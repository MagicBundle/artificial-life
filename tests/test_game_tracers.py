"""W2 tracer rig (DESIGN §32, spec §6.1, §9.8): passive tracers moved by the wind with the agents' coupling.

The rig is an observer: it draws from its own key, never reads or advances the state's, persists in the
sidecar and resumes exactly. Its motion is the engine's integrator on the unscaled coupling and the total
wind.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from alife import cli, runner
from alife import weather as W
from alife.config import Config, ConfigError, apply_overrides, dims
from alife.game import observer
from alife.game import tracers as T
from alife.game.observer import Observer
from alife.runio import RunDir, load_checkpoint
from alife.state import init_state, state_digest, to_numpy
from weather_helpers import fixed_wx, spun_ws, storm2d_config, storm3d_config

OBS = ("game.observe=true", "run.checkpoint_every=10", "run.metrics_every=5", "run.heavy_metrics_every=10")
TRC = ("game.tracers=32", "game.tracer_period=10")


def _scene(cfg: Config, n_alive: int = 100) -> tuple[mx.array, mx.array, mx.array]:
    """``(pos, buoy_cap, alive)`` over every row: row i at x = i, y = 2i (mod L), z = 5 + 0.5·(i mod 100); the
    first ``n_alive`` rows alive, the cap rising with the row."""
    d = dims(cfg)
    i = np.arange(d.n, dtype=np.float32)
    pos = np.stack([i, (2 * i) % cfg.world.size, 5.0 + 0.5 * (i % 100)], axis=1).astype(np.float32)
    cap = (i / d.n).astype(np.float32)
    return mx.array(pos[:, : d.nd]), mx.array(cap), mx.array(np.arange(d.n) < n_alive)


def _cfg(*extra: str) -> Config:
    return storm3d_config(*OBS, *TRC, *extra)


# --- reseed ---------------------------------------------------------------------------------------


def test_reseed_takes_only_live_rows_uniformly_with_their_coupling() -> None:
    cfg = _cfg("game.tracers=20000")
    pos, cap, alive = _scene(cfg, n_alive=100)
    alive = mx.array((np.arange(dims(cfg).n) % 5 == 0) & (np.arange(dims(cfg).n) < 200))  # 40 live rows
    tr = T.tracer_init(pos, cap, alive, cfg)
    mx.eval(*tr)
    row = np.rint(to_numpy(tr.pos)[:, 0]).astype(int)  # x identifies the row
    live = np.flatnonzero(to_numpy(alive))
    assert np.isin(row, live).all()  # dead rows are never picked
    counts = np.bincount(row, minlength=dims(cfg).n)[live]
    expected = 20000 / live.size
    chi2 = float(((counts - expected) ** 2 / expected).sum())
    assert chi2 < live.size - 1 + 6 * np.sqrt(2 * (live.size - 1))  # df 39: a loose bound, the key is fixed
    p = W.params_from_config(cfg)
    want = p.alpha0 + p.alpha1 * to_numpy(cap)[row]
    np.testing.assert_allclose(to_numpy(tr.alpha), want, rtol=1e-6)  # the coupling of the agent seeded from
    assert (to_numpy(tr.age) == 0).all()


def test_alpha_is_monotone_in_the_cap_and_unscaled() -> None:
    cfg = _cfg("weather.wind_scale=0.0")  # still air for the agents: the tracers' coupling is unchanged
    pos, cap, alive = _scene(cfg)
    tr = T.tracer_init(pos, cap, alive, cfg)
    mx.eval(*tr)
    row = np.rint(to_numpy(tr.pos)[:, 0]).astype(int)
    order = np.argsort(row)
    a = to_numpy(tr.alpha)[order]
    assert (np.diff(a) >= 0).all() and a.max() > a.min()
    p = W.params_from_config(cfg)
    np.testing.assert_allclose(a, p.alpha0 + p.alpha1 * to_numpy(cap)[row[order]], rtol=1e-6)


def test_reseed_draws_from_the_rigs_key_and_leaves_the_state_key_alone() -> None:
    cfg = _cfg()
    s = init_state(cfg)
    key0 = to_numpy(s.key).copy()
    tr = T.tracer_init(s.pos, s.buoy_cap, s.alive, cfg)
    tr2 = T.tracer_reseed(tr, s.pos, s.buoy_cap, s.alive, cfg)
    mx.eval(*tr, *tr2)
    assert np.array_equal(to_numpy(s.key), key0)  # the state's key is not even an input of the rig
    assert not np.array_equal(to_numpy(tr.key), to_numpy(tr2.key))  # the rig's key is split at each reseed
    again = T.tracer_init(s.pos, s.buoy_cap, s.alive, cfg)
    mx.eval(*again)
    assert all(
        np.array_equal(to_numpy(x), to_numpy(y)) for x, y in zip(tr, again, strict=True)
    )  # deterministic
    assert not np.array_equal(to_numpy(T.tracer_key(cfg.run.seed)), key0)


def test_no_live_row_leaves_the_tracers_somewhere_finite() -> None:
    cfg = _cfg()
    pos, cap, _ = _scene(cfg)
    tr = T.tracer_init(pos, cap, mx.zeros((dims(cfg).n,), dtype=mx.bool_), cfg)
    mx.eval(*tr)
    assert np.isfinite(to_numpy(tr.pos)).all() and np.isfinite(to_numpy(tr.alpha)).all()


# --- motion ---------------------------------------------------------------------------------------


def test_zero_wind_no_motion_and_age_counts() -> None:
    cfg = _cfg()
    pos, cap, alive = _scene(cfg)
    tr = T.tracer_init(pos, cap, alive, cfg)
    wx = fixed_wx(cfg)
    t2 = T.tracer_step(T.tracer_step(tr, wx, cfg), wx, cfg)
    mx.eval(*tr, *t2)
    assert np.array_equal(to_numpy(t2.pos), to_numpy(tr.pos))
    assert (to_numpy(t2.age) == 2).all()
    assert np.array_equal(to_numpy(t2.alpha), to_numpy(tr.alpha))


def test_a_uniform_wind_moves_each_tracer_at_alpha_times_the_wind() -> None:
    cfg = _cfg()
    pos, cap, alive = _scene(cfg)
    tr = T.tracer_init(pos, cap, alive, cfg)
    t2 = T.tracer_step(tr, fixed_wx(cfg, u1=0.1, v1=-0.05, u2=0.1, v2=-0.05), cfg)
    mx.eval(*tr, *t2)
    moved = to_numpy(t2.pos) - to_numpy(tr.pos)
    moved[:, :2] = (moved[:, :2] + cfg.world.size / 2) % cfg.world.size - cfg.world.size / 2  # undo the wrap
    z = to_numpy(tr.pos)[:, 2]
    keep = z >= 4.0  # above the surface-layer ramp phi(z) = 1
    a = to_numpy(tr.alpha)[keep]
    np.testing.assert_allclose(moved[keep, 0], a * 0.1, atol=2e-5)
    np.testing.assert_allclose(moved[keep, 1], -a * 0.05, atol=2e-5)


def test_speed_limit_and_clamps() -> None:
    cfg = _cfg()
    vmax = cfg.physics.v_max
    pos, cap, alive = _scene(cfg)
    d = dims(cfg)
    tr = T.tracer_init(pos, cap, alive, cfg)
    fast = tr._replace(alpha=tr.alpha * 0 + 10.0)  # no valid config couples that hard: the limit binds
    t2 = T.tracer_step(fast, fixed_wx(cfg, u1=0.3, v1=0.0, u2=0.3, v2=0.0), cfg)
    mx.eval(*tr, *t2)
    step = np.linalg.norm(to_numpy(t2.pos) - to_numpy(fast.pos), axis=1)
    assert (step <= vmax + 1e-5).all() and step.max() > 0.9 * vmax
    # a tracer on the floor or the lid stays in [z_lo, z_hi] and x, y wrap into [0, L)
    edge = tr._replace(
        pos=mx.array([[cfg.world.size - 0.01, 0.0, d.z_lo], [0.0, 0.0, d.z_hi]] * 16, dtype=mx.float32)
    )
    t3 = T.tracer_step(edge._replace(alpha=edge.alpha * 0 + 10.0), fixed_wx(cfg, u1=0.3, u2=0.3), cfg)
    mx.eval(*t3)
    p3 = to_numpy(t3.pos)
    assert ((p3[:, :2] >= 0) & (p3[:, :2] < cfg.world.size)).all()
    assert ((p3[:, 2] >= d.z_lo) & (p3[:, 2] <= d.z_hi)).all()


def test_tracers_move_with_the_unscaled_coupling_and_the_total_wind_in_every_arm() -> None:
    base = _cfg()
    wx = W.pack(spun_ws(base))
    pos, cap, alive = _scene(base)
    ref = T.tracer_step(T.tracer_init(pos, cap, alive, base), wx, base)
    mx.eval(*ref)
    for extra in (
        ("weather.wind_scale=0.0",),
        ("weather.wind_eddy_force=true",),
        ("weather.storm_eddy=true",),
    ):
        cfg = _cfg(*extra)
        out = T.tracer_step(T.tracer_init(pos, cap, alive, cfg), wx, cfg)
        mx.eval(*out)
        assert np.array_equal(to_numpy(out.pos), to_numpy(ref.pos)), extra
    assert (np.abs(to_numpy(ref.pos) - to_numpy(T.tracer_init(pos, cap, alive, base).pos)) > 0).any()


def test_compiled_equals_eager() -> None:
    cfg = _cfg()
    wx = W.pack(spun_ws(cfg))
    pos, cap, alive = _scene(cfg)
    tr = T.tracer_init(pos, cap, alive, cfg)
    step_c = mx.compile(lambda t, w: T.tracer_step(t, w, cfg))
    seed_c = mx.compile(lambda t, p, c, a: T.tracer_reseed(t, p, c, a, cfg))
    a, b = tr, tr
    for _ in range(3):
        a, b = step_c(a, wx), T.tracer_step(b, wx, cfg)
    a2, b2 = seed_c(a, pos, cap, alive), T.tracer_reseed(b, pos, cap, alive, cfg)
    mx.eval(*a, *b, *a2, *b2)
    for x, y in ((a, b), (a2, b2)):
        assert all(np.array_equal(to_numpy(u), to_numpy(v)) for u, v in zip(x, y, strict=True))


def test_two_d_tracers_follow_the_layer_one_wind() -> None:
    cfg = apply_overrides(storm2d_config(), [*OBS, *TRC])
    s = init_state(cfg)
    tr = T.tracer_init(s.pos, None, s.alive, cfg)
    t2 = T.tracer_step(tr, fixed_wx(cfg, u1=0.1, v1=0.05), cfg)
    mx.eval(*tr, *t2)
    p = W.params_from_config(cfg)
    moved = to_numpy(t2.pos) - to_numpy(tr.pos)
    moved = (moved + cfg.world.size / 2) % cfg.world.size - cfg.world.size / 2  # undo the wrap
    np.testing.assert_allclose(moved[:, 0], p.alpha0 * 0.1, atol=5e-5)
    np.testing.assert_allclose(moved[:, 1], p.alpha0 * 0.05, atol=5e-5)
    assert (to_numpy(tr.alpha) == np.float32(p.alpha0)).all()


# --- config ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        (["game.tracers=8"], "game.tracers > 0 needs game.observe"),
        (["game.tracers=8", "game.observe=true"], "game.tracers > 0 needs weather.enabled"),
        (["game.tracers=-1"], r"game.tracers must be in"),
        (["game.tracers=2000000", "game.observe=true"], r"game.tracers must be in"),
        (["game.tracer_period=0"], "game.tracer_period must be"),
        (["run.eco_every=10"], "run.eco_every > 0 needs game.observe"),
        (["run.eco_every=-1"], "run.eco_every must be"),
        (["run.wx_every=10"], "run.wx_every > 0 needs weather.enabled"),
    ],
)
def test_the_new_keys_are_validated(tiny_cfg: Config, overrides: list[str], match: str) -> None:
    with pytest.raises(ConfigError, match=match):
        apply_overrides(tiny_cfg, overrides)


def test_the_eco_dump_needs_a_3d_weather_world() -> None:
    with pytest.raises(ConfigError, match="needs a 3D world"):
        apply_overrides(storm2d_config(), ["game.observe=true", "run.eco_every=10"])
    ok = apply_overrides(storm3d_config(), ["game.observe=true", "run.eco_every=10", "run.wx_every=10"])
    assert (ok.run.eco_every, ok.run.wx_every) == (10, 10)
    assert (
        apply_overrides(storm2d_config(), ["run.wx_every=10"]).run.wx_every == 10
    )  # the weather dump is 2D too


def test_the_new_keys_do_not_move_the_config_hash() -> None:
    from alife.config import config_hash

    base = storm3d_config()
    on = apply_overrides(base, [*OBS, *TRC, "run.eco_every=10", "run.wx_every=10"])
    off = apply_overrides(base, ["game.observe=true"])
    assert config_hash(on) == config_hash(off) == config_hash(base)


# --- the observer ---------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def runs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    """Three 20-step runs of one start: observers with tracers and both dumps (``on``), observers alone
    (``off``), the same with tracers but no dumps (``plain``) and no observer at all (``bare``); returns their
    directories and final states.
    """
    root = tmp_path_factory.mktemp("trc")
    base = storm3d_config(
        *OBS, "weather.seed_gain=0.05", "weather.shade_gain=0.02", "weather.rain_gain=1000000"
    )
    start = init_state(base)._replace(wx=W.pack(spun_ws(base)))
    out: dict[str, object] = {}
    for name, extra in (
        ("on", [*TRC, "run.eco_every=10", "run.wx_every=10"]),
        ("off", []),
        ("plain", [*TRC]),
        ("bare", ["game.observe=false"]),
    ):
        cfg = apply_overrides(base, extra)
        rd = RunDir.create(root / name, cfg)
        res = runner.run(rd, cfg, start, 20, quiet=True)
        assert res.status == "done"
        out[name] = rd
        out[f"{name}_state"] = res.state
        out[f"{name}_cfg"] = cfg
    return out


def test_a_run_with_tracers_and_dumps_has_the_trajectory_of_a_run_without(runs: dict) -> None:
    digests = {name: state_digest(runs[f"{name}_state"]) for name in ("on", "off", "plain", "bare")}
    assert len(set(digests.values())) == 1  # tracers and both dumps on == observers alone == no observer
    rows = lambda rd: [  # noqa: E731
        {
            k: v
            for k, v in r.items()
            if k not in ("coupling", "observers", "wall_s", "steps_per_s", "peak_mem_gb")
        }
        for r in rd.metrics_rows()
    ]
    assert rows(runs["on"]) == rows(runs["off"]) == rows(runs["bare"])
    assert all("observers" not in r for r in runs["bare"].metrics_rows())


def test_the_coupling_block_is_in_the_rows_only_with_tracers(runs: dict) -> None:
    assert all("coupling" not in r for r in runs["off"].metrics_rows())
    for r in runs["on"].metrics_rows():
        c = r["coupling"]
        assert len(c["n_agents"]) == 4 and sum(c["n_tracers"]) == 32
        assert sum(c["n_agents"]) == round(r["n_alive"])
        assert 0.0 <= (c["ub_agents_all"] or 0.0) <= 1.0


def test_the_sidecar_holds_the_tracers_and_resume_is_exact(runs: dict, tmp_path: Path) -> None:
    rd, cfg = runs["on"], runs["on_cfg"]
    a = rd.read_observers(20)
    assert a is not None and {"tracer.pos", "tracer.alpha", "tracer.age", "tracer.key"} <= set(a)
    assert a["tracer.pos"].shape == (32, 3) and a["tracer.pos"].dtype == np.float32
    assert a["tracer.key"].dtype == np.uint32
    assert rd.read_observers(10) is not None
    # a run of 10 steps resumed for 10 more ends with the tracers of the uninterrupted run
    start = init_state(cfg)._replace(wx=W.pack(spun_ws(cfg)))
    split = RunDir.create(tmp_path / "split", cfg)
    runner.run(split, cfg, start, 10, quiet=True)
    state = load_checkpoint(split.latest_checkpoint(), cfg)
    split.truncate_after(int(to_numpy(state.step)))
    runner.run(split, cfg, state, 10, quiet=True)
    b = split.read_observers(20)
    assert b is not None
    assert all(np.array_equal(a[k], b[k]) for k in a if k.startswith("tracer."))
    assert json.dumps(rd.metrics_rows()[-1]["coupling"]) == json.dumps(split.metrics_rows()[-1]["coupling"])


def test_a_sidecar_without_tracers_restarts_only_the_tracers(runs: dict) -> None:
    rd, cfg = runs["off"], runs["on_cfg"]  # observers sidecar written with the rig off, resumed with it on
    state = load_checkpoint(rd.checkpoint_at(20), cfg)
    obs, note = Observer.open(rd, state, cfg)
    assert note is not None and "no valid tracers" in note
    assert obs.tr is not None and obs.tr.pos.shape == (32, 3)
    ref, none = Observer.open(rd, state, runs["off_cfg"])
    assert none is None and ref.tr is None
    # the labels came from the sidecar, not from a fresh start
    assert np.array_equal(to_numpy(obs.o.uid), to_numpy(ref.o.uid)) and int(to_numpy(obs.o.n_obs).max()) > 0
    # a changed count restarts them too
    rd_on = runs["on"]
    cfg64 = apply_overrides(runs["on_cfg"], ["game.tracers=64"])
    state_on = load_checkpoint(rd_on.checkpoint_at(20), cfg64)
    obs64, note64 = Observer.open(rd_on, state_on, cfg64)
    assert note64 is not None and obs64.tr is not None and obs64.tr.pos.shape == (64, 3)


def test_the_analysis_tools_leave_the_tracers_off(runs: dict) -> None:
    rd, cfg = runs["on"], runs["on_cfg"]
    state = load_checkpoint(rd.checkpoint_at(20), cfg)
    obs, note = Observer.open(rd, state, cfg, tracers=False)
    assert (
        note is None
        and obs.tr is None
        and len(obs.arrays()) == len(Observer.fresh(state, cfg, tracers=False).arrays())
    )
    assert Observer.fresh(state, cfg).tr is not None


def test_resume_accepts_the_tracer_keys(runs: dict, tmp_path: Path) -> None:
    copy = tmp_path / "run"
    shutil.copytree(runs["off"].path, copy)
    sets = ["game.tracers=16", "game.tracer_period=5", "run.eco_every=5", "run.wx_every=5"]
    code = cli.main(
        ["resume", str(copy), *[a for s in sets for a in ("--set", s)], "--steps", "10", "--quiet"]
    )
    assert code == 0
    rd = RunDir.open(copy)
    assert rd.config().game.tracers == 16 and rd.config().run.eco_every == 5
    assert [s for s, _ in rd.dump_files("eco")] == [25, 30] and [s for s, _ in rd.dump_files("wx")] == [
        25,
        30,
    ]


def test_truncate_after_removes_the_dumps_after_the_step(runs: dict, tmp_path: Path) -> None:
    copy = tmp_path / "run"
    shutil.copytree(runs["on"].path, copy)
    rd = RunDir.open(copy)
    assert [s for s, _ in rd.dump_files("eco")] == [10, 20] and [s for s, _ in rd.dump_files("wx")] == [
        10,
        20,
    ]
    rd.truncate_after(10)
    assert [s for s, _ in rd.dump_files("eco")] == [10] and [s for s, _ in rd.dump_files("wx")] == [10]


def test_the_default_observer_is_unchanged_by_the_new_code(runs: dict) -> None:
    assert observer.OBSERVER_FORMAT == 4
    rd = runs["off"]
    a = rd.read_observers(20)
    assert a is not None and not any(k.startswith("tracer.") for k in a)


def test_the_dump_files_hold_the_live_rows_at_their_step(runs: dict) -> None:
    rd, s = runs["on"], runs["on_state"]
    assert [st for st, _ in rd.dump_files("eco")] == [10, 20] and [st for st, _ in rd.dump_files("wx")] == [
        10,
        20,
    ]
    assert not any(runs[k].dump_files("eco") or runs[k].dump_files("wx") for k in ("off", "plain"))
    live = to_numpy(s.alive)
    e = rd.read_eco(20)
    assert {k: v.dtype for k, v in e.items()} == {
        "uid": np.int64, "clade": np.int32, "cap": np.float16, "z": np.float16, "vz_own": np.float16,
        "w_air": np.float16, "emit": np.int16, "tracer_z": np.float16, "tracer_age": np.int32,
    }  # fmt: skip
    np.testing.assert_array_equal(e["uid"], to_numpy(s.uid)[live])
    np.testing.assert_array_equal(e["clade"], to_numpy(s.clade)[live])
    np.testing.assert_array_equal(e["cap"], to_numpy(s.buoy_cap)[live].astype(np.float16))
    np.testing.assert_array_equal(e["z"], to_numpy(s.pos)[live, 2].astype(np.float16))
    assert e["tracer_z"].shape == (32,) and (e["tracer_age"] == 0).all()  # reseeded at step 20 (period 10)
    assert (e["emit"] >= 0).all() and (e["emit"] <= 768).all()
    w = rd.read_wx(20)
    assert w.shape == (10, 160, 160) and w.dtype == np.float32
    np.testing.assert_array_equal(w[:9], to_numpy(s.wx))
    assert float(w[9].max()) > 0.0  # seeding is on in this fixture
