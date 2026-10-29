"""Built structures, the observer PR S2 (DESIGN §38.9): the shelter in the observers' wind drift, the observer
channel (``Acts.ext.structures``, ``ActFlag.BUILDING``) and the purity of observing."""

from __future__ import annotations

from dataclasses import replace

import mlx.core as mx
import numpy as np
import pytest

from alife import runner, structures
from alife.analysis import labels
from alife.config import Config, apply_overrides, dims
from alife.game.observe import quantise_inputs, speed_unit
from alife.game.rules import RULES_V3, arch_layout
from alife.layout import B
from alife.metrics import snapshot_metrics
from alife.runio import RunDir
from alife.state import init_state, state_digest, to_numpy
from alife.step import ActFlag, make_step
from game_helpers import blank_acts, living, make_acts, make_oi
from helpers import make_scene
from structures_helpers import builders, cfg_for, units_for
from weather_helpers import fixed_wx, storm2d_config, storm3d_config


def _spots(nd: int) -> list[tuple[float, ...]]:
    z = (12.0,) if nd == 3 else ()
    return [(20.0 + 40.0 * i, 30.0 + 25.0 * i, *z) for i in range(6)]  # far apart: no contacts


def _scene(nd: int):
    mk = storm2d_config if nd == 2 else storm3d_config
    cfg = mk("structures.enabled=true", "structures.aperture=0.0", "structures.barrier_gain=0.0")
    cfg = apply_overrides(cfg, ["weather.storm_p=0.0"])
    d = dims(cfg)
    s = make_scene(cfg, [{"pos": p} for p in _spots(nd)], wx=fixed_wx(cfg, u1=0.05, v1=0.03))
    return cfg, d, s


@pytest.mark.parametrize("nd", [2, 3])
def test_the_observer_drift_follows_the_engines_shelter(nd: int) -> None:
    """Zero weights, no barrier, isolated agents: the engine's velocity in a full cell over the one in an
    empty
    cell is the shelter factor ``1 - shelter_gain`` (the grip factor of the zero brain cancels), and the
    observer's ``drift_at`` of the same two states has the same ratio, while the bare ``agent_drift`` (the
    audited bug) has ratio 1."""
    from alife import weather

    cfg, d, s = _scene(nd)
    n = len(_spots(nd))
    full = mx.full(d.field_shape, cfg.structures.cap, dtype=mx.int32)
    empty = mx.zeros(d.field_shape, dtype=mx.int32)
    cap = getattr(s, "buoy_cap", None)
    vel, drift = {}, {}
    for name, mat in (("full", full), ("empty", empty)):
        st = s._replace(mat=mat)
        s1, _ = make_step(cfg)(st)
        mx.eval(s1)
        vel[name] = to_numpy(s1.vel)[:n, :2]
        drift[name] = to_numpy(structures.drift_at(st.wx, st.pos, cap, st.alive, mat, cfg))[:n, :2]
    raw = to_numpy(weather.agent_drift(s.wx, s.pos, cap, s.alive, cfg))[:n, :2]
    sg = cfg.structures.shelter_gain
    assert np.abs(vel["empty"]).max() > 1e-4
    assert np.allclose(vel["full"], (1 - sg) * vel["empty"], rtol=1e-4, atol=1e-8)
    assert np.allclose(drift["full"], (1 - sg) * drift["empty"], rtol=1e-4, atol=1e-8)
    assert np.allclose(drift["empty"], raw)  # the empty grid is the old drift
    assert not np.allclose(raw, drift["full"], rtol=0.1)  # and the old drift over-states a sheltered one


@pytest.mark.parametrize("nd", [2, 3])
def test_an_empty_grid_leaves_the_drift_as_it_was(nd: int) -> None:
    from alife import weather

    cfg, d, s = _scene(nd)
    cap = getattr(s, "buoy_cap", None)
    a = structures.drift_at(s.wx, s.pos, cap, s.alive, mx.zeros(d.field_shape, mx.int32), cfg)
    b = weather.agent_drift(s.wx, s.pos, cap, s.alive, cfg)
    assert np.array_equal(to_numpy(a), to_numpy(b))


def test_structures_off_is_exactly_agent_drift() -> None:
    from alife import weather

    cfg = storm3d_config()
    s = make_scene(cfg, [{"pos": p} for p in _spots(3)], wx=fixed_wx(cfg, u1=0.05))
    a = structures.drift_at(s.wx, s.pos, s.buoy_cap, s.alive, None, cfg)
    assert np.array_equal(to_numpy(a), to_numpy(weather.agent_drift(s.wx, s.pos, s.buoy_cap, s.alive, cfg)))


def test_the_game_observers_subtract_the_sheltered_drift() -> None:
    """``quantise_inputs`` under a wind-relative rule set: the velocity halves of a free agent that drifts
    with
    the sheltered wind are zero and the WIND column is the sheltered drift's length."""
    cfg, d, s = _scene(3)
    n = len(_spots(3))
    full = mx.full(d.field_shape, cfg.structures.cap, dtype=mx.int32)
    drift = to_numpy(structures.drift_at(s.wx, s.pos, s.buoy_cap, s.alive, full, cfg)).astype(np.float32)
    arrays = living(cfg, n)
    arrays["pos"], arrays["buoy_cap"], arrays["wx"] = to_numpy(s.pos), to_numpy(s.buoy_cap), to_numpy(s.wx)
    arrays["mat"] = to_numpy(full)
    arrays["vel"] = drift
    x6, _, _ = quantise_inputs(
        make_oi(arrays), make_acts(blank_acts(cfg)), mx.zeros((d.n, B), mx.bool_), cfg, RULES_V3
    )
    x6, lay = to_numpy(x6), arch_layout(3, RULES_V3)
    scale = 256.0 / speed_unit(cfg)
    halves = x6[:n, lay.v_pos : lay.v_pos + 3].astype(np.int64) - x6[:n, lay.v_neg : lay.v_neg + 3]
    assert np.abs(halves).max() <= 1  # the agent moves with the sheltered wind: no own velocity
    assert x6[0, lay.wind] == round(float(np.linalg.norm(drift[0])) * scale) > 0


def test_obs_in_carries_the_material_for_a_wind_relative_rule_set() -> None:
    from alife.game.observe import obs_in

    _, _, s = _scene(3)
    oi = obs_in(s, RULES_V3)
    assert oi.mat is not None and oi.mat.shape == s.mat.shape
    assert obs_in(s, None).mat is None  # a rule set that reads no wind needs none


# --- the labels re-simulation -------------------------------------------------------------------------------


@pytest.mark.slow
def test_the_labels_study_passes_the_material_to_its_drift(tmp_path, monkeypatch) -> None:
    cfg: Config = apply_overrides(
        storm3d_config("structures.enabled=true", "population.n_init=128"),
        ["run.metrics_every=5", "run.checkpoint_every=10", "run.heavy_metrics_every=5", "game.observe=true"],
    )
    rd = RunDir.create(tmp_path / "run", cfg)
    assert runner.run(rd, cfg, init_state(cfg), 60, quiet=True).status == "done"
    calls: list[bool] = []
    real = structures.drift_at

    def spy(wx, pos, buoy_cap, alive, mat, c):
        calls.append(mat is not None)
        return real(wx, pos, buoy_cap, alive, mat, c)

    monkeypatch.setattr(structures, "drift_at", spy)
    study = {
        "start": 50, "steps": 120, "warmup": 40, "cohort_every": 40, "arch_horizon": 40, "arch_min_obs": 10,
        "intent_horizon": 10, "n_perm": 3,
    }  # fmt: skip
    cfg_run = RunDir.open(rd.path).config()
    assert cfg_run.structures.enabled
    rep = labels.study(rd.path, **study, log=lambda _: None)
    assert rep["rows_verified"] > 0
    assert calls and all(calls)  # every drift evaluation of the study saw the grid


# --- the observer channel -----------------------------------------------------------------------------------


@pytest.mark.parametrize("nd", [2, 3])
def test_acts_ext_structures_and_the_building_flag(nd: int) -> None:
    cfg = cfg_for(nd)
    s = builders(cfg, zero_weights=True)
    u = units_for(cfg)
    assert u > 0
    s1, out = make_step(cfg, observe=True)(s)
    mx.eval(s1, *out)
    a = out.acts
    assert a.ext is not None and a.ext.structures is not None
    alive = to_numpy(s.alive)
    build = to_numpy(a.ext.structures.build)
    assert build.dtype == np.int16 and np.array_equal(build, np.where(alive, u, 0))
    flags = to_numpy(a.flags).astype(np.int64)
    assert (flags[alive] & ActFlag.BUILDING.value).all() and not (
        flags[~alive] & ActFlag.BUILDING.value
    ).any()
    cover = to_numpy(a.ext.structures.cover)
    assert cover.dtype == np.int16 and cover.shape == alive.shape and (cover[alive] == 0).all()  # empty grid


def test_the_cover_is_the_start_of_step_cell() -> None:
    cfg = cfg_for(2)
    s = builders(cfg, zero_weights=True)
    d = dims(cfg)
    full = mx.full(d.field_shape, cfg.structures.cap, dtype=mx.int32)
    _, out = make_step(cfg, observe=True)(s._replace(mat=full))
    mx.eval(*out)
    alive = to_numpy(s.alive)
    assert (to_numpy(out.acts.ext.structures.cover)[alive] == 256).all()


@pytest.mark.parametrize("nd", [2, 3])
def test_observing_a_structures_world_changes_no_state_value(nd: int) -> None:
    cfg = replace(cfg_for(nd, compile=False))
    s = builders(cfg, bias=1.0)
    plain, obs = s, s
    f0, f1 = make_step(cfg), make_step(cfg, observe=True)
    for _ in range(6):
        plain, o0 = f0(plain)
        obs, o1 = f1(obs)
        mx.eval(plain, obs, o0, *o1)
        assert state_digest(plain) == state_digest(obs)
        assert np.array_equal(to_numpy(o0.counters), to_numpy(o1.counters))


def test_a_world_without_structures_has_no_structures_channel() -> None:
    cfg = replace(cfg_for(2), structures=replace(cfg_for(2).structures, enabled=False))
    s = init_state(cfg)
    _, out = make_step(cfg, observe=True)(s)
    mx.eval(*out)
    assert out.acts.ext is None or out.acts.ext.structures is None


# --- the dump and the metrics -------------------------------------------------------------------------------


@pytest.mark.parametrize("nd", [2, 3])
def test_the_mat_dump_is_the_grid_at_the_dump_steps_and_is_cut_on_resume(nd: int, tmp_path) -> None:
    cfg = cfg_for(
        nd, "run.mat_every=10", "run.metrics_every=10", "run.checkpoint_every=30", "run.keep_checkpoints=5"
    )
    s = builders(cfg, bias=2.0)
    rd = RunDir.create(tmp_path / "run", cfg)
    res = runner.run(rd, cfg, s, 30, quiet=True)
    assert res.status == "done"
    files = rd.dump_files("mat")
    assert [st for st, _ in files] == [10, 20, 30] and files[0][1].name == "mat_0000000010.npy"
    g = rd.read_grid("mat", 30)
    assert g.dtype == np.int32 and g.shape == dims(cfg).field_shape
    assert np.array_equal(g, to_numpy(res.state.mat)) and g.sum() > 0  # the exact grid; builders built
    with pytest.raises(ValueError, match="grid dump"):
        rd.read_grid("eco", 30)
    rd.truncate_after(20)
    assert [st for st, _ in rd.dump_files("mat")] == [
        10,
        20,
    ]  # the dumps after the resume step are regenerated


def test_the_mat_cadence_needs_structures_and_is_a_resume_key() -> None:
    from alife.cli import RESUME_KEYS, _resume_allowed
    from alife.config import ConfigError, config_hash, config_to_dict

    assert "mat_every" in RESUME_KEYS and _resume_allowed("run.mat_every=100")
    with pytest.raises(ConfigError, match="mat_every"):
        apply_overrides(cfg_for(2, "structures.enabled=false"), ["run.mat_every=5"])
    cfg = cfg_for(2)
    assert (
        "mat_every" not in config_to_dict(cfg, omit_defaults=True)["run"]
    )  # absent at 0, as before the dump
    assert (
        config_to_dict(apply_overrides(cfg, ["run.mat_every=5"]), omit_defaults=True)["run"]["mat_every"] == 5
    )
    assert config_hash(cfg) == config_hash(
        apply_overrides(cfg, ["run.mat_every=5"])
    )  # run is outside the hash


def test_the_snapshot_keys_of_a_structures_world(tmp_path) -> None:
    cfg = cfg_for(2, "run.metrics_every=20", "run.checkpoint_every=40", "run.heavy_metrics_every=40")
    rd = RunDir.create(tmp_path / "run", cfg)
    assert runner.run(rd, cfg, builders(cfg, bias=2.0), 40, quiet=True).status == "done"
    last = rd.metrics_rows()[-1]
    for k in ("build_here_mean", "build_in_frac", "build_cover_mean", "build_mat_max"):
        assert last[k] is not None, k
    assert 0.0 <= last["build_in_frac"] <= 1.0 and last["build_here_mean"] <= last["build_cover_mean"] + 1e-9
    assert 0 < last["build_mat_max"] <= cfg.structures.cap
    off = cfg_for(2, "structures.enabled=false")
    assert not [k for k in snapshot_metrics(init_state(off), off, heavy=False) if k.startswith("build_")]
