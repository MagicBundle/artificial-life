"""The sphere weather in the engine (DESIGN §46.6): the configuration rules and the readiness gate, the state
layout, the engine step eager on a weather-on sphere (invariants, the ledger, the rain, the wind that moves
agents), the weather-only spin-up, the metrics block. Compiled runs are in
``test_weather_sph_compiled.py``."""

from __future__ import annotations

import json
from dataclasses import replace

import mlx.core as mx
import numpy as np
import pytest

from alife import config, fields, invariants, metrics, weather, weather_sph, weather_spinup
from alife.config import ConfigError, dims, load_config
from alife.rng import InitStream, init_keys
from alife.state import expected_layout, init_state, rest_wx, to_numpy, wx_shape_sphere
from alife.step import make_step
from alife.weather import RainIn
from sphere_helpers import BASE, SPHERE_SETS
from sphere_weather_helpers import NW, weather_cfg

W64K = [*BASE, *SPHERE_SETS["64k3s"]]


def refused(*overrides: str, which: str = "tiny3s") -> str:
    with pytest.raises(ConfigError) as e:
        weather_cfg(which, *overrides)
    return str(e.value)


# --- configuration ------------------------------------------------------------------------------------------


def test_the_weather_layer_is_ready_on_a_sphere_and_the_rest_is_still_gated() -> None:
    assert frozenset({"weather"}) == config.SPHERE_READY
    assert "weather" in config.sphere_wanted(weather_cfg("tiny3s"))
    for ov, name, pr in (
        ("structures.enabled=true", "structures", "S3"),
        ("game.observe=true", "observers", "S5"),
        ("director.enabled=true", "director", "S5"),
    ):
        msg = refused(ov)
        assert (
            f"{name} is not available on a sphere yet (PR {pr})" in msg
            and "weather" not in msg.split("PR")[0].split(":")[-1][-12:]
        )


def test_the_gate_still_refuses_the_weather_when_it_is_not_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "SPHERE_READY", frozenset())
    assert "weather is not available on a sphere yet (PR S4)" in refused()


@pytest.mark.parametrize("which", ["tiny3s", "mini3s", "64k3s"])
def test_the_three_preset_geometries_validate_with_the_weather(which: str) -> None:
    cfg = weather_cfg(which)
    p = weather_sph.params_from_config(cfg)
    assert (cfg.weather.cells, p.n) == (NW[which], NW[which]) and dims(cfg).nside * 2 == NW[which]
    assert weather_sph.courant(p, cfg.world.radius) <= 0.8 * weather_sph.C_LIMIT


@pytest.mark.parametrize(
    ("override", "text"),
    [
        ("weather.cells=17", "Nf times a power of two"),
        ("weather.cells=24", "Nf times a power of two"),
        ("weather.cells=8", "L_d"),  # a coarse grid: the deformation radius is under three cells
        ("weather.cells=32", "Courant"),  # a finer grid at the same dt: C_used 0.97
        ("weather.pools=0,0,8,8,1,0,0.01", "vy"),
        ("weather.pools=0,0,8,8,1,0,0,0", "7 numbers per pool"),
        ("weather.pools=0,40,8,8,1,0,0", "latitude"),  # y/R = 1.1 rad = 63.7 degrees
        ("weather.pools=-5,0,8,8,1,0,0", "arc length east"),
        ("weather.shade_gain=0.01", "not available on a sphere yet"),
        ("weather.rain_zonal=true", "not available on a sphere yet"),
        ("weather.storm_eddy=true", "not available on a sphere yet"),
        ("weather.wind_eddy_force=true", "not available on a sphere yet"),
        ("weather.H1=0", "rule"),
        ("weather.seed_gain=0.5", "seed_gain"),
        ("run.wx_every=10", "wx_every"),
    ],
)
def test_the_sphere_weather_rules_name_their_problem(override: str, text: str) -> None:
    assert text in refused(override)


def test_weather_keys_are_inert_on_a_dry_sphere_and_the_dry_state_is_unchanged() -> None:
    dry = load_config(preset="tiny3d", overrides=[*BASE, *SPHERE_SETS["tiny3s"]])
    assert dry.weather.cells == 160
    with pytest.raises(ConfigError, match=r"inert while weather\.enabled is false"):
        load_config(preset="tiny3d", overrides=[*BASE, *SPHERE_SETS["tiny3s"], "weather.f0=0.02"])
    nf = dims(dry).nside
    assert expected_layout(dry)["wx"][0] == wx_shape_sphere(nf) == (11, 12 * (2 * nf) ** 2)
    assert not np.array(rest_wx(dry)).any()  # the placeholder of S1 and S2: its digest is unchanged


def test_the_stormsph_overlay_is_made_for_the_64k3s_world() -> None:
    assert "stormsph" in config.WORLD_OVERLAYS and "stormsph" in config.ALL_OVERLAYS
    with pytest.raises(ConfigError, match=r"Courant|latitude|arc length|power of two"):
        load_config(preset="tiny3s", overlays=["stormsph"])
    cfg = load_config(
        preset="tiny3s",
        overlays=["stormsph"],
        overrides=[
            *W64K[1:2],
            "world.size=2048",
            "world.radius=90",
            "world.height=40",
            "world.nside=21",
            "world.layers=18",
        ],
    )
    p = weather_sph.params_from_config(cfg)
    assert (cfg.weather.cells, len(cfg.weather.pools) // 7) == (42, 6)
    assert weather_sph.courant(p, 90.0) == pytest.approx(0.5067, abs=1e-3)
    # f at the pools: f0·sin(lat) = 0.025·sin(40 deg) = 0.0161, as 64k3d's
    assert p.f0 * np.sin(np.radians(40.0)) == pytest.approx(0.0161, abs=1e-4)


# --- state --------------------------------------------------------------------------------------------------


def test_the_state_carries_the_rest_weather_of_the_pools() -> None:
    cfg = weather_cfg("tiny3s")
    s = init_state(cfg)
    nw = NW["tiny3s"]
    assert (
        tuple(s.wx.shape)
        == wx_shape_sphere(dims(cfg).nside, nw)
        == expected_layout(cfg)["wx"][0]
        == (11, 12 * nw * nw)
    )
    wx = to_numpy(s.wx)
    assert np.array_equal(wx, to_numpy(weather.rest_array(cfg, 0))) and np.isfinite(wx).all()
    assert (wx[0] == 32.0).all() and (wx[1] == 48.0).all() and not wx[2:8].any() and not wx[9:].any()
    assert wx[8].min() == pytest.approx(0.7, abs=1e-3) and wx[8].max() <= 1.0  # Q_e(T_s) capped at Q_s
    assert invariants.full_check(s, cfg) == []
    # the step-dependent rest state: the pools drift (vx != 0), so the humidity map moves
    assert not np.array_equal(to_numpy(weather.rest_array(cfg, 5000)), wx)
    assert np.array_equal(
        to_numpy(weather.rest_array(cfg, 5000)), to_numpy(weather.rest_array(cfg, mx.array(5000)))
    )


def test_pack_and_unpack_are_inverse_and_dispatch_on_the_shape() -> None:
    cfg = weather_cfg("tiny3s")
    wx = weather.rest_array(cfg, 3)
    ws = weather.unpack(wx)
    assert isinstance(ws, weather_sph.WeatherStateS) and np.array_equal(
        to_numpy(weather.pack(ws)), to_numpy(wx)
    )
    assert weather.interface_height(ws, weather.params_from_config(cfg)).shape == (12 * 16 * 16,)


# --- the engine step ----------------------------------------------------------------------------------------


def run(cfg, n: int, state=None):
    fn = make_step(cfg, compile=False)
    s = init_state(cfg) if state is None else state
    prev = s
    for _ in range(n):
        prev = s
        s, out = fn(s)
    mx.eval(s, out)
    return prev, s, out


def test_the_engine_step_runs_with_the_weather_and_every_invariant_holds() -> None:
    cfg = weather_cfg("tiny3s", "weather.seed_rate=0.3", "weather.seed_gain=0.03")
    prev, s, out = run(cfg, 40)
    assert invariants.full_check(s, cfg, prev, out) == []
    wx = to_numpy(s.wx)
    assert np.isfinite(wx).all() and np.abs(wx[2:8]).max() > 0  # the seeds put winds in
    assert not np.array_equal(wx, to_numpy(init_state(cfg).wx))
    # no seed: the pools still move the humidity and, after condensation, the layers
    assert abs((wx[0] + wx[1]).sum() / (wx.shape[1] * 80.0) - 1.0) < 1e-5


def test_the_energy_ledger_closes_with_rain_feeding_the_resource() -> None:
    cfg = weather_cfg("tiny3s", "weather.rain_gain=500000")
    s = init_state(cfg)
    ws = weather_sph.unpack(s.wx)
    # a prescribed shower: rain everywhere in the cloud band, in the stored state, feeds the resource
    p = weather_sph.params_from_config(cfg)
    rain = RainIn(mx.full(ws.P.shape, 2e-3), mx.full(ws.P.shape, 32.0), mx.full(ws.P.shape, 20.0))
    base = fields.regen_rate(s.patch_pos, cfg, mx.array(0, dtype=mx.int32))
    with_rain = fields.regen_rate(s.patch_pos, cfg, mx.array(0, dtype=mx.int32), None, rain)
    extra = np.array(with_rain - base)
    assert extra.shape == base.shape and extra.min() >= 0 and extra.sum() > 0
    q = 500000 * 2e-3
    assert extra.sum(axis=0).max() <= q and extra.sum(axis=0).min() >= q - 2 * extra.shape[0]
    assert p.height == 80.0
    # and the engine books it: 60 steps with the rain on the sphere, the ledger of every step closes
    prev, s2, out = run(cfg, 60)
    assert invariants.full_check(s2, cfg, prev, out) == []
    assert int(np.array(s2.ledger_cum)[0:16].sum()) != 0


def test_a_prescribed_wind_moves_the_agents_by_alpha_times_the_wind() -> None:
    """Two steps from the same state, the stored weather the rest state or a prescribed solid rotation in both
    layers: the difference of the positions is the drift ``alpha·u_air`` (``agent_drift``), tangent."""
    cfg = weather_cfg("tiny3s", "weather.seed_rate=0.0", "weather.storm_p=0.0")
    s = init_state(cfg)
    x = weather_sph.weather_tab.weather_tab(NW["tiny3s"]).x
    u = 0.1 * np.cross([0.0, 0.0, 1.0], x)
    ws = weather_sph.unpack(s.wx)
    wx_w = weather_sph.pack(ws._replace(u1=mx.array(u, dtype=mx.float32), u2=mx.array(u, dtype=mx.float32)))
    fn = make_step(cfg, compile=False)
    a, _ = fn(s)
    b, _ = fn(s._replace(wx=wx_w))
    mx.eval(a, b)
    alive = np.array(s.alive)
    dpos = (np.array(b.pos) - np.array(a.pos))[alive]
    drift = np.array(weather.agent_drift(wx_w, s.pos, s.buoy_cap, s.alive, cfg))[alive]
    assert np.linalg.norm(drift, axis=1).mean() > 0.02
    cos = (dpos * drift).sum(1) / np.maximum(
        np.linalg.norm(dpos, axis=1) * np.linalg.norm(drift, axis=1), 1e-12
    )
    assert np.median(cos) > 0.9
    ratio = np.linalg.norm(dpos, axis=1).sum() / np.linalg.norm(drift, axis=1).sum()
    assert 0.5 < ratio < 1.5, ratio


def test_the_metrics_row_has_the_sphere_weather_block() -> None:
    cfg = weather_cfg("tiny3s", "weather.seed_gain=0.03")
    _, s, _ = run(cfg, 30)
    row = metrics.snapshot_metrics(s, cfg, heavy=False)
    blk = row["weather"]
    for k in (
        "wind_max_l1",
        "rain_total",
        "mass_rel",
        "q_total",
        "hyper_max",
        "storm_exposure",
        "seed_r_mean",
        "occ_hz0",
        "rain_band_share",
    ):
        assert k in blk, k
    assert abs(blk["mass_rel"]) < 1e-5 and json.dumps(row)  # plain JSON
    assert "storm_proxy" not in blk and "jet_max_l2" not in blk  # S5


# --- the spin-up --------------------------------------------------------------------------------------------


def _state_key(cfg) -> mx.array:
    return init_keys(cfg.run.seed)[InitStream.STATE_KEY]


def test_the_spinup_chain_is_the_engines_weather_bitwise() -> None:
    """Started from the engine's step 0 and state key, the chain gives the engine's ``wx`` after 30 steps
    (eager;
    the compiled chain is the child test). Agent-independent while seed_gain = 0."""
    c = weather_cfg("tiny3s", "weather.seed_rate=0.2", "run.seed=2")
    s = init_state(c)
    fn = make_step(c, compile=False)
    for _ in range(30):
        s, _ = fn(s)
    mx.eval(s)
    wx, key = weather_spinup.advance_chain(c, weather.rest_array(c), 0, _state_key(c), 30, compile=False)
    assert np.array_equal(to_numpy(wx), to_numpy(s.wx)) and np.array_equal(to_numpy(key), to_numpy(s.key))
    calm = weather_spinup.advance_chain(
        replace(c, weather=replace(c.weather, seed_rate=0.0)),
        weather.rest_array(c),
        0,
        _state_key(c),
        30,
        compile=False,
    )[0]
    assert not np.array_equal(to_numpy(calm), to_numpy(s.wx))  # a disturbance entered


def test_the_spinup_is_reproducible_ends_in_the_pools_of_the_start_step_and_depends_on_the_key() -> None:
    cfg = weather_cfg("tiny3s", "weather.seed_rate=0.1")
    key = _state_key(cfg)
    rest = to_numpy(weather.rest_array(cfg, 500))
    assert np.array_equal(to_numpy(weather_spinup.spinup(cfg, 0, step=500, key=key, compile=False)), rest)
    wx = to_numpy(weather_spinup.spinup(cfg, 60, step=500, key=key, compile=False))
    assert (
        wx.shape == rest.shape == (11, 12 * 16 * 16)
        and np.isfinite(wx).all()
        and not np.array_equal(wx, rest)
    )
    assert np.array_equal(wx, to_numpy(weather_spinup.spinup(cfg, 60, step=500, key=key, compile=False)))
    other = to_numpy(
        weather_spinup.spinup(cfg, 60, step=500, key=init_keys(7)[InitStream.STATE_KEY], compile=False)
    )
    assert not np.array_equal(wx, other)
    p = weather.params_from_config(cfg)
    assert abs((wx[0] + wx[1]).sum() / (wx.shape[1] * (p.H1 + p.H2)) - 1.0) < 1e-5
    # the engine resumes from it: a state with the spun-up weather steps and passes every check
    s = init_state(cfg)._replace(wx=mx.array(wx))
    fn = make_step(cfg, compile=False)
    s2, out = fn(s)
    assert invariants.full_check(s2, cfg, s, out) == []


def test_run_resume_and_inspect_work_with_the_weather_and_a_spinup(
    tmp_path, capsys: pytest.CaptureFixture
) -> None:
    """`alife run --preset tiny3s` with the weather on, `--weather-spinup`, the metrics rows, `resume` and
    `inspect` (eager: ``run.compile=false``)."""
    from pathlib import Path

    from alife import cli
    from alife.runio import RunDir
    from sphere_weather_helpers import pools

    out = Path(tmp_path) / "run"
    sets = [
        "brain.weight_dtype=float32",
        "run.compile=false",
        "weather.enabled=true",
        "weather.cells=16",
        "weather.pools=" + ",".join(f"{v:.6f}" for v in pools("tiny3s")),
        "weather.rain_gain=50000",
    ]
    argv = ["--preset", "tiny3s", *(a for s in sets for a in ("--set", s))]
    assert (
        cli.main(["run", *argv, "--steps", "20", "--weather-spinup", "12", "--out", str(out), "--quiet"]) == 0
    )
    rd = RunDir.open(out)
    assert rd.meta()["geometry"] == "sphere" and rd.meta()["weather_spinup"] == 12
    rows = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines() if line.strip()]
    assert rows and all(
        "weather" in r and r["ledger_error"] == 0 for r in rows if "weather" in r or "ledger_error" in r
    )
    assert cli.main(["inspect", str(out)]) == 0 and "ledger OK" in capsys.readouterr().out
    assert cli.main(["resume", str(out), "--until", "30", "--quiet"]) == 0
    assert RunDir.open(out).meta()["status"] == "done"
