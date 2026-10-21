"""W1 build step 4 (DESIGN §30, spec §7.1, §7.3, §9.13): the ``weather`` block of a metrics row, the hash-grid
occupancy helpers, ``invariants._check_weather`` and the weather rows of the staged benchmark.

CPU and fast. The weather block is checked against hand-built weather states (a Gaussian vortex for the storm
proxy), the occupancy against the engine's own neighbour search, the benchmark's staged pipeline against the
fused step bitwise in eager mode.
"""

from __future__ import annotations

import json
from dataclasses import replace

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from alife import bench, cli, invariants, metrics, spatial, weather_obs
from alife import weather as W
from alife.analysis import w0
from alife.config import Config, ConfigError, WeatherConfig, dims
from alife.state import CounterIdx, init_state, n_counters, state_digest, to_numpy
from alife.step import make_step
from weather_helpers import fixed_wx, storm2d_config, storm3d_config

KEYS_COMMON = {
    "wind_max_l1",
    "wind_max_l2",
    "jet_max_l2",
    "rain_total",
    "rain_cells",
    "vort_cells",
    "storm_proxy",
    "mass_rel",
    "q_total",
    "hyper_max",
    "storm_exposure",
    "air_speed_p99",
    "air_speed_max",
    "overlap_deep",
}
KEYS_OCC = {"occ_hz0", "occ_hz1_11", "occ_hz12"}


@pytest.fixture(scope="module", params=["3d", "2d"])
def cfg(request: pytest.FixtureRequest) -> Config:
    return storm3d_config() if request.param == "3d" else storm2d_config()


def _rest(cfg: Config) -> np.ndarray:
    return np.array(W.rest_array(cfg))


def _block(cfg: Config, wx: np.ndarray, speed=(), **kw) -> dict:
    return weather_obs.weather_block(wx, np.asarray(speed, dtype=np.float64), cfg, **kw)


def _vortex(cfg: Config, wx: np.ndarray, cx: float, cy: float, amp: float, sigma: float = 6.0) -> np.ndarray:
    """``wx`` with a Gaussian streamfunction vortex added to layer 1. ``amp < 0`` has a cyclonic (positive)
    vorticity at the centre in the northern hemisphere: ``ζ = ∇²ψ``, and ψ has a maximum at ``amp > 0``."""
    p = W.params_from_config(cfg)
    xs = (np.arange(p.n) + 1.0) * p.dx  # the corner coordinates
    psi = amp * np.exp(-((xs[None, :] - cx) ** 2 + (xs[:, None] - cy) ** 2) / (2.0 * sigma**2))
    du, dv = W.psi_winds(mx.array(psi.astype(np.float32)), p.dx)
    out = wx.copy()
    out[2] += np.array(du)
    out[3] += np.array(dv)
    return out


def _rain_at(cfg: Config, wx: np.ndarray, cx: float, cy: float, cells: int, rate: float = 1e-3) -> np.ndarray:
    """``wx`` with ``cells`` x ``cells`` raining cells around ``(cx, cy)``."""
    p = W.params_from_config(cfg)
    i, j = int(cx // p.dx), int(cy // p.dx)
    out = wx.copy()
    out[7, j : j + cells, i : i + cells] = rate
    return out


# --- the numpy helpers -----------------------------------------------------------------------------


def test_the_moved_helpers_keep_their_names_in_the_w0_driver() -> None:
    for name in ("label_periodic", "centre_winds", "corners_to_centres", "hemisphere_sign"):
        assert getattr(w0, name) is getattr(weather_obs, name), name
    assert w0.band_peaks is spatial.band_peaks


def test_the_numpy_vorticity_is_the_models(cfg: Config) -> None:
    p = W.params_from_config(cfg)
    rng = np.random.default_rng(3)
    ws = W.unpack(W.rest_array(cfg))
    ws = ws._replace(
        u1=mx.array(rng.normal(size=(p.n, p.n)).astype(np.float32)),
        v1=mx.array(rng.normal(size=(p.n, p.n)).astype(np.float32)),
    )
    ref = np.array(W.vorticity(ws, p))
    got = weather_obs.layer1_vorticity(np.array(ws.u1, np.float64), np.array(ws.v1, np.float64), p.dx)
    np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-4 * np.abs(ref).max())


def test_storm_exposure_is_the_mean_hazard_ramp() -> None:
    sp = np.array([0.0, 0.05, 0.06, 0.075, 0.09, 0.2])
    assert weather_obs.storm_exposure(sp, 0.06, 0.03) == pytest.approx((0 + 0 + 0 + 0.5 + 1 + 1) / 6)
    assert weather_obs.storm_exposure(np.zeros(0), 0.06, 0.03) == 0.0


# --- the block on hand-built weather states --------------------------------------------------------


def test_block_keys_3d_and_2d() -> None:
    c3, c2 = storm3d_config(), storm2d_config()
    b3 = _block(c3, _rest(c3), [0.01, 0.02], occupancy=np.array([3, 5, 2]), overlap_deep=4)
    b2 = _block(c2, _rest(c2), [0.01])
    assert set(b3) == KEYS_COMMON | KEYS_OCC and set(b2) == KEYS_COMMON
    assert (b3["occ_hz0"], b3["occ_hz1_11"], b3["occ_hz12"], b3["overlap_deep"]) == (3, 5, 2, 4)
    json.dumps(b3, allow_nan=False)  # plain floats and ints, nothing non-finite
    assert all(isinstance(v, int | float) for v in b3.values())


def test_the_rest_state_block(cfg: Config) -> None:
    wx = _rest(cfg)
    b = _block(cfg, wx)
    assert b["wind_max_l1"] == b["wind_max_l2"] == b["jet_max_l2"] == b["hyper_max"] == 0.0
    assert b["rain_total"] == b["rain_cells"] == b["vort_cells"] == 0.0 and b["storm_proxy"] == 0
    assert b["mass_rel"] == pytest.approx(0.0, abs=1e-7)
    assert b["q_total"] == pytest.approx(wx[6].astype(np.float64).sum())
    # no agents: the air-speed figures are 0.0, never None
    assert (b["storm_exposure"], b["air_speed_p99"], b["air_speed_max"]) == (0.0, 0.0, 0.0)


def test_winds_jet_and_hyperbolicity(cfg: Config) -> None:
    p = W.params_from_config(cfg)
    wx = np.array(fixed_wx(cfg, u1=0.03, v1=0.04, u2=0.01, v2=0.0))
    b = _block(cfg, wx)
    assert b["wind_max_l1"] == pytest.approx(0.05, rel=1e-6)
    assert b["wind_max_l2"] == pytest.approx(0.01, rel=1e-6)
    assert b["jet_max_l2"] == pytest.approx(0.01, rel=1e-6)  # the zonal mean of u2
    dv2 = (0.03 - 0.01) ** 2 + 0.04**2
    assert b["hyper_max"] == pytest.approx(dv2 / (p.g_prime * (p.H1 + p.H2)), rel=1e-5)
    # the jet is max_y |mean_x u2|: an eastward band at one latitude, westward at another
    wx2 = _rest(cfg)
    wx2[4, 10, :] = 0.05
    wx2[4, 20, :] = -0.07
    wx2[4, 30, ::2] = 0.2  # a zonal mean of 0.1 from a pattern that is not uniform in x
    assert _block(cfg, wx2)["jet_max_l2"] == pytest.approx(0.1, rel=1e-5)
    assert _block(cfg, wx2)["wind_max_l2"] == pytest.approx(0.2 / 2, rel=1e-5)  # the centre average of 0.2, 0


def test_rain_conservation_and_humidity(cfg: Config) -> None:
    p = W.params_from_config(cfg)
    wx = _rest(cfg)
    wx[7, 3, 4], wx[7, 3, 5], wx[7, 90, 9] = 2e-4, 5e-5, 1e-2  # two cells at or above p_th = 1e-4, one below
    b = _block(cfg, wx)
    assert b["rain_total"] == pytest.approx((2e-4 + 5e-5 + 1e-2) * p.dx**2, rel=1e-5)
    assert b["rain_cells"] == pytest.approx(2 / p.n**2)
    wx[0] *= 1.001
    assert _block(cfg, wx)["mass_rel"] == pytest.approx(0.001 * p.H1 / (p.H1 + p.H2), rel=2e-3)
    wx[6] += 0.5
    assert _block(cfg, wx)["q_total"] == pytest.approx(wx[6].astype(np.float64).sum())


def test_storm_proxy_counts_cyclonic_rainy_vortices_per_hemisphere(cfg: Config) -> None:
    """A cyclonic blob of area >= 25 with >= 4 raining cells is a candidate storm; without rain, with the
    wrong sign for its hemisphere, or too small, it is not. Two separate blobs count twice."""
    base = _rest(cfg)
    north, south = (80.0, 80.0), (80.0, 240.0)  # (x, y): the northern hemisphere has y < L/2
    wx = _vortex(cfg, base, *north, amp=-0.9)
    area = _block(cfg, wx)["vort_cells"] * W.params_from_config(cfg).n ** 2
    assert area >= 25, area  # the fixture is not vacuous: the vorticity mask is big enough
    assert _block(cfg, wx)["storm_proxy"] == 0  # cyclonic but dry
    assert _block(cfg, _rain_at(cfg, wx, *north, cells=2))["storm_proxy"] == 1  # 4 raining cells
    assert _block(cfg, _rain_at(cfg, wx, *north, cells=1))["storm_proxy"] == 0  # 1 raining cell < r_min
    two = _vortex(cfg, wx, 240.0, 80.0, amp=-0.9)
    two = _rain_at(cfg, _rain_at(cfg, two, *north, cells=2), 240.0, 80.0, cells=2)
    assert _block(cfg, two)["storm_proxy"] == 2
    anti = _rain_at(cfg, _vortex(cfg, base, *north, amp=+0.9), *north, cells=3)  # anticyclonic in the north
    assert _block(cfg, anti)["vort_cells"] == 0.0 and _block(cfg, anti)["storm_proxy"] == 0
    cyc_south = _rain_at(cfg, _vortex(cfg, base, *south, amp=+0.9), *south, cells=3)  # cyclonic in the south
    assert _block(cfg, cyc_south)["storm_proxy"] == 1
    tiny = _rain_at(cfg, _vortex(cfg, base, *north, amp=-0.9, sigma=2.0), *north, cells=2)
    b = _block(cfg, tiny)
    assert 0 < b["vort_cells"] * W.params_from_config(cfg).n ** 2 < 25 and b["storm_proxy"] == 0


def test_air_speed_figures(cfg: Config) -> None:
    sp = np.linspace(0.0, 0.1, 101)
    b = _block(cfg, _rest(cfg), sp)
    assert b["air_speed_max"] == pytest.approx(0.1) and b["air_speed_p99"] == pytest.approx(0.099)
    wc = cfg.weather
    ramp = np.clip((sp - wc.storm_u0) / wc.storm_span, 0, 1).mean()
    assert b["storm_exposure"] == pytest.approx(ramp)


# --- hash-grid occupancy ---------------------------------------------------------------------------


def test_hash_occupancy_masks_dead_rows_and_matches_the_search() -> None:
    cfg = storm3d_config()
    d = dims(cfg)
    s = init_state(cfg)
    counts = spatial.hash_occupancy(s.pos, s.alive, cfg)
    assert counts.shape == (d.g_hash_z, d.g_hash, d.g_hash) and int(counts.sum()) == int(s.alive.sum())
    nb = spatial.find_neighbors(s.pos, s.alive, cfg)
    assert int(counts.max()) == int(nb.max_occupancy)
    # every row counted (no mask) is w0's old binning; a dead row's stale position adds nothing with the mask
    assert int(spatial.hash_occupancy(s.pos, None, cfg).sum()) == d.n
    assert np.array_equal(
        np.array(w0.occupancy_counts(s.pos, cfg)), np.array(spatial.hash_occupancy(s.pos, None, cfg))
    )
    peaks, cells = spatial.band_peaks(counts)
    flat = np.array(counts).reshape(-1)
    assert all(flat[int(c)] == int(p) for p, c in zip(np.array(peaks), np.array(cells), strict=True) if p > 0)


# --- the snapshot's block --------------------------------------------------------------------------


def _snapshot(cfg: Config, steps: int = 3, prep=None):
    s = init_state(cfg)
    if prep is not None:
        s = prep(s)
    fn = make_step(cfg)
    for _ in range(steps):
        s, _ = fn(s)
    mx.eval(s)
    return s, metrics.snapshot_metrics(s, cfg, heavy=False)


def test_the_snapshot_has_the_block_only_with_weather_on(cfg: Config) -> None:
    _, snap = _snapshot(cfg)
    assert set(snap["weather"]) == KEYS_COMMON | (KEYS_OCC if dims(cfg).d3 else set())
    assert snap["weather"]["overlap_deep"] == 0
    assert snap["weather"]["mass_rel"] == pytest.approx(0.0, abs=1e-6)
    off = replace(cfg, weather=WeatherConfig())
    _, snap_off = _snapshot(off)
    assert "weather" not in snap_off
    assert set(snap) - {"weather"} == set(snap_off)  # nothing else about the row changes


def test_the_snapshot_block_follows_the_state(cfg: Config) -> None:
    """The air speed is the engine's sampling at the live agents; the wx figures are of the stored state."""

    def prep(s):
        return s._replace(wx=fixed_wx(cfg, u1=0.04, v1=0.0, u2=0.04, v2=0.0))

    s, snap = _snapshot(cfg, steps=1, prep=prep)
    b = snap["weather"]
    alive = to_numpy(s.alive)
    wind = W.agent_wind(W.unpack(s.wx), s.pos, s.buoy_cap if dims(cfg).d3 else None, s.alive, cfg)
    speed = to_numpy(wind.speed)[alive]
    assert b["air_speed_max"] == pytest.approx(float(speed.max()), rel=1e-6)
    assert b["air_speed_p99"] == pytest.approx(float(np.quantile(speed, 0.99)), rel=1e-6)
    # one weather step of friction has passed since the prescribed uniform wind
    assert b["air_speed_max"] > 0.0 and b["wind_max_l1"] == pytest.approx(0.04, rel=1e-2)
    assert b["jet_max_l2"] == pytest.approx(0.04, rel=1e-2)


def test_overlap_deep_counts_alive_agents_with_a_close_neighbour(cfg: Config) -> None:
    """Brute force on the positions: alive agents with another alive agent closer than 0.5 radius."""
    r = cfg.physics.radius

    def prep(s):
        pos = np.array(s.pos)
        pos[1] = pos[0] + np.eye(pos.shape[1])[0] * 0.3 * r  # rows 0 and 1 overlap deeply
        pos[3] = pos[2] + np.eye(pos.shape[1])[0] * 0.8 * r  # rows 2 and 3 touch, not deeply
        return s._replace(pos=mx.array(pos))

    s = prep(init_state(cfg))
    snap = metrics.snapshot_metrics(s, cfg, heavy=False)
    pos, alive = to_numpy(s.pos).astype(np.float64), to_numpy(s.alive)
    ids = np.flatnonzero(alive)
    dd = pos[ids][:, None, :] - pos[ids][None, :, :]
    dd = dd - cfg.world.size * np.round(dd / cfg.world.size) * np.array([1, 1, 0][: dd.shape[-1]])
    dist = np.sqrt((dd * dd).sum(-1)) + np.eye(ids.size) * 1e9
    want = int((dist.min(axis=1) < 0.5 * r).sum())
    assert want >= 2
    assert snap["weather"]["overlap_deep"] == want


def test_a_runner_row_carries_the_block(tmp_path) -> None:
    from alife import runner
    from alife.runio import RunDir

    cfg = storm3d_config("run.metrics_every=3", "run.heavy_metrics_every=6", "run.checkpoint_every=6")
    rd = RunDir.create(tmp_path / "run", cfg)
    res = runner.run(rd, cfg, init_state(cfg), 6, quiet=True)
    assert (res.status, res.exit_code) == ("done", runner.ExitCode.OK)
    rows = rd.metrics_rows()
    assert [r["step"] for r in rows] == [3, 6]
    for r in rows:
        assert set(r["weather"]) == KEYS_COMMON | KEYS_OCC and "deaths_storm" in r
        assert r["ledger_error"] == 0 and r["weather"]["mass_rel"] == pytest.approx(0.0, abs=1e-6)


# --- invariants._check_weather (spec §7.3) ---------------------------------------------------------


def test_full_check_flags_a_bad_weather_state(cfg: Config) -> None:
    s = init_state(cfg)
    assert invariants.full_check(s, cfg) == []

    def with_wx(mutate) -> list[str]:
        wx = np.array(s.wx)
        mutate(wx)
        return invariants.full_check(s._replace(wx=mx.array(wx)), cfg)

    def nan(wx):
        wx[2, 4, 4] = np.nan

    def thin(wx):
        wx[0, 5, 5] = 0.5  # below h_min = 2

    def dry(wx):
        wx[6, 7, 7] = -0.01

    def heavy(wx):
        wx[1] *= 1.01  # the column mass drifts by 0.6 percent

    assert any("non-finite" in m for m in with_wx(nan))
    assert any("h_min" in m for m in with_wx(thin))
    assert any("humidity" in m for m in with_wx(dry))
    assert any("column mass" in m for m in with_wx(heavy))
    # a state a few steps along is clean, and a weather-off config never reads the array
    s1, _ = _snapshot(cfg)
    assert invariants.full_check(s1, cfg) == []
    off = replace(cfg, weather=WeatherConfig())
    bad = np.array(s.wx)
    bad[2, 4, 4] = np.nan
    assert invariants.full_check(init_state(off)._replace(wx=mx.array(bad)), off) == []


# --- the benchmark (spec §8.2, §8.3) ---------------------------------------------------------------


def test_stages_for_weather_off_is_the_stage_table(tiny_cfg: Config) -> None:
    assert bench.stages_for(tiny_cfg) is bench.STAGES
    on = [name for name, _ in bench.stages_for(storm2d_config())]
    off = [name for name, _ in bench.STAGES]
    assert [n for n in on if n not in ("wind", "weather")] == off
    assert on.index("wind") == on.index("sense") + 1 and on.index("weather") == on.index("fields") - 1
    assert dict(bench.stages_for(storm2d_config()))["wind"] == "S3w"


def _staged_equals_fused(cfg: Config, steps: int = 4) -> None:
    cfg = replace(cfg, run=replace(cfg.run, compile=False))
    fused = make_step(cfg)
    staged = bench._StagedStep(cfg)
    s0 = init_state(cfg)
    s0 = s0._replace(
        birth_step=mx.where(s0.alive, -50, s0.birth_step),  # founders old enough to breed
        wx=fixed_wx(cfg, u1=0.06, v1=0.03, u2=0.06, v2=0.03),  # a wind that kills, breaks bonds and rains
    )
    mx.eval(s0)
    s, holder = s0, [s0]
    storm = births = 0
    for _ in range(steps):
        s, out = fused(s)
        out_staged = staged(holder)
        mx.eval(s, out, holder[0], out_staged)
        assert state_digest(holder[0]) == state_digest(s)
        for a, b in zip(tree_flatten(out), tree_flatten(out_staged), strict=True):
            assert a[0] == b[0]
            np.testing.assert_array_equal(to_numpy(a[1]), to_numpy(b[1]), err_msg=a[0])
        c = to_numpy(out.counters)
        births += int(c[CounterIdx.BIRTHS])
        storm += int(c[-1])
    assert out.counters.shape == (n_counters(dims(cfg).nd, True),)
    assert storm > 0, "the storm hazard was not exercised"
    assert births > 0
    assert set(staged.times) == {name for name, _ in bench.stages_for(cfg)}
    assert all(len(v) == steps for v in staged.times.values())


HAZARD = (
    "weather.storm_p=0.2",
    "weather.storm_u0=0.02",
    "weather.storm_span=0.02",
    "weather.storm_break_u=0.04",
    "weather.rain_gain=50000",
)


def test_staged_pipeline_equals_step_with_weather_2d() -> None:
    """The staged wiring mirrors every weather line of ``step.step`` (S3w, the break flag, the hazard mask,
    the wind force, the S17 update, the rain term and the health input): bitwise equal in eager mode."""
    _staged_equals_fused(storm2d_config(*HAZARD))


def test_staged_pipeline_equals_step_with_weather_3d() -> None:
    _staged_equals_fused(storm3d_config(*HAZARD))


def test_bench_weather_reports_the_marginal_cost() -> None:
    cfg = storm2d_config()
    r = bench.run_bench(cfg, steps=2, warmup=1, weather=True, stages=True)
    json.dumps(r, allow_nan=False)
    wt = r["weather"]
    assert set(wt) == {
        "warmup",
        "block_pairs",
        "steps_per_block",
        "off_ms_median",
        "on_ms_median",
        "marginal_ms_median",
        "off_ms_mean",
        "on_ms_mean",
        "marginal_ms_mean",
        "pair_marginal_ms",
        "steps_per_s_on",
    }
    assert wt["off_ms_median"] > 0 and wt["on_ms_median"] > 0
    assert wt["marginal_ms_median"] == pytest.approx(wt["on_ms_median"] - wt["off_ms_median"])
    assert wt["marginal_ms_mean"] == pytest.approx(wt["on_ms_mean"] - wt["off_ms_mean"])
    assert wt["steps_per_s_on"] == pytest.approx(1e3 / wt["on_ms_mean"])
    assert (wt["block_pairs"], wt["steps_per_block"], wt["warmup"]) == (2, 1, 1)
    assert len(wt["pair_marginal_ms"]) == 2
    assert (
        list(r["stages"]["ms"]) == [name for name, _ in bench.stages_for(cfg)] and "wind" in r["stages"]["ms"]
    )
    assert all(v > 0 for v in r["stages"]["ms"].values())
    text = bench.format_report(r)
    assert "weather (2 block pairs of 1 steps)" in text and "marginal" in text and "S3w" in text


def test_bench_weather_runs_the_blocks_off_on_on_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """The leading arm alternates in each pair, so a drift linear in time cancels from the difference."""
    cfg = storm2d_config()
    order: list[str] = []
    real = bench.init_state

    def spy(c: Config):
        order.append("on" if c.weather.enabled else "off")
        return real(c)

    monkeypatch.setattr(bench, "init_state", spy)
    bench._bench_weather(cfg, 2, 1)
    assert order == ["off", "on", "on", "off"]
    order.clear()
    wt = bench._bench_weather(cfg, 2, 1, pairs=3)
    assert order == ["off", "on", "on", "off", "off", "on"] and len(wt["pair_marginal_ms"]) == 3


def test_bench_weather_needs_the_weather_on(tiny_cfg: Config) -> None:
    with pytest.raises(ConfigError, match=r"weather\.enabled"):
        bench.run_bench(tiny_cfg, steps=1, warmup=0, weather=True)
    with pytest.raises(ValueError, match="weather_pairs"):
        bench.run_bench(storm2d_config(), steps=1, warmup=0, weather=True, weather_pairs=0)
    assert "weather" not in bench.run_bench(tiny_cfg, steps=1, warmup=0)


def test_the_bench_cli_takes_weather(capsys: pytest.CaptureFixture[str], tmp_path) -> None:
    args = ["bench", "--preset", "tiny", "--steps", "2", "--warmup", "1", "--weather"]
    storm = ["--set", "world.size=320", "--set", "weather.enabled=true", "--set", "weather.u_max=0.2"]
    out = tmp_path / "bench.json"
    assert cli.main([*args, *storm, "--json", str(out)]) == 0
    wt = json.loads(out.read_text())["weather"]
    assert wt["marginal_ms_median"] == pytest.approx(wt["on_ms_median"] - wt["off_ms_median"])
    capsys.readouterr()
    assert cli.main(args) == 2  # weather off: a config error, exit 2


# --- the smoke run (spec §9.11) --------------------------------------------------------------------

SMOKE = [
    *("run", "--preset", "64k3d", "--overlay", "storm", "--steps", "200", "--checks", "cheap", "--quiet"),
    *("--set", "population.n_max=1024", "--set", "population.n_init=256"),
    *("--set", "population.max_births_per_step=32", "--set", "run.metrics_every=25"),
    *("--set", "run.heavy_metrics_every=100", "--set", "run.checkpoint_every=100"),
]


@pytest.mark.slow
def test_the_smoke_run_writes_the_weather_block_and_matches_without_compile(tmp_path) -> None:
    """``alife run --preset 64k3d --overlay storm`` at 1,024 rows for 200 steps, compiled and eager: it
    finishes, no error diag bit, every row has the block and ``deaths_storm``, the ledger closes and the
    column mass holds. ``deaths_storm == 0`` is expected and not asserted (the hazard is inert at the
    overlay's values and the start is calm, spec §1.6). The compiled and eager states are **not** compared by
    digest: they diverge on the Mac CPU with the weather off as well (spec §9.8, stage 2 note), so what is
    compared is the agent-independent part, ``wx``, bitwise."""
    from alife.runio import RunDir, load_checkpoint
    from alife.state import ERROR_MASK

    final = {}
    for name, extra in (("compiled", []), ("eager", ["--no-compile"])):
        out = tmp_path / name
        assert cli.main([*SMOKE, *extra, "--out", str(out)]) == 0
        rd = RunDir.open(out)
        assert rd.meta()["status"] == "done"
        rows = rd.metrics_rows()
        assert [r["step"] for r in rows] == [25, 50, 75, 100, 125, 150, 175, 200]
        for r in rows:
            assert {"storm_exposure", "air_speed_p99", "air_speed_max", "jet_max_l2"} <= set(r["weather"])
            assert set(r["weather"]) == KEYS_COMMON | KEYS_OCC and "deaths_storm" in r
            assert r["ledger_error"] == 0 and not r["diag"] & ERROR_MASK
            assert abs(r["weather"]["mass_rel"]) < 1e-5
        final[name] = load_checkpoint(out / "checkpoints" / "step_0000000200.safetensors", rd.config())
    assert np.array_equal(to_numpy(final["compiled"].wx), to_numpy(final["eager"].wx))
