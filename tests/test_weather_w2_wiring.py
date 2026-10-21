"""W2 build step 2 (DESIGN §32, W2 spec §9.1(a), §9.2 to §9.6, §8.2, §13 item 2): the organism forcing wired
into the engine step, the metrics block additions and ``bench --weather-org``.

CPU and fast. The 3D case is ``64k3d`` + ``storm`` (256 rows, L = 320, FG = 80, FZ = 20); the 2D case is
``tiny`` at ``world.size = 320``. Part 1 pins the weather-on trajectory of the W1 engine (3799009): with every
new key at its default the step is that engine, bit for bit.
"""

from __future__ import annotations

import json
from dataclasses import replace

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from alife import bench, cli, invariants, metrics, weather_obs, weather_spinup
from alife import weather as W
from alife.config import Config, ConfigError, apply_overrides, dims
from alife.rng import InitStream, Stream, init_keys, step_keys
from alife.state import CounterIdx, init_state, state_digest, storm_counter_index, to_numpy
from alife.step import make_step
from helpers import run_checked
from weather_helpers import (
    STRONG,
    fixed_wx,
    spun_ws,
    storm2d_config,
    storm3d_config,
    trajectory_digest,
)

BIG_CAP = "fields.resource_cap=16000000"
# The couplings on, strong enough to bite at 64 agents: a small ``shade_n0`` and ``seed_c0``.
COUP = (
    "weather.seed_gain=0.05",
    "weather.seed_c0=0.5",
    "weather.shade_gain=0.03",
    "weather.shade_n0=1.0",
)


def _np(a: mx.array) -> np.ndarray:
    return to_numpy(a)


def _cfg(kind: str, *ov: str) -> Config:
    return storm3d_config(*ov) if kind == "3d" else storm2d_config(*ov)


@pytest.fixture(scope="module", params=["3d", "2d"])
def kind(request: pytest.FixtureRequest) -> str:
    return request.param


def _leaves(x) -> list[np.ndarray]:
    return [_np(a) for _, a in tree_flatten(x) if a is not None]


def _same(a, b) -> bool:
    la, lb = _leaves(a), _leaves(b)
    return len(la) == len(lb) and all(np.array_equal(p, q) for p, q in zip(la, lb, strict=True))


# --- 9.1(a): the weather-on trajectory is the W1 engine's -----------------------------------------

# (sha256 of the per-step state digests and step outputs, storm deaths, bonds broken) of 30 steps from a
# spun-up weather plus a 0.07 jet, computed with ``weather_helpers.trajectory_digest`` on a fresh ``git
# archive`` of 3799009 (the W1 engine, before any W2 key existed) and on the W2 head. CPU, MLX 0.32 stack.
PINS = {
    ("3d", "default", True): ("95721c288fdd70f4b1c516026c1d35b18cca1466d0fc11504d314ea76b9a0aea", 2, 1),
    ("3d", "default", False): ("5c8d7091bbef996de864897a965a2f8b19b0d582c4ad1c18dbc0a223a1940cf9", 2, 1),
    ("3d", "strong", True): ("c87ee6905663c8e0d80a284215079038ab6913414a13c3d6f444cc862f984d05", 60, 1),
    ("3d", "strong", False): ("d3d6e8b2ed090c3b10368975dd40d50d5b1f109dfb319ba05dea804534cde9f2", 60, 1),
    ("2d", "default", True): ("4fbfee1d744e214a74e9c1d4ed1066a0751342074b70c7638e2f3a04e45c38bf", 0, 1),
    ("2d", "default", False): ("5b41ead06a4fcf38258e8eb3c36f5b75e2146d92187aaf0efb6617ba1416af4d", 0, 1),
    ("2d", "strong", True): ("e69338b37064261fed9a94608894b8aa8ac32753da8a5d53e67c9a6f82310ae4", 198, 26),
    ("2d", "strong", False): ("c715b87c18074fe4d1b4fc1b36b331f89d339f7a6650b0b2123083aa9f3a47d0", 198, 26),
}


@pytest.mark.parametrize("compile", [True, False], ids=["compiled", "eager"])
@pytest.mark.parametrize("strength", ["default", "strong"])
def test_the_weather_on_trajectory_is_the_w1_engines(kind: str, strength: str, compile: bool) -> None:
    ov = [o for o in (STRONG if strength == "strong" else ()) if not (kind == "2d" and "rain_floor" in o)]
    cfg = _cfg(kind, *ov)
    wc = cfg.weather
    assert (wc.seed_gain, wc.shade_gain, wc.wind_scale) == (0.0, 0.0, 1.0)
    assert not (wc.rain_zonal or wc.storm_eddy or wc.wind_eddy_force)
    got = trajectory_digest(cfg, compile=compile)
    assert got == PINS[(kind, strength, compile)]
    if strength == "strong":
        assert got[1] > 0  # the hazard was exercised (the pin is not of a calm world)


# --- 9.3: advance ---------------------------------------------------------------------------------


def test_advance_without_the_couplings_is_w1s(kind: str) -> None:
    """Without ``qs`` and ``t_mult`` (what the step passes with both gains 0) ``advance`` is W1's expression:
    ``warm_pools`` and a ones ``qs``. Explicit ones are the same bitwise; a shade is not."""
    cfg = _cfg(kind)
    p = W.params_from_config(cfg)
    ws, key, t = spun_ws(cfg), mx.random.key(5), mx.array(7, dtype=mx.int32)
    ref = W.step(ws, W.WeatherForcing(W.warm_pools(p, t), mx.ones((p.n, p.n), dtype=mx.float32)), key, p)
    assert _same(W.advance(ws, t, key, cfg), ref)
    ones = mx.ones((p.n, p.n), dtype=mx.float32)
    assert _same(W.advance(ws, t, key, cfg, qs=ones, t_mult=ones), ref)
    assert W.organism_forcing(
        mx.zeros((3, 4, 4)), mx.zeros((1, 3)), mx.zeros(1, dtype=mx.bool_), ws, cfg
    ) == (
        None,
        None,
    )
    pos, alive = _agents(cfg, 40)
    strong = replace(cfg, weather=replace(cfg.weather, shade_gain=0.05, shade_n0=0.01))
    shade = W.shade_mult(pos, alive, strong)
    assert float((1.0 - shade).max()) > 0.01
    assert not _same(W.advance(ws, t, key, cfg, t_mult=shade)[0], ref[0])


def _agents(cfg: Config, n: int) -> tuple[mx.array, mx.array]:
    d = dims(cfg)
    rng = np.random.default_rng(3)
    pos = np.zeros((d.n, d.nd), np.float32)
    pos[:n, :2] = rng.uniform(100.0, 220.0, (n, 2))
    if d.d3:
        pos[:, 2] = d.z_lo
        pos[:n, 2] = rng.uniform(5.0, 70.0, n)
    return mx.array(pos), mx.array(np.arange(d.n) < n)


def test_organism_forcing_is_built_only_with_its_gain(kind: str) -> None:
    cfg = _cfg(kind)
    ws = spun_ws(cfg)
    chem = init_state(cfg).chem
    pos, alive = _agents(cfg, 20)
    fg = dims(cfg).fg
    for ov, want in (
        ((), (False, False)),
        (("weather.seed_gain=0.05",), (True, False)),
        (("weather.shade_gain=0.02",), (False, True)),
        (("weather.seed_gain=0.05", "weather.shade_gain=0.02"), (True, True)),
    ):
        c = _cfg(kind, *ov)
        qs, tm = W.organism_forcing(chem, pos, alive, ws, c)
        assert (qs is not None, tm is not None) == want
        n = W.params_from_config(c).n
        for a in (qs, tm):
            if a is not None:
                assert a.shape == (n, n) and a.dtype == mx.float32
    assert fg == 80


# --- 9.2, 9.5: the organism forcing is a function of the start-of-step state ----------------------


def _structured(cfg: Config, steps: int = 25):
    """A state with the weather spun up, some steps taken (agents have moved and emitted), and a c2 blob
    added so that the seeding has an excess to respond to."""
    s = init_state(cfg)._replace(wx=W.pack(spun_ws(cfg)))
    fn = make_step(cfg, compile=False)
    for _ in range(steps):
        s, out = fn(s)
        mx.eval(s, out)
    # the blob sits over the rainiest weather cell (the field lattice is half as fine as the weather grid)
    j, i = np.unravel_index(int(np.argmax(_np(W.unpack(s.wx).P))), (160, 160))
    j, i = j // 2, i // 2
    c = _np(s.chem).copy()
    if dims(cfg).d3:
        c[2, :, max(j - 3, 0) : j + 4, max(i - 3, 0) : i + 4] += 8.0
    else:
        c[2, max(j - 3, 0) : j + 4, max(i - 3, 0) : i + 4] += 8.0
    # twenty agents starve during the next step, so ``s.alive`` is not the end-of-step ``alive``
    rows = np.arange(dims(cfg).n) < 20
    energy = mx.where(mx.array(rows) & s.alive, 0, s.energy)
    return s._replace(chem=mx.array(c), energy=energy)


def _expected_wx(s, cfg: Config) -> mx.array:
    """The weather the step must produce, rebuilt from the start-of-step ``s`` with the engine's key chain."""
    _, ks = step_keys(s.key)
    kw = mx.random.split(ks[Stream.WEATHER], 2)
    ws = W.unpack(s.wx)
    qs, tm = W.organism_forcing(s.chem, s.pos, s.alive, ws, cfg)
    ws2, _ = W.advance(ws, s.step, kw[0], cfg, qs=qs, t_mult=tm)
    return W.pack(ws2)


def test_the_seeding_and_the_shade_read_the_start_of_step_state(kind: str) -> None:
    cfg = _cfg(kind, *COUP)
    s = _structured(cfg)
    s_next, _ = make_step(cfg, compile=False)(s)
    mx.eval(s_next)
    assert int(_np(s_next.alive).sum()) < int(_np(s.alive).sum())  # the end-of-step alive set differs
    assert np.array_equal(_np(s_next.wx), _np(_expected_wx(s, cfg)))
    # the forcing is not trivial, and reading the end-of-step state instead would give another weather
    ws = W.unpack(s.wx)
    qs, tm = W.organism_forcing(s.chem, s.pos, s.alive, ws, cfg)
    assert float((1.0 - qs).max()) > 0.0 and float((1.0 - tm).max()) > 0.0
    _, ks = step_keys(s.key)
    kw = mx.random.split(ks[Stream.WEATHER], 2)
    qs2, tm2 = W.organism_forcing(s_next.chem, s_next.pos, s_next.alive, ws, cfg)
    late, _ = W.advance(ws, s.step, kw[0], cfg, qs=qs2, t_mult=tm2)
    assert not np.array_equal(_np(s_next.wx), _np(W.pack(late)))
    # each coupling is wired on its own
    for one in COUP[:2], COUP[2:]:
        c1 = _cfg(kind, *one)
        n1, _ = make_step(c1, compile=False)(s)
        assert np.array_equal(_np(n1.wx), _np(_expected_wx(s, c1)))
        assert not np.array_equal(_np(n1.wx), _np(s_next.wx))


def test_the_couplings_change_the_weather_off_makes_it_agent_independent(kind: str) -> None:
    """Couplings on: another agent set gives another weather. Off: it does not (the W1 test 9.12 holds, and
    ``storm_p``, ``wind_scale``, ``wind_eddy_force`` and ``storm_eddy`` do not touch the weather either)."""

    def run(*ov: str, n_init: int = 64) -> np.ndarray:
        if kind == "3d":
            c = storm3d_config("run.seed=2", *ov, n_init=n_init)
        else:
            c = storm2d_config("run.seed=2", f"population.n_init={n_init}", *ov)
        s = init_state(c)._replace(wx=W.pack(spun_ws(c)))
        fn = make_step(c, compile=False)
        for _ in range(15):
            s, out = fn(s)
            mx.eval(s, out)
        return _np(s.wx)

    base = run()
    assert np.array_equal(base, run(n_init=12))  # W1: agent-independent
    for ov in (
        ("weather.storm_p=0.5", "weather.storm_u0=0.0", "weather.storm_span=0.02"),
        ("weather.wind_scale=0.0",),
        ("weather.wind_eddy_force=true",),
        ("weather.storm_eddy=true", "weather.storm_p=0.5", "weather.storm_u0=0.0", "weather.storm_span=0.02"),
    ):
        assert np.array_equal(base, run(*ov)), ov
    for coupling in (COUP[2:], COUP[:2], COUP):
        assert not np.array_equal(run(*coupling), run(*coupling, n_init=12)), coupling
        assert not np.array_equal(run(*coupling), base), coupling


def test_the_w1_agent_independence_tests_run_with_the_couplings_off() -> None:
    """Test 9.12 of W1 (the standalone chain reproduces ``wx``) is gated on both gains being 0."""
    for c in (storm3d_config("run.seed=2"), storm2d_config("run.seed=2")):
        assert c.weather.seed_gain == 0.0 and c.weather.shade_gain == 0.0


def test_a_fork_with_seeding_starts_from_the_same_spun_up_weather(kind: str) -> None:
    """``weather_spinup`` runs feedback-free: the spin-up of a config with seeding and shade on is the one
    without them (``advance`` is called without ``qs`` and ``t_mult``)."""
    key = init_keys(7)[InitStream.STATE_KEY]
    off, on = _cfg(kind), _cfg(kind, *COUP)
    a = weather_spinup.spinup(off, 12, step=40, key=key)
    b = weather_spinup.spinup(on, 12, step=40, key=key)
    assert np.array_equal(_np(a), _np(b))


# --- 9.4: wx unchanged by the eddy force ----------------------------------------------------------


def test_the_eddy_force_moves_the_agents_and_not_the_weather(kind: str) -> None:
    def run(*ov: str):
        c = _cfg(kind, *ov)
        s = init_state(c)._replace(wx=W.pack(spun_ws(c)))
        fn = make_step(c, compile=False)
        for _ in range(12):
            s, out = fn(s)
            mx.eval(s, out)
        return s

    a, b = run(), run("weather.wind_eddy_force=true")
    assert np.array_equal(_np(a.wx), _np(b.wx))
    assert not np.array_equal(_np(a.pos), _np(b.pos))  # the agents did feel the other force


# --- 9.6: the ledger ------------------------------------------------------------------------------


def test_the_ledger_closes_with_seeding_shade_rain_storms_catastrophe_and_immigration() -> None:
    cfg = storm3d_config(
        "director.enabled=true",
        "weather.rain_gain=10000000",
        "weather.storm_p=0.02",
        "weather.storm_u0=0.0",
        "weather.storm_span=0.02",
        "weather.storm_eddy=true",
        "weather.wind_eddy_force=true",
        "population.immigration_floor=120",
        *COUP,
        BIG_CAP,
    )
    c = cfg.world.size / 2
    s0 = init_state(cfg)._replace(
        wx=W.pack(spun_ws(cfg)),
        env_cat_center=mx.array([c, c], dtype=mx.float32),
        env_cat_radius=mx.array(cfg.world.size / 2, dtype=mx.float32),
        env_cat_kill_p=mx.array(0.02, dtype=mx.float32),
    )
    s, hist = run_checked(cfg, 300, s0)  # full_check(prev, out) every step, no error diag bit
    counters = [_np(o.counters) for _, _, o in hist]
    assert sum(int(c[storm_counter_index(3)]) for c in counters) > 0
    assert sum(int(c[CounterIdx.DEATHS_CATASTROPHE]) for c in counters) > 0
    assert sum(int(c[CounterIdx.IMMIGRANTS]) for c in counters) > 0
    assert invariants.full_check(s, cfg) == []
    # both couplings were live at the end
    assert float(W.shade_strength(s.pos, s.alive, cfg).max()) > 0.0
    assert float(W.seeding_response(s.chem, W.unpack(s.wx), cfg).max()) > 0.0


# --- 8.2: the metric definitions ------------------------------------------------------------------


def test_rain_shares_from_hand_sums() -> None:
    # layers 0..19 in 3D: the band is layers 8-13
    base = np.full(20, 1000, np.int64)
    rain = np.zeros(20, np.int64)
    rain[8:14] = 500
    rain[0] = 100
    real = rain // 2
    inc = base + real
    got = weather_obs.rain_shares(np.stack([base, rain, inc, real]))
    band = slice(8, 14)
    assert got["rain_band_share"] == pytest.approx(3000 / (3000 + 6000))
    assert got["rain_global_share"] == pytest.approx(3100 / (3100 + 20000))
    assert got["rain_real_band_share"] == pytest.approx(1500 / (inc[band].sum()))
    assert got["rain_real_global_share"] == pytest.approx(real.sum() / inc.sum())
    assert got["rain_clipped_frac"] == pytest.approx(1.0 - real.sum() / rain.sum())
    # no rain: shares 0, nothing clipped
    zero = np.zeros(20, np.int64)
    assert set(weather_obs.rain_shares(np.stack([base, zero, base, zero])).values()) == {0.0}
    # 2D: the single layer is the band
    one = weather_obs.rain_shares(np.array([[900], [100], [950], [50]]))
    assert one["rain_band_share"] == one["rain_global_share"] == pytest.approx(0.1)
    assert one["rain_clipped_frac"] == pytest.approx(0.5)


def test_the_rain_audit_on_a_hand_field(kind: str) -> None:
    """A saturated cell has ``rain_real = 0`` and an unsaturated one has the nominal rain; the nominal row is
    ``weather.rain_regen`` of the stored weather and the base row is the field's own ``regen_rate``."""
    cfg = _cfg(kind, "weather.rain_gain=1000000")
    ws = spun_ws(cfg)
    s = init_state(cfg)._replace(wx=W.pack(ws), step=mx.array(5, dtype=mx.int32))
    cap = cfg.fields.resource_cap
    empty = s._replace(resource=mx.zeros_like(s.resource))
    full = s._replace(resource=mx.full(s.resource.shape, cap, dtype=s.resource.dtype))
    rain = _np(W.rain_regen(W.rain_in(ws, cfg), cfg)).astype(np.int64)
    fg = dims(cfg).fg
    per_layer = rain.reshape(-1, fg, fg).sum(axis=(1, 2))
    assert per_layer.sum() > 0
    a0 = _np(metrics.rain_audit(empty, ws, cfg)).astype(np.int64)
    assert a0.shape == (4, rain.reshape(-1, fg, fg).shape[0])
    assert np.array_equal(a0[1], per_layer)
    assert np.array_equal(a0[3], a0[1])  # far below the cap: all of the rain is realised
    assert np.array_equal(a0[2], a0[0] + a0[1])
    from alife import fields

    base = _np(fields.regen_rate(s.patch_pos, cfg, mx.array(4, dtype=mx.int32), None)).astype(np.int64)
    assert np.array_equal(a0[0], base.reshape(-1, fg, fg).sum(axis=(1, 2)))
    a1 = _np(metrics.rain_audit(full, ws, cfg))
    assert (
        np.array_equal(a1[1], per_layer) and not a1[2].any() and not a1[3].any()
    )  # saturated: nothing booked
    # a mixed field: every cell has its own room; the realised rain is a numpy double of the engine's rule
    room = (np.arange(s.resource.size) * 37 % 300).reshape(s.resource.shape).astype(np.int64)
    mid = _np(metrics.rain_audit(s._replace(resource=mx.array((cap - room).astype(np.int32))), ws, cfg))
    base3 = base.reshape(rain.shape)
    inc_total = np.maximum(np.minimum(base3 + rain, room), 0)
    real = inc_total - np.maximum(np.minimum(base3, room), 0)
    sums = [a.reshape(-1, fg, fg).sum(axis=(1, 2)) for a in (base3, rain, inc_total, real)]
    assert np.array_equal(mid.astype(np.int64), np.stack(sums))
    assert (mid[3] <= mid[1]).all()


def _snapshot(cfg: Config, wx: mx.array, steps: int = 2):
    s = init_state(cfg)._replace(wx=wx)
    fn = make_step(cfg)
    for _ in range(steps):
        s, _ = fn(s)
    mx.eval(s)
    return s, metrics.snapshot_metrics(s, cfg, heavy=False)


SPEED_KEYS = {"total_speed_p99", "eddy_speed_p99"}
SEED_KEYS = {"seed_r_mean", "seed_r_hi_share", "seed_lower_mean"}
SHADE_KEYS = {"shade_s_mean", "shade_s_max"}
RAIN_KEYS = {
    "rain_band_share",
    "rain_global_share",
    "rain_real_band_share",
    "rain_real_global_share",
    "rain_clipped_frac",
}


def test_the_snapshot_reports_each_key_group_with_its_coupling(kind: str) -> None:
    wx = W.pack(spun_ws(_cfg(kind)))
    plain = _cfg(kind, "weather.rain_gain=0")
    _, snap = _snapshot(plain, wx)
    assert set(snap["weather"]) >= SPEED_KEYS and not (SEED_KEYS | SHADE_KEYS | RAIN_KEYS) & set(
        snap["weather"]
    )
    _, snap = _snapshot(_cfg(kind, "weather.rain_gain=1000000"), wx)
    assert set(snap["weather"]) >= RAIN_KEYS and not (SEED_KEYS | SHADE_KEYS) & set(snap["weather"])
    s, snap = _snapshot(_cfg(kind, *COUP), wx)
    b = snap["weather"]
    assert set(b) >= (SPEED_KEYS | SEED_KEYS | SHADE_KEYS)
    cfg = _cfg(kind, *COUP)
    r = _np(W.seeding_response(s.chem, W.unpack(s.wx), cfg))
    sh = _np(W.shade_strength(s.pos, s.alive, cfg))
    assert b["seed_r_mean"] == pytest.approx(float(r.mean()), rel=1e-6)
    assert b["seed_r_hi_share"] == pytest.approx(float((r > 0.5).mean()))
    assert b["seed_lower_mean"] == pytest.approx(cfg.weather.seed_gain * float(r.mean()), rel=1e-6)
    assert b["shade_s_max"] == pytest.approx(float(sh.max()), rel=1e-6) and b["shade_s_max"] > 0
    assert b["shade_s_mean"] == pytest.approx(float(sh.mean()), rel=1e-6)
    assert b["shade_s_max"] <= cfg.weather.shade_gain


def test_the_snapshot_rain_audit_reads_the_directors_regeneration_lever() -> None:
    """With the director on, the audit's ``base`` row is ``regen_rate`` with the director's regen lever."""
    cfg = storm3d_config("director.enabled=true", "weather.rain_gain=1000000")
    wx = W.pack(spun_ws(cfg))
    s0 = init_state(cfg)._replace(wx=wx)
    half = s0._replace(env_regen_mult=mx.array(0.5, dtype=mx.float32))
    a1 = _np(metrics.rain_audit(s0, W.unpack(wx), cfg)).astype(np.int64)
    a2 = _np(metrics.rain_audit(half, W.unpack(wx), cfg)).astype(np.int64)
    assert a2[0].sum() < a1[0].sum() and np.array_equal(
        a1[1], a2[1]
    )  # the lever scales the base, not the rain
    _, snap = _snapshot(cfg, wx)
    assert set(snap["weather"]) >= RAIN_KEYS


def test_the_snapshot_speeds_are_total_and_eddy_whatever_the_hazard_reads(kind: str) -> None:
    """A zonally uniform wind has zero eddy speed and the full total speed; the hazard keys (``air_speed_*``)
    follow ``storm_eddy`` and the new ones do not."""
    wx = fixed_wx(_cfg(kind), u1=0.04, v1=0.0, u2=0.04, v2=0.0)
    for eddy in (False, True):
        cfg = _cfg(kind, f"weather.storm_eddy={'true' if eddy else 'false'}")
        s, snap = _snapshot(cfg, wx, steps=1)
        b = snap["weather"]
        alive = _np(s.alive)
        ws = W.unpack(s.wx)
        tot, edd = W.air_speeds(ws, s.pos, s.buoy_cap if dims(cfg).d3 else None, s.alive, cfg)
        assert b["total_speed_p99"] == pytest.approx(float(np.quantile(_np(tot)[alive], 0.99)), rel=1e-6)
        assert b["eddy_speed_p99"] == pytest.approx(float(np.quantile(_np(edd)[alive], 0.99)), rel=1e-6)
        assert b["total_speed_p99"] > 0.01 and b["eddy_speed_p99"] < 0.05 * b["total_speed_p99"]
        want = b["eddy_speed_p99"] if eddy else b["total_speed_p99"]
        assert b["air_speed_p99"] == pytest.approx(want, rel=1e-6)


def test_air_speeds_equal_agent_wind_under_each_hazard_reference(kind: str) -> None:
    ws = spun_ws(_cfg(kind))
    for eddy in (False, True):
        cfg = _cfg(kind, f"weather.storm_eddy={'true' if eddy else 'false'}")
        s = init_state(cfg)
        bc = s.buoy_cap if dims(cfg).d3 else None
        tot, edd = W.air_speeds(ws, s.pos, bc, s.alive, cfg)
        f = W.agent_wind(ws, s.pos, bc, s.alive, cfg)
        assert np.array_equal(_np(f.speed), _np(edd if eddy else tot))
    assert float(edd.max()) > 0 and not np.array_equal(_np(tot), _np(edd))


# --- the staged benchmark mirrors the new S17 -----------------------------------------------------


def test_staged_pipeline_equals_step_with_the_couplings(kind: str) -> None:
    """``bench._StagedStep`` evaluates ``organism_forcing`` on the start-of-step state like ``step.step``:
    bitwise equal in eager mode with seeding, shade, rain and the eddy hazard on."""
    from test_weather_obs import HAZARD, _staged_equals_fused

    cfg = _cfg(
        kind,
        *HAZARD,
        *COUP,
        "weather.storm_eddy=true",
        "weather.wind_eddy_force=true",
        "weather.storm_u0=0.005",
    )
    _staged_equals_fused(cfg, wx=W.pack(spun_ws(cfg)))  # a zonally uniform wind has no eddy to kill with


# --- bench --weather-org --------------------------------------------------------------------------


def test_bench_weather_org_reports_the_marginal_cost_of_the_couplings() -> None:
    cfg = storm2d_config(*COUP)
    r = bench.run_bench(cfg, steps=2, warmup=1, weather_org=True)
    json.dumps(r, allow_nan=False)
    assert "weather" not in r
    wt = r["weather_org"]
    assert wt["off_ms_median"] > 0 and wt["on_ms_median"] > 0
    assert wt["marginal_ms_median"] == pytest.approx(wt["on_ms_median"] - wt["off_ms_median"])
    assert (wt["block_pairs"], wt["steps_per_block"], wt["warmup"]) == (2, 1, 1)
    text = bench.format_report(r)
    assert "weather-org (seeding and shade) (2 block pairs of 1 steps)" in text and "marginal" in text


def test_bench_weather_org_keeps_the_weather_on_in_the_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = storm2d_config(*COUP)
    seen: list[tuple[bool, float, float]] = []
    real = bench.init_state

    def spy(c: Config):
        seen.append((c.weather.enabled, c.weather.seed_gain, c.weather.shade_gain))
        return real(c)

    monkeypatch.setattr(bench, "init_state", spy)
    bench._bench_weather(cfg, 2, 1, org=True)
    assert seen == [(True, 0.0, 0.0), (True, 0.05, 0.03), (True, 0.05, 0.03), (True, 0.0, 0.0)]


def test_bench_weather_org_needs_a_coupling_on(tiny_cfg: Config) -> None:
    with pytest.raises(ConfigError, match=r"weather-org needs weather\.enabled"):
        bench.run_bench(tiny_cfg, steps=1, warmup=0, weather_org=True)
    with pytest.raises(ConfigError, match=r"seed_gain or weather\.shade_gain"):
        bench.run_bench(storm2d_config(), steps=1, warmup=0, weather_org=True)


def test_the_bench_cli_takes_weather_org(capsys: pytest.CaptureFixture[str], tmp_path) -> None:
    args = ["bench", "--preset", "tiny", "--steps", "2", "--warmup", "1", "--weather-org"]
    storm = ["--set", "world.size=320", "--set", "weather.enabled=true", "--set", "weather.u_max=0.2"]
    out = tmp_path / "bench.json"
    assert cli.main([*args, *storm, "--set", "weather.shade_gain=0.02", "--json", str(out)]) == 0
    wt = json.loads(out.read_text())["weather_org"]
    assert wt["marginal_ms_median"] == pytest.approx(wt["on_ms_median"] - wt["off_ms_median"])
    capsys.readouterr()
    assert cli.main([*args, *storm]) == 2  # no coupling on: a config error, exit 2
    assert cli.main(args) == 2  # weather off


def test_apply_overrides_reaches_the_new_keys() -> None:
    cfg = apply_overrides(storm3d_config(), ["weather.seed_gain=0.03", "weather.shade_gain=0.02"])
    assert (cfg.weather.seed_gain, cfg.weather.shade_gain) == (0.03, 0.02)
    assert state_digest(init_state(cfg)) == state_digest(
        init_state(storm3d_config())
    )  # the init is unchanged
