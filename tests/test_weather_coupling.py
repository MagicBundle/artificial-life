"""W1 build step 3 (DESIGN §30, spec §9.5 to §9.7, §9.9, §9.12): rain-fed regrowth, the storm hazard, the
``DEATHS_STORM`` counter and the wind bond-break flag.

The hazard and the bond rule are exercised with a prescribed wind (a ``fixed_wx`` state), because the
overlay's own values are numerically inert in a real storm field (spec §1.6). CPU; the 300-step ledger run is
``slow``.
"""

from __future__ import annotations

from dataclasses import replace

import mlx.core as mx
import numpy as np
import pytest

from alife import fields, lifecycle
from alife import weather as W
from alife.config import Config, WeatherConfig, apply_overrides, dims
from alife.metrics import carrying_capacity, floor_carrying_capacity, interval_row, rain_of
from alife.rng import InitStream, Stream, init_keys, step_keys
from alife.state import (
    N_COUNTERS,
    CounterIdx,
    DeathCause,
    Diag,
    LedgerIdx,
    init_state,
    n_counters,
    storm_counter_index,
    to_numpy,
)
from alife.step import make_step
from helpers import ON_GPU, ledger_net, make_scene, run_checked
from weather_helpers import fixed_wx, storm2d_config, storm3d_config

STORM_ALL = ("weather.storm_p=1", "weather.storm_u0=0", "weather.storm_span=1e-06")


def _np(a: mx.array) -> np.ndarray:
    return to_numpy(a)


@pytest.fixture(scope="module", params=["3d", "2d"])
def kind(request: pytest.FixtureRequest) -> str:
    return request.param


def _cfg(kind: str, *overrides: str) -> Config:
    return storm3d_config(*overrides) if kind == "3d" else storm2d_config(*overrides)


# --- counters (spec §6.2, §9.5) ---------------------------------------------------------------------------


def test_counter_widths_and_the_storm_index() -> None:
    assert (n_counters(2), n_counters(3)) == (27, 30) == (N_COUNTERS, N_COUNTERS + 3)
    assert (n_counters(2, True), n_counters(3, True)) == (28, 31)
    assert (storm_counter_index(2), storm_counter_index(3)) == (27, 30)
    assert DeathCause.STORM == 4 and [c.value for c in DeathCause] == [0, 1, 2, 3, 4]


def test_step_counters_have_the_weather_width_only_with_weather_on(kind: str) -> None:
    nd = 3 if kind == "3d" else 2
    on = _cfg(kind)
    off = replace(on, weather=WeatherConfig())
    for cfg, weather in ((on, True), (off, False)):
        s, out = make_step(cfg, compile=False)(init_state(cfg))
        mx.eval(s, out)
        assert out.counters.shape == (n_counters(nd, weather),) and out.counters.dtype == mx.int32
    # weather-off counters are the first entries of the weather-on ones (nothing moved), the storm count last
    assert n_counters(nd, True) == n_counters(nd) + 1


def test_interval_row_has_deaths_storm_only_with_weather() -> None:
    rs = np.random.default_rng(3)
    for nd in (2, 3):
        c = rs.integers(0, 50, (4, n_counters(nd, True))).astype(np.int32)
        lg = np.zeros((4, 16), np.int64)
        row = interval_row(8, c, lg, {}, {}, nd=nd, weather=True)
        assert row["deaths_storm"] == int(c[:, storm_counter_index(nd)].sum())
        keys = list(row)
        assert keys[keys.index("child_bonds") + 1] == "deaths_storm"
        with pytest.raises(ValueError, match="columns"):  # a 28/31-wide array is refused when weather is off
            interval_row(8, c, lg, {}, {}, nd=nd)
        with pytest.raises(ValueError, match="columns"):  # and a 27/30-wide one when it is on
            interval_row(8, c[:, :-1], lg, {}, {}, nd=nd, weather=True)
        off = interval_row(8, c[:, :-1], lg, {}, {}, nd=nd)
        assert "deaths_storm" not in off
        assert {k: v for k, v in row.items() if k != "deaths_storm"} == off


def test_status_line_vitals_and_phylo_count_storm_deaths() -> None:
    from alife.director import vitals
    from alife.runner import _status_line

    row = {
        "step": 10, "n_alive": 5.0, "births": 1, "bonds_total": None, "deaths_starved": 1,
        "deaths_killed": 2, "deaths_aged": 3, "deaths_catastrophe": 4, "deaths_storm": 5, "steps": 10,
        "ledger_error": 0,
    }  # fmt: skip
    assert "deaths 15" in _status_line(row)
    assert "deaths 10" in _status_line({k: v for k, v in row.items() if k != "deaths_storm"})
    w = vitals.aggregate([row])
    assert w.deaths == 2 + 1 + 3 + 4 + 5  # killed + starved + aged + catastrophe + storm


# --- the hazard mask (spec §6.1, §9.5) --------------------------------------------------------------------


def test_storm_mask_is_a_linear_ramp_in_the_air_speed() -> None:
    """``p = storm_p·clip((speed − u0)/span, 0, 1)``: the mean of the mask over many keys matches it per
    speed (binomial tolerance), is zero below ``u0`` and at ``storm_p = 0``, and dead rows never draw in."""
    cfg = storm3d_config("weather.storm_p=0.5", "weather.storm_u0=0.04", "weather.storm_span=0.04")
    speeds = np.array([0.0, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.3], np.float32)
    n_rows, reps = 4 * len(speeds), 400
    sp = np.tile(speeds, 4)
    alive = np.ones(n_rows, bool)
    alive[-len(speeds) :] = False  # the last block is dead
    hits = np.zeros(n_rows)
    for k in range(reps):
        key = mx.random.key(k)
        hits += _np(lifecycle.storm_mask(mx.array(sp), mx.array(alive), key, cfg))
    got = hits.reshape(4, -1)[:3].sum(axis=0) / (3 * reps)
    want = 0.5 * np.clip((speeds - 0.04) / 0.04, 0, 1)
    sigma = np.sqrt(np.maximum(want * (1 - want), 1e-12) / (3 * reps))
    assert (np.abs(got - want) <= 5 * sigma + 1e-9).all(), (got, want)
    assert got[:3].max() == 0.0 and got[-1] == pytest.approx(0.5, abs=5 * sigma[-1])
    assert hits.reshape(4, -1)[3].sum() == 0  # dead rows
    off = storm3d_config("weather.storm_p=0")
    assert not _np(lifecycle.storm_mask(mx.array(sp), mx.array(alive), mx.random.key(1), off)).any()


def _death_args(cfg: Config, rows: list[dict]):
    d = dims(cfg)
    s = make_scene(cfg, [{"row": r["row"]} for r in rows])
    n = d.n
    solvent, e = np.ones(n, bool), np.zeros(n, np.int32)
    lost = np.zeros(n, np.int32)
    for r in rows:
        solvent[r["row"]] = r.get("solvent", True)
        e[r["row"]] = r.get("E", 100)
        lost[r["row"]] = r.get("lost", 0)
    return s, mx.array(solvent), mx.array(e), mx.array(lost)


def test_deaths_storm_cause_precedence() -> None:
    """A storm kills only agents that would otherwise survive; a row in both the storm and the catastrophe
    mask dies as ``CATASTROPHE``. The corpse is ``E + S`` through the usual path."""
    cfg = storm3d_config()
    d, S = dims(cfg), cfg.energy.structure
    rows = [
        dict(row=0, E=100),  # in the storm: STORM
        dict(row=1, E=100, solvent=False),  # insolvent and in the storm: STARVED
        dict(row=2, E=0, lost=10),  # E == 0 after a loss and in the storm: KILLED
        dict(row=3, E=100),  # storm and catastrophe: CATASTROPHE
        dict(row=4, E=100),  # catastrophe only: CATASTROPHE
        dict(row=5, E=100),  # nothing: survives
        dict(row=6, E=0),  # E == 0, in the storm: STARVED
    ]
    s, solvent, e, lost = _death_args(cfg, rows)
    zeros = mx.zeros((d.n,), dtype=mx.int32)

    def run(storm, cat):
        return lifecycle.deaths(
            s.alive, solvent, e, lost, zeros, s.uid, s.generation, s.clade, s.partner, s.rev,
            s.resource, zeros, cfg, cat_mask=cat, storm_mask=storm,
        )  # fmt: skip

    def mask(rows_):
        m = np.zeros(d.n, bool)
        m[rows_] = True
        return mx.array(m)

    both = run(mask([0, 1, 2, 3, 6]), mask([3, 4]))
    c = _np(both.cause)
    assert c[:7].tolist() == [
        DeathCause.STORM, DeathCause.STARVED, DeathCause.KILLED, DeathCause.CATASTROPHE,
        DeathCause.CATASTROPHE, -1, DeathCause.STARVED,
    ]  # fmt: skip
    assert _np(both.corpse)[0] == 100 + S and _np(both.corpse)[5] == 0
    only = run(mask([0, 3]), None)  # no director: a storm row still dies as STORM
    assert _np(only.cause)[[0, 3, 5]].tolist() == [DeathCause.STORM, DeathCause.STORM, -1]
    plain = run(None, None)
    assert _np(plain.cause)[[0, 3, 5]].tolist() == [-1, -1, -1]
    nothing = run(mask([]), None)  # an all-false mask changes nothing
    assert np.array_equal(_np(nothing.cause), _np(plain.cause))
    assert np.array_equal(_np(nothing.corpse), _np(plain.corpse))


# --- the hazard in the engine step ------------------------------------------------------------------------


def _with_wind(cfg: Config, **winds: float):
    """A ``pre`` hook of ``run_checked`` that prescribes a uniform wind before every step."""
    wx = fixed_wx(cfg, **winds)
    return lambda i, s: s._replace(wx=wx)


def test_every_agent_in_the_wind_dies_as_storm_and_the_ledger_closes(kind: str) -> None:
    cfg = _cfg(kind, *STORM_ALL)
    s0 = init_state(cfg)
    n0 = int(_np(s0.alive.sum()))
    _, hist = run_checked(cfg, 1, s0, pre=_with_wind(cfg, u1=0.1, u2=0.1))
    prev, s1, out = hist[0]
    nd = 3 if kind == "3d" else 2
    c = _np(out.counters)
    assert c[storm_counter_index(nd)] == n0 and c[CounterIdx.N_ALIVE] == 0
    assert c[CounterIdx.DEATHS_STARVED] == c[CounterIdx.DEATHS_KILLED] == c[CounterIdx.DEATHS_AGED] == 0
    dv = _np(out.deaths.valid)
    assert (_np(out.deaths.cause)[dv] == DeathCause.STORM).all() and dv.sum() == min(n0, dims(cfg).dl)
    # the corpses (E + S) are booked in FLOW_CORPSE (run_checked checked that the ledger closes)
    led = _np(out.ledger)
    assert led[LedgerIdx.FLOW_CORPSE] >= n0 * cfg.energy.structure
    assert ledger_net(out.ledger, cfg) == int(_np(s1.total)) - int(_np(prev.total))
    assert int(_np(s1.diag)) & 0xFFFF == 0


def test_below_the_threshold_nobody_dies(kind: str) -> None:
    cfg = _cfg(kind, "weather.storm_p=1", "weather.storm_u0=0.2", "weather.storm_span=0.01")
    _, hist = run_checked(cfg, 3, None, pre=_with_wind(cfg, u1=0.1, u2=0.1))
    nd = 3 if kind == "3d" else 2
    for _, _, out in hist:
        assert _np(out.counters)[storm_counter_index(nd)] == 0
        assert (_np(out.deaths.cause)[_np(out.deaths.valid)] != DeathCause.STORM).all()


def test_storm_p_zero_and_the_unused_hazard_leave_the_run_alone(kind: str) -> None:
    """With the hazard at 0 the mask is all false (the draw is still made): no storm death, and the run is
    the inert-hazard run."""
    cfg = _cfg(kind, "weather.storm_p=0")
    _, hist = run_checked(cfg, 3, None, pre=_with_wind(cfg, u1=0.2, u2=0.2))
    nd = 3 if kind == "3d" else 2
    assert all(_np(o.counters)[storm_counter_index(nd)] == 0 for _, _, o in hist)


def test_a_director_catastrophe_beats_the_storm_in_the_engine() -> None:
    """3D with the director on: a catastrophe disc over every agent (``cat_kill_p = 1``) and a storm on every
    agent (``storm_p = 1``): all die as ``CATASTROPHE``, none as ``STORM``; the counters agree."""
    cfg = storm3d_config("director.enabled=true", *STORM_ALL)
    assert cfg.weather.rain_floor == 0.0 and cfg.director.enabled  # the C17 rule's allowed side
    c = cfg.world.size / 2
    agents = [{"pos": [c + 8.0 * (k % 5) - 16, c + 8.0 * (k // 5) - 16, 40.0]} for k in range(20)]
    s0 = make_scene(cfg, agents)._replace(
        env_cat_center=mx.array([c, c], dtype=mx.float32),
        env_cat_radius=mx.array(cfg.world.size / 2, dtype=mx.float32),
        env_cat_kill_p=mx.array(1.0, dtype=mx.float32),
    )
    n0 = int(_np(s0.alive.sum()))
    _, hist = run_checked(cfg, 1, s0, pre=_with_wind(cfg, u1=0.1, u2=0.1))
    out = hist[0][2]
    cnt = _np(out.counters)
    assert cnt[CounterIdx.DEATHS_CATASTROPHE] == n0 and cnt[storm_counter_index(3)] == 0
    dv = _np(out.deaths.valid)
    assert (_np(out.deaths.cause)[dv] == DeathCause.CATASTROPHE).all()


def test_the_hazard_draws_from_the_second_half_of_the_weather_split(kind: str) -> None:
    """``kw = split(ks[WEATHER], 2)``: ``kw[0]`` is the disturbance key (S17), ``kw[1]`` the hazard's. The
    storm deaths of one step are the mask of ``kw[1]``, recomputed here from the state key; the mask of
    ``kw[0]`` is a different set, so reading the wrong half fails."""
    cfg = _cfg(kind, "weather.storm_p=1", "weather.storm_u0=0", "weather.storm_span=0.25")
    nd = 3 if kind == "3d" else 2
    s = init_state(cfg)._replace(wx=fixed_wx(cfg, u1=0.1, u2=0.1))
    _, ks = step_keys(s.key)
    kw = mx.random.split(ks[Stream.WEATHER], 2)
    wind = W.agent_wind(W.unpack(s.wx), s.pos, s.buoy_cap if kind == "3d" else None, s.alive, cfg)
    m = [_np(lifecycle.storm_mask(wind.speed, s.alive, k, cfg)) for k in (kw[0], kw[1])]
    assert not np.array_equal(m[0], m[1]) and 0 < m[1].sum() < _np(s.alive).sum()
    s1, out = make_step(cfg, compile=False)(s)
    mx.eval(s1, out)
    assert _np(out.counters)[storm_counter_index(nd)] == m[1].sum()
    assert (_np(s1.alive) == (_np(s.alive) & ~m[1])).all()  # exactly the drawn agents are gone (no births)


# --- the wind breaks bonds (spec §6.3, §9.5) --------------------------------------------------------------


def _step_wind(cfg: Config, edge: int, speed: float) -> mx.array:
    """A wind of ``speed`` in both layers for the weather columns ``i <= edge``, zero beyond."""
    ws = W.unpack(W.rest_array(cfg))
    n = cfg.weather.cells
    u = mx.where(mx.arange(n)[None, :] <= edge, speed, 0.0) * mx.ones((n, n))
    return W.pack(ws._replace(u1=u, u2=u))


def _bonded_pair(cfg: Config, wx: mx.array, xa: float, xb: float):
    pos = lambda x: [x, 100.0, 40.0]  # noqa: E731
    s = make_scene(cfg, [{"pos": pos(xa)}, {"pos": pos(xb)}], bonds=[(0, 1)], wx=wx)
    return s


def _air_speeds(cfg: Config, s) -> np.ndarray:
    wind = W.agent_wind(W.unpack(s.wx), s.pos, s.buoy_cap, s.alive, cfg)
    return _np(wind.speed)[:2]


def test_a_strong_wind_at_either_end_breaks_the_bond_and_it_is_counted() -> None:
    cfg = storm3d_config("weather.storm_u0=0.02", "weather.storm_break_u=0.07")
    wx = _step_wind(cfg, edge=10, speed=0.08)
    fn = make_step(cfg, compile=False)
    for xa, xb in ((20.0, 23.0), (23.0, 20.0)):  # the windy end is row 0, then row 1
        s = _bonded_pair(cfg, wx, xa, xb)
        sp = _air_speeds(cfg, s)
        assert (
            sp.max() > cfg.weather.storm_break_u > sp.min() > 0.02
        )  # one end above, one below the threshold
        s2, out = fn(s)
        mx.eval(s2, out)
        assert (_np(s2.partner)[:2] == -1).all()
        assert _np(out.counters)[CounterIdx.BONDS_BROKEN] == 1
        assert _np(out.counters)[CounterIdx.BONDS_TOTAL] == 0
    # nothing else breaks it: the same pair with the threshold off, or above both ends, keeps its bond
    for over in ("weather.storm_break_u=0", "weather.storm_break_u=0.09"):
        c2 = apply_overrides(cfg, [over])
        s2, out = make_step(c2, compile=False)(_bonded_pair(c2, wx, 20.0, 23.0))
        mx.eval(s2, out)
        assert _np(s2.partner)[0, 0] == 1 and _np(out.counters)[CounterIdx.BONDS_BROKEN] == 0
    assert _np(out.counters)[CounterIdx.BONDS_TOTAL] == 1


def test_a_windy_row_breaks_the_bonds_in_every_slot() -> None:
    """Row 0 (in the wind) holds bonds in its slots 0 and 1, to rows 1 and 2 which are in calmer air: both
    break, and both are counted (the flag is per row, not per slot)."""
    cfg = storm3d_config("weather.storm_u0=0.02", "weather.storm_break_u=0.07")
    wx = _step_wind(cfg, edge=10, speed=0.08)
    pts = [[20.0, 100.0, 40.0], [23.0, 100.0, 40.0], [23.0, 99.0, 41.0]]
    s = make_scene(
        cfg, [{"pos": q} for q in pts], bonds=[(0, 0, 1, 0), (0, 1, 2, 0)], wx=wx
    )  # row 0: slot 0 -> row 1, slot 1 -> row 2
    sp = _np(W.agent_wind(W.unpack(s.wx), s.pos, s.buoy_cap, s.alive, cfg).speed)[:3]
    assert sp[0] > cfg.weather.storm_break_u > sp[1:].max() > 0.02
    assert (_np(s.partner)[0, :2] == [1, 2]).all() and _np(s.partner)[2, 0] == 0
    s2, out = make_step(cfg, compile=False)(s)
    mx.eval(s2, out)
    assert (_np(s2.partner)[:3] == -1).all()
    assert _np(out.counters)[CounterIdx.BONDS_BROKEN] == 2 and _np(out.counters)[CounterIdx.BONDS_TOTAL] == 0


def test_the_break_flag_is_the_air_speed_not_the_agents() -> None:
    """``breaks`` is the same array the hazard reads: ``speed > storm_break_u``, from the clipped air."""
    cfg = storm3d_config("weather.storm_u0=0.02", "weather.storm_break_u=0.07")
    s = _bonded_pair(cfg, _step_wind(cfg, edge=10, speed=0.08), 20.0, 23.0)
    wind = W.agent_wind(W.unpack(s.wx), s.pos, s.buoy_cap, s.alive, cfg)
    assert np.array_equal(_np(wind.breaks), _np(wind.speed) > np.float32(0.07))


# --- rain_regen (spec §5.2, §9.6) -------------------------------------------------------------------------


def _rain(cfg: Config, cells: dict[tuple[int, int], tuple[float, float, float]]) -> W.RainIn:
    """A ``RainIn`` that is zero except at weather cells ``(j, i)``: ``(P, zi, dc)``."""
    n = cfg.weather.cells
    p, zi, dc = (
        np.zeros((n, n), np.float32),
        np.full((n, n), 30.0, np.float32),
        np.full((n, n), 10.0, np.float32),
    )
    for (j, i), (a, b, c) in cells.items():
        p[j, i], zi[j, i], dc[j, i] = a, b, c
    return W.RainIn(mx.array(p), mx.array(zi), mx.array(dc))


def _band_split(zi: float, dc: float, fz: int, fc: float) -> np.ndarray:
    k = np.arange(fz)
    f = np.clip((np.minimum(zi + dc, fc * (k + 1)) - np.maximum(zi, fc * k)) / fc, 0, 1)
    return f / max(f.sum(), 1e-6)


def test_rain_regen_splits_a_column_over_its_cloud_band() -> None:
    cfg = storm3d_config("weather.rain_gain=1000000")
    d = dims(cfg)
    fc = cfg.world.field_cell
    p0, zi, dc = 0.0107, 28.3, 9.1
    r = _rain(cfg, {(10, 20): (p0, zi, dc)})
    out = _np(W.rain_regen(r, cfg))
    assert out.shape == (d.fz, d.fg, d.fg) and out.dtype == np.int32 and (out >= 0).all()
    q = 1e6 * p0 / 4.0  # a 2x2 block mean of one raining cell
    col = out[:, 5, 10]  # block (j//2, i//2)
    want = q * _band_split(zi, dc, d.fz, fc)
    assert (np.abs(col - want) <= 1.0001).all(), (col, want)  # the floors, within one quantum per layer
    assert col.sum() <= q + 1e-3 and col.sum() >= q - np.count_nonzero(want) - 1  # the floors bias it low
    assert np.count_nonzero(col) == np.count_nonzero(want) >= 3 and col[: int(zi // fc)].sum() == 0
    out[:, 5, 10] = 0
    assert not out.any()  # no other column rains


def test_rain_regen_block_means_are_rain_weighted() -> None:
    """A 2x2 block with one raining cell takes that cell's level and depth, not a quarter of its depth; the
    other cells' (dry) level and depth do not move the band."""
    cfg = storm3d_config("weather.rain_gain=1000000")
    d = dims(cfg)
    alone = _rain(cfg, {(10, 20): (0.01, 20.1, 24.0)})
    # same raining cell, three dry neighbours with a very different cloud
    n = cfg.weather.cells
    zi = np.full((n, n), 5.0, np.float32)
    dc = np.full((n, n), 60.0, np.float32)
    zi[10, 20], dc[10, 20] = 20.1, 24.0
    mixed = W.RainIn(alone.P, mx.array(zi), mx.array(dc))
    a, b = _np(W.rain_regen(alone, cfg)), _np(W.rain_regen(mixed, cfg))
    assert np.abs(a.astype(np.int64) - b).max() <= 1  # P-weighted: float rounding of x·P/P at most
    q = 1e6 * 0.01 / 4.0
    want = q * _band_split(20.1, 24.0, d.fz, cfg.world.field_cell)
    assert (np.abs(b[:, 5, 10] - want) <= 1.0001).all()
    # an unweighted mean of dc over the block (6 + 6 + 6 + 24)/4... would be 24 only for the raining cell:
    wrong = q * _band_split(0.25 * (20.1 + 3 * 5.0), 0.25 * (24.0 + 3 * 60.0), d.fz, cfg.world.field_cell)
    assert np.abs(b[:, 5, 10] - wrong).max() > 5


def test_rain_regen_floor_share_2d_zero_and_bounds() -> None:
    cfg = storm3d_config("weather.rain_gain=1000000", "weather.rain_floor=0.5")
    d = dims(cfg)
    r = _rain(cfg, {(10, 20): (0.0107, 28.3, 9.1), (100, 100): (0.02, 73.0, 6.9)})
    with_floor = _np(W.rain_regen(r, cfg)).astype(np.int64)
    base_cfg = replace(cfg, weather=replace(cfg.weather, rain_floor=0.0))
    no_floor = _np(W.rain_regen(r, base_cfg)).astype(np.int64)
    extra = with_floor - no_floor
    q1, q2 = 1e6 * 0.0107 / 4, 1e6 * 0.02 / 4
    assert extra[1:].sum() == 0 and abs(int(extra[0, 5, 10]) - np.floor(0.5 * q1)) <= 1
    assert abs(int(extra[0, 50, 50]) - np.floor(0.5 * q2)) <= 1
    assert no_floor[0].sum() == 0  # the cloud band starts at the condensation level: the floor layer is dry
    assert with_floor.shape == (d.fz, d.fg, d.fg)
    # a cloud reaching the top layer stays inside [0, FZ)
    top = _np(W.rain_regen(_rain(cfg, {(3, 3): (0.01, 70.0, 9.9)}), base_cfg))
    assert top[-1, 1, 1] > 0 and top.shape[0] == d.fz
    assert not _np(W.rain_regen(_rain(cfg, {}), cfg)).any()  # no rain -> zeros


def test_rain_regen_2d_is_the_floor_of_the_block_mean() -> None:
    cfg = storm2d_config("weather.rain_gain=1000000")
    d = dims(cfg)
    r = _rain(cfg, {(10, 20): (0.01, 28.0, 9.0), (11, 21): (0.02, 28.0, 9.0)})
    out = _np(W.rain_regen(r, cfg))
    assert out.shape == (d.fg, d.fg) and out.dtype == np.int32
    assert (
        out[5, 10] == int(np.floor(np.float32(1e6) * np.float32(0.03) / np.float32(4)))
        and out.sum() == out[5, 10]
    )


def test_rain_in_is_the_stored_rain_with_its_condensation_level_and_cloud_depth(kind: str) -> None:
    """``zi = H·h1/(h1 + h2)`` with ``H`` the world height (3D) or ``H1 + H2`` (2D), computed here from the
    layer thicknesses, ``P`` and ``Dc`` straight from the state."""
    cfg = _cfg(kind, "weather.rain_gain=1000000")
    rng = np.random.default_rng(4)
    n = cfg.weather.cells
    ws = W.unpack(fixed_wx(cfg))
    h1 = ws.h1 * mx.array(rng.uniform(0.8, 1.2, (n, n)).astype(np.float32))
    h2 = ws.h2 * mx.array(rng.uniform(0.8, 1.2, (n, n)).astype(np.float32))
    P = mx.array(rng.uniform(0, 0.01, (n, n)).astype(np.float32))
    Dc = mx.array(rng.uniform(0, 30, (n, n)).astype(np.float32))
    r = W.rain_in(ws._replace(h1=h1, h2=h2, P=P, Dc=Dc), cfg)
    height = cfg.world.height if kind == "3d" else cfg.weather.H1 + cfg.weather.H2
    a, b = _np(h1).astype(np.float64), _np(h2).astype(np.float64)
    np.testing.assert_allclose(_np(r.zi), height * a / (a + b), rtol=1e-5)
    assert np.array_equal(_np(r.P), _np(P)) and np.array_equal(_np(r.dc), _np(Dc))


def test_rain_regen_makes_no_nan_before_the_integer_cast(kind: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """No rain in a block (0/0 in the weighted means) and a cloud outside every layer must not reach
    ``floor_i32`` as a NaN: casting one is undefined (0 on ARM CPU, other values elsewhere)."""
    floor = W.ops.floor_i32
    seen: list[bool] = []

    def checked(x):
        seen.append(bool(np.isfinite(np.array(x)).all()))
        return floor(x)

    monkeypatch.setattr(W.ops, "floor_i32", checked)
    extra = ("weather.rain_floor=0.5",) if kind == "3d" else ()
    cfg = _cfg(kind, "weather.rain_gain=1000000", *extra)
    for cells in ({}, {(10, 20): (0.01, 28.0, 9.0)}, {(3, 3): (0.01, 79.9, 0.0)}):
        W.rain_regen(_rain(cfg, cells), cfg)
    assert seen and all(seen)


def test_rain_for_is_none_without_gain_and_regen_rate_composes() -> None:
    cfg = storm3d_config("weather.rain_gain=1000000")
    ws = W.unpack(fixed_wx(cfg))._replace(P=_rain(cfg, {(10, 20): (0.0107, 28.3, 9.1)}).P)
    off_gain = replace(cfg, weather=replace(cfg.weather, rain_gain=0.0))
    assert W.rain_for(ws, off_gain) is None and W.rain_for(ws, cfg) is not None
    s = init_state(cfg)
    rain = W.rain_in(ws, cfg)
    base = fields.regen_rate(s.patch_pos, cfg, s.step, None)
    assert np.array_equal(_np(fields.regen_rate(s.patch_pos, cfg, s.step, None, None)), _np(base))
    assert np.array_equal(
        _np(fields.regen_rate(s.patch_pos, cfg, s.step, None, rain)), _np(base) + _np(W.rain_regen(rain, cfg))
    )
    # weather off: the old value, floor(regen_max * fertility)
    off = replace(cfg, weather=WeatherConfig())
    want = np.floor(np.float32(off.fields.regen_max) * _np(fields.fertility(s.patch_pos, off, s.step, None)))
    assert np.array_equal(_np(fields.regen_rate(s.patch_pos, off, s.step, None)), want.astype(np.int32))
    assert rain_of(init_state(off), off) is None


def test_the_directors_regeneration_lever_does_not_scale_rain() -> None:
    cfg = storm3d_config("director.enabled=true", "weather.rain_gain=1000000")
    s = init_state(cfg)
    rain = _rain(cfg, {(10, 20): (0.0107, 28.3, 9.1), (30, 40): (0.005, 40.0, 12.0)})
    env1 = s.env
    env2 = env1._replace(regen_mult=mx.array(2.0, dtype=mx.float32))
    delta = [
        _np(fields.regen_rate(s.patch_pos, cfg, s.step, e, rain))
        - _np(fields.regen_rate(s.patch_pos, cfg, s.step, e))
        for e in (env1, env2)
    ]
    assert np.array_equal(delta[0], delta[1]) and delta[0].sum() > 0
    assert (
        _np(fields.regen_rate(s.patch_pos, cfg, s.step, env2)).sum()
        > _np(fields.regen_rate(s.patch_pos, cfg, s.step, env1)).sum()
    )


# --- the ledger books the rain (spec §5.1, §5.4, §9.6) ----------------------------------------------------

BIG_CAP = "fields.resource_cap=16000000"


def _expected_inject(prev, cfg: Config) -> int:
    """``Σ regen_rate(..., rain)`` of the step taken from ``prev``, rebuilt on the host side of the engine:
    the weather advanced from ``prev.wx`` with the engine's key chain, then the field's own functions."""
    _, ks = step_keys(prev.key)
    kw = mx.random.split(ks[Stream.WEATHER], 2)
    ws2, _ = W.advance(W.unpack(prev.wx), prev.step, kw[0], cfg)
    env = prev.env if cfg.director.enabled else None
    rate = fields.regen_rate(prev.patch_pos, cfg, prev.step, env, W.rain_for(ws2, cfg))
    return int(_np(rate).astype(np.int64).sum())


def test_inject_regen_books_the_rain_exactly(kind: str) -> None:
    """With a cap far above the cell contents, ``INJECT_REGEN`` is ``Σ regen_rate`` with the rain: the ledger
    needs no new term. A CPU equality (on a GPU the engine's fused float product may flip a floor by one
    quantum: the rebuild is within one quantum per raining cell then)."""
    cfg = _cfg(kind, "weather.rain_gain=10000000", BIG_CAP)
    fn = make_step(cfg, compile=True)
    d = dims(cfg)
    z = [40.0] if d.d3 else []
    s = make_scene(
        cfg, [{"pos": [20.0 + 30 * k, 50.0, *z]} for k in range(8)], resource=1000
    )  # far below the cap
    rained = 0
    for _ in range(5):
        prev = s
        s, out = fn(s)
        mx.eval(s, out)
        got = int(_np(out.ledger)[LedgerIdx.INJECT_REGEN])
        want = _expected_inject(prev, cfg)
        if ON_GPU:
            assert abs(got - want) <= int((_np(s.wx[W.WeatherState._fields.index("P")]) > 0).sum())
        else:
            assert got == want
        no_rain = replace(cfg, weather=replace(cfg.weather, rain_gain=0.0))
        base = int(_np(fields.regen_rate(prev.patch_pos, no_rain, prev.step, None)).astype(np.int64).sum())
        rained += got - base
    assert rained > 0  # the rain did feed the field
    assert int(_np(s.diag)) & Diag.LEDGER == 0


@pytest.mark.parametrize("steps", [40, 300])
def test_the_ledger_closes_with_rain_storms_catastrophe_and_immigration(steps: int) -> None:
    cfg = storm3d_config(
        "director.enabled=true",
        "weather.rain_gain=10000000",
        "weather.storm_p=0.02",
        "weather.storm_u0=0.0",
        "weather.storm_span=0.02",
        "population.immigration_floor=40",
        BIG_CAP,
    )
    c = cfg.world.size / 2
    s0 = init_state(cfg)._replace(
        env_cat_center=mx.array([c, c], dtype=mx.float32),
        env_cat_radius=mx.array(cfg.world.size / 2, dtype=mx.float32),
        env_cat_kill_p=mx.array(0.02, dtype=mx.float32),
    )
    fixed = fixed_wx(cfg, u1=0.03, v1=0.02, u2=-0.03, v2=0.01)

    def pre(i, s):  # a prescribed wind for the first half, the real weather after it
        return s._replace(wx=fixed) if i < steps // 2 else s

    _, hist = run_checked(cfg, steps, s0, pre=pre)  # full_check(prev, out) every step, no error diag bit
    storm = sum(int(_np(o.counters)[storm_counter_index(3)]) for _, _, o in hist)
    cat = sum(int(_np(o.counters)[CounterIdx.DEATHS_CATASTROPHE]) for _, _, o in hist)
    assert storm > 0 and cat > 0  # both death causes happened
    if steps >= 300:  # the population has been driven under the immigration floor by then
        assert sum(int(_np(o.counters)[CounterIdx.IMMIGRANTS]) for _, _, o in hist) > 0


# --- K_est rebuilds the rain (spec §5.3, §9.7) ------------------------------------------------------------


def _after_steps(cfg: Config, steps: int = 3):
    fn = make_step(cfg, compile=True)
    s = init_state(cfg)
    for _ in range(steps):
        s, out = fn(s)
        mx.eval(s, out)
    return s


def test_carrying_capacity_includes_the_stored_rain(kind: str) -> None:
    from alife.metrics import _c_ref, layer_occupancy_weights

    cfg = _cfg(kind, "weather.rain_gain=10000000", BIG_CAP)
    s = _after_steps(cfg)
    no_rain = replace(cfg, weather=replace(cfg.weather, rain_gain=0.0))
    rain = _np(W.rain_regen(W.rain_in(W.unpack(s.wx), cfg), cfg)).astype(np.int64)
    assert rain.sum() > 0 and rain_of(s, no_rain) is None and rain_of(s, cfg) is not None
    got = carrying_capacity(s, cfg) - carrying_capacity(s, no_rain)
    if kind == "3d":
        want = float((rain.reshape(rain.shape[0], -1).sum(axis=1) * layer_occupancy_weights(s, cfg)).sum())
    else:
        want = float(rain.sum())
    want /= _c_ref(cfg)
    tol = 1e-9 if not ON_GPU else 1e-6 * max(want, 1.0)
    assert got == pytest.approx(want, abs=tol, rel=1e-9) and got > 0
    # the same value through the weather-off formula: a config without weather ignores the stored wx
    off = replace(cfg, weather=WeatherConfig())
    assert rain_of(s, off) is None and carrying_capacity(s, off) == carrying_capacity(s, no_rain)


def test_floor_capacity_has_the_rain_only_with_rain_floor() -> None:
    from alife.metrics import _c_ref

    cfg = storm3d_config("weather.rain_gain=10000000", "weather.rain_floor=0.5", BIG_CAP)
    s = _after_steps(cfg)
    no_rain = replace(cfg, weather=replace(cfg.weather, rain_gain=0.0))
    rain = _np(W.rain_regen(W.rain_in(W.unpack(s.wx), cfg), cfg)).astype(np.int64)
    assert rain[0].sum() > 0 and rain[1:].sum() > 0
    got = floor_carrying_capacity(s, cfg) - floor_carrying_capacity(s, no_rain)
    assert got == pytest.approx(float(rain[0].sum()) / _c_ref(cfg), rel=1e-9) and got > 0
    zero = replace(cfg, weather=replace(cfg.weather, rain_floor=0.0))
    assert floor_carrying_capacity(s, zero) == floor_carrying_capacity(
        s, no_rain
    )  # the band is above the floor


# --- the weather stays agent-independent with the hazard on (spec §3.5, §9.9, §9.12) ---------------------


def _standalone_wx(c: Config, steps: int) -> mx.array:
    key = init_keys(c.run.seed)[InitStream.STATE_KEY]
    ws = W.unpack(W.rest_array(c))
    for t in range(steps):
        key, ks = step_keys(key)
        kw = mx.random.split(ks[Stream.WEATHER], 2)
        ws, _ = W.advance(ws, mx.array(t, dtype=mx.int32), kw[0], c)
    return W.pack(ws)


def _engine_run(c: Config, steps: int):
    fn = make_step(c, compile=True)
    s = init_state(c)
    storm = 0
    for _ in range(steps):
        s, out = fn(s)
        mx.eval(s, out)
        storm += int(_np(out.counters)[storm_counter_index(3)])
    return s, storm


def test_the_weather_is_agent_independent_with_agents_dying_of_storms() -> None:
    """30 engine steps with the hazard at its maximum (every agent in any wind dies) and with another
    ``n_init``: ``wx`` is bitwise the standalone model's; and toggling ``storm_p`` changes no stream's key and
    not the weather (spec §9.9)."""
    on = storm3d_config("run.seed=2", *STORM_ALL, "weather.rain_gain=10000000")
    ref = _np(_standalone_wx(on, 30))
    s_on, storm_on = _engine_run(on, 30)
    assert storm_on > 0  # agents did die of the storm (the winds are small but positive after step 0)
    assert np.array_equal(_np(s_on.wx), ref)
    few = storm3d_config("run.seed=2", *STORM_ALL, "weather.rain_gain=10000000", n_init=12)
    s_few, _ = _engine_run(few, 30)
    assert np.array_equal(_np(s_few.wx), ref)
    off = apply_overrides(on, ["weather.storm_p=0"])
    s_off, storm_off = _engine_run(off, 30)
    assert storm_off == 0 and np.array_equal(_np(s_off.wx), ref)
    # the state key chain does not depend on the lever (the keys are drawn from state.key alone)
    assert np.array_equal(_np(s_off.key), _np(s_on.key))


# --- the runner and the viewer see the weather-on width (spec §6.2) -------------------------------------


def test_a_runner_run_writes_deaths_storm_and_resumes(tmp_path) -> None:
    """A weather-on run (2D, a prescribed-free calm start): every metrics row has ``deaths_storm``; the
    status line, the checkpoint and ``resim.compare_row`` take the 28-wide counters."""
    from alife import runner
    from alife.analysis.resim import compare_row
    from alife.runio import RunDir

    cfg = storm2d_config("run.metrics_every=5", "run.checkpoint_every=10", "weather.rain_gain=1000000")
    rd = RunDir.create(tmp_path / "run", cfg)
    res = runner.run(rd, cfg, init_state(cfg), 10, quiet=True)
    assert (res.status, res.exit_code) == ("done", runner.ExitCode.OK)
    rows = rd.metrics_rows()
    assert [r["step"] for r in rows] == [5, 10] and all("deaths_storm" in r for r in rows)
    assert {r["ledger_error"] for r in rows} == {0}
    c = np.ones((3, n_counters(2, True)), np.int32)
    lg = np.zeros((3, 16), np.int64)
    stored = interval_row(7, c, lg, {}, {}, nd=2, weather=True)
    args = ({**stored, "steps": 3}, [mx.array(r) for r in c], [mx.array(r) for r in lg], 0)
    assert compare_row(*args, weather=True) == [] and "deaths_storm" in stored
    with pytest.raises(ValueError, match="columns"):
        compare_row(*args)  # the weather-off width is refused
