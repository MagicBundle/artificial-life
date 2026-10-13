"""§27.8 the 3D step: every compiled step passes ``full_check`` with the closures over slot 15
(``HEAT_BUOYANCY``), the step books the buoyancy payments there, the ``Counter3D`` counts match the state
transition, and ``INT_HEADROOM`` sees the occupancy of an x/y column (§27.7)."""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import numpy as np
import pytest

from alife import energy, invariants, layout, metrics
from alife.config import Config, apply_overrides, dims
from alife.state import (
    N_COUNTERS,
    Counter3D,
    CounterIdx,
    Diag,
    LedgerIdx,
    State3,
    init_state,
    n_counters,
    to_numpy,
)
from alife.step import StepOut, make_step
from helpers import ledger_net, make_scene, run_checked, set_output_bias
from test_step_invariants import _check_counters

STEPS = 60
H15 = LedgerIdx.HEAT_BUOYANCY
f32 = np.float32


def _np(a: mx.array) -> np.ndarray:
    return to_numpy(a)


def _e15() -> mx.array:
    return mx.array(np.eye(len(LedgerIdx), dtype=np.int64)[H15])


def _counter3d_ref(prev: State3, s: State3, cfg: Config) -> list[int]:
    """``Counter3D`` from the transition alone: a survivor keeps its row and uid; one stopped by the floor
    (surface) ends at ``z_lo`` (``z_hi``) with its stored ``v_z`` zeroed (§27.3); grounding is the
    start-of-step ``z ≤ z_ground`` of the rows that are alive at the start and survive the step (a row that
    dies is not counted, so ``GROUNDED ≤ N_ALIVE``)."""
    d = dims(cfg)
    a0, a1 = _np(prev.alive), _np(s.alive)
    sv = a0 & a1 & (_np(prev.uid) == _np(s.uid))
    z, vz = _np(s.pos)[:, 2], _np(s.vel)[:, 2]
    floor = sv & (z == f32(d.z_lo)) & (vz == 0)
    surface = sv & (z == f32(d.z_hi)) & (vz == 0)
    grounded = sv & (_np(prev.pos)[:, 2] <= f32(d.z_ground))
    return [int(floor.sum()), int(surface.sum()), int(grounded.sum())]


def _check_step_3d(prev: State3, s: State3, out: StepOut, cfg: Config) -> np.ndarray:
    """The 2D counter rules on the first 27, ``Counter3D`` against the transition, closures with slot 15."""
    c = _np(out.counters)
    assert c.shape == (n_counters(3),) == (N_COUNTERS + len(Counter3D),) and out.counters.dtype == mx.int32
    _check_counters(prev, s, out._replace(counters=out.counters[:N_COUNTERS]), cfg)
    assert c[N_COUNTERS:].tolist() == _counter3d_ref(prev, s, cfg)
    total0, total1 = int(_np(prev.total)), int(_np(s.total))
    assert total1 - total0 == ledger_net(out.ledger, cfg)  # per step, HEAT_BUOYANCY a sink
    assert total1 == int(_np(s.total0)) + ledger_net(s.ledger_cum, cfg)  # global
    return c


# --- the whole step ---------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def trajectory(micro3d_cfg: Config) -> tuple[State3, list[tuple[State3, State3, StepOut]]]:
    """``STEPS`` compiled, ``full_check``ed steps of micro3d, half the founders starting on the floor."""
    cfg = micro3d_cfg
    s = init_state(cfg)
    pos = _np(s.pos).copy()
    pos[: cfg.population.n_init // 2, 2] = dims(cfg).z_lo
    s = s._replace(pos=mx.array(pos))
    return run_checked(cfg, STEPS, s)


def test_every_3d_step_is_clean(micro3d_cfg: Config, trajectory) -> None:
    cfg = micro3d_cfg
    s, hist = trajectory
    totals = np.zeros(n_counters(3), np.int64)
    for prev, s1, out in hist:
        totals += _check_step_3d(prev, s1, out, cfg)
        assert _np(out.ledger)[H15] > 0  # every alive agent pays the upkeep of its cap
    assert isinstance(s, State3) and int(_np(s.step)) == STEPS
    assert totals[Counter3D.Z_FLOOR_CLAMPS] > 0 and totals[Counter3D.GROUNDED] > 0
    assert totals[CounterIdx.BIRTHS] > 0 and _np(s.alive).sum() > 0
    led = np.sum([_np(out.ledger) for _, _, out in hist], axis=0)
    assert np.array_equal(_np(s.ledger_cum), led)


def test_the_step_books_the_buoyancy_payments(micro3d_cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    """Slot 15 is ``Σ paid_buoy`` of S8, and every alive (solvent) agent pays the f32 due of §27.7."""
    cfg = micro3d_cfg
    seen: dict[str, Any] = {}
    real = energy.pay_costs3

    def spy(*args: Any) -> energy.Costs3:
        seen["args"], seen["costs"] = args, real(*args)
        return seen["costs"]

    monkeypatch.setattr(energy, "pay_costs3", spy)
    s = set_output_bias(init_state(cfg), cfg, {dims(cfg).lay.OUT_BUOYANCY: 1.0})
    s1, out = make_step(cfg, compile=False)(s)
    mx.eval(s1, out)
    _, _, mo, _, _, cap, alive, _ = seen["args"]
    c = seen["costs"]
    paid = _np(c.paid_buoy)
    assert int(_np(out.ledger)[H15]) == int(paid.sum()) > 0
    lift = _np(cap) * _np(mo.buoy)
    ec = cfg.energy
    due = np.floor(f32(ec.buoyancy_cost) * (lift * lift)).astype(np.int64)
    due += np.floor(f32(ec.buoyancy_cap_cost) * _np(cap)).astype(np.int64)
    alive = _np(alive)
    assert _np(c.solvent)[alive].all()
    np.testing.assert_array_equal(paid, np.where(alive, due, 0))
    assert (due[alive] > np.floor(f32(ec.buoyancy_cap_cost) * _np(cap)[alive])).any()  # the use term too


def test_heat_buoyancy_under_an_output_bias(micro3d_cfg: Config, trajectory) -> None:
    """Full ``OUT_BUOYANCY`` pays the use term on top of the upkeep every step; the ledger still closes."""
    cfg = micro3d_cfg
    s0 = trajectory[1][0][0]
    _, plain = run_checked(cfg, 10, s0)
    _, biased = run_checked(cfg, 10, set_output_bias(s0, cfg, {dims(cfg).lay.OUT_BUOYANCY: 4.0}))
    n0 = int(_np(s0.alive).sum())
    upkeep = int(np.floor(f32(cfg.energy.buoyancy_cap_cost) * f32(cfg.mutation.buoy_init)))
    h_plain = [int(_np(out.ledger)[H15]) for _, _, out in plain]
    h_biased = [int(_np(out.ledger)[H15]) for _, _, out in biased]
    assert h_biased[0] > n0 * upkeep and h_biased[0] > h_plain[0]
    assert all(h > 0 for h in h_biased) and sum(h_biased) > sum(h_plain)
    for prev, s1, out in biased:
        _check_step_3d(prev, s1, out, cfg)


def test_corrupting_heat_buoyancy_fails_the_closures(micro3d_cfg: Config, trajectory) -> None:
    cfg = micro3d_cfg
    prev, s, out = trajectory[1][-1]
    assert int(_np(out.ledger)[H15]) > 0 and invariants.full_check(s, cfg, prev, out) == []
    z = mx.array(0, dtype=mx.int32)

    def bits(ledger: mx.array) -> int:
        return int(_np(invariants.device_check(s, ledger, prev.total, z, z, z, z, cfg)))

    assert bits(out.ledger) & Diag.LEDGER == 0
    assert bits(out.ledger + _e15()) & Diag.LEDGER and bits(out.ledger - _e15()) & Diag.LEDGER
    bad_out = invariants.full_check(s, cfg, prev, out._replace(ledger=out.ledger + _e15()))
    assert any("out.ledger != ledger_cum" in p for p in bad_out), bad_out
    bad = invariants.full_check(s._replace(ledger_cum=s.ledger_cum + _e15()), cfg, prev)
    assert any("global closure fails" in p for p in bad) and any("per-step closure fails" in p for p in bad)


def test_a_2d_step_keeps_slot_15_at_zero(micro_cfg: Config) -> None:
    cfg = micro_cfg
    s, hist = run_checked(cfg, 10)
    for _, _, out in hist:
        assert out.counters.shape == (N_COUNTERS,) and int(_np(out.ledger)[LedgerIdx.RESERVED]) == 0
    assert int(_np(s.ledger_cum)[LedgerIdx.RESERVED]) == 0


# --- Counter3D ----------------------------------------------------------------------------------------------


def _w2(cfg: Config, **bias: float) -> np.ndarray:
    """A zero brain with output biases (``OUT_*`` names of the 3D layout)."""
    d = dims(cfg)
    w2 = np.zeros((d.w2_rows, d.lay.D_OUT), np.float32)
    for name, v in bias.items():
        w2[d.h, getattr(d.lay, name) if hasattr(d.lay, name) else getattr(layout, name)] = v
    return w2


def test_counter3d_counts_floor_surface_and_grounded(micro3d_cfg: Config) -> None:
    """Swimmers pushing up into the surface, agents pressed onto the floor by gravity, one in mid-water."""
    cfg = micro3d_cfg
    d = dims(cfg)
    up = _w2(cfg, OUT_THRUST=4.0, OUT_PITCH=4.0)
    pm = float(f32(layout.PITCH_MAX))
    agents = [
        *[dict(pos=(6.0 + 6 * k, 10.0, d.z_hi - 0.02), pitch=pm, W2=up, birth_step=-50) for k in range(3)],
        *[dict(pos=(6.0 + 6 * k, 30.0, d.z_lo), birth_step=-50) for k in range(4)],
        dict(pos=(40.0, 40.0, 12.0), birth_step=-50),
    ]
    s = make_scene(cfg, agents, resource=0)
    _, hist = run_checked(cfg, 3, s)
    for prev, s1, out in hist:
        c = _check_step_3d(prev, s1, out, cfg)
        assert c[Counter3D.Z_SURFACE_CLAMPS] == 3 and c[Counter3D.Z_FLOOR_CLAMPS] == 4
        assert c[Counter3D.GROUNDED] == 4
        assert (_np(s1.pos)[:3, 2] == f32(d.z_hi)).all() and (_np(s1.pos)[3:7, 2] == f32(d.z_lo)).all()


def test_grounded_counts_only_the_agents_that_survive(micro3d_cfg: Config) -> None:
    """A mass death: eight grounded agents starve in the step, two grounded agents live. `GROUNDED` is the
    two, not ten, so the interval's `grounded_frac` is a share (a start-of-step count against the end-of-step
    `N_ALIVE` gave 10 / 2)."""
    cfg = micro3d_cfg
    d = dims(cfg)
    dying = [dict(pos=(4.0 + 4 * k, 10.0, d.z_lo), energy=0, birth_step=-50) for k in range(8)]
    living = [dict(pos=(6.0 + 6 * k, 30.0, d.z_lo), birth_step=-50) for k in range(2)]
    s = make_scene(cfg, [*dying, *living], resource=0)
    _, hist = run_checked(cfg, 1, s)
    prev, s1, out = hist[0]
    c = _check_step_3d(prev, s1, out, cfg)
    assert int(_np(prev.alive).sum()) == 10 and int(_np(s1.alive).sum()) == 2
    assert c[CounterIdx.N_ALIVE] == 2 and c[Counter3D.GROUNDED] == 2
    row = metrics.interval_row(1, c[None, :], _np(out.ledger)[None, :], {}, {}, nd=3)
    assert row["grounded_frac"] == 1.0 and 0.0 <= row["grounded_frac"] <= 1.0


# --- the column bound of INT_HEADROOM (§27.7) ---------------------------------------------------------------


def test_a_crowded_column_sets_int_headroom(tiny3d_cfg: Config) -> None:
    """``occ_limit`` agents stacked over one x/y field cell, no 3D cell near the limit: only the column bound
    of §27.7 (a floor cell receives the corpses of the whole column) sees them."""
    cfg = apply_overrides(tiny3d_cfg, ["population.n_max=2048", "population.n_init=1100"])
    d = dims(cfg)
    n0 = cfg.population.n_init
    assert d.occ_limit <= n0
    s = init_state(cfg)
    pos = _np(s.pos).copy()
    rs = np.random.default_rng(7)
    pos[:n0, :2] = 60.0 + rs.uniform(0.0, cfg.world.field_cell, (n0, 2))
    pos[:n0, 2] = rs.uniform(d.z_lo, d.z_hi, n0)
    s = s._replace(pos=mx.array(pos.astype(np.float32)))
    cell = np.floor(pos[:n0] / cfg.world.field_cell).astype(np.int64)
    assert np.unique(cell[:, :2], axis=0).shape[0] == 1  # one column
    assert np.unique(cell, axis=0, return_counts=True)[1].max() < d.occ_limit  # every 3D cell below it
    s1, out = make_step(cfg, compile=False)(s)
    mx.eval(s1, out)
    assert int(_np(s1.diag)) & Diag.INT_HEADROOM
