"""§27.8 3D metrics: ``Counter3D`` and the per-dims counter width, interval rows with ``nd = 3`` (no silent
reshape, ``HEAT_BUOYANCY`` named in 3D and ``RESERVED`` kept in 2D), the snapshot over the volume (chemical
means, the global closure with slot 15), the carrying capacity with the cap's upkeep, and resim's
comparison."""

from __future__ import annotations

import json
import math

import mlx.core as mx
import numpy as np
import pytest

from alife import fields, layout, metrics
from alife.analysis import resim
from alife.config import Config, dims
from alife.state import N_COUNTERS, Counter3D, CounterIdx, LedgerIdx, ledger_names, n_counters, to_numpy
from helpers import as_3d, make_scene

C = CounterIdx


@pytest.fixture(scope="module")
def cfg3(micro_cfg: Config) -> Config:
    """``micro`` in 3D (FZ 6, FG 12) with an active buoyancy cap and its costs."""
    return as_3d(
        micro_cfg,
        physics={"buoyancy_max": 0.05},
        mutation={"buoy_init": 0.25, "buoy_sigma": 0.02},
        energy={"buoyancy_cost": 120, "buoyancy_cap_cost": 30},
    )


def _counters3(t: int, rs: np.random.Generator) -> np.ndarray:
    """(t, 30) counters: random event counts, then the three Counter3D columns."""
    c = rs.integers(0, 50, (t, n_counters(3))).astype(np.int32)
    c[:, C.N_ALIVE] = rs.integers(100, 200, t)
    return c


# --- counters -----------------------------------------------------------------------------------------------


def test_counter3d_follows_counteridx() -> None:
    assert [int(i) for i in Counter3D] == list(range(N_COUNTERS, N_COUNTERS + 3))
    assert [i.name for i in Counter3D] == ["Z_FLOOR_CLAMPS", "Z_SURFACE_CLAMPS", "GROUNDED"]
    assert len(CounterIdx) == N_COUNTERS == n_counters(2) == 27 and n_counters(3) == 30


# --- interval rows ------------------------------------------------------------------------------------------


def test_interval_row_3d_sums_the_shared_counters_and_names_slot_15() -> None:
    """T = 9 is the trap: 9·30 = 10·27, so a reshape by 27 would succeed with scrambled columns."""
    rs = np.random.default_rng(4)
    t = 9
    c = _counters3(t, rs)
    lg = rs.integers(0, 1 << 40, (t, len(LedgerIdx))).astype(np.int64)
    row = metrics.interval_row(90, c, lg, {"E_total": 1}, {}, nd=3)
    assert row["steps"] == t and row["births"] == int(c[:, C.BIRTHS].sum())
    assert row["n_alive"] == pytest.approx(c[:, C.N_ALIVE].mean()) and row["n_alive_min"] == c[:, 0].min()
    assert row["deaths_catastrophe"] == int(c[:, C.DEATHS_CATASTROPHE].sum())
    assert row["max_occupancy"] == int(c[:, C.MAX_OCCUPANCY].max())
    assert list(row["ledger"]) == list(ledger_names(3)) and "RESERVED" not in row["ledger"]
    assert row["ledger"]["HEAT_BUOYANCY"] == int(lg[:, LedgerIdx.HEAT_BUOYANCY].sum())
    assert all(row["ledger"][n] == int(lg[:, i].sum()) for i, n in enumerate(ledger_names(3)))
    json.dumps(row)
    with pytest.raises(ValueError, match="30 columns, a 2D world has 27"):
        metrics.interval_row(90, c, lg, {}, {})  # the default dims refuse 3D counters
    with pytest.raises(ValueError, match="27 columns, a 3D world has 30"):
        metrics.interval_row(90, c[:, :N_COUNTERS], lg, {}, {}, nd=3)


def test_interval_row_2d_keeps_reserved() -> None:
    c = np.zeros((2, N_COUNTERS), np.int32)
    lg = np.zeros((2, len(LedgerIdx)), np.int64)
    row = metrics.interval_row(2, c, lg, {}, {})
    assert list(row["ledger"]) == [i.name for i in LedgerIdx] == list(ledger_names(2))
    assert row["ledger"]["RESERVED"] == 0 and "HEAT_BUOYANCY" not in row["ledger"]
    assert row == metrics.interval_row(2, c, lg, {}, {}, nd=2)


# --- snapshot -----------------------------------------------------------------------------------------------


def test_snapshot_3d_means_over_the_volume_and_closes_with_slot_15(cfg3: Config) -> None:
    d = dims(cfg3)
    rs = np.random.default_rng(8)
    agents = [
        dict(pos=(rs.uniform(0, 48), rs.uniform(0, 48), rs.uniform(d.z_lo, d.z_hi)), energy=30_000 + k)
        for k in range(10)
    ]
    chem = rs.uniform(0, 4, (layout.NCH, *d.field_shape)).astype(np.float32)
    s = make_scene(cfg3, agents, bonds=[(0, 1), (1, 2)], chem=chem)
    spent = 1234  # quanta paid for buoyancy since t = 0: TOTAL fell, slot 15 holds them
    cum = np.zeros(len(LedgerIdx), np.int64)
    cum[LedgerIdx.HEAT_BUOYANCY] = spent
    s = s._replace(ledger_cum=mx.array(cum), total0=s.total + spent)
    m = metrics.snapshot_metrics(s, cfg3, heavy=True)
    assert m["ledger_error"] == 0
    assert m["chem_mean"] == pytest.approx(chem.reshape(layout.NCH, -1).mean(axis=1).tolist(), rel=1e-5)
    assert m["resource_mean"] == m["E_field"] / d.n_cells
    assert m["bodies_largest"] == 3 and m["weight_diversity"] is not None
    json.dumps(m)
    s2 = s._replace(ledger_cum=mx.zeros_like(s.ledger_cum))  # the payment without its sink
    assert metrics.snapshot_metrics(s2, cfg3, heavy=False)["ledger_error"] == spent


# --- carrying capacity --------------------------------------------------------------------------------------


def test_carrying_capacity_3d_includes_the_cap_upkeep(cfg3: Config) -> None:
    s = make_scene(cfg3, [])
    regen = int(to_numpy(fields.regen_rate(s.patch_pos, cfg3)).astype(np.int64).sum())
    assert to_numpy(fields.regen_rate(s.patch_pos, cfg3)).shape == dims(cfg3).field_shape
    upkeep = math.floor(30 * 0.25)  # floor(buoyancy_cap_cost·buoy_init) = 7
    c_ref = cfg3.energy.basal_cost + metrics.founder_brain_cost(cfg3) + layout.C_REF_MARGIN + upkeep
    assert metrics.carrying_capacity(s, cfg3) == pytest.approx(regen / c_ref, rel=1e-12)
    off = as_3d(cfg3, energy={"buoyancy_cap_cost": 0})
    assert metrics.carrying_capacity(s, off) == pytest.approx(regen / (c_ref - upkeep), rel=1e-12)


# --- resim --------------------------------------------------------------------------------------------------


def test_resim_compare_uses_the_3d_names() -> None:
    rs = np.random.default_rng(2)
    c = _counters3(5, rs)
    lg = rs.integers(0, 1000, (5, len(LedgerIdx))).astype(np.int64)
    stored = {**metrics.interval_row(50, c, lg, {}, {}, nd=3), "E_total": 77}
    stored = json.loads(json.dumps(stored))  # as read back from metrics.jsonl
    counters, ledgers = [mx.array(r) for r in c], [mx.array(r) for r in lg]
    assert resim._compare(stored, counters, ledgers, 77, nd=3) == []
    stored["ledger"]["HEAT_BUOYANCY"] += 1
    assert resim._compare(stored, counters, ledgers, 77, nd=3) == ["ledger"]
    with pytest.raises(ValueError, match="columns"):
        resim._compare(stored, counters, ledgers, 77)  # a 3D run compared as 2D is refused, not scrambled
