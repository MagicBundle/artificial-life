"""Slow 3D ecology check (§22.5, DESIGN §27.13): ``DEV3D``, the ``64k3d`` preset shrunk, for 2 500 steps.

The 3D counterpart of ``test_ecology.py`` and the harness for the ``64k3d`` defaults, to be passed before any
long 3D run. ``DEV3D`` keeps what sets the ecology of ``64k3d`` (gravity, the buoyancy actuator and its costs,
the vertical food profile, the regeneration, the hash capacity ``C = 48``, ``K = 8``) and shrinks what only
sets its size: a 160 x 160 x 40 column (a quarter of the floor, an eighth of the volume), four patches of
sigma 17 (the same share of the floor as eight patches of sigma 24 in 320), 16 384 rows and 4 096 founders.
Weights are float32 as in ``tiny3d_cfg`` (CPU fp16 matmul is about 3x slower, F19); the dynamics do not depend
on the storage dtype.

``n_init`` is about half of the occupancy-weighted ``K_est`` of a mostly benthic population (0.47 of 8 700,
the arithmetic of §27.13), as ``64k3d``'s is of its 33 600: the founders' first generation multiplies the
population by 1.4 to 1.7, and the cloud run seeded at 2.5x that ``K_est`` (5x the floor-only one) boomed and
crashed. Seeding at 1x was not tried: the value is a hedge that the owner signs off.
``C = 48`` is confirmed (owner decision 2026-10-16, DESIGN §27.13). It was 24 until this test failed its
overflow gate at that value on the cloud worker (cloud-7): one floor pile first passed 24 at step 2151, was
over it continuously from step 2207 and was still growing (peak occupancy 31, over capacity on 11.96% of the
steps). At 48 the test passes on that worker (cloud-8); a 20 000-step run of the same configuration never
overflowed (peak occupancy 43, at step 10 705) and ``64k3d`` itself ran 30 000 steps on the Mac with a peak
of 37. The gate is unchanged (at most 1% of the steps); a pile that keeps growing past 48 would fail it
again.
The assertions are the 2D ones that make sense in 3D (never extinct, not pinned at ``N_CAP``, births, bonds
and landed attacks, a clean state), plus the 3D terms: floor contacts (also late in the run),
``HEAT_BUOYANCY``, some agents resting on the floor and some swimming in the final state, and hash-cell
overflow on at most 1% of the steps.

Run by the nightly job (``-m slow``) and on the cloud worker; not by the fast suite, and not on the Mac while
its GPU is busy.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import numpy as np
import pytest

from alife import invariants, metrics, physics
from alife.config import Config, dims, load_config
from alife.state import (
    ERROR_MASK,
    Counter3D,
    CounterIdx,
    LedgerIdx,
    init_state,
    n_counters,
    to_numpy,
)
from alife.step import make_step

STEPS = 2500
C = CounterIdx
CELL_CAPACITY = 48  # 64k3d's hash capacity C, confirmed (DESIGN §27.13); DEV3D sets it too (checked equal)
DEV3D = (
    "population.n_max=16384",
    "population.max_births_per_step=256",
    "population.n_init=4096",
    "world.size=160",
    "world.height=40",
    "fields.n_patches=4",
    "fields.patch_sigma=17",
    "brain.weight_dtype=float32",
    f"sensing.cell_capacity={CELL_CAPACITY}",
)
# The share of the steps in which any hash cell holds more than C agents may be at most this. None was seen
# at C = 16 on tiny3d and the cloud DEV3D (peak occupancy 13), but those runs had a regen_max of 320; at the
# 1 700 of this preset the food alone can feed about 20 agents in the hottest floor hash cell (27 in DEV3D's
# quarter-size floor), and the cloud run at C = 24 had a floor pile of 31, over capacity on 11.96% of the
# steps (DESIGN §27.13), so this is the gate on C. A floor pile that overflows does so for many steps in a
# row (hundreds in the crowd tests), which 1% of 2 500 steps does not allow.
OVERFLOW_STEPS_MAX = 0.01
LATE = 500  # the last steps of the run: the steady state, where the floor is what the agents use


def dev3d_cfg() -> Config:
    """``64k3d`` shrunk to 16 384 rows (see the module docstring)."""
    return load_config(preset="64k3d", overrides=list(DEV3D))


def run_dev3d(cfg: Config, steps: int) -> dict[str, Any]:
    """``steps`` compiled steps from the initial state: the per-step counters and ledgers, the final state and
    the ``K_est`` of the initial one (the founders uniform in the volume)."""
    s = init_state(cfg)
    k0 = metrics.carrying_capacity(s, cfg)
    fn = make_step(cfg)
    counters = np.zeros((steps, n_counters(3)), np.int64)
    ledgers = np.zeros((steps, len(LedgerIdx)), np.int64)
    for i in range(steps):
        s, out = fn(s)
        mx.eval(s, out)
        counters[i], ledgers[i] = to_numpy(out.counters), to_numpy(out.ledger)
        assert counters[i, C.N_ALIVE] > 0, f"extinct at step {i}"
    return {"state": s, "counters": counters, "ledgers": ledgers, "k0": k0}


def check_dev3d(cfg: Config, run: dict[str, Any]) -> None:
    """The ecology assertions on a ``run_dev3d`` result."""
    d = dims(cfg)
    s, cnt, led = run["state"], run["counters"], run["ledgers"]
    n = cnt[:, C.N_ALIVE]
    totals = cnt.sum(axis=0)
    assert d.fz == 10 and d.q == 27 * CELL_CAPACITY and cfg.sensing.cell_capacity == CELL_CAPACITY
    assert d.k == 8  # DEV3D keeps 64k3d's K ...
    assert load_config(preset="64k3d").sensing.cell_capacity == CELL_CAPACITY  # ... and its C
    # --- the 2D checks of test_ecology.py ---
    assert (n == d.n_cap).mean() < 0.5, "population pinned at N_CAP"
    assert 0 < n[-1] < d.n_cap
    assert totals[C.BIRTHS] > 0
    assert totals[C.BONDS_FORMED] + totals[C.CHILD_BONDS] > 0 and cnt[-1, C.BONDS_TOTAL] > 0
    assert totals[C.ATTACK_HITS] > 0
    assert int(to_numpy(s.diag)) & ERROR_MASK == 0
    assert invariants.full_check(s, cfg) == []  # the ledger closes, z and pitch are in range, every row valid
    # --- the 3D terms ---
    assert totals[Counter3D.Z_FLOOR_CLAMPS] > 0, "nobody ever reached the floor"
    assert led[:, LedgerIdx.HEAT_BUOYANCY].sum() > 0, "the buoyancy actuator was never paid for"
    assert cnt[-LATE:, Counter3D.Z_FLOOR_CLAMPS].sum() > 0, "nobody reached the floor in the last steps"
    # the final state, not the run-wide ratio: the founders start uniform in z, so GROUNDED / N_ALIVE over the
    # whole run lies strictly between 0 and 1 whatever the population does afterwards
    alive = to_numpy(s.alive)
    resting = to_numpy(physics.grounded(s.pos, cfg))[alive]
    assert 0 < resting.sum() < alive.sum(), f"{resting.sum()} of {alive.sum()} rest on the floor at the end"
    over = cnt[:, C.CELL_OVERFLOW] > 0
    assert over.mean() <= OVERFLOW_STEPS_MAX, (
        f"hash cells over capacity in {over.sum()} of {over.size} steps (peak occupancy "
        f"{cnt[:, C.MAX_OCCUPANCY].max()}, {cnt[-LATE:, C.MAX_OCCUPANCY].max()} in the last {LATE}) "
        f"at C = {CELL_CAPACITY}"
    )
    # --- the sizing of regen_max: uniform founders see the whole column (15 600; 8 700 at 70% benthic) ---
    assert 12_000 < run["k0"] < 19_000, run["k0"]


@pytest.mark.slow
def test_dev3d_ecology_is_alive_and_unsaturated() -> None:
    cfg = dev3d_cfg()
    check_dev3d(cfg, run_dev3d(cfg, STEPS))
