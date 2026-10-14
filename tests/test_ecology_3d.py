"""Slow 3D ecology check (§22.5, DESIGN §27.13): ``DEV3D``, the ``64k3d`` preset shrunk, for 2 500 steps.

The 3D counterpart of ``test_ecology.py`` and the harness for the ``64k3d`` defaults, to be passed before any
long 3D run. ``DEV3D`` keeps what sets the ecology of ``64k3d`` (gravity, the buoyancy actuator and its costs,
the vertical food profile, the regeneration, the hash capacity ``C = 24``, ``K = 8``) and shrinks what only
sets its size: a 160 x 160 x 40 column (a quarter of the floor, an eighth of the volume), four patches of
sigma 17 (the same share of the floor as eight patches of sigma 24 in 320), 16 384 rows and 4 096 founders.
Weights are float32 as in ``tiny3d_cfg`` (CPU fp16 matmul is about 3x slower, F19); the dynamics do not depend
on the storage dtype.

``n_init`` is half of the occupancy-weighted ``K_est`` of a mostly benthic population (about 8 700, the
arithmetic of §27.13), as ``64k3d``'s is of its 33 000: the founders' first generation multiplies the
population by 1.4 to 1.6 and must not overshoot the food (the 4x over-seeding of the first cloud run did).
The assertions are the 2D ones that make sense in 3D (never extinct, not pinned at ``N_CAP``, births, bonds
and landed attacks, a clean state), plus the 3D terms: floor contacts, ``HEAT_BUOYANCY``, a grounded share
strictly between 0 and 1, and a hash-cell overflow rate near zero.

Run by the nightly job (``-m slow``) and on the cloud worker; not by the fast suite, and not on the Mac while
its GPU is busy.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import numpy as np
import pytest

from alife import invariants, metrics
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
DEV3D = (
    "population.n_max=16384",
    "population.max_births_per_step=256",
    "population.n_init=4096",
    "world.size=160",
    "world.height=40",
    "fields.n_patches=4",
    "fields.patch_sigma=17",
    "brain.weight_dtype=float32",
)
# At most this many hash-cell overflows per alive agent and step: none was seen at C = 16 on tiny3d and DEV3D
# (peak occupancy 13), and C = 24 has 1.8x that headroom; a crowded floor may still spill now and then.
OVERFLOW_RATE_MAX = 1e-4


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
    assert d.fz == 10 and d.q == 27 * 24 and cfg.sensing.cell_capacity == 24  # DEV3D keeps 64k3d's C and K
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
    grounded = totals[Counter3D.GROUNDED] / totals[C.N_ALIVE]
    assert 0.0 < grounded < 1.0, f"grounded share {grounded}: everyone or no one rests on the floor"
    overflow = totals[C.CELL_OVERFLOW] / totals[C.N_ALIVE]
    assert overflow < OVERFLOW_RATE_MAX, f"hash cells overflow {overflow:.2e} per agent and step at C = 24"
    # --- the sizing of regen_max: uniform founders see the whole column (15 600; 8 700 at 70% benthic) ---
    assert 12_000 < run["k0"] < 19_000, run["k0"]


@pytest.mark.slow
def test_dev3d_ecology_is_alive_and_unsaturated() -> None:
    cfg = dev3d_cfg()
    check_dev3d(cfg, run_dev3d(cfg, STEPS))
