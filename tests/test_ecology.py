"""Slow ecology check (§22.5): the ``dev`` preset with random founders for 3 000 steps. Owned by WP4.

This is the tuning harness for the §17 defaults: the population must neither go extinct nor sit pinned at
``N_CAP``, and births, bonds and landed attacks must all occur. Weights are float32 as in ``tiny_cfg``
(CPU fp16 matmul is ≈ 3× slower, F19); the dynamics do not depend on the storage dtype.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from alife import invariants
from alife.config import dims, load_config
from alife.state import ERROR_MASK, N_COUNTERS, CounterIdx, init_state, to_numpy
from alife.step import make_step

STEPS = 3000
C = CounterIdx


@pytest.mark.slow
def test_dev_ecology_is_alive_and_unsaturated() -> None:
    cfg = load_config(preset="dev", overrides=["brain.weight_dtype=float32"])
    n_cap = dims(cfg).n_cap
    fn = make_step(cfg)
    s = init_state(cfg)
    counters = np.zeros((STEPS, N_COUNTERS), np.int64)
    for i in range(STEPS):
        s, out = fn(s)
        mx.eval(s, out)
        counters[i] = to_numpy(out.counters)
        assert counters[i, C.N_ALIVE] > 0, f"extinct at step {i}"
    n = counters[:, C.N_ALIVE]
    assert (n == n_cap).mean() < 0.5, "population pinned at N_CAP"
    assert 0 < n[-1] < n_cap
    totals = counters.sum(axis=0)
    assert totals[C.BIRTHS] > 0
    assert totals[C.BONDS_FORMED] + totals[C.CHILD_BONDS] > 0 and counters[-1, C.BONDS_TOTAL] > 0
    assert totals[C.ATTACK_HITS] > 0
    assert int(to_numpy(s.diag)) & ERROR_MASK == 0
    assert invariants.full_check(s, cfg) == []
