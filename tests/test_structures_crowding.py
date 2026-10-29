"""E7: crowding in the material is reported, not fatal (DESIGN §38.7).

The barrier slows agents inside walls, so they can crowd into one hash cell and the neighbour search then
hides the agents beyond its capacity. That is the advisory ``Diag.CELL_OVERFLOW_WARN`` plus two counters: the
cells over capacity (``CELL_OVERFLOW``, as in every world) and, with structures on, the agents hidden
(``HASH_HIDDEN``). The run goes on; the runner prints an alert line and the metrics row carries
``build_hash_hidden``. Runs with persistent overflow are excluded from experiments (the measurement suite
gates on it).
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from alife import metrics
from alife.config import dims
from alife.state import ERROR_MASK, CounterIdx, Diag, counter_blocks, counter_index, to_numpy
from alife.step import make_step
from helpers import make_scene
from structures_helpers import cfg_for


def _crowd(cfg, n_agents: int):
    """``n_agents`` agents inside one hash cell (and one field cell)."""
    d = dims(cfg)
    z = (d.z_lo + 1.0,) if d.d3 else ()
    pos = [(10.0 + 0.01 * i, 10.0 + 0.01 * (i % 7), *z) for i in range(n_agents)]
    return make_scene(cfg, [{"pos": p} for p in pos])


def test_a_crowd_in_one_hash_cell_is_counted_and_warned_not_fatal() -> None:
    cfg = cfg_for(2)
    c = cfg.sensing.cell_capacity
    extra = 8
    s1, out = make_step(cfg)(_crowd(cfg, c + extra))
    mx.eval(s1, out)
    cnt = to_numpy(out.counters)
    assert int(cnt[CounterIdx.CELL_OVERFLOW]) >= 1
    assert int(cnt[counter_index(cfg, "HASH_HIDDEN")]) == extra  # the agents beyond C in the one full cell
    diag = int(to_numpy(s1.diag))
    assert diag & Diag.CELL_OVERFLOW_WARN and not diag & ERROR_MASK  # advisory: the run goes on


def test_without_structures_the_same_crowd_is_the_same_warning() -> None:
    off = cfg_for(2)
    off = type(off)(**{**off.__dict__, "structures": type(off.structures)()})
    s2, out = make_step(off)(_crowd(off, off.sensing.cell_capacity + 8))
    mx.eval(s2, out)
    assert int(to_numpy(s2.diag)) & Diag.CELL_OVERFLOW_WARN and not int(to_numpy(s2.diag)) & ERROR_MASK
    assert counter_blocks(off) == ()  # no hidden-agent counter outside the structures block


def test_the_metrics_row_sums_the_hidden_agents_in_64_bits() -> None:
    cfg = cfg_for(2)
    blocks = counter_blocks(cfg)
    t = 4000
    c = np.zeros((t, metrics.n_counters(2, False, blocks)), np.int32)
    c[:, counter_index(cfg, "HASH_HIDDEN")] = 1 << 20
    c[:, CounterIdx.N_ALIVE] = 10
    row = metrics.interval_row(4000, c, np.zeros((t, 16), np.int64), {}, {"wall_s": 1.0}, nd=2, blocks=blocks)
    assert row["build_hash_hidden"] == t * (1 << 20) > 2**31


def test_a_stress_run_of_slowed_builders_stays_within_the_hash_capacity() -> None:
    """Forced full builders on tiny: the barrier slows everyone inside walls to a quarter speed; the
    population must still fit the hash (no overflow, nobody hidden) and no error bit."""
    from structures_helpers import builders, run

    cfg = cfg_for(2)
    worst = hidden = 0
    for t, (_, s, out, _r) in enumerate(run(cfg, builders(cfg), 400)):
        worst = max(worst, int(to_numpy(out.counters)[CounterIdx.CELL_OVERFLOW]))
        hidden += int(to_numpy(out.counters)[counter_index(cfg, "HASH_HIDDEN")])
        assert not int(to_numpy(s.diag)) & ERROR_MASK, t
    assert worst == 0 and hidden == 0
