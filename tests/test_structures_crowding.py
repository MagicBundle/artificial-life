"""E7: crowding in the material must not hide neighbours silently (DESIGN §38.7): with structures on, a hash
cell over capacity is an error bit (``Diag.HASH_OVERFLOW``) as well as the ``CELL_OVERFLOW`` counter."""

from __future__ import annotations

import mlx.core as mx

from alife.config import dims
from alife.state import ERROR_MASK, CounterIdx, Diag, to_numpy
from alife.step import make_step
from helpers import make_scene
from structures_helpers import cfg_for


def _crowd(cfg, n_agents: int):
    """``n_agents`` agents inside one hash cell (and one field cell)."""
    d = dims(cfg)
    z = (d.z_lo + 1.0,) if d.d3 else ()
    pos = [(10.0 + 0.01 * i, 10.0 + 0.01 * (i % 7), *z) for i in range(n_agents)]
    return make_scene(cfg, [{"pos": p} for p in pos])


def test_a_crowd_in_one_hash_cell_fails_the_run_with_structures_on_but_only_warns_when_off() -> None:
    on = cfg_for(2)
    c = on.sensing.cell_capacity
    s = _crowd(on, c + 8)
    s1, out = make_step(on)(s)
    mx.eval(s1, out)
    assert int(to_numpy(out.counters)[CounterIdx.CELL_OVERFLOW]) >= 1
    assert int(to_numpy(s1.diag)) & Diag.HASH_OVERFLOW and int(to_numpy(s1.diag)) & ERROR_MASK
    # off: the same crowd is the advisory warning only (nothing changes for existing worlds)
    off = cfg_for(2)
    off = type(off)(**{**off.__dict__, "structures": type(off.structures)()})
    s2, _ = make_step(off)(_crowd(off, c + 8))
    mx.eval(s2)
    diag = int(to_numpy(s2.diag))
    assert diag & Diag.CELL_OVERFLOW_WARN and not diag & Diag.HASH_OVERFLOW


def test_a_stress_run_of_slowed_builders_stays_within_the_hash_capacity() -> None:
    """Forced full builders on tiny in the thickest material the barrier allows: the barrier slows everyone
    inside walls to a quarter speed, and the population must still fit the hash (no overflow, no error
    bit)."""
    from structures_helpers import builders, run

    cfg = cfg_for(2)
    worst = 0
    for t, (_, s, out, _) in enumerate(run(cfg, builders(cfg), 400)):
        worst = max(worst, int(to_numpy(out.counters)[CounterIdx.CELL_OVERFLOW]))
        assert not int(to_numpy(s.diag)) & ERROR_MASK, t
    assert worst == 0
