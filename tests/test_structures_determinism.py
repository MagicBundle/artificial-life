"""Built structures: determinism, compiled equals eager, resume equals the uninterrupted run, deposit order
independence (DESIGN §38.8)."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from alife import fields, structures
from alife.config import dims
from alife.runio import load_checkpoint, write_state_file
from alife.state import state_digest, to_numpy
from alife.step import make_step
from structures_helpers import builders, cfg_for

STEPS = 60


def _final(cfg, s, steps, compile=None):  # None: the config's own (an eager sphere, a compiled box)
    fn = make_step(cfg, compile=compile)
    outs = []
    for _ in range(steps):
        s, out = fn(s)
        mx.eval(s, out)
        outs.append((to_numpy(out.counters), to_numpy(out.ledger)))
    return s, outs


@pytest.mark.parametrize("nd", [2, 3, "s"])
def test_the_same_seed_gives_the_same_digest(nd: int | str) -> None:
    cfg = cfg_for(nd)
    a, oa = _final(cfg, builders(cfg), STEPS)
    b, ob = _final(cfg, builders(cfg), STEPS)
    assert state_digest(a) == state_digest(b)
    for (ca, la), (cb, lb) in zip(oa, ob, strict=True):
        assert np.array_equal(ca, cb) and np.array_equal(la, lb)


@pytest.mark.parametrize("nd", [2, 3])
def test_the_structures_functions_compiled_equal_eager(nd: int) -> None:
    """The whole step is not bit-equal compiled and eager with or without a feature (the brain's float
    reductions differ from step 0, see ``test_arms_step``), so the structures functions are compared on
    their own: the material update with its aperture gate, the sensing table, the barrier, the shelter and the
    deposit. They are integer or elementwise and must agree exactly on every backend."""
    cfg = cfg_for(nd, "structures.aperture=0.25")
    d = dims(cfg)
    rs = np.random.default_rng(8)
    n = 512
    m = mx.array((rs.integers(0, 4097, d.field_shape) * (rs.random(d.field_shape) < 0.5)).astype(np.int32))
    dep = mx.array((rs.integers(0, 60, d.field_shape) * (rs.random(d.field_shape) < 0.5)).astype(np.int32))
    force = mx.array(rs.normal(0, 0.1, (n, nd)).astype(np.float32))
    fcell = mx.array(rs.integers(0, d.n_cells, n).astype(np.int32))
    units = mx.array(rs.integers(0, 17, n).astype(np.int32))

    # MLX 0.32 fails to compile a graph that builds one roll of one array twice, so each function below
    # builds the material update once (the engine step does too: one ``update`` and one ``sense``)
    def f_update(m_, dep_, t_):
        up = fields.update_mat(m_, dep_, t_, cfg)
        return up.mat, up.wasted, up.decayed, up.nopen

    def f_effects(m_, t_, force_, fcell_, units_):
        sn = structures.sense(m_, fcell_, cfg)
        dm = structures.drag_mult(force_, sn, cfg)
        new, fl = structures.update(
            m_, structures.actions.Build(units_), fcell_, mx.ones((n,), mx.bool_), dm, t_, cfg
        )
        return sn.here, sn.grad, dm, new, fl.built, fl.trapped, fl.slowed, fl.violations

    for t in (0, 5, 17):
        tt = mx.array(t, mx.int32)
        for f, args in ((f_update, (m, dep, tt)), (f_effects, (m, tt, force, fcell, units))):
            for got, want in zip(mx.compile(f)(*args), f(*args), strict=True):
                assert np.array_equal(np.asarray(got), np.asarray(want))


@pytest.mark.parametrize("nd", [2, 3, "s"])
def test_resuming_from_a_checkpoint_equals_the_uninterrupted_run(nd: int | str, tmp_path: Path) -> None:
    cfg = cfg_for(nd)
    s0 = builders(cfg)
    straight, _ = _final(cfg, s0, 2 * STEPS)
    half, _ = _final(cfg, s0, STEPS)
    path = tmp_path / "half.safetensors"
    write_state_file(path, half, cfg)
    resumed, _ = _final(cfg, load_checkpoint(path, cfg), STEPS)
    assert state_digest(resumed) == state_digest(straight)
    assert np.array_equal(to_numpy(resumed.mat), to_numpy(straight.mat)) and to_numpy(straight.mat).any()


def test_the_deposit_does_not_depend_on_the_order_of_the_rows() -> None:
    cfg = cfg_for(2)
    d = dims(cfg)
    rs = np.random.default_rng(4)
    n = d.n
    fcell = rs.integers(0, d.n_cells // 8, n).astype(np.int32)  # crowd into few cells: many collisions
    units = rs.integers(0, 17, n).astype(np.int32)
    alive = rs.random(n) < 0.9
    m0 = mx.array(rs.integers(0, 3000, d.field_shape).astype(np.int32))
    alive0 = mx.array(alive)

    def run(perm):
        dec = structures.actions.Build(mx.array(units[perm]))
        new, fl = structures.update(m0, dec, mx.array(fcell[perm]), alive0, None, mx.array(7, mx.int32), cfg)
        return np.asarray(new), int(fl.built.item())

    ref, built = run(np.arange(n))
    for _ in range(4):
        got, b2 = run(rs.permutation(n))
        assert np.array_equal(got, ref) and b2 == built


def test_the_device_check_flags_a_corrupt_material() -> None:
    """A cell above the cap or below zero sets ``Diag.FIELD_RANGE`` (the device range check)."""
    from alife.state import Diag

    cfg = cfg_for(2)
    s = builders(cfg)
    bad = s._replace(mat=s.mat.at[0, 0].add(-5))  # a negative amount cannot be produced by the update
    s1, _ = make_step(cfg)(bad)
    mx.eval(s1)
    assert int(to_numpy(s1.diag)) & Diag.FIELD_RANGE
    good, _ = make_step(cfg)(s)
    mx.eval(good)
    assert not int(to_numpy(good.diag)) & Diag.FIELD_RANGE
