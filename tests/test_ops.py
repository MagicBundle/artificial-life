"""§16.2 ops: contention, exact sums, placement, scatters, setitem and torus math."""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from alife import ops
from alife.state import R_HARD


def _np(a: mx.array) -> np.ndarray:
    return np.array(a)


# --- proportional_take -----------------------------------------------------------------------------------


def _check_take(demand: np.ndarray, target: np.ndarray, active: np.ndarray, supply: np.ndarray) -> None:
    take, taken = ops.proportional_take(
        mx.array(demand), mx.array(target), mx.array(active), mx.array(supply)
    )
    take, taken = _np(take).astype(np.int64), _np(taken).astype(np.int64)
    p = supply.shape[0]
    d = np.where(active, demand, 0).astype(np.int64)
    t = np.where(active, target, 0)
    D = np.bincount(t, weights=d, minlength=p).astype(np.int64)
    contenders = np.bincount(t, weights=(d > 0), minlength=p).astype(np.int64)
    assert take.shape == demand.shape and taken.shape == supply.shape
    assert np.all(take >= 0) and np.all(take <= d), "0 ≤ take ≤ demand"
    assert np.all(take[~active] == 0)
    assert np.array_equal(taken, np.bincount(t, weights=take, minlength=p).astype(np.int64))
    assert np.all(taken <= supply), "taken ≤ supply"
    unc = supply >= D
    assert np.array_equal(take[unc[t]], d[unc[t]]), "uncontested pools served exactly"
    left = supply - taken
    bound = contenders + 2 * D / 2**15 + 1
    assert np.all(left[~unc] <= bound[~unc]), "contested leftover bound"


def test_proportional_take_random() -> None:
    rs = np.random.default_rng(0)
    for trial in range(500):
        n = int(rs.integers(1, 120))
        p = int(rs.integers(1, 12))
        dmax = int(rs.choice([1, 50, 4000, 2**16]))
        smax = int(rs.choice([1, 100, 10_000, 2**24]))
        demand = rs.integers(0, dmax, n).astype(np.int32)
        if trial % 7 == 0:
            demand[:] = 2**16 - 1
        supply = rs.integers(0, smax, p).astype(np.int32)
        if trial % 11 == 0:
            supply[:] = 0
        target = rs.integers(0, p, n).astype(np.int32)
        active = rs.random(n) < 0.8
        target = np.where(active, target, -1).astype(np.int32)  # inactive lanes may hold −1
        _check_take(demand, target, active, supply)


def test_proportional_take_edges() -> None:
    # Single pool, exact proportional split when it divides evenly.
    take, taken = ops.proportional_take(
        mx.array([100, 300], dtype=mx.int32),
        mx.array([0, 0], dtype=mx.int32),
        mx.array([True, True]),
        mx.array([200], dtype=mx.int32),
    )
    assert _np(take).sum() <= 200 and _np(taken)[0] == _np(take).sum()
    assert abs(int(_np(take)[0]) - 50) <= 1 and abs(int(_np(take)[1]) - 150) <= 1
    # Zero demand and zero supply.
    _check_take(np.zeros(5, np.int32), np.zeros(5, np.int32), np.ones(5, bool), np.array([7], np.int32))
    _check_take(np.full(5, 9, np.int32), np.zeros(5, np.int32), np.ones(5, bool), np.zeros(1, np.int32))
    # Max demand against max supply, many contenders.
    n = 4096
    _check_take(
        np.full(n, 2**16 - 1, np.int32),
        np.zeros(n, np.int32),
        np.ones(n, bool),
        np.array([2**24 - 1], np.int32),
    )
    with pytest.raises(TypeError):
        ops.proportional_take(mx.array([1.0]), mx.array([0]), mx.array([True]), mx.array([1], dtype=mx.int32))


# --- exact_sum -------------------------------------------------------------------------------------------


def test_exact_sum_max_range() -> None:
    n = 1 << 20
    x = np.full(n, 2**31 - 1, dtype=np.int32)
    got = ops.exact_sum(mx.array(x))
    assert got.dtype == mx.int64 and got.shape == ()
    assert int(_np(got)) == int(x.astype(np.int64).sum()) == (2**31 - 1) * n
    rs = np.random.default_rng(1)
    y = rs.integers(0, 2**31, n, dtype=np.int64).astype(np.int32)
    assert int(_np(ops.exact_sum(mx.array(y)))) == int(y.astype(np.int64).sum())


def test_exact_sum_axis() -> None:
    rs = np.random.default_rng(2)
    x = rs.integers(0, 2**31, (1000, 5), dtype=np.int64).astype(np.int32)
    got = _np(ops.exact_sum(mx.array(x), axis=0))
    assert got.dtype == np.int64 and np.array_equal(got, x.astype(np.int64).sum(0))
    got1 = _np(ops.exact_sum(mx.array(x), axis=1))
    assert np.array_equal(got1, x.astype(np.int64).sum(1))
    assert int(_np(ops.exact_sum(mx.array(x)))) == int(x.astype(np.int64).sum())


def test_exact_sum_rejects() -> None:
    with pytest.raises(TypeError):
        ops.exact_sum(mx.array([1.0, 2.0]))
    with pytest.raises(TypeError):
        ops.exact_sum(mx.array([1, 2], dtype=mx.int64))
    with pytest.raises(ValueError):
        ops.exact_sum(mx.zeros(((1 << 20) + 1,), dtype=mx.int32))


# --- rank_select -----------------------------------------------------------------------------------------


@pytest.mark.parametrize("p_true", [0.0, 0.05, 0.5, 1.0])
def test_rank_select_matches_flatnonzero(p_true: float) -> None:
    rs = np.random.default_rng(3)
    for n_sel in (1, 8, 64):
        mask = rs.random(200) < p_true
        idx, valid = ops.rank_select(mx.array(mask), n_sel)
        idx, valid = _np(idx), _np(valid)
        assert idx.dtype == np.int32 and idx.shape == (n_sel,) and valid.shape == (n_sel,)
        ref = np.flatnonzero(mask)[:n_sel]
        k = len(ref)
        assert np.array_equal(valid, np.arange(n_sel) < k)
        assert np.array_equal(idx[:k], ref)
        assert np.all(idx[k:] == 199)


# --- scatters, set_rows, place --------------------------------------------------------------------------


def test_scatter_add_duplicates_and_mask() -> None:
    idx = mx.array([0, 2, 2, 2, -1], dtype=mx.int32)
    val = mx.array([1, 2, 3, 4, 100], dtype=mx.int32)
    mask = mx.array([True, True, True, False, False])
    assert _np(ops.scatter_add_i32(4, idx, val, mask)).tolist() == [1, 0, 5, 0]
    assert _np(ops.scatter_add_i32(3, mx.array([1, 1, 2], dtype=mx.int32), 1)).tolist() == [0, 2, 1]
    out = ops.scatter_add_i32(3, mx.array([[0, 1], [1, 1]], dtype=mx.int32), mx.ones((2, 2), dtype=mx.int32))
    assert out.dtype == mx.int32 and _np(out).tolist() == [1, 3, 0]


def test_scatter_max_duplicates_and_mask() -> None:
    idx = mx.array([1, 1, 3, 0, -1], dtype=mx.int32)
    val = mx.array([5, 9, 2, 7, 50], dtype=mx.int32)
    mask = mx.array([True, True, True, False, False])
    assert _np(ops.scatter_max_i32(5, idx, val, mask)).tolist() == [-1, 9, -1, 2, -1]
    # uint32 indices (from argsort/permutation) are accepted.
    out = ops.scatter_max_i32(
        3, mx.array([2, 0], dtype=mx.uint32), mx.array([4, 1], dtype=mx.int32), mx.array([True, True])
    )
    assert _np(out).tolist() == [1, -1, 4]


def test_scatters_reject_floats() -> None:
    f = mx.array([1.0, 2.0])
    i = mx.array([0, 1], dtype=mx.int32)
    m = mx.array([True, True])
    with pytest.raises(TypeError):
        ops.scatter_add_i32(2, i, f)
    with pytest.raises(TypeError):
        ops.scatter_add_i32(2, f, i)
    with pytest.raises(TypeError):
        ops.scatter_max_i32(2, i, f, m)
    with pytest.raises(TypeError):
        ops.scatter_max_i32(2, f, i, m)
    with pytest.raises(TypeError):
        ops.scatter_add_i32(2, i, 1.5)  # type: ignore[arg-type]


def test_set_rows_does_not_mutate() -> None:
    a = mx.zeros((6, 3), dtype=mx.float16)
    rows = mx.ones((2, 3), dtype=mx.float16)
    out = ops.set_rows(a, mx.array([1, 4], dtype=mx.int32), rows)
    assert _np(a).sum() == 0, "input unchanged (F2)"
    assert _np(out)[[1, 4]].tolist() == [[1, 1, 1], [1, 1, 1]] and _np(out).sum() == 6
    with pytest.raises(TypeError):
        ops.set_rows(a, mx.array([1.0]), rows[:1])
    with pytest.raises(TypeError):
        ops.set_rows(a, mx.array([1], dtype=mx.int32), mx.ones((1, 3), dtype=mx.float32))
    fn = mx.compile(lambda x, i, r: ops.set_rows(x, i, r))
    out_c = fn(a, mx.array([0, 5], dtype=mx.int32), rows)
    assert _np(out_c)[[0, 5]].sum() == 6 and _np(a).sum() == 0


def test_place() -> None:
    arr = mx.array([10, 11, 12, 13], dtype=mx.int64)
    newborn = mx.array([False, True, False, True])
    lane = mx.array([-1, 1, -1, 0], dtype=mx.int32)
    vals = mx.array([100, 200], dtype=mx.int64)
    out = ops.place(arr, newborn, lane, vals)
    assert out.dtype == mx.int64 and _np(out).tolist() == [10, 200, 12, 100]
    assert _np(arr).tolist() == [10, 11, 12, 13]
    arr2 = mx.zeros((4, 2), dtype=mx.float32)
    out2 = ops.place(arr2, newborn, lane, mx.array([[1.0, 2.0], [3.0, 4.0]]))
    assert _np(out2).tolist() == [[0, 0], [3, 4], [0, 0], [1, 2]]
    with pytest.raises(TypeError):
        ops.place(arr, newborn, lane, mx.array([1, 2], dtype=mx.int32))
    with pytest.raises(TypeError):
        ops.place(arr, newborn, mx.array([0.0, 1, 0, 0]), vals)


def test_first_free() -> None:
    partner = mx.array([[-1, -1, -1, -1], [3, -1, 5, -1], [1, 2, 3, 4], [7, 8, 9, -1]], dtype=mx.int32)
    slot, has_free = ops.first_free(partner)
    assert slot.dtype == mx.int32
    assert _np(slot).tolist() == [0, 1, 0, 3]
    assert _np(has_free).tolist() == [True, True, False, True]


# --- torus math -----------------------------------------------------------------------------------------


def test_min_image_antisymmetric_bitwise() -> None:
    rs = np.random.default_rng(4)
    L = 128.0
    d = rs.uniform(-1.5 * L, 1.5 * L, 10000).astype(np.float32)
    d[:6] = [L / 2, -L / 2, 1.5 * L, -1.5 * L, 0.0, L]
    a = _np(ops.min_image(mx.array(d), L))
    b = _np(ops.min_image(mx.array(-d), L))
    assert np.array_equal(a, -b), "min_image(−d) == −min_image(d) bitwise"
    assert np.all(np.abs(a) <= L / 2)


def test_wrap_never_returns_L() -> None:
    L = 1024.0
    x = np.array(
        [-1e-8, -1e-30, -0.0, 0.0, L, -L, 2 * L - 1e-4, L - 1e-5, -3 * L - 1e-6, 5e6, -5e6], np.float32
    )
    rs = np.random.default_rng(5)
    x = np.concatenate([x, rs.uniform(-3 * L, 3 * L, 10000).astype(np.float32)])
    w = _np(ops.wrap(mx.array(x), L))
    assert np.all(w >= 0) and np.all(w < L), "0 ≤ wrap(x) < L (F8)"
    assert float(_np(ops.wrap(mx.array([-1e-8]), L))[0]) == 0.0


def test_wrap_angle_range() -> None:
    rs = np.random.default_rng(6)
    pi32 = float(np.float32(np.pi))
    a = np.concatenate(
        [
            np.array(
                [-pi32, pi32, -math.pi, math.pi, 0.0, 2 * pi32, -2 * pi32, 1e-8, -1e-8, 3 * pi32], np.float32
            ),
            rs.uniform(-50, 50, 10000).astype(np.float32),
        ]
    )
    w = _np(ops.wrap_angle(mx.array(a)))
    assert w.dtype == np.float32
    assert np.all(w >= -math.pi) and np.all(w < math.pi), "inside [−π, π) in float64 terms"
    assert np.all(w >= -np.float32(np.pi)) and np.all(w < np.float32(np.pi)), "and in float32 terms"
    # Equal to the input modulo 2π (away from the seam).
    inner = np.abs(a) < 3.0
    assert np.allclose(w[inner], a[inner], atol=1e-6)
    assert np.allclose(np.cos(w), np.cos(a), atol=1e-4) and np.allclose(np.sin(w), np.sin(a), atol=1e-4)


def test_to_body() -> None:
    v = mx.array([[1.0, 0.0], [0.0, 1.0], [3.0, 4.0]])
    th = mx.array([0.0, math.pi / 2, math.pi / 2])
    out = _np(ops.to_body(v, mx.cos(th), mx.sin(th)))
    assert np.allclose(out, [[1, 0], [1, 0], [4, -3]], atol=1e-6)
    vk = mx.ones((2, 3, 2))
    out_k = ops.to_body(vk, mx.cos(th[:2])[:, None], mx.sin(th[:2])[:, None])
    assert out_k.shape == (2, 3, 2)


# --- integer helpers ------------------------------------------------------------------------------------


def test_int_diffuse_conserves() -> None:
    rs = np.random.default_rng(7)
    fields = [
        rs.integers(0, 8000, (32, 32)).astype(np.int32),
        np.full((16, 16), R_HARD, np.int32),
        np.zeros((8, 8), np.int32),
        np.where(rs.random((12, 12)) < 0.1, R_HARD, 0).astype(np.int32),
        np.arange(9, dtype=np.int32).reshape(3, 3),
        rs.integers(R_HARD - 300, R_HARD + 1, (16, 16)).astype(np.int32),
    ]
    for r in fields:
        for d_q8 in (0, 1, 4, 64):
            out = ops.int_diffuse(mx.array(r), d_q8)
            o = _np(out)
            assert out.dtype == mx.int32
            assert o.astype(np.int64).sum() == r.astype(np.int64).sum(), "exact conservation"
            assert o.min() >= 0 and o.max() <= R_HARD
            q = (r.astype(np.int64) * d_q8) >> 8
            ref = r - 4 * q + sum(np.roll(q, s, a) for s in (1, -1) for a in (0, 1))
            assert np.array_equal(o, ref)
    spike = np.zeros((5, 5), np.int32)
    spike[2, 2] = 256
    o = _np(ops.int_diffuse(mx.array(spike), 4))
    assert o[2, 2] == 256 - 16 and o[1, 2] == o[3, 2] == o[2, 1] == o[2, 3] == 4


def test_q8_and_mul_q8() -> None:
    assert ops.q8(0.9) == 230 and ops.q8(0.7) == 179 and ops.q8(1.0) == 256 and ops.q8(0.0) == 0
    x = mx.array([0, 1, 255, 256, 1000, 1 << 20], dtype=mx.int32)
    assert _np(ops.mul_q8(x, 230)).tolist() == [(v * 230) >> 8 for v in [0, 1, 255, 256, 1000, 1 << 20]]
    with pytest.raises(TypeError):
        ops.mul_q8(mx.array([1.0]), 3)


def test_floor_i32() -> None:
    out = ops.floor_i32(mx.array([-1.5, -0.0, 0.9, 2.0, 7.99]))
    assert out.dtype == mx.int32 and _np(out).tolist() == [-2, 0, 0, 2, 7]
