"""§29.3 ``groups``: the hash, inherited labels, masked and exact bin counts, grid codes and plane cells.

These run on the default device, so the Metal run of the suite covers the uint64 hash and the integer scatters
(the coordinator's gate); under the CPU plugin they are the CPU check.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from alife import groups
from alife.analysis import phylo

SEEDS = (0, 1, 12345, 2**63 - 1)
BIG = (2**31 - 1, 2**31, 2**40 + 7, 2**62 + 3)
XS = np.concatenate([np.arange(5000, dtype=np.int64), np.array(BIG, dtype=np.int64)])


def _np(a: mx.array) -> np.ndarray:
    return np.array(a)


# --- splitmix64 and the founding cut -----------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_splitmix64_matches_phylo(seed: int) -> None:
    ref = phylo.splitmix64(XS.astype(np.uint64), seed)
    x = mx.array(XS)
    eager = groups.splitmix64(x, seed)
    assert eager.dtype == mx.uint64
    assert np.array_equal(_np(eager), ref)
    compiled = mx.compile(lambda a: groups.splitmix64(a, seed))
    assert np.array_equal(_np(compiled(x)), ref)


def test_splitmix64_wraps_negative_inputs_and_seeds_like_numpy() -> None:
    x = np.array([-1, -2, -(2**40), 0, 7], dtype=np.int64)
    for seed in (-1, -(2**62), 2**64 + 3):
        ref = phylo.splitmix64(x.astype(np.uint64), seed)
        assert np.array_equal(_np(groups.splitmix64(mx.array(x), seed)), ref)
    assert np.array_equal(  # an unsigned input needs no cast
        _np(groups.splitmix64(mx.array(XS.astype(np.uint64)), 5)), phylo.splitmix64(XS.astype(np.uint64), 5)
    )


def test_splitmix64_rejects_floats() -> None:
    with pytest.raises(TypeError, match="integer"):
        groups.splitmix64(mx.zeros((3,), dtype=mx.float32), 0)


@pytest.mark.parametrize("rate", [0.0, 2.0**-11, 0.3, 0.5, 0.75, 1.0])
def test_hash_below_is_the_marker_roots_hit(rate: float) -> None:
    seed = 12345
    got = _np(groups.hash_below(mx.array(XS), seed, rate))
    assert got.dtype == np.bool_
    u = phylo.splitmix64(XS.astype(np.uint64), seed)
    if rate >= 1.0:
        assert got.all()
    elif rate == 0.0:
        assert not got.any()
    else:
        assert np.array_equal(got, u < np.uint64(groups.hash_cut(rate)))
        assert got.any() and not got.all()
    # the same through marker_roots itself: a tree whose only root is agent 0 (a root is always "marked")
    n = XS.size
    tree = phylo.Tree(
        parent=np.concatenate([[-1], np.zeros(n - 1, np.int32)]).astype(np.int32),
        birth=np.zeros(n, np.int32),
        death=np.ones(n, np.int32),
        t_end=1,
    )
    uid = np.arange(n, dtype=np.int64)
    roots = phylo.marker_roots(tree, rate, seed)
    assert np.array_equal(_np(groups.hash_below(mx.array(uid), seed, rate))[1:], roots[1:])


def test_hash_below_is_strict_at_the_cut() -> None:
    # a hash with its low 11 bits zero is exact as rate * 2^64, so a row hashing to the cut is not below it
    xs = np.arange(200_000, dtype=np.uint64)
    u = phylo.splitmix64(xs, 0)
    hit = np.flatnonzero((u & np.uint64(2047)) == 0)
    assert hit.size > 10
    x, cut = int(xs[hit[0]]), int(u[hit[0]])
    rate = cut / 2.0**64
    assert groups.hash_cut(rate) == cut
    got = _np(groups.hash_below(mx.array([x, x + 1, x - 1 if x else x + 2], dtype=mx.int64), 0, rate))
    ref = phylo.splitmix64(np.array([x, x + 1, x - 1 if x else x + 2], np.uint64), 0) < np.uint64(cut)
    assert got.tolist() == ref.tolist() and not got[0]


def test_hash_cut_and_rate_bounds() -> None:
    assert groups.hash_cut(0.0) == 0
    assert groups.hash_cut(2.0**-11) == 2**53
    assert groups.hash_cut(0.5) == 2**63
    assert groups.hash_cut(1.0) == 2**64 - 1
    assert 2**63 < groups.hash_cut(0.75) < 2**64
    x = mx.arange(4, dtype=mx.int64)
    for bad in (-0.1, float("nan")):
        with pytest.raises(ValueError, match="rate"):
            groups.hash_below(x, 0, bad)
    assert _np(groups.hash_below(x, 0, 1.5)).all()  # "all True if rate >= 1"


# --- lanes and inheritance ---------------------------------------------------------------------------------


def test_lane_rows_and_inherit() -> None:
    n = 8
    born_row = mx.array([5, 2, -1, 7], dtype=mx.int32)  # lane 2 is padding
    lane = groups.lane_rows(n, born_row)
    assert lane.dtype == mx.int32
    assert _np(lane).tolist() == [-1, -1, 1, -1, -1, 0, -1, 3]
    prev = mx.array([10, 11, 12, 13, 14, 15, 16, 17], dtype=mx.int64)
    # rows 2 and 7 were dead at the step's start or are reused; row 5 is "kept" but a newborn wins
    keep = mx.array([True, True, False, True, False, True, True, False])
    lane_vals = mx.array([100, 101, 102, 103], dtype=mx.int64)
    out = groups.inherit(prev, keep, lane, lane_vals, -1)
    assert out.dtype == mx.int64
    assert _np(out).tolist() == [10, 11, 101, 13, -1, 100, 16, 103]
    # a row labelled by several columns (N, B): the newborn row takes its lane's whole value
    prev2 = mx.broadcast_to(prev[:, None], (n, 3)) + mx.array([0, 100, 200])
    lane_vals2 = mx.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=mx.int64)
    out2 = _np(groups.inherit(prev2, keep, lane, lane_vals2, -7))
    assert (
        out2[5].tolist() == [1, 2, 3] and out2[2].tolist() == [4, 5, 6] and out2[7].tolist() == [10, 11, 12]
    )
    assert out2[4].tolist() == [-7, -7, -7] and out2[0].tolist() == [10, 110, 210]
    # int32 stays int32
    assert groups.inherit(prev.astype(mx.int32), keep, lane, lane_vals.astype(mx.int32), -1).dtype == mx.int32


def test_lane_rows_with_no_lane_at_all() -> None:
    lane = groups.lane_rows(5, mx.full((3,), -1, dtype=mx.int32))
    assert _np(lane).tolist() == [-1] * 5


# --- masked counts -----------------------------------------------------------------------------------------


def _ref_bincount(size: int, idx: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.zeros(size, dtype=np.int64)
    np.add.at(out, idx[np.broadcast_to(mask, idx.shape)], 1)
    return out


def _case(name: str, rs: np.random.Generator, size: int, lanes: int) -> tuple[np.ndarray, np.ndarray]:
    idx = rs.integers(0, size, lanes).astype(np.int32)
    if name == "random":
        mask = rs.random(lanes) < 0.4
    elif name == "all_masked":
        mask = np.zeros(lanes, bool)
    elif name == "none_masked":
        mask = np.ones(lanes, bool)
    elif name == "one_bin":
        idx[:] = size - 1
        mask = np.ones(lanes, bool)
    elif name == "negative_on_masked":
        mask = rs.random(lanes) < 0.5
        idx = np.where(mask, idx, -rs.integers(1, 10**6, lanes)).astype(np.int32)
    elif name == "huge_on_masked":
        mask = rs.random(lanes) < 0.5
        idx = np.where(mask, idx, 2**31 - 1).astype(np.int32)
    else:
        raise AssertionError(name)
    return idx, mask


CASES = ("random", "all_masked", "none_masked", "one_bin", "negative_on_masked", "huge_on_masked")


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("size,lanes", [(1, 37), (7, 4096), (289, 5000), (3, 20000)])
def test_masked_bincount(name: str, size: int, lanes: int) -> None:
    rs = np.random.default_rng(size * 1000 + lanes)
    idx, mask = _case(name, rs, size, lanes)
    got = groups.masked_bincount(size, mx.array(idx), mx.array(mask))
    assert got.shape == (size,) and got.dtype == mx.int32
    assert np.array_equal(_np(got).astype(np.int64), _ref_bincount(size, idx, mask))
    compiled = mx.compile(lambda i, m: groups.masked_bincount(size, i, m))
    assert np.array_equal(_np(compiled(mx.array(idx), mx.array(mask))), _np(got))


def test_masked_bincount_shapes_and_dtypes() -> None:
    rs = np.random.default_rng(3)
    idx = rs.integers(0, 10, (50, 8)).astype(np.int32)
    row_mask = rs.random(50) < 0.5
    got = groups.masked_bincount(10, mx.array(idx), mx.array(row_mask)[:, None])  # a mask that broadcasts
    assert np.array_equal(_np(got), _ref_bincount(10, idx, row_mask[:, None]))
    got64 = groups.masked_bincount(10, mx.array(idx.astype(np.int64)), mx.array(row_mask)[:, None])
    assert np.array_equal(_np(got64), _np(got))
    assert _np(
        groups.masked_bincount(4, mx.zeros((0,), dtype=mx.int32), mx.zeros((0,), dtype=mx.bool_))
    ).tolist() == [
        0,
        0,
        0,
        0,
    ]


def test_masked_bincount_dump_region_never_leaks() -> None:
    size, lanes = 5, 5000  # 4,995 masked lanes spread over the 1,024 dump bins above the 5 real ones
    idx = np.zeros(lanes, np.int32)
    mask = np.zeros(lanes, bool)
    mask[::1000] = True
    got = _np(groups.masked_bincount(size, mx.array(idx), mx.array(mask)))
    assert got.tolist() == [5, 0, 0, 0, 0]


def test_masked_bincount_limits_are_static() -> None:
    # MLX arrays are lazy: a 2^31-lane input is never materialised because the check precedes any array work
    huge = mx.zeros((2**16, 2**15 + 1), dtype=mx.int8)  # 2^31 + 2^16 lanes
    with pytest.raises(ValueError, match="lanes"):
        groups.masked_bincount(4, huge, mx.ones((1, 1), dtype=mx.bool_))
    with pytest.raises(ValueError, match="lanes"):
        groups.limb_bits(huge.size)
    with pytest.raises(ValueError, match="dump"):
        groups.masked_bincount(2**31 - 100, mx.zeros((2,), dtype=mx.int32), mx.ones((2,), dtype=mx.bool_))


# --- exact sums --------------------------------------------------------------------------------------------


def test_limb_bits_and_masked_bincount_exact() -> None:
    assert groups.limb_bits(0) == 11 and groups.limb_bits(1) == 11
    assert groups.limb_bits(2**20) == 11
    for w in range(11, 1, -1):  # the largest lane count at which a w-bit limb is still exact
        top = (2**31 - 1) // (2**w - 1)
        assert groups.limb_bits(top) == w and groups.limb_bits(top + 1) == w - 1, w
    assert groups.limb_bits(2**21 + 3) == 10
    assert groups.limb_bits(2**21 + 1) == 10 and groups.limb_bits(2**22) == 9
    assert groups.limb_bits(2**31 - 1) == 1
    for lanes in (1, 5, 2**10, 2**20, 2**21 + 3, 2**22, 2**26, 2**31 - 1):
        w = groups.limb_bits(lanes)
        assert lanes * (2**w - 1) <= 2**31 - 1
        assert w == 11 or lanes * (2 ** (w + 1) - 1) > 2**31 - 1
    with pytest.raises(ValueError, match="lanes"):
        groups.limb_bits(2**31)

    def check(lanes: int, bits: int, size: int = 3) -> None:
        # every lane on one bin with the largest value: the worst case of each limb sum
        idx = mx.full((lanes,), size - 1, dtype=mx.int32)
        val = mx.full((lanes,), 2**bits - 1, dtype=mx.int32)
        mask = mx.ones((lanes,), dtype=mx.bool_)
        got = groups.masked_bincount_exact(size, idx, val, mask, bits)
        assert got.dtype == mx.int64 and got.shape == (size,)
        assert _np(got).tolist() == [0, 0, lanes * (2**bits - 1)]

    for bits in (15, 16, 31):
        check(2**18, bits)
    check(2**21 + 3, 16)
    check(2**22, 31)  # w = 9: four limbs


def test_masked_bincount_exact_against_numpy() -> None:
    rs = np.random.default_rng(11)
    for bits in (1, 3, 11, 12, 16, 22, 31):
        size, lanes = 17, 3000
        idx = rs.integers(0, size, lanes).astype(np.int32)
        val = rs.integers(0, 2**bits, lanes).astype(np.int32)
        mask = rs.random(lanes) < 0.6
        idx = np.where(mask, idx, -5).astype(np.int32)  # junk index on the masked lanes
        ref = np.zeros(size, np.int64)
        np.add.at(ref, idx[mask], val[mask].astype(np.int64))
        got = groups.masked_bincount_exact(size, mx.array(idx), mx.array(val), mx.array(mask), bits)
        assert np.array_equal(_np(got), ref), bits
    # a 2-D lane set with a per-row value and a mask that broadcast (the shape of the energy-shared sum)
    idx = rs.integers(0, 9, (200, 4)).astype(np.int32)
    val = rs.integers(0, 2**16, 200).astype(np.int32)
    mask = rs.random((200, 4)) < 0.5
    got = groups.masked_bincount_exact(9, mx.array(idx), mx.array(val)[:, None], mx.array(mask), 16)
    ref = np.zeros(9, np.int64)
    np.add.at(ref, idx[mask], np.broadcast_to(val[:, None], idx.shape)[mask].astype(np.int64))
    assert np.array_equal(_np(got), ref)


def test_masked_bincount_exact_rejects_bad_bits_and_dtypes() -> None:
    idx, val, mask = (
        mx.zeros((4,), dtype=mx.int32),
        mx.zeros((4,), dtype=mx.int32),
        mx.ones((4,), dtype=mx.bool_),
    )
    for bits in (0, 32):
        with pytest.raises(ValueError, match="bits"):
            groups.masked_bincount_exact(2, idx, val, mask, bits)
    with pytest.raises(TypeError, match="integer"):
        groups.masked_bincount_exact(2, idx, val.astype(mx.float32), mask, 8)


# --- grids -------------------------------------------------------------------------------------------------


def _np_grid_code(x: np.ndarray, lo: float, hi: float, bins: int) -> np.ndarray:
    scale = np.float32(bins / (hi - lo))
    c = np.clip(np.floor((x.astype(np.float32) - np.float32(lo)) * scale), 0, bins - 1).astype(np.int64)
    code = c[:, 0]
    for j in range(1, c.shape[1]):
        code = code * bins + c[:, j]
    return code


def test_grid_code_and_plane_cell() -> None:
    # -1, 1, every multiple of 1/8, and the float32 neighbours of each
    marks = np.arange(-8, 9, dtype=np.float32) / np.float32(8)
    vals = np.unique(
        np.concatenate(
            [marks, np.nextafter(marks, np.float32(-2)), np.nextafter(marks, np.float32(2)), [0.5, 0.0]]
        ).astype(np.float32)
    )
    rs = np.random.default_rng(0)
    cols = np.stack([rs.permutation(vals) for _ in range(4)], axis=1)
    got = _np(groups.grid_code(mx.array(cols), -1.0, 1.0, 16))
    assert got.dtype == np.int32
    assert np.array_equal(got.astype(np.int64), _np_grid_code(cols, -1.0, 1.0, 16))
    # the digits: column 0 is the most significant, exact bins at the marks (-1 -> 0, 1 -> 15 by the clip)
    probe = np.array([[-1, -1, -1, -1], [1, 1, 1, 1], [-1, -1, -1, -0.875], [-0.875, -1, -1, -1]], np.float32)
    assert _np(groups.grid_code(mx.array(probe), -1.0, 1.0, 16)).tolist() == [0, 16**4 - 1, 1, 16**3]
    # one column (N,) and a different box
    one = rs.uniform(-3, 5, 200).astype(np.float32)
    assert np.array_equal(
        _np(groups.grid_code(mx.array(one), -3.0, 5.0, 10)).astype(np.int64),
        _np_grid_code(one[:, None], -3.0, 5.0, 10),
    )
    with pytest.raises(ValueError, match="int32"):
        groups.grid_code(mx.zeros((2, 4)), -1.0, 1.0, 256)  # 256^4 = 2^32

    size, cells = 64.0, 32
    pos = np.array(
        [[0.0, 0.0, 5.0], [np.nextafter(np.float32(size), np.float32(0)), 0.0, 0.0], [63.9, 31.9, 1.0],
         [-0.5, size + 1, 2.0], [2.0, 4.0, 63.0], [2.0, 4.0, 63.99]],
        np.float32,
    )  # fmt: skip
    scale = np.float32(cells / size)

    def ref(a0: int, a1: int) -> np.ndarray:
        c0 = np.clip(np.floor(pos[:, a0] * scale), 0, cells - 1).astype(np.int64)
        c1 = np.clip(np.floor(pos[:, a1] * scale), 0, cells - 1).astype(np.int64)
        return c1 * cells + c0

    got = _np(groups.plane_cell(mx.array(pos), size, cells))
    assert got.dtype == np.int32 and np.array_equal(got, ref(0, 1))
    assert (
        got[0] == 0 and got[1] == cells - 1 and got[3] == (cells - 1) * cells
    )  # x < 0 -> 0, y > size -> top
    assert np.array_equal(_np(groups.plane_cell(mx.array(pos), size, cells, axes=(0, 2))), ref(0, 2))
    assert np.array_equal(_np(groups.plane_cell(mx.array(pos), size, cells, axes=(2, 1))), ref(2, 1))
    with pytest.raises(ValueError, match="int32"):
        groups.plane_cell(mx.array(pos), size, 2**16)


def test_compiled_equals_eager() -> None:
    """Every function is shape-static and branch-free on values, so ``mx.compile`` traces it once."""
    rs = np.random.default_rng(7)
    n, m, size = 300, 40, 12
    born_row = mx.array(np.where(rs.random(m) < 0.7, rs.permutation(n)[:m], -1).astype(np.int32))
    keep = mx.array(rs.random(n) < 0.6)
    prev = mx.array(rs.integers(0, 50, n).astype(np.int64))
    lane_vals = mx.array(rs.integers(100, 150, m).astype(np.int64))
    idx = mx.array(rs.integers(0, size, (n, 4)).astype(np.int32))
    mask = mx.array(rs.random((n, 4)) < 0.5)
    val = mx.array(rs.integers(0, 2**16, n).astype(np.int32))[:, None]
    tags = mx.array(rs.uniform(-1.2, 1.2, (n, 4)).astype(np.float32))
    pos = mx.array(rs.uniform(-1.0, 49.0, (n, 3)).astype(np.float32))

    def f(born_row, keep, prev, lane_vals, idx, mask, val, tags, pos):
        lane = groups.lane_rows(n, born_row)
        return (
            lane,
            groups.inherit(prev, keep, lane, lane_vals, -1),
            groups.masked_bincount(size, idx, mask),
            groups.masked_bincount_exact(size, idx, val, mask, 16),
            groups.grid_code(tags, -1.0, 1.0, 16),
            groups.plane_cell(pos, 48.0, 16, axes=(0, 2)),
            groups.hash_below(prev, 3, 0.3),
        )

    args = (born_row, keep, prev, lane_vals, idx, mask, val, tags, pos)
    for a, b in zip(f(*args), mx.compile(f)(*args), strict=True):
        assert a.dtype == b.dtype and np.array_equal(_np(a), _np(b))
