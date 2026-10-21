"""Generic grouping primitives in pure MLX (DESIGN §29.3): the device half of social tracking.

Hash-founded labels that newborn rows inherit, masked bin counts (exact int64 sums by limbs whose width
follows the lane count), grid codes and plane cells. Nothing here names a game: a later game whose tracked
systems are not agents (storms on a grid) uses the same functions. Like the simulation modules it reads no
value on the host and branches only on Python constants (shapes, rates, bit widths), and its scatters go
through ``ops``.

Constants above 2^63 are built as ``mx.array(np.uint64(c))``: a Python int of that size fails in MLX 0.32.3
(``std::bad_cast``), as does comparing a uint64 array with one.
"""

from __future__ import annotations

from typing import Final

import mlx.core as mx
import numpy as np

from alife import ops

DUMP: Final = 1024  # spread region for masked lanes (avoids one hot atomic address)

_I32_MAX: Final = 2**31 - 1
_MASK64: Final = 0xFFFFFFFFFFFFFFFF
_MAX_LIMB_BITS: Final = 11


def _u64(c: int) -> mx.array:
    return mx.array(np.uint64(c))


def _require_int(a: mx.array, name: str) -> None:
    if not mx.issubdtype(a.dtype, mx.integer):
        raise TypeError(f"{name} must have an integer dtype, got {a.dtype}")


# --- hashing ----------------------------------------------------------------------------------------------


def splitmix64(x: mx.array, seed: int) -> mx.array:
    """A stateless 64-bit hash (SplitMix64 finaliser) of ``x ^ seed``, uint64 with wrapping arithmetic.

    Bit-identical to ``analysis.phylo.splitmix64``: a negative ``x`` wraps as numpy's cast does, and ``seed``
    is taken modulo 2^64.
    """
    _require_int(x, "x")
    z = x.astype(mx.uint64) ^ _u64(seed & _MASK64)
    z = z + _u64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> 30)) * _u64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * _u64(0x94D049BB133111EB)
    return z ^ (z >> 31)


def hash_cut(rate: float) -> int:
    """The cut of ``phylo.marker_roots``: ``min(int(rate * 2^64), 2^64 - 1)``, a Python int in [0, 2^64)."""
    return min(int(rate * 2.0**64), 2**64 - 1)


def hash_below(x: mx.array, seed: int, rate: float) -> mx.array:
    """``splitmix64(x, seed) < rate * 2^64`` as a bool array: exactly the ``hit`` of ``phylo.marker_roots``.

    The cut is always a uint64 array, never a Python int (comparing a uint64 array with a Python int of 2^63
    or more raises in MLX 0.32.3, and every rate in [0.5, 1) has such a cut). All True if ``rate >= 1`` and
    all False if ``rate == 0`` (static branches on the float); a negative or NaN rate raises ``ValueError``.
    """
    if not rate >= 0.0:
        raise ValueError(f"hash rate {rate} must be >= 0")
    if rate >= 1.0:
        return mx.ones(x.shape, dtype=mx.bool_)
    if rate == 0.0:
        return mx.zeros(x.shape, dtype=mx.bool_)
    return splitmix64(x, seed) < _u64(hash_cut(rate))


# --- labels carried by rows -------------------------------------------------------------------------------


def lane_rows(n: int, born_row: mx.array) -> mx.array:
    """(N,) int32 row -> lane, -1 if no lane targets the row.

    ``scatter_max`` of the lane index at ``born_row`` over the lanes with ``born_row >= 0``. Child rows are
    distinct (lanes target distinct non-survivor rows, ``lifecycle.allocate_lanes``), so it is a placement.
    """
    m = born_row.shape[0]
    return ops.scatter_max_i32(n, mx.maximum(born_row, 0), mx.arange(m, dtype=mx.int32), born_row >= 0)


def inherit(prev: mx.array, keep: mx.array, lane: mx.array, lane_vals: mx.array, fill: int) -> mx.array:
    """Newborn rows take their lane's value, kept rows their previous value, every other row ``fill``.

    ``ops.place(where(keep, prev, fill), lane >= 0, lane, lane_vals)``; ``lane`` is ``lane_rows``'s output and
    ``lane_vals`` (one entry per lane) has ``prev``'s dtype. A newborn row is overwritten even if it is kept.
    """
    k = keep.reshape(keep.shape + (1,) * (prev.ndim - keep.ndim))
    return ops.place(mx.where(k, prev, fill), lane >= 0, lane, lane_vals)


# --- counts -----------------------------------------------------------------------------------------------


def _spread(size: int, idx: mx.array, mask: mx.array) -> mx.array:
    """Flat int32 bin per lane, with masked lanes sent to ``size + (lane & (DUMP - 1))``."""
    lanes = idx.size
    if lanes > _I32_MAX:
        raise ValueError(f"{lanes} lanes exceed the exact int32 bin count (2^31 - 1)")
    if size + DUMP > _I32_MAX:
        raise ValueError(f"{size} bins plus the dump region exceed int32")
    m = mx.broadcast_to(mask, idx.shape).reshape(-1)
    spread = size + (mx.arange(lanes, dtype=mx.int32) & (DUMP - 1))
    return mx.where(m, idx.reshape(-1).astype(mx.int32), spread)


def masked_bincount(size: int, idx: mx.array, mask: mx.array) -> mx.array:
    """(size,) int32 counts of ``idx`` over the lanes where ``mask``, whatever ``idx`` holds on masked lanes.

    ``idx`` may have any shape and ``mask`` broadcasts to it. Masked lanes go to
    ``size + (lane & (DUMP - 1))`` and ``ops.scatter_add_i32(size + DUMP, ...)[:size]`` drops that region, so
    a mostly masked input does not pile onto one atomic address. Exact while the lane count is at most
    2^31 - 1 (a static ``ValueError`` otherwise: unreachable under ``validate``, at most 24 * 2^20 lanes).
    """
    eff = _spread(size, idx, mask)
    return ops.scatter_add_i32(size + DUMP, eff, 1)[:size]


def limb_bits(lanes: int) -> int:
    """The largest ``w <= 11`` with ``lanes * (2^w - 1) <= 2^31 - 1``: each int32 bin sum of a limb is exact.

    Static. 11 up to 2^20 lanes (every 64k sum), 10 up to 2^21, 9 at 2^22; ``ValueError`` above 2^31 - 1.
    """
    if lanes > _I32_MAX:
        raise ValueError(f"{lanes} lanes exceed the exact int32 bin count (2^31 - 1)")
    for w in range(_MAX_LIMB_BITS, 0, -1):
        if max(lanes, 1) * ((1 << w) - 1) <= _I32_MAX:
            return w
    raise AssertionError("unreachable: w = 1 holds for every lane count up to 2^31 - 1")


def masked_bincount_exact(size: int, idx: mx.array, val: mx.array, mask: mx.array, bits: int) -> mx.array:
    """(size,) int64 exact sum of ``val`` per bin over the lanes where ``mask``.

    ``val`` is an integer array (cast to int32) with ``0 <= val < 2^bits`` and ``bits <= 31``; it and
    ``mask`` broadcast to ``idx``'s shape. With ``w = limb_bits(idx.size)`` the sum is split into
    ``ceil(bits / w)`` limbs of ``w`` bits; each limb is an ``ops.scatter_add_i32`` (masked lanes go to the
    dump region, which is dropped, and add at most ``2^w - 1`` there like any lane), exact in int32, cast to
    int64 and shifted. This generalises ``ops.exact_sum`` to any lane count: ``w = 11`` up to 2^20 lanes,
    ``w = 9`` for the 4N-lane sums at ``n_max = 2^20``.
    """
    if not 1 <= bits <= 31:
        raise ValueError(f"bits {bits} must be in [1, 31]")
    _require_int(val, "val")
    w = limb_bits(idx.size)
    eff = _spread(size, idx, mask)
    v = mx.broadcast_to(val.astype(mx.int32), idx.shape).reshape(-1)
    out = mx.zeros((size,), dtype=mx.int64)
    for j in range(-(-bits // w)):
        limb = (v >> (j * w)) & ((1 << w) - 1)
        out = out + ops.scatter_add_i32(size + DUMP, eff, limb)[:size].astype(mx.int64) * (1 << (j * w))
    return out


# --- grids ------------------------------------------------------------------------------------------------


def grid_code(x: mx.array, lo: float, hi: float, bins: int) -> mx.array:
    """(N,) int32 mixed-radix code of ``clip(floor((x - lo) * (bins / (hi - lo))), 0, bins - 1)`` per column.

    ``x`` is (N, T) (or (N,), one column) and finite; column 0 is the most significant digit. With
    ``lo = -1``, ``hi = 1``, ``bins = 16`` the scale is 8.0 and the code is exact on every IEEE backend.
    ``bins ** T`` must fit int32 (``ValueError`` otherwise).
    """
    if x.ndim == 1:
        x = x[:, None]
    t = x.shape[1]
    if bins**t > 2**31:
        raise ValueError(f"{bins}^{t} grid codes exceed int32")
    scale = bins / (hi - lo)
    c = mx.clip(mx.floor((x.astype(mx.float32) - lo) * scale), 0, bins - 1).astype(mx.int32)
    code = c[:, 0]
    for j in range(1, t):
        code = code * bins + c[:, j]
    return code


def plane_cell(pos: mx.array, size: float, cells: int, axes: tuple[int, int] = (0, 1)) -> mx.array:
    """(N,) int32 ``c1 * cells + c0``, ``c_j = clip(floor(pos[:, axes[j]] * (cells / size)), 0, cells - 1)``.

    The cell of a projection on the plane of two position columns. The game passes its floor axes ``(0, 1)``;
    column 2 is height in 3D. A position at 0 or below is in cell 0, one at ``size`` or more in ``cells - 1``.
    """
    if cells * cells > _I32_MAX:
        raise ValueError(f"{cells}^2 plane cells exceed int32")
    scale = cells / size
    p = pos.astype(mx.float32)
    c0 = mx.clip(mx.floor(p[:, axes[0]] * scale), 0, cells - 1).astype(mx.int32)
    c1 = mx.clip(mx.floor(p[:, axes[1]] * scale), 0, cells - 1).astype(mx.int32)
    return c1 * cells + c0
