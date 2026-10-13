"""Low-level array primitives: the only home of scatters and setitem (§16.1), plus torus math.

Every scatter is an int32 add or max, which is associative, so results do not depend on atomic order (F7).
Helpers that scatter or index check integer dtypes; the checks are on static dtypes, so they are free under
``mx.compile``.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np

_I32 = mx.int32
_LIMB_BITS = 11
_LIMB_MASK = (1 << _LIMB_BITS) - 1
_EXACT_SUM_MAX_LEN = 1 << 20
_TAKE_ONE = 1 << 15

# Smallest float32 ≥ −math.pi: float32(π) > math.pi, so −float32(π) is outside [−π, π) in float64 terms.
_NEG_PI_F32_SAFE = float(np.nextafter(np.float32(-math.pi), np.float32(0.0)))


def _require_int(a: mx.array, name: str) -> None:
    if not mx.issubdtype(a.dtype, mx.integer):
        raise TypeError(f"{name} must have an integer dtype, got {a.dtype}")


def _require_i32(a: mx.array, name: str) -> None:
    if a.dtype != _I32:
        raise TypeError(f"{name} must be int32, got {a.dtype}")


def _as_i32_values(val: mx.array | int, shape: tuple[int, ...]) -> mx.array:
    if isinstance(val, bool) or not isinstance(val, int | mx.array):
        raise TypeError(f"scatter values must be an int32 array or a Python int, got {type(val).__name__}")
    if isinstance(val, int):
        return mx.full(shape, val, dtype=_I32)
    _require_i32(val, "val")
    return mx.broadcast_to(val, shape)


def scatter_add_i32(size: int, idx: mx.array, val: mx.array | int, mask: mx.array | None = None) -> mx.array:
    """``zeros(size).at[idx].add(val)`` in int32; masked-out lanes add 0 at index 0. Duplicates accumulate."""
    _require_int(idx, "idx")
    v = _as_i32_values(val, idx.shape)
    if mask is not None:
        v = mx.where(mask, v, 0)
        idx = mx.where(mask, idx, 0)
    return mx.zeros((size,), dtype=_I32).at[idx.reshape(-1)].add(v.reshape(-1))


def scatter_max_i32(size: int, idx: mx.array, val: mx.array, mask: mx.array) -> mx.array:
    """``full(size, −1).at[idx].maximum(val)`` over lanes where ``mask``; requires ``val ≥ 0``.

    Invalid lanes write −1 to index 0, which never wins. With unique indices this is a placement (F7).
    """
    _require_int(idx, "idx")
    v = _as_i32_values(val, idx.shape)
    v = mx.where(mask, v, -1)
    idx = mx.where(mask, idx, 0)
    return mx.full((size,), -1, dtype=_I32).at[idx.reshape(-1)].maximum(v.reshape(-1))


def set_rows(a: mx.array, idx: mx.array, rows: mx.array) -> mx.array:
    """Return a copy of ``a`` with ``a[idx] = rows``; ``idx`` must be unique. Never mutates ``a`` (F2)."""
    _require_int(idx, "idx")
    if rows.dtype != a.dtype:
        raise TypeError(f"rows dtype {rows.dtype} must equal array dtype {a.dtype}")
    out = mx.array(a)
    out[idx] = rows
    return out


def proportional_take(
    demand: mx.array, target: mx.array, active: mx.array, supply: mx.array
) -> tuple[mx.array, mx.array]:
    """Share each pool ``supply[p]`` among its active demanders in proportion to demand (§10.4, F10).

    Requires ``0 ≤ demand < 2^16``, ``0 ≤ supply < 2^24`` and fewer than 2^15 contenders per pool.
    Returns ``(take (n,), taken (P,))`` with ``take ≤ demand`` and ``taken ≤ supply``; uncontested pools
    are served exactly.
    """
    _require_i32(demand, "demand")
    _require_int(target, "target")
    _require_i32(supply, "supply")
    p = supply.shape[0]
    d = mx.where(active, demand, 0)
    t = mx.where(active, target, 0)
    total = scatter_add_i32(p, t, d)
    q = supply.astype(mx.float32) * _TAKE_ONE / mx.maximum(total, 1).astype(mx.float32)
    # Clamp before the cast so uncontested pools (which ignore this value) cannot overflow int32.
    ratio = mx.floor(mx.minimum(q, _TAKE_ONE + 1)).astype(_I32) - 1
    ratio = mx.where(total <= supply, _TAKE_ONE, mx.clip(ratio, 0, _TAKE_ONE))
    take = (d * ratio[t]) >> 15
    taken = scatter_add_i32(p, t, take)
    return take, taken


def exact_sum(x: mx.array, axis: int | None = None) -> mx.array:
    """Exact int64 sum of a non-negative int32 array (F6) via three 11-bit limbs summed in int32.

    Exact for ``0 ≤ x < 2^31`` and a reduced length ≤ 2^20. No int64 reduction is issued.
    """
    _require_i32(x, "x")
    if axis is None:
        x = x.reshape(-1)
        axis = 0
    if x.shape[axis] > _EXACT_SUM_MAX_LEN:
        raise ValueError(f"exact_sum length {x.shape[axis]} exceeds 2^20")
    s0 = (x & _LIMB_MASK).sum(axis=axis).astype(mx.int64)
    s1 = ((x >> _LIMB_BITS) & _LIMB_MASK).sum(axis=axis).astype(mx.int64)
    s2 = (x >> (2 * _LIMB_BITS)).sum(axis=axis).astype(mx.int64)
    return s0 + s1 * (1 << _LIMB_BITS) + s2 * (1 << (2 * _LIMB_BITS))


def rank_select(mask: mx.array, n: int) -> tuple[mx.array, mx.array]:
    """Indices of the first ``n`` True entries of a 1-D mask: ``(idx (n,), valid (n,))``.

    Lanes past the number of True entries are invalid and hold ``len − 1``.
    """
    c = mx.cumsum(mask.astype(_I32))
    idx = mx.searchsorted(c, mx.arange(1, n + 1, dtype=_I32), side="left").astype(_I32)
    return mx.minimum(idx, mask.shape[0] - 1), mx.arange(n, dtype=_I32) < c[-1]


def place(arr: mx.array, newborn: mx.array, lane: mx.array, lane_vals: mx.array) -> mx.array:
    """Gather-based row write: ``where(newborn, lane_vals[max(lane, 0)], arr)`` (§12.3)."""
    _require_int(lane, "lane")
    if lane_vals.dtype != arr.dtype:
        raise TypeError(f"lane_vals dtype {lane_vals.dtype} must equal array dtype {arr.dtype}")
    vals = lane_vals[mx.maximum(lane, 0)]
    nb = newborn.reshape(newborn.shape + (1,) * (arr.ndim - newborn.ndim))
    return mx.where(nb, vals, arr)


def first_free(partner: mx.array) -> tuple[mx.array, mx.array]:
    """Lowest empty bond slot per row: ``(slot (N,) int32, has_free (N,) bool)``; full rows give slot 0."""
    _require_int(partner, "partner")
    free = partner < 0
    return mx.argmax(free.astype(_I32), axis=1).astype(_I32), free.any(axis=1)


def min_image(d: mx.array, L: float) -> mx.array:
    """Minimum-image displacement ``d − L·round(d/L)``; bitwise odd-symmetric (half-to-even round)."""
    return d - L * mx.round(d / L)


def wrap(x: mx.array, L: float) -> mx.array:
    """Map into ``[0, L)``, guarding the fp32 ``remainder(−ε, L) == L`` case (F8)."""
    r = mx.remainder(x, L)
    return mx.where(r >= L, r - L, r)


def wrap_xy_clamp_z(p: mx.array, L: float, z_lo: float, z_hi: float) -> mx.array:
    """3D positions ``(..., 3)``: x and y wrapped into ``[0, L)`` (``wrap``), z clamped to ``[z_lo, z_hi]``.

    z is never wrapped (−0.3 would become ``L − 0.3``); pass the float32 bounds ``Dims.z_lo``/``Dims.z_hi``.
    """
    assert p.shape[-1] == 3, "wrap_xy_clamp_z takes 3D positions"  # static
    return mx.concatenate([wrap(p[..., :2], L), mx.clip(p[..., 2:], z_lo, z_hi)], axis=-1)


def wrap_angle(a: mx.array) -> mx.array:
    """Map angles into ``[−π, π)``; the result lies inside that interval in float32 and float64 terms."""
    two_pi = 2.0 * math.pi
    r = mx.remainder(a + math.pi, two_pi)
    r = mx.where(r >= two_pi, r - two_pi, r)
    return mx.maximum(r - math.pi, _NEG_PI_F32_SAFE)


def to_body(v: mx.array, cos_h: mx.array, sin_h: mx.array) -> mx.array:
    """Rotate world vectors ``(..., 2)`` into the body frame ``(fwd, lat)``.

    ``cos_h``/``sin_h`` must broadcast against ``v[..., 0]`` (e.g. pass ``cos[:, None]`` for (N,K,2)).
    """
    assert v.shape[-1] == 2, "to_body got 3D vectors (to_body3)"  # static: z would be dropped silently
    vx, vy = v[..., 0], v[..., 1]
    return mx.stack([vx * cos_h + vy * sin_h, vy * cos_h - vx * sin_h], axis=-1)


def body_axes(
    cos_h: mx.array, sin_h: mx.array, cos_p: mx.array, sin_p: mx.array
) -> tuple[mx.array, mx.array, mx.array]:
    """The 3D body frame ``(fwd, lat, up)`` of yaw θ and pitch φ, each ``(..., 3)`` in world axes (§27.2).

    ``fwd = (cφ·cθ, cφ·sθ, sφ)``, ``lat = (−sθ, cθ, 0)`` (always horizontal) and ``up = fwd × lat =
    (−sφ·cθ, −sφ·sθ, cφ)``: right-handed and orthonormal, and ``(θ, φ) = (0, 0)`` gives the world axes.
    """
    fwd = mx.stack([cos_p * cos_h, cos_p * sin_h, sin_p], axis=-1)
    lat = mx.stack([-sin_h, cos_h, mx.zeros_like(cos_h)], axis=-1)
    up = mx.stack([-sin_p * cos_h, -sin_p * sin_h, cos_p], axis=-1)
    return fwd, lat, up


def to_body3(v: mx.array, cos_h: mx.array, sin_h: mx.array, cos_p: mx.array, sin_p: mx.array) -> mx.array:
    """Rotate world vectors ``(..., 3)`` into the body frame ``(fwd, lat, up)`` of ``body_axes`` (§27.2).

    Yaw first (``h`` and ``lat`` exactly as ``to_body``), then pitch in the vertical plane:
    ``fwd = h·cφ + vz·sφ`` and ``up = vz·cφ − h·sφ``. At φ = 0 with ``vz = 0`` the first two components reduce
    to ``to_body``'s. The angle arrays must broadcast against ``v[..., 0]``, as for ``to_body``.
    """
    assert v.shape[-1] == 3, "to_body3 got 2D vectors (to_body)"  # static
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]
    h, lat = vx * cos_h + vy * sin_h, vy * cos_h - vx * sin_h
    return mx.stack([h * cos_p + vz * sin_p, lat, vz * cos_p - h * sin_p], axis=-1)


def int_diffuse(r: mx.array, d_q8: int) -> mx.array:
    """Exact integer flux diffusion on a periodic grid (§13.2); conserves the total exactly.

    Requires ``0 ≤ r`` and ``0 ≤ d_q8 ≤ 64`` so that ``4q ≤ r`` and ``r·d_q8`` fits in int32.
    """
    _require_i32(r, "r")
    if d_q8 == 0:
        return r
    q = (r * d_q8) >> 8
    return (
        r
        - 4 * q
        + mx.roll(q, 1, axis=0)
        + mx.roll(q, -1, axis=0)
        + mx.roll(q, 1, axis=1)
        + mx.roll(q, -1, axis=1)
    )


def int_diffuse_3d(r: mx.array, d_q8: int) -> mx.array:
    """Exact integer flux diffusion on an (FZ,FG,FG) lattice, periodic in x and y, no flux in z (§27.6).

    Each cell sends ``q = (R·d_q8) >> 8`` to each of its four x/y neighbours and to each z neighbour that
    exists: ``R − (4 + n_z)·q + Σ roll₄(q) + q[k−1] + q[k+1]`` with ``n_z(k) = [k > 0] + [k < FZ − 1]`` and
    the missing layers contributing 0. z is never rolled, so nothing crosses the floor or the surface, and the
    total is conserved exactly. Requires ``0 ≤ r`` and ``0 ≤ d_q8 ≤ 42``: then ``6q ≤ r`` keeps every cell
    ``≥ 0`` and the cap ``R ≤ R_HARD`` holds (§27.6). With FZ = 1 it is ``int_diffuse`` on the single layer.
    """
    _require_i32(r, "r")
    if d_q8 == 0:
        return r
    fz = r.shape[0]
    q = (r * d_q8) >> 8
    n_z = mx.array([(k > 0) + (k < fz - 1) for k in range(fz)], dtype=_I32).reshape(fz, 1, 1)
    zero = mx.zeros_like(q[:1])
    return (
        r
        - (4 + n_z) * q
        + mx.roll(q, 1, axis=1)
        + mx.roll(q, -1, axis=1)
        + mx.roll(q, 1, axis=2)
        + mx.roll(q, -1, axis=2)
        + mx.concatenate([zero, q[:-1]], axis=0)  # from the layer below
        + mx.concatenate([q[1:], zero], axis=0)  # from the layer above
    )


def floor_i32(x: mx.array) -> mx.array:
    """``floor(x)`` cast to int32."""
    return mx.floor(x).astype(_I32)


def q8(f: float) -> int:
    """Host-side Q8 fixed-point constant ``round(f·256)``."""
    return round(f * 256)


def mul_q8(x: mx.array, q: int) -> mx.array:
    """``(x·q) >> 8`` in int32 (the caller guarantees ``x·q < 2^31``)."""
    _require_i32(x, "x")
    return (x * q) >> 8
