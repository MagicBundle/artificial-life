"""§27 the 3D ops helpers: ``wrap_xy_clamp_z`` (x and y periodic, z clamped to the float32 bounds)."""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from alife import ops

L, Z_LO, Z_HI = 64.0, 1.0, 23.0


def _np(a: mx.array) -> np.ndarray:
    return np.array(a)


def test_wrap_xy_clamp_z_wraps_xy_and_clamps_z() -> None:
    eps = float(np.float32(1e-7))
    p = mx.array(
        [
            [-eps, L, 0.0],  # F8: remainder(−ε, L) rounds to L; wrap maps it into [0, L)
            [-0.3, 2 * L + 1.5, -0.3],  # z below the floor is clamped, never wrapped to L − 0.3
            [10.0, 20.0, 30.0],  # above the surface
            [63.5, 0.25, 12.0],  # inside
        ],
        dtype=mx.float32,
    )
    q = _np(ops.wrap_xy_clamp_z(p, L, Z_LO, Z_HI))
    assert q.dtype == np.float32 and q.shape == (4, 3)
    assert np.all(q[:, :2] >= 0) and np.all(q[:, :2] < L)
    assert np.array_equal(q[:, :2], _np(ops.wrap(p[:, :2], L)))  # x, y exactly as ``wrap``
    assert q[:, 2].tolist() == [Z_LO, Z_LO, Z_HI, 12.0]  # z lands exactly on the bounds
    assert np.array_equal(_np(ops.wrap_xy_clamp_z(mx.array(q), L, Z_LO, Z_HI)), q)  # idempotent


def test_wrap_xy_clamp_z_broadcasts() -> None:
    p = mx.random.uniform(-2 * L, 2 * L, (5, 4, 3), key=mx.random.key(0))
    q = _np(ops.wrap_xy_clamp_z(p, L, Z_LO, Z_HI))
    assert q.shape == (5, 4, 3)
    assert np.all((q[..., :2] >= 0) & (q[..., :2] < L))
    assert np.all((q[..., 2] >= Z_LO) & (q[..., 2] <= Z_HI))
    inside = (_np(p)[..., 2] >= Z_LO) & (_np(p)[..., 2] <= Z_HI)
    assert np.array_equal(q[..., 2][inside], _np(p)[..., 2][inside])
