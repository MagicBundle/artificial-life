"""§27.2 the 3D ops helpers: ``wrap_xy_clamp_z`` (x and y periodic, z clamped to the float32 bounds), the body
frame (``body_axes``, ``to_body3``), ``min_image`` as the identity on z, and the 2D helpers' shape guards."""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from alife import layout, ops

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


def test_wrap_xy_clamp_z_refuses_2d_positions() -> None:
    with pytest.raises(AssertionError, match="3D positions"):
        ops.wrap_xy_clamp_z(mx.zeros((4, 2)), L, Z_LO, Z_HI)


# --- the body frame ---------------------------------------------------------------------------------------

PM = layout.PITCH_MAX
# (yaw, pitch): the named cases, then a random spread over the whole pitch range.
_ANGLES = np.concatenate(
    [
        np.array(
            [[0.0, 0.0], [math.pi / 2, 0.0], [0.0, math.pi / 4], [0.3, PM], [-2.0, -PM], [math.pi, 0.1]]
        ),
        np.stack(
            [
                np.random.default_rng(0).uniform(-math.pi, math.pi, 64),
                np.random.default_rng(1).uniform(-PM, PM, 64),
            ],
            axis=1,
        ),
    ]
).astype(np.float32)


def _trig(angles: np.ndarray) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    th, ph = mx.array(angles[:, 0]), mx.array(angles[:, 1])
    return mx.cos(th), mx.sin(th), mx.cos(ph), mx.sin(ph)


def test_body_axes_orthonormal_and_right_handed() -> None:
    fwd, lat, up = (_np(a).astype(np.float64) for a in ops.body_axes(*_trig(_ANGLES)))
    assert fwd.shape == lat.shape == up.shape == (len(_ANGLES), 3)
    for a, b in ((fwd, fwd), (lat, lat), (up, up)):
        np.testing.assert_allclose((a * b).sum(-1), 1.0, atol=1e-6)
    for a, b in ((fwd, lat), (fwd, up), (lat, up)):
        np.testing.assert_allclose((a * b).sum(-1), 0.0, atol=1e-6)
    np.testing.assert_allclose(np.cross(fwd, lat), up, atol=1e-6)  # up = fwd × lat
    assert np.all(lat[:, 2] == 0.0)  # the lateral axis stays horizontal
    th, ph = _ANGLES[:, 0].astype(np.float64), _ANGLES[:, 1].astype(np.float64)
    np.testing.assert_allclose(
        fwd, np.stack([np.cos(ph) * np.cos(th), np.cos(ph) * np.sin(th), np.sin(ph)], 1), atol=1e-6
    )
    assert np.array_equal(np.stack([fwd[0], lat[0], up[0]]), np.eye(3))  # (θ, φ) = (0, 0): the world axes


def test_to_body3_named_cases() -> None:
    c, s = math.cos(math.pi / 4), math.sin(math.pi / 4)
    cases = [  # (yaw, pitch, world vector, body (fwd, lat, up))
        (0.0, 0.0, (1.0, 2.0, 3.0), (1.0, 2.0, 3.0)),
        (math.pi / 2, 0.0, (1.0, 0.0, 0.0), (0.0, -1.0, 0.0)),
        (0.0, math.pi / 4, (0.0, 0.0, 1.0), (s, 0.0, c)),  # straight up, seen nose-up
        (0.0, math.pi / 4, (1.0, 0.0, 0.0), (c, 0.0, -s)),
        (0.0, PM, (0.0, 0.0, 1.0), (math.sin(PM), 0.0, math.cos(PM))),
        (0.0, -PM, (0.0, 0.0, 1.0), (-math.sin(PM), 0.0, math.cos(PM))),
    ]
    ang = np.array([[t, p] for t, p, _, _ in cases], np.float32)
    v = mx.array(np.array([w for _, _, w, _ in cases], np.float32))
    got = _np(ops.to_body3(v, *_trig(ang)))
    assert got.dtype == np.float32 and got.shape == (len(cases), 3)
    np.testing.assert_allclose(got, [b for _, _, _, b in cases], atol=1e-6)


def test_to_body3_projects_on_body_axes_and_inverts() -> None:
    rng = np.random.default_rng(2)
    v = rng.uniform(-6.0, 6.0, (len(_ANGLES), 3)).astype(np.float32)
    trig = _trig(_ANGLES)
    got = _np(ops.to_body3(mx.array(v), *trig)).astype(np.float64)
    fwd, lat, up = (_np(a).astype(np.float64) for a in ops.body_axes(*trig))
    v64 = v.astype(np.float64)
    want = np.stack([(v64 * fwd).sum(-1), (v64 * lat).sum(-1), (v64 * up).sum(-1)], -1)
    np.testing.assert_allclose(got, want, atol=1e-5)  # the same frame as body_axes (thrust and sensing agree)
    back = got[:, :1] * fwd + got[:, 1:2] * lat + got[:, 2:] * up  # rotation back to the world
    np.testing.assert_allclose(back, v64, atol=1e-5)
    np.testing.assert_allclose(np.linalg.norm(got, axis=1), np.linalg.norm(v64, axis=1), rtol=1e-6)


def test_to_body3_broadcasts_like_to_body() -> None:
    """(N,K,3) offsets with per-row angles, as the neighbour descriptor calls it."""
    v = mx.random.uniform(-1.0, 1.0, (5, 4, 3), key=mx.random.key(3))
    cos_h, sin_h, cos_p, sin_p = _trig(_ANGLES[:5])
    got = _np(ops.to_body3(v, cos_h[:, None], sin_h[:, None], cos_p[:, None], sin_p[:, None]))
    assert got.shape == (5, 4, 3)
    for k in range(4):
        row = _np(ops.to_body3(v[:, k], cos_h, sin_h, cos_p, sin_p))
        assert np.array_equal(got[:, k], row)


def test_to_body3_at_zero_pitch_is_to_body() -> None:
    """φ = 0 and vz = 0: (fwd, lat) equal the 2D rotation bit for bit and up is 0."""
    rng = np.random.default_rng(4)
    v2 = rng.uniform(-6.0, 6.0, (64, 2)).astype(np.float32)
    v3 = np.concatenate([v2, np.zeros((64, 1), np.float32)], axis=1)
    th = mx.array(rng.uniform(-math.pi, math.pi, 64).astype(np.float32))
    zero = mx.zeros((64,), dtype=mx.float32)
    got = _np(ops.to_body3(mx.array(v3), mx.cos(th), mx.sin(th), mx.cos(zero), mx.sin(zero)))
    assert np.array_equal(got[:, :2], _np(ops.to_body(mx.array(v2), mx.cos(th), mx.sin(th))))
    assert np.all(got[:, 2] == 0.0)


def test_2d_rotation_refuses_3d_vectors() -> None:
    """The 2D rotation would drop z silently; the 3D one would read past a 2D vector."""
    one = mx.ones((4,))
    with pytest.raises(AssertionError, match="to_body3"):
        ops.to_body(mx.zeros((4, 3)), one, one)
    with pytest.raises(AssertionError, match="2D vectors"):
        ops.to_body3(mx.zeros((4, 2)), one, one, one, one)


# --- min_image on z ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("size", [48.0, 128.0, 160.0])
def test_min_image_is_the_identity_below_half_the_period(size: float) -> None:
    """|d| < L/2 gives round(d/L) = ±0, so min_image(d) = d exactly: with 2·height ≤ L (§27.1) every z offset
    passes through unchanged, which is why 3D bonds and neighbours need no z-specific minimum image."""
    rng = np.random.default_rng(5)
    below = float(np.nextafter(np.float32(size / 2), np.float32(0)))
    d = np.concatenate([rng.uniform(-below, below, 4096), [0.0, -0.0, below, -below, 1e-30, -1e-30]]).astype(
        np.float32
    )
    got = _np(ops.min_image(mx.array(d), size))
    assert np.array_equal(got, d)
