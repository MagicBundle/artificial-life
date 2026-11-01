"""Motion on the sphere: the exponential-map step and the transport of the heading (``sphere.move``;
DESIGN §46.2; sphere series S0). The probes of the spec as tests: float32 against a float64 reference with the
exact trigonometric map, invariants over thousands of steps, the necessity of the renormalisation, the poles,
the clamps, the holonomy of a closed geodesic triangle (the area), the flat limit, the bearing convention.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from alife import sphere

R_LO, R_HI = 37.0, 129.0


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def ref_step(pos, hdg, v, turn, r_lo=R_LO, r_hi=R_HI):
    """Float64 reference: the EXACT great-circle map (sin and cos of the angle) and Rodrigues rotation about
    the axis ``u × u'``; the yaw turn about the new up. No series, no renormalisation."""
    r = np.linalg.norm(pos, axis=-1, keepdims=True)
    u = pos / r
    vr = (v * u).sum(-1, keepdims=True)
    w = (v - vr * u) / r
    sig = np.linalg.norm(w, axis=-1, keepdims=True)
    what = np.where(sig > 0, w / np.where(sig > 0, sig, 1), 0.0)
    u2 = np.cos(sig) * u + np.sin(sig) * what
    r2 = np.clip(r + vr, r_lo, r_hi)
    k = np.cross(u, what)  # unit axis where sig > 0, else zero
    h1 = (
        hdg * np.cos(sig)
        + np.cross(k, hdg) * np.sin(sig)
        + k * (k * hdg).sum(-1, keepdims=True) * (1 - np.cos(sig))
    )
    h2 = np.cos(turn)[:, None] * h1 + np.sin(turn)[:, None] * np.cross(u2, h1)
    return r2 * u2, h2


def step_f32_numpy(pos, hdg, v, turn, renorm: bool):
    """The formulas of ``sphere.move`` in numpy float32, the two normalisations optional (the probe)."""
    f = np.float32
    r = np.linalg.norm(pos, axis=-1, keepdims=True).astype(f)
    u = pos / r
    vr = (v * u).sum(-1, keepdims=True)
    w = (v - vr * u) / r
    s2 = (w * w).sum(-1, keepdims=True)
    c = f(1) - s2 / f(2) + s2 * s2 / f(24)
    s = f(1) - s2 / f(6) + s2 * s2 / f(120)
    u2 = c * u + s * w
    if renorm:
        u2 = u2 / np.sqrt((u2 * u2).sum(-1, keepdims=True))
    k = s * np.cross(u, w)
    h1 = hdg + np.cross(k, hdg) + np.cross(k, np.cross(k, hdg)) / (f(1) + c)
    h2 = np.cos(turn)[:, None].astype(f) * h1 + np.sin(turn)[:, None].astype(f) * np.cross(u2, h1)
    if renorm:
        h2 = h2 - (h2 * u2).sum(-1, keepdims=True) * u2
        h2 = h2 / np.sqrt((h2 * h2).sum(-1, keepdims=True))
    return (r + vr) * u2, h2


def _scene(n: int, seed: int, polar: int = 0):
    rng = np.random.default_rng(seed)
    u = _unit(rng.normal(size=(n, 3)))
    if polar:
        u[:polar] = _unit(np.array([0.0, 0.0, 1.0]) + 1e-3 * rng.normal(size=(polar, 3)))
    h = _unit(np.cross(u, rng.normal(size=(n, 3))))
    pos = (u * rng.uniform(R_LO + 2, R_HI - 2, size=(n, 1))).astype(np.float32)
    return rng, pos, h.astype(np.float32)


def _inputs(rng, steps: int, n: int):
    v = rng.normal(size=(steps, n, 3)).astype(np.float32)
    v *= (0.5 * rng.uniform(0.2, 1.0, (steps, n, 1)) / np.linalg.norm(v, axis=2, keepdims=True)).astype(
        np.float32
    )
    return v, rng.uniform(-0.1, 0.1, (steps, n)).astype(np.float32)


def _move(pos, hdg, v, turn, r_lo=R_LO, r_hi=R_HI):
    out = sphere.move(mx.array(pos), mx.array(hdg), mx.array(v), mx.array(turn), r_lo, r_hi)
    return tuple(np.array(a) for a in out)


def test_one_step_equals_the_exact_float64_map() -> None:
    rng, pos, hdg = _scene(20_000, 1, polar=2000)
    v, turn = _inputs(rng, 1, 20_000)
    p2, h2, *_ = _move(pos, hdg, v[0], turn[0])
    rp, rh = ref_step(
        pos.astype(np.float64), hdg.astype(np.float64), v[0].astype(np.float64), turn[0].astype(np.float64)
    )
    assert np.abs(p2 - rp).max() < 2e-5
    assert np.abs(h2 - rh).max() < 2e-6
    assert np.abs(np.linalg.norm(h2, axis=1) - 1).max() < 4e-7
    up = p2 / np.linalg.norm(p2, axis=1)[:, None]
    assert np.abs((h2 * up).sum(1)).max() < 4e-7  # tangent
    assert np.all(np.isfinite(p2)) and np.all(np.isfinite(h2))


def test_the_series_has_no_division_by_the_tangent_speed() -> None:
    """vt = 0 is exact: a purely radial step keeps the direction and the heading (up to the turn)."""
    pos = np.array([[0, 0, 60.0], [60.0, 0, 0], [0, -50.0, 0], [30.0, 30.0, 30.0]], dtype=np.float32)
    hdg = np.array([[1, 0, 0], [0, 0, 1], [0, 0, -1], [0, 0, 0]], dtype=np.float32)
    hdg[3] = np.cross(pos[3] / np.linalg.norm(pos[3]), [1.0, 0.0, 0.0])
    hdg[3] /= np.linalg.norm(hdg[3])
    v = (pos / np.linalg.norm(pos, axis=1)[:, None] * 0.3).astype(np.float32)
    p2, h2, vel, _, _ = _move(pos, hdg, v, np.zeros(4, np.float32))
    assert np.abs(p2 - pos * (1 + 0.3 / np.linalg.norm(pos, axis=1)[:, None])).max() < 5e-6
    assert np.abs(h2 - hdg).max() < 2e-7 and np.array_equal(vel, v)


def test_a_step_through_the_pole_is_regular() -> None:
    pos = np.array([[0, 0, 50.0], [0, 0, -50.0], [1e-4, 0, 50.0]], dtype=np.float32)
    hdg = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0]], dtype=np.float32)
    v = np.array([[0.4, 0, 0], [0, 0.4, 0], [-0.4, 0.0, 0]], dtype=np.float32)
    p2, h2, *_ = _move(pos, hdg, v, np.zeros(3, np.float32))
    assert np.all(np.isfinite(p2)) and np.all(np.isfinite(h2))
    assert np.abs(np.linalg.norm(p2, axis=1) - 50.0).max() < 1e-5
    sig = 0.4 / 50.0  # moving along the heading: the heading stays the direction of travel
    assert np.abs(h2[0] - [math.cos(sig), 0, -math.sin(sig)]).max() < 2e-6
    assert p2[0, 0] == pytest.approx(50.0 * math.sin(0.4 / 50.0), rel=1e-5)
    assert p2[2, 0] < 0  # crossed the pole


def test_radial_clamps_at_floor_and_lid_zero_the_outward_velocity() -> None:
    pos = np.array(
        [[R_LO + 0.05, 0, 0], [0, R_HI - 0.05, 0], [60.0, 0, 0], [R_LO + 0.05, 0, 0]], dtype=np.float32
    )
    hdg = np.array([[0, 1, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0]], dtype=np.float32)
    v = np.array([[-0.2, 0.1, 0], [0, 0.2, 0.1], [0.1, 0.1, 0], [0.2, 0.0, 0.0]], dtype=np.float32)
    p2, _, vel, at_floor, at_lid = _move(pos, hdg, v, np.zeros(4, np.float32))
    r2 = np.linalg.norm(p2, axis=1)
    assert at_floor.tolist() == [True, False, False, False] and at_lid.tolist() == [False, True, False, False]
    assert r2[0] == pytest.approx(R_LO, abs=1e-5) and r2[1] == pytest.approx(R_HI, abs=2e-5)
    assert abs(vel[0, 0]) < 1e-7 and abs(vel[1, 1]) < 1e-7  # the radial part is gone, the tangent part kept
    assert vel[0, 1] == pytest.approx(0.1) and vel[1, 2] == pytest.approx(0.1)
    assert np.array_equal(vel[2:], v[2:])  # unclamped rows keep v
    assert r2[3] == pytest.approx(R_LO + 0.25, abs=1e-5)  # moving up from the floor is not a clamp


def test_float32_invariants_hold_over_thousands_of_steps_and_match_float64() -> None:
    steps, n = 1500, 200
    rng, pos0, hdg0 = _scene(n, 4, polar=20)
    v, turn = _inputs(rng, steps, n)
    consts = mx.array(
        [R_LO, R_HI], dtype=mx.float32
    )  # runtime inputs: nothing printable for the compile cache

    @mx.compile
    def advance(pos, hdg, vi, ti, c):
        return sphere.move(pos, hdg, vi, ti, c[0], c[1])[:2]

    pos, hdg = mx.array(pos0), mx.array(hdg0)
    rp, rh = pos0.astype(np.float64), hdg0.astype(np.float64)
    for i in range(steps):
        pos, hdg = advance(pos, hdg, mx.array(v[i]), mx.array(turn[i]), consts)
        if i % 50 == 49:
            mx.eval(pos, hdg)
        rp, rh = ref_step(rp, rh, v[i].astype(np.float64), turn[i].astype(np.float64))
    p, h = np.array(pos), np.array(hdg).astype(np.float64)
    up = p / np.linalg.norm(p, axis=1)[:, None]
    assert np.abs(np.linalg.norm(h, axis=1) - 1).max() < 4e-7  # the spec's bounds for the state invariants
    assert np.abs((h * up).sum(1)).max() < 4e-7
    assert np.linalg.norm(p, axis=1).max() <= R_HI + 6e-5 and np.linalg.norm(p, axis=1).min() >= R_LO - 6e-5
    # against the float64 run on the same inputs: a random walk of rounding, far below the body radius
    assert np.linalg.norm(p - rp, axis=1).max() < 0.01
    ang = np.arccos(np.clip((h * rh).sum(1), -1, 1))
    assert ang.max() < 5e-4


def test_without_renormalisation_the_heading_drifts() -> None:
    """The reason the renormalisation is mandatory (spec probe: 0.11 after 3·10^5 steps; here 3 000)."""
    steps, n = 3000, 200
    rng, pos0, hdg0 = _scene(n, 5, polar=20)
    v, turn = _inputs(rng, steps, n)
    out = {}
    for renorm in (True, False):
        p, h = pos0.copy(), hdg0.copy()
        for i in range(steps):
            p, h = step_f32_numpy(p, h, v[i], turn[i], renorm)
            rr = np.linalg.norm(p, axis=1)[:, None]
            p = (p * np.clip(rr, R_LO, R_HI) / rr).astype(np.float32)
        out[renorm] = np.abs(np.linalg.norm(h, axis=1) - 1).max()
    assert out[True] < 4e-7
    assert out[False] > 20 * out[True] and out[False] > 5e-6


def test_holonomy_of_the_octant_triangle_is_its_area() -> None:
    """Pole -> equator at longitude 0 -> equator at longitude 90 -> pole, in steps of 0.25 units on the
    smallest planet (radius 45): transporting the heading round the loop turns it by the area, π/2, in
    float64 (the reference) and in float32 (the device step)."""
    r = 45.0
    corners = [np.array([0, 0, 1.0]), np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])]
    ang = {}
    for name in ("f64", "f32"):
        pos = (r * corners[0])[None].astype(np.float32 if name == "f32" else np.float64)
        hdg = np.array([[1.0, 0.0, 0.0]], dtype=pos.dtype)
        for target in corners[1:]:
            n_steps = math.ceil((math.pi / 2) * r / 0.25)
            for _ in range(n_steps):
                u = pos[0].astype(np.float64) / np.linalg.norm(pos[0].astype(np.float64))
                t = target - u * (u @ target)
                t /= np.linalg.norm(t)
                v = (t * ((math.pi / 2) * r / n_steps))[None].astype(
                    pos.dtype
                )  # tangent step toward the target
                if name == "f32":
                    pos, hdg, *_ = _move(pos, hdg, v, np.zeros(1, np.float32), 30.0, 60.0)
                else:
                    pos, hdg = ref_step(pos, hdg, v, np.zeros(1), 30.0, 60.0)
        h = hdg[0].astype(np.float64)
        ang[name] = math.atan2(np.cross([1.0, 0, 0], h)[2], h[0])
    assert (
        abs(abs(ang["f64"]) - math.pi / 2) < 1e-6
    )  # (the step is not exactly the arc: the walk is a polygon)
    assert abs(abs(ang["f32"]) - math.pi / 2) < 2e-4
    assert abs(ang["f32"] - ang["f64"]) < 2e-4


def test_flat_limit_reproduces_the_planar_step() -> None:
    """On a huge planet a step is the planar one: position by v, heading rotated by the turn about the up
    (the box's (cos θ, sin θ) with ẑ up), to 1 % of the step length (float32 resolves 1.2e-4 at 2 000)."""
    n = 500
    rng = np.random.default_rng(6)
    pos = np.tile([[0.0, 0.0, 2000.0]], (n, 1)).astype(np.float32)
    hdg = np.tile([[1.0, 0.0, 0.0]], (n, 1)).astype(np.float32)
    v = np.concatenate([rng.uniform(-0.3, 0.3, (n, 2)), np.zeros((n, 1))], axis=1).astype(np.float32)
    theta = rng.uniform(-0.3, 0.3, n).astype(np.float32)
    p2, h2, *_ = _move(pos, hdg, v, theta, 1900.0, 2100.0)
    step = np.linalg.norm(v, axis=1)
    assert (np.linalg.norm(p2 - (pos + v), axis=1) <= 0.01 * step + 1.3e-4).all()
    assert np.abs(h2[:, 0] - np.cos(theta)).max() < 1e-4 and np.abs(h2[:, 1] - np.sin(theta)).max() < 1e-4


def test_bearing_is_the_compass_angle_from_local_north() -> None:
    pos = np.array([[60.0, 0, 0]] * 4 + [[0, 0, 60.0], [0, 0, -60.0]], dtype=np.float32)
    hdg = np.array([[0, 0, 1], [0, 1, 0], [0, -1, 0], [0, 0, -1], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    b = np.array(sphere.bearing(mx.array(hdg), mx.array(pos)))
    # at x̂: east = ẑ × x̂ = ŷ, north = x̂ × ŷ = ẑ ; the range is [−π, π) and +π is −π
    assert b[:3] == pytest.approx([0.0, math.pi / 2, -math.pi / 2], abs=1e-6)
    assert b[3] == pytest.approx(-math.pi, abs=2e-6)
    assert np.all(np.isfinite(b[4:]))  # at the poles east is x̂ by convention
