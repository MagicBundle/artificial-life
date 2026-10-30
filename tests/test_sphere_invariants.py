"""The invariants of a sphere state (DESIGN §46.2; sphere series S2): ``device_check`` bits and
``full_check``.

A meta-test in the manner of ``test_invariants_3d``: the healthy state passes, and each way a row can leave
the
sphere's contract (radius outside the shell, a heading that is not a unit vector or not tangent, a stale
derived heading, a pitch out of range, a patch that left the unit sphere, a velocity pressing into the lid, a
non-finite value) is reported by name, on the host and in the compiled check.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from alife import invariants, layout
from alife.config import Config, dims, load_config
from alife.state import Diag, StateS, init_state, state_from_arrays, state_to_arrays, to_numpy
from sphere_rotation import random_rotation, rotate_state


def cfg_of() -> Config:
    return load_config(preset="tiny3s", overrides=["brain.weight_dtype=float32"])


def with_field(s: StateS, name: str, fn) -> StateS:
    """``s`` with ``name`` replaced by ``fn(numpy copy)``."""
    a = to_numpy(getattr(s, name)).copy()
    return s._replace(**{name: mx.array(fn(a))})


def bits(s: StateS, cfg: Config) -> int:
    z = mx.array(0, dtype=mx.int64)
    ledger = mx.zeros((16,), dtype=mx.int64)
    b = invariants.device_check(s, ledger, s.total, mx.array(0), mx.array(0), mx.array(0), mx.array(0), cfg)
    return int(to_numpy(b)) & ~int(Diag.LEDGER) if z is not None else 0


def test_a_healthy_state_passes_both_checks() -> None:
    cfg = cfg_of()
    s = init_state(cfg)
    assert invariants.full_check(s, cfg) == []
    assert bits(s, cfg) == 0


@pytest.mark.parametrize(
    ("name", "fn", "msg", "bit"),
    [
        (
            "pos",
            lambda a: np.where(np.arange(len(a))[:, None] == 3, a * 1.5, a),
            "|pos| outside [r_lo, r_hi]",
            Diag.POS_RANGE,
        ),
        (
            "pos",
            lambda a: np.where(np.arange(len(a))[:, None] == 3, a * 0.6, a),
            "|pos| outside [r_lo, r_hi]",
            Diag.POS_RANGE,
        ),
        (
            "hdg",
            lambda a: np.where(np.arange(len(a))[:, None] == 4, a * 1.01, a),
            "hdg: not a unit vector",
            Diag.POS_RANGE,
        ),
        (
            "hdg",
            lambda a: np.where(np.arange(len(a))[:, None] == 4, a * 0.999, a),
            "hdg: not a unit vector",
            Diag.POS_RANGE,
        ),
        ("pitch", lambda a: np.where(np.arange(len(a)) == 5, 2.0, a), "pitch: outside", Diag.POS_RANGE),
        (
            "buoy_cap",
            lambda a: np.where(np.arange(len(a)) == 5, 1.5, a),
            "buoy_cap: outside [0, 1]",
            Diag.NONFINITE & 0,
        ),
        ("patch_pos", lambda a: a * 1.01, "patch_pos: not a unit vector", Diag.NONFINITE & 0),
        (
            "hdg",
            lambda a: np.where(np.arange(len(a))[:, None] == 6, np.nan, a),
            "hdg: not a unit vector",
            Diag.NONFINITE,
        ),
        (
            "pos",
            lambda a: np.where(np.arange(len(a))[:, None] == 7, np.inf, a),
            "|pos| outside",
            Diag.NONFINITE,
        ),
    ],
)
def test_each_violation_is_named(name, fn, msg, bit) -> None:
    cfg = cfg_of()
    s = with_field(init_state(cfg), name, fn)
    problems = invariants.full_check(s, cfg)
    assert any(msg in p for p in problems), problems
    if bit:  # the compiled check covers the position, the heading, the pitch and the finiteness
        assert bits(s, cfg) & int(bit), (name, bits(s, cfg))


def test_a_heading_that_is_not_tangent_is_reported() -> None:
    cfg = cfg_of()
    s = init_state(cfg)
    pos = to_numpy(s.pos)
    up = pos / np.linalg.norm(pos, axis=1, keepdims=True)
    # tilt the heading of row 2 by 1e-3 rad towards the up direction and renormalise: unit, but not tangent
    hdg = to_numpy(s.hdg).copy()
    hdg[2] = (hdg[2] + 1e-3 * up[2]) / np.linalg.norm(hdg[2] + 1e-3 * up[2])
    bad = s._replace(hdg=mx.array(hdg.astype(np.float32)))
    assert any("not tangent" in p for p in invariants.full_check(bad, cfg))
    assert bits(bad, cfg) & int(Diag.POS_RANGE)


def test_the_derived_heading_is_compared_circularly_and_not_near_a_pole() -> None:
    cfg = cfg_of()
    s = init_state(cfg)
    h = to_numpy(s.heading).copy()
    h[1] = h[1] + 0.01  # a stale derived heading: 1e-2 off the bearing of hdg
    assert any(
        "heading: not the bearing of hdg" in p
        for p in invariants.full_check(s._replace(heading=mx.array(h)), cfg)
    )
    # +π and −π are the same bearing: a heading stored as the other branch passes
    pos = to_numpy(s.pos).astype(np.float64)
    up = pos / np.linalg.norm(pos, axis=1, keepdims=True)
    east = np.cross([0.0, 0.0, 1.0], up)
    east /= np.linalg.norm(east, axis=1, keepdims=True) + 1e-300
    north = np.cross(up, east)
    south_facing = (-north).astype(np.float32)  # the bearing of due south is ±π
    s2 = s._replace(hdg=mx.array(south_facing), heading=mx.full((dims(cfg).n,), -math.pi, dtype=mx.float32))
    ok = [p for p in invariants.full_check(s2, cfg) if "heading" in p and "bearing" in p]
    assert ok == []
    s3 = s2._replace(
        heading=mx.full((dims(cfg).n,), float(np.float32(math.pi) * (1 - 1e-7)), dtype=mx.float32)
    )
    assert [p for p in invariants.full_check(s3, cfg) if "bearing" in p] == []


def test_the_checks_do_not_depend_on_the_planet_axis() -> None:
    """The consumer-guard test of `invariants`: a rotated healthy state passes the same checks (a raw
    ``pos[:, 2]``
    in a range test would fail it at once)."""
    cfg = cfg_of()
    s = init_state(cfg)
    r = random_rotation(3)
    q = rotate_state(s, r)
    assert invariants.full_check(q, cfg) == []
    assert bits(q, cfg) == 0
    # and the same corruption is reported in the rotated frame
    bad = with_field(q, "pos", lambda a: np.where(np.arange(len(a))[:, None] == 3, a * 1.5, a))
    assert any("|pos| outside" in p for p in invariants.full_check(bad, cfg))


def test_layout_and_the_state_round_trip() -> None:
    cfg = cfg_of()
    s = init_state(cfg)
    assert type(s) is StateS and s.hdg.shape == (dims(cfg).n, 3)
    back = state_from_arrays({k: mx.array(to_numpy(v)) for k, v in state_to_arrays(s).items()}, cfg)
    assert isinstance(back, StateS) and invariants.full_check(back, cfg) == []
    assert layout.PITCH_MAX > 1.0


# --- the radial bookkeeping of a step, from the previous positions (review M2) ----------------------------


def radial_problems(prev: StateS, s: StateS, cfg: Config) -> list[str]:
    r = invariants._Report()
    a = {k: to_numpy(getattr(s, k)) for k in ("alive", "uid", "pos", "vel")}
    p = {k: to_numpy(getattr(prev, k)) for k in ("alive", "uid", "pos")}
    invariants._check_radial_sph(r, a, p, cfg)
    return r.problems


def test_the_radial_check_accepts_every_step_of_a_live_run_and_names_a_clamp_that_keeps_vr() -> None:
    from alife.step import step

    cfg = cfg_of()
    d = dims(cfg)
    s = init_state(cfg)
    clamped = 0
    for _ in range(120):
        prev = s
        s, out = step(s, cfg)
        mx.eval(s)
        assert radial_problems(prev, s, cfg) == []
        assert not [p for p in invariants.full_check(s, cfg, prev, out) if p.startswith(("radius", "clamp"))]
        clamped += int(to_numpy(out.counters)[-3] + to_numpy(out.counters)[-2])
    assert clamped > 0  # the floor was met: the clamp branch of the check ran on real rows

    # a synthetic step: row 0 rests on the floor and is pushed down; the engine zeroes the radial velocity
    prev = init_state(cfg)
    up = to_numpy(prev.pos)[0] / np.linalg.norm(to_numpy(prev.pos)[0])
    pos = to_numpy(prev.pos).copy()
    pos[0] = (up * float(d.r_lo)).astype(np.float32)
    prev = prev._replace(pos=mx.array(pos))
    ok = prev._replace(vel=mx.array(np.zeros_like(pos)))  # clamped: v_r zeroed, the radius stays on the floor
    assert radial_problems(prev, ok, cfg) == []
    vel = np.zeros_like(pos)
    vel[0] = (-0.01 * up).astype(
        np.float32
    )  # the clamp kept v_r = -0.01 while the radius stayed on the floor
    bad = radial_problems(prev, prev._replace(vel=mx.array(vel)), cfg)
    assert any("clamp: a row on a bound kept its radial velocity" in m for m in bad), bad
    # an unclamped landing: it came down 0.01 and stopped within 1e-6 of the floor: v_r kept, accepted
    high = pos.copy()
    high[0] = (up * (float(d.r_lo) + 0.01)).astype(np.float32)
    landed = prev._replace(pos=mx.array(pos))  # now on the floor
    came_from = prev._replace(pos=mx.array(high))
    assert radial_problems(came_from, landed._replace(vel=mx.array(vel)), cfg) == []
    # a radius that is not the integrated one (a free row teleported 0.5 outward)
    moved = pos.copy()
    moved[1] = (to_numpy(prev.pos)[1] * (1 + 0.5 / np.linalg.norm(to_numpy(prev.pos)[1]))).astype(np.float32)
    bad = radial_problems(prev, prev._replace(pos=mx.array(moved), vel=mx.array(np.zeros_like(pos))), cfg)
    assert any("radius: not the integrated one" in m for m in bad), bad


def test_the_heading_range_allows_the_device_wrap_and_nothing_more() -> None:
    """S7a, from a 3 000-step run of 64k3s (1 row in 65 536): ``sphere.bearing`` wraps ``b >= π − 1e-6`` to
    ``b − 2π``, which is down to 1e-6 (and a float32 ulp) below −π. That is not a violation; 1e-3 below is."""
    cfg = cfg_of()
    s = init_state(cfg)
    n = dims(cfg).n

    def range_problems(value: float) -> list[str]:
        h = np.zeros(n, np.float32)
        h[0] = value
        return [p for p in invariants.full_check(s._replace(heading=mx.array(h)), cfg) if "[−π, π)" in p]

    assert range_problems(float(np.float32(-math.pi - 1e-6))) == []
    assert range_problems(float(np.float32(-3.1415935))) == []  # the value seen: 8.6e-7 below −π
    assert range_problems(-math.pi - 1e-3) != [] and range_problems(math.pi) != []


def test_an_unclamped_landing_just_inside_the_tolerance_of_a_bound_is_not_a_clamp_that_kept_vr() -> None:
    """S7a, from the same run: a row lands 0.95·tol above the floor having commanded 1.05·tol above it
    (|r − rr| = 0.1·tol): it was not clamped, so keeping its radial velocity is right."""
    cfg = cfg_of()
    d = dims(cfg)
    tol = 4.0 * invariants.radius_tol(cfg)
    s = init_state(cfg)
    pos = to_numpy(s.pos).copy()
    up = pos[0] / np.linalg.norm(pos[0])
    r_old = float(d.r_lo) + 0.01
    r_new = float(d.r_lo) + 0.95 * tol
    rr = float(d.r_lo) + 1.05 * tol
    prev_pos = pos.copy()
    prev_pos[0] = (up * r_old).astype(np.float32)
    new_pos = pos.copy()
    new_pos[0] = (up * r_new).astype(np.float32)
    vel = np.zeros_like(pos)
    vel[0] = (up * (rr - r_old)).astype(np.float32)
    prev = s._replace(pos=mx.array(prev_pos))
    cur = s._replace(pos=mx.array(new_pos), vel=mx.array(vel))
    assert radial_problems(prev, cur, cfg) == []
    # a row pressed 0.01 into the floor, clamped to 0.5 tol above it but keeping v_r = -0.02, is the bug named
    far = new_pos.copy()
    far[0] = (up * (float(d.r_lo) + 0.5 * tol)).astype(np.float32)
    vel2 = vel.copy()
    vel2[0] = (up * (-0.02)).astype(np.float32)
    bad = radial_problems(prev, s._replace(pos=mx.array(far), vel=mx.array(vel2)), cfg)
    assert any("clamp: a row on a bound kept its radial velocity" in m for m in bad), bad
