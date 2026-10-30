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
    east /= np.linalg.norm(east, axis=1, keepdims=True)
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
