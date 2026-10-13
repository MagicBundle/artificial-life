"""§27.3 3D physics: gravity and buoyancy as body forces, the floor and surface clamps, grounding, grip on
the floor only, 3D repulsion and springs, the pitch clip, the level floor as the 2D physics, and compile."""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from alife import layout, physics
from alife.actions import Actions, Motion3
from alife.config import Config, dims
from alife.layout import MSG, NCH, B
from alife.spatial import Neighbors
from helpers import ON_GPU, as_3d

f32 = np.float32
PM32 = f32(layout.PITCH_MAX)
G, BMAX = 0.03, 0.05


@pytest.fixture(scope="module")
def cfg3(micro_cfg: Config) -> Config:
    """``micro`` in 3D (L 48, height 24) with gravity, a buoyancy actuator and a 0.1 grounding band."""
    return as_3d(micro_cfg, physics={"gravity": G, "buoyancy_max": BMAX, "floor_eps": 0.1})


@pytest.fixture(scope="module")
def level3(micro_cfg: Config) -> Config:
    """``micro`` in 3D without gravity or buoyancy."""
    return as_3d(micro_cfg)


def _actions(n: int, **kw: np.ndarray | float) -> Actions:
    """All-zero (inactive) actions with the given fields overridden."""
    z = np.zeros(n, np.float32)
    base = {
        "thrust": z, "turn": z, "msg": np.zeros((n, MSG), np.float32), "emit": np.zeros((n, NCH), np.float32),
        "harvest": z, "grip": z, "contract": z, "bond_want": np.zeros(n, bool),
        "release": np.zeros((n, B), bool), "share": np.zeros((n, B), np.float32), "attack": np.zeros(n, bool),
        "defend": z, "reproduce": np.zeros(n, bool), "child_bond": np.zeros(n, bool),
        "bond_k": np.full(n, -1, np.int32), "bond_target": np.full(n, -1, np.int32),
        "attack_k": np.full(n, -1, np.int32), "attack_target": np.full(n, -1, np.int32),
    }  # fmt: skip
    for k, v in kw.items():
        base[k] = np.broadcast_to(np.asarray(v, base[k].dtype), base[k].shape).copy()
    return Actions(**{k: mx.array(v) for k, v in base.items()})


def _motion(n: int, pitch: np.ndarray | float = 0.0, buoy: np.ndarray | float = 0.0) -> Motion3:
    def col(v: np.ndarray | float) -> mx.array:
        return mx.array(np.broadcast_to(f32(v), n).copy())

    return Motion3(col(pitch), col(buoy))


def _min_image(d: np.ndarray, L: float) -> np.ndarray:
    return (d - f32(L) * np.round(d / f32(L))).astype(np.float32)


def _neighbors(pos: np.ndarray, alive: np.ndarray, cfg: Config) -> Neighbors:
    """Brute-force K nearest alive agents within R in float32, for (N,2) or (N,3) positions (the §4 contract
    without cell truncation; ``min_image`` is the identity on z, §27.2)."""
    n, k, R = pos.shape[0], dims(cfg).k, cfg.sensing.radius
    off = _min_image(pos[None, :, :] - pos[:, None, :], cfg.world.size)  # (N,N,nd): me → j
    d2 = (off * off).sum(-1)
    ok = alive[:, None] & alive[None, :] & ~np.eye(n, dtype=bool) & (d2 < f32(R * R))
    d2 = np.where(ok, d2, np.inf)
    order = np.argsort(d2, axis=1, kind="stable")[:, :k]
    pd2 = np.take_along_axis(d2, order, 1)
    valid = np.isfinite(pd2)
    rows = np.arange(n)[:, None]
    idx = np.where(valid, order, rows).astype(np.int32)
    offk = np.where(valid[..., None], off[rows, order], 0).astype(np.float32)
    dist = np.where(valid, np.sqrt(np.where(valid, pd2, 0)), R).astype(np.float32)
    zero = mx.array(0, mx.int32)
    return Neighbors(mx.array(idx), mx.array(valid), mx.array(offk), mx.array(dist), zero, zero)


def _scene(cfg: Config, pts: list[tuple[float, ...]], bonds_: list[tuple[int, int]] = ()):
    """``(pos, alive, partner)`` with ``pts`` in the first rows; dead rows on the floor at (0, 0, z_lo)."""
    d = dims(cfg)
    pos = np.zeros((d.n, d.nd), np.float32)
    if d.d3:
        pos[:, 2] = d.z_lo
    pos[: len(pts)] = pts
    alive = np.arange(d.n) < len(pts)
    partner = np.full((d.n, B), -1, np.int32)
    for i, j in bonds_:
        si, sj = int(np.argmax(partner[i] < 0)), int(np.argmax(partner[j] < 0))
        partner[i, si], partner[j, sj] = j, i
    return pos, alive, partner


def _forces3(cfg, pos, heading, pitch, alive, partner, act, mo, cap) -> np.ndarray:
    nbr = _neighbors(pos, alive, cfg)
    f = physics.forces3(
        mx.array(pos), mx.array(heading), mx.array(pitch), act, mo, mx.array(cap), nbr, mx.array(alive),
        mx.array(partner), cfg,
    )  # fmt: skip
    return np.array(f)


def _integrate3(cfg, pos, force, *, grip=None, turn=None, pitch=None, dpitch=None, heading=None, alive=None):
    n = pos.shape[0]
    zf = np.zeros(n, np.float32)
    alive = np.ones(n, bool) if alive is None else alive
    out = physics.integrate3(
        mx.array(pos), mx.array(zf if heading is None else heading), mx.array(zf if pitch is None else pitch),
        mx.array(force.astype(np.float32)), mx.array(zf if grip is None else grip),
        mx.array(zf if turn is None else turn), mx.array(zf if dpitch is None else dpitch),
        physics.grounded(mx.array(pos), cfg), mx.array(alive), cfg,
    )  # fmt: skip
    return physics.Integrated3(*(np.array(a) for a in out))


# --- gravity, buoyancy and the clamps ----------------------------------------------------------------------


def test_sinking_and_rising_speed_is_net_force_over_drag(cfg3: Config) -> None:
    """Overdamped: an isolated agent sinks at g/γ per step, or rises at (b − g)/γ with buoyancy b."""
    d = dims(cfg3)
    caps = np.array([0.0, 1.0, 0.5], f32)
    buoy = np.array([0.0, 1.0, 0.3], f32)
    pos, alive, partner = _scene(cfg3, [(10.0, 10.0, 12.0), (30.0, 10.0, 12.0), (10.0, 30.0, 12.0)])
    n = d.n
    cap = np.zeros(n, f32)
    cap[:3] = caps
    b = np.zeros(n, f32)
    b[:3] = buoy
    lift = f32(BMAX) * caps * buoy - f32(G)
    act, mo, zf = _actions(n), _motion(n, buoy=b), np.zeros(n, f32)
    z0 = pos[:3, 2].astype(np.float64)
    for _ in range(20):
        f = _forces3(cfg3, pos, zf, zf, alive, partner, act, mo, cap)
        assert np.array_equal(f[:3, 2], lift) and not f[:3, :2].any() and not f[3:].any()
        out = _integrate3(cfg3, pos, f, alive=alive)
        assert np.array_equal(out.vel[:3, 2], lift / f32(cfg3.physics.drag))
        pos = out.pos
    assert np.allclose(pos[:3, 2], z0 + 20 * lift / cfg3.physics.drag, atol=1e-5)
    assert lift[0] < 0 < lift[1]  # sinks without buoyancy, rises at full cap and output


def test_floor_and_surface_clamps(cfg3: Config) -> None:
    d, L = dims(cfg3), cfg3.world.size
    pts = [
        (5.0, 5.0, d.z_lo + 0.2),  # 0: falls 0.5: lands on the floor, not 0.2 above it
        (5.0, 9.0, d.z_hi - 0.1),  # 1: rises 0.3: stops at the surface
        (9.0, 5.0, 12.0),  # 2: falls 0.4 in mid-water: not clamped
        (9.0, 9.0, d.z_lo),  # 3: resting on the floor without force: not clamped
        (13.0, 5.0, d.z_lo),  # 4: on the floor, moving up: not clamped
        (L - 0.1, 13.0, d.z_lo + 0.1),  # 5: falls and crosses the x seam: clamped and wrapped
        (13.0, 13.0, d.z_lo + 0.3),  # 6: dead, with a downward force: untouched, not counted
        (17.0, 5.0, d.z_hi),  # 7: at the surface, pushed down: leaves it
    ]
    pos, alive, _ = _scene(cfg3, pts)
    alive[6] = False
    force = np.zeros(pos.shape, np.float32)
    force[0] = (0.0, 0.0, -0.5)
    force[1] = (0.1, 0.0, 0.3)
    force[2] = (0.0, 0.0, -0.4)
    force[4] = (0.0, 0.0, 0.2)
    force[5] = (0.2, 0.0, -0.3)
    force[6] = (0.0, 0.0, -0.5)
    force[7] = (0.0, 0.0, -0.25)
    out = _integrate3(cfg3, pos, force, alive=alive)
    p, v = out.pos, out.vel
    assert p[0, 2] == f32(d.z_lo) and v[0, 2] == 0.0  # moved to the floor, stored v_z zeroed
    assert p[1, 2] == f32(d.z_hi) and v[1, 2] == 0.0 and np.isclose(p[1, 0], 5.1) and v[1, 0] == f32(0.1)
    assert np.isclose(p[2, 2], 11.6) and v[2, 2] == f32(-0.4)
    assert p[3, 2] == f32(d.z_lo) and not v[3].any()
    assert np.isclose(p[4, 2], d.z_lo + 0.2) and v[4, 2] == f32(0.2)
    assert p[5, 2] == f32(d.z_lo) and np.isclose(p[5, 0], 0.1, atol=1e-5) and v[5, 2] == 0.0
    assert v[5, 0] == f32(0.2)  # the horizontal part of a clamped row is kept
    assert np.array_equal(p[6], pos[6]) and not v[6].any()
    assert np.isclose(p[7, 2], d.z_hi - 0.25) and v[7, 2] == f32(-0.25)
    assert (out.z_floor.dtype, out.z_surface.dtype) == (np.int32, np.int32)
    assert int(out.z_floor) == 2 and int(out.z_surface) == 1
    assert np.all((p[:, :2] >= 0) & (p[:, :2] < L))
    assert np.all((p[:, 2] >= f32(d.z_lo)) & (p[:, 2] <= f32(d.z_hi)))


def test_a_falling_agent_lands_at_r_and_rests_there(cfg3: Config) -> None:
    """Under gravity an agent dropped in mid-water reaches z = r and stays there: each later step is one
    floor clamp with the stored v_z zeroed, and nothing pushes it below the floor."""
    d = dims(cfg3)
    pos, alive, partner = _scene(cfg3, [(20.0, 20.0, 2.0)])
    n = d.n
    act, mo, zf = _actions(n), _motion(n), np.zeros(n, np.float32)
    landed = None
    for t in range(80):
        f = _forces3(cfg3, pos, zf, zf, alive, partner, act, mo, zf)
        out = _integrate3(cfg3, pos, f, alive=alive)
        if landed is None and out.pos[0, 2] == f32(d.z_lo):
            landed = t
        if landed is not None:
            assert out.pos[0, 2] == f32(d.z_lo) and out.vel[0, 2] == 0.0 and int(out.z_floor) == 1
        else:
            assert int(out.z_floor) == 0 and out.vel[0, 2] == -f32(G)
        pos = out.pos
    assert landed == math.ceil((2.0 - d.z_lo) / G) - 1  # the step whose move would cross the floor
    assert np.array_equal(pos[0, :2], [20.0, 20.0])


def test_grounded_is_inclusive_at_z_ground_and_never_at_the_surface(cfg3: Config, level3: Config) -> None:
    d = dims(cfg3)
    assert d.z_ground == float(f32(cfg3.physics.radius + 0.1))
    above = float(np.nextafter(f32(d.z_ground), f32(np.inf)))
    z = [d.z_lo, d.z_ground, above, 12.0, d.z_hi]
    pos = np.array([(1.0, 1.0, zz) for zz in z], np.float32)
    assert np.array(physics.grounded(mx.array(pos), cfg3)).tolist() == [True, True, False, False, False]
    d0 = dims(level3)  # floor_eps = 0: grounded exactly at z = r
    assert d0.z_ground == d0.z_lo == 1.0
    up = float(np.nextafter(f32(1.0), f32(2.0)))
    pos0 = np.array([(1.0, 1.0, 1.0), (1.0, 1.0, up), (1.0, 1.0, d0.z_hi)], np.float32)
    assert np.array(physics.grounded(mx.array(pos0), level3)).tolist() == [True, False, False]


def test_grip_drags_x_and_y_only_and_only_when_grounded(cfg3: Config) -> None:
    ph, d = cfg3.physics, dims(cfg3)
    zg = d.z_lo + 0.05  # grounded (≤ r + floor_eps) and above the floor, so a small sink is not clamped
    pts = [(5.0, 5.0, zg), (9.0, 5.0, 12.0), (13.0, 5.0, zg), (17.0, 5.0, d.z_hi)]
    pos, _, _ = _scene(cfg3, pts)
    n = pos.shape[0]
    force = np.zeros((n, 3), np.float32)
    force[:4] = (0.3, -0.1, -0.02)
    grip = np.zeros(n, np.float32)
    grip[[0, 1, 3]] = 1.0  # row 2 is grounded without grip
    out = _integrate3(cfg3, pos, force, grip=grip)
    gam = f32(ph.drag * (1.0 + ph.grip_gain))
    assert np.array_equal(out.vel[0, :2], force[0, :2] / gam)  # grounded, gripping: slowed in x and y
    assert out.vel[0, 2] == f32(-0.02)  # ... but never slowed while sinking
    for row in (1, 2, 3):  # mid-water and the surface grant no grip; row 2 does not grip
        assert np.array_equal(out.vel[row, :2], force[row, :2] / f32(ph.drag)), row
    assert out.vel[1, 2] == f32(-0.02) and int(out.z_floor) == 0


def test_v_max_clips_the_3d_speed_before_the_clamp(cfg3: Config) -> None:
    d, vmax = dims(cfg3), cfg3.physics.v_max
    pos, _, _ = _scene(cfg3, [(5.0, 5.0, 12.0), (9.0, 5.0, 12.0), (13.0, 5.0, d.z_lo + 0.1)])
    force = np.zeros(pos.shape, np.float32)
    force[0] = (0.3, 0.0, -0.35)  # |v| < 0.5: not clipped
    force[1] = (0.3, 0.3, -0.4)  # clipped to 0.5
    force[2] = (0.4, 0.0, -0.9)  # clipped, then clamped at the floor
    out = _integrate3(cfg3, pos, force)
    assert int(out.v_clips) == 2 and out.v_clips.dtype == np.int32
    assert np.allclose(out.vel[0], [0.3, 0.0, -0.35])
    assert np.isclose(np.linalg.norm(out.vel[1]), vmax, rtol=1e-6)
    s = vmax / math.hypot(0.4, 0.9)
    assert np.isclose(out.vel[2, 0], 0.4 * s, rtol=1e-6) and out.vel[2, 2] == 0.0
    assert out.pos[2, 2] == f32(d.z_lo) and int(out.z_floor) == 1
    assert np.linalg.norm(out.vel, axis=1).max() <= vmax * (1 + 1e-6)


def test_pitch_is_clipped_and_dead_rows_keep_their_kinematics(cfg3: Config) -> None:
    pos, alive, _ = _scene(cfg3, [(5.0, 5.0, 12.0)] * 5)
    alive[4] = False
    n = pos.shape[0]
    pitch = np.zeros(n, np.float32)
    pitch[:5] = [1.3, -1.3, 0.2, PM32, 0.7]
    dpitch = np.zeros(n, np.float32)
    dpitch[:5] = [0.3, -0.3, -0.25, 0.0, 0.3]
    heading = np.zeros(n, np.float32)
    heading[:5] = [0.5, 3.1, -1.0, 0.0, 2.0]
    turn = np.zeros(n, np.float32)
    turn[:5] = [0.1, 0.2, 0.0, 0.0, 0.3]
    force = np.zeros(pos.shape, np.float32)
    force[4] = (0.3, 0.3, 0.3)
    out = _integrate3(cfg3, pos, force, pitch=pitch, dpitch=dpitch, heading=heading, turn=turn, alive=alive)
    assert out.pitch[0] == PM32 and out.pitch[1] == -PM32 and out.pitch[3] == PM32
    assert out.pitch[2] == f32(0.2) + f32(-0.25)
    assert np.all(np.abs(out.pitch) <= PM32)
    assert out.pitch[4] == f32(0.7) and out.heading[4] == f32(2.0)  # dead: unchanged
    assert np.array_equal(out.pos[4], pos[4]) and not out.vel[4].any()
    assert np.isclose(out.heading[1], 3.3 - 2 * math.pi, atol=1e-6) and np.isclose(out.heading[0], 0.6)


# --- forces -------------------------------------------------------------------------------------------------


def test_gravity_and_buoyancy_are_not_attenuated_by_contacts(cfg3: Config) -> None:
    """A fully attenuated agent (4 contacts) has no thrust, but the body forces act in full (§27.3)."""
    c = (20.0, 20.0, 12.0)
    ring = [(21.5, 20.0, 12.0), (18.5, 20.0, 12.0), (20.0, 21.5, 12.0), (20.0, 18.5, 12.0)]
    pos, alive, partner = _scene(cfg3, [c, *ring])
    n = pos.shape[0]
    thrust = np.zeros(n, np.float32)
    thrust[0] = 1.0
    pitch = np.zeros(n, np.float32)
    pitch[0] = 0.9  # nose up: full thrust would lift it
    cap = np.full(n, 0.6, np.float32)
    buoy = np.zeros(n, np.float32)
    buoy[0] = 0.7
    act, mo = _actions(n, thrust=thrust), _motion(n, buoy=buoy)
    f = _forces3(cfg3, pos, np.zeros(n, np.float32), pitch, alive, partner, act, mo, cap)
    assert f[0, 2] == f32(BMAX) * f32(0.6) * f32(0.7) - f32(G)
    assert np.allclose(f[0, :2], 0.0, atol=1e-7)  # the symmetric ring cancels in x and y; no thrust
    # One contact: thrust at 75% along the 3D forward axis, plus the same body force.
    pos, alive, partner = _scene(cfg3, [c, ring[0]])
    f = _forces3(cfg3, pos, np.zeros(n, np.float32), pitch, alive, partner, act, mo, cap)
    rep = -cfg3.physics.k_repel * (2.0 - 1.5)
    th = cfg3.physics.thrust_max * 0.75
    expect = (th * math.cos(0.9) + rep, 0.0, th * math.sin(0.9) + BMAX * 0.6 * 0.7 - G)
    assert np.allclose(f[0], expect, atol=1e-7)


@pytest.mark.parametrize("gravity", [0.0, G])
def test_repulsion_and_springs_are_antisymmetric_in_3d(cfg3: Config, level3: Config, gravity: float) -> None:
    """Pairs in contact or bonded across the x/y seams and in z: equal and opposite internal forces, bitwise
    (in x and y also under gravity, a body force on z only)."""
    cfg = cfg3 if gravity else level3
    L = cfg.world.size
    pts = [
        (0.3, L - 0.2, 10.0), (L - 0.1, 0.5, 11.2),  # contact across both seams and in z
        (24.0, 24.0, 5.0), (25.0, 24.5, 6.6),  # bonded, mostly vertical
        (L - 0.4, 30.0, 18.0), (1.2, 30.4, 15.9),  # bonded across the x seam, stretched in z
        (12.0, 12.0, dims(cfg).z_lo), (12.3, 12.1, 2.7),  # contact: one on the floor, one above it
    ]  # fmt: skip
    pos, alive, partner = _scene(cfg, pts, [(2, 3), (4, 5)])
    n = pos.shape[0]
    rs = np.random.default_rng(1)
    contract = np.where(alive, rs.uniform(-1, 1, n), 0.0).astype(np.float32)
    act, mo, zf = _actions(n, contract=contract), _motion(n), np.zeros(n, np.float32)
    f = _forces3(cfg, pos, zf, zf, alive, partner, act, mo, zf)
    for i in range(0, 8, 2):
        assert np.abs(f[i]).max() > 1e-3, i  # each pair really interacts
        if gravity:
            assert np.array_equal(f[i, :2], -f[i + 1, :2]), i
            assert np.allclose(f[i, 2] + f[i + 1, 2], -2 * gravity, atol=1e-7), i
        else:
            assert np.array_equal(f[i], -f[i + 1]), i
    assert np.any(f[2:6, 2] != 0)  # the vertical spring components are exercised


def test_level_floor_scene_is_the_2d_physics(micro_cfg: Config, level3: Config) -> None:
    """All agents on the floor at pitch 0 with g = b = 0: ``forces3``/``integrate3`` in x and y equal
    ``forces``/``integrate`` (grip included, since every agent is grounded), z stays on the floor and nothing
    is clamped. Bitwise on CPU. F11 (contraction alone does not move the centre of mass; phase-locked grip
    crawls) therefore carries over to the floor."""
    rs = np.random.default_rng(7)
    n2, d3 = dims(micro_cfg).n, dims(level3)
    xy = [tuple(rs.uniform(10.0, 26.0, 2)) for _ in range(24)]
    pos2, alive, partner = _scene(micro_cfg, xy, [(0, 1), (2, 3), (3, 4), (5, 6)])
    pos3 = np.concatenate([pos2, np.full((n2, 1), d3.z_lo, np.float32)], axis=1)
    act = _actions(
        n2, thrust=rs.uniform(-1, 1, n2), contract=rs.uniform(-1, 1, n2), grip=rs.uniform(0, 1, n2),
        turn=rs.uniform(-0.3, 0.3, n2),
    )  # fmt: skip
    heading = rs.uniform(-3, 3, n2).astype(np.float32)
    zf = np.zeros(n2, np.float32)
    f2 = physics.forces(
        mx.array(pos2), mx.array(heading), act, _neighbors(pos2, alive, micro_cfg), mx.array(alive),
        mx.array(partner), micro_cfg,
    )  # fmt: skip
    f3 = physics.forces3(
        mx.array(pos3), mx.array(heading), mx.array(zf), act, _motion(n2), mx.array(zf),
        _neighbors(pos3, alive, level3), mx.array(alive), mx.array(partner), level3,
    )  # fmt: skip
    p2, v2, h2, c2 = physics.integrate(
        mx.array(pos2), mx.array(heading), f2, act.grip, act.turn, mx.array(alive), micro_cfg
    )
    out = physics.integrate3(
        mx.array(pos3), mx.array(heading), mx.array(zf), f3, act.grip, act.turn, mx.array(zf),
        physics.grounded(mx.array(pos3), level3), mx.array(alive), level3,
    )  # fmt: skip
    f2, f3 = np.array(f2), np.array(f3)
    assert np.abs(f2).max() > 1e-3 and not f3[:, 2].any()
    p3, v3, h3 = np.array(out.pos), np.array(out.vel), np.array(out.heading)
    for a, b in ((f2, f3[:, :2]), (np.array(p2), p3[:, :2]), (np.array(v2), v3[:, :2]), (np.array(h2), h3)):
        if ON_GPU:
            np.testing.assert_allclose(a, b, atol=1e-6)
        else:
            assert np.array_equal(a, b)
    assert np.all(p3[:, 2] == f32(d3.z_lo)) and not v3[:, 2].any()
    assert int(c2) == int(out.v_clips) and int(out.z_floor) == int(out.z_surface) == 0


def test_integrate_refuses_3d_positions(level3: Config) -> None:
    n = dims(level3).n
    pos, z = mx.zeros((n, 3)), mx.zeros((n,))
    with pytest.raises(AssertionError, match="integrate3"):
        physics.integrate(pos, z, mx.zeros((n, 3)), z, z, mx.ones((n,), mx.bool_), level3)


def test_physics_3d_compiles(cfg3: Config) -> None:
    rs = np.random.default_rng(5)
    d = dims(cfg3)
    pts = [(*rs.uniform(14.0, 22.0, 2), d.z_lo if i < 10 else rs.uniform(d.z_lo, 6.0)) for i in range(30)]
    pos, alive, partner = _scene(cfg3, pts, [(0, 1), (2, 3), (4, 5)])
    n = pos.shape[0]
    nbr = _neighbors(pos, alive, cfg3)
    act = _actions(n, thrust=rs.uniform(-1, 1, n), contract=rs.uniform(-1, 1, n), grip=rs.uniform(0, 1, n),
                   turn=rs.uniform(-0.3, 0.3, n))  # fmt: skip
    mo = _motion(n, pitch=rs.uniform(-0.3, 0.3, n), buoy=rs.uniform(0, 1, n))
    heading = mx.array(rs.uniform(-3, 3, n).astype(np.float32))
    pitch = mx.array(rs.uniform(-1.3, 1.3, n).astype(np.float32))
    cap = mx.array(rs.uniform(0, 1, n).astype(np.float32))

    def f(pos, heading, pitch, act, mo, cap, nbr, sv, partner):
        F = physics.forces3(pos, heading, pitch, act, mo, cap, nbr, sv, partner, cfg3)
        g = physics.grounded(pos, cfg3)
        return physics.integrate3(pos, heading, pitch, F, act.grip, act.turn, mo.pitch, g, sv, cfg3)

    args = (mx.array(pos), heading, pitch, act, mo, cap, nbr, mx.array(alive), mx.array(partner))
    eager, comp = f(*args), mx.compile(f)(*args)
    assert int(eager.z_floor) > 0  # the scene exercises the floor clamp
    for name, a, b in zip(physics.Integrated3._fields, eager, comp, strict=True):
        x, y = np.array(a), np.array(b)
        if x.dtype.kind == "f":
            np.testing.assert_allclose(x, y, atol=1e-6, err_msg=name)
        else:
            assert np.array_equal(x, y), name
