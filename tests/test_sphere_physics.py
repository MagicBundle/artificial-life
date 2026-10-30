"""Forces and integration on the sphere (DESIGN §46.2, §46.3; sphere series S2).

``physics.forces_s`` / ``integrate_s`` against the 3D box expressions on a locally flat patch, against an
independent float64 reference of the exponential-map step, the clamps and their counts, the soft lid, the drag
anisotropy, the invariants over a long random walk, and the flat limit of the box (7.4 of the spec).
"""

from __future__ import annotations

import dataclasses
import math

import mlx.core as mx
import numpy as np
import pytest

from alife import layout, physics, sphere
from alife.actions import Actions, Motion3
from alife.config import Config, dims, load_config
from alife.spatial import Neighbors
from alife.state import to_numpy
from sphere_fields_helpers import dev  # noqa: F401  (a fixture)
from sphere_rotation import random_rotation, rot

N = 48
B = layout.B


def cfg_of(*extra: str) -> Config:
    return load_config(preset="tiny3s", overrides=["brain.weight_dtype=float32", *extra])


def actions(rng: np.random.Generator, n: int = N, **kw) -> Actions:
    z = mx.zeros((n,))
    f = lambda a: mx.array(np.asarray(a, np.float32))  # noqa: E731
    base = dict(
        thrust=f(rng.uniform(-1, 1, n)),
        turn=f(rng.uniform(-0.3, 0.3, n)),
        msg=mx.zeros((n, layout.MSG)),
        emit=mx.zeros((n, layout.NCH)),
        harvest=z,
        grip=f(rng.uniform(0, 1, n)),
        contract=z,
        bond_want=mx.zeros((n,), dtype=mx.bool_),
        release=mx.zeros((n, B), dtype=mx.bool_),
        share=mx.zeros((n, B)),
        attack=mx.zeros((n,), dtype=mx.bool_),
        defend=z,
        reproduce=mx.zeros((n,), dtype=mx.bool_),
        child_bond=mx.zeros((n,), dtype=mx.bool_),
        bond_k=mx.full((n,), -1, dtype=mx.int32),
        bond_target=mx.full((n,), -1, dtype=mx.int32),
        attack_k=mx.full((n,), -1, dtype=mx.int32),
        attack_target=mx.full((n,), -1, dtype=mx.int32),
    )
    return Actions(**{**base, **kw})


def motion(rng: np.random.Generator, n: int = N) -> Motion3:
    return Motion3(
        pitch=mx.array(rng.uniform(-0.2, 0.2, n).astype(np.float32)),
        buoy=mx.array(rng.uniform(0, 1, n).astype(np.float32)),
    )


def no_neighbours(n: int = N, k: int = 8, radius: float = 6.0) -> Neighbors:
    z = mx.zeros((n,), dtype=mx.int32)
    return Neighbors(
        idx=mx.broadcast_to(mx.arange(n, dtype=mx.int32)[:, None], (n, k)),
        valid=mx.zeros((n, k), dtype=mx.bool_),
        off=mx.zeros((n, k, 3)),
        dist=mx.full((n, k), radius),
        overflow_cells=z.sum(),
        max_occupancy=z.sum(),
        hidden_agents=z.sum(),
    )


def tangent_headings(rng: np.random.Generator, pos: np.ndarray) -> np.ndarray:
    """Random unit vectors tangent to the sphere at each position (float32)."""
    up = pos / np.linalg.norm(pos, axis=1, keepdims=True)
    v = rng.normal(size=pos.shape)
    v -= (v * up).sum(axis=1, keepdims=True) * up
    return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)


def shell_pos(
    rng: np.random.Generator, cfg: Config, n: int = N, lo: float = 1.5, hi: float = 14.0
) -> np.ndarray:
    z = rng.uniform(-1, 1, n)
    lam = rng.uniform(0, 2 * math.pi, n)
    st = np.sqrt(1 - z * z)
    u = np.stack([st * np.cos(lam), st * np.sin(lam), z], axis=1)
    return (u * (cfg.world.radius + rng.uniform(lo, hi, n))[:, None]).astype(np.float32)


def integrate(
    cfg, pos, hdg, force, *, grip=None, turn=None, on_floor=None, sv=None, pitch=None, heading=None
):
    """``physics.integrate_s`` with defaults (zero grip, turn, pitch, heading; nobody grounded; all
    survive)."""
    n = pos.shape[0]
    z = mx.zeros((n,))
    sv = mx.ones((n,), dtype=mx.bool_) if sv is None else sv
    on_floor = mx.zeros((n,), dtype=mx.bool_) if on_floor is None else on_floor
    return physics.integrate_s(
        pos,
        hdg,
        z if heading is None else heading,
        z if pitch is None else pitch,
        force,
        z if grip is None else grip,
        z if turn is None else turn,
        z,
        on_floor,
        sv,
        cfg,
    )


def unit(a: np.ndarray) -> np.ndarray:
    return a / np.linalg.norm(a, axis=1, keepdims=True)


# --- grounding and forces ------------------------------------------------------------------------------


@pytest.mark.usefixtures("dev")
def test_grounded_is_the_height_above_the_floor_sphere() -> None:
    cfg = cfg_of()
    d, rp = dims(cfg), cfg.world.radius
    rng = np.random.default_rng(1)
    u = unit(shell_pos(rng, cfg))
    for z, want in (
        (cfg.physics.radius, True),
        (d.z_ground - 1e-3, True),
        (d.z_ground + 1e-3, False),
        (8.0, False),
    ):
        pos = mx.array((u * (rp + z)).astype(np.float32))
        assert np.array_equal(np.array(physics.grounded(pos, cfg)), np.full(N, want)), z
    rest = mx.broadcast_to(mx.array([0.0, 0.0, d.r_lo]), (4, 3))
    assert np.array(physics.grounded(rest, cfg)).all()  # a dead row rests on the floor


@pytest.mark.usefixtures("dev")
def test_on_the_level_surface_the_force_is_the_box_force() -> None:
    """At the north pole the tangent plane is the xy plane and the local up is ẑ: ``forces_s`` must equal
    ``forces3`` with the same actions and neighbours (the flat-limit test of the frame, thrust along ``fwd``,
    gravity and buoyancy along up, contacts from the chord offsets)."""
    cfg_s, cfg_b = cfg_of(), load_config(preset="tiny3d", overrides=["brain.weight_dtype=float32"])
    rng = np.random.default_rng(2)
    theta = rng.uniform(-math.pi, math.pi, N).astype(np.float32)
    pos_b = np.stack([rng.uniform(0, 20, N), rng.uniform(0, 20, N), rng.uniform(1.5, 14, N)], axis=1)
    pos_s = np.stack([np.zeros(N), np.zeros(N), cfg_s.world.radius + pos_b[:, 2]], axis=1).astype(np.float32)
    hdg = np.stack([np.cos(theta), np.sin(theta), np.zeros(N)], axis=1).astype(np.float32)
    pitch = mx.array(rng.uniform(-1.2, 1.2, N).astype(np.float32))
    act, mo, cap = actions(rng), motion(rng), mx.array(rng.uniform(0, 1, N).astype(np.float32))
    nb = no_neighbours()
    off = rng.normal(0, 1.2, (N, 8, 3)).astype(np.float32)  # chord offsets of eight neighbours
    dist = np.linalg.norm(off, axis=-1).astype(np.float32)
    nb = nb._replace(
        valid=mx.array(rng.random((N, 8)) < 0.5),
        off=mx.array(off),
        dist=mx.array(dist),
        idx=mx.array(rng.integers(0, N, (N, 8)).astype(np.int32)),
    )
    sv = mx.array(rng.random(N) < 0.9)
    partner = mx.full((N, B), -1, dtype=mx.int32)
    fb = physics.forces3(
        mx.array(pos_b.astype(np.float32)), mx.array(theta), pitch, act, mo, cap, nb, sv, partner, cfg_b
    )
    fs = physics.forces_s(mx.array(pos_s), mx.array(hdg), pitch, act, mo, cap, nb, sv, partner, cfg_s)
    np.testing.assert_allclose(np.array(fs), np.array(fb), rtol=0, atol=2e-7)
    assert not np.array(fs)[~np.array(sv)].any()  # non-survivors get 0


@pytest.mark.usefixtures("dev")
def test_forces_and_the_step_are_rotation_equivariant() -> None:
    """Rotate the world (positions, headings, neighbour offsets): the force rotates with it, the new position
    and heading are the rotated ones, the counts are equal. Nothing in forces or integration may read the
    axis."""
    cfg = cfg_of()
    rng = np.random.default_rng(3)
    pos = shell_pos(rng, cfg)
    hdg = tangent_headings(rng, pos)
    pitch = mx.array(rng.uniform(-1, 1, N).astype(np.float32))
    act, mo, cap = actions(rng), motion(rng), mx.array(rng.uniform(0, 1, N).astype(np.float32))
    off = rng.normal(0, 1.5, (N, 8, 3)).astype(np.float32)
    nb = no_neighbours()._replace(
        valid=mx.array(rng.random((N, 8)) < 0.5),
        off=mx.array(off),
        dist=mx.array(np.linalg.norm(off, axis=-1).astype(np.float32)),
        idx=mx.array(rng.integers(0, N, (N, 8)).astype(np.int32)),
    )
    sv, partner = mx.ones((N,), dtype=mx.bool_), mx.full((N, B), -1, dtype=mx.int32)
    r = random_rotation(4)

    def run(p, h, o):
        n = nb._replace(off=o)
        f = physics.forces_s(p, h, pitch, act, mo, cap, n, sv, partner, cfg)
        gr = physics.grounded(p, cfg)
        return f, physics.integrate_s(
            p, h, mx.zeros((N,)), pitch, f, act.grip, act.turn, mo.pitch, gr, sv, cfg
        )

    f0, it0 = run(mx.array(pos), mx.array(hdg), mx.array(off))
    f1, it1 = run(rot(pos, r), rot(hdg, r), rot(off, r))
    np.testing.assert_allclose(np.array(f1), np.array(rot(f0, r)), atol=3e-6)
    np.testing.assert_allclose(np.array(it1.pos), np.array(rot(it0.pos, r)), atol=3e-5)
    np.testing.assert_allclose(np.array(it1.hdg), np.array(rot(it0.hdg, r)), atol=3e-6)
    np.testing.assert_allclose(np.array(it1.vel), np.array(rot(it0.vel, r)), atol=3e-6)
    assert np.array_equal(np.array(it1.pitch), np.array(it0.pitch))
    assert [int(x) for x in (it1.v_clips, it1.z_floor, it1.z_surface)] == [
        int(x) for x in (it0.v_clips, it0.z_floor, it0.z_surface)
    ]


@pytest.mark.usefixtures("dev")
def test_the_soft_lid_attenuates_only_the_outward_thrust_and_buoyancy() -> None:
    band = 4.0
    cfg = cfg_of(f"physics.ceiling_band={band}")
    d, rp = dims(cfg), cfg.world.radius
    n = 5
    rng = np.random.default_rng(5)
    # agents stacked on one radius line (north pole direction), at depths 0, 0.25, 0.5, 0.75, 1 of the band
    depth = np.array([-0.5, 0.0, 0.5, 1.0, 2.0])  # below the band, at its start, middle, at the lid, beyond
    z = d.z_hi - band + depth * band
    pos = np.stack([np.zeros(n), np.zeros(n), rp + z], axis=1).astype(np.float32)
    up = np.array([0.0, 0.0, 1.0], np.float32)
    # heading east; pitch +1 radian: the thrust has an outward component sin(1) per unit thrust
    hdg = np.tile(np.array([1.0, 0.0, 0.0], np.float32), (n, 1))
    pitch = mx.full((n,), 1.0)
    act = actions(rng, n, thrust=mx.ones((n,)), grip=mx.zeros((n,)))
    mo = Motion3(pitch=mx.zeros((n,)), buoy=mx.ones((n,)))
    cap = mx.ones((n,))
    nb, sv, partner = no_neighbours(n), mx.ones((n,), dtype=mx.bool_), mx.full((n, B), -1, dtype=mx.int32)
    f = np.array(physics.forces_s(mx.array(pos), mx.array(hdg), pitch, act, mo, cap, nb, sv, partner, cfg))
    ph = cfg.physics
    out, along = f @ up, f[:, 0]
    thrust_out, buoy = ph.thrust_max * math.sin(1.0), ph.buoyancy_max
    keep = 1.0 - np.clip(depth, 0, 1) ** 2 * (3.0 - 2.0 * np.clip(depth, 0, 1))
    # contacts: none. Radial force = outward thrust·keep + buoyancy·keep − gravity (att = 1: no neighbours)
    np.testing.assert_allclose(out, (thrust_out + buoy) * keep - ph.gravity, rtol=0, atol=2e-7)
    np.testing.assert_allclose(
        along, np.full(n, ph.thrust_max * math.cos(1.0)), atol=2e-7
    )  # tangent: untouched
    assert keep[3] == 0.0 and keep[0] == 1.0 and (np.diff(keep) <= 0).all()
    # an inward thrust (pitch −1) and gravity are never attenuated
    f_in = np.array(
        physics.forces_s(
            mx.array(pos),
            mx.array(hdg),
            mx.full((n,), -1.0),
            act,
            Motion3(mx.zeros((n,)), mx.zeros((n,))),
            cap,
            nb,
            sv,
            partner,
            cfg,
        )
    )
    np.testing.assert_allclose(f_in @ up, np.full(n, -thrust_out - ph.gravity), atol=2e-7)


# --- the step -----------------------------------------------------------------------------------------------


def reference_step(cfg: Config, pos, hdg, v, turn, r_lo, r_hi):
    """The float64 exponential-map step (exact trig) with clamps; the independent oracle of
    ``integrate_s``."""
    pos, hdg, v = (np.asarray(a, np.float64) for a in (pos, hdg, v))
    r = np.linalg.norm(pos, axis=1, keepdims=True)
    up = pos / r
    vr = (v * up).sum(axis=1, keepdims=True)
    vt = v - vr * up
    sig = np.linalg.norm(vt, axis=1, keepdims=True) / r
    that = np.where(sig > 0, vt / np.maximum(np.linalg.norm(vt, axis=1, keepdims=True), 1e-300), 0.0)
    up2 = np.cos(sig) * up + np.sin(sig) * that
    r2 = np.clip(r + vr, r_lo, r_hi)
    # parallel transport along the great circle: rotate by sig about the axis up × that
    ax = np.cross(up, that)
    axn = np.linalg.norm(ax, axis=1, keepdims=True)
    k = np.where(axn > 0, ax / np.maximum(axn, 1e-300), 0.0)
    ct, st = np.cos(sig), np.sin(sig)
    h1 = hdg * ct + np.cross(k, hdg) * st + k * (k * hdg).sum(axis=1, keepdims=True) * (1 - ct)
    tc, ts = np.cos(turn)[:, None], np.sin(turn)[:, None]
    h2 = tc * h1 + ts * np.cross(up2, h1)
    h2 -= (h2 * up2).sum(axis=1, keepdims=True) * up2
    return r2 * up2, h2 / np.linalg.norm(h2, axis=1, keepdims=True), r + vr, vr


@pytest.mark.usefixtures("dev")
def test_integrate_matches_the_float64_reference_with_anisotropic_drag() -> None:
    cfg = cfg_of()
    d, ph = dims(cfg), cfg.physics
    rng = np.random.default_rng(6)
    pos = shell_pos(rng, cfg, lo=3.0, hi=12.0)
    hdg = tangent_headings(rng, pos)
    force = (rng.normal(size=(N, 3)) * 0.03).astype(np.float32)
    grip = rng.uniform(0, 1, N).astype(np.float32)
    turn = rng.uniform(-0.3, 0.3, N).astype(np.float32)
    on_floor = np.arange(N) % 3 == 0
    sv = np.arange(N) % 7 != 0
    it = integrate(
        cfg,
        mx.array(pos),
        mx.array(hdg),
        mx.array(force),
        grip=mx.array(grip),
        turn=mx.array(turn),
        on_floor=mx.array(on_floor),
        sv=mx.array(sv),
    )
    # the commanded step v: tangent part over γ_t (grip on grounded rows), radial part over γ_r = drag
    up = pos.astype(np.float64) / np.linalg.norm(pos, axis=1, keepdims=True)
    fr = (force * up).sum(axis=1, keepdims=True)
    gam_t = ph.drag * (1.0 + ph.grip_gain * np.where(on_floor, grip, 0.0))[:, None]
    v = (force - fr * up) / gam_t + fr / ph.drag * up
    speed = np.linalg.norm(v, axis=1, keepdims=True)
    v = np.where(speed > ph.v_max, v * ph.v_max / np.maximum(speed, 1e-300), v)
    assert int(it.v_clips) == int(((np.linalg.norm(v, axis=1) > 0) & sv & (speed[:, 0] > ph.v_max)).sum())
    v = np.where(sv[:, None], v, 0.0)
    p2, h2, _, _ = reference_step(cfg, pos, hdg, v, np.where(sv, turn, 0.0), float(d.r_lo), float(d.r_hi))
    got_p, got_h, got_v = (np.array(a, np.float64) for a in (it.pos, it.hdg, it.vel))
    keep = ~sv
    np.testing.assert_array_equal(got_p[keep], pos[keep])  # non-survivors keep pos and hdg exactly
    np.testing.assert_array_equal(got_h[keep], hdg[keep])
    assert not got_v[keep].any()
    np.testing.assert_allclose(got_p[sv], p2[sv], rtol=0, atol=6e-6)
    np.testing.assert_allclose(got_h[sv], h2[sv], rtol=0, atol=6e-6)
    np.testing.assert_allclose(got_v[sv], v[sv], rtol=0, atol=1e-7)
    # grip slows the tangent motion of a grounded row and never the radial one
    r_get = (got_v * up).sum(axis=1)
    np.testing.assert_allclose(r_get[sv], (fr[:, 0] / ph.drag)[sv], atol=1e-7)
    # the derived heading is the bearing of the new hdg (compared circularly)
    want = np.array(sphere.bearing(mx.array(got_h.astype(np.float32)), mx.array(got_p.astype(np.float32))))
    diff = np.abs((np.array(it.heading)[sv] - want[sv] + math.pi) % (2 * math.pi) - math.pi)
    assert diff.max() < 1e-5
    assert np.array_equal(np.array(it.heading)[keep], np.zeros(keep.sum()))


@pytest.mark.usefixtures("dev")
def test_the_floor_and_the_lid_clamp_and_count() -> None:
    cfg = cfg_of()
    d, rp = dims(cfg), cfg.world.radius
    ph = cfg.physics
    rng = np.random.default_rng(7)
    n = 12
    u = unit(shell_pos(rng, cfg, n))
    # six agents resting on the floor pushed down, six at the lid pushed up, half of each also moving sideways
    rad = np.array([float(d.r_lo)] * 6 + [float(d.r_hi)] * 6)
    pos = (u * rad[:, None]).astype(np.float32)
    hdg = tangent_headings(rng, pos)
    push = np.array([-0.02] * 6 + [0.02] * 6, np.float32)
    side = np.where(np.arange(n) % 2 == 0, 0.01, 0.0).astype(np.float32)
    upv = pos / np.linalg.norm(pos, axis=1, keepdims=True)
    force = push[:, None] * upv + side[:, None] * hdg
    sv = np.ones(n, bool)
    sv[5] = sv[11] = False  # two rows are not survivors: no clamp counted for them
    it = integrate(cfg, mx.array(pos), mx.array(hdg), mx.array(force), sv=mx.array(sv))
    assert (int(it.z_floor), int(it.z_surface)) == (5, 5)
    r_new = np.linalg.norm(np.array(it.pos, np.float64), axis=1)
    eps = 4 * np.spacing(np.float32(rp + cfg.world.height))
    np.testing.assert_allclose(r_new[sv], rad[sv], rtol=0, atol=eps)  # on the bound (to float32 rounding)
    # the stored velocity has no radial part at the old point on a clamped row, the sideways part is kept
    vr = (np.array(it.vel, np.float64) * upv).sum(axis=1)
    assert np.abs(vr[sv]).max() < 1e-7
    tang = np.array(it.vel, np.float64)[sv] - vr[sv][:, None] * upv[sv]
    np.testing.assert_allclose(np.linalg.norm(tang, axis=1), side[sv] / ph.drag, atol=1e-7)
    # an unclamped agent keeps its radial speed: floor row pushed UP leaves the floor
    it2 = integrate(cfg, mx.array(pos[:6]), mx.array(hdg[:6]), mx.array(-force[:6]))
    assert (
        int(it2.z_floor) == 0
        and (np.linalg.norm(np.array(it2.pos, np.float64), axis=1) > rad[:6] + 1e-4).all()
    )


@pytest.mark.usefixtures("dev")
def test_clipping_counts_each_clipped_survivor() -> None:
    cfg = cfg_of()
    ph = cfg.physics
    rng = np.random.default_rng(8)
    pos = shell_pos(rng, cfg, 10, lo=4, hi=10)
    hdg = tangent_headings(rng, pos)
    big = np.tile(np.array([[0.0, 0.0, 0.0]], np.float32), (10, 1))
    big[:4] = hdg[:4] * (ph.v_max * ph.drag * 3.0)  # four rows far beyond v_max
    big[4:] = hdg[4:] * (ph.v_max * ph.drag * 0.2)  # six within
    sv = np.ones(10, bool)
    sv[3] = False
    it = integrate(cfg, mx.array(pos), mx.array(hdg), mx.array(big), sv=mx.array(sv))
    assert int(it.v_clips) == 3
    speed = np.linalg.norm(np.array(it.vel, np.float64), axis=1)
    assert speed[:3].max() <= ph.v_max * (1 + 1e-6) and speed[3] == 0.0


@pytest.mark.usefixtures("dev")
def test_a_long_random_walk_keeps_the_invariants_through_both_poles() -> None:
    """3 000 steps of strong random forces: ``|hdg| = 1``, tangent, the radius inside the bounds with the
    tolerances of ``invariants``, and the heading vector turns continuously across the poles."""
    cfg = cfg_of()
    d, rp = dims(cfg), cfg.world.radius
    n = 64
    rng = np.random.default_rng(9)
    pos = shell_pos(rng, cfg, n)
    pos[: n // 4] = np.array([0.0, 1e-3, 1.0]) * ((rp + 8.0) / 1.0)  # a quarter starts at the north pole
    pos[n // 4 : n // 2] = np.array([1e-3, 0.0, -1.0]) * (rp + 8.0)  # and a quarter at the south pole
    pos = pos.astype(np.float32)
    hdg = tangent_headings(rng, pos)
    p, h = mx.array(pos), mx.array(hdg)
    sv, ones = mx.ones((n,), dtype=mx.bool_), mx.zeros((n,))
    eps = 4 * np.spacing(np.float32(rp + cfg.world.height))
    worst = [0.0, 0.0, 0.0]
    for _ in range(3000):
        force = mx.array((rng.normal(size=(n, 3)) * 0.04).astype(np.float32))
        turn = mx.array(rng.uniform(-0.4, 0.4, n).astype(np.float32))
        gr = physics.grounded(p, cfg)
        it = physics.integrate_s(p, h, ones, ones, force, ones + 0.5, turn, ones, gr, sv, cfg)
        mx.eval(it.pos, it.hdg)
        p, h = it.pos, it.hdg
    pn, hn = np.array(p, np.float64), np.array(h, np.float64)
    rad = np.linalg.norm(pn, axis=1)
    worst = [abs(np.linalg.norm(hn, axis=1) - 1).max(), abs((hn * pn).sum(1) / rad).max()]
    assert worst[0] < 4e-7 and worst[1] < 4e-7, worst
    assert rad.min() >= float(d.r_lo) - eps and rad.max() <= float(d.r_hi) + eps


# --- the flat limit (spec 7.4) ----------------------------------------------------------------------------


@pytest.mark.usefixtures("dev")
def test_a_large_planet_is_the_box_over_a_patch() -> None:
    """Flat-limit test: the planet of radius 255 (the largest the validated range admits, ``Nh = 59``), agents
    confined to a patch of a few units at the pole, 20 steps of the sphere's forces and integration against
    the
    box's ``forces3`` and ``integrate3`` with the same actions. The commanded step of each agrees to 1 % of
    its
    length at every step, and the positions to 2 % of the path (curvature ``δ²/2R`` and the holonomy are
    below that): every sign, frame and transport error is far above it."""
    cfg_b = load_config(preset="tiny3d", overrides=["brain.weight_dtype=float32"])
    w = dataclasses.replace(load_config(preset="tiny3s").world, radius=255.0)  # not validated: the cell rules
    cfg_s = dataclasses.replace(load_config(preset="tiny3s"), world=w)
    rp = 255.0
    n = 32
    rng = np.random.default_rng(10)
    xy = rng.uniform(-0.5, 0.5, (n, 2))
    z0 = rng.uniform(2.0, 12.0, n)
    theta = rng.uniform(-math.pi, math.pi, n)
    centre = cfg_b.world.size / 2  # the box is a torus: start away from the seam at 0
    pos_b = np.concatenate([xy + centre, z0[:, None]], axis=1).astype(np.float32)
    pos_s = np.concatenate([xy, (rp + z0)[:, None]], axis=1)
    pos_s = (pos_s * ((rp + z0) / np.linalg.norm(pos_s, axis=1))[:, None]).astype(np.float32)
    up = pos_s / np.linalg.norm(pos_s, axis=1, keepdims=True)
    hdg0 = np.stack([np.cos(theta), np.sin(theta), np.zeros(n)], axis=1)
    hdg0 = (hdg0 - (hdg0 * up).sum(1, keepdims=True) * up).astype(np.float32)
    hdg0 = (hdg0 / np.linalg.norm(hdg0, axis=1, keepdims=True)).astype(np.float32)
    ps, hs, pb, hb = mx.array(pos_s), mx.array(hdg0), mx.array(pos_b), mx.array(theta.astype(np.float32))
    pitch_s = pitch_b = mx.array(rng.uniform(-0.6, 0.6, n).astype(np.float32))
    heading_s = mx.zeros((n,))
    sv, nb = mx.ones((n,), dtype=mx.bool_), no_neighbours(n)
    partner = mx.full((n, B), -1, dtype=mx.int32)
    cap = mx.array(rng.uniform(0.3, 0.7, n).astype(np.float32))
    path = np.zeros(n)
    for _step in range(20):
        act, mo = actions(rng, n), motion(rng, n)
        act = act._replace(thrust=mx.abs(act.thrust))
        gs, gb = physics.grounded(ps, cfg_s), physics.grounded(pb, cfg_b)
        assert np.array_equal(np.array(gs), np.array(gb))
        Fs = physics.forces_s(ps, hs, pitch_s, act, mo, cap, nb, sv, partner, cfg_s)
        Fb = physics.forces3(pb, hb, pitch_b, act, mo, cap, nb, sv, partner, cfg_b)
        its = physics.integrate_s(ps, hs, heading_s, pitch_s, Fs, act.grip, act.turn, mo.pitch, gs, sv, cfg_s)
        itb = physics.integrate3(pb, hb, pitch_b, Fb, act.grip, act.turn, mo.pitch, gb, sv, cfg_b)
        vs, vb = np.array(its.vel, np.float64), np.array(itb.vel, np.float64)
        step_len = np.linalg.norm(vb, axis=1)
        # compare in the common frame: the sphere's step is expressed in planet axes, the box's in world axes;
        # near the pole they differ by the rotation of the up vector (|Δup| < 0.04 over the patch)
        assert (np.linalg.norm(vs - vb, axis=1) <= 0.01 * step_len + 2e-5).all(), _step
        path += step_len
        ps, hs, heading_s, pitch_s = its.pos, its.hdg, its.heading, its.pitch
        pb, hb, pitch_b = itb.pos, itb.heading, itb.pitch
        assert np.array_equal(np.array(its.pitch), np.array(itb.pitch))
        assert int(its.z_floor) == int(itb.z_floor) and int(its.z_surface) == int(itb.z_surface)
    got = np.array(ps, np.float64) - np.array([0.0, 0.0, rp])
    want = np.array(pb, np.float64) - np.array([centre, centre, 0.0])
    assert (np.abs(got[:, :2] - want[:, :2]).max(axis=1) <= 0.02 * path + 1e-3).all()
    assert (np.abs(got[:, 2] - want[:, 2]) <= 0.02 * path + 1e-3).all()
    assert path.max() > 0.5  # the agents really moved


def test_a_sphere_state_is_the_planet_frame_not_the_box() -> None:
    """The box functions refuse nothing by themselves, so the dispatch is by ``dims(cfg).sph``:
    ``grounded`` of
    a sphere config uses the radius (a Cartesian z of 1.0 would be 'grounded' in a box)."""
    cfg = cfg_of()
    pos = mx.array([[0.0, 0.0, 1.0], [30.0, 0.0, 0.0]])  # |pos| = 1 (far below the floor), 30 (below too)
    assert np.array(physics.grounded(pos, cfg)).all()
    high = mx.array([[40.0, 0.0, 0.0], [0.0, 0.0, -44.0]])  # |pos| = 40, 44: height 4 and 8 above the floor
    assert not np.array(physics.grounded(high, cfg)).any()
    assert to_numpy(physics.grounded(high, cfg)).dtype == bool
