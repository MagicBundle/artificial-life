"""Overdamped physics with thrust, springs, grip and soft repulsion (§9); the 3D variants add gravity, the
buoyancy force, the floor and the surface (§27.3). Owned by WP3."""

from __future__ import annotations

from typing import NamedTuple

import mlx.core as mx

from alife import bonds, layout, ops, sphere
from alife.actions import Actions, Motion3
from alife.config import Config, dims
from alife.spatial import Neighbors


def forces(
    pos: mx.array,
    heading: mx.array,
    act: Actions,
    nbr: Neighbors,
    survivor: mx.array,
    partner: mx.array,
    cfg: Config,
    *,
    wind: mx.array | None = None,
) -> mx.array:
    """Total force per agent (N,2): contact-attenuated thrust, springs and repulsion (§9).

    Evaluated at start-of-step ``pos``/``heading`` with the post-S14 ``partner``. A contact is a valid
    neighbour that survives and is closer than ``2r``; thrust is scaled by
    ``max(0, 1 − n_contacts/thrust_contact_limit)`` (F18). Non-survivors get 0. ``wind`` ((N,2), the
    already-multiplied drag force ``alpha·drag·u_air``, W1 §30) is added unattenuated; ``None`` is the
    weather-off graph.
    """
    ph = cfg.physics
    two_r = 2.0 * ph.radius
    contact = nbr.valid & survivor[nbr.idx] & (nbr.dist < two_r)  # (N,K): the repulsion mask
    n_contact = contact.sum(axis=1).astype(mx.float32)
    att = mx.maximum(1.0 - n_contact / ph.thrust_contact_limit, 0.0)
    fwd = mx.stack([mx.cos(heading), mx.sin(heading)], axis=-1)
    f_thrust = (ph.thrust_max * act.thrust * att)[:, None] * fwd
    u = nbr.off / mx.maximum(nbr.dist, layout.EPS_LEN)[..., None]
    rep = mx.where(contact[..., None], (-ph.k_repel * (two_r - nbr.dist))[..., None] * u, 0.0).sum(axis=1)
    f = f_thrust + rep + bonds.spring_forces(pos, partner, act.contract, cfg)
    if wind is not None:  # static: weather on
        f = f + wind
    return mx.where(survivor[:, None], f, 0.0)


def integrate(
    pos: mx.array,
    heading: mx.array,
    force: mx.array,
    grip: mx.array,
    turn: mx.array,
    survivor: mx.array,
    cfg: Config,
    drag_mult: mx.array | None = None,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Explicit overdamped step for survivors: ``(pos, vel, heading, n_clipped () int32)`` (§9).

    ``v = F/γ`` with ``γ = drag·(1 + grip_gain·grip)``, clipped to ``‖v‖ ≤ v_max`` (each clip counted);
    ``pos ← wrap(pos + v)``, ``heading ← wrap_angle(heading + turn)``. Non-survivors keep ``pos`` and
    ``heading`` and get ``vel = 0``. ``drag_mult`` ((N,) f32 ≥ 1, the built material's barrier, §38.4.2)
    multiplies ``γ``; None (structures off, static) traces today's graph.
    """
    assert pos.shape[-1] == 2, "integrate got 3D positions (integrate3)"  # static: z would be wrapped
    ph = cfg.physics
    gamma = ph.drag * (1.0 + ph.grip_gain * grip)
    if drag_mult is not None:  # static: structures on
        gamma = gamma * drag_mult
    v = force / gamma[:, None]
    speed = mx.sqrt((v * v).sum(axis=-1))
    clip = survivor & (speed > ph.v_max)
    v = mx.where(clip[:, None], v * (ph.v_max / mx.maximum(speed, ph.v_max))[:, None], v)
    v = mx.where(survivor[:, None], v, 0.0)
    new_pos = mx.where(survivor[:, None], ops.wrap(pos + v, cfg.world.size), pos)
    new_heading = mx.where(survivor, ops.wrap_angle(heading + turn), heading)
    return new_pos, v, new_heading, clip.sum().astype(mx.int32)


# --- 3D (§27.3): gravity, the buoyancy force, the floor and the surface -------------------------------------


class Integrated3(NamedTuple):
    pos: mx.array  # (N,3): x, y wrapped; z clamped to [z_lo, z_hi]
    vel: mx.array  # (N,3): the clipped v, with v_z zeroed on the rows clamped this step
    heading: mx.array  # (N,)
    pitch: mx.array  # (N,) in [−PITCH_MAX, PITCH_MAX]
    v_clips: mx.array  # () int32: survivors whose ‖v‖ was clipped to v_max
    z_floor: mx.array  # () int32: survivors clamped at the floor
    z_surface: mx.array  # () int32: survivors clamped at the surface


def grounded(pos: mx.array, cfg: Config) -> mx.array:
    """(N,) bool: on the floor, ``z ≤ Dims.z_ground`` (§27.3): ``z ≤ r + floor_eps`` exactly, or ``z = z_lo``.

    The one grounding predicate (floor friction, ``IN_GROUNDED``, the ``GROUNDED`` counter), taken from the
    start-of-step position. The surface never grounds. On a sphere the height is ``|pos| − Rp`` (§46.2).
    """
    if dims(cfg).sph:  # static: the sphere world
        return sphere.geometry(cfg).height(pos) <= dims(cfg).z_ground
    return pos[:, 2] <= dims(cfg).z_ground


def forces3(
    pos: mx.array,
    heading: mx.array,
    pitch: mx.array,
    act: Actions,
    mo: Motion3,
    buoy_cap: mx.array,
    nbr: Neighbors,
    survivor: mx.array,
    partner: mx.array,
    cfg: Config,
    *,
    wind: mx.array | None = None,
) -> mx.array:
    """Total force per agent (N,3) in a 3D world (§27.3): ``forces`` plus gravity and buoyancy.

    Thrust acts along the body's forward axis (``ops.body_axes``) and is contact-attenuated as in 2D (F18);
    repulsion and springs use the (N,K,3) and (N,B,3) offsets. Gravity ``−gravity·ẑ`` and the buoyancy
    ``buoyancy_max·buoy_cap·mo.buoy·ẑ`` are body forces on z only and are not attenuated. Non-survivors get 0.
    ``wind`` ((N,3), the already-multiplied drag force ``alpha·drag·u_air``, W1 §30) is added after them,
    also unattenuated; ``None`` is the weather-off graph.
    """
    ph = cfg.physics
    two_r = 2.0 * ph.radius
    contact = nbr.valid & survivor[nbr.idx] & (nbr.dist < two_r)  # (N,K): the repulsion mask
    n_contact = contact.sum(axis=1).astype(mx.float32)
    att = mx.maximum(1.0 - n_contact / ph.thrust_contact_limit, 0.0)
    fwd, _, _ = ops.body_axes(mx.cos(heading), mx.sin(heading), mx.cos(pitch), mx.sin(pitch))
    f_thrust = (ph.thrust_max * act.thrust * att)[:, None] * fwd
    u = nbr.off / mx.maximum(nbr.dist, layout.EPS_LEN)[..., None]
    rep = mx.where(contact[..., None], (-ph.k_repel * (two_r - nbr.dist))[..., None] * u, 0.0).sum(axis=1)
    f = f_thrust + rep + bonds.spring_forces(pos, partner, act.contract, cfg)
    lift = ph.buoyancy_max * buoy_cap * mo.buoy - ph.gravity  # (N,) along +z
    f = mx.concatenate([f[:, :2], f[:, 2:] + lift[:, None]], axis=1)
    if wind is not None:  # static: weather on
        f = f + wind
    return mx.where(survivor[:, None], f, 0.0)


def integrate3(
    pos: mx.array,
    heading: mx.array,
    pitch: mx.array,
    force: mx.array,
    grip: mx.array,
    turn: mx.array,
    dpitch: mx.array,
    on_floor: mx.array,
    survivor: mx.array,
    cfg: Config,
    drag_mult: mx.array | None = None,
) -> Integrated3:
    """Explicit overdamped 3D step for survivors (§27.3).

    Anisotropic drag: ``γ_xy = drag·(1 + grip_gain·grip)`` on grounded rows (``on_floor``, from ``grounded``)
    and ``drag`` elsewhere; ``γ_z = drag`` always, so grip never slows sinking. ``‖v‖`` (3D) is clipped to
    ``v_max`` as in 2D. ``pos ← wrap_xy_clamp_z(pos + v)``: an agent that would cross the floor or the
    surface moves to it, and its stored ``v_z`` is zeroed (only on those rows; each clamp counted). Yaw as in
    2D; ``pitch ← clip(pitch + dpitch, ±PITCH_MAX)``. Non-survivors keep ``pos``, ``heading`` and ``pitch``
    and get ``vel = 0``. ``drag_mult`` ((N,) f32 ≥ 1, the built material's barrier, §38.4.2) multiplies
    both ``γ_xy`` and ``γ_z`` (a dense medium also slows sinking; grip stays floor-only); None traces
    today's graph.
    """
    ph, d = cfg.physics, dims(cfg)
    gamma_xy = ph.drag * (1.0 + ph.grip_gain * mx.where(on_floor, grip, 0.0))
    gamma_z = ph.drag
    if drag_mult is not None:  # static: structures on
        gamma_xy = gamma_xy * drag_mult
        gamma_z = (ph.drag * drag_mult)[:, None]
    v = mx.concatenate([force[:, :2] / gamma_xy[:, None], force[:, 2:] / gamma_z], axis=1)
    speed = mx.sqrt((v * v).sum(axis=-1))
    clip = survivor & (speed > ph.v_max)
    v = mx.where(clip[:, None], v * (ph.v_max / mx.maximum(speed, ph.v_max))[:, None], v)
    v = mx.where(survivor[:, None], v, 0.0)
    p = pos + v
    vz = v[:, 2]
    at_floor = survivor & (p[:, 2] <= d.z_lo) & (vz < 0.0)
    at_surface = survivor & (p[:, 2] >= d.z_hi) & (vz > 0.0)
    vel = mx.concatenate([v[:, :2], mx.where(at_floor | at_surface, 0.0, vz)[:, None]], axis=1)
    new_pos = mx.where(survivor[:, None], ops.wrap_xy_clamp_z(p, cfg.world.size, d.z_lo, d.z_hi), pos)
    new_heading = mx.where(survivor, ops.wrap_angle(heading + turn), heading)
    new_pitch = mx.where(survivor, mx.clip(pitch + dpitch, -layout.PITCH_MAX, layout.PITCH_MAX), pitch)
    return Integrated3(
        pos=new_pos,
        vel=vel,
        heading=new_heading,
        pitch=new_pitch,
        v_clips=clip.sum().astype(mx.int32),
        z_floor=at_floor.sum().astype(mx.int32),
        z_surface=at_surface.sum().astype(mx.int32),
    )


# --- the sphere (§46.2, §46.3): radial gravity, the floor and the hard lid, tangent motion ----------------


class IntegratedS(NamedTuple):
    pos: mx.array  # (N,3) planet frame, |pos| clamped to [r_lo, r_hi]
    vel: mx.array  # (N,3): the clipped commanded step, with its radial part zeroed on the clamped rows
    hdg: mx.array  # (N,3) tangent unit headings, transported and renormalised
    heading: mx.array  # (N,) derived bearing from local north (output only, never read by the dynamics)
    pitch: mx.array  # (N,) in [−PITCH_MAX, PITCH_MAX]
    v_clips: mx.array  # () int32: survivors whose ‖v‖ was clipped to v_max
    z_floor: mx.array  # () int32: survivors clamped at the floor (radius r_lo)
    z_surface: mx.array  # () int32: survivors clamped at the lid (radius r_hi)


def forces_s(
    pos: mx.array,
    hdg: mx.array,
    pitch: mx.array,
    act: Actions,
    mo: Motion3,
    buoy_cap: mx.array,
    nbr: Neighbors,
    survivor: mx.array,
    partner: mx.array,
    cfg: Config,
    *,
    wind: mx.array | None = None,
) -> mx.array:
    """Total force per agent (N,3) on the sphere (§46.2): ``forces3`` with the body frame of ``hdg`` and the
    local up ``û = pos/|pos|`` in place of (yaw, pitch; ẑ).

    Thrust acts along ``fwd = cφ·hdg + sφ·û`` (``ops.body_axes_s``), contact-attenuated (F18); repulsion and
    springs use the (N,K,3) and (N,B,3) chord offsets (``Neighbors.off`` is a plain difference: ``min_image``
    is the identity on a sphere's pair offsets). Gravity ``−gravity·û`` is radial and constant in magnitude;
    the buoyancy ``buoyancy_max·buoy_cap·mo.buoy·û`` is a body force along ``û`` and, like gravity, not
    attenuated by contacts. With ``physics.ceiling_band > 0`` (static) the *outward* part of the thrust and
    the buoyancy is multiplied by ``1 − smoothstep`` of the depth into the top band of the shell (the soft
    lid, §46.2); the hard lid is ``integrate_s``'s clamp. Non-survivors get 0.
    """
    ph, d = cfg.physics, dims(cfg)
    two_r = 2.0 * ph.radius
    contact = nbr.valid & survivor[nbr.idx] & (nbr.dist < two_r)  # (N,K): the repulsion mask
    n_contact = contact.sum(axis=1).astype(mx.float32)
    att = mx.maximum(1.0 - n_contact / ph.thrust_contact_limit, 0.0)
    up = pos / sphere.radius_of(pos)[:, None]
    fwd, _, _ = ops.body_axes_s(hdg, up, mx.cos(pitch), mx.sin(pitch))
    f_thrust = (ph.thrust_max * act.thrust * att)[:, None] * fwd
    buoy = ph.buoyancy_max * buoy_cap * mo.buoy
    if ph.ceiling_band > 0.0:  # static: the soft lid
        depth = mx.clip(
            (sphere.radius_of(pos) - cfg.world.radius - (d.z_hi - ph.ceiling_band)) / ph.ceiling_band,
            0.0,
            1.0,
        )
        keep = 1.0 - depth * depth * (3.0 - 2.0 * depth)  # 1 below the band, 0 at the lid
        out = (f_thrust * up).sum(axis=-1)
        f_thrust = f_thrust - ((1.0 - keep) * mx.maximum(out, 0.0))[:, None] * up
        buoy = buoy * keep
    u = nbr.off / mx.maximum(nbr.dist, layout.EPS_LEN)[..., None]
    rep = mx.where(contact[..., None], (-ph.k_repel * (two_r - nbr.dist))[..., None] * u, 0.0).sum(axis=1)
    f = f_thrust + rep + bonds.spring_forces(pos, partner, act.contract, cfg)
    f = f + (buoy - ph.gravity)[:, None] * up
    if wind is not None:  # static: weather on
        f = f + wind
    return mx.where(survivor[:, None], f, 0.0)


def integrate_s(
    pos: mx.array,
    hdg: mx.array,
    heading: mx.array,
    pitch: mx.array,
    force: mx.array,
    grip: mx.array,
    turn: mx.array,
    dpitch: mx.array,
    on_floor: mx.array,
    survivor: mx.array,
    cfg: Config,
    drag_mult: mx.array | None = None,
) -> IntegratedS:
    """Explicit overdamped step on the sphere (§46.2, §46.3), the geometry of ``sphere.move``.

    Anisotropic drag as in 3D: the tangent part of the force is divided by ``γ_t = drag·(1 + grip_gain·grip)``
    on grounded rows (``on_floor``) and by ``drag`` elsewhere, the radial part always by ``γ_r = drag`` (grip
    never slows sinking); ``drag_mult`` multiplies both. ``‖v‖`` is clipped to ``v_max`` (each clip counted),
    then ``sphere.move`` carries the position along the great circle, clamps the radius to ``[r_lo, r_hi]``
    (the floor and the hard lid, zeroing the radial velocity of the clamped row, each clamp counted),
    transports the heading by the minimal rotation, turns it by ``turn`` about the new up and renormalises it.
    The derived scalar ``heading`` is the bearing of the new ``hdg`` (output only). Non-survivors keep
    ``pos``, ``hdg``, ``heading`` and ``pitch`` and get ``vel = 0``.
    """
    ph, d = cfg.physics, dims(cfg)
    gamma_t = ph.drag * (1.0 + ph.grip_gain * mx.where(on_floor, grip, 0.0))
    gamma_r = ph.drag
    if drag_mult is not None:  # static: structures on
        gamma_t = gamma_t * drag_mult
        gamma_r = ph.drag * drag_mult
    up = pos / sphere.radius_of(pos)[:, None]
    f_r = (force * up).sum(axis=-1)
    f_t = force - f_r[:, None] * up
    v = f_t / gamma_t[:, None] + (f_r / gamma_r)[:, None] * up
    speed = mx.sqrt((v * v).sum(axis=-1))
    clip = survivor & (speed > ph.v_max)
    v = mx.where(clip[:, None], v * (ph.v_max / mx.maximum(speed, ph.v_max))[:, None], v)
    v = mx.where(survivor[:, None], v, 0.0)
    p2, h2, vel, at_floor, at_lid = sphere.move(pos, hdg, v, mx.where(survivor, turn, 0.0), d.r_lo, d.r_hi)
    at_floor, at_lid = survivor & at_floor, survivor & at_lid
    sv = survivor[:, None]
    new_pos, new_hdg = mx.where(sv, p2, pos), mx.where(sv, h2, hdg)
    return IntegratedS(
        pos=new_pos,
        vel=mx.where(sv, vel, 0.0),
        hdg=new_hdg,
        heading=mx.where(survivor, sphere.bearing(new_hdg, new_pos), heading),
        pitch=mx.where(survivor, mx.clip(pitch + dpitch, -layout.PITCH_MAX, layout.PITCH_MAX), pitch),
        v_clips=clip.sum().astype(mx.int32),
        z_floor=at_floor.sum().astype(mx.int32),
        z_surface=at_lid.sum().astype(mx.int32),
    )


# --- the barrier of built material (DESIGN §38.4.2, R1 and R3 of §38.4b) -----------------------------------

BARRIER_EPS: float = 1e-9  # the floor of a norm in the direction unit vectors
BARRIER_G0: float = 0.05  # R3: the gradient norm at which the confidence that there is a surface is 1/2


def barrier_drag(force: mx.array, here: mx.array, grad: mx.array | None, cfg: Config) -> mx.array:
    """The drag multiplier of the material at each agent's cell, ``(N,)`` f32 in ``[1, 1 + barrier_gain]``.

    ``1 + barrier_gain·here`` (R1: ``here`` is in [0, 1] and ``validate`` bounds ``barrier_gain`` by 3, so
    the slowdown is at most 4 and a closed cell is always crossable). With ``exit_ease > 0`` (R3, the
    ``grad`` argument is then required) the factor is ``1 + barrier_gain·here·(1 - exit_ease·out)`` where
    ``out = max(0, -c·g)·s`` is the share of the ACTUAL push ``force`` (thrust, wind, repulsion and gravity
    together: the same ``F`` the integrator divides, so displacement is what counts, not intended motion)
    that points down the gradient of the material, i.e. out of dense material, with ``s = |g|/(|g| + g0)``
    the confidence that a surface is there (a plateau has ``s = 0`` and pays the plain drag). No ``where``.
    """
    sc = cfg.structures
    if sc.barrier_gain <= 0:  # static: no barrier
        return mx.ones_like(here)
    plain = sc.barrier_gain * here
    if sc.exit_ease <= 0:  # static: R3 off
        return 1.0 + plain
    assert grad is not None, "barrier_drag with exit_ease > 0 needs the material gradient"  # static
    fn = mx.sqrt((force * force).sum(axis=-1))
    gn = mx.sqrt((grad * grad).sum(axis=-1))
    cos = (force * grad).sum(axis=-1) / (mx.maximum(fn, BARRIER_EPS) * mx.maximum(gn, BARRIER_EPS))
    out = mx.maximum(-cos, 0.0) * (gn / (gn + BARRIER_G0))
    return 1.0 + plain * (1.0 - sc.exit_ease * out)
