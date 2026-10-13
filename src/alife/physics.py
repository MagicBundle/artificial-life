"""Overdamped physics with thrust, springs, grip and soft repulsion (§9); the 3D variants add gravity, the
buoyancy force, the floor and the surface (§27.3). Owned by WP3."""

from __future__ import annotations

from typing import NamedTuple

import mlx.core as mx

from alife import bonds, layout, ops
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
) -> mx.array:
    """Total force per agent (N,2): contact-attenuated thrust, springs and repulsion (§9).

    Evaluated at start-of-step ``pos``/``heading`` with the post-S14 ``partner``. A contact is a valid
    neighbour that survives and is closer than ``2r``; thrust is scaled by
    ``max(0, 1 − n_contacts/thrust_contact_limit)`` (F18). Non-survivors get 0.
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
    return mx.where(survivor[:, None], f, 0.0)


def integrate(
    pos: mx.array,
    heading: mx.array,
    force: mx.array,
    grip: mx.array,
    turn: mx.array,
    survivor: mx.array,
    cfg: Config,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Explicit overdamped step for survivors: ``(pos, vel, heading, n_clipped () int32)`` (§9).

    ``v = F/γ`` with ``γ = drag·(1 + grip_gain·grip)``, clipped to ``‖v‖ ≤ v_max`` (each clip counted);
    ``pos ← wrap(pos + v)``, ``heading ← wrap_angle(heading + turn)``. Non-survivors keep ``pos`` and
    ``heading`` and get ``vel = 0``.
    """
    assert pos.shape[-1] == 2, "integrate got 3D positions (integrate3)"  # static: z would be wrapped
    ph = cfg.physics
    gamma = ph.drag * (1.0 + ph.grip_gain * grip)
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
    start-of-step position. The surface never grounds.
    """
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
) -> mx.array:
    """Total force per agent (N,3) in a 3D world (§27.3): ``forces`` plus gravity and buoyancy.

    Thrust acts along the body's forward axis (``ops.body_axes``) and is contact-attenuated as in 2D (F18);
    repulsion and springs use the (N,K,3) and (N,B,3) offsets. Gravity ``−gravity·ẑ`` and the buoyancy
    ``buoyancy_max·buoy_cap·mo.buoy·ẑ`` are body forces on z only and are not attenuated. Non-survivors get 0.
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
) -> Integrated3:
    """Explicit overdamped 3D step for survivors (§27.3).

    Anisotropic drag: ``γ_xy = drag·(1 + grip_gain·grip)`` on grounded rows (``on_floor``, from ``grounded``)
    and ``drag`` elsewhere; ``γ_z = drag`` always, so grip never slows sinking. ``‖v‖`` (3D) is clipped to
    ``v_max`` as in 2D. ``pos ← wrap_xy_clamp_z(pos + v)``: an agent that would cross the floor or the
    surface moves to it, and its stored ``v_z`` is zeroed (only on those rows; each clamp counted). Yaw as in
    2D; ``pitch ← clip(pitch + dpitch, ±PITCH_MAX)``. Non-survivors keep ``pos``, ``heading`` and ``pitch``
    and get ``vel = 0``.
    """
    ph, d = cfg.physics, dims(cfg)
    gamma_xy = ph.drag * (1.0 + ph.grip_gain * mx.where(on_floor, grip, 0.0))
    v = mx.concatenate([force[:, :2] / gamma_xy[:, None], force[:, 2:] / ph.drag], axis=1)
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
