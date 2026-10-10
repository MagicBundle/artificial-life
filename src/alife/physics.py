"""Overdamped physics with thrust, springs, grip and soft repulsion (§9). Owned by WP3."""

from __future__ import annotations

import mlx.core as mx

from alife import bonds, layout, ops
from alife.actions import Actions
from alife.config import Config
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
