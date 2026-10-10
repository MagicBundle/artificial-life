"""Overdamped physics with thrust, springs, grip and soft repulsion (§9). Owned by WP3."""

from __future__ import annotations

import mlx.core as mx

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
    """Total force per agent (N,2): contact-attenuated thrust, springs and repulsion (§9)."""
    raise NotImplementedError


def integrate(
    pos: mx.array,
    heading: mx.array,
    force: mx.array,
    grip: mx.array,
    turn: mx.array,
    survivor: mx.array,
    cfg: Config,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Explicit overdamped step for survivors: ``(pos, vel, heading, n_clipped () int32)`` (§9)."""
    raise NotImplementedError
