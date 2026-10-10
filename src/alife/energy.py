"""Energy: costs, harvest, combat, sharing and the cap (§10.3–§10.7, §11). Owned by WP3."""

from __future__ import annotations

from typing import NamedTuple

import mlx.core as mx

from alife.actions import Actions
from alife.config import Config


class Costs(NamedTuple):
    energy: mx.array  # (N,) int32 after paying
    solvent: mx.array  # (N,) bool
    paid_basal: mx.array  # (N,) int32
    paid_action: mx.array  # (N,) int32


class Combat(NamedTuple):
    energy: mx.array  # (N,) int32
    resource: mx.array  # (FG,FG) int32
    lost: mx.array  # (N,) int32
    gain: mx.array  # (N,) int32
    spill: mx.array  # (N,) int32
    attack: mx.array  # (N,) bool: attackers after the S9-partner gate
    hit: mx.array  # (N,) bool: grant > 0
    max_attackers: mx.array  # () int32


class Share(NamedTuple):
    energy: mx.array  # (N,) int32
    out: mx.array  # (N,) int32, summed over slots
    inflow: mx.array  # (N,) int32
    heat: mx.array  # (N,) int32, summed over slots


def pay_costs(
    energy: mx.array, act: Actions, n_bonds: mx.array, brain_cost: mx.array, alive: mx.array, cfg: Config
) -> Costs:
    """Basal and action costs with solvency (§10.3)."""
    raise NotImplementedError


def harvest(
    energy: mx.array, resource: mx.array, fcell: mx.array, act: Actions, solvent: mx.array, cfg: Config
) -> tuple[mx.array, mx.array, mx.array]:
    """Proportional harvest from the agent's field cell: ``(E, resource, got)`` (§10.4)."""
    raise NotImplementedError


def combat(
    energy: mx.array, resource: mx.array, fcell: mx.array, act: Actions, partner: mx.array, cfg: Config
) -> Combat:
    """Simultaneous theft with spill into the victim's cell; ``partner`` is post-S9 (§11)."""
    raise NotImplementedError


def share(
    energy: mx.array,
    partner: mx.array,
    rev: mx.array,
    partner0: mx.array,
    act: Actions,
    solvent: mx.array,
    cfg: Config,
) -> Share:
    """Per-slot gated transfers along bonds with 90% efficiency (§10.6)."""
    raise NotImplementedError


def cap_energy(energy: mx.array, cfg: Config) -> tuple[mx.array, mx.array]:
    """``E ≤ e_max``; the excess goes to heat: ``(E, overflow (N,))`` (§10.7)."""
    raise NotImplementedError
