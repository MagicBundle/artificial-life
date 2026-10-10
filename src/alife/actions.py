"""Decoding brain outputs into actions, including query-based target selection (§7). Owned by WP2."""

from __future__ import annotations

from typing import NamedTuple

import mlx.core as mx

from alife.config import Config
from alife.spatial import Neighbors


class Actions(NamedTuple):
    """Decoded actions, all (N, ...) and already masked by ``alive``."""

    thrust: mx.array  # (N,) f32 in (−1, 1)
    turn: mx.array  # (N,) f32 radians
    msg: mx.array  # (N,MSG) f32
    emit: mx.array  # (N,NCH) f32 in [0, 1)
    harvest: mx.array  # (N,) f32 in (0, 1)
    grip: mx.array  # (N,) f32 in (0, 1)
    contract: mx.array  # (N,) f32 in (−1, 1)
    bond_want: mx.array  # (N,) bool
    release: mx.array  # (N,B) bool
    share: mx.array  # (N,B) f32 in [0, 1)
    attack: mx.array  # (N,) bool
    defend: mx.array  # (N,) f32 in (0, 1)
    reproduce: mx.array  # (N,) bool
    child_bond: mx.array  # (N,) bool
    bond_k: mx.array  # (N,) int32, −1 = none
    bond_target: mx.array  # (N,) int32, −1 = none
    attack_k: mx.array  # (N,) int32, −1 = none
    attack_target: mx.array  # (N,) int32, −1 = none


def decode(
    y: mx.array, desc: mx.array, nbr: Neighbors, partner: mx.array, alive: mx.array, cfg: Config
) -> Actions:
    """Decode raw logits ``y (N,D_OUT)`` and select bond/attack targets (§7)."""
    raise NotImplementedError


def mask(act: Actions, keep: mx.array) -> Actions:
    """Cancel actions where ``~keep``: bools → False, floats → 0, k/targets → −1."""
    raise NotImplementedError
