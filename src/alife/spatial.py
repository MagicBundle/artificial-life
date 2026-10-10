"""Fixed-K neighbour search on a periodic hash grid in pure MLX (§4). Owned by WP1."""

from __future__ import annotations

from typing import NamedTuple

import mlx.core as mx

from alife.config import Config


class Neighbors(NamedTuple):
    idx: mx.array  # (N,K) int32; invalid entries hold the agent's own index
    valid: mx.array  # (N,K) bool
    off: mx.array  # (N,K,2) f32 minimum-image vector me → neighbour; 0 where invalid
    dist: mx.array  # (N,K) f32; R where invalid
    overflow_cells: mx.array  # () int32: hash cells with count > C
    max_occupancy: mx.array  # () int32: largest hash-cell count


def find_neighbors(pos: mx.array, alive: mx.array, cfg: Config) -> Neighbors:
    """K nearest alive agents within ``sensing.radius``, sorted by distance (§4)."""
    raise NotImplementedError
