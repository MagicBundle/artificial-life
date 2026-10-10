"""Sensing: neighbour descriptors and the D_IN = 69 input vector (§5). Owned by WP2."""

from __future__ import annotations

import mlx.core as mx

from alife.bonds import BondGeom
from alife.config import Config
from alife.fields import FieldObs
from alife.spatial import Neighbors
from alife.state import State


def neighbor_descriptors(s: State, nbr: Neighbors, cfg: Config) -> mx.array:
    """Descriptor ``desc (N,K,NF)`` f32 for all K neighbours, zeroed where invalid (§5)."""
    raise NotImplementedError


def build_inputs(
    s: State,
    age: mx.array,
    nbr: Neighbors,
    desc: mx.array,
    geom: BondGeom,
    fobs: FieldObs,
    key: mx.array,
    cfg: Config,
) -> mx.array:
    """Brain inputs ``x (N,D_IN)`` f32 with ``|x| ≤ 1`` (§5)."""
    raise NotImplementedError
