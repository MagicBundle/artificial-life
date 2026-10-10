"""Births, deaths and the deferred weight flush (§10.7, §12). Owned by WP4."""

from __future__ import annotations

from enum import IntEnum
from typing import NamedTuple

import mlx.core as mx

from alife.config import Config
from alife.state import State


class LaneKind(IntEnum):
    PAD = 0
    BIRTH = 1
    IMMIGRANT = 2


class DeathRecords(NamedTuple):  # (DL,) each, DL = 2M
    uid: mx.array
    age: mx.array
    cause: mx.array
    generation: mx.array
    clade: mx.array
    valid: mx.array


class BirthRecords(NamedTuple):  # (M,) each
    uid: mx.array
    parent_uid: mx.array
    clade: mx.array
    generation: mx.array
    slot: mx.array
    valid: mx.array  # bool: lane is a birth or an immigrant this step


class Deaths(NamedTuple):
    survivor: mx.array
    dead: mx.array
    energy: mx.array
    uid: mx.array
    partner: mx.array
    rev: mx.array
    resource: mx.array
    corpse: mx.array
    r_overflow: mx.array
    cause: mx.array
    records: DeathRecords


class Lanes(NamedTuple):
    parent: mx.array  # (M,)
    slot: mx.array  # (M,)
    kind: mx.array  # (M,) LaneKind
    n_req: mx.array  # ()
    n_birth: mx.array  # ()
    n_imm: mx.array  # ()


class Repro(NamedTuple):
    state: State
    records: BirthRecords
    endow: mx.array
    n_child_bonds: mx.array


def flush_pending(
    W1: mx.array, W2: mx.array, pend_slot: mx.array, pend_W1: mx.array, pend_W2: mx.array
) -> tuple[mx.array, mx.array]:
    """S0: write last step's child weight rows in place via ``ops.set_rows`` (§12.6)."""
    raise NotImplementedError


def deaths(
    alive: mx.array,
    solvent: mx.array,
    energy: mx.array,
    lost: mx.array,
    age: mx.array,
    uid: mx.array,
    generation: mx.array,
    clade: mx.array,
    partner: mx.array,
    rev: mx.array,
    resource: mx.array,
    fcell: mx.array,
    cfg: Config,
) -> Deaths:
    """S14: ``dead = alive & (~solvent | E == 0 | aged)``; corpses, canonical rows, bonds, records (§10.7)."""
    raise NotImplementedError


def allocate_lanes(req: mx.array, survivor: mx.array, key: mx.array, cfg: Config) -> Lanes:
    """Assign up to M birth/immigrant lanes to distinct dead rows under capacity pressure (§12.2)."""
    raise NotImplementedError


def reproduce(s: State, lanes: Lanes, child_bond: mx.array, ks: mx.array, cfg: Config) -> Repro:
    """S16: write newborn rows via ``ops.place``, debit parents, add child bonds, set ``pend_*`` (§12)."""
    raise NotImplementedError
