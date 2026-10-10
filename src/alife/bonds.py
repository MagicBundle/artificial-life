"""Bonds: geometry, breaking, the formation handshake, child bonds, death cleanup and springs (§8).

``drop_dead``, ``add_child_bonds`` and ``symmetric_ok`` are WP0 hand-over functions; WP3 owns this module.
Every bond operation is a gather through the stored back-pointer ``rev`` (flat index ``j·B + r``).
"""

from __future__ import annotations

from typing import NamedTuple

import mlx.core as mx

from alife import layout, ops
from alife.config import Config


class BondGeom(NamedTuple):
    off: mx.array  # (N,B,2) minimum-image vector to the partner; 0 where empty
    length: mx.array  # (N,B)
    valid: mx.array  # (N,B) bool: slot occupied


def bond_geometry(pos: mx.array, partner: mx.array, cfg: Config) -> BondGeom:
    """Start-of-step bond offsets and lengths (§3 S3)."""
    raise NotImplementedError


def n_bonds(partner: mx.array) -> mx.array:
    """Occupied slots per row, (N,) int32."""
    raise NotImplementedError


def break_bonds(partner: mx.array, rev: mx.array, flag: mx.array) -> tuple[mx.array, mx.array, mx.array]:
    """OR-symmetric per-slot breaking (§8.2): ``(partner, rev, broken (N,B) bool)``."""
    raise NotImplementedError


def form_bonds(
    partner: mx.array, rev: mx.array, can_bond: mx.array, target: mx.array, key: mx.array
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Propose/accept/mutual handshake (§8.3): ``(partner, rev, formed (N,) bool, as_proposer (N,) bool)``."""
    raise NotImplementedError


def drop_dead(partner: mx.array, rev: mx.array, survivor: mx.array) -> tuple[mx.array, mx.array]:
    """Drop every bond with a non-surviving end, on both sides, by gather (§8.5)."""
    keep = (partner >= 0) & survivor[:, None] & survivor[mx.maximum(partner, 0)]
    return mx.where(keep, partner, -1), mx.where(keep, rev, -1)


def add_child_bonds(
    partner: mx.array, rev: mx.array, child_of: mx.array, want: mx.array, born_parent: mx.array
) -> tuple[mx.array, mx.array, mx.array]:
    """Bond each newborn to its parent (§8.4): ``(partner, rev, made (N,) bool)``.

    ``child_of[p]`` is the child row of parent ``p`` (−1 otherwise); ``born_parent[c]`` is the parent row of
    newborn ``c`` (−1 for other rows and immigrants); ``want`` is the parent's ``child_bond`` decision. A bond
    is made iff the parent wants it and has a free slot; it goes into the parent's first free slot and the
    child's slot 0 (newborn rows are empty). ``made`` marks the parents that made one.
    """
    slots = mx.arange(layout.B, dtype=mx.int32)[None, :]
    ff, has_free = ops.first_free(partner)
    made = (child_of >= 0) & want & has_free
    put_p = made[:, None] & (slots == ff[:, None])
    bp = mx.maximum(born_parent, 0)
    child = (born_parent >= 0) & made[bp]
    put_c = child[:, None] & (slots == 0)
    partner = mx.where(put_p, child_of[:, None], mx.where(put_c, born_parent[:, None], partner))
    rev = mx.where(put_p, 0, mx.where(put_c, ff[bp][:, None], rev))
    return partner, rev, made


def spring_forces(pos: mx.array, partner: mx.array, contract: mx.array, cfg: Config) -> mx.array:
    """Spring forces with brain-controlled rest lengths (§9), (N,2); exactly antisymmetric per pair."""
    raise NotImplementedError


def symmetric_ok(partner: mx.array, rev: mx.array, alive: mx.array) -> mx.array:
    """Device check of the §8.1 bond invariant and no duplicate partners per row; () bool."""
    n, b = partner.shape
    occ = partner >= 0
    j = mx.clip(partner, 0, n - 1)
    r = mx.clip(rev, 0, b - 1)
    flat = j * b + r
    back_p = partner.reshape(-1)[flat]
    back_r = rev.reshape(-1)[flat]
    rows = mx.arange(n, dtype=mx.int32)[:, None]
    slots = mx.arange(b, dtype=mx.int32)[None, :]
    good = (
        alive[:, None]
        & alive[j]
        & (partner < n)
        & (partner != rows)
        & (rev >= 0)
        & (rev < b)
        & (back_p == rows)
        & (back_r == slots)
    )
    slot_ok = mx.where(occ, good, rev < 0)
    same = (partner[:, :, None] == partner[:, None, :]) & occ[:, :, None] & ~mx.eye(b, dtype=mx.bool_)[None]
    return mx.all(slot_ok) & ~mx.any(same)
