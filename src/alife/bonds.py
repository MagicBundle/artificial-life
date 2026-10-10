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


def _partner_offsets(pos: mx.array, partner: mx.array, L: float) -> tuple[mx.array, mx.array, mx.array]:
    """``(off (N,B,2), length (N,B), valid (N,B))``: minimum-image vector to each partner, 0 where empty."""
    valid = partner >= 0
    off = ops.min_image(pos[mx.maximum(partner, 0)] - pos[:, None, :], L)
    off = mx.where(valid[..., None], off, 0.0)
    return off, mx.sqrt((off * off).sum(-1)), valid


def bond_geometry(pos: mx.array, partner: mx.array, cfg: Config) -> BondGeom:
    """Start-of-step bond offsets and lengths (§3 S3); empty slots have ``off = 0`` and ``length = 0``."""
    off, length, valid = _partner_offsets(pos, partner, cfg.world.size)
    return BondGeom(off, length, valid)


def n_bonds(partner: mx.array) -> mx.array:
    """Occupied slots per row, (N,) int32."""
    return (partner >= 0).sum(axis=1).astype(mx.int32)


def _back_gather(a: mx.array, partner: mx.array, rev: mx.array) -> mx.array:
    """``a[partner[i,b], rev[i,b]]`` for every slot (garbage on empty slots; callers mask)."""
    b = partner.shape[1]
    flat = mx.maximum(partner, 0) * b + mx.maximum(rev, 0)
    return a.reshape(-1)[flat]


def break_bonds(partner: mx.array, rev: mx.array, flag: mx.array) -> tuple[mx.array, mx.array, mx.array]:
    """OR-symmetric per-slot breaking (§8.2): ``(partner, rev, broken (N,B) bool)``.

    A slot breaks if either end flags it, so both rows clear the bond in the same step whatever the
    floating-point details behind the flags.
    """
    valid = partner >= 0
    brk = valid & (flag | _back_gather(flag, partner, rev))
    return mx.where(brk, -1, partner), mx.where(brk, -1, rev), brk


def form_bonds(
    partner: mx.array, rev: mx.array, can_bond: mx.array, target: mx.array, key: mx.array
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Propose/accept/mutual handshake (§8.3): ``(partner, rev, formed (N,) bool, as_proposer (N,) bool)``.

    ``can_bond = alive & solvent & bond_want`` and ``target = act.bond_target`` (−1 = none); ``key`` is
    ``ks[BOND_PRIO]``. Each agent gains at most one bond, written into its own first free slot; the partner's
    slot is learned by gathering its ``first_free``. Proposals to self or to a current partner are dropped
    (a no-op under the targeting contract, kept as a guard for the §8.1 invariant).
    """
    n = partner.shape[0]
    rows = mx.arange(n, dtype=mx.int32)
    ff, has_free = ops.first_free(partner)
    can = can_bond & has_free
    tgt = mx.where(can & (target >= 0), target, -1)
    t0 = mx.maximum(tgt, 0)
    ok = (tgt >= 0) & can[t0] & (tgt != rows) & ~(partner == tgt[:, None]).any(axis=1)
    tgt = mx.where(ok, tgt, -1)
    t0 = mx.maximum(tgt, 0)
    prop = tgt >= 0
    prio = mx.random.permutation(n, key=key).astype(mx.int32)  # unique priorities in [0, N) (F4)
    best = ops.scatter_max_i32(n, tgt, prio, prop)  # best proposer priority per target; −1 if none
    won = prop & (best[t0] == prio)
    winner = ops.scatter_max_i32(n, tgt, rows, won)  # ≤ 1 winner per target: a placement
    can_accept = can & ~won  # a winning proposer never accepts
    mutual = (tgt[t0] == rows) & won[t0]
    as_prop = won & (can_accept[t0] | mutual)
    as_acc = can_accept & (winner >= 0)
    new_p = mx.where(as_prop, tgt, mx.where(as_acc, winner, -1))
    put = (new_p >= 0)[:, None] & (mx.arange(partner.shape[1], dtype=mx.int32)[None, :] == ff[:, None])
    partner = mx.where(put, new_p[:, None], partner)
    rev = mx.where(put, ff[mx.maximum(new_p, 0)][:, None], rev)
    return partner, rev, new_p >= 0, as_prop


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
    """Spring forces with brain-controlled rest lengths (§9), (N,2); exactly antisymmetric per pair.

    ``F_i = Σ_b k_s·(d_ib − ℓ_ib)·û_ib`` with ``ℓ_ib = L0·(1 + contract_gain·(c_i + c_j)/2)``. Both ends
    evaluate the same expression on bitwise-negated offsets (``min_image`` is odd-symmetric) and a commuted
    sum of contractions, so ``F_i + F_j == 0`` bitwise for an isolated pair.
    """
    bc = cfg.bonds
    off, d, valid = _partner_offsets(pos, partner, cfg.world.size)
    c_j = contract[mx.maximum(partner, 0)]
    rest = bc.rest_length * (1.0 + bc.contract_gain * ((contract[:, None] + c_j) * 0.5))
    u = off / mx.maximum(d, layout.EPS_LEN)[..., None]
    f = mx.where(valid[..., None], (bc.k_spring * (d - rest))[..., None] * u, 0.0)
    return f.sum(axis=1)


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
