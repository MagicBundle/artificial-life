"""Bonds: geometry, breaking, the formation handshake, child bonds, death cleanup and springs (§8).

``drop_dead``, ``add_child_bonds`` and ``symmetric_ok`` are WP0 hand-over functions; WP3 owns this module.
Every bond operation is a gather through the stored back-pointer ``rev`` (flat index ``j·B + r``).
"""

from __future__ import annotations

from typing import Any, NamedTuple

import mlx.core as mx

from alife import layout, ops
from alife.config import Config
from alife.ext.bonds import bonds_ext

__all__ = ["bonds_ext"]


class BondGeom(NamedTuple):
    off: mx.array  # (N,B,nd) minimum-image vector to the partner; 0 where empty
    length: mx.array  # (N,B)
    valid: mx.array  # (N,B) bool: slot occupied


def _partner_offsets(pos: mx.array, partner: mx.array, L: float) -> tuple[mx.array, mx.array, mx.array]:
    """``(off (N,B,nd), length (N,B), valid (N,B))``: minimum-image vector to each partner, 0 where empty.

    3D positions need no z-specific code: ``2·height ≤ L`` makes ``min_image`` the identity on z (§27.2).
    """
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
    """Spring forces with brain-controlled rest lengths (§9), (N,nd); exactly antisymmetric per pair.

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


# --- bonds that pay (DESIGN §40): the hooks of the registry row (ext/bonds.py) --------------------------

_I32, _F32 = mx.int32, mx.float32
ROLE_FORAGER, ROLE_GUARD = 1, 2  # State.role codes: 0 generalist (the founder state), 1 forager, 2 guard
I16_MAX = 32767


def role_code(role: mx.array) -> mx.array:
    """The brain-facing code of a role, (N,) f32: forager +1, guard −1, generalist 0."""
    return mx.where(role == ROLE_FORAGER, 1.0, mx.where(role == ROLE_GUARD, -1.0, 0.0)).astype(_F32)


def slot_kin(tag: mx.array, partner: mx.array, cfg: Config) -> mx.array:
    """The tag similarity of the two ends of every bond slot, (N, B) f32 in (0, 1], 0 on empty slots: the
    expression of ``sensing.neighbor_descriptors`` (``exp(−|Δtag|²/tag_scale²)``) gathered per slot."""
    dtag = tag[:, None, :] - tag[mx.maximum(partner, 0)]
    kin = mx.exp(-(dtag * dtag).sum(-1) / (cfg.sensing.tag_scale**2))
    return mx.where(partner >= 0, kin, 0.0)


def ext_inputs(s: Any, cfg: Config, ctx: Any) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """The thirteen brain inputs: own role code (N, 1), then per slot the partner's role code, the kin
    similarity and the last net flow (``bond_flow / share_rate``, clipped), each (N, B); 0 on empty slots."""
    valid = ctx.geom.valid
    pj = mx.maximum(s.partner, 0)
    code = role_code(s.role)
    role_in = mx.where(valid, code[pj], 0.0)
    kin_in = slot_kin(s.tag, s.partner, cfg)
    flow_in = mx.where(valid, mx.clip(s.bond_flow.astype(_F32) / cfg.bonds.share_rate, -1.0, 1.0), 0.0)
    return code[:, None], role_in, kin_in, flow_in


class BondFlows(NamedTuple):
    """The step's bonds intermediates for the counters and the observer channel (``_Flows.ext.bonds``)."""

    role0: mx.array  # (N,) int8: the start-of-step role (what the effects of the step read)
    role: mx.array  # (N,) int8: the role at the end of the step (S8 of the next step reads it)
    bond_flow: mx.array  # (N, B) int16: net received − sent per slot at S12, 0 where the slot is gone
    slots_active: mx.array  # () int32: slots with out > 0
    switched: mx.array  # (N,) bool: rows whose role changed this step
    shield: mx.array  # (N,) int16: round(256·guard), the protection fraction of the row as a victim
    shielded: mx.array  # () int32: attacks on a victim with guard > 0 (SHIELDED_BY_GUARD)


def state_after(
    s: Any,
    sv: mx.array,
    partner: mx.array,
    sl: Any,
    cfg: Config,
    dec: Any = None,
    guard: mx.array | None = None,
    shielded: mx.array | None = None,
) -> tuple[dict[str, mx.array], BondFlows]:
    """The new bonds fields of the survivors ``sv`` (S14), and the flows for the counters: dead rows are
    cleared to 0 and ``bond_flow`` is the net transfer of the slot, 0 where the bond is gone after S14 (a dead
    or released partner), so a reused row never inherits a memory (E15). ``partner`` is the post-S14 partner
    (``dt.partner``); ``sl`` the S12 slots. With the role machine (``dec``, the solvency-masked
    ``roles.RoleDec``) a survivor takes its requested role: ``switched`` resets ``role_dwell`` to 0, else it
    counts up to ``ramp_steps``; without it (``dec`` None) the roles stay as they are (all generalists)."""
    zero = (s.role * 0).astype(mx.int8)
    if dec is None:  # static: no role machine
        role = mx.where(sv, s.role, 0).astype(mx.int8)
        dwell = mx.where(sv, s.role_dwell, 0).astype(mx.int16)
        switched = sv & ~sv
    else:
        switched = sv & (dec.req != s.role)
        role = mx.where(sv, dec.req, 0).astype(mx.int8)
        grown = mx.minimum(s.role_dwell.astype(_I32) + 1, cfg.roles.ramp_steps)
        dwell = mx.where(sv & ~switched, grown, 0).astype(mx.int16)
    net = mx.clip(sl.got - sl.out, -I16_MAX, I16_MAX)
    flow = mx.where((partner >= 0) & sv[:, None], net, 0).astype(mx.int16)
    shield = mx.round(256.0 * guard).astype(mx.int16) if guard is not None else zero.astype(mx.int16)
    flows = BondFlows(
        s.role,
        role,
        flow,
        (sl.out > 0).astype(_I32).sum(),
        switched,
        shield,
        shielded if shielded is not None else zero.astype(_I32).sum(),
    )
    return {"role": role, "role_dwell": dwell, "bond_flow": flow}, flows


class BondActs(NamedTuple):
    """``Acts.ext.bonds`` (observer channel, DESIGN §40.5): the start-of-step role per row, the net flow per
    slot and the guard shield of the row as a victim."""

    role: mx.array  # (N,) int8
    bond_flow: mx.array  # (N, B) int16: net received − sent per slot
    shield: mx.array  # (N,) int16: round(256·guard)


def acts_of(f: BondFlows) -> BondActs:
    return BondActs(f.role0, f.bond_flow, f.shield)


def ext_count(s_new: Any, alive0: mx.array, flows: Any, cfg: Config) -> dict[str, mx.array]:
    """The counter block (names in ``ext.bonds._COUNTERS``). The population counts are of the rows alive at
    the end of the step."""
    f = flows.ext.bonds
    alive = s_new.alive
    return {
        "ROLE_SWITCHES": f.switched.astype(_I32).sum(),
        "N_FORAGERS": (alive & (s_new.role == ROLE_FORAGER)).astype(_I32).sum(),
        "N_GUARDS": (alive & (s_new.role == ROLE_GUARD)).astype(_I32).sum(),
        "SHIELDED_BY_GUARD": f.shielded,
        "SHARE_SLOTS_ACTIVE": f.slots_active,
    }
