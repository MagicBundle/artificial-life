"""Energy: costs, harvest, combat, sharing and the cap (§10.3–§10.7, §11). Owned by WP3."""

from __future__ import annotations

from typing import NamedTuple

import mlx.core as mx

from alife import arms as arms_mod
from alife import layout, ops
from alife.actions import Actions, Motion3
from alife.config import Config

_I32 = mx.int32


class Costs(NamedTuple):
    energy: mx.array  # (N,) int32 after paying
    solvent: mx.array  # (N,) bool
    paid_basal: mx.array  # (N,) int32
    paid_action: mx.array  # (N,) int32


class Costs3(NamedTuple):
    """``Costs`` plus the buoyancy payment (3D, §27.7); a parallel type, so ``Costs`` keeps four fields."""

    energy: mx.array  # (N,) int32 after paying
    solvent: mx.array  # (N,) bool
    paid_basal: mx.array  # (N,) int32
    paid_action: mx.array  # (N,) int32
    paid_buoy: mx.array  # (N,) int32: buoyancy use and cap upkeep (HEAT_BUOYANCY)


class Combat(NamedTuple):
    energy: mx.array  # (N,) int32
    resource: mx.array  # (FG,FG) int32; 3D (FZ,FG,FG)
    lost: mx.array  # (N,) int32
    gain: mx.array  # (N,) int32
    spill: mx.array  # (N,) int32, per attacker
    residue: mx.array  # (N,) int32, per fully drained victim: the remainder also spilled into its cell
    attack: mx.array  # (N,) bool: attackers after the S9-partner gate
    hit: mx.array  # (N,) bool: grant > 0
    max_attackers: mx.array  # () int32
    # Protection (DESIGN §43.6), None while no protection factor is passed; observer and counter inputs.
    prot: mx.array | None = None  # (N,) f32 per attacker: the protection product after the floor
    q: mx.array | None = None  # (N,) int32 per attacker: Q8 efficiency (arms only; else the scalar ``q8``)
    capped: mx.array | None = None  # (N,) bool: attackers whose product was raised to ``protect_floor``
    shielded: mx.array | None = None  # (N,) bool: attackers against a victim with shelter or guard > 0


class Share(NamedTuple):
    energy: mx.array  # (N,) int32
    out: mx.array  # (N,) int32, summed over slots
    inflow: mx.array  # (N,) int32
    heat: mx.array  # (N,) int32, summed over slots


def pay_costs(
    energy: mx.array,
    act: Actions,
    n_bonds: mx.array,
    brain_cost: mx.array,
    alive: mx.array,
    cfg: Config,
    extra_due: mx.array | None = None,
) -> Costs:
    """Basal and action costs with solvency (§10.3).

    ``n_bonds`` counts start-of-step bonds. Each float term is floored in f32. Insolvent agents
    (``alive & E < basal_due + action_due``) pay everything they hold and end at ``E = 0``; dead rows pay 0.
    Ledger: ``E_in − E_out = paid_basal + paid_action`` (``HEAT_BASAL``, ``HEAT_ACTION``).

    ``extra_due`` (DESIGN §43.4) is the sum of the feature dues, an (N,) int32 each already floored by its own
    module (``ext.sum_due``); it joins the action due after the legacy terms, so solvency, payment order and
    ``HEAT_ACTION`` stay the one path. None (every feature off) traces nothing.
    """
    ec, ph = cfg.energy, cfg.physics
    f = ops.floor_i32
    basal_due = ec.basal_cost + brain_cost + ec.bond_upkeep * n_bonds
    action_due = (
        f(ec.move_cost * (act.thrust * act.thrust))
        + f(ec.turn_cost * mx.abs(act.turn) / ph.turn_max)
        + f(ec.grip_cost * act.grip)
        + f(ec.contract_cost * mx.abs(act.contract) * n_bonds.astype(mx.float32))
        + f(ec.msg_cost * mx.abs(act.msg).sum(axis=1))
        + f(ec.emit_cost * act.emit.sum(axis=1))
        + f(ec.harvest_cost * act.harvest)
        + f(ec.defend_cost * act.defend)
        + ec.attack_cost * act.attack.astype(_I32)
    )
    if extra_due is not None:  # static
        action_due = action_due + extra_due
    basal_due = mx.where(alive, basal_due, 0).astype(_I32)
    action_due = mx.where(alive, action_due, 0).astype(_I32)
    solvent = alive & (energy >= basal_due + action_due)
    paid_basal = mx.minimum(basal_due, energy)
    paid_action = mx.minimum(action_due, energy - paid_basal)
    return Costs(energy - paid_basal - paid_action, solvent, paid_basal, paid_action)


def pay_costs3(
    energy: mx.array,
    act: Actions,
    mo: Motion3,
    n_bonds: mx.array,
    brain_cost: mx.array,
    buoy_cap: mx.array,
    alive: mx.array,
    cfg: Config,
    extra_due: mx.array | None = None,
) -> Costs3:
    """3D basal, action and buoyancy costs with solvency (§27.7); ``pay_costs`` is the 2D form.

    The dues are §10.3's except the turn term, ``floor(turn_cost·(|turn| + |Δφ|)/turn_max)`` with
    ``Δφ = mo.pitch``, plus the buoyancy due
    ``floor(buoyancy_cost·(buoy_cap·b)²) + floor(buoyancy_cap_cost·buoy_cap)`` with ``b = mo.buoy``: the
    quadratic use of the lift (``buoy_cap·b`` is its fraction of ``buoyancy_max``) and the cap's upkeep, paid
    every step even at ``b = 0``. Without the actuator (``buoyancy_max = 0``, static) the buoyancy terms are
    not traced and the due is 0. Solvency is on the total; payment order is basal, action, buoyancy. Dead
    rows pay 0. Ledger: ``E_in − E_out = paid_basal + paid_action + paid_buoy`` (``HEAT_BASAL``,
    ``HEAT_ACTION``, ``HEAT_BUOYANCY``). With ``Δφ = 0`` and ``buoy_cap = 0`` it equals ``pay_costs``.
    """
    ec, ph = cfg.energy, cfg.physics
    f = ops.floor_i32
    basal_due = ec.basal_cost + brain_cost + ec.bond_upkeep * n_bonds
    action_due = (
        f(ec.move_cost * (act.thrust * act.thrust))
        + f(ec.turn_cost * (mx.abs(act.turn) + mx.abs(mo.pitch)) / ph.turn_max)
        + f(ec.grip_cost * act.grip)
        + f(ec.contract_cost * mx.abs(act.contract) * n_bonds.astype(mx.float32))
        + f(ec.msg_cost * mx.abs(act.msg).sum(axis=1))
        + f(ec.emit_cost * act.emit.sum(axis=1))
        + f(ec.harvest_cost * act.harvest)
        + f(ec.defend_cost * act.defend)
        + ec.attack_cost * act.attack.astype(_I32)
    )
    if extra_due is not None:  # static: the feature dues (``pay_costs``)
        action_due = action_due + extra_due
    if ph.buoyancy_max > 0:  # static: the actuator exists
        lift = buoy_cap * mo.buoy  # the fraction of buoyancy_max used
        buoy_due = f(ec.buoyancy_cost * (lift * lift)) + f(ec.buoyancy_cap_cost * buoy_cap)
    else:  # 0; ``paid_buoy`` below derives from ``energy`` either way, never a constant output (I14)
        buoy_due = energy * 0
    basal_due = mx.where(alive, basal_due, 0).astype(_I32)
    action_due = mx.where(alive, action_due, 0).astype(_I32)
    buoy_due = mx.where(alive, buoy_due, 0).astype(_I32)
    solvent = alive & (energy >= basal_due + action_due + buoy_due)
    paid_basal = mx.minimum(basal_due, energy)
    paid_action = mx.minimum(action_due, energy - paid_basal)
    paid_buoy = mx.minimum(buoy_due, energy - paid_basal - paid_action)
    return Costs3(energy - paid_basal - paid_action - paid_buoy, solvent, paid_basal, paid_action, paid_buoy)


def harvest(
    energy: mx.array,
    resource: mx.array,
    fcell: mx.array,
    act: Actions,
    solvent: mx.array,
    cfg: Config,
    harvest_cap: mx.array | None = None,
) -> tuple[mx.array, mx.array, mx.array]:
    """Proportional harvest from the agent's field cell: ``(E, resource, got)`` (§10.4).

    ``demand = clip(min(floor(harvest·harvest_max), e_max − E), 0, harvest_max)`` for solvent agents;
    contested cells are shared by ``ops.proportional_take``. Ledger: ``Σ got`` moves from ``resource`` to
    ``E`` (``FLOW_HARVEST``). ``harvest_cap`` ((N,) int32, the roles of DESIGN §40.3) replaces the constant
    ``harvest_max`` per row in ``want`` and the final clip; it must stay below 2^16 (validated). None is
    today's graph (static).
    """
    ec = cfg.energy
    hm = ec.harvest_max if harvest_cap is None else harvest_cap
    want = mx.minimum(ops.floor_i32(act.harvest * hm), ec.e_max - energy)
    demand = mx.where(solvent, mx.clip(want, 0, hm), 0)
    take, taken = ops.proportional_take(demand, fcell, solvent, resource.reshape(-1))
    return energy + take, resource - taken.reshape(resource.shape), take


def protection(
    t0: mx.array,
    floor: float,
    *,
    arms_p: mx.array | None = None,
    shelter: mx.array | None = None,
    guard: mx.array | None = None,
    pierce: mx.array | None = None,
) -> tuple[mx.array | None, mx.array | None]:
    """The protection product ``P`` of each attacker's demand and the rows the floor raised (DESIGN §43.6).

    ``P`` is evaluated left to right in f32 in a fixed order: ``arms_p`` (the per-attacker armour, key/lock
    and
    toxin factor ``(1 − r)·K·T`` of the arms contest, already gathered), then ``shelter`` and ``guard`` (the
    victim's protection FRACTIONS in ``[0, 1)``, gathered at ``t0``), each as ``1 − fraction·(1 − pierce)``.
    ``pierce`` (per attacker, ``[0, 1)``) lifts the environmental protections only. The floor then bounds the
    product from below: ``P = max(P, floor)``, ``capped = P < floor``. A factor is traced only when passed
    (static), so with none the result is ``(None, None)`` and ``combat`` traces today's expression; a factor
    equal to exactly 1.0 multiplies exactly. ``floor == 0`` traces no cap (``capped`` is None).
    """
    p: mx.array | None = arms_p
    for frac in (shelter, guard):
        if frac is None:
            continue
        keep = 1.0 - frac[t0] if pierce is None else 1.0 - frac[t0] * (1.0 - pierce)
        p = keep if p is None else p * keep
    if p is None:
        return None, None
    if floor > 0:  # static
        return mx.maximum(p, floor), p < floor
    return p, None


def combat(
    energy: mx.array,
    resource: mx.array,
    fcell: mx.array,
    act: Actions,
    partner: mx.array,
    cfg: Config,
    *,
    arms: mx.array | None = None,
    shelter: mx.array | None = None,
    guard: mx.array | None = None,
) -> Combat:
    """Simultaneous theft with spill into the victim's cell; ``partner`` is post-S9 (§11).

    An attack whose target became a bond partner at S9 is dropped unless ``attack_bonded`` (its cost was paid
    at S8). Demand ``floor(attack_max·(1 − defend_efficiency·defend[target]))`` is served proportionally from
    the victims' post-harvest energy; 70% of each grant reaches the attacker and the rest spills into the
    field at the victim's cell. A victim whose attackers together demand at least all it holds is drained to
    exactly 0: the few quanta that ``proportional_take`` leaves in a contested pool (its −1 guard) are its
    ``residue``, spilled into its own cell, so it dies ``KILLED`` at S14 unless bond inflow rescues it.
    Ledger: ``Σ lost = Σ gain + Σ spill + Σ residue`` (``FLOW_STOLEN``, ``FLOW_ATTACK_GAIN``, ``FLOW_SPILL``
    = spill + residue); ``spill`` and ``gain`` are per attacker, ``lost`` and ``residue`` per victim.

    The one signature of the protection series (DESIGN §43.6). ``shelter`` and ``guard`` are (N,) f32 victim
    protection fractions (already scaled by their gains by the supplying module); the demand is multiplied by
    their product (``protection``), floored at ``combat.protect_floor``, before the behavioural ``defend``.
    With
    all three None (every feature off) the traced graph is the legacy one. ``arms`` (``State.arms``) adds the
    arms contest factor ``(1 − r)·K`` of the attacker against its target (``arms.contest_factor``, DESIGN
    §42.2) as the first factor of the product; it is the float 1.0 for zero traits and ``key_floor = 1``.
    """
    cc = cfg.combat
    n = energy.shape[0]
    tgt = act.attack_target
    attack = act.attack & (tgt >= 0)
    if not cc.attack_bonded:
        attack = attack & ~((partner == tgt[:, None]) & (partner >= 0)).any(axis=1)
    t0 = mx.maximum(tgt, 0)
    arms_p = None if arms is None else arms_mod.contest_factor(arms, t0, partner, cfg)  # static
    prot, capped = protection(t0, cc.protect_floor, arms_p=arms_p, shelter=shelter, guard=guard)
    if prot is None:  # static: nothing protects (every feature off)
        want = ops.floor_i32(cc.attack_max * (1.0 - cc.defend_efficiency * act.defend[t0]))
    else:
        want = ops.floor_i32(cc.attack_max * prot * (1.0 - cc.defend_efficiency * act.defend[t0]))
    want = mx.where(attack, mx.clip(want, 0, cc.attack_max), 0)
    grant, taken = ops.proportional_take(want, t0, attack, energy)
    demand = ops.scatter_add_i32(n, t0, want, attack)  # total demand per victim
    residue = mx.where(demand >= energy, energy - taken, 0)  # 0 when uncontested (taken == demand)
    lost = taken + residue
    gain = ops.mul_q8(grant, ops.q8(cc.efficiency))
    spill = grant - gain
    dep = ops.scatter_add_i32(resource.size, fcell[t0], spill, attack) + ops.scatter_add_i32(
        resource.size, fcell, residue
    )
    max_attackers = ops.scatter_add_i32(n, t0, 1, attack).max()
    return Combat(
        energy=energy - lost + gain,
        resource=resource + dep.reshape(resource.shape),
        lost=lost,
        gain=gain,
        spill=spill,
        residue=residue,
        attack=attack,
        hit=attack & (grant > 0),
        max_attackers=max_attackers,
        prot=prot,
        capped=None if capped is None else attack & capped,
        shielded=_shielded(t0, attack, shelter, guard),
    )


def _shielded(
    t0: mx.array, attack: mx.array, shelter: mx.array | None, guard: mx.array | None
) -> mx.array | None:
    """Attackers whose victim has a shelter or guard fraction > 0 (None when neither is passed)."""
    if shelter is None and guard is None:
        return None
    on = mx.zeros(attack.shape, dtype=mx.bool_)
    for frac in (shelter, guard):
        if frac is not None:
            on = on | (frac[t0] > 0.0)
    return attack & on


class ShareSlots(NamedTuple):
    """The per-slot flows of S12 (§10.6); ``share_from_slots`` sums them over the slots."""

    out: mx.array  # (N,B) int32, gated amount sent per slot
    inc: mx.array  # (N,B) int32, gross received per slot (the partner's ``out`` gathered through ``rev``)
    got: mx.array  # (N,B) int32, net received per slot: ``mul_q8(inc, q8(share_efficiency))``


def share_slots(
    energy: mx.array,
    partner: mx.array,
    rev: mx.array,
    partner0: mx.array,
    act: Actions,
    solvent: mx.array,
    cfg: Config,
    kin: mx.array | None = None,
) -> ShareSlots:
    """Per-slot gated transfers along bonds (§10.6; the conductive modes are DESIGN §40.2), before the slot
    sums.

    ``bonds.share_mode`` 0 (the legacy gift): a slot sends ``min(floor(share·share_rate), E // B)`` only if
    the sender is solvent and the slot's partner is unchanged since the start of the step (``partner ==
    partner0``). The receiver gathers the sender's gated ``out`` through ``rev`` and keeps
    ``mul_q8(in, q8(share_efficiency))``.

    Modes 1 and 2 (static) replace the amount by an equalising, antisymmetric flow: with the energies at the
    start of S12 (both ends read the same two values), only the richer end of a bond sends,
    ``min((gap·q8(share_kappa)) >> 8, cap, E // B)`` with ``gap = E_i − E_j > 0``. Mode 1 needs the consent of
    both ends (``s = floor(256·share) > 0`` at both, ``cap = (min(s_i, s_j)·share_rate) >> 8``); mode 2 is
    obligate (``cap = share_rate``, the share outputs are ignored). ``E // B`` per slot bounds a row's total
    send by ``E``: nobody goes below 0 on any bond graph (a ring or a star at full gap included), and
    ``share_kappa ≤ 1/(2B)`` bounds the receipts so that no row ends above its richest neighbour. The
    receiving end needs no solvency (a poor partner can be rescued). ``kin`` ((N, B) f32 in [0, 1], the
    tag similarity of the two ends) scales the efficiency when ``share_kin_floor < 1``:
    ``floor(256·share_efficiency·(floor + (1 − floor)·kin))``; the heat stays ``Σ(in − got)``.
    """
    bc = cfg.bonds
    valid = partner >= 0
    pj = mx.maximum(partner, 0)
    b = partner.shape[1]
    cap_e = (energy // layout.B)[:, None]
    if bc.share_mode == 0:  # static: the legacy gift
        gate = valid & (partner == partner0) & solvent[:, None]
        amount = mx.minimum(ops.floor_i32(act.share * bc.share_rate), cap_e)
        out = mx.where(gate, amount, 0)
    else:
        gate = valid & (partner == partner0) & solvent[:, None]
        gap = energy[:, None] - energy[pj]
        amount = mx.minimum((gap * ops.q8(bc.share_kappa)) >> 8, cap_e)
        if bc.share_mode == 1:  # static: mutual consent, the weaker of the two share outputs
            sq = mx.clip(ops.floor_i32(act.share * 256.0), 0, 255)
            sj = sq.reshape(-1)[pj * b + mx.maximum(rev, 0)]
            amount = mx.minimum(amount, (mx.minimum(sq, sj) * bc.share_rate) >> 8)
            gate = gate & (sq > 0) & (sj > 0)
        else:  # mode 2: obligate
            amount = mx.minimum(amount, bc.share_rate)
        out = mx.where(gate & (gap > 0), amount, 0)
    inc = mx.where(valid, out.reshape(-1)[pj * b + mx.maximum(rev, 0)], 0)
    if bc.share_mode != 0 and bc.share_kin_floor < 1.0:  # static: non-kin transfers lose more
        assert kin is not None, "share_kin_floor < 1 needs the per-slot kin similarity"
        f = bc.share_kin_floor
        eff = ops.floor_i32(256.0 * bc.share_efficiency * (f + (1.0 - f) * kin))
        g = ops.mul_q8(inc, eff)
    else:
        g = ops.mul_q8(inc, ops.q8(bc.share_efficiency))
    return ShareSlots(out, inc, g)


def share_from_slots(energy: mx.array, sl: ShareSlots) -> Share:
    """The slot sums of S12. Ledger: ``Σ out`` is ``FLOW_SHARED`` and ``Σ (in − g)`` is ``HEAT_SHARE``;
    ``E`` stays ≥ 0 by the ``// B`` cap."""
    out_sum, g_sum = sl.out.sum(axis=1), sl.got.sum(axis=1)
    return Share(energy - out_sum + g_sum, out_sum, g_sum, (sl.inc - sl.got).sum(axis=1))


def share(
    energy: mx.array,
    partner: mx.array,
    rev: mx.array,
    partner0: mx.array,
    act: Actions,
    solvent: mx.array,
    cfg: Config,
    kin: mx.array | None = None,
) -> Share:
    """Per-slot gated transfers along bonds with 90% efficiency (§10.6): ``share_from_slots`` of
    ``share_slots``. Ledger: ``Σ out`` is ``FLOW_SHARED`` and ``Σ (in − g)`` is ``HEAT_SHARE``."""
    return share_from_slots(energy, share_slots(energy, partner, rev, partner0, act, solvent, cfg, kin))


def cap_energy(energy: mx.array, cfg: Config) -> tuple[mx.array, mx.array]:
    """``E ≤ e_max``; the excess goes to heat: ``(E, overflow (N,))`` (§10.7, ``HEAT_OVERFLOW``)."""
    overflow = mx.maximum(energy - cfg.energy.e_max, 0)
    return energy - overflow, overflow
