"""Energy: costs, harvest, combat, sharing and the cap (§10.3–§10.7, §11). Owned by WP3."""

from __future__ import annotations

from typing import NamedTuple

import mlx.core as mx

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


class Share(NamedTuple):
    energy: mx.array  # (N,) int32
    out: mx.array  # (N,) int32, summed over slots
    inflow: mx.array  # (N,) int32
    heat: mx.array  # (N,) int32, summed over slots


def pay_costs(
    energy: mx.array, act: Actions, n_bonds: mx.array, brain_cost: mx.array, alive: mx.array, cfg: Config
) -> Costs:
    """Basal and action costs with solvency (§10.3).

    ``n_bonds`` counts start-of-step bonds. Each float term is floored in f32. Insolvent agents
    (``alive & E < basal_due + action_due``) pay everything they hold and end at ``E = 0``; dead rows pay 0.
    Ledger: ``E_in − E_out = paid_basal + paid_action`` (``HEAT_BASAL``, ``HEAT_ACTION``).
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
    energy: mx.array, resource: mx.array, fcell: mx.array, act: Actions, solvent: mx.array, cfg: Config
) -> tuple[mx.array, mx.array, mx.array]:
    """Proportional harvest from the agent's field cell: ``(E, resource, got)`` (§10.4).

    ``demand = clip(min(floor(harvest·harvest_max), e_max − E), 0, harvest_max)`` for solvent agents;
    contested cells are shared by ``ops.proportional_take``. Ledger: ``Σ got`` moves from ``resource`` to
    ``E`` (``FLOW_HARVEST``).
    """
    ec = cfg.energy
    want = mx.minimum(ops.floor_i32(act.harvest * ec.harvest_max), ec.e_max - energy)
    demand = mx.where(solvent, mx.clip(want, 0, ec.harvest_max), 0)
    take, taken = ops.proportional_take(demand, fcell, solvent, resource.reshape(-1))
    return energy + take, resource - taken.reshape(resource.shape), take


def combat(
    energy: mx.array, resource: mx.array, fcell: mx.array, act: Actions, partner: mx.array, cfg: Config
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
    """
    cc = cfg.combat
    n = energy.shape[0]
    tgt = act.attack_target
    attack = act.attack & (tgt >= 0)
    if not cc.attack_bonded:
        attack = attack & ~((partner == tgt[:, None]) & (partner >= 0)).any(axis=1)
    t0 = mx.maximum(tgt, 0)
    want = ops.floor_i32(cc.attack_max * (1.0 - cc.defend_efficiency * act.defend[t0]))
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
    )


def share(
    energy: mx.array,
    partner: mx.array,
    rev: mx.array,
    partner0: mx.array,
    act: Actions,
    solvent: mx.array,
    cfg: Config,
) -> Share:
    """Per-slot gated transfers along bonds with 90% efficiency (§10.6).

    A slot sends ``min(floor(share·share_rate), E // B)`` only if the sender is solvent and the slot's partner
    is unchanged since the start of the step (``partner == partner0``). The receiver gathers the sender's
    gated ``out`` through ``rev`` and keeps ``mul_q8(in, q8(share_efficiency))``. Ledger: ``Σ out`` is
    ``FLOW_SHARED`` and ``Σ (in − g)`` is ``HEAT_SHARE``; ``E`` stays ≥ 0 by the ``// B`` cap.
    """
    bc = cfg.bonds
    valid = partner >= 0
    gate = valid & (partner == partner0) & solvent[:, None]
    amount = mx.minimum(ops.floor_i32(act.share * bc.share_rate), (energy // layout.B)[:, None])
    out = mx.where(gate, amount, 0)
    b = partner.shape[1]
    inc = mx.where(valid, out.reshape(-1)[mx.maximum(partner, 0) * b + mx.maximum(rev, 0)], 0)
    g = ops.mul_q8(inc, ops.q8(bc.share_efficiency))
    out_sum, g_sum = out.sum(axis=1), g.sum(axis=1)
    return Share(energy - out_sum + g_sum, out_sum, g_sum, (inc - g).sum(axis=1))


def cap_energy(energy: mx.array, cfg: Config) -> tuple[mx.array, mx.array]:
    """``E ≤ e_max``; the excess goes to heat: ``(E, overflow (N,))`` (§10.7, ``HEAT_OVERFLOW``)."""
    overflow = mx.maximum(energy - cfg.energy.e_max, 0)
    return energy - overflow, overflow
