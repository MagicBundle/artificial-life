"""The pure, compiled simulation step (§3, §16.3). Owned by WP4.

``step`` wires the sub-steps S0–S18 in the normative order of §3; ``_finish`` (S18) assembles the exact
ledger, the per-step counters (including the kin counters) and the diag bits. There are no host syncs and no
Python branches on array values: the only branches are on config constants.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import mlx.core as mx

from alife import actions, bonds, brain, fields, invariants, layout, lifecycle, ops, physics, sensing, spatial
from alife import energy as en
from alife.actions import Actions
from alife.config import Config, dims
from alife.lifecycle import BirthRecords, DeathRecords
from alife.rng import Stream, step_keys
from alife.state import N_COUNTERS, CounterIdx, DeathCause, LedgerIdx, State, age, total_energy

__all__ = ["BirthRecords", "DeathRecords", "StepOut", "make_step", "step"]

_I32 = mx.int32
_I64 = mx.int64


class StepOut(NamedTuple):
    counters: mx.array  # (N_COUNTERS,) int32, CounterIdx (§19.1)
    ledger: mx.array  # (16,) int64, this step's ledger (LedgerIdx, §10.8)
    births: BirthRecords
    deaths: DeathRecords


class _Flows(NamedTuple):
    """Intermediate results of S0–S17 that S18 turns into the ledger, counters and diag."""

    costs: en.Costs
    got: mx.array  # (N,) harvest
    cb: en.Combat
    sh: en.Share
    e_over: mx.array  # (N,) cap overflow
    dt: lifecycle.Deaths
    lanes: lifecycle.Lanes
    rp: lifecycle.Repro
    regen_inc: mx.array  # (FG,FG)
    nbr: spatial.Neighbors
    v_clips: mx.array  # ()
    fcell: mx.array  # (N,) start-of-step field cell
    desc: mx.array  # (N,K,NF)
    x: mx.array  # (N,D_IN)
    act: Actions  # masked by alive and solvency
    as_prop: mx.array  # (N,) bool
    formed: mx.array  # (N,) bool
    broken: mx.array  # (N,B) bool, S9a
    slots_s9: mx.array  # () occupied slots after S9 (before S14)


def step(s: State, cfg: Config) -> tuple[State, StepOut]:
    """Advance the world by one step (S0–S18). Pure; no host syncs."""
    t = s.step
    W1, W2 = lifecycle.flush_pending(s.W1, s.W2, s.pend_slot, s.pend_W1, s.pend_W2)  # S0
    s = s._replace(W1=W1, W2=W2)
    key_next, ks = step_keys(s.key)  # S1
    a = age(t, s.birth_step)
    nbr = spatial.find_neighbors(s.pos, s.alive, cfg)  # S2
    geom = bonds.bond_geometry(s.pos, s.partner, cfg)  # S3
    fcell = fields.cell_index(s.pos, cfg)  # S4
    fobs = fields.sample_fields(s.resource, s.chem, fcell, s.heading, cfg)
    desc = sensing.neighbor_descriptors(s, nbr, cfg)  # S5
    x = sensing.build_inputs(s, a, nbr, desc, geom, fobs, ks[Stream.NOISE], cfg)
    h_new, y = brain.forward(s.W1, s.W2, x, s.h, cfg)  # S6
    act = actions.decode(y, desc, nbr, s.partner, s.alive, cfg)  # S7

    nb0 = bonds.n_bonds(s.partner)  # S8
    c = en.pay_costs(s.energy, act, nb0, s.brain_cost, s.alive, cfg)
    act = actions.mask(act, c.solvent)

    flag = act.release | (geom.length > cfg.bonds.break_length)  # S9
    partner, rev, broken = bonds.break_bonds(s.partner, s.rev, flag & geom.valid)
    partner, rev, formed, as_prop = bonds.form_bonds(
        partner, rev, s.alive & c.solvent & act.bond_want, act.bond_target, ks[Stream.BOND_PRIO]
    )
    slots_s9 = (partner >= 0).astype(_I32).sum()

    E, resource, got = en.harvest(c.energy, s.resource, fcell, act, c.solvent, cfg)  # S10
    cb = en.combat(E, resource, fcell, act, partner, cfg)  # S11
    sh = en.share(cb.energy, partner, rev, s.partner, act, c.solvent, cfg)  # S12
    E, e_over = en.cap_energy(sh.energy, cfg)  # S13
    dt = lifecycle.deaths(
        s.alive, c.solvent, E, cb.lost, a, s.uid, s.generation, s.clade, partner, rev, cb.resource, fcell, cfg
    )  # S14

    sv = dt.survivor  # S15
    F = physics.forces(s.pos, s.heading, act, nbr, sv, dt.partner, cfg)
    pos, vel, heading, v_clips = physics.integrate(s.pos, s.heading, F, act.grip, act.turn, sv, cfg)
    s1 = s._replace(
        alive=sv,
        energy=dt.energy,
        uid=dt.uid,
        partner=dt.partner,
        rev=dt.rev,
        resource=dt.resource,
        pos=pos,
        vel=vel,
        heading=heading,
        h=mx.where(sv[:, None], h_new, 0.0),
        msg=mx.where(sv[:, None], act.msg, 0.0),
        last_loss=mx.where(sv, cb.lost, 0),
        last_harvest=mx.where(sv, got, 0),
        last_inflow=mx.where(sv, sh.inflow, 0),
    )

    req = sv & act.reproduce & (dt.energy >= cfg.repro.min_energy) & (a >= cfg.repro.min_age)  # S16
    lanes = lifecycle.allocate_lanes(req, sv, ks[Stream.BIRTH_PRIO], cfg)
    rp = lifecycle.reproduce(s1, lanes, act.child_bond, ks, cfg)
    s2 = rp.state

    dep = fields.deposit_chem(act.emit, fcell, c.solvent, cfg)  # S17
    chem = fields.update_chem(s2.chem, dep, cfg)
    resource, regen_inc = fields.update_resource(s2.resource, s2.patch_pos, cfg)
    patch_pos = fields.drift_patches(s2.patch_pos, s2.patch_vel, cfg)
    s2 = s2._replace(chem=chem, resource=resource, patch_pos=patch_pos, key=key_next, step=t + 1)

    flows = _Flows(
        costs=c,
        got=got,
        cb=cb,
        sh=sh,
        e_over=e_over,
        dt=dt,
        lanes=lanes,
        rp=rp,
        regen_inc=regen_inc,
        nbr=nbr,
        v_clips=v_clips,
        fcell=fcell,
        desc=desc,
        x=x,
        act=act,
        as_prop=as_prop,
        formed=formed,
        broken=broken,
        slots_s9=slots_s9,
    )
    return _finish(s2, s, cfg, flows)  # S18


def _ledger(f: _Flows, cfg: Config) -> mx.array:
    """This step's (16,) int64 ledger from exact limb sums plus int64 count × constant terms (§10.8)."""
    S = cfg.energy.structure
    agent = mx.stack(
        [
            f.costs.paid_basal,
            f.costs.paid_action,
            f.got,
            f.cb.lost,
            f.cb.gain,
            f.cb.spill + f.cb.residue,
            f.sh.out,
            f.sh.heat,
            f.e_over,
            f.dt.corpse,
        ],
        axis=1,
    ).astype(_I32)
    a = ops.exact_sum(agent, axis=0)  # (10,) int64
    cell = mx.stack([f.regen_inc.reshape(-1), f.dt.r_overflow.reshape(-1)], axis=1).astype(_I32)
    c = ops.exact_sum(cell, axis=0)  # (2,) int64
    endow = ops.exact_sum(f.rp.endow)
    n_birth = f.lanes.n_birth.astype(_I64)
    n_imm = f.lanes.n_imm.astype(_I64)
    zero = mx.array(0, dtype=_I64)
    entries = {
        LedgerIdx.INJECT_REGEN: c[0],
        LedgerIdx.INJECT_IMMIGRATION: n_imm * (cfg.energy.e_init + S),
        LedgerIdx.HEAT_BASAL: a[0],
        LedgerIdx.HEAT_ACTION: a[1],
        LedgerIdx.HEAT_SHARE: a[7],
        LedgerIdx.HEAT_BIRTH: n_birth * cfg.repro.birth_cost,
        LedgerIdx.HEAT_OVERFLOW: a[8] + c[1],
        LedgerIdx.FLOW_HARVEST: a[2],
        LedgerIdx.FLOW_STOLEN: a[3],
        LedgerIdx.FLOW_ATTACK_GAIN: a[4],
        LedgerIdx.FLOW_SPILL: a[5],
        LedgerIdx.FLOW_SHARED: a[6],
        LedgerIdx.FLOW_CORPSE: a[9],
        LedgerIdx.FLOW_ENDOWMENT: endow,
        LedgerIdx.FLOW_STRUCTURE: n_birth * S,
        LedgerIdx.RESERVED: zero,
    }
    return mx.stack([entries[i].astype(_I64) for i in LedgerIdx])


def _kin_sum(desc: mx.array, k: mx.array, rows: mx.array) -> mx.array:
    """``Σ_rows round(256·desc[i, k_i, NBR_KIN])`` as int32 (``k ≥ 0`` on every selected row)."""
    kin = mx.take_along_axis(desc[..., layout.NBR_KIN], mx.maximum(k, 0)[:, None], axis=1)[:, 0]
    return mx.where(rows, mx.round(256.0 * kin), 0.0).astype(_I32).sum()


def _counters(s_new: State, alive0: mx.array, f: _Flows, cfg: Config) -> mx.array:
    """The (26,) int32 per-step counters of §19.1; ``alive0`` is the start-of-step ``alive``."""
    d = dims(cfg)
    act, dt, cb = f.act, f.dt, f.cb

    def count(mask: mx.array) -> mx.array:
        return mask.astype(_I32).sum()

    kin_mean = mx.round(256.0 * f.x[:, layout.IN_KIN_MEAN])
    n_dead = count(dt.dead)
    slots_s14 = count(dt.partner >= 0)
    emit_q = ops.floor_i32(act.emit * 256.0)  # the deposit quanta of §13.3 (act is masked by solvency)
    entries = {
        CounterIdx.N_ALIVE: count(s_new.alive),
        CounterIdx.BIRTHS: f.lanes.n_birth,
        CounterIdx.BIRTH_REQUESTS: f.lanes.n_req,
        CounterIdx.BIRTHS_BLOCKED: f.lanes.n_req - f.lanes.n_birth,
        CounterIdx.IMMIGRANTS: f.lanes.n_imm,
        CounterIdx.DEATHS_STARVED: count(dt.cause == DeathCause.STARVED.value),
        CounterIdx.DEATHS_KILLED: count(dt.cause == DeathCause.KILLED.value),
        CounterIdx.DEATHS_AGED: count(dt.cause == DeathCause.AGED.value),
        CounterIdx.DEATHS_UNLOGGED: mx.maximum(n_dead - d.dl, 0),
        CounterIdx.INSOLVENT: count(alive0 & ~f.costs.solvent),
        CounterIdx.ATTACKS: count(cb.attack),
        CounterIdx.ATTACK_HITS: count(cb.hit),
        CounterIdx.BONDS_FORMED: count(f.formed) // 2,
        CounterIdx.BONDS_BROKEN: (count(f.broken) + f.slots_s9 - slots_s14) // 2,
        CounterIdx.CHILD_BONDS: f.rp.n_child_bonds,
        CounterIdx.BONDS_TOTAL: count(s_new.partner >= 0) // 2,
        CounterIdx.SHARERS: count(f.sh.out > 0),
        CounterIdx.EMITTERS: count((emit_q > 0).any(axis=1)),
        CounterIdx.CELL_OVERFLOW: f.nbr.overflow_cells,
        CounterIdx.MAX_OCCUPANCY: f.nbr.max_occupancy,
        CounterIdx.V_CLIPS: f.v_clips,
        CounterIdx.KIN_ATTACK_SUM: _kin_sum(f.desc, act.attack_k, cb.attack),
        CounterIdx.KIN_ATTACK_BASE: mx.where(cb.attack, kin_mean, 0.0).astype(_I32).sum(),
        CounterIdx.KIN_BOND_SUM: _kin_sum(f.desc, act.bond_k, f.as_prop),
        CounterIdx.KIN_BOND_BASE: mx.where(f.as_prop, kin_mean, 0.0).astype(_I32).sum(),
        CounterIdx.BOND_PROPOSERS: count(f.as_prop),
    }
    assert len(entries) == N_COUNTERS
    return mx.stack([entries[i].astype(_I32) for i in CounterIdx])


def _finish(s_new: State, s: State, cfg: Config, f: _Flows) -> tuple[State, StepOut]:
    """S18: ledger, cumulative ledger, ``total``, counters and diag (§16.3)."""
    d = dims(cfg)
    ledger = _ledger(f, cfg)
    total = total_energy(s_new.energy, s_new.alive, s_new.resource, cfg)
    s_new = s_new._replace(total=total, ledger_cum=s.ledger_cum + ledger)
    counters = _counters(s_new, s.alive, f, cfg)
    if cfg.run.runtime_checks != "off":  # static
        max_cell_occ = ops.scatter_add_i32(d.fg * d.fg, f.fcell, 1, s.alive).max()
        bits = invariants.device_check(
            s_new,
            ledger,
            s.total,
            max_cell_occ,
            f.cb.max_attackers,
            f.nbr.overflow_cells,
            f.v_clips,
            cfg,
        )
        s_new = s_new._replace(diag=s.diag | bits)
    out = StepOut(counters=counters, ledger=ledger, births=f.rp.records, deaths=f.dt.records)
    return s_new, out


def make_step(cfg: Config, *, compile: bool | None = None) -> Callable[[State], tuple[State, StepOut]]:
    """The step closed over ``cfg``, compiled once when ``compile`` (default ``cfg.run.compile``)."""
    use_compile = cfg.run.compile if compile is None else compile

    def fn(s: State) -> tuple[State, StepOut]:
        return step(s, cfg)

    return mx.compile(fn) if use_compile else fn
