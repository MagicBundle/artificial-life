"""Births, deaths and the deferred weight flush (§10.7, §12). Owned by WP4.

Every contested write is an int32 scatter in ``ops``; newborn rows are written by gather (``ops.place``)
through int32 row → lane maps, so ``uid`` (int64) never needs a scatter (§12.3, D4). Child weights are
buffered in ``pend_*`` and written at S0 of the next step (§12.6, F1).
"""

from __future__ import annotations

import math
from enum import IntEnum
from typing import NamedTuple

import mlx.core as mx

from alife import bonds, brain, layout, ops
from alife.config import Config, dims
from alife.rng import Stream
from alife.state import R_HARD, DeathCause, State

_I32 = mx.int32


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
    survivor: mx.array  # (N,) bool: alive after S14
    dead: mx.array  # (N,) bool: died this step
    energy: mx.array  # (N,) int32, 0 on dead rows
    uid: mx.array  # (N,) int64, −1 on dead rows
    partner: mx.array  # (N,B) bonds with a dead end dropped on both sides
    rev: mx.array  # (N,B)
    resource: mx.array  # (FG,FG) int32 after corpses and the R_HARD cap
    corpse: mx.array  # (N,) int32: E + S on dead rows, else 0 (FLOW_CORPSE)
    r_overflow: mx.array  # (FG,FG) int32 removed by the hard cap (HEAT_OVERFLOW)
    cause: mx.array  # (N,) int32 DeathCause on dead rows, −1 elsewhere
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
    return ops.set_rows(W1, pend_slot, pend_W1), ops.set_rows(W2, pend_slot, pend_W2)


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
    """S14: ``dead = alive & (~solvent | E == 0 | aged)``; corpses, canonical rows, bonds, records (§10.7).

    Cause, first match wins: ``STARVED`` if insolvent, ``KILLED`` if ``E == 0`` after a loss this step,
    ``STARVED`` if ``E == 0``, else ``AGED``. Corpses (``E + S``) are scatter-added into the dead agent's
    field cell, then the field is hard-capped at ``R_HARD``. Death records hold the first ``DL`` dead rows by
    index.
    """
    d = dims(cfg)
    zero = energy == 0
    dead = alive & (~solvent | zero)
    if cfg.population.max_age > 0:  # static: senescence enabled
        dead = dead | (alive & (age >= cfg.population.max_age))
    survivor = alive & ~dead

    cause = mx.where(
        ~solvent,
        DeathCause.STARVED.value,
        mx.where(
            zero & (lost > 0),
            DeathCause.KILLED.value,
            mx.where(zero, DeathCause.STARVED.value, DeathCause.AGED.value),
        ),
    )
    cause = mx.where(dead, cause, -1).astype(_I32)

    corpse = mx.where(dead, energy + cfg.energy.structure, 0).astype(_I32)
    fg2 = d.fg * d.fg
    r = resource.reshape(-1) + ops.scatter_add_i32(fg2, fcell, corpse, dead)
    r_over = mx.maximum(r - R_HARD, 0)
    r = r - r_over

    partner, rev = bonds.drop_dead(partner, rev, survivor)

    idx, valid = ops.rank_select(dead, d.dl)
    records = DeathRecords(
        uid=mx.where(valid, uid[idx], -1),
        age=mx.where(valid, age[idx], 0),
        cause=mx.where(valid, cause[idx], 0),
        generation=mx.where(valid, generation[idx], 0),
        clade=mx.where(valid, clade[idx], 0),
        valid=valid,
    )
    return Deaths(
        survivor=survivor,
        dead=dead,
        energy=mx.where(dead, 0, energy),
        uid=mx.where(dead, -1, uid),
        partner=partner,
        rev=rev,
        resource=r.reshape(d.fg, d.fg),
        corpse=corpse,
        r_overflow=r_over.reshape(d.fg, d.fg),
        cause=cause,
        records=records,
    )


def allocate_lanes(req: mx.array, survivor: mx.array, key: mx.array, cfg: Config) -> Lanes:
    """Assign up to M birth/immigrant lanes to distinct dead rows under capacity pressure (§12.2).

    Birth lanes are a prefix, served in a uniformly random order of the requesters; immigrant lanes (only
    when ``immigration_floor > 0``) follow; the rest are padding. Every lane, padding included, targets one of
    the M lowest-index dead rows, which exist because the population never exceeds ``N_CAP = N − M``.
    Non-birth lanes use parent row 0.
    """
    d = dims(cfg)
    n, m = d.n, d.m
    perm = mx.random.permutation(n, key=key).astype(_I32)
    ppos, _ = ops.rank_select(req[perm], m)
    slot, _ = ops.rank_select(~survivor, m)
    n_surv = survivor.astype(_I32).sum()
    n_req = req.astype(_I32).sum()
    room = mx.maximum(d.n_cap - n_surv, 0)  # ≥ 0 whenever the cap held at the start of the step
    n_birth = mx.minimum(mx.minimum(n_req, m), room)
    pop = cfg.population
    if pop.immigration_floor > 0:  # static: immigration enabled
        want = mx.minimum(
            mx.minimum(pop.immigration_floor - n_surv - n_birth, pop.immigrants_per_step),
            mx.minimum(m - n_birth, room - n_birth),
        )
        n_imm = mx.clip(want, 0, m).astype(_I32)
    else:
        n_imm = mx.array(0, dtype=_I32)
    k = mx.arange(m, dtype=_I32)
    kind = mx.where(
        k < n_birth,
        LaneKind.BIRTH.value,
        mx.where(k < n_birth + n_imm, LaneKind.IMMIGRANT.value, LaneKind.PAD.value),
    ).astype(_I32)
    parent = mx.where(kind == LaneKind.BIRTH.value, perm[ppos], 0).astype(_I32)
    return Lanes(parent=parent, slot=slot, kind=kind, n_req=n_req, n_birth=n_birth.astype(_I32), n_imm=n_imm)


def _logit(p: mx.array) -> mx.array:
    return mx.log(p / (1.0 - p))


def _child_genes(s: State, p: mx.array, key: mx.array, cfg: Config) -> tuple[mx.array, ...]:
    """Mutated (σ, p, tag, frac) of each lane's parent row (§12.4); f32, (M,) / (M,T)."""
    mu, m = cfg.mutation, p.shape[0]
    k_sigma, k_rate, k_tag, k_frac = mx.random.split(key, 4)
    n1 = mx.random.normal((m,), key=k_sigma)
    n2 = mx.random.normal((m,), key=k_rate)
    n3 = mx.random.normal((m, layout.T), key=k_tag)
    n4 = mx.random.normal((m,), key=k_frac)
    sigma = mx.clip(s.mut_sigma[p] * mx.exp(mu.tau_sigma * n1), mu.sigma_min, mu.sigma_max)
    rate = mx.clip(mx.sigmoid(_logit(s.mut_rate[p]) + mu.tau_rate * n2), mu.rate_min, mu.rate_max)
    tag = mx.clip(s.tag[p] + mu.tag_sigma * n3, -1.0, 1.0)
    frac = mx.clip(mx.sigmoid(_logit(s.offspring_frac[p]) + mu.tau_frac * n4), mu.frac_min, mu.frac_max)
    return sigma, rate, tag, frac


def reproduce(s: State, lanes: Lanes, child_bond: mx.array, ks: mx.array, cfg: Config) -> Repro:
    """S16: write newborn rows via ``ops.place``, debit parents, add child bonds, set ``pend_*`` (§12).

    ``s`` is the in-flight state (``alive`` = survivors, post-physics ``pos``/``heading``, post-S14 bonds and
    energy, post-flush weights, ``step`` = this step's index). ``child_bond`` is the parents' (N,) decision.
    Returns the state with newborn rows, debited parents, child bonds, ``pend_*`` and ``next_uid`` advanced;
    the lane records; the per-lane endowment (0 off birth lanes, FLOW_ENDOWMENT); and the child-bond count.
    """
    d = dims(cfg)
    n, m = d.n, d.m
    en, mu, pop = cfg.energy, cfg.mutation, cfg.population
    L = cfg.world.size
    k = mx.arange(m, dtype=_I32)
    birth = lanes.kind == LaneKind.BIRTH.value
    imm = lanes.kind == LaneKind.IMMIGRANT.value
    used = birth | imm
    p, slot = lanes.parent, lanes.slot

    born_lane = ops.scatter_max_i32(n, slot, k, used)  # row → lane, −1 if not a newborn
    child_of = ops.scatter_max_i32(n, p, slot, birth)  # parent row → child row, −1 otherwise
    newborn = born_lane >= 0
    lane_of = mx.maximum(born_lane, 0)

    # Energy (§10.7): Dp = E − birth_cost − S ≥ 28 000 for eligible parents; endow = (Dp·frac_q8) >> 8.
    dp = s.energy[p] - cfg.repro.birth_cost - en.structure
    frac_q8 = mx.floor(s.offspring_frac[p] * 256.0 + 0.5).astype(_I32)
    endow = mx.where(birth, (dp * frac_q8) >> 8, 0).astype(_I32)
    parent_lane = born_lane[mx.maximum(child_of, 0)]
    debit = cfg.repro.birth_cost + en.structure + endow[mx.maximum(parent_lane, 0)]
    energy = mx.where(child_of >= 0, s.energy - debit, s.energy)
    energy = ops.place(energy, newborn, born_lane, mx.where(imm, en.e_init, endow).astype(_I32))

    # Genes, placement and weights per lane. Padding lanes compute mutated copies of row 0 (§12.6).
    sigma, rate, tag, frac = _child_genes(s, p, ks[Stream.MUT_GENES], cfg)
    phi = s.heading[p] + math.pi + mx.random.uniform(-0.5, 0.5, (m,), key=ks[Stream.PLACEMENT])
    step_vec = cfg.repro.offspring_distance * mx.stack([mx.cos(phi), mx.sin(phi)], axis=1)
    pos = ops.wrap(s.pos[p] + step_vec, L)
    heading = ops.wrap_angle(phi)
    w1, w2 = brain.mutate_rows(s.W1[p], s.W2[p], sigma, rate, ks[Stream.MUT_W1], ks[Stream.MUT_W2], cfg)
    if pop.immigration_floor > 0:  # static: immigrant lanes draw fresh founders (§12.7)
        k_pos, k_head, k_tag, k_w1, k_w2 = mx.random.split(ks[Stream.IMMIGRANT], 5)
        i_pos = ops.wrap(mx.random.uniform(0.0, L, (m, 2), key=k_pos), L)
        i_head = ops.wrap_angle(mx.random.uniform(-math.pi, math.pi, (m,), key=k_head))
        i_tag = mx.random.uniform(-1.0, 1.0, (m, layout.T), key=k_tag)
        i_w1, i_w2 = brain.init_weights(k_w1, k_w2, m, cfg)
        pos = mx.where(imm[:, None], i_pos, pos)
        heading = mx.where(imm, i_head, heading)
        tag = mx.where(imm[:, None], i_tag, tag)
        sigma = mx.where(imm, mu.sigma_init, sigma)
        rate = mx.where(imm, mu.rate_init, rate)
        frac = mx.where(imm, mu.frac_init, frac)
        w1 = mx.where(imm[:, None, None], i_w1, w1)
        w2 = mx.where(imm[:, None, None], i_w2, w2)
    brain_cost = brain.l1_cost(w1, w2, cfg)

    uid = s.next_uid + k.astype(mx.int64)
    parent_uid = mx.where(birth, s.uid[p], -1)
    clade = mx.where(birth, s.clade[p], pop.n_init).astype(_I32)
    generation = mx.where(birth, s.generation[p] + 1, 0).astype(_I32)

    def put(arr: mx.array, lane_vals: mx.array) -> mx.array:
        return ops.place(arr, newborn, born_lane, lane_vals.astype(arr.dtype))

    def clear(arr: mx.array, value: float | int = 0) -> mx.array:
        nb = newborn.reshape(newborn.shape + (1,) * (arr.ndim - 1))
        return mx.where(nb, mx.array(value, dtype=arr.dtype), arr)

    # Child bonds (§8.4): newborn rows start empty, the child side uses slot 0.
    born_parent = mx.where(newborn, mx.where(birth, p, -1)[lane_of], -1)
    partner, rev, made = bonds.add_child_bonds(
        clear(s.partner, -1), clear(s.rev, -1), child_of, child_bond, born_parent
    )
    n_new = (lanes.n_birth + lanes.n_imm).astype(mx.int64)
    state = s._replace(
        alive=s.alive | newborn,
        uid=put(s.uid, uid),
        parent_uid=put(s.parent_uid, parent_uid),
        clade=put(s.clade, clade),
        generation=put(s.generation, generation),
        birth_step=mx.where(newborn, s.step, s.birth_step),
        energy=energy,
        pos=put(s.pos, pos),
        vel=clear(s.vel),
        heading=put(s.heading, heading),
        h=clear(s.h),
        msg=clear(s.msg),
        last_loss=clear(s.last_loss),
        last_harvest=clear(s.last_harvest),
        last_inflow=clear(s.last_inflow),
        partner=partner,
        rev=rev,
        tag=put(s.tag, tag),
        mut_sigma=put(s.mut_sigma, sigma),
        mut_rate=put(s.mut_rate, rate),
        offspring_frac=put(s.offspring_frac, frac),
        brain_cost=put(s.brain_cost, brain_cost),
        pend_slot=slot,
        pend_W1=w1,
        pend_W2=w2,
        next_uid=s.next_uid + n_new,
    )
    records = BirthRecords(
        uid=mx.where(used, uid, -1),
        parent_uid=mx.where(used, parent_uid, -1),
        clade=mx.where(used, clade, 0),
        generation=mx.where(used, generation, 0),
        slot=mx.where(used, slot, -1),
        valid=used,
    )
    return Repro(state=state, records=records, endow=endow, n_child_bonds=made.astype(_I32).sum())
