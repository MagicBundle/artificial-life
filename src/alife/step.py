"""The pure, compiled simulation step (§3, §16.3). Owned by WP4.

``step`` wires the sub-steps S0–S18 in the normative order of §3; ``_finish`` (S18) assembles the exact
ledger, the per-step counters (including the kin counters) and the diag bits. There are no host syncs and no
Python branches on array values: the only branches are on config constants and on the static ``observe``
argument of ``step`` / ``make_step`` (the observer channel, §28.2).
"""

from __future__ import annotations

from collections.abc import Callable
from enum import IntFlag
from typing import Final, NamedTuple

import mlx.core as mx

from alife import (
    actions,
    arms,
    bonds,
    brain,
    ext,
    fields,
    invariants,
    layout,
    lifecycle,
    ops,
    physics,
    roles,
    sensing,
    spatial,
    structures,
    weather,
)
from alife import energy as en
from alife.actions import Actions
from alife.config import Config, dims
from alife.lifecycle import BirthRecords, DeathRecords
from alife.rng import Stream, step_keys
from alife.state import (
    N_COUNTERS,
    Counter3D,
    CounterIdx,
    DeathCause,
    Diag,
    Env,
    LedgerIdx,
    State,
    State3,
    age,
    total_energy,
)

__all__ = [
    "ActFlag",
    "Acts",
    "BirthRecords",
    "DeathRecords",
    "StepOut",
    "StepOutG",
    "make_step",
    "step",
]

_I32 = mx.int32
_I64 = mx.int64

KIN_Q_CLIP: Final = 127.0  # |y_kin| bound of ``Acts.q_kin``: round(256·127) = 32512 fits int16 (§29.2)


class StepOut(NamedTuple):
    # (n_counters(nd, weather),) int32: CounterIdx (§19.1), Counter3D in 3D (§27.8), DEATHS_STORM (§30)
    counters: mx.array
    ledger: mx.array  # (16,) int64, this step's ledger (LedgerIdx, §10.8)
    births: BirthRecords
    deaths: DeathRecords


class ActFlag(IntFlag):
    """The bits of ``Acts.flags`` (§28.2); each is the per-row form of one step counter."""

    ATTACKED = 1  # Combat.attack: attacked after the S9-partner gate (the ATTACKS counter)
    HIT = 2  # Combat.hit (ATTACK_HITS)
    FORMED = 4  # bond formed at S9, either end (2·BONDS_FORMED rows)
    BUDDED = 8  # parent of a birth lane at S16 (BIRTHS rows)
    EMITTED = 16  # some channel's deposit quanta floor(256·emit_c) > 0 (EMITTERS)
    PROPOSED = 32  # bonds.form_bonds as_prop: a proposal that formed a bond with bond_target (BOND_PROPOSERS)
    KILLED = 64  # died KILLED at S14 this step (DEATHS_KILLED)
    # The bits of the feature extensions, fixed by the extension index (``ext.ACT_FLAG_BASE << i``, DESIGN
    # §43.5) so that no feature PR edits this class; a feature's PR only sets its bit. 4096..32768 are free.
    CONTESTED = 128  # arms (index 0)
    BUILDING = 256  # structures (1)
    ROLE_SWITCHED = 512  # bonds (2)
    SPOKE = 1024  # comm (3)
    FARMING = 2048  # niche (4)


class ExtActs(NamedTuple):
    """The per-feature observer sub-tuples of ``Acts.ext`` (DESIGN §43.5): ``None`` while the feature is off,
    else a NamedTuple the feature defines in its own module. Replaces flat trailing ``Acts`` fields, so no two
    feature PRs edit the same lines of ``Acts`` and ``_acts``."""

    arms: NamedTuple | None = None
    structures: NamedTuple | None = None
    bonds: NamedTuple | None = None
    comm: NamedTuple | None = None
    niche: NamedTuple | None = None


class ExtFlows(NamedTuple):
    """The per-feature intermediates of S0–S17 that S18 turns into counters (``_Flows.ext``); like
    ``ExtActs``."""

    arms: NamedTuple | None = None
    structures: NamedTuple | None = None
    bonds: NamedTuple | None = None
    comm: NamedTuple | None = None
    niche: NamedTuple | None = None


class Acts(NamedTuple):
    """The observer channel (§28.2): this step's actions and flows per row, never fed back into the step."""

    attack_gain: mx.array  # (N,) int32: energy gained by attacking (Σ = FLOW_ATTACK_GAIN)
    share_out: mx.array  # (N,B) int32: sent per bond slot (Σ = FLOW_SHARED)
    share_got: mx.array  # (N,B) int32: net received per bond slot (Σ out − Σ got = HEAT_SHARE)
    flags: mx.array  # (N,) uint16: ActFlag bits (widened from uint8 for the extension bits, §43.5)
    thrust: mx.array  # (N,) int16: round(256·act.thrust) of the post-S8 actions
    defend: mx.array  # (N,) int16: round(256·act.defend) of the post-S8 actions
    emit: mx.array  # (N,) int16: Σ_c floor(256·act.emit_c), the S17 deposit quanta, in [0, 768]
    # Social tracking (§29.2), appended after G1's seven fields:
    attack_target: mx.array  # (N,) int32: post-S8 attack target row (start-of-step state), −1 none
    bond_target: mx.array  # (N,) int32: post-S8 bond target row, −1 none; the partner of a PROPOSED row
    q_kin: mx.array  # (N, 2) int16: round(256·clip(y_kin, ±KIN_Q_CLIP)), attack and bond query; 0 if dead
    nbr: mx.array  # (N, K) int32: sensed neighbour rows of the start-of-step state, −1 invalid
    born_row: mx.array  # (M,) int32: child row per lane, −1 padding
    born_parent: mx.array  # (M,) int32: parent row per birth lane, −1 for immigrant and padding lanes
    born_uid: mx.array  # (M,) int64: uid per lane, −1 padding
    # Roster (§29.2, stage 7), appended after the social-tracking fields:
    death_cause: mx.array  # (N,) int8: DeathCause of a row that died at S14 this step, −1 elsewhere
    # Storm world (§32), 3D only (None in 2D): the buoyancy use, appended after the roster field.
    buoy: mx.array | None = None  # (N,) int16: round(256·mo.buoy) of the pre-mask motion, 0 where not solvent
    # Feature extensions (§43.5): None while every feature is off, else an ``ExtActs`` with one sub-tuple per
    # enabled feature (None for a feature that is off).
    ext: ExtActs | None = None


class StepOutG(NamedTuple):
    """``StepOut``'s fields, then the observer channel (returned only by ``step(..., observe=True)``)."""

    counters: mx.array
    ledger: mx.array
    births: BirthRecords
    deaths: DeathRecords
    acts: Acts

    def base(self) -> StepOut:
        """The ``StepOut`` part, without the channel."""
        return StepOut(self.counters, self.ledger, self.births, self.deaths)


class _Flows(NamedTuple):
    """Intermediate results of S0–S17 that S18 turns into the ledger, counters and diag."""

    costs: en.Costs | en.Costs3  # Costs3 in 3D (paid_buoy, §27.7)
    got: mx.array  # (N,) harvest
    cb: en.Combat
    sh: en.Share
    e_over: mx.array  # (N,) cap overflow
    dt: lifecycle.Deaths
    lanes: lifecycle.Lanes
    rp: lifecycle.Repro
    regen_inc: mx.array  # d.field_shape
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
    # 3D only (None in 2D): the Counter3D inputs (§27.8).
    z_floor: mx.array | None = None  # () survivors clamped at the floor (Integrated3)
    z_surface: mx.array | None = None  # () survivors clamped at the surface
    grounded: mx.array | None = None  # (N,) bool, physics.grounded of the start-of-step pos (not masked)
    # Weather only (None when weather.enabled is false): the advisory WEATHER_WARN input (§30).
    weather_bad: mx.array | None = None  # () bool, weather.health
    # Feature extensions (§43.5): None while every feature is off, else an ``ExtFlows``.
    ext: ExtFlows | None = None


def step(
    s: State | State3, cfg: Config, *, observe: bool = False
) -> tuple[State | State3, StepOut | StepOutG]:
    """Advance the world by one step (S0–S18). Pure; no host syncs. A 3D world (``State3``) takes the static
    ``d3`` branches of §27.8; every 2D line is unchanged. With the static ``observe`` the step also returns
    the observer channel (``StepOutG``, §28.2); it changes no state value and no ``StepOut`` field."""
    d3 = dims(cfg).d3
    t = s.step
    W1, W2 = lifecycle.flush_pending(s.W1, s.W2, s.pend_slot, s.pend_W1, s.pend_W2)  # S0
    s = s._replace(W1=W1, W2=W2)
    key_next, ks = step_keys(s.key)  # S1
    a = age(t, s.birth_step)
    nbr = spatial.find_neighbors(s.pos, s.alive, cfg)  # S2
    geom = bonds.bond_geometry(s.pos, s.partner, cfg)  # S3
    wthr = cfg.weather.enabled  # static: the weather layer (§30)
    fcell = fields.cell_index(s.pos, cfg)  # S4 (above S3w: a pure function of s.pos, read by structures)
    st_on = cfg.structures.enabled  # static: built structures (§38)
    sn = structures.sense(s.mat, fcell, cfg) if st_on else None  # the material at the start-of-step cell
    if wthr:
        ws = weather.unpack(s.wx)  # S3w: the weather stored at the end of the previous step, unpacked once
        wind = weather.agent_wind(ws, s.pos, s.buoy_cap if d3 else None, s.alive, cfg)
        if st_on:  # static: shelter lowers the felt wind, the hazard speed and the bond break (§38.4.1)
            wind = structures.apply_shelter(wind, sn, cfg)
        kw = mx.random.split(ks[Stream.WEATHER], 2)  # kw[0]: the disturbance key (S17); kw[1]: the hazard
    if d3:  # static: the yaw-and-pitch frame and the start-of-step grounding (§27.6)
        gr = physics.grounded(s.pos, cfg)
        fobs = fields.sample_fields(s.resource, s.chem, fcell, s.heading, cfg, pitch=s.pitch)
    else:
        fobs = fields.sample_fields(s.resource, s.chem, fcell, s.heading, cfg)
    desc = sensing.neighbor_descriptors(s, nbr, cfg)  # S5
    ectx = ext.ExtCtx(nbr, desc, geom, fcell)
    ext_in = ext.step_inputs(s, cfg, ectx)  # the enabled features' brain-tail blocks, in TAIL_ORDER
    if d3:  # static
        x = sensing.build_inputs(
            s, a, nbr, desc, geom, fobs, ks[Stream.NOISE], cfg, grounded=gr, ext_in=ext_in
        )
    else:
        x = sensing.build_inputs(s, a, nbr, desc, geom, fobs, ks[Stream.NOISE], cfg, ext_in=ext_in)
    h_new, y = brain.forward(s.W1, s.W2, x, s.h, cfg)  # S6
    act = actions.decode(y, desc, nbr, s.partner, s.alive, cfg)  # S7
    xdec = ext.decode_all(
        s, y, ectx, cfg
    )  # the enabled features' own decodes ({} while every feature is off)
    if d3:  # static: the pitch change and the buoyancy fraction (§27.2)
        mo = actions.decode_motion3(y, s.alive, cfg)

    nb0 = bonds.n_bonds(s.partner)  # S8
    extra_due = ext.sum_due(spec.due(s, act, y, cfg) for spec in ext.active(cfg) if spec.due is not None)
    due_kw = (
        {} if extra_due is None else {"extra_due": extra_due}
    )  # static: the call is today's while it is None
    bcost = arms.basal_cost(s, cfg)  # the brain cost, plus the arms upkeep while it is on (§42.3)
    if d3:  # static: the pitch turn cost and the buoyancy due (§27.7)
        c = en.pay_costs3(s.energy, act, mo, nb0, bcost, s.buoy_cap, s.alive, cfg, **due_kw)
        buoy0 = mo.buoy if observe else None  # static: the pre-mask buoyancy, as thrust0 below (§32)
        mo = actions.mask(mo, c.solvent)
    else:
        c = en.pay_costs(s.energy, act, nb0, bcost, s.alive, cfg, **due_kw)
        buoy0 = None
    xdec = ext.mask_all(xdec, c.solvent, cfg)  # solvency, as ``actions.mask`` does for the base actions
    thrust0 = act.thrust if observe else None  # static: the decode output, which S8 also reads (§28.2)
    y0 = y if observe else None  # static: the brain output the kin queries are quantised from (§29.2)
    act = actions.mask(act, c.solvent)

    flag = act.release | (geom.length > cfg.bonds.break_length)  # S9
    if wthr and wind.breaks is not None:  # static: storm_break_u > 0, an explicit air-speed threshold (§30)
        flag = flag | wind.breaks[:, None]
    partner, rev, broken = bonds.break_bonds(s.partner, s.rev, flag & geom.valid)
    partner, rev, formed, as_prop = bonds.form_bonds(
        partner, rev, s.alive & c.solvent & act.bond_want, act.bond_target, ks[Stream.BOND_PRIO]
    )
    slots_s9 = (partner >= 0).astype(_I32).sum()

    bx = bonds.bonds_ext(cfg)  # static: bonds that pay (DESIGN §40)
    ef = roles.effects(s, act, partner, cfg) if cfg.roles.enabled else None  # static: the role effects
    E, resource, got = en.harvest(
        c.energy, s.resource, fcell, act, c.solvent, cfg, harvest_cap=None if ef is None else ef.harvest_cap
    )  # S10
    shield_kw = structures.combat_kw(sn, cfg) if st_on else {}  # static: the predator shield (§38.4.3)
    if ef is not None and ef.guard is not None:  # static: the guard shield joins the protection product
        shield_kw = {**shield_kw, "guard": ef.guard}
    act_c = act if ef is None else roles.combat_act(act, ef.defend_mult)  # the forager's exposed defence
    cb = en.combat(E, resource, fcell, act_c, partner, cfg, **arms.combat_kw(s, cfg), **shield_kw)  # S11
    if observe or bx:  # static: the per-slot flows are kept (channel, bond_flow); the graph equals en.share's
        kin = bonds.slot_kin(s.tag, partner, cfg) if bx and cfg.bonds.share_kin_floor < 1.0 else None
        sl = en.share_slots(cb.energy, partner, rev, s.partner, act, c.solvent, cfg, kin)  # S12
        sh = en.share_from_slots(cb.energy, sl)
    else:
        sh = en.share(cb.energy, partner, rev, s.partner, act, c.solvent, cfg)  # S12
    E, e_over = en.cap_energy(sh.energy, cfg)  # S13
    env = s.env if cfg.director.enabled else None  # static: the director's live levers
    cat_mask = (
        lifecycle.catastrophe_mask(s.pos, s.alive, env, ks[Stream.CATASTROPHE], cfg)
        if env is not None
        else None
    )
    # The hazard mask is built every step weather is on, even at storm_p = 0 (all false), so the stream use
    # does not depend on the lever.
    storm = lifecycle.storm_mask(wind.speed, s.alive, kw[1], cfg) if wthr else None
    dt = lifecycle.deaths(
        s.alive,
        c.solvent,
        E,
        cb.lost,
        a,
        s.uid,
        s.generation,
        s.clade,
        partner,
        rev,
        cb.resource,
        fcell,
        cfg,
        cat_mask=cat_mask,
        storm_mask=storm,
    )  # S14

    sv = dt.survivor  # S15
    wind_f = wind.force if wthr else None
    act_p = arms.physics_act(act, s, cfg)  # the armour slows the thrust (§42.3); ``act`` itself when off
    if d3:  # static: gravity, buoyancy, the floor and the surface (§27.3)
        F = physics.forces3(
            s.pos, s.heading, s.pitch, act_p, mo, s.buoy_cap, nbr, sv, dt.partner, cfg, wind=wind_f
        )
        dm = structures.drag_mult(F, sn, cfg) if st_on else None  # static: the barrier (§38.4.2)
        it = physics.integrate3(
            s.pos, s.heading, s.pitch, F, act.grip, act.turn, mo.pitch, gr, sv, cfg, drag_mult=dm
        )
        pos, vel, heading, v_clips = it.pos, it.vel, it.heading, it.v_clips
    else:
        F = physics.forces(s.pos, s.heading, act_p, nbr, sv, dt.partner, cfg, wind=wind_f)
        dm = structures.drag_mult(F, sn, cfg) if st_on else None  # static
        pos, vel, heading, v_clips = physics.integrate(
            s.pos, s.heading, F, act.grip, act.turn, sv, cfg, drag_mult=dm
        )
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
    if d3:  # static
        s1 = s1._replace(pitch=it.pitch)
    if bx:  # static: role, dwell and the slot memory of the survivors; dead rows cleared (E15)
        upd, bflows = bonds.state_after(
            s,
            sv,
            dt.partner,
            sl,
            cfg,
            dec=xdec["bonds"] if cfg.roles.enabled else None,
            guard=None if ef is None else ef.guard,
            shielded=roles.shielded_by_guard(None if ef is None else ef.guard, cb.attack, act.attack_target),
        )
        s1 = s1._replace(**upd)
        xdec = {**xdec, "bonds": bflows}

    req = sv & act.reproduce & (dt.energy >= cfg.repro.min_energy) & (a >= cfg.repro.min_age)  # S16
    imm_request = env.imm_request if env is not None else None
    lanes = lifecycle.allocate_lanes(req, sv, ks[Stream.BIRTH_PRIO], cfg, s.step, imm_request=imm_request)
    rp = lifecycle.reproduce(s1, lanes, act.child_bond, ks, cfg)
    s2 = rp.state

    dep = fields.deposit_chem(act.emit, fcell, c.solvent, cfg)  # S17
    chem = fields.update_chem(s2.chem, dep, cfg)
    rain = None
    if wthr:  # static: advance the weather; the new rain feeds the field
        # The organism forcing (DESIGN §32) reads the start-of-step ``s.chem``, ``s.pos``, ``s.alive``: one
        # step of lag, no draw. None while ``seed_gain``/``shade_gain`` are 0, so the weather then reads no
        # agent array (DESIGN §30).
        qs, t_mult = weather.organism_forcing(s.chem, s.pos, s.alive, ws, cfg)
        ws2, wd = weather.advance(ws, t, kw[0], cfg, qs=qs, t_mult=t_mult)
        rain = weather.rain_for(ws2, cfg)  # None when rain_gain == 0
    resource, regen_inc = fields.update_resource(s2.resource, s2.patch_pos, cfg, s2.step, env, rain)
    patch_pos = fields.drift_patches(s2.patch_pos, s2.patch_vel, cfg, env)
    s2 = s2._replace(chem=chem, resource=resource, patch_pos=patch_pos, key=key_next, step=t + 1)
    if st_on:  # static: deposit, cap, aperture gate and decay of the material (§38.2, §38.3)
        mat, sflows = structures.update(s.mat, xdec["structures"], fcell, s.alive, dm, t, cfg)
        s2 = s2._replace(mat=mat)
        xdec = {**xdec, "structures": sflows}
    if wthr:  # static
        s2 = s2._replace(wx=weather.pack(ws2))
    if env is not None:  # static: director enabled
        s2 = consume_env(s2, env, lanes)

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
    if xdec:  # static: the masked feature decodes reach S18 (counters) and the observer channel
        flows = flows._replace(ext=ExtFlows(**xdec))
    if d3:  # static
        flows = flows._replace(z_floor=it.z_floor, z_surface=it.z_surface, grounded=gr)
    if wthr:  # static
        flows = flows._replace(weather_bad=weather.health(wd, wind))
    s_new, out = _finish(s2, s, cfg, flows)  # S18
    if observe:  # static
        return s_new, StepOutG(*out, acts=_acts(flows, sl, thrust0, y0, s.alive, cfg, buoy0))
    return s_new, out


def _acts(
    f: _Flows,
    sl: en.ShareSlots,
    thrust0: mx.array,
    y: mx.array,
    alive0: mx.array,
    cfg: Config,
    buoy0: mx.array | None = None,
) -> Acts:
    """The observer channel (§28.2, §29.2). Every float is quantised from a value the step already
    materialises, and the thrust is masked by solvency in integers, so the channel adds no consumer to a fused
    float chain. ``y`` is the brain output (a kernel output read by two column slices), ``alive0`` the
    start-of-step alive mask (a step input). ``death_cause`` is the S14 cause array that the KILLED bit
    already reads, cast to int8 (the per-cause counts of the roster come from it, never from the capped death
    records). ``buoy0`` (3D) is the buoyancy output before the solvency mask, quantised and masked in integers
    exactly as the thrust (§32)."""
    n = dims(cfg).n
    birth = f.lanes.kind == lifecycle.LaneKind.BIRTH.value
    budded = ops.scatter_add_i32(n, f.lanes.parent, 1, birth) > 0  # distinct parents: Σ = n_birth
    emit_q = ops.floor_i32(f.act.emit * 256.0)  # the expression of _counters and fields.deposit_chem
    emitted = (emit_q > 0).any(axis=1)  # the expression of the EMITTERS counter
    killed = f.dt.cause == DeathCause.KILLED.value  # the expression of the DEATHS_KILLED counter
    u16, i16 = mx.uint16, mx.int16
    flags = (
        f.cb.attack.astype(u16) * ActFlag.ATTACKED.value
        | f.cb.hit.astype(u16) * ActFlag.HIT.value
        | f.formed.astype(u16) * ActFlag.FORMED.value
        | budded.astype(u16) * ActFlag.BUDDED.value
        | emitted.astype(u16) * ActFlag.EMITTED.value
        | f.as_prop.astype(u16) * ActFlag.PROPOSED.value
        | killed.astype(u16) * ActFlag.KILLED.value
    )
    if f.ext is not None and isinstance(f.ext.bonds, bonds.BondFlows):  # static: a role switched (§40.5)
        flags = flags | f.ext.bonds.switched.astype(u16) * ActFlag.ROLE_SWITCHED.value
    thrust = mx.where(f.costs.solvent, mx.round(256.0 * thrust0).astype(i16), 0)  # = round(256·masked thrust)
    defend = mx.round(256.0 * f.act.defend).astype(i16)
    # The kin columns of the attack and bond queries (the _select score is φ·q, with φ = desc[..., 1:], so the
    # kin feature NBR_KIN is column NBR_KIN − 1 of a query block). The brain evaluates dead rows too: mask in
    # integers, after the cast.
    qa = y[:, layout.OUT_Q_ATTACK + layout.NBR_KIN - 1]
    qb = y[:, layout.OUT_Q_BOND + layout.NBR_KIN - 1]
    q = mx.round(256.0 * mx.clip(mx.stack([qa, qb], axis=1), -KIN_Q_CLIP, KIN_Q_CLIP)).astype(i16)
    q_kin = mx.where(alive0[:, None], q, 0)
    rows = mx.arange(n, dtype=_I32)[:, None]
    nbr = mx.where(f.nbr.idx != rows, f.nbr.idx, -1)  # invalid entries hold the row's own index (spatial.py)
    born_parent = mx.where(birth, f.lanes.parent, -1)
    buoy = mx.where(f.costs.solvent, mx.round(256.0 * buoy0).astype(i16), 0) if buoy0 is not None else None
    xa = None
    if f.ext is not None and isinstance(f.ext.bonds, bonds.BondFlows):  # static: observer sub-tuples (§43.5)
        xa = ExtActs(bonds=bonds.acts_of(f.ext.bonds))
    return Acts(
        f.cb.gain, sl.out, sl.got, flags, thrust, defend, emit_q.sum(axis=1).astype(i16),
        f.act.attack_target, f.act.bond_target, q_kin, nbr,
        f.rp.records.slot, born_parent, f.rp.records.uid, f.dt.cause.astype(mx.int8), buoy, xa,
    )  # fmt: skip


def consume_env(s: State, env: Env, lanes: lifecycle.Lanes) -> State:
    """Write back the one-shot env levers after a step (§12.7, §12.9).

    ``env`` is the start-of-step view. The immigration request carries over what the lanes did not admit
    (``max(request − n_imm, 0)``), and the catastrophe kill probability resets to 0 so a catastrophe lasts one
    step. Centre and radius stay as set: they are inert at ``cat_kill_p = 0``. Both outputs derive from inputs
    (MLX 0.32 ``mx.compile`` fails on a constant output next to arrays that use it).
    """
    return s._replace(
        env_imm_request=mx.maximum(env.imm_request - lanes.n_imm, 0).astype(_I32),
        env_cat_kill_p=env.cat_kill_p * 0.0,
    )


def _ledger(f: _Flows, cfg: Config) -> mx.array:
    """This step's (16,) int64 ledger from exact limb sums plus int64 count × constant terms (§10.8).

    In 3D ``paid_buoy`` is an 11th per-agent column, booked in slot 15 (``HEAT_BUOYANCY``, §27.7); in 2D the
    slot is the constant 0 (``RESERVED``).
    """
    d3 = dims(cfg).d3
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
            *([f.costs.paid_buoy] if d3 else []),  # 3D: HEAT_BUOYANCY, last
        ],
        axis=1,
    ).astype(_I32)
    a = ops.exact_sum(agent, axis=0)  # (10,) int64; (11,) in 3D
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
        LedgerIdx.RESERVED: a[10] if d3 else zero,  # HEAT_BUOYANCY in 3D
    }
    return mx.stack([entries[i].astype(_I64) for i in LedgerIdx])


def _kin_sum(desc: mx.array, k: mx.array, rows: mx.array) -> mx.array:
    """``Σ_rows round(256·desc[i, k_i, NBR_KIN])`` as int32 (``k ≥ 0`` on every selected row)."""
    kin = mx.take_along_axis(desc[..., layout.NBR_KIN], mx.maximum(k, 0)[:, None], axis=1)[:, 0]
    return mx.where(rows, mx.round(256.0 * kin), 0.0).astype(_I32).sum()


def _counters(s_new: State, alive0: mx.array, f: _Flows, cfg: Config) -> mx.array:
    """The (27,) int32 per-step counters of §19.1; ``alive0`` is the start-of-step ``alive``. With weather on
    ``DEATHS_STORM`` is appended last (28 in 2D, 31 in 3D, ``state.n_counters(nd, True)``).

    In 3D the three ``Counter3D`` counts follow (30,): the floor and surface clamps of this step's integration
    and the agents that were grounded at its start and survived it (§27.8: a row that died this step is not
    counted, so ``Σ GROUNDED ≤ Σ N_ALIVE`` and the interval's ``grounded_frac`` lies in [0, 1]).
    """
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
        CounterIdx.DEATHS_CATASTROPHE: count(dt.cause == DeathCause.CATASTROPHE.value),
    }
    assert len(entries) == N_COUNTERS
    counters = mx.stack([entries[i].astype(_I32) for i in CounterIdx])
    if d.d3:  # static: appended, so the first 27 keep their meaning
        extra = {
            Counter3D.Z_FLOOR_CLAMPS: f.z_floor,
            Counter3D.Z_SURFACE_CLAMPS: f.z_surface,
            Counter3D.GROUNDED: count(f.grounded & alive0 & ~dt.dead),
        }
        counters = mx.concatenate([counters, mx.stack([extra[i].astype(_I32) for i in Counter3D])])
    if cfg.weather.enabled:  # static: DEATHS_STORM, last, so no index moves (state.storm_counter_index)
        counters = mx.concatenate(
            [counters, count(dt.cause == DeathCause.STORM.value).astype(_I32).reshape(1)]
        )
    if cfg.combat.protect_floor > 0:  # static: the ``protect`` block, first of the extension blocks (§43.4)
        capped = cb.capped if cb.capped is not None else mx.zeros_like(cb.attack)
        counters = mx.concatenate([counters, count(capped).astype(_I32).reshape(1)])
    for spec in ext.active(cfg):  # static: each enabled feature's block, in EXTENSIONS order
        if names := spec.counters(cfg):
            vals = spec.count(s_new, alive0, f, cfg)
            counters = mx.concatenate([counters, mx.stack([vals[n].astype(_I32) for n in names])])
    return counters


def _finish(s_new: State, s: State, cfg: Config, f: _Flows) -> tuple[State, StepOut]:
    """S18: ledger, cumulative ledger, ``total``, counters and diag (§16.3).

    ``INT_HEADROOM`` gets the largest start-of-step field-cell occupancy; in 3D the largest x/y column
    occupancy, since a floor cell collects the corpses of its whole column (§27.7).
    """
    d = dims(cfg)
    ledger = _ledger(f, cfg)
    total = total_energy(s_new.energy, s_new.alive, s_new.resource, cfg)
    s_new = s_new._replace(total=total, ledger_cum=s.ledger_cum + ledger)
    counters = _counters(s_new, s.alive, f, cfg)
    if cfg.run.runtime_checks != "off":  # static
        if d.d3:  # static: per x/y column; fcell mod FG² is the floor cell below (§27.5)
            max_cell_occ = ops.scatter_add_i32(d.fg * d.fg, f.fcell % (d.fg * d.fg), 1, s.alive).max()
        else:
            max_cell_occ = ops.scatter_add_i32(d.fg * d.fg, f.fcell, 1, s.alive).max()
        extra = {} if f.weather_bad is None else {"weather_bad": f.weather_bad}  # static: weather on
        if cfg.structures.enabled:  # static: the start-of-step aperture audit (§38.4b)
            extra = {**extra, "struct_viol": f.ext.structures.violations}
        bits = invariants.device_check(
            s_new,
            ledger,
            s.total,
            max_cell_occ,
            f.cb.max_attackers,
            f.nbr.overflow_cells,
            f.v_clips,
            cfg,
            **extra,
        )
        # The start-of-step levers, as the step used them (some are consumed by it).
        if cfg.director.enabled:  # static
            env_bad = invariants.env_out_of_range(s, cfg)
            bits = bits | mx.where(env_bad, Diag.FIELD_RANGE.value, 0).astype(_I32)
        s_new = s_new._replace(diag=s.diag | bits)
    out = StepOut(counters=counters, ledger=ledger, births=f.rp.records, deaths=f.dt.records)
    return s_new, out


def make_step(
    cfg: Config, *, compile: bool | None = None, observe: bool = False
) -> Callable[[State], tuple[State, StepOut | StepOutG]]:
    """The step closed over ``cfg``, compiled once when ``compile`` (default ``cfg.run.compile``). With the
    static ``observe`` it returns ``StepOutG`` (§28.2)."""
    use_compile = cfg.run.compile if compile is None else compile

    def fn(s: State) -> tuple[State, StepOut | StepOutG]:
        return step(s, cfg, observe=observe)

    return mx.compile(fn) if use_compile else fn
