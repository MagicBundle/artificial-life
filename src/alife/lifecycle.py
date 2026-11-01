"""Births, deaths and the deferred weight flush (§10.7, §12). Owned by WP4.

Every contested write is an int32 scatter in ``ops``; newborn rows are written by gather (``ops.place``)
through int32 row → lane maps, so ``uid`` (int64) never needs a scatter (§12.3, D4). Child weights are
buffered in ``pend_*`` and written at S0 of the next step (§12.6, F1).
"""

from __future__ import annotations

import math
from enum import IntEnum
from typing import Any, NamedTuple

import mlx.core as mx

from alife import bonds, brain, ext, layout, ops, sphere
from alife.config import Config, dims, immigration_possible
from alife.rng import Stream
from alife.state import R_HARD, DeathCause, Env, State, State3, sphere_kinematics, volume_kinematics

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
    resource: mx.array  # (FG,FG) int32 after corpses and the R_HARD cap; 3D (FZ,FG,FG)
    corpse: mx.array  # (N,) int32: E + S on dead rows, else 0 (FLOW_CORPSE)
    r_overflow: mx.array  # (FG,FG) int32 removed by the hard cap (HEAT_OVERFLOW); 3D (FZ,FG,FG)
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


def catastrophe_mask(pos: mx.array, alive: mx.array, env: Env, key: mx.array, cfg: Config) -> mx.array:
    """(N,) bool: alive agents inside the catastrophe disc that draw below ``env.cat_kill_p`` (§12.9).

    Distance is the torus minimum image from ``env.cat_center`` to the start-of-step position. One uniform
    per row is drawn from ``key`` (``Stream.CATASTROPHE``) every step the director is enabled, whether or not
    a catastrophe is active, so the stream use is independent of the levers. ``cat_radius = 0`` or
    ``cat_kill_p = 0`` kills nobody. In 3D the disc is a vertical column: only x and y count (§27.4). On a
    sphere (§46.4) it is a cap: the agents whose direction ``û`` is within the angle ``cat_radius/Rp`` of the
    centre ``env_cat_center = (lon, lat)``, at any height (the radial column of the cap), the same semantics.
    """
    if dims(cfg).sph:  # static: the cap, ``û·c >= cos(radius/Rp)`` (the radius is an arc length at the floor)
        lon, lat = env.cat_center[0], env.cat_center[1]
        c = mx.stack([mx.cos(lat) * mx.cos(lon), mx.cos(lat) * mx.sin(lon), mx.sin(lat)])
        up = pos / sphere.radius_of(pos)[:, None]
        cos_r = mx.cos(mx.minimum(env.cat_radius / cfg.world.radius, math.pi))
        inside = ((up * c).sum(axis=1) >= cos_r) & (env.cat_radius > 0.0)
        u = mx.random.uniform(shape=(pos.shape[0],), key=key)
        return alive & inside & (u < env.cat_kill_p)
    if dims(cfg).d3:  # static: the (2,) centre and the x/y distance, at any height
        pos = pos[:, :2]
    off = ops.min_image(pos - env.cat_center, cfg.world.size)
    inside = ((off * off).sum(axis=1) <= env.cat_radius * env.cat_radius) & (env.cat_radius > 0.0)
    u = mx.random.uniform(shape=(pos.shape[0],), key=key)
    return alive & inside & (u < env.cat_kill_p)


def storm_mask(speed: mx.array, alive: mx.array, key: mx.array, cfg: Config) -> mx.array:
    """(N,) bool: alive agents that draw below the storm hazard ``p_i`` (§30).

    ``p_i = storm_p·clip((speed_i − storm_u0)/storm_span, 0, 1)`` with ``speed`` the air speed at the agent
    (``weather.agent_wind``), not its own. One uniform per row is drawn from ``key`` (the hazard half of the
    ``Stream.WEATHER`` split) every step weather is on, whether or not any hazard is active, so the stream use
    is independent of the levers (``storm_p = 0`` gives an all-false mask).
    """
    wc = cfg.weather
    p = wc.storm_p * mx.clip((speed - wc.storm_u0) / wc.storm_span, 0.0, 1.0)
    u = mx.random.uniform(shape=(speed.shape[0],), key=key)
    return alive & (u < p)


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
    *,
    cat_mask: mx.array | None = None,
    storm_mask: mx.array | None = None,
) -> Deaths:
    """S14: ``dead = alive & (~solvent | E == 0 | aged)``; corpses, canonical rows, bonds, records (§10.7).

    Cause, first match wins: ``STARVED`` if insolvent, ``KILLED`` if ``E == 0`` after a loss this step,
    ``STARVED`` if ``E == 0``, else ``AGED``. ``cat_mask`` (the director's catastrophe, None when off)
    kills additional agents: an alive row in the mask that none of those causes already kills dies as
    ``CATASTROPHE``, so every other cause keeps precedence. ``storm_mask`` (the weather hazard, None when
    weather is off) does the same with ``STORM``, below the catastrophe: a row in both dies as
    ``CATASTROPHE``. Corpses (``E + S``) are scatter-added into the
    dead agent's field cell, then the field is hard-capped at ``R_HARD``. Death records hold the first ``DL``
    dead rows by index. In 3D corpses sink: they land in the floor cell ``(0, cy, cx)`` below the dead agent,
    ``fcell % FG²`` (§27.6); on a sphere the floor cell of the column is ``fcell % Npix`` (§46.4).
    """
    d = dims(cfg)
    zero = energy == 0
    dead = alive & (~solvent | zero)
    if cfg.population.max_age > 0:  # static: senescence enabled
        dead = dead | (alive & (age >= cfg.population.max_age))
    cat = None
    if cat_mask is not None:  # static: director enabled
        cat = alive & cat_mask & ~dead  # excess deaths only
    storm = None
    if storm_mask is not None:  # static: weather enabled
        storm = alive & storm_mask & ~dead  # excess deaths only
        if cat is not None:  # static
            storm = storm & ~cat  # the catastrophe keeps precedence
    if cat is not None:  # static
        dead = dead | cat
    if storm is not None:  # static
        dead = dead | storm
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
    if storm is not None:  # static: weather enabled
        cause = mx.where(storm, DeathCause.STORM.value, cause)
    if cat is not None:  # static: director enabled
        cause = mx.where(cat, DeathCause.CATASTROPHE.value, cause)
    cause = mx.where(dead, cause, -1).astype(_I32)

    corpse = mx.where(dead, energy + cfg.energy.structure, 0).astype(_I32)
    fg2 = d.npix if d.sph else d.fg * d.fg  # the cells of one layer (a sphere: Npix angular cells)
    if d.d3:  # static: corpses sink to the floor cell below
        r = resource.reshape(-1) + ops.scatter_add_i32(d.n_cells, fcell % fg2, corpse, dead)
    else:
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
        resource=r.reshape(resource.shape),
        corpse=corpse,
        r_overflow=r_over.reshape(resource.shape),
        cause=cause,
        records=records,
    )


def allocate_lanes(
    req: mx.array,
    survivor: mx.array,
    key: mx.array,
    cfg: Config,
    step: mx.array | None = None,
    *,
    imm_request: mx.array | None = None,
) -> Lanes:
    """Assign up to M birth/immigrant lanes to distinct dead rows under capacity pressure (§12.2).

    Birth lanes are a prefix, served in a uniformly random order of the requesters; immigrant lanes (only
    when immigration is on) follow; the rest are padding. Every lane, padding included, targets one of
    the M lowest-index dead rows, which exist because the population never exceeds ``N_CAP = N − M``.
    Non-birth lanes use parent row 0.

    Immigrant demand is the larger of the floor rule (``immigration_floor``: top up to the floor, at most
    ``immigrants_per_step``) and the periodic rule (``immigration_count`` on steps with
    ``(step + 1) % immigration_every == 0``, ``step`` being this step's index), then capped by the free lanes
    and the free capacity after births (§12.7). ``step`` is required when ``immigration_every > 0``.

    ``imm_request`` (the director's ``env.imm_request``, None when it is off) is a third demand: the
    demand is ``max(floor and periodic demand, imm_request)``, and the step carries
    ``max(imm_request − n_imm, 0)`` over to later steps.
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
    if immigration_possible(cfg):  # static: immigration enabled (config rules or the director)
        demand = mx.array(0, dtype=_I32)
        if pop.immigration_floor > 0:  # static: floor rule
            demand = mx.minimum(pop.immigration_floor - n_surv - n_birth, pop.immigrants_per_step)
        if pop.immigration_every > 0:  # static: periodic rule
            if step is None:
                raise ValueError("allocate_lanes needs `step` when population.immigration_every > 0")
            due = (step + 1) % pop.immigration_every == 0
            demand = mx.maximum(demand, mx.where(due, pop.immigration_count, 0))
        if imm_request is not None:  # static: director enabled
            demand = mx.maximum(demand, imm_request)
        want = mx.minimum(demand, mx.minimum(m - n_birth, room - n_birth))
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


def _child_genes(s: State | State3, p: mx.array, key: mx.array, cfg: Config) -> tuple[mx.array | None, ...]:
    """Mutated (σ, p, tag, frac, buoy_cap) of each lane's parent row (§12.4); f32, (M,) / (M,T).

    ``buoy_cap`` is None in 2D. In 3D (§27.4) the key splits into 5 whether or not the actuator is on, and the
    cap mutates additively with a clip, like ``tag``: ``clip(cap_p + buoy_sigma·n₅, 0, 1)``.
    """
    mu, m = cfg.mutation, p.shape[0]
    d3 = dims(cfg).d3
    if d3:  # static: a fifth gene, the buoyancy cap
        k_sigma, k_rate, k_tag, k_frac, k_cap = mx.random.split(key, 5)
    else:
        k_sigma, k_rate, k_tag, k_frac = mx.random.split(key, 4)
    n1 = mx.random.normal((m,), key=k_sigma)
    n2 = mx.random.normal((m,), key=k_rate)
    n3 = mx.random.normal((m, layout.T), key=k_tag)
    n4 = mx.random.normal((m,), key=k_frac)
    sigma = mx.clip(s.mut_sigma[p] * mx.exp(mu.tau_sigma * n1), mu.sigma_min, mu.sigma_max)
    rate = mx.clip(mx.sigmoid(_logit(s.mut_rate[p]) + mu.tau_rate * n2), mu.rate_min, mu.rate_max)
    tag = mx.clip(s.tag[p] + mu.tag_sigma * n3, -1.0, 1.0)
    frac = mx.clip(mx.sigmoid(_logit(s.offspring_frac[p]) + mu.tau_frac * n4), mu.frac_min, mu.frac_max)
    cap = None
    if d3:  # static
        n5 = mx.random.normal((m,), key=k_cap)
        cap = mx.clip(s.buoy_cap[p] + mu.buoy_sigma * n5, 0.0, 1.0)
    return sigma, rate, tag, frac, cap


def _offspring_kinematics_3d(
    pos_p: mx.array, heading_p: mx.array, pitch_p: mx.array, u: mx.array, cfg: Config
) -> tuple[mx.array, mx.array, mx.array]:
    """3D child placement (§27.4): ``(pos (M,3), heading (M,), pitch (M,))`` from the parents' rows.

    The child sits at ``offspring_distance`` in a cone behind the parent: yaw ``θ_p + π + u₀`` and pitch
    ``clip(u₁ − φ_p, ±PITCH_MAX)``, with ``u`` the (M,2) ``PLACEMENT`` jitter. When that point would leave
    ``[z_lo, z_hi]`` the pitch is negated (a reflection at the floor or the surface), so the offset keeps its
    length (by default the child bond's rest length); ``height − 2r ≥ 2·offspring_distance`` (§27.1) puts the
    reflected point in range, and the final clamp only absorbs rounding. The child faces away from the parent.
    With zero jitter and a level parent this is the 2D rule.
    """
    d, D, pm = dims(cfg), cfg.repro.offspring_distance, layout.PITCH_MAX
    yaw = heading_p + math.pi + u[:, 0]
    pitch = mx.clip(u[:, 1] - pitch_p, -pm, pm)
    z1 = pos_p[:, 2] + D * mx.sin(pitch)
    pitch = mx.where((z1 < d.z_lo) | (z1 > d.z_hi), -pitch, pitch)
    cp = mx.cos(pitch)
    step_vec = D * mx.stack([cp * mx.cos(yaw), cp * mx.sin(yaw), mx.sin(pitch)], axis=1)
    pos = ops.wrap_xy_clamp_z(pos_p + step_vec, cfg.world.size, d.z_lo, d.z_hi)
    return pos, ops.wrap_angle(yaw), pitch


def _offspring_kinematics_sph(
    pos_p: mx.array, hdg_p: mx.array, pitch_p: mx.array, u: mx.array, cfg: Config
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Sphere child placement (§46.4): ``(pos (M,3), hdg (M,3), heading (M,), pitch (M,))`` from the parents.

    The 3D rule on the sphere, with the same ``(M,2)`` ``PLACEMENT`` draws: the child sits at distance
    ``offspring_distance = D`` behind the parent in a cone. ``t̂`` is the direction ``−hdg`` of the parent
    turned about its up by the yaw jitter ``u₀``; the pitch is ``χ = clip(u₁ − φ_p, ±PITCH_MAX)``, negated
    when the child's height would leave ``[z_lo, z_hi]`` (a reflection at the floor and the lid, as in 3D).
    The child is at height ``z_p + D·sin χ`` and at the end of a great-circle arc along ``t̂`` whose length
    is chosen so that the chord to the parent is exactly ``D`` (``2 r asin(D cos χ / (2 sqrt(r r')))`` with
    ``r' = r + D sin χ``: the bond rest length is a chord, §46.4): the movement is ``sphere.move`` with the
    radial part ``D sin χ``. Its heading is ``t̂`` transported to the child's point (the child faces away from
    its parent) and its pitch ``χ``.
    """
    d, D, pm = dims(cfg), cfg.repro.offspring_distance, layout.PITCH_MAX
    r = sphere.radius_of(pos_p)
    up = pos_p / r[:, None]
    back = -hdg_p
    cu, su = mx.cos(u[:, 0])[:, None], mx.sin(u[:, 0])[:, None]
    that = cu * back + su * ops.cross3(up, back)  # tangent: a rotation of a tangent vector about up
    chi = mx.clip(u[:, 1] - pitch_p, -pm, pm)
    z1 = r - cfg.world.radius + D * mx.sin(chi)
    chi = mx.where((z1 < d.z_lo) | (z1 > d.z_hi), -chi, chi)
    dr = D * mx.sin(chi)
    arc = 2.0 * r * mx.arcsin(D * mx.cos(chi) / (2.0 * mx.sqrt(r * (r + dr))))
    v = arc[:, None] * that + dr[:, None] * up
    pos, hdg, _, _, _ = sphere.move(pos_p, that, v, mx.zeros_like(chi), d.r_lo, d.r_hi)
    return pos, hdg, sphere.bearing(hdg, pos), chi


def _on_birth(f: ext.FieldSpec, old: mx.array, cfg: Config, bctx: ext.BirthCtx, put: Any) -> mx.array:
    """The new value of a per-agent extension field after the births of this step (``FieldSpec.on_birth``)."""
    rest = f.rest(cfg)
    nb = bctx.newborn.reshape(bctx.newborn.shape + (1,) * (old.ndim - 1))
    if f.on_birth == "rest":  # newborn rows (births and immigrants) start fresh
        return mx.where(nb, rest.astype(old.dtype), old)
    if (
        f.on_birth == "inherit"
    ):  # a birth copies its parent's row (pre-birth values), an immigrant starts fresh
        lane = bctx.imm.reshape(bctx.imm.shape + (1,) * (old.ndim - 1))
        return put(old, mx.where(lane, rest[bctx.slot], old[bctx.parent]))
    if callable(f.on_birth):
        return f.on_birth(old, rest, bctx)
    raise ValueError(
        f"FieldSpec {f.name!r}: on_birth must be 'rest', 'inherit' or a hook, got {f.on_birth!r}"
    )


def reproduce(s: State | State3, lanes: Lanes, child_bond: mx.array, ks: mx.array, cfg: Config) -> Repro:
    """S16: write newborn rows via ``ops.place``, debit parents, add child bonds, set ``pend_*`` (§12).

    ``s`` is the in-flight state (``alive`` = survivors, post-physics ``pos``/``heading``, post-S14 bonds and
    energy, post-flush weights, ``step`` = this step's index). ``child_bond`` is the parents' (N,) decision.
    Returns the state with newborn rows, debited parents, child bonds, ``pend_*`` and ``next_uid`` advanced;
    the lane records; the per-lane endowment (0 off birth lanes, FLOW_ENDOWMENT); and the child-bond count.
    In 3D (§27.4) children go in a cone behind the parent, immigrants are uniform in the volume, and the
    ``pitch`` and ``buoy_cap`` rows are written too. On a sphere (§46.4) the same cone is built on the shell
    (``_offspring_kinematics_sph``), immigrants are uniform in the shell by volume and ``hdg`` is written too.
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
    sigma, rate, tag, frac, cap = _child_genes(s, p, ks[Stream.MUT_GENES], cfg)
    if d.sph:  # static: the same cone on the sphere, a chord D behind the parent (§46.4)
        u = mx.random.uniform(-0.5, 0.5, (m, 2), key=ks[Stream.PLACEMENT])
        pos, hdg, heading, pitch = _offspring_kinematics_sph(s.pos[p], s.hdg[p], s.pitch[p], u, cfg)
    elif d.d3:  # static: a cone behind the parent, reflected at the floor and the surface (§27.4)
        u = mx.random.uniform(-0.5, 0.5, (m, 2), key=ks[Stream.PLACEMENT])
        pos, heading, pitch = _offspring_kinematics_3d(s.pos[p], s.heading[p], s.pitch[p], u, cfg)
    else:
        phi = s.heading[p] + math.pi + mx.random.uniform(-0.5, 0.5, (m,), key=ks[Stream.PLACEMENT])
        step_vec = cfg.repro.offspring_distance * mx.stack([mx.cos(phi), mx.sin(phi)], axis=1)
        pos = ops.wrap(s.pos[p] + step_vec, L)
        heading = ops.wrap_angle(phi)
    w1, w2 = brain.mutate_rows(s.W1[p], s.W2[p], sigma, rate, ks[Stream.MUT_W1], ks[Stream.MUT_W2], cfg)
    if immigration_possible(cfg):  # static: immigrant lanes draw fresh founders (§12.7)
        k_pos, k_head, k_tag, k_w1, k_w2 = mx.random.split(ks[Stream.IMMIGRANT], 5)
        if d.sph:  # static: uniform in the shell by volume, like the founders (§46.5)
            i_pos, i_hdg, i_head, i_pitch = sphere_kinematics(k_pos, k_head, m, cfg)
        elif d.d3:  # static: uniform in the volume, like the founders (§27.4)
            i_pos, i_head, i_pitch = volume_kinematics(k_pos, k_head, m, cfg)
        else:
            i_pos = ops.wrap(mx.random.uniform(0.0, L, (m, 2), key=k_pos), L)
            i_head = ops.wrap_angle(mx.random.uniform(-math.pi, math.pi, (m,), key=k_head))
        i_tag = mx.random.uniform(-1.0, 1.0, (m, layout.T), key=k_tag)
        i_w1, i_w2 = brain.init_weights(k_w1, k_w2, m, cfg)
        pos = mx.where(imm[:, None], i_pos, pos)
        heading = mx.where(imm, i_head, heading)
        if d.sph:  # static
            hdg = mx.where(imm[:, None], i_hdg, hdg)
        tag = mx.where(imm[:, None], i_tag, tag)
        sigma = mx.where(imm, mu.sigma_init, sigma)
        rate = mx.where(imm, mu.rate_init, rate)
        frac = mx.where(imm, mu.frac_init, frac)
        w1 = mx.where(imm[:, None, None], i_w1, w1)
        w2 = mx.where(imm[:, None, None], i_w2, w2)
        if d.d3:  # static
            pitch = mx.where(imm, i_pitch, pitch)
            cap = mx.where(imm, mu.buoy_init, cap)
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
    if d.d3:  # static: the 3D-only rows (§27.4)
        state = state._replace(pitch=put(s.pitch, pitch), buoy_cap=put(s.buoy_cap, cap))
    if d.sph:  # static: the heading vector of the newborn rows (§46.4)
        state = state._replace(hdg=put(s.hdg, hdg))
    for f in ext.state_fields(cfg, d.nd):  # static: the per-agent extension fields' birth policy (§43.2)
        if f.per_agent:
            bctx = ext.BirthCtx(p, slot, birth, imm, newborn, born_lane, ks)
            state = state._replace(**{f.name: _on_birth(f, getattr(s, f.name), cfg, bctx, put)})
    records = BirthRecords(
        uid=mx.where(used, uid, -1),
        parent_uid=mx.where(used, parent_uid, -1),
        clade=mx.where(used, clade, 0),
        generation=mx.where(used, generation, 0),
        slot=mx.where(used, slot, -1),
        valid=used,
    )
    return Repro(state=state, records=records, endow=endow, n_child_bonds=made.astype(_I32).sum())
