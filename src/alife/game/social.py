"""Social state of the observer layer (docs/GAMES.md §6, DESIGN §29.4–§29.8): the device half. Pure MLX.

``update`` folds one step into ``SocState`` (every step, after G1's ``observe.update``): the house, kin-group
slot, colony slot and body size of every row, and the window matrices of the relations between kin-group
slots (attacks, hits, kills, formed bonds, bond-steps, energy shared, and the neighbour-exposure counts that
give the proximity baseline). ``snapshot`` (only at snapshot steps) derives the arrays the host needs to
cluster the tag space, label the bonded bodies, count territory cells and read the temperament, and
``relabel`` writes the host's answer (slots) back and resets the window. The host half is ``society.py``.

The roster (stage 7, DESIGN §29.14) adds three window counts to the same fold, all exact integer sums of the
channel: the births of each kin slot (a newborn counts for the slot it inherits), the deaths of each slot by
cause (``Acts.death_cause``, the S14 cause of every row, never the capped death records) and, per row, the
house its house was founded from (``house_from``). The snapshot also carries each row's primary archetype
label, so that the host can report a group's archetype mix.

Everything is shape-static and branch-free on array values, and a reader of the ``State`` sees no key and no
weight (``SocIn``). Integer counts only: the windows are exact int64 sums, so identities with the engine's
counters hold exactly (tests).

Lint (rule 5, ``tests/test_lint.py``): ``SocialRules`` fields are Python ints used directly (never
``int(sr.x)``: ``sr`` is not a safe root); enum constants are written ``X.Y.value``. No output of ``update``
aliases an input and every output derives from an input (the MLX 0.32 ``mx.compile`` quirk noted at
``step.consume_env``).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from enum import IntEnum
from typing import Final, NamedTuple

import mlx.core as mx

from alife import groups, layout, metrics
from alife.config import Config, dims
from alife.state import State, State3
from alife.step import ActFlag, Acts

__all__ = [
    "DEATH_CAUSES",
    "HOUSE_UNKNOWN",
    "KIN_NAME_SALT",
    "SOCIAL_V1",
    "SOCIAL_VERSION",
    "Expo",
    "Rel",
    "Snap",
    "SocIn",
    "SocState",
    "SocialRules",
    "death_causes",
    "init_soc",
    "relabel",
    "snapshot",
    "soc_in",
    "update",
]

SOCIAL_VERSION: Final = 1
HOUSE_UNKNOWN: Final = -2  # an immigrant lineage seeded mid-run (§29.4): never named, counted apart
KIN_NAME_SALT: Final = 0x6B696E6E616D65  # "kinname": the generated names of kin groups (§29.14)
# The rows of ``SocState.deaths`` with the weather off: the causes of death that exist without it, by
# ``DeathCause`` value. A literal, not an iteration of ``DeathCause``, so that ``DeathCause.STORM`` (W1,
# DESIGN §30.7) adds no row to the state and the sidecar of a weather-off run; ``death_causes(cfg)`` is the
# tuple a run uses.
DEATH_CAUSES: Final = ("starved", "killed", "aged", "catastrophe")

_I32 = mx.int32
_I64 = mx.int64
_I32_MAX: Final = 2**31 - 1


def death_causes(cfg: Config) -> tuple[str, ...]:
    """The rows of ``SocState.deaths`` for ``cfg``: ``DEATH_CAUSES``, plus ``"storm"`` (``DeathCause.STORM``
    = 4) when the weather layer is on (DESIGN §29.4, §30.7). Every row count of the social layer is its
    length."""
    return (*DEATH_CAUSES, "storm") if cfg.weather.enabled else DEATH_CAUSES


@dataclasses.dataclass(frozen=True)
class SocialRules:
    """The cadence and every threshold of the social layer (``SOCIAL_VERSION`` 1 = the defaults).

    Provisional until the calibration run (``python -m alife.analysis.social``). They are code constants, not
    config keys, as the label thresholds are (DESIGN §28.1): ``digest()`` names them, the sidecar stores it,
    and a changed value needs a new ``SOCIAL_VERSION``. A pair ``(a, s)`` is the threshold
    ``max(a, n_ref >> s)`` for the reference population ``n_ref``; ``(num, den)`` is a ratio, tested in
    integers.
    """

    snapshot_every: int = 250  # steps between snapshots, on the global step clock (GAMES §5.5 game cadence)
    slots: int = 16  # F: named slots per family; G = F + 1 with OTHER = F
    slot_slack: int = 4  # a slotted id keeps its slot while tracked and ranked < F + slot_slack
    kin_bins: int = 16  # bins per tag dimension over [-1, 1] (width 1/8); 16^4 = 65,536 bins
    kin_occ: tuple[int, int] = (2, 12)  # a bin is occupied iff count >= max(2, n_ref >> 12)
    kin_enter: tuple[int, int] = (8, 9)  # a kin group gets an id iff size >= max(8, n_ref >> 9)
    kin_leave: tuple[int, int] = (4, 10)  # a tracked kin group keeps it while size >= max(4, n_ref >> 10)
    n_band: int = 3  # n_ref follows n_alive only when |n_alive - n_ref| > n_ref >> 3 (12.5%)
    colony_enter: int = 4  # a colony gets an id iff it is a bond component of >= 4 alive agents
    colony_leave: int = 3  # a tracked colony keeps it while it has >= 3
    colony_iters: int = 32  # metrics.components rounds (the run key run.components_iters is not used)
    theta: tuple[int, int] = (3, 10)  # match iff Jaccard >= 0.3
    uid_spacing: int = 300  # uid matching iff snapshot_every <= 300, else footprint matching
    foot_cells: int = 128  # footprint grid per axis (floor plane)
    terr_cells: int = 32  # territory grid per axis (floor plane)
    terr_k: int = 2  # territory EMA shift (about 3·4 snapshots to 95%)
    terr_min: int = 1  # a cell is owned only if its smoothed total is >= 1 agent
    sim_sample: int = 64  # members per kin group in the mean tag similarity
    sim_distinct: tuple[int, int] = (1, 4)  # a pair is "distinct" iff mean similarity < 0.25
    events_max: int = 32  # events listed per record
    relations_max: int = 32  # relation pairs listed per record
    sim_salt: int = 0x6B696E73  # "kins": member choice of the similarity subsample
    name_salt: int = 0x686F757365  # "house": house names
    kin_name_salt: int = KIN_NAME_SALT  # kin groups named by no house (§29.14)

    def __post_init__(self) -> None:
        def req(ok: bool, msg: str) -> None:
            if not ok:
                raise ValueError(f"SocialRules: {msg}")

        req(self.snapshot_every >= 1, "snapshot_every must be >= 1")
        req(self.slots >= 1 and self.slot_slack >= 0, "slots must be >= 1 and slot_slack >= 0")
        req(self.kin_bins >= 2, "kin_bins must be >= 2")
        for name in ("kin_occ", "kin_enter", "kin_leave"):
            base, shift = getattr(self, name)
            req(base >= 1 and 0 <= shift <= 30, f"{name} must be (base >= 1, 0 <= shift <= 30)")
        req(
            self.kin_leave[0] <= self.kin_enter[0] and self.kin_leave[1] >= self.kin_enter[1],
            "kin_leave must not exceed kin_enter at any population (leave <= enter at every n)",
        )
        req(0 <= self.n_band <= 30, "n_band must be in [0, 30]")
        req(2 <= self.colony_leave <= self.colony_enter, "colony thresholds must satisfy 2 <= leave <= enter")
        req(self.colony_iters >= 1, "colony_iters must be >= 1")
        req(0 < self.theta[0] <= self.theta[1], "theta must be (num, den) with 0 < num <= den")
        req(self.uid_spacing >= 0, "uid_spacing must be >= 0")
        req(self.foot_cells >= 1 and self.terr_cells >= 1, "foot_cells and terr_cells must be >= 1")
        req(0 <= self.terr_k <= 20 and self.terr_min >= 0, "terr_k must be in [0, 20] and terr_min >= 0")
        req(self.sim_sample >= 1, "sim_sample must be >= 1")
        req(
            self.sim_distinct[0] >= 0 and self.sim_distinct[1] > 0, "sim_distinct must be (num >= 0, den > 0)"
        )
        req(self.events_max >= 0 and self.relations_max >= 0, "events_max and relations_max must be >= 0")

    def digest(self) -> str:
        """sha256 of the canonical JSON of every field (tuples as lists): the identity of these rules."""
        text = json.dumps(dataclasses.asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def check(self, cfg: Config) -> None:
        """Raise ``ValueError`` unless every static bin count and lane count of the device code fits int32:
        the exact bin counts of ``groups`` need at most 2^31 - 1 lanes, and every flat bin index (plus the
        dump region) must be an int32."""
        d = dims(cfg)
        n, k, b = d.n, d.k, layout.B
        g = self.slots + 1
        lanes = (
            ("relation counts", (4 + b) * n),  # attack, hit, kill, bond, and the B bond slots of every row
            ("neighbour exposure", 2 * n * k),
            ("energy shared", n * b),
            ("temperament and deaths by cause", 4 * n),  # (4, G) and (C, G), C = 4 or 5 causes
        )
        for name, count in lanes:
            if count > _I32_MAX:
                raise ValueError(f"SocialRules.check: {name} would use {count} lanes, over 2^31 - 1")
        bins = (
            ("kin bins", self.kin_bins**layout.T),
            ("relation bins", 5 * g * g),
            ("exposure bins", 2 * g * g * k),
            ("death bins", len(death_causes(cfg)) * g),
            ("territory bins", self.terr_cells**2 * g),
            ("footprint cells", self.foot_cells**2),
        )
        for name, size in bins:
            if size + groups.DUMP > _I32_MAX:
                raise ValueError(f"SocialRules.check: {name} ({size}) plus the dump region overflow int32")


SOCIAL_V1: Final = SocialRules()


class SocIn(NamedTuple):
    """The only ``State`` fields the social layer reads (never ``key``, never the weights)."""

    alive: mx.array  # (N,) bool
    uid: mx.array  # (N,) int64, -1 on dead rows
    parent_uid: mx.array  # (N,) int64, -1 for founders and immigrants
    clade: mx.array  # (N,) int32
    partner: mx.array  # (N,B) int32, -1 empty
    tag: mx.array  # (N,T) float32
    pos: mx.array  # (N,nd) float32
    next_uid: mx.array  # () int64


def soc_in(s: State | State3) -> SocIn:
    """The social layer's view of a state."""
    return SocIn(*(getattr(s, f) for f in SocIn._fields))


class Rel(IntEnum):
    """Rows of ``SocState.rel``: window counts per (actor slot, target slot)."""

    ATTACKS = 0
    HITS = 1
    KILLS = 2
    BONDS = 3
    BOND_SLOTS = 4


class Expo(IntEnum):
    """Rows of ``SocState.expo``: neighbour lanes of attackers and of proposers."""

    ATTACK = 0
    BOND = 1


class SocState(NamedTuple):
    uid: mx.array  # (N,) int64 where(alive, uid, -1) at the last update (the observed occupant)
    house: mx.array  # (N,) int64 house id (the founder's uid), HOUSE_UNKNOWN, -1 dead
    house_from: (
        mx.array
    )  # (N,) int64 the house of the founder's parent, inherited with the house: -1 none (founder of the run,
    #                 immigrant, observer start), HOUSE_UNKNOWN if the parent's house was unknown; -1 dead
    kin: (
        mx.array
    )  # (N,) int32 kin slot: 0..F-1 named, F = OTHER (untracked, unaffiliated, immigrant), -1 dead
    colony: mx.array  # (N,) int32 colony slot: 0..F-1 named, F = other tracked colony, -1 none or unknown
    body: mx.array  # (N,) int32 bond-component size at the last snapshot, 0 if unknown (reset rows)
    slot_group: mx.array  # (N,B) int32 kin slot of each slot's partner after the last update, -1 empty
    rel: mx.array  # (5,G,G) int64 window counts, Rel order, [actor slot, target slot]
    expo: mx.array  # (2,G,G,K) int64 window neighbour lanes of attackers / proposers,
    #                 [actor slot, neighbour slot, n_i - 1] with n_i the actor's valid sensed neighbours
    shared: mx.array  # (G,G) int64 window energy shared, [sender slot, receiver slot]
    births: mx.array  # (G,) int64 window births by the slot the newborn inherits (its parent's)
    deaths: mx.array  # (C,G) int64 window deaths by DeathCause (rows) and the dead row's slot


class Snap(NamedTuple):
    """The arrays of one snapshot, all derived (real ops: no ``State`` array is held)."""

    uid: mx.array  # (N,) int64 where(alive, uid, -1)
    parent_uid: mx.array  # (N,) int64 where(alive, ., -1)
    clade: mx.array  # (N,) int32 where(alive, ., -1)
    partner: mx.array  # (N,B) int32 where(alive, ., -1)
    tag: mx.array  # (N,T) float32 where(alive, ., 0)
    pos: mx.array  # (N,nd) float32 where(alive, ., 0)
    arch: mx.array  # (N,) int32 primary archetype label (rules.Archetype, 0 unlabelled) where(alive, ., 0)
    next_uid: mx.array  # () int64 maximum(next_uid, 0)
    comp: mx.array  # (N,) int32 metrics.components labels
    body: mx.array  # (N,) int32 component size per alive row, 0 on dead rows
    converged: mx.array  # () bool
    bins: mx.array  # (N,) int32 grid_code(tag)
    bin_counts: mx.array  # (kin_bins^T,) int32 alive rows per bin
    cell: mx.array  # (N,) int32 plane_cell at terr_cells
    foot: mx.array  # (N,) int32 plane_cell at foot_cells
    terr: mx.array  # (terr_cells^2 * G,) int32 counts [cell, slot] with the window's slots
    house: mx.array  # (N,) int64 so.house
    house_from: mx.array  # (N,) int64 so.house_from
    q_sums: mx.array  # (4,G) int64 temperament sums: attack+, attack-, bond+, bond- (units of 1/256)
    q_rows: mx.array  # (G,) int32 rows per start-of-step slot
    rel: mx.array  # (5,G,G) int64 the window matrices at the snapshot step
    expo: mx.array  # (2,G,G,K) int64
    shared: mx.array  # (G,G) int64
    births: mx.array  # (G,) int64
    deaths: mx.array  # (C,G) int64


# --- init -------------------------------------------------------------------------------------------------


def _slot_groups(partner: mx.array, kin: mx.array) -> mx.array:
    """(N,B) int32 kin slot of each bond slot's partner, -1 for an empty slot."""
    return mx.where(partner >= 0, kin[mx.maximum(partner, 0)], -1)


def init_soc(si: SocIn, cfg: Config, sr: SocialRules) -> SocState:
    """The social state of a state: every alive row in OTHER, no colony, no body, empty windows.

    A founder's house is its clade (clade = uid for the founders, ``state.py``). Immigrants and all their
    descendants carry clade ``n_init`` (``lifecycle.reproduce``), which is not a house: uid ``n_init`` is the
    run's first birth, an unrelated agent. An observer attached to a running world therefore seeds those
    lineages with ``HOUSE_UNKNOWN`` (DESIGN §29.4); a fresh observer at step 0 has none alive.
    """
    d = dims(cfg)
    g = sr.slots + 1
    alive = si.alive
    house = mx.where(
        alive, mx.where(si.clade == cfg.population.n_init, HOUSE_UNKNOWN, si.clade.astype(_I64)), -1
    )
    kin = mx.where(alive, sr.slots, -1).astype(_I32)
    return SocState(
        uid=mx.where(alive, si.uid, -1),
        house=house.astype(_I64),
        house_from=mx.full((d.n,), -1, dtype=_I64),
        kin=kin,
        colony=mx.full((d.n,), -1, dtype=_I32),
        body=mx.zeros((d.n,), dtype=_I32),
        slot_group=_slot_groups(si.partner, kin).astype(_I32),
        rel=mx.zeros((len(Rel), g, g), dtype=_I64),
        expo=mx.zeros((len(Expo), g, g, d.k), dtype=_I64),
        shared=mx.zeros((g, g), dtype=_I64),
        births=mx.zeros((g,), dtype=_I64),
        deaths=mx.zeros((len(death_causes(cfg)), g), dtype=_I64),
    )


# --- the per-step fold ------------------------------------------------------------------------------------


def update(so: SocState, si: SocIn, a: Acts, cfg: Config, sr: SocialRules) -> SocState:
    """Fold one step into the social state. ``si`` is the post-step state's view and ``a`` the step's channel
    (the same rows: a row's acts are those of the occupant it had at the start of the step).

    Houses, slots and bodies follow the reset rule of G1's observer: a row keeps its values only while the
    same occupant (same uid) lives in it; a newborn row takes its birth lane's. Relations are counted with
    the start-of-step slots ``so.kin`` (every acting row and every target was alive then) and added to the
    window matrices, which are exact int64 functions of the channel until ``relabel`` resets them. The births
    (by the slot the newborn inherits: its parent's at the start of the step) and the deaths by cause (by the
    dead row's start-of-step slot) are counted the same way.
    """
    d = dims(cfg)
    n, k = d.n, d.k
    g, other = sr.slots + 1, sr.slots

    keep = si.alive & (si.uid == so.uid)  # G1's reset rule: died, or a new uid in the row, resets it
    lane = groups.lane_rows(n, a.born_row)
    birth = a.born_parent >= 0
    p = mx.maximum(a.born_parent, 0)  # the parent row of a birth lane: a survivor, never a child row
    founded = (a.born_row >= 0) & (~birth | groups.hash_below(a.born_uid, cfg.run.seed, cfg.game.house_rate))
    house = groups.inherit(so.house, keep, lane, mx.where(founded, a.born_uid, so.house[p]), -1)
    # the house a new house comes from: the founder's parent's house (-1 for an immigrant), inherited with it
    house_from = groups.inherit(
        so.house_from, keep, lane, mx.where(founded, mx.where(birth, so.house[p], -1), so.house_from[p]), -1
    )
    kin = groups.inherit(so.kin, keep, lane, mx.where(birth, so.kin[p], other), -1)
    colony = mx.where(keep, so.colony, -1)
    body = mx.where(keep, so.body, 0)

    # Relations of this step, with the start-of-step slots g0 (every acting row and its target were alive).
    g0 = so.kin
    f = a.flags.astype(_I32)
    att = (f & ActFlag.ATTACKED.value) != 0
    hit = (f & ActFlag.HIT.value) != 0
    prop = (f & ActFlag.PROPOSED.value) != 0
    ta = mx.maximum(a.attack_target, 0)
    tb = mx.maximum(a.bond_target, 0)
    kill = att & ((f[ta] & ActFlag.KILLED.value) != 0)  # the victim died KILLED at the step it was a target
    valid = a.nbr >= 0
    gn = mx.where(valid, g0[mx.maximum(a.nbr, 0)], -1)  # (N,K) neighbour slots

    def pair(s: mx.array, t: mx.array) -> mx.array:
        return s * g + t

    gg = g * g
    bond_slot_ok = (g0[:, None] >= 0) & (so.slot_group >= 0)
    cidx = mx.concatenate(
        [
            Rel.ATTACKS.value * gg + pair(g0, g0[ta]),
            Rel.HITS.value * gg + pair(g0, g0[ta]),
            Rel.KILLS.value * gg + pair(g0, g0[ta]),
            Rel.BONDS.value * gg + pair(g0, g0[tb]),
            (Rel.BOND_SLOTS.value * gg + pair(g0[:, None], so.slot_group)).reshape(-1),
        ]
    )
    cmask = mx.concatenate([att, hit, kill, prop, bond_slot_ok.reshape(-1)])
    rel = so.rel + groups.masked_bincount(len(Rel) * gg, cidx, cmask).astype(_I64).reshape(len(Rel), g, g)

    # The neighbour exposure of attackers and proposers: for an actor with n_i valid sensed neighbours, each
    # valid neighbour lane in slot b adds one to [actor slot, b, n_i - 1].
    ni = mx.maximum(valid.astype(_I32).sum(axis=1), 1)[:, None] - 1  # n_i - 1; actors have n_i >= 1
    e = pair(g0[:, None], mx.maximum(gn, 0)) * k + ni  # (N,K)
    expo = so.expo + groups.masked_bincount(
        len(Expo) * gg * k,
        mx.concatenate([e, Expo.BOND.value * gg * k + e]),
        mx.concatenate([att[:, None] & valid, prop[:, None] & valid]),
    ).astype(_I64).reshape(len(Expo), g, g, k)

    # Energy shared, from the sender's slot to the receiver's (the bond slot's partner at the start).
    shared = so.shared + groups.masked_bincount_exact(
        gg,
        pair(g0[:, None], so.slot_group),
        a.share_out,
        (a.share_out > 0) & (so.slot_group >= 0),
        bits=16,
    ).reshape(g, g)

    # Births and deaths of the window, by the start-of-step slots. A birth lane has a child row and a parent
    # that was alive at the start; a row with a death cause was alive at the start (the slot guards only
    # keep an inconsistent synthetic input from counting in another cause's bins).
    parent_slot = g0[p]
    is_birth = (a.born_row >= 0) & birth & (parent_slot >= 0)
    births = so.births + groups.masked_bincount(g, parent_slot, is_birth).astype(_I64)
    cause = a.death_cause.astype(_I32)
    n_causes = len(death_causes(cfg))
    deaths = so.deaths + groups.masked_bincount(
        n_causes * g, cause * g + g0, (cause >= 0) & (cause < n_causes) & (g0 >= 0)
    ).astype(_I64).reshape(n_causes, g)

    slot_group = _slot_groups(si.partner, kin)
    return SocState(
        mx.where(si.alive, si.uid, -1),
        house,
        house_from,
        kin,
        colony,
        body,
        slot_group,
        rel,
        expo,
        shared,
        births,
        deaths,
    )


# --- the snapshot -----------------------------------------------------------------------------------------


def snapshot(
    kin0: mx.array,
    so: SocState,
    si: SocIn,
    a: Acts | None,
    cfg: Config,
    sr: SocialRules,
    arch: mx.array | None = None,
) -> Snap:
    """The arrays of a snapshot step (evaluated once, then pulled to the host by ``Society.settle``).

    ``kin0`` is the slot array before this step's update (the start-of-step slots, which the step's
    temperament is read with); ``so`` and ``si`` are the updated ones. ``a`` is None for the initial snapshot
    of a fresh observer (static): the temperament is then zero. The territory counts use ``so.kin``, the
    slots of the window, so that territory, relations and temperament of one record share one slot map.
    ``arch`` (N,) int32 is each row's primary archetype label (``Labels.arch_primary``, 0 where unlabelled);
    None (the initial snapshot of a fresh observer, direct callers) reads as all unlabelled.
    """
    d = dims(cfg)
    n = d.n
    g = sr.slots + 1
    alive = si.alive
    a1 = alive[:, None]

    comp, converged = metrics.components(si.partner, alive, sr.colony_iters)
    size = groups.masked_bincount(n, comp, alive)
    body = mx.where(alive, size[comp], 0)

    tag = mx.where(a1, si.tag, 0.0)
    pos = mx.where(a1, si.pos, 0.0)
    arch = mx.where(alive, si.clade * 0 if arch is None else arch, 0).astype(_I32)  # derived from an input
    bins = groups.grid_code(tag, -1.0, 1.0, sr.kin_bins)
    bin_counts = groups.masked_bincount(sr.kin_bins**layout.T, bins, alive)
    cell = groups.plane_cell(pos, cfg.world.size, sr.terr_cells, axes=(0, 1))
    foot = groups.plane_cell(pos, cfg.world.size, sr.foot_cells, axes=(0, 1))
    terr = groups.masked_bincount(sr.terr_cells**2 * g, cell * g + so.kin, alive & (so.kin >= 0))

    if a is None:  # no step to read: zeros derived from the window matrices (never a constant output)
        q_sums = so.rel[:4, :, 0] * 0
        q_rows = (so.rel[0, :, 0] * 0).astype(_I32)
    else:
        q = a.q_kin.astype(_I32)  # (N,2): attack, bond; units of 1/256
        parts = mx.stack(
            [
                mx.maximum(q[:, 0], 0),
                mx.maximum(-q[:, 0], 0),
                mx.maximum(q[:, 1], 0),
                mx.maximum(-q[:, 1], 0),
            ],
            axis=1,
        )
        idx = mx.arange(4, dtype=_I32)[None, :] * g + kin0[:, None]
        q_sums = groups.masked_bincount_exact(4 * g, idx, parts, (kin0 >= 0)[:, None], bits=15).reshape(4, g)
        q_rows = groups.masked_bincount(g, kin0, kin0 >= 0)

    return Snap(
        uid=mx.where(alive, si.uid, -1),
        parent_uid=mx.where(alive, si.parent_uid, -1),
        clade=mx.where(alive, si.clade, -1),
        partner=mx.where(a1, si.partner, -1),
        tag=tag,
        pos=pos,
        arch=arch,
        next_uid=mx.maximum(si.next_uid, 0),
        comp=comp,
        body=body,
        converged=converged,
        bins=bins,
        bin_counts=bin_counts,
        cell=cell,
        foot=foot,
        terr=terr,
        house=mx.where(alive, so.house, -1),
        house_from=mx.where(alive, so.house_from, -1),
        q_sums=q_sums,
        q_rows=q_rows,
        rel=so.rel + 0,
        expo=so.expo + 0,
        shared=so.shared + 0,
        births=so.births + 0,
        deaths=so.deaths + 0,
    )


def relabel(so: SocState, snap: Snap, kin_lut: mx.array, col_lut: mx.array) -> SocState:
    """Write the host's slots back (at settle) and reset the window.

    ``kin_lut`` (kin_bins^T,) int32 maps a tag bin to its group's slot (OTHER if none) and ``col_lut`` (N,)
    int32 maps a component label to its colony slot (F for a tracked colony without one, -1 if untracked).
    """
    alive = snap.uid >= 0
    kin = mx.where(alive, kin_lut[snap.bins], -1)
    colony = mx.where(alive, col_lut[snap.comp], -1)
    slot_group = _slot_groups(snap.partner, kin)
    return so._replace(
        uid=mx.where(alive, snap.uid, -1),
        house=mx.where(alive, so.house, -1),
        house_from=mx.where(alive, so.house_from, -1),
        kin=kin,
        colony=colony,
        body=snap.body + 0,
        slot_group=slot_group,
        rel=so.rel * 0,
        expo=so.expo * 0,
        shared=so.shared * 0,
        births=so.births * 0,
        deaths=so.deaths * 0,
    )
