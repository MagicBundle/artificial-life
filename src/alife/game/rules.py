"""Archetype and intent rules of the observer layer (docs/GAMES.md §4.4, DESIGN §28.4).

Pure, vectorised MLX over the windowed means of the observer accumulators (``observe.py``): every function
takes means and returns arrays, so each is unit-testable on synthetic ones. All arithmetic is integer, with
the thresholds as (numerator, denominator) pairs of a frozen ``RuleSet`` (``den·x >= num·y`` for the ratio
``num/den``), so a label is an exact function of the accumulators.

``INTENT_VERSION`` covers every archetype and intent rule, threshold, window and the order. ``RULES_V1`` is
its instance 1, provisional until the calibration run (the study tool, ``analysis/labels.py``);
``RuleSet.digest()`` of it is pinned in the tests, so a threshold edit without a version bump fails the
suite, and the observer sidecar stores the digest. Thresholds are code constants, not config keys: labels
mean the same in every world.

This module is the leaf of ``alife.game``: it also holds the accumulator column layout and the two bounds
that ``RuleSet.check`` needs (``observe.py`` re-exports them), so it imports nothing from ``observe``.

Lint (rule 5, ``tests/test_lint.py``): enum constants are written ``X.Y.value``, never ``int(X.Y)``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from enum import IntEnum
from typing import Final

import mlx.core as mx

from alife import layout
from alife.config import Config

__all__ = [
    "ARCHETYPE_NAMES",
    "ARCH_PRIORITY",
    "EVAL_BIT",
    "INTENT_NAMES",
    "INTENT_VERSION",
    "N_ARCH_BASE",
    "N_INTENT_COLS",
    "RULES_V1",
    "VQ_MAX",
    "ArchCol",
    "Archetype",
    "Intent",
    "IntentCol",
    "RuleSet",
    "archetype_labels",
    "archetype_matches",
    "first_intent",
    "hysteresis",
    "intent_matches",
    "n_arch_cols",
    "pack_bits",
]

INTENT_VERSION: Final = 1
VQ_MAX: Final = 1 << 15  # clip of the quantised velocity components and speed (units of u/256)
EVAL_BIT: Final = 1 << 30  # set in ``match_intent`` of every row that was evaluated (past intent warm-up)
_INT32_MAX: Final = (1 << 31) - 1

_I32 = mx.int32
_I64 = mx.int64


class ArchCol(IntEnum):
    """Archetype accumulator columns (``ObsState.arch``); then V_POS[c] = 5 + c and V_NEG[c] = 5 + nd + c."""

    HARVEST = 0
    GAIN = 1
    INFLOW = 2
    OUTFLOW = 3
    SPEED = 4


N_ARCH_BASE: Final = 5  # the columns before the velocity halves


def n_arch_cols(nd: int) -> int:
    """Archetype accumulator width: 9 in 2D, 11 in 3D."""
    return N_ARCH_BASE + 2 * nd


class IntentCol(IntEnum):
    """Intent accumulator columns (``ObsState.intent``)."""

    ATTACK = 0
    HURT = 1
    THRUST = 2
    DEFEND = 3
    EMIT = 4
    HARVEST = 5
    CARE = 6


N_INTENT_COLS: Final = 7


class Archetype(IntEnum):
    NONE = 0
    GATHERER = 1
    HUNTER = 2
    PROVIDER = 3
    TRADER = 4
    NOMAD = 5
    SETTLER = 6
    GENERALIST = 7


class Intent(IntEnum):
    NONE = 0
    BUDDING = 1
    HUNTING = 2
    FLEEING = 3
    DEFENDING = 4
    CARING = 5
    BONDING = 6
    SIGNALLING = 7
    FEEDING = 8
    ROAMING = 9
    RESTING = 10


# Income before movement (GAMES §4.1 "lives on"): the first match is the primary label.
ARCH_PRIORITY: Final = (
    Archetype.TRADER,
    Archetype.PROVIDER,
    Archetype.HUNTER,
    Archetype.GATHERER,
    Archetype.NOMAD,
    Archetype.SETTLER,
)
ARCHETYPE_NAMES: Final = tuple(a.name.lower() for a in Archetype if a is not Archetype.NONE)
INTENT_NAMES: Final = tuple(i.name.lower() for i in Intent if i is not Intent.NONE)
_N_ARCH_RULES: Final = 6  # Gatherer..Settler (Generalist is "none of them")
_N_INTENT_RULES: Final = 9  # Budding..Roaming (Resting is "none of them")


@dataclasses.dataclass(frozen=True)
class RuleSet:
    """Every window, threshold and period of the labels (``INTENT_VERSION`` 1 = the defaults).

    Windows: ``arch_k`` and ``intent_k`` are the accumulator shifts (about ``3·2^k`` steps to 95%). Labels are
    evaluated when ``state.step % every == 0``, and a displayed intent changes after ``hold`` consecutive
    evaluations. ``arch_warm`` and ``intent_warm`` are the observed steps before a row is labelled.

    A pair ``(num, den)`` is the ratio ``num/den`` and is tested as ``den·x >= num·y`` (``<`` for Settler),
    in integers. Inputs are the means ``m = A >> k``; a flag enters as 256 per set step; speed and velocity
    are in units of ``u/256`` (``observe.speed_unit``); ``I = m_harvest + m_gain + m_inflow`` is the windowed
    income.
    """

    arch_k: int = 6
    intent_k: int = 3
    every: int = 8
    hold: int = 4
    arch_warm: int = 192  # 3 << arch_k
    intent_warm: int = 24  # 3 << intent_k
    gatherer: tuple[int, int] = (7, 10)  # I >= 1 and m_harvest/I >= 0.7
    hunter: tuple[int, int] = (7, 20)  # m_gain >= 1 and m_gain/I >= 0.35
    provider: tuple[int, int] = (1, 5)  # m_outflow >= 1 and m_outflow/I >= 0.2
    trader: tuple[int, int] = (1, 2)  # Provider and a slot with min(out, in) >= 1 and min/max >= 0.5
    nomad_speed: tuple[int, int] = (512, 5)  # m_speed >= 102.4 units (0.4u)
    nomad_persist: tuple[int, int] = (9, 25)  # |mean velocity|^2 / m_speed^2 >= 0.36 (persistence >= 0.6)
    settler_speed: tuple[int, int] = (256, 10)  # m_speed < 25.6 units (0.1u); the colony clause is G2
    bud_window: int = 32  # Budding: steps since the last bud < this (= every * hold)
    hunt: int = 64  # Hunting: m_attack >= this (flag mean 0.25)
    flee_hurt: int = 8  # Fleeing: m_hurt >= this (hurt within about 11 steps) ...
    flee_thrust: int = 128  # ... and m_thrust >= this (0.5)
    defend: int = 128  # Defending: m_defend >= this (0.5)
    care: tuple[int, int] = (1, 10)  # Caring: m_care >= 1 and m_care >= harvest_max/10
    bond_age: tuple[int, int] = (24, 120)  # Bonding: a non-kin slot with lo <= age < hi
    signal: int = 128  # Signalling: m_emit >= this deposit quanta per step (about 0.5)
    feed: tuple[int, int] = (1, 2)  # Feeding: m_harvest(k_intent) >= 1 and >= harvest_max/2
    roam: tuple[int, int] = (768, 10)  # Roaming: m_thrust >= 76.8 (0.3)

    def __post_init__(self) -> None:
        def req(ok: bool, msg: str) -> None:
            if not ok:
                raise ValueError(f"RuleSet: {msg}")

        req(1 <= self.arch_k <= 30 and 1 <= self.intent_k <= 30, "arch_k and intent_k must be in [1, 30]")
        req(self.every >= 1 and self.hold >= 1, "every and hold must be >= 1")
        req(self.arch_warm >= 0 and self.intent_warm >= 0, "arch_warm and intent_warm must be >= 0")
        for name in _RATIOS:
            num, den = getattr(self, name)
            req(num >= 0 and den > 0, f"{name} must be a (numerator >= 0, denominator > 0) pair")
        req(0 <= self.bond_age[0] < self.bond_age[1], "bond_age must be (lo, hi) with 0 <= lo < hi")

    def digest(self) -> str:
        """sha256 of the canonical JSON of every field (tuples as lists): the identity of these rules."""
        text = json.dumps(dataclasses.asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def check(self, cfg: Config) -> None:
        """Raise ``ValueError`` unless every accumulator column fits int32: ``X·(2^k + 1) <= 2^31 − 1`` with
        ``X`` the column's per-step maximum under ``cfg`` (``A <= X·2^k`` is invariant and ``A + x`` is the
        largest intermediate). A calibration sweep with a large ``k`` fails loudly instead of wrapping."""
        flow = layout.B * cfg.bonds.share_rate  # a row's total over its B bond slots
        arch = (
            ("arch.harvest", cfg.energy.harvest_max),
            ("arch.gain", cfg.combat.attack_max),
            ("arch.inflow", flow),
            ("arch.outflow", flow),
            ("arch.speed", VQ_MAX),
            ("arch.velocity", VQ_MAX),
            ("arch_slot.share", cfg.bonds.share_rate),
        )
        intent = (
            ("intent.attack", 256),  # a flag enters as 256
            ("intent.hurt", 256),
            ("intent.thrust", 256),
            ("intent.defend", 256),
            ("intent.emit", layout.NCH * 255),  # Σ_c floor(256·emit_c) <= 765
            ("intent.harvest", cfg.energy.harvest_max),
            ("intent.care", flow),
        )
        for k, cols in ((self.arch_k, arch), (self.intent_k, intent)):
            for column, x in cols:
                if x * ((1 << k) + 1) > _INT32_MAX:
                    raise ValueError(
                        f"RuleSet.check: column {column} with k = {k} overflows int32: "
                        f"{x}·(2^{k} + 1) = {x * ((1 << k) + 1)} > 2^31 − 1"
                    )


_RATIOS: Final = (
    "gatherer",
    "hunter",
    "provider",
    "trader",
    "nomad_speed",
    "nomad_persist",
    "settler_speed",
    "care",
    "feed",
    "roam",
)

RULES_V1: Final = RuleSet()


def pack_bits(match: mx.array) -> mx.array:
    """(N, n) bool → (N,) int32 with bit j set where column j matched (``1 << array`` is unsupported)."""
    weights = mx.array([1 << j for j in range(match.shape[1])], dtype=_I32)
    return (match.astype(_I32) * weights).sum(axis=1)


# --- archetypes -----------------------------------------------------------------------------------------


def archetype_matches(m6: mx.array, m6s: mx.array, cfg: Config, rs: RuleSet) -> mx.array:
    """(N, 6) bool, one column per rule in enum order 1–6 (Gatherer..Settler), non-exclusive.

    ``m6`` is the (N, 5 + 2·nd) mean of ``arch`` (``ArchCol``, then the velocity halves V_POS and V_NEG),
    ``m6s`` the (N, B, 2) mean of ``arch_slot`` (share out, share got per bond slot).
    """
    del cfg  # no archetype threshold depends on the world
    nd = (m6.shape[1] - N_ARCH_BASE) // 2
    h, g, inflow, out, sp = (m6[:, c.value] for c in ArchCol)
    income = h + g + inflow
    gatherer = (income >= 1) & (rs.gatherer[1] * h >= rs.gatherer[0] * income)
    hunter = (g >= 1) & (rs.hunter[1] * g >= rs.hunter[0] * income)
    provider = (out >= 1) & (rs.provider[1] * out >= rs.provider[0] * income)
    sent, got = m6s[..., 0], m6s[..., 1]
    low, high = mx.minimum(sent, got), mx.maximum(sent, got)
    paired = ((low >= 1) & (rs.trader[1] * low >= rs.trader[0] * high)).any(axis=1)
    trader = provider & paired
    s64 = sp.astype(_I64)
    mean_v = (m6[:, N_ARCH_BASE : N_ARCH_BASE + nd] - m6[:, N_ARCH_BASE + nd :]).astype(_I64)
    v2 = (mean_v * mean_v).sum(axis=1)
    nomad = (rs.nomad_speed[1] * s64 >= rs.nomad_speed[0]) & (
        rs.nomad_persist[1] * v2 >= rs.nomad_persist[0] * s64 * s64
    )
    settler = rs.settler_speed[1] * s64 < rs.settler_speed[0]
    return mx.stack([gatherer, hunter, provider, trader, nomad, settler], axis=1)


def archetype_labels(match: mx.array) -> tuple[mx.array, mx.array]:
    """``(primary, secondary)`` (N,) int32 ``Archetype`` values from an ``archetype_matches`` matrix.

    Provider is masked out where Trader matches (a Trader is never a Provider secondary). The primary is the
    first match in ``ARCH_PRIORITY``, else GENERALIST; the secondary is the second match, else NONE.
    """
    trader = match[:, Archetype.TRADER.value - 1]
    cols = [
        match[:, a.value - 1] & ~trader if a is Archetype.PROVIDER else match[:, a.value - 1]
        for a in ARCH_PRIORITY
    ]
    pm = mx.stack(cols, axis=1)
    order = mx.arange(_N_ARCH_RULES, dtype=_I32)
    n = mx.cumsum(pm.astype(_I32), axis=1)
    first = mx.min(mx.where(pm, order, _N_ARCH_RULES), axis=1)
    second = mx.min(mx.where(pm & (n == 2), order, _N_ARCH_RULES), axis=1)
    values = mx.array([a.value for a in ARCH_PRIORITY], dtype=_I32)
    primary = mx.where(
        first < _N_ARCH_RULES, values[mx.minimum(first, _N_ARCH_RULES - 1)], Archetype.GENERALIST.value
    )
    secondary = mx.where(
        second < _N_ARCH_RULES, values[mx.minimum(second, _N_ARCH_RULES - 1)], Archetype.NONE.value
    )
    return primary.astype(_I32), secondary.astype(_I32)


# --- intents --------------------------------------------------------------------------------------------


def intent_matches(
    m3: mx.array,
    since_bud: mx.array,
    slot_age: mx.array,
    slot_kin: mx.array,
    cfg: Config,
    rs: RuleSet,
) -> mx.array:
    """(N, 9) bool, one column per predicate in the order of ``Intent`` 1–9 (Budding..Roaming).

    ``m3`` is the (N, 7) mean of ``intent`` (``IntentCol``); ``since_bud`` (N,), ``slot_age`` (N, B) and
    ``slot_kin`` (N, B) are the observer's exact (non-accumulated) inputs.
    """
    harvest_max = cfg.energy.harvest_max
    thrust = m3[:, IntentCol.THRUST.value]
    care, harvest = m3[:, IntentCol.CARE.value], m3[:, IntentCol.HARVEST.value]
    lo, hi = rs.bond_age
    cols = [
        since_bud < rs.bud_window,  # BUDDING
        m3[:, IntentCol.ATTACK.value] >= rs.hunt,  # HUNTING
        (m3[:, IntentCol.HURT.value] >= rs.flee_hurt) & (thrust >= rs.flee_thrust),  # FLEEING
        m3[:, IntentCol.DEFEND.value] >= rs.defend,  # DEFENDING
        (care >= 1) & (rs.care[1] * care >= rs.care[0] * harvest_max),  # CARING
        (~slot_kin & (slot_age >= lo) & (slot_age < hi)).any(axis=1),  # BONDING
        m3[:, IntentCol.EMIT.value] >= rs.signal,  # SIGNALLING
        (harvest >= 1) & (rs.feed[1] * harvest >= rs.feed[0] * harvest_max),  # FEEDING
        rs.roam[1] * thrust >= rs.roam[0],  # ROAMING
    ]
    return mx.stack(cols, axis=1)


def first_intent(match: mx.array) -> mx.array:
    """(N,) int32 ``Intent``: the first matching predicate of an ``intent_matches`` matrix, else RESTING."""
    order = mx.arange(_N_INTENT_RULES, dtype=_I32)
    first = mx.min(mx.where(match, order, _N_INTENT_RULES), axis=1)
    return mx.where(first < _N_INTENT_RULES, first + 1, Intent.RESTING.value).astype(_I32)


def hysteresis(
    label: mx.array, cand: mx.array, cand_n: mx.array, raw: mx.array, hold: int
) -> tuple[mx.array, mx.array, mx.array]:
    """One evaluation of the displayed intent: ``(label, cand, cand_n)`` after seeing ``raw``.

    A raw label that differs from the displayed one becomes the candidate; the label switches to it once it
    has held at ``hold`` consecutive evaluations. A return to the displayed label clears the candidate, and a
    ``raw`` of NONE (not warm, or reset) clears everything at once.
    """
    same = raw == label
    n = mx.where(same, 0, mx.where(raw == cand, cand_n + 1, 1))
    c = mx.where(same, label, raw)
    switch = n >= hold  # the candidate held at `hold` consecutive evaluations
    label, n = mx.where(switch, c, label), mx.where(switch, 0, n)  # after a switch the candidate is the label
    off = raw == Intent.NONE.value
    none = Intent.NONE.value
    return (
        mx.where(off, none, label).astype(_I32),
        mx.where(off, none, c).astype(_I32),
        mx.where(off, 0, n).astype(_I32),
    )
