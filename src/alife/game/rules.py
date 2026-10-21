"""Archetype and intent rules of the observer layer (docs/GAMES.md §4.4, DESIGN §28.4).

Pure, vectorised MLX over the windowed means of the observer accumulators (``observe.py``): every function
takes means and returns arrays, so each is unit-testable on synthetic ones. All arithmetic is integer, with
the thresholds as (numerator, denominator) pairs of a frozen ``RuleSet`` (``den·x >= num·y`` for the ratio
``num/den``), so a label is an exact function of the accumulators.

``INTENT_VERSION`` covers every archetype and intent rule, threshold, window and the order. ``RULES_V2`` (the
default, ``RuleSet()``) is its instance 2: the G1 rules with the calibration fixes of the labels that failed
G1's validity study, calibrated on the G1 study windows and validated on held-out ones (DESIGN §28.4, §28.8).
``RULES_V1`` stays defined, naming its version-1 values, and evaluates exactly as G1 did. ``RuleSet.digest()``
of both is pinned in the tests, so a threshold edit without a version bump fails the suite, and the observer
sidecar stores the digest. Thresholds are code constants, not config keys: labels mean the same in every
world.

A field added after version 1 has an inert value that reproduces version 1 (``_POST_V1_INERT``), and
``digest()`` leaves it out while it holds that value, so ``RULES_V1`` keeps its G1 pin. Version 2's rules are
those fields: ``settler_body`` (the Settler colony clause, DESIGN §28.4, §29: a row also matches Settler when
its bond component, as of the last social snapshot, has at least that many members, and the rows' body sizes
are required when it is positive), ``gatherer_level``, ``defend_hurt``, ``bond_rate``, ``bud_horizon`` with
``bud_cost`` and ``bud_overlay`` (the Budding forecast, which needs the post-step ``energy`` and
``brain_cost``), and ``intent_order``.

This module is the leaf of ``alife.game``: it also holds the accumulator column layout and the two bounds
that ``RuleSet.check`` needs (``observe.py`` re-exports them), so it imports nothing from ``observe``.

Lint (rule 5, ``tests/test_lint.py``): enum constants are written ``X.Y.value``, never ``int(X.Y)``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from enum import IntEnum
from types import MappingProxyType
from typing import Final, NamedTuple

import mlx.core as mx

from alife import layout
from alife.config import Config

__all__ = [
    "ARCHETYPE_DISPLAY",
    "ARCHETYPE_NAMES",
    "ARCH_PRIORITY",
    "EVAL_BIT",
    "INTENT_DISPLAY",
    "INTENT_NAMES",
    "INTENT_VERSION",
    "N_ARCH_BASE",
    "N_INTENT_COLS",
    "RULES_V1",
    "RULES_V2",
    "VQ_MAX",
    "ArchCol",
    "Archetype",
    "BudIn",
    "Intent",
    "IntentCol",
    "RuleSet",
    "archetype_labels",
    "archetype_matches",
    "budding_forecast",
    "first_intent",
    "hysteresis",
    "intent_matches",
    "n_arch_cols",
    "pack_bits",
]

INTENT_VERSION: Final = 2
VQ_MAX: Final = 1 << 15  # clip of the quantised velocity components and speed (units of u/256)
EVAL_BIT: Final = 1 << 30  # set in ``match_intent`` of every row that was evaluated (past intent warm-up)
_INT32_MAX: Final = (1 << 31) - 1
_N_INTENT_RULES: Final = 9  # Budding..Roaming (Resting is "none of them")
_V1_INTENT_ORDER: Final = (1, 2, 3, 4, 5, 6, 7, 8, 9)
# RuleSet fields added after version 1, with the value at which they are inert (it reproduces version 1
# exactly). ``digest()`` leaves such a field out while it holds that value, so the digest of a rule set that
# does not use it is the one it had before the field existed (the pattern of ``config_hash`` for post-v0.1
# keys).
_POST_V1_INERT: Final = {
    "settler_body": 0,
    "gatherer_level": (0, 1),
    "bud_horizon": 0,
    "bud_cost": 0,
    "bud_overlay": False,
    "defend_hurt": 0,
    "bond_rate": 0,
    "intent_order": _V1_INTENT_ORDER,
}

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
    FORMED = 7  # 256 per step with ActFlag.FORMED (Bonding, version 2)
    FLOW_POS = 8  # max(flow, 0), flow = the visible net energy flow of the step (Budding, version 2)
    FLOW_NEG = 9  # max(-flow, 0)


N_INTENT_COLS: Final = 10


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
# The names a player sees, by key. The keys (the enum names, lower case) stay what the reports, the metrics
# stream and the sidecar use; this is the one table of the human-facing words (a rename touches only it, the
# docs and the tests that pin it). "Unspecialised" is the label of a row that matched no income or movement
# rule; "Tethering" is forming and dropping bonds again and again (in 3D mostly re-forming a pair seen
# earlier; not measured in 2D): the formation-rate rule of version 2, not the lasting new partner that the
# word "bonding" promises).
ARCHETYPE_DISPLAY: Final = MappingProxyType(
    {
        "gatherer": "Gatherer",
        "hunter": "Hunter",
        "provider": "Provider",
        "trader": "Trader",
        "nomad": "Nomad",
        "settler": "Settler",
        "generalist": "Unspecialised",
    }
)
INTENT_DISPLAY: Final = MappingProxyType(
    {
        "budding": "Budding",
        "hunting": "Hunting",
        "fleeing": "Fleeing",
        "defending": "Defending",
        "caring": "Caring",
        "bonding": "Tethering",
        "signalling": "Signalling",
        "feeding": "Feeding",
        "roaming": "Roaming",
        "resting": "Resting",
    }
)
_N_ARCH_RULES: Final = 6  # Gatherer..Settler (Generalist is "none of them")


@dataclasses.dataclass(frozen=True)
class RuleSet:
    """Every window, threshold and period of the labels. The defaults are ``INTENT_VERSION`` 2 (``RULES_V2``);
    ``RULES_V1`` names the version-1 values.

    Windows: ``arch_k`` and ``intent_k`` are the accumulator shifts (about ``3·2^k`` steps to 95%). Labels are
    evaluated when ``state.step % every == 0``, and a displayed intent changes after ``hold`` consecutive
    evaluations. ``arch_warm`` and ``intent_warm`` are the observed steps before a row is labelled.

    A pair ``(num, den)`` is the ratio ``num/den`` and is tested as ``den·x >= num·y`` (``<`` for Settler),
    in integers. Inputs are the means ``m = A >> k``; a flag enters as 256 per set step; speed and velocity
    are in units of ``u/256`` (``observe.speed_unit``); ``I = m_harvest + m_gain + m_inflow`` is the windowed
    income.

    The fields added after version 1 (``_POST_V1_INERT``) qualify the v1 field next to them; at its inert
    value each leaves the version-1 rule exactly as it was, and ``digest()`` leaves it out.
    """

    arch_k: int = 6
    intent_k: int = 3
    every: int = 8
    hold: int = 4
    arch_warm: int = 192  # 3 << arch_k
    intent_warm: int = 24  # 3 << intent_k
    gatherer: tuple[int, int] = (0, 1)  # share: I >= 1 and m_harvest/I >= num/den (v1 (7, 10); (0, 1) = off)
    gatherer_level: tuple[int, int] = (3, 4)  # level: m_harvest >= 1 and >= harvest_max·num/den (v1: off)
    hunter: tuple[int, int] = (7, 20)  # m_gain >= 1 and m_gain/I >= 0.35
    provider: tuple[int, int] = (1, 5)  # m_outflow >= 1 and m_outflow/I >= 0.2
    trader: tuple[int, int] = (1, 2)  # Provider and a slot with min(out, in) >= 1 and min/max >= 0.5
    nomad_speed: tuple[int, int] = (512, 5)  # m_speed >= 102.4 units (0.4u)
    nomad_persist: tuple[int, int] = (9, 25)  # |mean velocity|^2 / m_speed^2 >= 0.36 (persistence >= 0.6)
    settler_speed: tuple[int, int] = (256, 10)  # m_speed < 25.6 units (0.1u); or settler_body below
    settler_body: int = 4  # Settler also if the row's bond component has >= this many members; 0 = off (v1)
    bud_window: int = 32  # v1 Budding: steps since the last bud < this; unused while bud_horizon > 0
    bud_horizon: int = 10  # h: > 0 selects the forecast (static): E < me, E + h·(flow − basal − cost) >= me
    bud_cost: int = 24  # c: quanta per step of action costs the observer does not see (the forecast's c)
    bud_overlay: bool = True  # Budding is displayed at every evaluation where it holds, outside the hold
    hunt: int = 64  # Hunting: m_attack >= this (flag mean 0.25)
    flee_hurt: int = 8  # Fleeing: m_hurt >= this (hurt within about 11 steps) ...
    flee_thrust: int = 128  # ... and m_thrust >= this (0.5)
    defend: int = 192  # Defending: m_defend >= this (0.75; v1 128)
    defend_hurt: int = 1  # Defending also needs m_hurt >= this (0: no condition, v1)
    care: tuple[int, int] = (1, 10)  # Caring: m_care >= 1 and m_care >= harvest_max/10
    # v1 Bonding: a non-kin slot with lo <= age < hi; unused while bond_rate > 0
    bond_age: tuple[int, int] = (24, 120)
    # Bonding: m_formed >= this (> 0 selects it, static); 64 = flag mean 0.25 (period 3 on, period 4 flickers)
    bond_rate: int = 64
    signal: int = 192  # Signalling: m_emit >= this deposit quanta per step (about 0.75; v1 128)
    feed: tuple[int, int] = (3, 4)  # Feeding: m_harvest(k_intent) >= 1 and >= harvest_max·3/4 (v1 (1, 2))
    roam: tuple[int, int] = (768, 10)  # Roaming: m_thrust >= 76.8 (0.3)
    intent_order: tuple[int, ...] = (1, 2, 3, 5, 4, 6, 8, 7, 9)  # Intent values in first-match priority

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
        req(0 <= self.settler_body <= _INT32_MAX, "settler_body must be in [0, 2^31 - 1]")
        req(self.defend_hurt >= 0 and self.bond_rate >= 0, "defend_hurt and bond_rate must be >= 0")
        req(self.bud_horizon >= 0 and self.bud_cost >= 0, "bud_horizon and bud_cost must be >= 0")
        req(isinstance(self.bud_overlay, bool), "bud_overlay must be a bool")
        order = self.intent_order
        req(
            isinstance(order, tuple)
            and all(type(i) is int for i in order)
            and sorted(order) == list(range(1, _N_INTENT_RULES + 1)),
            "intent_order must be a tuple permuting the Intent values 1..9",
        )

    def digest(self) -> str:
        """sha256 of the canonical JSON of every field (tuples as lists): the identity of these rules. A field
        added after version 1 is left out while it holds its inert value (``_POST_V1_INERT``)."""
        fields = {
            k: v
            for k, v in dataclasses.asdict(self).items()
            if k not in _POST_V1_INERT or v != _POST_V1_INERT[k]
        }
        text = json.dumps(fields, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def check(self, cfg: Config) -> None:
        """Raise ``ValueError`` unless every accumulator column fits int32: ``X·(2^k + 1) <= 2^31 − 1`` with
        ``X`` the column's per-step maximum under ``cfg`` (``A <= X·2^k`` is invariant and ``A + x`` is the
        largest intermediate), and unless every int32 ratio comparison ``den·x >= num·y`` does:
        ``max(num, den)·X <= 2^31 − 1`` with ``X`` the largest compared mean. A calibration sweep with a large
        ``k`` or a large ratio fails loudly instead of wrapping."""
        ec = cfg.energy
        flow = layout.B * cfg.bonds.share_rate  # a row's total over its B bond slots
        arch = (
            ("arch.harvest", ec.harvest_max),
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
            ("intent.emit", layout.NCH * 256),  # Σ_c floor(256·emit_c) <= 768 (emit_c = 1.0 is reachable)
            ("intent.harvest", ec.harvest_max),
            ("intent.care", flow),
            ("intent.formed", 256),
            # the visible net flow of a step: gains (harvest, attack gain, Σ got) less losses (the hit's
            # last_loss <= e_max, Σ out, and the four quantised costs, each at most its coefficient)
            ("intent.flow_pos", ec.harvest_max + cfg.combat.attack_max + flow),
            (
                "intent.flow_neg",
                ec.e_max + flow + ec.move_cost + ec.defend_cost + layout.NCH * ec.emit_cost + ec.attack_cost,
            ),
        )
        for k, cols in ((self.arch_k, arch), (self.intent_k, intent)):
            for column, x in cols:
                if x * ((1 << k) + 1) > _INT32_MAX:
                    raise ValueError(
                        f"RuleSet.check: column {column} with k = {k} overflows int32: "
                        f"{x}·(2^{k} + 1) = {x * ((1 << k) + 1)} > 2^31 − 1"
                    )
        harvest_max = ec.harvest_max
        income = harvest_max + cfg.combat.attack_max + flow  # the largest windowed income
        ratios = (  # (field, largest compared mean), the int32 ratio tests of the archetype and intent rules
            ("gatherer", income),
            ("gatherer_level", harvest_max),
            ("hunter", income),
            ("provider", income),
            ("trader", cfg.bonds.share_rate),
            ("care", max(flow, harvest_max)),  # den·care against num·harvest_max
            ("feed", harvest_max),
        )
        for name, x in ratios:
            num, den = getattr(self, name)
            if max(num, den) * x > _INT32_MAX:
                raise ValueError(
                    f"RuleSet.check: ratio {name} = ({num}, {den}) overflows int32: "
                    f"{max(num, den)}·{x} = {max(num, den) * x} > 2^31 − 1"
                )
        num, den = self.roam  # den·thrust (thrust <= 256) against the constant num
        if max(num, den * 256) > _INT32_MAX:
            raise ValueError(
                f"RuleSet.check: ratio roam = ({num}, {den}) overflows int32: "
                f"max({num}, {den}·256) > 2^31 − 1"
            )


_RATIOS: Final = (
    "gatherer",
    "gatherer_level",
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

# Version 1 names its values: every field the version-2 defaults changed, and every post-v1 field at its inert
# value, so ``RULES_V1.digest()`` is the G1 pin and ``RULES_V1`` evaluates exactly as G1 did.
RULES_V1: Final = RuleSet(gatherer=(7, 10), defend=128, signal=128, feed=(1, 2), **_POST_V1_INERT)
RULES_V2: Final = RuleSet()


def pack_bits(match: mx.array) -> mx.array:
    """(N, n) bool → (N,) int32 with bit j set where column j matched (``1 << array`` is unsupported)."""
    weights = mx.array([1 << j for j in range(match.shape[1])], dtype=_I32)
    return (match.astype(_I32) * weights).sum(axis=1)


# --- archetypes -----------------------------------------------------------------------------------------


def archetype_matches(
    m6: mx.array, m6s: mx.array, cfg: Config, rs: RuleSet, body: mx.array | None = None
) -> mx.array:
    """(N, 6) bool, one column per rule in enum order 1–6 (Gatherer..Settler), non-exclusive.

    ``m6`` is the (N, 5 + 2·nd) mean of ``arch`` (``ArchCol``, then the velocity halves V_POS and V_NEG),
    ``m6s`` the (N, B, 2) mean of ``arch_slot`` (share out, share got per bond slot). ``body`` is the (N,)
    bond-component size of each row at the last social snapshot (0 where unknown): Settler also matches where
    ``body >= rs.settler_body``. The gate is static on the rules: with ``rs.settler_body == 0`` the clause is
    off and ``body`` is ignored; with a positive value ``body`` is required (``ValueError`` if None), never
    speed-only labels under a digest that names the colony clause.

    Gatherer is a share of the income (``rs.gatherer``, off at (0, 1)) and a harvest level
    (``rs.gatherer_level``, a fraction of ``harvest_max``, off at (0, 1)); both must hold. Version 1 is the
    share alone, version 2 the level alone.
    """
    nd = (m6.shape[1] - N_ARCH_BASE) // 2
    h, g, inflow, out, sp = (m6[:, c.value] for c in ArchCol)
    income = h + g + inflow
    level_num, level_den = rs.gatherer_level
    gatherer = (
        (income >= 1)
        & (rs.gatherer[1] * h >= rs.gatherer[0] * income)  # share (v1 (7, 10); v2 (0, 1): off)
        & (h >= 1)
        & (level_den * h >= level_num * cfg.energy.harvest_max)  # level (v1 (0, 1): off; v2 (3, 4))
    )
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
    if rs.settler_body > 0:
        if body is None:
            raise ValueError(
                f"archetype_matches: settler_body = {rs.settler_body} needs the rows' body sizes (body=None)"
            )
        settler = settler | (body >= rs.settler_body)
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


class BudIn(NamedTuple):
    """The post-step values the Budding forecast reads at an evaluation (``budding_forecast``)."""

    energy: mx.array  # (N,) int32 State.energy
    brain_cost: mx.array  # (N,) int32 State.brain_cost
    n_bonds: mx.array  # (N,) int32 occupied slots of State.partner
    n_obs: mx.array  # (N,) int32 ObsState.n_obs


def budding_forecast(m3: mx.array, b: BudIn, cfg: Config, rs: RuleSet) -> mx.array:
    """(N,) bool, the row is about to bud: its energy is below ``repro.min_energy`` and, at the windowed net
    flow of the visible energy terms less the basal due and the allowance ``rs.bud_cost`` (the action costs
    the observer does not carry), reaches it within ``rs.bud_horizon`` steps; and it is old enough to bud.
    int64 (``brain_cost`` has no config bound).

    ``m3`` is the (N, 10) mean of ``intent``, so the windowed flow is ``(A_pos >> k) − (A_neg >> k)``. Reading
    ``n_obs >= min_age`` for the age is exact where it matters: ``n_obs <= age``.
    """
    me = cfg.repro.min_energy
    e = b.energy.astype(_I64)
    m_flow = m3[:, IntentCol.FLOW_POS.value].astype(_I64) - m3[:, IntentCol.FLOW_NEG.value].astype(_I64)
    basal = (
        cfg.energy.basal_cost + b.brain_cost.astype(_I64) + cfg.energy.bond_upkeep * b.n_bonds.astype(_I64)
    )
    reach = e + rs.bud_horizon * (m_flow - basal - rs.bud_cost)
    return (e < me) & (reach >= me) & (b.n_obs >= cfg.repro.min_age)


def intent_matches(
    m3: mx.array,
    since_bud: mx.array,
    slot_age: mx.array,
    slot_kin: mx.array,
    cfg: Config,
    rs: RuleSet,
    *,
    bud: BudIn | None = None,
) -> mx.array:
    """(N, 9) bool, one column per predicate in the order of ``Intent`` 1–9 (Budding..Roaming). The columns
    stay in ``Intent`` value order whatever the priority (``first_intent`` permutes).

    ``m3`` is the (N, 10) mean of ``intent`` (``IntentCol``); ``since_bud`` (N,), ``slot_age`` (N, B) and
    ``slot_kin`` (N, B) are the observer's exact (non-accumulated) inputs. Budding is the forecast of
    ``budding_forecast`` where ``rs.bud_horizon > 0`` (then ``bud``, the post-step values, is required: a
    ``ValueError`` without it), else the version-1 rule ``since_bud < rs.bud_window``. Bonding is the
    formation rate ``m_formed >= rs.bond_rate`` where that is positive, else the version-1 rule on the slots'
    ages.
    """
    harvest_max = cfg.energy.harvest_max
    thrust, hurt = m3[:, IntentCol.THRUST.value], m3[:, IntentCol.HURT.value]
    care, harvest = m3[:, IntentCol.CARE.value], m3[:, IntentCol.HARVEST.value]
    if rs.bud_horizon > 0:  # static: the forecast reads the post-step state
        if bud is None:
            raise ValueError(
                f"intent_matches: bud_horizon = {rs.bud_horizon} needs bud (the post-step values, BudIn)"
            )
        budding = budding_forecast(m3, bud, cfg, rs)
    else:
        budding = since_bud < rs.bud_window
    if rs.bond_rate > 0:  # static
        bonding = m3[:, IntentCol.FORMED.value] >= rs.bond_rate
    else:
        lo, hi = rs.bond_age
        bonding = (~slot_kin & (slot_age >= lo) & (slot_age < hi)).any(axis=1)
    cols = [
        budding,  # BUDDING
        m3[:, IntentCol.ATTACK.value] >= rs.hunt,  # HUNTING
        (hurt >= rs.flee_hurt) & (thrust >= rs.flee_thrust),  # FLEEING
        (hurt >= rs.defend_hurt) & (m3[:, IntentCol.DEFEND.value] >= rs.defend),  # DEFENDING
        (care >= 1) & (rs.care[1] * care >= rs.care[0] * harvest_max),  # CARING
        bonding,  # BONDING
        m3[:, IntentCol.EMIT.value] >= rs.signal,  # SIGNALLING
        (harvest >= 1) & (rs.feed[1] * harvest >= rs.feed[0] * harvest_max),  # FEEDING
        rs.roam[1] * thrust >= rs.roam[0],  # ROAMING
    ]
    return mx.stack(cols, axis=1)


def first_intent(match: mx.array, order: tuple[int, ...] = _V1_INTENT_ORDER) -> mx.array:
    """(N,) int32 ``Intent``: the first matching predicate of an ``intent_matches`` matrix in ``order``
    (``Intent`` values, the field ``RuleSet.intent_order``), else RESTING. The default order is the value
    order (version 1)."""
    cols = mx.stack([match[:, i - 1] for i in order], axis=1)  # a static column permutation
    idx = mx.arange(_N_INTENT_RULES, dtype=_I32)
    first = mx.min(mx.where(cols, idx, _N_INTENT_RULES), axis=1)
    values = mx.array(order, dtype=_I32)
    return mx.where(
        first < _N_INTENT_RULES, values[mx.minimum(first, _N_INTENT_RULES - 1)], Intent.RESTING.value
    ).astype(_I32)


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
