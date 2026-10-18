"""Observer state and its per-step update (docs/GAMES.md §4.2–§4.3, DESIGN §28.3). Pure MLX.

``update`` folds one step's observer channel (``step.Acts``) and a few post-step ``State`` fields
(``ObsIn``) into per-row fixed-point accumulators, and ``evaluate`` (every ``rs.every`` steps) turns the
windowed means into labels with ``rules``. Nothing here writes the ``State``, reads its RNG key or its
weights, or feeds anything back into the step. There are no host syncs, no Python branches on array values,
and no scatter or setitem.

The accumulator is ``A += x − (A >> k)`` in int32 with ``x >= 0``: a constant input converges exactly to
``x << k`` (the mean ``A >> k`` is then x, with no dead zone), and a spike decays to under ``2^k`` (mean 0)
without going negative. ``A <= X·2^k`` is invariant for inputs at most X (``rules.RuleSet.check`` enforces
int32 headroom). Over any n steps ``Σ (A_t >> k) = Σ x_t − (A_end − A_start)`` exactly, so the floor adds no
bias to a windowed mean beyond ``|A_end − A_start| / n``.

Identity: a row is *reset* (accumulators, counters and labels zero) when it is dead or its uid differs from
the observed one, which catches deaths, newborns, immigrants and a row reused in the step it died. The
per-slot flows of the channel occur only on the bond that occupied the slot at the start of the step, whose
partner uid is the previous update's ``slot_uid`` (DESIGN §28.3), so they are attributed exactly without the
S12 partner.

Lint (rule 5, ``tests/test_lint.py``): enum constants are written ``X.Y.value``, never ``int(X.Y)``. No output
of ``update`` aliases an input (``uid`` is a real op), and every output derives from an input (the MLX 0.32
``mx.compile`` quirk noted at ``step.consume_env``).
"""

from __future__ import annotations

from typing import Final, NamedTuple

import mlx.core as mx

from alife import layout
from alife.config import Config, dims
from alife.game import rules
from alife.game.rules import (
    EVAL_BIT,
    N_ARCH_BASE,
    N_INTENT_COLS,
    VQ_MAX,
    ArchCol,
    Archetype,
    Intent,
    IntentCol,
    RuleSet,
    n_arch_cols,
)
from alife.state import State, State3
from alife.step import ActFlag, Acts

__all__ = [
    "EVAL_BIT",
    "N_ARCH_BASE",
    "N_INTENT_COLS",
    "OBS_CAP",
    "VQ_MAX",
    "ArchCol",
    "IntentCol",
    "ObsIn",
    "ObsState",
    "care_out",
    "ema",
    "evaluate",
    "init_obs",
    "n_arch_cols",
    "obs_in",
    "quantise_inputs",
    "speed_unit",
    "update",
]

OBS_CAP: Final = 1 << 30  # saturation of counters, and "never" / "older than the observation"

_I32 = mx.int32
_I64 = mx.int64


class ObsIn(NamedTuple):
    """The only ``State`` fields an observer reads (never ``key``, never the weights)."""

    alive: mx.array  # (N,) bool
    uid: mx.array  # (N,) int64, -1 on dead rows
    parent_uid: mx.array  # (N,) int64, -1 for founders and immigrants
    partner: mx.array  # (N,B) int32, -1 empty
    vel: mx.array  # (N,nd) float32
    last_harvest: mx.array  # (N,) int32
    last_inflow: mx.array  # (N,) int32
    last_loss: mx.array  # (N,) int32


def obs_in(s: State | State3) -> ObsIn:
    """The observer's view of a state."""
    return ObsIn(*(getattr(s, f) for f in ObsIn._fields))


class ObsState(NamedTuple):
    uid: mx.array  # (N,) int64: where(alive, uid, -1) at the last update (the observed occupant)
    n_obs: mx.array  # (N,) int32: steps observed since the occupant appeared, saturating at OBS_CAP
    arch: mx.array  # (N, 5+2nd) int32 accumulators (ArchCol)
    arch_slot: mx.array  # (N, B, 2) int32 accumulators: [..., 0] share out, [..., 1] share got, per slot
    intent: mx.array  # (N, 7) int32 accumulators (IntentCol)
    since_bud: mx.array  # (N,) int32: steps since the last bud (OBS_CAP = never)
    slot_uid: mx.array  # (N, B) int64: partner uid per slot at the last update (-1 empty)
    slot_age: mx.array  # (N, B) int32: steps the slot's bond has been seen (OBS_CAP = predates observation)
    slot_kin: mx.array  # (N, B) bool: the slot's partner is this row's parent or child
    label_arch: mx.array  # (N, 2) int32: primary, secondary Archetype (NONE = 0)
    match_arch: mx.array  # (N,) int32: rule bits (bit a−1 = Archetype a) at the last eval
    label_intent: mx.array  # (N,) int32: displayed Intent (hysteresis)
    match_intent: mx.array  # (N,) int32: predicate bits (bit i−1 = Intent i) | EVAL_BIT at the last eval
    cand: mx.array  # (N,) int32: candidate intent
    cand_n: mx.array  # (N,) int32: consecutive evaluations the candidate has held


def ema(acc: mx.array, x: mx.array, k: int) -> mx.array:
    """``A + x − (A >> k)`` in int32, for ``x >= 0``; the mean is ``A >> k``."""
    return acc + x - (acc >> k)


def care_out(slot_kin_prev: mx.array, share_out: mx.array) -> mx.array:
    """(N,) int32: the energy a row sent this step to bond slots whose partner was its parent or child at the
    start of the step (``slot_kin`` of the previous update). Shared with ``analysis/labels.py``."""
    return mx.where(slot_kin_prev, share_out, 0).sum(axis=1)


def speed_unit(cfg: Config) -> float:
    """The velocity unit u: the terminal speed at full thrust ``thrust_max/drag`` (``v_max`` if no thrust)."""
    ph = cfg.physics
    return ph.thrust_max / ph.drag if ph.thrust_max > 0 else ph.v_max


def _slots(oi: ObsIn) -> tuple[mx.array, mx.array]:
    """``(slot_uid (N,B) int64, slot_kin (N,B) bool)`` of the post-step bonds, by gathers (no scatter)."""
    p = mx.maximum(oi.partner, 0)
    su = mx.where(oi.partner >= 0, oi.uid[p], -1)
    me = oi.uid[:, None]
    child = (oi.parent_uid[p] == me) & (me >= 0)  # the partner's parent is this row
    kin = (su >= 0) & (child | (oi.parent_uid[:, None] == su))  # ... or this row's parent is the partner
    return su, kin


def quantise_inputs(
    oi: ObsIn, a: Acts, slot_kin_prev: mx.array, cfg: Config
) -> tuple[mx.array, mx.array, mx.array]:
    """This step's non-negative accumulator inputs ``(x6 (N,5+2nd), xs (N,B,2), x3 (N,7))`` in int32.

    Velocity is in units of u/256, clipped to ±VQ_MAX; each component is split into its positive and negative
    parts, each with its own accumulator, so the mean component is ``m⁺ − m⁻`` and no negative number is ever
    shifted. Flags enter as 256 per set step; ``emit`` is in deposit quanta (Σ_c floor(256·emit_c)).
    """
    scale = 256.0 / speed_unit(cfg)
    vq = mx.clip(mx.round(oi.vel * scale), -VQ_MAX, VQ_MAX).astype(_I32)
    sp = mx.clip(mx.round(mx.sqrt((oi.vel * oi.vel).sum(axis=1)) * scale), 0, VQ_MAX).astype(_I32)
    sent = a.share_out.sum(axis=1)
    x6 = mx.concatenate(
        [
            mx.stack([oi.last_harvest, a.attack_gain, oi.last_inflow, sent, sp], axis=1),
            mx.maximum(vq, 0),
            mx.maximum(-vq, 0),
        ],
        axis=1,
    )
    xs = mx.stack([a.share_out, a.share_got], axis=2)
    f = a.flags.astype(_I32)
    x3 = mx.stack(
        [
            mx.where((f & ActFlag.ATTACKED.value) != 0, 256, 0),
            mx.where(oi.last_loss > 0, 256, 0),
            mx.abs(a.thrust.astype(_I32)),
            a.defend.astype(_I32),
            a.emit.astype(_I32),
            oi.last_harvest,
            care_out(slot_kin_prev, a.share_out),
        ],
        axis=1,
    )
    return x6.astype(_I32), xs.astype(_I32), x3.astype(_I32)


def init_obs(oi: ObsIn, cfg: Config) -> ObsState:
    """The empty observer state for a state: nothing accumulated, nothing labelled, existing bonds of unknown
    age (``OBS_CAP``: never "new")."""
    d = dims(cfg)
    n, b = d.n, layout.B
    su, kin = _slots(oi)
    zero = mx.zeros((n,), dtype=_I32)
    return ObsState(
        uid=mx.where(oi.alive, oi.uid, -1),
        n_obs=zero,
        arch=mx.zeros((n, n_arch_cols(d.nd)), dtype=_I32),
        arch_slot=mx.zeros((n, b, 2), dtype=_I32),
        intent=mx.zeros((n, N_INTENT_COLS), dtype=_I32),
        since_bud=mx.full((n,), OBS_CAP, dtype=_I32),
        slot_uid=su,
        slot_age=mx.where(su >= 0, OBS_CAP, 0).astype(_I32),
        slot_kin=kin,
        label_arch=mx.zeros((n, 2), dtype=_I32),
        match_arch=zero,
        label_intent=zero,
        match_intent=zero,
        cand=zero,
        cand_n=zero,
    )


def update(o: ObsState, oi: ObsIn, a: Acts, cfg: Config, rs: RuleSet) -> ObsState:
    """Fold one step into the observer state. ``oi`` is the post-step state's view and ``a`` the step's
    channel (same rows: a row's acts are those of the occupant it had at the start of the step)."""
    keep = oi.alive & (oi.uid == o.uid)  # the same occupant as last step; False: died, or a new uid
    k1 = keep[:, None]
    su, kin = _slots(oi)
    same = k1 & (su == o.slot_uid) & (su >= 0)  # the bond that existed at the start of this step persists
    x6, xs, x3 = quantise_inputs(oi, a, o.slot_kin, cfg)
    budded = (a.flags.astype(_I32) & ActFlag.BUDDED.value) != 0
    return ObsState(
        uid=mx.where(oi.alive, oi.uid, -1),
        n_obs=mx.where(keep, mx.minimum(o.n_obs + 1, OBS_CAP), 0),
        arch=mx.where(k1, ema(o.arch, x6, rs.arch_k), 0),
        arch_slot=mx.where(same[..., None], ema(o.arch_slot, xs, rs.arch_k), 0),
        intent=mx.where(k1, ema(o.intent, x3, rs.intent_k), 0),
        since_bud=mx.where(keep, mx.where(budded, 0, mx.minimum(o.since_bud + 1, OBS_CAP)), OBS_CAP),
        slot_uid=su,
        slot_age=mx.where(same, mx.minimum(o.slot_age + 1, OBS_CAP), mx.where(su >= 0, 1, 0)),
        slot_kin=kin,
        label_arch=mx.where(k1, o.label_arch, 0),  # a reset row is unlabelled at once
        match_arch=mx.where(keep, o.match_arch, 0),
        label_intent=mx.where(keep, o.label_intent, 0),
        match_intent=mx.where(keep, o.match_intent, 0),
        cand=mx.where(keep, o.cand, 0),
        cand_n=mx.where(keep, o.cand_n, 0),
    )


def evaluate(o: ObsState, cfg: Config, rs: RuleSet) -> ObsState:
    """Evaluate the labels from the windowed means (run at ``state.step % rs.every == 0``, on the step clock).

    A row is labelled once observed ``rs.arch_warm`` (archetypes) or ``rs.intent_warm`` (intents) steps. The
    archetype label is the rule outcome; the displayed intent goes through the hysteresis. ``match_arch`` and
    ``match_intent`` hold the per-rule bits (the Generalist and Resting bits stand for "no rule matched"), and
    ``match_intent`` also carries ``EVAL_BIT`` on every evaluated row (the denominator of the coverage).
    """
    m6 = o.arch >> rs.arch_k
    m6s = o.arch_slot >> rs.arch_k
    m3 = o.intent >> rs.intent_k

    ma = rules.archetype_matches(m6, m6s, cfg, rs)
    primary, secondary = rules.archetype_labels(ma)
    warm_a = o.n_obs >= rs.arch_warm
    generalist_bit = 1 << (Archetype.GENERALIST.value - 1)
    bits_a = rules.pack_bits(ma) | mx.where(ma.any(axis=1), 0, generalist_bit)
    label_arch = mx.where(warm_a[:, None], mx.stack([primary, secondary], axis=1), 0)

    mi = rules.intent_matches(m3, o.since_bud, o.slot_age, o.slot_kin, cfg, rs)
    warm_i = o.n_obs >= rs.intent_warm
    resting_bit = 1 << (Intent.RESTING.value - 1)
    bits_i = rules.pack_bits(mi) | mx.where(mi.any(axis=1), 0, resting_bit) | EVAL_BIT
    raw = mx.where(warm_i, rules.first_intent(mi), Intent.NONE.value)
    label, cand, cand_n = rules.hysteresis(o.label_intent, o.cand, o.cand_n, raw, rs.hold)

    return o._replace(
        label_arch=label_arch.astype(_I32),
        match_arch=mx.where(warm_a, bits_a, 0).astype(_I32),
        label_intent=label,
        match_intent=mx.where(warm_i, bits_i, 0).astype(_I32),
        cand=cand,
        cand_n=cand_n,
    )
