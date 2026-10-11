"""The rule tier: safety rules and the three personas, without an LLM (DIRECTOR.md §5, §6).

``decide(ctx, params)`` is a pure function of the vital signs, the pacing folds of the decision log, the live
environment values and a seed: the same inputs give the same decision. Randomness (a patch index, a
catastrophe centre, the Trickster's sampling) is stateless host numpy seeded by ``(run.seed, step, tag)``
(A13); the values it produces are logged, so replay never draws any.

Rule order at a boundary (the first that acts wins; one that triggers but is blocked leaves a journal note):
extinction rescue, runaway damper, monoculture breaker, stagnation response, then the Trickster's boredom.
``RuleParams`` holds every threshold, window, weight, range and budget; ``RULES_VERSION`` is recorded in the
log header and must change with any of them.

The objective J of DIRECTOR.md §6 is documented but not optimised here: without a forward model the rule
tier reacts to thresholds. Its terms are available as vital signs for the LLM tier (build step 3).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final, NamedTuple

import numpy as np

from alife.director.actions import IMM_PENDING_CAP, MENU, EnvView, lever_keys
from alife.director.pacing import (
    Pacing,
    budget_used,
    cooldown_ready,
    drama,
    steps_since_last,
)
from alife.director.vitals import Vitals

RULES_VERSION: Final = 1

Weights = tuple[tuple[str, float], ...]  # (action, weight) in menu order; ties go to the first


@dataclass(frozen=True)
class PersonaParams:
    """How a persona paces and chooses. Its weights also scale the sustain windows, cooldowns and budget."""

    name: str
    budget: int  # cost units per rolling 100k steps when director.budget_per_100k = 0
    budget_max: int  # the most a persona may be given by director.budget_per_100k
    window_scale: float  # scales how long monoculture / stagnation must last before the persona acts
    cooldown_scale: float  # scales every menu cooldown (the one-catastrophe-per-100k cap is never scaled)
    min_gap: int  # least steps between two discretionary interventions
    drama_max: float  # no discretionary intervention while the decayed cost of recent ones is at least this
    idle_gap: int  # 0 = never; else a "boredom" intervention after this many steps without one
    sampled: bool  # True: seeded weighted sampling and random parameters; False: argmax and fixed ``param_u``
    param_u: float  # deterministic personas: where in each parameter range they pick (0 = low end, 1 = high)
    monoculture: Weights
    stagnation: Weights


GARDENER: Final = PersonaParams(
    name="gardener",
    budget=2,
    budget_max=4,
    window_scale=1.0,
    cooldown_scale=1.5,
    min_gap=50_000,
    drama_max=3.0,
    idle_gap=0,
    sampled=False,
    param_u=0.3,
    monoculture=(("set_seasons", 2.0), ("habitat_loss", 1.0), ("immigration_pulse", 3.0)),
    stagnation=(
        ("set_seasons", 2.0),
        ("fertility_shift", 2.0),
        ("habitat_loss", 0.5),
        ("regen_pulse", 1.0),
        ("immigration_pulse", 3.0),
        ("drift_change", 1.0),
    ),
)
TRICKSTER: Final = PersonaParams(
    name="trickster",
    budget=6,
    budget_max=12,
    window_scale=0.5,
    cooldown_scale=0.5,
    min_gap=20_000,
    drama_max=8.0,
    idle_gap=150_000,
    sampled=True,
    param_u=0.5,
    monoculture=(("set_seasons", 2.0), ("habitat_loss", 2.0), ("immigration_pulse", 1.0)),
    stagnation=(
        ("set_seasons", 2.0),
        ("fertility_shift", 2.0),
        ("habitat_loss", 2.0),
        ("regen_pulse", 2.0),
        ("immigration_pulse", 1.5),
        ("catastrophe", 1.0),
        ("drift_change", 2.0),
    ),
)
ARCHITECT: Final = PersonaParams(
    name="architect",
    budget=4,
    budget_max=8,
    window_scale=1.0,
    cooldown_scale=1.0,
    min_gap=30_000,
    drama_max=5.0,
    idle_gap=0,
    sampled=False,
    param_u=0.5,
    monoculture=(("set_seasons", 2.0), ("habitat_loss", 1.0), ("immigration_pulse", 3.0)),
    stagnation=(
        ("set_seasons", 2.0),
        ("fertility_shift", 2.0),
        ("habitat_loss", 1.0),
        ("regen_pulse", 1.0),
        ("immigration_pulse", 2.0),
        ("drift_change", 1.0),
    ),
)


@dataclass(frozen=True)
class RuleParams:
    # Descriptor normalisation (vitals.descriptor): every component is clipped to [0, 1]
    bonds_per_agent_max: float = 4.0  # mean bonds per agent is divided by this
    predation_max: float = 0.25  # predation ratio (stolen / harvested energy) is divided by this
    sigma_range: tuple[float, float] = (1e-3, 0.5)  # gene medians are log-scaled onto these ranges
    rate_range: tuple[float, float] = (5e-4, 0.5)
    frac_range: tuple[float, float] = (0.05, 0.95)
    # Archive and novelty (archive.Archive)
    novelty_k: int = 5  # neighbours averaged in the novelty score
    novelty_exclude: int = 3  # the most recent archive entries a descriptor is not compared with
    archive_rho: float = 0.025  # a descriptor enters the archive when its novelty exceeds this (RMS distance)
    grid_bins: int = 10  # bins per axis of the coverage grid (emitters x bodies>=4 x predation)
    history_keep: int = 400_000  # steps of boundary history kept (the longest look-back is 200k)
    # Viability and extinction rescue
    risk_k_frac: float = 0.2  # extinction risk when the population is below this fraction of K_est ...
    risk_fall: float = 0.3  # ... or has fallen by more than this ...
    risk_fall_window: int = 50_000  # ... within this many steps
    rescue_override_gap: int = (
        200_000  # a rescue ignores budget and cooldown at most once per this many steps
    )
    rescue_regen: tuple[float, int] = (1.5, 20_000)  # regen_pulse (mult, duration) of a rescue
    rescue_imm_frac: float = 0.1  # rescue immigrants as a fraction of K_est
    # Runaway damper
    damper_pop_frac: float = 0.95  # population above this fraction of N_CAP ...
    damper_body_frac: float = 0.05  # ... or a body larger than this fraction of the population
    damper_mult: float = 0.8  # the damper's regen_pulse multiplier (< 1) ...
    damper_duration: int = 10_000  # ... and duration
    # Monoculture breaker and stagnation response
    mono_share: float = 0.9  # top clade share above this ...
    mono_window: int = 100_000  # ... for this many steps (times persona.window_scale)
    novelty_low: float = 0.02  # novelty below this and no new grid cell ...
    stagnation_window: int = 200_000  # ... for this many steps (times persona.window_scale)
    shift_window: int = 100_000  # look-back of the descriptor shift vital sign
    # Pacing
    drama_half_life: float = 50_000.0  # steps for a past intervention's drama to halve
    catastrophe_gap: int = 100_000  # at most one catastrophe per this many steps, whatever the persona
    # Parameter ranges of the discretionary responses (a persona picks inside them)
    seasons_period: tuple[float, float] = (10_000, 40_000)
    seasons_amp: tuple[float, float] = (0.3, 0.7)
    shift_gain: tuple[float, float] = (0.5, 1.6)
    shift_move_frac: tuple[float, float] = (0.4, 1.0)  # of the L/4 move limit
    habitat_duration: tuple[float, float] = (20_000, 100_000)
    regen_boom: tuple[float, float] = (1.2, 1.6)
    regen_boom_duration: tuple[float, float] = (10_000, 30_000)
    imm_frac: tuple[float, float] = (0.01, 0.04)  # of the current population
    cat_radius_frac: tuple[float, float] = (0.1, 0.25)  # of L
    cat_kill_p: tuple[float, float] = (0.3, 0.7)
    drift_mult: tuple[float, float] = (2.0, 6.0)
    personas: tuple[PersonaParams, ...] = (GARDENER, TRICKSTER, ARCHITECT)

    def persona(self, name: str) -> PersonaParams:
        return next(p for p in self.personas if p.name == name)


RULE_PARAMS: Final = RuleParams()


class Context(NamedTuple):
    """Everything a decision depends on."""

    t: int
    seed: int
    persona: str
    budget: int  # cost units per rolling 100k steps (already resolved from the config and the persona)
    world_size: float
    vitals: Vitals
    pacing: Pacing
    env: EnvView


class Proposal(NamedTuple):
    """A request for the controller to clamp, log and apply."""

    action: str
    raw: Mapping[str, Any]
    duration: int | None
    rule: str
    reason: str


class Decision(NamedTuple):
    proposals: tuple[Proposal, ...]
    notes: tuple[str, ...]  # triggered-but-blocked rules, for the journal


_TAGS: Final = {
    "extinction_rescue": 0,
    "runaway_damper": 1,
    "monoculture_breaker": 2,
    "stagnation_response": 3,
    "boredom": 4,
}


def rng_for(seed: int, t: int, tag: int) -> np.random.Generator:
    """The stateless generator of one decision stream (A13)."""
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, t, tag])))


def effective_budget(configured: int, persona: PersonaParams) -> int:
    """``director.budget_per_100k`` (0 = the persona's default), bounded by the persona's maximum."""
    return min(configured or persona.budget, persona.budget_max)


# --- eligibility ------------------------------------------------------------------------------------------


def _blocked(
    ctx: Context,
    p: RuleParams,
    persona: PersonaParams,
    action: str,
    params: Mapping[str, Any],
    *,
    safety: bool,
    override: bool = False,
) -> str | None:
    """Why ``action`` may not be taken now (None if it may). ``override`` waives budget, cooldown and the
    regen lever (a rescue superseding an older pulse); the catastrophe cap is never waived."""
    spec, t, pc = MENU[action], ctx.t, ctx.pacing
    floor = p.catastrophe_gap if action == "catastrophe" else 0
    if override:
        if floor and not cooldown_ready(pc, action, t, floor):
            return f"{action} is cooling down"
    else:
        if not cooldown_ready(pc, action, t, max(round(spec.cooldown * persona.cooldown_scale), floor)):
            return f"{action} is cooling down"
        if budget_used(pc, t) + spec.cost > ctx.budget:
            return f"budget {budget_used(pc, t)}/{ctx.budget} used"
        gap = steps_since_last(pc, t)
        if not safety and gap is not None and gap < persona.min_gap:
            return f"only {gap} steps since the last intervention"
        if not safety and drama(pc, t, p.drama_half_life) >= persona.drama_max:
            return "drama level too high"
    if action == "immigration_pulse" and IMM_PENDING_CAP - ctx.env.imm_request < spec.bounds["count"][0]:
        return "the immigration queue is full"
    busy = [k for k in lever_keys(action, params) if k in pc.active and not (override and k == "regen")]
    return f"{busy[0]} already has an active effect" if busy else None


def _free_patches(ctx: Context, *, unlocked: bool) -> list[int]:
    """Patches that are alive (gain > 0) and, if ``unlocked``, hold no timed effect."""
    gains = ctx.env.patch_gain
    return [
        i for i in range(len(gains)) if gains[i] > 0 and not (unlocked and f"patch:{i}" in ctx.pacing.active)
    ]


# --- discretionary responses ------------------------------------------------------------------------------


def _lerp(u: float, bounds: tuple[float, float]) -> float:
    return bounds[0] + u * (bounds[1] - bounds[0])


def _r(x: float | None) -> float | None:
    return None if x is None else round(x, 3)


def _request(
    action: str, ctx: Context, p: RuleParams, persona: PersonaParams, rng: np.random.Generator
) -> tuple[dict[str, Any], int | None] | None:
    """The raw parameters of a discretionary ``action`` for this persona, or None if it cannot be done now."""

    def u() -> float:
        return float(rng.random()) if persona.sampled else persona.param_u

    size, v = ctx.world_size, ctx.vitals
    if action == "set_seasons":
        return {"period": _lerp(u(), p.seasons_period), "amp": _lerp(u(), p.seasons_amp)}, None
    if action == "fertility_shift":
        gain_kind = bool(rng.random() < 0.5)  # a gain change takes the patch's lever; a move does not
        pool = _free_patches(ctx, unlocked=gain_kind)
        if not pool:
            return None
        patch = int(rng.choice(pool))
        if gain_kind:
            return {"patch": patch, "gain": _lerp(u(), p.shift_gain)}, None
        dist, angle = _lerp(u(), p.shift_move_frac) * size / 4, float(rng.random()) * 2 * math.pi
        return {"patch": patch, "dx": dist * math.cos(angle), "dy": dist * math.sin(angle)}, None
    if action == "habitat_loss":
        free = _free_patches(ctx, unlocked=True)
        if len(_free_patches(ctx, unlocked=False)) < 2 or not free:
            return None  # never remove the last patch
        return {"patch": int(rng.choice(free))}, round(_lerp(u(), p.habitat_duration))
    if action == "regen_pulse":
        return {"mult": _lerp(u(), p.regen_boom)}, round(_lerp(u(), p.regen_boom_duration))
    if action == "immigration_pulse":
        return {"count": _lerp(u(), p.imm_frac) * (v.n_alive or 0.0)}, None
    if action == "catastrophe":
        x, y = (float(c) for c in rng.uniform(0.0, size, 2))
        radius, kill_p = _lerp(u(), p.cat_radius_frac) * size, _lerp(u(), p.cat_kill_p)
        return {"x": x, "y": y, "radius": radius, "kill_p": kill_p}, None
    if action == "drift_change":
        return {"mult": _lerp(u(), p.drift_mult)}, None
    return None


def _respond(
    ctx: Context, p: RuleParams, persona: PersonaParams, rule: str, weights: Weights, reason: str
) -> tuple[tuple[Proposal, ...], str]:
    """Pick one eligible response from ``weights``: argmax for the deterministic personas, a seeded weighted
    draw (never the last action) for a sampled one. Returns the proposals (empty if all are blocked)."""
    params_rng = rng_for(ctx.seed, ctx.t, 2 * _TAGS[rule] + 1)
    options: list[tuple[str, float, Mapping[str, Any], int | None]] = []
    skipped: list[str] = []
    for action, weight in weights:
        req = _request(action, ctx, p, persona, params_rng) if weight > 0 else None
        if req is None:
            continue
        why = _blocked(ctx, p, persona, action, req[0], safety=False)
        if why is None:
            options.append((action, weight, *req))
        else:
            skipped.append(f"{action}: {why}")
    if persona.sampled and ctx.pacing.last_action is not None:
        options = [o for o in options if o[0] != ctx.pacing.last_action]
    if not options:
        return (), f"{rule} ({reason}): no eligible response" + (
            f" [{'; '.join(skipped)}]" if skipped else ""
        )
    if persona.sampled:
        w = np.asarray([o[1] for o in options], dtype=np.float64)
        pick = options[int(rng_for(ctx.seed, ctx.t, 2 * _TAGS[rule]).choice(len(options), p=w / w.sum()))]
    else:
        pick = max(options, key=lambda o: o[1])  # the first of equal weights
    action, _, raw, duration = pick
    return (Proposal(action, raw, duration, rule, reason),), f"{rule} ({reason}): {action}"


# --- the rules --------------------------------------------------------------------------------------------

Outcome = tuple[tuple[Proposal, ...], str] | None  # None: the rule did not trigger


def _extinction_rescue(ctx: Context, p: RuleParams, persona: PersonaParams) -> Outcome:
    v = ctx.vitals
    if not v.extinction_risk:
        return None
    last = ctx.pacing.last_rule.get("extinction_rescue")
    override = last is None or ctx.t - last >= p.rescue_override_gap
    why = f"n/K_est {_r(v.n_over_k)}, fall {_r(v.fall)}"
    mult, duration = p.rescue_regen
    count = round(p.rescue_imm_frac * v.k_est)
    parts = (("regen_pulse", {"mult": mult}, duration), ("immigration_pulse", {"count": count}, None))
    props, skipped = [], []
    for action, raw, dur in parts:
        blocked = _blocked(ctx, p, persona, action, raw, safety=True, override=override)
        if blocked is None:
            props.append(Proposal(action, raw, dur, "extinction_rescue", why))
        else:
            skipped.append(f"{action}: {blocked}")
    if override:
        props = [pr._replace(reason=f"{pr.reason} (override)") for pr in props]
    return tuple(props), f"extinction_rescue ({why}): blocked [{'; '.join(skipped)}]"


def _runaway_damper(ctx: Context, p: RuleParams, persona: PersonaParams) -> Outcome:
    v = ctx.vitals
    crowded = v.pop_frac_cap is not None and v.pop_frac_cap > p.damper_pop_frac
    bulky = v.largest_body_frac is not None and v.largest_body_frac > p.damper_body_frac
    if not (crowded or bulky):
        return None
    why = f"population {_r(v.pop_frac_cap)} of N_CAP, largest body {_r(v.largest_body_frac)} of it"
    raw = {"mult": p.damper_mult}
    blocked = _blocked(ctx, p, persona, "regen_pulse", raw, safety=True)
    if blocked is not None:
        return (), f"runaway_damper ({why}): blocked [{blocked}]"
    prop = Proposal("regen_pulse", raw, p.damper_duration, "runaway_damper", why)
    return (prop,), f"runaway_damper ({why}): regen_pulse"


def _sustained(since: int | None, last_fire: int | None, t: int, window: float) -> bool:
    """True when a condition that began at ``since`` has lasted ``window`` steps, counting from the rule's
    last firing (so it re-arms after acting)."""
    return since is not None and t - max(since, last_fire if last_fire is not None else since) >= window


def _monoculture(ctx: Context, p: RuleParams, persona: PersonaParams) -> Outcome:
    v = ctx.vitals
    last = ctx.pacing.last_rule.get("monoculture_breaker")
    if not _sustained(v.mono_since, last, ctx.t, p.mono_window * persona.window_scale):
        return None
    why = f"top clade {_r(v.top_clade_share)} since step {v.mono_since}"
    return _respond(ctx, p, persona, "monoculture_breaker", persona.monoculture, why)


def _stagnation(ctx: Context, p: RuleParams, persona: PersonaParams) -> Outcome:
    """Mean novelty below ``novelty_low`` and no new grid cell over the whole window, which the run must have
    covered; it re-arms one window after acting."""
    v, window = ctx.vitals, p.stagnation_window * persona.window_scale
    last = ctx.pacing.last_rule.get("stagnation_response")
    if v.origin is None or ctx.t - v.origin < window or (last is not None and ctx.t - last < window):
        return None
    if v.last_new_cell is not None and v.last_new_cell > ctx.t - window:
        return None
    seen = [n for s, n in v.novelty_trace if s > ctx.t - window and n is not None]
    if not seen or sum(seen) / len(seen) >= p.novelty_low:
        return None
    why = f"mean novelty {sum(seen) / len(seen):.4f}, no new cell for {ctx.t - (v.last_new_cell or 0)} steps"
    return _respond(ctx, p, persona, "stagnation_response", persona.stagnation, why)


def _boredom(ctx: Context, p: RuleParams, persona: PersonaParams) -> Outcome:
    if persona.idle_gap <= 0:
        return None
    anchor = ctx.pacing.last_step if ctx.pacing.last_step is not None else ctx.vitals.origin
    if anchor is None or ctx.t - anchor < persona.idle_gap:
        return None
    return _respond(ctx, p, persona, "boredom", persona.stagnation, f"{ctx.t - anchor} quiet steps")


_RULES: Final = (_extinction_rescue, _runaway_damper, _monoculture, _stagnation, _boredom)


def decide(ctx: Context, p: RuleParams = RULE_PARAMS) -> Decision:
    """The interventions to take at this boundary (usually none), and notes on rules that triggered but were
    blocked (the journal records them; a rule that acts is recorded by its entries)."""
    persona = p.persona(ctx.persona)
    notes: list[str] = []
    for rule in _RULES:
        out = rule(ctx, p, persona)
        if out is None:
            continue
        proposals, note = out
        if proposals:
            return Decision(proposals, tuple(notes))
        notes.append(note)
    return Decision((), tuple(notes))
