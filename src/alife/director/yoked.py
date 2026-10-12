"""The yoked ``random`` persona, arm R of DIRECTOR.md §9: a control that perturbs as often and as late as
another run's director did, but chooses nothing.

Arm R reads the ``director.jsonl`` of arm B (``director.yoke_log``) and intervenes at exactly the steps where
B made a *discretionary* move, with the same number of moves at each step. Reverts and the safety rules
(``rules.SAFETY_RULES``) are not yoked: R runs them itself, under the Gardener's budget, cooldowns and
thresholds, like any persona. What it draws at a yoked step is a uniformly random eligible menu action with
random parameters (``sample_params``), so the comparison with B isolates the quality of B's choices from the
amount and timing of perturbation.

Everything here is host-only and pure. The generator of a draw is stateless, seeded by
``(run.seed, step, RANDOM_TAG, draw)`` (A13): the values it produces are logged, so replay never draws any,
and a replay never reads the yoke log at all (``Director`` loads it in live mode only).
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final, NamedTuple

import numpy as np

from alife.config import Config
from alife.director import log, rules
from alife.director.actions import MENU, MOVE_MAX_FRAC, radius_bounds

RULE: Final = "random"  # the ``rule`` field of an entry R takes at a yoked step
RANDOM_TAG: Final = 100  # the decision stream of a draw; the Trickster's streams are tags 0 to 9
PARAMS_PERSONA: Final = "gardener"  # whose budget, cooldown scale and thresholds R lives by ("same as B")


class Yoke(NamedTuple):
    """The loaded yoke log: the path as given, the sha256 of its bytes, and the moves to make per step."""

    path: str
    sha256: str
    steps: Mapping[int, int]  # step_applied -> number of discretionary moves arm B made there


class Patches(NamedTuple):
    """The fertility patches a draw may take: ``alive`` (gain > 0) and ``free`` (alive, no timed effect)."""

    alive: tuple[int, ...]
    free: tuple[int, ...]

    @staticmethod
    def every(cfg: Config) -> Patches:
        everything = tuple(range(cfg.fields.n_patches))
        return Patches(everything, everything)


# --- the yoke log -----------------------------------------------------------------------------------------


def yoked_steps(records: Iterable[Any]) -> dict[int, int]:
    """Step -> number of discretionary moves in a director log (R2).

    Discretionary: an entry of ``source`` ``rule`` or ``llm`` that is not a safety rule's. Reverts never
    count, and neither do the safety rules (``rules.SAFETY_RULES``, by the entry's ``rule`` field). A log
    written before entries named their rule has no ``rule`` field to tell a safety move by, so its ``llm``
    entries count, and its ``rule`` entries too (unless they are reverts, which have their own source).
    Records that are not entries (the headers, anything malformed) are skipped here and left to
    ``log.validate_log``.
    """
    steps = (
        r["step_applied"]
        for r in records
        if isinstance(r, Mapping)
        and r.get("kind") != log.HEADER_KIND
        and r.get("source") in ("rule", "llm")
        and r.get("rule") not in rules.SAFETY_RULES
        and isinstance(r.get("step_applied"), int)
        and not isinstance(r.get("step_applied"), bool)
    )
    return dict(sorted(Counter(steps).items()))


def load(cfg: Config) -> Yoke:
    """Read, check and fingerprint ``cfg.director.yoke_log`` (``LogError`` / ``ActionError``, exit 2).

    The file must exist and pass the replay pre-validation against this run's config, and every yoked step
    must be a multiple of this run's ``director.rule_every`` (R can only act at its own boundaries).
    """
    path = Path(cfg.director.yoke_log)
    try:
        records = log.read_log(path)
        data = path.read_bytes()
    except UnicodeDecodeError as e:
        raise log.LogError(f"{path}: not a text log ({e})") from e
    steps = yoked_steps(records)
    every = cfg.director.rule_every
    off = [s for s in steps if s % every]
    if off:
        raise log.LogError(
            f"{path}: yoked step(s) {off[:5]} are not multiples of director.rule_every {every}; "
            "run the yoked arm with the cadence of the run it is yoked to"
        )
    log.validate_log(records, cfg)
    return Yoke(cfg.director.yoke_log, hashlib.sha256(data).hexdigest(), steps)


# --- the draw ---------------------------------------------------------------------------------------------


def rng_for(seed: int, t: int, draw: int) -> np.random.Generator:
    """The stateless generator of the ``draw``-th move at step ``t``."""
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, t, RANDOM_TAG, draw])))


def _probe(action: str, pools: Patches) -> dict[str, Any] | None:
    """Parameters that stand for ``action`` in ``rules.eligibility`` (which only reads the levers and the
    immigration count from them), or None if no choice of patch could make it eligible."""
    if action == "habitat_loss":  # an alive patch with no timed effect, and never the last alive patch
        return {"patch": pools.free[0]} if pools.free and len(pools.alive) >= 2 else None
    if action == "fertility_shift":  # the move form takes no lever, so it is always available
        return {"patch": 0, "dx": 0.0, "dy": 0.0}
    if action == "immigration_pulse":
        return {"count": MENU[action].bounds["count"][0]}
    return {}


def candidates(ctx: rules.Context, p: rules.RuleParams, cfg: Config) -> tuple[list[str], Patches, list[str]]:
    """The menu actions eligible right now (in menu order), the patches a draw may take, and why each of the
    others is not eligible. Eligibility is ``rules.eligibility``: budget, cooldown, one effect per lever,
    the catastrophe cap, a full immigration queue. The persona's minimum gap and drama do not apply.
    ``mutation_floor_nudge`` is not on the menu (``actions.MENU``), so it is never drawn."""
    gains, active = ctx.env.patch_gain, ctx.pacing.active
    alive = tuple(i for i in range(len(gains)) if gains[i] > 0)
    pools = Patches(alive, tuple(i for i in alive if f"patch:{i}" not in active))
    eligible: list[str] = []
    skipped: list[str] = []
    for action in MENU:
        probe = _probe(action, pools)
        why = "no patch it may take" if probe is None else rules.eligibility(ctx, p, action, probe)
        if why is None:
            eligible.append(action)
        else:
            skipped.append(f"{action}: {why}")
    return eligible, pools, skipped


def _uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(rng.uniform(lo, hi))


def _log_uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    return math.exp(_uniform(rng, math.log(lo), math.log(hi)))


def _pick(rng: np.random.Generator, pool: Sequence[int]) -> int:
    if not pool:
        raise ValueError("no patch to choose from")
    return int(pool[int(rng.integers(len(pool)))])


def sample_params(
    action: str, rng: np.random.Generator, cfg: Config, patches: Patches | None = None
) -> tuple[dict[str, Any], int | None]:
    """Random parameters for ``action``: ``(raw, duration)``, as a rule's ``Proposal`` carries them (R4).

    Drawn inside the live menu bounds (``actions.MENU``, ``radius_bounds``, the world size). Ranges that span
    a decade or more (season period, immigrant count, every duration, drift multiplier) are log-uniform,
    everything else uniform. Integers are rounded here; a duration is rounded up to ``rule_every`` and
    everything is brought inside its bounds by ``actions.clamp``, which is what gets logged. ``patches``
    limits the patch indices (default: all of them); a ``fertility_shift`` is a gain change or a move with
    probability 1/2 each (a move when no patch is free for a gain change).
    """
    if action not in MENU:
        raise ValueError(f"unknown action {action!r}")
    spec, size = MENU[action], cfg.world.size
    pools = patches if patches is not None else Patches.every(cfg)
    b = spec.bounds
    if action == "set_seasons":
        return {"period": round(_log_uniform(rng, *b["period"])), "amp": _uniform(rng, *b["amp"])}, None
    if action == "fertility_shift":
        gain_form = bool(rng.random() < 0.5) and bool(pools.free)
        if gain_form:
            return {"patch": _pick(rng, pools.free), "gain": _uniform(rng, *b["gain"])}, None
        patch = _pick(rng, pools.alive or Patches.every(cfg).alive)
        length, angle = _uniform(rng, 0.0, MOVE_MAX_FRAC * size), _uniform(rng, 0.0, 2 * math.pi)
        return {"patch": patch, "dx": length * math.cos(angle), "dy": length * math.sin(angle)}, None
    if spec.duration is not None:
        duration = round(_log_uniform(rng, *spec.duration))
        if action == "habitat_loss":
            return {"patch": _pick(rng, pools.free)}, duration
        return {"mult": _uniform(rng, *b["mult"])}, duration  # regen_pulse
    if action == "immigration_pulse":
        return {"count": round(_log_uniform(rng, *b["count"]))}, None
    if action == "catastrophe":
        x, y = (float(c) for c in rng.uniform(0.0, size, 2))
        return {
            "x": x,
            "y": y,
            "radius": _uniform(rng, *radius_bounds(cfg)),
            "kill_p": _uniform(rng, *b["kill_p"]),
        }, None
    return {"mult": _log_uniform(rng, *b["mult"])}, None  # drift_change
