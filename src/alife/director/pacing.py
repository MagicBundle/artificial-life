"""Director pacing: budget, cooldowns, active effects and drama as pure folds of the decision log (A6).

Nothing here is stored. ``fold(entries, t)`` replays the log entries with ``step_applied ≤ t`` and
returns what the rule tier needs, so a resumed run (log truncated to the checkpoint) and an
uninterrupted one agree by construction.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Final

from alife.director.actions import MENU, TIMED, lever_keys

BUDGET_WINDOW: Final = 100_000  # the budget is per this many steps, a rolling window over ``step_applied``


@dataclass(frozen=True)
class Effect:
    """A timed effect that has not been reverted: ``restore`` holds the values the revert writes back."""

    action: str
    key: str  # lever: "regen" or "patch:<p>"
    origin: int  # step_applied of the forward entry
    end: int  # origin + duration: the step of the revert
    restore: Mapping[str, Any]


@dataclass(frozen=True)
class Pacing:
    spent: tuple[tuple[int, int], ...]  # (step_applied, cost) of every forward entry
    ends: Mapping[str, int]  # per action: the latest end of any of its effects (step_applied if instant)
    active: Mapping[str, Effect]  # lever key -> the one active effect
    last_step: int | None  # step_applied of the latest forward entry
    last_action: str | None
    last_rule: Mapping[str, int]  # rule name -> step_applied of its latest forward entry


def fold(entries: Iterable[Mapping[str, Any]], t: int) -> Pacing:
    spent: list[tuple[int, int]] = []
    ends: dict[str, int] = {}
    active: dict[str, Effect] = {}
    last_rule: dict[str, int] = {}
    last: tuple[int, str] | None = None
    for e in entries:
        step, action, params = e["step_applied"], e["action"], e["params"]
        if step > t:
            continue
        if e["source"] == "revert":
            active = {k: v for k, v in active.items() if k not in lever_keys(action, params)}
            continue
        end = step + (e.get("duration") or 0)
        spent.append((step, MENU[action].cost))
        ends = {**ends, action: max(ends.get(action, end), end)}
        last = (step, action)
        if e.get("rule"):
            last_rule = {**last_rule, e["rule"]: step}
        if action in TIMED:  # a newer effect on the same lever supersedes the older one
            active = {
                **active,
                **{k: Effect(action, k, step, end, dict(e["prev"])) for k in lever_keys(action, params)},
            }
    return Pacing(tuple(spent), ends, active, last[0] if last else None, last[1] if last else None, last_rule)


def budget_used(p: Pacing, t: int) -> int:
    return sum(cost for step, cost in p.spent if t - BUDGET_WINDOW < step <= t)


def cooldown_ready(p: Pacing, action: str, t: int, cooldown: int) -> bool:
    """True once ``cooldown`` steps have passed since the end of the action's latest effect."""
    end = p.ends.get(action)
    return end is None or t >= end + cooldown


def drama(p: Pacing, t: int, half_life: float) -> float:
    """Recent intervention intensity: the costs of past interventions, halving every ``half_life`` steps."""
    return sum(cost * 0.5 ** ((t - step) / half_life) for step, cost in p.spent)


def steps_since_last(p: Pacing, t: int) -> int | None:
    return None if p.last_step is None else t - p.last_step


def due_reverts(p: Pacing, t: int) -> list[Effect]:
    """The active effects whose end has been reached, in the order their reverts are logged."""
    return sorted((e for e in p.active.values() if e.end <= t), key=lambda e: (e.end, e.key))
