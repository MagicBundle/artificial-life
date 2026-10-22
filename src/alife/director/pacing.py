"""Director pacing: budget, cooldowns, active effects and drama as pure folds of the decision log (A6).

Nothing here is stored. ``fold(entries, t)`` replays the log entries with ``step_applied ≤ t`` and
returns what the rule tier needs, so a resumed run (log truncated to the checkpoint) and an
uninterrupted one agree by construction.

**Scopes** (G3b spec §6.1.6). The records of the player channel (``actions.is_player``) and the rest are two
pacing scopes: ``fold(entries, t, "rule")`` ignores the player's records in ``spent``, ``ends``, ``last_*``
(and so in drama and cooldowns), ``fold(entries, t, "player")`` ignores everyone else's, and ``scope=None``
counts all of them. ``active`` (the levers a timed effect holds) is shared by every scope, so a lever is
exclusive whoever holds it; each ``Effect`` carries the scope it was made in, and ``due_reverts`` can ask for
one scope's, so the rule tier never reverts a player's effect nor the player tier a rule's. "Timed" is a
property of the entry (it has a duration), and a cost is the entry's own when it has one.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Final

from alife.director.actions import (
    GAME_SPECS,
    MENU,
    PLAYER_BUDGET,
    PLAYER_BUDGET_WINDOW,
    is_player,
    is_timed,
    lever_keys,
)

BUDGET_WINDOW: Final = 100_000  # the budget is per this many steps, a rolling window over ``step_applied``
SCOPES: Final = ("rule", "player")


@dataclass(frozen=True)
class Effect:
    """A timed effect that has not been reverted: ``restore`` holds the values the revert writes back."""

    action: str
    key: str  # lever: "regen" or "patch:<p>"
    origin: int  # step_applied of the forward entry
    end: int  # origin + duration: the step of the revert
    restore: Mapping[str, Any]
    scope: str = "rule"  # the scope of the entry that made it


@dataclass(frozen=True)
class Pacing:
    spent: tuple[tuple[int, int], ...]  # (step_applied, cost) of every forward entry
    ends: Mapping[str, int]  # per action: the latest end of any of its effects (step_applied if instant)
    active: Mapping[str, Effect]  # lever key -> the one active effect
    last_step: int | None  # step_applied of the latest forward entry
    last_action: str | None
    last_rule: Mapping[str, int]  # rule name -> step_applied of its latest forward entry


def fold(entries: Iterable[Mapping[str, Any]], t: int, scope: str | None = None) -> Pacing:
    """The pacing state at ``t``; ``scope`` (``rule`` | ``player``) keeps only that scope's forward entries
    in ``spent``, ``ends``, ``last_*``, ``None`` all of them. ``active`` is the same for every scope."""
    if scope is not None and scope not in SCOPES:
        raise ValueError(f"scope must be one of {SCOPES} or None, got {scope!r}")
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
        mine = "player" if is_player(e) else "rule"
        if scope is None or scope == mine:
            end = step + (e.get("duration") or 0)
            spent.append((step, e["cost"] if "cost" in e else MENU[action].cost))
            ends = {**ends, action: max(ends.get(action, end), end)}
            last = (step, action)
            if e.get("rule"):
                last_rule = {**last_rule, e["rule"]: step}
        if is_timed(e):  # a newer effect on the same lever supersedes the older one
            end = step + e["duration"]
            active = {
                **active,
                **{
                    k: Effect(action, k, step, end, dict(e["prev"]), mine) for k in lever_keys(action, params)
                },
            }
    return Pacing(tuple(spent), ends, active, last[0] if last else None, last[1] if last else None, last_rule)


def budget_used(p: Pacing, t: int, window: int = BUDGET_WINDOW) -> int:
    return sum(cost for step, cost in p.spent if t - window < step <= t)


def cooldown_ready(p: Pacing, action: str, t: int, cooldown: int) -> bool:
    """True once ``cooldown`` steps have passed since the end of the action's latest effect."""
    end = p.ends.get(action)
    return end is None or t >= end + cooldown


def drama(p: Pacing, t: int, half_life: float) -> float:
    """Recent intervention intensity: the costs of past interventions, halving every ``half_life`` steps."""
    return sum(cost * 0.5 ** ((t - step) / half_life) for step, cost in p.spent)


def steps_since_last(p: Pacing, t: int) -> int | None:
    return None if p.last_step is None else t - p.last_step


def due_reverts(p: Pacing, t: int, scope: str | None = None) -> list[Effect]:
    """The active effects whose end has been reached, in the order their reverts are logged; with a
    ``scope`` only that scope's (the rule tier never reverts a player's effect, nor the player a rule's)."""
    return sorted(
        (e for e in p.active.values() if e.end <= t and (scope is None or e.scope == scope)),
        key=lambda e: (e.end, e.key),
    )


def player_blocked(
    mine: Pacing, shared: Pacing, action: str, params: Mapping[str, Any], t: int
) -> str | None:
    """Why the player may not use ``action`` now, or None. ``mine`` is the player-scope fold (cooldown and
    influence are the player's alone: the rule tier's moves neither cool nor spend them), ``shared`` any fold
    (the levers are exclusive whoever holds them: a director's pulse blocks Food, and the rule tier sees a
    player-held lever as busy)."""
    spec = GAME_SPECS[action]
    if not cooldown_ready(mine, action, t, spec.cooldown):
        return f"{action} is cooling down until step {mine.ends[action] + spec.cooldown}"
    used = budget_used(mine, t, PLAYER_BUDGET_WINDOW)
    if used + spec.cost > PLAYER_BUDGET:
        return f"influence: {used} of {PLAYER_BUDGET} units are in use"
    busy = [k for k in lever_keys(action, params) if k in shared.active]
    return f"{busy[0]} already has an active effect" if busy else None
