"""The identity check of the player channel's pacing change (G3b spec §6.1.6): what ``fold``,
``due_reverts``, ``budget_used``, ``cooldown_ready`` and ``validate_log`` say about a research
``director.jsonl`` must not change.

``summary`` reads only what every version of ``director.pacing`` has (no scope), so the same function ran on
the code before the change (the golden digests in ``test_player_channel.py``) and runs on the code after it,
with the new ``scope`` argument as well.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from alife.config import Config, config_from_dict
from alife.director import actions, log, pacing

DATA = Path(__file__).parent / "data" / "director_logs"
LOGS = ("eval_s0_C", "eval_s1_C", "eval_s0_RC")


def load(name: str) -> tuple[list[dict[str, Any]], Config]:
    records = [json.loads(line) for line in (DATA / name / "director.jsonl").read_text().splitlines()]
    return records, config_from_dict(json.loads((DATA / name / "config.json").read_text()))


def steps_of(entries: Sequence[Mapping[str, Any]]) -> list[int]:
    """Every step at which the fold can change, and a grid beyond the last one."""
    out = {0}
    for e in entries:
        out |= {e["step_applied"], e["step_applied"] + (e.get("duration") or 0)}
        out |= {e["step_applied"] - 1, e["step_applied"] + 1}
    last = max(out)
    out |= set(range(0, last + 400_001, 25_000))
    return sorted(s for s in out if s >= 0)


def summary(
    entries: Sequence[Mapping[str, Any]],
    fold: Callable[[Sequence[Mapping[str, Any]], int], Any] = pacing.fold,
    due_reverts: Callable[[Any, int], Any] = pacing.due_reverts,
) -> str:
    """A digest of the fold, the budget, the cooldowns and the due reverts at every step of ``steps_of``."""
    rows: list[Any] = []
    for t in steps_of(entries):
        p = fold(entries, t)
        rows.append(
            [
                t,
                [list(x) for x in p.spent],
                sorted(p.ends.items()),
                sorted(
                    [k, e.action, e.key, e.origin, e.end, sorted(dict(e.restore).items())]
                    for k, e in p.active.items()
                ),
                p.last_step,
                p.last_action,
                sorted(p.last_rule.items()),
                pacing.budget_used(p, t),
                [pacing.cooldown_ready(p, a, t, spec.cooldown) for a, spec in sorted(actions.MENU.items())],
                [[e.action, e.key, e.origin, e.end] for e in due_reverts(p, t)],
            ]
        )
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def validates(records: Sequence[Mapping[str, Any]], cfg: Config) -> bool:
    log.validate_log(records, cfg)
    return True
