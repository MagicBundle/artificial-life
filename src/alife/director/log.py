"""The decision log ``director.jsonl`` and its header (A8), and the pre-validation of a log for replay (A9).

The log is the single source of truth for replay. Its first line is a header; every later line is one entry
(see ``actions``). A header is appended again whenever the policy changes on resume (persona, goal, budget,
cadence, rules version or yoke log), so a reader sees which settings were in force: entries also carry their
persona.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from alife.config import Config, ConfigError
from alife.director.actions import TIMED, lever_keys, validate_entry
from alife.director.rules import RULES_VERSION
from alife.runio import read_jsonl

HEADER_KIND: Final = "header"


class LogError(ConfigError):
    """A director log that cannot be replayed (exit 2 when it reaches the CLI)."""


def make_header(cfg: Config, step: int, budget: int, yoke_sha256: str | None = None) -> dict[str, Any]:
    """The header for a director starting at ``step``; ``budget`` is the effective cost budget per 100k.
    ``yoke_sha256`` (the random persona only) is the fingerprint of the yoke log it was started with."""
    d = cfg.director
    return {
        "kind": HEADER_KIND,
        "step": step,
        "rules_version": RULES_VERSION,
        "persona": d.persona,
        "goal": d.goal,
        "rule_every": d.rule_every,
        "budget_per_100k": budget,
        "seed": cfg.run.seed,
        # The LLM tier's policy is recorded only while it is on, so a log without it keeps its step-2 header.
        **({"llm_every": d.llm_every, "llm_model": cfg.run.llm_model} if d.llm_every else {}),
        # Likewise the yoked persona's provenance: the log it follows (as given) and what that file held.
        **({"yoke_log": d.yoke_log, "yoke_sha256": yoke_sha256} if yoke_sha256 is not None else {}),
    }


def needs_header(records: Sequence[Mapping[str, Any]], header: Mapping[str, Any]) -> bool:
    """True unless the latest header in ``records`` already says the same (the start step aside)."""
    last = next((r for r in reversed(records) if r.get("kind") == HEADER_KIND), None)
    return last is None or {k: v for k, v in last.items() if k != "step"} != {
        k: v for k, v in header.items() if k != "step"
    }


def entries_of(records: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """The intervention entries of a log (everything that is not a header), in order."""
    return [r for r in records if r.get("kind") != HEADER_KIND]


def read_log(path: Path) -> list[dict[str, Any]]:
    """The records of a log file; ``LogError`` if it is missing or a line other than the last is broken."""
    if not Path(path).is_file():
        raise LogError(f"{path}: no director log")
    try:
        return read_jsonl(Path(path))
    except json.JSONDecodeError as e:
        raise LogError(f"{path}: not valid JSON lines ({e})") from e


def validate_log(records: Sequence[Any], cfg: Config) -> None:
    """Pre-validate a whole log before a replay (A9): ``LogError`` / ``ActionError`` (both exit 2) if any
    record is malformed, any action unknown, any ``step_applied`` not a multiple of ``rule_every``, any value
    outside the menu bounds, the steps go backwards, or a revert has nothing to revert."""
    rule_every = cfg.director.rule_every
    active: set[str] = set()
    last = 0
    for i, rec in enumerate(records):
        if not isinstance(rec, Mapping):
            raise LogError(f"record {i}: not an object")
        if "kind" in rec:
            if rec["kind"] != HEADER_KIND:
                raise LogError(f"record {i}: unknown kind {rec['kind']!r}")
            continue
        validate_entry(rec, cfg, rule_every)
        if rec["step_applied"] < last:
            raise LogError(f"record {i}: step {rec['step_applied']} comes after step {last}")
        last = rec["step_applied"]
        keys = set(lever_keys(rec["action"], rec["params"]))
        if rec["source"] == "revert":
            if not keys or not keys <= active:
                raise LogError(
                    f"record {i}: a revert of {rec['action']} at step {last} with no active effect"
                )
            active -= keys
        elif rec["action"] in TIMED:
            active |= keys
