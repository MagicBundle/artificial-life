"""The decision log ``director.jsonl`` and its header (A8), and the pre-validation of a log for replay (A9).

The log is the single source of truth for replay. Its first line is a header; every later line is one entry
(see ``actions``). A header is appended again whenever the policy changes on resume (persona, goal, budget,
cadence or yoke log), so a reader sees which settings were in force: entries also carry their persona. The
rules version is in the header too, and a live run keeps the one of its latest header (``rules_version_of``):
a resume continues under the rules the run began with, and only a new run (or a fork) takes ``RULES_VERSION``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from alife.config import Config, ConfigError, is_3d
from alife.director.actions import TIMED, lever_keys, validate_entry
from alife.director.rules import RULES_VERSION, SUPPORTED_RULES_VERSIONS
from alife.runio import read_jsonl

HEADER_KIND: Final = "header"


class LogError(ConfigError):
    """A director log that cannot be replayed (exit 2 when it reaches the CLI)."""


def make_header(
    cfg: Config,
    step: int,
    budget: int,
    yoke_sha256: str | None = None,
    params_persona: str | None = None,
    rules_version: int = RULES_VERSION,
) -> dict[str, Any]:
    """The header for a director starting at ``step``; ``budget`` is the effective cost budget per 100k.
    ``yoke_sha256`` (the random persona only) is the fingerprint of the yoke log it was started with, and
    ``params_persona`` the persona whose pacing parameters it inherited from that log. ``rules_version`` is
    the rules the director runs under: the current ones for a new log, the log's own for a resumed run."""
    d = cfg.director
    return {
        "kind": HEADER_KIND,
        "step": step,
        "rules_version": rules_version,
        "persona": d.persona,
        "goal": d.goal,
        "rule_every": d.rule_every,
        "budget_per_100k": budget,
        "seed": cfg.run.seed,
        # The world's dimensions are recorded in 3D only (§27.10): a 2D header keeps its step-2 key set, so a
        # resumed 2D run never gets a new header for it.
        **({"dims": 3} if is_3d(cfg) else {}),
        # The LLM tier's policy is recorded only while it is on, so a log without it keeps its step-2 header.
        **({"llm_every": d.llm_every, "llm_model": cfg.run.llm_model} if d.llm_every else {}),
        # Likewise the yoked persona's provenance: the log it follows (as given) and what that file held.
        **({"yoke_log": d.yoke_log, "yoke_sha256": yoke_sha256} if yoke_sha256 is not None else {}),
        **({"params_persona": params_persona} if params_persona is not None else {}),
    }


def check_rules_version(version: Any, where: str = "header") -> int:
    """``version`` if this code implements it, else ``LogError`` (a log of newer code, or a malformed one)."""
    if isinstance(version, bool) or version not in SUPPORTED_RULES_VERSIONS:
        raise LogError(
            f"{where}: rules_version {version!r} is not one this code implements "
            f"({list(SUPPORTED_RULES_VERSIONS)}); it was written by another version of alife"
        )
    return int(version)


def rules_version_of(records: Sequence[Mapping[str, Any]]) -> int | None:
    """The rules version of the latest header in ``records``, or None if there is no header (a new log). A
    header with no ``rules_version`` is version 1. ``LogError`` for a version this code does not implement."""
    last = next((r for r in reversed(records) if r.get("kind") == HEADER_KIND), None)
    if last is None:
        return None
    return check_rules_version(last.get("rules_version", 1))


def needs_header(records: Sequence[Mapping[str, Any]], header: Mapping[str, Any]) -> bool:
    """True unless the latest header in ``records`` already says the same (the start step aside). A header
    with no ``rules_version`` says 1, as everywhere else, so resuming a log whose header predates the key
    under version 1 appends nothing."""
    last = next((r for r in reversed(records) if r.get("kind") == HEADER_KIND), None)

    def policy(h: Mapping[str, Any]) -> dict[str, Any]:
        return {"rules_version": 1, **{k: v for k, v in h.items() if k != "step"}}

    return last is None or policy(last) != policy(header)


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
    outside the menu bounds, the steps go backwards, a revert has nothing to revert, a header says the log
    was written for another ``world.dims`` than this run's, or for rules this code does not implement. Every
    implemented rules version validates: a replay applies the logged entries and never decides."""
    rule_every = cfg.director.rule_every
    active: set[str] = set()
    last = 0
    for i, rec in enumerate(records):
        if not isinstance(rec, Mapping):
            raise LogError(f"record {i}: not an object")
        if "kind" in rec:
            if rec["kind"] != HEADER_KIND:
                raise LogError(f"record {i}: unknown kind {rec['kind']!r}")
            check_rules_version(rec.get("rules_version", 1), f"record {i}")
            written = rec.get("dims", 2)  # absent = a 2D world (§27.10)
            if written != cfg.world.dims:
                raise LogError(
                    f"record {i}: the log was written for a world with dims = {written!r}, "
                    f"this run has world.dims = {cfg.world.dims}"
                )
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
