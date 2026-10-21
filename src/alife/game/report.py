"""The world report, offline (docs/DESIGN.md §31; the N1 step of docs/GAMES.md §9): a chapter report built
from a run's files, with every sentence written by code from a closed set of templates and checked again.

This module has three parts, all pure functions of the run's files (no engine call, no array, no device, no
write into the run directory; ``import alife.config`` loads ``mlx.core`` but nothing is evaluated):

1. The **fact sheet** (``build_sheet``): the exact counts, steps and shares of one chapter ``(a, b]`` of an
   observed run, each with the file field it comes from, the director's interventions as spans and as the
   levers' state (``fold_levers``, ``spans``), the event frames of ``tracking.Event``, and the gates that drop
   what a file cannot support (social records that do not cover the chapter, labels without a validated
   verdict, events at an unstable population, a window of a lever that shared the window with others). A
   sheet is a JSON-able dict; ``facts_digest`` is its identity.
2. The **candidates and the checker** (``candidates``, ``check_string``): every sentence is a template of
   ``game.templates`` filled with sheet fields; each rendered string is re-derived from the sheet and checked
   against the closed vocabulary, the number multiset, the per-fact gates and the lexicon of ``game.lexicon``
   (rules R1 to R7). A string that fails is excluded, never shown.
3. The **assembly** (``assemble``, ``template_report``, ``ticker``): the lead, the body in the fixed order of
   sections (the mandatory sentences, then the candidates by salience, displayed in the section order), the
   headline and the interventions block, as plain text lines.

An optional model may only choose among the candidates (the select-and-order mode: ``select_messages``,
``select_schema``, ``parse_choice``, ``narrate``): its reply is a list of candidate ids, validated entry by
entry and stored for audit, never rendered, quoted or put into a prompt. ``narrate`` is the one function of
this module with a side effect (the HTTP call of the ``Client`` it is given); ``verify_record`` replays a
stored record without ever calling a model.

Lint (rule 2, ``tests/test_lint.py``): no subscript assignment anywhere in this module, dict keys included.
Dicts and lists are built by literals, comprehensions and ``update``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, NamedTuple
from urllib.parse import urlsplit

import numpy as np

from alife.config import Config, DirectorConfig, GameConfig, config_hash, dims, immigration_on
from alife.game import lexicon, templates
from alife.game.rules import ARCHETYPE_DISPLAY, ARCHETYPE_NAMES, INTENT_DISPLAY
from alife.game.social import DEATH_CAUSES, SOCIAL_V1
from alife.game.templates import TEMPLATES, render
from alife.runio import RunDir

__all__ = [
    "CHECKER_VERSION",
    "KNOWN_SOCIAL",
    "LABEL_VERDICTS",
    "MAX_REPLY_CHARS",
    "REPORT_V1",
    "REPORT_VERSION",
    "SELECT_PROMPT_VERSION",
    "LeverState",
    "ReportError",
    "ReportRules",
    "RunView",
    "assemble",
    "body_k",
    "build_sheet",
    "candidates",
    "check_string",
    "compose_sheet",
    "facts_digest",
    "fold_levers",
    "is_chat_url",
    "is_point",
    "judge_attempt",
    "known_social_digest",
    "label_table_digest",
    "lead_of",
    "load_run",
    "narrate",
    "parse_choice",
    "record_versions",
    "select_messages",
    "select_schema",
    "select_seed",
    "settle",
    "spans",
    "template_report",
    "ticker",
    "verify_record",
]

REPORT_VERSION: Final = 1
CHECKER_VERSION: Final = 1


class ReportError(ValueError):
    """A report that cannot be built: ``code`` 2 is a bad parameter or a run that lacks what a report needs,
    ``code`` 1 a check of the run against the director's log that fails."""

    def __init__(self, message: str, code: int = 2) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ReportRules:
    """Every threshold of the sheet and the assembly; provisional values (docs/DESIGN.md §31) that the
    calibration and the owner read confirm. No config key: these are the report's own, pinned by
    ``digest``."""

    snapshot_every: int = 250  # the social cadence: a chapter is cut at a multiple of it
    label_warm: int = 192  # steps after an observer's start before its labels count (RULES_V1 arch_warm)
    catastrophe_margin: int = 250  # labels are withheld from this many steps before a catastrophe
    dip_mandatory: float = 0.8  # the lowest count below this share of the lower endpoint is mandatory
    window_min: int = 1_000  # shortest span of a window fact
    window_sentences: bool = False  # the W sentence of the block: off until the owner read
    labelled_min: int = 100  # labelled agents at a and at b for a label fact
    n_pos_min: int = 100  # positive outcomes in a label's validation
    margin_min: float = 0.10  # AUC minus the permutation null, twice the verdict's 0.05
    event_pop_gate: tuple[int, int] = (1, 20)  # an event is suspect if the population moved by 1/20 of a base
    kin_churn_factor: int = 2  # kin-group events below this multiple of kin_enter are churn
    colony_event_min: int = 32  # colony events below this size are churn
    relation_min_attacks: int = 10
    headline_change: float = 0.1  # |change| from which the headline is the population's
    headline_event_share: int = 50  # an event headlines if size * this >= the population at b
    quiet_change: float = 0.05
    k_body: int = 6
    k_quiet: int = 3
    block_cap: int = 24
    quota_events: int = 6
    quota_groups: int = 6
    quota_labels: int = 6
    quota_relations: int = 3
    quota_windows: int = 5
    max_mandatory_events: int = 3
    exact_tol: float = 1e-6  # a count recovered from a share must be this close to an integer
    max_chars: int = 240
    max_headline_chars: int = 100

    def digest(self) -> str:
        return hashlib.sha256(_canon(dataclasses.asdict(self)).encode("utf-8")).hexdigest()


REPORT_V1: Final = ReportRules()

# --- the label table (docs/DESIGN.md §31; GAMES §9) ---------------------------------------------------------

# (intent_version, rules_digest, dims, world_key) -> label -> (verdict, n_pos, margin, p, n_perm), from the
# two
# GPU studies of the version-1 rules (``alife.analysis.labels`` on runs/64k_s0 at 1,490,000 and
# runs/64k3d_s0 at
# 450,000, 6,000 steps each; the files' sha256 start 64303e05 and 527c6c1d). ``margin`` is the AUC minus the
# permutation null's mean, ``p`` the permutation p-value. The world key is the files' ``config_hash``
# (a config without ``run``; the director and game sections at their defaults). A new rule set or world adds
# rows by the change that validates it, never by this module.
_V1_DIGEST: Final = "156ad4963b84ffe9636e9ba4fdd7592632c669bf6e490687d8eb2b38b9aec009"
_W2D: Final = "69b4e21ad6b6016b17c11d6a1d9b3371c03b4b9f590fdb4a04752515c13a01d8"
_W3D: Final = "df54434fe947dea8f36e5e23223f9676df1ee784959806c31eafb235012ee242"
LABEL_VERDICTS: Final[
    Mapping[tuple[int, str, int, str], Mapping[str, tuple[str, int, float, float, int]]]
] = {
    (1, _V1_DIGEST, 2, _W2D): {
        "gatherer": ("fail", 506440, -0.1046, 1.0, 200),
        "hunter": ("pass", 2915, 0.2541, 0.004975, 200),
        "provider": ("pass", 2932, 0.4141, 0.004975, 200),
        "trader": ("pass", 438, 0.4692, 0.004975, 200),
        "nomad": ("pass", 865, 0.4499, 0.004975, 200),
        "settler": ("pass", 166662, 0.244, 0.004975, 200),
        "budding": ("fail", 17117, -0.0048, 1.0, 200),
        "hunting": ("pass", 931, 0.1807, 0.004975, 200),
        "fleeing": ("pass", 79, 0.2477, 0.004975, 200),
        "defending": ("fail", 486694, 0.413, 0.004975, 200),
        "caring": ("pass", 315, 0.4447, 0.004975, 200),
        "bonding": ("pass", 125, 0.0576, 0.00995, 200),
        "signalling": ("pass", 61488, 0.4792, 0.004975, 200),
        "feeding": ("fail", 16181, -0.0733, 1.0, 200),
        "roaming": ("pass", 7255, 0.1882, 0.004975, 200),
        "resting": ("pass", 19635, 0.2046, 0.004975, 200),
    },
    (1, _V1_DIGEST, 3, _W3D): {
        "gatherer": ("fail", 632063, -0.2939, 1.0, 200),
        "hunter": ("pass", 22159, 0.3011, 0.004975, 200),
        "provider": ("pass", 964, 0.4307, 0.004975, 200),
        "trader": ("pass", 32, 0.4817, 0.004975, 200),
        "nomad": ("pass", 8665, 0.2774, 0.004975, 200),
        "settler": ("pass", 34575, 0.4027, 0.004975, 200),
        "budding": ("fail", 32981, -0.0054, 1.0, 200),
        "hunting": ("pass", 32433, 0.2238, 0.004975, 200),
        "fleeing": ("pass", 712, 0.2644, 0.004975, 200),
        "defending": ("fail", 606816, 0.249, 0.004975, 200),
        "caring": ("insufficient", 8, 0.1156, 0.004975, 200),
        "bonding": ("fail", 769, -0.1055, 1.0, 200),
        "signalling": ("fail", 3181, -0.1291, 1.0, 200),
        "feeding": ("fail", 60696, -0.2034, 1.0, 200),
        "roaming": ("fail", 24893, 0.0433, 0.004975, 200),
        "resting": ("pass", 40596, 0.1324, 0.004975, 200),
    },
}

# social_digest -> the snapshot cadence and the slot count the report's grid and tables assume.
KNOWN_SOCIAL: Final[Mapping[str, tuple[int, int]]] = {
    SOCIAL_V1.digest(): (SOCIAL_V1.snapshot_every, SOCIAL_V1.slots)
}


def label_table_digest() -> str:
    return hashlib.sha256(
        _canon(
            [[list(k), {lb: list(v) for lb, v in t.items()}] for k, t in sorted(LABEL_VERDICTS.items())]
        ).encode()
    ).hexdigest()


def known_social_digest() -> str:
    return hashlib.sha256(_canon({k: list(v) for k, v in sorted(KNOWN_SOCIAL.items())}).encode()).hexdigest()


def _canon(x: Any) -> str:
    return json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def facts_digest(sheet: Mapping[str, Any]) -> str:
    """sha256 of the canonical JSON of a sheet (sorted keys, compact, ``allow_nan=False``: a non-finite number
    in a sheet is a bug that raises)."""
    return hashlib.sha256(_canon(sheet).encode("utf-8")).hexdigest()


# --- the run's files ---------------------------------------------------------------------------------------


class RunView(NamedTuple):
    """What a sheet reads of a run: the config, ``meta.json``, the metrics rows, the social records, the
    director's entries (log order, no header) and the observer sidecars' meta (step -> meta dict or a
    path)."""

    cfg: Config
    meta: Mapping[str, Any]
    rows: tuple[Mapping[str, Any], ...]
    records: tuple[Mapping[str, Any], ...]
    entries: tuple[Mapping[str, Any], ...]
    sidecars: Mapping[int, Any]

    @property
    def start_step(self) -> int:
        return int(self.meta.get("start_step") or 0)

    def row(self, step: int) -> Mapping[str, Any] | None:
        return next((r for r in self.rows if r["step"] == step), None)

    def sidecar_meta(self, at_most: int) -> Mapping[str, Any] | None:
        """The ``meta`` of the latest sidecar with step <= ``at_most`` (None if there is none or it cannot be
        read): only that one member of the file is read, never an array."""
        steps = [s for s in self.sidecars if s <= at_most]
        if not steps:
            return None
        v = self.sidecars[max(steps)]
        if isinstance(v, Mapping):
            return v
        try:
            z = np.load(Path(v), allow_pickle=False)
            with z:
                meta = json.loads(str(z["meta"]))
        except (OSError, ValueError, KeyError, EOFError):
            return None
        return meta if isinstance(meta, dict) else None


def load_run(path: str | Path) -> RunView:
    """The files of the run at ``path``, read-only. ``FileNotFoundError`` if it is not a run directory."""
    from alife.director.log import entries_of, read_log  # host-side; imported here to keep the import light

    rd = RunDir.open(Path(path))
    log = Path(path) / "director.jsonl"
    entries = tuple(entries_of(read_log(log))) if log.is_file() else ()
    return RunView(
        rd.config(),
        rd.meta(),
        tuple(rd.metrics_rows()),
        tuple(rd.social_records()),
        entries,
        {step: p for step, p in rd.observer_files()},
    )


# --- levers ------------------------------------------------------------------------------------------------

TIMED: Final = ("regen_pulse", "habitat_loss")
POINTS: Final = ("immigration_pulse", "catastrophe")


def is_point(action: str, params: Mapping[str, Any]) -> bool:
    """Whether an entry is a point event: it happens at one step and has no span. immigration_pulse and
    catastrophe write no lever; a patch move writes the position lever but is reported as a point too, since
    the patch drifts every step and a single write says nothing about where it is later."""
    return action in POINTS or (action == "fertility_shift" and "gain" not in params)


LEVER_ORDER: Final = ("regen", "seasons", "drift")


class LeverState(NamedTuple):
    value: Mapping[str, Any]  # the lever's values (``mult``, ``gain``, ``period``/``amp``, ``x``/``y``)
    step: int  # ``step_applied`` of the writing entry
    entry: int  # its index in the log


def lever_of(action: str, params: Mapping[str, Any]) -> str | None:
    """The lever an entry writes (``actions.lever_keys`` extended to the whole menu), None for a point."""
    if action == "regen_pulse":
        return "regen"
    if action == "habitat_loss" or (action == "fertility_shift" and "gain" in params):
        return f"patch:{int(params['patch'])}:gain"
    if action == "fertility_shift":
        return f"patch:{int(params['patch'])}:pos"
    if action == "set_seasons":
        return "seasons"
    if action == "drift_change":
        return "drift"
    return None


def _lever_value(lever: str, params: Mapping[str, Any]) -> dict[str, Any]:
    if lever in ("regen", "drift"):
        return {"mult": params["mult"]}
    if lever == "seasons":
        return {"period": params["period"], "amp": params["amp"]}
    if lever.endswith(":gain"):
        return {"gain": params["gain"]}
    return {"x": params["x"], "y": params["y"]}


def _lever_patch(lever: str) -> int | None:
    return int(lever.split(":")[1]) if lever.startswith("patch:") else None


def _lever_key(lever: str) -> tuple[int, int, int]:
    """The order of the levers in a block: regen, seasons, drift, then the patches (gain, then position)."""
    if lever in LEVER_ORDER:
        return (LEVER_ORDER.index(lever), 0, 0)
    p = _lever_patch(lever)
    return (len(LEVER_ORDER), p if p is not None else 0, 0 if lever.endswith(":gain") else 1)


def _in_fold_order(entries: Sequence[Mapping[str, Any]], start_step: int) -> list[tuple[int, int]]:
    """``(step_applied, log index)`` of the entries a replay applies (step_applied > start_step), in order."""
    return sorted(
        (int(e["step_applied"]), i) for i, e in enumerate(entries) if e["step_applied"] > start_step
    )


def fold_levers(
    entries: Sequence[Mapping[str, Any]], t: int, start_step: int, cfg: Config | None = None
) -> dict[str, LeverState]:
    """The engine's lever state as of the row at step ``t``: the one source of lever state for the env check
    and the block. Every entry with ``start_step < step_applied < t``, forward or revert (whatever entry the
    revert ends), in (``step_applied``, log index) order; each writes its lever with its ``params``. A replay
    fork applies every entry after ``start_step``, reverts included (``director/__init__.py``), so a revert
    of an entry before ``start_step`` is part of the fold. Points write no lever. ``cfg`` is accepted for
    symmetry with ``lever_default`` and unused."""
    out: dict[str, LeverState] = {}
    for s, i in _in_fold_order(entries, start_step):
        if s >= t:
            break
        e = entries[i]
        lv = lever_of(e["action"], e["params"])
        if lv is not None:
            out.update({lv: LeverState(_lever_value(lv, e["params"]), s, i)})
    return out


def lever_default(lever: str, cfg: Config) -> dict[str, Any] | None:
    """The values of a lever at the start of a run, None for a patch position (any move is "away")."""
    if lever in ("regen", "drift"):
        return {"mult": 1.0}
    if lever == "seasons":
        return {"period": cfg.fields.season_period, "amp": cfg.fields.season_amplitude}
    if lever.endswith(":gain"):
        return {"gain": 1.0}
    return None


def _f32(x: float) -> np.float32:
    return np.float32(x)


def away_from_default(lever: str, value: Mapping[str, Any], cfg: Config) -> bool:
    d = lever_default(lever, cfg)
    if d is None:
        return True
    return any(
        (int(value[k]) != int(v)) if k == "period" else (_f32(value[k]) != _f32(v)) for k, v in d.items()
    )


def _row_step_set(rd: RunView) -> frozenset[int]:
    return frozenset(r["step"] for r in rd.rows)


# --- the env check, spans and the block --------------------------------------------------------------------


def _implied_env(
    entries: Sequence[Mapping[str, Any]], t: int, start_step: int, cfg: Config
) -> dict[str, Any]:
    """The ``env`` levers of the row at ``t`` that the fold implies (``metrics._env_summary``)."""
    fold = fold_levers(entries, t, start_step, cfg)
    pp = dims(cfg).pp
    gains = [1.0] * pp
    for lv, st in fold.items():
        p = _lever_patch(lv)
        if lv.endswith(":gain") and p is not None and p < pp:
            gains = [st.value["gain"] if j == p else g for j, g in enumerate(gains)]
    g32 = np.array(gains, dtype=np.float32)
    dflt = lever_default("seasons", cfg) or {}
    seasons = fold.get("seasons")
    return {
        "regen_mult": fold["regen"].value["mult"] if "regen" in fold else 1.0,
        "season_amp": seasons.value["amp"] if seasons else dflt["amp"],
        "season_period": seasons.value["period"] if seasons else dflt["period"],
        "drift_mult": fold["drift"].value["mult"] if "drift" in fold else 1.0,
        "patch_gain_min": float(g32.min()),
        "patch_gain_max": float(g32.max()),
        "patches_off": int((g32 == 0).sum()),
    }


def check_env(rd: RunView, a: int, b: int) -> None:
    """The env check (MUST): in a run with a director every row r in [a, b] carries an ``env`` block equal, as
    float32, to the fold of the entries before r over the defaults; else ``ReportError`` (code 1)."""
    rows = [r for r in rd.rows if a <= r["step"] <= b]
    has_env = [r for r in rows if "env" in r]
    if not has_env:
        if rd.cfg.director.enabled and any(e["step_applied"] > rd.start_step for e in rd.entries):
            raise ReportError("the rows of a director run carry no env block: the log cannot be checked", 1)
        return
    for r in has_env:
        want = _implied_env(rd.entries, r["step"], rd.start_step, rd.cfg)
        for k, v in want.items():
            got = r["env"].get(k)
            same = got is not None and (
                int(got) == int(v) if k in ("season_period", "patches_off") else _f32(got) == _f32(v)
            )
            if not same:
                raise ReportError(
                    f"env.{k} of the row at step {r['step']} is {got!r}, the director log implies {v!r}", 1
                )


def _iv_fact(
    i: int,
    e: Mapping[str, Any],
    end: int | None,
    end_by: str | None,
    restored: Mapping[str, Any] | None,
    b: int,
) -> dict[str, Any]:
    return {
        "kind": "intervention",
        "entry": i,
        "action": e["action"],
        "source": e["source"],
        "rule": e.get("rule"),
        "src": templates.source_words(e["source"], e.get("rule")),
        "step": int(e["step_applied"]),
        "params": json.loads(json.dumps(e["params"])),
        "duration": e.get("duration"),
        "end": end,
        "end_by": end_by,
        "restored": None if restored is None else dict(restored),
        "open_to": b if end is None and not is_point(e["action"], e["params"]) else None,
    }


def spans(
    entries: Sequence[Mapping[str, Any]], a: int, b: int, start_step: int, cfg: Config | None = None
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """The intervention facts of the chapter ``(a, b]`` and what was left out.

    Forward entries (source ``rule`` or ``llm``) with ``step_applied`` in [a, b) (in force inside the
    chapter), then forward entries before ``a`` whose span contains the step ``a + 1`` (open, or ending after
    ``a``: an entry whose span ends at ``a`` has nothing in the chapter, and the lever it restored, if not to
    its default, is a standing line). Points (immigration_pulse, catastrophe and a patch move, see
    ``is_point``) are listed only inside the chapter and have no span. An entry at ``s`` is in force for
    steps s+1 to e, with e the ``step_applied`` of the next
    entry on its lever in the fold order (forward, or a revert of any entry) when that is at most ``b``, else
    open at ``b``; the director acts after its step's row, so the rows ``s`` and ``e`` bound the span
    exactly. ``end_by`` is ``elapsed`` (a revert whose ``reverts`` is ``s``; ``restored`` its params),
    ``replaced`` (any other later entry on the lever) or None. Entries with ``step_applied`` <=
    ``start_step`` are ignored as forward entries (a fork restarts the levers inert).

    The facts come in the order the block cap of 24 keeps them: the chapter's entries by (step, log index),
    then the in-force ones latest first; the caller cuts and re-sorts. The second result counts the entries
    ignored."""
    order = _in_fold_order(entries, start_step)
    ignored = sum(1 for e in entries if e["source"] != "revert" and e["step_applied"] <= min(start_step, b))
    made: list[dict[str, Any]] = []
    for pos, (s, i) in enumerate(order):
        e = entries[i]
        if e["source"] == "revert" or s >= b:
            continue
        lv = lever_of(e["action"], e["params"])
        point = is_point(e["action"], e["params"])
        end = end_by = restored = None
        if lv is not None and not point:
            for s2, j in order[pos + 1 :]:
                if s2 > b:
                    break
                e2 = entries[j]
                if lever_of(e2["action"], e2["params"]) == lv:
                    end = s2
                    if e2["source"] == "revert" and e2.get("reverts") == s:
                        end_by, restored = "elapsed", e2["params"]
                    else:
                        end_by = "replaced"
                    break
        listed = a <= s < b or (lv is not None and not point and s < a and (end is None or end > a))
        if listed:
            made.append(_iv_fact(i, e, end, end_by, restored, b))
    chapter = sorted((f for f in made if f["step"] >= a), key=lambda f: (f["step"], f["entry"]))
    inforce = sorted((f for f in made if f["step"] < a), key=lambda f: (-f["step"], -f["entry"]))
    return chapter + inforce, {"entries_before_start": ignored}


def _standing(lever: str, step: int, entry: int, value: Mapping[str, Any]) -> dict[str, Any]:
    p = _lever_patch(lever)
    return {"kind": "standing", "lever": lever, "step": step, "entry": entry, **value} | (
        {} if p is None else {"patch": p}
    )


def standing_facts(
    entries: Sequence[Mapping[str, Any]],
    a: int,
    start_step: int,
    cfg: Config,
    b: int | None = None,
    listed: Sequence[Mapping[str, Any]] = (),
) -> list[dict[str, Any]]:
    """The levers that a revert left away from their default: "from step {s}, ...", in lever order.

    The levers as of the row ``a`` that the latest write before it (a revert) left away from the default; a
    lever held by a forward entry is listed by that entry's own intervention fact. With ``b``, a lever that
    any entry writes inside the chapter ``[a, b)`` has no line of the first kind (the chapter overwrites it:
    "has been" would be false at ``b``), and the levers that the last write inside the chapter, a revert,
    leaves away from the default get a line when no listed intervention ends at that step (the forward entry
    was before ``start_step``: a replay fork applies its revert all the same; or it ended at ``a``). A patch
    position is never a standing lever (a move is a point event)."""
    inside: dict[str, tuple[int, int]] = {}
    if b is not None:
        for s, i in _in_fold_order(entries, start_step):
            lv = lever_of(entries[i]["action"], entries[i]["params"])
            if a <= s < b and lv is not None:
                inside.update({lv: (s, i)})
    out = [
        _standing(lv, st.step, st.entry, st.value)
        for lv, st in fold_levers(entries, a, start_step, cfg).items()
        if entries[st.entry]["source"] == "revert"
        and not lv.endswith(":pos")
        and lv not in inside
        and away_from_default(lv, st.value, cfg)
    ]
    if b is not None:
        ended = {(f["end"], lever_of(f["action"], f["params"])) for f in listed if f["end"] is not None}
        for lv, (s, i) in inside.items():
            e = entries[i]
            value = _lever_value(lv, e["params"])
            if (
                e["source"] == "revert"
                and not lv.endswith(":pos")
                and (s, lv) not in ended
                and away_from_default(lv, value, cfg)
            ):
                out.append(_standing(lv, s, i, value))
    return sorted(out, key=lambda f: (_lever_key(f["lever"]), f["step"]))


# --- the sheet ----------------------------------------------------------------------------------------------

FACT_KINDS: Final = (
    "chapter",
    "population",
    "deaths",
    "births",
    "lineages",
    "top_lineage",
    "intervention",
    "standing",
    "window",
    "label",
    "group",
    "event",
    "houses",
    "relation",
    "colonies",
    "run_end",
)
_KIN_FAMILY_RANK: Final = {"kin_group": 0, "colony": 1}


def _rows_of(table: Mapping[str, Sequence[Any]] | None) -> list[dict[str, Any]]:
    """The rows of a column-oriented record table (``society._table``)."""
    if not table:
        return []
    cols = list(table)
    n = len(table[cols[0]])
    return [{c: table[c][j] for c in cols} for j in range(n)]


def alive_of(rec: Mapping[str, Any]) -> int:
    """The population at a snapshot from its record: the agents outside every slotted kin group plus the
    slotted groups' rows (``society.py``: ``other`` is ``n_alive`` minus the slotted rows)."""
    return int(rec["summary"]["kin_groups"]["other"]) + sum(int(x) for x in rec["kin_groups"]["size"])


def _snapshot_steps(a: int, b: int, every: int) -> list[int]:
    return list(range((a // every + 1) * every, b + 1, every))


class _Ents:
    """The entities a sheet names: kin groups (by id, with the step bound of the record their name may come
    from) and houses (by id, with the name of the houses table)."""

    def __init__(self) -> None:
        self.kin: dict[int, int] = {}  # id -> the latest record step that cites it
        self.houses: dict[int, str] = {}

    def add_kin(self, kid: int, limit: int) -> None:
        self.kin.update({int(kid): max(limit, self.kin.get(int(kid), limit))})

    def add_house(self, hid: int, name: str) -> None:
        self.houses.update({int(hid): name})


def _kin_name(records: Sequence[Mapping[str, Any]], kid: int, limit: int) -> tuple[str | None, int | None]:
    """The name of kin group ``kid`` from the latest record at or before ``limit`` that lists it."""
    for rec in reversed(records):
        if rec["step"] > limit:
            continue
        ids = rec["kin_groups"]["id"]
        if kid in ids:
            return rec["kin_groups"]["name"][ids.index(kid)], int(rec["step"])
    return None, None


def _exact_count(share: float, denominator: int, tol: float) -> int | None:
    """``round(share * denominator)`` if the product is within ``tol`` of an integer, else None."""
    x = share * denominator
    n = round(x)
    return n if abs(x - n) <= tol else None


def _social_head(row: Mapping[str, Any]) -> Mapping[str, Any]:
    return row["observers"]["social"]


def _social_gate(rd: RunView, a: int, b: int, rules: ReportRules, by: Mapping[int, Mapping[str, Any]]):
    """``(records of (a, b], reason)``: the records the social facts need, or why there are none. Every
    snapshot step of (a, b] needs a record, all of one origin <= a, with a known ``social_digest`` and an
    alive identity that equals ``observers.alive`` where the step is a row's."""
    want = _snapshot_steps(a, b, rules.snapshot_every)
    recs = [r for r in rd.records if a < r["step"] <= b]
    have = [r["step"] for r in recs]
    missing = [s for s in want if s not in have]
    if missing or len(have) != len(set(have)):
        return [], f"no social record at step {missing[0]}" if missing else "a social record is repeated"
    origins = {r.get("origin") for r in recs}
    if not all(isinstance(o, int) and not isinstance(o, bool) for o in origins):
        return [], "a social record has no origin"
    if len(origins) != 1 or max(origins) > a:
        return [], f"observer restarted at step {max(origins)}"
    for r in recs:
        known = KNOWN_SOCIAL.get(r.get("social_digest"))
        if known is None or known[0] != rules.snapshot_every:
            return [], "unknown social digest"
    for r in recs:
        row = by.get(r["step"])
        if row is not None and alive_of(r) != row["observers"]["alive"]:
            return [], f"the alive count of the record at step {r['step']} differs from the metrics row"
    return recs, None


def _event_facts(
    rd: RunView, recs: Sequence[Mapping[str, Any]], rules: ReportRules, excl: dict[str, Any]
) -> list[dict[str, Any]]:
    """The event facts of the records (frames of ``tracking.Event``), after the suspect and churn gates."""
    out: list[dict[str, Any]] = []
    prev_alive = {r["step"]: alive_of(r) for r in rd.records if r.get("origin") == recs[0].get("origin")}
    for rec in recs:
        evs = _rows_of(rec["events"])
        th = rec["thresholds"]
        n_ref = th["n_ref"]
        num, den = rules.event_pop_gate
        now = alive_of(rec)
        before = prev_alive.get(rec["window_start"], n_ref)
        suspect = (
            bool(th["rescaled"])
            or abs(now - n_ref) * den > n_ref * num
            or abs(now - before) * den > before * num
        )
        pairs = {(e["family"], e["kind"], e["id"], e["other"]) for e in evs}
        for e in evs:
            fam, kind = e["family"], e["kind"]
            if fam == "kin_group":
                if suspect:
                    excl.update({"events_suspect": excl["events_suspect"] + 1})
                    continue
                if e["size"] < rules.kin_churn_factor * th["kin_enter"]:
                    excl.update({"events_churn": excl["events_churn"] + 1})
                    continue
            elif e["size"] < rules.colony_event_min:
                excl.update({"events_churn": excl["events_churn"] + 1})
                continue
            partner = {"born": "dissolved", "dissolved": "born"}.get(kind)
            reid = (
                partner is not None
                and e["other"] >= 0
                and ((fam, partner, e["other"], e["id"]) in pairs or 2 * e["overlap"] > e["size"])
            )
            out.append(
                {
                    "kind": "event",
                    "family": fam,
                    "event": kind,
                    "system": int(e["id"]),
                    "other": int(e["other"]),
                    "size": int(e["size"]),
                    "overlap": int(e["overlap"]),
                    "s0": int(rec["window_start"]),
                    "s1": int(rec["step"]),
                    "reid": bool(reid),
                }
            )
    out.sort(key=lambda f: (-f["size"], f["s1"], _KIN_FAMILY_RANK[f["family"]], f["system"]))
    return out


def _group_facts(
    recs: Sequence[Mapping[str, Any]], rb: Mapping[str, Any], alive_b: int
) -> list[dict[str, Any]]:
    table = _rows_of(rb["kin_groups"])
    table.sort(key=lambda g: (-g["size"], g["id"]))
    complete = rb["summary"]["kin_groups"]["tracked"] == len(table)
    cause = DEATH_CAUSES.index("killed")
    totals: dict[int, list[int]] = {}
    listed_all: dict[int, bool] = {}
    for g in table:
        gid = g["id"]
        rows = [next((x for x in _rows_of(rec["kin_groups"]) if x["id"] == gid), None) for rec in recs]
        ok = all(x is not None and x["births"] is not None and x["deaths"] is not None for x in rows)
        listed_all.update({gid: ok})
        if ok:
            totals.update(
                {
                    gid: [
                        sum(x["births"] for x in rows),
                        sum(sum(x["deaths"]) for x in rows),
                        sum(x["deaths"][cause] for x in rows),
                    ]
                }
            )
    out = []
    carried = complete
    for j, g in enumerate(table):
        nxt = table[j + 1]["size"] if j + 1 < len(table) else None
        f: dict[str, Any] = {
            "kind": "group",
            "entity": int(g["id"]),
            "size": int(g["size"]),
            "of": int(alive_b),
            "share": g["size"] / alive_b if alive_b else 0.0,
        }
        carried = carried and (nxt is None or g["size"] > nxt)
        if carried:
            f.update({"rank": j + 1})
        if listed_all[g["id"]]:
            born, died, killed = totals[g["id"]]
            f.update({"born": born, "died": died, "died_killed": killed})
        out.append(f)
    return out


def _house_facts(
    recs: Sequence[Mapping[str, Any]], ra: Mapping[str, Any], rb: Mapping[str, Any]
) -> list[dict[str, Any]]:
    he = [r["house_events"] for r in recs if r.get("house_events")]
    first = sum(len(h["founded"]["id"]) + h["founded_dropped"] for h in he)
    gone = sum(len(h["extinct"]["id"]) + h["extinct_dropped"] for h in he)
    sb = rb["summary"]["houses"]
    f: dict[str, Any] = {
        "kind": "houses",
        "alive_from": int(_social_head(ra)["houses"]["alive"]),
        "alive_to": int(sb["alive"]),
        "first_seen": first,
        "extinct": gone,
        "exact": bool(sb["exact"]),
        "origin": int(rb["origin"]),
    }
    table = sorted(_rows_of(rb["houses"]), key=lambda h: (-h["size"], h["id"]))
    if table:
        top = table[0]
        f.update(
            {
                "largest": int(top["id"]),
                "largest_size": int(top["size"]),
                "largest_strict": len(table) < 2 or top["size"] > table[1]["size"],
                "largest_name": top["name"],
            }
        )
    return [f]


def _relation_facts(recs: Sequence[Mapping[str, Any]], rules: ReportRules) -> list[dict[str, Any]]:
    if any(r["relations_dropped"] for r in recs):
        return []
    slotted = [set(r["kin_groups"]["id"]) for r in recs]
    sums: dict[tuple[int, int], list[int]] = {}
    last_distinct: dict[tuple[int, int], Any] = {}
    for rec in recs:
        for x in _rows_of(rec["relations"]):
            key = (x["a"], x["b"])
            cur = sums.get(key, [0, 0])
            sums.update({key: [cur[0] + x["attacks"], cur[1] + x["kills"]]})
            last_distinct.update({key: x["distinct"]})
    out = []
    for (ra, rb), (attacks, kills) in sums.items():
        if ra < 0 or rb < 0 or ra == rb or last_distinct[(ra, rb)] is not True:
            continue
        if attacks < rules.relation_min_attacks or not all(ra in s and rb in s for s in slotted):
            continue
        out.append({"kind": "relation", "a": int(ra), "b": int(rb), "attacks": attacks, "kills": kills})
    out.sort(key=lambda f: (-f["attacks"], f["a"], f["b"]))
    return out


def _label_display(key: str, version: int) -> str:
    if version == 1:
        return key.capitalize()
    return (ARCHETYPE_DISPLAY.get(key) or INTENT_DISPLAY.get(key) or key.capitalize()) if key else key


def _env_default(rd: RunView, a: int, b: int, rules: ReportRules) -> bool:
    """True if no lever of the chapter's rows is away from its default, no patch has moved before ``b`` and no
    catastrophe was applied within one snapshot window before ``a`` or inside the chapter."""
    cfg = rd.cfg
    want = _implied_env((), 0, 0, cfg)  # the defaults
    for r in rd.rows:
        if a <= r["step"] <= b and "env" in r:
            for k in (
                "regen_mult",
                "season_amp",
                "season_period",
                "drift_mult",
                "patch_gain_min",
                "patch_gain_max",
                "patches_off",
            ):
                got = r["env"].get(k)
                if got is None:
                    return False
                same = (
                    int(got) == int(want[k])
                    if k in ("season_period", "patches_off")
                    else _f32(got) == _f32(want[k])
                )
                if not same:
                    return False
    if any(lv.endswith(":pos") for lv in fold_levers(rd.entries, b, rd.start_step, cfg)):
        return False
    return not any(
        e["action"] == "catastrophe"
        and e["source"] != "revert"
        and a - rules.catastrophe_margin <= e["step_applied"] < b
        for e in rd.entries
    )


def _label_facts(
    rd: RunView,
    a: int,
    b: int,
    rules: ReportRules,
    ra: Mapping[str, Any],
    rb: Mapping[str, Any],
    world_key: str,
    excl: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = [r for r in rd.rows if a <= r["step"] <= b]
    versions = {r["observers"].get("intent_version") for r in rows}
    side = rd.sidecar_meta(b)
    version = next(iter(versions)) if len(versions) == 1 else None
    digest = None if side is None else side.get("rules_digest")
    env_default = _env_default(rd, a, b, rules)
    meta = {"intent_version": version, "rules_digest": digest, "key_found": False, "env_default": env_default}
    if len(versions) != 1 or (side is not None and side.get("intent_version") != version):
        excl.update({"labels": {"*": "rules differ"}})
        return [], meta
    if side is None or digest is None:
        return [], meta
    table = LABEL_VERDICTS.get((version, digest, rd.cfg.world.dims, world_key))
    if table is None:
        return [], meta
    meta.update({"key_found": True})
    citable = sorted(
        lb
        for lb, (verdict, n_pos, margin, p, n_perm) in table.items()
        if verdict == "pass"
        and n_pos >= rules.n_pos_min
        and margin >= rules.margin_min
        and abs(p - 1 / (n_perm + 1)) < 1e-6
    )
    oa, ob = ra["observers"], rb["observers"]
    why: dict[str, str] = {}
    origins = {r["observers"]["social"].get("origin") for r in rows}
    restarted = len(origins) != 1 or next(iter(origins)) is None or next(iter(origins)) > a - rules.label_warm
    out: list[dict[str, Any]] = []
    for lb in citable:
        fam = "archetype" if lb in ARCHETYPE_NAMES else "intent"
        shares, labelled = (
            ("archetype_match", "arch_labelled") if fam == "archetype" else ("intent", "intent_labelled")
        )
        if restarted:
            why.update({lb: "observer restarted"})
            continue
        n_a, n_b = oa.get(labelled), ob.get(labelled)
        if not n_a or not n_b or n_a < rules.labelled_min or n_b < rules.labelled_min:
            why.update({lb: "too few labelled"})
            continue
        if not env_default:
            why.update({lb: "environment changed"})
            continue
        sa, sb = (oa.get(shares) or {}).get(lb), (ob.get(shares) or {}).get(lb)
        ca, cb = (
            (None, None)
            if sa is None or sb is None
            else (_exact_count(sa, n_a, rules.exact_tol), _exact_count(sb, n_b, rules.exact_tol))
        )
        if ca is None or cb is None:
            excl.update(
                {"facts_dropped": [*excl["facts_dropped"], f"label {lb}: a share is not a whole count"]}
            )
            continue
        out.append(
            {
                "kind": "label",
                "label": _label_display(lb, version),
                "key": lb,
                "family": fam,
                "from_share": sa,
                "to_share": sb,
                "from_count": ca,
                "to_count": cb,
                "from_of": n_a,
                "to_of": n_b,
            }
        )
    out.sort(key=lambda f: (-abs(f["to_share"] - f["from_share"]), f["key"]))
    if why:
        excl.update({"labels": why})
    return out, meta


def _window_facts(
    rd: RunView,
    ivs: Sequence[Mapping[str, Any]],
    a: int,
    b: int,
    rules: ReportRules,
    by: Mapping[int, Mapping[str, Any]],
    excl: dict[str, Any],
) -> list[dict[str, Any]]:
    """A juxtaposition, gated: the population at the ends of the part of a timed effect's span that lies in
    the chapter, only if nothing else was in force or happening in that window (no other entry, no other lever
    away from its default, no arrivals, no catastrophe deaths)."""
    out = []
    for f in ivs:
        if f["action"] not in TIMED:
            continue
        w0, w1 = max(f["step"], a), min(f["end"] if f["end"] is not None else b, b)
        r0, r1 = by.get(w0), by.get(w1)
        if w1 - w0 < rules.window_min or (w0, w1) == (a, b) or r0 is None or r1 is None:
            continue
        lv = lever_of(f["action"], f["params"])
        others = [i for i, e in enumerate(rd.entries) if i != f["entry"] and w0 <= e["step_applied"] < w1]
        away = [
            k
            for k, st in fold_levers(rd.entries, w1, rd.start_step, rd.cfg).items()
            if away_from_default(k, st.value, rd.cfg)
        ]
        inside = [r for r in rd.rows if w0 < r["step"] <= w1]
        arrivals = sum(r.get("immigrants") or 0 for r in inside)
        killed = sum(r.get("deaths_catastrophe") or 0 for r in inside)
        if others or any(k != lv for k in away) or arrivals or killed:
            excl.update({"windows_confounded": excl["windows_confounded"] + 1})
            continue
        out.append(
            {
                "kind": "window",
                "entry": f["entry"],
                "w0": w0,
                "w1": w1,
                "from": r0["observers"]["alive"],
                "to": r1["observers"]["alive"],
            }
        )
    out.sort(key=lambda f: (-f["w1"], -f["entry"]))
    return out


def _entities(
    rd: RunView,
    kinds: Mapping[str, Sequence[Mapping[str, Any]]],
    origin: int | None,
) -> list[dict[str, Any]]:
    """The entities the facts name, with the raw name from the records (None if no record at or before the
    latest citing fact lists the group)."""
    limit: dict[int, int] = {}

    def want(kid: int, step: int) -> None:
        limit.update({int(kid): max(step, limit.get(int(kid), step))})

    for f in kinds.get("event", ()):
        if f["family"] == "kin_group":
            want(f["system"], f["s1"])
            if f["other"] >= 0:
                want(f["other"], f["s1"])
    for f in kinds.get("group", ()):
        want(f["entity"], kinds["chapter"][0]["end"])
    for f in kinds.get("relation", ()):
        want(f["a"], kinds["chapter"][0]["end"])
        want(f["b"], kinds["chapter"][0]["end"])
    mine = [r for r in rd.records if r.get("origin") == origin] if origin is not None else list(rd.records)
    ents = []
    for kid, lim in sorted(limit.items()):
        name, at = _kin_name(mine, kid, lim)
        ents.append({"kind": "kin_group", "id": kid, "name": name, "name_from": at})
    for f in kinds.get("houses", ()):
        if "largest" in f:
            ents.append({"kind": "house", "id": f["largest"], "name": f["largest_name"], "name_from": None})
    return ents


def _display(kind: str, ident: int, name: str | None) -> str:
    if kind == "house":
        return f"House {name}"
    return f"the {name} kin" if name else f"kin group {ident}"


def compose_sheet(
    *,
    header: Mapping[str, Any],
    kinds: Mapping[str, Sequence[Mapping[str, Any]]],
    entities: Sequence[Mapping[str, Any]],
    excluded: Mapping[str, Any],
    snapshots: Sequence[Mapping[str, Any]] = (),
    rules: ReportRules = REPORT_V1,
) -> dict[str, Any]:
    """The sheet of ``header`` (``run``, ``config_hash``, ``world_key``, ``dims``, ``chapter``, ``origin``,
    ``start_step``, ``labels``), the facts by kind (each kind in its own key order), the entities (``kind``,
    ``id``, ``name``, ``name_from``) and the exclusions.

    Ids ``F1`` to ``Fn`` follow the order of ``FACT_KINDS``; the display names are set here (two entities that
    would render alike all render "kin group {id}"); so are ``quiet`` and the digest of the rules.
    ``build_sheet`` and the tests' scripted sheets both end here."""
    a, b = header["chapter"]
    facts: list[dict[str, Any]] = []
    n = 0
    for kind in FACT_KINDS:
        for f in kinds.get(kind, ()):
            n += 1
            facts.append({"id": f"F{n}", **{k: v for k, v in f.items() if k != "kind"}} | {"kind": kind})
    by_entry = {f["entry"]: f["id"] for f in facts if f["kind"] == "intervention"}
    facts = [{**f, "intervention": by_entry[f["entry"]]} if f["kind"] == "window" else f for f in facts]
    shown = {(e["kind"], e["id"]): _display(e["kind"], e["id"], e["name"]) for e in entities}
    clash = sorted(
        {d for d in shown.values() if list(shown.values()).count(d) > 1 and not d.startswith("House ")}
    )
    ents = [
        {
            "ref": ("K" if e["kind"] == "kin_group" else "H") + str(e["id"]),
            "kind": e["kind"],
            "id": e["id"],
            "display": (
                f"kin group {e['id']}"
                if e["kind"] == "kin_group" and shown[(e["kind"], e["id"])] in clash
                else shown[(e["kind"], e["id"])]
            ),
            "name": None if shown[(e["kind"], e["id"])] in clash else e["name"],
            "name_from": None if shown[(e["kind"], e["id"])] in clash else e["name_from"],
        }
        for e in sorted(entities, key=lambda e: (e["kind"], e["id"]))
    ]
    pop = next((f for f in facts if f["kind"] == "population"), None)
    change = None if pop is None else pop.get("change")
    quiet = (
        pop is not None
        and change is not None
        and abs(change) < rules.quiet_change
        and not pop["deep_dip"]
        and not any(f["kind"] == "event" and f["family"] == "kin_group" for f in facts)
        and not any(f["kind"] == "intervention" and a <= f["step"] < b for f in facts)
    )
    return {
        "report_version": REPORT_VERSION,
        "report_rules_digest": rules.digest(),
        **header,
        "quiet": bool(quiet),
        "entities": ents,
        "facts": facts,
        "snapshots": [dict(s) for s in snapshots],
        "excluded": {**excluded, "name_clash": clash},
    }


def _check_chapter(
    rd: RunView, a: int, b: int, rules: ReportRules, by: Mapping[int, Mapping[str, Any]]
) -> None:
    if not (isinstance(a, int) and isinstance(b, int)) or a >= b:
        raise ReportError(f"the chapter ({a}, {b}] is empty or not made of whole steps")
    every = rules.snapshot_every
    hint = "an unobserved run: build the report on a fork with --set game.observe=true"
    for cut in (a, b):
        if cut % every:
            raise ReportError(f"the cut {cut} is not a multiple of the snapshot cadence {every}")
        row = by.get(cut)
        if row is None:
            raise ReportError(f"the cut {cut} is not the step of a metrics row")
        obs = row.get("observers")
        if not isinstance(obs, Mapping) or not isinstance(obs.get("social"), Mapping):
            raise ReportError(f"the row at step {cut} has no observers block ({hint})")
    inside = [r for r in rd.rows if a <= r["step"] <= b]
    if any(
        not isinstance(r.get("observers"), Mapping) or r["observers"].get("alive") is None for r in inside
    ):
        raise ReportError(f"a row of the chapter has no observers block ({hint})")
    if sum(r["steps"] for r in inside if r["step"] > a) != b - a:
        raise ReportError(f"the metrics rows of ({a}, {b}] do not cover the chapter")


def build_sheet(rd: RunView, a: int, b: int, rules: ReportRules = REPORT_V1) -> dict[str, Any]:
    """The fact sheet of the chapter ``(a, b]`` of ``rd`` (see the module doc). ``ReportError`` (code 2) if
    the chapter cannot be cut from the run's rows (not on the snapshot grid, a gap in the rows, an unobserved
    run), code 1 if a row's ``env`` disagrees with the director's log."""
    steps = [r["step"] for r in rd.rows]
    if steps != sorted(set(steps)):
        raise ReportError("the metrics rows are not in strictly increasing step order")
    by = {r["step"]: r for r in rd.rows}
    _check_chapter(rd, a, b, rules, by)
    check_env(rd, a, b)
    cfg = rd.cfg
    ra, rb = by[a], by[b]
    inside = [r for r in rd.rows if a < r["step"] <= b]
    excl: dict[str, Any] = {
        "labels": {},
        "social": None,
        "events_suspect": 0,
        "events_churn": 0,
        "interventions_dropped": 0,
        "windows_confounded": 0,
        "quota": {},
        "entries_before_start": 0,
        "facts_dropped": [],
    }
    quota: dict[str, int] = {}

    def cap(kind: str, facts: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
        if len(facts) > n:
            quota.update({kind: len(facts) - n})
        return facts[:n]

    kinds: dict[str, list[dict[str, Any]]] = {"chapter": [{"kind": "chapter", "start": a, "end": b}]}
    fa, fb = ra["observers"]["alive"], rb["observers"]["alive"]
    pop: dict[str, Any] = {"kind": "population", "from": fa, "to": fb}
    if fa > 0:
        pop.update({"change": (fb - fa) / fa})
    mins = [(r.get("n_alive_min"), r["step"], r["steps"]) for r in inside]
    deep = False
    if all(m[0] is not None for m in mins):
        lo = min(mins, key=lambda m: (m[0], m[1]))
        deep = lo[0] < rules.dip_mandatory * min(fa, fb)
        pop.update({"min": lo[0], "min_row": lo[1], "min_steps": lo[2]})
    pop.update({"deep_dip": deep})
    kinds.update({"population": [pop]})

    causes = ("starved", "killed", "aged", "catastrophe")
    if all(all(r.get(f"deaths_{c}") is not None for c in causes) for r in inside):
        d = {c: sum(r[f"deaths_{c}"] for r in inside) for c in causes}
        kinds.update({"deaths": [{"kind": "deaths", **d, "total": sum(d.values())}]})
    if all(r.get("births") is not None and r.get("immigrants") is not None for r in inside):
        kinds.update(
            {
                "births": [
                    {
                        "kind": "births",
                        "births": sum(r["births"] for r in inside),
                        "immigrants": sum(r["immigrants"] for r in inside),
                    }
                ]
            }
        )
    forward = [e for e in rd.entries if e["source"] != "revert" and rd.start_step < e["step_applied"] < b]
    bin_may = (
        immigration_on(cfg.population)
        or any(e["action"] == "immigration_pulse" for e in forward)
        or sum(r.get("immigrants") or 0 for r in inside) > 0
    )
    if bin_may:
        excl.update({"facts_dropped": [*excl["facts_dropped"], "immigrant lineage bin"]})
    else:
        if ra.get("clades_alive") is not None and rb.get("clades_alive") is not None:
            kinds.update(
                {"lineages": [{"kind": "lineages", "from": ra["clades_alive"], "to": rb["clades_alive"]}]}
            )
        sa, sb = ra.get("top_clade_share"), rb.get("top_clade_share")
        if sa is not None and sb is not None:
            ca, cb = _exact_count(sa, fa, rules.exact_tol), _exact_count(sb, fb, rules.exact_tol)
            if ca is None or cb is None:
                excl.update(
                    {"facts_dropped": [*excl["facts_dropped"], "top lineage: a share is not a whole count"]}
                )
            else:
                kinds.update(
                    {
                        "top_lineage": [
                            {
                                "kind": "top_lineage",
                                "from_share": sa,
                                "to_share": sb,
                                "from_count": ca,
                                "to_count": cb,
                                "from_of": fa,
                                "to_of": fb,
                            }
                        ]
                    }
                )

    # the director: spans, standing levers, windows
    ivs, ex = spans(rd.entries, a, b, rd.start_step, cfg)
    excl.update({"entries_before_start": ex["entries_before_start"]})
    standing = standing_facts(rd.entries, a, rd.start_step, cfg, b, ivs)
    kept = (ivs + standing)[: rules.block_cap]
    excl.update({"interventions_dropped": len(ivs) + len(standing) - len(kept)})
    ivs_kept = sorted((f for f in kept if f["kind"] == "intervention"), key=lambda f: (f["step"], f["entry"]))
    st_kept = [f for f in standing if f in kept]
    kinds.update({"intervention": ivs_kept, "standing": st_kept})
    wins = _window_facts(rd, ivs_kept, a, b, rules, by, excl)
    kinds.update({"window": cap("window", wins, rules.quota_windows)})

    # the world and its labels
    world_key = config_hash(dataclasses.replace(cfg, director=DirectorConfig(), game=GameConfig()))
    label_facts, label_meta = _label_facts(rd, a, b, rules, ra, rb, world_key, excl)
    kinds.update({"label": cap("label", label_facts, rules.quota_labels)})

    # the social facts, if the records cover the chapter
    recs, why = _social_gate(rd, a, b, rules, by)
    origin = _social_head(rb).get("origin")
    alive_at: dict[int, int] = {
        s: by[s]["observers"]["alive"] for s in _snapshot_steps(a, b, rules.snapshot_every) if s in by
    }
    if why is not None:
        excl.update({"social": why})
    else:
        rec_b = next(r for r in recs if r["step"] == b)
        alive_at = {r["step"]: alive_of(r) for r in recs}
        evs = _event_facts(rd, recs, rules, excl)
        kinds.update({"event": cap("event", evs, rules.quota_events)})
        kinds.update({"group": cap("group", _group_facts(recs, rec_b, alive_at[b]), rules.quota_groups)})
        kinds.update({"houses": _house_facts(recs, ra, rec_b)})
        kinds.update({"relation": cap("relation", _relation_facts(recs, rules), rules.quota_relations)})
        col = rec_b["summary"]["colonies"]
        if col["tracked"] > 0:
            kinds.update(
                {
                    "colonies": [
                        {"kind": "colonies", "tracked": col["tracked"], "largest_size": col["largest"]}
                    ]
                }
            )

    # the end of the run: the extinction stop writes a final off-grid row
    last = rd.rows[-1]
    obs_last = last.get("observers")
    if (
        last["step"] > b
        and last["step"] % rules.snapshot_every
        and not any(r["step"] % rules.snapshot_every == 0 for r in rd.rows if b < r["step"] <= last["step"])
        and isinstance(obs_last, Mapping)
        and obs_last.get("alive") == 0
    ):
        kinds.update({"run_end": [{"kind": "run_end", "step": last["step"], "alive": 0}]})

    excl.update({"quota": quota})
    header = {
        "run": {"seed": cfg.run.seed},
        "config_hash": config_hash(cfg),
        "world_key": world_key,
        "dims": cfg.world.dims,
        "chapter": [a, b],
        "origin": origin,
        "start_step": rd.start_step,
        "labels": label_meta,
    }
    snaps = [
        {"id": f"S{s}", "step": s, "alive": alive_at.get(s)}
        for s in _snapshot_steps(a, b, rules.snapshot_every)
    ]
    return compose_sheet(
        header=header,
        kinds=kinds,
        entities=_entities(rd, kinds, origin if why is None else None),
        excluded=excl,
        snapshots=snaps,
        rules=rules,
    )


# --- reading a sheet ----------------------------------------------------------------------------------------


def _idkey(fid: str) -> tuple[int, int, str]:
    return (0, int(fid[1:]), "") if fid[:1] == "F" and fid[1:].isdigit() else (1, 0, fid)


class _Ctx:
    """The lookups a sheet's candidates and checks share: facts by id (with the pseudo-facts ``excluded`` and
    the ticker's snapshots ``S<step>``), facts by kind, the display names by reference."""

    def __init__(self, sheet: Mapping[str, Any], rules: ReportRules = REPORT_V1) -> None:
        if sheet["report_rules_digest"] != rules.digest():
            raise ReportError("the sheet was built under other ReportRules than the ones given")
        self.sheet, self.rules = sheet, rules
        self.by_id: dict[str, Mapping[str, Any]] = {f["id"]: f for f in sheet["facts"]}
        self.by_id.update({s["id"]: {**s, "kind": "snapshot"} for s in sheet["snapshots"]})
        self.by_id.update({"excluded": {"id": "excluded", "kind": "excluded", **sheet["excluded"]}})
        self.kinds: dict[str, list[Mapping[str, Any]]] = {}
        for f in sheet["facts"]:
            self.kinds.setdefault(f["kind"], []).append(f)
        self.ents = {e["ref"]: e for e in sheet["entities"]}
        self.chapter = self.kinds["chapter"][0]

    def one(self, kind: str) -> Mapping[str, Any] | None:
        return next(iter(self.kinds.get(kind, ())), None)


def _get(fact: Mapping[str, Any], path: str) -> Any:
    cur: Any = fact
    for part in path.split("."):
        cur = cur[int(part)] if isinstance(cur, list) else cur[part]
    return cur


def _slot_raw(ctx: _Ctx, fact: Mapping[str, Any], slot: templates.Slot) -> Any:
    raw = _get(fact, slot.field)
    if slot.fmt == "kin":
        if int(raw) < 0:  # "no group": a gate refuses the template that names it (R5)
            return f"kin group {int(raw)}"
        return ctx.ents[f"K{int(raw)}"]["display"]
    if slot.fmt == "house":
        return ctx.ents[f"H{int(raw)}"]["display"]
    return raw


def _inst(ctx: _Ctx, tid: str, facts: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Template ``tid`` filled from ``facts`` (a fact per kind the template reads; the chapter fact and the
    exclusion counts are always at hand): the text, the slots as ``{name: [fact id, field]}`` and the
    cites."""
    t = TEMPLATES[tid]
    pool = {"chapter": ctx.chapter, "excluded": ctx.by_id["excluded"], **facts}
    values: dict[str, Any] = {}
    slots: dict[str, list[str]] = {}
    for name, sl in t.slots.items():
        f = pool[sl.kind]
        values.update({name: _slot_raw(ctx, f, sl)})
        slots.update({name: [f["id"], sl.field]})
    cites = sorted({s[0] for s in slots.values()}, key=_idkey)
    return {"template": tid, "cites": cites, "slots": slots, "text": render(tid, values)}


# --- the checker (R1 to R7) ---------------------------------------------------------------------------------

_NUMBER = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")
_MASK = "•"

# Words that need a fact to be true, and the templates that may use each (the base id, without "T:").
_GATED_WORDS: Final[Mapping[str, frozenset[str]]] = {
    "rose": frozenset({"P-up", "P-zero", "HP-up"}),
    "fell": frozenset({"P-down", "HP-down"}),
    "unchanged": frozenset({"HP-flat"}),
    "largest": frozenset({"G-r1", "G-r2", "G-r3", "TL", "H-2", "C"}),
    "second": frozenset({"G-r2"}),
    "third": frozenset({"G-r3"}),
    "lowest": frozenset({"P-low"}),
    "new": frozenset({"H-1n", "H-1ne", "H-1xn", "H-1xne"}),
    "overlapped": frozenset({"E-dissolved-o"}),
    "formed": frozenset({"E-born", "E-born-o", "CE-born", "HE-born"}),
    "split": frozenset({"E-split", "CE-split", "HE-split"}),
    "merged": frozenset({"E-merged", "CE-merged", "HE-merged"}),
    "dissolved": frozenset({"E-dissolved", "E-dissolved-o", "CE-dissolved", "HE-dissolved"}),
    "dipped": frozenset({"HP-dip"}),
    "stopped": frozenset({"END"}),
    "catastrophe": frozenset({"D-2", "I-cat"}),
    "replaced": frozenset({"X-replaced"}),
}


def _base(tid: str) -> str:
    return tid[2:] if tid.startswith("T:") else tid


def _sign(change: float) -> str:
    s = templates.fmt_change(change)
    return "0" if s in ("+0.0%", "-0.0%") else s[0]


def _gate_problems(ctx: _Ctx, tid: str, f: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """R5: what each template asserts, re-derived from the facts it cites and not from the condition that
    chose it. Returns the problems (empty if every gate holds)."""
    t, rules = _base(tid), ctx.rules
    out: list[str] = []

    def need(ok: bool, why: str) -> None:
        if not ok:
            out.append(f"R5:{t}:{why}")

    pop = f.get("population")
    if (
        t
        in (
            "P-up",
            "P-down",
            "P-flat",
            "P-near",
            "P-zero",
            "HP-up",
            "HP-down",
            "HP-flat",
            "HP-near",
            "HP-dip",
        )
        and pop
    ):
        fr, to, ch = pop["from"], pop["to"], pop.get("change")
        up = to > fr
        if t == "P-up":
            need(up and ch is not None and _sign(ch) == "+", "rose")
        if t == "HP-up":
            need(up and (ch is None or _sign(ch) == "+"), "rose")
        if t in ("P-down", "HP-down"):
            need(to < fr and ch is not None and _sign(ch) == "-", "fell")
        if t in ("P-flat", "HP-flat"):
            need(fr == to, "flat")
        if t in ("P-near", "HP-near"):
            need(fr != to and ch is not None and _sign(ch) == "0", "near")
        if t == "P-zero":
            need(fr == 0 and to > 0 and ch is None, "zero")
        if t == "HP-dip":
            need(bool(pop["deep_dip"]) and "min" in pop and pop["min"] < min(fr, to), "dip")
    if t == "P-low" and pop:
        need("min" in pop and pop["min"] < min(pop["from"], pop["to"]), "lowest")
    if t in ("D-1", "D-2"):
        d = f["deaths"]
        need(
            d["total"] == d["starved"] + d["killed"] + d["aged"] + d["catastrophe"] and d["total"] > 0,
            "total",
        )
        need((d["catastrophe"] > 0) == (t == "D-2"), "catastrophe")
    if t in ("B-1", "B-2"):
        need((f["births"]["immigrants"] > 0) == (t == "B-2"), "arrivals")
    if t == "LN":
        need(f["lineages"]["from"] != f["lineages"]["to"], "changed")
    if t == "HL":
        need(f["lineages"]["from"] != f["lineages"]["to"], "changed")
    g = f.get("group")
    if t in ("G-r1", "G-r2", "G-r3") and g:
        need(g.get("rank") == int(t[-1]), "rank")
    if t == "G-c" and g:
        need(all(g.get(k) is not None for k in ("born", "died", "died_killed")), "counts")
        need(g.get("died_killed", 0) <= g.get("died", 0), "killed")
    ev = f.get("event")
    if ev and (t.startswith(("E-", "CE-", "HE-"))):
        kind = t.split("-", 1)[1].split("-")[0]
        need(ev["event"] == kind, "event kind")
        need((ev["family"] == "colony") == t.startswith("CE-"), "family")
        o = ev["other"]
        if t in ("E-born-o", "E-dissolved-o", "E-split", "E-merged", "HE-split", "HE-merged"):
            need(o >= 0, "other")
        if t in ("E-born", "E-dissolved"):
            need(o < 0, "no other")
        if t == "E-born-o":
            need(0 <= ev["overlap"] <= ev["size"], "overlap")
        if t.startswith("HE-"):
            need(not ev["reid"] and ev["family"] == "kin_group", "reid")
    h = f.get("houses")
    if t.startswith("H-1") and h:
        suffix = t[3:]
        need((not suffix.startswith("x")) == h["exact"], "exact")
        need("n" not in suffix or h["first_seen"] >= 1, "new")
        need("e" not in suffix or h["extinct"] >= 1, "died out")
    if t == "H-2" and h:
        need(h.get("largest_strict") is True, "largest house")
    if t == "HH" and h:
        need(h["extinct"] >= 1 and h["exact"], "died out")
    lb = f.get("label")
    if t in ("LB-a", "LB-i") and lb:
        need(lb["family"] == ("archetype" if t == "LB-a" else "intent"), "family")
        need(_label_citable(ctx, lb["key"]), "citable")
        need(min(lb["from_of"], lb["to_of"]) >= rules.labelled_min, "labelled")
    r = f.get("relation")
    if t == "R" and r:
        need(
            r["a"] != r["b"] and r["attacks"] >= rules.relation_min_attacks and r["kills"] <= r["attacks"],
            "relation",
        )
    c = f.get("colonies")
    if t in ("C", "C-1") and c:
        need((c["tracked"] == 1) == (t == "C-1") and c["tracked"] >= 1 and c["largest_size"] >= 1, "colonies")
    e = f.get("run_end")
    if t == "END" and e:
        need(e["alive"] == 0 and e["step"] > ctx.chapter["end"], "run end")
    iv = f.get("intervention")
    if iv and t.startswith(("I-", "X-")) and t != "I-more":
        action = {
            "I-regen": "regen_pulse",
            "I-loss": "habitat_loss",
            "I-seasons": "set_seasons",
            "I-drift": "drift_change",
            "I-imm": "immigration_pulse",
            "I-cat": "catastrophe",
        }.get(t)
        if action:
            need(iv["action"] == action, "action")
        if t in ("I-fert", "I-move"):
            need(iv["action"] == "fertility_shift" and ("gain" in iv["params"]) == (t == "I-fert"), "action")
        if t.startswith("X-"):
            need(not is_point(iv["action"], iv["params"]), "point")
        if t.startswith("X-regen"):
            need(iv["end_by"] == "elapsed" and iv["action"] == "regen_pulse", "elapsed")
            back = (iv.get("restored") or {}).get("mult")
            need(back is not None and (_f32(back) == _f32(1.0)) == (t == "X-regen-1"), "base rate")
        if t.startswith("X-gain"):
            need(iv["end_by"] == "elapsed" and iv["action"] in ("habitat_loss", "fertility_shift"), "elapsed")
            back = (iv.get("restored") or {}).get("gain")
            need(back is not None and (_f32(back) == _f32(1.0)) == (t == "X-gain-1"), "base value")
        if t == "X-replaced":
            need(iv["end_by"] == "replaced" and iv["end"] is not None, "replaced")
        if t == "X-open":
            need(iv["end_by"] is None and not is_point(iv["action"], iv["params"]), "open")
    st = f.get("standing")
    if st and t.startswith("S-"):
        lever = {"S-regen": "regen", "S-seasons": "seasons", "S-drift": "drift"}.get(t)
        if lever:
            need(st["lever"] == lever, "lever")
        else:
            need(st["lever"].endswith(":gain"), "lever")
    if t == "I-more":
        need(ctx.by_id["excluded"]["interventions_dropped"] >= 1, "dropped")
    w = f.get("window")
    if t == "W" and w:
        iv2 = ctx.by_id.get(w["intervention"])
        a, b = ctx.sheet["chapter"]
        ok = rules.window_sentences and iv2 is not None and a <= w["w0"] < w["w1"] <= b
        if ok and iv2 is not None:
            ok = w["w0"] >= iv2["step"] and (iv2["end"] is None or w["w1"] <= iv2["end"])
        need(bool(ok), "window")
    sn = f.get("snapshot")
    if t == "T-pop":
        need(sn is not None and sn.get("alive") is not None, "alive")
    return out


def _label_citable(ctx: _Ctx, key: str) -> bool:
    lab = ctx.sheet["labels"]
    table = LABEL_VERDICTS.get(
        (lab["intent_version"], lab["rules_digest"], ctx.sheet["dims"], ctx.sheet["world_key"])
    )
    if not lab["key_found"] or table is None or key not in table:
        return False
    verdict, n_pos, margin, p, n_perm = table[key]
    r = ctx.rules
    return (
        verdict == "pass"
        and n_pos >= r.n_pos_min
        and margin >= r.margin_min
        and abs(p - 1 / (n_perm + 1)) < 1e-6
    )


def _mask(text: str, parts: Sequence[tuple[str, str]]) -> str:
    """``text`` with the name part of each display masked: ``parts`` is ``[(display, name part)]`` ("the
    Braelix kin", "Braelix"; "kin group 12", "12"). Only the display is searched, so a number that equals an
    id is not masked elsewhere; "the", "kin" and "House" stay visible to the vocabulary check."""
    out = text
    for display, name in sorted({p for p in parts if p[0]}, key=lambda p: len(p[0]), reverse=True):
        masked = display[: display.rindex(name)] + _MASK + display[display.rindex(name) + len(name) :]
        out = re.sub(
            r"(?<![A-Za-z0-9])" + re.escape(display) + r"(?![A-Za-z0-9])", masked, out, flags=re.IGNORECASE
        )
    return out


def _name_parts(ctx: _Ctx, slot_values: Sequence[tuple[templates.Slot, Any]]) -> list[tuple[str, str]]:
    """The displays to mask: of each entity slot (the name, or the id of a "kin group {id}" fallback) and the
    label of each label slot."""
    parts: list[tuple[str, str]] = []
    for sl, raw in slot_values:
        if sl.fmt == "label":
            parts.append((str(raw), str(raw)))
        elif sl.fmt in ("kin", "house"):
            ent = next((e for e in ctx.ents.values() if e["display"] == raw), None)
            parts.append(
                (str(raw), (ent["name"] if ent and ent["name"] else str(ent["id"]) if ent else str(raw)))
            )
    return parts


def _mask_words(text: str, words: Sequence[str]) -> str:
    out = text
    for w in sorted(words, key=len, reverse=True):
        out = re.sub(r"(?<![A-Za-z0-9_])" + re.escape(w) + r"(?![A-Za-z0-9_])", _MASK, out)
    return out


def check_string(
    text: str,
    slots: Mapping[str, Any] | None,
    sheet: Mapping[str, Any],
    block: bool = False,
    *,
    rules: ReportRules = REPORT_V1,
    ctx: _Ctx | None = None,
) -> list[str]:
    """The reasons ``text`` may not be shown (empty if it may): the second line of the report.

    ``slots`` is ``{"template": id, "slots": {name: [fact id, field]}}`` as ``candidates`` builds it, or None
    for a string with no template (the corpus and the crafted items: R1 fails, the text rules still run).
    ``block`` says the string is an interventions-block item or a ticker line (lever words and rule
    identifiers allowed).

    R1: the template id is in the table, its slots are exactly the template's, each reads its own field of a
    fact of the declared kind, the cites are those facts, and rendering the template from the sheet gives
    ``text``. R2: the template's slot boundaries. R3: every word of the text, with names and numbers masked,
    is in the closed vocabulary. R4: the number tokens are the rendered number slots. R5: the gates of each
    fact (``_gate_problems``, plus the gated words: "rose", "largest", "formed" and the like only in their
    templates). R6: the lexicon. R7: the length."""
    cx = ctx or _Ctx(sheet, rules)
    reasons: list[str] = []
    facts: dict[str, Mapping[str, Any]] = {}
    slot_values: list[tuple[templates.Slot, Any]] = []
    tid = None if slots is None else slots.get("template")
    role = None
    ok = False
    if slots is None:
        reasons.append("R1:no template")
    elif tid not in TEMPLATES:
        reasons.append(f"R1:unknown template {tid!r}")
    else:
        t = TEMPLATES[tid]
        role = t.role
        try:
            templates.validate({tid: t})
        except ValueError as e:
            reasons.append(f"R2:{e}")
        bound = slots.get("slots") or {}
        if set(bound) != set(t.slots):
            reasons.append("R1:slots differ from the template's")
        else:
            values: dict[str, Any] = {}
            ok = True
            for name, sl in t.slots.items():
                fid, field = bound[name]
                fact = cx.by_id.get(fid)
                if fact is None or fact["kind"] != sl.kind or field != sl.field:
                    reasons.append(f"R1:slot {name} does not read {sl.kind}.{sl.field}")
                    ok = False
                    continue
                try:
                    raw = _slot_raw(cx, fact, sl)
                except (KeyError, IndexError, ValueError):
                    reasons.append(f"R1:slot {name}: no such field")
                    ok = False
                    continue
                values.update({name: raw})
                slot_values.append((sl, raw))
                facts.update({sl.kind: fact})
            if ok:
                if render(tid, values) != text:
                    reasons.append("R1:the text is not the template filled from the sheet")
                cites = sorted({b[0] for b in bound.values()}, key=_idkey)
                if "cites" in slots and sorted(slots["cites"], key=_idkey) != cites:
                    reasons.append("R1:the cites are not the facts of the slots")
                if (role in templates.BLOCK_ROLES) != bool(block):
                    reasons.append("R1:block and body strings differ")
                reasons.extend(_gate_problems(cx, tid, facts))
    parts = (
        [(e["display"], e["name"] or str(e["id"])) for e in cx.ents.values()]
        if slots is None
        else _name_parts(cx, slot_values)
    )
    masked = _mask(text, parts)
    if block:
        masked = _mask_words(masked, sorted(templates.RULE_NAMES))
    if ok:
        words = set(re.findall(r"[a-z]+", masked.lower()))
        for word, allowed in _GATED_WORDS.items():
            if word in words and _base(tid) not in allowed:
                reasons.append(f"R5:the word {word!r} is not for {tid}")
    unnumbered = _NUMBER.sub(_MASK, masked)
    bad = sorted(
        {w for w in templates._WORD.findall(unnumbered.lower()) if w not in templates.TEMPLATE_VOCAB}
    )
    reasons.extend(f"R3:{w}" for w in bad)
    if ok:
        want = Counter(
            re.sub(r"^[+-]|%$", "", templates.format_value(sl, raw))
            for sl, raw in _numeric_slots(tid, slot_values)
        )
        if want != Counter(_NUMBER.findall(_mask(text, parts))):
            reasons.append("R4:the number tokens are not the rendered slots")
    for name, word in lexicon.scan(masked, lexicon.NEVER_LISTS):
        reasons.append(f"R6:{name}:{word}")
    for name, word in lexicon.scan(masked, lexicon.LEVER_LISTS):
        if not block and not (tid == "D-2" and word == "catastrophe"):
            reasons.append(f"R6:{name}:{word}")
    for name, word in lexicon.scan(masked, lexicon.LABEL_LISTS):
        reasons.append(f"R6:{name}:{word}")
    limit = rules.max_headline_chars if role == "headline" else rules.max_chars
    if len(text) > limit:
        reasons.append(f"R7:{len(text)} characters")
    return reasons


def _numeric_slots(tid: str, slot_values: Sequence[tuple[templates.Slot, Any]]):
    """``(slot, raw)`` for each numeric slot occurrence of the template, in text order (a slot used twice
    counts twice)."""
    t = TEMPLATES[tid]
    by_name = dict(zip(t.slots, slot_values, strict=True))
    return [
        by_name[n]
        for n in templates.tokens_of(t.text)
        if n in by_name and by_name[n][0].fmt in ("int", "id", "pct", "change", "lever")
    ]


# --- candidates ---------------------------------------------------------------------------------------------

_SECTION_INDEX: Final = {name: i for i, name in enumerate(templates.SECTION_ORDER)}
_SAL: Final = {name: i for i, name in enumerate(templates.SALIENCE_ORDER)}
_I_TEMPLATE: Final = {
    "regen_pulse": "I-regen",
    "habitat_loss": "I-loss",
    "set_seasons": "I-seasons",
    "drift_change": "I-drift",
    "immigration_pulse": "I-imm",
    "catastrophe": "I-cat",
}
_S_TEMPLATE: Final = {"regen": "S-regen", "seasons": "S-seasons", "drift": "S-drift"}


def _try(
    ctx: _Ctx,
    excluded: list[dict[str, Any]],
    tid: str,
    facts: Mapping[str, Mapping[str, Any]],
    block: bool = False,
):
    """A candidate of ``tid`` if its string passes the checker, else None (recorded in ``excluded``)."""
    c = _inst(ctx, tid, facts)
    why = check_string(c["text"], c, ctx.sheet, block, rules=ctx.rules, ctx=ctx)
    if why:
        excluded.append({"template": tid, "cites": c["cites"], "reasons": why})
        return None
    return c


def lead_of(sheet: Mapping[str, Any], rules: ReportRules = REPORT_V1) -> dict[str, Any] | None:
    """The lead sentence (always first): the population from the start to the end of the chapter, with a
    direction word that agrees with the rendered change."""
    ctx = _Ctx(sheet, rules)
    pop = ctx.one("population")
    if pop is None:
        return None
    ch = pop.get("change")
    if pop["from"] == pop["to"]:
        tid = "P-flat"
    elif ch is None:
        tid = "P-zero"
    elif _sign(ch) == "0":
        tid = "P-near"
    else:
        tid = "P-up" if ch > 0 else "P-down"
    out: list[dict[str, Any]] = []
    c = _try(ctx, out, tid, {"population": pop})
    if c is None:
        raise ReportError(f"the lead {tid} fails its check: {out[0]['reasons']}")
    return c


def _kin_events(ctx: _Ctx) -> list[Mapping[str, Any]]:
    return sorted(
        (f for f in ctx.kinds.get("event", ()) if f["family"] == "kin_group"),
        key=lambda f: (-f["size"], f["s1"], f["system"]),
    )


def _event_template(ev: Mapping[str, Any]) -> str | None:
    kind, o = ev["event"], ev["other"]
    pre = "E-" if ev["family"] == "kin_group" else "CE-"
    if ev["family"] == "colony":
        if ev["reid"] and kind in ("born", "dissolved"):
            return None  # a colony that went on under a new id: the colony sentences have no clause to say so
        return pre + kind
    if kind == "born":
        return "E-born-o" if o >= 0 else "E-born"
    if kind == "dissolved":
        return "E-dissolved-o" if o >= 0 else "E-dissolved"
    return pre + kind if o >= 0 else None


def _house_template(h: Mapping[str, Any]) -> str:
    return (
        "H-1"
        + ("" if h["exact"] else "x")
        + ("n" if h["first_seen"] > 0 else "")
        + ("e" if h["extinct"] > 0 else "")
    )


def _headline_picks(ctx: _Ctx) -> list[tuple[str, dict[str, Mapping[str, Any]]]]:
    """The template-mode headline first, then (only for a change that is small and not a crash) at most one
    alternative of another kind: the first eligible of an event, the houses that died out, the lineages."""
    rules = ctx.rules
    pop = ctx.one("population")
    if pop is None:
        return []
    ch = pop.get("change")
    fr, to = pop["from"], pop["to"]
    if fr == to:
        hp = "HP-flat"
    elif ch is not None and _sign(ch) == "0":
        hp = "HP-near"
    else:
        hp = "HP-up" if to > fr else "HP-down"
    if pop["deep_dip"] and not (ch is not None and ch <= -rules.headline_change):
        hp = "HP-dip"  # a crash that the endpoints hide (or a rise over it): say the dip, not the endpoints
    big = (
        pop["deep_dip"] or (ch is None and to != fr) or (ch is not None and abs(ch) >= rules.headline_change)
    )
    events = [e for e in _kin_events(ctx) if not e["reid"]]
    houses = ctx.one("houses")
    lineages = ctx.one("lineages")
    he = None
    if events and events[0]["system"] is not None:
        tpl = {"born": "HE-born", "split": "HE-split", "merged": "HE-merged", "dissolved": "HE-dissolved"}[
            events[0]["event"]
        ]
        if events[0]["event"] in ("split", "merged") and events[0]["other"] < 0:
            tpl = None
        he = None if tpl is None else (tpl, {"event": events[0]})
    hh = ("HH", {"houses": houses}) if houses and houses["extinct"] >= 1 and houses["exact"] else None
    hl = ("HL", {"lineages": lineages}) if lineages and lineages["from"] != lineages["to"] else None
    first = (hp, {"population": pop})
    if not big and he and events[0]["size"] * rules.headline_event_share >= to:
        first = he
    elif not big and hh:
        first = hh
    if big:
        return [first]
    alt = next((x for x in (he, hh, hl) if x is not None and x[0][:2] != first[0][:2]), None)
    return [first] if alt is None else [first, alt]


def candidates(
    sheet: Mapping[str, Any], rules: ReportRules = REPORT_V1
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """``(body, headlines, block, excluded)`` of a sheet.

    ``body``: the distinct candidate sentences (one per template and fact) in salience order, each ``{id,
    template, cites, slots, text, mandatory, salience, section, order}`` with ids ``C1``.. in that order. The
    mandatory ones are deaths when any, the lowest count when the dip is deep, every kin-group event tied at
    the largest size (three at most, by step and id) and the end of the run.

    ``headlines``: ``H1`` the template-mode headline and at most ``H2``, an alternative of another kind (none
    under a large change or a crash). ``block``: the interventions block, complete and in order (standing
    levers, then each intervention with its end and its window, then the note of what was left out).
    ``excluded``: the candidates whose string failed a check (a bug if not empty)."""
    ctx = _Ctx(sheet, rules)
    excluded: list[dict[str, Any]] = []
    items: list[tuple[tuple, dict[str, Any]]] = []

    def add(
        tid: str, facts: Mapping[str, Mapping[str, Any]], sal: tuple, order: tuple, mandatory: bool = False
    ):
        c = _try(ctx, excluded, tid, facts)
        if c is not None:
            c.update({"mandatory": mandatory, "section": templates.SECTION_OF[tid], "order": list(order)})
            items.append((sal, c))

    sx = _SECTION_INDEX
    d = ctx.one("deaths")
    if d and d["total"] > 0:
        add("D-2" if d["catastrophe"] > 0 else "D-1", {"deaths": d}, (_SAL["D"],), (sx["D"],), True)
    pop = ctx.one("population")
    if pop and "min" in pop and pop["min"] < min(pop["from"], pop["to"]):
        add("P-low", {"population": pop}, (_SAL["P-low"],), (sx["P-low"],), bool(pop["deep_dip"]))
    kin = _kin_events(ctx)
    top = [e for e in kin if kin and e["size"] == kin[0]["size"]]
    top = sorted(top, key=lambda e: (e["s1"], e["system"]))[: rules.max_mandatory_events]
    for e in ctx.kinds.get("event", ()):
        tid = _event_template(e)
        if tid is None:
            continue
        if e["family"] == "kin_group":
            add(
                tid,
                {"event": e},
                (_SAL["E"], -e["size"], e["s1"], e["system"]),
                (sx["E"], e["s1"], -e["size"], e["system"]),
                any(e is m for m in top),
            )
        else:
            add(
                tid,
                {"event": e},
                (_SAL["CE"], -e["size"], e["s1"], e["system"]),
                (sx["CE"], e["s1"], -e["size"], e["system"]),
            )
    for g in ctx.kinds.get("group", ()):
        ranked = g.get("rank") in (1, 2, 3)
        tid = f"G-r{g['rank']}" if ranked else "G-s"
        sal = (_SAL["G-top"] if g.get("rank") == 1 else _SAL["G-rest"], -g["size"], g["entity"])
        ordr = (sx["G"], -g["size"], g["entity"])
        add(tid, {"group": g}, (*sal, 0), (*ordr, 0))
        if all(g.get(k) is not None for k in ("born", "died", "died_killed")):
            add("G-c", {"group": g}, (*sal, 1), (*ordr, 1))
    b = ctx.one("births")
    if b:
        add("B-2" if b["immigrants"] > 0 else "B-1", {"births": b}, (_SAL["B"],), (sx["B"],))
    ln = ctx.one("lineages")
    if ln and ln["from"] != ln["to"]:
        add("LN", {"lineages": ln}, (_SAL["LN"],), (sx["LN"],))
    tl = ctx.one("top_lineage")
    if tl:
        add("TL", {"top_lineage": tl}, (_SAL["TL"],), (sx["TL"],))
    h = ctx.one("houses")
    if h:
        add(_house_template(h), {"houses": h}, (_SAL["H"], 0), (sx["H"], 0))
        if h.get("largest_strict") is True:
            add("H-2", {"houses": h}, (_SAL["H"], 1), (sx["H"], 1))
    for lb in ctx.kinds.get("label", ()):
        delta = -abs(lb["to_share"] - lb["from_share"])
        add(
            "LB-a" if lb["family"] == "archetype" else "LB-i",
            {"label": lb},
            (_SAL["LB"], delta, lb["key"]),
            (sx["LB"], delta, lb["key"]),
        )
    for r in ctx.kinds.get("relation", ()):
        add(
            "R",
            {"relation": r},
            (_SAL["R"], -r["attacks"], r["a"], r["b"]),
            (sx["R"], -r["attacks"], r["a"], r["b"]),
        )
    col = ctx.one("colonies")
    if col:
        add("C-1" if col["tracked"] == 1 else "C", {"colonies": col}, (_SAL["C"],), (sx["C"],))
    end = ctx.one("run_end")
    if end:
        add("END", {"run_end": end}, (_SAL["END"],), (sx["END"],), True)
    items.sort(key=lambda it: it[0])
    body = []
    for n, (_, c) in enumerate(items, start=1):
        body.append({"id": f"C{n}", "salience": n, **c})
    heads = []
    for n, (tid, facts) in enumerate(_headline_picks(ctx), start=1):
        c = _try(ctx, excluded, tid, facts)
        if c is not None:
            heads.append({"id": f"H{n}", **c})
    return body, heads, block_items(ctx, excluded), excluded


def block_items(ctx: _Ctx, excluded: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """The interventions block of a sheet (complete, in order): per standing lever its S-line, then per
    intervention fact by (step, log index) its I-line, its end (X-clause) and its window sentence if the
    sentences of windows are on, then I-more if the cap cut anything."""
    ex = excluded if excluded is not None else []
    out: list[dict[str, Any]] = []

    def add(tid: str, facts: Mapping[str, Mapping[str, Any]], fact_id: str) -> None:
        c = _try(ctx, ex, tid, facts, block=True)
        if c is not None:
            out.append({"fact": fact_id, **c})

    for st in ctx.kinds.get("standing", ()):
        lever = st["lever"]
        tid = _S_TEMPLATE.get(lever) or "S-gain"
        add(tid, {"standing": st}, st["id"])
    windows = {w["intervention"]: w for w in ctx.kinds.get("window", ())}
    for iv in ctx.kinds.get("intervention", ()):
        tid = _I_TEMPLATE.get(iv["action"]) or ("I-fert" if "gain" in iv["params"] else "I-move")
        add(tid, {"intervention": iv}, iv["id"])
        x = _end_template(iv)
        if x is not None:
            add(x, {"intervention": iv}, iv["id"])
        w = windows.get(iv["id"])
        if w is not None and ctx.rules.window_sentences:
            add("W", {"window": w}, w["id"])
    if ctx.by_id["excluded"]["interventions_dropped"] > 0:
        add("I-more", {}, "excluded")
    return out


def _end_template(iv: Mapping[str, Any]) -> str | None:
    if is_point(iv["action"], iv["params"]):
        return None
    if iv["end_by"] == "elapsed":
        r = iv["restored"]
        if iv["action"] == "regen_pulse":
            return "X-regen-1" if _f32(r["mult"]) == _f32(1.0) else "X-regen"
        return "X-gain-1" if _f32(r["gain"]) == _f32(1.0) else "X-gain"
    return "X-replaced" if iv["end_by"] == "replaced" else "X-open"


# --- assembly -----------------------------------------------------------------------------------------------


def _within(c: Mapping[str, Any], pos: Mapping[str, int] | None, group_pos: Mapping[int, int]) -> tuple:
    """The order of a chosen sentence inside its section: by its key, except that in select mode the groups,
    labels and relations keep the model's relative order (events are always in time order)."""
    sec = c["section"]
    if pos is None or sec in ("E", "CE") or len(c["order"]) < 2:
        return (0, *c["order"][1:])
    if sec == "G":
        gid = c["order"][2]
        return (0, group_pos.get(gid, len(pos)), c["order"][3])
    return (0, pos.get(c["id"], len(pos)), *c["order"][1:])


def body_k(
    sheet: Mapping[str, Any], body: Sequence[Mapping[str, Any]], rules: ReportRules = REPORT_V1
) -> int:
    """The number of body sentences of a chapter: ``k_body`` (6), ``k_quiet`` (3) in a quiet chapter, raised
    to the number of mandatory candidates when that is larger. The template mode, the select prompt, its
    schema and its validation all use this one number."""
    k = rules.k_quiet if sheet["quiet"] else rules.k_body
    return max(k, sum(1 for c in body if c["mandatory"]))


def assemble(
    sheet: Mapping[str, Any],
    cands: tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]],
    choice: Mapping[str, Any] | None = None,
    rules: ReportRules = REPORT_V1,
) -> dict[str, Any]:
    """The report of a sheet: ``{headline, lead, body, interventions}``.

    ``choice`` None is the template mode: the mandatory sentences plus the first candidates by salience up to
    ``k`` (6, or 3 in a quiet chapter; raised to the number of mandatory sentences). Otherwise ``choice`` is
    ``{"headline": id or None, "order": [candidate ids]}`` after the policy of the select mode: the sentences
    it names (``k`` at most, each once) plus the mandatory ones it left out.

    Either way the body is displayed in the fixed order of sections (``templates.SECTION_ORDER``), inside a
    section by the section's key; in the select mode the groups, labels and relations keep the order of the
    model, a group's own sentence before its G-c. Nothing here is written by a model: every text is a
    candidate."""
    body, heads, block, _ = cands
    k = body_k(sheet, body, rules)
    mand = [c for c in body if c["mandatory"]]
    by_id = {c["id"]: c for c in body}
    pos: dict[str, int] | None = None
    if choice is None:
        rest = [c for c in body if not c["mandatory"]][: max(0, k - len(mand))]
        chosen = [(c, "mandatory") for c in mand] + [(c, "template") for c in rest]
    else:
        kept: list[str] = []
        for cid in choice.get("order") or ():
            if isinstance(cid, str) and cid in by_id and cid not in kept and len(kept) < k:
                kept.append(cid)
        pos = {cid: i for i, cid in enumerate(kept)}
        chosen = [(by_id[cid], "model") for cid in kept] + [
            (c, "mandatory") for c in mand if c["id"] not in kept
        ]
    group_pos: dict[int, int] = {}
    if pos is not None:
        for c, _ in chosen:
            if c["section"] == "G" and c["id"] in pos:
                gid = c["order"][2]
                group_pos.update({gid: min(pos[c["id"]], group_pos.get(gid, pos[c["id"]]))})

    def key(cu: tuple[dict[str, Any], str]) -> tuple:
        c, origin = cu
        if origin == "mandatory" and pos is not None and c["section"] not in ("E", "CE"):
            return (c["order"][0], (1, *c["order"][1:]))  # what the model left out comes after what it chose
        return (c["order"][0], _within(c, pos, group_pos))

    chosen.sort(key=key)
    head = next((h for h in heads if choice is not None and h["id"] == choice.get("headline")), None)
    template_head = heads[0] if heads else None
    h = head or template_head
    lead = lead_of(sheet, rules)
    return {
        "headline": None
        if h is None
        else {"id": h["id"], "text": h["text"], "origin": "model" if head else "template"},
        "lead": None if lead is None else lead["text"],
        "body": [{"id": c["id"], "text": c["text"], "origin": origin} for c, origin in chosen],
        "interventions": [{"fact": x["fact"], "text": x["text"]} for x in block],
    }


def template_report(sheet: Mapping[str, Any], rules: ReportRules = REPORT_V1) -> dict[str, Any]:
    """The report in the template mode: no model, no network."""
    return assemble(sheet, candidates(sheet, rules), None, rules)


def _ticker_lines(sheet: Mapping[str, Any], rules: ReportRules = REPORT_V1) -> list[dict[str, Any]]:
    """The ticker's lines as candidates (``step``, ``template``, ``cites``, ``slots``, ``text``)."""
    ctx = _Ctx(sheet, rules)
    ex: list[dict[str, Any]] = []
    out: list[dict[str, Any]] = []
    for snap in ctx.sheet["snapshots"]:
        s = snap["step"]
        fact = ctx.by_id[snap["id"]]
        lines = []
        for iv in ctx.kinds.get("intervention", ()):
            if s - rules.snapshot_every <= iv["step"] < s:
                tid = _I_TEMPLATE.get(iv["action"]) or ("I-fert" if "gain" in iv["params"] else "I-move")
                lines.append(_try(ctx, ex, "T:" + tid, {"intervention": iv, "snapshot": fact}, block=True))
        if not lines:
            evs = sorted(
                (e for e in ctx.kinds.get("event", ()) if e["s1"] == s and e["family"] == "kin_group"),
                key=lambda e: (-e["size"], e["system"]),
            )
            tid = _event_template(evs[0]) if evs else None
            if tid is not None:
                lines.append(_try(ctx, ex, "T:" + tid, {"event": evs[0], "snapshot": fact}, block=True))
        if not lines and snap["alive"] is not None:
            lines.append(_try(ctx, ex, "T-pop", {"snapshot": fact}, block=True))
        out.extend({"step": s, **c} for c in lines if c is not None)
    return out


def ticker(sheet: Mapping[str, Any], rules: ReportRules = REPORT_V1) -> list[dict[str, Any]]:
    """The ticker of a chapter, template only: for each snapshot step ``s`` of (a, b], by priority, one line
    "Step {s}: " + the intervention line of each entry applied in [s - 250, s) (after the snapshot at its
    step; two entries on one step give two lines, in log order); else one line with the sentence of the
    largest kin-group event of the snapshot; else "Step {s}: the population was {n}." where the count is
    known. Every line passes the checker. The window sentence is never in the ticker."""
    return [{"step": x["step"], "text": x["text"]} for x in _ticker_lines(sheet, rules)]


# --- the select-and-order mode (docs/DESIGN.md §31; SELECT_PROMPT_VERSION 1) ------------------------------
#
# The model sees code-written candidate sentences and replies with candidate ids. Nothing it writes is shown,
# quoted or put into a prompt: the reply is parsed entry by entry (a bad entry is dropped, a bad structure
# makes the attempt ``invalid``), the policy picks the first attempt that is usable, and ``assemble`` renders
# only candidates, in the code's section order.

SELECT_PROMPT_VERSION: Final = 1
SELECT_SCHEMA_NAME: Final = "world_report_select"
SELECT_TEMPERATURE: Final = 0.4
SELECT_MAX_TOKENS: Final = 200
SELECT_SEED_TAG: Final = 17  # the stream tag of the request seed (the director's is 16)
MAX_REPLY_CHARS: Final = 4096  # a longer reply is invalid (the schema allows a few dozen characters)
DROP_REASONS: Final = ("not_a_string", "unknown_id", "repeat", "beyond_k")
OUTCOMES: Final = ("ok", "invalid", "failed", "too_few")
CHAT_PATH: Final = "/chat/completions"


def is_chat_url(url: str) -> bool:
    """Whether the path of ``url`` ends in ``/chat/completions``, the only endpoint a report request is sent
    to (the owner's LLM rule: no management endpoint is ever the target of a POST)."""
    try:
        return urlsplit(url).path.rstrip("/").endswith(CHAT_PATH)
    except ValueError:
        return False


RETRY_OUTCOMES: Final = ("invalid", "failed", "too_few")  # the outcomes after which the next attempt is made
ENTRY_CHARS: Final = 24  # characters of a dropped entry kept in a record

SELECT_SYSTEM: Final = (
    "You are the editor of a chapter report about a simulated world. Code wrote and checked every sentence, "
    "and you cannot add or change words. Choose the body sentences a reader most needs, most important "
    "first, and choose one headline. Code fixes the order between kinds of sentence, so the order you give "
    "matters only among sentences of the same kind. Sentences marked [always] are shown even if you leave "
    "them out. Do not choose two sentences that repeat each other. Reply with the ids only, as JSON."
)
# The lines of the user message (the format is part of the prompt digest).
_USER_FORMAT: Final = {
    "chapter": "Chapter: steps {a} to {b} of a {dims}D world.",
    "lead": "Lead (always first): {text}",
    "body": "Body candidates (choose {lo} to {hi}):",
    "body_all": "Body candidates (choose {lo}):",
    "line": "{id}: {text}",
    "always": "{id} [always]: {text}",
    "head": "Headline candidates (choose 1):",
}


def source_words_digest() -> str:
    """sha256 of the words that name the author of an intervention and of the rule identifiers (their content
    reaches the sheet's block, which the version numbers do not pin)."""
    return hashlib.sha256(
        _canon(
            {"source_words": dict(templates.SOURCE_WORDS), "rule_names": sorted(templates.RULE_NAMES)}
        ).encode("utf-8")
    ).hexdigest()


def select_prompt_digest() -> str:
    """sha256 of everything fixed in the select request: the system message, the user message's format, the
    schema's name, the sampling values, the seed stream and the reply limit."""
    return hashlib.sha256(
        _canon(
            {
                "version": SELECT_PROMPT_VERSION,
                "system": SELECT_SYSTEM,
                "user": dict(_USER_FORMAT),
                "schema_name": SELECT_SCHEMA_NAME,
                "temperature": SELECT_TEMPERATURE,
                "max_tokens": SELECT_MAX_TOKENS,
                "seed_tag": SELECT_SEED_TAG,
                "max_reply_chars": MAX_REPLY_CHARS,
                "drop_reasons": list(DROP_REASONS),
            }
        ).encode("utf-8")
    ).hexdigest()


def record_versions(rules: ReportRules = REPORT_V1) -> dict[str, Any]:
    """Every version number and table digest a record stores and ``verify`` compares (§8): the versions of
    the sheet, the templates, the checker and the select prompt, and the digests of the content they do not
    pin."""
    return {
        "report_version": REPORT_VERSION,
        "template_version": templates.TEMPLATE_VERSION,
        "checker_version": CHECKER_VERSION,
        "select_prompt_version": SELECT_PROMPT_VERSION,
        "report_rules_digest": rules.digest(),
        "template_digest": templates.TEMPLATE_DIGEST,
        "lexicon_digest": lexicon.LEXICON_DIGEST,
        "label_table_digest": label_table_digest(),
        "known_social_digest": known_social_digest(),
        "source_words_digest": source_words_digest(),
        "select_prompt_digest": select_prompt_digest(),
    }


def select_n_min(k: int, body: Sequence[Mapping[str, Any]]) -> int:
    """The fewest entries a usable reply holds: ``k - 1``, or every candidate when there are fewer (a one-id
    reply would leave only the lead and the mandatory sentences, N14)."""
    return min(k - 1, len(body))


def _vacuous(cands: tuple[Any, ...]) -> bool:
    """True when the model would have nothing to choose (no body or no headline candidate): no request."""
    return not cands[0] or not cands[1]


def select_messages(
    sheet: Mapping[str, Any],
    cands: tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]],
    rules: ReportRules = REPORT_V1,
) -> list[dict[str, str]]:
    """The chat messages of the select request: the fixed system message and a user message made of the
    chapter, the lead, the body candidates in salience order (the mandatory ones marked ``[always]``) and the
    headline candidates. Every text in it is a candidate string written by code; no intervention, window or
    setting and nothing a model wrote."""
    body, heads = cands[0], cands[1]
    lead = lead_of(sheet, rules)
    if lead is None or not body or not heads:
        raise ReportError("the sheet has no lead, body or headline candidate to put in a prompt")
    k = body_k(sheet, body, rules)
    a, b = sheet["chapter"]
    f = _USER_FORMAT
    lo, hi = select_n_min(k, body), min(k, len(body))
    lines = [
        f["chapter"].format(a=f"{a:,}", b=f"{b:,}", dims=sheet["dims"]),
        f["lead"].format(text=lead["text"]),
        "",
        f["body" if lo < hi else "body_all"].format(lo=lo, hi=hi),
        *[f["always" if c["mandatory"] else "line"].format(id=c["id"], text=c["text"]) for c in body],
        "",
        f["head"],
        *[f["line"].format(id=h["id"], text=h["text"]) for h in heads],
    ]
    return [{"role": "system", "content": SELECT_SYSTEM}, {"role": "user", "content": "\n".join(lines)}]


def select_schema(
    cands: tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]],
    k: int,
) -> dict[str, Any]:
    """The strict JSON schema of the reply: a headline id and 1 to ``k`` body ids (at least ``select_n_min``),
    every string and array bounded. ``uniqueItems`` is stated but not relied on (the parser drops repeats)."""
    body, heads = cands[0], cands[1]
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["headline", "order"],
        "properties": {
            "headline": {"type": "string", "maxLength": 4, "enum": [h["id"] for h in heads]},
            "order": {
                "type": "array",
                "minItems": select_n_min(k, body),
                "maxItems": k,
                "uniqueItems": True,
                "items": {"type": "string", "maxLength": 4, "enum": [c["id"] for c in body]},
            },
        },
    }


def select_seed(run_seed: int, step: int, attempt: int) -> int:
    """The ``seed`` request parameter of attempt ``attempt`` (0, 1, ..) for the chapter ending at ``step``: a
    non-negative int31 of ``(run.seed, step, 17, attempt)``. A server may ignore it."""
    return (
        int(np.random.SeedSequence([run_seed, step, SELECT_SEED_TAG, attempt]).generate_state(1)[0])
        & 0x7FFFFFFF
    )


def _no_constant(name: str) -> Any:
    raise ValueError(f"the JSON constant {name} is not allowed")


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    keys = [key for key, _ in pairs]
    if len(set(keys)) != len(keys):
        raise ValueError(f"duplicate key {next(k for k in keys if keys.count(k) > 1)!r}")
    return dict(pairs)


def _entry_repr(x: Any) -> str:
    """A dropped entry as a short, JSON-safe string for a record (never a nested value)."""
    if isinstance(x, str):
        return x.translate({c: " " for c in range(32)}).replace("\x7f", " ")[:ENTRY_CHARS]
    if x is None or isinstance(x, bool | int | float):
        return repr(x)[:ENTRY_CHARS]
    return f"<{type(x).__name__}>"


def parse_choice(
    text: str,
    cands: tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]],
    k: int,
) -> dict[str, Any]:
    """The model's reply as a choice: ``{"outcome": "ok"|"invalid"|"too_few", "error": str|None, "choice":
    {"headline": id|None, "order": [ids], "dropped": [{"entry", "reason"}]}|None}``.

    **Reply level** (``invalid``, no choice): not one JSON object, a duplicate key, ``NaN`` or
    ``Infinity``, keys other than exactly ``headline`` and ``order``, ``order`` not a list, a reply of more
    than ``MAX_REPLY_CHARS``. **Entry level** (the entry is dropped, never the reply): not a string, an
    unknown id, a repeat (the first stays), beyond ``k``. An unknown or non-string headline is ``None`` (the
    template headline is then used). Fewer kept entries than ``select_n_min`` is ``too_few`` (the choice is
    kept for the record)."""
    body, heads = cands[0], cands[1]
    if len(text) > MAX_REPLY_CHARS:
        return {
            "outcome": "invalid",
            "error": f"the reply is longer than {MAX_REPLY_CHARS} characters",
            "choice": None,
        }
    try:
        data = json.loads(text, parse_constant=_no_constant, object_pairs_hook=_unique_pairs)
    except (ValueError, RecursionError) as e:
        return {
            "outcome": "invalid",
            "error": f"not valid JSON ({type(e).__name__}: {str(e)[:80]})",
            "choice": None,
        }
    if not isinstance(data, dict):
        return {
            "outcome": "invalid",
            "error": f"the reply is a {type(data).__name__}, not an object",
            "choice": None,
        }
    if set(data) != {"headline", "order"}:
        return {
            "outcome": "invalid",
            "error": f"the keys are {sorted(data)[:4]}, not headline and order",
            "choice": None,
        }
    entries = data["order"]
    if not isinstance(entries, list):
        return {"outcome": "invalid", "error": "order is not a list", "choice": None}
    ids = {c["id"] for c in body}
    kept: list[str] = []
    dropped: list[dict[str, str]] = []
    for entry in entries:
        if not isinstance(entry, str):
            reason = "not_a_string"
        elif entry not in ids:
            reason = "unknown_id"
        elif entry in kept:
            reason = "repeat"
        elif len(kept) >= k:
            reason = "beyond_k"
        else:
            kept.append(entry)
            continue
        dropped.append({"entry": _entry_repr(entry), "reason": reason})
    head = data["headline"]
    head_id = head if isinstance(head, str) and head in {h["id"] for h in heads} else None
    choice = {"headline": head_id, "order": kept, "dropped": dropped}
    need = select_n_min(k, body)
    if len(kept) < need:
        return {"outcome": "too_few", "error": f"{len(kept)} entries kept, {need} needed", "choice": choice}
    return {"outcome": "ok", "error": None, "choice": choice}


def judge_attempt(
    attempt: Mapping[str, Any],
    cands: tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]],
    k: int,
) -> dict[str, Any]:
    """``{"outcome", "error", "choice"}`` of a stored attempt, derived again from its ``response``: an attempt
    with no response failed (the HTTP call, a timeout or an empty reply); any other is parsed."""
    response = attempt.get("response")
    if not isinstance(response, str):
        return {"outcome": "failed", "error": attempt.get("error"), "choice": None}
    return parse_choice(response, cands, k)


def settle(attempts: Sequence[Mapping[str, Any]]) -> tuple[int | None, str | None]:
    """The policy over the attempts in order: ``(index of the attempt used, fallback)``. The first attempt
    that is ``ok`` is used (``fallback`` "headline" if its headline was unknown, else None); with none, the
    report is the template report and ``fallback`` is the last attempt's outcome."""
    for i, a in enumerate(attempts):
        if a["outcome"] == "ok":
            return i, ("headline" if a["choice"]["headline"] is None else None)
    return None, (attempts[-1]["outcome"] if attempts else None)


def _attempt(
    ex: Any,
    seed: int,
    cands: tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]],
    k: int,
) -> dict[str, Any]:
    """One request's outcome as a record entry from the client's ``Exchange``. The reply text is cleaned
    (control characters dropped, whitespace collapsed) and stored; the cleaned text is what is parsed, so a
    replay parses the same bytes."""
    from alife.director.llm import clean_text  # lazy: keeps the module's import light

    latency = round(float(ex.latency_s), 3)
    finish = clean_text(ex.finish_reason, 40) if isinstance(ex.finish_reason, str) else None
    base = {"seed": seed, "latency_s": latency, "finish_reason": finish, "usage": ex.usage}
    if ex.error is not None or not isinstance(ex.text, str):
        error = clean_text(ex.error or "no reply", 200)
        return {**base, "outcome": "failed", "error": error, "response": None, "choice": None}
    response = clean_text(ex.text, MAX_REPLY_CHARS + 1)
    if not response:
        return {
            **base,
            "outcome": "failed",
            "error": "empty: the reply has no content",
            "response": None,
            "choice": None,
        }
    return {**base, "response": response, **parse_choice(response, cands, k)}


def _digest_of(x: Any) -> str:
    return hashlib.sha256(_canon(x).encode("utf-8")).hexdigest()


def _cand_views(
    cands: tuple[Any, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """The stored form of the candidates, the headlines and the strings the checker excluded."""
    body, heads, _block, excluded = cands
    return (
        [{k: c[k] for k in ("id", "template", "cites", "text", "mandatory", "salience")} for c in body],
        [{k: h[k] for k in ("id", "template", "cites", "text")} for h in heads],
        [{k: x[k] for k in ("template", "cites", "reasons")} for x in excluded],
    )


def narrate(
    sheet: Mapping[str, Any],
    client: Any | None,
    *,
    rules: ReportRules = REPORT_V1,
    attempts: int = 2,
) -> dict[str, Any]:
    """The record (§8) of a chapter's report. ``client`` None is the template mode: no request. Otherwise
    ``client`` (a ``director.llm.Client``: ``url``, ``model`` and ``chat(body)``) gets one select request per
    attempt, at most ``attempts``, stopping at the first that is usable; after the attempts the template
    report is used and ``fallback`` says why. The request body is built here (never ``Client.body``) with the
    model named explicitly. No global state; the client's HTTP call is the only side effect."""
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    if client is not None:
        if not isinstance(client.model, str) or not client.model.strip():
            raise ValueError("the client must name its model (a non-empty string): every request names it")
        if not isinstance(client.url, str) or not is_chat_url(client.url):
            raise ValueError(f"the client's URL must end in {CHAT_PATH}, got {client.url!r}")
    cands = candidates(sheet, rules)
    k = body_k(sheet, cands[0], rules)
    views = _cand_views(cands)
    log: list[dict[str, Any]] = []
    messages: list[dict[str, str]] | None = None
    request: dict[str, Any] | None = None
    if client is not None and not _vacuous(cands):
        messages = select_messages(sheet, cands, rules)
        request = {"temperature": SELECT_TEMPERATURE, "max_tokens": SELECT_MAX_TOKENS}
        fmt = {
            "type": "json_schema",
            "json_schema": {"name": SELECT_SCHEMA_NAME, "strict": True, "schema": select_schema(cands, k)},
        }
        a, b = sheet["chapter"]
        for i in range(attempts):
            seed = select_seed(sheet["run"]["seed"], b, i)
            payload = {
                "model": client.model,
                "messages": [dict(m) for m in messages],
                **request,
                "seed": seed,
                "stream": False,
                "response_format": fmt,
            }
            log.append(_attempt(client.chat(payload), seed, cands, k))
            if log[-1]["outcome"] == "ok":
                break
    used, fallback = settle(log)
    choice = None if used is None else log[used]["choice"]
    report = assemble(
        sheet,
        cands,
        None if choice is None else {"headline": choice["headline"], "order": choice["order"]},
        rules,
    )
    a, b = sheet["chapter"]
    return {
        "step": b,
        "chapter_start": a,
        **record_versions(rules),
        "facts_digest": facts_digest(sheet),
        "candidate_set_digest": _digest_of([views[0], views[1]]),
        "facts": sheet,
        "candidates": views[0],
        "headlines": views[1],
        "excluded_candidates": views[2],
        "mode": "template" if client is None else "select",
        "model": None if client is None else client.model,
        "url": None if client is None else client.url,
        "messages": messages,
        "request": request,
        "attempts": log,
        "fallback": fallback,
        "report": report,
        "ticker": ticker(sheet, rules),
    }


def _plain(x: Any) -> Any:
    """``x`` as it reads back from JSON (tuples are lists), to compare a computed value with a stored one."""
    return json.loads(json.dumps(x, allow_nan=False))


def verify_record(
    rec: Mapping[str, Any], sheet_now: Mapping[str, Any] | None, rules: ReportRules = REPORT_V1
) -> dict[str, Any]:
    """Replay of one stored record (§8), never calling a model. ``sheet_now`` is the sheet recomputed from the
    run for the record's chapter (None if the caller did not recompute it).

    ``{"status", "sheet_differs", "versions", "reasons"}``: ``versions_differ`` when a version number or a
    table digest of the record is not the code's (``versions`` lists them; the sheet comparison is still
    reported); else the replay from the **stored** facts: the facts against their digest, the candidates and
    headlines against the candidate-set digest and the stored views, a select record's messages rebuilt byte
    for byte, every attempt parsed again (outcome, choice) with the policy, then the report and the ticker
    rendered again; any difference is ``mismatch``. With nothing wrong, ``stale`` if the recomputed sheet's
    digest is not the record's, else ``ok``. The audit fields of the exchange (``model``, ``url``, each
    attempt's ``latency_s``, ``usage`` and ``finish_reason``) cannot be re-derived from the facts: only the
    shape of ``model`` and ``url`` is checked, so an edit of their values passes."""
    current = record_versions(rules)
    diffs = {k: [rec.get(k), v] for k, v in current.items() if rec.get(k) != v}
    sheet_differs = sheet_now is not None and facts_digest(sheet_now) != rec.get("facts_digest")
    if diffs:
        return {"status": "versions_differ", "sheet_differs": sheet_differs, "versions": diffs, "reasons": []}
    reasons: list[str] = []
    try:
        reasons = _replay(rec, rules)
    except (ReportError, KeyError, TypeError, ValueError, IndexError, AttributeError) as e:
        reasons = [f"the stored record cannot be replayed ({type(e).__name__}: {str(e)[:120]})"]
    status = "mismatch" if reasons else ("stale" if sheet_differs else "ok")
    return {"status": status, "sheet_differs": sheet_differs, "versions": {}, "reasons": reasons}


def _replay(rec: Mapping[str, Any], rules: ReportRules) -> list[str]:
    sheet = rec["facts"]
    reasons: list[str] = []

    def same(name: str, stored: Any, computed: Any) -> None:
        if stored != _plain(computed):
            reasons.append(f"{name}: the stored value is not the one the code computes from the stored facts")

    if facts_digest(sheet) != rec["facts_digest"]:
        reasons.append("facts_digest: the stored facts do not match their digest")
    if [sheet["chapter"][1], sheet["chapter"][0]] != [rec["step"], rec["chapter_start"]]:
        reasons.append("step: the record's chapter is not the chapter of its facts")
    cands = candidates(sheet, rules)
    views = _cand_views(cands)
    if _digest_of([views[0], views[1]]) != rec["candidate_set_digest"]:
        reasons.append(
            "candidate_set_digest: the stored candidates and headlines differ from the recomputed ones"
        )
    same("candidates", rec["candidates"], views[0])
    same("headlines", rec["headlines"], views[1])
    same("excluded_candidates", rec["excluded_candidates"], views[2])
    k = body_k(sheet, cands[0], rules)
    mode = rec["mode"]
    log = rec["attempts"]
    if mode == "template":
        if log or rec["model"] is not None or rec["messages"] is not None or rec["fallback"] is not None:
            reasons.append("mode: a template record holds a model, messages, attempts or a fallback")
        choice = None
    elif mode == "select":
        # model, url, latency_s, usage and finish_reason are audit data of the exchange: nothing in the
        # stored facts can reproduce them, so only the shape is checked (a model and a chat URL)
        if not isinstance(rec["model"], str) or not rec["model"].strip():
            reasons.append("model: a select record names no model")
        if not isinstance(rec["url"], str) or not is_chat_url(rec["url"]):
            reasons.append(f"url: a select record's URL does not end in {CHAT_PATH}")
        vacuous = _vacuous(cands)
        same("messages", rec["messages"], None if vacuous else select_messages(sheet, cands, rules))
        same(
            "request",
            rec["request"],
            None if vacuous else {"temperature": SELECT_TEMPERATURE, "max_tokens": SELECT_MAX_TOKENS},
        )
        if vacuous and log:
            reasons.append("attempts: a request was recorded for a chapter with nothing to choose")
        judged: list[dict[str, Any]] = []
        for i, att in enumerate(log):
            j = judge_attempt(att, cands, k)
            judged.append(j)
            if (att["outcome"], att["choice"]) != (j["outcome"], _plain(j["choice"])):
                reasons.append(
                    f"attempts[{i}]: the stored outcome or choice is not the one the stored reply gives"
                )
            if j["outcome"] != "failed" and att["error"] != j["error"]:
                reasons.append(f"attempts[{i}]: the stored error is not the one the stored reply gives")
            if att["seed"] != select_seed(sheet["run"]["seed"], rec["step"], i):
                reasons.append(f"attempts[{i}]: the seed is not the one of the chapter and attempt")
        used, fallback = settle(judged)
        if used is not None and used != len(log) - 1:
            reasons.append("attempts: an attempt follows the one that was used")
        if rec["fallback"] != fallback:
            reasons.append(f"fallback: stored {rec['fallback']!r}, replay {fallback!r}")
        choice = None if used is None else judged[used]["choice"]
    else:
        raise ReportError(f"unknown mode {mode!r}")
    report = assemble(
        sheet,
        cands,
        None if choice is None else {"headline": choice["headline"], "order": choice["order"]},
        rules,
    )
    same("report", rec["report"], report)
    same("ticker", rec["ticker"], ticker(sheet, rules))
    return reasons
