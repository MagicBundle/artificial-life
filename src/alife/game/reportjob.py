"""The live World Report of a played world: the report job and the ``RPRT`` object (G3b spec §5.1, §7.3,
DESIGN §34.4).

At a chapter end the play loop hands the chapter ``(a, b]`` to a ``ReportJob``, which builds the report on a
worker thread, appends the N1 record to ``reports.jsonl`` and hands the ``RPRT`` object to the stream. **The
live record is the offline record**: the job calls the functions ``python -m alife.analysis.world_report
build`` calls on the files of the run (``report.load_run``, ``build_sheet``, ``narrate`` in its template
mode), and nothing it uses is held in memory that the files do not have, with one exception made equal on
purpose: the sidecar of the observer at step ``b`` is written inside the checkpoint that has just started,
so the job is handed the ``meta`` it will hold (``RunView.sidecars``). So ``world_report verify RUN
--reports RUN/reports.jsonl`` passes on a played world, and a build of the same chapters writes the same
records.

The rows, the social record and the storm lines of step ``b`` are on disk before the checkpoint of ``b``
starts (the runner appends the row and the record, then checkpoints), so the job runs inside the save stall
and the card of the finished chapter is filled when the stall ends. It reads from the end of the logs (the
sheet of a chapter needs nothing older than ``a``) and from the kin thread table saved with the checkpoint
of ``a``; when that table is gone it reads the whole log (``report.TailError``).

Host code: no mlx, no array, no subscript assignment (``tests/test_lint.py``). The one thread besides the
loop's is the job's worker; it is not a daemon (a record being written is not cut off), and ``close`` joins it
with a timeout.
"""

from __future__ import annotations

import queue
import threading
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from alife.director import pacing
from alife.director.actions import PLAYER_BUDGET, PLAYER_BUDGET_WINDOW
from alife.game import lexicon, report, templates
from alife.runio import ALERTS_FILE, DIRECTOR_LOG_FILE, RunDir

__all__ = [
    "ARRIVAL_REASON",
    "BACKLOG_REPORTS",
    "CLOSE_TIMEOUT_S",
    "ReportJob",
    "chapter_number",
    "missing_chapters",
    "origin_chapters",
    "report_object",
    "since_of",
    "stored_objects",
    "withheld_object",
]

ARRIVAL_REASON: Final = "this world began part way through it"  # the notice of the first, partial chapter
BACKLOG_REPORTS: Final = 3  # the reports a world being opened is sent, and the chapters rebuilt if missing
CLOSE_TIMEOUT_S: Final = 5.0  # the join of the worker at shutdown

Emit = Callable[[int, dict[str, Any], bool], None]  # (step, RPRT object, backlog)


# --- the object ---------------------------------------------------------------------------------------------


def origin_chapters(origin_step: int, length: int) -> int:
    """``n_origin``: the absolute chapters before a world's own numbering starts, ``origin_step // length``
    (``origin_step`` is the planet's ``made_at_step``, 0 for a world with no planet). A fork made at 450,000
    has ``n_origin`` 75, so its first chapter, absolute 76, is "Chapter 1" (DESIGN §34.2.1)."""
    return max(origin_step, 0) // length


def chapter_number(b: int, length: int, n_origin: int) -> int:
    """The chapter the player sees for the chapter ending at ``b``: ``b // length - n_origin``. The one place
    the server derives it (the live ``RPRT``, a withheld one, the backlog rebuilt from ``reports.jsonl``); the
    client's bar and card subtract the same ``n_origin`` from ``CLCK.chapter`` (DESIGN §34.2.1)."""
    return max(0, b // length - n_origin)


def _versions() -> dict[str, int]:
    return {
        "report": report.REPORT_VERSION,
        "template": templates.TEMPLATE_VERSION,
        "checker": report.CHECKER_VERSION,
        "lexicon": lexicon.LEXICON_VERSION,
    }


def since_of(
    entries: Sequence[Mapping[str, Any]],
    alert_lines: Sequence[Mapping[str, Any]],
    a: int,
    b: int,
    length: int,
) -> dict[str, Any]:
    """The ``since`` strip of the chapter ``(a, b]`` (a pure function of the director's log and the alert log
    up to ``b``): the player's influence at ``b`` (the cap less the cost of the player's entries in the last
    ``PLAYER_BUDGET_WINDOW`` steps), the alerts of the alert surface raised in the chapter by class (the
    chapter alert itself and the rate limit's summary aside), and the timed effects, the director's and the
    player's, that end within the next chapter.

    The strip is built inside the checkpoint stall of ``b``, so it holds what exists then and nothing else,
    for the job and for a world being opened alike: a player's record at ``b`` is applied after the
    checkpoint (rule P) and is not in it, and the alerts of the row at ``b`` are raised after the checkpoint
    too, so the alerts counted are those of ``[a, b)`` (one at ``b`` is counted by the next chapter, one at
    ``a`` by this)."""
    visible = [e for e in entries if not (report.is_player(e) and int(e["step_applied"]) >= b)]
    mine = pacing.fold(visible, b, "player")
    used = pacing.budget_used(mine, b, PLAYER_BUDGET_WINDOW)
    raised = Counter(
        str(x["class"])
        for x in alert_lines
        if a <= int(x.get("step", -1)) < b
        and x.get("state") == "raised"
        and x.get("surface") == "alert"
        and x.get("class") not in ("chapter", "summary")
    )
    ending = [
        {"action": eff.action, "end": int(eff.end)}
        for eff in sorted(mine.active.values(), key=lambda e: (e.end, e.key))
        if eff.end <= b + length
    ]
    return {
        "influence": {"have": max(0, PLAYER_BUDGET - used), "cap": PLAYER_BUDGET},
        "alerts": {k: raised[k] for k in sorted(raised)},
        "ending": ending,
    }


def _z(fact: Mapping[str, Any]) -> dict[str, Any]:
    """The planet-frame ``z`` of a fact's place on a sphere (the facts of a box have none)."""
    return {"z": fact["z"]} if fact.get("z") is not None else {}


def _entity_of(sheet: Mapping[str, Any], fact: Mapping[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """The entities a cited fact names: ``(key, {kind, id, name, x, y})``. A thread and a group are kin
    groups (key ``K<kin id>``), a storm is ``S<id>``; the position is the one the fact holds (None if it holds
    none: the client finds the group in its cast)."""
    kind = fact["kind"]
    if kind == "thread":
        return [
            (
                f"K{fact['kin']}",
                {
                    "kind": "kin",
                    "id": fact["kin"],
                    "name": fact["name"],
                    "x": fact.get("x"),
                    "y": fact.get("y"),
                    **_z(fact),
                },
            )
        ]
    if kind == "storm":
        return [
            (
                f"S{fact['storm']}",
                {
                    "kind": "storm",
                    "id": fact["storm"],
                    "name": fact["name"],
                    "x": fact["x"],
                    "y": fact["y"],
                    **_z(fact),
                },
            )
        ]
    names = {e["ref"]: e.get("name") for e in sheet["entities"]}
    if kind == "group":
        key = f"K{fact['entity']}"
        return [
            (
                key,
                {
                    "kind": "kin",
                    "id": fact["entity"],
                    "name": names.get(key),
                    "x": fact.get("x"),
                    "y": fact.get("y"),
                    **_z(fact),
                },
            )
        ]
    if kind == "relation":
        return [
            (f"K{i}", {"kind": "kin", "id": i, "name": names.get(f"K{i}"), "x": None, "y": None})
            for i in (fact["a"], fact["b"])
        ]
    return []


def _refs(
    sheet: Mapping[str, Any],
    by_id: Mapping[str, Mapping[str, Any]],
    template: str | None,
    cites: Sequence[str],
    entities: dict[str, dict[str, Any]],
) -> list[str]:
    """The entity keys of a sentence (the facts it cites that are entities), added to ``entities``. The house
    of the largest-house sentence is an entity too."""
    refs: list[str] = []
    for fid in cites:
        fact = by_id.get(fid)
        if fact is None:
            continue
        found = _entity_of(sheet, fact)
        if template == "H-2" and fact["kind"] == "houses" and "largest" in fact:
            found = [
                (
                    f"H{fact['largest']}",
                    {
                        "kind": "house",
                        "id": fact["largest"],
                        "name": fact["largest_name"],
                        "x": None,
                        "y": None,
                    },
                )
            ]
        for key, ent in found:
            if key not in refs:
                refs.append(key)
            entities.update({key: ent})
    return refs


def _intervention_sentences(items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The interventions block as ``RPRT`` sentences. A fact can have several sentences (what was done, and
    that it was still in place), so the ids of a block are made unique: the fact's id for its first sentence,
    ``<fact>.2``, ``<fact>.3`` for the next ones. A client lists sentences by id."""
    out = []
    for i, x in enumerate(items):
        fact = str(x["fact"])
        nth = 1 + sum(
            1 for y in items[:i] if str(y["fact"]) == fact
        )  # (no subscript assignment in host code)
        out.append({"id": fact if nth == 1 else f"{fact}.{nth}", "text": x["text"], "refs": []})
    return out


def report_object(
    rec: Mapping[str, Any],
    *,
    n_origin: int,
    length: int,
    since: Mapping[str, Any],
    latency_s: float | None = None,
) -> dict[str, Any]:
    """The ``RPRT`` object (``streamrec.check_report``) of an N1 record: the template report of the chapter
    with the entities its sentences name (``refs``), the interventions block, the ticker, the ``since``
    strip and the versions the record was made under. ``chapter`` is the number the player sees (``n_abs -
    n_origin``)."""
    sheet = rec["facts"]
    a, b = rec["chapter_start"], rec["step"]
    by_id = {f["id"]: f for f in sheet["facts"]}
    cands = {c["id"]: c for c in rec["candidates"]}
    heads = {h["id"]: h for h in rec["headlines"]}
    entities: dict[str, dict[str, Any]] = {}
    rep = rec["report"]
    headline = rep["headline"]
    n_abs = b // length
    return {
        "chapter": chapter_number(b, length, n_origin),
        "n_abs": n_abs,
        "a": a,
        "b": b,
        "mode": 0,
        "final": True,
        "withheld": [],
        "headline": None
        if headline is None
        else {
            "id": headline["id"],
            "text": headline["text"],
            "refs": _refs(
                sheet, by_id, heads[headline["id"]]["template"], heads[headline["id"]]["cites"], entities
            ),
        },
        "lead": None if rep["lead"] is None else {"id": "lead", "text": rep["lead"], "refs": []},
        "body": [
            {
                "id": x["id"],
                "text": x["text"],
                "origin": x["origin"],
                "refs": _refs(sheet, by_id, cands[x["id"]]["template"], cands[x["id"]]["cites"], entities),
            }
            for x in rep["body"]
        ],
        "interventions": _intervention_sentences(rep["interventions"]),
        "ticker": [{"step": int(t["step"]), "text": t["text"]} for t in rec["ticker"]],
        "since": dict(since),
        "entities": entities,
        "versions": _versions(),
        "facts_digest": rec["facts_digest"],
        "latency_s": latency_s,
        "fallback": None,
    }


def withheld_object(
    a: int, b: int, *, n_origin: int, length: int, reason: str, since: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """The ``RPRT`` object of a chapter that cannot be cut from the files (a row missing, an unobserved run):
    empty, with the reason; never a guess."""
    n_abs = b // length
    return {
        "chapter": chapter_number(b, length, n_origin),
        "n_abs": n_abs,
        "a": a,
        "b": b,
        "mode": 0,
        "final": True,
        "withheld": [reason[:200]],
        "headline": None,
        "lead": None,
        "body": [],
        "interventions": [],
        "ticker": [],
        "since": dict(since) if since is not None else {"influence": None, "alerts": {}, "ending": []},
        "entities": {},
        "versions": _versions(),
        "facts_digest": "",
        "latency_s": None,
        "fallback": None,
    }


def missing_chapters(
    reports: Sequence[Mapping[str, Any]], t: int, start_step: int, length: int, grid: int
) -> list[int]:
    """The chapter ends ``b <= t`` (at most the last ``BACKLOG_REPORTS``) a world should have a report for and
    does not: the chapter must lie inside the world (``b - length >= start_step``: the partial chapter of a
    start off the chapter grid is an arrival chapter, with no report) and be cut on the snapshot grid."""
    if length % grid:
        return []
    ends = [b for b in range(length, t + 1, length) if b - length >= start_step]
    have = {int(r["step"]) for r in reports}
    return [b for b in ends[-BACKLOG_REPORTS:] if b not in have]


def stored_objects(path: Path, *, length: int, n_origin: int) -> list[tuple[int, dict[str, Any]]]:
    """The ``RPRT`` objects of the last ``BACKLOG_REPORTS`` records of ``reports.jsonl`` (what a world being
    opened is sent), in chapter order: ``(step, object)``. The ``since`` strip is the same function of the
    files as it was when the report was made."""
    from alife.director import log as director_log

    rd = RunDir.open(path)
    entries = director_log.entries_of(rd.read_jsonl(DIRECTOR_LOG_FILE))
    alert_lines = rd.read_jsonl(ALERTS_FILE)
    out = []
    for rec in rd.reports()[-BACKLOG_REPORTS:]:
        a, b = int(rec["chapter_start"]), int(rec["step"])
        since = since_of(entries, alert_lines, a, b, length)
        out.append((b, report_object(rec, n_origin=n_origin, length=length, since=since)))
    return out


# --- the job ------------------------------------------------------------------------------------------------


class ReportJob:
    """The chapter reports of one played world: ``start`` queues a chapter, one worker builds them in order
    (one at a time, a second waits), ``close`` lets it finish and joins it. ``emit(step, object, backlog)``
    hands the ``RPRT`` to the stream (from the worker thread, through the pacer's write lock); ``note``
    reports what went wrong without stopping the play (a report that cannot be built is a withheld ``RPRT``,
    never an exception in the loop)."""

    def __init__(
        self,
        path: Path,
        *,
        length: int,
        n_origin: int,
        emit: Emit,
        note: Callable[[str], None],
        rules: report.ReportRules = report.REPORT_V1,
        clock: Callable[[], float] = time.perf_counter,
        start_step: int = 0,
    ) -> None:
        self.path, self.length, self.n_origin = Path(path), length, n_origin
        self.start_step = start_step  # a chapter that began before it is an arrival chapter: no report
        self._emit, self._note, self._rules, self._clock = emit, note, rules, clock
        self._queue: queue.Queue[tuple[int, Mapping[str, Any] | None, bool]] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._closed = False
        self._abort = False
        self.built: list[int] = []  # the chapter ends whose report this job built (for the tests)

    # --- the loop thread --------------------------------------------------------------------------------

    def start(self, b: int, sidecar_meta: Mapping[str, Any] | None = None, *, backlog: bool = False) -> None:
        """Queue the report of the chapter ending at ``b`` (``sidecar_meta``: the ``meta`` the observer's
        sidecar of ``b`` will hold, taken on the loop thread). Cheap: it starts a worker if none runs. The
        first chapter of a world that started off the chapter grid began before the world did, so it has no
        rows to cut a report from: it gets a notice in words, written here, and no job."""
        if b - self.length < self.start_step:
            obj = withheld_object(
                b - self.length, b, n_origin=self.n_origin, length=self.length, reason=ARRIVAL_REASON
            )
            try:
                self._emit(b, obj, backlog)
            except Exception as e:  # the stream is gone or refuses it: nothing more to say
                self._note(f"the notice of the chapter ending at step {b} failed: {type(e).__name__}: {e}")
            return
        with self._lock:
            if self._closed:
                return
            self._queue.put((b, sidecar_meta, backlog))
            if self._thread is None:
                self._thread = threading.Thread(target=self._work, name="alife-report", daemon=False)
                self._thread.start()

    def close(self, timeout: float = CLOSE_TIMEOUT_S) -> bool:
        """Let the worker finish the chapters queued and join it. False if it did not finish within
        ``timeout`` seconds: it then stops before the next chapter (a resume rebuilds what is missing)."""
        with self._lock:
            self._closed = True
            thread = self._thread
        if thread is None:
            return True
        thread.join(timeout)
        if thread.is_alive():
            self._abort = True
            return False
        return True

    # --- the worker -------------------------------------------------------------------------------------

    def _work(self) -> None:
        """Build the queued chapters in order; leave when there is none (a thread that waited for more would
        keep the process alive: it is not a daemon). ``start`` makes a new one."""
        while True:
            with self._lock:
                if self._abort or self._queue.empty():
                    self._thread = None
                    return
                b, meta, backlog = self._queue.get_nowait()
            try:
                self._build(b, meta, backlog)
            except Exception as e:  # a report that cannot be built never stops the play
                self._note(f"the report of the chapter ending at step {b} failed: {type(e).__name__}: {e}")
                try:
                    reason = f"the report could not be built ({type(e).__name__})"
                    obj = withheld_object(
                        b - self.length, b, n_origin=self.n_origin, length=self.length, reason=reason
                    )
                    self._emit(b, obj, backlog)
                except Exception:  # the stream is gone or refuses it: nothing more to say
                    pass

    def _build(self, b: int, meta: Mapping[str, Any] | None, backlog: bool) -> None:
        t0 = self._clock()
        a = b - self.length
        sidecars = {} if meta is None else {b: meta}
        rd_files = RunDir.open(self.path)
        try:
            try:
                view = report.load_run(self.path, tail_from=a, sidecar_meta=sidecars)
                sheet = report.build_sheet(view, a, b, self._rules)
            except report.TailError:  # the saved kin thread table is gone: the whole log
                view = report.load_run(self.path, sidecar_meta=sidecars)
                sheet = report.build_sheet(view, a, b, self._rules)
        except report.ReportError as e:
            reason = withheld_object(a, b, n_origin=self.n_origin, length=self.length, reason=str(e))
            self._emit(b, reason, backlog)
            return
        rec = report.narrate(sheet, None, rules=self._rules)
        rd_files.put_report(rec)
        since = since_of(view.entries, rd_files.read_jsonl(ALERTS_FILE), a, b, self.length)
        obj = report_object(
            rec,
            n_origin=self.n_origin,
            length=self.length,
            since=since,
            latency_s=round(self._clock() - t0, 4),
        )
        self.built.append(b)
        self._emit(b, obj, backlog)
