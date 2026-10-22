"""The live World Report (docs/DESIGN.md §34.4; G3b spec §5.1, §7.3): the report job of a played world, the
``RPRT`` it sends, ``reports.jsonl``, and that the live record is the offline record. A tiny world with a chapter
of 500 steps (a whole number of social snapshots) on the CPU; the job's worker is joined by ``PlayPacer.close``
and every clock is injected or absent. CPU only, no socket."""

# ruff: noqa: E501  (expected report texts are long lines)

from __future__ import annotations

import io
import json
import random
import shutil
import threading
from pathlib import Path
from typing import Any

import pytest

import play_helpers as H
import report_synth as RS
from alife import play, runner, streamrec
from alife.analysis import world_report as W
from alife.game import report as R
from alife.game import reportjob
from alife.runio import REPORTS_FILE, RunDir, load_checkpoint, read_jsonl_from
from alife.state import init_state

CHAPTER = 500


def options(**kw: Any) -> play.PlayOptions:
    return play.PlayOptions(
        rate=float("inf"), paused=False, fresh=True, unattended=True, chapter=CHAPTER, **kw
    )


def config():
    return H.world("2d", "run.metrics_every=250", "run.heavy_metrics_every=250", "run.checkpoint_every=500")


def play_world(path: Path, steps: int, *, origin: bool = True) -> tuple[RunDir, play.PlayPacer, io.BytesIO]:
    """Play a tiny world for ``steps`` steps (3 chapters of 500 by default), as ``alife play`` would, with the
    origin row of a fresh world."""
    cfg = config()
    rd = RunDir.create(path, cfg)
    state = init_state(cfg)
    if origin:
        rd.update_meta(origin_row=play.origin_row(state, cfg))
    out = io.BytesIO()
    pacer = play.PlayPacer(out, H.ScriptControl([]), options(), clock=H.FakeClock(), note=lambda m: None)
    real = RunDir.save_checkpoint

    def slow_save(self: RunDir, *a: Any, **k: Any) -> Any:
        """A checkpoint that takes as long as the report: the stall of a real world is 0.9 s and the job 0.03 s,
        so the report is on the stream before the save ends (the order the tests check)."""
        thread = None if pacer.reports is None else pacer.reports._thread
        if thread is not None:
            thread.join(60)
        return real(self, *a, **k)

    RunDir.save_checkpoint = slow_save  # type: ignore[method-assign]
    try:
        runner.run(rd, cfg, state, steps, quiet=True, pacer=pacer)
    finally:
        RunDir.save_checkpoint = real  # type: ignore[method-assign]
        pacer.close()
    return rd, pacer, out


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> tuple[RunDir, play.PlayPacer, io.BytesIO]:
    return play_world(tmp_path_factory.mktemp("live") / "w", 1500)


def reprs(out: io.BytesIO) -> list[Any]:
    return [r for r in H.decoded(out).records if r.tag == "RPRT"]


def obj_of(rec: Any) -> dict[str, Any]:
    return rec.f["json"]


# --- the live record is the offline record -----------------------------------------------------------------------


def test_every_chapter_of_a_played_world_has_a_report_record_that_verifies(world) -> None:
    rd, pacer, _ = world
    recs = rd.reports()
    assert [(r["chapter_start"], r["step"]) for r in recs] == [(0, 500), (500, 1000), (1000, 1500)]
    assert pacer.reports is not None and pacer.reports.built == [500, 1000, 1500]
    code, summary = W.verify_reports(R.load_run(rd.path), recs)
    assert code == 0 and summary["counts"] == {"ok": 3}, summary
    assert all(r["mode"] == "template" and r["excluded_candidates"] == [] for r in recs)


def test_a_build_of_the_same_chapters_writes_the_same_records(world, tmp_path: Path) -> None:
    rd, _, _ = world
    out = tmp_path / "offline.jsonl"
    assert (
        W.main(
            [
                "build",
                str(rd.path),
                "--from",
                "0",
                "--to",
                "1500",
                "--chapter",
                str(CHAPTER),
                "--out",
                str(out),
            ]
        )
        == 0
    )
    assert [json.loads(line) for line in out.read_text().splitlines()] == rd.reports()
    assert (
        out.read_bytes() == (rd.path / REPORTS_FILE).read_bytes()
    )  # byte for byte: the file is the offline file


def test_the_first_chapter_is_reported_from_the_origin_row(world) -> None:
    rd, _, _ = world
    first = rd.reports()[0]
    assert first["facts"]["excluded"]["facts_dropped"] == ["no social baseline at the origin"]
    assert rd.meta()["origin_row"]["step"] == 0 and first["report"]["lead"].startswith("The population")


def test_the_report_of_a_world_with_no_origin_row_withholds_its_first_chapter(tmp_path: Path) -> None:
    rd, _, out = play_world(tmp_path / "w", 1000, origin=False)
    assert [(r["chapter_start"], r["step"]) for r in rd.reports()] == [(500, 1000)]
    first, second = reprs(out)
    obj = obj_of(first)
    assert (obj["a"], obj["b"], obj["chapter"]) == (0, 500, 1)
    assert obj["withheld"] == ["the cut 0 is not the step of a metrics row"]
    assert obj["headline"] is None and obj["lead"] is None and obj["body"] == [] and obj["facts_digest"] == ""
    assert obj_of(second)["withheld"] == [] and obj_of(second)["headline"] is not None
    streamrec.check_report(obj)  # a withheld object is a valid record


# --- the stream --------------------------------------------------------------------------------------------------


def test_the_rprt_follows_the_clock_that_opens_the_card_and_precedes_the_one_that_closes_it(world) -> None:
    _, _, out = world
    dec = H.decoded(out)
    assert "RPRT" in dec.header["play"]["records"]
    idx = [
        (i, r) for i, r in enumerate(dec.records) if r.tag in ("RPRT", "CLCK") and r.step in (500, 1000, 1500)
    ]
    for b in (500, 1000, 1500):
        at = [(i, r) for i, r in idx if r.step == b]
        kinds = [(r.tag, r.f.get("saving") if r.tag == "CLCK" else None) for _, r in at]
        # CLCK saving 1 (reason 3), RPRT, CLCK saving 0: the card is there when it is filled
        assert kinds == [("CLCK", 1), ("RPRT", None), ("CLCK", 0)], (b, kinds)
        assert at[0][1].f["reason"] == 3
    reports = reprs(out)
    assert [obj_of(r)["chapter"] for r in reports] == [1, 2, 3]
    for r in reports:
        assert r.flags & streamrec.FLAG_FINAL and not r.flags & streamrec.FLAG_BACKLOG
        obj = obj_of(r)
        assert obj["mode"] == 0 and obj["final"] is True and obj["n_abs"] == obj["b"] // CHAPTER
        assert obj["versions"] == {"report": 2, "template": 2, "checker": 1, "lexicon": 3}
        assert obj["latency_s"] is not None and obj["latency_s"] >= 0
        assert obj["since"]["influence"] == {"have": 6, "cap": 6} and obj["since"]["ending"] == []


def test_the_object_of_a_record_is_its_report(world) -> None:
    rd, _, out = world
    by_b = {obj_of(r)["b"]: obj_of(r) for r in reprs(out)}
    for rec in rd.reports():
        obj = by_b[rec["step"]]
        rep = rec["report"]
        assert obj["headline"]["text"] == rep["headline"]["text"] and obj["lead"]["text"] == rep["lead"]
        assert [x["text"] for x in obj["body"]] == [x["text"] for x in rep["body"]]
        assert [x["id"] for x in obj["body"]] == [x["id"] for x in rep["body"]]
        assert [x["text"] for x in obj["interventions"]] == [x["text"] for x in rep["interventions"]]
        assert obj["ticker"] == [{"step": t["step"], "text": t["text"]} for t in rec["ticker"]]
        assert obj["facts_digest"] == rec["facts_digest"]
        # a ref is the key of an entity the object holds, with the kin group's name and place
        for x in [obj["headline"], *obj["body"]]:
            assert set(x["refs"]) <= set(obj["entities"])
        for key, ent in obj["entities"].items():
            assert key[0] in "KSH" and ent["kind"] in ("kin", "storm", "house") and ent["id"] == int(key[1:])


def test_the_stored_reports_are_the_backlog_of_a_world_being_opened(world, tmp_path: Path) -> None:
    rd, _, _ = world
    run = tmp_path / "reopen"
    shutil.copytree(rd.path, run)
    cfg = config()
    rd2 = RunDir.open(run)
    state = load_checkpoint(rd2.latest_checkpoint(), cfg)  # type: ignore[arg-type]
    out = io.BytesIO()
    pacer = play.PlayPacer(out, H.ScriptControl([]), options(), clock=H.FakeClock(), note=lambda m: None)
    try:
        runner.run(rd2, cfg, state, 0, quiet=True, pacer=pacer)
    finally:
        pacer.close()
    got = reprs(out)
    assert [obj_of(r)["b"] for r in got] == [500, 1000, 1500] and all(
        r.flags & streamrec.FLAG_BACKLOG for r in got
    )
    assert pacer.reports is not None and pacer.reports.built == []  # nothing was missing: nothing rebuilt


def test_a_report_the_worker_did_not_write_is_rebuilt_when_the_world_is_opened(world, tmp_path: Path) -> None:
    rd, _, _ = world
    run = tmp_path / "reopen"
    shutil.copytree(rd.path, run)
    rd2 = RunDir.open(run)
    recs = rd2.reports()
    (run / REPORTS_FILE).write_text(
        "".join(json.dumps(r) + "\n" for r in recs[:2])
    )  # the last one is missing
    cfg = config()
    state = load_checkpoint(rd2.latest_checkpoint(), cfg)  # type: ignore[arg-type]
    out = io.BytesIO()
    pacer = play.PlayPacer(out, H.ScriptControl([]), options(), clock=H.FakeClock(), note=lambda m: None)
    try:
        runner.run(rd2, cfg, state, 0, quiet=True, pacer=pacer)
    finally:
        pacer.close()
    assert pacer.reports is not None and pacer.reports.built == [1500]
    assert rd2.reports() == recs  # the same record, byte for byte (a pure function of the files)
    got = reprs(out)
    assert sorted(obj_of(r)["b"] for r in got) == [500, 1000, 1500]
    assert all(r.flags & streamrec.FLAG_BACKLOG for r in got)


def test_the_header_lists_rprt_only_for_a_world_whose_chapter_is_a_whole_number_of_snapshots() -> None:
    cfg = H.world("2d")
    out = io.BytesIO()
    # a chapter of 30 steps (the other tests') is no multiple of the 250-step snapshots a report is cut at
    pacer = play.PlayPacer(
        out,
        H.ScriptControl([]),
        play.PlayOptions(rate=float("inf"), paused=False, fresh=True, unattended=True, chapter=30),
        clock=H.FakeClock(),
        note=lambda m: None,
    )
    assert pacer._reports_wanted() is False and pacer.reports is None
    assert cfg.game.observe


# --- the tail read and the saved thread table --------------------------------------------------------------------


def test_the_live_job_reads_from_the_end_of_the_logs(
    world, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rd, _, _ = world
    run = tmp_path / "again"
    shutil.copytree(rd.path, run)
    calls: list[dict[str, Any]] = []
    real = R.load_run

    def spy(path: Any, **kw: Any) -> Any:
        calls.append(kw)
        return real(path, **kw)

    monkeypatch.setattr(R, "load_run", spy)
    emitted: list[tuple[int, dict[str, Any], bool]] = []
    job = reportjob.ReportJob(
        run, length=CHAPTER, n_origin=0, emit=lambda *a: emitted.append(a), note=lambda m: None
    )
    (run / REPORTS_FILE).unlink()
    job.start(1000, None)
    assert job.close()
    assert [c.get("tail_from") for c in calls] == [
        500
    ]  # the saved table of 500 was there: no second, full read
    assert RunDir.open(run).reports()[0]["step"] == 1000 and emitted[0][1]["b"] == 1000


def test_without_the_saved_table_the_job_reads_the_whole_log_and_makes_the_same_record(
    world, tmp_path: Path
) -> None:
    rd, _, _ = world
    run = tmp_path / "again"
    shutil.copytree(rd.path, run)
    for p in (run / "observers").glob("kinthreads_*.json"):
        p.unlink()
    want = rd.reports()[2]
    (run / REPORTS_FILE).unlink()
    job = reportjob.ReportJob(run, length=CHAPTER, n_origin=0, emit=lambda *a: None, note=lambda m: None)
    job.start(1500, None)
    assert job.close()
    assert RunDir.open(run).reports() == [want]


def test_the_sidecar_meta_is_handed_over_while_its_file_is_still_to_be_written(world, tmp_path: Path) -> None:
    """At the chapter end the sidecar of ``b`` is not on disk yet; a job that read the files would take the one
    before. With ``meta`` in memory the sheet holds what the file will say."""
    rd, _, _ = world
    run = tmp_path / "again"
    shutil.copytree(rd.path, run)
    (run / "observers" / "step_0000001500.npz").unlink()  # not written yet

    def digest(meta: dict[str, Any] | None) -> str | None:
        (run / REPORTS_FILE).unlink(missing_ok=True)
        job = reportjob.ReportJob(run, length=CHAPTER, n_origin=0, emit=lambda *a: None, note=lambda m: None)
        job.start(1500, meta)
        assert job.close()
        return RunDir.open(run).reports()[0]["facts"]["labels"]["rules_digest"]

    from_files = digest(None)  # the sidecar of 1,250 is the latest at or before 1,500
    mine = {**RS.sidecar_meta(), "rules_digest": "ab" * 32}
    assert digest(mine) == "ab" * 32 and from_files != "ab" * 32


def test_the_pacer_hands_the_observers_meta_to_the_job_and_serve_play_closes_the_pacer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple[int, Any, bool]] = []
    real_start = reportjob.ReportJob.start

    def spy(self: reportjob.ReportJob, b: int, sidecar_meta: Any = None, *, backlog: bool = False) -> None:
        seen.append((b, sidecar_meta, backlog))
        real_start(self, b, sidecar_meta, backlog=backlog)

    closed: list[int] = []
    real_close = play.PlayPacer.close

    def close(self: play.PlayPacer) -> None:
        closed.append(1)
        real_close(self)

    monkeypatch.setattr(reportjob.ReportJob, "start", spy)
    monkeypatch.setattr(play.PlayPacer, "close", close)
    cfg = config()
    rd = RunDir.create(tmp_path / "w", cfg)
    state = init_state(cfg)
    rd.update_meta(origin_row=play.origin_row(state, cfg))
    code = play.serve_play(
        rd, cfg, [state], steps=500, options=options(), control=H.ScriptControl([]), out=io.BytesIO(),
        clock=H.FakeClock(),
    )  # fmt: skip
    assert code == 0 and closed == [1]
    ((b, meta, backlog),) = seen
    assert (b, backlog) == (500, False) and meta is not None and meta["step"] == 500


def test_the_chapter_number_counts_from_the_chapter_the_planet_was_made_in(tmp_path: Path) -> None:
    cfg = config()
    rd = RunDir.create(tmp_path / "w", cfg)
    state = init_state(cfg)
    rd.update_meta(origin_row=play.origin_row(state, cfg))
    out = io.BytesIO()
    opt = play.PlayOptions(
        rate=float("inf"), paused=False, fresh=True, unattended=True, chapter=CHAPTER,
        planet={"made_at_step": 500},
    )  # fmt: skip
    pacer = play.PlayPacer(out, H.ScriptControl([]), opt, clock=H.FakeClock(), note=lambda m: None)
    try:
        runner.run(rd, cfg, state, 1000, quiet=True, pacer=pacer)
    finally:
        pacer.close()
    assert [(obj_of(r)["n_abs"], obj_of(r)["chapter"]) for r in reprs(out)] == [(1, 0), (2, 1)]


def test_a_world_being_opened_is_sent_only_the_last_reports(world, monkeypatch: pytest.MonkeyPatch) -> None:
    rd, _, _ = world
    got = reportjob.stored_objects(rd.path, length=CHAPTER, n_origin=0)
    assert [b for b, _ in got] == [500, 1000, 1500]
    monkeypatch.setattr(reportjob, "BACKLOG_REPORTS", 2)
    assert [b for b, _ in reportjob.stored_objects(rd.path, length=CHAPTER, n_origin=0)] == [1000, 1500]


def test_a_closed_job_takes_no_more_chapters(world, tmp_path: Path) -> None:
    rd, _, _ = world
    run = tmp_path / "closed"
    shutil.copytree(rd.path, run)
    sent: list[int] = []
    job = reportjob.ReportJob(
        run, length=CHAPTER, n_origin=0, emit=lambda b, *a: sent.append(b), note=lambda m: None
    )
    assert job.close()
    job.start(1500, None)
    assert job._thread is None and job._queue.empty() and sent == [] and job.built == []


# --- the job: failures, the order, the shutdown ------------------------------------------------------------------


def test_a_report_that_fails_is_a_withheld_object_and_a_note_never_an_exception(
    world, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rd, _, _ = world
    run = tmp_path / "again"
    shutil.copytree(rd.path, run)
    notes: list[str] = []
    emitted: list[tuple[int, dict[str, Any], bool]] = []

    def boom(*a: Any, **k: Any) -> Any:
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(R, "narrate", boom)
    job = reportjob.ReportJob(
        run, length=CHAPTER, n_origin=0, emit=lambda *a: emitted.append(a), note=notes.append
    )
    job.start(1000, None)
    assert job.close()
    assert len(emitted) == 1 and emitted[0][1]["withheld"] == ["the report could not be built (RuntimeError)"]
    assert notes and "disk on fire" in notes[0]
    streamrec.check_report(emitted[0][1])


def test_the_worker_builds_in_order_one_at_a_time_and_close_waits_for_what_is_queued(
    world, tmp_path: Path
) -> None:
    rd, _, _ = world
    run = tmp_path / "again"
    shutil.copytree(rd.path, run)
    (run / REPORTS_FILE).unlink()
    order: list[int] = []
    active = []
    peak = [0]

    def emit(step: int, obj: dict[str, Any], backlog: bool) -> None:
        active.append(1)
        peak[0] = max(peak[0], len(active))
        order.append(step)
        active.pop()

    job = reportjob.ReportJob(run, length=CHAPTER, n_origin=0, emit=emit, note=lambda m: None)
    for b in (1500, 500, 1000):
        job.start(b, None)
    assert job.close() is True
    assert order == [1500, 500, 1000] and peak[0] == 1  # in the order they were asked for
    assert [r["step"] for r in RunDir.open(run).reports()] == [
        500,
        1000,
        1500,
    ]  # the file is in chapter order
    job.start(1500, None)  # a closed job takes nothing
    assert job.built == [1500, 500, 1000]


def test_close_gives_up_after_its_timeout_and_the_worker_stops_before_the_next_chapter(
    world, tmp_path: Path
) -> None:
    rd, _, _ = world
    run = tmp_path / "again"
    shutil.copytree(rd.path, run)
    gate = threading.Event()
    entered = threading.Event()
    seen: list[int] = []

    def slow(step: int, obj: dict[str, Any], backlog: bool) -> None:
        seen.append(step)
        entered.set()
        gate.wait(10)

    job = reportjob.ReportJob(run, length=CHAPTER, n_origin=0, emit=slow, note=lambda m: None)
    job.start(500, None)
    entered.wait(10)
    job.start(1000, None)
    assert job.close(timeout=0.05) is False  # the first is still being written
    gate.set()
    thread = job._thread
    if thread is not None:
        thread.join(10)
    assert seen == [500]  # the second was dropped: a resume rebuilds it (missing_chapters)
    assert not (thread and thread.is_alive())


def test_an_idle_worker_leaves_so_that_it_never_keeps_a_process_alive(world, tmp_path: Path) -> None:
    rd, _, _ = world
    run = tmp_path / "again"
    shutil.copytree(rd.path, run)
    job = reportjob.ReportJob(run, length=CHAPTER, n_origin=0, emit=lambda *a: None, note=lambda m: None)
    job.start(500, None)
    thread = job._thread
    assert thread is not None
    thread.join(30)
    assert not thread.is_alive() and job._thread is None
    job.start(1000, None)  # and a new one starts when there is work
    assert job._thread is not None
    assert job.close()


# --- the strips and the helpers ----------------------------------------------------------------------------------


def power(step: int, cost: int = 1) -> dict[str, Any]:
    return {"step_applied": step, "source": "player", "action": "regen_pulse", "params": {"mult": 0.5},
            "duration": 3000, "prev": {"mult": 1.0}, "cost": cost, "power": "food", "specs": "game-v0", "scope": "player"}  # fmt: skip


def test_since_has_the_influence_the_alerts_and_the_effects_that_end_soon() -> None:
    a, b = 1_000_000, 1_006_000
    rule = {"step_applied": 1_002_000, "source": "rule", "rule": "boredom", "action": "regen_pulse",
            "params": {"mult": 1.5}, "duration": 20_000, "prev": {"mult": 1.0}}  # fmt: skip
    # a player's power at 1,001,000 for 3,000 steps: ended at 1,004,000 by its revert; one at 1,005,000 is running
    entries = [
        power(1_001_000),
        {
            "step_applied": 1_004_000,
            "source": "revert",
            "scope": "player",
            "action": "regen_pulse",
            "params": {"mult": 1.0},
            "reverts": 1_001_000,
            "duration": None,
        },
        power(1_005_000, cost=2),
    ]
    alert_lines = [
        {"step": 1_000_500, "class": "storm_named", "state": "raised", "surface": "alert"},
        {"step": 1_000_750, "class": "storm_named", "state": "raised", "surface": "alert"},
        {"step": 1_001_000, "class": "storm_named", "state": "updated", "surface": "alert"},
        {"step": 1_001_250, "class": "lineage_formed", "state": "raised", "surface": "log"},
        {"step": 1_006_000, "class": "chapter", "state": "raised", "surface": "alert"},
        {"step": 999_000, "class": "pop_fall", "state": "raised", "surface": "alert"},
        {"step": 1_003_000, "class": "pop_fall", "state": "raised", "surface": "alert"},
    ]
    since = reportjob.since_of(entries, alert_lines, a, b, 6_000)
    assert since["influence"] == {"have": 6 - 1 - 2, "cap": 6}
    assert since["alerts"] == {
        "pop_fall": 1,
        "storm_named": 2,
    }  # raised, on the alert surface, in (a, b], not the chapter's
    assert since["ending"] == [
        {"action": "regen_pulse", "end": 1_008_000}
    ]  # the running power ends in the next chapter
    # a rule's effect that ends later than the next chapter is not listed; the cost of an old power frees
    far = reportjob.since_of([rule], [], a, b, 6_000)
    assert far["ending"] == [] and far["influence"]["have"] == 6
    old = reportjob.since_of([power(900_000)], [], a, b, 6_000)  # 106,000 steps ago: more than the window
    assert old["influence"]["have"] == 6
    within = reportjob.since_of(
        [power(990_000)], [], a, b, 6_000
    )  # 16,000 steps ago: in the window of 18,000
    assert within["influence"]["have"] == 5
    assert reportjob.since_of([], [], a, b, 6_000)["influence"] == {"have": 6, "cap": 6}
    # the edges: an effect that ends exactly at the end of the next chapter is listed (a rule's as well as the
    # player's); an alert raised at the first step of the chapter belongs to it, one at its last step to the
    # next: the strip is built before the alerts of the row at ``b`` exist, so the interval is ``[a, b)``
    edge = {**rule, "step_applied": 1_002_000, "duration": 10_000}
    assert reportjob.since_of([edge], [], a, b, 6_000)["ending"] == [
        {"action": "regen_pulse", "end": 1_012_000}
    ]
    late = {**edge, "duration": 10_001}
    assert reportjob.since_of([late], [], a, b, 6_000)["ending"] == []
    at_a = [{"step": a, "class": "pop_fall", "state": "raised", "surface": "alert"}]
    assert reportjob.since_of([], at_a, a, b, 6_000)["alerts"] == {"pop_fall": 1}
    at_b = [{"step": b, "class": "pop_fall", "state": "raised", "surface": "alert"}]
    assert reportjob.since_of([], at_b, a, b, 6_000)["alerts"] == {}
    # a player's record at ``b`` is after the checkpoint: its cost is not spent, its effect not held, and the
    # revert that ends an effect at ``b`` has not been written
    cast_at_b = power(b, cost=2)
    assert reportjob.since_of([cast_at_b], [], a, b, 6_000) == reportjob.since_of([], [], a, b, 6_000)
    ends_at_b = {**power(b - 3_000), "duration": 3_000}
    revert_at_b = {"step_applied": b, "source": "revert", "scope": "player", "action": "regen_pulse",
                   "params": {"mult": 1.0}, "reverts": b - 3_000, "duration": None}  # fmt: skip
    assert reportjob.since_of([ends_at_b, revert_at_b], [], a, b, 6_000) == reportjob.since_of(
        [ends_at_b], [], a, b, 6_000
    )
    assert reportjob.since_of([ends_at_b], [], a, b, 6_000)["ending"] == [{"action": "regen_pulse", "end": b}]


def test_missing_chapters_are_the_last_three_inside_the_world_on_the_snapshot_grid() -> None:
    have = [{"step": 12_000}]
    assert reportjob.missing_chapters([], 18_000, 0, 6_000, 250) == [6_000, 12_000, 18_000]
    assert reportjob.missing_chapters(have, 18_000, 0, 6_000, 250) == [6_000, 18_000]
    assert reportjob.missing_chapters([], 60_000, 0, 6_000, 250) == [
        48_000,
        54_000,
        60_000,
    ]  # at most the last three
    # a fork at 450,000: the chapter ending at 456,000 is its first; the one ending at 450,000 is the parent's
    assert reportjob.missing_chapters([], 456_000, 450_000, 6_000, 250) == [456_000]
    assert reportjob.missing_chapters([], 450_000, 450_000, 6_000, 250) == []
    # a start off the chapter grid: its partial chapter is an arrival chapter with no report
    assert reportjob.missing_chapters([], 462_000, 450_500, 6_000, 250) == [462_000]
    assert reportjob.missing_chapters([], 456_000, 450_500, 6_000, 250) == []
    # a chapter that is not a whole number of snapshots has no reports at all
    assert reportjob.missing_chapters([], 60, 0, 30, 250) == []


def test_the_object_of_the_p0_chapters_names_its_entities_with_their_places() -> None:
    view = R.load_run(Path(__file__).parent / "data" / "report_p0")
    sheet = R.build_sheet(view, 450250, 456250)
    rec = R.narrate(sheet, None)
    obj = reportjob.report_object(
        rec, n_origin=75, length=6000, since={"influence": {"have": 6, "cap": 6}, "alerts": {}, "ending": []}
    )
    streamrec.check_report(obj)
    assert (obj["chapter"], obj["n_abs"]) == (
        1,
        76,
    )  # 450,250 to 456,250 is the chapter 76 of the world, the 1st of the planet
    storm_line = next(x for x in obj["body"] if x["text"].startswith("Storm Phaitraern"))
    assert storm_line["refs"] == [
        next(k for k, e in obj["entities"].items() if e["kind"] == "storm" and e["name"] == "Phaitraern")
    ]
    ent = obj["entities"][storm_line["refs"][0]]
    assert ent["kind"] == "storm" and ent["x"] is not None and ent["y"] is not None
    kin_line = next(x for x in obj["body"] if x["text"].startswith("Pruashol merged"))
    (key,) = kin_line["refs"]
    assert (
        key.startswith("K")
        and obj["entities"][key]["name"] == "Pruashol"
        and obj["entities"][key]["x"] is not None
    )
    assert obj["headline"]["refs"] == []  # "3 storms reached named strength" names none of them
    # the lead names nobody; a group's sentence names its group
    assert obj["lead"]["refs"] == []
    big = reportjob.report_object(
        R.narrate(R.build_sheet(view, 462000, 468000), None), n_origin=75, length=6000, since=obj["since"]
    )
    assert any(x["refs"] for x in big["body"]) and big["interventions"][0]["refs"] == []
    streamrec.check_report(big)


def test_the_house_of_the_largest_house_sentence_is_an_entity_of_it() -> None:
    view = R.load_run(Path(__file__).parent / "data" / "report_p0")
    sheet = R.build_sheet(view, 450250, 456250)
    rec = R.narrate(sheet, None)
    cand = next(c for c in rec["candidates"] if c["template"] == "H-2")
    by_id = {f["id"]: f for f in sheet["facts"]}
    entities: dict[str, Any] = {}
    refs = reportjob._refs(sheet, by_id, "H-2", cand["cites"], entities)
    assert refs == ["H5165"] and entities == {
        "H5165": {"kind": "house", "id": 5165, "name": "Teshash", "x": None, "y": None}
    }
    # the same fact cited by another sentence names no house
    assert reportjob._refs(sheet, by_id, "H-1", cand["cites"], {}) == []


# --- the files ---------------------------------------------------------------------------------------------------


def test_read_jsonl_from_is_the_filter_of_the_whole_read(tmp_path: Path) -> None:
    rng = random.Random(7)
    for trial in range(60):
        steps = sorted(rng.sample(range(0, 5000), rng.randint(0, 40)))
        lines = [json.dumps({"step": s, "pad": "x" * rng.randint(0, 300)}) for s in steps]
        text = "\n".join(lines) + ("\n" if lines else "")
        if lines and rng.random() < 0.4:
            text += '{"step": 99999, "pa'  # a torn last line
        path = tmp_path / f"f{trial}.jsonl"
        path.write_text(text)
        for min_step in (0, rng.randint(0, 5000), 5001):
            for chunk in (16, 100, 1 << 20):
                want = (
                    [r for r in json.loads("[" + ",".join(lines) + "]") if r["step"] >= min_step]
                    if lines
                    else []
                )
                assert read_jsonl_from(path, min_step, chunk) == want, (trial, min_step, chunk)
    assert read_jsonl_from(tmp_path / "missing.jsonl", 0) == []


def test_read_jsonl_from_raises_for_a_corrupt_line_it_reads_and_never_sees_one_it_does_not(
    tmp_path: Path,
) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"step": 1}\n{"step": 2\n{"step": 3}\n')
    with pytest.raises(json.JSONDecodeError):
        read_jsonl_from(path, 0)
    with pytest.raises(json.JSONDecodeError):
        read_jsonl_from(path, 3)  # a corrupt line in what was read is an error, as in the whole read
    # a corrupt line far from the end is not read when the range is near it
    lines = ['{"step": 0'] + [json.dumps({"step": i, "pad": "x" * 100}) for i in range(1, 60)]
    far = tmp_path / "far.jsonl"
    far.write_text("\n".join(lines) + "\n")
    assert [r["step"] for r in read_jsonl_from(far, 50, 512)] == list(range(50, 60))


def test_put_report_replaces_sorts_and_keeps_the_other_lines(tmp_path: Path) -> None:
    cfg = config()
    rd = RunDir.create(tmp_path / "w", cfg)
    rec = lambda a, b, tag="x": {"chapter_start": a, "step": b, "tag": tag}  # noqa: E731
    path = rd.path / REPORTS_FILE
    rd.put_report(rec(500, 1000))
    rd.put_report(rec(0, 500))
    # lines that are not report records (a note, a step that is a boolean) are kept, after the reports
    path.write_text(path.read_text() + '{"note": "kept"}\n{"chapter_start": 0, "step": true}\n')
    rd.put_report(rec(500, 1000, "new"))  # replaces the record of the same chapter
    assert [(x["step"], x["tag"]) for x in rd.reports()] == [(500, "x"), (1000, "new")]
    rd.put_report(rec(1000, 1500))
    lines = [json.loads(x) for x in path.read_text().splitlines()]
    assert [(x.get("chapter_start"), x.get("step"), x.get("tag")) for x in lines] == [
        (0, 500, "x"), (500, 1000, "new"), (1000, 1500, "x"), (None, None, None), (0, True, None),
    ]  # fmt: skip
    assert lines[3] == {"note": "kept"} and [r["step"] for r in rd.reports()] == [500, 1000, 1500]
    assert not list(rd.path.glob(".*"))  # no temporary file is left


def test_truncate_after_cuts_the_reports_of_the_chapters_that_ended_after_the_step(
    world, tmp_path: Path
) -> None:
    rd, _, _ = world
    run = tmp_path / "cut"
    shutil.copytree(rd.path, run)
    rd2 = RunDir.open(run)
    rd2.truncate_after(1000)
    assert [r["step"] for r in rd2.reports()] == [500, 1000]
    rd2.truncate_after(750, keep_director=True)  # a replay cuts them as well: it regenerates them
    assert [r["step"] for r in rd2.reports()] == [500]
    # the report of a chapter that ended at the step is in the checkpoint at the step: kept
    run2 = tmp_path / "cut2"
    shutil.copytree(rd.path, run2)
    RunDir.open(run2).truncate_after(1500)
    assert [r["step"] for r in RunDir.open(run2).reports()] == [500, 1000, 1500]


def test_a_world_without_observers_has_no_report_job(tmp_path: Path) -> None:
    cfg = H.world(
        "2d",
        "run.metrics_every=250",
        "run.heavy_metrics_every=250",
        "run.checkpoint_every=250",
        observe=False,
    )
    rd = RunDir.create(tmp_path / "w", cfg)
    out = io.BytesIO()
    pacer = play.PlayPacer(out, H.ScriptControl([]), options(), clock=H.FakeClock(), note=lambda m: None)
    try:
        runner.run(rd, cfg, init_state(cfg), 500, quiet=True, pacer=pacer)
    finally:
        pacer.close()
    assert pacer.reports is None and not (rd.path / REPORTS_FILE).exists() and not reprs(out)
    assert "RPRT" not in H.decoded(out).header["play"]["records"]
