"""Chapters, the pause at their end and ``autopause`` (DESIGN §34.2), and the write lock (§34.2.4).

A chapter is the steps ``(L (n - 1), L n]``; the tests shorten ``L`` to 30 (``PlayOptions.chapter``, a
multiple of the checkpoint cadence of ``play_helpers.CADENCE``). The pause starts at the first ``before_step``
after the chapter's checkpoint, so no step is owed and the stall of the save is read, not waited for;
presentation never reaches the trajectory (the digest at every checkpoint equals a plain run's)."""

from __future__ import annotations

import io
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from alife import play, runio, runner, streamrec
from alife.play import Autopause, AutopauseLine, ChapterClock
from alife.runio import RunDir, load_checkpoint
from alife.state import init_state, state_digest
from play_fixtures import _report
from play_helpers import (
    FakeClock,
    ScriptControl,
    TimedControl,
    checkpoint_digests,
    decoded,
    rows,
    run_paced,
    run_plain,
    same_events,
    snapshots,
    social_text,
    world,
)

L = 30  # the chapter length of these worlds


def options(**kw: Any) -> play.PlayOptions:
    return play.PlayOptions(**{"rate": float("inf"), "paused": False, "fresh": True, "chapter": L, **kw})


def clcks(out: io.BytesIO) -> list[dict[str, Any]]:
    return [{"step": r.step, **r.f} for r in decoded(out).records if r.tag == "CLCK"]


# --- the clock ------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("t", "chapter", "in_chapter"),
    [
        (0, 1, 0),  # the start of a world: chapter 1, nothing of it done
        (1, 1, 1),
        (5999, 1, 5999),
        (6000, 1, 6000),  # a boundary belongs to the chapter it ends: the bar reads 100%
        (6001, 2, 1),
        (12000, 2, 6000),
        (450_000, 75, 6000),  # a fork at 450,000 starts at the end of absolute chapter 75 ...
        (450_001, 76, 1),  # ... and its first step is the first of 76 (Chapter 1 of the fork)
        (456_000, 76, 6000),
    ],
)
def test_the_clock_fields_have_the_one_based_meaning(t: int, chapter: int, in_chapter: int) -> None:
    clock = ChapterClock(6000)
    assert clock.position(t) == (chapter, in_chapter) and clock.chapter(t) == chapter
    assert clock.is_end(t) == (t > 0 and in_chapter == 6000)


def test_a_chapter_ends_on_a_multiple_of_its_length_and_never_at_step_zero() -> None:
    clock = ChapterClock(30)
    assert [t for t in range(0, 95) if clock.is_end(t)] == [30, 60, 90]
    assert not ChapterClock().is_end(0) and ChapterClock().length == play.CHAPTER == 6000


def test_clck_carries_the_chapter_fields_flags_and_length(tmp_path: Path) -> None:
    _, _, _, out = run_paced(world("2d"), tmp_path / "w", 45, [["autopause off"], ["run"]], options=options())
    got = clcks(out)
    assert {c["chapter_len"] for c in got} == {L}
    pos = {c["step"]: (c["chapter"], c["in_chapter"]) for c in got}
    assert pos[0] == (1, 0) and pos[30] == (1, 30) and pos[45] == (2, 15)
    assert all(
        (c["chapter"], c["in_chapter"])
        == (1 + max(0, c["step"] - 1) // L, c["step"] - L * (max(0, c["step"] - 1) // L))
        for c in got
    )


def test_the_header_carries_the_chapter_length_and_the_autopause_defaults(tmp_path: Path) -> None:
    _, _, _, out = run_paced(world("2d"), tmp_path / "w", 3, [[]], options=options())
    play_block = decoded(out).header["play"]
    assert play_block["chapter"] == L
    assert play_block["autopause"] == {"chapter": True, "alert": {}}
    _, _, _, out6k = run_paced(world("2d"), tmp_path / "w6", 3, [[]], options=options(chapter=play.CHAPTER))
    assert decoded(out6k).header["play"]["chapter"] == 6000


# --- autopause ------------------------------------------------------------------------------------------


def test_autopause_sets_the_chapter_pause_and_the_alert_classes() -> None:
    a = Autopause(alert={"pop_fall": True, "storm_named": False})
    assert a.flags == streamrec.CLCK_FLAG_CHAPTER_PAUSE | streamrec.CLCK_FLAG_ALERT_PAUSE
    assert a.apply(AutopauseLine("chapter", None, False)) is None
    assert a.flags == streamrec.CLCK_FLAG_ALERT_PAUSE and a.alert == {"pop_fall": True, "storm_named": False}
    assert a.apply(AutopauseLine("alert", "pop_fall", False)) is None
    assert a.flags == 0
    assert a.apply(AutopauseLine("alert", "storm_named", True)) is None
    assert a.flags == streamrec.CLCK_FLAG_ALERT_PAUSE and a.alert["storm_named"] is True
    assert a.apply(AutopauseLine("all", None, True)) is None
    assert a.chapter and a.alert == {"pop_fall": True, "storm_named": True}
    assert a.apply(AutopauseLine("all", None, False)) is None
    assert not a.chapter and a.alert == {"pop_fall": False, "storm_named": False} and a.flags == 0
    assert a.block() == {"chapter": False, "alert": {"pop_fall": False, "storm_named": False}}


def test_an_unknown_alert_class_is_noted_and_changes_nothing() -> None:
    a = Autopause(alert={"pop_fall": True})
    note = a.apply(AutopauseLine("alert", "no_such", True))
    assert note is not None and "no_such" in note
    assert a.alert == {"pop_fall": True} and a.chapter


def test_autopause_lines_reach_the_flags_of_the_next_clock(tmp_path: Path) -> None:
    notes: list[str] = []
    clock = FakeClock()
    cfg = world("2d")
    rd = RunDir.create(tmp_path / "w", cfg)
    out = io.BytesIO()
    script = [
        [],
        ["autopause chapter off"],
        ["autopause chapter on"],
        ["autopause alert nope on"],
        ["autopause off"],
    ]
    pacer = play.PlayPacer(out, ScriptControl(script), options(), clock=clock, note=notes.append)
    pacer.autopause.alert = {"pop_fall": True}  # (the alert engine's classes arrive with its stage)
    runner.run(rd, cfg, init_state(cfg), 8, quiet=True, pacer=pacer)
    flags = [c["flags"] for c in clcks(out)]
    assert flags[0] == 3  # the chapter pause and an armed class
    assert (
        2 in flags and 3 in flags[flags.index(2) :] and flags[-1] == 0
    )  # chapter off, on again, then all off
    assert any("nope" in n for n in notes)
    assert decoded(out).header["play"]["autopause"] == {"chapter": True, "alert": {"pop_fall": True}}


# --- the pause ------------------------------------------------------------------------------------------


def paused_run(tmp_path: Path, lines: list[tuple[float, str]], steps: int, **kw: Any) -> tuple[Any, ...]:
    clock = FakeClock()
    ctl = TimedControl(clock, lines)
    return (*run_paced(world("2d"), tmp_path, steps, ctl, clock=clock, options=options(**kw)), clock)


def test_the_world_pauses_at_the_end_of_a_chapter_and_a_run_resumes_it(tmp_path: Path) -> None:
    rd, res, pacer, out, clock = paused_run(tmp_path / "w", [(1005.0, "run"), (1012.0, "run")], 70)
    assert res.status == "done" and pacer.failure is None
    records = decoded(out).records
    got = clcks(out)
    # the chapter card: reason 3 while the checkpoint is written, again when it is done, held while paused
    at30 = [c for c in got if c["step"] == 30]
    assert [(c["reason"], c["saving"]) for c in at30[:2]] == [(3, 1), (3, 0)]
    held = [c for c in at30 if c["reason"] == 3]
    assert len(held) >= 16 and all(c["rate"] == 0 for c in held[2:]), "a CLCK every quarter second, rate 0"
    assert at30[-1]["reason"] == 0 and at30[-1]["rate"] == -1, "run: running again"
    # no step is owed: nothing of step 31 is written before the pause ends
    first31 = next(i for i, r in enumerate(records) if r.step == 31)
    pause_end = max(i for i, r in enumerate(records[:first31]) if r.tag == "CLCK" and r.f["reason"] == 3)
    resumed = next(i for i, r in enumerate(records[:first31]) if i > pause_end and r.tag == "CLCK")
    assert pause_end < resumed < first31
    # and the same again at the end of the second chapter
    assert any(c["step"] == 60 and c["reason"] == 3 and c["saving"] == 0 for c in got)
    assert clock.now >= 1012.0, "the world waited for the second run"
    assert [s for s, _ in rd.checkpoints()][-1] == 70


def test_the_pause_also_holds_a_world_at_the_maximum_rate_and_the_frame_is_sent(tmp_path: Path) -> None:
    _, _, pacer, out, _ = paused_run(tmp_path / "w", [(1004.0, "run"), (1010.0, "run")], 40)
    frames = [r.step for r in decoded(out).records if r.tag == "FRMA"]
    assert frames.count(30) == 1 and 30 in frames, "one frame of the state the player reads, not two"
    assert pacer.sched.late_steps == 0


def test_with_the_pause_off_the_world_runs_on_and_the_card_is_a_notification(tmp_path: Path) -> None:
    clock = FakeClock()
    _, res, pacer, out = run_paced(
        world("2d"),
        tmp_path / "w",
        70,
        ScriptControl([["autopause chapter off"]]),
        clock=clock,
        options=options(),
    )
    assert res.status == "done" and clock.waits == 0, "never waited"
    got = clcks(out)
    reason3 = [(c["step"], c["saving"]) for c in got if c["reason"] == 3]
    assert reason3 == [(30, 1), (30, 0), (60, 1), (60, 0)], (
        "the card is announced at each chapter end, once saved"
    )
    assert all(c["flags"] & 1 == 0 for c in got if c["step"] > 0), "and the clock says the pause is off"
    assert pacer.sched.paused is False


def test_a_step_batch_never_pauses_by_itself(tmp_path: Path) -> None:
    clock = FakeClock()
    opt = options(rate=1.0, paused=True)
    _, res, pacer, out = run_paced(
        world("2d"), tmp_path / "w", 40, ScriptControl([["step 40"]]), clock=clock, options=opt
    )
    assert res.status == "done"
    got = clcks(out)
    assert max(c["step"] for c in got) == 40
    assert [(c["step"], c["saving"]) for c in got if c["reason"] == 3] == [(30, 1), (30, 0)]
    assert pacer.sched.paused, "the batch ends paused, as always"


def test_a_step_batch_out_of_a_chapter_pause_ends_its_reason(tmp_path: Path) -> None:
    lines = [(1004.0, "step 3"), (1008.0, "run"), (1012.0, "rate 2")]
    _, res, _, out, _ = paused_run(tmp_path / "w", lines, 45)
    assert res.status == "done"
    at30 = [c["reason"] for c in clcks(out) if c["step"] == 30]
    assert set(at30[:-1]) <= {3} and 3 in at30[2:-1]
    after = [c["reason"] for c in clcks(out) if c["step"] == 33]
    assert 1 in after and 3 not in after, "the batch is done: an ordinary pause of the player's"
    assert after[-1] == 0, "and `run` resumes it"


def test_rate_from_a_chapter_pause_runs_the_world_at_that_rate(tmp_path: Path) -> None:
    _, res, _, out, _ = paused_run(tmp_path / "w", [(1003.0, "rate max")], 40)
    assert res.status == "done"
    last = clcks(out)[-1]
    assert last["step"] == 40 and last["reason"] != 3


def test_the_pause_that_the_player_asks_for_in_a_running_world_is_reason_one(tmp_path: Path) -> None:
    clock = FakeClock()
    ctl = TimedControl(clock, [(1000.5, "pause"), (1003.0, "run")])
    _, _, _, out = run_paced(
        world("2d"), tmp_path / "w", 12, ctl, clock=clock, options=options(rate=1.0, chapter=play.CHAPTER)
    )
    reasons = {c["reason"] for c in clcks(out)}
    assert 1 in reasons and 3 not in reasons


def test_the_end_of_the_run_is_not_a_chapter_pause(tmp_path: Path) -> None:
    """``--until`` at a chapter end: the stream says "end of steps" (reason 2), the world does not wait."""
    clock = FakeClock()
    opt = options(until=30)
    _, res, _, out = run_paced(world("2d"), tmp_path / "w", 30, [[]], clock=clock, options=opt)
    assert res.status == "done" and clock.waits == 0
    assert clcks(out)[-1]["reason"] == 2


def test_a_quit_at_a_chapter_end_ends_the_run_without_ceremony(tmp_path: Path) -> None:
    clock = FakeClock()
    script = [
        ["rate max"],
        *([[]] * 28),
        ["quit"],
    ]  # read before step 30: one more step runs, to a chapter end
    rd, res, pacer, out = run_paced(world("2d"), tmp_path / "w", 90, script, clock=clock, options=options())
    assert res.status == "interrupted" and pacer.stop_requested
    assert rd.checkpoints()[-1][0] == 30 and clock.waits == 0
    assert [c["reason"] for c in clcks(out) if c["step"] == 30][:2] == [3, 3], (
        "the card is announced, then it ends"
    )


def test_a_stop_checkpoint_off_a_chapter_end_starts_no_report_and_no_pause(tmp_path: Path) -> None:
    ends: list[int] = []
    clock = FakeClock()
    cfg = world("2d")
    rd = RunDir.create(tmp_path / "w", cfg)
    pacer = play.PlayPacer(
        io.BytesIO(), ScriptControl([[], [], ["quit"]]), options(), clock=clock, note=lambda m: None
    )
    pacer.on_chapter_end = ends.append
    res = runner.run(rd, cfg, init_state(cfg), 90, quiet=True, pacer=pacer)
    assert res.status == "interrupted" and ends == [] and not pacer._chapter_pending


def test_a_resume_at_a_boundary_pauses_for_no_chapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshots(monkeypatch, 10)
    cfg = world("2d")
    rd, res, *_ = run_paced(cfg, tmp_path / "w", 30, [[]], options=options(until=30))
    assert res.status == "done" and rd.checkpoints()[-1][0] == 30
    state = load_checkpoint(rd.latest_checkpoint(), cfg)
    clock = FakeClock()
    out = io.BytesIO()
    pacer = play.PlayPacer(out, ScriptControl([[]]), options(), clock=clock, note=lambda m: None)
    res2 = runner.run(rd, cfg, state, 20, quiet=True, pacer=pacer)
    assert res2.status == "done" and clock.waits == 0, "no pause for the chapter that had just ended"
    assert all(c["reason"] != 3 for c in clcks(out))
    assert clcks(out)[0]["step"] == 30 and clcks(out)[0]["chapter"] == 1 and clcks(out)[0]["in_chapter"] == 30


def test_the_report_hook_runs_in_the_checkpoint_just_after_the_clock_that_announces_it(
    tmp_path: Path,
) -> None:
    seen: list[tuple[int, bool, int]] = []
    clock = FakeClock()
    cfg = world("2d")
    rd = RunDir.create(tmp_path / "w", cfg)
    out = io.BytesIO()
    pacer = play.PlayPacer(
        out,
        TimedControl(clock, [(1003.0, "run"), (1006.0, "run")]),
        options(),
        clock=clock,
        note=lambda m: None,
    )

    def hook(t: int) -> None:
        seen.append((t, pacer._chapter_pending, len(clcks(out))))

    pacer.on_chapter_end = hook
    runner.run(rd, cfg, init_state(cfg), 70, quiet=True, pacer=pacer)
    assert [(t, pending) for t, pending, _ in seen] == [(30, True), (60, True)]
    # the CLCK that announces the card (saving, reason 3) is already in the stream when the hook runs
    announced = clcks(out)[seen[0][2] - 1]
    assert (announced["step"], announced["reason"], announced["saving"]) == (30, 3, 1)


# --- the stall ------------------------------------------------------------------------------------------


def slow_saves(monkeypatch: pytest.MonkeyPatch, clock: FakeClock, at: int, seconds: float) -> None:
    real = runio.write_state_file

    def slow(path: Path, s: Any, cfg: Any) -> None:
        if int(np.array(s.step)) == at:
            clock.now += seconds
        real(path, s, cfg)

    monkeypatch.setattr(runio, "write_state_file", slow)


def test_the_stall_of_a_chapter_end_with_the_pause_is_not_counted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = FakeClock()
    slow_saves(monkeypatch, clock, 30, 2.0)
    ctl = TimedControl(clock, [(1100.0, "run")])
    _, _, pacer, out = run_paced(world("2d"), tmp_path / "w", 40, ctl, clock=clock, options=options(rate=1.0))
    assert pacer.sched.save_stall_steps == 0 and pacer.sched.late_steps == 0
    assert {c["save_stall_steps"] for c in clcks(out)} == {0}


def test_the_stall_with_the_pause_off_is_the_hitch_it_always_was(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = FakeClock()
    slow_saves(monkeypatch, clock, 30, 2.0)
    _, _, pacer, out = run_paced(
        world("2d"), tmp_path / "w", 40, ScriptControl([["autopause chapter off"]]), clock=clock,
        options=options(rate=1.0),
    )  # fmt: skip
    assert pacer.sched.save_stall_steps == pytest.approx(2.0 * play.BASE_RATE, abs=2)
    assert pacer.sched.late_steps == 0 and clcks(out)[-1]["save_stall_steps"] == pacer.sched.save_stall_steps


def test_end_save_can_leave_the_stall_uncounted() -> None:
    clock = FakeClock()
    s = play.Scheduler(clock, base_rate=10.0, rate=1.0, paused=False)
    s.begin_save()
    clock.now += 2.0
    s.end_save(count=False)
    assert s.save_stall_steps == 0 and s.late_steps == 0
    assert s.poll() == pytest.approx(0.1, abs=1e-6), "the schedule starts again from the end of the save"
    s.begin_save()
    clock.now += 2.0
    s.end_save()
    assert s.save_stall_steps == 20


# --- presentation never reaches the trajectory ----------------------------------------------------------


def test_a_chapter_pause_and_autopause_lines_leave_the_trajectory_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshots(monkeypatch, 10)
    cfg = world("2d")
    plain, res_plain = run_plain(cfg, tmp_path / "plain", 70)
    clock = FakeClock()
    ctl = TimedControl(
        clock,
        [
            (1000.0, "autopause chapter on"),
            (1003.0, "run"),
            (1005.0, "autopause alert x on"),
            (1009.0, "run"),
        ],
    )
    paced, res, pacer, out = run_paced(cfg, tmp_path / "paced", 70, ctl, clock=clock, options=options())
    assert pacer.failure is None and any(c["reason"] == 3 for c in clcks(out))
    assert state_digest(res.state) == state_digest(res_plain.state)
    assert checkpoint_digests(plain) == checkpoint_digests(paced)
    assert rows(plain) == rows(paced) and same_events(plain, paced)
    assert social_text(plain) == social_text(paced)


# --- the write lock -------------------------------------------------------------------------------------


class ChoppedSink(io.BytesIO):
    """A stream that takes at most 5 bytes per write and offers the other threads the interpreter in between,
    as a pipe that is nearly full does: a record written without the lock is cut into pieces by the others."""

    def write(self, data: Any) -> int:
        threading.Event().wait(0)  # yield
        return super().write(bytes(memoryview(data)[:5]))


def test_two_threads_taking_different_tags_never_repeat_a_seq() -> None:
    seq = streamrec.Sequencer()
    n = 10_000
    got: dict[str, list[int]] = {"FRMA": [], "RPRT": []}
    barrier = threading.Barrier(2)

    def take(tag: str) -> None:
        barrier.wait()
        got[tag] = [seq.take(tag) for _ in range(n)]

    import sys

    old = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        threads = [threading.Thread(target=take, args=(tag,)) for tag in got]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        sys.setswitchinterval(old)
    for tag in got:
        assert got[tag] == list(range(n)), tag
    assert seq.take("FRMA") == n and seq.take("RPRT") == n and seq.take("CLCK") == 0


def test_a_record_posted_by_a_worker_is_never_cut_by_the_loops_write(tmp_path: Path) -> None:
    out = ChoppedSink()
    pacer = play.PlayPacer(out, ScriptControl([]), options(), clock=FakeClock(), note=lambda m: None)
    pacer._t = 5
    report = _report(1, 0, True)
    stop = threading.Event()
    posted: list[int] = []

    def worker() -> None:
        i = 0
        while not stop.is_set() and i < 60:
            pacer.post(
                "RPRT",
                lambda step, seq: streamrec.encode_rprt(step=step, seq=seq, obj=report),
                step=1000 + i,
            )
            posted.append(i)
            i += 1

    thread = threading.Thread(target=worker)
    thread.start()
    for k in range(60):  # the loop's own records, in several short writes each
        pacer._send(
            "CLCK",
            lambda step, seq, k=k: streamrec.encode_clck(
                step=step, seq=seq, rate=1.0, achieved=1.0, step_ms=1.0, frame_ms=1.0, late_steps=0,
                save_stall_steps=0, reason=0, saving=False, fe=1, chapter=1, in_chapter=k, chapter_len=L,
            ),
        )  # fmt: skip
    stop.set()
    thread.join()
    tags = []
    data = out.getvalue()
    i = 0
    while i < len(data):
        tag = data[i : i + 4].decode("ascii")
        length = int.from_bytes(data[i + 4 : i + 8], "little")
        assert tag in ("CLCK", "RPRT"), f"a record was cut at byte {i}"
        tags.append((tag, int.from_bytes(data[i + 16 : i + 20], "little")))
        i += 8 + length
    assert i == len(data)
    for tag in ("CLCK", "RPRT"):
        assert [seq for t, seq in tags if t == tag] == list(range(sum(1 for t, _ in tags if t == tag)))
    assert len([t for t, _ in tags if t == "RPRT"]) == len(posted) == 60
    assert pacer.bytes_by_tag["RPRT"] + pacer.bytes_by_tag["CLCK"] == len(data)


def test_a_reader_that_closes_during_a_report_stops_the_play_through_the_loop(tmp_path: Path) -> None:
    class Closing(io.BytesIO):
        def write(self, data: Any) -> int:
            raise BrokenPipeError("reader closed")

    pacer = play.PlayPacer(Closing(), ScriptControl([]), options(), clock=FakeClock(), note=lambda m: None)
    report = _report(1, 0, True)
    thread = threading.Thread(
        target=lambda: pacer.post(
            "RPRT", lambda step, seq: streamrec.encode_rprt(step=step, seq=seq, obj=report), step=30
        )
    )
    thread.start()
    thread.join()
    assert isinstance(pacer.failure, BrokenPipeError) and pacer.stop_requested
    # a later post writes nothing and raises nothing
    pacer.post("RPRT", lambda step, seq: streamrec.encode_rprt(step=step, seq=seq, obj=report), step=60)
    assert pacer.bytes_by_tag == {}, "nothing counted for a record that was not written"
    # the loop's next call takes the ordinary stop path
    assert pacer._stop_check() is False and pacer.stop_requested


def test_the_report_worker_runs_during_the_save_and_its_record_comes_before_the_second_clock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The chapter end on a slow save: the hook starts a worker that posts the report while the state file is
    written, so the record is in the stream after the first ``CLCK`` (saving) and before the second."""
    snapshots(monkeypatch, 10)
    clock = FakeClock()
    real = runio.write_state_file
    workers: list[threading.Thread] = []

    def slow(path: Path, s: Any, cfg: Any) -> None:
        if int(np.array(s.step)) == 30:
            for w in workers:
                w.join()
            clock.now += 1.0
        real(path, s, cfg)

    monkeypatch.setattr(runio, "write_state_file", slow)
    cfg = world("2d")
    rd = RunDir.create(tmp_path / "w", cfg)
    out = io.BytesIO()
    pacer = play.PlayPacer(
        out, TimedControl(clock, [(1100.0, "run")]), options(rate=1.0), clock=clock, note=lambda m: None
    )

    def start(t: int) -> None:
        report = _report(1, 0, True)
        w = threading.Thread(
            target=lambda: pacer.post(
                "RPRT", lambda step, seq: streamrec.encode_rprt(step=step, seq=seq, obj=report), step=t
            )
        )
        workers.append(w)
        w.start()

    pacer.on_chapter_end = start
    runner.run(rd, cfg, init_state(cfg), 40, quiet=True, pacer=pacer)
    seq = [
        (r.tag, r.f.get("saving"), r.f.get("reason"))
        for r in decoded(out).records
        if r.step == 30 and r.tag in ("CLCK", "RPRT")
    ]
    assert seq[:3] == [("CLCK", 1, 3), ("RPRT", None, None), ("CLCK", 0, 3)]


# --- the origin row -------------------------------------------------------------------------------------


def test_the_origin_row_of_a_world_on_the_snapshot_grid() -> None:
    cfg = world("2d")
    state = init_state(cfg)
    row = play.origin_row(state, cfg)
    assert row is not None
    assert set(row) == {"step", "steps", "origin", "observers", "clades_alive", "top_clade_share"}
    assert (row["step"], row["steps"], row["origin"]) == (0, 0, True)
    assert row["observers"] == {"alive": int(np.array(state.alive).sum())}
    assert row["clades_alive"] >= 1 and 0.0 < row["top_clade_share"] <= 1.0
    import json

    assert json.loads(json.dumps(row)) == row, "it is meta.json material"


def test_there_is_no_origin_row_off_the_grid_or_without_observers(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = world("2d")
    state = init_state(cfg)
    snapshots(monkeypatch, 250)
    assert play.origin_row(state, cfg) is not None
    snapshots(monkeypatch, 7)  # a cadence the start step 0 divides: still on the grid
    assert play.origin_row(state, cfg) is not None
    import mlx.core as mx

    moved = state._replace(step=mx.array(12, dtype=state.step.dtype))
    snapshots(monkeypatch, 250)
    assert play.origin_row(moved, cfg) is None, (
        "a start off the grid has no cut: its first chapter is an arrival"
    )
    snapshots(monkeypatch, 12)
    row = play.origin_row(moved, cfg)
    assert row is not None and row["step"] == 12
    assert play.origin_row(init_state(world("2d", observe=False)), world("2d", observe=False)) is None


# --- what the mutation checks found missing ---------------------------------------------------------------


@pytest.mark.parametrize("line", ["pause", "rate 0"])
def test_pause_and_rate_zero_inside_a_chapter_pause_keep_its_reason(tmp_path: Path, line: str) -> None:
    """The world is already paused for the chapter: the player's own pause line changes nothing, so the clocks
    that follow it still say reason 3 until the `run` that resumes it."""
    _, res, _, out, _ = paused_run(tmp_path / "w", [(1002.0, line), (1008.0, "run")], 40)
    assert res.status == "done"
    reasons = [c["reason"] for c in clcks(out) if c["step"] == 30]
    assert reasons[-1] == 0 and set(reasons[:-1]) == {3}, reasons


def test_the_pause_sends_the_frame_of_the_state_the_player_reads(tmp_path: Path) -> None:
    """With a frame every 7 steps none is due at step 30: the pause sends the state the card stands on."""
    _, res, _, out, _ = paused_run(tmp_path / "w", [(1000.0, "every 7"), (1005.0, "run")], 40)
    assert res.status == "done"
    frames = [r.step for r in decoded(out).records if r.tag == "FRMA"]
    assert frames.count(30) == 1 and 29 not in frames and 31 not in frames
    assert [f for f in frames if f > 30][:1] == [35], "the frames go on every 7 steps after the pause"


@pytest.mark.parametrize("how", ["stop_requested", "stop_check"])
def test_a_stop_asked_at_the_chapter_end_is_not_a_pause(how: str) -> None:
    """A quit, a closed reader or a SIGINT latched at the chapter end: the loop leaves at once; the pacer does
    not pause the scheduler, send a frame or announce a pause on the way out."""
    out = io.BytesIO()
    pacer = play.PlayPacer(out, ScriptControl([]), options(), clock=FakeClock(), note=lambda m: None)
    pacer._chapter_pending = True
    if how == "stop_requested":
        pacer.stop_requested = True
    else:
        pacer._stop_check = lambda: True
    assert not pacer._chapter_pause_follows()
    pacer._begin_chapter_pause(None)  # type: ignore[arg-type]  (a pause would need the state for its frame)
    assert not pacer._chapter_pending and pacer._pause_reason == 0 and not pacer.sched.paused
    assert out.getvalue() == b""


class SlowSeq(streamrec.Sequencer):
    """A sequencer whose critical section lets the other threads in (unlocked, two takers read one number)."""

    def take(self, tag: str) -> int:
        n = super().take(tag)
        threading.Event().wait(
            0.02 / (n + 1)
        )  # the first taker is the slowest: a number taken outside the lock
        return n  # is written after the ones taken later


def test_a_sequencer_hands_each_number_out_once_even_when_the_takers_are_interleaved() -> None:
    class Racy(dict):  # type: ignore[type-arg]
        def __getitem__(self, key: Any) -> Any:
            threading.Event().wait(0.01)  # a taker that has read the counter is overtaken here
            return super().__getitem__(key)

    seq = streamrec.Sequencer()
    seq._next = Racy(seq._next)  # type: ignore[assignment]
    got: list[int] = []
    barrier = threading.Barrier(2)

    def take() -> None:
        barrier.wait()
        got.append(seq.take("RPRT"))

    threads = [threading.Thread(target=take) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(got) == [0, 1]


def test_two_workers_posting_one_tag_reach_the_stream_in_order_of_their_numbers() -> None:
    out = ChoppedSink()
    pacer = play.PlayPacer(out, ScriptControl([]), options(), clock=FakeClock(), note=lambda m: None)
    pacer.seq = SlowSeq()
    report = _report(1, 0, True)

    def worker() -> None:
        for _ in range(8):
            pacer.post(
                "RPRT", lambda step, seq: streamrec.encode_rprt(step=step, seq=seq, obj=report), step=30
            )

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    data = out.getvalue()
    seqs, i = [], 0
    while i < len(data):
        seqs.append(int.from_bytes(data[i + 16 : i + 20], "little"))
        i += 8 + int.from_bytes(data[i + 4 : i + 8], "little")
    assert seqs == list(range(16)), "the number is taken inside the lock that spans the write"
