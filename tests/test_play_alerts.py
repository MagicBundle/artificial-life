"""Alerts in the play loop (DESIGN §34.3): the engine's hooks, ``ALRT`` records, ``alerts.jsonl``, the pause
by an alert (``CLCK`` reason 4), ``watch`` and ``ack``, the backlog of a resumed world, and the guarantees:
presentation never reaches the trajectory, a replay writes its wall-clock alerts apart, the logs are cut with
the checkpoints.

Worlds are the tiny ones of ``play_helpers`` with a chapter of 30 steps; a population collapse and a naming
are injected through a pacer subclass (the tiny worlds raise none of their own)."""

from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from alife import play, runner, storms, streamrec
from alife.game import alerts as al
from alife.runio import ALERTS_FILE, ALERTS_REPLAY_FILE, DIRECTOR_LOG_FILE, RunDir, load_checkpoint
from alife.state import init_state
from play_helpers import (
    FakeClock,
    ScriptControl,
    TimedControl,
    checkpoint_digests,
    decoded,
    run_paced,
    run_plain,
    snapshots,
    world,
)
from test_play_storms import scene

L = 30


def options(**kw: Any) -> play.PlayOptions:
    return play.PlayOptions(**{"rate": float("inf"), "paused": False, "fresh": True, "chapter": L, **kw})


def alrts(out: io.BytesIO) -> list[dict[str, Any]]:
    """The alerts of the stream, in order, each with the step and the backlog flag of its record."""
    return [
        {**a, "at": r.step, "backlog": r.f["backlog"]}
        for r in decoded(out).records
        if r.tag == "ALRT"
        for a in r.f["json"]
    ]


def clcks(out: io.BytesIO) -> list[dict[str, Any]]:
    return [{"step": r.step, **r.f} for r in decoded(out).records if r.tag == "CLCK"]


def tags(out: io.BytesIO) -> list[tuple[str, int]]:
    return [(r.tag, r.step) for r in decoded(out).records]


class Collapse(play.PlayPacer):
    """A pacer that hands the alert engine a collapsed population at the metrics row of ``at_step`` (a copy of
    the row: the run's own row, and the trajectory, are untouched)."""

    at_step = 20
    alive = 1

    def after_boundary(self, state: Any, row: dict[str, Any], t: int) -> None:
        if t == self.at_step:
            row = {**row, "observers": {"alive": self.alive}}
        super().after_boundary(state, row, t)


def paused_run(
    tmp_path: Path, lines: list[tuple[float, str]], steps: int, pacer_class: type = play.PlayPacer, **kw: Any
) -> tuple[Any, ...]:
    clock = FakeClock()
    ctl = TimedControl(clock, lines)
    return (
        *run_paced(
            world("2d"), tmp_path, steps, ctl, clock=clock, options=options(**kw), pacer_class=pacer_class
        ),
        clock,
    )


# --- the header ------------------------------------------------------------------------------------


def test_the_header_registers_the_classes_and_the_autopause_defaults(tmp_path: Path) -> None:
    _, _, _, out = run_paced(world("2d"), tmp_path / "w", 3, [[]], options=options())
    play_block = decoded(out).header["play"]
    assert "ALRT" in play_block["records"]
    assert play_block["alerts"] == streamrec.alerts_block(al.registry())
    assert play_block["autopause"] == {"chapter": True, "alert": al.default_autopause()}


# --- the chapter alert -----------------------------------------------------------------------------


def test_the_chapter_alert_precedes_the_card_names_the_pause_and_is_cleared_by_run(tmp_path: Path) -> None:
    rd, res, pacer, out, _ = paused_run(tmp_path / "w", [(1005.0, "run"), (1012.0, "run")], 70)
    assert res.status == "done" and pacer.failure is None
    got = [a for a in alrts(out) if a["class"] == "chapter"]
    assert [(a["state"], a["step"], a["version"]) for a in got] == [
        ("raised", 30, 1), ("cleared", 30, 2), ("raised", 60, 1), ("cleared", 60, 2),
    ]  # fmt: skip
    first = got[0]
    assert first["imp"] == 3 and first["pause"] is True and first["expires"] is None
    assert first["id"] == 30_600 and first["text"] == "The chapter ended at step 30."
    # the alert reaches the client before the card is announced (the clock with reason 3)
    seq = tags(out)
    i_alert = next(i for i, (t, s) in enumerate(seq) if t == "ALRT" and s == 30)
    i_card = next(i for i, (t, s) in enumerate(seq) if t == "CLCK" and s == 30 and i > i_alert - 3)
    assert i_alert < i_card
    # it is in the log (the tiny world raises a rise of its population of its own now and then: not asserted)
    lines = [m for m in rd.read_jsonl(ALERTS_FILE) if m["class"] == "chapter"]
    assert [(m["state"], m["step"]) for m in lines] == [
        ("raised", 30), ("cleared", 30), ("raised", 60), ("cleared", 60),
    ]  # fmt: skip


def test_with_the_chapter_pause_off_the_chapter_alert_says_it_does_not_pause(tmp_path: Path) -> None:
    clock = FakeClock()
    _, _, _, out = run_paced(
        world("2d"),
        tmp_path / "w",
        35,
        ScriptControl([["autopause chapter off"]]),
        clock=clock,
        options=options(),
    )
    (a,) = [a for a in alrts(out) if a["class"] == "chapter"]
    assert a["pause"] is False and a["state"] == "raised" and clock.waits == 0


def test_a_new_chapter_alert_replaces_one_the_player_never_cleared(tmp_path: Path) -> None:
    clock = FakeClock()
    _, _, _, out = run_paced(
        world("2d"),
        tmp_path / "w",
        65,
        ScriptControl([["autopause chapter off"]]),
        clock=clock,
        options=options(),
    )
    got = [(a["id"], a["step"], a["state"]) for a in alrts(out) if a["class"] == "chapter"]
    assert got == [(30_600, 30, "raised"), (30_600, 60, "cleared"), (60_600, 60, "raised")]


# --- the pause by an alert -------------------------------------------------------------------------


def test_a_population_collapse_pauses_with_reason_four_until_run(tmp_path: Path) -> None:
    rd, res, pacer, out, clock = paused_run(
        tmp_path / "w", [(1003.0, "run")], 28, pacer_class=Collapse, chapter=play.CHAPTER
    )
    assert res.status == "done" and pacer.failure is None
    got = {a["class"]: a for a in alrts(out) if a["state"] == "raised"}
    assert set(got) >= {"pop_low", "pop_fall"}
    low = got["pop_low"]
    assert low["imp"] == 3 and low["pause"] is True and low["step"] == 20
    assert low["text"].startswith("The population is 1, ")
    held = [c for c in clcks(out) if c["step"] == 20 and c["reason"] == 4]
    assert len(held) >= 8 and all(c["rate"] == 0 for c in held), (
        "held in the pause, a clock every quarter second"
    )
    assert clcks(out)[-1]["reason"] == 0 and clock.now >= 1003.0
    # one pause for the two alerts of the step, and none again for the same keys
    assert sum(1 for c in clcks(out) if c["reason"] == 4 and c["rate"] == 0 and c["step"] == 20) == len(held)
    assert not [c for c in clcks(out) if c["reason"] == 4 and c["step"] > 20]
    assert {m["class"] for m in rd.read_jsonl(ALERTS_FILE)} >= {"pop_low", "pop_fall"}


def test_an_alert_pause_is_presentation_only_the_trajectory_is_the_plain_runs(tmp_path: Path) -> None:
    cfg = world("2d")
    plain, _ = run_plain(cfg, tmp_path / "plain", 30)
    clock = FakeClock()
    ctl = TimedControl(clock, [(1003.0, "run")])
    paced, res, *_ = run_paced(
        cfg,
        tmp_path / "paced",
        30,
        ctl,
        clock=clock,
        options=options(chapter=play.CHAPTER),
        pacer_class=Collapse,
    )
    assert res.status == "done"
    assert checkpoint_digests(paced) == checkpoint_digests(plain)


def test_a_class_switched_off_raises_its_alert_without_a_pause(tmp_path: Path) -> None:
    clock = FakeClock()
    script = [["autopause alert pop_low off", "autopause alert pop_fall off"]]
    _, res, _, out = run_paced(
        world("2d"),
        tmp_path / "w",
        28,
        ScriptControl(script),
        clock=clock,
        options=options(chapter=play.CHAPTER),
        pacer_class=Collapse,
    )
    assert res.status == "done" and clock.waits == 0
    got = {a["class"]: a for a in alrts(out) if a["state"] == "raised"}
    assert got["pop_low"]["pause"] is False and got["pop_low"]["imp"] == 3
    assert all(c["reason"] != 4 for c in clcks(out))


def test_the_chapter_pause_wins_when_an_alert_comes_at_the_same_step(tmp_path: Path) -> None:
    class AtChapterEnd(Collapse):
        at_step = L

    _, res, _, out, _ = paused_run(
        tmp_path / "w", [(1003.0, "run"), (1006.0, "run")], 35, pacer_class=AtChapterEnd
    )
    assert res.status == "done"
    low = next(a for a in alrts(out) if a["class"] == "pop_low")
    assert low["pause"] is False, "the chapter pause is certain to follow: the alert does not claim one"
    assert {c["reason"] for c in clcks(out) if c["step"] == L and c["rate"] == 0} == {3}
    assert all(c["reason"] != 4 for c in clcks(out))


def test_a_step_batch_never_pauses_for_an_alert(tmp_path: Path) -> None:
    clock = FakeClock()
    opt = options(rate=1.0, paused=True, chapter=play.CHAPTER)
    _, res, pacer, out = run_paced(
        world("2d"),
        tmp_path / "w",
        25,
        ScriptControl([["step 25"]]),
        clock=clock,
        options=opt,
        pacer_class=Collapse,
    )
    assert res.status == "done"
    low = next(a for a in alrts(out) if a["class"] == "pop_low")
    assert low["pause"] is False
    assert all(c["reason"] != 4 for c in clcks(out)) and pacer.sched.paused


def test_unattended_worlds_arm_no_pause(tmp_path: Path) -> None:
    clock = FakeClock()
    _, res, pacer, out = run_paced(
        world("2d"),
        tmp_path / "w",
        28,
        [[]],
        clock=clock,
        options=options(unattended=True, chapter=play.CHAPTER),
        pacer_class=Collapse,
    )
    assert res.status == "done" and clock.waits == 0
    assert not any(pacer.autopause.alert.values()) and pacer.autopause.flags == 0
    assert decoded(out).header["play"]["autopause"]["alert"] == dict.fromkeys(al.default_autopause(), False)


# --- watch, ack ------------------------------------------------------------------------------------


def test_ack_logs_an_acknowledgement_and_notes_an_unknown_alert(tmp_path: Path) -> None:
    notes: list[str] = []
    clock = FakeClock()
    cfg = world("2d")
    rd = RunDir.create(tmp_path / "w", cfg)
    out = io.BytesIO()
    ctl = TimedControl(clock, [(1002.0, "ack 30600"), (1002.0, "ack 12345"), (1003.0, "run")])
    pacer = play.PlayPacer(out, ctl, options(), clock=clock, note=notes.append)
    runner.run(rd, cfg, init_state(cfg), 40, quiet=True, pacer=pacer)
    assert any("no alert 12345" in n for n in notes) and not any("no alert 30600" in n for n in notes)
    acks = [m for m in rd.read_jsonl(ALERTS_FILE) if m["state"] == "acknowledged"]
    assert [(m["id"], m["class"], m["step"]) for m in acks] == [(30_600, "chapter", 30)]


def test_watch_notes_a_group_the_world_does_not_list_and_unwatch_one_it_does_not_watch(
    tmp_path: Path,
) -> None:
    notes: list[str] = []
    cfg = world("2d")
    rd = RunDir.create(tmp_path / "w", cfg)
    pacer = play.PlayPacer(
        io.BytesIO(),
        ScriptControl([["watch 77", "unwatch 77", "watch off"], []]),
        options(chapter=play.CHAPTER),
        clock=FakeClock(),
        note=notes.append,
    )
    runner.run(rd, cfg, init_state(cfg), 5, quiet=True, pacer=pacer)
    assert [n.split(":")[0] for n in notes] == ["watch", "unwatch"]
    assert "no kin group 77" in notes[0] and "not watched" in notes[1]


# --- the log, the backlog, the cuts ----------------------------------------------------------------


class Naming(play.PlayPacer):
    """A pacer that hands the engine a storm naming at step 10 (an alert that stays 3,000 steps)."""

    def after_step(self, state: Any, out: Any, t: int, social: list[dict[str, Any]]) -> None:
        super().after_step(state, out, t, social)
        if t == 10:
            ev = {"kind": "named", "id": 4, "name": "Aa", "x": 40.0, "y": 5.0, "initial": False, "age": 0}
            self.alerts.on_tick(t, [{"id": 4, "x": 40.0, "y": 5.0, "r": 7.0, "zeta": 0.55}], [ev], [ev])
            self._emit_alerts()


def test_alerts_are_in_the_log_before_the_client_sees_them_and_a_resume_shows_the_unexpired_ones(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshots(monkeypatch, 10)
    cfg = world("2d")
    rd, res, _, out = run_paced(
        cfg,
        tmp_path / "w",
        40,
        [["autopause off"]],
        options=options(chapter=play.CHAPTER),
        pacer_class=Naming,
    )
    assert res.status == "done"
    named = [m for m in rd.read_jsonl(ALERTS_FILE) if m["class"] == "storm_named"]
    assert [(m["state"], m["step"], m["expires"]) for m in named] == [("raised", 10, 3010)]
    live = [a for a in alrts(out) if a["class"] == "storm_named"]
    assert len(live) == 1 and live[0]["backlog"] is False
    # a new session on the checkpoint at 30: the alert (expiring at 3,010) is on the stack, flagged backlog
    rd.truncate_after(30)
    out2 = io.BytesIO()
    pacer2 = play.PlayPacer(
        out2, ScriptControl([[]]), options(chapter=play.CHAPTER), clock=FakeClock(), note=print
    )
    runner.run(rd, cfg, load_checkpoint(rd.latest_checkpoint(), cfg), 5, quiet=True, pacer=pacer2)
    back = [a for a in alrts(out2) if a["backlog"] and a["class"] == "storm_named"]
    assert [(a["id"], a["pause"]) for a in back] == [(10_600, False)]
    seq = tags(out2)
    assert seq.index(("ALRT", 30)) < seq.index(("CLCK", 30)), "before the first clock"


def test_the_alert_logs_are_cut_with_the_checkpoint(tmp_path: Path) -> None:
    rd, res, *_ = run_paced(
        world("2d"), tmp_path / "w", 70, [["autopause chapter off"], []], options=options()
    )
    assert res.status == "done"
    steps = [m["step"] for m in rd.read_jsonl(ALERTS_FILE)]
    assert 30 in steps and 60 in steps
    (rd.path / ALERTS_REPLAY_FILE).write_text(
        '{"step": 40, "class": "engine", "state": "raised"}\n{"step": 20}\n'
    )
    rd.truncate_after(30)
    assert [m["step"] for m in rd.read_jsonl(ALERTS_FILE)] == [s for s in steps if s <= 30]
    assert [m["step"] for m in rd.read_jsonl(ALERTS_REPLAY_FILE)] == [20]
    rd.truncate_after(30, keep_director=True)  # a replay cuts them too: they belong to the trajectory
    assert [m["step"] for m in rd.read_jsonl(ALERTS_FILE)] == [s for s in steps if s <= 30]
    rd.truncate_after(0)
    assert rd.read_jsonl(ALERTS_FILE) == [] and rd.read_jsonl(ALERTS_REPLAY_FILE) == []


def test_a_torn_last_line_of_the_alert_log_is_dropped_by_the_cut(tmp_path: Path) -> None:
    rd, *_ = run_paced(world("2d"), tmp_path / "w", 40, [["autopause chapter off"]], options=options())
    with open(rd.path / ALERTS_FILE, "a", encoding="utf-8") as f:
        f.write('{"step": 35, "cla')
    rd.truncate_after(30)
    assert all(isinstance(m["step"], int) for m in rd.read_jsonl(ALERTS_FILE))


def test_a_replay_writes_its_wall_clock_alerts_apart(tmp_path: Path) -> None:
    pacer = play.PlayPacer(io.BytesIO(), ScriptControl([]), options(), clock=FakeClock(), note=print)
    cfg = world("2d")
    rd = RunDir.create(tmp_path / "w", cfg)
    pacer.rd = rd
    lines = [
        {"class": "engine", "step": 5, "state": "raised"},
        {"class": "chapter", "step": 30, "state": "raised"},
    ]
    pacer._replaying = False
    pacer._log_alerts(lines)
    assert [m["class"] for m in rd.read_jsonl(ALERTS_FILE)] == ["engine", "chapter"]
    assert rd.read_jsonl(ALERTS_REPLAY_FILE) == []
    pacer._replaying = True
    pacer._log_alerts(lines)
    assert [m["class"] for m in rd.read_jsonl(ALERTS_FILE)] == ["engine", "chapter", "chapter"]
    assert [m["class"] for m in rd.read_jsonl(ALERTS_REPLAY_FILE)] == ["engine"]


def test_the_engine_class_reads_the_injected_clock_of_the_loop(tmp_path: Path) -> None:
    """A world that cannot keep its rate raises an ``engine`` alert from the pacer's clock samples."""

    class Slow(play.PlayPacer):
        def after_step(self, state: Any, out: Any, t: int, social: list[dict[str, Any]]) -> None:
            self.clock.now += 0.2  # type: ignore[attr-defined]  # 5 steps a second against 14.4 requested
            super().after_step(state, out, t, social)

    clock = FakeClock()
    opt = play.PlayOptions(rate=1.0, paused=False, fresh=True, chapter=play.CHAPTER, unattended=True)
    _, res, _, out = run_paced(
        world("2d"), tmp_path / "w", 200, [[]], clock=clock, options=opt, pacer_class=Slow
    )
    assert res.status == "done"
    engine = [a for a in alrts(out) if a["class"] == "engine"]
    assert engine and engine[0]["imp"] == 1 and engine[0]["pause"] is False
    assert engine[0]["text"].startswith(("The play ran at ", "The play ran late"))
    assert len(engine) <= 2, "deduped for 300 seconds of the clock"


def test_a_presenter_failure_leaves_an_alert_in_the_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = play.PlayPacer._send_frame

    def boom(self: play.PlayPacer, state: Any) -> None:
        if self._t >= 6:
            raise RuntimeError("the frame cannot be made")
        original(self, state)

    monkeypatch.setattr(play.PlayPacer, "_send_frame", boom)
    rd, _, pacer, _ = run_paced(world("2d"), tmp_path / "w", 12, [[]], options=options(chapter=play.CHAPTER))
    assert isinstance(pacer.failure, RuntimeError)
    (m,) = [m for m in rd.read_jsonl(ALERTS_FILE) if m["class"] == "engine"]
    assert m["imp"] == 3 and "the frame cannot be made" in m["text"] and m["pause"] is False


def test_the_territory_alert_is_only_evaluated_with_the_experimental_flag(tmp_path: Path) -> None:
    pacer = play.PlayPacer(io.BytesIO(), ScriptControl([]), options(), clock=FakeClock(), note=print)
    assert pacer.alerts.experimental is False
    on = play.PlayPacer(
        io.BytesIO(), ScriptControl([]), options(alerts_experimental=True), clock=FakeClock(), note=print
    )
    assert on.alerts.experimental is True


# --- more of the loop ---------------------------------------------------------------------------------------


def test_an_alert_pause_sends_the_frame_the_player_reads(tmp_path: Path) -> None:
    lines = [(1000.0, "every 7"), (1003.0, "run")]  # frames at multiples of 7: none would be sent at step 20
    _, res, _, out, _ = paused_run(tmp_path / "w", lines, 28, pacer_class=Collapse, chapter=play.CHAPTER)
    assert res.status == "done"
    assert [r.step for r in decoded(out).records if r.tag == "FRMA"].count(20) == 1


def test_a_rate_change_in_a_running_world_does_not_acknowledge_the_chapter_alert(tmp_path: Path) -> None:
    script = [["autopause chapter off"]] + [[]] * 34 + [["rate max"]] + [[]] * 9 + [["run"]]
    _, _, _, out = run_paced(world("2d"), tmp_path / "w", 50, script, options=options())
    got = [(a["step"], a["state"]) for a in alrts(out) if a["class"] == "chapter"]
    assert got == [(30, "raised"), (45, "cleared")], "cleared by the run, not by the rate at step 35"


def test_a_step_batch_across_a_chapter_end_flags_no_pause_on_the_chapter_alert(tmp_path: Path) -> None:
    opt = options(rate=1.0, paused=True)
    _, res, _, out = run_paced(
        world("2d"), tmp_path / "w", 40, ScriptControl([["step 40"]]), clock=FakeClock(), options=opt
    )
    assert res.status == "done"
    (a,) = [a for a in alrts(out) if a["class"] == "chapter"]
    assert a["pause"] is False, "a batch is not paused by the chapter"


class Flood(play.PlayPacer):
    """Twenty tracker events told at one tick: more alerts than one ``ALRT`` carries."""

    def after_step(self, state: Any, out: Any, t: int, social: list[dict[str, Any]]) -> None:
        super().after_step(state, out, t, social)
        if t == 10:
            ev = [
                {"kind": "dissolved", "id": i, "name": f"S{i}", "age": 100, "x": 1.0, "y": 2.0}
                for i in range(1, 21)
            ]
            self.alerts.on_tick(t, [], ev, ev)
            self._emit_alerts()


def test_more_alerts_than_one_record_carries_go_out_in_batches(tmp_path: Path) -> None:
    _, _, _, out = run_paced(
        world("2d"), tmp_path / "w", 12, [[]], options=options(chapter=play.CHAPTER, unattended=True),
        pacer_class=Flood,
    )  # fmt: skip
    sizes = [
        len(r.f["json"])
        for r in decoded(out).records
        if r.tag == "ALRT" and r.step == 10 and r.f["json"][0]["class"] == "storm_ended"
    ]
    assert sizes == [16, 4]


def test_a_director_replay_is_known_by_its_mode() -> None:
    assert play.replaying(SimpleNamespace(mode="replay")) is True
    assert play.replaying(SimpleNamespace(mode="live")) is False
    assert play.replaying(SimpleNamespace(mode="player")) is False
    assert play.replaying(None) is False


class StubRun:
    """What ``_open_alerts`` reads of a run directory."""

    def __init__(self, rows: list[dict[str, Any]], meta: dict[str, Any]) -> None:
        self._rows, self._meta = rows, meta

    def metrics_rows(self) -> list[dict[str, Any]]:
        return self._rows

    def meta(self) -> dict[str, Any]:
        return self._meta


def test_a_resumed_world_primes_the_engine_from_its_files(tmp_path: Path) -> None:
    pacer = play.PlayPacer(
        io.BytesIO(), ScriptControl([]), options(chapter=play.CHAPTER), clock=FakeClock(), note=print
    )
    pacer.layer = None
    rows = [
        {
            "step": s,
            "steps": 250,
            "observers": {"alive": 1000 + s // 250},
            "deaths_starved": 7,
            "deaths_storm": 2,
        }
        for s in range(7_500, 17_501, 250)
    ]
    origin = {"step": 6_000, "steps": 0, "origin": True, "observers": {"alive": 4242}}
    rec = {"step": 7_500, "houses": {"id": [5], "name": ["Mycleav"], "size": [300]}}
    pacer._open_alerts(StubRun(rows, {"origin_row": origin}), [rec])  # type: ignore[arg-type]
    eng = pacer.alerts
    assert [s for s, _ in eng.rows] == [r["step"] for r in rows[-25:]]
    assert eng.opened is True
    assert list(eng.ends) == [(6_000, 4242), (12_000, 1048)], "the origin row is the first chapter end"
    assert eng.houses == {5: {"listings": 1, "peak": 300, "size": 300, "out": 0}}
    assert len(eng.starved) == 8 and len(eng.storm_deaths) == 8


def test_chapter_end_is_an_alert_class_with_the_chapter_switch() -> None:
    a = play.Autopause(alert=al.default_autopause())
    assert a.apply(play.AutopauseLine("alert", "chapter", False)) is None and a.chapter is False
    assert a.apply(play.AutopauseLine("alert", "chapter", True)) is None and a.chapter is True
    assert "chapter" not in a.alert


def test_an_ack_is_in_the_log_before_the_next_step(tmp_path: Path) -> None:
    cfg = world("2d")
    rd = RunDir.create(tmp_path / "w", cfg)
    pacer = play.PlayPacer(io.BytesIO(), ScriptControl([]), options(), clock=FakeClock(), note=print)
    pacer.rd = rd
    pacer.alerts.know([{"id": 600, "class": "storm_named"}])
    pacer._apply(["ack 600"], None)  # type: ignore[arg-type]
    assert [m["state"] for m in rd.read_jsonl(ALERTS_FILE)] == ["acknowledged"]


def test_the_engine_class_is_sampled_with_every_clock_and_its_alert_goes_out_at_once(tmp_path: Path) -> None:
    clock = FakeClock()
    out = io.BytesIO()
    rd = RunDir.create(tmp_path / "w", world("2d"))
    pacer = play.PlayPacer(out, ScriptControl([]), options(), clock=clock, note=print)
    pacer.rd = rd
    pacer._service()  # the first sample
    assert b"ALRT" not in out.getvalue()
    pacer.sched.late_steps = 9
    clock.now += 1.0
    pacer._service()
    assert b"ALRT" in out.getvalue() and out.getvalue().index(b"ALRT") < out.getvalue().rindex(b"CLCK")
    assert [m["class"] for m in rd.read_jsonl(ALERTS_FILE)] == ["engine"]


def test_the_alert_pause_waits_for_a_world_that_is_running_free(tmp_path: Path) -> None:
    """A pending alert pause is dropped, not taken, when a batch runs, the world is already paused or a stop
    was asked: the chapter pause or the player's own pause stands."""
    pacer = play.PlayPacer(io.BytesIO(), ScriptControl([]), options(), clock=FakeClock(), note=print)
    for setup in ("batch", "paused", "stop"):
        pacer.sched = play.Scheduler(FakeClock(), base_rate=14.4, rate=1.0, paused=False)
        pacer._pause_reason = 0
        pacer.stop_requested = False
        pacer._stop_check = lambda: False
        if setup == "batch":
            pacer.sched.step(5)
        elif setup == "paused":
            pacer.sched.pause()
        else:
            pacer.stop_requested = True
        pacer._alert_pause_pending = True
        pacer._begin_alert_pause(None)  # type: ignore[arg-type]
        assert pacer._alert_pause_pending is False and pacer._pause_reason == 0, setup


# --- the live engine and the fold of its own files agree ---------------------------------------------


def comparable(lines: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
    keys = ("id", "version", "class", "key", "state", "surface", "imp", "text", "step", "ref")
    return [tuple(m[k] for k in keys) for m in lines if m["class"] != "engine"]


def test_the_live_engine_raises_what_the_fold_of_its_files_raises_in_a_storm_world(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshots(monkeypatch, 10)
    dets = scene(3, 8)
    monkeypatch.setattr(storms.StormLayer, "tick_at", lambda self, ws, t: self.ingest(dets[t // 25 - 1]))
    rd, res, pacer, _ = run_paced(world("storm"), tmp_path / "w", 200, [[]], options=options(unattended=True))
    assert res.status == "done"
    live = rd.read_jsonl(ALERTS_FILE)
    assert {m["class"] for m in live} >= {"chapter", "storm_named"}, "the scene names storms"
    folded, _ = al.fold_logs(
        rd.metrics_rows(),
        rd.social_records(),
        rd.storms_lines(),
        chapter=L,
        house_salt=pacer.chron.house_salt,
    )
    assert comparable(live) == comparable(folded)


def test_the_live_engine_raises_what_the_fold_of_its_files_raises_with_a_director(tmp_path: Path) -> None:
    rd, res, pacer, _ = run_paced(
        world("2d", director=True), tmp_path / "w", 120, [[]], options=options(unattended=True)
    )
    assert res.status == "done"
    entries = [e for e in rd.read_jsonl(DIRECTOR_LOG_FILE) if e.get("kind") != "header"]
    live = rd.read_jsonl(ALERTS_FILE)
    assert any(m["class"] == "director_acted" for m in live), "the rule tier acted"
    folded, _ = al.fold_logs(
        rd.metrics_rows(), rd.social_records(), rd.storms_lines(), entries, chapter=L,
        house_salt=pacer.chron.house_salt,
    )  # fmt: skip
    assert comparable(live) == comparable(folded)
