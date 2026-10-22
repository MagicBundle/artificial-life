"""The control grammar of ``alife play`` (DESIGN §33.2.6) and what each line does to the stream.

The grammar is a table of valid and invalid lines; the channel (``stream.FdControl``) is exercised on a real
pipe for the 4,096-byte limit and ``quit``; the effects are read from the records a paced tiny world writes.
"""

from __future__ import annotations

import io
import os
import threading
from pathlib import Path

import pytest

from alife import play, runner, streamrec
from alife.play import BASE_RATE as B
from alife.play import AutopauseLine, PlayCommand, PowerLine, parse_play_command
from alife.runio import RunDir
from alife.state import init_state
from alife.stream import ControlError, FdControl
from play_helpers import FakeClock, ScriptControl, TimedControl, decoded, run_paced, snapshots, world

VALID = [
    ("pause", PlayCommand("pause")),
    ("run", PlayCommand("run")),
    ("idnt", PlayCommand("idnt")),
    ("quit", PlayCommand("quit")),
    ("step", PlayCommand("step", 1)),
    ("step 25", PlayCommand("step", 25)),
    ("rate 1", PlayCommand("rate", 1.0)),
    ("rate 0.25", PlayCommand("rate", 0.25)),
    ("rate 0", PlayCommand("rate", 0.0)),
    ("rate 16", PlayCommand("rate", 16.0)),
    ("rate max", PlayCommand("rate", "max")),
    ("color house", PlayCommand("color", "house")),
    ("color tag", PlayCommand("color", "tag")),
    ("every 3", PlayCommand("every", 3)),
    ("every 64", PlayCommand("every", 64)),
    ("every auto", PlayCommand("every", "auto")),
    ("focus 12345678901", PlayCommand("focus", 12345678901)),
    ("focus 0", PlayCommand("focus", 0)),
    ("focus off", PlayCommand("focus", "off")),
    ("follow 7", PlayCommand("follow", 7)),
    ("follow off", PlayCommand("follow", "off")),
    ("labels on", PlayCommand("labels", "on")),
    ("labels off", PlayCommand("labels", "off")),
    ("  pause   ", PlayCommand("pause")),
    ("rate\t2", PlayCommand("rate", 2.0)),
    # G3b (DESIGN §34.6): the lines of the decide half
    ("autopause on", PlayCommand("autopause", AutopauseLine("all", None, True))),
    ("autopause off", PlayCommand("autopause", AutopauseLine("all", None, False))),
    ("autopause chapter on", PlayCommand("autopause", AutopauseLine("chapter", None, True))),
    ("autopause chapter off", PlayCommand("autopause", AutopauseLine("chapter", None, False))),
    ("autopause alert pop_fall on", PlayCommand("autopause", AutopauseLine("alert", "pop_fall", True))),
    (
        "autopause alert storm_named off",
        PlayCommand("autopause", AutopauseLine("alert", "storm_named", False)),
    ),
    ("watch 12", PlayCommand("watch", 12)),
    ("watch off", PlayCommand("watch", "off")),
    ("unwatch 12", PlayCommand("unwatch", 12)),
    ("ack 466250600", PlayCommand("ack", 466250600)),
    ("ack 9223372036854775807", PlayCommand("ack", 9223372036854775807)),
    ("power food id=1", PlayCommand("power", PowerLine("apply", "food", 1))),
    (
        "power food id=1 mult=1.3 steps=6000",
        PlayCommand("power", PowerLine("apply", "food", 1, (("mult", 1.3), ("steps", 6000)))),
    ),
    (
        "power preview bloom id=0 patch=2 gain=1.5 steps=3000",
        PlayCommand(
            "power", PowerLine("preview", "bloom", 0, (("patch", 2), ("gain", 1.5), ("steps", 3000)))
        ),
    ),
    (
        "power blight id=7 patch=0 steps=12000",
        PlayCommand("power", PowerLine("apply", "blight", 7, (("patch", 0), ("steps", 12000)))),
    ),
    (
        "power strike id=9 x=100.5 y=20 radius=12.5 kill=0.4",
        PlayCommand(
            "power",
            PowerLine("apply", "strike", 9, (("x", 100.5), ("y", 20.0), ("radius", 12.5), ("kill", 0.4))),
        ),
    ),
    ("power cancel food id=4", PlayCommand("power", PowerLine("cancel", "food", 4))),
    ("intent make it rain", PlayCommand("intent", "make it rain")),
    ("intent  Make   more FOOD ", PlayCommand("intent", "Make   more FOOD")),
]

BLANK = ["", "   ", "# a comment", "  # rate nonsense"]
INVALID = [
    ("Pause", "unknown command"),
    ("pausee", "unknown command"),
    ("pause now", "takes 0"),
    ("run 1", "takes 0"),
    ("step 0", "N >= 1"),
    ("step -1", "non-negative integer"),
    ("step 1 2", "takes 0 to 1"),
    ("step x", "non-negative integer"),
    ("rate", "takes 1"),
    ("rate fast", "decimal"),
    ("rate -1", "decimal"),
    ("rate 0.01", "rate must be"),
    ("rate 17", "rate must be"),
    ("rate 1e1", "decimal"),
    ("rate MAX", "decimal"),
    ("color", "takes 1"),
    ("color red", "unknown colour mode"),
    ("color Tag", "unknown colour mode"),
    ("every 0", "1..64"),
    ("every 65", "1..64"),
    ("every x", "non-negative integer"),
    ("focus", "takes 1"),
    ("focus -3", "non-negative integer"),
    ("focus 1.5", "non-negative integer"),
    ("focus 9223372036854775808", "int64"),
    ("follow abc", "non-negative integer"),
    ("labels maybe", "'on' or 'off'"),
    ("labels", "takes 1"),
    ("fields on", "Lab"),
    ("fields z 1", "Lab"),
    ("idnt now", "takes 0"),
    ("quit now", "takes 0"),
    ("autopause", "autopause takes"),
    ("autopause maybe", "takes 'on' or 'off'"),
    ("autopause chapter", "autopause takes"),
    ("autopause chapter maybe", "takes 'on' or 'off'"),
    ("autopause alert pop_fall", "autopause takes"),
    ("autopause alert Pop on", "alert class name"),
    ("autopause alert 1x on", "alert class name"),
    ("autopause alert pop_fall maybe", "takes 'on' or 'off'"),
    ("autopause on off", "autopause takes"),
    ("watch", "takes 1"),
    ("watch -1", "non-negative integer"),
    ("watch 1 2", "takes 1"),
    ("watch 9223372036854775808", "int64"),
    ("unwatch off", "non-negative integer"),
    ("unwatch", "takes 1"),
    ("ack x", "non-negative integer"),
    ("ack 9223372036854775808", "int64"),
    ("ack", "takes 1"),
    ("power", "power name"),
    ("power preview", "power name"),
    ("power cancel", "power name"),
    ("power tea id=1", "unknown power"),
    ("power food", "needs id=N"),
    ("power food mult=1.3", "needs id=N"),
    ("power food id=-1", "non-negative integer"),
    ("power food id=1.5", "non-negative integer"),
    ("power food id=1 id=2", "twice"),
    ("power food id=1 mult", "key=value"),
    ("power food id=1 speed=2", "key=value"),
    ("power food id=1 mult=fast", "decimal"),
    ("power food id=1 mult=1e1", "decimal"),
    ("power food id=1 mult=-1", "decimal"),
    ("power food id=1 steps=6000.5", "non-negative integer"),
    ("power food id=1 patch=-2", "non-negative integer"),
    ("power cancel food id=4 mult=1.3", "only id=N"),
    ("power FOOD id=1", "unknown power"),
    ("intent", "takes a text"),
    ("intent   ", "takes a text"),
]


@pytest.mark.parametrize(("line", "expected"), VALID)
def test_valid_lines_parse(line: str, expected: PlayCommand) -> None:
    assert parse_play_command(line) == expected


@pytest.mark.parametrize("line", BLANK)
def test_blank_lines_and_comments_are_nothing(line: str) -> None:
    assert parse_play_command(line) is None


@pytest.mark.parametrize(("line", "why"), INVALID)
def test_invalid_lines_are_refused(line: str, why: str) -> None:
    with pytest.raises(ControlError, match=why):
        parse_play_command(line)


def test_the_colour_modes_are_the_headers() -> None:
    assert streamrec.COLOR_MODES == ("tag", "energy", "clade", "house", "kin", "archetype", "intent")
    for mode in streamrec.COLOR_MODES:
        assert parse_play_command(f"color {mode}") == PlayCommand("color", mode)


def test_the_grammar_commands_are_the_spec_list() -> None:
    assert set(play.COMMANDS) == {
        *("pause", "run", "step", "rate", "color", "every", "focus", "follow", "idnt", "labels", "quit"),
        *("autopause", "watch", "unwatch", "ack", "power", "intent"),
    }
    assert set(play.UNBUILT_LINES) <= set(play.COMMANDS)


# --- the channel ------------------------------------------------------------------------------------------


def read_all(control: FdControl, expect: int, timeout: float = 5.0) -> list[str]:
    got: list[str] = []
    done = threading.Event()

    def pull() -> None:
        while len(got) < expect:
            got.extend(control.poll(0.05))
        done.set()

    t = threading.Thread(target=pull, daemon=True)
    t.start()
    assert done.wait(timeout), f"only {got}"
    return got


def test_a_line_over_4096_bytes_is_dropped_with_a_note(capfd: pytest.CaptureFixture[str]) -> None:
    r, w = os.pipe()
    control = FdControl(r)
    os.write(w, b"pause\n" + b"x" * 5000 + b"\nrun\n")
    os.close(w)
    lines = read_all(control, 3)
    assert lines == ["pause", "run", "quit"]  # the long line is gone and EOF reads as quit
    assert "4096" in capfd.readouterr().err


def test_the_limit_is_exact() -> None:
    r, w = os.pipe()
    control = FdControl(r)
    ok = "# " + "y" * (4096 - 2)
    os.write(w, ok.encode() + b"\n")
    os.close(w)
    assert read_all(control, 2) == [ok, "quit"]


def test_lines_split_across_reads_are_joined() -> None:
    r, w = os.pipe()
    control = FdControl(r)
    for part in (b"ra", b"te ", b"max\npa", b"use\n"):
        os.write(w, part)
    os.close(w)
    assert [parse_play_command(line) for line in read_all(control, 3)[:2]] == [
        PlayCommand("rate", "max"),
        PlayCommand("pause"),
    ]


# --- effects ------------------------------------------------------------------------------------------------


def run(tmp_path: Path, script, steps: int = 12, **kw):
    clock = kw.pop("clock", FakeClock())
    opt = kw.pop("options", play.PlayOptions(rate=1.0, paused=False, fresh=True))
    return run_paced(world("2d", **kw), tmp_path / "w", steps, script, clock=clock, options=opt)


def tags(out: io.BytesIO) -> list[tuple[str, int]]:
    return [(r.tag, r.step) for r in decoded(out).records]


def test_pause_and_run_set_the_reason_and_the_rate(tmp_path: Path) -> None:
    _, _, _, out = run(tmp_path, [[], [], ["pause"], [], [], ["run"]])
    cl = [r.f for r in decoded(out).records if r.tag == "CLCK"]
    assert [c["reason"] for c in cl if c["reason"] == 1] and any(c["rate"] == 0.0 for c in cl)
    assert cl[-1]["rate"] == 1.0 and cl[-1]["reason"] == 0


def test_a_pause_holds_the_state_and_a_run_goes_on(tmp_path: Path) -> None:
    clock = FakeClock()
    ctl = TimedControl(clock, [(1000.5, "pause"), (1003.0, "run")])
    _, _, _, out = run(tmp_path, ctl, steps=12, clock=clock)
    recs = decoded(out).records
    frames = [r.step for r in recs if r.tag == "FRMA"]
    assert frames == list(range(13))  # no step is written twice, none is lost
    held = [r for r in recs if r.tag == "CLCK" and r.f["reason"] == 1]
    assert len({r.step for r in held}) == 1 and 3 <= held[0].step <= 9  # paused at one step, for 2.5 s
    assert 9 <= len(held) <= 13 and clock.now - 1000.0 > 3.0 + 12 / B * 0.4


def test_step_runs_a_batch_and_writes_one_frame_after_it(tmp_path: Path) -> None:
    opt = play.PlayOptions(rate=1.0, paused=True, fresh=True)
    script = [["step 4"], [], [], [], [], ["step 2"], [], [], ["run"], ["rate max"]]
    _, _, _, out = run(tmp_path, script, steps=10, options=opt)
    frames = [r.step for r in decoded(out).records if r.tag == "FRMA"]
    assert frames == [0, 4, 6, 7, 8, 9, 10]  # the first state, one frame per batch, then one per step


def test_rate_sets_the_multiplier_and_zero_pauses(tmp_path: Path) -> None:
    script = [["rate 2"], [], [], ["rate 0.25"], [], [], ["rate 0"], [], [], ["rate max"]]
    _, _, _, out = run(tmp_path, script, steps=30)
    cl = [r.f for r in decoded(out).records if r.tag == "CLCK"]
    rates = [c["rate"] for c in cl]
    assert 2.0 in rates and 0.25 in rates and 0.0 in rates and -1.0 in rates
    assert next(c for c in cl if c["rate"] == 0.0)["reason"] == 1


def test_color_redraws_at_once_and_marks_the_mode(tmp_path: Path) -> None:
    opt = play.PlayOptions(rate=1.0, paused=True, fresh=True)
    script = [
        ["color energy", "color house", "color intent", "color kin", "color archetype"],
        ["run"],
        ["rate max"],
    ]
    _, _, _, out = run(tmp_path, script, steps=4, options=opt)
    idx = [r.f["color_index"] for r in decoded(out).records if r.tag == "FRMA"]
    assert idx[:6] == [0, 1, 3, 6, 4, 5]  # the first frame, then one per colour line, in order


def test_an_observer_colour_without_observers_is_refused(tmp_path: Path) -> None:
    notes: list[str] = []
    clock = FakeClock()
    cfg = world("2d", observe=False)
    rd = RunDir.create(tmp_path / "w", cfg)
    out = io.BytesIO()
    pacer = play.PlayPacer(
        out,
        ScriptControl([["color house", "color energy"]]),
        play.PlayOptions(rate=1.0, paused=False),
        clock=clock,
        note=notes.append,
    )
    runner.run(rd, cfg, init_state(cfg), 3, quiet=True, pacer=pacer)
    assert any("needs the observers" in n for n in notes)
    recs = decoded(out).records
    assert not [r for r in recs if r.tag in ("IDNT", "INTN", "TERR")]
    assert [r.f["color_index"] for r in recs if r.tag == "FRMA"][1] == 1  # energy took effect
    assert decoded(out).header["play"]["records"] == ["FRMA", "EVNT", "ENVS", "CLCK", "FOCS"]


def test_invalid_lines_are_noted_and_ignored(tmp_path: Path) -> None:
    notes: list[str] = []
    clock = FakeClock()
    cfg = world("2d")
    rd = RunDir.create(tmp_path / "w", cfg)
    pacer = play.PlayPacer(
        io.BytesIO(), ScriptControl([["bogus", "rate 99", "fields on", "pause"], ["run"]]),
        play.PlayOptions(rate=1.0, paused=False), clock=clock, note=notes.append,
    )  # fmt: skip
    res = runner.run(rd, cfg, init_state(cfg), 3, quiet=True, pacer=pacer)
    assert res.status == "done"
    assert len(notes) == 3 and "fields" in notes[2] and all("ignored control line" in n for n in notes)


def test_the_decide_half_lines_are_noted_until_their_stage_and_change_nothing(tmp_path: Path) -> None:
    """``watch``, ``unwatch``, ``ack``, ``power`` and ``intent`` parse (the window sends them) and are
    answered with a note until the stage that builds their handler; they must not reach the focus branch or
    change the stream."""
    notes: list[str] = []
    lines = [
        "watch 3",
        "unwatch 3",
        "ack 466250600",
        "power food id=1 mult=1.3",
        "power preview strike id=2",
        "intent more food",
    ]
    script = [lines, [], ["rate max"]]
    clock = FakeClock()
    cfg = world("2d")
    rd = RunDir.create(tmp_path / "w", cfg)
    out = io.BytesIO()
    pacer = play.PlayPacer(
        out, ScriptControl(script), play.PlayOptions(rate=1.0, paused=False), clock=clock, note=notes.append
    )
    runner.run(rd, cfg, init_state(cfg), 4, quiet=True, pacer=pacer)
    assert [n.split(":")[0] for n in notes] == [ln.split()[0] for ln in lines]
    assert all("not available in this build yet" in n for n in notes)
    _, _, _, plain = run(tmp_path / "plain", [[], [], ["rate max"]], steps=4)
    key = lambda o: [(r.tag, r.step) for r in decoded(o).records if r.tag not in ("CLCK",)]  # noqa: E731
    assert key(out) == key(plain)


def test_every_sets_the_frame_period_and_clck_reports_it(tmp_path: Path) -> None:
    opt = play.PlayOptions(rate=float("inf"), paused=False)
    _, _, _, out = run(tmp_path, [["every 7"]], steps=20, options=opt)
    recs = decoded(out).records
    assert [r.step for r in recs if r.tag == "FRMA"] == [0, 7, 14]
    assert [r.f["fe"] for r in recs if r.tag == "CLCK"][-1] == 7


def test_labels_off_stops_intn_and_on_resumes_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    snapshots(monkeypatch, 10)
    script = [["labels off"], *[[]] * 30, ["labels on"]]
    opt = play.PlayOptions(rate=float("inf"), paused=False, fresh=True, tick=25)
    _, _, _, out = run(tmp_path, script, steps=100, options=opt)
    got = [r.step for r in decoded(out).records if r.tag == "INTN"]
    assert got == [0, 50, 75, 100]  # off from the first line to step 31: the tick at 25 sends nothing


def test_focus_of_an_unknown_uid_answers_once_with_the_alive_flag_clear(tmp_path: Path) -> None:
    _, _, _, out = run(
        tmp_path, [["focus 987654321"]], steps=6, options=play.PlayOptions(rate=float("inf"), paused=False)
    )
    focs = [r.f for r in decoded(out).records if r.tag == "FOCS"]
    assert len(focs) == 1 and focs[0]["uid"] == 987654321 and focs[0]["flags"] == 0


def test_idnt_resends_the_identity_records_now(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    snapshots(monkeypatch, 10)
    clock = FakeClock()
    ctl = TimedControl(clock, [(1001.0, "idnt")])
    _, _, _, out = run(tmp_path, ctl, steps=20, clock=clock)
    recs = decoded(out).records
    at = next(i for i, r in enumerate(recs) if r.tag == "IDNT" and r.step in (14, 15))
    assert [r.tag for r in recs[at - 1 : at + 5]] == ["FRMA", "IDNT", "INTN", "TERR", "ENVS", "FRMA"]
    assert recs[at].step == recs[at - 1].step  # at the step the line was read


def test_the_lines_of_a_chunk_act_as_if_they_came_one_by_one(tmp_path: Path) -> None:
    """While paused, lines delivered together or one per poll give the same records (apart from the clock)."""
    lines = ["color energy", "idnt", "focus off", "color house", "idnt", "labels off", "color tag"]
    opt = play.PlayOptions(rate=1.0, paused=True, fresh=True)
    quiet = [["run"], ["rate max"]]
    _, _, _, a = run(tmp_path / "a", [lines, *quiet], steps=2, options=opt)
    _, _, _, b = run(tmp_path / "b", [*[[ln] for ln in lines], *quiet], steps=2, options=opt)

    def key(out: io.BytesIO) -> list[tuple[str, int, int | None]]:
        return [(r.tag, r.step, r.f.get("color_index")) for r in decoded(out).records if r.tag != "CLCK"]

    assert key(a) == key(b)


def test_quit_drops_the_lines_after_it(tmp_path: Path) -> None:
    """``quit`` ends the batch of lines: the colour line after it is not applied, and one more step runs."""
    _, res, pacer, out = run(tmp_path, [["rate max", "quit", "color house", "idnt"]], steps=10)
    recs = decoded(out).records
    assert res.status == "interrupted" and pacer.stop_requested
    assert {r.f["color_index"] for r in recs if r.tag == "FRMA"} == {0}
    assert max(r.step for r in recs) == 1 and [r.tag for r in recs].count("IDNT") == 1  # only the bind's
