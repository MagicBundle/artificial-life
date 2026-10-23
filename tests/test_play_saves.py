"""The save mode of a played world: ``saves all|newest`` (DESIGN §34.10).

``all`` keeps the last ``run.keep_checkpoints`` saves and the origin (the default); ``newest`` keeps the
newest save and the origin, so a long session does not fill the disk. The switch deletes nothing by itself:
the next save writes its file, names it in ``LATEST`` and only then prunes. The mode is kept in ``play.json``
(absent: ``all``), logged to ``commands.jsonl`` and echoed to the client (``play.saves`` in the header,
``CLCK.flags`` bit 2). It is storage only: a world whose mode is toggled at random is the plain run on every
digest, row, event chunk and log it shares with it."""

from __future__ import annotations

import dataclasses
import io
import json
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import pytest

from alife import cli, play, runio, runner, streamrec
from alife.config import Config, apply_overrides
from alife.director import actions, make_director, pacing
from alife.game import powers
from alife.runio import COMMANDS_FILE, DIRECTOR_LOG_FILE, CheckpointError, RunDir, load_checkpoint
from alife.state import init_state, state_digest
from alife.stream import ControlError, NoControl
from play_decode import Decoded, decode
from play_helpers import (
    FakeClock,
    ScriptControl,
    checkpoint_digests,
    decoded,
    director_log,
    rows,
    run_paced,
    run_plain,
    same_events,
    same_sidecar,
    snapshots,
    social_text,
    spawn,
    world,
)

END = 150  # five checkpoints of the 30-step cadence of ``play_helpers.CADENCE``
TINY = ["--preset", "tiny", "--set", "brain.weight_dtype=float32"]


def at(lines: Mapping[int, Sequence[str]], n: int = END) -> list[list[str]]:
    """A control script with ``lines[t]`` at the top of iteration ``t`` of a running world (a poll a step)."""
    return [list(lines.get(t, ())) for t in range(n)]


def steps_of(rd: RunDir) -> list[int]:
    return [st for st, _ in rd.checkpoints()]


def commands(rd: RunDir) -> list[dict[str, Any]]:
    return rd.read_jsonl(COMMANDS_FILE) if (rd.path / COMMANDS_FILE).exists() else []


def clock_flags(out: io.BytesIO) -> list[int]:
    return [r.f["flags"] for r in decoded(out).records if r.tag == "CLCK"]


# --- the grammar --------------------------------------------------------------------------------------


def test_saves_takes_all_or_newest() -> None:
    assert play.parse_play_command("saves all") == play.PlayCommand("saves", "all")
    assert play.parse_play_command("saves newest") == play.PlayCommand("saves", "newest")
    assert play.parse_play_command("  saves   newest  ") == play.PlayCommand("saves", "newest")
    assert "saves" in play.COMMANDS and play.SAVE_MODES == ("all", "newest") == streamrec.SAVE_MODES
    for bad in ("saves", "saves maybe", "saves NEWEST", "saves all newest", "saves 1", "saves off"):
        with pytest.raises(ControlError, match="saves takes one of all, newest"):
            play.parse_play_command(bad)


# --- the files: pruning in both modes -----------------------------------------------------------------


def test_newest_keeps_the_origin_and_the_newest_and_all_keeps_the_last_ones(
    micro_cfg: Config, tmp_path: Path
) -> None:
    """A fork's origin (``start_step``) is never pruned, in either mode; a sidecar and a play-state file live
    as long as their checkpoint; and the newest save is in place before anything older goes."""
    cfg = apply_overrides(micro_cfg, ["run.keep_checkpoints=2"])
    rd = RunDir.create(tmp_path / "fork", cfg, parent="src:20")
    rd.update_meta(start_step=20)
    s = init_state(cfg)
    obs = {"x": np.arange(3)}

    def save(step: int) -> None:
        rd.write_play_state("kinthreads", step, {"step": step})
        rd.save_checkpoint(s._replace(step=mx.array(step, dtype=mx.int32)), cfg, observers=obs)

    def held() -> tuple[list[int], list[int], list[int]]:
        return (
            steps_of(rd),
            [st for st, _ in rd.observer_files()],
            [st for st, _ in rd.play_state_files("kinthreads")],
        )

    for step in (20, 30, 40):  # all: the origin and the newest two
        save(step)
    assert held() == ([20, 30, 40],) * 3
    rd.keep_newest = True
    assert steps_of(rd) == [20, 30, 40], "switching deletes nothing"
    save(50)  # the next save prunes
    assert held() == ([20, 50],) * 3
    save(51)  # a quit save replaces the chapter-end save
    assert held() == ([20, 51],) * 3
    assert rd.latest_checkpoint() == rd.checkpoint_at(51)
    rd.keep_newest = False  # and back: saves are kept from then on
    save(60)
    assert held() == ([20, 51, 60],) * 3
    save(70)
    assert held() == ([20, 60, 70],) * 3

    # a world from step 0: step 0 is its origin
    rd0 = RunDir.create(tmp_path / "zero", cfg)
    rd0.keep_newest = True
    for step in (0, 10, 20, 30):
        rd0.save_checkpoint(s._replace(step=mx.array(step, dtype=mx.int32)), cfg)
    assert steps_of(rd0) == [0, 30]


def test_a_new_run_dir_is_in_the_default_mode(micro_cfg: Config, tmp_path: Path) -> None:
    assert RunDir.create(tmp_path / "w", micro_cfg).keep_newest is False
    assert RunDir.open(tmp_path / "w").keep_newest is False


# --- the pacer ----------------------------------------------------------------------------------------


def paced(
    tmp_path: Path, lines: Mapping[int, Sequence[str]], *, name: str = "w", steps: int = END, **kw: Any
) -> tuple[RunDir, play.PlayPacer, io.BytesIO]:
    opts = play.PlayOptions(rate=float("inf"), paused=False, fresh=True, unattended=True, **kw)
    rd, res, pacer, out = run_paced(world("2d"), tmp_path / name, steps, at(lines, steps), options=opts)
    assert res.status == "done" and pacer.failure is None
    return rd, pacer, out


def test_a_world_in_newest_mode_keeps_the_origin_and_the_newest_save(tmp_path: Path) -> None:
    plain, _ = run_plain(world("2d"), tmp_path / "plain", END)
    assert steps_of(plain) == [0, 90, 120, 150]
    rd, pacer, _ = paced(tmp_path, {5: ["saves newest"]})
    assert steps_of(rd) == [0, 150] and pacer.saves == "newest" and rd.keep_newest
    assert [t for t, _ in rd.observer_files()] == [0, 150], "the sidecars follow the saves"
    assert rd.latest_checkpoint() == rd.checkpoint_at(150)
    # the other way round: a world that starts in newest mode (play.json says so)
    rd2, _, _ = paced(tmp_path, {}, name="started", saves="newest")
    assert steps_of(rd2) == [0, 150]


def test_the_switch_deletes_nothing_and_the_next_save_applies_it(tmp_path: Path) -> None:
    seen: list[tuple[str, list[int]]] = []

    class Spy(play.PlayPacer):
        def _set_saves(self, mode: str) -> None:
            before = steps_of(self.rd)
            super()._set_saves(mode)
            seen.append((mode, before))
            assert steps_of(self.rd) == before, "the line itself deletes no save"

    opts = play.PlayOptions(rate=float("inf"), paused=False, fresh=True, unattended=True)
    script = at({65: ["saves newest"], 100: ["saves all"], 101: ["saves all"]})
    rd, res, pacer, _ = run_paced(world("2d"), tmp_path / "w", END, script, options=opts, pacer_class=Spy)
    assert res.status == "done" and pacer.failure is None
    assert seen == [("newest", [0, 30, 60]), ("all", [0, 90]), ("all", [0, 90])], seen
    # newest up to the save at 90, all again from 100: 120 and 150 are added to 90
    assert steps_of(rd) == [0, 90, 120, 150]


def test_the_header_and_the_clock_say_what_the_mode_is(tmp_path: Path) -> None:
    _, _, out = paced(tmp_path, {40: ["saves newest"], 100: ["saves all"], 110: ["saves bogus"]})
    d = decoded(out)
    assert d.header["play"]["saves"] == {"mode": "all", "keep": 3}
    flags = clock_flags(out)
    newest = [bool(f & streamrec.CLCK_FLAG_SAVES_NEWEST) for f in flags]
    assert newest[0] is False and True in newest and newest[-1] is False
    first, last = newest.index(True), len(newest) - 1 - newest[::-1].index(True)
    assert all(newest[first : last + 1]), "the flag stays up while the mode is newest"
    _, _, out2 = paced(tmp_path, {}, name="started", saves="newest")
    d2 = decoded(out2)
    assert d2.header["play"]["saves"] == {"mode": "newest", "keep": 3}
    assert all(f & streamrec.CLCK_FLAG_SAVES_NEWEST for f in clock_flags(out2))


def test_a_change_is_logged_and_a_line_that_changes_nothing_is_not(tmp_path: Path) -> None:
    rd, _, _ = paced(
        tmp_path, {20: ["saves all"], 40: ["saves newest"], 41: ["saves newest"], 80: ["saves all"]}
    )
    assert [(c["t"], c["verb"], c["status"], c["mode"], c["was"]) for c in commands(rd)] == [
        (40, "saves", "applied", "newest", "all"),
        (80, "saves", "applied", "all", "newest"),
    ]


def test_the_mode_is_kept_in_play_json_and_absent_for_all(micro_cfg: Config, tmp_path: Path) -> None:
    rd = RunDir.create(tmp_path / "w", micro_cfg)
    pacer = play.PlayPacer(
        io.BytesIO(), NoControl(), play.PlayOptions(), clock=FakeClock(), note=lambda m: None
    )
    pacer.rd, pacer.cfg, pacer._t = rd, micro_cfg, 7
    pacer._set_saves("newest")  # no play.json (not a world made by `alife play`): nothing to keep it in
    assert rd.read_play_json() is None and rd.keep_newest and commands(rd)[0]["t"] == 7
    rd.write_play_json({"base_rate": 14.4, "tick": 25})
    pacer._set_saves("all")
    pacer._set_saves("newest")
    assert rd.read_play_json() == {"base_rate": 14.4, "tick": 25, "play_version": 1, "saves": "newest"}
    pacer._set_saves("all")
    assert rd.read_play_json() == {"base_rate": 14.4, "tick": 25, "play_version": 1}
    assert not rd.keep_newest


def test_a_replay_of_the_log_writes_no_input_line(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(play, "replaying", lambda director: True)
    opts = play.PlayOptions(rate=float("inf"), paused=False, fresh=True, unattended=True)
    rd, _, pacer, _ = run_paced(world("2d"), tmp_path / "w", 60, at({10: ["saves newest"]}, 60), options=opts)
    assert pacer.saves == "newest" and commands(rd) == []


# --- the trajectory -----------------------------------------------------------------------------------


def random_toggles(seed: int, steps: int) -> dict[int, list[str]]:
    rng = random.Random(seed)
    lines: dict[int, list[str]] = {}
    for _ in range(rng.randint(3, 9)):
        lines.setdefault(rng.randrange(steps), []).append(
            rng.choice(["saves all", "saves newest", "saves x"])
        )
    return lines


@pytest.mark.parametrize(
    ("kind", "observe", "seed"),
    [("2d", True, 1), ("2d", False, 2), ("3d", True, 3), ("3d", False, 4)],
    ids=["2d-observed", "2d-plain", "3d-observed", "3d-plain"],
)
def test_a_toggled_run_equals_a_plain_run(
    kind: str, observe: bool, seed: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The mode changes which saves stay on disk and nothing else: the state, the metrics rows, the event
    chunks, the social log, the director log and every sidecar that is still there are the plain run's."""
    snapshots(monkeypatch, 10)
    cfg = world(kind, observe=observe)
    plain, res_plain = run_plain(cfg, tmp_path / "plain", END)
    toggles = random_toggles(seed, END)
    opts = play.PlayOptions(rate=float("inf"), paused=False, fresh=True, unattended=True)
    toggled, res, pacer, _ = run_paced(cfg, tmp_path / "toggled", END, at(toggles), options=opts)
    assert pacer.failure is None and res.status == res_plain.status == "done"
    assert state_digest(res.state) == state_digest(res_plain.state)
    assert rows(plain) == rows(toggled) and same_events(plain, toggled)
    assert social_text(plain) == social_text(toggled) and director_log(plain) == director_log(toggled)
    kept, all_of_them = steps_of(toggled), steps_of(plain)
    assert set(kept) <= set(all_of_them) and kept[-1] == END and 0 in kept
    digests, plain_digests = checkpoint_digests(toggled), checkpoint_digests(plain)
    assert all(digests[st] == plain_digests[st] for st in kept)
    if observe:
        for st in kept:
            assert same_sidecar(plain, toggled, st)
        assert [st for st, _ in toggled.observer_files()] == kept
    # the toggles that changed the mode are the lines of commands.jsonl, in order
    modes = [m for line in sorted(toggles) for m in toggles[line] if m in ("saves all", "saves newest")]
    state = "saves all"
    changes = []
    for m in modes:
        if m != state:
            changes.append(m.split()[1])
            state = m
    assert [c["mode"] for c in commands(toggled)] == changes


# --- a player's power at a chapter end, and a resume --------------------------------------------------


@pytest.fixture
def small_powers(monkeypatch: pytest.MonkeyPatch) -> None:
    """The game table of ``test_play_powers``: durations of 30 to 120 steps, cooldowns of 20."""
    for name, spec in list(actions.GAME_SPECS.items()):
        small: dict[str, Any] = {"cooldown": 20}
        if spec.duration is not None:
            small |= {"duration": (30, 120), "quantum": 10}
        monkeypatch.setitem(actions.GAME_SPECS, name, dataclasses.replace(spec, **small))
    monkeypatch.setattr(pacing, "PLAYER_BUDGET_WINDOW", 400)
    monkeypatch.setattr(powers, "SINCE_AFTER", (20, 50))


@pytest.mark.usefixtures("small_powers")
def test_a_power_at_a_chapter_end_and_a_resume_in_newest_mode(tmp_path: Path) -> None:
    """Rule P (a power at the checkpoint step is applied after it) does not depend on how many saves are kept:
    in newest mode the chapter-end save is replaced by the quit save, and a resume from that one continues to
    the uninterrupted digest with the same director log."""
    cfg = apply_overrides(world("3d", director=True), ["run.director=player"])
    opts = play.PlayOptions(rate=float("inf"), paused=False, fresh=True, chapter=30, unattended=True)
    lines = {5: ["saves newest"], 30: ["power food id=1 mult=1.3 steps=40"]}
    full, res, *_ = run_paced(cfg, tmp_path / "full", 60, at(lines, 60), options=opts, game_specs=True)
    assert res.status == "done" and steps_of(full) == [0, 60]
    [entry] = [e for e in director_log(full) if e.get("source") == "player"]
    assert entry["step_applied"] == 30
    # a session that stops at 45 and resumes (the quit save at 45 replaced the one at 30)
    part, res1, *_ = run_paced(cfg, tmp_path / "part", 45, at(lines, 45), options=opts, game_specs=True)
    assert res1.status == "done" and steps_of(part) == [0, 45]
    assert [e["step_applied"] for e in director_log(part) if e.get("source") == "player"] == [30]
    part.truncate_after(45)
    state = load_checkpoint(part.latest_checkpoint(), cfg)  # type: ignore[arg-type]
    resumed = play.PlayPacer(
        io.BytesIO(),
        ScriptControl([]),
        play.PlayOptions(rate=float("inf"), paused=False, unattended=True, chapter=30, saves="newest"),
        clock=FakeClock(),
        note=lambda m: None,
    )
    resumed_res = runner.run(
        part,
        cfg,
        state,
        15,
        quiet=True,
        director=make_director(part, cfg, 45, game_specs=True),
        pacer=resumed,
    )
    resumed.close()
    assert resumed_res.status == "done" and steps_of(part) == [0, 60]
    assert state_digest(resumed_res.state) == state_digest(res.state)
    assert (part.path / DIRECTOR_LOG_FILE).read_bytes() == (full.path / DIRECTOR_LOG_FILE).read_bytes()
    assert checkpoint_digests(part)[60] == checkpoint_digests(full)[60]


# --- the chapter end and its report ---------------------------------------------------------------------


def test_the_reports_of_a_world_in_newest_mode_are_the_same_and_verify_from_its_logs(tmp_path: Path) -> None:
    """The report job reads the sidecar of a chapter's last step inside its save and the logs; the saves of
    earlier chapters are not an input. Three chapters of 500 steps in either mode give the same
    ``reports.jsonl``, and the reports of the world that kept the newest save only verify from its files."""
    from alife.analysis import world_report as W
    from alife.game import report as R

    cfg = world("2d", "run.metrics_every=250", "run.heavy_metrics_every=250", "run.checkpoint_every=500")

    def play_world(name: str, saves: str) -> RunDir:
        rd = RunDir.create(tmp_path / name, cfg)
        state = init_state(cfg)
        rd.update_meta(origin_row=play.origin_row(state, cfg))
        opts = play.PlayOptions(
            rate=float("inf"), paused=False, fresh=True, unattended=True, chapter=500, saves=saves
        )
        pacer = play.PlayPacer(io.BytesIO(), ScriptControl([]), opts, clock=FakeClock(), note=lambda m: None)
        try:
            runner.run(rd, cfg, state, 1500, quiet=True, pacer=pacer)
        finally:
            pacer.close()
        return rd

    everything, newest = play_world("all", "all"), play_world("newest", "newest")
    assert steps_of(everything) == [0, 500, 1000, 1500] and steps_of(newest) == [0, 1500]
    assert [r["step"] for r in newest.reports()] == [500, 1000, 1500]
    assert newest.reports() == everything.reports()
    code, summary = W.verify_reports(R.load_run(newest.path), newest.reports())
    assert code == 0 and summary["counts"] == {"ok": 3}, summary


# --- the command --------------------------------------------------------------------------------------


def session(*args: str, lines: str = "", until: int) -> tuple[int, Decoded, str]:
    """``alife play ARGS --rate max --until N`` with ``lines`` on its stdin, which stays open: the world ends
    by itself at its step (a closed stdin is a quit)."""
    p = spawn("play", *args, "--rate", "max", "--until", str(until))
    assert p.stdin is not None and p.stdout is not None and p.stderr is not None
    p.stdin.write(lines.encode())
    p.stdin.flush()
    data = p.stdout.read()
    code = p.wait(timeout=240)
    err = p.stderr.read().decode(errors="replace")
    p.stdin.close()
    return code, decode(data) if data else Decoded({}, [], []), err


def test_the_mode_survives_a_resume_and_the_flag_and_the_line_change_it(tmp_path: Path) -> None:
    out = tmp_path / "w"
    code, d, err = session(*TINY, "--out", str(out), until=10)
    assert code == 0, err
    assert d.header["play"]["saves"] == {"mode": "all", "keep": 3}
    rd = RunDir.open(out)
    assert steps_of(rd) == [0, 10] and "saves" not in rd.read_play_json()  # type: ignore[operator]
    # the player switches recording off; the quit save at 20 replaces the one at 10
    code, d, err = session(str(out), lines="saves newest\n", until=20)
    assert code == 0, err
    rd = RunDir.open(out)
    assert steps_of(rd) == [0, 20] and rd.read_play_json()["saves"] == "newest"  # type: ignore[index]
    assert [(c["verb"], c["mode"], c["was"]) for c in commands(rd)] == [("saves", "newest", "all")]
    assert any(r.tag == "CLCK" and r.f["flags"] & streamrec.CLCK_FLAG_SAVES_NEWEST for r in d.records)
    # a resume starts in the mode the world was left in; the header says so
    code, d, err = session(str(out), until=30)
    assert code == 0, err
    assert d.header["play"]["saves"] == {"mode": "newest", "keep": 3}
    rd = RunDir.open(out)
    assert steps_of(rd) == [0, 30] and rd.read_play_json()["saves"] == "newest"  # type: ignore[index]
    # back to all with the line: saves are kept from then on, and play.json forgets the key
    code, _, err = session(str(out), lines="saves all\n", until=40)
    assert code == 0, err
    rd = RunDir.open(out)
    assert steps_of(rd) == [0, 30, 40] and "saves" not in rd.read_play_json()  # type: ignore[operator]
    # the flag overrides play.json for the session and is kept
    code, _, err = session(str(out), "--saves", "newest", until=50)
    assert code == 0, err
    rd = RunDir.open(out)
    assert steps_of(rd) == [0, 50] and rd.read_play_json()["saves"] == "newest"  # type: ignore[index]


def test_a_play_json_with_another_mode_is_refused_before_anything_is_touched(tmp_path: Path) -> None:
    out = tmp_path / "w"
    code, _, err = session(*TINY, "--out", str(out), until=10)
    assert code == 0, err
    rd = RunDir.open(out)
    rd.write_play_json({**(rd.read_play_json() or {}), "saves": "some"})
    before = sorted(p.name for p in out.rglob("*") if p.is_file())
    code, _, err = session(str(out), until=20)
    assert code == 2 and "saves is one of" in err
    assert sorted(p.name for p in out.rglob("*") if p.is_file()) == before
    assert json.loads((out / "play.json").read_text())["saves"] == "some"
    assert cli.main(["play", str(out), "--saves", "maybe"]) == 2


def test_a_play_json_that_cannot_be_written_does_not_stop_the_world(
    micro_cfg: Config, tmp_path: Path
) -> None:
    rd = RunDir.create(tmp_path / "w", micro_cfg)
    rd.write_play_json({"base_rate": 14.4})
    notes: list[str] = []
    pacer = play.PlayPacer(
        io.BytesIO(), NoControl(), play.PlayOptions(), clock=FakeClock(), note=notes.append
    )
    pacer.rd, pacer.cfg = rd, micro_cfg
    rd.play_path.write_text("not json")  # (a torn file: reading it raises)
    pacer._set_saves("newest")
    assert pacer.saves == "newest" and rd.keep_newest
    assert any("could not keep the mode" in n for n in notes)


# --- a research run played in place (review fixes) ------------------------------------------------------


def research_run(path: Path, steps: int = 90, keep: int = 10) -> Path:
    args = ["run", *TINY, "--out", str(path), "--steps", str(steps), "--checkpoint-every", "30", "--quiet"]
    assert cli.main([*args, "--set", f"run.keep_checkpoints={keep}"]) == 0
    return path


def test_adopting_with_saves_newest_says_what_is_deleted(tmp_path: Path) -> None:
    """The note of ``--adopt`` used to say that nothing is pruned at the first save in every mode; with
    ``--saves newest`` the run's checkpoints go, and the note says which, before anything is touched."""
    out = research_run(tmp_path / "r")
    assert steps_of(RunDir.open(out)) == [0, 30, 60, 90]
    code, d, err = session(str(out), "--adopt", "--saves", "newest", until=100)
    assert code == 0, err
    assert "nothing is pruned" not in err
    assert "3 checkpoint(s) of the research run (steps 30 to 90) are DELETED at the first save" in err, err
    assert steps_of(RunDir.open(out)) == [0, 100]
    assert d.header["play"]["saves"] == {"mode": "newest", "keep": 10, "adopted": True}
    # in the default mode the note is the one it was
    out2 = research_run(tmp_path / "r2")
    code, _, err = session(str(out2), "--adopt", until=100)
    assert code == 0, err
    assert "nothing is pruned at the first save" in err and "DELETED" not in err


def test_the_flag_on_a_later_session_of_an_adopted_run_says_what_is_deleted(tmp_path: Path) -> None:
    """Once adopted, the directory has a ``play.json`` and is resumed without ``--adopt``; ``--saves newest``
    there still deletes the research's checkpoints at the first save, and says so (it used to be silent)."""
    out = research_run(tmp_path / "r")
    code, _, err = session(str(out), "--adopt", until=100)
    assert code == 0, err
    assert steps_of(RunDir.open(out)) == [0, 30, 60, 90, 100]
    code, _, err = session(str(out), "--saves", "newest", until=110)
    assert code == 0, err
    assert "4 checkpoint(s) of the research run (steps 30 to 100) are DELETED at the first save" in err, err
    assert steps_of(RunDir.open(out)) == [0, 110]
    # a world that is already in newest mode says nothing more: its first save replaces the newest, as usual
    code, _, err = session(str(out), "--saves", "newest", until=120)
    assert code == 0, err
    assert "DELETED" not in err and steps_of(RunDir.open(out)) == [0, 120]


@pytest.mark.parametrize(("steps", "keep"), [(90, 10), (30, 3)])
def test_a_saves_newest_line_does_not_delete_an_adopted_runs_checkpoints(
    tmp_path: Path, steps: int, keep: int
) -> None:
    """The line is refused in a research run played in place: its checkpoints are read by ``init-from
    RUN:STEP``, the houses and the activity analyses. ``(30, 3)`` is a run with the play value of
    ``keep_checkpoints`` and one checkpoint, which a rule on ``keep_checkpoints`` would not have caught; the
    header says ``adopted`` so that the window does not offer the switch."""
    out = research_run(tmp_path / "r", steps=steps, keep=keep)
    before = steps_of(RunDir.open(out))
    code, d, err = session(str(out), "--adopt", lines="saves newest\n", until=steps + 10)
    assert code == 0, err
    rd = RunDir.open(out)
    assert steps_of(rd) == [*before, steps + 10], "nothing was pruned"
    assert "saves: refused" in err, err
    assert rd.adopted
    assert "saves" not in (rd.read_play_json() or {}) and commands(rd) == []
    assert d.header["play"]["saves"]["adopted"] is True
    assert not any(r.tag == "CLCK" and r.f["flags"] & streamrec.CLCK_FLAG_SAVES_NEWEST for r in d.records)


def test_a_play_world_still_takes_the_line(tmp_path: Path) -> None:
    out = tmp_path / "w"
    code, d, err = session(*TINY, "--out", str(out), lines="saves newest\n", until=40)
    assert code == 0, err
    rd = RunDir.open(out)
    assert rd.read_play_json()["saves"] == "newest"  # type: ignore[index]
    assert d.header["play"]["saves"] == {"mode": "all", "keep": 3}, "no adopted key"
    assert not rd.adopted


def test_an_adopted_world_in_newest_mode_can_go_back_to_all(micro_cfg: Config, tmp_path: Path) -> None:
    """Only the line that would delete is refused: ``saves all`` (keep more) is taken, and a ``saves newest``
    line that changes nothing does nothing, refused or not."""
    rd = RunDir.create(tmp_path / "w", micro_cfg)
    (rd.path / runio.PRE_PLAY_CONFIG_FILE).write_text("{}")
    notes: list[str] = []
    opts = play.PlayOptions(saves="newest")
    pacer = play.PlayPacer(io.BytesIO(), NoControl(), opts, clock=FakeClock(), note=notes.append)
    pacer.rd, pacer.cfg, pacer._t = rd, micro_cfg, 7
    pacer._set_saves("newest")
    assert pacer.saves == "newest" and not notes
    pacer._set_saves("all")
    assert pacer.saves == "all" and not rd.keep_newest
    pacer._set_saves("newest")
    assert pacer.saves == "all" and not rd.keep_newest and "saves: refused" in notes[-1]


# --- the files: a bad new save and the debris of a killed one (review fixes) --------------------------


def test_a_new_save_that_is_not_whole_does_not_cost_the_last_good_one(
    micro_cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A write that returns normally and leaves a short file used to prune the last good save in newest
    mode; the check of the new file makes the prune the ``all`` one instead, and says so on stderr."""
    rd = RunDir.create(tmp_path / "w", micro_cfg, parent="src:10")
    rd.update_meta(start_step=10)
    rd.keep_newest = True
    s = init_state(micro_cfg)

    def save(step: int) -> None:
        rd.save_checkpoint(s._replace(step=mx.array(step, dtype=mx.int32)), micro_cfg)

    save(10)
    save(20)
    assert steps_of(rd) == [10, 20]
    real = runio.write_state_file

    def short(path: Path, state: Any, cfg: Config) -> None:
        real(path, state, cfg)
        data = path.read_bytes()
        path.write_bytes(data[: len(data) // 3])

    monkeypatch.setattr(runio, "write_state_file", short)
    save(30)
    monkeypatch.undo()
    assert steps_of(rd) == [10, 20, 30], "the last good save is still there"
    assert "does not read back whole; the older saves are kept" in capsys.readouterr().err
    with pytest.raises(CheckpointError):
        load_checkpoint(rd.latest_checkpoint(), micro_cfg)  # type: ignore[arg-type]
    load_checkpoint(rd.checkpoint_at(20), micro_cfg)  # type: ignore[arg-type]
    assert runio.state_file_is_complete(rd.checkpoint_at(20), 20)  # type: ignore[arg-type]
    assert not runio.state_file_is_complete(rd.checkpoint_at(20), 21), "another step"  # type: ignore[arg-type]
    assert not runio.state_file_is_complete(rd.checkpoint_at(30), 30)  # type: ignore[arg-type]
    assert not runio.state_file_is_complete(tmp_path / "missing", 20)
    save(40)  # a whole save: back to the newest only
    assert steps_of(rd) == [10, 40]


def test_newest_mode_puts_the_new_save_on_the_drive_before_it_deletes_the_old_one(
    micro_cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``os.fsync`` on macOS leaves the data in the drive's cache, where a power cut loses it; in newest mode
    the old save is the only fallback, so the new one is flushed past the cache (``F_FULLFSYNC``) while the
    old one still exists. ``all`` keeps older saves and pays nothing more."""
    rd = RunDir.create(tmp_path / "w", micro_cfg)
    s = init_state(micro_cfg)
    synced: list[tuple[str, list[int]]] = []
    real = runio._full_sync

    def spy(path: Path) -> None:
        synced.append((path.name, steps_of(rd)))
        real(path)

    monkeypatch.setattr(runio, "_full_sync", spy)
    for step in (0, 10, 20):
        rd.save_checkpoint(s._replace(step=mx.array(step, dtype=mx.int32)), micro_cfg)
    assert synced == [], "all mode"
    rd.keep_newest = True
    rd.save_checkpoint(s._replace(step=mx.array(30, dtype=mx.int32)), micro_cfg)
    assert synced == [("step_0000000030.safetensors", [0, 10, 20, 30])], "flushed while the old saves exist"
    assert steps_of(rd) == [0, 30]
    runio._full_sync(tmp_path / "missing")  # best effort: a file that is gone is no error


def test_a_tmp_file_of_a_killed_save_is_removed_by_the_next_save(micro_cfg: Config, tmp_path: Path) -> None:
    """The temporary files a save writes and a kill left behind are removed at the next save (a state file's
    is up to the size of a save); a ``*.tmp`` that is not a save's (a houses file an analysis may be
    writing) is left alone."""
    rd = RunDir.create(tmp_path / "w", micro_cfg)
    rd.keep_newest = True
    s = init_state(micro_cfg)
    rd.save_checkpoint(s._replace(step=mx.array(0, dtype=mx.int32)), micro_cfg, observers={"x": np.arange(2)})
    stray = [
        rd.checkpoint_dir / "step_0000000050.safetensors.tmp",
        rd.checkpoint_dir / "LATEST.tmp",
        rd.observers_dir / "step_0000000050.npz.tmp",
        rd.observers_dir / "kinthreads_0000000050.json.tmp",
        rd.observers_dir / "storms_0000000050.json.tmp",
    ]
    other = rd.observers_dir / "houses_0000000050.npz.tmp"
    for p in [*stray, other]:
        p.write_bytes(b"x" * 100)
    rd.save_checkpoint(s._replace(step=mx.array(100, dtype=mx.int32)), micro_cfg)
    assert not any(p.exists() for p in stray)
    assert other.exists()
    assert sorted(p.name for p in rd.checkpoint_dir.iterdir()) == [
        "LATEST",
        "step_0000000000.safetensors",
        "step_0000000100.safetensors",
    ]
