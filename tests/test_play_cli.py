"""``alife play`` as a process (DESIGN §33.2.8): a real subprocess on the CPU with real pipes.

The header must pass the legacy validation, the records must decode from byte 0, ``quit``, stdin EOF and a
closed reader must end with a checkpoint and exit 0, ``play.lock`` must keep a second process (and the Lab's
``resume``) out, a directory without ``play.json`` must be refused unless it is adopted, and an older
checkpoint must be refused unless it is truncated.
"""

from __future__ import annotations

import io
import signal
import time
from pathlib import Path

import mlx.core as mx
import pytest

from alife import cli, layout, play, stream
from alife.config import dims, load_config
from alife.runio import RunDir
from alife.state import init_state
from play_decode import Decoded, decode
from play_helpers import finish, spawn, wait_for

TINY = ["--preset", "tiny", "--set", "brain.weight_dtype=float32"]


def play_tiny(out: Path, send: str, *extra: str, until: str | None = "40") -> tuple[int, Decoded, str]:
    args = ["play", *TINY, "--out", str(out), *extra, *(["--until", until] if until else [])]
    code, data, err = finish(spawn(*args), send)
    return code, decode(data) if data else Decoded({}, [], []), err


def test_the_header_is_accepted_by_the_legacy_validation_and_the_records_decode_from_byte_0(
    tmp_path: Path,
) -> None:
    code, d, err = play_tiny(tmp_path / "w", "rate max\nrun\n" + "\n" * 0, "--rate", "max")
    assert code == 0, err
    h = d.header
    cfg = load_config(preset="tiny", overrides=["brain.weight_dtype=float32"])
    assert h["mode"] == "simulate" and h["source"] == "preset" and h["every"] == 1 and h["fields"] is False
    assert h["color"] == "tag" and h["color"] in h["color_modes"] and h["protocol"] == 1
    assert (h["dims"], h["N"], h["B"], h["NCH"]) == (2, dims(cfg).n, layout.B, layout.NCH)
    assert h["FG"] == dims(cfg).fg and h["FZ"] == dims(cfg).fz
    assert h["L"] == cfg.world.size and h["field_cell"] == cfg.world.field_cell
    assert (
        h["play"]["records"][0] == "FRMA"
        and h["play"]["tick"] == 25
        and h["play"]["weather"]["enabled"] is False
    )
    assert [r.tag for r in d.records[:2]] == ["FRMA", "IDNT"]  # the first state, written at start
    assert d.skipped == []  # every record was one the decoder knows


def test_quit_checkpoints_and_exits_0(tmp_path: Path) -> None:
    out = tmp_path / "w"
    code, d, err = play_tiny(out, "rate max\nrun\nquit\n", until=None)
    assert code == 0, err
    rd = RunDir.open(out)
    last = int(rd.metrics_rows()[-1]["step"])
    assert rd.meta()["status"] == "interrupted" and last >= 1
    assert [s for s, _ in rd.checkpoints()][-1] == last and rd.event_files("births")[-1][1] == last
    steps = [r.step for r in d.records if r.tag == "FRMA"]
    assert steps[0] == 0 and steps[-1] <= last
    assert (
        rd.read_play_json()["planet"] is None and (out / "play.lock").exists() is False
    )  # the lock is released


def test_stdin_eof_is_a_quit(tmp_path: Path) -> None:
    code, d, err = play_tiny(tmp_path / "w", "", until=None)
    assert code == 0, err
    assert RunDir.open(tmp_path / "w").meta()["status"] == "interrupted"
    assert [r.tag for r in d.records][:3] == ["FRMA", "IDNT", "INTN"]


def test_a_closed_reader_ends_with_a_checkpoint_and_exit_0(tmp_path: Path) -> None:
    out = tmp_path / "w"
    p = spawn("play", *TINY, "--out", str(out), "--rate", "max")
    assert p.stdout is not None and p.stdin is not None
    head = p.stdout.read(4096)
    assert head.startswith(b"ALSTRM")
    p.stdout.close()  # the reader goes away; stdin stays open
    code = p.wait(timeout=240)
    err = p.stderr.read().decode(errors="replace") if p.stderr else ""
    assert code == 0, err
    rd = RunDir.open(out)
    assert rd.meta()["status"] == "interrupted"
    assert [s for s, _ in rd.checkpoints()][-1] == int(rd.metrics_rows()[-1]["step"])


def test_stdout_stays_strictly_decodable_when_the_step_prints(tmp_path: Path) -> None:
    """The announcement of a new world and every other print go to stderr (protocol_output is the first act),
    so the stream decodes from byte 0 and ends exactly at a record."""
    code, d, err = play_tiny(tmp_path / "w", "rate max\nrun\n", "--rate", "max")
    assert code == 0 and "run " in err and "N=1024" in err  # the announcement is on stderr
    assert d.header["N"] == 1024


def test_a_second_play_and_the_labs_resume_are_refused_while_the_lock_is_held(tmp_path: Path) -> None:
    out = tmp_path / "w"
    first = spawn("play", *TINY, "--out", str(out))
    try:
        wait_for(out / "play.json")
        assert (out / "play.lock").exists()
        code, _, err = finish(spawn("play", str(out)), "")
        assert code == 2 and "being played" in err and "play.lock" in err
        assert cli.main(["resume", str(out), "--steps", "1", "--quiet"]) == 2  # the Lab's resume too
    finally:
        finish(first, "quit\n")
    assert not (out / "play.lock").exists()
    assert cli.main(["resume", str(out), "--steps", "0", "--quiet"]) == 0  # and again when it is free


def test_a_directory_without_play_json_is_refused_and_adoption_keeps_the_config(tmp_path: Path) -> None:
    research = tmp_path / "research"
    assert cli.main(["run", *TINY, "--out", str(research), "--steps", "20", "--quiet"]) == 0
    before = (research / "config.json").read_text()
    code, _, err = finish(spawn("play", str(research)), "")
    assert code == 2 and "play.json" in err and "--adopt" in err
    assert (research / "config.json").read_text() == before and not (research / "play.json").exists()
    code, data, err = finish(spawn("play", str(research), "--adopt"), "rate max\nrun\nquit\n")
    assert code == 0, err
    assert (research / "config.pre_play.json").read_text() == before  # the backup is the original config
    assert (research / "play.json").exists() and (research / "config.json").read_text() != before
    saved = RunDir.open(research).config()
    assert saved.run.metrics_every == 250 and saved.run.checkpoint_every == 6000 and saved.game.observe
    # adopting again (play.json removed) does not overwrite the backup, and a plain resume still works
    (research / "play.json").unlink()
    finish(spawn("play", str(research), "--adopt"), "quit\n")
    assert (research / "config.pre_play.json").read_text() == before and (research / "play.json").exists()
    assert cli.main(["resume", str(research), "--steps", "0", "--quiet"]) == 0
    assert decode(data).header["source"] == "run"


def test_an_older_checkpoint_needs_truncate(tmp_path: Path) -> None:
    out = tmp_path / "w"
    assert (
        cli.main(["run", *TINY, "--out", str(out), "--steps", "60", "--checkpoint-every", "30", "--quiet"])
        == 0
    )
    RunDir.open(out).write_play_json(
        {"base_rate": 14.4, "cadence": None, "tick": 25, "weather": None, "planet": None}
    )
    code, _, err = finish(spawn("play", str(out), "--checkpoint", "30"), "")
    assert code == 2 and "older than the newest (60)" in err and "--truncate" in err
    assert "checkpoints [60]" in err and "event chunk" in err  # what would be cut is listed
    rd = RunDir.open(out)
    assert [s for s, _ in rd.checkpoints()] == [0, 30, 60]  # nothing was touched
    code, _, err = finish(spawn("play", str(out), "--checkpoint", "30", "--truncate"), "quit\n")
    assert code == 0, err
    rd = RunDir.open(out)
    assert [s for s, _ in rd.checkpoints()] == [0, 30, 31]  # 60 was cut; the quit step was saved
    assert [r["step"] for r in rd.metrics_rows()] == [10, 20, 30, 31]


def test_the_flags_of_the_three_forms_are_checked_before_anything_is_written(tmp_path: Path) -> None:
    out = tmp_path / "w"
    cases = [
        (["play"], "exactly one of"),
        (["play", str(out), "--preset", "tiny"], "exactly one of"),
        (["play", "--preset", "tiny", "--checkpoint", "30"], "--checkpoint applies to a world directory"),
        (["play", "--preset", "tiny", "--adopt"], "--adopt applies"),
        (["play", "--preset", "tiny", "--replay"], "--replay applies"),
        (["play", str(out), "--set", "run.seed=1"], "make a new world"),
        (["play", str(out), "--replay", "--director", "rules"], "cannot also decide"),
    ]
    for args, why in cases:
        code, _, err = finish(spawn(*args), "")
        assert code == 2 and why in err, (args, err)
    assert not out.exists()


def test_the_rate_flags(tmp_path: Path) -> None:
    code, d, _ = play_tiny(tmp_path / "a", "quit\n", until=None)  # no --rate: starts paused
    assert code == 0
    clcks = [r for r in d.records if r.tag == "CLCK"]
    assert clcks[0].f["reason"] == 1 and clcks[0].f["rate"] == 0.0
    code, d, _ = play_tiny(tmp_path / "b", "quit\n", "--rate", "2", until=None)
    assert next(r for r in d.records if r.tag == "CLCK").f["rate"] == 2.0
    code, d, _ = play_tiny(tmp_path / "c", "quit\n", "--rate", "2", "--paused", until=None)
    assert next(r for r in d.records if r.tag == "CLCK").f["rate"] == 0.0
    code, d, _ = play_tiny(tmp_path / "d", "quit\n", "--rate", "max", until=None)
    assert next(r for r in d.records if r.tag == "CLCK").f["rate"] == -1.0
    pj = RunDir.open(tmp_path / "d").read_play_json()
    assert pj["cadence"] == {
        "metrics_every": 250,
        "heavy_metrics_every": 250,
        "checkpoint_every": 6000,
        "keep_checkpoints": 3,
    }
    code, _, _ = finish(spawn("play", *TINY, "--out", str(tmp_path / "e"), "--rate", "5000"), "")
    assert code == 2


def test_until_stops_at_the_step(tmp_path: Path) -> None:
    out = tmp_path / "w"
    p = spawn("play", *TINY, "--out", str(out), "--rate", "max", "--until", "12")
    assert p.stdin is not None
    # stdin stays open: the world ends by itself at its step; read to the end of the stream
    data = p.stdout.read() if p.stdout else b""
    code = p.wait(timeout=240)
    assert code == 0
    d = decode(data)
    assert [r.step for r in d.records if r.tag == "FRMA"][-1] == 12
    assert RunDir.open(out).meta()["status"] == "done"


def test_sigint_while_paused_checkpoints_and_exits_130(tmp_path: Path) -> None:
    out = tmp_path / "w"
    p = spawn("play", *TINY, "--out", str(out))
    wait_for(out / "play.json")
    time.sleep(0.5)  # in the paused wait
    p.send_signal(signal.SIGINT)  # stdin stays open: only the signal stops it
    code = p.wait(timeout=240)
    err = p.stderr.read().decode(errors="replace") if p.stderr else ""
    assert code == 130, err
    rd = RunDir.open(out)
    assert rd.meta()["status"] == "interrupted" and [s for s, _ in rd.checkpoints()][-1] == 1
    assert not (out / "play.lock").exists()


def test_a_failing_presenter_exits_1_after_the_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capfdbinary: pytest.CaptureFixture[bytes]
) -> None:
    calls = {"n": 0}
    original = play.PlayPacer._send_frame

    def boom(self: play.PlayPacer, state: object) -> None:
        calls["n"] += 1
        if calls["n"] == 8:
            raise RuntimeError("the presenter broke")
        original(self, state)

    monkeypatch.setattr(play.PlayPacer, "_send_frame", boom)
    out = tmp_path / "w"
    code = cli.main(["play", *TINY, "--out", str(out), "--no-control", "--rate", "max"])
    err = capfdbinary.readouterr().err.decode(errors="replace")
    assert code == 1 and "the presenter broke" in err
    rd = RunDir.open(out)
    last = int(rd.metrics_rows()[-1]["step"])
    assert rd.meta()["status"] == "interrupted" and [s for s, _ in rd.checkpoints()][-1] == last
    assert not (out / "play.lock").exists()


def test_an_extinct_world_ends_with_a_note_and_no_header(
    tmp_path: Path, capfdbinary: pytest.CaptureFixture[bytes]
) -> None:
    cfg = load_config(
        preset="tiny", overrides=["brain.weight_dtype=float32", "population.immigration_floor=0"]
    )
    rd = RunDir.create(tmp_path / "w", cfg)
    state = init_state(cfg)
    state = state._replace(
        alive=mx.zeros_like(state.alive), uid=mx.full(state.uid.shape, -1, dtype=state.uid.dtype)
    )
    out = io.BytesIO()
    code = play.serve_play(
        rd, cfg, [state], steps=10, options=play.PlayOptions(), control=stream.NoControl(), out=out
    )
    del state
    assert code == 0 and out.getvalue() == b"" and rd.meta()["status"] == "extinct"
    assert b"extinct" in capfdbinary.readouterr().err


def test_a_new_world_into_a_non_empty_directory_writes_nothing(tmp_path: Path) -> None:
    out = tmp_path / "w"
    out.mkdir()
    (out / "keep.txt").write_text("x")
    code, _, err = finish(spawn("play", *TINY, "--out", str(out)), "")
    assert code == 1 and "not empty" in err
    assert [p.name for p in out.iterdir()] == ["keep.txt"]  # not even a lock file
