"""The report's chapter counts from the planet's origin (DESIGN §34.2.1, §34.4).

Found on the G3b Mac day: ``cli.py`` built ``PlayOptions`` without the planet's ``made_at_step``, so the
server numbered ``RPRT.chapter`` from step 0 ("Chapter 79" on the Report tab of a planet made at 450,000)
while the bar and the card, which read ``play.json``, said "Chapter 4". The worlds of the other report tests
start at 0 and cannot see it, so these tests make a planet through ``cli.main`` (a fork at step 500, with
chapters of 500 steps: the same arithmetic as 450,000 and 6,000) and compare the stream, the backlog of a
resume, the files and the client's rule."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pytest

import play_helpers as H
from alife import cli, play, runner, streamrec
from alife.analysis import world_report as W
from alife.config import load_config
from alife.game import report as R
from alife.game import reportjob
from alife.runio import RunDir
from alife.state import init_state
from play_decode import decode
from weather_helpers import SMALL3D

CHAPTER = 500
MADE_AT = 500
CADENCE = {"metrics_every": 250, "heavy_metrics_every": 250, "checkpoint_every": 500, "keep_checkpoints": 3}


# --- the arithmetic -----------------------------------------------------------------------------------------


def test_a_planet_made_at_450000_numbers_468000_to_474000_as_chapter_4() -> None:
    n_origin = reportjob.origin_chapters(450_000, 6_000)
    assert n_origin == 75
    assert reportjob.chapter_number(474_000, 6_000, n_origin) == 4
    obj = reportjob.withheld_object(468_000, 474_000, n_origin=n_origin, length=6_000, reason="x")
    assert (obj["chapter"], obj["n_abs"], obj["a"], obj["b"]) == (4, 79, 468_000, 474_000)
    assert reportjob.chapter_number(456_000, 6_000, n_origin) == 1  # the first chapter of the planet
    # a start off the grid: the arrival chapter (450,250 to 456,000) is the planet's first too
    assert reportjob.chapter_number(456_000, 6_000, reportjob.origin_chapters(450_250, 6_000)) == 1


def test_a_world_without_a_planet_numbers_from_step_0() -> None:
    assert reportjob.origin_chapters(0, 6_000) == 0
    assert reportjob.chapter_number(6_000, 6_000, 0) == 1
    assert reportjob.chapter_number(474_000, 6_000, 0) == 79


@pytest.mark.parametrize(
    ("planet", "expected"),
    [
        (None, 0),  # a preset, an adopted run
        ({}, 0),
        ({"source": "x", "houses": "clade"}, 0),  # a play.json whose planet has no made_at_step
        ({"made_at_step": 450_000}, 450_000),
        ({"source": "x", "step": 450_000, "houses": "exact", "made_at_step": 450_000}, 450_000),
        ({"made_at_step": 0}, 0),
        ({"made_at_step": -6_000}, 0),
        ({"made_at_step": "450000"}, 0),
        ({"made_at_step": 450_000.0}, 0),
        ({"made_at_step": True}, 0),
        ({"made_at_step": None}, 0),
    ],
)
def test_the_origin_step_of_a_play_json_planet_and_its_fallback(planet: Any, expected: int) -> None:
    """A missing or unusable ``made_at_step`` is 0, which is also what the client takes from such a file."""
    assert play.planet_origin_step(planet) == expected


def test_a_preset_world_played_from_0_keeps_the_absolute_chapters() -> None:
    cfg = H.world("2d", "run.metrics_every=250", "run.heavy_metrics_every=250", "run.checkpoint_every=500")
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        rd = RunDir.create(Path(tmp) / "w", cfg)
        state = init_state(cfg)
        rd.update_meta(origin_row=play.origin_row(state, cfg))
        out = io.BytesIO()
        opt = play.PlayOptions(rate=float("inf"), paused=False, fresh=True, unattended=True, chapter=CHAPTER)
        pacer = play.PlayPacer(out, H.ScriptControl([]), opt, clock=H.FakeClock(), note=lambda m: None)
        try:
            runner.run(rd, cfg, state, 1000, quiet=True, pacer=pacer)
        finally:
            pacer.close()
    got = [(r.f["json"]["n_abs"], r.f["json"]["chapter"]) for r in H.decoded(out).records if r.tag == "RPRT"]
    assert got == [(1, 1), (2, 2)]


# --- a planet made by ``alife play`` ------------------------------------------------------------------------


def client_number(clock_chapter: int, play_json: dict[str, Any], length: int) -> int:
    """What the Swift client shows (``ChapterOrigin.chapter`` and ``ChapterBar``): the absolute chapter of the
    clock less ``made_at_step // length`` of ``play.json``'s planet, at least 1."""
    planet = play_json.get("planet")
    made = planet["made_at_step"] if isinstance(planet, dict) and "made_at_step" in planet else 0
    return max(clock_chapter - max(made, 0) // length, 1)


def rprt(data: bytes) -> list[Any]:
    return [r for r in decode(data).records if r.tag == "RPRT"]


@pytest.fixture(scope="module")
def source(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A weather-off 64k3d world of 256 rows with a checkpoint (and an observer sidecar) at step 500."""
    cfg = load_config(
        preset="64k3d",
        overrides=[
            *SMALL3D,
            "run.checkpoint_every=500",
            "run.metrics_every=250",
            "run.heavy_metrics_every=250",
            "game.observe=true",
        ],
    )
    rd = RunDir.create(tmp_path_factory.mktemp("src") / "src", cfg)
    runner.run(rd, cfg, init_state(cfg), MADE_AT, quiet=True)
    assert [s for s, _ in rd.checkpoints()][-1] == MADE_AT
    return rd.path


@pytest.fixture
def chapters_of_500(monkeypatch: pytest.MonkeyPatch) -> None:
    """``alife play`` with chapters of 500 and no pause nobody would answer (the CLI has no flag for them)."""
    real = play.PlayOptions
    monkeypatch.setattr(play, "PLAY_CADENCE", CADENCE)
    monkeypatch.setattr(
        play, "PlayOptions", lambda **kw: real(**{**kw, "chapter": CHAPTER, "unattended": True})
    )


@pytest.fixture
def planet(
    source: Path, tmp_path: Path, chapters_of_500: None, capfdbinary: pytest.CaptureFixture[bytes]
) -> tuple[Path, bytes]:
    out = tmp_path / "planet"
    args = [
        "play", "--init-from", f"{source}:{MADE_AT}", "--overlay", "storm", "--weather-spinup", "32",
        "--out", str(out), "--no-control", "--rate", "max", "--until", "1510",
    ]  # fmt: skip
    code = cli.main(args)
    cap = capfdbinary.readouterr()
    assert code == 0, cap.err.decode(errors="replace")
    return out, cap.out


def live_without_latency(objs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{**o, "latency_s": None} for o in objs]


def test_the_stream_of_a_planet_numbers_its_reports_from_the_planets_origin(
    planet: tuple[Path, bytes],
) -> None:
    out, data = planet
    play_json = json.loads((out / "play.json").read_text())
    assert play_json["planet"]["made_at_step"] == MADE_AT
    live = [r.f["json"] for r in rprt(data)]
    # (500, 1000] is the planet's first chapter, (1000, 1500] its second; the parent's numbers were 2 and 3
    assert [(o["a"], o["b"], o["n_abs"], o["chapter"]) for o in live] == [
        (500, 1000, 2, 1),
        (1000, 1500, 3, 2),
    ]
    # the client's number for the same chapters (the bar and the card read ``play.json``)
    assert [client_number(o["n_abs"], play_json, CHAPTER) for o in live] == [o["chapter"] for o in live]
    d = decode(data)
    assert {r.f["json"]["chapter"] for r in d.records if r.tag == "RPRT"} == {1, 2}
    clocks = [r for r in d.records if r.tag == "CLCK"]
    assert clocks  # the world's own clock stays absolute (§34.2.1): the client subtracts the origin


def test_live_equals_the_offline_objects_and_the_offline_verify(planet: tuple[Path, bytes]) -> None:
    out, data = planet
    play_json = json.loads((out / "play.json").read_text())
    n_origin = reportjob.origin_chapters(play.planet_origin_step(play_json["planet"]), CHAPTER)
    live = [r.f["json"] for r in rprt(data) if not r.f["backlog"]]
    stored = [o for _, o in reportjob.stored_objects(out, length=CHAPTER, n_origin=n_origin)]
    built = [o for o in live if not o["withheld"]]
    assert built, "the planet's chapters were not reported"
    by_b = {o["b"]: o for o in stored}
    for o in built:
        assert live_without_latency([o]) == live_without_latency([by_b[o["b"]]])
    assert [o["chapter"] for o in stored] == [o["chapter"] for o in built]
    for o in stored:
        streamrec.check_report(o)
    # offline: ``world_report verify`` replays the records of the run, which carry steps and no chapter number
    recs = RunDir.open(out).reports()
    code, summary = W.verify_reports(R.load_run(out), recs)
    assert code == 0 and summary["counts"] == {"ok": len(recs)}, summary


def test_a_resume_sends_the_same_numbers_as_its_backlog(
    planet: tuple[Path, bytes], capfdbinary: pytest.CaptureFixture[bytes]
) -> None:
    out, data = planet
    first = {o["b"]: o["chapter"] for o in (r.f["json"] for r in rprt(data))}
    code = cli.main(["play", str(out), "--no-control", "--rate", "max", "--until", "1520"])
    cap = capfdbinary.readouterr()
    assert code == 0, cap.err.decode(errors="replace")
    again = rprt(cap.out)
    assert again and all(r.f["backlog"] for r in again)
    assert {o["b"]: o["chapter"] for o in (r.f["json"] for r in again)} == first
    assert sorted(first.values()) == [1, 2]


def test_the_cli_gives_the_pacer_the_planets_origin_on_a_new_world_and_on_a_resume(
    source: Path, tmp_path: Path, chapters_of_500: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[play.PlayOptions] = []

    real_serve = play.serve_play

    def spy(rd: Any, cfg: Any, box: Any, *, steps: int, options: play.PlayOptions, **kw: Any) -> int:
        seen.append(options)
        return real_serve(rd, cfg, box, steps=steps, options=options, **kw)

    monkeypatch.setattr(play, "serve_play", spy)
    out = tmp_path / "p"
    new = [
        "play", "--init-from", f"{source}:{MADE_AT}", "--overlay", "storm", "--weather-spinup", "32",
        "--out", str(out), "--no-control", "--rate", "max", "--until", "1010",
    ]  # fmt: skip
    assert cli.main(new) == 0
    assert seen[-1].origin_step == MADE_AT
    assert seen[-1].planet == {
        "source": f"{source}:{MADE_AT}",
        "houses": seen[-1].planet["houses"],
    }  # the header
    # a resume reads the planet from the world's own play.json
    RunDir.open(out).write_play_json(
        {
            "base_rate": 14.4,
            "cadence": dict(CADENCE),
            "tick": 25,
            "weather": None,
            "planet": {**seen[-1].planet, "step": MADE_AT, "weather_spinup": 32, "made_at_step": 750},
        }
    )
    assert cli.main(["play", str(out), "--no-control", "--rate", "max", "--until", "1020"]) == 0
    assert seen[-1].origin_step == 750
    # without the key (a play.json of an older world) the numbering is the absolute one, like the client's
    pj = json.loads((out / "play.json").read_text())
    del pj["planet"]["made_at_step"]
    RunDir.open(out).write_play_json(pj)
    assert cli.main(["play", str(out), "--no-control", "--rate", "max", "--until", "1030"]) == 0
    assert seen[-1].origin_step == 0
