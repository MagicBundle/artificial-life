"""Planets (DESIGN §33.3.4): ``alife play --init-from SRC:STEP --overlay storm --weather-spinup N``.

A planet is a fork of a checkpoint with the weather turned on and spun up; ``play`` prepares it with the
extracted ``cli._prepare_fork`` (the same refusals as ``run --init-from``, and a spin-up of at least 1 step
when the fork turns the weather on), keeps its stdout strictly the protocol stream, takes the lock before it
writes anything and writes ``play.json`` last. These tests run ``cli.main`` in the process (the pytest fd
capture stands for the pipe) on the 256-row 64k3d world with a short cadence."""

from __future__ import annotations

from pathlib import Path

import pytest

from alife import cli, play, runner, weather_spinup
from alife.config import load_config
from alife.game import housefile
from alife.runio import RunDir
from alife.state import init_state
from play_decode import decode
from play_helpers import snapshots, spawn
from weather_helpers import SMALL3D

CADENCE = {"metrics_every": 10, "heavy_metrics_every": 10, "checkpoint_every": 10, "keep_checkpoints": 3}
SPINUP = 64


@pytest.fixture(scope="module")
def source(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A weather-off 64k3d world of 256 rows, checkpointed at step 12."""
    cfg = load_config(
        preset="64k3d",
        overrides=[*SMALL3D, "run.checkpoint_every=12", "run.metrics_every=12", "run.heavy_metrics_every=12"],
    )
    rd = RunDir.create(tmp_path_factory.mktemp("src") / "src", cfg)
    runner.run(rd, cfg, init_state(cfg), 12, quiet=True)
    assert [s for s, _ in rd.checkpoints()][-1] == 12
    return rd.path


@pytest.fixture
def short_cadence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(play, "PLAY_CADENCE", CADENCE)
    snapshots(monkeypatch, 10)


def planet_args(source: Path, out: Path, *extra: str) -> list[str]:
    return [
        "play", "--init-from", f"{source}:12", "--overlay", "storm", "--weather-spinup", str(SPINUP),
        "--out", str(out), "--no-control", "--rate", "max", "--until", "52", *extra,
    ]  # fmt: skip


@pytest.fixture
def planet(
    source: Path, tmp_path: Path, short_cadence: None, capfdbinary: pytest.CaptureFixture[bytes]
) -> tuple[Path, bytes, str]:
    out = tmp_path / "planet"
    code = cli.main(planet_args(source, out))
    cap = capfdbinary.readouterr()
    assert code == 0, cap.err.decode(errors="replace")
    return out, cap.out, cap.err.decode(errors="replace")


def test_a_planet_streams_from_byte_0_and_reports_its_progress_on_stderr(
    planet: tuple[Path, bytes, str], source: Path
) -> None:
    _, data, err = planet
    d = decode(data)  # strictly: the header is the first thing in the stream, and every byte is a record
    h = d.header["play"]
    assert h["weather"]["enabled"] is True and h["weather"]["refs"]["dc"] > 0
    assert h["planet"] == {"source": f"{source}:12", "houses": "exact"}  # the source has its events
    assert {"MAP1", "STRM"} <= set(h["records"]) and d.skipped == []
    assert f"weather spin-up: {SPINUP}/{SPINUP} steps" in err and "run " in err  # the progress is stderr's
    tags = [r.tag for r in d.records]
    assert tags[:4] == ["FRMA", "IDNT", "INTN", "ENVS"] and "MAP1" in tags[:8] and tags.count("MAP1") >= 2
    assert d.header["source"] == "run" and d.header["step"] == 12


def test_play_json_is_written_with_the_planet_and_the_cloud_scale(planet: tuple[Path, bytes, str]) -> None:
    out, data, _ = planet
    rd = RunDir.open(out)
    pj = rd.read_play_json()
    assert pj is not None and pj["cadence"] == CADENCE and pj["tick"] == 25
    assert pj["planet"]["step"] == 12 and pj["planet"]["weather_spinup"] == SPINUP
    assert pj["planet"]["houses"] == "exact" and pj["planet"]["made_at_step"] == 12
    assert pj["weather"]["dc_ref"] == decode(data).header["play"]["weather"]["refs"]["dc"] > 0
    assert rd.meta()["start_step"] == 12 and rd.meta()["weather_spinup"] == SPINUP
    assert "origin_row" not in rd.meta(), "step 12 is off the snapshot grid: the first chapter is an arrival"
    assert not (out / "play.lock").exists()


def test_the_sidecar_is_not_reused_and_the_houses_are_the_rebuilt_ones(
    planet: tuple[Path, bytes, str],
) -> None:
    out, data, _ = planet
    first = next(r for r in decode(data).records if r.tag == "IDNT")
    # fresh observers at the fork, seeded with the exact houses of the source's birth chain (DESIGN 33.3.5)
    assert first.f["json"]["exact"] is True and first.f["json"]["origin"] == 0
    assert RunDir.open(out).read_observers(12) is not None  # the planet's own sidecar at its first checkpoint
    assert [s for s, _ in housefile.files(RunDir.open(out))] == [12]  # and its house file


def test_the_spun_up_weather_is_the_start_steps(planet: tuple[Path, bytes, str]) -> None:
    out, data, _ = planet
    d = decode(data)
    envs = next(r for r in d.records if r.tag == "ENVS")
    cfg = RunDir.open(out).config()
    from alife import weather as W

    p = W.params_from_config(cfg)
    assert [(q["x"], q["y"]) for q in envs.f["json"]["pools"]] == [
        (round((x0 + vx * 12) % p.L, 2), round((y0 + vy * 12) % p.L, 2)) for x0, y0, *_, vx, vy in p.pools
    ]  # the pools are those of step 12, the step the spin-up ends in


def test_the_origin_checkpoint_survives_the_saves(planet: tuple[Path, bytes, str]) -> None:
    out, _, _ = planet
    rd = RunDir.open(out)
    steps = [s for s, _ in rd.checkpoints()]
    assert 12 in steps and steps[-1] == 52 and len(steps) == 4  # the origin and the newest three of ten
    assert len(rd.event_files("births")) >= 5  # (more than the saves a plain keep=3 could hold)
    assert RunDir.open(out).meta()["status"] == "done"


def test_a_planet_has_storm_files_and_a_storm_log(planet: tuple[Path, bytes, str]) -> None:
    out, _, _ = planet
    rd = RunDir.open(out)
    assert [s for s, _ in rd.storms_state_files()] == [s for s, _ in rd.checkpoints()]
    assert (
        rd.storms_path.exists() or rd.storms_lines() == []
    )  # (storms may take longer than 52 steps to form)


# --- refusals ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("spinup", [[], ["--weather-spinup", "0"]])
def test_a_fork_that_turns_the_weather_on_needs_a_spinup(
    spinup: list[str],
    source: Path,
    tmp_path: Path,
    short_cadence: None,
    capfdbinary: pytest.CaptureFixture[bytes],
) -> None:
    out = tmp_path / "p"
    args = [
        "play",
        "--init-from",
        f"{source}:12",
        "--overlay",
        "storm",
        "--out",
        str(out),
        "--no-control",
        "--rate",
        "max",
        "--until",
        "20",  # (a regression would play instead of refusing, and end)
        *spinup,
    ]
    assert cli.main(args) == 2
    cap = capfdbinary.readouterr()
    assert b"--weather-spinup" in cap.err and cap.out == b"" and not out.exists()


def test_a_weather_on_source_refuses_a_spinup_and_is_forked_as_it_is(
    planet: tuple[Path, bytes, str], tmp_path: Path, capfdbinary: pytest.CaptureFixture[bytes]
) -> None:
    src, _, _ = planet
    out = tmp_path / "again"
    refused = ["play", "--init-from", str(src), "--weather-spinup", "8", "--out", str(out), "--no-control"]
    assert cli.main([*refused, "--rate", "max", "--until", "60"]) == 2
    assert b"already has an evolving weather" in capfdbinary.readouterr().err and not out.exists()
    code = cli.main(
        ["play", "--init-from", str(src), "--out", str(out), "--no-control", "--rate", "max", "--until", "62"]
    )
    cap = capfdbinary.readouterr()
    assert code == 0, cap.err.decode(errors="replace")
    d = decode(cap.out)
    assert (
        d.header["step"] == 52 and d.header["play"]["weather"]["enabled"]
    )  # the planet's own end, no new spin-up


def test_the_fork_refusals_of_run_hold_and_nothing_is_written(
    source: Path, tmp_path: Path, short_cadence: None, capfdbinary: pytest.CaptureFixture[bytes]
) -> None:
    out = tmp_path / "p"
    args = planet_args(source, out, "--set", "population.n_init=32")
    assert cli.main(args) == 4  # CheckpointError: the clade bins depend on n_init
    cap = capfdbinary.readouterr()
    assert b"population.n_init must equal" in cap.err and cap.out == b"" and not out.exists()


def test_a_state_that_violates_the_new_config_writes_nothing(
    source: Path, tmp_path: Path, short_cadence: None, capfdbinary: pytest.CaptureFixture[bytes]
) -> None:
    out = tmp_path / "p"
    tight = ["--set", "mutation.sigma_init=0.005", "--set", "mutation.sigma_max=0.01"]
    assert cli.main(planet_args(source, out, *tight)) == 4
    cap = capfdbinary.readouterr()
    assert b"the state violates the new config" in cap.err and cap.out == b"" and not out.exists()


def test_an_interrupted_spinup_leaves_no_play_json(
    source: Path,
    tmp_path: Path,
    short_cadence: None,
    capfdbinary: pytest.CaptureFixture[bytes],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cancelled build is an unfinished directory: it has no ``play.json`` (written last), the lock is
    released and the exit code is the interrupt's."""

    def interrupted(*a: object, **k: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(weather_spinup, "spinup", interrupted)
    out = tmp_path / "p"
    assert cli.main(planet_args(source, out)) == 130
    cap = capfdbinary.readouterr()
    assert cap.out == b"" and b"interrupted during the weather spin-up" in cap.err
    assert out.is_dir() and not (out / "play.json").exists() and not (out / "play.lock").exists()
    assert RunDir.open(out).meta()["status"] == "interrupted"
    # such a directory is refused as a world
    assert cli.main(["play", str(out), "--no-control"]) == 2
    assert b"play.json" in capfdbinary.readouterr().err


def test_a_preset_world_with_the_weather_on(
    tmp_path: Path, short_cadence: None, capfdbinary: pytest.CaptureFixture[bytes]
) -> None:
    out = tmp_path / "p"
    args = [
        "play", "--preset", "64k3d", "--overlay", "storm", "--set", "population.n_max=256", "--set",
        "population.n_init=64", "--set", "population.max_births_per_step=32", "--set",
        "brain.weight_dtype=float32", "--weather-spinup", "16", "--out", str(out), "--no-control",
        "--rate", "max", "--until", "20",
    ]  # fmt: skip
    assert cli.main(args) == 0
    d = decode(capfdbinary.readouterr().out)
    assert d.header["source"] == "preset" and d.header["play"]["planet"] == {"source": None, "houses": "none"}
    assert RunDir.open(out).read_play_json()["planet"] is None


def test_a_planet_into_a_real_pipe_decodes_strictly_from_byte_0(source: Path, tmp_path: Path) -> None:
    """The same as the capture, through the pipe the window reads: the spin-up's progress is on stderr, the
    first byte of stdout is the magic, and the stream is whole records to its end."""
    p = spawn(
        "play", "--init-from", f"{source}:12", "--overlay", "storm", "--weather-spinup", "64",
        "--out", str(tmp_path / "planet"), "--rate", "max", "--until", "40",
    )  # fmt: skip
    assert p.stdout is not None and p.stderr is not None
    data = p.stdout.read()  # stdin stays open: the world ends at its step
    code = p.wait(timeout=300)
    err = p.stderr.read().decode(errors="replace")
    assert code == 0, err
    assert data[:8] == b"ALSTRM\x00\x01"
    assert "weather spin-up: 64/64 steps" in err
    d = decode(data)
    assert d.header["play"]["weather"]["enabled"] and [r.step for r in d.records if r.tag == "FRMA"][-1] == 40


def test_a_resumed_planet_keeps_its_map_scale_and_its_tracker(
    planet: tuple[Path, bytes, str], capfdbinary: pytest.CaptureFixture[bytes]
) -> None:
    """A later session reads ``dc_ref`` back from ``play.json`` (the map scale is the same across sessions),
    keeps the planet block, continues the tracker from its saved state (no ``renumbered``) and opens at the
    newest checkpoint."""
    out, data, _ = planet
    first = decode(data).header["play"]
    pj_before = RunDir.open(out).read_play_json()
    code = cli.main(["play", str(out), "--no-control", "--rate", "max", "--until", "62"])
    cap = capfdbinary.readouterr()
    assert code == 0, cap.err.decode(errors="replace")
    d = decode(cap.out)
    assert d.header["step"] == 52 and d.header["play"]["weather"] == first["weather"]
    assert d.header["play"]["planet"] == first["planet"]
    pj = RunDir.open(out).read_play_json()
    assert pj["planet"] == pj_before["planet"] and pj["weather"] == pj_before["weather"]
    texts = [e["text"] for r in d.records if r.tag == "EVNT" for e in r.f["json"]]
    assert not any("numbering restarted" in t for t in texts)
    assert [r.step for r in d.records if r.tag == "FRMA"][-1] == 62
