"""A planet of a sphere source (DESIGN §46.8): ``alife play --init-from SRC:STEP --overlay stormsph16
--weather-spinup N`` on a ``tiny3s`` source, as the Storm Worlds launcher builds it (``PlanetSpec``).

The launcher used to pass ``stormsph`` for every sphere, and that overlay is tuned for the ``64k3s`` planet
only: on a ``tiny3s`` source ``alife play`` refused it with 11 config problems (10 since C29: the Courant
number is 0.40 on every planet). These tests run ``cli.main`` in the process (the pytest fd capture stands
for the pipe) on a 12-step ``tiny3s`` source with a short cadence, and decode the stream with the
independent sphere decoder."""

from __future__ import annotations

from pathlib import Path

import pytest

import sphere_decode as sd
from alife import cli, play, runner
from alife.config import load_config
from alife.runio import RunDir
from alife.state import init_state
from play_helpers import snapshots

CADENCE = {"metrics_every": 10, "heavy_metrics_every": 10, "checkpoint_every": 10, "keep_checkpoints": 3}
SPINUP = 16


@pytest.fixture(scope="module")
def source(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A weather-off tiny3s world checkpointed at step 12."""
    cfg = load_config(
        preset="tiny3s",
        overrides=[
            "run.compile=false",
            "run.checkpoint_every=12",
            "run.metrics_every=12",
            "run.heavy_metrics_every=12",
        ],
    )
    rd = RunDir.create(tmp_path_factory.mktemp("src") / "src", cfg)
    runner.run(rd, cfg, init_state(cfg), 12, quiet=True)
    assert [s for s, _ in rd.checkpoints()][-1] == 12
    return rd.path


@pytest.fixture
def short_cadence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(play, "PLAY_CADENCE", CADENCE)
    snapshots(monkeypatch, 10)


def planet_args(source: Path, out: Path, overlay: str) -> list[str]:
    """``PlayLaunch.arguments(planet:)`` for a source of this shape (plus ``--until``)."""
    return [
        "play", "--init-from", f"{source}:12", "--overlay", overlay, "--set", "director.enabled=true",
        "--weather-spinup", str(SPINUP), "--houses", "auto", "--out", str(out), "--no-control",
        "--rate", "max", "--until", "40",
    ]  # fmt: skip


def test_stormsph16_launches_a_tiny3s_planet_and_streams_a_decodable_world(
    source: Path, tmp_path: Path, short_cadence: None, capfdbinary: pytest.CaptureFixture[bytes]
) -> None:
    out = tmp_path / "planet"
    code = cli.main(planet_args(source, out, "stormsph16"))
    cap = capfdbinary.readouterr()
    assert code == 0, cap.err.decode(errors="replace")
    assert f"weather spin-up: {SPINUP}/{SPINUP} steps" in cap.err.decode(errors="replace")
    d = sd.decode(cap.out)
    header = d.header
    assert header["play"]["weather"]["enabled"] and header["play"]["weather"]["nside"] == 16
    assert [r.step for r in d.records if r.tag == "FRMA"][-1] == 40
    cfg = RunDir.open(out).config()
    assert cfg.weather.enabled and cfg.weather.cells == 16 and cfg.world.geometry == "sphere"


def test_stormsph_is_refused_on_a_tiny3s_source_and_nothing_is_streamed(
    source: Path, tmp_path: Path, short_cadence: None, capfdbinary: pytest.CaptureFixture[bytes]
) -> None:
    """The bug of the launcher: the overlay of the 64k3s planet on a tiny3s source."""
    out = tmp_path / "planet"
    code = cli.main(planet_args(source, out, "stormsph"))
    cap = capfdbinary.readouterr()
    assert code == 2 and cap.out == b""
    err = cap.err.decode(errors="replace")
    # (the Courant problem of the first build is gone: the weather runs at Courant 0.40 on every planet, C29)
    assert "config problem" in err and "Nf times a power of two" in err and "latitude" in err
