"""A planet of the mini world (DESIGN §37): ``alife play --init-from SRC:STEP --overlay storm160``.

A 160-wide ``mini3d`` source with 256 rows is forked with the ``storm160`` weather (80 cells, ``f0`` 0.0165,
two pools per hemisphere). The stream header says what the layer is (``weather.cells`` 80 and not the 320
world's 160) while the reference scales stay the fixed ones (``coarse`` 40, ``f0`` and ``refs.zeta`` 0.025),
a ``MAP1`` decodes against them, the ``FRMA`` length formula holds (``play_decode`` asserts it for every
record), and no ``storm_strong`` is raised in the first ticks. Same harness as ``test_play_planet.py``
(``cli.main`` in the process)."""

from __future__ import annotations

from pathlib import Path

import pytest

from alife import cli, play, runner
from alife.config import load_config
from alife.runio import RunDir
from alife.state import init_state
from play_decode import decode
from play_helpers import snapshots
from weather_helpers import SMALL3D

CADENCE = {"metrics_every": 10, "heavy_metrics_every": 10, "checkpoint_every": 10, "keep_checkpoints": 3}
SPINUP = 64


@pytest.fixture(scope="module")
def source(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A weather-off ``mini3d`` world (L = 160) of 256 rows, checkpointed at step 12."""
    cfg = load_config(
        preset="mini3d",
        overrides=[*SMALL3D, "run.checkpoint_every=12", "run.metrics_every=12", "run.heavy_metrics_every=12"],
    )
    assert cfg.world.size == 160.0 and not cfg.weather.enabled
    rd = RunDir.create(tmp_path_factory.mktemp("mini") / "src", cfg)
    runner.run(rd, cfg, init_state(cfg), 12, quiet=True)
    assert [s for s, _ in rd.checkpoints()][-1] == 12
    return rd.path


@pytest.fixture
def planet(
    source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capfdbinary: pytest.CaptureFixture[bytes]
) -> tuple[Path, bytes]:
    monkeypatch.setattr(play, "PLAY_CADENCE", CADENCE)
    snapshots(monkeypatch, 10)
    out = tmp_path / "planet"
    args = [
        "play", "--init-from", f"{source}:12", "--overlay", "storm160", "--weather-spinup", str(SPINUP),
        "--out", str(out), "--no-control", "--rate", "max", "--until", "52",
    ]  # fmt: skip
    code = cli.main(args)
    cap = capfdbinary.readouterr()
    assert code == 0, cap.err.decode(errors="replace")
    return out, cap.out


def test_a_mini_planet_streams_the_layers_grid_and_the_reference_scales(planet: tuple[Path, bytes]) -> None:
    """One planet, built once (a 160-wide world is slow on the CPU), read five ways."""
    out, data = planet
    d = decode(data)
    # the header names the layer (80 cells at dx 2) and keeps the fixed reference scales
    w = d.header["play"]["weather"]
    assert d.header["L"] == 160.0 and d.skipped == []
    assert (w["enabled"], w["cells"], w["dx"], w["coarse"]) == (True, 80, 2.0, 40)
    assert (w["f0"], w["refs"]["zeta"], w["name_zeta"]) == (0.025, 0.025, 0.5)
    assert w["refs"]["dc"] > 0 and {"MAP1", "STRM"} <= set(d.header["play"]["records"])
    # play.json and the config say storm160
    rd = RunDir.open(out)
    pj = rd.read_play_json()
    assert pj is not None and pj["tick"] == 25 and pj["planet"]["weather_spinup"] == SPINUP
    assert pj["weather"]["dc_ref"] == w["refs"]["dc"]
    cfg = rd.config()
    assert (cfg.world.size, cfg.weather.cells, cfg.weather.f0, cfg.weather.seed_rate) == (
        160.0,
        80,
        0.0165,
        0.005,
    )
    assert len(cfg.weather.pools) == 4 * 7 and not (out / "play.lock").exists()
    # a MAP1 of the mini world decodes: 80 x 80 planes, 40 x 40 winds, quantised against 0.025
    maps = [r for r in d.records if r.tag == "MAP1"]
    assert len(maps) >= 2
    for rec in maps:
        assert (rec.f["n"], rec.f["nw"]) == (80, 40)
        assert rec.f["f0"] == pytest.approx(0.025) and rec.f["zeta_ref"] == pytest.approx(0.025)
        assert rec.a["P"].shape == rec.a["zeta"].shape == (80, 80) and rec.a["winds"].shape == (4, 40, 40)
    # the FRMA length formula (36 + 4 ceil(N/32) + 28 n + 8 bonds) is asserted by play_decode on every frame
    frames = [r for r in d.records if r.tag == "FRMA"]
    assert len(frames) >= 4
    # no strong-storm alert in the first ticks (f0 = 0.0165 would have made every storm strong at birth)
    raised = {a["class"] for r in d.records if r.tag == "ALRT" for a in r.f["json"]}
    assert "storm_strong" not in raised
    log = out / "alerts.jsonl"
    assert "storm_strong" not in (log.read_text() if log.exists() else "")


def test_the_storm_overlay_still_refuses_the_mini_source(
    source: Path, tmp_path: Path, capfdbinary: pytest.CaptureFixture[bytes]
) -> None:
    """No alias: the 320 world's overlay on a 160 source is a validation error; nothing is built."""
    out = tmp_path / "bad"
    args = [
        "play",
        "--init-from",
        f"{source}:12",
        "--overlay",
        "storm",
        "--weather-spinup",
        "8",
        "--out",
        str(out),
    ]
    assert cli.main([*args, "--no-control", "--rate", "max", "--until", "20"]) != 0
    cap = capfdbinary.readouterr()
    assert cap.out == b"" and not out.exists() and (b"rule 4" in cap.err or b"rule W-2" in cap.err)
