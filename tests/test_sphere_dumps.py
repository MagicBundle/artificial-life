"""A sphere run directory and its dumps (DESIGN §46.7; sphere series S5): ``geometry.json`` states the planet
for offline tools, the ``mat`` and ``niche`` dumps are ``(FZ, Npix)`` and index by ``layer·Npix + pixel``, a
box run directory has no ``geometry.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from alife import cli, geom
from alife.runio import GEOMETRY_FILE, RunDir

SETS = ["brain.weight_dtype=float32", "run.compile=false", "run.checkpoint_every=10", "run.metrics_every=10",
        "run.heavy_metrics_every=10", "structures.enabled=true", "structures.build_max=64", "niche.farm=true",
        "run.mat_every=10", "run.niche_every=10", "game.observe=true"]  # fmt: skip


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("dump") / "run"
    argv = ["run", "--preset", "tiny3s", *(x for s in SETS for x in ("--set", s)), "--steps", "30",
            "--out", str(out), "--quiet"]  # fmt: skip
    assert cli.main(argv) == 0
    return out


def test_geometry_json_states_the_planet(run: Path) -> None:
    rd = RunDir.open(run)
    g = rd.geometry()
    cfg = rd.config()
    assert g is not None and json.loads((run / GEOMETRY_FILE).read_text()) == g
    assert (g["planet_radius"], g["height"], g["nside"], g["layers"]) == (36.0, 16.0, 8, 8)
    assert g["npix"] == 12 * 8**2 and g["ring_order"] == "ring" and g["dump_shape"] == ["layers", "npix"]
    np.testing.assert_allclose(g["shell_edges"], geom.layer_radii(36.0, 16.0, 8))
    vol = np.diff(np.asarray(g["shell_edges"]) ** 3) * (4 * np.pi / 3) / g["npix"]  # one cell of each layer
    np.testing.assert_allclose(vol, g["layer_volume"], rtol=1e-9)
    assert g["nside_territory"] == geom.TERRITORY_NSIDE and g["nside_weather"] == 0
    assert cfg.world.geometry == "sphere" and rd.meta()["geometry"] == "sphere"


def test_the_grid_dumps_are_layer_by_pixel(run: Path) -> None:
    rd = RunDir.open(run)
    g = rd.geometry()
    assert [s for s, _ in rd.dump_files("mat")] == [10, 20, 30] == [s for s, _ in rd.dump_files("niche")]
    for kind in ("mat", "niche"):
        grid = rd.read_grid(kind, 30)
        assert grid.shape == (g["layers"], g["npix"]) and grid.dtype == np.int32 and grid.min() >= 0
    assert rd.read_grid("mat", 30).sum() > 0, "agents built something: the dump is not vacuous"
    # a dump is indexed like the engine's cells: a flat index layer*npix + pixel lands in the cell of a point
    from alife.config import dims

    cfg = rd.config()
    assert dims(cfg).field_shape == (g["layers"], g["npix"])


def test_a_box_run_has_no_geometry_json(tmp_path: Path) -> None:
    out = tmp_path / "box"
    assert cli.main(["run", "--preset", "tiny", "--steps", "10", "--out", str(out), "--quiet"]) == 0
    assert not (out / GEOMETRY_FILE).exists() and RunDir.open(out).geometry() is None


def test_resume_keeps_the_sidecar_and_the_observers_run(run: Path) -> None:
    before = (run / GEOMETRY_FILE).read_bytes()
    assert cli.main(["resume", str(run), "--until", "40", "--quiet"]) == 0
    assert (run / GEOMETRY_FILE).read_bytes() == before
    rd = RunDir.open(run)
    assert rd.meta()["status"] == "done" and (run / "observers").is_dir()
    assert [s for s, _ in rd.dump_files("mat")][-1] == 40
