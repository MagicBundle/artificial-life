"""A sphere config is refused at every place a config enters the engine, until S2 (DESIGN §46.9, review
L3, L4).

Every entry point funnels into ``config.validate`` (``load_config``, ``apply_overrides``, the config
embedded in a
run directory or in a checkpoint's metadata); these tests drive one of each, and the fork guard of the
geometry
keys with the sphere enabled.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from alife import cli, config
from alife.config import ConfigError, apply_overrides, config_to_dict, load_config
from alife.runio import RunDir
from sphere_helpers import sphere_cfg, sphere_enabled  # noqa: F401  (a fixture)

SPHERE_SET = ["--set", "world.geometry=sphere"]
MSG = "not enabled yet"


@pytest.mark.parametrize(
    "argv",
    [
        ["run", "--preset", "tiny3d", *SPHERE_SET, "--quiet"],
        ["bench", "--preset", "tiny3d", *SPHERE_SET],
        ["play", "--preset", "tiny3d", *SPHERE_SET],
    ],
    ids=["run", "bench", "play"],
)
def test_the_commands_that_take_a_preset_refuse_a_sphere(
    argv: list[str], capsys: pytest.CaptureFixture
) -> None:
    assert cli.main(argv) == 2
    assert MSG in capsys.readouterr().err


def test_overrides_and_files_refuse_a_sphere(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match=MSG):
        apply_overrides(load_config(preset="tiny3d"), ["world.geometry=sphere"])
    toml = tmp_path / "c.toml"
    toml.write_text('[world]\ngeometry = "sphere"\n')
    with pytest.raises(ConfigError, match=MSG):
        load_config(path=toml, preset="tiny3d")


def test_a_run_directory_or_a_checkpoint_naming_a_sphere_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with monkeypatch.context() as m:
        m.setattr(config, "GEOMETRY_SUPPORTED", ("box", "sphere"))
        blob = json.dumps(config_to_dict(sphere_cfg("tiny3s"), omit_defaults=True))
    # a config.json written by an engine that ran spheres, read by this one
    (tmp_path / "config.json").write_text(blob)
    (tmp_path / "meta.json").write_text(json.dumps({"format_version": 1}))
    with pytest.raises(ConfigError, match=MSG):
        RunDir(tmp_path).config()
    # a state file's embedded config (resume, fork, migrate and inspect read it through these two)
    monkeypatch.setattr(cli, "checkpoint_metadata", lambda path: {"config": blob})
    with pytest.raises(ConfigError, match=MSG):
        cli._embedded_config(tmp_path / "x.safetensors")
    from alife import migrate

    monkeypatch.setattr(migrate, "checkpoint_metadata", lambda path: {"config": blob})
    with pytest.raises(ConfigError, match=MSG):
        migrate.embedded_config(tmp_path / "x.safetensors")


@pytest.mark.usefixtures("sphere_enabled")
def test_a_fork_cannot_change_the_geometry_of_its_source(tmp_path: Path) -> None:
    """The fork guard covers the geometry keys (a state never crosses geometries): a box checkpoint
    forked into a
    sphere config is refused with the checkpoint error, exit 4, not silently loaded into the wrong shapes."""
    out = tmp_path / "src"
    assert cli.main(["run", "--preset", "tiny3d", "--set", "brain.weight_dtype=float32", "--steps", "50",
                     "--out", str(out), "--quiet"]) == 0  # fmt: skip
    ckpt = RunDir.open(out).checkpoint_at(50)
    sets = [
        "world.geometry=sphere", "world.size=1024", "world.radius=36", "world.height=16", "world.nside=8",
        "world.layers=8", "sensing.cell_capacity=32", "physics.floor_eps=0.05",
    ]  # fmt: skip
    argv = ["run", "--init-from", str(ckpt), "--out", str(tmp_path / "fork"), "--quiet", "--steps", "10"]
    for s in sets:
        argv += ["--set", s]
    assert cli.main(argv) == 4
