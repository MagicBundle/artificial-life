"""Where a sphere config enters the engine, and what S2 enables (DESIGN §46.9, §46.12; sphere series S2).

Every entry point funnels into ``config.validate`` (``load_config``, ``apply_overrides``, the config embedded
in a run directory or in a checkpoint's metadata). S2 widened ``GEOMETRY_SUPPORTED``: the refusal text and its
funnel are checked with the tuple narrowed again; a sphere ``run``, ``resume`` and ``bench`` work; the tools
that read the box geometry and the layers that are not converted (``SPHERE_READY``) refuse with the PR that
converts them. S5 converted the observers, the director, ``stream`` and ``play``; the weather (S4) is left.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from alife import cli, config, migrate
from alife.config import ConfigError, apply_overrides, config_to_dict, load_config
from alife.runio import RunDir
from sphere_helpers import sphere_cfg, sphere_enabled  # noqa: F401  (a fixture)

SPHERE_SET = ["--set", "world.geometry=sphere"]
MSG = "not enabled yet"


@pytest.fixture
def narrowed(monkeypatch: pytest.MonkeyPatch) -> None:
    """The state before S2: the funnel refuses a sphere with the text of ``_validate_sphere``."""
    monkeypatch.setattr(config, "GEOMETRY_SUPPORTED", ("box",))


@pytest.mark.usefixtures("narrowed")
@pytest.mark.parametrize(
    "argv",
    [
        ["run", "--preset", "tiny3d", *SPHERE_SET, "--quiet"],
        ["bench", "--preset", "tiny3d", *SPHERE_SET],
        ["play", "--preset", "tiny3d", *SPHERE_SET],
    ],
    ids=["run", "bench", "play"],
)
def test_the_funnel_refuses_a_sphere_when_geometry_is_narrowed(
    argv: list[str], capsys: pytest.CaptureFixture
) -> None:
    assert cli.main(argv) == 2
    assert MSG in capsys.readouterr().err


@pytest.mark.usefixtures("narrowed")
def test_overrides_and_files_refuse_a_narrowed_sphere(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match=MSG):
        apply_overrides(load_config(preset="tiny3d"), ["world.geometry=sphere"])
    toml = tmp_path / "c.toml"
    toml.write_text('[world]\ngeometry = "sphere"\n')
    with pytest.raises(ConfigError, match=MSG):
        load_config(path=toml, preset="tiny3d")


def test_a_run_directory_or_a_checkpoint_naming_a_sphere_is_refused_when_narrowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    blob = json.dumps(config_to_dict(sphere_cfg("tiny3s"), omit_defaults=True))
    monkeypatch.setattr(config, "GEOMETRY_SUPPORTED", ("box",))
    # a config.json written by an engine that ran spheres, read by one that does not
    (tmp_path / "config.json").write_text(blob)
    (tmp_path / "meta.json").write_text(json.dumps({"format_version": 1}))
    with pytest.raises(ConfigError, match=MSG):
        RunDir(tmp_path).config()
    # a state file's embedded config (resume, fork, migrate and inspect read it through these two)
    monkeypatch.setattr(cli, "checkpoint_metadata", lambda path: {"config": blob})
    with pytest.raises(ConfigError, match=MSG):
        cli._embedded_config(tmp_path / "x.safetensors")
    monkeypatch.setattr(migrate, "checkpoint_metadata", lambda path: {"config": blob})
    with pytest.raises(ConfigError, match=MSG):
        migrate.embedded_config(tmp_path / "x.safetensors")


def test_run_inspect_resume_and_bench_run_a_sphere(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """S2 enables the sphere: `run` writes a run directory with ``geometry`` in its meta and its checkpoints,
    `inspect` labels the planet, `resume` continues bitwise, `bench` times it."""
    out = tmp_path / "run"
    base = ["--preset", "tiny3s", "--set", "brain.weight_dtype=float32"]
    # not quiet: the announce line of `run` and `resume` names the planet, not a world size (S3 review H1)
    assert cli.main(["run", *base, "--steps", "20", "--out", str(out)]) == 0
    first = capsys.readouterr().out
    assert "R=36 Nside=8 height=16" in first and "steps 0 → 20" in first and "L=" not in first
    rd = RunDir.open(out)
    assert rd.meta()["geometry"] == "sphere" and "dims" in rd.meta()
    assert cli.main(["inspect", str(out)]) == 0
    text = capsys.readouterr().out
    assert "geometry=sphere" in text and "planet_radius=36" in text and "ledger OK" in text
    assert cli.main(["resume", str(out), "--until", "30"]) == 0
    assert "R=36 Nside=8" in capsys.readouterr().out
    assert RunDir.open(out).meta()["status"] == "done"
    assert cli.main(["bench", *base, "--steps", "3", "--warmup", "2"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["config"]["geometry"] == "sphere" and report["config"]["planet_radius"] == 36.0
    assert report["invariants_ok"] and "world_size" not in report["config"]


@pytest.mark.parametrize(
    ("argv", "pr"),
    [
        (["run", "--preset", "tiny3s", "--set", "weather.enabled=true"], "S4"),
        (["run", "--preset", "tiny3s", "--set", "game.observe=true"], "S5"),
        (["run", "--preset", "tiny3s", "--set", "director.enabled=true"], "S5"),
    ],
)
def test_layers_outside_the_readiness_set_are_refused_with_their_pr(
    argv: list[str], pr: str, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every layer of the series is ready (S5); the gate is still the readiness set, read at call time."""
    monkeypatch.setattr(config, "SPHERE_READY", frozenset())
    assert cli.main(argv if argv[0] == "play" else [*argv, "--quiet"]) == 2
    err = capsys.readouterr().err
    assert "is not available on a sphere yet" in err and f"(PR {pr})" in err


S3_LAYERS = {  # the layers PR S3 converts: the settings that switch each on (roles brings bonds)
    "structures": ["structures.enabled=true"],
    "arms": ["arms.enabled=true"],
    "comm": ["comm.enabled=true"],
    "niche": ["niche.farm=true"],
    "bonds": ["bonds.share_mode=2"],
    "roles": ["roles.enabled=true"],
}


@pytest.mark.parametrize("name", sorted(S3_LAYERS))
def test_a_layer_is_refused_with_its_pr_until_it_is_in_the_readiness_set(
    name: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """Outside ``SPHERE_READY`` a layer is refused with the PR that converts it (S3 for these), on the CLI
    too; inside it the gate lets the layer through (its own rules apply). The set is read at call time."""
    sets = S3_LAYERS[name]
    monkeypatch.setattr(config, "SPHERE_READY", config.SPHERE_READY - {name})
    with pytest.raises(ConfigError, match=rf"{name} is not available on a sphere yet \(PR S3\)"):
        load_config(preset="tiny3s", overrides=sets)
    argv = ["run", "--preset", "tiny3s", *(x for s in sets for x in ("--set", s)), "--quiet"]
    assert cli.main(argv) == 2
    assert f"{name} is not available on a sphere yet (PR S3)" in capsys.readouterr().err
    monkeypatch.setattr(config, "SPHERE_READY", config.SPHERE_READY | {"bonds", "roles", name})
    cfg = load_config(preset="tiny3s", overrides=[*sets, "brain.weight_dtype=float32"])
    assert name in config.sphere_wanted(cfg)


def test_a_sphere_has_no_migration(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    out = tmp_path / "run"
    assert cli.main(["run", "--preset", "tiny3s", "--steps", "10", "--out", str(out), "--quiet"]) == 0
    assert cli.main(["migrate", str(out), "--enable", "niche", "--out", str(tmp_path / "m")]) != 0
    assert "a sphere world has no migration" in capsys.readouterr().err


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


def test_a_sphere_checkpoint_never_loads_into_a_box_config_and_back(tmp_path: Path) -> None:
    """``load_checkpoint`` compares the file's ``geometry`` (absent = box) with ``world.geometry``: the shapes
    differ, so a state never crosses geometries (``CheckpointError`` in the library, exit 4 in the CLI)."""
    from alife.runio import CheckpointError, load_checkpoint, write_state_file
    from alife.state import init_state

    sph, box = load_config(preset="tiny3s"), load_config(preset="tiny3d")
    f_sph, f_box = tmp_path / "sph.safetensors", tmp_path / "box.safetensors"
    write_state_file(f_sph, init_state(sph), sph)
    write_state_file(f_box, init_state(box), box)
    with pytest.raises(CheckpointError, match="cross-geometry resume/fork is not supported"):
        load_checkpoint(f_sph, box, check_hash=False)
    with pytest.raises(CheckpointError, match="cross-geometry resume/fork is not supported"):
        load_checkpoint(f_box, sph, check_hash=False)
    assert load_checkpoint(f_sph, sph).pos.shape == (1024, 3)
