"""New worlds keep only their newest save (DESIGN §34.10, row I99, the owner's decision of 2026-11-05).

``run.saves`` is the research run's save mode (``all``, ``newest`` or ``single``). The dataclass default is
the old behaviour (a config.json without the key resumes as before, with the same hash); ``alife run`` writes
``single`` into a NEW run's config unless the user set it. ``alife play`` of a new world starts in ``single``;
a world that exists keeps its own mode, and an adopted research run keeps ``all``. All runs here are the
``tiny`` preset (checkpoint every 50 steps) on the CPU."""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from alife import cli
from alife.config import Config, ConfigError, config_hash, config_to_dict, load_config
from alife.runio import RunDir
from play_helpers import spawn

TINY = ["--preset", "tiny", "--set", "brain.weight_dtype=float32", "--set", "run.diversity_sample=64"]


@pytest.fixture(scope="module", autouse=True)
def cli_real_default(real_default_saves: Any) -> Iterator[None]:
    """The suite replaces ``cli._with_default_saves`` by the identity (``conftest.pytest_configure``); these
    tests are the ones that run the real default."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(cli, "_with_default_saves", real_default_saves)
        yield


def steps_of(rd: RunDir) -> list[int]:
    return [st for st, _ in rd.checkpoints()]


def run(out: Path, *extra: str, steps: int = 100) -> RunDir:
    assert cli.main(["run", *TINY, "--steps", str(steps), "--out", str(out), "--quiet", *extra]) == 0
    return RunDir.open(out)


@pytest.fixture(scope="module")
def fresh(tmp_path_factory: pytest.TempPathFactory) -> RunDir:
    return run(tmp_path_factory.mktemp("saves") / "fresh")


@pytest.fixture(scope="module")
def kept_all(tmp_path_factory: pytest.TempPathFactory) -> RunDir:
    return run(tmp_path_factory.mktemp("saves") / "all", "--set", "run.saves=all")


# --- the config key ------------------------------------------------------------------------------------


def test_the_default_is_the_old_behaviour_and_moves_no_hash() -> None:
    cfg = Config()
    assert cfg.run.saves == "all"
    assert "saves" not in config_to_dict(cfg, omit_defaults=True)["run"]  # an old config.json is byte-equal
    other = load_config(preset="tiny", overrides=["run.saves=single"])
    assert config_hash(other) == config_hash(load_config(preset="tiny"))
    assert config_to_dict(other, omit_defaults=True)["run"]["saves"] == "single"
    with pytest.raises(ConfigError, match=r"run\.saves"):
        load_config(preset="tiny", overrides=["run.saves=some"])
    assert "saves" in cli.RESUME_KEYS


# --- a fresh run, a fork, the opt-out ------------------------------------------------------------------


def test_a_fresh_run_keeps_only_its_newest_save(fresh: RunDir) -> None:
    assert fresh.config().run.saves == "single"
    assert json.loads((fresh.path / "config.json").read_text())["run"]["saves"] == "single"
    assert steps_of(fresh) == [100]  # step 0 and the save at 50 are gone
    assert (fresh.checkpoint_dir / "LATEST").read_text().strip() == "step_0000000100.safetensors"


def test_the_second_save_removes_step_0(tmp_path: Path) -> None:
    rd = run(tmp_path / "w", steps=50)  # the saves are step 0 (the start) and 50 (the end)
    assert steps_of(rd) == [50]


def test_a_fork_keeps_only_its_newest_save_and_not_its_origin(fresh: RunDir, tmp_path: Path) -> None:
    out = tmp_path / "fork"
    assert (
        cli.main(["run", "--init-from", str(fresh.path), "--steps", "50", "--out", str(out), "--quiet"]) == 0
    )
    rd = RunDir.open(out)
    assert rd.meta()["start_step"] == 100 and rd.config().run.saves == "single"
    assert steps_of(rd) == [150]  # the origin at 100 went too
    assert steps_of(RunDir.open(fresh.path)) == [100]  # the source is untouched


def test_saves_all_restores_the_old_pruning(kept_all: RunDir, tmp_path: Path) -> None:
    assert kept_all.config().run.saves == "all"
    assert "saves" not in json.loads((kept_all.path / "config.json").read_text())["run"]
    assert steps_of(kept_all) == [0, 50, 100]  # keep_checkpoints 3, plus step 0
    out = tmp_path / "fork"
    args = ["run", "--init-from", str(kept_all.path), "--steps", "100", "--out", str(out), "--quiet"]
    assert cli.main([*args, "--set", "run.saves=all", "--set", "run.keep_checkpoints=1"]) == 0
    assert steps_of(RunDir.open(out)) == [100, 200]  # the origin and the newest one


def test_a_config_file_that_sets_run_saves_is_respected(tmp_path: Path) -> None:
    cfg = tmp_path / "c.toml"
    cfg.write_text('[run]\nsaves = "all"\n')
    rd = run(tmp_path / "w", "--config", str(cfg))
    assert rd.config().run.saves == "all" and steps_of(rd) == [0, 50, 100]


# --- a resume --------------------------------------------------------------------------------------------


def test_an_old_run_resumes_with_the_old_pruning(kept_all: RunDir, tmp_path: Path) -> None:
    """A config.json made before the key (it is omitted at ``all``) loads as ``all`` with the same hash."""
    copy = tmp_path / "old"
    shutil.copytree(kept_all.path, copy)
    text = (copy / "config.json").read_text()
    assert '"saves"' not in text
    h = config_hash(RunDir.open(copy).config())
    assert h == json.loads((copy / "meta.json").read_text())["config_hash"]
    code = cli.main(["resume", str(copy), "--until", "200", "--set", "run.keep_checkpoints=2", "--quiet"])
    assert code == 0
    rd = RunDir.open(copy)
    assert steps_of(rd) == [0, 150, 200] and config_hash(rd.config()) == h
    assert '"saves"' not in (copy / "config.json").read_text()


def test_a_single_run_resumes_single_and_resume_may_change_the_mode(fresh: RunDir, tmp_path: Path) -> None:
    copy = tmp_path / "w"
    shutil.copytree(fresh.path, copy)
    assert cli.main(["resume", str(copy), "--until", "150", "--quiet"]) == 0
    assert steps_of(RunDir.open(copy)) == [150]
    assert cli.main(["resume", str(copy), "--until", "200", "--set", "run.saves=all", "--quiet"]) == 0
    rd = RunDir.open(copy)
    assert steps_of(rd) == [150, 200] and rd.config().run.saves == "all"


# --- a played world --------------------------------------------------------------------------------------


def play(*args: str, until: int) -> tuple[int, str]:
    p = spawn("play", *args, "--rate", "max", "--until", str(until))
    assert p.stdin is not None and p.stdout is not None and p.stderr is not None
    p.stdout.read()
    code = p.wait(timeout=240)
    err = p.stderr.read().decode(errors="replace")
    p.stdin.close()
    return code, err


def test_a_new_played_world_defaults_to_single(tmp_path: Path) -> None:
    out = tmp_path / "w"
    code, err = play(*TINY, "--out", str(out), until=10)
    assert code == 0, err
    rd = RunDir.open(out)
    assert rd.read_play_json()["saves"] == "single"  # type: ignore[index]
    assert steps_of(rd) == [10] and rd.config().run.saves == "all"  # play has its own mode
    # a resume keeps the world's mode
    code, err = play(str(out), until=20)
    assert code == 0, err
    assert steps_of(RunDir.open(out)) == [20]


def test_saves_all_on_a_new_world_is_the_old_default(tmp_path: Path) -> None:
    out = tmp_path / "w"
    code, err = play(*TINY, "--out", str(out), "--saves", "all", until=10)
    assert code == 0, err
    rd = RunDir.open(out)
    assert "saves" not in (rd.read_play_json() or {}) and steps_of(rd) == [0, 10]


def test_an_existing_play_json_keeps_its_mode(tmp_path: Path) -> None:
    """A world made before the change: its play.json says nothing, so it is ``all``."""
    out = tmp_path / "w"
    code, err = play(*TINY, "--out", str(out), "--saves", "all", until=10)
    assert code == 0, err
    rd = RunDir.open(out)
    saved = rd.read_play_json()
    assert saved is not None and "saves" not in saved
    code, err = play(str(out), until=20)
    assert code == 0, err
    rd = RunDir.open(out)
    assert steps_of(rd) == [0, 10, 20] and "saves" not in (rd.read_play_json() or {})


def test_adopting_a_research_run_protects_its_checkpoints(kept_all: RunDir, tmp_path: Path) -> None:
    copy = tmp_path / "adopted"
    shutil.copytree(kept_all.path, copy)
    code, err = play(str(copy), "--adopt", until=110)
    assert code == 0, err
    rd = RunDir.open(copy)
    assert rd.adopted and "saves" not in (rd.read_play_json() or {})
    assert steps_of(rd) == [0, 50, 100, 110]  # nothing of the research is pruned
