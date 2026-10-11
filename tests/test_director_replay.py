"""Replay (DIRECTOR.md §2.1): a ``director.jsonl`` replays bitwise, in a new directory and in place.

A scripted policy (a monkeypatched ``rules.decide``) takes every action, with timed effects whose reverts
fall after a checkpoint, so the whole path (clamp, log, apply, revert fold, checkpoint) is exercised on a tiny
world. The timed menu bounds (5k-200k steps) are shrunk for these tests by a module-scoped patch; the real
bounds are tested in ``test_director_replay_slow.py`` (a separate module, so the patch cannot leak into it).
"""

from __future__ import annotations

import dataclasses
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from alife import cli, runner
from alife.config import Config, apply_overrides
from alife.director import Director, actions
from alife.director.rules import Proposal
from alife.runio import (
    DIRECTOR_ARCHIVE_FILE,
    DIRECTOR_LOG_FILE,
    JOURNAL_FILE,
    RunDir,
    load_checkpoint,
    read_jsonl,
)
from alife.state import init_state, state_digest
from helpers import scripted_decide

PERF = {"wall_s", "steps_per_s", "peak_mem_gb"}
EVERY, END = 100, 1200


def prop(action: str, raw: dict, duration: int | None = None) -> Proposal:
    return Proposal(action, raw, duration, "scripted", f"scripted {action}")


PLAN = {
    100: [
        prop("set_seasons", {"period": 5000, "amp": 0.5}),
        prop("regen_pulse", {"mult": 1.4}, 300),  # reverts at 400, after the checkpoint at 300
        prop("fertility_shift", {"patch": 1, "gain": 1.5}),
    ],
    200: [
        prop("habitat_loss", {"patch": 2}, 500),
        prop("immigration_pulse", {"count": 60}),
    ],  # reverts at 700
    300: [prop("fertility_shift", {"patch": 0, "dx": 12.0, "dy": -9.0})],
    500: [
        prop("catastrophe", {"x": 40.0, "y": 70.0, "radius": 30.0, "kill_p": 0.6}),
        prop("drift_change", {"mult": 3.0}),
    ],
    600: [prop("regen_pulse", {"mult": 0.6}, 200)],  # reverts at 800
    900: [prop("immigration_pulse", {"count": 30}), prop("set_seasons", {"period": 3000, "amp": 0.2})],
}


@pytest.fixture(scope="module")
def cfg(tiny_cfg: Config) -> Config:
    return apply_overrides(
        tiny_cfg,
        [
            "director.enabled=true",
            f"director.rule_every={EVERY}",
            "run.director=live",
            "run.metrics_every=10",
            "run.heavy_metrics_every=100",
            "run.checkpoint_every=300",
            "run.keep_checkpoints=20",
            "run.diversity_sample=64",
        ],
    )


@pytest.fixture(scope="module")
def scripted() -> Iterator[None]:
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("alife.director.rules.decide", scripted_decide(PLAN))
        for name in ("regen_pulse", "habitat_loss"):  # reverts within a 1200-step test run
            mp.setitem(actions.MENU, name, dataclasses.replace(actions.MENU[name], duration=(100, 2000)))
        yield


@pytest.fixture(scope="module")
def live(cfg: Config, scripted: None, tmp_path_factory: pytest.TempPathFactory) -> Path:
    rd = RunDir.create(tmp_path_factory.mktemp("live") / "live", cfg)
    rd.update_meta(start_step=0)
    res = runner.run(rd, cfg, init_state(cfg), END, quiet=True, director=Director(rd, cfg, "live", 0))
    assert res.status == "done"
    return rd.path


def rows(path: Path) -> list[dict[str, Any]]:
    return [{k: v for k, v in r.items() if k not in PERF} for r in RunDir.open(path).metrics_rows()]


def digest(path: Path, step: int = END) -> str:
    rd = RunDir.open(path)
    return state_digest(load_checkpoint(rd.checkpoint_at(step), rd.config()))


def test_the_scripted_run_logs_every_action_and_a_revert_across_a_checkpoint(live: Path) -> None:
    recs = read_jsonl(live / DIRECTOR_LOG_FILE)
    assert recs[0]["kind"] == "header"
    entries = recs[1:]
    assert {e["action"] for e in entries} == {
        "set_seasons",
        "fertility_shift",
        "habitat_loss",
        "regen_pulse",
        "immigration_pulse",
        "catastrophe",
        "drift_change",
    }
    reverts = [(e["step_applied"], e["action"], e["reverts"]) for e in entries if e["source"] == "revert"]
    assert reverts == [(400, "regen_pulse", 100), (700, "habitat_loss", 200), (800, "regen_pulse", 600)]
    assert [e["step_applied"] for e in entries] == sorted(e["step_applied"] for e in entries)
    revert = next(e for e in entries if e["step_applied"] == 400)
    assert revert["params"] == {"mult": 1.0}  # explicit restore values
    move = next(e for e in entries if e["action"] == "fertility_shift" and "x" in e["params"])
    assert set(move["params"]) == {"patch", "x", "y"} and set(move["prev"]) == {"patch", "x", "y"}  # absolute
    assert all(e["source"] in ("rule", "revert") for e in entries)
    assert sorted({s for s, _ in RunDir.open(live).checkpoints()}) == [0, 300, 600, 900, 1200]
    assert (live / JOURNAL_FILE).exists() and (live / DIRECTOR_ARCHIVE_FILE).exists()
    assert cli.main(["inspect", str(live), "--check"]) == 0


def test_replay_into_a_fresh_directory_is_bitwise_identical(live: Path, cfg: Config, tmp_path: Path) -> None:
    rcfg = apply_overrides(cfg, ["run.director=replay"])
    rd = RunDir.create(tmp_path / "replay", rcfg)
    rd.update_meta(start_step=0)
    shutil.copyfile(live / DIRECTOR_LOG_FILE, rd.path / DIRECTOR_LOG_FILE)
    director = Director(rd, rcfg, "replay", 0)
    res = runner.run(rd, rcfg, init_state(rcfg), END, quiet=True, director=director)
    assert res.status == "done"
    assert digest(rd.path) == digest(live)
    assert rows(rd.path) == rows(live)  # every metrics row, env summary included, perf keys aside
    for step in (300, 600, 900):
        assert digest(rd.path, step) == digest(live, step)
    assert (rd.path / DIRECTOR_LOG_FILE).read_bytes() == (live / DIRECTOR_LOG_FILE).read_bytes()
    assert (
        not (rd.path / JOURNAL_FILE).exists() and not (rd.path / DIRECTOR_ARCHIVE_FILE).exists()
    )  # no new records
    assert cli.main(["inspect", str(rd.path), "--check"]) == 0


def test_the_director_changes_the_trajectory(live: Path, cfg: Config, tmp_path: Path) -> None:
    """The comparison above is not vacuous: the same run without the director ends elsewhere."""
    off = apply_overrides(cfg, ["run.director=off"])
    rd = RunDir.create(tmp_path / "off", off)
    runner.run(rd, off, init_state(off), END, quiet=True)
    assert digest(rd.path) != digest(live)


@pytest.mark.parametrize("from_ckpt", [300, 600, 900])
def test_replay_in_place_from_a_mid_run_checkpoint(live: Path, tmp_path: Path, from_ckpt: int) -> None:
    copy = tmp_path / "copy"
    shutil.copytree(live, copy)
    log_before = (copy / DIRECTOR_LOG_FILE).read_bytes()
    journal_before = (copy / JOURNAL_FILE).read_bytes()
    code = cli.main(
        [
            "resume",
            str(copy),
            "--director",
            "replay",
            "--checkpoint",
            str(from_ckpt),
            "--until",
            str(END),
            "--quiet",
        ]
    )
    assert code == 0
    assert (copy / DIRECTOR_LOG_FILE).read_bytes() == log_before  # a replay never truncates or writes the log
    assert (copy / JOURNAL_FILE).read_bytes() == journal_before
    assert (
        RunDir.open(copy).config().run.director == "replay"
    )  # the mode is written back, like any resume key
    assert digest(copy) == digest(live)
    assert rows(copy) == rows(live)


def test_replay_with_a_different_metrics_cadence_still_visits_the_log_steps(
    live: Path, tmp_path: Path
) -> None:
    out = tmp_path / "cadence"
    argv = [
        "run",
        "--init-from",
        f"{live}:0",
        "--director",
        "replay",
        "--replay-log",
        str(live / DIRECTOR_LOG_FILE),
    ]
    argv += ["--metrics-every", "20", "--steps", str(END), "--quiet", "--out", str(out)]
    assert cli.main(argv) == 0
    assert digest(out) == digest(live)
    assert len(rows(out)) == END // 20 and len(rows(live)) == END // 10
    assert (out / DIRECTOR_LOG_FILE).read_bytes() == (
        live / DIRECTOR_LOG_FILE
    ).read_bytes()  # copied verbatim
    # a cadence that skips a log step is not allowed to replay at all: rule_every must stay a multiple
    bad = [
        "run",
        "--init-from",
        f"{live}:0",
        "--director",
        "replay",
        "--replay-log",
        str(live / DIRECTOR_LOG_FILE),
    ]
    assert (
        cli.main([*bad, "--metrics-every", "30", "--steps", "100", "--quiet", "--out", str(tmp_path / "x")])
        == 2
    )


def test_a_replay_fork_ignores_entries_before_its_start_step(live: Path, tmp_path: Path) -> None:
    out = tmp_path / "late"
    argv = [
        "run",
        "--init-from",
        f"{live}:600",
        "--director",
        "replay",
        "--replay-log",
        str(live / DIRECTOR_LOG_FILE),
    ]
    assert cli.main([*argv, "--steps", "600", "--quiet", "--out", str(out)]) == 0
    assert (out / DIRECTOR_LOG_FILE).read_bytes() == (live / DIRECTOR_LOG_FILE).read_bytes()
    assert RunDir.open(out).meta()["start_step"] == 600 and cli.main(["inspect", str(out), "--check"]) == 0


def test_live_resume_from_a_checkpoint_regenerates_the_same_log(live: Path, tmp_path: Path) -> None:
    """The log is cut to the checkpoint and the pending reverts come from folding it: a live resume
    reproduces the uninterrupted run, with a revert (regen at 400) still pending at the checkpoint (300)."""
    copy = tmp_path / "copy"
    shutil.copytree(live, copy)
    code = cli.main(
        ["resume", str(copy), "--checkpoint", "300", "--until", str(END), "--director", "live", "--quiet"]
    )
    assert code == 0
    assert (copy / DIRECTOR_LOG_FILE).read_bytes() == (live / DIRECTOR_LOG_FILE).read_bytes()
    assert (copy / JOURNAL_FILE).read_bytes() == (live / JOURNAL_FILE).read_bytes()
    assert digest(copy) == digest(live) and rows(copy) == rows(live)


def test_real_rules_live_run_and_its_replay_are_bitwise_identical(tmp_path: Path) -> None:
    """Whether or not the thresholds fire on a tiny world, a live run replays exactly (here also resumed)."""
    common = [
        "--preset",
        "tiny",
        "--set",
        "brain.weight_dtype=float32",
        "--set",
        f"director.rule_every={EVERY}",
    ]
    common += ["--set", "run.heavy_metrics_every=100", "--set", "run.diversity_sample=64", "--quiet"]
    one, two, rep = tmp_path / "one", tmp_path / "two", tmp_path / "rep"
    steps = ["--steps", "600", "--set", "run.checkpoint_every=200"]
    assert cli.main(["run", *common, "--director", "live", *steps, "--out", str(one)]) == 0
    assert (
        cli.main(
            [
                "run",
                *common,
                "--director",
                "live",
                "--steps",
                "200",
                "--set",
                "run.checkpoint_every=200",
                "--out",
                str(two),
            ]
        )
        == 0
    )
    assert cli.main(["resume", str(two), "--until", "600", "--quiet"]) == 0  # inherits the live mode
    for name in (DIRECTOR_LOG_FILE, JOURNAL_FILE, DIRECTOR_ARCHIVE_FILE):
        a, b = one / name, two / name
        assert a.exists() == b.exists()
        if a.exists():
            assert a.read_bytes() == b.read_bytes(), (
                name
            )  # the rebuilt archive and the decisions are the same
    assert digest(one, 600) == digest(two, 600)
    argv = [
        "run",
        "--init-from",
        f"{one}:0",
        "--director",
        "replay",
        "--replay-log",
        str(one / DIRECTOR_LOG_FILE),
    ]
    assert (
        cli.main([*argv, "--steps", "600", "--set", "run.checkpoint_every=200", "--quiet", "--out", str(rep)])
        == 0
    )
    assert digest(rep, 600) == digest(one, 600)
    assert rows(rep) == rows(one)
