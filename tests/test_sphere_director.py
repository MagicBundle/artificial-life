"""The ecosystem director on a sphere world (DESIGN §46.7; sphere series S5): a scripted live run logs
sphere-form entries (a patch move and a catastrophe as longitude and latitude) and replays bitwise.

Eager (``run.compile=false``): a compiled sphere run belongs in a cold child process (docs/SPHERE.md 10.1).
"""

from __future__ import annotations

import dataclasses
import shutil
from pathlib import Path
from typing import Any

import pytest

from alife import cli, config, geom, runner
from alife.config import apply_overrides, load_config
from alife.director import Director, actions
from alife.director.rules import Proposal
from alife.runio import DIRECTOR_LOG_FILE, RunDir, load_checkpoint, read_jsonl
from alife.state import init_state, state_digest, to_numpy
from helpers import scripted_decide

PERF = {"wall_s", "steps_per_s", "peak_mem_gb"}
EVERY, END = 60, 240


def prop(action: str, raw: dict, duration: int | None = None) -> Proposal:
    return Proposal(action, raw, duration, "scripted", f"scripted {action}")


PLAN = {
    60: [prop("regen_pulse", {"mult": 1.4}, 120), prop("fertility_shift", {"patch": 1, "gain": 1.5})],
    120: [
        prop("fertility_shift", {"patch": 0, "dx": 9.0, "dy": -6.0}),
        prop("immigration_pulse", {"count": 40}),
    ],
    180: [prop("catastrophe", {"x": 100.0, "y": 50.0, "radius": 15.0, "kill_p": 0.6})],
}


@pytest.fixture(scope="module")
def cfg():
    saved = config.SPHERE_READY
    config.SPHERE_READY = saved | {"director", "observers"}
    try:
        return load_config(
            preset="tiny3s",
            overrides=[
                "brain.weight_dtype=float32", "director.enabled=true", f"director.rule_every={EVERY}",
                "run.director=live", "run.compile=false", "run.metrics_every=20",
                "run.heavy_metrics_every=60",
                "run.checkpoint_every=120", "run.keep_checkpoints=20", "run.diversity_sample=64",
            ],
        )  # fmt: skip
    finally:
        config.SPHERE_READY = saved


@pytest.fixture(scope="module")
def live(cfg, tmp_path_factory: pytest.TempPathFactory):
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("alife.director.rules.decide", scripted_decide(PLAN))
        mp.setitem(
            actions.MENU, "regen_pulse", dataclasses.replace(actions.MENU["regen_pulse"], duration=(60, 2000))
        )
        rd = RunDir.create(tmp_path_factory.mktemp("sd") / "live", cfg)
        rd.update_meta(start_step=0)
        res = runner.run(rd, cfg, init_state(cfg), END, quiet=True, director=Director(rd, cfg, "live", 0))
        assert res.status == "done"
        return rd.path


def _rows(path: Path) -> list[dict[str, Any]]:
    return [{k: v for k, v in r.items() if k not in PERF} for r in RunDir.open(path).metrics_rows()]


def _digest(path: Path, step: int = END) -> str:
    rd = RunDir.open(path)
    return state_digest(load_checkpoint(rd.checkpoint_at(step), rd.config()))


def test_the_log_holds_sphere_form_entries(live: Path, cfg) -> None:
    entries = [e for e in read_jsonl(live / DIRECTOR_LOG_FILE) if e.get("kind") != "header"]
    assert {e["action"] for e in entries} == {
        "regen_pulse", "fertility_shift", "immigration_pulse", "catastrophe"
    }  # fmt: skip
    move = next(e for e in entries if e["action"] == "fertility_shift" and "x" in e["params"])
    assert set(move["params"]) == {"patch", "x", "y"} and -90 <= move["params"]["y"] <= 90
    arc = float(
        geom.great_circle_distance(
            geom.from_latlon(move["prev"]["y"], move["prev"]["x"]),
            geom.from_latlon(move["params"]["y"], move["params"]["x"]),
            cfg.world.radius,
        )
    )
    assert arc == pytest.approx(10.8167, rel=1e-3)  # hypot(9, 6)
    cat = next(e for e in entries if e["action"] == "catastrophe")
    assert cat["params"]["center"] == [100.0, 50.0]
    assert cli.main(["inspect", str(live), "--check"]) == 0


def test_the_move_and_the_catastrophe_reach_the_state(live: Path, cfg) -> None:
    rd = RunDir.open(live)
    after = load_checkpoint(rd.checkpoint_at(120), rd.config())  # the move happens at the top of step 120
    before = load_checkpoint(rd.checkpoint_at(0), rd.config())
    p0, p1 = to_numpy(before.patch_pos), to_numpy(after.patch_pos)
    assert p1.shape == p0.shape == (cfg.fields.n_patches, 3)
    assert abs(float(p1[0] @ p0[0]) - 1.0) > 1e-6  # patch 0 moved off its start (patches also rotate: all do)
    end = load_checkpoint(rd.checkpoint_at(END), rd.config())
    import numpy as np

    np.testing.assert_allclose(to_numpy(end.env_cat_center), np.radians([100.0, 50.0]), atol=1e-6)


def test_replay_into_a_fresh_directory_is_bitwise_identical(live: Path, cfg, tmp_path: Path) -> None:
    rcfg = apply_overrides(cfg, ["run.director=replay"])
    rd = RunDir.create(tmp_path / "replay", rcfg)
    rd.update_meta(start_step=0)
    shutil.copyfile(live / DIRECTOR_LOG_FILE, rd.path / DIRECTOR_LOG_FILE)
    res = runner.run(rd, rcfg, init_state(rcfg), END, quiet=True, director=Director(rd, rcfg, "replay", 0))
    assert res.status == "done"
    assert _digest(rd.path) == _digest(live)
    assert _rows(rd.path) == _rows(live)
    assert _digest(rd.path, 120) == _digest(live, 120)
    assert (rd.path / DIRECTOR_LOG_FILE).read_bytes() == (live / DIRECTOR_LOG_FILE).read_bytes()


def test_the_director_changes_the_trajectory(live: Path, cfg, tmp_path: Path) -> None:
    off = apply_overrides(cfg, ["run.director=off"])
    rd = RunDir.create(tmp_path / "off", off)
    runner.run(rd, off, init_state(off), END, quiet=True)
    assert _digest(rd.path) != _digest(live)
