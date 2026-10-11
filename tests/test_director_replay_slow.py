"""Replay with the real timed menu bounds (DIRECTOR.md §2.1, §5): a 5k regen pulse and a 10k habitat loss,
reverted and replayed bitwise.

Kept apart from ``test_director_replay.py``, whose module-scoped fixture shrinks the timed bounds until that
module ends.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from alife import cli, runner
from alife.config import Config, apply_overrides
from alife.director import Director
from alife.director.rules import Proposal
from alife.runio import DIRECTOR_LOG_FILE, RunDir, load_checkpoint, read_jsonl
from alife.state import init_state, state_digest
from helpers import scripted_decide


def prop(action: str, raw: dict, duration: int | None = None) -> Proposal:
    return Proposal(action, raw, duration, "scripted", f"scripted {action}")


def digest(path: Path, step: int) -> str:
    rd = RunDir.open(path)
    return state_digest(load_checkpoint(rd.checkpoint_at(step), rd.config()))


@pytest.mark.slow
def test_replay_with_the_real_timed_bounds(
    tiny_cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unshrunk menu: a regen pulse of 5k and a habitat loss of 10k steps, reverted and replayed bitwise."""
    plan = {
        500: [prop("regen_pulse", {"mult": 1.3}, 5000), prop("habitat_loss", {"patch": 1}, 10_000)],
        3000: [prop("immigration_pulse", {"count": 50})],
    }
    monkeypatch.setattr("alife.director.rules.decide", scripted_decide(plan))
    cfg = apply_overrides(
        tiny_cfg,
        [
            "director.enabled=true",
            "director.rule_every=500",
            "run.director=live",
            "run.metrics_every=100",
            "run.heavy_metrics_every=500",
            "run.checkpoint_every=5000",
            "run.keep_checkpoints=10",
            "run.diversity_sample=64",
        ],
    )
    end = 11_000
    rd = RunDir.create(tmp_path / "live", cfg)
    runner.run(rd, cfg, init_state(cfg), end, quiet=True, director=Director(rd, cfg, "live", 0))
    entries = read_jsonl(rd.path / DIRECTOR_LOG_FILE)[1:]
    assert [(e["step_applied"], e["source"]) for e in entries if e["action"] != "immigration_pulse"] == [
        (500, "rule"),
        (500, "rule"),
        (5500, "revert"),
        (10_500, "revert"),
    ]
    rep = tmp_path / "rep"
    argv = [
        "run",
        "--init-from",
        f"{rd.path}:0",
        "--director",
        "replay",
        "--replay-log",
        str(rd.path / DIRECTOR_LOG_FILE),
    ]
    assert cli.main([*argv, "--steps", str(end), "--quiet", "--out", str(rep)]) == 0
    assert digest(rep, end) == digest(rd.path, end)
