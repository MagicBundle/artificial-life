"""In-memory re-simulation of ``alife.analysis.resim``: exact states, verified rows, a read-only run."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import mlx.core as mx
import pytest

from alife import runner
from alife.analysis import resim
from alife.config import Config, apply_overrides
from alife.runio import RunDir, load_checkpoint
from alife.state import init_state, state_digest
from alife.step import make_step


@pytest.fixture(scope="module")
def run_dir(tiny_cfg: Config, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A 40-step tiny run with metrics every 5 steps and checkpoints at 0, 20 and 40 only."""
    cfg = apply_overrides(
        tiny_cfg, ["run.metrics_every=5", "run.checkpoint_every=20", "run.heavy_metrics_every=5"]
    )
    path = tmp_path_factory.mktemp("resim") / "run"
    rd = RunDir.create(path, cfg)
    assert runner.run(rd, cfg, init_state(cfg), 40, quiet=True).status == "done"
    return path


def _files(path: Path) -> dict[str, int]:
    return {str(p.relative_to(path)): p.stat().st_mtime_ns for p in path.rglob("*") if p.is_file()}


def test_resim_matches_stepping_and_leaves_the_run_alone(run_dir: Path, tmp_path: Path) -> None:
    before = _files(run_dir)
    out = tmp_path / "cache"
    logs: list[str] = []
    paths = resim.resimulate(run_dir, [30, 35], out, log=logs.append)
    assert [p.name for p in paths] == ["step_0000000030.safetensors", "step_0000000035.safetensors"]
    assert _files(run_dir) == before  # nothing written into the run directory

    rd = RunDir.open(run_dir)
    cfg = rd.config()
    s = load_checkpoint(rd.checkpoint_at(20), cfg)
    step = make_step(cfg)
    for _ in range(15):
        s, _ = step(s)
        mx.eval(s)
    assert state_digest(load_checkpoint(paths[1], cfg)) == state_digest(s)
    assert [st for st, _ in resim.cached_states(out)] == [30, 35]
    report = json.loads((out / "resim.json").read_text())
    assert report["steps"] == [30, 35] and report["rows_verified_last_pass"] == 3  # rows 25, 30, 35

    assert resim.resimulate(run_dir, [30], out, log=logs.append) == [paths[0]]  # cached: no new pass


def test_resim_detects_a_diverging_row(run_dir: Path, tmp_path: Path) -> None:
    bad = tmp_path / "run"
    shutil.copytree(run_dir, bad)
    rows = [json.loads(line) for line in (bad / "metrics.jsonl").read_text().splitlines()]
    for r in rows:
        if r["step"] == 25:
            r["births"] += 1
    (bad / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(resim.ResimMismatch, match="step 25"):
        resim.resimulate(bad, [30], tmp_path / "cache", log=lambda _: None)
    assert resim.resimulate(bad, [30], tmp_path / "cache2", verify=False, log=lambda _: None)
