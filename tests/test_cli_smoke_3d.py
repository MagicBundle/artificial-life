"""§27.8 end-to-end smoke test of a 3D run through ``cli.main(argv)`` (no subprocesses).

One 100-step ``tiny3d`` run (f32 weights, as ``tiny3d_cfg``) is shared by the module; tests that change a run
directory work on a copy. The 2D-only tools (``render``, ``view``) refuse it, a fork keeps its dimensions,
height and radius, and ``analysis.resim`` re-simulates it.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from alife import cli, viewer
from alife.analysis import resim
from alife.config import THREE_D_KEYS, Config, config_to_dict, load_config
from alife.runio import RunDir, load_checkpoint
from alife.state import State3, state_digest, to_numpy

F32 = ["--set", "brain.weight_dtype=float32"]
SAMPLE = ["--set", "run.diversity_sample=64"]  # as test_cli_smoke.py: cheap heavy metrics on CPU
PERF_KEYS = {"wall_s", "steps_per_s", "peak_mem_gb"}


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("cli3d") / "run"
    code = cli.main(
        ["run", "--preset", "tiny3d", *F32, *SAMPLE, "--steps", "100", "--out", str(out), "--quiet"]
    )
    assert code == 0
    return out


def _digest(rd: RunDir, step: int) -> str:
    return state_digest(load_checkpoint(rd.checkpoint_at(step), rd.config()))


def _snapshot(path: Path) -> dict[str, bytes]:
    return {str(p.relative_to(path)): p.read_bytes() for p in sorted(path.rglob("*")) if p.is_file()}


def test_run_writes_a_clean_3d_run(run_dir: Path) -> None:
    rd = RunDir.open(run_dir)
    cfg = rd.config()
    assert cfg == load_config(
        preset="tiny3d", overrides=["brain.weight_dtype=float32", "run.diversity_sample=64", "run.steps=100"]
    )
    assert rd.meta()["status"] == "done" and rd.meta()["dims"] == 3
    rows = rd.metrics_rows()
    assert [r["step"] for r in rows] == list(range(10, 101, 10))
    assert all(r["ledger_error"] == 0 and r["diag"] & 0xFFFF == 0 for r in rows)
    assert all("HEAT_BUOYANCY" in r["ledger"] and "RESERVED" not in r["ledger"] for r in rows)
    assert all(r["ledger"]["HEAT_BUOYANCY"] > 0 for r in rows)
    assert [st for st, _ in rd.checkpoints()] == [0, 50, 100]
    s = load_checkpoint(rd.checkpoint_at(100), cfg)
    assert isinstance(s, State3) and int(to_numpy(s.total)) == rows[-1]["E_total"]
    assert to_numpy(s.alive).sum() > 0


def test_inspect_check_3d(run_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    capsys.readouterr()
    assert cli.main(["inspect", str(run_dir), "--check", "--json"]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["ledger_error"] == 0 and info["check"] == {"ok": True, "problems": []}
    assert info["step"] == 100 and info["checkpoints"] == [0, 50, 100] and info["n_alive"] > 0


def test_resume_truncates_and_is_bitwise_3d(run_dir: Path, tmp_path: Path) -> None:
    rd0 = RunDir.open(run_dir)
    ref_rows, ref_digest = rd0.metrics_rows(), _digest(rd0, 100)
    copy = tmp_path / "run"
    shutil.copytree(run_dir, copy)
    assert cli.main(["resume", str(copy), "--checkpoint", "50", "--steps", "20", "--quiet"]) == 0
    rd = RunDir.open(copy)
    assert [r["step"] for r in rd.metrics_rows()] == list(range(10, 71, 10))
    assert [st for st, _ in rd.checkpoints()] == [0, 50, 70]
    assert cli.main(["resume", str(copy), "--quiet"]) == 0
    assert _digest(rd, 100) == ref_digest
    rows = rd.metrics_rows()
    assert len(rows) == len(ref_rows)
    for a, b in zip(rows, ref_rows, strict=True):
        assert {k: v for k, v in a.items() if k not in PERF_KEYS} == {
            k: v for k, v in b.items() if k not in PERF_KEYS
        }
    for kind in ("births", "deaths"):
        new, ref = rd.read_events(kind), rd0.read_events(kind)
        for col in ref:
            np.testing.assert_array_equal(new[col], ref[col], err_msg=f"{kind}.{col}")


def test_render_and_view_refuse_a_3d_run(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[object] = []
    monkeypatch.setattr(viewer, "run_viewer", lambda *a, **k: calls.append(a))
    before = _snapshot(run_dir)
    capsys.readouterr()
    assert cli.main(["render", str(run_dir), "--steps", "2"]) == 2
    assert cli.main(["view", str(run_dir)]) == 2
    assert cli.main(["view", "--preset", "tiny3d"]) == 2
    err = capsys.readouterr().err
    assert err.count("supports 2D worlds only") == 3 and "docs/3D.md" in err
    assert calls == [] and _snapshot(run_dir) == before and not (run_dir / "media").exists()


def _to_2d() -> list[str]:
    """``--set`` items that make a 3D config a valid 2D one: every 3D key back to its default."""
    defaults = config_to_dict(Config())
    return [item for sec, key in THREE_D_KEYS for item in ("--set", f"{sec}.{key}={defaults[sec][key]}")]


def test_a_fork_cannot_change_dims_or_height(
    run_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    before = _snapshot(run_dir)
    out = tmp_path / "fork"
    base = ["run", "--init-from", str(run_dir), "--out", str(out), "--quiet", "--steps", "5"]
    capsys.readouterr()
    assert cli.main([*base, *_to_2d()]) == 4
    assert "world.dims must equal the source's 3" in capsys.readouterr().err
    assert cli.main([*base, "--set", "world.height=32"]) == 4
    assert "world.height must equal the source's 24.0" in capsys.readouterr().err
    assert not out.exists() and _snapshot(run_dir) == before
    assert cli.main([*base, "--set", "physics.gravity=0.02"]) == 0  # a dynamics change in 3D forks
    assert RunDir.open(out).config().physics.gravity == 0.02


def test_a_3d_fork_cannot_change_the_radius(
    run_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    before = _snapshot(run_dir)
    out = tmp_path / "fork"
    base = ["run", "--init-from", str(run_dir), "--out", str(out), "--quiet", "--steps", "5"]
    capsys.readouterr()
    for r in ("1.2", "0.9"):  # the radius sets z_lo and z_hi: the source's rows sit within its own bounds
        assert cli.main([*base, "--set", f"physics.radius={r}"]) == 4
        assert "physics.radius must equal the source's 1.0 in a 3D world" in capsys.readouterr().err
    assert not out.exists() and _snapshot(run_dir) == before


def test_resim_reproduces_a_3d_run(run_dir: Path, tmp_path: Path) -> None:
    """``analysis.resim`` re-simulates a 3D run from its step-50 checkpoint, verifies every stored row on the
    way (30 counters, ``HEAT_BUOYANCY``) and lands on the run's own step-100 state."""
    before = _snapshot(run_dir)
    out = tmp_path / "cache"
    paths = resim.resimulate(run_dir, [60, 100], out, log=lambda _: None)
    assert [p.name for p in paths] == ["step_0000000060.safetensors", "step_0000000100.safetensors"]
    assert json.loads((out / "resim.json").read_text())["rows_verified_last_pass"] == 5  # rows 60 to 100
    rd = RunDir.open(run_dir)
    assert state_digest(load_checkpoint(paths[1], rd.config())) == _digest(rd, 100)
    assert _snapshot(run_dir) == before
