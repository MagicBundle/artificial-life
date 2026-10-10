"""The run loop (§18, §21): unscheduled stops (SIGINT, extinction), violations and exit codes. WP5."""

from __future__ import annotations

import signal
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from alife import cli, runner
from alife.config import Config, apply_overrides
from alife.runio import RunDir, load_checkpoint, write_state_file
from alife.runner import ExitCode
from alife.state import Diag, init_state, state_digest, to_numpy
from helpers import make_scene


@pytest.fixture(scope="module")
def cfg(micro_cfg: Config) -> Config:
    return apply_overrides(
        micro_cfg, ["run.metrics_every=5", "run.checkpoint_every=10", "run.heavy_metrics_every=5"]
    )


def _steps(rd: RunDir) -> list[tuple[int, int]]:
    return [(r["step"], r["steps"]) for r in rd.metrics_rows()]


def test_sigint_stops_cleanly_and_resumes_bitwise(cfg: Config, tmp_path: Path, monkeypatch) -> None:
    real = runner.make_step

    def interrupting(c: Config):
        fn, calls = real(c), [0]

        def wrapped(s):
            calls[0] += 1
            if calls[0] == 7:  # Ctrl-C while step 7 (index 6) is being computed
                signal.raise_signal(signal.SIGINT)
            return fn(s)

        return wrapped

    before = signal.getsignal(signal.SIGINT)
    monkeypatch.setattr(runner, "make_step", interrupting)
    rd = RunDir.create(tmp_path / "run", cfg)
    res = runner.run(rd, cfg, init_state(cfg), 20, quiet=True)
    assert (res.status, res.exit_code) == ("interrupted", ExitCode.INTERRUPTED)
    assert signal.getsignal(signal.SIGINT) is before
    assert int(to_numpy(res.state.step)) == 7
    assert _steps(rd) == [(5, 5), (7, 2)]  # the step finished, then a short interval row
    assert [st for st, _ in rd.checkpoints()] == [0, 7]
    assert [(a, b) for a, b, _ in rd.event_files("births")] == [(0, 7)]
    assert rd.meta()["status"] == "interrupted"
    monkeypatch.undo()

    s = load_checkpoint(rd.latest_checkpoint(), cfg)
    res = runner.run(rd, cfg, s, 13, quiet=True)
    assert (res.status, res.exit_code) == ("done", ExitCode.OK)
    assert _steps(rd) == [(5, 5), (7, 2), (10, 3), (15, 5), (20, 5)]  # the first resumed interval is short
    assert [st for st, _ in rd.checkpoints()] == [0, 7, 10, 20]
    assert [(a, b) for a, b, _ in rd.event_files("deaths")] == [(0, 7), (7, 10), (10, 20)]

    fn, ref = runner.make_step(cfg), init_state(cfg)
    for _ in range(20):
        ref, out = fn(ref)
        mx.eval(ref, out)
    assert state_digest(res.state) == state_digest(ref)


def test_extinction_stops_with_status_extinct(cfg: Config, tmp_path: Path) -> None:
    s = make_scene(cfg, [{"pos": [10.0, 10.0], "energy": 1}])  # cannot pay its first step's costs
    rd = RunDir.create(tmp_path / "run", cfg)
    res = runner.run(rd, cfg, s, 100, quiet=True)
    assert (res.status, res.exit_code) == ("extinct", ExitCode.OK)
    assert int(to_numpy(res.state.step)) == 1 and not to_numpy(res.state.alive).any()
    assert _steps(rd) == [(1, 1)] and rd.metrics_rows()[0]["deaths_starved"] == 1
    assert [st for st, _ in rd.checkpoints()] == [0, 1]
    deaths = rd.read_events("deaths")
    assert list(deaths["uid"]) == [0] and list(deaths["step"]) == [0]
    assert rd.meta()["status"] == "extinct"
    # an extinct start runs nothing
    res = runner.run(rd, cfg, res.state, 10, quiet=True)
    assert res.status == "extinct" and _steps(rd) == [(1, 1)]


def test_extinction_with_immigration_continues(cfg: Config, tmp_path: Path) -> None:
    c = apply_overrides(cfg, ["population.immigration_floor=4", "population.immigrants_per_step=2"])
    s = make_scene(c, [{"pos": [10.0, 10.0], "energy": 1}])
    rd = RunDir.create(tmp_path / "run", c)
    res = runner.run(rd, c, s, 5, quiet=True)
    assert (res.status, res.exit_code) == ("done", ExitCode.OK)
    assert int(to_numpy(res.state.alive).sum()) > 0 and _steps(rd) == [(5, 5)]


def test_violation_dumps_state_and_exits_3(cfg: Config, tmp_path: Path, capsys) -> None:
    s = init_state(cfg)
    s = s._replace(diag=mx.array(int(Diag.BOND_ASYM | Diag.VCLIP_WARN), dtype=mx.int32))  # sticky
    rd = RunDir.create(tmp_path / "run", cfg)
    res = runner.run(rd, cfg, s, 20, quiet=True)
    assert (res.status, res.exit_code) == ("violation", ExitCode.INVARIANT)
    assert "BOND_ASYM" in capsys.readouterr().err
    assert (rd.diag_dir / "step_5.safetensors").is_file()
    assert [st for st, _ in rd.checkpoints()] == [0]  # an invalid state is never checkpointed
    assert _steps(rd) == [(5, 5)] and [(a, b) for a, b, _ in rd.event_files("births")] == [(0, 5)]
    assert rd.meta()["status"] == "violation"


def test_warning_bits_do_not_stop_a_run(cfg: Config, tmp_path: Path) -> None:
    s = init_state(cfg)._replace(diag=mx.array(int(Diag.CELL_OVERFLOW_WARN), dtype=mx.int32))
    rd = RunDir.create(tmp_path / "run", cfg)
    res = runner.run(rd, cfg, s, 5, quiet=True)
    assert res.status == "done" and rd.metrics_rows()[0]["diag"] == int(Diag.CELL_OVERFLOW_WARN)


def test_zero_steps_and_status_line(cfg: Config, tmp_path: Path, capsys) -> None:
    rd = RunDir.create(tmp_path / "run", cfg)
    res = runner.run(rd, cfg, init_state(cfg), 0)
    assert (res.status, res.exit_code) == ("done", ExitCode.OK)
    assert [st for st, _ in rd.checkpoints()] == [0] and rd.metrics_rows() == []
    rows = []
    runner.run(rd, cfg, res.state, 5, on_interval=lambda st, row: rows.append(row))
    line = capsys.readouterr().out.strip()
    assert line.startswith("step 5 | pop 8 |") and "largest body" in line and line.endswith("ledger OK")
    assert [r["step"] for r in rows] == [5] and "carrying_capacity_est" in rows[0]
    assert [st for st, _ in rd.checkpoints()] == [0, 5]
    with pytest.raises(ValueError):
        runner.run(rd, cfg, res.state, -1)


def test_diag_names() -> None:
    assert runner.diag_names(0) == []
    assert runner.diag_names(int(Diag.LEDGER | Diag.VCLIP_WARN)) == ["LEDGER", "VCLIP_WARN"]
    assert runner.diag_names(1 << 30) == [hex(1 << 30)]


def test_inspect_check_exits_3_on_violation(cfg: Config, tmp_path: Path) -> None:
    s = init_state(cfg)
    energy = to_numpy(s.energy)
    energy[-1] = 5  # a dead row holding energy
    path = tmp_path / "bad.safetensors"
    write_state_file(path, s._replace(energy=mx.array(energy)), cfg)
    assert cli.main(["inspect", str(path), "--check", "--json"]) == 3
    assert cli.main(["inspect", str(path)]) == 0
    good = tmp_path / "good.safetensors"
    write_state_file(good, s, cfg)
    assert cli.main(["inspect", str(good), "--check"]) == 0
    assert np.all(to_numpy(load_checkpoint(path, cfg).energy)[:-1] == to_numpy(s.energy)[:-1])
