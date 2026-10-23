"""The run loop (§18, §21): unscheduled stops (SIGINT, extinction), violations and exit codes. WP5."""

from __future__ import annotations

import signal
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from alife import cli, runner
from alife.config import Config, apply_overrides, dims
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


def test_sigint_stops_cleanly_and_resumes_bitwise_3d(
    micro3d_cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 3D run interrupted at step 7 and resumed from its checkpoint ends bitwise where an uninterrupted run
    does (``State3``, the 30 counters, ``HEAT_BUOYANCY``; §27.8)."""
    cfg = apply_overrides(
        micro3d_cfg, ["run.metrics_every=5", "run.checkpoint_every=10", "run.heavy_metrics_every=5"]
    )
    real = runner.make_step

    def interrupting(c: Config):
        fn, calls = real(c), [0]

        def wrapped(s):
            calls[0] += 1
            if calls[0] == 7:
                signal.raise_signal(signal.SIGINT)
            return fn(s)

        return wrapped

    monkeypatch.setattr(runner, "make_step", interrupting)
    rd = RunDir.create(tmp_path / "run", cfg)
    res = runner.run(rd, cfg, init_state(cfg), 20, quiet=True)
    assert (res.status, res.exit_code) == ("interrupted", ExitCode.INTERRUPTED)
    assert int(to_numpy(res.state.step)) == 7 and [st for st, _ in rd.checkpoints()] == [0, 7]
    monkeypatch.undo()

    s = load_checkpoint(rd.latest_checkpoint(), cfg)
    res = runner.run(rd, cfg, s, 13, quiet=True)
    assert (res.status, res.exit_code) == ("done", ExitCode.OK)
    assert _steps(rd) == [(5, 5), (7, 2), (10, 3), (15, 5), (20, 5)]
    assert all(r["ledger_error"] == 0 and r["ledger"]["HEAT_BUOYANCY"] > 0 for r in rd.metrics_rows())
    fn, ref = runner.make_step(cfg), init_state(cfg)
    for _ in range(20):
        ref, out = fn(ref)
        mx.eval(ref, out)
    assert state_digest(res.state) == state_digest(ref)


def test_immigrants_arrive_in_the_volume_3d(micro3d_cfg: Config, tmp_path: Path) -> None:
    """After a die-off, immigration refills a 3D world uniformly in the volume, at ``buoy_init`` (§27.4)."""
    c = apply_overrides(micro3d_cfg, ["population.immigration_floor=16", "population.immigrants_per_step=8"])
    d = dims(c)
    s = make_scene(c, [{"pos": [10.0, 10.0, d.z_lo], "energy": 1}])  # cannot pay its first step's costs
    rd = RunDir.create(tmp_path / "run", c)
    res = runner.run(rd, c, s, 2, quiet=True)
    assert (res.status, res.exit_code) == ("done", ExitCode.OK)
    assert sum(r["immigrants"] for r in rd.metrics_rows()) == 16
    alive = to_numpy(res.state.alive)
    z = to_numpy(res.state.pos)[alive, 2]
    assert alive.sum() == 16 and ((z >= d.z_lo) & (z <= d.z_hi)).all() and z.std() > 2.0
    assert (to_numpy(res.state.buoy_cap)[alive] == np.float32(c.mutation.buoy_init)).all()


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


def test_pending_director_immigration_request_prevents_extinction(cfg: Config, tmp_path: Path) -> None:
    c = apply_overrides(cfg, ["director.enabled=true"])
    empty = make_scene(c, [])
    rd = RunDir.create(tmp_path / "none", c)
    assert runner.run(rd, c, empty, 5, quiet=True).status == "extinct"  # nothing pending: extinct at once
    pending = empty._replace(env_imm_request=mx.array(6, dtype=mx.int32))
    rd = RunDir.create(tmp_path / "pending", c)
    res = runner.run(rd, c, pending, 5, quiet=True)
    assert (res.status, res.exit_code) == ("done", ExitCode.OK)
    assert int(to_numpy(res.state.alive).sum()) > 0 and int(to_numpy(res.state.env_imm_request)) == 0
    assert rd.metrics_rows()[0]["immigrants"] == 6 and rd.metrics_rows()[0]["deaths_catastrophe"] == 0


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


# --- the director hook ---


class SpyDirector:
    """Records the runner's calls; at a boundary it sets the regeneration multiplier to 1.5."""

    def __init__(self, every: int, act: bool = True) -> None:
        self.every, self.act, self.calls, self.applied = every, act, [], []
        self.pending: list[tuple[int, int, bool]] = []  # (t, the state's step, stopping) of apply_pending

    def due(self, t: int) -> bool:
        return t % self.every == 0

    def apply_pending(self, state, t, *, stopping=False):  # the player channel (G3b): nothing queued here
        self.pending.append((t, int(to_numpy(state.step)), stopping))
        return state

    def observe(self, row) -> None:
        self.calls.append(("observe", row["step"]))

    def on_boundary(self, state, row, t):
        self.calls.append(("boundary", t))
        self.applied = [{"step_applied": t}]
        return state._replace(env_regen_mult=mx.array(1.5, dtype=mx.float32)) if self.act else state

    def status_lines(self) -> list[str]:
        return [f"director @ {e['step_applied']}: spy" for e in self.applied]

    def on_checkpoint(self, t: int) -> None:
        self.calls.append(("checkpoint", t))


@pytest.fixture(scope="module")
def dcfg(cfg: Config) -> Config:
    return apply_overrides(cfg, ["director.enabled=true"])


def test_the_player_channel_is_called_at_the_top_of_every_iteration_with_the_state_at_that_step(
    dcfg: Config, tmp_path: Path
) -> None:
    """``apply_pending(state, t)`` runs once per loop iteration, before the step (so ``state.step == t``)."""
    spy = SpyDirector(every=10, act=False)
    rd = RunDir.create(tmp_path / "run", dcfg)
    assert runner.run(rd, dcfg, init_state(dcfg), 7, director=spy).status == "done"
    assert spy.pending == [(t, t, False) for t in range(7)]


def test_director_hook_runs_after_the_row_and_before_the_checkpoint(
    dcfg: Config, tmp_path: Path, capsys
) -> None:
    spy = SpyDirector(every=10)
    rd = RunDir.create(tmp_path / "run", dcfg)
    order = []
    real = rd.save_checkpoint
    rd.save_checkpoint = lambda s, c: (order.append(("save", int(to_numpy(s.step)))), real(s, c))[1]  # type: ignore[method-assign]
    res = runner.run(rd, dcfg, init_state(dcfg), 20, director=spy)
    assert res.status == "done"
    assert spy.calls == [
        ("observe", 5),
        ("observe", 10),
        ("boundary", 10),
        ("checkpoint", 10),
        ("observe", 15),
        ("observe", 20),
        ("boundary", 20),
        ("checkpoint", 20),
    ]  # every row is observed; the hook is on rule boundaries only, before the checkpoint it feeds
    assert order == [("save", 0), ("save", 10), ("save", 20)]
    ck10 = load_checkpoint(rd.checkpoint_at(10), dcfg)
    assert float(to_numpy(ck10.env_regen_mult)) == 1.5  # the checkpoint already holds the applied env
    assert float(to_numpy(res.state.env_regen_mult)) == 1.5
    out = capsys.readouterr().out.splitlines()
    assert out[1].startswith("step 10") and out[2] == "director @ 10: spy"  # printed after the status line


def test_a_director_that_does_nothing_changes_nothing(dcfg: Config, tmp_path: Path) -> None:
    quiet = SpyDirector(every=5, act=False)
    a = RunDir.create(tmp_path / "a", dcfg)
    b = RunDir.create(tmp_path / "b", dcfg)
    ra = runner.run(a, dcfg, init_state(dcfg), 20, quiet=True, director=quiet)
    rb = runner.run(b, dcfg, init_state(dcfg), 20, quiet=True)
    assert state_digest(ra.state) == state_digest(rb.state)
    perf = {"wall_s", "steps_per_s", "peak_mem_gb"}
    strip = lambda rd: [{k: v for k, v in r.items() if k not in perf} for r in rd.metrics_rows()]  # noqa: E731
    assert strip(a) == strip(b)


def test_the_director_never_sees_or_acts_on_a_violated_state(dcfg: Config, tmp_path: Path) -> None:
    s = init_state(dcfg)._replace(diag=mx.array(int(Diag.BOND_ASYM), dtype=mx.int32))
    spy = SpyDirector(every=5)
    rd = RunDir.create(tmp_path / "run", dcfg)
    res = runner.run(rd, dcfg, s, 20, quiet=True, director=spy)
    assert res.status == "violation"
    assert spy.calls == []  # the row of the violated step is not observed and no boundary runs
    assert [st for st, _ in rd.checkpoints()] == [0]
    # an out-of-range environment is a violation too (FIELD_RANGE), so it never reaches the director either
    bad = init_state(dcfg)._replace(env_regen_mult=mx.array(9.0, dtype=mx.float32))
    spy2 = SpyDirector(every=5)
    res = runner.run(RunDir.create(tmp_path / "run2", dcfg), dcfg, bad, 20, quiet=True, director=spy2)
    assert res.status == "violation" and spy2.calls == []


def test_the_director_does_not_act_on_an_extinct_world(dcfg: Config, tmp_path: Path) -> None:
    s = make_scene(dcfg, [{"pos": [10.0, 10.0], "energy": 1}])  # dies in its first step
    spy = SpyDirector(every=1)
    res = runner.run(RunDir.create(tmp_path / "run", dcfg), dcfg, s, 10, quiet=True, director=spy)
    assert res.status == "extinct"
    assert spy.calls == [
        ("observe", 1),
        ("checkpoint", 1),
    ]  # observed, but nothing is decided for a dead world
