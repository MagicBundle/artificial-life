"""The runner's pacer hook (DESIGN §33.2.4): where the callbacks sit in the loop, that a stop takes the
ordinary unscheduled-stop path, and that no callback can leave the run without its checkpoint.

``runner.run(pacer=None)`` is the old loop (every existing runner test runs it); the equality of a run with
and without a pacer is ``test_play_purity``.
"""

from __future__ import annotations

import gc
import io
import json
import os
import signal
import threading
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import pytest

from alife import play, runio, runner, storms
from alife.runio import RunDir, load_checkpoint
from alife.state import init_state, state_digest
from alife.step import StepOut
from play_helpers import (
    FailingSink,
    FakeClock,
    ScriptControl,
    TimedControl,
    decoded,
    run_paced,
    run_plain,
    same_events,
    snapshots,
    social_text,
    world,
)
from test_play_storms import scene

RUN = play.PlayOptions(rate=float("inf"), paused=False, fresh=True, unattended=True)


def test_a_quit_while_paused_runs_exactly_one_more_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The unscheduled-stop path is reached after a step, as it is for SIGINT: a short interval row, the
    events flushed, a checkpoint at that step, status ``interrupted``; and a resume continues bitwise."""
    snapshots(monkeypatch, 10)
    cfg = world("2d")
    whole, res_whole = run_plain(cfg, tmp_path / "whole", 20)
    rd, res, pacer, out = run_paced(
        cfg, tmp_path / "w", 20, [[], ["quit"]], options=play.PlayOptions(paused=True, fresh=True)
    )
    assert (res.status, res.exit_code) == ("interrupted", runner.ExitCode.INTERRUPTED)
    assert pacer.failure is None
    assert rd.meta()["status"] == "interrupted"
    assert [r["step"] for r in rd.metrics_rows()] == [1]  # the short row of the one step
    assert [(s0, s1) for s0, s1, _ in rd.event_files("births")] == [(0, 1)]  # the events were flushed
    assert [s for s, _ in rd.checkpoints()] == [0, 1]
    assert [r.step for r in decoded(out).records if r.tag == "FRMA"] == [
        0
    ]  # paused: the extra step is silent
    rd.truncate_after(1)
    final = runner.run(rd, cfg, load_checkpoint(rd.latest_checkpoint(), cfg), 19, quiet=True)
    assert state_digest(final.state) == state_digest(res_whole.state)
    assert social_text(rd) == social_text(whole)


def test_a_quit_while_running_stops_after_the_next_step(tmp_path: Path) -> None:
    rd, res, _, _ = run_paced(world("2d"), tmp_path / "w", 50, [["rate max", "run"], [], [], ["quit"]])
    assert res.status == "interrupted"
    stop = int(rd.metrics_rows()[-1]["step"])
    assert [s for s, _ in rd.checkpoints()][-1] == stop and 3 <= stop <= 6


class TimingPacer(play.PlayPacer):
    """Records when ``before_step`` was entered and when it returned (the SIGINT test)."""

    exits: list[float]
    entered: threading.Event

    def before_step(self, state: Any, t: int) -> None:
        self.entered.set()
        super().before_step(state, t)
        self.exits.append(time.perf_counter())


def test_sigint_while_paused_reaches_the_stop_path_within_a_tick(tmp_path: Path) -> None:
    """A real SIGINT while the world is paused: the paused wait polls the runner's latch at every 0.25 s tick,
    so the first Ctrl-C ends the wait (the stop ladder of the window has a working SIGINT rung).

    The signal is sent once the pacer is in its first ``before_step`` (not after a guessed delay, which a slow
    start-up would overrun), and the bound is in ticks of the pacer's wait, not in seconds of the machine."""
    from alife.stream import QueueControl

    cfg = world("2d")
    rd = RunDir.create(tmp_path / "w", cfg)
    pacer = TimingPacer(
        io.BytesIO(), QueueControl(), play.PlayOptions(paused=True, fresh=True), note=lambda m: None
    )
    pacer.exits = []
    pacer.entered = threading.Event()
    sent: list[float] = []

    def interrupt() -> None:
        assert pacer.entered.wait(300), "the play loop never began"
        time.sleep(play.WAIT_S)  # a tick into the paused wait (or past it: the latch is read at once then)
        sent.append(time.perf_counter())
        os.kill(os.getpid(), signal.SIGINT)

    threading.Thread(target=interrupt, daemon=True).start()
    res = runner.run(rd, cfg, init_state(cfg), 50, quiet=True, pacer=pacer)
    assert (res.status, res.exit_code) == ("interrupted", runner.ExitCode.INTERRUPTED)
    assert pacer.exits[0] - sent[0] < 4 * play.WAIT_S  # one tick, with three more for a loaded machine
    assert [s for s, _ in rd.checkpoints()][-1] == 1 and rd.meta()["status"] == "interrupted"


@pytest.mark.parametrize("after", [3, 9, 25])
def test_a_reader_closed_mid_run_ends_with_a_checkpoint(after: int, tmp_path: Path) -> None:
    """A BrokenPipe from the stream: the play stops quietly after the checkpoint, the event chunk and the
    status update, whatever write it was."""
    rd, res, pacer, _ = run_paced(world("2d"), tmp_path / "w", 40, [], out=FailingSink(after), options=RUN)
    assert isinstance(pacer.failure, BrokenPipeError) and pacer.stop_requested
    assert (res.status, res.exit_code) == ("interrupted", runner.ExitCode.INTERRUPTED)
    stop = int(rd.metrics_rows()[-1]["step"])
    assert stop < 40 and [s for s, _ in rd.checkpoints()][-1] == stop
    assert rd.event_files("births") and rd.event_files("births")[-1][1] == stop
    assert rd.meta()["status"] == "interrupted"


def test_a_raising_callback_ends_the_same_way_with_the_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls = {"n": 0}
    original = play.PlayPacer._send_frame

    def boom(self: play.PlayPacer, state: Any) -> None:
        calls["n"] += 1
        if calls["n"] == 6:
            raise RuntimeError("the presenter broke")
        original(self, state)

    monkeypatch.setattr(play.PlayPacer, "_send_frame", boom)
    rd, res, pacer, _ = run_paced(world("2d"), tmp_path / "w", 40, [], options=RUN)
    assert isinstance(pacer.failure, RuntimeError)
    assert res.status == "interrupted"
    stop = int(rd.metrics_rows()[-1]["step"])
    assert [s for s, _ in rd.checkpoints()][-1] == stop and stop < 40
    assert rd.event_files("births")[-1][1] == stop and rd.meta()["status"] == "interrupted"
    err = capsys.readouterr().err
    assert "the presenter broke" in err and "Traceback" in err


def test_a_failed_pacer_writes_nothing_more_and_does_not_wait(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After a failure nothing more is written and the callbacks that wait or present return at once."""
    monkeypatch.setattr(
        play.PlayPacer, "_send_frame", lambda self, state: (_ for _ in ()).throw(BrokenPipeError("gone"))
    )
    _, _, pacer, out = run_paced(world("2d"), tmp_path / "w", 20, [], options=RUN)
    assert isinstance(pacer.failure, BrokenPipeError)
    size = len(out.getvalue())
    pacer.before_step(None, 99)  # would raise on the missing state if it did anything
    pacer.after_boundary(None, {}, 99)
    assert len(out.getvalue()) == size and pacer.take_paused_s() == 0.0


def _serve(cfg: Any, path: Path, steps: int, out: Any) -> tuple[RunDir, int]:
    rd = RunDir.create(path, cfg)
    code = play.serve_play(
        rd,
        cfg,
        [init_state(cfg)],
        steps=steps,
        options=RUN,
        control=ScriptControl([]),
        out=out,
        clock=FakeClock(),
    )
    return rd, code


def test_an_oserror_that_is_not_a_closed_reader_is_a_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A full disk (ENOSPC from the storm log) is not "the reader closed": traceback, exit code 1, and the run
    still ends with its checkpoint; a closed reader is a normal stop (exit 0)."""
    import errno

    def full(self: Any, rd: Any, tick: Any) -> None:
        raise OSError(errno.ENOSPC, "No space left on device")

    cfg = world("storm")
    rd, code = _serve(cfg, tmp_path / "ok", 60, FailingSink(30))
    assert code == 0 and [s for s, _ in rd.checkpoints()][-1] < 60
    monkeypatch.setattr(storms.StormLayer, "log", full)
    rd, code = _serve(cfg, tmp_path / "full", 60, io.BytesIO())
    assert code == 1
    err = capsys.readouterr().err
    assert "No space left on device" in err and "Traceback" in err
    stop = int(rd.metrics_rows()[-1]["step"])
    assert stop < 60 and [s for s, _ in rd.checkpoints()][-1] == stop and rd.meta()["status"] == "interrupted"


def _storm_scene(monkeypatch: pytest.MonkeyPatch, ticks: int) -> None:
    """A storm world whose detections are the synthetic scene (the real weather of a 256-row world has no
    storm in 200 steps), so that the tracker has ids, names and events to carry across a stop."""
    dets = scene(3, ticks)
    monkeypatch.setattr(storms.StormLayer, "tick_at", lambda self, ws, t: self.ingest(dets[t // 25 - 1]))


def _strm_by_step(out: io.BytesIO) -> dict[int, str]:
    return {r.step: json.dumps(r.f["json"], sort_keys=True) for r in decoded(out).records if r.tag == "STRM"}


@pytest.mark.parametrize("after", [40, 60, 90])
def test_a_reader_closed_in_a_storm_world_resumes_the_same_play(
    after: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stop path writes the tracker state of the checkpoint and processes the tick of the last step even
    when the reader has gone, so the resumed play is the uninterrupted one: the same ``storms.jsonl`` and the
    same ``STRM`` objects, ids and names included, and no "numbering restarts" (DESIGN §33.3.2)."""
    snapshots(monkeypatch, 10)
    _storm_scene(monkeypatch, 8)
    cfg = world("storm")
    whole, _, pw, out_w = run_paced(cfg, tmp_path / "whole", 200, [], options=RUN)
    rd, _, pacer, _ = run_paced(cfg, tmp_path / "w", 200, [], out=FailingSink(after), options=RUN)
    assert isinstance(pacer.failure, BrokenPipeError)
    stop = [s for s, _ in rd.checkpoints()][-1]
    assert 0 < stop < 200 and rd.storms_state_path(stop).is_file(), "the tracker state is with the checkpoint"
    last_tick = stop // 25 * 25
    assert [r["step"] for r in rd.storms_lines()] == [
        r["step"] for r in whole.storms_lines() if r["step"] <= last_tick
    ]
    rd.truncate_after(stop)
    notes: list[str] = []
    out = io.BytesIO()
    options = play.PlayOptions(rate=float("inf"), paused=False, dc_ref=pw.layer.dc_ref, unattended=True)
    pacer2 = play.PlayPacer(out, ScriptControl([]), options, clock=FakeClock(), note=notes.append)
    runner.run(rd, cfg, load_checkpoint(rd.latest_checkpoint(), cfg), 200 - stop, quiet=True, pacer=pacer2)
    assert not any("restarts" in n for n in notes) and pacer2.failure is None
    assert rd.storms_lines() == whole.storms_lines()
    ref, got = _strm_by_step(out_w), _strm_by_step(out)
    first = next(r for r in decoded(out).records if r.tag == "STRM")
    assert first.f["json"]["detect_step"] == last_tick, (
        "the first STRM names the last tick, not the checkpoint"
    )
    later = [t for t in got if t > stop and t % 25 == 0]
    assert later and all(got[t] == ref[t] for t in later)


class SlowStartPacer(play.PlayPacer):
    """A pacer whose start-up takes one fake second and whose first step (the compile) takes three."""

    def bind(self, *args: Any, **kwargs: Any) -> None:
        self.clock.now += 1.0  # type: ignore[attr-defined]
        super().bind(*args, **kwargs)

    def after_step(self, state: Any, out: Any, t: int, social: list[dict[str, Any]]) -> None:
        if t == 1:
            self.clock.now += 3.0  # type: ignore[attr-defined]
        super().after_step(state, out, t, social)


def test_the_start_up_and_the_first_compile_are_not_lateness(tmp_path: Path) -> None:
    """The scheduler starts at the first step, not at the pacer's construction: at 1x a cold start is not
    "late N" for the rest of the session (the gate of §5.3 is ``late_steps == 0``)."""
    rate = play.PlayOptions(rate=1.0, paused=False, fresh=True)
    _, _, pacer, _ = run_paced(world("2d"), tmp_path / "w", 8, [], options=rate, pacer_class=SlowStartPacer)
    assert pacer.sched.late_steps == 0 and pacer._steps_taken == 8


class OrderPacer(play.PlayPacer):
    log: list[tuple[str, int]]

    def before_checkpoint(self, state: Any, t: int) -> None:
        super().before_checkpoint(state, t)
        self.log.append(("before", t))

    def after_checkpoint(self, t: int) -> None:
        super().after_checkpoint(t)
        self.log.append(("after", t))


def test_before_checkpoint_precedes_the_state_file_and_the_tracker_file_is_there(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshots(monkeypatch, 10)
    log: list[tuple[str, int]] = []
    seen: list[tuple[int, bool]] = []
    real = runio.write_state_file

    def spy(path: Path, s: Any, cfg: Any) -> None:
        step = int(np.array(s.step))
        log.append(("state", step))
        seen.append((step, (path.parent.parent / "observers" / f"storms_{step:010d}.json").exists()))
        real(path, s, cfg)

    monkeypatch.setattr(runio, "write_state_file", spy)
    cfg = world("storm")
    rd = RunDir.create(tmp_path / "w", cfg)
    pacer = OrderPacer(io.BytesIO(), ScriptControl([]), RUN, clock=FakeClock(), note=lambda m: None)
    pacer.log = log
    runner.run(rd, cfg, init_state(cfg), 40, quiet=True, pacer=pacer)
    saves = [e for e in log if e[0] in ("before", "state", "after") and e[1] in (30, 40)]
    assert saves == [
        ("before", 30),
        ("state", 30),
        ("after", 30),
        ("before", 40),
        ("state", 40),
        ("after", 40),
    ]
    assert dict(seen)[30] and dict(seen)[40]  # the tracker's file existed before the state file was written


class RecordingPacer(play.PlayPacer):
    def __init__(self, *a: Any, **k: Any) -> None:
        super().__init__(*a, **k)
        self.social: list[dict[str, Any]] = []
        self.outs: list[Any] = []
        self.steps: list[int] = []

    def after_step(self, state: Any, out: Any, t: int, social: list[dict[str, Any]]) -> None:
        self.social += social
        self.outs.append(type(out))
        self.steps.append(t)
        super().after_step(state, out, t, social)


def test_after_step_sees_the_records_the_log_got(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    snapshots(monkeypatch, 10)
    cfg = world("2d")
    rd, _, pacer, _ = run_paced(cfg, tmp_path / "w", 40, [], options=RUN, pacer_class=RecordingPacer)
    assert [r["step"] for r in pacer.social] == [10, 20, 30, 40]
    assert pacer.social == rd.social_records()
    assert set(pacer.outs) == {StepOut} and pacer.steps == list(range(1, 41))


def test_the_pacer_holds_no_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """State is borrowed: nothing the pacer keeps is, or contains, a device array (the donation rule)."""
    snapshots(monkeypatch, 10)
    _, _, pacer, _ = run_paced(world("storm"), tmp_path / "w", 30, [["focus 3"]], options=RUN)

    def holds_array(x: Any, depth: int = 0) -> bool:
        if isinstance(x, mx.array):
            return True
        if depth > 3:
            return False
        if isinstance(x, dict):
            return any(holds_array(v, depth + 1) for v in x.values())
        if isinstance(x, list | tuple | set | frozenset | __import__("collections").deque):
            return any(holds_array(v, depth + 1) for v in x)
        if hasattr(x, "__dict__") and type(x).__module__.startswith("alife"):
            return any(holds_array(v, depth + 1) for v in vars(x).values())
        return False

    held = {k: type(v).__name__ for k, v in vars(pacer).items() if holds_array(v)}
    # the observer and the runner's RunDir/Config are the run's own; a State or a raw array would be a leak
    assert set(held) <= {"obs", "director"}, held


class Hoarder(play.PlayPacer):
    """Negative control: keeps the state it is lent."""

    def after_step(self, state: Any, out: Any, t: int, social: list[dict[str, Any]]) -> None:
        self.keep = state
        super().after_step(state, out, t, social)


def _peak(cls: type[play.PlayPacer], tmp_path: Path, name: str) -> tuple[int, int]:
    cfg = world("3d", observe=False)
    state = init_state(cfg)
    weights = state.W1.nbytes + state.W2.nbytes
    del state
    gc.collect()
    mx.reset_peak_memory()
    run_paced(cfg, tmp_path / name, 25, [], options=RUN, pacer_class=cls)
    return mx.get_peak_memory(), weights


def test_a_pacer_run_stays_within_one_states_weights_of_a_plain_run(tmp_path: Path) -> None:
    """The donation health check: the peak memory of a pacer-driven run is that of a plain one (the runner is
    the only holder of the state when it steps); a pacer that keeps the lent state adds a copy of the weights.
    """
    cfg = world("3d", observe=False)
    gc.collect()
    mx.reset_peak_memory()
    run_plain(cfg, tmp_path / "plain", 25)
    plain = mx.get_peak_memory()
    paced, weights = _peak(play.PlayPacer, tmp_path, "paced")
    hoard, _ = _peak(Hoarder, tmp_path, "hoard")
    assert paced - plain < weights / 2
    assert hoard - plain > weights / 2  # the check is not vacuous


def test_a_pause_is_not_run_time(tmp_path: Path) -> None:
    """``take_paused_s`` reports the seconds waited while paused, once; a throttle sleep is run time."""
    paused: list[float] = []

    class Counting(play.PlayPacer):
        def take_paused_s(self) -> float:
            p = super().take_paused_s()
            paused.append(p)
            return p

    clock = FakeClock()
    ctl = TimedControl(clock, [(1000.5, "pause"), (1003.5, "run")])
    cfg = world("2d")
    rd = RunDir.create(tmp_path / "w", cfg)
    pacer = Counting(
        io.BytesIO(),
        ctl,
        play.PlayOptions(rate=1.0, paused=False, fresh=True),
        clock=clock,
        note=lambda m: None,
    )
    runner.run(rd, cfg, init_state(cfg), 12, quiet=True, pacer=pacer)
    assert 2.7 < sum(paused) < 3.3 and paused[0] > 2.7 and all(p == 0.0 for p in paused[1:])
    assert all(r["wall_s"] > 0 and r["steps_per_s"] > 0 for r in rd.metrics_rows())


def test_an_extinct_world_returns_before_the_pacer_is_bound(tmp_path: Path) -> None:
    cfg = world("2d", "population.immigration_floor=0", "population.n_init=8")
    state = init_state(cfg)
    state = state._replace(
        alive=mx.zeros_like(state.alive), uid=mx.full(state.uid.shape, -1, dtype=state.uid.dtype)
    )
    rd = RunDir.create(tmp_path / "w", cfg)
    out = io.BytesIO()
    pacer = play.PlayPacer(out, ScriptControl([]), RUN, clock=FakeClock(), note=lambda m: None)
    res = runner.run(rd, cfg, state, 10, quiet=True, pacer=pacer)
    assert res.status == "extinct" and out.getvalue() == b""


def test_a_pacer_run_has_the_files_of_a_plain_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    snapshots(monkeypatch, 10)
    cfg = world("2d")
    plain, _ = run_plain(cfg, tmp_path / "plain", 30)
    paced, *_ = run_paced(cfg, tmp_path / "paced", 30, [["rate max", "run"], ["pause"], [], ["run"]])
    assert same_events(plain, paced)
    names = sorted(p.name for p in plain.path.rglob("*") if p.is_file())
    played = sorted(p.name for p in paced.path.rglob("*") if p.is_file())
    # the one file a played world has besides: the kin thread table, with each checkpoint the loop saves (not
    # the initial one at step 0, whose table is the refold of an empty log)
    kin = [n for n in played if n.startswith("kinthreads_")]
    assert kin == [f"kinthreads_{st:010d}.json" for st, _ in paced.checkpoints() if st > 0]
    extra = [*kin, "alerts.jsonl"]  # and the alerts the world raised (a world that raised none has no file)
    assert sorted(n for n in played if n not in extra) == names


def test_a_checkpoint_stall_is_announced_and_counted_apart_from_lateness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A slow save (here 2 fake seconds) is sent as ``saving`` before it starts and ends, the steps it skipped
    are ``save_stall_steps`` and never ``late_steps``, and the world goes on at its own pace afterwards."""
    clock = FakeClock()
    real = runio.write_state_file

    def slow(path: Path, s: Any, cfg: Any) -> None:
        if int(np.array(s.step)) == 30:
            clock.now += 2.0
        real(path, s, cfg)

    monkeypatch.setattr(runio, "write_state_file", slow)
    opt = play.PlayOptions(rate=1.0, paused=False, fresh=True)
    _, _, pacer, out = run_paced(world("2d"), tmp_path / "w", 40, [], clock=clock, options=opt)
    clcks = [(r.step, r.f) for r in decoded(out).records if r.tag == "CLCK"]
    assert [s for s, f in clcks if f["saving"]] == [30, 40]  # each save is announced, at its own step
    before = next(f for s, f in clcks if s == 30 and f["saving"])
    assert before["save_stall_steps"] == 0  # announced before the stall begins
    after = next(f for s, f in clcks if s == 30 and not f["saving"] and f["save_stall_steps"])
    assert (
        after["save_stall_steps"] == pytest.approx(2.0 * play.BASE_RATE, abs=2) and after["late_steps"] == 0
    )
    assert pacer.sched.late_steps == 0 and pacer.sched.save_stall_steps == after["save_stall_steps"]
    assert clcks[-1][1]["saving"] == 0


def test_a_shaded_world_sends_the_shaded_sea_plane(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With ``weather.shade_gain > 0`` the ``MAP1`` plane ``T_s`` is the engine's forcing of the next step,
    ``warm_pools * shade_mult`` of the state at that step (DESIGN §33.1.7); at step 0 that is the initial
    population. With the gain 0 it is the bare pool map, and the plane differs between the two worlds."""
    from alife import weather as W

    snapshots(monkeypatch, 10)
    got = {}
    for gain in ("0.05", "0"):
        cfg = world("storm", f"weather.shade_gain={gain}", "weather.shade_n0=0.01")
        _, _, _, out = run_paced(cfg, tmp_path / gain, 2, [[]], options=RUN)
        got[gain] = next(r for r in decoded(out).records if r.tag == "MAP1" and r.step == 0).a["Ts"]
        s0 = init_state(cfg)
        warm = W.warm_pools(W.params_from_config(cfg), mx.array(0, dtype=mx.int32))
        t_s = warm * W.shade_mult(s0.pos, s0.alive, cfg) if float(gain) > 0 else warm
        want = np.rint(255 * np.clip(np.array(t_s), 0, 1)).astype(np.uint8)
        assert np.array_equal(got[gain], want), gain
    assert not np.array_equal(got["0.05"], got["0"])
