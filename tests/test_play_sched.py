"""The fixed-timestep scheduler (DESIGN §33.2.5), on a fake clock with no real waiting.

``Scheduler`` is driven here the way the pacer drives it: ``poll()`` says go or wait, a step costs the clock
whatever the test says, a wait advances it. Then ``PlayPacer`` on a real tiny world with the same fake clock:
the steps per second a rate gives, ``CLCK`` and its cadence, the frame period from the achieved rate, a pause
that still polls the stop check.
"""

from __future__ import annotations

import io
import math
import os
import signal
from itertools import pairwise
from pathlib import Path

import pytest

from alife import play, runner
from alife.play import BASE_RATE, WAIT_S, Scheduler
from alife.runio import RunDir
from alife.state import init_state
from play_helpers import CostedPacer, FakeClock, TimedControl, decoded, run_paced, snapshots, world

B = BASE_RATE
SLOT = 1 / B


def drive(
    sched: Scheduler, clock: FakeClock, n: int, cost: float | dict[int, float] = 0.0, *, limit: int = 100_000
) -> list[float]:
    """Issue ``n`` steps; ``cost`` is the time a step takes (per step index in a dict). Returns the issue
    times."""
    times: list[float] = []
    for _ in range(limit):
        if len(times) == n:
            return times
        w = sched.poll()
        if w is None:
            times.append(clock.now)
            clock.now += cost.get(len(times), 0.0) if isinstance(cost, dict) else cost
        else:
            assert 0 < w <= WAIT_S + 1e-12
            clock.now += w
    raise AssertionError("the scheduler did not issue the steps")


def spacings(times: list[float]) -> list[float]:
    return [b - a for a, b in pairwise(times)]


@pytest.mark.parametrize("rate", [0.25, 0.5, 1.0, 2.0])
def test_a_rate_issues_that_many_steps_per_second(rate: float) -> None:
    clock = FakeClock()
    s = Scheduler(clock, rate=rate, paused=False)
    times = drive(s, clock, 60)
    assert all(abs(d - 1 / (rate * B)) < 1e-6 for d in spacings(times))
    assert s.late_steps == 0 and s.save_stall_steps == 0
    # the first step is one slot after the anchor
    assert abs(times[0] - (1000.0 + 1 / (rate * B))) < 1e-6


def test_a_paused_scheduler_waits_and_issues_nothing() -> None:
    clock = FakeClock()
    s = Scheduler(clock, rate=1.0, paused=True)
    assert [s.poll() for _ in range(3)] == [WAIT_S] * 3
    assert s.requested() == 0.0


def test_pause_and_run_do_not_drift_or_burst() -> None:
    clock = FakeClock()
    s = Scheduler(clock, rate=1.0, paused=False)
    first = drive(s, clock, 10)
    s.pause()
    for _ in range(100):  # 25 s paused
        assert s.poll() == WAIT_S
        clock.now += WAIT_S
    s.run()
    after = drive(s, clock, 10)
    assert all(abs(d - SLOT) < 1e-6 for d in spacings(after))
    assert (
        after[0] - (first[-1] + 25.0) < 2 * SLOT
    )  # the first step after a run is one slot, not a burst of 360
    assert s.late_steps == 0


def test_a_rate_change_reanchors() -> None:
    clock = FakeClock()
    s = Scheduler(clock, rate=1.0, paused=False)
    drive(s, clock, 7)
    s.set_rate(2.0)
    nxt = drive(s, clock, 5)
    assert all(abs(d - SLOT / 2) < 1e-6 for d in spacings(nxt))
    s.set_rate(0.25)
    nxt = drive(s, clock, 5)
    assert all(abs(d - SLOT * 4) < 1e-6 for d in spacings(nxt))
    assert s.late_steps == 0


def test_rate_zero_pauses_and_a_positive_rate_runs() -> None:
    clock = FakeClock()
    s = Scheduler(clock, rate=1.0, paused=False)
    s.set_rate(0)
    assert s.paused and s.poll() == WAIT_S
    s.set_rate(2.0)
    assert not s.paused and s.rate == 2.0


def test_max_never_waits() -> None:
    clock = FakeClock()
    s = Scheduler(clock, rate=math.inf, paused=False)
    assert [s.poll() for _ in range(1000)] == [None] * 1000
    assert s.requested() == -1.0


def test_a_boundary_overrun_is_repaid_and_is_not_late() -> None:
    """One step takes 130 ms (the social settle and the runner's boundary) in a 69 ms slot: the next steps
    follow at once until the schedule is caught up, and nothing is counted late."""
    clock = FakeClock()
    s = Scheduler(clock, rate=1.0, paused=False)
    times = drive(s, clock, 40, {10: 0.130})
    gaps = spacings(times)
    assert s.late_steps == 0
    assert min(gaps[8:14]) < 0.2 * SLOT  # the debt is repaid by a step issued almost at once
    assert abs(sum(gaps[16:]) / len(gaps[16:]) - SLOT) < 1e-3  # and the pace is back to the slot


def test_a_debt_older_than_a_quarter_second_is_dropped_and_counted_late() -> None:
    clock = FakeClock()
    s = Scheduler(clock, rate=1.0, paused=False)
    times = drive(s, clock, 12, {5: 1.0})  # a one second stall: 14 slots owed
    owed = math.floor(1.0 * B + 1e-9)
    assert s.late_steps in (owed - 1, owed)  # all but the one step due now are dropped
    gaps = spacings(times)
    assert abs(gaps[4] - 1.0) < 1e-6  # the step after the stall is issued as it ends ...
    assert 0 < gaps[5] <= SLOT + 1e-9  # ... the next is due on the old grid, within a slot ...
    assert abs(gaps[6] - SLOT) < 1e-6  # ... and then the normal pace: it does not spiral


def test_a_slow_machine_settles_at_its_own_pace() -> None:
    """Steps that take 100 ms at a 69 ms slot: the loop runs as fast as it can and sheds the debt."""
    clock = FakeClock()
    s = Scheduler(clock, rate=1.0, paused=False)
    times = drive(s, clock, 200, 0.100)
    assert abs((times[-1] - times[0]) / 199 - 0.100) < 1e-6
    assert s.late_steps > 0


def test_a_checkpoint_stall_is_not_lateness() -> None:
    clock = FakeClock()
    s = Scheduler(clock, rate=1.0, paused=False)
    drive(s, clock, 10)
    s.begin_save()
    clock.now += 2.0  # 686 MB written
    s.end_save()
    assert s.late_steps == 0
    assert s.save_stall_steps == math.floor(2.0 * B)  # the steps the stall skipped
    nxt = drive(s, clock, 5)
    assert all(abs(d - SLOT) < 1e-6 for d in spacings(nxt))
    assert s.late_steps == 0


def test_a_save_while_paused_or_at_max_skips_nothing() -> None:
    clock = FakeClock()
    for s in (Scheduler(clock, paused=True), Scheduler(clock, rate=math.inf, paused=False)):
        s.begin_save()
        clock.now += 3.0
        s.end_save()
        assert s.save_stall_steps == 0


def test_step_batches_run_unthrottled_and_mark_their_last_step() -> None:
    clock = FakeClock()
    s = Scheduler(clock, rate=1.0, paused=False)
    s.step(3)
    assert s.paused and s.busy
    assert [s.poll() for _ in range(3)] == [None, None, None]
    assert s.batch_done and not s.busy
    assert s.poll() == WAIT_S  # paused again
    s.step(2)
    s.step(1)
    seen = []
    for _ in range(3):
        assert s.poll() is None
        seen.append(s.batch_done)
    assert seen == [False, True, True]  # a batch of 2 then a batch of 1: each ends with its own frame
    s.step(5)
    s.pause()  # cancels the batches
    assert not s.busy and s.poll() == WAIT_S
    s.step(5)
    s.run()
    assert not s.busy and not s.paused


# --- the pacer on a real world ---------------------------------------------------------------------------


def test_a_world_at_1x_takes_n_over_14_4_seconds(tmp_path: Path) -> None:
    clock = FakeClock()
    opt = play.PlayOptions(rate=1.0, paused=False, fresh=True, unattended=True)
    _, res, pacer, out = run_paced(world("2d"), tmp_path / "w", 30, [], clock=clock, options=opt)
    assert res.status == "done" and pacer.failure is None
    assert abs((clock.now - 1000.0) - 30 / B) < 0.3  # the first slot, 29 more, and the boundary waits
    steps = [r.step for r in decoded(out).records if r.tag == "FRMA"]
    assert steps == sorted(steps) and steps[0] == 0 and steps[-1] == 30


def test_the_pacer_starts_paused_without_a_rate(tmp_path: Path) -> None:
    """A world opened paused waits; the script's ``run`` starts it. Its first records are written at bind."""
    clock = FakeClock()
    opt = play.PlayOptions(rate=1.0, paused=True, fresh=True, unattended=True)
    _, res, _, out = run_paced(
        world("2d"), tmp_path / "w", 5, [[], [], [], ["run"]], clock=clock, options=opt
    )
    recs = decoded(out).records
    assert [r.tag for r in recs[:3]] == ["FRMA", "IDNT", "INTN"]
    clcks = [r for r in recs if r.tag == "CLCK"]
    assert clcks[0].f["reason"] == 1 and clcks[0].f["rate"] == 0.0  # paused by the user
    assert any(c.f["reason"] == 0 and c.f["rate"] == 1.0 for c in clcks)
    assert res.status == "done"


def test_clck_is_written_four_times_a_second_while_paused(tmp_path: Path) -> None:
    clock = FakeClock()
    ctl = TimedControl(clock, [(1002.0, "run"), (1002.0, "rate max")])
    cfg = world("2d")
    rd = RunDir.create(tmp_path / "w", cfg)
    out = io.BytesIO()
    pacer = play.PlayPacer(
        out,
        ctl,
        play.PlayOptions(rate=1.0, paused=True, fresh=True, unattended=True),
        clock=clock,
        note=lambda m: None,
    )
    runner.run(rd, cfg, init_state(cfg), 3, quiet=True, pacer=pacer)
    clcks = [r for r in decoded(out).records if r.tag == "CLCK"]
    paused = [c for c in clcks if c.f["reason"] == 1]
    assert 8 <= len(paused) <= 11  # about 2 s at 4 per second (and one per command)
    assert [c.seq for c in clcks] == list(range(len(clcks)))


def test_the_frame_period_follows_the_achieved_rate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """At max with steps that take 1/55 s the achieved rate is 55 steps/s: with a 30 Hz display a frame every
    second step, and the rate the machine cannot hold does not over-produce frames."""
    snapshots(monkeypatch, 10)
    clock = FakeClock()
    CostedPacer.cost = 1 / 55
    opt = play.PlayOptions(rate=math.inf, paused=False, fresh=True, unattended=True)
    cfg = world("2d", "run.metrics_every=50", "run.heavy_metrics_every=50", "run.checkpoint_every=100")
    _, _, pacer, out = run_paced(
        cfg, tmp_path / "w", 400, [], clock=clock, options=opt, pacer_class=CostedPacer
    )
    recs = decoded(out).records
    fes = [r.f["fe"] for r in recs if r.tag == "CLCK"]
    assert fes[0] == 1 and fes[-1] == 2
    late = [r.step for r in recs if r.tag == "FRMA" and r.step > 200]
    assert all(b - a == 2 for a, b in pairwise(late))
    assert pacer.sched.late_steps == 0
    achieved = [r.f["achieved"] for r in recs if r.tag == "CLCK"][-1]
    assert abs(achieved - 55) < 3


def test_a_finite_rate_caps_the_frame_period_at_the_commanded_rate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """At 2x (28.8 steps/s) with a 30 Hz display every step is a frame, and fe stays 1."""
    snapshots(monkeypatch, 10)
    clock = FakeClock()
    opt = play.PlayOptions(rate=2.0, paused=False, fresh=True, unattended=True)
    _, _, _, out = run_paced(world("2d"), tmp_path / "w", 120, [], clock=clock, options=opt)
    recs = decoded(out).records
    assert {r.f["fe"] for r in recs if r.tag == "CLCK"} == {1}
    steps = [r.step for r in recs if r.tag == "FRMA"]
    assert steps == list(range(0, 121))


def test_every_overrides_the_period_and_auto_restores_it(tmp_path: Path) -> None:
    clock = FakeClock()
    opt = play.PlayOptions(rate=math.inf, paused=False, fresh=True, unattended=True)
    script = [["every 5"], *[[]] * 40, ["every auto"]]
    _, _, _, out = run_paced(world("2d"), tmp_path / "w", 60, script, clock=clock, options=opt)
    recs = decoded(out).records
    steps = [r.step for r in recs if r.tag == "FRMA"]
    mid = [s for s in steps if 3 < s < 38]
    assert all(b - a == 5 for a, b in pairwise(mid)) and len(mid) >= 5
    assert 5 in {r.f["fe"] for r in recs if r.tag == "CLCK"}


def test_a_paused_wait_polls_the_stop_check(tmp_path: Path) -> None:
    """The paused wait looks at the runner's stop check at every tick, so a SIGINT is not dead while paused:
    the first one reaches the one-more-step path within one tick and the run checkpoints and ends
    ``interrupted``."""
    clock = FakeClock()

    class Interrupting(FakeClock):
        def wait(self, control, timeout):
            if self.waits == 5:
                os.kill(os.getpid(), signal.SIGINT)  # the runner's latch handler runs at once
            return super().wait(control, timeout)

    clock = Interrupting()
    cfg = world("2d")
    rd = RunDir.create(tmp_path / "w", cfg)
    pacer = play.PlayPacer(
        io.BytesIO(), TimedControl(clock, []), play.PlayOptions(paused=True), clock=clock, note=lambda m: None
    )
    res = runner.run(rd, cfg, init_state(cfg), 50, quiet=True, pacer=pacer)
    assert (res.status, res.exit_code) == ("interrupted", runner.ExitCode.INTERRUPTED)
    assert int(rd.metrics_rows()[-1]["step"]) == 1  # exactly one more step ran
    assert clock.waits == 6 and clock.now - 1000.0 < 2.0
    assert [s for s, _ in rd.checkpoints()][-1] == 1


def test_a_rate_the_machine_cannot_hold_does_not_over_produce_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """16x asks for 230 steps/s; steps that take 1/55 s give 55: the frame period follows what is achieved
    (every second step for a 30 Hz display), not what was asked, and the dropped debt is counted late."""
    snapshots(monkeypatch, 10)
    clock = FakeClock()
    CostedPacer.cost = 1 / 55
    opt = play.PlayOptions(rate=16.0, paused=False, fresh=True, unattended=True)
    cfg = world("2d", "run.metrics_every=50", "run.heavy_metrics_every=50", "run.checkpoint_every=100")
    _, _, pacer, out = run_paced(
        cfg, tmp_path / "w", 400, [], clock=clock, options=opt, pacer_class=CostedPacer
    )
    recs = decoded(out).records
    assert [r.f["fe"] for r in recs if r.tag == "CLCK"][-1] == 2
    late = [r.step for r in recs if r.tag == "FRMA" and r.step > 200]
    assert all(b - a == 2 for a, b in pairwise(late))
    assert pacer.sched.late_steps > 100  # the schedule cannot be kept and sheds what it cannot repay
