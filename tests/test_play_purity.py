"""Presentation never reaches the trajectory (DESIGN §33.2.4).

The same world is run with ``runner.run`` alone and with a ``PlayPacer`` that receives a seeded random control
script (pauses, rates, ``step`` batches, colours, ``every``, ``focus``, ``follow``, ``idnt``, ``labels`` and
some invalid lines) and builds every record. The state digests of every checkpoint, the metrics rows (every
key but the wall-clock ones), the event chunks, the social log and the director log must be equal, bit for
bit: observers on and off, 2D, 3D, the storm world, and a live rule-tier director run at the same cadence. A
quit at an arbitrary step followed by a resume equals the uninterrupted run on the state, the social log and
the sidecar. This is an empirical gate on the CPU in the suite and on Metal before merge (DESIGN §33.5)."""

from __future__ import annotations

from pathlib import Path

import pytest

from alife import runner
from alife.runio import RunDir, load_checkpoint
from alife.state import init_state, state_digest
from play_helpers import (
    checkpoint_digests,
    director_log,
    random_script,
    rows,
    run_paced,
    run_plain,
    same_events,
    same_sidecar,
    snapshots,
    social_text,
    uids,
    world,
)

STEPS = 60


def assert_same_run(a, b, steps: int, *, observed: bool) -> None:
    assert [st for st, _ in a.checkpoints()] == [st for st, _ in b.checkpoints()]
    assert checkpoint_digests(a) == checkpoint_digests(b)
    assert rows(a) == rows(b)
    assert same_events(a, b)
    assert social_text(a) == social_text(b)
    assert director_log(a) == director_log(b)
    if observed:
        for st, _ in a.observer_files():
            assert same_sidecar(a, b, st)


ARMS = [
    pytest.param("2d", True, False, 1, id="2d-observed"),
    pytest.param("2d", False, False, 2, id="2d-plain"),
    pytest.param("3d", True, False, 3, id="3d-observed"),
    pytest.param("3d", False, False, 4, id="3d-plain"),
    pytest.param("storm", True, False, 5, id="storm-observed"),
    pytest.param("storm", False, False, 6, id="storm-plain"),
    pytest.param("2d", True, True, 7, id="2d-director"),
    pytest.param("2d", False, True, 8, id="2d-director-plain"),
]


@pytest.mark.parametrize(("kind", "observe", "director", "seed"), ARMS)
def test_a_paced_run_equals_a_plain_run(
    kind: str, observe: bool, director: bool, seed: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshots(monkeypatch, 10)
    cfg = world(kind, observe=observe, director=director)
    plain, res_plain = run_plain(cfg, tmp_path / "plain", STEPS)
    script = random_script(seed, 70, uids(cfg)[:12])
    paced, res, pacer, out = run_paced(cfg, tmp_path / "paced", STEPS, script)
    assert (
        (res.status, res.exit_code) == (res_plain.status, res_plain.exit_code) == ("done", runner.ExitCode.OK)
    )
    assert pacer.failure is None
    assert state_digest(res.state) == state_digest(res_plain.state)
    assert_same_run(plain, paced, STEPS, observed=observe)
    assert len(out.getvalue()) > 0
    if director:
        assert len([e for e in director_log(plain) if e.get("kind") != "header"]) > 0  # the director acted


def test_the_play_files_are_additions_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The paced storm run has the plain run's files and, besides, the storm log, the tracker states and the
    kin thread tables."""
    snapshots(monkeypatch, 10)
    cfg = world("storm")
    plain, _ = run_plain(cfg, tmp_path / "plain", 40)
    paced, *_ = run_paced(cfg, tmp_path / "paced", 40, [["rate max", "run"]])

    def files(rd):
        return {str(p.relative_to(rd.path)) for p in rd.path.rglob("*") if p.is_file()}

    extra = files(paced) - files(plain)
    assert files(plain) <= files(paced)
    assert extra <= {
        "observers/storms.jsonl",
        *{f"observers/storms_{s:010d}.json" for s in (0, 30, 40)},
        *{f"observers/kinthreads_{s:010d}.json" for s in (30, 40)},  # G3b: the kin thread table
        "alerts.jsonl",  # G3b: the alerts the world raised (a world that raised none has no file)
    }
    assert {f for f in extra if f.endswith(".json")} >= {
        f"observers/storms_{s:010d}.json" for s in (0, 30, 40)
    }


def test_the_checkpoint_cadence_is_the_runs_not_the_pacers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A short ``run.checkpoint_every`` override decides when checkpoints are written, with or without a
    pacer."""
    snapshots(monkeypatch, 10)
    cfg = world("2d")
    plain, _ = run_plain(cfg, tmp_path / "plain", 70)
    paced, *_ = run_paced(cfg, tmp_path / "paced", 70, [["rate max", "run"]])
    assert [s for s, _ in plain.checkpoints()] == [s for s, _ in paced.checkpoints()]
    chunks = [(s0, s1) for s0, s1, _ in paced.event_files("births")]
    assert chunks == [(0, 30), (30, 60), (60, 70)] == [(s0, s1) for s0, s1, _ in plain.event_files("births")]


@pytest.mark.parametrize(("kind", "observe"), [("2d", True), ("3d", True), ("storm", True), ("2d", False)])
def test_a_quit_and_a_resume_equal_the_uninterrupted_run(
    kind: str, observe: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Quit at an arbitrary step (the step after the line is read), resume from the checkpoint it wrote, and
    finish: the state, the social log and the sidecar equal those of the run that never stopped."""
    snapshots(monkeypatch, 10)
    cfg = world(kind, observe=observe)
    whole, res_whole = run_plain(cfg, tmp_path / "whole", STEPS)
    # a quit after about 23 steps: the script is polled once per step while running
    rd, res, _, _ = run_paced(cfg, tmp_path / "stopped", STEPS, [["rate max", "run"], *([[]] * 22), ["quit"]])
    assert (res.status, res.exit_code) == ("interrupted", runner.ExitCode.INTERRUPTED)
    stop = int(rd.metrics_rows()[-1]["step"])
    assert stop == 24  # the quit line is read before step 24; exactly one more step runs
    assert [s for s, _ in rd.checkpoints()][-1] == stop
    rd.truncate_after(stop)
    state = load_checkpoint(rd.latest_checkpoint(), cfg)
    del res
    final = runner.run(rd, cfg, state, STEPS - stop, quiet=True)
    assert state_digest(final.state) == state_digest(res_whole.state)
    assert social_text(rd) == social_text(whole)
    if observe:
        assert same_sidecar(rd, whole, STEPS)
    assert checkpoint_digests(rd)[STEPS] == checkpoint_digests(whole)[STEPS]


def test_a_pacer_that_does_nothing_changes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The hook with every callback a no-op is the plain loop (the pacer=None path is the old code)."""
    snapshots(monkeypatch, 10)
    cfg = world("2d")

    class Null:
        stop_requested = False

        def bind(self, state, **kw): ...
        def before_step(self, state, t): ...
        def after_step(self, state, out, t, social): ...
        def before_checkpoint(self, state, t): ...
        def after_checkpoint(self, t): ...
        def after_boundary(self, state, row, t): ...
        def take_paused_s(self):
            return 0.0

    plain, res_plain = run_plain(cfg, tmp_path / "plain", 40)
    rd = RunDir.create(tmp_path / "null", cfg)
    res = runner.run(rd, cfg, init_state(cfg), 40, quiet=True, pacer=Null())
    assert state_digest(res.state) == state_digest(res_plain.state)
    assert_same_run(plain, rd, 40, observed=True)
