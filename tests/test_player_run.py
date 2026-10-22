"""The player channel in the run loop (G3b spec §6.1; DESIGN §26.8, §34.5): ``Director.apply_pending`` at the
top of the iteration, rule P, the player mode, replay equality, the stop check and the refusals.

A scripted pacer queues the player's requests at chosen steps on a tiny world; the game table is shrunk by an
autouse patch so that durations, cooldowns and the influence window fit a 120-step run (the real bounds are
tested in ``test_player_channel.py``). The digest matrix of the tree against its parent commit, the three
research logs and the plain-run equalities are what make the change safe for every run without a player.
"""

from __future__ import annotations

import dataclasses
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import mlx.core as mx
import pytest

from alife import cli, play, runner
from alife.config import Config, ConfigError, apply_overrides, load_config
from alife.director import Director, PlayerRequest, actions, log, make_director, pacing, rules
from alife.director.actions import GAME_SPECS, ActionError, env_view
from alife.runio import COMMANDS_FILE, DIRECTOR_LOG_FILE, JOURNAL_FILE, RunDir, load_checkpoint
from alife.state import State, init_state, state_digest
from alife.step import make_step
from helpers import scripted_decide
from play_helpers import CADENCE, checkpoint_digests, events, rows, snapshots

PLAY_CADENCE = {
    "metrics_every": 10,
    "heavy_metrics_every": 20,
    "checkpoint_every": 30,
    "keep_checkpoints": 50,
}
END = 120


def world(mode: str = "player", *extra: str) -> Config:
    """``tiny``, 24 organisms, a rule boundary every 20 steps, a checkpoint every 30."""
    return load_config(
        preset="tiny",
        overrides=[
            *[c for c in CADENCE if not c.startswith("run.heavy_metrics_every")],
            "run.heavy_metrics_every=20",
            "run.keep_checkpoints=50",
            "game.observe=false",
            "director.enabled=true",
            "director.rule_every=20",
            "population.n_init=24",
            f"run.director={mode}",
            *extra,
        ],
    )


def shrink(mp: pytest.MonkeyPatch) -> None:
    """Durations of 30 to 120 steps in multiples of 10, cooldowns of 20, an influence window of 400."""
    for name, spec in list(GAME_SPECS.items()):
        small: dict[str, Any] = {"cooldown": 20}
        if spec.duration is not None:
            small |= {"duration": (30, 120), "quantum": 10}
        mp.setitem(GAME_SPECS, name, dataclasses.replace(spec, **small))
    mp.setattr(pacing, "PLAYER_BUDGET_WINDOW", 400)


@pytest.fixture(autouse=True)
def small_powers(monkeypatch: pytest.MonkeyPatch) -> None:
    shrink(monkeypatch)


def food(command: int, mult: float = 1.3, duration: int = 40) -> PlayerRequest:
    return PlayerRequest(command, "food", {"mult": mult}, duration)


def bloom(command: int, patch: int = 1, gain: float = 1.8, duration: int = 60) -> PlayerRequest:
    return PlayerRequest(command, "bloom", {"patch": patch, "gain": gain}, duration)


def blight(command: int, patch: int = 2, duration: int = 30) -> PlayerRequest:
    return PlayerRequest(command, "blight", {"patch": patch}, duration)


def strike(command: int, x: float = 60.0, y: float = 60.0) -> PlayerRequest:
    return PlayerRequest(command, "strike", {"x": x, "y": y, "radius": 30.0, "kill_p": 0.5})


def cancel(command: int, power: str, **raw: Any) -> PlayerRequest:
    return PlayerRequest(command, power, raw, None, cancel=True)


class Script:
    """A pacer that queues requests at chosen steps (at the top of the iteration, in ``before_step``, as the
    play pacer will) and does nothing else; it keeps what the director reports. ``stop_at`` asks for a stop
    at that step."""

    def __init__(self, plan: Mapping[int, list[PlayerRequest]], stop_at: int | None = None) -> None:
        self.plan, self.stop_at = plan, stop_at
        self.stop_requested = False
        self.director: Director | None = None
        self.applied: list[tuple[int, dict[str, Any]]] = []
        self.refused: list[tuple[int, PlayerRequest, str]] = []
        self.planned: dict[
            tuple[int, int], list[dict[str, Any]] | str
        ] = {}  # (step, command) -> entries | why not
        self.preview: Mapping[int, list[PlayerRequest]] = {}

    def bind(self, state: State, **kw: Any) -> None:
        self.director = kw["director"]

    def before_step(self, state: State, t: int) -> None:
        if self.director is None:  # a world without a director: nothing to queue to
            return
        for request in self.plan.get(t, []):
            self.director.queue_player(request)
        for request in self.preview.get(
            t, []
        ):  # a dry run (after the queueing, so it sees it): changes nothing
            try:
                self.planned[t, request.command] = self.director.plan_player(state, request, t)
            except ActionError as e:
                self.planned[t, request.command] = str(e)
        if t == self.stop_at:
            self.stop_requested = True

    def after_step(self, state: State, out: Any, t: int, social: list[dict[str, Any]]) -> None:
        if self.director is None:
            return
        applied, refused = self.director.take_player()
        self.applied += [(t - 1, e) for e in applied]
        self.refused += [(t - 1, q, why) for q, why in refused]

    def before_checkpoint(self, state: State, t: int) -> None:
        pass

    def after_checkpoint(self, t: int) -> None:
        pass

    def after_boundary(self, state: State, row: dict[str, Any], t: int) -> None:
        pass

    def take_paused_s(self) -> float:
        return 0.0


def run(
    cfg: Config,
    path: Path,
    steps: int,
    plan: Mapping[int, list[PlayerRequest]] | None = None,
    *,
    stop_at: int | None = None,
    start: tuple[RunDir, int] | None = None,
    preview: Mapping[int, list[PlayerRequest]] | None = None,
    pacer: bool = True,
) -> tuple[RunDir, Script, runner.RunResult]:
    """Run ``steps`` steps of a new world at ``path`` (or on from the checkpoint ``start = (rd, step)``)."""
    if start is None:
        rd, state, at = RunDir.create(path, cfg), init_state(cfg), 0
        rd.update_meta(start_step=0)
    else:
        rd, at = start
        rd.truncate_after(at)
        state = load_checkpoint(rd.checkpoint_at(at), cfg)
    director = make_director(rd, cfg, at, game_specs=True) if cfg.run.director != "off" else None
    script = Script(plan or {}, stop_at)
    script.preview = preview or {}
    res = runner.run(rd, cfg, state, steps, quiet=True, director=director, pacer=script if pacer else None)
    return rd, script, res


def log_of(rd: RunDir) -> list[dict[str, Any]]:
    return [r for r in rd.read_jsonl(DIRECTOR_LOG_FILE) if r.get("kind") != "header"]


def steps_of(rd: RunDir) -> list[tuple[int, str, str]]:
    return [(e["step_applied"], e["source"], e["action"]) for e in log_of(rd)]


def env_at(rd: RunDir, step: int) -> Any:
    return env_view(load_checkpoint(rd.checkpoint_at(step), rd.config()))


SESSION = {
    13: [food(1)],  # off the metrics grid; ends at 53, between two checkpoints
    27: [bloom(2)],  # ends at 87
    41: [strike(3)],
    60: [blight(4)],  # on a checkpoint step; ends at 90, another checkpoint step
}


@pytest.fixture(scope="module")
def session(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The scripted session of four powers, one of each, three timed with off-grid or on-checkpoint ends."""
    with pytest.MonkeyPatch.context() as mp:
        shrink(mp)
        rd, script, res = run(world(), tmp_path_factory.mktemp("session") / "live", END, SESSION)
    assert res.status == "done" and not script.refused
    return rd.path


# --- the player mode ---


def test_the_player_mode_with_no_power_is_the_run_without_a_director(tmp_path: Path) -> None:
    """Inert env (DESIGN §26 D1): the same states, rows and events as ``off``, with or without a pacer."""
    off, _, _ = run(world("off"), tmp_path / "off", 60)
    for name, with_pacer in (("player", True), ("player_nopacer", False)):
        rd, script, res = run(world(), tmp_path / name, 60, pacer=with_pacer)
        assert res.status == "done" and not script.applied
        assert checkpoint_digests(rd) == checkpoint_digests(off)
        assert rows(rd) == rows(off)
        mine, theirs = events(rd), events(off)
        assert all((mine[k][c] == theirs[k][c]).all() for k in mine for c in mine[k])
        assert log_of(rd) == []
        [header] = rd.read_jsonl(DIRECTOR_LOG_FILE)
        assert header["game_specs"] == "game-v0" and header["kind"] == "header"


def test_the_player_mode_decides_nothing_and_writes_no_archive(tmp_path: Path) -> None:
    rd, _, _ = run(world(), tmp_path / "p", 60)
    assert not (rd.path / "director_archive.json").exists() and not (rd.path / JOURNAL_FILE).exists()
    d = Director(rd, world(), "player", 60)
    assert not any(d.due(t) for t in (20, 40, 60)) and d.mode == "player"
    assert Director(rd, world("live"), "live", 60).due(60)


def test_a_player_mode_needs_the_director_enabled(tiny_cfg: Config) -> None:
    with pytest.raises(ConfigError, match=r"needs director\.enabled"):
        apply_overrides(tiny_cfg, ["run.director=player"])
    apply_overrides(tiny_cfg, ["run.director=player", "director.enabled=true"])


# --- the loop ---


def test_the_scripted_session_is_logged_at_the_steps_it_was_applied(session: Path) -> None:
    rd = RunDir.open(session)
    assert steps_of(rd) == [
        (13, "player", "regen_pulse"),
        (27, "player", "fertility_shift"),
        (41, "player", "catastrophe"),
        (53, "revert", "regen_pulse"),  # 13 + 40, off the grid of rows, checkpoints and boundaries
        (60, "player", "habitat_loss"),
        (87, "revert", "fertility_shift"),  # 27 + 60
        (90, "revert", "habitat_loss"),  # 60 + 30: a checkpoint step
    ]
    header, *_ = rd.read_jsonl(DIRECTOR_LOG_FILE)
    assert header["game_specs"] == "game-v0" and header["rules_version"] >= 1
    entries = log_of(rd)
    assert all(e["scope"] == "player" for e in entries)
    powers = [e for e in entries if e["source"] == "player"]
    assert [(e["command"], e["power"], e["cost"], e["specs"]) for e in powers] == [
        (1, "food", 1, "game-v0"),
        (2, "bloom", 1, "game-v0"),
        (3, "strike", 1, "game-v0"),
        (4, "blight", 2, "game-v0"),
    ]
    assert [e["duration"] for e in powers] == [40, 60, None, 30]
    assert [e["reverts"] for e in entries if e["source"] == "revert"] == [13, 27, 60]
    assert all(e["vitals"]["n_alive"] is not None and e["vitals_step"] % 10 == 0 for e in entries)
    assert (
        entries[0]["reason"] == "the player used food"
        and entries[3]["reason"] == "end of the effect from step 13"
    )
    log.validate_log(rd.read_jsonl(DIRECTOR_LOG_FILE), rd.config())
    start, end = (
        env_at(rd, 0),
        env_at(rd, END),
    )  # the effects are gone at the end: Food, Bloom and Blight reverted
    assert end.regen_mult == start.regen_mult == 1.0 and (end.patch_gain == start.patch_gain).all()
    assert env_at(rd, 30).regen_mult == actions._f32(1.3) and env_at(rd, 60).patch_gain[1] == actions._f32(
        1.8
    )


def test_a_power_acts_on_the_step_that_starts_at_its_step(session: Path) -> None:
    """The semantics, replayed by hand with the compiled step alone: a record at ``s`` is applied between step
    ``s - 1`` and step ``s``, so the state at 60 is the plain steps with each record applied before the step
    that starts at its ``step_applied`` (a revert at 53 is the same)."""
    rd = RunDir.open(session)
    cfg = rd.config()
    step_fn = make_step(cfg)
    by_step: dict[int, list[dict[str, Any]]] = {}
    for e in log_of(rd):
        by_step.setdefault(e["step_applied"], []).append(e)
    s = init_state(cfg)
    for t in range(60):
        for e in by_step.get(t, []):
            s = actions.apply(s, e, cfg)
        s, _ = step_fn(s)
        mx.eval(s)
    assert int(s.step) == 60 and state_digest(s) == checkpoint_digests(rd)[60]


def test_rule_p_the_checkpoint_at_a_power_s_step_does_not_hold_it(session: Path) -> None:
    """Blight at 60 is applied at the top of iteration 60, after the checkpoint at 60 was written; its revert
    at 90 likewise, so the checkpoint at 90 still holds the lever it was about to free."""
    rd = RunDir.open(session)
    assert env_at(rd, 60).patch_gain[2] == 1.0 and env_at(rd, 90).patch_gain[2] == 0.0
    assert env_at(rd, END).patch_gain[2] == 1.0


def replay_cfg(cfg: Config) -> Config:
    return apply_overrides(cfg, ["run.director=replay"])


def test_a_replay_of_the_session_reaches_the_same_states_at_every_checkpoint(
    session: Path, tmp_path: Path
) -> None:
    live = RunDir.open(session)
    cfg = replay_cfg(live.config())
    rd = RunDir.create(tmp_path / "replay", cfg)
    rd.update_meta(start_step=0)
    shutil.copyfile(session / DIRECTOR_LOG_FILE, rd.path / DIRECTOR_LOG_FILE)
    director = Director(rd, cfg, "replay", 0)
    assert runner.run(rd, cfg, init_state(cfg), END, quiet=True, director=director).status == "done"
    assert checkpoint_digests(rd) == checkpoint_digests(live) and sorted(checkpoint_digests(rd)) == [
        0,
        30,
        60,
        90,
        120,
    ]
    assert rows(rd) == rows(live)
    a, b = events(rd), events(live)
    assert all((a[k][c] == b[k][c]).all() for k in a for c in a[k])
    assert (rd.path / DIRECTOR_LOG_FILE).read_bytes() == (session / DIRECTOR_LOG_FILE).read_bytes()
    assert not director.pending_player and not director.pending  # every record was applied


def test_play_replay_and_resume_replay_reach_the_live_digest(
    session: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capfdbinary: pytest.CaptureFixture[bytes]
) -> None:
    """Both commands over a played session: three powers at off-grid steps (one timed with an off-grid end)
    and a fourth at a checkpoint step."""
    monkeypatch.setattr(play, "PLAY_CADENCE", PLAY_CADENCE)
    snapshots(monkeypatch, 10)
    a, b = tmp_path / "a", tmp_path / "b"
    shutil.copytree(session, a)
    shutil.copytree(session, b)
    log_bytes = (session / DIRECTOR_LOG_FILE).read_bytes()
    assert (
        cli.main(
            ["resume", str(a), "--checkpoint", "0", "--director", "replay", "--steps", str(END), "--quiet"]
        )
        == 0
    )
    flags = ["--adopt", "--checkpoint", "0", "--truncate", "--replay", "--no-control"]
    code = cli.main(["play", str(b), *flags, "--rate", "max", "--until", str(END)])
    assert code == 0, capfdbinary.readouterr().err.decode(errors="replace")
    assert (a / DIRECTOR_LOG_FILE).read_bytes() == log_bytes == (b / DIRECTOR_LOG_FILE).read_bytes()
    live = checkpoint_digests(RunDir.open(session))
    for d in (a, b):
        got = checkpoint_digests(RunDir.open(d))
        assert got == live, d


def test_a_power_at_a_checkpoint_step_then_a_crash_resume_gives_the_world_before_it(
    session: Path, tmp_path: Path
) -> None:
    """Resuming from the checkpoint at 60 cuts Blight (applied at 60, after it) from the log with its revert;
    the world, the fold and the log are what they were at 60, and playing on with the same input at 60
    reproduces the session byte for byte."""
    w = tmp_path / "w"
    shutil.copytree(session, w)
    rd = RunDir.open(w)
    original = (w / DIRECTOR_LOG_FILE).read_bytes()
    rd.truncate_after(60)
    assert steps_of(rd) == [
        (13, "player", "regen_pulse"),
        (27, "player", "fertility_shift"),
        (41, "player", "catastrophe"),
        (53, "revert", "regen_pulse"),
    ]
    fold = pacing.fold(log_of(rd), 60)
    assert sorted(fold.active) == ["patch:1"]  # Bloom, still running at 60
    env = env_at(rd, 60)
    assert env.patch_gain[2] == 1.0 and env.patch_gain[1] == actions._f32(1.8) and env.regen_mult == 1.0
    # on from there: the player does it again at the same step
    cfg = rd.config()
    _, script, res = run(cfg, w, END - 60, {60: [blight(4)]}, start=(rd, 60))
    assert res.status == "done" and not script.refused
    assert (w / DIRECTOR_LOG_FILE).read_bytes() == original  # no second header, the same lines
    assert checkpoint_digests(RunDir.open(w)) == checkpoint_digests(RunDir.open(session))


def test_a_power_at_the_origin_step_is_logged_and_replays(tmp_path: Path) -> None:
    cfg = world()
    rd, script, _ = run(cfg, tmp_path / "live", 60, {0: [food(1, duration=30)]})
    assert steps_of(rd) == [(0, "player", "regen_pulse"), (30, "revert", "regen_pulse")]
    log.validate_log(rd.read_jsonl(DIRECTOR_LOG_FILE), cfg)
    assert env_at(rd, 0).regen_mult == 1.0  # the checkpoint at the origin does not hold it
    rcfg = replay_cfg(cfg)
    rrd = RunDir.create(tmp_path / "replay", rcfg)
    rrd.update_meta(start_step=0)
    shutil.copyfile(rd.path / DIRECTOR_LOG_FILE, rrd.path / DIRECTOR_LOG_FILE)
    runner.run(rrd, rcfg, init_state(rcfg), 60, quiet=True, director=Director(rrd, rcfg, "replay", 0))
    assert checkpoint_digests(rrd) == checkpoint_digests(rd)
    # a replay that starts at the origin of a played world applies a record at that very step
    assert [s for s, _ in script.applied] == [0, 30]


def test_a_replay_from_a_checkpoint_applies_the_records_at_and_after_it(tmp_path: Path) -> None:
    """Entries below the start step are in the checkpoint. At the start step a boundary entry is (applied
    before the checkpoint), a player's is not (applied at the top of the iteration that starts there)."""
    cfg = world()
    rd = RunDir.create(tmp_path / "r", cfg)
    mine = {
        "step_applied": 60,
        "source": "player",
        "scope": "player",
        "action": "regen_pulse",
        "params": {"mult": 1.3},
        "prev": {"mult": 1.0},
        "duration": 40,
        "command": 1,
        "power": "food",
        "specs": "game-v0",
        "cost": 1,
    }
    rule = {
        "step_applied": 60,
        "source": "rule",
        "action": "drift_change",
        "params": {"mult": 2.0},
        "prev": {"mult": 1.0},
        "duration": None,
    }
    header = {"kind": "header", "step": 0, "rules_version": 2, "game_specs": "game-v0"}
    rd.append_jsonl(DIRECTOR_LOG_FILE, [header, rule, mine, {**rule, "step_applied": 80}])
    rcfg = replay_cfg(cfg)
    at = Director(rd, rcfg, "replay", 60)
    assert {t: [e["source"] for e in es] for t, es in at.pending.items()} == {80: ["rule"]}
    assert {t: [e["source"] for e in es] for t, es in at.pending_player.items()} == {60: ["player"]}
    early = Director(rd, rcfg, "replay", 40)
    assert sorted(early.pending) == [60, 80] and sorted(early.pending_player) == [60]
    late = Director(rd, rcfg, "replay", 61)
    assert dict(late.pending) == {80: late.pending[80]} and dict(late.pending_player) == {}


def test_a_replay_still_applies_a_record_when_a_stop_is_requested_at_its_step(
    session: Path, tmp_path: Path
) -> None:
    """The loop runs one more step after a stop request; a replay applies what the log says at that step, or
    the checkpoint it writes would miss it and a resumed replay would never apply it."""
    cfg = replay_cfg(RunDir.open(session).config())

    def go(name: str, stop_at: int | None, steps: int) -> RunDir:
        rd = RunDir.create(tmp_path / name, cfg)
        rd.update_meta(start_step=0)
        shutil.copyfile(session / DIRECTOR_LOG_FILE, rd.path / DIRECTOR_LOG_FILE)
        runner.run(
            rd,
            cfg,
            init_state(cfg),
            steps,
            quiet=True,
            director=Director(rd, cfg, "replay", 0),
            pacer=Script({}, stop_at),
        )
        return rd

    stopped, plain = go("stopped", 13, 100), go("plain", None, 14)
    assert max(checkpoint_digests(stopped)) == 14 == max(checkpoint_digests(plain))
    assert checkpoint_digests(stopped)[14] == checkpoint_digests(plain)[14]


# --- the stop check ---


def test_a_stop_drops_the_queue_unlogged_and_still_ends_an_effect_that_is_due(tmp_path: Path) -> None:
    cfg = world()
    plan = {13: [food(1)], 53: [blight(2)]}  # Food ends at 53: the stop comes at its step
    rd, script, res = run(cfg, tmp_path / "s", END, plan, stop_at=53)
    assert (res.status, res.exit_code) == ("interrupted", runner.ExitCode.INTERRUPTED)
    assert steps_of(rd) == [(13, "player", "regen_pulse"), (53, "revert", "regen_pulse")]  # no Blight
    assert not script.refused and [e["step_applied"] for _, e in script.applied] == [13, 53]
    assert max(s for s, _ in rd.checkpoints()) == 54  # one more step ran; the world was consistent
    assert env_at(rd, 54).regen_mult == 1.0 and env_at(rd, 54).patch_gain[2] == 1.0
    # played on from the quit checkpoint, it is the uninterrupted world (the Blight request was never made)
    _, _, _ = run(cfg, tmp_path / "s", END - 54, {}, start=(rd, 54))
    whole, _, _ = run(cfg, tmp_path / "whole", END, {13: [food(1)]})
    assert checkpoint_digests(RunDir.open(tmp_path / "s"))[END] == checkpoint_digests(whole)[END]
    assert (tmp_path / "s" / DIRECTOR_LOG_FILE).read_bytes() == (whole.path / DIRECTOR_LOG_FILE).read_bytes()


def test_a_stop_with_a_queued_power_applies_nothing(tmp_path: Path) -> None:
    """``before_step`` returns at once on a ``quit`` but the loop runs one more step: a power queued in that
    iteration would be applied and logged for a world that is going away."""
    rd, script, res = run(world(), tmp_path / "q", END, {20: [strike(1)]}, stop_at=20)
    assert res.status == "interrupted" and steps_of(rd) == [] and not script.applied
    d = Director(rd, world(), "player", 21)
    assert not d.has_queued()
    assert env_at(rd, 21).regen_mult == 1.0


# --- eligibility and refusals ---


def test_refusals_are_reported_not_logged_and_the_rest_of_the_queue_goes_on(tmp_path: Path) -> None:
    plan = {
        13: [food(1), food(2), strike(3)],  # the second Food is cooling down; the Strike is fine
        50: [food(4)],  # Food has not ended (53), and its cooldown runs 20 steps beyond
        60: [food(5, mult=1.0), bloom(6, patch=9)],  # the dead zone; a patch that is not there
        70: [cancel(7, "bloom"), strike(8)],  # nothing to cancel; the Strike has cooled down
        80: [food(9)],  # past the cooldown (53 + 20 = 73)
    }
    rd, script, res = run(world(), tmp_path / "r", END, plan)
    assert res.status == "done"
    why = {q.command: m for _, q, m in script.refused}
    assert why[2] == why[4] == "regen_pulse is cooling down until step 73"
    assert "refused" in why[5] and "patch" in why[6] and why[7] == "no effect of bloom to cancel"
    assert 3 not in why and 8 not in why and 9 not in why
    assert steps_of(rd) == [
        (13, "player", "regen_pulse"),
        (13, "player", "catastrophe"),
        (53, "revert", "regen_pulse"),
        (70, "player", "catastrophe"),
        (80, "player", "regen_pulse"),  # its end, at 120, is where the run ends
    ]
    log.validate_log(rd.read_jsonl(DIRECTOR_LOG_FILE), rd.config())
    assert sorted(why) == [2, 4, 5, 6, 7]


def test_cooldown_runs_from_the_planned_end_and_influence_is_a_rolling_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pacing, "PLAYER_BUDGET", 3)
    plan = {
        13: [food(1, duration=30)],  # ends at 43, cooldown to 63
        50: [food(2)],  # cooling down
        63: [food(3, duration=30)],  # allowed (2 units used); ends 93
        64: [strike(4)],  # 3 units: allowed
        70: [bloom(5)],  # the influence is spent until 13 + 400
    }
    rd, script, _ = run(world(), tmp_path / "c", 100, plan)
    why = {q.command: m for _, q, m in script.refused}
    assert why[2] == "regen_pulse is cooling down until step 63"
    assert why[5] == "influence: 3 of 3 units are in use"
    assert [e["command"] for e in log_of(rd) if e["source"] == "player"] == [1, 3, 4]


def test_a_cancel_ends_the_effect_at_once_and_the_cooldown_stays_the_planned_one(tmp_path: Path) -> None:
    plan = {
        13: [food(1, duration=60)],  # planned end 73, cooldown to 93
        30: [cancel(2, "food")],
        80: [food(3)],  # cooling down although nothing runs
        93: [food(4, duration=30)],
    }
    rd, script, _ = run(world(), tmp_path / "x", 130, plan)
    assert steps_of(rd)[:3] == [
        (13, "player", "regen_pulse"),
        (30, "revert", "regen_pulse"),
        (93, "player", "regen_pulse"),
    ]
    revert = log_of(rd)[1]
    assert (revert["command"], revert["reverts"], revert["reason"], revert["scope"]) == (
        2,
        13,
        "cancelled by the player",
        "player",
    )
    assert revert["params"] == {"mult": 1.0} and env_at(rd, 30).regen_mult == actions._f32(
        1.3
    )  # the ckpt at 30 is before
    assert env_at(rd, 60).regen_mult == 1.0
    assert [m for _, q, m in script.refused] == ["regen_pulse is cooling down until step 93"]
    log.validate_log(
        rd.read_jsonl(DIRECTOR_LOG_FILE), rd.config()
    )  # a cancel is a valid revert of an active effect
    # no refund: the influence of the cancelled Food is still spent in the window
    assert pacing.budget_used(pacing.fold(log_of(rd), 100, "player"), 100, 400) == 2


def test_cancel_names_the_patch_of_a_bloom_or_ends_all_of_them(tmp_path: Path) -> None:
    plan = {
        13: [bloom(1, patch=0, duration=100)],
        14: [blight(2, patch=3, duration=100)],
        40: [cancel(3, "bloom", patch=2)],
        41: [cancel(4, "bloom", patch=0)],
        42: [cancel(5, "blight")],
    }
    rd, script, _ = run(world(), tmp_path / "p", 60, plan)
    assert [(q.command, m) for _, q, m in script.refused] == [(3, "no effect of bloom to cancel")]
    ends = [
        (e["step_applied"], e["action"], e["params"]["patch"], e["command"])
        for e in log_of(rd)
        if e["source"] == "revert"
    ]
    assert ends == [(41, "fertility_shift", 0, 4), (42, "habitat_loss", 3, 5)]


def test_a_dry_run_changes_nothing_and_sees_the_queue(tmp_path: Path) -> None:
    cfg = world()
    bare, _, _ = run(cfg, tmp_path / "bare", 60, {13: [food(1)]})
    preview = {13: [strike(9), food(8)], 30: [cancel(7, "food")], 31: [bloom(6, gain=0.1)]}
    rd, script, _ = run(cfg, tmp_path / "dry", 60, {13: [food(1)]}, preview=preview)
    assert checkpoint_digests(rd) == checkpoint_digests(bare) and steps_of(rd) == steps_of(bare)
    [dry_strike] = script.planned[13, 9]
    assert (dry_strike["action"], dry_strike["step_applied"], dry_strike["command"]) == ("catastrophe", 13, 9)
    assert (
        script.planned[13, 8] == "regen_pulse is cooling down until step 73"
    )  # the queued Food is folded in
    [dry_cancel] = script.planned[30, 7]
    assert (dry_cancel["source"], dry_cancel["reverts"], dry_cancel["command"]) == ("revert", 13, 7)
    [dry_bloom] = script.planned[31, 6]
    assert dry_bloom["params"]["gain"] == 1.5  # clamped, as it would be applied


def test_a_dry_run_that_is_not_possible_says_why(tmp_path: Path) -> None:
    cfg = world()
    rd = RunDir.create(tmp_path / "d", cfg)
    rd.update_meta(start_step=0)
    d = make_director(rd, cfg, 0)
    assert d is not None
    state = init_state(cfg)
    with pytest.raises(ActionError, match="refused"):
        d.plan_player(state, food(1, mult=1.0), 5)
    with pytest.raises(ActionError, match="unknown power"):
        d.plan_player(state, PlayerRequest(1, "weather", {}, None), 5)
    with pytest.raises(ActionError, match="no effect of food"):
        d.plan_player(state, cancel(1, "food"), 5)
    d.queue_player(bloom(2, patch=1))
    with pytest.raises(ActionError, match="patch:1 already has an active effect"):  # the queue is folded in
        d.plan_player(state, blight(3, patch=1), 5)
    assert d.plan_player(state, blight(3, patch=2), 5)[0]["params"] == {"patch": 2, "gain": 0.0}
    with pytest.raises(ActionError, match="unknown power"):
        d.queue_player(PlayerRequest(1, "weather", {}, None))
    assert d.has_queued() and log_of(rd) == []
    rep = Director(rd, replay_cfg(cfg), "replay", 0)
    with pytest.raises(ActionError, match="replay"):
        rep.queue_player(food(1))
    with pytest.raises(ActionError, match="replay"):
        rep.plan_player(state, food(1), 5)


# --- the rule tier beside a player ---


def test_the_rule_tier_and_the_player_keep_separate_budgets_and_share_levers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Under ``--director rules`` (live mode) a rule pulse holds the regen lever and the player's Food is
    refused while it runs; neither one's moves enter the other's budget, cooldowns or ``last_*``."""
    from alife.director.rules import Proposal

    monkeypatch.setitem(
        actions.MENU, "regen_pulse", dataclasses.replace(actions.MENU["regen_pulse"], duration=(100, 2000))
    )
    plan = {140: [Proposal("regen_pulse", {"mult": 0.6}, 100, "scripted", "a damper")]}
    monkeypatch.setattr("alife.director.rules.decide", scripted_decide(plan))
    cfg = world("live")
    rd, script, _ = run(
        cfg, tmp_path / "rules", 260, {20: [food(1, duration=40)], 150: [food(2)], 180: [strike(3)]}
    )
    assert steps_of(rd) == [
        (20, "player", "regen_pulse"),
        (60, "revert", "regen_pulse"),  # the player's, at its end: a boundary and a checkpoint step
        (140, "rule", "regen_pulse"),
        (180, "player", "catastrophe"),
        (240, "revert", "regen_pulse"),  # the rule tier's, at its boundary
    ]
    assert [(q.command, m) for _, q, m in script.refused] == [(2, "regen already has an active effect")]
    entries = log_of(rd)
    assert [e.get("scope") for e in entries] == ["player", "player", None, "player", None]
    # the rule tier did not revert the player's Food at the boundary 60: it ended after the checkpoint at 60
    assert env_at(rd, 60).regen_mult == actions._f32(1.3) and env_at(rd, 90).regen_mult == 1.0
    rule, mine = pacing.fold(entries, 260, "rule"), pacing.fold(entries, 260, "player")
    assert [c for _, c in rule.spent] == [1] and [c for _, c in mine.spent] == [1, 1]
    assert set(rule.ends) == {"regen_pulse"} and set(mine.ends) == {"regen_pulse", "catastrophe"}
    log.validate_log(rd.read_jsonl(DIRECTOR_LOG_FILE), rd.config())


def test_the_rule_tier_s_context_is_its_own_scope(tmp_path: Path) -> None:
    cfg = world("live")
    rd = RunDir.create(tmp_path / "c", cfg)
    rd.update_meta(start_step=0)
    d = Director(rd, cfg, "live", 0)
    d.entries = [held_food(duration=400)]
    ctx = d._context(init_state(cfg), None, 100, [])  # type: ignore[arg-type]
    assert ctx.pacing.spent == () and ctx.pacing.ends == {} and ctx.pacing.last_step is None
    assert sorted(ctx.pacing.active) == ["regen"] and ctx.pacing.active["regen"].scope == "player"


def held_food(duration: int) -> dict[str, Any]:
    """A logged Food of the player at 13."""
    return {
        "step_applied": 13,
        "source": "player",
        "scope": "player",
        "action": "regen_pulse",
        "params": {"mult": 1.3},
        "prev": {"mult": 1.0},
        "duration": duration,
        "cost": 1,
        "power": "food",
        "command": 1,
        "specs": "game-v0",
    }


def test_the_player_s_effect_makes_the_lever_busy_for_the_rule_tier_without_spending_its_budget() -> None:
    """The rule tier waits out a lever the player holds (a blocked note, not a supersede), except the
    extinction rescue, which may override the regen lever as it may any older pulse."""
    cfg = world("live")
    entries = [held_food(duration=400)]
    p = rules.RULE_PARAMS
    persona = p.persona("gardener")
    ctx = rules.Context(
        100,
        0,
        "gardener",
        6,
        cfg.world.size,
        None,
        pacing.fold(entries, 100, "rule"),
        env_view(init_state(cfg)),
    )  # type: ignore[arg-type]
    assert rules.eligibility(ctx, p, "regen_pulse", {"mult": 0.6}) == "regen is held by the player"
    assert rules._blocked(ctx, p, persona, "regen_pulse", {"mult": 1.5}, safety=True, override=True) is None
    assert pacing.budget_used(ctx.pacing, 100) == 0  # the player's unit is not the rule tier's
    assert rules.eligibility(ctx, p, "habitat_loss", {"patch": 1}) is None


def test_the_rule_tier_does_not_revert_a_player_effect_and_the_player_does_not_revert_the_rule_s() -> None:
    entries = [
        held_food(duration=40),
        {
            "step_applied": 20,
            "source": "rule",
            "action": "habitat_loss",
            "params": {"patch": 1, "gain": 0.0},
            "prev": {"patch": 1, "gain": 1.0},
            "duration": 100,
            "rule": "x",
        },
    ]
    p = pacing.fold(entries, 500)
    assert [e.key for e in pacing.due_reverts(p, 500, "rule")] == ["patch:1"]
    assert [e.key for e in pacing.due_reverts(p, 500, "player")] == ["regen"]


# --- a research loop is the old one ---


def test_without_a_player_apply_pending_is_one_comparison(tmp_path: Path) -> None:
    for mode in ("live", "replay"):
        cfg = world(mode)
        rd = RunDir.create(tmp_path / mode, cfg)
        rd.update_meta(start_step=0)
        d = Director(rd, cfg, mode, 0)
        s = init_state(cfg)
        before = {p.name: p.read_bytes() for p in rd.path.iterdir() if p.is_file()}
        assert d.apply_pending(s, 7) is s and d.apply_pending(s, 7, stopping=True) is s
        assert d.take_player() == ([], [])
        assert before == {p.name: p.read_bytes() for p in rd.path.iterdir() if p.is_file()}
    assert make_director(RunDir.create(tmp_path / "x", world("off")), world("off"), 0) is None


def test_a_research_resume_appends_no_header_and_a_played_one_appends_only_the_change(tmp_path: Path) -> None:
    rd = RunDir.create(tmp_path / "w", world("live"))
    rd.update_meta(start_step=0)
    Director(rd, world("live"), "live", 0)
    Director(rd, world("live"), "live", 20)  # a research resume: nothing appended
    assert (
        len(rd.read_jsonl(DIRECTOR_LOG_FILE)) == 1 and "game_specs" not in rd.read_jsonl(DIRECTOR_LOG_FILE)[0]
    )
    make_director(rd, world("live"), 20, game_specs=True)  # `alife play` adopting it: one header with the key
    heads = rd.read_jsonl(DIRECTOR_LOG_FILE)
    assert [h.get("game_specs") for h in heads] == [None, "game-v0"]
    make_director(rd, world("live"), 20, game_specs=True)
    assert len(rd.read_jsonl(DIRECTOR_LOG_FILE)) == 2
    log.validate_log(heads, world("live"))


def test_the_commands_file_is_not_written_by_the_channel(session: Path) -> None:
    assert not (session / COMMANDS_FILE).exists()  # the pacer of `alife play` writes it (spec §6.6)


# --- alife play ---


def test_the_default_launch_is_the_player_mode_when_the_world_has_a_director_and_off_otherwise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capfdbinary: pytest.CaptureFixture[bytes]
) -> None:
    monkeypatch.setattr(play, "PLAY_CADENCE", PLAY_CADENCE)
    snapshots(monkeypatch, 10)
    common = [
        "--set",
        "brain.weight_dtype=float32",
        "--set",
        "population.n_init=24",
        "--no-control",
        "--rate",
        "max",
        "--until",
        "60",
    ]
    on, off, rules = tmp_path / "on", tmp_path / "off", tmp_path / "rules"
    assert (
        cli.main(
            [
                "play",
                "--preset",
                "tiny",
                "--set",
                "director.enabled=true",
                "--set",
                "director.rule_every=20",
                "--out",
                str(on),
                *common,
            ]
        )
        == 0
    )
    assert cli.main(["play", "--preset", "tiny", "--out", str(off), *common]) == 0
    assert (
        cli.main(
            [
                "play",
                "--preset",
                "tiny",
                "--director",
                "rules",
                "--set",
                "director.rule_every=20",
                "--out",
                str(rules),
                *common,
            ]
        )
        == 0
    )
    capfdbinary.readouterr()
    assert (
        RunDir.open(on).config().run.director == "player" and RunDir.open(off).config().run.director == "off"
    )
    assert RunDir.open(rules).config().run.director == "live"
    [header] = RunDir.open(on).read_jsonl(DIRECTOR_LOG_FILE)
    assert header["game_specs"] == "game-v0"
    assert not (off / DIRECTOR_LOG_FILE).exists()
    assert (
        RunDir.open(rules).read_jsonl(DIRECTOR_LOG_FILE)[0]["game_specs"] == "game-v0"
    )  # `rules` takes powers too
    # inert: the same world as without a director
    assert checkpoint_digests(RunDir.open(on))[60] == checkpoint_digests(RunDir.open(off))[60]
    # and played again (a resume of a played world) it stays in the player mode and appends no header
    assert cli.main(["play", str(on), "--no-control", "--rate", "max", "--until", "90"]) == 0
    capfdbinary.readouterr()
    assert (
        RunDir.open(on).config().run.director == "player"
        and len(RunDir.open(on).read_jsonl(DIRECTOR_LOG_FILE)) == 1
    )
