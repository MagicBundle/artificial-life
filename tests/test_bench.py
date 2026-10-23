"""WP6 bench tests (§15.4, §21, §22.7): report shape, staged wiring, donation health, determinism, and the
observer blocks (§28, §29.12)."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from alife import bench
from alife.bench import STAGES, format_report, run_bench
from alife.config import BRAIN_IMPLS, WEIGHT_DTYPES, Config, apply_overrides
from alife.state import CounterIdx, LedgerIdx, init_state, n_counters, state_digest, to_numpy
from alife.step import make_step
from game_helpers import social_every

# tiny3d without the buoyancy actuator (validate requires its other keys at 0 then).
OFF_3D = [
    "physics.buoyancy_max=0",
    "energy.buoyancy_cost=0",
    "energy.buoyancy_cap_cost=0",
    "mutation.buoy_init=0",
    "mutation.buoy_sigma=0",
]


def test_smoke_all_parts_on_tiny(tiny_cfg: Config) -> None:
    r = run_bench(tiny_cfg, steps=3, warmup=2, stages=True, determinism=True, brain_variants=True)
    json.dumps(r, allow_nan=False)  # JSON-compatible as written by `alife bench --json`
    assert r["steps"] == 3 and r["warmup"] == 2
    assert r["config"]["n_max"] == 1024 and r["config"]["weight_dtype"] == "float32"
    for key in ("ms_per_step", "ms_per_step_median", "ms_per_step_p90", "steps_per_s", "peak_mem_gb"):
        assert r[key] > 0, key
    assert r["compile_s"] >= 0 and r["first_step_s"] > 0
    # population, births, immigrants and deaths all describe the timed window
    assert r["n_alive_end"] == r["n_alive_start"] + r["births"] + r["immigrants"] - r["deaths"] > 0
    assert r["diag_ok"] and r["invariants_ok"] and r["invariant_problems"] == []
    assert r["donation"]["ok"], r["donation"]
    assert r["weights_gb"] > 0 and r["state_gb"] > r["weights_gb"]

    st = r["stages"]
    assert list(st["ms"]) == [name for name, _ in STAGES] == list(st["labels"])
    assert all(v > 0 for v in st["ms"].values())
    assert sum(st["frac"].values()) == pytest.approx(1.0)
    assert st["total_ms"] == pytest.approx(sum(st["ms"].values())) and st["inflation"] > 0

    dt = r["determinism"]
    assert dt["match"] and dt["digest_a"] == dt["digest_b"] and dt["steps"] == 3

    variants = r["brain_variants"]
    assert [(v["impl"], v["weight_dtype"]) for v in variants] == [
        (i, w) for i in BRAIN_IMPLS for w in WEIGHT_DTYPES
    ]
    for v in variants:
        assert "error" not in v, v
        assert v["ms"] > 0 and v["weight_gb_per_s"] > 0

    text = format_report(r)
    for needle in ("steps/s", "donation health: ok", "determinism", "S8-S14", "bfloat16"):
        assert needle in text


def test_minimal_report_and_validation(micro_cfg: Config) -> None:
    r = run_bench(micro_cfg, steps=1, warmup=0)
    assert r["warmup"] == 1  # the compile call always runs untimed
    assert not {"stages", "determinism", "brain_variants", "observe"} & r.keys()
    assert "stages" not in format_report(r)
    with pytest.raises(ValueError, match="steps"):
        run_bench(micro_cfg, steps=0)
    with pytest.raises(ValueError, match="warmup"):
        run_bench(micro_cfg, steps=1, warmup=-1)


def test_observers_block(tiny_cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    """``observe=True`` adds the observers' cost against the plain step, as blocks in the order C O O C.

    At a snapshot cadence of 4 the formulas of §29.12 give a warm-up of ``max(warmup, 1, rules.every, S + 1)``
    and ``ceil(max(steps // 2, 1) / S) * S`` timed steps per block."""
    social_every(monkeypatch, 4)
    r = run_bench(tiny_cfg, steps=3, warmup=1, observe=True)
    json.dumps(r, allow_nan=False)
    ob = r["observe"]
    assert set(ob) == {
        "warmup",
        "block_pairs",
        "steps_per_block",
        "control_ms_median",
        "observed_ms_median",
        "overhead_ms_median",
        "control_ms_per_step",
        "observed_ms_per_step",
        "overhead_ms_mean",
        "observed_steps_per_s",
        "snapshot_step_ms_mean",
        "other_step_ms_mean",
        "trajectory_match",
        "social",
    }
    assert ob["observed_ms_per_step"] > 0 and ob["control_ms_per_step"] > 0 and ob["observed_steps_per_s"] > 0
    assert ob["overhead_ms_median"] == pytest.approx(ob["observed_ms_median"] - ob["control_ms_median"])
    assert ob["overhead_ms_mean"] == pytest.approx(ob["observed_ms_per_step"] - ob["control_ms_per_step"])
    assert ob["observed_steps_per_s"] == pytest.approx(1e3 / ob["observed_ms_per_step"])
    assert ob["block_pairs"] == 2 and ob["steps_per_block"] == 4  # ceil(max(steps // 2, 1) / 4) * 4
    assert ob["warmup"] == 8  # max(1, 1, rules.every, S + 1): the first evaluation is never timed
    assert ob["trajectory_match"] is True
    text = format_report(r)
    assert "observers (2 block pairs of 4 steps)" in text and "trajectory identical" in text
    assert "social (a snapshot every 4 steps)" in text and "ms/step amortised" in text
    assert run_bench(tiny_cfg, steps=3, warmup=20, observe=True)["observe"]["warmup"] == 20


@pytest.mark.parametrize(
    ("steps", "per_block"), [(1, 4), (2, 4), (8, 4), (9, 4), (10, 8), (17, 8), (18, 12), (30, 16)]
)
def test_the_timed_steps_of_a_block_hold_whole_snapshots(
    steps: int, per_block: int, tiny_cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``ceil(max(steps // 2, 1) / S) * S`` timed steps per block, always a whole number of snapshots."""
    social_every(monkeypatch, 4)
    ob = bench._bench_observe(tiny_cfg, steps, 1)
    assert ob["steps_per_block"] == per_block and per_block % 4 == 0


def test_median_ms_is_a_median(monkeypatch: pytest.MonkeyPatch) -> None:
    """The parts of a snapshot are reported by their median over the repetitions, not their mean."""
    durations = iter([0.001, 0.001, 0.010])  # one slow repetition
    clock = [0.0]

    def fake() -> float:
        return clock[0]

    calls = 0

    def work() -> None:
        nonlocal calls
        calls += 1
        clock[0] += next(durations)

    monkeypatch.setattr(bench.time, "perf_counter", fake)
    assert bench._median_ms(work, 3) == pytest.approx(1.0) and calls == 3


def test_the_device_part_of_a_snapshot_is_evaluated_when_timed(
    tiny_cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``snapshot_device_ms_median`` must include the evaluation of the snapshot's arrays, not only the
    dispatch of the closure: with twelve chained 1024 x 1024 products added to the closure's output, the
    evaluation takes milliseconds on any backend and the dispatch does not."""
    from alife.game.observer import Observer

    social_every(monkeypatch, 4)
    real_fresh = Observer.fresh

    def fresh(state, cfg, **kw):
        obs = real_fresh(state, cfg, **kw)
        inner = obs._snap_fn

        def heavy(*a):
            snap = inner(*a)
            x = mx.ones((1024, 1024), dtype=mx.float32) / 1024.0
            for _ in range(12):
                x = x @ x
            return snap._replace(uid=snap.uid + (x[0, 0] * 0).astype(snap.uid.dtype))

        obs._snap_fn = heavy
        return obs

    monkeypatch.setattr(Observer, "fresh", staticmethod(fresh))
    ob = bench._bench_observe(tiny_cfg, 2, 1)
    assert ob["social"]["snapshot_device_ms_median"] > 2.0, ob["social"]


def test_the_social_part_of_the_observers_block(tiny_cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    """``observe.social`` times the parts of one snapshot; the step times split into snapshot steps and
    others, and the amortised cost is (device + host) per step of the cadence."""
    social_every(monkeypatch, 4)
    ob = bench._bench_observe(tiny_cfg, 4, 1)
    json.dumps(ob, allow_nan=False)
    assert ob["snapshot_step_ms_mean"] > 0 and ob["other_step_ms_mean"] > 0
    # a block of 4 timed steps holds exactly one snapshot step, so the two classes of observed steps average
    # to the observed mean with weights 1 and 3
    assert ob["steps_per_block"] == 4
    expect = (ob["snapshot_step_ms_mean"] * 1 + ob["other_step_ms_mean"] * 3) / 4
    assert ob["observed_ms_per_step"] == pytest.approx(expect)
    so = ob["social"]
    assert set(so) == {
        "snapshot_every",
        "reps",
        "components_ms_median",
        "snapshot_device_ms_median",
        "snapshot_host_ms_median",
        "heavy_ms_median",
        "amortised_ms_per_step",
    }
    assert so["snapshot_every"] == 4 and so["reps"] == bench.SOCIAL_REPS == 10
    for key in ("components_ms_median", "snapshot_device_ms_median", "snapshot_host_ms_median"):
        assert so[key] > 0, key
    assert so["heavy_ms_median"] > 0
    assert so["amortised_ms_per_step"] == pytest.approx(
        (so["snapshot_device_ms_median"] + so["snapshot_host_ms_median"]) / 4
    )
    assert "heavy metrics" in format_report({**_report_stub(), "observe": ob})


def test_observers_block_runs_the_observer_in_every_step(
    tiny_cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The observed arm folds every step (warm-up included) of each of its blocks into a fresh observer, and
    the social part steps ``warm`` more steps with a third."""
    from alife.game.observer import Observer

    social_every(monkeypatch, 4)
    seen: list[int] = []
    real = Observer.update

    def spy(self: Observer, state, acts, t: int) -> None:
        seen.append(t)
        real(self, state, acts, t)

    monkeypatch.setattr(Observer, "update", spy)
    ob = bench._bench_observe(tiny_cfg, 2, 1)  # warm-up 8 (rules.every), 4 timed steps per block (S = 4)
    assert ob["warmup"] == 8 and ob["steps_per_block"] == 4
    # two observed blocks (8 warm-up and 4 timed steps each), then the social part's `warm` steps
    assert seen == [*range(1, 13), *range(1, 13), *range(1, 9)]


def test_observers_block_takes_the_records_after_every_eval_and_writes_nothing(
    tiny_cfg: Config, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """As the runner does: ``take_records()`` after each step's eval, so that a snapshot settles at its own
    step; the records are dropped and the bench writes no file."""
    from alife.game.observer import Observer
    from alife.game.society import Society
    from alife.runio import RunDir

    social_every(monkeypatch, 4)
    monkeypatch.chdir(tmp_path)
    events: list[tuple[str, int]] = []
    real_update, real_take, real_settle = Observer.update, Observer.take_records, Society.settle

    def update(self: Observer, state, acts, t: int) -> None:
        events.append(("update", t))
        real_update(self, state, acts, t)

    def take(self: Observer):
        events.append(("take", self.step))
        return real_take(self)

    def settle(self: Society, h, step: int):
        events.append(("settle", step))
        return real_settle(self, h, step)

    def forbidden(self: RunDir, records) -> None:
        raise AssertionError("the bench must not write the social log")

    monkeypatch.setattr(Observer, "update", update)
    monkeypatch.setattr(Observer, "take_records", take)
    monkeypatch.setattr(Society, "settle", settle)
    monkeypatch.setattr(RunDir, "append_social", forbidden)
    ob = bench._bench_observe(tiny_cfg, 2, 1)
    # every update is followed by a take_records of the same step, before the next update
    steps = [e for e in events if e[0] != "settle"]
    assert steps[0::2] == [("update", t) for _, t in steps[0::2]]
    pairs = list(zip(steps[0::2], steps[1::2], strict=True))
    assert all(u[0] == "update" and k[0] == "take" and u[1] == k[1] for u, k in pairs)
    assert len(steps) % 2 == 0 and len(steps) == 2 * (2 * 12 + 8)
    # a snapshot is settled by the take of its own step (never by the next update); the last SOCIAL_REPS
    # events are the timed settles of the social part, on copies of the society, one cadence after the last
    # settled snapshot (step 8, from the social part's own 8 warm-up steps)
    copies, taken = events[-bench.SOCIAL_REPS :], events[: -bench.SOCIAL_REPS]
    assert copies == [("settle", 12)] * bench.SOCIAL_REPS
    for i, e in enumerate(taken):
        if e[0] == "settle":
            assert taken[i - 1] == ("take", e[1]), (i, taken[i - 1], e)
    snapshots = [e[1] for e in taken if e[0] == "settle"]
    assert snapshots and all(t % 4 == 0 for t in snapshots) and ob["trajectory_match"] is True
    assert list(tmp_path.iterdir()) == []


def test_snapshot_steps_are_timed_apart_from_the_others(
    tiny_cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A snapshot's settle falls on the snapshot step (``take_records`` is inside the timed span): with 50 ms
    added to every ``take_records`` that settles one, the snapshot steps average over 50 ms and the others do
    not."""
    from alife.game.observer import Observer

    social_every(monkeypatch, 4)
    real = Observer.take_records

    def slow(self: Observer):
        records = real(self)
        if records:
            time.sleep(0.05)
        return records

    monkeypatch.setattr(Observer, "take_records", slow)
    ob = bench._bench_observe(tiny_cfg, 4, 1)
    assert ob["snapshot_step_ms_mean"] > 50 and ob["other_step_ms_mean"] < 40
    assert ob["observed_ms_per_step"] > ob["control_ms_per_step"]


def test_the_social_parts_use_the_social_rounds_and_the_heavy_metrics(
    tiny_cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The components alone are timed with ``social.colony_iters`` rounds (not the run's
    ``components_iters``, which the heavy metrics use), and the heavy metrics with ``heavy=True``."""
    from alife.game import observer

    rules = replace(observer.DEFAULT_SOCIAL, snapshot_every=4, colony_iters=7)
    monkeypatch.setattr(observer, "DEFAULT_SOCIAL", rules)
    assert tiny_cfg.run.components_iters != 7
    comps: list[tuple[str, int]] = []
    heavy_calls: list[bool] = []
    real_comps, real_snap = bench.metrics.components, bench.metrics.snapshot_metrics

    def comps_spy(partner, alive, iters: int):
        comps.append((sys._getframe(1).f_code.co_name, iters))
        return real_comps(partner, alive, iters)

    def snap_spy(s, cfg, *, heavy: bool):
        heavy_calls.append(heavy)
        return real_snap(s, cfg, heavy=heavy)

    monkeypatch.setattr(bench.metrics, "components", comps_spy)
    monkeypatch.setattr(bench.metrics, "snapshot_metrics", snap_spy)
    bench._bench_observe(tiny_cfg, 2, 1)
    assert ("comps", 7) in comps and all(it == 7 for who, it in comps if who == "comps")
    assert ("snapshot_metrics", tiny_cfg.run.components_iters) in comps
    assert heavy_calls == [True] * (bench.SOCIAL_REPS + 1)  # one untimed (compile) and the timed repetitions


def test_the_first_snapshot_of_each_block_is_in_the_warm_up(
    tiny_cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The first call of the snapshot closure with real ``Acts`` compiles it: it falls in the untimed warm-up
    of each observed block (warm = S + 1 = 11 at S = 10), and the timed steps hold whole snapshots."""
    from alife.game.observer import Observer

    social_every(monkeypatch, 10)
    blocks: list[list[int]] = []  # per observer: the steps whose update made a snapshot
    real = Observer.update

    def spy(self: Observer, state, acts, t: int) -> None:
        if t == 1:
            blocks.append([])
        real(self, state, acts, t)
        if self._snap is not None:  # the snapshot closure was called by this update
            blocks[-1].append(t)

    monkeypatch.setattr(Observer, "update", spy)
    ob = bench._bench_observe(tiny_cfg, 2, 1)
    warm, per_block = ob["warmup"], ob["steps_per_block"]
    assert warm == 11 and per_block % 10 == 0 and per_block == 10
    assert len(blocks) == 3  # two observed blocks and the social part's observer
    for snaps in blocks[:2]:
        assert snaps[0] < warm  # the first snapshot (step 10) is untimed
        timed = [t for t in snaps if t > warm]
        assert len(timed) == per_block // 10  # whole snapshots in the timed steps
        assert snaps == list(range(10, warm + per_block + 1, 10))
    assert blocks[2][0] < warm


def test_observers_block_sees_a_changed_trajectory(tiny_cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    """If observing ever changed the state, the report says so."""
    social_every(monkeypatch, 4)
    real = bench.step_mod.make_step

    def make_step_bad(cfg: Config, **kw):
        fn = real(cfg, **kw)
        if not kw.get("observe"):
            return fn

        def perturbed(s):
            s2, out = fn(s)
            return s2._replace(total=s2.total + 1), out

        return perturbed

    monkeypatch.setattr(bench.step_mod, "make_step", make_step_bad)
    ob = bench._bench_observe(tiny_cfg, 2, 1)
    assert ob["trajectory_match"] is False
    assert "trajectory DIFFERS" in format_report({**_report_stub(), "observe": ob})


def _report_stub() -> dict:
    """The fields ``format_report`` needs before its optional parts."""
    return {
        "alife_version": "x",
        "mlx_version": "x",
        "platform": "x",
        "device": "x",
        "config": {
            "n_max": 1,
            "max_births_per_step": 1,
            "n_init": 1,
            "world_size": 1.0,
            "k_neighbors": 1,
            "cell_capacity": 1,
            "hidden": 1,
            "weight_dtype": "float32",
            "brain_impl": "x",
            "compile": True,
            "runtime_checks": False,
        },
        "compile_s": 0.0,
        "ms_per_step": 1.0,
        "ms_per_step_median": 1.0,
        "ms_per_step_p90": 1.0,
        "steps_per_s": 1.0,
        "peak_mem_gb": 0.0,
        "n_alive_start": 1,
        "n_alive_end": 1,
        "births": 0,
        "deaths": 0,
        "diag_ok": True,
        "diag_flags": [],
        "invariants_ok": True,
        "donation": {"verdict": "ok", "excess_gb": 0.0, "excess_over_weights": 0.0},
    }


def test_staged_pipeline_equals_step(tiny_cfg: Config) -> None:
    """The stage wiring is exactly ``step.step``: bitwise equal states and StepOuts in eager mode."""
    cfg = replace(tiny_cfg, run=replace(tiny_cfg.run, compile=False))
    fused = make_step(cfg)
    staged = bench._StagedStep(cfg)
    s0 = init_state(cfg)
    s0 = s0._replace(birth_step=mx.where(s0.alive, -50, s0.birth_step))  # founders old enough to breed
    mx.eval(s0)
    s, holder = s0, [s0]
    births = 0
    for _ in range(4):
        s, out = fused(s)
        out_staged = staged(holder)
        mx.eval(s, out, holder[0], out_staged)
        assert state_digest(holder[0]) == state_digest(s)
        for a, b in zip(tree_flatten(out), tree_flatten(out_staged), strict=True):
            assert a[0] == b[0]
            np.testing.assert_array_equal(to_numpy(a[1]), to_numpy(b[1]), err_msg=a[0])
        births += int(to_numpy(out.counters)[CounterIdx.BIRTHS])
    assert births > 0  # the reproduction stage and the S0 flush of its pending rows were exercised
    assert all(len(v) == 4 for v in staged.times.values())


@pytest.mark.parametrize("actuator", [True, False])
def test_staged_pipeline_equals_step_3d(tiny3d_cfg: Config, actuator: bool) -> None:
    """The stage wiring mirrors every 3D branch of ``step.step`` (§27.8): bitwise equal states and StepOuts in
    eager mode, with and without the buoyancy actuator."""
    cfg = replace(tiny3d_cfg, run=replace(tiny3d_cfg.run, compile=False))
    if not actuator:
        cfg = apply_overrides(cfg, OFF_3D)
    fused = make_step(cfg)
    staged = bench._StagedStep(cfg)
    s0 = init_state(cfg)
    s0 = s0._replace(birth_step=mx.where(s0.alive, -50, s0.birth_step))  # founders old enough to breed
    mx.eval(s0)
    s, holder = s0, [s0]
    births = 0
    for _ in range(4):
        s, out = fused(s)
        out_staged = staged(holder)
        mx.eval(s, out, holder[0], out_staged)
        assert state_digest(holder[0]) == state_digest(s)
        for a, b in zip(tree_flatten(out), tree_flatten(out_staged), strict=True):
            assert a[0] == b[0]
            np.testing.assert_array_equal(to_numpy(a[1]), to_numpy(b[1]), err_msg=a[0])
        births += int(to_numpy(out.counters)[CounterIdx.BIRTHS])
    assert births > 0 and out.counters.shape == (n_counters(3),)
    assert (int(to_numpy(s.ledger_cum)[LedgerIdx.HEAT_BUOYANCY]) > 0) == actuator


def test_smoke_all_parts_on_tiny3d(tiny3d_cfg: Config) -> None:
    """Every part on ``tiny3d``, compiled; the stages also without the actuator, whose ``paid_buoy`` is 0 by
    construction (and derived from ``energy``, so never a constant compile output, I14)."""
    r = run_bench(tiny3d_cfg, steps=3, warmup=2, stages=True, determinism=True, brain_variants=True)
    json.dumps(r, allow_nan=False)
    assert r["n_alive_end"] == r["n_alive_start"] + r["births"] + r["immigrants"] - r["deaths"] > 0
    assert r["diag_ok"] and r["invariants_ok"] and r["invariant_problems"] == []
    assert r["determinism"]["match"] and all("error" not in v for v in r["brain_variants"])
    assert list(r["stages"]["ms"]) == [name for name, _ in STAGES]
    off = run_bench(apply_overrides(tiny3d_cfg, OFF_3D), steps=2, warmup=1, stages=True)
    assert off["diag_ok"] and off["invariants_ok"] and all(v > 0 for v in off["stages"]["ms"].values())


def test_donation_health_flags_a_retained_state(tiny_cfg: Config) -> None:
    """A step whose caller keeps the previous state forces a weight copy; the check must see it (F1)."""
    fn = make_step(tiny_cfg)
    healthy = bench._throughput(fn, tiny_cfg, 1, 2)["donation"]
    assert healthy["ok"] and healthy["saved_over_weights"] > 2 * bench.DONATION_LIMIT, healthy

    kept: list = [None]

    def leaky(s):
        kept[0] = s  # an observer holding on to the old state (the bug the check exists for)
        return fn(s)

    leak = bench._throughput(leaky, tiny_cfg, 1, 2)["donation"]
    assert not leak["ok"] and "WARNING" in leak["verdict"], leak


def test_donation_verdict_arithmetic() -> None:
    w = 1000
    ok = bench._donation(excess=600, excess_copy=1400, weights=w)
    assert ok["ok"] and ok["saved_over_weights"] == pytest.approx(0.8)
    assert ok["excess_over_weights"] == pytest.approx(0.6)
    bad = bench._donation(excess=1350, excess_copy=1400, weights=w)
    assert not bad["ok"] and bad["saved_over_weights"] == pytest.approx(0.05)
    edge = bench._donation(excess=0, excess_copy=int(bench.DONATION_LIMIT * w), weights=w)
    assert not edge["ok"]  # the copying step must be strictly more than the limit above


# --- bench --play (G3a, DESIGN §33.5) ---------------------------------------------------------------------

PLAY_CADENCE = {"metrics_every": 10, "heavy_metrics_every": 10, "checkpoint_every": 30, "keep_checkpoints": 3}


@pytest.fixture
def short_play(monkeypatch: pytest.MonkeyPatch) -> None:
    """Play with a metrics row and a snapshot every 10 steps and a checkpoint every 30."""
    from alife import play
    from play_helpers import snapshots

    monkeypatch.setattr(play, "PLAY_CADENCE", PLAY_CADENCE)
    snapshots(monkeypatch, 10)


def planet_template(tmp_path: Path, *overrides: str) -> tuple[Config, Path]:
    """A 256-row storm world (the planet's configuration in miniature, director levers compiled in) and its
    state saved as the bench's template."""
    from alife.runio import write_state_file
    from weather_helpers import storm3d_config

    cfg = storm3d_config("director.enabled=true", "director.rule_every=20", *overrides)
    path = tmp_path / "template.safetensors"
    write_state_file(path, init_state(cfg), cfg)
    return cfg, path


def test_play_bench_reports_every_arm_and_the_digests_equal_the_plain_run(
    tmp_path: Path, short_play: None
) -> None:
    from alife import play

    cfg, template = planet_template(tmp_path)
    r = bench.run_play_bench(
        cfg, template, tmp_path / "w", steps=40, warmup=10, rates=("max", "1"), directors=("off", "rules")
    )
    json.dumps(r, allow_nan=False)  # JSON-compatible, like every report
    assert [(a["director"], a["rate"]) for a in r["arms"]] == [
        ("off", "max"), ("off", "1"), ("rules", "max"), ("rules", "1"),
    ]  # fmt: skip
    assert r["digests_equal"] and set(r["plain_digest"]) == {"off", "rules"}
    assert r["engine_ms"] > 0 and r["cadence"] == PLAY_CADENCE
    for a in r["arms"]:
        assert a["digest_equal"] and a["digest"] == r["plain_digest"][a["director"]]
        assert a["steps_measured"] >= 39 and a["steps_per_s"] > 0 and a["seconds"] > 0
        q = a["step_ms"]
        assert 0 < q["p50"] <= q["p99"] <= q["max"]
        c = a["components_ms"]
        for key in ("engine", "loop_typical", "frma", "write", "tables", "boundary"):
            assert c[key] is not None and c[key] > 0, (a["name"], key)
        assert c["observer"] is not None and 0 <= c["observer"] <= c["loop_typical"], (
            "derived: step less the engine"
        )
        assert a["bytes_per_s"]["HEAD"] == 0, "the header went out before the measured window began"
        assert c["settle"] is not None, "a snapshot every 10 steps"
        assert a["bytes_per_s"]["FRMA"] > 0 and a["bytes_per_s"]["CLCK"] > 0 and a["mb_per_s"] > 0
        assert a["bytes_per_s"]["IDNT"] > 0 and a["bytes_per_s"]["MAP1"] > 0 and a["bytes_per_s"]["STRM"] > 0
        assert a["frames"] > 0 and a["late_steps"] >= 0 and a["save_stall_steps"] >= 0
        # 50 steps from step 0 (10 of warm-up), a boundary and a snapshot every 10, a tick every 25, a
        # checkpoint every 30 and at the end: what the stopwatch counted is the schedule's, not the machine's.
        assert a["step_counts"] == {"typical": 36, "snapshot": 0, "boundary": 3}, (
            "boundary steps are snapshot steps"
        )
        calls = a["part_calls"]
        assert (calls["boundary"], calls["tables"], calls["save"], calls["tick_storms"], calls["write"]) == (
            4,
            4,
            2,
            2,
            40,
        )
        assert a["drained_bytes"] >= sum(a["bytes_per_s"].values()) * a["seconds"] * 0.5, (
            "the reader got the stream"
        )
    # Pacing is a ceiling, not a speed: a slow machine can only fall short of it, so these hold on any
    # runner. (That "max" is unthrottled is the scheduler's own test, on a fake clock.)
    paced = [a for a in r["arms"] if a["rate"] == "1"]
    assert len(paced) == 2 and all(a["achieved_x"] <= 1.25 for a in paced), "a paced arm cannot beat its rate"
    assert all(a["seconds"] >= 0.8 * (a["steps_measured"] - 1) / play.BASE_RATE for a in paced)
    text = "\n".join(bench._format_play(r))
    assert "play loop" in text and "director rules" in text and "equal to the plain run" in text


def test_play_bench_gate_reads_the_documented_numbers() -> None:
    arm = {"rate": "1", "achieved_x": 0.99, "late_steps": 0, "step_ms": {"p99": 60.0}, "seconds": 310.0}
    g = bench._play_gate(arm)
    assert g["applies"] and g["pass"] and g["five_minutes"] and g["want"] == bench.PLAY_GATE_1X
    assert not bench._play_gate({**arm, "achieved_x": 0.97})["pass"]
    assert not bench._play_gate({**arm, "late_steps": 1})["pass"]
    assert not bench._play_gate({**arm, "step_ms": {"p99": 67.0}})["pass"]
    assert not bench._play_gate({**arm, "seconds": 120.0})["five_minutes"]
    fast = {"rate": "max", "achieved_x": 3.6, "late_steps": 0, "step_ms": {"p99": 40.0}, "seconds": 20.0}
    assert bench._play_gate(fast)["pass"] and not bench._play_gate({**fast, "achieved_x": 2.4})["pass"]
    assert bench._play_gate({"rate": "0.5", **{k: v for k, v in arm.items() if k != "rate"}}) == {
        "applies": False
    }


def test_play_bench_needs_the_levers_for_the_rule_tier(tmp_path: Path, short_play: None) -> None:
    from alife.config import ConfigError
    from alife.runio import write_state_file
    from weather_helpers import storm3d_config

    cfg = storm3d_config()
    path = tmp_path / "t.safetensors"
    write_state_file(path, init_state(cfg), cfg)
    with pytest.raises(ConfigError, match=r"director\.enabled"):
        bench.run_play_bench(
            cfg, path, tmp_path / "w", steps=5, warmup=1, rates=("max",), directors=("rules",)
        )
    for kwargs in ({"steps": 0}, {"warmup": -1}, {"rates": ()}, {"directors": ()}):
        args = {"steps": 5, "warmup": 1, "rates": ("max",), "directors": ("off",), **kwargs}
        with pytest.raises(ValueError):
            bench.run_play_bench(cfg, path, tmp_path / "w2", **args)
    with pytest.raises(ValueError, match="rate"):
        bench._rate_value("99")
    assert bench._rate_value("max") == float("inf") and bench._rate_value("0.25") == 0.25


def test_the_bench_pacer_changes_nothing_it_measures(tmp_path: Path, short_play: None) -> None:
    """A stopwatch pacer is a presenter like any other: the run is the plain run's, bit for bit (the report's
    `digests_equal` is that claim)."""
    cfg, template = planet_template(tmp_path)
    r = bench.run_play_bench(
        cfg, template, tmp_path / "w", steps=20, warmup=5, rates=("max",), directors=("off",)
    )
    assert r["digests_equal"] and r["arms"][0]["gate"]["applies"]


def test_alife_bench_play_runs_from_the_command_line(
    tmp_path: Path, short_play: None, capsys: pytest.CaptureFixture[str]
) -> None:
    from alife import cli
    from weather_helpers import SMALL3D

    argv = [
        "bench", "--play", "--preset", "64k3d", "--overlay", "storm", "--steps", "25", "--play-warmup", "5",
        "--play-rates", "max", "--play-spinup", "20",
        "--play-out", str(tmp_path / "arms"), "--json", str(tmp_path / "bench.json"),
        "--set", "director.enabled=true", "--set", "director.rule_every=20",
        *[x for o in SMALL3D for x in ("--set", o)],
    ]  # fmt: skip
    assert cli.main(argv) == 0
    report = json.loads(capsys.readouterr().out)
    assert json.loads((tmp_path / "bench.json").read_text()) == report
    play = report["play"]
    assert play["digests_equal"] and [a["director"] for a in play["arms"]] == ["off", "rules", "player"]
    assert play["workdir"] == str(tmp_path / "arms") and (tmp_path / "arms" / "template.safetensors").exists()
    assert report["config"]["n_max"] == 256, "the bench report carries the environment"
    assert "stages" not in report and "ms_per_step" not in report, "--play is a report of its own"


def test_alife_bench_play_refuses_what_it_cannot_run(tmp_path: Path, short_play: None) -> None:
    from alife import cli
    from weather_helpers import SMALL3D

    base = [
        "bench",
        "--play",
        "--preset",
        "64k3d",
        "--steps",
        "5",
        "--play-out",
        str(tmp_path / "a"),
        *[x for o in SMALL3D for x in ("--set", o)],
    ]
    assert cli.main([*base, "--play-rates", "99"]) == 2, "a rate outside [0.05, 16]"
    assert cli.main([*base, "--play-directors", "llm"]) == 2
    assert cli.main([*base, "--play-spinup", "10"]) == 2, "a spin-up needs the weather on"
    assert cli.main([*base, "--play-directors", "rules"]) == 2, "the rule tier needs director.enabled"


def test_alife_bench_play_runs_only_the_plain_arm_without_the_director_levers(
    tmp_path: Path, short_play: None, capsys: pytest.CaptureFixture[str]
) -> None:
    from alife import cli
    from weather_helpers import SMALL3D

    argv = [
        "bench", "--play", "--preset", "64k3d", "--steps", "4", "--play-warmup", "1", "--play-rates", "max",
        "--play-out", str(tmp_path / "arms"), *[x for o in SMALL3D for x in ("--set", o)],
    ]  # fmt: skip
    assert cli.main(argv) == 0
    play = json.loads(capsys.readouterr().out)["play"]
    assert [(a["director"], a["rate"]) for a in play["arms"]] == [("off", "max")], "no levers, no rule tier"
    assert play["digests_equal"] and play["arms"][0]["bytes_per_s"].get("MAP1") is None, (
        "a dry world sends no map"
    )


def test_play_bench_says_so_when_a_played_world_differs_from_the_plain_run(
    tmp_path: Path, short_play: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`digests_equal` is a measurement, not a constant: against a wrong plain digest it is false, and the
    report says so (`alife bench --play` exits 1)."""
    cfg, template = planet_template(tmp_path)
    monkeypatch.setattr(bench, "_plain_digest", lambda *a, **k: "not the digest")
    r = bench.run_play_bench(
        cfg, template, tmp_path / "w", steps=6, warmup=2, rates=("max",), directors=("off",)
    )
    assert r["digests_equal"] is False and r["arms"][0]["digest_equal"] is False
    assert "DIFFER from the plain run" in "\n".join(bench._format_play(r))


def test_play_bench_does_not_count_saving_as_boundary_work(
    tmp_path: Path, short_play: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The checkpoint is its own part: with a save at every boundary and a slow disk, `save` is the disk's and
    `boundary` is the runner's work without it (the gate reports saving apart from `late_steps`).

    No millisecond bound: the injected delay is long against the runner's own boundary work on any machine
    that runs the suite, so the claim is a ratio. A bench that counted the save inside `boundary` would have
    `boundary >= save` (a boundary's span holds the save), and the bound sits at three quarters of it
    (1.5 s of runner work per boundary, eight times what the slowest CI runner measured)."""
    from alife import play
    from alife.runio import RunDir

    delay_s = 2.0
    monkeypatch.setattr(play, "PLAY_CADENCE", {**PLAY_CADENCE, "checkpoint_every": 10})
    real = RunDir.save_checkpoint

    def slow(self: RunDir, *args: object, **kwargs: object) -> object:
        time.sleep(delay_s)
        return real(self, *args, **kwargs)

    monkeypatch.setattr(RunDir, "save_checkpoint", slow)
    cfg, template = planet_template(tmp_path)
    r = bench.run_play_bench(
        cfg, template, tmp_path / "w", steps=25, warmup=5, rates=("max",), directors=("off",)
    )
    c = r["arms"][0]["components_ms"]
    assert c["save"] >= 1e3 * delay_s, "the save part holds the disk's time"
    assert c["boundary"] < 0.75 * c["save"], (c["save"], c["boundary"])


def test_the_bench_pacer_never_waits_at_a_chapter_end() -> None:
    """A bench arm has no reader to press ``run``: its pacer starts with the chapter pause off (§34.2), so an
    arm that crosses a chapter end measures the loop instead of waiting."""
    import io

    from alife import play, stream

    pacer = bench._bench_pacer_class()(io.BytesIO(), stream.NoControl(), play.PlayOptions())
    assert pacer.autopause.chapter is False and pacer.autopause.flags == 0
    assert play.PlayPacer(io.BytesIO(), stream.NoControl(), play.PlayOptions()).autopause.chapter is True


# --- bench --play, G3b: the player arm, scripted powers, the chapter and report line (DESIGN §34.8) ------

CHAPTER = 30  # a multiple of the checkpoint cadence of `short_play`


def test_the_player_arm_is_the_plain_run_without_powers(tmp_path: Path, short_play: None) -> None:
    """The player mode decides nothing: its arm, with no power sent, is the plain run bit for bit."""
    cfg, template = planet_template(tmp_path)
    r = bench.run_play_bench(
        cfg, template, tmp_path / "w", steps=40, warmup=10, rates=("max",),
        directors=("off", "rules", "player"), chapter=CHAPTER,
    )  # fmt: skip
    assert [a["director"] for a in r["arms"]] == ["off", "rules", "player"] and r["digests_equal"]
    player = r["arms"][2]
    assert player["powers"] is None and player["expected"] == "plain" and player["digest_equal"]
    assert r["replay_digest"] == {} and r["powers"] is False
    # the chapter of 30 steps ends once in the 50 steps, and the engine of alerts was timed (a short world
    # has no
    # report job: a chapter is a whole number of snapshots, DESIGN §34.4)
    assert player["chapters"]["ends"] == 1 and player["chapters"]["length"] == CHAPTER
    assert player["chapters"]["report_ms"] == []
    assert player["alerts"]["calls"] > 0 and player["alerts"]["ms_total"] > 0
    json.dumps(r, allow_nan=False)


def test_scripted_powers_are_applied_and_the_arm_equals_the_replay_of_its_own_log(
    tmp_path: Path, short_play: None
) -> None:
    from alife.runio import DIRECTOR_LOG_FILE

    cfg, template = planet_template(tmp_path)
    r = bench.run_play_bench(
        cfg, template, tmp_path / "w", steps=40, warmup=10, rates=("max", "1"), directors=("off", "player"),
        chapter=CHAPTER, powers=True,
    )  # fmt: skip
    arms = {a["name"]: a for a in r["arms"]}
    assert set(arms) == {"off_max", "off_1", "player_max", "player_1"} and r["powers"] is True
    assert arms["off_max"]["powers"] is None and arms["off_max"]["expected"] == "plain"
    for name in ("player_max", "player_1"):
        arm = arms[name]
        powers = arm["powers"]
        # a Food half way between the end of the warm-up (10) and the chapter's end (30), and a Strike at the
        # chapter-end
        # step itself: the script and what the world said of it
        assert [item["step"] for item in powers["script"]] == [20, CHAPTER]
        assert powers["script"][0]["line"].startswith("power food id=9001 ")
        assert powers["script"][1]["line"].startswith("power strike id=9002 ")
        assert powers["applied"] == [9001, 9002] and powers["all_applied"] and powers["refused"] == []
        assert len(powers["latency_s"]) == 2 and all(
            0 < v < bench.PLAY_POWER_LIMIT_S for v in powers["latency_s"]
        )
        statuses = [(p["id"], p["status"]) for p in powers["plans"] if p["id"] == 9002]
        assert statuses[0] == (9002, "queued") and (9002, "applied") in statuses
        # both are in the arm's own log, as the player's, applied at the steps they were sent at
        entries = [
            json.loads(line) for line in (Path(arm["run"]) / DIRECTOR_LOG_FILE).read_text().splitlines()
        ]
        player = [e for e in entries if e.get("source") == "player"]
        assert [(e["step_applied"], e["action"]) for e in player][:2] == [
            (20, "regen_pulse"),
            (CHAPTER, "catastrophe"),
        ]
        # a world with the player's powers is no longer the plain run's: its digest is its log's, replayed
        assert (
            arm["expected"] == "replay" and arm["digest_equal"] and arm["digest"] == r["replay_digest"][name]
        )
        assert arm["digest"] != r["plain_digest"]["player"], (
            "the powers changed the trajectory (they are not presentation)"
        )
    assert r["digests_equal"]
    text = "\n".join(bench._format_play(r))
    assert "a run with powers: to its log replayed" in text and "powers: applied [9001, 9002]" in text
    assert "chapter, report, alerts:" in text and "click to applied" in text


def test_the_scripted_arm_needs_a_chapter_end_in_its_measured_part(tmp_path: Path, short_play: None) -> None:
    from alife.config import ConfigError

    cfg, template = planet_template(tmp_path)
    args = {"rates": ("max",), "directors": ("player",), "powers": True}
    with pytest.raises(ConfigError, match="chapter end"):  # 10 + 15 steps from 0: the chapter ends at 30
        bench.run_play_bench(cfg, template, tmp_path / "a", steps=15, warmup=10, chapter=CHAPTER, **args)
    with pytest.raises(ConfigError, match="chapter end"):  # the warm-up is past it
        bench.run_play_bench(cfg, template, tmp_path / "b", steps=40, warmup=35, chapter=CHAPTER, **args)
    with pytest.raises(
        ConfigError, match="chapter end"
    ):  # the warm-up ends at it: nothing of the chapter is measured
        bench.run_play_bench(cfg, template, tmp_path / "b2", steps=20, warmup=30, chapter=CHAPTER, **args)
    with pytest.raises(
        ConfigError, match="chapter end"
    ):  # the arm stops at it: no step to see the strike applied
        bench.run_play_bench(cfg, template, tmp_path / "b3", steps=20, warmup=10, chapter=CHAPTER, **args)
    with pytest.raises(ConfigError, match="player director"):
        bench.run_play_bench(
            cfg,
            template,
            tmp_path / "c",
            steps=40,
            warmup=10,
            rates=("max",),
            directors=("off",),
            powers=True,
        )
    flat = bench.power_script(cfg, start=1000, warmup=100, total=300, chapter=300)
    assert [step for step, _ in flat] == [(1100 + 1200) // 2, 1200], "the next end after the start, 1,200"


def test_the_player_mode_needs_the_levers_like_the_rule_tier(tmp_path: Path, short_play: None) -> None:
    from alife.config import ConfigError
    from alife.runio import write_state_file
    from weather_helpers import storm3d_config

    cfg = storm3d_config()
    path = tmp_path / "t.safetensors"
    write_state_file(path, init_state(cfg), cfg)
    with pytest.raises(ConfigError, match=r"director\.enabled"):
        bench.run_play_bench(
            cfg, path, tmp_path / "w", steps=5, warmup=1, rates=("max",), directors=("player",)
        )


def test_a_scripted_control_hands_a_line_over_when_the_world_reaches_its_step() -> None:
    control = bench.ScriptedControl(
        [(30, "power strike id=9002 x=1 y=1"), (20, "power food id=9001 mult=1.3"), (20, "rate 2")]
    )
    assert control.poll(0) == [], "the world is not there yet"
    control.now = 19
    assert control.poll(0) == []
    control.now = 20
    assert control.poll(0) == ["power food id=9001 mult=1.3", "rate 2"], "in order"
    control = bench.ScriptedControl(
        [(30, "power strike id=9002 x=1 y=1"), (20, "power food id=9001 mult=1.3")]
    )
    control.now = 25
    assert control.poll(0) == ["power food id=9001 mult=1.3"] and set(control.clicked) == {9001}
    assert control.poll(0) == [], "once"
    control.now = 99
    assert control.poll(0) == ["power strike id=9002 x=1 y=1"] and set(control.clicked) == {9001, 9002}
    assert control.clicked[9002] >= control.clicked[9001]
    control.close()


def test_the_gate_reads_the_power_latency_and_the_report_time() -> None:
    arm = {"rate": "1", "achieved_x": 0.99, "late_steps": 0, "step_ms": {"p99": 60.0}, "seconds": 310.0}
    scripted = {"latency_s": [0.07, 0.12], "all_applied": True}
    chapters = {"report_ms": [31.0]}
    gate = bench._play_gate({**arm, "powers": scripted, "chapters": chapters})
    assert gate["pass"] and gate["want"]["power_applied_p95_s"] == 0.5 and gate["want"]["report_ms"] == 500.0
    slow = bench._play_gate({**arm, "powers": {**scripted, "latency_s": [0.07, 0.9]}, "chapters": chapters})
    assert not slow["pass"] and not slow["checks"]["power_applied_p95_s"]
    assert not bench._play_gate({**arm, "powers": {**scripted, "all_applied": False}})["pass"]
    assert not bench._play_gate({**arm, "powers": {**scripted, "latency_s": []}})["pass"], (
        "no click was answered"
    )
    late = bench._play_gate({**arm, "powers": scripted, "chapters": {"report_ms": [31.0, 720.0]}})
    assert not late["pass"] and not late["checks"]["report_ms"]
    # an arm without powers or chapters is read as before
    assert bench._play_gate(arm)["want"] == bench.PLAY_GATE_1X
    assert "report_ms" not in bench._play_gate({**arm, "chapters": {"report_ms": []}})["want"]


def test_alife_bench_play_sends_scripted_powers_from_the_command_line(
    tmp_path: Path, short_play: None, capsys: pytest.CaptureFixture[str]
) -> None:
    from alife import cli
    from weather_helpers import SMALL3D

    argv = [
        "bench", "--play", "--preset", "64k3d", "--overlay", "storm", "--steps", "40", "--play-warmup", "10",
        "--play-rates", "max", "--play-spinup", "20", "--play-directors", "player", "--play-powers",
        "--play-chapter", str(CHAPTER), "--play-out", str(tmp_path / "arms"),
        "--json", str(tmp_path / "bench.json"),
        "--set", "director.enabled=true", "--set", "director.rule_every=20",
        *[x for o in SMALL3D for x in ("--set", o)],
    ]  # fmt: skip
    assert cli.main(argv) == 0
    play = json.loads(capsys.readouterr().out)["play"]
    assert [a["name"] for a in play["arms"]] == ["player_max"] and play["powers"] and play["digests_equal"]
    assert play["arms"][0]["powers"]["applied"] == [9001, 9002]
    assert cli.main([*argv, "--play-directors", "off"]) == 2, "scripted powers need the player director mode"
    assert cli.main([*argv[: argv.index("--play-chapter")], "--play-chapter", "7"]) == 2, (
        "no chapter end in the arm"
    )


def test_alife_bench_play_resumes_a_played_worlds_checkpoint(
    tmp_path: Path, short_play: None, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--play-resume` takes a state file (or a run's checkpoint) as it is: the arms start from its step."""
    from alife import cli

    _, template = planet_template(tmp_path)
    argv = [
        "bench", "--play", "--play-resume", str(template), "--steps", "40", "--play-warmup", "10",
        "--play-rates", "max", "--play-directors", "player", "--play-powers", "--play-chapter", str(CHAPTER),
        "--play-out", str(tmp_path / "arms"),
    ]  # fmt: skip
    assert cli.main(argv) == 0
    play = json.loads(capsys.readouterr().out)["play"]
    assert play["arms"][0]["powers"]["all_applied"] and play["digests_equal"]
    assert cli.main([*argv, "--play-from", "somewhere"]) == 2, "two ways to name the world"


def test_the_bench_pacer_reads_a_report_and_a_plan_from_the_records_it_writes() -> None:
    """The stopwatch parses the records it passes on: the first report that is not the backlog ends a
    chapter's timer, and an `applied` plan stamps its request (the click-to-applied latency is read from
    these)."""
    import io
    import struct
    import threading

    from alife import play, stream, streamrec

    pacer = bench._bench_pacer_class()(io.BytesIO(), stream.NoControl(), play.PlayOptions())
    pacer.timing = bench._Timing(0)

    def rprt(flags: int) -> bytes:
        return streamrec.record("RPRT", 6000, 0, flags, b"\x00" * 16)

    pacer._chapter_t0 = time.perf_counter() - 0.25
    pacer._write("RPRT", rprt(1 | 2))  # the backlog of a world being opened: not this chapter's
    assert pacer.report_ms == [] and pacer._chapter_t0 is not None
    loop_write = pacer._write_s
    worker = threading.Thread(target=pacer._write, args=("RPRT", rprt(2)))  # the report job's thread
    worker.start()
    worker.join()
    assert pacer._write_s == loop_write, "only the loop's writes count as the loop's write time"
    assert len(pacer.report_ms) == 1 and 250 <= pacer.report_ms[0] < 5000 and pacer._chapter_t0 is None
    pacer._write("RPRT", rprt(2))
    assert len(pacer.report_ms) == 1, "no chapter is waiting for one"

    def plan(request: int, status: str) -> bytes:
        js = streamrec.json_bytes({"id": request, "status": status})
        return streamrec.record("PLAN", 100, 0, 0, struct.pack("<I", len(js)) + streamrec.pad4(js))

    pacer._write("PLAN", plan(9001, "queued"))
    pacer._write("PLAN", plan(9001, "applied"))
    first = pacer.applied_at[9001]
    for request, status in ((9001, "applied"), (9002, "refused")):
        pacer._write("PLAN", plan(request, status))
    assert pacer.applied_at[9001] == first, "a second `applied` for the request does not move its stamp"
    assert pacer.plans == [(9001, "queued"), (9001, "applied"), (9001, "applied"), (9002, "refused")]
    assert set(pacer.applied_at) == {9001}, "refused is not applied; the first `applied` stands"


def test_a_refused_power_is_not_applied_and_fails_the_gate(
    tmp_path: Path, short_play: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`all_applied` is what the arm saw, not a constant: a second Food while the first runs is refused."""
    lines = [
        (20, "power food id=9001 mult=1.3 steps=3000"),
        (CHAPTER, "power food id=9002 mult=1.3 steps=3000"),
    ]
    monkeypatch.setattr(bench, "power_script", lambda *a, **k: lines)
    cfg, template = planet_template(tmp_path)
    r = bench.run_play_bench(
        cfg, template, tmp_path / "w", steps=40, warmup=10, rates=("max",), directors=("player",),
        chapter=CHAPTER, powers=True,
    )  # fmt: skip
    arm = r["arms"][0]
    powers = arm["powers"]
    assert powers["applied"] == [9001] and powers["refused"] == [9002] and powers["all_applied"] is False
    assert len(powers["latency_s"]) == 1
    assert arm["gate"]["checks"]["power_applied_p95_s"] is False, (
        "a power that was not applied fails the gate"
    )
    assert arm["digest_equal"], (
        "(a refused request changes nothing: the arm still equals the replay of its log)"
    )
