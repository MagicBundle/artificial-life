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
