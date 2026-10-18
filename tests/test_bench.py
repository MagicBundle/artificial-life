"""WP6 bench tests (§15.4, §21, §22.7): report shape, staged wiring, donation health, determinism."""

from __future__ import annotations

import json
from dataclasses import replace

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from alife import bench
from alife.bench import STAGES, format_report, run_bench
from alife.config import BRAIN_IMPLS, WEIGHT_DTYPES, Config, apply_overrides
from alife.state import CounterIdx, LedgerIdx, init_state, n_counters, state_digest, to_numpy
from alife.step import make_step

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


def test_observers_block(tiny_cfg: Config) -> None:
    """``observe=True`` adds the observers' cost against the plain step, as blocks in the order C O O C."""
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
        "trajectory_match",
    }
    assert ob["observed_ms_per_step"] > 0 and ob["control_ms_per_step"] > 0 and ob["observed_steps_per_s"] > 0
    assert ob["overhead_ms_median"] == pytest.approx(ob["observed_ms_median"] - ob["control_ms_median"])
    assert ob["overhead_ms_mean"] == pytest.approx(ob["observed_ms_per_step"] - ob["control_ms_per_step"])
    assert ob["observed_steps_per_s"] == pytest.approx(1e3 / ob["observed_ms_per_step"])
    assert ob["block_pairs"] == 2 and ob["steps_per_block"] == 1  # max(steps // 2, 1)
    assert ob["warmup"] == 8  # the first evaluation (rules.every) is in the untimed warm-up, never timed
    assert ob["trajectory_match"] is True
    assert "observers (2 block pairs of 1 steps)" in format_report(
        r
    ) and "trajectory identical" in format_report(r)
    assert run_bench(tiny_cfg, steps=3, warmup=20, observe=True)["observe"]["warmup"] == 20


def test_observers_block_runs_the_observer_in_every_step(
    tiny_cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The observed arm folds every step (warm-up included) of each of its blocks into a fresh observer."""
    from alife.game.observer import Observer

    seen: list[int] = []
    real = Observer.update

    def spy(self: Observer, state, acts, t: int) -> None:
        seen.append(t)
        real(self, state, acts, t)

    monkeypatch.setattr(Observer, "update", spy)
    ob = bench._bench_observe(tiny_cfg, 2, 1)  # warm-up 8 (rules.every), 1 timed step per block
    assert ob["warmup"] == 8
    assert seen == [*range(1, 10), *range(1, 10)]  # two observed blocks, each from a fresh observer


def test_observers_blocks_run_control_observed_observed_control(
    tiny_cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """C O O C, not C O C O: with a drift linear in time the observed blocks of the second order sit on
    average one block later than the control blocks, which biases the overhead by one block's drift."""
    from alife.game.observer import Observer

    order: list[str] = []
    real_init, real_fresh = bench.init_state, Observer.fresh

    def init_spy(cfg: Config):
        order.append("control")  # every block starts from init_state; an observed one renames it below
        return real_init(cfg)

    def fresh_spy(state, cfg: Config, **kw):
        order[-1] = "observed"
        return real_fresh(state, cfg, **kw)

    monkeypatch.setattr(bench, "init_state", init_spy)
    monkeypatch.setattr(Observer, "fresh", staticmethod(fresh_spy))
    ob = bench._bench_observe(tiny_cfg, 2, 1)
    assert order == ["control", "observed", "observed", "control"] and ob["block_pairs"] == 2


def test_observers_block_sees_a_changed_trajectory(tiny_cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    """If observing ever changed the state, the report says so."""
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
