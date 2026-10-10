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
from alife.config import BRAIN_IMPLS, WEIGHT_DTYPES, Config
from alife.state import CounterIdx, init_state, state_digest, to_numpy
from alife.step import make_step


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
    assert not {"stages", "determinism", "brain_variants"} & r.keys()
    assert "stages" not in format_report(r)
    with pytest.raises(ValueError, match="steps"):
        run_bench(micro_cfg, steps=0)
    with pytest.raises(ValueError, match="warmup"):
        run_bench(micro_cfg, steps=1, warmup=-1)


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
