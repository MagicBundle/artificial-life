"""The emit/msg recomputation and the per-agent table of ``alife.analysis.signals``."""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from alife import layout
from alife.analysis import signals
from alife.config import Config
from alife.state import CounterIdx, State, init_state, to_numpy
from alife.step import make_step

WARMUP = 15


@pytest.fixture(scope="module")
def warm(tiny_cfg: Config) -> State:
    """A tiny world after a few steps, so messages, bonds and births are under way."""
    s = init_state(tiny_cfg)
    step = make_step(tiny_cfg)
    for _ in range(WARMUP):
        s, _ = step(s)
        mx.eval(s)
    return s


def test_decode_reproduces_msg_and_emitters(tiny_cfg: Config, warm: State) -> None:
    """One step stores exactly the recomputed ``msg`` for its survivors and counts the same emitters."""
    dec = signals.decode(warm, tiny_cfg)
    s1, out = make_step(tiny_cfg, compile=False)(warm)
    q = signals.emit_quanta(dec.act.emit)
    mx.eval(dec, s1, out, q)
    survivor = to_numpy(s1.alive) & (to_numpy(s1.uid) == to_numpy(warm.uid))
    assert survivor.sum() > 50
    msg = to_numpy(dec.act.msg)
    assert np.array_equal(to_numpy(s1.msg)[survivor], msg[survivor])
    assert np.abs(msg[survivor]).max() > 0
    emitters = int(((to_numpy(q) > 0).any(axis=1) & to_numpy(warm.alive)).sum())
    assert emitters == int(to_numpy(out.counters)[CounterIdx.EMITTERS]) > 0


def test_decode_has_no_side_effects(tiny_cfg: Config, warm: State) -> None:
    before = {name: to_numpy(getattr(warm, name)) for name in ("key", "step", "W1", "msg")}
    mx.eval(signals.decode(warm, tiny_cfg))
    for name, v in before.items():
        assert np.array_equal(to_numpy(getattr(warm, name)), v), name


def test_agent_table(tiny_cfg: Config, warm: State) -> None:
    t = signals.agent_table(warm, tiny_cfg)
    n = int(to_numpy(warm.alive).sum())
    assert {*signals.SIGNALS, *signals.TARGETS, "row", "patch", "clade"} <= set(t)
    assert all(v.shape[0] == n for v in t.values())
    k = tiny_cfg.sensing.k_neighbors
    np.testing.assert_array_equal(t["kin"] + t["nonkin"], t["crowd"])
    assert np.all(t["crowd"] <= k) and np.all(t["close_kin"] <= t["crowd"])
    for c in range(layout.NCH):
        assert t[f"emit{c}"].min() >= 0 and t[f"emit{c}"].max() <= 255
    survived = np.isfinite(t["thrust_next"])
    assert survived.mean() > 0.9
    np.testing.assert_array_equal(np.isfinite(t["attack_next"]), survived)
    assert set(np.unique(t["attack_next"][survived])) <= {0.0, 1.0}
    res = to_numpy(warm.resource).reshape(-1)
    assert np.isin(t["resource"], res).all()
    assert t["patch"].max() < to_numpy(warm.patch_pos).shape[0]


def test_nearest_patch_wraps() -> None:
    patches = np.array([[1.0, 1.0], [50.0, 50.0]])
    pos = np.array([[99.0, 99.0], [52.0, 49.0]])
    np.testing.assert_array_equal(signals.nearest_patch(pos, patches, 100.0), [0, 1])


def test_jobs_and_run_job(tiny_cfg: Config, warm: State) -> None:
    t = signals.agent_table(warm, tiny_cfg)
    jobs = signals.jobs_for(7, t, n_sample=200, seed=0)
    n_emit = sum(s.startswith("emit") for s in signals.SIGNALS)
    per_signal = 3 * len(signals.TARGETS) + len(signals.KIN_TARGETS)  # all, patch, clade (+ crowd)
    assert len(jobs) == len(signals.SIGNALS) * per_signal + n_emit * 2 * len(signals.TARGETS)
    assert {j.variant for j in jobs} == set(signals.VARIANTS)
    rec = signals.run_job(jobs[0], n_perm=3, k=5, seed=0, min_stratum=10)
    assert rec["step"] == 7 and rec["variant"] == "all" and "mi_mm" in rec
    const = signals.Job(7, "emit0", "energy", "all", np.zeros(100), np.arange(100.0), None)
    assert math.isnan(signals.run_job(const, n_perm=3, k=5, seed=0, min_stratum=10)["mi"])


def _fake_report(step: int, z: float) -> dict:
    results = [
        {
            "step": step,
            "signal": sig,
            "target": tgt,
            "variant": v,
            "n": 10,
            "mi": 0.1,
            "null_mean": 0.01,
            "z": z,
        }
        for sig in signals.SIGNALS
        for tgt in signals.TARGETS
        for v in signals.VARIANTS
        if (v not in ("emits", "emitters") or sig.startswith("emit"))
        and (v != "crowd" or tgt in signals.KIN_TARGETS)
    ]
    return {"run": "r", "params": {"k": 10}, "summaries": [{"step": step}], "results": results}


def test_merge_and_render() -> None:
    rep = signals.merge_reports([_fake_report(500, 1.0), _fake_report(100, 9.0)])
    assert [s["step"] for s in rep["summaries"]] == [100, 500]
    md = signals.markdown_tables(rep, z_sig=4.0)
    assert md.index("Step 100") < md.index("Step 500") and "**9.0**" in md and "**1.0**" not in md
    trend = signals.trend_table(rep, "all", z_sig=4.0)
    assert "| emit0 | resource | 0.090* | 0.090 |" in trend
    with pytest.raises(ValueError):
        signals.merge_reports([_fake_report(100, 1.0), _fake_report(100, 1.0)])
    other = {**_fake_report(200, 1.0), "params": {"k": 5}}
    with pytest.raises(ValueError):
        signals.merge_reports([_fake_report(100, 1.0), other])
