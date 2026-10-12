"""The director evaluation metrics (``alife.analysis.evaluation``, docs/EVALUATION.md)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from alife import runner
from alife.analysis import evaluation as ev
from alife.config import Config, apply_overrides
from alife.director import Director
from alife.runio import RunDir
from alife.state import init_state
from test_analysis_activity import _alive_checkpoint, _hand_run

EVERY, END = 100, 1200


@pytest.fixture(scope="module")
def live_run(tiny_cfg: Config, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A real live director (Gardener, rule tier only) on the tiny world: its start-up boom and bust makes the
    extinction rescue fire, so the log has entries with vitals."""
    cfg = apply_overrides(
        tiny_cfg,
        [
            "director.enabled=true",
            f"director.rule_every={EVERY}",
            "run.director=live",
            "run.metrics_every=10",
            "run.heavy_metrics_every=100",
            "run.checkpoint_every=600",
            "run.diversity_sample=64",
        ],
    )
    rd = RunDir.create(tmp_path_factory.mktemp("eval") / "live", cfg)
    rd.update_meta(start_step=0)
    res = runner.run(rd, cfg, init_state(cfg), END, quiet=True, director=Director(rd, cfg, "live", 0))
    assert res.status == "done"
    return rd.path


def test_recomputed_vitals_equal_the_live_directors(live_run: Path) -> None:
    """The offline fold of metrics.jsonl gives exactly the novelty, coverage, fall and shift the live director
    logged with each of its entries."""
    bds, archive = ev.boundaries(RunDir.open(live_run).metrics_rows(), 0, EVERY)
    assert [b["step"] for b in bds] == list(range(EVERY, END + 1, EVERY))
    entries = ev.log_entries(live_run)
    assert entries, "the tiny world's start-up bust should trigger a rescue"
    check = ev.check_against_log(bds, entries)
    assert check["entries_checked"] == sum(1 for e in entries if e.get("vitals")) >= 1
    assert check["mismatched"] == 0, check["first_mismatch"]
    logged = json.loads((live_run / "director_archive.json").read_text())
    assert archive.coverage == logged["coverage"] and len(archive.entries) == len(logged["entries"])


def test_check_against_log_reports_a_mismatch() -> None:
    bds = [{"step": 1000, "novelty": 0.05, "coverage": 3, "fall": 0.1, "b_shift": None}]
    good = {"vitals_step": 1000, "vitals": {"novelty": 0.05, "coverage": 3, "fall": 0.1, "b_shift": None}}
    bad = {"vitals_step": 1000, "vitals": {"novelty": 0.05, "coverage": 4, "fall": 0.1, "b_shift": None}}
    assert ev.check_against_log(bds, [good])["mismatched"] == 0
    r = ev.check_against_log(bds, [good, bad])
    assert r["entries_checked"] == 2 and r["mismatched"] == 1 and r["first_mismatch"]["key"] == "coverage"


def test_persistent_novelty_windows() -> None:
    t = np.arange(1000, 400_001, 1000)
    nov = np.where(t > 150_000, 0.05, 0.01)  # a lasting rise after step 150k
    ctrl = (t, np.full(t.size, 0.02))
    rows = ev.persistent_novelty(t, nov, [100_000, 200_000, 300_000], ctrl)
    # 300k is dropped: its after-window (400k, 450k] runs past the end
    assert [r["step"] for r in rows] == [100_000, 200_000]
    assert rows[0]["before"] == pytest.approx(0.01) and rows[0]["after"] == pytest.approx(0.05)
    assert rows[1]["delta"] == pytest.approx(0.0) and rows[1]["vs_control"] == pytest.approx(0.03)
    s = ev._pn_summary(rows, "vs_control")
    assert s == {"anchors": 2, "mean": pytest.approx(0.03), "frac_above": 1.0}
    assert ev._pn_summary([], "delta")["anchors"] == 0


def test_excursion_is_zero_on_the_reference_itself() -> None:
    rng = np.random.default_rng(0)
    ref = rng.random((50, 10)).tolist()
    bds = [{"b": b} for b in ref[:5]] + [{"b": None}, {"b": [2.0] * 10}]
    e = ev.excursion(bds, ref, k=1)
    np.testing.assert_allclose(e[:5], 0.0)
    assert math.isnan(e[5]) and e[6] > 1.0


ENTRIES = [
    {"step_applied": 1000, "source": "rule", "rule": "extinction_rescue", "action": "regen_pulse"},
    {"step_applied": 1000, "source": "rule", "rule": "extinction_rescue", "action": "immigration_pulse"},
    {"step_applied": 2000, "source": "llm", "action": "catastrophe"},
    {"step_applied": 2000, "source": "llm", "action": "drift_change"},
    {"step_applied": 3000, "source": "rule", "rule": "stagnation_response", "action": "drift_change"},
    {"step_applied": 21000, "source": "revert", "action": "regen_pulse"},
]


def test_intervention_stats() -> None:
    s = ev.intervention_stats(ENTRIES, 200_000)
    assert s["all"]["count"] == 5 and s["all"]["steps"] == 3 and s["reverts"] == 1
    assert s["all"]["cost"] == 1 + 1 + 3 + 1 + 1 and s["all"]["cost_per_100k"] == pytest.approx(3.5)
    assert s["discretionary"]["by_action"] == {"catastrophe": 1, "drift_change": 2}
    assert s["discretionary"]["entropy_bits"] == pytest.approx(
        -(1 / 3) * math.log2(1 / 3) - (2 / 3) * math.log2(2 / 3)
    )
    assert s["safety"]["count"] == 2 and s["rescues"] == 2 and s["rescue_steps"] == 1
    assert s["by_source"] == {"llm": 2, "rule": 3}
    assert ev.anchor_steps(ENTRIES) == [1000, 2000, 3000]
    assert ev.anchor_steps(ENTRIES, discretionary_only=True) == [2000, 3000]


def _call(step: int, outcome: str, accepted: int = 0, clamps: int = 0, rejected: int = 0, lat: float = 3.0):
    return {
        "kind": "llm_call",
        "step": step,
        "outcome": outcome,
        "latency_s": lat,
        "accepted": [{"action": "drift_change", "params": {"mult": 2.0}}] * accepted,
        "clamps": [{"param": "mult"}] * clamps,
        "rejected": [{"action": "catastrophe", "why": "budget 6/6 used"}] * rejected,
        "fallback": "rule" if outcome in ("failed", "invalid") else None,
        "journal": f"journal at {step}",
        "usage": {"prompt_tokens": 1500},
    }


def test_llm_stats() -> None:
    journal = [
        {"step": 1, "text": "a rule one-liner"},
        _call(25_000, "ok", accepted=2, clamps=1, lat=2.0),
        _call(50_000, "none", lat=4.0),
        _call(75_000, "rejected", rejected=1, lat=3.0),
        _call(100_000, "invalid", lat=10.0),
    ]
    s = ev.llm_stats(journal)
    assert s["calls"] == 4 and s["valid"] == 3 and s["validity_rate"] == pytest.approx(0.75)
    assert s["clamp_rate"] == pytest.approx(0.5) and s["calls_with_clamp"] == 1
    assert s["rejection_reasons"] == {"budget 6/6 used": 1} and s["fallbacks"] == 1
    assert s["latency_s"]["median"] == pytest.approx(3.5) and s["latency_s"]["max"] == 10.0
    assert ev.llm_stats([{"step": 1, "text": "x"}]) is None


def test_journal_sample_is_seeded_and_in_step_order() -> None:
    journal = [_call(25_000 * i, "ok", accepted=1) for i in range(1, 31)]
    a = ev.journal_sample(journal, 10, seed=3)
    assert a == ev.journal_sample(journal, 10, seed=3) and len(a) == 10
    assert [x["step"] for x in a] == sorted(x["step"] for x in a)
    assert a[0]["accepted"] == ["drift_change(mult=2.0)"]
    assert len(ev.journal_sample(journal[:4], 10)) == 4


def test_lineage_stats_of_a_fork(tmp_path: Path) -> None:
    """The hand tree as a fork at step 10 (test_analysis_activity): founders 0–3; at the end 2, 3 and 4 (a
    child of 2) are alive, so two founder lineages survive."""
    uid_of = [100, 101, 102, 103, 200, 201]
    rd = _hand_run(tmp_path / "fork", uid_of=uid_of, start=10, cuts=(10, 20, 30))
    _alive_checkpoint(rd, 10, [100, 101, 102, 103])
    s = ev.lineage_stats(rd.path, 10, tail=18, head=5)
    assert s["founders"] == 4 and s["alive_end"] == 3 and s["founder_lineages_alive"] == 2
    assert s["immigrants"] == 0 and s["immigrant_descended_frac_end"] == 0.0
    assert s["generation_head"] == 1.0 and s["generations_gained"] == 0.0 and s["coverage_ok"]


def test_arm_parse() -> None:
    a = ev.Arm.parse("1:B:runs/eval_s1_B")
    assert (a.seed, a.name, a.run, a.start) == (1, "B", Path("runs/eval_s1_B"), None)
    assert ev.Arm.parse("0:A:runs/64k_s0:120000").start == 120_000
    with pytest.raises(ValueError):
        ev.Arm.parse("0:A")


def test_activity_classes_of_a_fork_report(tmp_path: Path) -> None:
    from alife.analysis import report

    rd = _hand_run(tmp_path / "fork", uid_of=[100, 101, 102, 103, 200, 201], start=10, cuts=(10, 20, 30))
    _alive_checkpoint(rd, 10, [100, 101, 102, 103])
    st = report.Settings(grid_every=5, marker_rates=(0.5,), cohort_horizons=(10,), shadows=1, windows=())
    report.run_report(rd.path, rd.path / ev.ACTIVITY_DIR, st, plots=False, log=lambda m: None)
    a = ev.activity_classes(rd.path / ev.ACTIVITY_DIR / "summary.json", ("marker_0.5",))
    cell = a["spaces"]["marker_0.5"][report.window_label(10, 30)]
    assert a["t0"] == 10 and cell["class"] in report.CLASSES and set(cell["sweep"]) == set(report.CLASSES)
    assert ev.activity_classes(tmp_path / "missing.json") is None


def test_markdown_table_orders_arms_and_marks_missing_values() -> None:
    summary = {
        "arms": [
            {"arm": "C", "seed": 1, "coverage": {"final": 80}},
            {"arm": "A", "seed": 0, "coverage": {"final": 23}, "novelty": {"mean": 0.02}},
            {"arm": "C", "seed": 0, "coverage": {"final": 87}},
        ]
    }
    cols = (("coverage.final", "coverage", "{:.0f}"), ("novelty.mean", "novelty", "{:.3f}"))
    t = ev.markdown_table(summary, cols, order=("A", "B", "C")).splitlines()
    assert t[0] == "| arm | seed | coverage | novelty |"
    assert t[2:] == ["| A | 0 | 23 | 0.020 |", "| C | 0 | 87 | – |", "| C | 1 | 80 | – |"]
    assert ev.dig(summary["arms"][0], "coverage.final") == 80 and ev.dig({}, "a.b") is None
