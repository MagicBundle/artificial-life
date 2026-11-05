"""The second diel pilot's rule, lineage fitness (DESIGN 49.14), on synthetic metrics.jsonl files."""

import json
import math

import numpy as np
import pytest

from alife.analysis import diel_pilot2 as d2

PERIOD = 100


def pops(o: np.ndarray, c: np.ndarray) -> list[list[int]]:
    """Per day, the six diel_orc_pop entries (O1 O2 O3 C1 C2 C3) from (days, 3) arrays."""
    return [[round(float(x)) for x in (*o[d], *c[d])] for d in range(o.shape[0])]


def make_run(path, daily: list[list[int]], start: int = 0, schedule: int = 2):
    """A run with 2 identical rows per day holding the given populations."""
    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps({"diel": {"period": PERIOD, "oracle_schedule": schedule}}))
    if start:
        (path / "meta.json").write_text(json.dumps({"start_step": start}))
    rows = [
        json.dumps({"step": start + d * PERIOD + k * 50, "diel_orc_pop": p})
        for d, p in enumerate(daily)
        for k in range(2)
    ]
    (path / "metrics.jsonl").write_text("\n".join(rows) + "\n")
    return path


def growth_runs(tmp_path, rates, noise=0.02, days=40, o0=200.0, c0=800.0):
    out = []
    for i in range(3):
        rng = np.random.default_rng(i)
        t = np.arange(days)[:, None]
        o = o0 * np.exp(np.array(rates) * t + rng.normal(0, noise, (days, 3)))
        c = c0 * np.exp(rng.normal(0, noise, (days, 3)))
        out.append(make_run(tmp_path / f"r{i}", pops(o, c)))
    return out


def test_selection_for_the_oracle_is_go(tmp_path):
    r = d2.analyse(growth_runs(tmp_path, (0.05, 0.05, 0.05)), boot=300)
    assert r.decision == "GO" and r.reason == ""
    assert r.w["pooled"] == pytest.approx(0.05, abs=0.005) and r.lo["pooled"] >= d2.GO_LOWER


def test_selection_against_the_oracle_is_stop(tmp_path):
    r = d2.analyse(growth_runs(tmp_path, (-0.05, -0.05, -0.05)), boot=300)
    assert r.decision == "STOP"
    assert all(r.hi[k] < d2.STOP_STRATUM for k in "123") and r.hi["pooled"] < d2.STOP_POOLED


def test_no_selection_is_small(tmp_path):
    r = d2.analyse(growth_runs(tmp_path, (0.0, 0.0, 0.0), noise=0.05), boot=300)
    assert r.decision == "SMALL"
    assert r.lo["pooled"] < 0 < r.hi["pooled"]


def test_one_stratum_against_others_neutral_is_small(tmp_path):
    r = d2.analyse(growth_runs(tmp_path, (-0.05, 0.0, 0.0), noise=0.05), boot=300)
    assert r.decision == "SMALL" and r.w["1"] < -0.04


def test_censoring_clamps_the_absorbing_step_and_stops():
    """O of stratum 1 falls from 100 to 0 on day 3: the increments of days 0-1 and 1-2 count, day 2-3 counts
    with O raised to N_MIN, and nothing after. Stratum 2: both groups die together on day 2 (only day 0-1
    counts). Stratum 3: the comparison is absorbed (a positive clamped step)."""
    o = np.array([[100, 50, 100], [80, 40, 100], [60, 5, 100], [0, 0, 100], [0, 0, 100], [0, 0, 100]], float)
    c = np.array([[100, 50, 100], [100, 40, 50], [100, 5, 20], [100, 0, 4], [120, 0, 0], [90, 0, 0]], float)
    fit = d2.fit_rows(
        [{"step": d * PERIOD, "diel_orc_pop": p} for d, p in enumerate(pops(o, c))], PERIOD, 0, (0, 6)
    )
    inc = fit.inc
    assert inc.shape == (5, 3)
    np.testing.assert_allclose(
        inc[:3, 0], [math.log(0.8), math.log(60 / 80), math.log(10 / 100) - math.log(0.6)]
    )
    assert np.isnan(inc[3:, 0]).all()
    assert inc[0, 1] == pytest.approx(0.0) and np.isnan(inc[1:, 1]).all()  # 40/40 then both below 10
    np.testing.assert_allclose(
        inc[:3, 2], [math.log(2), math.log(100 / 20) - math.log(2), math.log(100 / 10) - math.log(5)]
    )
    assert np.isnan(inc[3:, 2]).all()
    assert fit.extinct == (3, 3, None) and fit.founders == (100, 50, 100)
    w = d2._stats([fit], None)
    assert w[0] == pytest.approx(np.mean(inc[:3, 0])) and w[1] == pytest.approx(0.0)


def test_extinction_reads_strongly_negative(tmp_path):
    """The first pilot's pattern: the oracle lineages collapse within days while the comparison holds."""
    days = 40
    o = np.zeros((days, 3))
    o[:5] = [[500, 600, 300], [300, 60, 20], [120, 40, 5], [40, 8, 0], [5, 0, 0]]
    c = np.full((days, 3), 2000.0)
    ps = [make_run(tmp_path / f"r{i}", pops(o, c)) for i in range(3)]
    r = d2.analyse(ps, boot=300)
    assert r.decision == "STOP" and r.w["pooled"] < -0.5


def test_too_few_founders_is_invalid(tmp_path):
    o = np.full((40, 3), 200.0)
    o[0, 2] = 30
    c = np.full((40, 3), 800.0)
    ps = [make_run(tmp_path / f"r{i}", pops(o, c)) for i in range(3)]
    r = d2.analyse(ps, boot=50)
    assert r.decision == "INVALID" and "fewer than 50 oracle founders" in r.reason and "stratum 3" in r.reason


def test_a_stratum_with_no_counted_increment_is_invalid(tmp_path):
    o = np.full((40, 3), 200.0)
    c = np.full((40, 3), 800.0)
    c[:, 1] = 5  # the comparison of the mid stratum is below N_MIN throughout
    ps = [make_run(tmp_path / f"r{i}", pops(o, c)) for i in range(3)]
    r = d2.analyse(ps, boot=50)
    assert r.decision == "INVALID" and "no counted increment in stratum 2" in r.reason


def test_window_and_fork_start(tmp_path):
    o = np.full((45, 3), 200.0)
    o[41:] = 400  # outside the window [0, 40): no effect
    c = np.full((45, 3), 800.0)
    p = make_run(tmp_path / "a", pops(o, c), start=40_000)
    fit = d2.load_run(p)
    assert fit.inc.shape == (39, 3) and np.allclose(fit.inc, 0) and fit.schedule == 2


def test_bootstrap_reproducible(tmp_path):
    ps = growth_runs(tmp_path, (0.01, 0.0, -0.01), noise=0.05)
    a, b = d2.analyse(ps, boot=200, seed=3), d2.analyse(ps, boot=200, seed=3)
    assert a.lo == b.lo and a.hi == b.hi


def test_rule_hash_pinned():
    assert d2.rule_hash() == "0f4b7da91a64ab4f135ee21fbd7acbc70537671f1ddb24244d24653ae3d7fd9e"
    assert d2.rule_hash(boot=500) != d2.rule_hash()


def test_cli(tmp_path, capsys):
    ps = growth_runs(tmp_path, (0.05, 0.05, 0.05))
    out = tmp_path / "o.json"
    assert d2.main([*map(str, ps), "--boot", "50", "--json", str(out)]) == 0
    text = capsys.readouterr().out
    assert "decision: GO" in text and "lineage fitness" in text
    got = json.loads(out.read_text())
    assert got["decision"] == "GO" and got["schedules"] == {"r0": 2, "r1": 2, "r2": 2}
