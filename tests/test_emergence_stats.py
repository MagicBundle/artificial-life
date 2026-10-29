"""Statistics of the emergence suite (DESIGN §44.8): known answers, coverage, size and power simulations."""

from __future__ import annotations

import math

import numpy as np
import pytest

from alife.analysis import w2
from alife.analysis.emergence import compare, prereg, stats, synth
from alife.analysis.emergence.panel import PanelError


def test_holm_and_bh_match_hand_computation() -> None:
    p = [0.01, 0.04, 0.03, 0.005]
    assert stats.holm(p) == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert stats.bh(p) == pytest.approx([0.02, 0.04, 0.04, 0.02])
    nan = stats.holm([0.01, float("nan"), 0.02])
    assert nan[0] == pytest.approx(0.02) and math.isnan(nan[1]) and nan[2] == pytest.approx(0.02)
    assert stats.holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])


def test_stouffer_matches_the_closed_form() -> None:
    z = 1.959963984540054  # P(Z > z) = 0.025
    want = 0.5 * math.erfc(3 * z / math.sqrt(3) / math.sqrt(2))
    assert stats.stouffer([0.025, 0.025, 0.025]) == pytest.approx(want, rel=1e-9)
    assert stats.stouffer([0.5, 0.5, 0.5]) == pytest.approx(0.5)
    assert stats.stouffer([0.01, None, 0.02]) is None
    # one very small p is not cancelled by two large ones, but is diluted
    assert stats.stouffer([1e-6, 0.5, 0.5]) > 1e-6


def test_distribution_functions_against_tables() -> None:
    assert stats.norm_isf(0.025) == pytest.approx(1.959963984540054, abs=1e-9)
    assert stats.norm_sf(stats.norm_isf(1e-9)) == pytest.approx(1e-9, rel=1e-6)
    assert stats.t_sf(1.0, 1) == pytest.approx(0.25, abs=1e-12)  # Cauchy
    assert stats.t_sf(1.0, 2) == pytest.approx(0.5 * (1 - 1 / math.sqrt(3)), abs=1e-12)
    for df in (3, 7, 15, 30):  # the project's own 97.5% table
        assert stats.t_sf(w2.t_crit(df), df) == pytest.approx(0.025, abs=6e-4)
    assert stats.t_sf(-2.0, 7) == pytest.approx(1 - stats.t_sf(2.0, 7))


def test_paired_test_reports_a_known_contrast() -> None:
    a = np.array([2.1, 2.3, 1.9, 1.8, 2.4, 2.0, 2.2, 2.1])  # widening off: known-value check
    b = np.zeros(8)
    r = stats.paired_test(a, b, widen=False)
    d = a - b
    assert r["mean"] == pytest.approx(d.mean())
    assert r["d_z"] == pytest.approx(d.mean() / d.std(ddof=1))
    assert r["t"] == pytest.approx(d.mean() / (d.std(ddof=1) / math.sqrt(8)))
    assert r["p"] < 1e-6 and r["lo"] > 0 and not r["widened"]
    flip = stats.paired_test(a, b, direction=-1, widen=False)
    assert flip["mean"] == pytest.approx(-d.mean()) and flip["p"] > 0.999999


def test_t_interval_coverage_on_ar1_block_means() -> None:
    """Eight block means of a step-level AR(1) series (rho 0.3, 25 steps per block, as the recorded blocks are
    long against the correlation time): the 95% interval covers at least 93%. At the block level the same rho
    would cover about 87% at 8 blocks, which is why the widening rule exists (intervals are indicative)."""
    rng = np.random.default_rng(3)
    rho, per, n, reps, hit = 0.3, 25, 8, 3000, 0
    s = math.sqrt(1 - rho * rho)
    for _ in range(reps):
        e = rng.standard_normal(n * per + 50)
        x = np.zeros(n * per + 50)
        for i in range(1, len(x)):
            x[i] = rho * x[i - 1] + s * e[i]
        iv = stats.t_interval(x[50:].reshape(n, per).mean(axis=1))
        hit += iv["lo"] <= 0.0 <= iv["hi"]
    assert hit / reps >= 0.93


def test_autocorrelation_widens_the_blocks_and_flags_it() -> None:
    ramp = np.linspace(0, 1, 8) + 0.01 * np.arange(8) % 2  # strong positive lag-1
    r = stats.paired_test(ramp, np.zeros(8))
    assert r["widened"] and r["n"] == 4 and abs(r["r1"]) > 0.5
    plain = stats.paired_test(ramp, np.zeros(8), widen=False)
    assert not plain["widened"] and plain["n"] == 8
    alt = np.tile([1.0, -1.0], 4) + np.linspace(0, 0.1, 8)
    assert stats.paired_test(alt, np.zeros(8))["widened"]  # negative lag-1 is widened too
    rng = np.random.default_rng(0)
    white = rng.standard_normal(8)
    assert abs(stats.lag1(white)) < 0.6 or stats.paired_test(white, np.zeros(8))["widened"] is True
    assert stats.lag1(np.ones(8)) is None and stats.lag1(np.arange(3.0)) is None


def test_blocking_helpers_reuse_w2() -> None:
    b = stats.window_blocking(480_000, 490_000)
    assert (b.lo, b.width, b.n) == (480_000, 1250, 8)
    steps = np.arange(480_001, 490_001)
    assert stats.block_mean(b, steps, steps.astype(float)).shape == (8,)
    with pytest.raises(ValueError):
        stats.window_blocking(0, 1001, 8)
    r = stats.run_blocking(500_000)
    assert (r.lo, r.width, r.n) == (520_000, 10_000, 8)


def test_seed_call_uses_the_floor_and_the_sign() -> None:
    def seed(mean: float, sd: float = 0.2) -> dict:
        rng = np.random.default_rng(int(mean * 100) % 97)
        x = rng.standard_normal(8)
        x = (x - x.mean()) / x.std(ddof=1) * sd + mean
        return stats.paired_test(x, np.zeros(8))

    up = [seed(1.0), seed(1.1), seed(0.9)]
    assert stats.seed_call(up)["label"] == "supported"
    assert stats.seed_call(up, floor=5.0)["label"] != "supported"  # below the seed-twin floor
    assert stats.seed_call(up)["p"] < 1e-6
    down = [seed(-1.0), seed(-1.1), seed(-0.9)]
    assert stats.seed_call(down)["label"] == "reversed"
    mixed = [seed(1.0), seed(0.0), seed(1.0)]
    assert stats.seed_call(mixed)["label"] in ("mixed", "not supported")


def test_noise_floor_is_the_mean_magnitude_of_twin_contrasts() -> None:
    assert stats.noise_floor([np.array([1.0, 3.0]), np.array([-1.0, -1.0])]) == pytest.approx(1.5)
    assert stats.noise_floor([np.array([np.nan])]) is None


# --- the comparison on synthetic panels ---


def _frozen() -> tuple[dict, str]:
    pr = prereg.default_prereg()
    pr["frozen"] = True
    pr[prereg.HASH_KEY] = h = prereg.prereg_hash(pr)
    return pr, h


def _calls(rng: np.random.Generator, effects: dict, pr: dict, h: str) -> dict[str, str]:
    c = compare.compare_panels(synth.synth_panels(rng, effects=effects, prereg_hash=h), pr)
    return {f: c["features"][f]["call"] for f in c["order"]}


def test_planted_feature_is_found_and_the_others_are_not() -> None:
    pr, h = _frozen()
    eff = {("bonds", e): 3.0 for e in ("P1", "P4", "P5")}
    calls = _calls(np.random.default_rng(5), eff, pr, h)
    assert calls["bonds"] == "supported"
    assert all(v == "not supported" for f, v in calls.items() if f != "bonds")


def test_cost_only_effect_is_not_supported() -> None:
    """A build-up and leave-one-out difference with none against the inert twin is cost, not function."""
    pr, h = _frozen()
    rng = np.random.default_rng(6)
    panels = synth.synth_panels(
        rng, effects={("comm", "P4"): 3.0, ("comm", "P1"): 3.0, ("comm", "P5"): 3.0}, prereg_hash=h
    )
    for p in panels:  # the inert twin shows the same endpoint levels as the full stack
        if p["header"]["arm"] == "I_comm":
            top = next(
                q for q in panels if q["header"]["arm"] == "U5" and q["header"]["seed"] == p["header"]["seed"]
            )
            p["endpoints"] = {k: dict(v) for k, v in top["endpoints"].items()}
    c = compare.compare_panels(panels, pr)
    assert c["features"]["comm"]["call"] != "supported"


def test_unevaluated_scorecard_cannot_confirm() -> None:
    pr, h = _frozen()
    rng = np.random.default_rng(7)
    panels = synth.synth_panels(
        rng, effects={("arms", e): 3.0 for e in ("P1", "P4", "P5")}, prereg_hash=h, scorecards=False
    )
    c = compare.compare_panels(panels, pr)
    assert c["features"]["arms"]["call"] == "mixed" and "scorecard" in c["features"]["arms"]["reason"]


def test_exploratory_endpoints_take_no_part_in_the_call() -> None:
    pr, h = _frozen()
    eff = {("niche", e): 3.0 for e in ("P2", "P3")}  # only the exploratory (O8) endpoints move
    calls = _calls(np.random.default_rng(8), eff, pr, h)
    assert calls["niche"] != "supported"


def _hits(rng: np.random.Generator, reps: int, d: float, h: str, pr: dict, **kw) -> float:
    eff = {("comm", e): d for e in ("P1", "P4", "P5")}  # planted on the confirmatory family
    ok = 0
    for _ in range(reps):
        c = compare.compare_panels(synth.synth_panels(rng, effects=eff, prereg_hash=h, **kw), pr)
        ok += c["features"]["comm"]["call"] == "supported"
    return ok / reps


def test_null_family_wise_rate_and_power() -> None:
    """Size 0 of 250 (0 of 2000 slow); power at d = 1.5 with 16 blocks (the prereg) >= 0.80. The honest
    8-block number: widening at |r1| > 0.3 halves the blocks and power falls to about 0.56."""
    pr, h = _frozen()
    rng = np.random.default_rng(11)
    reps, fw = 250, 0
    for _ in range(reps):
        fw += any(v == "supported" for v in _calls(rng, {}, pr, h).values())
    assert fw / reps <= 0.05
    assert _hits(rng, 200, 1.5, h, pr, n_blocks=16) >= 0.80  # measured 0.998
    assert 0.40 <= _hits(rng, 200, 1.5, h, pr, n_blocks=8) < 0.80  # measured 0.556: 8 blocks are not enough


@pytest.mark.slow
def test_null_and_power_with_2000_replications() -> None:
    pr, h = _frozen()
    rng = np.random.default_rng(12)
    fw = sum(any(v == "supported" for v in _calls(rng, {}, pr, h).values()) for _ in range(2000))
    assert fw / 2000 <= 0.05
    # 16 blocks, as the prereg uses after the review fixes (the 8-block default measures about 0.56)
    assert _hits(rng, 2000, 1.5, h, pr, n_blocks=16) >= 0.80


def test_prereg_hash_mismatch_refuses_supported() -> None:
    pr, _ = _frozen()
    eff = {("bonds", e): 3.0 for e in ("P1", "P4", "P5")}
    rng = np.random.default_rng(9)
    wrong = synth.synth_panels(rng, effects=eff, prereg_hash="0" * 64)
    with pytest.raises(prereg.PreregError, match="differ"):
        compare.compare_panels(wrong, pr)
    expl = compare.compare_panels(wrong, pr, exploratory=True)
    assert expl["prereg"]["exploratory"] and expl["features"]["bonds"]["exploratory"]
    unfrozen = prereg.default_prereg()
    with pytest.raises(prereg.PreregError, match="not frozen"):
        compare.compare_panels(synth.synth_panels(rng, prereg_hash=prereg.prereg_hash(unfrozen)), unfrozen)


def test_prereg_file_round_trip_and_tamper_check(tmp_path) -> None:
    path = tmp_path / "prereg.json"
    h = prereg.write_prereg(path, prereg.default_prereg())
    assert prereg.load_prereg(path)["sha256"] == h == prereg.prereg_hash(prereg.load_prereg(path))
    path.write_text(path.read_text().replace('"alpha": 0.05', '"alpha": 0.5'))
    with pytest.raises(prereg.PreregError, match="edited"):
        prereg.load_prereg(path)


def test_mixed_panel_versions_are_refused() -> None:
    pr, h = _frozen()
    ps = synth.synth_panels(np.random.default_rng(1), prereg_hash=h)
    ps[0]["header"]["panel_version"] = 99
    with pytest.raises(PanelError):
        compare.compare_panels(ps, pr)


def test_missing_arms_are_reported_not_filled() -> None:
    pr, h = _frozen()
    ps = [
        p
        for p in synth.synth_panels(np.random.default_rng(2), prereg_hash=h)
        if p["header"]["arm"] != "I_bonds"
    ]
    c = compare.compare_panels(ps, pr)
    assert c["features"]["bonds"]["call"] != "supported"
    assert any("I_bonds" in m for m in c["missing"])
    cell = c["features"]["bonds"]["endpoints"]["P1"]["series"]["control"]
    assert cell["label"] == "not evaluated"


def test_widening_at_0_3_and_nan_blocks_are_explicit() -> None:
    rng = np.random.default_rng(4)
    e = rng.standard_normal(40)
    x = np.zeros(40)
    for i in range(1, 40):  # AR(1) 0.5: r1 of an 8-block draw is usually above 0.3
        x[i] = 0.5 * x[i - 1] + e[i]
    d = x[:8] + 0.6
    r1 = stats.lag1(d)
    assert stats.WIDEN_AT == 0.3
    assert (abs(r1) > 0.3) == stats.paired_test(d, np.zeros(8))["widened"]
    nan = d.copy()
    nan[3] = np.nan
    r = stats.paired_test(nan, np.zeros(8))
    assert r["nan_blocks"] == 1 and r["r1"] is not None
    ramp = np.linspace(0, 1, 8)
    ramp[2] = np.nan
    w = stats.paired_test(ramp, np.zeros(8))
    assert (
        w["widened"] and w["n"] == 4 and w["nan_blocks"] == 1
    )  # a NaN block no longer disables the widening
    assert list(np.isnan(stats.merge_pairs(np.array([np.nan, np.nan, 1.0, 3.0])))) == [True, False]
    assert stats.merge_pairs(np.array([np.nan, 2.0, 1.0, 3.0])).tolist() == [2.0, 2.0]


def test_noise_floor_is_the_median_not_the_mean() -> None:
    twins = [np.array([0.4]), np.array([0.5]), np.array([0.3]), np.array([10.0])]
    assert stats.noise_floor(twins) == pytest.approx(0.45)  # the mean would be 2.8


def test_a_call_needs_three_seeds_and_a_finite_floor() -> None:
    pr, h = _frozen()
    eff = {("bonds", e): 3.0 for e in ("P1", "P4", "P5")}
    rng = np.random.default_rng(13)
    two = synth.synth_panels(rng, effects=eff, seeds=(0, 1), prereg_hash=h)
    c = compare.compare_panels(two, pr)["features"]["bonds"]
    assert c["call"] == "mixed" and "at least 3 seeds" in c["reason"]
    three = synth.synth_panels(rng, effects=eff, prereg_hash=h)
    no_twins = [p for p in three if p["header"]["arm"] not in ("T0", "T5")]
    c = compare.compare_panels(no_twins, pr)["features"]["bonds"]
    assert c["call"] == "mixed" and "noise floor" in c["reason"]
    assert compare.compare_panels(three, pr)["features"]["bonds"]["call"] == "supported"


def test_unevaluated_endpoint_counts_as_p_one_in_holm() -> None:
    pr, h = _frozen()
    eff = {("bonds", "P1"): 3.0, ("bonds", "P4"): 3.0, ("bonds", "P5"): 3.0}
    ps = synth.synth_panels(np.random.default_rng(14), effects=eff, prereg_hash=h)
    for p in ps:
        p["endpoints"]["P5"] = {**p["endpoints"]["P5"], "status": "needs_missing", "blocks": None}
    t = compare.compare_panels(ps, pr)["features"]["bonds"]["endpoints"]
    p1, p4 = (t[e]["series"]["control"]["p"] for e in ("P1", "P4"))
    assert t["P1"]["series"]["control"]["p_holm"] == pytest.approx(stats.holm([p1, p4, 1.0])[0])
    assert t["P1"]["series"]["control"]["p_holm"] > stats.holm([p1, p4])[0] or p1 == 0.0


def test_a_significantly_negative_feature_is_reversed() -> None:
    pr, h = _frozen()
    eff = {("comm", e): -3.0 for e in ("P1", "P4", "P5")}
    c = compare.compare_panels(synth.synth_panels(np.random.default_rng(15), effects=eff, prereg_hash=h), pr)
    assert c["features"]["comm"]["call"] == "reversed" and "lower" in c["features"]["comm"]["reason"]
