"""The labels study (``alife.analysis.labels``, DESIGN §28.8, docs/GAMES.md §4.5): the AUC and the matched
random label, exposure-normalised archetype scoring, intent scoring, the cohort trackers against a numpy
reference (on synthetic sequences and on real steps), and the study end to end on tiny and tiny3d."""

from __future__ import annotations

import dataclasses
import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import pytest

from alife import layout, runner
from alife.analysis import labels, mi
from alife.analysis.resim import ResimMismatch
from alife.config import Config, apply_overrides
from alife.game.observe import EVAL_BIT
from alife.game.observer import Observer
from alife.game.rules import ARCHETYPE_NAMES, INTENT_NAMES, RULES_V1, RULES_V2, Archetype, Intent
from alife.runio import RunDir, load_checkpoint
from alife.state import init_state, to_numpy
from alife.step import ActFlag, make_step
from game_helpers import blank_social_acts, make_acts
from helpers import set_output_bias

STUDY: dict[str, Any] = {
    "start": 50,
    "steps": 300,
    "warmup": 200,
    "cohort_every": 40,
    "arch_horizon": 60,
    "arch_min_obs": 20,
    "intent_horizon": 10,
    "n_perm": 10,
}
RUN = ["run.metrics_every=5", "run.checkpoint_every=10", "run.heavy_metrics_every=5"]
TOP_KEYS = {
    "run",
    "config_hash",
    "start",
    "steps",
    "params",
    "intent_version",
    "rules_digest",
    "device",
    "mlx_version",
    "observer_start",
    "rows_verified",
    "cohorts",
    "archetypes",
    "intents",
    "coverage",
    "flips_per_1000",
}
SCORE_KEYS = {"n", "n_pos", "auc", "null_mean", "null_sd", "z", "p"}
ARCH_KEYS = {
    "pool",
    "prevalence_match",
    "prevalence_primary",
    "outcome",
    "horizon",
    "min_obs",
    *SCORE_KEYS,
    "per_cohort",
    "verdict",
    "auc_complete",
    "auc_survived",
    "auc_censored",
    "exposure",
}
INTENT_KEYS = {
    "outcome",
    "horizon",
    "n_pool",
    "n_labelled",
    "censored_frac",
    "prevalence_label",
    "prevalence_rule",
    "auc_label",
    "auc_rule",
    "coverage_max",
    "verdict",
    "verdict_rule",
}
VERDICTS = {"pass", "fail", "insufficient"}


def _make_run(
    cfg: Config, path: Path, *extra: str, steps: int = 60, bias: Mapping[int, float] | None = None
) -> Path:
    cfg = apply_overrides(cfg, [*RUN, *extra])
    rd = RunDir.create(path, cfg)
    state = init_state(cfg) if bias is None else set_output_bias(init_state(cfg), cfg, bias)
    assert runner.run(rd, cfg, state, steps, quiet=True).status == "done"
    return path


@pytest.fixture(scope="module")
def run_dir(tiny_cfg: Config, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A 60-step tiny run with metrics every 5 steps and checkpoints every 10."""
    return _make_run(tiny_cfg, tmp_path_factory.mktemp("labels") / "run")


# founders' output biases that make bonds, shares, attacks, signals and births (and with them kin bonds and
# energy sent to kin) happen, so that every outcome column of the trackers is exercised on real steps
BIAS = {
    layout.OUT_HARVEST: 2.0,
    layout.OUT_BOND: 2.0,
    **{layout.OUT_SHARE + b: 1.0 for b in range(layout.B)},
    layout.OUT_REPRODUCE: 2.0,
    layout.OUT_CHILD_BOND: 2.0,
    layout.OUT_ATTACK: 1.0,
    layout.OUT_EMIT: 1.0,
}


@pytest.fixture(scope="module")
def biased_run_dir(tiny_cfg: Config, tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("labels_biased") / "run"
    return _make_run(
        tiny_cfg, path, "population.immigration_every=5", "population.immigration_count=8", bias=BIAS
    )


@pytest.fixture(scope="module")
def run_dir_3d(tiny3d_cfg: Config, tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _make_run(tiny3d_cfg, tmp_path_factory.mktemp("labels3d") / "run")


def _files(path: Path) -> dict[str, tuple[int, int]]:
    return {
        str(p.relative_to(path)): (p.stat().st_size, p.stat().st_mtime_ns)
        for p in sorted(path.rglob("*"))
        if p.is_file()
    }


# --- the AUC and the matched random label ------------------------------------------------------------------


def _brute_auc(x: np.ndarray, y: np.ndarray) -> float:
    pos, neg = y[x], y[~x]
    return float(((pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])).mean())


def test_auc_basics() -> None:
    x = np.array([True, True, False, False])
    assert labels.auc(x, np.array([3.0, 4.0, 1.0, 2.0])) == 1.0
    assert labels.auc(x, np.array([1.0, 2.0, 3.0, 4.0])) == 0.0
    assert labels.auc(x, np.zeros(4)) == 0.5  # all tied
    assert labels.auc(~x, np.array([3.0, 4.0, 1.0, 2.0])) == 0.0
    assert np.isnan(labels.auc(np.ones(4, bool), np.arange(4.0))) and np.isnan(
        labels.auc(~np.ones(4, bool), np.arange(4.0))
    )
    rng = np.random.default_rng(5)
    for _ in range(20):  # integer outcomes, many ties
        n = int(rng.integers(5, 60))
        xx = rng.random(n) < rng.uniform(0.2, 0.8)
        yy = rng.integers(0, 6, n).astype(float)
        if xx.any() and not xx.all():
            assert labels.auc(xx, yy) == pytest.approx(_brute_auc(xx, yy), abs=1e-12)


def test_matched_random_null() -> None:
    rng = np.random.default_rng(1)
    # unrelated label and outcome: the null centres on 0.5, and so does the AUC
    x, y = rng.random(300) < 0.4, rng.normal(size=300)
    s = labels._score(x, y, np.zeros(300, int), 300, np.random.default_rng(2))
    assert s["n"] == 300 and s["n_pos"] == int(x.sum())
    assert 0.45 < s["null_mean"] < 0.55 and abs(s["auc"] - s["null_mean"]) < 4 * s["null_sd"]
    assert s["p"] > 0.01

    # a label that only tracks a stratum (prevalence varies across strata, so does the outcome): the naive AUC
    # is high but the matched null has the same mean, and the label is not significant
    strata = rng.integers(0, 8, 600)
    x = rng.random(600) < np.linspace(0.1, 0.9, 8)[strata]
    y = strata + rng.normal(0, 0.5, 600)
    assert labels.auc(x, y) > 0.7
    s = labels._score(x, y, strata, 200, np.random.default_rng(3))
    assert s["null_mean"] == pytest.approx(s["auc"], abs=0.03) and s["p"] > 0.05
    assert labels.verdict({**s, "n": 600}) in VERDICTS - {"pass"}

    # a real effect inside the strata is found
    y2 = strata + 1.5 * x + rng.normal(0, 0.5, 600)
    s2 = labels._score(x, y2, strata, 200, np.random.default_rng(4))
    assert s2["p"] <= 0.01 and s2["auc"] - s2["null_mean"] >= 0.05 and labels.verdict(s2) == "pass"


def test_the_permutation_keeps_each_strata_prevalence() -> None:
    rng = np.random.default_rng(6)
    strata = rng.integers(0, 5, 200)
    x = rng.random(200) < np.linspace(0.1, 0.9, 5)[strata]
    for _ in range(5):
        xp = x[mi.permute_within(rng, 200, strata)]
        for s in range(5):
            assert xp[strata == s].sum() == x[strata == s].sum()


def test_strata_are_cohort_and_age_quartile_of_the_scored_rows() -> None:
    cohort = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
    age = np.array([5, 6, 7, 8, 9, 10, 11, 12, 100, 100, 100, 100, 7, 7, 7, 7])
    scored = np.ones(16, bool)
    scored[[0, 15]] = False  # unscored rows are outside the quartiles and sit in quartile 0
    quart, strata = labels._age_strata(cohort, age, scored)
    assert np.array_equal(strata, cohort * 4 + quart) and quart[0] == quart[15] == 0
    assert quart[scored].min() == 0 and quart[scored].max() == 3
    assert quart[8] == quart[9] == quart[10] == quart[11]  # tied ages share a bin
    assert np.array_equal(quart, labels._age_strata(cohort, age, scored)[0])  # deterministic


def test_score_is_none_for_a_single_class() -> None:
    s = labels._score(np.zeros(10, bool), np.arange(10.0), np.zeros(10, int), 5, np.random.default_rng(0))
    assert s["n"] == 10 and s["n_pos"] == 0 and s["auc"] is None and s["p"] is None
    assert labels.verdict(s) == "insufficient"


@pytest.mark.parametrize(
    ("n", "n_pos", "p", "auc", "null", "cov", "intent", "want"),
    [
        (100, 30, 0.01, 0.56, 0.50, None, False, "pass"),  # p at its limit and margin above 0.05: pass
        (100, 30, 0.011, 0.9, 0.50, None, False, "fail"),
        (100, 30, 0.001, 0.54, 0.50, None, False, "fail"),  # margin below 0.05
        (100, 29, 0.001, 0.9, 0.50, None, False, "insufficient"),  # too few positives
        (100, 71, 0.001, 0.9, 0.50, None, False, "insufficient"),  # 29 negatives
        (100, 50, 0.001, 0.9, 0.50, 0.6, True, "pass"),  # coverage at its limit
        (100, 50, 0.001, 0.9, 0.50, 0.61, True, "fail"),
        (100, 50, 0.001, 0.9, 0.50, 0.99, False, "pass"),  # archetypes have no coverage clause
        # the margin is over the matched random label, not over 0.5
        (100, 50, 0.001, 0.62, 0.60, None, False, "fail"),
        (100, 50, 0.001, 0.62, 0.55, None, False, "pass"),
        (100, 50, 0.001, 0.62, 0.57, None, False, "pass"),  # a margin of exactly 0.05
    ],
)
def test_verdict_thresholds(
    n: int, n_pos: int, p: float, auc: float, null: float, cov: float | None, intent: bool, want: str
) -> None:
    s = {"n": n, "n_pos": n_pos, "auc": auc, "null_mean": null, "null_sd": 0.05, "z": 5.0, "p": p}
    assert labels.verdict(s, cov, is_intent=intent) == want


@pytest.mark.parametrize(
    ("n_perm", "want"),
    [
        (10, "insufficient"),
        (50, "insufficient"),
        (98, "insufficient"),
        (99, "pass"),
        (200, "pass"),
        (None, "pass"),
    ],
)
def test_too_few_permutations_cannot_judge_a_label(n_perm: int | None, want: str) -> None:
    """p >= 1/(1 + n_perm) > 0.01 below 99 permutations: a perfectly separating label would otherwise be
    reported as a failure (and dropped from the UI) by a quick calibration."""
    x = np.r_[np.ones(100, bool), np.zeros(100, bool)]
    y = np.r_[np.ones(100), np.zeros(100)]
    n_perm_run = 99 if n_perm is None else n_perm
    s = labels._score(x, y, np.zeros(200, int), n_perm_run, np.random.default_rng(1))
    assert s["auc"] == 1.0 and s["p"] == pytest.approx(1 / (1 + n_perm_run))
    assert labels.verdict(s, n_perm=n_perm) == want
    # and through the scoring function, which knows its n_perm (ages interleave the classes: the strata
    # hold both, and the label separates the outcome inside each of them)
    pos = np.arange(200) % 2 == 0
    zero = np.zeros(200, np.int64)
    tab = {
        "cohort": zero,
        "age": np.arange(200),
        "primary": zero,
        "match": np.where(pos, 1 << (Archetype.GATHERER.value - 1), 0),
        "n_ok": np.full(200, 100),
        "rg2_min": np.zeros(200),
        "rg2_full": np.zeros(200),
        "harvest": np.where(pos, 5000, 100),
        "attack": zero,
        "out": zero,
        "got": zero,
    }
    g = labels.score_archetypes(tab, arch_horizon=100, arch_min_obs=20, n_perm=n_perm_run, seed=0)["gatherer"]
    assert g["auc"] == 1.0 and g["verdict"] == (want if n_perm is not None else "pass")


# --- the matched random label inside the scoring functions -----------------------------------------------

AGE_PREVALENCE = np.array([0.1, 0.3, 0.6, 0.9])  # the label's prevalence in each age quartile
DEFENDING_BIT = 1 << (Intent.DEFENDING.value - 1)


def _age_confounded(effect: float, n: int = 4000, seed: int = 0) -> tuple[dict[str, Any], dict[str, Any]]:
    """An archetype table (Gatherer) and an intent table (Defending) with two cohorts, in which the label's
    prevalence and the outcome both rise with the age quartile and, by ``effect``, with the label itself.
    The quartiles are exact (four age bands of n/4 rows), half the rows are censored whatever their age."""
    rng = np.random.default_rng(seed)
    cohort = rng.integers(0, 2, n)
    q = rng.permutation(np.tile(np.arange(4), n // 4))
    age = q * 1000 + rng.integers(0, 100, n)
    x = rng.random(n) < AGE_PREVALENCE[q]
    rate = (q * 10 + effect * x + rng.normal(0, 2, n)).clip(0)
    n_ok = np.where(rng.random(n) < 0.5, 100, 50)
    zero = np.zeros(n, np.int64)
    arch = {
        "cohort": cohort,
        "age": age,
        "primary": np.where(x, Archetype.GATHERER.value, Archetype.GENERALIST.value),
        "match": np.where(x, 1 << (Archetype.GATHERER.value - 1), 0),
        "n_ok": n_ok,
        "rg2_min": np.zeros(n),
        "rg2_full": np.zeros(n),
        "harvest": np.round(rate * n_ok).astype(np.int64),  # totals: the rate over the observed steps
        "attack": zero,
        "out": zero,
        "got": zero,
    }
    intent = {
        "cohort": cohort,
        "age": age,
        "label": np.where(x, Intent.DEFENDING.value, Intent.RESTING.value),
        "match": EVAL_BIT | np.where(x, DEFENDING_BIT, 1 << (Intent.RESTING.value - 1)),
        "complete": np.ones(n, bool),
        **{k: zero for k in ("bud", "attack", "formed", "thrust", "emit", "harvest", "care")},
        "defend": np.round(rate * 10).astype(np.int64),
    }
    return arch, intent


def _scores_of(arch: dict[str, Any], intent: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Every score the two scoring functions compute for the one label, by name."""
    g = labels.score_archetypes(arch, arch_horizon=100, arch_min_obs=20, n_perm=200, seed=0)["gatherer"]
    d = labels.score_intents(intent, n_perm=200, seed=0, intent_horizon=10)["defending"]
    named = {
        "arch pooled": g,
        "arch complete": g["auc_complete"],
        "arch survived": g["auc_survived"],
        "arch censored": g["auc_censored"],
        "intent label": d["auc_label"],
        "intent rule": d["auc_rule"],
    }
    for name, s in (("arch", g), ("intent label", d["auc_label"]), ("intent rule", d["auc_rule"])):
        assert len(s["per_cohort"]) == 2
        named.update({f"{name} cohort {i}": c for i, c in enumerate(s["per_cohort"])})
    return named


def test_every_score_is_against_the_age_matched_random_label() -> None:
    """A label that only tracks the age quartile scores like the random label on every score the tools
    compute: pooled, per cohort, complete cases, survivors and censored, archetype and intent. Against an
    unstratified null (the permutation over all rows) the same table gives a high AUC over a null of 0.5."""
    arch, intent = _age_confounded(effect=0.0)
    assert labels.auc(arch["match"] > 0, arch["harvest"] / arch["n_ok"]) > 0.8  # the naive AUC is high
    for name, s in _scores_of(arch, intent).items():
        assert s["auc"] is not None and abs(s["auc"] - s["null_mean"]) < 0.03, (name, s)
        assert labels.verdict(s, n_perm=200) != "pass", name


def test_a_real_effect_inside_the_age_strata_passes() -> None:
    """The control: the same table with the outcome raised by the label itself passes everywhere."""
    arch, intent = _age_confounded(effect=6.0)
    scores = _scores_of(arch, intent)
    for name, s in scores.items():
        assert s["auc"] - s["null_mean"] >= 0.05 and s["p"] <= 0.01, (name, s)
    for name in ("arch pooled", "intent label", "intent rule"):
        assert labels.verdict(scores[name], n_perm=200) == "pass", name


# --- exposure-normalised archetype scoring -----------------------------------------------------------------


def _arch_table(groups: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    """One archetype-table row per group member (``count`` rows with the group's fields)."""
    cols: dict[str, list[Any]] = {
        k: []
        for k in (
            "cohort",
            "age",
            "primary",
            "match",
            "n_ok",
            "rg2_min",
            "rg2_full",
            "harvest",
            "attack",
            "out",
            "got",
        )
    }
    for g in groups:
        for _ in range(g["count"]):
            row = {
                "cohort": 0,
                "primary": 0,
                "rg2_min": 0.0,
                "rg2_full": 0.0,
                "attack": 0,
                "out": 0,
                "got": 0,
                **g,
            }
            row["age"] = len(cols["age"])
            for k, col in cols.items():
                col.append(row.get(k, 0))
    return {k: np.array(v, dtype=np.float64 if k.startswith("rg2") else np.int64) for k, v in cols.items()}


def test_exposure_normalised_scoring() -> None:
    gatherer = 1 << (Archetype.GATHERER.value - 1)
    tab = _arch_table(
        [
            # (match bit, observed steps, harvest total) -> rate
            {"count": 12, "match": gatherer, "n_ok": 100, "harvest": 5000},  # complete positives: rate 50
            {"count": 8, "match": gatherer, "n_ok": 25, "harvest": 500},  # censored positives: rate 20
            {"count": 12, "match": 0, "n_ok": 100, "harvest": 1000},  # complete negatives: rate 10
            {"count": 14, "match": 0, "n_ok": 30, "harvest": 150},  # censored negatives: rate 5
            {"count": 5, "match": gatherer, "n_ok": 10, "harvest": 0},  # short positives (not scored)
            {"count": 5, "match": 0, "n_ok": 10, "harvest": 10**6},  # short negatives (not scored)
        ]
    )
    out = labels.score_archetypes(tab, arch_horizon=100, arch_min_obs=20, n_perm=5, seed=0)
    g = out["gatherer"]
    # primary: rows observed >= 20 steps, harvest per observed step: positives (50, 20) beat negatives
    # (10, 5); on totals the censored positives (500) would lose to the complete negatives (1000)
    assert (g["n"], g["n_pos"], g["auc"]) == (46, 20, 1.0)
    assert g["pool"] == 56 and g["prevalence_match"] == pytest.approx(25 / 56)
    # complete cases: only the rows observed all 100 steps, on the totals
    assert (g["auc_complete"]["n"], g["auc_complete"]["n_pos"], g["auc_complete"]["auc"]) == (24, 12, 1.0)
    # survived: the primary outcome among the rows observed to the horizon; censored: among the other scored
    assert (g["auc_survived"]["n"], g["auc_survived"]["auc"]) == (24, 1.0)
    assert (g["auc_censored"]["n"], g["auc_censored"]["n_pos"], g["auc_censored"]["auc"]) == (22, 8, 1.0)
    pos, neg = g["exposure"]["pos"], g["exposure"]["neg"]
    assert (pos["n_pool"], pos["n_short"]) == (25, 5) and pos["censored_frac"] == pytest.approx(1 - 12 / 25)
    assert pos["n_ok_median"] == 100.0 and neg["n_ok_median"] == 30.0  # over the scored rows only
    assert (neg["n_pool"], neg["n_short"]) == (31, 5) and neg["censored_frac"] == pytest.approx(1 - 12 / 31)
    assert g["verdict"] == "insufficient" and g["outcome"] == "harvest_rate"


def test_movement_is_scored_on_the_same_exposure_for_every_row() -> None:
    nomad, settler = 1 << (Archetype.NOMAD.value - 1), 1 << (Archetype.SETTLER.value - 1)
    # Rg² over the first arch_min_obs steps (primary) separates the labels; the full-track Rg² (complete
    # cases) says the opposite. Nomad predicts a large Rg², Settler a small one.
    tab = _arch_table(
        [
            {"count": 15, "match": nomad, "n_ok": 100, "rg2_min": 100.0, "rg2_full": 1.0},
            {"count": 15, "match": settler, "n_ok": 100, "rg2_min": 1.0, "rg2_full": 100.0},
            {"count": 15, "match": 0, "n_ok": 100, "rg2_min": 50.0, "rg2_full": 50.0},
        ]
    )
    out = labels.score_archetypes(tab, arch_horizon=100, arch_min_obs=20, n_perm=5, seed=0)
    assert out["nomad"]["auc"] == 1.0 and out["nomad"]["auc_complete"]["auc"] == 0.0
    assert out["settler"]["auc"] == 1.0 and out["settler"]["auc_complete"]["auc"] == 0.0
    assert out["nomad"]["outcome"] == "rg2" and out["settler"]["outcome"] == "neg_rg2"
    assert out["generalist"].keys() == {"pool", "prevalence_match", "prevalence_primary"}  # prevalence only


def test_trader_and_provider_outcomes() -> None:
    trader = 1 << (Archetype.TRADER.value - 1)
    tab = _arch_table(
        [
            {"count": 15, "match": trader, "n_ok": 50, "out": 500, "got": 400},  # min(out, got) rate 8
            {"count": 15, "match": 0, "n_ok": 50, "out": 5000, "got": 100},  # out is big, min is 2
        ]
    )
    out = labels.score_archetypes(tab, arch_horizon=50, arch_min_obs=10, n_perm=5, seed=0)
    assert out["trader"]["auc"] == 1.0 and out["trader"]["outcome"] == "min_out_got_rate"
    provider = 1 << (Archetype.PROVIDER.value - 1)
    tab = _arch_table(
        [
            {"count": 15, "match": provider, "n_ok": 50, "out": 5000, "got": 0},
            {"count": 15, "match": 0, "n_ok": 50, "out": 500, "got": 0},
        ]
    )
    assert (
        labels.score_archetypes(tab, arch_horizon=50, arch_min_obs=10, n_perm=5, seed=0)["provider"]["auc"]
        == 1.0
    )


def test_scoring_an_empty_table() -> None:
    tab = _arch_table([])
    out = labels.score_archetypes(tab, arch_horizon=100, arch_min_obs=20, n_perm=5, seed=0)
    assert set(out) == set(ARCHETYPE_NAMES) and out["gatherer"]["n"] == 0
    assert out["gatherer"]["verdict"] == "insufficient" and out["gatherer"]["prevalence_match"] is None


# --- intent scoring ----------------------------------------------------------------------------------------


def test_intent_scoring_pools_and_outcomes() -> None:
    n = 100
    idx = np.arange(n)
    defending, resting = Intent.DEFENDING.value, Intent.RESTING.value
    label = np.zeros(n, int)
    label[:60] = defending  # rows 0-9 are not complete; 10-59 defend
    label[60:90] = resting
    # rows 90-99: evaluated, no displayed label yet
    match = np.zeros(n, int) | EVAL_BIT
    match[10:60] |= 1 << (defending - 1)
    match[60:90] |= 1 << (resting - 1)
    tab = {
        "cohort": np.zeros(n, int),
        "age": idx,
        "label": label,
        "match": match,
        "complete": idx >= 10,
        **{k: np.zeros(n, int) for k in ("bud", "attack", "formed", "emit", "harvest", "care")},
        "defend": np.where(np.isin(idx, range(10, 60)), 200, 10),
        "thrust": np.where(np.isin(idx, range(60, 90)), 0, 100),
    }
    out = labels.score_intents(tab, n_perm=5, seed=0, intent_horizon=10)
    assert list(out) == list(INTENT_NAMES) and set(out["defending"]) == INTENT_KEYS
    d = out["defending"]
    # the label's pool is the complete rows with a displayed label (80); the rule's is every complete row
    assert (d["auc_label"]["n"], d["auc_label"]["n_pos"], d["auc_label"]["auc"]) == (80, 50, 1.0)
    assert (d["auc_rule"]["n"], d["auc_rule"]["n_pos"], d["auc_rule"]["auc"]) == (90, 50, 1.0)
    assert d["censored_frac"] == pytest.approx(0.1) and d["n_pool"] == 100 and d["n_labelled"] == 90
    assert d["outcome"] == "defend" and d["horizon"] == 10
    r = out["resting"]  # Resting predicts less thrust: the outcome is -thrust
    assert r["outcome"] == "-thrust" and r["auc_label"]["auc"] == 1.0 and r["auc_rule"]["auc"] == 1.0
    assert out["roaming"]["outcome"] == "thrust" and out["roaming"]["auc_label"]["n_pos"] == 0
    # verdicts: 50 positives and 30 negatives pass the size gate; coverage decides
    cov = {"intent": {"defending": 0.9}, "intent_match": {"defending": 0.4}}
    out = labels.score_intents(tab, n_perm=200, seed=0, intent_horizon=10, coverage_max=cov)
    d = out["defending"]
    assert d["coverage_max"] == {"label": 0.9, "rule": 0.4}
    assert (
        d["verdict"] == "fail" and d["verdict_rule"] == "pass"
    )  # the label covers too much, the rule does not


def test_per_cohort_scores() -> None:
    n = 120
    cohort = np.repeat([0, 1], n // 2)
    tab = {
        "cohort": cohort,
        "age": np.tile(np.arange(n // 2), 2),
        "label": np.where(np.arange(n) % 2 == 0, Intent.HUNTING.value, Intent.RESTING.value),
        "match": np.full(n, EVAL_BIT),
        "complete": np.ones(n, bool),
        **{k: np.zeros(n, int) for k in ("bud", "formed", "thrust", "defend", "emit", "harvest", "care")},
        "attack": np.where(np.arange(n) % 2 == 0, 5, 0),
    }
    out = labels.score_intents(tab, n_perm=5, seed=0, intent_horizon=10, cohort_steps=[256, 512])
    per = out["hunting"]["auc_label"]["per_cohort"]
    assert [c["cohort"] for c in per] == [256, 512] and all(c["n"] == n // 2 and c["auc"] == 1.0 for c in per)


# --- coverage and flips ------------------------------------------------------------------------------------


def test_flips_count_labelled_pairs_of_the_same_row_only() -> None:
    f = labels._Flips()
    uid = np.array([1, 2, 3, 4, -1])
    f.add(uid, np.array([1, 1, 2, 0, 0]), np.array([1, 1, 1, 1, 0]))
    assert f.per_1000(8) == {"intent": None, "archetype": None}  # no pair yet
    # row 1 changes both labels; row 2 changes only its intent; row 3 was unlabelled for archetypes; row 0 is
    # a new occupant (uid 9)
    f.add(np.array([9, 2, 3, 4, -1]), np.array([1, 2, 2, 3, 0]), np.array([1, 2, 2, 1, 0]))
    assert (f.pairs["archetype"], f.flips["archetype"]) == (2, 1)  # rows 1 and 2
    assert (f.pairs["intent"], f.flips["intent"]) == (3, 2)  # rows 1, 2 and 3
    assert f.per_1000(8)["archetype"] == pytest.approx(1000 * 1 / (2 * 8))
    assert f.per_1000(8)["intent"] == pytest.approx(1000 * 2 / (3 * 8))


def test_coverage_mean_and_max_skip_empty_shares() -> None:
    c = labels._Coverage()
    block = {
        "intent_match": {n: 0.0 for n in INTENT_NAMES} | {"defending": 0.2},
        "intent": None,
        "archetype": {n: 0.1 for n in ARCHETYPE_NAMES},
        "archetype_match": None,
    }
    c.add(block)
    c.add(
        {
            **block,
            "intent_match": {n: 0.0 for n in INTENT_NAMES} | {"defending": 0.6},
            "intent": {n: 0.5 for n in INTENT_NAMES},
        }
    )
    rep = c.report()
    assert rep["evaluations"] == 2
    assert rep["intent_match"]["defending"] == {"mean": pytest.approx(0.4), "max": 0.6}
    assert rep["intent"]["resting"] == {"mean": 0.5, "max": 0.5}  # the first evaluation had no shares
    assert rep["archetype_match"]["trader"] == {"mean": None, "max": None}
    assert c.maxima()["intent_match"]["defending"] == 0.6


# --- the cohort trackers -----------------------------------------------------------------------------------


def _np_min_image(d: np.ndarray, size: float) -> np.ndarray:
    return d - size * np.round(d / size)


def test_tracker_follows_the_same_occupant_and_unwraps_the_track() -> None:
    n, nd, size, steps = 6, 2, 10.0, 8
    rng = np.random.default_rng(11)
    cuid = np.array([10, 11, 12, -1, 14, -1], dtype=np.int64)
    member = cuid >= 0
    alive = np.ones((steps, n), bool)
    uid = np.broadcast_to(np.arange(10, 16, dtype=np.int64), (steps, n)).copy()
    alive[3:, 1] = False  # row 1 dies at the 4th step
    uid[2:, 2] = 99  # row 2's row is reused at the 3rd step
    uid[:, 3] = 13  # row 3 is not a member
    alive[:, 5] = False
    uid[:, 5] = -1
    pos = np.zeros((steps, n, nd))
    start = np.array([[8.0, 1.0], [2.0, 2.0], [5.0, 5.0], [1.0, 1.0], [3.0, 9.0], [0.0, 0.0]])
    jitter = np.cumsum(rng.normal(0, 0.4, (steps, nd)), axis=0)
    for k in range(steps):  # row 0 walks +1.5 in x across the periodic boundary; row 4 jitters
        pos[k] = start
        pos[k, 0, 0] = start[0, 0] + 1.5 * (k + 1)
        pos[k, 4] = start[4] + jitter[k]
        pos[k, 1:4] += 0.3 * (k + 1)
    pos = np.mod(pos, size).astype(np.float32)
    oc = rng.integers(0, 50, (steps, n, len(labels.OUTCOMES))).astype(np.int32)

    start_pos = np.mod(start, size).astype(np.float32)
    trk = labels.track_start(mx.array(member), mx.array(start_pos))
    for k in range(steps):
        trk = labels.track_step(
            mx.array(cuid), trk, mx.array(alive[k]), mx.array(uid[k]), mx.array(pos[k]), mx.array(oc[k]), size
        )
    mx.eval(*trk)
    host = labels._host_track(trk)

    # numpy reference in float64
    ok = member.copy()
    n_ok = np.zeros(n, int)
    tot = np.zeros((n, len(labels.OUTCOMES)), int)
    last = start_pos.astype(np.float64)
    u = np.zeros((n, nd))
    track: list[list[np.ndarray]] = [[] for _ in range(n)]
    for k in range(steps):
        ok = ok & alive[k] & (uid[k] == cuid)
        n_ok += ok
        tot += np.where(ok[:, None], oc[k], 0)
        step = _np_min_image(pos[k].astype(np.float64) - last, size)
        u = u + np.where(ok[:, None], step, 0.0)
        last = np.where(ok[:, None], pos[k], last)
        for i in np.flatnonzero(ok):
            track[i].append(u[i].copy())
    assert n_ok.tolist() == [8, 3, 2, 0, 8, 0] and np.array_equal(host["n_ok"], n_ok)
    for name, i in labels._OC.items():
        assert np.array_equal(host[name], tot[:, i]), name
    for i in range(n):
        if n_ok[i] == 0:
            assert host["rg2"][i] == 0.0
            continue
        pts = np.array(track[i])
        ref = ((pts - pts.mean(axis=0)) ** 2).sum(axis=1).mean()
        assert host["rg2"][i] == pytest.approx(ref, rel=2e-4, abs=1e-6), i
    # row 0 walks 1.5 per step along a line across the boundary: Rg² is that of the points 1.5·(1..8)
    assert host["rg2"][0] == pytest.approx(1.5**2 * (8**2 - 1) / 12, rel=1e-5)


def test_tracker_compiled_equals_eager() -> None:
    n, nd, size = 16, 3, 20.0
    rng = np.random.default_rng(3)
    member = rng.random(n) < 0.7
    cuid = np.where(member, np.arange(n), -1).astype(np.int64)
    pos0 = rng.uniform(0, size, (n, nd)).astype(np.float32)
    seq = [
        (
            rng.random(n) < 0.97,
            np.arange(n),
            np.mod(pos0 + rng.normal(0, 1, (n, nd)).astype(np.float32), size),
            rng.integers(0, 9, (n, len(labels.OUTCOMES))).astype(np.int32),
        )
        for _ in range(12)
    ]

    def fn(
        cuid_: mx.array, tr: labels.TrackState, a: mx.array, u: mx.array, p: mx.array, oc: mx.array
    ) -> Any:
        return labels.track_step(cuid_, tr, a, u, p, oc, size)

    out = []
    for step in (fn, mx.compile(fn)):
        trk = labels.track_start(mx.array(member), mx.array(pos0))
        for alive, uid, pos, oc in seq:
            trk = step(
                mx.array(cuid),
                trk,
                mx.array(alive),
                mx.array(uid.astype(np.int64)),
                mx.array(pos),
                mx.array(oc),
            )
            mx.eval(*trk)
        out.append([to_numpy(a) for a in trk])
    for a, b in zip(*out, strict=True):  # integers and flags exactly; the float track to rounding
        if a.dtype == np.float32:
            np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5)
        else:
            assert np.array_equal(a, b)


def test_step_outcomes() -> None:
    n = 4
    a = {
        "attack_gain": np.zeros(n, np.int32),
        "share_out": np.zeros((n, layout.B), np.int32),
        "share_got": np.zeros((n, layout.B), np.int32),
        "flags": np.zeros(n, np.uint8),
        "thrust": np.zeros(n, np.int16),
        "defend": np.zeros(n, np.int16),
        "emit": np.zeros(n, np.int16),
        **blank_social_acts(n, 1, 1),  # the outcomes read none of the social additions
    }
    a["flags"][:] = [0, ActFlag.BUDDED.value, ActFlag.ATTACKED.value | ActFlag.FORMED.value, 0xFF]
    a["thrust"][:] = [0, -200, 256, -1]
    a["defend"][:] = [0, 3, 128, 256]
    a["emit"][:] = [0, 765, 1, 0]
    a["share_out"][:, 0], a["share_out"][:, 1] = [0, 10, 20, 30], [0, 1, 2, 3]
    a["share_got"][:, 0], a["share_got"][:, 2] = [0, 5, 6, 7], [0, 1, 1, 1]
    kin = np.zeros((n, layout.B), dtype=bool)
    kin[1, 0] = kin[2, 1] = kin[3, 0] = True  # the energy sent to kin: 10 (row 1), 2 (row 2), 30 (row 3)
    harvest = np.array([0, 4, 0, 400], dtype=np.int32)
    oc = to_numpy(labels.step_outcomes(make_acts(a), mx.array(harvest), mx.array(kin)))
    assert oc.shape == (n, len(labels.OUTCOMES)) and oc.dtype == np.int32
    col = {name: oc[:, i] for name, i in labels._OC.items()}
    assert col["bud"].tolist() == [0, 1, 0, 1]
    assert col["attack"].tolist() == [0, 0, 1, 1] and col["formed"].tolist() == [0, 0, 1, 1]
    assert col["thrust"].tolist() == [0, 200, 256, 1]  # |thrust|
    assert col["defend"].tolist() == [0, 3, 128, 256] and col["emit"].tolist() == [0, 765, 1, 0]
    assert col["harvest"].tolist() == harvest.tolist()
    assert col["care"].tolist() == [0, 10, 2, 30]
    assert col["out"].tolist() == [0, 11, 22, 33] and col["got"].tolist() == [0, 6, 7, 8]


# --- the study ---------------------------------------------------------------------------------------------


def _documented(rep: Mapping[str, Any]) -> None:
    assert set(rep) == TOP_KEYS
    assert list(rep["archetypes"]) == list(ARCHETYPE_NAMES) and list(rep["intents"]) == list(INTENT_NAMES)
    for name, a in rep["archetypes"].items():
        if name == "generalist":
            assert set(a) == {"pool", "prevalence_match", "prevalence_primary"}
            continue
        assert set(a) == ARCH_KEYS and a["verdict"] in VERDICTS, name
        for col in ("auc_complete", "auc_survived", "auc_censored"):
            assert set(a[col]) == SCORE_KEYS
    for name, i in rep["intents"].items():
        assert set(i) == INTENT_KEYS and i["verdict"] in VERDICTS and i["verdict_rule"] in VERDICTS, name
        assert set(i["auc_label"]) == {*SCORE_KEYS, "per_cohort"} and set(i["auc_rule"]) == {
            *SCORE_KEYS,
            "per_cohort",
        }
    assert rep["intent_version"] == 2 and rep["rules_digest"] == RULES_V2.digest()
    assert set(rep["flips_per_1000"]) == {"intent", "archetype"}
    cov = rep["coverage"]
    assert set(cov) == {"evaluations", "intent_match", "intent", "archetype", "archetype_match"}
    assert list(cov["intent_match"]) == list(INTENT_NAMES) and list(cov["archetype_match"]) == list(
        ARCHETYPE_NAMES
    )
    json.dumps(rep, allow_nan=False)  # JSON-safe: no NaN or infinity


def test_study_on_tiny(run_dir: Path) -> None:
    before = _files(run_dir)
    logs: list[str] = []
    rep = labels.study(run_dir, **STUDY, log=logs.append)
    _documented(rep)
    assert _files(run_dir) == before  # read only: names, sizes and mtimes
    assert rep["rows_verified"] > 0 and rep["start"] == 50 and rep["steps"] == 300
    assert rep["observer_start"]["step"] == 50 and rep["observer_start"]["source"] == "fresh"
    assert [c["step"] for c in rep["cohorts"]] == [
        256
    ]  # first multiple of 8 at or after 50 + 200; 296 > 350 - 60
    assert rep["cohorts"][0]["n_archetype_pool"] > 0 and rep["cohorts"][0]["n_intent_pool"] > 0
    assert rep["params"]["n_perm"] == 10 and rep["params"]["rules"]["arch_k"] == 6
    assert rep["coverage"]["evaluations"] == 12  # evaluation steps 256, 264, ..., 344
    assert any("labels:" in m for m in logs)
    # the same seed gives the identical report; another seed changes the matched nulls but not the AUCs
    again = labels.study(run_dir, **STUDY, log=lambda _: None)
    assert json.dumps(rep) == json.dumps(again)
    other = labels.study(run_dir, **{**STUDY, "seed": 1}, log=lambda _: None)
    assert other["archetypes"]["provider"]["auc"] == rep["archetypes"]["provider"]["auc"]
    assert other["archetypes"]["provider"]["null_mean"] != rep["archetypes"]["provider"]["null_mean"]


def test_study_on_tiny3d(run_dir_3d: Path) -> None:
    rep = labels.study(run_dir_3d, **{**STUDY, "n_perm": 5}, log=lambda _: None)
    _documented(rep)
    assert rep["rows_verified"] > 0 and rep["cohorts"][0]["n_archetype_pool"] > 0
    assert rep["archetypes"]["nomad"]["pool"] == rep["archetypes"]["gatherer"]["pool"]


def test_study_can_skip_verification(run_dir: Path) -> None:
    short = {**STUDY, "steps": 200, "arch_horizon": 20, "arch_min_obs": 10, "warmup": 100}
    rep = labels.study(run_dir, **short, verify=False, log=lambda _: None)
    assert rep["rows_verified"] == 0 and rep["params"]["verify"] is False


def test_study_detects_a_diverging_row(run_dir: Path, tmp_path: Path) -> None:
    bad = tmp_path / "run"
    shutil.copytree(run_dir, bad)
    rows = [json.loads(line) for line in (bad / "metrics.jsonl").read_text().splitlines()]
    for r in rows:
        if r["step"] == 55:
            r["births"] += 1
    (bad / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ResimMismatch, match="step 55"):
        labels.study(bad, **STUDY, log=lambda _: None)


def test_study_rejects_bad_parameters(run_dir: Path) -> None:
    with pytest.raises(ValueError, match=r"multiple of rules\.every"):
        labels.study(run_dir, **{**STUDY, "cohort_every": 12})
    with pytest.raises(ValueError, match="no cohort fits"):
        labels.study(run_dir, **{**STUDY, "steps": 100})
    with pytest.raises(ValueError, match="arch_min_obs"):
        labels.study(run_dir, **{**STUDY, "arch_min_obs": 61})
    with pytest.raises(ValueError, match="n_perm"):
        labels.study(run_dir, **{**STUDY, "n_perm": 0})
    with pytest.raises(FileNotFoundError):
        labels.study(run_dir, **{**STUDY, "start": 51})  # no checkpoint there


def test_study_uses_a_valid_sidecar_and_never_one_of_other_rules(tiny_cfg: Config, tmp_path: Path) -> None:
    path = _make_run(tiny_cfg, tmp_path / "observing", "game.observe=true")
    quick = {**STUDY, "steps": 280, "n_perm": 5}  # one cohort at 256
    rep = labels.study(path, **quick, log=lambda _: None)
    assert rep["observer_start"] == {"step": 50, "source": "sidecar", "note": None}
    other = dataclasses.replace(RULES_V1, arch_k=5, arch_warm=96)
    rep = labels.study(path, **quick, rules=other, log=lambda _: None)
    assert rep["observer_start"]["source"] == "fresh" and "rules_digest" in rep["observer_start"]["note"]
    assert rep["rules_digest"] == other.digest() != RULES_V1.digest()


def test_study_with_the_version_1_rules_carries_the_g1_pin(run_dir: Path) -> None:
    """``RULES_V1`` stays a rule set the tool can run: the report names the G1 digest and the version-1
    values, and has the documented shape (the version-2 rules are the default of the other tests)."""
    rep = labels.study(run_dir, **STUDY, rules=RULES_V1, log=lambda _: None)
    assert set(rep) == TOP_KEYS and rep["rows_verified"] > 0
    assert rep["rules_digest"] == "156ad4963b84ffe9636e9ba4fdd7592632c669bf6e490687d8eb2b38b9aec009"
    assert rep["rules_digest"] == RULES_V1.digest() != RULES_V2.digest()
    assert rep["params"]["rules"] == dataclasses.asdict(RULES_V1) and rep["params"]["rules"]["gatherer"] == (
        7,
        10,
    )
    assert list(rep["archetypes"]) == list(ARCHETYPE_NAMES) and list(rep["intents"]) == list(INTENT_NAMES)
    json.dumps(rep, allow_nan=False)
    default = labels.study(run_dir, **STUDY, log=lambda _: None)  # the default rules are version 2's
    assert default["rules_digest"] == RULES_V2.digest() and default["params"]["rules"]["bud_horizon"] == 10


def test_study_from_a_cached_state(run_dir: Path, tmp_path: Path) -> None:
    from alife.analysis import resim

    cache = tmp_path / "cache"
    resim.resimulate(run_dir, [55], cache, log=lambda _: None)
    rep = labels.study(
        run_dir, **{**STUDY, "start": 55, "steps": 295, "warmup": 200}, states_dir=cache, log=lambda _: None
    )
    assert rep["observer_start"]["source"] == "fresh" and rep["rows_verified"] > 0


def test_cli(run_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "out" / "report.json"
    argv = [str(run_dir), "--from", "50", "--steps", "280", "--warmup", "200", "--cohort-every", "40"]
    argv += ["--arch-horizon", "60", "--arch-min-obs", "20", "--n-perm", "5", "--out", str(out)]
    assert labels.main(argv) == 0
    rep = json.loads(out.read_text())
    assert set(rep) == TOP_KEYS and rep["params"]["n_perm"] == 5
    assert "wrote" in capsys.readouterr().out
    assert labels.main([*argv[:-2], "--out", str(tmp_path / "r2.json"), "--cohort-every", "12"]) == 2
    with pytest.raises(SystemExit) as e:  # the report must not land in the run directory
        labels.main([*argv[:-2], "--out", str(run_dir / "report.json")])
    assert e.value.code == 2 and not (run_dir / "report.json").exists()


# --- the tables of a real study against a numpy reference ---------------------------------------------------


def test_cohort_tables_equal_a_numpy_reference(biased_run_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """On real steps the device trackers' tables (totals, observed steps, Rg²) equal an independent numpy
    tally of the same steps."""
    cap: dict[str, dict[str, np.ndarray]] = {}
    real_a, real_i = labels.score_archetypes, labels.score_intents

    def spy_a(tab: Mapping[str, np.ndarray], **kw: Any) -> Any:
        cap["a"] = dict(tab)
        return real_a(tab, **kw)

    def spy_i(tab: Mapping[str, np.ndarray], **kw: Any) -> Any:
        cap["i"] = dict(tab)
        return real_i(tab, **kw)

    monkeypatch.setattr(labels, "score_archetypes", spy_a)
    monkeypatch.setattr(labels, "score_intents", spy_i)
    horizon, min_obs, ih = 60, 20, 10
    # the study ends exactly one horizon after its only cohort (256 + 60): that cohort must still be taken
    labels.study(biased_run_dir, **{**STUDY, "steps": 266, "n_perm": 2}, log=lambda _: None)

    rd = RunDir.open(biased_run_dir)
    cfg = rd.config()
    size = cfg.world.size
    state = load_checkpoint(rd.checkpoint_at(50), cfg)
    fn = make_step(cfg, observe=True)
    obs, _ = Observer.open(rd, state, cfg)
    t, T = 50, 256
    snap: dict[str, np.ndarray] = {}
    rec: list[dict[str, np.ndarray]] = []
    while t < T + horizon:
        kin_prev = to_numpy(obs.o.slot_kin)
        state, out = fn(state)
        obs.update(state, out.acts, t + 1)
        mx.eval(state, out.base(), *obs.arrays())
        t += 1
        if snap:
            rec.append(
                {
                    "alive": to_numpy(state.alive),
                    "uid": to_numpy(state.uid),
                    "pos": to_numpy(state.pos).astype(np.float64),
                    "harvest": to_numpy(state.last_harvest),
                    "flags": to_numpy(out.acts.flags).astype(np.int64),
                    "thrust": np.abs(to_numpy(out.acts.thrust).astype(np.int64)),
                    "defend": to_numpy(out.acts.defend).astype(np.int64),
                    "emit": to_numpy(out.acts.emit).astype(np.int64),
                    "out": to_numpy(out.acts.share_out).astype(np.int64),
                    "got": to_numpy(out.acts.share_got).astype(np.int64),
                    "kin": kin_prev,
                }
            )
        if t == T:
            lab = obs.host_labels()
            snap = {
                "uid": to_numpy(obs.o.uid),
                "pos": to_numpy(state.pos).astype(np.float64),
                "age": T - 1 - to_numpy(state.birth_step).astype(np.int64),
                "primary": lab["arch_primary"],
                "arch_match": lab["arch_match"],
                "label": lab["intent"],
                "intent_match": lab["intent_match"],
            }
    pool_a, pool_i = snap["primary"] != 0, (snap["intent_match"] & EVAL_BIT) != 0
    member = pool_a | pool_i
    assert pool_a.sum() > 50 and pool_i.sum() > 50
    cuid = np.where(member, snap["uid"], -1)

    n, nd = snap["uid"].size, snap["pos"].shape[1]
    ok = member.copy()
    n_ok = np.zeros(n, np.int64)
    col = {k: np.zeros(n, np.int64) for k in labels.OUTCOMES}
    at_intent: dict[str, np.ndarray] = {}
    last, u = snap["pos"].copy(), np.zeros((n, nd))
    path = np.zeros((len(rec), n, nd))
    oks = np.zeros((len(rec), n), bool)
    for k, r in enumerate(rec, start=1):
        ok = ok & r["alive"] & (r["uid"] == cuid)
        n_ok += ok
        f = r["flags"]
        for name, v in {
            "bud": (f & ActFlag.BUDDED.value) != 0,
            "attack": (f & ActFlag.ATTACKED.value) != 0,
            "formed": (f & ActFlag.FORMED.value) != 0,
            "thrust": r["thrust"],
            "defend": r["defend"],
            "emit": r["emit"],
            "harvest": r["harvest"],
            "care": np.where(r["kin"], r["out"], 0).sum(axis=1),
            "out": r["out"].sum(axis=1),
            "got": r["got"].sum(axis=1),
        }.items():
            col[name] += np.where(ok, v, 0)
        u = u + np.where(ok[:, None], _np_min_image(r["pos"] - last, size), 0.0)
        last = np.where(ok[:, None], r["pos"], last)
        path[k - 1], oks[k - 1] = u, ok
        if k == ih:
            at_intent = {"n_ok": n_ok.copy(), **{name: v.copy() for name, v in col.items()}}

    def rg2(rows: np.ndarray, first: int) -> np.ndarray:
        out = np.zeros(rows.size)
        for j, i in enumerate(rows):
            pts = path[:first][oks[:first, i], i]
            if pts.size:
                out[j] = ((pts - pts.mean(axis=0)) ** 2).sum(axis=1).mean()
        return out

    ra, ri = np.flatnonzero(pool_a), np.flatnonzero(pool_i)
    a = cap["a"]
    assert a["cohort"].tolist() == [0] * ra.size
    assert np.array_equal(a["n_ok"], n_ok[ra]) and a["n_ok"].max() <= horizon
    assert (a["n_ok"] < horizon).any() and (a["n_ok"] == horizon).any()  # both censored and complete rows
    for name in ("harvest", "attack", "out", "got"):
        assert np.array_equal(a[name], col[name][ra]), name
    assert a["harvest"].sum() > 0 and a["out"].sum() > 0
    assert np.array_equal(a["age"], snap["age"][ra]) and np.array_equal(a["primary"], snap["primary"][ra])
    assert np.array_equal(a["match"], snap["arch_match"][ra])
    np.testing.assert_allclose(a["rg2_full"], rg2(ra, horizon), rtol=2e-3, atol=1e-5)
    scored = a["n_ok"] >= min_obs
    np.testing.assert_allclose(a["rg2_min"][scored], rg2(ra, min_obs)[scored], rtol=2e-3, atol=1e-5)

    i = cap["i"]
    assert np.array_equal(i["complete"], at_intent["n_ok"][ri] >= ih)
    for name in ("bud", "attack", "formed", "thrust", "defend", "emit", "harvest", "care"):
        assert np.array_equal(i[name], at_intent[name][ri]), name
    for name in ("harvest", "thrust", "care", "attack", "bud", "formed", "emit", "defend"):
        assert i[name].sum() > 0, name  # the biased run exercises every column
    assert np.array_equal(i["label"], snap["label"][ri]) and np.array_equal(
        i["match"], snap["intent_match"][ri]
    )
