"""Mutual-information estimators of ``alife.analysis.mi`` on synthetic data with known MI."""

from __future__ import annotations

import math

import numpy as np
import pytest

from alife.analysis import mi

N = 1500


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(12345)


def test_digamma_int() -> None:
    euler = 0.5772156649015329
    np.testing.assert_allclose(mi.digamma_int(np.array([1, 2, 3])), [-euler, 1 - euler, 1.5 - euler])
    with pytest.raises(ValueError):
        mi.digamma_int(np.array([0]))


def test_rank_ties(rng: np.random.Generator) -> None:
    x = np.array([3.0, 1.0, 3.0, 2.0])
    np.testing.assert_array_equal(mi.rank(x), [2.5, 0.0, 2.5, 1.0])
    r = mi.rank(x, rng)  # ties broken at random: a strictly increasing map of the order, off the lattice
    assert np.unique(r).size == 4
    assert r[1] < r[3] < min(r[0], r[2])
    assert np.all(np.abs(r - np.round(r)) <= 0.25)


def test_independent_continuous_is_zero(rng: np.random.Generator) -> None:
    x, y = rng.normal(size=N), rng.normal(size=N)
    assert abs(mi.mi_knn(x, y)) < 0.03
    assert abs(mi.mi_binned_mm(x, y)) < 0.02


def test_independent_zero_inflated_integers_are_zero(rng: np.random.Generator) -> None:
    """Emit-like quanta (half exactly 0, the rest on 255 levels) against a count: no tie bias."""
    z = rng.normal(size=N)
    q = np.where(z > 0, np.floor(255 * np.tanh(z)), 0.0)
    count = rng.binomial(8, 0.3, N).astype(float)
    flag = (rng.random(N) < 0.1).astype(float)
    assert abs(mi.mi_knn(q, count)) < 0.03
    assert abs(mi.mi_knn(q, flag)) < 0.03


def test_gaussian_matches_closed_form(rng: np.random.Generator) -> None:
    r = 0.6
    x = rng.normal(size=N)
    y = r * x + math.sqrt(1 - r * r) * rng.normal(size=N)
    assert mi.mi_knn(x, y) == pytest.approx(-0.5 * math.log2(1 - r * r), abs=0.05)


@pytest.mark.parametrize("levels", [2, 4, 8])
def test_deterministic_channel_is_the_entropy(rng: np.random.Generator, levels: int) -> None:
    d = rng.integers(0, levels, N).astype(float)
    h = math.log2(levels)
    assert mi.mi_knn(d, d) == pytest.approx(h, rel=0.08)
    assert mi.mi_knn(d, 3.0 * d - 1.0) == pytest.approx(h, rel=0.08)  # any injective relabelling
    assert mi.mi_binned_mm(d, d, bins=levels) == pytest.approx(h, abs=0.02)


def test_monotone_invariance(rng: np.random.Generator) -> None:
    x = rng.normal(size=N)
    y = x + rng.normal(size=N)
    assert mi.mi_knn(np.exp(x), y**3) == pytest.approx(mi.mi_knn(x, y), abs=1e-12)


def test_perm_test(rng: np.random.Generator) -> None:
    x = rng.normal(size=800)
    dep = mi.perm_test(mi.mi_knn, x, x + rng.normal(size=800), 20, rng)
    assert dep.z > 10 and dep.p == pytest.approx(1 / 21) and dep.n == 800
    ind = mi.perm_test(mi.mi_knn, x, rng.normal(size=800), 20, rng)
    assert abs(ind.z) < 4 and ind.p > 1 / 21


def test_permute_within_keeps_strata(rng: np.random.Generator) -> None:
    strata = rng.integers(0, 5, 200)
    perm = mi.permute_within(rng, 200, strata)
    assert sorted(perm) == list(range(200))
    np.testing.assert_array_equal(strata[perm], strata)
    assert (perm != np.arange(200)).any()


def test_cmi_removes_a_shared_stratum(rng: np.random.Generator) -> None:
    """X and Y both depend on the stratum only: large I(X; Y), I(X; Y | S) about 0."""
    s = rng.integers(0, 4, 4 * N)
    x = s + 0.3 * rng.normal(size=s.size)
    y = s + 0.3 * rng.normal(size=s.size)
    assert mi.mi_knn(x, y) > 1.0
    cmi, cov = mi.cmi_strata(x, y, s, mi.mi_knn, min_size=50)
    assert abs(cmi) < 0.03 and cov == 1.0
    assert math.isnan(mi.cmi_strata(x, y, s, mi.mi_knn, min_size=10 * N)[0])
