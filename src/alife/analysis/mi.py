"""Mutual-information estimators with shuffle nulls, numpy only. Results are in bits.

- ``mi_knn``: Kraskov–Stögbauer–Grassberger estimator 1 on tie-broken ranks. Both variables are first mapped
  to ranks with ties broken at random (``rank``). MI is invariant under strictly monotone maps, and random
  tie-breaking preserves it exactly: within a tie the order is independent noise, and the original value is
  recovered from the rank, so ``I(rank X; rank Y) = I(X; Y)``. For an integer variable this is the same as
  dithering it with ``U(−½, ½)``. It turns the zero-inflated, integer-valued signals and targets of the
  simulation (deposit quanta, counts, booleans) into continuous ones, where KSG is consistent; on such data
  the discrete-continuous estimator of Gao et al. (2017) was biased upward by about 0.07 bits at n = 3000
  (independent inputs), dithered KSG by less than 0.001.
- ``mi_binned_mm``: plug-in MI on equal-frequency bins with the Miller–Madow correction, a cross-check.
- ``perm_test``: a shuffle null (permute ``x`` across samples, optionally within strata), with a z-score and
  a permutation p-value.
- ``cmi_strata``: ``I(X; Y | S) = Σ_s p(s) I(X; Y | S = s)`` for a discrete conditioning variable.

Both variables are one-dimensional. The kNN search is a chunked brute force, O(n²) time and O(n·chunk)
memory, which suits the few thousand samples used per estimate.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable
from typing import NamedTuple

import numpy as np

LOG2E = 1.0 / math.log(2.0)
_EULER = 0.5772156649015329


def digamma_int(n: np.ndarray) -> np.ndarray:
    """``ψ(n)`` for positive integers: ``−γ + Σ_{j<n} 1/j``, exact to float64 rounding."""
    n = np.asarray(n, dtype=np.int64)
    if n.size and n.min() < 1:
        raise ValueError("digamma_int needs positive integers")
    top = int(n.max(initial=1))
    harmonic = np.concatenate([[0.0], np.cumsum(1.0 / np.arange(1, top, dtype=np.float64))])
    return harmonic[n - 1] - _EULER


def rank(x: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    """Ranks as float64 (0-based).

    Without ``rng``: average ranks, ties share one value. With ``rng``: ties are broken in random order, and
    every rank gets a draw of ``U(−0.25, 0.25)``, which keeps the order but takes the ranks off the integer
    lattice, whose many equal distances would break the kNN estimator's no-ties assumption.
    """
    x = np.asarray(x)
    if rng is not None:
        order = np.lexsort((rng.random(x.size), x))
        r = np.empty(x.size, dtype=np.float64)
        r[order] = np.arange(x.size) + rng.uniform(-0.25, 0.25, x.size)
        return r
    order = np.argsort(x, kind="stable")
    xs = x[order]
    new = np.concatenate([[True], xs[1:] != xs[:-1]])
    group = np.cumsum(new) - 1
    first = np.flatnonzero(new)
    last = np.concatenate([first[1:], [x.size]]) - 1
    r = np.empty(x.size, dtype=np.float64)
    r[order] = (0.5 * (first + last))[group]
    return r


def _kth_joint_distance(x: np.ndarray, y: np.ndarray, k: int, chunk: int) -> np.ndarray:
    """Max-norm distance from each point to its k-th nearest other point in the (x, y) plane."""
    n = x.size
    out = np.empty(n, dtype=np.float64)
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        d = np.maximum(np.abs(x[a:b, None] - x[None, :]), np.abs(y[a:b, None] - y[None, :]))
        d[np.arange(b - a), np.arange(a, b)] = np.inf  # exclude self
        out[a:b] = np.partition(d, k - 1, axis=1)[:, k - 1]
    return out


def _count_strict(sorted_v: np.ndarray, v: np.ndarray, r: np.ndarray) -> np.ndarray:
    """``#{j : |v_j − v_i| < r_i}`` including ``i`` itself, from the sorted values."""
    return np.searchsorted(sorted_v, v + r, side="left") - np.searchsorted(sorted_v, v - r, side="right")


def mi_knn(
    x: np.ndarray, y: np.ndarray, k: int = 10, *, ranked: bool = False, seed: int = 0, chunk: int = 1024
) -> float:
    """KSG estimator 1 of ``I(X; Y)`` in bits, on tie-broken ranks. May be slightly negative.

    ``I = ψ(k) + ψ(n) − ⟨ψ(n_x + 1) + ψ(n_y + 1)⟩``, with ``n_x``, ``n_y`` the other points strictly inside
    the max-norm distance to the k-th joint neighbour. The inputs are ranked with tie-breaking drawn from
    ``seed``, unless ``ranked`` says the caller already did that (so a permutation test ranks once).
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n = x.size
    if y.size != n:
        raise ValueError(f"x and y differ in length ({n} vs {y.size})")
    if n <= k:
        raise ValueError(f"need more than k = {k} samples, got {n}")
    if not ranked:
        rng = np.random.default_rng(seed)
        x, y = rank(x, rng), rank(y, rng)
    rho = _kth_joint_distance(x, y, k, chunk)
    nx = _count_strict(np.sort(x), x, rho)  # includes the point itself: n_x + 1
    ny = _count_strict(np.sort(y), y, rho)
    xi = digamma_int(np.array([k, n])).sum() - digamma_int(nx) - digamma_int(ny)
    return float(xi.mean() * LOG2E)


def _entropy_mm(labels: np.ndarray) -> float:
    """Plug-in entropy (nats) of integer labels plus the Miller–Madow term ``(m − 1)/(2n)``."""
    cnt = np.unique(labels, axis=0, return_counts=True)[1]
    n = cnt.sum()
    p = cnt / n
    return float(-(p * np.log(p)).sum() + (cnt.size - 1) / (2.0 * n))


def quantile_bins(x: np.ndarray, bins: int) -> np.ndarray:
    """Equal-frequency bin labels in ``[0, bins)``; tied values always share a bin."""
    r = rank(x)
    return np.minimum((r * bins / x.size).astype(np.int64), bins - 1)


def mi_binned_mm(x: np.ndarray, y: np.ndarray, bins: int = 8) -> float:
    """Plug-in ``I(X; Y)`` in bits on equal-frequency bins, each entropy Miller–Madow corrected."""
    bx, by = quantile_bins(np.asarray(x), bins), quantile_bins(np.asarray(y), bins)
    joint = np.stack([bx, by], axis=1)
    return (_entropy_mm(bx) + _entropy_mm(by) - _entropy_mm(joint)) * LOG2E


def cmi_strata(
    x: np.ndarray,
    y: np.ndarray,
    strata: np.ndarray,
    estimator: Callable[[np.ndarray, np.ndarray], float],
    min_size: int,
) -> tuple[float, float]:
    """``(Σ_s w_s I(X; Y | S = s), coverage)`` over strata with at least ``min_size`` samples.

    ``w_s`` is the stratum's share of the covered samples and ``coverage`` the covered share of all samples.
    Returns ``(nan, 0)`` when no stratum is large enough.
    """
    _, inv, cnt = np.unique(strata, return_inverse=True, return_counts=True)
    keep = np.flatnonzero(cnt >= min_size)
    covered = int(cnt[keep].sum())
    if covered == 0:
        return math.nan, 0.0
    total = 0.0
    for s in keep:
        rows = inv == s
        total += cnt[s] * estimator(x[rows], y[rows])
    return total / covered, covered / x.size


class PermResult(NamedTuple):
    mi: float  # bits
    null_mean: float
    null_sd: float
    z: float  # (mi − null_mean) / null_sd; nan when the null has no spread
    p: float  # (1 + #{null ≥ mi}) / (1 + n_perm)
    n: int  # samples


def permute_within(rng: np.random.Generator, n: int, strata: np.ndarray | None) -> np.ndarray:
    """A random permutation of ``range(n)`` that keeps every index inside its stratum (all one if None)."""
    if strata is None:
        return rng.permutation(n)
    perm = np.arange(n)
    order = np.argsort(strata, kind="stable")
    s = strata[order]
    bounds = np.flatnonzero(np.concatenate([[True], s[1:] != s[:-1], [True]]))
    for a, b in itertools.pairwise(bounds):
        idx = order[a:b]
        perm[idx] = idx[rng.permutation(b - a)]
    return perm


def perm_test(
    stat: Callable[[np.ndarray, np.ndarray], float],
    x: np.ndarray,
    y: np.ndarray,
    n_perm: int,
    rng: np.random.Generator,
    *,
    strata: np.ndarray | None = None,
) -> PermResult:
    """``stat(x, y)`` against ``stat(x[π], y)`` for ``n_perm`` random ``π`` (within ``strata`` if given)."""
    mi = stat(x, y)
    null = np.array([stat(x[permute_within(rng, x.size, strata)], y) for _ in range(n_perm)])
    mean = float(null.mean()) if n_perm else math.nan
    sd = float(null.std(ddof=1)) if n_perm > 1 else math.nan
    z = (mi - mean) / sd if sd > 0 else math.nan
    p = (1.0 + float((null >= mi).sum())) / (1.0 + n_perm)
    return PermResult(mi, mean, sd, z, p, int(x.size))
