"""Statistics of the emergence suite (DESIGN §44.8): paired block intervals, the seed rule, multiplicity.

Numpy and the standard library only (scipy is not a dependency). The interval machinery is ``analysis.w2``'s
(``t_interval``, ``Blocking``, ``classify``, ``paired``); this module adds what the arm comparison needs on
top of it: one-sided p-values from the t distribution, the lag-1 autocorrelation check that widens the blocks,
the seed-twin noise floor, Stouffer's combination of the seeds, and the Holm and Benjamini-Hochberg
corrections. Nothing here reads a run.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from alife.analysis import w2
from alife.analysis.w2 import Blocking, block_mean, block_ratio, classify, paired, t_interval

__all__ = [
    "Blocking",
    "bh",
    "block_mean",
    "block_ratio",
    "classify",
    "holm",
    "lag1",
    "merge_pairs",
    "noise_floor",
    "norm_isf",
    "norm_sf",
    "paired",
    "paired_test",
    "run_blocking",
    "seed_call",
    "stouffer",
    "t_interval",
    "t_sf",
    "window_blocking",
]

ALPHA = 0.05  # family-wise level of the confirmatory calls (§44.8)
BH_Q = 0.10  # false discovery rate of the across-feature context line
WIDEN_AT = (
    0.3  # |lag-1 autocorrelation| of the paired differences above which the blocks are doubled in width
)


def norm_sf(z: float) -> float:
    """The standard normal survival function ``P(Z > z)``."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def norm_isf(p: float) -> float:
    """The ``z`` with ``P(Z > z) = p`` (Acklam's rational start, two Halley steps; absolute error below"
    "1e-9 for p >= 1e-6 and about 5e-8 at p = 1e-12)."""
    if not 0.0 < p < 1.0:
        return math.inf if p <= 0.0 else -math.inf
    q = 1.0 - p  # work with the lower-tail quantile of q
    a = (
        -39.69683028665376,
        220.9460984245205,
        -275.9285104469687,
        138.357751867269,
        -30.66479806614716,
        2.506628277459239,
    )
    b = (-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572)
    c = (
        -0.007784894002430293,
        -0.3223964580411365,
        -2.400758277161838,
        -2.549732539343734,
        4.374664141464968,
        2.938163982698783,
    )
    d = (0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416)
    lo = 0.02425
    if q < lo:
        t = math.sqrt(-2.0 * math.log(q))
        x = (((((c[0] * t + c[1]) * t + c[2]) * t + c[3]) * t + c[4]) * t + c[5]) / (
            (((d[0] * t + d[1]) * t + d[2]) * t + d[3]) * t + 1.0
        )
    elif q <= 1.0 - lo:
        t = q - 0.5
        r = t * t
        x = (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
            * t
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        )
    else:
        t = math.sqrt(-2.0 * math.log(p))
        x = -(((((c[0] * t + c[1]) * t + c[2]) * t + c[3]) * t + c[4]) * t + c[5]) / (
            (((d[0] * t + d[1]) * t + d[2]) * t + d[3]) * t + 1.0
        )
    for _ in range(2):  # Halley refinement against the exact cdf
        e = 0.5 * math.erfc(-x / math.sqrt(2.0)) - q
        u = e * math.sqrt(2.0 * math.pi) * math.exp(x * x / 2.0)
        x -= u / (1.0 + x * u / 2.0)
    return x


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction of the incomplete beta function (modified Lentz)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 400):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    """The regularised incomplete beta function ``I_x(a, b)``."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_sf(t: float, df: int) -> float:
    """``P(T > t)`` for Student's t with ``df`` degrees of freedom."""
    if df < 1:
        return math.nan
    if not math.isfinite(t):
        return 0.0 if t > 0 else 1.0
    tail = 0.5 * _betainc(df / 2.0, 0.5, df / (df + t * t))
    return tail if t >= 0 else 1.0 - tail


# --- multiplicity and combination ---


def holm(p: Sequence[float]) -> list[float]:
    """Holm's step-down adjusted p-values, in the input order (NaN stays NaN and takes no part)."""
    v = np.asarray(p, dtype=np.float64)
    out = np.full(v.shape, np.nan)
    ok = np.flatnonzero(np.isfinite(v))
    order = ok[np.argsort(v[ok], kind="stable")]
    m, run = order.size, 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * v[i])
        out[i] = min(1.0, run)
    return out.tolist()


def bh(p: Sequence[float]) -> list[float]:
    """Benjamini-Hochberg adjusted p-values (q-values), in the input order (NaN stays NaN)."""
    v = np.asarray(p, dtype=np.float64)
    out = np.full(v.shape, np.nan)
    ok = np.flatnonzero(np.isfinite(v))
    order = ok[np.argsort(-v[ok], kind="stable")]  # descending
    m, run = order.size, 1.0
    for k, i in enumerate(order):
        rank = m - k
        run = min(run, v[i] * m / rank)
        out[i] = min(1.0, run)
    return out.tolist()


def stouffer(p: Sequence[float | None]) -> float | None:
    """Stouffer's combination of one-sided p-values (equal weights): ``Z = sum z_i / sqrt(k)``, the combined
    one-sided p. None if any p is missing (a seed that could not be evaluated cannot be dropped silently)."""
    if not len(p) or any(x is None or not math.isfinite(x) for x in p):
        return None
    z = [norm_isf(min(max(float(x), 1e-300), 1.0 - 1e-16)) for x in p]
    return norm_sf(sum(z) / math.sqrt(len(z)))


# --- blocks ---


def window_blocking(t0: int, t1: int, n: int = 8) -> Blocking:
    """``n`` equal blocks of a recorded window ``(t0, t1]`` (10 000 steps -> 8 blocks of 1250)."""
    if n < 1 or (t1 - t0) % n:
        raise ValueError(f"the window ({t0}, {t1}] does not split into {n} equal blocks")
    return Blocking(start=t0, transient=0, width=(t1 - t0) // n, n=n)


def run_blocking(start: int, *, transient: int = 20_000, width: int = 10_000, n: int = 8) -> Blocking:
    """The run-level blocks of the open-endedness endpoints (§44.8): 8 x 10 000 after a 20 000 transient."""
    return Blocking(start=start, transient=transient, width=width, n=n)


def lag1(d: np.ndarray) -> float | None:
    """Lag-1 autocorrelation of ``d`` over adjacent pairs of finite blocks (a NaN block breaks adjacency"
    "instead
    of joining its neighbours); None under 4 finite values or zero variance."""
    v = np.asarray(d, dtype=np.float64)
    ok = np.isfinite(v)
    if ok.sum() < 4:
        return None
    x = v - v[ok].mean()
    den = float((x[ok] ** 2).sum())
    if den <= 0.0:
        return None
    pair = ok[:-1] & ok[1:]
    return float((x[:-1][pair] * x[1:][pair]).sum() / den) if pair.any() else None


def merge_pairs(d: np.ndarray) -> np.ndarray:
    """Adjacent blocks merged in pairs (the mean of the finite blocks of a pair, NaN if both are NaN; an"
    "odd last
    block is dropped): doubles the width."""
    v = np.asarray(d, dtype=np.float64)
    k = v.size // 2
    if not k:
        return v[:0]
    w = v[: 2 * k].reshape(k, 2)
    ok = np.isfinite(w)
    n = ok.sum(axis=1)
    return np.where(n > 0, np.where(ok, w, 0.0).sum(axis=1) / np.maximum(n, 1), np.nan)


def paired_test(a: np.ndarray, b: np.ndarray, *, direction: int = 1, widen: bool = True) -> dict[str, Any]:
    """The paired block contrast ``direction * (a - b)`` of one seed (§44.8.1).

    Returns the ``t_interval`` of the differences plus ``d_z`` (mean / sd), ``r1`` (lag-1 autocorrelation of
    the differences), ``widened`` (|r1| > 0.3 doubled the block width and the interval was recomputed; the
    flag is reported, never silent), ``nan_blocks`` (blocks missing from the contrast), ``t`` and the
    one-sided ``p`` for "the mean is positive". The interval
    and p are None below two finite blocks.
    """
    d = direction * paired(a, b)
    r1 = lag1(d)
    widened = False
    if widen and r1 is not None and abs(r1) > WIDEN_AT and d.size >= 4:
        d, widened = merge_pairs(d), True
    iv = t_interval(d)
    out: dict[str, Any] = {
        **iv,
        "d_z": None,
        "r1": r1,
        "widened": widened,
        "nan_blocks": int((~np.isfinite(paired(a, b))).sum()),
        "t": None,
        "p": None,
    }
    if iv["sd"] is not None and iv["n"] >= 2:
        if iv["sd"] > 0:
            out["d_z"] = iv["mean"] / iv["sd"]
            out["t"] = iv["mean"] / (iv["sd"] / math.sqrt(iv["n"]))
            out["p"] = t_sf(out["t"], iv["n"] - 1)
        else:  # constant differences: the interval is a point
            out["p"] = 0.0 if iv["mean"] > 0 else 1.0
    return out


def noise_floor(twin_diffs: Sequence[np.ndarray]) -> float | None:
    """The seed-twin noise floor of an endpoint: the **median** magnitude of the twins' mean paired"
    "differences
    (``U0`` vs ``T0``, ``U5`` vs ``T5``, per seed; the W2 convention; the median so that one outlying twin
    does
    not lift the floor of every contrast). None without a finite twin contrast."""
    mags = []
    for d in twin_diffs:
        v = np.asarray(d, dtype=np.float64)
        v = v[np.isfinite(v)]
        if v.size:
            mags.append(abs(float(v.mean())))
    return float(np.median(mags)) if mags else None


def seed_call(per_seed: Sequence[Mapping[str, Any]], floor: float | None = None) -> dict[str, Any]:
    """The decision over seeds of one contrast (§44.8.3).

    ``per_seed`` are ``paired_test`` results with the direction already applied (positive = the feature
    increases the endpoint). ``label``: ``supported`` when ``w2.classify`` says every seed's interval excludes
    0 with one sign and the mean exceeds ``floor``, and that sign is positive; ``reversed`` when it is
    supported with a negative sign; ``mixed`` / ``not supported`` as in ``w2.classify``. ``p`` is Stouffer's
    combination of the seeds' one-sided p-values (a sign test across 3 seeds cannot go below 0.125).
    """
    ivs = [s if s.get("lo") is not None else None for s in per_seed]
    label = classify(ivs, floor)
    if label == "supported" and not all(iv["mean"] > 0 for iv in ivs if iv is not None):
        label = "reversed"
    return {
        "label": label,
        "p": stouffer([s.get("p") for s in per_seed]),
        "floor": floor,
        "mean": float(np.mean([s["mean"] for s in per_seed if s.get("mean") is not None]))
        if any(s.get("mean") is not None for s in per_seed)
        else None,
        "seeds": len(per_seed),
    }


assert w2.classify is classify  # one decision rule for the project
