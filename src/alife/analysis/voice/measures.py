"""Information measures of the voice analysis and their shuffle nulls (numpy only; results in bits).

Variables are small integer codes. ``cmi_codes`` is the plug-in conditional mutual information with the
Miller-Madow correction per stratum (``mi.mi_discrete_mm`` is its unconditional form). A **family** is a
list of
(x, y, strata) pairs tested together: the statistic of a pair is ``I(x; y | strata)``, its null permutes ``x``
inside the strata (so every marginal and every dependence on the conditioning variable stays), and the family
p-value is that of the largest permutation z over the pairs (max-T), so that looking at many (context,
response) pairs is paid for. The pairs share the random keys of each permutation, which couples their nulls
the
way a common permutation of the rows would. The estimator and the null use the same statistic, so the
Miller-Madow and plug-in biases cancel in the *excess* ``mi - null_mean``, the effect size reported.

Nothing here reads a run or touches MLX.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, NamedTuple

import numpy as np

LOG2E = 1.0 / math.log(2.0)


class Pair(NamedTuple):
    """One tested pair: codes ``x`` in ``[0, nx)`` (permuted) and ``y`` in ``[0, ny)``, stratum ids."""

    name: str
    x: np.ndarray
    nx: int
    y: np.ndarray
    ny: int
    strata: np.ndarray  # (n,) non-negative ints; conditioning set and permutation block


def compact(ids: np.ndarray) -> tuple[np.ndarray, int]:
    """Stratum ids as ``0..m-1`` (order of first appearance not preserved: sorted unique values)."""
    u, inv = np.unique(np.asarray(ids), return_inverse=True)
    return inv.astype(np.int64).reshape(-1), int(u.size)


def combine(*parts: np.ndarray) -> np.ndarray:
    """One id per distinct combination of the integer arrays ``parts``."""
    key = np.zeros(len(parts[0]), dtype=np.int64)
    for p in parts:
        p = np.asarray(p).astype(np.int64)
        key = key * (int(p.max(initial=0)) + 1) + p
    return compact(key)[0]


def _entropy(cnt: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Plug-in entropy (nats) per row of ``cnt`` (S, m), ``n`` the row sums."""
    with np.errstate(divide="ignore", invalid="ignore"):
        p = cnt / n[:, None]
        t = np.where(cnt > 0, p * np.log(p), 0.0)
    return -t.sum(axis=1)


def counts3(x: np.ndarray, y: np.ndarray, nx: int, ny: int, sid: np.ndarray, ns: int) -> np.ndarray:
    """The (S, nx, ny) table of joint counts per stratum."""
    j = (sid.astype(np.int64) * nx + x.astype(np.int64)) * ny + y.astype(np.int64)
    return np.bincount(j, minlength=ns * nx * ny).reshape(ns, nx, ny)


def cmi_from_counts(c: np.ndarray, min_size: int = 1) -> tuple[float, float]:
    """``(I(X;Y|S) in bits, covered share)`` from the (S, nx, ny) counts; strata with fewer than ``min_size``
    rows are left out. Miller-Madow per stratum: the entropy of a stratum of ``n`` rows with ``m`` occupied
    cells is corrected by ``(m - 1) / (2n)``. ``(nan, 0)`` when no stratum is large enough."""
    n_s = c.sum(axis=(1, 2))
    keep = n_s >= max(min_size, 1)
    total = int(n_s.sum())
    if not keep.any():
        return math.nan, 0.0
    c, n_s = c[keep], n_s[keep].astype(np.float64)
    s = c.shape[0]
    px, py, pxy = c.sum(axis=2), c.sum(axis=1), c.reshape(s, -1)
    info = _entropy(px, n_s) + _entropy(py, n_s) - _entropy(pxy, n_s)
    mm = ((px > 0).sum(1) + (py > 0).sum(1) - (pxy > 0).sum(1) - 1) / (2.0 * n_s)
    return float(((info + mm) * n_s).sum() / n_s.sum() * LOG2E), float(n_s.sum() / max(total, 1))


def cmi_codes(
    x: np.ndarray, y: np.ndarray, nx: int, ny: int, strata: np.ndarray | None = None, min_size: int = 1
) -> float:
    """``I(X; Y | strata)`` in bits (plug-in, Miller-Madow per stratum); ``strata`` None is unconditional."""
    if strata is None:
        sid, ns = np.zeros(len(x), dtype=np.int64), 1
    else:
        sid, ns = compact(strata)
    return cmi_from_counts(counts3(x, y, nx, ny, sid, ns), min_size)[0]


def mi_codes(x: np.ndarray, y: np.ndarray, nx: int, ny: int) -> float:
    """Unconditional Miller-Madow ``I(X; Y)`` in bits of two code arrays."""
    return cmi_codes(x, y, nx, ny, None, 1)


def entropy_bits(x: np.ndarray, nx: int) -> float:
    """Miller-Madow entropy (bits) of a code array."""
    cnt = np.bincount(np.asarray(x, dtype=np.int64), minlength=nx)[None, :]
    n = np.array([float(cnt.sum())])
    return float((_entropy(cnt, n)[0] + ((cnt > 0).sum() - 1) / (2.0 * n[0])) * LOG2E)


def shuffle_within(strata: np.ndarray, u: np.ndarray, base: np.ndarray) -> np.ndarray:
    """A permutation of ``range(n)`` that keeps every index inside its stratum. ``u`` are n random keys in
    [0, 1) and ``base = argsort(strata, stable)``, computed once per stratum set."""
    rnd = np.argsort(strata.astype(np.float64) + u, kind="stable")
    perm = np.empty(len(strata), dtype=np.int64)
    perm[base] = rnd
    return perm


class FamilyResult(NamedTuple):
    names: list[str]
    mi: np.ndarray  # per pair, bits
    null_mean: np.ndarray
    null_sd: np.ndarray
    excess: np.ndarray  # mi - null_mean
    z: np.ndarray
    p: np.ndarray  # per pair, uncorrected: (1 + #{null >= mi}) / (1 + n_perm)
    best: int  # index of the largest z
    p_family: float  # max-T
    n: int
    coverage: np.ndarray  # share of rows in strata large enough

    def table(self) -> list[dict[str, Any]]:
        return [
            {
                "pair": nm,
                "mi": _f(self.mi[i]),
                "null_mean": _f(self.null_mean[i]),
                "null_sd": _f(self.null_sd[i]),
                "excess": _f(self.excess[i]),
                "z": _f(self.z[i]),
                "p": _f(self.p[i]),
                "coverage": _f(self.coverage[i]),
            }
            for i, nm in enumerate(self.names)
        ]

    def best_summary(self) -> dict[str, Any]:
        i = self.best
        return {
            "pair": self.names[i],
            "mi": _f(self.mi[i]),
            "null_mean": _f(self.null_mean[i]),
            "null_sd": _f(self.null_sd[i]),
            "excess": _f(self.excess[i]),
            "z": _f(self.z[i]),
            "p_pair": _f(self.p[i]),
            "p_family": _f(self.p_family),
            "n": self.n,
            "n_pairs": len(self.names),
        }


def _f(v: Any) -> float | None:
    v = float(v)
    return v if math.isfinite(v) else None


def family_test(
    pairs: Sequence[Pair], n_perm: int, rng: np.random.Generator, min_size: int = 1
) -> FamilyResult:
    """Observed ``I(x; y | strata)`` of every pair, its permutation null, the per-pair z and p, and the max-T
    family p. Pairs on the same rows (all of one length); ``n_perm`` >= 2."""
    if not pairs:
        raise ValueError("a family needs at least one pair")
    if n_perm < 2:
        raise ValueError("a permutation null needs n_perm >= 2")
    n = len(pairs[0].x)
    k = len(pairs)
    prep = []
    cache: dict[int, tuple[np.ndarray, np.ndarray, int]] = {}
    for p in pairs:
        if len(p.x) != n or len(p.y) != n or len(p.strata) != n:
            raise ValueError(f"pair {p.name}: arrays differ in length")
        key = id(p.strata)
        if key not in cache:
            sid, ns = compact(p.strata)
            cache[key] = (sid, np.argsort(sid, kind="stable"), ns)
        sid, base, ns = cache[key]
        prep.append((sid, base, ns))
    obs = np.empty(k)
    cov = np.empty(k)
    for i, (p, (sid, _, ns)) in enumerate(zip(pairs, prep, strict=True)):
        obs[i], cov[i] = cmi_from_counts(counts3(p.x, p.y, p.nx, p.ny, sid, ns), min_size)
    null = np.empty((n_perm, k))
    for it in range(n_perm):
        u = rng.random(n)
        done: dict[int, np.ndarray] = {}
        for i, (p, (sid, base, ns)) in enumerate(zip(pairs, prep, strict=True)):
            key = id(p.strata)
            if key not in done:
                done[key] = shuffle_within(sid, u, base)
            xp = p.x[done[key]]
            null[it, i] = cmi_from_counts(counts3(xp, p.y, p.nx, p.ny, sid, ns), min_size)[0]
    mean = null.mean(axis=0)
    sd = null.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(sd > 0, (obs - mean) / sd, np.nan)
        zn = np.where(sd > 0, (null - mean) / sd, np.nan)
    p_pair = (1.0 + (null >= obs[None, :]).sum(axis=0)) / (1.0 + n_perm)
    zf = np.where(np.isfinite(z), z, -np.inf)
    best = int(np.argmax(zf))
    zmax_null = np.nanmax(np.where(np.isfinite(zn), zn, -np.inf), axis=1)
    p_fam = float((1.0 + (zmax_null >= zf[best]).sum()) / (1.0 + n_perm))
    return FamilyResult([p.name for p in pairs], obs, mean, sd, obs - mean, z, p_pair, best, p_fam, n, cov)


def block_excess(
    pair: Pair,
    block: np.ndarray,
    n_blocks: int,
    n_perm: int,
    rng: np.random.Generator,
    min_size: int = 1,
    use: Sequence[int] | None = None,
) -> np.ndarray:
    """``mi - null_mean`` of one pair, per block (NaN for a block without enough rows). ``block`` is the block
    index of every row (-1 = none); ``use`` restricts the blocks computed (the others stay NaN)."""
    out = np.full(n_blocks, np.nan)
    for b in range(n_blocks) if use is None else use:
        m = block == b
        if int(m.sum()) < max(2 * min_size, 4):
            continue
        sid, ns = compact(pair.strata[m])
        base = np.argsort(sid, kind="stable")
        x, y = pair.x[m], pair.y[m]
        obs = cmi_from_counts(counts3(x, y, pair.nx, pair.ny, sid, ns), min_size)[0]
        if not math.isfinite(obs):
            continue
        nul = []
        for _ in range(n_perm):
            xp = x[shuffle_within(sid, rng.random(len(x)), base)]
            nul.append(cmi_from_counts(counts3(xp, y, pair.nx, pair.ny, sid, ns), min_size)[0])
        out[b] = obs - float(np.nanmean(nul))
    return out


# --- discretisation ---------------------------------------------------------------------------------------


def quantile_edges(x: np.ndarray, levels: int) -> np.ndarray:
    """Distinct cut points of ``levels`` equal-frequency bins (fewer when the data are tied)."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0 or levels < 2:
        return np.empty(0)
    return np.unique(np.quantile(x, np.arange(1, levels) / levels))


def code_by_edges(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """The number of cut points strictly below each value: codes ``0..len(edges)`` (int8)."""
    return np.searchsorted(edges, np.asarray(x, dtype=np.float64), side="left").astype(np.int8)


def symbol_threshold(voice: np.ndarray) -> float:
    """The median of the loudest channel over the non-silent calls (the intensity split of ``symbols``)."""
    top = voice.max(axis=1)
    top = top[top > 0]
    return float(np.median(top)) if top.size else 0.0


def symbols(voice: np.ndarray, thr: float) -> np.ndarray:
    """The call type of each row of a ``(n, W)`` uint8 voice: 0 = silent, else ``1 + 2 * argmax_c + [loudest
    channel above thr]``; alphabet ``1 + 2W``."""
    voice = np.asarray(voice)
    top = voice.max(axis=1)
    sym = 1 + 2 * voice.argmax(axis=1) + (top > thr)
    return np.where(top > 0, sym, 0).astype(np.int8)


def alphabet(width: int) -> int:
    return 1 + 2 * width


def nanmean(a: np.ndarray, axis: int = 0) -> np.ndarray:
    """``np.nanmean`` that is NaN, without a warning, where nothing is finite."""
    a = np.asarray(a, dtype=np.float64)
    ok = np.isfinite(a)
    n = ok.sum(axis=axis)
    return np.where(n > 0, np.where(ok, a, 0.0).sum(axis=axis) / np.maximum(n, 1), np.nan)


def jensen_shannon(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence in bits of two count or probability vectors (0 for two empty ones)."""
    p, q = np.asarray(p, dtype=np.float64), np.asarray(q, dtype=np.float64)
    if p.sum() <= 0 or q.sum() <= 0:
        return math.nan
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        s = a > 0
        return float((a[s] * np.log(a[s] / b[s])).sum())

    return 0.5 * (kl(p, m) + kl(q, m)) * LOG2E
