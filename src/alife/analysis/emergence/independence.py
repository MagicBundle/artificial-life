"""Independence of two hash-salted streams (edge case E18: no spurious "language" from shared hash salts).

Two features that draw pseudo-random values from the same hash with the same salt would be perfectly
correlated and look like communication. ``hash_stream`` is the stateless stream ``splitmix64(key ^ salt)``
the engine's observers use; ``independence`` tests two value arrays on the same keys: Pearson correlation with
its z (the null sd is ``1/sqrt(n)``), the mutual information of their quantile bins against a permutation
null, and the fraction of 64 bit positions whose agreement departs from 1/2 by more than 4 sd. Numpy only; the
language-score adapter of the comm scorecard (E6) calls it on the streams it compares.
"""

from __future__ import annotations

import math
from typing import Any, Final

import numpy as np

from alife.analysis.phylo import splitmix64

INDEPENDENCE_VERSION: Final = 1
Z_ROT: Final = 5.5  # per bit-pair z for the 4096 rotation pairs (family false rate about 1e-3)
BITS: Final = np.uint64(1) << np.arange(64, dtype=np.uint64)


def hash_stream(keys: np.ndarray, salt: int) -> np.ndarray:
    """The uint64 stream of ``keys`` under ``salt`` (SplitMix64 finaliser of ``key ^ salt``)."""
    return splitmix64(np.asarray(keys, dtype=np.uint64), salt)


def _quantile_bins(x: np.ndarray, bins: int) -> np.ndarray:
    r = np.argsort(np.argsort(x, kind="stable"), kind="stable")
    return (r * bins // len(x)).astype(np.int64)


def mutual_information(a: np.ndarray, b: np.ndarray, bins: int = 8) -> float:
    """Mutual information (nats) of the quantile-binned values."""
    ia, ib = _quantile_bins(a, bins), _quantile_bins(b, bins)
    j = np.bincount(ia * bins + ib, minlength=bins * bins).reshape(bins, bins).astype(np.float64)
    j /= j.sum()
    pa, pb = j.sum(1, keepdims=True), j.sum(0, keepdims=True)
    s = j > 0
    return float((j[s] * np.log(j[s] / (pa @ pb)[s])).sum())


def independence(
    a: np.ndarray, b: np.ndarray, *, bins: int = 8, n_perm: int = 200, seed: int = 0, z_max: float = 4.0
) -> dict[str, Any]:
    """Test two streams on the same keys for independence. ``independent`` is False when |r| z exceeds
    ``z_max``, the MI exceeds every permutation, or (uint64 inputs) a bit position agrees with the other
    stream's bit far from half the time."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.ndim != 1 or len(a) < 100:
        raise ValueError("two equal-length 1-D streams of at least 100 values are needed")
    n = len(a)
    af, bf = a.astype(np.float64), b.astype(np.float64)
    r = float(np.corrcoef(af, bf)[0, 1]) if af.std() > 0 and bf.std() > 0 else 1.0
    z = r * math.sqrt(n)
    mi = mutual_information(af, bf, bins)
    rng = np.random.default_rng(seed)
    null = np.array([mutual_information(af, rng.permutation(bf), bins) for _ in range(n_perm)])
    p_mi = float((1 + (null >= mi).sum()) / (n_perm + 1))
    bit_dev = bit_rot = None
    if a.dtype == np.uint64 and b.dtype == np.uint64:
        agree = np.array([((a & m) == (b & m)).mean() for m in BITS])
        bit_dev = int((np.abs(agree - 0.5) * 2 * math.sqrt(n) > z_max).sum())
        # bit k of one stream against bit (k + r) of the other, for every rotation r: a rotated copy (or a
        # stream sharing words with a shift) agrees perfectly at one offset. 4096 pairs, so a stricter z.
        bit_rot = 0
        for rot in range(64):
            x = a ^ (b if rot == 0 else ((b << np.uint64(rot)) | (b >> np.uint64(64 - rot))))
            frac = np.array([((x >> np.uint64(k)) & np.uint64(1)).mean() for k in range(64)])
            bit_rot += int((np.abs(frac - 0.5) * 2 * math.sqrt(n) > Z_ROT).sum())
    ok = abs(z) <= z_max and p_mi > 0.005 and not bit_dev and not bit_rot
    return {
        "version": INDEPENDENCE_VERSION,
        "n": n,
        "r": r,
        "z": z,
        "mi": mi,
        "mi_null_mean": float(null.mean()),
        "mi_p": p_mi,
        "bit_positions_off_half": bit_dev,
        "bit_rotation_pairs_off_half": bit_rot,
        "independent": bool(ok),
    }


def salts_independent(keys: np.ndarray, salts: list[int], **kw: Any) -> dict[tuple[int, int], dict[str, Any]]:
    """Pairwise ``independence`` of the hash streams of ``keys`` under every pair of ``salts``; a repeated
    salt gives identical streams and fails."""
    out = {}
    for i, s in enumerate(salts):
        for t in salts[i + 1 :]:
            out[(s, t)] = independence(hash_stream(keys, s), hash_stream(keys, t), **kw)
    return out
