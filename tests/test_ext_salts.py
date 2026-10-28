"""Hash salts of the feature extensions (DESIGN 43.3): frozen values, pairwise independent streams (E18)."""

from __future__ import annotations

import math
from itertools import combinations

import mlx.core as mx
import numpy as np

from alife import ops, rng

FROZEN = {
    "arms": 0x20D53604,
    "structures": 0x243C7C36,
    "bonds": 0x9E1A3B74,
    "comm": 0x6E00A046,
    "niche": 0x20622137,
}
N = 20000
Z_MAX = 4.5  # per statistic; the data are deterministic, so this is a fixed outcome, not a flake rate
Z_ROT = 6.5  # 1024 bit pairs per stream pair (one chance 5.8 seen at N=20000, 3.3 at N=200000)


def _mutual_information(a: np.ndarray, b: np.ndarray, bins: int = 8) -> float:
    def q(x: np.ndarray) -> np.ndarray:
        return (np.argsort(np.argsort(x, kind="stable"), kind="stable") * bins // len(x)).astype(np.int64)

    j = np.bincount(q(a) * bins + q(b), minlength=bins * bins).reshape(bins, bins).astype(np.float64)
    j /= j.sum()
    pa, pb = j.sum(1, keepdims=True), j.sum(0, keepdims=True)
    s = j > 0
    return float((j[s] * np.log(j[s] / (pa @ pb)[s])).sum())


def independence_report(a: np.ndarray, b: np.ndarray, n_perm: int = 100) -> dict:
    """Port of ``alife.analysis.emergence.independence.independence`` for uint32 streams on the same keys:
    Pearson z, mutual information of quantile bins against a permutation null, bit positions whose agreement
    departs from 1/2, and bit k of one against bit k+r of the other for every rotation r (32 x 32 pairs)."""
    n = len(a)
    af, bf = a.astype(np.float64), b.astype(np.float64)
    z = float(np.corrcoef(af, bf)[0, 1]) * math.sqrt(n)
    mi = _mutual_information(af, bf)
    rg = np.random.default_rng(0)
    null = [_mutual_information(af, rg.permutation(bf)) for _ in range(n_perm)]
    p_mi = (1 + sum(v >= mi for v in null)) / (n_perm + 1)
    one = np.uint32(1)
    agree = np.array([(((a ^ b) >> np.uint32(k)) & one).mean() for k in range(32)])
    bit_dev = int((np.abs(agree - 0.5) * 2 * math.sqrt(n) > Z_MAX).sum())
    bit_rot = 0
    for r in range(32):
        rb = b if r == 0 else ((b << np.uint32(r)) | (b >> np.uint32(32 - r)))
        x = a ^ rb
        frac = np.array([((x >> np.uint32(k)) & one).mean() for k in range(32)])
        bit_rot += int((np.abs(frac - 0.5) * 2 * math.sqrt(n) > Z_ROT).sum())
    return {"z": z, "p_mi": p_mi, "bit_dev": bit_dev, "bit_rot": bit_rot}


def _stream(salt: int, scheme: int) -> np.ndarray:
    i = mx.arange(N, dtype=mx.int32)
    with mx.stream(mx.cpu):
        # scheme 0: consecutive agent indices, one cell; scheme 1: index against step-like second argument
        h = ops.hash32(i, 7, salt) if scheme == 0 else ops.hash32(i % 97, i // 97, salt)
        mx.eval(h)
    return np.array(h)


def test_salt_table_is_frozen() -> None:
    """Changing a salt changes the trajectory of every run with that feature on: never edit this table."""
    assert dict(rng.HASH_SALTS) == FROZEN
    assert len(set(FROZEN.values())) == len(FROZEN) and all(0 <= v < 1 << 32 for v in FROZEN.values())


def test_salts_are_well_separated() -> None:
    for (na, a), (nb, b) in combinations(FROZEN.items(), 2):
        d = (a - b) & 0xFFFFFFFF
        assert bin(a ^ b).count("1") >= 8 and min(d, (1 << 32) - d) > 1 << 20, (na, nb)


def test_every_pair_of_salts_gives_independent_streams() -> None:
    for scheme in (0, 1):
        streams = {k: _stream(v, scheme) for k, v in rng.HASH_SALTS.items()}
        for (na, a), (nb, b) in combinations(streams.items(), 2):
            r = independence_report(a, b)
            assert abs(r["z"]) <= Z_MAX and r["p_mi"] > 0.005 and r["bit_dev"] == 0 and r["bit_rot"] == 0, (
                na,
                nb,
                scheme,
                r,
            )


def test_the_check_detects_dependent_streams() -> None:
    """Sanity of the helper: identical streams and a stream against its own complement-shift fail."""
    a = _stream(rng.HASH_SALTS["comm"], 0)
    r = independence_report(a, a)
    assert r["z"] > 100 and r["bit_dev"] > 0
    r2 = independence_report(a, (a << np.uint32(1)) | (a >> np.uint32(31)))
    assert r2["bit_rot"] > 0
