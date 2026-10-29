"""§14.1 RNG key layout."""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from alife import rng


def _np(a: mx.array) -> np.ndarray:
    return np.array(a)


def test_step_keys_deterministic_and_distinct() -> None:
    key = rng.init_keys(3)[rng.InitStream.STATE_KEY]
    k1, ks1 = rng.step_keys(key)
    k2, ks2 = rng.step_keys(key)
    assert k1.shape == (2,) and k1.dtype == mx.uint32
    assert ks1.shape == (rng.NUM_STREAMS, 2) and ks1.dtype == mx.uint32
    assert np.array_equal(_np(k1), _np(k2)) and np.array_equal(_np(ks1), _np(ks2))
    rows = {tuple(r) for r in _np(ks1).tolist()} | {tuple(_np(k1).tolist()), tuple(_np(key).tolist())}
    assert len(rows) == rng.NUM_STREAMS + 2, "streams, next key and input key are pairwise distinct"
    k3, _ = rng.step_keys(k1)
    assert not np.array_equal(_np(k3), _np(k1))


def test_step_keys_matches_documented_split() -> None:
    key = mx.random.key(11)
    k = mx.random.split(key, 2)
    key_next, ks = rng.step_keys(key)
    assert np.array_equal(_np(key_next), _np(k[0]))
    assert np.array_equal(_np(ks), _np(mx.random.split(k[1], 16)))


def test_init_keys() -> None:
    a, b = rng.init_keys(0), rng.init_keys(1)
    assert a.shape == (16, 2) and a.dtype == mx.uint32
    assert not np.array_equal(_np(a), _np(b))
    assert np.array_equal(_np(a), _np(rng.init_keys(0)))
    assert np.array_equal(_np(a), _np(mx.random.split(mx.random.key(0), 16)))
    assert len({tuple(r) for r in _np(a).tolist()}) == 16


def test_observer_key() -> None:
    k = rng.observer_key(5, 100)
    assert np.array_equal(_np(k), _np(mx.random.key((5 * 1_000_003 + 100) % 2**63)))
    assert not np.array_equal(_np(k), _np(rng.observer_key(5, 101)))
    rng.observer_key(2**62, 2**40)  # large inputs stay in range


def test_layout_frozen() -> None:
    """Changing the stream layout must come with an RNG_LAYOUT_VERSION bump (update this snapshot then)."""
    assert rng.RNG_LAYOUT_VERSION == 3
    assert {1, 2, 3, 4} == set(rng.RNG_COMPATIBLE_VERSIONS)  # 4: arms.enabled (DESIGN §42.4)
    assert rng.NUM_STREAMS == 16
    assert {s.name: s.value for s in rng.Stream} == {
        "NOISE": 0,
        "BOND_PRIO": 1,
        "BIRTH_PRIO": 2,
        "PLACEMENT": 3,
        "MUT_W1": 4,
        "MUT_W2": 5,
        "MUT_GENES": 6,
        "IMMIGRANT": 7,
        "CATASTROPHE": 8,
        "WEATHER": 9,
        "ARMS": 10,
    }
    assert {s.name: s.value for s in rng.InitStream} == {
        "POS": 0,
        "HEADING": 1,
        "W1": 2,
        "W2": 3,
        "TAG": 4,
        "PATCH_POS": 5,
        "PATCH_VEL": 6,
        "STATE_KEY": 7,
        "ARMS": 8,
    }
