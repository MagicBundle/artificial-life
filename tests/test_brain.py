"""§6 brain: init distribution, forward vs a numpy reference, recurrence, mutation statistics, l1 cost."""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from alife import brain, layout
from alife.config import Config, apply_overrides, dims


def _np(a: mx.array) -> np.ndarray:
    return np.array(a.astype(mx.float32)) if a.dtype in (mx.float16, mx.bfloat16) else np.array(a)


def _keys(seed: int) -> tuple[mx.array, mx.array]:
    k = mx.random.split(mx.random.key(seed), 2)
    return k[0], k[1]


@pytest.mark.parametrize("fixture", ["tiny_cfg", "tiny_cfg_f16"])
def test_init_weights_statistics(fixture: str, request: pytest.FixtureRequest) -> None:
    cfg: Config = request.getfixturevalue(fixture)
    d = dims(cfg)
    w1, w2 = brain.init_weights(*_keys(0), 400, cfg)
    assert w1.shape == (400, d.w1_rows, d.h) and w2.shape == (400, d.w2_rows, layout.D_OUT)
    assert w1.dtype == d.wdtype and w2.dtype == d.wdtype
    a1, a2 = _np(w1), _np(w2)
    s1 = cfg.brain.init_scale / math.sqrt(d.w1_rows)
    s2 = cfg.brain.init_scale / math.sqrt(d.w2_rows)
    assert abs(a1[:, : layout.D_IN].std() / s1 - 1) < 0.05
    assert abs(a1[:, layout.D_IN :].std() / (s1 * cfg.brain.recurrent_scale) - 1) < 0.05
    assert abs(a2.std() / s2 - 1) < 0.05
    assert abs(a1.mean()) < 0.01 and abs(a2.mean()) < 0.01
    # Deterministic per key, different across keys.
    w1b, _ = brain.init_weights(*_keys(0), 400, cfg)
    assert np.array_equal(_np(w1b), a1)
    assert not np.array_equal(_np(brain.init_weights(*_keys(1), 400, cfg)[0]), a1)


def test_init_weights_clip(tiny_cfg: Config) -> None:
    cfg = apply_overrides(tiny_cfg, ["brain.init_scale=200"])
    w1, w2 = brain.init_weights(*_keys(2), 16, cfg)
    for w in (_np(w1), _np(w2)):
        assert np.abs(w).max() == pytest.approx(cfg.brain.weight_clip)
        assert np.all(np.abs(w) <= cfg.brain.weight_clip)


def test_l1_cost_matches_numpy(tiny_cfg: Config, tiny_cfg_f16: Config) -> None:
    for cfg in (tiny_cfg, tiny_cfg_f16):
        w1, w2 = brain.init_weights(*_keys(3), 64, cfg)
        cost = brain.l1_cost(w1, w2, cfg)
        assert cost.dtype == mx.int32 and cost.shape == (64,)
        stored1, stored2 = _np(w1).astype(np.float64), _np(w2).astype(np.float64)
        ref = np.floor(
            cfg.energy.brain_cost_per_l1 * (np.abs(stored1).sum((1, 2)) + np.abs(stored2).sum((1, 2)))
        )
        assert np.max(np.abs(np.array(cost) - ref)) <= 1  # f32 summation order may flip a floor
        assert np.mean(np.array(cost) == ref) > 0.9
        # A founder pays ≈ 37 q/step at the defaults (§6).
        assert 30 <= np.array(cost).mean() <= 45
    z = mx.zeros((3, dims(tiny_cfg).w1_rows, dims(tiny_cfg).h))
    z2 = mx.zeros((3, dims(tiny_cfg).w2_rows, layout.D_OUT))
    assert np.array(brain.l1_cost(z, z2, tiny_cfg)).tolist() == [0, 0, 0]
    ones1, ones2 = mx.ones(z.shape), mx.ones(z2.shape)
    want = math.floor(0.1 * (z.size // 3 + z2.size // 3))
    assert np.array(brain.l1_cost(ones1, ones2, tiny_cfg)).tolist() == [want] * 3


def _parents(cfg: Config, m: int, seed: int = 4) -> tuple[mx.array, mx.array]:
    return brain.init_weights(*_keys(seed), m, cfg)


def test_mutate_sigma_zero_is_identity(tiny_cfg: Config, tiny_cfg_f16: Config) -> None:
    for cfg in (tiny_cfg, tiny_cfg_f16):
        w1, w2 = _parents(cfg, 8)
        sigma, rate = mx.zeros((8,)), mx.full((8,), 0.5)
        c1, c2 = brain.mutate_rows(w1, w2, sigma, rate, *_keys(5), cfg)
        assert c1.dtype == w1.dtype and c2.dtype == w2.dtype
        assert np.array_equal(_np(c1), _np(w1)) and np.array_equal(_np(c2), _np(w2))
        c1, c2 = brain.mutate_rows(w1, w2, mx.full((8,), 0.3), mx.zeros((8,)), *_keys(5), cfg)
        assert np.array_equal(_np(c1), _np(w1)), "rate 0 changes nothing"


def test_mutate_changed_fraction(tiny_cfg: Config) -> None:
    m = 32
    w1, w2 = _parents(tiny_cfg, m)
    rate = mx.array(np.linspace(0.01, 0.5, m, dtype=np.float32))
    c1, c2 = brain.mutate_rows(w1, w2, mx.full((m,), 0.1), rate, *_keys(6), tiny_cfg)
    for c, w in ((c1, w1), (c2, w2)):
        changed = (_np(c) != _np(w)).reshape(m, -1)
        k = changed.shape[1]
        frac = changed.mean(1)
        p = np.array(rate)
        bound = 5 * np.sqrt(p * (1 - p) / k) + 1e-3  # 5-sigma binomial bound per row
        assert np.all(np.abs(frac - p) <= bound)


def test_mutate_clipped_and_scaled(tiny_cfg: Config) -> None:
    m = 16
    d = dims(tiny_cfg)
    w1 = mx.full((m, d.w1_rows, d.h), 3.9)
    w2 = mx.full((m, d.w2_rows, layout.D_OUT), -3.9)
    c1, c2 = brain.mutate_rows(w1, w2, mx.full((m,), 0.3), mx.full((m,), 0.5), *_keys(7), tiny_cfg)
    a1, a2 = _np(c1), _np(c2)
    assert np.all(np.abs(a1) <= 4.0) and np.all(np.abs(a2) <= 4.0)
    assert np.any(a1 == 4.0) and np.any(a2 == -4.0)
    delta = (a1 - 3.9)[a1 < 3.9]  # the unclipped half-normal side keeps the σ scale: E|δ| = σ·√(2/π)
    assert abs(-delta.mean() / (0.3 * math.sqrt(2 / math.pi)) - 1) < 0.05


def test_mutate_matches_reference_and_casts_last(tiny_cfg_f16: Config) -> None:
    """Child = cast(clip(f32(W_p) + 1[u < p]·σ·ε)) with u, ε from split(key, 2): the cast happens last."""
    cfg = tiny_cfg_f16
    m = 6
    w1, w2 = _parents(cfg, m, seed=8)
    sigma = mx.array(np.linspace(0.002, 0.3, m, dtype=np.float32))
    rate = mx.array(np.linspace(0.05, 0.5, m, dtype=np.float32))
    k1, k2 = _keys(9)
    c1, c2 = brain.mutate_rows(w1, w2, sigma, rate, k1, k2, cfg)
    for c, w, k in ((c1, w1, k1), (c2, w2, k2)):
        km, ke = mx.random.split(k, 2)
        u = np.array(mx.random.uniform(shape=w.shape, key=km))
        eps = np.array(mx.random.normal(w.shape, key=ke))
        wp = _np(w).astype(np.float32)
        s, p = np.array(sigma)[:, None, None], np.array(rate)[:, None, None]
        ref = (
            np.clip(wp + np.where(u < p, s * eps, np.float32(0)), -4, 4).astype(np.float32).astype(np.float16)
        )
        assert c.dtype == mx.float16
        got = np.array(c.astype(mx.float32)).astype(np.float16)
        assert np.mean(got == ref) > 0.999  # identical up to rare f32 rounding-order ties
        assert np.max(np.abs(got.astype(np.float32) - ref.astype(np.float32))) <= 4e-3


# --- forward (WP2) ----------------------------------------------------------------------------------------


def _ref_forward(
    w1: np.ndarray, w2: np.ndarray, x: np.ndarray, h: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """float64 reference of §6: ``h' = tanh([x, h] @ W1)``, ``y = [h', 1] @ W2``."""
    z = np.concatenate([x, h], 1).astype(np.float64)
    hn = np.tanh(np.einsum("nr,nrh->nh", z, w1.astype(np.float64)))
    hz = np.concatenate([hn, np.ones((hn.shape[0], 1))], 1)
    return hn, np.einsum("nr,nro->no", hz, w2.astype(np.float64))


def _inputs(cfg: Config, n: int, seed: int) -> tuple[mx.array, mx.array]:
    rs = np.random.default_rng(seed)
    x = rs.uniform(-1, 1, (n, layout.D_IN)).astype(np.float32)
    x[:, layout.IN_BIAS] = 1.0
    h = rs.uniform(-1, 1, (n, dims(cfg).h)).astype(np.float32)
    return mx.array(x), mx.array(h)


@pytest.mark.parametrize(
    ("fixture", "tol"), [("tiny_cfg", 1e-5), ("tiny_cfg_f16", 2e-2), ("tiny_cfg_bf16", 1e-1)]
)
@pytest.mark.parametrize("impl", ["matmul", "einsum"])
def test_forward_matches_reference(
    fixture: str, tol: float, impl: str, request: pytest.FixtureRequest
) -> None:
    cfg = apply_overrides(request.getfixturevalue(fixture), [f"brain.impl={impl}"])
    d, n = dims(cfg), 128
    w1, w2 = brain.init_weights(*_keys(10), n, cfg)
    x, h = _inputs(cfg, n, 10)
    h_new, y = brain.forward(w1, w2, x, h, cfg)
    assert h_new.shape == (n, d.h) and y.shape == (n, layout.D_OUT)
    assert h_new.dtype == mx.float32 and y.dtype == mx.float32
    rh, ry = _ref_forward(_np(w1), _np(w2), np.array(x), np.array(h))
    assert np.max(np.abs(np.array(h_new) - rh)) < tol
    assert np.max(np.abs(np.array(y) - ry)) < tol * max(1.0, float(np.abs(ry).max()))
    assert np.all(np.abs(np.array(h_new)) <= 1.0)


@pytest.mark.parametrize("fixture", ["tiny_cfg", "tiny_cfg_f16"])
def test_forward_impls_agree(fixture: str, request: pytest.FixtureRequest) -> None:
    cfg: Config = request.getfixturevalue(fixture)
    w1, w2 = brain.init_weights(*_keys(11), 64, cfg)
    x, h = _inputs(cfg, 64, 11)
    a = brain.forward(w1, w2, x, h, apply_overrides(cfg, ["brain.impl=matmul"]))
    b = brain.forward(w1, w2, x, h, apply_overrides(cfg, ["brain.impl=einsum"]))
    tol = 1e-5 if cfg.brain.weight_dtype == "float32" else 1e-2
    for u, v in zip(a, b, strict=True):
        assert np.max(np.abs(np.array(u) - np.array(v))) < tol


def test_forward_bias_paths_and_recurrence(tiny_cfg: Config) -> None:
    """``x[0] = 1`` carries the hidden bias (W1 row 0), the constant hidden unit the output bias (W2 row H),
    and rows ``D_IN..`` of W1 read the previous hidden state."""
    cfg, d, n = tiny_cfg, dims(tiny_cfg), 4
    rs = np.random.default_rng(12)
    w1 = np.zeros((n, d.w1_rows, d.h), np.float32)
    w2 = np.zeros((n, d.w2_rows, layout.D_OUT), np.float32)
    w1[:, layout.IN_BIAS] = rs.uniform(-1, 1, (n, d.h))
    w2[:, d.h] = rs.uniform(-1, 1, (n, layout.D_OUT))
    x = np.zeros((n, layout.D_IN), np.float32)
    x[:, layout.IN_BIAS] = 1.0
    h0 = np.zeros((n, d.h), np.float32)
    hn, y = brain.forward(mx.array(w1), mx.array(w2), mx.array(x), mx.array(h0), cfg)
    assert np.allclose(np.array(hn), np.tanh(w1[:, layout.IN_BIAS]), atol=1e-6)
    assert np.allclose(np.array(y), w2[:, d.h], atol=1e-6)  # W2 rows < H see h' but are zero here
    # Pure recurrence: identity recurrent block, no bias → h' = tanh(h); rolled over 3 steps.
    w1 = np.zeros_like(w1)
    w1[:, layout.D_IN :] = np.eye(d.h, dtype=np.float32)
    x[:] = 0.0
    h = rs.uniform(-1, 1, (n, d.h)).astype(np.float32)
    want = h.astype(np.float64)
    hm = mx.array(h)
    for _ in range(3):
        hm, _ = brain.forward(mx.array(w1), mx.array(w2), mx.array(x), hm, cfg)
        want = np.tanh(want)
    assert np.allclose(np.array(hm), want, atol=1e-6)
    # Rows are independent: each row uses only its own weights.
    w1r = rs.normal(0, 0.3, (n, d.w1_rows, d.h)).astype(np.float32)
    full, _ = brain.forward(mx.array(w1r), mx.array(w2), mx.array(x), mx.array(h), cfg)
    one, _ = brain.forward(mx.array(w1r[2:3]), mx.array(w2[2:3]), mx.array(x[2:3]), mx.array(h[2:3]), cfg)
    assert np.allclose(np.array(full)[2], np.array(one)[0], atol=1e-6)


def test_forward_compiles(tiny_cfg_f16: Config) -> None:
    cfg = tiny_cfg_f16
    w1, w2 = brain.init_weights(*_keys(13), 32, cfg)
    x, h = _inputs(cfg, 32, 13)
    eager = brain.forward(w1, w2, x, h, cfg)
    comp = mx.compile(lambda a, b, c, e: brain.forward(a, b, c, e, cfg))(w1, w2, x, h)
    for u, v in zip(eager, comp, strict=True):
        assert u.dtype == v.dtype == mx.float32
        assert np.allclose(np.array(u), np.array(v), atol=1e-3)


def test_mutate_perturbation_scale(tiny_cfg: Config) -> None:
    """Changed weights move by N(0, σ_c²) per row (row-specific σ), before clipping."""
    m = 8
    d = dims(tiny_cfg)
    w1 = mx.zeros((m, d.w1_rows, d.h))
    w2 = mx.zeros((m, d.w2_rows, layout.D_OUT))
    sigma = np.array([0.002, 0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3], np.float32)
    c1, _ = brain.mutate_rows(w1, w2, mx.array(sigma), mx.full((m,), 0.5), *_keys(14), tiny_cfg)
    a = _np(c1).reshape(m, -1)
    for r in range(m):
        moved = a[r][a[r] != 0]
        assert abs(moved.std() / sigma[r] - 1) < 0.1
        assert abs(moved.mean()) < 0.1 * sigma[r]
