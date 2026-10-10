"""Per-agent Elman RNN controllers: init, forward, mutation and weight cost (§6).

``init_weights``, ``l1_cost`` and ``mutate_rows`` are WP0 hand-over functions; WP2 owns this module.
"""

from __future__ import annotations

import math

import mlx.core as mx

from alife import layout
from alife.config import Config, dims


def init_weights(key_w1: mx.array, key_w2: mx.array, n_rows: int, cfg: Config) -> tuple[mx.array, mx.array]:
    """Random founder/immigrant weights ``(W1 (n_rows, W1_ROWS, H), W2 (n_rows, W2_ROWS, D_OUT))`` in wdtype.

    ``W1 ~ N(0, init_scale²/W1_ROWS)`` with the recurrent rows scaled by ``recurrent_scale``,
    ``W2 ~ N(0, init_scale²/W2_ROWS)``; both clipped to ``±weight_clip``. No output bias is seeded.
    """
    d, bc = dims(cfg), cfg.brain
    w1 = mx.random.normal((n_rows, d.w1_rows, d.h), key=key_w1) * (bc.init_scale / math.sqrt(d.w1_rows))
    row_scale = mx.where(mx.arange(d.w1_rows) >= layout.D_IN, bc.recurrent_scale, 1.0).astype(mx.float32)
    w1 = w1 * row_scale[None, :, None]
    w2 = mx.random.normal((n_rows, d.w2_rows, layout.D_OUT), key=key_w2) * (bc.init_scale / math.sqrt(d.w2_rows))
    clip = bc.weight_clip
    return mx.clip(w1, -clip, clip).astype(d.wdtype), mx.clip(w2, -clip, clip).astype(d.wdtype)


def forward(W1: mx.array, W2: mx.array, x: mx.array, h: mx.array, cfg: Config) -> tuple[mx.array, mx.array]:
    """Batched RNN step: ``(h_new (N,H) f32, y (N,D_OUT) f32)`` (§6)."""
    raise NotImplementedError


def _mutate(w: mx.array, sigma: mx.array, rate: mx.array, key: mx.array, clip: float) -> mx.array:
    k_mask, k_eps = mx.random.split(key, 2)
    u = mx.random.uniform(shape=w.shape, key=k_mask)
    eps = mx.random.normal(w.shape, key=k_eps)
    delta = mx.where(u < rate[:, None, None], sigma[:, None, None] * eps, 0.0)
    return mx.clip(w.astype(mx.float32) + delta, -clip, clip)


def mutate_rows(
    W1p: mx.array,
    W2p: mx.array,
    sigma: mx.array,
    rate: mx.array,
    key_w1: mx.array,
    key_w2: mx.array,
    cfg: Config,
) -> tuple[mx.array, mx.array]:
    """Child weights ``clip(W_p + 1[u < p_c]·σ_c·ε, ±weight_clip)``, computed in f32 and cast to wdtype.

    ``W1p``/``W2p`` are the (M, rows, cols) parent rows, ``sigma``/``rate`` the child's mutated genes (M,).
    Each key is split into ``(k_mask, k_eps)`` (§6, §14.1).
    """
    d, clip = dims(cfg), cfg.brain.weight_clip
    w1 = _mutate(W1p, sigma, rate, key_w1, clip)
    w2 = _mutate(W2p, sigma, rate, key_w2, clip)
    return w1.astype(d.wdtype), w2.astype(d.wdtype)


def l1_cost(W1r: mx.array, W2r: mx.array, cfg: Config) -> mx.array:
    """Per-row brain upkeep ``floor(brain_cost_per_l1·(Σ|W1| + Σ|W2|))`` from the stored weights, (rows,) int32."""
    l1 = mx.abs(W1r.astype(mx.float32)).sum(axis=(1, 2)) + mx.abs(W2r.astype(mx.float32)).sum(axis=(1, 2))
    return mx.floor(cfg.energy.brain_cost_per_l1 * l1).astype(mx.int32)
