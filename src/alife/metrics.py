"""Emergence metrics: bodies, snapshots, interval rows and carrying capacity (§19, §10.10). Owned by WP5."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import mlx.core as mx
import numpy as np

from alife.config import Config
from alife.state import State


def components(partner: mx.array, alive: mx.array, iters: int) -> tuple[mx.array, mx.array]:
    """Bond-graph connected components by min-label propagation: ``(labels, converged)`` (§19.2)."""
    raise NotImplementedError


def snapshot_metrics(s: State, cfg: Config, *, heavy: bool) -> dict[str, Any]:
    """Host floats/ints/lists describing the current state, with one host transfer (§19.2)."""
    raise NotImplementedError


def interval_row(
    step: int,
    counters: np.ndarray,
    ledgers: np.ndarray,
    snap: Mapping[str, Any],
    perf: Mapping[str, float],
) -> dict[str, Any]:
    """One ``metrics.jsonl`` row from stacked counters (T,26), ledgers (T,16) and a snapshot (§19.2)."""
    raise NotImplementedError


def carrying_capacity(s: State, cfg: Config) -> float:
    """``K_est = Σ_c regen_rate_c / c_ref`` (§10.10)."""
    raise NotImplementedError
