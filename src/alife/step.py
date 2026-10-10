"""The pure, compiled simulation step (§3, §16.3). Owned by WP4."""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import mlx.core as mx

from alife.config import Config
from alife.lifecycle import BirthRecords, DeathRecords
from alife.state import State

__all__ = ["BirthRecords", "DeathRecords", "StepOut", "make_step", "step"]


class StepOut(NamedTuple):
    counters: mx.array  # (N_COUNTERS,) int32, CounterIdx (§19.1)
    ledger: mx.array  # (16,) int64, this step's ledger (LedgerIdx, §10.8)
    births: BirthRecords
    deaths: DeathRecords


def step(s: State, cfg: Config) -> tuple[State, StepOut]:
    """Advance the world by one step (S0–S18). Pure; no host syncs."""
    raise NotImplementedError


def make_step(cfg: Config, *, compile: bool | None = None) -> Callable[[State], tuple[State, StepOut]]:
    """The step closed over ``cfg``, compiled once when ``compile`` (default ``cfg.run.compile``)."""
    raise NotImplementedError
