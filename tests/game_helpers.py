"""Synthetic observer inputs for the game tests (§28): hand-built ``ObsIn`` and ``Acts`` over a config's
sizes, and a driver that runs the observer's ``update`` and ``evaluate`` on the step clock without a
simulation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import mlx.core as mx
import numpy as np

from alife import layout
from alife.config import Config, dims
from alife.game.observe import ObsIn, ObsState, evaluate, init_obs, update
from alife.game.rules import RULES_V1, RuleSet
from alife.state import to_numpy
from alife.step import Acts


def blank_oi(cfg: Config) -> dict[str, np.ndarray]:
    """Numpy arrays of an all-dead ``ObsIn`` (``make_oi`` turns them into one)."""
    d = dims(cfg)
    return {
        "alive": np.zeros(d.n, bool),
        "uid": np.full(d.n, -1, np.int64),
        "parent_uid": np.full(d.n, -1, np.int64),
        "partner": np.full((d.n, layout.B), -1, np.int32),
        "vel": np.zeros((d.n, d.nd), np.float32),
        "last_harvest": np.zeros(d.n, np.int32),
        "last_inflow": np.zeros(d.n, np.int32),
        "last_loss": np.zeros(d.n, np.int32),
    }


def make_oi(arrays: Mapping[str, Any]) -> ObsIn:
    return ObsIn(**{f: mx.array(np.asarray(arrays[f])) for f in ObsIn._fields})


def blank_social_acts(n: int, k: int, m: int) -> dict[str, np.ndarray]:
    """The social-tracking additions of ``Acts`` (§29.2) for ``n`` rows, ``k`` neighbours and ``m`` lanes: no
    target, no neighbour, no kin query, no birth lane."""
    return {
        "attack_target": np.full(n, -1, np.int32),
        "bond_target": np.full(n, -1, np.int32),
        "q_kin": np.zeros((n, 2), np.int16),
        "nbr": np.full((n, k), -1, np.int32),
        "born_row": np.full(m, -1, np.int32),
        "born_parent": np.full(m, -1, np.int32),
        "born_uid": np.full(m, -1, np.int64),
    }


def blank_acts(cfg: Config) -> dict[str, np.ndarray]:
    d = dims(cfg)
    return {
        "attack_gain": np.zeros(d.n, np.int32),
        "share_out": np.zeros((d.n, layout.B), np.int32),
        "share_got": np.zeros((d.n, layout.B), np.int32),
        "flags": np.zeros(d.n, np.uint8),
        "thrust": np.zeros(d.n, np.int16),
        "defend": np.zeros(d.n, np.int16),
        "emit": np.zeros(d.n, np.int16),
        **blank_social_acts(d.n, d.k, d.m),
    }


def make_acts(arrays: Mapping[str, Any]) -> Acts:
    return Acts(**{f: mx.array(np.asarray(arrays[f])) for f in Acts._fields})


def living(cfg: Config, rows: int, **fields: Any) -> dict[str, np.ndarray]:
    """``blank_oi`` with rows ``0..rows-1`` alive (uid = row) and ``fields`` (name → array or scalar,
    broadcast over the alive rows) set."""
    oi = blank_oi(cfg)
    oi["alive"][:rows] = True
    oi["uid"][:rows] = np.arange(rows)
    for name, value in fields.items():
        oi[name][:rows] = value
    return oi


def np_state(o: ObsState) -> dict[str, np.ndarray]:
    return {f: to_numpy(getattr(o, f)) for f in ObsState._fields}


class Driver:
    """One observer on a synthetic constant (or per-step) input, evaluated on the global step clock."""

    def __init__(self, cfg: Config, oi: ObsIn, *, start: int = 0, rs: RuleSet = RULES_V1) -> None:
        self.cfg, self.rs, self.t = cfg, rs, start
        self.o = init_obs(oi, cfg)
        mx.eval(*self.o)

    def step(self, oi: ObsIn, acts: Acts) -> ObsState:
        """Update with one step's inputs; evaluate if the new step count is a multiple of ``rs.every``."""
        self.t += 1
        self.o = update(self.o, oi, acts, self.cfg, self.rs)
        if self.t % self.rs.every == 0:
            self.o = evaluate(self.o, self.cfg, self.rs)
        mx.eval(*self.o)
        return self.o
