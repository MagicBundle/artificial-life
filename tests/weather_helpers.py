"""Helpers for the W1 engine tests (DESIGN §30): the two weather-on configs of the spec's §9, a prescribed
uniform wind, and hand-built physics inputs."""

from __future__ import annotations

import functools

import mlx.core as mx
import numpy as np

from alife import weather as W
from alife.actions import Actions, Motion3
from alife.config import Config, dims, load_config
from alife.layout import MSG, NCH, B
from alife.spatial import Neighbors

SMALL3D = (
    "population.n_max=256",
    "population.n_init=64",
    "population.max_births_per_step=32",
    "brain.weight_dtype=float32",
)


def storm3d_config(*overrides: str, n_init: int = 64) -> Config:
    """``64k3d`` + the ``storm`` overlay with 256 rows (L = 320, so the confirmed 160-cell weather is valid),
    float32 weights (the whole-step default, F19)."""
    ov = [o for o in SMALL3D if not o.startswith("population.n_init")]
    return load_config(
        preset="64k3d", overlays=["storm"], overrides=[*ov, f"population.n_init={n_init}", *overrides]
    )


def storm2d_config(*overrides: str) -> Config:
    """``tiny`` on a 320-wide world with the weather on and ``u_max = 0.2`` (rule R4), float32 weights."""
    return load_config(
        preset="tiny",
        overrides=[
            "world.size=320",
            "weather.enabled=true",
            "weather.u_max=0.2",
            "brain.weight_dtype=float32",
            *overrides,
        ],
    )


def fixed_wx(cfg: Config, u1: float = 0.0, v1: float = 0.0, u2: float = 0.0, v2: float = 0.0) -> mx.array:
    """The rest state of ``cfg``'s weather with uniform winds ``(u1, v1)`` in layer 1 and ``(u2, v2)`` in
    layer 2 (uniform winds and thicknesses have no divergence and no interface slope, so ``w = 0``)."""
    ws = W.unpack(W.rest_array(cfg))
    full = mx.full(ws.h1.shape, 1.0, dtype=mx.float32)
    return W.pack(ws._replace(u1=u1 * full, v1=v1 * full, u2=u2 * full, v2=v2 * full))


def zero_actions(n: int) -> Actions:
    """All-zero (inactive) actions."""
    z = np.zeros(n, np.float32)
    base = {
        "thrust": z, "turn": z, "msg": np.zeros((n, MSG), np.float32), "emit": np.zeros((n, NCH), np.float32),
        "harvest": z, "grip": z, "contract": z, "bond_want": np.zeros(n, bool),
        "release": np.zeros((n, B), bool), "share": np.zeros((n, B), np.float32), "attack": np.zeros(n, bool),
        "defend": z, "reproduce": np.zeros(n, bool), "child_bond": np.zeros(n, bool),
        "bond_k": np.full(n, -1, np.int32), "bond_target": np.full(n, -1, np.int32),
        "attack_k": np.full(n, -1, np.int32), "attack_target": np.full(n, -1, np.int32),
    }  # fmt: skip
    return Actions(**{k: mx.array(v) for k, v in base.items()})


def zero_motion(n: int) -> Motion3:
    z = mx.zeros((n,), dtype=mx.float32)
    return Motion3(z, z)


def no_neighbors(n: int, k: int, radius: float) -> Neighbors:
    """Nobody has a neighbour: every slot invalid, pointing at the row itself."""
    zero = mx.array(0, mx.int32)
    idx = mx.broadcast_to(mx.arange(n, dtype=mx.int32)[:, None], (n, k))
    return Neighbors(
        idx,
        mx.zeros((n, k), dtype=mx.bool_),
        mx.zeros((n, k, 3), dtype=mx.float32),
        mx.full((n, k), radius, dtype=mx.float32),
        zero,
        zero,
    )


def floor_scene(cfg: Config, pts: list[tuple[float, ...]]) -> tuple[mx.array, mx.array]:
    """``(pos, alive)`` with ``pts`` in the first rows and dead rows resting at ``(0, 0, z_lo)``."""
    d = dims(cfg)
    pos = np.zeros((d.n, d.nd), np.float32)
    if d.d3:
        pos[:, 2] = d.z_lo
    pos[: len(pts)] = pts
    return mx.array(pos), mx.array(np.arange(d.n) < len(pts))


@functools.cache
def spun_ws(cfg: Config, steps: int = 500) -> W.WeatherState:
    """The weather of ``cfg`` after ``steps`` standalone weather steps from the rest state (one key chain,
    compiled): about 4% of the cells rain and the winds are non-trivial after 500 steps (CPU: 2.6 s)."""
    fn = mx.compile(lambda ws, t, k: W.advance(ws, t, k, cfg))
    ws, key = W.unpack(W.rest_array(cfg)), mx.random.key(1)
    for t in range(1, steps + 1):
        key, k = mx.random.split(key)
        ws, _ = fn(ws, mx.array(t, dtype=mx.int32), k)
    mx.eval(ws)
    return ws
