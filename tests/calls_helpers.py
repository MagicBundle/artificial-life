"""Configs and scenes of the event-triggered calls tests (DESIGN §47, step K1).

``WORLDS`` are the six worlds of the K1 matrix: the box in 2D and 3D and the tiny sphere, each without and
with the weather. ``calls_cfg`` builds a validated config of one of them with ``comm`` (width = the number of
events) and ``calls.trigger`` on; a sphere is never compiled in the test process (docs/SPHERE.md 10.1, F21
item 6): its compiled step is in ``test_calls_sphere_compiled``. ``patch_wind`` replaces the engine's wind
sampler by a fixed speed per row, so the storm event can be driven exactly (also on a sphere, whose weather
is slow).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import mlx.core as mx
import numpy as np
import pytest

from alife import weather
from alife.config import Config, load_config
from alife.ext.calls import event_names
from weather_helpers import storm2d_config, storm3d_config

F32 = "brain.weight_dtype=float32"
SIZES = (
    "population.n_max=64",
    "population.max_births_per_step=8",
    "population.n_init=8",
)
WORLDS = ("box2", "box2w", "box3", "box3w", "sph", "sphw")
WEATHER = {"box2w", "box3w", "sphw"}


def calls_cfg(
    world: str = "box2", *extra: str, events: int | None = None, cost: float = 32.0, micro: bool = False
) -> Config:
    """A validated config of ``world`` (one of ``WORLDS``) with comm and the calls trigger on. ``events``
    defaults to 5 (hurt and storm) with the weather and 3 (hurt and food) without; ``comm.width`` is the
    number of events. ``micro`` shrinks the population to ``SIZES`` (a hand-built scene)."""
    wx = world in WEATHER
    events = (5 if wx else 3) if events is None else events
    width = len(event_names(events))
    over = [
        F32,
        "comm.enabled=true",
        f"comm.width={width}",
        f"comm.cost={cost}",
        "calls.enabled=true",
        "calls.trigger=true",
        f"calls.events={events}",
        *extra,
    ]
    if micro:
        over = [*SIZES, *over]
    if world.startswith("box2"):
        cfg = (
            storm2d_config(*over)
            if wx
            else load_config(preset="tiny", overrides=[*(["world.size=48"] if micro else []), *over])
        )
    elif world.startswith("box3"):
        cfg = (
            storm3d_config(*over, n_init=8 if micro else 64)
            if wx
            else load_config(preset="tiny3d", overrides=[*(["world.size=48"] if micro else []), *over])
        )
    else:
        from sphere_weather_helpers import weather_cfg

        cfg = weather_cfg("tiny3s", *over) if wx else load_config(preset="tiny3s", overrides=over)
    if world.startswith("sph"):  # never compiled in the test process
        cfg = dataclasses.replace(cfg, run=dataclasses.replace(cfg.run, compile=False))
    return cfg


def patch_wind(
    monkeypatch: pytest.MonkeyPatch, speed: Callable[[int], Any], dims_nd: int | None = None
) -> None:
    """``weather.agent_wind`` returns zero force and the air speed ``speed(n)`` ((n,) floats, a numpy array or
    a Python list) for the ``n`` rows: the storm event and the hazard read exactly that speed. The step, the
    bench mirror and the metrics all call it through the module."""

    def fake(ws: Any, pos: Any, buoy_cap: Any, alive: Any, cfg: Config) -> weather.AgentForce:
        n = pos.shape[0]
        nd = pos.shape[1]
        sp = mx.array(np.asarray(speed(n), np.float32))
        brk = sp > cfg.weather.storm_break_u if cfg.weather.storm_break_u > 0 else None
        return weather.AgentForce(mx.zeros((n, nd), mx.float32), sp, brk, mx.array(False))

    monkeypatch.setattr(weather, "agent_wind", fake)


def storm_threshold(cfg: Config) -> np.float32:
    """The float32 the engine compares the air speed with: ``storm_frac * storm_u0`` rounded once."""
    return np.float32(cfg.calls.storm_frac * cfg.weather.storm_u0)
