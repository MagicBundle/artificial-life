"""Configs of the sphere weather tests (S4b): the three preset geometries with the weather on.

The pools of the small planets are narrower than the box's 32 x 16 (S4a: ``tiny3s`` needs 2 to 3 times
narrower pools, ``mini3s`` narrower ones): the numbers here are for the tests, the study of the storms is S4c.
"""

from __future__ import annotations

import math

from alife.config import Config, load_config
from sphere_helpers import BASE, SPHERE_SETS

NW = {"tiny3s": 16, "mini3s": 22, "64k3s": 42}
RADIUS = {"tiny3s": 36.0, "mini3s": 48.0, "64k3s": 90.0}
POOL_SIZE = {"tiny3s": (10.0, 6.0), "mini3s": (16.0, 10.0), "64k3s": (32.0, 16.0)}


def pools(which: str, drift: float = -0.009) -> list[float]:
    """Six pools at latitude +-40 degrees, three per hemisphere (120 degrees apart, the southern offset by
    60), as the flat 7-tuples of ``weather.pools`` (arc lengths east and north of the origin)."""
    r = RADIUS[which]
    sx, sy = POOL_SIZE[which]
    out: list[float] = []
    for lon, lat in ((60, 40), (180, 40), (300, 40), (0, -40), (120, -40), (240, -40)):
        out += [r * math.radians(lon), r * math.radians(lat), sx, sy, 1.0, drift, 0.0]
    return out


def weather_cfg(which: str = "tiny3s", *overrides: str) -> Config:
    """A validated sphere config with the weather on (rain food on, hazard on, the pools of ``pools``)."""
    pl = ",".join(f"{v:.6f}" for v in pools(which))
    return load_config(
        preset="tiny3d",
        overrides=[
            *BASE,
            *SPHERE_SETS[which],
            "weather.enabled=true",
            f"weather.cells={NW[which]}",
            f"weather.pools={pl}",
            "weather.rain_gain=50000",
            "weather.storm_p=0.002",
            *overrides,
        ],
    )
