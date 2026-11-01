"""Where things are, for the host side of the game layer, in either geometry (DESIGN §46.7).

The social records, the events, the alerts and the reports state positions. In a box world a position is
``(x, y)`` on the torus, ``z`` the height; on a sphere it is the planet-frame Cartesian point ``(x, y, z)``
(the stream's convention, the same as ``FRMA``), and text states it as latitude, longitude and height
(``geom.place_text``). This module is the one place the game layer reads that difference: a centroid of
members (a circular mean on the torus, the mean direction at the mean radius on a sphere), a height, a
distance between places, and the sentence form. A consumer calls these with the config and never
indexes a position itself.

Host side, numpy, float64. Pure functions: no state, no I/O.
"""

from __future__ import annotations

import numpy as np

from alife import geom
from alife.config import Config
from alife.tracking import circular_mean


def is_sphere(cfg: Config) -> bool:
    return cfg.world.geometry == "sphere"


def centroid(pos: np.ndarray, cfg: Config) -> tuple[float, ...]:
    """The centre of the rows ``pos`` (n, 2 or 3), n >= 1: ``(x, y)`` as the circular mean on the torus of a
    box world, or ``(x, y, z)`` on a sphere: the mean direction of the members at their mean radius. Floats,
    unrounded."""
    if is_sphere(cfg):
        p = np.asarray(pos, dtype=np.float64)
        direction, _ = geom.mean_direction(p)
        return tuple(float(v) for v in direction * np.linalg.norm(p, axis=1).mean())
    size = cfg.world.size
    return (circular_mean(pos[:, 0], size), circular_mean(pos[:, 1], size))


def mean_height(pos: np.ndarray, cfg: Config) -> float:
    """The mean height of the rows ``pos`` (n, 3): above the floor in a box, above the planet on a sphere."""
    return float(np.asarray(geom.geometry(cfg).height(np.asarray(pos, dtype=np.float64))).mean())


def heights(pos: np.ndarray, cfg: Config) -> np.ndarray:
    """The height of every row of ``pos`` (n, 3) as float64."""
    return np.asarray(geom.geometry(cfg).height(np.asarray(pos, dtype=np.float64)), dtype=np.float64)


def distance(a: np.ndarray, b: np.ndarray, cfg: Config) -> np.ndarray:
    """The planar distance between places ``a`` and ``b`` (..., 2 or 3): the torus minimum image of the
    floor-plane offset in a box, the great-circle arc on a sphere (at the mean radius)."""
    g = geom.geometry(cfg)
    if is_sphere(cfg):
        return np.asarray(g.arc(np.asarray(a, np.float64), np.asarray(b, np.float64)))
    d = np.asarray(g.displacement(np.asarray(a, np.float64)[..., :2], np.asarray(b, np.float64)[..., :2]))
    return np.linalg.norm(d, axis=-1)


def text(place: tuple[float, ...] | list[float] | np.ndarray, cfg: Config) -> str:
    """A place as words: ``"x 12.0 y 34.0"`` style is the box's business; on a sphere ``"35.3N 45.0E"``."""
    if is_sphere(cfg):
        return geom.place_text(np.asarray(place, dtype=np.float64)[:3])
    return f"({float(place[0]):.0f}, {float(place[1]):.0f})"
