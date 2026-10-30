"""Storm detection and tracking on the sphere's weather cells (DESIGN §46.7, STORMS §6 on a planet).

The box tracker (``storms.py``) reads ``(N, N)`` periodic arrays: the footprint of a storm is a labelled
region of the cyclonic-vorticity mask on a torus, its centre a circular mean, its distances the torus'
minimum image. None of that exists on a sphere. The weather lives on the ``12·Nw²`` ring-ordered HEALPix
cells (equal area), so:

* a **footprint** is a connected component of the mask ``s·ζ >= zeta_th`` over the **edge graph** of the
  cells (``geom.label_components``), the hemispheres kept apart (``s`` is the sign of the Coriolis
  parameter, +1 where ``z >= 0``), so a region may cross a pole, a face seam or a trivalent vertex of the
  grid without notice;
* its **centre** is the weighted mean direction of its cells (``geom.mean_direction``, weights
  ``s·ζ − zeta_th``), stated as the planet-frame point ``(x, y, z)`` at the planet radius; its **radius**
  is ``sqrt(ΣA_c/π)`` with ``A_c`` the area of a cell at the floor, so a storm of a given size has the same
  radius at every place;
* a region is a **band** (excluded from storms, like the box's) if its mean resultant length is below 0.5 or
  its diameter (the largest arc between two of its cells, found by two sweeps from the centre) exceeds a
  quarter of a great circle, ``π R / 2`` (the sphere's "wider than half the world"; a ring round a pole is
  a band though its centre is the pole);
* the **tracker** is the box tracker with the sphere's geometry: distances are great-circle arcs, a
  prediction extrapolates the least-squares angular velocity of the last four detections along a great
  circle (``SphereTracker.predict``), and a detection carries its centre as ``x, y, z``. The overlap
  matching, the splits and mergers, the coasting and the numbering are the box code.

Everything is a function of directions and arcs, so a rotation of the planet rotates every result (tested on
synthetic vortices crossing the poles and the vertices of the grid). Host code, numpy only, float64.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np

from alife import geom, streamrec
from alife.storms import Comp, Det, Detection, Track, Tracker

__all__ = [
    "SphereComp",
    "SphereDet",
    "SphereFrame",
    "SphereParams",
    "SphereTracker",
    "detect",
]

BAND_RESULTANT = 0.5  # a footprint whose mean resultant length is below this is a band
BAND_EXTENT = (
    0.25  # ... or whose diameter exceeds this fraction of a great circle (2 pi R): wider than a quarter
)


@dataclass(frozen=True)
class SphereParams:
    """What the sphere's detection and tracking read: the weather ``nside``, the planet radius ``radius`` (the
    length unit of every arc), and the thresholds of ``WeatherParams`` (``zeta_th`` in 1/time, ``p_th`` rain,
    ``a_min`` and ``r_min`` in cells, ``miss_grace`` detections)."""

    nside: int
    radius: float
    zeta_th: float = 0.00625
    p_th: float = 1e-4
    a_min: int = 25
    r_min: int = 4
    miss_grace: int = 2

    @property
    def n_cells(self) -> int:
        return geom.npix(self.nside)

    @property
    def cell_area(self) -> float:
        """The area of a cell at the floor, ``4π R² / Npix`` (arc units squared)."""
        return 4.0 * math.pi * self.radius**2 / self.n_cells


class SphereFrame(NamedTuple):
    """A host snapshot of the weather at a detection time, on the ring-ordered cells: ``zeta`` the layer-1
    relative vorticity at the cell centres, ``P`` the rain rate, ``h1`` the layer-1 thickness and ``spd1`` the
    layer-1 wind speed at the centres; all ``(Npix,)`` float."""

    t: int
    zeta: np.ndarray
    P: np.ndarray
    h1: np.ndarray
    spd1: np.ndarray


@dataclass(frozen=True)
class SphereComp(Comp):
    """A ``Comp`` of the sphere: ``x, y, z`` is the centre at the planet radius (``rx`` and ``ry`` both
    hold the mean resultant length; ``ext_x`` and ``ext_y`` the diameter)."""

    z: float = 0.0


@dataclass
class SphereDet(Det):
    """A ``Det`` of the sphere: ``x, y, z`` the centre at the planet radius; ``ux, uy`` repeat ``x, y``
    (there is
    no unwrapped track)."""

    z: float = 0.0


def _arcs(centres: np.ndarray, direction: np.ndarray, radius: float) -> np.ndarray:
    """Great-circle arc lengths from ``direction`` to every cell centre."""
    cr = np.cross(centres, direction)
    return radius * np.arctan2(np.linalg.norm(cr, axis=1), centres @ direction)


def _side(centres: np.ndarray) -> np.ndarray:
    """The hemisphere sign of each cell, +1 north (``z >= 0``) else −1 (the sign of the Coriolis
    parameter)."""
    return np.where(centres[:, 2] >= 0.0, 1, -1).astype(np.int8)


def detect(fr: SphereFrame, p: SphereParams) -> Detection:
    """Storm footprints, bands and per-detection quantities from a frame (the box ``detect`` on the cell
    graph).

    Mask ``s·ζ >= zeta_th``, components over the edge graph per hemisphere, a band or a storm by the module
    rules; a storm needs ``area >= a_min`` cells and ``r_min`` cells with ``P >= p_th``. Centres are mean
    directions with weight ``s·ζ − zeta_th``. ``wind_max`` is the largest layer-1 speed within ``1.5 r_eq`` of
    the centre, ``dh1`` the mean ``h1`` over the annulus ``r_eq < d <= 2 r_eq`` less the footprint minimum."""
    n = p.n_cells
    centres = geom.pixel_centres(p.nside)
    s = _side(centres)
    zeta = np.asarray(fr.zeta, np.float64)
    sz = s * zeta
    th = p.zeta_th
    mask = sz >= th
    lab = np.zeros(n, np.int64)
    root = geom.label_components(p.nside, mask, group=s)
    ids = np.unique(root[mask])
    if ids.size == 0:
        return Detection(fr.t, lab, [], [], 0, 0)
    # labels 1..k in the order of the smallest pixel id of each component (deterministic)
    lut = np.zeros(n + 1, np.int64)
    lut[ids] = np.arange(1, ids.size + 1)
    lab = np.where(mask, lut[np.where(mask, root, n)], 0)
    k = ids.size
    h1 = np.asarray(fr.h1, np.float64)
    pr = np.asarray(fr.P, np.float64)
    spd = np.asarray(fr.spd1, np.float64)
    cell_area = p.cell_area
    storms: list[SphereComp] = []
    bands: list[SphereComp] = []
    n_weak = 0
    for q in range(k):
        cells = np.flatnonzero(lab == q + 1)
        w = np.maximum(sz[cells] - th, 0.0) + 1e-9 * th  # the floor only guards an all-zero weight sum
        direction, resultant = geom.mean_direction(centres[cells], w)
        area = int(cells.size)
        r_eq = math.sqrt(area * cell_area / math.pi)
        arc_f = _arcs(centres[cells], direction, p.radius)
        r_max = float(arc_f.max())
        far = cells[int(np.argmax(arc_f))]  # the cell farthest from the centre, then the one farthest from it
        diameter = float(_arcs(centres[cells], centres[far], p.radius).max())
        band = bool(resultant < BAND_RESULTANT or diameter > BAND_EXTENT * 2.0 * math.pi * p.radius)
        n_rain = int((pr[cells] >= p.p_th).sum())
        storm = (not band) and area >= p.a_min and n_rain >= p.r_min
        if not band and not storm:
            n_weak += 1
            continue
        hemi = 0 if s[cells[0]] > 0 else 1
        zmax = float(sz[cells].max()) / streamrec.F0
        h1min = float(h1[cells].min())
        rain = float(pr[cells].sum()) * cell_area
        px, py, pz = (p.radius * direction).tolist()
        if band:
            bands.append(
                SphereComp(q + 1, hemi, px, py, area, r_eq, 0.0, zmax, 0.0, h1min, 0.0, rain, n_rain,
                           resultant, resultant, diameter, diameter, cells, pz)
            )  # fmt: skip
            continue
        d_all = _arcs(centres, direction, p.radius)
        wind_max = float(spd[d_all <= 1.5 * r_eq].max(initial=0.0))
        ring = (d_all > r_eq) & (d_all <= 2.0 * r_eq)
        dh1 = float(h1[ring].mean() - h1min) if ring.any() else 0.0
        storms.append(
            SphereComp(q + 1, hemi, px, py, area, r_eq, r_max, zmax, wind_max, h1min, dh1, rain, n_rain,
                       resultant, resultant, diameter, diameter, cells, pz)
        )  # fmt: skip
    return Detection(fr.t, lab, storms, bands, k, n_weak)


class SphereTracker(Tracker):
    """The box ``Tracker`` with great-circle distances and a great-circle extrapolation. ``p`` is a
    ``SphereParams``; ``Tracker.update`` reads only its ``miss_grace``."""

    def __init__(self, p: SphereParams) -> None:
        super().__init__(p)  # type: ignore[arg-type]
        self.p: SphereParams = p

    @staticmethod
    def _dir(x: float, y: float, z: float) -> np.ndarray:
        return geom.unit(np.array([x, y, z], dtype=np.float64))

    def _gap(self, c: Comp, last: Det) -> float:
        return float(
            geom.great_circle_distance(
                self._dir(c.x, c.y, c.z),
                self._dir(last.x, last.y, last.z),
                self.p.radius,  # type: ignore[attr-defined]
            )
        )

    def _gap_to(self, c: Comp, predicted: tuple[float, ...]) -> float:
        return float(
            geom.great_circle_distance(
                self._dir(c.x, c.y, c.z),
                self._dir(*predicted),
                self.p.radius,  # type: ignore[attr-defined]
            )
        )

    def predict(self, tr: Track, t: int) -> tuple[float, float, float]:  # type: ignore[override]
        """The position of ``tr`` at ``t`` on the planet: its last centre moved along a great circle by the
        least-squares angular velocity of its last four detections times the elapsed time. Each earlier centre
        is taken to the tangent plane of the last one (the logarithm map, an arc in the direction of the
        centre), the slope of those vectors against time is the velocity, the exponential map brings the
        displacement back."""
        d = tr.dets[-4:]
        last = d[-1]
        u0 = self._dir(last.x, last.y, last.z)  # type: ignore[attr-defined]
        if len(d) < 2:
            return tuple((self.p.radius * u0).tolist())  # type: ignore[return-value]
        v = []
        for q in d:
            u = self._dir(q.x, q.y, q.z)  # type: ignore[attr-defined]
            theta = math.atan2(float(np.linalg.norm(np.cross(u0, u))), float(u0 @ u))
            tang = u - (u @ u0) * u0
            nrm = float(np.linalg.norm(tang))
            v.append(theta * tang / nrm if nrm > 1e-15 else np.zeros(3))
        ts = np.array([q.t for q in d], dtype=np.float64)
        tc = ts - ts.mean()
        den = float((tc * tc).sum())
        slope = (tc[:, None] * np.array(v)).sum(axis=0) / den  # radians per step, in the tangent plane of u0
        step = slope * (t - last.t)
        phi = float(np.linalg.norm(step))
        if phi < 1e-15:
            return tuple((self.p.radius * u0).tolist())  # type: ignore[return-value]
        moved = math.cos(phi) * u0 + math.sin(phi) * step / phi
        return tuple((self.p.radius * moved).tolist())  # type: ignore[return-value]

    def _append(self, tr: Track, t: int, c: Comp) -> None:
        z = float(getattr(c, "z", 0.0))
        tr.dets.append(
            SphereDet(
                t, c.x, c.y, c.x, c.y, c.r_eq, c.area, c.hemi, c.zeta_max_f0, c.wind_max, c.h1_min, c.dh1,
                c.rain, z,
            )
        )  # fmt: skip
        tr.last_flat = c.flat
        tr.missed = 0


def vortex_field(
    nside: int, centre: np.ndarray, sigma: float, radius: float, amplitude: float = 0.05
) -> np.ndarray:
    """A synthetic cyclone for tests and calibration: ``ζ = s·A·exp(−d²/2σ²)`` on the cells, ``d`` the arc to
    ``centre`` (a direction) and ``s`` the hemisphere sign of the **cell** (cyclonic is positive
    vorticity in the
    north and negative in the south, the box convention)."""
    centres = geom.pixel_centres(nside)
    d = _arcs(centres, geom.unit(np.asarray(centre, np.float64)), radius)
    return _side(centres) * amplitude * np.exp(-(d**2) / (2.0 * sigma**2))


def frame_of_vortices(nside: int, t: int, vortices: list[dict[str, Any]], radius: float) -> SphereFrame:
    """A ``SphereFrame`` that is the sum of synthetic vortices (each ``{"centre", "sigma", "amp"}``):
    vorticity as ``vortex_field``, rain ``0.004`` where the vorticity is above half its threshold scale,
    ``h1`` 1 less the vorticity, ``spd1`` the sum of ``σ·ζ``-scaled peaks. Pure function of the
    arguments."""
    zeta = np.zeros(geom.npix(nside))
    for v in vortices:
        zeta = zeta + vortex_field(nside, v["centre"], v["sigma"], radius, v.get("amp", 0.05))
    s = _side(geom.pixel_centres(nside))
    return SphereFrame(
        t,
        zeta,
        np.where(s * zeta > 0.005, 0.004, 0.0),
        1.0 - np.abs(zeta),
        np.abs(zeta) * 10.0,
    )
