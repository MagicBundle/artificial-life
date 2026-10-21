"""The ``weather`` block of a metrics row and the host helpers it uses (W1, DESIGN §30).

Host-side numpy over one snapshot of ``State.wx`` (the nine weather arrays), the air speed the agents feel and
a few engine counts. It is not a simulation module (not under the §22.6 lint scope): the runner calls it once
per metrics row, after the one device transfer of ``metrics.snapshot_metrics``. The block describes the
weather and its coupling to the organisms; it never feeds back into the step.

``label_periodic``, ``centre_winds``, ``corners_to_centres`` and ``hemisphere_sign`` were written for the W0
study driver (``analysis/w0.py``, which keeps the names importable) and moved here so that the engine's
metrics do not import the 4,000-line driver.
"""

from __future__ import annotations

import math
from typing import Any, Final

import numpy as np

from alife import weather
from alife.config import Config
from alife.weather import WeatherParams

TWO_PI: Final = 2.0 * math.pi
BAND_KEYS: Final = ("occ_hz0", "occ_hz1_11", "occ_hz12")  # peak agents per hash cell: floor, mid, surface

# --- storm detection helpers (moved from analysis/w0.py) --------------------------------------------------

_NEIGHBOURS = tuple((dj, di) for dj in (-1, 0, 1) for di in (-1, 0, 1) if dj or di)


def label_periodic(mask: np.ndarray, group: np.ndarray | None = None) -> tuple[np.ndarray, int]:
    """Periodic 8-connected component labelling by min-label propagation (deterministic).

    Every masked cell starts with its flat index plus one and takes the minimum label over its eight rolled
    neighbours until a fixed point (with pointer jumping, which only shortens the iteration), so seams and
    corners join. Cells join only when ``group`` agrees (the study passes the hemisphere sign, so a region
    never crosses an equator). Labels are renumbered ``1 .. K`` in order of the smallest flat index of each
    component; ``0`` is the background. Returns ``(labels (N, M) int32, K)``."""
    mask = np.asarray(mask, dtype=bool)
    size = mask.size
    big = size + 1
    lab = np.where(mask, np.arange(1, size + 1, dtype=np.int64).reshape(mask.shape), 0)
    if group is not None:
        group = np.broadcast_to(np.asarray(group), mask.shape)
        same = {s: np.roll(group, s, (0, 1)) == group for s in _NEIGHBOURS}
    while True:
        cur = np.where(mask, lab, big)
        new = cur.copy()
        for s in _NEIGHBOURS:
            nb = np.roll(cur, s, (0, 1))
            if group is not None:
                nb = np.where(same[s], nb, big)
            new = np.minimum(new, nb)
        new = np.where(mask, new, 0)
        jump = np.where(mask, new.ravel()[np.maximum(new.ravel() - 1, 0)].reshape(mask.shape), 0)
        new = np.where(mask, np.minimum(new, np.where(jump > 0, jump, new)), 0)
        if np.array_equal(new, lab):
            break
        lab = new
    out = np.zeros(mask.shape, dtype=np.int32)
    if mask.any():
        _, inv = np.unique(lab[mask], return_inverse=True)
        out[mask] = inv.reshape(-1) + 1
    return out, int(out.max())


def centre_winds(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """u and v averaged to the cell centres: ``½(u + u[i-1])`` and ``½(v + v[j-1])``."""
    return 0.5 * (u + np.roll(u, 1, 1)), 0.5 * (v + np.roll(v, 1, 0))


def corners_to_centres(z: np.ndarray) -> np.ndarray:
    """``¼(ζ + W(ζ) + S(ζ) + S(W(ζ)))``: the four corners around each centre."""
    zw = np.roll(z, 1, 1)
    return 0.25 * (z + zw + np.roll(z, 1, 0) + np.roll(zw, 1, 0))


def hemisphere_sign(p: WeatherParams) -> np.ndarray:
    """``sign f(y_c)`` per centre row, (N, 1) int8; never zero (centre rows avoid y = 0 and L/2)."""
    yc = (np.arange(p.n) + 0.5) * p.dx
    return np.where(np.sin(TWO_PI * yc / p.L) >= 0, 1, -1).astype(np.int8)[:, None]


# --- the block ----------------------------------------------------------------------------------------------


def layer1_vorticity(u1: np.ndarray, v1: np.ndarray, dx: float) -> np.ndarray:
    """Layer-1 relative vorticity at the corners ``((i+1)dx, (j+1)dx)``, the numpy form of
    ``weather.vorticity`` (float64 in, float64 out)."""
    return (np.roll(v1, -1, 1) - v1) / dx - (np.roll(u1, -1, 0) - u1) / dx


def storm_exposure(speed: np.ndarray, u0: float, span: float) -> float:
    """The mean hazard ramp ``clip((speed - u0)/span, 0, 1)`` over the given air speeds (0 for none): what
    ``storm_p`` multiplies to give the population-mean storm mortality per step."""
    if speed.size == 0:
        return 0.0
    return float(np.clip((speed.astype(np.float64) - u0) / span, 0.0, 1.0).mean())


def weather_block(
    wx: np.ndarray,
    air_speed: np.ndarray,
    cfg: Config,
    *,
    occupancy: np.ndarray | None = None,
    overlap_deep: int = 0,
) -> dict[str, Any]:
    """The ``weather`` block of one metrics row (DESIGN §30), plain floats and ints.

    ``wx`` is the stored ``State.wx`` ``(N_WX, n, n)``; ``air_speed`` the air speed (jet included) of the
    **alive** rows at their positions (1-D, possibly empty); ``occupancy`` the 3D peak agents per hash cell in
    the floor, mid-column and surface bands (``(3,)`` ints, or None in 2D, which then has no ``occ_*`` keys);
    ``overlap_deep`` the alive agents with a neighbour closer than ``0.5·radius``.

    The wind and rain maxima and the conservation figures are of the stored state, the state after the last
    step. ``storm_proxy`` counts the 8-connected periodic components of the cyclonic-vorticity mask (per
    hemisphere) of area ``a_min`` or more with at least ``r_min`` raining cells: a count of candidate storms,
    not of tracks. ``storm_exposure`` and the air-speed quantiles are at the snapshot's positions and are the
    table the calibration of ``storm_u0`` reads.
    """
    p = weather.params_from_config(cfg)
    wc = cfg.weather
    h1, h2, u1, v1, u2, v2, q, rain = (np.asarray(wx[i], dtype=np.float64) for i in range(8))
    n, dx = p.n, p.dx

    uc1, vc1 = centre_winds(u1, v1)
    uc2, vc2 = centre_winds(u2, v2)
    s = hemisphere_sign(p)
    zc = s * corners_to_centres(layer1_vorticity(u1, v1, dx))
    vort = zc >= p.zeta_th
    raining = rain >= p.p_th
    lab, k = label_periodic(vort, group=np.broadcast_to(s, (n, n)))
    storms = 0
    if k:
        area = np.bincount(lab.ravel(), minlength=k + 1)[1:]
        wet = np.bincount(lab[raining & vort].ravel(), minlength=k + 1)[1:]
        storms = int(((area >= p.a_min) & (wet >= p.r_min)).sum())

    h = h1 + h2
    dwx, dwy = uc1 - uc2, vc1 - vc2
    speed = np.asarray(air_speed, dtype=np.float64).reshape(-1)
    block: dict[str, Any] = {
        "wind_max_l1": float(np.hypot(uc1, vc1).max()),
        "wind_max_l2": float(np.hypot(uc2, vc2).max()),
        "jet_max_l2": float(np.abs(u2.mean(axis=1)).max()),
        "rain_total": float(rain.sum() * dx * dx),
        "rain_cells": float(raining.mean()),
        "vort_cells": float(vort.mean()),
        "storm_proxy": storms,
        "mass_rel": float(h.sum() / (n * n * (p.H1 + p.H2)) - 1.0),
        "q_total": float(q.sum()),
        "hyper_max": float(((dwx * dwx + dwy * dwy) / (p.g_prime * h)).max()),
    }
    if occupancy is not None:
        block.update({key: int(v) for key, v in zip(BAND_KEYS, occupancy, strict=True)})
    block.update(
        {
            "storm_exposure": storm_exposure(speed, wc.storm_u0, wc.storm_span),
            "air_speed_p99": float(np.quantile(speed, 0.99)) if speed.size else 0.0,
            "air_speed_max": float(speed.max()) if speed.size else 0.0,
            "overlap_deep": int(overlap_deep),
        }
    )
    return block
