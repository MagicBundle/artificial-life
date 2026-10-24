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
from collections.abc import Mapping
from typing import Any, Final

import numpy as np

from alife import weather
from alife.config import Config
from alife.weather import WeatherParams

TWO_PI: Final = 2.0 * math.pi
BAND_KEYS: Final = ("occ_hz0", "occ_hz1_11", "occ_hz12")  # peak agents per hash cell: floor, mid, surface
RAIN_BAND: Final = (8, 14)  # the cloud band of the rain shares: field layers 8-13 (z 32-56 at field_cell 4)
# The z bands of the coupling block and the dumps' analysis, as fractions of the world height (0-4, 4-32,
# 32-56, 56-80 at height 80): the floor layer, the lower column, the cloud band, the upper layer.
Z_BAND_FRACTIONS: Final = (0.0, 0.05, 0.4, 0.7, 1.0)
UB_QUANTILE: Final = 0.9  # the tracers' per-band top decile of g that marks an updraft boundary
HOLD_W_MIN: Final = 0.002  # the smallest |w| of the agents that enter the hold regression

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


def label_channel(mask: np.ndarray, group: np.ndarray | None = None) -> tuple[np.ndarray, int]:
    """``label_periodic`` for a channel: periodic in x only. The mask is padded with one empty row on each
    side before labelling and the padding is dropped after, so a region never joins across a wall (the
    labels are renumbered by the smallest flat index, whose order the padding keeps)."""
    mask = np.asarray(mask, dtype=bool)
    pad = ((1, 1), (0, 0))
    grp = None if group is None else np.pad(np.broadcast_to(np.asarray(group), mask.shape), pad)
    lab, k = label_periodic(np.pad(mask, pad), group=grp)
    return np.ascontiguousarray(lab[1:-1]), k


def centre_winds(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """u and v averaged to the cell centres: ``½(u + u[i-1])`` and ``½(v + v[j-1])``."""
    return 0.5 * (u + np.roll(u, 1, 1)), 0.5 * (v + np.roll(v, 1, 0))


def corners_to_centres(z: np.ndarray) -> np.ndarray:
    """``¼(ζ + W(ζ) + S(ζ) + S(W(ζ)))``: the four corners around each centre."""
    zw = np.roll(z, 1, 1)
    return 0.25 * (z + zw + np.roll(z, 1, 0) + np.roll(zw, 1, 0))


def hemisphere_sign(p: WeatherParams) -> np.ndarray:
    """``sign f(y_c)`` per centre row, (ny, 1) int8; never zero (centre rows avoid y = 0 and L/2). In a
    channel ``f`` is odd about the equator ``y = Ly/2``, so the north is ``y >= Ly/2``."""
    yc = (np.arange(p.ny) + 0.5) * p.dx
    if p.channel:
        return np.where(yc >= 0.5 * p.Ly, 1, -1).astype(np.int8)[:, None]
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


def _share(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def rain_shares(audit: np.ndarray) -> dict[str, float]:
    """The nominal and realised rain shares of DESIGN §32 from the per-layer sums of ``metrics.rain_audit``.

    ``audit`` is ``(4, layers)``: the regeneration ``base`` without rain, the nominal ``rain`` term, the
    realised increment ``inc_total = clip(min(base + rain, cap - R), 0)`` and the realised rain
    ``rain_real = inc_total - inc_base``, each summed over a layer. The band is ``RAIN_BAND`` (all of a 2D
    world's single layer). Nominal: ``rain / (rain + base)``, a statement about the forcing; realised:
    ``rain_real / inc_total``; ``rain_clipped_frac = 1 - Σ rain_real / Σ rain`` (0 without rain)."""
    a = np.asarray(audit, dtype=np.int64)
    base, rain, inc, real = a[0], a[1], a[2], a[3]
    lo, hi = RAIN_BAND if base.size > 1 else (0, 1)
    b = slice(lo, hi)
    return {
        "rain_band_share": _share(rain[b].sum(), rain[b].sum() + base[b].sum()),
        "rain_global_share": _share(rain.sum(), rain.sum() + base.sum()),
        "rain_real_band_share": _share(real[b].sum(), inc[b].sum()),
        "rain_real_global_share": _share(real.sum(), inc.sum()),
        "rain_clipped_frac": 1.0 - _share(real.sum(), rain.sum()) if rain.sum() else 0.0,
    }


def weather_block(
    wx: np.ndarray,
    air_speed: np.ndarray,
    cfg: Config,
    *,
    occupancy: np.ndarray | None = None,
    overlap_deep: int = 0,
    total_speed: np.ndarray | None = None,
    eddy_speed: np.ndarray | None = None,
    seed_r: np.ndarray | None = None,
    shade_s: np.ndarray | None = None,
    rain_audit: np.ndarray | None = None,
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

    W2 (DESIGN §32), each key group present only when its input is given (the engine's snapshot gives them
    when the coupling is on): ``total_speed`` and ``eddy_speed`` (the alive rows' ``|u_air|`` and eddy speed)
    give ``total_speed_p99`` and ``eddy_speed_p99`` (``air_speed_*`` and ``storm_exposure`` stay on the hazard
    speed); ``seed_r`` (the ``(FG, FG)`` seeding response) gives ``seed_r_mean``, ``seed_r_hi_share`` (columns
    with ``r > 0.5``) and ``seed_lower_mean`` (``seed_gain·mean r``); ``shade_s`` (the ``(FG, FG)`` shade)
    gives ``shade_s_mean`` and ``shade_s_max``; ``rain_audit`` (see ``rain_shares``) the rain shares.
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
    if total_speed is not None and eddy_speed is not None:
        for name, v in (("total", total_speed), ("eddy", eddy_speed)):
            v = np.asarray(v, dtype=np.float64).reshape(-1)
            block[f"{name}_speed_p99"] = float(np.quantile(v, 0.99)) if v.size else 0.0
    if seed_r is not None:
        r = np.asarray(seed_r, dtype=np.float64)
        block["seed_r_mean"] = float(r.mean())
        block["seed_r_hi_share"] = float((r > 0.5).mean())
        block["seed_lower_mean"] = float(wc.seed_gain * r.mean())
    if shade_s is not None:
        sh = np.asarray(shade_s, dtype=np.float64)
        block["shade_s_mean"] = float(sh.mean())
        block["shade_s_max"] = float(sh.max())
    if rain_audit is not None:
        block.update(rain_shares(rain_audit))
    return block


# --- the coupling block (W2, DESIGN §32) ---


def z_band_edges(height: float) -> np.ndarray:
    """The four z bands' five edges, ``Z_BAND_FRACTIONS · height``."""
    return np.asarray(Z_BAND_FRACTIONS, dtype=np.float64) * height


def z_band(z: np.ndarray, height: float) -> np.ndarray:
    """The band index 0-3 of each height (a height at the lid belongs to the top band)."""
    edges = z_band_edges(height)
    return np.clip(np.searchsorted(edges, np.asarray(z, dtype=np.float64), side="right") - 1, 0, 3)


def _opt(x: float) -> float | None:
    return None if not math.isfinite(x) else float(x)


def hold_index(vz: np.ndarray, w: np.ndarray) -> float | None:
    """``-Cov(vz, w)/Var(w)``: minus the slope of the agents' own vertical velocity on the air's, i.e. the
    **fraction of alpha countered**. The own velocity is ``vel_z`` minus the drift ``alpha·w`` (``alpha``
    0.25 to 1, about 0.2 to 0.6 for the s0 caps), so an agent that holds its altitude (``vel_z = 0``) has
    ``hold = alpha``, not 1; 0 ignores the wind (tracers and the WB arm are 0 by construction). Compare
    runs on the scale of the agents' ``alpha``, not of 1. None for fewer than two rows or no variance."""
    v, x = np.asarray(vz, dtype=np.float64), np.asarray(w, dtype=np.float64)
    if x.size < 2 or not x.var() > 0:
        return None
    return -float(((v - v.mean()) * (x - x.mean())).mean() / x.var())


def coupling_block(
    agents: Mapping[str, np.ndarray], tracers: Mapping[str, np.ndarray], height: float, field_cell: float
) -> dict[str, Any]:
    """The ``coupling`` block of a metrics row (DESIGN §32): the live agents against the tracers, stratified
    by z band (``Z_BAND_FRACTIONS``).

    ``agents`` holds 1-D arrays of the **live** rows: ``z``, ``g`` (the updraft-boundary index ``|∇_h w|`` at
    the agent), ``w`` (the air's vertical velocity there) and ``vz_own`` (the agent's own vertical velocity,
    ``vel_z`` minus the wind drift's); ``tracers`` holds ``z``, ``g`` and ``w``. Keys, each list of four per
    band:

    * ``n_agents``, ``n_tracers``: rows per band;
    * ``ub_threshold``: the tracers' ``UB_QUANTILE`` of ``g`` in the band (None if it has no tracer);
    * ``ub_agents``: the share of the band's agents with ``g`` above it (None without agents or threshold). A
      population drawn from the tracers' distribution gives 0.10; ``ub_agents_all`` is the share over all the
      agents of bands with a threshold (the null is 0.10 for any composition);
    * ``w_mean_agents``, ``w_mean_tracers``: the mean ``w`` per band;
    * ``hold`` / ``hold_band``: ``hold_index`` over the agents with ``|w| >= HOLD_W_MIN`` and more than one
    field   layer from the floor and the lid, overall and per band;
    * ``z_mean_agents``, ``z_mean_tracers``: the mean height.
    """
    az, ag, aw, av = (np.asarray(agents[k], dtype=np.float64) for k in ("z", "g", "w", "vz_own"))
    tz, tg, tw = (np.asarray(tracers[k], dtype=np.float64) for k in ("z", "g", "w"))
    ab, tb = z_band(az, height), z_band(tz, height)
    thr: list[float | None] = []
    ub: list[float | None] = []
    above = total = 0
    for b in range(4):
        gt = tg[tb == b]
        ga = ag[ab == b]
        t = float(np.quantile(gt, UB_QUANTILE)) if gt.size else math.nan
        thr.append(_opt(t))
        if gt.size and ga.size:
            hits = int((ga > t).sum())
            ub.append(hits / ga.size)
            above, total = above + hits, total + ga.size
        else:
            ub.append(None)
    sel = (np.abs(aw) >= HOLD_W_MIN) & (az >= field_cell) & (az < height - field_cell)
    return {
        "n_agents": [int((ab == b).sum()) for b in range(4)],
        "n_tracers": [int((tb == b).sum()) for b in range(4)],
        "ub_threshold": thr,
        "ub_agents": ub,
        "ub_agents_all": above / total if total else None,
        "w_mean_agents": [_opt(float(aw[ab == b].mean())) if (ab == b).any() else None for b in range(4)],
        "w_mean_tracers": [_opt(float(tw[tb == b].mean())) if (tb == b).any() else None for b in range(4)],
        "hold": hold_index(av[sel], aw[sel]),
        "hold_band": [hold_index(av[sel & (ab == b)], aw[sel & (ab == b)]) for b in range(4)],
        "z_mean_agents": _opt(float(az.mean())) if az.size else None,
        "z_mean_tracers": _opt(float(tz.mean())) if tz.size else None,
    }
