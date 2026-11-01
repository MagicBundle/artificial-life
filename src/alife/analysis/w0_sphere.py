"""The W0-style study driver of the sphere's weather (DESIGN §46.6, §46.9, docs/SPHERE.md S4c).

``python -m alife.analysis.w0_sphere run --preset 64k3s --overlay stormsph --steps 100000 --out DIR`` advances
the engine's own sphere weather alone (``weather_spinup.advance_chain`` from the calm rest state, the engine's
key chain, so the weather is the one a run with that seed would see), detects and tracks storms with the
storm layer's detector and tracker (``storms_sph``), and writes, per hemisphere:

* the **count** of storms at every detection tick of the window, read three ways: the *named* storms active at
  the tick (named in real time, as the storm layer of ``alife play`` names them: a track whose peak
  lower-layer vorticity so far reaches ``NAME_ZETA`` = 0.5 f0), every active track, and the W0 criterion's
  own D5 count (qualifying tracks, 3 detections or more, alive between their first and last detection,
  ``w0.storm_counts``); the share of ticks inside 2 to 6 in both hemispheres;
* the **lifetimes**: the Kaplan-Meier median with left truncation of ``w0.lifetime_stats`` over the named
  storms (peak over the whole track, the ruling C11), over the storms named in real time (age from the
  naming) and over every qualifying track, with the criterion's 3 000 steps;
* the **tracks** (``tracks.csv``): id, hemisphere, first and last step, lifetime, peak vorticity, the great
  circle length of the track, the latitude, whether the storm was named and when;
* the **exposure** of the named storms to the grid's irregularities (``exposure``): the 8 trivalent vertices
  of HEALPix (at ``z = ±2/3``, longitudes 0, 90, 180 and 270 degrees), the cells of the face seams (cells
  touching a cell of another one of the 12 base faces), the 8 pole cells, and the latitude circle
  ``|z| = 2/3`` where the rings change from the polar caps' to the equatorial belt's. Observed shares are
  set against a latitude-matched null (a storm of the same latitude at a uniformly drawn longitude);
* the **climate** (``series.csv``): mean kinetic energy, humidity, rain and the wind maxima every few ticks,
  which is how the spin-up length is read.

``sweep`` runs a list of candidates (JSON) in parallel worker processes within a memory budget, one
``run`` per candidate and seed. ``table`` prints a comparison of finished runs.

Study code, never imported by the simulation: it reads arrays to the host whenever it needs to, writes only
inside ``--out`` and touches no run directory. Time is in engine steps; one step is ``dt`` units of weather
time (DESIGN §46.6), the pool drift ``omega`` is per unit of weather time.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import functools
import json
import math
import os
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from alife import geom, weather_sph
from alife.analysis import w0
from alife.config import Config, load_config, validate
from alife.rng import InitStream, init_keys
from alife.storms import NAME_ZETA, Track
from alife.storms_sph import SphereFrame, SphereParams, SphereTracker, detect
from alife.weather_spinup import advance_chain, chain_step, spinup_key

__all__ = [
    "POOL_LATITUDE",
    "Exposure",
    "StudySettings",
    "build_config",
    "count_stats",
    "default_omega",
    "exposure_stats",
    "make_pools",
    "run_study",
    "seam_cells",
    "trivalent_vertices",
]

POOL_LATITUDE = 40.0  # degrees: the latitude of the pools of every sphere overlay (f at the pool 0.0162)
COUNT_LO, COUNT_HI = 2, 6  # the STORMS_W0 count criterion, per hemisphere
LIFETIME_MIN = 3000.0  # the STORMS_W0 lifetime criterion, steps
MIN_DETS = 3  # the W0 qualifying-track minimum (WeatherParams.min_dets)
RING_Z = 2.0 / 3.0  # |z| where HEALPix's rings change from the polar caps' to the belt's


# --- the pools ----------------------------------------------------------------------------------------------


def default_omega() -> float:
    """The angular drift of the pools of ``stormsph``: ``-0.009`` arc per unit of weather time along the
    latitude circle of 40 degrees on the planet of radius 90, i.e. ``-0.009/(90 cos 40°)`` rad (the other
    overlays use the same angular rate: -0.0048 at Rp 48, -0.0036 at Rp 36)."""
    return -0.009 / (90.0 * math.cos(math.radians(POOL_LATITUDE)))


def make_pools(
    radius: float,
    per_hemisphere: int,
    sx: float,
    sy: float,
    *,
    lat: float = POOL_LATITUDE,
    amp: float = 1.0,
    omega: float | None = None,
    phase: float = 60.0,
    south_offset: float | None = None,
) -> tuple[float, ...]:
    """``weather.pools`` for ``per_hemisphere`` equal pools in each hemisphere at latitude ``±lat`` degrees,
    spaced ``360/per_hemisphere`` degrees apart in longitude, the northern first at ``phase`` degrees and
    the southern offset by ``south_offset`` (default half a spacing) from them. ``sx`` and ``sy`` are the
    Gaussian widths in length, ``amp`` the amplitude and ``omega`` the angular drift in radians per unit of
    weather time (default ``default_omega()``; the pool's ``vx`` is ``omega·R·cos(lat)``, the arc speed
    along its own latitude circle)."""
    if per_hemisphere < 1:
        raise ValueError("per_hemisphere must be >= 1")
    om = default_omega() if omega is None else omega
    off = 180.0 / per_hemisphere if south_offset is None else south_offset
    vx = om * radius * math.cos(math.radians(lat))
    out: list[float] = []
    for sign in (1.0, -1.0):
        lons = sorted(
            (phase + 360.0 * i / per_hemisphere + (0.0 if sign > 0 else off)) % 360.0
            for i in range(per_hemisphere)
        )
        for lon in lons:
            out += [radius * math.radians(lon), sign * radius * math.radians(lat), sx, sy, amp, vx, 0.0]
    return tuple(round(v, 6) for v in out)


# --- the configuration of a candidate -----------------------------------------------------------------------


def build_config(
    preset: str,
    overlays: Sequence[str] = (),
    sets: Sequence[str] = (),
    pools: dict[str, Any] | Sequence[float] | None = None,
) -> Config:
    """The config of a candidate: preset, overlays, ``section.key=value`` overrides, then ``pools`` (the
    keywords of ``make_pools`` as a dict, taking the radius from the planet, or a flat list), validated."""
    cfg = load_config(preset=preset, overlays=list(overlays), overrides=list(sets))
    if pools is not None:
        flat = (
            make_pools(cfg.world.radius, **pools)  # type: ignore[arg-type]
            if isinstance(pools, dict)
            else tuple(float(v) for v in pools)
        )
        cfg = dataclasses.replace(cfg, weather=dataclasses.replace(cfg.weather, pools=flat))
    validate(cfg)
    if not cfg.weather.enabled:
        raise ValueError("the study needs weather.enabled (compose a stormsph overlay)")
    return cfg


# --- the grid's irregularities ------------------------------------------------------------------------------


@functools.cache
def trivalent_vertices(nside: int) -> np.ndarray:
    """The 8 trivalent vertices of the grid as unit vectors ``(8, 3)``: the corners shared by exactly three
    cells, at ``z = ±2/3`` and longitudes 0, 90, 180 and 270 degrees. Taken from the cell tables and the exact
    corner positions of ``geom.xyf2vec``."""
    t = geom.tables(nside)
    flat = np.asarray(t.vertex).reshape(-1)
    val = np.bincount(flat, minlength=t.n_vertices)
    ix, iy, face = geom.pix2xyf(nside, np.arange(t.npix))
    corner = np.stack(
        [geom.xyf2vec((ix + cx) / nside, (iy + cy) / nside, face) for cx, cy in geom.CORNERS], axis=1
    ).reshape(-1, 3)
    first = np.unique(flat[val[flat] == 3], return_index=True)[1]
    return corner[val[flat] == 3][first]


@functools.cache
def seam_cells(nside: int) -> np.ndarray:
    """Boolean ``(Npix,)``: the cell touches a cell of another one of the 12 base faces (the cells along the
    seams of the grid). The 8 pole cells are marked by ``pole_cells``."""
    t = geom.tables(nside)
    _, _, face = geom.pix2xyf(nside, np.arange(t.npix))
    tc = np.asarray(t.touch)
    other = np.where(tc >= 0, face[np.maximum(tc, 0)] != face[:, None], False)
    return other.any(axis=1)


@functools.cache
def pole_cells(nside: int) -> np.ndarray:
    """Boolean ``(Npix,)``: the 4 cells at each pole, those that share the pole vertex (the 4 cells of the
    ring nearest the pole)."""
    z = np.sin(np.radians(geom.to_latlon(geom.pixel_centres(nside))[0]))
    out = np.zeros(z.size, dtype=bool)
    out[np.argsort(-z)[:4]] = True
    out[np.argsort(z)[:4]] = True
    return out


@dataclass(frozen=True)
class Exposure:
    """What the exposure statistics are set against: the unit-sphere geometry of one grid (read-only)."""

    nside: int
    side: float  # the mean cell side (length units)
    radius: float
    vertices: np.ndarray  # (8, 3)
    seam_centres: np.ndarray  # (n, 3) centres of the seam cells
    seam_mask: np.ndarray  # (Npix,)
    pole_mask: np.ndarray  # (Npix,)

    @classmethod
    def of(cls, nside: int, radius: float) -> Exposure:
        c = geom.pixel_centres(nside)
        sm = seam_cells(nside)
        return cls(nside, weather_sph.cell_side(radius, nside), radius, trivalent_vertices(nside), c[sm], sm,
                   pole_cells(nside))  # fmt: skip


def _near(points: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """The smallest angle (radians) from each of ``points`` (m, 3) to any of ``targets`` (n, 3) (unit
    vectors)."""
    out = np.full(points.shape[0], np.inf)
    for i in range(0, targets.shape[0], 512):
        d = points @ targets[i : i + 512].T
        out = np.minimum(out, np.arccos(np.clip(d.max(axis=1), -1.0, 1.0)))
    return out


EXPOSURE_THRESHOLDS = (1.5, 3.0, 6.0)  # cell sides: the distances within which exposure is counted


def _null_shares(ex: Exposure, lat_deg: np.ndarray, n_lon: int = 720) -> dict[str, np.ndarray]:
    """For each latitude (degrees, 1-degree bins), the share of longitudes whose point lies within each
    threshold of a vertex and of a seam cell, and in a seam cell, a pole cell: the expectation for a storm
    of that latitude placed at a uniform longitude. Returns arrays of shape ``(len(lat_deg), n_thr)``."""
    bins = np.round(lat_deg).astype(int)
    uniq = np.unique(bins)
    lon = np.linspace(0.0, 2.0 * math.pi, n_lon, endpoint=False)
    res_v = np.zeros((uniq.size, len(EXPOSURE_THRESHOLDS)))
    res_s = np.zeros_like(res_v)
    res_cell = np.zeros((uniq.size, 2))
    for k, la in enumerate(uniq):
        lr = math.radians(float(la))
        pts = np.stack(
            [math.cos(lr) * np.cos(lon), math.cos(lr) * np.sin(lon), np.full(n_lon, math.sin(lr))], 1
        )
        dv = _near(pts, ex.vertices) * ex.radius / ex.side
        ds = _near(pts, ex.seam_centres) * ex.radius / ex.side
        res_v[k] = [(dv <= th).mean() for th in EXPOSURE_THRESHOLDS]
        res_s[k] = [(ds <= th).mean() for th in EXPOSURE_THRESHOLDS]
        cell = geom.ang2pix(pts, ex.nside)
        res_cell[k] = [ex.seam_mask[cell].mean(), ex.pole_mask[cell].mean()]
    idx = np.searchsorted(uniq, bins)
    return {"vertex": res_v[idx], "seam": res_s[idx], "cells": res_cell[idx]}


def exposure_stats(
    ex: Exposure,
    centres: np.ndarray,
    track_id: np.ndarray,
    zeta: np.ndarray | None = None,
    boot: int = 400,
    seed: int = 0,
) -> dict[str, Any]:
    """The exposure of storms to the grid: ``centres`` (m, 3) are the unit directions of the detections of the
    named storms, ``track_id`` (m,) the track of each. For each distance threshold (in cell sides) the
    observed share of detections within it of a trivalent vertex and of a seam cell, the latitude-matched null
    share, and their ratio; the shares of detections in a seam cell and in a pole cell; the quantiles of
    ``|lat|``; the share within one cell side of the ring seam ``|z| = 2/3``. The shares are per detection
    and per track (each track weighted equally, which is what a bootstrap over tracks resamples; the 95 %
    interval of the ratio is that bootstrap). Detections of one track are strongly correlated, so the number
    of independent samples is the number of tracks: a ratio is not to be read without its interval. Storms
    ride their pools, which keep one latitude and sweep every longitude at a fixed rate whatever the seed, so
    the draws of different seeds are not independent either, and the ring statistic has no null (it is the
    latitude of the pools against ``|z| = 2/3``): the controlled test of a seam is a pool placed on it and one
    placed off it (``w0_sphere run --pools`` with ``omega`` 0)."""
    m = centres.shape[0]
    if m == 0:
        return {"detections": 0, "tracks": 0}
    lat = geom.to_latlon(centres)[0]
    null = _null_shares(ex, lat)
    dv = _near(centres, ex.vertices) * ex.radius / ex.side
    ds = _near(centres, ex.seam_centres) * ex.radius / ex.side
    cell = geom.ang2pix(centres, ex.nside)
    ids = np.unique(track_id)
    k = ids.size
    # per-track weighted means: each detection weighs 1/(its track's detections)
    inv = np.searchsorted(ids, track_id)
    count = np.bincount(inv, minlength=k).astype(np.float64)
    w_track = 1.0 / count[inv]

    def share(obs: np.ndarray, nul: np.ndarray) -> dict[str, Any]:
        o_det, n_det = float(obs.mean()), float(nul.mean())
        o_tr = np.bincount(inv, weights=obs.astype(np.float64) * w_track, minlength=k)
        n_tr = np.bincount(inv, weights=nul * w_track, minlength=k)
        rng = np.random.default_rng(seed)
        ratios = []
        for _ in range(boot):
            pick = rng.integers(0, k, size=k)
            nn = n_tr[pick].mean()
            ratios.append(o_tr[pick].mean() / nn if nn > 0 else math.nan)
        lo, hi = (
            (float(v) for v in np.nanpercentile(ratios, [2.5, 97.5]))
            if np.isfinite(ratios).any()
            else (math.nan,) * 2
        )
        nt = float(n_tr.mean())
        return {
            "observed": o_det,
            "null": n_det,
            "ratio": o_det / n_det if n_det > 0 else math.nan,
            "observed_per_track": float(o_tr.mean()),
            "null_per_track": nt,
            "ratio_per_track": float(o_tr.mean() / nt) if nt > 0 else math.nan,
            "ratio_per_track_ci95": [lo, hi],
        }

    out: dict[str, Any] = {
        "detections": int(m),
        "tracks": int(k),
        "thresholds_cell_sides": list(EXPOSURE_THRESHOLDS),
        "vertex": {
            f"within_{th:g}": share(dv <= th, null["vertex"][:, i])
            for i, th in enumerate(EXPOSURE_THRESHOLDS)
        },
        "seam": {
            f"within_{th:g}": share(ds <= th, null["seam"][:, i]) for i, th in enumerate(EXPOSURE_THRESHOLDS)
        },
        "in_seam_cell": share(ex.seam_mask[cell], null["cells"][:, 0]),
        "in_pole_cell": {
            "observed": float(ex.pole_mask[cell].mean()),
            "null": float(null["cells"][:, 1].mean()),
        },
        "abs_lat_quantiles": {
            str(q): float(np.percentile(np.abs(lat), q)) for q in (0, 5, 25, 50, 75, 95, 100)
        },
        "nearest_vertex_cell_sides_median": float(np.median(dv)),
    }
    ring_lat = math.degrees(math.asin(RING_Z))
    d_ring = np.abs(np.abs(lat) - ring_lat) * math.pi / 180.0 * ex.radius / ex.side
    out["ring_seam"] = {
        "latitude": ring_lat,
        "within_1_cell_side": float((d_ring <= 1.0).mean()),
        "median_distance_cell_sides": float(np.median(d_ring)),
    }
    if zeta is not None:
        near = dv <= EXPOSURE_THRESHOLDS[1]
        out["peak_zeta_near_vertex"] = {
            "within_3": float(zeta[near].mean()) if near.any() else None,
            "elsewhere": float(zeta[~near].mean()) if (~near).any() else None,
        }
    return out


# --- the study ----------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class StudySettings:
    """One study: ``steps`` engine steps of weather alone from the rest state, the statistics over the window
    from ``spinup`` on, a detection every ``tick`` steps (25 is the engine's ``detect_every``), the state key
    of ``seed``, the climate sampled every ``series_every`` ticks."""

    steps: int = 100_000
    spinup: int = 20_000
    seed: int = 0
    tick: int = 25
    series_every: int = 20
    min_dets: int = MIN_DETS
    exposure: bool = True


def _mean_latitude(tr: Track, radius: float) -> float:
    """The mean latitude (degrees) of the detections of a track."""
    z = np.array([d.z for d in tr.dets], dtype=np.float64) / radius  # type: ignore[attr-defined]
    return float(np.degrees(np.arcsin(np.clip(z, -1.0, 1.0))).mean())


def _arc_length(tr: Track, radius: float) -> float:
    """The great circle length of a track (the sum of the arcs between consecutive detections)."""
    if len(tr.dets) < 2:
        return 0.0
    v = np.array([[d.x, d.y, d.z] for d in tr.dets], dtype=np.float64)  # type: ignore[attr-defined]
    return float(geom.great_circle_distance(v[:-1], v[1:], radius).sum())


def _climate(wx: mx.array, p: Any) -> dict[str, float]:
    a = np.array(wx, dtype=np.float64)
    u1 = (a[2:5] ** 2).sum(0)
    u2 = (a[5:8] ** 2).sum(0)
    return {
        "ke": float(0.5 * (u1 + u2).mean()),
        "ke1": float(0.5 * u1.mean()),
        "ke2": float(0.5 * u2.mean()),
        "umax1": float(np.sqrt(u1.max())),
        "umax2": float(np.sqrt(u2.max())),
        "q_mean": float(a[8].mean()),
        "p_mean": float(a[9].mean()),
        "p_max": float(a[9].max()),
        "rain_share": float((a[9] >= p.p_th).mean()),
        "dc_mean": float(a[10].mean()),
        "h1_min": float(a[0].min()),
        "h1_max": float(a[0].max()),
    }


def count_stats(
    n_named: np.ndarray, n_all: np.ndarray, lo: int = COUNT_LO, hi: int = COUNT_HI
) -> dict[str, Any]:
    """Counts per hemisphere over the ticks of the window: ``n_named`` and ``n_all`` are ``(ticks, 2)``
    integer arrays (north, south). Per reading: the mean, minimum and maximum per hemisphere, the share of
    ticks inside ``lo..hi`` per hemisphere and in both at once, and the histogram."""

    def one(n: np.ndarray) -> dict[str, Any]:
        ok = (n >= lo) & (n <= hi)
        d: dict[str, Any] = {"in_range_both": float(ok.all(axis=1).mean()) if len(n) else math.nan}
        for h, name in enumerate(("north", "south")):
            d[name] = {
                "mean": float(n[:, h].mean()),
                "min": int(n[:, h].min()),
                "max": int(n[:, h].max()),
                "in_range": float(ok[:, h].mean()),
                "histogram": {
                    int(k): int(v) for k, v in zip(*np.unique(n[:, h], return_counts=True), strict=True)
                },
            }
        return d

    if len(n_named) == 0:
        return {}
    return {"named": one(n_named), "active": one(n_all)}


def _lifetime(tracks: Sequence[Track], t_spin: int, min_dets: int) -> dict[str, Any]:
    st = w0.lifetime_stats(list(tracks), t_spin, min_dets, LIFETIME_MIN)
    keys = (
        *("n", "events", "censored", "median", "median_text", "max_age"),
        *("n_risk_at_median", "thin_age", "median_reliable", "survival_before_threshold", "pass"),
    )
    out = {k: st[k] for k in keys}
    out["by_hemisphere"] = st["by_hemisphere"]
    out["share_detections_in_long_storms"] = st["companions"]["share_detections_in_long_storms"]
    return out


class StormBook:
    """The storm layer's bookkeeping without its compaction and records: the sphere tracker over the ticks,
    the peak vorticity of every track so far and the step each track was first named (real time: the first
    detection whose peak so far reaches ``NAME_ZETA``, as ``StormLayer.ingest`` names a track). ``update``
    folds one frame in and returns the named and the active counts per hemisphere and the detections of the
    named storms active at the tick, ``(t, track, hemi, x, y, z, zeta_f0, r_eq, rain)``."""

    def __init__(self, sp: SphereParams) -> None:
        self.sp = sp
        self.tracker = SphereTracker(sp)
        self.peak: dict[int, float] = {}
        self.named_at: dict[int, int] = {}

    def update(
        self, fr: SphereFrame
    ) -> tuple[
        tuple[int, int], tuple[int, int], list[tuple[int, int, int, float, float, float, float, float, float]]
    ]:
        t = fr.t
        det = detect(fr, self.sp)
        ids = self.tracker.update(det)
        for tid, c in zip(ids, det.storms, strict=True):
            self.peak[tid] = max(self.peak.get(tid, 0.0), c.zeta_max_f0)
            if self.peak[tid] >= NAME_ZETA and tid not in self.named_at:
                self.named_at[tid] = t
        n_named, n_all = [0, 0], [0, 0]
        rows = []
        for tr in self.tracker.active():
            d = tr.dets[-1]
            if d.t != t:
                continue
            n_all[tr.hemi] += 1
            if tr.id in self.named_at:
                n_named[tr.hemi] += 1
                rows.append((t, tr.id, tr.hemi, d.x, d.y, d.z, d.zeta_max_f0, d.r_eq, d.rain))  # type: ignore[attr-defined]
        return (n_named[0], n_named[1]), (n_all[0], n_all[1]), rows


def run_study(cfg: Config, st: StudySettings, progress: Any = None) -> dict[str, Any]:
    """Run one study and return its record (JSON-ready) plus the arrays under the keys ``_series``, ``_dets``,
    ``_counts`` and ``_tracks`` (rows for the CSV and npz outputs), which ``write_outputs`` takes out."""
    wp = weather_sph.params_from_config(cfg)
    radius = cfg.world.radius
    sp = SphereParams.from_config(cfg)
    book = StormBook(sp)
    step_fn = chain_step(cfg)
    key = spinup_key(init_keys(st.seed)[InitStream.STATE_KEY])
    wx = weather_sph.rest_array(cfg, 0)
    ex = Exposure.of(sp.nside, radius)
    n_ticks = st.steps // st.tick
    rows_named: list[tuple[int, int]] = []
    rows_all: list[tuple[int, int]] = []
    times: list[int] = []
    dets: list[tuple[int, int, int, float, float, float, float, float, float]] = []
    series: list[dict[str, float]] = []
    t = 0
    t0 = time.time()
    for k in range(n_ticks):
        wx, key = advance_chain(cfg, wx, t, key, st.tick, step_fn=step_fn)
        t += st.tick
        ws = weather_sph.unpack(wx)
        zeta = np.array(weather_sph.vorticity(ws, wp, radius), dtype=np.float64)
        spd = np.array(mx.sqrt((ws.u1 * ws.u1).sum(axis=-1)), dtype=np.float64)
        fr = SphereFrame(t, zeta, np.array(ws.P, dtype=np.float64), np.array(ws.h1, dtype=np.float64), spd)
        n_named, n_all, rows = book.update(fr)
        dets += rows
        rows_named.append(n_named)
        rows_all.append(n_all)
        times.append(t)
        if k % st.series_every == 0:
            series.append({"t": float(t), **_climate(wx, wp)})
        if progress is not None and k % 200 == 0:
            progress(t, st.steps, time.time() - t0)
    mx.eval(wx)
    tracks = book.tracker.finish()
    peak, named_rt = book.peak, book.named_at
    times_a = np.asarray(times, dtype=np.int64)
    win = times_a >= st.spinup
    nn, na = np.asarray(rows_named, dtype=np.int64), np.asarray(rows_all, dtype=np.int64)
    counts = count_stats(nn[win], na[win])
    d5 = w0.storm_counts(tracks, times_a[win].tolist(), st.min_dets)
    counts["d5"] = {
        "in_range_both": d5["in_range_fraction"],
        **{
            h: {
                "mean": d5[h]["mean"],
                "min": d5[h]["min"],
                "max": d5[h]["max"],
                "in_range": d5[h]["in_range_fraction"],
            }
            for h in ("north", "south")
        },
    }
    qual = w0.qualifying(tracks, st.min_dets)
    named = [tr for tr in qual if peak.get(tr.id, 0.0) >= NAME_ZETA]
    life = {
        "named_hindsight": _lifetime(named, st.spinup, st.min_dets),
        "all_qualifying": _lifetime(qual, st.spinup, st.min_dets),
    }
    rt = _realtime_lifetime(named, named_rt, st.spinup)
    life["named_realtime"] = rt
    rec_tracks = []
    for tr in tracks:
        n_at = named_rt.get(tr.id)
        rec_tracks.append({
            "id": tr.id, "hemi": tr.hemi, "t_first": tr.t_first, "t_last": tr.t_last,
            "life": tr.t_last - tr.t_first, "dets": len(tr.dets), "ended": int(tr.ended),
            "end_reason": tr.end_reason, "parent": tr.parent, "peak_f0": round(peak.get(tr.id, 0.0), 4),
            "named_at": -1 if n_at is None else n_at, "arc": round(_arc_length(tr, radius), 2),
            "lat_mean": round(_mean_latitude(tr, radius), 2),
        })  # fmt: skip
    lens = np.array(
        [
            r["life"]
            for r in rec_tracks
            if r["named_at"] >= 0 and r["t_last"] >= st.spinup and r["dets"] >= st.min_dets
        ],
        float,
    )
    arcs = np.array(
        [
            r["arc"]
            for r in rec_tracks
            if r["named_at"] >= 0 and r["t_last"] >= st.spinup and r["dets"] >= st.min_dets
        ],
        float,
    )
    track_summary = {
        "tracks_total": len(tracks),
        "qualifying": len(qual),
        "named": len(named),
        "named_in_window": int(lens.size),
        "named_life_quantiles": [float(v) for v in np.percentile(lens, [10, 50, 90])] if lens.size else None,
        "named_life_max": float(lens.max()) if lens.size else 0.0,
        "named_arc_quantiles_planet_units": [float(v) for v in np.percentile(arcs, [10, 50, 90])]
        if arcs.size
        else None,
        "named_arc_max": float(arcs.max()) if arcs.size else 0.0,
        "split_born": sum(1 for tr in tracks if tr.parent),
        **_turnover(rec_tracks, st),
    }
    da = np.array(dets, dtype=np.float64).reshape(-1, 9)
    inwin = da[:, 0] >= st.spinup
    expo: dict[str, Any] = {}
    if st.exposure and inwin.any():
        cen = geom.unit(da[inwin][:, 3:6])
        expo = exposure_stats(ex, cen, da[inwin][:, 1].astype(np.int64), da[inwin][:, 6])
    wall = time.time() - t0
    return {
        "settings": dataclasses.asdict(st),
        "planet": {
            "radius": radius,
            "nw": sp.nside,
            "cell_side": ex.side,
            "dt": wp.dt,
            "cells": geom.npix(sp.nside),
        },
        "weather": {
            k: getattr(cfg.weather, k)
            for k in (
                "seed_rate",
                "seed_v",
                "seed_sigma",
                "q_pool",
                "rain_gain",
                "storm_p",
                "storm_u0",
                "storm_span",
                "tau_rad",
                "r1",
                "r2",
                "c_d",
                "gamma",
            )
        },
        "pools": [list(p) for p in wp.pools],
        "wall_s": round(wall, 1),
        "ticks_in_window": int(win.sum()),
        "counts": counts,
        "lifetime": life,
        "tracks": track_summary,
        "exposure": expo,
        "climate_window": _window_climate(series, st.spinup),
        "climate_cv": _window_cv(series, st.spinup),
        "spinup_steps": spinup_stats(series, np.column_stack([times_a, nn]), st.steps),
        "_series": series,
        "_dets": da,
        "_counts": np.column_stack([times_a, nn, na]),
        "_tracks": rec_tracks,
    }


def _realtime_lifetime(named: Sequence[Track], named_rt: dict[int, int], t_spin: int) -> dict[str, Any]:
    """Kaplan-Meier lifetime with the age counted from the real-time naming (the first detection whose peak
    so far reaches ``NAME_ZETA``) to the last detection."""
    q = [tr for tr in named if tr.t_last >= t_spin and tr.id in named_rt]
    if not q:
        return {"n": 0}
    dur = [tr.t_last - named_rt[tr.id] for tr in q]
    ev = [tr.ended for tr in q]
    ent = [max(0, t_spin - named_rt[tr.id]) for tr in q]
    km = w0.kaplan_meier(dur, ev, ent)
    s = km.summary()
    s["pass"] = bool(km.reached >= LIFETIME_MIN)
    return s


def plateau_time(
    t: np.ndarray, y: np.ndarray, *, tol: float = 0.05, ref_from: float, smooth: int = 5
) -> float | None:
    """The first time from which ``y`` stays within the relative ``tol`` of its late mean (the mean over
    ``t >= ref_from``) for good: the spin-up length of a quantity. ``y`` is smoothed by a moving average of
    ``smooth`` samples (centred, shortened at the ends) first. None when it never settles, or the late part
    is empty."""
    t, y = np.asarray(t, dtype=np.float64), np.asarray(y, dtype=np.float64)
    late = t >= ref_from
    if not late.any() or t.size < smooth:
        return None
    k = np.ones(smooth) / smooth
    ys = np.convolve(np.pad(y, (smooth // 2, (smooth - 1) // 2), mode="edge"), k, mode="valid")
    ref = float(ys[late].mean())
    if ref == 0.0:
        return None
    out = np.abs(ys - ref) > tol * abs(ref)
    if not out.any():
        return float(t[0])
    last = int(np.flatnonzero(out).max())
    return float(t[last + 1]) if last + 1 < t.size else None


def spinup_stats(series: list[dict[str, float]], counts: np.ndarray, steps: int) -> dict[str, Any]:
    """The spin-up length of the weather in engine steps. A quantity settles at the time from which its
    smoothed series stays within a tolerance of its late mean (the last 40 % of the run, ``plateau_time``):
    the kinetic energy within 15 % (a moving average of 10 samples, 5 000 steps: storms come and go), the
    mean humidity within 1 % (3 samples), the mean rain within 25 % (10 samples). The storms have formed when
    the number of named storms per hemisphere (block means of 20 ticks) first reaches 1 of its late mean
    (``_first_reach``). ``counts`` are the rows ``(t, n_north, n_south, ...)``. ``settled`` is the latest of
    the kinetic energy, the humidity and the two counts: the spin-up after which the weather is the weather
    of the study."""
    ref = 0.6 * steps
    t = np.array([r["t"] for r in series])

    def col(k: str) -> np.ndarray:
        return np.array([r[k] for r in series])

    out: dict[str, Any] = {
        "ke": plateau_time(t, col("ke"), tol=0.15, ref_from=ref, smooth=10),
        "q_mean": plateau_time(t, col("q_mean"), tol=0.01, ref_from=ref, smooth=3),
        "p_mean": plateau_time(t, col("p_mean"), tol=0.25, ref_from=ref, smooth=10),
    }
    block = 20
    n = (counts.shape[0] // block) * block
    if n:
        tb = counts[:n, 0].reshape(-1, block)[:, -1]
        for h, name in enumerate(("north", "south")):
            yb = counts[:n, 1 + h].reshape(-1, block).mean(axis=1)
            out[f"named_{name}"] = _first_reach(tb, yb, ref)
    parts = [out[k] for k in ("ke", "q_mean", "named_north", "named_south") if out.get(k) is not None]
    out["settled"] = max(parts) if parts else None
    return out


def _first_reach(t: np.ndarray, y: np.ndarray, ref_from: float, band: float = 1.0) -> float | None:
    """The first time a block mean ``y`` reaches ``band`` of its late mean (the mean over ``t >= ref_from``)
    from below: the storms have formed. (The count of a world whose storms come and go wanders around its mean
    for tens of thousands of steps, so "stays within" would only ever measure the length of the run.)"""
    late = t >= ref_from
    if not late.any():
        return None
    hit = np.flatnonzero(y >= y[late].mean() - band)
    return float(t[hit[0]]) if hit.size else None


def _turnover(rec_tracks: list[dict[str, Any]], st: StudySettings) -> dict[str, float]:
    """The life cycle of the named storms in the window: births (a qualifying track first named at or after
    the end of the spin-up), deaths (one that ended, by dissipation or merger, in the window) and deaths per
    10 000 steps (the box's storms die at 1.8 to 2.0 per 10 000 steps, STORMS_W0 §2.2; a frozen weather
    at 0)."""
    named = [r for r in rec_tracks if r["named_at"] >= 0 and r["dets"] >= st.min_dets]
    births = sum(1 for r in named if r["named_at"] >= st.spinup)
    deaths = sum(1 for r in named if r["ended"] and st.spinup <= r["t_last"] <= st.steps)
    span = max(st.steps - st.spinup, 1)
    return {
        "named_births_in_window": births,
        "named_deaths_in_window": deaths,
        "named_deaths_per_10k": 1e4 * deaths / span,
    }


def _window_cv(series: list[dict[str, float]], t_spin: int) -> dict[str, float]:
    """The coefficient of variation of the kinetic energy and of the largest wind over the window: a weather
    that does not evolve (frozen, or stuck in a permanent state) has none."""
    rows = [r for r in series if r["t"] >= t_spin]
    out = {}
    for k in ("ke", "umax1"):
        v = np.array([r[k] for r in rows], dtype=np.float64)
        out[k] = float(v.std() / v.mean()) if v.size > 1 and v.mean() > 0 else 0.0
    return out


def _window_climate(series: list[dict[str, float]], t_spin: int) -> dict[str, float]:
    rows = [r for r in series if r["t"] >= t_spin]
    if not rows:
        return {}
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != "t"}


# --- outputs ------------------------------------------------------------------------------------------------


def write_outputs(rec: dict[str, Any], out: Path) -> None:
    """``summary.json``, ``tracks.csv``, ``series.csv`` and ``dets.npz`` in ``out`` (created)."""
    out.mkdir(parents=True, exist_ok=True)
    series, dets, tracks = rec.pop("_series"), rec.pop("_dets"), rec.pop("_tracks")
    counts = rec.pop("_counts")
    (out / "summary.json").write_text(json.dumps(rec, indent=1, default=float) + "\n", encoding="utf-8")
    if tracks:
        with (out / "tracks.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(tracks[0]))
            w.writeheader()
            w.writerows(tracks)
    if series:
        with (out / "series.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(series[0]))
            w.writeheader()
            w.writerows(series)
    np.savez_compressed(out / "dets.npz", counts=counts, dets=dets, columns=np.array(
        ["t", "track", "hemi", "x", "y", "z", "zeta_f0", "r_eq", "rain"]))  # fmt: skip


def one_line(rec: dict[str, Any]) -> str:
    c = rec["counts"]["named"]
    lt = rec["lifetime"]["named_hindsight"]
    return (
        f"named N {c['north']['mean']:.2f} [{c['north']['min']}-{c['north']['max']}] "
        f"S {c['south']['mean']:.2f} [{c['south']['min']}-{c['south']['max']}] "
        f"in2-6 both {c['in_range_both']:.3f} | "
        f"D5 both {rec['counts']['d5']['in_range_both']:.3f} | life med {lt['median_text']} (n {lt['n']}) "
        f"max {rec['tracks']['named_life_max']:.0f} | {rec['wall_s']}s"
    )


# --- the parallel sweep -------------------------------------------------------------------------------------


def _cand_args(base: dict[str, Any], cand: dict[str, Any], seed: int, out: Path) -> list[str]:
    """The command line of one ``run`` for a candidate of a sweep spec."""
    merged = {**base, **{k: v for k, v in cand.items() if k not in ("name", "seeds")}}
    args = [sys.executable, "-m", "alife.analysis.w0_sphere", "run", "--preset", merged["preset"],
            "--steps", str(merged.get("steps", 100_000)), "--spinup", str(merged.get("spinup", 20_000)),
            "--tick", str(merged.get("tick", 25)), "--seed", str(seed), "--out", str(out)]  # fmt: skip
    for ov in merged.get("overlays", []):
        args += ["--overlay", ov]
    for s in merged.get("set", []):
        args += ["--set", s]
    if "pools" in merged:
        args += ["--pools", json.dumps(merged["pools"])]
    if not merged.get("exposure", True):
        args += ["--no-exposure"]
    return args


def sweep(spec: dict[str, Any], out: Path, jobs: int, mem_gb: float, per_proc_gb: float) -> list[Path]:
    """Run every (candidate, seed) of ``spec`` as a worker process, at most ``jobs`` at a time and at most
    ``mem_gb / per_proc_gb`` (the budget; a worker of this study holds well under a GB). Workers are polled by
    their own handles. Returns the output directories in the order of the spec."""
    base = spec.get("base", {})
    work: list[tuple[str, list[str], Path]] = []
    for cand in spec["candidates"]:
        for seed in cand.get("seeds", base.get("seeds", [0])):
            d = out / f"{cand['name']}_s{seed}"
            work.append((f"{cand['name']}_s{seed}", _cand_args(base, cand, seed, d), d))
    slots = max(1, min(jobs, int(mem_gb // per_proc_gb)))
    out.mkdir(parents=True, exist_ok=True)
    running: list[tuple[str, subprocess.Popen[bytes], Any]] = []
    queue = list(work)
    done: list[Path] = []
    t0 = time.time()
    while queue or running:
        while queue and len(running) < slots:
            name, args, d = queue.pop(0)
            if (d / "summary.json").exists():
                print(f"[{time.time() - t0:6.0f}s] skip {name}: done", flush=True)
                done.append(d)
                continue
            d.mkdir(parents=True, exist_ok=True)
            log = (d / "run.log").open("wb")
            proc = subprocess.Popen(
                args, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, "PYTHONUNBUFFERED": "1"}
            )
            running.append((name, proc, log))
            print(f"[{time.time() - t0:6.0f}s] start {name} pid {proc.pid}", flush=True)
        still = []
        for name, proc, log in running:
            try:
                rc = proc.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                still.append((name, proc, log))
                continue
            log.close()
            print(f"[{time.time() - t0:6.0f}s] end {name} rc {rc}", flush=True)
            if rc == 0:
                done.append(out / name)
        running = still
    return [d for _, _, d in work if (d / "summary.json").exists()]


# --- the command line ---------------------------------------------------------------------------------------


def _run_cmd(a: argparse.Namespace) -> int:
    pools = json.loads(a.pools) if a.pools else None
    cfg = build_config(a.preset, a.overlay, a.set, pools)
    st = StudySettings(steps=a.steps, spinup=a.spinup, seed=a.seed, tick=a.tick, exposure=not a.no_exposure)
    if st.steps % st.tick:
        raise SystemExit("--steps must be a multiple of --tick")
    rec = run_study(cfg, st, progress=lambda t, n, s: print(f"  step {t}/{n} {s:.0f}s", flush=True))
    rec["argv"] = sys.argv[1:]
    print(one_line(rec))
    write_outputs(rec, Path(a.out))
    return 0


def _table_cmd(a: argparse.Namespace) -> int:
    for d in a.dirs:
        p = Path(d) / "summary.json"
        if not p.exists():
            print(f"{d}: no summary")
            continue
        print(f"{Path(d).name:40s} {one_line(json.loads(p.read_text()))}")
    return 0


def _sweep_cmd(a: argparse.Namespace) -> int:
    spec = json.loads(Path(a.spec).read_text(encoding="utf-8"))
    dirs = sweep(spec, Path(a.out), a.jobs, a.mem_gb, a.proc_gb)
    for d in dirs:
        rec = json.loads((d / "summary.json").read_text())
        print(f"{d.name:40s} {one_line(rec)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="alife.analysis.w0_sphere", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="one study")
    r.add_argument("--preset", required=True)
    r.add_argument("--overlay", action="append", default=[])
    r.add_argument("--set", action="append", default=[], help="section.key=value (repeatable)")
    r.add_argument("--pools", help="JSON: the keywords of make_pools, or a flat list of pools")
    r.add_argument("--steps", type=int, default=100_000)
    r.add_argument("--spinup", type=int, default=20_000, help="the window of the statistics starts here")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--tick", type=int, default=25, help="steps between detections (the engine's is 25)")
    r.add_argument("--no-exposure", action="store_true")
    r.add_argument("--out", required=True)
    r.set_defaults(fn=_run_cmd)
    s = sub.add_parser("sweep", help="many studies in parallel worker processes")
    s.add_argument(
        "--spec", required=True, help="JSON {base: {...}, candidates: [{name, set, pools, seeds}]}"
    )
    s.add_argument("--out", required=True)
    s.add_argument("--jobs", type=int, default=6)
    s.add_argument("--mem-gb", type=float, default=30.0, help="the budget for all workers together")
    s.add_argument("--proc-gb", type=float, default=2.0, help="the memory assumed per worker")
    s.set_defaults(fn=_sweep_cmd)
    t = sub.add_parser("table", help="one line per finished study")
    t.add_argument("dirs", nargs="+")
    t.set_defaults(fn=_table_cmd)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    return int(a.fn(a))


if __name__ == "__main__":
    raise SystemExit(main())
