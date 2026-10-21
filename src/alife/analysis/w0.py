"""The W0 study driver (docs/STORMS.md §8, W0 spec §5 to §9): runs ``alife.weather`` offline on MLX.

``python -m alife.analysis.w0 --params FILE --steps N --out DIR [--tracers RUN --from STEP |
--uniform-tracers M]`` advances the weather model, detects and tracks storms on the host (numpy), moves
passive tracers on exactly the agents' wind (``weather.sample_wind``) and bins them as the engine's
neighbour search does, and writes the six exit criteria, the diagnostics, the health gate, the figures and
the movies to ``DIR``. ``--bench`` measures the cost, ``--vortex-decay`` the unforced-vortex table of spec
§3.1, and ``--impulse``, ``--pool-shift`` and ``--pool-gain`` the steerability deliverables of spec §9a.

This module is the offline MLX study driver of the prototype: it is never imported by the simulation step
and is not under the §22.6 lint scope, so it may read arrays to the host (``float(...)``, ``np.asarray``)
whenever it needs to. Nothing here writes outside ``--out`` (tool caches excepted), and a tracer source run
is only read.

Layout of the file: utilities, storm detection (``label_periodic``, ``detect``), the tracker, the statistics
(counts, Kaplan-Meier, forecast error with its jitter floor, drift, latitudes), the tracers and the occupancy
binning, the accumulators and the run loop (``Sim``), the study recorder, the experiments, the bench, the
outputs and the CLI.
"""

from __future__ import annotations

import argparse
import copy
import csv
import dataclasses
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

import mlx.core as mx
import numpy as np

from alife import ops, spatial, weather_obs
from alife import weather as W
from alife.config import Config, dims, load_config
from alife.metrics import N_BANDS, z_band
from alife.weather import WeatherForcing, WeatherParams, WeatherState

TWO_PI = 2.0 * math.pi
TRACER_LIFE = 1000  # steps; agent lifetimes are 700 to 950 (STORMS §2)
STEPS_PER_S = (14.4, 30.0)  # the two playback speeds of the wall-clock columns (spec §9a)
BAND_NAMES = ("hz0", "hz1_11", "hz12")  # the occupancy bands by hash layer: floor, mid-column, surface
BAND_LABELS = (
    "floor (hash layer 0)",
    "mid-column (hash layers 1 to G_z - 2)",
    "surface (hash layer G_z - 1)",
)
FORECAST_LEADS = (250, 500, 1000)
FORECAST_ORIGIN_EVERY = 250
SPEED_BINS = 2000  # the background-wind histograms: [0, SPEED_MAX_BIN] in SPEED_BINS bins
SPEED_MAX_BIN = 0.5
KM_NEVER = "> max observed age"
KM_MIN_RISK = 5  # a Kaplan-Meier curve is not read at or after an age where fewer subjects are at risk
SPLIT_NEAR_TIE = 0.9  # tracker rule 4: overlaps within 10% of a track's best count as tied (ZETA decides)


# --- small utilities ----------------------------------------------------------------------------------------


def wall(steps: float) -> dict[str, float]:
    """A time in steps with its wall-clock value in seconds and minutes at 14.4 and at 30 steps/s."""
    out: dict[str, float] = {"steps": float(steps)}
    for rate in STEPS_PER_S:
        tag = f"{rate:g}"
        out[f"s_at_{tag}"] = float(steps) / rate
        out[f"min_at_{tag}"] = float(steps) / rate / 60.0
    return out


def min_image_np(d: Any, length: float) -> Any:
    """Minimum-image displacement ``d - L*round(d/L)`` (numpy; half-to-even rounding, like
    ``ops.min_image``)."""
    return d - length * np.round(np.asarray(d) / length)


def _clean(obj: Any) -> Any:
    """``obj`` as strict JSON: numpy scalars and arrays to Python, NaN and infinities to ``None``."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating | float):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def dumps(obj: Any) -> str:
    return json.dumps(_clean(obj), indent=2, sort_keys=False, allow_nan=False) + "\n"


def _quantiles(
    a: Sequence[float] | np.ndarray, qs: Sequence[float] = (0.25, 0.5, 0.75, 0.9)
) -> dict[str, float]:
    arr = np.asarray(a, dtype=np.float64)
    if arr.size == 0:
        return {"n": 0, **{f"p{round(100 * q)}": math.nan for q in qs}}
    return {"n": int(arr.size), **{f"p{round(100 * q)}": float(np.quantile(arr, q)) for q in qs}}


def params_hash(p: WeatherParams) -> str:
    """sha256 of the canonical JSON of the full parameter set."""
    blob = json.dumps(p.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _git_rev() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


# --- storm detection (spec §6) ------------------------------------------------------------------------------

# The periodic labelling and the centre/corner helpers moved to ``alife.weather_obs`` in W1 (the engine's
# metrics rows use them); the names stay importable from here.
label_periodic = weather_obs.label_periodic
centre_winds = weather_obs.centre_winds
corners_to_centres = weather_obs.corners_to_centres
hemisphere_sign = weather_obs.hemisphere_sign


class Frame(NamedTuple):
    """A host snapshot of the weather state at a detection time: float32 ``(N, N)`` arrays, ``zeta`` at the
    corners (layer 1), the rest as in ``WeatherState`` plus the derived vorticity."""

    t: int
    zeta: np.ndarray
    P: np.ndarray
    h1: np.ndarray
    h2: np.ndarray
    u1: np.ndarray
    v1: np.ndarray
    u2: np.ndarray
    v2: np.ndarray
    Q: np.ndarray


def read_frame(ws: WeatherState, p: WeatherParams, t: int) -> Frame:
    """The nine fields a detection reads (spec §6), in one device sync."""
    z = W.vorticity(ws, p)
    arrs = (z, ws.P, ws.h1, ws.h2, ws.u1, ws.v1, ws.u2, ws.v2, ws.Q)
    mx.eval(*arrs)
    return Frame(t, *(np.array(a) for a in arrs))


@dataclass(frozen=True)
class Comp:
    """A labelled region of the cyclonic-vorticity mask: a storm candidate or a band."""

    label: int
    hemi: int  # 0 north (y < L/2), 1 south
    x: float  # circular centroid, [0, L)
    y: float
    area: int  # cells
    r_eq: float  # Δx √(area/π)
    r_max: float  # largest distance of a footprint cell from the centroid
    zeta_max_f0: float
    wind_max: float  # max layer-1 centre speed within 1.5 r_eq of the centroid
    h1_min: float  # min layer-1 thickness over the footprint
    dh1: float  # mean h1 over the annulus r_eq < d <= 2 r_eq, minus h1_min
    rain: float  # Σ P Δx² over the footprint
    n_rain: int  # footprint cells with P >= p_th
    rx: float  # mean resultant length per axis (1 = compact, 0 = spread over the period)
    ry: float
    ext_x: float  # occupied columns × Δx
    ext_y: float
    flat: np.ndarray  # flat indices of the footprint cells


@dataclass
class Detection:
    """The result of ``detect`` at one time: the storms, the bands and the label image."""

    t: int
    lab: np.ndarray
    storms: list[Comp]
    bands: list[Comp]
    n_regions: int
    n_weak: int  # regions that are neither storms nor bands (too small, or without rain)


def _window(
    cx: float, cy: float, reach: float, n: int, dx: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rows, columns (wrapped indices) and the distance of each cell centre from ``(cx, cy)`` for the square
    window of half-width ``reach``."""
    m = max(min(math.ceil(reach / dx) + 1, n // 2 - 1), 1)
    ic, jc = int(cx // dx), int(cy // dx)
    cols = np.arange(ic - m, ic + m + 1)
    rows = np.arange(jc - m, jc + m + 1)
    ddx = (cols + 0.5) * dx - cx
    ddy = (rows + 0.5) * dx - cy
    return rows % n, cols % n, np.hypot(ddx[None, :], ddy[:, None])


def detect(fr: Frame, p: WeatherParams) -> Detection:
    """Storm footprints, bands and per-detection quantities from a frame (spec §6).

    Mask ``s·ζ_c >= zeta_th`` (``ζ_c`` the corner average at the centres, ``s`` the hemisphere sign)
    labelled with ``label_periodic`` per hemisphere. A region with a circular mean resultant length below
    0.5 on an axis or an extent above ``L/2`` is a **band** (its centroid is undefined): excluded from
    storms. Otherwise a storm needs ``area >= a_min`` and ``r_min`` cells with ``P >= p_th``. Centroids are
    circular means with weight ``s·ζ - zeta_th``. ``wind_max`` is the maximum layer-1 centre speed within
    ``1.5 r_eq`` of the centroid (a balanced vortex peaks near the footprint edge, so the footprint alone
    would miss it); ``dh1`` uses the annulus ``r_eq < d <= 2 r_eq``."""
    n, dx, length = p.n, p.dx, p.L
    s = hemisphere_sign(p)
    zc = corners_to_centres(fr.zeta.astype(np.float64))
    sz = s * zc
    th = p.zeta_th
    lab, k = label_periodic(sz >= th, group=np.broadcast_to(s, (n, n)))
    if k == 0:
        return Detection(fr.t, lab, [], [], 0, 0)
    jj, ii = np.nonzero(lab)
    lb = lab[jj, ii]
    order = np.argsort(lb, kind="stable")
    jj, ii, lb = jj[order], ii[order], lb[order]
    starts = np.searchsorted(lb, np.arange(1, k + 2))
    head = starts[:-1]
    area = np.diff(starts)
    w = np.maximum(sz[jj, ii] - th, 0.0) + 1e-9 * th  # the tiny floor only guards an all-zero weight sum
    ax = TWO_PI * (ii + 0.5) * dx / length
    ay = TWO_PI * (jj + 0.5) * dx / length
    sw = np.add.reduceat(w, head)
    cxs, sxs = np.add.reduceat(w * np.cos(ax), head), np.add.reduceat(w * np.sin(ax), head)
    cys, sys_ = np.add.reduceat(w * np.cos(ay), head), np.add.reduceat(w * np.sin(ay), head)
    cx = (np.arctan2(sxs, cxs) * length / TWO_PI) % length
    cy = (np.arctan2(sys_, cys) * length / TWO_PI) % length
    rx, ry = np.hypot(cxs, sxs) / sw, np.hypot(cys, sys_) / sw
    cols = np.bincount(np.unique(lb * n + ii) // n, minlength=k + 1)[1:] * dx
    rows = np.bincount(np.unique(lb * n + jj) // n, minlength=k + 1)[1:] * dx
    zmax = np.maximum.reduceat(sz[jj, ii], head) / p.f0
    h1min = np.minimum.reduceat(fr.h1[jj, ii].astype(np.float64), head)
    prain = fr.P[jj, ii].astype(np.float64)
    n_rain = np.add.reduceat((prain >= p.p_th).astype(np.int64), head)
    rain = np.add.reduceat(prain, head) * dx * dx
    uc1, vc1 = centre_winds(fr.u1, fr.v1)
    spd1 = np.hypot(uc1, vc1)
    flat_all = jj * n + ii
    storms: list[Comp] = []
    bands: list[Comp] = []
    n_weak = 0
    for q in range(k):
        label = q + 1
        sl = slice(int(starts[q]), int(starts[q + 1]))
        band = bool(rx[q] < 0.5 or ry[q] < 0.5 or cols[q] > length / 2 or rows[q] > length / 2)
        storm = (not band) and int(area[q]) >= p.a_min and int(n_rain[q]) >= p.r_min
        if not band and not storm:
            n_weak += 1
            continue
        hemi = 0 if s[int(jj[starts[q]]), 0] > 0 else 1
        r_eq = dx * math.sqrt(int(area[q]) / math.pi)
        flat = flat_all[sl]
        if band:
            bands.append(
                Comp(label, hemi, float(cx[q]), float(cy[q]), int(area[q]), r_eq, 0.0, float(zmax[q]), 0.0,
                     float(h1min[q]), 0.0, float(rain[q]), int(n_rain[q]), float(rx[q]), float(ry[q]),
                     float(cols[q]), float(rows[q]), flat)
            )  # fmt: skip
            continue
        ddx = min_image_np((ii[sl] + 0.5) * dx - cx[q], length)
        ddy = min_image_np((jj[sl] + 0.5) * dx - cy[q], length)
        r_max = float(np.hypot(ddx, ddy).max())
        rows_w, cols_w, d = _window(float(cx[q]), float(cy[q]), 2.0 * r_eq, n, dx)
        win = np.ix_(rows_w, cols_w)
        wind_max = float(spd1[win][d <= 1.5 * r_eq].max(initial=0.0))
        ring = (d > r_eq) & (d <= 2.0 * r_eq)
        dh1 = float(fr.h1[win][ring].mean() - h1min[q]) if ring.any() else 0.0
        storms.append(
            Comp(label, hemi, float(cx[q]), float(cy[q]), int(area[q]), r_eq, r_max, float(zmax[q]), wind_max,
                 float(h1min[q]), dh1, float(rain[q]), int(n_rain[q]), float(rx[q]), float(ry[q]),
                 float(cols[q]), float(rows[q]), flat)
        )  # fmt: skip
    return Detection(fr.t, lab, storms, bands, k, n_weak)


# --- the tracker (spec §6) ----------------------------------------------------------------------------------


@dataclass
class Det:
    """One detection of a storm on a track (``ux, uy`` are unwrapped along the track)."""

    t: int
    x: float
    y: float
    ux: float
    uy: float
    r_eq: float
    area: int
    hemi: int
    zeta_max_f0: float
    wind_max: float
    h1_min: float
    dh1: float
    rain: float


@dataclass
class Track:
    """A storm over time. ``dets`` are real detections only (coasting times are the gaps between them).

    ``end_reason`` is ``""`` while the track is active, ``"dissipated"`` once it has missed more than
    ``miss_grace`` consecutive detections (it ends at its last detection), or ``"merged"`` (then
    ``merged_into`` names the track it joined). ``parent`` is the id of the track a split-off storm came from.
    """

    id: int
    hemi: int
    dets: list[Det]
    parent: int = 0
    merged_into: int = 0
    end_reason: str = ""
    missed: int = 0
    last_flat: np.ndarray | None = None

    @property
    def ended(self) -> bool:
        return bool(self.end_reason)

    @property
    def t_first(self) -> int:
        return self.dets[0].t

    @property
    def t_last(self) -> int:
        return self.dets[-1].t

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """``(t, ux, uy, r_eq)`` of the real detections."""
        d = self.dets
        return (
            np.array([q.t for q in d], dtype=np.float64),
            np.array([q.ux for q in d]),
            np.array([q.uy for q in d]),
            np.array([q.r_eq for q in d]),
        )


class Tracker:
    """Persistent storm ids across detections (spec §6).

    Order of one ``update``: (1) overlaps between each active track's last footprint and the new storm
    components, greedy by descending overlap, where an overlap within 10% of that track's best counts as tied
    with it and ties go to the component with the higher peak vorticity (a split keeps the id with the core,
    not with a larger but dying fragment), then centroid distance, lower track id, component;
    (3) an active track left without a component that overlaps components taken by other tracks has merged
    into the track of its best-overlap component and ends there; (4) a component left without a track that
    overlaps a track becomes a new track with ``parent`` = the track it overlaps most (a split); (2) the
    tracks and components that remain, unexplained by any overlap, are matched by the predicted position
    (last centroid plus the least-squares velocity of the last four detections times the elapsed time)
    inside a gate of ``max(r_track, r_det)`` in the same hemisphere, greedy by distance; (5) the other
    components are births; (6) the other tracks coast, and a track that has missed more than ``miss_grace``
    consecutive detections ends at its last detection as dissipated. The numbering follows the spec; the
    explained cases (3, 4) are settled before the position fallback (2) so that a merging or splitting storm
    is never matched by distance to an unrelated track."""

    def __init__(self, p: WeatherParams) -> None:
        self.p = p
        self.tracks: list[Track] = []
        self.next_id = 1

    def active(self) -> list[Track]:
        return [tr for tr in self.tracks if not tr.ended]

    def clone(self) -> Tracker:
        """An independent copy (a forked run continues the same ids)."""
        return copy.deepcopy(self)

    def _predict(self, tr: Track, t: int) -> tuple[float, float]:
        d = tr.dets[-4:]
        last = d[-1]
        if len(d) < 2:
            return last.ux, last.uy
        ts = np.array([q.t for q in d], dtype=np.float64)
        tc = ts - ts.mean()
        den = float((tc * tc).sum())
        bx = float((tc * np.array([q.ux for q in d])).sum()) / den
        by = float((tc * np.array([q.uy for q in d])).sum()) / den
        dt = t - last.t
        return last.ux + bx * dt, last.uy + by * dt

    def _append(self, tr: Track, t: int, c: Comp) -> None:
        last = tr.dets[-1] if tr.dets else None
        if last is None:
            ux, uy = c.x, c.y
        else:
            ux = last.ux + float(min_image_np(c.x - last.x, self.p.L))
            uy = last.uy + float(min_image_np(c.y - last.y, self.p.L))
        tr.dets.append(
            Det(
                t,
                c.x,
                c.y,
                ux,
                uy,
                c.r_eq,
                c.area,
                c.hemi,
                c.zeta_max_f0,
                c.wind_max,
                c.h1_min,
                c.dh1,
                c.rain,
            )
        )
        tr.last_flat = c.flat
        tr.missed = 0

    def _new(self, t: int, c: Comp, parent: int = 0) -> Track:
        tr = Track(self.next_id, c.hemi, [], parent=parent)
        self.next_id += 1
        self._append(tr, t, c)
        self.tracks.append(tr)
        return tr

    def update(self, det: Detection) -> list[int]:
        """Advance the tracks with a detection; returns the track id of each ``det.storms`` entry."""
        p, t, comps = self.p, det.t, det.storms
        lab_flat = det.lab.ravel()
        by_label = {c.label: i for i, c in enumerate(comps)}
        act = self.active()
        # (-effective overlap, -peak vorticity of the component, distance, track id, component index)
        pairs: list[tuple[int, float, float, int, int]] = []
        for tr in act:
            labs = lab_flat[tr.last_flat]
            labs = labs[labs > 0]
            if labs.size == 0:
                continue
            ul, cnt = np.unique(labs, return_counts=True)
            last = tr.dets[-1]
            rows = [(lbl, cn) for lbl, cn in zip(ul.tolist(), cnt.tolist(), strict=True) if lbl in by_label]
            if not rows:
                continue  # it overlaps only bands or weak regions: not storms
            best = max(cn for _, cn in rows)
            for lbl, cn in rows:
                ci = by_label[lbl]
                dist = float(
                    np.hypot(min_image_np(comps[ci].x - last.x, p.L), min_image_np(comps[ci].y - last.y, p.L))
                )
                # a piece within 10% of the track's best overlap counts as tied with it, and the stronger
                # piece (peak vorticity) wins the tie, so a dying polar fragment does not take the id
                eff = best if cn >= SPLIT_NEAR_TIE * best else cn
                pairs.append((-eff, -comps[ci].zeta_max_f0, dist, tr.id, ci))
        pairs.sort()
        track_of_comp: dict[int, int] = {}
        comp_of_track: dict[int, int] = {}
        for _, _, _, tid, ci in pairs:
            if tid in comp_of_track or ci in track_of_comp:
                continue
            comp_of_track[tid] = ci
            track_of_comp[ci] = tid
        by_id = {tr.id: tr for tr in self.tracks}

        # (3) tracks explained by a merger; (4) components explained by a split
        explained_tracks: set[int] = set()
        for _, _, _, tid, ci in pairs:
            if tid in comp_of_track or tid in explained_tracks:
                continue
            tr = by_id[tid]
            tr.end_reason, tr.merged_into = "merged", track_of_comp[ci]
            explained_tracks.add(tid)
        parent_of_comp: dict[int, int] = {}
        for _, _, _, tid, ci in pairs:
            if ci not in track_of_comp and ci not in parent_of_comp:
                parent_of_comp[ci] = tid

        # (2) the position fallback among what no overlap explains
        free_tracks = [tr for tr in act if tr.id not in comp_of_track and tr.id not in explained_tracks]
        free_comps = [i for i in range(len(comps)) if i not in track_of_comp and i not in parent_of_comp]
        cand: list[tuple[float, int, int]] = []
        for tr in free_tracks:
            px, py = self._predict(tr, t)
            last = tr.dets[-1]
            for ci in free_comps:
                c = comps[ci]
                if c.hemi != tr.hemi:
                    continue
                dist = float(np.hypot(min_image_np(c.x - px, p.L), min_image_np(c.y - py, p.L)))
                if dist <= max(last.r_eq, c.r_eq):
                    cand.append((dist, tr.id, ci))
        cand.sort()
        for _, tid, ci in cand:
            if tid in comp_of_track or ci in track_of_comp:
                continue
            comp_of_track[tid] = ci
            track_of_comp[ci] = tid

        ids = [0] * len(comps)
        for tid, ci in comp_of_track.items():
            self._append(by_id[tid], t, comps[ci])
            ids[ci] = tid
        for ci, c in enumerate(comps):  # (5) births, split-offs included
            if ci not in track_of_comp:
                ids[ci] = self._new(t, c, parent_of_comp.get(ci, 0)).id
        for tr in act:  # (6) coasting and dissipation
            if tr.id in comp_of_track or tr.id in explained_tracks:
                continue
            tr.missed += 1
            if tr.missed > p.miss_grace:
                tr.end_reason = "dissipated"
        return ids

    def finish(self) -> list[Track]:
        """The tracks, in id order. Tracks still active are alive (censored) at the end of the run."""
        return sorted(self.tracks, key=lambda tr: tr.id)


# --- statistics (spec §6) -----------------------------------------------------------------------------------


def qualifying(tracks: Iterable[Track], min_dets: int) -> list[Track]:
    """Tracks with at least ``min_dets`` real detections (the others are transients, excluded everywhere)."""
    return [tr for tr in tracks if len(tr.dets) >= min_dets]


def detection_times(t_from: int, t_to: int, every: int) -> list[int]:
    """The detection times ``every, 2 every, ...`` inside ``[t_from, t_to]``."""
    first = -(-max(t_from, 1) // every) * every
    return list(range(first, t_to + 1, every))


def storm_counts(
    tracks: Sequence[Track], times: Sequence[int], min_dets: int, lo: int = 2, hi: int = 6
) -> dict[str, Any]:
    """Qualifying tracks alive per hemisphere at each time in ``times`` (spec §6, D5).

    A track is alive from its first to its last detection: a coasting time inside a gap that ends in a
    re-detection counts, and the coasting tail of a dissipated track (after its last detection) never does.
    Criterion: ``lo <= n <= hi`` in both hemispheres at every time.
    """
    q = qualifying(tracks, min_dets)
    t = np.asarray(times, dtype=np.int64)
    n = np.zeros((2, t.size), dtype=np.int64)
    for tr in q:
        n[tr.hemi] += (t >= tr.t_first) & (t <= tr.t_last)
    if t.size == 0:
        return {
            "times": [],
            "n_north": [],
            "n_south": [],
            "all_in_range": False,
            "in_range_fraction": math.nan,
        }
    ok = (n >= lo) & (n <= hi)
    out: dict[str, Any] = {
        "times": t,
        "n_north": n[0],
        "n_south": n[1],
        "lo": lo,
        "hi": hi,
        "all_in_range": bool(ok.all()),
        "in_range_fraction": float(ok.all(axis=0).mean()),
    }
    for h, name in enumerate(("north", "south")):
        out[name] = {
            "min": int(n[h].min()),
            "max": int(n[h].max()),
            "mean": float(n[h].mean()),
            "in_range_fraction": float(ok[h].mean()),
            "histogram": {int(k): int(v) for k, v in zip(*np.unique(n[h], return_counts=True), strict=True)},
        }
    return out


@dataclass
class KaplanMeier:
    """A Kaplan-Meier curve with delayed entry, plus the subjects it was built from.

    ``ages`` are the event ages, ``survival``, ``n_risk`` and ``n_events`` the curve, its risk set and its
    event count at each of them. ``median`` is ``None`` when the curve never reaches 0.5. A curve is not to be
    read where fewer than ``min_risk`` subjects are at risk (``thin_age`` is the first such event age): a
    median at or after it is flagged ``median_reliable = False``.
    """

    ages: np.ndarray
    survival: np.ndarray
    n_risk: np.ndarray
    n_events: np.ndarray
    median: float | None  # None: the curve never reaches 0.5
    max_age: float
    dur: np.ndarray = field(default_factory=lambda: np.zeros(0))
    event: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    entry: np.ndarray = field(default_factory=lambda: np.zeros(0))
    min_risk: int = KM_MIN_RISK

    def median_text(self) -> str:
        return f"{self.median:g}" if self.median is not None else f"{KM_NEVER} ({self.max_age:g})"

    @property
    def thin_age(self) -> float | None:
        """The first event age with fewer than ``min_risk`` subjects at risk (``None``: never)."""
        idx = np.flatnonzero(self.n_risk < self.min_risk)
        return float(self.ages[idx[0]]) if idx.size else None

    @property
    def n_risk_at_median(self) -> int | None:
        if self.median is None:
            return None
        return int(self.n_risk[int(np.flatnonzero(self.ages == self.median)[0])])

    @property
    def median_reliable(self) -> bool:
        """False when the median is at or after the first age whose risk set is thinner than ``min_risk``."""
        thin = self.thin_age
        return self.median is None or thin is None or self.median < thin

    @property
    def reached(self) -> float:
        """The median, or the longest observed age when the curve stays above 0.5 (a lower bound)."""
        return self.median if self.median is not None else self.max_age

    def at_risk_at(self, age: float) -> int:
        """Subjects at risk at ``age``: entered before it and still observed at it."""
        return int(((self.entry < age) & (self.dur >= age)).sum())

    def survival_before(self, age: float) -> float:
        """The curve just before ``age`` (1 before the first event): ``median >= threshold`` is the same as
        ``survival_before(threshold) > 0.5``."""
        done = self.survival[self.ages < age]
        return float(done[-1]) if done.size else 1.0

    def summary(self) -> dict[str, Any]:
        """The numbers that say how far a median can be trusted (JSON-ready)."""
        return {
            "n": int(self.dur.size),
            "events": int(self.event.sum()),
            "censored": int(self.dur.size - self.event.sum()),
            "median": self.median,
            "median_text": self.median_text(),
            "reached": self.reached,
            "max_age": self.max_age,
            "n_risk_at_median": self.n_risk_at_median,
            "min_risk": self.min_risk,
            "thin_age": self.thin_age,
            "median_reliable": self.median_reliable,
        }


def kaplan_meier(
    duration: Sequence[float], event: Sequence[bool], entry: Sequence[float] | None = None
) -> KaplanMeier:
    """Kaplan-Meier survival with delayed entry (left truncation).

    A subject is at risk at age ``a`` when ``entry < a <= duration``; ``event`` False is a censoring at
    ``duration``. An event at the subject's own entry age (``entry == duration``) was never at risk and is not
    counted. The median is the first age with ``S <= 0.5``, ``None`` if the curve stays above it.
    """
    dur = np.asarray(duration, dtype=np.float64)
    ev = np.asarray(event, dtype=bool)
    ent = np.zeros_like(dur) if entry is None else np.asarray(entry, dtype=np.float64)
    ages, surv, risk, nev = [], [], [], []
    s = 1.0
    for tau in np.unique(dur[ev]):
        at_risk = int(((ent < tau) & (dur >= tau)).sum())
        d = int(((dur == tau) & ev & (ent < tau)).sum())
        if at_risk > 0:
            s *= 1.0 - d / at_risk
        ages.append(float(tau))
        surv.append(s)
        risk.append(at_risk)
        nev.append(d)
    median = next((a for a, v in zip(ages, surv, strict=True) if v <= 0.5), None)
    return KaplanMeier(
        np.array(ages),
        np.array(surv),
        np.array(risk),
        np.array(nev),
        median,
        float(dur.max()) if dur.size else 0.0,
        dur,
        ev,
        ent,
    )


def _km_of(q: Sequence[Track], t_spin: int | None) -> KaplanMeier:
    """The curve of the tracks ``q``: duration ``t_last - t_first``, event = ended, entry age
    ``max(0, t_spin - t_first)`` (``t_spin`` ``None``: no truncation, full histories)."""
    if not q:
        return kaplan_meier([], [], [])
    dur = [tr.t_last - tr.t_first for tr in q]
    ev = [tr.ended for tr in q]
    ent = [0 if t_spin is None else max(0, t_spin - tr.t_first) for tr in q]
    return kaplan_meier(dur, ev, ent)


def lifetime_stats(
    tracks: Sequence[Track], t_spin: int, min_dets: int, threshold: float = 3000.0
) -> dict[str, Any]:
    """Storm lifetimes over every qualifying track alive at any time in the window (spec §6).

    Duration ``t_last - t_first``; the event is a dissipation or a merger, and a track still active at the end
    is censored. Storms born in the spin-up enter at age ``t_spin - t_first`` (left truncation). The median
    passes when it is at least ``threshold``, or when the curve stays above 0.5 and the longest observed age
    reaches ``threshold``; that is decided by the curve before ``threshold`` alone.

    The median is a reading of storm lifetimes only where the risk set is not thin: ``n_risk_at_median``,
    ``thin_age`` (the first event age with fewer than ``KM_MIN_RISK`` at risk) and ``median_reliable`` say
    whether it is, and ``reading`` puts it in words. The criterion is judged on the median as written
    whatever the flag says; companions that do not depend on a few short fragments are reported beside it
    (``companions``): the share of the window's storm detections that belong to storms observed living at
    least ``threshold`` steps (and whether it exceeds one half, which is the criterion read on detections
    instead of storms), the curve over full histories (every qualifying track of the run, no
    truncation, the spin-up genesis included), the curve without split-born tracks, and the curve per
    hemisphere (``by_hemisphere``).
    """
    qual = qualifying(tracks, min_dets)
    q = [tr for tr in qual if tr.t_last >= t_spin]
    km = _km_of(q, t_spin)
    done = [tr.t_last - tr.t_first for tr in q if tr.ended]
    win_dets = [sum(1 for d in tr.dets if d.t >= t_spin) for tr in q]
    n_dets = sum(win_dets)
    long_dets = sum(n for tr, n in zip(q, win_dets, strict=True) if tr.t_last - tr.t_first >= threshold)
    split_born = [tr for tr in q if tr.parent]
    if km.median is None:
        reading = (
            f"median not reached: survival stays above 0.5 up to the longest observed age {km.max_age:g}"
        )
    elif km.median_reliable:
        reading = f"median {km.median:g} read with {km.n_risk_at_median} at risk"
    else:
        reading = (
            f"NOT READABLE: median {km.median:g} is at or after age {km.thin_age:g}, where fewer than "
            f"{km.min_risk} storms are at risk ({km.n_risk_at_median} at the median)"
        )
    companions = {
        "share_detections_in_long_storms": long_dets / n_dets if n_dets else math.nan,
        # the criterion read on detections instead of storms: the median over detections of the lifetime of
        # the storm they belong to is at least ``threshold`` exactly when more than half are in long storms
        "detection_weighted_pass": bool(n_dets and 2 * long_dets > n_dets),
        "detections": n_dets,
        "detections_in_long_storms": long_dets,
        "long_storm_steps": threshold,
        "full_histories": _km_of(qual, None).summary(),
        "without_split_born": _km_of([tr for tr in q if not tr.parent], t_spin).summary(),
        "split_born": {
            "n": len(split_born),
            "events": sum(1 for tr in split_born if tr.ended),
            "median_duration": float(np.median([tr.t_last - tr.t_first for tr in split_born]))
            if split_born
            else None,
        },
    }
    return {
        "km": km,
        "n": len(q),
        "events": int(sum(tr.ended for tr in q)),
        "censored": int(sum(not tr.ended for tr in q)),
        "median": km.median,
        "median_text": km.median_text(),
        "max_age": km.max_age,
        "n_risk_at_median": km.n_risk_at_median,
        "thin_age": km.thin_age,
        "median_reliable": km.median_reliable,
        "reading": reading,
        "n_risk_at_threshold": km.at_risk_at(threshold),
        "survival_before_threshold": km.survival_before(threshold),
        "naive_median_completed": float(np.median(done)) if done else None,
        "pass": bool(q) and km.reached >= threshold,
        "threshold": threshold,
        "companions": companions,
        "by_hemisphere": {
            name: _km_of([tr for tr in q if tr.hemi == h], t_spin).summary()
            for h, name in enumerate(("north", "south"))
        },
    }


def _fit(
    t: np.ndarray, x: np.ndarray, y: np.ndarray
) -> tuple[float, float, float, float, float, float, float]:
    """Least-squares lines ``x(t)`` and ``y(t)`` through ``(t, .)`` with ``t`` relative to the origin:
    ``(ax, bx, ay, by, rss, t_mean, s_tt)`` where ``a`` is the value at ``t = 0`` and ``b`` the slope."""
    tm = float(t.mean())
    tc = t - tm
    stt = float((tc * tc).sum())
    bx, by = float((tc * x).sum()) / stt, float((tc * y).sum()) / stt
    ax, ay = float(x.mean()) - bx * tm, float(y.mean()) - by * tm
    rss = float(((x - ax - bx * t) ** 2).sum() + ((y - ay - by * t) ** 2).sum())
    return ax, bx, ay, by, rss, tm, stt


def forecast_stats(
    tracks: Sequence[Track],
    p: WeatherParams,
    t_spin: int,
    t_end: int,
    min_dets: int,
    leads: Sequence[int] = FORECAST_LEADS,
    origin_every: int = FORECAST_ORIGIN_EVERY,
    detect_every: int | None = None,
) -> dict[str, Any]:
    """Track extrapolation against persistence at each lead, with the jitter floor (spec §6, D9).

    Origins are multiples of ``origin_every`` inside the window; a track qualifies at an origin ``t0`` when it
    is detected at ``t0`` (not coasting) and has at least ``slots - 2`` detections among the ``slots`` =
    ``origin_every / detect_every + 1`` detection times in ``[t0 - origin_every, t0]`` (9 of 11 by default).
    The forecast is the least-squares line through the unwrapped positions, evaluated at ``t0 + lead``, and is
    verified against the same track's detection at ``t0 + lead`` (dropped, with the cause counted, if there is
    none). Errors are in units of the track's ``r_eq`` at ``t0``. The jitter floor is the median 2D error that
    pure centroid noise of the fitted size would give: ``√(2 ln 2) σ̂ √(1/n + (lead - t̄)²/S_tt + 1)``. The
    floor assumes white centroid noise: a storm that meanders slowly (an anchored one) has correlated centroid
    errors, the fit residuals understate them, and the floor reads low for it (the start set: 0.10 to 0.13
    radii against errors of 0.58 to 0.81). Only the skill condition protects the D9 reading there.
    """
    every = detect_every or p.detect_every
    slots = origin_every // every + 1
    need = max(slots - 2, 3)
    rows: dict[int, list[dict[str, float]]] = {lead: [] for lead in leads}
    drops: dict[int, dict[str, int]] = {lead: {} for lead in leads}

    def drop(lead: int, why: str) -> None:
        drops[lead][why] = drops[lead].get(why, 0) + 1

    for tr in qualifying(tracks, min_dets):
        t, ux, uy, r = tr.arrays()
        index = {int(tt): i for i, tt in enumerate(t)}
        for t0 in range(origin_every * math.ceil(max(t_spin, 1) / origin_every), t_end + 1, origin_every):
            if t0 < tr.t_first or t0 > tr.t_last:
                continue
            i0 = index.get(t0)
            for lead in leads:
                if i0 is None:
                    drop(lead, "coasting at the origin")
                    continue
                sel = (t >= t0 - origin_every) & (t <= t0)
                if int(sel.sum()) < need:
                    drop(lead, "short history")
                    continue
                target = t0 + lead
                j = index.get(target)
                if j is None:
                    if target > t_end:
                        why = "beyond the end of the run"
                    elif tr.ended and target > tr.t_last:
                        why = "track merged" if tr.end_reason == "merged" else "track dissipated"
                    else:  # inside the run: a gap in the detections (an active track coasting at the target)
                        why = "coasting at the target"
                    drop(lead, why)
                    continue
                ax, bx, ay, by, rss, tm, stt = _fit(t[sel] - t0, ux[sel], uy[sel])
                nfit = int(sel.sum())
                fx, fy = ax + lead * bx, ay + lead * by
                err = float(np.hypot(min_image_np(ux[j] - fx, p.L), min_image_np(uy[j] - fy, p.L))) / r[i0]
                pers = (
                    float(np.hypot(min_image_np(ux[j] - ux[i0], p.L), min_image_np(uy[j] - uy[i0], p.L)))
                    / r[i0]
                )
                sigma = math.sqrt(rss / (2.0 * max(nfit - 2, 1)))
                floor = (
                    math.sqrt(2.0 * math.log(2.0))
                    * sigma
                    * math.sqrt(1.0 / nfit + (lead - tm) ** 2 / stt + 1.0)
                )
                rows[lead].append(
                    {
                        "id": tr.id,
                        "hemi": tr.hemi,
                        "t0": t0,
                        "err": err,
                        "persist": pers,
                        "floor": floor / r[i0],
                        "r_eq": float(r[i0]),
                    }
                )
    out: dict[str, Any] = {"leads": {}, "slots": slots, "need": need}
    for lead in leads:
        rs = rows[lead]
        err = np.array([x["err"] for x in rs])
        per = np.array([x["persist"] for x in rs])
        flo = np.array([x["floor"] for x in rs])
        med_e = float(np.median(err)) if err.size else math.nan
        med_p = float(np.median(per)) if per.size else math.nan
        hemi = np.array([x["hemi"] for x in rs])
        by_hemi: dict[str, Any] = {}
        for h, name in enumerate(("north", "south")):
            sel = hemi == h
            by_hemi[name] = {
                "n": int(sel.sum()),
                "median_error": float(np.median(err[sel])) if sel.any() else math.nan,
                "median_persistence": float(np.median(per[sel])) if sel.any() else math.nan,
            }
        out["leads"][lead] = {
            "n": int(err.size),
            "by_hemisphere": by_hemi,
            "median_error": med_e,
            "p25": float(np.quantile(err, 0.25)) if err.size else math.nan,
            "p75": float(np.quantile(err, 0.75)) if err.size else math.nan,
            "median_persistence": med_p,
            "median_floor": float(np.median(flo)) if flo.size else math.nan,
            "skill": (1.0 - med_e / med_p) if err.size and med_p > 0 else math.nan,
            "drift_radii_median": med_p,
            "drops": dict(sorted(drops[lead].items())),
            "rows": rs,
        }
    return out


def drift_stats(
    tracks: Sequence[Track], p: WeatherParams, t_lo: int, t_hi: int, min_dets: int, gap: int = 250
) -> dict[str, Any]:
    """Displacement between detections ``gap`` steps apart on a track, inside the window (spec §6).

    Magnitude (units and radii of the starting ``r_eq``), the zonal component (east positive) and the poleward
    component (toward ``L/4`` in the north, ``3L/4`` in the south), median and 90th percentile by hemisphere.
    """
    rec: list[tuple[int, float, float, float, float]] = []  # hemi, units, radii, zonal, poleward
    for tr in qualifying(tracks, min_dets):
        t, ux, uy, r = tr.arrays()
        index = {int(tt): i for i, tt in enumerate(t)}
        pole = p.L / 4 if tr.hemi == 0 else 3 * p.L / 4
        for i, tt in enumerate(t):
            j = index.get(int(tt) + gap)
            if j is None or tt < t_lo or tt + gap > t_hi:
                continue
            dx_, dy_ = ux[j] - ux[i], uy[j] - uy[i]
            sgn = 1.0 if (pole - (uy[i] % p.L)) >= 0 else -1.0
            rec.append((tr.hemi, math.hypot(dx_, dy_), math.hypot(dx_, dy_) / r[i], dx_, sgn * dy_))
    out: dict[str, Any] = {
        "gap": gap,
        "n": len(rec),
        "records": np.array(rec, dtype=np.float64).reshape(-1, 5),
    }
    for name, sel in (("all", None), ("north", 0), ("south", 1)):
        sub = [x for x in rec if sel is None or x[0] == sel]
        out[name] = {
            "n": len(sub),
            "units": _quantiles([x[1] for x in sub], (0.5, 0.9)),
            "radii": _quantiles([x[2] for x in sub], (0.5, 0.9)),
            "zonal": _quantiles([x[3] for x in sub], (0.5, 0.9)),
            "poleward": _quantiles([x[4] for x in sub], (0.5, 0.9)),
        }
    return out


def pole_distance(y: np.ndarray, hemi: np.ndarray, length: float) -> np.ndarray:
    """Distance of latitude ``y`` from the pole of its hemisphere (``L/4`` north, ``3L/4`` south)."""
    pole = np.where(np.asarray(hemi) == 0, length / 4, 3 * length / 4)
    return np.abs(np.asarray(y) - pole)


def latitude_stats(
    tracks: Sequence[Track], p: WeatherParams, t_lo: int, t_hi: int, min_dets: int
) -> dict[str, Any]:
    """Where the storms are (spec §6): a histogram of the centroid latitude of every qualifying detection in
    the window (8-unit bins), the pole distances, the genesis latitudes and the share within 16 units of a
    pole."""
    ys, hs = [], []
    gen_y, gen_h = [], []
    for tr in qualifying(tracks, min_dets):
        gen_y.append(tr.dets[0].y)
        gen_h.append(tr.hemi)
        for d in tr.dets:
            if t_lo <= d.t <= t_hi:
                ys.append(d.y)
                hs.append(d.hemi)
    ys_a, hs_a = np.array(ys), np.array(hs)
    edges = np.arange(0.0, p.L + 8.0, 8.0)
    out: dict[str, Any] = {
        "bin_edges": edges,
        "hist": np.histogram(ys_a, edges)[0] if ys_a.size else np.zeros(len(edges) - 1, dtype=int),
        "equators": [0.0, p.L / 2],
        "poles": [p.L / 4, 3 * p.L / 4],
        "n": int(ys_a.size),
        "genesis_y": np.array(gen_y),
        "genesis_hemi": np.array(gen_h),
    }
    for h, name in enumerate(("north", "south")):
        d = pole_distance(ys_a[hs_a == h], hs_a[hs_a == h], p.L) if ys_a.size else np.zeros(0)
        out[name] = {
            "n": int(d.size),
            "median_pole_distance": float(np.median(d)) if d.size else math.nan,
            "within_16_of_pole": float((d <= 16.0).mean()) if d.size else math.nan,
        }
    d_all = pole_distance(ys_a, hs_a, p.L) if ys_a.size else np.zeros(0)
    out["median_pole_distance"] = float(np.median(d_all)) if d_all.size else math.nan
    out["within_16_of_pole"] = float((d_all <= 16.0).mean()) if d_all.size else math.nan
    return out


# --- tracers and the occupancy binning (spec §5) ------------------------------------------------------------


def occupancy_counts(pos: mx.array, cfg: Config) -> mx.array:
    """Agents per hash cell, binned exactly as ``spatial._find_neighbors_3d`` does (spec §5): ``(G_z, G, G)``
    int32. The binning moved to ``spatial.hash_occupancy`` (W1), which also takes an ``alive`` mask."""
    return spatial.hash_occupancy(pos, None, cfg)


band_peaks = spatial.band_peaks  # moved to ``spatial`` in W1; the name stays importable from here


def decode_cell(cid: int, cfg: Config) -> tuple[int, int, int]:
    """``(cx, cy, cz)`` of a flat hash-cell id ``(cz·G + cy)·G + cx``."""
    g = dims(cfg).g_hash
    return cid % g, (cid // g) % g, cid // (g * g)


def baseline_occupancy(pos: np.ndarray, cfg: Config) -> dict[str, Any]:
    """Peak occupancy per band and agents per band / per ``z_band`` quarter of a set of positions."""
    d = dims(cfg)
    counts = occupancy_counts(mx.array(pos, dtype=mx.float32), cfg)
    mxs, cells = band_peaks(counts)
    c = np.array(counts)
    gz = c.shape[0]
    per_band = [int(c[0].sum()), int(c[1 : gz - 1].sum()) if gz > 2 else 0, int(c[gz - 1].sum())]
    quarters = np.bincount(z_band(pos[:, 2], cfg.world.height), minlength=N_BANDS)
    return {
        "peak": [int(v) for v in np.array(mxs)],
        "cell": [int(v) for v in np.array(cells)],
        "agents": per_band,
        "total": int(pos.shape[0]),
        "z_band_quarters": [int(v) for v in quarters],
        "capacity": cfg.sensing.cell_capacity,
        "g_hash": d.g_hash,
        "g_hash_z": d.g_hash_z,
    }


def resampled_peaks(
    draw: Callable[[np.random.Generator], np.ndarray], cfg: Config, draws: int = 20, seed: int = 0
) -> dict[str, Any]:
    """Peak occupancy per band of independent draws of the tracer positions: what tracers show with no wind at
    all (a tracer is reseeded from a uniformly drawn source row, so its steady-state position is an
    independent draw). The checkpoint's own peak is lower than this, because drawing rows with replacement
    adds Poisson noise to the hot cells. ``draw(rng)`` returns one ``(M, 3)`` set of positions."""
    rng = np.random.default_rng(seed)
    peaks = np.zeros((draws, 3), dtype=np.int64)
    for k in range(draws):
        mxs, _ = band_peaks(occupancy_counts(mx.array(draw(rng), dtype=mx.float32), cfg))
        peaks[k] = np.array(mxs)
    return {
        "draws": draws,
        "peak_median": [float(v) for v in np.median(peaks, axis=0)],
        "peak_max": [int(v) for v in peaks.max(axis=0)],
    }


def load_tracer_source(run: Path, step: int) -> tuple[Config, np.ndarray, np.ndarray]:
    """``(cfg, pos (M, 3), buoy_cap (M,))`` of the live rows of one checkpoint of a 3D run. Read only."""
    from alife.runio import RunDir, load_checkpoint

    rd = RunDir.open(Path(run))
    cfg = rd.config()
    s = load_checkpoint(rd.checkpoint_at(step), cfg)
    alive = np.array(s.alive).astype(bool)
    return cfg, np.array(s.pos)[alive].astype(np.float32), np.array(s.buoy_cap)[alive].astype(np.float32)


def tracer_move(
    ws: WeatherState, pos: mx.array, al: mx.array, p: WeatherParams, cfg: Config, floor_held: bool = False
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """The tracers' motion on the agents' wind: ``(pos', n_clip_h, n_clip_w, n_edge)``. Shared by the rig and
    the bench's sampler timing."""
    d = dims(cfg)
    fc = cfg.world.field_cell
    aw = W.sample_wind(ws, pos, p, field_cell=fc)
    z = pos[:, 2]
    fz = round(p.height / fc)
    edge = (z < fc) | (z >= (fz - 1) * fc)
    v = aw.u
    if floor_held:
        v = mx.concatenate([v[:, :2], mx.where((z < d.cell_hash_z)[:, None], 0.0, v[:, 2:])], axis=1)
    v = al[:, None] * v
    speed = mx.sqrt((v * v).sum(axis=-1))
    v = v * (cfg.physics.v_max / mx.maximum(speed, cfg.physics.v_max))[:, None]
    new = ops.wrap_xy_clamp_z(pos + v, cfg.world.size, d.z_lo, d.z_hi)
    return new, aw.n_clip_h, aw.n_clip_w, edge.astype(mx.int32).sum()


class TracerRig:
    """Passive tracers on exactly the agents' wind (spec §5).

    One compiled step per weather step, on the weather state at the start of the step:
    ``v = α_i·sample_wind(ws, pos).u`` (clipped as §4), ``v *= min(1, v_max/‖v‖)`` (integrate3's 3D clip),
    ``pos = wrap_xy_clamp_z(pos + v)``. Gravity, buoyancy, thrust, contacts and grip are ignored on purpose.
    ``floor_held`` zeroes ``w`` in hash layer 0 (the gravity-bound bracket of spec §5). A tracer ``i`` is
    reseeded at window steps ``t`` with ``(t + i) mod life == 0``, before it moves, from a uniformly drawn
    source row (``pos``, ``buoy_cap``) or, in the uniform option, from a fresh uniform draw; the key chain is
    independent of the weather's. After each step the three band maxima and their cells are kept, as a running
    peak and as a per-step history that the host reads in blocks (no per-step sync).
    """

    def __init__(
        self,
        p: WeatherParams,
        cfg: Config,
        src_pos: np.ndarray | None,
        src_cap: np.ndarray | None,
        *,
        m: int | None = None,
        key: int = 0,
        floor_held: bool = False,
        life: int = TRACER_LIFE,
        compile: bool = True,
    ) -> None:
        self.p, self.cfg, self.d = p, cfg, dims(cfg)
        self.uniform = src_pos is None
        self.floor_held, self.life = floor_held, life
        if self.uniform:
            assert m is not None and m > 0, "uniform tracers need a count"
            self.m = m
            self.src_pos = self.src_cap = None
        else:
            assert src_pos is not None and src_cap is not None
            self.m = int(src_pos.shape[0])
            self.src_pos, self.src_cap = mx.array(src_pos), mx.array(src_cap)
        self.key = mx.random.key((key + 1) * 7919 + 104729)  # independent of the weather key chain
        self.key, k0 = mx.random.split(self.key)
        if self.uniform:
            pos, cap = self._uniform(k0, self.m)
        else:
            pos, cap = self.src_pos, self.src_cap
        self.pos, self.alpha = pos, W.alpha(cap, p)
        self.peak = mx.zeros((3, 3), dtype=mx.int32)  # per band: value, cell, step
        self.clip_acc = mx.zeros((3,), dtype=mx.int32)  # n_clip_h, n_clip_w, n_edge since the last read
        self._hist: list[mx.array] = []
        self.hist_steps: list[int] = []
        self._fn = mx.compile(self._step) if compile else self._step
        self._peak_fn = mx.compile(self._peak_update) if compile else self._peak_update
        mx.eval(self.pos, self.alpha)

    def _uniform(self, key: mx.array, m: int) -> tuple[mx.array, mx.array]:
        d, length = self.d, self.p.L
        u = mx.random.uniform(shape=(m, 4), key=key)
        pos = mx.stack([u[:, 0] * length, u[:, 1] * length, d.z_lo + u[:, 2] * (d.z_hi - d.z_lo)], axis=-1)
        return pos, u[:, 3]

    def _step(
        self, ws: WeatherState, pos: mx.array, al: mx.array, t: mx.array, key: mx.array, *src: mx.array
    ):
        p, cfg = self.p, self.cfg
        key, k1 = mx.random.split(key)
        due = ((t + mx.arange(self.m, dtype=mx.int32)) % self.life) == 0
        if self.uniform:
            npos, ncap = self._uniform(k1, self.m)
        else:
            row = mx.random.randint(0, self.m, (self.m,), key=k1)
            npos, ncap = src[0][row], src[1][row]
        pos = mx.where(due[:, None], npos, pos)
        al = mx.where(due, W.alpha(ncap, p), al)
        pos, n_h, n_w, n_edge = tracer_move(ws, pos, al, p, cfg, self.floor_held)
        mxs, cells = band_peaks(occupancy_counts(pos, cfg))
        occ = mx.concatenate([mxs, cells])
        clip = mx.stack([n_h, n_w, n_edge]).astype(mx.int32)
        return pos, al, key, occ, clip

    @staticmethod
    def _peak_update(peak: mx.array, occ: mx.array, t: mx.array) -> mx.array:
        better = occ[:3] > peak[:, 0]
        tt = mx.broadcast_to(t.astype(mx.int32), (3,))
        return mx.stack(
            [
                mx.where(better, occ[:3], peak[:, 0]),
                mx.where(better, occ[3:], peak[:, 1]),
                mx.where(better, tt, peak[:, 2]),
            ],
            axis=1,
        )

    def step(self, ws: WeatherState, t: int) -> tuple[mx.array, ...]:
        """Queue one tracer step (lazy) on the state ``ws`` at the start of window step ``t``; returns the
        arrays the caller must evaluate."""
        tt = mx.array(t, dtype=mx.int32)
        src = () if self.uniform else (self.src_pos, self.src_cap)
        self.pos, self.alpha, self.key, occ, clip = self._fn(ws, self.pos, self.alpha, tt, self.key, *src)
        self.peak = self._peak_fn(self.peak, occ, tt)
        self.clip_acc = self.clip_acc + clip
        self._hist.append(occ)
        self.hist_steps.append(t)
        return self.pos, self.alpha, self.key, self.peak, self.clip_acc, occ

    def reset_peak(self) -> None:
        self.peak = mx.zeros((3, 3), dtype=mx.int32)

    def read(self) -> dict[str, Any]:
        """Block read at a detection: the per-step band maxima and cells since the last read (steps, (k, 6)),
        the clip counters since the last read (reset), the running peak and the ``z_band`` quarter counts."""
        occ = np.array(mx.stack(self._hist)) if self._hist else np.zeros((0, 6), dtype=np.int32)
        steps = np.array(self.hist_steps, dtype=np.int64)
        self._hist, self.hist_steps = [], []
        clip = np.array(self.clip_acc).astype(np.int64)
        self.clip_acc = mx.zeros((3,), dtype=mx.int32)
        zq = ops.scatter_add_i32(
            N_BANDS,
            mx.clip(
                mx.floor(self.pos[:, 2] * N_BANDS / self.cfg.world.height).astype(mx.int32), 0, N_BANDS - 1
            ),
            1,
        )
        mx.eval(zq, self.peak)
        return {"steps": steps, "occ": occ, "clip": clip, "peak": np.array(self.peak), "zq": np.array(zq)}

    def maps(self) -> np.ndarray:
        """Per hash column the largest count in each band, ``(3, G, G)`` (the tracer movies)."""
        c = occupancy_counts(self.pos, self.cfg)
        gz = c.shape[0]
        mid = c[1 : gz - 1].max(axis=0) if gz > 2 else c[0] * 0
        m = mx.stack([c[0], mid, c[gz - 1]])
        mx.eval(m)
        return np.array(m)


# --- device accumulators and the run loop -------------------------------------------------------------------

ACC_MAX = ("courant_max", "speed_max", "hyper_max", "eps_local_max")
ACC_CNT = ("n_limited", "n_clamp", "n_cond_limited", "n_relax_limited", "n_evap_limited", "n_safety")
ACC_SUM = ("clamp_mass", "q_clamp", "sum_E", "sum_C", "sum_Qseed", "seed_on")


class Acc(NamedTuple):
    """The step diagnostics folded on the device: running maxima (never reset), and counters and sums since
    the last read (the host adds them up in int64 and float64)."""

    mx_: mx.array  # (4,) f32
    cnt: mx.array  # (6,) i32
    sums: mx.array  # (6,) f32


def new_acc() -> Acc:
    return Acc(
        mx.zeros((len(ACC_MAX),), dtype=mx.float32),
        mx.zeros((len(ACC_CNT),), dtype=mx.int32),
        mx.zeros((len(ACC_SUM),), dtype=mx.float32),
    )


def fold_diag(acc: Acc, d: W.WeatherDiag) -> Acc:
    """Fold one step's diagnostics into the accumulators (all derived from inputs, none constant)."""
    return Acc(
        mx.maximum(acc.mx_, mx.stack([getattr(d, k) for k in ACC_MAX])),
        acc.cnt + mx.stack([getattr(d, k).astype(mx.int32) for k in ACC_CNT]),
        acc.sums + mx.stack([getattr(d, k).astype(mx.float32) for k in ACC_SUM]),
    )


_FUSED: dict[tuple[WeatherParams, bool, bool], Callable[..., Any]] = {}


def make_fused(p: WeatherParams, *, compile: bool = True, momentum: bool = False) -> Callable[..., Any]:
    """``fn(ws, fo, key, acc) -> (ws', key', acc')``: the key-chain split (``k_run, k_t = split(k_run)``), the
    weather step and the fold of its diagnostics in one compiled graph. Cached per ``(p, compile, momentum)``
    so a forked run reuses the trace (``momentum`` only documents the forcing structure of the caller)."""
    ck = (p, compile, momentum)
    if ck not in _FUSED:

        def fn(ws: WeatherState, fo: WeatherForcing, key: mx.array, acc: Acc):
            key, k_t = mx.random.split(key)
            new, d = W.step(ws, fo, k_t, p)
            return new, key, fold_diag(acc, d)

        _FUSED[ck] = mx.compile(fn) if compile else fn
    return _FUSED[ck]


@dataclass
class Snapshot:
    """A restart point of a run: state, key, tracker and the schedule-dependent forcing inputs."""

    t: int
    ws: WeatherState
    key: mx.array
    tracker: Tracker
    pools: tuple[tuple[float, ...], ...]
    frame: Frame | None = None
    detection: Detection | None = None
    ids: list[int] | None = None


class Sim:
    """A weather run: state, key chain, forcing schedule, accumulators, detection and tracking.

    ``advance(t_end)`` steps one at a time (one ``mx.eval`` per step on the state and the accumulators, host
    reads only at detections). Actions in ``schedule[t]`` run at the start of step ``t`` (pool shifts, gains,
    impulses). Every ``detect_every`` steps the frame is read, storms are detected and tracked, and each
    callback gets ``(sim, frame, detection, ids, interval)``. With ``momentum`` the forcing always carries
    momentum arrays (zeros when idle), so the compiled graph never changes (spec §9a).
    """

    def __init__(
        self,
        p: WeatherParams,
        key: int = 0,
        *,
        momentum: bool = False,
        rig: TracerRig | None = None,
        rig_floor_held: TracerRig | None = None,
        compile: bool = True,
        detect_every: int | None = None,
    ) -> None:
        self.p = p
        self.detect_every = detect_every or p.detect_every
        self.momentum = momentum
        self.rig = rig
        self.rig_floor_held = rig_floor_held
        self.pools = p.pools
        self.fo = W.default_forcing(p)
        if momentum:
            self._set_momentum(*([mx.zeros((p.n, p.n), dtype=mx.float32)] * 4))
        self.ws = W.init_state(p, self.fo)
        self.key = mx.random.key(key)
        self.acc = new_acc()
        self.t = 0
        self.tracker = Tracker(p)
        self.schedule: dict[int, list[Callable[[Sim], None]]] = {}
        self.callbacks: list[Callable[..., None]] = []
        self.fused = make_fused(p, compile=compile, momentum=momentum)
        self.compile = compile
        self._drift = any(row[5] != 0.0 or row[6] != 0.0 for row in p.pools)
        self._warm: Callable[[mx.array], mx.array] | None = None
        self.totals_cnt = np.zeros(len(ACC_CNT), dtype=np.int64)
        self.totals_sum = np.zeros(len(ACC_SUM), dtype=np.float64)
        self.max_run = np.zeros(len(ACC_MAX), dtype=np.float64)
        mx.eval(self.ws, self.fo.T_s)

    # --- schedule and forcing ---

    def at(self, t: int, action: Callable[[Sim], None]) -> None:
        self.schedule.setdefault(t, []).append(action)

    def _set_momentum(self, fu1: mx.array, fv1: mx.array, fu2: mx.array, fv2: mx.array) -> None:
        self.fo = self.fo._replace(fu1=fu1, fv1=fv1, fu2=fu2, fv2=fv2)

    def set_pools(self, pools: tuple[tuple[float, ...], ...]) -> None:
        """Replace the pool list and recompute ``T_s`` (a host-schedule branch, never on array values)."""
        self.pools = pools
        pe = dataclasses.replace(self.p, pools=pools)
        if self._drift:
            self._warm = (
                mx.compile(lambda t: W.warm_pools(pe, t)) if self.compile else lambda t: W.warm_pools(pe, t)
            )
        else:
            self.fo = self.fo._replace(T_s=W.warm_pools(pe))
            mx.eval(self.fo.T_s)

    def _forcing(self) -> WeatherForcing:
        if not self._drift:
            return self.fo
        if self._warm is None:
            self.set_pools(self.pools)
        assert self._warm is not None
        return self.fo._replace(T_s=self._warm(mx.array(self.t, dtype=mx.int32)))

    # --- stepping ---

    def advance(self, t_end: int) -> None:
        while self.t < t_end:
            for action in self.schedule.pop(self.t, ()):
                action(self)
            fo = self._forcing()
            old = self.ws
            extra: tuple[mx.array, ...] = ()
            for rig in (self.rig, self.rig_floor_held):
                if rig is not None:
                    extra += rig.step(old, self.t)
            self.ws, self.key, self.acc = self.fused(old, fo, self.key, self.acc)
            mx.eval(self.ws, self.key, *self.acc, *extra)
            self.t += 1
            if self.t % self.detect_every == 0:
                self._detect()

    def read_interval(self) -> dict[str, np.ndarray]:
        """Read and reset the per-interval counters and sums; the maxima are running."""
        a = self.acc
        mx.eval(*a)
        cnt, sums, mxv = (
            np.array(a.cnt).astype(np.int64),
            np.array(a.sums).astype(np.float64),
            np.array(a.mx_),
        )
        self.totals_cnt += cnt
        self.totals_sum += sums
        self.max_run = np.maximum(self.max_run, mxv.astype(np.float64))
        self.acc = Acc(a.mx_, mx.zeros_like(a.cnt), mx.zeros_like(a.sums))
        return {"cnt": cnt, "sums": sums, "max": self.max_run.copy()}

    def _detect(self) -> None:
        interval = self.read_interval()
        fr = read_frame(self.ws, self.p, self.t)
        det = detect(fr, self.p)
        ids = self.tracker.update(det)
        for cb in self.callbacks:
            cb(self, fr, det, ids, interval)

    # --- forks ---

    def snapshot(
        self, frame: Frame | None = None, det: Detection | None = None, ids: list[int] | None = None
    ) -> Snapshot:
        return Snapshot(self.t, self.ws, self.key, self.tracker.clone(), self.pools, frame, det, ids)

    @classmethod
    def from_snapshot(cls, snap: Snapshot, p: WeatherParams, **kw: Any) -> Sim:
        """A run that continues a snapshot (same state, key chain, tracker and pools)."""
        sim = cls(p, 0, **kw)
        sim.ws, sim.key, sim.t = snap.ws, snap.key, snap.t
        sim.tracker = snap.tracker.clone()
        if snap.pools != p.pools:
            sim.set_pools(snap.pools)
        return sim


# --- the study recorder -------------------------------------------------------------------------------------

ZETA_BINS = np.linspace(-1.5, 3.0, 91)  # s·ζ_c / f0
P_BINS = np.concatenate([[0.0], np.logspace(-8, -1, 36)])  # rain rate, with the zero bin first
WIND_BINS = 300  # |u_h| per layer in [0, u_max]


def hyperbolic_real(h1: np.ndarray, h2: np.ndarray, delta: np.ndarray, p: WeatherParams) -> np.ndarray:
    """The exact long-wave hyperbolicity test (spec §2.5 rule 17): True where the 4 × 4 flux Jacobian of the
    two-layer system along the direction of the relative wind has only real eigenvalues.

    State ``(h1, h2, q1, q2)`` with the normal velocities ``q1 - q2 = |u1 - u2|`` (a common velocity only
    shifts the eigenvalues), pressure ``M2 = g_top (h1 + h2)``, ``M1 = M2 + g' h1``.
    """
    h1, h2, delta = (np.atleast_1d(np.asarray(a, dtype=np.float64)) for a in (h1, h2, delta))
    gt, gp = p.g_top, p.g_prime
    a = np.zeros((h1.size, 4, 4))
    a[:, 0, 0] = a[:, 2, 2] = delta
    a[:, 0, 2], a[:, 1, 3] = h1, h2
    a[:, 2, 0], a[:, 2, 1] = gt + gp, gt
    a[:, 3, 0], a[:, 3, 1] = gt, gt
    return np.abs(np.linalg.eigvals(a).imag).max(axis=1) <= 1e-9


def grid_noise_index(zeta: np.ndarray) -> float:
    """``Σ(ζ - B(ζ))² / Σζ²`` with ``B`` the 3 × 3 periodic box mean: the grid-scale share of the
    vorticity."""
    z = zeta.astype(np.float64)
    box = sum(np.roll(z, (dj, di), (0, 1)) for dj in (-1, 0, 1) for di in (-1, 0, 1)) / 9.0
    den = float((z * z).sum())
    return float(((z - box) ** 2).sum() / den) if den > 0 else 0.0


class WindStats:
    """Per-layer statistics of the agent wind from ``weather.wind_layers`` (spec §6): mean ``|u_h|``, a
    histogram for the 99th percentile, the largest ``|w|`` and the share of samples whose 3D norm exceeds
    ``u_max`` (for W1 to decide whether ``u_max`` should bound the 3D norm). Mid-column layers are 1 .. fz -
    2."""

    def __init__(self, p: WeatherParams, fz: int, field_cell: float) -> None:
        self.p, self.fz, self.fc = p, fz, field_cell
        self.n = 0
        self.mean_sum = np.zeros(fz)
        self.hist = np.zeros((fz, WIND_BINS))
        self.w_max = np.zeros(fz)
        self.over3 = np.zeros(fz)

    def add(self, ws: WeatherState) -> None:
        lay = W.wind_layers(ws, self.p, self.fz, self.fc)
        uh = mx.sqrt(lay[:, 0] * lay[:, 0] + lay[:, 1] * lay[:, 1])
        w = lay[:, 2]
        norm = mx.sqrt(uh * uh + w * w)
        mean = uh.reshape(self.fz, -1).mean(axis=1)
        wmax = mx.abs(w).reshape(self.fz, -1).max(axis=1)
        over = (norm > self.p.u_max).astype(mx.float32).reshape(self.fz, -1).mean(axis=1)
        mx.eval(uh, mean, wmax, over)
        self.n += 1
        self.mean_sum += np.array(mean)
        self.w_max = np.maximum(self.w_max, np.array(wmax))
        self.over3 += np.array(over)
        idx = np.clip((np.array(uh) / self.p.u_max * WIND_BINS).astype(np.int64), 0, WIND_BINS - 1).reshape(
            self.fz, -1
        )
        for k in range(self.fz):
            self.hist[k] += np.bincount(idx[k], minlength=WIND_BINS)

    def summary(self) -> dict[str, Any]:
        if self.n == 0:
            return {"n": 0}
        cum = np.cumsum(self.hist, axis=1)
        p99 = [float((np.searchsorted(c / c[-1], 0.99) + 1) * self.p.u_max / WIND_BINS) for c in cum]
        pooled = np.cumsum(self.hist.sum(axis=0))
        mid = slice(1, self.fz - 1)
        return {
            "n": self.n,
            "layers": self.fz,
            "mean_uh": (self.mean_sum / self.n).tolist(),
            "p99_uh": p99,
            "p99_uh_all": float((np.searchsorted(pooled / pooled[-1], 0.99) + 1) * self.p.u_max / WIND_BINS),
            "max_abs_w": self.w_max.tolist(),
            "max_abs_w_mid": float(self.w_max[mid].max()) if self.fz > 2 else None,
            "share_norm3_over_umax": (self.over3 / self.n).tolist(),
            "share_norm3_over_umax_mid": float((self.over3[mid] / self.n).mean()) if self.fz > 2 else None,
        }


class ArmRecord:
    """The host side of one tracer arm: the per-step band maxima and cells (read in blocks at each
    detection), the clip totals over the window and the device's running peak. ``prefix`` names the arm's
    columns in the time series ("" for the plain arm, "fh_" for the floor-held arm)."""

    def __init__(self, rig: TracerRig, prefix: str) -> None:
        self.rig, self.prefix = rig, prefix
        self.occ_tau: list[np.ndarray] = []
        self.occ_val: list[np.ndarray] = []
        self.clip_total = np.zeros(3, dtype=np.int64)
        self.peak_device = np.zeros((3, 3), dtype=np.int64)

    def read(self, row: dict[str, Any], t: int, t_spin: int) -> dict[str, Any]:
        """Read the arm at detection ``t``: store the block and put the interval maxima in the row."""
        rd = self.rig.read()
        self.peak_device = rd["peak"].astype(np.int64)
        if rd["occ"].shape[0]:
            self.occ_tau.append(rd["steps"] + 1)
            self.occ_val.append(rd["occ"])
            blk = rd["occ"]
            for b, name in enumerate(BAND_NAMES):
                j = int(blk[:, b].argmax())
                row[f"{self.prefix}occ_{name}"] = int(blk[j, b])
                row[f"{self.prefix}occ_{name}_cell"] = int(blk[j, 3 + b])
        if t > t_spin:
            self.clip_total += rd["clip"]
        return rd

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """``(tau, occ)`` over every tracer step: the time after the step and the six band values."""
        if not self.occ_tau:
            return np.zeros(0, dtype=np.int64), np.zeros((0, 6), dtype=np.int32)
        return np.concatenate(self.occ_tau), np.concatenate(self.occ_val)


class Recorder:
    """The per-detection recording of a study (a ``Sim`` callback).

    Per detection: the time-series row (storms and bands, kinetic energy zonal and eddy per layer, rain, mean
    thickness, hyperbolicity, running maxima, grid-noise index, occupancy and tracer counts), the exact
    hyperbolicity test on the 16 centres with the largest ratio, the movie and snapshot frames, restart
    snapshots for the experiments, and, inside the window ``t >= t_spin``, the accumulators of the ζ and P
    histograms, the background wind outside storms, the zonal means and the kinetic-energy sums.
    """

    def __init__(
        self,
        p: WeatherParams,
        *,
        t_spin: int,
        frame_times: Iterable[int] = (),
        snap_times: Iterable[int] = (),
        rig: TracerRig | None = None,
        rig_floor_held: TracerRig | None = None,
        cfg: Config | None = None,
        wind_every: int = 250,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        self.p, self.t_spin, self.rig, self.cfg = p, t_spin, rig, cfg
        self.arms: dict[str, ArmRecord] = {}
        if rig is not None:
            self.arms["plain"] = ArmRecord(rig, "")
        if rig_floor_held is not None:
            self.arms["floor_held"] = ArmRecord(rig_floor_held, "fh_")
        self.frame_times, self.snap_times = set(frame_times), set(snap_times)
        self.wind_every = wind_every
        self.progress = progress
        self.rows: list[dict[str, Any]] = []
        self.frames: dict[int, dict[str, Any]] = {}
        self.snaps: dict[int, Snapshot] = {}
        n = p.n
        self.zeta_hist = np.zeros(len(ZETA_BINS) - 1)
        self.p_hist = np.zeros(len(P_BINS) - 1)
        self.bg_hist = np.zeros((2, SPEED_BINS))
        self.zonal_sum = np.zeros((2, n))
        self.ke_sum = np.zeros((2, 2))  # layer × (zonal, eddy)
        self.n_win = 0
        self.band_log: list[tuple[int, int, int, float]] = []  # t, hemi, area, y
        self.eig_checked = 0
        self.eig_complex = 0
        self.hyper_snap_max = 0.0
        self.mass0 = self.q0 = 0.0
        self.mass_dev = 0.0
        self.mass_last = self.q_last = 0.0
        self.winds: WindStats | None = None
        if cfg is not None:
            self.winds = WindStats(p, round(p.height / cfg.world.field_cell), cfg.world.field_cell)
        else:
            self.winds = WindStats(p, round(p.height / 4.0), 4.0)

    def start(self, sim: Sim) -> None:
        """Record the initial mass and moisture (the budgets close against them)."""
        self.mass0 = float(
            (np.array(sim.ws.h1, dtype=np.float64) + np.array(sim.ws.h2, dtype=np.float64)).sum()
        )
        self.q0 = float(np.array(sim.ws.Q, dtype=np.float64).sum())
        self.mass_last, self.q_last = self.mass0, self.q0

    def __call__(
        self, sim: Sim, fr: Frame, det: Detection, ids: list[int], interval: dict[str, np.ndarray]
    ) -> None:
        p, t = self.p, fr.t
        in_window = t >= self.t_spin
        uc1, vc1 = centre_winds(fr.u1, fr.v1)
        uc2, vc2 = centre_winds(fr.u2, fr.v2)
        spd = (np.hypot(uc1, vc1), np.hypot(uc2, vc2))
        zc = corners_to_centres(fr.zeta.astype(np.float64))
        s = hemisphere_sign(p)

        # domain kinetic energy per layer, unweighted: zonal-mean part and the rest
        ke = np.zeros((2, 2))
        ubar = []
        for k, (u, v) in enumerate(((fr.u1, fr.v1), (fr.u2, fr.v2))):
            ub = u.astype(np.float64).mean(axis=1)
            ubar.append(ub)
            ke[k, 0] = 0.5 * float((ub**2).mean())
            ke[k, 1] = 0.5 * float(((u - ub[:, None]) ** 2).mean() + (v.astype(np.float64) ** 2).mean())
        mass = float((fr.h1.astype(np.float64) + fr.h2.astype(np.float64)).sum())
        qsum = float(fr.Q.astype(np.float64).sum())
        self.mass_last, self.q_last = mass, qsum
        self.mass_dev = max(self.mass_dev, abs(mass - self.mass0) / self.mass0)

        # hyperbolicity: host snapshot ratio, and the exact test on the 16 largest
        d2 = (uc1 - uc2) ** 2 + (vc1 - vc2) ** 2
        hyp = d2 / (p.g_prime * (fr.h1.astype(np.float64) + fr.h2))
        top = np.argpartition(hyp.ravel(), -16)[-16:]
        real = hyperbolic_real(fr.h1.ravel()[top], fr.h2.ravel()[top], np.sqrt(d2.ravel()[top]), p)
        self.eig_checked += int(real.size)
        self.eig_complex += int((~real).sum())
        self.hyper_snap_max = max(self.hyper_snap_max, float(hyp.max()))
        gn = grid_noise_index(fr.zeta)

        nb = [sum(1 for c in det.storms if c.hemi == h) for h in (0, 1)]
        bb = [sum(1 for c in det.bands if c.hemi == h) for h in (0, 1)]
        for c in det.bands:
            self.band_log.append((t, c.hemi, c.area, c.y))
        row: dict[str, Any] = {
            "t": t,
            "n_storms_north": math.nan,
            "n_storms_south": math.nan,
            "detections_north": nb[0],
            "detections_south": nb[1],
            "bands_north": bb[0],
            "bands_south": bb[1],
            "weak_regions": det.n_weak,
            "ke_zonal_1": ke[0, 0],
            "ke_eddy_1": ke[0, 1],
            "ke_zonal_2": ke[1, 0],
            "ke_eddy_2": ke[1, 1],
            "rain": float(fr.P.astype(np.float64).mean()),
            "mean_h1": float(fr.h1.astype(np.float64).mean()),
            "min_h1": float(fr.h1.min()),
            "hyper_snapshot": float(hyp.max()),
            "courant_max": float(interval["max"][0]),
            "speed_max": float(interval["max"][1]),
            "hyper_run_max": float(interval["max"][2]),
            "eps_local_max": float(interval["max"][3]),
            "n_clamp": int(interval["cnt"][1]),
            "n_safety": int(interval["cnt"][5]),
            "grid_noise": gn,
        }
        for arm in self.arms.values():
            rd = arm.read(row, t, self.t_spin)
            if arm.prefix == "":
                for q in range(N_BANDS):
                    row[f"zq{q}"] = int(rd["zq"][q])
        self.rows.append(row)

        if in_window:
            self.n_win += 1
            self.zeta_hist += np.histogram(s * zc / p.f0, ZETA_BINS)[0]
            pv = fr.P.ravel()
            self.p_hist[0] += float((pv < P_BINS[1]).sum())
            self.p_hist[1:] += np.histogram(pv[pv >= P_BINS[1]], P_BINS[1:])[0]
            inside = np.zeros((p.n, p.n), dtype=bool)
            for c in det.storms:
                rows_w, cols_w, d = _window(c.x, c.y, 2.0 * (c.r_max + c.r_eq), p.n, p.dx)
                inside[np.ix_(rows_w, cols_w)] |= d <= c.r_max + c.r_eq
            for k in range(2):
                v = spd[k][~inside]
                self.bg_hist[k] += np.bincount(
                    np.clip((v / SPEED_MAX_BIN * SPEED_BINS).astype(np.int64), 0, SPEED_BINS - 1),
                    minlength=SPEED_BINS,
                )
                self.zonal_sum[k] += ubar[k]
            self.ke_sum += ke
            if self.winds is not None and (t - self.t_spin) % self.wind_every == 0:
                self.winds.add(sim.ws)

        if t in self.frame_times:
            self.frames[t] = {
                "h1": (fr.h1 - p.H1).astype(np.float32),
                "zeta": (zc / p.f0).astype(np.float32),
                "P": fr.P.astype(np.float32),
                "storms": [(i, c.x, c.y, c.r_eq) for i, c in zip(ids, det.storms, strict=True)],
                "maps": self.rig.maps() if self.rig is not None else None,
            }
        if t in self.snap_times:
            self.snaps[t] = sim.snapshot(fr, det, ids)
        if self.progress is not None and t % 1000 == 0:
            self.progress(
                f"t={t} storms N/S {nb[0]}/{nb[1]} bands {sum(bb)} hyper {row['hyper_snapshot']:.3f} "
                f"speed_max {row['speed_max']:.3f} min_h1 {row['min_h1']:.1f}"
            )

    # --- results used by the assembly ---

    def quantile_bg(self, k: int, q: float) -> float:
        c = np.cumsum(self.bg_hist[k])
        if c[-1] <= 0:
            return math.nan
        return float((np.searchsorted(c / c[-1], q) + 1) * SPEED_MAX_BIN / SPEED_BINS)


# --- organism lifetimes (the storm-to-organism ratio, spec §9a) ---------------------------------------------


def organism_lifetime(run: Path, step_from: int, span: int = 50_000) -> dict[str, Any]:
    """The age at death of the organisms of a run, from its deaths event files covering ``[from, from +
    span)`` (read only). The median is the organism lifetime the storm-to-organism ratio divides by."""
    ev = Path(run) / "events"
    ages = []
    files = []
    for f in sorted(ev.glob("deaths_*.npz")):
        a, b = (int(x) for x in f.stem.split("_")[1:3])
        if b > step_from and a < step_from + span:
            with np.load(f) as z:
                ages.append(np.asarray(z["age"], dtype=np.int64))
            files.append(f.name)
    if not ages:
        return {"source": "deaths event files", "run": str(run), "files": [], "n_deaths": 0}
    a = np.concatenate(ages)
    return {
        "source": "deaths event files (age at death, all causes)",
        "run": str(run),
        "files": files,
        "n_deaths": int(a.size),
        "median": float(np.median(a)),
        "mean": float(a.mean()),
        "p25": float(np.quantile(a, 0.25)),
        "p75": float(np.quantile(a, 0.75)),
    }


# --- vortex decay (spec §3.1, round 0) ----------------------------------------------------------------------


def balanced_vortex(p: WeatherParams, zeta0: float = 0.55, sigma: float = 9.0) -> WeatherState:
    """A balanced Gaussian-vorticity vortex on the north pole row: ζ at the corners (peak ``zeta0·f0``,
    radius ``sigma``), ψ from the discrete Poisson problem of ζ minus its mean, ``h1 = H1 + f0 ψ/g'``."""
    n, dx, length = p.n, p.dx, p.L
    xc = (np.arange(n) + 1.0) * dx
    r2 = (xc[None, :] - length / 2) ** 2 + (xc[:, None] - length / 4) ** 2
    zeta = zeta0 * p.f0 * np.exp(-r2 / (2.0 * sigma**2))
    zeta -= zeta.mean()
    k = TWO_PI * np.fft.fftfreq(n)
    lam = (2.0 * np.cos(k)[None, :] + 2.0 * np.cos(k)[:, None] - 4.0) / dx**2
    lam[0, 0] = 1.0
    ph = np.fft.fft2(zeta) / lam
    ph[0, 0] = 0.0
    psi = np.real(np.fft.ifft2(ph))
    u = -(psi - np.roll(psi, 1, 0)) / dx
    v = (psi - np.roll(psi, 1, 1)) / dx
    psi_c = 0.25 * (psi + np.roll(psi, 1, 0) + np.roll(psi, 1, 1) + np.roll(np.roll(psi, 1, 0), 1, 1))
    h1 = p.H1 + p.f0 * psi_c / p.g_prime
    fo = W.default_forcing(p)
    ws = W.init_state(p, fo)

    def f32(a: np.ndarray) -> mx.array:
        return mx.array(a, dtype=mx.float32)

    return ws._replace(h1=f32(h1), h2=f32(p.height - h1), u1=f32(u), v1=f32(v))


def vortex_decay_time(p: WeatherParams, limit: int, every: int = 25, threshold: float = 0.3) -> int | None:
    """The first multiple of ``every`` steps at which the centre-averaged peak ζ₁ of the unforced, dry vortex
    is below ``threshold·f0``; None if it lasts to ``limit``."""
    q = dataclasses.replace(p, gamma=0.0, k_E=0.0, tau_rad=0.0, seed_rate=0.0)
    ws = balanced_vortex(q)
    fo = W.default_forcing(q)
    fn = W.make_step(q)
    key = mx.random.key(0)
    for t in range(1, limit + 1):
        key, k = mx.random.split(key)
        ws, _ = fn(ws, fo, k)
        if t % every == 0:
            mx.eval(ws)
            z = corners_to_centres(np.array(W.vorticity(ws, q), dtype=np.float64))
            if z.max() / q.f0 < threshold:
                return t
    return None


def vortex_decay_table(p: WeatherParams, limit: int = 5000, every: int = 25) -> dict[str, Any]:
    """The unforced-vortex decay table (spec §3.1): default numerics, ``nu`` 0.02 with ``kappa_d`` 0, no
    physical damping, and cubic SL, each dry on ``p``'s grid, a balanced vortex of ζ 0.55 f0 and σ 9 on the
    north pole row, and the step at which ζ_max falls below 0.3 f0."""
    lines = {
        "default numerics": p,
        "nu 0.02, kappa_d 0 (the probe)": dataclasses.replace(p, nu=0.02, kappa_d=0.0),
        "no physical damping": dataclasses.replace(p, r1=0.0, c_d=0.0, r2=0.0),
        "cubic SL (kappa_d 0.10)": dataclasses.replace(p, sl_order=3, kappa_d=max(p.kappa_d, 0.10), nu=0.0),
    }
    out = {}
    for name, q in lines.items():
        t = vortex_decay_time(q, limit, every)
        out[name] = {
            "decay_step": t,
            "text": f"{t}" if t is not None else f"> {limit}",
            "sl_order": q.sl_order,
            "kappa_d": q.kappa_d,
            "nu": q.nu,
            "r1": q.r1,
            "c_d": q.c_d,
            "r2": q.r2,
        }
    return {"limit": limit, "every": every, "n": p.n, "zeta0_f0": 0.55, "sigma": 9.0, "lines": out}


# --- steerability experiments (spec §9a) --------------------------------------------------------------------


class LightLog:
    """A ``Sim`` callback for forked runs: the storms (id, x, y, r_eq, wind_max, hemisphere) per detection
    time."""

    def __init__(self) -> None:
        self.rows: dict[int, list[tuple[int, float, float, float, float, int]]] = {}

    def __call__(
        self, sim: Sim, fr: Frame, det: Detection, ids: list[int], interval: dict[str, np.ndarray]
    ) -> None:
        self.rows[fr.t] = [
            (i, c.x, c.y, c.r_eq, c.wind_max, c.hemi) for i, c in zip(ids, det.storms, strict=True)
        ]


@dataclass(frozen=True)
class ImpulseSpec:
    """``--impulse T0,X,Y,UX,UY,RADIUS,DURATION[,LAYERS]``: a Gaussian disc of momentum, ``(UX, UY)/DURATION``
    added per step to the winds of ``layers`` ("12" both, "1", "2") for ``DURATION`` steps from ``T0``."""

    t0: int
    x: float
    y: float
    ux: float
    uy: float
    radius: float
    duration: int
    layers: str = "12"

    @classmethod
    def parse(cls, text: str) -> ImpulseSpec:
        f = [s.strip() for s in text.split(",")]
        if len(f) not in (7, 8):
            raise ValueError("--impulse needs T0,X,Y,UX,UY,RADIUS,DURATION[,LAYERS]")
        layers = f[7] if len(f) == 8 else "12"
        if layers not in ("1", "2", "12"):
            raise ValueError("impulse LAYERS must be 1, 2 or 12")
        return cls(
            int(f[0]), float(f[1]), float(f[2]), float(f[3]), float(f[4]), float(f[5]), int(f[6]), layers
        )


def impulse_arrays(p: WeatherParams, s: ImpulseSpec) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """The per-step accelerations ``(fu1, fv1, fu2, fv2)`` of an impulse: ``(U/DURATION)·exp(-d²/2R²)`` at the
    u- and v-points, so the winds gain ``(UX, UY)`` at the centre over ``DURATION`` steps."""
    n, dx, length = p.n, p.dx, p.L
    xu = (np.arange(n) + 1.0) * dx  # u-points: x at east faces, y at centres
    yc = (np.arange(n) + 0.5) * dx
    xc = yc
    yv = xu

    def gauss(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        d2 = min_image_np(xs[None, :] - s.x, length) ** 2 + min_image_np(ys[:, None] - s.y, length) ** 2
        return np.exp(-d2 / (2.0 * s.radius**2))

    gu, gv = gauss(xu, yc), gauss(xc, yv)
    scale = 1.0 / (s.duration * p.dt)

    def f32(a: np.ndarray) -> mx.array:
        return mx.array(a, dtype=mx.float32)

    on1, on2 = "1" in s.layers, "2" in s.layers
    zero = np.zeros((n, n))
    return (
        f32(s.ux * scale * gu if on1 else zero),
        f32(s.uy * scale * gv if on1 else zero),
        f32(s.ux * scale * gu if on2 else zero),
        f32(s.uy * scale * gv if on2 else zero),
    )


@dataclass(frozen=True)
class PoolOp:
    """A timed change of one warm pool: ``--pool-shift T0,INDEX,DX,DY`` or ``--pool-gain T0,INDEX,FACTOR``."""

    t0: int
    index: int
    kind: str  # "shift" or "gain"
    dx: float = 0.0
    dy: float = 0.0
    factor: float = 1.0

    @classmethod
    def parse_shift(cls, text: str) -> PoolOp:
        f = [s.strip() for s in text.split(",")]
        if len(f) != 4:
            raise ValueError("--pool-shift needs T0,INDEX,DX,DY")
        return cls(int(f[0]), int(f[1]), "shift", dx=float(f[2]), dy=float(f[3]))

    @classmethod
    def parse_gain(cls, text: str) -> PoolOp:
        f = [s.strip() for s in text.split(",")]
        if len(f) != 3:
            raise ValueError("--pool-gain needs T0,INDEX,FACTOR")
        return cls(int(f[0]), int(f[1]), "gain", factor=float(f[2]))

    def apply(self, pools: tuple[tuple[float, ...], ...], length: float) -> tuple[tuple[float, ...], ...]:
        if not 0 <= self.index < len(pools):
            raise ValueError(f"pool index {self.index} out of range (the set has {len(pools)} pools)")
        x, y, sx, sy, amp, vx, vy = pools[self.index]
        if self.kind == "shift":
            x, y = (x + self.dx) % length, (y + self.dy) % length
        else:
            amp = amp * self.factor
        new = list(pools)
        new[self.index] = (x, y, sx, sy, amp, vx, vy)
        return tuple(new)


def pool_value(pool: tuple[float, ...], x: float, y: float, length: float) -> float:
    """``T_s`` of a single pool at a point (unclipped), the shape ``warm_pools`` sums."""
    x0, y0, sx, sy, amp, _, _ = pool
    e = -(float(min_image_np(y - y0, length)) ** 2) / (2.0 * sy * sy)
    if sx > 0:
        e -= float(min_image_np(x - x0, length)) ** 2 / (2.0 * sx * sx)
    return amp * math.exp(e)


def supersaturated_threshold(p: WeatherParams) -> float:
    """``T_s`` above which ``Q_e > Q_s``: ``(1 - q_bg)/(q_pool - q_bg)`` (the footprint of a pool)."""
    return (1.0 - p.q_bg) / (p.q_pool - p.q_bg) if p.q_pool > p.q_bg else math.inf


def det_at(tr: Track | None, t: int) -> Det | None:
    if tr is None:
        return None
    for d in tr.dets:
        if d.t == t:
            return d
    return None


def run_fork(
    snap: Snapshot,
    p: WeatherParams,
    horizon: int,
    *,
    impulse: ImpulseSpec | None = None,
    pool_ops: Sequence[PoolOp] = (),
    compile: bool = True,
) -> tuple[list[Track], LightLog]:
    """Continue a snapshot for ``horizon`` steps with the same key chain, optionally with an impulse and
    timed pool changes; the momentum arrays are always passed (zeros when idle), so the graph never
    changes."""
    sim = Sim.from_snapshot(snap, p, momentum=True, compile=compile)
    log = LightLog()
    sim.callbacks.append(log)
    if impulse is not None:
        arrays = impulse_arrays(p, impulse)
        zeros = tuple(mx.zeros((p.n, p.n), dtype=mx.float32) for _ in range(4))
        sim.at(impulse.t0, lambda s, a=arrays: s._set_momentum(*a))
        sim.at(impulse.t0 + impulse.duration, lambda s, z=zeros: s._set_momentum(*z))
    for op in pool_ops:
        sim.at(op.t0, lambda s, o=op: s.set_pools(o.apply(s.pools, p.L)))
    sim.advance(snap.t + horizon)
    return sim.tracker.finish(), log


def core_momentum(snap: Snapshot, p: WeatherParams, ci: int, layers: str) -> dict[str, float]:
    """The storm core's momentum scale at the snapshot: ``Σ_footprint Σ_layers h_k |u_k| Δx²``
    (centre-averaged winds), and the thickness-weighted area ``Σ_layers mean(h_k)`` that turns an impulse
    velocity into momentum."""
    fr, c = snap.frame, snap.detection.storms[ci]  # type: ignore[union-attr]
    assert fr is not None
    flat = c.flat
    total, hsum = 0.0, 0.0
    for k, (u, v, h) in enumerate(((fr.u1, fr.v1, fr.h1), (fr.u2, fr.v2, fr.h2)), start=1):
        if str(k) not in layers:
            continue
        uc, vc = centre_winds(u, v)
        total += float((h.ravel()[flat] * np.hypot(uc, vc).ravel()[flat]).sum()) * p.dx**2
        hsum += float(h.ravel()[flat].mean())
    return {"core_momentum": total, "thickness_sum": hsum, "r_eq": c.r_eq}


def _horizon_row(
    ctrl: Sequence[Track], pert: Sequence[Track], tid: int, t0: int, h: int, r_eq: float, p: WeatherParams
) -> dict[str, Any]:
    a = det_at(next((x for x in ctrl if x.id == tid), None), t0 + h)
    b = det_at(next((x for x in pert if x.id == tid), None), t0 + h)
    ta = next((x for x in ctrl if x.id == tid), None)
    tb = next((x for x in pert if x.id == tid), None)
    row: dict[str, Any] = {
        "horizon": h,
        "wall": wall(h),
        "survived_control": bool(ta is not None and ta.t_last >= t0 + h),
        "survived_perturbed": bool(tb is not None and tb.t_last >= t0 + h),
        "available": a is not None and b is not None,
    }
    if a is not None and b is not None:
        d = float(np.hypot(min_image_np(b.x - a.x, p.L), min_image_np(b.y - a.y, p.L)))
        row.update(displacement_units=d, displacement_radii=d / r_eq)
    return row


def impulse_experiment(
    snap: Snapshot,
    p: WeatherParams,
    specs: Sequence[tuple[float, ImpulseSpec]],
    *,
    horizon: int = 1000,
    ctrl: list[Track] | None = None,
    compile: bool = True,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Paired runs without and with each impulse, from one snapshot (spec §9a): the matched displacement of
    the storm nearest the impulse centre at +250, +500 and +1 000 steps, whether it survived, and the same
    for its two nearest neighbours as a control. ``specs`` are ``(factor, spec)`` pairs (``factor`` None
    when the caller gave the speed directly); the control runs once, or is passed in as ``ctrl`` (the tracks
    of a continuation from the same snapshot)."""
    assert snap.detection is not None and snap.ids is not None
    spec0 = specs[0][1]
    hist = {tr.id: len(tr.dets) for tr in snap.tracker.tracks}
    cand = [(i, c) for i, c in enumerate(snap.detection.storms) if hist.get(snap.ids[i], 0) >= p.min_dets]
    if not cand:
        return {"error": f"no qualifying storm detected at t0 = {snap.t}"}

    def dist(c: Comp) -> float:
        return float(np.hypot(min_image_np(c.x - spec0.x, p.L), min_image_np(c.y - spec0.y, p.L)))

    cand.sort(key=lambda ic: dist(ic[1]))
    ti, tc = cand[0]
    tid = snap.ids[ti]
    neighbours = sorted(
        (ic for ic in cand[1:]),
        key=lambda ic: float(np.hypot(min_image_np(ic[1].x - tc.x, p.L), min_image_np(ic[1].y - tc.y, p.L))),
    )[:2]
    if ctrl is None:
        if progress:
            progress(f"impulse: control continuation from t0 = {snap.t} ({horizon} steps)")
        ctrl, _ = run_fork(snap, p, horizon, compile=compile)
    runs = []
    for factor, spec in specs:
        if progress:
            progress(f"impulse: factor {factor} |U| = {math.hypot(spec.ux, spec.uy):.5f}")
        pert, _ = run_fork(snap, p, horizon, impulse=spec, compile=compile)
        cm = core_momentum(snap, p, ti, spec.layers)
        added = cm["thickness_sum"] * math.hypot(spec.ux, spec.uy) * TWO_PI * spec.radius**2
        runs.append(
            {
                "factor": factor,
                "momentum_ratio": added / cm["core_momentum"] if cm["core_momentum"] > 0 else None,
                "spec": dataclasses.asdict(spec),
                "speed": math.hypot(spec.ux, spec.uy),
                "target": {"id": tid, "x": tc.x, "y": tc.y, "r_eq": tc.r_eq},
                "horizons": [
                    _horizon_row(ctrl, pert, tid, snap.t, h, tc.r_eq, p)
                    for h in (250, 500, 1000)
                    if h <= horizon
                ],
                "neighbours": [
                    {
                        "id": snap.ids[i],
                        "distance": float(
                            np.hypot(min_image_np(c.x - tc.x, p.L), min_image_np(c.y - tc.y, p.L))
                        ),
                        "horizons": [
                            _horizon_row(ctrl, pert, snap.ids[i], snap.t, h, c.r_eq, p)
                            for h in (250, 500, 1000)
                            if h <= horizon
                        ],
                    }
                    for i, c in neighbours
                ],
            }
        )
    return {"t0": snap.t, "horizon": horizon, "target_id": tid, "runs": runs}


def auto_impulses(
    snap: Snapshot,
    p: WeatherParams,
    t0: int,
    x: float,
    y: float,
    direction_deg: float,
    duration: int = 50,
    factors: Sequence[float] = (0.5, 1.0, 2.0),
    layers: str = "12",
) -> list[tuple[float, ImpulseSpec]]:
    """The three impulses of spec §9a: ``RADIUS`` = the nearest storm's ``r_eq``, ``DURATION`` 50, and the
    speed ``|U|`` such that the disc's added momentum ``Σ_layers h̄_k·|U|·2πR²`` is ``factor`` × the
    core's."""
    assert snap.detection is not None and snap.ids is not None
    hist = {tr.id: len(tr.dets) for tr in snap.tracker.tracks}
    cand = [(i, c) for i, c in enumerate(snap.detection.storms) if hist.get(snap.ids[i], 0) >= p.min_dets]
    if not cand:
        raise ValueError(f"no qualifying storm detected at t0 = {t0}")
    ti, tc = min(
        cand, key=lambda ic: float(np.hypot(min_image_np(ic[1].x - x, p.L), min_image_np(ic[1].y - y, p.L)))
    )
    cm = core_momentum(snap, p, ti, layers)
    per_unit_speed = cm["thickness_sum"] * TWO_PI * tc.r_eq**2
    ang = math.radians(direction_deg)
    out = []
    for f in factors:
        speed = f * cm["core_momentum"] / per_unit_speed
        out.append(
            (
                f,
                ImpulseSpec(
                    t0, tc.x, tc.y, speed * math.cos(ang), speed * math.sin(ang), tc.r_eq, duration, layers
                ),
            )
        )
    return out


def _causal_stats(
    tracks: Sequence[Track], t: int, min_dets: int, hemi: int | None, window: int = 250
) -> dict[str, Any]:
    """Tracks detected at ``t`` with at least ``min_dets`` detections so far (a count that needs no future),
    and the median ``wind_max`` of their detections in ``[t - window, t]``."""
    alive = [
        tr for tr in tracks if det_at(tr, t) is not None and sum(1 for d in tr.dets if d.t <= t) >= min_dets
    ]
    if hemi is not None:
        alive = [tr for tr in alive if tr.hemi == hemi]
    winds = [d.wind_max for tr in alive for d in tr.dets if t - window <= d.t <= t]
    return {"count": len(alive), "core_wind": float(np.median(winds)) if winds else math.nan}


def pool_experiment(
    snap: Snapshot,
    p: WeatherParams,
    op: PoolOp,
    *,
    horizon: int | None = None,
    ctrl: list[Track] | None = None,
    after: Sequence[int] = (2000, 5000),
    compile: bool = True,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """One timed warm-pool change against a control continuation from the same snapshot (spec §9a).

    A shift reports the **genesis lag** (steps from ``t0`` to the first detection of a storm born after
    ``t0`` whose centroid lies inside the shifted pool's supersaturated footprint, ``T_s > (1 -
    q_bg)/(q_pool - q_bg)``, and the **first-detection lag** of any storm there), and the **displacement
    lag**: steps until the storm anchored to the old pool (the nearest at ``t0``) has moved half of the
    shift along it, ``None`` ("never") within the horizon. A gain reports the change of storm count and core
    wind against the control after 2 000 and 5 000 steps. Counts are causal (tracks with at least
    ``min_dets`` detections so far)."""
    assert snap.detection is not None and snap.ids is not None
    horizon = horizon or (10_000 if op.kind == "shift" else 5_000)
    if progress:
        progress(f"pool {op.kind} at t0 = {op.t0}: perturbed continuation ({horizon} steps)")
    if ctrl is None:
        ctrl, _ = run_fork(snap, p, horizon, compile=compile)
    pert, _ = run_fork(snap, p, horizon, pool_ops=[op], compile=compile)
    old = p.pools[op.index]
    new = op.apply(snap.pools, p.L)[op.index]
    hemi = 0 if new[1] < p.L / 2 else 1
    thr = supersaturated_threshold(p)
    out: dict[str, Any] = {
        "op": dataclasses.asdict(op),
        "t0": snap.t,
        "horizon": horizon,
        "pool_before": list(snap.pools[op.index]),
        "pool_after": list(new),
        "hemisphere": "north" if hemi == 0 else "south",
    }
    if op.kind == "shift":
        inside = lambda d: pool_value(new, d.x, d.y, p.L) > thr  # noqa: E731
        born = [(d.t, tr.id) for tr in pert if tr.t_first > snap.t for d in tr.dets if inside(d)]
        anyone = [(d.t, tr.id) for tr in pert for d in tr.dets if d.t > snap.t and inside(d)]
        gen = min(born) if born else None
        first = min(anyone) if anyone else None
        hist = {tr.id: len(tr.dets) for tr in snap.tracker.tracks}
        cand = [
            (i, c)
            for i, c in enumerate(snap.detection.storms)
            if hist.get(snap.ids[i], 0) >= p.min_dets and c.hemi == hemi
        ]
        anchored = None
        if cand:
            ai, ac = min(
                cand,
                key=lambda ic: float(
                    np.hypot(min_image_np(ic[1].x - old[0], p.L), min_image_np(ic[1].y - old[1], p.L))
                ),
            )
            tr = next((x for x in pert if x.id == snap.ids[ai]), None)
            shift = np.array([min_image_np(new[0] - old[0], p.L), min_image_np(new[1] - old[1], p.L)])
            s2 = float(shift @ shift)
            lag = None
            moved = []
            if tr is not None and s2 > 0:
                for d in tr.dets:
                    if d.t <= snap.t:
                        continue
                    disp = np.array([min_image_np(d.x - ac.x, p.L), min_image_np(d.y - ac.y, p.L)])
                    frac = float(disp @ shift) / s2
                    moved.append((d.t, frac))
                    if lag is None and frac >= 0.5:
                        lag = d.t - snap.t
            anchored = {
                "id": snap.ids[ai],
                "distance_to_old_pool": float(
                    np.hypot(min_image_np(ac.x - old[0], p.L), min_image_np(ac.y - old[1], p.L))
                ),
                "displacement_lag": lag,
                "displacement_lag_text": f"{lag}" if lag is not None else f"never within {horizon}",
                "displacement_lag_wall": wall(lag) if lag is not None else None,
                "end": None if tr is None else {"t_last": tr.t_last, "reason": tr.end_reason or "alive"},
                "max_fraction_of_shift": max((m for _, m in moved), default=None),
            }
        out.update(
            shift_units=float(np.hypot(op.dx, op.dy)),
            genesis_lag=None if gen is None else gen[0] - snap.t,
            genesis_track=None if gen is None else gen[1],
            genesis_lag_wall=None if gen is None else wall(gen[0] - snap.t),
            first_detection_lag=None if first is None else first[0] - snap.t,
            first_detection_track=None if first is None else first[1],
            anchored=anchored,
        )
    else:
        rows = []
        for dt in after:
            if dt > horizon:
                continue
            t = snap.t + dt
            cp, pp = _causal_stats(ctrl, t, p.min_dets, None), _causal_stats(pert, t, p.min_dets, None)
            ch, ph = _causal_stats(ctrl, t, p.min_dets, hemi), _causal_stats(pert, t, p.min_dets, hemi)
            rows.append(
                {
                    "after": dt,
                    "wall": wall(dt),
                    "count_control": cp["count"],
                    "count_perturbed": pp["count"],
                    "count_change": pp["count"] - cp["count"],
                    "count_change_hemisphere": ph["count"] - ch["count"],
                    "core_wind_control": ch["core_wind"],
                    "core_wind_perturbed": ph["core_wind"],
                    "core_wind_change": ph["core_wind"] - ch["core_wind"],
                }
            )
        out["gain"] = {"factor": op.factor, "rows": rows}
    return out


# --- the cost measurement (spec §7) -------------------------------------------------------------------------


def run_bench(
    p: WeatherParams,
    cfg: Config,
    *,
    m: int = 65_536,
    steps: int = 2000,
    warmup: int = 200,
    debug: bool = False,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Cost of the weather layer (spec §7), compiled, ``debug`` off by default.

    (A) 2 000 steps synced (``mx.eval(ws, diag)`` after each, perf_counter per step: median, p90, mean);
    (B) 2 000 steps chained, the step's diagnostics folded into device accumulators and ``mx.eval`` every 10
    steps; (C) the agent-wind sampler (``sample_wind`` with its fields, ``alpha``, the clips, and the position
    update) at ``m`` positions, chained and evaluated every 10 steps on a fixed weather state, and synced.
    The criterion is (B) + (C) <= 1 ms. Both (A) and (B) evaluate the diagnostics (MLX computes only what the
    evaluated arrays need); ``bench["evaluated"]`` records what each timing evaluated.
    """
    say = progress or (lambda s: None)
    fo = W.default_forcing(p)
    ws = W.init_state(p, fo)
    key = mx.random.key(0)
    fn = W.make_step(p, compile=True, debug=debug)
    ks = mx.random.split(key, warmup + 2 * steps + 8)
    say(f"bench: warm-up ({warmup} steps, compile included)")
    t0 = time.perf_counter()
    for i in range(warmup):
        ws, d = fn(ws, fo, ks[i])
        mx.eval(ws, d)
    warm_s = time.perf_counter() - t0
    mx.reset_peak_memory()

    say(f"bench: (A) synced, {steps} steps")
    times = []
    for i in range(steps):
        t1 = time.perf_counter()
        ws, d = fn(ws, fo, ks[warmup + i])
        mx.eval(ws, d)
        times.append(time.perf_counter() - t1)
    a_ms = np.array(times) * 1e3

    say(f"bench: (B) chained, {steps} steps")
    fused = make_fused(p, compile=True)
    acc = new_acc()
    k = mx.random.key(1)
    ws2, k, acc = fused(ws, fo, k, acc)
    mx.eval(ws2, k, *acc)
    t1 = time.perf_counter()
    for i in range(steps):
        ws2, k, acc = fused(ws2, fo, k, acc)
        if (i + 1) % 10 == 0:
            mx.eval(ws2, k, *acc)
    mx.eval(ws2, k, *acc)
    b_ms = (time.perf_counter() - t1) * 1e3 / steps
    peak_step = mx.get_peak_memory()

    say(f"bench: (C) sampler at M = {m}")
    rng_pos = mx.random.uniform(shape=(m, 3), key=mx.random.key(2))
    d_ = dims(cfg)
    pos = mx.stack(
        [rng_pos[:, 0] * p.L, rng_pos[:, 1] * p.L, d_.z_lo + rng_pos[:, 2] * (d_.z_hi - d_.z_lo)], axis=-1
    )
    caps = mx.random.uniform(shape=(m,), key=mx.random.key(3))

    def sampler(w: WeatherState, ps: mx.array, cap: mx.array, cnt: mx.array):
        new, nh, nw, ne = tracer_move(w, ps, W.alpha(cap, p), p, cfg)  # alpha is part of the timed graph
        return new, cnt + mx.stack([nh, nw, ne]).astype(mx.int32)

    sfn = mx.compile(sampler)
    cnt = mx.zeros((3,), dtype=mx.int32)
    pos, cnt = sfn(ws2, pos, caps, cnt)
    mx.eval(pos, cnt)
    t1 = time.perf_counter()
    for i in range(steps):
        pos, cnt = sfn(ws2, pos, caps, cnt)
        if (i + 1) % 10 == 0:
            mx.eval(pos, cnt)
    mx.eval(pos, cnt)
    c_ms = (time.perf_counter() - t1) * 1e3 / steps
    cs = []
    for _ in range(min(steps, 500)):
        t1 = time.perf_counter()
        pos, cnt = sfn(ws2, pos, caps, cnt)
        mx.eval(pos, cnt)
        cs.append(time.perf_counter() - t1)
    c_sync = np.array(cs) * 1e3
    peak = mx.get_peak_memory()
    total = b_ms + c_ms
    return {
        "device": str(mx.default_device()),
        "debug": debug,
        "n": p.n,
        "m_positions": m,
        "steps": steps,
        "warmup_steps": warmup,
        "warmup_wall_s": warm_s,
        "A_synced_ms": {
            "median": float(np.median(a_ms)),
            "p90": float(np.quantile(a_ms, 0.9)),
            "mean": float(a_ms.mean()),
        },
        "B_chained_ms": b_ms,
        "C_sampler_chained_ms": c_ms,
        "C_sampler_synced_ms": {"median": float(np.median(c_sync)), "p90": float(np.quantile(c_sync, 0.9))},
        "B_plus_C_ms": total,
        "criterion_ms": 1.0,
        "pass": bool(total <= 1.0),
        "peak_memory_gb": peak / 1e9,
        "peak_memory_step_gb": peak_step / 1e9,
        "evaluated": {
            "A": "weather state and all diagnostics (ws, diag) after every step",
            "B": "weather state, key and the folded diagnostic accumulators, every 10 steps "
            "(step and fold fused)",
            "C": "sampler positions and clip counters, every 10 steps (chained) or every step (synced), "
            "on the fixed weather state at the end of (B); wind_fields, sample_wind, alpha, clips and "
            "the position update",
        },
    }


# --- the study ----------------------------------------------------------------------------------------------


@dataclass
class Settings:
    """What a study run is asked to do (the CLI's arguments after parsing)."""

    steps: int
    t_spin: int = 15_000
    detect_every: int = 25
    frames_every: int = 100
    movie: tuple[int, int] | None = None  # None: the last min(10 000, steps) steps
    key: int = 0
    tracers_run: Path | None = None
    tracers_from: int | None = None
    uniform_tracers: int | None = None
    floor_held: bool = False
    movies: bool = True
    life_run: Path | None = None
    life_from: int | None = None
    snap_times: tuple[int, ...] = ()
    compile: bool = True
    argv: tuple[str, ...] = ()

    def movie_range(self) -> tuple[int, int]:
        if self.movie is not None:
            return self.movie
        return max(self.steps - 10_000, 0), self.steps

    def validate(self, p: WeatherParams) -> list[str]:
        bad = []
        if self.steps < 0:
            bad.append("--steps must be >= 0")
        if self.detect_every < 1:
            bad.append("--detect-every must be >= 1")
        if self.detect_every >= 1 and self.frames_every % self.detect_every:
            bad.append("--frames-every must be a multiple of --detect-every")
        if self.steps and not 0 <= self.t_spin < self.steps:
            bad.append("--spin-up must be in [0, steps)")
        if self.tracers_run is not None and self.tracers_from is None:
            bad.append("--tracers needs --from STEP")
        if self.tracers_run is not None and self.uniform_tracers is not None:
            bad.append("--tracers and --uniform-tracers are exclusive")
        if self.floor_held and self.tracers_run is None and self.uniform_tracers is None:
            bad.append("--floor-held needs tracers")
        return bad

    def snapshot_times(self) -> list[int]:
        """The four times of the PNG snapshots: a quarter, half, three quarters and all of the run, on
        detections."""
        e = self.detect_every
        return sorted({t for f in (0.25, 0.5, 0.75, 1.0) if (t := int(self.steps * f) // e * e) > 0})

    def frame_times(self) -> list[int]:
        """Movie frames (every ``frames_every`` steps in the movie range) and the four snapshot times."""
        out = set(self.snapshot_times())
        if self.movies:
            a, b = self.movie_range()
            out |= set(range((a // self.frames_every + 1) * self.frames_every, b + 1, self.frames_every))
        return sorted(out)


@dataclass
class StudyResult:
    p: WeatherParams
    st: Settings
    sim: Sim
    rec: Recorder
    tracks: list[Track]
    cfg: Config | None
    baseline: dict[str, Any] | None
    wall_s: float
    tracer_count: int = 0


def forbid_out(out: Path, run: Path | None) -> None:
    """The study never writes inside a tracer source run or under a ``runs`` directory (spec §5)."""
    o = out.resolve()
    if run is not None:
        r = Path(run).resolve()
        if o == r or r in o.parents:
            raise ValueError(f"--out {out} is inside the tracer source run {run}")
    if "runs" in o.parts:
        raise ValueError(f"--out {out} is under a runs/ directory; the run directories are read only")


def run_study(p: WeatherParams, st: Settings, progress: Callable[[str], None] | None = None) -> StudyResult:
    """Advance the weather ``st.steps`` steps with detection, tracking, tracers and recording.

    With ``floor_held`` a second arm runs beside the plain one: the same tracers and reseeds with ``w`` set to
    0 while in hash layer 0 (spec §5)."""
    say = progress or (lambda s: None)
    cfg: Config | None = None
    rigs: list[TracerRig] = []
    baseline: dict[str, Any] | None = None
    if st.tracers_run is not None:
        assert st.tracers_from is not None
        say(f"loading the tracer source {st.tracers_run} at step {st.tracers_from}")
        cfg, pos, cap = load_tracer_source(st.tracers_run, st.tracers_from)
        _check_world(p, cfg)
        baseline = baseline_occupancy(pos, cfg)
        baseline["resampled"] = resampled_peaks(lambda rng: pos[rng.integers(0, len(pos), len(pos))], cfg)
        for held in (False, True)[: 2 if st.floor_held else 1]:
            rigs.append(TracerRig(p, cfg, pos, cap, key=st.key, floor_held=held, compile=st.compile))
    elif st.uniform_tracers is not None:
        cfg = uniform_cfg(p)
        _check_world(p, cfg)
        m = st.uniform_tracers
        d = dims(cfg)

        def uniform(rng: np.random.Generator) -> np.ndarray:
            u = rng.random((m, 3))
            return np.stack([u[:, 0] * p.L, u[:, 1] * p.L, d.z_lo + u[:, 2] * (d.z_hi - d.z_lo)], axis=-1)

        for held in (False, True)[: 2 if st.floor_held else 1]:
            rigs.append(TracerRig(p, cfg, None, None, m=m, key=st.key, floor_held=held, compile=st.compile))
        baseline = baseline_occupancy(np.array(rigs[0].pos), cfg)
        baseline["resampled"] = resampled_peaks(uniform, cfg)
    sim = Sim(
        p,
        st.key,
        rig=rigs[0] if rigs else None,
        rig_floor_held=rigs[1] if len(rigs) > 1 else None,
        detect_every=st.detect_every,
        compile=st.compile,
    )
    rec = Recorder(
        p,
        t_spin=st.t_spin,
        frame_times=st.frame_times(),
        snap_times=st.snap_times,
        rig=sim.rig,
        rig_floor_held=sim.rig_floor_held,
        cfg=cfg,
        progress=say,
    )
    rec.start(sim)
    sim.callbacks.append(rec)
    if rigs:

        def reset(s: Sim) -> None:
            for rig in (s.rig, s.rig_floor_held):
                if rig is not None:
                    rig.reset_peak()

        sim.at(st.t_spin, reset)
    t0 = time.perf_counter()
    sim.advance(st.steps)
    wall_s = time.perf_counter() - t0
    return StudyResult(p, st, sim, rec, sim.tracker.finish(), cfg, baseline, wall_s, rigs[0].m if rigs else 0)


def uniform_cfg(p: WeatherParams) -> Config:
    """The ``64k3d`` world for uniform tracers; its size follows the weather domain when that is not 320 (the
    height must match: ``2·height <= L`` is a world rule, so small test domains need a small ``height``)."""
    if abs(p.L - 320.0) < 1e-9:
        return load_config(preset="64k3d")
    return load_config(preset="64k3d", overrides=[f"world.size={p.L:g}"])


def _check_world(p: WeatherParams, cfg: Config) -> None:
    if abs(cfg.world.size - p.L) > 1e-9 or abs(cfg.world.height - p.height) > 1e-9:
        raise ValueError(
            f"the weather domain ({p.L:g} x {p.height:g}) does not match the world of the tracers "
            f"({cfg.world.size:g} x {cfg.world.height:g})"
        )


# --- assembly of the metrics --------------------------------------------------------------------------------


def _strip(d: dict[str, Any], *keys: str) -> dict[str, Any]:
    return {k: v for k, v in d.items() if k not in keys}


def occupancy_summary(res: StudyResult, arm: str = "plain") -> dict[str, Any] | None:
    """Per hash-layer band: baseline, window peak with its step and cell, p99 of the per-step maxima, steps
    over 0.75 C and over C, the clip shares (spec §5); the criterion is a peak below C in each band. ``arm``
    is "plain" or "floor_held"."""
    rec, st, cfg = res.rec, res.st, res.cfg
    if res.baseline is None or cfg is None or arm not in rec.arms:
        return None
    ar = rec.arms[arm]
    tau, occ = ar.arrays()
    sel = (tau > st.t_spin) & (tau <= st.steps)
    cap = cfg.sensing.cell_capacity
    n_steps = int(sel.sum())
    bands = {}
    for b, name in enumerate(BAND_NAMES):
        v = occ[sel, b]
        if v.size == 0:
            bands[name] = {"label": BAND_LABELS[b], "baseline": res.baseline["peak"][b], "peak": None}
            continue
        j = int(np.argmax(v))
        cell = int(occ[sel, 3 + b][j])
        bands[name] = {
            "label": BAND_LABELS[b],
            "baseline_peak": res.baseline["peak"][b],
            "baseline_agents": res.baseline["agents"][b],
            "resampled_peak_median": res.baseline["resampled"]["peak_median"][b],
            "resampled_peak_max": res.baseline["resampled"]["peak_max"][b],
            "peak": int(v[j]),
            "peak_step": int(tau[sel][j]),
            "peak_cell": cell,
            "peak_cell_xyz": decode_cell(cell, cfg) if cell >= 0 else None,
            "p99_of_step_maxima": float(np.quantile(v, 0.99)),
            "steps_over_0.75C": int((v > 0.75 * cap).sum()),
            "steps_over_C": int((v > cap).sum()),
            "pass": bool(v[j] < cap),
        }
    m = res.tracer_count
    clip = ar.clip_total
    zq = [r for r in rec.rows if r["t"] > st.t_spin and "zq0" in r]
    return {
        "capacity": cap,
        "threshold": f"< {cap} in each band",
        "tracers": m,
        "window_steps": n_steps,
        "bands": bands,
        "pass": bool(all(b.get("pass", False) for b in bands.values())),
        "clip": {
            "u_max_share": float(clip[0] / max(m * n_steps, 1)),
            "w_edge_share_of_samples": float(clip[1] / max(m * n_steps, 1)),
            "w_edge_share_of_edge_samples": float(clip[1] / max(clip[2], 1)),
            "edge_sample_share": float(clip[2] / max(m * n_steps, 1)),
        },
        "z_band_quarters": {
            "baseline": res.baseline["z_band_quarters"],
            "window_median": [
                float(np.median([r[f"zq{q}"] for r in zq])) if zq else None for q in range(N_BANDS)
            ],
        },
        "device_peak_check": [[int(x) for x in row] for row in ar.peak_device],
        "floor_held": arm == "floor_held",
    }


def stationarity(
    rows: Sequence[dict[str, Any]], t_spin: int, t_end: int, block: int = 5000
) -> dict[str, Any]:
    """The last three ``block``-step blocks of the window agree within 20% in kinetic energy, rain and storm
    count (spec §7); not evaluated when the window is shorter than three blocks."""
    t0 = t_end - 3 * block
    if t0 < t_spin:
        return {"evaluated": False, "reason": f"the window is shorter than {3 * block} steps"}
    out: dict[str, Any] = {"evaluated": True, "block": block}
    ok = True
    for name, fn in (
        ("kinetic_energy", lambda r: r["ke_zonal_1"] + r["ke_eddy_1"] + r["ke_zonal_2"] + r["ke_eddy_2"]),
        ("rain", lambda r: r["rain"]),
        ("storm_count", lambda r: r["n_storms_north"] + r["n_storms_south"]),
    ):
        means = []
        for b in range(3):
            lo, hi = t0 + b * block, t0 + (b + 1) * block
            vals = [fn(r) for r in rows if lo < r["t"] <= hi]
            means.append(float(np.mean(vals)) if vals else math.nan)
        mean = float(np.nanmean(means))
        spread = (max(means) - min(means)) / mean if mean > 0 else math.nan
        out[name] = {"block_means": means, "spread": spread, "pass": bool(spread <= 0.2)}
        ok = ok and bool(spread <= 0.2)
    out["pass"] = ok
    return out


HYPER_GATE = 0.8  # the health gate on max (u1 - u2)^2 / (g'(h1 + h2)); the exact eigenvalue test also gates
MASS_GATE = 1e-5  # relative drift of the column mass
LIMITER_SHARE_GATE = 0.05  # share of qualifying detections that are limiter-bound
LIMITER_H_FACTOR = 1.5  # a detection is limiter-bound when min h1 <= LIMITER_H_FACTOR * h_min (the outcrop)
FORECAST_BAND = (0.5, 2.0)  # the 1000-step median error must lie in this many radii (STORMS 8)
MOBILITY_RADII = 0.5  # median drift per 1000 steps below this: the error is mostly centroid jitter


def forecast_criterion(f1000: dict[str, Any]) -> dict[str, Any]:
    """Criterion 3 from the 1000-step row of ``forecast_stats``, with the D9 reading.

    ``in_range`` is the criterion as written (median error between 0.5 and 2 radii); the criterion counts as
    met (``evidence``) only if the median also exceeds the median jitter floor and the skill against
    persistence is positive. ``mobility`` flags a median drift under 0.5 radii per 1000 steps.
    """
    n = f1000["n"]
    med = f1000["median_error"]
    in_range = bool(n > 0 and FORECAST_BAND[0] <= med <= FORECAST_BAND[1])
    exceeds = bool(n > 0 and med > f1000["median_floor"])
    skill_pos = bool(n > 0 and f1000["skill"] > 0)
    evidence = in_range and exceeds and skill_pos
    if not in_range:
        reading = "not met"
    elif evidence:
        reading = "met, and above the jitter floor with positive skill over persistence"
    else:
        reading = "met as written, not evidence of forecastable motion"
    return {
        "in_range": in_range,
        "exceeds_jitter_floor": exceeds,
        "skill_positive": skill_pos,
        "evidence": evidence,
        "mobility": bool(n == 0 or f1000["drift_radii_median"] < MOBILITY_RADII),
        "reading": reading,
    }


def health_gate(
    *,
    n_safety: int,
    hyper_max: float,
    eig_complex: int,
    eps_local_max: float,
    mass_dev: float,
    limiter_share: float,
) -> dict[str, bool]:
    """The run's health gate (spec §3.4): every entry must hold. A NaN fails its entry."""
    return {
        "no_safety_clips": n_safety == 0,
        "hyperbolicity_below_0.8": bool(hyper_max < HYPER_GATE),
        "exact_eigenvalue_test_real": eig_complex == 0,
        "eps_local_max_below_1": bool(eps_local_max < 1.0),
        "mass_drift_below_1e-5": bool(mass_dev < MASS_GATE),
        "limiter_bound_share_at_most_5%": bool(limiter_share <= LIMITER_SHARE_GATE),
    }


def early_stop_inputs(
    cnt: dict[str, Any], life: dict[str, Any], fc: dict[str, Any], min_median: float = 1000.0
) -> dict[str, Any]:
    """The D10 early-stop test per hemisphere (spec §3.4): at least one storm on average, a Kaplan-Meier
    median lifetime of at least ``min_median`` steps (the longest observed age if the curve stays above 0.5)
    and a median drift of at least 0.5 radii per 1000 steps. ``run_meets`` is true when both hemispheres meet
    all three: the early stop fires when, after round 2, no run does. Each value is reported with the
    risk-set figures that say whether it can be read."""
    f1000 = fc["leads"][1000]
    out: dict[str, Any] = {}
    for name in ("north", "south"):
        km = life["by_hemisphere"][name]
        mean_storms = cnt[name]["mean"] if name in cnt else math.nan
        drift = f1000["by_hemisphere"][name]["median_persistence"]
        ok_storms = bool(mean_storms >= 1.0)
        ok_life = bool(km["n"] > 0 and km["reached"] >= min_median)
        ok_drift = bool(drift >= MOBILITY_RADII)
        out[name] = {
            "mean_storms": mean_storms,
            "lifetime_km": km,
            "drift_radii_per_1000": drift,
            "drift_n": f1000["by_hemisphere"][name]["n"],
            "meets": {"storms": ok_storms, "lifetime": ok_life, "drift": ok_drift},
            "ok": ok_storms and ok_life and ok_drift,
        }
    out["run_meets"] = all(out[h]["ok"] for h in ("north", "south"))
    return out


def assemble_metrics(
    res: StudyResult,
    *,
    bench: dict[str, Any] | None = None,
    steer: dict[str, Any] | None = None,
    vortex: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The metrics of a study (``metrics.json``) and the arrays the figures need.

    Returns ``(metrics, arrays)``; ``arrays`` holds numpy data that is not JSON (counts, Kaplan-Meier curve,
    forecast rows, latitude histogram, zonal means, histograms).
    """
    p, st, rec, tracks, sim = res.p, res.st, res.rec, res.tracks, res.sim
    t_spin, t_end, every = st.t_spin, st.steps, st.detect_every
    times = detection_times(t_spin, t_end, every)
    cnt = storm_counts(tracks, times, p.min_dets)
    full = storm_counts(tracks, [r["t"] for r in rec.rows], p.min_dets)
    for r, nn, ns in zip(rec.rows, full["n_north"], full["n_south"], strict=True):
        r["n_storms_north"], r["n_storms_south"] = int(nn), int(ns)
    life = lifetime_stats(tracks, t_spin, p.min_dets)
    fc = forecast_stats(tracks, p, t_spin, t_end, p.min_dets, detect_every=every)
    dr = drift_stats(tracks, p, t_spin, t_end, p.min_dets)
    lat = latitude_stats(tracks, p, t_spin, t_end, p.min_dets)
    occ = occupancy_summary(res)
    occ_fh = occupancy_summary(res, "floor_held")
    timed: list[dict[str, Any]] = []

    def timed_add(label: str, steps: float | None) -> None:
        if steps is not None and math.isfinite(steps):
            timed.append({"label": label, **wall(steps)})

    # --- criterion 1: storm count
    c1 = {
        "value": {h: cnt[h] for h in ("north", "south")} if times else None,
        "in_range_fraction": cnt.get("in_range_fraction"),
        "threshold": "2 to 6 storms per hemisphere at every detection time of the window",
        "pass": bool(cnt.get("all_in_range", False)),
        "detail": {"window_times": len(times), "lo": 2, "hi": 6},
    }
    # --- criterion 2: lifetime
    km: KaplanMeier = life["km"]
    c2 = {
        "value": life["median"] if life["median"] is not None else life["median_text"],
        "threshold": ">= 3000 steps",
        "pass": bool(life["pass"]),
        "detail": {
            "n": life["n"],
            "events": life["events"],
            "censored": life["censored"],
            "median_text": life["median_text"],
            "max_age": life["max_age"],
            "naive_median_completed": life["naive_median_completed"],
            "wall": wall(life["median"] if life["median"] is not None else life["max_age"]),
            "n_risk_at_median": life["n_risk_at_median"],
            "thin_age": life["thin_age"],
            "min_risk": KM_MIN_RISK,
            "median_reliable": life["median_reliable"],
            "reading": life["reading"],
            "n_risk_at_threshold": life["n_risk_at_threshold"],
            "survival_before_threshold": life["survival_before_threshold"],
            "companions": life["companions"],
            "by_hemisphere": life["by_hemisphere"],
        },
    }
    timed_add(
        "storm lifetime, Kaplan-Meier median"
        if life["median"] is not None
        else "longest observed storm age (median not reached)",
        life["median"] if life["median"] is not None else life["max_age"],
    )
    timed_add("storm lifetime, naive median of completed tracks", life["naive_median_completed"])
    # --- criterion 3: forecast
    f1000 = fc["leads"][1000]
    med = f1000["median_error"]
    fcrit = forecast_criterion(f1000)
    mobility = fcrit["mobility"]
    c3 = {
        "value": med,
        "threshold": "0.5 <= median extrapolation error at a 1000-step lead <= 2 storm radii "
        "(D9: also above the jitter floor, skill > 0)",
        "pass": fcrit["evidence"],
        "detail": {
            "pass_as_written": fcrit["in_range"],
            "d9_reading": fcrit["reading"],
            "exceeds_jitter_floor": fcrit["exceeds_jitter_floor"],
            "median_jitter_floor": f1000["median_floor"],
            "skill_vs_persistence": f1000["skill"],
            "median_persistence_error": f1000["median_persistence"],
            "p25": f1000["p25"],
            "p75": f1000["p75"],
            "n": f1000["n"],
            "drops": f1000["drops"],
            "mobility_flag_median_drift_below_0.5_radii": mobility,
            "median_drift_per_1000_radii": f1000["drift_radii_median"],
            "leads": {str(k): _strip(v, "rows") | {"wall": wall(k)} for k, v in fc["leads"].items()},
        },
    }
    for k in fc["leads"]:
        timed_add(f"forecast lead {k}", k)
    # --- criterion 4: occupancy
    if occ is not None:
        c4 = {
            "value": {name: b.get("peak") for name, b in occ["bands"].items()},
            "threshold": occ["threshold"],
            "pass": occ["pass"],
            "detail": _strip(occ, "pass"),
        }
    else:
        c4 = {"value": None, "threshold": "< C in each band", "pass": None, "detail": "tracers not run"}
    # --- criterion 5: cost
    if bench is not None:
        c5 = {
            "value": bench["B_plus_C_ms"],
            "threshold": "(B) + (C) <= 1.0 ms",
            "pass": bench["pass"],
            "detail": bench,
        }
    else:
        c5 = {"value": None, "threshold": "(B) + (C) <= 1.0 ms", "pass": None, "detail": "not measured"}
    # --- criterion 6: latitudes (reported)
    c6 = {
        "value": {
            "median_pole_distance": lat["median_pole_distance"],
            "within_16_of_pole": lat["within_16_of_pole"],
            "north": lat["north"],
            "south": lat["south"],
        },
        "threshold": "reported (clustering at L/4 and 3L/4)",
        "pass": bool(lat["n"] > 0),
        "detail": {"n_detections": lat["n"], "bin": 8.0},
    }
    criteria = {
        "storm_count": c1,
        "lifetime": c2,
        "forecast": c3,
        "occupancy": c4,
        "cost": c5,
        "latitudes": c6,
    }

    timed_add("drift gap (displacement between detections)", dr["gap"])

    # --- dynamics (spec §8 diagnostics)
    qd = [d for tr in qualifying(tracks, p.min_dets) for d in tr.dets if t_spin <= d.t <= t_end]
    wmax = np.array([d.wind_max for d in qd])
    froude = wmax / p.c_bc if wmax.size else wmax
    dh = np.array([d.dh1 / p.H1 for d in qd])
    n_win = max(rec.n_win, 1)
    rows_y = (np.arange(p.n) + 0.5) * p.dx
    zonal = rec.zonal_sum / n_win
    jets = {}
    for k in range(2):
        j = int(np.argmax(np.abs(zonal[k])))
        jets[f"layer{k + 1}"] = {
            "max_abs_zonal_mean": float(abs(zonal[k][j])),
            "at_y": float(rows_y[j]),
            "signed": float(zonal[k][j]),
            "zonal_to_eddy_ke": float(rec.ke_sum[k, 0] / rec.ke_sum[k, 1])
            if rec.ke_sum[k, 1] > 0
            else math.nan,
            "profile_y": rows_y,
            "profile_zonal_mean": zonal[k],
        }
    bg = {f"layer{k + 1}": {"p50": rec.quantile_bg(k, 0.5), "p90": rec.quantile_bg(k, 0.9)} for k in range(2)}
    window_bands = [b for b in rec.band_log if b[0] >= t_spin]
    n_det_win = max(sum(1 for r in rec.rows if r["t"] >= t_spin), 1)
    bands = {
        "detections_with_band_share": len({b[0] for b in window_bands}) / n_det_win,
        "count": len(window_bands),
        "per_hemisphere": {
            name: {
                "count": sum(1 for b in window_bands if b[1] == h),
                "mean_area_cells": float(np.mean([b[2] for b in window_bands if b[1] == h]))
                if any(b[1] == h for b in window_bands)
                else None,
                "median_latitude": float(np.median([b[3] for b in window_bands if b[1] == h]))
                if any(b[1] == h for b in window_bands)
                else None,
            }
            for h, name in enumerate(("north", "south"))
        },
    }
    diagnostics = {
        "waves": p.derived(),
        "storm_climate": {
            "tracks": len(tracks),
            "qualifying": len(qualifying(tracks, p.min_dets)),
            "transients": len(tracks) - len(qualifying(tracks, p.min_dets)),
            "births_in_window": sum(
                1 for tr in qualifying(tracks, p.min_dets) if t_spin <= tr.t_first <= t_end
            ),
            "splits": sum(1 for tr in tracks if tr.parent),
            "mergers": sum(1 for tr in tracks if tr.end_reason == "merged"),
            "dissipations": sum(1 for tr in tracks if tr.end_reason == "dissipated"),
            "counts": _strip(cnt, "times", "n_north", "n_south"),
            "bands": bands,
        },
        "forecast": {"slots": fc["slots"], "need": fc["need"]},
        "early_stop": early_stop_inputs(cnt, life, fc),
        "drift": _strip(dr, "records"),
        "latitudes": lat,
        "core": {
            "n_detections": int(wmax.size),
            "froude": _quantiles(froude, (0.5, 0.9))
            | {"max": float(froude.max()) if froude.size else math.nan},
            "dh1_over_H1": _quantiles(dh, (0.5, 0.9)) | {"max": float(dh.max()) if dh.size else math.nan},
            "wind": _quantiles(wmax, (0.5, 0.9)) | {"max": float(wmax.max()) if wmax.size else math.nan},
        },
        "hyperbolicity": {
            "device_running_max": float(sim.max_run[2]),
            "host_snapshot_max": rec.hyper_snap_max,
            "gate": 0.8,
            "rigid_lid_bound": 1.0,
            "exact_checked": rec.eig_checked,
            "exact_complex": rec.eig_complex,
        },
        "background_wind": bg,
        "jets": jets,
        "agent_wind": rec.winds.summary() if rec.winds is not None else None,
        "tracers": occ,
        "tracers_floor_held": occ_fh,
        "histograms": {
            "zeta_bins_f0": ZETA_BINS.tolist(),
            "zeta_counts": rec.zeta_hist.tolist(),
            "p_bins": P_BINS.tolist(),
            "p_counts": rec.p_hist.tolist(),
        },
    }
    # --- health
    cnts = {k: int(v) for k, v in zip(ACC_CNT, sim.totals_cnt, strict=True)}
    sums = {k: float(v) for k, v in zip(ACC_SUM, sim.totals_sum, strict=True)}
    qd_bound = sum(1 for d in qd if d.h1_min <= LIMITER_H_FACTOR * p.h_min)
    limiter_share = qd_bound / len(qd) if qd else 0.0
    q_res = rec.q_last - rec.q0 - (sums["sum_E"] - sums["sum_C"] + sums["sum_Qseed"] + sums["q_clamp"])
    gn = [r["grid_noise"] for r in rec.rows if r["t"] >= t_spin]
    stat = stationarity(rec.rows, t_spin, t_end)
    gate = health_gate(
        n_safety=cnts["n_safety"],
        hyper_max=float(sim.max_run[2]),
        eig_complex=rec.eig_complex,
        eps_local_max=float(sim.max_run[3]),
        mass_dev=rec.mass_dev,
        limiter_share=limiter_share,
    )
    health = {
        "counters": cnts,
        "sums": sums,
        "courant_max": float(sim.max_run[0]),
        "speed_max": float(sim.max_run[1]),
        "hyper_max": float(sim.max_run[2]),
        "eps_local_max": float(sim.max_run[3]),
        "mass_drift_max": rec.mass_dev,
        "mass_drift_final": abs(rec.mass_last - rec.mass0) / rec.mass0 if rec.mass0 else math.nan,
        "q_budget_residual": q_res,
        "q_budget_residual_relative": q_res / rec.q0 if rec.q0 else math.nan,
        "limiter_bound_share": limiter_share,
        "limiter_bound_detections": qd_bound,
        "grid_noise": {
            "window_median": float(np.median(gn)) if gn else math.nan,
            "window_max": float(np.max(gn)) if gn else math.nan,
            "first": rec.rows[0]["grid_noise"] if rec.rows else math.nan,
        },
        "stationarity": stat,
        "gate": gate,
        "pass": bool(all(gate.values())),
    }
    # --- numerics, deviations, ratio
    numerics = {
        "sl_order": p.sl_order,
        "kappa_d": p.kappa_d,
        "nu": p.nu,
        "vortex_decay": vortex,
    }
    d3 = (
        f"D3: q_pool = {p.q_pool:g} > 1, so Q_e > Q_s over warm-pool cores (STORMS 3.2 has Q_e <= Q_s); a "
        "steady storm with eps < 1 needs evaporation inside the saturated core"
        if p.q_pool > 1.0
        else "D3: q_pool <= 1 (no supersaturated warm pools; no storm is expected to form)"
    )
    deviations = [
        d3,
        f"D7: the moist-thickness limiter on evaporation (m = h1 - gamma Q >= m_min = {p.m_min:g}) is on; "
        "it is not in STORMS 3.2",
        f"divergence damping kappa_d = {p.kappa_d:g} in place of the probe's nu (spec S7)",
    ]
    if any(row[5] != 0.0 or row[6] != 0.0 for row in p.pools):
        deviations.append("D8: warm-pool drift is on (a last-resort mobility lever)")
    if p.sl_order == 3:
        deviations.append(
            "sl_order 3 (Catmull-Rom) in place of STORMS 3.2 step 1's bilinear semi-lagrangian advection"
        )
    base = W.START_SET  # the detection thresholds of the spec's starting set, not of the (tuned) defaults
    changed = [
        k
        for k in ("zeta_th", "p_th", "a_min", "r_min", "miss_grace", "min_dets", "detect_every")
        if getattr(p, k) != getattr(base, k)
    ]
    if changed:
        deviations.append(
            "detection thresholds changed from the start set: "
            + ", ".join(f"{k} = {getattr(p, k):g}" for k in changed)
        )
    ratio = None
    if st.life_run is not None or st.tracers_run is not None:
        run = st.life_run if st.life_run is not None else st.tracers_run
        assert run is not None
        step_from = st.life_from if st.life_from is not None else (st.tracers_from or 0)
        org = organism_lifetime(Path(run), step_from)
        storm_med = life["median"] if life["median"] is not None else None
        ratio = {
            "organism": org,
            "storm_km_median": storm_med,
            "storm_max_age": life["max_age"],
            "ratio_median_to_median": (storm_med / org["median"])
            if storm_med is not None and org.get("median")
            else None,
            "ratio_lower_bound_max_age": (life["max_age"] / org["median"])
            if storm_med is None and org.get("median")
            else None,
            "ratio_to_mean_lifetime": (storm_med / org["mean"])
            if storm_med is not None and org.get("mean")
            else None,
            "which": "Storm lifetime is the Kaplan-Meier median; organism lifetime is the median age at "
            "death over the deaths of the source run in the listed event files (the mean is beside it)",
        }
        if org.get("median"):
            timed_add("organism median lifetime (source run)", org["median"])
    timed_add("run length", t_end)
    timed_add("spin-up", t_spin)
    timed_add("window", t_end - t_spin)
    verdict_parts = {k: v["pass"] for k, v in criteria.items()}
    failed = [k for k, v in verdict_parts.items() if v is False]
    unmeasured = [k for k, v in verdict_parts.items() if v is None]
    if not health["pass"]:
        failed.append("health")
    verdict = {
        "pass": not failed and not unmeasured,
        "failed": failed,
        "not_measured": unmeasured,
        "text": (
            "PASS"
            if not failed and not unmeasured
            else "PASS on every measured criterion (not measured: " + ", ".join(unmeasured) + ")"
            if not failed
            else "FAIL: " + ", ".join(failed)
        ),
    }
    metrics = {
        "run": {
            "params_hash": params_hash(p),
            "key": st.key,
            "steps": t_end,
            "spin_up": t_spin,
            "window": [t_spin, t_end],
            "detect_every": every,
            "device": str(mx.default_device()),
            "n": p.n,
            "wall_s": res.wall_s,
            "steps_per_s": t_end / res.wall_s if res.wall_s > 0 else None,
            "tracers": res.tracer_count,
            "wall_clock_of_run": wall(t_end),
        },
        "criteria": criteria,
        "diagnostics": diagnostics,
        "health": health,
        "bench": bench,
        "steerability": steer,
        "lifetime_ratio": ratio,
        "numerics": numerics,
        "deviations": deviations,
        "wall_clock": timed,
        "verdict": verdict,
    }
    arrays = {
        "counts_times": cnt.get("times", np.zeros(0)),
        "counts_north": cnt.get("n_north", np.zeros(0)),
        "counts_south": cnt.get("n_south", np.zeros(0)),
        "km": km,
        "forecast": fc,
        "drift": dr,
        "lat": lat,
        "zonal_mean": zonal,
        "rows_y": rows_y,
        "occupancy": occ,
        "occupancy_floor_held": occ_fh,
    }
    return metrics, arrays


# --- steerability driver ------------------------------------------------------------------------------------


def run_steerability(
    res: StudyResult,
    impulses: Sequence[ImpulseSpec],
    autos: Sequence[tuple[int, float, float, float, int]],
    shifts: Sequence[PoolOp],
    gains: Sequence[PoolOp],
    progress: Callable[[str], None] | None = None,
    horizons: dict[str, int] | None = None,
    after: Sequence[int] = (2000, 5000),
) -> dict[str, Any]:
    """The steerability deliverables (spec §9a) from the study's restart snapshots: one control continuation
    per snapshot (shared by every experiment from it), the impulse pairs, the pool shifts and gains.
    ``horizons`` gives the continuation lengths ("impulse" 1000, "shift" 10 000, "gain" 5 000 steps) and
    ``after`` the times after a gain at which counts and winds are compared."""
    p = res.p
    hz = {"impulse": 1000, "shift": 10_000, "gain": 5_000, **(horizons or {})}
    say = progress or (lambda s: None)
    out: dict[str, Any] = {"impulse": [], "pool_shift": [], "pool_gain": []}
    by_t0: dict[int, dict[str, list[Any]]] = {}
    for spec in impulses:
        by_t0.setdefault(spec.t0, {"imp": [], "ops": []})["imp"].append((None, spec))
    for t0, x, y, ang, dur in autos:
        snap = res.rec.snaps.get(t0)
        if snap is None:
            out["impulse"].append({"error": f"no snapshot at t0 = {t0}"})
            continue
        for f, spec in auto_impulses(snap, p, t0, x, y, ang, dur):
            by_t0.setdefault(t0, {"imp": [], "ops": []})["imp"].append((f, spec))
    for op in (*shifts, *gains):
        by_t0.setdefault(op.t0, {"imp": [], "ops": []})["ops"].append(op)
    main = {tr.id: tr for tr in res.tracks}
    for t0, grp in sorted(by_t0.items()):
        snap = res.rec.snaps.get(t0)
        if snap is None:
            for _ in grp["imp"]:
                out["impulse"].append({"error": f"no snapshot at t0 = {t0}"})
            continue
        need = max([hz["impulse"]] * bool(grp["imp"]) + [hz[o.kind] for o in grp["ops"]])
        say(f"steerability: control continuation from t0 = {t0} ({need} steps)")
        ctrl, _ = run_fork(snap, p, need, compile=res.st.compile)
        diff, n_cmp = 0.0, 0
        for tr in ctrl:
            m = main.get(tr.id)
            if m is None:
                continue
            for d in tr.dets:
                e = det_at(m, d.t)
                if e is not None:
                    n_cmp += 1
                    diff = max(
                        diff, float(np.hypot(min_image_np(d.x - e.x, p.L), min_image_np(d.y - e.y, p.L)))
                    )
        check = {
            "t0": t0,
            "n_compared_detections": n_cmp,
            "max_position_difference_units": diff,
            "bitwise_reproduced": bool(n_cmp > 0 and diff == 0.0),
        }
        out.setdefault("control_vs_study", []).append(check)
        if grp["imp"]:
            r = impulse_experiment(
                snap, p, grp["imp"], horizon=hz["impulse"], ctrl=ctrl, compile=res.st.compile, progress=say
            )
            out["impulse"].append(r)
        for op in grp["ops"]:
            r = pool_experiment(
                snap,
                p,
                op,
                ctrl=ctrl,
                horizon=hz[op.kind],
                after=after,
                compile=res.st.compile,
                progress=say,
            )
            out["pool_shift" if op.kind == "shift" else "pool_gain"].append(r)
    return out


# --- outputs ------------------------------------------------------------------------------------------------


def write_tracks_csv(path: Path, tracks: Sequence[Track], p: WeatherParams, every: int) -> None:
    """One row per track and detection; a gap that ends in a re-detection adds coasting rows at the
    interpolated position (spec §9)."""
    cols = [
        "id", "t", "x", "y", "r_eq", "area", "hemisphere", "zeta_max_f0", "wind_max", "h1_min", "dh1",
        "rain", "coasting", "event", "parent", "merged_into",
    ]  # fmt: skip
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for tr in tracks:
            hemi = "N" if tr.hemi == 0 else "S"
            last = len(tr.dets) - 1
            for i, d in enumerate(tr.dets):
                if i > 0:
                    prev = tr.dets[i - 1]
                    n_gap = (d.t - prev.t) // every - 1
                    for k in range(1, n_gap + 1):
                        f = k / (n_gap + 1)
                        gx = (prev.ux + f * (d.ux - prev.ux)) % p.L
                        gy = (prev.uy + f * (d.uy - prev.uy)) % p.L
                        gap_row = [tr.id, prev.t + k * every, f"{gx:.4f}", f"{gy:.4f}"]
                        w.writerow([*gap_row, "", "", hemi, "", "", "", "", "", 1, "coast", "", ""])
                ev = ""
                if i == 0:
                    ev = "split" if tr.parent else "birth"
                if i == last:
                    ev = (ev + "+" if ev else "") + (tr.end_reason or "alive")
                row = [tr.id, d.t, f"{d.x:.4f}", f"{d.y:.4f}", f"{d.r_eq:.4f}", d.area, hemi]
                row += [f"{d.zeta_max_f0:.5f}", f"{d.wind_max:.6f}", f"{d.h1_min:.4f}", f"{d.dh1:.4f}"]
                row += [f"{d.rain:.6e}", 0, ev, tr.parent or "", tr.merged_into or ""]
                w.writerow(row)


def write_storms_csv(path: Path, tracks: Sequence[Track], p: WeatherParams, t_end: int) -> None:
    cols = [
        "id", "hemisphere", "birth", "end", "lifetime", "end_reason", "censored", "qualifying",
        "detections", "genesis_x", "genesis_y", "peak_zeta_f0", "peak_wind", "parent", "merged_into",
    ]  # fmt: skip
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for tr in tracks:
            g = tr.dets[0]
            row = [tr.id, "N" if tr.hemi == 0 else "S", tr.t_first, tr.t_last, tr.t_last - tr.t_first]
            row += [tr.end_reason or "alive", int(not tr.ended), int(len(tr.dets) >= p.min_dets)]
            row += [len(tr.dets), f"{g.x:.3f}", f"{g.y:.3f}", f"{max(d.zeta_max_f0 for d in tr.dets):.5f}"]
            row += [f"{max(d.wind_max for d in tr.dets):.6f}", tr.parent or "", tr.merged_into or ""]
            w.writerow(row)


def write_timeseries_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    cols: list[str] = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.8g}" if isinstance(v, float) else v) for k, v in r.items()})


def _f(x: Any, nd: int = 3) -> str:
    """A number for the report: ``n/a`` for None and NaN, integers as is, floats to ``nd`` significant
    digits."""
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, int):
        return str(x)
    return f"{x:.0f}" if abs(x) >= 1000 else f"{x:.{nd}g}"


def _fw(steps: float | None) -> str:
    """A time in steps with its wall-clock value at 14.4 and at 30 steps/s."""
    if steps is None or not math.isfinite(steps):
        return "n/a"
    return (
        f"{steps:g} steps ({steps / 14.4:.0f} s = {steps / 14.4 / 60:.1f} min at 14.4/s; "
        f"{steps / 30:.0f} s = {steps / 30 / 60:.1f} min at 30/s)"
    )


def _yn(v: bool | None) -> str:
    return "not measured" if v is None else ("yes" if v else "NO")


def _climate_lines(m: dict[str, Any], p: WeatherParams) -> list[str]:
    cr, dg = m["criteria"], m["diagnostics"]
    c1, c2 = cr["storm_count"], cr["lifetime"]
    sc = dg["storm_climate"]
    out = ["## Storm climate", ""]
    out.append(
        f"- Tracks {sc['tracks']} ({sc['qualifying']} qualifying with at least {p.min_dets} detections, "
        f"{sc['transients']} transients); births in the window {sc['births_in_window']}, splits "
        f"{sc['splits']}, mergers {sc['mergers']}, dissipations {sc['dissipations']}."
    )
    for h in ("north", "south") if c1["value"] else ():
        v = c1["value"][h]
        out.append(
            f"- {h}: min {v['min']}, max {v['max']}, mean {_f(v['mean'])}, in range "
            f"{_f(100 * v['in_range_fraction'], 4)}%; histogram {v['histogram']}."
        )
    b = sc["bands"]
    out.append(
        f"- Bands (zonal strips, excluded from storms): {b['count']} band detections in the window, in "
        f"{_f(100 * b['detections_with_band_share'], 3)}% of detection times; north "
        f"{b['per_hemisphere']['north']}, south {b['per_hemisphere']['south']}."
    )
    d2 = c2["detail"]
    out.append(
        f"- Kaplan-Meier lifetime median: {d2['median_text']} steps; naive median of completed tracks "
        f"{_f(d2['naive_median_completed'])}; longest observed age {_fw(d2['max_age'])}."
    )
    out.append(
        f"- Reading of the median: {d2['reading']}. At risk at the criterion age (3000): "
        f"{d2['n_risk_at_threshold']}; survival just before it {_f(d2['survival_before_threshold'])}; "
        f"first age with fewer than {d2['min_risk']} at risk: "
        f"{_f(d2['thin_age']) if d2['thin_age'] is not None else 'none'}."
    )
    cm = d2["companions"]
    out.append(
        f"- Companions (not the criterion): {_f(100 * cm['share_detections_in_long_storms'], 3)}% of the "
        f"window's {cm['detections']} storm detections belong to storms observed living at least "
        f"{cm['long_storm_steps']:g} steps (the criterion read on detections: "
        f"{_yn(cm['detection_weighted_pass'])}); Kaplan-Meier over full histories (no truncation, spin-up "
        f"included) median {cm['full_histories']['median_text']} (n {cm['full_histories']['n']}, "
        f"reliable {_yn(cm['full_histories']['median_reliable'])}); without split-born tracks median "
        f"{cm['without_split_born']['median_text']} (n {cm['without_split_born']['n']}); "
        f"split-born tracks {cm['split_born']['n']} (median duration "
        f"{_f(cm['split_born']['median_duration'])})."
    )
    for h in ("north", "south"):
        v = d2["by_hemisphere"][h]
        out.append(
            f"- Lifetime, {h}: n {v['n']}, events {v['events']}, median {v['median_text']}, at risk at the "
            f"median {v['n_risk_at_median'] if v['n_risk_at_median'] is not None else '-'}, reliable "
            f"{_yn(v['median_reliable'])}."
        )
    es = dg["early_stop"]
    out.append(
        "- Early-stop test (D10, per hemisphere: mean storms >= 1, Kaplan-Meier median >= 1000, median drift "
        f">= 0.5 radii per 1000 steps): run meets it {_yn(es['run_meets'])}; "
        + "; ".join(
            f"{h}: storms {_f(es[h]['mean_storms'])}, lifetime "
            f"{es[h]['lifetime_km']['median_text']}, drift {_f(es[h]['drift_radii_per_1000'])}"
            for h in ("north", "south")
        )
        + "."
    )
    lat = dg["latitudes"]
    out.append(
        f"- Latitudes (centroid y, 8-unit bins in `lat_hist.png`; poles at {p.L / 4:g} and {3 * p.L / 4:g}): "
        f"median pole distance N {_f(lat['north']['median_pole_distance'])}, "
        f"S {_f(lat['south']['median_pole_distance'])}; share within 16 units of a pole "
        f"N {_f(lat['north']['within_16_of_pole'])}, S {_f(lat['south']['within_16_of_pole'])}."
    )
    return [*out, ""]


def _forecast_lines(m: dict[str, Any]) -> list[str]:
    c3 = m["criteria"]["forecast"]
    d3 = c3["detail"]
    out = ["## Forecasting", ""]
    out.append(
        f"- Median extrapolation error at 1000 steps: {_f(c3['value'])} radii (p25 {_f(d3['p25'])}, "
        f"p75 {_f(d3['p75'])}, n {d3['n']}); persistence {_f(d3['median_persistence_error'])}; "
        f"jitter floor {_f(d3['median_jitter_floor'])}; skill against persistence "
        f"{_f(d3['skill_vs_persistence'])}."
    )
    out.append(
        f"- D9 reading: {d3['d9_reading']}. Passes as written: {_yn(d3['pass_as_written'])}; above the "
        f"floor: {_yn(d3['exceeds_jitter_floor'])}."
    )
    out.append(
        f"- Mobility: median drift per 1000 steps {_f(d3['median_drift_per_1000_radii'])} radii; flag (under "
        f"0.5 radii, the error is mostly centroid jitter): "
        f"{_yn(d3['mobility_flag_median_drift_below_0.5_radii'])}."
    )
    bh = d3["leads"]["1000"]["by_hemisphere"]
    out.append(
        "- Median drift per 1000 steps by hemisphere (radii): "
        + ", ".join(f"{h} {_f(bh[h]['median_persistence'])} (n {bh[h]['n']})" for h in ("north", "south"))
        + "."
    )
    out.append(
        "- The jitter floor assumes white centroid noise; a slowly meandering (anchored) storm has "
        "correlated centroid errors that the fit residuals understate, so the floor reads low for it and "
        "only the skill condition protects the D9 reading there."
    )
    out.append(f"- Drops at the 1000-step lead by cause: {d3['drops']}.")
    for lead, v in d3["leads"].items():
        out.append(
            f"- Lead {lead} ({_fw(float(lead))}): median {_f(v['median_error'])} radii, n {v['n']}, "
            f"persistence {_f(v['median_persistence'])}, floor {_f(v['median_floor'])}, "
            f"skill {_f(v['skill'])}."
        )
    dr = m["diagnostics"]["drift"]
    for hname in ("north", "south"):
        v = dr[hname]
        out.append(
            f"- Drift per 250 steps ({_fw(250)}), {hname}: n {v['n']}, magnitude median "
            f"{_f(v['units']['p50'])} units ({_f(v['radii']['p50'])} radii), p90 {_f(v['units']['p90'])}; "
            f"zonal (east +) median {_f(v['zonal']['p50'])}; poleward median {_f(v['poleward']['p50'])}."
        )
    return [*out, ""]


def _dynamics_lines(m: dict[str, Any]) -> list[str]:
    dg = m["diagnostics"]
    w, co, hy, bg = dg["waves"], dg["core"], dg["hyperbolicity"], dg["background_wind"]
    out = ["## Dynamics (STORMS §8)", ""]
    out.append(
        f"- Wave speeds: c_ext {_f(w['c_ext'])}, c_bc {_f(w['c_bc'])}, c_moist {_f(w['c_moist'])}; "
        f"deformation radii L_d {_f(w['ld_pole'])} (pole) and {_f(w['ld_pool'])} (pools), moist "
        f"{_f(w['ld_moist_pole'])} / {_f(w['ld_moist_pool'])}; inertial period "
        f"{_f(w['inertial_period_pole'])} steps; eps {_f(w['eps'])}; potential intensity "
        f"{_f(w['potential_intensity'])}."
    )
    out.append(
        f"- Cores ({co['n_detections']} qualifying detections in the window): Froude (max wind / c_bc) "
        f"p50 {_f(co['froude']['p50'])}, p90 {_f(co['froude']['p90'])}, max {_f(co['froude']['max'])}; "
        f"dh1/H1 p50 {_f(co['dh1_over_H1']['p50'])}, p90 {_f(co['dh1_over_H1']['p90'])}, "
        f"max {_f(co['dh1_over_H1']['max'])}; core wind p50 {_f(co['wind']['p50'])}, "
        f"p90 {_f(co['wind']['p90'])}, max {_f(co['wind']['max'])}."
    )
    out.append(
        f"- Hyperbolicity max (u1 - u2)^2/(g'(h1 + h2)): device running max "
        f"{_f(hy['device_running_max'])}, host snapshots {_f(hy['host_snapshot_max'])} (gate "
        f"{hy['gate']}, the STORMS bound is {hy['rigid_lid_bound']}); exact 4x4 eigenvalue test on the 16 "
        f"largest per snapshot: {hy['exact_checked']} checked, {hy['exact_complex']} complex."
    )
    out.append(
        f"- Background wind outside storms: layer 1 p50 {_f(bg['layer1']['p50'])}, "
        f"p90 {_f(bg['layer1']['p90'])}; layer 2 p50 {_f(bg['layer2']['p50'])}, "
        f"p90 {_f(bg['layer2']['p90'])}."
    )
    for k in ("layer1", "layer2"):
        j = dg["jets"][k]
        out.append(
            f"- Jets, {k}: largest time-mean zonal-mean |u| {_f(j['max_abs_zonal_mean'])} at y = "
            f"{_f(j['at_y'])} (signed {_f(j['signed'])}); zonal-to-eddy kinetic energy "
            f"{_f(j['zonal_to_eddy_ke'])}."
        )
    aw = dg["agent_wind"]
    if aw and aw.get("n"):
        out.append(
            f"- Agent wind (`wind_layers`, {aw['n']} snapshots): p99 |u_h| over all layers "
            f"{_f(aw['p99_uh_all'])}; max |w| mid-column {_f(aw['max_abs_w_mid'])}; share of mid-column "
            f"samples with |(u_h, w)| above u_max {_f(aw['share_norm3_over_umax_mid'])}."
        )
    return [*out, ""]


def _tracer_table(t: dict[str, Any]) -> list[str]:
    out = [
        "| band | start peak (agents) | no-wind draws peak (median, max) | window peak (step, cell) "
        "| p99 of step maxima | steps over 0.75 C | steps over C |",
        "|---|---|---|---|---|---|---|",
    ]
    for bd in t["bands"].values():
        if bd.get("peak") is None:
            out.append(f"| {bd['label']} | {bd.get('baseline_peak', 'n/a')} | n/a | n/a | n/a | n/a | n/a |")
            continue
        out.append(
            f"| {bd['label']} | {bd['baseline_peak']} ({bd['baseline_agents']}) | "
            f"{_f(bd['resampled_peak_median'])}, {bd['resampled_peak_max']} | {bd['peak']} (step "
            f"{bd['peak_step']}, cell {bd['peak_cell_xyz']}) | {_f(bd['p99_of_step_maxima'])} | "
            f"{bd['steps_over_0.75C']} | {bd['steps_over_C']} |"
        )
    cl = t["clip"]
    out += [
        "",
        f"- Clip shares: u_max {_f(cl['u_max_share'])} of samples; w_edge "
        f"{_f(cl['w_edge_share_of_samples'])} of samples ({_f(cl['w_edge_share_of_edge_samples'])} of the "
        f"edge-layer samples, which are {_f(cl['edge_sample_share'])} of all).",
    ]
    return out


def _tracer_lines(m: dict[str, Any]) -> list[str]:
    t, fh = m["diagnostics"]["tracers"], m["diagnostics"]["tracers_floor_held"]
    out = ["## Tracers", ""]
    if t is None:
        return [*out, "Not run (no `--tracers` or `--uniform-tracers`).", ""]
    zb = t["z_band_quarters"]
    out += [
        f"{t['tracers']} tracers on exactly the agents' wind (clip included), {t['window_steps']} window "
        "steps. The no-wind column is the peak of independent draws of the tracer positions (what the "
        "turnover alone produces: the checkpoint's own peak is lower because drawing rows with replacement "
        "adds Poisson noise to its hot cells).",
        "",
        *_tracer_table(t),
        f"- Tracers per `z_band` quarter (RESULTS_3D's bands): at the start {zb['baseline']}, window median "
        f"{zb['window_median']}.",
        "",
    ]
    if fh is not None:
        out += [
            "Floor-held arm (the same tracers with w = 0 in hash layer 0, the gravity-bound bracket):",
            "",
            *_tracer_table(fh),
            "",
        ]
    return out


def _cost_lines(m: dict[str, Any]) -> list[str]:
    bn = m["bench"]
    out = ["## Cost", ""]
    if bn is None:
        return [*out, "Not measured (`--bench`).", ""]
    a, cs, ev = bn["A_synced_ms"], bn["C_sampler_synced_ms"], bn["evaluated"]
    out += [
        f"- Device {bn['device']}, debug {bn['debug']}, sampler at M = {bn['m_positions']}; {bn['steps']} "
        f"steps per timing.",
        f"- (B) chained {_f(bn['B_chained_ms'])} ms + (C) sampler chained "
        f"{_f(bn['C_sampler_chained_ms'])} ms = **{_f(bn['B_plus_C_ms'])} ms** against "
        f"{bn['criterion_ms']} ms: {_yn(bn['pass'])}.",
        f"- (A) synced: median {_f(a['median'])}, p90 {_f(a['p90'])}, mean {_f(a['mean'])} ms. Sampler "
        f"synced: median {_f(cs['median'])}, p90 {_f(cs['p90'])} ms.",
        f"- Peak memory {_f(bn['peak_memory_gb'])} GB. What each timing evaluated: A {ev['A']}; B {ev['B']}; "
        f"C {ev['C']}.",
        "",
    ]
    return out


def _health_lines(m: dict[str, Any]) -> list[str]:
    hl = m["health"]
    cn, sm, gn, st = hl["counters"], hl["sums"], hl["grid_noise"], hl["stationarity"]
    gate = ", ".join(f"{k}: {_yn(v)}" for k, v in hl["gate"].items())
    out = ["## Health", ""]
    out += [
        f"- Gate: {'**clean**' if hl['pass'] else '**NOT clean**'} ({gate}).",
        f"- Counters over the run: limited {cn['n_limited']}, clamps {cn['n_clamp']}, condensation-limited "
        f"{cn['n_cond_limited']}, relaxation-limited {cn['n_relax_limited']}, evaporation-limited "
        f"{cn['n_evap_limited']}, safety clips {cn['n_safety']}; clamp mass {_f(sm['clamp_mass'])}, clamped "
        f"moisture {_f(sm['q_clamp'])}.",
        f"- Courant max {_f(hl['courant_max'])}, speed max {_f(hl['speed_max'])}, eps_local max "
        f"{_f(hl['eps_local_max'])}; mass drift max {_f(hl['mass_drift_max'])} (final "
        f"{_f(hl['mass_drift_final'])}); moisture budget residual {_f(hl['q_budget_residual'])} "
        f"({_f(hl['q_budget_residual_relative'])} of the initial total).",
        f"- Limiter-bound qualifying detections (min h1 <= 1.5 h_min): {hl['limiter_bound_detections']} "
        f"({_f(100 * hl['limiter_bound_share'], 3)}%).",
        f"- Grid-noise index: window median {_f(gn['window_median'])}, window max {_f(gn['window_max'])}, "
        f"first snapshot {_f(gn['first'])}.",
    ]
    if st["evaluated"]:
        parts = "; ".join(
            f"{k} {_f(v['spread'])} ({_yn(v['pass'])})"
            for k, v in st.items()
            if isinstance(v, dict) and "spread" in v
        )
        out.append(f"- Stationarity (last three 5000-step blocks, spread within 20%): {parts}.")
    else:
        out.append(f"- Stationarity: not evaluated ({st['reason']}).")
    return [*out, ""]


def _numerics_lines(m: dict[str, Any]) -> list[str]:
    nm = m["numerics"]
    out = ["## Numerics", "", f"sl_order {nm['sl_order']}, kappa_d {nm['kappa_d']}, nu {nm['nu']}."]
    v = nm["vortex_decay"]
    if v:
        out += [
            "",
            f"Unforced-vortex decay, dry, N = {v['n']}, zeta 0.55 f0, sigma 9, north pole row; steps until "
            f"zeta_max < 0.3 f0 (resolution {v['every']}, limit {v['limit']}):",
            "",
            "| line | decay step | wall-clock |",
            "|---|---|---|",
        ]
        for name, ln in v["lines"].items():
            wc = _fw(ln["decay_step"]) if ln["decay_step"] is not None else "beyond the limit"
            out.append(f"| {name} | {ln['text']} | {wc} |")
    return [*out, ""]


def _ratio_lines(m: dict[str, Any]) -> list[str]:
    r = m["lifetime_ratio"]
    if not r:
        return []
    o = r["organism"]
    out = ["## Lifetime ratio", ""]
    if not o.get("median"):
        return [*out, f"- No deaths found in the event files of `{o['run']}`.", ""]
    ratio = r["ratio_median_to_median"]
    shown = _f(ratio) if ratio is not None else "> " + _f(r["ratio_lower_bound_max_age"])
    out += [
        f"- Organism lifetime: median age at death {_f(o['median'])} steps (mean {_f(o['mean'])}, p25 "
        f"{_f(o['p25'])}, p75 {_f(o['p75'])}; {o['n_deaths']} deaths in {o['files']} of `{o['run']}`).",
        f"- Storm to organism lifetime ratio: {shown} (storm median over organism median; a bound when the "
        f"storm median is not reached), against the organism mean {_f(r['ratio_to_mean_lifetime'])}. "
        f"{r['which']}.",
        "",
    ]
    return out


def render_report(m: dict[str, Any], p: WeatherParams) -> str:
    """``report.md`` from the metrics (the template of spec §9, filled)."""
    run, cr = m["run"], m["criteria"]
    out: list[str] = []
    a = out.append
    a(
        f"# W0 run {run['params_hash'][:8]} (key {run['key']}, {run['steps']} steps, spin-up "
        f"{run['spin_up']}, {run['device']})"
    )
    a("")
    a(
        f"Verdict: **{m['verdict']['text']}**. Window {run['window'][0]} to {run['window'][1]} "
        f"({_fw(run['window'][1] - run['window'][0])}); run wall time {_f(run['wall_s'], 4)} s."
    )
    a("")
    c1, c2, c3, c4, c5, c6 = (
        cr[k] for k in ("storm_count", "lifetime", "forecast", "occupancy", "cost", "latitudes")
    )
    a("| criterion | value | threshold | pass |")
    a("|---|---|---|---|")
    if c1["value"]:
        n, s = c1["value"]["north"], c1["value"]["south"]
        v1 = (
            f"N {n['min']} to {n['max']} (mean {_f(n['mean'])}), S {s['min']} to {s['max']} "
            f"(mean {_f(s['mean'])}); in range {_f(100 * c1['in_range_fraction'], 4)}% of "
            f"{c1['detail']['window_times']} detection times"
        )
    else:
        v1 = "no detection times in the window"
    a(f"| storms per hemisphere | {v1} | {c1['threshold']} | {_yn(c1['pass'])} |")
    d2, d3 = c2["detail"], c3["detail"]
    a(
        f"| median storm lifetime (Kaplan-Meier) | {d2['median_text']} steps (n {d2['n']}, events "
        f"{d2['events']}, censored {d2['censored']}) | {c2['threshold']} | {_yn(c2['pass'])} |"
    )
    a(
        f"| forecast error at 1000 steps | median {_f(c3['value'])} radii (jitter floor "
        f"{_f(d3['median_jitter_floor'])}, skill {_f(d3['skill_vs_persistence'])}, n {d3['n']}) | "
        f"{c3['threshold']} | {_yn(c3['pass'])} |"
    )
    if c4["value"] is not None:
        v4 = ", ".join(f"{k} {_f(v)}" for k, v in c4["value"].items())
        a(
            f"| tracer peak occupancy per hash cell | {v4} (C = {c4['detail']['capacity']}) | "
            f"{c4['threshold']} | {_yn(c4['pass'])} |"
        )
    else:
        a("| tracer peak occupancy per hash cell | not measured (no tracers) | < C in each band | n/a |")
    cost = f"{_f(c5['value'])} ms" if c5["value"] is not None else "not measured"
    a(f"| cost, (B) + (C) | {cost} | {c5['threshold']} | {_yn(c5['pass'])} |")
    v6 = c6["value"]
    a(
        f"| storm latitudes | median pole distance N {_f(v6['north']['median_pole_distance'])}, "
        f"S {_f(v6['south']['median_pole_distance'])}; within 16 of a pole {_f(v6['within_16_of_pole'])} | "
        f"{c6['threshold']} | reported |"
    )
    a("")
    out += _climate_lines(m, p)
    out += _forecast_lines(m)
    out += _dynamics_lines(m)
    out += _tracer_lines(m)
    out += _cost_lines(m)
    out += _health_lines(m)
    out += _numerics_lines(m)
    out += _ratio_lines(m)
    if m["steerability"]:
        out += ["## Steerability (spec §9a)", "", render_steerability(m["steerability"]), ""]
    out += ["## Deviations", ""]
    out += [f"- {d}" for d in m["deviations"]]
    out += [
        "",
        "## Wall-clock of the time-valued results",
        "",
        "| quantity | steps | seconds at 14.4/s | minutes at 14.4/s | seconds at 30/s | minutes at 30/s |",
        "|---|---|---|---|---|---|",
    ]
    for e in m["wall_clock"]:
        out.append(
            f"| {e['label']} | {e['steps']:g} | {_f(e['s_at_14.4'])} | {_f(e['min_at_14.4'])} | "
            f"{_f(e['s_at_30'])} | {_f(e['min_at_30'])} |"
        )
    out += [
        "",
        "## Notes",
        "",
        "Figures: `lat_hist.png`, `counts.png`, `km.png`, `forecast.png`, `drift.png`, `occupancy.png`, "
        "`jets.png`, `hist.png`; frames in `frames/`; movies `h1.gif`, `vort.gif`, `rain.gif`, "
        "`tracers_hz0.gif`, `tracers_hz1_11.gif`, `tracers_hz12.gif`.",
        "",
    ]
    return "\n".join(out)


def _neighbour_text(h: dict[str, Any]) -> str:
    if not h["available"]:
        return f"+{h['horizon']}: n/a"
    return f"+{h['horizon']}: {_f(h['displacement_units'])} units ({_f(h['displacement_radii'])} radii)"


def _horizon_text(h: dict[str, Any]) -> str:
    if not h["available"]:
        return (
            f"+{h['horizon']}: not available (control present {_yn(h['survived_control'])}, "
            f"perturbed {_yn(h['survived_perturbed'])})"
        )
    return (
        f"+{h['horizon']} ({_fw(h['horizon'])}): displacement {_f(h['displacement_units'])} units = "
        f"{_f(h['displacement_radii'])} radii; survived: control {_yn(h['survived_control'])}, "
        f"perturbed {_yn(h['survived_perturbed'])}"
    )


def render_steerability(s: dict[str, Any]) -> str:
    out: list[str] = []
    a = out.append
    for chk in s.get("control_vs_study", []):
        a(
            f"- Control continuation from t0 = {chk['t0']} against the study: max position difference "
            f"{_f(chk['max_position_difference_units'])} units (bitwise reproduced: "
            f"{_yn(chk['bitwise_reproduced'])})."
        )
    for g in s.get("impulse", []):
        if "error" in g:
            a(f"- Impulse: {g['error']}.")
            continue
        a(f"- Wind impulses from t0 = {g['t0']} on storm {g['target_id']} (control shared):")
        for r in g["runs"]:
            sp = r["spec"]
            a(
                f"  - factor {_f(r['factor'])} (momentum ratio {_f(r['momentum_ratio'])}), |U| "
                f"{_f(r['speed'])}, radius {_f(sp['radius'])}, duration {sp['duration']}, layers "
                f"{sp['layers']}:"
            )
            out += [f"    - {_horizon_text(h)}." for h in r["horizons"]]
            for nb in r["neighbours"]:
                txt = "; ".join(_neighbour_text(h) for h in nb["horizons"])
                a(f"    - neighbour {nb['id']} at {_f(nb['distance'])} units: {txt}.")
    for r in s.get("pool_shift", []):
        gl, anc = r["genesis_lag"], r["anchored"]
        end = ""
        if anc and anc["end"] and anc["end"]["reason"] != "alive":
            end = f" (the track {anc['end']['reason']} at step {anc['end']['t_last']})"
        a(
            f"- Pool shift {r['op']['index']} at t0 = {r['t0']} by ({_f(r['op']['dx'])}, "
            f"{_f(r['op']['dy'])}) ({r['hemisphere']}): genesis lag "
            f"{(str(gl) + ' = ' + _fw(gl)) if gl is not None else 'none'}; first detection in the new "
            f"footprint {r['first_detection_lag'] if r['first_detection_lag'] is not None else 'none'}; "
            f"anchored storm: {anc['displacement_lag_text'] if anc else 'none at t0'}{end}."
        )
    for r in s.get("pool_gain", []):
        for row in r["gain"]["rows"]:
            a(
                f"- Pool gain x{_f(r['gain']['factor'])} on pool {r['op']['index']} at t0 = {r['t0']}, "
                f"after {_fw(row['after'])}: storm count {row['count_perturbed']} against "
                f"{row['count_control']} (change {row['count_change']:+d}; in the pool's hemisphere "
                f"{row['count_change_hemisphere']:+d}); core wind {_f(row['core_wind_perturbed'])} against "
                f"{_f(row['core_wind_control'])} (change {_f(row['core_wind_change'])})."
            )
    return "\n".join(out)


def write_study_outputs(
    out: Path,
    res: StudyResult,
    metrics: dict[str, Any],
    arrays: dict[str, Any],
    *,
    plots: bool = True,
    movies: bool = True,
    progress: Callable[[str], None] | None = None,
) -> list[str]:
    """Write every file of spec §9 into ``out``; returns the file names written."""
    p, st = res.p, res.st
    written: list[str] = []

    def put(name: str, text: str) -> None:
        (out / name).write_text(text, encoding="utf-8")
        written.append(name)

    put("params.json", params_record(p, st))
    put("metrics.json", dumps(metrics))
    write_tracks_csv(out / "tracks.csv", res.tracks, p, st.detect_every)
    write_storms_csv(out / "storms.csv", res.tracks, p, st.steps)
    write_timeseries_csv(out / "timeseries.csv", res.rec.rows)
    written += ["tracks.csv", "storms.csv", "timeseries.csv"]
    put("report.md", render_report(metrics, p))
    if plots:
        try:  # matplotlib is the optional analysis extra: w0_plots imports it on its first figure
            from alife.analysis import w0_plots

            written += w0_plots.write_all(
                out, res, metrics, arrays, movies=movies and st.movies, progress=progress
            )
        except ImportError as e:
            (progress or print)(f"skipping figures and movies: {e}")
    return written


def params_record(p: WeatherParams, st: Settings | None = None, extra: dict[str, Any] | None = None) -> str:
    """``params.json``: the full parameter set, its hash, the derived values and the provenance."""
    rec = {
        "params": p.to_dict(),
        "hash": params_hash(p),
        "derived": p.derived(),
        "validate": p.validate(),
        "cli": list(st.argv) if st is not None else [],
        "git_rev": _git_rev(),
        "mlx_version": mx.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": str(mx.default_device()),
        **(extra or {}),
    }
    return dumps(rec)


MARKER = ".w0"


def prepare_out(out: Path, overwrite: bool) -> None:
    """``out`` must be new or empty, or hold a ``.w0`` marker (then ``--overwrite`` removes only the files the
    marker lists); the marker is rewritten at the end of a run."""
    if out.exists():
        entries = [x for x in out.iterdir()]
        marker = out / MARKER
        if entries and not marker.exists():
            raise FileExistsError(f"{out} is not empty and has no {MARKER} marker; choose a new directory")
        if entries and not overwrite:
            raise FileExistsError(f"{out} holds a previous run; pass --overwrite to replace its files")
        if marker.exists():
            try:
                listed = json.loads(marker.read_text(encoding="utf-8")).get("files", [])
            except (OSError, json.JSONDecodeError):
                listed = []
            for name in listed:
                f = (out / name).resolve()
                if out.resolve() in f.parents and f.is_file():
                    f.unlink()
    out.mkdir(parents=True, exist_ok=True)
    (out / MARKER).write_text(dumps({"files": []}), encoding="utf-8")


def finish_out(out: Path, written: Sequence[str]) -> None:
    """Record the files of this run in the ``.w0`` marker: the names written, plus the snapshots in
    ``frames/`` (a figure step that fails half way may have left some). Nothing else that happens to be in
    ``out`` is recorded, so ``--overwrite`` never removes a file this tool did not write."""
    frames = (str(f.relative_to(out)) for f in (out / "frames").glob("t*.png") if f.is_file())
    (out / MARKER).write_text(dumps({"files": sorted({*written, *frames})}), encoding="utf-8")


# --- command line -------------------------------------------------------------------------------------------


def _load_params(path: str | None, sets: Sequence[str]) -> WeatherParams:
    """The parameters of a study. A ``--params`` file is a sparse override of ``weather.START_SET`` (the
    spec's starting set, which the tuning rounds' files were written against: ``{}`` is the start set).
    Without one the base is the ``WeatherParams`` defaults (the W0 confirmed set). ``--set`` overrides go on
    top of either."""
    d: dict[str, Any] = {}
    if path:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(d, dict):
            raise ValueError("--params must hold a JSON object (a sparse override)")
    for s in sets:
        k, _, v = s.partition("=")
        if not _:
            raise ValueError(f"--set needs KEY=VALUE, got {s!r}")
        d[k.strip()] = json.loads(v)
    return WeatherParams.from_dict(d, base=W.START_SET if path else None)


def _load_json(path: str | None, flag: str) -> dict[str, Any] | None:
    """The JSON object at ``path`` (``None`` when the flag is absent). A missing file, bad JSON or a value
    that is not an object is a ``ValueError`` that names the flag."""
    if not path:
        return None
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ValueError(f"{flag} {path}: {e}") from e
    if not isinstance(data, dict):
        raise ValueError(f"{flag} {path} must hold a JSON object")
    return data


def _parse_auto(text: str) -> tuple[int, float, float, float, int]:
    """``--impulse-auto T0,X,Y,DIR_DEG[,DURATION]`` as ``(t0, x, y, direction in degrees, duration)``."""
    f = [x.strip() for x in text.split(",")]
    if len(f) not in (4, 5):
        raise ValueError("--impulse-auto needs T0,X,Y,DIR_DEG[,DURATION]")
    return int(f[0]), float(f[1]), float(f[2]), float(f[3]), int(f[4]) if len(f) == 5 else 50


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m alife.analysis.w0",
        description="The W0 storm study: runs the weather model offline on MLX (docs/STORMS.md, W0 spec).",
    )
    ap.add_argument(
        "--params",
        help="sparse JSON override of weather.START_SET, the spec's starting set ({} gives it; the tuning "
        "rounds' files are written against it). Without --params the base is the WeatherParams defaults "
        "(the W0 confirmed set)",
    )
    ap.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE", help="one parameter override (JSON value)"
    )
    ap.add_argument(
        "--steps",
        type=int,
        default=None,
        help="study length (default 60000; 0, or --bench alone, skips the study)",
    )
    ap.add_argument(
        "--out", required=True, help="output directory (new or empty, or a previous .w0 run with --overwrite)"
    )
    ap.add_argument("--tracers", metavar="RUN", help="tracer source run directory (read only); needs --from")
    ap.add_argument(
        "--from", dest="tracers_from", type=int, metavar="STEP", help="checkpoint step of the tracer source"
    )
    ap.add_argument("--uniform-tracers", type=int, metavar="M", help="M tracers uniform in the 64k3d domain")
    ap.add_argument(
        "--floor-held", action="store_true", help="the floor-held tracer arm (w = 0 in hash layer 0)"
    )
    ap.add_argument("--key", type=int, default=0)
    ap.add_argument("--spin-up", type=int, default=15_000)
    ap.add_argument(
        "--detect-every", type=int, default=None, help="default: the parameter set's detect_every (25)"
    )
    ap.add_argument("--frames-every", type=int, default=100)
    ap.add_argument("--movie-steps", metavar="A:B", help="movie segment (default: the last 10000 steps)")
    ap.add_argument("--no-movies", action="store_true", help="skip the GIFs")
    ap.add_argument("--no-plots", action="store_true", help="skip figures and movies")
    ap.add_argument("--device", choices=("gpu", "cpu"), help="default: MLX's default device")
    ap.add_argument("--bench", action="store_true", help="measure the cost (spec §7)")
    ap.add_argument("--bench-steps", type=int, default=2000)
    ap.add_argument("--bench-m", type=int, default=65_536, help="positions of the sampler timing (C)")
    ap.add_argument("--bench-json", metavar="FILE", help="attach an earlier bench.json to the study")
    ap.add_argument(
        "--vortex-decay", action="store_true", help="the unforced-vortex decay table instead of a study"
    )
    ap.add_argument("--vortex-json", metavar="FILE", help="attach an earlier vortex_decay.json to the study")
    ap.add_argument(
        "--life-run",
        metavar="RUN",
        help="run whose deaths give the organism lifetime (default: the tracer source)",
    )
    ap.add_argument(
        "--life-from",
        type=int,
        metavar="STEP",
        help="first step of the 50000-step span of deaths (default --from)",
    )
    ap.add_argument("--impulse", action="append", default=[], metavar="T0,X,Y,UX,UY,RADIUS,DURATION[,LAYERS]")
    ap.add_argument(
        "--impulse-auto",
        action="append",
        default=[],
        metavar="T0,X,Y,DIR_DEG[,DURATION]",
        help="three impulses of 0.5x, 1x and 2x the nearest storm core's momentum, RADIUS = its r_eq",
    )
    ap.add_argument("--impulse-horizon", type=int, default=1000, help="steps each impulse run continues")
    ap.add_argument("--shift-horizon", type=int, default=10_000, help="steps each pool-shift run continues")
    ap.add_argument("--gain-horizon", type=int, default=5_000, help="steps each pool-gain run continues")
    ap.add_argument("--gain-after", default="2000,5000", help="steps after a gain to compare count and wind")
    ap.add_argument("--pool-shift", action="append", default=[], metavar="T0,INDEX,DX,DY")
    ap.add_argument("--pool-gain", action="append", default=[], metavar="T0,INDEX,FACTOR")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    say: Callable[[str], None] = (lambda s: None) if args.quiet else (lambda s: print(s, flush=True))
    if args.device:
        mx.set_default_device(mx.gpu if args.device == "gpu" else mx.cpu)
    try:
        p = _load_params(args.params, args.set)
    except (OSError, ValueError, json.JSONDecodeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    bad = p.validate()
    if bad:
        print("invalid weather parameters:\n  " + "\n  ".join(bad), file=sys.stderr)
        return 2
    out = Path(args.out)
    steps = args.steps if args.steps is not None else (0 if (args.bench or args.vortex_decay) else 60_000)
    try:
        movie = None
        if args.movie_steps:
            lo, _, hi = args.movie_steps.partition(":")
            movie = (int(lo), int(hi))
        impulses = [ImpulseSpec.parse(x) for x in args.impulse]
        autos = [_parse_auto(x) for x in args.impulse_auto]
        shifts = [PoolOp.parse_shift(x) for x in args.pool_shift]
        gains = [PoolOp.parse_gain(x) for x in args.pool_gain]
        gain_after = tuple(int(x) for x in args.gain_after.split(",") if x.strip())
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    snap_times = tuple(
        sorted({s.t0 for s in impulses} | {a[0] for a in autos} | {o.t0 for o in (*shifts, *gains)})
    )
    st = Settings(
        steps=steps,
        t_spin=args.spin_up,
        detect_every=args.detect_every if args.detect_every is not None else p.detect_every,
        frames_every=args.frames_every,
        movie=movie,
        key=args.key,
        tracers_run=Path(args.tracers) if args.tracers else None,
        tracers_from=args.tracers_from,
        uniform_tracers=args.uniform_tracers,
        floor_held=args.floor_held,
        movies=not args.no_movies,
        life_run=Path(args.life_run) if args.life_run else None,
        life_from=args.life_from,
        snap_times=snap_times,
        argv=tuple(argv if argv is not None else sys.argv[1:]),
    )
    if steps and not args.vortex_decay and (bad := st.validate(p)):
        print("error: " + "; ".join(bad), file=sys.stderr)
        return 2
    for t0 in snap_times:
        if t0 % st.detect_every or not 0 < t0 <= steps:
            print(
                f"error: experiment time {t0} must be a multiple of --detect-every inside the run",
                file=sys.stderr,
            )
            return 2
    try:  # read what is attached first: --overwrite may delete these very files (a study re-run in place)
        bench = _load_json(args.bench_json, "--bench-json")
        vortex = _load_json(args.vortex_json, "--vortex-json")
        forbid_out(out, st.tracers_run)
        prepare_out(out, args.overwrite)
    except (OSError, ValueError) as e:  # FileExistsError and JSONDecodeError are subclasses
        print(f"error: {e}", file=sys.stderr)
        return 2
    written: list[str] = []
    if args.vortex_decay:
        say("vortex decay table")
        table = vortex_decay_table(p, steps or 5000)
        (out / "vortex_decay.json").write_text(dumps(table), encoding="utf-8")
        (out / "params.json").write_text(params_record(p, st), encoding="utf-8")
        written += ["vortex_decay.json", "params.json"]
        say(dumps({k: v["text"] for k, v in table["lines"].items()}))
        finish_out(out, written)
        return 0
    for name, attached in (("bench.json", bench), ("vortex_decay.json", vortex)):
        if attached is not None:  # keep the measurement beside the study (it may have been in this directory)
            (out / name).write_text(dumps(attached), encoding="utf-8")
            written.append(name)
    if args.bench:
        cfg = uniform_cfg(p)
        try:
            _check_world(p, cfg)
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        bench = run_bench(p, cfg, m=args.bench_m, steps=args.bench_steps, progress=say)
        (out / "bench.json").write_text(dumps(bench), encoding="utf-8")
        written.append("bench.json")
        say(f"bench: (B) + (C) = {bench['B_plus_C_ms']:.3f} ms (criterion {bench['criterion_ms']} ms)")
    if steps == 0:
        (out / "params.json").write_text(params_record(p, st), encoding="utf-8")
        written.append("params.json")
        finish_out(out, written)
        return 0
    from alife.runio import CheckpointError

    try:
        res = run_study(p, st, say)
    except (ValueError, OSError, CheckpointError) as e:  # a bad tracer source or a domain that does not match
        print(f"error: {e}", file=sys.stderr)
        return 2
    steer = None
    if impulses or autos or shifts or gains:
        steer = run_steerability(
            res,
            impulses,
            autos,
            shifts,
            gains,
            say,
            horizons={
                "impulse": args.impulse_horizon,
                "shift": args.shift_horizon,
                "gain": args.gain_horizon,
            },
            after=gain_after,
        )
    metrics, arrays = assemble_metrics(res, bench=bench, steer=steer, vortex=vortex)
    written += write_study_outputs(
        out, res, metrics, arrays, plots=not args.no_plots, movies=not args.no_movies, progress=say
    )
    finish_out(out, written)
    say(f"verdict: {metrics['verdict']['text']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
