"""Storm detection, the storm tracker and the storm layer of a played world (DESIGN §33.3).

**The tracker** (``Frame``, ``read_frame``, ``Comp``, ``Detection``, ``detect``, ``Det``, ``Track``,
``Tracker``) moved here from ``analysis/w0.py``, the W0 study driver, which imports and re-exports the names
(``tests/test_analysis_w0.py`` runs unchanged). The study driver is 4,000 lines of offline MLX tooling; the
play loop needs only the tracker, and ``weather_obs.py`` was split out of it for the same reason. Behaviour is
the W0 tracker's: ``Tracker.update`` differs only in how it fills its dictionaries (no subscript assignment,
the lint rule of this module) and ``_predict`` is public as ``predict``. Detection, tracking and the rules of
§30 / STORMS §6 are unchanged.

**The layer** (``StormLayer``) is the one stateful host component besides the observer. Per tick (the
detection cadence, ``WeatherParams.detect_every``) it reads the weather once, detects and tracks storms, and
produces the ``STRM`` object, the ``MAP1`` planes, the tracker events for the chronicle and the line of
``observers/storms.jsonl``. It names a storm when its peak lower-layer vorticity reaches ``NAME_ZETA`` (0.5
f0; the lower tracks are disturbances). Its **complete state** (``StormLayer.state``) is what makes a stopped
and resumed play the same play: every field ``Tracker.update`` reads, and the peak, first detection, name and
extent of each active track. It is written next to each checkpoint (``RunDir.write_storms_state``) and read
back by ``StormLayer.open``.

**Initial storms** (G3b, DESIGN §34.1). The first tick of a layer made by this run (a new world, or one
whose tracker state was lost) finds the storms that already existed; their ``born`` and ``named`` events
carry ``"initial": true`` in the tick's line so that the chronicle does not tell them as happenings
(``game.chronicle``; the persistence rule, ``STORM_MIN_DETECTIONS``, is applied there, on the logged lines).
The flag is the layer's ``initial_pending``, saved in the state while it is set, so that a stop before the
first tick does not change what the first tick says.

Host code: numpy only apart from ``read_frame_extra`` (one ``mx.eval`` per tick). No subscript assignment
(``tests/test_lint.py``): arrays by concatenation and comprehension, dictionaries by literals.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, Final, NamedTuple

import mlx.core as mx
import numpy as np

from alife import streamrec, weather_obs
from alife import weather as W
from alife.game.society import raw_house_names
from alife.weather import WeatherParams, WeatherState

__all__ = [
    "KEEP_DETS",
    "NAME_ZETA",
    "STORMS_FORMAT",
    "STORM_MIN_DETECTIONS",
    "STORM_NAME_SALT",
    "STRM_LEADS",
    "TRACK_POINTS",
    "Comp",
    "Det",
    "Detection",
    "Frame",
    "StormLayer",
    "StormTick",
    "Track",
    "Tracker",
    "dc_reference",
    "detect",
    "min_image_np",
    "read_frame",
    "read_frame_extra",
]

TWO_PI = 2.0 * math.pi
SPLIT_NEAR_TIE = 0.9  # tracker rule 4: overlaps within 10% of a track's best count as tied (ZETA decides)

# The periodic labelling and the centre/corner helpers live in ``alife.weather_obs`` (the engine's metrics
# rows use them too).
label_periodic = weather_obs.label_periodic
centre_winds = weather_obs.centre_winds
corners_to_centres = weather_obs.corners_to_centres
hemisphere_sign = weather_obs.hemisphere_sign


def min_image_np(d: Any, length: float) -> Any:
    """Minimum-image displacement ``d - L*round(d/L)`` (numpy; half-to-even rounding, like
    ``ops.min_image``)."""
    return d - length * np.round(np.asarray(d) / length)


# --- storm detection (W0 spec §6) ---------------------------------------------------------------------------


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


def read_frame_extra(
    ws: WeatherState, p: WeatherParams, t: int, extra: Sequence[mx.array] = ()
) -> tuple[Frame, tuple[np.ndarray, ...]]:
    """The nine fields a detection reads (W0 spec §6) and the arrays of ``extra``, in one device sync."""
    z = W.vorticity(ws, p)
    arrs = (z, ws.P, ws.h1, ws.h2, ws.u1, ws.v1, ws.u2, ws.v2, ws.Q)
    mx.eval(*arrs, *extra)
    return Frame(t, *(np.array(a) for a in arrs)), tuple(np.array(a) for a in extra)


def read_frame(ws: WeatherState, p: WeatherParams, t: int) -> Frame:
    """The nine fields a detection reads (spec §6), in one device sync."""
    return read_frame_extra(ws, p, t)[0]


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
    # In units of the REFERENCE f0 (0.025, streamrec.F0), not the config's: NAME_ZETA, STRONG_ZETA and the
    # zeta plane of MAP1 keep one meaning in every world (DESIGN §37.4). Identical where f0 is 0.025.
    zmax = np.maximum.reduceat(sz[jj, ii], head) / streamrec.F0
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

    def predict(self, tr: Track, t: int) -> tuple[float, float]:
        """The unwrapped position of ``tr`` at ``t``: its last centroid plus the least-squares velocity of its
        last four detections times the elapsed time (the tracker's own extrapolation, the baseline of STORMS
        §6.3)."""
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

    # The two distances the matching reads (the geometry of the world): the torus here, the great circle in
    # ``storms_sph.SphereTracker`` (DESIGN §46.7), which overrides them, ``predict`` and ``_append``.
    def _gap(self, c: Comp, last: Det) -> float:
        """The distance from a component's centroid to a track's last detection."""
        return float(np.hypot(min_image_np(c.x - last.x, self.p.L), min_image_np(c.y - last.y, self.p.L)))

    def _gap_to(self, c: Comp, predicted: tuple[float, ...]) -> float:
        """The distance from a component's centroid to a predicted position (``predict``'s result)."""
        return float(
            np.hypot(min_image_np(c.x - predicted[0], self.p.L), min_image_np(c.y - predicted[1], self.p.L))
        )

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
                dist = self._gap(comps[ci], last)
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
            comp_of_track.setdefault(tid, ci)
            track_of_comp.setdefault(ci, tid)
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
            if ci not in track_of_comp:
                parent_of_comp.setdefault(ci, tid)

        # (2) the position fallback among what no overlap explains
        free_tracks = [tr for tr in act if tr.id not in comp_of_track and tr.id not in explained_tracks]
        free_comps = [i for i in range(len(comps)) if i not in track_of_comp and i not in parent_of_comp]
        cand: list[tuple[float, int, int]] = []
        for tr in free_tracks:
            predicted = self.predict(tr, t)
            last = tr.dets[-1]
            for ci in free_comps:
                c = comps[ci]
                if c.hemi != tr.hemi:
                    continue
                dist = self._gap_to(c, predicted)
                if dist <= max(last.r_eq, c.r_eq):
                    cand.append((dist, tr.id, ci))
        cand.sort()
        for _, tid, ci in cand:
            if tid in comp_of_track or ci in track_of_comp:
                continue
            comp_of_track.setdefault(tid, ci)
            track_of_comp.setdefault(ci, tid)

        for tid, ci in comp_of_track.items():
            self._append(by_id[tid], t, comps[ci])
        # (5) births, split-offs included, in component order
        born = {
            ci: self._new(t, c, parent_of_comp.get(ci, 0)).id
            for ci, c in enumerate(comps)
            if ci not in track_of_comp
        }
        ids = [track_of_comp.get(ci) or born.get(ci, 0) for ci in range(len(comps))]
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

    def compact(self, keep: int) -> None:
        """Drop the ended tracks and all but the last ``keep`` detections of the others: the play loop's
        memory bound. ``update`` reads only the last detection (its footprint, centroid and extent) and the
        last four (``predict``), so a ``keep`` of four or more changes no result."""
        self.tracks = [tr for tr in self.tracks if not tr.ended]
        for tr in self.tracks:
            tr.dets = tr.dets[-keep:]


# --- the storm layer ----------------------------------------------------------------------------------------

STORMS_FORMAT: Final = "alife-storms"  # the ``format`` of ``observers/storms_<t>.json``
STORMS_VERSION: Final = 1
STORM_MIN_DETECTIONS: Final = (
    8  # detections before the chronicle tells a track's birth (200 steps at tick 25)
)
KEEP_DETS: Final = 40  # detections of an active track kept (and persisted)
TRACK_POINTS: Final = 24  # points of a STRM track polyline
STRM_LEADS: Final = (250, 500, 1000)  # the STRM extrapolation leads (steps)
NAME_ZETA: Final = streamrec.NAME_ZETA  # a storm is named when its peak lower-layer vorticity reaches 0.5 f0
STORM_NAME_SALT: Final = 0x73746F726D6E616D  # "stormnam": the generated names of storms
DC_REF_FLOOR: Final = 1e-3
_DET_FIELDS: Final = (
    "t", "x", "y", "ux", "uy", "r_eq", "area", "hemi", "zeta_max_f0", "wind_max", "h1_min", "dh1", "rain",
)  # fmt: skip


def dc_reference(dc: np.ndarray) -> float:
    """The cloud-depth scale of ``MAP1`` (``play.weather.refs.dc``): the 99.9th percentile of ``D_c`` over
    the cells of a spun-up weather, or its maximum if that is zero, and never below ``DC_REF_FLOOR``. Computed
    once when a world is made and stored in ``play.json``, so every session of a world has the same scale."""
    a = np.asarray(dc, np.float64)
    q = float(np.percentile(a, 99.9))
    return max(q if q > 0.0 else float(a.max(initial=0.0)), DC_REF_FLOOR)


def _params_hash(p: WeatherParams) -> str:
    blob = json.dumps(p.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Note:
    """What the layer knows of an active track beyond ``Track``: the first detection, the peak
    ``zeta_max_f0`` over its whole life (the 40 stored detections cannot un-name a storm or reset its age),
    its name and the extent of its footprint."""

    t_first: int
    peak: float
    name: str | None
    ext: tuple[float, float]


class StormTick(NamedTuple):
    """One tick of the layer."""

    step: int
    strm: dict[str, Any]  # the STRM object
    planes: dict[str, np.ndarray] | None  # the quantised MAP1 planes (None if the maps were not read)
    events: list[dict[str, Any]]  # tracker events, the chronicle's input
    line: dict[str, Any] | None  # the line of observers/storms.jsonl (None: no storm and no event)


def _round(x: float, nd: int) -> float:
    return round(float(x), nd)


@dataclass
class StormLayer:
    """Detects, tracks and names the storms of a weather world, tick by tick (module docstring)."""

    p: WeatherParams
    dc_ref: float
    tick: int
    tracker: Tracker = field(init=False)
    notes: dict[int, Note] = field(default_factory=dict)
    step: int = 0  # the step of the last tick (the checkpoint step for a restored layer)
    restarted: bool = False  # opened at a step > 0 with no tracker state: ids begin again at 1
    initial_pending: bool = False  # the next tick finds storms that already existed (module docstring)
    # the detection record of the layer (a sphere's adds ``z``): persisted fields, their types, the class
    det_fields: ClassVar[tuple[str, ...]] = ()
    det_types: ClassVar[tuple[Callable[[Any], Any], ...]] = ()
    det_cls: ClassVar[type[Det]] = Det

    def __post_init__(self) -> None:
        if not (self.dc_ref > 0 and math.isfinite(self.dc_ref)):
            raise ValueError(f"dc_ref must be positive and finite, got {self.dc_ref}")
        if self.tick < 1:
            raise ValueError(f"tick must be >= 1, got {self.tick}")
        self.tracker = Tracker(self.p)

    # --- one tick -------------------------------------------------------------------------------------------

    def tick_at(self, ws: WeatherState, t: int, t_mult: mx.array | None = None) -> StormTick:
        """Read the weather of step ``t`` (one device sync), detect, track, name and build the records.

        ``t_mult`` is the engine's shade on ``T_s`` (``weather.shade_mult`` of the state at step ``t``, i.e.
        the next step's start-of-step state; None while ``shade_gain`` is 0). The ``MAP1`` plane is then the
        ``T_s`` that step ``t`` will force the weather with, ``warm_pools * t_mult``, not the bare pools."""
        p = self.p
        warm = self._sea(t, t_mult)
        fr, (dc, ts) = read_frame_extra(ws, p, t, (ws.Dc, warm))
        det = detect(fr, p)
        maps = {
            "p": fr.P,
            "dc": dc,
            "zeta": corners_to_centres(fr.zeta.astype(np.float64)),
            "ts": ts,
            "winds": np.stack([fr.u1, fr.v1, fr.u2, fr.v2]),
        }
        return self.ingest(det, maps)

    def _sea(self, t: int, t_mult: mx.array | None) -> mx.array:
        """``T_s`` of step ``t``: the warm pools, times the shade when the world has one (static)."""
        warm = W.warm_pools(self.p, mx.array(t, dtype=mx.int32))
        return warm if t_mult is None else warm * t_mult

    def current(self, ws: WeatherState, t: int, t_mult: mx.array | None = None) -> StormTick:
        """The records of step ``t`` without a detection: the planes read from ``ws`` and the ``STRM`` object
        of the tracks as the last tick left them (``detect_step`` is that tick), no event, no tracker change.
        The play loop sends it at start and on ``idnt``, when no tick has just run. ``t_mult``: as
        ``tick_at``."""
        p = self.p
        warm = self._sea(t, t_mult)
        fr, (dc, ts) = read_frame_extra(ws, p, t, (ws.Dc, warm))
        planes = self._planes(
            {
                "p": fr.P,
                "dc": dc,
                "zeta": corners_to_centres(fr.zeta.astype(np.float64)),
                "ts": ts,
                "winds": np.stack([fr.u1, fr.v1, fr.u2, fr.v2]),
            }
        )
        strm = self._strm(t, [])
        return StormTick(t, {**strm, "detect_step": self.step}, planes, [], None)

    def ingest(self, det: Detection, maps: Mapping[str, np.ndarray] | None = None) -> StormTick:
        """Fold one detection into the tracker and the notes. ``maps`` (the arrays of ``quantise_map1``) add
        the quantised planes to the result."""
        tr, t = self.tracker, det.t
        before = {x.id for x in tr.active()}
        first_new = tr.next_id
        ids = tr.update(det)
        extent = {tid: (c.ext_x, c.ext_y) for tid, c in zip(ids, det.storms, strict=True)}
        old = self.notes
        named_now: list[int] = []
        live_names = {n.name for n in old.values() if n.name}
        notes: dict[int, Note] = {}
        for x in sorted(tr.tracks, key=lambda x: x.id):
            if x.ended and x.id not in before:
                continue
            prev, last = old.get(x.id), x.dets[-1]
            peak = last.zeta_max_f0 if prev is None else max(prev.peak, last.zeta_max_f0)
            name = None if prev is None else prev.name
            if not x.ended and name is None and peak >= NAME_ZETA:
                name = self._name(x.id, live_names)
                live_names = live_names | {name}
                named_now.append(x.id)
            notes = {
                **notes,
                x.id: Note(
                    x.dets[0].t if prev is None else prev.t_first,
                    peak,
                    name,
                    extent.get(x.id) or (prev.ext if prev else (0.0, 0.0)),
                ),
            }
        events = self._events(t, first_new, before, named_now, notes)
        if self.initial_pending:  # the first tick of this layer: what it finds was there before
            events = [{**e, "initial": True} for e in events]
            self.initial_pending = False
        tr.compact(KEEP_DETS)
        self.notes = {i: n for i, n in notes.items() if i in {x.id for x in tr.tracks}}
        self.step = t
        planes = None if maps is None else self._planes(maps)
        return StormTick(t, self._strm(t, events), planes, events, self._line(t, ids, det, events))

    def _planes(self, maps: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        """The quantised weather map planes of ``maps`` (``MAP1``; a sphere's layer makes ``MAPH``'s)."""
        return streamrec.quantise_map1(
            p=maps["p"],
            dc=maps["dc"],
            zeta=maps["zeta"],
            ts=maps["ts"],
            winds=maps["winds"],
            dc_ref=self.dc_ref,
        )

    @staticmethod
    def _at(c: Any) -> dict[str, float]:
        """The place of a detection or component in a record: ``x`` and ``y`` (a sphere's adds ``z``)."""
        return {"x": _round(c.x, 2), "y": _round(c.y, 2)}

    def _name(self, ident: int, taken: set[str | None]) -> str:
        """The name of a storm: the house-name generator on ``splitmix64(id, STORM_NAME_SALT)``, with the id
        appended if a live storm already has it."""
        name = raw_house_names(np.array([ident]), STORM_NAME_SALT)[0]
        return name if name not in taken else f"{name} {ident}"

    def _events(
        self, t: int, first_new: int, before: set[int], named_now: Sequence[int], notes: Mapping[int, Note]
    ) -> list[dict[str, Any]]:
        """The tick's tracker events: births and splits, named, merged, dissolved, each group by id."""
        tracks = sorted(self.tracker.tracks, key=lambda x: x.id)

        def event(kind: str, x: Track, *, name: str | None, other: int = 0, age: int = 0) -> dict[str, Any]:
            last = x.dets[-1]
            return {
                "kind": kind,
                "step": t,
                "id": x.id,
                "name": name,
                "other": other,
                "other_name": notes[other].name if other in notes else None,
                **self._at(last),
                "hemi": x.hemi,
                "age": age,
            }

        out = [
            event("split" if x.parent else "born", x, name=None, other=x.parent)
            for x in tracks
            if x.id >= first_new
        ]
        out += [event("named", x, name=notes[x.id].name) for x in tracks if x.id in named_now]
        out += [
            event("merged", x, name=notes[x.id].name, other=x.merged_into)
            for x in tracks
            if x.ended and x.id in before and x.end_reason == "merged"
        ]
        out += [
            event("dissolved", x, name=notes[x.id].name, age=x.dets[-1].t - notes[x.id].t_first)
            for x in tracks
            if x.ended and x.id in before and x.end_reason == "dissipated"
        ]
        return out

    # --- the records ----------------------------------------------------------------------------------------

    def _storm(self, x: Track, t: int, points: int, pred: bool) -> dict[str, Any]:
        n, last = self.notes[x.id], x.dets[-1]
        length = self.p.L
        ahead = (
            [
                [lead, _round(px % length, 2), _round(py % length, 2)]
                for lead in STRM_LEADS
                for px, py in [self.tracker.predict(x, t + lead)]
            ]
            if pred and len(x.dets) >= 2
            else []
        )
        return {
            "id": x.id,
            "name": n.name,
            "named": n.name is not None,
            "hemi": x.hemi,
            "x": _round(last.x, 2),
            "y": _round(last.y, 2),
            "r": _round(last.r_eq, 2),
            "ext": [_round(n.ext[0], 2), _round(n.ext[1], 2)],
            "zeta": _round(n.peak, 4),
            "zeta_now": _round(last.zeta_max_f0, 4),
            "wind": _round(last.wind_max, 4),
            "rain": _round(last.rain, 6),
            "age": t - n.t_first,
            "parent": x.parent,
            "state": "active" if last.t == t else "coasting",
            "track": [[d.t, _round(d.ux, 2), _round(d.uy, 2)] for d in x.dets[-points:]],
            "pred": ahead,
        }

    def _strm(self, t: int, events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """The ``STRM`` object (§33.1.6): the active tracks, at most 32 (the highest peaks), shortened to fit
        the record's byte cap if need be."""
        active = sorted(self.tracker.active(), key=lambda x: (-self.notes[x.id].peak, x.id))[
            : streamrec.MAX_STORMS
        ]
        active = sorted(active, key=lambda x: x.id)
        ended = [
            {"id": e["id"], "reason": "merged" if e["kind"] == "merged" else "dissipated", "into": e["other"]}
            for e in events
            if e["kind"] in ("merged", "dissolved")
        ]
        for points, pred in ((TRACK_POINTS, True), (8, True), (4, False), (2, False)):
            obj = {
                "detect_step": t,
                "storms": [self._storm(x, t, points, pred) for x in active],
                "ended": ended,
            }
            if len(streamrec.json_bytes(obj)) <= streamrec.STRM_JSON_CAP:
                return obj
        return obj

    def _line(
        self, t: int, ids: Sequence[int], det: Detection, events: Sequence[Mapping[str, Any]]
    ) -> dict[str, Any] | None:
        storms = [
            {
                "id": tid,
                **self._at(c),
                "r": _round(c.r_eq, 2),
                "zeta": _round(c.zeta_max_f0, 4),
                "wind": _round(c.wind_max, 4),
                "rain": _round(c.rain, 6),
            }
            for tid, c in zip(ids, det.storms, strict=True)
        ]
        if not storms and not events:
            return None
        return {"step": t, "storms": storms, "events": list(events)}

    # --- persistence ----------------------------------------------------------------------------------------

    def state(self) -> dict[str, Any]:
        """The complete state, as of the last tick, as plain JSON values: ``next_id``, and per active track
        everything ``Tracker.update`` and the records read (``hemi``, ``parent``, ``missed``, the footprint,
        the last ``KEEP_DETS`` detections) and the notes (first detection, peak, name, extent)."""
        tracks = [
            {
                "id": x.id,
                "hemi": x.hemi,
                "parent": x.parent,
                "missed": x.missed,
                "t_first": self.notes[x.id].t_first,
                "peak": self.notes[x.id].peak,
                "name": self.notes[x.id].name,
                "ext": list(self.notes[x.id].ext),
                "last_flat": [] if x.last_flat is None else [int(v) for v in x.last_flat],
                "dets": [[getattr(d, f) for f in self.det_fields] for d in x.dets[-KEEP_DETS:]],
            }
            for x in sorted(self.tracker.active(), key=lambda x: x.id)
        ]
        return {
            "format": STORMS_FORMAT,
            "version": STORMS_VERSION,
            "params": _params_hash(self.p),
            "tick": self.tick,
            "step": self.step,
            "tick_step": self.step,
            "next_id": self.tracker.next_id,
            "tracks": tracks,
            **({"initial": True} if self.initial_pending else {}),
        }

    @classmethod
    def restore(cls, p: WeatherParams, dc_ref: float, state: Mapping[str, Any]) -> StormLayer:
        """The layer of ``state`` (``ValueError`` if it is not a state of this version and parameters)."""
        if state.get("format") != STORMS_FORMAT or state.get("version") != STORMS_VERSION:
            raise ValueError("not a storm tracker state of this version")
        if state.get("params") != _params_hash(p):
            raise ValueError("the tracker state was made with other weather parameters")
        layer = cls(p, dc_ref, int(state["tick"]))
        tracks, notes = [], {}
        for d in state["tracks"]:
            dets = [
                cls.det_cls(*(f(v) for f, v in zip(cls.det_types, row, strict=True))) for row in d["dets"]
            ]
            tracks.append(
                Track(
                    d["id"], d["hemi"], dets, parent=d["parent"], missed=d["missed"],
                    last_flat=np.array(d["last_flat"], dtype=np.int64),
                )
            )  # fmt: skip
            notes = {**notes, d["id"]: Note(d["t_first"], d["peak"], d["name"], (d["ext"][0], d["ext"][1]))}
        layer.tracker.tracks = tracks
        layer.tracker.next_id = int(state["next_id"])
        layer.initial_pending = bool(state.get("initial", False))
        layer.notes = notes
        layer.step = int(state.get("tick_step", state["step"]))  # the last tick (``step``: the checkpoint's)
        return layer

    @classmethod
    def open(
        cls, rd: Any, p: WeatherParams, dc_ref: float, step: int, tick: int | None = None
    ) -> tuple[StormLayer, str | None]:
        """The layer for a play that continues at ``step``: the tracker state saved with the checkpoint of
        ``step`` if it is usable, else a fresh layer (``restarted`` when ``step > 0``: the ids begin again at
        1) and the reason. ``rd`` is a ``RunDir``. Never an error."""
        tick = p.detect_every if tick is None else tick
        reason: str | None = None
        try:
            state = rd.read_storms_state(step)
            if state is None:
                reason = f"no tracker state at step {step}"
            elif state.get("step") != step or state.get("tick") != tick:
                reason = "the tracker state is for another step or tick"
            else:
                return cls.restore(p, dc_ref, state), None
        except (ValueError, KeyError, TypeError, IndexError) as e:
            reason = f"the tracker state at step {step} is unusable ({e})"
        layer = cls(p, dc_ref, tick)
        layer.step = step
        layer.restarted = step > 0
        layer.initial_pending = (
            layer.restarted
        )  # a restart finds storms that were there; ``fresh`` sets it too
        return layer, reason

    def save(self, rd: Any, step: int) -> None:
        """Write the state next to the checkpoint of ``step`` (call it before the state file is written)."""
        rd.write_storms_state(step, {**self.state(), "step": step})

    def log(self, rd: Any, tick: StormTick) -> None:
        """Append the tick's line to ``observers/storms.jsonl`` (nothing if it has none)."""
        if tick.line is not None:
            rd.append_storms([tick.line])


_DET_TYPES: Final[tuple[Callable[[Any], Any], ...]] = (
    int, float, float, float, float, float, int, int, float, float, float, float, float,
)  # fmt: skip


StormLayer.det_fields, StormLayer.det_types = _DET_FIELDS, _DET_TYPES
