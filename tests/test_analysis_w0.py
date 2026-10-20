"""The W0 study driver (``alife.analysis.w0``, spec §5 to §9a): labelling, detection, tracking, the
statistics, the tracer step and the occupancy binning, the experiments and the CLI.

CPU and fast. The host-side tests use synthetic footprints and tracks with hand-computable answers; the device
tests use small grids (N = 32 to 64) and the ``tiny3d`` hash geometry.
"""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from alife import spatial
from alife import weather as W
from alife.analysis import w0
from alife.config import dims, load_config
from alife.runio import RunDir
from alife.state import init_state
from alife.weather import START_SET, WeatherParams
from helpers import ON_GPU

P0 = START_SET  # the spec's starting set (spec 3.3); the defaults are the W0 result (weather.DEFAULT_POOLS)
SMALL = {
    "n": 32,
    "dx": 5.0,
    "f0": 0.012,
    "gamma": 16.0,
    "tau_c": 20.0,
    "pools": [
        [40, 40, 16, 8, 1, 0, 0],
        [120, 40, 16, 8, 1, 0, 0],
        [0, 120, 16, 8, 1, 0, 0],
        [80, 120, 16, 8, 1, 0, 0],
    ],
    "zeta_th": 0.0036,
    "a_min": 6,
    "r_min": 2,
}  # a 32 x 32 grid of 5-unit cells (L = 160, the world of the smallest valid 64k3d)


# --- 19. label_periodic --------------------------------------------------------------------------------


def _bfs_labels(mask: np.ndarray, group: np.ndarray | None = None) -> np.ndarray:
    """Reference: periodic 8-connected components by breadth-first search, labelled in raster order."""
    n, m = mask.shape
    out = np.zeros(mask.shape, dtype=np.int32)
    k = 0
    for j in range(n):
        for i in range(m):
            if not mask[j, i] or out[j, i]:
                continue
            k += 1
            out[j, i] = k
            stack = [(j, i)]
            while stack:
                a, b = stack.pop()
                for dj in (-1, 0, 1):
                    for di in (-1, 0, 1):
                        c, d = (a + dj) % n, (b + di) % m
                        if mask[c, d] and not out[c, d] and (group is None or group[c, d] == group[a, b]):
                            out[c, d] = k
                            stack.append((c, d))
    return out


def test_label_periodic_joins_seams_corners_and_diagonals() -> None:
    mask = np.zeros((10, 10), dtype=bool)
    mask[0, 0] = mask[9, 9] = True  # the four corners meet across both seams (a diagonal join)
    mask[0, 9] = mask[9, 0] = True
    mask[4, 4] = mask[5, 5] = True  # a plain diagonal
    mask[2, 7] = True  # separate
    lab, k = w0.label_periodic(mask)
    assert k == 3
    assert lab[0, 0] == lab[9, 9] == lab[0, 9] == lab[9, 0] == 1
    assert lab[2, 7] == 2  # components are numbered by their first cell in raster order
    assert lab[4, 4] == lab[5, 5] == 3
    assert (lab[~mask] == 0).all()
    again, _ = w0.label_periodic(mask)
    assert np.array_equal(lab, again)  # deterministic


@pytest.mark.parametrize("seed", range(4))
def test_label_periodic_matches_a_breadth_first_search(seed: int) -> None:
    rng = np.random.default_rng(seed)
    mask = rng.random((24, 20)) < 0.35
    lab, k = w0.label_periodic(mask)
    ref = _bfs_labels(mask)
    assert k == int(ref.max())
    assert np.array_equal(lab, ref)  # both number components by their first cell in raster order


def test_label_periodic_group_never_joins_across_groups() -> None:
    mask = np.ones((8, 6), dtype=bool)
    group = np.where(np.arange(8) < 4, 1, -1)[:, None] * np.ones((1, 6), dtype=int)
    lab, k = w0.label_periodic(mask, group=group)
    assert k == 2  # the two halves touch across both row seams but stay apart
    assert np.array_equal(lab, _bfs_labels(mask, group))
    assert w0.label_periodic(mask)[1] == 1


# --- detection (spec §6) -------------------------------------------------------------------------------


def _frame(p: WeatherParams, t: int = 100) -> w0.Frame:
    z = np.zeros((p.n, p.n), dtype=np.float32)
    return w0.Frame(
        t, z.copy(), z.copy(), z + p.H1, z + p.H2, z.copy(), z.copy(), z.copy(), z.copy(), z.copy()
    )


def _bump(p: WeatherParams, fr: w0.Frame, rows: range, cols: range, value: float = 0.02, rain: float = 1e-3):
    """Constant corner vorticity over a block (rows and columns wrap), with rain over the same cells."""
    z, pp = fr.zeta.copy(), fr.P.copy()
    for j in rows:
        for i in cols:
            z[j % p.n, i % p.n] = value
            pp[j % p.n, i % p.n] = rain
    return fr._replace(zeta=z, P=pp)


def test_detect_centroid_across_the_seam_and_hemisphere() -> None:
    p = replace(P0, n=40)  # L = 80; centre rows 0..19 are the north hemisphere
    fr = _bump(p, _frame(p), range(8, 14), range(37, 45))  # columns 37..44 straddle the x seam
    det = w0.detect(fr, p)
    assert len(det.storms) == 1 and not det.bands
    c = det.storms[0]
    assert c.hemi == 0
    # the block of corners 37..44 gives centres 38..44 (corners on both sides): centre x = (i + 0.5) dx
    expected_x = ((41.0 + 0.5) * p.dx) % p.L  # the middle of the 38..44 run, wrapping 40 -> 0
    assert abs(min(abs(c.x - expected_x), p.L - abs(c.x - expected_x))) < 1.5
    assert 8 * p.dx < c.y < 14 * p.dx
    assert c.rx > 0.9 and c.ry > 0.9
    assert c.r_eq == pytest.approx(p.dx * math.sqrt(c.area / math.pi))
    # the same bump in the south (rows 20..39) is cyclonic with the opposite sign of zeta
    fr_s = _bump(p, _frame(p), range(28, 34), range(10, 16), value=-0.02)
    ds = w0.detect(fr_s, p)
    assert len(ds.storms) == 1 and ds.storms[0].hemi == 1
    assert w0.detect(_bump(p, _frame(p), range(28, 34), range(10, 16), value=0.02), p).storms == []


def test_detect_needs_rain_and_area() -> None:
    p = replace(P0, n=40)
    dry = _bump(p, _frame(p), range(8, 14), range(10, 16), rain=0.0)
    d = w0.detect(dry, p)
    assert d.storms == [] and d.n_weak == 1  # no rain: a vorticity region, not a storm
    tiny = _bump(p, _frame(p), range(8, 10), range(10, 12))
    assert w0.detect(tiny, p).storms == []  # below a_min


def test_detect_classifies_zonal_strips_and_long_regions_as_bands() -> None:
    p = replace(P0, n=40)
    strip = _bump(p, _frame(p), range(8, 11), range(0, 40))  # a full zonal ring: its x centroid is undefined
    d = w0.detect(strip, p)
    assert d.storms == [] and len(d.bands) == 1
    assert d.bands[0].rx < 0.5
    # 23 of 40 columns: the circular resultant is still above 0.5 but the extent exceeds L/2
    wide = _bump(p, _frame(p), range(8, 12), range(0, 23))
    d = w0.detect(wide, p)
    assert d.storms == [] and len(d.bands) == 1
    assert d.bands[0].rx >= 0.5 and d.bands[0].ext_x > p.L / 2
    # a compact storm and a band at once: the storm still counts
    both = _bump(p, strip, range(15, 19), range(30, 35))
    d = w0.detect(both, p)
    assert len(d.storms) == 1 and len(d.bands) == 1


def test_detect_splits_regions_at_the_equator() -> None:
    p = replace(P0, n=40, a_min=6)
    z = np.zeros((p.n, p.n), dtype=np.float32)
    z[17:19, 10:16] = 0.02  # north side, cyclonic (positive); centre rows 18 and 19
    z[20:22, 10:16] = -0.02  # south side, cyclonic (negative); centre rows 20 and 21
    fr = _frame(p)._replace(zeta=z, P=np.full_like(z, 1e-3))
    # centre rows 19 (north) and 20 (south) touch across the equator, and both are in the mask
    zc = w0.corners_to_centres(z.astype(np.float64))
    mask = (w0.hemisphere_sign(p) * zc) >= p.zeta_th
    assert mask[19].any() and mask[20].any() and w0.label_periodic(mask)[1] == 1  # one blob without the split
    det = w0.detect(fr, p)
    assert det.n_regions == 2 and sorted(c.hemi for c in det.storms) == [0, 1]
    north, south = sorted(det.storms, key=lambda c: c.hemi)
    assert north.y < p.L / 2 < south.y


def test_detect_on_a_balanced_vortex_reads_its_core_values() -> None:
    """The balanced vortex of spec 3.1 (zeta 0.55 f0, sigma 9, peak wind 0.056) on the north pole row."""
    p = replace(P0, n=64, gamma=0.0, k_E=0.0, tau_rad=0.0, seed_rate=0.0, p_th=0.0, r_min=1, a_min=12)
    fr = w0.read_frame(w0.balanced_vortex(p), p, 0)
    det = w0.detect(fr, p)
    assert len(det.storms) == 1 and not det.bands
    c = det.storms[0]
    assert (c.x, c.y) == pytest.approx((p.L / 2, p.L / 4), abs=0.05)  # the circular centroid of the footprint
    uc, vc = w0.centre_winds(fr.u1, fr.v1)
    assert c.wind_max == pytest.approx(
        float(np.hypot(uc, vc).max()), rel=2e-3
    )  # the maximum sits in 1.5 r_eq
    assert 0.5 < c.zeta_max_f0 < 0.55 and 0.04 < c.wind_max < 0.06
    assert c.h1_min == pytest.approx(float(fr.h1.min())) and 1.0 < c.dh1 < 5.0
    assert c.r_eq == pytest.approx(p.dx * math.sqrt(c.area / math.pi)) and c.r_max >= c.r_eq * 0.9
    assert c.rx > 0.97 and c.ry > 0.97 and c.n_rain == c.area  # p_th = 0: every cell counts as rainy


def _block_frame(p: WeatherParams) -> tuple[w0.Frame, np.ndarray]:
    """An 8 x 8 block of corner vorticity rising to the east (0.01 to 0.08), with rain over and around it."""
    fr = _frame(p)
    z, pp = fr.zeta.copy(), fr.P.copy()
    for c in range(6, 14):
        z[6:14, c] = 0.01 * (c - 5)
    pp[5:16, 5:16] = 1e-3
    return fr._replace(zeta=z, P=pp), z


def test_detect_centroid_is_weighted_by_the_excess_vorticity() -> None:
    p = replace(P0, n=40)
    fr, z = _block_frame(p)
    (c,) = w0.detect(fr, p).storms
    zc = w0.corners_to_centres(z.astype(np.float64))
    jj, ii = np.nonzero(zc >= p.zeta_th)
    ang = 2.0 * math.pi * (ii + 0.5) * p.dx / p.L

    def circ(w: np.ndarray) -> float:
        return float(math.atan2((w * np.sin(ang)).sum(), (w * np.cos(ang)).sum()) * p.L / (2.0 * math.pi))

    x_weighted, x_plain = circ(zc[jj, ii] - p.zeta_th), circ(np.ones(ii.size))
    assert abs(x_weighted - x_plain) > 0.3  # the weights matter in this fixture
    assert c.x == pytest.approx(x_weighted, abs=1e-6)


def test_detect_dh1_is_the_ring_between_one_and_two_equivalent_radii_above_the_footprint_minimum() -> None:
    p = replace(P0, n=40)
    fr, _ = _block_frame(p)
    (c,) = w0.detect(fr, p).storms
    j, i = np.mgrid[0 : p.n, 0 : p.n]
    d = np.hypot((i + 0.5) * p.dx - c.x, (j + 0.5) * p.dx - c.y)
    h1 = np.full((p.n, p.n), 32.0, dtype=np.float32)
    h1[c.flat // p.n, c.flat % p.n] = 31.0
    h1[(d > c.r_eq) & (d <= 2 * c.r_eq)] = 30.0  # the ring (footprint cells beyond r_eq included)
    h1[(d > 2 * c.r_eq) & (d <= 3 * c.r_eq)] = 10.0  # beyond it: must not count
    core = c.flat[d.ravel()[c.flat] <= c.r_eq][0]
    h1[core // p.n, core % p.n] = 28.0  # the footprint minimum, inside r_eq
    (c2,) = w0.detect(fr._replace(h1=h1), p).storms
    assert c2.h1_min == 28.0 and c2.dh1 == pytest.approx(30.0 - 28.0, abs=1e-5)


# --- 21. the tracker -----------------------------------------------------------------------------------

N40 = 40
P40 = replace(P0, n=N40)  # L = 80, miss_grace 2, min_dets 3


def _comp(label: int, rect: tuple[int, int, int, int], hemi: int = 0, zeta: float = 1.0) -> w0.Comp:
    j0, i0, h, w = rect
    jj, ii = np.meshgrid(np.arange(j0, j0 + h), np.arange(i0, i0 + w), indexing="ij")
    area = h * w
    return w0.Comp(
        label=label,
        hemi=hemi,
        x=(ii.mean() + 0.5) * P40.dx,
        y=(jj.mean() + 0.5) * P40.dx,
        area=area,
        r_eq=P40.dx * math.sqrt(area / math.pi),
        r_max=3.0,
        zeta_max_f0=zeta,
        wind_max=0.1,
        h1_min=30.0,
        dh1=1.0,
        rain=1.0,
        n_rain=5,
        rx=1.0,
        ry=1.0,
        ext_x=w * P40.dx,
        ext_y=h * P40.dx,
        flat=(jj * N40 + ii).ravel(),
    )


def _det(
    t: int, rects: list[tuple[int, int, int, int]], zetas: Sequence[float] | None = None
) -> w0.Detection:
    lab = np.zeros((N40, N40), dtype=np.int32)
    comps = []
    for k, r in enumerate(rects, start=1):
        c = _comp(k, r, zeta=1.0 if zetas is None else zetas[k - 1])
        lab.ravel()[c.flat] = k
        comps.append(c)
    return w0.Detection(t, lab, comps, [], len(comps), 0)


def _run(tracker: w0.Tracker, frames: dict[int, list[tuple[int, int, int, int]]]) -> dict[int, list[int]]:
    return {
        t: tracker.update(_det(t, frames.get(t, []))) for t in range(25, 25 * (max(frames) // 25) + 1, 25)
    }


def test_tracker_births_continuation_and_ids() -> None:
    tr = w0.Tracker(P40)
    ids = _run(tr, {25: [(10, 10, 4, 4)], 50: [(10, 11, 4, 4)], 75: [(10, 12, 4, 4), (25, 30, 4, 4)]})
    assert ids[25] == [1] and ids[50] == [1] and ids[75] == [1, 2]
    tracks = tr.finish()
    assert [len(t.dets) for t in tracks] == [3, 1]
    assert tracks[0].parent == 0 and tracks[1].parent == 0 and not tracks[0].ended


def test_tracker_merger_ends_the_smaller_track() -> None:
    tr = w0.Tracker(P40)
    a, b = (10, 10, 4, 6), (10, 18, 4, 4)
    _run(tr, {25: [a, b], 50: [a, b], 75: [a, b]})
    merged = (10, 11, 4, 11)  # overlaps a by 5 columns and b by 4
    ids = tr.update(_det(100, [merged]))
    assert ids == [1]  # the larger overlap (track 1, the bigger footprint) continues
    t1, t2 = tr.finish()
    assert t2.end_reason == "merged" and t2.merged_into == 1 and t2.t_last == 75
    assert not t1.ended and t1.t_last == 100


def test_tracker_split_starts_a_child_with_its_parent() -> None:
    tr = w0.Tracker(P40)
    big = (10, 10, 4, 12)
    _run(tr, {25: [big], 50: [big], 75: [big]})
    ids = tr.update(
        _det(100, [(10, 10, 4, 8), (10, 20, 4, 4)])
    )  # the left part continues, the right splits off
    assert ids[0] == 1 and ids[1] == 2
    t1, t2 = tr.finish()
    assert t2.parent == 1 and t2.t_first == 100 and not t1.ended


def _split_pair(zetas: tuple[float, float], right: tuple[int, int, int, int]) -> list[int]:
    """Track 1 holds a 19 x 10 footprint; it splits into a left piece (19 x 4, overlap 76) and ``right``."""
    tr = w0.Tracker(P40)
    big = (1, 10, 19, 10)
    _run(tr, {25: [big], 50: [big], 75: [big]})
    ids = tr.update(_det(100, [(1, 10, 19, 4), right], zetas=zetas))
    assert sorted(ids) == [1, 2]
    return ids


def test_tracker_split_near_tie_keeps_the_id_with_the_stronger_core() -> None:
    """The right piece overlaps 72 cells against 76 (within 10%): the stronger of the two keeps the id, so a
    dying fragment with a few more overlapping cells does not take the storm's identity."""
    right = (2, 15, 18, 4)  # 18 x 4 = 72 cells
    assert _split_pair((0.3, 0.6), right) == [2, 1]  # the right piece is the core: it continues track 1
    assert _split_pair((0.6, 0.3), right) == [1, 2]  # the left piece is both larger and stronger
    tr = w0.Tracker(P40)
    big = (1, 10, 19, 10)
    _run(tr, {25: [big], 50: [big], 75: [big]})
    tr.update(_det(100, [(1, 10, 19, 4), right], zetas=(0.3, 0.6)))
    t1, t2 = tr.finish()
    assert t2.parent == 1 and t2.t_first == 100 and t1.t_last == 100 and not t1.ended


def test_tracker_split_without_a_tie_keeps_the_largest_overlap_whatever_the_vorticity() -> None:
    right = (10, 15, 10, 4)  # 40 cells against 76: not within 10%
    assert _split_pair((0.3, 0.9), right) == [1, 2]
    assert _split_pair((0.9, 0.3), right) == [1, 2]


def test_tracker_merger_near_tie_between_pieces_still_follows_the_larger_overlap() -> None:
    """Two tracks overlap one component: the track with the larger overlap continues, vorticity plays no part
    (it is the component's, the same for both)."""
    tr = w0.Tracker(P40)
    a, b = (10, 10, 4, 6), (10, 18, 4, 4)
    _run(tr, {25: [a, b], 50: [a, b], 75: [a, b]})
    ids = tr.update(_det(100, [(10, 11, 4, 11)], zetas=(0.2,)))
    assert ids == [1]


def test_tracker_one_detection_gap_continues_and_counts() -> None:
    tr = w0.Tracker(P40)
    r = (10, 10, 4, 4)
    ids = _run(tr, {25: [r], 50: [r], 75: [], 100: [r]})
    assert ids[100] == [1]  # the same track after a one-detection gap
    (track,) = tr.finish()
    assert [d.t for d in track.dets] == [25, 50, 100] and not track.ended
    c = w0.storm_counts([track], [25, 50, 75, 100], min_dets=3, lo=1, hi=6)
    assert list(c["n_north"]) == [1, 1, 1, 1]  # the coasting time 75 counts: the gap ends in a re-detection


def test_tracker_three_detection_gap_ends_the_track_and_its_tail_is_not_counted() -> None:
    tr = w0.Tracker(P40)
    r = (10, 10, 4, 4)
    frames: dict[int, list[tuple[int, int, int, int]]] = {
        25: [r],
        50: [r],
        75: [r],
        175: [r],
        200: [r],
        225: [r],
    }
    ids = _run(tr, frames)
    assert ids[175] == [2]  # a new track: the first one ended after three missed detections
    t1, t2 = tr.finish()
    assert t1.end_reason == "dissipated" and t1.t_last == 75
    assert t2.t_first == 175 and not t2.ended
    c = w0.storm_counts([t1, t2], [75, 100, 125, 150, 175], min_dets=3, lo=1, hi=6)
    assert list(c["n_north"]) == [1, 0, 0, 0, 1]  # the coasting tail of track 1 never counts


def test_tracker_two_detection_gap_still_continues() -> None:
    """``miss_grace`` 2 tolerates two consecutive misses: only the third ends a track (the tests above pin the
    one- and three-detection gaps; this one pins the boundary)."""
    tr = w0.Tracker(P40)
    r = (10, 10, 4, 4)
    ids = _run(tr, {25: [r], 50: [r], 75: [r], 150: [r]})
    assert ids[150] == [1]
    (track,) = tr.finish()
    assert [d.t for d in track.dets] == [25, 50, 75, 150] and not track.ended


def test_tracker_unwraps_positions_across_the_seam() -> None:
    tr = w0.Tracker(P40)  # L = 80, 40 columns of 2 units
    _run(tr, {25: [(10, 36, 4, 3)], 50: [(10, 38, 4, 3)], 75: [(10, 40 - 1, 4, 3)], 100: [(10, 0, 4, 3)]})
    ids = tr.update(_det(125, [(10, 2, 4, 3)]))
    assert ids == [1]
    (track,) = tr.finish()
    assert len(track.dets) == 5
    ux = np.array([d.ux for d in track.dets])
    assert np.all(np.diff(ux) > 0) and np.all(np.diff(ux) < 5.0)  # continuous, no jump of L
    assert track.dets[-1].ux > P40.L and track.dets[-1].x < P40.L  # unwrapped past the seam, stored wrapped


def test_tracker_follows_a_fast_storm_by_predicted_position() -> None:
    tr = w0.Tracker(P40)
    # 8 x 8 footprints (r_eq 9 units): 4 columns per detection, then a jump of 8 that misses the last
    # footprint; the least-squares velocity predicts 4 of them, and the 8-unit miss is inside the gate
    cols = {25: 2, 50: 6, 75: 10, 100: 18}
    ids = _run(tr, {t: [(10, c, 8, 8)] for t, c in cols.items()})
    assert [ids[t] for t in sorted(ids)] == [[1], [1], [1], [1]]
    (track,) = tr.finish()
    assert [d.t for d in track.dets] == [25, 50, 75, 100]
    # a jump of 20 columns is outside the gate: a new storm
    tr2 = w0.Tracker(P40)
    ids2 = _run(tr2, {25: [(10, 2, 8, 8)], 50: [(10, 6, 8, 8)], 75: [(10, 10, 8, 8)], 100: [(10, 30, 8, 8)]})
    assert ids2[100] == [2]


def test_tracker_fallback_never_crosses_hemispheres() -> None:
    tr = w0.Tracker(P40)
    for t, col in ((25, 2), (50, 6), (75, 10)):
        tr.update(_det(t, [(10, col, 8, 8)]))
    south = _det(100, [(10, 18, 8, 8)])  # where the prediction puts the storm, but in the other hemisphere
    south.storms[0] = w0.Comp(**{**south.storms[0].__dict__, "hemi": 1})
    assert tr.update(south) == [2]  # a different hemisphere is a different storm
    assert tr.finish()[0].missed == 1


def test_tracker_fallback_gate_is_the_larger_of_the_two_radii() -> None:
    """A small storm and a much larger one 8 units away, with no cell in common, are the same storm: the gate
    is max(r_track, r_det) (8.7 here), not the smaller radius (2.3)."""
    tr = w0.Tracker(P40)
    for t in (25, 50, 75):
        tr.update(_det(t, [(10, 10, 2, 2)]))  # r_eq 2.3
    big = (6, 12, 10, 6)  # 60 cells, r_eq 8.7, centroid 8 units east of the small storm's
    assert tr.update(_det(100, [big])) == [1]
    (track,) = tr.finish()
    assert len(track.dets) == 4 and track.dets[-1].r_eq > 8.0
    far = w0.Tracker(P40)
    for t in (25, 50, 75):
        far.update(_det(t, [(10, 10, 2, 2)]))
    assert far.update(_det(100, [(6, 20, 10, 6)])) == [2]  # 24 units away: outside even the larger radius


# --- storm counts, 22. Kaplan-Meier, lifetimes ---------------------------------------------------------


def _track(tid: int, times, xs, ys, r=5.0, hemi: int = 0, end: str = "") -> w0.Track:
    """A track at the given positions; ``r`` is one radius or one per detection."""
    rs = np.broadcast_to(np.asarray(r, dtype=float), (len(times),))
    dets = [
        w0.Det(
            int(t),
            float(x) % 320,
            float(y) % 320,
            float(x),
            float(y),
            float(rr),
            80,
            hemi,
            1.0,
            0.1,
            30.0,
            1.0,
            1.0,
        )
        for t, x, y, rr in zip(times, xs, ys, rs, strict=True)
    ]
    return w0.Track(tid, hemi, dets, end_reason=end)


def test_storm_counts_by_hemisphere_and_range() -> None:
    a = _track(1, range(0, 500, 25), [10] * 20, [40] * 20, hemi=0)
    b = _track(2, range(100, 500, 25), [50] * 16, [40] * 16, hemi=0)
    c = _track(3, range(0, 500, 25), [90] * 20, [250] * 20, hemi=1)
    d = _track(4, [0, 25], [1, 1], [1, 1], hemi=0)  # a transient (2 detections): never counted
    times = list(range(0, 500, 25))
    cnt = w0.storm_counts([a, b, c, d], times, min_dets=3, lo=1, hi=2)
    n, s = np.array(cnt["n_north"]), np.array(cnt["n_south"])
    assert n[0] == 1 and n[-1] == 2 and (s == 1).all()
    assert cnt["all_in_range"] and cnt["in_range_fraction"] == 1.0
    assert cnt["north"]["histogram"] == {1: 4, 2: 16}
    cnt2 = w0.storm_counts([a, b, c], times, min_dets=3)  # default 2..6: the south has one storm
    assert not cnt2["all_in_range"] and cnt2["south"]["in_range_fraction"] == 0.0


def test_kaplan_meier_hand_example_and_delayed_entry() -> None:
    km = w0.kaplan_meier([2, 3, 3, 5, 7], [1, 1, 0, 1, 0])
    assert list(km.ages) == [2.0, 3.0, 5.0]
    assert np.allclose(km.survival, [0.8, 0.6, 0.3]) and km.median == 5.0
    # delayed entry (left truncation): C and D enter at age 3 and are not at risk before it
    km = w0.kaplan_meier([2, 4, 5, 8], [1, 1, 1, 0], [0, 0, 3, 3])
    assert list(km.ages) == [2.0, 4.0, 5.0]
    assert np.allclose(km.survival, [0.5, 1.0 / 3.0, 1.0 / 6.0]) and km.median == 2.0
    plain = w0.kaplan_meier([2, 4, 5, 8], [1, 1, 1, 0])  # ignoring the entry gives different numbers
    assert np.allclose(plain.survival, [0.75, 0.5, 0.25]) and plain.median == 4.0


def test_kaplan_meier_median_beyond_the_longest_age() -> None:
    km = w0.kaplan_meier([10, 20, 30, 40], [1, 0, 0, 0])
    assert km.median is None and km.max_age == 40.0 and "max observed age" in km.median_text()


def test_lifetime_stats_truncation_and_the_beyond_max_age_pass() -> None:
    long_ = _track(1, range(0, 3500, 25), [10] * 140, [40] * 140)  # alive at the end, 3475 steps
    long2 = _track(2, range(100, 3000, 25), [20] * 116, [40] * 116)  # alive, 2900 steps
    other = _track(3, range(500, 1725, 25), [30] * 49, [40] * 49, end="dissipated")  # one event at age 1200
    died = _track(4, range(0, 300, 25), [40] * 12, [40] * 12, end="dissipated")  # ended before the window
    st = w0.lifetime_stats([long_, long2, other, died], t_spin=1000, min_dets=3)
    assert st["n"] == 3 and st["events"] == 1 and st["censored"] == 2  # the early death is not in the window
    # at age 1200 all three are at risk (the two long tracks entered at 1000 and 900): S = 2/3 stays above 0.5
    assert st["median"] is None and st["max_age"] == 3475 and st["pass"]
    assert st["naive_median_completed"] == 1200.0
    short = _track(5, range(4000, 4100, 25), [50] * 4, [50] * 4, end="dissipated")
    only_short = w0.lifetime_stats([short, died], t_spin=1000, min_dets=3)
    assert only_short["median"] == 75.0 and not only_short["pass"]


def test_kaplan_meier_counts_no_event_at_the_entry_age() -> None:
    """A subject whose event is at its own entry age was never at risk: counting it would give 1/3 here and a
    negative curve when such events outnumber the risk set."""
    km = w0.kaplan_meier([1000, 1000, 2000, 3000], [1, 1, 1, 0], [1000, 0, 0, 0])
    assert km.n_risk[0] == 3 and km.n_events[0] == 1 and km.survival[0] == pytest.approx(2.0 / 3.0)
    km = w0.kaplan_meier([5, 5, 5, 10], [1, 1, 1, 0], [5, 5, 0, 0])
    assert km.survival[0] == pytest.approx(0.5) and km.n_events[0] == 1  # not 1 - 3/2


def test_kaplan_meier_risk_set_flags_and_helpers() -> None:
    km = w0.kaplan_meier([10, 20, 30, 40, 50, 60, 70], [1, 1, 1, 1, 0, 0, 0])
    assert list(km.n_risk) == [7, 6, 5, 4]
    assert km.thin_age == 40.0 and km.min_risk == w0.KM_MIN_RISK == 5
    assert km.median == 40.0 and km.n_risk_at_median == 4 and not km.median_reliable  # at the first thin age
    assert km.reached == 40.0
    assert km.at_risk_at(25.0) == 5 and km.at_risk_at(40.0) == 4 and km.at_risk_at(71.0) == 0
    assert km.survival_before(10.0) == 1.0 and km.survival_before(11.0) == pytest.approx(6 / 7)
    assert km.survival_before(40.0) == pytest.approx(4 / 7)
    ok = w0.kaplan_meier([10, 20, 30, 40, 50, 60, 70, 80, 90], [1, 1, 1, 1, 1, 0, 0, 0, 0])
    assert list(ok.n_risk) == [9, 8, 7, 6, 5] and ok.median == 50.0
    assert ok.thin_age is None and ok.median_reliable and ok.n_risk_at_median == 5
    few = w0.kaplan_meier([10, 20, 30], [1, 0, 0])  # three subjects: always a thin risk set, no median
    assert few.median is None and few.median_reliable and few.n_risk_at_median is None
    assert few.reached == 30.0 and few.thin_age == 10.0
    s = ok.summary()
    assert s["n"] == 9 and s["events"] == 5 and s["censored"] == 4 and s["median_reliable"] is True
    empty = w0.kaplan_meier([], [], [])
    assert empty.median is None and empty.summary()["n"] == 0 and empty.survival_before(5.0) == 1.0


def test_lifetime_stats_applies_the_entry_age_of_storms_born_in_the_spin_up() -> None:
    """A long storm born at t = 0 is not at risk at the young ages of the storms born in the window (its entry
    age is 3000); ignoring that entry gives a median of 200 instead of 100."""
    a = _track(1, range(3000, 3125, 25), [10] * 5, [40] * 5, end="dissipated")  # age 100
    b = _track(2, range(3200, 3425, 25), [20] * 9, [40] * 9, end="dissipated")  # age 200
    long_ = _track(3, range(0, 5025, 25), [30] * 201, [40] * 201)  # alive, age 5000 at the end
    st = w0.lifetime_stats([a, b, long_], t_spin=3000, min_dets=3)
    assert st["median"] == 100.0 and st["n_risk_at_median"] == 2
    assert w0.kaplan_meier([100, 200, 5000], [1, 1, 0]).median == 200.0  # what ignoring the entry gives
    assert st["thin_age"] == 100.0 and not st["median_reliable"]
    assert st["reading"].startswith("NOT READABLE") and not st["pass"]


def _fragments_and_storms() -> tuple[list[w0.Track], int]:
    """The start-set pathology: four storms born in the spin-up that live to the end, and 21 split-born
    fragments of 75 to 575 steps that are the only storms at risk at the young ages."""
    t_spin = 1000
    tracks = []
    for k, (x, y, h) in enumerate(((10, 40, 0), (50, 40, 0), (10, 250, 1), (50, 250, 1))):
        tracks.append(_track(k + 1, range(0, 30025, 25), [x] * 1201, [y] * 1201, hemi=h))
    for k in range(21):
        dur = 75 + 25 * k
        first = 2000 + 100 * k
        tr = _track(
            10 + k, range(first, first + dur + 25, 25), [90] * (dur // 25 + 1), [40] * (dur // 25 + 1)
        )
        tr.end_reason, tr.parent = "dissipated", 1
        tracks.append(tr)
    return tracks, t_spin


def test_lifetime_median_is_flagged_and_companions_show_the_long_storms() -> None:
    tracks, t_spin = _fragments_and_storms()
    st = w0.lifetime_stats(tracks, t_spin=t_spin, min_dets=3)
    assert st["n"] == 25 and st["events"] == 21 and st["censored"] == 4
    assert st["median"] == 325.0 and not st["pass"]  # the fragments decide the median, as written
    assert st["n_risk_at_median"] == 11 and st["median_reliable"] and st["thin_age"] == 500.0
    assert st["reading"] == "median 325 read with 11 at risk"
    # the criterion age: only the four old storms are at risk, and the curve has fallen to zero before it
    assert st["n_risk_at_threshold"] == 4 and st["survival_before_threshold"] == pytest.approx(0.0)
    cm = st["companions"]
    frag_dets = sum(4 + k for k in range(21))
    long_dets = 4 * ((30000 - t_spin) // 25 + 1)
    assert cm["detections"] == frag_dets + long_dets and cm["detections_in_long_storms"] == long_dets
    assert cm["share_detections_in_long_storms"] == pytest.approx(long_dets / (frag_dets + long_dets))
    assert cm["share_detections_in_long_storms"] > 0.9 and cm["detection_weighted_pass"] is True
    assert cm["without_split_born"]["n"] == 4 and cm["without_split_born"]["median"] is None
    assert cm["without_split_born"]["reached"] == 30000.0
    assert cm["split_born"] == {"n": 21, "events": 21, "median_duration": 325.0}
    # over full histories the four old storms are at risk at the young ages too, which moves the median
    assert cm["full_histories"]["n"] == 25 and cm["full_histories"]["median"] == 375.0
    bh = st["by_hemisphere"]
    assert bh["north"]["n"] == 2 + 21 and bh["south"]["n"] == 2 and bh["south"]["median"] is None
    assert bh["south"]["reached"] == 30000.0


def test_lifetime_criterion_is_decided_by_the_curve_before_the_threshold() -> None:
    """median >= 3000 is the same as survival > 0.5 before 3000: a single late death, the only storm at risk
    at 6000, gives a median that is flagged unreadable and does not change the verdict."""
    tracks = [_track(k, range(0, 4025, 25), [10 * k] * 161, [40] * 161) for k in range(1, 5)]
    tracks += [_track(9, range(0, 6025, 25), [90] * 241, [40] * 241, end="dissipated")]
    st = w0.lifetime_stats(tracks, t_spin=100, min_dets=3)
    assert st["median"] == 6000.0 and st["n_risk_at_median"] == 1 and not st["median_reliable"]
    assert st["pass"] and st["survival_before_threshold"] == 1.0 and st["n_risk_at_threshold"] == 5


# --- 23. forecast error --------------------------------------------------------------------------------


def _fc(tracks, t_spin=1000, t_end=2000, **kw):
    return w0.forecast_stats(tracks, P0, t_spin, t_end, 3, **kw)


def test_forecast_constant_velocity_has_no_error_and_persistence_does() -> None:
    t = np.arange(0, 2025, 25)
    tr = _track(1, t, 0.01 * t, 100 + 0.0 * t)
    f = _fc([tr])["leads"]
    assert f[1000]["n"] == 1 and f[1000]["median_error"] == pytest.approx(0.0, abs=1e-9)
    assert f[1000]["median_persistence"] == pytest.approx(10.0 / 5.0)  # 1000 steps at 0.01 over r_eq 5
    assert f[250]["n"] >= 3 and f[250]["median_error"] == pytest.approx(0.0, abs=1e-9)
    assert f[1000]["skill"] == pytest.approx(1.0)


def test_forecast_after_a_velocity_change_is_the_hand_value() -> None:
    t = np.arange(0, 2025, 25)
    x = np.where(t <= 1000, 0.01 * t, 10.0 + 0.02 * (t - 1000))
    f = _fc([_track(1, t, x, 100 + 0.0 * t)])["leads"][1000]
    # origin 1000: the fit sees speed 0.01 and forecasts x = 20; the track is at 30: 10 units = 2 radii
    assert f["n"] == 1 and f["median_error"] == pytest.approx(2.0, rel=1e-6)
    assert f["median_persistence"] == pytest.approx(4.0, rel=1e-6)  # it moved 20 units
    assert f["skill"] == pytest.approx(0.5, rel=1e-6)


def test_forecast_has_no_origin_at_a_coasting_time_or_with_a_short_history() -> None:
    t = np.array([x for x in np.arange(0, 2025, 25) if x != 1000])  # a one-detection gap at the origin
    tr = _track(1, t, 0.01 * t, 100 + 0.0 * t)
    f = _fc([tr])["leads"][1000]
    assert f["n"] == 0 and f["drops"].get("coasting at the origin") == 1
    t2 = np.array([x for x in np.arange(0, 2025, 25) if x not in (800, 825, 850)])  # 8 of 11 slots at 1000
    f2 = _fc([_track(1, t2, 0.01 * t2, 100 + 0.0 * t2)])["leads"][1000]
    assert f2["n"] == 0 and f2["drops"].get("short history") == 1
    t3 = np.arange(0, 1500, 25)  # ends before the target
    f3 = _fc([_track(1, t3, 0.01 * t3, 100 + 0.0 * t3, end="dissipated")])["leads"][1000]
    assert f3["n"] == 0 and f3["drops"].get("track dissipated") == 1


def test_forecast_jitter_floor_matches_the_analytic_value_for_a_stationary_track() -> None:
    rng = np.random.default_rng(3)
    sigma, r = 1.0, 10.0
    t = np.arange(0, 2525, 25)
    tracks = [
        _track(
            k, t, 100 + sigma * rng.standard_normal(t.size), 100 + sigma * rng.standard_normal(t.size), r=r
        )
        for k in range(200)
    ]
    f = w0.forecast_stats(tracks, P0, 750, 2500, 3)["leads"][1000]
    assert f["n"] > 500
    expected = math.sqrt(2.0 * math.log(2.0)) * sigma * math.sqrt(19.5) / r  # 5.20 sigma / r
    assert f["median_floor"] == pytest.approx(expected, rel=0.05)
    assert f["median_error"] == pytest.approx(expected, rel=0.08)  # pure jitter produces the floor
    assert f["skill"] < -1.5  # the extrapolation is worse than persistence (the analytic value is -2.1)


def test_forecast_unwraps_across_the_seam() -> None:
    t = np.arange(0, 2025, 25)
    x = 300 + 0.04 * t  # crosses x = 320 at t = 500
    f = _fc([_track(1, t, x, 100 + 0.0 * t)])["leads"][1000]
    assert f["median_error"] == pytest.approx(0.0, abs=1e-6)


def test_forecast_error_is_in_radii_of_the_storm_at_the_origin() -> None:
    t = np.arange(0, 2025, 25)
    x = 0.01 * t + np.where(t == 2000, 10.0, 0.0)  # 10 units off the line at the target
    r = np.where(t <= 1000, 5.0, 10.0)  # the storm doubles its radius by the target
    f = _fc([_track(1, t, x, 100 + 0.0 * t, r=r)])["leads"][1000]
    assert f["n"] == 1 and f["median_error"] == pytest.approx(10.0 / 5.0)  # r_eq at t0, not at the target
    assert f["median_persistence"] == pytest.approx(20.0 / 5.0)


def test_forecast_drop_causes_inside_and_beyond_the_run_and_the_hemisphere_split() -> None:
    t = np.arange(0, 2000, 25)  # an active track whose last detection is 1975, in a run that ends at 2000
    f = _fc([_track(1, t, 0.01 * t, 100 + 0.0 * t)])["leads"][1000]
    assert f["drops"] == {"coasting at the target": 1, "beyond the end of the run": 3}
    t = np.arange(0, 2025, 25)
    north = _track(1, t, 0.01 * t, 100 + 0.0 * t, hemi=0)
    south = _track(2, t, 50 + 0.0 * t, 250 + 0.0 * t, hemi=1)
    f = _fc([north, south])["leads"][1000]
    assert f["n"] == 2
    assert f["by_hemisphere"]["north"]["median_persistence"] == pytest.approx(2.0)
    assert f["by_hemisphere"]["south"]["median_persistence"] == pytest.approx(0.0)
    assert f["by_hemisphere"]["north"]["n"] == f["by_hemisphere"]["south"]["n"] == 1


# --- 24. drift -----------------------------------------------------------------------------------------


def test_drift_components_and_the_poleward_sign_in_both_hemispheres() -> None:
    t = np.arange(0, 600, 25)
    # north, y = 60 (below the pole at 80): moving +y is poleward; y = 100 (beyond it): +y is away
    n_toward = _track(1, t, 50 + 0.004 * t, 60 + 0.004 * t, hemi=0)
    n_away = _track(2, t, 50 + 0.0 * t, 100 + 0.004 * t, hemi=0)
    # south, pole at 240: y = 200 moving +y is poleward; y = 280 moving -y is poleward
    s_toward = _track(3, t, 50 + 0.0 * t, 200 + 0.004 * t, hemi=1)
    s_back = _track(4, t, 50 + 0.0 * t, 280 - 0.004 * t, hemi=1)
    d = w0.drift_stats([n_toward, n_away, s_toward, s_back], P0, 0, 600, 3)
    north, south = d["north"], d["south"]
    assert d["gap"] == 250 and north["n"] > 0 and south["n"] > 0
    assert north["units"]["p50"] == pytest.approx(math.hypot(1.0, 1.0)) or north["units"]["p50"] > 0
    rec = np.asarray(d["records"])
    pole = {int(r[0]): [] for r in rec}
    for r in rec:
        pole[int(r[0])].append(r[4])
    assert max(pole[0]) > 0.9 and min(pole[0]) < -0.9  # both signs appear in the north
    assert all(v > 0.9 for v in pole[1])  # the south tracks both move toward the pole
    zonal_n = [r[3] for r in rec if r[0] == 0]
    assert max(zonal_n) == pytest.approx(1.0, abs=1e-9) and min(zonal_n) == pytest.approx(0.0, abs=1e-9)


# --- latitudes -----------------------------------------------------------------------------------------


def test_latitude_stats_pole_distance_and_bins() -> None:
    t = np.arange(0, 300, 25)
    tracks = [
        _track(1, t, 10 + 0 * t, 70 + 0 * t, hemi=0),
        _track(2, t, 10 + 0 * t, 250 + 0 * t, hemi=1),
    ]
    lat = w0.latitude_stats(tracks, P0, 0, 300, 3)
    assert lat["n"] == 24
    assert lat["north"]["median_pole_distance"] == pytest.approx(10.0)
    assert lat["south"]["median_pole_distance"] == pytest.approx(10.0)
    assert lat["within_16_of_pole"] == 1.0
    assert lat["hist"].sum() == 24 and lat["hist"][70 // 8] == 12 and lat["hist"][250 // 8] == 12
    d = w0.pole_distance(np.array([10.0, 70.0, 250.0, 300.0]), np.array([0, 0, 1, 1]), 320.0)
    assert np.allclose(d, [70.0, 10.0, 10.0, 60.0])


# --- utilities -----------------------------------------------------------------------------------------


def test_wall_clock_columns() -> None:
    w = w0.wall(1440)
    assert w["s_at_14.4"] == pytest.approx(100.0) and w["min_at_14.4"] == pytest.approx(100.0 / 60.0)
    assert w["s_at_30"] == pytest.approx(48.0) and w["min_at_30"] == pytest.approx(0.8)


def test_params_hash_of_the_defaults_and_the_start_set() -> None:
    """``{}`` reproduces the confirmed W0 set and ``START_SET`` round 0's start set (docs/STORMS_W0.md)."""
    assert w0.params_hash(WeatherParams.from_dict({})).startswith("7832e060")
    assert w0.params_hash(P0).startswith("bf468155")


def _params_file(tmp_path: Path, d: dict) -> str:
    f = tmp_path / "p.json"
    f.write_text(json.dumps(d))
    return str(f)


def test_load_params_resolves_sparse_files_against_the_start_set(tmp_path: Path) -> None:
    """The tuning rounds' sparse files are overrides of ``START_SET`` (their runs' recorded sets), not of the
    defaults, which are the W0 result since the results commit; without a file the base is the defaults."""
    # {} is the start set (round 0's r0_start), not the confirmed set
    p = w0._load_params(_params_file(tmp_path, {}), [])
    assert p == START_SET and w0.params_hash(p).startswith("bf468155")
    # a one-key file (round 0's r0_qpool105) changes that key of the start set only
    p = w0._load_params(_params_file(tmp_path, {"q_pool": 1.05}), [])
    assert p == replace(START_SET, q_pool=1.05)
    assert (p.gamma, p.zeta_th, p.pools) == (24.0, 0.0075, W.START_POOLS)
    assert p != replace(WeatherParams(), q_pool=1.05)  # what the defaults as the base would have given
    # a file that spells out the four changed fields resolves to the confirmed set
    result = {"gamma": 21.6, "q_pool": 1.11, "pools": [list(r) for r in W.DEFAULT_POOLS], "zeta_th": 0.00625}
    p = w0._load_params(_params_file(tmp_path, result), [])
    assert p == WeatherParams() and w0.params_hash(p).startswith("7832e060")
    # no file: the defaults, and --set goes on top of whichever base applies
    assert w0._load_params(None, []) == WeatherParams()
    assert w0._load_params(None, ["q_pool=1.05"]) == replace(WeatherParams(), q_pool=1.05)
    assert w0._load_params(_params_file(tmp_path, {}), ["q_pool=1.05"]) == replace(START_SET, q_pool=1.05)
    with pytest.raises(ValueError, match="unknown weather parameter"):
        w0._load_params(_params_file(tmp_path, {"no_such_key": 1}), [])
    help_ = next(a.help for a in w0.build_parser()._actions if "--params" in a.option_strings)
    assert "START_SET" in help_ and "defaults" in help_


def test_min_image_and_json_cleaning() -> None:
    assert w0.min_image_np(np.array([170.0, -170.0, 10.0]), 320.0) == pytest.approx([-150.0, 150.0, 10.0])
    out = json.loads(
        w0.dumps({"a": np.float32(1.5), "b": float("nan"), "c": np.arange(3), "d": float("inf")})
    )
    assert out == {"a": 1.5, "b": None, "c": [0, 1, 2], "d": None}


def test_exact_hyperbolicity_threshold_matches_the_spec() -> None:
    # spec 2.5 rule 17: the 4 x 4 flux Jacobian turns complex at 0.871 (h1 32, h2 48) of g'(h1 + h2)
    bound = math.sqrt(0.871 * P0.g_prime * 80.0)
    flags = w0.hyperbolic_real(
        np.array([32.0] * 3), np.array([48.0] * 3), np.array([0.0, 0.98 * bound, 1.02 * bound]), P0
    )
    assert list(flags) == [True, True, False]
    # a thinner h1 lowers the threshold (spec: 0.804 at h1 20, h2 60)
    thin = math.sqrt(0.804 * P0.g_prime * 80.0)
    flags = w0.hyperbolic_real(
        np.array([20.0, 20.0]), np.array([60.0, 60.0]), np.array([0.98 * thin, 1.02 * thin]), P0
    )
    assert list(flags) == [True, False]


def test_grid_noise_index_separates_grid_scale_from_smooth_fields() -> None:
    n = 32
    y, x = np.mgrid[0:n, 0:n]
    smooth = np.sin(2 * np.pi * x / n) * np.cos(2 * np.pi * y / n)
    checker = (-1.0) ** (x + y)
    assert w0.grid_noise_index(smooth) < 0.2
    assert w0.grid_noise_index(checker) == pytest.approx(
        64.0 / 81.0
    )  # the 3 x 3 mean of a checkerboard is c/9


def test_balanced_vortex_has_the_requested_strength() -> None:
    p = replace(P0, n=64)
    ws = w0.balanced_vortex(p)
    z = w0.corners_to_centres(np.array(W.vorticity(ws, p), dtype=np.float64))
    assert 0.45 < z.max() / p.f0 < 0.6  # 0.55 f0 at the corners, averaged to the centres
    j, i = np.unravel_index(z.argmax(), z.shape)
    assert abs((i + 0.5) * p.dx - p.L / 2) < 3 and abs((j + 0.5) * p.dx - p.L / 4) < 3
    h = np.array(ws.h1) + np.array(ws.h2)
    assert np.allclose(h, p.height, atol=1e-4)  # the layers keep the column depth
    assert np.array(ws.h1)[j, i] < p.H1 - 3.0  # a cyclone: layer 1 is thin in the core (geostrophic balance)


# --- organism lifetimes, output-directory rules --------------------------------------------------------


def test_organism_lifetime_reads_only_the_deaths_in_the_span(tmp_path: Path) -> None:
    ev = tmp_path / "events"
    ev.mkdir()
    for a, b, ages in ((0, 100, [1, 2, 3]), (100, 200, [10, 20, 30, 40, 50]), (200, 300, [1000])):
        np.savez(ev / f"deaths_{a:010d}_{b:010d}.npz", age=np.array(ages, dtype=np.int32))
    out = w0.organism_lifetime(tmp_path, 100, span=100)
    assert (
        out["files"] == ["deaths_0000000100_0000000200.npz"]
        and out["median"] == 30.0
        and out["n_deaths"] == 5
    )
    assert w0.organism_lifetime(tmp_path, 5000)["n_deaths"] == 0


def test_out_directory_rules(tmp_path: Path) -> None:
    run = tmp_path / "somewhere" / "run"
    run.mkdir(parents=True)
    with pytest.raises(ValueError, match="inside the tracer source"):
        w0.forbid_out(run / "w0", run)
    with pytest.raises(ValueError, match="runs/"):
        w0.forbid_out(tmp_path / "runs" / "w0_a", None)
    w0.forbid_out(tmp_path / "out", run)  # a sibling is fine
    out = tmp_path / "new"
    w0.prepare_out(out, False)  # new
    assert (out / w0.MARKER).is_file()
    (out / "report.md").write_text("old")
    w0.finish_out(out, ["report.md"])
    with pytest.raises(FileExistsError, match="--overwrite"):
        w0.prepare_out(out, False)
    stray = out / "keep.txt"
    stray.write_text("not ours")
    w0.prepare_out(out, True)  # removes only the files the marker lists
    assert not (out / "report.md").exists() and stray.exists()
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "x").write_text("1")
    with pytest.raises(FileExistsError, match="marker"):
        w0.prepare_out(foreign, True)
    empty = tmp_path / "empty"
    empty.mkdir()
    w0.prepare_out(empty, False)


# --- 25. the occupancy binning equals the engine's -----------------------------------------------------


@pytest.fixture(scope="module")
def tiny_overflow_cfg():
    return load_config(preset="tiny3d", overrides=["brain.weight_dtype=float32", "sensing.cell_capacity=2"])


def test_hash_binning_equals_the_engines_neighbour_search(tiny_overflow_cfg) -> None:
    cfg = tiny_overflow_cfg
    d = dims(cfg)
    rng = np.random.default_rng(0)
    length, height = cfg.world.size, cfg.world.height
    pos = (rng.random((d.n, 3)) * [length, length, height]).astype(np.float32)
    pos[0] = [length - 1e-3, 0.0, height]  # the seam and the surface
    pos[1] = [0.0, length - 1e-3, 0.0]
    pos[2] = [length, length, height + 1.0]  # x = L wraps to cell 0, z above the top clamps to the top layer
    alive = mx.ones((d.n,), dtype=mx.bool_)
    nb = spatial.find_neighbors(mx.array(pos), alive, cfg)
    counts = w0.occupancy_counts(mx.array(pos), cfg)
    assert counts.shape == (d.g_hash_z, d.g_hash, d.g_hash)
    assert int(counts.sum()) == d.n
    # the whole array against an independent binning (a transposed x and y leave the maximum unchanged)
    cx = np.floor(pos[:, 0] / np.float32(d.cell_hash)).astype(np.int64) % d.g_hash
    cy = np.floor(pos[:, 1] / np.float32(d.cell_hash)).astype(np.int64) % d.g_hash
    cz = np.clip(np.floor(pos[:, 2] / np.float32(d.cell_hash_z)).astype(np.int64), 0, d.g_hash_z - 1)
    ref = np.zeros((d.g_hash_z, d.g_hash, d.g_hash), dtype=np.int64)
    np.add.at(ref, (cz, cy, cx), 1)
    assert np.array_equal(np.array(counts), ref)
    assert not np.array_equal(ref, ref.transpose(0, 2, 1))  # x and y are told apart by this fixture
    assert int(counts.max()) == int(nb.max_occupancy)
    assert int((counts > cfg.sensing.cell_capacity).sum()) == int(nb.overflow_cells) > 0  # not vacuous
    # the recorded argmax cell holds the band maximum, inside the band's layers
    peaks, cells = w0.band_peaks(counts)
    flat = np.array(counts).reshape(-1)
    g, gz = d.g_hash, d.g_hash_z
    layers = [(0, 1), (1, gz - 1), (gz - 1, gz)]
    for b, (lo, hi) in enumerate(layers):
        band = np.array(counts)[lo:hi]
        assert int(peaks[b]) == int(band.max())
        assert flat[int(cells[b])] == int(peaks[b])
        assert lo * g * g <= int(cells[b]) < hi * g * g
    assert w0.decode_cell(int(cells[0]), cfg)[2] == 0
    baseline = w0.baseline_occupancy(pos, cfg)
    assert baseline["peak"] == [int(v) for v in np.array(peaks)] and baseline["total"] == d.n
    assert sum(baseline["agents"]) == d.n and sum(baseline["z_band_quarters"]) == d.n


# --- 26. the tracer step ----------------------------------------------------------------------------------


def _sine_state(p: WeatherParams, amp: float = 0.1) -> W.WeatherState:
    """u1 = amp sin(2 pi x / L): a divergence that drives w, over the rest state."""
    fo = W.default_forcing(p)
    ws = W.init_state(p, fo)
    x = (np.arange(p.n) + 1.0) * p.dx
    u1 = np.tile(amp * np.sin(2.0 * np.pi * x / p.L)[None, :], (p.n, 1))
    return ws._replace(u1=mx.array(u1, dtype=mx.float32))


def _reference_move(
    ws, pos: np.ndarray, al: np.ndarray, p: WeatherParams, cfg, floor_held: bool
) -> np.ndarray:
    """The tracer motion written out from ``sample_wind`` with numpy: v = α u, the 3D clip, wrap and clamp."""
    d = dims(cfg)
    aw = W.sample_wind(ws, mx.array(pos), p, field_cell=cfg.world.field_cell)
    u = np.array(aw.u, dtype=np.float64)
    if floor_held:
        u[pos[:, 2] < d.cell_hash_z, 2] = 0.0
    v = al[:, None] * u
    sp = np.sqrt((v * v).sum(axis=1))
    v = v * (cfg.physics.v_max / np.maximum(sp, cfg.physics.v_max))[:, None]
    new = pos.astype(np.float64) + v
    new[:, :2] = np.mod(new[:, :2], cfg.world.size)
    new[:, 2] = np.clip(new[:, 2], d.z_lo, d.z_hi)
    return new


@pytest.fixture(scope="module")
def world_160():
    p = WeatherParams.from_dict({**SMALL, "seed_rate": 0.0})
    return p, load_config(preset="64k3d", overrides=["world.size=160"])


@pytest.mark.parametrize("floor_held", [False, True])
def test_tracer_step_is_wrap_and_clamp_of_the_clipped_wind(world_160, floor_held: bool) -> None:
    p, cfg = world_160
    d = dims(cfg)
    ws = _sine_state(p, amp=0.9)  # large: the 3D clip and the u_max clip both act on some tracers
    rng = np.random.default_rng(2)
    m = 4000
    pos = (rng.random((m, 3)) * [p.L, p.L, d.z_hi - d.z_lo] + [0, 0, d.z_lo]).astype(np.float32)
    pos[:200, 2] = d.z_lo  # on the floor
    pos[200:400, 2] = d.z_hi  # at the surface
    pos[400:410, 0] = p.L - 1e-4  # at the seam
    al = rng.uniform(0.25, 1.0, m).astype(np.float32)
    new, n_h, _, n_edge = w0.tracer_move(ws, mx.array(pos), mx.array(al), p, cfg, floor_held)
    ref = _reference_move(ws, pos, al, p, cfg, floor_held)
    assert np.allclose(np.array(new), ref, atol=2e-5)
    assert int(n_h) > 0  # the u_max clip acted (not vacuous)
    z = np.array(new)[:, 2]
    assert z.min() >= d.z_lo and z.max() <= d.z_hi
    assert int(n_edge) == int(((pos[:, 2] < cfg.world.field_cell) | (pos[:, 2] >= 76.0)).sum())
    if floor_held:  # in hash layer 0 the tracer does not rise or sink
        low = pos[:, 2] < d.cell_hash_z
        assert np.allclose(np.array(new)[low, 2], pos[low, 2], atol=1e-6)


def test_tracer_rig_reseeds_every_tracer_once_per_life_at_its_offset(world_160) -> None:
    p, cfg = world_160
    life, m = 10, 64
    rig = w0.TracerRig(p, cfg, None, None, m=m, key=3, life=life)
    ws = W.init_state(p, W.default_forcing(p))  # no wind: only the reseeds move tracers
    last = np.array(rig.pos)
    changed = np.zeros((2 * life, m), dtype=bool)
    for t in range(2 * life):
        mx.eval(*rig.step(ws, t))
        now = np.array(rig.pos)
        changed[t] = np.any(now != last, axis=1)
        last = now
    expected = ((np.arange(2 * life)[:, None] + np.arange(m)[None, :]) % life) == 0
    assert np.array_equal(changed, expected)
    assert (changed.sum(axis=0) == 2).all()  # exactly once per tracer_life steps
    blk = rig.read()
    assert blk["occ"].shape == (2 * life, 6) and list(blk["steps"]) == list(range(2 * life))


def test_tracer_rig_source_rows_and_the_running_peak(world_160) -> None:
    p, cfg = world_160
    d = dims(cfg)
    rng = np.random.default_rng(5)
    m = 300
    src = (rng.random((m, 3)) * [p.L, p.L, d.z_hi - d.z_lo] + [0, 0, d.z_lo]).astype(np.float32)
    cap = rng.random(m).astype(np.float32)
    rig = w0.TracerRig(p, cfg, src, cap, key=1, life=5)
    ws = W.init_state(p, W.default_forcing(p))
    rows = {tuple(r) for r in np.round(src, 5)}
    for t in range(12):
        mx.eval(*rig.step(ws, t))
    pos = np.round(np.array(rig.pos), 5)
    assert all(tuple(r) in rows for r in pos)  # still on source rows: reseeds copy a row, no wind moves them
    blk = rig.read()
    # the device running peak is the maximum of the per-step maxima, with the first step and its cell
    for b in range(3):
        j = int(np.argmax(blk["occ"][:, b]))
        assert blk["peak"][b, 0] == blk["occ"][j, b] and blk["peak"][b, 1] == blk["occ"][j, 3 + b]
        assert blk["peak"][b, 2] == blk["steps"][j]
    assert int(blk["zq"].sum()) == m


def test_uniform_resampling_baseline_is_above_the_start_peak_in_expectation(world_160) -> None:
    p, cfg = world_160
    d = dims(cfg)
    m = 3000

    def uniform(rng: np.random.Generator) -> np.ndarray:
        u = rng.random((m, 3))
        return np.stack([u[:, 0] * p.L, u[:, 1] * p.L, d.z_lo + u[:, 2] * (d.z_hi - d.z_lo)], axis=-1)

    out = w0.resampled_peaks(uniform, cfg, draws=5)
    assert out["draws"] == 5 and len(out["peak_median"]) == 3 and all(v >= 1 for v in out["peak_max"])


# --- 14b (tracer part). a steady tilted interface keeps uniform tracers uniform ----------------------------


def _shell_density(monkeypatch, drop_j: bool, steps: int = 200, m: int = 60_000) -> float:
    """Density of uniform tracers within 0.5 of a steady tilted interface, against uniform, after ``steps``
    steps in the frozen wind of spec test 14b (a uniform flux under ``h1 = H1 + 8 sin(2 pi x / L)``)."""
    n = 64
    p = replace(P0, n=n, dx=2.5, gamma=0.0, k_E=0.0, tau_rad=0.0, seed_rate=0.0, z_sl=1e-3, f0=0.0)
    cfg = load_config(preset="64k3d", overrides=[f"world.size={p.L:g}"])
    d = dims(cfg)
    x = (np.arange(n) + 0.5) * p.dx
    h1c = p.H1 + 8.0 * np.sin(2.0 * np.pi * x / p.L)  # centres
    h1 = np.tile(h1c[None, :], (n, 1))
    u1 = 0.05 * p.H1 / (0.5 * (h1 + np.roll(h1, -1, 1)))  # a uniform flux at the u faces: h1 stays steady
    ws = W.init_state(p, W.default_forcing(p))
    f32 = lambda a: mx.array(a, dtype=mx.float32)  # noqa: E731
    ws = ws._replace(h1=f32(h1), h2=f32(p.height - h1), u1=f32(u1))
    if drop_j:
        real = W.wind_fields
        monkeypatch.setattr(W, "wind_fields", lambda w, q: real(w, q)._replace(jump=real(w, q).jump * 0.0))
    rng = np.random.default_rng(11)
    pos = np.stack(
        [rng.random(m) * p.L, rng.random(m) * p.L, d.z_lo + rng.random(m) * (d.z_hi - d.z_lo)], axis=-1
    ).astype(np.float32)
    al = mx.ones((m,), dtype=mx.float32)
    move = mx.compile(lambda ps: w0.tracer_move(ws, ps, al, p, cfg)[0])
    ps = mx.array(pos)
    for _ in range(steps):
        ps = move(ps)
        mx.eval(ps)
    out = np.array(ps)
    zi = np.interp(out[:, 0] % p.L, np.concatenate([x - p.L, x, x + p.L]), np.tile(h1c, 3))
    near = np.abs(out[:, 2] - zi) <= 0.5
    return float(near.mean() * (d.z_hi - d.z_lo))  # against a unit-thick shell's share of the 78-unit column


def test_uniform_tracers_keep_their_density_near_a_tilted_interface(monkeypatch) -> None:
    """Spec 14b: with the Leibniz term J the steady tilted interface is a material surface and uniform tracers
    stay uniform near it (the density is 1 within the Poisson noise of 3.6%); without J they pile up there."""
    assert 0.9 < _shell_density(monkeypatch, drop_j=False) < 1.1
    assert _shell_density(monkeypatch, drop_j=True) > 1.25


# --- the run loop, forks and the experiments -----------------------------------------------------------


def _vortex_params(n: int = 48) -> WeatherParams:
    """A dry, unforced vortex run that detects its vortex as a storm (no rain needed: p_th = 0)."""
    pool = ((n * P0.dx / 2, n * P0.dx / 4, 16.0, 8.0, 1.0, 0.0, 0.0),)  # one pool under the vortex
    return replace(
        P0, n=n, gamma=0.0, k_E=0.0, tau_rad=0.0, seed_rate=0.0, p_th=0.0, r_min=1, a_min=12, pools=pool
    )


def _vortex_sim(p: WeatherParams, upto: int, snap_at: int | None = None, **kw):
    sim = w0.Sim(p, 0, **kw)
    sim.ws = w0.balanced_vortex(p)
    holder: dict[str, w0.Snapshot] = {}

    def cb(s, fr, det, ids, interval):
        if snap_at is not None and fr.t == snap_at:
            holder["snap"] = s.snapshot(fr, det, ids)

    sim.callbacks.append(cb)
    sim.advance(upto)
    return sim, holder.get("snap")


def test_fork_continues_the_run_bitwise_and_detects_the_vortex() -> None:
    p = _vortex_params()
    sim, snap = _vortex_sim(p, 150, snap_at=75)
    tracks = sim.tracker.finish()
    assert len(tracks) == 1 and len(tracks[0].dets) == 6  # the vortex is one storm, detected every 25 steps
    assert snap is not None and snap.t == 75
    ctrl, log = w0.run_fork(snap, p, 75)  # the control continuation: zero momentum arrays instead of none
    assert [d.t for d in ctrl[0].dets] == [d.t for d in tracks[0].dets]
    for a, b in zip(ctrl[0].dets, tracks[0].dets, strict=True):
        if mx.default_device() != mx.gpu:
            assert (a.x, a.y) == (b.x, b.y)  # bitwise on the CPU, as the weather's zero-forcing test
        else:
            assert abs(a.x - b.x) < 1e-3
    assert sorted(log.rows) == [100, 125, 150]


def test_impulse_displaces_the_storm_against_the_shared_control() -> None:
    p = _vortex_params()
    _, snap = _vortex_sim(p, 75, snap_at=75)
    assert snap is not None
    specs = w0.auto_impulses(snap, p, 75, p.L / 2, p.L / 4, 0.0, duration=25, factors=(1.0, 2.0))
    assert [f for f, _ in specs] == [1.0, 2.0]
    s1, s2 = specs[0][1], specs[1][1]
    assert s2.ux == pytest.approx(2 * s1.ux) and s1.uy == pytest.approx(0.0, abs=1e-12)  # twice the momentum
    assert s1.radius == pytest.approx(snap.detection.storms[0].r_eq) and s1.duration == 25
    res = w0.impulse_experiment(snap, p, specs, horizon=250)
    assert res["target_id"] == 1 and len(res["runs"]) == 2 and res["runs"][0]["neighbours"] == []
    for r in res["runs"]:
        assert r["momentum_ratio"] == pytest.approx(r["factor"], rel=1e-9)
        assert [h["horizon"] for h in r["horizons"]] == [250]
        assert r["horizons"][0]["available"] and r["horizons"][0]["survived_perturbed"]
    d1, d2 = (r["horizons"][0]["displacement_units"] for r in res["runs"])
    assert d1 > 0.05 and d2 > d1  # the push moves the storm, and a stronger push moves it more
    assert res["runs"][0]["horizons"][0]["displacement_radii"] == pytest.approx(d1 / s1.radius)


def test_core_momentum_is_the_footprint_sum_of_h_times_speed() -> None:
    p = _vortex_params()
    _, snap = _vortex_sim(p, 75, snap_at=75)
    assert snap is not None and snap.frame is not None
    fr, c = snap.frame, snap.detection.storms[0]
    uc1, vc1 = w0.centre_winds(fr.u1, fr.v1)
    uc2, vc2 = w0.centre_winds(fr.u2, fr.v2)
    j, i = c.flat // p.n, c.flat % p.n
    m1 = (fr.h1[j, i] * np.hypot(uc1[j, i], vc1[j, i])).sum() * p.dx**2
    m2 = (fr.h2[j, i] * np.hypot(uc2[j, i], vc2[j, i])).sum() * p.dx**2
    both = w0.core_momentum(snap, p, 0, "12")
    assert both["core_momentum"] == pytest.approx(float(m1 + m2), rel=1e-6)
    assert w0.core_momentum(snap, p, 0, "1")["core_momentum"] == pytest.approx(float(m1), rel=1e-6)
    assert both["thickness_sum"] == pytest.approx(float(fr.h1[j, i].mean() + fr.h2[j, i].mean()), rel=1e-6)
    assert both["r_eq"] == c.r_eq


def test_impulse_arrays_peak_and_total_velocity() -> None:
    p = replace(P0, n=64)
    s = w0.ImpulseSpec.parse("100,64,32,0.04,-0.02,8,40")
    fu1, fv1, fu2, fv2 = (np.array(a) for a in w0.impulse_arrays(p, s))
    assert s.layers == "12" and np.array_equal(fu1, fu2) and np.array_equal(fv1, fv2)
    assert fu1.max() == pytest.approx(0.04 / 40, rel=0.02)  # the disc peak is U / DURATION per step
    assert fv1.min() == pytest.approx(-0.02 / 40, rel=0.02)
    j, i = np.unravel_index(fu1.argmax(), fu1.shape)
    assert abs((i + 1.0) * p.dx - 64) <= p.dx and abs((j + 0.5) * p.dx - 32) <= p.dx  # u-point positions
    only1 = w0.ImpulseSpec.parse("100,64,32,0.04,-0.02,8,40,1")
    a = [np.array(x) for x in w0.impulse_arrays(p, only1)]
    assert np.abs(a[2]).max() == 0 and np.abs(a[3]).max() == 0 and np.abs(a[0]).max() > 0
    with pytest.raises(ValueError):
        w0.ImpulseSpec.parse("1,2,3")
    with pytest.raises(ValueError):
        w0.ImpulseSpec.parse("1,2,3,4,5,6,7,9")


def test_pool_operations_and_the_supersaturated_footprint() -> None:
    p = P0
    shift = w0.PoolOp.parse_shift("500,2,-100,60")
    pools = shift.apply(p.pools, p.L)
    assert pools[2][:2] == ((0.0 - 100.0) % 320.0, (272.0 + 60.0) % 320.0) and pools[0] == p.pools[0]
    gain = w0.PoolOp.parse_gain("500,0,0.7")
    assert gain.apply(p.pools, p.L)[0][4] == pytest.approx(0.7)
    with pytest.raises(ValueError, match="out of range"):
        w0.PoolOp.parse_gain("1,9,1.0").apply(p.pools, p.L)
    with pytest.raises(ValueError):
        w0.PoolOp.parse_shift("1,2,3")
    thr = w0.supersaturated_threshold(p)
    assert thr == pytest.approx(0.75)  # (1 - 0.7) / (1.1 - 0.7)
    pool = p.pools[0]
    assert w0.pool_value(pool, pool[0], pool[1], p.L) == pytest.approx(1.0)
    assert w0.pool_value(pool, pool[0] + 24.0, pool[1], p.L) == pytest.approx(math.exp(-0.5 * 0.75**2))
    # the semi-axes of the supersaturated ellipse are 24 x 12 (spec 3.3): sigma_y 16 at 12 is 0.75 sigma
    assert w0.pool_value(pool, pool[0], pool[1] + 12.0, p.L) == pytest.approx(math.exp(-0.5 * 0.75**2))


def test_causal_storm_stats() -> None:
    t = np.arange(0, 400, 25)
    a = _track(1, t, 10 + 0 * t, 40 + 0 * t, hemi=0)
    b = _track(2, t[8:], 50 + 0 * t[8:], 40 + 0 * t[8:], hemi=0)  # starts at t = 200: 3 detections by 250
    c = _track(3, t, 90 + 0 * t, 250 + 0 * t, hemi=1)
    for tr, w in ((a, 0.1), (b, 0.2), (c, 0.3)):
        for d in tr.dets:
            d.wind_max = w
    s = w0._causal_stats([a, b, c], 225, 3, None)
    assert s["count"] == 2  # b has only 2 detections so far at 225
    s = w0._causal_stats([a, b, c], 250, 3, None)
    assert s["count"] == 3 and s["core_wind"] == pytest.approx(0.2)
    assert w0._causal_stats([a, b, c], 250, 3, 1)["count"] == 1


def _anchored_snapshot(p: WeatherParams) -> w0.Snapshot:
    """A snapshot at t = 500 with one established storm (track 7, hemisphere 0) just east of pool 0."""
    comp = replace(_comp(1, (10, 10, 4, 4)), x=82.0, y=50.0)
    tracker = w0.Tracker(p)
    tracker.tracks = [_track(7, [450, 475, 500], [82.0] * 3, [50.0] * 3)]
    det = w0.Detection(500, np.zeros((p.n, p.n), dtype=np.int32), [comp], [], 1, 0)
    return w0.Snapshot(500, None, None, tracker, p.pools, None, det, [7])  # type: ignore[arg-type]


def test_pool_shift_lags_are_read_from_the_tracks_of_the_perturbed_run(monkeypatch) -> None:
    """Genesis lag: first detection of a storm born after t0 inside the shifted pool's supersaturated
    footprint. Displacement lag: first detection where the anchored storm has covered half of the shift (the
    projection on the shift vector, 0.5 and not another fraction)."""
    p = P0
    snap = _anchored_snapshot(p)
    op = w0.PoolOp.parse_shift("500,0,40,0")  # pool 0 from (80, 48) to (120, 48)
    t = np.arange(500, 1025, 25)
    anchored = _track(7, t, 82.0 + 0.12 * (t - 500), 50.0 + 0 * t)  # 0.12 units per step along the shift
    inside = _track(9, range(700, 900, 25), [121.0] * 8, [49.0] * 8)  # born at 700 in the new footprint
    outside = _track(10, range(600, 800, 25), [250.0] * 8, [49.0] * 8)  # born earlier, far from it
    monkeypatch.setattr(w0, "run_fork", lambda *a, **k: ([anchored, inside, outside], w0.LightLog()))
    out = w0.pool_experiment(snap, p, op, horizon=525, ctrl=[])
    assert out["shift_units"] == 40.0 and out["hemisphere"] == "north"
    assert (
        out["genesis_lag"] == 200 and out["genesis_track"] == 9
    )  # the storm born at 700, not the one at 600
    assert (
        out["first_detection_lag"] is not None and out["first_detection_lag"] < 200
    )  # the old storm arrives
    anc = out["anchored"]
    assert anc["id"] == 7 and anc["displacement_lag"] == 175  # 21 units = 0.525 of 40 at +175 (0.45 at +150)
    assert anc["max_fraction_of_shift"] == pytest.approx(0.12 * 500 / 40.0)
    never = _track(7, t, 82.0 + 0.001 * (t - 500), 50.0 + 0 * t)
    monkeypatch.setattr(w0, "run_fork", lambda *a, **k: ([never], w0.LightLog()))
    out = w0.pool_experiment(snap, p, op, horizon=525, ctrl=[])
    assert out["anchored"]["displacement_lag"] is None and out["genesis_lag"] is None


def test_pool_experiment_and_run_steerability_on_a_vortex() -> None:
    p = _vortex_params()
    sim = w0.Sim(p, 0)
    sim.ws = w0.balanced_vortex(p)
    rec = w0.Recorder(p, t_spin=25, snap_times=[75])
    rec.start(sim)
    sim.callbacks.append(rec)
    sim.advance(100)
    st = w0.Settings(steps=100, t_spin=25)
    res = w0.StudyResult(p, st, sim, rec, sim.tracker.finish(), None, None, 0.0)
    out = w0.run_steerability(
        res,
        impulses=[w0.ImpulseSpec(75, p.L / 2, p.L / 4, 0.03, 0.0, 10.0, 25)],
        autos=[],
        shifts=[w0.PoolOp.parse_shift("75,0,12,0")],
        gains=[w0.PoolOp.parse_gain("75,0,0.7")],
        horizons={"impulse": 250, "shift": 100, "gain": 100},
        after=(50, 100),
    )
    chk = out["control_vs_study"][0]
    assert chk["n_compared_detections"] > 0 and chk["max_position_difference_units"] < 1e-3
    (imp,) = out["impulse"]
    assert imp["runs"][0]["factor"] is None and imp["runs"][0]["horizons"][0]["available"]
    (sh,) = out["pool_shift"]
    assert sh["shift_units"] == pytest.approx(12.0) and sh["genesis_lag"] is None  # a dry run grows no storm
    assert sh["anchored"] is not None and sh["anchored"]["displacement_lag"] is None
    (gn,) = out["pool_gain"]
    assert [r["after"] for r in gn["gain"]["rows"]] == [50, 100]
    assert all(math.isfinite(r["core_wind_control"]) for r in gn["gain"]["rows"])
    text = w0.render_steerability(out)
    assert "Wind impulses from t0 = 75" in text and "Pool shift 0" in text and "Pool gain x0.7" in text


def test_vortex_decay_table_lines_and_limit() -> None:
    p = replace(P0, n=48)
    tab = w0.vortex_decay_table(p, limit=50, every=25)
    assert list(tab["lines"]) == [
        "default numerics",
        "nu 0.02, kappa_d 0 (the probe)",
        "no physical damping",
        "cubic SL (kappa_d 0.10)",
    ]
    assert all(v["decay_step"] is None and v["text"] == "> 50" for v in tab["lines"].values())
    assert tab["lines"]["cubic SL (kappa_d 0.10)"]["sl_order"] == 3
    assert tab["lines"]["nu 0.02, kappa_d 0 (the probe)"]["kappa_d"] == 0.0


# --- the criteria, the health gate and the early-stop test -----------------------------------------------


def _row(**kw) -> dict:
    base = {"n": 10, "median_error": 1.0, "median_floor": 0.5, "skill": 0.2, "drift_radii_median": 1.0}
    return {**base, **kw}


def test_forecast_criterion_reads_every_branch_of_d9() -> None:
    fcrit = w0.forecast_criterion
    ok = fcrit(_row())
    assert ok["evidence"] and ok["in_range"] and ok["exceeds_jitter_floor"] and ok["skill_positive"]
    assert ok["reading"].startswith("met, and above the jitter floor") and not ok["mobility"]
    below = fcrit(_row(median_floor=1.2))  # in range, but the jitter alone gives that much
    assert below["in_range"] and not below["evidence"] and not below["exceeds_jitter_floor"]
    assert below["reading"] == "met as written, not evidence of forecastable motion"
    equal = fcrit(_row(median_floor=1.0))  # the floor must be exceeded, not matched
    assert not equal["evidence"] and equal["reading"].startswith("met as written")
    noskill = fcrit(_row(skill=-0.1))
    assert noskill["in_range"] and not noskill["evidence"] and not noskill["skill_positive"]
    assert not fcrit(_row(skill=0.0))["evidence"]
    for med in (0.49, 2.01):
        r = fcrit(_row(median_error=med))
        assert not r["in_range"] and not r["evidence"] and r["reading"] == "not met"
    for med in (0.5, 2.0):  # the band is closed
        assert fcrit(_row(median_error=med, median_floor=0.1))["evidence"]
    assert (
        fcrit(_row(drift_radii_median=0.49))["mobility"]
        and not fcrit(_row(drift_radii_median=0.5))["mobility"]
    )
    none = fcrit(
        _row(n=0, median_error=math.nan, median_floor=math.nan, skill=math.nan, drift_radii_median=math.nan)
    )
    assert not none["in_range"] and not none["evidence"] and none["mobility"] and none["reading"] == "not met"


GATE_OK = dict(n_safety=0, hyper_max=0.5, eig_complex=0, eps_local_max=0.9, mass_dev=1e-6, limiter_share=0.05)
GATE_BAD = {
    "no_safety_clips": dict(n_safety=1),
    "hyperbolicity_below_0.8": dict(hyper_max=0.8),
    "exact_eigenvalue_test_real": dict(eig_complex=1),
    "eps_local_max_below_1": dict(eps_local_max=1.0),
    "mass_drift_below_1e-5": dict(mass_dev=1e-5),
    "limiter_bound_share_at_most_5%": dict(limiter_share=0.0501),
}


def test_health_gate_has_these_thresholds_and_fails_on_nan() -> None:
    assert all(w0.health_gate(**GATE_OK).values()) and set(w0.health_gate(**GATE_OK)) == set(GATE_BAD)
    for name, bad in GATE_BAD.items():
        gate = w0.health_gate(**{**GATE_OK, **bad})
        assert [k for k, v in gate.items() if not v] == [name], name
    just_in = dict(hyper_max=0.7999, eps_local_max=0.9999, mass_dev=9.9e-6, limiter_share=0.05)
    assert all(w0.health_gate(**{**GATE_OK, **just_in}).values())
    for key in ("hyper_max", "eps_local_max", "mass_dev", "limiter_share"):
        assert not all(w0.health_gate(**{**GATE_OK, key: math.nan}).values()), key


def _hemisphere_tracks() -> list[w0.Track]:
    t = np.arange(0, 3025, 25)
    return [
        _track(1, t, 0.01 * t, 100 + 0 * t, hemi=0),
        _track(2, t, 0.01 * t + 150, 100 + 0 * t, hemi=0),
        _track(3, t, 50 + 0 * t, 250 + 0 * t, hemi=1),
        _track(4, t, 150 + 0 * t, 250 + 0 * t, hemi=1),
    ]


def test_early_stop_inputs_are_per_hemisphere() -> None:
    tracks = _hemisphere_tracks()
    times = w0.detection_times(1000, 3000, 25)
    cnt = w0.storm_counts(tracks, times, 3)
    life = w0.lifetime_stats(tracks, 1000, 3)
    fc = w0.forecast_stats(tracks, P0, 1000, 3000, 3)
    es = w0.early_stop_inputs(cnt, life, fc)
    assert es["north"]["ok"] and es["north"]["meets"] == {"storms": True, "lifetime": True, "drift": True}
    assert es["north"]["mean_storms"] == 2.0 and es["north"]["drift_radii_per_1000"] == pytest.approx(2.0)
    assert es["south"]["meets"] == {"storms": True, "lifetime": True, "drift": False}  # stationary storms
    assert es["south"]["drift_radii_per_1000"] == pytest.approx(0.0) and es["south"]["drift_n"] > 0
    assert not es["run_meets"] and es["south"]["lifetime_km"]["reached"] == 3000.0
    # the lifetime bound is on the Kaplan-Meier median (or the longest age when it is not reached)
    short = [
        *_hemisphere_tracks()[:2],
        _track(
            5,
            np.arange(0, 600, 25),
            0 * np.arange(0, 600, 25) + 9,
            0 * np.arange(0, 600, 25) + 250,
            hemi=1,
            end="dissipated",
        ),
    ]
    life2 = w0.lifetime_stats(short, 1000, 3)
    cnt2 = w0.storm_counts(short, times, 3)
    es2 = w0.early_stop_inputs(cnt2, life2, fc)
    assert not es2["south"]["meets"]["storms"] and not es2["south"]["meets"]["lifetime"]


@pytest.fixture(scope="module")
def mini_study():
    p = WeatherParams.from_dict(SMALL)
    st = w0.Settings(steps=300, t_spin=100, uniform_tracers=200)
    return w0.run_study(p, st)


def _gate(m: dict) -> dict:
    return m["health"]["gate"]


def test_assemble_metrics_passes_a_clean_run_and_fails_each_gate_entry(mini_study, monkeypatch) -> None:
    res = mini_study
    metrics, _ = w0.assemble_metrics(res)
    assert all(_gate(metrics).values()) and metrics["health"]["pass"]
    n_safety = w0.ACC_CNT.index("n_safety")

    def run(**ov) -> dict:
        cnt = res.sim.totals_cnt.copy()
        cnt[n_safety] = ov.get("n_safety", 0)
        mxr = np.array([0.4, 0.2, ov.get("hyper", 0.5), ov.get("eps", 0.9)])
        monkeypatch.setattr(res.sim, "totals_cnt", cnt)
        monkeypatch.setattr(res.sim, "max_run", mxr)
        monkeypatch.setattr(res.rec, "eig_complex", ov.get("eig", 0))
        monkeypatch.setattr(res.rec, "mass_dev", ov.get("mass", 1e-7))
        return w0.assemble_metrics(res)[0]

    assert all(_gate(run()).values())
    cases = {
        "no_safety_clips": dict(n_safety=3),
        "hyperbolicity_below_0.8": dict(hyper=0.81),
        "exact_eigenvalue_test_real": dict(eig=2),
        "eps_local_max_below_1": dict(eps=1.01),
        "mass_drift_below_1e-5": dict(mass=2e-5),
    }
    for name, ov in cases.items():
        m = run(**ov)
        assert [k for k, v in _gate(m).items() if not v] == [name], name
        assert not m["health"]["pass"] and "health" in m["verdict"]["failed"]
    m = run(hyper=0.79, mass=9e-6)
    assert all(_gate(m).values())
    assert m["health"]["hyper_max"] == pytest.approx(0.79) and m["health"]["eps_local_max"] == pytest.approx(
        0.9
    )


def test_assemble_metrics_reads_the_forecast_criterion_in_each_d9_branch(mini_study, monkeypatch) -> None:
    """The three readings through the whole assembly, on synthetic tracks with known forecast errors."""
    res = mini_study
    monkeypatch.setattr(res.st, "t_spin", 1000)
    monkeypatch.setattr(res.st, "steps", 2000)
    t = np.arange(0, 2025, 25)
    rng = np.random.default_rng(5)
    still = [_track(k, t, 100 + 0.0 * t, 100 + 0.0 * t) for k in range(1, 4)]  # no motion: error 0
    jitter = [
        _track(k, t, 100 + 1.2 * rng.standard_normal(t.size), 100 + 1.2 * rng.standard_normal(t.size), r=10.0)
        for k in range(1, 40)
    ]  # centroid noise only: 0.5 to 2 radii of error, at the floor, worse than persistence
    x = np.where(t <= 1000, 0.01 * t, 10.0 + 0.018 * (t - 1000))
    turning = [
        _track(k, t, x + 20 * k, 100 + 0.0 * t) for k in range(1, 4)
    ]  # a speed change after the origin

    def reading(tracks: list[w0.Track]) -> dict:
        monkeypatch.setattr(res, "tracks", tracks)
        c3 = w0.assemble_metrics(res)[0]["criteria"]["forecast"]
        return {"pass": c3["pass"], **c3["detail"]}

    d = reading(still)
    assert d["d9_reading"] == "not met" and d["pass"] is False and d["pass_as_written"] is False
    d = reading(jitter)
    assert d["d9_reading"] == "met as written, not evidence of forecastable motion"
    assert d["pass_as_written"] is True and d["pass"] is False and d["skill_vs_persistence"] < 0
    d = reading(turning)
    assert d["d9_reading"].startswith("met, and above the jitter floor") and d["pass"] is True
    assert 0.5 <= d["leads"]["1000"]["median_error"] <= 2.0 and d["skill_vs_persistence"] > 0


def test_assemble_metrics_limiter_share_counts_h1_at_or_below_one_and_a_half_h_min(
    mini_study, monkeypatch
) -> None:
    res = mini_study
    p = res.p
    tracks = [_track(k, range(100, 301, 25), [10 * k] * 9, [40] * 9) for k in range(1, 13)]  # 108 detections
    dets = [d for tr in tracks for d in tr.dets]
    for d in dets[:5]:
        d.h1_min = 1.5 * p.h_min  # limiter-bound: exactly at the threshold counts
    for d in dets[5:25]:
        d.h1_min = 2.5 * p.h_min  # thin, but not bound: must not count
    monkeypatch.setattr(res, "tracks", tracks)
    m, _ = w0.assemble_metrics(res)
    assert m["health"]["limiter_bound_detections"] == 5 and len(dets) == 108
    assert m["health"]["limiter_bound_share"] == pytest.approx(5 / 108)
    assert _gate(m)["limiter_bound_share_at_most_5%"]
    dets[5].h1_min = 1.4 * p.h_min
    m, _ = w0.assemble_metrics(res)
    assert m["health"]["limiter_bound_detections"] == 6 and not _gate(m)["limiter_bound_share_at_most_5%"]


def test_assemble_metrics_moisture_residual_includes_the_clamped_moisture(mini_study, monkeypatch) -> None:
    res = mini_study
    sums = np.zeros(len(w0.ACC_SUM))
    for name, v in (("q_clamp", 1.0), ("sum_E", 20.0), ("sum_C", 14.0), ("sum_Qseed", 2.0)):
        sums[w0.ACC_SUM.index(name)] = v
    monkeypatch.setattr(res.sim, "totals_sum", sums)
    monkeypatch.setattr(res.rec, "q0", 100.0)
    monkeypatch.setattr(res.rec, "q_last", 105.0)
    m, _ = w0.assemble_metrics(res)
    assert m["health"]["q_budget_residual"] == pytest.approx(105.0 - 100.0 - (20.0 - 14.0 + 2.0 + 1.0))
    assert m["health"]["q_budget_residual_relative"] == pytest.approx(-0.04)


@pytest.mark.filterwarnings("ignore:Mean of empty slice")  # the stationarity blocks of a 300-step study
def test_assemble_metrics_reports_the_lifetime_flags_and_the_early_stop(mini_study, monkeypatch) -> None:
    res = mini_study
    tracks, _ = _fragments_and_storms()
    monkeypatch.setattr(res, "tracks", tracks)
    monkeypatch.setattr(res.st, "t_spin", 1000)
    monkeypatch.setattr(res.st, "steps", 30000)
    m, _ = w0.assemble_metrics(res)
    c2 = m["criteria"]["lifetime"]
    assert c2["pass"] is False and c2["value"] == 325.0
    d = c2["detail"]
    assert d["median_reliable"] and d["n_risk_at_median"] == 11 and d["n_risk_at_threshold"] == 4
    assert d["companions"]["share_detections_in_long_storms"] > 0.9 and d["reading"].startswith("median 325")
    es = m["diagnostics"]["early_stop"]
    assert set(es) >= {"north", "south", "run_meets"} and es["north"]["lifetime_km"]["n"] == 23
    json.dumps(m, default=str)  # serialisable (the CLI writes it with w0.dumps)
    assert json.loads(w0.dumps(m))["criteria"]["lifetime"]["detail"]["companions"]["split_born"]["n"] == 21


DETECTION_KEYS = ("zeta_th", "p_th", "a_min", "r_min", "miss_grace", "min_dets", "detect_every")


def test_deviations_compare_the_detection_thresholds_with_the_start_set(mini_study) -> None:
    """The report's "detection thresholds changed" line is against ``START_SET`` (the spec's thresholds), not
    the defaults: a confirmed-set study lists ``zeta_th``, a start-set study lists nothing."""
    start = {k: getattr(START_SET, k) for k in DETECTION_KEYS}

    def line(**ov: float) -> list[str]:
        res = replace(mini_study, p=replace(mini_study.p, **{**start, **ov}))
        devs = w0.assemble_metrics(res)[0]["deviations"]
        return [d for d in devs if d.startswith("detection thresholds changed")]

    assert START_SET.zeta_th != WeatherParams().zeta_th  # the two bases differ, so the test tells them apart
    assert line() == []  # a start-set study: nothing changed (the defaults as the base listed zeta_th 0.0075)
    assert line(zeta_th=WeatherParams().zeta_th) == [
        "detection thresholds changed from the start set: zeta_th = 0.00625"
    ]  # the confirmed set (the defaults as the base listed nothing)
    assert line(zeta_th=0.00625, a_min=6) == [
        "detection thresholds changed from the start set: zeta_th = 0.00625, a_min = 6"
    ]
    assert line(zeta_th=0.0036, r_min=2) == [
        "detection thresholds changed from the start set: zeta_th = 0.0036, r_min = 2"
    ]


# --- 27. the command line ------------------------------------------------------------------------------


STUDY_FILES = ["params.json", "metrics.json", "tracks.csv", "storms.csv", "timeseries.csv", "report.md"]
METRIC_KEYS = [
    "run", "criteria", "diagnostics", "health", "bench", "steerability", "lifetime_ratio", "numerics",
    "deviations", "wall_clock", "verdict",
]  # fmt: skip
CRITERIA = ["storm_count", "lifetime", "forecast", "occupancy", "cost", "latitudes"]


def _write_small(tmp_path: Path) -> Path:
    f = tmp_path / "small.json"
    f.write_text(json.dumps(SMALL))
    return f


def test_cli_smoke_writes_every_output(tmp_path: Path) -> None:
    out = tmp_path / "study"
    rc = w0.main(
        [
            "--params", str(_write_small(tmp_path)), "--steps", "300", "--spin-up", "100",
            "--uniform-tracers", "500", "--floor-held", "--out", str(out), "--no-plots", "--quiet",
        ]
    )  # fmt: skip
    assert rc == 0
    for name in STUDY_FILES:
        assert (out / name).is_file(), name
    m = json.loads((out / "metrics.json").read_text())
    assert list(m) == METRIC_KEYS
    assert list(m["criteria"]) == CRITERIA
    for c in m["criteria"].values():
        assert {"value", "threshold", "pass", "detail"} <= set(c)
    assert m["criteria"]["cost"]["pass"] is None and m["bench"] is None  # not measured
    assert m["criteria"]["occupancy"]["pass"] is True
    assert {"hz0", "hz1_11", "hz12"} <= set(m["criteria"]["occupancy"]["value"])
    d = m["diagnostics"]
    assert d["tracers"]["tracers"] == 500 and d["tracers_floor_held"]["floor_held"] is True
    assert d["tracers"]["device_peak_check"][0][0] == d["tracers"]["bands"]["hz0"]["peak"]
    for key in ("waves", "jets", "storm_climate", "hyperbolicity", "latitudes", "forecast"):
        assert key in d
    assert m["health"]["pass"] and m["health"]["counters"]["n_safety"] == 0
    assert m["run"]["steps"] == 300 and m["run"]["window"] == [100, 300] and "wall_clock_of_run" in m["run"]
    assert any(e["label"] == "run length" and e["steps"] == 300 for e in m["wall_clock"])
    ts = (out / "timeseries.csv").read_text().splitlines()
    assert ts[0].startswith("t,n_storms_north") and len(ts) == 1 + 300 // 25
    assert "occ_hz0" in ts[0] and "fh_occ_hz12" in ts[0] and "zq3" in ts[0]
    pj = json.loads((out / "params.json").read_text())
    # a --params file is a sparse override of START_SET (the CLI's contract), not of the defaults
    expected = WeatherParams.from_dict(SMALL, base=START_SET)
    assert pj["hash"] == w0.params_hash(expected) and pj["validate"] == []
    assert pj["mlx_version"] and "cli" in pj and "derived" in pj
    report = (out / "report.md").read_text()
    assert report.startswith("# W0 run ") and "14.4" in report and "## Wall-clock" in report
    marker = json.loads((out / w0.MARKER).read_text())
    assert "report.md" in marker["files"]
    # an existing run needs --overwrite
    again = [
        "--params",
        str(_write_small(tmp_path)),
        "--steps",
        "50",
        "--spin-up",
        "25",
        "--out",
        str(out),
        "--quiet",
    ]
    assert w0.main(again) == 2
    assert w0.main([*again, "--overwrite", "--no-plots"]) == 0
    assert json.loads((out / "metrics.json").read_text())["run"]["steps"] == 50


BAD_ARGUMENTS = [
    ["--set", "gamma=40"],
    ["--set", "bogus=1"],
    ["--set", "no_equals_sign"],
    ["--spin-up", "100"],
    ["--tracers", "x"],
    ["--tracers", "x", "--from", "0", "--uniform-tracers", "5"],
    ["--floor-held"],
    ["--detect-every", "0"],
    ["--frames-every", "30"],
    ["--impulse", "30,1,2,3,4,5,6"],  # t0 is not a detection time
    ["--impulse", "50,1,2,3,4,5,6,9"],
    ["--impulse", "a,b"],
    ["--impulse-auto", "50,1,2"],
    ["--pool-shift", "50,0,1"],
    ["--pool-gain", "50,0"],
    ["--pool-gain", "500,0,0.7"],  # t0 beyond the run
    ["--gain-after", "x"],
    ["--movie-steps", "a:b"],
]


@pytest.mark.parametrize("extra", BAD_ARGUMENTS, ids=lambda e: " ".join(e))
def test_cli_rejects_bad_arguments(tmp_path: Path, capsys, extra: list[str]) -> None:
    base = ["--params", str(_write_small(tmp_path)), "--steps", "100", "--spin-up", "10", "--quiet"]
    assert w0.main([*base, *extra, "--out", str(tmp_path / "o")]) == 2
    assert capsys.readouterr().err.startswith(("error", "invalid"))
    assert not (tmp_path / "o" / "metrics.json").exists()


def test_cli_refuses_an_output_under_a_runs_directory(tmp_path: Path, capsys) -> None:
    args = ["--params", str(_write_small(tmp_path)), "--steps", "100", "--spin-up", "10", "--quiet"]
    assert w0.main([*args, "--out", str(tmp_path / "runs" / "w0_a")]) == 2
    assert "runs/" in capsys.readouterr().err


MICRO_WEATHER = {  # the weather domain of ``micro3d_cfg``: L = 48, height 24 = H1 + H2, the same wave speeds
    "n": 24,
    "dx": 2.0,
    "height": 24.0,
    "H1": 9.0,
    "H2": 15.0,
    "g_top": 0.03095911544707478,
    "g_prime": 0.017442358807800592,
    "h_min": 1.0,
    "gamma": 4.0,
    "seed_sigma": 6.0,
    "pools": [
        [12, 12, 8, 4, 1, 0, 0],
        [36, 12, 8, 4, 1, 0, 0],
        [0, 36, 8, 4, 1, 0, 0],
        [24, 36, 8, 4, 1, 0, 0],
    ],
    "zeta_th": 0.0075,
    "a_min": 4,
    "r_min": 2,
}


def _listing(root: Path) -> list[tuple[str, int, int]]:
    return sorted((str(f), f.stat().st_mtime_ns, f.stat().st_size) for f in root.rglob("*"))


def test_load_tracer_source_reads_live_rows_and_writes_nothing(tmp_path: Path, micro3d_cfg) -> None:
    run = tmp_path / "src"
    rd = RunDir.create(run, micro3d_cfg)
    s = init_state(micro3d_cfg)
    rd.save_checkpoint(s, micro3d_cfg)
    before = _listing(run)
    cfg, pos, cap = w0.load_tracer_source(run, 0)
    assert _listing(run) == before  # nothing under the run changed (spec 5)
    alive = np.array(s.alive).astype(bool)
    assert pos.shape == (int(alive.sum()), 3) and cap.shape == (pos.shape[0],) and pos.dtype == np.float32
    assert np.array_equal(pos, np.array(s.pos)[alive]) and np.array_equal(cap, np.array(s.buoy_cap)[alive])
    assert cfg.world.size == 48.0
    with pytest.raises(FileNotFoundError):
        w0.load_tracer_source(run, 123)


def test_cli_tracers_from_a_checkpoint_never_touches_the_run(tmp_path: Path, micro3d_cfg, capsys) -> None:
    run = tmp_path / "src"
    RunDir.create(run, micro3d_cfg).save_checkpoint(init_state(micro3d_cfg), micro3d_cfg)
    params = tmp_path / "micro.json"
    params.write_text(json.dumps(MICRO_WEATHER))
    before = _listing(run)
    out = tmp_path / "study"
    args = ["--params", str(params), "--steps", "75", "--spin-up", "25", "--tracers", str(run), "--from", "0"]
    assert w0.main([*args, "--out", str(out), "--no-plots", "--floor-held", "--quiet"]) == 0
    assert _listing(run) == before
    m = json.loads((out / "metrics.json").read_text())
    assert m["diagnostics"]["tracers"]["tracers"] == 8 and m["run"]["tracers"] == 8
    assert m["lifetime_ratio"]["organism"]["n_deaths"] == 0  # no events directory in this run
    assert "No deaths found" in (out / "report.md").read_text()
    # a weather domain that does not match the tracers' world is refused, and so is an output inside the run
    wrong = tmp_path / "wrong.json"
    wrong.write_text(json.dumps(SMALL))
    assert w0.main(["--params", str(wrong), *args[2:], "--out", str(tmp_path / "bad"), "--quiet"]) == 2
    assert "does not match" in capsys.readouterr().err
    assert w0.main([*args, "--out", str(run / "w0"), "--quiet"]) == 2
    assert _listing(run) == before


def test_cli_plots_and_movies(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    out = tmp_path / "plots"
    rc = w0.main(
        [
            "--params", str(_write_small(tmp_path)), "--steps", "300", "--spin-up", "100",
            "--uniform-tracers", "500", "--movie-steps", "100:300", "--out", str(out), "--quiet",
        ]
    )  # fmt: skip
    assert rc == 0
    for name in ["lat_hist", "counts", "km", "forecast", "drift", "occupancy", "jets", "hist"]:
        assert (out / f"{name}.png").stat().st_size > 1000, name
    assert sorted(p.name for p in (out / "frames").glob("*.png")) == [
        "t0000075.png",
        "t0000150.png",
        "t0000225.png",
        "t0000300.png",
    ]
    for name in ["h1", "vort", "rain", "tracers_hz0", "tracers_hz1_11", "tracers_hz12"]:
        size = (out / f"{name}.gif").stat().st_size
        assert 1000 < size < 10_000_000, name
    from PIL import Image

    with Image.open(out / "h1.gif") as im:
        assert im.n_frames == 2  # frames at 200 and 300 inside (100, 300]


def test_cli_vortex_decay_and_bench_only(tmp_path: Path) -> None:
    small = _write_small(tmp_path)
    out = tmp_path / "vortex"
    assert (
        w0.main(["--params", str(small), "--vortex-decay", "--steps", "50", "--out", str(out), "--quiet"])
        == 0
    )
    tab = json.loads((out / "vortex_decay.json").read_text())
    assert tab["n"] == 32 and set(tab["lines"]) >= {"default numerics"}
    out2 = tmp_path / "bench"
    assert (
        w0.main(
            [
                "--params",
                str(small),
                "--bench",
                "--bench-steps",
                "10",
                "--bench-m",
                "300",
                "--out",
                str(out2),
                "--quiet",
            ]
        )
        == 0
    )
    b = json.loads((out2 / "bench.json").read_text())
    assert b["m_positions"] == 300 and b["steps"] == 10 and b["debug"] is False
    assert b["B_plus_C_ms"] == pytest.approx(b["B_chained_ms"] + b["C_sampler_chained_ms"])
    assert {"A", "B", "C"} == set(b["evaluated"]) and b["A_synced_ms"]["median"] > 0
    # a study can attach the bench: the cost criterion then has a value
    out3 = tmp_path / "attach"
    attach = ["--params", str(small), "--steps", "75", "--spin-up", "25", "--out", str(out3), "--no-plots"]
    attach += ["--bench-json", str(out2 / "bench.json"), "--vortex-json", str(out / "vortex_decay.json")]
    assert w0.main([*attach, "--quiet"]) == 0
    m = json.loads((out3 / "metrics.json").read_text())
    assert (
        m["criteria"]["cost"]["value"] == pytest.approx(b["B_plus_C_ms"])
        and m["numerics"]["vortex_decay"]["n"] == 32
    )
    assert "Unforced-vortex decay" in (out3 / "report.md").read_text()


# --- the command line, review fixes ---------------------------------------------------------------------


def test_cli_attached_json_is_read_before_overwrite_and_kept(tmp_path: Path, capsys) -> None:
    """A study re-run in its own directory deletes the files its marker lists, bench.json included, so the
    attached JSON is read first and written back beside the new study (the bench is a one-off GPU run)."""
    small = _write_small(tmp_path)
    out = tmp_path / "x"
    bench_args = ["--params", str(small), "--bench", "--bench-steps", "10", "--bench-m", "300"]
    assert w0.main([*bench_args, "--out", str(out), "--quiet"]) == 0
    bench = out / "bench.json"
    first = json.loads(bench.read_text())
    args = ["--params", str(small), "--steps", "50", "--spin-up", "25", "--out", str(out), "--no-plots"]
    assert w0.main([*args, "--overwrite", "--bench-json", str(bench), "--quiet"]) == 0
    assert (
        json.loads(bench.read_text()) == first
        and "bench.json" in json.loads((out / w0.MARKER).read_text())["files"]
    )
    m = json.loads((out / "metrics.json").read_text())
    assert m["criteria"]["cost"]["value"] == pytest.approx(first["B_plus_C_ms"]) and m["bench"] == first
    capsys.readouterr()
    # an unreadable attachment is an error with return code 2, and nothing in the directory is removed
    before = sorted(f.name for f in out.iterdir())
    for bad in (tmp_path / "missing.json", tmp_path / "broken.json", tmp_path / "list.json"):
        if bad.name == "broken.json":
            bad.write_text("{")
        elif bad.name == "list.json":
            bad.write_text("[1, 2]")
        assert w0.main([*args, "--overwrite", "--bench-json", str(bad), "--quiet"]) == 2, bad.name
        assert capsys.readouterr().err.startswith("error: --bench-json"), bad.name
        assert sorted(f.name for f in out.iterdir()) == before
    assert w0.main([*args, "--overwrite", "--vortex-json", str(tmp_path / "missing.json"), "--quiet"]) == 2
    assert capsys.readouterr().err.startswith("error: --vortex-json")
    assert (out / "metrics.json").is_file()


def test_cli_default_run_survives_a_missing_matplotlib(tmp_path: Path, monkeypatch, capsys) -> None:
    """matplotlib is an optional extra: the plots module imports it on its first figure, so the run must
    skip the figures with a message, finish, and record its files, not crash after writing the report."""
    monkeypatch.setitem(sys.modules, "matplotlib", None)  # `import matplotlib` now raises ImportError
    out = tmp_path / "nompl"
    args = ["--params", str(_write_small(tmp_path)), "--steps", "75", "--spin-up", "25", "--out", str(out)]
    assert w0.main(args) == 0
    assert "skipping figures and movies" in capsys.readouterr().out
    assert (out / "report.md").is_file() and not list(out.glob("*.png")) and not list(out.glob("*.gif"))
    files = json.loads((out / w0.MARKER).read_text())["files"]
    assert "report.md" in files and "metrics.json" in files


def test_marker_lists_what_this_run_wrote_and_its_frames_only(tmp_path: Path) -> None:
    out = tmp_path / "o"
    w0.prepare_out(out, False)
    (out / "report.md").write_text("ours")
    (out / "stray.txt").write_text("not ours")
    (out / "frames").mkdir()
    (out / "frames" / "t0000100.png").write_bytes(b"png")
    (out / "frames" / "notes.txt").write_text("not ours")
    w0.finish_out(out, ["report.md"])
    assert json.loads((out / w0.MARKER).read_text())["files"] == ["frames/t0000100.png", "report.md"]
    w0.prepare_out(out, True)
    assert not (out / "report.md").exists() and not (out / "frames" / "t0000100.png").exists()
    assert (out / "stray.txt").exists() and (out / "frames" / "notes.txt").exists()


def _same(a, b, what: str = "") -> None:
    x, y = np.asarray(a), np.asarray(b)
    if ON_GPU and x.dtype.kind == "f":
        np.testing.assert_allclose(x, y, rtol=1e-6, atol=1e-6, err_msg=what)
    else:
        assert np.array_equal(x, y), what


def _tracked_sim(compile: bool) -> w0.Sim:
    """The driver's whole compiled composition: key split + step + fold, the tracer rigs (plain and
    floor-held) and a non-zero momentum forcing (an impulse from step 25 for 25 steps), with detection."""
    p = WeatherParams.from_dict({**SMALL, "seed_rate": 0.05})
    cfg = w0.uniform_cfg(p)
    rigs = [
        w0.TracerRig(p, cfg, None, None, m=300, key=1, floor_held=h, compile=compile) for h in (False, True)
    ]
    sim = w0.Sim(p, 3, momentum=True, rig=rigs[0], rig_floor_held=rigs[1], compile=compile)
    arrays = w0.impulse_arrays(p, w0.ImpulseSpec(25, p.L / 2, p.L / 4, 0.04, 0.02, 20.0, 25))
    zeros = tuple(mx.zeros((p.n, p.n), dtype=mx.float32) for _ in range(4))
    sim.at(25, lambda s: s._set_momentum(*arrays))
    sim.at(50, lambda s: s._set_momentum(*zeros))
    sim.advance(100)
    return sim


def test_compiled_driver_equals_eager_with_tracers_and_an_impulse() -> None:
    """The graphs W1 composes from this module (fused key + step + fold, the tracer step, a momentum forcing)
    give the same run compiled and eager."""
    a, b = _tracked_sim(True), _tracked_sim(False)
    for f in a.ws._fields:
        _same(getattr(a.ws, f), getattr(b.ws, f), f)
    _same(a.key, b.key, "key")
    for rig_a, rig_b in ((a.rig, b.rig), (a.rig_floor_held, b.rig_floor_held)):
        _same(rig_a.pos, rig_b.pos, "tracer positions")
        _same(rig_a.peak, rig_b.peak, "running peak")
    a.read_interval(), b.read_interval()
    _same(a.totals_cnt, b.totals_cnt, "counters")
    _same(a.totals_sum, b.totals_sum, "sums")
    _same(a.max_run, b.max_run, "maxima")
    ta, tb = a.tracker.finish(), b.tracker.finish()
    assert [(t.id, len(t.dets)) for t in ta] == [(t.id, len(t.dets)) for t in tb]
    assert float(np.abs(np.array(a.ws.u1)).max()) > 0.0 and a.totals_cnt.sum() >= 0
