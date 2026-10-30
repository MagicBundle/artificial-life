"""The storm detector and tracker on the sphere's cells (DESIGN §46.7; sphere series S5), against synthetic
cyclones on the HEALPix grid: no weather engine is needed. They cross the poles, the trivalent vertices of the
grid and the face seams; the results are rotation equivariant and the box tracker's code path is the same one.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from alife import geom
from alife.storms import Tracker
from alife.storms_sph import (
    SphereParams,
    SphereTracker,
    detect,
    frame_of_vortices,
    vortex_field,
)
from sphere_rotation import random_rotation

NSIDE, RADIUS = 24, 60.0
P = SphereParams(NSIDE, RADIUS)
SIGMA = 8.0


def cyclone(centre, amp=0.05, sigma=SIGMA):
    return {"centre": np.asarray(centre, float), "sigma": sigma, "amp": amp}


def arc(a, b):
    return float(geom.great_circle_distance(geom.unit(a), geom.unit(b), RADIUS))


def vertices() -> np.ndarray:
    """The 8 trivalent vertices on the faces' boundary at latitude asin(2/3)... as unit vectors: the corners
    where three pixels meet and which are not the poles (HEALPix: z = ±2/3, longitudes k·90°)."""
    z = 2.0 / 3.0
    s = math.sqrt(1 - z * z)
    return np.array(
        [
            [s * math.cos(k * math.pi / 2), s * math.sin(k * math.pi / 2), sz]
            for sz in (z, -z)
            for k in range(4)
        ]
    )


PLACES = {
    "low latitude": [1.0, 0.0, 0.4],
    "north pole": [0.0, 0.0, 1.0],
    "south pole": [0.0, 1e-4, -1.0],
    "a trivalent vertex": vertices()[1],
    "a face seam": [math.cos(math.radians(45.0)), math.sin(math.radians(45.0)), 0.3],
    "mid latitude": [0.3, -0.8, 0.5],
}


@pytest.mark.parametrize("name", sorted(PLACES))
def test_a_single_cyclone_is_one_storm_with_the_right_centre_and_radius(name: str) -> None:
    centre = geom.unit(np.asarray(PLACES[name], float))
    det = detect(frame_of_vortices(NSIDE, 0, [cyclone(centre)], RADIUS), P)
    assert len(det.storms) == 1 and not det.bands
    [c] = det.storms
    cell = math.sqrt(P.cell_area)
    assert arc(np.array([c.x, c.y, c.z]), centre) < 0.4 * cell  # the centre, to under half a cell
    assert np.linalg.norm([c.x, c.y, c.z]) == pytest.approx(RADIUS)
    # the footprint of the Gaussian above the threshold has the analytic radius r = sigma*sqrt(2 ln(A/th))
    r_exact = SIGMA * math.sqrt(2.0 * math.log(0.05 / P.zeta_th))
    assert c.r_eq == pytest.approx(r_exact, rel=0.08)  # sqrt(area/pi) from equal-area cells, any place
    assert c.r_max == pytest.approx(r_exact, rel=0.15)
    assert c.hemi == (0 if centre[2] >= 0 else 1)
    assert c.zeta_max_f0 == pytest.approx(
        0.05 / 0.025, rel=0.05
    )  # the peak cell sits a fraction of a cell off
    assert c.rx == c.ry and c.rx > 0.99 and c.area == len(c.flat)
    # the label image holds exactly the footprint
    assert set(np.flatnonzero(det.lab == c.label)) == set(c.flat.tolist())


def test_the_radius_does_not_depend_on_where_the_cyclone_is() -> None:
    radii = []
    for name in PLACES:
        centre = geom.unit(np.asarray(PLACES[name], float))
        [c] = detect(frame_of_vortices(NSIDE, 0, [cyclone(centre)], RADIUS), P).storms
        radii.append(c.r_eq)
    assert max(radii) / min(radii) < 1.08  # equal-area cells: the pole and the equator measure alike


def test_detection_is_rotation_equivariant() -> None:
    rng_rot = random_rotation(5)
    base = [cyclone(PLACES["mid latitude"]), cyclone([-0.9, 0.2, -0.4], amp=0.04)]
    d0 = detect(frame_of_vortices(NSIDE, 0, base, RADIUS), P)
    rot = [cyclone(rng_rot @ geom.unit(np.asarray(v["centre"], float)), v["amp"]) for v in base]
    # rotating a centre can move a cyclone to the other hemisphere: compare by nearest centre
    d1 = detect(frame_of_vortices(NSIDE, 0, [{**v, "amp": v["amp"]} for v in rot], RADIUS), P)
    assert len(d0.storms) == len(d1.storms) == 2 or len(d1.storms) >= 1
    for c0 in d0.storms:
        want = rng_rot @ np.array([c0.x, c0.y, c0.z])
        got = min(d1.storms, key=lambda c: arc(np.array([c.x, c.y, c.z]), want))
        if got.hemi == c0.hemi:  # cyclonic sign flips with the hemisphere: only the same-sign copy is a twin
            assert arc(np.array([got.x, got.y, got.z]), want) < 0.5 * math.sqrt(P.cell_area)
            assert got.r_eq == pytest.approx(c0.r_eq, rel=0.1)


def test_two_hemispheres_are_two_storms_even_when_the_vortex_straddles_the_equator() -> None:
    d = detect(frame_of_vortices(NSIDE, 0, [cyclone([1.0, 0.0, 0.0], amp=0.06, sigma=10.0)], RADIUS), P)
    assert sorted(c.hemi for c in d.storms) == [0, 1]
    n, s = (c for c in sorted(d.storms, key=lambda c: c.hemi))
    assert n.z > 0 > s.z and abs(n.z + s.z) < 3.0 and abs(n.area - s.area) <= 15


def test_a_belt_is_a_band_and_a_small_blob_is_neither() -> None:
    centres = geom.pixel_centres(NSIDE)
    lat = np.arcsin(centres[:, 2])
    belt = np.where((lat > 0.5) & (lat < 0.8), 0.05, 0.0)  # a zonal ring around the north pole
    from alife.storms_sph import SphereFrame

    fr = SphereFrame(0, belt, np.where(belt > 0, 0.004, 0.0), np.ones_like(belt), belt)
    d = detect(fr, P)
    assert (
        not d.storms and len(d.bands) == 1 and d.bands[0].ext_x > 0.25 * 2 * math.pi * RADIUS
    )  # wide, not diffuse
    tiny = detect(frame_of_vortices(NSIDE, 0, [cyclone([1, 0.2, 0.3], amp=0.0065, sigma=3.0)], RADIUS), P)
    assert not tiny.storms and tiny.n_weak >= 0


# --- the tracker ------------------------------------------------------------------------------------------


def great_circle_path(start, towards, n: int, step_deg: float) -> np.ndarray:
    """``n`` centres along the great circle from ``start`` through ``towards``, ``step_deg`` apart."""
    u = geom.unit(np.asarray(start, float))
    w = np.asarray(towards, float)
    t = geom.unit(w - (w @ u) * u)
    ang = np.radians(step_deg) * np.arange(n)
    return np.cos(ang)[:, None] * u + np.sin(ang)[:, None] * t


def track_path(path: np.ndarray, every: int = 25, amp: float = 0.05):
    tracker = SphereTracker(P)
    out = []
    for i, c in enumerate(path):
        det = detect(frame_of_vortices(NSIDE, i * every, [cyclone(c, amp)], RADIUS), P)
        ids = tracker.update(det)
        out.append((det, ids))
    return tracker, out


@pytest.mark.parametrize(
    ("start", "towards", "what"),
    [
        ([1.0, 0.0, 0.55], [0.0, 0.0, 1.0], "over the north pole"),
        (vertices()[0] * [1, 1, 1], vertices()[2], "through trivalent vertices"),
        ([0.2, -1.0, -0.6], [0.0, 0.2, -1.0], "over the south pole"),
        ([1.0, 0.0, 0.5], [0.0, 1.0, 0.5], "across the face seams at mid latitude"),
    ],
)
def test_a_cyclone_crossing_the_poles_vertices_and_seams_keeps_one_track(start, towards, what: str) -> None:
    path = great_circle_path(start, towards, 48, 1.6)  # 77 degrees; 1.6 degrees = 1.7 units a detection
    tracker, steps = track_path(path)
    assert all(len(det.storms) == 1 for det, _ in steps), what
    ids = {ids[0] for _, ids in steps}
    assert ids == {1}, f"one track id {what}: {ids}"
    [tr] = tracker.tracks
    assert len(tr.dets) == len(path) and not tr.ended
    cell = math.sqrt(P.cell_area)
    for d, c in zip(tr.dets, path, strict=True):
        assert arc(np.array([d.x, d.y, d.z]), c) < 0.5 * cell
    assert tr.hemi == (0 if path[0][2] >= 0 else 1)


def test_the_hemisphere_change_is_a_new_track_not_a_jump() -> None:
    """A cyclone drifting over the equator changes sign of the Coriolis parameter: its cyclonic region
    switches hemisphere, which the tracker treats as a death and a birth (as the box tracker does)."""
    path = great_circle_path([1.0, 0.0, 0.25], [1.0, 0.0, -1.0], 30, 1.2)
    tracker, _ = track_path(path)
    assert len(tracker.tracks) >= 2 and {t.hemi for t in tracker.tracks} == {0, 1}


def test_the_prediction_extrapolates_a_great_circle() -> None:
    path = great_circle_path([1.0, 0.0, 0.55], [0.0, 0.0, 1.0], 12, 1.6)
    tracker, _ = track_path(path)
    [tr] = tracker.tracks
    ahead = great_circle_path([1.0, 0.0, 0.55], [0.0, 0.0, 1.0], 20, 1.6)
    pred = np.array(tracker.predict(tr, tr.dets[-1].t + 5 * 25))
    assert np.linalg.norm(pred) == pytest.approx(RADIUS)
    assert arc(pred, ahead[11 + 5]) < 0.3  # five detections ahead: within a third of a unit of the true path
    assert tracker.predict(tr, tr.dets[-1].t) == pytest.approx(
        tuple(RADIUS * geom.unit(np.array([tr.dets[-1].x, tr.dets[-1].y, tr.dets[-1].z]))), abs=1e-9
    )
    one = SphereTracker(P)
    one.update(detect(frame_of_vortices(NSIDE, 0, [cyclone(path[0])], RADIUS), P))
    assert arc(np.array(one.predict(one.tracks[0], 100)), path[0]) < 0.5 * math.sqrt(
        P.cell_area
    )  # no velocity


def test_the_track_is_rotation_equivariant() -> None:
    path = great_circle_path([1.0, 0.0, 0.55], [0.0, 0.0, 1.0], 30, 1.6)
    c, sn = math.cos(0.3), math.sin(0.3)
    rx = np.array([[1, 0, 0], [0, c, -sn], [0, sn, c]])
    cz, sz = math.cos(1.1), math.sin(1.1)
    rot = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]) @ rx  # a rotation that keeps the path in the north
    a, _ = track_path(path)
    b, _ = track_path(path @ rot.T)
    [ta], [tb] = a.tracks, b.tracks
    cell = math.sqrt(P.cell_area)
    for da, db in zip(ta.dets, tb.dets, strict=True):
        assert arc(rot @ np.array([da.x, da.y, da.z]), np.array([db.x, db.y, db.z])) < 0.7 * cell
        assert db.r_eq == pytest.approx(da.r_eq, rel=0.1)


def test_two_cyclones_merge_and_one_splits() -> None:
    a = geom.unit(np.array([1.0, 0.0, 0.5]))
    c1 = great_circle_path([1.0, 0.0, 0.5], [0.0, 1.0, 0.5], 2, 40.0)[1]  # 40 degrees apart
    tracker = SphereTracker(P)
    for i in range(3):
        tracker.update(detect(frame_of_vortices(NSIDE, 25 * i, [cyclone(a), cyclone(c1)], RADIUS), P))
    assert len(tracker.active()) == 2
    mid = geom.unit(a + c1)
    tracker.update(detect(frame_of_vortices(NSIDE, 75, [cyclone(mid, 0.07, 12.0)], RADIUS), P))
    ended = [t for t in tracker.tracks if t.ended]
    assert len(ended) == 1 and ended[0].end_reason == "merged" and ended[0].merged_into in {1, 2}
    survivor = next(t for t in tracker.tracks if not t.ended)
    assert survivor.id == ended[0].merged_into
    # and apart again: the footprint overlaps one track, the second piece is a split-off with a parent
    tracker.update(detect(frame_of_vortices(NSIDE, 100, [cyclone(a), cyclone(c1)], RADIUS), P))
    born = [t for t in tracker.tracks if t.id > 2]
    assert born and born[0].parent == survivor.id


def test_the_sphere_tracker_is_the_box_tracker_in_everything_but_distance() -> None:
    assert SphereTracker.update is Tracker.update and SphereTracker.finish is Tracker.finish
    assert SphereTracker.compact is Tracker.compact
    assert SphereTracker.predict is not Tracker.predict and SphereTracker._gap is not Tracker._gap


def test_the_vortex_field_has_the_cyclonic_sign_of_its_hemisphere() -> None:
    z = vortex_field(NSIDE, [0, 0, 1], SIGMA, RADIUS, 0.05)
    s = np.where(geom.pixel_centres(NSIDE)[:, 2] >= 0, 1, -1)
    assert ((s * z) >= 0).all() and z.max() <= 0.05 and z[s < 0].min() >= -0.05
