"""The exact coverage of the hash stencils, the hash Nside and the cell capacity derived from it, and the
equal-volume layers (``alife.geom``; DESIGN §46.1 and §46.3, sphere series S0).

The coverage of cell p is the geodesic distance between the boundary of p and the boundary of the union of its
stencil: every ball of that angular radius around a point of p lies inside the stencil. The tests check the
numbers of the sphere spec, the method against the reviewer's sampled implementation (``hp_oracle``), the
definition itself by brute force around the witness points, and the search guarantee (F13) at its margin.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

import hp_oracle as hp
from alife import geom

R_SENSE, R_BODY = 6.0, 1.0
# (name, Rp, H, FZ, Nf, margin of the exact reach-2 coverage over the rigorous angle, G_z, min C)
PRESETS = [
    ("tiny3s", 36.0, 16.0, 8, 8, 0.120, 5, 28),
    ("mini3s", 48.0, 24.0, 12, 11, 0.068, 8, 27),
    ("64k3s", 90.0, 40.0, 18, 21, 0.023, 13, 25),
]


def _chord_to_angle(chord: np.ndarray) -> np.ndarray:
    return 2.0 * np.arcsin(chord / 2.0)


@pytest.mark.parametrize(
    "nside,reach,expected",
    [(21, 2, 1.385), (11, 2, 1.406), (8, 2, 1.421), (21, 1, 0.685), (4, 1, 0.711)],
)
def test_exact_coverage_reproduces_the_reviewers_numbers(nside: int, reach: int, expected: float) -> None:
    # the review quotes chords of a sampled boundary: the geodesic angle is 0.1 to 0.2 % larger
    assert geom.coverage(nside, reach).min_rel == pytest.approx(expected, abs=0.004)


def test_coverage_falls_with_nside_and_reach_two_is_about_twice_reach_one() -> None:
    r2 = [geom.coverage(n, 2).min_rel for n in (4, 8, 11, 16)]
    assert all(a > b for a, b in itertools.pairwise(r2))  # 1.466 .. 1.385 and below
    assert all(1.3 < v < 1.5 for v in r2)
    assert geom.coverage(11, 1).min_rel < 0.72


@pytest.mark.parametrize("nside,reach", [(5, 2), (6, 1), (8, 2)])
def test_exact_coverage_equals_the_independent_sampled_one_cell_by_cell(nside: int, reach: int) -> None:
    """The reviewer's method samples both boundaries (m = 24 points per edge) and takes the closest pair of
    samples, in chords: an upper bound of the exact distance that is tight to O(1/m²). Per cell, the exact
    geodesic value must be at most the sampled one (converted to an angle) and within 5e-4 sides of it."""
    sampled, size = hp.sampled_coverage(nside, reach)
    cov = geom.coverage(nside, reach)
    mine = cov.angle / cov.unit
    theirs = _chord_to_angle(sampled * cov.unit) / cov.unit
    assert (mine <= theirs + 1e-9).all()
    assert (theirs - mine).max() < 5e-4
    assert size.max() == (geom.edge_stencil(nside, reach) >= 0).sum(axis=1).max()


def _circle(centre: np.ndarray, radius: float, n: int = 720) -> np.ndarray:
    """``n`` points at geodesic distance ``radius`` from the unit vector ``centre``."""
    e1 = np.cross(centre, [0.3, 0.5, 0.8])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(centre, e1)
    t = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return centre * math.cos(radius) + (np.outer(np.cos(t), e1) + np.outer(np.sin(t), e2)) * math.sin(radius)


@pytest.mark.parametrize("nside,reach", [(8, 2), (11, 1), (13, 2)])
def test_the_witnesses_are_the_nearest_boundary_points(nside: int, reach: int) -> None:
    """``witness_a`` lies on the boundary of the cell, ``witness_b`` on the boundary of the stencil union (a
    tiny circle around each has points on both sides), and their distance is the coverage."""
    cov = geom.coverage(nside, reach)
    tb = geom.tables(nside)
    st = geom.edge_stencil(nside, reach)
    ang = np.arctan2(
        np.linalg.norm(np.cross(cov.witness_a, cov.witness_b), axis=1), (cov.witness_a * cov.witness_b).sum(1)
    )
    assert np.abs(ang - cov.angle).max() < 1e-12
    pick = np.unique(np.concatenate([[cov.worst], np.random.default_rng(0).integers(0, tb.npix, 60)]))
    for p in pick:
        members = st[p][st[p] >= 0]
        around_a = geom.ang2pix(_circle(cov.witness_a[p], 1e-7), nside)
        assert (around_a == p).any() and (around_a != p).any()
        around_b = np.isin(geom.ang2pix(_circle(cov.witness_b[p], 1e-7), nside), members)
        assert around_b.any() and not around_b.all()


@pytest.mark.parametrize("nside,reach", [(8, 2), (21, 2), (11, 1)])
def test_a_ball_of_the_coverage_radius_stays_inside_the_stencil(nside: int, reach: int) -> None:
    """The definition by brute force: around the nearest point of the worst cell (and of a few others) the
    circle of radius coverage − 1e-6 lies in the stencil in all 4 000 directions; the circle 1 % of a cell
    side wider does not."""
    cov = geom.coverage(nside, reach)
    st = geom.edge_stencil(nside, reach)
    rng = np.random.default_rng(1)
    for p in [cov.worst, *rng.choice(len(cov.angle), 5, replace=False)]:
        members = st[p][st[p] >= 0]
        a = cov.witness_a[p]
        inside = np.isin(geom.ang2pix(_circle(a, cov.angle[p] - 1e-6, 4000), nside), members)
        outside = np.isin(geom.ang2pix(_circle(a, cov.angle[p] + 0.01 * cov.unit, 4000), nside), members)
        assert inside.all() and not outside.all()


def test_the_rigorous_angle_is_the_worst_case_of_two_bodies_closer_than_r_above_the_floor() -> None:
    """Two centres closer than R, both at least r_lo from the planet centre, subtend at most
    2·asin(R/(2·r_lo)): the maximum over a grid of radii (equal radii at the floor are the worst case)."""
    r_lo = 37.0
    rho = np.linspace(r_lo, r_lo + 6, 241)
    ri, rj = np.meshgrid(rho, rho)
    c = (ri**2 + rj**2 - R_SENSE**2) / (2 * ri * rj)  # law of cosines for a separation exactly R
    ok = np.abs(ri - rj) <= R_SENSE
    theta = np.where(ok, np.arccos(np.clip(c, -1, 1)), 0.0)
    assert theta.max() == pytest.approx(geom.rigorous_angle(36.0, R_BODY, R_SENSE), rel=1e-12)
    i, j = np.unravel_index(theta.argmax(), theta.shape)
    assert ri[i, j] == rj[i, j] == r_lo


@pytest.mark.parametrize("name,rp,h,fz,nf,margin,gz,c_min", PRESETS)
def test_the_hash_nside_of_the_presets_is_the_field_nside(
    name: str, rp: float, h: float, fz: int, nf: int, margin: float, gz: int, c_min: int
) -> None:
    angle = geom.rigorous_angle(rp, R_BODY, R_SENSE)
    nh = geom.derive_nh(rp, R_BODY, R_SENSE)
    assert nh == nf
    assert geom.coverage(nh, 2).min_angle >= angle > geom.coverage(nh + 1, 2).min_angle  # the largest Nside
    assert geom.hash_margin(nh, angle) == pytest.approx(margin, abs=0.004)
    assert geom.hash_layers(h, R_SENSE)[0] == gz and geom.hash_layers(h, R_SENSE)[1] >= R_SENSE / 2
    assert geom.min_cell_capacity(rp, h, R_SENSE, nh) == c_min


def test_no_hash_nside_for_a_planet_the_sensing_radius_cannot_cover() -> None:
    assert geom.derive_nh(5.0, 1.0, 6.0) == 0  # a ball larger than the planet: not even Nside 2 covers it


@pytest.mark.parametrize("name,rp,h,fz,nf,margin,gz,c_min", PRESETS[:2])
def test_the_search_finds_every_neighbour_at_the_margin(
    name: str, rp: float, h: float, fz: int, nf: int, margin: float, gz: int, c_min: int
) -> None:
    """F13 at the margin: put an agent at the extreme point of the worst cell, on the floor, and a second
    one at sensing.radius − 1e-3 from it in the direction of the nearest stencil boundary (and in 720 other
    directions):
    the second one's cell is in the first one's stencil. With the radius chosen so that the rigorous
    angle equals the
    exact coverage, one more millimetre of reach leaves the stencil."""
    nh = geom.derive_nh(rp, R_BODY, R_SENSE)
    cov = geom.coverage(nh, 2)
    st = geom.tables(nh).stencil2
    r_lo = rp + R_BODY
    p = cov.worst
    a, b = cov.witness_a[p], cov.witness_b[p]
    toward = b - a * (a @ b)
    toward /= np.linalg.norm(toward)
    inward = geom.pixel_centres(nh)[p] - a
    a_in = a + 1e-7 * inward / np.linalg.norm(inward)  # strictly inside the cell
    a_in /= np.linalg.norm(a_in)
    ci = int(geom.ang2pix(a_in, nh))
    members = st[ci][st[ci] >= 0]

    def partner(sense: float, direction: np.ndarray) -> np.ndarray:
        theta = 2.0 * math.asin(sense / (2.0 * r_lo))  # two bodies at the floor, `sense` apart
        u = a_in * math.cos(theta) + direction * math.sin(theta)
        return u / np.linalg.norm(u)

    pi_ = r_lo * a_in
    for sense in (R_SENSE - 1e-3, R_SENSE * 0.5):
        assert np.linalg.norm(r_lo * partner(sense, toward) - pi_) < R_SENSE
        assert int(geom.ang2pix(partner(sense, toward), nh)) in members
    e1 = np.cross(a_in, toward)
    directions = [
        toward * math.cos(t) + e1 * math.sin(t) for t in np.linspace(0, 2 * math.pi, 720, endpoint=False)
    ]
    for d in directions:
        assert int(geom.ang2pix(partner(R_SENSE - 1e-3, d), nh)) in members
    # tightness: with the sensing radius whose rigorous angle is exactly the coverage, 1 mm less is still
    # found in
    # every direction; 1 % more is not (some direction leaves the stencil)
    sense_star = 2.0 * r_lo * math.sin(cov.angle[p] / 2.0)
    assert all(int(geom.ang2pix(partner(sense_star - 1e-3, d), nh)) in members for d in directions)
    assert not all(int(geom.ang2pix(partner(sense_star * 1.01, d), nh)) in members for d in directions)


@pytest.mark.parametrize("name,rp,h,fz,nf,margin,gz,c_min", PRESETS)
def test_equal_volume_layers(
    name: str, rp: float, h: float, fz: int, nf: int, margin: float, gz: int, c_min: int
) -> None:
    r = geom.layer_radii(rp, h, fz)
    assert r[0] == pytest.approx(rp) and r[-1] == pytest.approx(rp + h)
    vol = (4 * math.pi / 3) * np.diff(r**3)
    assert vol.max() / vol.min() == pytest.approx(1.0, abs=1e-12)  # equal shell volume per layer
    assert np.all(np.diff(r) > 0) and np.diff(r)[0] > np.diff(r)[-1]  # thinner upward
    cell = geom.layer_volume(rp, h, fz, nf)
    assert cell == pytest.approx(vol[0] / geom.npix(nf), rel=1e-12)
    assert geom.layer_edges(rp, h, fz)[0] == pytest.approx(0.0, abs=1e-9)
    # the sizing table of the spec: cell volumes 63 to 65, thickness of the floor and the lid layer
    assert 62.5 < cell < 65.5
    assert np.diff(r)[0] == pytest.approx({"tiny3s": 2.8, "mini3s": 3.0, "64k3s": 3.2}[name], abs=0.06)
    assert np.diff(r)[-1] == pytest.approx({"tiny3s": 1.5, "mini3s": 1.4, "64k3s": 1.6}[name], abs=0.06)
    # the layer of a distance, just above and just below each interface; the clamp holds at both ends
    assert np.array_equal(geom.layer_of(r[:-1] * (1 + 1e-12), rp, h, fz), np.arange(fz))
    assert np.array_equal(geom.layer_of(r[1:-1] * (1 - 1e-12), rp, h, fz), np.arange(fz - 1))
    assert geom.layer_of(np.array([rp - 3.0, rp + h + 3.0]), rp, h, fz).tolist() == [0, fz - 1]


def test_cell_capacity_rule_follows_the_lid_cell() -> None:
    nh = 8
    v = geom.lid_cell_volume(36.0, 16.0, R_SENSE, nh)
    assert v == pytest.approx(geom.pixel_solid_angle(nh) * 52.0**2 * (16.0 / 5), rel=1e-12)
    assert geom.min_cell_capacity(36.0, 16.0, R_SENSE, nh) == math.ceil(0.167 * v) + 4
