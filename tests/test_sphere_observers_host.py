"""The host half of the sphere's observers (DESIGN §46.7; sphere series S5): directions, the coarse cell graph,
the territory map against a brute-force reference, and the rule set of version 4.

Pure numpy and eager MLX on small inputs; the full observer run is ``test_sphere_observers_run.py``.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from alife import geom, groups
from alife.game import place, rules
from alife.tracking import Dominance
from sphere_rotation import random_rotation


def _unit(rng: np.random.Generator, n: int) -> np.ndarray:
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


# --- directions --------------------------------------------------------------------------------------------


def test_latlon_round_trip_and_the_poles() -> None:
    rng = np.random.default_rng(0)
    u = _unit(rng, 500)
    lat, lon = geom.to_latlon(u)
    assert lat.min() >= -90 and lat.max() <= 90 and lon.min() >= -180 and lon.max() < 180
    np.testing.assert_allclose(geom.from_latlon(lat, lon), u, atol=1e-12)
    lat0, lon0 = geom.to_latlon(np.array([[0.0, 0.0, 5.0], [3.0, 0.0, 0.0], [0.0, -2.0, 0.0]]))
    np.testing.assert_allclose(lat0, [90.0, 0.0, 0.0])
    np.testing.assert_allclose(lon0, [0.0, 0.0, -90.0], atol=1e-12)


def test_mean_direction_is_rotation_equivariant_and_has_the_right_resultant() -> None:
    rng = np.random.default_rng(1)
    pts = _unit(rng, 40) * rng.uniform(36, 52, (40, 1))
    w = rng.uniform(0.5, 2.0, 40)
    d, r = geom.mean_direction(pts, w)
    rot = random_rotation(3)
    d2, r2 = geom.mean_direction(pts @ rot.T, w)
    np.testing.assert_allclose(d2, rot @ d, atol=1e-12)
    assert abs(r - r2) < 1e-12 and 0.0 <= r <= 1.0
    # a tight cluster around a direction: its mean is that direction and R is near 1
    c = _unit(rng, 1)[0]
    near = c + 0.01 * rng.normal(size=(30, 3))
    dn, rn = geom.mean_direction(near)
    assert np.dot(dn, c) > 0.9999 and rn > 0.999
    # the cancelling pair: the first point's direction, R = 0
    dz, rz = geom.mean_direction(np.array([[1.0, 0, 0], [-1.0, 0, 0]]))
    assert rz < 1e-12 and np.allclose(dz, [1, 0, 0])
    with pytest.raises(ValueError):
        geom.mean_direction(np.zeros((0, 3)))


def test_great_circle_distance_small_antipodal_and_text() -> None:
    a = np.array([1.0, 0.0, 0.0])
    assert geom.great_circle_distance(a, a * 7, 10.0) == pytest.approx(0.0, abs=1e-12)
    assert geom.great_circle_distance(a, -a, 10.0) == pytest.approx(math.pi * 10.0)
    b = np.array([math.cos(1e-9), math.sin(1e-9), 0.0])
    assert geom.great_circle_distance(a, b, 1.0) == pytest.approx(1e-9, rel=1e-6)  # arctan2, not arccos
    assert geom.place_text(np.array([0.0, 0.0, -45.0]), 36.0) == "90.0S 0.0E h9.0"
    assert geom.place_text(np.array([1.0, 1.0, 1.0])) == "35.3N 45.0E"
    assert geom.place_text(np.array([-1.0, -1.0, 0.0])) == "0.0N 135.0W"


# --- the cell graph ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("nside", [2, 3, 5, 9])
def test_components_on_the_coarse_graph_agree_with_a_breadth_first_search(nside: int) -> None:
    rng = np.random.default_rng(nside)
    n = geom.npix(nside)
    nb = geom.coarse_graph(nside)
    assert nb.shape == (n, 4) and (nb >= 0).all()
    for density in (0.15, 0.4, 0.6):
        member = rng.random(n) < density
        got = geom.label_components(nside, member)
        want = np.full(n, -1)
        for start in range(n):  # the reference: a queue, smallest pixel id of each component
            if not member[start] or want[start] >= 0:
                continue
            queue, seen = [start], {start}
            while queue:
                p = queue.pop()
                for q in nb[p]:
                    if member[q] and q not in seen:
                        seen.add(int(q))
                        queue.append(int(q))
            for p in seen:
                want[p] = start
        np.testing.assert_array_equal(got, want)
    np.testing.assert_array_equal(geom.label_components(nside, np.ones(n, bool)), np.zeros(n, int))
    assert (geom.label_components(nside, np.zeros(n, bool)) == -1).all()
    with pytest.raises(ValueError):
        geom.label_components(nside, np.ones(n + 1, bool))


# --- the territory map on the cell graph ---------------------------------------------------------------------


def _reference_dominance(counts: np.ndarray, ids: np.ndarray, nb: np.ndarray, k: int, min_count: int):
    """The smoothed dominance map written out cell by cell (first update: every column starts at its steady
    state ``x << k``), and the borders against each table column."""
    x = counts.astype(np.int64) << 8
    e = x << k
    owner = np.full(len(e), -1, np.int64)
    purity = np.zeros(len(e), np.int64)
    for c in range(len(e)):
        total = int(e[c].sum())
        top = int(np.argmax(e[c]))  # the first maximum: ties to the lower column
        if top < len(ids) and total >= (min_count << 8 << k) and total > 0:
            owner[c] = ids[top]
            purity[c] = int(e[c, top]) * 65536 // total
    border = np.zeros(nb.shape, bool)
    for c in range(len(e)):
        for j in range(nb.shape[1]):
            o = owner[nb[c, j]]
            border[c, j] = owner[c] != o and (owner[c] >= 0 or o >= 0)
    return owner, purity, border


def test_the_territory_map_on_the_coarse_graph_matches_a_brute_force_reference() -> None:
    nside, rng = 5, np.random.default_rng(7)
    n, nb = geom.npix(nside), geom.coarse_graph(nside)
    ids = np.array([11, 4, 29])
    counts = rng.integers(0, 6, (n, ids.size + 1))
    counts[rng.random(n) < 0.3] = 0
    dom = Dominance((n, 1), k=2, min_count=2, graph=nb)
    got = dom.update(ids, counts)
    owner, purity, border = _reference_dominance(counts, ids, nb, 2, 2)
    np.testing.assert_array_equal(got.owner, owner)
    np.testing.assert_array_equal(got.purity_q16, purity)
    np.testing.assert_array_equal(got.border, border)
    assert got.border.shape == (n, 4) and (owner >= 0).sum() > 10
    # the state round-trips: a restored map answers the same
    again = Dominance((n, 1), k=2, min_count=2, graph=nb, state=dom.state()).current()
    np.testing.assert_array_equal(again.owner, owner)
    with pytest.raises(ValueError, match="graph"):
        Dominance((n, 1), k=2, min_count=2, graph=nb[:-1])


def test_a_graph_map_with_box_arguments_is_the_box_map() -> None:
    """No graph: the lattice path is unchanged (east and north borders, shape (C, 2))."""
    dom = Dominance((4, 3), k=1, min_count=0)
    out = dom.update(np.array([5]), np.ones((12, 2), np.int64))
    assert out.border.shape == (12, 2)


def test_sphere_cell_agrees_with_the_float64_pixel_except_next_to_a_boundary() -> None:
    rng = np.random.default_rng(3)
    pos = _unit(rng, 4000) * rng.uniform(36, 52, (4000, 1))
    for nside in (9, 32):
        got = np.array(groups.sphere_cell(mx.array(pos.astype(np.float32)), nside))
        want = geom.ang2pix(pos, nside)
        bad = got != want
        assert bad.mean() < 5e-3
        touch = geom.tables(nside).touch
        for g, w in zip(got[bad], want[bad], strict=True):
            assert g in touch[w]  # a float32 rounding lands in a neighbour, never farther
    with pytest.raises(ValueError, match="int32"):
        groups.sphere_cell(mx.zeros((2, 3)), 14000)


# --- where things are, for the game layer ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny3s_cfg():
    from alife.config import load_config

    return load_config(preset="tiny3s", overrides=["brain.weight_dtype=float32"])


def test_place_centroid_height_distance_and_text_on_a_sphere(tiny3s_cfg) -> None:
    cfg = tiny3s_cfg
    rp = cfg.world.radius
    rng = np.random.default_rng(4)
    pts = _unit(rng, 50) * rng.uniform(rp + 1, rp + 14, (50, 1))
    c = np.array(place.centroid(pts, cfg))
    assert c.shape == (3,)
    d, _ = geom.mean_direction(pts)
    np.testing.assert_allclose(c / np.linalg.norm(c), d, atol=1e-12)
    assert np.linalg.norm(c) == pytest.approx(np.linalg.norm(pts, axis=1).mean())
    rot = random_rotation(5)
    np.testing.assert_allclose(np.array(place.centroid(pts @ rot.T, cfg)), rot @ c, atol=1e-9)  # equivariant
    np.testing.assert_allclose(place.heights(pts, cfg), np.linalg.norm(pts, axis=1) - rp)
    assert place.mean_height(pts, cfg) == pytest.approx(float(np.mean(np.linalg.norm(pts, axis=1) - rp)))
    a, b = pts[0], pts[1]
    ang = geom.great_circle_distance(a, b)
    assert place.distance(a, b, cfg) == pytest.approx(ang * 0.5 * (np.linalg.norm(a) + np.linalg.norm(b)))
    assert place.text(np.array([rp, 0.0, 0.0]), cfg) == "0.0N 0.0E"


def test_place_in_a_box_world_is_the_old_expressions() -> None:
    from alife.config import load_config
    from alife.tracking import circular_mean

    cfg = load_config(preset="tiny3d")
    pts = np.random.default_rng(1).uniform(0, cfg.world.size, (30, 3))
    c = place.centroid(pts, cfg)
    assert len(c) == 2 and c[0] == circular_mean(pts[:, 0], cfg.world.size)
    assert place.mean_height(pts, cfg) == float(pts[:, 2].mean())
    d = place.distance(pts[0], pts[1], cfg)
    off = pts[1, :2] - pts[0, :2]
    off -= cfg.world.size * np.rint(off / cfg.world.size)
    assert d == pytest.approx(float(np.linalg.norm(off)))
    assert place.text(np.array([12.4, 3.6]), cfg) == "(12, 4)"


# --- the rule set of version 4 --------------------------------------------------------------------------------


def test_rules_v4_is_v3_on_the_sphere_columns() -> None:
    v3, v4 = rules.RULES_V3, rules.RULES_V4
    assert v4.sphere and not v3.sphere and v4.digest() != v3.digest()
    assert rules.intent_version(v4) == rules.INTENT_VERSION_SPHERE == 4
    assert rules.intent_version(v3) == rules.INTENT_VERSION == 3
    # every threshold is version 3's: only the sphere flag differs
    import dataclasses

    assert dataclasses.replace(v4, sphere=False) == v3
    lay3, lay4 = rules.arch_layout(3, v3), rules.arch_layout(3, v4)
    assert lay3.vrad is None and lay4.vrad == lay3.width and lay4.width == lay3.width + 2
    assert rules.n_arch_cols(3, v4) == 16
    assert rules.RULES_V3.digest() == "3515d96e" + rules.RULES_V3.digest()[8:]  # the box pin is untouched
    with pytest.raises(ValueError, match="needs vertical"):
        dataclasses.replace(v4, vertical=False, floater_lift=(0, 1), sinker_lift=(0, 1))


def test_floater_and_sinker_read_the_radial_velocity_on_a_sphere(tiny3s_cfg) -> None:
    cfg, rs = tiny3s_cfg, rules.RULES_V4
    lay = rules.arch_layout(3, rs)
    m6 = np.zeros((4, lay.width), np.int32)
    m6[:, lay.zpos] = 200  # high above the floor
    m6[:, lay.lift] = np.array([100, 100, 5, 5])  # hover lift (>= 77) or little lift (< 26)
    # a planet-frame z velocity that is NOT radial must not matter: the Cartesian halves carry a big z mean
    m6[:, lay.v_pos + 2] = 400
    m6[2, lay.vrad + 1] = 60  # row 2 descends at the radial speed 60 (>= sinker_vz = 19)
    m6[1, lay.vrad] = 50  # row 1 climbs (|vz| > floater_vz = 19)
    m6[:, rules.ArchCol.SPEED] = 1
    body = mx.zeros((4,), dtype=mx.int32)
    out = np.array(rules.archetype_matches(mx.array(m6), mx.zeros((4, 4, 2), dtype=mx.int32), cfg, rs, body))
    floater, sinker = out[:, rules.Archetype.FLOATER.value - 1], out[:, rules.Archetype.SINKER.value - 1]
    assert floater.tolist() == [True, False, False, False]
    assert sinker.tolist() == [False, False, True, False]
    # the same numbers read as the box would (the z half) give the opposite: the test is not vacuous
    box_like = np.array(
        rules.archetype_matches(
            mx.array(m6[:, : lay.vrad]), mx.zeros((4, 4, 2), dtype=mx.int32), cfg, rules.RULES_V3, body,
            lay=rules.arch_layout(3, rules.RULES_V3),
        )
    )  # fmt: skip
    assert box_like[:, rules.Archetype.FLOATER.value - 1].tolist() != floater.tolist()


def test_a_rule_set_must_match_the_geometry(tiny3s_cfg) -> None:
    rules.RULES_V4.check(tiny3s_cfg)
    with pytest.raises(ValueError, match="sphere = False"):
        rules.RULES_V3.check(tiny3s_cfg)
    from alife.config import load_config

    with pytest.raises(ValueError, match="sphere = True"):
        rules.RULES_V4.check(load_config(preset="tiny3d"))
