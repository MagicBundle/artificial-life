"""The HEALPix scheme and its exact tables (``alife.geom``; DESIGN §46.1, sphere series S0).

Checked against the independent implementation ``tests/hp_oracle.py`` (the reviewer's, rounded-coordinate
neighbours) and against combinatorial facts: the Euler count of vertices, valences 3 and 4, 4 edge and 8 (24
cells: 7) touching neighbours, symmetric tables, the exact refinement for any integer ratio.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import hp_oracle as hp
from alife import geom

NSIDES_ALL = (2, 3, 4, 5, 6, 7, 8, 11, 16, 21, 32, 64)  # round trips and equal area
NSIDES_TABLES = (2, 3, 4, 5, 8, 11, 16, 21, 32)
NSIDES_ORACLE = (2, 3, 4, 5, 6, 7, 8, 11)  # the oracle's neighbours are O(Npix) Python loops


@pytest.mark.parametrize("nside", NSIDES_ALL)
def test_ring_to_facet_round_trip_is_a_bijection(nside: int) -> None:
    n = geom.npix(nside)
    pix = np.arange(n)
    ix, iy, face = geom.pix2xyf(nside, pix)
    assert ix.min() >= 0 and iy.min() >= 0 and ix.max() < nside and iy.max() < nside
    assert face.min() == 0 and face.max() == 11
    assert np.array_equal(geom.xyf2pix(nside, ix, iy, face), pix)
    assert len(np.unique((face * nside + iy) * nside + ix)) == n  # every (face, iy, ix) once


@pytest.mark.parametrize("nside", NSIDES_ALL)
def test_centres_round_trip_and_match_the_oracle(nside: int) -> None:
    n = geom.npix(nside)
    c = geom.pix2vec(nside)
    assert c.shape == (n, 3) and np.abs(np.linalg.norm(c, axis=1) - 1).max() < 1e-14
    assert np.array_equal(geom.ang2pix(c, nside), np.arange(n))  # exact over all pixels
    if nside <= 32:
        assert np.abs(c - hp.pix2vec(nside)).max() < 1e-13


@pytest.mark.parametrize("nside", (1, 2, 5, 13, 21))
def test_ang2pix_agrees_with_the_oracle_on_random_points(nside: int) -> None:
    rng = np.random.default_rng(nside)
    v = rng.normal(size=(200_000, 3))
    v /= np.linalg.norm(v, axis=1)[:, None]
    assert np.array_equal(geom.ang2pix(v, nside), hp.ang2pix(v, nside))
    # unnormalised input gives the same pixel
    assert np.array_equal(geom.ang2pix(v * 77.7, nside), geom.ang2pix(v, nside))


def _fibonacci(m: int) -> np.ndarray:
    i = np.arange(m) + 0.5
    z = 1 - 2 * i / m
    phi = math.pi * (1 + 5**0.5) * i
    s = np.sqrt(1 - z * z)
    return np.stack([s * np.cos(phi), s * np.sin(phi), z], axis=1)


@pytest.mark.parametrize("nside", (2, 5, 8, 11))
def test_pixels_have_equal_area(nside: int) -> None:
    """Gauss-Legendre quadrature of the surface element |v_x × v_y| over each pixel's square (a finite
    difference of the HEALPix map), and an independent lattice count with ``ang2pix``."""
    n = geom.npix(nside)
    ix, iy, face = geom.pix2xyf(nside, np.arange(n))
    nodes, weights = np.polynomial.legendre.leggauss(6)
    nodes, weights = 0.5 * (nodes + 1), 0.5 * weights
    area = np.zeros(n)
    h = 1e-6
    for a, wa in zip(nodes, weights, strict=True):
        for b, wb in zip(nodes, weights, strict=True):
            x, y = (ix + a) / nside, (iy + b) / nside
            vx = (geom.xyf2vec(x + h, y, face) - geom.xyf2vec(x - h, y, face)) / (2 * h)
            vy = (geom.xyf2vec(x, y + h, face) - geom.xyf2vec(x, y - h, face)) / (2 * h)
            area += wa * wb * np.linalg.norm(np.cross(vx, vy), axis=1) / nside**2
    assert np.abs(area / geom.pixel_solid_angle(nside) - 1).max() < 2e-7
    assert abs(area.sum() / (4 * math.pi) - 1) < 1e-7
    m = 1500 * n
    counts = np.bincount(geom.ang2pix(_fibonacci(m), nside), minlength=n)
    assert np.abs(counts / (m / n) - 1).max() < 0.03


@pytest.mark.parametrize("nside", NSIDES_TABLES)
def test_neighbour_structure(nside: int) -> None:
    tb = geom.tables(nside)
    n = tb.npix
    assert tb.n_vertices == n + 2  # Euler: V − E + F = 2 with E = 2F
    val = np.bincount(tb.vertex.reshape(-1))
    assert set(np.unique(val)) == {3, 4} and (val == 3).sum() == 8  # 8 trivalent vertices
    # edge neighbours: exactly four, distinct, never the pixel itself, symmetric
    assert tb.edge.shape == (n, 4) and (tb.edge >= 0).all() and (tb.edge != np.arange(n)[:, None]).all()
    assert np.all(np.sort(tb.edge, axis=1)[:, 1:] != np.sort(tb.edge, axis=1)[:, :-1])
    assert all(p in tb.edge[q] for p in range(0, n, max(1, n // 400)) for q in tb.edge[p])
    # touching: 8, or 7 for the 24 pixels around the trivalent vertices
    assert set(np.unique(tb.n_touch)) == {7, 8} and (tb.n_touch == 7).sum() == 24
    assert all(set(tb.edge[p]) <= set(tb.touch[p][tb.touch[p] >= 0]) for p in range(0, n, max(1, n // 400)))
    # symmetric, no self, ascending with −1 padding at the end
    pairs = {(p, int(q)) for p in range(n) for q in tb.touch[p] if q >= 0}
    assert all((q, p) in pairs for p, q in pairs) and all(p != q for p, q in pairs)
    assert all(np.all(np.diff(row[row >= 0]) > 0) for row in tb.touch)
    # the seven-neighbour pixels are the three around each trivalent vertex
    vids = np.flatnonzero(val == 3)
    around = {int(p) for v in vids for p in np.flatnonzero((tb.vertex == v).any(axis=1))}
    assert around == set(np.flatnonzero(tb.n_touch == 7).tolist())
    # the trivalent vertices sit at z = ±2/3 and φ = k·90°
    ix, iy, face = geom.pix2xyf(nside, np.arange(n))
    corner = {}
    for k, (dx, dy) in enumerate(geom.CORNERS):
        corner.update(
            zip(
                tb.vertex[:, k].tolist(),
                geom.xyf2vec((ix + dx) / nside, (iy + dy) / nside, face),
                strict=True,
            )
        )
    for v in vids:
        z, ph = corner[int(v)][2], math.atan2(corner[int(v)][1], corner[int(v)][0])
        assert abs(abs(z) - 2 / 3) < 1e-12 and abs(((ph / (math.pi / 2)) + 0.5) % 1 - 0.5) < 1e-9


@pytest.mark.parametrize("nside", (2, 3, 8))
def test_vertex_keys_are_the_geometric_identity(nside: int) -> None:
    """Corners share a key iff they are the same point of the sphere (integers against float geometry)."""
    n = geom.npix(nside)
    ix, iy, face = geom.pix2xyf(nside, np.arange(n))
    key = geom.vertex_keys(nside).reshape(-1)
    pts = np.concatenate(
        [geom.xyf2vec((ix + dx) / nside, (iy + dy) / nside, face)[:, None] for dx, dy in geom.CORNERS], axis=1
    ).reshape(-1, 3)
    uniq, inv = np.unique(key, return_inverse=True)
    first = np.zeros(len(uniq), dtype=np.int64)
    first[inv[::-1]] = np.arange(len(inv))[::-1]
    assert np.abs(pts - pts[first][inv]).max() < 1e-12  # same key, same point
    centres = pts[first]
    d = np.linalg.norm(centres[:, None, :] - centres[None, :, :], axis=-1) + 10 * np.eye(len(uniq))
    assert d.min() > 0.5 * geom.pixel_side(nside) * 0.2  # different keys, different points


@pytest.mark.parametrize("nside", NSIDES_ORACLE)
def test_tables_match_the_independent_oracle(nside: int) -> None:
    tb = geom.tables(nside)
    touch, edge, _, _ = hp.neighbours(nside)
    for p in range(tb.npix):
        assert set(tb.touch[p][tb.touch[p] >= 0].tolist()) == touch[p]
        assert set(tb.edge[p].tolist()) == edge[p]
        s2 = set().union(*[touch[q] | {q} for q in touch[p] | {p}])
        assert set(tb.stencil2[p][tb.stencil2[p] >= 0].tolist()) == s2
        assert tb.stencil2[p, 0] == p


@pytest.mark.parametrize("nside", NSIDES_TABLES)
def test_reach_two_stencil(nside: int) -> None:
    tb = geom.tables(nside)
    s = tb.stencil2
    sizes = (s >= 0).sum(axis=1)
    assert sizes.max() == (25 if nside >= 3 else 23) and sizes.min() >= 19
    assert (s[:, 0] == np.arange(tb.npix)).all()
    assert all(np.all(np.diff(row[1:][row[1:] >= 0]) > 0) for row in s)  # ascending after self
    assert np.array_equal(geom.edge_stencil(nside, 1)[:, 1:], tb.touch)


@pytest.mark.parametrize("nside", (2, 3, 8, 21))
def test_fan_is_a_counter_clockwise_ring_of_triangles(nside: int) -> None:
    tb = geom.tables(nside)
    n = tb.npix
    cen = geom.pixel_centres(nside)
    e, nn = geom.local_frame(cen)
    k = np.arange(8)[None, :]
    assert np.array_equal(
        np.sort(np.where(tb.fan >= 0, tb.fan, n), axis=1),
        np.sort(np.where(tb.touch >= 0, tb.touch, n), axis=1),
    )
    # consecutive neighbours are themselves touching (cyclically), so the fan has no gap
    nxt = np.take_along_axis(tb.fan, np.where(k + 1 < tb.n_touch[:, None], k + 1, 0), axis=1)
    ok = k < tb.n_touch[:, None]
    p_idx = np.broadcast_to(np.arange(n)[:, None], tb.fan.shape)
    touching = (tb.touch[tb.fan.clip(0)] == nxt[..., None]).any(-1)
    assert touching[ok].all() and (tb.fan[ok] != nxt[ok]).all()
    # the angles at the centre are positive (counter-clockwise) and sum to a full turn
    a = cen[tb.fan.clip(0)]
    b = cen[nxt]
    ta = np.arctan2(((a - cen[:, None]) * nn[:, None]).sum(-1), ((a - cen[:, None]) * e[:, None]).sum(-1))
    tb_ = np.arctan2(((b - cen[:, None]) * nn[:, None]).sum(-1), ((b - cen[:, None]) * e[:, None]).sum(-1))
    turn = np.mod(tb_ - ta, 2 * math.pi)
    assert (turn[ok] > 0).all() and (turn[ok] < math.pi).all()
    assert np.abs(np.where(ok, turn, 0).sum(1) - 2 * math.pi).max() < 1e-9
    tri = geom.fan_triangles(nside)
    assert tri.shape == (n, 8, 3) and (tri[..., 0][ok] == p_idx[ok]).all() and (tri[~ok] == -1).all()
    assert np.array_equal(tri[..., 1][ok], tb.fan[ok]) and np.array_equal(tri[..., 2][ok], nxt[ok])


@pytest.mark.parametrize("nside", (8, 16, 24))
def test_lsq_weights_reproduce_a_smooth_gradient(nside: int) -> None:
    cen = geom.pixel_centres(nside)
    g = np.array([0.3, -0.5, 0.8])  # f(c) = g·c, whose tangent gradient at c is g − (g·c)c
    f = cen @ g
    exact = g[None, :] - (cen @ g)[:, None] * cen
    errs = {}
    for kind in ("edge", "touch"):
        tb = geom.tables(nside)
        nb = tb.edge if kind == "edge" else tb.touch
        w = geom.lsq_weights(nside, kind)
        est = np.einsum("pjk,pj->pk", w, np.where(nb >= 0, f[nb.clip(0)] - f[:, None], 0.0))
        assert np.abs((est * cen).sum(1)).max() < 1e-12  # tangent
        errs[kind] = np.abs(est - exact).max()
    assert errs["touch"] < 0.025 * 24 / nside and errs["edge"] < 0.06 * 24 / nside  # O(1/Nside)
    # constants have zero gradient; the weights ignore padded neighbours
    w = geom.lsq_weights(nside, "touch")
    assert np.all(w[geom.tables(nside).touch < 0] == 0)


def test_lsq_condition_of_the_touching_neighbours() -> None:
    """The conditioning of the 8-neighbour tangent matrix quoted in the spec (max 2.2, mean 1.3 at 24)."""
    cond = geom.lsq_condition(24, "touch")
    assert cond.min() >= 1.0 and cond.max() < 2.3 and abs(cond.mean() - 1.3) < 0.1


@pytest.mark.parametrize("fine,ratio", [(22, 2), (42, 2), (28, 4), (63, 3), (30, 6), (16, 2), (8, 8)])
def test_refinement_by_any_integer_ratio_is_exact(fine: int, ratio: int) -> None:
    coarse = fine // ratio
    par = geom.parent_table(fine, ratio)
    kids = geom.children_table(coarse, ratio)
    assert kids.shape == (geom.npix(coarse), ratio * ratio)
    assert np.array_equal(np.bincount(par), np.full(geom.npix(coarse), ratio * ratio))
    assert np.array_equal(par[kids], np.repeat(np.arange(geom.npix(coarse))[:, None], ratio * ratio, 1))
    # geometric truth: a point well inside a fine pixel lies in its parent
    ix, iy, face = geom.pix2xyf(fine, np.arange(geom.npix(fine)))
    for dx, dy in ((0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9), (0.5, 0.5)):
        pts = geom.xyf2vec((ix + dx) / fine, (iy + dy) / fine, face)
        assert np.array_equal(geom.ang2pix(pts, coarse), par)


@pytest.mark.parametrize("nside,n_colours", [(8, 15), (11, 16), (21, 15)])
def test_greedy_colouring_separates_cells_within_two_steps(nside: int, n_colours: int) -> None:
    colour, k = geom.greedy_colouring(nside)
    tb = geom.tables(nside)
    assert colour.min() == 0 and colour.max() == k - 1 and k == n_colours and k >= 9
    near = tb.stencil2[:, 1:]
    assert (colour[:, None] != colour[near.clip(0)])[near >= 0].all()
    # the oracle's two-step sets agree with the table, so the check above is against an independent graph
    touch, _, _, _ = hp.neighbours(nside) if nside <= 11 else (None, None, None, None)
    if touch is not None:
        for p in range(0, tb.npix, 7):
            s2 = set().union(*[touch[q] | {q} for q in touch[p] | {p}]) - {p}
            assert all(colour[q] != colour[p] for q in s2)
