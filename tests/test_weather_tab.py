"""The host tables of the sphere weather (``alife.weather_tab``, DESIGN §46.6): the combinatorics against the
independent HEALPix oracle, the edge geometry by properties that need none of the formulas of the module, the
least-squares weights and the fan triangles by what they must reproduce."""

from __future__ import annotations

import numpy as np
import pytest

import hp_oracle
from alife import geom, weather_tab

NSIDES = (2, 3, 4, 8, 16)


@pytest.mark.parametrize("ns", NSIDES)
def test_the_neighbour_tables_equal_the_independent_oracle(ns: int) -> None:
    t = weather_tab.weather_tab(ns)
    touch, edge, _valence, _vid = hp_oracle.neighbours(ns)
    assert t.npix == 12 * ns * ns
    for p in range(t.npix):
        assert set(t.nb[p]) == edge[p]
        assert set(t.nb8[p]) == touch[p]
    assert np.abs(t.x - hp_oracle.pix2vec(ns)).max() < 1e-12


@pytest.mark.parametrize("ns", NSIDES)
def test_edges_are_shared_symmetric_and_the_ring_has_seven_or_eight(ns: int) -> None:
    t = weather_tab.weather_tab(ns)
    p = np.arange(t.npix)[:, None]
    assert (t.nb[t.nb, t.rev] == p).all() and (t.nb != p).all()
    assert (t.ell[t.nb, t.rev] == t.ell).all() or np.allclose(t.ell[t.nb, t.rev], t.ell, atol=1e-13)
    assert np.allclose(t.d[t.nb, t.rev], t.d, atol=1e-13)
    assert np.allclose(t.dx[t.nb, t.rev], t.dx, atol=1e-13)  # the same point of the same edge
    assert set(np.unique(t.cnt)) == {7, 8} and (t.cnt == 7).sum() == 24
    for p_ in range(t.npix):  # padding repeats the last neighbour and carries no weight
        c = t.cnt[p_]
        assert (t.nb8[p_, c:] == t.nb8[p_, c - 1]).all() and not t.w8[p_, c:].any()


@pytest.mark.parametrize("ns", (4, 8, 16))
def test_edge_normals_are_unit_tangent_outward_and_flip_across_the_edge(ns: int) -> None:
    t = weather_tab.weather_tab(ns)
    corn = np.stack(
        [hp_oracle.corners(ns)[:, k] for k in range(4)], axis=1
    )  # independent corner positions (oracle)
    mid = corn + np.roll(corn, -1, axis=1)
    mid /= np.linalg.norm(mid, axis=-1, keepdims=True)
    assert np.allclose(np.linalg.norm(t.n, axis=-1), 1.0, atol=1e-12)
    assert np.abs((t.n * mid).sum(-1)).max() < 1e-12  # tangent at the edge midpoint
    assert ((t.n * (mid - t.x[:, None, :])).sum(-1) > 0).all()  # out of the cell
    assert np.allclose(t.n[t.nb, t.rev], -t.n, atol=1e-12)
    ell = np.arccos(np.clip((corn * np.roll(corn, -1, axis=1)).sum(-1), -1, 1))
    assert np.abs(ell - t.ell).max() < 1e-12
    # the unit tangent towards the neighbour's centre, at the edge midpoint
    assert np.allclose(np.linalg.norm(t.e, axis=-1), 1.0, atol=1e-12)
    assert np.abs((t.e * mid).sum(-1)).max() < 1e-12
    assert (
        (t.e * t.n).sum(-1) > 0.5
    ).all()  # the centre-to-centre direction is within 60 degrees of the normal


def test_the_divergence_theorem_on_a_cell() -> None:
    """For a constant vector ``a`` the tangent field ``a - (a·x)x`` has divergence ``-2 a·x``, so the edge sum
    ``Σ ell·(n·a)`` over the great-circle edges is ``-2 (a·x) A`` up to the cell's shape: the median error is
    second order in the side, the largest one (outside the 8 pole cells, whose great-circle polygon has 0.9 of
    the area 4π/N: the S4a finding behind the pole-cell entry of gate G1) decreasing."""
    a = np.array([0.3, -0.5, 0.8])
    med, mx_ = [], []
    for ns in (8, 16, 32):
        t = weather_tab.weather_tab(ns)
        err = np.abs((t.ell * (t.n @ a)).sum(-1) + 2.0 * (t.x @ a) * t.area) / t.area
        polar = (np.abs(hp_oracle.corners(ns)[..., 2]) > 1 - 1e-9).any(axis=1)  # a corner at a pole
        assert polar.sum() == 8 and err[polar].max() > 0.1  # the pole cells are the outliers
        med.append(np.median(err))
        mx_.append(err[~polar].max())
        per = t.ell.sum(-1)  # the cells enclose about four sides
        assert 3.3 * np.sqrt(t.area) < per.min() and per.max() < 4.6 * np.sqrt(t.area)
    assert med[0] < 0.01 and med[0] / med[1] > 3 and med[1] / med[2] > 3
    assert mx_[0] < 0.07 and mx_[0] > mx_[1] > mx_[2] and mx_[2] < 0.025


@pytest.mark.parametrize("ns", (8, 16, 32))
def test_the_lsq_weights_reproduce_the_gradient_of_a_linear_function(ns: int) -> None:
    t = weather_tab.weather_tab(ns)
    a = np.array([0.3, -0.5, 0.8])
    f = t.x @ a
    g = ((f[t.nb8] - f[:, None])[..., None] * t.w8).sum(1)
    exact = a - (t.x @ a)[:, None] * t.x  # the tangent gradient of a·x on the unit sphere
    err = np.abs(g - exact).max()
    assert err < 3.0 * np.sqrt(t.area)  # first order in the cell side at worst


def test_the_gradient_error_halves_with_the_cell_side() -> None:
    errs = []
    a = np.array([0.3, -0.5, 0.8])
    for ns in (8, 16, 32):
        t = weather_tab.weather_tab(ns)
        f = t.x @ a
        g = ((f[t.nb8] - f[:, None])[..., None] * t.w8).sum(1)
        exact = a - (t.x @ a)[:, None] * t.x
        errs.append(np.abs(g - exact).max())
    assert errs[0] / errs[1] > 1.6 and errs[1] / errs[2] > 1.6


@pytest.mark.parametrize("ns", (4, 8))
def test_the_fan_inverses_give_barycentric_coordinates_and_the_fan_covers_the_cell(ns: int) -> None:
    t = weather_tab.weather_tab(ns)
    dv = t.x[t.nb8] - t.x[:, None, :]
    dv = dv - (dv * t.x[:, None, :]).sum(-1, keepdims=True) * t.x[:, None, :]
    p2 = np.stack([(dv * t.ex[:, None, :]).sum(-1), (dv * t.ey[:, None, :]).sum(-1)], -1)  # (N, 8, 2)
    for p in range(0, t.npix, 3):
        for j in range(int(t.cnt[p])):
            nxt = (j + 1) % int(t.cnt[p])
            assert np.allclose(t.tinv[p, j] @ p2[p, j], [1, 0], atol=1e-10)
            assert np.allclose(t.tinv[p, j] @ p2[p, nxt], [0, 1], atol=1e-10)
    # a point inside the ring polygon is in at least one triangle (all three coordinates >= 0)
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(t.npix, 2))
    pts *= (0.4 * np.sqrt(t.area) / np.linalg.norm(pts, axis=1))[:, None]
    lam = np.einsum("pkij,pj->pki", t.tinv, pts)
    w0 = 1 - lam.sum(-1)
    inside = (np.minimum(np.minimum(lam[..., 0], lam[..., 1]), w0) >= -1e-12).any(axis=1)
    assert inside.all()


@pytest.mark.parametrize("ns", (4, 8, 16))
def test_the_laplacian_weights_are_symmetric_with_unit_mean(ns: int) -> None:
    t = weather_tab.weather_tab(ns)
    assert abs(t.wl.mean() - 1.0) < 1e-12
    assert np.allclose(
        t.wl[t.nb, t.rev], t.wl, atol=1e-12
    )  # symmetric: the biharmonic filter conserves the sum
    f = np.random.default_rng(1).normal(size=t.npix)
    lap = (t.wl * (f[t.nb] - f[:, None])).sum(-1)
    assert abs(lap.sum()) < 1e-10  # equal-area cells: the Laplacian sums to zero


def test_the_tables_are_read_only_and_cached() -> None:
    t = weather_tab.weather_tab(4)
    assert weather_tab.weather_tab(4) is t
    with pytest.raises(ValueError):
        t.nb[0, 0] = 0
    assert t.area == pytest.approx(geom.pixel_solid_angle(4))
    with pytest.raises(ValueError):
        weather_tab.weather_tab(1)
