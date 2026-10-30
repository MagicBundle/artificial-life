"""Host tables of the sphere weather (numpy, float64, unit sphere; DESIGN §46.6, true-sphere series S4b).

Everything the weather kernels of ``alife.weather_sph`` need that is not a function of the state, for the
HEALPix grid of Nside ``Nw`` (``N = 12·Nw²`` equal-area cells). Lengths are on the unit sphere: the kernels
multiply them by the planet radius. Built from the exact combinatorial tables of ``alife.geom`` (shared
corners, so a neighbour is never found by a distance test) and the corner positions of ``geom.xyf2vec``;
checked against the independent float64 prototype of S4a (``tests/test_weather_tab.py``: edge and touching
tables, the geometry of every edge, the least-squares weights and the fan triangles).

Per cell ``p``:

* ``nb[p, k]`` the 4 edge neighbours (edge ``k`` joins corner ``k`` to ``k+1`` of ``geom.CORNERS``), ``rev``
  the slot of ``p`` in ``nb[nb[p, k]]``;
* per edge: the arc length ``ell``, the **true** edge normal ``n`` (the tangent at the edge midpoint,
  perpendicular to the edge, pointing out of the cell: not the centre-to-centre direction, whose divergence
  error grows like 1/h), the unit tangent ``e`` at the midpoint of the direction from the centre to the
  neighbour's centre, the centre distance ``d`` and ``dx = x_f - x_mid`` (the edge midpoint minus the
  normalised midpoint of the two centres: up to 0.42 √A at the polar-cap facet seams);
* the 8 touching neighbours ``nb8`` (7 for 24 cells; counter-clockwise, padded with the last one), their
  ``1/d²``-weighted least-squares gradient weights ``W8`` (tangent 3-vectors), and the fan triangles'
  inverses ``tinv`` (``(centre, nb8[k], nb8[k+1])`` in the local basis ``(ex, ey)``; the padding triangles
  repeat the last one);
* ``wl`` the symmetric weights ``ell/d`` of the 4-neighbour Laplacian of the biharmonic filter, normalised to
  a mean of 1; ``bm`` the contraction ``W8[p, k']·dx[p, m]`` that turns the face correction of a gradient
  into a sum over the touching differences.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np

from alife import geom


@dataclass(frozen=True)
class WeatherTab:
    nside: int
    npix: int
    x: np.ndarray  # (N, 3) cell centres
    nb: np.ndarray  # (N, 4) int32
    rev: np.ndarray  # (N, 4) int32
    ell: np.ndarray  # (N, 4)
    n: np.ndarray  # (N, 4, 3)
    e: np.ndarray  # (N, 4, 3)
    d: np.ndarray  # (N, 4)
    dx: np.ndarray  # (N, 4, 3)
    nb8: np.ndarray  # (N, 8) int32
    cnt: np.ndarray  # (N,) int32 (7 or 8)
    w8: np.ndarray  # (N, 8, 3)
    ex: np.ndarray  # (N, 3) local east
    ey: np.ndarray  # (N, 3) local north
    tinv: np.ndarray  # (N, 8, 2, 2)
    wl: np.ndarray  # (N, 4)
    bm: np.ndarray  # (N, 4, 8)
    area: float  # the (equal) cell area on the unit sphere, 4π/N


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


@functools.cache
def weather_tab(nside: int) -> WeatherTab:
    """The tables of Nside ``nside`` (>= 2), cached; all arrays read-only."""
    ns = int(nside)
    tb = geom.tables(ns)
    n_pix = tb.npix
    x = geom.pixel_centres(ns)
    ix, iy, face = geom.pix2xyf(ns, np.arange(n_pix))
    corn = np.stack(
        [geom.xyf2vec((ix + cx) / ns, (iy + cy) / ns, face) for cx, cy in geom.CORNERS], axis=1
    )  # (N, 4, 3)
    nb = tb.edge.astype(np.int64)
    # the slot of p in its neighbour's row (an edge is shared by exactly two pixels, so it is unique)
    match = nb[nb] == np.arange(n_pix)[:, None, None]  # (N, 4, 4): [p, k, k'] nb[nb[p, k], k'] == p
    if not (match.sum(axis=2) == 1).all():
        raise AssertionError("an edge neighbour does not point back exactly once")
    rev = match.argmax(axis=2)
    # the neighbour across edge k shares the two corners of edge k
    vid = tb.vertex.astype(np.int64)
    a_lo, a_hi = np.minimum(vid, np.roll(vid, -1, 1)), np.maximum(vid, np.roll(vid, -1, 1))
    b = nb * 4 + rev  # row of the neighbour's edge
    if not (np.array_equal(a_lo.reshape(-1)[b], a_lo) and np.array_equal(a_hi.reshape(-1)[b], a_hi)):
        raise AssertionError("the edge tables disagree about the shared corners")
    # --- the edges
    ca, cb = corn, np.roll(corn, -1, axis=1)
    mid = _unit(ca + cb)
    ell = np.arccos(np.clip((ca * cb).sum(-1), -1.0, 1.0))
    t = cb - ca
    t = t - (t * mid).sum(-1, keepdims=True) * mid
    nn = _unit(np.cross(mid, t))
    sgn = np.sign((nn * (mid - x[:, None, :])).sum(-1))
    if not (sgn != 0).all():
        raise AssertionError("an edge normal is perpendicular to the direction out of its cell")
    nn = nn * sgn[..., None]
    xq = x[nb]
    dd = xq - x[:, None, :]
    dd = dd - (dd * mid).sum(-1, keepdims=True) * mid
    e = _unit(dd)
    dist = np.arccos(np.clip((x[:, None, :] * xq).sum(-1), -1.0, 1.0))
    dxm = mid - _unit(x[:, None, :] + xq)
    # --- the touching ring, its least-squares weights and the fan triangles
    cnt = tb.n_touch.astype(np.int64)
    if not set(np.unique(cnt)) <= {7, 8}:
        raise AssertionError("a cell does not have 7 or 8 touching neighbours")
    last = cnt - 1
    idx = np.minimum(np.arange(8)[None, :], last[:, None])
    nb8 = np.take_along_axis(tb.fan.astype(np.int64), idx, axis=1)
    valid = np.arange(8)[None, :] < cnt[:, None]
    ex, ey = geom.local_frame(x)
    dv = x[nb8] - x[:, None, :]
    dv = dv - (dv * x[:, None, :]).sum(-1, keepdims=True) * x[:, None, :]  # onto the tangent plane at p
    w = np.where(valid, 1.0 / (dv * dv).sum(-1), 0.0)
    m = np.einsum("pk,pki,pkj->pij", w, dv, dv)
    w8 = np.einsum("pij,pkj->pki", np.linalg.pinv(m, rcond=1e-10), dv * w[..., None])
    w8 = np.where(valid[..., None], w8, 0.0)
    p2 = np.stack([(dv * ex[:, None, :]).sum(-1), (dv * ey[:, None, :]).sum(-1)], -1)  # (N, 8, 2)
    jj = np.arange(8)[None, :]
    jc = np.minimum(jj, last[:, None])
    nxt = np.where(jc + 1 < cnt[:, None], jc + 1, 0)
    a0 = np.take_along_axis(p2, jc[..., None], axis=1)
    a1 = np.take_along_axis(p2, nxt[..., None], axis=1)
    tinv = np.linalg.inv(np.stack([a0, a1], axis=-1))  # (N, 8, 2, 2): the columns are the triangle's vertices
    wl = ell / dist
    wl = wl / wl.mean()
    bm = np.einsum("pmj,pkj->pmk", dxm, w8)
    tab = WeatherTab(
        nside=ns,
        npix=n_pix,
        x=x,
        nb=nb.astype(np.int32),
        rev=rev.astype(np.int32),
        ell=ell,
        n=nn,
        e=e,
        d=dist,
        dx=dxm,
        nb8=nb8.astype(np.int32),
        cnt=cnt.astype(np.int32),
        w8=w8,
        ex=ex,
        ey=ey,
        tinv=tinv,
        wl=wl,
        bm=bm,
        area=geom.pixel_solid_angle(ns),
    )
    for f in tab.__dataclass_fields__:
        v = getattr(tab, f)
        if isinstance(v, np.ndarray):
            v.flags.writeable = False
    return tab
