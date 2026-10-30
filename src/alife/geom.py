"""Host-side geometry of the sphere world (numpy, float64; DESIGN §46, docs of the true-sphere series, S0).

Nothing here touches the engine: box worlds never import this module's tables. It holds

* the HEALPix RING scheme for **any** integer Nside, implemented from the published formulas (Gorski et al.,
  ApJ 622, 759, 2005): ``pix2xyf``/``xyf2pix`` (ring index and facet coordinates), ``xyf2vec``, ``ang2pix``,
  ``pix2vec``;
* **exact** neighbour tables built from shared pixel corners (a corner is identified by integer ring
  arithmetic, never by rounding coordinates): touching (8, or 7 at 24 pixels), edge (4), the reach-2 hash
  stencil (self + at most 24), the counter-clockwise fan of touching neighbours and its triangles, the
  tangent least-squares gradient weights, the parent-child tables of an integer refinement, and a greedy
  colouring of the distance-2 touching graph;
* the **exact coverage** of the reach-1 and reach-2 hash stencils (the distance on the sphere between the
  boundary of a cell and the boundary of its stencil union, minimised over two curves with a golden-section
  refinement) and the derivation of the hash Nside ``Nh`` and of the cell capacity ``C`` from it;
* the equal-volume shell layers;
* the geometry accessors of the host (``Box`` and ``Sphere``; ``geometry(cfg)``), the single place where a
  consumer reads a height, an up direction, a displacement or a cell without caring about the world's shape.

The device twin (float32 MLX) is ``alife.sphere``. The tests check it against this module and this module
against an independent implementation (``tests/hp_oracle.py``).
"""

from __future__ import annotations

import functools
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from alife.config import Config

JRLL = np.array(
    [2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4], dtype=np.int64
)  # ring of the facet's top corner / Nside
JPLL = np.array([1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7], dtype=np.int64)  # longitude of the facet's corner
CORNERS = (
    (0, 0),
    (1, 0),
    (1, 1),
    (0, 1),
)  # facet-coordinate corner k of a pixel; edge k joins corner k to k+1
EDGE_NEIGHBOUR_OFFSET = ((0, -1), (1, 0), (0, 1), (-1, 0))  # (dix, diy) across edge k
TOUCH_W = 8  # width of the touching table
EDGE_W = 4
STENCIL2_W = 25  # self + at most 24
TWO_PI = 2.0 * math.pi


def npix(nside: int) -> int:
    """Number of pixels ``12·Nside²``."""
    return 12 * int(nside) * int(nside)


def pixel_solid_angle(nside: int) -> float:
    """The area of every pixel on the unit sphere, ``4π/Npix``."""
    return 4.0 * math.pi / npix(nside)


def pixel_side(nside: int) -> float:
    """``sqrt(4π/Npix)``: the unit in which coverages are quoted."""
    return math.sqrt(pixel_solid_angle(nside))


def _isqrt(a: np.ndarray) -> np.ndarray:
    """Exact integer square root of non-negative int64 values."""
    r = np.floor(np.sqrt(a.astype(np.float64))).astype(np.int64)
    r = np.where(r * r > a, r - 1, r)
    return np.where((r + 1) * (r + 1) <= a, r + 1, r)


# --- ring index <-> facet coordinates ----------------------------------------------------------------------


def pix2xyf(nside: int, pix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """RING pixel index -> ``(ix, iy, face)``: the pixel's column and row inside its base facet (0..N−1) and
    the facet (0..11)."""
    n = int(nside)
    pix = np.asarray(pix, dtype=np.int64)
    ncap, tot = 2 * n * (n - 1), 12 * n * n
    north, south = pix < ncap, pix >= tot - ncap
    equ = ~(north | south)
    # polar caps: ring ir has 4·ir pixels
    pn = np.where(north, pix, 0)
    ir_n = (1 + _isqrt(1 + 2 * pn)) >> 1
    ip_n = pn + 1 - 2 * ir_n * (ir_n - 1)
    face_n = (ip_n - 1) // np.maximum(ir_n, 1)
    ps = np.where(south, tot - pix, 1)
    nr_s = (1 + _isqrt(2 * ps - 1)) >> 1
    ip_s = 4 * nr_s + 1 - (ps - 2 * nr_s * (nr_s - 1))
    face_s = 8 + (ip_s - 1) // np.maximum(nr_s, 1)
    ir_s = 4 * n - nr_s
    # equatorial belt: 4N pixels per ring
    q = np.where(equ, pix - ncap, 0)
    t = q // (4 * n)
    ir_e = t + n
    ip_e = q - t * 4 * n + 1
    ks_e = (ir_e + n) & 1
    ire = t + 1
    irm = 2 * n + 2 - ire
    ifm = (ip_e - (ire >> 1) + n - 1) // n
    ifp = (ip_e - (irm >> 1) + n - 1) // n
    face_e = np.where(ifp == ifm, ifp | 4, np.where(ifp < ifm, ifp, ifm + 8))
    iring = np.where(north, ir_n, np.where(south, ir_s, ir_e))
    iphi = np.where(north, ip_n, np.where(south, ip_s, ip_e))
    kshift = np.where(equ, ks_e, 0)
    nr = np.where(north, ir_n, np.where(south, nr_s, n))
    face = np.where(north, face_n, np.where(south, face_s, face_e))
    irt = iring - JRLL[face] * n + 1
    ipt = 2 * iphi - JPLL[face] * nr - kshift - 1
    ipt = np.where(ipt >= 2 * n, ipt - 8 * n, ipt)
    return (ipt - irt) >> 1, (-(ipt + irt)) >> 1, face


def xyf2pix(nside: int, ix: np.ndarray, iy: np.ndarray, face: np.ndarray) -> np.ndarray:
    """``(ix, iy, face)`` -> RING pixel index (the exact inverse of ``pix2xyf``)."""
    n = int(nside)
    ix, iy, face = (np.asarray(a, dtype=np.int64) for a in (ix, iy, face))
    tot, ncap, nl4 = 12 * n * n, 2 * n * (n - 1), 4 * n
    jr = JRLL[face] * n - ix - iy - 1
    top, bot = jr < n, jr > 3 * n
    nr = np.where(top, jr, np.where(bot, nl4 - jr, n))
    before = np.where(top, 2 * nr * (nr - 1), np.where(bot, tot - 2 * (nr + 1) * nr, ncap + (jr - n) * nl4))
    kshift = np.where(top | bot, 0, (jr - n) & 1)
    jp = (JPLL[face] * nr + ix - iy + 1 + kshift) >> 1
    jp = np.where(jp > nl4, jp - nl4, np.where(jp < 1, jp + nl4, jp))
    return before + jp - 1


def xyf2vec(fx: np.ndarray, fy: np.ndarray, face: np.ndarray) -> np.ndarray:
    """Continuous facet coordinates ``(fx, fy)`` in [0, 1]² of a facet -> unit vector (the HEALPix map)."""
    fx, fy = np.asarray(fx, dtype=np.float64), np.asarray(fy, dtype=np.float64)
    face = np.asarray(face, dtype=np.int64)
    jr = JRLL[face] - fx - fy
    nr = np.where(jr < 1.0, jr, np.where(jr > 3.0, 4.0 - jr, 1.0))
    z = np.where(
        jr < 1.0, 1.0 - nr * nr / 3.0, np.where(jr > 3.0, nr * nr / 3.0 - 1.0, (2.0 - jr) * (2.0 / 3.0))
    )
    safe = nr > 1e-15
    # the longitude in units of pi/4, wrapped AFTER the division by nr (a defensive form: in a polar cap
    # (nr < 1) wrapping the numerator by 8 would be wrong whenever it is negative at rounding level; the
    # review could not trigger it with the original order, and both orders agree on every tested point)
    t = (JPLL[face] * nr + fx - fy) / np.where(safe, nr, 1.0)
    t = np.where(t < 0.0, t + 8.0, np.where(t >= 8.0, t - 8.0, t))
    phi = np.where(safe, 0.25 * math.pi * t, 0.0)
    st = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.stack([st * np.cos(phi), st * np.sin(phi), z], axis=-1)


def pix2vec(nside: int, pix: np.ndarray | None = None) -> np.ndarray:
    """Pixel centres (all pixels when ``pix`` is None), unit vectors, float64."""
    if pix is None:
        return pixel_centres(int(nside))
    ix, iy, face = pix2xyf(nside, pix)
    return xyf2vec((ix + 0.5) / nside, (iy + 0.5) / nside, face)


@functools.cache
def pixel_centres(nside: int) -> np.ndarray:
    out = pix2vec(nside, np.arange(npix(nside)))
    out.flags.writeable = False
    return out


def ang2pix(vec: np.ndarray, nside: int) -> np.ndarray:
    """RING pixel index of each direction (float64; the reference the float32 device lookup is held to). The
    vectors need not be normalised."""
    n = int(nside)
    v = np.asarray(vec, dtype=np.float64).reshape(-1, 3)
    r = np.sqrt((v * v).sum(axis=1))
    z = np.clip(v[:, 2] / r, -1.0, 1.0)
    phi = np.mod(np.arctan2(v[:, 1], v[:, 0]), TWO_PI)
    tt = phi / (0.5 * math.pi)  # [0, 4]
    za = np.abs(z)
    tot, ncap = 12 * n * n, 2 * n * (n - 1)
    out = np.empty(len(z), dtype=np.int64)
    eq = za <= 2.0 / 3.0
    t1, t2 = n * (0.5 + tt[eq]), n * z[eq] * 0.75
    jp, jm = np.floor(t1 - t2).astype(np.int64), np.floor(t1 + t2).astype(np.int64)
    ir = n + 1 + jp - jm
    ks = 1 - (ir & 1)
    ip = ((jp + jm - n + ks + 1) >> 1) % (4 * n)
    out[eq] = ncap + (ir - 1) * 4 * n + ip
    pc = ~eq
    tp = tt[pc] - np.floor(tt[pc])
    tmp = n * np.sqrt(3.0 * (1.0 - za[pc]))
    jp = np.floor(tp * tmp).astype(np.int64)
    jm = np.floor((1.0 - tp) * tmp).astype(np.int64)
    ir = np.clip(jp + jm + 1, 1, n)
    ip = np.floor(tt[pc] * ir).astype(np.int64) % (4 * ir)
    out[pc] = np.where(z[pc] > 0, 2 * ir * (ir - 1) + ip, tot - 2 * ir * (ir + 1) + ip)
    return out.reshape(np.shape(vec)[:-1])


# --- exact tables from shared corners ----------------------------------------------------------------------


@dataclass(frozen=True)
class Tables:
    """Combinatorial structure of one Nside (all arrays read-only, int32 unless noted).

    ``vertex[p, k]`` is the id of corner k of pixel p (0..V−1, ``V = 12N²+2``); ``edge[p, k]`` the pixel
    across
    edge k (edge k joins corner k to k+1: ``-y, +x, +y, -x`` in facet coordinates); ``touch[p]`` the
    pixels that
    share at least one corner (ascending, padded with −1); ``stencil2[p]`` is p itself followed by the pixels
    within two touching steps (ascending, padded with −1); ``fan[p]`` is the touching ring in
    counter-clockwise
    order seen from outside (by the direction of the neighbour's centre from local east, in (−π, π]; padded
    with −1).
    """

    nside: int
    npix: int
    vertex: np.ndarray  # (N, 4)
    n_vertices: int
    edge: np.ndarray  # (N, 4)
    touch: np.ndarray  # (N, 8)
    n_touch: np.ndarray  # (N,)
    stencil2: np.ndarray  # (N, 25)
    fan: np.ndarray  # (N, 8)


def _frozen(a: np.ndarray) -> np.ndarray:
    a.flags.writeable = False
    return a


def vertex_keys(nside: int) -> np.ndarray:
    """(N, 4) int64 key of the corner k of each pixel. Two corners are the same point of the sphere iff their
    keys are equal: the key is ``(ring, longitude index)`` of the corner in exact integer arithmetic (the
    longitude index counts eighths of a quarter turn per unit of the ring's width, wrapped modulo the
    ring)."""
    n = int(nside)
    ix, iy, face = pix2xyf(n, np.arange(12 * n * n))
    out = np.empty((len(ix), 4), dtype=np.int64)
    for k, (dx, dy) in enumerate(CORNERS):
        x, y = ix + dx, iy + dy
        jr = JRLL[face] * n - x - y  # 0..4N
        nr = np.where(jr < n, jr, np.where(jr > 3 * n, 4 * n - jr, n))
        tmp = np.where(nr > 0, (JPLL[face] * nr + x - y) % np.maximum(8 * nr, 1), 0)
        out[:, k] = jr * (8 * n) + tmp
    return out


def _pairs_to_table(a: np.ndarray, b: np.ndarray, n: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Directed pairs (a -> b), de-duplicated, as a padded ascending table (n, width) and the row counts."""
    key = np.unique(a.astype(np.int64) * n + b)
    ra, rb = key // n, key % n
    cnt = np.bincount(ra, minlength=n)
    if cnt.max() > width:
        raise AssertionError(f"a pixel has {cnt.max()} neighbours, the table is {width} wide")
    start = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    col = np.arange(len(ra)) - start[ra]
    tab = np.full((n, width), -1, dtype=np.int32)
    tab[ra, col] = rb
    return tab, cnt


@functools.cache
def tables(nside: int) -> Tables:
    """The exact combinatorial tables of ``nside`` (>= 2), cached."""
    n = int(nside)
    if n < 2:
        raise ValueError(
            "the neighbour tables need Nside >= 2 (at Nside 1 a pixel has 6 touching neighbours)"
        )
    tot = 12 * n * n
    uniq, inv = np.unique(vertex_keys(n), return_inverse=True)
    vid = inv.reshape(tot, 4)
    nv = len(uniq)
    # edges: each joins two corners and belongs to exactly two pixels
    a, b = vid, np.roll(vid, -1, axis=1)
    ekey = (np.minimum(a, b).astype(np.int64) * nv + np.maximum(a, b)).reshape(-1)
    order = np.argsort(ekey, kind="stable")
    ks = ekey[order]
    if not (np.array_equal(ks[0::2], ks[1::2]) and np.all(ks[1::2][:-1] != ks[0::2][1:])):
        raise AssertionError("an edge is not shared by exactly two pixels")
    first, second = order[0::2], order[1::2]
    edge = np.empty(4 * tot, dtype=np.int32)
    edge[first], edge[second] = second // 4, first // 4
    edge = edge.reshape(tot, 4)
    # touching: pixels around a common corner (valence 3 or 4)
    flat = vid.reshape(-1)
    o = np.argsort(flat, kind="stable")
    val = np.bincount(flat, minlength=nv)
    start = np.concatenate([[0], np.cumsum(val)[:-1]])
    pa, pb = [], []
    for v in np.unique(val):
        vs = np.flatnonzero(val == v)
        members = o[start[vs][:, None] + np.arange(v)[None, :]] // 4  # (m, v) pixels around each corner
        for i in range(v):
            for j in range(v):
                if i != j:
                    pa.append(members[:, i])
                    pb.append(members[:, j])
    touch, n_touch = _pairs_to_table(np.concatenate(pa), np.concatenate(pb), tot, TOUCH_W)
    # reach-2 stencil: self + touching of (self + touching)
    s1 = np.concatenate([np.arange(tot, dtype=np.int32)[:, None], touch], axis=1)  # (N, 9)
    cand = touch[s1.clip(0)]  # (N, 9, 8)
    cand = np.where((s1 >= 0)[:, :, None], cand, -1).reshape(tot, -1)
    cand = np.concatenate([s1, cand], axis=1).astype(np.int64)
    cand = np.where(cand < 0, tot, cand)
    cand.sort(axis=1)
    dup = np.zeros_like(cand, dtype=bool)
    dup[:, 1:] = cand[:, 1:] == cand[:, :-1]
    cand = np.where(dup, tot, cand)
    cand.sort(axis=1)
    if (cand[:, STENCIL2_W] < tot).any():
        raise AssertionError("a reach-2 stencil has more than 25 cells")
    rest = cand[:, :STENCIL2_W]
    rest = np.where(rest == np.arange(tot)[:, None], tot, rest)  # self goes first
    rest.sort(axis=1)
    stencil2 = np.concatenate([np.arange(tot)[:, None], rest[:, : STENCIL2_W - 1]], axis=1)
    stencil2 = np.where(stencil2 >= tot, -1, stencil2).astype(np.int32)
    fan = _fan_order(n, touch, n_touch)
    return Tables(
        nside=n,
        npix=tot,
        vertex=_frozen(vid.astype(np.int32)),
        n_vertices=nv,
        edge=_frozen(edge),
        touch=_frozen(touch),
        n_touch=_frozen(n_touch.astype(np.int32)),
        stencil2=_frozen(stencil2),
        fan=_frozen(fan),
    )


def local_frame(c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """East and north unit vectors at the unit vectors ``c`` (..., 3): ``east = normalise(ẑ × c)``, ``north =
    c × east``; x̂ replaces east where ``|ẑ × c| < 1e-6`` (the poles)."""
    c = np.asarray(c, dtype=np.float64)
    e = np.stack([-c[..., 1], c[..., 0], np.zeros_like(c[..., 0])], axis=-1)
    ne = np.linalg.norm(e, axis=-1, keepdims=True)
    e = np.where(ne < 1e-6, np.array([1.0, 0.0, 0.0]), e / np.maximum(ne, 1e-300))
    return e, np.cross(c, e)


def _fan_order(nside: int, touch: np.ndarray, n_touch: np.ndarray) -> np.ndarray:
    """The touching neighbours of each pixel in counter-clockwise order, by the direction of the neighbour's
    centre in the pixel's tangent plane (east = 0), padded with −1."""
    cen = pixel_centres(nside)
    e, nn = local_frame(cen)
    t = touch.clip(0)
    d = cen[t] - cen[:, None, :]  # (N, 8, 3)
    ang = np.arctan2((d * nn[:, None, :]).sum(-1), (d * e[:, None, :]).sum(-1))
    ang = np.where(touch >= 0, ang, np.inf)
    order = np.argsort(ang, axis=1, kind="stable")
    fan = np.take_along_axis(touch, order, axis=1)
    return fan.astype(np.int32)


def fan_triangles(nside: int) -> np.ndarray:
    """(N, 8, 3) int32: the triangle fan of each pixel, ``(pixel, fan[i], fan[i+1 mod n_touch])``; −1 rows
    past the 7th for the 24 pixels with 7 touching neighbours."""
    tb = tables(nside)
    k = np.arange(TOUCH_W)[None, :]
    nxt = np.take_along_axis(tb.fan, np.where(k + 1 < tb.n_touch[:, None], k + 1, 0), axis=1)
    tri = np.stack(
        [np.broadcast_to(np.arange(tb.npix, dtype=np.int32)[:, None], tb.fan.shape), tb.fan, nxt], axis=-1
    )
    return np.where((k < tb.n_touch[:, None])[:, :, None], tri, -1).astype(np.int32)


def edge_stencil(nside: int, reach: int) -> np.ndarray:
    """(N, 9) for reach 1 (self + touching), (N, 25) for reach 2; self first, then ascending, −1 padded."""
    tb = tables(nside)
    if reach == 1:
        return np.concatenate([np.arange(tb.npix, dtype=np.int32)[:, None], tb.touch], axis=1)
    if reach == 2:
        return tb.stencil2
    raise ValueError("reach must be 1 or 2")


def lsq_weights(nside: int, kind: str = "edge") -> np.ndarray:
    """Tangent least-squares gradient weights ``W (N, k, 3)`` on the unit sphere: for a scalar f,
    ``∇f ≈ Σ_j W[p, j]·(f[nb_j] − f[p])`` (a tangent 3-vector, per radian; divide by the radius for per unit
    length). ``kind`` is ``"edge"`` (the 4 edge neighbours, k = 4) or ``"touch"`` (the 8 touching neighbours,
    k = 8; missing neighbours carry weight 0). Float64 normal equations in the tangent plane (a pseudo-inverse
    of the 3x3 matrix, which has the normal as its null direction)."""
    tb = tables(nside)
    nb = tb.edge if kind == "edge" else tb.touch if kind == "touch" else None
    if nb is None:
        raise ValueError("kind must be 'edge' or 'touch'")
    cen = pixel_centres(nside)
    d = cen[nb.clip(0)] - cen[:, None, :]
    d = d - (d * cen[:, None, :]).sum(-1, keepdims=True) * cen[:, None, :]  # onto the tangent plane at p
    d = np.where((nb >= 0)[..., None], d, 0.0)
    a = np.einsum("pji,pjk->pik", d, d)
    return np.einsum("pik,pjk->pji", np.linalg.pinv(a), d)


def lsq_condition(nside: int, kind: str = "touch") -> np.ndarray:
    """Condition number (largest/smallest non-zero singular value) of the tangent neighbour matrix of each
    pixel, the conditioning of ``lsq_weights``."""
    tb = tables(nside)
    nb = tb.edge if kind == "edge" else tb.touch
    cen = pixel_centres(nside)
    d = cen[nb.clip(0)] - cen[:, None, :]
    d = np.where(
        (nb >= 0)[..., None], d - (d * cen[:, None, :]).sum(-1, keepdims=True) * cen[:, None, :], 0.0
    )
    s = np.linalg.svd(d, compute_uv=False)  # (N, 3), the third is 0
    return s[:, 0] / s[:, 1]


def parent_table(nside_fine: int, ratio: int) -> np.ndarray:
    """(Npix_fine,) int32: the pixel of Nside ``nside_fine/ratio`` that contains each fine pixel, for ANY
    integer ratio: the grid of Nside ``r·N`` refines the grid of Nside ``N`` exactly, so the parent is
    ``(ix // r, iy // r, face)``."""
    if nside_fine % ratio:
        raise ValueError("nside_fine must be a multiple of ratio")
    ix, iy, face = pix2xyf(nside_fine, np.arange(npix(nside_fine)))
    return xyf2pix(nside_fine // ratio, ix // ratio, iy // ratio, face).astype(np.int32)


def children_table(nside_coarse: int, ratio: int) -> np.ndarray:
    """(Npix_coarse, ratio²) int32: the fine pixels of each coarse pixel, ascending."""
    par = parent_table(nside_coarse * ratio, ratio)
    order = np.argsort(par, kind="stable")
    return order.reshape(npix(nside_coarse), ratio * ratio).astype(np.int32)


def greedy_colouring(nside: int) -> tuple[np.ndarray, int]:
    """A proper colouring of the graph whose edges join pixels within two touching steps (the cells of one
    colour are pairwise at least three apart). Greedy in two orders (ring order, and largest conflict degree
    first); the one with fewer colours wins (ties: ring order). Returns ``(colour (N,) int32, n_colours)``.
    The lower bound is 9; the greedy gives 15, 16, 15 at Nside 8, 11, 21 (``min_colouring``, S3, gives 11)."""
    tb = tables(nside)
    conflict = tb.stencil2[:, 1:]
    deg = (conflict >= 0).sum(axis=1)
    best: tuple[np.ndarray, int] | None = None
    for order in (np.arange(tb.npix), np.argsort(-deg, kind="stable")):
        col = np.full(tb.npix, -1, dtype=np.int64)
        for p in order.tolist():
            used = col[conflict[p][conflict[p] >= 0]]
            used = used[used >= 0]
            mark = np.zeros(int(used.max()) + 2 if used.size else 1, dtype=bool)
            mark[used] = True
            col[p] = int(np.argmin(mark))
        k = int(col.max()) + 1
        if best is None or k < best[1]:
            best = (col.astype(np.int32), k)
    assert best is not None
    return best


COLOURING_TABU_ITERS = 20000  # tabu-search iterations tried at each colour count below the iterated greedy


def _greedy_order(order: np.ndarray, conflict: np.ndarray) -> np.ndarray:
    col = np.full(len(conflict), -1, dtype=np.int64)
    for p in order.tolist():
        c = conflict[p][conflict[p] >= 0]
        used = col[c]
        used = used[used >= 0]
        mark = np.zeros(int(used.max()) + 2 if used.size else 1, dtype=bool)
        mark[used] = True
        col[p] = int(np.argmin(mark))
    return col


def _dsatur(conflict: np.ndarray) -> np.ndarray:
    """DSATUR (largest saturation first, ties by degree then by index): a proper colouring."""
    import heapq

    n = len(conflict)
    deg = (conflict >= 0).sum(axis=1)
    col = np.full(n, -1, dtype=np.int64)
    sat: list[set[int]] = [set() for _ in range(n)]
    heap = [(0, -int(deg[p]), p) for p in range(n)]
    heapq.heapify(heap)
    while heap:
        s, _, p = heapq.heappop(heap)
        if col[p] >= 0:
            continue
        if -s != len(sat[p]):  # a stale entry: the saturation grew since it was pushed
            heapq.heappush(heap, (-len(sat[p]), -int(deg[p]), p))
            continue
        c = 0
        while c in sat[p]:
            c += 1
        col[p] = c
        for q in conflict[p].tolist():
            if q >= 0 and col[q] < 0 and c not in sat[q]:
                sat[q].add(c)
                heapq.heappush(heap, (-len(sat[q]), -int(deg[q]), q))
    return col


def _tabucol(conflict: np.ndarray, k: int, init: np.ndarray, iters: int) -> np.ndarray | None:
    """Tabu search (Hertz and de Werra) for a proper ``k``-colouring from ``init`` (colours ``>= k`` folded
    onto ``k - 1``); the colouring if one is found within ``iters`` moves, else None. Deterministic: the
    legacy ``RandomState`` stream is frozen by NumPy's compatibility guarantee."""
    n = len(conflict)
    rs = np.random.RandomState(12345 + k)
    nb = [c[c >= 0] for c in conflict]
    col = np.minimum(init, k - 1).astype(np.int64)
    gamma = np.zeros((n, k), dtype=np.int32)  # gamma[p, c]: neighbours of p with colour c
    for p in range(n):
        np.add.at(gamma[p], col[nb[p]], 1)
    tabu = np.zeros((n, k), dtype=np.int64)
    rows = np.arange(n)
    f = int(gamma[rows, col].sum() // 2)  # conflicting pairs
    best, big = f, 1 << 20
    for it in range(iters):
        if f == 0:
            return col
        bad = np.flatnonzero(gamma[rows, col] > 0)
        delta = gamma[bad] - gamma[bad, col[bad]][:, None]  # (m, k): change of f for each move
        delta[np.arange(len(bad)), col[bad]] = big
        ok = (tabu[bad] <= it) | (f + delta < best)  # tabu unless it beats the best (aspiration)
        d2 = np.where(ok, delta, big)
        m = int(d2.min())
        if m >= big:
            continue
        cand = np.argwhere(d2 == m)
        i, c2 = cand[rs.randint(len(cand))]
        p, old = int(bad[i]), int(col[bad[i]])
        col[p] = c2
        np.subtract.at(gamma[:, old], nb[p], 1)
        np.add.at(gamma[:, c2], nb[p], 1)
        f += m
        tabu[p, old] = it + int(0.6 * len(bad)) + rs.randint(10)
        best = min(best, f)
    return col if f == 0 else None


@functools.cache
def min_colouring(nside: int) -> tuple[np.ndarray, int]:
    """A proper colouring of the distance-2 touching graph with few colours (read-only):
    ``(colour (N,) int32, n_colours)``.

    DSATUR, improved by iterated greedy (Culberson: the classes in reverse, by size, rotated), then a tabu
    search for one colour less while it succeeds within ``COLOURING_TABU_ITERS`` moves. Fully deterministic.
    ``greedy_colouring`` (S0) gives 15, 16 and 15 colours at Nside 8, 11 and 21 (the ring order alone 15, 16,
    18); this gives 11 at the three. The lower bound is 9: a pixel and its 8 neighbours are mutually within
    two steps. The box's closing cycle is ``px² (·3)`` phases, 27 to 48; a sphere's is
    ``min(FZ, 3)·n_colours``.
    """
    tb = tables(nside)
    conflict = tb.stencil2[:, 1:]
    best = _dsatur(conflict)
    for r in range(30):  # iterated greedy never uses more colours than its input
        classes = [np.flatnonzero(best == c) for c in range(int(best.max()) + 1)]
        if r % 3 == 0:
            classes = classes[::-1]
        elif r % 3 == 1:
            classes = sorted(classes, key=lambda a: -len(a))
        else:
            classes = classes[r // 3 % len(classes) :] + classes[: r // 3 % len(classes)]
        new = _greedy_order(np.concatenate(classes), conflict)
        if new.max() <= best.max():
            best = new
    k = int(best.max()) + 1
    while k > 9:  # the clique of a pixel and its 8 neighbours
        found = _tabucol(conflict, k - 1, best, COLOURING_TABU_ITERS)
        if found is None:
            break
        best, k = found, k - 1
    col = _frozen(best.astype(np.int32))
    return col, int(col.max()) + 1


def check_colouring(nside: int, colour: np.ndarray) -> bool:
    """True iff cells of one colour are pairwise at least three touching steps apart (a proper colouring of
    the distance-2 graph)."""
    conflict = tables(nside).stencil2[:, 1:]
    ok = conflict >= 0
    return bool((colour[:, None] != colour[np.where(ok, conflict, 0)])[ok].all())


# --- the exact coverage of the hash stencils ---------------------------------------------------------------


def _edge_endpoints(nside: int) -> tuple[np.ndarray, np.ndarray]:
    """Per (pixel, edge): the two ends in integer facet coordinates ``(N, 4, 4)`` = (x0, y0, x1, y1) and the
    face ``(N,)``."""
    ix, iy, face = pix2xyf(nside, np.arange(npix(nside)))
    ends = np.empty((len(ix), 4, 4), dtype=np.int64)
    for k in range(4):
        (dx0, dy0), (dx1, dy1) = CORNERS[k], CORNERS[(k + 1) % 4]
        ends[:, k] = np.stack([ix + dx0, iy + dy0, ix + dx1, iy + dy1], axis=1)
    return ends, face


def _curve(nside: int, face: np.ndarray, ends: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Points of the edge curves at parameters ``s`` (broadcast with the arrays)."""
    x = (ends[..., 0] + (ends[..., 2] - ends[..., 0]) * s) / nside
    y = (ends[..., 1] + (ends[..., 3] - ends[..., 1]) * s) / nside
    return xyf2vec(x, y, face)


def _angle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Angle between unit vectors, accurate for small angles."""
    return np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1), (a * b).sum(-1))


def _golden(f, lo: np.ndarray, hi: np.ndarray, iters: int) -> np.ndarray:
    """Vectorised golden-section minimisation of f over [lo, hi]; returns the arg-min among the final bracket
    points and the two ends (a minimum at a curve's end point is then exact, not converged to 1e-8)."""
    lo0, hi0 = lo, hi
    r = 0.5 * (math.sqrt(5.0) - 1.0)
    x1, x2 = hi - r * (hi - lo), lo + r * (hi - lo)
    f1, f2 = f(x1), f(x2)
    for _ in range(iters):
        left = f1 < f2  # the minimum lies in [lo, x2]
        hi = np.where(left, x2, hi)
        lo = np.where(left, lo, x1)
        new = np.where(left, hi - r * (hi - lo), lo + r * (hi - lo))
        fn = f(new)
        x1, x2, f1, f2 = (
            np.where(left, new, x2),
            np.where(left, x1, new),
            np.where(left, fn, f2),
            np.where(left, f1, fn),
        )
    cand = np.stack([x1, x2, lo0, hi0])
    val = np.stack([f1, f2, f(lo0), f(hi0)])
    return np.take_along_axis(cand, val.argmin(axis=0)[None], axis=0)[0]


@dataclass(frozen=True)
class Coverage:
    """``angle[p]``: the geodesic distance on the unit sphere between the boundary of cell p and the boundary
    of its reach-``reach`` stencil union (every point of p has a ball of that angular radius inside the
    stencil); ``witness_a[p]`` / ``witness_b[p]`` the nearest pair of boundary points."""

    nside: int
    reach: int
    angle: np.ndarray  # (N,) radians
    witness_a: np.ndarray  # (N, 3) on the boundary of the cell
    witness_b: np.ndarray  # (N, 3) on the boundary of the stencil union

    @property
    def unit(self) -> float:
        """``sqrt(4π/Npix)``, the unit of the quoted coverages."""
        return pixel_side(self.nside)

    @property
    def worst(self) -> int:
        return int(np.argmin(self.angle))

    @property
    def min_rel(self) -> float:
        """The minimum coverage in units of ``sqrt(4π/Npix)``."""
        return float(self.angle.min() / self.unit)

    @property
    def min_angle(self) -> float:
        return float(self.angle.min())


COARSE_SAMPLES = 16  # samples per edge in the first (bracketing) stage
REFINE_ITERS = 24  # golden-section iterations per parameter (the bracket shrinks by 0.618^24 ~ 1e-5)
REFINE_WINDOW = 0.005  # pairs within this many cell sides of the pixel's coarse minimum are refined (the
# refinement gains at most ~0.002 sides over the best sample pair; a window of 0.02 gives the same table
# to 1e-15)


@functools.cache
def coverage(nside: int, reach: int) -> Coverage:
    """The exact coverage table of the reach-1 or reach-2 hash stencil at ``nside`` (cached; about 3 s at
    Nside 21, scaling with Npix).

    Stage 1 samples every pixel edge at ``COARSE_SAMPLES + 1`` points and finds, for each cell edge and each
    boundary edge of the stencil union, the closest sample pair. Stage 2 refines every pair whose coarse
    distance is within ``REFINE_WINDOW`` of the pixel's coarse minimum by nested golden-section search
    over the
    two edge parameters (the edges are curves, not great arcs), starting from the bracket of one sample step
    around the coarse pair. The minimum over the refined pairs is the coverage. Boundary edges of the
    union are
    the pixel edges whose far side is not in the stencil; a stencil is a closed set, so its boundary is made
    of whole pixel edges.
    """
    n = int(nside)
    tb = tables(n)
    st = edge_stencil(n, reach)
    tot = tb.npix
    ends, face = _edge_endpoints(n)
    m = COARSE_SAMPLES
    sgrid = np.linspace(0.0, 1.0, m + 1)
    pts = _curve(n, face[:, None, None], ends[:, :, None, :], sgrid[None, None, :])  # (N, 4, m+1, 3)
    unit = pixel_side(n)
    picked: list[tuple[np.ndarray, ...]] = []
    chunk = max(1, 2_000_000 // (m + 1) ** 2 // 20)
    for c0 in range(0, tot, chunk):
        p = np.arange(c0, min(tot, c0 + chunk))
        s_c = st[p]  # (c, S)
        valid = s_c >= 0
        en = tb.edge[s_c.clip(0)]  # (c, S, 4) the pixel across each member edge
        inside = (en[..., None] == s_c[:, None, None, :]).any(-1)
        bmask = (valid[..., None] & ~inside).reshape(len(p), -1)  # boundary edges of the union
        nb = bmask.sum(axis=1)
        bmax = int(nb.max())
        sel = np.argsort(~bmask, axis=1, kind="stable")[:, :bmax]  # boundary slots first
        bq = np.take_along_axis(s_c, sel // 4, axis=1)  # (c, B) pixel owning the boundary edge
        bk = sel % 4
        bvalid = np.arange(bmax)[None, :] < nb[:, None]
        bq = np.where(bvalid, bq, 0)
        bpts = pts[bq, bk]  # (c, B, m+1, 3)
        own = pts[p]  # (c, 4, m+1, 3)
        # dot products between every own sample and every boundary sample: (c, 4, m+1, B, m+1)
        dots = np.einsum("caix,cbjx->caibj", own, bpts)
        dots = np.where(bvalid[:, None, None, :, None], dots, -2.0)
        flat = dots.transpose(0, 1, 3, 2, 4).reshape(len(p), 4, bmax, -1)  # (c, 4, B, (m+1)^2)
        best = flat.argmax(axis=-1)  # (c, 4, B)
        bd = np.take_along_axis(flat, best[..., None], axis=-1)[..., 0]
        coarse = np.arccos(np.clip(bd, -1.0, 1.0))
        coarse = np.where(bvalid[:, None, :], coarse, np.inf)
        cmin = coarse.min(axis=(1, 2))
        pick = coarse <= (cmin + REFINE_WINDOW * unit)[:, None, None]
        ic, ia, ib = np.nonzero(pick)
        bi, bj = np.divmod(best[ic, ia, ib], m + 1)  # sample index on the own edge, on the boundary edge
        pix = p[ic]
        fa, ea = face[pix], ends[pix, ia]
        fb, eb = face[bq[ic, ib]], ends[bq[ic, ib], bk[ic, ib]]
        picked.append((pix, fa, ea, fb, eb, bi, bj))
    pix, fa, ea, fb, eb, bi, bj = (np.concatenate(x) for x in zip(*picked, strict=True))
    h = 1.0 / m
    s0, t0 = bi * h, bj * h
    lo_s, hi_s = np.clip(s0 - h, 0.0, 1.0), np.clip(s0 + h, 0.0, 1.0)
    lo_t, hi_t = np.clip(t0 - h, 0.0, 1.0), np.clip(t0 + h, 0.0, 1.0)

    def inner(t):
        b = _curve(n, fb, eb, t)
        s = _golden(lambda x: _angle(_curve(n, fa, ea, x), b), lo_s, hi_s, REFINE_ITERS)
        return s, _angle(_curve(n, fa, ea, s), b)

    t_star = _golden(lambda t: inner(t)[1], lo_t, hi_t, REFINE_ITERS)
    s_star, ang = inner(t_star)
    out_angle = np.full(tot, np.inf)
    np.minimum.at(out_angle, pix, ang)
    # witnesses: the refined pair that attained the minimum of its pixel
    won = np.flatnonzero(ang <= out_angle[pix])
    out_a, out_b = np.zeros((tot, 3)), np.zeros((tot, 3))
    out_a[pix[won]] = _curve(n, fa[won], ea[won], s_star[won])
    out_b[pix[won]] = _curve(n, fb[won], eb[won], t_star[won])
    return Coverage(n, reach, _frozen(out_angle), _frozen(out_a), _frozen(out_b))


# --- the hash derivation: Nh and C -------------------------------------------------------------------------


def rigorous_angle(radius: float, r: float, sense: float) -> float:
    """The largest angle between the directions of two bodies whose centres are closer than ``sense`` and lie
    at least ``radius + r`` from the planet centre: ``2·asin(R / (2·r_lo))`` (equal radii at the floor are the
    worst case; the tangent configuration of unequal radii gives ``atan(R / r_lo)``, which is smaller)."""
    # pi when the ball is wider than the planet
    return 2.0 * math.asin(min(1.0, sense / (2.0 * (radius + r))))


# bump when ``coverage`` changes its definition: invalidates the shipped table and the disk cache
COVERAGE_ALGORITHM = 1
_COVERAGE_TABLE = Path(__file__).with_name("data") / "healpix_coverage_reach2.json"
_min_angle_memo: dict[tuple[int, int], float] = {}


@functools.cache
def _shipped_coverage() -> dict[int, float]:
    """The shipped table ``Nside -> exact reach-2 coverage angle`` (Nside 2..72), regenerated by
    ``tests/test_geom_coverage.py`` samples; empty if the file is missing or of another algorithm."""
    try:
        data = json.loads(_COVERAGE_TABLE.read_text(encoding="utf-8"))
        if data.get("algorithm") == COVERAGE_ALGORITHM and data.get("reach") == 2:
            return {int(k): float(v) for k, v in data["min_angle"].items()}
    except (OSError, ValueError, KeyError):
        pass
    return {}


def _disk_cache_dir() -> Path:
    base = os.environ.get("ALIFE_CACHE_DIR") or os.path.join(os.path.expanduser("~"), ".cache", "alife")
    return Path(base) / "geom"


def min_coverage_angle(nside: int, reach: int = 2) -> float:
    """The exact minimum coverage angle (radians) of the reach-``reach`` stencil at ``nside``: the one
    number the
    hash derivation needs. Looked up in this order: the in-process memo, the shipped table (reach 2,
    Nside up to
    72), a persistent disk cache (``$ALIFE_CACHE_DIR`` or ``~/.cache/alife/geom``, keyed by Nside, reach and
    ``COVERAGE_ALGORITHM``), and only then computed with ``coverage`` (seconds; about 40 s at Nside 64) and
    written back. A disk failure is silent: the cache is an optimisation."""
    key = (int(nside), int(reach))
    if key in _min_angle_memo:
        return _min_angle_memo[key]
    value = _shipped_coverage().get(key[0]) if reach == 2 else None
    path = _disk_cache_dir() / f"cov_a{COVERAGE_ALGORITHM}_r{reach}_n{nside}.json"
    if value is None:
        try:
            value = float(json.loads(path.read_text(encoding="utf-8"))["min_angle"])
        except (OSError, ValueError, KeyError):
            value = None
    if value is None:
        value = coverage(key[0], key[1]).min_angle
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(f".{os.getpid()}.tmp")
            tmp.write_text(json.dumps({"min_angle": value}), encoding="utf-8")
            os.replace(tmp, path)
        except OSError:
            pass
    _min_angle_memo[key] = value
    return value


def hash_margin(nside: int, angle: float, reach: int = 2) -> float:
    """Exact reach-``reach`` coverage at ``nside`` over the rigorous ``angle``, minus 1."""
    return min_coverage_angle(nside, reach) / angle - 1.0


def estimate_nh(angle: float) -> int:
    """The Nside at which the reach-2 coverage (about 1.40·sqrt(4π/Npix) = 1.40·sqrt(π/3)/N) equals ``angle``:
    the starting point of ``derive_nh`` and a cheap upper bound for refusing absurd planets."""
    return max(2, int(math.sqrt(math.pi / 3.0) * 1.40 / angle))


def derive_nh(radius: float, r: float, sense: float, reach: int = 2) -> int:
    """The hash Nside ``Nh``: the largest Nside whose exact coverage angle ``cov·sqrt(4π/Npix)`` is at
    least the
    rigorous angle. The search starts from the asymptotic estimate and walks to the boundary, looking up
    the exact
    angles of the Nsides it visits (``min_coverage_angle``: shipped table, disk cache, or computed).
    Returns 0 when
    even Nside 2 fails, which includes a ball wider than the planet (the angle is then π)."""
    angle = rigorous_angle(radius, r, sense)
    n = estimate_nh(angle)
    while n > 2 and min_coverage_angle(n, reach) < angle:
        n -= 1
    if min_coverage_angle(n, reach) < angle:
        return 0
    while min_coverage_angle(n + 1, reach) >= angle:
        n += 1
    return n


def hash_layers(height: float, sense: float) -> tuple[int, float]:
    """``(G_z, T_h)``: ``G_z = floor(2H/R)`` equal-thickness hash layers, ``T_h = H/G_z >= R/2``."""
    gz = math.floor(2.0 * height / sense)
    return gz, (height / gz if gz > 0 else math.inf)


AGENT_DENSITY_MAX = 0.167  # agents per unit^3 reached by the repo's crowd tests (36 in a 216 cell)


def lid_cell_volume(radius: float, height: float, sense: float, nh: int) -> float:
    """Volume of the largest hash cell (the lid cell): ``Ω_cell · r_top² · T_h``."""
    _, th = hash_layers(height, sense)
    return pixel_solid_angle(nh) * (radius + height) ** 2 * th


def min_cell_capacity(radius: float, height: float, sense: float, nh: int) -> int:
    """The smallest admissible ``sensing.cell_capacity``: ``ceil(0.167·V_lid) + 4``."""
    return math.ceil(AGENT_DENSITY_MAX * lid_cell_volume(radius, height, sense, nh)) + 4


# --- equal-volume shell layers -----------------------------------------------------------------------------


def layer_radii(radius: float, height: float, fz: int) -> np.ndarray:
    """(FZ+1,) float64: ``r_k = (Rp³ + k·(r_top³ − Rp³)/FZ)^(1/3)``; layer k is ``r_k <= r < r_{k+1}``."""
    rt = radius + height
    return np.cbrt(radius**3 + np.arange(fz + 1) * ((rt**3 - radius**3) / fz))


def layer_edges(radius: float, height: float, fz: int) -> np.ndarray:
    """(FZ+1,) heights ``z_k = r_k − Rp`` of the layer interfaces."""
    return layer_radii(radius, height, fz) - radius


def layer_volume(radius: float, height: float, fz: int, nside: int) -> float:
    """Volume of every (angular cell, layer): ``(4π/3)(r_top³ − Rp³)/(Npix·FZ)``."""
    rt = radius + height
    return (4.0 * math.pi / 3.0) * (rt**3 - radius**3) / (npix(nside) * fz)


def layer_of(rho: np.ndarray, radius: float, height: float, fz: int) -> np.ndarray:
    """Layer index of each distance from the planet centre: ``clip(floor((ρ³ − Rp³)/ΔV), 0, FZ−1)``."""
    dv = ((radius + height) ** 3 - radius**3) / fz
    return np.clip(np.floor((np.asarray(rho, dtype=np.float64) ** 3 - radius**3) / dv), 0, fz - 1).astype(
        np.int64
    )


# --- the geometry accessors of the host (DESIGN §46.12) ----------------------------------------------------


def _min_image(d: np.ndarray, size: float) -> np.ndarray:
    return d - size * np.rint(d / size)  # rint rounds half to even, like mx.round


class Box:
    """The torus-and-floor world, as the engine's present expressions written once (host, numpy)."""

    name = "box"

    def __init__(self, size: float, field_cell: float, dims: int, height: float):
        self.size, self.field_cell, self.dims, self.height_max = size, field_cell, dims, height
        self.fg = round(size / field_cell)
        self.fz = round(height / field_cell) if dims == 3 else 1

    def height(self, pos):
        """z of a 3D position; a 2D world has no height (ValueError)."""
        pos = np.asarray(pos)
        if pos.shape[-1] != 3:
            raise ValueError("Box.height needs 3D positions (a 2D world has no height)")
        return pos[..., 2]

    def up(self, pos):
        pos = np.asarray(pos)
        out = np.zeros(pos.shape, dtype=pos.dtype)
        out[..., 2] = 1
        return out

    def radial(self, vec, pos):
        """The vertical component of a 3D vector (ValueError in 2D)."""
        vec = np.asarray(vec)
        if vec.shape[-1] != 3:
            raise ValueError("Box.radial needs 3D vectors (a 2D world has no vertical)")
        return vec[..., 2]

    def tangent(self, vec, pos):
        v = np.array(vec, copy=True)
        v[..., 2] = 0
        return v

    def displacement(self, a, b):
        return _min_image(np.asarray(b) - np.asarray(a), self.size)

    def distance(self, a, b):
        return np.linalg.norm(self.displacement(a, b), axis=-1)

    def arc(self, a, b):
        return self.distance(a, b)

    def cell_of(self, pos):
        pos = np.asarray(pos)
        fc, fg = self.field_cell, self.fg
        cxy = np.floor(pos[..., :2] / fc).astype(np.int64) % fg
        if self.dims == 3:
            cz = np.clip(np.floor(pos[..., 2] / fc).astype(np.int64), 0, self.fz - 1)
            return (cz * fg + cxy[..., 1]) * fg + cxy[..., 0]
        return cxy[..., 1] * fg + cxy[..., 0]

    def centroid(self, points, weights=None):
        """Circular mean on the torus in x and y, weighted mean in z (3D)."""
        p = np.asarray(points, dtype=np.float64)
        w = np.ones(len(p)) if weights is None else np.asarray(weights, dtype=np.float64)
        ang = p[:, :2] * (TWO_PI / self.size)
        mean = np.arctan2((w[:, None] * np.sin(ang)).sum(0), (w[:, None] * np.cos(ang)).sum(0))
        xy = (mean % TWO_PI) * self.size / TWO_PI % self.size
        out = np.concatenate([xy, [(w * p[:, 2]).sum() / w.sum()]]) if p.shape[1] == 3 else xy
        return out

    def bearing(self, vec, pos):
        """The engine's heading angle of the horizontal part of ``vec``: ``atan2(vy, vx)``."""
        v = np.asarray(vec)
        return np.arctan2(v[..., 1], v[..., 0])


class Sphere:
    """The sphere world (host, numpy): a shell of thickness H on a planet of radius Rp, HEALPix cells of Nside
    ``nside`` and ``layers`` equal-volume layers."""

    name = "sphere"

    def __init__(self, radius: float, height: float, nside: int, layers: int):
        self.radius, self.height_max, self.nside, self.layers = radius, height, nside, layers

    def _norm(self, pos):
        return np.linalg.norm(np.asarray(pos, dtype=np.float64), axis=-1, keepdims=True)

    def height(self, pos):
        return self._norm(pos)[..., 0] - self.radius

    def up(self, pos):
        p = np.asarray(pos, dtype=np.float64)
        return p / np.maximum(self._norm(p), 1e-300)

    def radial(self, vec, pos):
        return (np.asarray(vec, dtype=np.float64) * self.up(pos)).sum(-1)

    def tangent(self, vec, pos):
        v = np.asarray(vec, dtype=np.float64)
        u = self.up(pos)
        return v - (v * u).sum(-1, keepdims=True) * u

    def displacement(self, a, b):
        return np.asarray(b) - np.asarray(a)

    def distance(self, a, b):
        return np.linalg.norm(self.displacement(a, b), axis=-1)

    def arc(self, a, b):
        """Great-circle arc between the directions of ``a`` and ``b`` at their mean radius."""
        ua, ub = self.up(a), self.up(b)
        ang = np.arctan2(np.linalg.norm(np.cross(ua, ub), axis=-1), (ua * ub).sum(-1))
        return ang * 0.5 * (self._norm(a)[..., 0] + self._norm(b)[..., 0])

    def cell_of(self, pos):
        """Flat field-cell index ``layer·Npix + pixel`` (z-major, as the box)."""
        p = np.asarray(pos, dtype=np.float64)
        rho = self._norm(p)[..., 0]
        lay = layer_of(rho, self.radius, self.height_max, self.layers)
        return lay * npix(self.nside) + ang2pix(p, self.nside)

    def centroid(self, points, weights=None):
        """Weighted centroid on the sphere: the direction of the weighted sum of unit vectors, at the weighted
        mean radius (the first point's direction if the sum cancels)."""
        p = np.asarray(points, dtype=np.float64)
        w = np.ones(len(p)) if weights is None else np.asarray(weights, dtype=np.float64)
        u = self.up(p)
        s = (w[:, None] * u).sum(0)
        nrm = np.linalg.norm(s)
        d = s / nrm if nrm > 1e-12 * w.sum() else u[0]
        return d * ((w * self._norm(p)[:, 0]).sum() / w.sum())

    def bearing(self, vec, pos):
        """Compass bearing of the tangent part of ``vec`` from local north: ``atan2(v·east, v·north)`` in
        [−π, π) (+π mapped to −π). A different convention from the box's heading angle."""
        u = self.up(pos)
        e, nn = local_frame(u)
        v = np.asarray(vec, dtype=np.float64)
        b = np.arctan2((v * e).sum(-1), (v * nn).sum(-1))
        return np.where(b >= math.pi, b - TWO_PI, b)


def geometry(cfg: Config) -> Box | Sphere:
    """The static geometry object of a config (host)."""
    w = cfg.world
    if w.geometry == "sphere":
        return Sphere(w.radius, w.height, w.nside, w.layers)
    return Box(w.size, w.field_cell, w.dims, w.height)
