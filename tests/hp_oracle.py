"""An independent HEALPix (RING) implementation: a TEST ORACLE for ``alife.geom`` and ``alife.sphere``.

Credit: the independent implementation written by the reviewer of the true-sphere series (``hp.py`` and
``t2_cov.py`` of the sphere review, 2026-10-29), ported with two changes. (1) ``xyf2loc`` wraps the longitude
AFTER the division by the ring width (a defensive form: the original wrapped ``tmp`` by 8 before it, wrong in
principle inside a polar cap when ``tmp`` is negative at rounding level; the two orders agree on every point
the review tested). (2) The sampled coverage of ``t2_cov.py`` is exposed as ``sampled_coverage``.

It shares no code with ``alife.geom``: the neighbour sets come from rounded corner coordinates (``geom``
identifies corners in integer arithmetic) and ``ang2pix`` is a separate float64 closed form. astropy-healpix
was NOT added as a dependency (owner decision D10 was amended: this file and combinatorial checks replace
it), and healpy (GPL-2.0) is not used anywhere.
"""

from collections import defaultdict

import numpy as np

JRLL = np.array([2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4])
JPLL = np.array([1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7])


def isqrt(a):
    r = np.floor(np.sqrt(a.astype(np.float64))).astype(np.int64)
    r = np.where(r * r > a, r - 1, r)
    r = np.where((r + 1) * (r + 1) <= a, r + 1, r)
    return r


def ring2xyf(nside, pix):
    pix = np.asarray(pix, dtype=np.int64)
    npix = 12 * nside * nside
    ncap = 2 * nside * (nside - 1)
    nl2 = 2 * nside
    iring = np.zeros_like(pix)
    iphi = np.zeros_like(pix)
    kshift = np.zeros_like(pix)
    nr = np.zeros_like(pix)
    face = np.zeros_like(pix)
    n = pix < ncap
    p = pix[n]
    ir = (1 + isqrt(1 + 2 * p)) >> 1
    ip = (p + 1) - 2 * ir * (ir - 1)
    iring[n], iphi[n], kshift[n], nr[n], face[n] = ir, ip, 0, ir, (ip - 1) // ir
    e = (pix >= ncap) & (pix < npix - ncap)
    p = pix[e] - ncap
    tmp = p // (4 * nside)
    ir = tmp + nside
    ip = p - tmp * 4 * nside + 1
    ks = (ir + nside) & 1
    ire = tmp + 1
    irm = nl2 + 2 - ire
    ifm = ip - (ire >> 1) + nside - 1
    ifp = ip - (irm >> 1) + nside - 1
    ifm //= nside
    ifp //= nside
    f = np.where(ifp == ifm, ifp | 4, np.where(ifp < ifm, ifp, ifm + 8))
    iring[e], iphi[e], kshift[e], nr[e], face[e] = ir, ip, ks, nside, f
    s = pix >= npix - ncap
    p = npix - pix[s]
    ir = (1 + isqrt(2 * p - 1)) >> 1
    ip = 4 * ir + 1 - (p - 2 * ir * (ir - 1))
    iring[s], iphi[s], kshift[s], nr[s] = nl2 * 2 - ir, ip, 0, ir
    face[s] = 8 + (ip - 1) // ir
    irt = iring - (JRLL[face] * nside) + 1
    ipt = 2 * iphi - JPLL[face] * nr - kshift - 1
    ipt = np.where(ipt >= nl2, ipt - 8 * nside, ipt)
    ix = (ipt - irt) >> 1
    iy = (-(ipt + irt)) >> 1
    return ix, iy, face


def xyf2loc(x, y, face):
    """x,y in [0,1] facet coordinates -> unit vector"""
    jr = JRLL[face] - x - y
    nr = np.where(jr < 1, jr, np.where(jr > 3, 4 - jr, 1.0))
    z = np.where(jr < 1, 1 - nr * nr / 3.0, np.where(jr > 3, nr * nr / 3.0 - 1, (2 - jr) * 2.0 / 3.0))
    t = (JPLL[face] * nr + x - y) / np.where(
        nr < 1e-15, 1, nr
    )  # longitude in units of pi/4, wrapped after the division
    t = np.where(t < 0, t + 8, t)
    t = np.where(t >= 8, t - 8, t)
    phi = np.where(nr < 1e-15, 0.0, 0.25 * np.pi * t)
    st = np.sqrt(np.maximum(0.0, 1 - z * z))
    return np.stack([st * np.cos(phi), st * np.sin(phi), z], -1)


def pix2vec(nside):
    npix = 12 * nside * nside
    ix, iy, f = ring2xyf(nside, np.arange(npix))
    return xyf2loc((ix + 0.5) / nside, (iy + 0.5) / nside, f)


def corners(nside):
    """(npix,4,3) corner vectors, order (0,0),(1,0),(1,1),(0,1) in facet coords"""
    npix = 12 * nside * nside
    ix, iy, f = ring2xyf(nside, np.arange(npix))
    out = []
    for dx, dy in ((0, 0), (1, 0), (1, 1), (0, 1)):
        out.append(xyf2loc((ix + dx) / nside, (iy + dy) / nside, f))
    return np.stack(out, 1)


def ang2pix(v, nside):
    v = np.asarray(v, dtype=np.float64)
    z = np.clip(v[:, 2], -1, 1)
    phi = np.mod(np.arctan2(v[:, 1], v[:, 0]), 2 * np.pi)
    tt = phi / (np.pi / 2)
    za = np.abs(z)
    npix = 12 * nside * nside
    ncap = 2 * nside * (nside - 1)
    pix = np.empty(len(z), np.int64)
    eq = za <= 2 / 3
    t1 = nside * (0.5 + tt[eq])
    t2 = nside * z[eq] * 0.75
    jp = np.floor(t1 - t2).astype(np.int64)
    jm = np.floor(t1 + t2).astype(np.int64)
    ir = nside + 1 + jp - jm
    ks = 1 - (ir & 1)
    ip = ((jp + jm - nside + ks + 1) // 2) % (4 * nside)
    pix[eq] = ncap + (ir - 1) * 4 * nside + ip
    pc = ~eq
    tp = tt[pc] - np.floor(tt[pc])
    tmp = nside * np.sqrt(3 * (1 - za[pc]))
    jp = np.floor(tp * tmp).astype(np.int64)
    jm = np.floor((1 - tp) * tmp).astype(np.int64)
    ir = jp + jm + 1
    ip = np.floor(tt[pc] * ir).astype(np.int64) % (4 * ir)
    north = z[pc] > 0
    pix[pc] = np.where(north, 2 * ir * (ir - 1) + ip, npix - 2 * ir * (ir + 1) + ip)
    return pix


def vertex_ids(nside):
    """unique vertex index for each (pixel, corner) by rounding coordinates"""
    c = corners(nside)
    key = np.round(c.reshape(-1, 3) * 1e8).astype(np.int64)
    # canonicalise -0
    key = key + 0
    _, inv = np.unique(key, axis=0, return_inverse=True)
    return inv.reshape(-1, 4)


def neighbours(nside):
    """touching (shared >=1 vertex) and edge (shared 2 vertices) neighbour sets, exact."""
    npix = 12 * nside * nside
    vid = vertex_ids(nside)
    nv = vid.max() + 1
    byv = defaultdict(list)
    for p in range(npix):
        for v in vid[p]:
            byv[int(v)].append(p)
    touch = [set() for _ in range(npix)]
    shared = [defaultdict(int) for _ in range(npix)]
    for ps in byv.values():
        for a in ps:
            for b in ps:
                if a != b:
                    shared[a][b] += 1
    for a in range(npix):
        touch[a] = set(shared[a].keys())
    edge = [set(b for b, c in shared[a].items() if c >= 2) for a in range(npix)]
    valence = np.bincount(vid.reshape(-1), minlength=nv)
    return touch, edge, valence, vid


# --- the reviewer's sampled coverage (t2_cov.py) ---


def _edge_samples(nside, m):
    npix = 12 * nside * nside
    ix, iy, f = ring2xyf(nside, np.arange(npix))
    t = np.linspace(0, 1, m + 1)
    # four edges of each pixel: param
    segs = []
    for x0, y0, x1, y1 in ((0, 0, 1, 0), (1, 0, 1, 1), (1, 1, 0, 1), (0, 1, 0, 0)):
        pts = []
        for s in t:
            x = (ix + x0 + (x1 - x0) * s) / nside
            y = (iy + y0 + (y1 - y0) * s) / nside
            pts.append(xyf2loc(x, y, f))
        segs.append(np.stack(pts, 1))  # (npix,m+1,3)
    return np.stack(segs, 1)  # (npix,4,m+1,3)


def sampled_coverage(nside, reach, m=24):
    """(coverage / sqrt(4 pi/Npix) per pixel as a CHORD on the unit sphere, stencil sizes): the reviewer's
    sampled boundary-to-boundary method (an upper bound of the exact value, tight to O(1/m²))."""
    npix = 12 * nside * nside
    touch, _edge, _val, vid = neighbours(nside)
    S = [set([p]) | touch[p] for p in range(npix)]
    if reach == 2:
        S = [set().union(*[S[q] for q in S[p]]) for p in range(npix)]
    es = _edge_samples(nside, m)  # (npix,4,m+1,3)
    # edges as vertex-id pairs
    ev = np.stack([vid[:, [0, 1]], vid[:, [1, 2]], vid[:, [2, 3]], vid[:, [3, 0]]], 1)  # (npix,4,2)
    key = np.sort(ev, axis=2)
    # map edge key -> list of (pix, edgeidx)
    ed = defaultdict(list)
    for p in range(npix):
        for k in range(4):
            ed[(key[p, k, 0], key[p, k, 1])].append((p, k))
    res = np.empty(npix)
    for p in range(npix):
        sp = S[p]
        # boundary edges of the stencil union: edges with exactly one adjacent pixel in sp
        bpts = []
        for q in sp:
            for k in range(4):
                nb = [a for a, _ in ed[(key[q, k, 0], key[q, k, 1])]]
                if sum(1 for a in nb if a in sp) == 1:
                    bpts.append(es[q, k])
        B = np.concatenate(bpts, 0)  # (nb,3)
        C = es[p].reshape(-1, 3)
        d = np.linalg.norm(C[:, None, :] - B[None, :, :], axis=2).min()
        res[p] = d
    a = np.sqrt(4 * np.pi / npix)
    return res / a, np.array([len(s) for s in S])
