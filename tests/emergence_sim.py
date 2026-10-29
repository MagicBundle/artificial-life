"""Reversible populations with turnover on a lattice, as Frames (harness for the entropy-production tests).

Adapted from the independent review of E3 (its sim.py): a reversible continuous-time chain with a
heterogeneous stationary law on a G x G torus; organisms die with probability ``p_die`` per frame and are
replaced by births drawn from a chosen cell law. Reversible movement with spatially biased births is the case
the cross-fitted estimator of the closed chain read as irreversibility.
"""

from __future__ import annotations

import numpy as np

from alife.analysis.emergence import frames as fm

SIZE = 64.0
G = 32


def chain(rng: np.random.Generator, hetero_pi: float = 1.0, hetero_c: float = 1.0, dt: float = 0.002):
    """Reversible chain on the torus; returns (cumulative transition rows, stationary law)."""
    n = G * G
    pi = np.exp(hetero_pi * rng.standard_normal(n))
    pi /= pi.sum()
    c = np.zeros((n, n))
    idx = np.arange(n).reshape(G, G)
    for dx, dy in ((1, 0), (0, 1)):
        a = idx.ravel()
        b = np.roll(idx, (-dx, -dy), axis=(0, 1)).ravel()
        w = np.exp(hetero_c * rng.standard_normal(n))
        c[a, b] += w
        c[b, a] += w
    q = c / pi[:, None]
    np.fill_diagonal(q, 0.0)
    q[np.arange(n), np.arange(n)] = -q.sum(1)
    s = np.sqrt(pi)
    sym = q * s[:, None] / s[None, :]
    sym = 0.5 * (sym + sym.T)
    w_, v = np.linalg.eigh(sym)
    p = (v * np.exp(w_ * dt)) @ v.T
    p = np.clip(p * (s[None, :] / s[:, None]), 0, None)
    p /= p.sum(1, keepdims=True)
    return p.cumsum(1), pi


def hot_spot() -> np.ndarray:
    """A Gaussian hot spot of cells at the middle of the torus (a birth law)."""
    ncell = G * G
    ix, iy = np.arange(ncell) // G, np.arange(ncell) % G
    hot = np.exp(-(((ix - 16) ** 2 + (iy - 16) ** 2) / (2 * 4.0**2)))
    return hot / hot.sum()


def simulate(rng, cum, start_dist, birth_dist, n=1200, nf=151, p_die=0.067):
    """Per frame: move, die with probability ``p_die``, refill to ``n`` from ``birth_dist`` (fresh uids)."""
    ncell = G * G
    cur = rng.choice(ncell, n, p=start_dist)
    uid = np.arange(n)
    nxt = n
    us, cells = [uid.copy()], [cur.copy()]
    for _ in range(nf - 1):
        u = rng.random(len(cur))
        cur = np.minimum((u[:, None] > cum[cur]).sum(1), ncell - 1)
        keep = rng.random(len(cur)) > p_die
        cur, uid = cur[keep], uid[keep]
        nb = n - len(cur)
        if nb > 0:
            cur = np.concatenate([cur, rng.choice(ncell, nb, p=birth_dist)])
            uid = np.concatenate([uid, np.arange(nxt, nxt + nb)])
            nxt += nb
        us.append(uid.copy())
        cells.append(cur.copy())
    return us, cells


def to_frames(us, cells, rng, step: int = 100, drift: float | None = None) -> fm.Frames:
    f = SIZE / G
    ps = [np.stack([(c // G + rng.random(len(c))) * f, (c % G + rng.random(len(c))) * f], 1) for c in cells]
    allp = np.concatenate(ps)
    dr = None
    if drift is not None:
        dr = np.zeros((len(allp), 2), np.float16)
        dr[:, 0] = drift
    return fm.Frames(
        meta={"size": SIZE, "height": 0.0},
        step=np.arange(len(us), dtype=np.int64) * step,
        off=np.concatenate([[0], np.cumsum([len(u) for u in us])]).astype(np.int64),
        uid=np.concatenate(us).astype(np.int64),
        row=np.zeros(len(allp), np.int32),
        pos=fm.quantise(allp, SIZE),
        clade=np.zeros(len(allp), np.int32),
        body=np.full(len(allp), -1, np.int32),
        drift=dr,
    )
