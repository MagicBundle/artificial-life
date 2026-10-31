"""Scenes and float64 references of the far-hearing tests (DESIGN §47.7, step K2).

Everything here is written independently of ``alife.calltab`` and ``alife.calls``: the reference table is a
double loop over pixel centres from ``hp_oracle`` (a separate HEALPix implementation) or over torus images,
the reference pipeline a Python loop over the callers and the table row, and the exact reference a sum over
caller-hearer pairs of the true positions. ``scene`` builds the namespace that ``calls.hear_far`` reads
(``alive``, ``pos``, ``hdg`` / ``heading``) without a whole ``State``.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import mlx.core as mx
import numpy as np

import hp_oracle
from alife import calls
from alife.config import Config, dims


def kernel(d: float, hear: float, far: float, cell: float, atten: float) -> float:
    """The far weight of a centre distance (the formula of DESIGN §47.7), one pair, plain Python."""
    if d < hear or d > far:
        return 0.0
    return min(max((d - hear) / cell, 0.0), 1.0) * max(1.0 - d / far, 0.0) ** atten


def ref_table(cfg: Config) -> tuple[list[list[tuple[int, float, np.ndarray]]], float]:
    """``(rows, cell)``: for every column the list ``(p', w, off)`` of its entries in ascending ``p'``, by
    brute force over every pair of columns (float64). The sphere's centres are ``hp_oracle.pix2vec``; the box
    takes the nearest of the nine torus images."""
    w, hear, far, atten = cfg.world, float(cfg.sensing.radius), float(cfg.calls.far_range), cfg.comm.atten_pow
    rows: list[list[tuple[int, float, np.ndarray]]] = []
    if w.geometry == "sphere":
        c = hp_oracle.pix2vec(w.nside) * w.radius
        cell = w.radius * math.sqrt(math.pi / 3.0) / w.nside
        for p in range(len(c)):
            row = []
            for q in range(len(c)):
                dv = c[q] - c[p]
                d = float(np.sqrt((dv * dv).sum()))
                if hear <= d <= far:
                    row.append((q, kernel(d, hear, far, cell, atten), dv / far))
            rows.append(row)
        return rows, cell
    fg = round(w.size / w.field_cell)
    return box_rows(fg, w.size, hear, far, atten), w.size / fg


def box_rows(
    fg: int, size: float, hear: float, far: float, atten: float
) -> list[list[tuple[int, float, np.ndarray]]]:
    """The box's rows by brute force: for every pair of columns the nearest of the nine torus images, one
    numpy row at a time (an explicit image search, not the rounding of ``min_image`` that ``calltab``
    uses)."""
    cell = size / fg
    cx, cy = (np.arange(fg * fg) % fg + 0.5) * cell, (np.arange(fg * fg) // fg + 0.5) * cell
    shifts = [(ix * size, iy * size) for ix in (-1, 0, 1) for iy in (-1, 0, 1)]
    rows = []
    for p in range(fg * fg):
        dx = np.stack([cx + sx - cx[p] for sx, _ in shifts])  # (9, ncol)
        dy = np.stack([cy + sy - cy[p] for _, sy in shifts])
        k = np.argmin(dx * dx + dy * dy, axis=0)
        ddx, ddy = dx[k, np.arange(fg * fg)], dy[k, np.arange(fg * fg)]
        d = np.sqrt(ddx * ddx + ddy * ddy)
        rows.append(
            [
                (int(q), kernel(float(d[q]), hear, far, cell, atten), np.array([ddx[q], ddy[q]]) / far)
                for q in np.flatnonzero((d >= hear) & (d <= far))
            ]
        )
    return rows


def scene(cfg: Config, n_alive: int, seed: int, *, callers: np.ndarray | None = None) -> SimpleNamespace:
    """Positions, headings and liveness of ``dims(cfg).n`` rows, ``n_alive`` of them alive and uniform in the
    world (the shell between the floor and the lid on a sphere); dead rows rest at the origin. ``callers``
    (N,) bool selects nothing here, it is for the caller to build a voice."""
    d, rs = dims(cfg), np.random.default_rng(seed)
    n = d.n
    alive = np.zeros(n, bool)
    alive[rs.choice(n, n_alive, replace=False)] = True
    ns = SimpleNamespace(alive=mx.array(alive), n=n)
    if d.sph:
        u = rs.normal(size=(n, 3))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        rho = np.cbrt(rs.uniform(d.r_lo**3, d.r_hi**3, n))
        pos = u * rho[:, None]
        t = rs.normal(size=(n, 3))
        t -= (t * u).sum(axis=1, keepdims=True) * u
        t /= np.linalg.norm(t, axis=1, keepdims=True)
        ns.pos, ns.hdg = mx.array(pos.astype(np.float32)), mx.array(t.astype(np.float32))
        ns.heading = mx.zeros((n,), mx.float32)
    else:
        size = cfg.world.size
        pos = rs.uniform(0.0, size, (n, d.nd))
        if d.d3:
            pos[:, 2] = rs.uniform(d.z_lo, d.z_hi, n)
        ns.pos = mx.array(pos.astype(np.float32))
        ns.heading = mx.array(rs.uniform(-math.pi, math.pi, n).astype(np.float32))
    ns.step = mx.array(0, mx.int32)
    return ns


def columns(cfg: Config, ns: SimpleNamespace) -> mx.array:
    """The column of every row: the engine's own path (``fields.cell_index`` then ``calls.column_of``)."""
    from alife import fields

    return calls.column_of(fields.cell_index(ns.pos, cfg), cfg, calls.runtime(cfg))


def hear(cfg: Config, ns: SimpleNamespace, voice: np.ndarray | mx.array) -> tuple[np.ndarray, np.ndarray]:
    """``calls.hear_far`` on a scene: ``(voice_far, far_bearing)`` as numpy."""
    v = voice if isinstance(voice, mx.array) else mx.array(np.asarray(voice, np.float32))
    h, b = calls.hear_far(ns, v, columns(cfg, ns), calls.runtime(cfg), cfg)
    mx.eval(h, b)
    return np.array(h), np.array(b)


def frame(cfg: Config, ns: SimpleNamespace, i: int) -> tuple[np.ndarray, np.ndarray]:
    """The planet-frame axes ``(fwd, lat)`` of row ``i`` (float64): ``hdg`` and ``up × hdg`` on a sphere, the
    yaw ``(cos, sin)`` and ``(-sin, cos)`` in a box (a 0 z in 3D)."""
    d = dims(cfg)
    if d.sph:
        pos, hdg = np.array(ns.pos[i], np.float64), np.array(ns.hdg[i], np.float64)
        up = pos / np.linalg.norm(pos)
        return hdg, np.cross(up, hdg)
    th = float(ns.heading[i])
    f, la = np.array([math.cos(th), math.sin(th)]), np.array([-math.sin(th), math.cos(th)])
    return (np.append(f, 0.0), np.append(la, 0.0)) if d.d3 else (f, la)


def ref_hear(
    cfg: Config, ns: SimpleNamespace, voice: np.ndarray, rows: list[list[tuple[int, float, np.ndarray]]]
) -> tuple[np.ndarray, np.ndarray]:
    """The float64 reference of ``hear_far``: deposit the callers (by a Python loop), sum each hearer's table
    row, ``1 - exp(-x / far_xref)``, the bearing in the hearer's frame, 0 for a dead row."""
    d = dims(cfg)
    col = np.array(columns(cfg, ns))
    alive = np.array(ns.alive)
    n, w = d.n, voice.shape[1]
    ncol = len(rows)
    dep = np.zeros((w, ncol))
    for i in range(n):
        if alive[i]:
            for c in range(w):
                dep[c, col[i]] += min(255.0, math.floor(float(np.float32(voice[i, c])) * 256.0))
    heard = np.zeros((n, w))
    bear = np.zeros((n, 2 * w))
    for i in range(n):
        if not alive[i]:
            continue
        fwd, lat = frame(cfg, ns, i)
        for c in range(w):
            x, num = 0.0, 0.0
            for q, wt, off in rows[col[i]]:
                x += wt * dep[c, q] / 256.0
                num = num + wt * dep[c, q] / 256.0 * off
            b = num / max(x, 1e-3)
            if d.sph:
                up = np.array(ns.pos[i], np.float64)
                up /= np.linalg.norm(up)
                b = b - (b @ up) * up
            heard[i, c] = 1.0 - math.exp(-x / cfg.calls.far_xref)
            bear[i, 2 * c] = b @ fwd[: len(b)]  # a box's offsets are horizontal: a 3D frame has a 0 z
            bear[i, 2 * c + 1] = b @ lat[: len(b)]
    return heard, bear


def exact_sum(
    cfg: Config, ns: SimpleNamespace, voice: np.ndarray, hearers: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The exact far sum of the true positions: ``x[h, c] = Σ_callers kernel(d(h, caller)) · q / 256`` with
    ``d`` the chord of the two DIRECTIONS on the floor sphere (a sphere) or the torus distance of the
    horizontal positions (a box), no grid, and the matching exact bearing ``Σ kernel·q·(c - h) / far / max(Σ,
    1e-3)`` (the horizontal offset). Returns ``(x (len(hearers), W), bearing (len(hearers), W, nt))``."""
    d = dims(cfg)
    w = cfg.world
    hear_r, far, atten = float(cfg.sensing.radius), float(cfg.calls.far_range), cfg.comm.atten_pow
    src = np.flatnonzero(np.array(ns.alive))  # only the alive rows can call
    pos = np.array(ns.pos, np.float64)[src]
    hpos = np.array(ns.pos, np.float64)[hearers]
    if d.sph:
        u = pos / np.linalg.norm(pos, axis=1, keepdims=True) * w.radius
        hu = hpos / np.linalg.norm(hpos, axis=1, keepdims=True) * w.radius
        cell = w.radius * math.sqrt(math.pi / 3.0) / w.nside
        vec = u[None, :, :] - hu[:, None, :]
    else:
        fg = round(w.size / w.field_cell)
        cell = w.size / fg
        dv = pos[None, :, :2] - hpos[:, None, :2]
        vec = dv - w.size * np.rint(dv / w.size)
    dist = np.sqrt((vec * vec).sum(axis=-1))
    ramp = np.clip((dist - hear_r) / cell, 0.0, 1.0)
    wt = np.where((dist >= hear_r) & (dist <= far), ramp * np.clip(1.0 - dist / far, 0.0, 1.0) ** atten, 0.0)
    q = np.minimum(255.0, np.floor(np.asarray(voice, np.float64)[src] * 256.0))
    x = np.einsum("hn,nc->hc", wt, q) / 256.0
    num = np.einsum("hn,nc,hnt->hct", wt, q, vec / far) / 256.0
    return x, num / np.maximum(x, 1e-3)[..., None]
