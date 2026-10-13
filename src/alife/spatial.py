"""Fixed-K neighbour search on a periodic hash grid in pure MLX (§4; 3D: §27.2). Owned by WP1."""

from __future__ import annotations

from typing import NamedTuple

import mlx.core as mx

from alife import ops
from alife.config import Config, dims

_I32 = mx.int32
# 3×3 stencil offsets, row-major (dy outer, dx inner).
_DX9 = (-1, 0, 1, -1, 0, 1, -1, 0, 1)
_DY9 = (-1, -1, -1, 0, 0, 0, 1, 1, 1)
# 3×3×3 stencil offsets (3D, §27.2): dz outermost, then the 3×3 block above.
_DX27 = _DX9 * 3
_DY27 = _DY9 * 3
_DZ27 = (-1,) * 9 + (0,) * 9 + (1,) * 9


class Neighbors(NamedTuple):
    idx: mx.array  # (N,K) int32; invalid entries hold the agent's own index
    valid: mx.array  # (N,K) bool
    off: mx.array  # (N,K,nd) f32 minimum-image vector me → neighbour; 0 where invalid
    dist: mx.array  # (N,K) f32; R where invalid
    overflow_cells: mx.array  # () int32: hash cells with count > C
    max_occupancy: mx.array  # () int32: largest hash-cell count


def find_neighbors(pos: mx.array, alive: mx.array, cfg: Config) -> Neighbors:
    """K nearest alive agents within ``sensing.radius``, sorted by distance (§4).

    Alive agents are sorted by hash cell; each agent scans the first C sorted entries of each of its 9
    surrounding cells (Q = 9C candidates, gathered from the cell-sorted position copy) and keeps the K
    nearest strictly inside R. Equals brute-force kNN whenever no cell holds more than C agents (F13);
    otherwise only the first C agents of an overfull cell are visible, which ``overflow_cells`` reports.
    A 3D world (``pos`` (N,3)) uses the 3×3×3 search of ``_find_neighbors_3d`` (§27.2).
    """
    d = dims(cfg)
    if d.d3:
        return _find_neighbors_3d(pos, alive, cfg)
    assert pos.shape[-1] == 2, "the 2D neighbour search got 3D positions"  # static: it would ignore z
    n, g, c, k, q = d.n, d.g_hash, d.c, d.k, d.q
    L, R = cfg.world.size, cfg.sensing.radius

    cxy = mx.floor(pos / d.cell_hash).astype(_I32) % g  # (N,2); % G maps pos ≈ L to cell 0
    cid = mx.where(alive, cxy[:, 1] * g + cxy[:, 0], g * g)  # dead → sentinel after all real cells
    order = mx.argsort(cid).astype(_I32)  # (N,)
    start = mx.searchsorted(cid[order], mx.arange(g * g + 1, dtype=_I32), side="left").astype(_I32)
    count = start[1:] - start[:-1]  # (G²,) occupancy of real cells
    spos = pos[order]  # cell-sorted copy: each cell's run is contiguous
    inv = ops.scatter_add_i32(n, order, mx.arange(n, dtype=_I32))  # inv[i] = sorted rank of row i

    ncx = (cxy[:, None, 0] + mx.array(_DX9, dtype=_I32)) % g  # (N,9)
    ncy = (cxy[:, None, 1] + mx.array(_DY9, dtype=_I32)) % g
    ncid = ncy * g + ncx
    s, e = start[ncid], start[ncid + 1]  # (N,9) run [s, e) of each neighbouring cell
    slot3 = s[..., None] + mx.arange(c, dtype=_I32)  # (N,9,C) sorted positions
    ok = (slot3 < e[..., None]).reshape(n, q)
    slot = slot3.reshape(n, q)
    ok = ok & (slot != inv[:, None]) & alive[:, None]
    sl = mx.minimum(slot, n - 1)

    off = ops.min_image(spos[sl] - pos[:, None, :], L)  # (N,Q,2) neighbour − self
    d2 = (off * off).sum(-1)
    d2 = mx.where(ok & (d2 < R * R), d2, mx.inf)
    part = mx.argpartition(d2, kth=k - 1, axis=1)[:, :k].astype(_I32)  # K smallest, unordered
    pd2 = mx.take_along_axis(d2, part, 1)
    o = mx.argsort(pd2, axis=1).astype(_I32)  # nearest first
    part = mx.take_along_axis(part, o, 1)
    pd2 = mx.take_along_axis(pd2, o, 1)
    valid = pd2 < mx.inf

    self_ = mx.arange(n, dtype=_I32)[:, None]
    idx = mx.where(valid, order[mx.take_along_axis(sl, part, 1)], self_)  # ids resolved only for K winners
    offk = mx.where(valid[..., None], mx.take_along_axis(off, part[..., None], 1), 0.0)
    dist = mx.where(valid, mx.sqrt(pd2), R)
    return Neighbors(
        idx=idx,
        valid=valid,
        off=offk,
        dist=dist,
        overflow_cells=(count > c).sum().astype(_I32),
        max_occupancy=count.max().astype(_I32),
    )


def _find_neighbors_3d(pos: mx.array, alive: mx.array, cfg: Config) -> Neighbors:
    """The §4 search in a 3D world (§27.2): ``pos`` (N,3), a G²·G_z hash grid and a 3×3×3 stencil (Q = 27C).

    x and y are periodic as in 2D. z is bounded: the hash layer ``cz = floor(z / cell_hash_z)`` is clamped to
    ``[0, G_z − 1]`` and never wrapped, and a stencil layer outside the grid has an empty run (a clamped layer
    would otherwise be visited twice), so ``G_z`` may be 1 or 2. Distances are summed per axis, never as an
    (N,Q,3) array; only the K winners' offsets are gathered. z needs no minimum image: ``2·height ≤ L``
    (§27.1) makes ``ops.min_image`` the identity on every in-range ``dz``. Otherwise as ``find_neighbors``.
    """
    d = dims(cfg)
    n, g, gz, c, k, q = d.n, d.g_hash, d.g_hash_z, d.c, d.k, d.q
    L, R = cfg.world.size, cfg.sensing.radius
    n_hash = g * g * gz

    cxy = mx.floor(pos[:, :2] / d.cell_hash).astype(_I32) % g  # (N,2); % G maps pos ≈ L to cell 0
    cz = mx.clip(mx.floor(pos[:, 2] / d.cell_hash_z).astype(_I32), 0, gz - 1)  # clamped, never wrapped
    cid = mx.where(alive, (cz * g + cxy[:, 1]) * g + cxy[:, 0], n_hash)  # z-major; dead → sentinel
    order = mx.argsort(cid).astype(_I32)  # (N,)
    start = mx.searchsorted(cid[order], mx.arange(n_hash + 1, dtype=_I32), side="left").astype(_I32)
    count = start[1:] - start[:-1]  # (G²·G_z,) occupancy of real cells
    spos = pos[order]  # cell-sorted copy: each cell's run is contiguous
    inv = ops.scatter_add_i32(n, order, mx.arange(n, dtype=_I32))  # inv[i] = sorted rank of row i

    ncx = (cxy[:, None, 0] + mx.array(_DX27, dtype=_I32)) % g  # (N,27)
    ncy = (cxy[:, None, 1] + mx.array(_DY27, dtype=_I32)) % g
    ncz = cz[:, None] + mx.array(_DZ27, dtype=_I32)
    zin = (ncz >= 0) & (ncz < gz)  # no layer below the floor or above the surface
    ncid = (mx.clip(ncz, 0, gz - 1) * g + ncy) * g + ncx
    s = start[ncid]
    e = mx.where(zin, start[ncid + 1], s)  # (N,27) run [s, e); empty for an out-of-range layer
    slot3 = s[..., None] + mx.arange(c, dtype=_I32)  # (N,27,C) sorted positions
    ok = (slot3 < e[..., None]).reshape(n, q)
    slot = slot3.reshape(n, q)
    ok = ok & (slot != inv[:, None]) & alive[:, None]
    sl = mx.minimum(slot, n - 1)

    dx = ops.min_image(spos[:, 0][sl] - pos[:, None, 0], L)  # (N,Q) per axis, neighbour − self
    dy = ops.min_image(spos[:, 1][sl] - pos[:, None, 1], L)
    dz = spos[:, 2][sl] - pos[:, None, 2]
    d2 = dx * dx + dy * dy + dz * dz
    d2 = mx.where(ok & (d2 < R * R), d2, mx.inf)
    part = mx.argpartition(d2, kth=k - 1, axis=1)[:, :k].astype(_I32)  # K smallest, unordered
    pd2 = mx.take_along_axis(d2, part, 1)
    o = mx.argsort(pd2, axis=1).astype(_I32)  # nearest first
    part = mx.take_along_axis(part, o, 1)
    pd2 = mx.take_along_axis(pd2, o, 1)
    valid = pd2 < mx.inf

    self_ = mx.arange(n, dtype=_I32)[:, None]
    idx = mx.where(valid, order[mx.take_along_axis(sl, part, 1)], self_)  # ids resolved only for K winners
    offk = mx.stack([mx.take_along_axis(a, part, 1) for a in (dx, dy, dz)], axis=-1)  # (N,K,3)
    offk = mx.where(valid[..., None], offk, 0.0)
    dist = mx.where(valid, mx.sqrt(pd2), R)
    return Neighbors(
        idx=idx,
        valid=valid,
        off=offk,
        dist=dist,
        overflow_cells=(count > c).sum().astype(_I32),
        max_occupancy=count.max().astype(_I32),
    )
