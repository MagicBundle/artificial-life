"""Fixed-K neighbour search on a periodic hash grid in pure MLX (§4). Owned by WP1."""

from __future__ import annotations

from typing import NamedTuple

import mlx.core as mx

from alife import ops
from alife.config import Config, dims

_I32 = mx.int32
# 3×3 stencil offsets, row-major (dy outer, dx inner).
_DX9 = (-1, 0, 1, -1, 0, 1, -1, 0, 1)
_DY9 = (-1, -1, -1, 0, 0, 0, 1, 1, 1)


class Neighbors(NamedTuple):
    idx: mx.array  # (N,K) int32; invalid entries hold the agent's own index
    valid: mx.array  # (N,K) bool
    off: mx.array  # (N,K,2) f32 minimum-image vector me → neighbour; 0 where invalid
    dist: mx.array  # (N,K) f32; R where invalid
    overflow_cells: mx.array  # () int32: hash cells with count > C
    max_occupancy: mx.array  # () int32: largest hash-cell count


def find_neighbors(pos: mx.array, alive: mx.array, cfg: Config) -> Neighbors:
    """K nearest alive agents within ``sensing.radius``, sorted by distance (§4).

    Alive agents are sorted by hash cell; each agent scans the first C sorted entries of each of its 9
    surrounding cells (Q = 9C candidates, gathered from the cell-sorted position copy) and keeps the K
    nearest strictly inside R. Equals brute-force kNN whenever no cell holds more than C agents (F13);
    otherwise only the first C agents of an overfull cell are visible, which ``overflow_cells`` reports.
    """
    d = dims(cfg)
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
