"""Fixed-K neighbour search on a periodic hash grid in pure MLX (§4; 3D: §27.2), with a fused Metal kernel for
the scan and selection on a Metal GPU (``knn_metal``, §4 "Fused Metal path"). Owned by WP1."""

from __future__ import annotations

import os
from typing import NamedTuple

import mlx.core as mx

from alife import knn_metal, ops
from alife.config import Config, dims

_I32 = mx.int32
# 3×3 stencil offsets, row-major (dy outer, dx inner).
_DX9 = (-1, 0, 1, -1, 0, 1, -1, 0, 1)
_DY9 = (-1, -1, -1, 0, 0, 0, 1, 1, 1)
# 3×3×3 stencil offsets (3D, §27.2): dz outermost, then the 3×3 block above.
_DX27 = _DX9 * 3
_DY27 = _DY9 * 3
_DZ27 = (-1,) * 9 + (0,) * 9 + (1,) * 9

_KNN_IMPLS = ("auto", "mlx", "metal")


class Neighbors(NamedTuple):
    idx: mx.array  # (N,K) int32; invalid entries hold the agent's own index
    valid: mx.array  # (N,K) bool
    off: mx.array  # (N,K,nd) f32 minimum-image vector me → neighbour; 0 where invalid
    dist: mx.array  # (N,K) f32; R where invalid
    overflow_cells: mx.array  # () int32: hash cells with count > C
    max_occupancy: mx.array  # () int32: largest hash-cell count


def _on_gpu() -> bool:
    """The default device is the Metal GPU (a trace-time constant: it reads no array)."""
    return mx.default_device().type == mx.gpu


def knn_impl(cfg: Config, impl: str | None = None) -> str:
    """``"metal"`` or ``"mlx"``: the search ``find_neighbors`` runs for ``cfg`` (§4 "Fused Metal path").

    Decided on the host when the step is traced, from the environment, MLX's capability queries, the
    default device and a config constant (``sensing.k_neighbors``; ``world.dims`` no longer matters), so it
    is a trace-time constant like the ``d3`` branches: nothing retraces on it and a compiled step keeps the
    path it was traced with. ``impl`` (tests, bench) overrides the ``ALIFE_KNN`` environment variable,
    which defaults to ``"auto"``.

    - ``"mlx"``: the pure-MLX search, the reference (and the only one on the CPU).
    - ``"auto"``: the kernel, in 2D and in 3D, on Metal with the GPU as default device, a verified MLX,
      a passing self-check and ``K ≤ knn_metal.K_MAX``; the pure-MLX search otherwise. (2D joined 3D as
      a default by the owner's decision of 2026-10-21, I52; the results are bitwise identical.)
    - ``"metal"``: the kernel, or ``RuntimeError`` where it cannot run: no Metal, no
      ``mx.fast.metal_kernel``, an MLX not in ``knn_metal.VERIFIED_MLX``, a failed ``self_check``, a
      CPU default device or ``K > K_MAX``.
    """
    want = impl if impl is not None else os.environ.get("ALIFE_KNN", "auto")
    if want not in _KNN_IMPLS:
        raise ValueError(f"ALIFE_KNN / impl must be one of {_KNN_IMPLS}, got {want!r}")
    if want == "mlx":
        return "mlx"
    d = dims(cfg)
    ok = knn_metal.available() and _on_gpu() and d.k <= knn_metal.K_MAX
    if want == "metal" and not ok:
        raise RuntimeError(
            "the Metal neighbour search needs Metal, mx.fast.metal_kernel, a verified MLX "
            f"(one of {sorted(knn_metal.VERIFIED_MLX)}, got {mx.__version__}), a passing self-check, "
            f"the GPU as default device and sensing.k_neighbors <= {knn_metal.K_MAX} (got {d.k})"
        )
    return "metal" if ok else "mlx"


def prepare_knn(cfg: Config) -> str:
    """The search ``cfg`` will use, after ``knn_metal.self_check`` has run once for its dimension (host side).

    Called by the CLI entry points before ``make_step`` (and by any library caller that wants the self-check;
    one that skips it gets only the version guard): when the kernel would be selected, it is first run against
    the pure-MLX search on a small scene with exact ties; on a mismatch it warns and the kernel is off for the
    process (``ALIFE_KNN=metal`` then raises, as it does wherever the kernel cannot run). Where the kernel
    would have been chosen but the MLX version is not in ``knn_metal.VERIFIED_MLX``, it warns once (the MLX
    search is about 5 times slower in 3D and 1.5 times in 2D). Returns ``knn_impl``.
    """
    d = dims(cfg)
    if knn_impl(cfg) == "metal":
        knn_metal.self_check(d.nd)
    elif d.k <= knn_metal.K_MAX and _on_gpu() and os.environ.get("ALIFE_KNN", "auto") == "auto":
        knn_metal.note_unverified()  # the kernel is what auto would have taken
    return knn_impl(cfg)


def find_neighbors(pos: mx.array, alive: mx.array, cfg: Config, *, impl: str | None = None) -> Neighbors:
    """K nearest alive agents within ``sensing.radius``, sorted by distance (§4).

    Alive agents are sorted by hash cell; each agent scans the first C sorted entries of each of its 9
    surrounding cells (Q = 9C candidates, gathered from the cell-sorted position copy) and keeps the K
    nearest strictly inside R. Equals brute-force kNN whenever no cell holds more than C agents (F13);
    otherwise only the first C agents of an overfull cell are visible, which ``overflow_cells`` reports.
    A 3D world (``pos`` (N,3)) uses the 3×3×3 search of ``_find_neighbors_3d`` (§27.2). ``impl`` overrides
    ``ALIFE_KNN`` (``knn_impl``); both searches return identical arrays on Metal, the kernel only runs faster.
    """
    d = dims(cfg)
    metal = knn_impl(cfg, impl) == "metal"  # static: a Python bool, fixed when the step is traced
    if d.d3:
        return _find_neighbors_3d(pos, alive, cfg, metal=metal)
    assert pos.shape[-1] == 2, "the 2D neighbour search got 3D positions"  # static: it would ignore z
    n, g, c, k, q = d.n, d.g_hash, d.c, d.k, d.q
    L, R = cfg.world.size, cfg.sensing.radius

    cxy = mx.floor(pos / d.cell_hash).astype(_I32) % g  # (N,2); % G maps pos ≈ L to cell 0
    cid = mx.where(alive, cxy[:, 1] * g + cxy[:, 0], g * g)  # dead → sentinel after all real cells
    order = mx.argsort(cid).astype(_I32)  # (N,)
    start = mx.searchsorted(cid[order], mx.arange(g * g + 1, dtype=_I32), side="left").astype(_I32)
    count = start[1:] - start[:-1]  # (G²,) occupancy of real cells
    spos = pos[order]  # cell-sorted copy: each cell's run is contiguous
    if metal:  # static: the scan, mask, top-K and gather as one kernel; no ``inv``, the thread knows its rank
        return _metal_tail(alive, cfg, cxy, None, order, start, count, spos)
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


def _metal_tail(
    alive: mx.array,
    cfg: Config,
    cxy: mx.array,
    cz: mx.array | None,
    order: mx.array,
    start: mx.array,
    count: mx.array,
    spos: mx.array,
) -> Neighbors:
    """Scan and selection by ``knn_metal.scan_select`` on the arrays of the shared MLX preamble (the hash, the
    sort, ``start`` and ``count`` are the very same graph nodes in both searches). ``cz`` is None in 2D."""
    d = dims(cfg)
    idx, valid, off, dist = knn_metal.scan_select(
        spos,
        start,
        order,
        cxy,
        cz if cz is not None else mx.zeros((1,), dtype=_I32),
        alive,
        nd=d.nd,
        g=d.g_hash,
        gz=d.g_hash_z,
        c=d.c,
        k=d.k,
        n=d.n,
        L=cfg.world.size,
        R=cfg.sensing.radius,
    )
    return Neighbors(
        idx=idx,
        valid=valid,
        off=off,
        dist=dist,
        overflow_cells=(count > d.c).sum().astype(_I32),
        max_occupancy=count.max().astype(_I32),
    )


def _find_neighbors_3d(pos: mx.array, alive: mx.array, cfg: Config, *, metal: bool = False) -> Neighbors:
    """The §4 search in a 3D world (§27.2): ``pos`` (N,3), a G²·G_z hash grid and a 3×3×3 stencil (Q = 27C).

    x and y are periodic as in 2D. z is bounded: the hash layer ``cz = floor(z / cell_hash_z)`` is clamped to
    ``[0, G_z − 1]`` and never wrapped, and a stencil layer outside the grid has an empty run (a clamped layer
    would otherwise be visited twice), so ``G_z`` may be 1 or 2. Distances are summed per axis, never as an
    (N,Q,3) array; only the K winners' offsets are gathered. z needs no minimum image: ``2·height ≤ L``
    (§27.1) makes ``ops.min_image`` the identity on every in-range ``dz``. Otherwise as ``find_neighbors``;
    ``metal`` (static) hands the sorted data to the fused kernel after ``spos``.
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
    if metal:  # static: one kernel for the scan, mask, top-K and gather (empty out-of-range layers included)
        return _metal_tail(alive, cfg, cxy, cz, order, start, count, spos)
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


# --- occupancy of the 3D hash grid (W1, DESIGN §30) ---------------------------------------------------------


def hash_occupancy(pos: mx.array, alive: mx.array | None, cfg: Config) -> mx.array:
    """Agents per hash cell, binned exactly as ``_find_neighbors_3d`` does: ``(G_z, G, G)`` int32. ``alive``
    keeps dead rows (their stale positions) out of the count; ``None`` counts every row. The two binning lines
    are the search's own, so the maximum equals ``Neighbors.max_occupancy``."""
    d = dims(cfg)
    g, gz = d.g_hash, d.g_hash_z
    cxy = mx.floor(pos[:, :2] / d.cell_hash).astype(_I32) % g
    cz = mx.clip(mx.floor(pos[:, 2] / d.cell_hash_z).astype(_I32), 0, gz - 1)
    cid = (cz * g + cxy[:, 1]) * g + cxy[:, 0]
    return ops.scatter_add_i32(g * g * gz, cid, 1, alive).reshape(gz, g, g)


def band_peaks(counts: mx.array) -> tuple[mx.array, mx.array]:
    """Per band (floor = hash layer 0, mid = layers 1 to G_z - 2, surface = layer G_z - 1) the largest cell
    count and the flat id of the argmax cell in the ``(G_z, G, G)`` volume. Both ``(3,)`` int32. A band that
    does not exist (``G_z`` below 3 has no mid band) reads 0 and -1."""
    gz, g = counts.shape[0], counts.shape[1]
    zero = counts[0, 0, 0] * 0
    bands: list[tuple[mx.array | None, int]] = [
        (counts[0:1], 0),
        (counts[1 : gz - 1] if gz > 2 else None, g * g),
        (counts[gz - 1 : gz], (gz - 1) * g * g),
    ]
    mxs, cells = [], []
    for sub, off in bands:
        if sub is None:
            mxs.append(zero)
            cells.append(zero - 1)
        else:
            flat = sub.reshape(-1)
            mxs.append(flat.max())
            cells.append(flat.argmax().astype(_I32) + off)
    return mx.stack(mxs).astype(_I32), mx.stack(cells).astype(_I32)
