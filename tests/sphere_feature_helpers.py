"""Helpers of the sphere feature tests (DESIGN §46.5; sphere series S3): configs, eager runs, and independent
numpy references of the built-material rule on the shell.

The references are written as explicit loops over the touching table, independent of the gather-and-mask form
of ``fields_sph`` (and ``adjacency_brute`` goes further: it rebuilds the touching relation from the corner
ids of the pixels, independent of ``geom.tables``' own touching table).
"""

from __future__ import annotations

import functools
import math
from collections.abc import Iterator
from dataclasses import replace

import mlx.core as mx
import numpy as np

from alife import geom
from alife.config import Config, load_config
from alife.state import counter_index, to_numpy
from alife.step import make_step


def cfg_of(*extra: str, compile: bool = False) -> Config:
    """tiny3s with float32 weights and ``extra`` overrides; eager unless ``compile``."""
    cfg = load_config(preset="tiny3s", overrides=["brain.weight_dtype=float32", *extra])
    return cfg if compile else replace(cfg, run=replace(cfg.run, compile=False))


def eager_steps(cfg: Config, s, steps: int) -> Iterator[tuple]:
    """``(prev, s, out)`` per eager step (evaluated)."""
    fn = make_step(cfg, compile=False)
    for _ in range(steps):
        prev = s
        s, out = fn(s)
        mx.eval(s, out)
        yield prev, s, out


def counters(cfg: Config, out, *names: str) -> dict[str, int]:
    c = to_numpy(out.counters)
    return {n: int(c[counter_index(cfg, n)]) for n in names}


# --- the shell's neighbourhood, written out ---------------------------------------------------------------


def adjacency_brute(nside: int) -> list[set[int]]:
    """Touching pixels from the corner ids alone: two pixels touch iff they share a corner (O(N^2), small
    Nside only). Independent of ``geom.tables().touch``."""
    vtx = geom.tables(nside).vertex
    sets = [set(map(int, row)) for row in vtx]
    n = len(sets)
    return [{q for q in range(n) if q != p and sets[p] & sets[q]} for p in range(n)]


@functools.cache
def neighbour_index(nside: int, layers: int) -> np.ndarray:
    """(FZ·Npix, 26) int64: for every cell (flat ``k·Npix + p``) the flat indices of its neighbour cells,
    centre excluded, padded with -1: the touching pixels of the layers ``k-1..k+1`` (those that exist) and
    the pixel itself in the layers above and below. Explicit loops; the reference of the tests."""
    touch = geom.tables(nside).touch
    npix = touch.shape[0]
    out = np.full((layers * npix, 26), -1, np.int64)
    for k in range(layers):
        for p in range(npix):
            cell = []
            for k2 in (k - 1, k, k + 1):
                if not 0 <= k2 < layers:
                    continue
                if k2 != k:
                    cell.append(k2 * npix + p)
                cell += [k2 * npix + int(q) for q in touch[p] if q >= 0]
            out[k * npix + p, : len(cell)] = cell
    return out


def ref_nopen(m: np.ndarray, thr: int, nside: int) -> np.ndarray:
    """OPEN neighbours of every cell by the explicit list (``m`` is (FZ, Npix))."""
    idx = neighbour_index(nside, m.shape[0])
    op = (m < thr).astype(np.int64).reshape(-1)
    return np.where(idx >= 0, op[np.maximum(idx, 0)], 0).sum(axis=1).reshape(m.shape)


def ref_boxmin(a: np.ndarray, nside: int) -> np.ndarray:
    """Minimum over the cell and its neighbours of every cell."""
    idx = neighbour_index(nside, a.shape[0])
    flat = a.reshape(-1)
    big = np.iinfo(np.int64).max
    return np.minimum(flat, np.where(idx >= 0, flat[np.maximum(idx, 0)], big).min(axis=1)).reshape(a.shape)


def ref_phase(layers: int, nside: int) -> tuple[np.ndarray, int]:
    colour, n_col = geom.min_colouring(nside)
    return colour[None, :].astype(np.int64) + n_col * (np.arange(layers)[:, None] % 3), min(layers, 3) * n_col


def ref_update(m: np.ndarray, dep: np.ndarray, t: int, cfg: Config):
    """The numpy form of ``fields.update_mat`` on the shell: ``(mat', wasted, decayed, nopen)``."""
    sc = cfg.structures
    nside = cfg.world.nside
    thr = math.ceil(sc.open_frac * sc.cap)
    n = ref_nopen(m, thr, nside)
    pre = m + dep
    m1 = np.minimum(pre, sc.cap)
    if sc.aperture > 0:
        n_nbr = ref_nopen(np.zeros_like(m), thr, nside)  # every cell open: the neighbour counts
        kmap = np.ceil(sc.aperture * n_nbr - 1e-9).astype(np.int64)
        slack = ref_boxmin(n - kmap, nside) >= 1
        phase, nph = ref_phase(m.shape[0], nside)
        active = phase == (t % nph)
        want = (m < thr) & (m1 >= thr)
        m1 = np.where(want & ~(active & slack), thr - 1, m1)
    loss = (m1 * sc.decay_q12 + 4095) >> 12
    return m1 - loss, int((pre - m1).sum()), int(loss.sum()), n
