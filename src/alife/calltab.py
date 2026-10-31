"""Host table of the column voice grid (numpy, float64 cast to float32; DESIGN §47.7, calls step K2).

Far hearing deposits the calls of every row into the angular **columns** of the field lattice (the HEALPix
pixels of a sphere, the ``FG x FG`` columns of a box; the vertical position is dropped) and reads, for every
column ``p``, the columns ``p'`` whose centre lies at a distance ``d`` in ``[hear_range, far_range]`` (the
chord on the floor sphere; the torus minimum image in a box), weighted by

    w(d) = clip((d - hear_range) / cell, 0, 1) * (1 - d / far_range) ** atten_pow

The ramp over one column side ``cell`` makes the far weight 0 at ``d = hear_range``, where the near weight of
``comm.hear`` is also 0, so the heard loudness is continuous at the near range. Per column the entries are
sorted by ``p'`` and padded to the row maximum ``S`` with weight 0 (index 0, offset 0); the sum over ``S`` in
the step is therefore in a fixed order and a padded entry adds exactly 0. ``off[p, s]`` is the centre offset
``centre(p') - centre(p)`` divided by ``far_range`` (3 components on a sphere, in the planet frame at the
column centre; 2 in a box, the minimum image), so ``|off| <= 1``.

Pure host code (it runs at trace time and in ``config.validate``); the device arrays are cached by
``alife.calls``. Cached per ``(kind, n, scale, hear_range, far_range, atten_pow)``.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass

import numpy as np

from alife import geom

S_MAX = 512  # most entries per column (validation)
TABLE_MAX_BYTES = 64 << 20  # idx + w + off, all columns (validation)
CELLS_MIN = 4.0  # far_range >= CELLS_MIN column sides (the sparse-caller accuracy tables, DESIGN §47.7)
_CHUNK = 256  # rows of the pairwise distance block (a sphere of 19 200 columns stays under 120 MB)


@dataclass(frozen=True)
class CallTab:
    kind: str  # "sphere" or "box"
    ncol: int  # number of columns
    s: int  # entries per column (the row maximum, >= 1)
    cell: float  # the column side
    idx: np.ndarray  # (ncol, s) int32, ascending in the real entries, 0 in the padding
    w: np.ndarray  # (ncol, s) float32, 0 in the padding
    off: np.ndarray  # (ncol, s, nt) float32, nt = 3 (sphere) or 2 (box)
    count: np.ndarray  # (ncol,) int32: the real entries of each column (``w`` can be 0 at the two edges)

    @property
    def nbytes(self) -> int:
        return int(self.idx.nbytes + self.w.nbytes + self.off.nbytes)


def column_side(kind: str, n: int, scale: float) -> float:
    """The side of one column: ``sqrt`` of the pixel area on a sphere of radius ``scale``, the lattice cell
    ``scale / n`` of a box of side ``scale``."""
    if kind == "sphere":
        return scale * math.sqrt(math.pi / 3.0) / n  # sqrt(4 pi R^2 / (12 n^2))
    return scale / n


def ncol_of(kind: str, n: int) -> int:
    return geom.npix(n) if kind == "sphere" else n * n


def weight(d: np.ndarray, hear_range: float, far_range: float, cell: float, atten_pow: float) -> np.ndarray:
    """The far weight of a centre distance ``d`` (float64): 0 outside ``[hear_range, far_range]``."""
    ramp = np.clip((d - hear_range) / cell, 0.0, 1.0)
    fall = np.clip(1.0 - d / far_range, 0.0, 1.0)
    w = ramp * fall**atten_pow
    return np.where((d >= hear_range) & (d <= far_range), w, 0.0)


def _distances(kind: str, n: int, scale: float, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(d, off)`` of the columns ``rows`` against every column: the centre distance ``(len(rows), ncol)``
    and the centre offset ``(len(rows), ncol, nt)`` (float64, in length units)."""
    if kind == "sphere":
        c = geom.pixel_centres(n)
        diff = (c[None, :, :] - c[rows][:, None, :]) * scale
        return np.sqrt((diff * diff).sum(axis=-1)), diff
    fg = n
    cell = scale / fg
    cx, cy = np.arange(fg * fg) % fg, np.arange(fg * fg) // fg
    dx = (cx[None, :] - cx[rows][:, None]) * cell
    dy = (cy[None, :] - cy[rows][:, None]) * cell
    size = scale
    dx = dx - size * np.rint(dx / size)  # the minimum image, half to even as ``ops.min_image``
    dy = dy - size * np.rint(dy / size)
    return np.sqrt(dx * dx + dy * dy), np.stack([dx, dy], axis=-1)


def probe(kind: str, n: int, scale: float, hear_range: float, far_range: float) -> tuple[int, int]:
    """``(s, nbytes)`` lower bounds of the table of ``build`` from eight columns, so that validation can
    refuse an oversized table before spending its memory (``build`` holds float64 members of every column;
    256 x 256 columns at range 24 peak at 24 GB)."""
    rows = np.unique(np.linspace(0, ncol_of(kind, n) - 1, 8).astype(np.int64))
    d, _ = _distances(kind, n, scale, rows)
    s = max(int(((d >= hear_range) & (d <= far_range)).sum(axis=1).max()), 1)
    nt = 3 if kind == "sphere" else 2
    return s, ncol_of(kind, n) * s * (4 + 4 + 4 * nt)


@functools.cache
def build(kind: str, n: int, scale: float, hear_range: float, far_range: float, atten_pow: float) -> CallTab:
    """The table of a geometry (``kind`` "sphere": Nside ``n`` and radius ``scale``; "box": ``n`` x ``n``
    columns of a torus of side ``scale``) for the near range ``hear_range`` and the far range ``far_range``.
    All arrays are read-only."""
    if kind not in ("sphere", "box"):
        raise ValueError(f"unknown column geometry {kind!r}")
    if not 0.0 < hear_range < far_range:
        raise ValueError(f"need 0 < hear_range < far_range, got {hear_range}, {far_range}")
    ncol = ncol_of(kind, n)
    cell = column_side(kind, n, scale)
    nt = 3 if kind == "sphere" else 2
    members: list[np.ndarray] = []
    for lo in range(0, ncol, _CHUNK):
        rows = np.arange(lo, min(lo + _CHUNK, ncol))
        d, off = _distances(kind, n, scale, rows)
        keep = (d >= hear_range) & (d <= far_range)
        wgt = weight(d, hear_range, far_range, cell, atten_pow)
        for r in range(len(rows)):
            j = np.flatnonzero(keep[r])  # ascending
            members.append(np.concatenate([j[:, None], wgt[r, j][:, None], off[r, j] / far_range], axis=1))
    count = np.array([len(m) for m in members], dtype=np.int32)
    s = max(int(count.max()), 1)
    idx = np.zeros((ncol, s), np.int32)
    w = np.zeros((ncol, s), np.float32)
    off = np.zeros((ncol, s, nt), np.float32)
    for p, m in enumerate(members):
        k = len(m)
        idx[p, :k] = m[:, 0].astype(np.int32)
        w[p, :k] = m[:, 1].astype(np.float32)
        off[p, :k] = m[:, 2:].astype(np.float32)
    for a in (idx, w, off, count):
        a.flags.writeable = False
    return CallTab(kind, ncol, s, cell, idx, w, off, count)


def geometry_key(cfg) -> tuple[str, int, float]:
    """``(kind, n, scale)`` of a config's columns from the world keys alone (safe before ``dims``)."""
    w = cfg.world
    if w.geometry == "sphere":
        return "sphere", int(w.nside), float(w.radius)
    return "box", round(w.size / w.field_cell), float(w.size)


def table_for(cfg) -> CallTab:
    """The table of a config with ``calls.far_range > 0`` (the near range is the sensing radius: far hearing
    needs ``comm.range == 0``)."""
    kind, n, scale = geometry_key(cfg)
    return build(
        kind, n, scale, float(cfg.sensing.radius), float(cfg.calls.far_range), float(cfg.comm.atten_pow)
    )
