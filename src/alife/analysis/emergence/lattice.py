"""Lattices and organism transitions for the irreversibility block (DESIGN §44.3). Numpy only.

``Lattice`` is the periodic horizontal grid of side ``cell`` (world units). ``own_positions`` turns the frames
into per-row unwrapped **own** coordinates (``pos`` less the integrated wind drift, the construction of
``labels.track_step``: ``u_k = u_{k-1} + min_image(Δpos) - drift``; with no drift they are the raw track).
``extract_transitions`` pairs the organisms alive at two frames (matched by ``uid``, never by row, so a reused
row creates no transition) and returns their cell pairs.

Drift convention: the recorded ``drift`` is a velocity (units per step); between two frames ``dt`` steps apart
its integral is approximated by ``dt`` times the mean of the two frames' values (the trapezoid; the error is
second order in the frame spacing and is not the quantity under test).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from alife.analysis.emergence.frames import Frames, dequantise

LATTICE_VERSION: Final = 1
ORIGIN_GRID: Final = 8  # coarse cells per side of the origin grid that defines the organism blocks
HORIZONTAL: Final = "xy"
VERTICAL: Final = "z"


@dataclass(frozen=True)
class Lattice:
    """A periodic ``ncell x ncell`` horizontal lattice over ``[0, size)^2`` (cell side ``size / ncell``)."""

    size: float
    ncell: int

    @property
    def n_cells(self) -> int:
        return self.ncell * self.ncell

    @property
    def cell(self) -> float:
        return self.size / self.ncell

    def index(self, xy: np.ndarray) -> np.ndarray:
        """Cell code ``ix * ncell + iy`` (int32) of positions ``(n, >=2)``; the lattice wraps."""
        g = np.floor(np.mod(xy[:, :2], self.size) / self.size * self.ncell).astype(np.int64) % self.ncell
        return (g[:, 0] * self.ncell + g[:, 1]).astype(np.int32)

    def centres(self) -> np.ndarray:
        """Cell centres ``(n_cells, 2)`` in code order."""
        c = (np.arange(self.ncell) + 0.5) * self.cell
        gx, gy = np.meshgrid(c, c, indexing="ij")
        return np.stack([gx.ravel(), gy.ravel()], axis=1)

    def merge(self, factor: int) -> Lattice:
        """The lattice with ``factor`` times larger cells (``ncell`` must be divisible by ``factor``)."""
        if factor < 1 or self.ncell % factor:
            raise ValueError(f"{self.ncell} cells do not merge by {factor}")
        return Lattice(self.size, self.ncell // factor)


def lattice_for(size: float, cell: float) -> Lattice:
    """The lattice of cell side nearest ``cell`` that tiles ``size`` exactly."""
    return Lattice(float(size), max(1, round(size / cell)))


def n_bands(height: float, band: float = 8.0) -> int:
    """Vertical bands of height about ``band`` (``H / 8`` in the spec)."""
    return max(1, round(height / band))


def band_index(z: np.ndarray, height: float, bands: int) -> np.ndarray:
    """Band code (int32) of heights, clipped into ``[0, bands)`` (z does not wrap)."""
    return np.clip(np.floor(np.asarray(z) / height * bands), 0, bands - 1).astype(np.int32)


def min_image(d: np.ndarray, size: float) -> np.ndarray:
    return d - size * np.rint(d / size)


def _match(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Indices ``(ia, ib)`` of the uids present in both (uids are unique within a frame)."""
    _, ia, ib = np.intersect1d(a, b, assume_unique=True, return_indices=True)
    return ia, ib


def own_positions(fr: Frames, *, own: bool = True) -> np.ndarray:
    """Unwrapped coordinates ``(n_rows, nd)`` float64 of every row of every frame.

    An organism's track starts at its first recorded position and then advances by the minimum-image
    displacement between consecutive frames, less (``own=True``) the integrated drift. The horizontal axes are
    unwrapped (take ``mod size`` for a cell), z is not wrapped. Rows of a frame whose uid is absent from the
    previous frame (births, or entry) restart from their recorded position.
    """
    nd = fr.nd
    out = np.empty((len(fr.uid), nd), np.float64)
    s0 = fr.span(0)
    out[s0] = fr.positions(0)
    for i in range(1, fr.n_frames):
        s, p = fr.span(i), fr.span(i - 1)
        cur = fr.positions(i)
        rows = out[s]
        rows[:] = cur  # restart
        ia, ib = _match(fr.uid[p], fr.uid[s])
        if ia.size:
            prev_raw = dequantise(fr.pos[p], fr.size, fr.height)[ia]
            d = cur[ib] - prev_raw
            d[:, :2] = min_image(d[:, :2], fr.size)
            if own and fr.drift is not None:
                dt = float(fr.step[i] - fr.step[i - 1])
                dr = 0.5 * dt * (fr.drift[p][ia].astype(np.float64) + fr.drift[s][ib].astype(np.float64))
                d = d - dr
            rows[ib] = out[p][ia] + d
        out[s] = rows
    return out


@dataclass
class Transitions:
    """The cell-to-cell moves of organisms alive at two frames ``dt`` steps apart.

    ``ci``, ``cj`` are cell codes at the earlier and later frame; ``frame`` the earlier frame's index;
    ``uid``, ``clade`` and ``body_size`` (members of the organism's body at the earlier frame, 0 if unbonded)
    identify subsets; ``move`` the horizontal displacement (own coordinates, world units) per transition.
    """

    n_cells: int
    dt: int
    n_pairs: int  # frame pairs used
    skipped: int  # frame pairs skipped (uneven spacing)
    ci: np.ndarray
    cj: np.ndarray
    frame: np.ndarray
    uid: np.ndarray
    clade: np.ndarray
    body_size: np.ndarray
    move: np.ndarray
    origin: np.ndarray | None = (
        None  # coarse cell (ORIGIN_GRID^2) where the organism first appears: the block unit
    )

    def __len__(self) -> int:
        return len(self.ci)

    def select(self, mask: np.ndarray) -> Transitions:
        return Transitions(
            self.n_cells,
            self.dt,
            self.n_pairs,
            self.skipped,
            self.ci[mask],
            self.cj[mask],
            self.frame[mask],
            self.uid[mask],
            self.clade[mask],
            self.body_size[mask],
            self.move[mask],
            None if self.origin is None else self.origin[mask],
        )

    def reversed(self) -> Transitions:
        """The time-reversed transitions (``ci`` and ``cj`` swapped)."""
        t = self.select(np.ones(len(self), bool))
        t.ci, t.cj = self.cj.copy(), self.ci.copy()
        return t


def _body_sizes(body: np.ndarray) -> np.ndarray:
    out = np.zeros(len(body), np.int32)
    m = body >= 0
    if m.any():
        _, inv, cnt = np.unique(body[m], return_inverse=True, return_counts=True)
        out[m] = cnt[inv]
    return out


def extract_transitions(
    fr: Frames,
    *,
    cell: float = 8.0,
    own: bool = True,
    stride: int = 1,
    axis: str = HORIZONTAL,
    band: float = 8.0,
    tracks: np.ndarray | None = None,
) -> Transitions:
    """Transitions between frames ``i`` and ``i + stride`` for the organisms alive at both.

    ``axis`` ``"xy"`` uses the periodic horizontal lattice of side ``cell``; ``"z"`` uses vertical bands of
    height about ``band`` in the own z (secondary number; the ambient drift and gravity make it large).
    ``stride`` k pairs frames k apart (tau = k times the recorded tau); pairs whose step difference is not k
    times the median spacing are skipped and counted. ``tracks`` lets a caller pass the ``own_positions`` once
    for several lattices.
    """
    if fr.n_frames < stride + 1:
        raise ValueError("fewer frames than stride + 1")
    if axis == VERTICAL and fr.nd < 3:
        raise ValueError("vertical bands need a 3D file")
    u = own_positions(fr, own=own) if tracks is None else tracks
    spacing = float(np.median(np.diff(fr.step)))
    lat = lattice_for(fr.size, cell)
    if axis == HORIZONTAL:
        codes = lat.index(u)
        n_cells = lat.n_cells
    else:
        bands = n_bands(fr.height, band)
        codes = band_index(u[:, 2], fr.height, bands)
        n_cells = bands
    first_uid, first_row = np.unique(fr.uid, return_index=True)
    og = Lattice(fr.size, ORIGIN_GRID)
    origin = og.index(dequantise(fr.pos[first_row], fr.size, fr.height))
    bs = np.concatenate([_body_sizes(fr.body[fr.span(i)]) for i in range(fr.n_frames)])
    parts: dict[str, list[np.ndarray]] = {
        k: [] for k in ("ci", "cj", "frame", "uid", "clade", "bs", "move", "origin")
    }
    used = skipped = 0
    for i in range(fr.n_frames - stride):
        j = i + stride
        if int(fr.step[j] - fr.step[i]) != round(spacing * stride):
            skipped += 1
            continue
        a, b = fr.span(i), fr.span(j)
        ia, ib = _match(fr.uid[a], fr.uid[b])
        used += 1
        if not ia.size:
            continue
        ra, rb = a.start + ia, b.start + ib
        parts["ci"].append(codes[ra])
        parts["cj"].append(codes[rb])
        parts["frame"].append(np.full(ia.size, i, np.int32))
        parts["uid"].append(fr.uid[ra])
        parts["clade"].append(fr.clade[ra])
        parts["bs"].append(bs[ra])
        parts["origin"].append(origin[np.searchsorted(first_uid, fr.uid[ra])])
        parts["move"].append(np.linalg.norm(u[rb, :2] - u[ra, :2], axis=1).astype(np.float32))

    def cat(k: str, dt: type) -> np.ndarray:
        return np.concatenate(parts[k]) if parts[k] else np.zeros(0, dt)

    return Transitions(
        n_cells,
        round(spacing * stride),
        used,
        skipped,
        cat("ci", np.int32),
        cat("cj", np.int32),
        cat("frame", np.int32),
        cat("uid", np.int64),
        cat("clade", np.int32),
        cat("bs", np.int32),
        cat("move", np.float32),
        cat("origin", np.int32),
    )


def coarsen(t: Transitions, factor: int, ncell: int) -> Transitions:
    """The same transitions on a lattice ``factor`` times coarser (codes ``ix * ncell + iy`` of an
    ``ncell x ncell`` lattice merged by blocks of ``factor``): the coarse-graining of the scale curve."""
    m = ncell // factor

    def f(c: np.ndarray) -> np.ndarray:
        return ((c // ncell) // factor * m + (c % ncell) // factor).astype(np.int32)

    out = t.select(np.ones(len(t), bool))
    out.ci, out.cj, out.n_cells = f(t.ci), f(t.cj), m * m
    return out


def advection_slope(fr: Frames) -> float | None:
    """Slope of the horizontal displacement of organisms between frames on the integrated recorded drift
    (``dt x`` the mean of the two frames' drift): about 1 for a population the wind advects, about 0 when the
    drift is recorded but not applied to the motion (subtracting it would then create a spurious own-motion
    current). None without drift, or when the drift is zero."""
    if fr.drift is None:
        return None
    num = den = 0.0
    for i in range(fr.n_frames - 1):
        a, b = fr.span(i), fr.span(i + 1)
        ia, ib = _match(fr.uid[a], fr.uid[b])
        if not ia.size:
            continue
        d = (
            dequantise(fr.pos[b], fr.size, fr.height)[ib, :2]
            - dequantise(fr.pos[a], fr.size, fr.height)[ia, :2]
        )
        d = min_image(d, fr.size)
        dr = (
            0.5
            * float(fr.step[i + 1] - fr.step[i])
            * (fr.drift[a][ia, :2].astype(np.float64) + fr.drift[b][ib, :2].astype(np.float64))
        )
        num += float((d * dr).sum())
        den += float((dr * dr).sum())
    return num / den if den > 0 else None
