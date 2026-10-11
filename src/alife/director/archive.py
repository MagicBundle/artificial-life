"""The behaviour archive: novelty search and a coverage grid over descriptors in [0, 1] (DIRECTOR.md §4).

A descriptor ``b`` is one vector per rule window (``vitals.descriptor``). The archive keeps the descriptors
that were novel when they arrived, scores every new one by its mean distance to its nearest archive members,
and records which cells of a coarse grid over three descriptor axes the run has ever visited ("coverage").
It is a deterministic fold of the descriptors it is fed, so a resumed run rebuilds it exactly from the
on-disk metrics rows (A7); ``director_archive.json`` is written for analysis and never read back.
"""

from __future__ import annotations

import math
from typing import Any, NamedTuple

import numpy as np


class Observation(NamedTuple):
    novelty: float | None  # None while the archive (minus the most recent entries) is empty
    added: bool  # the descriptor entered the archive
    new_cell: bool  # the descriptor visited a grid cell for the first time


class Archive:
    """Descriptors in ``[0, 1]^d`` (``entries``) and the grid cells ever visited (``cells``).

    - **Novelty** of ``b``: the mean distance to its ``k`` nearest archive members, excluding the most recent
      ``exclude`` entries (so a descriptor is not scored against its own near-duplicates). Distances are
      Euclidean divided by √d, so they lie in ``[0, 1]``.
    - **Membership:** ``b`` is added when its novelty exceeds ``rho``, or when there is nothing to compare to.
    - **Coverage:** the grid has ``bins`` equal bins on each of the three descriptor ``axes``; every observed
      descriptor (added or not) marks its cell.
    """

    def __init__(self, k: int, exclude: int, rho: float, bins: int, axes: tuple[int, ...]) -> None:
        self.k, self.exclude, self.rho, self.bins, self.axes = k, exclude, rho, bins, axes
        self.entries: list[tuple[int, tuple[float, ...]]] = []
        self.cells: dict[tuple[int, ...], int] = {}

    @property
    def coverage(self) -> int:
        return len(self.cells)

    def cell(self, b: tuple[float, ...]) -> tuple[int, ...]:
        return tuple(min(int(b[a] * self.bins), self.bins - 1) for a in self.axes)

    def novelty(self, b: tuple[float, ...]) -> float | None:
        pool = self.entries[: -self.exclude] if self.exclude else self.entries
        if not pool:
            return None
        points = np.asarray([e[1] for e in pool], dtype=np.float64)
        dist = np.linalg.norm(points - np.asarray(b, dtype=np.float64), axis=1) / math.sqrt(len(b))
        return float(np.sort(dist)[: self.k].mean())

    def observe(self, step: int, b: tuple[float, ...]) -> Observation:
        """Score ``b``, add it if novel, and mark its grid cell (first visit recorded at ``step``)."""
        novelty = self.novelty(b)
        added = novelty is None or novelty > self.rho
        if added:
            self.entries.append((step, b))
        cell = self.cell(b)
        new_cell = cell not in self.cells
        self.cells.setdefault(cell, step)
        return Observation(novelty, added, new_cell)

    @property
    def last_new_cell(self) -> int | None:
        """The step at which the latest cell was first visited."""
        return max(self.cells.values(), default=None)

    def new_cells_since(self, step: int) -> int:
        """Cells first visited after ``step``."""
        return sum(1 for first in self.cells.values() if first > step)

    def to_json(self, as_of_step: int) -> dict[str, Any]:
        return {
            "as_of_step": as_of_step,
            "k": self.k,
            "exclude": self.exclude,
            "rho": self.rho,
            "bins": self.bins,
            "axes": list(self.axes),
            "coverage": self.coverage,
            "cells": [[*cell, first] for cell, first in sorted(self.cells.items())],
            "entries": [[step, list(b)] for step, b in self.entries],
        }
