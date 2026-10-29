"""Storm and organism-storm winding numbers (DESIGN §44.4). Numpy only; exploratory, descriptive.

Input is a set of storm tracks: ``StormTrack(id, t, x, y, r)`` with the tick steps, the (wrapped) centroid
and the equivalent radius. ``read_storms_jsonl`` builds them from ``observers/storms.jsonl`` (the tick lines
of the storm layer). **Source decision (O6).** A plain ``alife run`` writes no ``storms.jsonl`` (only a
played weather world does); the arm runs are plain runs, so the tracks of an arm come from its
``run.wx_every`` weather dumps replayed through the storm tracker (an MLX step, to be added to the recorder
at the calibration stage, E7). Until then this module takes tracks as data from either source, and every
quantity is exploratory: a handful of storms per run cannot support an inference.

Quantities: the signed zonal circuits of a track (``trunc(total unwrapped dx / L)``; truncation rather than
the floor so the count is odd-symmetric), the pair winding ``W_ab = (1/2 pi) sum Delta arg(r_a - r_b)`` over
consecutive ticks where both exist and the minimum-image separation is below ``L/2``, the Burau entropy of the
storm braid over intervals with at least three coexisting storms (descriptive), and the winding of organisms
about a storm (those within ``1.5 r_eq`` at two consecutive frames).
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

from alife.analysis.emergence.braid import braid_stats
from alife.analysis.emergence.frames import Frames
from alife.analysis.emergence.lattice import min_image

WINDING_VERSION: Final = 1
NEAR: Final = 1.5  # organism within NEAR * r_eq of a track
MAX_TURN: Final = 0.9 * math.pi  # a larger angular step per frame is ambiguous: skipped and counted
ENTRAIN_AT: Final = 0.5


@dataclass(frozen=True)
class StormTrack:
    id: int
    t: np.ndarray  # tick steps, ascending
    x: np.ndarray  # wrapped centroid
    y: np.ndarray
    r: np.ndarray

    def unwrapped(self, size: float) -> tuple[np.ndarray, np.ndarray]:
        ux = self.x[0] + np.concatenate([[0.0], np.cumsum(min_image(np.diff(self.x), size))])
        uy = self.y[0] + np.concatenate([[0.0], np.cumsum(min_image(np.diff(self.y), size))])
        return ux, uy

    def at(self, step: float, size: float) -> tuple[float, float, float] | None:
        """Interpolated wrapped centroid and radius at ``step`` (None outside the track)."""
        if step < self.t[0] or step > self.t[-1]:
            return None
        ux, uy = self.unwrapped(size)
        return (
            float(np.interp(step, self.t, ux)) % size,
            float(np.interp(step, self.t, uy)) % size,
            float(np.interp(step, self.t, self.r)),
        )


def read_storms_jsonl(path: Path) -> list[StormTrack]:
    """Tracks from the tick lines ``{"step", "storms": [{"id", "x", "y", "r", ...}]}`` of ``storms.jsonl``."""
    acc: dict[int, list[tuple[int, float, float, float]]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            d = json.loads(line)
        except ValueError:
            break  # a torn last line
        for s in d.get("storms", []):
            acc.setdefault(int(s["id"]), []).append(
                (int(d["step"]), float(s["x"]), float(s["y"]), float(s["r"]))
            )
    out = []
    for i, rows in sorted(acc.items()):
        a = np.array(sorted(rows), np.float64)
        out.append(StormTrack(i, a[:, 0].astype(np.int64), a[:, 1], a[:, 2], a[:, 3]))
    return out


def zonal_circuits(tr: StormTrack, size: float) -> int:
    """Signed whole circuits of the cylinder (``trunc`` of the unwrapped x displacement over ``L``)."""
    ux, _ = tr.unwrapped(size)
    return int(np.trunc((ux[-1] - ux[0]) / size))


def _turn(d0: np.ndarray, d1: np.ndarray) -> np.ndarray:
    """Signed angle from ``d0`` to ``d1`` (rows of 2-vectors) in ``(-pi, pi]``."""
    return np.arctan2(d0[..., 0] * d1[..., 1] - d0[..., 1] * d1[..., 0], (d0 * d1).sum(-1))


def pair_winding(a: StormTrack, b: StormTrack, size: float) -> dict[str, Any]:
    """``W_ab`` in turns over the common ticks with separation below ``L/2`` (a gap or a separation of
    ``L/2`` or more breaks the chain; the pieces add). ``ticks`` counts the angular steps used."""
    ts, ia, ib = np.intersect1d(a.t, b.t, return_indices=True)
    if len(ts) < 2:
        return {"w": 0.0, "ticks": 0}
    v = min_image(np.stack([a.x[ia] - b.x[ib], a.y[ia] - b.y[ib]], 1), size)
    near = np.hypot(v[:, 0], v[:, 1]) < size / 2
    step_ok = near[:-1] & near[1:]
    # consecutive common ticks must also be consecutive in both tracks (no gap in either)
    ca = np.diff(ia) == 1
    cb = np.diff(ib) == 1
    ok = step_ok & ca & cb
    ang = _turn(v[:-1], v[1:])
    return {"w": float(ang[ok].sum() / (2 * math.pi)), "ticks": int(ok.sum())}


def storm_braid(tracks: Sequence[StormTrack], size: float, min_storms: int = 3) -> list[dict[str, Any]]:
    """The Burau lower bound over each maximal tick interval where at least ``min_storms`` storms coexist
    (descriptive: a few events per run)."""
    ticks = np.unique(np.concatenate([t.t for t in tracks])) if tracks else np.zeros(0, np.int64)
    alive = (
        np.array([[t.t[0] <= s <= t.t[-1] for t in tracks] for s in ticks])
        if len(ticks)
        else np.zeros((0, 0), bool)
    )
    good = alive.sum(1) >= min_storms if alive.size else np.zeros(0, bool)
    out = []
    i = 0
    while i < len(ticks):
        if not good[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(ticks) and good[j + 1]:
            j += 1
        members = [k for k in range(len(tracks)) if alive[i : j + 1, k].all()]
        if len(members) >= min_storms and j > i:
            ts = ticks[i : j + 1]
            xy = []
            for k in members:
                ux, uy = tracks[k].unwrapped(size)
                xy.append((np.interp(ts, tracks[k].t, ux), np.interp(ts, tracks[k].t, uy)))
            x = np.stack([p[0] for p in xy], 1)
            y = np.stack([p[1] for p in xy], 1)
            st = braid_stats(x, y, float(ts[-1] - ts[0]))
            out.append({"t0": int(ts[0]), "t1": int(ts[-1]), "storms": len(members), **st})
        i = j + 1
    return out


def organism_winding(fr: Frames, tracks: Sequence[StormTrack]) -> dict[str, Any]:
    """Winding of organisms about storms over the frames of a window.

    An organism is *near* a storm at a frame if it lies within ``NEAR * r_eq`` of the interpolated centre. For
    each (organism, storm) the angle of ``pos - centre`` is followed over consecutive frames at which it is
    near; the winding is the summed signed angle over ``2 pi``. ``entrained`` is the share of pairs (with at
    least two near frames) with ``|W| >= 0.5``; ``turns`` the summed ``|W|``; ``skipped_steps`` the angular
    steps larger than ``0.9 pi`` (ambiguous) that were dropped."""
    size = fr.size
    acc: dict[tuple[int, int], list[float]] = {}  # (uid, storm) -> [w, n]
    skipped = 0
    prev: dict[tuple[int, int], tuple[int, np.ndarray]] = {}
    for i in range(fr.n_frames):
        s = fr.span(i)
        pos = fr.positions(i)[:, :2]
        uid = fr.uid[s]
        cur: dict[tuple[int, int], tuple[int, np.ndarray]] = {}
        for tr in tracks:
            c = tr.at(float(fr.step[i]), size)
            if c is None:
                continue
            d = min_image(pos - np.array(c[:2]), size)
            near = np.hypot(d[:, 0], d[:, 1]) < NEAR * c[2]
            for u, v in zip(uid[near], d[near], strict=True):
                key = (int(u), tr.id)
                cur[key] = (i, v)
                p = prev.get(key)
                if p is not None and p[0] == i - 1:
                    ang = float(_turn(p[1][None], v[None])[0])
                    if abs(ang) > MAX_TURN:
                        skipped += 1
                    else:
                        a = acc.setdefault(key, [0.0, 0.0])
                        a[0] += ang
                        a[1] += 1
        prev = cur
    w = np.array([v[0] / (2 * math.pi) for v in acc.values()]) if acc else np.zeros(0)
    return {
        "pairs": len(w),
        "entrained_frac": float((np.abs(w) >= ENTRAIN_AT).mean()) if len(w) else None,
        "turns": float(np.abs(w).sum()),
        "turns_per_storm": float(np.abs(w).sum() / max(len({k[1] for k in acc}), 1)) if len(w) else None,
        "skipped_steps": skipped,
    }


def analyse(tracks: Sequence[StormTrack], size: float, fr: Frames | None = None) -> dict[str, Any]:
    """The winding table of a set of tracks (and, with ``fr``, of organisms about them). Descriptive."""
    circ = {tr.id: zonal_circuits(tr, size) for tr in tracks}
    pairs = []
    for i, a in enumerate(tracks):
        for b in tracks[i + 1 :]:
            w = pair_winding(a, b, size)
            if w["ticks"]:
                pairs.append({"a": a.id, "b": b.id, **w})
    return {
        "version": WINDING_VERSION,
        "status": "exploratory",
        "storms": len(tracks),
        "zonal_circuits": circ,
        "circuits_total": int(sum(circ.values())),
        "pairs": pairs,
        "co_rotating_pairs": int(sum(abs(p["w"]) >= 1.0 for p in pairs)),
        "storm_braids": storm_braid(tracks, size),
        "organisms": organism_winding(fr, tracks) if fr is not None else None,
        "note": "a handful of storms per run cannot support an inference; descriptive only",
    }
