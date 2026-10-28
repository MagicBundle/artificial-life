"""Braids of organism groups: words, the Burau (t = -1) entropy lower bound, surrogates (DESIGN §44.4). Numpy
only.

A group of ``n`` strands with horizontal tracks ``(x, y)`` is ordered by ``x``; between consecutive frames
the old order is sorted into the new one by adjacent swaps, each the generator ``sigma_j^{+-1}`` (``+`` when
the strand that was on the left, moving right, passes at larger ``y``; the entropy does not depend on the
convention, only on its being consistent). The finite-time exponent ``h_T = (1/T) log ||B(w_m) ...
B(w_1)||_2`` of the Burau matrices at ``t = -1`` (real; Fried-Kolev, Band-Boyland) is a lower bound of the
topological entropy of long words (not the Dynnikov-coordinate growth, which can only be larger). It is
reported per step and per crossing.

The statistic is a **coordination index**: independent strands braid chaotically, so the real value is set
against surrogates that keep each strand's path statistics and break the coordination between strands. Either
sign of ``real - surrogate`` is information (``> 0`` stirring beyond independent motion, ``< 0`` ordered
collective flow); neither is "more emergent" a priori.

Surrogates (``kind``): ``"increment"`` (default) rolls each strand's *displacement increments* by an
independent offset and re-integrates from its start (no jump, so it is valid for the short sub-windows the
fast turnover of the mini world forces); ``"position"`` is the time-shift of positions of the specification
(a cyclic shift creates a jump of the track, so it is meant for long windows).

Groups must restart inside a window: the organisms of the mini world turn over with a half-life of about 1000
steps, so a group is formed from the body labels of one frame (labels are per frame and not stable) and
followed only over a short sub-window (``analyse_blocks``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from alife.analysis.emergence.frames import Frames, dequantise
from alife.analysis.emergence.lattice import min_image

BRAID_VERSION: Final = 1
T_BURAU: Final = -1.0
RESCALE_EVERY: Final = 64
RESCALE_AT: Final = 1e50
N_SURR: Final = 20
MIN_STRANDS: Final = 4
MIN_GROUP: Final = 8
MAX_STRANDS: Final = 250
MAX_GROUPS: Final = 40
EXTENT_FRAC: Final = 1.0 / 3.0
SEED: Final = 0xB8A1D

CAVEATS: Final = (
    "Braid entropy is a lower-bound estimate (Burau at t = -1), not the Dynnikov growth.",
    "It is a coordination index (real against surrogate, either sign), not a score of more braiding.",
    "Groups are bodies of one frame followed over a short sub-window (organisms turn over fast).",
)


# --- words ---


def word_from_tracks(x: np.ndarray, y: np.ndarray) -> tuple[list[tuple[int, int]], float]:
    """The braid word of tracks ``x``, ``y`` of shape ``(T, n)`` as ``[(slot j, sign +-1), ...]``.

    Also returns ``undersampled``, the share of adjacent pairs (in the order of the earlier frame) whose
    relative horizontal displacement over the frame exceeds their separation: those pairs could have crossed
    twice unseen (the frame cadence is too coarse for them). Insertion-sorts the old order into the new one;
    ties are broken by strand index.
    """
    t, n = x.shape
    order = list(np.argsort(x[0], kind="stable"))
    word: list[tuple[int, int]] = []
    under = tot = 0
    for k in range(1, t):
        rank = np.empty(n, np.int64)
        rank[np.argsort(x[k], kind="stable")] = np.arange(n)
        o = np.asarray(order)
        if n > 1:
            a, b = o[:-1], o[1:]
            sep = np.hypot(x[k - 1, b] - x[k - 1, a], y[k - 1, b] - y[k - 1, a])
            rel = np.hypot(
                (x[k, b] - x[k - 1, b]) - (x[k, a] - x[k - 1, a]),
                (y[k, b] - y[k - 1, b]) - (y[k, a] - y[k - 1, a]),
            )
            under += int((rel > sep).sum())
            tot += n - 1
        rk = [int(rank[s]) for s in order]
        cur = list(order)
        for i in range(1, n):  # insertion sort by rank: one adjacent swap per inversion
            j = i
            while j > 0 and rk[j - 1] > rk[j]:
                a, b = cur[j - 1], cur[j]
                sign = 1 if (y[k - 1, a] + y[k, a]) > (y[k - 1, b] + y[k, b]) else -1
                word.append((j - 1, sign))
                cur[j - 1], cur[j] = b, a
                rk[j - 1], rk[j] = rk[j], rk[j - 1]
                j -= 1
        order = cur
    return word, (under / tot if tot else 0.0)


def apply_generator(m: np.ndarray, i: int, sign: int, t: float = T_BURAU) -> None:
    """Left-multiply ``m`` by the Burau matrix of ``sigma_i^sign`` (rows ``i``, ``i + 1`` only, O(n))."""
    r0, r1 = m[i].copy(), m[i + 1].copy()
    if sign > 0:
        m[i] = (1 - t) * r0 + t * r1
        m[i + 1] = r0
    else:
        m[i] = r1
        m[i + 1] = r0 / t + ((t - 1) / t) * r1


def log_growth(word: list[tuple[int, int]], n: int) -> float:
    """``log ||B(w)||_2`` of the Burau product (at least 0), rescaled every 64 crossings against overflow."""
    m = np.eye(n)
    logs = 0.0
    for k, (i, s) in enumerate(word):
        apply_generator(m, i, s)
        if k % RESCALE_EVERY == RESCALE_EVERY - 1:
            nrm = float(np.abs(m).max())
            if nrm > RESCALE_AT:
                m /= nrm
                logs += math.log(nrm)
    nrm = float(np.linalg.norm(m, 2))
    return logs + math.log(max(nrm, 1.0))


def braid_stats(x: np.ndarray, y: np.ndarray, steps: float) -> dict[str, Any]:
    """Word, crossings and entropy lower bound of one set of tracks over ``steps`` steps."""
    word, under = word_from_tracks(x, y)
    n = x.shape[1]
    lg = log_growth(word, n) if word else 0.0
    return {
        "n_strands": n,
        "n_cross": len(word),
        "log_norm": lg,
        "h_step": lg / steps if steps > 0 else None,
        "h_cross": lg / len(word) if word else None,
        "multi_cross_frac": under,
    }


# --- surrogates ---


def surrogate(
    x: np.ndarray, y: np.ndarray, rng: np.random.Generator, kind: str = "increment"
) -> tuple[np.ndarray, np.ndarray]:
    """Tracks with each strand's path statistics kept and the coordination between strands broken."""
    t, n = x.shape
    sx, sy = np.empty_like(x), np.empty_like(y)
    if kind == "increment":
        dx, dy = np.diff(x, axis=0), np.diff(y, axis=0)
        for s in range(n):
            r = int(rng.integers(t - 1)) if t > 2 else 0
            sx[:, s] = x[0, s] + np.concatenate([[0.0], np.cumsum(np.roll(dx[:, s], r))])
            sy[:, s] = y[0, s] + np.concatenate([[0.0], np.cumsum(np.roll(dy[:, s], r))])
    elif kind == "position":
        for s in range(n):
            r = int(rng.integers(t))
            sx[:, s], sy[:, s] = np.roll(x[:, s], r), np.roll(y[:, s], r)
    else:
        raise ValueError(f"unknown surrogate kind {kind!r}")
    return sx, sy


def coordination(
    x: np.ndarray,
    y: np.ndarray,
    steps: float,
    rng: np.random.Generator,
    *,
    n_surr: int = N_SURR,
    kind: str = "increment",
) -> dict[str, Any]:
    """Real braid statistics against ``n_surr`` surrogates: ``coord = h_real / mean(h_surr) - 1`` and the
    contrasts ``h_real - mean(h_surr)`` per step and per crossing (the P5 ingredient)."""
    real = braid_stats(x, y, steps)
    sur = [braid_stats(*surrogate(x, y, rng, kind), steps) for _ in range(n_surr)]
    hs = np.array([s["h_step"] for s in sur], np.float64)
    pc = np.array([s["h_cross"] if s["h_cross"] is not None else np.nan for s in sur], np.float64)
    ms = float(hs.mean())
    mpc = float(np.nanmean(pc)) if np.isfinite(pc).any() else None
    return {
        **real,
        "h_surr": ms,
        "h_surr_sd": float(hs.std(ddof=1)) if n_surr > 1 else None,
        "h_cross_surr": mpc,
        "cross_surr": float(np.mean([s["n_cross"] for s in sur])),
        "coord": real["h_step"] / ms - 1.0 if ms > 0 and real["h_step"] is not None else None,
        "contrast_step": real["h_step"] - ms if real["h_step"] is not None else None,
        "contrast_cross": real["h_cross"] - mpc if real["h_cross"] is not None and mpc is not None else None,
        "kind": kind,
    }


# --- groups from frames ---


@dataclass
class Group:
    uids: np.ndarray
    reason: str | None = None  # why the group was skipped


def groups_at(
    fr: Frames, i: int, *, min_size: int = MIN_GROUP, max_size: int = MAX_STRANDS
) -> tuple[list[np.ndarray], dict[str, int]]:
    """The bodies of frame ``i`` with ``min_size <= members <= max_size`` as uid arrays (body labels are
    per frame and are used at this frame only), and the counts of bodies skipped as too small or too large."""
    s = fr.span(i)
    body, uid = fr.body[s], fr.uid[s]
    ok = body >= 0
    out: list[np.ndarray] = []
    skipped = {"small": 0, "large": 0}
    if ok.any():
        lab, inv, cnt = np.unique(body[ok], return_inverse=True, return_counts=True)
        order = np.argsort(inv, kind="stable")
        bounds = np.concatenate([[0], np.cumsum(cnt)])
        u = uid[ok][order]
        for j in range(len(lab)):
            if cnt[j] < min_size:
                skipped["small"] += 1
            elif cnt[j] > max_size:
                skipped["large"] += 1
            else:
                out.append(np.sort(u[bounds[j] : bounds[j + 1]]))
    return out, skipped


def tracks_of(fr: Frames, uids: np.ndarray, i0: int, i1: int) -> tuple[np.ndarray, np.ndarray] | str:
    """Horizontal tracks ``(x, y)`` of shape ``(i1 - i0 + 1, n)`` of the group members alive at every frame
    of ``[i0, i1]``, unwrapped about the first frame's centroid; or a string naming why the group is skipped
    (fewer than ``MIN_STRANDS`` survivors, or an extent above ``L / 3`` where the local frame is invalid)."""
    size = fr.size
    keep = np.asarray(uids)
    cols = []
    for i in range(i0, i1 + 1):
        s = fr.span(i)
        u = fr.uid[s]
        o = np.argsort(u)
        idx = np.searchsorted(u[o], keep)
        idx = np.clip(idx, 0, len(u) - 1)
        found = u[o][idx] == keep
        keep, idx = keep[found], idx[found]
        cols = [c[found] for c in cols]
        rows = s.start + o[idx]
        cols.append(rows)
        if len(keep) < MIN_STRANDS:
            return "survivors"
    pos = np.stack([dequantise(fr.pos[r], size, fr.height)[:, :2] for r in cols])  # (T, n, 2)
    d = min_image(np.diff(pos, axis=0), size)
    first = pos[0] - pos[0].mean(0)
    first = min_image(first, size)
    rel = np.concatenate([first[None], first[None] + np.cumsum(d, axis=0)])
    ext = float((rel.max(1) - rel.min(1)).max())
    if ext > size * EXTENT_FRAC:
        return "extent"
    return rel[..., 0], rel[..., 1]


def analyse_blocks(
    fr: Frames,
    *,
    n_blocks: int = 8,
    n_surr: int = N_SURR,
    kind: str = "increment",
    max_groups: int = MAX_GROUPS,
    seed: int = SEED,
) -> dict[str, Any]:
    """Block C of one frames file. The window is cut into ``n_blocks`` sub-windows of consecutive frames; at
    the first frame of each, the bodies of 8 to 250 members form the groups, followed over the sub-window on
    the members that survive it. ``blocks`` are the P5 block values: the mean over groups of the per-crossing
    contrast ``h_real - h_surr`` (``None`` for a block with no usable group)."""
    f = fr.n_frames
    w = (f - 1) // n_blocks
    if w < 2:
        raise ValueError(f"{f} frames are too few for {n_blocks} braid blocks")
    steps = float(fr.step[w] - fr.step[0])
    blocks: list[float | None] = []
    per_block = []
    skipped = {"small": 0, "large": 0, "survivors": 0, "extent": 0}
    for b in range(n_blocks):
        i0, i1 = b * w, b * w + w
        groups, sk = groups_at(fr, i0)
        skipped["small"] += sk["small"]
        skipped["large"] += sk["large"]
        # a deterministic subset: the groups with the lowest hash of their smallest uid
        groups.sort(key=lambda g: (int(g[0]) * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF)
        done = []
        for gi, g in enumerate(groups[:max_groups]):
            tr = tracks_of(fr, g, i0, i1)
            if isinstance(tr, str):
                skipped[tr] += 1
                continue
            rng = np.random.default_rng(np.random.SeedSequence([seed, b, gi]))
            r = coordination(tr[0], tr[1], steps, rng, n_surr=n_surr, kind=kind)
            r["size0"] = len(g)
            done.append(r)
        cc = [r["contrast_cross"] for r in done if r["contrast_cross"] is not None]
        blocks.append(float(np.mean(cc)) if cc else None)
        per_block.append(
            {
                "block": b,
                "groups": len(done),
                "contrast_cross": blocks[-1],
                "contrast_step": _mean([r["contrast_step"] for r in done]),
                "h_real": _mean([r["h_step"] for r in done]),
                "h_surr": _mean([r["h_surr"] for r in done]),
                "coord": _mean([r["coord"] for r in done]),
                "n_cross": _mean([r["n_cross"] for r in done]),
                "multi_cross_frac": _mean([r["multi_cross_frac"] for r in done]),
                "strands": _mean([r["n_strands"] for r in done]),
            }
        )
    return {
        "version": BRAID_VERSION,
        "sub_window_frames": w,
        "sub_window_steps": steps,
        "surrogate": kind,
        "n_surr": n_surr,
        "blocks": blocks,
        "per_block": per_block,
        "skipped": skipped,
        "groups_total": int(sum(p["groups"] for p in per_block)),
        "caveats": list(CAVEATS),
    }


def _mean(v: list[Any]) -> float | None:
    v = [x for x in v if x is not None and math.isfinite(x)]
    return float(np.mean(v)) if v else None
