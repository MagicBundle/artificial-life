"""Movement irreversibility: the cross-fitted entropy-production estimator (DESIGN §44.3). Numpy only.

For organisms moving between lattice cells, ``sigma`` is the entropy production of the sampled two-step
process,
estimated from the transition counts ``C_ij`` of organisms alive at two consecutive frames (births and deaths
excluded). Two estimators:

* plug-in, antisymmetric: ``sum_{i<j} (C_ij - C_ji)/N ln((C_ij + a)/(C_ji + a))`` (reported as a diagnostic);
* **cross-fitted** (primary): the organisms are split into halves A and B; the rate ratios come from one half
  and the net flux from the other, ``sigma = 1/2 [ sum d^B_ij/N_B ln((C^A_ij+a)/(C^A_ji+a)) + (A <-> B) ]``
  averaged over ``R`` random splits. Its expectation is exactly zero under detailed balance (the two factors
  are independent), so no null-bias subtraction is needed. The standard error is a **cluster bootstrap over
  organism blocks** (organisms are hashed into ``K`` blocks; a resample is a multinomial weighting of the
  blocks, and halves are drawn over the blocks before weighting so no block sits in both halves).

It is a **lower bound** on the entropy production of the fine-grained dynamics (coarse-graining and lag
sampling only lose irreversibility), valid for a stationary window; the scale curve (cell 4, 8, 16, 32) shows
the dependence on the lattice and the headline is fixed at cell 8. Units of the reported rate: nats per
organism per 1000 steps.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from alife.analysis.emergence.frames import Frames
from alife.analysis.emergence.lattice import (
    HORIZONTAL,
    VERTICAL,
    Lattice,
    Transitions,
    coarsen,
    extract_transitions,
    lattice_for,
    min_image,
    own_positions,
)
from alife.analysis.phylo import splitmix64

EPR_VERSION: Final = 1
PSEUDO: Final = 0.5  # Jeffreys pseudocount
WINDOW_BLOCKS: Final = 16  # time blocks of a window endpoint (prereg decision.blocks.window): 16 x 625 steps
K_BLOCKS: Final = 20
R_SPLITS: Final = 8
N_BOOT: Final = 100
BOOT_SPLITS: Final = 2
MIN_PER_PAIR: Final = 50  # a subset needs N >= 50 x (active pairs)
JS_THRESHOLD: Final = 0.02  # provisional stationarity threshold (nats), calibrated on the seed twins at E7
SCALES: Final = (4.0, 8.0, 16.0, 32.0)
HEADLINE_CELL: Final = 8.0
SEED: Final = 0xE9B0

CAVEATS: Final = (
    "Entropy production here is a lower bound: coarse-graining the lattice and sampling at a lag only lose "
    "irreversibility. It is not 'the' entropy production of the organisms.",
    "Valid for a window with stationary occupancy; a window flagged nonstationary is not used in inference.",
    "The headline is the own-motion value (wind drift subtracted); raw minus own is the ambient part.",
    "A faster population raises the rate; ep_per_move and activity say whether speed explains a difference.",
)


# --- edge tables ---


@dataclass
class EdgeTable:
    """Transition counts of undirected cell pairs, per organism block.

    ``fwd[k, e]`` counts moves ``lo -> hi`` and ``bwd[k, e]`` moves ``hi -> lo`` of edge ``e = (lo, hi)`` in
    block ``k``; ``n[k]`` is every transition of the block (stays included); ``moves[k]`` the
    cell-changing ones.
    """

    n_cells: int
    edges: np.ndarray  # (E, 2) int64, lo < hi
    fwd: np.ndarray  # (K, E) float64
    bwd: np.ndarray
    n: np.ndarray  # (K,)
    moves: np.ndarray  # (K,)

    @property
    def k(self) -> int:
        return int(self.n.shape[0])

    @property
    def active_pairs(self) -> int:
        return int(((self.fwd + self.bwd).sum(0) > 0).sum())

    def total(self) -> int:
        return int(self.n.sum())


def block_ids(uid: np.ndarray, k: int = K_BLOCKS, seed: int = SEED) -> np.ndarray:
    """The organism block of each uid (a hash, so one organism is always in one block)."""
    return (splitmix64(uid, seed) % np.uint64(k)).astype(np.int64)


def edge_table(ci: np.ndarray, cj: np.ndarray, blk: np.ndarray, n_cells: int, k: int = K_BLOCKS) -> EdgeTable:
    """Count transitions into an ``EdgeTable`` (``blk`` in ``[0, k)`` per transition)."""
    ci = np.asarray(ci, np.int64)
    cj = np.asarray(cj, np.int64)
    blk = np.asarray(blk, np.int64)
    n = np.bincount(blk, minlength=k).astype(np.float64)
    mv = ci != cj
    moves = np.bincount(blk[mv], minlength=k).astype(np.float64)
    lo, hi = np.minimum(ci[mv], cj[mv]), np.maximum(ci[mv], cj[mv])
    fw = ci[mv] < cj[mv]
    key = lo * n_cells + hi
    uniq, inv = np.unique(key, return_inverse=True)
    e = len(uniq)
    b = blk[mv]
    fwd = np.bincount((b * e + inv)[fw], minlength=k * e).reshape(k, e).astype(np.float64)
    bwd = np.bincount((b * e + inv)[~fw], minlength=k * e).reshape(k, e).astype(np.float64)
    edges = np.stack([uniq // n_cells, uniq % n_cells], axis=1) if e else np.zeros((0, 2), np.int64)
    return EdgeTable(n_cells, edges, fwd, bwd, n, moves)


def table_of(t: Transitions, k: int = K_BLOCKS, seed: int = SEED) -> EdgeTable:
    return edge_table(t.ci, t.cj, block_ids(t.uid, k, seed), t.n_cells, k)


# --- estimators ---


def plugin(tab: EdgeTable, a: float = PSEUDO) -> float:
    """The plug-in antisymmetric estimate in nats per transition (nan with no transitions)."""
    n = tab.n.sum()
    if n <= 0 or tab.fwd.shape[1] == 0:
        return float("nan") if n <= 0 else 0.0
    f, b = tab.fwd.sum(0), tab.bwd.sum(0)
    return float(((f - b) / n * np.log((f + a) / (b + a))).sum())


def _halves(tab: EdgeTable, wa: np.ndarray, wb: np.ndarray, a: float) -> np.ndarray:
    """Cross-fitted estimates for stacked weightings: ``wa``, ``wb`` ``(S, K)`` weights of halves A and B."""
    fa, ba, fb, bb = wa @ tab.fwd, wa @ tab.bwd, wb @ tab.fwd, wb @ tab.bwd
    na, nb = wa @ tab.n, wb @ tab.n
    with np.errstate(divide="ignore", invalid="ignore"):
        s1 = ((fb - bb) / nb[:, None] * np.log((fa + a) / (ba + a))).sum(1)
        s2 = ((fa - ba) / na[:, None] * np.log((fb + a) / (bb + a))).sum(1)
    out = 0.5 * (s1 + s2)
    out[(na <= 0) | (nb <= 0)] = np.nan
    return out


def _split_masks(rng: np.random.Generator, k: int, r: int) -> np.ndarray:
    m = np.zeros((r, k))
    for i in range(r):
        m[i, rng.permutation(k)[: k // 2]] = 1.0
    return m


def crossfit(tab: EdgeTable, rng: np.random.Generator, r: int = R_SPLITS, a: float = PSEUDO) -> float:
    """The cross-fitted estimate in nats per transition, averaged over ``r`` random organism splits."""
    if tab.k < 2 or tab.n.sum() <= 0:
        return float("nan")
    m = _split_masks(rng, tab.k, r)
    v = _halves(tab, m, 1.0 - m, a)
    return float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")


def bootstrap(
    tab: EdgeTable,
    rng: np.random.Generator,
    n_boot: int = N_BOOT,
    splits: int = BOOT_SPLITS,
    a: float = PSEUDO,
) -> np.ndarray:
    """Cluster-bootstrap replicates of the cross-fitted estimate (organism blocks resampled)."""
    k = tab.k
    if k < 2 or n_boot < 1:
        return np.zeros(0)
    w = rng.multinomial(k, np.full(k, 1.0 / k), size=n_boot).astype(np.float64)  # (B, K)
    m = _split_masks(rng, k, splits)  # (R, K)
    wa = (w[:, None, :] * m[None]).reshape(-1, k)
    wb = (w[:, None, :] * (1.0 - m)[None]).reshape(-1, k)
    v = _halves(tab, wa, wb, a).reshape(n_boot, splits)
    ok = np.isfinite(v)
    cnt = ok.sum(1)
    return np.where(cnt > 0, np.where(ok, v, 0.0).sum(1) / np.maximum(cnt, 1), np.nan)


def rate(sigma: float, dt: int) -> float:
    """Nats per transition to nats per organism per 1000 steps."""
    return sigma / dt * 1000.0


def _rng(seed: int, *salt: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed, *salt]))


def estimate(
    tab: EdgeTable,
    dt: int,
    *,
    seed: int = SEED,
    salt: int = 0,
    n_boot: int = N_BOOT,
    r: int = R_SPLITS,
) -> dict[str, Any]:
    """The estimate of one table: ``rate`` (nats/organism/1000 steps), ``se`` (cluster bootstrap), ``z``,
    the plug-in diagnostic, the share of cell-changing transitions and the counts."""
    rng = _rng(seed, salt)
    s = crossfit(tab, rng, r)
    bt = bootstrap(tab, rng, n_boot)
    se = float(np.nanstd(bt, ddof=1)) if np.isfinite(bt).sum() >= 2 else float("nan")
    n = tab.total()
    share = float(tab.moves.sum() / n) if n else float("nan")
    rt = rate(s, dt)
    se_rate = rate(se, dt)
    return {
        "rate": _f(rt),
        "se": _f(se_rate),
        "z": _f(rt / se_rate) if se_rate and math.isfinite(se_rate) and se_rate > 0 else None,
        "plugin": _f(rate(plugin(tab), dt)),
        "move_share": _f(share),
        "ep_per_move": _f(rt / share) if share and math.isfinite(share) and share > 0 else None,
        "n": n,
        "active_pairs": tab.active_pairs,
    }


def _f(x: float) -> float | None:
    return float(x) if x is not None and math.isfinite(x) else None


# --- stationarity, subsets, blocks ---


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence (nats) of two count vectors; nan if either is empty."""
    p, q = np.asarray(p, np.float64), np.asarray(q, np.float64)
    if p.sum() <= 0 or q.sum() <= 0:
        return float("nan")
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)

    def kl(x: np.ndarray) -> float:
        s = x > 0
        return float((x[s] * np.log(x[s] / m[s])).sum())

    return 0.5 * kl(p) + 0.5 * kl(q)


def stationarity(t: Transitions, threshold: float = JS_THRESHOLD) -> dict[str, Any]:
    """JS divergence between the occupancy (cell of the earlier frame) of the first and second halves of the
    window, less its finite-sample bias ``(K - 1) / 8 (1/n1 + 1/n2)`` for ``K`` occupied cells (the expected
    JS of two samples of one law). Organisms recur across frames, so the real excess is larger (about
    14 / n_organisms on a diffusing toy: 0.005 at 3000) and the threshold is to be set from the seed twins.
    ``nonstationary`` when the excess exceeds ``threshold`` (provisional)."""
    frames = np.unique(t.frame)
    if t.n_pairs < 2 or not len(t) or len(frames) < 2:
        return {"js": None, "js_bias": None, "js_excess": None, "threshold": threshold, "nonstationary": None}
    cut = frames[len(frames) // 2]
    first = np.bincount(t.ci[t.frame < cut], minlength=t.n_cells)
    second = np.bincount(t.ci[t.frame >= cut], minlength=t.n_cells)
    js = js_divergence(first, second)
    k = int(((first + second) > 0).sum())
    bias = (k - 1) / 8.0 * (1.0 / first.sum() + 1.0 / second.sum())
    ex = js - bias
    return {
        "js": _f(js),
        "js_bias": _f(bias),
        "js_excess": _f(ex),
        "threshold": threshold,
        "nonstationary": bool(ex > threshold),
    }


def subset(
    t: Transitions, mask: np.ndarray, dt: int, *, seed: int = SEED, salt: int = 0, n_boot: int = N_BOOT
) -> dict[str, Any]:
    """The estimate on a subset of transitions, or ``null`` with the reason when ``N < 50 x active pairs``."""
    sub = t.select(mask)
    if not len(sub):
        return {"rate": None, "reason": "no transitions", "n": 0}
    tab = table_of(sub)
    pairs = tab.active_pairs
    if len(sub) < MIN_PER_PAIR * max(pairs, 1):
        return {
            "rate": None,
            "reason": f"N = {len(sub)} < {MIN_PER_PAIR} x {pairs} active pairs",
            "n": len(sub),
            "active_pairs": pairs,
        }
    return estimate(tab, dt, seed=seed, salt=salt, n_boot=n_boot)


def time_blocks(t: Transitions, n_blocks: int = WINDOW_BLOCKS, *, seed: int = SEED) -> list[float | None]:
    """The own-motion rate in ``n_blocks`` equal time blocks of the window (the paired units of the
    comparison, DESIGN §44.8.1): each is the cross-fitted estimate on the transitions of its frame pairs."""
    out: list[float | None] = []
    npairs = max(t.n_pairs, 1)
    blk = (t.frame.astype(np.int64) * n_blocks) // npairs
    for b in range(n_blocks):
        sel = t.select(blk == b)
        if not len(sel):
            out.append(None)
            continue
        s = crossfit(table_of(sel), _rng(seed, 7, b))
        out.append(_f(rate(s, t.dt)))
    return out


# --- flux ---


def flux_field(t: Transitions, lat: Lattice) -> np.ndarray:
    """The net-flux arrows ``(ncell, ncell, 2)``: per cell, the sum over its edges of the net outflow per
    transition times the unit-less displacement to the neighbour (min image). Negated by time reversal."""
    tab = table_of(t, 1)
    out = np.zeros((lat.n_cells, 2))
    if not tab.edges.shape[0] or not tab.total():
        return out.reshape(lat.ncell, lat.ncell, 2)
    cen = lat.centres()
    net = (tab.fwd[0] - tab.bwd[0]) / tab.total()  # lo -> hi
    lo, hi = tab.edges[:, 0], tab.edges[:, 1]
    vec = min_image(cen[hi] - cen[lo], lat.size)
    np.add.at(out, lo, net[:, None] * vec)
    np.add.at(out, hi, net[:, None] * vec)  # outflow of hi is -net along -vec
    return out.reshape(lat.ncell, lat.ncell, 2)


# --- the window analysis ---


def displacement_stats(t: Transitions, lat: Lattice) -> dict[str, Any]:
    """Median horizontal displacement in cell units and the share of cell-changing transitions (the tau
    calibration targets: median 0.25 to 0.5 cell, share 20 to 60%)."""
    if not len(t):
        return {"median_cells": None, "move_share": None}
    return {
        "median_cells": _f(float(np.median(t.move)) / lat.cell),
        "move_share": _f(float((t.ci != t.cj).mean())),
    }


def analyse_window(
    fr: Frames,
    *,
    cell: float = HEADLINE_CELL,
    stride: int = 1,
    scales: Sequence[float] = SCALES,
    top_clades: int = 8,
    n_boot: int = N_BOOT,
    seed: int = SEED,
    js_threshold: float = JS_THRESHOLD,
    blocks: int = WINDOW_BLOCKS,
) -> dict[str, Any]:
    """Block B of one frames file: headline own-motion and raw rates, the ambient part, scale curve,
    vertical bands, subsets (top clades; bodied against free), stationarity, and the 8 time-block values."""
    lat = lattice_for(fr.size, cell)
    own = own_positions(fr, own=True)
    has_drift = fr.has_drift
    raw = own_positions(fr, own=False) if has_drift else own
    t_own = extract_transitions(fr, cell=cell, own=True, stride=stride, tracks=own)
    t_raw = extract_transitions(fr, cell=cell, own=False, stride=stride, tracks=raw) if has_drift else t_own
    dt = t_own.dt
    e_own = estimate(table_of(t_own), dt, seed=seed, salt=1, n_boot=n_boot)
    e_raw = estimate(table_of(t_raw), dt, seed=seed, salt=2, n_boot=n_boot) if has_drift else e_own
    amb = (
        _f(e_raw["rate"] - e_own["rate"]) if e_raw["rate"] is not None and e_own["rate"] is not None else None
    )
    curve = []
    for c in scales:
        tc = extract_transitions(fr, cell=c, own=True, stride=stride, tracks=own)
        ec = estimate(table_of(tc), dt, seed=seed, salt=10 + int(c), n_boot=n_boot)
        curve.append({"cell": c, "n_cells": tc.n_cells, **ec})
    vertical = None
    if fr.nd == 3:
        tv = extract_transitions(fr, own=True, stride=stride, axis=VERTICAL, tracks=own)
        vertical = {
            "axis": VERTICAL,
            "n_bands": tv.n_cells,
            **estimate(table_of(tv), dt, seed=seed, salt=3, n_boot=n_boot),
        }
    # subsets, own-motion headline lattice
    clades: list[dict[str, Any]] = []
    if len(t_own):
        cl, cnt = np.unique(t_own.clade, return_counts=True)
        for j in np.argsort(-cnt, kind="stable")[:top_clades]:
            r = subset(t_own, t_own.clade == cl[j], dt, seed=seed, salt=100 + int(cl[j]) % 997, n_boot=n_boot)
            clades.append({"clade": int(cl[j]), **r})
    bodied = subset(t_own, t_own.body_size >= 8, dt, seed=seed, salt=4, n_boot=n_boot)
    free = subset(t_own, t_own.body_size == 0, dt, seed=seed, salt=5, n_boot=n_boot)
    stat = stationarity(t_own, js_threshold)
    return {
        "version": EPR_VERSION,
        "cell": cell,
        "tau": dt,
        "stride": stride,
        "lattice": {"ncell": lat.ncell, "n_cells": lat.n_cells},
        "has_drift": has_drift,
        "frame_pairs": t_own.n_pairs,
        "skipped_pairs": t_own.skipped,
        "own": e_own,
        "raw": e_raw,
        "ambient": amb,
        "activity": _f(float(t_own.move.mean()) / dt) if len(t_own) else None,
        "displacement": displacement_stats(t_own, lat),
        "scale_curve": curve,
        "vertical": vertical,
        "clades": clades,
        "bodies": {"bodied_ge8": bodied, "free": free},
        "stationarity": stat,
        "blocks": time_blocks(t_own, blocks, seed=seed),
        "unit": "nats per organism per 1000 steps",
        "caveats": list(CAVEATS),
    }


def flux_for_plot(fr: Frames, cell: float = HEADLINE_CELL, stride: int = 1) -> tuple[np.ndarray, Lattice]:
    lat = lattice_for(fr.size, cell)
    return flux_field(extract_transitions(fr, cell=cell, own=True, stride=stride), lat), lat


__all__ = [
    "EPR_VERSION",
    "HORIZONTAL",
    "EdgeTable",
    "analyse_window",
    "bootstrap",
    "coarsen",
    "crossfit",
    "edge_table",
    "estimate",
    "flux_field",
    "plugin",
    "stationarity",
    "subset",
    "table_of",
    "time_blocks",
]
