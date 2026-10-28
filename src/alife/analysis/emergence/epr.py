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
    advection_slope,
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
MIN_EDGE: float = 0.0  # edges with fewer fit-half transitions are left out of the circulating sum
N_BOOT: Final = 100
BOOT_SPLITS: Final = 2
MIN_PER_PAIR: Final = 50  # a subset needs N >= 50 x (active pairs)
JS_THRESHOLD: Final = 0.02  # provisional stationarity threshold (nats), calibrated on the seed twins at E7
SCALES: Final = (4.0, 8.0, 16.0, 32.0)
HEADLINE_CELL: Final = 8.0
SEED: Final = 0xE9B0

CAVEATS: Final = (
    "Entropy production here is a lower bound: coarse-graining the lattice and sampling at a lag only "
    "lose irreversibility. It is not 'the' entropy production of the organisms.",
    "Valid for a window with stationary occupancy; a nonstationary window is not used in inference.",
    "The headline is the CIRCULATING part of the own-motion current (wind drift subtracted). The total "
    "cross-fitted value also counts source-sink currents that births and deaths drive through reversible "
    "movement (hot-spot births alone give a large total and a circulating part of about 0); the total, "
    "the source-sink part and a turnover-matched reversible null are reported beside it.",
    "A faster population raises the rate; ep_per_move and activity say whether speed explains a difference.",
    "The error is the larger of an organism-block (body-aware) and a time-block bootstrap; for independent "
    "organisms it is about 1.8 times conservative, so z is understated.",
)


# --- edge tables ---


@dataclass
class EdgeTable:
    """Transition counts of undirected cell pairs per bootstrap unit.

    A unit is ``(time block t, organism block k)``, laid out ``u = t * n_org + k``. ``fwd[u, e]`` counts moves
    ``lo -> hi`` of edge ``e = (lo, hi)`` and ``bwd[u, e]`` moves ``hi -> lo``; ``n[u]`` is every transition
    of the unit (stays included); ``moves[u]`` the cell-changing ones.
    """

    n_cells: int
    edges: np.ndarray  # (E, 2) int64, lo < hi
    fwd: np.ndarray  # (U, E) float64
    bwd: np.ndarray
    n: np.ndarray  # (U,)
    moves: np.ndarray  # (U,)
    n_org: int = K_BLOCKS
    n_time: int = 1

    @property
    def k(self) -> int:
        return self.n_org

    @property
    def active_pairs(self) -> int:
        return int(((self.fwd + self.bwd).sum(0) > 0).sum())

    def total(self) -> int:
        return int(self.n.sum())

    def active_nodes(self) -> int:
        return int(np.unique(self.edges).size) if len(self.edges) else 0


def block_ids(uid: np.ndarray, k: int = K_BLOCKS, seed: int = SEED) -> np.ndarray:
    """The organism block of each uid (a hash, so one organism is always in one block)."""
    return (splitmix64(uid, seed) % np.uint64(k)).astype(np.int64)


def edge_table(
    ci: np.ndarray,
    cj: np.ndarray,
    blk: np.ndarray,
    n_cells: int,
    k: int = K_BLOCKS,
    tblk: np.ndarray | None = None,
    n_time: int = 1,
) -> EdgeTable:
    """Count transitions into an ``EdgeTable`` (``blk`` in ``[0, k)`` the organism block, ``tblk`` in
    ``[0, n_time)`` the time block of each transition)."""
    ci = np.asarray(ci, np.int64)
    cj = np.asarray(cj, np.int64)
    unit = np.asarray(blk, np.int64) + (0 if tblk is None else np.asarray(tblk, np.int64) * k)
    u_n = k * n_time
    n = np.bincount(unit, minlength=u_n).astype(np.float64)
    mv = ci != cj
    moves = np.bincount(unit[mv], minlength=u_n).astype(np.float64)
    lo, hi = np.minimum(ci[mv], cj[mv]), np.maximum(ci[mv], cj[mv])
    fw = ci[mv] < cj[mv]
    key = lo * n_cells + hi
    uniq, inv = np.unique(key, return_inverse=True)
    e = len(uniq)
    b = unit[mv]
    fwd = np.bincount((b * e + inv)[fw], minlength=u_n * e).reshape(u_n, e).astype(np.float64)
    bwd = np.bincount((b * e + inv)[~fw], minlength=u_n * e).reshape(u_n, e).astype(np.float64)
    edges = np.stack([uniq // n_cells, uniq % n_cells], axis=1) if e else np.zeros((0, 2), np.int64)
    return EdgeTable(n_cells, edges, fwd, bwd, n, moves, k, n_time)


def group_keys(t: Transitions) -> np.ndarray:
    """The organism-block key of each transition: the coarse cell where the organism first appears in the
    window (births included). Organisms that move as a group (offspring, flocks, bodies) appear together, so
    they share a block and are resampled and cross-fitted together; hashing individual uids would put the
    members of a group in both halves of the cross-fit and make the error far too small. Organisms still
    wander over the whole lattice, so each half sees every cell pair. Falls back to the uid without
    origins."""
    if t.origin is None:
        return t.uid.astype(np.int64)
    return t.origin.astype(np.int64)


def table_of(t: Transitions, k: int = K_BLOCKS, seed: int = SEED, n_time: int = 1) -> EdgeTable:
    npairs = max(t.n_pairs, 1)
    tblk = (t.frame.astype(np.int64) * n_time) // npairs if n_time > 1 else None
    return edge_table(t.ci, t.cj, block_ids(group_keys(t), k, seed), t.n_cells, k, tblk, n_time)


# --- estimators ---


def plugin(tab: EdgeTable, a: float = PSEUDO) -> float:
    """The plug-in antisymmetric estimate in nats per transition (nan with no transitions)."""
    n = tab.n.sum()
    if n <= 0 or tab.fwd.shape[1] == 0:
        return float("nan") if n <= 0 else 0.0
    f, b = tab.fwd.sum(0), tab.bwd.sum(0)
    return float(((f - b) / n * np.log((f + a) / (b + a))).sum())


IRLS_STEPS: Final = 4


def affinity_residual(fa: np.ndarray, ba: np.ndarray, edges: np.ndarray, a: float = PSEUDO) -> np.ndarray:
    """The affinity field ``A_e = ln((f + a)/(b + a))`` of stacked count rows ``(S, E)`` less its fitted
    gradient.

    For a reversible movement kernel the log-ratio of the counts is exactly a potential difference,
    ``ln(C_ij / C_ji) = psi_i - psi_j`` with ``psi = ln(p / pi)``, however the occupancy ``p`` is fed by
    births and deaths; a source-sink current is therefore a pure gradient field. The potential is fitted by
    binomial maximum likelihood on the graph of active cell pairs (``f ~ Bin(f + b, sigma(psi_lo -
    psi_hi))``, IRLS with ``IRLS_STEPS`` weighted least-squares steps) and the residual is ``A_e`` less the
    affinity of the fitted expected counts, which carries the same pseudocount shrinkage and so cancels it.
    What remains is the cycle-space (circulating) part. The sum of ``J_e`` times the gradient part equals
    ``sum_i psi_i div J_i``: zero for a closed stationary chain, the source-sink contribution of an open
    one."""
    s_n, e_n = fa.shape
    out = np.zeros_like(fa)
    if e_n == 0:
        return out
    nodes, inv = np.unique(edges.ravel(), return_inverse=True)
    m = len(nodes)
    li = inv.reshape(-1, 2)
    n_e = fa + ba
    a_e = np.log((fa + a) / (ba + a))
    p_hat = np.divide(fa, n_e, out=np.full_like(fa, 0.5), where=n_e > 0)
    chunk = max(1, int(2.5e7 // (m * m)))
    ar = np.arange(m)
    for s0 in range(0, s_n, chunk):
        sl = slice(s0, min(s0 + chunk, s_n))
        nc, pc = n_e[sl], p_hat[sl]
        c = nc.shape[0]
        rows = np.arange(c)[:, None]
        psi = np.zeros((c, m))
        eta = np.zeros_like(nc)
        for _ in range(IRLS_STEPS):
            sig = np.clip(1.0 / (1.0 + np.exp(-eta)), 1e-3, 1 - 1e-3)
            wc = nc * sig * (1 - sig)
            z = eta + (pc - sig) / (sig * (1 - sig))
            diag = np.zeros((c, m))
            np.add.at(diag, (rows, li[:, 0][None]), wc)
            np.add.at(diag, (rows, li[:, 1][None]), wc)
            rhs = np.zeros((c, m))
            np.add.at(rhs, (rows, li[:, 0][None]), wc * z)
            np.add.at(rhs, (rows, li[:, 1][None]), -wc * z)
            lap = np.zeros((c, m, m))
            lap[:, li[:, 0], li[:, 1]] = -wc
            lap[:, li[:, 1], li[:, 0]] = -wc
            lap[:, ar, ar] = diag + 1e-6 * (diag.mean(1, keepdims=True) + 1e-12) + 1e-9
            psi = np.linalg.solve(lap, rhs[..., None])[..., 0]
            eta = np.clip(psi[:, li[:, 0]] - psi[:, li[:, 1]], -12.0, 12.0)
        sig = 1.0 / (1.0 + np.exp(-eta))
        pred = np.log((nc * sig + a) / (nc * (1 - sig) + a))
        out[sl] = a_e[sl] - pred
    return out


def _halves(
    tab: EdgeTable, wa: np.ndarray, wb: np.ndarray, a: float, hodge: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Cross-fitted ``(total, circulating)`` estimates for stacked unit weightings ``wa``, ``wb`` ``(S, U)``.

    The affinity comes from one half, the net flux from the other. ``total`` uses the full affinity;
    ``circulating`` the affinity less its fitted gradient part."""
    fa, ba, fb, bb = wa @ tab.fwd, wa @ tab.bwd, wb @ tab.fwd, wb @ tab.bwd
    na, nb = wa @ tab.n, wb @ tab.n
    with np.errstate(divide="ignore", invalid="ignore"):
        aff_a = np.log((fa + a) / (ba + a))
        aff_b = np.log((fb + a) / (bb + a))
        jb, ja = (fb - bb) / nb[:, None], (fa - ba) / na[:, None]
        tot = 0.5 * ((jb * aff_a).sum(1) + (ja * aff_b).sum(1))
        if hodge:
            ra = affinity_residual(fa, ba, tab.edges, a)
            rb = affinity_residual(fb, bb, tab.edges, a)
            ma, mb = (fa + ba >= MIN_EDGE), (fb + bb >= MIN_EDGE)
            circ = 0.5 * ((jb * ra * ma).sum(1) + (ja * rb * mb).sum(1))
        else:
            circ = tot
    bad = (na <= 0) | (nb <= 0)
    tot[bad] = np.nan
    circ[bad] = np.nan
    return tot, circ


def _split_masks(rng: np.random.Generator, k: int, r: int) -> np.ndarray:
    m = np.zeros((r, k))
    for i in range(r):
        m[i, rng.permutation(k)[: k // 2]] = 1.0
    return m


def _unit_org(tab: EdgeTable) -> np.ndarray:
    return np.tile(np.arange(tab.n_org), tab.n_time)


def _unit_time(tab: EdgeTable) -> np.ndarray:
    return np.repeat(np.arange(tab.n_time), tab.n_org)


def crossfit(
    tab: EdgeTable, rng: np.random.Generator, r: int = R_SPLITS, a: float = PSEUDO, hodge: bool = True
) -> tuple[float, float]:
    """``(total, circulating)`` cross-fitted estimates in nats per transition, averaged over ``r`` random
    organism splits (the splits are over organism blocks; every time block is in both halves)."""
    if tab.n_org < 2 or tab.n.sum() <= 0:
        return float("nan"), float("nan")
    m = _split_masks(rng, tab.n_org, r)[:, _unit_org(tab)]
    tot, circ = _halves(tab, m, 1.0 - m, a, hodge)

    def mean(v: np.ndarray) -> float:
        return float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")

    return mean(tot), mean(circ)


def bootstrap(
    tab: EdgeTable,
    rng: np.random.Generator,
    n_boot: int = N_BOOT,
    splits: int = BOOT_SPLITS,
    a: float = PSEUDO,
    over: str = "organisms",
    hodge: bool = True,
) -> np.ndarray:
    """Bootstrap replicates of the circulating estimate. ``over`` ``"organisms"`` resamples the organism
    blocks (which hash the body for bonded organisms); ``"time"`` resamples the time blocks (organisms that
    move as a group are then resampled together). Halves are drawn over organism blocks before weighting, so
    no block is in both."""
    uo, ut = _unit_org(tab), _unit_time(tab)
    k = tab.n_org if over == "organisms" else tab.n_time
    if k < 2 or n_boot < 1 or tab.n_org < 2:
        return np.zeros(0)
    w = rng.multinomial(k, np.full(k, 1.0 / k), size=n_boot).astype(np.float64)  # (B, k)
    m = _split_masks(rng, tab.n_org, splits)[:, uo]  # (R, U)
    wu = w[:, uo] if over == "organisms" else w[:, ut]  # (B, U)
    wa = (wu[:, None, :] * m[None]).reshape(-1, wu.shape[1])
    wb = (wu[:, None, :] * (1.0 - m)[None]).reshape(-1, wu.shape[1])
    _, circ = _halves(tab, wa, wb, a, hodge)
    v = circ.reshape(n_boot, splits)
    ok = np.isfinite(v)
    cnt = ok.sum(1)
    return np.where(cnt > 0, np.where(ok, v, 0.0).sum(1) / np.maximum(cnt, 1), np.nan)


def rate(sigma: float, dt: int) -> float:
    """Nats per transition to nats per organism per 1000 steps."""
    return sigma / dt * 1000.0


def _rng(seed: int, *salt: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed, *salt]))


def boot_budget(n_boot: int, nodes: int) -> int:
    """Bootstrap replicates affordable for a graph of ``nodes`` cells (the weighted fit solves a dense
    ``nodes x nodes`` system per replicate): full at 500 nodes or fewer, scaling with the cube beyond."""
    return n_boot if nodes <= 500 else max(10, int(n_boot * (500.0 / nodes) ** 3))


def estimate(
    tab: EdgeTable,
    dt: int,
    *,
    seed: int = SEED,
    salt: int = 0,
    n_boot: int = N_BOOT,
    r: int = R_SPLITS,
) -> dict[str, Any]:
    """The estimate of one table.

    ``rate`` is the circulating cross-fitted value (nats per organism per 1000 steps); ``rate_total`` the
    total (circulating plus source-sink), ``source_sink`` their difference. ``se`` is the larger of the
    organism-block and the time-block bootstrap (``se_org``, ``se_time``); ``z`` is ``rate / se``. Also the
    plug-in diagnostic, the share of cell-changing transitions and the counts."""
    rng = _rng(seed, salt)
    _, circ = crossfit(tab, rng, r)
    nb = boot_budget(n_boot, tab.active_nodes())
    b_org = bootstrap(tab, rng, nb, over="organisms")
    b_time = bootstrap(tab, rng, nb, over="time") if tab.n_time > 1 else np.zeros(0)

    def sd(v: np.ndarray) -> float:
        return float(np.nanstd(v, ddof=1)) if np.isfinite(v).sum() >= 2 else float("nan")

    se_o, se_t = sd(b_org), sd(b_time)
    ses = [x for x in (se_o, se_t) if math.isfinite(x)]
    se = max(ses) if ses else float("nan")
    n = tab.total()
    share = float(tab.moves.sum() / n) if n else float("nan")
    rt, rtot, se_rate = rate(circ, dt), rate(plugin(tab), dt), rate(se, dt)
    return {
        "rate": _f(rt),
        "se": _f(se_rate),
        "se_org": _f(rate(se_o, dt)),
        "se_time": _f(rate(se_t, dt)),
        "z": _f(rt / se_rate) if se_rate and math.isfinite(se_rate) and se_rate > 0 else None,
        "rate_total": _f(rtot),
        "source_sink": _f(rtot - rt),
        "plugin": _f(rtot),
        "move_share": _f(share),
        "ep_per_move": _f(rt / share) if share and math.isfinite(share) and share > 0 else None,
        "n": n,
        "active_pairs": tab.active_pairs,
        "boot": nb,
    }


def _f(x: float) -> float | None:
    return float(x) if x is not None and math.isfinite(x) else None


def gradient_null(t: Transitions, rng: np.random.Generator, a: float = PSEUDO) -> Transitions:
    """A turnover-matched reversible surrogate of the transitions (the gradient-only counterpart of the data).

    Every transition keeps its start cell, organism and frame; its destination is redrawn from the kernel
    whose directed counts keep the observed traffic of every cell pair but carry only the **gradient part**
    of the affinity (the circulating part removed, ``affinity_residual``), with the observed stays. Movement
    is then reversible, the start-cell law is the observed one (births and deaths included): the surrogate
    shows the source-sink current that reversible movement under the observed turnover gives (its total) and,
    on it, the estimator should read zero circulation (its circulating value is the estimator's bias in this
    window)."""
    n_c = t.n_cells
    mv = t.ci != t.cj
    lo, hi = np.minimum(t.ci[mv], t.cj[mv]).astype(np.int64), np.maximum(t.ci[mv], t.cj[mv]).astype(np.int64)
    fw = (t.ci[mv] < t.cj[mv]).astype(np.float64)
    uniq, inv = np.unique(lo * n_c + hi, return_inverse=True)
    f = np.bincount(inv, weights=fw, minlength=len(uniq))
    b = np.bincount(inv, weights=1.0 - fw, minlength=len(uniq))
    edges = np.stack([uniq // n_c, uniq % n_c], axis=1)
    if len(uniq):
        res = affinity_residual(f[None], b[None], edges, a)[0]
        grad = np.log((f + a) / (b + a)) - res
        wsum = f + b
        f2 = wsum / (1.0 + np.exp(-grad))
        b2 = wsum - f2
    else:
        f2 = b2 = np.zeros(0)
    stays = np.bincount(t.ci[~mv], minlength=n_c).astype(np.float64)
    src = np.concatenate([uniq // n_c, uniq % n_c, np.arange(n_c)])
    dst = np.concatenate([uniq % n_c, uniq // n_c, np.arange(n_c)])
    wt = np.concatenate([f2, b2, stays])
    keep = wt > 0
    src, dst, wt = src[keep], dst[keep], wt[keep]
    o = np.argsort(src, kind="stable")
    src, dst, wt = src[o], dst[o], wt[o]
    tot = np.bincount(src, weights=wt, minlength=n_c)
    cum = np.cumsum(wt)
    start = np.concatenate([[0.0], cum])[np.searchsorted(src, np.arange(n_c), side="left")]
    keys = src + np.minimum((cum - start[src]) / tot[src], 1.0)
    u = rng.random(len(t)) * (1.0 - 1e-12) + 1e-13
    idx = np.minimum(np.searchsorted(keys, t.ci.astype(np.float64) + u, side="left"), len(keys) - 1)
    out = t.select(np.ones(len(t), bool))
    out.cj = dst[idx].astype(np.int32)
    return out


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
    tab = table_of(sub, n_time=WINDOW_BLOCKS)
    pairs = tab.active_pairs
    if len(sub) < MIN_PER_PAIR * max(pairs, 1):
        return {
            "rate": None,
            "reason": f"N = {len(sub)} < {MIN_PER_PAIR} x {pairs} active pairs",
            "n": len(sub),
            "active_pairs": pairs,
        }
    return estimate(tab, dt, seed=seed, salt=salt, n_boot=n_boot)


def time_blocks(
    t: Transitions, n_blocks: int = WINDOW_BLOCKS, *, seed: int = SEED, r: int = R_SPLITS
) -> list[float | None]:
    """The circulating own-motion rate in ``n_blocks`` equal time blocks of the window (the paired units of
    the comparison, DESIGN §44.8.1).

    The affinity (and its gradient part) is fitted on the **whole window** of one organism half, the net flux
    is that of the other half's transitions **inside the block**, and the two halves swap, averaged over
    ``r`` random splits. A block's own table is too sparse to separate circulating from gradient flux (on the
    tree-like part of a sparse graph every edge is fitted exactly and counts as gradient), so fitting the
    block alone would bias its value by the block's sample size; this way a block differs from the others only
    through its flux, which is unbiased."""
    tab = table_of(t, n_time=n_blocks)
    n_org, e = tab.n_org, tab.edges.shape[0]
    if t.n_pairs < n_blocks or e == 0:
        return [None] * n_blocks
    rng = _rng(seed, 7)
    fw = tab.fwd.reshape(n_blocks, n_org, e)
    bw = tab.bwd.reshape(n_blocks, n_org, e)
    nn = tab.n.reshape(n_blocks, n_org)
    masks = _split_masks(rng, n_org, r)  # (R, K) 1 = half A
    fa = np.einsum("rk,tke->re", masks, fw)
    ba = np.einsum("rk,tke->re", masks, bw)
    fb = np.einsum("rk,tke->re", 1 - masks, fw)
    bb = np.einsum("rk,tke->re", 1 - masks, bw)
    res_a = affinity_residual(fa, ba, tab.edges)
    res_b = affinity_residual(fb, bb, tab.edges)
    out: list[float | None] = []
    for b in range(n_blocks):
        jb = np.einsum("rk,ke->re", 1 - masks, fw[b] - bw[b])
        ja = np.einsum("rk,ke->re", masks, fw[b] - bw[b])
        nb_ = (1 - masks) @ nn[b]
        na_ = masks @ nn[b]
        ok = (nb_ > 0) & (na_ > 0)
        if not ok.any():
            out.append(None)
            continue
        v = 0.5 * ((jb * res_a).sum(1) / np.maximum(nb_, 1) + (ja * res_b).sum(1) / np.maximum(na_, 1))
        out.append(_f(rate(float(v[ok].mean()), t.dt)))
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
    vertical bands, subsets (top clades; bodied against free), stationarity, and the 16 time-block values."""
    lat = lattice_for(fr.size, cell)
    has_drift = fr.has_drift
    slope = advection_slope(fr) if has_drift else None
    # recorded drift that does not advect the organisms must not be subtracted (a spurious current)
    drift_applied = has_drift and (slope is None or abs(slope - 1.0) <= 0.5)
    raw = own_positions(fr, own=False)
    own = own_positions(fr, own=True) if drift_applied else raw
    t_own = extract_transitions(fr, cell=cell, own=True, stride=stride, tracks=own)
    t_raw = (
        extract_transitions(fr, cell=cell, own=False, stride=stride, tracks=raw) if drift_applied else t_own
    )
    dt = t_own.dt
    e_own = estimate(table_of(t_own, n_time=blocks), dt, seed=seed, salt=1, n_boot=n_boot)
    e_raw = (
        estimate(table_of(t_raw, n_time=blocks), dt, seed=seed, salt=2, n_boot=n_boot)
        if drift_applied
        else e_own
    )
    amb = (
        _f(e_raw["rate"] - e_own["rate"]) if e_raw["rate"] is not None and e_own["rate"] is not None else None
    )
    curve = []
    for c in scales:
        tc = extract_transitions(fr, cell=c, own=True, stride=stride, tracks=own)
        ec = estimate(table_of(tc, n_time=blocks), dt, seed=seed, salt=10 + int(c), n_boot=n_boot)
        curve.append({"cell": c, "n_cells": tc.n_cells, **ec})
    vertical = None
    if fr.nd == 3:
        tv = extract_transitions(fr, own=True, stride=stride, axis=VERTICAL, tracks=own)
        vertical = {
            "axis": VERTICAL,
            "n_bands": tv.n_cells,
            **estimate(table_of(tv, n_time=blocks), dt, seed=seed, salt=3, n_boot=n_boot),
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
    nulls = []
    for j in range(2):
        tn = gradient_null(t_own, _rng(seed, 77, j))
        tabn = table_of(tn)
        _, c0 = crossfit(tabn, _rng(seed, 78, j))
        nulls.append((rate(plugin(tabn), dt), rate(c0, dt)))
    null = {
        "rate_total": _f(float(np.mean([x[0] for x in nulls]))),
        "rate": _f(float(np.mean([x[1] for x in nulls]))),
        "note": "kernel with the observed traffic and only the gradient part of the affinity, observed "
        "start-cell law (births and deaths): its total is the source-sink current of reversible movement "
        "under this turnover, its circulating value the estimator bias here",
    }
    return {
        "version": EPR_VERSION,
        "cell": cell,
        "tau": dt,
        "stride": stride,
        "lattice": {"ncell": lat.ncell, "n_cells": lat.n_cells},
        "has_drift": has_drift,
        "drift_removal": {"slope": _f(slope) if slope is not None else None, "applied": bool(drift_applied)},
        "frame_pairs": t_own.n_pairs,
        "skipped_pairs": t_own.skipped,
        "own": e_own,
        "raw": e_raw,
        "ambient": amb,
        "gradient_null": null,
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
    "affinity_residual",
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
