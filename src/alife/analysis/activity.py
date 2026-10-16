"""Bedau–Packard evolutionary-activity statistics on lineage components (docs/ACTIVITY.md).

Definitions follow Bedau, Snyder & Packard 1998 (ALife VI) and, for the normalisation, Channon 2002 (ALife
VIII). For a component ``i`` present at grid time ``t``:

- activity ``a_i(t)``: the cumulative number of births by members of ``i`` (reproduction events whose parent
  is in ``i``), reset to 0 once ``i`` is extinct (1998 eq. 1 with the reset of ECAL97 eq. 2; the increment
  is usage, as the task specifies, not Bedau's presence indicator). The presence version, ``a_i = t − t_i``
  (the component's age), is computed too, as a sensitivity check;
- diversity ``D(t)``: the number of present components; total ``A_cum = Σ a_i``; mean ``Ā = A_cum / D``;
- new activity ``A_new(t) = (1/D) Σ_{a0 ≤ a_i ≤ a1} a_i`` (1998 eq. 7), evaluated post hoc from per-grid
  log-binned histograms of ``a`` (so ``a0``, ``a1`` snap to the bin edges, 10 per decade);
- component-normalised activity (Channon 2002 eqs. 9–15) ``aN_i = a_i − e_i``, with ``e_i`` the expected
  births of ``i`` under Bedau's neutral shadow reset to the real state at every grid time (Channon's reset
  shadow, its expectation computed exactly instead of sampled). In a neutral shadow a component's share of
  the population is a martingale, so over the window ``[g_j, g_{j+1})`` it expects ``B_j · n_i(g_j) / N(g_j)``
  of the window's ``B_j`` births. ``aN`` therefore rewards both fecundity and survival within a window.
  ``Σ_i aN_i = 0`` over all components, extinct ones included, so ``A^N_cum`` of the present components
  equals the shortfall of the extinct ones. New normalised activity is counted at the grid time a component
  first reaches ``aN ≥ θ`` (Channon 2002), for a sweep of ``θ``;
- the fecundity-only variant ``aF_i = a_i − f_i`` (a reset every step): each step's ``b_u`` births go to
  uniformly random eligible agents (alive, age ≥ gap), so an agent expects its exposure ``Σ b_u / E_u`` over
  its eligible life and ``f_i`` sums the members' exposures. Survival differences do not enter it.

Component definitions (both heritable and identical in the shadows, which share the agents):

- ``cohort``: for each grid boundary ``τ_k`` the agents alive at ``τ_k`` root one component each, holding all
  their later descendants (the lineage subtree). Every later birth belongs to exactly one component of each
  earlier cohort, like alleles at one locus per cohort. Statistics are kept per cohort (aligned on age) and
  pooled over a sliding horizon: at time ``t`` the pool holds the cohorts aged ``0 < t − τ_k ≤ H``. Pooling
  every cohort instead would grow ``D`` and the totals with the number of cohorts by construction, because
  each cohort keeps at least the one lineage that holds the population.
- ``marker``: paraphyletic, genotype-like components. Each birth founds a new component with probability
  ``rate`` (a fixed hash of the agent index, ``phylo.marker_roots``), else joins its parent's. This is the
  genealogy seen through a neutral infinite-alleles marker with mutation rate ``rate`` per birth; ``1/rate``
  sets the component lifetime in generations.

Everything is computed in one pass over grid windows ``[g_j, g_{j+1})`` (a trailing partial window is
dropped); only the agents alive at a boundary and the births of one window are touched at a time, plus one
int32 label per agent for each ``marker`` rate. Memory is dominated by the tree (three int32 arrays) and the
``[agent alive at g_j, cohort]`` table of component ids (about 33k × 150 int32 on ``64k_s0``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from alife.analysis.phylo import Tree, eligible_from, marker_roots, min_gap, nearest_root

# Log-spaced histogram edges for activity: bin 0 is [0, 1), then 10 bins per decade up to 1e10.
A_EDGES = np.concatenate([[0.0], 10.0 ** np.arange(0.0, 10.0001, 0.1)])
N_BINS = A_EDGES.size  # the last bin is [1e10, ∞)
THETA_A = 10.0 ** np.arange(1.0, 7.01, 0.5)  # raw-activity crossing thresholds
THETA_N = 10.0 ** np.arange(0.5, 6.51, 0.5)  # normalised-activity crossing thresholds


def _bins(x: np.ndarray) -> np.ndarray:
    return np.clip(np.searchsorted(A_EDGES, x, side="right") - 1, 0, N_BINS - 1)


def grid_times(t_end: int, every: int) -> np.ndarray:
    """``0, every, 2·every, …`` up to ``t_end``. A trailing partial window is dropped (its events are not
    analysed): a short window would bias the per-window reset shadow and the cohort pools."""
    if every <= 0 or every > t_end:
        raise ValueError(f"grid spacing {every} must be in [1, t_end = {t_end}]")
    return np.arange(0, t_end + 1, every, dtype=np.int64)


def exposure_cdf(tree: Tree, gap: int) -> np.ndarray:
    """``F[t] = Σ_{u<t} b_u / E_u`` (float64, length ``t_end + 1``): one agent's expected births over the
    steps before ``t`` if it is eligible throughout; ``b_u`` are the non-root births of step ``u`` and ``E_u``
    the agents eligible in step ``u`` (``eligible_from ≤ u < death``: ``birth + gap``, 0 for a founder)."""
    t_end = tree.t_end
    start = eligible_from(tree.birth, gap)
    end = tree.death.astype(np.int64)
    ok = start < end
    e_count = np.cumsum(
        np.bincount(start[ok], minlength=t_end + 1)[: t_end + 1]
        - np.bincount(end[ok], minlength=t_end + 1)[: t_end + 1]
    )
    kids = tree.birth[tree.parent >= 0]
    b = np.bincount(kids, minlength=t_end + 1)[: t_end + 1].astype(np.float64)
    if np.any((b > 0) & (e_count <= 0)):
        raise ValueError("births in a step without eligible agents (gap too large?)")
    rate = np.divide(b, e_count, out=np.zeros_like(b), where=e_count > 0)
    return np.concatenate([[0.0], np.cumsum(rate)[:-1]])


def _exposure(tree: Tree, idx: np.ndarray, gap: int, cdf: np.ndarray, t0: int, t1: int) -> np.ndarray:
    lo = np.clip(eligible_from(tree.birth[idx], gap), t0, t1)
    hi = np.clip(tree.death[idx].astype(np.int64), t0, t1)
    return np.where(hi > lo, cdf[hi] - cdf[lo], 0.0)


# --- per-space accumulation --------------------------------------------------------------------------------


class Snapshot(NamedTuple):
    """The present components of a space at one grid time."""

    a: np.ndarray  # activity (births by members)
    an: np.ndarray  # normalised activity, reset shadow
    af: np.ndarray  # normalised activity, exposure (fecundity only)
    p: np.ndarray  # presence activity (age, steps)
    new_a: np.ndarray  # [i, θ] first reaching a ≥ THETA_A[θ] now
    new_n: np.ndarray  # [i, θ] first reaching aN ≥ THETA_N[θ] now
    created: int  # components first present now


@dataclass
class Series:
    """Statistics of one component space at every grid time (row ``g`` = grid time ``grid[g]``)."""

    n_grid: int
    D: np.ndarray = field(init=False)
    A_cum: np.ndarray = field(init=False)
    A_med: np.ndarray = field(init=False)
    AN_cum: np.ndarray = field(init=False)
    AN_med: np.ndarray = field(init=False)
    AN_min: np.ndarray = field(init=False)
    AN_max: np.ndarray = field(init=False)
    AF_cum: np.ndarray = field(init=False)  # fecundity-only normalisation (exposure)
    AF_med: np.ndarray = field(init=False)
    P_cum: np.ndarray = field(init=False)  # presence activity (component age, steps)
    P_med: np.ndarray = field(init=False)
    created: np.ndarray = field(init=False)  # components first present at this grid time
    hist_n: np.ndarray = field(init=False)  # [g, bin] count of present components by a
    hist_sum: np.ndarray = field(init=False)  # [g, bin] Σ a of those components
    hist_p_n: np.ndarray = field(init=False)  # [g, bin] by presence activity
    hist_p_sum: np.ndarray = field(init=False)
    hist_np_n: np.ndarray = field(init=False)  # [g, bin] by aN, components with aN > 0
    hist_np_sum: np.ndarray = field(init=False)
    cross_a_n: np.ndarray = field(init=False)  # [g, θ] components first reaching a ≥ θ at g
    cross_a_sum: np.ndarray = field(init=False)
    cross_n_n: np.ndarray = field(init=False)  # [g, θ] first reaching aN ≥ θ at g
    cross_n_sum: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        g = self.n_grid
        for name in (
            "D",
            "A_cum",
            "A_med",
            "AN_cum",
            "AN_med",
            "AN_min",
            "AN_max",
            "AF_cum",
            "AF_med",
            "P_cum",
            "P_med",
            "created",
        ):
            setattr(self, name, np.zeros(g))
        for name in ("hist_n", "hist_sum", "hist_p_n", "hist_p_sum", "hist_np_n", "hist_np_sum"):
            setattr(self, name, np.zeros((g, N_BINS)))
        for name in ("cross_a_n", "cross_a_sum"):
            setattr(self, name, np.zeros((g, THETA_A.size)))
        for name in ("cross_n_n", "cross_n_sum"):
            setattr(self, name, np.zeros((g, THETA_N.size)))

    def arrays(self, prefix: str = "") -> dict[str, np.ndarray]:
        return {prefix + k: v for k, v in vars(self).items() if isinstance(v, np.ndarray)}

    def fill(self, g: int, snaps: Sequence[Snapshot]) -> None:
        """Row ``g`` from the union of ``snaps`` (one space, or the cohorts of a pool)."""
        self.created[g] = sum(sn.created for sn in snaps)
        snaps = [sn for sn in snaps if sn.a.size]
        if not snaps:
            return
        a = np.concatenate([sn.a for sn in snaps])
        an = np.concatenate([sn.an for sn in snaps])
        af = np.concatenate([sn.af for sn in snaps])
        p = np.concatenate([sn.p for sn in snaps])
        new_a = np.concatenate([sn.new_a for sn in snaps])
        new_n = np.concatenate([sn.new_n for sn in snaps])
        self.D[g] = a.size
        self.A_cum[g], self.A_med[g] = a.sum(), np.median(a)
        self.AN_cum[g], self.AN_med[g] = an.sum(), np.median(an)
        self.AN_min[g], self.AN_max[g] = an.min(), an.max()
        self.AF_cum[g], self.AF_med[g] = af.sum(), np.median(af)
        self.P_cum[g], self.P_med[g] = p.sum(), np.median(p)
        ba, bp = _bins(a), _bins(p)
        self.hist_n[g] = np.bincount(ba, minlength=N_BINS)
        self.hist_sum[g] = np.bincount(ba, weights=a, minlength=N_BINS)
        self.hist_p_n[g] = np.bincount(bp, minlength=N_BINS)
        self.hist_p_sum[g] = np.bincount(bp, weights=p, minlength=N_BINS)
        pos = an[an > 0]
        bn = _bins(pos)
        self.hist_np_n[g] = np.bincount(bn, minlength=N_BINS)
        self.hist_np_sum[g] = np.bincount(bn, weights=pos, minlength=N_BINS)
        self.cross_a_n[g] = new_a.sum(axis=0)
        self.cross_a_sum[g] = (new_a * a[:, None]).sum(axis=0)
        self.cross_n_n[g] = new_n.sum(axis=0)
        self.cross_n_sum[g] = (new_n * an[:, None]).sum(axis=0)


class Space:
    """Cumulative activity of ``C`` components; ``snapshot`` reads the present ones and advances the running
    maxima used for the threshold crossings."""

    def __init__(self, n_comp: int, created_at: np.ndarray) -> None:
        self.a = np.zeros(n_comp, np.int64)
        self.e = np.zeros(n_comp, np.float64)  # expected births, reset shadow (grid interval)
        self.f = np.zeros(n_comp, np.float64)  # expected births, exposure (reset every step)
        self.max_a = np.zeros(n_comp, np.int64)  # running maxima at the previous snapshot
        self.max_n = np.full(n_comp, -np.inf)
        self.seen = np.zeros(n_comp, bool)
        self.created_at = created_at  # the time each component starts (presence activity)

    def add(self, births: np.ndarray, expected: np.ndarray, exposure: np.ndarray) -> None:
        self.a += births
        self.e += expected
        self.f += exposure

    def snapshot(self, t: int, present: np.ndarray) -> Snapshot:
        a = self.a[present]
        an = a - self.e[present]
        prev_a, prev_n = self.max_a[present], self.max_n[present]
        new_a = (prev_a[:, None] < THETA_A[None, :]) & (a[:, None] >= THETA_A[None, :])
        new_n = (prev_n[:, None] < THETA_N[None, :]) & (an[:, None] >= THETA_N[None, :])
        self.max_a[present] = np.maximum(prev_a, a)
        self.max_n[present] = np.maximum(prev_n, an)
        created = int(np.count_nonzero(~self.seen[present]))
        self.seen[present] = True
        return Snapshot(
            a=a,
            an=an,
            af=a - self.f[present],
            p=(t - self.created_at[present]).astype(np.float64),
            new_a=new_a,
            new_n=new_n,
            created=created,
        )


COHORT_FIELDS = ("D", "A_cum", "A_med", "AN_cum", "AN_med", "AF_cum", "P_cum")


def _cohort_row(sn: Snapshot) -> dict[str, float]:
    if sn.a.size == 0:
        return dict.fromkeys(COHORT_FIELDS, 0.0)
    return {
        "D": sn.a.size,
        "A_cum": sn.a.sum(),
        "A_med": np.median(sn.a),
        "AN_cum": sn.an.sum(),
        "AN_med": np.median(sn.an),
        "AF_cum": sn.af.sum(),
        "P_cum": sn.p.sum(),
    }


# --- the window pass ---------------------------------------------------------------------------------------


def _window_ancestors(parent: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """For agents ``lo … hi−1``: the nearest ancestor with index < ``lo`` (alive at the window start), or −1
    if the chain reaches a root born inside the window (an immigrant) or the agent is such a root."""
    ptr = parent[lo:hi].astype(np.int64)
    active = np.flatnonzero(ptr >= lo)
    while active.size:
        ptr[active] = ptr[ptr[active] - lo]
        active = active[ptr[active] >= lo]
    return ptr


@dataclass
class Result:
    """The statistics of one tree.

    ``cohorts[f]`` is ``[k, g]``: field ``f`` (``COHORT_FIELDS``) of cohort ``k`` (rooted at ``grid[k]``) at
    grid time ``grid[g]`` (0 before the cohort starts). ``cohort_pooled[H]`` pools, at each grid time ``t``,
    the cohorts of age ``0 < t − τ_k ≤ H``: a sliding window of cohorts, so its statistics stay stationary
    when each cohort's are (pooling every cohort would grow ``D`` and the totals with the number of cohorts by
    construction).
    """

    grid: np.ndarray
    n_alive: np.ndarray
    births: np.ndarray  # non-root births per window (row g: window ending at grid[g])
    cohorts: dict[str, np.ndarray]
    cohort_pooled: dict[int, Series]
    marker: dict[float, Series]


def analyse(
    tree: Tree,
    grid_every: int,
    marker_rates: Sequence[float] = (),
    cohort_horizons: Sequence[int] = (),
    *,
    marker_seed: int = 0,
    cohorts: bool = True,
    gap: int | None = None,
) -> Result:
    """Every statistic of the module docstring for one tree (the run or a shadow)."""
    gap = min_gap(tree) if gap is None else gap
    grid = grid_times(tree.t_end, grid_every)
    n_grid = grid.size
    cdf = exposure_cdf(tree, gap)
    lo_of = np.searchsorted(tree.birth, grid, side="left")  # first agent born at or after grid[g]
    parent, birth, death = tree.parent, tree.birth, tree.death

    # Marker spaces: global labels, compact component ids, creation times.
    marker: dict[float, Series] = {}
    mspaces: dict[float, tuple[np.ndarray, Space]] = {}
    for rate in marker_rates:
        roots = marker_roots(tree, rate, marker_seed)
        rank = np.cumsum(roots, dtype=np.int64) - 1
        label = rank[nearest_root(parent, roots)].astype(np.int32)
        created = birth[roots].astype(np.int64) + 1
        marker[rate] = Series(n_grid)
        mspaces[rate] = (label, Space(int(roots.sum()), created))

    alive = np.flatnonzero(birth < 0).astype(np.int64)  # alive at grid[0] = 0: the founders
    rows = np.arange(alive.size, dtype=np.int32)[:, None]  # [agent of `alive`, cohort k] → component
    cohort_spaces: list[Space] = []
    n_cohorts = n_grid - 1 if cohorts else 0
    tracks = {f: np.zeros((n_cohorts, n_grid)) for f in COHORT_FIELDS}
    pooled = {h: Series(n_grid) for h in cohort_horizons} if cohorts else {}
    n_alive = np.zeros(n_grid)
    births = np.zeros(n_grid)
    n_alive[0] = alive.size

    def new_cohort(g: int) -> None:
        sp = Space(alive.size, np.full(alive.size, grid[g], np.int64))
        cohort_spaces.append(sp)
        for f, v in _cohort_row(sp.snapshot(int(grid[g]), np.arange(alive.size))).items():
            tracks[f][g, g] = v

    if cohorts:
        new_cohort(0)
    for rate, (label, sp) in mspaces.items():
        marker[rate].fill(0, [sp.snapshot(0, np.unique(label[alive]))])

    for j in range(n_grid - 1):
        t0, t1 = int(grid[j]), int(grid[j + 1])
        lo, hi = int(lo_of[j]), int(lo_of[j + 1])
        born = np.arange(lo, hi, dtype=np.int64)
        kid = parent[lo:hi] >= 0
        births[j + 1] = np.count_nonzero(kid)
        expo_alive = _exposure(tree, alive, gap, cdf, t0, t1)
        expo_born = _exposure(tree, born, gap, cdf, t0, t1)
        surv = death[alive] >= t1
        born_surv = death[lo:hi] >= t1
        next_alive = np.concatenate([alive[surv], born[born_surv]])
        n_alive[j + 1] = next_alive.size

        share = births[j + 1] / alive.size if alive.size else 0.0  # reset shadow: B_j · n_i / N
        for rate, (label, sp) in mspaces.items():
            c = sp.a.size
            par = parent[lo:hi][kid]
            sp.add(
                np.bincount(label[par], minlength=c),
                share * np.bincount(label[alive], minlength=c),
                np.bincount(label[alive], weights=expo_alive, minlength=c)
                + np.bincount(label[born], weights=expo_born, minlength=c),
            )
            marker[rate].fill(j + 1, [sp.snapshot(t1, np.unique(label[next_alive]))])

        if cohorts:
            anc = _window_ancestors(parent, lo, hi)  # −1: immigrant lineage born in this window
            pos = np.searchsorted(alive, np.maximum(anc, 0))
            in_r = anc >= 0
            n_r = alive.size
            b_r = np.bincount(pos[in_r & kid], minlength=n_r)
            e_r = expo_alive + np.bincount(pos[in_r], weights=expo_born[in_r], minlength=n_r)
            born_rows = np.full((hi - lo, rows.shape[1]), -1, np.int32)
            born_rows[in_r] = rows[pos[in_r]]
            next_rows = np.concatenate([rows[surv], born_rows[born_surv]])
            snaps: list[Snapshot] = []
            for k, sp in enumerate(cohort_spaces):
                col = rows[:, k]
                m = col >= 0
                ck = sp.a.size
                sp.add(
                    np.bincount(col[m], weights=b_r[m], minlength=ck).astype(np.int64),
                    share * np.bincount(col[m], minlength=ck),
                    np.bincount(col[m], weights=e_r[m], minlength=ck),
                )
                nxt = next_rows[:, k]
                sn = sp.snapshot(t1, np.flatnonzero(np.bincount(nxt[nxt >= 0], minlength=ck)))
                snaps.append(sn)
                for f, v in _cohort_row(sn).items():
                    tracks[f][k, j + 1] = v
            for h, series in pooled.items():
                series.fill(j + 1, [sn for k, sn in enumerate(snaps) if 0 < t1 - grid[k] <= h])
            alive = next_alive
            rows = np.concatenate([next_rows, np.arange(alive.size, dtype=np.int32)[:, None]], axis=1)
            if j + 1 < n_grid - 1:
                new_cohort(j + 1)
            else:
                rows = rows[:, :-1]
        else:
            alive = next_alive

    return Result(
        grid=grid, n_alive=n_alive, births=births, cohorts=tracks, cohort_pooled=pooled, marker=marker
    )


# --- derived statistics ------------------------------------------------------------------------------------


def a_new(
    hist_n: np.ndarray, hist_sum: np.ndarray, d: np.ndarray, a0: float, a1: float, *, count: bool = False
) -> np.ndarray:
    """Bedau 1998 eq. 7, ``(1/D) Σ_{a0 ≤ a ≤ a1} a`` per grid time, from the ``[g, bin]`` histograms of a
    ``Series`` (``hist_n``/``hist_sum``, or the presence ones). ``a0`` and ``a1`` snap to bin edges: the bins
    whose lower edge lies in ``[a0, a1]``. ``count`` gives the eq. 8 reading (components in the window
    over ``D``)."""
    sel = (a0 * (1 - 1e-9) <= A_EDGES) & (a1 * (1 + 1e-9) >= A_EDGES)
    num = (hist_n if count else hist_sum)[:, sel].sum(axis=1)
    return np.divide(num, d, out=np.zeros_like(num, dtype=np.float64), where=d > 0)


def crossing_threshold(real_hist_n: np.ndarray, shadow_hist_n: np.ndarray) -> tuple[float, float, float]:
    """Bedau 1998's new-activity window from the time-collapsed activity distributions.

    Each distribution (``[g, bin]`` counts summed over grid times) is divided by the total count of both.
    ``a'`` is the lower edge of the lowest bin (above bin 0) where the curves cross, i.e. where the real
    density reaches the shadow's after being below it in the previous bin; if they never cross that way,
    the lowest bin where the real density is at least the shadow's. ``a_max`` is the highest bin where either
    is positive, and ``[a0, a1] = a' ± 0.05 (a_max − a')`` (Bedau, Snyder & Packard 1998, Fig. 1). Returns
    ``(a0, a1, a')``, all infinite if no bin qualifies.
    """
    rn, sn = real_hist_n.sum(axis=0), shadow_hist_n.sum(axis=0)
    tot = rn.sum() + sn.sum()
    if tot == 0:
        return (np.inf, np.inf, np.inf)
    rd, sd = rn / tot, sn / tot
    pos = np.flatnonzero((rd > 0) | (sd > 0))
    a_max = A_EDGES[pos[-1]]
    i = np.arange(N_BINS)
    above = (rd >= sd) & (rd > 0)
    below_prev = np.concatenate([[False], rd[:-1] < sd[:-1]])
    cross = np.flatnonzero((i > 0) & above & below_prev)
    if cross.size == 0:
        cross = np.flatnonzero((i > 0) & above)
    if cross.size == 0:
        return (np.inf, np.inf, np.inf)
    a_p = A_EDGES[cross[0]]
    w = 0.05 * (a_max - a_p)
    return (max(a_p - w, 0.0), a_p + w, a_p)


def shadow_theta(shadow_hist_np_sum: np.ndarray, q: float) -> float:
    """The smallest crossing threshold of ``THETA_N`` above which at most a fraction ``q`` of the shadow's
    positive normalised activity lies (time-collapsed), the analogue of Channon 2001's window placed above
    nearly all shadow activity. A shadow without positive activity gives the smallest θ; ``inf`` means no θ
    of the sweep leaves at most ``q`` above it."""
    mass = shadow_hist_np_sum.sum(axis=0)
    tot = mass.sum()
    if tot <= 0:
        return float(THETA_N[0])
    above = np.cumsum(mass[::-1])[::-1] / tot  # fraction in bins with lower edge ≥ A_EDGES[i]
    for th in THETA_N:
        i = int(np.searchsorted(A_EDGES, th * (1 - 1e-9)))
        if i >= N_BINS or above[i] <= q:
            return float(th)
    return float("inf")
