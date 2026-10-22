"""The W2 results pipeline: every pre-registered statistic of docs/STORMS.md W2 from run directories (DESIGN
§32).

Usage::

    python -m alife.analysis.w2 [ARM@SEED=]RUN... --out FILE.json [--start STEP]

``RUN`` is a run directory (read only); an arm and a seed may be given as ``A@0=runs/a_s0``, else the arm is
read from the run's config (``infer_arm``) and the seed from the fork's parent name (``..._s<k>``) or
``run.seed``. The arms are A (the storm world), N (no seeding), S (still air, zonal rain), WB (wind-blind: no
wind force, live rain), A1 (coupling independent of the cap) and E (eddy-only wind force); J and A2 need an
explicit label. **The definitions, windows and thresholds here are fixed by the W2 spec (§8) and by
``tests/test_analysis_w2.py`` on synthetic runs before any W2b run is read.**

**Intervals.** The post-transient steps (the first ``TRANSIENT`` = 20,000 after the fork are dropped) are cut
into ``N_BLOCKS`` = 8 blocks of ``BLOCK`` = 10,000 steps; a statistic is computed per block and its interval
is the t-interval of the block values (7 df, 95%). A contrast between two runs uses the paired block
differences (the arms share the fork and the weather clock, so block ``b`` of both covers the same steps). A
16 x 5,000 version is reported beside each as a sensitivity check (``sens``). A claim is *supported* if both
seeds' intervals (s0 and s1) exclude 0 with the same sign, *mixed* if exactly one does or they disagree, and
*not supported* otherwise. For the agent statistics (``hold``, ``ub``, the selection gradients, the cap trend,
the biosphere), a contrast also has to exceed in magnitude the **noise floor**: the magnitude of the s1 A - N
contrast of the same statistic (seeding is near-inert on seed 1, so that contrast is the run-to-run spread of
two runs from one fork). Weather totals reproduce to under 0.5%, so for them the thresholds carry the test.

**Statistics.** P1a: ``log(rain_total_A / rain_total_N)``, early (steps 1-3,000 after the fork) and late
(post-transient); P1b: per ``wx`` dump the slope of ``P_A - P_N`` on A's seeding strength (plane 9), block
t-interval over dumps, and the cell-wise correlation of ``P_A`` and ``P_N``; P1c: the emission above the free
level (``emit - 12``) by z band and clade, and the share of agents paying; P2: ``ub_agents_all`` of the
``coupling`` blocks per arm and its contrasts A - WB (wind) and WB - S (rain pattern), with the thresholds
0.12 and 0.02; P3: ``hold`` and A - WB with the threshold 0.05, and the 250-step ``|dz|`` of agents in drafts
against tracers; P4: the standardised selection gradient of the buoyancy cap on births and on 250-step
survival per block, the contrasts A - WB, A - A1, A - E and WB - S, the cap trend and the mean ``alpha``; P5:
the biosphere (population, speed, births, deaths, storm deaths and their share, mid-column occupancy) with the
contrasts A - WB, A - E, E - WB and WB - S, and the storm deaths by cause, z band and clade; P6: the safety
figures against the limits of §4.1. Each is reported only if the data it needs are in the run (``coupling``
blocks need ``game.tracers``; the dump statistics need ``run.eco_every`` or ``run.wx_every``).

**Approximations (stated).** P1c does not read the c2 field (only checkpoints hold it): it reports the paid
share of the emission, the quantity that decides "crowds make rain" against "genes make rain". A storm death's
z band is that of the agent at the last ecology dump before its death (agents born since are unattributed).
``WEATHER_WARN`` is a sticky ``diag`` bit; P6 reports the share of rows carrying it and, separately, the share
of rows whose ``hyper_max`` is at the health gate (0.8), which is the rate a stop rule should read.

Host numpy only; reads the run directory, never writes into it.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

from alife.config import Config
from alife.runio import RunDir
from alife.state import ERROR_MASK, DeathCause, Diag
from alife.weather_obs import z_band

__all__ = [
    "BLOCK",
    "N_BLOCKS",
    "TRANSIENT",
    "Blocking",
    "Run",
    "analyse",
    "classify",
    "infer_arm",
    "load_run",
    "paired",
    "selection_gradient",
    "t_interval",
]

TRANSIENT: Final = 20_000  # steps after the fork that no statistic reads
BLOCK: Final = 10_000
N_BLOCKS: Final = 8
EARLY: Final = 3_000  # the early window of P1a, steps after the fork
FREE_EMIT: Final = 12  # the deposit (quanta of 1/256) at the free-emission limit, Σ emit = 0.05
ECO_WINDOW: Final = 250  # the survival and persistence window, the eco dump cadence of the arms
DRAFT_W: Final = 0.002  # |w| from which an agent is "in a draft"
UB_MIN: Final = 0.12  # P2: ub_A at least
UB_DIFF: Final = 0.02  # P2: ub_A - ub_WB at least
HOLD_DIFF: Final = 0.05  # P3: hold_A - hold_WB at least
HYPER_GATE: Final = 0.8  # the weather layer's hyperbolicity gate (weather._HYPER_GATE)
SEEDS: Final = (0, 1)  # the seeds a claim is read on
ARMS: Final = ("A", "N", "S", "WB", "A1", "E")

# 0.975 quantiles of Student's t for 1..30 df
_T975: Final = (
    12.706,
    4.303,
    3.182,
    2.776,
    2.571,
    2.447,
    2.365,
    2.306,
    2.262,
    2.228,
    2.201,
    2.179,
    2.160,
    2.145,
    2.131,
    2.120,
    2.110,
    2.101,
    2.093,
    2.086,
    2.080,
    2.074,
    2.069,
    2.064,
    2.060,
    2.056,
    2.052,
    2.048,
    2.045,
    2.042,
)


# --- intervals, blocks and claims ---


def t_crit(df: int) -> float:
    """The two-sided 95% critical value of Student's t with ``df`` degrees of freedom (1.96 above 30)."""
    return _T975[df - 1] if 1 <= df <= len(_T975) else 1.96


def t_interval(values: Sequence[float] | np.ndarray) -> dict[str, Any]:
    """Mean and 95% t-interval of the finite values: ``n``, ``mean``, ``sd`` (ddof 1), ``lo``, ``hi``. The
    interval is None for fewer than two values."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    out: dict[str, Any] = {"n": int(v.size), "mean": float(v.mean()) if v.size else None, "sd": None}
    out.update(lo=None, hi=None)
    if v.size >= 2:
        sd = float(v.std(ddof=1))
        half = t_crit(v.size - 1) * sd / math.sqrt(v.size)
        out.update(sd=sd, lo=float(v.mean()) - half, hi=float(v.mean()) + half)
    return out


def paired(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The paired block differences ``a - b`` (NaN wherever either block is empty)."""
    return np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)


def _sign(iv: Mapping[str, Any] | None, floor: float | None) -> int:
    """+1 / -1 if the interval excludes 0 on that side (and, with a ``floor``, the mean's magnitude exceeds
    it), else 0."""
    if iv is None or iv.get("lo") is None:
        return 0
    s = 1 if iv["lo"] > 0 else -1 if iv["hi"] < 0 else 0
    return s if s and (floor is None or abs(iv["mean"]) > floor) else 0


def classify(ivs: Sequence[Mapping[str, Any] | None], floor: float | None = None) -> str:
    """``supported`` if every interval (one per seed) excludes 0 with the same sign, ``mixed`` if some do or
    they disagree, ``not supported`` otherwise. With a ``floor`` the mean's magnitude must also exceed it."""
    signs = [_sign(iv, floor) for iv in ivs]
    if signs and all(s != 0 for s in signs) and len(set(signs)) == 1 and len(signs) >= 2:
        return "supported"
    return "mixed" if any(signs) else "not supported"


@dataclass(frozen=True)
class Blocking:
    """The post-transient blocks of a run that forked at step ``start``: block ``b`` holds the steps in
    ``(start + transient + b·width, start + transient + (b+1)·width]``."""

    start: int
    transient: int
    width: int
    n: int

    @property
    def lo(self) -> int:
        return self.start + self.transient

    def index(self, steps: np.ndarray) -> np.ndarray:
        """The block of each step, -1 outside the blocks."""
        b = (np.asarray(steps, dtype=np.int64) - self.lo - 1) // self.width
        return np.where((np.asarray(steps) > self.lo) & (b < self.n), b, -1)

    def sensitivity(self) -> Blocking:
        """The same steps in twice as many blocks of half the width (16 x 5,000)."""
        return Blocking(self.start, self.transient, self.width // 2, self.n * 2)


def block_mean(blk: Blocking, steps: np.ndarray, values: np.ndarray) -> np.ndarray:
    """The mean of ``values`` per block (NaN for a block with no finite value)."""
    steps, v = np.asarray(steps), np.asarray(values, dtype=np.float64)
    ok = np.isfinite(v)
    idx = blk.index(steps)
    out = np.full(blk.n, np.nan)
    for b in range(blk.n):
        m = ok & (idx == b)
        if m.any():
            out[b] = v[m].mean()
    return out


def block_ratio(blk: Blocking, steps: np.ndarray, num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """``Σ num / Σ den`` per block (NaN where ``Σ den`` is 0)."""
    idx = blk.index(steps)
    out = np.full(blk.n, np.nan)
    for b in range(blk.n):
        m = idx == b
        d = float(np.asarray(den, dtype=np.float64)[m].sum())
        if m.any() and d != 0:
            out[b] = float(np.asarray(num, dtype=np.float64)[m].sum()) / d
    return out


# --- runs ---


def infer_arm(cfg: Config) -> str:
    """The arm of a run from its config: S (``wind_scale 0`` with zonal rain), WB (``wind_scale 0``), E (the
    eddy-only force), A1 (``alpha1 0``), N (no seeding), else A. J and A2 are not inferable."""
    w = cfg.weather
    if w.wind_scale == 0.0:
        return "S" if w.rain_zonal else "WB"
    if w.wind_eddy_force:
        return "E"
    if w.alpha1 == 0.0:
        return "A1"
    return "N" if w.seed_gain == 0.0 else "A"


@dataclass
class Run:
    """One run directory read for the analysis."""

    arm: str
    seed: int
    path: Path
    cfg: Config
    rows: list[dict[str, Any]]
    start: int  # the step the run forked at

    @property
    def rd(self) -> RunDir:
        return RunDir.open(self.path)  # cheap: the directory is read lazily

    def blocking(self) -> Blocking:
        """The blocks of the run (the module constants as of this call)."""
        return Blocking(self.start, TRANSIENT, BLOCK, N_BLOCKS)

    def steps(self) -> np.ndarray:
        return np.array([r["step"] for r in self.rows], dtype=np.int64)

    def series(self, key: str) -> np.ndarray:
        """The metrics rows' value at the dotted ``key`` (``weather.rain_total``; NaN where absent)."""
        return np.array([_get(r, key) for r in self.rows], dtype=np.float64)

    def eco(self) -> Iterator[tuple[int, dict[str, np.ndarray]]]:
        """``(step, columns)`` of every ecology dump, ascending (one in memory at a time)."""
        rd = self.rd
        for st, _ in rd.dump_files("eco"):
            yield st, rd.read_eco(st)

    def wx_steps(self) -> list[int]:
        return [st for st, _ in self.rd.dump_files("wx")]


def _get(row: Mapping[str, Any], key: str) -> float:
    """The number at a dotted key of a row, NaN if absent or not a number."""
    cur: Any = row
    for part in key.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return math.nan
        cur = cur[part]
    return float(cur) if isinstance(cur, int | float) and not isinstance(cur, bool) else math.nan


def _parse_spec(text: str) -> tuple[str | None, int | None, Path]:
    m = re.fullmatch(r"(?:(?P<arm>[A-Za-z0-9]+)@(?P<seed>\d+)=)?(?P<path>.+)", text)
    assert m is not None
    return m["arm"], (int(m["seed"]) if m["seed"] is not None else None), Path(m["path"])


def load_run(spec: str | Path, *, start: int | None = None) -> Run:
    """Read a run directory. ``spec`` is ``[ARM@SEED=]PATH``. ``start`` defaults to the first row's step minus
    its interval (the step the run forked at)."""
    arm, seed, path = _parse_spec(str(spec))
    rd = RunDir.open(path)
    cfg = rd.config()
    rows = rd.metrics_rows()
    if not rows:
        raise ValueError(f"{path}: no metrics rows")
    if start is None:
        start = int(rows[0]["step"]) - int(rows[0].get("steps", 0))
    if seed is None:
        parent = str(rd.meta().get("parent") or "")
        m = re.findall(r"_s(\d+)", parent)
        seed = int(m[-1]) if m else cfg.run.seed
    return Run(arm or infer_arm(cfg), seed, path, cfg, rows, start)


# --- statistics: a spec maps a run and a blocking to the per-block values ---

Spec = Callable[[Run, Blocking], np.ndarray]


def mean_of(key: str) -> Spec:
    """The block mean of a metrics row value."""
    return lambda run, blk: block_mean(blk, run.steps(), run.series(key))


def rate_of(*keys: str) -> Spec:
    """Events per step: ``Σ keys / Σ steps`` over the rows of a block."""

    def f(run: Run, blk: Blocking) -> np.ndarray:
        num = np.nansum([run.series(k) for k in keys], axis=0)
        return block_ratio(blk, run.steps(), num, run.series("steps"))

    return f


def share_of(num: Sequence[str], den: Sequence[str]) -> Spec:
    """``Σ num / Σ den`` over the rows of a block."""

    def f(run: Run, blk: Blocking) -> np.ndarray:
        return block_ratio(
            blk,
            run.steps(),
            np.nansum([run.series(k) for k in num], axis=0),
            np.nansum([run.series(k) for k in den], axis=0),
        )

    return f


def summarise(spec: Spec, run: Run) -> dict[str, Any]:
    """The per-block values, the t-interval and the 16 x 5,000 sensitivity interval of a run's statistic."""
    blk = run.blocking()
    vals = spec(run, blk)
    return {"blocks": vals.tolist(), **t_interval(vals), "sens": t_interval(spec(run, blk.sensitivity()))}


def contrast(spec: Spec, a: Run, b: Run) -> dict[str, Any]:
    """The paired block differences ``a - b`` of a statistic: values, interval and sensitivity interval."""
    blk = a.blocking()
    d = paired(spec(a, blk), spec(b, b.blocking()))
    ds = paired(spec(a, blk.sensitivity()), spec(b, b.blocking().sensitivity()))
    return {"blocks": d.tolist(), **t_interval(d), "sens": t_interval(ds)}


class Runs:
    """The runs of an analysis by ``(arm, seed)``."""

    def __init__(self, runs: Sequence[Run]) -> None:
        self.by = {(r.arm, r.seed): r for r in runs}

    def get(self, arm: str, seed: int) -> Run | None:
        return self.by.get((arm, seed))

    def seeds(self, arm: str) -> list[int]:
        return sorted(s for a, s in self.by if a == arm)

    def has(self, arm: str, *more: str, seed: int) -> bool:
        return all((x, seed) in self.by for x in (arm, *more))


def noise_floor(runs: Runs, spec: Spec) -> float | None:
    """The magnitude of the s1 A - N contrast of ``spec`` (None if either run is missing): the run-to-run
    spread of two runs from one fork, since seeding is near-inert on seed 1."""
    a, n = runs.get("A", 1), runs.get("N", 1)
    if a is None or n is None:
        return None
    m = contrast(spec, a, n)["mean"]
    return None if m is None else abs(m)


def compare(
    runs: Runs, spec: Spec, a: str, b: str, *, floor: bool, seeds: Sequence[int] = SEEDS
) -> dict[str, Any]:
    """The contrast ``a - b`` of a statistic on each seed that has both arms, and its claim over ``seeds``."""
    per = {
        str(s): contrast(spec, ra, rb)
        for s in runs.seeds(a)
        if (ra := runs.get(a, s)) is not None and (rb := runs.get(b, s)) is not None
    }
    fl = noise_floor(runs, spec) if floor else None
    ivs = [per.get(str(s)) for s in seeds]
    return {
        "contrast": f"{a} - {b}",
        "seeds": per,
        "noise_floor": fl,
        "claim": classify(ivs, fl),
        "claim_unfloored": classify(ivs, None) if floor else None,
    }


def per_arm(runs: Runs, spec: Spec, arms: Sequence[str]) -> dict[str, Any]:
    """``summarise`` for every ``arm`` / seed present: ``{arm: {seed: summary}}``."""
    return {
        arm: {str(s): summarise(spec, r) for s in runs.seeds(arm) if (r := runs.get(arm, s)) is not None}
        for arm in arms
        if runs.seeds(arm)
    }


# --- the dump statistics: series of (step, value) turned into block values ---


def series_spec(fn: Callable[[Run], tuple[np.ndarray, np.ndarray]]) -> Spec:
    """A statistic given as ``(steps, values)`` of a run (one per dump), block-averaged. The series of a run
    is read from its dumps once and kept: the statistic is evaluated for every contrast and both layouts."""
    cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def f(run: Run, blk: Blocking) -> np.ndarray:
        if id(run) not in cache:
            cache.update({id(run): fn(run)})
        steps, vals = cache[id(run)]
        return block_mean(blk, steps, vals)

    return f


def selection_gradient(trait: np.ndarray, fitness: np.ndarray) -> float | None:
    """The standardised selection gradient ``Cov(w/w̄, (x - x̄)/sd(x))`` of a trait ``x`` on a fitness ``w``
    (population moments). None for fewer than two rows, a constant trait or a zero mean fitness."""
    x, w = np.asarray(trait, dtype=np.float64), np.asarray(fitness, dtype=np.float64)
    if x.size < 2 or not x.std() > 0 or not w.mean() > 0:
        return None
    return float(np.mean((w / w.mean()) * (x - x.mean())) / x.std())


def _eco_pairs(run: Run) -> Iterator[tuple[int, dict[str, np.ndarray], dict[str, np.ndarray]]]:
    """``(t, dump at t, dump at t + ECO_WINDOW)`` for consecutive ecology dumps one window apart."""
    prev: tuple[int, dict[str, np.ndarray]] | None = None
    for st, cols in run.eco():
        if prev is not None and st - prev[0] == ECO_WINDOW:
            yield prev[0], prev[1], cols
        prev = (st, cols)


def births_per_parent(parent_uid: np.ndarray, uids: np.ndarray) -> np.ndarray:
    """For each uid, how many of the ``parent_uid`` entries equal it."""
    if parent_uid.size == 0:
        return np.zeros(uids.size, dtype=np.int64)
    u, c = np.unique(parent_uid, return_counts=True)
    pos = np.minimum(np.searchsorted(u, uids), u.size - 1)
    return np.where(u[pos] == uids, c[pos], 0)


def match_uid(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each entry of ``a`` the index of the equal entry of ``b`` and whether there is one (``b`` holds
    distinct values)."""
    if b.size == 0:
        return np.zeros(a.size, dtype=np.int64), np.zeros(a.size, dtype=bool)
    order = np.argsort(b)
    j = order[np.minimum(np.searchsorted(b, a, sorter=order), b.size - 1)]
    return j, b[j] == a


def gradient_series(run: Run, what: str) -> tuple[np.ndarray, np.ndarray]:
    """``(t, β_cap)`` per ecology-dump pair: the selection gradient of ``cap`` on ``births`` (children born in
    the next window, by parent uid, from the birth log) or on ``survival`` (alive at the next dump)."""
    births = run.rd.read_events("births") if what == "births" else None
    steps, vals = [], []
    for t, a, b in _eco_pairs(run):
        if what == "survival":
            w = np.isin(a["uid"], b["uid"]).astype(np.float64)
        else:
            assert births is not None
            m = (births["step"] > t) & (births["step"] <= t + ECO_WINDOW)
            w = births_per_parent(births["parent_uid"][m], a["uid"]).astype(np.float64)
        g = selection_gradient(a["cap"].astype(np.float64), w)
        steps.append(t + ECO_WINDOW)
        vals.append(math.nan if g is None else g)
    return np.array(steps, dtype=np.int64), np.array(vals)


def persistence_series(run: Run, who: str) -> tuple[np.ndarray, np.ndarray]:
    """``(t, mean |dz| over ECO_WINDOW steps)`` per dump pair, for the agents in a draft at the first dump
    (``|w_air| >= DRAFT_W``, matched by uid) or for the tracers (those not reseeded in between)."""
    steps, vals = [], []
    for t, a, b in _eco_pairs(run):
        if who == "agents":
            j, hit = match_uid(a["uid"], b["uid"])
            m = hit & (np.abs(a["w_air"].astype(np.float64)) >= DRAFT_W)
            dz = np.abs(b["z"][j].astype(np.float64) - a["z"].astype(np.float64))[m]
        else:
            if "tracer_z" not in a or "tracer_z" not in b:
                continue
            if not (b["tracer_age"] - a["tracer_age"] == ECO_WINDOW).all():
                continue  # the tracers were reseeded in between
            dz = np.abs(b["tracer_z"].astype(np.float64) - a["tracer_z"].astype(np.float64))
        steps.append(t + ECO_WINDOW)
        vals.append(float(dz.mean()) if dz.size else math.nan)
    return np.array(steps, dtype=np.int64), np.array(vals)


def cap_series(run: Run, how: str) -> tuple[np.ndarray, np.ndarray]:
    """``(step, value)`` per ecology dump of the live agents' cap: ``median``, ``q90`` or ``alpha`` (the mean
    coupling ``alpha0 + alpha1·mean cap``)."""
    w = run.cfg.weather
    steps, vals = [], []
    for st, cols in run.eco():
        cap = cols["cap"].astype(np.float64)
        steps.append(st)
        if not cap.size:
            vals.append(math.nan)
        elif how == "median":
            vals.append(float(np.median(cap)))
        elif how == "q90":
            vals.append(float(np.quantile(cap, 0.9)))
        else:
            vals.append(w.alpha0 + w.alpha1 * float(cap.mean()))
    return np.array(steps, dtype=np.int64), np.array(vals)


def zshare_series(run: Run, band: int) -> tuple[np.ndarray, np.ndarray]:
    """``(step, share of live agents in z band 'band')`` per ecology dump (band 2: z 32-56, the clouds)."""
    height = run.cfg.world.height
    steps, vals = [], []
    for st, cols in run.eco():
        z = cols["z"].astype(np.float64)
        steps.append(st)
        vals.append(float((z_band(z, height) == band).mean()) if z.size else math.nan)
    return np.array(steps, dtype=np.int64), np.array(vals)


def cap_trend(run: Run) -> dict[str, Any]:
    """The least-squares slope per 10,000 steps of the median and the q90 of the cap over the post-transient
    dumps, and the change from the first to the last block."""
    out: dict[str, Any] = {}
    blk = run.blocking()
    for how in ("median", "q90", "alpha"):
        st, v = cap_series(run, how)
        ok = (blk.index(st) >= 0) & np.isfinite(v)
        trend = float(np.polyfit(st[ok].astype(np.float64), v[ok], 1)[0]) * BLOCK if ok.sum() >= 3 else None
        bv = block_mean(blk, st, v)
        fin = bv[np.isfinite(bv)]
        out[how] = {
            "slope_per_10k": trend,
            "first_block": float(fin[0]) if fin.size else None,
            "last_block": float(fin[-1]) if fin.size else None,
        }
    return out


# --- the predictions ---


def p1a(runs: Runs) -> dict[str, Any]:
    """Seeding raises rain (a manipulation check of an imposed coupling): ``log(rain_total_A/rain_total_N)``,
    early (steps 1-3,000 after the fork) and late (post-transient), per seed that has both arms."""
    out: dict[str, Any] = {}
    for s in runs.seeds("A"):
        a, n = runs.get("A", s), runs.get("N", s)
        if a is None or n is None:
            continue
        ra, rn = a.series("weather.rain_total"), n.series("weather.rain_total")

        def window(r: Run, v: np.ndarray, lo: int, hi: int) -> float:
            m = (r.steps() > lo) & (r.steps() <= hi) & np.isfinite(v)
            return float(v[m].mean()) if m.any() else math.nan

        early_a, early_n = window(a, ra, a.start, a.start + EARLY), window(n, rn, n.start, n.start + EARLY)
        late_a, late_n = (
            window(a, ra, a.start + TRANSIENT, 10**12),
            window(n, rn, n.start + TRANSIENT, 10**12),
        )
        spec = mean_of("weather.rain_total")
        pair = contrast(spec, a, n)
        blocks_a, blocks_n = spec(a, a.blocking()), spec(n, n.blocking())
        ratio = t_interval(np.log(blocks_a / blocks_n))
        out[str(s)] = {
            "early_ratio": early_a / early_n if early_n else None,
            "late_ratio": late_a / late_n if late_n else None,
            "log_early": math.log(early_a / early_n) if early_a > 0 and early_n > 0 else None,
            "log_late": math.log(late_a / late_n) if late_a > 0 and late_n > 0 else None,
            "late_log_ratio_blocks": ratio,
            "late_difference_blocks": pair,
            "early_in_calibration_band": bool(1.10 <= early_a / early_n <= 2.0) if early_n else None,
            "same_sign": bool(
                early_a > 0
                and early_n > 0
                and late_a > 0
                and late_n > 0
                and (math.log(early_a / early_n) > 0) == (math.log(late_a / late_n) > 0)
            ),
        }
    return out


def slope(y: np.ndarray, x: np.ndarray) -> float | None:
    """The least-squares slope of ``y`` on ``x`` (None without variance in ``x``)."""
    xv, yv = np.asarray(x, dtype=np.float64).ravel(), np.asarray(y, dtype=np.float64).ravel()
    var = xv.var()
    return float(((xv - xv.mean()) * (yv - yv.mean())).mean() / var) if var > 0 else None


def p1b(runs: Runs) -> dict[str, Any]:
    """Dose-response, cell-wise, in the ``wx`` dumps both arms share (post-transient): per dump the slope of
    ``P_A - P_N`` on A's seeding strength (plane 9), and the correlation of ``P_A`` and ``P_N``; block
    t-interval over dumps."""
    out: dict[str, Any] = {}
    for s in runs.seeds("A"):
        a, n = runs.get("A", s), runs.get("N", s)
        if a is None or n is None:
            continue
        blk = a.blocking()
        common = sorted(set(a.wx_steps()) & set(n.wx_steps()))
        steps, slopes, corrs = [], [], []
        for st in common:
            if blk.index(np.array([st]))[0] < 0:
                continue
            wa, wn = a.rd.read_wx(st), n.rd.read_wx(st)
            pa, pn = wa[7].astype(np.float64), wn[7].astype(np.float64)
            sl = slope(pa - pn, wa[9])
            c = (
                float(np.corrcoef(pa.ravel(), pn.ravel())[0, 1])
                if pa.std() > 0 and pn.std() > 0
                else math.nan
            )
            steps.append(st)
            slopes.append(math.nan if sl is None else sl)
            corrs.append(c)
        st_arr = np.array(steps, dtype=np.int64)
        out[str(s)] = {
            "dumps": len(steps),
            "slope": _series_interval(blk, st_arr, np.array(slopes)),
            "pair_correlation": _series_interval(blk, st_arr, np.array(corrs)),
        }
    return out


def _series_interval(blk: Blocking, steps: np.ndarray, vals: np.ndarray) -> dict[str, Any]:
    b = block_mean(blk, steps, vals)
    return {"blocks": b.tolist(), **t_interval(b)}


def p1c(runs: Runs) -> dict[str, Any]:
    """Does the crowd seed, or do the genes? The emission above the free level (``emit - 12`` quanta) over the
    post-transient ecology dumps: per z band the mean emission, the paid share ``Σ max(emit-12, 0)/Σ emit``
    and
    the share of agents paying, and the paid share per clade (the ten largest), for each run."""
    out: dict[str, Any] = {}
    for (arm, seed), r in sorted(runs.by.items()):
        blk = r.blocking()
        tot = np.zeros(4)
        paid = np.zeros(4)
        payers = np.zeros(4)
        n = np.zeros(4)
        clade_tot: dict[int, list[float]] = {}
        for st, cols in r.eco():
            if blk.index(np.array([st]))[0] < 0:
                continue
            emit = cols["emit"].astype(np.float64)
            pd = np.maximum(emit - FREE_EMIT, 0.0)
            zb = z_band(cols["z"].astype(np.float64), r.cfg.world.height)
            for b in range(4):
                m = zb == b
                tot[b] += emit[m].sum()
                paid[b] += pd[m].sum()
                payers[b] += (emit[m] > FREE_EMIT).sum()
                n[b] += m.sum()
            for c in np.unique(cols["clade"]):
                m = cols["clade"] == c
                acc = clade_tot.setdefault(int(c), [0.0, 0.0, 0.0])
                acc[0] += float(m.sum())
                acc[1] += float(emit[m].sum())
                acc[2] += float(pd[m].sum())
        top = sorted(clade_tot.items(), key=lambda kv: -kv[1][0])[:10]
        out.setdefault(arm, {})[str(seed)] = {
            "free_emit": FREE_EMIT,
            "agents": n.tolist(),
            "emit_mean": [float(t / k) if k else None for t, k in zip(tot, n, strict=True)],
            "paid_share": [float(p / t) if t else None for p, t in zip(paid, tot, strict=True)],
            "payer_share": [float(p / k) if k else None for p, k in zip(payers, n, strict=True)],
            "paid_share_all": float(paid.sum() / tot.sum()) if tot.sum() else None,
            "clades": {
                str(c): {
                    "agent_dumps": v[0],
                    "emit_mean": v[1] / v[0],
                    "paid_share": v[2] / v[1] if v[1] else None,
                }
                for c, v in top
            },
        }
    return out


def p2(runs: Runs) -> dict[str, Any]:
    """Agents at updraft boundaries against tracers: ``ub_agents_all`` per arm, A - WB (wind) and WB - S (rain
    pattern), against ``ub_A >= 0.12`` and ``ub_A - ub_WB >= 0.02``."""
    spec = mean_of("coupling.ub_agents_all")
    per = per_arm(runs, spec, ARMS)
    if not per:
        return {}
    out: dict[str, Any] = {
        "ub": per,
        "A_minus_WB": compare(runs, spec, "A", "WB", floor=True),
        "WB_minus_S": compare(runs, spec, "WB", "S", floor=True),
    }
    ub_a = {s: v["mean"] for s, v in per.get("A", {}).items()}
    out["ub_A_at_least_0.12"] = {s: (m is not None and m >= UB_MIN) for s, m in ub_a.items()}
    out["wind_contrast_at_least_0.02"] = {
        s: (c["mean"] is not None and c["mean"] >= UB_DIFF) for s, c in out["A_minus_WB"]["seeds"].items()
    }
    return out


def p3(runs: Runs) -> dict[str, Any]:
    """Altitude holding relative to the vertical wind: ``hold`` per arm, A - WB (against 0.05 and the s1 A - N
    spread), and the 250-step ``|dz|`` of agents in drafts against tracers."""
    spec = mean_of("coupling.hold")
    per = per_arm(runs, spec, ARMS)
    if not per:
        return {}
    c = compare(runs, spec, "A", "WB", floor=True)
    pers_a = series_spec(lambda r: persistence_series(r, "agents"))
    pers_t = series_spec(lambda r: persistence_series(r, "tracers"))
    out: dict[str, Any] = {
        "hold": per,
        "A_minus_WB": c,
        "hold_A_positive": {
            s: (v["mean"] is not None and v["mean"] > 0) for s, v in per.get("A", {}).items()
        },
        "wind_contrast_at_least_0.05": {
            s: (v["mean"] is not None and v["mean"] >= HOLD_DIFF) for s, v in c["seeds"].items()
        },
    }
    if any(r.rd.dump_files("eco") for r in runs.by.values()):
        out["abs_dz_250_agents_in_drafts"] = per_arm(runs, pers_a, ARMS)
        out["abs_dz_250_tracers"] = per_arm(runs, pers_t, ARMS)
    return out


def p4(runs: Runs) -> dict[str, Any]:
    """Selection on the buoyancy cap: the standardised gradient of ``cap`` on births (next 250 steps) and on
    survival (250 steps) per block, the contrasts A - WB (wind), A - A1 (cap link), A - E (jet) and WB - S
    (rain pattern), the cap trend and the mean ``alpha``."""
    if not any(r.rd.dump_files("eco") for r in runs.by.values()):
        return {}
    out: dict[str, Any] = {}
    for what in ("births", "survival"):
        spec = series_spec(lambda r, w=what: gradient_series(r, w))
        out[f"beta_{what}"] = {
            "per_arm": per_arm(runs, spec, ARMS),
            "contrasts": {
                name: compare(runs, spec, a, b, floor=True)
                for name, (a, b) in {
                    "A_minus_WB": ("A", "WB"),
                    "A_minus_A1": ("A", "A1"),
                    "A_minus_E": ("A", "E"),
                    "WB_minus_S": ("WB", "S"),
                }.items()
            },
        }
    out["cap_trend"] = {f"{r.arm}.{r.seed}": cap_trend(r) for r in runs.by.values() if r.rd.dump_files("eco")}
    return out


def storm_deaths(run: Run) -> dict[str, Any]:
    """The storm deaths of a run by clade (the ten largest) and, from the last ecology dump before the death,
    by z band; ``unattributed`` counts deaths of agents not in that dump."""
    d = run.rd.read_events("deaths")
    m = (d["cause"] == DeathCause.STORM.value) & (d["step"] > run.start + TRANSIENT)
    st, uid, clade = d["step"][m], d["uid"][m], d["clade"][m]
    by_clade = {str(c): int((clade == c).sum()) for c in np.unique(clade)}
    top = dict(sorted(by_clade.items(), key=lambda kv: -kv[1])[:10])
    band = np.full(st.size, -1)
    for t, cols in run.eco():
        w = np.flatnonzero((st > t) & (st <= t + ECO_WINDOW) & (band < 0))
        if not w.size:
            continue
        j, hit = match_uid(uid[w], cols["uid"])
        band[w[hit]] = z_band(cols["z"][j[hit]].astype(np.float64), run.cfg.world.height)
    return {
        "total": int(m.sum()),
        "by_clade": top,
        "by_z_band": [int((band == b).sum()) for b in range(4)],
        "unattributed": int((band < 0).sum()),
    }


def p5(runs: Runs) -> dict[str, Any]:
    """Does the storm world change the biosphere: population, speed, births, deaths, storm deaths and their
    share, the cloud band's occupancy share, per arm and for the contrasts A - WB (hazard and the rest of the
    wind), A - E (the jet), E - WB (the eddy wind) and WB - S (the rain pattern)."""
    deaths = ("deaths_starved", "deaths_killed", "deaths_aged", "deaths_catastrophe", "deaths_storm")
    specs: dict[str, Spec] = {
        "population": mean_of("n_alive"),
        "speed_mean": mean_of("speed_mean"),
        "births_per_step": rate_of("births"),
        "deaths_per_step": rate_of(*deaths),
        "storm_deaths_per_step": rate_of("deaths_storm"),
        "storm_death_share": share_of(["deaths_storm"], deaths),
        "resource_mean": mean_of("resource_mean"),
        "height_mean": mean_of("height_mean"),
    }
    if any(r.rd.dump_files("eco") for r in runs.by.values()):
        specs["cloud_band_share"] = series_spec(lambda r: zshare_series(r, 2))
    pairs = {
        "A_minus_WB": ("A", "WB"),
        "A_minus_E": ("A", "E"),
        "E_minus_WB": ("E", "WB"),
        "WB_minus_S": ("WB", "S"),
    }
    out: dict[str, Any] = {}
    for name, spec in specs.items():
        out[name] = {
            "per_arm": per_arm(runs, spec, ARMS),
            "contrasts": {k: compare(runs, spec, a, b, floor=True) for k, (a, b) in pairs.items()},
        }
    out["storm_deaths"] = {f"{r.arm}.{r.seed}": storm_deaths(r) for r in runs.by.values()}
    return out


def p6(runs: Runs) -> dict[str, Any]:
    """Safety: the largest occupancy, the overflow count, ``overlap_deep`` (mean share of the live agents, the
    largest share in any row, and against the WB arm), the ledger, the error bits, ``v_clips`` and the weather
    warning (the sticky ``diag`` bit and the rows at the hyperbolicity gate), against the limits of §4.1."""
    out: dict[str, Any] = {}
    wb = {s: runs.get("WB", s) for s in runs.seeds("WB")}
    for (arm, seed), r in sorted(runs.by.items()):
        alive = r.series("n_alive")
        post = r.steps() > r.start + TRANSIENT
        od = r.series("weather.overlap_deep")
        share = od / alive
        diag = np.array([int(row.get("diag", 0)) for row in r.rows], dtype=np.int64)
        hyper = r.series("weather.hyper_max")
        occ = max(
            np.nanmax(r.series(k)) if np.isfinite(r.series(k)).any() else 0.0
            for k in ("max_occupancy", "weather.occ_hz0", "weather.occ_hz1_11", "weather.occ_hz12")
        )
        w_od = None
        if seed in wb and wb[seed] is not None:
            w = wb[seed]
            w_od = float(np.nanmean(w.series("weather.overlap_deep")))
        entry = {
            "rows": len(r.rows),
            "max_occupancy": float(occ),
            "max_occupancy_ok": bool(occ < 43),
            "cell_overflow_total": float(np.nansum(r.series("cell_overflow"))),
            "overlap_deep_mean_share": float(np.nanmean(share[post])) if post.any() else None,
            "overlap_deep_max_share": float(np.nanmax(share)) if np.isfinite(share).any() else None,
            "overlap_deep_mean": float(np.nanmean(od)),
            "overlap_deep_wb_mean": w_od,
            "ledger_error_max": float(np.nanmax(r.series("ledger_error"))),
            "error_bits_set": bool((diag & ERROR_MASK).any()),
            "v_clips_total": float(np.nansum(r.series("v_clips"))),
            "weather_warn_row_share": float(((diag & int(Diag.WEATHER_WARN)) != 0).mean()),
            "hyper_gate_row_share": float((hyper >= HYPER_GATE).mean()) if np.isfinite(hyper).any() else None,
            "population_final": float(alive[-1]),
        }
        entry["overlap_deep_ok"] = bool(
            (
                entry["overlap_deep_mean_share"] is not None
                and entry["overlap_deep_mean_share"] < 0.001
                and entry["overlap_deep_max_share"] is not None
                and entry["overlap_deep_max_share"] < 0.01
            )
            or (w_od is not None and entry["overlap_deep_mean"] <= 3 * w_od)
        )
        entry["safe"] = bool(
            entry["max_occupancy_ok"]
            and entry["cell_overflow_total"] == 0
            and entry["ledger_error_max"] == 0
            and not entry["error_bits_set"]
            and entry["overlap_deep_ok"]
        )
        out.setdefault(arm, {})[str(seed)] = entry
    return out


def analyse(runs: Sequence[Run]) -> dict[str, Any]:
    """Every statistic that the runs' data support."""
    rs = Runs(runs)
    parts: dict[str, Any] = {
        "P1a": p1a(rs),
        "P1b": p1b(rs),
        "P1c": p1c(rs),
        "P2": p2(rs),
        "P3": p3(rs),
        "P4": p4(rs),
        "P5": p5(rs),
        "P6": p6(rs),
    }
    return {
        "design": {
            "transient": TRANSIENT,
            "block": BLOCK,
            "blocks": N_BLOCKS,
            "sensitivity": [2 * N_BLOCKS, BLOCK // 2],
            "early": EARLY,
            "free_emit": FREE_EMIT,
            "eco_window": ECO_WINDOW,
        },
        "runs": [
            {"arm": r.arm, "seed": r.seed, "path": str(r.path), "start": r.start, "rows": len(r.rows)}
            for r in runs
        ],
        **{k: v for k, v in parts.items() if v},
    }


def _clean(x: Any) -> Any:
    """JSON-safe: NaN and infinities become None, numpy scalars Python numbers."""
    if isinstance(x, Mapping):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, list | tuple | np.ndarray):
        return [_clean(v) for v in (x.tolist() if isinstance(x, np.ndarray) else x)]
    if isinstance(x, np.generic):
        return _clean(x.item())
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m alife.analysis.w2", description=__doc__.split("\n")[0])
    ap.add_argument("run", nargs="+", help="[ARM@SEED=]RUN directories (read only)")
    ap.add_argument("--out", type=Path, required=True, help="report JSON (outside the run directories)")
    ap.add_argument(
        "--start", type=int, default=None, help="the step the runs forked at (default: from the rows)"
    )
    a = ap.parse_args(argv)
    try:
        runs = [load_run(spec, start=a.start) for spec in a.run]
    except (ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    for r in runs:
        if a.out.resolve().is_relative_to(r.path.resolve()):
            ap.error("--out must not be inside a run directory")
    report = _clean(analyse(runs))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
