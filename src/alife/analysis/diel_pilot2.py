"""The second diel oracle pilot's decision rule: lineage fitness (DESIGN 49.14, PR D1c).

Usage::

    python -m alife.analysis.diel_pilot2 RUN [RUN ...] [--days 1 40] [--boot 2000] [--seed 0] [--json OUT]

The first pilot (D1b's schedule, rule ``alife.analysis.diel_pilot``) could not be read: the oracle lineages
died out before its day window, and its measure ``G`` (the energy of survivors aged 300 to 500) does not see
mortality. This rule replaces ``G`` as the decision by the oracle lineages' fitness relative to their
comparison group; ``G`` is still reported. The rule was set after the first pilot's result was seen (post
hoc, recorded in DESIGN 49.14) and before the second pilot ran.

Host side, pure numpy; a run directory is read only (``metrics.jsonl``, ``config.json``, ``meta.json``). A
row falls on pilot day ``(step - start_step) // period``.

**Statistic.** Per run, stratum ``s`` (1 floor, 2 mid, 3 top) and day ``d``: ``N_O``, ``N_C`` the means over
the day's rows of ``diel_orc_pop`` (alive oracle and comparison rows of the stratum, all ages). The data are
the daily increments ``ln(N_O/N_C)(d+1) - ln(N_O/N_C)(d)`` for consecutive days ``d, d+1`` of the window
``[day_lo, day_hi)`` (default days 0 to 39; day 0's increment holds the schedule's first application).

*Censoring.* An increment counts only while both groups hold at least ``N_MIN`` (10) agents on day ``d``.
When one group falls below ``N_MIN`` on day ``d+1`` and the other does not, that last increment counts with
the fallen group's count raised to ``N_MIN`` (an understatement of the loss, never a made-up floor), and the
stratum's later increments of that run do not count (the lineage is absorbed; a ratio near a count of 0 says
nothing). When both fall below ``N_MIN`` together, no further increment counts. ``w_s`` is the mean of the
counted increments pooled over the runs (each increment weighs the same, so a run with more counted days
weighs more), in units of 1/day; the pooled ``w`` is the mean of
the ``w_s`` weighted by stratum size, the mean over the window's days and runs of ``N_O + N_C``.

**Bootstrap.** Whole days are resampled with replacement within each run (as many as the run has
increments; the same days for the three strata and the size weights); ``B`` replicates (default 2000) from
``numpy.random.default_rng(seed)``; bounds are the 5th and 95th percentiles (90% two-sided).

**Decision.** INVALID if a run's first metrics row has fewer than ``MIN_FOUNDERS`` (50) oracle rows in a
stratum, or a stratum has no counted increment in any run; else GO if the pooled lower bound >= ``GO_LOWER``
(+0.01/day); else STOP if the pooled upper bound < ``STOP_POOLED`` (0) and every stratum's upper bound <
``STOP_STRATUM`` (+0.005); else SMALL (stage 1 only). ``rule_hash`` freezes these."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from alife.analysis import diel_pilot

STRATA = (1, 2, 3)
GO_LOWER = 0.01  # GO: the pooled lower 90% bound of w at or above this (1/day)
STOP_POOLED = 0.0  # STOP: the pooled upper bound below this ...
STOP_STRATUM = 0.005  # ... and every stratum's upper bound below this
MIN_FOUNDERS = 50  # INVALID: fewer oracle rows than this in a stratum at the pilot's first metrics row
N_MIN = 10  # an increment counts while both groups hold at least this many agents (censoring)
LEVEL = 0.90
DAY_WINDOW = (0, 40)
DEFAULT_BOOT = 2000
SCHEDULE = 2  # the oracle schedule the rule was set for (diel.oracle_schedule)


def rule_params(boot: int = DEFAULT_BOOT, days: tuple[int, int] = DAY_WINDOW) -> dict:
    return {
        "measure": "lineage_fitness",
        "schedule": SCHEDULE,
        "go_lower": GO_LOWER,
        "stop_pooled": STOP_POOLED,
        "stop_stratum": STOP_STRATUM,
        "min_founders": MIN_FOUNDERS,
        "n_min": N_MIN,
        "censoring": "absorb_clamped",
        "level": LEVEL,
        "strata": list(STRATA),
        "window_days": list(days),
        "bootstrap": boot,
    }


def rule_hash(boot: int = DEFAULT_BOOT, days: tuple[int, int] = DAY_WINDOW) -> str:
    """sha256 of the canonical JSON (sorted keys, no spaces) of the frozen rule (pre-registration)."""
    s = json.dumps(rule_params(boot, days), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(s.encode()).hexdigest()


@dataclass
class RunFit:
    """One run, per window day ``d`` with a day ``d + 1``: ``inc`` (n, 3) the increment of ``ln(N_O/N_C)``
    (NaN where it does not count, censoring), ``size`` (n, 3) the stratum size ``N_O + N_C`` on day ``d``;
    ``founders`` (3,) the oracle rows of the first metrics row; ``extinct`` the first window day with no
    oracle row per stratum (None if never)."""

    name: str
    inc: np.ndarray
    size: np.ndarray
    founders: tuple[int, int, int]
    extinct: tuple[int | None, int | None, int | None]
    schedule: int


def _daily(rows: list[dict], period: int, start: int) -> dict[int, np.ndarray]:
    """Per pilot day, the (6,) mean of ``diel_orc_pop`` (O1 O2 O3 C1 C2 C3) over the day's rows."""
    acc: dict[int, list[np.ndarray]] = {}
    for r in rows:
        if "diel_orc_pop" not in r or "step" not in r:
            continue
        d = (int(r["step"]) - start) // period
        acc.setdefault(d, []).append(np.array([v or 0 for v in r["diel_orc_pop"]], dtype=np.float64))
    return {d: np.mean(v, axis=0) for d, v in acc.items()}


def fit_rows(
    rows: list[dict], period: int, start: int, days: tuple[int, int], name: str = "run", schedule: int = 0
) -> RunFit:
    """The increments of one run's rows over the window ``[day_lo, day_hi)``."""
    daily = _daily(rows, period, start)
    first = next((r for r in rows if "diel_orc_pop" in r), None)
    founders = tuple(int(first["diel_orc_pop"][j] or 0) for j in range(3)) if first else (0, 0, 0)
    lo, hi = days
    inc, size = [], []
    live = np.ones(3, bool)  # the strata not yet absorbed
    for d in range(lo, hi - 1):
        if d in daily and d + 1 in daily:
            a, b = daily[d], daily[d + 1]
            row = np.full(3, np.nan)
            for j in range(3):
                if not live[j] or min(a[j], a[3 + j]) < N_MIN:
                    live[j] = False
                    continue
                o, c = b[j], b[3 + j]
                if o < N_MIN and c < N_MIN:  # both absorbed together: no information
                    live[j] = False
                    continue
                if o < N_MIN or c < N_MIN:  # one absorbed: the last increment, clamped at N_MIN
                    live[j] = False
                row[j] = math.log(max(o, N_MIN) / max(c, N_MIN)) - math.log(a[j] / a[3 + j])
            inc.append(row)
            size.append(a[:3] + a[3:])
    extinct = tuple(
        next((d for d in range(lo, hi) if d in daily and daily[d][j] == 0), None) for j in range(3)
    )
    return RunFit(
        name,
        np.array(inc).reshape(-1, 3),
        np.array(size).reshape(-1, 3),
        founders,  # type: ignore[arg-type]
        extinct,  # type: ignore[arg-type]
        schedule,
    )


def load_run(run: str | Path, days: tuple[int, int] = DAY_WINDOW) -> RunFit:
    run = Path(run)
    cfg = diel_pilot._read_json(run / "config.json")
    period = int(cfg.get("diel", {}).get("period", 0))
    if period <= 0:
        raise ValueError(f"{run}: config.json has no diel.period (the cycle is off)")
    start = int(diel_pilot._read_json(run / "meta.json").get("start_step") or 0)
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines() if line.strip()]
    sched = int(cfg.get("diel", {}).get("oracle_schedule", 1))
    return fit_rows(rows, period, start, days, run.name, sched)


def _stats(runs: list[RunFit], picks: list[np.ndarray] | None) -> np.ndarray:
    """``[w_1, w_2, w_3, w_pooled]`` from the runs' increments at ``picks`` (or all)."""
    inc = np.concatenate([r.inc if picks is None else r.inc[picks[i]] for i, r in enumerate(runs)])
    size = np.concatenate([r.size if picks is None else r.size[picks[i]] for i, r in enumerate(runs)])
    if inc.shape[0] == 0:
        return np.full(4, np.nan)
    cnt = np.isfinite(inc).sum(axis=0)
    w = np.where(cnt > 0, np.nansum(inc, axis=0) / np.maximum(cnt, 1), np.nan)
    sz = size.mean(axis=0)
    ok = np.isfinite(w) & (sz > 0)
    pooled = float(np.sum(w[ok] * sz[ok]) / np.sum(sz[ok])) if ok.any() else float("nan")
    return np.array([*w, pooled])


def bootstrap(runs: list[RunFit], boot: int = DEFAULT_BOOT, seed: int = 0) -> np.ndarray:
    """``(boot, 4)`` replicates of ``[w_1, w_2, w_3, w_pooled]`` from the day-block bootstrap."""
    rng = np.random.default_rng(seed)
    out = np.empty((boot, 4))
    for b in range(boot):
        picks = [rng.integers(0, len(r.inc), len(r.inc)) for r in runs]
        out[b] = _stats(runs, picks)
    return out


@dataclass(frozen=True)
class Pilot2Result:
    decision: str  # "GO" | "SMALL" | "STOP" | "INVALID"
    reason: str  # why INVALID ("" otherwise)
    runs: tuple[str, ...]
    days: tuple[int, int]
    boot: int
    seed: int
    w: dict[str, float | None]  # "1","2","3","pooled", in 1/day
    lo: dict[str, float | None]
    hi: dict[str, float | None]
    founders: dict[str, list[int]]  # per run, the oracle rows per stratum at the first metrics row
    extinct: dict[str, list[int | None]]  # per run, the first window day with no oracle row per stratum
    schedules: dict[str, int]  # per run, diel.oracle_schedule
    g: dict[str, float | None]  # the energy measure of 49.7 over the same days (reported, not decisive)
    photic_share: dict[str, float | None]
    rule_hash: str


def _f(x) -> float | None:
    return None if x is None or not math.isfinite(x) else float(x)


def analyse(
    run_dirs: list[str | Path], days: tuple[int, int] = DAY_WINDOW, boot: int = DEFAULT_BOOT, seed: int = 0
) -> Pilot2Result:
    runs = [load_run(d, days) for d in run_dirs]
    names = [str(s) for s in STRATA] + ["pooled"]
    est = _stats(runs, None)
    counted = sum((np.isfinite(r.inc).sum(axis=0) for r in runs), np.zeros(3, int))
    if counted.all():
        reps = bootstrap(runs, boot, seed)
        with np.errstate(invalid="ignore"):  # a replicate may draw no counted day of a stratum
            lo = np.nanpercentile(reps, 100 * (1 - LEVEL) / 2, axis=0)
            hi = np.nanpercentile(reps, 100 * (1 + LEVEL) / 2, axis=0)
    else:
        lo = hi = np.full(4, np.nan)
    reason = ""
    few = [
        f"{r.name} stratum {s}"
        for r in runs
        for s, n in zip(STRATA, r.founders, strict=True)
        if n < MIN_FOUNDERS
    ]
    if few:
        reason = f"fewer than {MIN_FOUNDERS} oracle founders: {', '.join(few)}"
    elif not counted.all():
        missing = [str(s) for s, n in zip(STRATA, counted, strict=True) if not n]
        reason = "no counted increment in stratum " + ", ".join(missing)
    if reason:
        decision = "INVALID"
    elif lo[3] >= GO_LOWER:
        decision = "GO"
    elif hi[3] < STOP_POOLED and all(h < STOP_STRATUM for h in hi[:3]):
        decision = "STOP"
    else:
        decision = "SMALL"
    try:  # the energy measure over the same days, for the record
        g_runs = [diel_pilot.load_run(d, *days) for d in run_dirs]
        g_s, g_p = diel_pilot._stats(*diel_pilot._pool(g_runs, None))
        g = dict(zip(names, [*map(_f, g_s), _f(g_p)], strict=True))
        share = diel_pilot.photic_share(g_runs)
    except ValueError:
        g, share = dict.fromkeys(names), {}
    return Pilot2Result(
        decision=decision,
        reason=reason,
        runs=tuple(r.name for r in runs),
        days=tuple(days),  # type: ignore[arg-type]
        boot=boot,
        seed=seed,
        w=dict(zip(names, map(_f, est), strict=True)),
        lo=dict(zip(names, map(_f, lo), strict=True)),
        hi=dict(zip(names, map(_f, hi), strict=True)),
        founders={r.name: list(r.founders) for r in runs},
        extinct={r.name: list(r.extinct) for r in runs},
        schedules={r.name: r.schedule for r in runs},
        g=g,
        photic_share=share,
        rule_hash=rule_hash(boot, tuple(days)),  # type: ignore[arg-type]
    )


def _fmt(x: float | None, nd: int = 4) -> str:
    return "n/a" if x is None else f"{x:+.{nd}f}"


def format_result(r: Pilot2Result) -> str:
    lab = {"1": "floor", "2": "mid", "3": "top", "pooled": "pooled"}
    out = [
        f"diel oracle pilot 2 (lineage fitness): runs {', '.join(r.runs)}; days {r.days[0]} to {r.days[1]}; "
        f"B={r.boot} seed={r.seed}; schedules {sorted(set(r.schedules.values()))}",
        f"{'':8}{'w/day':>10}{'lo90':>10}{'hi90':>10}{'G':>9}",
    ]
    for k in ("1", "2", "3", "pooled"):
        g = r.g.get(k)
        out.append(f"{lab[k]:8}{_fmt(r.w[k]):>10}{_fmt(r.lo[k]):>10}{_fmt(r.hi[k]):>10}{_fmt(g, 3):>9}")
    for name in r.runs:
        out.append(f"  {name}: founders {r.founders[name]}, oracle extinct on day {r.extinct[name]}")
    if r.photic_share:
        cells = []
        for g in ("o", "c"):
            for s in STRATA:
                d, n = r.photic_share.get(f"{g}{s}_day"), r.photic_share.get(f"{g}{s}_night")
                ds, ns = ("n/a" if v is None else f"{v:.2f}" for v in (d, n))
                cells.append(f"{g.upper()}{s} {ds}/{ns}")
        out.append("photic harvest share (day / night): " + "  ".join(cells))
    tag = r.decision + (f" ({r.reason})" if r.reason else "")
    out.append(f"decision: {tag}   rule sha256 {r.rule_hash[:16]}")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m alife.analysis.diel_pilot2", description=__doc__.split("\n")[0]
    )
    ap.add_argument("runs", nargs="+", help="pilot run directories")
    ap.add_argument("--days", nargs=2, type=int, default=list(DAY_WINDOW), metavar=("LO", "HI"))
    ap.add_argument("--boot", type=int, default=DEFAULT_BOOT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None, help="write the full result as JSON")
    a = ap.parse_args(argv)
    res = analyse(a.runs, (a.days[0], a.days[1]), a.boot, a.seed)
    print(format_result(res))
    if a.json:
        Path(a.json).write_text(json.dumps(asdict(res), indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
