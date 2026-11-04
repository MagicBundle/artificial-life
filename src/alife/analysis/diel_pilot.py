"""The offline analysis of the diel oracle pilot (docs/DIEL.md 7.0, DESIGN 49.7).

Usage::

    python -m alife.analysis.diel_pilot RUN [RUN ...] [--days 10 40] [--boot 2000] [--seed 0] [--json OUT]

Host side, pure numpy. A run directory is read only: ``metrics.jsonl``, ``config.json``
(``diel.period``) and ``meta.json`` (``start_step`` of a fork). A row falls on pilot day ``(step -
start_step) // period``; the window is the days ``[day_lo, day_hi)`` (default 10 to 40).

**Statistic.** Per stratum ``s`` (1 floor, 2 mid, 3 top) and per day, the sums ``sum(n * energy)`` and
``sum(n)`` of the oracle (O) and comparison (C) groups over the rows of the day (``diel_orc_energy`` is
the mean energy of the agents aged 300 to 500, ``diel_orc_n`` their count; rows with a null energy or a
zero count are skipped, per group). ``G_s = ln(mean_O / mean_C)`` with the means taken from the sums
pooled over the window's days and over all runs (the count-weighted mean). The pooled ``G`` is the mean
of the ``G_s`` weighted by stratum size, the mean over the window's rows and runs of ``diel_orc_pop`` of
O plus C of the stratum; a stratum with no O or no C data is left out of that mean.

**Bootstrap.** Whole days are resampled with replacement within each run (as many as the run has days in
the window; the same days for the three strata and for the size weights, so the strata keep their
correlation), the sums are re-pooled over runs and the statistics recomputed. ``B`` replicates (default
2000) from ``numpy.random.default_rng(seed)``. Bounds are the 5th and 95th percentiles (90% two-sided,
so the lower bound is a one-sided 95% bound).

**Holm.** For each stratum ``p_s = (1 + #{b: G_s^b <= 0.05}) / (B + 1)``, the one-sided bootstrap
p-value of H0: ``G_s <= 0.05``. Sorted ascending ``p_(1) <= p_(2) <= p_(3)``, the adjusted value is
``max_{j <= i} min(1, (3 - j + 1) p_(j))`` and a stratum passes if it is below 0.05. A stratum with no
estimate has ``p = 1``.

**Decision.** GO if the pooled lower bound >= 0.05; else SMALL if some stratum passes Holm; else STOP if
the pooled upper bound < 0.02 and every stratum's upper bound < 0.02; else SMALL with
``straddling=True`` (stage 1 only, never GO).

**Photic share** (reported, not decisive): ``photic / all`` of the ``diel_harv_*`` interval counters
summed over the window's rows, per group, stratum and phase (day, night); None when ``all`` is 0."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

STRATA = (1, 2, 3)
GROUPS = ("o", "c")
PHASES = ("day", "night")
GO_LOWER = 0.05  # GO: the pooled lower 90% bound at or above this
STOP_UPPER = 0.02  # STOP: every upper 90% bound below this
LEVEL = 0.90  # two-sided; the bounds are the 5th and 95th percentiles
AGE_WINDOW = (300, 500)  # the age window of diel_orc_energy (engine side)
DAY_WINDOW = (10, 40)
DEFAULT_BOOT = 2000


def rule_params(boot: int = DEFAULT_BOOT, days: tuple[int, int] = DAY_WINDOW) -> dict:
    return {
        "go_lower": GO_LOWER,
        "stop_upper": STOP_UPPER,
        "level": LEVEL,
        "multiplicity": "holm",
        "strata": list(STRATA),
        "window_days": list(days),
        "age_window": list(AGE_WINDOW),
        "bootstrap": boot,
    }


def rule_hash(boot: int = DEFAULT_BOOT, days: tuple[int, int] = DAY_WINDOW) -> str:
    """sha256 of the canonical JSON (sorted keys, no spaces) of the frozen thresholds (pre-registration)."""
    s = json.dumps(rule_params(boot, days), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(s.encode()).hexdigest()


@dataclass
class RunDays:
    """Per-day sums of one run: arrays of shape (n_days, 3). ``days`` are the day indices present."""

    name: str
    days: list[int]
    wsum: dict[str, np.ndarray]  # "o"/"c" -> sum(n * energy)
    nsum: dict[str, np.ndarray]  # "o"/"c" -> sum(n)
    pop: np.ndarray  # sum over rows of (pop_o + pop_c)
    rows: np.ndarray  # number of rows (n_days,)
    harv: dict[str, np.ndarray] = field(default_factory=dict)  # "o_photic_day" etc. -> (n_days, 3)


def _read_json(p: Path) -> dict:
    return json.loads(p.read_text()) if p.exists() else {}


def _group_rows(rows: list[dict], period: int, start: int, lo: int, hi: int, name: str) -> RunDays:
    by_day: dict[int, list[dict]] = {}
    for r in rows:
        if "diel_orc_energy" not in r or "step" not in r:
            continue
        d = (int(r["step"]) - start) // period
        if lo <= d < hi:
            by_day.setdefault(d, []).append(r)
    days = sorted(by_day)
    nd = len(days)
    w = {g: np.zeros((nd, 3)) for g in GROUPS}
    n = {g: np.zeros((nd, 3)) for g in GROUPS}
    pop = np.zeros((nd, 3))
    nrows = np.zeros(nd)
    harv = {f"{g}_{k}_{ph}": np.zeros((nd, 3)) for g in GROUPS for k in ("photic", "all") for ph in PHASES}
    for i, d in enumerate(days):
        for r in by_day[d]:
            nrows[i] += 1
            e, cnt, p = r["diel_orc_energy"], r["diel_orc_n"], r["diel_orc_pop"]
            for j in range(3):
                pop[i, j] += (p[j] or 0) + (p[3 + j] or 0)
                for gi, g in enumerate(GROUPS):
                    k = 3 * gi + j
                    if e[k] is None or not cnt[k] or not math.isfinite(e[k]):
                        continue
                    w[g][i, j] += cnt[k] * e[k]
                    n[g][i, j] += cnt[k]
                for key in harv:
                    g, kind, ph = key.split("_")
                    harv[key][i, j] += r.get(f"diel_harv_{g}{j + 1}_{kind}_{ph}", 0) or 0
    return RunDays(name, days, w, n, pop, nrows, harv)


def load_run(run: str | Path, day_lo: int = DAY_WINDOW[0], day_hi: int = DAY_WINDOW[1]) -> RunDays:
    """Read a run directory and cut its metrics rows to the pilot days ``[day_lo, day_hi)``."""
    run = Path(run)
    cfg = _read_json(run / "config.json")
    period = int(cfg.get("diel", {}).get("period", 0))
    if period <= 0:
        raise ValueError(f"{run}: config.json has no diel.period (the cycle is off)")
    start = int(_read_json(run / "meta.json").get("start_step") or 0)
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines() if line.strip()]
    return _group_rows(rows, period, start, day_lo, day_hi, run.name)


def _stats(sw_o, sn_o, sw_c, sn_c, pop, nrows) -> tuple[np.ndarray, float]:
    """``G_s`` (3,) (nan where a group has no data) and the pooled G from pooled sums (arrays (3,))."""
    with np.errstate(divide="ignore", invalid="ignore"):
        mo, mc = sw_o / sn_o, sw_c / sn_c
        g = np.log(mo / mc)
        g = np.where((sn_o > 0) & (sn_c > 0) & (mo > 0) & (mc > 0), g, np.nan)
        size = pop / max(nrows, 1)
    ok = np.isfinite(g) & (size > 0)
    pooled = float(np.sum(g[ok] * size[ok]) / np.sum(size[ok])) if ok.any() else float("nan")
    return g, pooled


def _pool(runs: list[RunDays], picks: list[np.ndarray] | None):
    """Sum the per-day arrays of the runs over ``picks`` (day indices per run, with repeats) or all days."""
    acc = [np.zeros(3) for _ in range(4)] + [np.zeros(3)]
    nrows = 0.0
    for ri, r in enumerate(runs):
        ix = np.arange(len(r.days)) if picks is None else picks[ri]
        for a, arr in zip(acc, (r.wsum["o"], r.nsum["o"], r.wsum["c"], r.nsum["c"], r.pop), strict=True):
            a += arr[ix].sum(axis=0)
        nrows += r.rows[ix].sum()
    return (*acc, nrows)


def bootstrap(runs: list[RunDays], boot: int = DEFAULT_BOOT, seed: int = 0) -> np.ndarray:
    """``(boot, 4)`` replicates of ``[G_1, G_2, G_3, G_pooled]`` from the day-block bootstrap."""
    rng = np.random.default_rng(seed)
    out = np.empty((boot, 4))
    for b in range(boot):
        picks = [rng.integers(0, len(r.days), len(r.days)) if r.days else np.zeros(0, int) for r in runs]
        g, p = _stats(*_pool(runs, picks))
        out[b, :3], out[b, 3] = g, p
    return out


def holm(p: list[float], alpha: float = 0.05) -> tuple[list[float], list[bool]]:
    """Holm step-down adjusted p-values (in the input order) and whether each is below ``alpha``."""
    m = len(p)
    order = sorted(range(m), key=lambda i: p[i])
    adj = [0.0] * m
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * p[i]))
        adj[i] = run
    return adj, [a < alpha for a in adj]


def photic_share(runs: list[RunDays]) -> dict[str, float | None]:
    """``photic/all`` harvest per ``{o|c}{1|2|3}_{day|night}`` summed over the window and the runs."""
    out: dict[str, float | None] = {}
    for g in GROUPS:
        for ph in PHASES:
            for j, s in enumerate(STRATA):
                ph_sum = sum(r.harv[f"{g}_photic_{ph}"][:, j].sum() for r in runs)
                all_sum = sum(r.harv[f"{g}_all_{ph}"][:, j].sum() for r in runs)
                out[f"{g}{s}_{ph}"] = float(ph_sum / all_sum) if all_sum > 0 else None
    return out


@dataclass(frozen=True)
class PilotResult:
    decision: str  # "GO" | "SMALL" | "STOP"
    straddling: bool  # SMALL because the bounds straddle (stage 1 only), not because a stratum passed
    runs: tuple[str, ...]
    days: tuple[int, int]
    boot: int
    seed: int
    g: dict[str, float | None]  # "1","2","3","pooled"
    lo: dict[str, float | None]
    hi: dict[str, float | None]
    p: dict[str, float]  # one-sided p-values of H0 G_s <= 0.05
    p_holm: dict[str, float]
    holm_pass: dict[str, bool]
    n_days: dict[str, int]  # days with rows, per run
    photic_share: dict[str, float | None]
    rule_hash: str


def _f(x) -> float | None:
    return None if x is None or not math.isfinite(x) else float(x)


def analyse(
    run_dirs: list[str | Path],
    days: tuple[int, int] = DAY_WINDOW,
    boot: int = DEFAULT_BOOT,
    seed: int = 0,
) -> PilotResult:
    runs = [load_run(d, *days) for d in run_dirs]
    if not any(r.days for r in runs):
        raise ValueError("no metrics rows with diel_orc_* fall in the day window")
    g, pooled = _stats(*_pool(runs, None))
    reps = bootstrap(runs, boot, seed)
    with np.errstate(invalid="ignore"):
        lo = np.nanpercentile(reps, 100 * (1 - LEVEL) / 2, axis=0)
        hi = np.nanpercentile(reps, 100 * (1 + LEVEL) / 2, axis=0)
    names = [str(s) for s in STRATA] + ["pooled"]
    est = [*map(_f, g), _f(pooled)]
    p = []
    for j in range(3):
        col = reps[:, j]
        valid = np.isfinite(col)
        p.append(1.0 if not valid.any() else (1 + float(np.sum(col[valid] <= GO_LOWER))) / (valid.sum() + 1))
    adj, ok = holm(p)
    s_lo, s_hi = [_f(x) for x in lo], [_f(x) for x in hi]
    go = s_lo[3] is not None and s_lo[3] >= GO_LOWER
    stop = all(h is not None and h < STOP_UPPER for h in s_hi)
    if go:
        decision, straddling = "GO", False
    elif any(ok):
        decision, straddling = "SMALL", False
    elif stop:
        decision, straddling = "STOP", False
    else:
        decision, straddling = "SMALL", True
    return PilotResult(
        decision=decision,
        straddling=straddling,
        runs=tuple(r.name for r in runs),
        days=tuple(days),
        boot=boot,
        seed=seed,
        g=dict(zip(names, est, strict=True)),
        lo=dict(zip(names, s_lo, strict=True)),
        hi=dict(zip(names, s_hi, strict=True)),
        p={str(s): p[j] for j, s in enumerate(STRATA)},
        p_holm={str(s): adj[j] for j, s in enumerate(STRATA)},
        holm_pass={str(s): bool(ok[j]) for j, s in enumerate(STRATA)},
        n_days={r.name: len(r.days) for r in runs},
        photic_share=photic_share(runs),
        rule_hash=rule_hash(boot, tuple(days)),
    )


def _fmt(x: float | None, nd: int = 3) -> str:
    return "  n/a" if x is None else f"{x:+.{nd}f}"


def format_result(r: PilotResult) -> str:
    lab = {"1": "floor", "2": "mid", "3": "top", "pooled": "pooled"}
    out = [
        f"diel oracle pilot: runs {', '.join(r.runs)}; days {r.days[0]} to {r.days[1]}; "
        f"B={r.boot} seed={r.seed}",
        f"{'':8}{'G':>8}{'lo90':>8}{'hi90':>8}{'p':>8}{'p_holm':>8}  pass",
    ]
    for k in ("1", "2", "3", "pooled"):
        extra = (
            f"{r.p[k]:8.3f}{r.p_holm[k]:8.3f}  {'yes' if r.holm_pass[k] else 'no'}" if k != "pooled" else ""
        )
        out.append(f"{lab[k]:8}{_fmt(r.g[k]):>8}{_fmt(r.lo[k]):>8}{_fmt(r.hi[k]):>8}{extra}")
    out.append("photic harvest share (day / night):")
    for g in GROUPS:
        cells = []
        for s in STRATA:
            d, n = r.photic_share[f"{g}{s}_day"], r.photic_share[f"{g}{s}_night"]
            cells.append(
                f"{g.upper()}{s} {'n/a' if d is None else f'{d:.2f}'}/{'n/a' if n is None else f'{n:.2f}'}"
            )
        out.append("  " + "   ".join(cells))
    tag = r.decision + (" (stage 1 only, bounds straddle)" if r.straddling else "")
    out.append(f"decision: {tag}   rule sha256 {r.rule_hash[:16]}")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m alife.analysis.diel_pilot", description=__doc__.split("\n")[0]
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
