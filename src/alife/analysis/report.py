"""The evolutionary-activity report of a run directory (``alife activity RUN``, docs/ACTIVITY.md).

``compute`` analyses the run's phylogeny and its shadows (``activity.analyse``) and returns flat arrays (saved
as ``activity.npz``, reusable with ``--reuse``); ``summarise`` turns them into the classification tables and
their threshold and normalisation sensitivity (``summary.json``); ``plot`` draws the figures (matplotlib, the
optional ``analysis`` extra).

Classification (Bedau, Snyder & Packard 1998, with Channon's split of the unbounded class). The 1998 paper's
operational definitions are asymptotic (``f`` unbounded iff ``limsup max_{s≤t} f(s) / t > 0``; positive iff
its running time average stays above 0); on a finite window they become trends, with these rules:

- **Bedau** (raw activity, normalised by subtracting the shadow's population-level statistic, 1998):
  new activity is *positive* when the window mean of ``A_new_real − A_new_shadow`` is > 0; ``Ā`` is
  *unbounded* when the OLS slope of ``Ā_real − Ā_shadow`` over the window, in births per component per step,
  exceeds ``ρ0`` times the births per component per step (``B̄ / D̄``): ``ρ = slope · D̄ / B̄`` is the share of
  each step's new activity that the mean cumulative activity keeps; ``D`` is *unbounded* when its fitted
  relative growth across the window, ``slope · |W| / D̄``, exceeds ``g0``. Class: *none* if new activity is
  not positive, else *unbounded* if ``D`` or ``Ā`` is unbounded (Channon's 3a/3b/3c), else *bounded*.
- **Channon** (component-normalised activity, Channon 2002 and the 2024 procedure): *unbounded* iff
  ``A^N_cum`` and the median ``Ã^N`` both grow (``slope / B̄ > ρ0`` and ``slope · D̄ / B̄ > ρ0``) and
  ``A^N_new`` is persistently positive (positive in ≥ ``persist`` of the window's blocks of
  ``PERSIST_BLOCK`` grid times, and its window mean above the neutral shadow's); *bounded* if only the last
  holds; else *none*.

Every threshold is swept and the number of settings giving each class is reported next to the primary one,
over the full sweep and over its informative part (Bedau: activity windows the real or the shadow reaches;
Channon: thresholds where the real run has crossings and the neutral shadow's ``A^N_new`` is at most 20% of
the real one).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

from alife.analysis import activity as act
from alife.analysis import phylo

DEFAULT_RATES: tuple[float, ...] = (0.003, 0.01, 0.03, 0.1)
DEFAULT_HORIZONS: tuple[int, ...] = (100_000, 200_000, 500_000)
PRIMARY_SPACES = ("cohort_h200k", "marker_0.01")
CLASSES = ("none", "bounded", "unbounded")

# Primary settings and sweeps.
RHO0, RHO_SWEEP = 0.01, (0.003, 0.01, 0.03)
G0, G_SWEEP = 0.5, (0.25, 0.5, 1.0)
Q0, Q_SWEEP = 0.05, (0.01, 0.05, 0.2)
PERSIST0, PERSIST_SWEEP = 0.5, (0.5, 0.75)
PERSIST_BLOCK = 5  # grid times per block for "persistently positive"
MIN_POINTS = 3  # grid times a window needs to be classified
A0_SWEEP = 10.0 ** np.arange(1.0, 6.51, 0.5)  # fixed Bedau windows [a0, a0 · WINDOW_RATIO]
WINDOW_RATIO = 10**0.2  # two histogram bins
COHORT_AGES = (20_000, 50_000, 100_000, 200_000)


@dataclass(frozen=True)
class Settings:
    grid_every: int = 10_000
    marker_rates: tuple[float, ...] = DEFAULT_RATES
    cohort_horizons: tuple[int, ...] = DEFAULT_HORIZONS
    shadows: int = 2
    seed: int = 0
    windows: tuple[tuple[int, int | None], ...] = ((0, 120_000), (120_000, 1_000_000), (1_000_000, None))


def horizon_name(h: int) -> str:
    """``cohort_h200k`` for 200 000 steps, ``cohort_h500`` for 500."""
    return f"cohort_h{h // 1000}k" if h % 1000 == 0 else f"cohort_h{h}"


def space_names(st: Settings) -> list[str]:
    return [
        *(horizon_name(h) for h in st.cohort_horizons),
        *(f"marker_{r:g}" for r in st.marker_rates),
    ]


def tree_names(shadows: int) -> list[str]:
    return [
        "real",
        *(f"neutral{i}" for i in range(1, shadows + 1)),
        *(f"permuted{i}" for i in range(1, shadows + 1)),
    ]


# --- computation ------------------------------------------------------------------------------------------


def _flatten(name: str, r: act.Result) -> dict[str, np.ndarray]:
    out = {f"{name}/grid": r.grid, f"{name}/n_alive": r.n_alive, f"{name}/births": r.births}
    for h, s in r.cohort_pooled.items():
        for k, v in s.arrays().items():
            out[f"{name}/{horizon_name(h)}/{k}"] = v
    for rate, s in r.marker.items():
        for k, v in s.arrays().items():
            out[f"{name}/marker_{rate:g}/{k}"] = v
    for f, v in r.cohorts.items():
        out[f"{name}/cohorts/{f}"] = v
    return out


def compute(
    tree: phylo.Tree, st: Settings, log: Callable[[str], None] = print
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """The statistics of the run and its shadows as flat arrays ``<tree>/<space>/<field>``, plus timings."""
    flat: dict[str, np.ndarray] = {}
    timing: dict[str, float] = {}
    rng = np.random.default_rng(st.seed)
    gap = phylo.min_gap(tree)

    def run(name: str, t: phylo.Tree) -> None:
        t0 = time.perf_counter()
        flat.update(
            _flatten(name, act.analyse(t, st.grid_every, st.marker_rates, st.cohort_horizons, gap=gap))
        )
        timing[name] = time.perf_counter() - t0
        log(f"  {name}: {timing[name]:.1f} s")

    run("real", tree)
    for i in range(1, st.shadows + 1):
        t0 = time.perf_counter()
        sh = phylo.neutral_shadow(tree, rng, gap)
        timing[f"neutral{i}_shadow"] = time.perf_counter() - t0
        run(f"neutral{i}", sh)
        del sh
        run(f"permuted{i}", phylo.permuted_shadow(tree, rng))
    return flat, timing


# --- helpers ------------------------------------------------------------------------------------------------


class Data:
    """Read access to the flat arrays, with shadow means."""

    def __init__(self, flat: dict[str, np.ndarray], shadows: int) -> None:
        self.flat = flat
        self.shadows = shadows
        self.grid = flat["real/grid"].astype(np.int64)
        self.births = flat["real/births"]

    def get(self, tree: str, space: str, name: str) -> np.ndarray:
        return self.flat[f"{tree}/{space}/{name}"]

    def reps(self, kind: str) -> list[str]:
        return [f"{kind}{i}" for i in range(1, self.shadows + 1)]

    def mean(self, kind: str, fn: Callable[[str], np.ndarray]) -> np.ndarray:
        return np.mean([fn(t) for t in self.reps(kind)], axis=0)


def _div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, np.float64)
    return np.divide(a, b, out=np.full_like(a, np.nan), where=np.asarray(b) > 0)


def _slope(t: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    return float(np.polyfit(t[ok].astype(np.float64), y[ok], 1)[0])


def pool_horizon(space: str) -> int:
    """The horizon ``H`` of a cohort-pool space (``horizon_name``), 0 for the others."""
    if not space.startswith("cohort_h"):
        return 0
    h = space.removeprefix("cohort_h")
    return int(h[:-1]) * 1000 if h.endswith("k") else int(h)


def window_label(w0: int, w1: int) -> str:
    return f"{w0 // 1000}k-{w1 // 1000}k"


def _mask(grid: np.ndarray, w0: int, w1: int) -> np.ndarray:
    return (grid > w0) & (grid <= w1)


def _f(x: Any) -> Any:
    """JSON-safe floats."""
    if isinstance(x, float | np.floating):
        return None if not np.isfinite(x) else float(x)
    return x


# --- classification ---------------------------------------------------------------------------------------


@dataclass
class Window:
    w0: int
    w1: int
    mask: np.ndarray
    t: np.ndarray
    b_rate: float  # births per step


def _window(d: Data, w0: int, w1: int) -> Window:
    m = _mask(d.grid, w0, w1)
    return Window(w0, w1, m, d.grid[m], float(d.births[m].sum()) / max(w1 - w0, 1))


def bedau_stats(
    d: Data, space: str, w: Window, shadow: str, a0: float, a1: float, *, presence: bool = False
) -> dict[str, Any]:
    """The raw window statistics behind a Bedau classification (any thresholds can be applied later)."""
    hn, hs, cum = ("hist_p_n", "hist_p_sum", "P_cum") if presence else ("hist_n", "hist_sum", "A_cum")

    def new(tree: str) -> np.ndarray:
        return act.a_new(d.get(tree, space, hn), d.get(tree, space, hs), d.get(tree, space, "D"), a0, a1)

    def mean_act(tree: str) -> np.ndarray:
        return _div(d.get(tree, space, cum), d.get(tree, space, "D"))

    m = w.mask
    D = d.get("real", space, "D")[m]
    new_r = new("real")[m]
    new_s = [new(t)[m] for t in d.reps(shadow)]
    abar_excess = mean_act("real")[m] - d.mean(shadow, mean_act)[m]
    d_mean = float(np.mean(D)) if D.size else float("nan")
    rate_scale = 1.0 if presence else w.b_rate / max(d_mean, 1e-12)  # new activity per component per step
    return {
        "a_new_real": float(new_r.mean()) if m.any() else float("nan"),
        "a_new_shadow": [float(x.mean()) for x in new_s],
        "a_new_excess": float(new_r.mean() - np.mean(new_s)) if m.any() else float("nan"),
        "a_new_shadow_spread": float(np.ptp([x.mean() for x in new_s])) if len(new_s) > 1 else 0.0,
        "abar_real_end": float(mean_act("real")[m][-1]) if m.any() else float("nan"),
        "abar_excess_end": float(abar_excess[-1]) if m.any() else float("nan"),
        "rho": _slope(w.t, abar_excess) / rate_scale,
        # Decomposition of rho: the run's own trend and the shadow's (rho = rho_real − rho_shadow).
        "rho_real": _slope(w.t, mean_act("real")[m]) / rate_scale,
        "rho_shadow": _slope(w.t, d.mean(shadow, mean_act)[m]) / rate_scale,
        "D_mean": d_mean,
        "D_growth": _slope(w.t, D) * (w.w1 - w.w0) / max(d_mean, 1e-12),
    }


def primary_window(crossing: Sequence[float | None]) -> tuple[float, float]:
    """The primary Bedau window ``[a', 10^0.2 a']`` from a ``crossing_threshold`` triple ``(a0, a1, a')``.

    The literal 1998 window ``a' ± 0.05 (a_max − a')`` assumes a short tail: in Bedau's example ``a_max / a'``
    is about 5. On ``64k_s0`` it is about 40, so the lower edge falls to 0 and ``A_new`` becomes ``Ā``. The
    primary window therefore starts at the crossing and is as wide as the swept windows; the literal one is
    part of the sweep.
    """
    a_p = crossing[2]
    if a_p is None or not np.isfinite(a_p):
        return (np.inf, np.inf)
    return (float(a_p), float(a_p) * WINDOW_RATIO)


def bedau_class(s: dict[str, Any], rho0: float, g0: float) -> str:
    if not s["a_new_excess"] > 0:
        return "none"
    return "unbounded" if (s["D_growth"] > g0 or s["rho"] > rho0) else "bounded"


def bedau_subclass(s: dict[str, Any], rho0: float, g0: float) -> str:
    c = bedau_class(s, rho0, g0)
    if c != "unbounded":
        return c
    d_unb, a_unb = s["D_growth"] > g0, s["rho"] > rho0
    return (
        "unbounded (3c: D and Ā)"
        if d_unb and a_unb
        else ("unbounded (3a: D)" if d_unb else "unbounded (3b: Ā)")
    )


def channon_stats(d: Data, space: str, w: Window, theta: float) -> dict[str, Any]:
    m = w.mask
    if not np.isfinite(theta):
        return {"theta": None}
    j = int(np.argmin(np.abs(act.THETA_N - theta)))

    def new(tree: str) -> np.ndarray:
        return _div(d.get(tree, space, "cross_n_sum")[:, j], d.get(tree, space, "D"))

    D = d.get("real", space, "D")[m]
    d_mean = float(np.mean(D)) if D.size else float("nan")
    an_cum = d.get("real", space, "AN_cum")[m]
    an_med = d.get("real", space, "AN_med")[m]
    nr = new("real")[m]
    out = {
        "theta": float(act.THETA_N[j]),
        "a_new_real": float(np.nanmean(nr)) if m.any() else float("nan"),
        "a_new_positive_frac": _block_positive(nr),
        "a_new_neutral": float(np.nanmean(d.mean("neutral", new)[m])) if m.any() else float("nan"),
        "a_new_permuted": float(np.nanmean(d.mean("permuted", new)[m])) if m.any() else float("nan"),
        "an_cum_end": float(an_cum[-1]) if m.any() else float("nan"),
        "an_med_end": float(an_med[-1]) if m.any() else float("nan"),
        "rho_cum": _slope(w.t, an_cum) / max(w.b_rate, 1e-12),
        "rho_med": _slope(w.t, an_med) * d_mean / max(w.b_rate, 1e-12),
    }
    return out


def _block_positive(x: np.ndarray) -> float:
    """The fraction of consecutive blocks of ``PERSIST_BLOCK`` values (a shorter last block included) whose
    sum is positive; nan for an empty series."""
    if x.size == 0:
        return float("nan")
    sums = np.add.reduceat(np.nan_to_num(x), np.arange(0, x.size, PERSIST_BLOCK))
    return float(np.mean(sums > 0))


def channon_informative(s: dict[str, Any]) -> bool:
    """A threshold where the real run has crossings and the neutral shadow's new activity is ≤ 20% of it."""
    return bool(
        s.get("theta") is not None and s["a_new_real"] > 0 and s["a_new_neutral"] <= 0.2 * s["a_new_real"]
    )


def channon_class(s: dict[str, Any], rho0: float, persist: float) -> str:
    if s.get("theta") is None:
        return "none"
    positive = s["a_new_positive_frac"] >= persist and s["a_new_real"] > s["a_new_neutral"]
    if not positive:
        return "none"
    return "unbounded" if (s["rho_cum"] > rho0 and s["rho_med"] > rho0) else "bounded"


def _count(classes: Iterable[str]) -> dict[str, int]:
    out = dict.fromkeys(CLASSES, 0)
    for c in classes:
        out[c] += 1
    return out


def summarise(
    flat: dict[str, np.ndarray], st: Settings, coverage: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Classification tables, thresholds, sensitivity and the cohort-age summary (``summary.json``)."""
    d = Data(flat, st.shadows)
    t_end = int(d.grid[-1])
    windows = [(0, t_end)] + [(a, t_end if b is None else b) for a, b in st.windows]
    spaces = [s for s in space_names(st) if f"real/{s}/D" in flat]
    out: dict[str, Any] = {"settings": asdict(st), "coverage": coverage, "t_analysed": t_end, "spaces": {}}

    for space in spaces:
        thr: dict[str, Any] = {}
        for shadow in ("neutral", "permuted"):
            for presence in (False, True):
                hn = "hist_p_n" if presence else "hist_n"
                rh = d.get("real", space, hn)
                sh = np.sum([d.get(t, space, hn) for t in d.reps(shadow)], axis=0) / d.shadows
                thr[f"{shadow}_{'presence' if presence else 'births'}"] = [
                    _f(x) for x in act.crossing_threshold(rh, sh)
                ]
        sh_np = np.sum([d.get(t, space, "hist_np_sum") for t in d.reps("neutral")], axis=0)
        thetas = {q: act.shadow_theta(sh_np, q) for q in Q_SWEEP}
        an_min = float(np.min(d.get("real", space, "AN_min")))
        thr["channon_theta_q"] = {str(q): _f(v) for q, v in thetas.items()}
        thr["channon_minus_min_rule"] = _f(-an_min)

        sp_out: dict[str, Any] = {"thresholds": thr, "windows": {}}
        settle = pool_horizon(space)  # a cohort pool is filling up before its horizon
        for w0, w1 in windows:
            w = _window(d, max(w0, settle), w1)
            if w.t.size < MIN_POINTS:
                sp_out["windows"][window_label(w0, w1)] = {"valid": False, "valid_from": max(w0, settle)}
                continue
            a0, a1 = primary_window(thr["neutral_births"])
            prim_b = bedau_stats(d, space, w, "neutral", a0, a1)
            prim_c = channon_stats(d, space, w, thetas[Q0])
            cell: dict[str, Any] = {
                "valid": True,
                "valid_from": w.w0,
                "births_per_step": w.b_rate,
                "bedau": {
                    **{k: _f(v) if not isinstance(v, list) else v for k, v in prim_b.items()},
                    "window": [a0, a1],
                    "class": bedau_subclass(prim_b, RHO0, G0),
                },
                "channon": {
                    **{k: _f(v) for k, v in prim_c.items()},
                    "class": channon_class(prim_c, RHO0, PERSIST0),
                },
            }
            # Sensitivity: Bedau over shadow × increment × window × ρ0 × g0.
            bed: list[str] = []
            bed_all: list[str] = []
            bed_by: dict[str, list[str]] = {}
            for shadow, presence in product(("neutral", "permuted"), (False, True)):
                key = f"{shadow}_{'presence' if presence else 'births'}"
                cw = thr[key]
                wins = [(a, a * WINDOW_RATIO) for a in A0_SWEEP]
                if cw[2] is not None:  # the crossing windows: [a', 1.58 a'] and the literal 1998 one
                    wins += [primary_window(cw), (cw[0], cw[1])]
                for a0_, a1_ in wins:
                    s = bedau_stats(d, space, w, shadow, a0_, a1_, presence=presence)
                    cls = [bedau_class(s, r0, g0) for r0, g0 in product(RHO_SWEEP, G_SWEEP)]
                    bed_all += cls
                    if not (s["a_new_real"] > 0 or max(s["a_new_shadow"]) > 0):
                        continue  # no component in this activity window at all: uninformative
                    bed += cls
                    bed_by.setdefault(key, []).extend(cls)
            # Sensitivity: Channon over θ × ρ0 × persistence.
            ch: list[str] = []
            ch_all: list[str] = []
            for th in act.THETA_N:
                s = channon_stats(d, space, w, th)
                cls = [channon_class(s, r0, p) for r0, p in product(RHO_SWEEP, PERSIST_SWEEP)]
                ch_all += cls
                if channon_informative(s):
                    ch += cls
            cell["sensitivity"] = {
                "bedau": _count(bed),
                "bedau_full": _count(bed_all),
                "bedau_by_shadow_increment": {k: _count(v) for k, v in bed_by.items()},
                "channon": _count(ch),
                "channon_full": _count(ch_all),
            }
            # The A_new excess across the fixed windows (threshold sweep).
            cell["a_new_excess_by_a0"] = {
                f"{a:g}": _f(bedau_stats(d, space, w, "neutral", a, a * 10**0.2)["a_new_excess"])
                for a in A0_SWEEP
            }
            cell["an_new_by_theta"] = {
                f"{th:g}": {
                    k: _f(v)
                    for k, v in channon_stats(d, space, w, th).items()
                    if k in ("a_new_real", "a_new_neutral", "a_new_permuted", "a_new_positive_frac")
                }
                for th in act.THETA_N
            }
            sp_out["windows"][window_label(w0, w1)] = cell
        out["spaces"][space] = sp_out

    if "real/cohorts/D" in flat and flat["real/cohorts/D"].size:
        out["cohort_ages"] = cohort_age_summary(d, windows[1:], st.grid_every)
    return out


def cohort_age_table(d: Data, age: int, grid_every: int) -> dict[str, np.ndarray]:
    """Per cohort (start ``grid[k]``) at ``age`` steps: real/shadow ratios of D and Ā, and
    ``A^N_cum / A_cum``."""
    lag = age // grid_every
    K = d.flat["real/cohorts/D"].shape[0]
    ks = np.arange(K)
    ks = ks[ks + lag < d.grid.size]

    def at(tree: str, f: str) -> np.ndarray:
        return d.flat[f"{tree}/cohorts/{f}"][ks, ks + lag]

    def abar(tree: str) -> np.ndarray:
        return _div(at(tree, "A_cum"), at(tree, "D"))

    return {
        "start": d.grid[ks],
        "D_real": at("real", "D"),
        "D_neutral": d.mean("neutral", lambda t: at(t, "D")),
        "D_permuted": d.mean("permuted", lambda t: at(t, "D")),
        "abar_ratio_neutral": abar("real") / d.mean("neutral", abar),
        "abar_ratio_permuted": abar("real") / d.mean("permuted", abar),
        "excess_frac_real": _div(at("real", "AN_cum"), at("real", "A_cum")),
        "excess_frac_neutral": d.mean("neutral", lambda t: _div(at(t, "AN_cum"), at(t, "A_cum"))),
        "excess_frac_permuted": d.mean("permuted", lambda t: _div(at(t, "AN_cum"), at(t, "A_cum"))),
    }


def cohort_age_summary(d: Data, windows: Sequence[tuple[int, int]], grid_every: int) -> dict[str, Any]:
    """Means over the cohorts started in each window, and the trend over cohort start (per million steps)."""
    out: dict[str, Any] = {}
    for age in COHORT_AGES:
        tab = cohort_age_table(d, age, grid_every)
        per: dict[str, Any] = {}
        for w0, w1 in windows:
            m = (tab["start"] >= w0) & (tab["start"] < w1)
            if m.sum() == 0:
                continue
            row: dict[str, Any] = {"cohorts": int(m.sum())}
            for k, v in tab.items():
                if k == "start":
                    continue
                row[k] = _f(float(np.nanmean(v[m])))
                if m.sum() >= 3:
                    row[k + "_trend_per_M"] = _f(_slope(tab["start"][m], v[m]) * 1e6)
            per[window_label(w0, w1)] = row
        out[f"age_{age // 1000}k"] = per
    return out


# --- output ------------------------------------------------------------------------------------------------


def save_flat(path: Path, flat: dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **flat)


def load_flat(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.write_text(json.dumps(summary, indent=1, default=_f) + "\n", encoding="utf-8")


def classification_table(summary: dict[str, Any]) -> str:
    """A plain-text table: one row per space and window with both classes and their sensitivity counts."""
    lines = [
        f"{'space':<14} {'window':<14} {'Bedau (primary)':<24} {'Bedau sweep n/b/u':<18} "
        f"{'Channon (primary)':<18} {'Channon sweep n/b/u'}"
    ]
    for space, sp in summary["spaces"].items():
        for wl, c in sp["windows"].items():
            if not c["valid"]:
                lines.append(f"{space:<14} {wl:<14} n/a (cohort pool still filling before {c['valid_from']})")
                continue
            sb, sc = c["sensitivity"]["bedau"], c["sensitivity"]["channon"]
            lines.append(
                f"{space:<14} {wl:<14} {c['bedau']['class']:<24} "
                f"{sb['none']}/{sb['bounded']}/{sb['unbounded']:<12} {c['channon']['class']:<18} "
                f"{sc['none']}/{sc['bounded']}/{sc['unbounded']}"
            )
    return "\n".join(lines)


def run_report(
    run: Path,
    out: Path,
    st: Settings,
    *,
    reuse: bool = False,
    plots: bool = True,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """The whole analysis: load, compute (or reuse ``activity.npz``), summarise, plot."""
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    log(f"loading {run}")
    tree, coverage = phylo.load_tree(run)
    coverage["load_s"] = time.perf_counter() - t0
    npz = out / "activity.npz"
    if reuse and npz.is_file():
        log(f"reusing {npz}")
        flat = load_flat(npz)
        timing: dict[str, float] = {}
    else:
        log("analysing the run and its shadows")
        flat, timing = compute(tree, st, log)
        save_flat(npz, flat)
    del tree
    summary = summarise(flat, st, coverage)
    summary["timing_s"] = {**timing, "total": time.perf_counter() - t0}
    write_summary(out / "summary.json", summary)
    table = classification_table(summary)
    (out / "classification.txt").write_text(table + "\n", encoding="utf-8")
    log(table)
    if plots:
        from alife.analysis import plots as plots_mod  # needs the optional 'analysis' extra

        for p in plots_mod.plot_all(flat, summary, st, out):
            log(f"  wrote {p}")
    return summary
