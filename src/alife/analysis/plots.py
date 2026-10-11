"""Figures of the evolutionary-activity report (matplotlib, the optional ``analysis`` extra; Agg backend).

Three series throughout, always in the same colours: the run (blue), the neutral shadow (orange) and the
heritability null, i.e. the permuted shadow (aqua); shadows are the mean over their replicates.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from alife.analysis import activity as act
from alife.analysis.report import (
    PRIMARY_SPACES,
    Data,
    Settings,
    cohort_age_table,
    pool_horizon,
    primary_window,
)

SURFACE = "#fcfcfb"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COLORS = {"real": "#2a78d6", "neutral": "#eb6834", "permuted": "#1baf7a"}
LABELS = {"real": "run", "neutral": "neutral shadow", "permuted": "heritability null"}


def _setup() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": INK2,
            "axes.labelcolor": INK,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "text.color": INK,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 1.6,
            "font.size": 9,
            "axes.titlesize": 10,
            "legend.frameon": False,
            "legend.fontsize": 8,
        }
    )
    return plt


def _div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.divide(a, b, out=np.full(np.shape(a), np.nan), where=np.asarray(b) > 0)


def _windows(ax: Any, st: Settings, t_end: int) -> None:
    for w0, _ in st.windows[1:]:
        ax.axvline(w0 / 1e6, color=INK2, lw=0.8, ls=":", zorder=0)


def _three(ax: Any, d: Data, fn: Any, t_min: int = 0) -> None:
    x = d.grid / 1e6
    m = d.grid >= t_min
    ax.plot(x[m], fn("real")[m], color=COLORS["real"], label=LABELS["real"])
    for kind in ("neutral", "permuted"):
        ax.plot(x[m], d.mean(kind, fn)[m], color=COLORS[kind], label=LABELS[kind])


def fig_bedau(d: Data, summary: dict[str, Any], st: Settings, out: Path, plt: Any) -> Path:
    """D, mean cumulative activity and Bedau A_new (1998 crossing window) for the primary spaces."""
    fig, axes = plt.subplots(len(PRIMARY_SPACES), 3, figsize=(11, 3.1 * len(PRIMARY_SPACES)), squeeze=False)
    for row, space in zip(axes, PRIMARY_SPACES, strict=True):
        t_min = pool_horizon(space)
        a0, a1 = primary_window(summary["spaces"][space]["thresholds"]["neutral_births"])

        def D(t: str, space: str = space) -> np.ndarray:
            return d.get(t, space, "D")

        def abar(t: str, space: str = space) -> np.ndarray:
            return _div(d.get(t, space, "A_cum"), d.get(t, space, "D"))

        def anew(t: str, space: str = space, a0: float = a0, a1: float = a1) -> np.ndarray:
            g = d.get
            return act.a_new(g(t, space, "hist_n"), g(t, space, "hist_sum"), g(t, space, "D"), a0, a1)

        for ax, fn, title in zip(
            row,
            (D, abar, anew),
            ("diversity D", "mean cumulative activity Ā", f"new activity A_new, [{a0:.3g}, {a1:.3g}]"),
            strict=True,
        ):
            _three(ax, d, fn, t_min)
            _windows(ax, st, int(d.grid[-1]))
            ax.set_title(f"{space}\n{title}", loc="left")
            ax.set_xlabel("step (millions)")
        row[0].legend(loc="best")
    fig.tight_layout()
    path = out / "bedau.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def fig_channon(d: Data, summary: dict[str, Any], st: Settings, out: Path, plt: Any) -> Path:
    """Component-normalised statistics for the primary spaces: Ā^N, median Ã^N, A^N_new at θ(q = 5%)."""
    fig, axes = plt.subplots(len(PRIMARY_SPACES), 3, figsize=(11, 3.1 * len(PRIMARY_SPACES)), squeeze=False)
    for row, space in zip(axes, PRIMARY_SPACES, strict=True):
        t_min = pool_horizon(space)
        theta = summary["spaces"][space]["thresholds"]["channon_theta_q"]["0.05"]
        j = int(np.argmin(np.abs(act.THETA_N - theta)))

        def abar(t: str, space: str = space) -> np.ndarray:
            return _div(d.get(t, space, "AN_cum"), d.get(t, space, "D"))

        def med(t: str, space: str = space) -> np.ndarray:
            return d.get(t, space, "AN_med")

        def new(t: str, space: str = space, j: int = j) -> np.ndarray:
            return _div(d.get(t, space, "cross_n_sum")[:, j], d.get(t, space, "D"))

        for ax, fn, title in zip(
            row,
            (abar, med, new),
            (
                "mean normalised activity Ā^N",
                "median normalised activity Ã^N",
                f"new normalised activity A^N_new, θ = {theta:.3g}",
            ),
            strict=True,
        ):
            _three(ax, d, fn, t_min)
            _windows(ax, st, int(d.grid[-1]))
            ax.axhline(0, color=INK2, lw=0.8)
            ax.set_title(f"{space}\n{title}", loc="left")
            ax.set_xlabel("step (millions)")
        row[0].legend(loc="best")
    fig.tight_layout()
    path = out / "channon.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def fig_thresholds(d: Data, summary: dict[str, Any], st: Settings, out: Path, plt: Any) -> Path:
    """Threshold sensitivity: window means of A^N_new against θ, and of the Bedau A_new excess against a0."""
    wins = list(summary["spaces"][PRIMARY_SPACES[0]]["windows"])[1:]  # the windows, not the whole run
    fig, axes = plt.subplots(len(PRIMARY_SPACES), 2, figsize=(11, 3.2 * len(PRIMARY_SPACES)), squeeze=False)
    styles = ["-", "--", ":"]
    for row, space in zip(axes, PRIMARY_SPACES, strict=True):
        cells = summary["spaces"][space]["windows"]
        ax = row[0]
        for wl, ls in zip(wins, styles, strict=False):
            c = cells[wl]
            if not c.get("valid"):
                continue
            th = np.array([float(k) for k in c["an_new_by_theta"]])
            for kind, key in (
                ("real", "a_new_real"),
                ("neutral", "a_new_neutral"),
                ("permuted", "a_new_permuted"),
            ):
                y = np.array(
                    [v[key] if v[key] is not None else np.nan for v in c["an_new_by_theta"].values()]
                )
                ax.plot(th, y, color=COLORS[kind], ls=ls, label=f"{LABELS[kind]}, {wl}")
        theta = summary["spaces"][space]["thresholds"]["channon_theta_q"]["0.05"]
        ax.axvline(theta, color=INK2, lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("threshold θ on normalised activity a^N")
        ax.set_title(f"{space}\nwindow mean of A^N_new", loc="left")
        ax.legend(fontsize=6, ncol=2)
        ax = row[1]
        for wl, ls in zip(wins, styles, strict=False):
            c = cells[wl]
            if not c.get("valid"):
                continue
            a0 = np.array([float(k) for k in c["a_new_excess_by_a0"]])
            y = np.array([v if v is not None else np.nan for v in c["a_new_excess_by_a0"].values()])
            ax.plot(a0, y, color=COLORS["real"], ls=ls, label=wl)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=1.0)
        ax.set_xlabel("lower edge a0 of the activity window [a0, 1.58·a0]")
        ax.set_title(f"{space}\nBedau A_new, run − neutral shadow (window mean)", loc="left")
        ax.legend()
    fig.tight_layout()
    path = out / "thresholds.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def fig_distribution(d: Data, summary: dict[str, Any], st: Settings, out: Path, plt: Any) -> Path:
    """Time-collapsed activity distributions (Bedau 1998 Fig. 1) with the crossing a'."""
    fig, axes = plt.subplots(1, len(PRIMARY_SPACES), figsize=(11, 3.2), squeeze=False)
    for ax, space in zip(axes[0], PRIMARY_SPACES, strict=True):
        rn = d.get("real", space, "hist_n").sum(axis=0)
        sn = d.mean("neutral", lambda t, space=space: d.get(t, space, "hist_n")).sum(axis=0)
        pn = d.mean("permuted", lambda t, space=space: d.get(t, space, "hist_n")).sum(axis=0)
        tot = rn.sum() + sn.sum()
        x = np.maximum(act.A_EDGES, 0.5)
        for kind, y in (("real", rn), ("neutral", sn), ("permuted", pn)):
            ax.step(x, np.where(y > 0, y / tot, np.nan), where="post", color=COLORS[kind], label=LABELS[kind])
        ap = summary["spaces"][space]["thresholds"]["neutral_births"][2]
        if ap is not None:
            ax.axvline(ap, color=INK2, lw=0.8)
            ax.annotate(
                f"a' = {ap:.3g}",
                (ap, 0.5),
                xycoords=("data", "axes fraction"),
                color=INK2,
                xytext=(4, 0),
                textcoords="offset points",
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("activity a (births by members)")
        ax.set_title(f"{space}\ncomponent-activity distribution", loc="left")
    axes[0][0].legend()
    fig.tight_layout()
    path = out / "distribution.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def fig_cohorts(d: Data, summary: dict[str, Any], st: Settings, out: Path, plt: Any) -> Path:
    """Cohort lineage sorting: surviving lineages per cohort against cohort start, at three ages, and the
    share of the survivors' births that is excess over the reset shadow."""
    ages = (50_000, 100_000, 200_000)
    fig, axes = plt.subplots(1, len(ages) + 1, figsize=(13, 3.2), squeeze=False)
    for ax, age in zip(axes[0], ages, strict=False):
        tab = cohort_age_table(d, age, st.grid_every)
        x = tab["start"] / 1e6
        for kind in ("real", "neutral", "permuted"):
            ax.plot(x, tab[f"D_{kind}"], color=COLORS[kind], label=LABELS[kind])
        _windows(ax, st, int(d.grid[-1]))
        ax.set_yscale("log")
        ax.set_xlabel("cohort start (million steps)")
        ax.set_title(f"surviving lineages at age {age // 1000}k", loc="left")
    axes[0][0].legend()
    ax = axes[0][-1]
    tab = cohort_age_table(d, 100_000, st.grid_every)
    for kind in ("real", "neutral", "permuted"):
        ax.plot(tab["start"] / 1e6, tab[f"excess_frac_{kind}"], color=COLORS[kind], label=LABELS[kind])
    _windows(ax, st, int(d.grid[-1]))
    ax.set_xlabel("cohort start (million steps)")
    ax.set_title("A^N_cum / A_cum of the survivors\nat age 100k", loc="left")
    fig.tight_layout()
    path = out / "cohorts.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_all(flat: dict[str, np.ndarray], summary: dict[str, Any], st: Settings, out: Path) -> list[Path]:
    plt = _setup()
    d = Data(flat, st.shadows)
    figs = [fig_bedau, fig_channon, fig_thresholds, fig_distribution]
    if "real/cohorts/D" in flat and flat["real/cohorts/D"].size:
        figs.append(fig_cohorts)
    return [f(d, summary, st, out, plt) for f in figs]
