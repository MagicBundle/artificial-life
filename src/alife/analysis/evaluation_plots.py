"""Figures of the director evaluation (``alife.analysis.evaluation``; matplotlib, the ``analysis`` extra).

Arms keep one colour everywhere: the control A is the neutral ink, dashed (a reference, not a peer series);
the directed arms take the categorical slots in a fixed order (B blue, C orange, R aqua, RC magenta,
D yellow), so a seed without an arm does not repaint the others.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from alife.analysis import evaluation as ev
from alife.analysis.plots import INK2, _setup

ARM_COLORS = {"A": INK2, "B": "#2a78d6", "C": "#eb6834", "R": "#1baf7a", "RC": "#e87ba4", "D": "#eda100"}
ARM_LABELS = {
    "A": "A  off",
    "B": "B  Gardener (rules)",
    "C": "C  Trickster (rules + LLM)",
    "R": "R  random, yoked to B",
    "RC": "RC random, yoked to C",
    "D": "D  Architect (rules + LLM)",
}
SMOOTH = 25  # rule boundaries (25k steps) in the novelty running mean


def _color(name: str) -> str:
    return ARM_COLORS.get(name, "#e87ba4")


def _style(name: str) -> dict[str, Any]:
    return {"color": _color(name), "ls": "--" if name == "A" else "-", "lw": 1.3 if name == "A" else 1.6}


def _col(bds: Sequence[Mapping[str, Any]], key: str) -> np.ndarray:
    return np.array([np.nan if b[key] is None else float(b[key]) for b in bds])


def _running_mean(y: np.ndarray, n: int) -> np.ndarray:
    out = np.full(y.size, np.nan)
    ok = np.isfinite(y)
    c = np.cumsum(np.where(ok, y, 0.0))
    k = np.cumsum(ok)
    for i in range(y.size):
        lo = max(i - n + 1, 0)
        cnt = k[i] - (k[lo - 1] if lo else 0)
        if cnt:
            out[i] = (c[i] - (c[lo - 1] if lo else 0)) / cnt
    return out


def fig_timeseries(
    seed: int,
    arms: Sequence[ev.Arm],
    series: Mapping[tuple[int, str], Sequence[Mapping[str, Any]]],
    out: Path,
) -> Path:
    """Coverage, novelty (running mean), population and top-clade share over time for one seed, with a tick
    under the novelty panel for every intervention step of each directed arm."""
    plt = _setup()
    fig, axes = plt.subplots(4, 1, figsize=(9, 10), sharex=True, layout="constrained")
    mine = [a for a in arms if a.seed == seed]
    for a in mine:
        bds = series[(seed, a.name)]
        x = _col(bds, "step") / 1e6
        st = _style(a.name)
        axes[0].plot(x, _col(bds, "coverage"), label=ARM_LABELS.get(a.name, a.name), **st)
        axes[1].plot(x, _running_mean(_col(bds, "novelty"), SMOOTH), **st)
        axes[2].plot(x, _col(bds, "n_alive") / 1e3, **st)
        axes[3].plot(x, _col(bds, "top_clade_share"), **st)
    _end_labels(axes[0], [(a.name, _col(series[(seed, a.name)], "coverage")[-1]) for a in mine])
    lo = axes[1].get_ylim()[0]
    for i, a in enumerate(x for x in mine if x.name != "A"):
        steps = np.array(ev.anchor_steps(ev.log_entries(a.run)), np.float64) / 1e6
        axes[1].plot(steps, np.full(steps.size, lo), "|", color=_color(a.name), ms=7 + 3 * i, mew=1.2)
    axes[0].set_ylabel("coverage (grid cells)")
    axes[1].set_ylabel(f"novelty ({SMOOTH}k-step mean)")
    axes[2].set_ylabel("population (thousands)")
    axes[3].set_ylabel("top clade share")
    axes[3].set_xlabel("step (millions)")
    axes[0].set_title(f"Seed {seed}: arms from the fork at 120k (ticks: intervention steps)", loc="left")
    axes[0].legend(loc="upper left", ncols=2)
    path = out / f"timeseries_s{seed}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _end_labels(ax: Any, ends: Sequence[tuple[str, float]]) -> None:
    """Direct labels at the right end of the lines, nudged apart so that close ends do not collide."""
    lo, hi = ax.get_ylim()
    gap = 0.045 * (hi - lo)
    placed: list[float] = []
    for name, y in sorted(ends, key=lambda e: e[1]):
        y_lab = max(y, placed[-1] + gap) if placed else y
        placed.append(y_lab)
        ax.annotate(
            f" {name}",
            (1.0, y_lab),
            xycoords=("axes fraction", "data"),
            color=INK2,
            fontsize=8,
            va="center",
            annotation_clip=False,
        )


SUMMARY_METRICS: tuple[tuple[str, str], ...] = (
    ("coverage.final", "final coverage (cells)"),
    ("persistent_novelty.vs_control.mean", "persistent novelty vs A"),
    ("novelty.mean", "mean novelty"),
    ("viability.pop_min_over_ref", "lowest population / reference"),
)


def fig_summary(summary: Mapping[str, Any], out: Path) -> Path:
    """One panel per headline metric: each arm's value on every seed (circle seed 0, square seed 1)."""
    plt = _setup()
    rows = summary["arms"]
    names = [n for n in ARM_COLORS if any(r["arm"] == n for r in rows)]
    fig, axes = plt.subplots(1, len(SUMMARY_METRICS), figsize=(12, 3.2), layout="constrained")
    markers = {0: "o", 1: "s"}
    for ax, (key, title) in zip(axes, SUMMARY_METRICS, strict=True):
        for i, n in enumerate(names):
            for r in (r for r in rows if r["arm"] == n):
                v = ev.dig(r, key)
                if v is None:
                    continue
                ax.plot(i + (r["seed"] - 0.5) * 0.18, v, markers.get(r["seed"], "D"), color=_color(n), ms=8)
        ax.set_xticks(range(len(names)), names)
        ax.set_xlim(-0.6, len(names) - 0.4)
        ax.set_title(title, loc="left")
    axes[0].plot([], [], "o", color=INK2, label="seed 0")
    axes[0].plot([], [], "s", color=INK2, label="seed 1")
    axes[0].legend(loc="upper left")
    path = out / "summary.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_all(
    summary: Mapping[str, Any],
    series: Mapping[tuple[int, str], Sequence[Mapping[str, Any]]],
    arms: Sequence[ev.Arm],
    out: Path,
) -> list[Path]:
    paths = [fig_timeseries(s, arms, series, out) for s in sorted({a.seed for a in arms})]
    return [*paths, fig_summary(summary, out)]
