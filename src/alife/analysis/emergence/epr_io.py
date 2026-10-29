"""Outputs of block B (DESIGN §44.3): ``epr.json``, ``epr_scale.csv`` and the plots (matplotlib, optional).

Writes are atomic (tmp file then rename). The plots need the optional ``analysis`` extra; without it
``plot_*`` return ``None``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from alife.analysis.emergence.epr import flux_for_plot
from alife.analysis.emergence.frames import Frames


def _atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


SCALE_COLUMNS = (
    "cell",
    "n_cells",
    "rate",
    "se",
    "z",
    "plugin",
    "move_share",
    "ep_per_move",
    "n",
    "active_pairs",
)


def scale_csv(result: dict[str, Any]) -> str:
    """The scale curve (own motion) as CSV text."""
    rows = [",".join(SCALE_COLUMNS)]
    for r in result["scale_curve"]:
        rows.append(",".join("" if r.get(c) is None else repr(r[c]) for c in SCALE_COLUMNS))
    return "\n".join(rows) + "\n"


def write_outputs(
    out: Path, result: dict[str, Any], fr: Frames | None = None, *, plot: bool = True
) -> list[Path]:
    """Write ``epr.json``, ``epr_scale.csv`` and (with matplotlib and ``fr``) ``epr_flux.png``,
    ``epr_scale.png``; returns the paths written."""
    out = Path(out)
    paths = [out / "epr.json", out / "epr_scale.csv"]
    _atomic(paths[0], json.dumps(result, indent=1, sort_keys=True) + "\n")
    _atomic(paths[1], scale_csv(result))
    if plot:
        for p in (plot_scale(out / "epr_scale.png", result), plot_flux(out / "epr_flux.png", fr, result)):
            if p is not None:
                paths.append(p)
    return paths


def _plt() -> Any | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    return plt


def plot_scale(path: Path, result: dict[str, Any]) -> Path | None:
    """Own-motion rate against cell size with its bootstrap error bars (a lower bound at every scale)."""
    plt = _plt()
    if plt is None:
        return None
    c = [r["cell"] for r in result["scale_curve"] if r.get("rate") is not None]
    y = [r["rate"] for r in result["scale_curve"] if r.get("rate") is not None]
    e = [r["se"] or 0.0 for r in result["scale_curve"] if r.get("rate") is not None]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.errorbar(c, y, yerr=e, marker="o", color="#2a78d6")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("cell side (world units)")
    ax.set_ylabel("EP, nats / organism / 1000 steps")
    ax.set_title("scale curve (lower bound at every scale)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_flux(path: Path, fr: Frames | None, result: dict[str, Any]) -> Path | None:
    """The net-flux arrows on the headline lattice (own motion)."""
    plt = _plt()
    if plt is None or fr is None:
        return None
    field, lat = flux_for_plot(fr, result["cell"], result["stride"])
    cen = lat.centres().reshape(lat.ncell, lat.ncell, 2)
    fig, ax = plt.subplots(figsize=(5, 5))
    mag = np.linalg.norm(field, axis=2)
    ax.imshow(mag.T, origin="lower", extent=(0, lat.size, 0, lat.size), cmap="Blues")
    ax.quiver(cen[..., 0], cen[..., 1], field[..., 0], field[..., 1], color="#0b0b0b", angles="xy")
    ax.set_title("net probability flux (own motion)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
