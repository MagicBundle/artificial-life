"""Evolutionary activity of an arm: the adapter over ``activity.py`` and ``report.py`` (DESIGN §44.5).

Nothing is re-implemented. ``analyse_arm`` loads the run's phylogeny (``phylo.load_tree``, from the fork step
for a fork), runs ``report.compute`` (the run, neutral shadows and permuted shadows) and
``report.summarise``, then reads the endpoint P1 out of the flat arrays: the **Channon normalised new
activity** ``A^N_new`` of the primary space at the threshold ``theta`` the neutral shadow fixes
(``act.shadow_theta`` at ``report.Q0``), per block of the run (8 blocks of 10 000 steps after a 20 000-step
transient, ``stats.run_blocking``). Next to it: the same series of the neutral and permuted shadows (the
excess over the neutral shadow is reported per block), the class (``none``, ``bounded``, ``unbounded``) with
the number of sweep settings giving each, and the slope ``rho``.

Caveat (kept from ``docs/ACTIVITY.md``): the components are lineages, so activity measures persistent
differential reproduction of lineages, not new phenotypes; a feature that creates new phenotype classes
inside a lineage is seen by MODES, not here.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any, Final

import numpy as np

from alife.analysis import activity as act
from alife.analysis import phylo, report
from alife.analysis import w2 as w2mod
from alife.analysis.emergence import stats

ACTIVITY_VERSION: Final = 1
LONG_ARM: Final = 300_000  # steps from the fork above which the 200k-cohort space has settled

CAVEATS: Final = (
    "Components are lineages: A^N_new measures persistent differential reproduction of lineages, not new "
    "phenotypes (MODES sees those).",
    "Thresholds (theta from the neutral shadow at q = 0.05, the class rules) are those of analysis/report.py "
    "and provisional until the calibration run.",
)


def settings_for(grid_every: int = 10_000, shadows: int = 2, seed: int = 0) -> report.Settings:
    """The ``report.Settings`` of an arm: only the whole post-fork window (no fixed sub-windows)."""
    return report.Settings(grid_every=grid_every, shadows=shadows, seed=seed, windows=())


def pick_space(flat: dict[str, np.ndarray], span: int) -> str:
    """The primary space: ``cohort_h200k`` for arms of at least ``LONG_ARM`` steps, else ``marker_0.01``."""
    want = report.PRIMARY_SPACES[0] if span >= LONG_ARM else report.PRIMARY_SPACES[1]
    if f"real/{want}/D" in flat:
        return want
    for s in report.PRIMARY_SPACES:
        if f"real/{s}/D" in flat:
            return s
    raise KeyError("no primary activity space in the computed arrays")


def p1_series(
    flat: dict[str, np.ndarray], st: report.Settings, space: str, q: float = report.Q0
) -> dict[str, Any]:
    """``A^N_new`` per grid time for the run and the shadow means, at the neutral shadow's ``theta``."""
    d = report.Data(flat, st.shadows)
    sh = np.sum([d.get(t, space, "hist_np_sum") for t in d.reps("neutral")], axis=0)
    theta = act.shadow_theta(sh, q)
    if not np.isfinite(theta):
        return {"theta": None, "grid": d.grid, "real": None, "neutral": None, "permuted": None}
    j = int(np.argmin(np.abs(act.THETA_N - theta)))

    def new(tree: str) -> np.ndarray:
        return report._div(d.get(tree, space, "cross_n_sum")[:, j], d.get(tree, space, "D"))

    return {
        "theta": float(act.THETA_N[j]),
        "grid": d.grid,
        "real": new("real"),
        "neutral": d.mean("neutral", new),
        "permuted": d.mean("permuted", new),
    }


def block_values(blk: w2mod.Blocking, grid: np.ndarray, series: np.ndarray | None) -> list[float | None]:
    """Block means of a grid-time series (``None`` for a block with no finite value)."""
    if series is None:
        return [None] * blk.n
    return [None if not np.isfinite(v) else float(v) for v in w2mod.block_mean(blk, grid, series)]


def endpoint(
    flat: dict[str, np.ndarray],
    st: report.Settings,
    summary: dict[str, Any],
    start: int,
    *,
    space: str | None = None,
    blocking: w2mod.Blocking | None = None,
) -> dict[str, Any]:
    """The P1 endpoint and its context from computed arrays and their summary."""
    t_end = int(summary["t_analysed"])
    space = space or pick_space(flat, t_end - start)
    blk = blocking or stats.run_blocking(start)
    s = p1_series(flat, st, space)
    real = block_values(blk, s["grid"], s["real"])
    neu = block_values(blk, s["grid"], s["neutral"])
    per = block_values(blk, s["grid"], s["permuted"])
    excess = [None if a is None or b is None else a - b for a, b in zip(real, neu, strict=True)]
    win = next(iter(summary["spaces"][space]["windows"].values()), {})
    ch = win.get("channon", {}) if win.get("valid") else {}
    return {
        "space": space,
        "theta": s["theta"],
        "blocking": {"start": blk.start, "transient": blk.transient, "width": blk.width, "n": blk.n},
        "blocks": real,
        "neutral_blocks": neu,
        "permuted_blocks": per,
        "excess_blocks": excess,
        "channon_class": ch.get("class"),
        "rho_cum": ch.get("rho_cum"),
        "rho_med": ch.get("rho_med"),
        "sensitivity": win.get("sensitivity", {}).get("channon"),
        "valid": bool(win.get("valid")),
        "t_analysed": t_end,
    }


def fingerprint(run: Path, origin: int, cov: dict[str, Any], st: report.Settings) -> dict[str, Any]:
    """What the cached arrays depend on: the run (path and config hash), the fork step, the end of the events,
    and every setting of ``report.compute`` (shadows, grid, spaces, seed). A cache with another fingerprint is
    recomputed, never reused."""
    meta_p = Path(run) / "meta.json"
    ch = None
    if meta_p.is_file():
        ch = json.loads(meta_p.read_text(encoding="utf-8")).get("config_hash")
    return {
        "run": str(Path(run).resolve()),
        "config_hash": ch,
        "start": int(origin),
        "t_end": cov.get("t_end"),
        "settings": json.loads(json.dumps(asdict(st), default=list)),
        "version": ACTIVITY_VERSION,
    }


def analyse_arm(
    run: Path,
    *,
    start: int | None = None,
    st: report.Settings | None = None,
    space: str | None = None,
    blocking: w2mod.Blocking | None = None,
    out: Path | None = None,
    reuse: bool = True,
    log: Callable[[str], None] = lambda m: None,
) -> dict[str, Any]:
    """The activity block of a run: tree, ``report.compute`` and ``summarise``, then the P1 endpoint.

    ``out`` caches the flat arrays (``activity.npz``) so a reuse skips the computation. Timings (seconds) of
    the load and of ``report.compute`` per tree are returned (the cost to budget for an arm)."""
    st = st or settings_for()
    t0 = time.perf_counter()
    tree, cov = phylo.load_tree(Path(run), start=start)
    t_load = time.perf_counter() - t0
    origin = int(cov["t0"])
    npz = Path(out) / "activity.npz" if out else None
    timing: dict[str, float] = {"load": t_load}
    fp = fingerprint(Path(run), origin, cov, st)
    fp_path = Path(out) / "activity.fp.json" if out else None
    cached = (
        reuse
        and npz is not None
        and npz.is_file()
        and fp_path is not None
        and fp_path.is_file()
        and json.loads(fp_path.read_text(encoding="utf-8")) == fp
    )
    if cached:
        flat = report.load_flat(npz)  # type: ignore[arg-type]
    else:
        flat, tm = report.compute(tree, st, log, step0=origin, gap=cov["min_parent_child_gap"])
        timing.update(tm)
        if npz is not None and fp_path is not None:
            Path(out).mkdir(parents=True, exist_ok=True)  # type: ignore[arg-type]
            report.save_flat(npz, flat)
            fp_path.write_text(json.dumps(fp, sort_keys=True), encoding="utf-8")
    n_agents = len(tree.birth) if hasattr(tree, "birth") else None
    del tree
    t1 = time.perf_counter()
    summary = report.summarise(flat, st, cov)
    timing["summarise"] = time.perf_counter() - t1
    timing["total"] = time.perf_counter() - t0
    ep = endpoint(flat, st, summary, origin, space=space, blocking=blocking)
    return {
        "version": ACTIVITY_VERSION,
        "start": origin,
        "agents": n_agents,
        "coverage": {k: cov[k] for k in ("t0", "t_end", "min_parent_child_gap") if k in cov},
        "endpoint": ep,
        "timing_s": timing,
        "caveats": list(CAVEATS),
    }
