"""Figures, snapshots and movies of a W0 study (``alife.analysis.w0``; matplotlib, the optional ``analysis``
extra).

Matplotlib is imported lazily (``_setup``), so importing this module needs only numpy. GIFs are written with
``matplotlib.animation.PillowWriter`` (Pillow comes with matplotlib). The palette and rcParams are those of
``analysis/plots.py:_setup``, copied rather than imported from a private helper; diverging maps are
``RdBu_r``, sequential maps ``Blues``; figures are saved at dpi 130."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

SURFACE = "#fcfcfb"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GIF_MAX_BYTES = 10_000_000
BAND_TITLES = (
    "floor (hash layer 0)",
    "mid-column (hash layers 1 to G_z - 2)",
    "surface (hash layer G_z - 1)",
)


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


def _save(fig: Any, plt: Any, path: Path) -> str:
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path.name


def lat_hist(out: Path, plt: Any, lat: dict[str, Any], length: float) -> str:
    """Centroid latitude of every qualifying detection (8-unit bins), equators and poles marked."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.4))
    edges = np.asarray(lat["bin_edges"])
    a.bar(
        edges[:-1],
        lat["hist"],
        width=np.diff(edges),
        align="edge",
        color=BLUE,
        edgecolor=SURFACE,
        linewidth=0.4,
    )
    ymax = float(edges[-1])  # the y extent: L on the torus, Ly in a channel
    equators, poles = lat["equators"], lat["poles"]
    for y in (*equators, *(() if len(equators) == 1 else (ymax,))):
        a.axvline(y, color=INK2, ls="--", lw=0.9)
    for y in poles:
        a.axvline(y, color=ORANGE, ls=":", lw=1.2)
    a.set(
        xlabel="centroid y",
        ylabel="detections",
        title="storm latitudes (dashed: equators, dotted: poles)",
        xlim=(0, ymax),
    )
    gy, gh = np.asarray(lat["genesis_y"]), np.asarray(lat["genesis_hemi"])
    pd = np.abs(gy - np.where(gh == 0, poles[0], poles[1])) if gy.size else gy
    if pd.size:
        top = (
            length / 4 if len(equators) == 2 else ymax / 2
        )  # a torus pole is a quarter period from its equator
        b.hist(pd, bins=np.arange(0, top + 8, 8), color=AQUA, edgecolor=SURFACE)
    b.set(
        xlabel="genesis distance from the pole of its hemisphere", ylabel="storms", title="where storms form"
    )
    fig.tight_layout()
    return _save(fig, plt, out / "lat_hist.png")


def counts(out: Path, plt: Any, arrays: dict[str, Any]) -> str:
    fig, ax = plt.subplots(figsize=(9, 3.2))
    t = np.asarray(arrays["counts_times"])
    ax.axhspan(2, 6, color=GRID, alpha=0.7, label="criterion 2 to 6")
    ax.step(t, arrays["counts_north"], where="post", color=BLUE, label="north")
    ax.step(t, arrays["counts_south"], where="post", color=ORANGE, label="south")
    ax.set(xlabel="step", ylabel="storms alive", title="storms per hemisphere", ylim=(0, None))
    ax.legend(ncol=3)
    fig.tight_layout()
    return _save(fig, plt, out / "counts.png")


def km_plot(out: Path, plt: Any, km: Any, median_text: str) -> str:
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ages = np.concatenate([[0.0], km.ages])
    surv = np.concatenate([[1.0], km.survival])
    ax.step(ages, surv, where="post", color=BLUE)
    ax.axhline(0.5, color=INK2, ls="--", lw=0.9)
    ax.axvline(3000, color=ORANGE, ls=":", lw=1.2, label="3000 steps")
    ax.set(
        xlabel="storm age (steps)",
        ylabel="survival",
        title=f"Kaplan-Meier, median {median_text}",
        ylim=(0, 1.02),
    )
    ax.legend()
    fig.tight_layout()
    return _save(fig, plt, out / "km.png")


def forecast_plot(out: Path, plt: Any, fc: dict[str, Any]) -> str:
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.6))
    rows = fc["leads"][1000]["rows"]
    err = np.array([r["err"] for r in rows])
    per = np.array([r["persist"] for r in rows])
    flo = np.array([r["floor"] for r in rows])
    if err.size:
        a.boxplot(
            [err, per, flo], tick_labels=["extrapolation", "persistence", "jitter floor"], showfliers=False
        )
        b.scatter(flo, err, s=10, color=BLUE)
        top = max(err.max(), flo.max()) * 1.05
        b.plot([0, top], [0, top], color=INK2, ls="--", lw=0.9)
    a.axhspan(0.5, 2.0, color=GRID, alpha=0.7)
    a.set(ylabel="error at 1000 steps (storm radii)", title="forecast error (band: criterion 0.5 to 2)")
    b.set(
        xlabel="jitter floor (radii)", ylabel="extrapolation error (radii)", title="error against the floor"
    )
    fig.tight_layout()
    return _save(fig, plt, out / "forecast.png")


def drift_plot(out: Path, plt: Any, dr: dict[str, Any]) -> str:
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.6))
    rec = np.asarray(dr["records"]).reshape(-1, 5)
    if rec.size:
        for h, name, col in ((0, "north", BLUE), (1, "south", ORANGE)):
            s = rec[rec[:, 0] == h]
            a.hist(s[:, 2], bins=30, alpha=0.7, color=col, label=name)
            b.scatter(s[:, 3], s[:, 4], s=6, alpha=0.5, color=col, label=name)
        a.legend()
    a.set(xlabel="drift per 250 steps (storm radii)", ylabel="count", title="storm drift")
    b.axhline(0, color=INK2, lw=0.8)
    b.axvline(0, color=INK2, lw=0.8)
    b.set(xlabel="zonal (east +), units", ylabel="poleward, units", title="drift per 250 steps")
    fig.tight_layout()
    return _save(fig, plt, out / "drift.png")


def occupancy_plot(
    out: Path,
    plt: Any,
    rows: Sequence[dict[str, Any]],
    occ: dict[str, Any] | None,
    occ_fh: dict[str, Any] | None = None,
) -> str | None:
    """Per band, the maximum cell count over each detection interval against time (floor-held arm dashed)."""
    if occ is None or not rows or "occ_hz0" not in rows[0]:
        return None
    fig, axes = plt.subplots(3, 1, figsize=(9, 6.4), sharex=True)
    t = np.array([r["t"] for r in rows])
    cap = occ["capacity"]
    for ax, name, title in zip(axes, ("hz0", "hz1_11", "hz12"), BAND_TITLES, strict=True):
        v = np.array([r[f"occ_{name}"] for r in rows])
        ax.plot(t, v, color=BLUE, label="tracers")
        top = float(v.max()) if v.size else 0.0
        if occ_fh is not None and f"fh_occ_{name}" in rows[0]:
            vf = np.array([r[f"fh_occ_{name}"] for r in rows])
            ax.plot(t, vf, color=AQUA, ls="--", label="floor-held arm")
            top = max(top, float(vf.max()))
        ax.axhline(cap, color=ORANGE, ls="-", lw=1.0)
        ax.axhline(0.75 * cap, color=ORANGE, ls=":", lw=1.0)
        base = occ["bands"][name].get("baseline_peak")
        if base is not None:
            ax.axhline(base, color=INK2, ls="--", lw=0.9)
        ax.set(ylabel="max per cell", title=title, ylim=(0, max(cap * 1.1, top * 1.1)))
    if occ_fh is not None:
        axes[0].legend(ncol=2, loc="center right")
    axes[-1].set_xlabel(
        "step (max over each detection interval; solid: C, dotted: 0.75 C, dashed: checkpoint)"
    )
    fig.tight_layout()
    return _save(fig, plt, out / "occupancy.png")


def jets_plot(
    out: Path,
    plt: Any,
    y: np.ndarray,
    zonal: np.ndarray,
    length: float,
    equators: Sequence[float] | None = None,
    poles: Sequence[float] | None = None,
) -> str:
    """The time-mean zonal-mean winds by y. ``length`` is the y extent; ``equators`` and ``poles`` default to
    the torus's (``0, L/2, L`` and ``L/4, 3L/4``)."""
    fig, ax = plt.subplots(figsize=(4.8, 5.2))
    ax.plot(zonal[0], y, color=BLUE, label="u1")
    ax.plot(zonal[1], y, color=ORANGE, label="u2")
    eq = (0.0, length / 2, length) if equators is None else equators
    pl = (length / 4, 3 * length / 4) if poles is None else poles
    for yy in eq:
        ax.axhline(yy, color=INK2, ls="--", lw=0.8)
    for yy in pl:
        ax.axhline(yy, color=ORANGE, ls=":", lw=1.0)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set(xlabel="time-mean zonal-mean wind", ylabel="y", title="jets", ylim=(0, length))
    ax.locator_params(axis="x", nbins=5)
    ax.legend()
    fig.tight_layout()
    return _save(fig, plt, out / "jets.png")


def hist_plot(out: Path, plt: Any, h: dict[str, Any], p: Any) -> str:
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.4))
    zb = np.asarray(h["zeta_bins_f0"])
    a.bar(zb[:-1], np.asarray(h["zeta_counts"]), width=np.diff(zb), align="edge", color=BLUE)
    a.axvline(p.zeta_th / p.f0, color=ORANGE, ls=":", lw=1.3, label="zeta_th")
    a.set(yscale="log", xlabel="s zeta_c / f0", ylabel="cells", title="vorticity (window)")
    a.legend()
    pb = np.asarray(h["p_bins"])
    pc = np.asarray(h["p_counts"])
    b.bar(np.arange(len(pc)), pc, color=AQUA)
    ticks = [0, *range(1, len(pc), 5)]
    b.set_xticks(ticks, ["0"] + [f"{pb[i]:.0e}" for i in ticks[1:]], rotation=60)
    b.set(yscale="log", xlabel="rain rate P", ylabel="cells", title=f"rain (p_th = {p.p_th:g})")
    fig.tight_layout()
    return _save(fig, plt, out / "hist.png")


def _annotate(
    ax: Any, storms: Sequence[tuple[int, float, float, float]], length: float, store: list[Any]
) -> None:
    for artist in store:
        artist.remove()
    store.clear()
    from matplotlib.patches import Circle

    for sid, x, y, r in storms:
        store.append(ax.add_patch(Circle((x, y), r, fill=False, ec=INK, lw=0.8)))
        store.append(ax.text(x, y, str(sid), ha="center", va="center", fontsize=6, color=INK))


def snapshots(
    out: Path, plt: Any, frames: dict[int, dict[str, Any]], times: Sequence[int], p: Any
) -> list[str]:
    """PNG snapshots (h1 - H1, zeta/f0, rain, with storm ids) at the given times, in ``frames/``."""
    (out / "frames").mkdir(exist_ok=True)
    names = []
    for t in times:
        fr = frames.get(t)
        if fr is None:
            continue
        if p.channel:
            fig, axes = plt.subplots(3, 1, figsize=(9, 3 * 9 * p.Ly / p.L + 2.4))
        else:
            fig, axes = plt.subplots(1, 3, figsize=(12, 4.1))
        ext = (0, p.L, 0, p.Ly)
        specs = (
            ("h1 - H1", fr["h1"], "RdBu_r", 8.0),
            ("zeta_1 / f0 (cyclones: red north, blue south)", fr["zeta"], "RdBu_r", 1.0),
            ("rain P", fr["P"], "Blues", float(max(np.percentile(fr["P"], 99.9), 1e-6))),
        )
        for ax, (title, img, cmap, lim) in zip(axes, specs, strict=True):
            vmin = 0.0 if cmap == "Blues" else -lim
            im = ax.imshow(
                img, origin="lower", extent=ext, cmap=cmap, vmin=vmin, vmax=lim, interpolation="nearest"
            )
            ax.set(title=f"{title}, step {t}", xlabel="x", ylabel="y")
            ax.grid(False)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            _annotate(ax, fr["storms"], p.L, [])
        fig.tight_layout()
        path = out / "frames" / f"t{t:07d}.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        names.append(f"frames/{path.name}")
    return names


def _gif(
    out: Path,
    plt: Any,
    name: str,
    times: Sequence[int],
    images: Sequence[np.ndarray],
    storms: Sequence[Sequence[tuple[int, float, float, float]]] | None,
    *,
    cmap: str,
    vmin: float,
    vmax: float,
    title: str,
    extent: tuple[float, float, float, float],
    length: float,
) -> str:
    from matplotlib.animation import PillowWriter

    path = out / name
    aspect = (extent[3] - extent[2]) / (extent[1] - extent[0])
    for dpi in (80, 60, 45, 32):
        fig, ax = plt.subplots(figsize=(4.4, 4.2) if aspect > 0.9 else (6.4, 6.4 * aspect + 1.0))
        im = ax.imshow(
            images[0], origin="lower", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest"
        )
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        ttl = ax.set_title(title)
        store: list[Any] = []
        writer = PillowWriter(fps=8)
        with writer.saving(fig, str(path), dpi):
            for k, t in enumerate(times):
                im.set_data(images[k])
                ttl.set_text(f"{title}, step {t}")
                if storms is not None:
                    _annotate(ax, storms[k], length, store)
                writer.grab_frame()
        plt.close(fig)
        if path.stat().st_size < GIF_MAX_BYTES:
            break
    return name


def write_movies(
    out: Path, plt: Any, frames: dict[int, dict[str, Any]], times: Sequence[int], p: Any, cap: int | None
) -> list[str]:
    """``h1.gif``, ``vort.gif``, ``rain.gif`` and, with tracers, ``tracers_<band>.gif`` over ``times`` (8
    fps)."""
    names: list[str] = []
    ts = [t for t in times if t in frames]
    if not ts:
        return names
    ext = (0, p.L, 0, p.Ly)
    storms = [frames[t]["storms"] for t in ts]
    rain_hi = float(max(np.percentile(np.stack([frames[t]["P"] for t in ts]), 99.9), 1e-6))
    names.append(
        _gif(
            out,
            plt,
            "h1.gif",
            ts,
            [frames[t]["h1"] for t in ts],
            None,
            cmap="RdBu_r",
            vmin=-8,
            vmax=8,
            title="h1 - H1",
            extent=ext,
            length=p.L,
        )
    )
    names.append(
        _gif(
            out,
            plt,
            "vort.gif",
            ts,
            [frames[t]["zeta"] for t in ts],
            storms,
            cmap="RdBu_r",
            vmin=-1.0,
            vmax=1.0,
            title="zeta_1 / f0",
            extent=ext,
            length=p.L,
        )
    )
    names.append(
        _gif(
            out,
            plt,
            "rain.gif",
            ts,
            [frames[t]["P"] for t in ts],
            None,
            cmap="Blues",
            vmin=0.0,
            vmax=rain_hi,
            title="rain P",
            extent=ext,
            length=p.L,
        )
    )
    if cap is not None and all(frames[t]["maps"] is not None for t in ts):
        for b, nm in enumerate(("hz0", "hz1_11", "hz12")):
            maps = [frames[t]["maps"][b] for t in ts]
            names.append(
                _gif(
                    out,
                    plt,
                    f"tracers_{nm}.gif",
                    ts,
                    maps,
                    None,
                    cmap="Blues",
                    vmin=0,
                    vmax=cap,
                    title=f"tracers per cell, {nm}",
                    extent=ext,
                    length=p.L,
                )
            )
    return names


def write_all(
    out: Path,
    res: Any,
    metrics: dict[str, Any],
    arrays: dict[str, Any],
    *,
    movies: bool = True,
    progress: Callable[[str], None] | None = None,
) -> list[str]:
    """Every figure, snapshot and movie of a study; returns the file names written (relative to ``out``)."""
    say = progress or (lambda s: None)
    plt = _setup()
    p, st = res.p, res.st
    names: list[str] = []
    names.append(lat_hist(out, plt, arrays["lat"], p.L))
    names.append(counts(out, plt, arrays))
    names.append(km_plot(out, plt, arrays["km"], metrics["criteria"]["lifetime"]["detail"]["median_text"]))
    names.append(forecast_plot(out, plt, arrays["forecast"]))
    names.append(drift_plot(out, plt, arrays["drift"]))
    occ = occupancy_plot(out, plt, res.rec.rows, arrays["occupancy"], arrays.get("occupancy_floor_held"))
    if occ:
        names.append(occ)
    names.append(
        jets_plot(
            out,
            plt,
            arrays["rows_y"],
            arrays["zonal_mean"],
            p.Ly,
            arrays["lat"]["equators"],
            arrays["lat"]["poles"],
        )
    )
    names.append(hist_plot(out, plt, metrics["diagnostics"]["histograms"], p))
    say("writing snapshots")
    names += snapshots(out, plt, res.rec.frames, st.snapshot_times(), p)
    if movies:
        a, b = st.movie_range()
        times = [t for t in sorted(res.rec.frames) if a <= t <= b and t % st.frames_every == 0]
        say(f"writing movies ({len(times)} frames)")
        cap = arrays["occupancy"]["capacity"] if arrays["occupancy"] else None
        names += write_movies(out, plt, res.rec.frames, times, p, cap)
    return names
