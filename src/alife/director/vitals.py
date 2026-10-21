"""Vital signs from metrics rows (DIRECTOR.md §4): window aggregates, the behaviour descriptor, the tracker.

Everything is a pure function of ``metrics.jsonl`` rows (plus the live inputs, which are not stored:
``K_est`` and, in 3D, the floor layer's ``k_floor``, §27.10), so the ``Tracker`` rebuilt from the rows on
disk equals the one a live run built (A7). Rows are aggregated over the rule window and weighted by
``row["steps"]`` (short rows exist after an interrupted run); the wall-clock keys (``wall_s``,
``steps_per_s``, ``peak_mem_gb``) are never read. A heavy key that is absent or None (an empty world, a
non-heavy row) is skipped, and a descriptor component that is missing counts as 0.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from alife.director import seasons
from alife.director.archive import Archive, Observation

if TYPE_CHECKING:  # rules imports this module; the annotation alone must not create an import cycle
    from alife.director.rules import RuleParams

# Descriptor layout; the coverage grid uses emitters_frac, frac_in_bodies_ge4 and predation_ratio.
DESCRIPTOR_NAMES = (
    "emitters_frac",
    "sharers_frac",
    "bonds_per_agent",
    "frac_in_bodies_ge4",
    "singleton_frac",
    "kill_share",
    "predation_ratio",
    "sigma_med",
    "rate_med",
    "frac_med",
)
GRID_AXES = (0, 3, 6)


@dataclass(frozen=True)
class Window:
    """Aggregates of the metrics rows since the previous rule boundary."""

    step: int  # step of the last row
    steps: int  # steps covered
    n_alive: float | None  # step-weighted mean of the rows' means
    n_alive_min: float | None  # lowest per-step population in the window
    n_alive_last: float | None  # the last row's mean population
    births: int
    deaths: int  # starved + killed + aged + catastrophe
    kill_share: float | None  # deaths_killed / deaths, catastrophe deaths in the denominator
    emitters_frac: float | None
    sharers_frac: float | None
    bonds_per_agent: float | None
    predation_ratio: float | None
    # Snapshots: the last non-None value in the window
    top_clade_share: float | None
    clades_alive: int | None
    clade_entropy: float | None
    weight_diversity: float | None
    frac_in_bodies_ge4: float | None
    singleton_frac: float | None
    bodies_largest: int | None
    sigma_med: float | None
    rate_med: float | None
    frac_med: float | None


def _wmean(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    num = den = 0.0
    for r in rows:
        if (v := r.get(key)) is not None:
            w = r.get("steps") or 1
            num, den = num + w * v, den + w
    return num / den if den else None


def _last(rows: Sequence[Mapping[str, Any]], key: str) -> Any:
    return next((v for r in reversed(rows) if (v := r.get(key)) is not None), None)


def _unit(x: float | None, scale: float = 1.0) -> float | None:
    """``x / scale`` clipped to [0, 1]; None stays None."""
    return None if x is None else min(max(x / scale, 0.0), 1.0)


def _singleton_frac(rows: Sequence[Mapping[str, Any]], n_last: float | None) -> float | None:
    hist = _last(rows, "bodies_hist")
    if not hist or not n_last:
        return None
    return _unit(hist.get("1", 0) / n_last)


def aggregate(rows: Sequence[Mapping[str, Any]]) -> Window:
    """The ``Window`` of the rows of one rule window (at least one)."""
    steps = sum(r.get("steps") or 1 for r in rows)
    killed = sum(r.get("deaths_killed", 0) for r in rows)
    deaths = killed + sum(
        r.get(k, 0)
        for r in rows
        for k in ("deaths_starved", "deaths_aged", "deaths_catastrophe", "deaths_storm")
    )
    lows = [v for r in rows if (v := r.get("n_alive_min", r.get("n_alive"))) is not None]
    n_last = rows[-1].get("n_alive")
    return Window(
        step=rows[-1]["step"],
        steps=steps,
        n_alive=_wmean(rows, "n_alive"),
        n_alive_min=min(lows) if lows else None,
        n_alive_last=n_last,
        births=sum(r.get("births", 0) for r in rows),
        deaths=deaths,
        kill_share=killed / deaths if deaths else None,
        emitters_frac=_wmean(rows, "emitters_frac"),
        sharers_frac=_wmean(rows, "sharers_frac"),
        bonds_per_agent=_wmean(rows, "mean_bonds_per_agent"),
        predation_ratio=_wmean(rows, "predation_ratio"),
        top_clade_share=_last(rows, "top_clade_share"),
        clades_alive=_last(rows, "clades_alive"),
        clade_entropy=_last(rows, "clade_entropy"),
        weight_diversity=_last(rows, "weight_diversity"),
        frac_in_bodies_ge4=_last(rows, "frac_in_bodies_ge4"),
        singleton_frac=_singleton_frac(rows, n_last),
        bodies_largest=_last(rows, "bodies_largest"),
        sigma_med=_last(rows, "sigma_q50"),
        rate_med=_last(rows, "rate_q50"),
        frac_med=_last(rows, "frac_q50"),
    )


def _log_scaled(x: float | None, lo: float, hi: float) -> float:
    if x is None or x <= 0:
        return 0.0
    return min(max((math.log(x) - math.log(lo)) / (math.log(hi) - math.log(lo)), 0.0), 1.0)


def descriptor(w: Window, p: RuleParams) -> tuple[float, ...] | None:
    """The behaviour descriptor ``b`` of a window: ten components in [0, 1] (``DESCRIPTOR_NAMES``), or None.

    Fractions are clipped to [0, 1]; bonds per agent is divided by ``p.bonds_per_agent_max`` and the predation
    ratio by ``p.predation_max``; the gene medians are log-scaled onto their fixed ranges in ``p``.
    """
    if not w.n_alive_last:
        return None

    def unit(x: float | None, scale: float = 1.0) -> float:
        return _unit(x, scale) or 0.0

    return (
        unit(w.emitters_frac),
        unit(w.sharers_frac),
        unit(w.bonds_per_agent, p.bonds_per_agent_max),
        unit(w.frac_in_bodies_ge4),
        unit(w.singleton_frac),
        unit(w.kill_share),
        unit(w.predation_ratio, p.predation_max),
        _log_scaled(w.sigma_med, *p.sigma_range),
        _log_scaled(w.rate_med, *p.rate_range),
        _log_scaled(w.frac_med, *p.frac_range),
    )


@dataclass(frozen=True)
class Boundary:
    """One rule boundary: its window, descriptor and what the archive made of it."""

    step: int
    window: Window
    b: tuple[float, ...] | None
    obs: Observation | None
    regime: seasons.Regime | None = None  # (season amplitude, period) the window ran under; None: no seasons
    regen_mult: float = 1.0  # the regeneration multiplier the window ran with (``env.regen_mult``)


@dataclass(frozen=True)
class Vitals:
    """What the rules see at a boundary (the window plus history, the archive and the live ``K_est``, with the
    floor layer's ``k_floor`` in 3D)."""

    step: int
    origin: int | None  # start of the first window this run directory ever saw
    window: Window
    n_alive: float | None
    n_alive_min: float | None
    k_est: float  # capacity estimate; 3D: occupancy-weighted (§27.10), read by the rescue size and the prompt
    n_over_k: float | None  # n_alive / k_est
    fall: float | None  # 1 − (window minimum / highest window mean) over the last ``risk_fall_window`` steps
    # ``fall`` is the observed fall in every version. From rules version 2, with seasons in force in the fall
    # window, ``fall_seasonal`` is the part of it the seasons force (at most what they and the regen pulses
    # together force) and ``fall_excess`` the rest (seasons.py);
    # the flag then reads the excess. Both are None without seasons (and under version 1): ``fall`` is read.
    extinction_risk: bool  # n_over_k_risk < ``risk_k_frac`` or the fall (its excess) > ``risk_fall``
    pop_frac_cap: float | None  # population / N_CAP
    largest_body_frac: float | None
    top_clade_share: float | None
    mono_since: int | None  # start of the current run of boundaries above ``mono_share``
    novelty: float | None
    novelty_trace: tuple[tuple[int, float | None], ...]  # (step, novelty) of the recent boundaries
    coverage: int
    new_cells_200k: int
    last_new_cell: int | None  # step the latest grid cell was first visited
    b_shift: float | None  # RMS distance between b now and b about ``shift_window`` steps ago
    # 3D only (None in 2D, which has one K): the floor layer's capacity, the K the extinction-risk test reads
    # (§27.10, owner decision 2026-10-15), and the population over it.
    k_floor: float | None = None
    n_over_k_floor: float | None = None
    fall_seasonal: float | None = None
    fall_excess: float | None = None

    @property
    def n_over_k_risk(self) -> float | None:
        """The ratio the extinction-risk test compares with ``risk_k_frac``: the floor layer's in 3D, the one
        ``K_est`` in 2D."""
        return self.n_over_k if self.k_floor is None else self.n_over_k_floor

    def compact(self) -> dict[str, Any]:
        """The legible subset written into each log entry (rounded to 4 places)."""

        def r(x: Any) -> Any:
            return round(x, 4) if isinstance(x, float) else x

        # 2D: the keys of build step 2, byte for byte. 3D adds the floor layer's K the risk test reads, and a
        # window under seasons (rules version 2) the split of the fall into the forced part and the excess.
        floor = ("k_floor", "n_over_k_floor") if self.k_floor is not None else ()
        seasonal = ("fall_seasonal", "fall_excess") if self.fall_seasonal is not None else ()
        return {
            k: r(getattr(self, k))
            for k in (
                "n_alive",
                "k_est",
                "n_over_k",
                *floor,
                "fall",
                *seasonal,
                "extinction_risk",
                "top_clade_share",
                "novelty",
                "coverage",
                "new_cells_200k",
                "b_shift",
            )
        }


class Tracker:
    """The director's fold of the metrics rows: window buffer, boundary history, archive and streaks.

    Live mode calls ``observe`` for every row and ``close`` at every rule boundary; a rebuild from disk calls
    the same two methods on the stored rows, so both produce the same tracker.
    """

    def __init__(self, p: RuleParams) -> None:
        self.p = p
        self.archive = Archive(p.novelty_k, p.novelty_exclude, p.archive_rho, p.grid_bins, GRID_AXES)
        self.rows: list[Mapping[str, Any]] = []
        self.history: list[Boundary] = []
        self.origin: int | None = None
        self.mono_since: int | None = None

    def observe(self, row: Mapping[str, Any]) -> None:
        self.rows.append(row)

    def close(self, t: int) -> Boundary:
        """Fold the buffered rows (the window ending at ``t``) into the history, archive and streaks."""
        w = aggregate(self.rows)
        # The regime and the regeneration multiplier of the window: a metrics row's ``env`` is what its
        # interval ran with, and the director changes the environment only at a boundary, so the window's last
        # row speaks for all of it.
        env = self.rows[-1].get("env")
        regime, regen_mult = seasons.regime_of(env), seasons.regen_mult_of(env)
        self.rows = []
        start = t - w.steps
        self.origin = start if self.origin is None else self.origin
        b = descriptor(w, self.p)
        obs = self.archive.observe(t, b) if b is not None else None
        mono = w.top_clade_share is not None and w.top_clade_share > self.p.mono_share
        self.mono_since = (start if self.mono_since is None else self.mono_since) if mono else None
        boundary = Boundary(t, w, b, obs, regime, regen_mult)
        kept = [h for h in self.history if h.step > t - self.p.history_keep]
        self.history = [*kept, boundary]
        return boundary

    def _fall(self, bd: Boundary) -> float | None:
        recent = [
            h.window.n_alive
            for h in self.history
            if bd.step - self.p.risk_fall_window < h.step <= bd.step and h.window.n_alive is not None
        ]
        low = bd.window.n_alive_min
        return 1.0 - low / max(recent) if recent and low is not None and max(recent) > 0 else None

    def season_amp(self, bd: Boundary) -> float:
        """The largest season amplitude of the windows in the fall window ending at ``bd`` (0: no seasons)."""
        return max(
            (
                h.regime[0]
                for h in self.history
                if h.regime is not None and bd.step - self.p.risk_fall_window < h.step <= bd.step
            ),
            default=0.0,
        )

    def _forced_fall(self, bd: Boundary, shape: seasons.CapacityShape) -> float | None:
        """The fall the seasons force over the same windows as ``_fall`` (seasons.py): the modelled capacity
        through the vital sign's own formula, each window under its own regime and regeneration multiplier
        (the lower of the fall with and without the multipliers, so a regen pulse never widens the exemption).
        None where ``_fall`` is, and where the window of ``bd`` has no mean population."""
        wins = [
            h
            for h in self.history
            if bd.step - self.p.risk_fall_window < h.step <= bd.step and h.window.n_alive is not None
        ]
        if not wins or wins[-1].step != bd.step or bd.window.n_alive_min is None:
            return None
        return seasons.forced_fall(
            shape, [(h.regime, h.step - h.window.steps, h.step, h.regen_mult) for h in wins]
        )

    def _shift(self, bd: Boundary) -> float | None:
        if bd.b is None:
            return None
        past = [h.b for h in self.history if h.step <= bd.step - self.p.shift_window and h.b is not None]
        if not past:
            return None
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(bd.b, past[-1], strict=True)) / len(bd.b))

    def vitals(
        self,
        bd: Boundary,
        k_est: float,
        n_cap: int,
        k_floor: float | None = None,
        shape: seasons.CapacityShape | None = None,
    ) -> Vitals:
        """The vital signs at boundary ``bd`` (the one ``close`` just returned) for a live ``K_est``.

        ``k_floor`` is given in 3D only (the floor layer's capacity, §27.10): the extinction-risk test then
        divides the population by it instead of by ``k_est``, which every other reader keeps. Without it (2D)
        the one ``K_est`` is both.

        ``shape`` is the capacity over a seasonal cycle (``seasons.CapacityShape``), given by a live director
        of rules version 2 when seasons are in force in the fall window. The extinction-risk test then reads
        the excess of the fall over what the seasons force (``fall_seasonal``, ``fall_excess``); ``fall``
        itself stays the observed value. Without it the test reads ``fall`` as it always did.
        """
        w, p = bd.window, self.p
        n = w.n_alive_last
        n_over_k = n / k_est if n is not None and k_est > 0 else None
        n_over_k_floor = n / k_floor if n is not None and k_floor is not None and k_floor > 0 else None
        risk_ratio = n_over_k if k_floor is None else n_over_k_floor
        fall = self._fall(bd)
        forced = self._forced_fall(bd, shape) if shape is not None and p.season_aware else None
        excess = seasons.excess_fall(fall, forced) if fall is not None and forced is not None else None
        judged = fall if excess is None else excess
        risk = (risk_ratio is not None and risk_ratio < p.risk_k_frac) or (
            judged is not None and judged > p.risk_fall
        )
        return Vitals(
            step=bd.step,
            origin=self.origin,
            window=w,
            n_alive=n,
            n_alive_min=w.n_alive_min,
            k_est=k_est,
            n_over_k=n_over_k,
            fall=fall,
            extinction_risk=risk,
            pop_frac_cap=n / n_cap if n is not None else None,
            largest_body_frac=w.bodies_largest / n if n and w.bodies_largest is not None else None,
            top_clade_share=w.top_clade_share,
            mono_since=self.mono_since,
            novelty=bd.obs.novelty if bd.obs else None,
            novelty_trace=tuple((h.step, h.obs.novelty if h.obs else None) for h in self.history),
            coverage=self.archive.coverage,
            new_cells_200k=self.archive.new_cells_since(bd.step - p.stagnation_window),
            last_new_cell=self.archive.last_new_cell,
            b_shift=self._shift(bd),
            k_floor=k_floor,
            n_over_k_floor=n_over_k_floor,
            fall_seasonal=None if excess is None else forced,
            fall_excess=excess,
        )
