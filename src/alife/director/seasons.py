"""The fall that the seasons themselves force on the population (DESIGN §26.5, rules version 2).

Under seasons the lattice's regeneration, and with it the capacity the population tracks, swings every
period. A population that follows it falls by a large fraction between a peak and the next trough without
anything being wrong, which the rule tier's fall criterion (``fall`` > ``risk_fall`` within
``risk_fall_window`` steps) cannot tell from a crash. This module models the forced part so that the
extinction-risk flag can be set on the **excess** only.

The model, from how the engine applies a season (``fields.fertility``, ``fields.season_weights``):

- A patch's weight is ``1 + A sin(2 pi t / P + phi_p)`` and the fertility is
  ``clip(base + sum_p gain_p w_p G_p, 0, 1)``, so the regeneration ``floor(regen_max fertility)`` is *not*
  proportional to the weight: the base fertility and the clip at 1 damp and skew it (on
  ``runs/eval_s0_C``, amplitude 0.6, the capacity swings by 0.40 to 0.45 of its mean, not 0.6, and with a
  second harmonic). The lattice is therefore **sampled through the engine** over one cycle
  (``metrics.seasonal_capacity_table``): the table gives the capacity at ``len(table)`` equally spaced
  phases relative to its value without seasons. It is a function of the phase only (also with
  ``season_mode = per_patch``, where the patches' sines cancel in the total), so it serves every period.
- The population is taken to follow the capacity without lag, and the capacity of a regime of another
  amplitude ``A`` to be the table's deviation from 1 scaled by ``A / table amplitude`` (exact for the one
  regime of an ordinary run; first order when the amplitude changed within the fall window).
- ``forced_fall`` applies the vital sign's own ``fall`` formula to that modelled capacity:
  ``1 - (lowest capacity of the current window) / (highest window mean over the fall window)``, each window
  under the regime in force in it. It is the fall the seasons force at *this* phase (zero at a peak, the
  full swing at a trough), and includes the jump of a regime that has just been switched on.
- The regeneration multiplier (``env.regen_mult``, a regen pulse's lever) scales the capacity of a window
  and is recorded per window in the rows' ``env``. The exemption never exceeds what the seasons and the
  multiplier **together** force: ``forced_fall`` is the lower of the fall with the seasons alone and the
  fall with each window's multiplier. A pulse begun after the last peak raises the capacity of the current
  window against the peak's, so the seasons' share is overstated and the second is the lower; a pulse that
  has ended drops the capacity back, which no season forced, and the first is the lower (the drop stays
  flagged, as in version 1). With the multiplier 1 in every window the two are the same number.
- The **excess** divides the forced ratio out of the observed one: ``1 - (1 - fall) / (1 - forced)``. A fall
  of 0.50 where the seasons force 0.42 is an excess of 0.14; a collapse to a tenth of the reference where
  they force 0.42 is an excess of 0.84.

Everything is a pure function of the table and the regimes (amplitude and period of each window, read from
the metrics rows' ``env``), so the tracker's fold stays deterministic.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Final, NamedTuple

import numpy as np

TABLE_SIZE: Final = 32  # phases per cycle in the capacity table
WINDOW_PARTS: Final = 16  # equal parts of a rule window; the modelled capacity is read at their ends

Regime = tuple[float, int]  # (season amplitude, season period) in force in a window; None = no seasons


class CapacityShape(NamedTuple):
    """The capacity over one cycle at the phases ``j / len(table)``, relative to its value without
    seasons, for seasons of amplitude ``amp`` (``metrics.seasonal_capacity_table``)."""

    amp: float
    table: tuple[float, ...]


def regen_mult_of(env: Mapping[str, Any] | None) -> float:
    """The regeneration multiplier a metrics row's ``env`` summary records; 1.0 if it records none (or
    something that is not a positive finite number, which the engine's lever range does not allow)."""
    mult = env.get("regen_mult") if env else None
    ok = isinstance(mult, int | float) and not isinstance(mult, bool) and math.isfinite(mult) and mult > 0
    return float(mult) if ok else 1.0


def regime_of(env: Mapping[str, Any] | None) -> Regime | None:
    """The ``(amplitude, period)`` a metrics row's ``env`` summary records, or None if seasons are off (or the
    row has no ``env``)."""
    if not env:
        return None
    amp, period = env.get("season_amp") or 0.0, env.get("season_period") or 0
    return (float(amp), int(period)) if amp > 0 and period > 0 else None


def capacity(shape: CapacityShape, regime: Regime | None, steps: np.ndarray) -> np.ndarray:
    """The modelled capacity at ``steps`` (floats allowed) under ``regime``, relative to no seasons."""
    if regime is None:
        return np.ones(np.shape(steps), dtype=np.float64)
    amp, period = regime
    table = np.asarray(shape.table, dtype=np.float64)
    phase = np.mod(np.asarray(steps, dtype=np.float64), period) / period
    dev = np.interp(phase, np.arange(len(table)) / len(table), table, period=1.0) - 1.0
    return 1.0 + (amp / shape.amp) * dev


def window_stats(
    shape: CapacityShape, regime: Regime | None, lo: float, hi: float, mult: float = 1.0
) -> tuple[float, float]:
    """The mean and the lowest value of the modelled capacity over the window ``(lo, hi]``, from its values
    at the ends of ``WINDOW_PARTS`` equal parts (the mean by the trapezoid rule). The ends are read because
    the population's window minimum is a per-step minimum, which on a rising or falling stretch sits at an
    edge. ``mult`` is the regeneration multiplier the window ran with (the capacity is proportional to it)."""
    c = capacity(shape, regime, np.linspace(lo, hi, WINDOW_PARTS + 1))
    if mult != 1.0:
        c = c * mult
    return float((c.sum() - 0.5 * (c[0] + c[-1])) / WINDOW_PARTS), float(c.min())


def forced_fall(
    shape: CapacityShape,
    windows: Sequence[tuple[Regime | None, float, float] | tuple[Regime | None, float, float, float]],
    current: int = -1,
) -> float:
    """The fall the seasons force: ``1 - (lowest capacity of the current window) / (highest window mean)``.

    ``windows`` are the ``(regime, lo, hi)`` of the rule windows of the fall window, oldest first, each with
    an optional fourth item, the regeneration multiplier the window ran with (default 1); ``current`` indexes
    the one whose lowest capacity counts (the last by default). The result is the lower of the fall with the
    seasons alone and the fall with each window's multiplier (see the module docstring): never more than the
    seasons and the levers together force.
    """

    def fall(mults: Sequence[float]) -> float:
        stats = [
            window_stats(shape, regime, lo, hi, m)
            for (regime, lo, hi, *_), m in zip(windows, mults, strict=True)
        ]
        return 1.0 - stats[current][1] / max(mean for mean, _ in stats)

    levers = [w[3] if len(w) > 3 else 1.0 for w in windows]
    seasons_only = fall([1.0] * len(windows))
    return seasons_only if all(m == 1.0 for m in levers) else min(seasons_only, fall(levers))


def excess_fall(fall: float, forced: float) -> float:
    """The part of an observed ``fall`` that the seasons do not force: ``1 - (1 - fall) / (1 - forced)``,
    never below 0. With nothing forced it is ``fall`` itself, exactly; if the seasons force the capacity to
    zero there is nothing to compare with and the excess is 0."""
    if forced <= 0.0:
        return fall
    if 1.0 - forced <= 1e-9:
        return 0.0
    return max(0.0, 1.0 - (1.0 - fall) / (1.0 - forced))
