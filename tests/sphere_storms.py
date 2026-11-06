"""A weather-alone storm study on a planet (DESIGN §46.6, S4c): the engine's own weather chain from the rest
state and the storm statistics of ``alife.analysis.w0_sphere`` (the sphere's tracker, named storms in real
time, the W0 count and lifetime criteria), at the cadence of ``alife play`` (a detection every 25 steps) or a
coarser one for a cheaper test.

``study`` is the instrument of the slow gates in ``tests/test_sphere_presets_slow.py``: one run, returning the
numbers the criteria of STORMS_W0 are read from.
"""

from __future__ import annotations

from typing import Any

from alife.analysis import w0_sphere
from alife.analysis.w0_sphere import LIFETIME_MIN
from alife.config import Config

# The named-storm count band per hemisphere by planet (I100): the same storm density as the 64k3s, whose pools
# per hemisphere are 3, so 4, 6 and 9 pools on the larger planets and the 2 to 6 band in proportion.
BANDS: dict[str, tuple[int, int]] = {
    "64k3s": (2, 6),
    "mini3s": (2, 6),
    "tiny3s": (2, 6),
    "128k3s": (3, 8),
    "256k3s": (4, 12),
    "576k3s": (6, 18),
}


def study(
    cfg: Config, steps: int, spinup: int, seed: int = 0, tick: int = 50, lo: int = 2, hi: int = 6
) -> dict[str, Any]:
    """The summary of ``w0_sphere.run_study`` for ``cfg`` over ``steps`` steps, the statistics over the
    window from ``spinup`` on (the study's arrays are dropped), the count band ``lo`` to ``hi``."""
    st = w0_sphere.StudySettings(
        steps=steps, spinup=spinup, seed=seed, tick=tick, exposure=False, count_lo=lo, count_hi=hi
    )
    rec = w0_sphere.run_study(cfg, st)
    return {k: v for k, v in rec.items() if not k.startswith("_")}


def criteria(rec: dict[str, Any], lo: int = 2, hi: int = 6) -> dict[str, Any]:
    """The STORMS_W0 criteria on a study record: the mean and range of the named storms per hemisphere, the
    share of ticks with ``lo`` to ``hi`` (2 to 6 by default, ``BANDS`` by planet) in both hemispheres, the
    lifetime (the median of the Kaplan-Meier curve of the named storms, or the longest age observed when the
    curve stays above one half), whether it reaches 3 000 steps and whether that reading is reliable
    (``median_reliable``), and what tells a live weather from a frozen one: the births and deaths of named
    storms in the window and the variation of the kinetic energy and of the largest wind over it. A frozen
    weather has storms that never die, a count always in range and no variation. The record must have been
    counted with the same band (``run_study`` with ``count_lo`` and ``count_hi``)."""
    band = tuple(rec.get("count_band", (2, 6)))  # a record from before I100 used 2 to 6
    if band != (lo, hi):
        raise ValueError(
            f"the record was counted with the band {band}, not {(lo, hi)}: run the study with it"
        )
    named = rec["counts"]["named"]
    life = rec["lifetime"]["named_hindsight"]
    track = rec["tracks"]
    return {
        "band": (lo, hi),
        "north_mean": named["north"]["mean"],
        "south_mean": named["south"]["mean"],
        "north_range": (named["north"]["min"], named["north"]["max"]),
        "south_range": (named["south"]["min"], named["south"]["max"]),
        "in_range_both": named["in_range_both"],
        "lifetime_pass": life["pass"],
        "lifetime": life["median"] if life["median"] is not None else life["max_age"],
        "median_reliable": life["median_reliable"],
        # the median may be unreadable (a thin risk set) and the criterion still decided: a median at or after
        # an age where the risk set is thin, which is itself 3 000 or more, is 3 000 or more
        "lifetime_readable": bool(
            life["median_reliable"] or (life["thin_age"] is not None and life["thin_age"] >= LIFETIME_MIN)
        ),
        # the criterion read on detections instead of storms: the share of the window's detections that belong
        # to a storm observed living 3 000 steps or more (above one half is a pass)
        "long_share": life["share_detections_in_long_storms"],
        "tracks": track["named"],
        "births": track["named_births_in_window"],
        "deaths": track["named_deaths_in_window"],
        "ke_cv": rec["climate_cv"]["ke"],
        "umax1_cv": rec["climate_cv"]["umax1"],
    }
