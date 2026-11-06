"""The S7 gate of the sphere series, per preset (DESIGN §46.9; docs/SPHERE.md 10): 3 000 compiled steps of
``mini3s`` and of ``64k3s`` from their founders close the ledger on every host check (the 16 terms and every
invariant, every 250 steps), never overflow a hash cell, keep the diag word at 0 and keep the population
inside its band. The band is the founders' boom and the crash after it, measured in the calibration of S7a
(a boom to 1.5 to 1.7 times ``n_init`` within 100 steps, a trough at 0.8 to 0.93 of it, 0.83 to 1.1 at step
3 000): from 0.6 to 2 times ``n_init`` all the way, and 0.6 to 1.3 times at the end. A third case runs
``mini3s`` with the five features and the weather on (the source run of a planet).

Each runs in a cold child process (``tests/sphere_run.py``, the compile-collision fallback of
docs/SPHERE.md 10.1). ``64k3s`` is Metal-only here: on the CPU the 65 536 agents take hours.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

import sphere_run
from alife.config import dims, load_config
from alife.state import LedgerIdx
from sphere_storms import BANDS, criteria, study

FEATURES = ["build", "bonds", "comm", "niche3d"]
CASES = {
    "mini3s": dict(preset="mini3s", overlays=[], overrides=[]),
    "64k3s": dict(preset="64k3s", overlays=[], overrides=[]),
    "mini3s_features_weather": dict(
        preset="mini3s", overlays=[*FEATURES, "stormsph22"], overrides=["arms.enabled=true"]
    ),
}


def _run(case: str, tmp_path, steps: int = 3000):
    cfg = load_config(**CASES[case])
    return cfg, sphere_run.spawn("trajectory", cfg, tmp_path / "cold", steps, 250, "", timeout=20000)


@pytest.mark.slow
@pytest.mark.parametrize("case", list(CASES))
def test_three_thousand_steps_of_a_preset_close_the_ledger_overflow_nothing_and_keep_the_band(
    case: str, tmp_path
) -> None:
    if case == "64k3s" and mx.default_device().type != mx.gpu:
        pytest.skip("64k3s is a Metal gate: 65 536 agents for 3 000 steps take hours on the CPU")
    cfg, r = _run(case, tmp_path)
    d = dims(cfg)
    n_init = cfg.population.n_init
    assert r["problems"] == [], r["problems"][
        :5
    ]  # full_check: the per-step ledger closure, ranges, invariants
    assert r["diag"] == 0
    assert (
        r["overflow"] == 0 and r["max_occupancy"] <= 0.75 * d.c
    )  # no hash cell over capacity, a margin of a quarter
    ledger = r["ledger"]
    sources = ledger[LedgerIdx.INJECT_REGEN] + ledger[LedgerIdx.INJECT_IMMIGRATION]
    sinks = (
        sum(ledger[i] for i in range(LedgerIdx.HEAT_BASAL, LedgerIdx.HEAT_OVERFLOW + 1))
        + ledger[LedgerIdx.HEAT_BUOYANCY]
    )
    assert (
        sources > 0 and sinks > 0 and ledger[LedgerIdx.HEAT_BUOYANCY] > 0
    )  # the 16-term ledger is exercised
    assert 0.6 * n_init <= r["alive_min"] and r["alive_max"] <= 2.0 * n_init, (
        r["alive_min"],
        r["alive_max"],
    )
    assert 0.6 * n_init <= r["alive"] <= 1.3 * n_init and r["alive_max"] < 0.75 * cfg.population.n_max


# The STORMS_W0 criteria on each calibrated planet (DESIGN §46.6, S4c): the weather alone from the rest
# state for ``steps`` steps (the engine's chain, a detection every 50 steps), the statistics over the window
# after the first ``SPINUP`` steps, for each of ``seeds``. ``steps`` is modest: the calibration is 100 000
# steps, every 25. ``births`` is the least number of named storms that must be born in the window: the
# storms of a live weather come and go (the measured births in the window are 2 to 6 per 30 000 steps on
# ``64k3s``, 0 to 6 on ``mini3s`` (seed 0 is its quiet seed: 0 to 1, hence seed 1 here) and 5 to 11 on
# ``tiny3s``), those of a frozen one do not, and a frozen weather passes the count and the lifetime (the
# Kaplan-Meier curve never falls).
#
# The three extra-large planets (I101) are read as they were calibrated: the window after a 20 000-step
# spin-up of 80 000 steps, a detection every 25, the band of ``BANDS`` (the same storm density as the 64k3s),
# seed 0, and a lower in-band share: the five calibration seeds of the chosen overlays gave 0.92 / 0.60 /
# 0.88 / 0.88 / 0.80 (128k3s), 1.000 on all (256k3s) and 0.77 / 0.76 / 0.95 / 0.82 / 0.70 (576k3s), so
# ``in_band`` is a little under seed 0's value. These cases take about 20 minutes (128k3s) to an hour
# (576k3s) each on the M5 Pro and under 10 GB.
STORM_CASES = {
    "64k3s": dict(overlay="stormsph", steps=60_000, seeds=(0,), births=2),
    "mini3s": dict(overlay="stormsph22", steps=45_000, seeds=(1,), births=2),
    "tiny3s": dict(overlay="stormsph16", steps=45_000, seeds=(0, 1), births=2),
    "128k3s": dict(
        overlay="stormsph128", steps=80_000, seeds=(0,), births=2, spinup=20_000, tick=25, in_band=0.85
    ),
    "256k3s": dict(
        overlay="stormsph256", steps=80_000, seeds=(0,), births=2, spinup=20_000, tick=25, in_band=0.95
    ),
    "576k3s": dict(
        overlay="stormsph576", steps=80_000, seeds=(0,), births=2, spinup=20_000, tick=25, in_band=0.70
    ),
}
SPINUP = 15_000  # the storms of every planet have formed by then (the measured spin-up is 10 000 to 14 000)


@pytest.mark.slow
@pytest.mark.parametrize("name", list(STORM_CASES))
def test_a_calibrated_planet_holds_two_to_six_named_storms_per_hemisphere_that_live_past_3000_steps(
    name: str,
) -> None:
    """The criteria of STORMS_W0 on the sphere (S4c): 2 to 6 named storms in each hemisphere (the band of
    ``BANDS`` on the extra-large planets) at 90 % of the detection times of the window or more (``in_band``
    where a case sets it), and a named-storm lifetime (the Kaplan-Meier median, or the
    longest age while the curve stays above one half) of at least 3 000 steps, read where it can be read
    (``median_reliable``); and a living weather: storms born in the window, a kinetic energy and a largest
    wind that vary over it. The exact numbers of the calibration runs are in DESIGN §46.6; a weaker pool, a
    larger one, or a retuned coupling falls out of the band (a 28 x 18 pool on ``mini3s`` and a 20 x 14 one
    on ``tiny3s`` hold no storm at all), and a weather that does not evolve fails the last two."""
    case = STORM_CASES[name]
    if "spinup" in case and mx.default_device().type != mx.gpu:
        pytest.skip(
            f"{name} is a Metal gate: 100 000 weather steps of the extra-large planet take hours on the CPU"
        )
    lo, hi = BANDS[name]
    cfg = load_config(preset=name, overlays=[case["overlay"]])
    for seed in case["seeds"]:
        rec = study(
            cfg, case["steps"], case.get("spinup", SPINUP), seed=seed, tick=case.get("tick", 50), lo=lo, hi=hi
        )
        c = criteria(rec, lo, hi)
        assert c["in_range_both"] >= case.get("in_band", 0.9), (seed, c)
        assert lo <= c["north_mean"] <= hi and lo <= c["south_mean"] <= hi, (seed, c)
        assert c["lifetime_pass"] and c["lifetime"] >= 3000, (seed, c)
        # a short window cannot always read the median (few storms at risk where it falls): the lifetime is
        # then read on detections (more than half of them in storms that lived 3 000 steps or more)
        assert c["lifetime_readable"] or c["long_share"] > 0.5, (seed, c)
        assert c["births"] >= case["births"], (seed, c)
        assert c["ke_cv"] > 0.02 and c["umax1_cv"] > 0.02, (seed, c)
