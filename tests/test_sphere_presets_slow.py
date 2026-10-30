"""The S7 gate of the sphere series, per preset (DESIGN §46.9; docs/SPHERE.md 10): 3 000 compiled steps of
``mini3s`` and of ``64k3s`` from their founders close the ledger on every host check (the 16 terms and every
invariant, every 250 steps), never overflow a hash cell, keep the diag word at 0 and keep the population
inside its band. The band is the founders' boom and the crash after it, measured in the calibration of S7a
(a boom to 1.5 times ``n_init`` within 100 steps, a trough at 0.8 to 0.93 of it, 0.83 to 0.95 at step 3 000):
from 0.6 to 1.75 times ``n_init`` all the way, and 0.6 to 1.3 times at the end. A third case runs
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
    assert 0.6 * n_init <= r["alive_min"] and r["alive_max"] <= 1.75 * n_init, (
        r["alive_min"],
        r["alive_max"],
    )
    assert 0.6 * n_init <= r["alive"] <= 1.3 * n_init and r["alive_max"] < 0.75 * cfg.population.n_max
