"""Synthetic panels with planted effects (DESIGN §44.9): fixtures for the statistics tests and the demo.

``synth_panels`` returns the panels of the 18-arm design (``U0..U5``, ``L_<f>``, ``I_<f>``, ``T0``, ``T5``)
over several seeds. Every arm-seed series of block values is unit-variance AR(1) noise around a level; the
level of ``U_i`` is the sum of the planted effects of the first ``i`` features, ``L_f`` and ``I_f`` are
``U_top`` minus that feature's effect, the twins ``T0``, ``T5`` repeat ``U0``, ``U_top`` with fresh noise. An
effect ``d`` is in units of the sd of a paired difference (so the planted ``d_z`` of a contrast is ``d``).
Deterministic given ``rng``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

from alife.analysis.emergence.panel import endpoint_entry, new_panel
from alife.analysis.emergence.prereg import BUILDUP_ORDER, ENDPOINTS
from alife.analysis.emergence.scorecards import FEATURES, ScorecardResult


def _ar1(rng: np.random.Generator, n: int, rho: float) -> np.ndarray:
    e = rng.standard_normal(n)
    x = np.empty(n)
    x[0] = e[0]
    s = math.sqrt(1.0 - rho * rho)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + s * e[i]
    return x


def arm_list(order: Sequence[str] = BUILDUP_ORDER) -> list[str]:
    n = len(order)
    return [
        *(f"U{i}" for i in range(n + 1)),
        *(f"L_{f}" for f in order),
        *(f"I_{f}" for f in order),
        "T0",
        f"T{n}",
    ]


def synth_panels(
    rng: np.random.Generator,
    *,
    effects: Mapping[tuple[str, str], float] | None = None,
    seeds: Sequence[int] = (0, 1, 2),
    order: Sequence[str] = BUILDUP_ORDER,
    n_blocks: int = 8,
    rho: float = 0.3,
    prereg_hash: str | None = None,
    scorecards: bool = True,
) -> list[dict]:
    eff = dict(effects or {})
    shift = math.sqrt(2.0)  # sd of a paired difference of two unit-variance series
    n = len(order)

    def level(arm: str, e: str) -> float:
        d = {f: eff.get((f, e), 0.0) * shift for f in order}
        if arm.startswith("U"):
            return sum(d[order[i]] for i in range(int(arm[1:])))
        if arm.startswith("T"):
            return sum(d[order[i]] for i in range(int(arm[1:])))
        top = sum(d.values())
        feat = arm[2:]
        return top - d[feat]  # L_f and I_f: the feature's effect removed (the inert twin keeps cost only)

    panels = []
    for seed in seeds:
        for arm in arm_list(order):
            ep = {}
            for e in ENDPOINTS:
                b = level(arm, e) + _ar1(rng, n_blocks, rho)
                ep[e] = endpoint_entry(e, value=float(b.mean()), blocks=b.tolist())
            cards = []
            if scorecards and (arm == f"U{n}" or arm.startswith("I_")):
                for f in FEATURES:
                    on = arm == f"U{n}" or arm != f"I_{f}"
                    val = float(any(eff.get((f, e), 0.0) > 0 for e in ENDPOINTS) and on)
                    cards.append(
                        ScorecardResult(f, "synthetic", None, "synthetic", {"primary": val}, provisional=True)
                    )
            panels.append(
                new_panel(
                    run=f"synth_{arm}_s{seed}",
                    arm=arm,
                    seed=seed,
                    config_hash="synthetic",
                    fork_step=500000,
                    windows=[(0, 10000)],
                    prereg_hash=prereg_hash,
                    endpoints=ep,
                    scorecards=cards,
                )
            )
    return panels
