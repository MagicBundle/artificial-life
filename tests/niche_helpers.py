"""Helpers of the farming tests (DESIGN §41)."""

from __future__ import annotations

from dataclasses import replace

import mlx.core as mx
import numpy as np

from alife.config import Config, dims, load_config
from alife.state import counter_index, init_state, to_numpy
from alife.step import make_step
from helpers import set_output_bias

PRESET = {2: "tiny", 3: "tiny3d", "s": "tiny3s"}  # "s": the sphere world, run eager (spec 10.1 fallback)
NAMES = ("FARMERS", "FARM_YIELD", "SOIL_PUT", "SOIL_WASTED", "SOIL_DECAYED", "SOIL_TOTAL")
# the cost of a unit scales with regen_max (320 in tiny, 1700 in tiny3d?): the design point is ROI 5.5
COST = {2: 30, 3: 30}


def cfg_for(nd: int | str, *extra: str, compile: bool = True) -> Config:
    """tiny / tiny3d / tiny3s ("s"), float32 weights, farming on; ``extra`` are ``--set`` overrides. A
    sphere is never compiled in the test process (docs/SPHERE.md 10.1): its compiled step is in
    ``test_sphere_features_compiled``."""
    cfg = load_config(preset=PRESET[nd], overrides=["brain.weight_dtype=float32", "niche.farm=true", *extra])
    return cfg if compile and nd != "s" else replace(cfg, run=replace(cfg.run, compile=False))


def farmers(cfg: Config, bias: float = 4.0, *, zero_weights: bool = False):
    """A fresh state whose alive rows all have the output bias ``bias`` on ``farm_put`` (4: near full output).

    ``zero_weights`` also zeroes the brains, so the output is exactly the bias for every row and nobody moves
    or reproduces."""
    s = init_state(cfg)
    if zero_weights:
        s = s._replace(W1=mx.zeros_like(s.W1), W2=mx.zeros_like(s.W2))
    return set_output_bias(s, cfg, {dims(cfg).lay.OUT_FARM_PUT: bias})


def counters(cfg: Config, out) -> dict[str, int]:
    c = to_numpy(out.counters)
    return {n: int(c[counter_index(cfg, n)]) for n in NAMES}


def run(cfg: Config, s, steps: int):
    """Yield ``(prev, s, out, row)`` per compiled step."""
    fn = make_step(cfg)
    for _ in range(steps):
        prev = s
        s, out = fn(s)
        mx.eval(s, out)
        yield prev, s, out, counters(cfg, out)


def np_soil(s) -> np.ndarray:
    return to_numpy(s.soil).astype(np.int64)
