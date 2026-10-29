"""Helpers of the built-structures tests (DESIGN §38)."""

from __future__ import annotations

from dataclasses import replace

import mlx.core as mx
import numpy as np

from alife.config import Config, dims, load_config
from alife.state import counter_index, init_state, to_numpy
from alife.step import make_step
from helpers import set_output_bias

PRESET = {2: "tiny", 3: "tiny3d"}
NAMES = (
    "BUILD_UNITS",
    "MAT_WASTED",
    "MAT_DECAYED",
    "MAT_TOTAL",
    "MAT_CELLS",
    "SHIELDED_ATTACKS",
    "TRAPPED",
    "APERTURE_VIOLATIONS",
    "SLOWED",
    "MAT_CLOSED",
)


def cfg_for(nd: int, *extra: str, compile: bool = True) -> Config:
    """tiny / tiny3d, float32 weights, structures on; ``extra`` are ``--set`` overrides."""
    cfg = load_config(
        preset=PRESET[nd], overrides=["brain.weight_dtype=float32", "structures.enabled=true", *extra]
    )
    return cfg if compile else replace(cfg, run=replace(cfg.run, compile=False))


def builders(cfg: Config, bias: float = 4.0, *, zero_weights: bool = False):
    """A fresh state whose alive rows all have the output bias ``bias`` on ``build`` (4: near full output).

    ``zero_weights`` also zeroes the brains, so the output is exactly the bias for every row (``units_for``)
    and nobody moves or reproduces.
    """
    s = init_state(cfg)
    if zero_weights:
        s = s._replace(W1=mx.zeros_like(s.W1), W2=mx.zeros_like(s.W2))
    return set_output_bias(s, cfg, {dims(cfg).lay.OUT_BUILD: bias})


def units_for(cfg: Config, bias: float = 4.0) -> int:
    """The units a zero-weight builder with output bias ``bias`` deposits (the bias is clipped to the weight
    clip, tanh'd and floored)."""
    b = max(np.tanh(np.float32(min(bias, cfg.brain.weight_clip))), 0.0)
    return int(np.floor(np.float32(b) * np.float32(cfg.structures.build_max)))


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


def np_mat(s) -> np.ndarray:
    return to_numpy(s.mat).astype(np.int64)
