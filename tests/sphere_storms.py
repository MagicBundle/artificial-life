"""A weather-alone storm count on a planet (DESIGN §46.6, S7a): the engine's own weather chain from the rest
state (``weather_spinup.advance_chain``) and the storm layer of ``alife play`` (``StormLayerS``), ticked every
100 steps; the named, active storms per hemisphere at each tick. The instrument that chose the pools of
``stormsph22`` (16 x 10 gave 0.15 and 0.04 per hemisphere, 20 x 12 gave 1.0 and 0.6, 24 x 16 2.8 and 2.9).
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from alife import weather_sph
from alife.config import Config
from alife.storms import dc_reference
from alife.storms_sph import SphereParams, StormLayerS
from alife.weather import rest_array
from alife.weather_spinup import advance_chain, spinup_key


def count_storms(cfg: Config, steps: int, seed: int, every: int = 100) -> dict[str, float | int]:
    """Mean and maximum of the named, active storms per hemisphere over the last two thirds of ``steps``, and
    the number of named tracks seen."""
    params = SphereParams.from_config(cfg)
    wx = rest_array(cfg, 0)
    key = spinup_key(mx.random.key(seed))
    ws0 = weather_sph.unpack(wx)
    mx.eval(ws0.Dc)
    layer = StormLayerS(params, dc_reference(np.array(ws0.Dc)), params.detect_every)
    counts: list[tuple[int, int]] = []
    named: set[int] = set()
    t = 0
    while t < steps:
        wx, key = advance_chain(cfg, wx, t, key, every)
        t += every
        tick = layer.tick_at(weather_sph.unpack(wx), t)
        storms = [s for s in tick.strm["storms"] if s["named"] and s["state"] == "active"]
        counts.append((sum(s["hemi"] == 0 for s in storms), sum(s["hemi"] == 1 for s in storms)))
        named |= {s["id"] for s in tick.strm["storms"] if s["named"]}
    late = np.array(counts[len(counts) // 3 :])
    return {
        "north_mean": float(late[:, 0].mean()),
        "south_mean": float(late[:, 1].mean()),
        "north_max": int(late[:, 0].max()),
        "south_max": int(late[:, 1].max()),
        "tracks": len(named),
    }
