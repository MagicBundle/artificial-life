"""Built structures off is the engine without them (DESIGN §38, §43.9).

The trajectory-level proof is the identity matrix against a fresh ``git archive`` of the base commit
(``tests/identity_matrix.py``); these are the in-tree guards that nothing of the feature exists in an off
config.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from alife import config, ext
from alife.config import dims, load_config
from alife.state import (
    N_COUNTERS,
    CounterIdx,
    State,
    State3,
    counter_blocks,
    init_state,
    n_counters,
    state_cls,
)
from alife.step import make_step

ALL_PRESETS = config.PRESETS


@pytest.mark.parametrize("preset", ["tiny", "tiny3d"])
def test_an_off_world_has_no_structures_footprint(preset: str) -> None:
    cfg = load_config(preset=preset, overrides=["brain.weight_dtype=float32"])
    d = dims(cfg)
    assert "structures" not in ext.active(cfg) and ext.layout_features(cfg) == ""
    assert (d.lay.IN_MATERIAL, d.lay.OUT_BUILD) == (None, None)
    assert ((69, 35) if preset == "tiny" else (84, 39)) == (d.lay.D_IN, d.lay.D_OUT)
    assert state_cls(cfg) is (State if preset == "tiny" else State3)  # the exact class: no `mat`
    s = init_state(cfg)
    assert "mat" not in type(s)._fields
    s1, out = make_step(cfg)(s)
    mx.eval(s1, out)
    nd = d.nd
    assert out.counters.shape == (n_counters(nd, cfg.weather.enabled),) and counter_blocks(cfg) == ()
    assert len(CounterIdx) == N_COUNTERS


def test_every_preset_overlay_pair_without_build_has_no_structures_section_in_its_file() -> None:
    for preset in ALL_PRESETS:
        cfg = load_config(preset=preset)
        assert not cfg.structures.enabled
        assert "structures" not in config.config_to_dict(cfg, omit_defaults=True)
