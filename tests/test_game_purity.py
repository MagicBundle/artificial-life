"""§28.7 purity: observing never changes the trajectory (G1).

With ``observe`` on the step returns the observer channel next to the same ``StepOut``, and a consumer folds
the channel into per-row accumulators in the same ``mx.eval`` as the state (the runner's eval pattern). The
state digest and every ``StepOut`` field must equal those of a run with observers off, bit for bit, at every
step: 2D and 3D, compiled and eager, float32 and float16 weights, the director's static branches off and on.
This is an empirical gate (the channel reads only arrays that are already kernel boundaries, or integers,
§28.7); it runs on the CPU in the suite and on Metal before merge.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from alife.config import Config, load_config
from alife.game.observe import EVAL_BIT, ObsIn, evaluate, init_obs, obs_in, update
from alife.game.rules import RULES_V1
from alife.state import CounterIdx as C
from alife.state import State, init_state, state_digest, to_numpy
from alife.step import Acts, StepOut, make_step

COMPILED_STEPS = 60
EAGER_STEPS = 20


class _Observer:
    """The real observer update (``game.observe``) driven as the runner drives it: ``update`` after every step
    and ``evaluate`` at multiples of ``every``, lazily, so that its graph is evaluated in the same
    ``mx.eval`` as the state. It reads the whole channel and the post-step state fields it needs. It runs
    eagerly: the step's fusion is decided by the step's own compile, which the observer's ops cannot enter."""

    def __init__(self, state: State, cfg: Config) -> None:
        self.cfg, self.rs = cfg, RULES_V1
        self.obs = init_obs(obs_in(state), cfg)

    def update(self, state: State, acts: Acts, t: int) -> None:
        self.obs = update(self.obs, obs_in(state), acts, self.cfg, self.rs)
        if t % self.rs.every == 0:
            self.obs = evaluate(self.obs, self.cfg, self.rs)

    def arrays(self) -> tuple[mx.array, ...]:
        return tuple(self.obs)


def _same(a, b) -> bool:
    fa, fb = tree_flatten(a), tree_flatten(b)
    pairs = zip(fa, fb, strict=True)
    return [k for k, _ in fa] == [k for k, _ in fb] and all(
        x.dtype == y.dtype and np.array_equal(to_numpy(x), to_numpy(y)) for (_, x), (_, y) in pairs
    )


def _config(d3: bool, dtype: str, director: bool) -> Config:
    overrides = [f"brain.weight_dtype={dtype}", *(["director.enabled=true"] if director else [])]
    return load_config(preset="tiny3d" if d3 else "tiny", overrides=overrides)


def _levers(s: State, cfg: Config) -> State:
    """Nonzero director levers: an immigration request and a half-strength catastrophe over the centre.

    Both are set from the config alone, so the observing and the control state get identical edits.
    """
    half = cfg.world.size / 2
    return s._replace(
        env_imm_request=mx.array(16, dtype=mx.int32),
        env_cat_center=mx.array([half, half], dtype=mx.float32),
        env_cat_radius=mx.array(40.0, dtype=mx.float32),
        env_cat_kill_p=mx.array(0.5, dtype=mx.float32),
    )


# 3D, weight dtype, director on, compiled. Eager mode has no fusion, so only the float32 director-off arms are
# run eagerly. Every arm takes 1 to 4 s on the CPU, so none is marked slow.
ARMS = [
    pytest.param(False, "float32", False, True, id="2d-f32-off"),
    pytest.param(True, "float32", False, True, id="3d-f32-off"),
    pytest.param(False, "float32", False, False, id="2d-f32-off-eager"),
    pytest.param(True, "float32", False, False, id="3d-f32-off-eager"),
    pytest.param(False, "float16", False, True, id="2d-f16-off"),
    pytest.param(False, "float32", True, True, id="2d-f32-on"),
    pytest.param(False, "float16", True, True, id="2d-f16-on"),
    pytest.param(True, "float16", True, True, id="3d-f16-on"),
]


@pytest.mark.parametrize(("d3", "dtype", "director", "compiled"), ARMS)
def test_observing_does_not_change_the_trajectory(
    d3: bool, dtype: str, director: bool, compiled: bool
) -> None:
    cfg = _config(d3, dtype, director)
    steps = COMPILED_STEPS if compiled else EAGER_STEPS
    off = make_step(cfg, compile=compiled)
    on = make_step(cfg, compile=compiled, observe=True)
    s_off, s_on = init_state(cfg), init_state(cfg)
    fold = _Observer(s_on, cfg)
    catastrophes = immigrants = 0
    for t in range(steps):
        if director and t in (0, steps // 2):  # the one-step levers are consumed by the step: set them again
            s_off, s_on = _levers(s_off, cfg), _levers(s_on, cfg)
        s_off, o_off = off(s_off)
        mx.eval(s_off, o_off)
        s_on, o_on = on(s_on)
        fold.update(s_on, o_on.acts, t + 1)
        mx.eval(s_on, o_on.base(), *fold.arrays())  # the runner's pattern: one blocking eval per step
        where = f"step {t}"
        assert type(o_off) is StepOut and type(o_on.base()) is StepOut
        assert state_digest(s_off) == state_digest(s_on), where
        assert _same(o_off, o_on.base()), where
        c = to_numpy(o_off.counters)
        catastrophes += int(c[C.DEATHS_CATASTROPHE])
        immigrants += int(c[C.IMMIGRANTS])
    assert (
        catastrophes > 0 and immigrants > 0
    ) == director  # the director arms run the levers, the others not
    assert int(to_numpy(s_on.step)) == steps
    # the observer actually ran: it accumulated, and (past the intent warm-up) evaluated rows
    assert int(to_numpy(fold.obs.n_obs).max()) == steps and int(to_numpy(fold.obs.arch).max()) > 0
    assert (steps < RULES_V1.intent_warm) == (not (to_numpy(fold.obs.match_intent) & EVAL_BIT).any())


def test_observer_inputs_exclude_the_key() -> None:
    """The observer reads a handful of State fields and never the RNG key or the weights (§28.1)."""
    assert set(ObsIn._fields) == {
        "alive", "uid", "parent_uid", "partner", "vel", "last_harvest", "last_inflow", "last_loss",
    }  # fmt: skip
    assert "key" not in ObsIn._fields
    assert not {"W1", "W2", "pend_W1", "pend_W2", "h", "msg", "tag"} & set(ObsIn._fields)
    s = init_state(load_config(preset="tiny", overrides=["brain.weight_dtype=float32"]))
    oi = obs_in(s)
    assert all(a is getattr(s, f) for f, a in zip(ObsIn._fields, oi, strict=True))
