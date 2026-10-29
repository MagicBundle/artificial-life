"""Communication, the observer PR C2 (DESIGN §39.8): ``Acts.ext.comm`` (voice, heard, audible), the observer's
retained heard half and the ``FSIG`` source values."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

import helpers
from alife import comm
from alife.config import dims, load_config
from alife.game.observer import Observer
from alife.state import counter_index, to_numpy
from alife.step import ActFlag, make_step

PRESET = {2: "tiny", 3: "tiny3d"}


def _cfg(nd: int, *extra: str):
    return load_config(
        preset=PRESET[nd],
        overrides=["brain.weight_dtype=float32", "comm.enabled=true", "comm.width=3", "comm.cost=0", *extra],
    )


def _pair(nd: int, *extra: str):
    """Two silent-weight agents 1 apart, both speaking all channels at full volume (a bias on the outputs),
    and
    a third far away."""
    cfg = _cfg(nd, *extra)
    d = dims(cfg)
    z = (5.0,) if nd == 3 else ()
    s = helpers.make_scene(
        cfg, [{"pos": (10.0, 10.0, *z)}, {"pos": (11.0, 10.0, *z)}, {"pos": (30.0, 30.0, *z)}]
    )
    bias = {d.lay.OUT_VOICE + c: 6.0 for c in range(d.voice_w)}
    return cfg, helpers.set_output_bias(s, cfg, bias)


@pytest.mark.parametrize("nd", [2, 3])
def test_the_channel_carries_what_was_said_what_was_heard_and_who_was_audible(nd: int) -> None:
    cfg, s = _pair(nd)
    fn = make_step(cfg, observe=True)
    s1, o1 = fn(s)
    mx.eval(s1, *o1)
    x1 = o1.acts.ext.comm
    assert (x1.voice.dtype, x1.heard.dtype, x1.audible.dtype) == (mx.uint8, mx.uint8, mx.uint8)
    assert x1.voice.shape == (dims(cfg).n, 3) and x1.audible.shape == (dims(cfg).n,)
    voice = to_numpy(x1.voice)
    assert (voice[:3] == 255).all() and not voice[3:].any()  # floor(256·0.99999) clipped to 255
    assert not to_numpy(x1.heard).any()  # step 1: nobody had spoken yet
    assert to_numpy(x1.audible)[:2].tolist() == [
        1,
        1,
    ]  # audible is the geometry: who is in range, spoken or not
    s2, o2 = fn(s1)
    mx.eval(s2, *o2)
    x2 = o2.acts.ext.comm
    heard, aud = to_numpy(x2.heard), to_numpy(x2.audible)
    # the pair hears each other: weight 1 - 1/range of a full call; the far agent hears nobody
    rng_ = float(dims(cfg).ext("comm").hear_range)
    expect = int(np.rint(255.0 * (1.0 - 1.0 / rng_) * 0.99999994))
    assert (np.abs(heard[:2].astype(int) - expect) <= 1).all() and not heard[2:].any()
    assert aud[:2].tolist() == [1, 1] and not aud[2:].any()
    flags = to_numpy(o2.acts.flags).astype(np.int64)
    spoke = (flags & ActFlag.SPOKE.value) != 0
    assert np.array_equal(spoke, (to_numpy(x2.voice) > 0).any(axis=1))
    assert int(spoke.sum()) == int(to_numpy(o2.counters)[counter_index(cfg, "COMM_SPEAKERS")])


@pytest.mark.parametrize("control", ["deaf", "decoy"])
def test_the_controls_show_in_the_channel(control: str) -> None:
    cfg, s = _pair(2, f"comm.control={control}")
    fn = make_step(cfg, observe=True)
    s, _ = fn(s)
    s, o = fn(s)
    mx.eval(s, *o)
    x = o.acts.ext.comm
    if control == "deaf":
        assert not to_numpy(x.heard).any() and not to_numpy(x.audible).any()
    else:  # decoy: the listeners hear a stranger's call, so some are audible-and-hearing, voice unchanged
        assert to_numpy(x.voice)[:3].min() == 255


def test_the_observer_keeps_the_latest_heard_half_for_the_play_loop() -> None:
    cfg, s = _pair(2, "game.observe=true")
    obs = Observer.fresh(s, cfg)
    assert obs.comm_last is None
    fn = make_step(cfg, observe=True)
    s, o = fn(s)
    obs.update(s, o.acts, 1)
    s, o = fn(s)
    obs.update(s, o.acts, 2)
    mx.eval(*obs.arrays())
    heard, aud = obs.comm_last
    assert np.array_equal(to_numpy(heard), to_numpy(o.acts.ext.comm.heard))
    assert np.array_equal(to_numpy(aud), to_numpy(o.acts.ext.comm.audible))
    assert to_numpy(aud)[:2].tolist() == [1, 1]


def test_without_comm_there_is_no_channel_and_no_comm_last() -> None:
    cfg = load_config(preset="tiny", overrides=["brain.weight_dtype=float32"])
    s = helpers.make_scene(cfg, [{"pos": (10.0, 10.0)}])
    s, o = make_step(cfg, observe=True)(s)
    mx.eval(s, *o)
    assert o.acts.ext is None


def test_the_unobserved_step_does_not_build_the_heard_half() -> None:
    cfg, s = _pair(2)
    s1, o = make_step(cfg)(s)
    mx.eval(s1, o)
    assert comm.Voice._fields[-2:] == ("heard", "audible")  # trailing optional fields: None unless observed
