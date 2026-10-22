"""§32 the version-3 observers on the engine's state: the wind drift and the velocity relative to it, the
extra accumulator columns, ``Acts.buoy``, the identity with version 2 where there is no wind, and the
sidecar of format 4."""

from __future__ import annotations

import dataclasses
import json

import mlx.core as mx
import numpy as np
import pytest

from alife import actions, energy, layout, weather
from alife import step as step_mod
from alife.config import Config, apply_overrides, dims, load_config
from alife.game.observe import (
    VQ_MAX,
    ArchCol,
    ObsIn,
    evaluate,
    init_obs,
    n_arch_cols,
    obs_in,
    quantise_inputs,
    speed_unit,
    update,
)
from alife.game.observer import OBSERVER_FORMAT, Observer
from alife.game.rules import RULES_V2, RULES_V3, Archetype, Intent, arch_layout, for_2d
from alife.state import init_state, to_numpy
from alife.step import make_step
from game_helpers import blank_acts, living, make_acts, make_oi
from helpers import set_output_bias
from weather_helpers import SMALL3D, fixed_wx, storm2d_config, storm3d_config

B = layout.B
I32 = mx.int32


def np_(a: mx.array) -> np.ndarray:
    return to_numpy(a)


def _oi(cfg: Config, rows: int, **fields) -> ObsIn:
    """A hand-built version-3 ``ObsIn``: ``rows`` living agents with a ``pos`` and a ``buoy_cap``."""
    arrays = living(cfg, rows)
    d = dims(cfg)
    arrays["pos"] = np.zeros((d.n, d.nd), np.float32)
    if d.d3:
        arrays["buoy_cap"] = np.full(d.n, 0.5, np.float32)
    for name, value in fields.items():
        arrays[name] = np.asarray(value)
    return make_oi(arrays)


# --- the drift ----------------------------------------------------------------------------------------------


def test_agent_drift_is_the_force_over_drag_of_the_engine() -> None:
    """The velocity the wind alone gives a free agent: ``agent_wind(...).force / drag``, with the eddy wind
    under ``wind_eddy_force`` and the scale under ``wind_scale``."""
    for extra in ([], ["weather.wind_scale=0.5"], ["weather.wind_eddy_force=true"], ["weather.wind_scale=0"]):
        cfg = storm3d_config(*extra)
        d = dims(cfg)
        wx = fixed_wx(cfg, u1=0.02, v1=-0.01, u2=0.05, v2=0.03)
        rng = np.random.default_rng(1)
        pos = mx.array(
            rng.uniform([0, 0, 1], [cfg.world.size, cfg.world.size, 70], (d.n, 3)).astype(np.float32)
        )
        cap = mx.array(rng.uniform(0, 1, d.n).astype(np.float32))
        alive = mx.ones((d.n,), dtype=mx.bool_)
        force = weather.agent_wind(weather.unpack(wx), pos, cap, alive, cfg).force
        drift = weather.agent_drift(wx, pos, cap, alive, cfg)
        assert drift.shape == (d.n, 3) and drift.dtype == mx.float32
        assert np.array_equal(np_(drift), np_(force / cfg.physics.drag)), extra
        if "weather.wind_scale=0" in extra:
            assert not np_(drift).any()
        elif "weather.wind_eddy_force=true" in extra:
            assert np.abs(np_(drift)[:, :2]).max() < 1e-6  # a zonally uniform wind has no eddy
        else:
            assert np.abs(np_(drift)[:, :2]).max() > 0.005


def test_agent_drift_in_2d() -> None:
    cfg = storm2d_config()
    wx = fixed_wx(cfg, u1=0.04, v1=0.0)
    d = dims(cfg)
    pos = mx.array(np.random.default_rng(2).uniform(0, cfg.world.size, (d.n, 2)).astype(np.float32))
    drift = weather.agent_drift(wx, pos, None, mx.ones((d.n,), dtype=mx.bool_), cfg)
    assert drift.shape == (d.n, 2)
    want = cfg.weather.alpha0 * 0.04
    assert np.allclose(np_(drift)[:, 0], want, rtol=1e-5, atol=1e-7) and not np_(drift)[:, 1].any()


# --- the velocity relative to the wind ----------------------------------------------------------------------


def test_v_rel_is_zero_for_a_free_drifter_and_the_own_velocity_otherwise() -> None:
    """A lone agent in a uniform wind with zero thrust moves with the drift: v_rel = 0; one with thrust moves
    with drift + own velocity: v_rel = own."""
    cfg = storm3d_config()
    wx = fixed_wx(cfg, u1=0.03, v1=0.01, u2=0.06, v2=-0.02)
    pos = np.zeros((dims(cfg).n, 3), np.float32)
    pos[0] = [100.0, 100.0, 60.0]  # above the interface (80·h1/(h1+h2)): the layer-2 wind
    pos[1] = [100.0, 100.0, 5.0]
    drift = np_(
        weather.agent_drift(
            wx, mx.array(pos), mx.full((dims(cfg).n,), 0.5), mx.ones((dims(cfg).n,), dtype=mx.bool_), cfg
        )
    )
    own = np.zeros_like(drift)
    own[1] = [0.02, -0.03, 0.005]
    scale = 256.0 / speed_unit(cfg)
    cap = np.full(dims(cfg).n, 0.5, np.float32)
    for vel, want in ((drift, np.zeros_like(drift)), (drift + own, own)):
        arrays = living(cfg, 2, vel=vel[:2])
        arrays["pos"], arrays["buoy_cap"], arrays["wx"] = pos, cap, np_(wx)
        arrays["vel"] = vel.astype(np.float32)
        x6, _, _ = quantise_inputs(
            make_oi(arrays), make_acts(blank_acts(cfg)), mx.zeros((dims(cfg).n, B), mx.bool_), cfg, RULES_V3
        )
        x6 = np_(x6)
        lay = arch_layout(3, RULES_V3)
        halves = x6[:, lay.v_pos : lay.v_pos + 3].astype(np.int64) - x6[:, lay.v_neg : lay.v_neg + 3]
        assert np.abs(halves[:2] - np.round(want[:2] * scale)).max() <= 1, (halves[:2], want[:2] * scale)
        speed = np.sqrt((want[:2] ** 2).sum(axis=1)) * scale
        assert np.abs(x6[:2, ArchCol.SPEED.value] - np.round(speed)).max() <= 1
        assert x6[0, lay.wind] > 0 and x6[0, lay.wind] == round(float(np.linalg.norm(drift[0])) * scale)


def test_the_extra_columns_follow_numpy_doubles() -> None:
    cfg = storm3d_config()
    n, h = dims(cfg).n, cfg.world.height
    wx = fixed_wx(cfg, u1=0.02, u2=0.04)
    rng = np.random.default_rng(5)
    pos = rng.uniform([0, 0, 0], [cfg.world.size, cfg.world.size, h], (n, 3)).astype(np.float32)
    cap = rng.uniform(0, 1, n).astype(np.float32)
    buoy = rng.integers(0, 257, n).astype(np.int16)
    arrays = living(cfg, n)
    arrays["pos"], arrays["buoy_cap"], arrays["wx"] = pos, cap, np_(wx)
    acts = blank_acts(cfg)
    acts["buoy"] = buoy
    x6, _, _ = quantise_inputs(make_oi(arrays), make_acts(acts), mx.zeros((n, B), mx.bool_), cfg, RULES_V3)
    x6, lay = np_(x6), arch_layout(3, RULES_V3)
    assert x6.shape == (n, 14) and x6.dtype == np.int32
    assert np.array_equal(x6[:, lay.zpos], np.clip(np.round(pos[:, 2] * np.float32(256.0 / h)), 0, 256))
    capq = np.round(cap * np.float32(256.0)).astype(np.int64)
    assert np.array_equal(x6[:, lay.lift], (buoy.astype(np.int64) * capq) >> 8)
    assert x6[:, lay.lift].max() <= 256 and x6[:, lay.wind].max() <= VQ_MAX
    # without the channel's buoyancy (variant 3) the lift column is the capacity alone
    acts.pop("buoy")
    x6b, _, _ = quantise_inputs(make_oi(arrays), make_acts(acts), mx.zeros((n, B), mx.bool_), cfg, RULES_V3)
    assert np.array_equal(np_(x6b)[:, lay.lift], capq)


def test_wind_rel_without_weather_is_version_2_on_the_first_columns() -> None:
    """A world with no weather has no drift: the velocity columns of the wind-relative rules are bitwise the
    version-2 ones, WIND is zero, and ZPOS and LIFT follow the state."""
    cfg = load_config(preset="64k3d", overrides=list(SMALL3D))  # the storm world's size, no weather
    assert not cfg.weather.enabled
    n = dims(cfg).n
    rng = np.random.default_rng(3)
    arrays = living(cfg, n, vel=rng.normal(0, 0.05, (n, 3)).astype(np.float32))
    arrays["pos"] = rng.uniform(0, 70, (n, 3)).astype(np.float32)
    arrays["buoy_cap"] = rng.uniform(0, 1, n).astype(np.float32)
    acts = make_acts({**blank_acts(cfg), "buoy": np.zeros(n, np.int16)})
    kin = mx.zeros((n, B), mx.bool_)
    a2, s2, i2 = quantise_inputs(make_oi(arrays), acts, kin, cfg, RULES_V2)
    a3, s3, i3 = quantise_inputs(make_oi(arrays), acts, kin, cfg, RULES_V3)
    assert a2.shape == (n, n_arch_cols(3)) and a3.shape == (n, n_arch_cols(3, RULES_V3))
    assert np.array_equal(np_(a3)[:, : a2.shape[1]], np_(a2))  # the first 5 + 2·nd columns, bitwise
    assert not np_(a3)[:, arch_layout(3, RULES_V3).wind].any()
    assert np.array_equal(np_(s2), np_(s3)) and np.array_equal(np_(i2), np_(i3))


# --- Acts.buoy ----------------------------------------------------------------------------------------------


def test_acts_buoy_is_the_pre_mask_buoyancy_quantised_and_masked_by_solvency() -> None:
    cfg = apply_overrides(storm3d_config("population.n_init=64"), ["brain.weight_dtype=float32"])
    s = set_output_bias(init_state(cfg), cfg, {dims(cfg).lay.OUT_BUOYANCY: 0.4})
    seen: dict[str, mx.array] = {}
    real_motion, real_pay = actions.decode_motion3, energy.pay_costs3

    def motion(*a, **k):
        seen["mo"] = real_motion(*a, **k)
        return seen["mo"]

    def pay(*a, **k):
        seen["c"] = real_pay(*a, **k)
        return seen["c"]

    with pytest.MonkeyPatch.context() as m:
        m.setattr(actions, "decode_motion3", motion)
        m.setattr(energy, "pay_costs3", pay)
        s1, out = step_mod.step(s, cfg, observe=True)
    mx.eval(s1, out)
    b = np_(out.acts.buoy)
    assert b.dtype == np.int16 and b.shape == (dims(cfg).n,)
    want = np.where(np_(seen["c"].solvent), np.round(np_(seen["mo"].buoy) * np.float32(256.0)), 0)
    assert np.array_equal(b, want.astype(np.int16))
    assert (b > 0).any() and b.max() <= 256 and (b[~np_(s.alive)] == 0).all()
    # an insolvent row has none: starve one
    starved = s._replace(energy=mx.where(mx.arange(dims(cfg).n) == 0, 0, s.energy))
    s2, out2 = step_mod.step(starved, cfg, observe=True)
    mx.eval(s2, out2)
    assert int(np_(out2.acts.buoy)[0]) in (0,)  # row 0 cannot pay the due: no buoyancy use is recorded


def test_acts_buoy_is_absent_in_2d(tiny_cfg: Config) -> None:
    _, out = step_mod.step(init_state(tiny_cfg), tiny_cfg, observe=True)
    assert out.acts.buoy is None


# --- version 2 against version 3 where there is no wind ---------------------------------------------------


def test_version_3_changes_only_the_new_labels_without_weather(tiny3d_cfg: Config) -> None:
    """Weather-off observed tiny3d for 200 steps: the ten intents and the four income archetypes are
    bit-identical under the two rule sets; the primary archetype differs only where a new label matches, and
    then it was Generalist; Generalist shrinks by exactly those rows."""
    cfg = tiny3d_cfg
    bias = {dims(cfg).lay.OUT_BUOYANCY: 1.5, layout.OUT_HARVEST: 1.0}
    s = set_output_bias(init_state(cfg), cfg, bias)
    fn = make_step(cfg, observe=True)
    obs = {name: Observer.fresh(s, cfg, rules=rs) for name, rs in (("v2", RULES_V2), ("v3", RULES_V3))}
    assert obs["v3"].o.arch.shape[1] == 14 and obs["v2"].o.arch.shape[1] == 11
    for t in range(200):
        s, out = fn(s)
        for o in obs.values():
            o.update(s, out.acts, t + 1)
        mx.eval(s, out.base(), *[a for o in obs.values() for a in o.arrays()])
    a2, a3 = obs["v2"], obs["v3"]
    assert np.array_equal(np_(a2.o.arch), np_(a3.o.arch)[:, :11])  # no wind: the first 11 columns, bitwise
    l2, l3 = a2.host_labels(), a3.host_labels()
    for field in ("intent", "intent_match"):
        assert np.array_equal(l2[field], l3[field]), field
    p2, p3 = l2["arch_primary"], l3["arch_primary"]
    new = (Archetype.DRIFTER.value, Archetype.FLOATER.value, Archetype.SINKER.value)
    changed = p2 != p3
    assert np.isin(p3[changed], new).all() and (p2[changed] == Archetype.GENERALIST.value).all()
    # a V2 primary that is not Generalist never changes
    assert np.array_equal(p2[p2 != Archetype.GENERALIST.value], p3[p2 != Archetype.GENERALIST.value])
    gen2, gen3 = int((p2 == Archetype.GENERALIST.value).sum()), int((p3 == Archetype.GENERALIST.value).sum())
    assert gen2 - gen3 == int(changed.sum())
    # the match bits of the six older rules are the same; the new bits are set only in V3
    m2, m3 = l2["arch_match"], l3["arch_match"]
    older = (1 << 6) - 1
    assert np.array_equal(m2 & older, m3 & older)
    assert not (m2 >> 7).any()
    # the probe is not vacuous: some row is a Floater or a Sinker
    assert (
        int(
            ((m3 >> (Archetype.FLOATER.value - 1)) & 1).sum()
            + ((m3 >> (Archetype.SINKER.value - 1)) & 1).sum()
        )
        > 0
    )
    assert int(changed.sum()) > 0
    assert not (np_(a3.o.arch)[:, arch_layout(3, RULES_V3).wind]).any()  # drift is exactly 0 without weather


def test_an_observer_with_weather_accumulates_the_wind() -> None:
    """On a weather world with a prescribed wind the Drifter's WIND column converges to the drift of the
    row, and a free agent that moves with the wind has a small relative speed."""
    cfg = storm3d_config()
    s = init_state(cfg)
    s = s._replace(wx=fixed_wx(cfg, u1=0.03, u2=0.05))
    fn = make_step(cfg, observe=True)
    obs = Observer.fresh(s, cfg)
    for t in range(120):
        s, out = fn(s)
        s = s._replace(wx=fixed_wx(cfg, u1=0.03, u2=0.05))
        obs.update(s, out.acts, t + 1)
        mx.eval(s, out.base(), *obs.arrays())
    lay = arch_layout(3, RULES_V3)
    mean = np_(obs.o.arch) >> RULES_V3.arch_k
    alive = np_(s.alive)
    assert mean[alive, lay.wind].max() > 0
    drift = np_(weather.agent_drift(s.wx, s.pos, s.buoy_cap, s.alive, cfg))
    scale = 256.0 / speed_unit(cfg)
    last = np.round(np.linalg.norm(drift, axis=1) * scale)
    # the accumulator is a windowed mean of a slowly varying quantity: it follows the last value
    assert np.corrcoef(mean[alive, lay.wind], last[alive])[0, 1] > 0.9


# --- the sidecar --------------------------------------------------------------------------------------------


def test_the_sidecar_names_the_rules_and_the_format(tiny3d_cfg: Config) -> None:
    cfg = tiny3d_cfg
    s = init_state(cfg)
    obs = Observer.fresh(s, cfg)
    side = obs.sidecar()
    meta = json.loads(str(side["meta"]))
    assert OBSERVER_FORMAT == 4 and meta["observer_format"] == 4 and meta["intent_version"] == 3
    assert meta["wind_rel"] is True and meta["vertical"] is True and meta["rules_digest"] == RULES_V3.digest()
    assert side["arch"].shape == (dims(cfg).n, 14)
    # the same observer, 2D: no vertical labels, 10 columns
    assert for_2d(RULES_V3).wind_rel and n_arch_cols(2, for_2d(RULES_V3)) == 10


def test_every_label_is_in_the_row_block_and_an_empty_row_is_a_settler(tiny3d_cfg: Config) -> None:
    cfg = tiny3d_cfg
    s = init_state(cfg)
    assert Observer.fresh(s, cfg).row_block()["intent_version"] == 3
    # evaluate on an empty accumulator: every row is at rest, so a Settler; the vertical rules cannot match a
    # lift of 0 as a Floater (lift 0 < 77) and need a fall to be a Sinker
    o = init_obs(obs_in(s, RULES_V3), cfg, RULES_V3)
    oi = obs_in(s, RULES_V3)
    ev = evaluate(
        o._replace(n_obs=mx.full_like(o.n_obs, 10_000)), cfg, RULES_V3, body=mx.zeros_like(o.n_obs), oi=oi
    )
    prim = np_(ev.label_arch)[:, 0][np_(s.alive)]
    assert (prim == Archetype.SETTLER.value).all()
    # the row block names all ten archetypes once rows are labelled
    obs = Observer.fresh(s, cfg)
    fn = make_step(cfg, observe=True)
    for t in range(200):
        s, out = fn(s)
        obs.update(s, out.acts, t + 1)
        mx.eval(s, out.base(), *obs.arrays())
    names = ("gatherer", "hunter", "provider", "trader", "nomad", "settler", "generalist")
    block = obs.row_block()
    assert (
        tuple(block["archetype"])
        == (*names, "drifter", "floater", "sinker")
        == tuple(block["archetype_match"])
    )
    assert abs(sum(block["archetype"].values()) - 1.0) < 1e-9 and Intent.RESTING.value == 10


def test_update_signature_roundtrip(tiny3d_cfg: Config) -> None:
    """update/evaluate with the version-3 rules run on a 3D state without weather (no drift is evaluated)."""
    cfg = tiny3d_cfg
    s = init_state(cfg)
    o = init_obs(obs_in(s, RULES_V3), cfg, RULES_V3)
    fn = make_step(cfg, observe=True)
    s, out = fn(s)
    o = update(o, obs_in(s, RULES_V3), out.acts, cfg, RULES_V3)
    mx.eval(*o)
    assert np_(o.n_obs)[np_(s.alive)].max() == 1
    assert dataclasses.is_dataclass(RULES_V3)


def test_the_observer_in_a_weather_world_is_the_reference_fold_and_resumes_exactly(tmp_path) -> None:
    """The compiled observer (the drift evaluated in its own fold) equals the eager reference ``update`` and
    ``evaluate`` on the storm world, and a run resumed from its format-4 sidecar continues identically."""
    from dataclasses import replace

    from alife.game.observe import ObsState
    from alife.game.social import SOCIAL_V1
    from alife.runio import RunDir, load_checkpoint

    cfg = storm3d_config("population.n_init=128")
    soc15 = replace(SOCIAL_V1, snapshot_every=15)
    s = init_state(cfg)
    s = s._replace(wx=fixed_wx(cfg, u1=0.02, v1=0.01, u2=0.05))  # a prescribed wind: drift everywhere
    fn = make_step(cfg, observe=True)
    obs = Observer.fresh(s, cfg, social=soc15)
    assert obs.rules is RULES_V3
    ref = init_obs(obs_in(s, RULES_V3), cfg, RULES_V3)
    for t in range(1, 41):
        s, out = fn(s)
        s = s._replace(wx=fixed_wx(cfg, u1=0.02, v1=0.01, u2=0.05))
        obs.update(s, out.acts, t)
        ref = update(ref, obs_in(s, RULES_V3), out.acts, cfg, RULES_V3)
        if t % RULES_V3.every == 0:
            ref = evaluate(ref, cfg, RULES_V3, body=obs.so.body, oi=obs_in(s, RULES_V3))
        mx.eval(s, out.base(), *obs.arrays(), *ref)
        obs.take_records()
        if t % RULES_V3.every != 0:
            for f in ("arch", "intent", "n_obs"):
                assert np.array_equal(np_(getattr(obs.o, f)), np_(getattr(ref, f))), (t, f)
    lay = arch_layout(3, RULES_V3)
    assert np_(obs.o.arch)[:, lay.wind].max() > 0  # the drift was in the fold
    # resume: write the checkpoint and sidecar, open, and continue both for 10 steps
    rd = RunDir.create(tmp_path / "run", cfg)
    path = rd.save_checkpoint(s, cfg, observers=obs.sidecar())
    state = load_checkpoint(path, cfg)
    opened, note = Observer.open(rd, state, cfg, social=soc15)
    assert note is None and opened.step == 40
    s2 = state
    for t in range(41, 51):
        s, out = fn(s)
        s = s._replace(wx=fixed_wx(cfg, u1=0.02, v1=0.01, u2=0.05))
        s2, out2 = fn(s2)
        s2 = s2._replace(wx=fixed_wx(cfg, u1=0.02, v1=0.01, u2=0.05))
        obs.update(s, out.acts, t)
        opened.update(s2, out2.acts, t)
        mx.eval(s, s2, out.base(), out2.base(), *obs.arrays(), *opened.arrays())
        obs.take_records()
        opened.take_records()
    for f in ObsState._fields:
        assert np.array_equal(np_(getattr(obs.o, f)), np_(getattr(opened.o, f))), f
