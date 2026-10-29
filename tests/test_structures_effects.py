"""Built structures: decay, the build action, the sensor and the three effects (DESIGN §38.2 to §38.4)."""

from __future__ import annotations

from dataclasses import replace

import mlx.core as mx
import numpy as np
import pytest

from alife import actions, fields, ops, physics, structures, weather
from alife import energy as en
from alife.config import apply_overrides, dims
from alife.state import init_state, to_numpy
from alife.step import make_step
from helpers import make_scene
from structures_helpers import builders, cfg_for, np_mat, run, units_for
from weather_helpers import fixed_wx, storm2d_config, storm3d_config, zero_actions

# --- decay (section 3) ----------------------------------------------------------------------------


def _decay_ref(m: int, q: int = 16) -> int:
    return m - ((m * q + 4095) >> 12)


@pytest.mark.parametrize("nd", [2, 3])
def test_a_full_cell_with_no_builders_follows_the_ceil_form_and_vanishes_in_864_steps(nd: int) -> None:
    cfg = cfg_for(nd)
    d = dims(cfg)
    m = mx.zeros(d.field_shape, dtype=mx.int32)
    m = m.at[(0,) * len(d.field_shape)].add(cfg.structures.cap)
    dep = mx.zeros(d.field_shape, dtype=mx.int32)
    ref, n = cfg.structures.cap, 0
    cell = (0,) * len(d.field_shape)
    while ref > 0:
        up = fields.update_mat(m, dep, mx.array(n, mx.int32), cfg)
        m = up.mat
        ref = _decay_ref(ref)
        n += 1
        assert int(m[cell].item()) == ref and int(m.sum().item()) == ref, n
    assert n == 864


def test_a_steady_builder_holds_the_cell_at_the_probe_levels() -> None:
    """16 units per step hold a cell at 3825 of 4096 (93 %); 8 hold 1785; 1 holds nothing (the floor)."""
    cfg = cfg_for(2)
    for units, level in ((16, 3825), (8, 1785), (4, 765), (2, 255), (1, 0)):
        m, dep = mx.zeros((32, 32), mx.int32), mx.zeros((32, 32), mx.int32).at[3, 3].add(units)
        for t in range(4000):
            m = fields.update_mat(m, dep, mx.array(t, mx.int32), replace_aperture_off(cfg)).mat
        assert int(m[3, 3].item()) == level, units


def replace_aperture_off(cfg):
    return replace(cfg, structures=replace(cfg.structures, aperture=0.0))


# --- the build action and the sensor (sections 2 and 5) -------------------------------------------


def test_decode_build_bounds_and_zero() -> None:
    cfg = cfg_for(2)
    d = dims(cfg)
    y = mx.zeros((4, d.lay.D_OUT), mx.float32)
    y = y.at[:, d.lay.OUT_BUILD].add(mx.array([-3.0, 0.0, 0.3, 50.0]))
    b = actions.decode_build(y, mx.array([True, True, True, False]), cfg)
    assert b.units.tolist() == [0, 0, int(np.floor(np.float32(np.tanh(np.float32(0.3))) * 16)), 0]
    y2 = y.at[3, d.lay.OUT_BUILD].add(0.0)
    assert actions.decode_build(y2, mx.ones((4,), mx.bool_), cfg).units.tolist()[3] == 16  # tanh(50) = 1


def test_the_sensor_is_the_material_level_at_the_agents_cell() -> None:
    cfg = cfg_for(2)
    d = dims(cfg)
    s = init_state(cfg)
    rs = np.random.default_rng(1)
    mat = mx.array(rs.integers(0, cfg.structures.cap + 1, d.field_shape).astype(np.int32))
    s = s._replace(mat=mat)
    from alife.ext import ExtCtx

    fcell = fields.cell_index(s.pos, cfg)
    (col,) = structures.inputs(s, cfg, ExtCtx(None, None, None, fcell))
    want = np.asarray(mat).reshape(-1)[np.asarray(fcell)] / cfg.structures.cap
    assert np.allclose(np.asarray(col)[:, 0], want, atol=1e-7)
    assert d.lay.IN_MATERIAL == d.lay.D_IN - 1  # appended last, with structures alone


def test_a_forced_builder_deposits_at_its_own_cell_and_pays_for_it() -> None:
    cfg = cfg_for(2)
    s = builders(cfg, zero_weights=True)
    u = units_for(cfg)
    assert 0 < u <= 16
    prev, s1, _, row = next(run(cfg, s, 1))
    alive = to_numpy(prev.alive)
    fc = np.asarray(fields.cell_index(prev.pos, cfg))[alive]
    expect = np.zeros(dims(cfg).n_cells, np.int64)
    np.add.at(expect, fc, u)
    # one step later: deposit then decay (the cap cannot bind: 16 per builder, a few per cell)
    got = np_mat(s1).reshape(-1)
    pre = np.minimum(expect, cfg.structures.cap)
    assert np.array_equal(got, pre - ((pre * 16 + 4095) >> 12))
    assert row["BUILD_UNITS"] == u * alive.sum()


# --- shelter (4.1) --------------------------------------------------------------------------------


def test_shelter_wind_scales_force_speed_and_recomputes_breaks() -> None:
    cfg = storm2d_config("structures.enabled=true", "weather.storm_break_u=0.5")
    n = 4
    force = mx.ones((n, 2), mx.float32)
    speed = mx.array([0.4, 0.8, 1.2, 2.0], mx.float32)
    w = weather.AgentForce(force, speed, speed > 0.5, mx.array(False))
    k = mx.array([1.0, 0.5, 0.25, 0.2], mx.float32)
    sw = weather.shelter_wind(w, k, cfg)
    assert np.allclose(np.asarray(sw.force), np.asarray(force * k[:, None]))
    assert np.allclose(np.asarray(sw.speed), np.asarray(speed * k))
    assert np.asarray(sw.breaks).tolist() == [False, False, False, False]  # 0.4, 0.4, 0.3, 0.4 < 0.5
    assert np.asarray(w.breaks).tolist() == [False, True, True, True]
    none = weather.AgentForce(force, speed, None, mx.array(False))
    assert weather.shelter_wind(none, k, cfg).breaks is None


def test_a_sheltered_row_has_a_lower_storm_probability() -> None:
    cfg = storm2d_config(
        "structures.enabled=true", "weather.storm_p=0.5", "weather.storm_u0=0.02", "weather.storm_span=0.2"
    )
    wc = cfg.weather
    sp = mx.array([0.05, 0.075, 0.1, 0.12], mx.float32)

    def p(speed):
        return np.clip((np.asarray(speed) - wc.storm_u0) / wc.storm_span, 0, 1) * wc.storm_p

    k = 1.0 - cfg.structures.shelter_gain * 1.0
    assert np.all(p(sp * k) <= p(sp)) and p(sp * k).sum() < p(sp).sum()


@pytest.mark.parametrize("nd", [2, 3])
def test_in_the_step_a_full_cell_cuts_the_wind_drift_to_the_shelter_and_barrier_factors(nd: int) -> None:
    """Zero weights (no thrust, no turn), a uniform wind, material full everywhere: the velocity is the
    unsheltered one times ``(1 - shelter_gain)/(1 + barrier_gain)`` (a plateau: R3 has no gradient)."""
    mk = storm2d_config if nd == 2 else storm3d_config
    base = mk("structures.enabled=true", "structures.aperture=0.0", "weather.storm_p=0.0")
    d = dims(base)
    z = 12.0 if nd == 3 else None
    spots = [(20.0 + 40.0 * i, 30.0 + 25.0 * i) + ((z,) if nd == 3 else ()) for i in range(6)]  # far apart
    zero = make_scene(base, [{"pos": p} for p in spots], wx=fixed_wx(base, u1=0.05, v1=0.03))
    full = mx.full(d.field_shape, base.structures.cap, dtype=mx.int32)

    def vel(cfg, mat):
        s = zero._replace(mat=mat)
        s1, _ = make_step(cfg)(s)
        mx.eval(s1)
        return to_numpy(s1.vel)[: len(spots)]

    empty = mx.zeros(d.field_shape, dtype=mx.int32)
    v0 = vel(base, empty)
    sheltered = vel(replace(base, structures=replace(base.structures, barrier_gain=0.0)), full)
    both = vel(base, full)
    assert np.abs(v0[:, :2]).max() > 1e-3
    sg, bg = base.structures.shelter_gain, base.structures.barrier_gain
    assert np.allclose(sheltered[:, :2], (1 - sg) * v0[:, :2], rtol=1e-4, atol=1e-7)
    assert np.allclose(both[:, :2], (1 - sg) / (1 + bg) * v0[:, :2], rtol=1e-4, atol=1e-7)


def test_shelter_is_inert_in_a_dry_world() -> None:
    cfg = cfg_for(2)  # weather off
    assert not cfg.weather.enabled and "cover" not in fields.mat_columns(replace_gain(cfg, shield=0.0))


def replace_gain(cfg, shield):
    return replace(cfg, structures=replace(cfg.structures, shield_gain=shield))


# --- barrier (4.2) --------------------------------------------------------------------------------


def test_barrier_drag_scales_the_displacement_by_one_over_the_factor() -> None:
    cfg = apply_overrides(cfg_for(2), ["structures.exit_ease=0"])
    n = 3
    force = mx.array([[0.04, 0.0]] * n, mx.float32)
    here = mx.array([0.0, 0.5, 1.0], mx.float32)
    dm = physics.barrier_drag(force, here, None, cfg)
    assert np.allclose(np.asarray(dm), 1.0 + 3.0 * np.asarray(here))
    pos = mx.full((n, 2), 10.0, mx.float32)
    out = lambda m: physics.integrate(  # noqa: E731
        pos, mx.zeros((n,)), force, mx.zeros((n,)), mx.zeros((n,)), mx.ones((n,), mx.bool_), cfg, drag_mult=m
    )
    free, slow = out(None)[0], out(dm)[0]
    gain = np.asarray(slow - pos)[:, 0] / np.asarray(free - pos)[:, 0]
    assert np.allclose(gain, 1.0 / np.asarray(dm), rtol=1e-3)  # pos + v rounds in float32


def test_barrier_in_3d_slows_sinking_too() -> None:
    cfg = cfg_for(3)
    d = dims(cfg)
    n = 2
    force = mx.array([[0.0, 0.0, -0.01]] * n, mx.float32)
    pos = mx.array([[10.0, 10.0, 12.0]] * n, mx.float32)
    dm = mx.array([1.0, 4.0], mx.float32)
    z = mx.zeros((n,))
    it = physics.integrate3(
        pos, z, z, force, z, z, z, mx.zeros((n,), mx.bool_), mx.ones((n,), mx.bool_), cfg, drag_mult=dm
    )
    vz = np.asarray(it.vel)[:, 2]
    assert d.d3 and np.isclose(vz[0] / vz[1], 4.0, rtol=1e-4)


# --- shield (4.3) ---------------------------------------------------------------------------------


def _combat(cfg, cover, defend=0.0):
    """Agent 0 attacks agent 1 (energy 5000 each); the victim's cover is ``cover``."""
    d = dims(cfg)
    n = d.n
    e = np.zeros(n, np.int32)
    e[:2] = 5000
    tgt = np.full(n, -1, np.int32)
    tgt[0] = 1
    att = np.zeros(n, bool)
    att[0] = True
    dfd = np.zeros(n, np.float32)
    dfd[1] = defend
    act = zero_actions(n)._replace(attack=mx.array(att), attack_target=mx.array(tgt), defend=mx.array(dfd))
    res = mx.zeros(d.field_shape, mx.int32)
    fcell = mx.zeros((n,), mx.int32)
    cov = np.zeros(n, np.float32)
    cov[1] = cover
    sn = structures.Sense(mx.array(cov), mx.array(cov), None)
    partner = mx.full((n, 4), -1, mx.int32)
    return en.combat(mx.array(e), res, fcell, act, partner, cfg, **structures.combat_kw(sn, cfg))


def test_a_victim_in_a_full_cell_loses_the_shielded_demand() -> None:
    cfg = cfg_for(2)
    cc, sc = cfg.combat, cfg.structures
    plain = _combat(cfg, 0.0)
    shielded = _combat(cfg, 1.0)
    assert int(plain.lost[1].item()) == cc.attack_max
    want = int(np.floor(np.float32(cc.attack_max) * np.float32(1.0 - sc.shield_gain * 1.0)))
    assert int(shielded.lost[1].item()) == want
    # the integer pipeline is unchanged: lost = gain + spill + residue
    assert int(shielded.lost.sum().item()) == int(
        (shielded.gain + shielded.spill + shielded.residue).sum().item()
    )
    assert bool(shielded.shielded[0].item()) and not bool(
        plain.shielded is not None and plain.shielded[0].item()
    )
    # composes multiplicatively with the behavioural defend
    both = _combat(cfg, 1.0, defend=1.0)
    w2 = int(
        np.floor(np.float32(cc.attack_max) * np.float32(1.0 - 0.5) * np.float32(1.0 - cc.defend_efficiency))
    )
    assert int(both.lost[1].item()) == w2


def test_the_attackers_cost_is_still_paid_so_a_shield_makes_attacks_less_profitable() -> None:
    cfg = cfg_for(2)
    cc = cfg.combat
    gain_plain, gain_shield = int(_combat(cfg, 0.0).gain[0].item()), int(_combat(cfg, 1.0).gain[0].item())
    assert gain_shield < gain_plain
    assert gain_shield == int(
        ops.mul_q8(mx.array([cc.attack_max // 2], mx.int32), ops.q8(cc.efficiency))[0].item()
    )
