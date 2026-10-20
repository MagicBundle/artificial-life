"""W1 build step 2 (DESIGN §30, spec §9.4): the wind as a drag force. ``physics.forces`` / ``forces3`` with
``wind=``, ``weather.agent_wind`` against ``sample_wind``, the clips and their counts, the closed forms for a
lone grounded agent, and the engine invariants under a strong prescribed wind.

CPU and fast; physics-level tests call the functions directly, the invariant run steps the engine eagerly.
"""

from __future__ import annotations

import math
from dataclasses import replace

import mlx.core as mx
import numpy as np
import pytest

from alife import invariants, physics
from alife import weather as W
from alife.config import Config, dims
from alife.layout import B
from alife.state import ERROR_MASK, Diag, init_state, to_numpy
from alife.step import make_step
from weather_helpers import (
    fixed_wx,
    floor_scene,
    no_neighbors,
    storm2d_config,
    storm3d_config,
    zero_actions,
    zero_motion,
)

f32 = np.float32


@pytest.fixture(scope="module")
def cfg3() -> Config:
    return storm3d_config()


@pytest.fixture(scope="module")
def cfg2() -> Config:
    return storm2d_config()


def wavy_ws(cfg: Config, amp: float) -> W.WeatherState:
    """A rest state with a divergent layer-1 wind: ``u1 = amp·sin(2π x/L)``, ``v1 = amp·cos(2π y/L)`` on the
    faces, and a layer-2 wind ``u2 = -u1/2``. Divergence is non-zero, so ``w`` is too."""
    ws = W.unpack(W.rest_array(cfg))
    p = W.params_from_config(cfg)
    x = (np.arange(p.n) + 1.0) * p.dx  # u faces at (i + 1) dx
    y = (np.arange(p.n) + 1.0) * p.dx  # v faces at (j + 1) dx
    u1 = np.broadcast_to(amp * np.sin(2 * math.pi * x / p.L)[None, :], (p.n, p.n))
    v1 = np.broadcast_to(amp * np.cos(2 * math.pi * y / p.L)[:, None], (p.n, p.n))
    return ws._replace(
        u1=mx.array(u1.astype(f32)),
        v1=mx.array(v1.astype(f32)),
        u2=mx.array((-0.5 * u1).astype(f32)),
        v2=mx.array((0.25 * v1).astype(f32)),
    )


def _nbr(cfg: Config):
    d = dims(cfg)
    return no_neighbors(d.n, d.k, cfg.sensing.radius)


def _forces3(cfg: Config, pos, alive, *, thrust=0.0, cap=0.0, nbr=None, wind=None, kw_wind=True):
    d = dims(cfg)
    n = d.n
    z = mx.zeros((n,), dtype=mx.float32)
    act = zero_actions(n)._replace(thrust=mx.full((n,), thrust, dtype=mx.float32))
    partner = mx.full((n, B), -1, dtype=mx.int32)
    args = (pos, z, z, act, zero_motion(n), mx.full((n,), cap, dtype=mx.float32))
    args += (_nbr(cfg) if nbr is None else nbr, alive, partner, cfg)
    return physics.forces3(*args, wind=wind) if kw_wind else physics.forces3(*args)


# --- forces with wind ------------------------------------------------------------------------------------


def test_forces3_without_wind_is_the_old_graph_and_a_zero_wind_changes_no_value(cfg3: Config) -> None:
    pos, alive = floor_scene(cfg3, [(100.0, 100.0, 10.0), (200.0, 50.0, 30.0), (50.0, 250.0, 2.0)])
    base = _forces3(cfg3, pos, alive, thrust=0.7, cap=0.4, kw_wind=False)
    none = _forces3(cfg3, pos, alive, thrust=0.7, cap=0.4, wind=None)
    zero = _forces3(cfg3, pos, alive, thrust=0.7, cap=0.4, wind=mx.zeros(pos.shape, dtype=mx.float32))
    assert np.array_equal(to_numpy(base), to_numpy(none))  # bitwise (the same graph)
    assert np.array_equal(to_numpy(base), to_numpy(zero))  # equal in value (== treats -0.0 as 0.0)
    assert abs(float(to_numpy(base)[0, 0])) > 0  # the thrust is real: the comparison is not vacuous


def test_forces_2d_without_wind_and_with_zero_wind(cfg2: Config) -> None:
    d = dims(cfg2)
    pos, alive = floor_scene(cfg2, [(100.0, 100.0), (200.0, 50.0)])
    z = mx.zeros((d.n,), dtype=mx.float32)
    act = zero_actions(d.n)._replace(thrust=mx.full((d.n,), 0.6, dtype=mx.float32))
    partner = mx.full((d.n, B), -1, dtype=mx.int32)
    nbr = no_neighbors(d.n, d.k, cfg2.sensing.radius)
    nbr = nbr._replace(off=nbr.off[..., :2])
    base = physics.forces(pos, z, act, nbr, alive, partner, cfg2)
    none = physics.forces(pos, z, act, nbr, alive, partner, cfg2, wind=None)
    zero = physics.forces(pos, z, act, nbr, alive, partner, cfg2, wind=mx.zeros_like(pos))
    assert np.array_equal(to_numpy(base), to_numpy(none)) and np.array_equal(to_numpy(base), to_numpy(zero))
    wind = mx.full(pos.shape, 0.01, dtype=mx.float32)
    got = to_numpy(physics.forces(pos, z, act, nbr, alive, partner, cfg2, wind=wind))
    # the dead rows get 0 whatever the wind is; the live rows get thrust + wind
    assert not got[2:].any()
    np.testing.assert_allclose(got[:2], to_numpy(base)[:2] + f32(0.01), rtol=1e-6)


def test_wind_is_not_contact_attenuated(cfg3: Config) -> None:
    """Four contacts attenuate the thrust to 0 (``thrust_contact_limit`` 4); the wind force stays intact."""
    d = dims(cfg3)
    pts = [
        (100.0, 100.0, 10.0),
        (101.5, 100.0, 10.0),
        (98.5, 100.0, 10.0),
        (100.0, 101.5, 10.0),
        (100.0, 98.5, 10.0),
    ]
    pos, alive = floor_scene(cfg3, pts)
    P = to_numpy(pos)
    idx = np.tile(np.arange(d.n, dtype=np.int32)[:, None], (1, d.k))
    valid = np.zeros((d.n, d.k), bool)
    off = np.zeros((d.n, d.k, 3), f32)
    dist = np.full((d.n, d.k), cfg3.sensing.radius, f32)
    for k, j in enumerate((1, 2, 3, 4)):  # agent 0 sees the four others, all within 2 r
        idx[0, k], valid[0, k], off[0, k] = j, True, P[j] - P[0]
        dist[0, k] = np.linalg.norm(off[0, k])
    zero = mx.array(0, mx.int32)
    nbr = type(no_neighbors(d.n, d.k, 1.0))(
        mx.array(idx), mx.array(valid), mx.array(off), mx.array(dist), zero, zero
    )
    wind = np.zeros(P.shape, f32)
    wind[0] = (0.03, -0.02, 0.01)
    f_wind = to_numpy(_forces3(cfg3, pos, alive, thrust=1.0, nbr=nbr, wind=mx.array(wind)))
    f_none = to_numpy(_forces3(cfg3, pos, alive, thrust=1.0, nbr=nbr))
    assert np.array_equal(f_none[0, :2], np.zeros(2, f32))  # thrust fully attenuated, repulsion cancels
    assert np.array_equal(f_wind[0, :2], wind[0, :2])  # the wind force is exactly what was added
    np.testing.assert_allclose(f_wind[0, 2] - f_none[0, 2], 0.01, atol=1e-7)
    lone = to_numpy(_forces3(cfg3, pos, alive, thrust=1.0))  # control: without contacts the thrust acts
    assert np.linalg.norm(lone[0, :2]) > 0.05


# --- agent_wind against sample_wind ------------------------------------------------------------------------


def _agents(cfg: Config, n_live: int = 12):
    rng = np.random.default_rng(3)
    pts = [
        (float(rng.uniform(0, 320)), float(rng.uniform(0, 320)), float(rng.uniform(1.0, 79.0)))
        for _ in range(n_live)
    ]
    pos, alive = floor_scene(cfg, pts)
    cap = mx.array(np.concatenate([rng.uniform(0, 1, n_live), np.zeros(dims(cfg).n - n_live)]).astype(f32))
    return pos, alive, cap


def test_agent_wind_3d_is_alpha_drag_times_the_sampled_wind(cfg3: Config) -> None:
    p = W.params_from_config(cfg3)
    ws = wavy_ws(cfg3, 0.2)
    pos, alive, cap = _agents(cfg3)
    aw = W.agent_wind(ws, pos, cap, alive, cfg3)
    ref = W.sample_wind(ws, pos, p, field_cell=cfg3.world.field_cell, valid=alive)
    a = to_numpy(W.alpha(cap, p))[:, None]
    np.testing.assert_array_equal(to_numpy(aw.force), a * f32(cfg3.physics.drag) * to_numpy(ref.u))
    u = to_numpy(ref.u)
    np.testing.assert_allclose(to_numpy(aw.speed), np.sqrt((u * u).sum(-1)), rtol=1e-6)
    assert aw.force.shape == (dims(cfg3).n, 3) and aw.breaks is None  # storm_break_u = 0
    assert not bool(aw.clip_bad)
    assert np.abs(to_numpy(aw.force)[:12, 2]).max() > 0  # w is not zero: the state is divergent
    # the break flag is the air speed against the threshold (3D)
    on = W.agent_wind(
        ws, pos, cap, alive, storm3d_config("weather.storm_u0=0.01", "weather.storm_break_u=0.05")
    )
    assert on.breaks is not None and np.array_equal(to_numpy(on.breaks), to_numpy(aw.speed) > f32(0.05))


def test_alpha_is_monotone_in_the_cap_and_alpha1_zero_removes_it() -> None:
    cfg = storm3d_config()
    p = W.params_from_config(cfg)
    caps = mx.array(np.linspace(0, 1, 11, dtype=f32))
    al = to_numpy(W.alpha(caps, p))
    assert np.all(np.diff(al) > 0) and np.isclose(al[0], p.alpha0) and np.isclose(al[-1], p.alpha0 + p.alpha1)
    cfg0 = storm3d_config("weather.alpha1=0.0")
    ws = wavy_ws(cfg0, 0.2)
    pos, alive, cap = _agents(cfg0)
    f0 = to_numpy(W.agent_wind(ws, pos, cap, alive, cfg0).force)
    f0_cap0 = to_numpy(W.agent_wind(ws, pos, mx.zeros_like(cap), alive, cfg0).force)
    assert np.array_equal(f0, f0_cap0)  # every agent feels alpha0·u whatever its cap
    ref = to_numpy(W.sample_wind(ws, pos, W.params_from_config(cfg0), valid=alive).u)
    np.testing.assert_allclose(f0, f32(cfg0.weather.alpha0) * f32(cfg0.physics.drag) * ref, rtol=1e-6)


def test_agent_wind_2d_is_the_layer_one_wind_with_alpha0(cfg2: Config) -> None:
    """2D: ``u_air`` is the layer-1 wind bilinear on its C-grid, ``alpha = alpha0``, ``alpha1`` inert."""
    p = W.params_from_config(cfg2)
    ws = wavy_ws(cfg2, 0.15)
    rng = np.random.default_rng(5)
    pts = [(float(rng.uniform(0, 320)), float(rng.uniform(0, 320))) for _ in range(10)]
    pos, alive = floor_scene(cfg2, pts)
    aw = W.agent_wind(ws, pos, None, alive, cfg2)
    # an independent bilinear sample of u1 at (i + 1, j + 1/2) dx and v1 at (i + 1/2, j + 1) dx
    u1, v1, n = to_numpy(ws.u1), to_numpy(ws.v1), p.n

    def bil(a, gx, gy):
        i0, j0 = math.floor(gx), math.floor(gy)
        ax, ay = gx - i0, gy - j0
        g = lambda j, i: float(a[j % n, i % n])  # noqa: E731
        return (1 - ay) * ((1 - ax) * g(j0, i0) + ax * g(j0, i0 + 1)) + ay * (
            (1 - ax) * g(j0 + 1, i0) + ax * g(j0 + 1, i0 + 1)
        )

    P = to_numpy(pos)
    ref = np.array(
        [
            [bil(u1, x / p.dx - 1.0, y / p.dx - 0.5), bil(v1, x / p.dx - 0.5, y / p.dx - 1.0)]
            for x, y in P[:10]
        ]
    )
    np.testing.assert_allclose(
        to_numpy(aw.force)[:10], p.alpha0 * cfg2.physics.drag * ref, rtol=1e-4, atol=1e-7
    )
    np.testing.assert_allclose(to_numpy(aw.speed)[:10], np.sqrt((ref * ref).sum(-1)), rtol=1e-4, atol=1e-7)
    assert aw.force.shape == (dims(cfg2).n, 2) and not bool(aw.clip_bad)
    cfg_a1 = storm2d_config("weather.alpha1=0.5")
    aw1 = W.agent_wind(ws, pos, None, alive, cfg_a1)
    assert np.array_equal(to_numpy(aw1.force)[:10], to_numpy(aw.force)[:10])  # alpha1 is inert in 2D


# --- the closed forms for a lone grounded agent ------------------------------------------------------------


@pytest.mark.parametrize("grip", [0.0, 1.0])
def test_lone_grounded_agent_drifts_at_alpha_u_over_the_grip_drag(cfg3: Config, grip: float) -> None:
    """3D, on the floor: ``Δxy = alpha·φ(z)·u1/(1 + grip_gain·grip)`` via ``forces3`` and ``integrate3``."""
    d, ph = dims(cfg3), cfg3.physics
    n = d.n
    pos, alive = floor_scene(cfg3, [(160.0, 160.0, d.z_lo)])
    cap = mx.full((n,), 0.5, dtype=mx.float32)
    ws = W.unpack(fixed_wx(cfg3, u1=0.1, v1=-0.05))
    aw = W.agent_wind(ws, pos, cap, alive, cfg3)
    p = W.params_from_config(cfg3)
    alpha = p.alpha0 + p.alpha1 * 0.5
    phi = min(d.z_lo / p.z_sl, 1.0)  # the surface-layer factor at the agent's z
    F = physics.forces3(
        pos, mx.zeros((n,)), mx.zeros((n,)), zero_actions(n), zero_motion(n), cap, _nbr(cfg3), alive,
        mx.full((n, B), -1, dtype=mx.int32), cfg3, wind=aw.force,
    )  # fmt: skip
    it = physics.integrate3(
        pos, mx.zeros((n,)), mx.zeros((n,)), F, mx.full((n,), grip), mx.zeros((n,)), mx.zeros((n,)),
        physics.grounded(pos, cfg3), alive, cfg3,
    )  # fmt: skip
    want = alpha * phi * np.array([0.1, -0.05]) / (1.0 + ph.grip_gain * grip)
    np.testing.assert_allclose(to_numpy(it.vel)[0, :2], want, rtol=1e-5)
    dxy = to_numpy(it.pos)[0, :2] - to_numpy(pos)[0, :2]
    np.testing.assert_allclose(dxy, want, atol=2e-5)  # the position is float32 near 160
    assert to_numpy(it.pos)[0, 2] == f32(d.z_lo)  # gravity presses it on the floor
    assert int(it.v_clips) == 0


def test_vertical_wind_is_never_gripped(cfg3: Config) -> None:
    """``integrate3``: ``gamma_z = drag`` whatever the grip, while the horizontal drag grows on the floor."""
    d, ph = dims(cfg3), cfg3.physics
    n = d.n
    pos, alive = floor_scene(cfg3, [(160.0, 160.0, d.z_lo)])
    force = np.zeros((n, 3), f32)
    force[0] = (0.1, 0.0, 0.05)  # the wind force of an updraft over a grounded agent
    out = {}
    for grip in (0.0, 1.0):
        it = physics.integrate3(
            pos, mx.zeros((n,)), mx.zeros((n,)), mx.array(force), mx.full((n,), grip), mx.zeros((n,)),
            mx.zeros((n,)), physics.grounded(pos, cfg3), alive, cfg3,
        )  # fmt: skip
        out[grip] = to_numpy(it.vel)[0]
    assert out[0.0][2] == out[1.0][2] == f32(0.05 / ph.drag)
    assert np.isclose(out[0.0][0] / out[1.0][0], 1.0 + ph.grip_gain)
    # and agent_wind passes w through: a divergent state gives a non-zero vertical force = alpha·drag·w
    ws = wavy_ws(cfg3, 0.2)
    pts = [(40.0 * k + 7.0, 31.0 * k + 5.0, 3.0 + 6.0 * k) for k in range(12)]
    pos2, alive2 = floor_scene(cfg3, pts)
    cap = mx.full((n,), 0.5, dtype=mx.float32)
    aw = W.agent_wind(ws, pos2, cap, alive2, cfg3)
    ref = W.sample_wind(ws, pos2, W.params_from_config(cfg3), valid=alive2)
    np.testing.assert_array_equal(
        to_numpy(aw.force)[:, 2], to_numpy(W.alpha(cap, W.params_from_config(cfg3)) * ph.drag * ref.u[:, 2])
    )


def test_lone_agent_2d_drifts_at_alpha0_u_over_the_grip_drag(cfg2: Config) -> None:
    d, ph = dims(cfg2), cfg2.physics
    n = d.n
    pos, alive = floor_scene(cfg2, [(160.0, 160.0)])
    ws = W.unpack(fixed_wx(cfg2, u1=0.15, v1=0.05))
    aw = W.agent_wind(ws, pos, None, alive, cfg2)
    nbr = no_neighbors(n, d.k, 1.0)._replace(off=mx.zeros((n, d.k, 2)))
    for grip in (0.0, 1.0):
        F = physics.forces(
            pos, mx.zeros((n,)), zero_actions(n), nbr, alive,
            mx.full((n, B), -1, dtype=mx.int32), cfg2, wind=aw.force,
        )  # fmt: skip
        new_pos, v, _, clips = physics.integrate(
            pos, mx.zeros((n,)), F, mx.full((n,), grip), mx.zeros((n,)), alive, cfg2
        )
        want = cfg2.weather.alpha0 * np.array([0.15, 0.05]) / (1.0 + ph.grip_gain * grip)
        np.testing.assert_allclose(to_numpy(v)[0], want, rtol=1e-5)
        np.testing.assert_allclose(to_numpy(new_pos)[0] - 160.0, want, atol=2e-5)
        assert int(clips) == 0


# --- clips and their counts ---------------------------------------------------------------------------------


def test_horizontal_clip_keeps_the_direction_and_dead_rows_are_not_counted(cfg3: Config) -> None:
    p = W.params_from_config(cfg3)
    ws = W.unpack(fixed_wx(cfg3, u1=0.4, v1=0.3))  # |u_h| = 0.5 above u_max = 0.3 away from the floor
    pos, alive = floor_scene(
        cfg3, [(50.0, 50.0, 10.0), (60.0, 60.0, 12.0), (70.0, 70.0, 20.0), (80.0, 80.0, 10.0)]
    )
    alive = mx.array(np.arange(dims(cfg3).n) < 3)  # row 3 is dead but keeps its stale mid-water position
    plain = W.sample_wind(ws, pos, p, field_cell=cfg3.world.field_cell)
    masked = W.sample_wind(ws, pos, p, field_cell=cfg3.world.field_cell, valid=alive)
    assert int(plain.n_clip_h) == 4 and int(masked.n_clip_h) == 3  # valid=None counts every row
    u = to_numpy(masked.u)[:4]
    np.testing.assert_allclose(np.hypot(u[:, 0], u[:, 1]), p.u_max, rtol=1e-6)  # the clip itself is unchanged
    np.testing.assert_allclose(u[:, 1] / u[:, 0], 0.3 / 0.4, rtol=1e-5)  # the direction is kept
    cap = mx.full((dims(cfg3).n,), 0.5, dtype=mx.float32)
    aw = W.agent_wind(ws, pos, cap, alive, cfg3)
    assert bool(aw.clip_bad)
    quiet = W.agent_wind(W.unpack(fixed_wx(cfg3, u1=0.1)), pos, cap, alive, cfg3)
    assert not bool(quiet.clip_bad)


def test_vertical_clip_applies_in_the_edge_layers_only(cfg3: Config) -> None:
    cfg = storm3d_config("weather.w_edge=0.001")
    p = W.params_from_config(cfg)
    ws = wavy_ws(cfg, 0.2)
    free = replace(p, w_edge=1.0)  # the unclipped reference
    xs = np.linspace(5.0, 315.0, 24, dtype=f32)
    edge = [(float(x), 100.0, 2.0) for x in xs] + [
        (float(x), 100.0, 78.0) for x in xs
    ]  # floor and surface layers
    mid = [(float(x), 100.0, 40.0) for x in xs]  # mid-column
    pos, alive = floor_scene(cfg, edge + mid)
    clipped = W.sample_wind(ws, pos, p, field_cell=cfg.world.field_cell, valid=alive)
    ref = W.sample_wind(ws, pos, free, field_cell=cfg.world.field_cell, valid=alive)
    w, w0 = to_numpy(clipped.u)[:, 2], to_numpy(ref.u)[:, 2]
    n_e = len(edge)
    assert np.all(np.abs(w[:n_e]) <= f32(0.001) + 1e-9)
    assert np.array_equal(w[n_e : n_e + len(mid)], w0[n_e : n_e + len(mid)])  # mid-column w is not clipped
    assert np.abs(w0[n_e : n_e + len(mid)]).max() > 0.001  # and it exceeds w_edge: the test is not vacuous
    assert int(clipped.n_clip_w) == int((np.abs(w0[:n_e]) > f32(0.001)).sum()) > 0
    # dead rows are not counted
    none_alive = mx.zeros_like(alive)
    assert int(W.sample_wind(ws, pos, p, field_cell=cfg.world.field_cell, valid=none_alive).n_clip_w) == 0


def test_a_wind_force_above_the_speed_budget_is_clipped_by_integrate3(cfg3: Config) -> None:
    """R3 keeps the horizontal budget under ``v_max``; the 3D norm clip of ``integrate3`` is the net."""
    d, ph = dims(cfg3), cfg3.physics
    n = d.n
    pos, alive = floor_scene(cfg3, [(100.0, 100.0, 30.0)])
    force = np.zeros((n, 3), f32)
    force[0] = (0.6, 0.0, 0.3)
    it = physics.integrate3(
        pos, mx.zeros((n,)), mx.zeros((n,)), mx.array(force), mx.zeros((n,)), mx.zeros((n,)), mx.zeros((n,)),
        physics.grounded(pos, cfg3), alive, cfg3,
    )  # fmt: skip
    v = to_numpy(it.vel)[0]
    assert int(it.v_clips) == 1 and np.isclose(np.linalg.norm(v), ph.v_max, rtol=1e-6)
    assert np.isclose(v[0] / v[2], 2.0, rtol=1e-5)  # the direction is kept


# --- the engine under a strong prescribed wind --------------------------------------------------------------


def test_engine_invariants_hold_under_a_strong_wind_for_200_steps(cfg3: Config) -> None:
    """Eager 3D steps from a uniform strong wind (0.27 in layer 1, 0.2 in layer 2): ``full_check`` is clean on
    the final state and every step, no ``POS_RANGE``, ``|vel| <= v_max (1 + 1e-6)``, finite weather."""
    ph = cfg3.physics
    s = init_state(cfg3)._replace(wx=fixed_wx(cfg3, u1=0.25, v1=0.1, u2=-0.2, v2=0.05))
    step = make_step(cfg3, compile=False)
    wind0 = W.agent_wind(W.unpack(s.wx), s.pos, s.buoy_cap, s.alive, cfg3)
    assert float(to_numpy(wind0.speed).max()) > 0.1  # the wind the agents start in is strong
    start = to_numpy(s.pos).copy()
    for t in range(200):
        prev = s
        s, out = step(s)
        mx.eval(s, out)
        v = np.linalg.norm(to_numpy(s.vel).astype(np.float64), axis=-1)
        assert v.max() <= ph.v_max * (1 + 1e-6), t
        if t % 20 == 19 or t < 3:
            assert invariants.full_check(s, cfg3, prev, out) == [], t
    assert invariants.full_check(s, cfg3) == []
    diag = int(to_numpy(s.diag))
    assert not diag & ERROR_MASK and not diag & Diag.POS_RANGE
    alive = to_numpy(s.alive)
    assert alive.sum() > 0 and np.isfinite(to_numpy(s.wx)).all()
    # the wind moved the population: the mean drift of the survivors is the wind's, not a random walk's
    end = to_numpy(s.pos)
    both = alive & (to_numpy(s.uid) < 64)
    assert both.any() and np.abs(((end[both, 0] - start[both, 0] + 160.0) % 320.0) - 160.0).mean() > 5.0
