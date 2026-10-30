"""The wind-coupling experiment of S4c (``alife.analysis.w0_sphere_wind``): its geometry helpers, and the
clock-consistent coupling it measures (DESIGN §46.6, "Time and the Courant number"): the wind force on an
agent and the step of a tracer carry the weather time of a step, ``dt``."""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from alife import geom, weather_sph
from alife.analysis import w0_sphere_wind as ww
from alife.analysis.w0_sphere import build_config
from alife.game.tracers import TracerState, tracer_step
from alife.weather import rest_array


def test_disk_points_are_uniform_in_the_cap_and_the_azimuth_winds_the_right_way() -> None:
    rng = np.random.default_rng(1)
    c = geom.unit(np.array([0.3, -0.5, 0.8]))
    pts = ww._disk_points(c, 8.0, 90.0, 4000, rng)
    arc = geom.great_circle_distance(pts, c, 90.0)
    assert np.allclose(np.linalg.norm(pts, axis=1), 1.0) and arc.max() <= 8.0 + 1e-9
    assert (
        abs(np.mean(arc <= 8.0 / math.sqrt(2)) - 0.5) < 0.03
    )  # equal area: half the points within r/sqrt(2)
    east, north = geom.local_frame(c[None])

    def at(a: float) -> float:
        return ww._azimuth(
            np.cos(0.05) * c + np.sin(0.05) * (math.cos(a) * east[0] + math.sin(a) * north[0]), c
        )

    assert at(0.0) == pytest.approx(0.0, abs=1e-9) and at(math.pi / 2) == pytest.approx(math.pi / 2)
    assert ww._fold(np.array([3.5, -3.5, 7.0])) == pytest.approx(
        [3.5 - 2 * math.pi, 2 * math.pi - 3.5, 7 - 2 * math.pi]
    )


def test_a_sphere_agent_and_a_tracer_move_by_alpha_u_dt_not_alpha_u() -> None:
    """A step is ``dt`` of a unit of weather time: the force of the wind on an agent (``agent_wind``) and the
    displacement of a tracer are the wind times ``dt`` times the coupling; the air speed is not scaled."""
    cfg = build_config("tiny3s", ["stormsph16"], ["game.tracers=8", "game.observe=true"])
    p = weather_sph.params_from_config(cfg)
    assert 0.78 < p.dt < 0.86
    rng = np.random.default_rng(0)
    wx = np.array(rest_array(cfg, 0))
    wx[2:5] = 0.02 * rng.normal(
        size=wx[2:5].shape
    )  # a wind field: the layer-1 velocity, tangent-projected below
    ws = weather_sph.unpack(mx.array(wx.astype(np.float32)))
    tab = weather_sph.dev_for(p, cfg.world.radius)
    ws = ws._replace(u1=weather_sph._proj(tab.x, ws.u1), u2=weather_sph._proj(tab.x, ws.u1))
    geo = weather_sph.WeatherGeometry.from_config(cfg)
    pos = mx.array(((cfg.world.radius + 4.0) * geom.unit(rng.normal(size=(16, 3)))).astype(np.float32))
    alive = mx.ones((16,), dtype=mx.bool_)
    cap = mx.zeros((16,), dtype=mx.float32)
    air = weather_sph.sample_wind(ws, pos, p, geo, valid=alive).u
    af = weather_sph.agent_wind(ws, pos, cap, alive, cfg)
    alpha = cfg.weather.alpha0
    assert np.allclose(
        np.array(af.force), alpha * cfg.physics.drag * p.dt * np.array(air), rtol=1e-5, atol=1e-9
    )
    assert np.allclose(
        np.array(af.speed), np.linalg.norm(np.array(air), axis=1), rtol=1e-5
    )  # the air's, unscaled
    tr = TracerState(
        pos,
        mx.full((16,), alpha, dtype=mx.float32),
        mx.zeros((16,), dtype=mx.int32),
        mx.zeros((2,), dtype=mx.uint32),
    )
    moved = np.array(tracer_step(tr, weather_sph.pack(ws), cfg).pos) - np.array(pos)
    radial = np.sum(moved * np.array(pos) / np.linalg.norm(np.array(pos), axis=1, keepdims=True), axis=1)
    # the tracer is put back inside the shell; the horizontal part of the step is alpha * u * dt
    tangent = moved - radial[:, None] * np.array(pos) / np.linalg.norm(np.array(pos), axis=1, keepdims=True)
    want = alpha * p.dt * np.array(air)
    want_t = want - np.sum(want * np.array(pos), axis=1, keepdims=True) * np.array(pos) / np.sum(
        np.array(pos) ** 2, axis=1, keepdims=True
    )
    assert np.allclose(tangent, want_t, atol=2e-6)
