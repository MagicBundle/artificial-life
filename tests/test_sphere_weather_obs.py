"""The weather consumers of the sphere (DESIGN §46.7; sphere series S5): the storm proxy of the metrics block,
the storm layer on the cell graph (STRM, MAPH, persistence), the tracers and the coupling block, the run
dumps, and a played weather world decoded by the independent decoder. Eager (``run.compile=false``).
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

import sphere_decode as sd
from alife import cli, geom, metrics, play, streamrec, weather_sph
from alife.config import dims
from alife.game import observer
from alife.game import tracers as trc
from alife.game.social import SOCIAL_V1
from alife.runio import RunDir
from alife.state import init_state, to_numpy
from alife.storms_sph import SphereParams, StormLayerS
from play_helpers import CADENCE, run_paced
from sphere_weather_helpers import weather_cfg

NW = 16
RADIUS = 36.0


@pytest.fixture(scope="module")
def cfg():
    return weather_cfg("tiny3s", "brain.weight_dtype=float32", "run.compile=false", "game.observe=true")


def cyclone_wx(cfg, centre, amp=0.6, sigma=6.0, rain=0.004) -> np.ndarray:
    """A rest ``wx`` with a solid-body-like cyclone in layer 1 about ``centre``: ``u1 = A (c × x̂) g(d)`` with
    ``g`` a Gaussian of the arc ``d`` (``x̂`` the unit cell centre; the sign of the hemisphere's Coriolis
    parameter, so that it is cyclonic) and rain in its core. The vorticity at the centre is about ``2A/R``."""
    wx = np.array(weather_sph.rest_array(cfg), dtype=np.float64)
    tab = weather_sph.dev_for(weather_sph.params_from_config(cfg), RADIUS)
    x = np.array(tab.x, dtype=np.float64)  # unit cell centres
    c = geom.unit(np.asarray(centre, float))
    d = RADIUS * np.arctan2(np.linalg.norm(np.cross(x, c), axis=1), x @ c)
    g = np.exp(-(d**2) / (2 * sigma**2))
    sign = 1.0 if c[2] >= 0 else -1.0
    wx[2:5] = (sign * amp * np.cross(c, x) * g[:, None]).T
    wx[9] = np.where(d < 1.2 * sigma, rain, 0.0)
    return wx.astype(np.float32)


# --- the metrics block ------------------------------------------------------------------------------------


def test_the_storm_proxy_counts_cyclonic_rainy_footprints_of_the_cell_graph(cfg) -> None:
    s = init_state(cfg)
    assert metrics.snapshot_metrics(s, cfg, heavy=False)["weather"]["storm_proxy"] == 0
    north = cyclone_wx(cfg, [1.0, 0.3, 0.8])
    both = np.maximum(north, 0) * 0 + north  # copy
    south = cyclone_wx(cfg, [-0.4, -0.9, -0.5])
    both[2:5] = north[2:5] + south[2:5]
    both[9] = np.maximum(north[9], south[9])
    for wx, want in ((north, 1), (both, 2)):
        blk = metrics.snapshot_metrics(s._replace(wx=mx.array(wx)), cfg, heavy=False)["weather"]
        assert blk["storm_proxy"] == want and 0.0 < blk["vort_cells"] < 0.5 and json.dumps(blk)
    dry = north.copy()
    dry[9] = 0.0
    assert (
        metrics.snapshot_metrics(s._replace(wx=mx.array(dry)), cfg, heavy=False)["weather"]["storm_proxy"]
        == 0
    )  # no rain


# --- the storm layer --------------------------------------------------------------------------------------


def test_the_storm_layer_tracks_a_cyclone_across_ticks_and_states_places_on_the_planet(cfg, tmp_path) -> None:
    params = SphereParams.from_config(cfg)
    layer = StormLayerS(params, 1e-3, 25)
    path = [
        geom.unit(np.array([1.0, 0.0, 0.7])),
        geom.unit(np.array([1.0, 0.12, 0.7])),
        geom.unit(np.array([1.0, 0.24, 0.7])),
    ]
    ticks = []
    for i, c in enumerate(path):
        ws = weather_sph.unpack(mx.array(cyclone_wx(cfg, c)))
        ticks.append(layer.tick_at(ws, 25 * (i + 1)))
    kinds = [[e["kind"] for e in t.events] for t in ticks]
    assert kinds[0] == ["born", "named"] and kinds[1] == [] and kinds[2] == []  # strong: named at once
    ids = {s["id"] for t in ticks for s in t.strm["storms"]}
    assert ids == {1}
    last = ticks[-1].strm["storms"][0]
    assert {"x", "y", "z", "r", "track", "pred"} <= set(last) and "lon" not in last
    assert math.isclose(math.sqrt(last["x"] ** 2 + last["y"] ** 2 + last["z"] ** 2), RADIUS, rel_tol=1e-3)
    assert len(last["track"]) == 3 and all(len(p) == 4 for p in last["track"]) and last["pred"]
    born = ticks[0].events[0]
    assert {"x", "y", "z"} <= set(born) and ticks[0].line is not None and "z" in ticks[0].line["storms"][0]
    # the wire: STRM and MAPH decode under the independent decoder, planes on the field Nside
    p = ticks[-1].planes
    assert p["P"].shape == (12 * dims(cfg).nside ** 2,) and p["winds"].shape == (4, 12 * dims(cfg).nside ** 2)
    rec = streamrec.encode_maph(step=75, seq=1, planes=p, nside=dims(cfg).nside, k=1, dc_ref=1e-3)
    strm = streamrec.encode_strm(step=75, seq=2, tick=25, obj=ticks[-1].strm)
    head = {
        "protocol": 2, "dims": 3, "height": 16.0, "N": 4, "B": 4, "radius": 1.0, "field_cell": 4.0, "FG": 0,
        "FZ": dims(cfg).fz, "geometry": "sphere", "planet_radius": RADIUS,
        "shell_edges": geom.layer_radii(RADIUS, 16.0, dims(cfg).fz).tolist(), "nside_field": dims(cfg).nside,
        "nside_weather": NW, "nside_territory": 9, "ring_order": "ring", "f0": 0.025, "weather_depth": 80.0,
        "play": {"records": ["MAPH", "STRM"], "versions": {"MAPH": 1, "STRM": 1},
                 "weather": {"enabled": True, "nside": NW, "coarse": dims(cfg).nside, "f0": 0.025,
                             "name_zeta": 0.5, "refs": {}}},
    }  # fmt: skip
    dec = sd.decode(streamrec.header_bytes(head) + rec + strm)
    assert [r.tag for r in dec.records] == ["MAPH", "STRM"]
    assert dec.records[0].a["winds"].shape == (4, 12 * dims(cfg).nside ** 2)
    # the persisted state restores the same tracker (ids, footprints, notes) and refuses other parameters
    state = layer.state()
    again = StormLayerS.restore(params, 1e-3, json.loads(json.dumps(state)))
    assert again.tracker.next_id == layer.tracker.next_id
    assert [d.z for d in again.tracker.tracks[0].dets] == [d.z for d in layer.tracker.tracks[0].dets]
    nxt_a = layer.tick_at(
        weather_sph.unpack(mx.array(cyclone_wx(cfg, geom.unit(np.array([1.0, 0.36, 0.7]))))), 100
    )
    nxt_b = again.tick_at(
        weather_sph.unpack(mx.array(cyclone_wx(cfg, geom.unit(np.array([1.0, 0.36, 0.7]))))), 100
    )
    assert nxt_a.strm == nxt_b.strm
    with pytest.raises(ValueError, match="other weather parameters"):
        StormLayerS.restore(dataclasses.replace(params, radius=40.0), 1e-3, state)


# --- the tracers and the coupling -------------------------------------------------------------------------


def test_tracers_move_with_the_wind_in_the_shell_and_the_coupling_inputs_are_planet_frame(cfg) -> None:
    rcfg = dataclasses.replace(cfg, game=dataclasses.replace(cfg.game, tracers=24, tracer_period=500))
    s = init_state(rcfg)
    wx = np.array(weather_sph.rest_array(rcfg))
    east = np.cross(
        [0.0, 0.0, 1.0], np.array(weather_sph.dev_for(weather_sph.params_from_config(rcfg), RADIUS).x)
    )
    east /= np.maximum(np.linalg.norm(east, axis=1, keepdims=True), 1e-9)
    wx[2:5], wx[5:8] = (0.05 * east).T, (0.05 * east).T  # a steady zonal wind in both layers
    s = s._replace(wx=mx.array(wx.astype(np.float32)))
    d = dims(rcfg)
    tr = trc.tracer_init(s.pos, s.buoy_cap, s.alive, rcfg)
    tr2 = trc.tracer_step(tr, s.wx, rcfg)
    mx.eval(tr, tr2)
    p0, p1 = np.array(tr.pos), np.array(tr2.pos)
    rho = np.linalg.norm(p1, axis=1)
    assert (
        (rho >= float(d.r_lo) - 1e-4).all() and (rho <= float(d.r_hi) + 1e-4).all() and np.isfinite(p1).all()
    )
    step_len = np.linalg.norm(p1 - p0, axis=1)
    assert step_len.max() > 1e-3, "the wind moved the tracers"
    assert step_len.max() <= rcfg.physics.v_max * 1.0001  # the speed limit
    # eastward: the displacement is along the local east (planet axis z), not along a coordinate axis
    up = p0 / np.linalg.norm(p0, axis=1, keepdims=True)
    east_dir = np.cross([0.0, 0.0, 1.0], up)
    ok = (np.linalg.norm(east_dir, axis=1) > 0.3) & (step_len > 1e-4)
    cosang = (
        (p1 - p0)[ok] @ np.zeros(3)
        if False
        else ((p1 - p0)[ok] * east_dir[ok]).sum(1) / (step_len[ok] * np.linalg.norm(east_dir[ok], axis=1))
    )
    assert ok.sum() > 5 and cosang.min() > 0.8
    # coupling inputs: heights are heights, the updraft gradient is nonnegative on (FZ, N), g is gathered
    g = np.array(trc.updraft_gradient(s.wx, rcfg))
    assert g.shape == (d.fz, 12 * NW * NW) and (g >= 0).all() and g.max() > 0
    inp = trc.coupling_inputs(s, tr2, rcfg)
    mx.eval(*inp)
    alive = np.array(inp.alive)
    assert np.array(inp.z)[alive].min() >= 0.0 and np.array(inp.z)[alive].max() <= rcfg.world.height
    assert (
        np.array(inp.tz).min() >= -1e-3
        and np.isfinite(np.array(inp.tg)).all()
        and np.isfinite(np.array(inp.w)).all()
    )
    # z is the height above the planet, not a coordinate
    np.testing.assert_allclose(
        np.array(inp.z)[alive], np.linalg.norm(to_numpy(s.pos)[alive], axis=1) - RADIUS, atol=1e-4
    )
    planes = np.array(trc.wx_planes(s, rcfg))
    assert planes.shape == (12, 12 * NW * NW)


# --- the run and the play ---------------------------------------------------------------------------------


def test_a_run_writes_the_sphere_weather_dumps_and_the_coupling_block(tmp_path: Path) -> None:
    out = tmp_path / "run"
    wc = weather_cfg("tiny3s")
    sets = [
        "world.geometry=sphere", "world.size=1024", "sensing.cell_capacity=32", "physics.floor_eps=0.05",
        "world.radius=36", "world.height=16", "world.nside=8", "world.layers=8", "weather.enabled=true",
        f"weather.cells={NW}", "weather.pools=" + ",".join(str(v) for v in wc.weather.pools),
        "weather.rain_gain=50000", "weather.storm_p=0.002", "brain.weight_dtype=float32", "run.compile=false",
        "game.observe=true", "game.tracers=16", "run.eco_every=10", "run.wx_every=10", "run.metrics_every=10",
        "run.heavy_metrics_every=10", "run.checkpoint_every=10",
    ]  # fmt: skip
    argv = ["run", "--preset", "tiny3d", *(x for kv in sets for x in ("--set", kv))]
    argv += ["--steps", "20", "--out", str(out), "--quiet"]
    assert cli.main(argv) == 0
    rd = RunDir.open(out)
    wx = np.load(rd.dump_path("wx", 20))
    assert wx.shape == (12, 12 * NW * NW) and wx.dtype == np.float32
    eco = np.load(rd.dump_path("eco", 20))
    assert {"z", "vz_own", "w_air", "tracer_z"} <= set(eco.files)
    assert eco["z"].min() >= -1e-3 and eco["z"].max() <= 16.0 + 1e-3  # heights above the planet
    row = rd.metrics_rows()[-1]
    assert (
        "storm_proxy" in row["weather"]
        and row["coupling"]["n_tracers"]
        and len(row["coupling"]["n_agents"]) == 4
    )


def test_a_played_weather_world_sends_maph_and_strm_the_decoder_accepts(tmp_path: Path) -> None:
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(observer, "DEFAULT_SOCIAL", dataclasses.replace(SOCIAL_V1, snapshot_every=25))
        cfg = weather_cfg("tiny3s", *CADENCE, "game.observe=true", "run.compile=false")
        opts = play.PlayOptions(rate=float("inf"), paused=False, fresh=True, unattended=True, tick=25)
        _rd, _res, _pacer, out = run_paced(cfg, tmp_path / "w", 50, [], options=opts)
    dec = sd.decode(out.getvalue())
    tags = [r.tag for r in dec.records]
    assert "MAPH" in tags and "STRM" in tags and "MAP1" not in tags
    h = dec.header
    assert h["nside_weather"] == NW and h["weather_depth"] == cfg.weather.H1 + cfg.weather.H2
    assert h["play"]["weather"]["enabled"] is True and h["play"]["weather"]["coarse"] == dims(cfg).nside
    maph = [r for r in dec.records if r.tag == "MAPH"][-1]
    assert maph.f["nside"] == dims(cfg).nside and maph.f["k"] == 1
    envs = next(r for r in dec.records if r.tag == "ENVS").f["json"]
    assert len(envs["pools"]) == 6 and all({"x", "y", "z", "lon", "lat"} <= set(p) for p in envs["pools"])


def test_the_pools_of_envs_are_where_the_warm_pool_map_has_them_at_a_later_step(cfg) -> None:
    """Review L5: ``play.sphere_pools`` at ``t > 0`` is the centre of ``weather_sph.warm_pools`` (the drift in
    longitude is ``vx·t/(R cos lat)``): the argmax cell of the map is within one cell of the stated place."""

    one = weather_cfg(
        "tiny3s",
        f"weather.pools={RADIUS * math.radians(60):.6f},{RADIUS * math.radians(40):.6f},10,6,1,-0.009,0",
    )
    wp = weather_sph.params_from_config(one)
    x = np.array(weather_sph.dev_for(wp, RADIUS).x, dtype=np.float64)
    cell = math.sqrt(4 * math.pi * RADIUS**2 / (12 * NW * NW))
    for t in (0, 5000):
        [pool] = play.sphere_pools(wp.pools, RADIUS, t)
        ts = np.array(
            weather_sph.warm_pools(wp, RADIUS, mx.array(x, dtype=mx.float32), mx.array(t, dtype=mx.int32))
        )
        peak = x[int(np.argmax(ts))]
        centre = np.array([pool["x"], pool["y"], pool["z"]]) / RADIUS
        arc = RADIUS * math.acos(min(1.0, float(peak @ centre)))
        assert arc < 1.2 * cell, (t, arc, cell)
        assert math.isclose(math.hypot(pool["x"], pool["y"], pool["z"]), RADIUS, rel_tol=1e-3)
    lon0 = play.sphere_pools(wp.pools, RADIUS, 0)[0]["lon"]
    lon1 = play.sphere_pools(wp.pools, RADIUS, 5000)[0]["lon"]
    assert (
        abs(
            ((lon1 - lon0 + 180) % 360)
            - 180
            - math.degrees(-0.009 * 5000 / (RADIUS * math.cos(math.radians(40))))
        )
        < 0.1
    )
