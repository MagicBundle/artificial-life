"""The coupling of the sphere weather to the engine (``alife.weather_sph``, DESIGN §46.6) on prescribed
fields:
the vertical map of the shell onto the atmosphere and the rain/seeding audit of docs/SPHERE.md 4.2 for every
preset, the wind formula against the box's, the clips, the force, the hazard and the shelter."""

from __future__ import annotations

from dataclasses import replace

import mlx.core as mx
import numpy as np
import pytest

from alife import lifecycle, weather, weather_sph, weather_tab
from alife.weather import RainIn, WeatherParams
from sphere_weather_helpers import NW, RADIUS, weather_cfg

PRESETS = ("tiny3s", "mini3s", "64k3s")
H1, H2 = 32.0, 48.0


def geo_of(which: str) -> weather_sph.WeatherGeometry:
    return weather_sph.WeatherGeometry.from_config(weather_cfg(which))


def rest_ws(nw: int, **over: np.ndarray) -> weather_sph.WeatherStateS:
    n = 12 * nw * nw
    base = dict(
        h1=np.full(n, H1),
        h2=np.full(n, H2),
        u1=np.zeros((n, 3)),
        u2=np.zeros((n, 3)),
        Q=np.full(n, 0.7),
        P=np.zeros(n),
        Dc=np.zeros(n),
    )
    base.update(over)
    return weather_sph.WeatherStateS(
        *(mx.array(base[k], dtype=mx.float32) for k in weather_sph.WeatherStateS._fields)
    )


def positions(cfg, which: str, z: float, n: int = 400, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1)[:, None]
    return v * (RADIUS[which] + z)


# --- the vertical map ---------------------------------------------------------------------------------------


def test_the_vertical_map_ratio_of_the_presets() -> None:
    got = {w: geo_of(w).s for w in PRESETS}
    assert (
        got["tiny3s"] == pytest.approx(5.0)
        and got["mini3s"] == pytest.approx(80 / 24)
        and got["64k3s"] == pytest.approx(2.0)
    )


@pytest.mark.parametrize("which", PRESETS)
def test_every_preset_has_rain_food_and_the_first_draft_rule_did_not(which: str) -> None:
    """The first-draft rule overlapped the cloud [zi, zi + dc] (weather units) with the shell's layers:
    for tiny3s and mini3s no layer overlaps [32, 52] (a storm world without food). The map divides by s."""
    geo = geo_of(which)
    edges = geo.z_edges
    assert (geo.height < 32.0) == (
        which in ("tiny3s", "mini3s")
    )  # the shell is thinner than the rest interface
    zi, dc = mx.full((5,), 32.0), mx.full((5,), 20.0)
    w = np.array(geo.cloud_weights(zi, dc))  # (FZ, 5)
    assert w.shape == (geo.layers, 5) and np.allclose(w.sum(axis=0), 1.0, atol=1e-6) and (w >= 0).all()
    lo, hi = 32.0 / geo.s, 52.0 / geo.s
    used = np.flatnonzero(w[:, 0] > 0)
    assert used.size >= 1 and edges[used[0]] < hi and edges[used[-1] + 1] > lo
    # outside the cloud nothing is fed; inside, the share follows the overlap fraction (non-uniform layers)
    assert w[0, 0] == 0.0 or edges[1] > lo
    first_draft = np.clip(
        (np.minimum(52.0, edges[1:]) - np.maximum(32.0, edges[:-1])) / np.diff(edges), 0, 1
    ).sum()
    if which in ("tiny3s", "mini3s"):
        assert first_draft == 0.0
    else:
        assert first_draft > 0.0  # 64k3s's first draft fed only the top few layers


@pytest.mark.parametrize("which", PRESETS)
def test_the_column_food_is_rain_gain_times_the_rain_less_the_floors(which: str) -> None:
    cfg = weather_cfg(which, "weather.rain_gain=40000")
    geo = weather_sph.WeatherGeometry.from_config(cfg)
    nw = NW[which]
    n = 12 * nw * nw
    rng = np.random.default_rng(3)
    p_rate = rng.uniform(0.0, 4e-3, n).astype(np.float32)
    zi = rng.uniform(28.0, 36.0, n).astype(np.float32)
    dc = rng.uniform(5.0, 30.0, n).astype(np.float32)
    out = np.array(weather_sph.rain_regen(RainIn(mx.array(p_rate), mx.array(zi), mx.array(dc)), cfg))
    assert out.dtype == np.int32 and out.shape == (geo.layers, 12 * cfg.world.nside**2)
    q = 40000.0 * np.array(geo.block_mean(mx.array(p_rate)), dtype=np.float64)
    col = out.sum(axis=0)
    used = (out > 0).sum(axis=0)
    assert (col <= q + 1e-3).all()
    assert (col >= q - 1.0 * np.maximum(used, 1) - 1).all()  # each used layer loses less than one quantum
    assert (col > 0.5 * q[q > 100].min()).any() and col.sum() > 0.9 * q.sum() - out.shape[0] * out.shape[1]


@pytest.mark.parametrize("which", PRESETS)
def test_the_seeding_band_is_not_empty_where_it_does_not_rain(which: str) -> None:
    """Where it does not rain ``Dc`` is 0, so the band uses ``max(mean Dc, seed_dmin)``: always some
    layers."""
    cfg = weather_cfg(which, "weather.seed_gain=0.05")
    geo = weather_sph.WeatherGeometry.from_config(cfg)
    ws = rest_ws(NW[which])
    nf = cfg.world.nside
    nl = geo.layers
    chem = np.zeros((3, nl, 12 * nf * nf), np.float32)
    chem[2] = 1.0  # uniform c2: no excess, no seeding
    r = np.array(weather_sph.seeding_response(mx.array(chem), ws, cfg))
    assert r.shape == (12 * nf * nf,) and (r == 0.0).all()
    p = weather_sph.params_from_config(cfg)
    zi = mx.full((3,), 32.0)
    w = np.array(geo.cloud_weights(zi, mx.maximum(mx.zeros((3,)), p.seed_dmin)))
    assert (w.sum(axis=0) > 0.999).all() and (w > 0).sum(axis=0).min() >= 1
    # an excess of c2 in the cloud layers of one column seeds exactly that column
    used = np.flatnonzero(w[:, 0] > 0)
    chem[2, used, 7] = 40.0
    r = np.array(weather_sph.seeding_response(mx.array(chem), ws, cfg))
    assert r[7] > 0.5 and r.sum() - r[7] < 1e-6
    qs = np.array(weather_sph.seeding_qs(mx.array(chem), ws, cfg))
    assert qs.shape == (12 * NW[which] ** 2,) and qs.min() < 1.0 - 0.02 and qs.max() == 1.0
    assert (qs < 1.0).sum() == geo.ratio**2  # the pixel's children, and only them


def test_children_and_parents_agree() -> None:
    geo = geo_of("tiny3s")
    d = geo.dev
    ch = np.array(d.children).reshape(-1, geo.ratio**2)
    assert (np.array(d.parent)[ch] == np.arange(ch.shape[0])[:, None]).all()
    a = mx.array(np.random.default_rng(0).normal(size=12 * NW["tiny3s"] ** 2), dtype=mx.float32)
    assert np.allclose(np.array(geo.block_sum(a)), np.array(geo.block_mean(a)) * geo.ratio**2, atol=1e-5)
    assert np.array(geo.spread(geo.block_mean(a))).shape == a.shape


def test_the_rain_band_of_the_audit_is_a_set_of_layers_of_the_cloud() -> None:
    for w in PRESETS:
        lo, hi = geo_of(w).rain_band()
        assert 0 <= lo < hi <= geo_of(w).layers


# --- the wind on prescribed fields --------------------------------------------------------------------------


def _divergent(nw: int, k1: float, k2: float) -> weather_sph.WeatherStateS:
    x = weather_tab.weather_tab(nw).x
    g = np.array([0.0, 0.0, 1.0])[None, :] - x[:, 2:3] * x  # the tangent gradient of z: divergence -2z
    return rest_ws(nw, u1=k1 * g, u2=k2 * g)


@pytest.mark.parametrize("which", PRESETS)
def test_the_wind_profile_is_the_boxs_in_the_relative_height(which: str) -> None:
    """The sphere's sampled wind at height z equals the box formula at z_w = s·z: the horizontal part
    ``φ(z_w)·u1`` below the interface and ``u2`` above, and ``w_shell = w_w/s`` from the same divergences."""
    cfg = weather_cfg(which)
    p = weather_sph.params_from_config(cfg)
    geo = weather_sph.WeatherGeometry.from_config(cfg)
    ws = _divergent(NW[which], 0.03, -0.02)
    wf = weather_sph.wind_fields(ws, p, geo.radius)
    nw = NW[which]
    for z_frac in (0.01, 0.1, 0.3, 0.45, 0.6, 0.8, 0.97):
        z = z_frac * geo.height
        pos = positions(cfg, which, z)
        # keep away from the 8 pole cells (the divergence of the gradient field is 10% off there, S4a G1)
        pos = pos[np.abs(pos[:, 2]) < 0.9 * (RADIUS[which] + z)]
        aw = weather_sph.sample_wind(ws, mx.array(pos, dtype=mx.float32), p, geo, fields=wf)
        u = np.array(aw.u)
        up = pos / np.linalg.norm(pos, axis=1)[:, None]
        w_sh = (u * up).sum(1)
        cell = np.array(weather_sph.sphere.ang2pix(mx.array(pos, dtype=mx.float32), nw))
        rows = np.array(wf.rows)[cell]
        zw = geo.s * z
        zi = rows[:, 0]
        # the box's formula at (zw, zi, d1, d2, jump), scalars per row
        _, _, w_box = weather._wind_formula(
            mx.full((len(cell),), zw),
            mx.array(zi),
            (mx.zeros(1),) * 4,
            mx.array(rows[:, 1]),
            mx.array(rows[:, 2]),
            mx.array(rows[:, 3]),
            p,
        )
        w_box = np.array(w_box)
        edge = (z < geo.z_first) or (z >= geo.z_last)
        if edge:
            w_box = np.clip(w_box, -p.w_edge, p.w_edge)
        assert np.abs(w_sh * geo.s - w_box).max() < 2e-6 + 1e-5 * np.abs(w_box).max(), z_frac
        # horizontal: below the interface phi(zw)·u1 (re-projected), above u2
        u1c, u2c = rows[:, 4:7], rows[:, 7:10]
        phi = min(zw / p.z_sl, 1.0)
        want = np.where((zw < zi)[:, None], phi * u1c, u2c)
        want = want - (want * up).sum(1)[:, None] * up
        got = u - w_sh[:, None] * up
        assert np.abs(got - want).max() < 1e-6


@pytest.mark.parametrize("which", PRESETS)
def test_w_vanishes_at_the_floor_and_the_lid_and_is_continuous_in_the_interior(which: str) -> None:
    cfg = weather_cfg(which)
    p = replace(weather_sph.params_from_config(cfg), w_edge=1.0)  # no clip: look at the formula
    geo = weather_sph.WeatherGeometry.from_config(cfg)
    ws = _divergent(NW[which], 0.03, -0.02)
    pos0 = positions(cfg, which, 0.0, 200)
    pos0 = pos0[np.abs(pos0[:, 2]) < 0.8 * RADIUS[which]]
    for z, tol in ((0.0, 1e-5), (geo.height, 1e-4)):
        pos = pos0 / np.linalg.norm(pos0, axis=1)[:, None] * (RADIUS[which] + z)
        aw = weather_sph.sample_wind(ws, mx.array(pos, dtype=mx.float32), p, geo)
        up = pos / np.linalg.norm(pos, axis=1)[:, None]
        assert np.abs((np.array(aw.u) * up).sum(1)).max() < tol


def test_the_wind_is_clipped_counted_and_tangent_on_a_solid_rotation() -> None:
    which = "tiny3s"
    cfg = weather_cfg(which)
    p = weather_sph.params_from_config(cfg)
    geo = weather_sph.WeatherGeometry.from_config(cfg)
    nw = NW[which]
    x = weather_tab.weather_tab(nw).x
    omega = np.array([0.0, 0.0, 1.0])
    ws = rest_ws(nw, u1=0.5 * np.cross(omega, x), u2=0.1 * np.cross(omega, x))
    z = 0.2 * geo.height  # above the shear layer (z_w = 5 > z_sl), below the interface
    pos = positions(cfg, which, z, 300)
    pos = pos[np.abs(pos[:, 2]) < 0.95 * (RADIUS[which] + z)]  # off the poles, where u1 ~ 0 anyway
    alive = np.ones(len(pos), bool)
    alive[::2] = False
    aw = weather_sph.sample_wind(ws, mx.array(pos, dtype=mx.float32), p, geo, valid=mx.array(alive))
    u = np.array(aw.u)
    up = pos / np.linalg.norm(pos, axis=1)[:, None]
    assert np.abs((u * up).sum(1)).max() < 5e-4  # horizontal; the discrete divergence of the rotation is ~0
    sp = np.linalg.norm(u, axis=1)
    assert sp.max() <= p.u_max * (1 + 1e-5)
    fast = np.linalg.norm(np.cross(omega, pos / np.linalg.norm(pos, axis=1)[:, None]), axis=1) * 0.5 > p.u_max
    assert (
        abs(int(aw.n_clip_h) - int((fast & alive).sum())) <= 3 and int(aw.n_clip_w) == 0
    )  # cell-constant wind
    # the direction is kept: eastward
    east = np.cross(omega, up)
    east /= np.maximum(np.linalg.norm(east, axis=1), 1e-9)[:, None]
    assert ((u * east).sum(1) / np.maximum(sp, 1e-9) > 0.99).all()


def test_the_edge_clip_on_the_vertical_wind_is_counted_over_the_valid_rows() -> None:
    which = "tiny3s"
    cfg = weather_cfg(which)
    p = weather_sph.params_from_config(cfg)
    geo = weather_sph.WeatherGeometry.from_config(cfg)
    ws = _divergent(
        NW[which], 0.2, -0.2
    )  # strong convergence/divergence: |w| beyond w_edge in the edge layers
    pos = positions(cfg, which, 0.9 * geo.z_first, 200)
    pos = pos[np.abs(pos[:, 2]) < 0.8 * RADIUS[which]]
    aw = weather_sph.sample_wind(ws, mx.array(pos, dtype=mx.float32), p, geo)
    up = pos / np.linalg.norm(pos, axis=1)[:, None]
    w = (np.array(aw.u) * up).sum(1)
    assert np.abs(w).max() <= p.w_edge / geo.s * (1 + 1e-5) and int(aw.n_clip_w) > 0


@pytest.mark.parametrize("which", ("tiny3s", "64k3s"))
def test_the_force_is_alpha_drag_times_the_wind_and_the_hazard_reads_its_speed(which: str) -> None:
    cfg = weather_cfg(
        which,
        "weather.storm_p=0.5",
        "weather.storm_u0=0.01",
        "weather.storm_span=0.02",
        "weather.wind_scale=1.0",
    )
    p = weather_sph.params_from_config(cfg)
    geo = weather_sph.WeatherGeometry.from_config(cfg)
    x = weather_tab.weather_tab(NW[which]).x
    ws = rest_ws(NW[which], u1=0.05 * np.cross([0, 0, 1.0], x), u2=0.05 * np.cross([0, 0, 1.0], x))
    pos = positions(cfg, which, 0.6 * geo.height, 64)
    buoy = mx.array(np.linspace(0, 1, len(pos)), dtype=mx.float32)
    alive = mx.ones((len(pos),), dtype=mx.bool_)
    wind = weather.agent_wind(ws, mx.array(pos, dtype=mx.float32), buoy, alive, cfg)
    aw = weather_sph.sample_wind(ws, mx.array(pos, dtype=mx.float32), p, geo, valid=alive)
    alpha = p.alpha0 + p.alpha1 * np.array(buoy)
    assert np.allclose(
        np.array(wind.force), alpha[:, None] * cfg.physics.drag * np.array(aw.u), rtol=1e-5, atol=1e-8
    )
    assert np.allclose(np.array(wind.speed), np.linalg.norm(np.array(aw.u), axis=1), atol=1e-7)
    assert wind.breaks is None and not bool(wind.clip_bad)
    # the drift the observers see is the force over the drag
    drift = weather_sph.agent_drift(weather_sph.pack(ws), mx.array(pos, dtype=mx.float32), buoy, alive, cfg)
    assert np.allclose(np.array(drift), np.array(wind.force) / cfg.physics.drag, atol=1e-8)
    # the hazard: zero below storm_u0, p at u0 + span, a Bernoulli draw per row
    for sp_, want in ((0.0, 0.0), (0.01, 0.0), (0.03, 0.5), (0.5, 0.5)):
        speed = mx.full((20000,), sp_)
        hit = np.array(lifecycle.storm_mask(speed, mx.ones((20000,), dtype=mx.bool_), mx.random.key(1), cfg))
        assert abs(hit.mean() - want) < 0.02, (sp_, hit.mean())


def test_wind_scale_zero_gives_still_air_and_the_break_threshold_reads_the_speed() -> None:
    which = "tiny3s"
    cfg = weather_cfg(which, "weather.wind_scale=0.0", "weather.storm_u0=0.01", "weather.storm_break_u=0.03")
    x = weather_tab.weather_tab(NW[which]).x
    ws = rest_ws(NW[which], u1=0.05 * np.cross([0, 0, 1.0], x), u2=0.05 * np.cross([0, 0, 1.0], x))
    pos = positions(cfg, which, 5.0, 40)
    z = mx.zeros((40,))
    wind = weather.agent_wind(ws, mx.array(pos, dtype=mx.float32), z, mx.ones((40,), dtype=mx.bool_), cfg)
    assert not np.array(wind.force).any() and float(np.array(wind.speed).max()) > 0
    assert wind.breaks is not None and np.array_equal(np.array(wind.breaks), np.array(wind.speed) > 0.03)


def test_shelter_scales_the_force_the_speed_and_the_break_on_a_sphere() -> None:
    which = "tiny3s"
    cfg = weather_cfg(which, "weather.storm_u0=0.01", "weather.storm_break_u=0.03")
    x = weather_tab.weather_tab(NW[which]).x
    ws = rest_ws(NW[which], u1=0.08 * np.cross([0, 0, 1.0], x), u2=0.08 * np.cross([0, 0, 1.0], x))
    pos = positions(cfg, which, 5.0, 50)
    alive = mx.ones((50,), dtype=mx.bool_)
    wind = weather.agent_wind(ws, mx.array(pos, dtype=mx.float32), mx.zeros((50,)), alive, cfg)
    k = mx.array(np.linspace(0.2, 1.0, 50), dtype=mx.float32)
    sh = weather.shelter_wind(wind, k, cfg)
    kk = np.array(k)
    assert np.allclose(np.array(sh.force), np.array(wind.force) * kk[:, None], atol=1e-9)
    assert np.allclose(np.array(sh.speed), np.array(wind.speed) * kk, atol=1e-9)
    assert np.array_equal(np.array(sh.breaks), np.array(sh.speed) > 0.03)
    # drift of the observers with the shelter factor
    d0 = weather_sph.agent_drift(
        weather_sph.pack(ws), mx.array(pos, dtype=mx.float32), mx.zeros((50,)), alive, cfg
    )
    d1 = weather_sph.agent_drift(
        weather_sph.pack(ws), mx.array(pos, dtype=mx.float32), mx.zeros((50,)), alive, cfg, shelter=k
    )
    assert np.allclose(np.array(d1), np.array(d0) * kk[:, None], atol=1e-9)


def test_the_vorticity_and_divergence_diagnostics_of_a_solid_rotation() -> None:
    nw = 12
    p = weather_sph.params_from_config(weather_cfg("tiny3s"))
    p = WeatherParams(
        n=nw, dx=weather_sph.cell_side(36.0, nw), height=80.0, **{k: getattr(p, k) for k in ("dt",)}
    )
    x = weather_tab.weather_tab(nw).x
    om = np.array([0.3, -0.5, 0.8]) * 0.01
    ws = rest_ws(nw, u1=np.cross(om, x) * 36.0)
    z = np.array(weather_sph.vorticity(ws, p, 36.0))
    zeta = 2 * (x @ om)  # the normal curl of a solid rotation of angular velocity om: 2 om·x
    non_pole = np.abs(x[:, 2]) < 0.97
    assert np.abs(z - zeta)[non_pole].max() < 0.03 * np.abs(zeta).max()
    d = np.array(weather_sph.divergence(ws, p, 36.0))
    assert np.abs(d[:, non_pole]).max() < 0.03 * np.abs(zeta).max()
