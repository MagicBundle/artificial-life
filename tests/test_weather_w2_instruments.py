"""W2 instruments (DESIGN §32, spec §6.2, §9.9): the ``coupling`` block, the ecology and weather dumps.

The block's definitions are fixed here on synthetic fields before any run is read: a population drawn from the
tracers' own distribution sits at the null (0.10 above the tracers' top decile), one drawn from the top decile
at 1.0, a counter-steering population has a positive ``hold`` and a blind one 0; the dumps' arrays are what
the step produces.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from alife import weather as W
from alife import weather_obs as WO
from alife.config import Config, apply_overrides
from alife.game import tracers as T
from alife.state import init_state, to_numpy
from alife.step import make_step
from weather_helpers import spun_ws, storm2d_config, storm3d_config

HEIGHT, FC = 80.0, 4.0


def _population(rng: np.random.Generator, n: int, g_scale: float = 1.0) -> dict[str, np.ndarray]:
    """``n`` rows with z uniform over the world and a lognormal updraft-boundary index per row."""
    return {
        "z": rng.uniform(0.0, HEIGHT, n),
        "g": rng.lognormal(-8.0, 1.0, n) * g_scale,
        "w": rng.normal(0.0, 0.01, n),
        "vz_own": np.zeros(n),
    }


# --- the block ------------------------------------------------------------------------------------


def test_bands_are_the_four_z_layers() -> None:
    assert WO.z_band_edges(HEIGHT).tolist() == [0.0, 4.0, 32.0, 56.0, 80.0]
    z = np.array([0.0, 3.99, 4.0, 31.9, 32.0, 55.9, 56.0, 79.9, 80.0])
    assert WO.z_band(z, HEIGHT).tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 3]  # the lid belongs to the top band


def test_agents_drawn_from_the_tracers_distribution_sit_at_the_null() -> None:
    rng = np.random.default_rng(0)
    tr, ag = _population(rng, 60_000), _population(rng, 60_000)
    b = WO.coupling_block(ag, {k: tr[k] for k in ("z", "g", "w")}, HEIGHT, FC)
    assert b["ub_agents_all"] == pytest.approx(0.10, abs=0.006)
    assert all(u == pytest.approx(0.10, abs=0.02) for u in b["ub_agents"])
    assert b["n_agents"] == [int(x) for x in np.bincount(WO.z_band(ag["z"], HEIGHT), minlength=4)]
    assert sum(b["n_tracers"]) == 60_000
    # the threshold is the tracers' own top decile of the band
    zb = WO.z_band(tr["z"], HEIGHT)
    for k in range(4):
        assert b["ub_threshold"][k] == pytest.approx(np.quantile(tr["g"][zb == k], 0.9))


def test_agents_drawn_from_the_top_decile_are_all_above_it() -> None:
    rng = np.random.default_rng(1)
    tr = _population(rng, 40_000)
    zb = WO.z_band(tr["z"], HEIGHT)
    ag = _population(rng, 40_000)
    thr = np.array([np.quantile(tr["g"][zb == k], 0.9) for k in range(4)])
    ag["g"] = thr[WO.z_band(ag["z"], HEIGHT)] * (
        1.0 + rng.uniform(0.01, 3.0, 40_000)
    )  # above the band's decile
    b = WO.coupling_block(ag, {k: tr[k] for k in ("z", "g", "w")}, HEIGHT, FC)
    assert b["ub_agents_all"] == 1.0 and b["ub_agents"] == [1.0] * 4


def test_ub_is_none_for_a_band_without_tracers_or_agents() -> None:
    rng = np.random.default_rng(2)
    tr = _population(rng, 1000)
    tr["z"] = np.full(1000, 10.0)  # every tracer in band 1
    ag = _population(rng, 1000)
    b = WO.coupling_block(ag, {k: tr[k] for k in ("z", "g", "w")}, HEIGHT, FC)
    assert b["ub_agents"][0] is None and b["ub_threshold"][0] is None and b["ub_agents"][1] is not None
    assert b["w_mean_tracers"][0] is None and b["w_mean_tracers"][1] is not None
    none = WO.coupling_block(
        {k: np.zeros(0) for k in ("z", "g", "w", "vz_own")}, {k: tr[k] for k in ("z", "g", "w")}, HEIGHT, FC
    )
    assert none["ub_agents_all"] is None and none["hold"] is None and none["z_mean_agents"] is None


def test_w_and_z_means_per_band() -> None:
    ag = {
        "z": np.array([1.0, 2.0, 10.0, 70.0]),
        "g": np.ones(4),
        "w": np.array([0.01, 0.03, -0.02, 0.5]),
        "vz_own": np.zeros(4),
    }
    tr = {"z": np.array([1.0, 1.0, 60.0]), "g": np.ones(3), "w": np.array([0.1, 0.3, -0.2])}
    b = WO.coupling_block(ag, tr, HEIGHT, FC)
    assert b["w_mean_agents"] == [pytest.approx(0.02), pytest.approx(-0.02), None, pytest.approx(0.5)]
    assert b["w_mean_tracers"] == [pytest.approx(0.2), None, None, pytest.approx(-0.2)]
    assert b["z_mean_agents"] == pytest.approx(20.75) and b["z_mean_tracers"] == pytest.approx(62.0 / 3)


def test_hold_is_minus_the_slope_of_own_vertical_velocity_on_the_air() -> None:
    rng = np.random.default_rng(3)
    n = 20_000
    ag = _population(rng, n)
    ag["z"] = rng.uniform(FC, HEIGHT - FC, n)  # clear of the floor and lid layers
    ag["w"] = rng.normal(0.0, 0.01, n)
    tr = {k: _population(rng, 100)[k] for k in ("z", "g", "w")}
    for k_true in (0.0, 0.3, 1.0, -0.5):
        ag["vz_own"] = -k_true * ag["w"] + rng.normal(0.0, 0.001, n)
        b = WO.coupling_block(ag, tr, HEIGHT, FC)
        assert b["hold"] == pytest.approx(k_true, abs=0.05)
        assert all(h == pytest.approx(k_true, abs=0.15) for h in b["hold_band"] if h is not None)


def test_hold_excludes_weak_wind_and_the_floor_and_lid_layers() -> None:
    n = 2000
    rng = np.random.default_rng(4)
    z = rng.uniform(FC, HEIGHT - FC, n)
    w = rng.normal(0.0, 0.01, n)
    vz = -1.0 * w
    # rows that would spoil the slope if they were counted: weak wind, floor layer, lid layer
    z2 = np.concatenate([z, np.full(500, 20.0), np.full(500, 1.0), np.full(500, 79.0)])
    w2 = np.concatenate(
        [w, rng.normal(0.0, 0.0005, 500), rng.normal(0.0, 0.01, 500), rng.normal(0.0, 0.01, 500)]
    )
    vz2 = np.concatenate(
        [vz, rng.normal(0.0, 0.5, 500), rng.normal(0.0, 0.5, 500), rng.normal(0.0, 0.5, 500)]
    )
    ag = {"z": z2, "g": np.ones(z2.size), "w": w2, "vz_own": vz2}
    tr = {"z": z, "g": np.ones(n), "w": w}
    assert WO.coupling_block(ag, tr, HEIGHT, FC)["hold"] == pytest.approx(1.0, abs=1e-9)
    assert WO.hold_index(np.array([1.0]), np.array([0.1])) is None
    assert WO.hold_index(np.array([1.0, 2.0]), np.array([0.1, 0.1])) is None  # no variance in w


# --- the device half ------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def stepped() -> tuple[Config, object, object, mx.array]:
    """A storm3d state a few steps after a spun-up weather, with its observer channel."""
    cfg = apply_overrides(
        storm3d_config("weather.seed_gain=0.05", "weather.shade_gain=0.02", "weather.rain_gain=1000000"),
        ["game.observe=true", "game.tracers=32"],
    )
    s = init_state(cfg)._replace(wx=W.pack(spun_ws(cfg)))
    fn = make_step(cfg, observe=True)
    for _ in range(5):
        s, out = fn(s)
        mx.eval(s, out)
    tr = T.tracer_init(s.pos, s.buoy_cap, s.alive, cfg)
    mx.eval(*tr)
    return cfg, s, out, tr


def test_the_updraft_gradient_is_the_central_difference_of_the_layer_winds(stepped) -> None:
    cfg, s, _, _ = stepped
    p = W.params_from_config(cfg)
    g = to_numpy(T.updraft_gradient(s.wx, cfg)).astype(np.float64)
    w = to_numpy(W.wind_layers(W.unpack(s.wx), p, 20, cfg.world.field_cell))[:, 2].astype(np.float64)
    gx = (np.roll(w, -1, axis=-1) - np.roll(w, 1, axis=-1)) / (2 * p.dx)
    gy = (np.roll(w, -1, axis=-2) - np.roll(w, 1, axis=-2)) / (2 * p.dx)
    assert g.shape == (20, p.n, p.n)
    np.testing.assert_allclose(g, np.hypot(gx, gy), rtol=1e-4, atol=1e-9)
    assert g.max() > 0


def test_g_is_read_at_the_nearest_layer_and_cell(stepped) -> None:
    cfg, s, _, _ = stepped
    p = W.params_from_config(cfg)
    g = T.updraft_gradient(s.wx, cfg)
    pos = mx.array([[0.5, 0.5, 0.5], [319.9, 3.0, 79.9], [100.0, 200.0, 40.0]], dtype=mx.float32)
    got = to_numpy(T._g_at(g, pos, cfg))
    gn = to_numpy(g)
    want = [gn[0, 0, 0], gn[19, 1, 159], gn[10, 100, 50]]
    assert p.dx == 2.0
    np.testing.assert_array_equal(got, np.array(want, dtype=np.float32))


def test_coupling_inputs_are_the_agents_and_tracers_at_their_positions(stepped) -> None:
    cfg, s, _, tr = stepped
    c = T.coupling_inputs(s, tr, cfg)
    mx.eval(*c)
    p = W.params_from_config(cfg)
    ws = W.unpack(s.wx)
    aw = W.sample_wind(ws, s.pos, p, field_cell=cfg.world.field_cell, valid=s.alive)
    drift = W.agent_drift(s.wx, s.pos, s.buoy_cap, s.alive, cfg)
    np.testing.assert_array_equal(to_numpy(c.w), to_numpy(aw.u[:, 2]))
    np.testing.assert_array_equal(to_numpy(c.vz_own), to_numpy(s.vel[:, 2] - drift[:, 2]))
    np.testing.assert_array_equal(to_numpy(c.z), to_numpy(s.pos[:, 2]))
    np.testing.assert_array_equal(to_numpy(c.tz), to_numpy(tr.pos[:, 2]))
    tw = W.sample_wind(ws, tr.pos, p, field_cell=cfg.world.field_cell).u[:, 2]
    np.testing.assert_array_equal(to_numpy(c.tw), to_numpy(tw))
    assert to_numpy(c.tg).shape == (32,) and (to_numpy(c.tg) >= 0).all()


def test_eco_arrays_are_the_live_rows_columns_and_the_channels_emission(stepped) -> None:
    cfg, s, out, tr = stepped
    cols = T.eco_arrays(s, out.acts, tr, cfg)
    mx.eval(*cols.values())
    assert set(cols) == {
        "alive",
        "uid",
        "clade",
        "cap",
        "z",
        "vz_own",
        "w_air",
        "emit",
        "tracer_z",
        "tracer_age",
    }
    np.testing.assert_array_equal(to_numpy(cols["emit"]), to_numpy(out.acts.emit))  # the channel tap, whole
    np.testing.assert_array_equal(to_numpy(cols["uid"]), to_numpy(s.uid))
    np.testing.assert_array_equal(to_numpy(cols["cap"]), to_numpy(s.buoy_cap))
    # the two columns of the pre-registered P3 statistic and the offline hold analysis: the same quantities
    # as ``coupling_inputs`` (own vertical velocity minus the drift's z; the air's z at the agent)
    p = W.params_from_config(cfg)
    aw = W.sample_wind(W.unpack(s.wx), s.pos, p, field_cell=cfg.world.field_cell, valid=s.alive)
    drift_z = W.agent_drift(s.wx, s.pos, s.buoy_cap, s.alive, cfg)[:, 2]
    assert float(mx.abs(drift_z).max()) > 0 and float(mx.abs(aw.u[:, 1]).max()) > 0  # a mutant is visible
    np.testing.assert_array_equal(to_numpy(cols["vz_own"]), to_numpy(s.vel[:, 2] - drift_z))
    np.testing.assert_array_equal(to_numpy(cols["w_air"]), to_numpy(aw.u[:, 2]))
    assert set(T.eco_arrays(s, out.acts, None, cfg)) == set(cols) - {"tracer_z", "tracer_age"}
    emit = to_numpy(cols["emit"])[to_numpy(s.alive)]
    assert emit.min() >= 0 and emit.max() <= 768 and emit.max() > 0


def test_the_tenth_plane_is_one_minus_qs_and_zero_without_seeding(stepped) -> None:
    cfg, s, _, _ = stepped
    # a crowd in the cloud band over a flat background seeds only itself
    chem = np.full(tuple(s.chem.shape), 1.0, np.float32)
    chem[2, 8:14, 40:46, 40:46] = 9.0
    s2 = s._replace(chem=mx.array(chem))
    planes = T.wx_planes(s2, cfg)
    qs = W.seeding_qs(s2.chem, W.unpack(s2.wx), cfg)
    mx.eval(planes, qs)
    assert planes.shape == (10, 160, 160) and planes.dtype == mx.float32
    np.testing.assert_array_equal(to_numpy(planes[:9]), to_numpy(s2.wx))
    np.testing.assert_array_equal(to_numpy(planes[9]), to_numpy(1.0 - qs))
    p9 = to_numpy(planes[9])
    assert p9.max() > 0.0 and p9.max() <= 0.05 and (p9 >= 0).all()
    assert (p9[:60] == 0).all()  # nothing seeds away from the crowd (rows 80-92 of the weather grid)
    off = apply_overrides(cfg, ["weather.seed_gain=0.0"])
    assert float(to_numpy(T.wx_planes(s2, off)[9]).max()) == 0.0


def test_the_weather_dump_works_in_2d() -> None:
    cfg = apply_overrides(storm2d_config(), ["weather.seed_gain=0.05"])
    s = init_state(cfg)
    planes = T.wx_planes(s, cfg)
    mx.eval(planes)
    assert planes.shape == (10, 160, 160)
