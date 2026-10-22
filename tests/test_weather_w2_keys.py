"""W2 build step 1 (DESIGN §32, W2 spec §9.1(b)(c), §9.2-§9.4): the eight coupling keys and the pure
functions behind them, with the engine step untouched.

Keys and validation; ``cloud_weights`` and the bitwise-unchanged ``rain_regen``; ``rain_zonal``;
``seeding_qs``; ``shade_mult`` and its separable box; the eddy wind (``WindFields.zbar``,
``sample_wind(eddy=)``) against a numpy double; ``agent_wind`` under ``wind_scale``, ``wind_eddy_force`` and
``storm_eddy``; the hazard and the snapshot exposure following the hazard speed.

CPU and fast. The 3D case is ``64k3d`` + ``storm`` (256 rows, L = 320, FG = 80, FZ = 20); the 2D case is
``tiny`` at ``world.size = 320``. The wiring into the step (S17), the metrics block and the ledger tests
are the next build step's.
"""

from __future__ import annotations

import dataclasses
import itertools
from dataclasses import fields, replace

import mlx.core as mx
import numpy as np
import pytest

from alife import lifecycle, metrics, ops
from alife import weather as W
from alife.config import (
    Config,
    ConfigError,
    WeatherConfig,
    config_hash,
    dims,
    load_config,
    validate,
)
from alife.state import init_state, to_numpy
from weather_helpers import fixed_wx, floor_scene, spun_ws, storm2d_config, storm3d_config

NEW_KEYS = (
    "seed_gain",
    "seed_c0",
    "shade_gain",
    "shade_n0",
    "wind_scale",
    "rain_zonal",
    "storm_eddy",
    "wind_eddy_force",
)
# config_hash of `64k3d` + the `storm` overlay at 3799009 (W1): the eight keys at their defaults leave it.
STORM_W1_HASH = "f07fb3d12a99f66c152fdb60a2a75824eee6ffbd874dad770cfb889570ffa0c9"


def _np(a: mx.array) -> np.ndarray:
    return np.asarray(a)


def cw(cfg: Config, **kw) -> Config:
    return replace(cfg, weather=replace(cfg.weather, **kw))


@pytest.fixture(scope="module")
def cfg3() -> Config:
    return storm3d_config()


@pytest.fixture(scope="module")
def cfg2() -> Config:
    return storm2d_config()


@pytest.fixture(scope="module", params=["3d", "2d"])
def kind(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture
def cfg(kind: str, cfg3: Config, cfg2: Config) -> Config:
    return cfg3 if kind == "3d" else cfg2


# --- keys, hash, validation (spec §9.1 b, c) -------------------------------------------------------------


def test_the_eight_keys_are_the_last_of_the_section_with_inert_defaults() -> None:
    names = [f.name for f in fields(WeatherConfig)]
    assert tuple(names[-8:]) == NEW_KEYS
    wc = WeatherConfig()
    assert (wc.seed_gain, wc.shade_gain, wc.wind_scale) == (0.0, 0.0, 1.0)
    assert (wc.rain_zonal, wc.storm_eddy, wc.wind_eddy_force) == (False, False, False)
    assert (wc.seed_c0, wc.shade_n0) == (2.0, 8.0)


def test_the_config_hash_omits_the_eight_keys_at_their_defaults(cfg3: Config) -> None:
    assert config_hash(load_config(preset="64k3d", overlays=["storm"])) == STORM_W1_HASH
    assert config_hash(cfg3) != STORM_W1_HASH  # 256 rows: a different config, so the pin above is not vacuous
    on = config_hash(load_config(preset="64k3d", overlays=["storm"], overrides=["weather.storm_eddy=true"]))
    assert on != STORM_W1_HASH


@pytest.mark.parametrize("key", NEW_KEYS)
def test_every_new_key_is_rejected_while_the_weather_is_off(key: str) -> None:
    """W-0: inert while ``weather.enabled`` is false, so it must keep its default."""
    base = load_config(preset="tiny")
    val = {"rain_zonal": True, "storm_eddy": True, "wind_eddy_force": True}.get(key, 0.05)
    bad = replace(base, weather=replace(base.weather, **{key: val}))
    with pytest.raises(ConfigError, match=f"weather.{key} is inert while weather.enabled is false"):
        validate(bad)


@pytest.mark.parametrize(
    ("key", "ok", "bad"),
    [
        ("seed_gain", [0.0, 0.03, 0.1], [-0.01, 0.11, float("nan"), float("inf")]),
        ("seed_c0", [0.5, 2.0, 40.0], [0.0, -1.0, float("nan"), float("inf")]),
        ("shade_gain", [0.0, 0.02, 0.05], [-0.01, 0.051, float("nan")]),
        ("shade_n0", [0.5, 8.0, 100.0], [0.0, -2.0, float("nan"), float("inf")]),
        ("wind_scale", [0.0, 0.5, 1.0], [-0.1, 1.01, float("nan")]),
    ],
)
def test_the_value_rules(kind: str, cfg3: Config, cfg2: Config, key: str, ok: list, bad: list) -> None:
    c = cfg3 if kind == "3d" else cfg2
    for v in ok:
        assert W.validate_config(cw(c, **{key: v})) == [], (key, v)
    for v in bad:
        msgs = W.validate_config(cw(c, **{key: v}))
        assert any(f"weather.{key}" in m for m in msgs), (key, v, msgs)


def test_the_boolean_controls_validate_in_both_dimensions(cfg: Config) -> None:
    for key in ("rain_zonal", "storm_eddy", "wind_eddy_force"):
        assert W.validate_config(cw(cfg, **{key: True})) == [], key


def test_the_break_threshold_message_names_the_hazard_speed(cfg3: Config) -> None:
    msgs = W.validate_config(cw(cfg3, storm_u0=0.05, storm_break_u=0.02))
    assert len(msgs) == 1
    assert "hazard speed" in msgs[0] and "weather.storm_eddy" in msgs[0] and "layer-2 jet" in msgs[0]
    assert W.validate_config(cw(cfg3, storm_u0=0.05, storm_break_u=0.02, storm_eddy=True)) != []  # same rule


def test_seed_dmin_is_engine_fixed_and_reaches_the_parameters(cfg3: Config) -> None:
    assert W.ENGINE_FIXED["seed_dmin"] == 8.0 == 2 * cfg3.world.field_cell
    p = W.params_from_config(cfg3)
    assert p.seed_dmin == 8.0 and p == W.WeatherParams()


def test_seed_dmin_leaves_the_w0_parameter_dict_alone() -> None:
    """``to_dict`` omits the post-W0 field at its default (the W0 params hashes and files are unchanged) and
    keeps it when it is not the default; ``from_dict`` round-trips both."""
    p = W.WeatherParams()
    assert "seed_dmin" not in p.to_dict()
    q = replace(p, seed_dmin=6.0)
    assert q.to_dict()["seed_dmin"] == 6.0
    assert W.WeatherParams.from_dict(q.to_dict()) == q and W.WeatherParams.from_dict(p.to_dict()) == p


# --- cloud_weights and rain_regen (spec §1.1 item 1, §9.2) ------------------------------------------------


def _rain_regen_w1(rain: W.RainIn, cfg: Config) -> mx.array:
    """``weather.rain_regen`` as W1 built it (3799009), kept verbatim as the reference of the refactor."""
    wc, w = cfg.weather, cfg.world
    fg = round(w.size / w.field_cell)
    pc = W._blocks(rain.P, fg).mean(axis=(1, 3))
    q = wc.rain_gain * pc
    if w.dims != 3:
        return ops.floor_i32(q)
    fz, fc = round(w.height / w.field_cell), w.field_cell
    ps = W._blocks(rain.P, fg).sum(axis=(1, 3))
    den = mx.maximum(ps, 1e-30)
    zi = W._blocks(rain.P * rain.zi, fg).sum(axis=(1, 3)) / den
    dc = W._blocks(rain.P * rain.dc, fg).sum(axis=(1, 3)) / den
    k = mx.arange(fz, dtype=mx.float32).reshape(fz, 1, 1)
    frac = mx.clip((mx.minimum(zi + dc, fc * (k + 1.0)) - mx.maximum(zi, fc * k)) / fc, 0.0, 1.0)
    share = frac / mx.maximum(frac.sum(axis=0), 1e-6)
    out = ops.floor_i32(q * share)
    if wc.rain_floor > 0:
        floor_q = ops.floor_i32(wc.rain_floor * q)
        out = out + mx.concatenate([floor_q[None], mx.zeros((fz - 1, fg, fg), dtype=mx.int32)])
    return out


def _hand_rain(cfg: Config) -> W.RainIn:
    """Ten raining cells with their own level and depth, in blocks with one to four raining cells."""
    p = W.params_from_config(cfg)
    n = p.n
    P, zi, dc = (np.zeros((n, n), np.float32) for _ in range(3))
    zi[:] = 32.0
    cells = [
        (3, 4, 0.01, 20.0, 12.0),
        (3, 5, 0.02, 24.0, 5.0),
        (40, 40, 0.005, 30.0, 25.0),
        (41, 41, 0.007, 33.0, 9.0),
        (100, 7, 0.03, 40.0, 30.0),
        (100, 6, 0.01, 10.0, 70.0),
        (159, 159, 0.02, 60.0, 15.0),
        (0, 0, 0.004, 3.0, 2.0),
        (120, 80, 0.015, 78.0, 4.0),
        (60, 61, 0.012, 1.0, 100.0),
    ]
    for j, i, rate, z, d in cells:
        P[j, i], zi[j, i], dc[j, i] = rate, z, d
    return W.RainIn(mx.array(P), mx.array(zi), mx.array(dc))


@pytest.mark.parametrize("floor", [0.0, 0.25])
@pytest.mark.parametrize("gain", [5e4, 1e6])
def test_rain_regen_is_bitwise_what_w1_built_on_hand_columns_and_a_spun_up_state(
    cfg3: Config, gain: float, floor: float
) -> None:
    c = cw(cfg3, rain_gain=gain, rain_floor=floor)
    spun = W.rain_in(spun_ws(cfg3), c)
    assert float(_np(spun.P).max()) > 0 and (_np(spun.P) > 1e-4).mean() > 0.01  # real rain: not vacuous
    for rain in (_hand_rain(c), spun):
        new, old = W.rain_regen(rain, c), _rain_regen_w1(rain, c)
        mx.eval(new, old)
        assert new.dtype == old.dtype and np.array_equal(_np(new), _np(old))
        assert int(_np(new).sum()) > 0


def test_rain_regen_2d_is_bitwise_what_w1_built(cfg2: Config) -> None:
    c = cw(cfg2, rain_gain=1e6)
    for rain in (_hand_rain(c), W.rain_in(spun_ws(cfg2), c)):
        assert np.array_equal(_np(W.rain_regen(rain, c)), _np(_rain_regen_w1(rain, c)))


def _band_split(zi: float, dc: float, fz: int, fc: float) -> np.ndarray:
    """The share of a cloud ``[zi, zi + dc]`` in each layer, numpy float64."""
    lo, hi = fc * np.arange(fz), fc * (np.arange(fz) + 1)
    frac = np.clip((np.minimum(zi + dc, hi) - np.maximum(zi, lo)) / fc, 0, 1)
    return frac / max(frac.sum(), 1e-6)


def test_cloud_weights_is_the_layer_overlap_share(cfg3: Config) -> None:
    fg, fz, fc = 80, 20, cfg3.world.field_cell
    zi = np.full((fg, fg), 32.0, np.float32)
    dc = np.full((fg, fg), 8.0, np.float32)
    zi[5, 7], dc[5, 7] = 21.5, 13.0
    zi[9, 9], dc[9, 9] = 77.0, 30.0  # runs past the lid
    w = _np(W.cloud_weights(mx.array(zi), mx.array(dc), cfg3))
    assert w.shape == (fz, fg, fg) and w.dtype == np.float32
    for j, i in [(0, 0), (5, 7), (9, 9)]:
        np.testing.assert_allclose(w[:, j, i], _band_split(zi[j, i], dc[j, i], fz, fc), atol=1e-6)
    np.testing.assert_allclose(w.sum(axis=0), 1.0, atol=1e-6)
    assert (w[8:10, 0, 0] == 0.5).all() and w[:8, 0, 0].sum() == 0 and w[10:, 0, 0].sum() == 0


def test_seeding_and_feeding_call_one_cloud_weights(cfg3: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    """The seeding band is the rain food's band: both go through ``weather.cloud_weights``."""
    calls: list[str] = []
    real = W.cloud_weights

    def spy(zi, dc, c):
        calls.append("w")
        return real(zi, dc, c)

    monkeypatch.setattr(W, "cloud_weights", spy)
    c = cw(cfg3, rain_gain=1e6, seed_gain=0.03)
    W.rain_regen(_hand_rain(c), c)
    assert calls == ["w"]
    W.seeding_qs(mx.zeros((3, 20, 80, 80)), spun_ws(cfg3), c)
    assert calls == ["w", "w"]


# --- rain_zonal (spec §3) ---------------------------------------------------------------------------------


def test_a_zonally_uniform_rain_gives_the_same_food_with_rain_zonal(cfg: Config) -> None:
    p = W.params_from_config(cfg)
    n = p.n
    rate = (0.002 * (1 + np.arange(n) % 5))[:, None] * np.ones((1, n))
    zi = (20.0 + 3.0 * (np.arange(n) % 7))[:, None] * np.ones((1, n))
    dc = (8.0 + 2.0 * (np.arange(n) % 3))[:, None] * np.ones((1, n))
    rain = W.RainIn(*(mx.array(a.astype(np.float32)) for a in (rate, zi, dc)))
    c = cw(cfg, rain_gain=1e6)
    plain, zonal = _np(W.rain_regen(rain, c)), _np(W.rain_regen(rain, cw(c, rain_zonal=True)))
    assert plain.shape == zonal.shape and int(plain.sum()) > 10_000
    # the x-mean of identical floats is the float up to its last bit: at most one quantum off in a cell
    assert np.abs(plain - zonal).max() <= 1
    assert abs(int(plain.sum()) - int(zonal.sum())) <= 0.001 * int(plain.sum())


def test_a_point_rain_is_spread_over_its_row_with_the_total_preserved(cfg3: Config) -> None:
    c = cw(cfg3, rain_gain=1e7)
    n = 160
    P = np.zeros((n, n), np.float32)
    zi = np.full((n, n), 32.0, np.float32)
    dc = np.full((n, n), 12.0, np.float32)
    P[50, 33] = 0.05
    rain = W.RainIn(*(mx.array(a) for a in (P, zi, dc)))
    plain, zonal = _np(W.rain_regen(rain, c)), _np(W.rain_regen(rain, cw(c, rain_zonal=True)))
    assert (plain.sum(axis=0) > 0).sum() == 1  # one column
    cols = zonal.sum(axis=0)  # (FG, FG): y, x
    assert (cols > 0).sum() == 80 and (cols[25] > 0).all() and not np.delete(cols, 25, axis=0).any()
    assert abs(int(zonal.sum()) - int(plain.sum())) <= 0.01 * int(
        plain.sum()
    )  # the floors of 3 layers x 80 cells


def test_the_zonal_food_depends_on_latitude_only(cfg: Config) -> None:
    c = cw(cfg, rain_gain=1e6, rain_zonal=True)
    out = _np(W.rain_regen(W.rain_in(spun_ws(cfg), c), c))
    assert int(out.sum()) > 0 and np.array_equal(out, np.broadcast_to(out[..., :1], out.shape))
    plain = _np(W.rain_regen(W.rain_in(spun_ws(cfg), c), cw(c, rain_zonal=False)))
    assert not np.array_equal(plain, np.broadcast_to(plain[..., :1], plain.shape))  # the control varies in x


# --- seeding_qs (spec §1.1, §9.2) -------------------------------------------------------------------------


def _chem(cfg: Config, fill: float | np.ndarray = 0.0) -> mx.array:
    d = dims(cfg)
    return mx.array(np.full((3, *d.field_shape), fill, np.float32))


def _np_seed(chem: np.ndarray, ws: W.WeatherState, cfg: Config) -> np.ndarray:
    """``seeding_qs`` written out in numpy float64 (the band by `_band_split`, not `cloud_weights`)."""
    wc, w = cfg.weather, cfg.world
    p = W.params_from_config(cfg)
    fg = round(w.size / w.field_cell)
    r = p.n // fg
    c2 = chem[2].astype(np.float64)
    if w.dims == 3:
        zi_c = (p.height * _np(ws.h1) / (_np(ws.h1) + _np(ws.h2))).astype(np.float64)
        zi_b = zi_c.reshape(fg, r, fg, r).mean(axis=(1, 3))
        dc_b = np.maximum(_np(ws.Dc).astype(np.float64).reshape(fg, r, fg, r).mean(axis=(1, 3)), 8.0)
        fz = c2.shape[0]
        wts = np.zeros_like(c2)
        for j in range(fg):
            for i in range(fg):
                wts[:, j, i] = _band_split(zi_b[j, i], dc_b[j, i], fz, w.field_cell)
        b = (wts * c2).sum(axis=0)
    else:
        b = c2
    e = np.maximum(b - b.mean() * (1 + 1e-4), 0.0)
    q = 1.0 - wc.seed_gain * e / (e + wc.seed_c0)
    return np.kron(q, np.ones((r, r)))


@pytest.mark.parametrize("fill", [0.0, 0.1, 3.7, 6.03, 16.0])
def test_a_uniform_c2_seeds_nothing_exactly(cfg: Config, fill: float) -> None:
    c = cw(cfg, seed_gain=0.1)
    qs = W.seeding_qs(_chem(c, fill), W.unpack(W.rest_array(c)), c)
    assert qs.shape == (160, 160) and qs.dtype == mx.float32
    assert np.array_equal(_np(qs), np.ones((160, 160), np.float32))
    # a spun-up weather gives every column its own band, whose weights sum to 1 only to rounding
    spun = W.seeding_qs(_chem(c, fill), spun_ws(c), c)
    assert np.array_equal(_np(spun), np.ones((160, 160), np.float32))


def test_a_point_crowd_lowers_qs_only_inside_its_footprint(cfg: Config) -> None:
    c = cw(cfg, seed_gain=0.05, seed_c0=2.0)
    ws = W.unpack(
        W.rest_array(c)
    )  # 3D rest: zi = 32, D_c = 0 -> floored to 8: layers 8 and 9 (weight 1/2 each)
    ch = np.zeros((3, *dims(c).field_shape), np.float32)
    if dims(c).d3:
        ch[2, 8:10, 10, 20] = 10.0
    else:
        ch[2, 10, 20] = 10.0
    qs = _np(W.seeding_qs(mx.array(ch), ws, c))
    b = 10.0
    e = b - (b / 6400.0) * (1 + 1e-4)
    want = 1.0 - 0.05 * e / (e + 2.0)
    inside = (slice(20, 22), slice(40, 42))
    assert np.allclose(qs[inside], want, rtol=1e-6)
    out = qs.copy()
    out[inside] = 1.0
    assert np.array_equal(out, np.ones_like(out))  # exactly 1 everywhere else
    assert 1 - 0.05 <= qs.min() < 1.0


def test_qs_is_bounded_monotone_and_saturating_in_the_excess(cfg: Config) -> None:
    c = cw(cfg, seed_gain=0.08, seed_c0=2.0)
    ws = W.unpack(W.rest_array(c))
    amps = [0.0, 0.05, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    vals = []
    for a in amps:
        ch = np.zeros((3, *dims(c).field_shape), np.float32)
        if dims(c).d3:
            ch[2, 8:10, 3, 3] = a
        else:
            ch[2, 3, 3] = a
        vals.append(float(_np(W.seeding_qs(mx.array(ch), ws, c))[6, 6]))
    assert vals[0] == 1.0 and all(b <= a for a, b in itertools.pairwise(vals))
    assert min(vals) >= 1 - 0.08
    # saturating: the step from 8 to 16 is a fraction of the step from 0.5 to 1 per unit of excess
    assert (vals[-2] - vals[-1]) / 8.0 < 0.25 * (vals[2] - vals[3]) / 0.5
    e = 16.0 * (1 - (1 + 1e-4) / 6400.0)
    assert vals[-1] == pytest.approx(1 - 0.08 * e / (e + 2.0), rel=1e-6)


def test_columns_below_the_lattice_mean_are_not_anti_seeded(cfg: Config) -> None:
    c = cw(cfg, seed_gain=0.1)
    ch = np.full((3, *dims(c).field_shape), 5.0, np.float32)
    if dims(c).d3:
        ch[2, :, 30:32, :] = 0.0
    else:
        ch[2, 30:32, :] = 0.0
    qs = _np(W.seeding_qs(mx.array(ch), spun_ws(c), c))
    assert (qs[60:64] == 1.0).all()  # the low rows
    assert (qs <= 1.0).all()


def test_seeding_equals_the_numpy_double_on_a_spun_up_state(cfg: Config) -> None:
    c = cw(cfg, seed_gain=0.1, seed_c0=1.5)
    rng = np.random.default_rng(3)
    ch = np.clip(rng.gamma(2.0, 1.5, (3, *dims(c).field_shape)), 0, 16).astype(np.float32)
    ws = spun_ws(c)
    got = _np(W.seeding_qs(mx.array(ch), ws, c))
    want = _np_seed(ch, ws, c)
    np.testing.assert_allclose(got, want, rtol=2e-6, atol=2e-7)
    assert got.min() < 1.0 - 0.01 and got.min() >= 0.9 - 1e-6


def test_a_column_with_a_thin_cloud_uses_the_depth_floor(cfg3: Config) -> None:
    """D_c = 0 and D_c = 2 both read as the floor 8 (the same band); D_c = 20 is its own band."""
    c = cw(cfg3, seed_gain=0.1)
    ws = W.unpack(W.rest_array(c))
    ch = np.zeros((3, 20, 80, 80), np.float32)
    ch[2, 8:10, 10, 10] = 16.0  # layers 8 and 9 = [32, 40]
    Dc = np.zeros((160, 160), np.float32)
    got = {}
    for d in (0.0, 2.0, 7.9, 8.0, 20.0):
        Dc[20:22, 20:22] = d
        qs = _np(W.seeding_qs(mx.array(ch), ws._replace(Dc=mx.array(Dc)), c))
        got[d] = qs[20, 20]
    assert got[0.0] == got[2.0] == got[7.9] == got[8.0] < 1.0
    assert (
        got[20.0] > got[8.0]
    )  # a deeper cloud spreads the band over five layers: a fifth of the c2 per layer
    for d, b in ((8.0, 16.0), (20.0, 16.0 * 0.4)):  # the band [32, 52] has weight 0.2 on layers 8 and 9
        e = b - (b / 6400.0) * (1 + 1e-4)
        assert got[d] == pytest.approx(1 - 0.1 * e / (e + 2.0), rel=1e-5)


def test_the_block_means_are_plain_not_rain_weighted(cfg3: Config) -> None:
    """One raining weather cell of a block: the seeding band takes the block mean of D_c (a quarter of it)."""
    c = cw(cfg3, seed_gain=0.1)
    ws = W.unpack(W.rest_array(c))
    Dc = np.zeros((160, 160), np.float32)
    Dc[20, 20] = 40.0  # block mean 10 -> band [32, 42]; a rain-weighted mean would give 40
    ch = np.zeros((3, 20, 80, 80), np.float32)
    ch[2, 8:11, 10, 10] = 16.0
    got = _np(W.seeding_qs(mx.array(ch), ws._replace(Dc=mx.array(Dc)), c))[20, 20]
    w = _band_split(32.0, 10.0, 20, 4.0)
    b = 16.0 * w[8:11].sum()
    e = b - (b / 6400.0) * (1 + 1e-4)
    assert got == pytest.approx(1 - 0.1 * e / (e + 2.0), rel=1e-5)


def test_seeding_compiled_equals_eager(cfg: Config) -> None:
    c = cw(cfg, seed_gain=0.1)
    rng = np.random.default_rng(5)
    ch = mx.array(np.clip(rng.gamma(2.0, 1.5, (3, *dims(c).field_shape)), 0, 16).astype(np.float32))
    wx = W.pack(spun_ws(c))
    f = lambda chem, wx: W.seeding_qs(chem, W.unpack(wx), c)  # noqa: E731
    a, b = f(ch, wx), mx.compile(f)(ch, wx)
    mx.eval(a, b)
    assert np.array_equal(_np(a), _np(b))


# --- shade_mult (spec §2, §9.3) ---------------------------------------------------------------------------


def _scene(nd: int, m: int = 400, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0, 320, (m, nd)).astype(np.float32)
    pos[:150, :2] = rng.normal([100.0, 200.0], 6.0, (150, 2)).astype(np.float32) % 320  # a crowd
    pos[:3, :2] = [[0.0, 0.0], [319.99, 319.99], [0.5, 319.5]]  # the seams
    alive = rng.random(m) < 0.8
    return pos, alive


def _np_shade(pos: np.ndarray, alive: np.ndarray, gain: float, n0: float, cfg: Config) -> np.ndarray:
    fg = round(cfg.world.size / cfg.world.field_cell)
    ix = np.floor(pos[alive, 0] / cfg.world.field_cell).astype(int) % fg
    iy = np.floor(pos[alive, 1] / cfg.world.field_cell).astype(int) % fg
    cnt, _, _ = np.histogram2d(iy, ix, bins=fg, range=[[0, fg], [0, fg]])
    d = sum(np.roll(cnt, (dj, di), (0, 1)) for dj in (-1, 0, 1) for di in (-1, 0, 1)) / 9.0
    return np.kron(1.0 - gain * d / (d + n0), np.ones((160 // fg, 160 // fg)))


@pytest.mark.parametrize("nd", [3, 2])
def test_shade_equals_the_numpy_histogram_and_nine_roll_box(nd: int, cfg3: Config, cfg2: Config) -> None:
    c = cw(cfg3 if nd == 3 else cfg2, shade_gain=0.05, shade_n0=6.0)
    pos, alive = _scene(nd)
    got = _np(W.shade_mult(mx.array(pos), mx.array(alive), c))
    assert got.shape == (160, 160) and got.dtype == np.float32
    np.testing.assert_allclose(got, _np_shade(pos, alive, 0.05, 6.0, c), rtol=1e-6, atol=1e-7)
    assert got.min() < 1 - 0.01  # the crowd shades: not vacuous


def test_dead_rows_do_not_shade(cfg3: Config) -> None:
    c = cw(cfg3, shade_gain=0.05)
    pos, alive = _scene(3)
    hot = pos.copy()
    hot[~alive] = [100.0, 200.0, 5.0]  # stale positions of dead rows inside the crowd
    a = _np(W.shade_mult(mx.array(pos), mx.array(alive), c))
    b = _np(W.shade_mult(mx.array(hot), mx.array(alive), c))
    assert np.array_equal(a, b)
    alive_only = _np(W.shade_mult(mx.array(pos[alive]), mx.array(np.ones(alive.sum(), bool)), c))
    assert np.array_equal(a, alive_only)


def test_the_separable_box_is_the_nine_roll_mean_and_conserves_the_sum() -> None:
    rng = np.random.default_rng(1)
    a = rng.integers(0, 30, (80, 80)).astype(np.float32)
    got = _np(W._box3(mx.array(a)))
    want = sum(np.roll(a, (dj, di), (0, 1)) for dj in (-1, 0, 1) for di in (-1, 0, 1)) / 9.0
    np.testing.assert_allclose(got, want, rtol=1e-6)
    assert got.sum() == pytest.approx(a.sum(), rel=1e-6)
    one = np.zeros((80, 80), np.float32)
    one[0, 0] = 9.0  # across the seams
    box = _np(W._box3(mx.array(one)))
    assert (box > 0).sum() == 9 and box[79, 79] == pytest.approx(1.0) and box[1, 1] == pytest.approx(1.0)


def test_shade_bounds_zero_agents_and_zero_gain(cfg3: Config) -> None:
    c = cw(cfg3, shade_gain=0.05, shade_n0=4.0)
    pos, alive = _scene(3, m=3000)
    t = _np(W.shade_mult(mx.array(pos), mx.array(alive), c))
    assert 1 - 0.05 <= t.min() < t.max() <= 1.0
    none = W.shade_mult(mx.array(pos), mx.zeros(pos.shape[0], dtype=mx.bool_), c)
    assert np.array_equal(_np(none), np.ones((160, 160), np.float32))
    off = W.shade_mult(mx.array(pos), mx.array(alive), cw(c, shade_gain=0.0))
    assert np.array_equal(_np(off), np.ones((160, 160), np.float32))


def test_shade_is_a_function_of_the_agents_alone(cfg3: Config) -> None:
    c = cw(cfg3, shade_gain=0.03)
    pos, alive = _scene(3)
    a = W.shade_mult(mx.array(pos), mx.array(alive), c)
    f = mx.compile(lambda p, al: W.shade_mult(p, al, c))
    b = f(mx.array(pos), mx.array(alive))
    mx.eval(a, b)
    assert np.array_equal(_np(a), _np(b))


def test_the_shade_and_the_weather_step_compile_in_one_graph(cfg3: Config) -> None:
    """The once-per-array rule: a graph with the shade box and the whole weather step builds (the MLX
    duplicate-roll compile bug would raise here) and equals its eager run."""
    c = cw(cfg3, shade_gain=0.05)
    p = W.params_from_config(c)
    pos, alive = _scene(3)
    ws = spun_ws(c)

    def f(pos, alive, ws, t, key):
        tm = W.shade_mult(pos, alive, c)
        fo = W.WeatherForcing(W.warm_pools(p, t) * tm, mx.ones((p.n, p.n), dtype=mx.float32))
        return W.step(ws, fo, key, p)

    args = (mx.array(pos), mx.array(alive), ws, mx.array(10, dtype=mx.int32), mx.random.key(2))
    eager = f(*args)
    comp = mx.compile(f)(*args)
    mx.eval(eager, comp)
    for a, b in zip(eager[0], comp[0], strict=True):
        assert np.array_equal(_np(a), _np(b))
    # the shade reaches the weather: the unshaded step differs
    plain = W.step(ws, W.default_forcing(p, args[3]), args[4], p)
    assert not np.array_equal(_np(plain[0].Q), _np(eager[0].Q))


# --- the eddy wind (spec §3.1, §9.4) ----------------------------------------------------------------------


def _jetted(cfg: Config) -> W.WeatherState:
    """The spun-up weather plus zonal jets that vary with y in both layers and in v (a non-trivial ``ū``)."""
    ws = spun_ws(cfg)
    n = 160
    y = (np.arange(n) + 0.5)[:, None] * 2.0
    one = np.ones((1, n))
    add = lambda a: mx.array(a.astype(np.float32))  # noqa: E731
    return ws._replace(
        u1=ws.u1 + add(0.03 * np.cos(2 * np.pi * y / 320) * one),
        u2=ws.u2 + add(0.05 * np.sin(2 * np.pi * y / 320) * one + 0.02),
        v1=ws.v1 + add(0.01 * np.sin(4 * np.pi * y / 320) * one),
        v2=ws.v2 + add(0.02 * np.cos(2 * np.pi * y / 320) * one),
    )


def _agents(m: int = 300, seed: int = 4) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0, 320, (m, 3)).astype(np.float32)
    pos[:, 2] = rng.choice([0.5, 1.0, 3.9, 4.5, 10.0, 25.0, 31.0, 34.0, 50.0, 70.0, 79.5], m)
    return pos


def _np_bilinear(t: np.ndarray, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    n = t.shape[0]
    fx, fy = np.floor(gx), np.floor(gy)
    ax, ay = gx - fx, gy - fy
    i0, j0 = fx.astype(int) % n, fy.astype(int) % n
    i1, j1 = (i0 + 1) % n, (j0 + 1) % n
    return (1 - ay) * ((1 - ax) * t[j0, i0] + ax * t[j0, i1]) + ay * ((1 - ax) * t[j1, i0] + ax * t[j1, i1])


def _np_interp(t: np.ndarray, g: np.ndarray) -> np.ndarray:
    n = t.shape[0]
    f = np.floor(g)
    a = g - f
    i0 = f.astype(int) % n
    return (1 - a) * t[i0] + a * t[(i0 + 1) % n]


def test_zbar_is_the_x_mean_of_the_staggered_arrays(cfg3: Config) -> None:
    ws = _jetted(cfg3)
    p = W.params_from_config(cfg3)
    wf = W.wind_fields(ws, p)
    assert len(W.WindFields._fields) == 9 and W.WindFields._field_defaults == {"zbar": None}
    zb = _np(wf.zbar)
    assert zb.shape == (4, 160)
    for k, a in enumerate((ws.u1, ws.v1, ws.u2, ws.v2)):
        np.testing.assert_allclose(zb[k], _np(a).astype(np.float64).mean(axis=1), rtol=1e-5, atol=1e-8)
    # not the centre-averaged winds: v1 centre-averaging shifts it half a row
    assert not np.allclose(zb[1], _np(wf.v1).mean(axis=1), atol=1e-6)


def test_the_eddy_wind_equals_a_numpy_double_in_3d(cfg3: Config) -> None:
    ws = _jetted(cfg3)
    p = W.params_from_config(cfg3)
    pos = _agents()
    aw = W.sample_wind(ws, mx.array(pos), p, eddy=True)
    assert aw.u_eddy.shape == (300, 2) and W.sample_wind(ws, mx.array(pos), p).u_eddy is None
    h1, h2 = (_np(a).astype(np.float64) for a in (ws.h1, ws.h2))
    gx, gy = pos[:, 0].astype(np.float64) / p.dx, pos[:, 1].astype(np.float64) / p.dx
    z = pos[:, 2].astype(np.float64)
    zi = (
        p.height
        * _np_bilinear(h1, gx - 0.5, gy - 0.5)
        / (_np_bilinear(h1, gx - 0.5, gy - 0.5) + _np_bilinear(h2, gx - 0.5, gy - 0.5))
    )
    mean = lambda a: _np(a).astype(np.float64).mean(axis=1)  # noqa: E731
    ub1, ub2 = _np_interp(mean(ws.u1), gy - 0.5), _np_interp(mean(ws.u2), gy - 0.5)
    vb1, vb2 = _np_interp(mean(ws.v1), gy - 1.0), _np_interp(mean(ws.v2), gy - 1.0)
    phi = np.minimum(z / p.z_sl, 1.0)
    below = z < zi
    ub = np.where(below, phi * ub1, ub2)
    vb = np.where(below, phi * vb1, vb2)
    want = _np(aw.u)[:, :2].astype(np.float64) - np.stack([ub, vb], axis=-1)
    np.testing.assert_allclose(_np(aw.u_eddy), want, rtol=1e-4, atol=2e-6)
    assert below.any() and (~below).any() and (z < 4).any()  # all three branches are exercised
    assert np.abs(_np(aw.u_eddy)).max() > 0.01 and np.abs(ub).max() > 0.01  # and the means are not trivial
    # the precomputed fields give the same answer
    wf = W.wind_fields(ws, p)
    b = W.sample_wind(ws, mx.array(pos), p, fields=wf, eddy=True)
    assert np.array_equal(_np(aw.u_eddy), _np(b.u_eddy)) and np.array_equal(_np(aw.u), _np(b.u))


def test_the_eddy_flag_does_not_change_the_wind(cfg: Config) -> None:
    ws = _jetted(cfg)
    p = W.params_from_config(cfg)
    if dims(cfg).d3:
        pos = mx.array(_agents())
        a, b = W.sample_wind(ws, pos, p), W.sample_wind(ws, pos, p, eddy=True)
    else:
        pos = mx.array(_agents()[:, :2])
        a, b = W.sample_wind_h(ws, pos, p), W.sample_wind_h(ws, pos, p, eddy=True)
    assert np.array_equal(_np(a.u), _np(b.u)) and int(a.n_clip_h) == int(b.n_clip_h)


def test_the_eddy_wind_equals_a_numpy_double_in_2d(cfg2: Config) -> None:
    ws = _jetted(cfg2)
    p = W.params_from_config(cfg2)
    pos = _agents()[:, :2]
    aw = W.sample_wind_h(ws, mx.array(pos), p, eddy=True)
    gy = pos[:, 1].astype(np.float64) / p.dx
    mean = lambda a: _np(a).astype(np.float64).mean(axis=1)  # noqa: E731
    want = _np(aw.u).astype(np.float64) - np.stack(
        [_np_interp(mean(ws.u1), gy - 0.5), _np_interp(mean(ws.v1), gy - 1.0)], axis=-1
    )
    np.testing.assert_allclose(_np(aw.u_eddy), want, rtol=1e-4, atol=2e-6)
    assert np.abs(want).max() > 0.01


def _zonal_ws(cfg: Config) -> W.WeatherState:
    """Winds that depend on the row only, ``v`` included (so the divergence, hence ``w``, is not zero)."""
    ws = W.unpack(W.rest_array(cfg))
    y = (np.arange(160) + 0.5)[:, None] * 2.0
    one = np.ones((1, 160))
    f = lambda a: mx.array(a.astype(np.float32))  # noqa: E731
    return ws._replace(
        u1=f(0.04 * np.cos(2 * np.pi * y / 320) * one),
        v1=f(0.03 * np.sin(4 * np.pi * y / 320) * one),
        u2=f(0.06 * np.sin(2 * np.pi * y / 320) * one + 0.01),
        v2=f(0.02 * np.cos(4 * np.pi * y / 320) * one),
    )


def test_a_zonally_uniform_wind_has_no_eddy_even_with_v_varying_in_y(cfg: Config) -> None:
    ws = _zonal_ws(cfg)
    p = W.params_from_config(cfg)
    if dims(cfg).d3:
        aw = W.sample_wind(ws, mx.array(_agents()), p, eddy=True)
    else:
        aw = W.sample_wind_h(ws, mx.array(_agents()[:, :2]), p, eddy=True)
    assert np.abs(_np(aw.u)[:, :2]).max() > 0.02  # there is wind to subtract
    # float32 rounding of u - zonal mean (Linux CPU measured 1.2e-7, one ulp at 1.0): far below the 0.02 wind
    assert np.abs(_np(aw.u_eddy)).max() < 1e-6


def test_a_wind_with_zero_zonal_mean_is_all_eddy(cfg: Config) -> None:
    ws = W.unpack(W.rest_array(cfg))
    x = (np.arange(160) + 0.5)[None, :] * 2.0
    one = np.ones((160, 1))
    f = lambda a: mx.array(a.astype(np.float32))  # noqa: E731
    ws = ws._replace(
        u1=f(0.05 * np.sin(2 * np.pi * x / 320) * one),
        v1=f(0.04 * np.cos(4 * np.pi * x / 320) * one),
        u2=f(-0.03 * np.sin(2 * np.pi * x / 320) * one),
        v2=f(0.02 * np.sin(6 * np.pi * x / 320) * one),
    )
    p = W.params_from_config(cfg)
    if dims(cfg).d3:
        aw = W.sample_wind(ws, mx.array(_agents()), p, eddy=True)
    else:
        aw = W.sample_wind_h(ws, mx.array(_agents()[:, :2]), p, eddy=True)
    np.testing.assert_allclose(_np(aw.u_eddy), _np(aw.u)[:, :2], atol=2e-7)
    assert np.abs(_np(aw.u_eddy)).max() > 0.02


# --- agent_wind under the controls (spec §3, §9.4) --------------------------------------------------------


def _state_at(cfg: Config, pos: np.ndarray):
    """A state whose first rows sit at ``pos`` (alive), the rest dead, with random caps."""
    d = dims(cfg)
    s = init_state(cfg)
    alive = np.arange(d.n) < len(pos)
    full = to_numpy(s.pos).copy()
    full[: len(pos)] = pos[:, : d.nd]
    return s._replace(pos=mx.array(full), alive=mx.array(alive))


def _aw(cfg: Config, ws: W.WeatherState, s) -> W.AgentForce:
    d = dims(cfg)
    return W.agent_wind(ws, s.pos, s.buoy_cap if d.d3 else None, s.alive, cfg)


def test_wind_scale_scales_the_force_and_nothing_else(cfg: Config) -> None:
    ws = _jetted(cfg)
    s = _state_at(cfg, _agents(200))
    base = _aw(cfg, ws, s)
    half = _aw(cw(cfg, wind_scale=0.5), ws, s)
    zero = _aw(cw(cfg, wind_scale=0.0, storm_break_u=0.06), ws, s)
    np.testing.assert_allclose(_np(half.force), 0.5 * _np(base.force), rtol=1e-6, atol=1e-12)
    assert (_np(zero.force) == 0).all() and np.abs(_np(base.force)).max() > 0
    for a in (half, zero):
        assert np.array_equal(_np(a.speed), _np(base.speed))  # the hazard speed is the air's, not the force's
    assert zero.breaks is not None and (_np(zero.breaks) == (_np(base.speed) > 0.06)).all()


def test_the_defaults_are_the_w1_expressions(cfg: Config) -> None:
    """With every new key at its default ``agent_wind`` is W1's: force ``a·drag·u_air``, speed ``|u_air|``."""
    ws = _jetted(cfg)
    s = _state_at(cfg, _agents(200))
    got = _aw(cfg, ws, s)
    p = W.params_from_config(cfg)
    d = dims(cfg)
    if d.d3:
        aw = W.sample_wind(ws, s.pos, p, field_cell=cfg.world.field_cell, valid=s.alive)
        a = W.alpha(s.buoy_cap, p)[:, None]
    else:
        aw, a = W.sample_wind_h(ws, s.pos, p, valid=s.alive), p.alpha0
    assert np.array_equal(_np(got.force), _np(a * cfg.physics.drag * aw.u))
    assert np.array_equal(_np(got.speed), _np(mx.sqrt((aw.u * aw.u).sum(axis=-1))))


def test_the_eddy_force_drives_the_agents_with_the_eddy_wind(cfg: Config) -> None:
    ws = _jetted(cfg)
    s = _state_at(cfg, _agents(200))
    p = W.params_from_config(cfg)
    d = dims(cfg)
    got = _aw(cw(cfg, wind_eddy_force=True), ws, s)
    if d.d3:
        aw = W.sample_wind(ws, s.pos, p, field_cell=cfg.world.field_cell, valid=s.alive, eddy=True)
        a = W.alpha(s.buoy_cap, p)[:, None]
        drive = np.concatenate([_np(aw.u_eddy), _np(aw.u)[:, 2:3]], axis=-1)
    else:
        aw, a = W.sample_wind_h(ws, s.pos, p, valid=s.alive, eddy=True), p.alpha0
        drive = _np(aw.u_eddy)
    np.testing.assert_allclose(_np(got.force), _np(a) * cfg.physics.drag * drive, rtol=1e-5, atol=1e-9)
    base = _aw(cfg, ws, s)
    assert np.abs(_np(got.force) - _np(base.force)).max() > 1e-4  # the jet is gone from the force
    # the hazard speed is independent of the force key
    assert np.array_equal(_np(got.speed), _np(base.speed))
    # the vertical force is the full wind's
    if d.d3:
        assert np.array_equal(_np(got.force)[:, 2], _np(base.force)[:, 2])
    # wind_scale composes
    both = _aw(cw(cfg, wind_eddy_force=True, wind_scale=0.5), ws, s)
    np.testing.assert_allclose(_np(both.force), 0.5 * _np(got.force), rtol=1e-6, atol=1e-12)


def test_a_zonally_uniform_wind_gives_no_horizontal_eddy_force(cfg: Config) -> None:
    ws = _zonal_ws(cfg)
    s = _state_at(cfg, _agents(200))
    full = _aw(cfg, ws, s)
    eddy = _aw(cw(cfg, wind_eddy_force=True), ws, s)
    assert np.abs(_np(full.force)[:, :2]).max() > 1e-3
    assert np.abs(_np(eddy.force)[:, :2]).max() < 1e-6  # float32 rounding, as above


def test_storm_eddy_makes_the_hazard_speed_the_eddy_speed(cfg: Config) -> None:
    ws = _jetted(cfg)
    s = _state_at(cfg, _agents(200))
    p = W.params_from_config(cfg)
    d = dims(cfg)
    for force_key in (False, True):
        c = cw(cfg, storm_eddy=True, wind_eddy_force=force_key)
        got = _aw(c, ws, s)
        if d.d3:
            aw = W.sample_wind(ws, s.pos, p, field_cell=cfg.world.field_cell, valid=s.alive, eddy=True)
            want = np.sqrt((_np(aw.u_eddy) ** 2).sum(-1) + _np(aw.u)[:, 2] ** 2)
        else:
            aw = W.sample_wind_h(ws, s.pos, p, valid=s.alive, eddy=True)
            want = np.sqrt((_np(aw.u_eddy) ** 2).sum(-1))
        np.testing.assert_allclose(_np(got.speed), want, rtol=1e-5, atol=1e-9)
    total = _aw(cfg, ws, s).speed
    assert (_np(total) > _np(got.speed) + 1e-3).any()  # and it is not the total speed
    # the bond rule reads the same speed
    c = cw(cfg, storm_eddy=True, storm_break_u=0.05)
    br = _aw(c, ws, s).breaks
    assert br is not None and (_np(br) == (_np(_aw(c, ws, s).speed) > 0.05)).all()


def test_the_hazard_and_the_exposure_follow_the_hazard_speed(cfg: Config) -> None:
    """A uniform jet (zonal mean = the wind) is all exposure under the total speed and none under the eddy."""
    wx = fixed_wx(cfg, u1=0.08, u2=0.08)
    nd = dims(cfg).nd
    pos, alive = floor_scene(
        cfg, [p[:nd] for p in [(100.0, 100.0, 10.0), (200.0, 50.0, 50.0), (50.0, 250.0, 20.0)]]
    )
    key = mx.random.key(0)
    out = {}
    for eddy in (False, True):
        c = cw(cfg, storm_p=1.0, storm_u0=0.03, storm_eddy=eddy)
        sp = W.agent_wind(
            W.unpack(wx), pos, mx.zeros((pos.shape[0],)) if dims(c).d3 else None, alive, c
        ).speed
        out[eddy] = (_np(sp)[_np(alive)], _np(lifecycle.storm_mask(sp, alive, key, c)))
    assert out[False][0].min() > 0.07 and out[True][0].max() < 1e-6
    assert out[False][1][:3].all() and not out[True][1].any()
    # the snapshot's exposure (the metrics row) is the mean ramp of the same speed
    s = init_state(cfg)._replace(wx=wx)
    exp = {
        e: metrics.snapshot_metrics(s, cw(cfg, storm_eddy=e), heavy=False)["weather"] for e in (False, True)
    }
    assert exp[False]["storm_exposure"] > 0.5 and exp[True]["storm_exposure"] < 1e-6
    assert exp[True]["air_speed_max"] < 1e-6 < exp[False]["air_speed_max"]


def test_the_new_keys_are_dataclass_fields_of_the_config_only() -> None:
    """None of the eight is a ``WeatherParams`` field: they are read from the config by name."""
    assert not set(NEW_KEYS) & {f.name for f in dataclasses.fields(W.WeatherParams)}
