"""The channel (equatorial band) weather, W-band-0 (DESIGN §37): the torus is untouched, the channel obeys its
wall, its symmetry and its conservation, and the study driver and the detector read it.

CPU and fast. The torus branch is pinned two ways: the parameter hash and ``to_dict`` (the channel fields
are in ``weather._POST_W0``), and a trajectory digest recorded from the shipped step (``c916615``) that is
platform-bound like ``test_weather_w2_wiring.PINS``. "Bitwise" means bitwise on CPU and ``allclose`` on the
GPU (``helpers.ON_GPU``).
"""

from __future__ import annotations

import hashlib
import math
import platform
from dataclasses import replace

import mlx.core as mx
import numpy as np
import pytest

from alife import weather as W
from alife.analysis import w0
from alife.weather import START_SET, WeatherParams, WeatherState

BAND = W.band_params(96.0)  # Ly = 96, 160 x 48, the recommended candidate (DESIGN §37.4)


def _np(a) -> np.ndarray:
    return np.asarray(a, dtype=np.float64)


# --- the torus is untouched ---------------------------------------------------------------------------------


def test_defaults_keep_the_w0_parameter_hash_and_record() -> None:
    """The channel fields are in ``_POST_W0``: omitted at their defaults, so the W0 hashes and parameter files
    are the old ones (``analysis/w0.params_hash(WeatherParams()) == 7832e060``)."""
    assert w0.params_hash(WeatherParams()).startswith("7832e060")
    assert w0.params_hash(START_SET).startswith("bf468155")
    d = WeatherParams().to_dict()
    for k in W._POST_W0:
        assert k not in d
    assert "seed_dmin" not in d
    on = BAND.to_dict()
    for k in ("cells_y", "channel", "wall_margin", "sponge_r"):
        assert k in on
    assert "sponge_w" not in on and "sponge_profile" not in on  # at their defaults (4.0, "exp")
    assert replace(BAND, sponge_w=3.0).to_dict()["sponge_w"] == 3.0
    assert replace(BAND, sponge_profile="quad").to_dict()["sponge_profile"] == "quad"
    assert WeatherParams.from_dict(on) == BAND  # the record round-trips
    assert WeatherParams.from_dict({"channel": True}).channel is True  # a bool stays a bool
    assert w0.params_hash(BAND) != w0.params_hash(WeatherParams())


def test_ny_is_derived_not_an_independent_default() -> None:
    """``ny`` follows ``n`` on the torus (the probe's independent default of 160 built (160, 64) arrays for
    ``n = 64``)."""
    for n, dx in ((64, 5.0), (96, 320.0 / 96), (160, 2.0)):
        p = WeatherParams(n=n, dx=dx)
        assert p.ny == n and p.Ly == p.L
        ws = W.init_state(p, W.default_forcing(p))
        assert all(a.shape == (n, n) for a in ws)
        assert W.default_forcing(p).T_s.shape == (n, n)
    assert (BAND.ny, BAND.n, BAND.Ly, BAND.L) == (48, 160, 96.0, 320.0)
    ws = W.init_state(BAND, W.default_forcing(BAND))
    assert all(a.shape == (48, 160) for a in ws)


# platform-bound: sha256 of the nine arrays after 200 compiled steps (seeds on, drifting pools, key 11) of
# ``WeatherParams(n=64, dx=5)`` and ``WeatherParams(n=96, dx=320/96)``, recorded from the shipped
# ``weather.py`` of c916615 (the module before the channel) on CPU, MLX 0.32.3, Darwin arm64
_SHIPPED = {
    (64, 5.0): "9968d8e204f4bb38c603b6b39b9d29952dd33e8e151695ef4512cd5e71e90aff",
    (96, 320.0 / 96): "7b3c5bf92dff0a0695753d02c48ec6619e3851f3977841769995bbaa18589b92",
}


def _digest(p: WeatherParams, steps: int) -> str:
    ws = W.init_state(p, W.default_forcing(p))
    step = W.make_step(p)
    key = mx.random.key(11)
    for t in range(steps):
        key, kt = mx.random.split(key)
        ws, _ = step(ws, W.default_forcing(p, mx.array(t, dtype=mx.int32)), kt)
    mx.eval(ws)
    h = hashlib.sha256()
    for a in ws:
        h.update(np.array(a).tobytes())
    return h.hexdigest()


@pytest.mark.parametrize("n,dx", list(_SHIPPED), ids=["n64", "n96"])
def test_torus_branch_is_the_shipped_step(n: int, dx: float) -> None:
    """Gate G0.1: with the channel off the step is the shipped one, bit for bit, at ``n`` other than 160."""
    if (mx.__version__, platform.system(), platform.machine(), mx.default_device().type == mx.cpu) != (
        "0.32.3",
        "Darwin",
        "arm64",
        True,
    ):
        pytest.skip("the digest is platform-bound: it is recorded on CPU, MLX 0.32.3, Darwin arm64")
    assert _digest(WeatherParams(n=n, dx=dx), 200) == _SHIPPED[(n, dx)]


def test_explicit_square_rows_change_nothing() -> None:
    """``cells_y = n`` is the torus: the same trajectory as the default, bitwise."""
    a, b = WeatherParams(n=48, dx=320.0 / 48), WeatherParams(n=48, dx=320.0 / 48, cells_y=48)
    assert a.ny == b.ny == 48
    assert _digest(a, 40) == _digest(b, 40)


# --- validation ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("ly", [80.0, 96.0, 112.0])
def test_band_layouts_are_valid(ly: float) -> None:
    p = W.band_params(ly)
    assert p.validate() == []
    assert (p.wall_margin, p.sponge_r, p.sponge_w, p.sponge_profile) == (16.0, 0.05, 4.0, "exp")
    assert p.ny == round(ly / 2) and p.Ly == ly
    phi_max = {80.0: 45.0, 96.0: 54.0, 112.0: 63.0}[ly]
    assert math.degrees(0.5 * ly / p.a_rad) == pytest.approx(phi_max, abs=0.01)
    # the pools sit well off the equator and well inside the walls
    assert all(abs(row[1] - 0.5 * ly) in (20.0, 30.0, 36.0) for row in p.pools)
    assert p.f_pool == pytest.approx(p.f0 * math.sin(abs(p.pools[0][1] - 0.5 * ly) / p.a_rad))


def test_channel_rules_reject_what_breaks_the_wall() -> None:
    def bad(**kw) -> str:
        msgs = replace(BAND, **kw).validate()
        assert msgs, kw
        return " | ".join(msgs)

    assert "C2" in bad(cells_y=90)  # 2 Ly > Lx
    assert "C3" in bad(sl_order=3, kappa_d=0.10)
    assert "C4" in bad(pools=((10.0, 120.0, 32.0, 14.0, 1.0, 0.0, 0.0),))  # y outside [0, Ly)
    assert "C4" in bad(pools=((10.0, 60.0, 32.0, 14.0, 1.0, 0.0, 0.01),))  # drifts in y
    assert "C5" in bad(wall_margin=30.0)
    assert "C6" in bad(sponge_r=0.5)
    assert "C6" in bad(sponge_w=1.0)
    assert "C6" in bad(sponge_profile="gauss")
    assert "C1" in bad(cells_y=4)
    # the channel keys are inert without a channel: no parameter set changes its hash without its behaviour
    assert "C0" in replace(WeatherParams(), wall_margin=12.0).validate()[0]
    assert "C0" in replace(WeatherParams(), sponge_r=0.05).validate()[0]
    assert "C0" in replace(WeatherParams(), cells_y=48).validate()[0]


def test_the_engine_refuses_a_channel_and_has_no_wind_for_it() -> None:
    from alife.config import load_config

    cfg = load_config(preset="64k3d", overlays=["storm"])
    assert any("channel" in m for m in W.check_world(BAND, cfg))
    assert not any("channel" in m for m in W.check_world(WeatherParams(), cfg))
    assert W.params_from_config(cfg) == WeatherParams()  # a config never builds a channel
    ws = W.init_state(BAND, W.default_forcing(BAND))
    with pytest.raises(NotImplementedError):
        W.wind_fields(ws, BAND)
    with pytest.raises(NotImplementedError):
        W.sample_wind(ws, mx.zeros((4, 3)), BAND)


# --- the latitude table and the rotation --------------------------------------------------------------------


def test_latitude_is_odd_about_the_equator_and_monotone() -> None:
    p = BAND
    ys = np.arange(0.0, p.Ly + 1e-9, p.dx / 2)
    f = np.array([p.f_of_y(y) for y in ys])
    assert p.f_of_y(0.5 * p.Ly) == 0.0
    assert np.all(np.diff(f) > 0)  # monotone: one equator, no fold
    np.testing.assert_allclose(f, -f[::-1], atol=1e-15)  # odd about Ly/2
    assert p.f_of_y(p.Ly) == pytest.approx(p.f0 * math.sin(math.radians(54.0)))  # 0.81 f0 at the walls
    assert p.pole_ys == (p.Ly, 0.0) and p.equator_ys == (0.5 * p.Ly,)
    assert W.WeatherParams().pole_ys == (80.0, 240.0) and W.WeatherParams().equator_ys == (0.0, 160.0)
    # the torus keeps its f
    t = WeatherParams()
    assert t.f_of_y(80.0) == pytest.approx(t.f0)


def test_rotation_tables_follow_the_table_and_skip_the_wall_pairs() -> None:
    p = BAND
    cs, ss, cn, sn = (np.array(x)[0, :, 0] for x in W.rotation_tables(p))
    assert ss[0] == 0.0 and cs[0] == 1.0 and sn[-1] == 0.0 and cn[-1] == 1.0  # the wall pairs are identity
    ny = p.ny
    # f is odd about the equator: the north angle of row ny-1-j is minus the south angle of row j
    np.testing.assert_allclose(sn[::-1][1:], -ss[1:], atol=1e-9)
    j = np.arange(ny)
    want = 0.25 * p.dt * np.array([p.f_of_y((x + 0.25) * p.dx) for x in j])
    np.testing.assert_allclose(np.arcsin(ss)[1:], want[1:], rtol=1e-6)
    assert np.all(np.diff(np.arcsin(ss)[1:]) > 0)  # monotone: no sign change except at the equator


def test_rotation_never_increases_kinetic_energy_and_leaves_the_wall_face() -> None:
    p = BAND
    r = np.random.default_rng(3)
    u = mx.array(r.uniform(-1, 1, (2, p.ny, p.n)), dtype=mx.float32)
    v = mx.array(r.uniform(-1, 1, (2, p.ny, p.n)), dtype=mx.float32)
    tab = W.rotation_tables(p)
    un, vn = W.rotate_half(u, v, tab)
    mx.eval(un, vn)
    assert np.array_equal(np.array(vn[:, -1]), np.array(v[:, -1]))  # the wall face is not rotated
    ke0 = float((u * u).sum() + (v * v).sum())
    ke1 = float((un * un).sum() + (vn * vn).sum())
    assert ke1 <= ke0 * (1 + 1e-6)  # exact disjoint pair rotations averaged: the energy cannot grow
    assert ke1 > 0.5 * ke0


# --- the wall -----------------------------------------------------------------------------------------------


def _run(p: WeatherParams, steps: int, seed: int = 0) -> tuple[WeatherState, WeatherState]:
    ws0 = W.init_state(p, W.default_forcing(p))
    ws = ws0
    step = W.make_step(p)
    key = mx.random.key(seed)
    for t in range(steps):
        key, kt = mx.random.split(key)
        ws, _ = step(ws, W.default_forcing(p, mx.array(t, dtype=mx.int32)), kt)
    mx.eval(ws)
    return ws0, ws


def test_the_wall_face_is_exactly_zero_after_every_stage() -> None:
    p = replace(BAND, seed_rate=1.0)  # a disturbance every step, the harshest case for the wall
    # each stage on its own, fed a v that is not zero on the wall row
    r = np.random.default_rng(1)
    u = mx.array(r.uniform(-0.05, 0.05, (2, p.ny, p.n)), dtype=mx.float32)
    v = mx.array(r.uniform(-0.05, 0.05, (2, p.ny, p.n)), dtype=mx.float32)
    h = mx.stack([mx.full((p.ny, p.n), p.H1), mx.full((p.ny, p.n), p.H2)]) + mx.array(
        r.uniform(-0.2, 0.2, (2, p.ny, p.n)), dtype=mx.float32
    )
    va = W._advect(u, v, p)[1]
    vk = W._pressure_kick(u, v, h, p)[1]
    vd = W._damp(u, v, h, p)[1]
    mx.eval(va, vk, vd)
    for name, x in (("advect", va), ("pressure kick", vk), ("damping", vd)):
        assert np.array_equal(np.array(x[:, -1]), np.zeros((2, p.n))), name
    psi = W._seed_fields(mx.random.key(5), p)[0]
    _, dv = W.psi_winds(psi, p.dx)
    mx.eval(psi, dv)
    assert np.array(psi[-1]).max() == 0.0 and np.array(dv[-1]).max() == 0.0
    # and the whole step, every step
    ws = W.init_state(p, W.default_forcing(p))
    step = W.make_step(p)
    key = mx.random.key(0)
    for t in range(150):
        key, kt = mx.random.split(key)
        ws, _ = step(ws, W.default_forcing(p, mx.array(t, dtype=mx.int32)), kt)
        assert float(mx.abs(ws.v1[-1]).max()) == 0.0 and float(mx.abs(ws.v2[-1]).max()) == 0.0, t
    assert float(mx.abs(W.vorticity(ws, p)[-1]).max()) == 0.0  # the corner row is both walls: free slip


def test_mass_is_conserved_across_walls() -> None:
    """A zero face gives zero flux in ``h``: the column mass drifts only by float rounding and the (counted)
    positivity clamps."""
    p = BAND
    ws0, ws = _run(p, 400)
    m0 = float((_np(ws0.h1) + _np(ws0.h2)).sum())
    m1 = float((_np(ws.h1) + _np(ws.h2)).sum())
    assert abs(m1 - m0) / m0 < 1e-6


def _mirror(ws: WeatherState, ny: int) -> WeatherState:
    """y -> Ly - y, v -> -v, f odd: scalars and u flip rows; v faces map k <-> ny-2-k; wall row stays 0."""
    out = []
    for name, a in zip(ws._fields, ws, strict=True):
        a = np.array(a)
        if name in ("v1", "v2"):
            m = np.zeros_like(a)
            m[: ny - 1] = -a[: ny - 1][::-1]
            out.append(mx.array(m))
        else:
            out.append(mx.array(a[::-1].copy()))
    return WeatherState(*out)


@pytest.mark.parametrize("sponge", [0.0, 0.05], ids=["no_sponge", "sponge"])
def test_mirror_symmetry(sponge: float) -> None:
    """The channel step is symmetric under ``y -> Ly - y, v -> -v`` (``f`` odd): the semi-Lagrangian gather
    sees a zero south face (a stored copy of ``v[0]`` there would break it by 1e-1). Tolerance 1e-4
    relative to the field's range, 50 and 400 steps, no seeds."""
    p = replace(BAND, seed_rate=0.0, sponge_r=sponge)
    pm = replace(p, pools=tuple((x, p.Ly - y, sx, sy, a, vx, -vy) for (x, y, sx, sy, a, vx, vy) in p.pools))
    r = np.random.default_rng(1)

    def smooth(a: np.ndarray) -> np.ndarray:
        for _ in range(3):
            a = 0.25 * (np.roll(a, 1, 0) + np.roll(a, -1, 0)) + 0.25 * (np.roll(a, 1, 1) + np.roll(a, -1, 1))
        return a

    ws = W.init_state(p, W.default_forcing(p))
    v1 = 0.1 * smooth(r.standard_normal((p.ny, p.n)))
    v1[-1] = 0.0
    ws = ws._replace(
        h1=mx.array(np.array(ws.h1) + 1.5 * smooth(r.standard_normal((p.ny, p.n))), dtype=mx.float32),
        u1=mx.array(0.1 * smooth(r.standard_normal((p.ny, p.n))), dtype=mx.float32),
        v1=mx.array(v1, dtype=mx.float32),
    )
    a, b = ws, _mirror(ws, p.ny)
    fa = W.make_step(p)
    fb = W.make_step(pm)
    key = mx.random.key(0)
    t = 0
    for stop in (50, 400):
        while t < stop:
            a, _ = fa(a, W.default_forcing(p, mx.array(t, dtype=mx.int32)), key)
            b, _ = fb(b, W.default_forcing(pm, mx.array(t, dtype=mx.int32)), key)
            t += 1
        mx.eval(a, b)
        ma = _mirror(a, p.ny)
        for name in ("h1", "u1", "v1", "Q"):
            x, y = _np(getattr(ma, name)), _np(getattr(b, name))
            scale = np.abs(x - x.mean()).max() if name in ("h1", "Q") else np.abs(x).max()
            assert np.abs(x - y).max() / scale < 1e-4, (name, stop, np.abs(x - y).max() / scale)


def test_the_gather_never_reads_across_a_wall() -> None:
    """A uniform u advected by a backtrace that leaves the grid stays itself (free slip); a uniform v takes
    the zero of the wall face; the zero offset is the identity."""
    ny, n = 12, 16
    a = np.ones((4, ny, n), dtype=np.float32)
    a[2:, -1] = 0.0  # the stored north wall face of the v-like fields
    ones = mx.array(a)
    zero = mx.zeros((4, ny, n))
    out = W.semi_lagrangian(ones, zero, zero, 1, 2)
    assert np.array_equal(np.array(out), np.array(ones))
    for sign in (+1.0, -1.0):
        oy = mx.full((4, ny, n), sign * 5.0)
        out = np.array(W.semi_lagrangian(ones, zero, oy, 1, 2))
        np.testing.assert_allclose(out[:2], 1.0)  # u-like: clamped to its edge row
        assert out[2:].min() >= -1e-6 and out[2:].max() <= 1.0 + 1e-6
        row = out[2:, 0] if sign < 0 else out[2:, -1]
        np.testing.assert_allclose(row, 0.0, atol=1e-6)  # v-like: the wall face's zero


# --- seeds and the sponge -----------------------------------------------------------------------------------


def test_seeds_keep_their_margin_and_take_the_sign_of_f() -> None:
    p = BAND
    ys = []
    for s in range(120):
        y = W._seed_fields(mx.random.key(s), p)[4]
        ys.append(float(y))
    ys = np.array(ys)
    assert ys.min() >= p.wall_margin - 1e-4 and ys.max() <= p.Ly - p.wall_margin + 1e-4
    assert ys.min() < p.wall_margin + 3 and ys.max() > p.Ly - p.wall_margin - 3  # the range is used
    # the cyclonic sign follows the hemisphere: a disturbance at y >= Ly/2 has the opposite psi of one below
    sign: dict[bool, set[float]] = {True: set(), False: set()}
    for s in range(200):
        psi, _, _, _, y = W._seed_fields(mx.random.key(s), replace(p, seed_rate=1.0))
        arr = np.array(psi)
        peak = np.unravel_index(np.abs(arr).argmax(), arr.shape)
        sign[float(y) >= 0.5 * p.Ly].add(float(np.sign(arr[peak])))
    assert sign[True] == {-1.0} and sign[False] == {1.0}


def test_sponge_profiles() -> None:
    p = BAND
    ru, rv = W.sponge_rates(p)
    assert len(ru) == len(rv) == p.ny
    np.testing.assert_allclose(ru, ru[::-1], rtol=1e-12)  # symmetric about the equator
    assert ru[0] == pytest.approx(p.sponge_r * math.exp(-1.0 / p.sponge_w))  # u row 0 is 1 unit from the wall
    assert rv[0] == pytest.approx(p.sponge_r * math.exp(-2.0 / p.sponge_w))
    assert rv[-1] == pytest.approx(p.sponge_r)  # the wall face itself (masked anyway)
    pool_row = round(18.0 / p.dx)  # the pools' friction is the point of the window (DESIGN §37.5)
    assert ru[pool_row] < 1e-3 and ru[pool_row] == pytest.approx(0.05 * math.exp(-(pool_row + 0.5) * 2 / 4))
    q = W.sponge_rates(replace(p, sponge_profile="quad", sponge_w=12.0))[0]
    assert q[0] == pytest.approx(0.05 * (1 - 1.0 / 12.0) ** 2)
    assert q[6] == 0.0 and q[p.ny // 2] == 0.0  # zero beyond w
    assert W.sponge_rates(replace(p, sponge_r=0.0))[0] == [0.0] * p.ny
