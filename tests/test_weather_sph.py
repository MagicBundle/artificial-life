"""The sphere weather step (``alife.weather_sph``, DESIGN §46.6) against the float64 reference of
``tests/sphere_weather_ref.py``, its invariants (mass and moisture closure, the calm rest state, tangency,
the 90 degree symmetry of the grid), the pools and the seeded vortex against their closed forms, and the
stability measurements of S4b (float32 noise tests; the dense one-step map)."""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

import sphere_weather_stab as S
from alife import geom, weather_sph, weather_tab
from alife.weather import ENGINE_FIXED, WeatherForcing, WeatherParams
from sphere_weather_ref import Ref, RefModel

R = 36.0


def params(nw: int, radius: float = R, **kw: float) -> WeatherParams:
    return WeatherParams(n=nw, dx=weather_sph.cell_side(radius, nw), height=80.0, **{**ENGINE_FIXED, **kw})


def active(p: WeatherParams, seed: int = 1, amp: float = 1.0) -> Ref:
    """A dynamically active smooth state: a solid-rotation wind of ~0.05 in each layer, smooth thickness
    anomalies, humidity from the rest state."""
    t = weather_tab.weather_tab(p.n)
    x = t.x
    om = np.array([0.3, -0.5, 0.8])
    u = np.stack([np.cross(om, x) * 0.05, np.cross(-om, x) * 0.03]) * amp
    h = np.array([[p.H1], [p.H2]]) + amp * np.stack([1.2 * x[:, 0] * x[:, 2], -0.9 * x[:, 1]])
    q = np.full(t.npix, 0.9) + 0.1 * x[:, 0]
    zeros = np.zeros(t.npix)
    return Ref(h, u, q, zeros, zeros)


def to_mlx(s: Ref) -> weather_sph.WeatherStateS:
    f = lambda a: mx.array(np.asarray(a), dtype=mx.float32)  # noqa: E731
    return weather_sph.WeatherStateS(f(s.h[0]), f(s.h[1]), f(s.u[0]), f(s.u[1]), f(s.q), f(s.p), f(s.dc))


def forcing(p: WeatherParams, ts: np.ndarray, qs: np.ndarray | None = None) -> WeatherForcing:
    n = ts.shape[0]
    return WeatherForcing(
        mx.array(ts, dtype=mx.float32), mx.ones((n,)) if qs is None else mx.array(qs, dtype=mx.float32)
    )


def compare(a: weather_sph.WeatherStateS, b: Ref) -> dict[str, float]:
    mx.eval(*a)
    return {
        "h": max(np.abs(np.array(a.h1) - b.h[0]).max(), np.abs(np.array(a.h2) - b.h[1]).max()),
        "u": max(np.abs(np.array(a.u1) - b.u[0]).max(), np.abs(np.array(a.u2) - b.u[1]).max()),
        "q": np.abs(np.array(a.Q) - b.q).max(),
    }


# --- MLX equals the float64 reference to float32 rounding ---------------------------------------------------


@pytest.mark.parametrize("nw,radius", [(4, 36.0), (8, 36.0), (12, 48.0)])
def test_the_moist_step_equals_the_float64_reference(nw: int, radius: float) -> None:
    p = params(nw, radius)
    model = RefModel(p, radius)
    ts = np.clip(np.exp(-((weather_tab.weather_tab(nw).x[:, 0] - 0.5) ** 2) / 0.1), 0, 1)
    ref, mlx_s = active(p), to_mlx(active(p))
    fo = forcing(p, ts)
    key = mx.random.key(0)
    for i in range(1, 41):
        ref, _ = model.step(ref, ts, np.ones(len(ts)))
        mlx_s, _ = weather_sph.step(mlx_s, fo, key, p, radius, seeds=False)
        if i in (1, 10, 40):
            err = compare(mlx_s, ref)
            # float32 rounding of values of 32 and 48 is 4e-6 and 8e-6 per operation: a few ulps accumulate
            assert err["h"] < 3e-5 * i**0.5 and err["u"] < 3e-6 * i**0.5 and err["q"] < 3e-6 * i**0.5, (
                i,
                err,
            )


def test_the_dry_step_equals_the_float64_reference_and_so_does_the_unadvected_one() -> None:
    p = params(8)
    model = RefModel(p, R)
    ts = np.zeros(model.n)
    fo = forcing(p, ts)
    key = mx.random.key(0)
    for advect in (True, False):
        ref, mlx_s = active(p, amp=0.5), to_mlx(active(p, amp=0.5))
        for _ in range(15):
            ref, _ = model.step(ref, ts, np.ones(model.n), advect=advect, dry=True)
            mlx_s, _ = weather_sph.step(mlx_s, fo, key, p, R, advect=advect, dry=True, seeds=False)
        err = compare(mlx_s, ref)
        assert err["h"] < 1e-4 and err["u"] < 1e-5, (advect, err)


def test_the_reference_conserves_what_the_port_conserves_with_the_filter_off() -> None:
    """a4 = 0 on both: the unfiltered step still equals the reference (the filter is an addition, not a
    crutch)."""
    p = params(8)
    model = RefModel(p, R, a4=0.0)
    ts = np.zeros(model.n)
    ref, mlx_s = active(p, amp=0.3), to_mlx(active(p, amp=0.3))
    fo, key = forcing(p, ts), mx.random.key(0)
    for _ in range(10):
        ref, _ = model.step(ref, ts, np.ones(model.n), dry=True)
        mlx_s, _ = weather_sph.step(mlx_s, fo, key, p, R, dry=True, seeds=False, a4=0.0)
    err = compare(mlx_s, ref)
    assert err["h"] < 1e-4 and err["u"] < 1e-5, err


# --- invariants ---------------------------------------------------------------------------------------------


def test_the_calm_rest_state_is_a_fixed_point_without_pools_or_seeds() -> None:
    p = params(8)
    n = 12 * 64
    fo = forcing(p, np.zeros(n))
    s0 = weather_sph.init_state(p, fo)
    s = s0
    for _ in range(30):
        s, d = weather_sph.step(s, fo, mx.random.key(0), p, R, seeds=False)
    mx.eval(*s)
    for a, b in zip(s, s0, strict=True):
        assert np.array_equal(np.array(a), np.array(b))
    assert int(d.n_limited) == int(d.n_clamp) == 0


def thin_front() -> tuple[WeatherParams, Ref, np.ndarray]:
    """A scene that drives the limiter and the clamp: layer 1 is 2.2 thick (``h_min`` is 2) on one side of a
    smooth front and 32 on the other, with winds of 0.25 and 0.1, over a warm pool, and a short condensation
    time."""
    p = params(8, q_pool=1.3, k_E=0.5, tau_c=2.0)
    x = weather_tab.weather_tab(8).x
    h1 = 2.2 + 29.8 / (1.0 + np.exp(-8.0 * x[:, 0]))
    u = np.stack([0.25 * np.cross([0, 0, 1.0], x), 0.1 * np.cross([0, 1.0, 0], x)])
    zeros = np.zeros(len(x))
    ts = np.clip(2.0 * np.exp(-((x[:, 0] - 0.7) ** 2 + x[:, 1] ** 2) / 0.1), 0, 1)
    return p, Ref(np.stack([h1, 80.0 - h1]), u, np.full(len(x), 0.9), zeros, zeros), ts


def test_mass_and_moisture_close_with_the_clamp_terms() -> None:
    """Σ(h1+h2) changes only by ``clamp_mass`` (the positivity clamp adds mass) and ΣQ by evaporation minus
    condensation plus ``q_clamp``; every other term is a flux or an exchange (conservative to rounding). The
    limiter does not guarantee ``h >= h_min`` with the filter and the corrected faces (S4a G6): the clamp
    acts."""
    p, ref, ts = thin_front()
    fo = forcing(p, ts)
    s = to_mlx(ref)
    mass0, q0 = _tot(s)
    cm = qc = ev = co = 0.0
    lim = clamps = 0
    for i in range(120):
        s, d = weather_sph.step(s, fo, mx.random.key(i), p, R)
        mx.eval(*s, d.clamp_mass)
        cm += float(d.clamp_mass)
        qc += float(d.q_clamp)
        ev += float(d.sum_E)
        co += float(d.sum_C)
        lim += int(d.n_limited)
        clamps += int(d.n_clamp)
    mass1, q1 = _tot(s)
    assert lim > 0 and clamps > 0 and cm > 0, (
        "the scene never reached the limiter and the clamp: it proves nothing"
    )
    assert abs((mass1 - mass0) - cm) < 2e-6 * mass0, ((mass1 - mass0), cm)
    # condensation moves vapour to layer 2: Q changes by evap - cond (+ the seeds' Q: none) + q_clamp
    assert abs((q1 - q0) - (ev - co + qc)) < 3e-5 * max(q0, 1.0), ((q1 - q0), ev - co + qc)


def test_the_limited_scene_still_equals_the_reference() -> None:
    """The limiter and the clamp branch the same way in float32 and in float64 (their counts agree)."""
    p, ref, ts = thin_front()
    model = RefModel(p, R)
    fo = forcing(p, ts)
    s = to_mlx(ref)
    for _ in range(25):
        ref, _ = model.step(ref, ts, np.ones(len(ts)))
        s, _ = weather_sph.step(s, fo, mx.random.key(0), p, R, seeds=False)
    err = compare(s, ref)
    assert err["h"] < 5e-3 and err["u"] < 5e-4, err  # a thin layer amplifies rounding; still the same flow


def test_the_limiter_alone_keeps_the_layers_positive() -> None:
    """With the filter off (a4 = 0) the limiter is the only thing between an outflow beyond availability and
    ``h < h_min``: after a step of the thin-front scene ``h >= h_min`` to rounding, nothing is clamped, and
    the
    count of limited cells equals the float64 reference's (review M2: swapping the donor's and the receiver's
    factors lost positivity, with a clamp mass of 20, and no closure test noticed)."""
    p, ref, ts = thin_front()
    model = RefModel(p, R, a4=0.0)
    fo = forcing(p, ts)
    s1 = to_mlx(ref)
    n_ref = n_mlx = 0
    hmin = 1e9
    for i in range(120):
        ref, aux = model.step(ref, ts, np.ones(len(ts)))
        s1, d = weather_sph.step(s1, fo, mx.random.key(0), p, R, seeds=False, a4=0.0)
        mx.eval(*s1, d.clamp_mass)
        n_ref, n_mlx = n_ref + aux["n_limited"], n_mlx + int(d.n_limited)
        hmin = min(hmin, float(mx.minimum(s1.h1.min(), s1.h2.min())))
        assert float(d.clamp_mass) == 0.0 and int(d.n_clamp) == 0 and aux["clamp_mass"] == 0.0, i
    assert n_ref > 3000 and abs(n_mlx - n_ref) <= 0.01 * n_ref  # the limiter acts, and acts alike
    assert hmin >= p.h_min - 1e-5
    assert np.abs(np.array(s1.h1) - ref.h[0]).max() < 5e-4 and np.abs(np.array(s1.u1) - ref.u[0]).max() < 5e-5


def test_the_moisture_clamp_closes_the_water_identity() -> None:
    """A scene with ``q_clamp > 0``: the upwind flux never drives ``Q`` negative by itself (the limiter bounds
    the outflow by the cell's own water), so some cells start with a slightly negative ``Q`` (a corrupted
    state); the clamp adds it back and ``Δ ΣQ = Σ(sum_E - sum_C + q_clamp)`` with the term, not without."""
    p, ref, ts = thin_front()
    q = ref.q.copy()
    q[::17] = -0.02
    s = to_mlx(ref._replace(q=q))
    fo = forcing(p, ts)
    _, q0 = _tot(s)
    s2, d = weather_sph.step(s, fo, mx.random.key(0), p, R, seeds=False)
    mx.eval(*s2, d.q_clamp)
    _, q1 = _tot(s2)
    assert float(d.q_clamp) > 0.0 and int(d.n_clamp) > 0
    want = float(d.sum_E) - float(d.sum_C) + float(d.q_clamp)
    assert abs((q1 - q0) - want) < 3e-5 * q0
    assert abs((q1 - q0) - (want - float(d.q_clamp))) > 0.5 * float(
        d.q_clamp
    )  # without the term it would not close


def _tot(s: weather_sph.WeatherStateS) -> tuple[float, float]:
    h = np.array(s.h1, dtype=np.float64) + np.array(s.h2, dtype=np.float64)
    return float(h.sum()), float(np.array(s.Q, dtype=np.float64).sum())


def test_the_winds_stay_tangent_and_under_the_tripwire() -> None:
    p = params(8)
    ts = np.clip(weather_tab.weather_tab(8).x[:, 0] + 0.2, 0, 1)
    fo = forcing(p, ts)
    s = to_mlx(active(p, amp=4.0))
    x = np.array(weather_tab.weather_tab(8).x)
    for i in range(60):
        s, d = weather_sph.step(s, fo, mx.random.key(i), p, R)
    mx.eval(*s)
    for u in (np.array(s.u1), np.array(s.u2)):
        assert np.abs((u * x).sum(1)).max() < 3e-7
        assert np.linalg.norm(u, axis=1).max() <= p.u_safe + p.seed_v * (1 + p.seed_jitter) + 1e-6
    assert int(d.n_safety) >= 0 and np.isfinite(float(d.hyper_max))


def _rot_z90(v: np.ndarray) -> np.ndarray:
    """Rotation by +90 degrees about z: (x, y, z) -> (-y, x, z)."""
    return np.stack([-v[..., 1], v[..., 0], v[..., 2]], axis=-1)


def test_the_step_commutes_with_the_quarter_turn_about_the_axis() -> None:
    """The HEALPix grid maps onto itself under a rotation of 90 degrees about z, and f = f0·z is invariant:
    the
    step on the rotated state is the rotated step (T_s = 0 and no seeds). It tests the tables and the kernels
    against each other and not against the reference."""
    nw = 8
    p = params(nw)
    t = weather_tab.weather_tab(nw)
    perm = geom.ang2pix(_rot_z90(t.x), nw)  # pixel p goes to perm[p]
    assert np.allclose(t.x[perm], _rot_z90(t.x), atol=1e-12) and len(set(perm)) == t.npix
    s = active(p, amp=1.5)
    fo = forcing(p, np.zeros(t.npix))

    def rotated(r: Ref) -> Ref:
        ph = np.empty_like(r.h)
        pu, pq = np.empty_like(r.u), np.empty_like(r.q)
        ph[:, perm] = r.h
        pu[:, perm] = _rot_z90(r.u)
        pq[perm] = r.q
        return Ref(ph, pu, pq, r.p, r.dc)

    a = to_mlx(s)
    b = to_mlx(rotated(s))
    for _ in range(8):
        a, _ = weather_sph.step(a, fo, mx.random.key(0), p, R, seeds=False)
        b, _ = weather_sph.step(b, fo, mx.random.key(0), p, R, seeds=False)
    mx.eval(*a, *b)
    ra = rotated(
        Ref(
            np.stack([np.array(a.h1), np.array(a.h2)]),
            np.stack([np.array(a.u1), np.array(a.u2)]),
            np.array(a.Q),
            s.p,
            s.dc,
        )
    )
    assert np.abs(ra.h - np.stack([np.array(b.h1), np.array(b.h2)])).max() < 3e-4
    assert np.abs(ra.u - np.stack([np.array(b.u1), np.array(b.u2)])).max() < 3e-5


# --- the pools and the seeded vortex ------------------------------------------------------------------------


def _pool_ref(x: np.ndarray, radius: float, pools, t: int) -> np.ndarray:
    out = np.zeros(len(x))
    for px, py, sx, sy, amp, vx, _vy in pools:
        lat = py / radius
        lon = px / radius + vx * t / (radius * math.cos(lat))
        c = np.array([math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)])
        east = np.array([-math.sin(lon), math.cos(lon), 0.0])
        north = np.array([-math.sin(lat) * math.cos(lon), -math.sin(lat) * math.sin(lon), math.cos(lat)])
        ch = radius * (x - c)
        e = -((ch @ north) ** 2) / (2 * sy * sy)
        if sx > 0:
            e = e - (ch @ east) ** 2 / (2 * sx * sx)
        out += amp * np.exp(e) * (x @ c > 0)
    return np.clip(out, 0, 1)


def test_the_warm_pools_equal_their_closed_form_and_drift_in_longitude() -> None:
    radius = 90.0
    pools = (
        (94.2, 62.8, 32.0, 16.0, 1.0, -0.009, 0.0),
        (0.0, -62.8, 32.0, 16.0, 0.8, 0.0, 0.0),
        (200.0, 10.0, 0.0, 12.0, 0.5, 0.0, 0.0),
    )
    p = params(16, radius, pools=pools)
    x = weather_tab.weather_tab(16).x
    for t in (0, 777, 123456):
        got = np.array(
            weather_sph.warm_pools(p, radius, mx.array(x, dtype=mx.float32), mx.array(t, dtype=mx.int32))
        )
        assert np.abs(got - _pool_ref(x, radius, pools, t)).max() < 2e-5, t
    t0 = np.array(
        weather_sph.warm_pools(p, radius, mx.array(x, dtype=mx.float32), mx.array(0, dtype=mx.int32))
    )
    assert t0.max() <= 1.0 and t0.min() >= 0.0
    # the far side of a pool is dark: the Gaussian in the chord components alone would light the antipode
    one = params(16, radius, pools=((94.2, 62.8, 32.0, 16.0, 1.0, 0.0, 0.0),))
    ts = np.array(weather_sph.warm_pools(one, radius, mx.array(x, dtype=mx.float32)))
    lat, lon = 62.8 / radius, 94.2 / radius
    c = np.array([math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)])
    assert ts[(x @ c) < 0].max() == 0.0 and ts[np.argmax(x @ c)] > 0.95
    with pytest.raises(ValueError, match="vy"):
        weather_sph.warm_pools(
            params(16, radius, pools=((0.0, 0.0, 8.0, 8.0, 1.0, 0.0, 0.1),)),
            radius,
            mx.array(x, dtype=mx.float32),
        )


def test_the_seeded_vortex_is_the_closed_form_and_cyclonic_for_its_hemisphere() -> None:
    radius = 90.0
    p = params(16, radius, seed_rate=1.0)
    x = weather_tab.weather_tab(16).x
    xm = mx.array(x, dtype=mx.float32)
    for k in range(6):
        key = mx.random.key(k)
        u, on = weather_sph._seed_vortex(xm, key, p, radius)
        mx.eval(u, on)
        assert float(on) == 1.0
        _, k_x, k_y, k_a = mx.random.split(key, 4)
        xs = np.array(
            weather_sph.fields_sph.sphere_point(mx.random.uniform(key=k_x), mx.random.uniform(key=k_y)),
            dtype=np.float64,
        )
        a = 1.0 + p.seed_jitter * (2.0 * float(mx.random.uniform(key=k_a)) - 1.0)
        sgn = 1.0 if xs[2] >= 0 else -1.0
        chord = radius * (x - xs)
        amp = -sgn * a * p.seed_v * p.seed_sigma * math.sqrt(math.e)
        psi = amp * np.exp(-(chord**2).sum(1) / (2 * p.seed_sigma**2))
        grad = -(psi / p.seed_sigma**2)[:, None] * (chord - (chord * x).sum(1)[:, None] * x)
        assert np.abs(np.array(u) - np.cross(x, grad)).max() < 1e-6
        assert (
            np.linalg.norm(np.array(u), axis=1).max() < 1.0 * a * p.seed_v * 1.02
        )  # peak |u| = a·seed_v at r = sigma
    off, o = weather_sph._seed_vortex(xm, mx.random.key(0), params(16, radius, seed_rate=0.0), radius)
    assert float(o) == 0.0 and not np.array(off).any()


# --- stability in float32 (S4b first task; the table is in docs and the build notes) ------------------------


def _dt_for(nw: int, radius: float, c: float) -> float:
    return c * 0.81 * weather_sph.cell_side(radius, nw) / WeatherParams().c_ext


def noise_ratio(nw: int, radius: float, c: float, nit: int, f0: float = 0.0, seed: int = 0) -> float:
    """Growth of the largest |h - h0| of linear noise (flat rest state, no advection, dry, no friction) over
    ``nit`` steps at the Courant number ``c``: the S4a measurement on the MLX step (``inf``: grew by 10³)."""
    p = params(nw, radius, dt=_dt_for(nw, radius, c), f0=f0)
    n = 12 * nw * nw
    base = np.array([[p.H1], [p.H2]], np.float32)
    h = base + 0.01 * np.random.default_rng(seed).normal(size=(2, n)).astype(np.float32)
    z = mx.zeros((n, 3))
    s = weather_sph.WeatherStateS(
        mx.array(h[0]), mx.array(h[1]), z, z, mx.zeros((n,)), mx.zeros((n,)), mx.zeros((n,))
    )
    fo = forcing(p, np.zeros(n))
    a0 = float(np.abs(h - base).max())
    b = mx.array(base)
    key = mx.random.key(0)
    for i in range(1, nit + 1):
        s, _ = weather_sph.step(s, fo, key, p, radius, dry=True, advect=False, seeds=False)
        if i % 25 == 0 or i == nit:
            e = float(mx.abs(mx.stack([s.h1, s.h2]) - b).max())
            if not math.isfinite(e) or e > 1e3 * a0:
                return math.inf
    return e / a0


def c_used(nw: int, radius: float) -> float:
    return WeatherParams().c_ext / (0.81 * weather_sph.cell_side(radius, nw))


def test_the_courant_numbers_of_the_presets() -> None:
    """The table of docs/SPHERE.md 7.3 with the corrected ``c_ext`` of the two-layer model (0.900)."""
    got = [round(c_used(nw, r), 3) for nw, r in ((16, 36.0), (22, 48.0), (42, 90.0))]
    assert got == [0.483, 0.498, 0.507]
    assert max(got) <= 0.8 * weather_sph.C_LIMIT


def test_linear_noise_decays_at_the_courant_number_of_use_and_at_1_2_times_it() -> None:
    for c in (1.0, 1.2):
        r = noise_ratio(16, 36.0, c * c_used(16, 36.0), 400)
        assert r < 0.5, (c, r)  # the largest value of the noise decays by the filter within hundreds of steps


def test_linear_noise_grows_past_the_limit() -> None:
    """A control: the test can fail. At C = 0.80, well above the float32 blow-up limit 0.68 to 0.72, the noise
    grows by 10³."""
    assert noise_ratio(16, 36.0, 0.80, 700) == math.inf


@pytest.mark.slow
@pytest.mark.parametrize("nw,radius", [(16, 36.0), (22, 48.0), (42, 90.0)])
def test_the_float32_courant_limit_is_at_least_1_25_times_the_courant_number_of_use(
    nw: int, radius: float
) -> None:
    """The stability rule C_used <= 0.8·C_limit with ``C_limit`` re-measured in float32 on the MLX step
    (the S4a
    decay criterion: the largest |h - h0| after 1500 steps of linear noise is below its start). Measured at
    S4b: 0.6547 / 0.650 / 0.650 for Nw 16 / 22 / 42 (bisection to 0.005 from [0.50, 0.80]); the stop rule
    ``C_limit < 1.25·C_used`` would have fired at 0.633 for Nw 42."""
    cu = c_used(nw, radius)
    assert 1.25 * cu * 0.99 <= weather_sph.C_LIMIT  # the constant used by `validate`
    stable = noise_ratio(nw, radius, weather_sph.C_LIMIT, 1500)
    assert stable < 1.0, stable  # the decay criterion at the constant itself
    for c in (1.0, 1.2):
        assert noise_ratio(nw, radius, c * cu, 1500, seed=1) < 1.0
    assert noise_ratio(nw, radius, 0.72, 1500) > 1.0  # beyond the limit (a control)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("nw", "radius", "bound"),
    [(16, 36.0, 2e-5), (22, 48.0, 2e-5), (42, 90.0, 1.5e-4)],
)
def test_the_dry_rotating_noise_grows_at_most_at_the_known_rate(nw: int, radius: float, bound: float) -> None:
    """f0 = 0.025, **40 000** steps (S4a and the 4 000-step version of this test stop before the onset),
    dry and
    frictionless, at C_used: the slope of ``log rms(h - h0)`` over the last 5 blocks of 4 000 steps. Nw 16
    and 22
    decay or are neutral (measured -0.8 and -0.7e-5 per step); **Nw 42 grows at +8.6e-5 per step** (the
    scheme's
    own rotation-coupled mode, review M1, float32 CPU/Metal and float64 agree): the bound 1.5e-4 only stops it
    from getting worse. In the production step friction cancels it (next test)."""
    c = S.WeatherParams().c_ext / (0.81 * weather_sph.cell_side(radius, nw))
    blocks = S.rms_blocks(nw, radius, c, 40000)
    tail = float(np.mean(blocks[-5:]))
    assert tail < bound, (nw, tail, blocks)


@pytest.mark.slow
def test_the_production_weather_at_nw_42_is_neutral_and_the_friction_is_what_makes_it_so() -> None:
    """The dominant growth rate of the production map (r1 2e-4, r2 1e-4, tau_rad 2000, tripwire) about the
    rest
    state at Nw 42, C_used, by power iteration (24 blocks of 4 000 steps; the mean of the last 12). Measured:
    +0.8e-5 per step (blocks wander between -1.1 and +3.1e-5; 16 blocks gave -0.7e-5, the reviewer's 20 gave
    -0.3e-5, so the estimate is neutral within about +-2e-5), against +6.7e-5 with r1 and r2 halved
    (e-folding in
    15 000 steps). The bound +3e-5 (e-folding in 33 000 steps) fails if the margin that friction buys is
    eaten;
    the control asserts that the instrument sees it. **r1, r2 and tau_rad are stability parameters of this
    weather**; S4c's calibration re-runs this test after changing them, Nw, Rp or dt."""
    c = S.WeatherParams().c_ext / (0.81 * weather_sph.cell_side(90.0, 42))
    nominal = S.dominant_rate(42, 90.0, c, 96000, dry=False)
    assert float(np.mean(nominal[12:])) < 3e-5, nominal
    halved = S.dominant_rate(42, 90.0, c, 96000, dry=False, r1=1e-4, r2=5e-5)
    assert float(np.mean(halved[12:])) > 3e-5, halved


def _dense_rho(nw: int, c: float, f0: float, radius: float = 90.0) -> float:
    """Spectral radius of the one-step map linearised at the rest state (no advection, dry), by central
    differences of the float32 MLX step (columns for h1, h2 and the three components of each velocity)."""
    p = params(nw, radius, dt=_dt_for(nw, radius, c), f0=f0)
    n = 12 * nw * nw
    fo = forcing(p, np.zeros(n))
    key = mx.random.key(0)
    base = weather_sph.init_state(p, fo)
    eps = 0.05

    def one(s: weather_sph.WeatherStateS) -> np.ndarray:
        o, _ = weather_sph.step(s, fo, key, p, radius, dry=True, advect=False, seeds=False)
        mx.eval(*o)
        return np.concatenate(
            [np.array(o.h1), np.array(o.h2), np.array(o.u1).ravel(), np.array(o.u2).ravel()]
        ).astype(np.float64)

    cols = []
    names = ["h1", "h2", "u1", "u2"]
    for name in names:
        arr = np.array(getattr(base, name))
        for idx in np.ndindex(arr.shape):
            pair = []
            for sg in (1.0, -1.0):
                a2 = arr.copy()
                a2[idx] += sg * eps
                pair.append(one(base._replace(**{name: mx.array(a2)})))
            cols.append((pair[0] - pair[1]) / (2 * eps))
    a = np.array(cols).T
    return float(np.abs(np.linalg.eigvals(a)).max())


@pytest.mark.parametrize("f0", [0.0, 0.025])
def test_the_dense_one_step_map_has_no_growing_mode_at_nside_3(f0: float) -> None:
    """ρ of the linearised one-step map at the rest state (dry, no advection) at Nside 3 (dimension 8·108): no
    mode grows beyond the finite-difference noise of float32 (the S4a float64 value of ρ - 1 is 1e-11; with
    eps
    = 0.05 the float32 evaluation resolves ρ - 1 to about 1e-3)."""
    assert _dense_rho(3, 0.5, f0) < 1.0 + 2e-3


@pytest.mark.slow
@pytest.mark.parametrize("nw,f0", [(4, 0.0), (4, 0.025), (6, 0.0), (6, 0.025)])
def test_the_dense_one_step_map_has_no_growing_mode_up_to_nside_6(nw: int, f0: float) -> None:
    for c in (0.3, 0.5, 0.6):
        assert _dense_rho(nw, c, f0) < 1.0 + 2e-3, (nw, c, f0)


# --- S4a gate G1 on the MLX step, the dense map in float64, pack/unpack (review M5, M8) ---------------------


@pytest.mark.parametrize("nw,radius", [(16, 36.0), (22, 48.0)])
def test_a_solid_rotation_through_the_production_faces_is_divergence_free(nw: int, radius: float) -> None:
    """S4a gate G1 on the shipped step: with the filter and the rotation off, uniform thickness and a
    solid-body
    rotation, one step changes ``h`` by ``-dt·h·div u``; the divergence the gradient-corrected faces see is
    below
    3 % of ``|ω|`` at its maximum (S4a: 3 %) and 1.5 % rms. With δ flipped it is 60 %, with the plain faces
    30 %."""
    p = params(nw, radius, dt=0.5, f0=0.0)
    t = weather_tab.weather_tab(nw)
    omega = 1e-3
    u = np.cross([0.0, 0.0, omega], t.x) * radius
    n = t.npix
    f = lambda a: mx.array(np.asarray(a), dtype=mx.float32)  # noqa: E731
    ws = weather_sph.WeatherStateS(
        f(np.full(n, 32.0)),
        f(np.full(n, 48.0)),
        f(u),
        f(u),
        f(np.full(n, 0.7)),
        f(np.zeros(n)),
        f(np.zeros(n)),
    )
    fo = forcing(p, np.zeros(n))
    o, _ = weather_sph.step(ws, fo, mx.random.key(0), p, radius, dry=True, advect=False, seeds=False, a4=0.0)
    mx.eval(*o)
    div = (np.array(ws.h1, dtype=np.float64) - np.array(o.h1, dtype=np.float64)) / (p.dt * 32.0)
    assert np.abs(div).max() / omega < 0.03 and np.sqrt((div**2).mean()) / omega < 0.015
    # the control: the same step with the wrong sign of the correction is far worse (the test can fail)
    tab = weather_sph.dev_for(p, radius)
    flipped = tab._replace(dx=-tab.dx, bm=-tab.bm)
    orig = weather_sph.dev_for
    weather_sph.dev_for = lambda p_, r_: flipped
    try:
        o2, _ = weather_sph.step(
            ws, fo, mx.random.key(0), p, radius, dry=True, advect=False, seeds=False, a4=0.0
        )
        mx.eval(*o2)
    finally:
        weather_sph.dev_for = orig
    bad = (np.array(ws.h1, dtype=np.float64) - np.array(o2.h1, dtype=np.float64)) / (p.dt * 32.0)
    assert np.abs(bad).max() / omega > 0.3


def test_pack_and_unpack_round_trip_a_random_state_plane_by_plane() -> None:
    rng = np.random.default_rng(5)
    n = 12 * 6 * 6
    f = lambda *shape: mx.array(rng.normal(size=shape), dtype=mx.float32)  # noqa: E731
    ws = weather_sph.WeatherStateS(f(n), f(n), f(n, 3), f(n, 3), f(n), f(n), f(n))
    wx = weather_sph.pack(ws)
    assert wx.shape == (weather_sph.N_WX, n)
    back = weather_sph.unpack(wx)
    for a, b in zip(ws, back, strict=True):
        assert np.array_equal(np.array(a), np.array(b))
    w = np.array(wx)  # the documented plane order: h1, h2, u1 xyz, u2 xyz, Q, P, Dc
    assert np.array_equal(w[2:5].T, np.array(ws.u1)) and np.array_equal(w[5:8].T, np.array(ws.u2))
    assert (
        np.array_equal(w[0], np.array(ws.h1))
        and np.array_equal(w[8], np.array(ws.Q))
        and np.array_equal(w[10], np.array(ws.Dc))
    )


@pytest.fixture
def float64_step(monkeypatch: pytest.MonkeyPatch):
    """The step in float64 on the CPU (MLX computes float64 on the CPU only): the module's float type, its
    cached tables and the default device are switched for one test."""
    old = mx.default_device()
    mx.set_default_device(mx.cpu)
    monkeypatch.setattr(weather_sph, "_F32", mx.float64)
    weather_sph.device_tables.cache_clear()
    yield
    weather_sph.device_tables.cache_clear()
    mx.set_default_device(old)


def _dense_rho64(nw: int, c: float, f0: float, radius: float = 90.0) -> tuple[float, int]:
    """ρ of the linearised dry one-step map at the rest state in float64 (ε = 1e-6), and the count of
    eigenvalues beyond ``1 + 1e-9`` (the conserved mass modes sit at 1 to rounding)."""
    p = params(nw, radius, dt=_dt_for(nw, radius, c), f0=f0)
    n = 12 * nw * nw
    fo = WeatherForcing(mx.zeros((n,), dtype=mx.float64), mx.ones((n,), dtype=mx.float64))
    key = mx.random.key(0)
    base = {"h1": np.full(n, p.H1), "h2": np.full(n, p.H2), "u1": np.zeros((n, 3)), "u2": np.zeros((n, 3))}

    def one(**ov: np.ndarray) -> np.ndarray:
        d = {**base, **ov}
        s = weather_sph.WeatherStateS(
            *(mx.array(d[k], dtype=mx.float64) for k in ("h1", "h2", "u1", "u2")),
            mx.ones((n,), dtype=mx.float64),
            mx.zeros((n,), dtype=mx.float64),
            mx.zeros((n,), dtype=mx.float64),
        )
        o, _ = weather_sph.step(s, fo, key, p, radius, dry=True, advect=False, seeds=False)
        mx.eval(*o)
        return np.concatenate(
            [np.array(o.h1), np.array(o.h2), np.array(o.u1).ravel(), np.array(o.u2).ravel()]
        )

    eps, cols = 1e-6, []
    for name, arr in base.items():
        for idx in np.ndindex(arr.shape):
            pair = []
            for sg in (1.0, -1.0):
                a2 = arr.copy()
                a2[idx] += sg * eps
                pair.append(one(**{name: a2}))
            cols.append((pair[0] - pair[1]) / (2 * eps))
    r = np.abs(np.linalg.eigvals(np.array(cols).T))
    return float(r.max()) - 1.0, int((r > 1 + 1e-9).sum())


@pytest.mark.parametrize("f0", [0.0, 0.025])
def test_the_dense_one_step_map_in_float64_has_no_growing_mode(float64_step, f0: float) -> None:
    """The float32 dense test cannot tell a mode growing at 1.001 from rounding (review M8); in float64 the
    spectral radius of the linearised map exceeds 1 by less than 1e-6 (measured 1e-9: the conserved modes)."""
    for c in (0.5, 0.64):
        excess, _ = _dense_rho64(3, c, f0)
        assert excess < 1e-6, (c, f0, excess)


@pytest.mark.slow
@pytest.mark.parametrize("nw,f0", [(4, 0.025), (6, 0.0), (6, 0.025)])
def test_the_dense_one_step_map_in_float64_up_to_nside_6(float64_step, nw: int, f0: float) -> None:
    for c in (0.3, 0.5, 0.6, 0.64):
        excess, _ = _dense_rho64(nw, c, f0)
        assert excess < 1e-6, (nw, c, f0, excess)
