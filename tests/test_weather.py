"""The W0 weather model (``alife.weather``, STORMS §3.2; spec §8): paper values, budgets, numerics, sampler.

CPU and fast. "Bitwise" means bitwise on CPU and ``assert_allclose(rtol=1e-6, atol=1e-6)`` on the GPU
(``helpers.ON_GPU``), the pattern of ``test_lifecycle_compiles_and_matches_eager``. Parameter sets are built
with ``dataclasses.replace`` and not validated (the numerics tests switch the damping off on purpose).
"""

from __future__ import annotations

import ast
import json
import math
from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from alife import weather as W
from alife.config import load_config
from alife.weather import WeatherForcing, WeatherParams, WeatherState
from helpers import ON_GPU

P0 = WeatherParams()
TWO_PI = 2.0 * math.pi


def _np(a) -> np.ndarray:
    return np.asarray(a, dtype=np.float64)


def small(n: int = 32, **kw) -> WeatherParams:
    """The start set on an ``n`` × ``n`` grid (L = 2n), seeds off unless asked, with overrides."""
    kw.setdefault("seed_rate", 0.0)
    return replace(P0, n=n, **kw)


def dry(n: int = 32, **kw) -> WeatherParams:
    """No moisture feedback, no evaporation, no physical damping, no radiation: pure shallow water."""
    base = dict(gamma=0.0, k_E=0.0, tau_rad=0.0, r1=0.0, c_d=0.0, r2=0.0, nu=0.0)
    return small(n, **{**base, **kw})


def rest(p: WeatherParams, **fields: mx.array) -> tuple[WeatherState, WeatherForcing]:
    fo = W.default_forcing(p)
    ws = W.init_state(p, fo)
    return ws._replace(**fields), fo


def rand_state(p: WeatherParams, seed: int = 0, u: float = 0.05, dh: float = 0.5) -> WeatherState:
    r = np.random.default_rng(seed)
    n = p.n

    def f(s: float) -> mx.array:
        return mx.array(r.uniform(-1, 1, (n, n)) * s, dtype=mx.float32)

    zero = mx.zeros((n, n), dtype=mx.float32)
    q = mx.array(r.uniform(0.6, 1.05, (n, n)), dtype=mx.float32)
    return WeatherState(p.H1 + f(dh), p.H2 + f(dh), f(u), f(u), f(u), f(u), q, zero, zero)


def keys(count: int, seed: int = 0) -> mx.array:
    return mx.random.split(mx.random.key(seed), count)


def run(fn, ws, fo, ks) -> tuple[WeatherState, list]:
    diags = []
    for i in range(ks.shape[0]):
        ws, d = fn(ws, fo, ks[i])
        mx.eval(ws, d)
        diags.append(d)
    return ws, diags


def same(a, b, what: str = "") -> None:
    x, y = np.asarray(a), np.asarray(b)
    if ON_GPU and x.dtype.kind == "f":
        np.testing.assert_allclose(x, y, rtol=1e-6, atol=1e-6, err_msg=what)
    else:
        assert np.array_equal(x, y), what


def same_state(a, b, what: str = "") -> None:
    for f in a._fields:
        same(getattr(a, f), getattr(b, f), f"{what} {f}")


def same_diag(a, b, what: str = "") -> None:
    for f in a._fields:
        x, y = getattr(a, f), getattr(b, f)
        assert (x is None) == (y is None), f
        if x is not None:
            same(x, y, f"{what} {f}")


# --- 1. paper values, validation, JSON --------------------------------------------------------------------


def test_paper_values() -> None:
    p = WeatherParams()
    assert p.validate() == []
    assert p.c_ext == pytest.approx(0.900, abs=1e-3)
    assert p.c_bc == pytest.approx(0.300, abs=1e-3)
    assert p.ld_pole == pytest.approx(12.0, abs=0.01)
    assert p.courant == pytest.approx(0.45, abs=1e-3)
    assert p.eps == pytest.approx(0.75)
    assert p.c_moist == pytest.approx(0.1557, abs=1e-3)
    assert p.ld_moist_pool == pytest.approx(7.71, abs=0.05)
    assert p.L == 320.0
    d = p.derived()
    assert d["inertial_period_pole"] == pytest.approx(251.3, abs=0.1)
    assert d["potential_intensity"] == pytest.approx(0.11, abs=0.01)
    assert d["mean_flow_limit"] == 0.352


RULE_CASES = [
    ("rule 1", {"n": 4}),
    ("rule 1", {"dx": 0.0}),
    ("rule 2", {"h_min": 0.0}),
    ("rule 2", {"H1": 33.0}),
    ("rule 3", {"g_prime": 0.01}),
    ("rule 4", {"dt": 1.2}),
    ("rule 5", {"f0": 0.6}),
    ("rule 6", {"f0": 0.06}),
    ("rule 6b", {"gamma": 28.8}),
    ("rule 7", {"tau_c": 6.0}),
    ("rule 8", {"gamma": 32.0}),
    ("rule 9", {"q_pool": 1.3}),
    ("rule 10", {"k_E": 4.0}),
    ("rule 11", {"tau_rad": 0.5}),
    ("rule 11", {"nu": 0.4}),
    ("rule 12", {"seed_rate": 1.5}),
    ("rule 12", {"seed_sigma": 1.0}),
    ("rule 13", {"m_min": -1.0}),
    ("rule 13", {"pools": ((0.0, 48.0, 32.0, 0.0, 1.0, 0.0, 0.0),)}),
    ("rule 14", {"kappa_d": 0.03}),
    ("rule 14", {"sl_order": 3, "kappa_d": 0.08}),
    ("rule 14", {"u_safe": 0.31}),
    ("rule 14", {"sl_order": 2}),
    ("rule 16", {"alpha1": 0.8}),
]


@pytest.mark.parametrize(("tag", "override"), RULE_CASES, ids=lambda x: str(x)[:40])
def test_each_rule_fails_on_a_violating_value(tag: str, override: dict) -> None:
    msgs = replace(P0, **override).validate()
    assert any(m.startswith(tag + ":") for m in msgs), (tag, msgs)


def test_params_json_round_trip_and_hash() -> None:
    p = replace(P0, f0=0.02, pools=((10.0, 40.0, 20.0, 10.0, 0.9, 0.001, 0.0),))
    q = WeatherParams.from_dict(json.loads(json.dumps(p.to_dict())))
    assert q == p
    assert hash(q) == hash(p)
    assert isinstance(q.pools, tuple) and all(isinstance(r, tuple) for r in q.pools)
    assert WeatherParams.from_dict({}) == P0
    assert WeatherParams.from_dict({"sl_order": 3.0}).sl_order == 3
    with pytest.raises(ValueError, match="unknown"):
        WeatherParams.from_dict({"f_zero": 1.0})
    with pytest.raises(ValueError, match="integer"):
        WeatherParams.from_dict({"sl_order": 1.5})
    assert {p: 1}[q] == 1  # usable as a dict / functools.cache key


# --- 13, 13b. compile equals eager (built first) ----------------------------------------------------------

BRANCHES = {
    "start": {},
    "tau_rad_off": {"tau_rad": 0.0},
    "gamma_0": {"gamma": 0.0, "m_min": 0.0},
    "nu": {"nu": 0.01},
    "kappa_0": {"kappa_d": 0.0},
    "cubic": {"sl_order": 3, "kappa_d": 0.10},
    "seed_q": {"seed_q": 0.05, "seed_rate": 0.3},
    "seed_w2": {"seed_w2": 0.5, "seed_rate": 0.3},
    "no_seeds": {"seed_rate": 0.0},
    "all_off": {"gamma": 0.0, "tau_rad": 0.0, "kappa_d": 0.0, "seed_rate": 0.0},
}


@pytest.mark.parametrize("name", list(BRANCHES))
def test_compiled_equals_eager_every_static_branch(name: str) -> None:
    """The full step, compiled and eager, 20 steps from a noisy state: the same bits on CPU (spec test 13)."""
    p = small(24, **{"seed_rate": 0.1, **BRANCHES[name]})
    fo = W.default_forcing(p)
    ws = rand_state(p, seed=1)
    ks = keys(20)
    eager, ed = run(W.make_step(p, compile=False), ws, fo, ks)
    comp, cd = run(W.make_step(p, compile=True), ws, fo, ks)
    same_state(eager, comp, name)
    for i in (0, 9, 19):
        same_diag(ed[i], cd[i], f"{name} step {i}")


@pytest.mark.parametrize("debug", [False, True])
def test_compiled_equals_eager_debug_off_and_on(debug: bool) -> None:
    p = small(24, seed_rate=0.1, seed_q=0.05)
    fo = W.default_forcing(p)
    ws = rand_state(p, seed=2)
    ks = keys(10)
    eager, ed = run(W.make_step(p, compile=False, debug=debug), ws, fo, ks)
    comp, cd = run(W.make_step(p, compile=True, debug=debug), ws, fo, ks)
    same_state(eager, comp)
    same_diag(ed[-1], cd[-1])
    assert (cd[-1].E is not None) == debug


def test_compiled_equals_eager_with_pool_drift_and_no_retrace() -> None:
    drift = ((20.0, 20.0, 12.0, 8.0, 1.0, 0.3, -0.1), (60.0, 70.0, 0.0, 8.0, 1.0, 0.0, 0.0))
    p = small(24, seed_rate=0.1, pools=drift)
    ws = rand_state(p, seed=3)
    ks = keys(20)
    traces = []

    def pools_at(t: mx.array) -> mx.array:
        traces.append(1)
        return W.warm_pools(p, t)

    cpools = mx.compile(pools_at)
    cstep, estep = W.make_step(p, compile=True), W.make_step(p, compile=False)
    qs = mx.ones((p.n, p.n), dtype=mx.float32)
    ce, cc = ws, ws
    for i in range(20):
        t = mx.array(i, dtype=mx.int32)
        t_eager = W.warm_pools(p, t)
        t_comp = cpools(t)
        same(t_eager, t_comp, f"T_s {i}")
        ce, _ = estep(ce, WeatherForcing(t_eager, qs), ks[i])
        cc, _ = cstep(cc, WeatherForcing(t_comp, qs), ks[i])
        mx.eval(ce, cc)
        if i == 0:
            first = len(traces)
    same_state(ce, cc)
    assert len(traces) == first == 1, "warm_pools(p, t) retraced with a new int32 t"
    assert not np.array_equal(_np(W.warm_pools(p, mx.array(0, dtype=mx.int32))), _np(t_comp))  # it moves


def _east(a: mx.array) -> mx.array:
    return mx.roll(a, -1, -1)


def _west(a: mx.array) -> mx.array:
    return mx.roll(a, 1, -1)


def _north(a: mx.array) -> mx.array:
    return mx.roll(a, -1, -2)


def _south(a: mx.array) -> mx.array:
    return mx.roll(a, 1, -2)


@pytest.mark.xfail(
    raises=IndexError, strict=True, reason="MLX 0.32.3 mx.compile duplicate-roll bug (spec §2.1)"
)
def test_mlx_duplicate_roll_compile_bug_is_still_there() -> None:
    """Pins the bug the once-per-array rule works around (CPU device). If an MLX upgrade fixes it this XPASSes
    (strict) and fails, which shows that the rule in ``alife.weather`` can be relaxed."""

    def f(u: mx.array) -> tuple[mx.array, mx.array]:
        u = u + 0.02 * (_east(u) + _west(u) + _north(u) + _south(u) - 4 * u)
        return (u - _west(u)) / 2, 0.5 * (u + _west(u))  # the second _west(u) is the duplicate

    with mx.stream(mx.cpu):
        u = mx.random.normal((32, 32), key=mx.random.key(0))
        mx.eval(mx.compile(f)(u))


def test_the_once_per_array_form_compiles_and_matches_eager() -> None:
    """The same function with the shift held in a variable: compiles, and equals eager bitwise."""

    def f(u: mx.array) -> tuple[mx.array, mx.array]:
        u = u + 0.02 * (_east(u) + _west(u) + _north(u) + _south(u) - 4 * u)
        wu = _west(u)
        return (u - wu) / 2, 0.5 * (u + wu)

    with mx.stream(mx.cpu):
        u = mx.random.normal((32, 32), key=mx.random.key(0))
        comp, eager = mx.compile(f)(u), f(u)
        mx.eval(comp, eager)
    assert all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(comp, eager, strict=True))


# --- 2, 3, 3b, 3c, 4. conservation, budgets, sign, positivity ---------------------------------------------


def test_dry_mass_conservation() -> None:
    p = dry(32)
    r = np.random.default_rng(0)
    n = p.n

    def f(s: float) -> mx.array:
        return mx.array(r.uniform(-1, 1, (n, n)) * s, dtype=mx.float32)

    ws, fo = rest(p, h1=p.H1 + f(2.0), h2=p.H2 + f(2.0), u1=f(0.1), v1=f(0.1), u2=f(0.1), v2=f(0.1))
    m0 = (_np(ws.h1).sum(), _np(ws.h2).sum())
    out, diags = run(W.make_step(p), ws, fo, keys(200))
    for k, h in enumerate((out.h1, out.h2)):
        assert abs(_np(h).sum() - m0[k]) / m0[k] <= 1e-6, k
    assert sum(int(d.n_clamp.item()) for d in diags) == 0


def _moist_params() -> WeatherParams:
    pool = ((48.0, 24.0, 14.0, 10.0, 1.0, 0.0, 0.0),)
    return small(48, pools=pool, seed_rate=0.2, seed_q=0.1)


def test_moist_budgets_close_with_and_without_debug_arrays() -> None:
    p = _moist_params()
    fo = W.default_forcing(p)
    ws = W.init_state(p, fo)
    q0, m0 = _np(ws.Q).sum(), _np(ws.h1).sum() + _np(ws.h2).sum()
    out, diags = run(W.make_step(p, debug=True), ws, fo, keys(200))
    sums = {k: 0.0 for k in ("E", "C", "Qseed")}
    diag_sums = {k: 0.0 for k in ("sum_E", "sum_C", "sum_Qseed", "q_clamp")}
    for d in diags:
        sums["E"] += _np(d.E).sum() * p.dt
        sums["C"] += _np(d.C).sum() * p.dt
        sums["Qseed"] += _np(d.Qseed).sum()
        for k in diag_sums:
            diag_sums[k] += float(_np(getattr(d, k)))
        assert abs(float(_np(d.sum_E)) - _np(d.E).sum() * p.dt) <= 1e-6 * max(_np(d.E).sum(), 1e-12)
        assert abs(float(_np(d.sum_C)) - _np(d.C).sum() * p.dt) <= 1e-6 * max(_np(d.C).sum(), 1e-12)
    assert sums["C"] > 0 and sums["E"] > 0 and sums["Qseed"] > 0  # not vacuous
    assert abs(_np(out.h1).sum() + _np(out.h2).sum() - m0) / m0 <= 1e-6
    dq = _np(out.Q).sum() - q0
    assert abs(dq - (sums["E"] - sums["C"] + sums["Qseed"])) <= 1e-5 * q0
    assert abs(
        dq - (diag_sums["sum_E"] - diag_sums["sum_C"] + diag_sums["sum_Qseed"] + diag_sums["q_clamp"])
    ) <= (1e-5 * q0)
    assert sum(int(d.n_clamp.item()) for d in diags) == 0


def test_moist_response_has_the_sign_of_latent_heating() -> None:
    """A supersaturated disc on the north pole row condenses into a cyclone with a thinned lower layer (the
    budget test passes with the heating reversed, this one does not)."""
    p = replace(small(64, k_E=0.0, tau_rad=0.0), pools=())
    n, dx = p.n, p.dx
    xc = (np.arange(n) + 0.5) * dx
    r2 = (xc[None, :] - 64.0) ** 2 + (xc[:, None] - 32.0) ** 2
    q = mx.array(np.where(r2 <= 8.0**2, 1.1, 0.7), dtype=mx.float32)
    ws, fo = rest(p, Q=q)
    out, diags = run(W.make_step(p), ws, fo, keys(300))
    h1c = _np(out.h1)[16, 31:33].mean()  # the rows around y = 32 (the pole row)
    assert h1c < p.H1 - 0.1
    zeta = _np(W.vorticity(out, p))[15, 31]  # the corner at (64, 32)
    assert zeta > 0.05 * p.f0  # cyclonic in the northern hemisphere
    dm = _np(out.h1).sum() - _np(ws.h1).sum()
    assert dm == pytest.approx(-p.gamma * sum(float(_np(d.sum_C)) for d in diags), rel=1e-5)


def test_mass_exchange_conserves_face_momentum_and_stays_convex() -> None:
    n = 16
    r = np.random.default_rng(5)
    h = mx.array(r.uniform(5, 30, (2, n, n)), dtype=mx.float32)
    u = mx.array(r.uniform(-0.1, 0.1, (2, n, n)), dtype=mx.float32)
    v = mx.array(r.uniform(-0.1, 0.1, (2, n, n)), dtype=mx.float32)

    def face(hh, uu, vv):
        hh, uu, vv = _np(hh), _np(uu), _np(vv)
        hu = 0.5 * (hh + np.roll(hh, -1, 2))  # east
        hv = 0.5 * (hh + np.roll(hh, -1, 1))  # north
        return (hu * uu).sum(0), (hv * vv).sum(0)

    # condensation: mu in (0, 0.4 h1)
    mu = mx.array(r.uniform(0, 0.4, (n, n)) * _np(h)[0], dtype=mx.float32)
    h2, u2, v2 = W.exchange_condense(h, u, v, mu)
    for a, b in zip(face(h, u, v), face(h2, u2, v2), strict=True):
        np.testing.assert_allclose(b, a, rtol=1e-6, atol=1e-7)
    assert np.array_equal(_np(u2)[0], _np(u)[0]) and np.array_equal(_np(v2)[0], _np(v)[0])  # donor unchanged
    for new, old in ((u2, u), (v2, v)):
        lo, hi = np.minimum(_np(old)[0], _np(old)[1]), np.maximum(_np(old)[0], _np(old)[1])
        assert (_np(new)[1] >= lo - 1e-7).all() and (_np(new)[1] <= hi + 1e-7).all()
    # radiative return: positive r (layer 1 receives), negative r (layer 2 receives), and mixed signs
    for sign in (1.0, -1.0, 0.0):
        rr = r.uniform(0, 1, (n, n)) * (
            np.where(r.uniform(size=(n, n)) < 0.5, -1.0, 1.0) if sign == 0 else sign
        )
        rm = mx.array(0.3 * rr * np.where(rr > 0, _np(h)[1], _np(h)[0]), dtype=mx.float32)
        h3, u3, v3 = W.exchange_relax(h, u, v, rm)
        for a, b in zip(face(h, u, v), face(h3, u3, v3), strict=True):
            np.testing.assert_allclose(b, a, rtol=1e-6, atol=1e-7)
        if sign != 0:
            donor, recv = (1, 0) if sign > 0 else (0, 1)
            assert np.array_equal(_np(u3)[donor], _np(u)[donor])
            for new, old in ((u3, u), (v3, v)):
                lo, hi = np.minimum(_np(old)[0], _np(old)[1]), np.maximum(_np(old)[0], _np(old)[1])
                assert (_np(new)[recv] >= lo - 1e-7).all() and (_np(new)[recv] <= hi + 1e-7).all()


def test_positivity_limiter_engages_and_holds() -> None:
    p = small(32)
    n = p.n
    x_u = (np.arange(n) + 1.0) * p.dx
    u1 = mx.array(np.tile(-0.25 * np.sin(TWO_PI * x_u / p.L), (n, 1)), dtype=mx.float32)
    ws, fo = rest(p, h1=mx.full((n, n), 2.5), u1=u1, Q=mx.full((n, n), 1.05))
    fn = W.make_step(p)
    lim = clamps = 0
    for k in keys(30):
        ws, d = fn(ws, fo, k)
        mx.eval(ws, d)
        lim += int(d.n_limited.item())
        clamps += int(d.n_clamp.item())  # the limiter alone keeps h >= h_min: the rounding clamp stays idle
        assert _np(ws.h1).min() >= p.h_min and _np(ws.h2).min() >= p.h_min and _np(ws.Q).min() >= 0.0
        assert int(d.n_safety.item()) == 0
    assert lim > 0 and clamps == 0
    assert np.isfinite(_np(ws.u1)).all()


# --- 5, 5b. gravity waves and mean-flow stability ---------------------------------------------------------


@pytest.mark.parametrize("mode", ["external", "baroclinic"])
def test_gravity_wave_speed_and_neutrality(mode: str) -> None:
    p = dry(64, f0=0.0, kappa_d=0.0)
    c = p.c_ext if mode == "external" else p.c_bc
    ratio = (c * c - p.H2 * p.g_top) / (p.H2 * p.g_top)  # h1'/h2' of the mode
    n, dx, length = p.n, p.dx, p.L
    k = TWO_PI / length
    xc = (np.arange(n) + 0.5) * dx
    shape = np.tile(np.cos(k * xc), (n, 1))
    amp = 1e-3
    ws, fo = rest(
        p,
        h1=p.H1 + amp * mx.array(shape, dtype=mx.float32),
        h2=p.H2 + amp / ratio * mx.array(shape, dtype=mx.float32),
    )
    omega = 2.0 * math.asin(c * p.dt / dx * math.sin(k * dx / 2.0)) / p.dt
    period = TWO_PI / omega
    steps = int(1.75 * period) + 2
    fn = W.make_step(p)
    series = [(_np(ws.h1) * shape).sum() * 2.0 / n**2]
    for kk in keys(steps):
        ws, _ = fn(ws, fo, kk)
        mx.eval(ws)
        series.append((_np(ws.h1) * shape).sum() * 2.0 / n**2)
    s = np.array(series)
    cross = [i + s[i] / (s[i] - s[i + 1]) for i in range(len(s) - 1) if s[i] * s[i + 1] < 0]
    assert len(cross) >= 3
    measured = 2.0 * (cross[-1] - cross[0]) / (len(cross) - 1)
    assert measured == pytest.approx(period, rel=0.005)
    assert measured == pytest.approx(TWO_PI / (c * k), rel=0.01)
    om = TWO_PI / measured
    t = np.arange(len(s), dtype=np.float64)

    def amplitude(lo: float, hi: float) -> float:
        m = (t >= lo) & (t <= hi)
        design = np.stack([np.cos(om * t[m]), np.sin(om * t[m])], axis=1)
        coef = np.linalg.lstsq(design, s[m], rcond=None)[0]
        return float(np.hypot(*coef))

    a1, a2 = amplitude(0.0, 0.75 * measured), amplitude(0.75 * measured, 1.5 * measured)
    assert abs(a2 - a1) / a1 <= 0.01  # undamped forward-backward is neutral
    assert a1 == pytest.approx(amp, rel=0.02)


def _growth_factor(p: WeatherParams, base: dict[str, tuple[float, float]], steps: int = 2000) -> float:
    """Product of the renormalisation factors of a 1e-4 perturbation about a uniform flow (spec test 5b)."""
    n = p.n
    fields = ("h1", "h2", "u1", "v1", "u2", "v2")
    const = {
        "h1": p.H1,
        "h2": p.H2,
        "u1": base["u"][0],
        "v1": base["u"][1],
        "u2": base["v"][0],
        "v2": base["v"][1],
    }
    ws, fo = rest(p)
    ws = ws._replace(**{f: mx.full((n, n), const[f], dtype=mx.float32) for f in fields})
    ref = {f: _np(getattr(ws, f)) for f in fields}
    r = np.random.default_rng(0)
    noise = {f: 1e-4 * r.standard_normal((n, n)) for f in fields}
    ws = ws._replace(**{f: mx.array(ref[f] + noise[f], dtype=mx.float32) for f in fields})
    fn = W.make_step(p)
    prod, norm0 = 1.0, math.sqrt(sum((v**2).sum() for v in noise.values()))
    last = norm0
    kk = keys(steps)
    for blk in range(steps // 100):
        for i in range(100):
            ws, d = fn(ws, fo, kk[blk * 100 + i])
            mx.eval(ws)
            assert int(d.n_safety.item()) == 0
        delta = {f: _np(getattr(ws, f)) - ref[f] for f in fields}
        norm = math.sqrt(sum((v**2).sum() for v in delta.values()))
        prod *= norm / last
        ws = ws._replace(
            **{f: mx.array(ref[f] + delta[f] * (norm0 / norm), dtype=mx.float32) for f in fields}
        )
        last = norm0
    return prod


@pytest.mark.parametrize(
    ("order", "kappa"), [(1, 0.04), (3, 0.10)], ids=["bilinear_kd_0.04", "cubic_kd_0.10"]
)
@pytest.mark.parametrize("case", ["baro_0", "baro_45", "layer1_0", "layer1_45"])
def test_mean_flow_perturbations_do_not_grow(order: int, kappa: float, case: str) -> None:
    p = dry(32, f0=0.0, sl_order=order, kappa_d=kappa)
    speed = 0.29
    ang = 0.0 if case.endswith("_0") else math.pi / 4
    uu, vv = speed * math.cos(ang), speed * math.sin(ang)
    base = {"u": (uu, vv), "v": (uu, vv) if case.startswith("baro") else (0.0, 0.0)}
    assert _growth_factor(p, base) <= 1.0 + 1e-3


def test_mean_flow_check_is_not_vacuous_without_divergence_damping() -> None:
    p = dry(32, f0=0.0, kappa_d=0.0)
    uu = 0.29 / math.sqrt(2.0)
    assert _growth_factor(p, {"u": (uu, uu), "v": (uu, uu)}) > 10.0


# --- 6, 6b. geostrophic jet and unforced-vortex decay -----------------------------------------------------


def _jet(p: WeatherParams, balanced: bool) -> WeatherState:
    n, dx, length = p.n, p.dx, p.L
    y = (np.arange(n) + 0.5) * dx  # centres and u-points share these rows
    u1 = 0.01 * np.sin(2.0 * TWO_PI * y / length)
    h1 = np.full(n, p.H1)
    if balanced:
        h1 = p.H1 - (p.f0 * 0.01 / p.g_prime) * (length / (2.0 * TWO_PI)) * (
            np.sin(TWO_PI * y / length) - np.sin(3.0 * TWO_PI * y / length) / 3.0
        )
    tile = lambda a: mx.array(np.tile(a[:, None], (1, n)), dtype=mx.float32)  # noqa: E731
    ws, _ = rest(p)
    return ws._replace(u1=tile(u1), h1=tile(h1), h2=tile(p.height - h1))


def test_geostrophic_jet_is_a_fixed_point_and_flat_h_is_not() -> None:
    p = dry(64, f0=0.025)
    fo = W.default_forcing(p)
    out = {}
    for balanced in (True, False):
        ws = _jet(p, balanced)
        u0 = _np(ws.u1)
        ws, _ = run(W.make_step(p), ws, fo, keys(500))
        out[balanced] = np.linalg.norm(_np(ws.u1) - u0) / np.linalg.norm(u0)
    assert out[True] <= 0.02
    assert out[False] >= 0.2


def _vortex(p: WeatherParams, zeta0: float = 0.55, sigma: float = 9.0) -> WeatherState:
    """A balanced Gaussian-vorticity vortex on the north pole row: psi from the discrete Poisson problem."""
    n, dx, length = p.n, p.dx, p.L
    xc = (np.arange(n) + 1.0) * dx  # corners
    x0, y0 = length / 2, length / 4
    r2 = (xc[None, :] - x0) ** 2 + (xc[:, None] - y0) ** 2
    zeta = zeta0 * p.f0 * np.exp(-r2 / (2.0 * sigma**2))
    zeta -= zeta.mean()
    kx = TWO_PI * np.fft.fftfreq(n)
    lam = (2.0 * np.cos(kx)[None, :] + 2.0 * np.cos(kx)[:, None] - 4.0) / dx**2
    lam[0, 0] = 1.0
    ph = np.fft.fft2(zeta) / lam
    ph[0, 0] = 0.0
    psi = np.real(np.fft.ifft2(ph))
    u = -(psi - np.roll(psi, 1, 0)) / dx
    v = (psi - np.roll(psi, 1, 1)) / dx
    psi_c = 0.25 * (psi + np.roll(psi, 1, 0) + np.roll(psi, 1, 1) + np.roll(np.roll(psi, 1, 0), 1, 1))
    h1 = p.H1 + p.f0 * psi_c / p.g_prime
    ws, _ = rest(p)
    f32 = lambda a: mx.array(a, dtype=mx.float32)  # noqa: E731
    return ws._replace(h1=f32(h1), h2=f32(p.height - h1), u1=f32(u), v1=f32(v))


def _decay_time(p: WeatherParams, limit: int, every: int = 25) -> int:
    """First multiple of ``every`` steps at which the centre-averaged peak zeta of layer 1 is below 0.3 f0."""
    ws = _vortex(p)
    fo = W.default_forcing(p)
    fn = W.make_step(p)
    kk = keys(limit)
    for t in range(1, limit + 1):
        ws, _ = fn(ws, fo, kk[t - 1])
        if t % every == 0:
            mx.eval(ws)
            z = _np(W.vorticity(ws, p))
            zc = 0.25 * (z + np.roll(z, 1, 1) + np.roll(z, 1, 0) + np.roll(np.roll(z, 1, 0), 1, 1))
            if zc.max() / p.f0 < 0.3:
                return t
    return limit + 1


def _fplane(p: WeatherParams):
    phi = 0.25 * p.dt * p.f0  # the pole-row angle, constant in y
    c = mx.full((1, p.n, 1), math.cos(phi), dtype=mx.float32)
    s = mx.full((1, p.n, 1), math.sin(phi), dtype=mx.float32)
    return c, s, c, s


def test_unforced_vortex_outlives_the_nu_damped_one_on_an_f_plane(monkeypatch) -> None:
    """Regression for spec §3.1 (the f-plane check): with the default numerics a storm-sized vortex keeps
    zeta >= 0.3 f0 for at least 1 200 steps and 1.8 times as long as with the probe's nu = 0.02 and no
    divergence damping (measured here at N = 64: 1 375 and 700, ratio 1.96; the check read 1 400 and 700)."""
    monkeypatch.setattr(W, "rotation_tables", _fplane)
    base = dict(gamma=0.0, k_E=0.0, tau_rad=0.0)
    t_default = _decay_time(small(64, **base), 2000)
    t_nu = _decay_time(small(64, nu=0.02, kappa_d=0.0, **base), 2000)
    assert t_default >= 1200, (t_default, t_nu)
    assert t_default >= 1.8 * t_nu, (t_default, t_nu)


def test_unforced_vortex_lives_1200_steps_on_the_beta_plane() -> None:
    """The same vortex with the real f(y) on the north pole row (N = 64: 1 525 steps; Rossby-wave radiation
    shortens the lifetime of both variants, so the ratio of the f-plane test does not carry over: 1.74)."""
    assert _decay_time(small(64, gamma=0.0, k_E=0.0, tau_rad=0.0), 2000) >= 1200


# --- 7, 8, 9, 10, 11, 12. rotation, symmetries, seams, determinism, seeds ---------------------------------


def test_rotation_never_adds_energy_and_matches_the_exact_rotation() -> None:
    p = small(32)
    r = np.random.default_rng(7)
    u = mx.array(r.standard_normal((2, 32, 32)), dtype=mx.float32)
    v = mx.array(r.standard_normal((2, 32, 32)), dtype=mx.float32)
    un, vn = W.rotate_half(u, v, W.rotation_tables(p))
    ke = lambda a, b: (_np(a) ** 2).sum() + (_np(b) ** 2).sum()  # noqa: E731
    assert ke(un, vn) <= ke(u, v) * (1.0 + 1e-6)
    assert ke(un, vn) < ke(u, v)  # the varying f does smooth the field a little
    # constant angle: uniform fields rotate exactly by 2 phi = f dt / 2 (spec S3)
    phi = 0.012
    c = mx.full((1, 32, 1), math.cos(phi), dtype=mx.float32)
    s = mx.full((1, 32, 1), math.sin(phi), dtype=mx.float32)
    ua, va = mx.full((2, 32, 32), 0.3), mx.full((2, 32, 32), -0.1)
    un, vn = W.rotate_half(ua, va, (c, s, c, s))
    ang = 2.0 * phi
    assert np.allclose(_np(un), 0.3 * math.cos(ang) + -0.1 * math.sin(ang), atol=1e-6)
    assert np.allclose(_np(vn), -0.3 * math.sin(ang) + -0.1 * math.cos(ang), atol=1e-6)
    # zero angles: the identity, bitwise
    one, zero = mx.ones((1, 32, 1)), mx.zeros((1, 32, 1))
    un, vn = W.rotate_half(u, v, (one, zero, one, zero))
    assert np.array_equal(_np(un), _np(u)) and np.array_equal(_np(vn), _np(v))


def _mirror(ws: WeatherState) -> WeatherState:
    def centre(a):
        return mx.array(np.asarray(a)[::-1].copy())

    def vface(a):
        return mx.array(-np.roll(np.asarray(a)[::-1], -1, axis=0))

    return ws._replace(
        h1=centre(ws.h1), h2=centre(ws.h2), u1=centre(ws.u1), u2=centre(ws.u2),
        v1=vface(ws.v1), v2=vface(ws.v2), Q=centre(ws.Q), P=centre(ws.P), Dc=centre(ws.Dc),
    )  # fmt: skip


def test_hemispheric_mirror_equivariance() -> None:
    p = small(48)
    ws = rand_state(p, seed=11)
    fo = W.default_forcing(p)
    fo_m = WeatherForcing(mx.array(np.asarray(fo.T_s)[::-1].copy()), fo.qs)
    ws_m = _mirror(ws)
    fn = W.make_step(p)
    ks = keys(20)
    for steps, tol in ((1, 2e-6), (20, 5e-5)):
        a, b = ws, ws_m
        for i in range(steps):
            a, _ = fn(a, fo, ks[i])
            b, _ = fn(b, fo_m, ks[i])
        mx.eval(a, b)
        a_m = _mirror(a)
        for f in ("h1", "h2", "u1", "u2", "v1", "v2", "Q"):
            x, y = _np(getattr(a_m, f)), _np(getattr(b, f))
            assert np.abs(x - y).max() <= tol * np.abs(x).max(), (steps, f)


def _roll(ws: WeatherState, shift: int, axis: int) -> WeatherState:
    return WeatherState(*(mx.roll(getattr(ws, f), shift, axis=axis) for f in ws._fields))


@pytest.mark.parametrize(("axis", "f0"), [(1, 0.025), (1, 0.0), (0, 0.0)], ids=["x", "x_f0_0", "y_f0_0"])
def test_translation_equivariance(axis: int, f0: float) -> None:
    p = small(32, f0=f0)
    ws = rand_state(p, seed=4)
    fo = W.default_forcing(p)
    fo_r = WeatherForcing(mx.roll(fo.T_s, 7, axis=axis), mx.roll(fo.qs, 7, axis=axis))
    fn = W.make_step(p)
    ks = keys(20)
    a, _ = run(fn, ws, fo, ks)
    b, _ = run(fn, _roll(ws, 7, axis), fo_r, ks)
    same_state(_roll(a, 7, axis), b)


def _sl_reference(a: np.ndarray, ox: np.ndarray, oy: np.ndarray, order: int) -> np.ndarray:
    k, n = a.shape[0], a.shape[-1]
    jj, ii = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    out = np.empty(a.shape)
    for q in range(k):
        fx, fy = np.floor(ox[q]), np.floor(oy[q])
        ax, ay = ox[q] - fx, oy[q] - fy
        j0, i0 = (jj + fy.astype(int)), (ii + fx.astype(int))

        def at(dj, di, q=q, j0=j0, i0=i0):
            return a[q][(j0 + dj) % n, (i0 + di) % n]

        bil = (1 - ay) * ((1 - ax) * at(0, 0) + ax * at(0, 1)) + ay * ((1 - ax) * at(1, 0) + ax * at(1, 1))
        if order == 1:
            out[q] = bil
            continue

        def w(t):
            return [
                (-(t**3) + 2 * t**2 - t) / 2,
                (3 * t**3 - 5 * t**2 + 2) / 2,
                (-3 * t**3 + 4 * t**2 + t) / 2,
                (t**3 - t**2) / 2,
            ]

        wx, wy = w(ax), w(ay)
        val = sum(wy[m] * sum(wx[c] * at(m - 1, c - 1) for c in range(4)) for m in range(4))
        cs = np.stack([at(0, 0), at(0, 1), at(1, 0), at(1, 1)])
        out[q] = np.clip(val, cs.min(0), cs.max(0))
    return out


@pytest.mark.parametrize("order", [1, 3])
def test_semi_lagrangian_matches_a_float64_periodic_reference(order: int) -> None:
    n = 16
    r = np.random.default_rng(order)
    a = r.uniform(-1, 1, (3, n, n)).astype(np.float32)
    ox = r.uniform(-2.3, 2.3, (3, n, n)).astype(np.float32)
    oy = r.uniform(-2.3, 2.3, (3, n, n)).astype(np.float32)
    ox[0, :2], oy[0, :2] = -0.3, -0.7  # offsets that index -1 at the seams
    got = W.semi_lagrangian(mx.array(a), mx.array(ox), mx.array(oy), order)
    ref = _sl_reference(a.astype(np.float64), ox.astype(np.float64), oy.astype(np.float64), order)
    assert np.abs(_np(got) - ref).max() <= 1e-6


@pytest.mark.parametrize("direction", [1.0, -1.0], ids=["east_north", "west_south"])
@pytest.mark.parametrize("order", [1, 3])
def test_a_blob_crosses_both_seams_at_the_flow_speed(order: int, direction: float) -> None:
    """Content moves by -offset per call: 40 calls at 0.25 cells carry a blob 10 cells over both seams."""
    n = 32
    x = np.arange(n)
    start = (29.0, 30.0) if direction > 0 else (2.0, 1.0)
    d = lambda c: (x - c + n / 2) % n - n / 2  # noqa: E731
    blob = np.exp(-(d(start[0])[None, :] ** 2 + d(start[1])[:, None] ** 2) / (2 * 2.5**2)).astype(np.float32)
    a = mx.array(blob)[None]
    off = mx.full((1, n, n), -0.25 * direction)
    for _ in range(40):
        a = W.semi_lagrangian(a, off, off, order)
    b = _np(a[0])

    def centre(profile: np.ndarray) -> float:
        return (np.angle((profile * np.exp(TWO_PI * 1j * x / n)).sum()) % TWO_PI) * n / TWO_PI

    # a uniform shift by bilinear weights is a convolution, so mass is conserved; the cubic clamp trims a bit
    assert b.sum() == pytest.approx(blob.sum(), rel=1e-4 if order == 1 else 0.01)
    assert centre(b.sum(0)) == pytest.approx((start[0] + 10.0 * direction) % n, abs=0.05)
    assert centre(b.sum(1)) == pytest.approx((start[1] + 10.0 * direction) % n, abs=0.05)


def test_determinism_and_key_sensitivity() -> None:
    p = small(24, seed_rate=1.0)
    ws, fo = rest(p)
    fn = W.make_step(p)
    ks = keys(5)
    a, _ = run(fn, ws, fo, ks)
    b, _ = run(fn, ws, fo, ks)
    same_state(a, b)
    c, _ = run(fn, ws, fo, keys(5, seed=1))
    assert not np.array_equal(_np(a.u1), _np(c.u1))


def test_seeds_are_divergence_free_cyclonic_and_the_right_size() -> None:
    p = small(160, seed_rate=1.0, seed_jitter=0.0)
    for s in range(4):
        psi, on, a, xs, ys = W._seed_fields(mx.random.key(s), p)
        assert float(_np(on)) == 1.0 and float(_np(a)) == 1.0
        du, dv = W.psi_winds(psi, p.dx)
        du_, dv_ = _np(du), _np(dv)
        div = (du_ - np.roll(du_, 1, 1) + dv_ - np.roll(dv_, 1, 0)) / p.dx
        assert np.abs(div).max() <= 1e-7
        uc, vc = 0.5 * (du_ + np.roll(du_, 1, 1)), 0.5 * (dv_ + np.roll(dv_, 1, 0))
        assert np.hypot(uc, vc).max() == pytest.approx(p.seed_v, rel=0.02)
        ws, _ = rest(p, u1=du, v1=dv)
        zeta = _np(W.vorticity(ws, p))
        i, j = round(float(_np(xs)) / p.dx) - 1, round(float(_np(ys)) / p.dx) - 1
        sign_f = 1.0 if math.sin(TWO_PI * float(_np(ys)) / p.L) >= 0 else -1.0
        assert zeta[j % p.n, i % p.n] * sign_f > 0.0
        assert np.abs(zeta).max() == pytest.approx(abs(zeta[j % p.n, i % p.n]), rel=0.05)


def test_a_seed_step_changes_only_the_winds_it_adds_to() -> None:
    p_on, p_off = small(24, seed_rate=1.0), small(24, seed_rate=0.0)
    ws = rand_state(p_on, seed=8)
    fo = W.default_forcing(p_on)
    k = mx.random.key(3)
    a, _ = W.step(ws, fo, k, p_on)
    b, _ = W.step(ws, fo, k, p_off)
    for f in ("h1", "h2", "Q", "P", "Dc", "u2", "v2"):
        same(getattr(a, f), getattr(b, f), f)  # seed_w2 = 0: layer 2 untouched
    assert not np.array_equal(_np(a.u1), _np(b.u1))


# --- the optional momentum forcing (STORMS steerability) --------------------------------------------------


def _forcing_with(p: WeatherParams, scale: float) -> WeatherForcing:
    fo = W.default_forcing(p)
    z = mx.zeros((p.n, p.n), dtype=mx.float32)
    r = np.random.default_rng(0)
    push = mx.array(r.uniform(-1, 1, (p.n, p.n)) * scale, dtype=mx.float32)
    return fo._replace(fu1=push, fv1=z + scale, fu2=-push, fv2=z)


def test_zero_momentum_forcing_equals_no_forcing_bitwise() -> None:
    p = small(24, seed_rate=0.1)
    fo = W.default_forcing(p)
    fo_zero = _forcing_with(p, 0.0)
    ws = rand_state(p, seed=9)
    ks = keys(10)
    for compile_ in (False, True):
        fn = W.make_step(p, compile=compile_)
        a, da = run(fn, ws, fo, ks)
        b, db = run(fn, ws, fo_zero, ks)
        same_state(a, b, f"compile={compile_}")
        same_diag(da[-1], db[-1])


def test_momentum_forcing_adds_dt_times_f_and_does_not_recompile() -> None:
    p = dry(24, f0=0.0, kappa_d=0.0, dt=0.5, h_min=2.0)
    ws, fo = rest(p)
    push = mx.full((p.n, p.n), 1e-3, dtype=mx.float32)
    z = mx.zeros((p.n, p.n), dtype=mx.float32)
    fo_push = fo._replace(fu1=push, fv1=z, fu2=z, fv2=-push)
    out, _ = W.step(ws, fo_push, mx.random.key(0), p)
    assert np.allclose(_np(out.u1), p.dt * 1e-3, atol=1e-9) and np.allclose(
        _np(out.v2), -p.dt * 1e-3, atol=1e-9
    )
    assert np.allclose(_np(out.v1), 0.0, atol=1e-9) and np.allclose(_np(out.u2), 0.0, atol=1e-9)

    pr = small(24, seed_rate=0.1)
    traces = []

    def fn(ws_, fo_, k_):
        traces.append(1)
        return W.step(ws_, fo_, k_, pr)

    cfn = mx.compile(fn)
    ws = rand_state(pr, seed=2)
    ks = keys(6)
    scales = [0.0, 0.0, 1e-3, 5e-3, 0.0, 2e-3]
    for i, s in enumerate(scales):
        ws, d = cfn(ws, _forcing_with(pr, s), ks[i])
        mx.eval(ws, d)
        if i == 0:
            first = len(traces)
    assert len(traces) == first == 1, "a non-zero forcing recompiled the step"
    assert np.isfinite(_np(ws.u1)).all()


# --- 14, 14b, 15. the wind the organisms feel -------------------------------------------------------------


def _pos(xs, ys, zs) -> mx.array:
    return mx.array(np.stack([xs, ys, zs], axis=-1), dtype=mx.float32)


def _cell_pos(p: WeatherParams, i: int, j: int, zs) -> mx.array:
    zs = np.asarray(zs, dtype=np.float64)
    return _pos(np.full(zs.shape, (i + 0.5) * p.dx), np.full(zs.shape, (j + 0.5) * p.dx), zs)


def _cap_phi(z, zs):
    return np.where(z <= zs, z * z / (2 * zs), z - 0.5 * zs)


def _w_ref(z, zi, d1, d2, j, p):
    z = np.asarray(z, dtype=np.float64)
    big_h = p.height
    dbar = (d1 * _cap_phi(zi, p.z_sl) + d2 * (big_h - zi) + j) / big_h
    return -(d1 * _cap_phi(np.minimum(z, zi), p.z_sl) + np.where(z > zi, d2 * (z - zi) + j, 0.0) - dbar * z)


def test_sampler_hand_values_uniform_winds() -> None:
    p = small(64)
    n = p.n
    full = lambda x: mx.full((n, n), x, dtype=mx.float32)  # noqa: E731
    ws, _ = rest(p, u1=full(0.07), v1=full(0.03), u2=full(-0.05), v2=full(0.02))
    got = W.sample_wind(ws, _pos([37.3] * 3, [51.1] * 3, [1.0, 10.0, 50.0]), p)
    ref = np.array([[0.25 * 0.07, 0.25 * 0.03], [0.07, 0.03], [-0.05, 0.02]])
    np.testing.assert_allclose(_np(got.u)[:, :2], ref, atol=1e-6)
    assert np.abs(_np(got.u)[:, 2]).max() <= 1e-7
    assert int(got.n_clip_h.item()) == 0 and int(got.n_clip_w.item()) == 0


def test_sampler_divergence_over_a_flat_interface_matches_the_formula() -> None:
    p = small(64)
    n = p.n
    x_u = (np.arange(n) + 1.0) * p.dx
    u1 = np.tile(0.05 * np.sin(TWO_PI * x_u / p.L), (n, 1))
    u2 = np.tile(0.04 * np.sin(TWO_PI * x_u / p.L + 0.7), (n, 1))
    ws, _ = rest(p, u1=mx.array(u1, dtype=mx.float32), u2=mx.array(u2, dtype=mx.float32))
    i, j = 11, 20
    d1 = (_np(ws.u1)[j, i] - _np(ws.u1)[j, i - 1]) / p.dx
    d2 = (_np(ws.u2)[j, i] - _np(ws.u2)[j, i - 1]) / p.dx
    zs = np.array([2.0, 20.0, 32.0, 60.0, 80.0])
    got = W.sample_wind(ws, _cell_pos(p, i, j, zs), p)
    ref = _w_ref(zs, 32.0, d1, d2, 0.0, p)
    np.testing.assert_allclose(_np(got.u)[:, 2], ref, atol=1e-6)
    assert abs(_np(got.u)[-1, 2]) <= 1e-7  # w(H) = 0 without a taper
    assert abs(ref[0]) < p.w_edge  # the z = 2 check is not clipped
    assert np.abs(ref).max() > 1e-3  # not vacuous


def _tilted(p: WeatherParams, u: float = 0.05) -> WeatherState:
    n = p.n
    x_c = (np.arange(n) + 0.5) * p.dx
    h1 = np.tile(p.H1 + 8.0 * np.sin(TWO_PI * x_c / p.L), (n, 1))
    ws, _ = rest(p)
    f32 = lambda a: mx.array(a, dtype=mx.float32)  # noqa: E731
    return ws._replace(h1=f32(h1), h2=f32(p.height - h1), u1=mx.full((n, n), u, dtype=mx.float32))


def test_sampler_tilted_interface_gives_the_leibniz_term_and_a_jump() -> None:
    p = small(64)
    ws = _tilted(p)
    i, j = 5, 9  # on the rising flank of the sinusoid
    h1, h2 = _np(ws.h1), _np(ws.h2)
    zi = p.height * h1 / (h1 + h2)
    dzdx = (zi[j, i + 1] - zi[j, i - 1]) / (2 * p.dx)
    jj = min(1.0, zi[j, i] / p.z_sl) * 0.05 * dzdx  # J = phi(z_i) u1 dz_i/dx (u2 = 0)
    eps = 0.01
    below, above = zi[j, i] - eps, zi[j, i] + eps
    got = _np(W.sample_wind(ws, _cell_pos(p, i, j, [below, above, 80.0]), p).u)[:, 2]
    ref = _w_ref([below, above, 80.0], zi[j, i], 0.0, 0.0, jj, p)
    np.testing.assert_allclose(got, ref, atol=1e-6)
    assert got[0] - got[1] == pytest.approx(
        jj - 2 * eps * jj / p.height, abs=1e-6
    )  # the jump is -J (+ the slope)
    assert abs(jj) > 1e-3 and abs(got[2]) <= 1e-7


def test_sampler_clips_are_applied_and_counted() -> None:
    p = replace(small(64), u_max=0.3, w_edge=0.004)
    n = p.n
    full = lambda x: mx.full((n, n), x, dtype=mx.float32)  # noqa: E731
    ws, _ = rest(p, u1=full(0.4), v1=full(0.3))
    out = W.sample_wind(ws, _pos([40.0], [40.0], [10.0]), p)
    np.testing.assert_allclose(_np(out.u)[0, :2], [0.24, 0.18], atol=1e-6)  # norm 0.5 -> 0.3, direction kept
    assert int(out.n_clip_h.item()) == 1
    # w above w_edge: clipped in field layers 0 and 19, not in layer 10
    x_u = (np.arange(n) + 1.0) * p.dx
    u2 = mx.array(np.tile(0.3 * np.sin(TWO_PI * x_u / p.L), (n, 1)), dtype=mx.float32)
    ws, _ = rest(p, u2=u2)
    i, j = 10, 10
    d2 = (_np(ws.u2)[j, i] - _np(ws.u2)[j, i - 1]) / p.dx
    zs = np.array([2.0, 78.0, 40.0])
    ref = _w_ref(zs, 32.0, 0.0, d2, 0.0, p)
    out = W.sample_wind(ws, _cell_pos(p, i, j, zs), p)
    w = _np(out.u)[:, 2]
    assert abs(ref[0]) > p.w_edge and abs(ref[1]) > p.w_edge and abs(ref[2]) > p.w_edge
    np.testing.assert_allclose(w[:2], np.sign(ref[:2]) * p.w_edge, atol=1e-7)
    assert w[2] == pytest.approx(ref[2], abs=1e-6)
    assert int(out.n_clip_w.item()) == 2


def test_precomputed_wind_fields_give_the_same_wind() -> None:
    p = small(32)
    ws = rand_state(p, seed=12, u=0.05, dh=3.0)
    pos = _pos([3.3, 40.1, 63.9], [10.0, 33.3, 1.0], [5.0, 33.0, 70.0])
    wf = W.wind_fields(ws, p)
    a, b = W.sample_wind(ws, pos, p), W.sample_wind(ws, pos, p, fields=wf)
    same(a.u, b.u)
    same(W.wind_layers(ws, p), W.wind_layers(ws, p, fields=wf))


def test_wind_layers_agree_with_the_point_sampler_at_cell_centres() -> None:
    p = small(32)
    ws = rand_state(p, seed=6, u=0.05, dh=3.0)
    lay = _np(W.wind_layers(ws, p))
    assert lay.shape == (20, 3, 32, 32)
    i, j = 7, 12
    zs = (np.arange(20) + 0.5) * 4.0
    pt = _np(W.sample_wind(ws, _cell_pos(p, i, j, zs), p).u)
    np.testing.assert_allclose(lay[:, :, j, i], pt, atol=1e-6)


def test_steady_tilted_interface_is_a_material_surface() -> None:
    """Uniform layer-1 flux through a tilted interface: w above it vanishes and just below it is u1·dz_i/dx
    (kinematic condition). The formula without the Leibniz term J fails both."""
    p = replace(small(64), z_sl=1e-3)
    n = p.n
    x_c = (np.arange(n) + 0.5) * p.dx
    h1 = p.H1 + 8.0 * np.sin(TWO_PI * x_c / p.L)
    h1_u = 0.5 * (h1 + np.roll(h1, -1))
    u1 = np.tile(1.6 / h1_u, (n, 1))  # F = 1.6: |u1| about 0.05
    ws, _ = rest(p)
    f32 = lambda a: mx.array(a, dtype=mx.float32)  # noqa: E731
    ws = ws._replace(h1=f32(np.tile(h1, (n, 1))), h2=f32(np.tile(p.height - h1, (n, 1))), u1=f32(u1))
    ii = np.arange(2, n - 2)
    j = 10
    h1f, h2f = _np(ws.h1), _np(ws.h2)
    zi = p.height * h1f / (h1f + h2f)
    dzdx = (np.roll(zi, -1, 1) - np.roll(zi, 1, 1)) / (2 * p.dx)
    u1c = 0.5 * (_np(ws.u1) + np.roll(_np(ws.u1), 1, 1))
    jmax = np.abs(u1c * dzdx).max()
    below = np.array([_np(W.sample_wind(ws, _cell_pos(p, i, j, [zi[j, i] - 0.01]), p).u)[0, 2] for i in ii])
    above = np.array([_np(W.sample_wind(ws, _cell_pos(p, i, j, [zi[j, i] + 0.01]), p).u)[0, 2] for i in ii])
    target = (u1c * dzdx)[j, ii]
    assert np.abs(above).max() <= 0.02 * jmax
    big = np.abs(target) > 0.2 * jmax
    np.testing.assert_allclose(below[big], target[big], rtol=0.02)
    # without J (the review's probe): w is wrong above and below the interface
    d1 = (_np(ws.u1) - np.roll(_np(ws.u1), 1, 1)) / p.dx
    w_no_j_above = _w_ref(zi[j, ii] + 0.01, zi[j, ii], d1[j, ii], 0.0, 0.0, p)
    assert np.abs(w_no_j_above).max() > 0.2 * jmax


def test_sampler_is_seamless_across_the_periodic_edges() -> None:
    p = small(32)
    n = p.n
    r = np.random.default_rng(3)
    u1 = r.uniform(-0.05, 0.05, (n, n))
    v1 = r.uniform(-0.05, 0.05, (n, n))
    ws, _ = rest(p, u1=mx.array(u1, dtype=mx.float32), v1=mx.array(v1, dtype=mx.float32))
    eps = 1e-3
    xs = np.array([eps, p.L - eps, eps, p.L - eps, 17.3, eps, p.L - eps, 40.0])
    ys = np.array([eps, eps, p.L - eps, p.L - eps, eps, 23.7, 30.1, p.L - eps])
    got = _np(W.sample_wind(ws, _pos(xs, ys, np.full(8, 10.0)), p).u)[:, :2]  # z = 10: layer 1, phi = 1
    u1f, v1f = _np(ws.u1), _np(ws.v1)

    def bilinear(a, gx, gy):
        fx, fy = np.floor(gx), np.floor(gy)
        ax, ay = gx - fx, gy - fy
        i0, j0 = int(fx) % n, int(fy) % n
        i1, j1 = (i0 + 1) % n, (j0 + 1) % n
        return (1 - ay) * ((1 - ax) * a[j0, i0] + ax * a[j0, i1]) + ay * (
            (1 - ax) * a[j1, i0] + ax * a[j1, i1]
        )

    ref = np.array(
        [
            [bilinear(u1f, x / p.dx - 1.0, y / p.dx - 0.5), bilinear(v1f, x / p.dx - 0.5, y / p.dx - 1.0)]
            for x, y in zip(xs, ys, strict=True)
        ]
    )
    np.testing.assert_allclose(got, ref, atol=1e-6)


# --- 16, 17, 17b, 18. world rules, pools, the moist-thickness limiter, purity -----------------------------


def test_check_world_on_64k3d() -> None:
    cfg = load_config(preset="64k3d")
    assert W.check_world(P0, cfg) == []
    assert any("rule 16" in m for m in W.check_world(replace(P0, w_edge=0.035), cfg))
    assert any("rule 16" in m for m in W.check_world(replace(P0, u_max=0.41), cfg))
    assert any("rule 15" in m for m in W.check_world(replace(P0, n=100), cfg))
    assert any("rule 15" in m for m in W.check_world(replace(P0, z_sl=5.0), cfg))
    assert any("rule 15" in m for m in W.check_world(P0, load_config(preset="tiny")))  # a 2D world
    assert W.alpha(mx.array([0.0, 1.0]), P0).tolist() == pytest.approx([0.25, 1.0])


def test_warm_pools() -> None:
    p = P0
    t = _np(W.warm_pools(p))
    assert t.shape == (160, 160) and t.min() >= 0.0 and t.max() <= 1.0
    for x, y, *_ in p.pools:  # the maximum of each pool is within a cell of its centre
        j, i = round(y / p.dx - 0.5), round(x / p.dx - 0.5)
        win = t[np.ix_([(j + d) % 160 for d in range(-10, 11)], [(i + d) % 160 for d in range(-10, 11)])]
        jj, ii = np.unravel_index(win.argmax(), win.shape)
        assert abs(jj - 10) <= 1 and abs(ii - 10) <= 1
        assert win.max() == pytest.approx(
            1.0, abs=5e-3
        )  # a centre is up to a cell from the nearest cell centre
    # the pool at x = 0 is symmetric across the seam (cell centres x = 1 and x = 319)
    np.testing.assert_allclose(t[:, 0], t[:, -1], rtol=1e-6, atol=1e-9)
    band = _np(W.warm_pools(replace(p, pools=((0.0, 48.0, 0.0, 16.0, 1.0, 0.0, 0.0),))))
    assert np.array_equal(band, np.tile(band[:, :1], (1, 160))) and band.max() > 0.99
    drift = replace(p, pools=((316.0, 48.0, 32.0, 16.0, 1.0, 0.5, 0.0),))
    moved = _np(W.warm_pools(drift, mx.array(20, dtype=mx.int32)))  # the centre is at x = 326 = 6
    assert abs((np.argmax(moved[24]) + 0.5) * p.dx - 6.0) <= p.dx
    assert np.array_equal(_np(W.warm_pools(p, mx.array(5, dtype=mx.int32))), t)  # static pools ignore t


def test_moist_thickness_limiter() -> None:
    # f0 = 0 keeps the uniform wind steady (rotation would make divergence, which thins h1 after the limiter)
    p = small(24, f0=0.0, q_pool=1.2, pools=((24.0, 24.0, 0.0, 1e4, 1.0, 0.0, 0.0),))  # T_s = 1 everywhere
    n = p.n
    ws, fo = rest(
        p, h1=mx.full((n, n), 20.0), h2=mx.full((n, n), 60.0), Q=mx.full((n, n), 0.5), u1=mx.full((n, n), 0.2)
    )
    fn = W.make_step(p)
    total = 0
    for k in keys(50):
        ws, d = fn(ws, fo, k)
        mx.eval(ws, d)
        total += int(d.n_evap_limited.item())
    assert total > 0
    assert (_np(ws.h1) - p.gamma * _np(ws.Q)).min() >= p.m_min - 1e-5
    # the start set at rest (m0 = 8) never triggers it
    p2 = small(48, seed_rate=0.02)
    ws, fo = rest(p2)
    _, diags = run(W.make_step(p2), ws, fo, keys(100))
    assert sum(int(d.n_evap_limited.item()) for d in diags) == 0
    # gamma = 0 divides by nothing
    p3 = small(24, gamma=0.0)
    ws, fo = rest(p3, u1=mx.full((24, 24), 0.1))
    out, _ = run(W.make_step(p3), ws, fo, keys(10))
    assert all(np.isfinite(_np(getattr(out, f))).all() for f in out._fields)


def test_weather_is_pure_mlx_and_lint_scanned() -> None:
    src = Path(W.__file__).read_text(encoding="utf-8")
    imported = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "numpy" not in imported and "np" not in imported
    assert imported <= {"__future__", "functools", "math", "dataclasses", "typing", "mlx", "alife"}
    tree = ast.parse(src)
    assert not any(isinstance(nd, ast.BinOp) and isinstance(nd.op, ast.MatMult) for nd in ast.walk(tree))
    banned = {
        "matmul",
        "einsum",
        "tensordot",
        "addmm",
        "linalg",
        "item",
        "tolist",
        "eval",
        "async_eval",
        "at",
    }
    assert not [nd.attr for nd in ast.walk(tree) if isinstance(nd, ast.Attribute) and nd.attr in banned]
    from test_lint import SIM_MODULES  # the lint scans weather under rule 5

    assert "weather" in SIM_MODULES
    # nothing in the engine imports weather (W0 is offline): only analysis/w0.py may
    root = Path(W.__file__).parent
    for path in root.rglob("*.py"):
        if path.name in ("weather.py", "w0.py", "w0_plots.py"):
            continue
        assert not any(
            isinstance(nd, ast.ImportFrom | ast.Import) and "weather" in ast.unparse(nd)
            for nd in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        ), path
