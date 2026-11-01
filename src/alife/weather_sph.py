"""The sphere weather (device, pure MLX; DESIGN §46.6, true-sphere series S4b).

The two stacked shallow-water layers with moisture of ``weather.py`` on the HEALPix grid of Nside ``Nw``
(``N = 12·Nw²`` equal-area cells, collocated), after the S4a float64 prototype (``docs/SPHERE.md`` 4.1, report
``s4a/report.md``). Per cell: the layer thicknesses ``h1, h2``, the layers' velocities as **tangent
3-vectors** in the planet frame, the lower-layer humidity ``Q``, and the rain rate ``P`` and cloud depth
``Dc`` of the last step (``N_WX = 11`` planes of ``State.wx``).

One step (single formulas, ``R = Rp``, unit-sphere tables times ``R``)::

    S1  semi-Lagrangian momentum on the fan interpolation (8 triangles around the cell, the one with the
        largest minimum barycentric coordinate), re-projected on the tangent plane
    S2  flux-form continuity: gradient-corrected face values of ``u`` and ``h`` at the true edge midpoints,
        the Rhie-Chow compact pressure term in the face normal velocity, the box's positivity limiter on the
        outflow of each cell, ``Q`` advected with the same fluxes (upwind ``q = Q/h1``)
    S3  half Coriolis rotation about the local up (``f = f0·z``, exact), pressure kick with the new ``h``,
        half rotation, re-projection
    S4  biharmonic filter ``a4`` on ``h`` (before the clamp) and on ``u`` (``ENGINE_FIXED``)
    S5  evaporation, condensation with momentum-conserving mass transfer, radiative return, implicit
        friction, the safety tripwire, seeded vortices (pointwise, as the box)

Differences from the box: no divergence damping (``kappa_d`` is not carried; not needed by the S4a gates), the
biharmonic filter, the Courant number ``C = c_ext·dt/(0.81·√A)`` of the **mean cell side** ``√A`` (limit
0.655 ±
0.005 at ``a4 = 3e-3``, float64 and float32 alike; ``C_LIMIT`` = 0.650 is the last neutral value, measured
by the
slow tests of ``tests/test_weather_sph.py``), and the warm pools on the
sphere (arc-length widths at the pool latitude, masked to the hemisphere facing the centre). Pure MLX like the
simulation modules: no host syncs, no branching on array values, fixed shapes; gathers only (no scatter, no
matmul: F20).

Time (owner decision C29). One engine step advances the weather once, by ``dt`` = ``weather_dt(p)`` =
``SPHERE_COURANT·0.81·√A/c_ext`` units of weather time (``SPHERE_COURANT`` = 0.40: ``dt`` = 0.829 / 0.804 /
0.789 at the presets), derived and not configured. All rates of the model are per unit of weather time: the
step takes them as ``rate·dt``; the pools drift ``vx·dt`` per engine step, a seed fires with probability
``seed_rate·dt``, the rain handed to the resource field is ``rain_gain·P·dt``. The wind that moves agents is
the weather's velocity applied once per engine step, unscaled (a calibration of S4c).

The coupling to the engine (wind at the agents through the vertical map of ``WeatherGeometry``, rain,
seeding, the validation) is in the second half of this module.
"""

from __future__ import annotations

import dataclasses
import functools
import math
from typing import TYPE_CHECKING, Any, Final, NamedTuple

import mlx.core as mx
import numpy as np

from alife import fields_sph, geom, ops, sphere, weather, weather_tab
from alife.weather import (
    ENGINE_FIXED,
    AgentForce,
    AgentWind,
    RainIn,
    WeatherDiag,
    WeatherForcing,
    WeatherParams,
)

if TYPE_CHECKING:
    from alife.config import Config

_F32 = mx.float32
_I32 = mx.int32

N_WX: Final = 11  # h1, h2, u1 (3), u2 (3), Q, P, Dc
A4: Final = 3e-3  # the biharmonic filter strength on h and u (ENGINE_FIXED: the Courant limit needs it, S4a)
C_LIMIT: Final = (
    0.650  # float32 limit of the S4a noise test at a4 = 3e-3: least over Nw 16, 22, 42 (0.6547, 0.650)
)
# The Courant number the sphere weather runs at, at every Nw (owner decision C29, DESIGN §46.6): the weather
# time step per engine step is derived from it, ``dt_w = SPHERE_COURANT·0.81·√A/c_ext`` (0.829 / 0.804 / 0.789
# for Nw 16 / 22 / 42 at the presets' radii), so one engine step advances the weather by ``dt_w`` < 1 units of
# weather time. Not a config key (the box's ``ENGINE_FIXED["dt"]`` stays 1); a later change makes it one.
SPHERE_COURANT: Final = 0.40
E_SQRT: Final = math.sqrt(math.e)
PLANET_AXIS: Final = np.array(
    [0.0, 0.0, 1.0]
)  # the rotation axis of the planet (no tilt: docs/SPHERE.md 1.4)
POOL_LAT_MAX: Final = math.radians(60.0)  # pool centres stay within this latitude (docs/SPHERE.md 4.1)


class WeatherStateS(NamedTuple):
    """The prognostic fields on the sphere: thicknesses (N,), velocities as tangent 3-vectors (N, 3), the
    lower-layer humidity ``Q`` and the last step's rain rate ``P`` and cloud depth ``Dc`` (N,)."""

    h1: mx.array
    h2: mx.array
    u1: mx.array
    u2: mx.array
    Q: mx.array
    P: mx.array
    Dc: mx.array


def pack(ws: WeatherStateS) -> mx.array:
    """``State.wx``: ``(N_WX, N)`` float32 in the order h1, h2, u1 xyz, u2 xyz, Q, P, Dc."""
    return mx.concatenate(
        [
            ws.h1[None],
            ws.h2[None],
            ws.u1.transpose(1, 0),
            ws.u2.transpose(1, 0),
            ws.Q[None],
            ws.P[None],
            ws.Dc[None],
        ]
    )


def unpack(wx: mx.array) -> WeatherStateS:
    """The inverse of ``pack``."""
    return WeatherStateS(wx[0], wx[1], wx[2:5].transpose(1, 0), wx[5:8].transpose(1, 0), wx[8], wx[9], wx[10])


# --- parameters ---------------------------------------------------------------------------------------------


def n_cells(nside: int) -> int:
    return 12 * nside * nside


def cell_side(radius: float, nw: int) -> float:
    """The mean cell side ``√A`` with ``A = 4π·R²/N`` (the unit of the Courant number and of ``dx``)."""
    return radius * math.sqrt(4.0 * math.pi / n_cells(nw))


def weather_dt(p: WeatherParams) -> float:
    """The weather time step of one engine step, ``SPHERE_COURANT·0.81·√A/c_ext`` (``p.dx`` is ``√A``):
    ``C_used = SPHERE_COURANT`` at every ``Nw``. Derived from the geometry and the layer constants, not
    configured. Every per-step quantity of the weather is a rate times this ``dt``: the dynamics, friction,
    relaxation and moisture take ``p.dt``; the pools drift ``vx·dt`` and the seeds fire with probability
    ``seed_rate·dt`` per engine step; the rain handed to the resource field is ``rain_gain·P·dt``."""
    return SPHERE_COURANT * 0.81 * p.dx / p.c_ext


@functools.cache
def _params(wc: Any, radius: float) -> WeatherParams:
    nw = wc.cells
    pools = tuple(tuple(wc.pools[i : i + 7]) for i in range(0, len(wc.pools), 7))
    from dataclasses import fields as dc_fields

    own = {f.name for f in dc_fields(WeatherParams)}
    kw = {f.name: getattr(wc, f.name) for f in dc_fields(wc) if f.name in own and f.name != "pools"}
    base = WeatherParams(
        n=nw,
        dx=cell_side(radius, nw),
        height=wc.H1 + wc.H2,
        pools=pools,
        **ENGINE_FIXED,
        **kw,
    )
    return dataclasses.replace(base, dt=weather_dt(base))


def params_from_config(cfg: Config) -> WeatherParams:
    """The ``WeatherParams`` of a sphere config: ``n = Nw`` (the weather Nside), ``dx = √A`` (the mean cell
    side), ``height = H1 + H2`` (the weather's own depth; the shell's is ``world.height``, see
    ``WeatherGeometry``), ``ENGINE_FIXED``. Cached on the weather section and the planet radius."""
    return _params(cfg.weather, cfg.world.radius)


def courant(p: WeatherParams, radius: float, dt: float | None = None) -> float:
    """``C = c_ext·dt/(0.81·√A)`` (docs/SPHERE.md 4.3; ``c_ext`` of the two-layer model, 0.900)."""
    return p.c_ext * (p.dt if dt is None else dt) / (0.81 * cell_side(radius, p.n))


# --- the device tables --------------------------------------------------------------------------------------


class Dev(NamedTuple):
    """The tables of one (Nw, R, f0, dt) as float32 / int32 device arrays (host float64, cast once)."""

    x: mx.array  # (N, 3)
    nb: mx.array  # (N, 4) int32
    revflat: mx.array  # (N·4,) int32: the flat slot [nb[p, k], rev[p, k]]
    ell_r: mx.array  # (N, 4): the edge length times R
    d_r: mx.array  # (N, 4): the centre distance times R
    n: mx.array  # (N, 4, 3)
    e: mx.array  # (N, 4, 3)
    en: mx.array  # (N, 4): e·n
    dx: mx.array  # (N, 4, 3) (unit sphere)
    bm: mx.array  # (N, 4, 8)
    nb8: mx.array  # (N, 8) int32
    cnt: mx.array  # (N,) int32
    w8: mx.array  # (N, 8, 3): unit sphere (per radian)
    w8r: mx.array  # (N, 8, 3): per unit length (divided by R)
    ex: mx.array  # (N, 3)
    ey: mx.array  # (N, 3)
    tinv: mx.array  # (N, 32): the 8 fan inverses, row-major 2x2
    wl: mx.array  # (N, 4)
    cos_r: mx.array  # (N,): cos(-f dt/2)
    sin_r: mx.array  # (N,)
    ar2: float  # the cell area times R² (a static constant)


@functools.cache
def device_tables(nside: int, radius: float, f0: float, dt: float) -> Dev:
    t = weather_tab.weather_tab(nside)
    n = t.npix
    ang = -f0 * (t.x * PLANET_AXIS).sum(axis=-1) * dt / 2.0  # f = f0·(x̂·axis), the half-step angle
    return Dev(
        x=mx.array(t.x, dtype=_F32),
        nb=mx.array(t.nb, dtype=_I32),
        revflat=mx.array((t.nb * 4 + t.rev).reshape(-1), dtype=_I32),
        ell_r=mx.array(t.ell * radius, dtype=_F32),
        d_r=mx.array(t.d * radius, dtype=_F32),
        n=mx.array(t.n, dtype=_F32),
        e=mx.array(t.e, dtype=_F32),
        en=mx.array((t.e * t.n).sum(-1), dtype=_F32),
        dx=mx.array(t.dx, dtype=_F32),
        bm=mx.array(t.bm, dtype=_F32),
        nb8=mx.array(t.nb8, dtype=_I32),
        cnt=mx.array(t.cnt, dtype=_I32),
        w8=mx.array(t.w8, dtype=_F32),
        w8r=mx.array(t.w8 / radius, dtype=_F32),
        ex=mx.array(t.ex, dtype=_F32),
        ey=mx.array(t.ey, dtype=_F32),
        tinv=mx.array(t.tinv.reshape(n, 32), dtype=_F32),
        wl=mx.array(t.wl, dtype=_F32),
        cos_r=mx.array(np.cos(ang), dtype=_F32),
        sin_r=mx.array(np.sin(ang), dtype=_F32),
        ar2=float(t.area * radius * radius),
    )


def dev_for(p: WeatherParams, radius: float) -> Dev:
    return device_tables(p.n, float(radius), float(p.f0), float(p.dt))


# --- small vector helpers -----------------------------------------------------------------------------------


def _dot(a: mx.array, b: mx.array) -> mx.array:
    return (a * b).sum(axis=-1)


def _proj(x: mx.array, u: mx.array) -> mx.array:
    """The tangent part of ``u`` at the unit vector ``x`` (broadcasting)."""
    return u - _dot(x, u)[..., None] * x


def _unit(v: mx.array) -> mx.array:
    return v / mx.maximum(mx.sqrt(_dot(v, v)), 1e-30)[..., None]


# --- operators ----------------------------------------------------------------------------------------------


def _grad(f: mx.array, tab: Dev, per_length: bool) -> mx.array:
    """Least-squares tangent gradient of scalars ``f`` (K, N) over the 8 touching neighbours: (K, N, 3), per
    radian, or per unit length (divided by R) with ``per_length``."""
    df = mx.take(f, tab.nb8, axis=1) - f[:, :, None]  # (K, N, 8)
    w = tab.w8r if per_length else tab.w8
    return (df[..., None] * w[None]).sum(axis=2)


def _lap4(f: mx.array, tab: Dev) -> mx.array:
    """The 4-neighbour Laplacian with the symmetric weights ``ell/d`` (mean 1): scalars (K, N) or vectors
    (K, N, 3)."""
    g = mx.take(f, tab.nb, axis=1)  # (K, N, 4) or (K, N, 4, 3)
    if f.ndim == 3:
        return (tab.wl[None, :, :, None] * (g - f[:, :, None, :])).sum(axis=2)
    return (tab.wl[None] * (g - f[:, :, None])).sum(axis=2)


def _bih(f: mx.array, tab: Dev) -> mx.array:
    """The biharmonic operator ``-L²`` (conserves the sum of equal-area scalars)."""
    return -_lap4(_lap4(f, tab), tab)


def _rot(x: mx.array, u: mx.array, cos_r: mx.array, sin_r: mx.array) -> mx.array:
    """Rotate the tangent vectors ``u`` about the local up ``x`` by the angle of the tables (exact)."""
    return cos_r[..., None] * u + sin_r[..., None] * ops.cross3(mx.broadcast_to(x, u.shape), u)


def _fan(
    xd: mx.array, tab: Dev, nw: int
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
    """The fan interpolation of departure points ``xd`` (K, N, 3), unit vectors: ``(c, j1, j2, w0, l1, l2)``
    with the cell ``c`` of the point, the two further vertices of the winning triangle and the weights."""
    k, n = xd.shape[0], xd.shape[1]
    c = sphere.ang2pix(xd, nw)  # (K, N)
    cf = c.reshape(-1)
    xc = mx.take(tab.x, cf, axis=0).reshape(k, n, 3)
    dv = _proj(xc, xd - xc)
    p2 = mx.stack(
        [
            _dot(dv, mx.take(tab.ex, cf, axis=0).reshape(k, n, 3)),
            _dot(dv, mx.take(tab.ey, cf, axis=0).reshape(k, n, 3)),
        ],
        axis=-1,
    )  # (K, N, 2)
    ti = mx.take(tab.tinv, cf, axis=0).reshape(k, n, 8, 2, 2)
    lam = (ti * p2[:, :, None, None, :]).sum(axis=-1)  # (K, N, 8, 2): the barycentric coordinates
    w0 = 1.0 - lam[..., 0] - lam[..., 1]
    score = mx.minimum(mx.minimum(lam[..., 0], lam[..., 1]), w0)
    kb = mx.argmax(score, axis=-1)[..., None]  # (K, N, 1)
    l1 = mx.take_along_axis(lam[..., 0], kb, axis=-1)[..., 0]
    l2 = mx.take_along_axis(lam[..., 1], kb, axis=-1)[..., 0]
    nb8c = mx.take(tab.nb8, cf, axis=0).reshape(k, n, 8)
    cntc = mx.take(tab.cnt, cf, axis=0).reshape(k, n)
    k1 = kb[..., 0]
    k2 = mx.where(k1 + 1 < cntc, k1 + 1, 0)
    j1 = mx.take_along_axis(nb8c, kb, axis=-1)[..., 0]
    j2 = mx.take_along_axis(nb8c, k2[..., None], axis=-1)[..., 0]
    return c, j1, j2, 1.0 - l1 - l2, l1, l2


def _apply_fan(u: mx.array, fw: tuple[mx.array, ...]) -> mx.array:
    """``u`` (K, N, 3) at the departure points of ``fw``."""
    c, j1, j2, w0, l1, l2 = fw
    k, n = u.shape[0], u.shape[1]
    flat = u.reshape(k * n, 3)
    off = (mx.arange(k, dtype=_I32) * n).reshape(k, 1)

    def at(idx: mx.array) -> mx.array:
        return mx.take(flat, (idx + off).reshape(-1), axis=0).reshape(k, n, 3)

    return w0[..., None] * at(c) + l1[..., None] * at(j1) + l2[..., None] * at(j2)


# --- the warm pools and the seeds ---------------------------------------------------------------------------


def warm_pools(p: WeatherParams, radius: float, x: mx.array, t: mx.array | None = None) -> mx.array:
    """The warm-pool map ``T_s`` at the cell centres ``x`` (N, 3), float32 in [0, 1].

    A pool ``(x, y, sx, sy, amp, vx, vy)`` has its centre at the arc lengths ``x`` east of longitude 0 and
    ``y`` north of the equator (``lon = x/R``, ``lat = y/R``), drifts in longitude at its own latitude
    (``vx`` arc length per unit of weather time, so ``vx·dt`` per engine step; ``vy = 0`` required), and is
    Gaussian in the east and north components of the chord vector to the cell centre (widths ``sx``, ``sy`` in
    length; ``sx <= 0``: a zonal band). The Gaussian is masked to the hemisphere facing the centre (``x·c >
    0``): the chord's tangent components are not monotone in the angle and would light the antipode (S4a).
    ``t`` is an int32 () array (a Python int
    would be a compile constant); with every drift 0 the map ignores it."""
    total = mx.zeros((x.shape[0],), dtype=_F32)
    drift = any(row[5] != 0.0 for row in p.pools)
    tf = t.astype(_F32) if (drift and t is not None) else None
    for px, py, sx, sy, amp, vx, vy in p.pools:
        if vy != 0.0:
            raise ValueError("a pool with vy != 0 leaves the validated band of the sphere weather")
        lat = py / radius
        lon = px / radius
        clat, slat = math.cos(lat), math.sin(lat)
        if tf is None or vx == 0.0:  # static: a fixed pool
            cl, sl = math.cos(lon), math.sin(lon)
            cen = mx.array([clat * cl, clat * sl, slat], dtype=_F32)
            east = mx.array([-sl, cl, 0.0], dtype=_F32)
            north = mx.array([-slat * cl, -slat * sl, clat], dtype=_F32)
        else:
            lon_t = lon + (vx * p.dt / (radius * clat)) * tf
            cl, sl = mx.cos(lon_t), mx.sin(lon_t)
            zero = lon_t * 0.0
            cen = mx.stack([clat * cl, clat * sl, zero + slat])
            east = mx.stack([-sl, cl, zero])
            north = mx.stack([-slat * cl, -slat * sl, zero + clat])
        ch = radius * (x - cen)
        de, dn = _dot(ch, east), _dot(ch, north)
        e = -(dn * dn) / (2.0 * sy * sy)
        if sx > 0:
            e = e - (de * de) / (2.0 * sx * sx)
        total = total + amp * mx.exp(e) * (_dot(x, cen) > 0).astype(_F32)
    return mx.clip(total, 0.0, 1.0)


def default_forcing(p: WeatherParams, radius: float, t: mx.array | None = None) -> WeatherForcing:
    """``T_s`` from the pools and ``qs`` = 1."""
    tab = dev_for(p, radius)
    return WeatherForcing(warm_pools(p, radius, tab.x, t), mx.ones((tab.x.shape[0],), dtype=_F32))


def init_state(p: WeatherParams, fo: WeatherForcing) -> WeatherStateS:
    """The rest state: ``h = (H1, H2)``, ``u = 0``, ``Q = Q_e(T_s)`` capped at ``Q_s``, ``P = Dc = 0``."""
    assert fo.T_s.dtype == _F32 and fo.qs.dtype == _F32, "weather forcing must be float32"
    n = fo.T_s.shape[0]
    zero = mx.zeros((n,), dtype=_F32)
    qe = p.Qs * (p.q_bg + (p.q_pool - p.q_bg) * fo.T_s)
    return WeatherStateS(
        h1=mx.full((n,), p.H1, dtype=_F32),
        h2=mx.full((n,), p.H2, dtype=_F32),
        u1=mx.zeros((n, 3), dtype=_F32),
        u2=mx.zeros((n, 3), dtype=_F32),
        Q=mx.minimum(qe, p.Qs),
        P=zero,
        Dc=zero,
    )


def rest_array(cfg: Config, step: mx.array | int = 0) -> mx.array:
    """The rest ``State.wx`` of a sphere config under the pools of step ``step`` (see
    ``weather.rest_array``)."""
    p = params_from_config(cfg)
    t = (step if isinstance(step, mx.array) else mx.array(step)).astype(_I32)
    return pack(init_state(p, default_forcing(p, cfg.world.radius, t)))


def _seed_vortex(x: mx.array, key: mx.array, p: WeatherParams, radius: float) -> tuple[mx.array, mx.array]:
    """The tangent velocity ``u = û × ∇ψ`` of this step's disturbance, (N, 3), and ``on`` () float32.

    ``ψ = amp·exp(-|R(x - x_s)|²/2σ²)`` about a random point ``x_s`` (equal-area draw), cyclonic for the
    hemisphere of the point (``amp = -sign(z_s)·on·a·seed_v·σ·√e``); zero when not ``on``."""
    k_on, k_x, k_y, k_a = mx.random.split(key, 4)
    on = (mx.random.uniform(key=k_on) < p.seed_rate * p.dt).astype(_F32)  # a rate per unit of weather time
    xs = fields_sph.sphere_point(mx.random.uniform(key=k_x), mx.random.uniform(key=k_y))  # (3,)
    a = 1.0 + p.seed_jitter * (2.0 * mx.random.uniform(key=k_a) - 1.0)
    sgn = mx.where(xs[2] >= 0, 1.0, -1.0)
    chord = radius * (x - xs)
    r2 = _dot(chord, chord)
    amp = -sgn * on * a * p.seed_v * p.seed_sigma * E_SQRT
    psi = amp * mx.exp(-r2 / (2.0 * p.seed_sigma**2))
    gradpsi = -(psi / p.seed_sigma**2)[:, None] * _proj(x, chord)
    return ops.cross3(x, gradpsi), on


# --- the step -----------------------------------------------------------------------------------------------


def step(
    ws: WeatherStateS,
    fo: WeatherForcing,
    key: mx.array,
    p: WeatherParams,
    radius: float,
    *,
    debug: bool = False,
    advect: bool = True,
    seeds: bool = True,
    a4: float = A4,
    dry: bool = False,
    kappa_d: float = 0.0,
) -> tuple[WeatherStateS, WeatherDiag]:
    """One weather step (S1 to S8 of the module docstring). ``key`` is a (2,) MLX key used only by the seeds;
    the step returns no key. The keywords are static test hooks (the engine passes none): ``advect=False``
    switches the momentum advection off (the linear stability test), ``seeds=False`` the disturbances, ``a4``
    sets the filter strength (the engine uses ``A4``), ``dry=True`` stops after the dynamics (S1 to S3 and the
    filter: no moisture, relaxation, friction or seeds), which is what the S4a noise and balance tests ran.
    ``kappa_d`` > 0 is an experiment, not adopted (docs: ``s4b_build/m1_options.md``): the box's divergence
    damping ``u += kappa_d·dx²·∇D`` (``D`` the divergence per length, ``dx`` the mean cell side) after the
    filter."""
    tab = dev_for(p, radius)
    nw = p.n
    dt = p.dt
    n = tab.x.shape[0]
    zero = ws.h1[0] * 0.0  # a zero derived from the inputs (never return a constant array from compile)
    zero_i = zero.astype(_I32)
    x = tab.x
    h = mx.stack([ws.h1, ws.h2])  # (2, N)
    u = mx.stack([ws.u1, ws.u2])  # (2, N, 3)
    big_q = ws.Q

    # S1: semi-Lagrangian momentum advection (Euler backtrace with the winds at n)
    if advect:
        xd = _unit(x[None] - (dt / radius) * u)
        ua = _proj(x, _apply_fan(u, _fan(xd, tab, nw)))
    else:
        ua = _proj(x, u)

    # S2: continuity with the old velocities; Rhie-Chow with the old potential
    m2 = p.g_top * (h[0] + h[1])
    m0 = mx.stack([m2 + p.g_prime * h[0], m2])
    gm = _grad(m0, tab, True)  # (2, N, 3) per length
    gh = _grad(h, tab, False)  # per radian
    du = mx.take(u, tab.nb8, axis=1) - u[:, :, None, :]  # (2, N, 8, 3)
    # t[k, p, m] = n[p, m] . J[k, p] . dx[p, m] with J = sum_k' du[k'] (x) w8[k']: contracted through bm
    nd = (tab.n[None, :, :, None, :] * du[:, :, None, :, :]).sum(axis=-1)  # (2, N, 4, 8)
    t = (tab.bm[None] * nd).sum(axis=-1)  # (2, N, 4)
    tq = mx.take(t.reshape(2, n * 4), tab.revflat, axis=1).reshape(2, n, 4)
    uq = mx.take(u, tab.nb, axis=1)  # (2, N, 4, 3)
    un = 0.5 * (_dot(u[:, :, None, :], tab.n[None]) + _dot(uq, tab.n[None])) + 0.5 * (t - tq)
    mq = mx.take(m0, tab.nb, axis=1)  # (2, N, 4)
    gmq = mx.take(gm, tab.nb, axis=1)  # (2, N, 4, 3)
    comp = (mq - m0[:, :, None]) / tab.d_r[None] - 0.5 * (
        _dot(gm[:, :, None, :], tab.e[None]) + _dot(gmq, tab.e[None])
    )
    un = un - dt * comp * tab.en[None]
    hq = mx.take(h, tab.nb, axis=1)
    ghq = mx.take(gh, tab.nb, axis=1)
    hf = 0.5 * (h[:, :, None] + hq) + 0.5 * (_dot(gh[:, :, None, :], tab.dx[None]) + _dot(ghq, tab.dx[None]))
    flux = tab.ell_r[None] * hf * un  # outward flux per unit time
    out = mx.maximum(flux, 0.0).sum(axis=-1)
    avail = mx.maximum(h - p.h_min, 0.0) * tab.ar2
    over = dt * out > avail
    lam = mx.where(over, avail / (dt * mx.maximum(out, 1e-30)), 1.0)
    lamq = mx.take(lam, tab.nb, axis=1)
    flux = flux * mx.where(flux > 0, lam[:, :, None], lamq)
    hn = h - dt * flux.sum(axis=-1) / tab.ar2
    q = big_q / h[0]
    qq = mx.take(q, tab.nb, axis=0)  # (N, 4)
    qm = mx.where(flux[0] > 0, q[:, None], qq)
    qn = big_q - dt * (flux[0] * qm).sum(axis=-1) / tab.ar2
    n_limited = over.astype(_I32).sum()
    if a4:
        hn = hn + a4 * _bih(hn, tab)
    n_clamp = (hn < p.h_min).astype(_I32).sum() + (qn < 0.0).astype(_I32).sum()
    clamp_mass = mx.maximum(p.h_min - hn, 0.0).sum()
    q_clamp = mx.maximum(-qn, 0.0).sum()
    h = mx.maximum(hn, p.h_min)
    big_q = mx.maximum(qn, 0.0)

    # S3: half rotation, pressure kick with the new h, half rotation
    v = _rot(x, ua, tab.cos_r, tab.sin_r)
    m2n = p.g_top * (h[0] + h[1])
    mn = mx.stack([m2n + p.g_prime * h[0], m2n])
    v = v - dt * _grad(mn, tab, True)
    v = _rot(x, v, tab.cos_r, tab.sin_r)
    v = _proj(x, v)
    if a4:
        v = _proj(x, v + a4 * _bih(v, tab))
    ua = v
    if kappa_d:  # static, off in the engine: the box's divergence damping (the S4b stability options)
        div = (mx.take(ua, tab.nb8, axis=1) - ua[:, :, None, :]) * tab.w8r[None]
        ua = _proj(x, ua + (kappa_d * p.dx**2) * _grad(div.sum(axis=(2, 3)), tab, True))
    if dry:  # static: the dynamics alone
        zero_d = zero.astype(_I32)
        return WeatherStateS(h[0], h[1], ua[0], ua[1], big_q, ws.P, ws.Dc), WeatherDiag(
            n_limited,
            n_clamp,
            clamp_mass,
            q_clamp,
            zero_d,
            zero_d,
            zero_d,
            zero_d,
            zero,
            zero,
            zero,
            zero,
            zero,
            zero,
            zero,
            zero,
        )

    # S4: evaporation over the warm pools, with the moist-thickness limiter
    sp1 = mx.sqrt(_dot(ua[0], ua[0]))
    qe = p.Qs * (p.q_bg + (p.q_pool - p.q_bg) * fo.T_s)
    evap = p.k_E * (sp1 + p.u_gust) * mx.maximum(qe - big_q, 0.0)
    cap = mx.maximum(h[0] - p.gamma * big_q - p.m_min, 0.0) / (p.gamma * dt)
    n_evap = (evap > cap).astype(_I32).sum()
    evap = mx.minimum(evap, cap)
    big_q = big_q + dt * evap

    # S5: condensation and mass transfer to layer 2 (with its momentum)
    cond = mx.maximum(big_q - p.Qs * fo.qs, 0.0) / p.tau_c
    bound = (h[0] - p.h_min) / (p.gamma * dt)
    n_cond = (cond > bound).astype(_I32).sum()
    cond = mx.minimum(cond, bound)
    big_q = big_q - dt * cond
    mu = p.gamma * dt * cond
    h1n, h2n = h[0] - mu, h[1] + mu
    u2 = ua[1] + (mu / h2n)[:, None] * (ua[0] - ua[1])
    h = mx.stack([h1n, h2n])
    ua = mx.stack([ua[0], u2])

    # S6: radiative mass return
    n_relax = zero_i
    if p.tau_rad > 0:
        r = dt * (p.H1 - h[0]) / p.tau_rad
        lo, hi = -(h[0] - p.h_min), h[1] - p.h_min
        n_relax = ((r < lo) | (r > hi)).astype(_I32).sum()
        r = mx.minimum(mx.maximum(r, lo), hi)
        hn2 = mx.stack([h[0] + r, h[1] - r])
        rr = mx.stack([mx.maximum(r, 0.0), mx.maximum(-r, 0.0)])
        ua = ua + (rr / hn2)[..., None] * (ua[::-1] - ua)
        h = hn2

    # S7: damping (implicit friction: layer 1 linear + quadratic, layer 2 linear) and the tripwire
    sp = mx.sqrt(_dot(ua, ua))  # (2, N)
    speed = sp.max()
    cour = speed * (dt / p.dx)
    d12 = ua[0] - ua[1]
    hyper = (_dot(d12, d12) / (p.g_prime * (h[0] + h[1]))).max()
    n_safety = (sp > p.u_safe).astype(_I32).sum()
    lin = mx.array([p.r1, p.r2], dtype=_F32).reshape(2, 1)
    quad = mx.array([p.c_d, 0.0], dtype=_F32).reshape(2, 1)
    ua = ua / (1.0 + dt * (lin + quad * sp))[..., None]
    sp2 = mx.sqrt(_dot(ua, ua))
    ua = ua * (p.u_safe / mx.maximum(sp2, p.u_safe))[..., None]

    # S8: disturbances
    seed_on = zero
    if seeds and p.seed_rate > 0:
        dv, seed_on = _seed_vortex(x, key, p, radius)
        ua = mx.stack([ua[0] + dv, ua[1] + p.seed_w2 * dv]) if p.seed_w2 else mx.stack([ua[0] + dv, ua[1]])

    # S9: stored diagnostics
    zi = p.height * h[0] / (h[0] + h[1])
    cloud = (p.height - zi) * cond / (cond + p.c_half)
    new = WeatherStateS(h[0], h[1], ua[0], ua[1], big_q, cond, cloud)
    diag = WeatherDiag(
        n_limited=n_limited,
        n_clamp=n_clamp,
        clamp_mass=clamp_mass,
        q_clamp=q_clamp,
        n_cond_limited=n_cond,
        n_relax_limited=n_relax,
        n_evap_limited=n_evap,
        n_safety=n_safety,
        courant_max=cour,
        speed_max=speed,
        hyper_max=hyper,
        eps_local_max=(p.gamma * big_q / h[0]).max(),
        seed_on=seed_on,
        sum_E=evap.sum() * dt,
        sum_C=cond.sum() * dt,
        sum_Qseed=zero,
        E=evap if debug else None,
        C=cond if debug else None,
        Qseed=(big_q * 0.0) if debug else None,
    )
    return new, diag


def make_step(p: WeatherParams, radius: float, *, compile: bool = True, debug: bool = False, **kw: Any):
    """``fn(ws, fo, key) -> (ws', diag)`` closed over ``p``, ``radius`` and the test hooks ``kw``
    (``advect``, ``seeds``, ``a4``), compiled once."""

    def fn(ws: WeatherStateS, fo: WeatherForcing, key: mx.array) -> tuple[WeatherStateS, WeatherDiag]:
        return step(ws, fo, key, p, radius, debug=debug, **kw)

    return mx.compile(fn) if compile else fn


def advance(
    ws: WeatherStateS,
    t: mx.array,
    key: mx.array,
    cfg: Config,
    qs: mx.array | None = None,
    t_mult: mx.array | None = None,
) -> tuple[WeatherStateS, WeatherDiag]:
    """One weather step at engine step ``t`` (an int32 () array): the warm pools of step ``t`` (times the
    shade ``t_mult``, refused on a sphere before S4c), the seeding factor ``qs`` on ``Q_s``."""
    p = params_from_config(cfg)
    radius = cfg.world.radius
    tab = dev_for(p, radius)
    t_s = warm_pools(p, radius, tab.x, t)
    if t_mult is not None:  # static
        t_s = t_s * t_mult
    qs_ = qs if qs is not None else mx.ones((tab.x.shape[0],), dtype=_F32)
    return step(ws, WeatherForcing(t_s, qs_), key, p, radius)


# --- diagnostics of a state ---------------------------------------------------------------------------------


def divergence(ws: WeatherStateS, p: WeatherParams, radius: float) -> mx.array:
    """Horizontal divergence ``Σ_k W8_k·(u_k - u_c)`` of both layers at the centres, (2, N), per length."""
    tab = dev_for(p, radius)
    u = mx.stack([ws.u1, ws.u2])
    du = mx.take(u, tab.nb8, axis=1) - u[:, :, None, :]
    return (du * tab.w8r[None]).sum(axis=(2, 3))


def vorticity(ws: WeatherStateS, p: WeatherParams, radius: float) -> mx.array:
    """Layer-1 relative vorticity ``x̂·Σ_k W8_k × (u_k - u_c)`` (the normal component of the curl), (N,)."""
    tab = dev_for(p, radius)
    du = mx.take(ws.u1, tab.nb8, axis=0) - ws.u1[:, None, :]  # (N, 8, 3)
    cr = ops.cross3(mx.broadcast_to(tab.w8r, du.shape), du)
    return (cr * tab.x[:, None, :]).sum(axis=(1, 2))


def interface_height(ws: WeatherStateS, p: WeatherParams) -> mx.array:
    """``z_i = (H1 + H2)·h1/(h1 + h2)`` in weather units, (N,)."""
    return p.height * ws.h1 / (ws.h1 + ws.h2)


# --- the vertical map: the shell onto the atmosphere (docs/SPHERE.md 4.2) -----------------------------------


class GeoDev(NamedTuple):
    """The device constants of a ``WeatherGeometry``."""

    zl: mx.array  # (FZ, 1): the layers' lower interface heights
    zu: mx.array  # (FZ, 1): the upper ones
    children: mx.array  # (Npix_f · r²,) int32: weather cells of each field pixel, ascending
    parent: mx.array  # (Npix_w,) int32: the field pixel of each weather cell


class WeatherGeometry:
    """The shell-to-atmosphere map of one sphere world (host constants, device tables on demand).

    The weather layers are ``H1 + H2`` deep (80 at the confirmed set); the shell is ``H`` (16, 24, 40 for the
    presets): with ``s = (H1 + H2)/H`` the heights of the weather are ``z_w = s·z`` of the shell and the
    vertical wind is ``w_shell = w_w/s``. The cloud ``[zi, zi + dc]`` (weather units) occupies
    ``[zi/s, (zi + dc)/s]`` of the shell and is overlapped with the **actual** layer intervals
    ``[z_k, z_{k+1}]`` (non-uniform: equal volume), so every preset has rain food and a seeding band
    (docs/SPHERE.md 4.2; the first-draft rule gave tiny3s and mini3s none). The field pixels are the
    parents of
    the weather cells (``geom.parent_table``): the rain and the seeding act on the means over the ``r²``
    children of a pixel.
    """

    def __init__(self, radius: float, height: float, layers: int, nside_f: int, nw: int, h_w: float):
        if nw % nside_f:
            raise ValueError(f"the weather Nside {nw} must be a multiple of the field Nside {nside_f}")
        self.radius, self.height, self.layers = float(radius), float(height), int(layers)
        self.nside_f, self.nw, self.h_w = int(nside_f), int(nw), float(h_w)
        self.ratio = nw // nside_f
        self.s = self.h_w / self.height
        self.z_edges = geom.layer_edges(radius, height, layers)  # (FZ+1,) shell heights
        self.z_first = float(self.z_edges[1])
        self.z_last = float(self.z_edges[layers - 1])
        self._dev: GeoDev | None = None

    @classmethod
    def from_config(cls, cfg: Config) -> WeatherGeometry:
        g, wc = geom.geometry(cfg), cfg.weather
        return _geometry(g.radius, g.height_max, g.layers, g.nside, wc.cells, wc.H1 + wc.H2)

    @property
    def dev(self) -> GeoDev:
        if self._dev is None:
            edges = self.z_edges
            self._dev = GeoDev(
                zl=mx.array(edges[:-1].reshape(-1, 1), dtype=_F32),
                zu=mx.array(edges[1:].reshape(-1, 1), dtype=_F32),
                children=mx.array(geom.children_table(self.nside_f, self.ratio).reshape(-1), dtype=_I32),
                parent=mx.array(geom.parent_table(self.nw, self.ratio), dtype=_I32),
            )
        return self._dev

    def cloud_weights(self, zi: mx.array, dc: mx.array) -> mx.array:
        """The share of a column's cloud in each field layer, ``(FZ, ...)``: ``zi`` and ``dc`` are in weather
        units with the shape of the columns; the cloud's shell interval is ``[zi, zi + dc]/s``, ``frac_k`` the
        overlap over the layer thickness, weight ``frac_k/Σ frac`` (``weather.cloud_weights`` with the layer
        table in place of ``field_cell·k``)."""
        g = self.dev
        lo, hi = zi / self.s, (zi + dc) / self.s
        pad = (1,) * zi.ndim
        zl, zu = g.zl.reshape((-1, *pad)), g.zu.reshape((-1, *pad))
        frac = mx.clip((mx.minimum(hi, zu) - mx.maximum(lo, zl)) / (zu - zl), 0.0, 1.0)
        return frac / mx.maximum(frac.sum(axis=0), 1e-6)

    def rain_band(self) -> tuple[int, int]:
        """``(lo, hi)``: the field layers whose middle lies in the box's cloud band ``[32, 56]`` of the
        weather
        heights (``RAIN_BAND``: layers 8-13 at ``field_cell`` 4), mapped to the shell by ``/s``; at least one
        layer."""
        mid = 0.5 * (self.z_edges[:-1] + self.z_edges[1:])
        inside = np.flatnonzero((mid >= 32.0 / self.s) & (mid < 56.0 / self.s))
        if inside.size == 0:
            k = int(np.argmin(np.abs(mid - 44.0 / self.s)))
            return k, k + 1
        return int(inside[0]), int(inside[-1]) + 1

    def block_mean(self, a: mx.array) -> mx.array:
        """(Npix_w,) -> (Npix_f,): the mean over the ``r²`` weather children of each field pixel."""
        return mx.take(a, self.dev.children).reshape(-1, self.ratio * self.ratio).mean(axis=-1)

    def block_sum(self, a: mx.array) -> mx.array:
        return mx.take(a, self.dev.children).reshape(-1, self.ratio * self.ratio).sum(axis=-1)

    def spread(self, a: mx.array) -> mx.array:
        """(Npix_f,) -> (Npix_w,): each field pixel's value on its weather children."""
        return mx.take(a, self.dev.parent)


@functools.cache
def _geometry(
    radius: float, height: float, layers: int, nside_f: int, nw: int, h_w: float
) -> WeatherGeometry:
    return WeatherGeometry(radius, height, layers, nside_f, nw, h_w)


# --- the wind the organisms feel ----------------------------------------------------------------------------


class WindFieldsS(NamedTuple):
    """The per-cell fields of the wind formula, packed as the rows an agent gathers: ``rows (N, 10)`` =
    ``[zi, d1, d2, jump, u1 (3), u2 (3)]`` (the interface height in weather units, the layer divergences, the
    interface term, the two centre winds); one table, so a sample is one gather."""

    rows: mx.array  # (N, 10)


def wind_fields(ws: WeatherStateS, p: WeatherParams, radius: float) -> WindFieldsS:
    """Build the cell fields of the wind formula: the interface height ``zi`` (weather units), the layer
    divergences ``d1, d2`` (per length), the interface (Leibniz) term ``J = (φ(zi)·ū1 − ū2)·∇zi`` and the
    centre winds. Rolls nothing; gathers only, so it is safe to build once per step (never twice from one
    state in one compiled graph, as the box's)."""
    tab = dev_for(p, radius)
    u = mx.stack([ws.u1, ws.u2])
    du = mx.take(u, tab.nb8, axis=1) - u[:, :, None, :]
    div = (du * tab.w8r[None]).sum(axis=(2, 3))  # (2, N)
    zi = interface_height(ws, p)
    gz = _grad(zi[None], tab, True)[0]  # (N, 3)
    phi = mx.minimum(zi / p.z_sl, 1.0)
    jump = _dot(phi[:, None] * ws.u1 - ws.u2, gz)
    rows = mx.concatenate(
        [zi[:, None], div[0][:, None], div[1][:, None], jump[:, None], ws.u1, ws.u2], axis=-1
    )
    return WindFieldsS(rows)


def _cap_phi(x: mx.array, zs: float) -> mx.array:
    return mx.where(x <= zs, x * x / (2.0 * zs), x - 0.5 * zs)


def sample_wind(
    ws: WeatherStateS,
    pos: mx.array,
    p: WeatherParams,
    geo: WeatherGeometry,
    *,
    fields: WindFieldsS | None = None,
    valid: mx.array | None = None,
) -> AgentWind:
    """The wind at agent positions ``pos`` (M, 3), planet frame: pure gathers, any M (docs/SPHERE.md 4.1,
    4.2).

    The cell of the position (``sphere.ang2pix`` at the weather Nside) gives the layer fields; the agent's
    height ``z`` is mapped to the weather's ``z_w = s·z``; the formula of the box then gives the horizontal
    wind
    (``φ(z_w)·u1`` below the interface ``zi``, ``u2`` above; a tangent vector, **re-projected onto the agent's
    tangent plane**) and the vertical wind ``w_w = −[D1·Φ(min(z_w, zi)) + 1[z_w > zi]·(D2·(z_w − zi) + J) −
    D̄·z_w]`` with ``D̄ = [D1·Φ(zi) + D2·(H_w − zi) + J]/H_w`` (``w(0) = w(H) = 0``), clipped to ``|w_w| <=
    w_edge`` in the floor and lid field layers (``z < z_1`` and ``z >= z_{FZ-1}`` of the shell), and
    ``w = w_w/s`` along the local up. ``||u_h|| <= u_max`` is clipped with the direction kept. Both clips are
    counted over the ``valid`` rows (dead rows rest at stale positions). The result ``u`` is the Cartesian
    vector ``u_h + w·up`` (M, 3)."""
    wf = wind_fields(ws, p, geo.radius) if fields is None else fields
    nw = p.n
    cell = sphere.ang2pix(pos, nw)
    r = mx.take(wf.rows, cell, axis=0)  # (M, 10)
    zi, d1, d2, jump = (r[:, k] for k in range(4))
    u1, u2 = r[:, 4:7], r[:, 7:10]
    rho = mx.sqrt(_dot(pos, pos))
    up = pos / mx.maximum(rho, 1e-30)[:, None]
    z = rho - geo.radius
    zw = geo.s * z
    zs, hw = p.z_sl, p.height
    phi = mx.minimum(zw / zs, 1.0)
    below = zw < zi
    uh = mx.where(below[:, None], phi[:, None] * u1, u2)
    uh = _proj(up, uh)
    dbar = (d1 * _cap_phi(zi, zs) + d2 * (hw - zi) + jump) / hw
    above = mx.where(zw > zi, d2 * (zw - zi) + jump, 0.0)
    w = -(d1 * _cap_phi(mx.minimum(zw, zi), zs) + above - dbar * zw)
    edge = (z < geo.z_first) | (z >= geo.z_last)
    sp = mx.sqrt(_dot(uh, uh))
    over = sp > p.u_max
    w_over = edge & (mx.abs(w) > p.w_edge)
    if valid is not None:  # static
        over, w_over = over & valid, w_over & valid
    w = mx.where(edge, mx.clip(w, -p.w_edge, p.w_edge), w)
    uh = uh * (p.u_max / mx.maximum(sp, p.u_max))[:, None]
    u = uh + (w / geo.s)[:, None] * up
    return AgentWind(u, over.astype(_I32).sum(), w_over.astype(_I32).sum())


def agent_wind(
    ws: WeatherStateS, pos: mx.array, buoy_cap: mx.array, alive: mx.array, cfg: Config
) -> AgentForce:
    """The wind force on every row at ``pos`` (start-of-step positions) from the stored sphere weather: the
    coupling ``alpha0 + alpha1·buoy_cap`` times ``drag`` times the wind of ``sample_wind`` (Cartesian, planet
    frame, in length per unit of weather time) times ``dt``, the weather time of one engine step
    (``weather_dt``; ``physics.integrate_s`` divides by the same ``drag``, so a free agent drifts at
    ``alpha·dt·u_air`` per step, ``alpha`` times the displacement of the air parcel: DESIGN §46.6, S4c).
    ``wind_scale`` multiplies the force; the hazard speed is ``|u_air|``, the air's own, not scaled (the eddy
    controls are refused on a sphere by ``validate_config``)."""
    from alife.weather import alpha

    p = params_from_config(cfg)
    wc = cfg.weather
    geo = WeatherGeometry.from_config(cfg)
    aw = sample_wind(ws, pos, p, geo, valid=alive)
    a = alpha(buoy_cap, p)[:, None]
    force = (a * cfg.physics.drag * p.dt) * aw.u  # S4c: the wind is per unit of weather time, a step is dt
    if wc.wind_scale != 1.0:  # static
        force = wc.wind_scale * force
    speed = mx.sqrt(_dot(aw.u, aw.u))
    breaks = speed > wc.storm_break_u if wc.storm_break_u > 0 else None
    return AgentForce(force, speed, breaks, (aw.n_clip_h + aw.n_clip_w) > 0)


def air_speeds(
    ws: WeatherStateS, pos: mx.array, buoy_cap: mx.array, alive: mx.array, cfg: Config
) -> tuple[mx.array, mx.array]:
    """``(total, eddy)`` air speeds of every row. The sphere has no zonal mean in S4b: the eddy speed is
    reported equal to the total (the eddy controls are refused), for the metrics rows."""
    p = params_from_config(cfg)
    aw = sample_wind(ws, pos, p, WeatherGeometry.from_config(cfg), valid=alive)
    sp = mx.sqrt(_dot(aw.u, aw.u))
    return sp, sp


def agent_drift(
    wx: mx.array,
    pos: mx.array,
    buoy_cap: mx.array,
    alive: mx.array,
    cfg: Config,
    shelter: mx.array | None = None,
) -> mx.array:
    """The velocity the wind alone would give a free agent, (N, 3): ``agent_wind(...).force / drag``."""
    force = agent_wind(unpack(wx), pos, buoy_cap, alive, cfg).force
    if shelter is not None:  # static: structures on
        force = force * shelter[:, None]
    return force / cfg.physics.drag


# --- rain-fed regrowth and rain seeding (the field pixels are the parents of the weather cells) -------------


def rain_regen(rain: RainIn, cfg: Config) -> mx.array:
    """The rain as integer regeneration quanta per field cell, ``(FZ, Npix_f)`` int32 (the sphere's
    ``weather.rain_regen``): ``q = rain_gain·dt·mean(P)`` over the ``r²`` weather children of a pixel (``P``
    is a rate per unit of weather time, one engine step is ``dt`` of it); the cloud
    ``[zi, zi + dc]`` (rain-weighted means over the children, weather units) is mapped onto the shell
    layers by
    ``WeatherGeometry.cloud_weights``; layer ``k`` takes ``floor(q·share_k)``, so a column's food is
    ``rain_gain·dt·P`` whatever the cloud depth (less the floors: about half a quantum per used layer);
    ``rain_floor·q`` more goes to layer 0."""
    wc = cfg.weather
    p = params_from_config(cfg)
    geo = WeatherGeometry.from_config(cfg)
    ps = geo.block_sum(rain.P)
    q = wc.rain_gain * p.dt * geo.block_mean(rain.P)  # P is a rate per unit of weather time
    den = mx.maximum(ps, 1e-30)
    zi = geo.block_sum(rain.P * rain.zi) / den
    dc = geo.block_sum(rain.P * rain.dc) / den
    out = ops.floor_i32(q * geo.cloud_weights(zi, dc))
    if wc.rain_floor > 0:  # static
        floor_q = ops.floor_i32(wc.rain_floor * q)
        out = out + mx.concatenate([floor_q[None], mx.zeros((geo.layers - 1, floor_q.shape[0]), dtype=_I32)])
    return out


# The relative dead band of the seeding excess (``weather._SEED_DEADBAND``): above the float32 rounding of
# a mean.
_SEED_DEADBAND: Final = 1e-4


def seeding_response(chem: mx.array, ws: WeatherStateS, cfg: Config) -> mx.array:
    """The saturating seeding response ``r`` on the field pixels, ``(Npix_f,)`` float32 in ``[0, 1)``: the
    signal is the c2 averaged over the cloud layers, ``b = Σ_k w_k·chem[2, k]``, at the interface height
    ``zi``
    (mean over the pixel's weather children) and the cloud depth ``max(mean Dc, seed_dmin)``; the excess over
    the mean (equal-area pixels) goes through ``e⁺/(e⁺ + seed_c0)`` with the dead band of the box."""
    wc = cfg.weather
    p = params_from_config(cfg)
    geo = WeatherGeometry.from_config(cfg)
    zi = geo.block_mean(interface_height(ws, p))
    dc = mx.maximum(geo.block_mean(ws.Dc), p.seed_dmin)
    b = (geo.cloud_weights(zi, dc) * chem[2]).sum(axis=0)
    e = mx.maximum(b - b.mean() * (1.0 + _SEED_DEADBAND), 0.0)
    return e / (e + wc.seed_c0)


def seeding_qs(chem: mx.array, ws: WeatherStateS, cfg: Config) -> mx.array:
    """The rain-seeding factor on the threshold ``Q_s``, (Nw cells,) float32 in ``[1 - seed_gain, 1]``."""
    geo = WeatherGeometry.from_config(cfg)
    return geo.spread(1.0 - cfg.weather.seed_gain * seeding_response(chem, ws, cfg))


# --- validation and the host block --------------------------------------------------------------------------


def validate_config(cfg: Config) -> list[str]:
    """Messages for every violated weather rule of an enabled sphere config (empty = valid; never raises).

    The box rules that name the lattice (rule 15: ``L = world.size``, ``W-1``/``W-2``) are replaced by:
    ``weather.cells = Nw = r·Nf`` with ``r`` a power of two (the parent-child table is exact for any ratio;
    the power of two is the spec's), the Courant rule ``C_used <= 0.8·C_limit`` (``C_limit`` is the float32
    value of the S4a noise test, ``C_LIMIT``), pools within |lat| <= 60 deg with ``vy = 0``, ``H1 + H2 > 0``
    (the vertical map replaces ``H1 + H2 = height``), the load rules of the box with ``w_edge/s``, and the
    coupling keys (those of the box, minus the eddy and shade controls, which are not built on a sphere).
    """
    wc, w, ph = cfg.weather, cfg.world, cfg.physics
    gm = geom.geometry(cfg)
    bad: list[str] = []
    nw, nf = wc.cells, w.nside
    if not (
        math.isfinite(gm.radius)
        and gm.radius > 0
        and math.isfinite(gm.height_max)
        and gm.height_max > 0
        and gm.layers >= 2
        and nf >= 2
    ):  # the sphere rules report these; the weather rules need a shell to measure against
        return ["weather needs a valid sphere shell (world.radius, height, nside, layers)"]
    if not (isinstance(nw, int) and not isinstance(nw, bool) and nw >= 2 and nf >= 2):
        return [f"weather.cells must be an integer Nside >= 2 on a sphere, got {nw!r}"]
    if nw > 256:
        return [f"weather.cells {nw} is not a plausible weather Nside (<= 256)"]
    if len(wc.pools) % 7 != 0:
        return [
            "weather.pools must hold 7 numbers per pool (x, y, sx, sy, amp, vx, vy), "
            f"got {len(wc.pools)} values"
        ]
    if not all(math.isfinite(x) for x in wc.pools):
        return ["weather.pools must be finite numbers"]
    if nw % nf or (nw // nf) & (nw // nf - 1):
        bad.append(
            f"weather.cells {nw} must be Nf times a power of two (Nf = world.nside = {nf}): the rain is a "
            "mean over the children of a field pixel"
        )
    p = params_from_config(cfg)
    bad += _params_rules(p, w.radius)
    for i in range(0, len(wc.pools), 7):
        x, y, vy = wc.pools[i], wc.pools[i + 1], wc.pools[i + 6]
        lat = y / w.radius
        if not abs(lat) <= POOL_LAT_MAX + 1e-9:
            bad.append(f"rule S-2: pool {i // 7} latitude {math.degrees(lat):.1f} deg must lie within 60 deg")
        if vy != 0.0:
            bad.append(f"rule S-2: pool {i // 7} has vy = {vy:g}, which must be 0 on a sphere")
        if not 0.0 <= x < 2.0 * math.pi * w.radius:
            bad.append(
                f"rule S-2: pool {i // 7} x = {x:g} is an arc length east of longitude 0 and must lie in "
                f"[0, 2·pi·radius = {2.0 * math.pi * w.radius:.4g})"
            )
    # the vertical map and the load rules
    height = gm.height_max
    s_ratio = (wc.H1 + wc.H2) / height
    if not p.H1 + p.H2 > 0:
        bad.append("rule S-3: H1 + H2 must be > 0")
    else:
        s = s_ratio
        a_max = p.alpha0 + p.alpha1
        g, bmax = ph.gravity, ph.buoyancy_max
        wind_w = a_max * ph.drag * p.dt * p.w_edge / s  # the force is alpha·drag·dt·u (S4c)
        load = ph.thrust_max + max(g + wind_w, bmax - g + wind_w)
        if not load < 2 * ph.radius * ph.k_repel:
            bad.append(
                f"rule 16: thrust_max + max(g + a*drag*dt*w_edge/s, bmax - g + a*drag*dt*w_edge/s) "
                f"= {load:.4g} "
                f"must be < 2*radius*k_repel = {2 * ph.radius * ph.k_repel:.4g}"
            )
        speed = (ph.thrust_max + a_max * ph.drag * p.dt * p.u_max) / ph.drag if ph.drag > 0 else math.inf
        if not speed <= ph.v_max:
            bad.append(
                f"rule 16: (thrust_max + a*drag*dt*u_max)/drag = {speed:.4g} must be <= v_max = {ph.v_max:g}"
            )
    z_floor = float(geom.layer_edges(gm.radius, gm.height_max, gm.layers)[1])
    if not 0 < p.z_sl <= s_ratio * z_floor:
        bad.append(
            f"rule 15: need 0 < z_sl <= s·z_1 = {s_ratio * z_floor:.4g} (the shear layer lies within the "
            f"floor field layer, in weather units), got {p.z_sl:g}"
        )
    # the coupling keys (the box's rules, minus what a sphere does not have)
    if not 0.0 <= wc.storm_p <= 1.0:
        bad.append(f"weather.storm_p must be in [0, 1], got {wc.storm_p}")
    if not wc.storm_u0 >= 0.0:
        bad.append(f"weather.storm_u0 must be >= 0, got {wc.storm_u0}")
    if not wc.storm_span > 0.0:
        bad.append(f"weather.storm_span must be > 0, got {wc.storm_span}")
    if not (wc.storm_break_u == 0.0 or wc.storm_break_u >= wc.storm_u0):
        bad.append(
            f"weather.storm_break_u {wc.storm_break_u} must be 0 (off) or >= weather.storm_u0 {wc.storm_u0}"
        )
    if not 0.0 <= wc.rain_floor <= 1.0:
        bad.append(f"weather.rain_floor must be in [0, 1], got {wc.rain_floor}")
    if cfg.director.enabled and wc.rain_floor != 0.0:
        bad.append("weather.rain_floor must be 0 when director.enabled")
    if not wc.rain_gain >= 0.0:
        bad.append(f"weather.rain_gain must be >= 0, got {wc.rain_gain}")
    elif wc.gamma > 0 and not wc.rain_gain * (wc.H1 + wc.H2 - 2.0 * p.h_min) / wc.gamma <= weather._INT_BOUND:
        bad.append(
            f"weather.rain_gain {wc.rain_gain:g} too large: "
            "rain_gain*(H1 + H2 - 2*h_min)/gamma must be <= 2^30"
        )
    if not 0.0 <= wc.seed_gain <= 0.1:
        bad.append(f"weather.seed_gain must be in [0, 0.1], got {wc.seed_gain}")
    if not (wc.seed_c0 > 0.0 and math.isfinite(wc.seed_c0)):
        bad.append(f"weather.seed_c0 must be finite and > 0, got {wc.seed_c0}")
    if not 0.0 <= wc.wind_scale <= 1.0:
        bad.append(f"weather.wind_scale must be in [0, 1], got {wc.wind_scale}")
    for name in ("shade_gain", "rain_zonal", "storm_eddy", "wind_eddy_force"):
        if getattr(wc, name):
            bad.append(
                f"weather.{name} is not available on a sphere yet (the shade, the zonal mean and the "
                "eddy wind are calibration controls of the box: PR S4c)"
            )
    return bad


def _params_rules(p: WeatherParams, radius: float) -> list[str]:
    """The ``WeatherParams`` rules that do not name the box: layers, speeds, the Courant rule, moisture,
    damping, seeds (the box's ``validate`` rules 1-14 with the sphere's cell side for ``dx`` and no
    ``L``)."""
    bad: list[str] = []

    def req(ok: bool, msg: str) -> None:
        if not ok:
            bad.append(msg)

    req(p.n >= 2 and p.dx > 0 and p.dt > 0, "rule 1: need Nw >= 2, dx > 0, dt > 0")
    req(0 < p.h_min < p.H1 and p.h_min < p.H2, "rule 2: need 0 < h_min < H1 and h_min < H2")
    req(
        p.g_top > 0 and p.g_prime > 0 and p.g_prime < p.g_top,
        "rule 3: need g_top > 0, g_prime > 0 and g_prime < g_top",
    )
    if bad:
        return bad
    c_ext, c_bc = p._speeds()
    c_used = c_ext * p.dt / (0.81 * p.dx)
    req(
        c_used <= 0.8 * C_LIMIT,
        f"rule S-1: the Courant number C_used = c_ext·dt/(0.81·√A) = {c_used:.3f} must be <= 0.8·C_limit = "
        f"{0.8 * C_LIMIT:.3f} (C_limit {C_LIMIT} is the float32 limit of the noise test at a4 = {A4:g})",
    )
    req(p.f0 >= 0 and p.f0 * p.dt <= 0.5, "rule 5: need 0 <= f0 and f0*dt <= 0.5")
    if p.f0 > 0:
        req(c_bc / p.f0 >= 3 * p.dx, f"rule 6: L_d {c_bc / p.f0:.3g} < 3 dx (c_bc/f0 >= 3dx)")
    req(
        p.tau_c >= max(p.dt, p.dx / c_bc), f"rule 7: tau_c {p.tau_c:g} < max(dt, dx/c_bc = {p.dx / c_bc:.3g})"
    )
    req(
        p.gamma >= 0 and p.eps < 1 and p.H1 * (1 - p.eps) > p.m_min,
        "rule 8: need gamma >= 0, eps = gamma*Qs/H1 < 1 and H1*(1 - eps) > m_min",
    )
    req(p.gamma > 0, "rule 8: the sphere weather needs gamma > 0 (the moist limiters divide by it)")
    req(0 <= p.q_bg <= p.q_pool <= 1.2, "rule 9: need 0 <= q_bg <= q_pool <= 1.2")
    req(
        p.k_E >= 0 and p.u_gust >= 0 and p.dt * p.k_E * (p.u_safe + p.u_gust) <= 1,
        "rule 10: need k_E, u_gust >= 0 and dt*k_E*(u_safe + u_gust) <= 1",
    )
    req(p.tau_rad == 0 or p.tau_rad >= p.dt, "rule 11: tau_rad must be 0 (off) or >= dt")
    req(min(p.r1, p.c_d, p.r2) >= 0, "rule 11: need r1, c_d, r2 >= 0")
    req(
        0 <= p.seed_rate <= 1
        and p.seed_v >= 0
        and p.dx <= p.seed_sigma <= math.pi * radius / 4
        and 0 <= p.seed_jitter < 1,
        "rule 12: seed parameters out of range (rate in [0,1], dx <= sigma <= pi·R/4, jitter in [0,1))",
    )
    req(
        all(len(row) == 7 and row[4] >= 0 and row[3] > 0 for row in p.pools),
        "rule 13: every pool needs 7 entries, amp >= 0 and sy > 0",
    )
    req(0 < p.u_safe <= 0.3, "rule 14: need 0 < u_safe <= 0.3")
    req(
        p.alpha0 >= 0 and p.alpha1 >= 0 and p.alpha0 + p.alpha1 <= 1,
        "rule 16: need alpha0, alpha1 >= 0 and alpha_max = alpha0 + alpha1 <= 1",
    )
    req(p.u_max > 0 and p.w_edge >= 0 and p.z_sl > 0, "rule 18: need u_max > 0, w_edge >= 0 and z_sl > 0")
    return bad


def weather_block(
    wx: np.ndarray,
    air_speed: np.ndarray,
    cfg: Config,
    *,
    occupancy: np.ndarray | None = None,
    overlap_deep: int = 0,
    total_speed: np.ndarray | None = None,
    eddy_speed: np.ndarray | None = None,
    seed_r: np.ndarray | None = None,
    shade_s: np.ndarray | None = None,
    rain_audit: np.ndarray | None = None,
) -> dict[str, Any]:
    """The ``weather`` block of one metrics row on a sphere (plain floats and ints; the sphere's
    ``weather_obs.weather_block``; the storm proxy is the number of cyclonic-vorticity components of the cell
    graph, ``storms_sph.proxy``).

    ``wx`` is the stored ``(N_WX, N)``; ``air_speed`` the hazard speed of the alive rows. The keys are those
    of the box block that have a sphere meaning: the wind maxima of the two layers, the total rain
    (``Σ P·A``, ``A`` the cell area), the raining share of the cells, ``mass_rel`` (the relative drift of the
    column mass from ``N·(H1 + H2)``), ``q_total``, ``hyper_max``, the hazard exposure and the air-speed
    quantiles, the occupancy of the hash bands, the W2 coupling summaries that exist (``seed_r``, the rain
    shares), ``rain_cells``, ``vort_cells`` (the share of cells in the cyclonic-vorticity mask) and
    ``storm_proxy`` (its components of ``a_min`` cells with ``r_min`` raining ones, per hemisphere). No
    ``jet_max_l2`` (no zonal mean)."""
    from alife import weather_obs

    p = params_from_config(cfg)
    wc = cfg.weather
    w = np.asarray(wx, dtype=np.float64)
    h1, h2, u1, u2 = w[0], w[1], w[2:5], w[5:8]
    q, rain = w[8], w[9]
    n = w.shape[1]
    area = 4.0 * math.pi * cfg.world.radius**2 / n
    sp1, sp2 = np.sqrt((u1 * u1).sum(0)), np.sqrt((u2 * u2).sum(0))
    d = u1 - u2
    speed = np.asarray(air_speed, dtype=np.float64).reshape(-1)
    from alife import storms_sph

    zeta = np.array(vorticity(unpack(mx.array(np.asarray(wx, np.float32))), p, cfg.world.radius))
    storms, vort_cells = storms_sph.proxy(zeta, rain, storms_sph.SphereParams.from_config(cfg))
    block: dict[str, Any] = {
        "wind_max_l1": float(sp1.max()),
        "wind_max_l2": float(sp2.max()),
        "rain_total": float(rain.sum() * area),
        "rain_cells": float((rain >= p.p_th).mean()),
        "vort_cells": vort_cells,
        "storm_proxy": storms,
        "mass_rel": float((h1 + h2).sum() / (n * (p.H1 + p.H2)) - 1.0),
        "q_total": float(q.sum()),
        "hyper_max": float(((d * d).sum(0) / (p.g_prime * (h1 + h2))).max()),
    }
    if occupancy is not None:
        block.update({key: int(v) for key, v in zip(weather_obs.BAND_KEYS, occupancy, strict=True)})
    block.update(
        {
            "storm_exposure": weather_obs.storm_exposure(speed, wc.storm_u0, wc.storm_span),
            "air_speed_p99": float(np.quantile(speed, 0.99)) if speed.size else 0.0,
            "air_speed_max": float(speed.max()) if speed.size else 0.0,
            "overlap_deep": int(overlap_deep),
        }
    )
    if total_speed is not None and eddy_speed is not None:
        for name, v in (("total", total_speed), ("eddy", eddy_speed)):
            v = np.asarray(v, dtype=np.float64).reshape(-1)
            block[f"{name}_speed_p99"] = float(np.quantile(v, 0.99)) if v.size else 0.0
    if seed_r is not None:
        r = np.asarray(seed_r, dtype=np.float64)
        block["seed_r_mean"] = float(r.mean())
        block["seed_r_hi_share"] = float((r > 0.5).mean())
        block["seed_lower_mean"] = float(wc.seed_gain * r.mean())
    if rain_audit is not None:
        block.update(weather_obs.rain_shares(rain_audit, WeatherGeometry.from_config(cfg).rain_band()))
    return block
