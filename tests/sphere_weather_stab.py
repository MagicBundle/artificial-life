"""The long stability measurements of the sphere weather (S4b review M1; shared by the slow tests and the
S4c calibration): linear noise about the rest state, the growth of its rms over blocks of steps, and the
dominant growth rate of the tangent map by power iteration (renormalised every 100 steps).

Why both. The dry, frictionless scheme at ``Nw`` 42 with rotation has a slowly growing mode (about +1e-4 per
step at the first draft's C = 0.507 after 8 000 steps, +5e-5 at the C = 0.40 the engine runs at since C29; the
S4a gates and the 4 000-step tests stop before it sets in). In the production step the implicit friction and
the radiative return remove it: the dominant rate of the production map was 0 +- 2e-5 per step at C = 0.507
(no margin) and is -0.3e-5 at C = 0.40, still negative with ``r1`` and ``r2`` halved. ``r1``, ``r2`` and
``tau_rad`` are therefore stability parameters of the ``Nw`` 42 weather, and the power iteration is what
bounds the margin.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np

from alife import weather_sph, weather_tab
from alife.weather import ENGINE_FIXED, WeatherForcing, WeatherParams

BLOCK = 4000  # steps per reported block
RENORM = 100  # steps between renormalisations of the power iteration


def make_params(nw: int, radius: float, c: float, f0: float = 0.025, **over: float) -> WeatherParams:
    """The weather parameters at the Courant number ``c`` (``dt = c·0.81·sqrt(A)/c_ext``); ``over`` replaces
    ``ENGINE_FIXED`` or ``WeatherParams`` fields (``r1``, ``r2``, ``tau_rad``, ...)."""
    dt = c * 0.81 * weather_sph.cell_side(radius, nw) / WeatherParams().c_ext
    return WeatherParams(
        n=nw,
        dx=weather_sph.cell_side(radius, nw),
        height=80.0,
        **{**ENGINE_FIXED, "dt": dt, "f0": f0, **over},
    )


def _noise_state(p: WeatherParams, seed: int = 0, dh: float = 0.01, du: float = 1e-4):
    n = 12 * p.n * p.n
    x = weather_tab.weather_tab(p.n).x
    rng = np.random.default_rng(seed)
    h = np.array([[p.H1], [p.H2]]) + dh * rng.normal(size=(2, n))
    u = rng.normal(size=(2, n, 3))
    u = (u - (u * x).sum(-1, keepdims=True) * x) * du
    f = lambda a: mx.array(np.asarray(a, np.float32))  # noqa: E731
    s = weather_sph.WeatherStateS(
        f(h[0]), f(h[1]), f(u[0]), f(u[1]), mx.ones((n,)), mx.zeros((n,)), mx.zeros((n,))
    )
    return s, WeatherForcing(mx.zeros((n,)), mx.ones((n,))), mx.array(np.array([[p.H1], [p.H2]], np.float32))


def rms_blocks(
    nw: int, radius: float, c: float, nsteps: int, *, dry: bool = True, kappa_d: float = 0.0, **over: float
) -> list[float]:
    """Slope per step of ``log rms(h - h0)`` over consecutive blocks of ``BLOCK`` steps of linear noise (no
    advection, no seeds; ``dry``: the dynamics alone, as the S4a noise test)."""
    p = make_params(nw, radius, c, **over)
    s, fo, base = _noise_state(p)
    key = mx.random.key(0)
    out, prev = [], None
    for i in range(1, nsteps + 1):
        s, _ = weather_sph.step(s, fo, key, p, radius, dry=dry, advect=False, seeds=False, kappa_d=kappa_d)
        if i % BLOCK == 0:
            d = mx.stack([s.h1, s.h2]) - base
            r = math.log(float(mx.sqrt((d * d).mean())))
            if prev is not None:
                out.append((r - prev) / BLOCK)
            prev = r
    return out


def dominant_rate(
    nw: int, radius: float, c: float, nsteps: int, *, dry: bool = False, kappa_d: float = 0.0, **over: float
) -> list[float]:
    """Growth rate per step of the dominant mode of the one-step map about the rest state, per block of
    ``BLOCK`` steps: perturb, step, renormalise every ``RENORM`` steps and accumulate the log of the growth.
    ``dry=False`` is the production physics (friction, relaxation, tripwire, moisture at saturation)."""
    p = make_params(nw, radius, c, **over)
    s, fo, base = _noise_state(p, dh=0.05, du=1e-3)
    key = mx.random.key(0)
    target, acc, out = 0.05, 0.0, []
    for i in range(1, nsteps + 1):
        s, _ = weather_sph.step(s, fo, key, p, radius, dry=dry, advect=False, seeds=False, kappa_d=kappa_d)
        if i % RENORM == 0:
            d = mx.stack([s.h1, s.h2]) - base
            nrm = max(
                float(mx.sqrt((d * d).mean())),
                50.0 * float(mx.sqrt((mx.stack([s.u1, s.u2]) ** 2).sum(-1).mean())),
            )
            sc = target / nrm
            acc += math.log(nrm / target)
            s = weather_sph.WeatherStateS(
                base[0] + (s.h1 - base[0]) * sc,
                base[1] + (s.h2 - base[1]) * sc,
                s.u1 * sc,
                s.u2 * sc,
                s.Q,
                s.P,
                s.Dc,
            )
            mx.eval(*s)
        if i % BLOCK == 0:
            out.append(acc / BLOCK)
            acc = 0.0
    return out
