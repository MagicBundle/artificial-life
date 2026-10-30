"""A float64 numpy reference of the sphere weather step (S4b tests): the scheme of ``docs/SPHERE.md`` 4.1
written
the way the S4a prototype writes it (Jacobian first, then ``n·J·dx``; ``einsum`` over the tables), on the
tables
of ``alife.weather_tab``. The MLX step contracts the same operators differently (the ``bm`` table, gathers)
and
in float32, so agreement to float32 rounding tests the port and not a copy of it. The geometry itself is
tested against the independent HEALPix oracle in ``tests/test_weather_tab.py``, and against the S4a prototype
by the scripts logged in the S4b build notes (``xcheck_step.py``).
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

from alife import geom, weather_tab
from alife.weather import WeatherParams

E_SQRT = math.sqrt(math.e)


class Ref(NamedTuple):
    h: np.ndarray  # (2, N)
    u: np.ndarray  # (2, N, 3)
    q: np.ndarray  # (N,)
    p: np.ndarray  # rain (N,)
    dc: np.ndarray  # cloud depth (N,)


def _proj(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    return u - (u * x).sum(-1, keepdims=True) * x


class RefModel:
    def __init__(self, p: WeatherParams, radius: float, a4: float = 3e-3):
        self.p, self.r, self.a4 = p, radius, a4
        self.t = weather_tab.weather_tab(p.n)
        n = self.t.npix
        self.n = n
        self.ar2 = self.t.area * radius**2
        self.f = p.f0 * self.t.x[:, 2]
        self.revflat = (self.t.nb * 4 + self.t.rev).reshape(-1)

    def grad(self, f: np.ndarray) -> np.ndarray:
        df = f[..., self.t.nb8] - f[..., :, None]
        return np.einsum("...pk,pkj->...pj", df, self.t.w8)

    def jac(self, u: np.ndarray) -> np.ndarray:
        du = u[..., self.t.nb8, :] - u[..., :, None, :]
        return np.einsum("...pki,pkj->...pij", du, self.t.w8)

    def lap(self, f: np.ndarray, vec: bool = False) -> np.ndarray:
        if vec:
            return (self.t.wl[..., None] * (f[..., self.t.nb, :] - f[..., :, None, :])).sum(-2)
        return (self.t.wl * (f[..., self.t.nb] - f[..., :, None])).sum(-1)

    def bih(self, f: np.ndarray, vec: bool = False) -> np.ndarray:
        return -self.lap(self.lap(f, vec), vec)

    def fan(self, xd: np.ndarray):
        t = self.t
        c = geom.ang2pix(xd, t.nside)
        xc = t.x[c]
        dv = _proj(xc, xd - xc)
        p2 = np.stack([(dv * t.ex[c]).sum(1), (dv * t.ey[c]).sum(1)], 1)
        lam = np.einsum("mkij,mj->mki", t.tinv[c], p2)
        w0 = 1 - lam.sum(2)
        score = np.minimum(np.minimum(lam[..., 0], lam[..., 1]), w0)
        kb = score.argmax(1)
        r = np.arange(len(c))
        l1, l2 = lam[r, kb, 0], lam[r, kb, 1]
        j1 = t.nb8[c, kb]
        j2 = t.nb8[c, np.where(kb + 1 < t.cnt[c], kb + 1, 0)]
        return c, j1, j2, 1 - l1 - l2, l1, l2

    def step(
        self, s: Ref, ts: np.ndarray, qs: np.ndarray, seed_u: np.ndarray | None = None, advect=True, dry=False
    ):
        p, t, R, dt = self.p, self.t, self.r, self.p.dt
        h, u, bq = s.h, s.u, s.q
        x = t.x
        if advect:
            ua = np.empty_like(u)
            for k in range(2):
                xd = x - dt * u[k] / R
                xd /= np.linalg.norm(xd, axis=1)[:, None]
                c, j1, j2, w0, l1, l2 = self.fan(xd)
                ua[k] = _proj(x, w0[:, None] * u[k][c] + l1[:, None] * u[k][j1] + l2[:, None] * u[k][j2])
        else:
            ua = _proj(x, u)
        m2 = p.g_top * (h[0] + h[1])
        m0 = np.stack([m2 + p.g_prime * h[0], m2])
        gm = self.grad(m0) / R
        jac = self.jac(u)
        tt = np.einsum("pmi,kpij,pmj->kpm", t.n, jac, t.dx)
        tq = tt.reshape(2, -1)[:, self.revflat].reshape(2, self.n, 4)
        nb = t.nb
        un = 0.5 * (
            np.einsum("kpi,pmi->kpm", u, t.n) + np.einsum("kpmi,pmi->kpm", u[:, nb, :], t.n)
        ) + 0.5 * (tt - tq)
        comp = (m0[:, nb] - m0[:, :, None]) / (t.d * R) - 0.5 * (
            np.einsum("kpi,pmi->kpm", gm, t.e) + np.einsum("kpmi,pmi->kpm", gm[:, nb, :], t.e)
        )
        un = un - dt * comp * (t.e * t.n).sum(-1)
        gh = self.grad(h)
        hf = 0.5 * (h[:, :, None] + h[:, nb]) + 0.5 * (
            np.einsum("kpi,pmi->kpm", gh, t.dx) + np.einsum("kpmi,pmi->kpm", gh[:, nb, :], t.dx)
        )
        flux = t.ell * R * hf * un
        out = np.maximum(flux, 0).sum(-1)
        avail = np.maximum(h - p.h_min, 0) * self.ar2
        over = dt * out > avail
        lam = np.where(over, avail / (dt * np.maximum(out, 1e-300)), 1.0)
        flux = flux * np.where(flux > 0, lam[:, :, None], lam[:, nb])
        hn = h - dt * flux.sum(-1) / self.ar2
        q = bq / h[0]
        qm = np.where(flux[0] > 0, q[:, None], q[nb])
        qn = bq - dt * (flux[0] * qm).sum(-1) / self.ar2
        if self.a4:
            hn = hn + self.a4 * self.bih(hn)
        clamp = np.maximum(p.h_min - hn, 0).sum()
        qclamp = np.maximum(-qn, 0).sum()
        h = np.maximum(hn, p.h_min)
        bq = np.maximum(qn, 0)
        th = -self.f * dt / 2

        def rot(v):
            return np.cos(th)[:, None] * v + np.sin(th)[:, None] * np.cross(x, v)

        v = rot(ua)
        m2n = p.g_top * (h[0] + h[1])
        v = v - dt * self.grad(np.stack([m2n + p.g_prime * h[0], m2n])) / R
        v = _proj(x, rot(v))
        if self.a4:
            v = _proj(x, v + self.a4 * self.bih(v, True))
        ua = v
        if dry:
            return Ref(h, ua, bq, s.p, s.dc), {"clamp_mass": clamp, "q_clamp": qclamp}
        sp1 = np.sqrt((ua[0] * ua[0]).sum(-1))
        qe = p.Qs * (p.q_bg + (p.q_pool - p.q_bg) * ts)
        evap = p.k_E * (sp1 + p.u_gust) * np.maximum(qe - bq, 0)
        cap = np.maximum(h[0] - p.gamma * bq - p.m_min, 0) / (p.gamma * dt)
        evap = np.minimum(evap, cap)
        bq = bq + dt * evap
        cond = np.maximum(bq - p.Qs * qs, 0) / p.tau_c
        cond = np.minimum(cond, (h[0] - p.h_min) / (p.gamma * dt))
        bq = bq - dt * cond
        mu = p.gamma * dt * cond
        h1n, h2n = h[0] - mu, h[1] + mu
        u2 = ua[1] + (mu / h2n)[:, None] * (ua[0] - ua[1])
        h = np.stack([h1n, h2n])
        ua = np.stack([ua[0], u2])
        r = dt * (p.H1 - h[0]) / p.tau_rad
        r = np.minimum(np.maximum(r, -(h[0] - p.h_min)), h[1] - p.h_min)
        hn2 = np.stack([h[0] + r, h[1] - r])
        rr = np.stack([np.maximum(r, 0), np.maximum(-r, 0)])
        ua = ua + (rr / hn2)[..., None] * (ua[::-1] - ua)
        h = hn2
        sp = np.sqrt((ua * ua).sum(-1))
        lin = np.array([p.r1, p.r2])[:, None]
        quad = np.array([p.c_d, 0.0])[:, None]
        ua = ua / (1 + dt * (lin + quad * sp))[..., None]
        sp2 = np.sqrt((ua * ua).sum(-1))
        ua = ua * (p.u_safe / np.maximum(sp2, p.u_safe))[..., None]
        if seed_u is not None:
            ua = ua.copy()
            ua[0] = ua[0] + seed_u
        zi = p.height * h[0] / (h[0] + h[1])
        cloud = (p.height - zi) * cond / (cond + p.c_half)
        return Ref(h, ua, bq, cond, cloud), {
            "clamp_mass": clamp,
            "q_clamp": qclamp,
            "evap": evap,
            "cond": cond,
        }
