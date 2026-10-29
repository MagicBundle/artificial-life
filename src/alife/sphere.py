"""Device-side geometry of the sphere world (pure MLX, float32; DESIGN §46, true-sphere series S0).

Position to cell, the layer index, the hash and field cell ids, the exponential-map step with the transport of
the heading, and the device geometry accessors (``Box``, ``Sphere``; ``geometry(cfg)``). The host twin, the
exact tables and the derivations are ``alife.geom``; every function here is checked against it.

Nothing in the step uses this module yet (S2 does). Conventions of the simulation modules apply: no host
syncs, no Python branching on array values, fixed shapes. The scalar parameters (Nside, radii, ...) are plain
Python numbers in the engine; they may also be 0-d ``mx.array`` values, which is how a *compiled* test passes
them (no printable constant for the compile cache to confuse, DESIGN F21 item 6).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import mlx.core as mx

from alife import ops
from alife.config import dims

if TYPE_CHECKING:
    from alife.config import Config

_I32 = mx.int32
_HALF_PI = 0.5 * math.pi
_TWO_PI = 2.0 * math.pi


# --- position to cell ---------------------------------------------------------------------------------------


def ang2pix(pos: mx.array, nside: Any) -> mx.array:
    """RING pixel index of the direction of each position, (...,) int32 (HEALPix, any Nside).

    ``pos`` (..., 3) need not be normalised. The two branches (equatorial belt ``|z| <= 2/3`` and the polar
    caps) are both evaluated and ``mx.where`` selects; the cap radius uses ``1 − |z| = s2/(1 + |z|)`` with
    ``s2 = (x² + y²)/r²``, so it is stable at the poles. A point within float32 rounding of a pixel
    boundary may
    land in the adjacent pixel: the hash stencil covers that (margin of ``geom.coverage``) and a field
    cell is a
    bin. The result is clipped to ``[0, Npix)``.
    """
    x, y, z = pos[..., 0], pos[..., 1], pos[..., 2]
    xy2 = x * x + y * y
    r2 = mx.maximum(xy2 + z * z, 1e-30)
    zc = z / mx.sqrt(r2)
    s2 = xy2 / r2
    phi = mx.arctan2(y, x)
    phi = mx.where(phi < 0, phi + _TWO_PI, phi)
    tt = phi / _HALF_PI  # [0, 4]
    za = mx.abs(zc)
    npix = 12 * nside * nside
    ncap = 2 * nside * (nside - 1)
    # equatorial belt
    t1 = nside * (0.5 + tt)
    t2 = nside * zc * 0.75
    jp = mx.floor(t1 - t2).astype(_I32)
    jm = mx.floor(t1 + t2).astype(_I32)
    ir = nside + 1 + jp - jm
    ks = 1 - (ir & 1)
    ip = ((jp + jm - nside + ks + 1) >> 1) % (4 * nside)
    pe = ncap + (mx.clip(ir, 1, 2 * nside + 1) - 1) * 4 * nside + ip
    # polar caps
    tp = tt - mx.floor(tt)
    tmp = nside * mx.sqrt(3.0 * s2 / (1.0 + za))
    jpp = mx.floor(tp * tmp).astype(_I32)
    jmm = mx.floor((1.0 - tp) * tmp).astype(_I32)
    irr = mx.clip(jpp + jmm + 1, 1, nside)
    ipp = mx.floor(tt * irr.astype(mx.float32)).astype(_I32) % (4 * irr)
    pp = mx.where(zc > 0, 2 * irr * (irr - 1) + ipp, npix - 2 * irr * (irr + 1) + ipp)
    return mx.clip(mx.where(za <= 2.0 / 3.0, pe, pp), 0, npix - 1).astype(_I32)


def _isqrt(a: mx.array) -> mx.array:
    """Integer square root of non-negative int32 values below 2^23 (float32 square root, two corrections)."""
    r = mx.floor(mx.sqrt(a.astype(mx.float32))).astype(_I32)
    r = mx.where(r * r > a, r - 1, r)
    return mx.where((r + 1) * (r + 1) <= a, r + 1, r)


def pix2vec(pix: mx.array, nside: Any) -> mx.array:
    """Pixel centres of RING pixel indices, (..., 3) float32 unit vectors (closed form; ``Npix < 2^23``)."""
    npix = 12 * nside * nside
    ncap = 2 * nside * (nside - 1)
    n2 = nside * nside
    # north cap
    pn = mx.minimum(pix, ncap)
    ir_n = (1 + _isqrt(1 + 2 * pn)) >> 1
    ip_n = pn + 1 - 2 * ir_n * (ir_n - 1)
    z_n = 1.0 - (ir_n * ir_n).astype(mx.float32) / (3.0 * n2)
    ph_n = (ip_n.astype(mx.float32) - 0.5) * (math.pi / 2.0) / ir_n.astype(mx.float32)
    # south cap (mirror)
    ps = mx.maximum(npix - pix, 1)
    ir_s = (1 + _isqrt(2 * ps - 1)) >> 1
    ip_s = 4 * ir_s + 1 - (ps - 2 * ir_s * (ir_s - 1))
    z_s = -1.0 + (ir_s * ir_s).astype(mx.float32) / (3.0 * n2)
    ph_s = (ip_s.astype(mx.float32) - 0.5) * (math.pi / 2.0) / ir_s.astype(mx.float32)
    # equatorial belt
    pe = mx.clip(pix - ncap, 0, npix - 2 * ncap - 1)
    ir_e = pe // (4 * nside) + nside
    ip_e = pe % (4 * nside) + 1
    fodd = mx.where(((ir_e + nside) & 1) == 1, 1.0, 0.5)
    z_e = (2 * nside - ir_e).astype(mx.float32) * (2.0 / 3.0) / nside
    ph_e = (ip_e.astype(mx.float32) - fodd) * (math.pi / 2.0) / nside
    north, south = pix < ncap, pix >= npix - ncap
    z = mx.where(north, z_n, mx.where(south, z_s, z_e))
    phi = mx.where(north, ph_n, mx.where(south, ph_s, ph_e))
    st = mx.sqrt(mx.maximum(1.0 - z * z, 0.0))
    return mx.stack([st * mx.cos(phi), st * mx.sin(phi), z], axis=-1)


def radius_of(pos: mx.array) -> mx.array:
    """Distance from the planet centre, (...,)."""
    return mx.sqrt((pos * pos).sum(axis=-1))


def layer_index(pos: mx.array, rp3: Any, dv: Any, layers: Any) -> mx.array:
    """Equal-volume layer of each position: ``clip(floor((|p|³ − Rp³)/ΔV), 0, FZ−1)``, (...,) int32.
    ``rp3 = Rp³`` and ``dv = ΔV`` are host constants (``geom.layer_of`` is the reference); ``|p|³`` is
    ``r²·sqrt(r²)``, one square root."""
    r2 = (pos * pos).sum(axis=-1)
    p3 = r2 * mx.sqrt(r2)
    return mx.clip(mx.floor((p3 - rp3) / dv).astype(_I32), 0, layers - 1)


def hash_cell(pos: mx.array, alive: mx.array, rp: Any, th: Any, gz: Any, nside_h: Any) -> mx.array:
    """Hash cell id ``clip(floor(z/T_h), 0, G_z−1)·Npix_h + pixel`` (equal-thickness radial layers); dead
    rows get the sentinel ``Npix_h·G_z`` (one past the last cell)."""
    z = radius_of(pos) - rp
    lay = mx.clip(mx.floor(z / th).astype(_I32), 0, gz - 1)
    npix_h = 12 * nside_h * nside_h
    return mx.where(alive, lay * npix_h + ang2pix(pos, nside_h), npix_h * gz).astype(_I32)


def field_cell(pos: mx.array, rp3: Any, dv: Any, layers: Any, nside: Any) -> mx.array:
    """Flat field-cell index ``layer·Npix + pixel`` (z-major, as the box's ``(cz·FG + cy)·FG + cx``)."""
    return (layer_index(pos, rp3, dv, layers) * (12 * nside * nside) + ang2pix(pos, nside)).astype(_I32)


# --- motion: the exponential-map step and the transport of the heading (§46.2) ---------------------------


def _cross(a: mx.array, b: mx.array) -> mx.array:
    return mx.stack(
        [
            a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
            a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
            a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0],
        ],
        axis=-1,
    )


def _dot(a: mx.array, b: mx.array) -> mx.array:
    return (a * b).sum(axis=-1, keepdims=True)


def move(
    pos: mx.array, hdg: mx.array, v: mx.array, turn: mx.array, r_lo: Any, r_hi: Any
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
    """One step of every agent on the sphere (the geometry of DESIGN §46.2; forces and counters are S2's).

    ``pos`` (N,3) positions, ``hdg`` (N,3) tangent unit headings, ``v`` (N,3) the commanded displacement
    (already clipped to ``v_max``), ``turn`` (N,) the yaw increment. The tangent part of ``v`` moves the
    direction along the great circle by the exponential map, as a series in ``σ² = |vt|²/r²`` (truncation
    error σ⁶/720, far below float32 rounding for any step the engine takes; the result is renormalised): no
    trig and no division by ``|vt|``, so ``vt = 0`` is exact. The radial part changes the distance, clamped to
    ``[r_lo, r_hi]``. The heading is transported by the minimal rotation taking the old up to the new one
    (exact parallel transport along the great circle), turned about the new up, and renormalised (mandatory:
    without it ``|hdg|`` drifts by 0.11 over 3·10^5 steps).

    A row with ``pos = 0`` gives ``0/0``: the caller keeps dead rows at ``(0, 0, r_lo)`` (the engine's
    convention) or masks the result with ``where``.

    Returns ``(pos', hdg', vel', at_floor, at_lid)``: ``vel'`` is ``v`` with its radial part set to 0 on a
    clamped row (the box's surface rule made radial).
    """
    r = radius_of(pos)[..., None]
    up = pos / r
    vr = _dot(v, up)
    vt = v - vr * up
    w = vt / r
    s2 = _dot(w, w)
    c = 1.0 - s2 / 2.0 + s2 * s2 / 24.0
    s = 1.0 - s2 / 6.0 + s2 * s2 / 120.0
    u2 = c * up + s * w
    u2 = u2 / mx.sqrt(_dot(u2, u2))
    r_new = r + vr
    at_floor = (r_new <= r_lo) & (vr < 0)
    at_lid = (r_new >= r_hi) & (vr > 0)
    r_new = mx.clip(r_new, r_lo, r_hi)
    pos2 = r_new * u2
    # minimal rotation up -> up2: axis k = up × up2 = s·(up × w), cosine c (no cancellation of two unit
    # vectors)
    k = s * _cross(up, w)
    h1 = hdg + _cross(k, hdg) + _cross(k, _cross(k, hdg)) / (1.0 + c)
    ct, st = mx.cos(turn)[..., None], mx.sin(turn)[..., None]
    h2 = ct * h1 + st * _cross(u2, h1)
    h2 = h2 - _dot(h2, u2) * u2
    h2 = h2 / mx.sqrt(_dot(h2, h2))
    clamped = at_floor | at_lid
    vel = mx.where(clamped, v - vr * up, v)
    return pos2, h2, vel, at_floor[..., 0], at_lid[..., 0]


def bearing(hdg: mx.array, pos: mx.array) -> mx.array:
    """Compass bearing of the heading from local north, ``atan2(hdg·east, hdg·north)`` in [−π, π)
    (``east = normalise(ẑ × û)``, x̂ within 1e-6 of a pole; ``north = û × east``). A bearing within 1e-6 of +π
    is returned as −π (float32 rounding of the south), whereas the host maps ``>= π``: compare circularly."""
    up = pos / radius_of(pos)[..., None]
    ex, ey = -up[..., 1], up[..., 0]
    ne = mx.sqrt(ex * ex + ey * ey)
    polar = ne < 1e-6
    safe = mx.where(polar, 1.0, ne)
    east = mx.stack(
        [mx.where(polar, 1.0, ex / safe), mx.where(polar, 0.0, ey / safe), mx.zeros_like(ne)], axis=-1
    )
    north = _cross(up, east)
    b = mx.arctan2((hdg * east).sum(axis=-1), (hdg * north).sum(axis=-1))
    return mx.where(b >= math.pi - 1e-6, b - _TWO_PI, b)  # [−π, π) up to float32 rounding of the south


# --- the accessors (DESIGN §46.12) ------------------------------------------------------------------------


class Box:
    """The torus-and-floor world: today's expressions, written once (device). Converting a consumer to these
    methods changes no bit (the identity matrix proves it)."""

    name = "box"

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.size = cfg.world.size

    def height(self, pos: mx.array) -> mx.array:
        return pos[..., 2]

    def up(self, pos: mx.array) -> mx.array:
        return mx.broadcast_to(mx.array([0.0, 0.0, 1.0], dtype=pos.dtype), pos.shape)

    def radial(self, vec: mx.array, pos: mx.array) -> mx.array:
        return vec[..., 2]

    def tangent(self, vec: mx.array, pos: mx.array) -> mx.array:
        return mx.concatenate([vec[..., :2], mx.zeros_like(vec[..., 2:])], axis=-1)

    def displacement(self, a: mx.array, b: mx.array) -> mx.array:
        return ops.min_image(b - a, self.size)

    def distance(self, a: mx.array, b: mx.array) -> mx.array:
        d = self.displacement(a, b)
        return mx.sqrt((d * d).sum(axis=-1))

    def arc(self, a: mx.array, b: mx.array) -> mx.array:
        return self.distance(a, b)

    def cell_of(self, pos: mx.array) -> mx.array:
        from alife import fields  # deferred: fields is a heavier module than this one

        return fields.cell_index(pos, self.cfg)

    def bearing(self, vec: mx.array, pos: mx.array) -> mx.array:
        return mx.arctan2(vec[..., 1], vec[..., 0])


class Sphere:
    """The sphere world (device): constants from the config and ``geom``, methods on float32 positions."""

    name = "sphere"

    def __init__(self, cfg: Config):
        from alife import geom  # host derivations (Nh, layers)

        d = dims(cfg)
        w = cfg.world
        self.radius, self.height_max = w.radius, w.height
        self.nside, self.layers, self.npix = d.nside, d.fz, d.npix
        rt = w.radius + w.height
        self.rp3 = w.radius**3
        self.dv = (rt**3 - w.radius**3) / d.fz
        self.nside_h, self.npix_h, self.gz, self.th = d.nside_h, d.npix_h, d.g_hash_z, d.cell_hash_z
        self.r_lo, self.r_hi = d.r_lo, d.r_hi
        self.layer_radii = geom.layer_radii(w.radius, w.height, d.fz)

    def height(self, pos: mx.array) -> mx.array:
        return radius_of(pos) - self.radius

    def up(self, pos: mx.array) -> mx.array:
        return pos / radius_of(pos)[..., None]

    def radial(self, vec: mx.array, pos: mx.array) -> mx.array:
        return (vec * self.up(pos)).sum(axis=-1)

    def tangent(self, vec: mx.array, pos: mx.array) -> mx.array:
        u = self.up(pos)
        return vec - _dot(vec, u) * u

    def displacement(self, a: mx.array, b: mx.array) -> mx.array:
        return b - a  # min_image is the identity on a sphere's pair offsets (world.size > 4(Rp + H))

    def distance(self, a: mx.array, b: mx.array) -> mx.array:
        d = b - a
        return mx.sqrt((d * d).sum(axis=-1))

    def arc(self, a: mx.array, b: mx.array) -> mx.array:
        ua, ub = self.up(a), self.up(b)
        ang = mx.arctan2(mx.sqrt((_cross(ua, ub) ** 2).sum(axis=-1)), (ua * ub).sum(axis=-1))
        return ang * 0.5 * (radius_of(a) + radius_of(b))

    def cell_of(self, pos: mx.array) -> mx.array:
        return field_cell(pos, self.rp3, self.dv, self.layers, self.nside)

    def hash_cell_of(self, pos: mx.array, alive: mx.array) -> mx.array:
        return hash_cell(pos, alive, self.radius, self.th, self.gz, self.nside_h)

    def bearing(self, vec: mx.array, pos: mx.array) -> mx.array:
        return bearing(vec, pos)


def geometry(cfg: Config) -> Box | Sphere:
    """The static geometry object of a config (device); a trace-time constant."""
    return Sphere(cfg) if cfg.world.geometry == "sphere" else Box(cfg)
