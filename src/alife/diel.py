"""The day-night light cycle on the photic food band (DESIGN §49.3, §49.5, §49.8; docs/DIEL.md §2).

A light factor ``L(t, cell)`` in [0, 1] multiplies only the photic term of the vertical food profile:
``V = min(1, B + L·Pk)`` with ``B[k] = exp(−z_k/benthic_lambda)`` and ``Pk[k] = photic_amp·exp(−(z_top −
z_k)/photic_lambda)``, the two terms of ``fields.vertical_profile`` built apart on the host. Where the light
is full (the smoothstep ``S`` is exactly 1) the profile **is** the static one, element for element: today's
profile is summed and clipped in float64 and rounded once, which a float32 ``B + Pk`` would not reproduce bit
for bit, so ``profile`` selects the static array there (``mx.where``, no branch). Food stays the integer
``INJECT_REGEN`` source; no ledger term is added (§49.3).

Everything here is pure MLX in the step (no host sync, fixed shapes, branches on config constants only); the
host tables (``split``, ``v_bar``) are float64 numpy, cast once and cached per config, as
``fields_sph.profile``. This module imports neither ``fields`` nor ``state`` (``fields`` imports it).
"""

from __future__ import annotations

import functools
import math
from typing import TYPE_CHECKING

import mlx.core as mx
import numpy as np

from alife import fields_sph, fieldtab, geom
from alife.config import dims

if TYPE_CHECKING:
    from alife.config import Config

_F32 = mx.float32
VBAR_PHASES = 720  # the phases of the day mean V_bar (§49.5), equally spaced


def on(cfg: Config) -> bool:
    """The cycle is on (``diel.period > 0``): the static switch of every diel expression."""
    return cfg.diel.period > 0


def live(s: object, cfg: Config) -> mx.array | None:
    """The effective day fraction a state's steps use (``diel_ctl[1]``), or None while the cycle is off; what
    the host probes of ``metrics`` pass as ``diel=`` so that they reproduce the step's regeneration."""
    return s.diel_ctl[1] if on(cfg) else None  # type: ignore[attr-defined]


# --- host tables ------------------------------------------------------------------------------------


def split(cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """``(B, Pk)``, (FZ,) float64: the benthic and photic terms of the vertical profile, the same per-element
    arithmetic as ``fields.vertical_profile`` (box) and ``fieldtab.vertical_profile`` (sphere), before their
    sum and the clip at 1."""
    fi = cfg.fields
    if dims(cfg).sph:  # static: the layers' lower interface heights (§46.5)
        radius, height, _, layers = fieldtab.shell(cfg)
        z = geom.layer_edges(radius, height, layers)[:-1]
        return np.exp(-z / fi.benthic_lambda), fi.photic_amp * np.exp(-(z[-1] - z) / fi.photic_lambda)
    fz, fc = dims(cfg).fz, cfg.world.field_cell
    b = [math.exp(-k * fc / fi.benthic_lambda) for k in range(fz)]
    p = [fi.photic_amp * math.exp(-(fz - 1 - k) * fc / fi.photic_lambda) for k in range(fz)]
    return np.array(b, dtype=np.float64), np.array(p, dtype=np.float64)


def _shape(cfg: Config) -> tuple[int, ...]:
    """The shape of a per-layer column: (FZ,1,1) in the box, (FZ,1) on the sphere."""
    return (-1, 1) if dims(cfg).sph else (-1, 1, 1)


def smooth_light(h: np.ndarray, day_frac: float, cfg: Config) -> np.ndarray:
    """The host float64 ``L`` of the sun's sine ``h`` (any shape): ``h0 = cos(pi·day_frac)``, ``x = clip((h −
    h0)/tw + 1/2, 0, 1)``, ``S = x²(3 − 2x)``, ``L = n + (1 − n)·S``. The reference of ``light``."""
    c = cfg.diel
    x = np.clip((h - math.cos(math.pi * day_frac)) / c.twilight + 0.5, 0.0, 1.0)
    return c.night_level + (1.0 - c.night_level) * (x * x * (3.0 - 2.0 * x))


def sun_sine(phase: np.ndarray | float, cfg: Config) -> np.ndarray:
    """The host float64 sine of the sun's elevation at the day phase ``th`` (radians): ``cos th`` (global,
    shape of ``th``) or ``u_c · (cos th, −sin th, 0)`` per pixel (terminator, (..., Npix))."""
    th = np.asarray(phase, dtype=np.float64)
    if cfg.diel.mode == "terminator":  # static
        u = geom.pixel_centres(cfg.world.nside)
        return u[:, 0] * np.cos(th)[..., None] - u[:, 1] * np.sin(th)[..., None]
    return np.cos(th)


@functools.cache
def v_bar_host(cfg: Config) -> np.ndarray:
    """The day-mean profile ``V_bar[k, c] = mean_j min(1, B[k] + L(th_j, c)·Pk[k])`` over ``VBAR_PHASES``
    equally spaced phases ``th_j = 2·pi·j/720`` at the configured ``day_frac``, float64: (FZ,) in global mode,
    (FZ, Npix) in terminator mode (§49.5). Accumulated phase by phase (a (720, FZ, Npix) array would be
    large)."""
    b, p = split(cfg)
    acc = None
    for j in range(VBAR_PHASES):
        lt = smooth_light(sun_sine(2.0 * math.pi * j / VBAR_PHASES, cfg), cfg.diel.day_frac, cfg)
        v = np.minimum(1.0, b[:, None] + np.atleast_1d(lt)[None, :] * p[:, None])
        acc = v if acc is None else acc + v
    out = acc / VBAR_PHASES
    return out[:, 0] if cfg.diel.mode != "terminator" else out


@functools.cache
def v_bar(cfg: Config) -> mx.array:
    """``v_bar_host`` cast to float32 in the profile's shape: (FZ,1,1) box, (FZ,1) sphere global, (FZ, Npix)
    terminator. The profile of the matched control (``control = "mean"``) and of the time-average (``step``
    None: the initial resource and the carrying-capacity estimate)."""
    v = v_bar_host(cfg)
    if cfg.diel.mode == "terminator":  # static
        return mx.array(v, dtype=_F32)
    return mx.array(v, dtype=_F32).reshape(_shape(cfg))


@functools.cache
def _terms(cfg: Config) -> tuple[mx.array, mx.array]:
    b, p = split(cfg)
    return mx.array(b, dtype=_F32).reshape(_shape(cfg)), mx.array(p, dtype=_F32).reshape(_shape(cfg))


# --- in the step ------------------------------------------------------------------------------------


def light(step: mx.array, day_eff: mx.array | None, cfg: Config) -> tuple[mx.array, mx.array]:
    """``(L, S)`` at step ``step`` (int32) for the effective day fraction ``day_eff`` (float32 scalar; None =
    the configured ``day_frac``): scalars in global mode, (Npix,) in terminator mode (§49.3). ``th =
    2·pi·(t mod P)/P`` with the integer modulo first; the terminator's ``h = u_c · s(t)`` is three elementwise
    products and a sum (no matmul, F20)."""
    c = cfg.diel
    if day_eff is None:
        day_eff = mx.array(c.day_frac, dtype=_F32)
    th = 2.0 * math.pi * (step % c.period).astype(_F32) / c.period
    if c.mode == "terminator":  # static
        u = fields_sph.cell_centres(cfg.world.nside)
        h = u[:, 0] * mx.cos(th) + u[:, 1] * (-mx.sin(th))
    else:
        h = mx.cos(th)
    h0 = mx.cos(math.pi * day_eff)
    x = mx.clip((h - h0) / c.twilight + 0.5, 0.0, 1.0)
    s = x * x * (3.0 - 2.0 * x)
    return c.night_level + (1.0 - c.night_level) * s, s


def profile(cfg: Config, step: mx.array, day_eff: mx.array | None, static: mx.array) -> mx.array:
    """The vertical food profile of step ``step`` with the cycle on: ``min(1, B + L·Pk)``, the static profile
    ``static`` (``fields.vertical_profile``) wherever ``S`` is 1, and ``V_bar`` under ``control = "mean"``.
    Shapes: (FZ,1,1) box, (FZ,1) sphere global, (FZ, Npix) terminator."""
    if cfg.diel.control == "mean":  # static: the matched control (§49.5)
        return v_bar(cfg)
    lt, s = light(step, day_eff, cfg)
    b, p = _terms(cfg)
    if cfg.diel.mode == "terminator":  # static: one light per pixel
        lt, s = lt[None, :], s[None, :]
    v = mx.minimum(b + lt * p, 1.0)
    return mx.where(s >= 1.0, static, v)


def ramp(ctl: mx.array, cfg: Config) -> mx.array:
    """``diel_ctl`` one step later: the effective day fraction moves toward the target by at most ``1/(2P)``
    (§49.8), so the light never jumps when Long Night writes a target. ``ctl`` = (target, effective)."""
    lim = 1.0 / (2.0 * cfg.diel.period)
    eff = ctl[1] + mx.clip(ctl[0] - ctl[1], -lim, lim)
    return mx.stack([ctl[0], eff])
