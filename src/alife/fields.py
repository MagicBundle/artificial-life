"""Lattice fields: fertility patches, resource and chemicals (§13).

``init_patches``, ``fertility`` and ``regen_rate`` are WP0 hand-over functions; WP1 owns this module.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, NamedTuple

import mlx.core as mx

from alife import layout, ops
from alife.config import Config, dims, seasons_on

if TYPE_CHECKING:  # state imports this module; the annotation alone must not create an import cycle
    from alife.state import Env


class FieldObs(NamedTuple):
    res: mx.array  # (N,)
    res_grad: mx.array  # (N,2) body frame, normalised
    chem: mx.array  # (N,NCH)
    chem_grad: mx.array  # (N,NCH,2) body frame, normalised


def init_patches(key_pos: mx.array, key_vel: mx.array, cfg: Config) -> tuple[mx.array, mx.array]:
    """Patch centres ``U[0, L)²`` and constant drifts ``patch_drift·(cos ψ, sin ψ)``, ψ ~ U[0, 2π)."""
    pp, L = dims(cfg).pp, cfg.world.size
    pos = ops.wrap(mx.random.uniform(0.0, L, (pp, 2), key=key_pos), L)
    psi = mx.random.uniform(0.0, 2.0 * math.pi, (pp,), key=key_vel)
    vel = cfg.fields.patch_drift * mx.stack([mx.cos(psi), mx.sin(psi)], axis=-1)
    return pos, vel


def season_weights(step: mx.array, cfg: Config, env: Env | None = None) -> mx.array:
    """Per-patch seasonal weights ``w_p = 1 + A·sin(2π·t/P + φ_p)``, (PP,) f32 (§13.1).

    ``φ_p = 0`` for ``season_mode = "global"`` and ``2π·p/PP`` for ``"per_patch"``. ``t mod P`` is taken in
    integers first so the phase stays exact for long runs. Only called when seasons are on, or (with ``env``,
    i.e. the director enabled) always: then ``A`` and ``P`` come from ``env.season_amp`` and
    ``env.season_period``, and ``P = 0`` means off (all weights 1; the modulo is guarded against it).
    """
    fi = cfg.fields
    pp = dims(cfg).pp
    offset = (2.0 * math.pi / pp) * mx.arange(pp, dtype=mx.float32) if fi.season_mode == "per_patch" else 0.0
    if env is None:
        phase = 2.0 * math.pi * (step % fi.season_period).astype(mx.float32) / fi.season_period
        return 1.0 + fi.season_amplitude * mx.sin(phase + offset * mx.ones((pp,), mx.float32))
    period = mx.maximum(env.season_period, 1)
    phase = 2.0 * math.pi * (step % period).astype(mx.float32) / period.astype(mx.float32)
    w = 1.0 + env.season_amp * mx.sin(phase + offset * mx.ones((pp,), mx.float32))
    return mx.where(env.season_period > 0, w, 1.0)


def fertility(
    patch_pos: mx.array, cfg: Config, step: mx.array | None = None, env: Env | None = None
) -> mx.array:
    """``clip(fertility_base + Σ_p w_p·torus-Gaussian_p, 0, 1)``, (FG,FG) f32 indexed [y, x].

    ``w_p`` is the seasonal weight at ``step`` (``season_weights``); it is 1 when seasons are off or ``step``
    is None (the time-average, used for initial fields and the carrying-capacity estimate). With seasons off
    the computation is exactly the unweighted sum. With ``env`` (the director enabled) and a ``step``, the
    weight is ``env.patch_gain[p]·w_p`` with the env season parameters (gain 0 disables a patch).

    Separable: ``Σ_p gy_p ⊗ gx_p`` with per-axis minimum-image offsets from cell centres. The sum is
    elementwise rather than ``gyᵀ @ gx`` because float32 matmul runs at reduced precision on some Apple GPUs
    (≈ 8e-4 relative on M5), which would perturb the integer regeneration map well beyond rounding.

    In 3D (§27.5) the result is (FZ,FG,FG) indexed [z, y, x]: this 2D fertility times ``vertical_profile``,
    whose floor entry is exactly 1, so layer 0 equals the 2D fertility bit for bit.
    """
    fg, L, fc = dims(cfg).fg, cfg.world.size, cfg.world.field_cell
    centres = (mx.arange(fg, dtype=mx.float32) + 0.5) * fc
    inv = -1.0 / (2.0 * cfg.fields.patch_sigma**2)
    dx = ops.min_image(centres[None, :] - patch_pos[:, 0:1], L)  # (PP,FG)
    dy = ops.min_image(centres[None, :] - patch_pos[:, 1:2], L)
    gx, gy = mx.exp(dx * dx * inv), mx.exp(dy * dy * inv)
    terms = gy[:, :, None] * gx[:, None, :]
    if step is not None and env is not None:  # static: director enabled
        terms = (env.patch_gain * season_weights(step, cfg, env))[:, None, None] * terms
    elif step is not None and seasons_on(cfg.fields):  # static: seasons enabled
        terms = season_weights(step, cfg)[:, None, None] * terms
    fert = mx.clip(cfg.fields.fertility_base + terms.sum(axis=0), 0.0, 1.0)
    if dims(cfg).d3:  # static: broadcast over the layers; never a (PP,FZ,FG,FG) array
        return fert[None] * vertical_profile(cfg)
    return fert


def vertical_profile(cfg: Config) -> mx.array:
    """The 3D food profile ``V`` by layer index ``k``, (FZ,1,1) f32 (§27.5).

    ``V[k] = min(1, exp(−k·fc/benthic_lambda) + photic_amp·exp(−(FZ−1−k)·fc/photic_lambda))`` with
    ``fc = field_cell``: a benthic band at the floor plus an optional photic band under the surface (its term
    exists only when ``photic_amp > 0``). ``V[0] = 1`` exactly. Built on the host from config constants.
    """
    d, fi, fc = dims(cfg), cfg.fields, cfg.world.field_cell
    v = [math.exp(-k * fc / fi.benthic_lambda) for k in range(d.fz)]
    if fi.photic_amp > 0:  # static: the photic band is on
        v = [b + fi.photic_amp * math.exp(-(d.fz - 1 - k) * fc / fi.photic_lambda) for k, b in enumerate(v)]
    return mx.array([min(1.0, x) for x in v], dtype=mx.float32).reshape(d.fz, 1, 1)


def regen_rate(
    patch_pos: mx.array, cfg: Config, step: mx.array | None = None, env: Env | None = None
) -> mx.array:
    """Per-cell regeneration ``floor(regen_max·fertility)``, (FG,FG) int32 (seasonal at ``step`` if given).

    With ``env`` the product is scaled by ``env.regen_mult`` after ``regen_max·fertility``, so 1.0 is exact.
    In 3D it has the fertility's shape (FZ,FG,FG) (§27.5).
    """
    rate = cfg.fields.regen_max * fertility(patch_pos, cfg, step, env)
    if env is not None:  # static: director enabled
        rate = rate * env.regen_mult
    return ops.floor_i32(rate)


def cell_index(pos: mx.array, cfg: Config) -> mx.array:
    """Flat field-cell index ``cy·FG + cx`` of each position, (N,) int32 (§13).

    ``% FG`` is a second seam guard: a position that rounds to ``L`` maps to cell 0. In 3D the index is
    z-major, ``(cz·FG + cy)·FG + cx`` (``_cell_index_3d``, §27.5).
    """
    if dims(cfg).d3:
        return _cell_index_3d(pos, cfg)
    assert pos.shape[-1] == 2, "the 2D cell index got 3D positions"  # static: z would be dropped silently
    fg = dims(cfg).fg
    cxy = ops.floor_i32(pos / cfg.world.field_cell) % fg
    return cxy[:, 1] * fg + cxy[:, 0]


def _cell_index_3d(pos: mx.array, cfg: Config) -> mx.array:
    """Z-major flat index ``(cz·FG + cy)·FG + cx`` of 3D positions, (N,) int32 (§27.5).

    x and y keep the ``% FG`` seam guard; ``cz`` is clamped to ``[0, FZ − 1]``, never wrapped. Layer 0's index
    equals the 2D index, and the floor cell below any position is ``fcell % FG²``.
    """
    d, fc = dims(cfg), cfg.world.field_cell
    cxy = ops.floor_i32(pos[:, :2] / fc) % d.fg
    cz = mx.clip(ops.floor_i32(pos[:, 2] / fc), 0, d.fz - 1)
    return (cz * d.fg + cxy[:, 1]) * d.fg + cxy[:, 0]


def _nbr_cells(fcell: mx.array, fg: int) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Flat indices of the periodic neighbour cells ``(x+1, x−1, y+1, y−1)`` of each flat cell index."""
    cy, cx = fcell // fg, fcell % fg
    xp, xm = (cx + 1) % fg, (cx - 1 + fg) % fg
    yp, ym = (cy + 1) % fg, (cy - 1 + fg) % fg
    return cy * fg + xp, cy * fg + xm, yp * fg + cx, ym * fg + cx


def sample_fields(
    resource: mx.array, chem: mx.array, fcell: mx.array, heading: mx.array, cfg: Config
) -> FieldObs:
    """Field inputs at each agent's start-of-step cell, already normalised as in §5.

    - ``res``: ``min(R / resource_cap, RES_SAT) / RES_SAT`` (``IN_RES``).
    - ``res_grad``: ``clip(RES_GRAD_GAIN·to_body(g), −1, 1)`` with the central difference
      ``g = (R[y,x+1] − R[y,x−1], R[y+1,x] − R[y−1,x]) / (2·resource_cap)`` (``IN_RES_GRAD``).
    - ``chem``: ``c / (c + chem_half_sat)`` per channel (``IN_CHEM``).
    - ``chem_grad``: ``tanh(to_body(central difference / 2) / chem_half_sat)`` per channel (``IN_CHEM_GRAD``).

    Dead rows get stale values (from their stale ``pos``); consumers mask by ``alive``.
    """
    assert resource.ndim == 2, "2D field sampling got a 3D lattice"  # static: _nbr_cells would fold z into y
    fg = dims(cfg).fg
    fi, half = cfg.fields, cfg.sensing.chem_half_sat
    cos_h, sin_h = mx.cos(heading), mx.sin(heading)
    xp, xm, yp, ym = _nbr_cells(fcell, fg)

    r = resource.reshape(-1).astype(mx.float32)  # R ≤ R_HARD < 2^24: exact in f32
    res = mx.minimum(r[fcell] / fi.resource_cap, layout.RES_SAT) / layout.RES_SAT
    g = mx.stack([r[xp] - r[xm], r[yp] - r[ym]], axis=-1) / (2.0 * fi.resource_cap)  # (N,2) world frame
    res_grad = mx.clip(layout.RES_GRAD_GAIN * ops.to_body(g, cos_h, sin_h), -1.0, 1.0)

    c = chem.reshape(layout.NCH, fg * fg)
    cv = c[:, fcell].T  # (N,NCH)
    cg = mx.stack([(c[:, xp] - c[:, xm]).T, (c[:, yp] - c[:, ym]).T], axis=-1) * 0.5  # (N,NCH,2) world
    chem_grad = mx.tanh(ops.to_body(cg, cos_h[:, None], sin_h[:, None]) / half)
    return FieldObs(res=res, res_grad=res_grad, chem=cv / (cv + half), chem_grad=chem_grad)


def deposit_chem(emit: mx.array, fcell: mx.array, active: mx.array, cfg: Config) -> mx.array:
    """Integer chemical deposit quanta per channel and cell, (NCH·FG·FG,) int32 (§13.3).

    Each active agent adds ``floor(emit_c·256)`` quanta at flat index ``c·FG² + fcell`` (int scatter-add, so
    the result is independent of atomic order).
    """
    fg2 = dims(cfg).fg ** 2
    quanta = mx.maximum(ops.floor_i32(emit * 256.0), 0)  # (N,NCH); emit ∈ [0, 1) → [0, 255]
    flat = fcell[:, None] + mx.arange(layout.NCH, dtype=mx.int32) * fg2  # (N,NCH)
    mask = mx.broadcast_to(active[:, None], flat.shape)
    return ops.scatter_add_i32(layout.NCH * fg2, flat, quanta, mask)


def update_chem(chem: mx.array, dep_q: mx.array, cfg: Config) -> mx.array:
    """Deposit, then diffuse and decay, then clamp the chemical channels (§13.3).

    ``chem_c ← clip((1 − λ_c)·(chem_c + D_c·(roll₄(chem_c) − 4·chem_c)), 0, chem_max)`` after
    ``chem += dep_q·chem_deposit_scale``.
    """
    fi, fg = cfg.fields, dims(cfg).fg
    diff = mx.array(fi.chem_diffusion, dtype=mx.float32).reshape(layout.NCH, 1, 1)
    keep = mx.array([1.0 - lam for lam in fi.chem_decay], dtype=mx.float32).reshape(layout.NCH, 1, 1)
    c = chem + dep_q.reshape(layout.NCH, fg, fg).astype(mx.float32) * fi.chem_deposit_scale
    lap = (
        mx.roll(c, 1, axis=1)
        + mx.roll(c, -1, axis=1)
        + mx.roll(c, 1, axis=2)
        + mx.roll(c, -1, axis=2)
        - 4.0 * c
    )
    return mx.clip(keep * (c + diff * lap), 0.0, fi.chem_max)


def update_resource(
    resource: mx.array,
    patch_pos: mx.array,
    cfg: Config,
    step: mx.array | None = None,
    env: Env | None = None,
) -> tuple[mx.array, mx.array]:
    """Regenerate then integer-diffuse the resource: ``(resource', inc (FG,FG) int32)`` (§13.2, §10.2).

    ``inc = clip(min(regen_rate, resource_cap − R), 0, ∞)`` (``INJECT_REGEN = Σ inc``), so regeneration stops
    at ``resource_cap``; the exact flux stencil ``ops.int_diffuse`` then conserves the total. ``step`` (the
    step index during the step) selects the seasonal weights; it is ignored with seasons off. ``env`` is the
    director's live levers (None when the director is off).
    """
    fi = cfg.fields
    inc = mx.maximum(mx.minimum(regen_rate(patch_pos, cfg, step, env), fi.resource_cap - resource), 0)
    return ops.int_diffuse(resource + inc, fi.resource_diffusion_q8), inc


def drift_patches(patch_pos: mx.array, patch_vel: mx.array, cfg: Config, env: Env | None = None) -> mx.array:
    """``wrap(patch_pos + patch_vel)`` into ``[0, L)`` (§13.1); ``env`` scales the drift."""
    if env is not None:  # static: director enabled
        patch_vel = patch_vel * env.drift_mult
    return ops.wrap(patch_pos + patch_vel, cfg.world.size)
