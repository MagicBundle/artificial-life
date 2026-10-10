"""Lattice fields: fertility patches, resource and chemicals (§13).

``init_patches``, ``fertility`` and ``regen_rate`` are WP0 hand-over functions; WP1 owns this module.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import mlx.core as mx

from alife import layout, ops
from alife.config import Config, dims


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


def fertility(patch_pos: mx.array, cfg: Config) -> mx.array:
    """``clip(fertility_base + Σ_p torus-Gaussian_p, 0, 1)`` on the field grid, (FG,FG) f32 indexed [y, x].

    Separable: ``gyᵀ @ gx`` with per-axis minimum-image offsets from cell centres.
    """
    fg, L, fc = dims(cfg).fg, cfg.world.size, cfg.world.field_cell
    centres = (mx.arange(fg, dtype=mx.float32) + 0.5) * fc
    inv = -1.0 / (2.0 * cfg.fields.patch_sigma**2)
    dx = ops.min_image(centres[None, :] - patch_pos[:, 0:1], L)  # (PP,FG)
    dy = ops.min_image(centres[None, :] - patch_pos[:, 1:2], L)
    gx, gy = mx.exp(dx * dx * inv), mx.exp(dy * dy * inv)
    return mx.clip(cfg.fields.fertility_base + gy.T @ gx, 0.0, 1.0)


def regen_rate(patch_pos: mx.array, cfg: Config) -> mx.array:
    """Per-cell regeneration ``floor(regen_max·fertility)``, (FG,FG) int32."""
    return ops.floor_i32(cfg.fields.regen_max * fertility(patch_pos, cfg))


def cell_index(pos: mx.array, cfg: Config) -> mx.array:
    """Flat field-cell index ``cy·FG + cx`` of each position, (N,) int32 (§13).

    ``% FG`` is a second seam guard: a position that rounds to ``L`` maps to cell 0.
    """
    fg = dims(cfg).fg
    cxy = ops.floor_i32(pos / cfg.world.field_cell) % fg
    return cxy[:, 1] * fg + cxy[:, 0]


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


def update_resource(resource: mx.array, patch_pos: mx.array, cfg: Config) -> tuple[mx.array, mx.array]:
    """Regenerate then integer-diffuse the resource: ``(resource', inc (FG,FG) int32)`` (§13.2, §10.2).

    ``inc = clip(min(regen_rate, resource_cap − R), 0, ∞)`` (``INJECT_REGEN = Σ inc``), so regeneration stops
    at ``resource_cap``; the exact flux stencil ``ops.int_diffuse`` then conserves the total.
    """
    fi = cfg.fields
    inc = mx.maximum(mx.minimum(regen_rate(patch_pos, cfg), fi.resource_cap - resource), 0)
    return ops.int_diffuse(resource + inc, fi.resource_diffusion_q8), inc


def drift_patches(patch_pos: mx.array, patch_vel: mx.array, cfg: Config) -> mx.array:
    """``wrap(patch_pos + patch_vel)`` into ``[0, L)`` (§13.1)."""
    return ops.wrap(patch_pos + patch_vel, cfg.world.size)
