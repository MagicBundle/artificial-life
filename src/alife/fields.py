"""Lattice fields: fertility patches, resource and chemicals (§13).

``init_patches``, ``fertility`` and ``regen_rate`` are WP0 hand-over functions; WP1 owns this module.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import mlx.core as mx

from alife import ops
from alife.config import Config, dims


class FieldObs(NamedTuple):
    res: mx.array  # (N,)
    res_grad: mx.array  # (N,2) body frame, normalised
    chem: mx.array  # (N,NCH)
    chem_grad: mx.array  # (N,NCH,2) body frame, normalised


def init_patches(key_pos: mx.array, key_vel: mx.array, cfg: Config) -> tuple[mx.array, mx.array]:
    """Patch centres ``U[0, L)²`` and constant drift velocities ``patch_drift·(cos ψ, sin ψ)``, ψ ~ U[0, 2π)."""
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
    """Flat field-cell index ``cy·FG + cx`` of each position, (N,) int32."""
    raise NotImplementedError


def sample_fields(resource: mx.array, chem: mx.array, fcell: mx.array, heading: mx.array, cfg: Config) -> FieldObs:
    """Resource and chemical values and body-frame gradients at each agent's cell (§5, §13)."""
    raise NotImplementedError


def deposit_chem(emit: mx.array, fcell: mx.array, active: mx.array, cfg: Config) -> mx.array:
    """Integer chemical deposit quanta per channel and cell, (NCH·FG·FG,) int32 (§13.3)."""
    raise NotImplementedError


def update_chem(chem: mx.array, dep_q: mx.array, cfg: Config) -> mx.array:
    """Deposit, diffuse, decay and clamp the chemical channels (§13.3)."""
    raise NotImplementedError


def update_resource(resource: mx.array, patch_pos: mx.array, cfg: Config) -> tuple[mx.array, mx.array]:
    """Regenerate then integer-diffuse the resource: ``(resource', inc (FG,FG) int32)`` (§13.2)."""
    raise NotImplementedError


def drift_patches(patch_pos: mx.array, patch_vel: mx.array, cfg: Config) -> mx.array:
    """``wrap(patch_pos + patch_vel)`` (§13.1)."""
    raise NotImplementedError
