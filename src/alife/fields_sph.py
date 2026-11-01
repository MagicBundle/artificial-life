"""The fields of the sphere world (device, pure MLX; DESIGN §46.5, true-sphere series S1).

The resource, the chemicals and the fertility patches on the shell ``(FZ layers) × (12·Nf² angular cells)``:

* **fertility** from patch centres that are unit vectors (chord distance in length units), times the vertical
  profile at the layers' lower interface heights; patches **rotate rigidly** about a stored angular velocity
  (a small circle for ever);
* the **resource** step: integer regeneration, then ``ops.int_diffuse_sph`` (int64 flux, the symmetric
  geometric pair weights of ``fieldtab``), exactly conserving;
* the **chemical** update: the same pair weights as float conductances, **sub-stepped** statically so that the
  explicit scheme is stable on the thin outer layers, conserving mass to rounding without decay;
* the **agent inputs**: the cell values and the least-squares tangent gradient plus the radial difference,
  projected on the body frame (``ops.to_body_s``), scaled so a box and a sphere feed the brain the same
  magnitudes.

The tables are host constants (``fieldtab``, cached per shell); every kernel takes them as arguments, and its
scalar parameters may be 0-d ``mx.array`` values, which is how a *compiled* test passes them (no printable
constant for the compile cache to confuse, DESIGN F21 item 6). Nothing here is wired into the step before S2:
``alife.fields`` dispatches to this module when ``dims(cfg).sph``.
"""

from __future__ import annotations

import functools
import math
from typing import TYPE_CHECKING, Any, NamedTuple

import mlx.core as mx
import numpy as np

from alife import fieldtab, geom, layout, ops

if TYPE_CHECKING:
    from alife.config import Config
    from alife.fields import FieldObs
    from alife.state import Env

_F32 = mx.float32
_I32 = mx.int32


# --- the cached tables of a config -----------------------------------------------------------------


class DiffTables(NamedTuple):
    """Device tables of the integer resource diffusion (``ops.int_diffuse_sph``)."""

    nbr: mx.array  # (Npix, 4) int32
    a_t: mx.array  # (FZ, Npix, 4) int32: d_q8·w
    a_z: mx.array  # (FZ-1, 1) int32


class ChemTables(NamedTuple):
    """Device tables of the chemical update."""

    nbr: mx.array  # (Npix, 4) int32
    c_t: mx.array  # (FZ, Npix, 4) float32: w / 256
    c_z: mx.array  # (FZ-1, 1) float32
    n_sub: int  # static number of sub-steps


class GradTables(NamedTuple):
    """Device tables of the field gradient."""

    nbr: mx.array  # (Npix, 4) int32
    w: mx.array  # (Npix, 4, 3) float32: tangent weights per radian
    inv_r: mx.array  # (FZ,) float32: 1 / layer radius
    dz2: mx.array  # (FZ,) float32: radial divisor of the central difference


@functools.cache
def diffusion_tables(cfg: Config) -> DiffTables:
    """The integer diffusion tables of ``cfg``: the pair weights times ``resource_diffusion_q8``."""
    c = fieldtab.conductance(*fieldtab.shell(cfg))
    d = cfg.fields.resource_diffusion_q8
    return DiffTables(
        nbr=mx.array(c.nbr, dtype=_I32),
        a_t=mx.array(c.wt.astype(np.int32) * d, dtype=_I32),
        a_z=mx.array((c.wz.astype(np.int32) * d).reshape(-1, 1), dtype=_I32),
    )


@functools.cache
def chem_tables(cfg: Config) -> ChemTables:
    """The chemical conductances ``w / 256`` and the sub-step count for the largest ``chem_diffusion``."""
    c = fieldtab.conductance(*fieldtab.shell(cfg))
    return ChemTables(
        nbr=mx.array(c.nbr, dtype=_I32),
        c_t=mx.array(c.wt.astype(np.float64) / 256.0, dtype=_F32),
        c_z=mx.array((c.wz.astype(np.float64) / 256.0).reshape(-1, 1), dtype=_F32),
        n_sub=c.chem_substeps(max(cfg.fields.chem_diffusion)),
    )


@functools.cache
def grad_tables(cfg: Config) -> GradTables:
    g = fieldtab.gradient(*fieldtab.shell(cfg))
    return GradTables(
        nbr=mx.array(fieldtab.conductance(*fieldtab.shell(cfg)).nbr, dtype=_I32),
        w=mx.array(g.w, dtype=_F32),
        inv_r=mx.array(g.inv_r, dtype=_F32),
        dz2=mx.array(g.dz2, dtype=_F32),
    )


@functools.cache
def cell_centres(nside: int) -> mx.array:
    """(Npix, 3) float32 unit vectors of the pixel centres (host float64, cast)."""
    return mx.array(geom.pixel_centres(nside), dtype=_F32)


@functools.cache
def profile(cfg: Config) -> mx.array:
    """The vertical food profile ``V`` of the layers, (FZ, 1) f32 (``fieldtab.vertical_profile``)."""
    fi = cfg.fields
    v = fieldtab.vertical_profile(
        *fieldtab.shell(cfg)[:2], cfg.world.layers, fi.benthic_lambda, fi.photic_amp, fi.photic_lambda
    )
    return mx.array(v.reshape(-1, 1), dtype=_F32)


# --- patches ---------------------------------------------------------------------------------------


def sphere_point(a: mx.array, b: mx.array) -> mx.array:
    """The point of the unit sphere of two uniform draws in [0, 1): ``cos θ = 2a − 1``, ``λ = 2π b`` (equal
    area by Archimedes), (..., 3) float32. The map of the founders and the patches (DESIGN §46.5)."""
    cos_t = 2.0 * a - 1.0
    sin_t = mx.sqrt(mx.maximum(1.0 - cos_t * cos_t, 0.0))
    lam = 2.0 * math.pi * b
    return mx.stack([sin_t * mx.cos(lam), sin_t * mx.sin(lam), cos_t], axis=-1)


def init_patches(key_pos: mx.array, key_vel: mx.array, cfg: Config) -> tuple[mx.array, mx.array]:
    """``(patch_pos (PP,3), patch_vel (PP,3))``: centres uniform on the sphere from one ``(PP, 2)`` uniform
    draw of ``key_pos`` (``sphere_point``), and the drift as an angular velocity ``ω = (û × t̂)·patch_drift /
    Rp`` with ``t̂ = cos ψ·east + sin ψ·north``, ``ψ ~ U[0, 2π)`` from ``key_vel`` (the box's draw). ``ω ⟂ û``
    and ``|ω|·Rp`` is the drift speed on the floor sphere."""
    pp = cfg.fields.n_patches
    u = mx.random.uniform(0.0, 1.0, (pp, 2), key=key_pos)
    pos = sphere_point(u[:, 0], u[:, 1])
    psi = mx.random.uniform(0.0, 2.0 * math.pi, (pp,), key=key_vel)
    east, north = ops.local_frame_s(pos)
    t = mx.cos(psi)[:, None] * east + mx.sin(psi)[:, None] * north
    om = ops.cross3(pos, t)
    return pos, om * (cfg.fields.patch_drift / cfg.world.radius)


def rotate(p: mx.array, om: mx.array) -> mx.array:
    """The rigid rotation of unit vectors ``p`` (..., 3) about the angular velocity ``om`` (..., 3) by
    ``|om|`` radians (Rodrigues with series coefficients: the truncation is below 3e-11 for ``|om| <= 0.05``
    and 2e-9 for ``|om| <= 0.1``, under the float32 rounding):
    ``c·p + s·(ω × p) + g·(ω·p)·ω`` with ``c = 1 − σ²/2 + σ⁴/24``, ``s = 1 − σ²/6 + σ⁴/120``,
    ``g = 1/2 − σ²/24 + σ⁴/720``, then renormalised. For ``ω ⟂ p`` (always, here) the third term is rounding
    noise that keeps the rotation axis exact."""
    s2 = (om * om).sum(axis=-1, keepdims=True)
    c = 1.0 - s2 * (0.5 - s2 / 24.0)
    s = 1.0 - s2 * (1.0 / 6.0 - s2 / 120.0)
    g = 0.5 - s2 * (1.0 / 24.0 - s2 / 720.0)
    cross = ops.cross3(om, p)
    along = (om * p).sum(axis=-1, keepdims=True)
    q = c * p + s * cross + g * along * om
    return q / mx.sqrt((q * q).sum(axis=-1, keepdims=True))


def drift_patches(patch_pos: mx.array, patch_vel: mx.array, cfg: Config, env: Env | None = None) -> mx.array:
    """The patch centres one step later: ``rotate(patch_pos, patch_vel)``; ``env.drift_mult`` scales ω."""
    if env is not None:  # static: director enabled
        patch_vel = patch_vel * env.drift_mult
    return rotate(patch_pos, patch_vel)


# --- fertility -------------------------------------------------------------------------------------


def fertility_cells(
    patch_pos: mx.array,
    weights: mx.array | None,
    centres: mx.array,
    radius: Any,
    sigma: Any,
    base: Any,
) -> mx.array:
    """``clip(base + Σ_p w_p·exp(−d²_{p,c}/2σ²), 0, 1)`` per angular cell, (Npix,) float32.

    ``d²`` is the squared **chord** between the patch centre and the cell centre in length units,
    ``Rp²·|û_c − û_p|²``: three squared differences and sums, elementwise on (PP, Npix), no matmul (F20) and
    no cancellation of ``2 − 2 cos``. ``weights`` is the per-patch factor (seasons, the director's gain) or
    None for all 1."""
    d = centres[None, :, :] - patch_pos[:, None, :]
    d2 = (d * d).sum(axis=-1) * (radius * radius)
    terms = mx.exp(d2 * (-1.0 / (2.0 * sigma * sigma)))
    if weights is not None:
        terms = weights[:, None] * terms
    return mx.clip(base + terms.sum(axis=0), 0.0, 1.0)


def fertility(patch_pos: mx.array, weights: mx.array | None, cfg: Config) -> mx.array:
    """The sphere's ``fields.fertility``: ``(FZ, Npix)`` float32, the cell fertility times the vertical
    profile ``V[k]`` (``V[0] = 1`` exactly, so layer 0 is the angular fertility)."""
    fi = cfg.fields
    cells = fertility_cells(
        patch_pos, weights, cell_centres(cfg.world.nside), cfg.world.radius, fi.patch_sigma, fi.fertility_base
    )
    return cells[None, :] * profile(cfg)


# --- the resource and the chemicals ----------------------------------------------------------------


def diffuse(resource: mx.array, cfg: Config) -> mx.array:
    """``ops.int_diffuse_sph`` with the tables of ``cfg`` (identity for ``resource_diffusion_q8 = 0``)."""
    if cfg.fields.resource_diffusion_q8 == 0:  # static
        return resource
    t = diffusion_tables(cfg)
    return ops.int_diffuse_sph(resource, t.nbr, t.a_t, t.a_z)


def chem_laplacian(c: mx.array, nbr: mx.array, c_t: mx.array, c_z: mx.array) -> mx.array:
    """``lap_i = Σ_j C_ij (c_j − c_i)`` over the 4 edge neighbours and the layers above and below, for
    ``c`` (NCH, FZ, Npix). The pair conductances are symmetric, so ``Σ_i lap_i = 0`` up to rounding; the top
    and bottom layers have no outer neighbour (no flux through the floor or the lid)."""
    lap = (c_t[None] * (mx.take(c, nbr, axis=2) - c[..., None])).sum(axis=3)
    flux = c_z[None] * (c[:, 1:] - c[:, :-1])  # from layer k+1 into k (and minus into k+1)
    zero = mx.zeros((c.shape[0], 1, c.shape[-1]), dtype=c.dtype)
    return lap + mx.concatenate([flux, zero], axis=1) - mx.concatenate([zero, flux], axis=1)


def update_chem_arrays(
    chem: mx.array,
    dep_q: mx.array,
    t: ChemTables,
    diff: mx.array,
    keep: mx.array,
    deposit_scale: Any,
    chem_max: Any,
) -> mx.array:
    """Deposit, then ``n_sub`` explicit sub-steps ``c ← c + (D_c/n_sub)·lap(c)``, then decay and clamp:
    ``clip(keep·c, 0, chem_max)``. ``chem`` is (NCH, FZ, Npix), ``dep_q`` the integer deposit of the same
    shape, ``diff`` and ``keep`` are (NCH, 1, 1). Each sub-step keeps every coefficient of the update
    non-negative (``n_sub = ceil(D·max Σ_j C_ij)``), the decay is applied once."""
    c = chem + dep_q.astype(_F32) * deposit_scale
    for _ in range(t.n_sub):  # static
        c = c + (diff / t.n_sub) * chem_laplacian(c, t.nbr, t.c_t, t.c_z)
    return mx.clip(keep * c, 0.0, chem_max)


def update_chem(chem: mx.array, dep_q: mx.array, cfg: Config) -> mx.array:
    """The sphere's ``fields.update_chem``: ``dep_q`` is the flat ``(NCH·n_cells,)`` deposit of
    ``fields.deposit_chem``; ``chem`` is (NCH, FZ, Npix)."""
    fi = cfg.fields
    diff = mx.array(fi.chem_diffusion, dtype=_F32).reshape(layout.NCH, 1, 1)
    keep = mx.array([1.0 - lam for lam in fi.chem_decay], dtype=_F32).reshape(layout.NCH, 1, 1)
    return update_chem_arrays(
        chem, dep_q.reshape(chem.shape), chem_tables(cfg), diff, keep, fi.chem_deposit_scale, fi.chem_max
    )


# --- the agent inputs ------------------------------------------------------------------------------


def _gradient(
    flat: mx.array, fcell: mx.array, t: GradTables, layers: int, npix: int, up: mx.array
) -> mx.array:
    """The gradient vector (planet axes, per unit length) of fields ``flat`` (..., FZ·Npix) at the cells
    ``fcell`` (N,) of the agents with local up ``up`` (N, 3): the tangent least-squares sum over the 4 edge
    neighbours at the layer's radius plus the radial central difference. Result (..., N, 3)."""
    k, p = fcell // npix, fcell % npix
    here = flat[..., fcell]  # (..., N)
    nb = flat[..., k[:, None] * npix + t.nbr[p]]  # (..., N, 4)
    gt = ((nb - here[..., None])[..., None] * t.w[p]).sum(axis=-2) * t.inv_r[k][:, None]  # (..., N, 3)
    gt = gt - (gt * up).sum(axis=-1, keepdims=True) * up  # onto the agent's own tangent plane
    kp, km = mx.minimum(k + 1, layers - 1), mx.maximum(k - 1, 0)
    gr = (flat[..., kp * npix + p] - flat[..., km * npix + p]) / t.dz2[k]  # (..., N)
    return gt + gr[..., None] * up


def sample_fields(
    resource: mx.array,
    chem: mx.array,
    fcell: mx.array,
    pos: mx.array,
    hdg: mx.array,
    pitch: mx.array,
    cfg: Config,
) -> FieldObs:
    """The sphere's ``fields.sample_fields``: the field inputs at each agent's start-of-step cell, already
    normalised as in §5, with the gradients in the body frame ``(fwd, lat, up)`` of ``ops.body_axes_s``.

    ``res`` and ``chem`` are as in the box. ``res_grad = clip(RES_GRAD_GAIN·g, −1, 1)`` and ``chem_grad =
    tanh(g / chem_half_sat)`` with ``g = ∇f·field_cell / scale`` (``resource_cap`` for the resource, 1 for a
    chemical): the box's central difference over ``2·field_cell`` is ``∇f·field_cell``, so a box and a sphere
    world feed the brain the same magnitudes. ``∇f`` is the tangent least-squares gradient of the 4 edge
    neighbours (``fieldtab.gradient``) plus ``(f_{k+1} − f_{k−1}) / (zc_{k+1} − zc_{k−1})`` on ``û`` with
    the box's mirrored boundary layers. Dead rows get stale values; consumers mask by ``alive``.
    """
    from alife.fields import FieldObs  # deferred: fields imports this module

    fi, half = cfg.fields, cfg.sensing.chem_half_sat
    fc = cfg.world.field_cell
    npix, layers = geom.npix(cfg.world.nside), cfg.world.layers
    t = grad_tables(cfg)
    up = pos / mx.sqrt((pos * pos).sum(axis=-1, keepdims=True))
    cos_p, sin_p = mx.cos(pitch), mx.sin(pitch)

    r = resource.reshape(-1).astype(_F32)  # R <= R_HARD < 2^24: exact in f32
    res = mx.minimum(r[fcell] / fi.resource_cap, layout.RES_SAT) / layout.RES_SAT
    g = _gradient(r, fcell, t, layers, npix, up) * (fc / fi.resource_cap)
    res_grad = mx.clip(layout.RES_GRAD_GAIN * ops.to_body_s(g, hdg, up, cos_p, sin_p), -1.0, 1.0)

    c = chem.reshape(layout.NCH, layers * npix)
    cv = c[:, fcell].T  # (N, NCH)
    cg = _gradient(c, fcell, t, layers, npix, up[None]) * fc  # (NCH, N, 3)
    cg = mx.transpose(cg, (1, 0, 2))  # (N, NCH, 3)
    chem_grad = mx.tanh(ops.to_body_s(cg, hdg[:, None], up[:, None], cos_p[:, None], sin_p[:, None]) / half)
    return FieldObs(res=res, res_grad=res_grad, chem=cv / (cv + half), chem_grad=chem_grad)
