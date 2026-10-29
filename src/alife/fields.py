"""Lattice fields: fertility patches, resource and chemicals (§13).

``init_patches``, ``fertility`` and ``regen_rate`` are WP0 hand-over functions; WP1 owns this module.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, NamedTuple

import mlx.core as mx
import numpy as np

from alife import fields_sph, layout, ops, sphere, weather
from alife.config import Config, dims, seasons_on

if TYPE_CHECKING:  # state imports this module; the annotation alone must not create an import cycle
    from alife.state import Env


class FieldObs(NamedTuple):
    res: mx.array  # (N,)
    res_grad: mx.array  # (N,2) body frame (fwd, lat), normalised; 3D (N,3) (fwd, lat, up) (§27.6)
    chem: mx.array  # (N,NCH)
    chem_grad: mx.array  # (N,NCH,2) body frame, normalised; 3D (N,NCH,3)


def init_patches(key_pos: mx.array, key_vel: mx.array, cfg: Config) -> tuple[mx.array, mx.array]:
    """Patch centres ``U[0, L)²`` and constant drifts ``patch_drift·(cos ψ, sin ψ)``, ψ ~ U[0, 2π).

    On a sphere (§46.5, ``fields_sph.init_patches``): unit-vector centres (PP,3), angular velocities (PP,3).
    """
    if dims(cfg).sph:  # static: the sphere world
        return fields_sph.init_patches(key_pos, key_vel, cfg)
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
    whose floor entry is exactly 1, so layer 0 equals the 2D fertility bit for bit. On a sphere (§46.5) it is
    (FZ,Npix): the chord-Gaussian of the unit-vector patches times the profile (``fields_sph.fertility``),
    with the same seasonal and director weights.
    """
    if dims(cfg).sph:  # static: the sphere world
        w = None
        if step is not None and env is not None:  # static: director enabled
            w = env.patch_gain * season_weights(step, cfg, env)
        elif step is not None and seasons_on(cfg.fields):  # static: seasons enabled
            w = season_weights(step, cfg)
        return fields_sph.fertility(patch_pos, w, cfg)
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
    On a sphere it is (FZ,1), evaluated at the layers' lower interface heights ``z_k`` (§46.5).
    """
    if dims(cfg).sph:  # static: the sphere world
        return fields_sph.profile(cfg)
    d, fi, fc = dims(cfg), cfg.fields, cfg.world.field_cell
    v = [math.exp(-k * fc / fi.benthic_lambda) for k in range(d.fz)]
    if fi.photic_amp > 0:  # static: the photic band is on
        v = [b + fi.photic_amp * math.exp(-(d.fz - 1 - k) * fc / fi.photic_lambda) for k, b in enumerate(v)]
    return mx.array([min(1.0, x) for x in v], dtype=mx.float32).reshape(d.fz, 1, 1)


def regen_rate(
    patch_pos: mx.array,
    cfg: Config,
    step: mx.array | None = None,
    env: Env | None = None,
    rain: weather.RainIn | None = None,
) -> mx.array:
    """Per-cell regeneration ``floor(regen_max·fertility)``, (FG,FG) int32 (seasonal at ``step`` if given).

    With ``env`` the product is scaled by ``env.regen_mult`` after ``regen_max·fertility``, so 1.0 is exact.
    In 3D it has the fertility's shape (FZ,FG,FG) (§27.5). With ``rain`` (the weather's, None when weather is
    off or ``rain_gain`` is 0) the integer ``weather.rain_regen`` is added after the floor: the director's
    regeneration lever does not scale it (§30).
    """
    rate = cfg.fields.regen_max * fertility(patch_pos, cfg, step, env)
    if env is not None:  # static: director enabled
        rate = rate * env.regen_mult
    base = ops.floor_i32(rate)
    if rain is not None:  # static: rain feeds the resource
        assert not dims(cfg).sph, "the weather's rain is not defined on the sphere before S4"
        return base + weather.rain_regen(rain, cfg)
    return base


def cell_index(pos: mx.array, cfg: Config) -> mx.array:
    """Flat field-cell index ``cy·FG + cx`` of each position, (N,) int32 (§13).

    ``% FG`` is a second seam guard: a position that rounds to ``L`` maps to cell 0. In 3D the index is
    z-major, ``(cz·FG + cy)·FG + cx`` (``_cell_index_3d``, §27.5).
    """
    if dims(cfg).sph:  # static: the sphere world, layer·Npix + pixel (§46.1)
        return sphere.geometry(cfg).cell_of(pos)
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


def _nbr_cells_3d(fcell: mx.array, fg: int, fz: int) -> tuple[mx.array, ...]:
    """Flat z-major indices of the neighbour cells ``(x+1, x−1, y+1, y−1, z+1, z−1)`` (§27.6).

    x and y are periodic; z is clamped to ``[0, FZ − 1]``, so a boundary layer is its own outer neighbour (the
    mirror of the no-flux boundary).
    """
    fg2 = fg * fg
    cz, cyx = fcell // fg2, fcell % fg2
    cy, cx = cyx // fg, cyx % fg
    xp, xm = (cx + 1) % fg, (cx - 1 + fg) % fg
    yp, ym = (cy + 1) % fg, (cy - 1 + fg) % fg
    zp, zm = mx.minimum(cz + 1, fz - 1), mx.maximum(cz - 1, 0)
    row = cz * fg2 + cy * fg
    return (
        row + xp,
        row + xm,
        cz * fg2 + yp * fg + cx,
        cz * fg2 + ym * fg + cx,
        zp * fg2 + cyx,
        zm * fg2 + cyx,
    )


def sample_fields(
    resource: mx.array,
    chem: mx.array,
    fcell: mx.array,
    heading: mx.array,
    cfg: Config,
    *,
    pitch: mx.array | None = None,
    pos: mx.array | None = None,
    hdg: mx.array | None = None,
) -> FieldObs:
    """Field inputs at each agent's start-of-step cell, already normalised as in §5.

    - ``res``: ``min(R / resource_cap, RES_SAT) / RES_SAT`` (``IN_RES``).
    - ``res_grad``: ``clip(RES_GRAD_GAIN·to_body(g), −1, 1)`` with the central difference
      ``g = (R[y,x+1] − R[y,x−1], R[y+1,x] − R[y−1,x]) / (2·resource_cap)`` (``IN_RES_GRAD``).
    - ``chem``: ``c / (c + chem_half_sat)`` per channel (``IN_CHEM``).
    - ``chem_grad``: ``tanh(to_body(central difference / 2) / chem_half_sat)`` per channel (``IN_CHEM_GRAD``).

    Dead rows get stale values (from their stale ``pos``); consumers mask by ``alive``. In 3D the gradients
    gain a z difference and are rotated by yaw and ``pitch`` (required) into (fwd, lat, up)
    (``_sample_fields_3d``, §27.6). On a sphere (§46.5) ``pos`` (N,3) and the heading vector ``hdg`` (N,3)
    are required (the frame is built from them; ``heading`` is not read) and the gradients are (fwd, lat, up).
    """
    if dims(cfg).sph:  # static: the sphere world
        assert pos is not None and hdg is not None and pitch is not None, (
            "sphere sampling needs pos, hdg, pitch"
        )
        return fields_sph.sample_fields(resource, chem, fcell, pos, hdg, pitch, cfg)
    if dims(cfg).d3:
        return _sample_fields_3d(resource, chem, fcell, heading, pitch, cfg)
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


def _sample_fields_3d(
    resource: mx.array, chem: mx.array, fcell: mx.array, heading: mx.array, pitch: mx.array, cfg: Config
) -> FieldObs:
    """``sample_fields`` on the (FZ,FG,FG) lattice (§27.6): the same §5 normalisation, with a z difference.

    ``g_z = (R[z+1] − R[z−1]) / (2·resource_cap)`` with the z index clamped and the divisor kept at 2 (the
    mirror of the no-flux boundary, so a boundary layer sees half its one-sided slope); likewise for the
    chemicals. The 3-vectors are rotated into (fwd, lat, up) by ``ops.to_body3``: ``res_grad`` (N,3) and
    ``chem_grad`` (N,NCH,3).
    """
    assert pitch is not None, "3D field sampling needs the pitch"  # static
    d = dims(cfg)
    fi, half = cfg.fields, cfg.sensing.chem_half_sat
    cos_h, sin_h = mx.cos(heading), mx.sin(heading)
    cos_p, sin_p = mx.cos(pitch), mx.sin(pitch)
    xp, xm, yp, ym, zp, zm = _nbr_cells_3d(fcell, d.fg, d.fz)

    r = resource.reshape(-1).astype(mx.float32)  # R ≤ R_HARD < 2^24: exact in f32
    res = mx.minimum(r[fcell] / fi.resource_cap, layout.RES_SAT) / layout.RES_SAT
    g = mx.stack([r[xp] - r[xm], r[yp] - r[ym], r[zp] - r[zm]], axis=-1) / (2.0 * fi.resource_cap)
    res_grad = mx.clip(layout.RES_GRAD_GAIN * ops.to_body3(g, cos_h, sin_h, cos_p, sin_p), -1.0, 1.0)

    c = chem.reshape(layout.NCH, d.n_cells)
    cv = c[:, fcell].T  # (N,NCH)
    cg = mx.stack([(c[:, xp] - c[:, xm]).T, (c[:, yp] - c[:, ym]).T, (c[:, zp] - c[:, zm]).T], axis=-1) * 0.5
    rot = ops.to_body3(cg, cos_h[:, None], sin_h[:, None], cos_p[:, None], sin_p[:, None])
    return FieldObs(res=res, res_grad=res_grad, chem=cv / (cv + half), chem_grad=mx.tanh(rot / half))


def deposit_chem(emit: mx.array, fcell: mx.array, active: mx.array, cfg: Config) -> mx.array:
    """Integer chemical deposit quanta per channel and cell, (NCH·n_cells,) int32 (§13.3).

    Each active agent adds ``floor(emit_c·256)`` quanta at flat index ``c·n_cells + fcell`` (int scatter-add,
    so the result is independent of atomic order). ``n_cells`` is FG² in 2D and FZ·FG² in 3D (§27.6).
    """
    n_cells = dims(cfg).n_cells
    quanta = mx.maximum(ops.floor_i32(emit * 256.0), 0)  # (N,NCH); emit ∈ [0, 1] → [0, 256]
    flat = fcell[:, None] + mx.arange(layout.NCH, dtype=mx.int32) * n_cells  # (N,NCH)
    mask = mx.broadcast_to(active[:, None], flat.shape)
    return ops.scatter_add_i32(layout.NCH * n_cells, flat, quanta, mask)


def update_chem(chem: mx.array, dep_q: mx.array, cfg: Config) -> mx.array:
    """Deposit, then diffuse and decay, then clamp the chemical channels (§13.3).

    ``chem_c ← clip((1 − λ_c)·(chem_c + D_c·(roll₄(chem_c) − 4·chem_c)), 0, chem_max)`` after
    ``chem += dep_q·chem_deposit_scale``. 3D: ``_update_chem_3d`` (§27.6).
    """
    if dims(cfg).sph:  # static: the sphere world, sub-stepped (§46.5)
        return fields_sph.update_chem(chem, dep_q, cfg)
    if dims(cfg).d3:
        return _update_chem_3d(chem, dep_q, cfg)
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


def _update_chem_3d(chem: mx.array, dep_q: mx.array, cfg: Config) -> mx.array:
    """``update_chem`` on the (NCH,FZ,FG,FG) lattice with a 7-point stencil, no flux in z (§27.6).

    x and y are rolled (periodic); z uses mirrored ghost layers (``c[−1] = c[0]``, ``c[FZ] = c[FZ−1]``), so a
    boundary layer has the 5-point stencil of its four x/y neighbours and its one z neighbour, nothing crosses
    the floor or the surface, and diffusion conserves mass. ``chem_diffusion ≤ 1/6`` keeps it stable (§27.1).
    """
    fi, d = cfg.fields, dims(cfg)
    diff = mx.array(fi.chem_diffusion, dtype=mx.float32).reshape(layout.NCH, 1, 1, 1)
    keep = mx.array([1.0 - lam for lam in fi.chem_decay], dtype=mx.float32).reshape(layout.NCH, 1, 1, 1)
    c = chem + dep_q.reshape(layout.NCH, d.fz, d.fg, d.fg).astype(mx.float32) * fi.chem_deposit_scale
    c_up = mx.concatenate([c[:, 1:], c[:, -1:]], axis=1)  # the layer above; the top layer mirrors itself
    c_dn = mx.concatenate([c[:, :1], c[:, :-1]], axis=1)  # the layer below; the floor layer mirrors itself
    lap = (
        mx.roll(c, 1, axis=2)
        + mx.roll(c, -1, axis=2)
        + mx.roll(c, 1, axis=3)
        + mx.roll(c, -1, axis=3)
        + c_up
        + c_dn
        - 6.0 * c
    )
    return mx.clip(keep * (c + diff * lap), 0.0, fi.chem_max)


def update_resource(
    resource: mx.array,
    patch_pos: mx.array,
    cfg: Config,
    step: mx.array | None = None,
    env: Env | None = None,
    rain: weather.RainIn | None = None,
) -> tuple[mx.array, mx.array]:
    """Regenerate then integer-diffuse the resource: ``(resource', inc (FG,FG) int32)`` (§13.2, §10.2).

    ``inc = clip(min(regen_rate, resource_cap − R), 0, ∞)`` (``INJECT_REGEN = Σ inc``), so regeneration stops
    at ``resource_cap``; the exact flux stencil ``ops.int_diffuse`` then conserves the total. ``step`` (the
    step index during the step) selects the seasonal weights; it is ignored with seasons off. ``env`` is the
    director's live levers (None when the director is off); ``rain`` is the weather's rain term for
    ``regen_rate`` (``INJECT_REGEN`` then books it, §30). In 3D both are (FZ,FG,FG) and the stencil is
    ``ops.int_diffuse_3d``, with no flux through the floor and the surface (§27.6).
    """
    fi = cfg.fields
    inc = mx.maximum(mx.minimum(regen_rate(patch_pos, cfg, step, env, rain), fi.resource_cap - resource), 0)
    if dims(cfg).sph:  # static: the sphere world, int64 flux with geometric weights (§46.5)
        return fields_sph.diffuse(resource + inc, cfg), inc
    if dims(cfg).d3:
        return ops.int_diffuse_3d(resource + inc, fi.resource_diffusion_q8), inc
    return ops.int_diffuse(resource + inc, fi.resource_diffusion_q8), inc


def update_resource_farm(
    resource: mx.array,
    patch_pos: mx.array,
    bonus: mx.array,
    cfg: Config,
    step: mx.array | None = None,
    env: Env | None = None,
    rain: weather.RainIn | None = None,
) -> tuple[mx.array, mx.array, mx.array]:
    """``update_resource`` with the farm bonus inside the clip: ``(resource', inc, inc0)`` (DESIGN §41).

    ``inc = clip(min(regen_rate + bonus, resource_cap − R), 0, ∞)``, so ``INJECT_REGEN = Σ inc`` books the
    bonus and the resource cap still binds; ``inc0`` is the same clip without the bonus, so that
    ``Σ(inc − inc0)`` is the yield of the soil after the cap (``FARM_YIELD``). ``bonus`` is an int32 array of
    the shape of the resource, >= 0; the director's ``regen_mult`` does not scale it (as the rain)."""
    fi = cfg.fields
    base = regen_rate(patch_pos, cfg, step, env, rain)
    room = fi.resource_cap - resource
    inc0 = mx.maximum(mx.minimum(base, room), 0)
    inc = mx.maximum(mx.minimum(base + bonus, room), 0)
    if dims(cfg).sph:  # static: the sphere world (§46.5)
        return fields_sph.diffuse(resource + inc, cfg), inc, inc0
    if dims(cfg).d3:
        return ops.int_diffuse_3d(resource + inc, fi.resource_diffusion_q8), inc, inc0
    return ops.int_diffuse(resource + inc, fi.resource_diffusion_q8), inc, inc0


def drift_patches(patch_pos: mx.array, patch_vel: mx.array, cfg: Config, env: Env | None = None) -> mx.array:
    """``wrap(patch_pos + patch_vel)`` into ``[0, L)`` (§13.1); ``env`` scales the drift. On a sphere the
    centre rotates rigidly about its angular velocity (§46.5)."""
    if dims(cfg).sph:  # static: the sphere world
        return fields_sph.drift_patches(patch_pos, patch_vel, cfg, env)
    if env is not None:  # static: director enabled
        patch_vel = patch_vel * env.drift_mult
    return ops.wrap(patch_pos + patch_vel, cfg.world.size)


# --- built material (DESIGN section 38): the grid State.mat on the food lattice ------------------------

_BIG: int = 1 << 30  # the "missing neighbour" value of a min pass (never the minimum)


def mat_phases(fg: int, nd: int) -> tuple[int, int]:
    """``(px, n_phase)`` of the R2 colouring (§38.4b).

    ``px`` is the smallest divisor of ``FG`` that is at least 3, so the colour ``cx mod px`` is
    consistent across the periodic wrap; ``n_phase`` is ``px²`` in 2D and ``3·px²`` in 3D (z is never
    wrapped and cycles through 3). Raises ``ValueError`` when ``FG`` is below 4 or prime (no short
    colouring cycle); ``validate`` reports it as a config error.
    """
    if fg < 4 or all(fg % q for q in range(2, math.isqrt(fg) + 1)):
        raise ValueError(f"the field lattice side FG = {fg} is below 4 or prime: no R2 colouring")
    px = next(p for p in range(3, fg + 1) if fg % p == 0)
    return px, px * px * (3 if nd == 3 else 1)


def mat_colour(cfg: Config) -> mx.array:
    """The R2 colour of every cell, ``field_shape`` int32.

    ``cx mod px + px·(cy mod px) + px²·(cz mod 3)`` (no z term in 2D); a constant of the trace, built on
    the host from the config.
    """
    d = dims(cfg)
    px, _ = mat_phases(d.fg, d.nd)
    c = np.indices(d.field_shape)
    code = c[-1] % px + px * (c[-2] % px)
    if d.d3:
        code = code + px * px * (c[0] % 3)
    return mx.array(code.astype(np.int32))


def mat_thr(cfg: Config) -> int:
    """The closed threshold in units: a cell is OPEN iff ``m < ceil(open_frac·cap)`` (exact for ints)."""
    return math.ceil(cfg.structures.open_frac * cfg.structures.cap)


def mat_k(cfg: Config) -> int:
    """R2's required open neighbours ``ceil(aperture·n_nbr)`` (8 or 26); 0 when ``aperture`` is 0.

    A 1e-9 guard keeps a product that is exact in decimal (0.25 · 8) from rounding up.
    """
    sc = cfg.structures
    if sc.aperture <= 0:
        return 0
    return math.ceil(sc.aperture * (26 if dims(cfg).d3 else 8) - 1e-9)


def _face_neighbours(a: mx.array, d3: bool) -> tuple[mx.array, ...]:
    """``(x+1, x-1, y+1, y-1[, z+1, z-1])`` of a lattice, each built exactly once per array.

    MLX 0.32 fails to compile a graph that builds one roll of one array twice (``weather``). x and y
    are periodic; z is mirrored (the boundary layer is its own outer neighbour, as ``_nbr_cells_3d``).
    """
    ax_y, ax_x = (1, 2) if d3 else (0, 1)
    out = [
        mx.roll(a, -1, axis=ax_x),
        mx.roll(a, 1, axis=ax_x),
        mx.roll(a, -1, axis=ax_y),
        mx.roll(a, 1, axis=ax_y),
    ]
    if d3:
        out += [mx.concatenate([a[1:], a[-1:]], axis=0), mx.concatenate([a[:1], a[:-1]], axis=0)]
    return tuple(out)


def mat_columns(cfg: Config) -> tuple[str, ...]:
    """The static column set of ``mat_table``: ``here`` always; ``cover`` and ``grad`` when used."""
    sc = cfg.structures
    cols = ["here"]
    if sc.reach > 0 and (sc.shield_gain > 0 or (sc.shelter_gain > 0 and cfg.weather.enabled)):
        cols.append("cover")
    if sc.barrier_gain > 0 and sc.exit_ease > 0:
        cols.append("grad")
    return tuple(cols)


def mat_table(m: mx.array, cfg: Config) -> mx.array:
    """The per-cell table the agents gather once, ``(n_cells, C)`` f32 (§38.4).

    Columns (``mat_columns``): ``here = m/cap``; ``cover = max(m, reach·max over the face
    neighbours)/cap`` (with ``reach = 0`` it is ``here`` and is not built); the central-difference
    gradient of ``m/cap``, ``nd`` columns with divisor 2 as ``sample_fields`` (only with a positive
    ``barrier_gain`` and ``exit_ease``).
    """
    sc, d = cfg.structures, dims(cfg)
    cols = mat_columns(cfg)
    inv = 1.0 / sc.cap

    def flat(a: mx.array) -> mx.array:
        return a.astype(mx.float32).reshape(-1)

    out = [flat(m) * inv]
    if "cover" in cols or "grad" in cols:
        xp, xm, yp, ym, *z = _face_neighbours(m, d.d3)
    if "cover" in cols:
        nmax = mx.maximum(mx.maximum(xp, xm), mx.maximum(yp, ym))
        for a in z:
            nmax = mx.maximum(nmax, a)
        out.append(mx.maximum(flat(m), sc.reach * flat(nmax)) * inv)
    if "grad" in cols:
        half = 0.5 * inv
        out += [flat(xp - xm) * half, flat(yp - ym) * half]
        if d.d3:
            out.append(flat(z[0] - z[1]) * half)
    return mx.stack(out, axis=1)


def _box_sum(a: mx.array, d3: bool) -> mx.array:
    """Sum over the 3x3(x3) box including the centre of an int32 lattice (missing z layer counts 0).

    Separable: one pass per axis, each on a fresh array.
    """
    ax_y, ax_x = (1, 2) if d3 else (0, 1)
    for ax in (ax_x, ax_y):
        a = a + mx.roll(a, 1, axis=ax) + mx.roll(a, -1, axis=ax)
    if d3:
        zero = mx.zeros_like(a[:1])
        a = a + mx.concatenate([zero, a[:-1]], axis=0) + mx.concatenate([a[1:], zero], axis=0)
    return a


def _box_min(a: mx.array, d3: bool) -> mx.array:
    """Minimum over the 3x3(x3) box including the centre; a missing z layer counts as ``_BIG``."""
    ax_y, ax_x = (1, 2) if d3 else (0, 1)
    for ax in (ax_x, ax_y):
        a = mx.minimum(a, mx.minimum(mx.roll(a, 1, axis=ax), mx.roll(a, -1, axis=ax)))
    if d3:
        big = mx.full(a[:1].shape, _BIG, dtype=a.dtype)
        above = mx.concatenate([a[1:], big], axis=0)
        below = mx.concatenate([big, a[:-1]], axis=0)
        a = mx.minimum(a, mx.minimum(below, above))
    return a


def mat_nopen(m: mx.array, cfg: Config) -> mx.array:
    """``nopen[c]``: the OPEN cells (``m < thr``) among the neighbours of ``c`` (centre excluded), int32.

    The full neighbourhood has 8 cells (26 in 3D; a floor or surface cell has 17); only cells that
    exist are counted.
    """
    op = (m < mat_thr(cfg)).astype(mx.int32)
    return _box_sum(op, dims(cfg).d3) - op


class MatUpdate(NamedTuple):
    mat: mx.array  # field_shape int32: the new material
    wasted: mx.array  # () int32: units lost to the cap and to refused closings (sum of pre - m1)
    decayed: mx.array  # () int32: units lost to decay (sum of loss)
    nopen: mx.array  # field_shape int32: the open-neighbour counts of the START-of-step material


def update_mat(m: mx.array, dep: mx.array, step: mx.array, cfg: Config) -> MatUpdate:
    """Deposit, cap, gate (R2), decay: the material of the next step (§38.2 to §38.4b).

    ``pre = m + dep`` and ``m1 = min(pre, cap)``. With ``aperture > 0`` a cell that would turn from
    OPEN to CLOSED (``m < thr <= m1``) may do so only when it is active in this step's colour phase
    and every cell of its box has at least ``k + 1`` open neighbours (``box_min(nopen) >= k + 1``);
    otherwise it stays at ``thr - 1``. Then ``loss = (m1·decay_q12 + 4095) >> 12`` and
    ``m' = m1 - loss``, all exact integers. ``wasted = Σ(pre - m1)`` and ``decayed = Σ loss`` give
    the identity ``Σm' = Σm + Σdep - wasted - decayed``.
    """
    sc, d = cfg.structures, dims(cfg)
    thr, k = mat_thr(cfg), mat_k(cfg)
    nopen = mat_nopen(m, cfg)
    pre = m + dep
    m1 = mx.minimum(pre, sc.cap)
    if k > 0:  # static: R2 (§38.4b)
        _, n_phase = mat_phases(d.fg, d.nd)
        slack = _box_min(nopen, d.d3) >= k + 1
        active = mat_colour(cfg) == (step % n_phase)
        want_close = (m < thr) & (m1 >= thr)
        m1 = mx.where(want_close & ~(active & slack), thr - 1, m1)
    loss = (m1 * sc.decay_q12 + 4095) >> 12
    return MatUpdate(m1 - loss, (pre - m1).sum(), loss.sum(), nopen)
