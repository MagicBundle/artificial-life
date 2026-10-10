"""Sensing: neighbour descriptors and the D_IN = 69 input vector (§5). Owned by WP2.

Everything is float32 and in the agent's body frame. Absent neighbours and empty bond slots give all-zero
blocks with ``present = 0``. Every input is finite with ``|x| ≤ 1``.
"""

from __future__ import annotations

import math

import mlx.core as mx

from alife import layout, ops
from alife.bonds import BondGeom
from alife.config import Config, dims
from alife.fields import FieldObs
from alife.spatial import Neighbors
from alife.state import State

_F32 = mx.float32


def _q0(cfg: Config) -> float:
    """Energy offset of the log ratios; ``max(basal_cost, 1)`` keeps them defined when ``basal_cost = 0``."""
    return float(max(cfg.energy.basal_cost, 1))


def _log_ratio(a: mx.array, b: mx.array, q0: float) -> mx.array:
    """``lr(a, b) = clip(log((a + q0)/(b + q0)) / LR_SCALE, −1, 1)`` for int32 energies."""
    ratio = (a.astype(_F32) + q0) / (b.astype(_F32) + q0)
    return mx.clip(mx.log(ratio) / layout.LR_SCALE, -1.0, 1.0)


def _sat(x: mx.array, scale: int) -> mx.array:
    """``min(x / scale, 1)`` for a non-negative int32 counter; ``scale`` 0 is read as 1."""
    return mx.minimum(x.astype(_F32) / float(max(scale, 1)), 1.0)


def _col(v: mx.array) -> mx.array:
    return v.astype(_F32)[:, None]


def _neighbor_mask(s: State, nbr: Neighbors) -> mx.array:
    """``nbr.valid`` ANDed with both ends' ``alive`` (a no-op under the §4 contract, kept as a guard)."""
    return nbr.valid & s.alive[:, None] & s.alive[nbr.idx]


def neighbor_descriptors(s: State, nbr: Neighbors, cfg: Config) -> mx.array:
    """Descriptor ``desc (N,K,NF)`` f32 for all K neighbours, zeroed where invalid (§5).

    Features: present, body-frame offset / R (dx, dy), kin similarity, ``lr(E_j, E_i)``, ``msg_j[0]`` and
    whether ``j`` is a current bond partner.
    """
    valid = _neighbor_mask(s, nbr)
    j = nbr.idx
    cos_h, sin_h = mx.cos(s.heading), mx.sin(s.heading)
    rel = mx.clip(ops.to_body(nbr.off, cos_h[:, None], sin_h[:, None]) / cfg.sensing.radius, -1.0, 1.0)
    dtag = s.tag[:, None, :] - s.tag[j]  # (N,K,T)
    kin = mx.exp(-(dtag * dtag).sum(-1) / (cfg.sensing.tag_scale**2))
    rele = _log_ratio(s.energy[j], s.energy[:, None], _q0(cfg))
    msg0 = s.msg[:, 0][j]
    bonded = (j[:, :, None] == s.partner[:, None, :]).any(-1)  # partner −1 never equals a row index
    desc = mx.stack(
        [
            mx.ones(j.shape, dtype=_F32),
            rel[..., 0],
            rel[..., 1],
            kin,
            rele,
            msg0,
            bonded.astype(_F32),
        ],
        axis=-1,
    )
    return mx.where(valid[..., None], desc, 0.0)


def build_inputs(
    s: State,
    age: mx.array,
    nbr: Neighbors,
    desc: mx.array,
    geom: BondGeom,
    fobs: FieldObs,
    key: mx.array,
    cfg: Config,
) -> mx.array:
    """Brain inputs ``x (N,D_IN)`` f32 with ``|x| ≤ 1`` (§5).

    ``desc`` comes from ``neighbor_descriptors``; ``geom`` holds start-of-step bond offsets for ``s.partner``;
    ``fobs`` holds rows 12–23 as computed by ``fields.sample_fields`` (§16.2), copied unchanged; ``key`` is
    ``ks[Stream.NOISE]``.
    """
    d, n = dims(cfg), s.alive.shape[0]
    q0, sc, en = _q0(cfg), cfg.sensing, cfg.energy
    cos_h, sin_h = mx.cos(s.heading), mx.sin(s.heading)

    # --- self (0..11) ---
    energy = mx.clip(mx.log1p(s.energy.astype(_F32) / q0) / math.log1p(en.e_max / q0), 0.0, 1.0)
    age_in = mx.tanh(age.astype(_F32) / layout.AGE_SCALE)
    vel = mx.clip(ops.to_body(s.vel, cos_h, sin_h) / cfg.physics.v_max, -1.0, 1.0)
    occupied = s.partner >= 0
    nbonds = occupied.sum(1).astype(_F32) / layout.B
    hurt = _sat(s.last_loss, cfg.combat.attack_max)
    fed = _sat(s.last_harvest, en.harvest_max)
    gift = _sat(s.last_inflow, cfg.bonds.share_rate)
    noise = mx.random.uniform(-1.0, 1.0, (n,), key=key)
    # Phase from the integer age modulo the period: exact for any age, unlike 2π·age/P in f32.
    phase = mx.remainder(age, sc.clock_period).astype(_F32) * (2.0 * math.pi / sc.clock_period)

    # --- fields (12..23): sample_fields already returns the final §5 values (§16.2) ---
    chem_grad = fobs.chem_grad.astype(_F32).reshape(n, 2 * layout.NCH)  # (c0 fwd, c0 lat, c1 fwd, ...)

    # --- pooled neighbourhood (24..27) ---
    valid = desc[..., layout.NBR_PRESENT] > 0.5
    cnt = valid.sum(1).astype(_F32)
    inv = 1.0 / mx.maximum(cnt, 1.0)
    crowd = cnt / d.k
    kin_mean = desc[..., layout.NBR_KIN].sum(1) * inv
    msg_mean = mx.where(valid[..., None], s.msg[nbr.idx], 0.0).sum(1) * inv[:, None]

    # --- nearest NS neighbours (28..48) ---
    near = desc[:, : layout.NS, :].reshape(n, layout.NS * layout.NF)

    # --- bond slots (49..68) ---
    pj = mx.maximum(s.partner, 0)
    brel = mx.clip(ops.to_body(geom.off, cos_h[:, None], sin_h[:, None]) / cfg.bonds.break_length, -1.0, 1.0)
    bond = mx.stack(
        [
            mx.ones(pj.shape, dtype=_F32),
            brel[..., 0],
            brel[..., 1],
            s.msg[:, 0][pj],
            _log_ratio(s.energy[pj], s.energy[:, None], q0),
        ],
        axis=-1,
    )
    bond = mx.where(geom.valid[..., None], bond, 0.0).reshape(n, layout.B * layout.BOND_W)

    x = mx.concatenate(
        [
            mx.ones((n, 1), dtype=_F32),  # IN_BIAS
            _col(energy),  # IN_ENERGY
            _col(age_in),  # IN_AGE
            vel,  # IN_VEL
            _col(nbonds),  # IN_NBONDS
            _col(hurt),  # IN_HURT
            _col(fed),  # IN_FED
            _col(gift),  # IN_GIFT
            _col(noise),  # IN_NOISE
            _col(mx.sin(phase)),  # IN_CLOCK sin
            _col(mx.cos(phase)),  # IN_CLOCK cos
            _col(fobs.res),  # IN_RES
            fobs.res_grad.astype(_F32),  # IN_RES_GRAD
            fobs.chem.astype(_F32),  # IN_CHEM
            chem_grad,  # IN_CHEM_GRAD
            _col(crowd),  # IN_CROWD
            _col(kin_mean),  # IN_KIN_MEAN
            msg_mean,  # IN_MSG_MEAN
            near,  # IN_NBR
            bond,  # IN_BOND
        ],
        axis=1,
    )
    assert x.shape == (n, layout.D_IN), x.shape  # static shape check, free under compile
    return x
