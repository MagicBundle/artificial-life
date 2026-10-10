"""§5 sensing: exact input layout on hand-built scenes, normalisation, masking and the |x| ≤ 1 contract.

Neighbour lists, bond geometry and field observations are built directly in numpy (the package does not wait
for WP1/WP3); ``_ref_inputs`` is a plain-numpy float64 reference of the whole §5 table.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from alife import fields, layout, sensing
from alife.bonds import BondGeom
from alife.config import Config, apply_overrides, dims
from alife.fields import FieldObs
from alife.layout import D_IN, MSG, NCH, NF, NS, B
from alife.spatial import Neighbors
from alife.state import State, age, to_numpy
from helpers import make_scene

L_ = layout


# --- builders --------------------------------------------------------------------------------------------


def _min_image(d: np.ndarray, L: float) -> np.ndarray:
    return d - L * np.round(d / L)


def _neighbors(s: State, cfg: Config) -> Neighbors:
    """Brute-force kNN (alive, within R, nearest first, ties by row) in the §4 Neighbors format."""
    d = dims(cfg)
    pos, alive = np.array(s.pos, np.float64), np.array(s.alive)
    n, k, R, L = d.n, d.k, cfg.sensing.radius, cfg.world.size
    idx = np.tile(np.arange(n, dtype=np.int32)[:, None], (1, k))
    valid = np.zeros((n, k), bool)
    off = np.zeros((n, k, 2), np.float32)
    dist = np.full((n, k), R, np.float32)
    live = np.flatnonzero(alive)
    for i in live:
        o = _min_image(pos[live] - pos[i], L)
        dd = np.sqrt((o * o).sum(1))
        cand = [(dd[c], int(live[c]), o[c]) for c in range(len(live)) if live[c] != i and dd[c] < R]
        cand.sort(key=lambda t: (t[0], t[1]))
        for slot, (dj, j, oj) in enumerate(cand[:k]):
            idx[i, slot], valid[i, slot], off[i, slot], dist[i, slot] = j, True, oj, dj
    return Neighbors(
        mx.array(idx),
        mx.array(valid),
        mx.array(off),
        mx.array(dist),
        mx.array(0, mx.int32),
        mx.array(0, mx.int32),
    )


def _geom(s: State, cfg: Config) -> BondGeom:
    pos, partner = np.array(s.pos, np.float64), np.array(s.partner)
    occ = partner >= 0
    off = _min_image(pos[np.maximum(partner, 0)] - pos[:, None, :], cfg.world.size)
    off = np.where(occ[..., None], off, 0.0).astype(np.float32)
    return BondGeom(mx.array(off), mx.array(np.sqrt((off * off).sum(-1))), mx.array(occ))


def _fobs(n: int, res=0.0, res_grad=(0.0, 0.0), chem=0.0, chem_grad=0.0) -> FieldObs:
    def full(v, shape):
        return mx.array(np.broadcast_to(np.asarray(v, np.float32), shape).copy())

    return FieldObs(
        full(res, (n,)), full(res_grad, (n, 2)), full(chem, (n, NCH)), full(chem_grad, (n, NCH, 2))
    )


def _sense(s: State, cfg: Config, fobs: FieldObs | None = None, nbr: Neighbors | None = None, seed: int = 0):
    nbr = _neighbors(s, cfg) if nbr is None else nbr
    fobs = _fobs(dims(cfg).n) if fobs is None else fobs
    desc = sensing.neighbor_descriptors(s, nbr, cfg)
    x = sensing.build_inputs(
        s, age(s.step, s.birth_step), nbr, desc, _geom(s, cfg), fobs, mx.random.key(seed), cfg
    )
    mx.eval(desc, x)
    assert desc.shape == (dims(cfg).n, dims(cfg).k, NF) and desc.dtype == mx.float32
    assert x.shape == (dims(cfg).n, D_IN) and x.dtype == mx.float32
    return np.array(desc), np.array(x), nbr


def _nbr_col(n: int, f: int) -> int:
    return L_.IN_NBR + NF * n + f


def _bond_col(b: int, f: int) -> int:
    return L_.IN_BOND + L_.BOND_W * b + f


def _lr(a: float, b: float, cfg: Config) -> float:
    q0 = max(cfg.energy.basal_cost, 1)
    return float(np.clip(np.log((a + q0) / (b + q0)) / L_.LR_SCALE, -1, 1))


# --- numpy reference of §5 ------------------------------------------------------------------------------


def _ref_inputs(s: State, cfg: Config, nbr: Neighbors, fobs: FieldObs, noise: np.ndarray) -> np.ndarray:
    d = dims(cfg)
    g = {k: np.array(to_numpy(v)).astype(np.float64) for k, v in s._asdict().items() if v.ndim >= 1}
    alive = np.array(s.alive)
    partner = np.array(s.partner)
    idx, valid = np.array(nbr.idx), np.array(nbr.valid)
    off = np.array(nbr.off, np.float64)
    a = (int(s.step) - 1 - g["birth_step"]).astype(np.float64)
    q0 = max(cfg.energy.basal_cost, 1)
    th = g["heading"]
    fwd = np.stack([np.cos(th), np.sin(th)], -1)
    lat = np.stack([-np.sin(th), np.cos(th)], -1)
    geom = _geom(s, cfg)
    goff = np.array(geom.off, np.float64)

    def body(v: np.ndarray) -> np.ndarray:  # (N,...,2) world → body
        sh = (slice(None),) + (None,) * (v.ndim - 2)
        return np.stack([(v * fwd[sh]).sum(-1), (v * lat[sh]).sum(-1)], -1)

    def lr(x, y):
        return np.clip(np.log((x + q0) / (y + q0)) / L_.LR_SCALE, -1, 1)

    E, msg, tag = g["energy"], g["msg"], g["tag"]
    x = np.zeros((d.n, D_IN))
    x[:, L_.IN_BIAS] = 1
    x[:, L_.IN_ENERGY] = np.log1p(E / q0) / np.log1p(cfg.energy.e_max / q0)
    x[:, L_.IN_AGE] = np.tanh(a / L_.AGE_SCALE)
    x[:, L_.IN_VEL : L_.IN_VEL + 2] = np.clip(body(g["vel"]) / cfg.physics.v_max, -1, 1)
    x[:, L_.IN_NBONDS] = (partner >= 0).sum(1) / B
    x[:, L_.IN_HURT] = np.minimum(g["last_loss"] / cfg.combat.attack_max, 1)
    x[:, L_.IN_FED] = np.minimum(g["last_harvest"] / cfg.energy.harvest_max, 1)
    x[:, L_.IN_GIFT] = np.minimum(g["last_inflow"] / cfg.bonds.share_rate, 1)
    x[:, L_.IN_NOISE] = noise
    ph = 2 * np.pi * a / cfg.sensing.clock_period
    x[:, L_.IN_CLOCK], x[:, L_.IN_CLOCK + 1] = np.sin(ph), np.cos(ph)
    x[:, L_.IN_RES] = np.array(fobs.res)
    x[:, L_.IN_RES_GRAD : L_.IN_RES_GRAD + 2] = np.array(fobs.res_grad)
    x[:, L_.IN_CHEM : L_.IN_CHEM + NCH] = np.array(fobs.chem)
    x[:, L_.IN_CHEM_GRAD : L_.IN_CHEM_GRAD + 2 * NCH] = np.array(fobs.chem_grad).reshape(d.n, -1)

    desc = np.zeros((d.n, d.k, NF))
    rel = np.clip(body(off) / cfg.sensing.radius, -1, 1)
    for i in range(d.n):
        for k in range(d.k):
            if not valid[i, k]:
                continue
            j = idx[i, k]
            sim = np.exp(-((tag[i] - tag[j]) ** 2).sum() / cfg.sensing.tag_scale**2)
            bonded = float(j in partner[i])
            desc[i, k] = [1, rel[i, k, 0], rel[i, k, 1], sim, lr(E[j], E[i]), msg[j, 0], bonded]
    cnt = valid.sum(1)
    x[:, L_.IN_CROWD] = cnt / d.k
    x[:, L_.IN_KIN_MEAN] = desc[..., L_.NBR_KIN].sum(1) / np.maximum(cnt, 1)
    x[:, L_.IN_MSG_MEAN : L_.IN_MSG_MEAN + MSG] = (
        np.where(valid[..., None], msg[idx], 0).sum(1) / np.maximum(cnt, 1)[:, None]
    )
    x[:, L_.IN_NBR : L_.IN_NBR + NS * NF] = desc[:, :NS].reshape(d.n, -1)

    brel = np.clip(body(goff) / cfg.bonds.break_length, -1, 1)
    pj = np.maximum(partner, 0)
    blk = np.stack(
        [np.ones_like(brel[..., 0]), brel[..., 0], brel[..., 1], msg[pj, 0], lr(E[pj], E[:, None])], -1
    )
    x[:, L_.IN_BOND :] = np.where((partner >= 0)[..., None], blk, 0).reshape(d.n, -1)
    del alive
    return x


# --- hand-built scenes ------------------------------------------------------------------------------------


def test_rotated_neighbour_offsets_and_absent_blocks(micro_cfg: Config) -> None:
    cfg, R = micro_cfg, micro_cfg.sensing.radius
    # Agent 0 faces +y. Agent 1 is 3 u ahead (world +y), agent 2 is 2 u to the world −x, i.e. to its left.
    s = make_scene(
        cfg,
        [
            {"pos": (10.0, 10.0), "heading": math.pi / 2},
            {"pos": (10.0, 13.0)},
            {"pos": (8.0, 10.0)},
            {"pos": (30.0, 30.0)},  # isolated: no neighbours at all
        ],
    )
    desc, x, _ = _sense(s, cfg)
    # Nearest first: agent 2 (d = 2) then agent 1 (d = 3).
    assert x[0, _nbr_col(0, L_.NBR_PRESENT)] == 1.0
    # Body frame: fwd = (0, 1), lat = (−1, 0). Agent 2 offset (−2, 0) → fwd 0, lat +2.
    assert x[0, _nbr_col(0, L_.NBR_DX)] == pytest.approx(0.0, abs=1e-6)
    assert x[0, _nbr_col(0, L_.NBR_DY)] == pytest.approx(2.0 / R, abs=1e-6)
    # Agent 1 offset (0, 3) → fwd 3, lat 0.
    assert x[0, _nbr_col(1, L_.NBR_PRESENT)] == 1.0
    assert x[0, _nbr_col(1, L_.NBR_DX)] == pytest.approx(3.0 / R, abs=1e-6)
    assert x[0, _nbr_col(1, L_.NBR_DY)] == pytest.approx(0.0, abs=1e-6)
    # Third block absent: all zeros, present = 0.
    assert np.all(x[0, _nbr_col(2, 0) : _nbr_col(2, NF)] == 0.0)
    assert np.all(desc[0, 2:] == 0.0)
    assert x[0, L_.IN_CROWD] == pytest.approx(2 / dims(cfg).k)
    # Isolated agent: whole neighbourhood block zero.
    assert np.all(x[3, L_.IN_CROWD : L_.IN_NBR + NS * NF] == 0.0)


def test_neighbour_across_the_seam(micro_cfg: Config) -> None:
    cfg, Lw, R = micro_cfg, micro_cfg.world.size, micro_cfg.sensing.radius
    s = make_scene(cfg, [{"pos": (0.5, 20.0), "heading": 0.0}, {"pos": (Lw - 1.0, 20.0), "heading": math.pi}])
    _, x, _ = _sense(s, cfg)
    # Agent 0 faces +x; the neighbour is 1.5 u behind it through the seam.
    assert x[0, _nbr_col(0, L_.NBR_DX)] == pytest.approx(-1.5 / R, abs=1e-6)
    assert x[0, _nbr_col(0, L_.NBR_DY)] == pytest.approx(0.0, abs=1e-6)
    # Agent 1 faces −x; agent 0 is 1.5 u to its world +x through the seam, i.e. behind it as well.
    assert x[1, _nbr_col(0, L_.NBR_DX)] == pytest.approx(-1.5 / R, abs=1e-6)


def test_kin_energy_and_message_features(micro_cfg: Config) -> None:
    cfg = micro_cfg
    tag = (0.3, -0.2, 0.5, 0.1)
    other = (-0.3, 0.4, 0.0, 0.1)
    s = make_scene(
        cfg,
        [
            {"pos": (10.0, 10.0), "tag": tag, "energy": 900, "msg": (0.25, -0.5)},
            {"pos": (11.0, 10.0), "tag": tag, "energy": 2618, "msg": (0.75, 0.5)},  # clone, d = 1
            {"pos": (10.0, 12.0), "tag": other, "energy": 50, "msg": (-0.5, 1.0)},  # stranger, d = 2
        ],
    )
    _, x, _ = _sense(s, cfg)
    sim = math.exp(-sum((a - b) ** 2 for a, b in zip(tag, other, strict=True)) / cfg.sensing.tag_scale**2)
    assert x[0, _nbr_col(0, L_.NBR_KIN)] == pytest.approx(1.0, abs=1e-7)  # clone
    assert x[0, _nbr_col(1, L_.NBR_KIN)] == pytest.approx(sim, rel=1e-5)
    assert x[0, _nbr_col(0, L_.NBR_RELE)] == pytest.approx(_lr(2618, 900, cfg), abs=1e-6)
    assert x[0, _nbr_col(0, L_.NBR_RELE)] == pytest.approx(0.5, abs=1e-4)  # log(2718/1000)/2 ≈ 1/2
    assert x[0, _nbr_col(1, L_.NBR_RELE)] == pytest.approx(_lr(50, 900, cfg), abs=1e-6)
    assert x[0, _nbr_col(0, L_.NBR_MSG0)] == 0.75 and x[0, _nbr_col(1, L_.NBR_MSG0)] == -0.5
    assert x[0, L_.IN_KIN_MEAN] == pytest.approx((1 + sim) / 2, rel=1e-6)
    assert x[0, L_.IN_MSG_MEAN] == pytest.approx(0.125) and x[0, L_.IN_MSG_MEAN + 1] == pytest.approx(0.75)
    assert x[0, _nbr_col(0, L_.NBR_BONDED)] == 0.0
    # Extreme energy ratios clip to ±1.
    s2 = make_scene(
        cfg, [{"pos": (10.0, 10.0), "energy": 1}, {"pos": (11.0, 10.0), "energy": cfg.energy.e_max}]
    )
    _, x2, _ = _sense(s2, cfg)
    assert x2[0, _nbr_col(0, L_.NBR_RELE)] == 1.0 and x2[1, _nbr_col(0, L_.NBR_RELE)] == -1.0


def test_bonded_partner_slot_block(micro_cfg: Config) -> None:
    cfg, Lb = micro_cfg, micro_cfg.bonds.break_length
    # Agent 0 faces −x (heading π); partner 1 sits 2 u ahead of it (world −x) in agent 0's slot 2.
    s = make_scene(
        cfg,
        [
            {"pos": (20.0, 20.0), "heading": math.pi, "energy": 4000},
            {"pos": (18.0, 20.0), "heading": 0.0, "energy": 1000, "msg": (-0.6, 0.2)},
        ],
        bonds=[(0, 2, 1, 0)],
    )
    desc, x, _ = _sense(s, cfg)
    assert x[0, L_.IN_NBONDS] == pytest.approx(0.25) and x[1, L_.IN_NBONDS] == pytest.approx(0.25)
    blk = x[0, _bond_col(2, 0) : _bond_col(2, L_.BOND_W)]
    assert blk[L_.BOND_PRESENT] == 1.0
    assert blk[L_.BOND_DX] == pytest.approx(2.0 / Lb, abs=1e-6)
    assert blk[L_.BOND_DY] == pytest.approx(0.0, abs=1e-6)
    assert blk[L_.BOND_MSG0] == pytest.approx(-0.6)
    assert blk[L_.BOND_RELE] == pytest.approx(_lr(1000, 4000, cfg), abs=1e-6)
    # Empty slots of agent 0 are zero; agent 1 (facing +x) sees agent 0 in its slot 0, 2 u ahead of it.
    for b in (0, 1, 3):
        assert np.all(x[0, _bond_col(b, 0) : _bond_col(b, L_.BOND_W)] == 0.0)
    assert x[1, _bond_col(0, L_.BOND_DX)] == pytest.approx(2.0 / Lb, abs=1e-6)
    assert x[1, _bond_col(0, L_.BOND_RELE)] == pytest.approx(_lr(4000, 1000, cfg), abs=1e-6)
    # The neighbour descriptor flags the partner as bonded on both sides.
    assert desc[0, 0, L_.NBR_BONDED] == 1.0 and desc[1, 0, L_.NBR_BONDED] == 1.0
    assert x[0, _nbr_col(0, L_.NBR_BONDED)] == 1.0


def test_stretched_bond_offsets_are_clipped(micro_cfg: Config) -> None:
    cfg = micro_cfg
    # A 4.5-long bond (stretched past break_length = 4 by the previous step's physics, R10).
    s = make_scene(
        cfg,
        [{"pos": (20.0, 20.0), "heading": 0.0}, {"pos": (24.5, 20.0), "heading": math.pi / 2}],
        bonds=[(0, 1)],
    )
    _, x, _ = _sense(s, cfg)
    assert x[0, _bond_col(0, L_.BOND_DX)] == 1.0 and x[0, _bond_col(0, L_.BOND_DY)] == pytest.approx(
        0, abs=1e-6
    )
    # Agent 1 faces +y: agent 0 (world −x) lies to its left (lat = +4.5 → +1).
    assert x[1, _bond_col(0, L_.BOND_DY)] == 1.0
    assert np.all(np.abs(x) <= 1.0)


def test_self_features(micro_cfg: Config) -> None:
    cfg = apply_overrides(micro_cfg, ["energy.basal_cost=100"])
    e_max, vmax = cfg.energy.e_max, cfg.physics.v_max
    s = make_scene(
        cfg,
        [
            {"pos": (5.0, 5.0), "energy": e_max, "birth_step": 3, "heading": math.pi / 2, "vel": (0.0, 0.25)},
            {"pos": (25.0, 5.0), "energy": 1, "birth_step": -1, "vel": (3.0, -3.0)},
            {
                "pos": (5.0, 25.0),
                "last_loss": cfg.combat.attack_max // 2,
                "last_harvest": 10 * cfg.energy.harvest_max,
                "last_inflow": cfg.bonds.share_rate,
            },
        ],
        step=12,
    )
    _, x, _ = _sense(s, cfg)
    assert np.all(x[:3, L_.IN_BIAS] == 1.0)
    assert x[0, L_.IN_ENERGY] == pytest.approx(1.0, abs=1e-6)
    q0 = cfg.energy.basal_cost
    assert x[1, L_.IN_ENERGY] == pytest.approx(math.log1p(1 / q0) / math.log1p(e_max / q0), rel=1e-5)
    # Age during step 12 = 12 − 1 − birth_step: 8 for agent 0 (a quarter clock period), 12 for founders.
    assert x[0, L_.IN_AGE] == pytest.approx(math.tanh(8 / L_.AGE_SCALE), rel=1e-6)
    assert x[0, L_.IN_CLOCK] == pytest.approx(1.0, abs=1e-6) and x[0, L_.IN_CLOCK + 1] == pytest.approx(
        0, abs=1e-6
    )
    ph = 2 * math.pi * 12 / cfg.sensing.clock_period
    assert x[1, L_.IN_CLOCK] == pytest.approx(math.sin(ph), abs=1e-6)
    # Velocity (0, 0.25) while facing +y is straight ahead; a fast diagonal clips per component.
    assert x[0, L_.IN_VEL] == pytest.approx(0.25 / vmax, abs=1e-6) and abs(x[0, L_.IN_VEL + 1]) < 1e-6
    assert x[1, L_.IN_VEL] == 1.0 and x[1, L_.IN_VEL + 1] == -1.0
    assert x[2, L_.IN_HURT] == pytest.approx(0.5, abs=1e-4)
    assert x[2, L_.IN_FED] == 1.0 and x[2, L_.IN_GIFT] == 1.0
    assert np.all(x[:2, L_.IN_HURT : L_.IN_GIFT + 1] == 0.0)


def test_clock_phase_is_exact_for_large_ages(micro_cfg: Config) -> None:
    big = 2**30 + 8  # ≡ 8 mod 32: a quarter period
    s = make_scene(micro_cfg, [{"pos": (5.0, 5.0), "birth_step": -1}], step=big)  # age = step − 1 + 1
    _, x, _ = _sense(s, micro_cfg)
    assert x[0, L_.IN_CLOCK] == pytest.approx(1.0, abs=1e-6) and abs(x[0, L_.IN_CLOCK + 1]) < 1e-6


def test_noise_stream(micro_cfg: Config) -> None:
    s = make_scene(micro_cfg, [{"pos": (5.0, 5.0)}])
    _, xa, _ = _sense(s, micro_cfg, seed=1)
    _, xb, _ = _sense(s, micro_cfg, seed=1)
    _, xc, _ = _sense(s, micro_cfg, seed=2)
    n = dims(micro_cfg).n
    want = np.array(mx.random.uniform(-1.0, 1.0, (n,), key=mx.random.key(1)))
    assert np.array_equal(xa[:, L_.IN_NOISE], want)
    assert np.array_equal(xa, xb) and not np.array_equal(xa[:, L_.IN_NOISE], xc[:, L_.IN_NOISE])
    assert np.all((xa[:, L_.IN_NOISE] >= -1) & (xa[:, L_.IN_NOISE] < 1))


def test_field_observations_are_copied(micro_cfg: Config) -> None:
    cfg = micro_cfg
    n = dims(cfg).n
    s = make_scene(cfg, [{"pos": (5.0, 5.0)}])
    rs = np.random.default_rng(3)
    fo = FieldObs(
        mx.array(rs.uniform(0, 1, n).astype(np.float32)),
        mx.array(rs.uniform(-1, 1, (n, 2)).astype(np.float32)),
        mx.array(rs.uniform(0, 1, (n, NCH)).astype(np.float32)),
        mx.array(rs.uniform(-1, 1, (n, NCH, 2)).astype(np.float32)),
    )
    _, x, _ = _sense(s, cfg, fobs=fo)
    assert np.array_equal(x[:, L_.IN_RES], np.array(fo.res))
    assert np.array_equal(x[:, L_.IN_RES_GRAD : L_.IN_RES_GRAD + 2], np.array(fo.res_grad))
    assert np.array_equal(x[:, L_.IN_CHEM : L_.IN_CHEM + NCH], np.array(fo.chem))
    cg = np.array(fo.chem_grad)
    for c in range(NCH):  # (c0 fwd, c0 lat, c1 fwd, ...)
        assert np.array_equal(x[:, L_.IN_CHEM_GRAD + 2 * c], cg[:, c, 0])
        assert np.array_equal(x[:, L_.IN_CHEM_GRAD + 2 * c + 1], cg[:, c, 1])


def test_resource_ramp_gradient(micro_cfg: Config) -> None:
    """Through ``fields.sample_fields``: a ramp along +x gives the §5 gradient in the body frame."""
    cfg = micro_cfg
    d = dims(cfg)
    cap, fc = cfg.fields.resource_cap, cfg.world.field_cell
    ramp = np.tile((np.arange(d.fg, dtype=np.int32) * 100)[None, :], (d.fg, 1))  # R[y, x] = 100·x
    s = make_scene(
        cfg,
        [
            {"pos": (5.5 * fc, 3.5 * fc), "heading": 0.0},
            {"pos": (5.5 * fc, 7.5 * fc), "heading": math.pi / 2},
        ],
        resource=ramp,
    )
    try:
        fcell = fields.cell_index(s.pos, cfg)
        fo = fields.sample_fields(s.resource, s.chem, fcell, s.heading, cfg)
        mx.eval(fo)
    except NotImplementedError:
        pytest.skip("fields.sample_fields not implemented yet (WP1)")
    _, x, _ = _sense(s, cfg, fobs=fo)
    g = L_.RES_GRAD_GAIN * 200.0 / (2 * cap)  # central difference of the ramp, body-frame gain
    assert x[0, L_.IN_RES_GRAD] == pytest.approx(g, rel=1e-5) and abs(x[0, L_.IN_RES_GRAD + 1]) < 1e-6
    # Facing +y, a +x gradient lies to the right: lat = −g.
    assert abs(x[1, L_.IN_RES_GRAD]) < 1e-6 and x[1, L_.IN_RES_GRAD + 1] == pytest.approx(-g, rel=1e-5)
    assert x[0, L_.IN_RES] == pytest.approx(min(500 / cap, L_.RES_SAT) / L_.RES_SAT, rel=1e-6)


def test_dead_and_padded_neighbours_are_zeroed(micro_cfg: Config) -> None:
    cfg = micro_cfg
    d = dims(cfg)
    s = make_scene(cfg, [{"pos": (10.0, 10.0)}, {"pos": (11.0, 10.0)}, {"pos": (10.0, 11.5), "row": 5}])
    nbr = _neighbors(s, cfg)
    # Kill row 5 without fixing the neighbour list, and corrupt the padding lanes' geometry: neither may leak.
    alive = np.array(s.alive)
    alive[5] = False
    s = s._replace(alive=mx.array(alive))
    off = np.array(nbr.off)
    off[~np.array(nbr.valid)] = 7.0
    nbr = nbr._replace(off=mx.array(off))
    desc, x, _ = _sense(s, cfg, nbr=nbr)
    assert np.array(nbr.valid)[0, :2].all()  # row 5 is still listed as valid for agent 0
    assert desc[0, 0, L_.NBR_PRESENT] == 1.0  # agent 1 (d = 1)
    assert np.all(desc[0, 1:] == 0.0)  # dead row 5 and padding
    assert x[0, L_.IN_CROWD] == pytest.approx(1 / d.k)
    dead_rows = np.flatnonzero(~alive)
    assert np.all(desc[dead_rows] == 0.0)
    assert np.all(x[dead_rows, L_.IN_CROWD : L_.IN_BOND + B * L_.BOND_W] == 0.0)


def test_random_scene_matches_reference(tiny_cfg: Config) -> None:
    """Every index on a random scene with bonds (some stretched), seams, crowds and dead rows."""
    cfg = tiny_cfg
    d = dims(cfg)
    rs = np.random.default_rng(11)
    n_agents = 300
    Lw = cfg.world.size
    centres = rs.uniform(0, Lw, (12, 2))
    agents = []
    for k in range(n_agents):
        pos = np.mod(centres[k % 12] + rs.normal(0, 3.0, 2), Lw).astype(np.float32)
        agents.append(
            {
                "row": int(k * 3 % d.n_cap),
                "pos": pos,
                "heading": rs.uniform(-math.pi, math.pi),
                "vel": rs.uniform(-0.7, 0.7, 2),
                "energy": int(rs.integers(1, cfg.energy.e_max + 1)),
                "msg": rs.uniform(-1, 1, MSG),
                "tag": np.clip(rs.normal(0, 0.3, 4), -1, 1),
                "birth_step": int(rs.integers(-1, 40)),
                "last_loss": int(rs.integers(0, 2 * cfg.combat.attack_max)),
                "last_harvest": int(rs.integers(0, 2 * cfg.energy.harvest_max)),
                "last_inflow": int(rs.integers(0, 2 * cfg.bonds.share_rate)),
            }
        )
    rows = [a["row"] for a in agents]
    assert len(set(rows)) == n_agents
    pairs, used = [], set()
    for _ in range(250):
        i, j = rs.choice(rows, 2, replace=False)
        if (i, j) in used or (j, i) in used:
            continue
        cnt = sum(1 for p in pairs if i in p), sum(1 for p in pairs if j in p)
        if max(cnt) >= B:
            continue
        used.add((i, j))
        pairs.append((int(i), int(j)))
    s = make_scene(cfg, agents, bonds=pairs, step=40)
    fo = FieldObs(
        mx.array(rs.uniform(0, 1, d.n).astype(np.float32)),
        mx.array(rs.uniform(-1, 1, (d.n, 2)).astype(np.float32)),
        mx.array(rs.uniform(0, 1, (d.n, NCH)).astype(np.float32)),
        mx.array(rs.uniform(-1, 1, (d.n, NCH, 2)).astype(np.float32)),
    )
    _, x, nbr = _sense(s, cfg, fobs=fo, seed=5)
    assert np.array(nbr.valid).sum() > 3 * n_agents  # crowded enough to fill several slots
    assert np.isfinite(x).all() and np.abs(x).max() <= 1.0
    noise = np.array(mx.random.uniform(-1.0, 1.0, (d.n,), key=mx.random.key(5)))
    ref = _ref_inputs(s, cfg, nbr, fo, noise)
    live = np.array(s.alive)
    err = np.abs(x[live] - ref[live])
    worst = np.unravel_index(err.argmax(), err.shape)
    assert err.max() < 2e-5, (worst, L_.INPUT_NAMES[worst[1]], x[live][worst], ref[live][worst])


def test_contract_holds_on_all_rows_and_zero_scales(micro_cfg: Config) -> None:
    """|x| ≤ 1 and finite everywhere, dead rows included, even with zero basal cost and zero scales."""
    cfg = apply_overrides(micro_cfg, ["energy.basal_cost=0", "energy.harvest_max=0", "bonds.share_rate=0"])
    s = make_scene(
        cfg, [{"pos": (10.0, 10.0), "last_harvest": 5, "energy": 3}, {"pos": (11.0, 10.0), "energy": 1}]
    )
    _, x, _ = _sense(s, cfg)
    assert np.isfinite(x).all() and np.abs(x).max() <= 1.0
    assert x[0, L_.IN_FED] == 1.0 and x[0, L_.IN_GIFT] == 0.0


def test_compiled_matches_eager(micro_cfg: Config) -> None:
    cfg = micro_cfg
    s = make_scene(cfg, [{"pos": (10.0, 10.0), "heading": 1.0}, {"pos": (11.0, 11.0)}], bonds=[(0, 1)])
    nbr, geom, fo = _neighbors(s, cfg), _geom(s, cfg), _fobs(dims(cfg).n, 0.5, (0.1, -0.2), 0.3, 0.05)
    a = age(s.step, s.birth_step)
    key = mx.random.key(4)

    def f(s, a, nbr, geom, fo, key):
        desc = sensing.neighbor_descriptors(s, nbr, cfg)
        return desc, sensing.build_inputs(s, a, nbr, desc, geom, fo, key, cfg)

    e = f(s, a, nbr, geom, fo, key)
    c = mx.compile(f)(s, a, nbr, geom, fo, key)
    for u, v in zip(e, c, strict=True):
        assert np.allclose(np.array(u), np.array(v), atol=1e-6)
