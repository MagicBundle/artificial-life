"""§27.2 the 3D neighbour descriptor: the seven 2D features with the offset in the yaw-and-pitch body frame,
then ``NBR_DZ`` (the up component / R). §27.6 the 3D input vector: the 69 2D inputs from the 3D frame, then
the tail 69–83, against a float64 reference of all 84 columns."""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from alife import layout, physics, sensing
from alife.bonds import BondGeom
from alife.config import Config, dims
from alife.fields import FieldObs
from alife.layout import MSG, NCH, NS, B
from alife.spatial import Neighbors
from alife.state import State, State3, age, to_numpy
from helpers import ON_GPU, as_3d, make_scene

PM = layout.PITCH_MAX


@pytest.fixture(scope="module")
def cfg3(micro_cfg: Config) -> Config:
    return as_3d(micro_cfg)


def _neighbors(s: State | State3, cfg: Config) -> Neighbors:
    """Brute-force kNN (alive, within R, nearest first, ties by row); minimum image on x and y only."""
    d = dims(cfg)
    pos, alive = np.array(s.pos, np.float64), np.array(s.alive)
    n, k, R, L = d.n, d.k, cfg.sensing.radius, cfg.world.size
    idx = np.tile(np.arange(n, dtype=np.int32)[:, None], (1, k))
    valid = np.zeros((n, k), bool)
    off = np.zeros((n, k, d.nd), np.float32)
    dist = np.full((n, k), R, np.float32)
    live = np.flatnonzero(alive)
    for i in live:
        o = pos[live] - pos[i]
        o[:, :2] -= L * np.round(o[:, :2] / L)
        dd = np.sqrt((o * o).sum(1))
        cand = sorted((dd[c], int(live[c]), c) for c in range(len(live)) if live[c] != i and dd[c] < R)
        for slot, (dj, j, c) in enumerate(cand[:k]):
            idx[i, slot], valid[i, slot], off[i, slot], dist[i, slot] = j, True, o[c], dj
    zero = mx.array(0, mx.int32)
    return Neighbors(mx.array(idx), mx.array(valid), mx.array(off), mx.array(dist), zero, zero)


def _agents(rng: np.random.Generator, n: int, z: np.ndarray | None) -> list[dict]:
    """``n`` agents in a 14×14 patch (so most see neighbours) with random heading, tag, energy, msg; 3D when
    ``z`` is given (then also a random pitch)."""
    out = []
    for i in range(n):
        a = {
            "pos": [*rng.uniform(17.0, 31.0, 2), *([] if z is None else [z[i]])],
            "heading": rng.uniform(-math.pi, math.pi),
            "tag": rng.normal(0, 0.3, layout.T),
            "energy": int(rng.integers(100, 5000)),
            "msg": rng.uniform(-1, 1, layout.MSG),
        }
        out.append(a)
    return out


def test_descriptor_3d_is_8_wide_and_level_scenes_match_2d(micro_cfg: Config, cfg3: Config) -> None:
    """All agents at one height with pitch 0: the first seven features equal the 2D descriptor of the same
    scene bit for bit (the 3D rotation reduces to the 2D one), and NBR_DZ is 0."""
    rng = np.random.default_rng(0)
    ag2 = _agents(rng, 30, None)
    bonds = [(0, 1), (2, 3), (4, 5)]
    for i, j in bonds:  # partners side by side, so each lists the other
        ag2[j]["pos"] = [ag2[i]["pos"][0] + 0.8, ag2[i]["pos"][1] + 0.3]
    ag3 = [{**a, "pos": [*a["pos"], 12.0], "pitch": 0.0} for a in ag2]
    s2, s3 = make_scene(micro_cfg, ag2, bonds=bonds), make_scene(cfg3, ag3, bonds=bonds)
    assert isinstance(s3, State3)
    nbr2, nbr3 = _neighbors(s2, micro_cfg), _neighbors(s3, cfg3)
    assert np.array_equal(np.array(nbr2.idx), np.array(nbr3.idx)) and np.array(nbr2.valid).sum() > 100
    d2 = np.array(sensing.neighbor_descriptors(s2, nbr2, micro_cfg))
    d3 = np.array(sensing.neighbor_descriptors(s3, nbr3, cfg3))
    lay = dims(cfg3).lay
    assert d3.shape == (dims(cfg3).n, dims(cfg3).k, lay.NF) and lay.NF == 8 and d3.dtype == np.float32
    assert np.array_equal(d3[..., : layout.NF], d2)
    assert np.all(d3[..., lay.NBR_DZ] == 0.0)
    assert d2[..., layout.NBR_BONDED].sum() == 2 * len(bonds)  # the bonded feature is exercised


def test_descriptor_3d_body_frame(cfg3: Config) -> None:
    """Random yaw and pitch in a 3D scene: (dx, dy, dz) are the offset's (fwd, lat, up) components / R, the
    other features are the 2D ones, invalid lanes are zero and every feature is in [−1, 1]."""
    rng = np.random.default_rng(1)
    ag = _agents(rng, 40, rng.uniform(6.0, 18.0, 40))
    for a in ag:
        a["pitch"] = rng.uniform(-PM, PM)
    s = make_scene(cfg3, ag, bonds=[(0, 1), (6, 7)])
    nbr = _neighbors(s, cfg3)
    desc = np.array(sensing.neighbor_descriptors(s, nbr, cfg3))
    lay, R = dims(cfg3).lay, cfg3.sensing.radius
    valid = np.array(nbr.valid)
    assert valid.sum() > 100
    th, ph = np.array(s.heading, np.float64), np.array(s.pitch, np.float64)
    fwd = np.stack([np.cos(ph) * np.cos(th), np.cos(ph) * np.sin(th), np.sin(ph)], -1)[:, None, :]
    lat = np.stack([-np.sin(th), np.cos(th), np.zeros_like(th)], -1)[:, None, :]
    up = np.stack([-np.sin(ph) * np.cos(th), -np.sin(ph) * np.sin(th), np.cos(ph)], -1)[:, None, :]
    off = np.array(nbr.off, np.float64)
    want = np.stack([(off * fwd).sum(-1), (off * lat).sum(-1), (off * up).sum(-1)], -1) / R
    got = desc[..., [layout.NBR_DX, layout.NBR_DY, lay.NBR_DZ]]
    np.testing.assert_allclose(got[valid], want[valid], atol=1e-6)
    assert np.all(desc[..., layout.NBR_PRESENT] == valid)
    assert np.all(desc[~valid] == 0.0)
    assert np.all(np.abs(desc) <= 1.0)
    # The other features do not depend on the frame: the same scene seen level gives the same values.
    level = s._replace(pitch=mx.zeros_like(s.pitch))
    flat = np.array(sensing.neighbor_descriptors(level, nbr, cfg3))
    others = [layout.NBR_PRESENT, layout.NBR_KIN, layout.NBR_RELE, layout.NBR_MSG0, layout.NBR_BONDED]
    assert np.array_equal(flat[..., others], desc[..., others])


def test_descriptor_3d_named_case(cfg3: Config) -> None:
    """Nose 45° up, facing +x: a neighbour 3 straight above is ahead-and-up, one 3 ahead is ahead-and-down."""
    s = make_scene(
        cfg3,
        [
            {"pos": [10.0, 10.0, 10.0], "heading": 0.0, "pitch": math.pi / 4},
            {"pos": [10.0, 10.0, 13.0]},
            {"pos": [13.5, 10.0, 10.0]},
        ],
    )
    desc = np.array(sensing.neighbor_descriptors(s, _neighbors(s, cfg3), cfg3))
    lay, c = dims(cfg3).lay, math.cos(math.pi / 4)
    cols = [layout.NBR_DX, layout.NBR_DY, lay.NBR_DZ]
    np.testing.assert_allclose(desc[0, 0, cols], [3 * c / 6, 0.0, 3 * c / 6], atol=1e-6)  # j = 1, distance 3
    np.testing.assert_allclose(desc[0, 1, cols], [3.5 * c / 6, 0.0, -3.5 * c / 6], atol=1e-6)  # j = 2


def test_descriptor_3d_compiles(cfg3: Config) -> None:
    rng = np.random.default_rng(2)
    ag = _agents(rng, 30, rng.uniform(6.0, 18.0, 30))
    for a in ag:
        a["pitch"] = rng.uniform(-PM, PM)
    s = make_scene(cfg3, ag)
    nbr = _neighbors(s, cfg3)
    eager = np.array(sensing.neighbor_descriptors(s, nbr, cfg3))
    comp = np.array(mx.compile(lambda s_, n_: sensing.neighbor_descriptors(s_, n_, cfg3))(s, nbr))
    np.testing.assert_allclose(comp, eager, atol=1e-6)


# --- the 3D input vector (§27.6) ---------------------------------------------------------------------------

L_ = layout


def _geom3(s: State3, cfg: Config) -> BondGeom:
    """Start-of-step bond offsets: minimum image in x and y, plain in z; empty slots 0."""
    pos, partner, L = np.array(s.pos, np.float64), np.array(s.partner), cfg.world.size
    occ = partner >= 0
    off = pos[np.maximum(partner, 0)] - pos[:, None, :]
    off[..., :2] -= L * np.round(off[..., :2] / L)
    off = np.where(occ[..., None], off, 0.0).astype(np.float32)
    return BondGeom(mx.array(off), mx.array(np.sqrt((off * off).sum(-1))), mx.array(occ))


def _fobs3(n: int, rs: np.random.Generator | None = None) -> FieldObs:
    """Field observations (3D shapes), random when ``rs`` is given, else zero."""

    def draw(lo: float, shape: tuple[int, ...]) -> mx.array:
        v = np.zeros(shape) if rs is None else rs.uniform(lo, 1.0, shape)
        return mx.array(v.astype(np.float32))

    return FieldObs(draw(0.0, (n,)), draw(-1.0, (n, 3)), draw(0.0, (n, NCH)), draw(-1.0, (n, NCH, 3)))


def _sense3(
    s: State3,
    cfg: Config,
    fobs: FieldObs | None = None,
    nbr: Neighbors | None = None,
    geom: BondGeom | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, Neighbors]:
    d = dims(cfg)
    nbr = _neighbors(s, cfg) if nbr is None else nbr
    geom = _geom3(s, cfg) if geom is None else geom
    fobs = _fobs3(d.n) if fobs is None else fobs
    desc = sensing.neighbor_descriptors(s, nbr, cfg)
    x = sensing.build_inputs(
        s, age(s.step, s.birth_step), nbr, desc, geom, fobs, mx.random.key(seed), cfg,
        grounded=physics.grounded(s.pos, cfg),
    )  # fmt: skip
    mx.eval(desc, x)
    assert x.shape == (d.n, d.lay.D_IN) == (d.n, 84) and x.dtype == mx.float32
    return np.array(desc), np.array(x), nbr


def _ref_inputs3(s: State3, cfg: Config, nbr: Neighbors, fobs: FieldObs, noise: np.ndarray) -> np.ndarray:
    """float64 reference of all 84 inputs: the §5 formulas with every vector taken in the (fwd, lat, up) frame
    of §27.2 (its first two components fill the 2D blocks), then the §27.6 tail."""
    d, lay = dims(cfg), dims(cfg).lay
    g = {k: np.array(to_numpy(v)).astype(np.float64) for k, v in s._asdict().items() if v.ndim >= 1}
    partner = np.array(s.partner)
    idx, valid = np.array(nbr.idx), np.array(nbr.valid)
    off = np.array(nbr.off, np.float64)
    goff = np.array(_geom3(s, cfg).off, np.float64)
    a = int(s.step) - 1 - g["birth_step"]
    q0 = max(cfg.energy.basal_cost, 1)
    th, ph = g["heading"], g["pitch"]
    axes = (
        np.stack([np.cos(ph) * np.cos(th), np.cos(ph) * np.sin(th), np.sin(ph)], -1),
        np.stack([-np.sin(th), np.cos(th), np.zeros_like(th)], -1),
        np.stack([-np.sin(ph) * np.cos(th), -np.sin(ph) * np.sin(th), np.cos(ph)], -1),
    )

    def body(v: np.ndarray) -> np.ndarray:  # (N,...,3) world → (fwd, lat, up)
        sh = (slice(None),) + (None,) * (v.ndim - 2)
        return np.stack([(v * ax[sh]).sum(-1) for ax in axes], -1)

    def lr(u, w):
        return np.clip(np.log((u + q0) / (w + q0)) / L_.LR_SCALE, -1, 1)

    E, msg, tag = g["energy"], g["msg"], g["tag"]
    vel = np.clip(body(g["vel"]) / cfg.physics.v_max, -1, 1)
    fo = {k: np.array(v, np.float64) for k, v in fobs._asdict().items()}
    x = np.zeros((d.n, lay.D_IN))
    x[:, L_.IN_BIAS] = 1
    x[:, L_.IN_ENERGY] = np.log1p(E / q0) / np.log1p(cfg.energy.e_max / q0)
    x[:, L_.IN_AGE] = np.tanh(a / L_.AGE_SCALE)
    x[:, L_.IN_VEL : L_.IN_VEL + 2] = vel[:, :2]
    x[:, L_.IN_NBONDS] = (partner >= 0).sum(1) / B
    x[:, L_.IN_HURT] = np.minimum(g["last_loss"] / cfg.combat.attack_max, 1)
    x[:, L_.IN_FED] = np.minimum(g["last_harvest"] / cfg.energy.harvest_max, 1)
    x[:, L_.IN_GIFT] = np.minimum(g["last_inflow"] / cfg.bonds.share_rate, 1)
    x[:, L_.IN_NOISE] = noise
    phase = 2 * np.pi * a / cfg.sensing.clock_period
    x[:, L_.IN_CLOCK], x[:, L_.IN_CLOCK + 1] = np.sin(phase), np.cos(phase)
    x[:, L_.IN_RES] = fo["res"]
    x[:, L_.IN_RES_GRAD : L_.IN_RES_GRAD + 2] = fo["res_grad"][:, :2]
    x[:, L_.IN_CHEM : L_.IN_CHEM + NCH] = fo["chem"]
    x[:, L_.IN_CHEM_GRAD : L_.IN_CHEM_GRAD + 2 * NCH] = fo["chem_grad"][..., :2].reshape(d.n, -1)

    desc = np.zeros((d.n, d.k, lay.NF))
    rel = np.clip(body(off) / cfg.sensing.radius, -1, 1)
    for i in range(d.n):
        for k in range(d.k):
            if valid[i, k]:
                j = idx[i, k]
                sim = np.exp(-((tag[i] - tag[j]) ** 2).sum() / cfg.sensing.tag_scale**2)
                bonded = float(j in partner[i])
                desc[i, k] = [
                    1,
                    rel[i, k, 0],
                    rel[i, k, 1],
                    sim,
                    lr(E[j], E[i]),
                    msg[j, 0],
                    bonded,
                    rel[i, k, 2],
                ]
    cnt = valid.sum(1)
    x[:, L_.IN_CROWD] = cnt / d.k
    x[:, L_.IN_KIN_MEAN] = desc[..., L_.NBR_KIN].sum(1) / np.maximum(cnt, 1)
    x[:, L_.IN_MSG_MEAN : L_.IN_MSG_MEAN + MSG] = (
        np.where(valid[..., None], msg[idx], 0).sum(1) / np.maximum(cnt, 1)[:, None]
    )
    x[:, L_.IN_NBR : L_.IN_NBR + NS * L_.NF] = desc[:, :NS, : L_.NF].reshape(d.n, -1)

    occ = partner >= 0
    brel = np.clip(body(goff) / cfg.bonds.break_length, -1, 1)
    pj = np.maximum(partner, 0)
    blk = np.stack(
        [np.ones_like(brel[..., 0]), brel[..., 0], brel[..., 1], msg[pj, 0], lr(E[pj], E[:, None])], -1
    )
    x[:, L_.IN_BOND : L_.IN_BOND + B * L_.BOND_W] = np.where(occ[..., None], blk, 0).reshape(d.n, -1)

    x[:, lay.IN_VEL_UP] = vel[:, 2]
    x[:, lay.IN_RES_GRAD_UP] = fo["res_grad"][:, 2]
    x[:, lay.IN_CHEM_GRAD_UP : lay.IN_CHEM_GRAD_UP + NCH] = fo["chem_grad"][..., 2]
    x[:, lay.IN_NBR_DZ : lay.IN_NBR_DZ + NS] = desc[:, :NS, lay.NBR_DZ]
    x[:, lay.IN_BOND_DZ : lay.IN_BOND_DZ + B] = np.where(occ, brel[..., 2], 0)
    x[:, lay.IN_HEIGHT] = g["pos"][:, 2] / cfg.world.height
    x[:, lay.IN_PITCH] = np.sin(ph)
    x[:, lay.IN_GROUNDED] = g["pos"][:, 2] <= np.float32(dims(cfg).z_ground)
    return x


def test_build_inputs_3d_random_scene_matches_reference(cfg3: Config) -> None:
    """Every one of the 84 columns on a random volume scene: bonds (some stretched), pitches over the whole
    range, agents on the floor and at the surface, velocities beyond v_max, dead rows."""
    d, lay = dims(cfg3), dims(cfg3).lay
    rs = np.random.default_rng(11)
    zs = np.concatenate([[d.z_lo] * 6, [d.z_hi] * 4, rs.uniform(d.z_lo, 14.0, 30)])
    ag = _agents(rs, len(zs), zs)
    for k, a in enumerate(ag):
        a.update(
            row=int(k * 3 % d.n_cap),
            pitch=rs.uniform(-PM, PM),
            vel=rs.uniform(-0.7, 0.7, 3),
            birth_step=int(rs.integers(-1, 40)),
            last_loss=int(rs.integers(0, 2 * cfg3.combat.attack_max)),
            last_harvest=int(rs.integers(0, 2 * cfg3.energy.harvest_max)),
            last_inflow=int(rs.integers(0, 2 * cfg3.bonds.share_rate)),
        )
    rows = [a["row"] for a in ag]
    pairs = [(rows[i], rows[i + 1]) for i in range(0, 24, 2)] + [(rows[0], rows[30]), (rows[2], rows[31])]
    s = make_scene(cfg3, ag, bonds=pairs, step=40)
    fo = _fobs3(d.n, rs)
    _, x, nbr = _sense3(s, cfg3, fobs=fo, seed=5)
    assert np.array(nbr.valid).sum() > 100
    assert np.isfinite(x).all() and np.abs(x).max() <= 1.0
    noise = np.array(mx.random.uniform(-1.0, 1.0, (d.n,), key=mx.random.key(5)))
    ref = _ref_inputs3(s, cfg3, nbr, fo, noise)
    live = np.array(s.alive)
    err = np.abs(x[live] - ref[live])
    worst = np.unravel_index(err.argmax(), err.shape)
    assert err.max() < 2e-5, (worst, lay.INPUT_NAMES[worst[1]], x[live][worst], ref[live][worst])
    # The field blocks are copied, not recomputed: the up components land in the tail unchanged.
    rg, cg = np.array(fo.res_grad), np.array(fo.chem_grad)
    assert np.array_equal(x[:, L_.IN_RES_GRAD : L_.IN_RES_GRAD + 2], rg[:, :2])
    assert np.array_equal(x[:, lay.IN_RES_GRAD_UP], rg[:, 2])
    assert np.array_equal(x[:, L_.IN_CHEM_GRAD : L_.IN_CHEM_GRAD + 2 * NCH], cg[..., :2].reshape(d.n, -1))
    assert np.array_equal(x[:, lay.IN_CHEM_GRAD_UP : lay.IN_CHEM_GRAD_UP + NCH], cg[..., 2])
    on_floor = live & (np.array(s.pos)[:, 2] == np.float32(d.z_lo))
    assert on_floor.sum() == 6 and np.all(x[on_floor, lay.IN_GROUNDED] == 1.0)
    assert (
        np.array(s.partner)[live].max() >= 0
        and np.abs(x[live, lay.IN_BOND_DZ : lay.IN_BOND_DZ + B]).max() > 0
    )


def test_build_inputs_3d_level_scene_prefix_is_2d(micro_cfg: Config, cfg3: Config) -> None:
    """All agents at one height, level, moving horizontally: the 69 prefix inputs equal the 2D inputs of the
    same scene (bit for bit on CPU), and the tail holds only the field up components, height and grounded."""
    rs = np.random.default_rng(12)
    ag2 = _agents(rs, 30, None)
    for a in ag2:
        a["vel"] = rs.uniform(-0.4, 0.4, 2)
    bonds = [(0, 1), (2, 3), (4, 9)]
    s2 = make_scene(micro_cfg, ag2, bonds=bonds, step=7)
    s3 = make_scene(
        cfg3, [{**a, "pos": [*a["pos"], 9.0], "vel": [*a["vel"], 0.0]} for a in ag2], bonds=bonds, step=7
    )
    fo3 = _fobs3(dims(cfg3).n, rs)
    fo2 = fo3._replace(res_grad=fo3.res_grad[:, :2], chem_grad=fo3.chem_grad[..., :2])
    nbr2 = Neighbors(*_neighbors(s2, micro_cfg))
    geom2 = BondGeom(_geom3(s3, cfg3).off[..., :2], *_geom3(s3, cfg3)[1:])
    desc2 = sensing.neighbor_descriptors(s2, nbr2, micro_cfg)
    x2 = np.array(
        sensing.build_inputs(
            s2, age(s2.step, s2.birth_step), nbr2, desc2, geom2, fo2, mx.random.key(1), micro_cfg
        )
    )
    _, x3, _ = _sense3(s3, cfg3, fobs=fo3, seed=1)
    lay = dims(cfg3).lay
    if ON_GPU:
        np.testing.assert_allclose(x3[:, :69], x2, atol=1e-6)
    else:
        assert np.array_equal(x3[:, :69], x2)
    assert np.abs(x2[:, L_.IN_BOND : L_.IN_BOND + B * L_.BOND_W]).max() > 0  # bond blocks exercised
    for col in (lay.IN_VEL_UP, lay.IN_PITCH, *range(lay.IN_NBR_DZ, lay.IN_NBR_DZ + NS + B)):
        assert not x3[:, col].any(), lay.INPUT_NAMES[col]
    live = np.array(s3.alive)
    assert np.all(x3[live, lay.IN_HEIGHT] == np.float32(9.0) / np.float32(cfg3.world.height))
    assert not x3[live, lay.IN_GROUNDED].any()


def test_build_inputs_3d_pitch_rotation(cfg3: Config) -> None:
    """Nose 45° up facing +x: a neighbour, bond partner and velocity along (1, 0, 1) all lie straight ahead
    (fwd = √2 in units, up = 0)."""
    R, bl, vm = cfg3.sensing.radius, cfg3.bonds.break_length, cfg3.physics.v_max
    s = make_scene(
        cfg3,
        [
            {"pos": [10.0, 10.0, 10.0], "heading": 0.0, "pitch": math.pi / 4, "vel": [0.2, 0.0, 0.2]},
            {"pos": [11.0, 10.0, 11.0]},
        ],
        bonds=[(0, 1)],
    )
    _, x, _ = _sense3(s, cfg3)
    lay, r2 = dims(cfg3).lay, math.sqrt(2.0)
    nb = L_.IN_NBR
    np.testing.assert_allclose(
        x[0, [nb + L_.NBR_DX, nb + L_.NBR_DY, lay.IN_NBR_DZ]], [r2 / R, 0, 0], atol=1e-6
    )
    bd = L_.IN_BOND
    np.testing.assert_allclose(
        x[0, [bd + L_.BOND_DX, bd + L_.BOND_DY, lay.IN_BOND_DZ]], [r2 / bl, 0, 0], atol=1e-6
    )
    np.testing.assert_allclose(
        x[0, [L_.IN_VEL, L_.IN_VEL + 1, lay.IN_VEL_UP]], [0.2 * r2 / vm, 0, 0], atol=1e-6
    )
    assert x[0, lay.IN_PITCH] == pytest.approx(math.sin(math.pi / 4), abs=1e-7)
    # Seen level by the neighbour (pitch 0, facing +x), the agent is behind and below.
    np.testing.assert_allclose(
        x[1, [nb + L_.NBR_DX, nb + L_.NBR_DY, lay.IN_NBR_DZ]], [-1 / R, 0, -1 / R], atol=1e-6
    )


def test_build_inputs_3d_grounded_and_height(cfg3: Config) -> None:
    """``IN_GROUNDED`` is ``physics.grounded``: inclusive at ``z_ground``, false one float32 step above it and
    at the surface; ``IN_HEIGHT`` is z / height."""
    cfg = as_3d(cfg3, physics={"floor_eps": 0.5})
    d = dims(cfg)
    above = float(np.nextafter(np.float32(d.z_ground), np.float32(np.inf)))
    zs = [d.z_lo, d.z_ground, above, 12.0, d.z_hi]
    assert d.z_lo < d.z_ground < above
    s = make_scene(cfg, [{"pos": [6.0 * i + 3.0, 5.0, z]} for i, z in enumerate(zs)])
    _, x, _ = _sense3(s, cfg)
    lay = d.lay
    assert x[: len(zs), lay.IN_GROUNDED].tolist() == [1.0, 1.0, 0.0, 0.0, 0.0]
    np.testing.assert_allclose(x[: len(zs), lay.IN_HEIGHT], np.float32(zs) / cfg.world.height, rtol=1e-7)
    assert x[:, lay.IN_HEIGHT].min() > 0 and x[:, lay.IN_HEIGHT].max() < 1


def test_build_inputs_3d_needs_grounded(cfg3: Config) -> None:
    s = make_scene(cfg3, [{"pos": [5.0, 5.0, 5.0]}])
    nbr = _neighbors(s, cfg3)
    desc = sensing.neighbor_descriptors(s, nbr, cfg3)
    with pytest.raises(AssertionError, match="grounded"):
        sensing.build_inputs(s, age(s.step, s.birth_step), nbr, desc, _geom3(s, cfg3), _fobs3(dims(cfg3).n),
                             mx.random.key(0), cfg3)  # fmt: skip


def test_dead_and_padded_neighbours_zero_the_dz_inputs(cfg3: Config) -> None:
    d = dims(cfg3)
    s = make_scene(
        cfg3,
        [{"pos": [10.0, 10.0, 10.0]}, {"pos": [10.0, 10.0, 11.0]}, {"pos": [10.0, 10.0, 8.5], "row": 5}],
    )
    nbr = _neighbors(s, cfg3)
    # Kill row 5 without fixing the neighbour list, and corrupt the padding lanes' and the empty bond slots'
    # offsets: none may leak into the DZ columns.
    alive = np.array(s.alive)
    alive[5] = False
    s = s._replace(alive=mx.array(alive))
    off = np.array(nbr.off)
    off[~np.array(nbr.valid)] = 7.0
    nbr = nbr._replace(off=mx.array(off))
    geom = _geom3(s, cfg3)
    geom = geom._replace(off=mx.where(geom.valid[..., None], geom.off, 7.0))
    desc, x, _ = _sense3(s, cfg3, nbr=nbr, geom=geom)
    lay = d.lay
    assert np.array(nbr.valid)[0, :2].all()  # row 5 is still listed for agent 0
    assert desc[0, 0, layout.NBR_PRESENT] == 1.0 and desc[0, 0, lay.NBR_DZ] == pytest.approx(1 / 6)
    assert x[0, lay.IN_NBR_DZ] == pytest.approx(1 / 6) and not x[0, lay.IN_NBR_DZ + 1 :][: NS - 1].any()
    assert np.all(desc[0, 1:] == 0.0)
    assert not x[:, lay.IN_BOND_DZ : lay.IN_BOND_DZ + B].any()  # no bonds: every slot is empty
    dead_rows = np.flatnonzero(~alive)
    assert not x[dead_rows, lay.IN_NBR_DZ : lay.IN_NBR_DZ + NS].any()


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_inputs_stay_bounded_at_the_pitch_limit(cfg3: Config, sign: float) -> None:
    """|x| ≤ 1 and finite on every row at pitch ±PITCH_MAX (float32), with velocities beyond v_max, bonds
    stretched past break_length in z, saturated field gradients, agents on the floor and at the surface."""
    d = dims(cfg3)
    pm = sign * float(np.float32(PM))
    agents = [
        {"pos": [10.0, 10.0, d.z_lo], "pitch": pm, "vel": [0.0, 0.0, -3.0]},
        {"pos": [10.0, 10.5, d.z_lo + 5.5], "pitch": pm, "vel": [2.0, -2.0, 3.0]},
        {"pos": [10.5, 10.0, d.z_hi], "pitch": -pm, "vel": [0.0, 1.0, 3.0]},
        {"pos": [10.0, 9.5, d.z_hi - 5.0], "pitch": pm},
    ]
    s = make_scene(cfg3, agents, bonds=[(0, 1), (2, 3)])
    fo = FieldObs(
        mx.ones((d.n,)), mx.full((d.n, 3), sign), mx.ones((d.n, NCH)), mx.full((d.n, NCH, 3), -sign)
    )
    _, x, _ = _sense3(s, cfg3, fobs=fo)
    lay = d.lay
    assert np.isfinite(x).all() and np.abs(x).max() <= 1.0
    assert x[0, lay.IN_PITCH] == pytest.approx(sign * math.sin(PM), abs=1e-6)
    assert abs(x[0, L_.IN_BOND + L_.BOND_DX]) == 1.0  # 5.5 up is nearly straight ahead at 80°: clipped


def test_build_inputs_3d_compiles(cfg3: Config) -> None:
    rs = np.random.default_rng(13)
    ag = _agents(rs, 20, rs.uniform(1.0, 12.0, 20))
    for a in ag:
        a.update(pitch=rs.uniform(-PM, PM), vel=rs.uniform(-0.5, 0.5, 3))
    s = make_scene(cfg3, ag, bonds=[(0, 1), (2, 3)])
    nbr, geom, fo = _neighbors(s, cfg3), _geom3(s, cfg3), _fobs3(dims(cfg3).n, rs)
    a_ = age(s.step, s.birth_step)
    key = mx.random.key(4)

    def f(s, a, nbr, geom, fo, key):
        desc = sensing.neighbor_descriptors(s, nbr, cfg3)
        gr = physics.grounded(s.pos, cfg3)
        return desc, sensing.build_inputs(s, a, nbr, desc, geom, fo, key, cfg3, grounded=gr)

    e = f(s, a_, nbr, geom, fo, key)
    c = mx.compile(f)(s, a_, nbr, geom, fo, key)
    for u, v in zip(e, c, strict=True):
        np.testing.assert_allclose(np.array(u), np.array(v), atol=1e-6)
