"""§27.4 3D births: the cone placement behind the parent with its vertical reflection, immigrants uniform in
the volume, the buoyancy-cap gene, complete newborn rows, the catastrophe column and compile. §27.6 corpses
sink to the floor cell below the dead agent."""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from alife import bonds, fields, layout, lifecycle, ops, rng
from alife.config import Config, apply_overrides, dims
from alife.layout import T
from alife.lifecycle import LaneKind
from alife.rng import Stream
from alife.state import AGENT_FIELDS_3D, R_HARD, State3, init_state, to_numpy, total_energy
from helpers import ON_GPU, as_3d, make_scene, volume_draws

f32 = np.float32
PM32 = f32(layout.PITCH_MAX)
BUOY = {"physics": {"buoyancy_max": 0.05}, "mutation": {"buoy_init": 0.25, "buoy_sigma": 0.02}}


@pytest.fixture(scope="module")
def cfg3(micro_cfg: Config) -> Config:
    """``micro`` in 3D (N 64, M 8, L 48, height 24) with an active buoyancy cap."""
    return as_3d(micro_cfg, **BUOY)


@pytest.fixture(scope="module")
def imm3(micro_cfg: Config) -> Config:
    return as_3d(apply_overrides(micro_cfg, ["population.immigration_floor=40"]), **BUOY)


def _np(a: mx.array) -> np.ndarray:
    return to_numpy(a)


def _host(s: State3) -> dict[str, np.ndarray]:
    return {k: _np(v) for k, v in s._asdict().items()}


def _reproduce(s: State3, req: np.ndarray, child_bond: np.ndarray, cfg: Config, seed: int = 3):
    _, ks = rng.step_keys(mx.random.key(seed))
    lanes = lifecycle.allocate_lanes(mx.array(req), s.alive, ks[Stream.BIRTH_PRIO], cfg)
    return ks, lanes, lifecycle.reproduce(s, lanes, mx.array(child_bond), ks, cfg)


def _place(cfg: Config, pos_p, head_p, pitch_p, u) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    args = (mx.array(np.asarray(a, f32)) for a in (pos_p, head_p, pitch_p, u))
    return tuple(_np(a) for a in lifecycle._offspring_kinematics_3d(*args, cfg))


def _fwd(yaw: np.ndarray, pitch: np.ndarray) -> np.ndarray:
    y, p = yaw.astype(np.float64), pitch.astype(np.float64)
    return np.stack([np.cos(p) * np.cos(y), np.cos(p) * np.sin(y), np.sin(p)], axis=-1)


def _offset(child: np.ndarray, parent: np.ndarray, L: float) -> np.ndarray:
    """float64 parent → child offsets: minimum image in x and y, plain in z."""
    off = child.astype(np.float64) - parent
    off[..., :2] -= L * np.round(off[..., :2] / L)
    return off


# --- placement ---------------------------------------------------------------------------------------------


def test_level_parent_without_pitch_jitter_is_the_2d_rule(cfg3: Config) -> None:
    """Pitch 0 and no pitch jitter: x and y are §12.3's 2D placement (bit for bit on CPU) and z is the
    parent's; so zero jitter is exactly the 2D rule."""
    d, L, D = dims(cfg3), cfg3.world.size, cfg3.repro.offspring_distance
    rs = np.random.default_rng(1)
    m = d.m
    pos_p = np.stack([rs.uniform(0, L, m), rs.uniform(0, L, m), rs.uniform(6, 18, m)], 1).astype(f32)
    pos_p[0, :2] = (L - 0.3, 0.2)  # the child crosses a seam
    head_p = rs.uniform(-math.pi, math.pi, m).astype(f32)
    u0 = rs.uniform(-0.5, 0.5, m).astype(f32)
    for jitter in (np.zeros(m, f32), u0):
        pos, heading, pitch = _place(cfg3, pos_p, head_p, np.zeros(m), np.stack([jitter, np.zeros(m)], 1))
        phi = mx.array(head_p) + math.pi + mx.array(jitter)  # the 2D rule (lifecycle.reproduce, §12.3)
        pos2 = _np(ops.wrap(mx.array(pos_p[:, :2]) + D * mx.stack([mx.cos(phi), mx.sin(phi)], axis=1), L))
        if ON_GPU:
            np.testing.assert_allclose(pos[:, :2], pos2, atol=1e-5)
        else:
            assert np.array_equal(pos[:, :2], pos2)
        assert np.array_equal(heading, _np(ops.wrap_angle(phi)))
        assert np.array_equal(pos[:, 2], pos_p[:, 2]) and not pitch.any()


def test_placement_keeps_the_offset_and_reflects_at_the_floor_and_surface(cfg3: Config) -> None:
    d, L, D = dims(cfg3), cfg3.world.size, cfg3.repro.offspring_distance
    rs = np.random.default_rng(2)
    zl, zh = d.z_lo, d.z_hi
    # floor, nose up (0, 1); surface, nose down (2, 3); mid-water (4, 5); floor, level (6); near it, steep (7)
    z_p = np.array([zl, zl, zh, zh, 12.0, 9.0, zl, zl + 1.0], f32)
    pitch_p = np.array([0.6, 0.45, -0.6, -0.5, 0.4, -0.55, 0.0, 1.2], f32)
    u = rs.uniform(-0.5, 0.5, (8, 2)).astype(f32)
    u[6, 1] = -0.4  # a level parent on the floor with a downward jitter
    pos_p = np.stack([rs.uniform(0, L, 8), rs.uniform(0, L, 8), z_p], 1).astype(f32)
    pos_p[1, :2] = (0.1, L - 0.1)  # the seams
    head_p = rs.uniform(-math.pi, math.pi, 8).astype(f32)
    pos, heading, pitch = _place(cfg3, pos_p, head_p, pitch_p, u)
    off = _offset(pos, pos_p, L)
    assert np.allclose(np.linalg.norm(off, axis=1), D, rtol=1e-5)  # the child bond's rest length, never less
    assert np.all((pos[:, :2] >= 0) & (pos[:, :2] < L))
    assert np.all((pos[:, 2] >= f32(zl)) & (pos[:, 2] <= f32(zh)))
    assert np.allclose(D * _fwd(heading, pitch), off, atol=2e-5)  # the child faces away from its parent
    assert np.all((heading >= -math.pi) & (heading < math.pi)) and np.all(np.abs(pitch) <= PM32)
    behind = np.clip(u[:, 1] - pitch_p, -PM32, PM32)
    reflected = np.array([1, 1, 1, 1, 0, 0, 1, 1], bool)
    assert np.allclose(pitch, np.where(reflected, -behind, behind), atol=1e-7)
    assert np.all(pos[[0, 1, 6, 7], 2] > pos_p[[0, 1, 6, 7], 2])  # reflected off the floor
    assert np.all(pos[[2, 3], 2] < pos_p[[2, 3], 2])  # reflected off the surface
    back = -_fwd(head_p, pitch_p)  # the cone behind the parent (no pitch clip on these rows)
    keep = ~reflected
    assert np.all((off[keep] / D * back[keep]).sum(1) >= math.cos(0.75))


def test_reproduce_3d_writes_every_child_row(cfg3: Config) -> None:
    """Birth lanes: placement from ``PLACEMENT`` (M,2), the five genes from ``split(MUT_GENES, 5)``, the
    pitch and buoy_cap rows; unstressed child bonds; every other row untouched."""
    d, L = dims(cfg3), cfg3.world.size
    parents = [
        dict(pos=(5.0, 5.0, d.z_lo), heading=0.5, pitch=0.6, buoy_cap=0.0),
        dict(pos=(L - 0.5, 30.0, d.z_hi), heading=-2.0, pitch=-0.4, buoy_cap=1.0),
        dict(pos=(20.0, 0.3, 12.0), heading=3.0, pitch=1.3, buoy_cap=0.4),
        dict(pos=(30.0, 20.0, 7.0), heading=-0.1, pitch=-1.2, buoy_cap=0.7),
    ]
    others = [dict(pos=(10.0 + 3 * i, 40.0, 4.0 + i), energy=1_000, pitch=0.1 * i) for i in range(4)]
    agents = [dict(a, energy=60_000, birth_step=-50) for a in parents] + others
    s = make_scene(cfg3, agents, bonds=[(1, 4)], step=7, next_uid=100)
    req = np.arange(d.n) < 4
    ks, lanes, rp = _reproduce(s, req, req, cfg3)
    ns = rp.state
    assert type(ns) is State3
    p, c = _np(lanes.parent)[:4], _np(lanes.slot)[:4]
    assert sorted(p) == [0, 1, 2, 3] and np.all(_np(lanes.kind)[:4] == LaneKind.BIRTH)
    a0, a1 = _host(s), _host(ns)

    u = mx.random.uniform(-0.5, 0.5, (d.m, 2), key=ks[Stream.PLACEMENT])
    par = lanes.parent
    pl = lifecycle._offspring_kinematics_3d(s.pos[par], s.heading[par], s.pitch[par], u, cfg3)
    for name, ref in zip(("pos", "heading", "pitch"), pl, strict=True):
        assert np.array_equal(a1[name][c], _np(ref)[:4]), name
    off = _offset(a1["pos"][c], a0["pos"][p], L)
    assert np.allclose(np.linalg.norm(off, axis=1), cfg3.repro.offspring_distance, rtol=1e-5)
    assert not a1["vel"][c].any() and a1["vel"].shape == (d.n, 3)

    genes = _gene_ref3(s, _np(lanes.parent), ks[Stream.MUT_GENES], cfg3)
    for name, ref in zip(("mut_sigma", "mut_rate", "tag", "offspring_frac", "buoy_cap"), genes, strict=True):
        np.testing.assert_allclose(a1[name][c], ref[:4], rtol=1e-5, atol=1e-7, err_msg=name)
    assert np.all((a1["buoy_cap"][c] >= 0) & (a1["buoy_cap"][c] <= 1))

    geom = bonds.bond_geometry(ns.pos, ns.partner, cfg3)
    ln, valid = _np(geom.length), _np(geom.valid)
    for ch in c:
        assert valid[ch, 0] and np.isclose(ln[ch, 0], cfg3.bonds.rest_length, rtol=1e-5)  # unstressed

    other = np.ones(d.n, bool)
    other[c] = False
    for name in AGENT_FIELDS_3D:
        rows = other.copy()
        if name in ("energy", "partner", "rev"):
            rows[p] = False
        assert np.array_equal(a0[name][rows], a1[name][rows]), name


def _gene_ref3(s: State3, p: np.ndarray, key: mx.array, cfg: Config) -> tuple[np.ndarray, ...]:
    """numpy reference of the 3D ``_child_genes``: ``split(key, 5)``, the 2D formulas, then the cap."""
    mu, m = cfg.mutation, p.shape[0]
    k = mx.random.split(key, 5)
    n1, n2 = _np(mx.random.normal((m,), key=k[0])), _np(mx.random.normal((m,), key=k[1]))
    n3, n4 = _np(mx.random.normal((m, T), key=k[2])), _np(mx.random.normal((m,), key=k[3]))
    n5 = _np(mx.random.normal((m,), key=k[4]))

    def logit(x: np.ndarray) -> np.ndarray:
        return np.log(x / (f32(1) - x))

    def sig(x: np.ndarray) -> np.ndarray:
        return f32(1) / (f32(1) + np.exp(-x))

    a = _host(s)
    sigma = np.clip(a["mut_sigma"][p] * np.exp(f32(mu.tau_sigma) * n1), f32(mu.sigma_min), f32(mu.sigma_max))
    rate = np.clip(sig(logit(a["mut_rate"][p]) + f32(mu.tau_rate) * n2), f32(mu.rate_min), f32(mu.rate_max))
    tag = np.clip(a["tag"][p] + f32(mu.tag_sigma) * n3, -1, 1)
    frac = np.clip(
        sig(logit(a["offspring_frac"][p]) + f32(mu.tau_frac) * n4), f32(mu.frac_min), f32(mu.frac_max)
    )
    cap = np.clip(a["buoy_cap"][p] + f32(mu.buoy_sigma) * n5, f32(0), f32(1))
    return sigma, rate, tag, frac, cap


def test_child_genes_3d_match_the_reference_and_clip_at_0_and_1(micro_cfg: Config) -> None:
    cfg = as_3d(micro_cfg, physics={"buoyancy_max": 0.05}, mutation={"buoy_init": 0.5, "buoy_sigma": 0.5})
    caps = [0.0, 1.0, 0.0, 1.0, 0.5, 0.02, 0.98, 0.3]
    s = make_scene(cfg, [dict(buoy_cap=c, tag=(0.3, -0.2, 0.1, 0.0)) for c in caps])
    p = mx.arange(8, dtype=mx.int32)
    hit0 = hit1 = False
    for seed in range(6):
        key = mx.random.key(seed)
        got = [_np(g) for g in lifecycle._child_genes(s, p, key, cfg)]
        for name, a, b in zip(("sigma", "rate", "tag", "frac", "cap"), got, _gene_ref3(s, _np(p), key, cfg),
                              strict=True):  # fmt: skip
            np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-7, err_msg=name)
        cap = got[4]
        assert cap.dtype == np.float32 and np.all((cap >= 0) & (cap <= 1))
        hit0, hit1 = hit0 or bool((cap == 0).any()), hit1 or bool((cap == 1).any())
    assert hit0 and hit1  # the clip makes both ends reachable (and, being additive, leavable)

    # The fifth draw exists whenever dims = 3: with the actuator off the other genes are drawn identically.
    off = as_3d(micro_cfg)
    s_off = make_scene(off, [dict(buoy_cap=c, tag=(0.3, -0.2, 0.1, 0.0)) for c in caps])
    key = mx.random.key(4)
    on_g = [_np(g) for g in lifecycle._child_genes(s, p, key, cfg)]
    off_g = [_np(g) for g in lifecycle._child_genes(s_off, p, key, off)]
    for a, b in zip(on_g[:4], off_g[:4], strict=True):
        assert np.array_equal(a, b)
    assert np.array_equal(off_g[4], np.array(caps, f32))  # buoy_sigma = 0: the cap is inherited unchanged
    assert lifecycle._child_genes(make_scene(micro_cfg, [{}]), p, key, micro_cfg)[4] is None  # 2D: four genes


# --- immigrants and newborn rows ----------------------------------------------------------------------------


def test_immigrants_are_uniform_in_the_volume_with_buoy_init(imm3: Config) -> None:
    d, L, mu = dims(imm3), imm3.world.size, imm3.mutation
    agents = [
        dict(energy=80_000, birth_step=-40, pos=(10.0, 10.0, 5.0), pitch=0.3, buoy_cap=0.9),
        dict(energy=5_000, pos=(20.0, 20.0, 10.0)),
    ]
    s = make_scene(imm3, agents, step=4)
    req = np.arange(d.n) == 0
    ks, lanes, rp = _reproduce(s, req, np.zeros(d.n, bool), imm3)
    kind, slot = _np(lanes.kind), _np(lanes.slot)
    assert kind[0] == LaneKind.BIRTH and np.all(kind[1:] == LaneKind.IMMIGRANT)
    a1, im = _host(rp.state), slot[1:]
    k = mx.random.split(ks[Stream.IMMIGRANT], 5)
    ref = [x[1:] for x in volume_draws(k[0], k[1], d.m, imm3)]  # lane j takes row j of the IMMIGRANT draws
    for name, r in zip(("pos", "heading", "pitch"), ref, strict=True):
        assert np.array_equal(a1[name][im], r), name
    pos = a1["pos"][im]
    assert np.all((pos[:, :2] >= 0) & (pos[:, :2] < L))
    assert np.all((pos[:, 2] >= f32(d.z_lo)) & (pos[:, 2] <= f32(d.z_hi)))
    assert pos[:, 2].std() > 2.0  # spread through the volume, not on the floor
    assert np.all(np.abs(a1["pitch"][im]) <= PM32) and not a1["vel"][im].any()
    assert np.all(a1["buoy_cap"][im] == f32(mu.buoy_init))  # immigrants start at buoy_init, no draw
    cap = _gene_ref3(s, _np(lanes.parent), ks[Stream.MUT_GENES], imm3)[4]
    np.testing.assert_allclose(
        a1["buoy_cap"][slot[0]], cap[0], atol=1e-7
    )  # the birth lane mutates its parent's


@pytest.mark.parametrize("lane", ["birth", "immigrant"])
def test_newborn_rows_are_complete_in_3d(cfg3: Config, imm3: Config, lane: str) -> None:
    """A dead row full of sentinel garbage becomes a clean newborn after reproduce and the next S0 flush,
    including the 3D-only ``pitch`` and ``buoy_cap`` (a missed write would leave a stale in-range value)."""
    cfg = cfg3 if lane == "birth" else imm3
    d = dims(cfg)
    energy = 90_000 if lane == "birth" else 5_000
    s = make_scene(cfg, [dict(energy=energy, pos=(5.0, 5.0, 8.0), birth_step=-40)], step=2)
    target = 1  # the lowest dead row
    sentinel = {
        "uid": 987_654_321, "parent_uid": 55_555, "clade": 77, "generation": 999, "birth_step": 4242,
        "energy": 31_337, "pos": 999.0, "vel": 9.5, "heading": 99.0, "h": 9.0, "msg": 9.0,
        "last_loss": 31_337, "last_harvest": 31_337, "last_inflow": 31_337, "partner": 33, "rev": 3,
        "tag": 9.0, "mut_sigma": 9.0, "mut_rate": 9.0, "offspring_frac": 9.0, "brain_cost": 31_337,
        "W1": 3.25, "W2": 3.25, "pitch": 99.0, "buoy_cap": 9.0,
    }  # fmt: skip
    assert set(sentinel) == set(AGENT_FIELDS_3D) - {"alive"}
    garbage = {}
    for name, v in sentinel.items():
        a = _np(getattr(s, name)).copy()
        a[target] = v
        garbage[name] = mx.array(a)
    s = s._replace(**garbage)
    req = np.arange(d.n) == 0 if lane == "birth" else np.zeros(d.n, bool)
    _, lanes, rp = _reproduce(s, req, req, cfg)
    assert int(_np(lanes.slot)[0]) == target
    assert _np(lanes.kind)[0] == (LaneKind.BIRTH if lane == "birth" else LaneKind.IMMIGRANT)
    ns = rp.state
    W1, W2 = lifecycle.flush_pending(ns.W1, ns.W2, ns.pend_slot, ns.pend_W1, ns.pend_W2)  # next step's S0
    ns = ns._replace(W1=W1, W2=W2)
    for name, v in sentinel.items():
        row = _np(getattr(ns, name))[target]
        assert not np.any(row == v), name
    assert bool(_np(ns.alive)[target])


# --- catastrophe -------------------------------------------------------------------------------------------


def test_catastrophe_is_a_vertical_column(cfg3: Config, micro_cfg: Config) -> None:
    d, L = dims(cfg3), cfg3.world.size
    pos = np.zeros((d.n, 3), np.float32)
    pos[:, 2] = d.z_lo
    pos[:6] = [
        [1.0, 1.0, d.z_lo],  # inside the disc, on the floor
        [1.0, 1.0, d.z_hi],  # inside the disc, at the surface
        [L - 1.0, L - 1.0, 12.0],  # inside, across both seams
        [L / 2, L / 2, 5.0],  # outside
        [10.0, 1.0, 20.0],  # outside
        [0.0, 0.0, 15.0],  # at the centre
    ]
    env = init_state(cfg3).env._replace(
        cat_center=mx.array([0.0, 0.0]), cat_radius=mx.array(3.0), cat_kill_p=mx.array(1.0)
    )
    assert env.cat_center.shape == (2,)
    alive = mx.array(np.arange(d.n) < 6)
    mask = _np(lifecycle.catastrophe_mask(mx.array(pos), alive, env, mx.random.key(1), cfg3))
    assert mask[:6].tolist() == [True, True, True, False, False, True] and not mask[6:].any()
    flat = _np(lifecycle.catastrophe_mask(mx.array(pos[:, :2]), alive, env, mx.random.key(1), micro_cfg))
    assert np.array_equal(mask, flat)  # the column is the 2D disc at every height


# --- corpses sink (§27.6) -----------------------------------------------------------------------------------


def _deaths(s: State3, solvent: np.ndarray, e: np.ndarray, cfg: Config) -> lifecycle.Deaths:
    zero = mx.zeros(s.alive.shape, mx.int32)
    return lifecycle.deaths(
        s.alive, mx.array(solvent), mx.array(e), zero, zero, s.uid, s.generation, s.clade, s.partner, s.rev,
        s.resource, fields.cell_index(s.pos, cfg), cfg,
    )  # fmt: skip


def _column_scene(cfg: Config, resource: np.ndarray, floor: int) -> tuple[State3, np.ndarray, np.ndarray]:
    """One agent per layer stacked over the cell (cx, cy) = (2, 3), plus one survivor elsewhere; the floor
    cell of the column holds ``floor``. Rows 1..FZ−1 above the floor die (insolvent, E = 100·(row + 1))."""
    d, fc = dims(cfg), cfg.world.field_cell
    resource = resource.copy()
    resource[0, 3, 2] = floor
    zs = [d.z_lo, *((k + 0.5) * fc for k in range(1, d.fz - 1)), d.z_hi]  # layer k holds row k
    agents = [{"pos": [2.5 * fc, 3.5 * fc, z]} for z in zs] + [{"pos": [9.5 * fc, 9.5 * fc, 10.0]}]
    s = make_scene(cfg, agents, resource=resource)
    n, rows = dims(cfg).n, np.arange(1, d.fz)
    solvent, e = np.array(s.alive), np.zeros(n, np.int32)
    solvent[rows] = False
    e[: len(agents)] = 100 * np.arange(1, len(agents) + 1)
    return s, solvent, e


def test_corpses_sink_to_the_floor_cell(cfg3: Config) -> None:
    """A corpse from layer k lands in the floor cell (0, cy, cx) below it, not in its own cell; every other
    cell is untouched and the corpse term is unchanged (FLOW_CORPSE)."""
    d, S = dims(cfg3), cfg3.energy.structure
    resource = np.random.default_rng(1).integers(0, 8000, d.field_shape).astype(np.int32)
    s, solvent, e = _column_scene(cfg3, resource, 500)
    fcell = _np(fields.cell_index(s.pos, cfg3))
    assert sorted(fcell[: d.fz] // (d.fg * d.fg)) == list(range(d.fz))  # one agent per layer
    assert len(set(fcell[: d.fz] % (d.fg * d.fg))) == 1  # one column
    dt = _deaths(s, solvent, e, cfg3)
    dead = _np(dt.dead)
    assert np.array_equal(np.flatnonzero(dead), np.arange(1, d.fz))
    corpse = np.where(dead, e + S, 0)
    assert np.array_equal(_np(dt.corpse), corpse)
    want = _np(s.resource).astype(np.int64)
    want[0, 3, 2] += corpse.sum()
    assert dt.resource.shape == d.field_shape and dt.resource.dtype == mx.int32
    assert np.array_equal(_np(dt.resource), want)
    assert dt.r_overflow.shape == d.field_shape and not _np(dt.r_overflow).any()
    # The conserved total does not move: corpses are internal (§10.8).
    before = int(e.sum()) + S * int(_np(s.alive).sum()) + int(_np(s.resource).astype(np.int64).sum())
    assert before == int(_np(total_energy(dt.energy, dt.survivor, dt.resource, cfg3)))


def test_corpses_on_a_full_floor_cell_are_capped(cfg3: Config) -> None:
    """The R_HARD cap applies at the floor cell the corpses sink into, and only there; the excess is the
    overflow (HEAT_OVERFLOW). Compiled equals eager (integers)."""
    d, S = dims(cfg3), cfg3.energy.structure
    resource = np.full(d.field_shape, 1000, np.int32)
    s, solvent, e = _column_scene(cfg3, resource, R_HARD - 100)
    corpse = int(np.where(~solvent & _np(s.alive), e + S, 0).sum())
    assert corpse > 100
    dt = _deaths(s, solvent, e, cfg3)
    over = np.zeros(d.field_shape, np.int64)
    over[0, 3, 2] = corpse - 100
    assert np.array_equal(_np(dt.r_overflow), over)
    want = _np(s.resource).astype(np.int64)
    want[0, 3, 2] = R_HARD
    assert np.array_equal(_np(dt.resource), want)
    comp = mx.compile(lambda s_: _deaths(s_, solvent, e, cfg3))(s)
    for a, b in zip(dt, comp, strict=True):
        if isinstance(a, mx.array):
            assert np.array_equal(_np(a), _np(b))


# --- compile -----------------------------------------------------------------------------------------------


def test_reproduce_3d_compiles_and_matches_eager(imm3: Config) -> None:
    """S0 + the catastrophe mask + S16 in 3D: compiled equals eager (integers exactly; floats to rounding,
    since fused kernels may contract the weight mutation)."""
    cfg = imm3
    agents = [
        dict(
            energy=30_000 + 7_000 * i,
            pos=(2.0 * i, 3.0, 1.0 + 1.9 * i),
            pitch=0.1 * i - 0.5,
            buoy_cap=0.08 * i,
            birth_step=-25,
        )
        for i in range(12)
    ]
    s = make_scene(cfg, agents, bonds=[(0, 1), (2, 3)], step=4)
    env = s.env._replace(cat_center=mx.array([4.0, 3.0]), cat_radius=mx.array(5.0), cat_kill_p=mx.array(0.5))

    def core(s: State3) -> tuple:
        W1, W2 = lifecycle.flush_pending(s.W1, s.W2, s.pend_slot, s.pend_W1, s.pend_W2)
        s = s._replace(W1=W1, W2=W2)
        _, ks = rng.step_keys(s.key)
        cat = lifecycle.catastrophe_mask(s.pos, s.alive, env, ks[Stream.CATASTROPHE], cfg)
        survivor = s.alive & ~cat
        lanes = lifecycle.allocate_lanes(survivor & (s.energy >= 80_000), survivor, ks[2], cfg)
        rp = lifecycle.reproduce(s._replace(alive=survivor), lanes, survivor, ks, cfg)
        return rp.state, rp.records, lanes, rp.endow, cat

    eager, comp = core(s), mx.compile(core)(s)
    mx.eval(eager, comp)
    for a, b in zip(tree_flatten(eager), tree_flatten(comp), strict=True):
        x, y = _np(a[1]), _np(b[1])
        assert a[0] == b[0], a[0]
        if x.dtype.kind == "f":
            np.testing.assert_allclose(x, y, rtol=1e-6, atol=1e-6, err_msg=a[0])
        else:
            assert np.array_equal(x, y), a[0]
    lanes = eager[2]
    assert int(_np(lanes.n_birth)) > 0 and int(_np(lanes.n_imm)) > 0 and _np(eager[4]).any()
