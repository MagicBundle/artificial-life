"""Sensing on the sphere (DESIGN §46.4; sphere series S2): the descriptors and the brain inputs in the body
frame
of the heading vector and the local up.

A hand-built scene fixes the frame (fwd, lat, up), the box's own descriptors are the reference on a locally
flat
patch, and every input stays finite with ``|x| <= 1`` over a stepped state.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from alife import bonds, fields, layout, sensing, spatial
from alife.config import Config, dims, load_config
from alife.fields import FieldObs
from alife.state import StateS, init_state, to_numpy
from alife.step import step
from sphere_fields_helpers import dev  # noqa: F401  (a fixture)


def cfg_of(*extra: str) -> Config:
    return load_config(preset="tiny3s", overrides=["brain.weight_dtype=float32", *extra])


def zero_fields(n: int) -> FieldObs:
    """Field inputs of zeros: the grid-bound part of the brain input removed, to look at the rest."""
    return FieldObs(
        res=mx.zeros((n,)),
        res_grad=mx.zeros((n, 3)),
        chem=mx.zeros((n, layout.NCH)),
        chem_grad=mx.zeros((n, layout.NCH, 3)),
    )


def with_agents(s: StateS, cfg: Config, rows: dict[int, tuple[np.ndarray, np.ndarray, float]]) -> StateS:
    """``s`` with the rows given as ``row: (pos, hdg, pitch)`` placed there (other rows alive where they
    were)."""
    pos, hdg, pitch = (to_numpy(a).copy() for a in (s.pos, s.hdg, s.pitch))
    for r, (p, h, ph) in rows.items():
        pos[r], hdg[r], pitch[r] = p, h, ph
    return s._replace(pos=mx.array(pos), hdg=mx.array(hdg), pitch=mx.array(pitch))


@pytest.mark.usefixtures("dev")
def test_a_neighbour_is_seen_in_the_body_frame_ahead_left_and_above() -> None:
    cfg = cfg_of()
    d, R = dims(cfg), cfg.sensing.radius
    s = init_state(cfg)
    r0 = cfg.world.radius + 6.0
    # agent 0 on the equator at longitude 0: up = x̂, heading north (ẑ): fwd = ẑ, lat = up × hdg = x̂ × ẑ = −ŷ
    me = (np.array([r0, 0, 0], np.float32), np.array([0, 0, 1], np.float32), 0.0)
    ahead = (np.array([r0, 0, 3.0], np.float32), np.array([0, 0, 1], np.float32), 0.0)  # 3 units north
    left = (np.array([r0, -2.5, 0], np.float32), np.array([0, 0, 1], np.float32), 0.0)  # along −ŷ = lat
    above = (np.array([r0 + 2.0, 0, 0], np.float32), np.array([0, 0, 1], np.float32), 0.0)  # along up
    s = with_agents(s, cfg, {0: me, 1: ahead, 2: left, 3: above})
    # keep only these four alive and in view: everything else far away
    alive = np.zeros(d.n, bool)
    alive[:4] = True
    s = s._replace(alive=mx.array(alive))
    pos = to_numpy(s.pos).copy()
    pos[4:] = np.array([0.0, 0.0, -(cfg.world.radius + 6.0)], np.float32)
    s = s._replace(pos=mx.array(pos))
    nb = spatial.find_neighbors(s.pos, s.alive, cfg, impl="mlx")
    desc = np.array(sensing.neighbor_descriptors(s, nb, cfg))
    idx, valid = np.array(nb.idx[0]), np.array(nb.valid[0])
    assert set(idx[valid]) == {1, 2, 3}
    by = {int(idx[k]): desc[0, k] for k in range(len(idx)) if valid[k]}
    # NBR_DX = fwd, NBR_DY = lat, NBR_DZ = up (units of R); the chord of an arc is slightly shorter than
    # the arc
    fwd, lat, up = layout.NBR_DX, layout.NBR_DY, d.lay.NBR_DZ  # NBR_DZ is the last column of the descriptor
    np.testing.assert_allclose(by[1][[fwd, lat, up]], [3.0 / R, 0.0, 0.0], atol=5e-3 / R * 3)
    np.testing.assert_allclose(by[2][[fwd, lat, up]], [0.0, 2.5 / R, 0.0], atol=5e-3 / R * 3)
    np.testing.assert_allclose(by[3][[fwd, lat, up]], [0.0, 0.0, 2.0 / R], atol=2e-6)
    assert all(v[layout.NBR_PRESENT] == 1.0 for v in by.values())


@pytest.mark.usefixtures("dev")
def test_on_a_level_patch_the_descriptors_and_inputs_are_the_boxs() -> None:
    """At the north pole the tangent plane is the xy plane and the up is ẑ: the sphere's brain input vector
    equals the box's for the same scene (the field blocks are zero in both; the heights use the same H)."""
    cs = cfg_of()
    cb = load_config(preset="tiny3d", overrides=["brain.weight_dtype=float32", "world.height=16"])
    ds, db = dims(cs), dims(cb)
    assert ds.lay.D_IN == db.lay.D_IN == 84
    rng = np.random.default_rng(1)
    n = 40
    z = rng.uniform(1.5, 14.0, n)
    theta = rng.uniform(-math.pi, math.pi, n).astype(np.float32)
    pitch = rng.uniform(-1.2, 1.2, n).astype(np.float32)
    ss, sb = init_state(cs), init_state(cb)
    # the sphere's agents on the polar axis (up = ẑ exactly), the box's anywhere: the frame is the same
    pos_s = np.zeros((ds.n, 3), np.float32)
    pos_s[:, 2] = cs.world.radius + 1.0
    pos_s[:n, 2] = cs.world.radius + z
    pos_b = np.zeros((ds.n, 3), np.float32)
    pos_b[:, :2] = rng.uniform(0, 100, (ds.n, 2))
    pos_b[:, 2] = 1.0
    pos_b[:n, 2] = z
    hdg = np.zeros((ds.n, 3), np.float32)
    hdg[:, 0] = 1.0
    hdg[:n] = np.stack([np.cos(theta), np.sin(theta), np.zeros(n)], axis=1)
    alive = np.arange(ds.n) < n
    vel = (rng.normal(size=(ds.n, 3)) * 0.02).astype(np.float32)
    head = np.zeros(ds.n, np.float32)
    head[:n] = theta
    pit = np.zeros(ds.n, np.float32)
    pit[:n] = pitch
    sb = sb._replace(
        pos=mx.array(pos_b),
        alive=mx.array(alive),
        vel=mx.array(vel),
        heading=mx.array(head),
        pitch=mx.array(pit),
    )
    ss = ss._replace(
        pos=mx.array(pos_s),
        alive=mx.array(alive),
        vel=mx.array(vel),
        heading=mx.array(head),
        pitch=mx.array(pit),
        hdg=mx.array(hdg),
    )
    # one hand-made neighbour list in both worlds (the descriptor reads only the chord offsets)
    k = ds.k
    off = rng.normal(0, 2.5, (ds.n, k, 3)).astype(np.float32)
    nbs = spatial.Neighbors(
        idx=mx.array(rng.integers(0, n, (ds.n, k)).astype(np.int32)),
        valid=mx.array(rng.random((ds.n, k)) < 0.6),
        off=mx.array(off),
        dist=mx.array(np.linalg.norm(off, axis=-1)),
        overflow_cells=mx.array(0, dtype=mx.int32),
        max_occupancy=mx.array(0, dtype=mx.int32),
        hidden_agents=mx.array(0, dtype=mx.int32),
    )
    nbb = nbs
    gb, gs = bonds.bond_geometry(sb.pos, sb.partner, cb), bonds.bond_geometry(ss.pos, ss.partner, cs)
    desc_b = sensing.neighbor_descriptors(sb, nbb, cb)
    desc_s = sensing.neighbor_descriptors(ss, nbs, cs)
    np.testing.assert_allclose(np.array(desc_s), np.array(desc_b), atol=2e-6)
    age = mx.zeros((ds.n,), dtype=mx.int32)
    key = mx.random.key(3)
    grounded_b, grounded_s = (
        mx.array(np.array(z_ <= d_.z_ground))
        for z_, d_ in ((pos_b[:, 2], db), (pos_s[:, 2] - cs.world.radius, ds))
    )
    xb = sensing.build_inputs(sb, age, nbb, desc_b, gb, zero_fields(ds.n), key, cb, grounded=grounded_b)
    xs = sensing.build_inputs(ss, age, nbs, desc_s, gs, zero_fields(ds.n), key, cs, grounded=grounded_s)
    np.testing.assert_allclose(np.array(xs)[:n], np.array(xb)[:n], atol=2e-6)


@pytest.mark.usefixtures("dev")
def test_the_inputs_of_a_stepped_state_are_finite_and_bounded() -> None:
    cfg = cfg_of()
    s = init_state(cfg)
    for _ in range(8):
        s, _ = step(s, cfg)
    mx.eval(s)
    nb = spatial.find_neighbors(s.pos, s.alive, cfg, impl="mlx")
    fcell = fields.cell_index(s.pos, cfg)
    fobs = fields.sample_fields(
        s.resource, s.chem, fcell, s.heading, cfg, pitch=s.pitch, pos=s.pos, hdg=s.hdg
    )
    desc = sensing.neighbor_descriptors(s, nb, cfg)
    geom = bonds.bond_geometry(s.pos, s.partner, cfg)
    from alife import physics

    gr = physics.grounded(s.pos, cfg)
    x = np.array(
        sensing.build_inputs(
            s,
            mx.ones((dims(cfg).n,), dtype=mx.int32),
            nb,
            desc,
            geom,
            fobs,
            mx.random.key(0),
            cfg,
            grounded=gr,
        )
    )
    assert x.shape == (dims(cfg).n, 84) and np.isfinite(x).all() and np.abs(x).max() <= 1.0
    # IN_HEIGHT is (|pos| − Rp)/height, in [0, 1], and IN_GROUNDED is the predicate
    alive = np.array(s.alive)
    h = (np.linalg.norm(to_numpy(s.pos), axis=1) - cfg.world.radius) / cfg.world.height
    np.testing.assert_allclose(x[alive, dims(cfg).lay.IN_HEIGHT], h[alive], atol=1e-5)
