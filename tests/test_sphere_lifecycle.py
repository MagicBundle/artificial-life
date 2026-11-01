"""Births, immigrants, deaths and the catastrophe on the sphere (DESIGN §46.4, §46.5; sphere series S2).

Child placement at a chord ``D`` behind the parent in a cone, reflected at the floor and the lid; founders and
immigrants uniform in the shell by volume; the completeness of a newborn row (every field set); corpses to the
floor cell of the column; the catastrophe as a cap of the sphere.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from alife import fields, layout, lifecycle, sphere
from alife.config import Config, dims, load_config
from alife.rng import step_keys
from alife.state import Env, StateS, init_state, to_numpy
from sphere_fields_helpers import dev  # noqa: F401  (a fixture)
from sphere_rotation import random_rotation


def cfg_of(*extra: str) -> Config:
    return load_config(preset="tiny3s", overrides=["brain.weight_dtype=float32", *extra])


def parents(cfg: Config, rng: np.random.Generator, n: int, radii: np.ndarray | None = None):
    """``(pos (n,3), hdg (n,3), pitch (n,))`` of parents at random points of the shell."""
    z = rng.uniform(-1, 1, n)
    lam = rng.uniform(0, 2 * math.pi, n)
    st = np.sqrt(1 - z * z)
    u = np.stack([st * np.cos(lam), st * np.sin(lam), z], axis=1)
    rad = cfg.world.radius + (rng.uniform(1.5, 14.0, n) if radii is None else radii)
    pos = (u * rad[:, None]).astype(np.float32)
    v = rng.normal(size=(n, 3))
    v -= (v * u).sum(axis=1, keepdims=True) * u
    hdg = (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)
    pitch = rng.uniform(-layout.PITCH_MAX, layout.PITCH_MAX, n).astype(np.float32)
    return pos, hdg, pitch


@pytest.mark.usefixtures("dev")
def test_a_child_sits_at_a_chord_d_behind_its_parent_in_a_cone() -> None:
    cfg = cfg_of()
    d, D = dims(cfg), cfg.repro.offspring_distance
    n = 256
    rng = np.random.default_rng(1)
    pos, hdg, pitch = parents(cfg, rng, n)
    u = rng.uniform(-0.5, 0.5, (n, 2)).astype(np.float32)
    cp, ch, cheading, cpitch = lifecycle._offspring_kinematics_sph(
        mx.array(pos), mx.array(hdg), mx.array(pitch), mx.array(u), cfg
    )
    cp, ch, cpitch = (np.array(a, np.float64) for a in (cp, ch, cpitch))
    pos64, hdg64 = pos.astype(np.float64), hdg.astype(np.float64)
    chord = np.linalg.norm(cp - pos64, axis=1)
    np.testing.assert_allclose(chord, D, rtol=2e-5)  # a chord of length D (the bond rest length)
    rad = np.linalg.norm(cp, axis=1)
    assert rad.min() >= float(d.r_lo) - 2e-5 and rad.max() <= float(d.r_hi) + 2e-5  # inside the shell
    assert abs(np.linalg.norm(ch, axis=1) - 1).max() < 4e-7
    assert abs((ch * cp).sum(1) / rad).max() < 4e-7  # the child's heading is a tangent unit vector
    assert (np.abs(cpitch) <= layout.PITCH_MAX + 1e-7).all()
    # behind the parent: the chord points along -hdg turned by the jitter u0 (|u0| <= 0.5) and tilted by
    # the pitch
    horiz = (cp - pos64) - (
        ((cp - pos64) * (pos64 / np.linalg.norm(pos64, axis=1, keepdims=True))).sum(1, keepdims=True)
    ) * (pos64 / np.linalg.norm(pos64, axis=1, keepdims=True))
    horiz /= np.linalg.norm(horiz, axis=1, keepdims=True)
    ang = np.arccos(np.clip((horiz * -hdg64).sum(axis=1), -1, 1))
    assert ang.max() < 0.5 + 0.05
    # the child faces away from the parent: its heading points along the way from the parent (to the jitter)
    away = (ch * (cp - pos64)).sum(1) / chord
    assert away.min() > math.cos(0.5 + 0.1 + layout.PITCH_MAX) - 0.05
    # the derived heading is the bearing of the child's heading vector
    want = np.array(sphere.bearing(mx.array(ch.astype(np.float32)), mx.array(cp.astype(np.float32))))
    assert (np.abs((np.array(cheading) - want + math.pi) % (2 * math.pi) - math.pi) < 1e-5).all()


@pytest.mark.usefixtures("dev")
def test_with_no_jitter_the_child_is_straight_behind_and_level() -> None:
    cfg = cfg_of()
    D = cfg.repro.offspring_distance
    n = 32
    pos, hdg, _ = parents(cfg, np.random.default_rng(2), n)
    zeros = mx.zeros((n,))
    cp, _ch, _, cpitch = lifecycle._offspring_kinematics_sph(
        mx.array(pos), mx.array(hdg), zeros, mx.zeros((n, 2)), cfg
    )
    cp = np.array(cp, np.float64)
    off = cp - pos.astype(np.float64)
    assert np.array(cpitch).max() == 0.0  # u1 − φ_p = 0
    np.testing.assert_allclose(np.linalg.norm(off, axis=1), D, rtol=2e-5)
    cosang = (off / np.linalg.norm(off, axis=1, keepdims=True) * -hdg.astype(np.float64)).sum(axis=1)
    assert cosang.min() > 1 - 2e-3  # the chord of an arc D/2R off the tangent: sin(σ/2) <= 0.042
    # level: the child is at its parent's height (to the chord's curvature term)
    h_par = np.linalg.norm(pos.astype(np.float64), axis=1)
    np.testing.assert_allclose(np.linalg.norm(cp, axis=1), h_par, atol=1e-4)


@pytest.mark.usefixtures("dev")
def test_the_cone_reflects_at_the_floor_and_the_lid() -> None:
    """A parent on the floor whose pitch would put the child under it gets the reflected pitch (and a lid
    parent
    likewise), so the child is inside the shell with the chord still D."""
    cfg = cfg_of()
    d, rp, D = dims(cfg), cfg.world.radius, cfg.repro.offspring_distance
    n = 64
    rng = np.random.default_rng(3)
    floor = np.full(n, float(d.r_lo) - rp, np.float32)
    lid = np.full(n, float(d.r_hi) - rp, np.float32)
    for radii, sign in ((floor, -1.0), (lid, +1.0)):
        pos, hdg, _ = parents(cfg, rng, n, radii)
        # u1 = ±0.5 and parent pitch ∓ (PITCH_MAX − 0.5 .. ) chosen so that χ = clip(u1 − φ_p) points out
        # of the shell
        pitch = np.full(n, -sign * 0.8, np.float32)
        u = np.stack([rng.uniform(-0.5, 0.5, n), np.full(n, sign * 0.5)], axis=1).astype(np.float32)
        cp, _, _, cpitch = lifecycle._offspring_kinematics_sph(
            mx.array(pos), mx.array(hdg), mx.array(pitch), mx.array(u), cfg
        )
        r_new = np.linalg.norm(np.array(cp, np.float64), axis=1)
        assert r_new.min() >= float(d.r_lo) - 2e-5 and r_new.max() <= float(d.r_hi) + 2e-5
        assert (np.sign(np.array(cpitch)) == -sign).all()  # reflected: χ = −clip(...) points into the shell
        np.testing.assert_allclose(
            np.linalg.norm(np.array(cp, np.float64) - pos.astype(np.float64), axis=1), D, rtol=2e-5
        )


def test_a_child_born_at_a_pole_is_well_defined() -> None:
    cfg = cfg_of()
    n = 4
    pos = np.array([[0.0, 0.0, 44.0], [0.0, 1e-4, -44.0], [1e-5, 0.0, 44.0], [0.0, 0.0, -44.0]], np.float32)
    hdg = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], np.float32)
    cp, ch, cheading, _ = lifecycle._offspring_kinematics_sph(
        mx.array(pos), mx.array(hdg), mx.zeros((n,)), mx.zeros((n, 2)), cfg
    )
    cp, ch = np.array(cp, np.float64), np.array(ch, np.float64)
    assert np.isfinite(cp).all() and np.isfinite(ch).all() and np.isfinite(np.array(cheading)).all()
    assert (
        abs(np.linalg.norm(ch, axis=1) - 1).max() < 4e-7
        and abs((ch * cp).sum(1) / np.linalg.norm(cp, axis=1)).max() < 4e-7
    )


def lanes(cfg: Config, kinds: list[int], parent_rows: list[int]) -> lifecycle.Lanes:
    m = dims(cfg).m
    kind = np.zeros(m, np.int32)
    kind[: len(kinds)] = kinds
    parent = np.zeros(m, np.int32)
    parent[: len(parent_rows)] = parent_rows
    from alife import ops

    return lifecycle.Lanes(
        parent=mx.array(parent),
        slot=ops.rank_select(~mx.array(np.arange(dims(cfg).n) < cfg.population.n_init), m)[0],
        kind=mx.array(kind),
        n_req=mx.array(len(kinds), dtype=mx.int32),
        n_birth=mx.array(sum(k == lifecycle.LaneKind.BIRTH.value for k in kinds), dtype=mx.int32),
        n_imm=mx.array(sum(k == lifecycle.LaneKind.IMMIGRANT.value for k in kinds), dtype=mx.int32),
    )


@pytest.mark.usefixtures("dev")
def test_a_newborn_row_is_complete_births_and_immigrants() -> None:
    """Every per-agent field of a newborn row is written: births from their parent (a chord D away, its energy
    and genes, ``generation + 1``), immigrants fresh (uniform in the shell, heading consistent); the row is a
    valid ``StateS`` row (unit tangent heading, radius in range, pitch in range, velocity and memory zero)."""
    cfg = cfg_of("population.immigration_every=1", "population.immigration_count=8")
    d, D = dims(cfg), cfg.repro.offspring_distance
    s = init_state(cfg)
    assert isinstance(s, StateS)
    births, imms = [0, 1, 2, 3, 4, 5], 6
    ln = lanes(
        cfg, [lifecycle.LaneKind.BIRTH.value] * 6 + [lifecycle.LaneKind.IMMIGRANT.value] * imms, births
    )
    ks = step_keys(s.key)[1]
    child_bond = mx.zeros((d.n,), dtype=mx.bool_)
    rp = lifecycle.reproduce(s._replace(step=mx.array(7, dtype=mx.int32)), ln, child_bond, ks, cfg)
    new = rp.state
    slots = to_numpy(ln.slot)[: 6 + imms]
    a = {
        name: to_numpy(getattr(new, name))
        for name in type(new)._fields
        if name
        in (
            "alive",
            "uid",
            "parent_uid",
            "clade",
            "generation",
            "birth_step",
            "energy",
            "pos",
            "vel",
            "hdg",
            "heading",
            "pitch",
            "buoy_cap",
            "h",
            "msg",
            "last_loss",
            "last_harvest",
            "last_inflow",
            "partner",
            "rev",
            "tag",
            "mut_sigma",
            "mut_rate",
            "offspring_frac",
            "brain_cost",
        )
    }
    old = {name: to_numpy(getattr(s, name)) for name in a}
    assert a["alive"][slots].all() and not old["alive"][slots].any()
    assert (a["uid"][slots] == np.arange(int(to_numpy(s.next_uid)), int(to_numpy(s.next_uid)) + 12)).all()
    assert (a["birth_step"][slots] == 7).all()
    for i, slot in enumerate(slots):
        assert not a["vel"][slot].any() and not a["h"][slot].any() and not a["msg"][slot].any()
        assert (a["partner"][slot] == -1).all() and (a["rev"][slot] == -1).all()
        assert a["last_loss"][slot] == a["last_harvest"][slot] == a["last_inflow"][slot] == 0
        pos = a["pos"][slot].astype(np.float64)
        hdg = a["hdg"][slot].astype(np.float64)
        r = np.linalg.norm(pos)
        assert float(d.r_lo) - 2e-5 <= r <= float(d.r_hi) + 2e-5
        assert abs(np.linalg.norm(hdg) - 1) < 4e-7 and abs(hdg @ pos / r) < 4e-7
        assert abs(a["pitch"][slot]) <= layout.PITCH_MAX and 0.0 <= a["buoy_cap"][slot] <= 1.0
        want = float(
            sphere.bearing(mx.array(hdg[None].astype(np.float32)), mx.array(pos[None].astype(np.float32)))[0]
        )
        assert abs((a["heading"][slot] - want + math.pi) % (2 * math.pi) - math.pi) < 1e-5
        if i < 6:  # a birth: a chord D from the parent, the parent's lineage
            par = old["pos"][births[i]].astype(np.float64)
            assert abs(np.linalg.norm(pos - par) - D) < 5e-5 * D
            assert (
                a["parent_uid"][slot] == old["uid"][births[i]]
                and a["generation"][slot] == old["generation"][births[i]] + 1
            )
            assert a["clade"][slot] == old["clade"][births[i]]
            assert a["energy"][slot] < old["energy"][births[i]] and a["energy"][slot] > 0
        else:  # an immigrant: no parent, a fresh clade, the initial energy, the initial genes
            assert a["parent_uid"][slot] == -1 and a["generation"][slot] == 0
            assert a["energy"][slot] == cfg.energy.e_init and a["clade"][slot] == cfg.population.n_init
            assert a["mut_sigma"][slot] == np.float32(cfg.mutation.sigma_init)
    assert int(to_numpy(new.next_uid)) == int(to_numpy(s.next_uid)) + 12
    # rows that did not get a lane are untouched
    keep = np.ones(d.n, bool)
    keep[slots] = False
    for name in ("pos", "hdg", "heading", "pitch", "vel", "alive"):
        if name != "alive":
            assert np.array_equal(
                a[name][keep & (np.arange(d.n) >= 12)], old[name][keep & (np.arange(d.n) >= 12)]
            )


@pytest.mark.usefixtures("dev")
def test_immigrants_are_uniform_in_the_shell_by_volume() -> None:
    """The founders' draw (``state.sphere_kinematics``) is what an immigrant lane uses: the direction is equal
    area and ``r³`` is uniform between the bounds (S1 tests the distribution of the founders; the
    immigrants of a
    reproduce call must be the same function of their keys)."""
    from alife.state import sphere_kinematics

    cfg = cfg_of("population.immigration_every=1", "population.immigration_count=8")
    ks = step_keys(init_state(cfg).key)[1]
    k_pos, k_head, *_ = mx.random.split(ks[lifecycle.Stream.IMMIGRANT.value], 5)
    pos, hdg, yaw, _pitch = sphere_kinematics(k_pos, k_head, 32, cfg)
    s = init_state(cfg)
    ln = lanes(cfg, [lifecycle.LaneKind.IMMIGRANT.value] * 32, [])
    new = lifecycle.reproduce(s, ln, mx.zeros((dims(cfg).n,), dtype=mx.bool_), ks, cfg).state
    slots = to_numpy(ln.slot)[:32]
    assert np.array_equal(to_numpy(new.pos)[slots], to_numpy(pos))
    assert np.array_equal(to_numpy(new.hdg)[slots], to_numpy(hdg))
    assert np.array_equal(to_numpy(new.heading)[slots], to_numpy(yaw))


# --- corpses and the catastrophe ----------------------------------------------------------------------------


@pytest.mark.usefixtures("dev")
def test_corpses_land_in_the_floor_cell_of_their_column() -> None:
    cfg = cfg_of()
    d = dims(cfg)
    s = init_state(cfg)
    n0 = cfg.population.n_init
    fcell = fields.cell_index(s.pos, cfg)
    layers = to_numpy(fcell)[:n0] // d.npix
    assert layers.max() >= 3  # founders live at all heights
    solvent = np.ones(d.n, bool)
    dead = np.zeros(d.n, bool)
    dead[:n0:3] = True
    solvent[dead] = False
    res0 = mx.zeros(d.field_shape, dtype=mx.int32)
    out = lifecycle.deaths(
        s.alive,
        mx.array(solvent),
        s.energy,
        mx.zeros((d.n,), dtype=mx.int32),
        mx.zeros((d.n,), dtype=mx.int32),
        s.uid,
        s.generation,
        s.clade,
        s.partner,
        s.rev,
        res0,
        fcell,
        cfg,
    )
    r = to_numpy(out.resource)
    corpse = to_numpy(out.corpse)
    want = np.zeros(d.field_shape, np.int64)
    for i in np.flatnonzero(dead):
        want[0, to_numpy(fcell)[i] % d.npix] += int(corpse[i])
    assert np.array_equal(r.astype(np.int64), want) and r[1:].sum() == 0  # every corpse is on layer 0
    assert corpse[dead].min() == int(to_numpy(s.energy)[0]) + cfg.energy.structure


def env_with(cfg: Config, lon: float, lat: float, radius: float, p: float) -> Env:
    e = init_state(cfg).env
    return e._replace(
        cat_center=mx.array([lon, lat], dtype=mx.float32),
        cat_radius=mx.array(radius, dtype=mx.float32),
        cat_kill_p=mx.array(p, dtype=mx.float32),
    )


@pytest.mark.usefixtures("dev")
def test_the_catastrophe_is_a_cap_of_the_sphere_at_every_height() -> None:
    cfg = cfg_of()
    rp = cfg.world.radius
    n = 1024
    rng = np.random.default_rng(4)
    pos, _, _ = parents(cfg, rng, n)
    alive = mx.ones((n,), dtype=mx.bool_)
    key = mx.random.key(5)
    lon, lat, radius = 0.7, -0.4, 14.0
    c = np.array([math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)])
    up = pos / np.linalg.norm(pos, axis=1, keepdims=True)
    ang = np.arccos(np.clip(up @ c, -1, 1))
    inside = ang * rp <= radius  # an arc length at the planet radius, at any height
    near = np.abs(ang * rp - radius) < 1e-3
    got = np.array(
        lifecycle.catastrophe_mask(mx.array(pos), alive, env_with(cfg, lon, lat, radius, 1.0), key, cfg)
    )
    assert np.array_equal(got[~near], inside[~near]) and 20 < inside.sum() < 400
    assert not np.array(
        lifecycle.catastrophe_mask(mx.array(pos), alive, env_with(cfg, lon, lat, radius, 0.0), key, cfg)
    ).any()
    assert not np.array(
        lifecycle.catastrophe_mask(mx.array(pos), alive, env_with(cfg, lon, lat, 0.0, 1.0), key, cfg)
    ).any()
    # the kill probability thins the cap by the same draws; dead rows are never killed
    half = np.array(
        lifecycle.catastrophe_mask(mx.array(pos), alive, env_with(cfg, lon, lat, radius, 0.5), key, cfg)
    )
    assert half.sum() < got.sum() and (half <= got).all()
    odd = mx.array(np.arange(n) % 2 == 1)  # the even rows are dead
    assert not np.array(
        lifecycle.catastrophe_mask(mx.array(pos), odd, env_with(cfg, lon, lat, radius, 1.0), key, cfg)
    )[::2].any()
    # a cap larger than half the planet is the whole sphere (the angle saturates at π)
    big = np.array(
        lifecycle.catastrophe_mask(mx.array(pos), alive, env_with(cfg, 0.0, 0.0, 10.0 * rp, 1.0), key, cfg)
    )
    assert big.all()


@pytest.mark.usefixtures("dev")
def test_the_cap_is_rotation_equivariant_and_the_poles_are_ordinary() -> None:
    cfg = cfg_of()
    n = 512
    rng = np.random.default_rng(6)
    pos, _, _ = parents(cfg, rng, n)
    alive, key = mx.ones((n,), dtype=mx.bool_), mx.random.key(1)
    r = random_rotation(2)
    lon, lat, radius = 0.3, 1.2, 12.0
    c = np.array([math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)])
    c2 = r @ c
    lon2, lat2 = math.atan2(c2[1], c2[0]), math.asin(np.clip(c2[2], -1, 1))
    a = np.array(
        lifecycle.catastrophe_mask(mx.array(pos), alive, env_with(cfg, lon, lat, radius, 1.0), key, cfg)
    )
    pos2 = (pos.astype(np.float64) @ r.T).astype(np.float32)
    b = np.array(
        lifecycle.catastrophe_mask(mx.array(pos2), alive, env_with(cfg, lon2, lat2, radius, 1.0), key, cfg)
    )
    assert (a != b).sum() <= 2  # only a point within rounding of the rim can flip
    pole = np.array(
        lifecycle.catastrophe_mask(
            mx.array(pos), alive, env_with(cfg, 0.0, math.pi / 2, radius, 1.0), key, cfg
        )
    )
    assert (
        pole.sum()
        == (pos[:, 2] / np.linalg.norm(pos, axis=1) >= math.cos(radius / cfg.world.radius) - 1e-6).sum()
    )
