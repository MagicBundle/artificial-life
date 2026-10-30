"""Bonds on the shell (DESIGN §40 on §46.3; sphere series S3): the springs, the break test and the child bond
on chord offsets (``ops.min_image`` is the identity on a pair, ``world.size > 4(Rp + H)``), at the places a
torus wrap or a planar frame would break: the poles, the antimeridian, the ring seams.

The graph logic of the bonds (handshake, conductive share, roles) has no geometry; the box suites run on the
sphere through ``tests/test_bonds_step.py`` and ``tests/test_roles.py`` (parameter ``"s"``).
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from alife import bonds, physics
from alife.config import Config, load_config
from alife.layout import B
from alife.state import to_numpy
from test_sphere_physics import actions, motion, no_neighbours, tangent_headings


def cfg_of(*extra: str) -> Config:
    return load_config(preset="tiny3s", overrides=["brain.weight_dtype=float32", *extra])


def graph(n: int, edges: list[tuple[int, int]]):
    partner = np.full((n, B), -1, np.int32)
    rev = np.full((n, B), -1, np.int32)
    for i, j in edges:
        si, sj = int(np.argmax(partner[i] < 0)), int(np.argmax(partner[j] < 0))
        partner[i, si], rev[i, si], partner[j, sj], rev[j, sj] = j, sj, i, si
    return partner, rev


def at(cfg: Config, lat_deg: float, lon_deg: float, height: float) -> np.ndarray:
    """A point of the shell by latitude, longitude (degrees) and height above the planet radius."""
    la, lo = math.radians(lat_deg), math.radians(lon_deg)
    r = cfg.world.radius + height
    return np.array([r * math.cos(la) * math.cos(lo), r * math.cos(la) * math.sin(lo), r * math.sin(la)])


def offset_pair(cfg: Config, lat: float, lon: float, h: float, d: float, along: str) -> tuple:
    """Two points a chord ``d`` apart, one at ``(lat, lon, h)``, the other displaced along the local ``east``,
    ``north`` or ``up`` by ``d`` (then put back on the shell by the chord, not the arc: the pair is exact)."""
    p = at(cfg, lat, lon, h)
    up = p / np.linalg.norm(p)
    east = np.cross([0.0, 0.0, 1.0], up)
    east = east / np.linalg.norm(east) if np.linalg.norm(east) > 1e-9 else np.array([0.0, 1.0, 0.0])
    north = np.cross(up, east)
    q = p + d * {"east": east, "north": north, "up": up}[along]
    return p, q


PLACES = [
    ("mid", 20.0, 40.0),
    ("north pole", 89.99, 10.0),
    ("south pole", -89.99, 200.0),
    ("antimeridian", 5.0, 179.999),
    ("lon 0", -30.0, 0.0),
    ("ring seam", 41.8, 45.0),  # the HEALPix equatorial/polar boundary latitude
]


@pytest.mark.parametrize(("name", "lat", "lon"), PLACES)
@pytest.mark.parametrize("along", ["east", "north", "up"])
def test_a_spring_on_a_chord_is_the_box_spring_wherever_the_pair_sits(name, lat, lon, along) -> None:
    """The spring force of a pair at a chord ``d`` is ``k_s·(d − L0)`` along the chord, antisymmetric to the
    last bit, whatever the longitude, the latitude or the direction; the torus wrap that would break at the
    antimeridian is not in the pair code."""
    cfg = cfg_of()
    bc = cfg.bonds
    for d in (0.9 * bc.rest_length, bc.rest_length, 1.4 * bc.rest_length, 0.95 * bc.break_length):
        p, q = offset_pair(cfg, lat, lon, 6.0, d, along)
        pos = np.zeros((6, 3), np.float32)
        pos[0], pos[1] = p, q
        pos[2:] = at(cfg, -60.0, 100.0, 3.0)  # bystanders far from the pair
        partner, _ = graph(6, [(0, 1)])
        f = to_numpy(bonds.spring_forces(mx.array(pos), mx.array(partner), mx.zeros(6), cfg)).astype(
            np.float64
        )
        chord = pos[1].astype(np.float64) - pos[0].astype(np.float64)
        dist = np.linalg.norm(chord)
        want = bc.k_spring * (dist - bc.rest_length) * chord / dist
        np.testing.assert_allclose(f[0], want, rtol=2e-3, atol=2e-6)
        assert np.array_equal(f[0], -f[1]) and not f[2:].any()


def test_the_bond_geometry_of_a_pair_is_its_chord_and_never_a_torus_image() -> None:
    cfg = cfg_of()
    rs = np.random.default_rng(1)
    n = 24
    z = rs.uniform(-1, 1, n)
    lam = rs.uniform(0, 2 * math.pi, n)
    u = np.stack([np.sqrt(1 - z * z) * np.cos(lam), np.sqrt(1 - z * z) * np.sin(lam), z], axis=1)
    pos = (u * (cfg.world.radius + rs.uniform(1.5, 14.0, n))[:, None]).astype(np.float32)
    far = [(i, (i + 7) % n) for i in range(0, n, 2)]  # partners on the other side of the planet
    partner, _ = graph(n, far)
    g = bonds.bond_geometry(mx.array(pos), mx.array(partner), cfg)
    length = to_numpy(g.length).astype(np.float64)
    ii, bb = np.nonzero(partner >= 0)
    want = np.linalg.norm(pos[partner[ii, bb]].astype(np.float64) - pos[ii], axis=1)
    np.testing.assert_allclose(length[ii, bb], want, rtol=1e-5)
    assert want.max() > cfg.world.radius  # genuinely far: a wrap at size/2 would have changed the offset


def test_bonded_forces_differ_from_unbonded_by_exactly_the_spring() -> None:
    """In ``forces_s`` a bond adds the spring and nothing else (gravity, repulsion, thrust unchanged)."""
    cfg = cfg_of()
    rs = np.random.default_rng(3)
    n = 48
    p, q = offset_pair(cfg, 30.0, 70.0, 7.0, 3.1, "east")  # stretched past the rest length 2.0
    pos = np.zeros((n, 3), np.float32)
    pos[0], pos[1] = p, q
    for k in range(2, n):  # the others spread over the shell, away from the pair
        pos[k] = at(cfg, -10.0 + 0.7 * k, -100.0 + 5.0 * k, 3.0 + (k % 9))
    hdg = tangent_headings(rs, pos)
    act, mo = actions(rs), motion(rs)
    cap, sv = mx.full((n,), 0.5), mx.ones((n,), dtype=mx.bool_)
    nb = no_neighbours(n)
    none = mx.full((n, B), -1, dtype=mx.int32)
    partner, _ = graph(n, [(0, 1)])
    pitch = mx.zeros((n,))
    f0 = to_numpy(physics.forces_s(mx.array(pos), mx.array(hdg), pitch, act, mo, cap, nb, sv, none, cfg))
    f1 = to_numpy(
        physics.forces_s(mx.array(pos), mx.array(hdg), pitch, act, mo, cap, nb, sv, mx.array(partner), cfg)
    )
    spring = to_numpy(bonds.spring_forces(mx.array(pos), mx.array(partner), act.contract, cfg))
    assert np.abs(spring[0]).max() > 0
    np.testing.assert_allclose(f1 - f0, spring, rtol=1e-4, atol=1e-6)
    assert np.array_equal(f1[2:], f0[2:])  # nobody else notices the bond


def test_a_break_flag_is_a_chord_longer_than_break_length_on_the_far_side_of_the_planet() -> None:
    """The step breaks a bond whose chord exceeds ``bonds.break_length``: the geometry that feeds the test is
    the chord, so a pair at the same arc and a bigger chord breaks and a closer one does not."""
    cfg = cfg_of()
    bl = cfg.bonds.break_length
    n = 6
    pos = np.zeros((n, 3), np.float32)
    pos[0], pos[1] = offset_pair(cfg, 10.0, 170.0, 6.0, 0.97 * bl, "north")
    pos[2], pos[3] = offset_pair(cfg, -20.0, -170.0, 6.0, 1.03 * bl, "north")
    pos[4:] = at(cfg, 60.0, 0.0, 4.0)
    partner, _ = graph(n, [(0, 1), (2, 3)])
    g = bonds.bond_geometry(mx.array(pos), mx.array(partner), cfg)
    too_long = to_numpy(g.length > bl) & to_numpy(g.valid)
    assert too_long[2].any() and not too_long[0].any() and too_long.sum() == 2
