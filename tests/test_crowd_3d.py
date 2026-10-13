"""3D crowding probes through the real 3D neighbour search and physics (§27.2, §27.3; docs/3D.md §12 Q15).

300 agents in the ``tiny`` geometry made 3D (L = 128, height 24, K = 8) run ``find_neighbors``, ``forces3``
and ``integrate3``: a column that sinks into a floor pile under gravity, and a ball that thrusts toward its
centre without gravity. Each starts below 16 agents per hash cell and runs at the default C = 16 and at
C = 64, where no cell overflows, so that the occupancy measured there is what the physics produces. The tests
are informational, for sizing C and K of ``64k3d``: they print the worst hash-cell occupancy against C and the
contacts per agent against K (also as junit properties), and assert only that the scene really piled or
converged and stayed finite and in range.
"""

from __future__ import annotations

import math
from typing import Any

import mlx.core as mx
import numpy as np
import pytest

from alife import layout, ops, physics, spatial
from alife.actions import Actions, Motion3
from alife.config import Config, apply_overrides, dims
from alife.layout import MSG, NCH, B
from helpers import as_3d

pytestmark = pytest.mark.slow

N_AGENTS = 300
CHUNK = 50


def _crowd3(tiny_cfg: Config, c: int, gravity: float) -> Config:
    """``tiny`` in 3D with just enough rows for the crowd: G_h = 21, G_z = 4, K = 8, C = ``c``."""
    cfg = apply_overrides(
        tiny_cfg,
        ["population.n_max=320", "population.max_births_per_step=16", "population.n_init=16",
         f"sensing.cell_capacity={c}"],
    )  # fmt: skip
    cfg = as_3d(cfg, physics={"gravity": gravity})
    d = dims(cfg)
    assert (d.c, d.k, d.g_hash, d.g_hash_z) == (c, 8, 21, 4)
    return cfg


def _start(cfg: Config, shape: str) -> tuple[mx.array, mx.array]:
    """300 agents in a vertical cylinder of radius 12 over the whole height (``pile``) or in a ball of radius
    15 around the centre of the volume, flattened to fit the height (``sphere``); the other rows are dead on
    the floor."""
    d, L, h = dims(cfg), cfg.world.size, cfg.world.height
    rs = np.random.default_rng(5)
    pos = np.zeros((d.n, 3), np.float32)
    pos[:, 2] = d.z_lo
    if shape == "pile":
        r = 12.0 * np.sqrt(rs.uniform(size=N_AGENTS))
        a = rs.uniform(0.0, 2.0 * math.pi, N_AGENTS)
        z = rs.uniform(d.z_lo, d.z_hi, N_AGENTS)
        pos[:N_AGENTS] = np.stack([L / 2 + r * np.cos(a), L / 2 + r * np.sin(a), z], axis=1)
    else:
        v = rs.normal(size=(N_AGENTS, 3))
        v *= (15.0 * rs.uniform(size=(N_AGENTS, 1)) ** (1 / 3)) / np.linalg.norm(v, axis=1, keepdims=True)
        v[:, 2] *= (h / 2 - 1.5) / 15.0  # flattened to fit between the floor and the surface
        pos[:N_AGENTS] = v + np.array([L / 2, L / 2, h / 2])
    return mx.array(pos), mx.array(np.arange(d.n) < N_AGENTS)


def _crowd_step(cfg: Config, thrust: float):
    d, L, h = dims(cfg), cfg.world.size, cfg.world.height
    n, two_r = d.n, 2.0 * cfg.physics.radius
    zf, zb = mx.zeros((n,), mx.float32), mx.zeros((n,), mx.bool_)
    none = mx.full((n,), -1, mx.int32)
    partner = mx.full((n, B), -1, mx.int32)
    centre = mx.array([L / 2, L / 2, h / 2], dtype=mx.float32)
    mo = Motion3(pitch=zf, buoy=zf)

    def fn(pos: mx.array, alive: mx.array) -> tuple[mx.array, ...]:
        nbr = spatial.find_neighbors(pos, alive, cfg)
        to = ops.min_image(centre[None, :] - pos, L)  # the identity on z (2·height ≤ L)
        heading = mx.arctan2(to[:, 1], to[:, 0])  # everyone aims at the centre
        pitch = mx.arctan2(to[:, 2], mx.sqrt(to[:, 0] ** 2 + to[:, 1] ** 2))
        pitch = mx.clip(pitch, -layout.PITCH_MAX, layout.PITCH_MAX)
        act = Actions(
            thrust=mx.where(alive, thrust, 0.0), turn=zf, msg=mx.zeros((n, MSG)), emit=mx.zeros((n, NCH)),
            harvest=zf, grip=zf, contract=zf, bond_want=zb, release=mx.zeros((n, B), mx.bool_),
            share=mx.zeros((n, B)), attack=zb, defend=zf, reproduce=zb, child_bond=zb, bond_k=none,
            bond_target=none, attack_k=none, attack_target=none,
        )  # fmt: skip
        f = physics.forces3(pos, heading, pitch, act, mo, zf, nbr, alive, partner, cfg)
        out = physics.integrate3(pos, heading, pitch, f, zf, zf, zf, physics.grounded(pos, cfg), alive, cfg)
        listed = (nbr.valid & (nbr.dist < two_r)).sum(axis=1)  # contacts in the K list
        return out.pos, nbr.overflow_cells, nbr.max_occupancy, listed.max(), (listed >= d.k).sum()

    return mx.compile(fn)


def _true_contacts(p: np.ndarray, cfg: Config) -> np.ndarray:
    """Contacts per agent (centre distance < 2r) by brute force in float64, not truncated at K."""
    d = p[:, None, :].astype(np.float64) - p[None, :, :]
    d[..., :2] -= cfg.world.size * np.round(d[..., :2] / cfg.world.size)
    dist = np.sqrt((d * d).sum(-1))
    np.fill_diagonal(dist, np.inf)
    return (dist < 2.0 * cfg.physics.radius).sum(1)


def _run(cfg: Config, shape: str, thrust: float, steps: int) -> dict[str, Any]:
    fn = _crowd_step(cfg, thrust)
    pos, alive = _start(cfg, shape)
    rows: list[tuple[int, ...]] = []
    true_max = 0
    for _ in range(steps // CHUNK):
        chunk = []
        for _ in range(CHUNK):
            pos, *stats = fn(pos, alive)
            chunk.append(stats)
        mx.eval(pos, chunk)
        rows += [tuple(int(x) for x in st) for st in chunk]
        true_max = max(true_max, int(_true_contacts(np.array(pos)[:N_AGENTS], cfg).max()))
    over, occ, listed, saturated = np.array(rows).T
    p = np.array(pos)[:N_AGENTS]
    gap = p[:, None, :].astype(np.float64) - p[None, :, :]
    gap[..., :2] -= cfg.world.size * np.round(gap[..., :2] / cfg.world.size)
    sep = np.sqrt((gap * gap).sum(-1))
    np.fill_diagonal(sep, np.inf)
    return {
        "pos": p,
        "start_occupancy": int(occ[0]),  # the first search sees the starting positions
        "max_occupancy": int(occ.max()),
        "overflow_steps": int((over > 0).sum()),
        "max_listed_contacts": int(listed.max()),
        "max_saturated_agents": int(saturated.max()),
        "max_true_contacts": true_max,
        "min_separation": float(sep.min()),
    }


def _report(name: str, cfg: Config, r: dict[str, Any], record_property: Any) -> None:
    d = dims(cfg)
    line = (
        f"{name}: max occupancy {r['max_occupancy']} (C = {d.c}; {r['start_occupancy']} at the start), "
        f"overflow on {r['overflow_steps']} steps; "
        f"contacts max {r['max_true_contacts']} true / {r['max_listed_contacts']} listed (K = {d.k}), "
        f"up to {r['max_saturated_agents']} agents with a K list full of contacts; "
        f"min centre distance {r['min_separation']:.3f} (2r = {2 * cfg.physics.radius})"
    )
    print(line)
    for k, v in r.items():
        if k != "pos":
            record_property(f"{name}.{k}", v)


@pytest.mark.parametrize("c", [16, 64], ids=["C16", "C64"])
def test_floor_pile_under_gravity(tiny_cfg: Config, c: int, record_property: Any) -> None:
    """A column sinks at g = 0.03 into a pile on the floor (no thrust)."""
    cfg = _crowd3(tiny_cfg, c, 0.03)
    d = dims(cfg)
    r = _run(cfg, "pile", 0.0, 900)  # 900 steps: a fall of 22 at 0.03 per step takes ~730
    _report(f"pile C={c}", cfg, r, record_property)
    assert r["start_occupancy"] <= 16
    p = r["pos"]
    assert np.isfinite(p).all() and np.all((p[:, :2] >= 0) & (p[:, :2] < cfg.world.size))
    assert np.all((p[:, 2] >= np.float32(d.z_lo)) & (p[:, 2] <= np.float32(d.z_hi)))
    assert np.median(p[:, 2]) < 8.0  # the column really came down
    assert (p[:, 2] == np.float32(d.z_lo)).sum() > 20  # a bottom layer rests on the floor


@pytest.mark.parametrize("c", [16, 64], ids=["C16", "C64"])
def test_converging_sphere(tiny_cfg: Config, c: int, record_property: Any) -> None:
    """A ball thrusts toward its centre at full strength, without gravity (the 3D twin of test_crowd)."""
    cfg = _crowd3(tiny_cfg, c, 0.0)
    d = dims(cfg)
    r = _run(cfg, "sphere", 1.0, 600)
    _report(f"sphere C={c}", cfg, r, record_property)
    assert r["start_occupancy"] <= 16
    p = r["pos"]
    assert np.isfinite(p).all() and np.all((p[:, 2] >= np.float32(d.z_lo)) & (p[:, 2] <= np.float32(d.z_hi)))
    centre = np.array([cfg.world.size / 2, cfg.world.size / 2, cfg.world.height / 2])
    assert np.median(np.linalg.norm(p - centre, axis=1)) < 8.0  # it really converged (from a radius of 15)
