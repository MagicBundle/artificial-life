"""Converging crowd under the real neighbour search and physics (§4 failure mode 1, §9, F18). Owned by WP4.

300 agents thrust at full strength toward one point for 600 steps through ``spatial.find_neighbors``
(C = 16, K = 8), ``physics.forces`` and ``physics.integrate``. Contact-attenuated thrust must keep every hash
cell within capacity; with attenuation effectively off the same scene must overflow (the test is not vacuous).
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from alife import ops, physics, spatial
from alife.actions import Actions
from alife.config import Config, apply_overrides, dims
from alife.layout import MSG, NCH, B

N_AGENTS = 300
STEPS = 600
CHUNK = 50


@pytest.fixture(scope="module")
def crowd_cfg(tiny_cfg: Config) -> Config:
    """``tiny`` geometry (L = 128, G_h = 21, C = 16, K = 8) with just enough rows for the crowd."""
    cfg = apply_overrides(
        tiny_cfg, ["population.n_max=320", "population.max_births_per_step=16", "population.n_init=16"]
    )
    d = dims(cfg)
    assert (d.c, d.k, d.g_hash) == (16, 8, 21)
    return cfg


def _start(cfg: Config) -> tuple[mx.array, mx.array]:
    """300 agents uniformly in a disc of radius 25 around the world centre; the remaining rows are dead."""
    n, L = dims(cfg).n, cfg.world.size
    rs = np.random.default_rng(5)
    r = 25.0 * np.sqrt(rs.uniform(size=N_AGENTS))
    a = rs.uniform(0.0, 2.0 * math.pi, N_AGENTS)
    pos = np.zeros((n, 2), np.float32)
    pos[:N_AGENTS] = np.stack([L / 2 + r * np.cos(a), L / 2 + r * np.sin(a)], axis=1)
    return mx.array(pos), mx.array(np.arange(n) < N_AGENTS)


def _crowd_step(cfg: Config, grip: float):
    n, L = dims(cfg).n, cfg.world.size
    zf, zb = mx.zeros((n,), mx.float32), mx.zeros((n,), mx.bool_)
    none = mx.full((n,), -1, mx.int32)
    partner = mx.full((n, B), -1, mx.int32)
    centre = mx.array([L / 2, L / 2], dtype=mx.float32)

    def fn(pos: mx.array, alive: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        nbr = spatial.find_neighbors(pos, alive, cfg)
        to = ops.min_image(centre[None, :] - pos, L)
        heading = mx.arctan2(to[:, 1], to[:, 0])  # everyone aims at the centre
        act = Actions(
            thrust=mx.where(alive, 1.0, 0.0), turn=zf, msg=mx.zeros((n, MSG)), emit=mx.zeros((n, NCH)),
            harvest=zf, grip=mx.where(alive, grip, 0.0), contract=zf, bond_want=zb,
            release=mx.zeros((n, B), mx.bool_), share=mx.zeros((n, B)), attack=zb, defend=zf, reproduce=zb,
            child_bond=zb, bond_k=none, bond_target=none, attack_k=none, attack_target=none,
        )  # fmt: skip
        f = physics.forces(pos, heading, act, nbr, alive, partner, cfg)
        pos, _, _, _ = physics.integrate(pos, heading, f, act.grip, act.turn, alive, cfg)
        return pos, nbr.overflow_cells, nbr.max_occupancy

    return mx.compile(fn)


def _run(
    cfg: Config, grip: float, *, stop_on_overflow: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(final pos of the crowd, overflow_cells per step, max_occupancy per step)``."""
    fn = _crowd_step(cfg, grip)
    pos, alive = _start(cfg)
    over, occ = [], []
    for _ in range(STEPS // CHUNK):
        o_chunk, m_chunk = [], []
        for _ in range(CHUNK):
            pos, o, m = fn(pos, alive)
            o_chunk.append(o)
            m_chunk.append(m)
        mx.eval(pos, *o_chunk, *m_chunk)
        over += [int(x) for x in o_chunk]
        occ += [int(x) for x in m_chunk]
        if stop_on_overflow and max(over) > 0:
            break
    return np.array(pos)[:N_AGENTS], np.array(over), np.array(occ)


def _median_nn_spacing(p: np.ndarray, L: float) -> float:
    d = p[:, None, :].astype(np.float64) - p[None, :, :]
    d -= L * np.round(d / L)
    dist = np.sqrt((d * d).sum(-1))
    np.fill_diagonal(dist, np.inf)
    return float(np.median(dist.min(axis=1)))


@pytest.mark.parametrize("grip", [0.0, 1.0], ids=["no-grip", "grip"])
def test_converging_crowd_never_overflows(crowd_cfg: Config, grip: float) -> None:
    pos, over, occ = _run(crowd_cfg, grip)
    C = dims(crowd_cfg).c
    assert over.shape == (STEPS,)
    assert (over == 0).all(), f"cell overflow at steps {np.flatnonzero(over)[:5]}"
    assert occ.max() <= C, f"max occupancy {occ.max()} > C = {C}"
    assert np.isfinite(pos).all() and (pos >= 0).all() and (pos < crowd_cfg.world.size).all()
    L = crowd_cfg.world.size
    centre_dist = np.linalg.norm(pos - L / 2, axis=1)
    assert np.median(centre_dist) < 15.0  # the crowd really converged (it started spread over r ≤ 25)
    assert _median_nn_spacing(pos, L) >= 1.6


def test_crowd_overflows_without_attenuation(crowd_cfg: Config) -> None:
    """With ``thrust_contact_limit`` huge (attenuation off), the same crowd collapses past C (F18)."""
    cfg = apply_overrides(crowd_cfg, ["physics.thrust_contact_limit=1e9"])
    _, over, occ = _run(cfg, 0.0, stop_on_overflow=True)
    assert over.max() > 0
    assert occ.max() > dims(cfg).c
