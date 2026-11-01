"""Farming on the shell (DESIGN §41 on §46.5; sphere series S3): the soil lives on the cells of the shell
(``(FZ, Npix)``, the flat index ``layer·Npix + pixel``); an organism puts it into the cell it stands in,
reads it back from that cell, and the regrowth bonus lands in the same cell's resource.

The ledger, the soil identity, the controls and the cost of the box suites run on the sphere through
``tests/test_niche_farm.py`` and ``tests/test_niche_step.py`` (parameter ``"s"``); here the cell of an agent
is computed on the host by the formulas of the geometry (``geom.ang2pix``, ``geom.layer_of``), independently
of the device lookup.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np

import helpers
from alife import config, ext, fields, geom, invariants, niche
from alife.config import dims, load_config
from alife.state import init_state, to_numpy, total_energy
from alife.step import make_step
from niche_helpers import cfg_for, np_soil, run


def host_cell(cfg, pos: np.ndarray) -> np.ndarray:
    """Flat cell index ``layer·Npix + pixel`` of positions, by the host formulas."""
    w = cfg.world
    rho = np.linalg.norm(pos.astype(np.float64), axis=1)
    pix = geom.ang2pix(pos.astype(np.float64) / rho[:, None], w.nside)
    return geom.layer_of(rho, w.radius, w.height, w.layers) * geom.npix(w.nside) + pix


def spread_agents(cfg, k: int = 24):
    """A state whose first ``k`` rows are alive at known points of the shell, nobody moving or breeding."""
    d = dims(cfg)
    s = init_state(cfg)
    s = s._replace(W1=mx.zeros_like(s.W1), W2=mx.zeros_like(s.W2))
    rs = np.random.default_rng(4)
    z = rs.uniform(-1, 1, k)
    lam = rs.uniform(0, 2 * math.pi, k)
    u = np.stack([np.sqrt(1 - z * z) * np.cos(lam), np.sqrt(1 - z * z) * np.sin(lam), z], axis=1)
    pos = np.zeros((d.n, 3), np.float32)
    pos[:] = np.array([0.0, 0.0, -(cfg.world.radius + 6.0)], np.float32)  # the dead rows
    pos[:k] = (u * (cfg.world.radius + rs.uniform(1.5, 14.0, k))[:, None]).astype(np.float32)
    hdg = np.zeros((d.n, 3), np.float32)
    hdg[:, 0] = 1.0  # tangent at the dead rows' point (the south pole)
    v = rs.normal(size=(k, 3))
    v -= (v * u).sum(axis=1, keepdims=True) * u
    hdg[:k] = v / np.linalg.norm(v, axis=1, keepdims=True)
    alive = np.zeros(d.n, bool)
    alive[:k] = True
    al = mx.array(alive)
    energy = mx.where(al, s.energy, 0)  # a dead row holds nothing and no uid; TOTAL follows the live rows
    tot = total_energy(energy, al, s.resource, cfg)
    s = s._replace(
        pos=mx.array(pos),
        hdg=mx.array(hdg),
        alive=al,
        energy=energy,
        uid=mx.where(al, s.uid, -1),
        total=tot,
        total0=tot,
    )
    mx.eval(s)
    return s, pos[:k]


def test_a_farmer_puts_soil_into_the_cell_it_stands_in() -> None:
    cfg = cfg_for("s", "niche.farm_cost=0")
    s, pos = spread_agents(cfg)
    s = helpers.set_output_bias(s, cfg, {dims(cfg).lay.OUT_FARM_PUT: 4.0})  # zero brains, full farm_put
    assert not np_soil(s).any()
    s1, out = make_step(cfg, compile=False)(s)
    mx.eval(s1, out)
    soil = np_soil(s1).reshape(-1)
    cells = host_cell(cfg, pos)
    # zero brains with the full output bias put 7 or 8 units each: every occupied cell, and only those
    assert set(np.flatnonzero(soil).tolist()) == set(cells.tolist())
    assert soil.sum() > 0 and invariants.full_check(s1, cfg, s, out) == []


def test_the_soil_sensor_reads_the_cell_of_the_agent() -> None:
    cfg = cfg_for("s")
    s, pos = spread_agents(cfg)
    cells = host_cell(cfg, pos)
    soil = np.zeros(tuple(s.soil.shape), np.int32).reshape(-1)
    for k, c in enumerate(cells):
        soil[c] = 100 * (k + 1)
    s = s._replace(soil=mx.array(soil.reshape(s.soil.shape)))
    ctx = ext.ExtCtx(None, None, None, fields.cell_index(s.pos, cfg))
    (inp,) = niche.inputs(s, cfg, ctx)
    got = to_numpy(inp)[: len(cells), 0]
    want = np.array([soil[c] for c in cells]) / cfg.niche.farm_cap
    np.testing.assert_allclose(got, want, rtol=1e-6)


def test_the_device_cell_equals_the_host_cell_over_the_shell() -> None:
    cfg = cfg_for("s")
    rs = np.random.default_rng(9)
    n = 4000
    z = rs.uniform(-1, 1, n)
    lam = rs.uniform(0, 2 * math.pi, n)
    u = np.stack([np.sqrt(1 - z * z) * np.cos(lam), np.sqrt(1 - z * z) * np.sin(lam), z], axis=1)
    pos = (u * (cfg.world.radius + rs.uniform(0.2, 15.8, n))[:, None]).astype(np.float32)
    dev = to_numpy(fields.cell_index(mx.array(pos), cfg))
    host = host_cell(cfg, pos)
    assert (dev == host).mean() > 0.999  # float32 against float64 differs only on a boundary
    assert dev.min() >= 0 and dev.max() < dims(cfg).n_cells


def test_the_bonus_lands_in_the_cell_of_the_soil() -> None:
    """Soil in one cell raises the regrowth of that cell and no other (the diffusion is switched off to see
    the cell itself)."""
    cfg = cfg_for("s", "fields.resource_diffusion_q8=0")
    ab = cfg_for("s", "fields.resource_diffusion_q8=0", "niche.control=ablate")
    s = init_state(cfg)
    s = s._replace(W1=mx.zeros_like(s.W1), W2=mx.zeros_like(s.W2))
    cell = 3 * geom.npix(cfg.world.nside) + 100
    soil = np.zeros(tuple(s.soil.shape), np.int32).reshape(-1)
    soil[cell] = cfg.niche.farm_cap
    s = s._replace(soil=mx.array(soil.reshape(s.soil.shape)), resource=mx.zeros_like(s.resource))
    out = []
    for c in (cfg, ab):
        s1, _ = make_step(c, compile=False)(s)
        mx.eval(s1)
        out.append(to_numpy(s1.resource).astype(np.int64).reshape(-1))
    diff = out[0] - out[1]  # the bonus alone: same actions (zero), same harvest (none to speak of)
    assert diff[cell] > 0
    assert (np.flatnonzero(diff) == cell).all()
    assert diff[cell] <= cfg.niche.farm_bonus_frac * cfg.fields.regen_max + 1


def test_niche_validates_on_a_shell_and_counts_its_cells() -> None:
    cfg = load_config(preset="tiny3s", overlays=["niche3d"])
    assert cfg.niche.farm and config.field_cells(cfg) == dims(cfg).n_cells == 8 * 768
    assert tuple(init_state(cfg).soil.shape) == (8, 768)
    assert next(run(cfg_for("s"), init_state(cfg_for("s")), 1))[3]["SOIL_TOTAL"] == 0
