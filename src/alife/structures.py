"""Built structures: material the agents deposit, and what it does (DESIGN §38).

``State.mat`` is an int32 grid on the food lattice (``dims.field_shape``). An agent with the brain output
``build`` deposits up to ``build_max`` units into the cell it occupies at ``build_cost`` quanta per unit
(booked in ``HEAT_ACTION`` through ``extra_due``); the material decays by an exact integer rule
(``fields.update_mat``) and holds no energy, so the 16-term energy ledger is unchanged and the material has
its own exact identity ``Σmat' = Σmat + BUILD_UNITS - MAT_WASTED - MAT_DECAYED``.

Three effects read the START-of-step material at the agent's cell (``sense``, one gather of a per-cell
table):

* shelter: the wind force and the air speed that feeds the storm hazard and the bond break are multiplied by
  ``1 - shelter_gain·cover`` (``weather.shelter_wind``; inert without weather);
* barrier: the drag of the overdamped integrator is multiplied by ``1 + barrier_gain·here`` (R1), eased for
  an agent that is being pushed out of the material (R3, ``physics.barrier_drag``);
* shield: the damage demand of a predation attack on the victim is multiplied by ``1 - shield_gain·cover``
  (``energy.combat(shelter=)``, inside the protection product and under ``combat.protect_floor``).

Anti-trapping (§38.4b): R1 bounds the slowdown by ``1 + barrier_gain <= 4``, R2 (``fields.update_mat``)
keeps ``k`` open neighbours around every cell by construction, R3 eases the exit. ``TRAPPED`` (agents in a
cell with no open neighbour) and ``APERTURE_VIOLATIONS`` (cells with fewer than ``k``) are counters; the
second is also a device error (``Diag.FIELD_RANGE``) and a host check.

Pure MLX, no RNG, no host sync; the only scatter is one ``ops.scatter_add_i32``. The numpy functions at the
end are the host checks and the references of the tests.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import mlx.core as mx
import numpy as np

from alife import actions, fields, ops, physics, weather
from alife.config import Config, dims

COUNTERS: tuple[str, ...] = (
    "BUILD_UNITS",
    "MAT_WASTED",
    "MAT_DECAYED",
    "MAT_TOTAL",
    "MAT_CELLS",
    "SHIELDED_ATTACKS",
    "TRAPPED",
    "APERTURE_VIOLATIONS",
    "SLOWED",
    "MAT_CLOSED",
)

_I32 = mx.int32


class Sense(NamedTuple):
    """The material at each agent's start-of-step cell (one gather of ``fields.mat_table``)."""

    here: mx.array  # (N,) f32 in [0, 1]: m/cap
    cover: (
        mx.array | None
    )  # (N,) f32 in [0, 1]: max(m, reach·max over face neighbours)/cap; here if reach = 0
    grad: mx.array | None  # (N, nd) f32: central difference of m/cap (world frame); only with R3


def sense(mat: mx.array, fcell: mx.array, cfg: Config) -> Sense:
    """Gather the per-cell table of ``mat`` at ``fcell`` (the start-of-step cell of every row)."""
    cols = fields.mat_columns(cfg)
    row = fields.mat_table(mat, cfg)[fcell]  # (N, C): the one gather
    here = row[:, 0]
    i = 1
    cover = here if cfg.structures.reach == 0 else None
    if "cover" in cols:
        cover, i = row[:, i], i + 1
    grad = row[:, i : i + dims(cfg).nd] if "grad" in cols else None
    return Sense(here, cover, grad)


# --- the effects ---------------------------------------------------------------------------------------


def shelter_k(sn: Sense, cfg: Config) -> mx.array:
    """The wind factor ``1 - shelter_gain·cover`` of every row, (N,) f32 in ``[1 - shelter_gain, 1]``."""
    assert sn.cover is not None, "shelter needs the cover column"  # static
    return 1.0 - cfg.structures.shelter_gain * sn.cover


def apply_shelter(wind: weather.AgentForce, sn: Sense, cfg: Config) -> weather.AgentForce:
    """S3w: the wind the material lets through; ``wind`` itself when ``shelter_gain`` is 0 (static)."""
    if cfg.structures.shelter_gain <= 0:
        return wind
    return weather.shelter_wind(wind, shelter_k(sn, cfg), cfg)


def combat_kw(sn: Sense, cfg: Config) -> dict[str, mx.array]:
    """The ``energy.combat`` keyword of the predator shield: the victim's protection fraction
    ``shield_gain·cover`` (empty when ``shield_gain`` is 0)."""
    if cfg.structures.shield_gain <= 0:
        return {}
    assert sn.cover is not None, "shield needs the cover column"  # static
    return {"shelter": cfg.structures.shield_gain * sn.cover}


def drag_mult(force: mx.array, sn: Sense, cfg: Config) -> mx.array | None:
    """The barrier of every row for ``physics.integrate(3)``; None when ``barrier_gain`` is 0 (static)."""
    if cfg.structures.barrier_gain <= 0:
        return None
    return physics.barrier_drag(force, sn.here, sn.grad, cfg)


def shelter_k_at(mat: mx.array, pos: mx.array, cfg: Config) -> mx.array:
    """The wind factor of the material ``mat`` at the positions ``pos`` (observers: ``agent_drift``)."""
    if cfg.structures.shelter_gain <= 0:  # static
        return mx.ones(pos.shape[0], dtype=mx.float32)
    return shelter_k(sense(mat, fields.cell_index(pos, cfg), cfg), cfg)


# --- the hooks of the registry row (ext/structures.py) ---------------------------------------------------


def inputs(s: Any, cfg: Config, ctx: Any) -> tuple[mx.array]:
    """The brain input ``material``: ``m/cap`` at the agent's cell, (N, 1) f32."""
    here = s.mat.reshape(-1)[ctx.fcell].astype(mx.float32) * (1.0 / cfg.structures.cap)
    return (here[:, None],)


def decode(s: Any, y: mx.array, ctx: Any, cfg: Config) -> actions.Build:
    """S7: the build output."""
    return actions.decode_build(y, s.alive, cfg)


def due(s: Any, act: Any, y: mx.array, cfg: Config) -> mx.array | None:
    """The action due of the build: ``units·build_cost`` quanta, an int32 product (None when free)."""
    if cfg.structures.build_cost == 0:
        return None
    return actions.decode_build(y, s.alive, cfg).units * cfg.structures.build_cost


def mask(dec: actions.Build, solvent: mx.array, cfg: Config) -> actions.Build:
    """Solvency: an insolvent row deposits nothing (its cost was due and is paid as far as it holds)."""
    return actions.Build(mx.where(solvent, dec.units, 0))


class StructFlows(NamedTuple):
    """The step's structures intermediates for the counters (``_Flows.ext.structures``)."""

    units: mx.array  # (N,) int32: deposited units (masked by solvency)
    built: mx.array  # () int32: Σ units
    wasted: mx.array  # () int32
    decayed: mx.array  # () int32
    total: mx.array  # () int32: Σ mat after the step
    cells: mx.array  # () int32: cells with material after the step
    closed: mx.array  # () int32: cells at or above the closed threshold after the step
    violations: mx.array  # () int32: cells with nopen < k in the START-of-step material
    trapped: mx.array  # () int32: alive rows (start of step) in a cell with no open neighbour
    slowed: mx.array  # () int32: alive rows (start of step) with drag_mult > 2


def update(
    mat: mx.array,
    dec: actions.Build,
    fcell: mx.array,
    alive0: mx.array,
    dmult: mx.array | None,
    step: mx.array,
    cfg: Config,
) -> tuple[mx.array, StructFlows]:
    """S17: deposit the masked units at the start-of-step cells, cap, gate (R2), decay; the new grid and the
    counters' inputs. ``dmult`` is the barrier used by the physics (None when ``barrier_gain`` is 0)."""
    d = dims(cfg)
    dep = ops.scatter_add_i32(d.n_cells, fcell, dec.units).reshape(d.field_shape)
    upd = fields.update_mat(mat, dep, step, cfg)

    def count(x: mx.array) -> mx.array:
        return x.astype(_I32).sum()

    k = fields.mat_k(cfg)
    zero = mx.array(0, dtype=_I32)
    return upd.mat, StructFlows(
        units=dec.units,
        built=dec.units.sum(),
        wasted=upd.wasted,
        decayed=upd.decayed,
        total=upd.mat.sum(),
        cells=count(upd.mat > 0),
        closed=count(upd.mat >= fields.mat_thr(cfg)),
        violations=count(upd.nopen < k) if k > 0 else zero,
        trapped=count(alive0 & (upd.nopen.reshape(-1)[fcell] < 1)),
        slowed=count(alive0 & (dmult > 2.0)) if dmult is not None else zero,
    )


def count(s_new: Any, alive0: mx.array, flows: Any, cfg: Config) -> dict[str, mx.array]:
    """The counter block (names in ``COUNTERS``)."""
    f = flows.ext.structures
    shielded = flows.cb.shielded
    return {
        "BUILD_UNITS": f.built,
        "MAT_WASTED": f.wasted,
        "MAT_DECAYED": f.decayed,
        "MAT_TOTAL": f.total,
        "MAT_CELLS": f.cells,
        "SHIELDED_ATTACKS": shielded.astype(_I32).sum() if shielded is not None else mx.array(0, dtype=_I32),
        "TRAPPED": f.trapped,
        "APERTURE_VIOLATIONS": f.violations,
        "SLOWED": f.slowed,
        "MAT_CLOSED": f.closed,
    }


# --- host checks and numpy references ----------------------------------------------------------------------


def nopen_np(mat: np.ndarray, cfg: Config) -> np.ndarray:
    """Numpy ``nopen`` by explicit neighbour offsets (x and y periodic, z clamped, missing cells not open):
    the reference of the tests and the host check. Independent of the separable box sums of the engine."""
    d = dims(cfg)
    thr = fields.mat_thr(cfg)
    op = (np.asarray(mat) < thr).astype(np.int64)
    n = np.zeros(op.shape, np.int64)
    axes = range(op.ndim)
    for off in np.ndindex(*(3,) * op.ndim):
        o = [x - 1 for x in off]
        if not any(o):
            continue
        v, ex = op, np.ones(op.shape, bool)
        for ax, dlt in zip(axes, o, strict=True):
            if dlt == 0:
                continue
            v, ex = np.roll(v, -dlt, axis=ax), np.roll(ex, -dlt, axis=ax)
            if d.d3 and ax == 0:  # z is not periodic: the wrapped layer does not exist
                sl = [slice(None)] * op.ndim
                sl[0] = slice(op.shape[0] - 1, None) if dlt == 1 else slice(0, 1)
                ex[tuple(sl)] = False
        n += v * ex
    return n


def check(arrays: Any, cfg: Config) -> list[str]:
    """Host check of the numpy ``mat``: in ``[0, cap]`` and every cell keeps ``k`` open neighbours."""
    m = np.asarray(arrays["mat"])
    problems = []
    if (m < 0).any() or (m > cfg.structures.cap).any():
        problems.append(f"mat: values outside [0, {cfg.structures.cap}] (min {m.min()}, max {m.max()})")
    k = fields.mat_k(cfg)
    if k > 0:
        bad = int((nopen_np(m, cfg) < k).sum())
        if bad:
            problems.append(f"mat: {bad} cell(s) with fewer than {k} open neighbours (R2 aperture)")
    return problems
