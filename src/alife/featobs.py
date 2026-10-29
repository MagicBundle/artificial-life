"""Snapshot metrics of the feature extensions (DESIGN §38.9, §40.6, §42.6, §41.14): the host-side observers.

Each feature registers a ``snap_dev(s, cfg, heavy)`` (the device arrays that join the one transfer of
``metrics.snapshot_metrics``, keyed ``<feature>.<name>``) and a ``snap_row(h, cfg)`` (the keys of the row,
all starting with the feature's ``metric_prefix``). Nothing here touches a trajectory: the functions read a
finished state, once per metrics row, and the numpy helpers are the references of the tests.

Conventions. A statistic that is not defined (nobody alive, a constant array, no bond) is ``None``, never
NaN, so a row is always valid JSON. ``alive`` is the ``h["alive"]`` mask of the snapshot.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

import mlx.core as mx
import numpy as np

from alife import arms as arms_mod
from alife import bonds, fields, structures
from alife.config import Config

NULL_PERMS: Final = 20  # permutations of the role-mix null (bonds, DESIGN §40.6)
BODY_MIN: Final = 4  # smallest body the role entropy is computed over
EDGE_EPS: Final = 0.02  # a trait within this of 0 or 1 sits at its boundary (arms)


def _q(name: str, x: np.ndarray) -> dict[str, float | None]:
    from alife.metrics import _quantiles

    return _quantiles(name, x)


def _mean(x: np.ndarray) -> float | None:
    return float(x.mean()) if x.size else None


# --- structures ---------------------------------------------------------------------------------------------


def structures_dev(s: Any, cfg: Config, heavy: bool) -> dict[str, mx.array]:
    """Per row ``here`` (m/cap) and ``cover`` at the row's cell, and the grid maximum."""
    sn = structures.sense(s.mat, fields.cell_index(s.pos, cfg), cfg)
    return {
        "structures.here": sn.here,
        "structures.cover": sn.cover if sn.cover is not None else sn.here,
        "structures.mat_max": s.mat.max(),
    }


def structures_row(h: Mapping[str, np.ndarray], cfg: Config) -> dict[str, Any]:
    alive = h["alive"].astype(bool)
    here, cover = h["structures.here"][alive], h["structures.cover"][alive]
    # with reach > 0 but no effect reading the cover column the step never builds it: no value, not ``here``
    has_cover = cfg.structures.reach == 0 or "cover" in fields.mat_columns(cfg)
    return {
        "build_here_mean": _mean(here),
        "build_in_frac": _mean((here > 0).astype(np.float64)),
        "build_cover_mean": _mean(cover) if has_cover else None,
        "build_mat_max": int(h["structures.mat_max"]),
    }


# --- arms ---------------------------------------------------------------------------------------------------


def circular(x: np.ndarray) -> tuple[float | None, float | None]:
    """``(R, mean angle in turns)`` of phases ``x`` in turns: ``R = |mean exp(2πi x)|`` in [0, 1] (1 = all
    aligned) and the mean direction in [0, 1)."""
    if not x.size:
        return None, None
    z = np.exp(2j * np.pi * x.astype(np.float64)).mean()
    return float(abs(z)), float((np.angle(z) / (2 * np.pi)) % 1.0)


def pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    """Pearson correlation, None when either side has no variance or fewer than two points."""
    if a.size < 2:
        return None
    a, b = a.astype(np.float64), b.astype(np.float64)
    sa, sb = a.std(), b.std()
    if sa == 0.0 or sb == 0.0:
        return None
    return float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb))


def arms_dev(s: Any, cfg: Config, heavy: bool) -> dict[str, mx.array]:
    return {"arms.traits": s.arms, "arms.n_bonds": (s.partner >= 0).sum(axis=1)}


def arms_row(h: Mapping[str, np.ndarray], cfg: Config) -> dict[str, Any]:
    alive = h["alive"].astype(bool)
    tr = h["arms.traits"][alive].astype(np.float64)
    out: dict[str, Any] = {}
    for name, col in (
        ("atk", arms_mod.ATK),
        ("arm", arms_mod.ARM),
        ("tox", arms_mod.TOX),
        ("sig", arms_mod.SIG),
    ):
        x = tr[:, col]
        out[f"arms_{name}_mean"] = _mean(x)
        out.update(_q(f"arms_{name}", x))
    for name, col in (("key", arms_mod.KEY), ("lock", arms_mod.LOCK)):
        r, ang = circular(tr[:, col])
        out[f"arms_{name}_R"], out[f"arms_{name}_mean"] = r, ang
    out["arms_arm_bonds_corr"] = pearson(tr[:, arms_mod.ARM], h["arms.n_bonds"][alive])
    for name, col in (("atk", arms_mod.ATK), ("arm", arms_mod.ARM)):
        x = tr[:, col]
        out[f"arms_{name}_edge_frac"] = _mean(((x < EDGE_EPS) | (x > 1.0 - EDGE_EPS)).astype(np.float64))
    return out


# --- bonds --------------------------------------------------------------------------------------------------

N_ROLES: Final = 3  # generalist, forager, guard (germ is deferred)


def bonds_dev(s: Any, cfg: Config, heavy: bool) -> dict[str, mx.array]:
    out = {
        "bonds.role": s.role,
        "bonds.partner": s.partner,
        "bonds.flow": s.bond_flow,
        "bonds.kin": bonds.slot_kin(s.tag, s.partner, cfg),
        "bonds.harvest": s.last_harvest,
    }
    return out  # the bodies of a heavy snapshot are ``labels``, which ``snapshot_metrics`` computes once


def gini(x: np.ndarray) -> float | None:
    """Gini coefficient of a non-negative sample (0 equality, 1 one holds everything); None when empty or all
    0."""
    x = np.sort(np.asarray(x, np.float64))
    if not x.size or x.sum() <= 0:
        return None
    n = x.size
    return float((2.0 * (np.arange(1, n + 1) * x).sum() / (n * x.sum())) - (n + 1) / n)


def edges(partner: np.ndarray, alive: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(i, b)`` of every undirected bond between alive rows listed once, from the lower row ``i`` (<
    partner)."""
    rows = np.arange(partner.shape[0])[:, None]
    drawn = (partner > rows) & alive[:, None] & alive[np.maximum(partner, 0)]
    ii, bb = np.nonzero(drawn)
    return ii, bb


def assortativity(ra: np.ndarray, rb: np.ndarray, k: int = N_ROLES) -> float | None:
    """Newman's discrete assortativity of the categories over undirected edges ``(ra[e], rb[e])``:
    ``(Σ e_ii − Σ a_i b_i) / (1 − Σ a_i b_i)`` with the symmetrised mixing matrix; None for no edge or when
    the
    denominator vanishes (every end in one category)."""
    if not ra.size:
        return None
    e = np.zeros((k, k))
    np.add.at(e, (ra, rb), 1.0)
    e = (e + e.T) / (2.0 * ra.size)
    a = e.sum(axis=1)
    denom = 1.0 - float((a * a).sum())
    if denom <= 1e-12:
        return None
    return float((np.trace(e) - (a * a).sum()) / denom)


def role_entropy(
    role: np.ndarray, labels: np.ndarray, alive: np.ndarray
) -> tuple[float | None, np.ndarray, np.ndarray]:
    """``(size-weighted mean Shannon entropy in bits of the role mix over bodies of size >= BODY_MIN, the
    member
    rows, their body ids)``; None when there is no such body."""
    rows = np.flatnonzero(alive)
    lab = labels[rows]
    _, inv, sizes = np.unique(lab, return_inverse=True, return_counts=True)
    big = sizes[inv] >= BODY_MIN
    rows = rows[big]
    if not rows.size:
        return None, rows, inv[big]
    _, body = np.unique(inv[big], return_inverse=True)  # contiguous ids 0..k-1 over the kept bodies
    return _body_entropy(role[rows], body), rows, body


def _body_entropy(role: np.ndarray, body: np.ndarray) -> float:
    k = int(body.max()) + 1
    cnt = np.zeros((k, N_ROLES))
    np.add.at(cnt, (body, role), 1.0)
    n = cnt.sum(axis=1)
    p = cnt / n[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -np.where(p > 0, p * np.log2(p), 0.0).sum(axis=1)
    return float((h * n).sum() / n.sum())


def role_diff_z(role: np.ndarray, body: np.ndarray, observed: float, seed: int) -> float | None:
    """``(null mean − observed) / null sd`` of the body role entropy, the null being ``NULL_PERMS``
    permutations
    of the roles among the members of the bodies (fixed ``seed``); None when the null has no spread."""
    rng = np.random.default_rng(seed)
    null = np.array([_body_entropy(rng.permutation(role), body) for _ in range(NULL_PERMS)])
    sd = null.std()
    if sd <= 1e-12:
        return None
    return float((null.mean() - observed) / sd)


def flow_matrix(partner: np.ndarray, role: np.ndarray, flow: np.ndarray, alive: np.ndarray) -> np.ndarray:
    """(3, 3) quanta that moved last step from a sender of role ``r`` to a receiver of role ``c``: every slot
    with
    a positive net flow ``f`` (received) counts ``f`` at ``[role of the partner, role of the row]``."""
    m = np.zeros((N_ROLES, N_ROLES), np.int64)
    ii, bb = np.nonzero((partner >= 0) & (flow > 0) & alive[:, None] & alive[np.maximum(partner, 0)])
    np.add.at(m, (role[partner[ii, bb]], role[ii]), flow[ii, bb].astype(np.int64))
    return m


def bonds_row(h: Mapping[str, np.ndarray], cfg: Config) -> dict[str, Any]:
    alive = h["alive"].astype(bool)
    role = np.clip(h["bonds.role"].astype(np.int64), 0, N_ROLES - 1)
    partner, flow = h["bonds.partner"], h["bonds.flow"].astype(np.int64)
    n = int(alive.sum())
    share = np.bincount(role[alive], minlength=N_ROLES) / n if n else np.full(N_ROLES, np.nan)
    out: dict[str, Any] = {
        "bond_role_frac_gen": float(share[0]) if n else None,
        "bond_role_frac_forager": float(share[1]) if n else None,
        "bond_role_frac_guard": float(share[2]) if n else None,
    }
    ii, bb = edges(partner, alive)
    jj = partner[ii, bb]
    ra, rb = role[ii], role[jj]
    out["bond_role_assort_r"] = assortativity(ra, rb)
    if ii.size:  # forager-guard edges against random mixing over the edge ends
        ends = np.bincount(np.concatenate([ra, rb]), minlength=N_ROLES) / (2.0 * ii.size)
        out["bond_fg_edge_frac"] = float(((ra == 1) & (rb == 2) | (ra == 2) & (rb == 1)).mean())
        out["bond_fg_edge_expect"] = float(2.0 * ends[1] * ends[2])
    else:
        out["bond_fg_edge_frac"] = out["bond_fg_edge_expect"] = None
    fm = flow_matrix(partner, role, flow, alive)
    out["bond_flow_by_role"] = fm.tolist()
    inflow = fm.sum()
    bonded = alive & (partner >= 0).any(axis=1)
    net = np.where(partner >= 0, flow, 0).sum(axis=1)
    out["bond_net_giver_frac"] = _mean((net[bonded] < 0).astype(np.float64))
    out["bond_flow_gini"] = gini(np.abs(net[bonded]))
    # within-role flow to the poorer harvester, as a share of all inflow (the free-riding index)
    pos_i, pos_b = np.nonzero((partner >= 0) & (flow > 0) & alive[:, None] & alive[np.maximum(partner, 0)])
    pj = partner[pos_i, pos_b]
    harvest = h["bonds.harvest"].astype(np.int64)
    same = role[pos_i] == role[pj]
    poorer = harvest[pos_i] < harvest[pj]
    cheat = flow[pos_i, pos_b][same & poorer].sum()
    out["bond_cheat_idx"] = float(cheat / inflow) if inflow > 0 else None
    nonkin = flow[pos_i, pos_b][h["bonds.kin"][pos_i, pos_b] < 0.5].sum()
    out["bond_nonkin_flow_share"] = float(nonkin / inflow) if inflow > 0 else None
    guard = alive & (role == 2)
    near = guard | (
        ((partner >= 0) & alive[np.maximum(partner, 0)] & guard[np.maximum(partner, 0)]).any(axis=1) & alive
    )
    out["bond_shield_cover"] = _mean(near[alive].astype(np.float64))
    if "labels" in h:  # heavy snapshots: the role mix of the bodies
        ent, rows, body = role_entropy(role, h["labels"], alive)
        out["bond_body_role_entropy"] = ent
        step = int(h["step"]) if "step" in h else 0
        out["bond_role_diff_z"] = None if ent is None else role_diff_z(role[rows], body, ent, seed=step)
    return out


# --- farming ------------------------------------------------------------------------------------------------


def niche_dev(s: Any, cfg: Config, heavy: bool) -> dict[str, mx.array]:
    """The soil's reductions (it holds no agent data: the per-cell level is the state)."""
    return {
        "niche.cells": (s.soil > 0).astype(mx.int32).sum(),
        "niche.total": s.soil.astype(mx.int64).sum(),
        "niche.max": s.soil.max(),
    }


def niche_row(h: Mapping[str, np.ndarray], cfg: Config) -> dict[str, Any]:
    """``farm_cells`` (cells with soil), ``farm_soil_mean`` (mean level over all cells, in units) and
    ``farm_soil_max`` (the largest level); the interval keys of the soil are the counters' (``metrics``)."""
    n_cells = h["resource"].size
    return {
        "farm_cells": int(h["niche.cells"]),
        "farm_soil_mean": float(h["niche.total"]) / n_cells,
        "farm_soil_max": int(h["niche.max"]),
    }


# --- the registry rows --------------------------------------------------------------------------------------

__all__ = [
    "arms_dev",
    "arms_row",
    "assortativity",
    "bonds_dev",
    "bonds_row",
    "circular",
    "gini",
    "niche_dev",
    "niche_row",
    "pearson",
    "structures_dev",
    "structures_row",
]
