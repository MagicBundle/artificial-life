"""Emergence metrics: bodies, snapshots, interval rows and carrying capacity (§19, §10.10). Owned by WP5.

Observers only: nothing here touches ``state.key`` or feeds back into the simulation. Random sampling uses
``rng.observer_key(seed, step)`` (§14.1).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Final

import mlx.core as mx
import numpy as np

from alife import fields, layout, ops, rng
from alife.config import Config, dims, is_3d
from alife.state import (
    ENV_FIELDS,
    CounterIdx,
    Env,
    LedgerIdx,
    State,
    ledger_names,
    n_counters,
    sinks,
    to_numpy,
)

# Upper bounds of the body-size histogram bins 1, 2, 3–4, 5–8, 9–16, 17–32, 33–64, 65+ (§19.2).
BODY_BIN_EDGES: Final = (1, 2, 4, 8, 16, 32, 64)
BODY_BIN_NAMES: Final = ("1", "2", "3-4", "5-8", "9-16", "17-32", "33-64", "65+")


# --- bodies ---------------------------------------------------------------------------------------------


def _label_round(lab: mx.array, nbr: mx.array, valid: mx.array, n: int) -> mx.array:
    """One round: min-label propagation over bonds, root hooking, then one pointer jump.

    Hooking also lowers the label of each row's current root ``lab[i]`` to ``i``'s smallest neighbour label
    (an int32 scatter-max of ``n − 1 − label``, i.e. a scatter-min). Without it a long chain laid over random
    rows needs O(length) rounds; with it a shuffled 2 000-chain converges in about ten.
    """
    m = mx.where(valid, lab[nbr], n).min(axis=1)
    has = m < n
    best = ops.scatter_max_i32(n, lab, mx.where(has, n - 1 - m, 0), has)
    lab = mx.minimum(mx.where(best >= 0, mx.minimum(lab, n - 1 - best), lab), m)
    return lab[lab]


def components(partner: mx.array, alive: mx.array, iters: int) -> tuple[mx.array, mx.array]:
    """Bond-graph connected components by min-label propagation: ``(labels, converged)`` (§19.2).

    ``labels`` (N,) int32 holds, for every row, a row index in the same component; at convergence it is the
    smallest row index of the component. Bonds touching a dead row are ignored, so dead rows are singletons.
    ``converged`` () bool is True when one extra round changes nothing.
    """
    n = partner.shape[0]
    nbr = mx.maximum(partner, 0)
    valid = (partner >= 0) & alive[:, None] & alive[nbr]
    lab = mx.arange(n, dtype=mx.int32)
    for _ in range(iters):
        lab = _label_round(lab, nbr, valid, n)
    return lab, mx.all(_label_round(lab, nbr, valid, n) == lab)


def _body_stats(sizes: np.ndarray, n_alive: int, converged: bool) -> dict[str, Any]:
    """Body metrics from per-label component sizes (zeros for non-root labels)."""
    sizes = sizes[sizes > 0].astype(np.int64)
    bodies = sizes[sizes >= 2]
    bins = np.digitize(sizes, BODY_BIN_EDGES, right=True)
    hist = np.bincount(bins, minlength=len(BODY_BIN_NAMES))
    return {
        "bodies_n": int(bodies.size),
        "bodies_largest": int(sizes.max(initial=0)),
        "bodies_mean_size": float((bodies * bodies).sum() / bodies.sum()) if bodies.size else None,
        "bodies_hist": dict(zip(BODY_BIN_NAMES, (int(c) for c in hist), strict=True)),
        "frac_in_bodies_ge4": float(sizes[sizes >= 4].sum() / n_alive) if n_alive else None,
        "bodies_converged": converged,
    }


# --- weight diversity -----------------------------------------------------------------------------------


def _diversity_arrays(s: State, step: int, cfg: Config) -> tuple[mx.array, mx.array]:
    """Device ``(Σ pairwise L2 distance, n_pairs)`` over the observer sample of flushed alive rows (§19.2).

    Rows born during the last step still hold stale weights until the next S0 flush, so they are excluded
    (for ``step ≥ 1``; at step 0 nothing is pending, §12.6).
    """
    n, k = s.alive.shape[0], min(cfg.run.diversity_sample, s.alive.shape[0])
    eligible = s.alive & ((s.birth_step < step - 1) | (step == 0))
    perm = mx.random.permutation(n, key=rng.observer_key(cfg.run.seed, step)).astype(mx.int32)
    pick, valid = ops.rank_select(eligible[perm], k)
    rows = perm[pick]
    x = mx.concatenate(
        [s.W1[rows].astype(mx.float32).reshape(k, -1), s.W2[rows].astype(mx.float32).reshape(k, -1)], axis=1
    )
    # Centring removes the component related genomes share, which would otherwise cancel in the Gram trick.
    # The Gram matmul runs on the CPU stream: float32 matmul on some Apple GPUs is reduced precision (≈ 8e-4
    # relative on M5). Squared norms are exact elementwise sums. This is an observer-side metric, so the
    # extra stream costs nothing in the step.
    w = valid.astype(mx.float32)[:, None]
    x = x - (x * w).sum(axis=0) / mx.maximum(w.sum(), 1.0)
    g = mx.matmul(x, x.T, stream=mx.cpu)
    sq = (x * x).sum(axis=1)
    dist = mx.sqrt(mx.maximum(sq[:, None] + sq[None, :] - 2.0 * g, 0.0))
    idx = mx.arange(k)
    pair = valid[:, None] & valid[None, :] & (idx[:, None] < idx[None, :])
    return mx.where(pair, dist, 0.0).sum(), pair.astype(mx.int32).sum()


# --- snapshot -------------------------------------------------------------------------------------------


def _quantiles(name: str, x: np.ndarray) -> dict[str, float | None]:
    """``<name>_q10/q50/q90`` of an alive-masked gene (None when nobody is alive)."""
    if not x.size:
        return {f"{name}_q{p}": None for p in (10, 50, 90)}
    q = np.quantile(x.astype(np.float64), (0.1, 0.5, 0.9))
    return {f"{name}_q{p}": float(v) for p, v in zip((10, 50, 90), q, strict=True)}


def _env_summary(h: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """The live environment levers at the snapshot (``env`` on rows of a director-enabled run).

    The row is written before the director acts at that step, so these are the values the last interval ran
    with; an action taken at the step shows up in the next row (and in ``director.jsonl``).
    """
    gain = h["env_patch_gain"]
    return {
        "regen_mult": float(h["env_regen_mult"]),
        "season_amp": float(h["env_season_amp"]),
        "season_period": int(h["env_season_period"]),
        "drift_mult": float(h["env_drift_mult"]),
        "imm_request": int(h["env_imm_request"]),
        "cat_radius": float(h["env_cat_radius"]),
        "cat_kill_p": float(h["env_cat_kill_p"]),
        "patch_gain_min": float(gain.min()),
        "patch_gain_max": float(gain.max()),
        "patches_off": int((gain == 0).sum()),
    }


def snapshot_metrics(s: State, cfg: Config, *, heavy: bool) -> dict[str, Any]:
    """Host floats/ints/lists describing the current state, with one host transfer (§19.2).

    Small device reductions (clade bins, chem means and, when ``heavy``, body sizes and weight diversity) and
    the per-agent arrays the host statistics need are evaluated together and read once. Energy totals are
    exact host int64; ``ledger_error`` is the absolute discrepancy of the global closure (§10.8), 0 when it
    holds (its sinks are ``state.sinks``: ``HEAT_BUOYANCY`` too in 3D). ``clade_entropy`` is in nats;
    ``tag_var`` is the trace of the population (ddof 0) covariance. ``resource_mean`` is per field cell (every
    layer in 3D) and ``chem_mean`` per channel over all its cells.
    """
    n_init = cfg.population.n_init
    step = int(to_numpy(s.step))
    dev: dict[str, mx.array] = {
        "alive": s.alive,
        "energy": s.energy,
        "birth_step": s.birth_step,
        "generation": s.generation,
        "tag": s.tag,
        "mut_sigma": s.mut_sigma,
        "mut_rate": s.mut_rate,
        "offspring_frac": s.offspring_frac,
        "brain_cost": s.brain_cost,
        "resource": s.resource,
        "chem_mean": s.chem.mean(axis=tuple(range(1, s.chem.ndim))),  # (1, 2) in 2D, (1, 2, 3) in 3D
        "clade_counts": ops.scatter_add_i32(n_init + 1, mx.clip(s.clade, 0, n_init), 1, s.alive),
        "total": s.total,
        "total0": s.total0,
        "ledger_cum": s.ledger_cum,
        "diag": s.diag,
    }
    if (
        cfg.director.enabled
    ):  # static: the director's live levers (a few scalars; one more entry in the transfer)
        dev.update({name: getattr(s, name) for name in ENV_FIELDS})
    if heavy:
        labels, converged = components(s.partner, s.alive, cfg.run.components_iters)
        dev["body_sizes"] = ops.scatter_add_i32(s.alive.shape[0], labels, 1, s.alive)
        dev["bodies_converged"] = converged
        dev["div_sum"], dev["div_pairs"] = _diversity_arrays(s, step, cfg)
    mx.eval(*dev.values())
    h = {name: to_numpy(a) for name, a in dev.items()}

    alive = h["alive"]
    n_alive = int(alive.sum())
    e = h["energy"].astype(np.int64)
    e_agents = int(e.sum())
    e_structure = n_alive * cfg.energy.structure
    e_field = int(h["resource"].astype(np.int64).sum())
    e_total = e_agents + e_structure + e_field
    cum = h["ledger_cum"].astype(np.int64)
    expected = int(h["total0"]) + int(cum[LedgerIdx.INJECT_REGEN] + cum[LedgerIdx.INJECT_IMMIGRATION])
    expected -= int(sum(int(cum[i]) for i in sinks(dims(cfg).nd)))
    ledger_error = abs(e_total - expected) + abs(int(h["total"]) - e_total)

    ea = e[alive]
    age = step - 1 - h["birth_step"][alive].astype(np.int64)
    gen = h["generation"][alive].astype(np.int64)
    counts = h["clade_counts"].astype(np.int64)
    if n_alive:
        p = counts[counts > 0] / n_alive
        clade = {
            "clades_alive": int(p.size),
            "clade_entropy": float(-(p * np.log(p)).sum()),
            "clade_simpson": float(1.0 - (p * p).sum()),
            "top_clade_share": float(p.max()),
        }
        tag = h["tag"][alive].astype(np.float64)
        tag_var = float(tag.var(axis=0).sum())
    else:
        clade = {"clades_alive": 0, "clade_entropy": None, "clade_simpson": None, "top_clade_share": None}
        tag_var = None

    out: dict[str, Any] = {
        "E_agents": e_agents,
        "E_structure": e_structure,
        "E_field": e_field,
        "E_total": e_total,
        "ledger_error": ledger_error,
        "energy_mean": float(ea.mean()) if n_alive else None,
        "energy_median": float(np.median(ea)) if n_alive else None,
        "age_mean": float(age.mean()) if n_alive else None,
        "generation_max": int(gen.max()) if n_alive else None,
        "generation_mean": float(gen.mean()) if n_alive else None,
        **clade,
        "tag_var": tag_var,
        **_quantiles("sigma", h["mut_sigma"][alive]),
        **_quantiles("rate", h["mut_rate"][alive]),
        **_quantiles("frac", h["offspring_frac"][alive]),
        "brain_cost_mean": float(h["brain_cost"][alive].mean()) if n_alive else None,
        "resource_mean": e_field / h["resource"].size,
        "chem_mean": [float(c) for c in h["chem_mean"]],
        "diag": int(h["diag"]),
    }
    if cfg.director.enabled:
        out["env"] = _env_summary(h)
    if heavy:
        out.update(_body_stats(h["body_sizes"], n_alive, bool(h["bodies_converged"])))
        pairs = int(h["div_pairs"])
        out["weight_diversity"] = float(h["div_sum"]) / pairs if pairs else None
    return out


# --- interval row ---------------------------------------------------------------------------------------

_SUMMED: Final = (
    ("births", CounterIdx.BIRTHS),
    ("birth_requests", CounterIdx.BIRTH_REQUESTS),
    ("births_blocked", CounterIdx.BIRTHS_BLOCKED),
    ("immigrants", CounterIdx.IMMIGRANTS),
    ("deaths_starved", CounterIdx.DEATHS_STARVED),
    ("deaths_killed", CounterIdx.DEATHS_KILLED),
    ("deaths_aged", CounterIdx.DEATHS_AGED),
    ("deaths_catastrophe", CounterIdx.DEATHS_CATASTROPHE),
    ("deaths_unlogged", CounterIdx.DEATHS_UNLOGGED),
    ("insolvent", CounterIdx.INSOLVENT),
    ("attacks", CounterIdx.ATTACKS),
    ("attack_hits", CounterIdx.ATTACK_HITS),
    ("bonds_formed", CounterIdx.BONDS_FORMED),
    ("bonds_broken", CounterIdx.BONDS_BROKEN),
    ("child_bonds", CounterIdx.CHILD_BONDS),
)


def _ratio(num: int | float, den: int | float, scale: float = 1.0) -> float | None:
    return float(num) / (scale * float(den)) if den else None


def interval_row(
    step: int,
    counters: np.ndarray,
    ledgers: np.ndarray,
    snap: Mapping[str, Any],
    perf: Mapping[str, float],
    *,
    nd: int = 2,
) -> dict[str, Any]:
    """One ``metrics.jsonl`` row from stacked counters (T,27), ledgers (T,16) and a snapshot (§19.2).

    Event counters are interval sums; the level counters ``N_ALIVE`` and ``BONDS_TOTAL`` are interval
    means, and the per-agent fractions are weighted by ``N_ALIVE`` over the interval. ``MAX_OCCUPANCY`` is an
    interval max. ``perf`` supplies ``wall_s`` and ``peak_mem_gb``. Ratios with a zero denominator are None.
    The snapshot keys (and ``carrying_capacity_est`` when the caller adds it) follow; "step" is the first key.
    ``nd`` is the world's dims (§27.8): a 3D world's counters are (T,30) and its ledger names slot 15
    ``HEAT_BUOYANCY``; counters of another width are refused rather than reshaped.
    """
    width = n_counters(nd)
    c = np.asarray(counters, dtype=np.int64)
    if c.ndim == 2 and c.shape[1] != width:
        raise ValueError(f"counters have {c.shape[1]} columns, a {nd}D world has {width}")
    c = c.reshape(-1, width)
    lg = np.asarray(ledgers, dtype=np.int64).reshape(-1, len(LedgerIdx))
    if c.shape[0] != lg.shape[0]:
        raise ValueError(f"counters ({c.shape[0]} steps) and ledgers ({lg.shape[0]} steps) differ in length")
    t = c.shape[0]
    cs, ls = c.sum(axis=0), lg.sum(axis=0)
    wall = perf.get("wall_s")
    alive_steps = int(cs[CounterIdx.N_ALIVE])

    row: dict[str, Any] = {
        "step": int(step),
        "steps": t,
        "wall_s": None if wall is None else float(wall),
        "steps_per_s": t / float(wall) if wall else None,
        "peak_mem_gb": None if perf.get("peak_mem_gb") is None else float(perf["peak_mem_gb"]),
        "n_alive": _ratio(alive_steps, t),
        "n_alive_min": int(c[:, CounterIdx.N_ALIVE].min()) if t else None,
    }
    row.update({name: int(cs[idx]) for name, idx in _SUMMED})
    row.update(
        {
            "bonds_total": _ratio(int(cs[CounterIdx.BONDS_TOTAL]), t),
            "mean_bonds_per_agent": _ratio(2 * int(cs[CounterIdx.BONDS_TOTAL]), alive_steps),
            "sharers_frac": _ratio(int(cs[CounterIdx.SHARERS]), alive_steps),
            "emitters_frac": _ratio(int(cs[CounterIdx.EMITTERS]), alive_steps),
            "predation_ratio": _ratio(int(ls[LedgerIdx.FLOW_STOLEN]), int(ls[LedgerIdx.FLOW_HARVEST])),
            "kin_attack_bias": _ratio(
                int(cs[CounterIdx.KIN_ATTACK_SUM] - cs[CounterIdx.KIN_ATTACK_BASE]),
                int(cs[CounterIdx.ATTACKS]),
                256.0,
            ),
            "kin_bond_bias": _ratio(
                int(cs[CounterIdx.KIN_BOND_SUM] - cs[CounterIdx.KIN_BOND_BASE]),
                int(cs[CounterIdx.BOND_PROPOSERS]),
                256.0,
            ),
            "ledger": {name: int(ls[i]) for i, name in enumerate(ledger_names(nd))},
            "cell_overflow": int(cs[CounterIdx.CELL_OVERFLOW]),
            "max_occupancy": int(c[:, CounterIdx.MAX_OCCUPANCY].max(initial=0)),
            "v_clips": int(cs[CounterIdx.V_CLIPS]),
        }
    )
    clash = sorted(set(row) & set(snap))
    if clash:
        raise ValueError(f"snapshot keys collide with interval keys: {clash}")
    row.update(snap)
    return row


# --- carrying capacity ----------------------------------------------------------------------------------


def _mean_abs_clipped_normal(sigma: float, clip: float) -> float:
    """``E|clip(X, ±clip)|`` for ``X ~ N(0, σ²)``."""
    if sigma <= 0.0:
        return 0.0
    z = clip / (sigma * math.sqrt(2.0))
    tail = clip * math.erfc(z) if math.isfinite(clip) else 0.0
    return sigma * math.sqrt(2.0 / math.pi) * (1.0 - math.exp(-z * z)) + tail


def founder_brain_cost(cfg: Config) -> float:
    """Expected founder ``brain_cost`` under the §6 init distribution (before flooring and dtype rounding)."""
    d, bc = dims(cfg), cfg.brain
    s1, s2 = bc.init_scale / math.sqrt(d.w1_rows), bc.init_scale / math.sqrt(d.w2_rows)
    l1 = (
        d.lay.D_IN * d.h * _mean_abs_clipped_normal(s1, bc.weight_clip)
        + d.h * d.h * _mean_abs_clipped_normal(s1 * bc.recurrent_scale, bc.weight_clip)
        + d.w2_rows * d.lay.D_OUT * _mean_abs_clipped_normal(s2, bc.weight_clip)
    )
    return cfg.energy.brain_cost_per_l1 * l1


def carrying_capacity(s: State, cfg: Config, step: mx.array | None = None, env: Env | None = None) -> float:
    """``K_est = Σ_c regen_rate_c / c_ref``, ``c_ref = basal_cost + E[founder brain_cost] + C_REF_MARGIN``.

    Regeneration uses the state's current patch positions, so the estimate is specific to the seed (and
    drifts with the patches); ``E[founder brain_cost]`` is the analytic expectation (``founder_brain_cost``).
    In 3D ``c_ref`` also holds a founder's buoyancy-cap upkeep ``floor(buoyancy_cap_cost·buoy_init)`` (§27.7),
    paid every step, and the regeneration is summed over every layer.
    With ``step`` and ``env`` (the director's rule tier) the regeneration is the live one: seasonal at
    ``step`` and scaled by the environment levers, as ``fields.regen_rate`` defines it.
    """
    regen = int(to_numpy(fields.regen_rate(s.patch_pos, cfg, step, env)).astype(np.int64).sum())
    c_ref = cfg.energy.basal_cost + founder_brain_cost(cfg) + layout.C_REF_MARGIN
    if is_3d(cfg):  # float32, as pay_costs3 computes it
        c_ref += int(np.floor(np.float32(cfg.energy.buoyancy_cap_cost) * np.float32(cfg.mutation.buoy_init)))
    return regen / c_ref
