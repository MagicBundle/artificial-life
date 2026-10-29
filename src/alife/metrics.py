"""Emergence metrics: bodies, snapshots, interval rows and carrying capacity (§19, §10.10). Owned by WP5.

Observers only: nothing here touches ``state.key`` or feeds back into the simulation. Random sampling uses
``rng.observer_key(seed, step)`` (§14.1).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any, Final

import mlx.core as mx
import numpy as np

from alife import arms, ext, fields, layout, ops, rng, spatial, weather, weather_obs
from alife.config import Config, config_hash, dims, is_3d
from alife.state import (
    ENV_FIELDS,
    Counter3D,
    CounterBlocks,
    CounterIdx,
    Env,
    LedgerIdx,
    State,
    State3,
    ledger_names,
    n_counters,
    sinks,
    storm_counter_index,
    to_numpy,
)

# The key prefixes the registered features own in a snapshot or interval row (DESIGN §43.5): ``arms_``,
# ``build_``,
# ``bond_``, ``comm_``, ``farm_``; ``ext.check_registry`` rejects a prefix claimed twice.
METRIC_PREFIXES: Final = tuple(spec.metric_prefix for spec in ext.REGISTRY if spec.metric_prefix)

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


def _quantiles(name: str, x: np.ndarray, ps: tuple[int, ...] = (10, 50, 90)) -> dict[str, float | None]:
    """``<name>_q<p>`` for each percentile in ``ps`` of an alive-masked array (None when nobody is alive).

    ``p / 100`` is the correctly rounded quotient of two exact integers, so it is the same float as the
    literal ``0.1``, ``0.5``, ``0.9`` of the default percentiles.
    """
    if not x.size:
        return {f"{name}_q{p}": None for p in ps}
    q = np.quantile(x.astype(np.float64), [p / 100 for p in ps])
    return {f"{name}_q{p}": float(v) for p, v in zip(ps, q, strict=True)}


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


# --- 3D snapshot keys (§27.9) -----------------------------------------------------------------------------

# Equal z bands of ``band_frac``: band b is [b·height/4, (b+1)·height/4) and the top one holds the surface.
N_BANDS: Final = 4


def z_band(z: np.ndarray, height: float) -> np.ndarray:
    """The z band ``0..N_BANDS − 1`` of each height ``z``: ``floor(N_BANDS·z / height)``, clipped.

    ``z·N_BANDS`` is exact (a power of two), so a height on a band edge belongs to the band above it.
    """
    return np.clip(np.floor((z.astype(np.float64) * N_BANDS) / height), 0, N_BANDS - 1).astype(np.int64)


def _mutual_information(a: np.ndarray, b: np.ndarray, na: int, nb: int) -> float:
    """Plug-in mutual information in nats of two integer labellings (``a`` in [0, na), ``b`` in [0, nb)).

    Counts are integers and the ratio ``n_ab·n / (n_a·n_b)`` is formed from exact int64 products, so
    independent labellings (including a single clade) give exactly 0 rather than rounding noise.
    """
    joint = np.bincount(a * nb + b, minlength=na * nb).reshape(na, nb).astype(np.int64)
    n = int(joint.sum())
    na_, nb_ = joint.sum(axis=1)[:, None], joint.sum(axis=0)[None, :]
    nz = joint > 0
    ratio = (joint * n)[nz] / (na_ * nb_)[nz]
    return max(0.0, float((joint[nz] / n * np.log(ratio)).sum()))


def _snapshot_3d(h: Mapping[str, np.ndarray], cfg: Config) -> dict[str, Any]:
    """The 3D-only snapshot keys over the alive agents of a host-side snapshot (§27.9).

    Heights are the centre ``z`` in world units. ``surface_frac`` is the fraction within ``floor_eps`` of the
    surface bound, ``z ≥ z_hi − floor_eps`` (the mirror of the grounding band, although the surface never
    grounds; with ``floor_eps = 0`` it is the fraction resting on the clamp). ``band_frac`` is the fraction
    in each of ``N_BANDS`` equal z bands. ``clade_height_mi`` is the plug-in mutual information in nats
    between an agent's clade (immigrants binned at ``n_init``, as ``clade_counts``) and its z band; it is
    biased upwards by about ``(clades − 1)(bands − 1)/(2·n)`` nats for small samples. All None with nobody
    alive.
    """
    d, n_init = dims(cfg), cfg.population.n_init
    alive = h["alive"]
    z = h["pos_z"][alive].astype(np.float64)
    keys = ("height_mean", "height_q25", "height_q50", "height_q75", "surface_frac", "speed_mean")
    keys += ("pitch_abs_mean", "buoy_cap_q10", "buoy_cap_q50", "buoy_cap_q90", "band_frac", "clade_height_mi")
    if not z.size:
        return dict.fromkeys(keys)
    vel = h["vel"][alive].astype(np.float64)
    band = z_band(z, cfg.world.height)
    return {
        "height_mean": float(z.mean()),
        **_quantiles("height", z, (25, 50, 75)),
        "surface_frac": float((z >= d.z_hi - cfg.physics.floor_eps).mean()),
        "speed_mean": float(np.sqrt((vel * vel).sum(axis=1)).mean()),
        "pitch_abs_mean": float(np.abs(h["pitch"][alive].astype(np.float64)).mean()),
        **_quantiles("buoy_cap", h["buoy_cap"][alive]),
        "band_frac": [float(c) / z.size for c in np.bincount(band, minlength=N_BANDS)],
        "clade_height_mi": _mutual_information(
            np.clip(h["clade"][alive].astype(np.int64), 0, n_init), band, n_init + 1, N_BANDS
        ),
    }


def _body_z_extent(labels: np.ndarray, sizes: np.ndarray, alive: np.ndarray, z: np.ndarray) -> dict[str, Any]:
    """``body_z_extent_mean`` / ``_max`` over the bodies (components of ≥ 2 alive agents): ``max z − min z``
    of a body's member centres. z is bounded, never wrapped, so no unwrapping is needed (§27.9)."""
    member = alive & (sizes[labels] >= 2)
    if not member.any():
        return {"body_z_extent_mean": None, "body_z_extent_max": None}
    lab, zz = labels[member], z[member].astype(np.float64)
    order = np.argsort(lab, kind="stable")
    lab, zz = lab[order], zz[order]
    starts = np.concatenate([[0], np.flatnonzero(np.diff(lab)) + 1])
    ext = np.maximum.reduceat(zz, starts) - np.minimum.reduceat(zz, starts)
    return {"body_z_extent_mean": float(ext.mean()), "body_z_extent_max": float(ext.max())}


def _overlap_deep_fn(cfg: Config) -> Callable[[mx.array, mx.array], mx.array]:
    """The compiled count of alive agents whose nearest neighbour is closer than ``0.5·radius`` (one neighbour
    search; compiled so that its (N, Q) intermediates fuse as in the step), cached per config and per
    ``run.compile`` (the config hash ignores the latter)."""
    key = (config_hash(cfg), cfg.run.compile)
    fn = _OVERLAP_FNS.get(key)
    if fn is None:
        half = 0.5 * cfg.physics.radius

        def count(pos: mx.array, alive: mx.array) -> mx.array:
            nb = spatial.find_neighbors(pos, alive, cfg)
            return ((nb.dist[:, 0] < half) & alive).astype(mx.int32).sum()

        fn = _OVERLAP_FNS[key] = mx.compile(count) if cfg.run.compile else count
    return fn


_OVERLAP_FNS: dict[tuple[str, bool], Callable[[mx.array, mx.array], mx.array]] = {}


def rain_audit(s: State | State3, ws: weather.WeatherState, cfg: Config) -> mx.array:
    """``(4, layers)`` int32 per-layer sums for ``weather_obs.rain_shares`` (DESIGN §32), eager at a snapshot.

    Rows: ``base`` (``fields.regen_rate`` without the rain term), ``rain`` (the nominal ``weather.rain_regen``
    of the stored weather), ``inc_total = clip(min(base + rain, cap - R), 0)`` (what ``update_resource``
    books) and ``rain_real = inc_total - inc_base``. ``R`` is the end-of-step resource where the engine used
    the start-of-step one, the season is that of the last step and the patches are one drift later: exact for
    cells far from saturation or saturated, off by the one-step change otherwise (an approximation, stated).
    A layer sum is below ``2³¹`` (6,400 cells of at most 8,000 quanta each); the host adds them in int64."""
    fg = dims(cfg).fg
    env = s.env if cfg.director.enabled else None  # static
    step = s.step - 1
    base = fields.regen_rate(s.patch_pos, cfg, step, env)
    rain = weather.rain_regen(weather.rain_in(ws, cfg), cfg)
    room = cfg.fields.resource_cap - s.resource
    inc_total = mx.maximum(mx.minimum(base + rain, room), 0)
    real = inc_total - mx.maximum(mx.minimum(base, room), 0)
    rows = [a.reshape(-1, fg, fg).sum(axis=(1, 2)) for a in (base, rain, inc_total, real)]
    return mx.stack(rows).astype(mx.int32)


def snapshot_metrics(s: State | State3, cfg: Config, *, heavy: bool) -> dict[str, Any]:
    """Host floats/ints/lists describing the current state, with one host transfer (§19.2).

    Small device reductions (clade bins, chem means and, when ``heavy``, body sizes and weight diversity) and
    the per-agent arrays the host statistics need are evaluated together and read once. Energy totals are
    exact host int64; ``ledger_error`` is the absolute discrepancy of the global closure (§10.8), 0 when it
    holds (its sinks are ``state.sinks``: ``HEAT_BUOYANCY`` too in 3D). ``clade_entropy`` is in nats;
    ``tag_var`` is the trace of the population (ddof 0) covariance. ``resource_mean`` is per field cell (every
    layer in 3D) and ``chem_mean`` per channel over all its cells. A 3D world adds the keys of
    ``_snapshot_3d`` (and, when ``heavy``, ``body_z_extent_*``) after ``diag``; the 2D dict is unchanged
    (§27.9). With ``weather.enabled`` the stored ``wx``, the air speed at the live agents, the peak hash-cell
    occupancy (3D) and the count of deeply overlapping agents join the one transfer, and the dict gains the
    ``weather`` block of ``weather_obs.weather_block`` after ``env``; a weather-off dict is unchanged (§30).
    """
    n_init = cfg.population.n_init
    d3 = dims(cfg).d3
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
    if d3:  # static: the per-agent arrays of the 3D keys (§27.9)
        dev.update(pos_z=s.pos[:, 2], vel=s.vel, pitch=s.pitch, buoy_cap=s.buoy_cap, clade=s.clade)
    if cfg.weather.enabled:  # static: the weather block's inputs join the one transfer (§30)
        ws = weather.unpack(s.wx)
        buoy = s.buoy_cap if d3 else None
        wind = weather.agent_wind(ws, s.pos, buoy, s.alive, cfg)
        dev.update(wx=s.wx, air_speed=wind.speed, overlap_deep=_overlap_deep_fn(cfg)(s.pos, s.alive))
        dev["total_speed"], dev["eddy_speed"] = weather.air_speeds(ws, s.pos, buoy, s.alive, cfg)  # §32
        if cfg.weather.seed_gain > 0:  # static: the organism couplings' strengths (§32)
            dev["seed_r"] = weather.seeding_response(s.chem, ws, cfg)
        if cfg.weather.shade_gain > 0:  # static
            dev["shade_s"] = weather.shade_strength(s.pos, s.alive, cfg)
        if cfg.weather.rain_gain > 0:  # static
            dev["rain_audit"] = rain_audit(s, ws, cfg)
        if d3:
            dev["occupancy"] = spatial.band_peaks(spatial.hash_occupancy(s.pos, s.alive, cfg))[0]
    if heavy:
        labels, converged = components(s.partner, s.alive, cfg.run.components_iters)
        dev["body_sizes"] = ops.scatter_add_i32(s.alive.shape[0], labels, 1, s.alive)
        dev["bodies_converged"] = converged
        if d3:
            dev["labels"] = labels
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
    if d3:
        out.update(_snapshot_3d(h, cfg))
    if cfg.director.enabled:
        out["env"] = _env_summary(h)
    if cfg.weather.enabled:
        out["weather"] = weather_obs.weather_block(
            h["wx"],
            h["air_speed"][alive],
            cfg,
            occupancy=h.get("occupancy"),
            overlap_deep=int(h["overlap_deep"]),
            total_speed=h["total_speed"][alive],
            eddy_speed=h["eddy_speed"][alive],
            seed_r=h.get("seed_r"),
            shade_s=h.get("shade_s"),
            rain_audit=h.get("rain_audit"),
        )
    if heavy:
        out.update(_body_stats(h["body_sizes"], n_alive, bool(h["bodies_converged"])))
        if d3:
            out.update(_body_z_extent(h["labels"], h["body_sizes"], alive, h["pos_z"]))
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
    weather: bool = False,
    blocks: CounterBlocks = (),
) -> dict[str, Any]:
    """One ``metrics.jsonl`` row from stacked counters (T,27), ledgers (T,16) and a snapshot (§19.2).

    Event counters are interval sums; the level counters ``N_ALIVE`` and ``BONDS_TOTAL`` are interval
    means, and the per-agent fractions are weighted by ``N_ALIVE`` over the interval. ``MAX_OCCUPANCY`` is an
    interval max. ``perf`` supplies ``wall_s`` and ``peak_mem_gb``. Ratios with a zero denominator are None.
    The snapshot keys (and ``carrying_capacity_est`` when the caller adds it) follow; "step" is the first key.
    ``nd`` is the world's dims (§27.8): a 3D world's counters are (T,30) and its ledger names slot 15
    ``HEAT_BUOYANCY``; counters of another width are refused rather than reshaped. A 3D row also has
    ``z_floor_clamps`` and ``z_surface_clamps`` (interval sums), ``grounded_frac`` (``GROUNDED``, the agents
    that were grounded at the start of a step and survived it, over the ``N_ALIVE`` steps: the share of
    agent-steps spent grounded, in [0, 1] whatever dies) and ``buoy_heat`` (the interval sum of ledger
    slot 15, ``HEAT_BUOYANCY``), after ``v_clips`` (§27.9). ``weather`` (the run's ``weather.enabled``) adds
    the last counter, ``DEATHS_STORM``, to the expected width and ``deaths_storm`` (interval sum) to the row
    after ``child_bonds``; a weather-off row has neither (§30).
    """
    width = n_counters(nd, weather, blocks)
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
    if weather:  # DEATHS_STORM is the last counter (storm_counter_index), right after child_bonds in the row
        row["deaths_storm"] = int(cs[storm_counter_index(nd)])
    if blocks and blocks[0][0] == "protect":  # PROT_CAPPED, the first extension counter (DESIGN §43.4, §43.6)
        row["prot_capped_frac"] = _ratio(int(cs[n_counters(nd, weather)]), int(cs[CounterIdx.ATTACKS]))
    at = n_counters(nd, weather)
    for bname, names in blocks:  # the extension counter blocks, in step order (DESIGN §43.4)
        if bname == "structures":  # built structures (§38.9): sums for the flows, means for the levels
            by = {n: i for i, n in enumerate(names)}
            col = lambda n, at=at, by=by: cs[at + by[n]]  # noqa: E731
            row.update(
                {
                    "build_units": int(col("BUILD_UNITS")),
                    "build_wasted": int(col("MAT_WASTED")),
                    "build_decayed": int(col("MAT_DECAYED")),
                    "build_mat_total": _ratio(int(col("MAT_TOTAL")), t),
                    "build_mat_cells": _ratio(int(col("MAT_CELLS")), t),
                    "build_mat_closed": _ratio(int(col("MAT_CLOSED")), t),
                    "build_shielded": int(col("SHIELDED_ATTACKS")),
                    "build_trapped": int(col("TRAPPED")),
                    "build_aperture_violations": int(col("APERTURE_VIOLATIONS")),
                    "build_slowed_frac": _ratio(int(col("SLOWED")), alive_steps),
                    "build_hash_hidden": int(col("HASH_HIDDEN")),  # agent-steps hidden by a full hash cell
                }
            )
        elif bname == "bonds":  # bonds that pay (DESIGN §40.6): sums for the flows, means for the levels
            by = {n: i for i, n in enumerate(names)}
            col = lambda n, at=at, by=by: cs[at + by[n]]  # noqa: E731
            row.update(
                {
                    "bond_flux": int(ls[LedgerIdx.FLOW_SHARED]),  # q sent along bonds in the interval
                    "bond_share_slots": _ratio(int(col("SHARE_SLOTS_ACTIVE")), t),
                    "bond_role_switches": int(col("ROLE_SWITCHES")),
                    "bond_forager_frac": _ratio(int(col("N_FORAGERS")), alive_steps),
                    "bond_guard_frac": _ratio(int(col("N_GUARDS")), alive_steps),
                }
            )
        elif bname == "comm":  # the voice channel (§39.7): the share of the alive rows that spoke, the price
            by = {n: i for i, n in enumerate(names)}
            col = lambda n, at=at, by=by: cs[at + by[n]]  # noqa: E731
            row.update(
                {
                    "comm_speaker_frac": _ratio(int(col("COMM_SPEAKERS")), alive_steps),
                    "comm_due": int(col("COMM_DUE")),  # voice quanta paid in the interval (in heat_action)
                    "comm_due_share": _ratio(int(col("COMM_DUE")), int(ls[LedgerIdx.HEAT_ACTION])),
                }
            )
        at += len(names)
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
    if nd == 3:  # the 3D keys follow the shared ones (§27.9); a 2D row is unchanged
        row.update(
            {
                "z_floor_clamps": int(cs[Counter3D.Z_FLOOR_CLAMPS]),
                "z_surface_clamps": int(cs[Counter3D.Z_SURFACE_CLAMPS]),
                "grounded_frac": _ratio(int(cs[Counter3D.GROUNDED]), alive_steps),
                "buoy_heat": int(ls[LedgerIdx.HEAT_BUOYANCY]),
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


def layer_occupancy_weights(s: State3, cfg: Config) -> np.ndarray:
    """The 3D layer weights ``w_k = min(1, FZ·p_k)`` of ``carrying_capacity``, (FZ,) float64 (§27.10).

    ``p_k`` is the share of the alive agents whose field cell is in layer ``k``: ``fields.cell_index // FG²``,
    the layer ``clip(floor(z / field_cell), 0, FZ − 1)`` that harvest reads and deposits in (the same float32
    computation, so a position on a layer edge falls where the engine puts it). A layer holding at least its
    even share ``1 / FZ`` of the agents counts fully (``w_k = 1``), a sparser one in proportion to its
    occupancy. With ``n ≤ FZ`` alive every occupied layer has ``w_k = 1`` (a few survivors saturate the
    weights, so ``K_est`` is not continuous at ``n = 0``). With nobody alive there is no occupancy to weigh,
    and the floor layer alone counts.
    """
    d = dims(cfg)
    alive = to_numpy(s.alive)
    n = int(alive.sum())
    if n == 0:
        return (np.arange(d.fz) == 0).astype(np.float64)
    layer = to_numpy(fields.cell_index(s.pos, cfg)) // (d.fg * d.fg)
    p = np.bincount(layer[alive], minlength=d.fz) / n
    return np.minimum(1.0, d.fz * p)


def rain_of(s: State | State3, cfg: Config) -> weather.RainIn | None:
    """The rain term the state's last step fed the resource with, rebuilt from the stored ``wx`` (``P``,
    ``Dc``, ``h1``, ``h2``): None with weather off or ``rain_gain == 0`` (§30)."""
    return weather.rain_for(weather.unpack(s.wx), cfg) if cfg.weather.enabled else None


def _layer_regen(s: State | State3, cfg: Config, step: mx.array | None, env: Env | None) -> np.ndarray:
    """Integer regeneration per field cell, ``regen_rate``'s: (FG, FG) in 2D, (FZ, FG, FG) in 3D. With the
    weather's rain on it includes the rain of the state's last step, exactly (``rain_of``)."""
    return to_numpy(fields.regen_rate(s.patch_pos, cfg, step, env, rain_of(s, cfg))).astype(np.int64)


def _c_ref(cfg: Config) -> float:
    """``c_ref = basal_cost + E[founder brain_cost] + C_REF_MARGIN`` (and the founder's cap upkeep in 3D)."""
    c_ref = cfg.energy.basal_cost + founder_brain_cost(cfg) + layout.C_REF_MARGIN + arms.founder_upkeep(cfg)
    if is_3d(cfg):  # float32, as pay_costs3 computes it
        c_ref += int(np.floor(np.float32(cfg.energy.buoyancy_cap_cost) * np.float32(cfg.mutation.buoy_init)))
    return c_ref


def carrying_capacity(
    s: State | State3, cfg: Config, step: mx.array | None = None, env: Env | None = None
) -> float:
    """``K_est = Σ_c regen_rate_c / c_ref``, ``c_ref = basal_cost + E[founder brain_cost] + C_REF_MARGIN``.

    Regeneration uses the state's current patch positions, so the estimate is specific to the seed (and
    drifts with the patches); ``E[founder brain_cost]`` is the analytic expectation (``founder_brain_cost``).
    In 3D the regeneration is occupancy-weighted (§27.10): ``Σ_k regen_k · min(1, FZ·p_k)``, where
    ``regen_k`` is the integer regeneration of layer ``k`` (the sum of ``regen_rate(…)[k]``) and ``p_k`` the
    share of the alive agents living in it (``layer_occupancy_weights``). It equals the floor layer's alone
    (the benthic analogue of the 2D lattice) when every agent is in the floor layer, every layer's when the
    column is used evenly, and the floor layer's when nobody is alive, so it depends on the state's agents as
    well as on its patches. ``c_ref`` also holds a founder's buoyancy-cap upkeep
    ``floor(buoyancy_cap_cost·buoy_init)`` (§27.7), paid every step.
    With ``step`` and ``env`` (the director's rule tier) the regeneration is the live one: seasonal at
    ``step`` and scaled by the environment levers, as ``fields.regen_rate`` defines it.
    """
    rate = _layer_regen(s, cfg, step, env)
    regen: float
    if is_3d(cfg):  # static: weigh the per-layer sums by where the agents live
        per_layer = rate.reshape(rate.shape[0], -1).sum(axis=1).astype(np.float64)
        regen = float((per_layer * layer_occupancy_weights(s, cfg)).sum())
    else:
        regen = int(rate.sum())
    return regen / _c_ref(cfg)


def floor_carrying_capacity(
    s: State | State3, cfg: Config, step: mx.array | None = None, env: Env | None = None
) -> float:
    """The floor layer's ``K_est``: ``regen_0 / c_ref``, whoever is alive and wherever they are (§27.10).

    The director's extinction-risk test divides the population by this in 3D (owner decision 2026-10-15):
    the floor layer's food is the stable base of a population, while the occupancy-weighted
    ``carrying_capacity`` moves with where the agents are (a rescue's uniform immigrants raise it, so it
    stayed above the population that the rescue had saved) and reads a multiple of the realised population
    on ``DEV3D``. It is PR 4's rule, with the live regeneration under ``step`` and ``env`` as for
    ``carrying_capacity``. In 2D there is one lattice and one ``K_est``, so this is ``carrying_capacity``
    (the same value, the same integer sum).
    """
    if not is_3d(cfg):
        return carrying_capacity(s, cfg, step, env)
    rate = _layer_regen(s, cfg, step, env)
    return float(rate[0].sum()) / _c_ref(cfg)


_SEASON_PROBE_PERIOD: Final = (
    1 << 20
)  # a multiple of every table size we use: sample j is at exactly phase j / n


def seasonal_capacity_table(s: State | State3, cfg: Config, env: Env, amp: float, n: int = 32) -> np.ndarray:
    """The lattice's regeneration over one seasonal cycle, ``n`` equally spaced phases ``j / n``, relative to
    its value with no seasons: ``Σ regen_rate(step = j·P/n; env season = (amp, P)) / Σ regen_rate(env season
    off)`` for the state's patches, gains and levers (the director's rules version 2, ``director.seasons``).

    It is the engine's own map, so it carries what a plain sine of the weight would miss: the base fertility
    and the clip at 1 damp and skew the swing, and with ``season_mode = per_patch`` the patches' sines cancel
    in the total. Only the phase matters (the weights depend on ``step mod P`` through ``t / P``), so the
    table serves any period; it is evaluated at an artificial period that ``n`` divides, so phases are exact.
    The floor layer stands for the column in 3D (every layer is the 2D fertility times a fixed profile, so
    they swing alike). ``amp`` is the amplitude to evaluate at (the model scales other amplitudes linearly).
    """
    if n < 2 or _SEASON_PROBE_PERIOD % n:
        raise ValueError(
            f"the capacity table needs a size n ≥ 2 that divides {_SEASON_PROBE_PERIOD}, got {n}"
        )
    f32, i32 = mx.float32, mx.int32
    probe = env._replace(
        season_amp=mx.array(amp, dtype=f32), season_period=mx.array(_SEASON_PROBE_PERIOD, dtype=i32)
    )
    off = env._replace(season_amp=mx.array(0.0, dtype=f32), season_period=mx.array(0, dtype=i32))

    def total(step: int, e: Env) -> int:
        rate = fields.regen_rate(s.patch_pos, cfg, mx.array(step, dtype=i32), e)
        return int(to_numpy(rate[0] if is_3d(cfg) else rate).astype(np.int64).sum())

    base = total(0, off)
    if base <= 0:
        return np.ones(n, dtype=np.float64)  # no regeneration at all: nothing for the seasons to modulate
    return np.array(
        [total(j * (_SEASON_PROBE_PERIOD // n), probe) for j in range(n)], dtype=np.float64
    ) / float(base)
