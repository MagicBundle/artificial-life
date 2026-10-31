"""Replay a window of a run with the observers on and record the voice tables (the only MLX module here).

The run directory is read only. From the checkpoint at ``start`` (or a cached state) the step is replayed as
``labels.study`` does, with ``observe=True`` (the observer channel changes no state value, DESIGN §28 and
§39.8). Every ``every`` steps the three observed steps around the sample are turned into table rows
(``tables.build_sample``). Every ``cf_every``-th sample also runs the **counterfactual**: the same start
state is
stepped once, eagerly, under each ``comm.control`` (``none``, ``deaf``, ``decoy``) and the decisions are
compared with the run's own control, which says what hearing the real calls (or a stranger's) changes at that
moment. The counterfactual steps are separate pure calls on the same immutable state, so the replayed
trajectory is untouched; the eager step avoids a second compiled graph in the process.

Verification: with ``verify`` every stored ``metrics.jsonl`` row on the way is recomputed and compared
(``resim.compare_row``), and when the run has a checkpoint at the window's end its digest is compared with the
replayed state's. Usage is through ``python -m alife.analysis.voice log``.
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from alife import runio, spatial, structures
from alife.analysis import resim
from alife.analysis.signals import load_state
from alife.analysis.voice import spec, tables
from alife.analysis.voice.tables import FLAG_ATTACKED, FLAG_SPOKE, Geo, Snap, Tables
from alife.config import Config, config_hash, dims
from alife.geom import geometry as host_geometry
from alife.runio import RunDir
from alife.state import counter_blocks, state_digest, to_numpy
from alife.step import make_step

CF_METRICS = ("d_thrust", "flip_attack", "flip_bond", "flip_speak", "d_voice", "d_vel", "n")
CONTROLS = ("none", "deaf", "decoy")


def with_control(cfg: Config, control: str) -> Config:
    return dataclasses.replace(cfg, comm=dataclasses.replace(cfg.comm, control=control))


def geo_of(cfg: Config) -> Geo:
    c = cfg.comm
    return Geo(
        geo=host_geometry(cfg),
        d3=dims(cfg).d3,
        hear_range=c.range if c.range > 0 else cfg.sensing.radius,
        atten_pow=c.atten_pow,
    )


def snapshot(
    t: int, s: Any, acts: Any, cfg: Config, resource_flat: np.ndarray, geo: Any, lineage: np.ndarray
) -> Snap:
    """The host copy of the observed step that starts at ``t`` from the state ``s`` (arrays already
    evaluated)."""
    c = acts.ext.comm
    pos = to_numpy(s.pos)
    wind = np.zeros(pos.shape[0], np.float32)
    if cfg.weather.enabled:  # the wind drift speed at the start-of-step positions
        drift = structures.drift_at(
            s.wx, s.pos, getattr(s, "buoy_cap", None), s.alive, getattr(s, "mat", None), cfg
        )
        wind = np.linalg.norm(to_numpy(drift), axis=1).astype(np.float32)
    return Snap(
        step=t,
        alive=to_numpy(s.alive),
        uid=to_numpy(s.uid),
        clade=to_numpy(s.clade),
        lineage=lineage.copy(),
        energy=to_numpy(s.energy).astype(np.float32),
        loss=to_numpy(s.last_loss),
        harvest=to_numpy(s.last_harvest),
        pos=pos,
        bonds=(to_numpy(s.partner) >= 0).sum(axis=1).astype(np.int8),
        food=resource_flat[geo.cell_of(pos)].astype(np.float32),
        wind=wind,
        voice=to_numpy(c.voice),
        heard=to_numpy(c.heard),
        audible=to_numpy(c.audible),
        nbr=to_numpy(acts.nbr),
        flags=to_numpy(acts.flags),
        thrust=to_numpy(acts.thrust),
    )


def _decisions(s1: Any, out: Any) -> dict[str, np.ndarray]:
    a = out.acts
    return {
        "thrust": to_numpy(a.thrust).astype(np.int32),
        "flags": to_numpy(a.flags),
        "bond": to_numpy(a.bond_target),
        "voice": to_numpy(a.ext.comm.voice).astype(np.int32),
        "audible": to_numpy(a.ext.comm.audible),
        "vel": to_numpy(s1.vel),
    }


def counterfactual(
    state: Any, own: str, steps: dict[str, Callable[[Any], Any]], alive: np.ndarray
) -> dict[str, np.ndarray]:
    """``(2 modes, 2 groups, len(CF_METRICS))`` effect sizes of replacing the run's control ``own`` by each
    other
    control for one step from ``state``. Groups: rows that have a real audible neighbour under ``none`` (0)
    and
    rows that have none (1, the control group: ``deaf`` must change nothing there). The metrics are the mean
    absolute thrust change (thrust is a quantum of 1/256), the share of rows whose attack, bond target or
    speaking flipped, the mean summed absolute voice change, the mean change of the next velocity, and the
    group size."""
    dec: dict[str, dict[str, np.ndarray]] = {}
    for mode, fn in steps.items():
        s1, out = fn(state)
        mx.eval(s1.vel, out.acts.thrust, out.acts.flags, out.acts.bond_target, out.acts.ext.comm.voice)
        dec[mode] = _decisions(s1, out)
    ref = dec[own]
    groups = (alive & (dec["none"]["audible"] > 0), alive & (dec["none"]["audible"] == 0))
    res = np.full((2, 2, len(CF_METRICS)), np.nan, np.float32)
    others = [m for m in CONTROLS if m != own]
    for i, mode in enumerate(others):
        d = dec[mode]
        per_row = {
            "d_thrust": np.abs(d["thrust"] - ref["thrust"]) / 256.0,
            "flip_attack": ((d["flags"] ^ ref["flags"]) & FLAG_ATTACKED) != 0,
            "flip_bond": d["bond"] != ref["bond"],
            "flip_speak": ((d["flags"] ^ ref["flags"]) & FLAG_SPOKE) != 0,
            "d_voice": np.abs(d["voice"] - ref["voice"]).sum(axis=1) / 256.0,
            "d_vel": np.linalg.norm(d["vel"] - ref["vel"], axis=1),
        }
        for g, m in enumerate(groups):
            n = int(m.sum())
            for k, name in enumerate(CF_METRICS[:-1]):
                res[i, g, k] = float(per_row[name][m].mean()) if n else np.nan
            res[i, g, -1] = n
    return res


def record(
    run: Path,
    *,
    start: int,
    steps: int,
    out: Path,
    states_dir: Path | None = None,
    every: int = spec.SAMPLE_EVERY,
    rows: int = spec.SAMPLE_ROWS,
    seed: int = spec.SEED,
    cf_every: int = 4,
    groups: int = spec.LINEAGE_GROUPS,
    verify: bool = True,
    log: Callable[[str], None] = print,
) -> Tables:
    """Replay ``steps`` steps of ``run`` from ``start``, write ``out/tables.npz`` and return the tables."""
    rd = RunDir.open(Path(run))
    cfg = rd.config()
    if not cfg.comm.enabled:
        raise ValueError(f"{rd.path}: comm.enabled is false, there is no voice to record")
    if every < 2 or steps < 2 * every or rows < 1:
        raise ValueError("need every >= 2, steps >= 2 * every and rows >= 1")
    end = start + steps
    g = geo_of(cfg)
    state = load_state(rd, cfg, states_dir, start)
    sample_steps = list(range(start + every, end - 1, every))  # u: needs observed u-1, u, u+1 <= end - 1
    sample_set = set(sample_steps)
    need = {u + d for u in sample_steps for d in (-1, 0, 1)}
    cf_steps = set(sample_steps[::cf_every]) if cf_every > 0 else set()
    spatial.prepare_knn(cfg)
    step_fn = make_step(cfg, observe=True)
    own = cfg.comm.control
    cf_fns = (
        {m: make_step(with_control(cfg, m), compile=False, observe=True) for m in CONTROLS}
        if cf_steps
        else {}
    )
    rows_stored = {int(r["step"]): r for r in rd.metrics_rows()} if verify else {}
    last_row = max(rows_stored, default=-1)
    log(
        f"voice: {rd.path} from {start} to {end}: {len(sample_steps)} samples every {every}, "
        f"{len(cf_steps)} counterfactual, verify={verify}, own control={own}"
    )

    # lineage: the kin tags of the rows alive at the start are clustered once (k-means), and a birth
    # inherits its parent's group, so a group is a line of descent that the window follows (immigrants: -1)
    alive0 = to_numpy(state.alive)
    labels, centroids = tables.kin_groups(to_numpy(state.tag)[alive0], groups, seed)
    grp = np.full(alive0.shape[0], -1, np.int8)
    grp[alive0] = labels
    group_sizes = np.bincount(labels, minlength=len(centroids)).tolist()

    snaps: dict[int, Snap] = {}
    send_parts, hear_parts, cf_rows, cf_at, pop = [], [], [], [], []
    counters: list[mx.array] = []
    ledgers: list[mx.array] = []
    checked = 0
    t_wall, t_log = time.perf_counter(), start
    t = start
    while t < end:
        if t in cf_steps:
            alive = to_numpy(state.alive)
            cf_rows.append(counterfactual(state, own, cf_fns, alive))
            cf_at.append(t)
        prev_state = state if t in need else None
        state, o = step_fn(state)
        mx.eval(state, o.base(), *o.acts.ext.comm, o.acts.born_row, o.acts.born_parent)
        if prev_state is not None:
            flat = to_numpy(prev_state.resource).reshape(-1)
            snaps[t] = snapshot(t, prev_state, o.acts, cfg, flat, g.geo, grp)
        del prev_state
        born, parent = to_numpy(o.acts.born_row), to_numpy(o.acts.born_parent)  # descent: the child's group
        ok = born >= 0
        grp[born[ok]] = np.where(parent[ok] >= 0, grp[np.maximum(parent[ok], 0)], -1)
        if t < last_row:
            counters.append(o.counters)
            ledgers.append(o.ledger)
        t += 1
        stored = rows_stored.get(t)
        if stored is not None:
            if stored["steps"] == len(counters):
                bad = resim.compare_row(
                    stored,
                    counters,
                    ledgers,
                    int(to_numpy(state.total)),
                    nd=dims(cfg).nd,
                    weather=cfg.weather.enabled,
                    blocks=counter_blocks(cfg),
                )
                if bad:
                    raise resim.ResimMismatch(
                        f"step {t}: the observed replay differs from metrics.jsonl in {bad}"
                    )
                checked += 1
            counters, ledgers = [], []
        u = t - 2  # the step just observed started at t - 1 = u + 1: the last snapshot of the sample at u
        if u in sample_set and (u - 1) in snaps and u in snaps and (u + 1) in snaps:
            rng = np.random.default_rng([seed, u])
            s_rows, h_rows = tables.build_sample(snaps[u - 1], snaps[u], snaps[u + 1], g, rows, rng)
            send_parts.append(s_rows)
            hear_parts.append(h_rows)
            pop.append((u, int(snaps[u].alive.sum()), float((snaps[u].audible[snaps[u].alive] > 0).mean())))
        for k in [k for k in snaps if k < t - 3]:
            del snaps[k]
        if t - t_log >= 1000:
            rate = (t - t_log) / (time.perf_counter() - t_wall)
            log(f"voice: step {t}, {rate:.1f} steps/s, {len(send_parts)} samples, {checked} rows verified")
            t_wall, t_log = time.perf_counter(), t

    final_digest = state_digest(state)
    end_digest = None
    if verify:
        ck = dict(rd.checkpoints()).get(end)
        if ck is not None:
            ref = state_digest(runio.load_checkpoint(ck, cfg))
            end_digest = {"checkpoint": str(ck), "equal": ref == final_digest}
            if not end_digest["equal"]:
                raise resim.ResimMismatch(f"the replayed state at {end} differs from the run's checkpoint")
    meta = {
        "run": str(rd.path),
        "config_hash": config_hash(cfg),
        "control": own,
        "width": cfg.comm.width,
        "window": [start, end],
        "every": every,
        "rows": rows,
        "seed": seed,
        "cf_every": cf_every,
        "lineage_groups": {
            "k": len(centroids),
            "sizes_at_start": group_sizes,
            "centroids": centroids.tolist(),
        },
        "samples": len(send_parts),
        "population": pop,
        "rows_verified": checked,
        "end_digest": end_digest,
        "end_state_digest": final_digest,
        "device": str(mx.default_device()),
        "mlx_version": mx.__version__,
        "geometry": cfg.world.geometry,
        "dims": dims(cfg).nd,
        "hear_range": g.hear_range,
        "cf_modes": [m for m in CONTROLS if m != own],
        "cf_metrics": list(CF_METRICS),
    }
    cf = {"step": np.asarray(cf_at, np.int32), "stats": np.asarray(cf_rows, np.float32)} if cf_rows else {}
    tab = Tables(tables.concat(send_parts), tables.concat(hear_parts), meta, cf)
    tab.save(Path(out) / "tables.npz")
    return tab
