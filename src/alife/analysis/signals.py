"""Do the signals agents still produce carry information about their situation? (``docs/SIGNALS.md``)

For a state ``s`` the chemical emission and the message are recomputed by re-running the start of the step
(S0–S8: flush, sensing, brain, decode, solvency) as a pure function, exactly as ``step.step`` does, because
``Actions.emit`` is not stored. The signals are the deposit quanta ``floor(256·emit_c)`` per chemical channel
(what reaches the field, §13.3) and ``msg_m`` per message component. Each is paired with the agent's
situation at the start of the step (resource at its field cell, ``last_harvest``, energy, the number of
sensed neighbours and how many of them are same-clade, other-clade or close kin) and with its behaviour one
step later (thrust, turn and attack decoded from the next state), and the mutual information is estimated
with ``mi.mi_knn`` against a shuffle null.

Variants per (signal, target): ``all`` (a random sample of alive agents), ``emits`` (the indicator
``q_c > 0`` on the same sample), ``emitters`` (``q_c`` among agents with ``q_c > 0`` only), and three
conditional MIs on the ``all`` sample, each with its null shuffled within the strata: ``patch`` (given the
nearest fertility patch: the shared environment), ``clade`` (given the agent's own clade: heritable
differences in signalling) and, for the kin counts only, ``crowd`` (given the number of sensed neighbours:
kin composition beyond local density). For an emit channel the chain rule gives
``I(q; Y) = I(1[q>0]; Y) + P(q>0)·I(q; Y | q>0)``.

Nothing here writes into a run directory. Usage::

    python -m alife.analysis.signals RUN --states analysis_cache/64k_s0 --steps 0 120000 ... --out FILE.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, NamedTuple

import mlx.core as mx
import numpy as np

from alife import actions, bonds, brain, fields, layout, lifecycle, ops, runio, sensing, spatial
from alife import energy as en
from alife.actions import Actions
from alife.analysis import mi
from alife.analysis.resim import cached_states
from alife.config import Config
from alife.rng import Stream, step_keys
from alife.runio import RunDir
from alife.spatial import Neighbors
from alife.state import State, age, to_numpy
from alife.step import make_step

SIGNALS: tuple[str, ...] = (*(f"emit{c}" for c in range(layout.NCH)), *(f"msg{m}" for m in range(layout.MSG)))
TARGETS: tuple[str, ...] = (
    "resource",
    "harvest",
    "energy",
    "crowd",
    "kin",
    "nonkin",
    "close_kin",
    "thrust_next",
    "turn_next",
    "attack_next",
)
VARIANTS: tuple[str, ...] = ("all", "emits", "emitters", "patch", "clade", "crowd")
KIN_TARGETS: tuple[str, ...] = ("kin", "nonkin", "close_kin")


class Decoded(NamedTuple):
    act: Actions  # masked by alive and solvency, as the step uses it from S8 on
    nbr: Neighbors
    fcell: mx.array  # (N,) start-of-step field cell


def decode(s: State, cfg: Config) -> Decoded:
    """The actions step ``s → s+1`` takes, by S0–S8 of ``step.step`` (no state change, no RNG advance)."""
    W1, W2 = lifecycle.flush_pending(s.W1, s.W2, s.pend_slot, s.pend_W1, s.pend_W2)  # S0
    _, ks = step_keys(s.key)  # S1
    a = age(s.step, s.birth_step)
    nbr = spatial.find_neighbors(s.pos, s.alive, cfg)  # S2
    geom = bonds.bond_geometry(s.pos, s.partner, cfg)  # S3
    fcell = fields.cell_index(s.pos, cfg)  # S4
    fobs = fields.sample_fields(s.resource, s.chem, fcell, s.heading, cfg)
    desc = sensing.neighbor_descriptors(s, nbr, cfg)  # S5
    x = sensing.build_inputs(s, a, nbr, desc, geom, fobs, ks[Stream.NOISE], cfg)
    _, y = brain.forward(W1, W2, x, s.h, cfg)  # S6
    act = actions.decode(y, desc, nbr, s.partner, s.alive, cfg)  # S7
    c = en.pay_costs(s.energy, act, bonds.n_bonds(s.partner), s.brain_cost, s.alive, cfg)  # S8
    return Decoded(actions.mask(act, c.solvent), nbr, fcell)


def emit_quanta(emit: mx.array) -> mx.array:
    """The deposit quanta ``floor(256·emit)`` of §13.3, (N,NCH) int32."""
    return ops.floor_i32(emit * 256.0)


def nearest_patch(pos: np.ndarray, patch_pos: np.ndarray, size: float) -> np.ndarray:
    """Index of the nearest patch centre on the torus, (N,) int64."""
    d = pos[:, None, :] - patch_pos[None, :, :]
    d -= size * np.round(d / size)
    return np.argmin((d * d).sum(-1), axis=1)


def agent_table(s: State, cfg: Config, step_fn: Any = None) -> dict[str, np.ndarray]:
    """Per alive agent: signals, situation and next-step behaviour (NaN where the agent did not survive).

    ``step_fn`` advances ``s`` one step for the next-step targets (default: ``make_step(cfg)``).
    """
    dec = decode(s, cfg)
    s1 = (step_fn or make_step(cfg))(s)[0]
    dec1 = decode(s1, cfg)
    q = emit_quanta(dec.act.emit)
    mx.eval(dec, dec1, q)

    h = {name: to_numpy(getattr(s, name)) for name in ("alive", "uid", "parent_uid", "clade", "energy")}
    alive = h["alive"]
    rows = np.flatnonzero(alive)
    idx, valid = to_numpy(dec.nbr.idx), to_numpy(dec.nbr.valid)
    uid, puid, clade = h["uid"], h["parent_uid"], h["clade"]
    same_clade = clade[idx] == clade[:, None]
    sibling = (puid[idx] == puid[:, None]) & (puid[:, None] >= 0)
    parent_child = (puid[idx] == uid[:, None]) | (uid[idx] == puid[:, None])
    close = sibling | parent_child

    survived = to_numpy(s1.alive) & (to_numpy(s1.uid) == uid)
    act1 = dec1.act

    def nxt(a: mx.array) -> np.ndarray:
        return np.where(survived, to_numpy(a).astype(np.float64), np.nan)

    resource = to_numpy(s.resource).reshape(-1)[to_numpy(dec.fcell)]
    table = {
        "row": rows,
        **{f"emit{c}": to_numpy(q)[:, c] for c in range(layout.NCH)},
        **{f"msg{m}": to_numpy(dec.act.msg)[:, m] for m in range(layout.MSG)},
        "resource": resource,
        "harvest": to_numpy(s.last_harvest),
        "energy": h["energy"],
        "crowd": valid.sum(1),
        "kin": (valid & same_clade).sum(1),
        "nonkin": (valid & ~same_clade).sum(1),
        "close_kin": (valid & close).sum(1),
        "thrust_next": nxt(act1.thrust),
        "turn_next": nxt(act1.turn),
        "attack_next": nxt(act1.attack),
        "patch": nearest_patch(to_numpy(s.pos), to_numpy(s.patch_pos), cfg.world.size),
        "clade": clade,
    }
    return {k: (v if k == "row" else v[rows]) for k, v in table.items()}


# --- the analysis -------------------------------------------------------------------------------------


class Job(NamedTuple):
    step: int
    signal: str
    target: str
    variant: str
    x: np.ndarray
    y: np.ndarray
    strata: np.ndarray | None


def _seed(*parts: Any) -> int:
    return int.from_bytes(hashlib.sha256(repr(parts).encode()).digest()[:8], "little")


def jobs_for(step: int, table: dict[str, np.ndarray], n_sample: int, seed: int) -> list[Job]:
    """All (signal, target, variant) estimates of one state, on fixed random samples of alive agents."""
    rng = np.random.default_rng(_seed(seed, step, "sample"))
    n = table["row"].size
    sample = rng.permutation(n)[: min(n_sample, n)]
    jobs: list[Job] = []
    for sig in SIGNALS:
        is_emit = sig.startswith("emit")
        for tgt in TARGETS:
            y_all = table[tgt].astype(np.float64)
            ok = sample[np.isfinite(y_all[sample])]
            x, y = table[sig][ok].astype(np.float64), y_all[ok]
            jobs.append(Job(step, sig, tgt, "all", x, y, None))
            jobs.append(Job(step, sig, tgt, "patch", x, y, table["patch"][ok]))
            jobs.append(Job(step, sig, tgt, "clade", x, y, table["clade"][ok]))
            if tgt in KIN_TARGETS:
                jobs.append(Job(step, sig, tgt, "crowd", x, y, table["crowd"][ok]))
            if is_emit:
                jobs.append(Job(step, sig, tgt, "emits", (x > 0).astype(np.float64), y, None))
                em = np.flatnonzero((table[sig] > 0) & np.isfinite(y_all))
                em = np.random.default_rng(_seed(seed, step, sig, "emitters")).permutation(em)[:n_sample]
                jobs.append(
                    Job(step, sig, tgt, "emitters", table[sig][em].astype(np.float64), y_all[em], None)
                )
    return jobs


def run_job(job: Job, n_perm: int, k: int, seed: int, min_stratum: int) -> dict[str, Any]:
    """One estimate with its shuffle null; degenerate inputs (too few rows, a constant) give NaNs."""
    rec: dict[str, Any] = {
        "step": job.step,
        "signal": job.signal,
        "target": job.target,
        "variant": job.variant,
    }
    rec["n"] = int(job.x.size)
    rng = np.random.default_rng(_seed(seed, job.step, job.signal, job.target, job.variant))
    constant = job.x.size == 0 or np.ptp(job.x) == 0 or np.ptp(job.y) == 0
    if job.x.size <= 4 * k or constant:
        return {**rec, **dict.fromkeys(("mi", "null_mean", "null_sd", "z", "p"), math.nan)}
    if job.strata is None:

        def stat(x: np.ndarray, y: np.ndarray) -> float:
            return mi.mi_knn(x, y, k, ranked=True)

        x, y = mi.rank(job.x, rng), mi.rank(job.y, rng)
        res = mi.perm_test(stat, x, y, n_perm, rng)
        if job.variant == "all":
            mm = mi.perm_test(mi.mi_binned_mm, job.x, job.y, n_perm, rng)
            rec.update(mi_mm=mm.mi, mm_null_mean=mm.null_mean, mm_z=mm.z)
    else:
        strata = job.strata
        cov = mi.cmi_strata(job.x, job.y, strata, lambda a, b: mi.mi_knn(a, b, k), min_stratum)[1]

        def stat(x: np.ndarray, y: np.ndarray) -> float:
            return mi.cmi_strata(x, y, strata, lambda a, b: mi.mi_knn(a, b, k), min_stratum)[0]

        res = mi.perm_test(stat, job.x, job.y, n_perm, rng, strata=strata)
        rec["coverage"] = cov
    return {**rec, **{f: getattr(res, f) for f in ("mi", "null_mean", "null_sd", "z", "p")}}


def summary(step: int, table: dict[str, np.ndarray]) -> dict[str, Any]:
    """Descriptive statistics of the signals at one state."""
    out: dict[str, Any] = {"step": step, "n_alive": int(table["row"].size)}
    emits_any = np.zeros(table["row"].size, dtype=bool)
    for c in range(layout.NCH):
        q = table[f"emit{c}"]
        emits_any |= q > 0
        out[f"emit{c}_frac"] = float((q > 0).mean())
        out[f"emit{c}_q_median_emitters"] = float(np.median(q[q > 0])) if (q > 0).any() else None
    out["emitters_frac"] = float(emits_any.mean())
    for m in range(layout.MSG):
        v = table[f"msg{m}"]
        out[f"msg{m}_mean"] = float(v.mean())
        out[f"msg{m}_sd"] = float(v.std())
        out[f"msg{m}_saturated_frac"] = float((np.abs(v) > 0.999).mean())
    out["patches_occupied"] = int(np.unique(table["patch"]).size)
    clades, counts = np.unique(table["clade"], return_counts=True)
    out["clades"] = int(clades.size)
    out["top_clade_share"] = float(counts.max() / counts.sum())
    out["crowd_mean"] = float(table["crowd"].mean())
    out["kin_share_of_neighbours"] = float(table["kin"].sum() / max(int(table["crowd"].sum()), 1))
    out["survived_frac"] = float(np.isfinite(table["thrust_next"]).mean())
    return out


def load_state(rd: RunDir, cfg: Config, states_dir: Path | None, step: int) -> State:
    """The run's checkpoint at ``step``, else a cached re-simulated state from ``states_dir``."""
    cands = dict(rd.checkpoints())
    if states_dir is not None:
        cands.update({st: p for st, p in cached_states(states_dir) if st not in cands})
    if step not in cands:
        raise FileNotFoundError(f"no checkpoint or cached state at step {step}")
    return runio.load_checkpoint(cands[step], cfg)


def analyse(
    run: Path,
    steps: Sequence[int],
    *,
    states_dir: Path | None,
    n_sample: int = 3000,
    n_perm: int = 100,
    k: int = 10,
    seed: int = 0,
    min_stratum: int = 50,
    workers: int | None = None,
    log: Any = print,
) -> dict[str, Any]:
    """Tables and summaries for ``steps``; see the module docstring."""
    rd = RunDir.open(Path(run))
    cfg = rd.config()
    step_fn = make_step(cfg)
    results: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    workers = workers or max(1, min(8, (os.cpu_count() or 2) // 2))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for step in steps:
            t0 = time.perf_counter()
            table = agent_table(load_state(rd, cfg, states_dir, step), cfg, step_fn)
            summaries.append(summary(step, table))
            jobs = jobs_for(step, table, n_sample, seed)
            del table
            futs = [pool.submit(run_job, j, n_perm, k, seed, min_stratum) for j in jobs]
            results.extend(f.result() for f in futs)
            log(f"signals: step {step}: {len(jobs)} estimates in {time.perf_counter() - t0:.0f} s")
    params = {"n_sample": n_sample, "n_perm": n_perm, "k": k, "seed": seed, "min_stratum": min_stratum}
    return {"run": str(rd.path), "params": params, "summaries": summaries, "results": results}


# --- reporting ----------------------------------------------------------------------------------------


def _fmt(v: Any, nd: int = 3) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "–"
    return f"{v:.{nd}f}"


def _zcell(r: dict[str, Any] | None, z_sig: float) -> str:
    if r is None:
        return ""
    z = _fmt(r["z"], 1)
    return f"**{z}**" if math.isfinite(r["z"]) and r["z"] >= z_sig else z


def _cells(r: dict[str, Any] | None, z_sig: float) -> list[str]:
    """``MI | null | z`` of one estimate (blank when absent)."""
    if r is None:
        return ["", "", ""]
    return [_fmt(r["mi"]), _fmt(r["null_mean"]), _zcell(r, z_sig)]


def markdown_tables(report: dict[str, Any], z_sig: float) -> str:
    """Per state: MI, null mean and z of every (signal, target) for the ``all``, ``emitters``, ``patch`` and
    ``clade`` variants, then the kin counts given the crowd. ``emits`` is in the JSON only."""
    by = {(r["step"], r["signal"], r["target"], r["variant"]): r for r in report["results"]}
    out: list[str] = []
    for summ in report["summaries"]:
        step = summ["step"]
        out.append(f"#### Step {step:,}\n")
        out.append(
            "| signal | target | MI | null | z | emitters MI | null | z | n | CMI patch | null | z "
            "| CMI clade | null | z |"
        )
        out.append("|" + "---|" * 15)
        for sig in SIGNALS:
            for tgt in TARGETS:
                m = by.get((step, sig, tgt, "emitters"))
                row = [sig, tgt, *_cells(by[(step, sig, tgt, "all")], z_sig), *_cells(m, z_sig)]
                row.append(str(m["n"]) if m else "")
                row += _cells(by[(step, sig, tgt, "patch")], z_sig)
                row += _cells(by[(step, sig, tgt, "clade")], z_sig)
                out.append("| " + " | ".join(row) + " |")
        out.append("")
        out.append(f"Kin counts given the number of sensed neighbours, step {step:,} (CMI, null, z):\n")
        out.append("| signal | " + " | ".join(f"{t} | null | z" for t in KIN_TARGETS) + " |")
        out.append("|" + "---|" * (1 + 3 * len(KIN_TARGETS)))
        for sig in SIGNALS:
            row = [sig]
            for tgt in KIN_TARGETS:
                row += _cells(by[(step, sig, tgt, "crowd")], z_sig)
            out.append("| " + " | ".join(row) + " |")
        out.append("")
    return "\n".join(out)


def trend_table(report: dict[str, Any], variant: str, z_sig: float) -> str:
    """Excess MI (``MI − null mean``, bits) of one variant across the states, a ``*`` where ``z ≥ z_sig``.

    Rows are the (signal, target) pairs significant at one state at least.
    """
    by = {(r["step"], r["signal"], r["target"], r["variant"]): r for r in report["results"]}
    steps = [s["step"] for s in report["summaries"]]
    out = [
        "| signal | target | " + " | ".join(f"{st:,}" for st in steps) + " |",
        "|" + "---|" * (2 + len(steps)),
    ]
    for sig in SIGNALS:
        for tgt in TARGETS:
            recs = [by.get((st, sig, tgt, variant)) for st in steps]
            if not any(r and math.isfinite(r["z"]) and r["z"] >= z_sig for r in recs):
                continue
            row = [sig, tgt]
            for r in recs:
                if r is None or not math.isfinite(r["mi"]):
                    row.append("–")
                    continue
                star = "*" if math.isfinite(r["z"]) and r["z"] >= z_sig else ""
                row.append(f"{r['mi'] - r['null_mean']:.3f}{star}")
            out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def merge_reports(reports: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """One report from several passes over the same run with the same parameters, ordered by step."""
    if len({json.dumps(r["params"], sort_keys=True) for r in reports}) > 1:
        raise ValueError("reports were made with different parameters")
    summaries = sorted((s for r in reports for s in r["summaries"]), key=lambda s: s["step"])
    steps = [s["step"] for s in summaries]
    if len(set(steps)) != len(steps):
        raise ValueError(f"a step appears in more than one report: {steps}")
    results = sorted((x for r in reports for x in r["results"]), key=lambda x: x["step"])
    return {**reports[0], "summaries": summaries, "results": results}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m alife.analysis.signals", description=__doc__.split("\n")[0])
    ap.add_argument("run", type=Path, nargs="?", help="run directory (read only)")
    ap.add_argument("--states", type=Path, default=None, help="directory of re-simulated states (resim)")
    ap.add_argument("--steps", nargs="+", type=int)
    ap.add_argument("--out", type=Path, help="results JSON (outside the run directory)")
    ap.add_argument("--render", nargs="+", type=Path, help="merge these results files instead of running")
    ap.add_argument("--md", type=Path, default=None, help="also write the markdown tables here")
    ap.add_argument("--n-sample", type=int, default=3000)
    ap.add_argument("--n-perm", type=int, default=100)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--z-sig", type=float, default=4.0, help="z at or above which a cell is bold")
    a = ap.parse_args(argv)
    if a.render:
        report = merge_reports([json.loads(p.read_text(encoding="utf-8")) for p in a.render])
    else:
        if a.run is None or not a.steps or a.out is None:
            ap.error("a run needs RUN, --steps and --out (or use --render)")
        if a.out.resolve().is_relative_to(a.run.resolve()):
            ap.error("--out must not be inside the run directory")
        report = analyse(
            a.run,
            a.steps,
            states_dir=a.states,
            n_sample=a.n_sample,
            n_perm=a.n_perm,
            k=a.k,
            seed=a.seed,
            workers=a.workers,
            log=lambda m: print(m, flush=True),
        )
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    if a.md is not None:
        trends = [
            f"#### Trend: {v}\n\n{trend_table(report, v, a.z_sig)}\n"
            for v in ("all", "emitters", "patch", "clade")
        ]
        a.md.write_text("\n".join(trends) + "\n" + markdown_tables(report, a.z_sig) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
