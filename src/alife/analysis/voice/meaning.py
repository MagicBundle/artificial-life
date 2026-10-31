"""Do the calls mean something? The meaning test of DESIGN §39.9 and §39.14, rules version 3.

Input: forks of one checkpoint with ``comm.control`` ``none``, ``deaf`` and ``decoy`` (the same seeds in every
arm), each with its voice log (``tables.npz``) over the same window, and the fork's ``metrics.jsonl``. The
voice is paid for in every arm, so a contrast isolates what the *information* is worth.

Rules version 3 is a revision decided after the v2 results were read (the amendments are in the
pre-registration; the v2 rule is in commit ``1be5a33``). The unit of inference is the seed, one evolved
checkpoint per world.

* **Call-specific transfer** (the language score): how much of the caller's situation reaches the hearer's
  behaviour *through the call content*. The none arm minus the **decoy** arm (the content is cut; presence,
  loudness and bearing are kept), per seed on the odd blocks, for the transfer pair with the largest such
  contrast on the **even** blocks (so the pair is not the shared environment's champion, and there is no
  winner's curse). The none arm minus the **deaf** arm is the secondary contrast: it also contains the
  knock-out of inputs the evolved brains are tuned to. Every arm of a seed is coded with the none arm's cut
  points. Across seeds: the mean, the one-sided 95% bounds (t, df = seeds - 1) and the seed rule of the suite
  (``stats.seed_call``) with the seed-twin noise floor.
* **Positive**: the seed rule says ``supported`` and the lower bound is above 0. A **meaning**: positive with
  a mean of at least ``min_effect`` bits. **Equivalent to nothing**: the upper bound is below ``equivalence``.
* **Use**: the one-step counterfactual (none to decoy) changes the audible rows' thrust by at least
  ``use_thrust`` in every seed, and the rows with no audible neighbour by exactly 0.
* **Fitness**: none beats deaf and decoy on one readout that passes in both (seed rule with the floor, Holm
  over the readouts). The deaf contrast is a knock-out (removing the heard inputs costs something whatever
  they carry); only the decoy contrast separates the content.
* **Cost matched** in every arm.

Verdicts: ``language`` (speaking, use, meaning, the secondary contrast not negative, fitness, cost);
``signalling with trace meaning`` (the primary contrast is positive, but not a language); ``signalling, no
detectable meaning (bound X)`` (the upper bound is below the equivalence bound); ``no evidence`` (nobody
speaks, or the data neither show a contrast nor bound it). ``code`` (sender side) and ``reception``
(confounded by the geometry of hearing, even with the audible count and the distance in its strata) are
reported, not decided on.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from alife.analysis.emergence import prereg as pr
from alife.analysis.emergence import stats
from alife.analysis.voice import analyse, spec
from alife.analysis.voice import measures as ms
from alife.analysis.voice.tables import Tables
from alife.analysis.w2 import Blocking, block_mean, block_ratio

MEANING_VERSION = 3


class Rules:
    """The decision constants: read from a frozen pre-registration, else the module defaults (exploratory)."""

    def __init__(self, prereg: dict[str, Any] | None = None):
        d = prereg or spec.default_prereg()
        if d.get("rules_version") != spec.RULES_VERSION:
            raise ValueError(
                f"the pre-registration is rules version {d.get('rules_version')}; this code decides version "
                f"{spec.RULES_VERSION} (the v2 rule is in commit 1be5a33)"
            )
        self.confirmatory = bool(prereg and prereg.get("frozen"))
        self.hash = pr.prereg_hash(prereg) if prereg else None
        self.alpha = d["decision"]["alpha"]
        self.cost_tol = d["cost"]["tolerance"]
        self.n_perm = d["tests"]["n_perm"]
        self.n_perm_block = d["tests"]["n_perm_block"]
        self.fitness_blocks = d["fitness"]["blocks"]
        self.fork_steps = d["arms"]["fork_steps"]
        self.table_window = tuple(d["arms"]["table_window"])
        self.n_blocks = d["sampling"]["blocks"]
        self.seed = d["sampling"]["seed"]
        self.every = d["sampling"]["every"]
        self.rows = d["sampling"]["rows"]
        self.readouts = list(d["fitness"]["readouts"])
        self.min_effect = d["contrast"]["min_effect_bits"]
        self.equivalence = d["contrast"]["equivalence_bits"]
        self.alpha_one_sided = d["contrast"]["alpha_one_sided"]
        self.use_thrust = d["use"]["use_thrust"]
        self.speaking_min = d["speaking_min"]
        self.amendments = d.get("amendments", [])
        self.revised_after_results = any(a.get("results_read") for a in self.amendments)

    def tests_params(self) -> dict[str, Any]:
        return {
            "n_perm": self.n_perm,
            "n_perm_block": self.n_perm_block,
            "seed": self.seed,
            "n_blocks": self.n_blocks,
        }


# --- one arm-seed -----------------------------------------------------------------------------------------


def load_fitness(run: Path, rules: Rules) -> dict[str, np.ndarray]:
    """Block series (``fitness_blocks`` blocks of the fork's steps) of the fitness readouts and the voice cost
    from a fork's ``metrics.jsonl``. A fork's step counter starts at the source's step."""
    rows = [
        json.loads(ln) for ln in (Path(run) / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if ln
    ]
    meta = json.loads((Path(run) / "meta.json").read_text(encoding="utf-8"))
    start = int(meta["start_step"])
    width = rules.fork_steps // rules.fitness_blocks
    blk = Blocking(start=start, transient=0, width=width, n=rules.fitness_blocks)
    step = np.array([r["step"] for r in rows], dtype=np.int64)
    col = lambda k: np.array([r.get(k, np.nan) for r in rows], dtype=np.float64)  # noqa: E731
    n, steps = col("n_alive"), col("steps")
    return {
        "population": block_mean(blk, step, n),
        "births": block_ratio(blk, step, col("births"), steps * n),
        "energy": block_mean(blk, step, col("energy_mean")),
        "comm_due": block_ratio(blk, step, col("comm_due"), steps * n),
        "last_step": np.array([step.max(initial=-1) - start]),
    }


class ArmRun:
    """One fork: its path, its tables and its fitness blocks."""

    def __init__(self, arm: str, seed: int, run: Path, tables_path: Path, rules: Rules):
        self.arm, self.seed, self.run = arm, seed, Path(run)
        self.tab = Tables.load(tables_path)
        self.window = tuple(self.tab.meta["window"])
        self.fit = load_fitness(run, rules)
        self.prep: analyse.Prepared | None = None
        meta = json.loads((self.run / "meta.json").read_text(encoding="utf-8"))
        self.fork_start = int(meta["start_step"])

    def as_designed(self, rules: Rules) -> bool:
        """The fork ran the pre-registered length, its log covers the pre-registered window and was made
        with the
        pre-registered sampling (every, rows, seed)."""
        lo, hi = rules.table_window
        m = self.tab.meta
        return bool(
            int(self.fit["last_step"][0]) == rules.fork_steps
            and tuple(self.window) == (self.fork_start + lo, self.fork_start + hi)
            and (m.get("every"), m.get("rows"), m.get("seed")) == (rules.every, rules.rows, rules.seed)
        )


def check_tests(tests: Mapping[str, Any], rules: Rules, name: str) -> None:
    """A stored ``summary.json`` ``tests`` block must have been run with the pre-registered permutations, seed
    and blocks (otherwise every p-value in it means something else)."""
    have = tests.get("params")
    if have != rules.tests_params():
        raise ValueError(
            f"{name}: the stored tests ran with {have}, the pre-registration says {rules.tests_params()}; "
            "rebuild the summary (python -m alife.analysis.voice summarize DIR)"
        )


def analyse_arm(
    a: ArmRun, rules: Rules, tests: dict[str, Any] | None = None, log: Callable[[str], None] = print
) -> dict:
    """The pooled tests of one fork (``tests`` from its summary.json if already computed; reported, not
    decided on)."""
    if tests is None:
        tests = analyse.run_all(
            a.tab,
            a.window,
            n_perm=rules.n_perm,
            n_perm_block=rules.n_perm_block,
            seed=rules.seed,
            n_blocks=rules.n_blocks,
            log=lambda m: log(f"{a.arm}_s{a.seed}: {m}"),
        )
    else:
        check_tests(tests, rules, f"{a.arm}_s{a.seed}")
    fam = {k: (v["best"] if v else None) for k, v in tests["families"].items()}
    blocks = {k: (v["blocks"] if v else None) for k, v in tests["families"].items()}
    return {"families": fam, "blocks": blocks}


# --- the contrast pair ------------------------------------------------------------------------------------


def select_contrast_pair(
    preps: Mapping[int, Mapping[str, analyse.Prepared]], rules: Rules
) -> tuple[str | None, dict[str, float]]:
    """The transfer pair with the largest **none minus decoy** contrast of the raw information on the **even**
    blocks, averaged over the seeds' blocks and seeds; only pairs present in every arm of every seed. Returns
    the name and the scores of all pairs."""
    per_seed: list[dict[str, float]] = []
    for arms in preps.values():
        a = analyse.block_information(arms["none"], "transfer", parity=0)
        b = analyse.block_information(arms["decoy"], "transfer", parity=0)
        d = {}
        for name in a.keys() & b.keys():
            diff = a[name] - b[name]
            if np.isfinite(diff).any():
                d[name] = float(np.nanmean(diff))
        per_seed.append(d)
    names = set.intersection(*(set(d) for d in per_seed)) if per_seed else set()
    if not names:
        return None, {}
    score = {n: float(np.mean([d[n] for d in per_seed])) for n in sorted(names)}
    return max(score, key=lambda n: score[n]), score


# --- contrasts --------------------------------------------------------------------------------------------


def holm_flags(ps: Sequence[float | None], alpha: float) -> tuple[list[float | None], list[bool]]:
    adj = stats.holm([np.nan if p is None else p for p in ps])
    adj_n = [None if not math.isfinite(v) else v for v in adj]
    return adj_n, [bool(v is not None and v < alpha) for v in adj_n]


def twin_floor(values: Mapping[str, Mapping[int, np.ndarray]]) -> float | None:
    """The seed-twin noise floor (DESIGN 44.8): the median magnitude of the mean same-arm difference between
    two seeds, over all arms and seed pairs (``stats.noise_floor``)."""
    diffs = []
    for by_seed in values.values():
        seeds = sorted(by_seed)
        for i, s in enumerate(seeds):
            for t in seeds[i + 1 :]:
                diffs.append(np.asarray(by_seed[s], float) - np.asarray(by_seed[t], float))
    return stats.noise_floor(diffs)


def seed_bounds(means: Sequence[float], alpha: float) -> dict[str, Any]:
    """Across-seed mean and the one-sided ``1 - alpha`` bounds (t, df = seeds - 1) of per-seed means."""
    v = np.asarray([m for m in means if m is not None and math.isfinite(m)], float)
    out: dict[str, Any] = {
        "n": int(v.size),
        "mean": float(v.mean()) if v.size else None,
        "lower": None,
        "upper": None,
    }
    if v.size >= 2:
        half = ms.t_upper(alpha, v.size - 1) * float(v.std(ddof=1)) / math.sqrt(v.size)
        out.update(lower=float(v.mean()) - half, upper=float(v.mean()) + half, sd=float(v.std(ddof=1)))
    return out


def contrast_call(
    none: Mapping[int, np.ndarray], ctrl: Mapping[int, np.ndarray], seeds: Sequence[int], floor: float | None
) -> dict[str, Any]:
    """Per-seed paired block tests of ``none - ctrl`` (positive = the real calls help), the seed rule with the
    noise floor, and the across-seed bounds of the seed means."""
    per = [stats.paired_test(none[s], ctrl[s], direction=1) for s in seeds]
    call = stats.seed_call(per, floor)
    means = [p["mean"] for p in per]
    return {"per_seed": per, "call": call, "bounds": seed_bounds(means, 0.05), "floor": floor}


def classify_contrast(c: Mapping[str, Any], rules: Rules) -> dict[str, Any]:
    """positive / meaning / equivalent for a primary contrast (see the module docstring)."""
    b = c["bounds"]
    bounds = seed_bounds([p["mean"] for p in c["per_seed"]], rules.alpha_one_sided)
    positive = bool(c["call"]["label"] == "supported" and bounds["lower"] is not None and bounds["lower"] > 0)
    meaning = bool(positive and bounds["mean"] >= rules.min_effect)
    equivalent = bool(bounds["upper"] is not None and bounds["upper"] < rules.equivalence)
    return {
        "positive": positive,
        "meaning": meaning,
        "equivalent": equivalent,
        "mean": bounds["mean"],
        "lower": bounds["lower"],
        "upper": bounds["upper"],
        "bounds": bounds,
        "two_sided_bounds": b,
    }


def transfer_contrasts(
    T: Mapping[str, Mapping[int, np.ndarray]], seeds: Sequence[int], rules: Rules
) -> dict[str, Any]:
    """The primary (none minus decoy) and secondary (none minus deaf) contrasts of the odd-block transfer
    values ``T[arm][seed]``, with the seed-twin noise floor of the same values (same-arm seed differences)."""
    floor = twin_floor({n: {s: T[n][s][1::2] for s in seeds} for n in T})
    out: dict[str, Any] = {}
    for c in ("decoy", "deaf"):
        ct = contrast_call(T["none"], T[c], seeds, floor)
        if c == "decoy":
            ct["class"] = classify_contrast(ct, rules)
        out[f"contrast_{c}"] = ct
    return out


def use_clause(arms: Mapping[str, Mapping[int, ArmRun]], rules: Rules) -> dict[str, Any]:
    """The counterfactual use clause from the none arm's logs (see the module docstring)."""
    per = []
    for s, a in sorted(arms["none"].items()):
        st = a.tab.cf.get("stats")
        modes = a.tab.meta.get("cf_modes", [])
        names = a.tab.meta.get("cf_metrics", [])
        if st is None or "decoy" not in modes:
            per.append({"seed": s, "d_thrust": None, "control_zero": None})
            continue
        st = st.astype(np.float64)
        i, k = modes.index("decoy"), names.index("d_thrust")
        aud = ms.nanmean(st[:, i, 0, :], axis=0)
        silent = st[:, i, 1, : len(names) - 1]
        per.append(
            {
                "seed": s,
                "d_thrust": float(aud[k]),
                "flip_attack": float(aud[names.index("flip_attack")]),
                "control_zero": bool(np.all(np.nan_to_num(silent, nan=0.0) == 0.0)),
            }
        )
    ok = bool(
        per
        and all(
            p["d_thrust"] is not None and p["d_thrust"] >= rules.use_thrust and p["control_zero"] for p in per
        )
    )
    return {"passed": ok, "per_seed": per, "use_thrust": rules.use_thrust}


def decide(res: dict[str, Any], rules: Rules) -> dict[str, Any]:
    """The clauses and the verdict from the assembled numbers (see the module docstring)."""
    a = rules.alpha
    prim = res["contrast_decoy"]["class"]
    sec = res["contrast_deaf"]
    sec_ok = bool(sec["bounds"]["mean"] is not None and sec["bounds"]["mean"] > 0)
    speaking = bool(res["speaker_frac"]["min"] >= rules.speaking_min)
    use = res["use"]
    fit = res["fitness"]
    passing: dict[str, dict[str, bool]] = {r: {} for r in rules.readouts}
    for c in ("deaf", "decoy"):
        adj, ok = holm_flags([fit[r][c]["call"]["p"] for r in rules.readouts], a)
        for i, r in enumerate(rules.readouts):
            fit[r][c]["call"]["p_holm"] = adj[i]
            passing[r][c] = fit[r][c]["call"]["label"] == "supported" and ok[i]
    fitness_ok = any(all(passing[r].values()) for r in rules.readouts)
    beats = {c: [r for r in rules.readouts if passing[r][c]] for c in ("deaf", "decoy")}
    cost_ok = bool(res["cost"]["matched"])
    clauses = {
        "speaking": {"passed": speaking, "min_share": res["speaker_frac"]["min"]},
        "use": {"passed": use["passed"]},
        "meaning": {
            "passed": prim["meaning"],
            "positive": prim["positive"],
            "equivalent": prim["equivalent"],
            "mean": prim["mean"],
            "lower": prim["lower"],
            "upper": prim["upper"],
        },
        "secondary_contrast": {"passed": sec_ok, "mean": sec["bounds"]["mean"]},
        "fitness": {"passed": bool(fitness_ok), "by_readout": passing},
        "cost_matched": {"passed": cost_ok},
    }
    bound = prim["upper"]
    fit_facts = "; ".join(
        f"none beats {c} on {', '.join(v) if v else 'no readout'}" for c, v in beats.items()
    )
    miss = [k for k, v in clauses.items() if not v["passed"]]
    if all(c["passed"] for c in clauses.values()):
        verdict, label = "language", "language"
        detail = (
            f"the real calls carry {prim['mean']:.4f} bits of call-specific transfer (lower bound "
            f"{prim['lower']:.4f}), hearers use them, and hearing them pays ({fit_facts})"
        )
    elif not speaking:
        verdict = label = "no evidence"
        detail = f"only {res['speaker_frac']['min'] * 100:.1f} % of the sampled rows speak"
    elif prim["positive"]:
        verdict = label = "signalling with trace meaning"
        size = "above" if prim["meaning"] else "below"
        detail = (
            f"the call-specific transfer (none minus decoy) is positive, {prim['mean']:.4f} bits, {size} the "
            f"minimum effect {rules.min_effect} (lower bound {prim['lower']:.4f}, "
            f"upper {prim['upper']:.4f}); "
            f"clauses that failed: {', '.join(miss) if miss else 'none'}; {fit_facts}"
        )
    elif prim["equivalent"]:
        verdict = "signalling, no detectable meaning"
        label = f"signalling, no detectable meaning (bound {bound:.4f} bits)"
        detail = (
            f"the call-specific transfer (none minus decoy) is {prim['mean']:.4f} bits with a one-sided 95% "
            f"upper bound of {bound:.4f}, below the equivalence bound {rules.equivalence}; hearers "
            f"{'do' if use['passed'] else 'do not'} change their decisions with the content; {fit_facts}"
        )
    else:
        verdict = label = "no evidence"
        detail = (
            f"the call-specific transfer is {prim['mean']} bits, not positive, and its upper bound "
            f"{bound} does not fall below the equivalence bound {rules.equivalence}: not enough to say"
        )
    if not cost_ok:
        detail += "; the voice cost is not matched across arms, so the fitness clause cannot be read"
    return {
        "verdict": verdict,
        "label": label,
        "detail": detail,
        "clauses": clauses,
        "confirmatory": bool(rules.confirmatory and res["as_designed"]),
        "revised_after_results": rules.revised_after_results,
    }


# --- the whole test ---------------------------------------------------------------------------------------


def run_meaning(
    arms: Mapping[str, Mapping[int, ArmRun]],
    rules: Rules,
    tests: Mapping[tuple[str, int], dict[str, Any]] | None = None,
    log: Callable[[str], None] = print,
    v2: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The full test over ``arms[name][seed]`` (names ``none``, ``deaf``, ``decoy``; the same seeds in each).
    ``v2`` is the archived result of the earlier rule, shown beside the verdict."""
    for name in spec.ARMS:
        if name not in arms or not arms[name]:
            raise ValueError(f"the arm {name!r} is missing")
    seeds = sorted(arms["none"])
    for name in spec.ARMS:
        if sorted(arms[name]) != seeds:
            raise ValueError(f"arm {name} has seeds {sorted(arms[name])}, none has {seeds}")
    if len(seeds) < 2:
        raise ValueError("a seed rule needs at least two seeds")
    windows = {tuple(a.window) for d in arms.values() for a in d.values()}
    steps_rel = {a.fit["last_step"][0] for d in arms.values() for a in d.values()}
    res: dict[str, Any] = {
        "version": MEANING_VERSION,
        "rules_version": spec.RULES_VERSION,
        "seeds": seeds,
        "windows": sorted(windows),
        "prereg_hash": rules.hash,
        "confirmatory": rules.confirmatory,
        "as_designed": all(a.as_designed(rules) for d in arms.values() for a in d.values()),
        "fork_steps_seen": sorted(int(x) for x in steps_rel),
        "arms": {},
        "v2": dict(v2) if v2 else None,
    }
    for name, d in arms.items():
        res["arms"][name] = {}
        for s, a in d.items():
            res["arms"][name][s] = analyse_arm(a, rules, (tests or {}).get((name, s)), log)
    # coding: every arm of a seed with the none arm's cut points
    preps: dict[int, dict[str, analyse.Prepared]] = {}
    for s in seeds:
        base = arms["none"][s]
        base.prep = analyse.prepare(base.tab, base.window, rules.n_blocks)
        preps[s] = {"none": base.prep}
        for name in ("deaf", "decoy"):
            a = arms[name][s]
            a.prep = analyse.prepare(
                a.tab,
                a.window,
                rules.n_blocks,
                edges=base.prep.edges,
                sym_thr=base.prep.sym_thr,
                heard_thr=base.prep.heard_thr,
                dist_edges=base.prep.dist_edges,
            )
            preps[s][name] = a.prep
    pair, scores = select_contrast_pair(preps, rules)
    log(f"contrast pair (none minus decoy, even blocks): {pair}")
    res["contrast_pair"] = pair
    res["contrast_pair_scores"] = dict(sorted(scores.items(), key=lambda kv: -kv[1])[:8])
    T: dict[str, dict[int, np.ndarray]] = {n: {} for n in spec.ARMS}
    for s in seeds:
        for name in spec.ARMS:
            rng = np.random.default_rng([rules.seed, s, 11])
            T[name][s] = (
                analyse.pair_blocks(preps[s][name], "transfer", pair, rules.n_perm_block, rng, parity=1)
                if pair
                else np.full(rules.n_blocks, np.nan)
            )
    res["transfer_blocks"] = {
        n: {str(s): [None if not np.isfinite(v) else float(v) for v in T[n][s]] for s in seeds} for n in T
    }
    res.update(transfer_contrasts(T, seeds, rules))
    # use
    res["use"] = use_clause(arms, rules)
    sp = {
        s: float((arms["none"][s].prep.s_sym > 0).mean()) if arms["none"][s].prep is not None else math.nan
        for s in seeds
    }
    res["speaker_frac"] = {"per_seed": {str(s): v for s, v in sp.items()}, "min": min(sp.values())}
    # fitness and cost
    res["fitness"] = {}
    for r in rules.readouts:
        res["fitness"][r] = {}
        fl = twin_floor({n: {s: arms[n][s].fit[r] for s in seeds} for n in spec.ARMS})
        for c in ("deaf", "decoy"):
            f_none = {s: arms["none"][s].fit[r] for s in seeds}
            f_ctrl = {s: arms[c][s].fit[r] for s in seeds}
            per = [stats.paired_test(f_none[s], f_ctrl[s], direction=1) for s in seeds]
            rel = [
                None
                if not np.isfinite(np.nanmean(f_ctrl[s])) or np.nanmean(f_ctrl[s]) == 0
                else float((np.nanmean(f_none[s]) - np.nanmean(f_ctrl[s])) / abs(np.nanmean(f_ctrl[s])))
                for s in seeds
            ]
            res["fitness"][r][c] = {
                "per_seed": per,
                "call": stats.seed_call(per, fl),
                "relative_per_seed": rel,
                "floor": fl,
            }
        res["fitness"][r]["means"] = {n: [_nm(arms[n][s].fit[r]) for s in seeds] for n in spec.ARMS}
    due = {n: float(np.nanmean([np.nanmean(arms[n][s].fit["comm_due"]) for s in seeds])) for n in spec.ARMS}
    base_due = due["none"]
    rel = {n: (due[n] - base_due) / base_due if base_due else math.nan for n in spec.ARMS}
    res["cost"] = {
        "comm_due_per_agent_step": due,
        "relative_to_none": {n: (None if not math.isfinite(v) else v) for n, v in rel.items()},
        "tolerance": rules.cost_tol,
        "matched": bool(all(math.isfinite(v) and abs(v) <= rules.cost_tol for v in rel.values())),
    }
    res["counterfactual"] = _cf_summary(arms)
    res["decision"] = decide(res, rules)
    res["confirmatory"] = res["decision"]["confirmatory"]
    return res


def _nm(x: np.ndarray) -> float | None:
    v = float(np.nanmean(x)) if np.isfinite(x).any() else math.nan
    return v if math.isfinite(v) else None


def _cf_summary(arms: Mapping[str, Mapping[int, ArmRun]]) -> dict[str, Any]:
    """Mean counterfactual effect per arm (own control against each other control), over seeds and samples."""
    out: dict[str, Any] = {}
    for name, d in arms.items():
        agg: dict[str, list[list[float]]] = {}
        names: list[str] = []
        for a in d.values():
            if "stats" not in a.tab.cf:
                continue
            names = a.tab.meta["cf_metrics"]
            for i, mode in enumerate(a.tab.meta["cf_modes"]):
                st = a.tab.cf["stats"][:, i, 0, :].astype(np.float64)  # audible rows
                agg.setdefault(mode, []).append(ms.nanmean(st, axis=0).tolist())
        out[name] = {
            mode: {n: _nm(np.array([v[j] for v in vs])) for j, n in enumerate(names)}
            for mode, vs in agg.items()
        }
    return out


# --- the report -------------------------------------------------------------------------------------------


def _f(v: Any, nd: int = 3) -> str:
    return "n/a" if v is None else f"{v:.{nd}f}"


def _p(v: Any) -> str:
    return "n/a" if v is None else "<1e-4" if v < 1e-4 else f"{v:.4f}"


def _row(*cells: Any) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def _ci(x: Mapping[str, Any], nd: int = 4) -> str:
    return f"{_f(x['mean'], nd)} [{_f(x['lo'], nd)}, {_f(x['hi'], nd)}]"


def to_markdown(res: dict[str, Any]) -> str:
    """The report: verdict (and the earlier rule's), clauses with their numbers, contrasts per seed,
    fitness."""
    d, seeds, cl = res["decision"], res["seeds"], res["decision"]["clauses"]
    if res["confirmatory"] and d["revised_after_results"]:
        status = "Frozen pre-registration, but revised after results (not blind)"
    elif res["confirmatory"]:
        status = "Confirmatory"
    elif not res["as_designed"]:
        status = "EXPLORATORY (a fork or a log does not match the pre-registered length, window or sampling)"
    else:
        status = "EXPLORATORY (the pre-registration is missing or not frozen)"
    lines = [
        "# Voice meaning test (rules version 3)",
        "",
        f"**Verdict: {d['label']}.** {d['detail']}.",
        "",
    ]
    if d["revised_after_results"]:
        lines += [
            "*Revised rule, decided after seeing the v2 results (pre-registration amendment A2).* The "
            "minimum effect and the equivalence bound were chosen knowing the v2 numbers.",
            "",
        ]
    if res.get("v2"):
        v2 = res["v2"]
        lines += [
            f"Earlier rule (v2, archived in `{v2.get('source')}`): **{v2.get('verdict')}**. "
            f"{v2.get('detail')}",
            "",
        ]
    lines += [
        f"{status}, pre-registration sha256 `{res['prereg_hash']}`. Seeds {seeds}, windows {res['windows']}; "
        f"the forks ran {res['fork_steps_seen']} steps. The unit of inference is the seed: one evolved "
        "checkpoint per world, so the seeds are near-replicates and the result is conditional on that "
        "checkpoint.",
        "",
        "## Clauses",
        "",
        _row("clause", "holds", "numbers"),
        _row("---", "---", "---"),
        _row(
            "speaking",
            cl["speaking"]["passed"],
            f"share of rows that speak, lowest seed {_f(cl['speaking']['min_share'], 3)}",
        ),
    ]
    use = res["use"]
    nums = "; ".join(
        f"s{p['seed']}: |thrust| change {_f(p['d_thrust'], 4)}, control group exactly 0: {p['control_zero']}"
        for p in use["per_seed"]
    )
    lines.append(
        _row(
            "use (counterfactual none to decoy)",
            cl["use"]["passed"],
            f"{nums} (needs >= {use['use_thrust']})",
        )
    )
    m = cl["meaning"]
    lines.append(
        _row(
            "call-specific transfer (none - decoy)",
            m["passed"],
            f"pair `{res['contrast_pair']}`; mean {_f(m['mean'], 4)} bits, one-sided 95% bounds "
            f"[{_f(m['lower'], 4)}, {_f(m['upper'], 4)}]; positive {m['positive']}, equivalent to nothing "
            f"(upper < bound) {m['equivalent']}",
        )
    )
    lines.append(
        _row(
            "secondary: none - deaf",
            cl["secondary_contrast"]["passed"],
            f"mean {_f(cl['secondary_contrast']['mean'], 4)} bits",
        )
    )
    lines.append(_row("fitness", cl["fitness"]["passed"], "see below"))
    cost = res["cost"]
    rel = ", ".join(f"{k} {_f(v, 3)}" for k, v in cost["relative_to_none"].items())
    lines.append(
        _row(
            "cost matched",
            cl["cost_matched"]["passed"],
            f"voice cost relative to none: {rel} (tolerance {cost['tolerance']})",
        )
    )
    cd, cu = res["contrast_decoy"], res["contrast_deaf"]
    lines += [
        "",
        f"## Contrasts of the pair `{res['contrast_pair']}` on the odd blocks (bits, 95% CI over blocks)",
        "",
        f"Seed-twin noise floor {_f(cd['floor'], 4)} bits. Seed rule (with the floor): none - decoy "
        f"`{cd['call']['label']}`, none - deaf `{cu['call']['label']}`.",
        "",
        _row("seed", "none - decoy", "none - deaf"),
        _row(*["---"] * 3),
    ]
    for i, s in enumerate(seeds):
        lines.append(_row(s, _ci(cd["per_seed"][i]), _ci(cu["per_seed"][i])))
    lines += [
        "",
        "Pair scores on the even blocks (none minus decoy, raw bits; top): "
        + ", ".join(f"{k} {_f(v, 4)}" for k, v in res["contrast_pair_scores"].items()),
        "",
        "## Fitness readouts (none minus control, per seed; relative to the control's mean)",
        "",
        "Removing the heard inputs (deaf) is a knock-out of evolved brains: a gap against deaf is the cost "
        "of "
        "the knock-out plus any value of information; replacing the content (decoy) is the content test.",
        "",
        _row("readout", "contrast", "label", "p", "Holm p", "mean difference per seed", "relative"),
        _row(*["---"] * 7),
    ]
    for r, byc in res["fitness"].items():
        for c in ("deaf", "decoy"):
            x = byc[c]
            per = "; ".join(_ci(p) for p in x["per_seed"])
            relv = ", ".join(_f(v * 100, 2) + "%" if v is not None else "n/a" for v in x["relative_per_seed"])
            call = x["call"]
            lines.append(
                _row(r, f"none-{c}", call["label"], _p(call["p"]), _p(call.get("p_holm")), per, relv)
            )
    lines += ["", "Arm means per seed:", ""]
    for r, byc in res["fitness"].items():
        arms = "; ".join(f"{n} " + ", ".join(_f(v, 4) for v in vs) for n, vs in byc["means"].items())
        lines.append(f"- {r}: {arms}")
    lines += ["", "## Reported, not decided on (best pair per arm, mean excess bits over seeds)", ""]
    lines += [_row("family", "none", "deaf", "decoy"), _row(*["---"] * 4)]
    for fam in ("code", "reception", "transfer"):
        vals = []
        for n in spec.ARMS:
            xs = [res["arms"][n][s]["families"].get(fam) for s in seeds]
            ex = [x["excess"] for x in xs if x and x.get("excess") is not None]
            pr_ = ", ".join(x["pair"] for x in xs[:1] if x)
            vals.append(f"{_f(float(np.mean(ex)) if ex else None, 4)} ({pr_})")
        lines.append(_row(fam, *vals))
    lines += [
        "",
        "`code` is sender-side (what the call says about the caller), not communication. `reception` is "
        "stratified on the audible count and the distance of the loudest source and is still reported only; "
        "the pooled z and "
        "p are non-binding (every z is far above any threshold, and the rows are not exchangeable).",
        "",
        "## Counterfactual decisions (audible rows; own control against each other control)",
        "",
    ]
    for n, modes in res["counterfactual"].items():
        for mode, mm in modes.items():
            lines.append(
                f"- {n} -> {mode}: |thrust| change {_f(mm.get('d_thrust'), 4)}, attack flips "
                f"{_f(mm.get('flip_attack'), 4)}, bond flips {_f(mm.get('flip_bond'), 4)}, speech flips "
                f"{_f(mm.get('flip_speak'), 4)}, call change {_f(mm.get('d_voice'), 4)}"
            )
    lines += [
        "",
        "## Reading it",
        "",
        "- The shuffle null removes any link between a caller's situation and a hearer's action. The shared "
        "environment (neighbours are in similar places) survives conditioning on the hearer's own situation, "
        "so the transfer in the none arm alone is not evidence of a call; the none-minus-decoy contrast, "
        "which "
        "keeps the environment, the presence, the loudness and the bearing and cuts only the content, is.",
        "- The thresholds (minimum effect, equivalence bound, use) are provisional and were chosen after the "
        "v2 results; the effect sizes are the excess over the null mean, in bits.",
    ]
    return "\n".join(lines) + "\n"
