"""Do the calls mean something? The meaning test of DESIGN §39.9 and §39.14 (numpy and the standard library).

Input: forks of one checkpoint with ``comm.control`` ``none``, ``deaf`` and ``decoy`` (the same seeds in every
arm), each with its voice log (``tables.npz``) over the same window, and the fork's ``metrics.jsonl``. The
voice is paid for in every arm, so a contrast isolates what the *information* is worth.

The clauses of the operational definition (spec section 7.2), each tested the way the emergence suite tests an
arm contrast (paired block differences per seed, ``w2.classify`` over the seeds, Stouffer's combined p, Holm
where several tests share a family, the thresholds read from the frozen pre-registration):

* **code**, ``none`` arm: the call type carries information about the caller's context (max-T permutation
  test,
  every seed).
* **transfer**, ``none`` arm: the situation of the loudest audible caller reaches the hearer's behaviour
  beyond
  the hearer's own situation: at least 0.05 bits and z >= 4 above the shuffle null (the spec's number).
* **transfer contrast**: that information is absent in ``deaf`` and at the ``decoy`` level or below. The pair
  is chosen on the even blocks of the ``none`` arm and estimated on the odd blocks of every arm (coded with
  the
  ``none`` arm's cut points), so the contrast has no winner's curse; ``none`` minus ``deaf`` and ``none``
  minus
  ``decoy`` are positive in every seed.
* **reception**, ``none`` arm: hearers act differently on different calls, given their own situation.
* **fitness**: ``none`` beats ``deaf`` and ``decoy`` on one readout (population, births per agent-step, mean
  energy) that passes in both contrasts, with Holm over the readouts.
* **cost matched**: the voice costs the same per agent-step in every arm (it is paid whether heard or not).

The verdict: ``language`` when every clause holds; ``signalling without meaning`` when the calls carry
information or are used but a clause fails (the detail names it: information without use, use without
benefit); ``no evidence`` otherwise. Information without use, or use without benefit, is never called a
language (spec 7.2).
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

MEANING_VERSION = 1


class Rules:
    """The decision constants: read from a frozen pre-registration, else the module defaults (exploratory)."""

    def __init__(self, prereg: dict[str, Any] | None = None):
        d = prereg or spec.default_prereg()
        self.confirmatory = bool(prereg and prereg.get("frozen"))
        self.hash = pr.prereg_hash(prereg) if prereg else None
        self.thresholds = d["thresholds"]
        self.alpha = d["decision"]["alpha"]
        self.cost_tol = d["cost"]["tolerance"]
        self.n_perm = d["tests"]["n_perm"]
        self.n_perm_block = d["tests"]["n_perm_block"]
        self.fitness_blocks = d["fitness"]["blocks"]
        self.fork_steps = d["arms"]["fork_steps"]
        self.table_window = tuple(d["arms"]["table_window"])
        self.n_blocks = d["sampling"]["blocks"]
        self.seed = d["sampling"]["seed"]
        self.readouts = list(d["fitness"]["readouts"])


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
        """The fork ran the pre-registered length and its log covers the pre-registered window."""
        lo, hi = rules.table_window
        return bool(
            int(self.fit["last_step"][0]) == rules.fork_steps
            and tuple(self.window) == (self.fork_start + lo, self.fork_start + hi)
        )


def _pass(best: Mapping[str, Any] | None, thr: Mapping[str, float]) -> dict[str, Any]:
    """The threshold test of a family's best pair: ``passed`` and the numbers behind it."""
    if best is None:
        return {"passed": False, "reason": "not enough rows"}
    z, ex, p = best["z"], best["excess"], best["p_family"]
    ok = None not in (z, ex, p) and z >= thr["z"] and ex >= thr["bits"] and p <= thr["p"]
    return {"passed": bool(ok), "pair": best["pair"], "excess": ex, "z": z, "p_family": p, "mi": best["mi"]}


def analyse_arm(
    a: ArmRun, rules: Rules, tests: dict[str, Any] | None = None, log: Callable[[str], None] = print
) -> dict:
    """The pooled tests of one fork (``tests`` from its summary.json if already computed)."""
    if tests is None:
        tests = analyse.run_all(
            a.tab,
            a.window,
            n_perm=rules.n_perm,
            n_perm_block=rules.n_perm_block,
            seed=rules.seed,
            log=lambda m: log(f"{a.arm}_s{a.seed}: {m}"),
        )
    fam = {k: (v["best"] if v else None) for k, v in tests["families"].items()}
    return {"tests": tests, "families": fam}


def select_transfer_pair(p: analyse.Prepared, rules: Rules) -> str | None:
    """The transfer pair with the largest z on the **even** blocks of a ``none`` arm (50 permutations)."""
    pairs = analyse.build_family(p, "transfer", mask=(p.h_block >= 0) & (p.h_block % 2 == 0))
    if not pairs or len(pairs[0].x) < 100:
        return None
    res = ms.family_test(pairs, 50, np.random.default_rng([rules.seed, 7]), spec.MIN_STRATUM)
    return res.names[res.best]


# --- contrasts --------------------------------------------------------------------------------------------


def _interval(per_seed: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The decision over seeds of one contrast (``stats.seed_call``)."""
    return stats.seed_call(per_seed)


def contrast_blocks(
    none: Mapping[int, np.ndarray], ctrl: Mapping[int, np.ndarray], seeds: Sequence[int]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Per-seed paired block tests of ``none - ctrl`` (positive = hearing the real calls helps) and the
    call."""
    per = [stats.paired_test(none[s], ctrl[s], direction=1) for s in seeds]
    return per, _interval(per)


def holm_flags(ps: Sequence[float | None], alpha: float) -> tuple[list[float | None], list[bool]]:
    adj = stats.holm([np.nan if p is None else p for p in ps])
    adj_n = [None if not math.isfinite(v) else v for v in adj]
    return adj_n, [bool(v is not None and v < alpha) for v in adj_n]


def decide(res: dict[str, Any], rules: Rules) -> dict[str, Any]:
    """The clauses and the verdict from the assembled numbers (see the module docstring)."""
    thr, a = rules.thresholds, rules.alpha
    seeds = res["seeds"]
    none = res["arms"]["none"]
    code = [_pass(none[s]["families"].get("code"), thr["code"]) for s in seeds]
    trn = [_pass(none[s]["families"].get("transfer"), thr["transfer"]) for s in seeds]
    rec = [_pass(none[s]["families"].get("reception"), thr["reception"]) for s in seeds]
    # the arm contrast on the transfer pair, Holm over the two contrasts
    tc = res["transfer_contrast"]
    adj, ok_adj = holm_flags([tc[c]["call"]["p"] for c in ("deaf", "decoy")], a)
    contrast_ok = all(
        tc[c]["call"]["label"] == "supported" and ok_adj[i] for i, c in enumerate(("deaf", "decoy"))
    )
    for i, c in enumerate(("deaf", "decoy")):
        tc[c]["call"]["p_holm"] = adj[i]
    # fitness: Holm over the readouts within each contrast; one readout must pass in both
    fit = res["fitness"]
    passing: dict[str, dict[str, bool]] = {r: {} for r in rules.readouts}
    for c in ("deaf", "decoy"):
        adj_f, ok_f = holm_flags([fit[r][c]["call"]["p"] for r in rules.readouts], a)
        for i, r in enumerate(rules.readouts):
            fit[r][c]["call"]["p_holm"] = adj_f[i]
            passing[r][c] = fit[r][c]["call"]["label"] == "supported" and ok_f[i]
    fitness_ok = any(all(passing[r].values()) for r in rules.readouts)
    cost = res["cost"]
    cost_ok = bool(cost["matched"])
    clauses = {
        "code": {"passed": all(c["passed"] for c in code), "per_seed": code},
        "transfer": {"passed": all(c["passed"] for c in trn), "per_seed": trn},
        "transfer_contrast": {"passed": bool(contrast_ok), "p_holm": adj},
        "reception": {"passed": all(c["passed"] for c in rec), "per_seed": rec},
        "fitness": {"passed": bool(fitness_ok), "by_readout": passing},
        "cost_matched": {"passed": cost_ok},
    }
    info = clauses["code"]["passed"] or clauses["transfer"]["passed"] or clauses["reception"]["passed"]
    used = clauses["reception"]["passed"] or clauses["transfer_contrast"]["passed"]
    miss = [k for k, v in clauses.items() if not v["passed"]]
    beats = {c: [r for r in rules.readouts if passing[r][c]] for c in ("deaf", "decoy")}

    def fit_facts() -> str:
        part = [f"none beats {c} on {', '.join(v) if v else 'no readout'}" for c, v in beats.items()]
        return "; ".join(part)

    if all(c["passed"] for c in clauses.values()):
        verdict, detail = "language", "every clause of the operational definition holds"
    elif info or used:
        verdict = "signalling without meaning"
        if not used:
            detail = (
                "information without use: the calls carry information, but hearers' behaviour does not use it"
            )
        elif not clauses["fitness"]["passed"]:
            if beats["deaf"] and not beats["decoy"]:
                kind = "use with a benefit of hearing but none specific to the calls"
            elif beats["deaf"] or beats["decoy"]:
                kind = "use with a partial benefit"
            else:
                kind = "use without benefit"
            detail = f"{kind}: hearers use the calls; {fit_facts()}"
        else:
            detail = "partial: " + ", ".join(miss) + " did not hold"
        detail += "; clauses that failed: " + ", ".join(miss)
        if not cost_ok:
            detail += "; the voice cost is not matched across arms, so the fitness clause cannot be read"
    else:
        verdict = "no evidence"
        detail = "no information clause passes"
        if clauses["fitness"]["passed"]:
            detail += " (hearing is worth something in fitness, but no measured call information explains it)"
    return {
        "verdict": verdict,
        "detail": detail,
        "clauses": clauses,
        "confirmatory": bool(rules.confirmatory and res["as_designed"]),
    }


# --- the whole test ---------------------------------------------------------------------------------------


def run_meaning(
    arms: Mapping[str, Mapping[int, ArmRun]],
    rules: Rules,
    tests: Mapping[tuple[str, int], dict[str, Any]] | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """The full test over ``arms[name][seed]`` (names ``none``, ``deaf``, ``decoy``; the same seeds in
    each)."""
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
        "seeds": seeds,
        "windows": sorted(windows),
        "prereg_hash": rules.hash,
        "confirmatory": rules.confirmatory,
        "as_designed": all(a.as_designed(rules) for d in arms.values() for a in d.values()),
        "fork_steps_seen": sorted(int(x) for x in steps_rel),
        "arms": {},
    }
    for name, d in arms.items():
        res["arms"][name] = {}
        for s, a in d.items():
            tj = (tests or {}).get((name, s))
            res["arms"][name][s] = analyse_arm(a, rules, tj, log)
    # the transfer pair: the none arm's even blocks, per seed; estimated on odd blocks of every arm of the
    # seed
    T: dict[str, dict[int, np.ndarray]] = {n: {} for n in spec.ARMS}
    pairs: dict[int, str | None] = {}
    for s in seeds:
        base = arms["none"][s]
        base.prep = analyse.prepare(base.tab, base.window, rules.n_blocks)
        pair = select_transfer_pair(base.prep, rules)
        pairs[s] = pair
        for name in spec.ARMS:
            a = arms[name][s]
            a.prep = (
                base.prep
                if name == "none"
                else analyse.prepare(
                    a.tab,
                    a.window,
                    rules.n_blocks,
                    edges=base.prep.edges,
                    sym_thr=base.prep.sym_thr,
                    heard_thr=base.prep.heard_thr,
                    dist_edges=base.prep.dist_edges,
                )
            )
            rng = np.random.default_rng([rules.seed, s, 11])
            T[name][s] = (
                analyse.pair_blocks(a.prep, "transfer", pair, rules.n_perm_block, rng, parity=1)
                if pair
                else np.full(rules.n_blocks, np.nan)
            )
        log(f"seed {s}: transfer pair {pair}")
    res["transfer_pairs"] = {str(s): p for s, p in pairs.items()}
    res["transfer_blocks"] = {
        n: {str(s): [None if not np.isfinite(v) else float(v) for v in T[n][s]] for s in seeds} for n in T
    }
    res["transfer_contrast"] = {}
    for c in ("deaf", "decoy"):
        per, call = contrast_blocks(T["none"], T[c], seeds)
        res["transfer_contrast"][c] = {"per_seed": per, "call": call}
    # fitness and cost
    res["fitness"] = {}
    for r in rules.readouts:
        res["fitness"][r] = {}
        for c in ("deaf", "decoy"):
            f_none = {s: arms["none"][s].fit[r] for s in seeds}
            f_ctrl = {s: arms[c][s].fit[r] for s in seeds}
            per, call = contrast_blocks(f_none, f_ctrl, seeds)
            rel = [
                None
                if not np.isfinite(np.nanmean(f_ctrl[s])) or np.nanmean(f_ctrl[s]) == 0
                else float((np.nanmean(f_none[s]) - np.nanmean(f_ctrl[s])) / abs(np.nanmean(f_ctrl[s])))
                for s in seeds
            ]
            res["fitness"][r][c] = {"per_seed": per, "call": call, "relative_per_seed": rel}
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
    # counterfactual effects, descriptive
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
    """The report: verdict, clauses with their numbers, contrasts per seed, fitness readouts."""
    d, seeds, cl = res["decision"], res["seeds"], res["decision"]["clauses"]
    if res["confirmatory"]:
        status = "Confirmatory"
    elif not res["as_designed"]:
        status = "EXPLORATORY (a fork or a log does not match the pre-registered length or window)"
    else:
        status = "EXPLORATORY (the pre-registration is missing or not frozen)"
    lines = [
        "# Voice meaning test",
        "",
        f"**Verdict: {d['verdict']}.** {d['detail']}.",
        "",
        f"{status}, pre-registration sha256 `{res['prereg_hash']}`. Seeds {seeds}, windows {res['windows']}; "
        f"the forks ran {res['fork_steps_seen']} steps.",
        "",
        "## Clauses",
        "",
        _row("clause", "holds", "numbers"),
        _row("---", "---", "---"),
    ]
    for k in ("code", "transfer", "reception"):
        nums = "; ".join(
            f"s{s}: {c.get('pair', '-')} excess {_f(c.get('excess'), 4)} bits, z {_f(c.get('z'), 1)}, "
            f"p {_p(c.get('p_family'))}"
            for s, c in zip(seeds, cl[k]["per_seed"], strict=True)
        )
        lines.append(_row(f"{k} (none arm)", cl[k]["passed"], nums))
    tc = res["transfer_contrast"]

    def contrast(c: str) -> str:
        means = [x["mean"] for x in tc[c]["per_seed"] if x["mean"] is not None]
        call = tc[c]["call"]
        return (
            f"none-{c}: mean {_f(np.mean(means) if means else None, 4)} bits, label {call['label']}, "
            f"p {_p(call['p'])}, Holm {_p(call.get('p_holm'))}"
        )

    lines.append(
        _row(
            "transfer contrast",
            cl["transfer_contrast"]["passed"],
            "; ".join(contrast(c) for c in ("deaf", "decoy")),
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
    lines += [
        "",
        "## Transfer contrast on the odd blocks (bits, 95% interval over blocks), per seed",
        "",
        _row("seed", "pair", "none - deaf", "none - decoy"),
        _row(*["---"] * 4),
    ]
    for i, s in enumerate(seeds):
        lines.append(
            _row(
                s,
                res["transfer_pairs"][str(s)],
                _ci(tc["deaf"]["per_seed"][i]),
                _ci(tc["decoy"]["per_seed"][i]),
            )
        )
    lines += [
        "",
        "## Fitness readouts (none minus control, per seed; relative to the control's mean)",
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
    lines += ["", "## Counterfactual decisions (audible rows; own control against each other control)", ""]
    for n, modes in res["counterfactual"].items():
        for mode, m in modes.items():
            lines.append(
                f"- {n} -> {mode}: |thrust| change {_f(m.get('d_thrust'), 4)}, attack flips "
                f"{_f(m.get('flip_attack'), 4)}, bond flips {_f(m.get('flip_bond'), 4)}, speech flips "
                f"{_f(m.get('flip_speak'), 4)}, call change {_f(m.get('d_voice'), 4)}"
            )
    lines += [
        "",
        "## Reading it",
        "",
        "- The shuffle null removes any link between a caller's situation and a hearer's action. The shared "
        "environment (neighbours are in similar places) survives conditioning on the hearer's own situation, "
        "so a transfer above the null is not yet evidence of a call; the contrast with `deaf` and `decoy`, "
        "which keep the environment and cut the calls, is.",
        "- Thresholds are provisional (DESIGN 44.8) and come from the frozen pre-registration; effect sizes "
        "are the excess over the null mean, in bits, with 95% t-intervals over blocks of one seed.",
    ]
    return "\n".join(lines) + "\n"
