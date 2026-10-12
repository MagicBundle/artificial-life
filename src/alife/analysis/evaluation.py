"""The ecosystem-director evaluation (``docs/DIRECTOR.md`` §9, results in ``docs/EVALUATION.md``).

Every arm is a run directory analysed from its fork step on (``start``): for arm A of seed 0 that is
``runs/64k_s0`` from step 120,000; for the other arms the fork itself. All the director's own vital signs are
recomputed here from ``metrics.jsonl`` with the director's code (``director.vitals.Tracker`` with the default
``RuleParams``), so the control arm, which ran without a director, is measured with the same yardstick as the
directed arms, and every arm's archive starts empty at the fork step, as a live director's does.
``check_against_log`` confirms the recomputation equals the vital signs a live director logged.

Per arm (``arm_summary``), over the whole span and over ``BLOCKS``:

- **coverage**: grid cells of the behaviour archive ever visited (``Archive.coverage``), and the cells
  outside the control arm's (``cells_outside_control``);
- **novelty**: the director's novelty at each rule boundary (k-NN distance to the arm's own archive), and
  **excursion**: the k-NN distance of each boundary's descriptor to every descriptor of the control arm of
  the same seed (``excursion``), which does not depend on the arm's own history;
- **persistent novelty** (§9: novelty still above the baseline 100k steps after an intervention), for each
  intervention step ``s`` (every applied entry but a revert) whose window ends in the run, two baselines
  (``persistent_novelty``):
  - *vs control* (primary): the arm's mean novelty over ``(s + lag, s + lag + span]`` minus the control arm's
    over the same steps: still above the undirected world. It is 0 for the control by construction;
  - *vs before*: the same window minus the arm's own mean over ``(s − base, s]``. The control has no
    interventions; it is evaluated at the intervention steps of every arm of its seed (``seed_anchors``),
    which gives the drift without a director;
- **turnover**: ``b_shift`` (RMS distance between the descriptor now and 100k steps earlier) and archive
  additions per 100k steps;
- **viability**: the lowest population (also over the control arm's mean), the share of rule windows at
  risk (the director's ``fall`` above 0.3, a fall of more than 30% within 50k steps) and the extinction
  rescues;
- **clades**: ``clades_alive``, ``top_clade_share`` and ``clade_entropy`` (all immigrants share one clade id,
  ``population.n_init``, so immigration adds at most one clade);
- **interventions** (``intervention_stats``): counts by action and source, distinct actions, the Shannon
  entropy of the action mix and the budget cost (``actions.MENU``), safety and discretionary apart;
- **LLM** (``llm_stats``): calls, outcomes, the JSON-validity rate (replies that parsed and passed the
  schema), the clamp rate (clamped parameters per accepted proposal), rejections and latency;
- **lineages** (``lineage_stats``, from the event logs): fork-founder lineages surviving at the end, the
  immigrant-descended share of the population, and generations gained since the fork;
- **evolutionary activity** (``activity_classes``): the Channon classes of ``alife activity RUN --start``
  (``docs/ACTIVITY.md``), read from ``RUN/ACTIVITY_DIR/summary.json`` when it exists.

Usage::

    python -m alife.analysis.evaluation --arm 0:A:runs/64k_s0:120000 --arm 0:B:runs/eval_s0_B ... \\
        --out docs/evaluation [--lineages] [--plots]
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from alife.director import actions as dactions
from alife.director import rules
from alife.director.archive import Archive
from alife.director.vitals import GRID_AXES, Tracker
from alife.runio import RunDir

DIRECTOR_LOG = "director.jsonl"
ACTIVITY_DIR = "activity_eval"  # alife activity RUN --out RUN/activity_eval (the evaluation's windows)
ACTIVITY_SPACES = ("cohort_h200k", "marker_0.01")  # report.PRIMARY_SPACES
JOURNAL = "journal.jsonl"
RULE_EVERY = 1_000
BLOCKS: tuple[tuple[int, int], ...] = ((120_000, 500_000), (500_000, 1_000_000), (1_000_000, 1_500_000))
PN_LAG, PN_SPAN, PN_BASE = 100_000, 50_000, 50_000
CRASH_FALL = 0.3  # rules.RuleParams.risk_fall: the director's extinction-risk fall


@dataclass(frozen=True)
class Arm:
    seed: int
    name: str
    run: Path
    start: int | None = None  # the fork step; default: the run's start_step

    @classmethod
    def parse(cls, text: str) -> Arm:
        """``SEED:NAME:RUN[:START]``."""
        parts = text.split(":")
        if len(parts) not in (3, 4):
            raise ValueError(f"expected SEED:NAME:RUN[:START], got {text!r}")
        start = int(parts[3]) if len(parts) == 4 else None
        return cls(int(parts[0]), parts[1], Path(parts[2]), start)

    def start_step(self) -> int:
        if self.start is not None:
            return self.start
        return int(RunDir.open(self.run).meta().get("start_step") or 0)


# --- the director's vital signs, recomputed -----------------------------------------------------------------


def boundaries(
    rows: Iterable[Mapping[str, Any]], start: int, rule_every: int = RULE_EVERY
) -> tuple[list[dict[str, Any]], Archive]:
    """One record per rule boundary after ``start``, folded exactly as a live director folds the rows
    (``Director._rebuild`` then ``observe``/``close``); ``K_est`` is not stored, so the risk flags that need
    it are left out. Returns the records and the final archive."""
    tr = Tracker(rules.RULE_PARAMS)
    out: list[dict[str, Any]] = []
    imm = 0
    for r in rows:
        if r["step"] <= start:
            continue
        tr.observe(r)
        imm += int(r.get("immigrants", 0) or 0)
        if r["step"] % rule_every:
            continue
        bd = tr.close(r["step"])
        v = tr.vitals(bd, 0.0, 1)
        w = bd.window
        out.append(
            {
                "step": bd.step,
                "n_alive": w.n_alive_last,
                "n_alive_min": w.n_alive_min,
                "births": w.births,
                "deaths": w.deaths,
                "immigrants": imm,
                "fall": v.fall,
                "novelty": v.novelty,
                "coverage": v.coverage,
                "added": bool(bd.obs.added) if bd.obs else False,
                "new_cell": bool(bd.obs.new_cell) if bd.obs else False,
                "b": list(bd.b) if bd.b is not None else None,
                "b_shift": v.b_shift,
                "new_cells_200k": v.new_cells_200k,
                "top_clade_share": w.top_clade_share,
                "clades_alive": w.clades_alive,
                "clade_entropy": w.clade_entropy,
                "weight_diversity": w.weight_diversity,
                "bodies_largest": w.bodies_largest,
                "frac_in_bodies_ge4": w.frac_in_bodies_ge4,
            }
        )
        imm = 0
    return out, tr.archive


def check_against_log(
    bds: Sequence[Mapping[str, Any]], entries: Iterable[Mapping[str, Any]]
) -> dict[str, Any]:
    """Compare the recomputed novelty, coverage, ``fall`` and ``b_shift`` with the ``vitals`` a live
    director logged in its entries (rounded to 4 places there)."""
    by_step = {b["step"]: b for b in bds}
    checked = mismatched = 0
    first: dict[str, Any] | None = None
    for e in entries:
        v, b = e.get("vitals"), by_step.get(e.get("vitals_step"))
        if not v or b is None:
            continue
        checked += 1
        for key in ("novelty", "coverage", "fall", "b_shift"):
            want, got = v.get(key), b[key]
            if want is None and got is None:
                continue
            if want is None or got is None or abs(round(got, 4) - want) > 1.5e-4:
                mismatched += 1
                first = first or {"step": e["vitals_step"], "key": key, "logged": want, "recomputed": got}
                break
    return {"entries_checked": checked, "mismatched": mismatched, "first_mismatch": first}


def _series(bds: Sequence[Mapping[str, Any]], key: str) -> tuple[np.ndarray, np.ndarray]:
    t = np.array([b["step"] for b in bds], np.int64)
    y = np.array([np.nan if b[key] is None else float(b[key]) for b in bds])
    return t, y


def _mean(y: np.ndarray) -> float | None:
    ok = np.isfinite(y)
    return float(y[ok].mean()) if ok.any() else None


def excursion(
    bds: Sequence[Mapping[str, Any]], reference: Sequence[Sequence[float]], k: int = 5
) -> np.ndarray:
    """Per boundary: the mean RMS distance of its descriptor to the ``k`` nearest reference descriptors (the
    archive's distance, without the archive's membership rule), NaN where there is no descriptor."""
    ref = np.asarray(reference, np.float64)
    out = np.full(len(bds), np.nan)
    if ref.size == 0:
        return out
    for i, b in enumerate(bds):
        if b["b"] is None:
            continue
        d = np.linalg.norm(ref - np.asarray(b["b"]), axis=1) / math.sqrt(ref.shape[1])
        out[i] = float(np.sort(d)[: min(k, d.size)].mean())
    return out


def persistent_novelty(
    t: np.ndarray,
    novelty: np.ndarray,
    anchors: Iterable[int],
    control: tuple[np.ndarray, np.ndarray] | None = None,
    lag: int = PN_LAG,
    span: int = PN_SPAN,
    base: int = PN_BASE,
) -> list[dict[str, Any]]:
    """For each anchor step ``s`` whose after-window ends by the last boundary: the mean novelty before
    (``(s − base, s]``) and ``lag`` steps after (``(s + lag, s + lag + span]``), ``delta`` = after − before,
    and, with ``control`` = (steps, novelty) of the control arm, its mean over the after-window and
    ``vs_control`` = after − that."""
    out = []
    for s in sorted(set(anchors)):
        if s + lag + span > t[-1]:
            continue
        before = _mean(novelty[(t > s - base) & (t <= s)])
        after = _mean(novelty[(t > s + lag) & (t <= s + lag + span)])
        if before is None or after is None:
            continue
        row = {"step": int(s), "before": before, "after": after, "delta": after - before}
        if control is not None:
            ct, cn = control
            c_after = _mean(cn[(ct > s + lag) & (ct <= s + lag + span)])
            row["control_after"] = c_after
            row["vs_control"] = None if c_after is None else after - c_after
        out.append(row)
    return out


def _pn_summary(rows: Sequence[Mapping[str, Any]], key: str = "delta") -> dict[str, Any]:
    d = np.array([r[key] for r in rows if r.get(key) is not None], np.float64)
    if d.size == 0:
        return {"anchors": 0, "mean": None, "frac_above": None}
    return {"anchors": int(d.size), "mean": float(d.mean()), "frac_above": float((d > 0).mean())}


# --- interventions and the LLM tier -------------------------------------------------------------------------


def log_entries(run: Path) -> list[dict[str, Any]]:
    path = run / DIRECTOR_LOG
    if not path.is_file():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [r for r in rows if r.get("kind") != "header"]


def is_safety(e: Mapping[str, Any]) -> bool:
    return e.get("rule") in rules.SAFETY_RULES


def anchor_steps(entries: Iterable[Mapping[str, Any]], *, discretionary_only: bool = False) -> list[int]:
    """The steps at which the arm applied anything but a revert (optionally only discretionary moves)."""
    return sorted(
        {
            int(e["step_applied"])
            for e in entries
            if e["source"] != "revert" and not (discretionary_only and is_safety(e))
        }
    )


def _entropy(counts: Iterable[int]) -> float:
    c = np.array([x for x in counts if x > 0], np.float64)
    if c.size == 0:
        return 0.0
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


def intervention_stats(entries: Sequence[Mapping[str, Any]], span_steps: int) -> dict[str, Any]:
    """Counts by action, source and rule; distinct actions, the action entropy (bits) and the budget cost,
    for all applied entries and for the discretionary ones (reverts are not interventions)."""
    applied = [e for e in entries if e["source"] != "revert"]
    disc = [e for e in applied if not is_safety(e)]

    def part(es: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        by_action = Counter(e["action"] for e in es)
        cost = sum(dactions.MENU[e["action"]].cost for e in es)
        return {
            "count": len(es),
            "steps": len({e["step_applied"] for e in es}),
            "by_action": dict(sorted(by_action.items())),
            "distinct_actions": len(by_action),
            "entropy_bits": _entropy(by_action.values()),
            "cost": cost,
            "cost_per_100k": cost * 100_000 / span_steps if span_steps else None,
        }

    return {
        "all": part(applied),
        "discretionary": part(disc),
        "safety": part([e for e in applied if is_safety(e)]),
        "by_source": dict(sorted(Counter(e["source"] for e in applied).items())),
        "by_rule": dict(sorted(Counter(str(e.get("rule")) for e in applied).items())),
        "rescues": sum(1 for e in applied if e.get("rule") == rules.EXTINCTION_RESCUE),
        "rescue_steps": len({e["step_applied"] for e in applied if e.get("rule") == rules.EXTINCTION_RESCUE}),
        "reverts": sum(1 for e in entries if e["source"] == "revert"),
    }


def llm_stats(journal: Iterable[Mapping[str, Any]]) -> dict[str, Any] | None:
    """The LLM tier's record (``journal.jsonl`` ``llm_call`` records), or None for an arm without it.

    - validity: replies that arrived, parsed and passed the schema (outcome ``ok``, ``none`` or
      ``rejected``) over all calls; ``failed`` (transport) and ``invalid`` (parse or schema) are not valid;
    - clamp rate: clamped parameters per accepted proposal, and the share of calls with any clamp."""
    calls = [j for j in journal if j.get("kind") == "llm_call"]
    if not calls:
        return None
    outcomes = Counter(c["outcome"] for c in calls)
    valid = sum(outcomes[k] for k in ("ok", "none", "rejected"))
    accepted = sum(len(c.get("accepted") or []) for c in calls)
    clamps = sum(len(c.get("clamps") or []) for c in calls)
    lat = np.array([c["latency_s"] for c in calls if c.get("latency_s") is not None], np.float64)
    reasons = Counter(r.get("why", "") for c in calls for r in c.get("rejected") or [])
    return {
        "calls": len(calls),
        "outcomes": dict(sorted(outcomes.items())),
        "valid": valid,
        "validity_rate": valid / len(calls),
        "accepted_proposals": accepted,
        "rejected_proposals": sum(reasons.values()),
        "rejection_reasons": dict(reasons.most_common()),
        "clamped_params": clamps,
        "clamp_rate": clamps / accepted if accepted else 0.0,
        "calls_with_clamp": sum(1 for c in calls if c.get("clamps")),
        "fallbacks": sum(1 for c in calls if c.get("fallback")),
        "latency_s": {
            "median": float(np.median(lat)),
            "p90": float(np.percentile(lat, 90)),
            "max": float(lat.max()),
            "min": float(lat.min()),
        }
        if lat.size
        else None,
        "prompt_tokens_median": float(
            np.median([c["usage"]["prompt_tokens"] for c in calls if c.get("usage")] or [np.nan])
        ),
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def journal_sample(journal: Iterable[Mapping[str, Any]], n: int = 10, seed: int = 0) -> list[dict[str, Any]]:
    """``n`` LLM journal entries drawn uniformly without replacement (seeded), in step order, for the
    legibility rating of §9: the step, the model's journal text and what it chose."""
    calls = [j for j in journal if j.get("kind") == "llm_call" and j.get("journal")]
    rng = np.random.default_rng(seed)
    pick = sorted(rng.choice(len(calls), size=min(n, len(calls)), replace=False)) if calls else []
    return [
        {
            "step": calls[i]["step"],
            "journal": calls[i]["journal"],
            "accepted": [f"{a['action']}({_params(a['params'])})" for a in calls[i].get("accepted") or []],
            "outcome": calls[i]["outcome"],
        }
        for i in pick
    ]


def _params(p: Mapping[str, Any]) -> str:
    return ", ".join(f"{k}={round(v, 3) if isinstance(v, float) else v}" for k, v in p.items())


# --- lineages from the event logs ---------------------------------------------------------------------------


def lineage_stats(run: Path, start: int, tail: int = 50_000, head: int = 10_000) -> dict[str, Any]:
    """From the phylogeny since ``start`` (``phylo.load_tree``) and the births' ``generation`` column:
    fork-founder lineages alive at the end, immigrant-descended share, and the mean generation of the
    founder-descended births in the first ``head`` and the last ``tail`` steps."""
    from alife.analysis import phylo  # local: numpy-heavy; not needed for the rest of the module

    tree, rep = phylo.load_tree(run, start=start, check_checkpoint=False)
    n_f = int(np.count_nonzero(tree.birth < 0))
    root = phylo.nearest_root(tree.parent, tree.parent < 0)
    alive = tree.death >= tree.t_end
    founder_line = root < n_f
    gen = _birth_generations(run, start)
    if gen.size != tree.n - n_f:
        raise ValueError(f"{run}: {gen.size} generation records for {tree.n - n_f} births")
    born = np.arange(n_f, tree.n)
    b = tree.birth[n_f:]
    fl = founder_line[n_f:] & (tree.parent[n_f:] >= 0)

    def mean_gen(m: np.ndarray) -> float | None:
        return float(gen[m].mean()) if m.any() else None

    g_head = mean_gen(fl & (b < head))
    g_tail = mean_gen(fl & (b >= tree.t_end - tail))
    return {
        "founders": n_f,
        "agents": tree.n,
        "alive_end": int(alive.sum()),
        "founder_lineages_alive": int(np.unique(root[alive & founder_line]).size),
        "immigrants": int(np.count_nonzero((tree.parent < 0) & (tree.birth >= 0))),
        "immigrant_descended_frac_end": float((alive & ~founder_line).sum() / max(alive.sum(), 1)),
        "generation_head": g_head,
        "generation_tail": g_tail,
        "generations_gained": None if g_head is None or g_tail is None else g_tail - g_head,
        "births": int(born.size),
        "coverage_ok": bool(rep["chunks_tile"] and rep["births_match_metrics"]),
    }


def _birth_generations(run: Path, start: int) -> np.ndarray:
    rd = RunDir.open(run)
    out = []
    for _, s1, path in rd.event_files("births"):
        if s1 <= start:
            continue
        with np.load(path) as z:
            out.append(z["generation"][z["step"] >= start].astype(np.int32))
    return np.concatenate(out) if out else np.zeros(0, np.int32)


def activity_classes(path: Path, spaces: Sequence[str] = ACTIVITY_SPACES) -> dict[str, Any] | None:
    """Per primary space and window of an ``alife activity`` summary: the Channon class and its sweep counts,
    the new normalised activity of the run, the neutral shadow and the heritability null (per component per
    grid interval), and the growth ratios of ``A^N_cum`` and the median."""
    if not path.is_file():
        return None
    s = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, Any] = {"t0": s.get("t0"), "spaces": {}}
    for space in spaces:
        cells = s["spaces"].get(space, {}).get("windows", {})
        out["spaces"][space] = {
            w: None
            if not c.get("valid")
            else {
                "class": c["channon"]["class"],
                "sweep": c["sensitivity"]["channon"],
                "a_new_real": c["channon"]["a_new_real"],
                "a_new_neutral": c["channon"]["a_new_neutral"],
                "a_new_permuted": c["channon"]["a_new_permuted"],
                "rho_cum": c["channon"]["rho_cum"],
                "rho_med": c["channon"]["rho_med"],
                "bedau_class": c["bedau"]["class"],
            }
            for w, c in cells.items()
        }
    return out


# --- one arm ------------------------------------------------------------------------------------------------


def _block_means(t: np.ndarray, y: np.ndarray, blocks: Sequence[tuple[int, int]]) -> dict[str, float | None]:
    return {f"{a // 1000}k-{b // 1000}k": _mean(y[(t > a) & (t <= b)]) for a, b in blocks}


def _at(t: np.ndarray, y: np.ndarray, step: int) -> float | None:
    i = np.searchsorted(t, step, side="right") - 1
    return None if i < 0 or not np.isfinite(y[i]) else float(y[i])


def arm_summary(
    arm: Arm,
    bds: Sequence[Mapping[str, Any]],
    archive: Archive,
    control: tuple[Sequence[Mapping[str, Any]], Archive] | None,
    seed_anchors: Sequence[int],
    blocks: Sequence[tuple[int, int]] = BLOCKS,
) -> dict[str, Any]:
    """Every metric of the module docstring for one arm (``control``: the seed's control arm, None for the
    control itself; ``seed_anchors``: the intervention steps of every arm of the seed)."""
    run = arm.run
    start = arm.start_step()
    entries = log_entries(run)
    t, nov = _series(bds, "novelty")
    end = int(t[-1])
    out: dict[str, Any] = {"seed": arm.seed, "arm": arm.name, "run": str(run), "start": start, "end": end}
    meta = RunDir.open(run).meta()
    out["git_rev"] = meta.get("git_rev")
    _, cov = _series(bds, "coverage")
    out["coverage"] = {
        "final": int(archive.coverage),
        "at": {f"{s // 1000}k": _at(t, cov, s) for s in (250_000, 500_000, 1_000_000, end)},
    }
    if control is not None:
        out["coverage"]["cells_outside_control"] = len(set(archive.cells) - set(control[1].cells))
        ref = [b["b"] for b in control[0] if b["b"] is not None]
        exc = excursion(bds, ref)
        out["excursion"] = {"mean": _mean(exc), "blocks": _block_means(t, exc, blocks)}
    else:
        out["coverage"]["cells_outside_control"] = 0
    out["novelty"] = {"mean": _mean(nov), "blocks": _block_means(t, nov, blocks)}
    own = anchor_steps(entries)
    ctrl_nov = _series(control[0], "novelty") if control is not None else None
    pn_own = persistent_novelty(t, nov, own, ctrl_nov)
    pn_disc = persistent_novelty(t, nov, anchor_steps(entries, discretionary_only=True), ctrl_nov)
    none = {"anchors": 0, "mean": None, "frac_above": None}
    out["persistent_novelty"] = {
        # the control's own value is 0 by construction
        "vs_control": _pn_summary(pn_own, "vs_control") if control is not None else {**none, "mean": 0.0},
        "vs_before": _pn_summary(pn_own),
        "discretionary_vs_control": _pn_summary(pn_disc, "vs_control") if control is not None else none,
        "discretionary_vs_before": _pn_summary(pn_disc),
        "seed_anchors_vs_before": _pn_summary(persistent_novelty(t, nov, seed_anchors)),
        "rows": pn_own,
    }
    _, shift = _series(bds, "b_shift")
    added = np.array([b["added"] for b in bds])
    span = end - start
    out["turnover"] = {
        "b_shift_mean": _mean(shift),
        "b_shift_blocks": _block_means(t, shift, blocks),
        "archive_size": len(archive.entries),
        "archive_additions_per_100k": float(added.sum() * 100_000 / span),
    }
    _, pop = _series(bds, "n_alive")
    _, low = _series(bds, "n_alive_min")
    _, fall = _series(bds, "fall")
    risk = np.isfinite(fall) & (fall > CRASH_FALL)
    ctrl_pop = _mean(_series(control[0], "n_alive")[1]) if control is not None else _mean(pop)
    out["viability"] = {
        "pop_mean": _mean(pop),
        "pop_blocks": _block_means(t, pop, blocks),
        "pop_min": float(np.nanmin(low)),
        "pop_min_step": int(t[np.nanargmin(low)]),
        "pop_end": _at(t, pop, end),
        "pop_min_over_control_mean": float(np.nanmin(low)) / ctrl_pop if ctrl_pop else None,
        "at_risk_frac": float(risk.mean()),
        "extinct": bool(np.nanmin(low) == 0),
        "immigrants": int(sum(b["immigrants"] for b in bds)),
    }
    clades: dict[str, Any] = {}
    for key in ("clades_alive", "top_clade_share", "clade_entropy"):
        _, y = _series(bds, key)
        clades[key] = {
            "start": _at(t, y, start + RULE_EVERY),
            "end": _at(t, y, end),
            "blocks": _block_means(t, y, blocks),
        }
    out["clades"] = clades
    out["interventions"] = intervention_stats(entries, span)
    out["llm"] = llm_stats(read_jsonl(run / JOURNAL))
    out["log_check"] = check_against_log(bds, entries) if entries else None
    out["activity"] = activity_classes(run / ACTIVITY_DIR / "summary.json")
    return out


# --- the evaluation -----------------------------------------------------------------------------------------


def evaluate(
    arms: Sequence[Arm], control: str = "A", lineages: bool = False, log: Any = print
) -> tuple[dict[str, Any], dict[tuple[int, str], list[dict[str, Any]]]]:
    """Summaries of every arm (``{"arms": [...], "seeds": {...}}``) and the per-boundary records."""
    series: dict[tuple[int, str], list[dict[str, Any]]] = {}
    archives: dict[tuple[int, str], Archive] = {}
    for a in arms:
        log(f"folding {a.seed}:{a.name} ({a.run})")
        bds, arc = boundaries(RunDir.open(a.run).metrics_rows(), a.start_step())
        series[(a.seed, a.name)], archives[(a.seed, a.name)] = bds, arc
    out: dict[str, Any] = {"arms": [], "settings": _settings(control)}
    for a in arms:
        ctrl = (a.seed, control)
        has_ctrl = ctrl in series and a.name != control
        seed_anchors = sorted({s for b in arms if b.seed == a.seed for s in anchor_steps(log_entries(b.run))})
        summ = arm_summary(
            a,
            series[(a.seed, a.name)],
            archives[(a.seed, a.name)],
            (series[ctrl], archives[ctrl]) if has_ctrl else None,
            seed_anchors,
        )
        if lineages:
            log(f"lineages of {a.seed}:{a.name}")
            summ["lineages"] = lineage_stats(a.run, a.start_step())
        out["arms"].append(summ)
    return out, series


def _settings(control: str) -> dict[str, Any]:
    p = rules.RULE_PARAMS
    return {
        "control_arm": control,
        "rule_every": RULE_EVERY,
        "blocks": [list(b) for b in BLOCKS],
        "persistent_novelty": {"lag": PN_LAG, "span": PN_SPAN, "base": PN_BASE},
        "crash_fall": CRASH_FALL,
        "archive": {
            "k": p.novelty_k,
            "exclude": p.novelty_exclude,
            "rho": p.archive_rho,
            "bins": p.grid_bins,
        },
        "grid_axes": list(GRID_AXES),
    }


def write_series(path: Path, series: Mapping[tuple[int, str], Sequence[Mapping[str, Any]]]) -> None:
    """The per-boundary records as compressed arrays ``s<seed>_<arm>/<key>``."""
    keys = (
        "step",
        "n_alive",
        "n_alive_min",
        "novelty",
        "coverage",
        "b_shift",
        "top_clade_share",
        "clades_alive",
        "fall",
    )
    flat = {}
    for (seed, name), bds in series.items():
        for k in keys:
            flat[f"s{seed}_{name}/{k}"] = np.array(
                [np.nan if b[k] is None else b[k] for b in bds], np.float64
            )
    np.savez_compressed(path, **flat)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m alife.analysis.evaluation", description=__doc__.split("\n")[0]
    )
    ap.add_argument("--arm", action="append", required=True, metavar="SEED:NAME:RUN[:START]")
    ap.add_argument("--control", default="A")
    ap.add_argument("--out", required=True, metavar="DIR")
    ap.add_argument(
        "--lineages", action="store_true", help="also read the event logs (slow, about 1 min per arm)"
    )
    ap.add_argument("--plots", action="store_true", help="draw the figures (needs the analysis extra)")
    ap.add_argument(
        "--journal-sample",
        action="append",
        default=[],
        metavar="SEED:NAME",
        help="sample 10 LLM journal entries",
    )
    a = ap.parse_args(argv)
    arms = [Arm.parse(x) for x in a.arm]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summary, series = evaluate(arms, a.control, a.lineages)
    (out / "evaluation.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    write_series(out / "series.npz", series)
    for spec in a.journal_sample:
        seed, name = spec.split(":")
        arm = next(x for x in arms if x.seed == int(seed) and x.name == name)
        sample = journal_sample(read_jsonl(arm.run / JOURNAL))
        path = out / f"journal_sample_s{seed}_{name}.json"
        path.write_text(json.dumps(sample, indent=1) + "\n", encoding="utf-8")
    if a.plots:
        from alife.analysis import evaluation_plots

        for p in evaluation_plots.plot_all(summary, series, arms, out):
            print(f"wrote {p}")
    print(f"wrote {out / 'evaluation.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
