"""The phylogeny of a run, rebuilt from its event logs (DESIGN §12.8), and the null-model shadows.

Agents are indexed in uid order: the founders first, then every later agent (child or immigrant), which is
logged in ``births_*.npz`` with a uid assigned in birth order, so a parent's index is always smaller than its
child's. In a run from step 0 the founders are the ``n_init`` agents with uids ``0 … n_init − 1``; in a fork
(``run --init-from``) they are the agents alive in its first checkpoint, which keep their source uids; with
``start`` (``load_tree``), the agents alive at that step. Steps are counted from that start, ``t0``: a step
index ``u`` below is the run's step ``t0 + u``. A ``Tree`` holds three int32 arrays over all agents:

- ``parent``: the parent's index, −1 for a root (a founder or an immigrant);
- ``birth``: the index of the step during which the agent was born, −1 for the founders (their true age is
  not used: a founder may be a parent from step 0 on, ``eligible_from``);
- ``death``: the index of the step during which it died, ``t_end`` for agents alive at the end.

Grid times are ``state.step`` values: an agent is alive at grid time ``s`` iff ``birth < s ≤ death``, and the
births counted up to ``s`` are those with ``birth < s`` (the convention of the event chunks, ``runio``). An
agent can be a parent during steps ``birth + gap ≤ u < death``; on ``runs/64k_s0`` the smallest parent–child
gap is 21 steps (``repro.min_age`` = 20 after a founder's birth at −1) and every parent outlives its child's
birth step.

Shadows (same agents, same birth steps, same per-step birth and death counts; only ``parent`` and, for the
neutral shadow, ``death`` change):

- ``neutral_shadow``: Bedau's neutral shadow (Bedau, Snyder & Packard 1998): each step the logged number of
  deaths removes uniformly random alive agents, then each logged birth picks a uniformly random eligible
  parent (alive, age ≥ gap). Selection is random, so it has no adaptation by construction.
- ``permuted_shadow``: a heritability null. The children born in a step are reassigned at random among that
  step's real parents (a permutation within the step). Every agent keeps its own life span, its birth steps
  as a parent and its offspring count, so the life history and the individual variance in fecundity are
  exactly the run's; what is destroyed is the correlation of reproductive success along lineages.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from alife.runio import RunDir


class Tree(NamedTuple):
    parent: np.ndarray  # int32 (n,), −1 for roots (founders and immigrants)
    birth: np.ndarray  # int32 (n,), step index of birth, −1 for the founders
    death: np.ndarray  # int32 (n,), step index of death, t_end if alive at the end
    t_end: int  # state.step at the end of the record

    @property
    def n(self) -> int:
        return int(self.parent.size)


def validate_tree(tree: Tree) -> None:
    """Structural checks; ``ValueError`` on the first violation."""
    n = tree.n
    if not (tree.birth.size == tree.death.size == n):
        raise ValueError("parent, birth and death differ in length")
    if n and np.any(np.diff(tree.birth) < 0):
        raise ValueError("agents are not in birth order")
    if np.any(tree.death <= tree.birth) or np.any(tree.death > tree.t_end):
        raise ValueError("a death is not after its birth or is after t_end")
    child = np.flatnonzero(tree.parent >= 0)
    p = tree.parent[child]
    if np.any(p >= child):
        raise ValueError("a parent's index is not smaller than its child's")
    if np.any(tree.death[p] < tree.birth[child]):
        raise ValueError("a parent died before its child's birth step")


def min_gap(tree: Tree, *, founder_ages: bool = True) -> int:
    """The smallest parent–child birth-step gap (≥ 1); the eligibility age of the shadows and exposures.

    With ``founder_ages=False`` (a fork, or a tree from a ``start`` step, whose founders were born at unknown
    steps before the start) the founders' children are left out when other children exist, since their gap
    says nothing.
    """
    child = np.flatnonzero(tree.parent >= 0)
    if child.size == 0:
        return 1
    gaps = tree.birth[child] - tree.birth[tree.parent[child]]
    if not founder_ages:
        aged = tree.birth[tree.parent[child]] >= 0
        gaps = gaps[aged] if aged.any() else gaps
    return max(int(gaps.min()), 1)


def eligible_from(birth: np.ndarray, gap: int) -> np.ndarray:
    """The first step index (int64) in which each agent may be a parent: ``birth + gap``, and 0 for the
    founders (``birth < 0``), which may be of any age at the start."""
    b = birth.astype(np.int64)
    return np.where(b < 0, 0, b + gap)


# --- loading ----------------------------------------------------------------------------------------------


def _columns(path: Path, names: tuple[str, ...]) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k] for k in names}


def _chunks(
    rd: RunDir, kind: str, names: tuple[str, ...]
) -> Iterator[tuple[int, int, dict[str, np.ndarray]]]:
    for s0, s1, path in rd.event_files(kind):
        yield s0, s1, _columns(path, names)


def _alive_uids(path: Path) -> np.ndarray:
    """The sorted uids (int64) of the agents alive in a checkpoint."""
    import mlx.core as mx  # local: the analysis is numpy-only except for reading a checkpoint

    from alife.state import to_numpy

    arrs = mx.load(str(path), format="safetensors")
    alive = to_numpy(arrs["alive"]).astype(bool)
    return np.sort(to_numpy(arrs["uid"]).astype(np.int64)[alive])


def _index_of(uid_all: np.ndarray, u: np.ndarray, what: str) -> np.ndarray:
    """Indices (int64) of the uids ``u`` in the strictly increasing ``uid_all``; ``ValueError`` if any is
    missing."""
    pos = np.searchsorted(uid_all, u)
    ok = pos < uid_all.size
    ok[ok] = uid_all[pos[ok]] == u[ok]
    if not ok.all():
        raise ValueError(f"{int((~ok).sum())} {what} uids are not agents of the run (first {u[~ok][0]})")
    return pos.astype(np.int64)


def load_tree(
    run: Path | str, *, start: int | None = None, check_checkpoint: bool = True
) -> tuple[Tree, dict[str, Any]]:
    """The run's phylogeny and an event-log coverage report.

    The chunks are read one at a time and only the needed columns are kept (int32, about 0.5 GB for the 43M
    agents of ``64k_s0``). The report compares the logged events with the ``metrics.jsonl`` counters
    (``births``, ``immigrants``, the three death causes, ``deaths_unlogged``), checks that the chunks tile
    ``[origin, end)``, and, if a checkpoint at the end exists, that its alive uids are exactly the agents
    without a death record. Agents that died without a record (``deaths_unlogged``) cannot be told apart from
    survivors without that checkpoint; when it exists they get the lower bound ``death = max(birth, last
    child birth) + 1`` and are counted in the report.

    ``origin`` is the run's first step (``meta.json`` ``start_step``: 0, or the fork step of a fork, whose
    founders are read from its checkpoint at that step). ``start`` (default: ``origin``) restricts the tree to
    the agents alive at that step, which become the founders, and their descendants (``restrict``). The tree
    counts steps from ``start``; the report's ``t0`` is ``start``.
    """
    rd = RunDir.open(Path(run))
    cfg = rd.config()
    origin = int(rd.meta().get("start_step") or 0)
    start = origin if start is None else int(start)
    if start < origin:
        raise ValueError(f"start {start} is before the run's first step {origin}")
    if origin == 0:
        founders = np.arange(cfg.population.n_init, dtype=np.int64)
    else:
        founders = _alive_uids(rd.checkpoint_at(origin))
    n_init = int(founders.size)
    spans: dict[str, list[tuple[int, int]]] = {"births": [], "deaths": []}

    steps, uids, parents = [], [], []
    for s0, s1, c in _chunks(rd, "births", ("step", "uid", "parent_uid")):
        spans["births"].append((s0, s1))
        steps.append(c["step"].astype(np.int64) - origin)
        uids.append(c["uid"].astype(np.int64))
        parents.append(c["parent_uid"].astype(np.int64))
    b_uid = np.concatenate(uids) if uids else np.zeros(0, np.int64)
    n = n_init + b_uid.size
    if origin == 0:
        if not np.array_equal(b_uid, np.arange(n_init, n, dtype=np.int64)):
            raise ValueError("birth uids are not contiguous from n_init in log order")
    elif np.any(np.diff(b_uid) <= 0) or (b_uid.size and n_init and b_uid[0] <= founders[-1]):
        raise ValueError("birth uids are not increasing in log order after the founders' uids")
    uid_all = np.concatenate([founders, b_uid])
    del uids, b_uid
    parent = np.full(n, -1, np.int32)
    birth = np.full(n, -1, np.int32)
    if steps:
        pu = np.concatenate(parents)
        known = pu >= 0  # −1 (immigrant) stays −1
        par = np.full(pu.size, -1, np.int64)
        par[known] = _index_of(uid_all, pu[known], "parent")
        parent[n_init:] = par.astype(np.int32)
        birth[n_init:] = np.concatenate(steps).astype(np.int32)
    del steps, parents

    t_end = max((s1 for _, s1 in spans["births"]), default=origin) - origin
    death = np.full(n, t_end, np.int32)
    has_death = np.zeros(n, bool)
    n_death_records = 0
    for s0, s1, c in _chunks(rd, "deaths", ("step", "uid")):
        spans["deaths"].append((s0, s1))
        u = _index_of(uid_all, c["uid"].astype(np.int64), "dead")
        death[u] = c["step"].astype(np.int64) - origin
        has_death[u] = True
        n_death_records += u.size

    def tiles(sp: list[tuple[int, int]]) -> bool:
        return bool(sp) and sp[0][0] == origin and all(a[1] == b[0] for a, b in itertools.pairwise(sp))

    rows = rd.metrics_rows()
    totals = {
        k: int(sum(r.get(k, 0) for r in rows))
        for k in ("births", "immigrants", "deaths_starved", "deaths_killed", "deaths_aged", "deaths_unlogged")
    }
    deaths_catastrophe = int(sum(r.get("deaths_catastrophe", 0) for r in rows))
    n_imm = int(np.count_nonzero((parent < 0) & (birth >= 0)))
    report: dict[str, Any] = {
        "run": str(rd.path),
        "origin": origin,
        "t0": start,
        "t_end": t_end + origin,
        "n_init": n_init,
        "agents": n,
        "births_logged": int(n - n_init - n_imm),
        "immigrants_logged": n_imm,
        "deaths_logged": n_death_records,
        "metrics_totals": totals,
        "metrics_last_step": rows[-1]["step"] if rows else None,
        "chunks_tile": tiles(spans["births"]) and tiles(spans["deaths"]),
        "chunk_spans_equal": spans["births"] == spans["deaths"],
        "duplicate_death_records": int(n_death_records - np.count_nonzero(has_death)),
    }
    if deaths_catastrophe:
        report["metrics_totals"]["deaths_catastrophe"] = deaths_catastrophe
    deaths_metric = (
        totals["deaths_starved"] + totals["deaths_killed"] + totals["deaths_aged"] + deaths_catastrophe
    )
    report["births_match_metrics"] = (
        report["births_logged"] + n_imm == totals["births"] + totals["immigrants"]
    )
    report["deaths_match_metrics"] = n_death_records + totals["deaths_unlogged"] == deaths_metric

    unlogged_found = 0
    report["checkpoint_check"] = None
    if check_checkpoint:
        try:
            ck = rd.checkpoint_at(t_end + origin)
        except FileNotFoundError:
            ck = None
        if ck is not None:
            alive_uid = _alive_uids(ck)
            pos = np.searchsorted(uid_all, alive_uid)
            inside = pos < n
            inside[inside] = uid_all[pos[inside]] == alive_uid[inside]
            alive_idx = pos[inside]
            no_record = np.flatnonzero(~has_death)
            missing = np.setdiff1d(no_record, alive_idx)  # dead per checkpoint, no death record
            unknown_alive = np.setdiff1d(alive_idx, no_record)
            unlogged_found = int(missing.size)
            if missing.size:
                last_child = np.full(n, -1, np.int32)
                kids = np.flatnonzero(parent >= 0)
                np.maximum.at(last_child, parent[kids], birth[kids])
                death[missing] = np.minimum(np.maximum(birth[missing], last_child[missing]) + 1, t_end - 1)
            report["checkpoint_check"] = {
                "checkpoint": ck.name,
                "alive": int(alive_uid.size),
                "alive_without_death_record": int(no_record.size),
                "dead_without_record": unlogged_found,
                "alive_with_death_record": int(unknown_alive.size),
                "alive_not_in_log": int((~inside).sum()),
                "ok": unlogged_found == 0 and unknown_alive.size == 0 and bool(inside.all()),
            }
    report["dead_without_record_patched"] = unlogged_found
    tree = Tree(parent=parent, birth=birth, death=death, t_end=t_end)
    validate_tree(tree)
    if start > origin:
        tree = restrict(tree, start - origin)
        validate_tree(tree)
        report["founders_at_t0"] = int(np.count_nonzero(tree.birth < 0))
    report["min_parent_child_gap"] = min_gap(tree, founder_ages=start == 0)
    return tree, report


def restrict(tree: Tree, s: int) -> Tree:
    """The tree from step index ``s`` on: the agents alive at grid time ``s`` (``birth < s ≤ death``) become
    the founders (parent −1, birth −1) and keep their descendants; earlier agents are dropped and every step
    index is shifted by ``−s``.

    A parent of an agent born at or after ``s`` was alive at that birth step, so it is alive at ``s`` or born
    after it: no kept agent loses its parent.
    """
    if not 0 < s < tree.t_end:
        raise ValueError(f"restrict step {s} not in (0, t_end = {tree.t_end})")
    root = (tree.birth < s) & (tree.death >= s)
    keep = np.flatnonzero(root | (tree.birth >= s))
    new_of = np.full(tree.n, -1, np.int64)
    new_of[keep] = np.arange(keep.size)
    is_root = root[keep]
    par = tree.parent[keep].astype(np.int64)
    mapped = np.where(par >= 0, new_of[np.maximum(par, 0)], -1)
    if np.any(~is_root & (par >= 0) & (mapped < 0)):
        raise ValueError("an agent's parent is neither alive at the restrict step nor born after it")
    return Tree(
        parent=np.where(is_root, -1, mapped).astype(np.int32),
        birth=np.where(is_root, -1, tree.birth[keep].astype(np.int64) - s).astype(np.int32),
        death=(tree.death[keep].astype(np.int64) - s).astype(np.int32),
        t_end=tree.t_end - s,
    )


def coverage_json(report: dict[str, Any]) -> str:
    return json.dumps(report, indent=2, sort_keys=True)


# --- lineage labels ---------------------------------------------------------------------------------------


def nearest_root(parent: np.ndarray, is_root: np.ndarray, start: int = 0) -> np.ndarray:
    """For agents ``start … n−1``: the nearest ancestor-or-self with ``is_root`` (pointer jumping).

    Every agent below ``start`` must be a root, as must every agent with ``parent == −1``. Returns an int32
    array of length ``n − start`` of global indices. O(n log depth).
    """
    n = parent.size
    idx = np.arange(start, n, dtype=np.int32)
    root_sub = is_root[start:]
    if np.any(~root_sub & (parent[start:] < 0)):
        raise ValueError("an agent without a parent is not marked as a root")
    if start and not bool(is_root[:start].all()):
        raise ValueError("every agent below start must be a root")
    ptr = np.where(root_sub, idx, parent[start:]).astype(np.int32)
    active = np.flatnonzero(~is_root[ptr])
    while active.size:
        p = ptr[active]  # ≥ start, since every agent below start is a root
        ptr[active] = ptr[p - start]
        active = active[~is_root[ptr[active]]]
    return ptr


def splitmix64(x: np.ndarray, seed: int) -> np.ndarray:
    """A stateless 64-bit hash (SplitMix64 finaliser) of ``x ^ seed``; uint64, wrapping arithmetic."""
    z = np.asarray(x, dtype=np.uint64) ^ np.uint64(seed & 0xFFFFFFFFFFFFFFFF)
    z = z + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def marker_roots(tree: Tree, rate: float, seed: int = 0) -> np.ndarray:
    """Roots of the marker components: every root of the tree, plus each birth with ``hash(uid) < rate``.

    The founding events are a fixed property of the agent index, independent of the tree, so a shadow with
    the same agents has exactly the same founding events.
    """
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"marker rate {rate} not in [0, 1]")
    u = splitmix64(np.arange(tree.n, dtype=np.uint64), seed)
    cut = np.uint64(min(int(rate * 2.0**64), 2**64 - 1))
    hit = u < cut if rate < 1.0 else np.ones(tree.n, bool)
    return (tree.parent < 0) | hit


# --- shadows ----------------------------------------------------------------------------------------------


def _step_ranges(birth: np.ndarray, t_end: int) -> np.ndarray:
    """``lo[t] … lo[t+1]−1`` are the agents born during step t (agents are in birth order); length t_end+1."""
    return np.searchsorted(birth, np.arange(t_end + 1), side="left").astype(np.int64)


def permuted_shadow(tree: Tree, rng: np.random.Generator) -> Tree:
    """The heritability null: children reassigned at random among their birth step's real parents."""
    child = np.flatnonzero(tree.parent >= 0)
    if child.size == 0:
        return tree
    st = tree.birth[child].astype(np.float64)
    order = np.argsort(st + rng.random(child.size), kind="stable")  # random order within each step
    parent = tree.parent.copy()
    parent[child] = tree.parent[child[order]]
    return Tree(parent=parent, birth=tree.birth, death=tree.death, t_end=tree.t_end)


def neutral_shadow(tree: Tree, rng: np.random.Generator, gap: int | None = None) -> Tree:
    """Bedau's neutral shadow with the run's per-step birth, immigrant and death counts (module docstring).

    Within a step the deaths come first (a parent outlives its child's birth step, as in the run), then the
    births, whose parents are uniform over the alive agents of age ≥ ``gap`` (default ``min_gap(tree)``; the
    founders at any age, ``eligible_from``);
    newborns join the pool after the step. Immigrants (roots born after step 0) are kept as roots.
    """
    gap = min_gap(tree) if gap is None else gap
    n, t_end = tree.n, tree.t_end
    birth = tree.birth
    lo = _step_ranges(birth, t_end)
    n0 = int(np.count_nonzero(birth < 0))
    is_child = tree.parent >= 0
    deaths_per_step = np.bincount(tree.death[tree.death < t_end], minlength=t_end)
    first = eligible_from(birth, gap)
    parent = np.full(n, -1, np.int32)
    death = np.full(n, t_end, np.int32)

    pool = np.empty(n, np.int32)  # alive agents in pool[:size]; capacity covers every agent
    pool[:n0] = np.arange(n0, dtype=np.int32)
    size = n0
    event_steps = np.flatnonzero((np.diff(lo) > 0) | (deaths_per_step > 0))
    for t in event_steps:
        t = int(t)
        nd = int(deaths_per_step[t])
        if nd:
            if nd > size:
                raise ValueError(f"step {t}: {nd} deaths but only {size} alive in the shadow")
            pos = np.sort(rng.choice(size, nd, replace=False))
            death[pool[pos]] = t
            new_size = size - nd
            holes = pos[pos < new_size]
            keep = np.ones(nd, bool)
            keep[pos[pos >= new_size] - new_size] = False
            pool[holes] = pool[new_size:size][keep]
            size = new_size
        a, b = int(lo[t]), int(lo[t + 1])
        if b > a:
            kids = np.arange(a, b)
            kids = kids[is_child[a:b]]
            if kids.size:
                parent[kids] = _eligible_draw(pool[:size], first, t, kids.size, rng)
            pool[size : size + (b - a)] = np.arange(a, b, dtype=np.int32)
            size += b - a
    return Tree(parent=parent, birth=birth.copy(), death=death, t_end=t_end)


def _eligible_draw(
    alive: np.ndarray, first: np.ndarray, t: int, k: int, rng: np.random.Generator
) -> np.ndarray:
    """``k`` uniform draws (with replacement) from the agents of ``alive`` that may be parents in step ``t``
    (``first``: ``eligible_from``)."""
    out = np.empty(0, np.int32)
    for _ in range(64):
        cand = alive[rng.integers(0, alive.size, size=2 * (k - out.size) + 16)]
        out = np.concatenate([out, cand[first[cand] <= t]])
        if out.size >= k:
            return out[:k]
    eligible = alive[first[alive] <= t]  # rare: few eligible agents, draw from them directly
    if eligible.size == 0:
        raise ValueError(f"step {t}: no eligible parent in the neutral shadow")
    return eligible[rng.integers(0, eligible.size, size=k)]
