"""Evolutionary-activity analysis (``alife.analysis``, docs/ACTIVITY.md) on synthetic phylogenies."""

from __future__ import annotations

import itertools
import json

import numpy as np
import pytest

from alife.analysis import activity as act
from alife.analysis import phylo, report
from alife.analysis.phylo import Tree
from alife.config import load_config
from alife.runio import RunDir

# --- a hand-built tree with known answers -------------------------------------------------------------------
#
#   founders 0, 1 (alive at step 0)
#   step 5:  0 → 2        step 12: 2 → 4
#   step 6:  0 → 3        step 13: 1 → 5
#   deaths: 1 in step 15, 5 in step 18, 0 in step 20; 2, 3, 4 survive to t_end = 30
#   grid every 10 → grid times 0, 10, 20, 30


def hand_tree() -> Tree:
    return Tree(
        parent=np.array([-1, -1, 0, 0, 2, 1], np.int32),
        birth=np.array([-1, -1, 5, 6, 12, 13], np.int32),
        death=np.array([20, 15, 30, 30, 30, 18], np.int32),
        t_end=30,
    )


def test_hand_tree_structure() -> None:
    t = hand_tree()
    phylo.validate_tree(t)
    assert phylo.min_gap(t) == 6  # 2 born at 5 to a founder born at −1
    np.testing.assert_array_equal(phylo.nearest_root(t.parent, t.parent < 0), [0, 1, 0, 0, 0, 1])
    roots = np.array([1, 1, 0, 0, 1, 0], bool)  # agent 4 founds its own component
    np.testing.assert_array_equal(phylo.nearest_root(t.parent, roots), [0, 1, 0, 0, 4, 1])


def test_validate_tree_rejects_bad_trees() -> None:
    t = hand_tree()
    with pytest.raises(ValueError, match="died before"):
        phylo.validate_tree(t._replace(death=np.array([20, 10, 30, 30, 30, 18], np.int32)))
    with pytest.raises(ValueError, match="not after its birth"):
        phylo.validate_tree(t._replace(death=np.array([20, 15, 5, 30, 30, 18], np.int32)))


def test_hand_tree_cohort_statistics() -> None:
    r = act.analyse(hand_tree(), 10, marker_rates=(0.0, 1.0), cohort_horizons=(10, 20))
    np.testing.assert_array_equal(r.grid, [0, 10, 20, 30])
    np.testing.assert_array_equal(r.n_alive, [2, 4, 4, 3])
    np.testing.assert_array_equal(r.births, [0, 2, 2, 0])
    c = r.cohorts
    # Cohort 0 (roots 0, 1): births 2, 3 by 0's lineage, then 4 (0's) and 5 (1's); 1's line dies in step 18.
    np.testing.assert_array_equal(c["D"][0], [2, 2, 1, 1])
    np.testing.assert_array_equal(c["A_cum"][0], [0, 2, 3, 3])
    # Reset shadow: window 1 gives each founder 2·1/2 = 1 expected birth; window 2 gives 0's lineage (three
    # members alive at 10 of four) 2·3/4 = 1.5 and 1's lineage 0.5. aN of 0's lineage: (2−1) + (1−1.5) = 0.5.
    np.testing.assert_allclose(c["AN_cum"][0], [0, 0, 0.5, 0.5])
    # Exposure (gap 6): founders expect 1/2 + 1/2 (+ 1/4 + 1/4 after 10), agents 2 and 3 expect 1/4 + 1/4.
    np.testing.assert_allclose(c["AF_cum"][0], [0, (2 - 1.0) + (0 - 1.0), 3 - 2.5, 3 - 2.5])
    # Presence activity of cohort 0's survivor: its age.
    np.testing.assert_array_equal(c["P_cum"][0], [0, 20, 20, 30])
    # Cohort 1 (roots 0, 1, 2, 3 alive at 10): 2 has child 4 and 1 has child 5 in window 2, and 1's line dies.
    np.testing.assert_array_equal(c["D"][1], [0, 4, 3, 2])
    np.testing.assert_array_equal(c["A_cum"][1], [0, 0, 1, 1])
    # Cohort 2 (roots 0, 2, 3, 4 alive at 20): no births; 0 dies in step 20, so three remain at 30.
    np.testing.assert_array_equal(c["D"][2], [0, 0, 4, 3])
    # Pools: horizon 10 holds only the cohort aged 10 at each grid time.
    pool = r.cohort_pooled[10]
    np.testing.assert_array_equal(pool.D, [0, 2, 3, 3])
    np.testing.assert_array_equal(r.cohort_pooled[20].D, [0, 2, 1 + 3, 2 + 3])
    # Cohort 1 at 20: four members alive at 10 share the 2 births (0.5 each); 0, 2 (one child) and 3 remain.
    np.testing.assert_allclose(r.cohort_pooled[20].AN_cum[2], 0.5 + (-0.5 + 0.5 - 0.5))


def test_marker_components_limits() -> None:
    t = hand_tree()
    r = act.analyse(t, 10, marker_rates=(0.0, 1.0))
    m0 = r.marker[0.0]  # no founding events: the components are the founder clades, as cohort 0
    np.testing.assert_array_equal(m0.D, r.cohorts["D"][0])
    np.testing.assert_array_equal(m0.A_cum, r.cohorts["A_cum"][0])
    np.testing.assert_allclose(m0.AN_cum, r.cohorts["AN_cum"][0])
    m1 = r.marker[1.0]  # every agent its own component: D = alive agents, activity = own offspring
    np.testing.assert_array_equal(m1.D, r.n_alive)
    np.testing.assert_array_equal(m1.A_cum, [0, 2, 3, 1])  # 0 (two children) is alive until 20, 2 has one
    np.testing.assert_array_equal(m1.created, [2, 2, 1, 0])  # 5 lives between grid times: never present


def test_marker_roots_are_a_fixed_hash() -> None:
    t = hand_tree()
    big = Tree(np.full(100_000, -1, np.int32), np.full(100_000, -1, np.int32), np.ones(100_000, np.int32), 1)
    frac = phylo.marker_roots(big._replace(parent=np.zeros(100_000, np.int32)), 0.1).mean()
    assert abs(frac - 0.1) < 0.005
    np.testing.assert_array_equal(phylo.marker_roots(t, 0.3), phylo.marker_roots(t, 0.3))
    assert phylo.marker_roots(t, 0.0).sum() == 2  # only the founders


def test_crossing_threshold_and_a_new() -> None:
    real = np.zeros((2, act.N_BINS))
    shadow = np.zeros((2, act.N_BINS))
    real[:, 5], shadow[:, 5] = 1, 4  # shadow-dominated at low activity
    real[:, 30], shadow[:, 30] = 3, 1  # real-dominated from bin 30 (a = 10^2.9) on
    real[:, 50] = 1
    a0, a1, ap = act.crossing_threshold(real, shadow)
    assert ap == pytest.approx(act.A_EDGES[30])
    assert a0 < ap < a1
    hist_sum = real * act.A_EDGES
    d = np.array([5.0, 5.0])
    np.testing.assert_allclose(
        act.a_new(real, hist_sum, d, act.A_EDGES[30], act.A_EDGES[30]), 3 * act.A_EDGES[30] / 5
    )
    np.testing.assert_allclose(
        act.a_new(real, hist_sum, d, act.A_EDGES[30], act.A_EDGES[30], count=True), 0.6
    )


# --- synthetic populations ----------------------------------------------------------------------------------


def _population(
    n0: int, steps: int, b: int, rng: np.random.Generator, select: float = 0.0, gap: int = 5
) -> Tree:
    """Constant population: each step ``b`` uniform random deaths, then ``b`` births. Parents are chosen with
    probability ∝ ``exp(select · z)`` among agents of age ≥ ``gap``, where ``z`` is a heritable trait
    (child = parent + N(0, 0.1²)); ``select = 0`` is a neutral tree."""
    n = n0 + (steps - gap) * b  # no events before the founders are old enough to reproduce
    parent = np.full(n, -1, np.int32)
    birth = np.full(n, -1, np.int32)
    death = np.full(n, steps, np.int32)
    z = np.zeros(n)
    alive = list(range(n0))
    nxt = n0
    for t in range(gap, steps):
        for i in sorted(rng.choice(len(alive), b, replace=False), reverse=True):
            death[alive[i]] = t
            alive[i] = alive[-1]
            alive.pop()
        pool = np.array([a for a in alive if birth[a] <= t - gap], dtype=np.int64)
        w = np.exp(select * (z[pool] - z[pool].max()))
        for p in rng.choice(pool, b, p=w / w.sum()):
            parent[nxt], birth[nxt], z[nxt] = p, t, z[p] + rng.normal(0, 0.1)
            alive.append(nxt)
            nxt += 1
    return Tree(parent, birth, death, steps)


@pytest.fixture(scope="module")
def neutral_tree() -> Tree:
    return _population(300, 3000, 2, np.random.default_rng(1))


@pytest.fixture(scope="module")
def selected_tree() -> Tree:
    return _population(300, 3000, 2, np.random.default_rng(2), select=8.0)


def test_shadows_keep_counts(neutral_tree: Tree) -> None:
    rng = np.random.default_rng(0)
    for sh in (phylo.neutral_shadow(neutral_tree, rng), phylo.permuted_shadow(neutral_tree, rng)):
        phylo.validate_tree(sh)
        np.testing.assert_array_equal(sh.birth, neutral_tree.birth)
        np.testing.assert_array_equal(
            np.bincount(sh.death, minlength=3001), np.bincount(neutral_tree.death, minlength=3001)
        )
        assert phylo.min_gap(sh) >= phylo.min_gap(neutral_tree)
    perm = phylo.permuted_shadow(neutral_tree, rng)
    np.testing.assert_array_equal(perm.death, neutral_tree.death)
    kids = neutral_tree.parent >= 0
    for t in (100, 1500, 2999):  # each step's multiset of parents is unchanged
        m = kids & (neutral_tree.birth == t)
        np.testing.assert_array_equal(np.sort(perm.parent[m]), np.sort(neutral_tree.parent[m]))


def _excess(tree: Tree, space: str, seed: int = 0) -> dict[str, float]:
    st = report.Settings(
        grid_every=100,
        marker_rates=(0.02,),
        cohort_horizons=(500,),
        shadows=2,
        seed=seed,
        windows=((0, 1500), (1500, None)),
    )
    flat, _ = report.compute(tree, st, log=lambda m: None)
    d = report.Data(flat, st.shadows)
    m = d.grid >= 1000

    def abar(t: str) -> np.ndarray:
        return d.get(t, space, "A_cum") / np.maximum(d.get(t, space, "D"), 1)

    def mean_an(t: str) -> np.ndarray:
        return d.get(t, space, "AN_cum") / np.maximum(d.get(t, space, "D"), 1)

    return {
        "abar_rel": float(np.mean(abar("real")[m] / d.mean("neutral", abar)[m]) - 1),
        "d_rel": float(
            np.mean(d.get("real", space, "D")[m] / d.mean("neutral", lambda t: d.get(t, space, "D"))[m]) - 1
        ),
        "an_real": float(np.mean(mean_an("real")[m])),
        "an_neutral": float(np.mean(d.mean("neutral", mean_an)[m])),
        "abar": float(np.mean(abar("real")[m])),
    }


@pytest.mark.parametrize("space", ["marker_0.02", "cohort_h500"])
def test_neutral_tree_has_no_excess(neutral_tree: Tree, space: str) -> None:
    """A neutral phylogeny is indistinguishable from its neutral shadow: normalised activity is near zero."""
    e = _excess(neutral_tree, space)
    assert abs(e["abar_rel"]) < 0.25
    assert abs(e["d_rel"]) < 0.25
    assert abs(e["an_real"] - e["an_neutral"]) < 0.15 * e["abar"]


def test_selected_tree_has_excess(neutral_tree: Tree, selected_tree: Tree) -> None:
    """A heritable fecundity advantage shows as fewer, more active lineages and positive normalised excess."""
    e = _excess(selected_tree, "cohort_h500")
    assert e["d_rel"] < -0.3
    assert e["abar_rel"] > 0.3
    assert e["an_real"] - e["an_neutral"] > 0.15 * e["abar"]


def test_summary_and_table(selected_tree: Tree) -> None:
    st = report.Settings(
        grid_every=100,
        marker_rates=(0.02,),
        cohort_horizons=(500,),
        shadows=1,
        windows=((0, 1500), (1500, None)),
    )
    flat, _ = report.compute(selected_tree, st, log=lambda m: None)
    s = report.summarise(flat, st)
    json.dumps(s, default=report._f)  # serialisable
    cell = s["spaces"]["cohort_h500"]["windows"]["1k-3k"]
    assert cell["valid"] and cell["bedau"]["class"] and cell["channon"]["class"] in report.CLASSES
    assert sum(cell["sensitivity"]["channon_full"].values()) == act.THETA_N.size * 6
    assert "cohort_h500" in report.classification_table(s)


# --- loading from a run directory ---------------------------------------------------------------------------


def _part(cols: dict, lo: int, hi: int) -> dict:
    m = (cols["step"] >= lo) & (cols["step"] < hi)
    return {k: np.asarray(v)[m] for k, v in cols.items()}


def _hand_run(path, *, uid_of=None, start: int = 0, cuts=(0, 15, 30)) -> RunDir:
    """A run directory holding the hand tree's events from ``start`` on; ``uid_of`` maps agent → uid."""
    cfg = load_config(preset="tiny", overrides=["population.n_init=2"])
    rd = RunDir.create(path, cfg)
    t = hand_tree()
    uid = np.arange(t.n) if uid_of is None else np.asarray(uid_of)
    kids = np.flatnonzero(t.birth >= start)
    births = {
        "step": t.birth[kids],
        "uid": uid[kids],
        "parent_uid": uid[t.parent[kids]],
        "clade": np.zeros(kids.size),
        "generation": np.ones(kids.size),
        "slot": kids,
    }
    dead = np.flatnonzero((t.death < t.t_end) & (t.death >= start))
    deaths = {
        "step": t.death[dead],
        "uid": uid[dead],
        "age": np.zeros(dead.size),
        "cause": np.zeros(dead.size),
        "generation": np.zeros(dead.size),
        "clade": np.zeros(dead.size),
    }
    for lo, hi in itertools.pairwise(cuts):
        rd.write_events(lo, hi, _part(births, lo, hi), _part(deaths, lo, hi))
    rd.append_metrics(
        [
            {
                "step": 30,
                "births": int(kids.size),
                "immigrants": 0,
                "deaths_starved": int(dead.size),
                "deaths_killed": 0,
                "deaths_aged": 0,
                "deaths_unlogged": 0,
            }
        ]
    )
    if start:
        rd.update_meta(start_step=start)
    return rd


def _alive_checkpoint(rd: RunDir, step: int, uids) -> None:
    """A checkpoint stand-in with only the ``alive`` and ``uid`` arrays (all ``load_tree`` reads)."""
    import mlx.core as mx

    rd.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    u = np.asarray(uids, np.int64)
    mx.save_safetensors(
        str(rd.checkpoint_dir / f"step_{step:010d}.safetensors"),
        {"alive": mx.array(np.r_[np.ones(u.size, bool), False]), "uid": mx.array(np.r_[u, -1])},
    )


def _assert_trees_equal(a: Tree, b: Tree) -> None:
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x, y)


def test_load_tree_from_run_dir(tmp_path) -> None:
    rd = _hand_run(tmp_path / "run")
    tree, rep = phylo.load_tree(rd.path, check_checkpoint=False)
    _assert_trees_equal(tree, hand_tree())
    assert rep["chunks_tile"] and rep["births_match_metrics"] and rep["deaths_match_metrics"]
    assert rep["metrics_totals"]["deaths_unlogged"] == 0 and rep["min_parent_child_gap"] == 6
    assert rep["origin"] == rep["t0"] == 0


# --- forks and start steps ----------------------------------------------------------------------------------

#   restrict(hand_tree, 10): the agents alive at 10 (0, 1, 2, 3) become founders; 4 (born 12 to 2) and 5 (born
#   13 to 1) keep their parents; every step index moves by −10.
RESTRICTED_10 = Tree(
    parent=np.array([-1, -1, -1, -1, 2, 1], np.int32),
    birth=np.array([-1, -1, -1, -1, 2, 3], np.int32),
    death=np.array([10, 5, 20, 20, 20, 8], np.int32),
    t_end=20,
)


def test_restrict_hand_tree() -> None:
    r = phylo.restrict(hand_tree(), 10)
    phylo.validate_tree(r)
    _assert_trees_equal(r, RESTRICTED_10)
    assert phylo.min_gap(r, founder_ages=False) == 3  # only founders' children: fall back to them
    with pytest.raises(ValueError, match="restrict step"):
        phylo.restrict(hand_tree(), 30)


def test_restricted_cohort_equals_the_original_cohort() -> None:
    """Cohort 0 of the tree restricted to step 10 is cohort 1 (rooted at 10) of the whole tree."""
    whole = act.analyse(hand_tree(), 10, cohort_horizons=(10,), gap=6).cohorts
    part = act.analyse(phylo.restrict(hand_tree(), 10), 10, cohort_horizons=(10,), gap=6).cohorts
    for f in ("D", "A_cum", "AN_cum", "AF_cum"):
        np.testing.assert_allclose(part[f][0], whole[f][1][1:], err_msg=f)


def test_founders_may_reproduce_from_the_start() -> None:
    """In a fork, founders are of unknown age: a birth before ``gap`` steps have passed must not fail."""
    r = phylo.restrict(hand_tree(), 10)  # births at steps 2 and 3, gap 6
    np.testing.assert_array_equal(phylo.eligible_from(r.birth, 6), [0, 0, 0, 0, 8, 9])
    act.exposure_cdf(r, 6)  # no "births without eligible agents"
    sh = phylo.neutral_shadow(r, np.random.default_rng(0), gap=6)
    phylo.validate_tree(sh)
    assert np.all(sh.parent[4:] < 4)  # the only eligible parents are the founders


def test_load_tree_with_start(tmp_path) -> None:
    rd = _hand_run(tmp_path / "run")
    tree, rep = phylo.load_tree(rd.path, start=10, check_checkpoint=False)
    _assert_trees_equal(tree, RESTRICTED_10)
    assert rep["t0"] == 10 and rep["origin"] == 0 and rep["founders_at_t0"] == 4
    assert rep["min_parent_child_gap"] == 3


def test_load_tree_of_a_fork(tmp_path) -> None:
    """A fork's founders are the agents alive in its first checkpoint, with their source uids."""
    uid_of = [100, 101, 102, 103, 200, 201]  # founders keep source uids, births continue from next_uid
    rd = _hand_run(tmp_path / "fork", uid_of=uid_of, start=10, cuts=(10, 20, 30))
    _alive_checkpoint(rd, 10, [100, 101, 102, 103])
    _alive_checkpoint(rd, 30, [102, 103, 200])
    tree, rep = phylo.load_tree(rd.path)
    _assert_trees_equal(tree, RESTRICTED_10)
    assert rep["origin"] == rep["t0"] == 10 and rep["t_end"] == 30 and rep["n_init"] == 4
    assert rep["chunks_tile"] and rep["births_match_metrics"] and rep["deaths_match_metrics"]
    assert rep["checkpoint_check"]["ok"]
    with pytest.raises(ValueError, match="before the run's first step"):
        phylo.load_tree(rd.path, start=5)


def test_fork_report_uses_run_steps(tmp_path) -> None:
    uid_of = [100, 101, 102, 103, 200, 201]
    rd = _hand_run(tmp_path / "fork", uid_of=uid_of, start=10, cuts=(10, 20, 30))
    _alive_checkpoint(rd, 10, [100, 101, 102, 103])
    st = report.Settings(
        grid_every=5, marker_rates=(0.5,), cohort_horizons=(10,), shadows=1, windows=((10, None),)
    )
    s = report.run_report(rd.path, tmp_path / "out", st, plots=False, log=lambda m: None)
    flat = report.load_flat(tmp_path / "out" / "activity.npz")
    np.testing.assert_array_equal(flat["real/grid"], [10, 15, 20, 25, 30])
    assert s["t0"] == 10 and s["t_analysed"] == 30
    cells = s["spaces"]["marker_0.5"]["windows"]
    assert list(cells) == [report.window_label(10, 30)]  # the whole run starts at the fork step, not at 0
    assert cells[report.window_label(10, 30)]["valid"]
