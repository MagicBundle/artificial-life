"""Director vital signs (DIRECTOR.md §4): window aggregates, descriptor and tracker, from synthetic rows."""

from __future__ import annotations

import dataclasses
import json
import math
from typing import Any

import pytest

from alife.director.rules import RULE_PARAMS
from alife.director.vitals import DESCRIPTOR_NAMES, Tracker, aggregate, descriptor

P = RULE_PARAMS
PERF = {"wall_s": 12.5, "steps_per_s": 80.0, "peak_mem_gb": 3.3}


def row(step: int, steps: int = 10, **kw: Any) -> dict[str, Any]:
    """A metrics row with the keys the vitals read, all at harmless values."""
    base: dict[str, Any] = {
        "step": step,
        "steps": steps,
        "n_alive": 100.0,
        "n_alive_min": 100,
        "births": 0,
        "deaths_starved": 0,
        "deaths_killed": 0,
        "deaths_aged": 0,
        "deaths_catastrophe": 0,
        "emitters_frac": 0.5,
        "sharers_frac": 0.1,
        "mean_bonds_per_agent": 1.0,
        "predation_ratio": 0.1,
        "top_clade_share": 0.4,
        "clades_alive": 5,
        "clade_entropy": 1.2,
        "sigma_q50": 0.02,
        "rate_q50": 0.02,
        "frac_q50": 0.5,
        **PERF,
    }
    return {**base, **kw}


def heavy(step: int, **kw: Any) -> dict[str, Any]:
    return row(
        step,
        frac_in_bodies_ge4=0.3,
        bodies_largest=9,
        weight_diversity=4.0,
        bodies_hist={"1": 60, "2": 10, "3-4": 5},
        **kw,
    )


def test_aggregate_weights_rows_by_their_steps() -> None:
    w = aggregate(
        [
            row(90, steps=90, emitters_frac=0.2, n_alive=100.0),
            row(100, steps=10, emitters_frac=0.6, n_alive=200.0),
        ]
    )
    assert w.step == 100 and w.steps == 100
    assert w.emitters_frac == pytest.approx(0.24)
    assert w.n_alive == pytest.approx(110.0) and w.n_alive_last == 200.0


def test_aggregate_skips_none_and_absent_heavy_keys() -> None:
    w = aggregate([row(10, emitters_frac=None), row(20, emitters_frac=0.4)])
    assert w.emitters_frac == pytest.approx(0.4)  # None rows drop out of the mean, whatever their weight
    assert w.frac_in_bodies_ge4 is None and w.bodies_largest is None and w.singleton_frac is None
    assert w.weight_diversity is None
    assert aggregate([row(10, emitters_frac=None)]).emitters_frac is None


def test_snapshots_are_the_last_non_none_value() -> None:
    rows = [heavy(10, top_clade_share=0.3), row(20, top_clade_share=0.8), row(30, top_clade_share=None)]
    w = aggregate(rows)
    assert w.top_clade_share == 0.8  # the latest row that has one
    assert w.frac_in_bodies_ge4 == 0.3 and w.bodies_largest == 9 and w.weight_diversity == 4.0


def test_kill_share_counts_catastrophe_deaths_in_the_denominator() -> None:
    w = aggregate([row(10, deaths_killed=3, deaths_starved=4, deaths_aged=1, deaths_catastrophe=2)])
    assert w.deaths == 10 and w.kill_share == pytest.approx(0.3)
    assert aggregate([row(10)]).kill_share is None  # nobody died
    old = row(10, deaths_killed=1, deaths_starved=1, deaths_aged=0)
    del old["deaths_catastrophe"]  # rows from before the key existed
    assert aggregate([old]).kill_share == pytest.approx(0.5)


def test_singleton_fraction_is_hist_one_over_population() -> None:
    w = aggregate([heavy(10, n_alive=80.0)])
    assert w.singleton_frac == pytest.approx(60 / 80)
    assert aggregate([heavy(10, n_alive=40.0)]).singleton_frac == 1.0  # clipped to [0, 1]


def test_population_minimum_is_the_lowest_row_minimum() -> None:
    w = aggregate([row(10, n_alive=100.0, n_alive_min=90), row(20, n_alive=100.0, n_alive_min=60)])
    assert w.n_alive_min == 60
    old = row(10, n_alive=77.0)
    del old["n_alive_min"]  # a row from before n_alive_min: the mean stands in
    assert aggregate([old]).n_alive_min == 77.0


def test_wall_clock_keys_are_never_read() -> None:
    a = aggregate([row(10), heavy(20)])
    slow = [{**r, "wall_s": 1e9, "steps_per_s": 0.001, "peak_mem_gb": 99.0} for r in (row(10), heavy(20))]
    bare = [{k: v for k, v in r.items() if k not in PERF} for r in (row(10), heavy(20))]
    assert aggregate(slow) == a == aggregate(bare)


def test_descriptor_components_are_normalised_into_the_unit_interval() -> None:
    w = aggregate([heavy(10, emitters_frac=0.5, mean_bonds_per_agent=2.0, predation_ratio=0.125)])
    b = descriptor(w, P)
    assert b is not None and len(b) == len(DESCRIPTOR_NAMES) == 10
    named = dict(zip(DESCRIPTOR_NAMES, b, strict=True))
    assert named["emitters_frac"] == 0.5 and named["bonds_per_agent"] == pytest.approx(0.5)
    assert named["predation_ratio"] == pytest.approx(0.5)  # 0.125 / predation_max 0.25
    assert named["frac_in_bodies_ge4"] == 0.3 and named["kill_share"] == 0.0
    extreme = descriptor(aggregate([heavy(10, mean_bonds_per_agent=99.0, predation_ratio=5.0)]), P)
    assert extreme is not None and all(0.0 <= x <= 1.0 for x in extreme)
    assert dict(zip(DESCRIPTOR_NAMES, extreme, strict=True))["bonds_per_agent"] == 1.0


@pytest.mark.parametrize(
    ("name", "rng", "key"),
    [
        ("sigma_med", P.sigma_range, "sigma_q50"),
        ("rate_med", P.rate_range, "rate_q50"),
        ("frac_med", P.frac_range, "frac_q50"),
    ],
)
def test_gene_medians_are_log_scaled_onto_their_fixed_ranges(
    name: str, rng: tuple[float, float], key: str
) -> None:
    def comp(x: float) -> float:
        b = descriptor(aggregate([heavy(10, **{key: x})]), P)
        assert b is not None
        return b[DESCRIPTOR_NAMES.index(name)]

    lo, hi = rng
    assert comp(lo) == pytest.approx(0.0, abs=1e-12) and comp(hi) == pytest.approx(1.0)
    assert comp(lo / 10) == 0.0 and comp(hi * 10) == 1.0  # clipped outside the range
    assert comp(math.sqrt(lo * hi)) == pytest.approx(0.5)  # log scale: the geometric mean is the middle


def test_missing_components_count_as_zero_and_an_empty_world_has_no_descriptor() -> None:
    bare = row(10)
    for key in ("emitters_frac", "predation_ratio", "sigma_q50"):
        bare[key] = None
    b = descriptor(aggregate([bare]), P)
    assert b is not None
    named = dict(zip(DESCRIPTOR_NAMES, b, strict=True))
    assert named["emitters_frac"] == named["predation_ratio"] == named["sigma_med"] == 0.0
    assert descriptor(aggregate([row(10, n_alive=0.0)]), P) is None
    assert descriptor(aggregate([row(10, n_alive=None)]), P) is None


# --- tracker ---


def feed(tr: Tracker, rows: list[dict[str, Any]], every: int) -> list:
    out = []
    for r in rows:
        tr.observe(r)
        if r["step"] % every == 0:
            out.append(tr.close(r["step"]))
    return out


def test_tracker_closes_one_window_per_boundary() -> None:
    tr = Tracker(P)
    rows = [row(s, emitters_frac=0.1 * (s // 10)) for s in range(10, 61, 10)]
    bds = feed(tr, rows, 30)
    assert [b.step for b in bds] == [30, 60]
    assert bds[0].window.steps == 30 and bds[0].window.emitters_frac == pytest.approx(0.2)
    assert bds[1].window.emitters_frac == pytest.approx(0.5)  # only the rows after the first boundary
    assert tr.origin == 0 and len(tr.history) == 2 and tr.rows == []


def test_tracker_history_is_pruned_to_the_keep_horizon() -> None:
    p = dataclasses.replace(P, history_keep=100)
    tr = Tracker(p)
    feed(tr, [row(s) for s in range(10, 401, 10)], 10)
    assert [h.step for h in tr.history] == list(range(310, 401, 10))


def test_monoculture_streak_starts_at_the_window_start_and_resets() -> None:
    tr = Tracker(P)
    seq = [0.5, 0.95, 0.97, 0.92, 0.6, 0.95]
    starts = []
    for i, share in enumerate(seq, 1):
        tr.observe(row(i * 100, steps=100, top_clade_share=share))
        tr.close(i * 100)
        starts.append(tr.mono_since)
    assert starts == [None, 100, 100, 100, None, 500]


def test_extinction_risk_from_carrying_capacity_and_from_a_fall() -> None:
    tr = Tracker(P)
    tr.observe(row(100, steps=100, n_alive=500.0, n_alive_min=480))
    bd = tr.close(100)
    calm = tr.vitals(bd, k_est=800.0, n_cap=2000)
    assert (
        not calm.extinction_risk and calm.n_over_k == pytest.approx(500 / 800) and calm.pop_frac_cap == 0.25
    )
    assert tr.vitals(bd, k_est=5000.0, n_cap=2000).extinction_risk  # 0.1 < 0.2 of K_est
    tr.observe(row(200, steps=100, n_alive=300.0, n_alive_min=250))
    fall = tr.vitals(tr.close(200), k_est=800.0, n_cap=2000)
    assert fall.fall == pytest.approx(1 - 250 / 500) and fall.extinction_risk
    # a fall older than the fall window no longer counts
    far = 200 + P.risk_fall_window + 100
    tr.observe(row(far, steps=100, n_alive=300.0, n_alive_min=290))
    assert not tr.vitals(tr.close(far), k_est=800.0, n_cap=2000).extinction_risk


COMPACT_2D_KEYS = [
    "n_alive",
    "k_est",
    "n_over_k",
    "fall",
    "extinction_risk",
    "top_clade_share",
    "novelty",
    "coverage",
    "new_cells_200k",
    "b_shift",
]


def test_2d_vitals_have_one_k_and_the_log_keys_of_build_step_2() -> None:
    """2D has one K: no second capacity is carried, the risk ratio is ``n / K_est``, and the logged vitals are
    exactly the keys of build step 2 in their order (so a 2D ``director.jsonl`` is byte-identical, §27.10)."""
    tr = Tracker(P)
    tr.observe(row(100, steps=100, n_alive=500.0, n_alive_min=480))
    v = tr.vitals(tr.close(100), k_est=800.0, n_cap=2000)
    assert v.k_floor is None and v.n_over_k_floor is None
    assert v.n_over_k_risk == v.n_over_k == pytest.approx(500 / 800)
    assert list(v.compact()) == COMPACT_2D_KEYS


def test_3d_extinction_risk_reads_the_floor_layer_k_and_every_other_reader_keeps_k_est() -> None:
    """A spread population: the occupancy-weighted ``K_est`` is 5 000, the floor layer's capacity 800, and 500
    agents are alive. ``n / K_est`` is 0.1 (under ``risk_k_frac``) but the flag reads ``n / K_floor`` = 0.625
    and is clear; with the capacities the other way round it is set. ``k_est`` and ``n_over_k`` (the viability
    line, the rescue's size) are the occupancy-weighted ones in both."""
    tr = Tracker(P)
    tr.observe(row(100, steps=100, n_alive=500.0, n_alive_min=480))
    bd = tr.close(100)
    spread = tr.vitals(bd, k_est=5000.0, n_cap=2000, k_floor=800.0)
    assert spread.n_over_k == pytest.approx(0.1) and spread.n_over_k < P.risk_k_frac
    assert (
        spread.k_est == 5000.0 and spread.k_floor == 800.0 and spread.n_over_k_floor == pytest.approx(0.625)
    )
    assert spread.n_over_k_risk == spread.n_over_k_floor and not spread.extinction_risk
    assert spread.pop_frac_cap == 0.25
    sparse = tr.vitals(
        bd, k_est=800.0, n_cap=2000, k_floor=5000.0
    )  # the floor layer's food is the scarce one
    assert sparse.n_over_k == pytest.approx(0.625) and sparse.n_over_k_floor == pytest.approx(0.1)
    assert sparse.extinction_risk and sparse.k_est == 800.0
    # the same K on both sides is the 2D rule
    assert tr.vitals(bd, k_est=5000.0, n_cap=2000, k_floor=5000.0).extinction_risk
    assert not tr.vitals(bd, k_est=800.0, n_cap=2000, k_floor=800.0).extinction_risk
    # the threshold is the same number as in 2D: just under and just over risk_k_frac of the floor layer's K
    edge = 500 / P.risk_k_frac
    assert tr.vitals(bd, 800.0, 2000, k_floor=edge * 1.001).extinction_risk
    assert not tr.vitals(bd, 800.0, 2000, k_floor=edge * 0.999).extinction_risk


def test_3d_fall_criterion_is_independent_of_the_capacities() -> None:
    tr = Tracker(P)
    tr.observe(row(100, steps=100, n_alive=500.0, n_alive_min=480))
    tr.vitals(tr.close(100), 800.0, 2000, k_floor=800.0)
    tr.observe(row(200, steps=100, n_alive=300.0, n_alive_min=250))
    v = tr.vitals(tr.close(200), k_est=800.0, n_cap=2000, k_floor=800.0)
    assert v.fall == pytest.approx(1 - 250 / 500) and v.n_over_k_floor > P.risk_k_frac
    assert v.extinction_risk  # the fall alone sets the flag


def test_3d_compact_adds_the_floor_k_after_n_over_k_and_is_json_ready() -> None:
    tr = Tracker(P)
    tr.observe(row(100, steps=100, n_alive=123.456789))
    v = tr.vitals(tr.close(100), k_est=777.7777777, n_cap=2000, k_floor=222.2222222)
    c = v.compact()
    assert list(c) == [*COMPACT_2D_KEYS[:3], "k_floor", "n_over_k_floor", *COMPACT_2D_KEYS[3:]]
    assert c["k_floor"] == 222.2222 and c["n_over_k_floor"] == round(123.456789 / 222.2222222, 4)
    assert c["k_est"] == 777.7778 and c["n_over_k"] == round(123.456789 / 777.7777777, 4)
    assert json.loads(json.dumps(c, allow_nan=False)) == c
    # nobody alive: no ratio, no flag from the capacities
    empty = Tracker(P)
    empty.observe(row(100, steps=100, n_alive=None, n_alive_min=None))
    e = empty.vitals(empty.close(100), k_est=800.0, n_cap=2000, k_floor=0.0)
    assert e.n_over_k_floor is None and e.n_over_k_risk is None and not e.extinction_risk


def test_descriptor_shift_compares_with_the_boundary_a_shift_window_ago() -> None:
    p = dataclasses.replace(P, shift_window=200)
    tr = Tracker(p)
    rows = [row(s, steps=100, emitters_frac=0.1 if s <= 200 else 0.5) for s in (100, 200, 300, 400)]
    vs = []
    for r in rows:
        tr.observe(r)
        vs.append(tr.vitals(tr.close(r["step"]), 800.0, 2000))
    assert [v.b_shift for v in vs[:2]] == [None, None]  # no boundary 200 steps back yet
    want = math.sqrt(0.4**2 / 10)  # one of ten components moved by 0.4: step 300 vs 100, step 400 vs 200
    assert vs[2].b_shift == pytest.approx(want) and vs[3].b_shift == pytest.approx(want)


def test_vitals_compact_is_rounded_and_json_ready() -> None:
    tr = Tracker(P)
    tr.observe(row(100, steps=100, n_alive=123.456789))
    v = tr.vitals(tr.close(100), 777.7777777, 2000)
    c = v.compact()
    assert c["n_alive"] == 123.4568 and c["k_est"] == 777.7778 and c["novelty"] is None
    assert set(c) >= {"extinction_risk", "coverage", "new_cells_200k", "top_clade_share", "b_shift"}
