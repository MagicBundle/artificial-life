"""The voice analysis (DESIGN §39.14) on synthetic worlds with a planted code: the information measures and
their
shuffle nulls, the sampler, the arm contrast and the verdict (rules version 3), the scorecard adapter. Host
numpy
only."""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import voice_synth as vs
from alife.analysis import comm as comm_card
from alife.analysis.emergence import prereg as pr
from alife.analysis.emergence import scorecards
from alife.analysis.voice import analyse, meaning, spec, summary, tables
from alife.analysis.voice import measures as ms
from alife.analysis.voice.__main__ import jsonable
from alife.step import ActFlag

WINDOW = (0, 80)
PKG = Path(__file__).resolve().parents[1] / "src" / "alife" / "analysis" / "voice"


# --- the primitives ---


def test_the_flag_constants_are_the_engines() -> None:
    assert ActFlag.ATTACKED.value == tables.FLAG_ATTACKED and ActFlag.FORMED.value == tables.FLAG_FORMED
    assert ActFlag.BUDDED.value == tables.FLAG_BUDDED and ActFlag.SPOKE.value == tables.FLAG_SPOKE


def test_mutual_information_of_codes() -> None:
    rng = np.random.default_rng(0)
    x = rng.integers(0, 4, 20000)
    assert abs(ms.mi_codes(x, rng.integers(0, 3, 20000), 4, 3)) < 0.002  # independent: Miller-Madow bias ~ 0
    assert ms.mi_codes(x, x, 4, 4) == pytest.approx(2.0, abs=0.003)  # a copy: H(X) = 2 bits
    y = (x + (rng.random(20000) < 0.25) * rng.integers(1, 4, 20000)) % 4  # a noisy copy
    assert 0.5 < ms.mi_codes(x, y, 4, 4) < 2.0
    assert ms.entropy_bits(x, 4) == pytest.approx(2.0, abs=0.003)


def test_conditioning_removes_what_the_stratum_explains() -> None:
    rng = np.random.default_rng(1)
    s = rng.integers(0, 2, 40000)
    x = (s + (rng.random(40000) < 0.1)) % 2  # x and y both copy s: related only through s
    y = (s + (rng.random(40000) < 0.1)) % 2
    assert ms.cmi_codes(x, y, 2, 2) > 0.3
    assert abs(ms.cmi_codes(x, y, 2, 2, strata=s)) < 0.003


def test_the_shuffle_keeps_every_index_inside_its_stratum() -> None:
    rng = np.random.default_rng(2)
    strata = rng.integers(0, 7, 500)
    base = np.argsort(strata, kind="stable")
    perm = ms.shuffle_within(strata, rng.random(500), base)
    assert (
        sorted(perm) == list(range(500)) and (strata[perm] == strata).all() and (perm != np.arange(500)).any()
    )


def test_symbols_and_codes() -> None:
    v = np.array([[0, 0, 0, 0], [10, 0, 0, 0], [200, 0, 0, 0], [0, 0, 90, 5], [0, 255, 0, 0]], np.uint8)
    thr = ms.symbol_threshold(v)  # the median of the loudest channel over the non-silent rows
    assert thr == pytest.approx(145.0)
    assert ms.symbols(v, thr).tolist() == [0, 1, 2, 5, 4] and ms.alphabet(4) == 9
    assert ms.code_by_edges(np.array([0.0, 1.0, 5.0, 9.0]), np.array([1.0, 5.0])).tolist() == [0, 0, 1, 2]
    assert ms.quantile_edges(np.zeros(100), 3).tolist() == [
        0.0
    ]  # tied data gives fewer levels, not empty bins
    assert ms.jensen_shannon(np.array([1, 0]), np.array([0, 1])) == pytest.approx(1.0)
    assert ms.jensen_shannon(np.array([3, 1]), np.array([3, 1])) == pytest.approx(0.0)


def test_the_family_test_finds_a_planted_dependence_and_pays_for_the_search() -> None:
    rng = np.random.default_rng(3)
    n = 4000
    strata = rng.integers(0, 8, n)
    x = rng.integers(0, 3, n)
    pairs = [
        ms.Pair("planted", x, 3, ((x + (rng.random(n) < 0.2)) % 3).astype(np.int8), 3, strata),
        *[ms.Pair(f"null{i}", x, 3, rng.integers(0, 3, n), 3, strata) for i in range(5)],
    ]
    res = ms.family_test(pairs, 100, np.random.default_rng(4), min_size=10)
    assert res.names[res.best] == "planted" and res.excess[res.best] > 0.5 and res.p_family <= 0.02
    nulls = ms.family_test(pairs[1:], 100, np.random.default_rng(5), min_size=10)
    assert nulls.p_family > 0.05 or nulls.z[nulls.best] < 3.5


def test_a_family_needs_pairs_and_permutations() -> None:
    x = np.zeros(10, np.int8)
    with pytest.raises(ValueError, match="at least one pair"):
        ms.family_test([], 10, np.random.default_rng(0))
    with pytest.raises(ValueError, match="n_perm"):
        ms.family_test([ms.Pair("p", x, 1, x, 1, x)], 1, np.random.default_rng(0))


# --- the sampler ---


def test_a_sample_aligns_the_call_with_the_callers_context_and_the_source_with_the_loudest_neighbour() -> (
    None
):
    rng = np.random.default_rng(5)
    snaps = vs.world(rng, n=120, steps=6, mode="planted")
    g = vs.geo()
    s, h = tables.build_sample(snaps[1], snaps[2], snaps[3], g, 60, np.random.default_rng(0))
    # senders: the call spoken at u-1 = 1 with the context of u-1
    uid = s["uid"]
    assert (s["step"] == 2).all()
    hurt1 = snaps[1].loss[uid] > 0
    assert (s["c_loss"] > 0).tolist() == hurt1.tolist()
    assert (s["voice"] == snaps[1].voice[uid]).all()
    # hearers: the heard vector of u = 2, the source is the loudest weighted neighbour, its call from u-1
    hid = h["uid"]
    has = h["has_src"] > 0
    assert has.any() and (h["src_uid"][~has] == -1).all()
    cur, prev = snaps[2], snaps[1]
    for i in np.flatnonzero(has)[:25]:
        row = int(hid[i])
        best, best_w = -1, 0.0
        for j in cur.nbr[row]:
            if j < 0:
                continue
            d = cur.pos[j] - cur.pos[row]
            d -= vs.SIZE * np.round(d / vs.SIZE)
            w = max(0.0, 1.0 - np.linalg.norm(d) / vs.RANGE) * prev.voice[j].astype(float).sum()
            if w > best_w:
                best, best_w = int(j), w
        assert h["src_uid"][i] == best
        assert (h["src_voice"][i] == prev.voice[best]).all()
        assert (h["src_c_loss"][i] > 0) == (prev.loss[best] > 0)
    assert (h["heard"] == cur.heard[hid]).all()


def test_the_approach_response_follows_the_planted_rule() -> None:
    snaps = vs.world(np.random.default_rng(6), n=150, steps=6, mode="planted")
    _, h = tables.build_sample(snaps[1], snaps[2], snaps[3], vs.geo(), 150, np.random.default_rng(1))
    has = h["has_src"] > 0
    called0 = h["src_voice"][:, 0] > 120  # the planted rule: toward a channel-0 caller, away from the rest
    moved = h["r_approach"] != 0
    assert (h["r_approach"][has & moved & called0] == 1).mean() > 0.93
    assert (h["r_approach"][has & moved & ~called0] == -1).mean() > 0.93
    assert (h["r_approach"][~has] == 0).all()


def test_tables_round_trip(tmp_path: Path) -> None:
    tab = vs.sample_tables(np.random.default_rng(7), "planted", n=80, steps=12, rows=40)
    tab.save(tmp_path / "t.npz")
    back = tables.Tables.load(tmp_path / "t.npz")
    assert back.meta["run"] == tab.meta["run"] and back.width == vs.WIDTH
    for k, v in tab.hearers.items():
        assert np.array_equal(back.hearers[k], v, equal_nan=True), k
    assert set(back.senders) == set(tab.senders)


# --- the tests on planted worlds ---


@pytest.fixture(scope="module")
def worlds() -> dict[str, tables.Tables]:
    return {
        m: vs.sample_tables(np.random.default_rng(1), m) for m in ("planted", "random", "tic", "environment")
    }


def _run(tab: tables.Tables) -> dict[str, dict | None]:
    r = analyse.run_all(tab, WINDOW, n_perm=80, n_perm_block=10)
    return {k: (v["best"] if v else None) for k, v in r["families"].items()}


def test_a_planted_code_is_found_by_the_code_and_the_transfer_test(worlds: dict[str, tables.Tables]) -> None:
    b = _run(worlds["planted"])
    assert b["code"]["pair"] == "S~hurt" and b["code"]["excess"] > 0.6 and b["code"]["p_family"] <= 0.02
    assert b["transfer"]["pair"] == "hurt>approach" and b["transfer"]["excess"] > 0.1
    assert b["transfer"]["z"] > 4.0 and b["reception"]["p_family"] <= 0.02


def test_random_calls_give_the_null(worlds: dict[str, tables.Tables]) -> None:
    b = _run(worlds["random"])
    for fam in ("code", "reception", "transfer"):
        assert b[fam]["excess"] < 0.02 and b[fam]["p_family"] > 0.01, fam


def test_a_tic_is_information_without_use(worlds: dict[str, tables.Tables]) -> None:
    b = _run(worlds["tic"])
    assert b["code"]["excess"] > 0.6  # the calls say the caller's situation
    assert b["transfer"]["excess"] < 0.02 and b["reception"]["excess"] < 0.02  # nobody acts on them


def test_the_shared_environment_reaches_the_transfer_test_but_not_the_calls(
    worlds: dict[str, tables.Tables],
) -> None:
    b = _run(worlds["environment"])
    assert b["transfer"]["excess"] > 0.05  # the caveat that the arm contrast exists for
    assert b["code"]["excess"] < 0.02 and b["reception"]["excess"] < 0.02


def test_the_pooled_test_is_not_grossly_wrong_under_the_null() -> None:
    """A gross-failure guard (25 % at the 5 % level over 24 worlds) for the code and reception families; the
    tight check of the largest family is the slow test below."""
    hits = {f: 0 for f in ("code", "reception", "transfer")}
    n = 24
    for seed in range(n):
        tab = vs.sample_tables(np.random.default_rng(100 + seed), "random", n=150, steps=40, rows=150)
        p = analyse.prepare(tab, (0, 40))
        for f in hits:
            r = analyse.run_family(p, f, 60, np.random.default_rng(seed), 30)
            hits[f] += int(r is not None and r.p_family <= 0.05)
    assert all(h <= 0.25 * n for h in hits.values()), hits


@pytest.mark.slow
def test_the_largest_family_is_calibrated_on_independent_null_worlds() -> None:
    """At the 5 % level the max-T family p of ``transfer`` rejects at most 12 % of 150 independent null worlds
    (observed 6 %, Wilson 95% interval 3 to 11 %). This is a nominal check on independent rows. In the real
    logs
    the rows are 199 samples drawn from the same 42 000 animals, so they are not exchangeable and every pooled
    z (100 to 1200) and p (the floor) is non-binding: the verdict does not use them."""
    hits, n = 0, 150
    for seed in range(n):
        tab = vs.sample_tables(np.random.default_rng(1000 + seed), "random", n=120, steps=40, rows=120)
        p = analyse.prepare(tab, (0, 40))
        r = analyse.run_family(p, "transfer", 40, np.random.default_rng(seed), 20)
        hits += int(r is not None and r.p_family <= 0.05)
    assert hits <= 0.12 * n, hits


def test_block_estimates_follow_the_pooled_test(worlds: dict[str, tables.Tables]) -> None:
    p = analyse.prepare(worlds["planted"], WINDOW)
    blocks = analyse.pair_blocks(p, "transfer", "hurt>approach", 10, np.random.default_rng(0))
    assert np.isfinite(blocks).sum() >= 12 and np.nanmean(blocks) > 0.1
    odd = analyse.pair_blocks(p, "transfer", "hurt>approach", 10, np.random.default_rng(0), parity=1)
    assert np.isnan(odd[0::2]).all() and np.isfinite(odd[1::2]).sum() >= 6
    assert np.isnan(analyse.pair_blocks(p, "transfer", "no>such", 10, np.random.default_rng(0))).all()


def test_the_summary_reports_lineages_and_drift(worlds: dict[str, tables.Tables]) -> None:
    s = summary.build(worlds["planted"], WINDOW, n_perm=40, n_perm_block=8, top=3, with_tests=False)
    assert (
        s["population"]["speaker_frac"] == pytest.approx(1.0, abs=0.1) and s["population"]["vocabulary"] >= 2
    )
    assert len(s["lineages"]) == 3 and abs(sum(c["share"] for c in s["lineages"]) - 0.6) < 0.2
    top = s["lineages"][0]["top_calls"][0]
    assert top["call"] in {"ch0+", "ch1-", "ch1+", "ch0-"} and 0 < top["share"] <= 1
    d = s["drift"]
    assert d["population_js"] is not None and d["population_js"] < 0.02  # a stationary world does not drift
    json.dumps(jsonable(s), allow_nan=False)
    md = summary.to_markdown(s)
    assert "## Tag clusters" in md and "Dialect drift" in md
    cmp = summary.compare_windows(s, s)
    assert cmp["population_js"] == pytest.approx(0.0, abs=1e-9) and cmp[
        "code_table_correlation"
    ] == pytest.approx(1.0)


# --- the pieces of the decision ---


def test_the_block_index_follows_the_window() -> None:
    step = np.array([100, 101, 109, 110, 111, 119, 120, 130, 131, 90])
    b = tables.block_of(step, 100, 30, 3)  # blocks of 10 steps in (100, 130]
    assert b.tolist() == [-1, 0, 0, 0, 1, 1, 1, 2, -1, -1]


def test_the_excess_is_the_information_minus_the_null_mean() -> None:
    rng = np.random.default_rng(11)
    x = rng.integers(0, 3, 600).astype(np.int8)
    y = ((x + (rng.random(600) < 0.3)) % 3).astype(np.int8)
    strata = rng.integers(0, 20, 600)
    r = ms.family_test([ms.Pair("a", x, 3, y, 3, strata)], 30, np.random.default_rng(0), min_size=5)
    assert r.excess[0] == pytest.approx(r.mi[0] - r.null_mean[0], abs=1e-12) and r.null_mean[0] > 0.0
    blk = np.zeros(600, int)
    ex = ms.block_excess(ms.Pair("a", x, 3, y, 3, strata), blk, 1, 30, np.random.default_rng(0), 5)
    assert ex[0] < ms.block_info(ms.Pair("a", x, 3, y, 3, strata), blk, 1, 5)[0]  # the bias is subtracted


def test_the_null_permutes_inside_the_strata() -> None:
    """When x is constant inside every stratum a within-stratum shuffle changes nothing, so the null mean is
    exactly the observed value's null (0); a shuffle across strata would not be."""
    rng = np.random.default_rng(12)
    strata = np.repeat(np.arange(40), 5)
    x = (strata % 3).astype(np.int8)
    y = rng.integers(0, 2, 200).astype(np.int8)
    r = ms.family_test([ms.Pair("a", x, 3, y, 2, strata)], 20, np.random.default_rng(0), min_size=1)
    assert r.null_sd[0] == 0.0 and r.null_mean[0] == pytest.approx(r.mi[0])
    ex = ms.block_excess(
        ms.Pair("a", x, 3, y, 2, strata), np.zeros(200, int), 1, 10, np.random.default_rng(0), 1
    )
    assert ex[0] == pytest.approx(0.0, abs=1e-12)


def test_the_transfer_is_conditioned_on_the_hearers_own_situation() -> None:
    """If the source's context equals the hearer's own and the response depends on the own context only, the
    transfer given the own context is nil; without the conditioning it would be the whole entropy."""
    tab = vs.sample_tables(np.random.default_rng(13), "random", n=150, steps=40, rows=150)
    h = tab.hearers
    h["src_c_loss"] = h["c_loss"].copy()
    h["r_attack"] = (h["c_loss"] > 0).astype(np.int8)
    p = analyse.prepare(tab, (0, 40))
    r = analyse.run_family(p, "transfer", 30, np.random.default_rng(0), 20)
    ex = dict(zip(r.names, r.excess, strict=True))
    assert abs(ex["hurt>attack"]) < 0.01
    raw = ms.mi_codes(p.h_src_ctx["hurt"][p.h_has_src], p.h_resp["attack"][p.h_has_src], 2, 2)
    assert raw > 0.5


def test_the_reception_strata_hold_the_geometry() -> None:
    """The reception test is stratified on the audible count and the distance of the loudest source, so a
    heard call that only tells how near the neighbours are carries nothing."""
    tab = vs.sample_tables(np.random.default_rng(14), "random", n=150, steps=40, rows=150)
    h = tab.hearers
    near = h["audible"] >= 7  # the response follows the audible count, and so does the call (its channel)
    h["r_attack"] = near.astype(np.int8)
    h["heard"] = np.zeros_like(h["heard"])
    h["heard"][near, 0] = 200
    h["heard"][~near, 1] = 200
    p = analyse.prepare(tab, (0, 40))
    raw = ms.mi_codes(p.h_heard, p.h_resp["attack"], p.n_sym, 2)
    r = analyse.run_family(p, "reception", 30, np.random.default_rng(0), 20)
    assert raw > 0.2 and abs(dict(zip(r.names, r.excess, strict=True))["H~attack"]) < 0.02


def _contrast(means: list[float], label: str = "supported") -> dict:
    return {"per_seed": [{"mean": m} for m in means], "call": {"label": label}, "bounds": {}}


def test_the_minimum_effect_and_the_equivalence_gate() -> None:
    rules = meaning.Rules()
    x = rules.min_effect

    def cls(mean: float, spread: float = 0.0005, label: str = "supported") -> dict:
        return meaning.classify_contrast(_contrast([mean - spread, mean, mean + spread], label), rules)

    assert cls(x + 0.001)["meaning"] and cls(x + 0.001)["positive"]
    below = cls(
        x - 0.001, 0.001
    )  # positive, but a trace: below the minimum effect and not bounded below 0.005
    assert below["positive"] and not below["meaning"] and not below["equivalent"]
    assert not cls(x + 0.001, label="mixed")["positive"]  # the seed rule decides, not the mean alone
    none = cls(0.0001, label="not supported")
    assert none["equivalent"] and not none["positive"] and none["upper"] < rules.equivalence
    wide = meaning.classify_contrast(_contrast([-0.02, 0.0, 0.02], "not supported"), rules)
    assert not wide["equivalent"] and not wide["positive"]  # too noisy to bound: not enough to say
    lower = meaning.classify_contrast(_contrast([0.0, 0.002, 0.004], "supported"), rules)
    assert not lower["positive"]  # the seed rule says supported, the one-sided lower bound does not exclude 0


def test_the_seed_twin_floor_and_the_bounds() -> None:
    f = meaning.twin_floor(
        {"a": {1: np.array([0.0, 0.0]), 2: np.array([0.01, 0.01]), 3: np.array([0.0, 0.0])}}
    )
    assert f == pytest.approx(0.01)  # pairs (1,2) 0.01, (1,3) 0, (2,3) 0.01: the median magnitude
    b = meaning.seed_bounds([0.01, 0.02, 0.03], 0.05)
    assert b["mean"] == pytest.approx(0.02) and b["lower"] < 0.02 < b["upper"]
    assert b["upper"] == pytest.approx(0.02 + 2.919986 * 0.01 / np.sqrt(3), abs=1e-5)  # t(0.95, df 2)


def test_the_noise_floor_of_the_seed_twins_enters_the_seed_rule() -> None:
    """The seeds of an arm differ among themselves by more than the contrast: a consistent 0.002-bit contrast
    is then inside the twin noise and is not called supported; without the offsets it is."""
    rules = meaning.Rules()
    rng = np.random.default_rng(16)

    def arrays(offsets: tuple[float, float, float], contrast: float) -> dict:
        ctrl = {
            s: np.where(np.arange(16) % 2 == 1, o + rng.normal(0, 0.0003, 16), np.nan)
            for s, o in zip((1, 2, 3), offsets, strict=True)
        }
        none = {s: ctrl[s] + contrast for s in ctrl}
        return {"none": none, "decoy": ctrl, "deaf": ctrl}

    quiet = meaning.transfer_contrasts(arrays((0.0, 0.0, 0.0), 0.002), (1, 2, 3), rules)
    assert (
        quiet["contrast_decoy"]["call"]["label"] == "supported"
        and quiet["contrast_decoy"]["class"]["positive"]
    )
    noisy = meaning.transfer_contrasts(arrays((0.0, 0.02, -0.02), 0.002), (1, 2, 3), rules)
    assert noisy["contrast_decoy"]["floor"] > 0.01
    assert (
        noisy["contrast_decoy"]["call"]["label"] != "supported"
        and not noisy["contrast_decoy"]["class"]["positive"]
    )


def test_the_primary_contrast_is_against_the_decoy_and_the_secondary_against_the_deaf() -> None:
    rules = meaning.Rules()
    rng = np.random.default_rng(17)
    base = {s: np.where(np.arange(16) % 2 == 1, 0.03 + rng.normal(0, 0.0003, 16), np.nan) for s in (1, 2, 3)}
    shift = lambda d, x: {s: v + x for s, v in d.items()}  # noqa: E731
    # the real world is as informative as the decoy one and 0.02 bits above the deaf one
    r = meaning.transfer_contrasts(
        {"none": base, "decoy": shift(base, 0.0), "deaf": shift(base, -0.02)}, (1, 2, 3), rules
    )
    assert abs(r["contrast_decoy"]["class"]["mean"]) < 0.001 and not r["contrast_decoy"]["class"]["meaning"]
    assert r["contrast_deaf"]["bounds"]["mean"] == pytest.approx(0.02, abs=1e-3)
    # and the other way round
    r = meaning.transfer_contrasts(
        {"none": base, "decoy": shift(base, -0.02), "deaf": shift(base, 0.0)}, (1, 2, 3), rules
    )
    assert r["contrast_decoy"]["class"]["meaning"] and abs(r["contrast_deaf"]["bounds"]["mean"]) < 0.001


def _fabricated(rules: meaning.Rules) -> dict:
    """A result in which every clause holds, to switch one clause at a time."""
    call = {"label": "supported", "p": 1e-6}
    fit = {r: {c: {"call": dict(call)} for c in ("deaf", "decoy")} for r in rules.readouts}
    return {
        "as_designed": True,
        "contrast_decoy": {
            "class": {
                "positive": True,
                "meaning": True,
                "equivalent": False,
                "mean": 0.02,
                "lower": 0.01,
                "upper": 0.03,
            }
        },
        "contrast_deaf": {"bounds": {"mean": 0.03}},
        "speaker_frac": {"min": 0.9},
        "use": {"passed": True},
        "fitness": fit,
        "cost": {"matched": True},
    }


def test_every_clause_of_the_language_is_needed() -> None:
    rules = meaning.Rules()

    def verdict(edit) -> str:
        res = _fabricated(rules)
        edit(res)
        return meaning.decide(res, rules)["verdict"]

    assert verdict(lambda r: None) == "language"
    assert verdict(lambda r: r["speaker_frac"].update(min=0.01)) == "no evidence"
    assert verdict(lambda r: r["use"].update(passed=False)) == "signalling with trace meaning"
    assert (
        verdict(lambda r: r["contrast_deaf"]["bounds"].update(mean=-0.01)) == "signalling with trace meaning"
    )
    assert verdict(lambda r: r["cost"].update(matched=False)) == "signalling with trace meaning"
    one = lambda r: r["fitness"]["population"]["decoy"]["call"].update(label="mixed")  # noqa: E731
    assert verdict(one) == "language"  # another readout still passes in both contrasts

    def all_mixed(r: dict) -> None:
        for readout in r["fitness"]:
            r["fitness"][readout]["decoy"]["call"].update(label="mixed")

    assert verdict(all_mixed) == "signalling with trace meaning"
    for (
        readout
    ) in rules.readouts:  # a seed rule that says supported with an unconvincing p is not enough (Holm)
        res = _fabricated(rules)
        for r in rules.readouts:
            res["fitness"][r]["deaf"]["call"]["p"] = res["fitness"][r]["decoy"]["call"]["p"] = 0.9
        assert meaning.decide(res, rules)["verdict"] != "language", readout
    res = _fabricated(rules)
    res["contrast_decoy"]["class"].update(
        positive=False, meaning=False, equivalent=True, mean=0.0, upper=0.002
    )
    d = meaning.decide(res, rules)
    assert d["verdict"] == "signalling, no detectable meaning" and d["label"].endswith("(bound 0.0020 bits)")
    res["contrast_decoy"]["class"].update(equivalent=False, upper=0.02)
    assert meaning.decide(res, rules)["verdict"] == "no evidence"


def test_the_rules_come_from_a_version_3_preregistration_only() -> None:
    d = spec.default_prereg()
    assert d["rules_version"] == 3 and [a["id"] for a in d["amendments"]] == ["A1", "A2"]
    assert d["amendments"][1]["results_read"] is True and d["amendments"][0]["results_read"] is False
    r = meaning.Rules()
    assert r.revised_after_results and (r.min_effect, r.equivalence) == (0.005, 0.005)
    old = {k: v for k, v in d.items() if k != "rules_version"}
    with pytest.raises(ValueError, match="rules version"):
        meaning.Rules(old)


def test_the_stored_tests_must_have_run_with_the_registered_parameters() -> None:
    rules = meaning.Rules()
    ok = {"params": rules.tests_params()}
    meaning.check_tests(ok, rules, "x")
    for bad in (
        {},
        {"params": {**rules.tests_params(), "n_perm": 20}},
        {"params": {**rules.tests_params(), "seed": 9}},
    ):
        with pytest.raises(ValueError, match="rebuild the summary"):
            meaning.check_tests(bad, rules, "x")


def test_as_designed_checks_the_sampling_of_the_log() -> None:
    rules = meaning.Rules()
    a = object.__new__(meaning.ArmRun)
    a.fit, a.fork_start = {"last_step": np.array([rules.fork_steps])}, 1000
    lo, hi = rules.table_window
    a.window = (1000 + lo, 1000 + hi)
    a.tab = SimpleNamespace(meta={"every": rules.every, "rows": rules.rows, "seed": rules.seed})
    assert a.as_designed(rules)
    for k, v in (("every", 25), ("rows", 1024), ("seed", 3)):
        a.tab.meta[k] = v
        assert not a.as_designed(rules), k
        a.tab.meta[k] = getattr(rules, "every" if k == "every" else k)
    a.window = (1000 + lo, 1000 + hi - 1)
    assert not a.as_designed(rules)


# --- the arm contrast and the verdict ---


class FakeArm(meaning.ArmRun):
    def __init__(self, arm: str, seed: int, tab: tables.Tables, fit: dict[str, np.ndarray]):
        self.arm, self.seed, self.tab, self.fit = arm, seed, tab, fit
        self.window, self.prep, self.run = WINDOW, None, Path("x")
        self.fork_start, self.designed = 0, True

    def as_designed(self, rules: meaning.Rules) -> bool:
        return self.designed


CF_METRICS = ["d_thrust", "flip_attack", "flip_bond", "flip_speak", "d_voice", "d_vel", "n"]


def _fit(rng: np.random.Generator, level: float, due: float = 100.0) -> dict[str, np.ndarray]:
    return {
        "population": level + rng.normal(0, 1, 12),
        "births": 0.01 + 0.001 * rng.normal(0, 1, 12),
        "energy": 500 + rng.normal(0, 5, 12),
        "comm_due": np.full(12, due) + rng.normal(0, 1, 12),
        "last_step": np.array([30000]),
    }


_TABLES: dict[tuple[str, int], tables.Tables] = {}


def _table(mode: str, seed: int, d_thrust: float | None = None, silent: bool = False) -> tables.Tables:
    """A cached synthetic table; ``d_thrust`` adds the counterfactual block of a none arm, ``silent`` silences
    every sender."""
    key = (mode, seed, d_thrust, silent)
    if key not in _TABLES:  # type: ignore[comparison-overlap]
        base = vs.sample_tables(np.random.default_rng(seed), mode, n=200, steps=80, rows=200)
        senders = dict(base.senders)
        if silent:
            senders["voice"] = np.zeros_like(senders["voice"])
        cf = {}
        meta = dict(base.meta)
        if d_thrust is not None:
            st = np.zeros((6, 2, 2, 7), np.float32)
            st[:, :, 0, 0] = d_thrust
            st[:, :, 0, -1] = 1000
            st[:, :, 1, -1] = 100  # the silent control group changes by exactly 0
            cf = {"step": np.arange(6, dtype=np.int32), "stats": st}
            meta.update(cf_modes=["deaf", "decoy"], cf_metrics=CF_METRICS)
        _TABLES[key] = tables.Tables(senders, base.hearers, meta, cf)  # type: ignore[index]
    return _TABLES[key]  # type: ignore[index]


def _arms(
    modes: tuple[str, str, str],
    benefit: float | tuple[float, float] = 15.0,
    due: dict[str, float] | None = None,
    d_thrust: float | None = 0.04,
    silent: bool = False,
):
    rules = meaning.Rules()
    rules.n_perm, rules.n_perm_block = 60, 8
    benefit_deaf, benefit_decoy = benefit if isinstance(benefit, tuple) else (benefit, benefit)
    due = due or {}
    arms: dict[str, dict[int, meaning.ArmRun]] = {n: {} for n in spec.ARMS}
    for s in (1, 2, 3):
        rng = np.random.default_rng(50 + s)
        for i, name in enumerate(spec.ARMS):
            tab = _table(modes[i], 10 * s + i, d_thrust if name == "none" else None, silent)
            level = 1000 + {"none": 0.0, "deaf": -benefit_deaf, "decoy": -benefit_decoy}[name]
            arms[name][s] = FakeArm(name, s, tab, _fit(rng, level, due.get(name, 100.0)))
    return rules, arms


def _verdict(rules, arms):
    return meaning.run_meaning(arms, rules, log=lambda m: None)


@pytest.fixture(scope="module")
def language_case():
    return _arms(("planted", "random", "random"))


def test_a_planted_language_gets_the_verdict_language(language_case) -> None:
    rules, arms = language_case
    res = _verdict(rules, arms)
    d = res["decision"]
    assert d["verdict"] == "language", json.dumps(jsonable(d["clauses"]), indent=1)[:1800]
    assert all(c["passed"] for c in d["clauses"].values()) and d["revised_after_results"]
    assert res["contrast_decoy"]["class"]["meaning"] and res["contrast_decoy"]["call"]["label"] == "supported"
    for arm_blocks in res["transfer_blocks"].values():  # the estimate uses the odd blocks only
        for v in arm_blocks.values():
            assert all(x is None for x in v[0::2]) and all(x is not None for x in v[1::2])
    json.dumps(jsonable(res), allow_nan=False)
    md = meaning.to_markdown(res)
    assert "Verdict: language" in md and "decided after seeing the v2 results" in md


def test_the_arms_are_coded_with_the_none_arms_cut_points(language_case) -> None:
    rules, arms = language_case
    _verdict(rules, arms)
    for s in (1, 2, 3):
        base = arms["none"][s].prep
        for n in ("deaf", "decoy"):
            assert arms[n][s].prep.edges == base.edges and arms[n][s].prep.sym_thr == base.sym_thr
        own = tables.fit_edges(arms["decoy"][s].tab.senders, arms["decoy"][s].tab.hearers)
        assert own != base.edges  # the arm's own cut points would differ, so the sharing is not vacuous


def test_the_pair_is_chosen_on_the_even_blocks_and_estimated_on_the_odd_ones() -> None:
    """Altering the odd blocks of a table (a binary response, so the cut points do not move) cannot change the
    selected pair; altering the even blocks cannot change the odd-block estimate."""
    rules = meaning.Rules()
    rules.n_blocks = 16
    rng = np.random.default_rng(15)

    def preps(tabs):
        return {1: {"none": analyse.prepare(tabs[0], WINDOW), "decoy": analyse.prepare(tabs[1], WINDOW)}}

    t_none, t_decoy = _table("planted", 31), _table("random", 32)

    def altered(tab, parity):
        h = {k: v.copy() for k, v in tab.hearers.items()}
        blk = tables.block_of(h["step"], 0, 80, 16)
        sel = (blk >= 0) & (blk % 2 == parity)
        h["r_attack"][sel] = rng.integers(0, 2, int(sel.sum())).astype(np.int8)
        h["r_bond"][sel] = rng.integers(0, 2, int(sel.sum())).astype(np.int8)
        h["r_approach"][sel] = rng.integers(-1, 2, int(sel.sum())).astype(np.int8)
        return tables.Tables(tab.senders, h, tab.meta)

    base_pair, base_scores = meaning.select_contrast_pair(preps((t_none, t_decoy)), rules)
    odd_changed = meaning.select_contrast_pair(preps((altered(t_none, 1), altered(t_decoy, 1))), rules)
    assert odd_changed[0] == base_pair and odd_changed[1][base_pair] == pytest.approx(base_scores[base_pair])
    even_changed = meaning.select_contrast_pair(preps((altered(t_none, 0), altered(t_decoy, 0))), rules)
    assert even_changed[1][base_pair] != pytest.approx(
        base_scores[base_pair]
    )  # the selection does read the even blocks
    p0 = analyse.prepare(t_none, WINDOW)
    p1 = analyse.prepare(altered(t_none, 0), WINDOW)
    a = analyse.pair_blocks(p0, "transfer", "hurt>approach", 5, np.random.default_rng(3), parity=1)
    b = analyse.pair_blocks(p1, "transfer", "hurt>approach", 5, np.random.default_rng(3), parity=1)
    assert np.allclose(a[1::2], b[1::2], equal_nan=True) and np.isnan(a[0::2]).all()


def test_the_decoy_contrast_is_a_clause_of_its_own() -> None:
    """The decoy world carries the same behaviour as the real one while the deaf world does not: the content
    adds nothing beyond what the decoy already gives, so the primary contrast is nil and there is no language,
    although none would beat deaf on the planted pair."""
    rules, arms = _arms(("planted", "random", "planted"))
    rules.equivalence = 0.05
    res = _verdict(rules, arms)
    cl = res["decision"]["clauses"]
    assert not cl["meaning"]["passed"] and cl["meaning"]["equivalent"]
    assert res["decision"]["verdict"] == "signalling, no detectable meaning"
    assert res["decision"]["label"].startswith("signalling, no detectable meaning (bound ")


def test_a_meaning_without_a_fitness_benefit_is_only_a_signalling_with_trace_meaning() -> None:
    rules, arms = _arms(("planted", "random", "random"), benefit=0.0)
    d = _verdict(rules, arms)["decision"]
    assert d["verdict"] == "signalling with trace meaning" and "fitness" in d["detail"]
    assert d["clauses"]["meaning"]["passed"] and not d["clauses"]["fitness"]["passed"]


def test_a_benefit_of_hearing_that_the_decoy_shares_is_not_a_benefit_of_the_calls() -> None:
    rules, arms = _arms(("planted", "random", "random"), benefit=(15.0, 0.0))
    d = _verdict(rules, arms)["decision"]
    assert d["verdict"] != "language" and "none beats deaf on population" in d["detail"]
    assert "none beats decoy on no readout" in d["detail"]


def test_unmatched_cost_withholds_the_language() -> None:
    rules, arms = _arms(("planted", "random", "random"), due={"deaf": 60.0})
    res = _verdict(rules, arms)
    assert not res["cost"]["matched"] and res["decision"]["verdict"] != "language"
    assert "cost" in res["decision"]["detail"]


def test_a_use_that_the_counterfactual_does_not_show_withholds_the_language() -> None:
    rules, arms = _arms(("planted", "random", "random"), d_thrust=0.001)
    d = _verdict(rules, arms)["decision"]
    assert not d["clauses"]["use"]["passed"] and d["verdict"] != "language"


def test_calls_that_nobody_makes_are_no_evidence() -> None:
    rules, arms = _arms(("planted", "random", "random"), silent=True)
    d = _verdict(rules, arms)["decision"]
    assert d["verdict"] == "no evidence" and not d["clauses"]["speaking"]["passed"]


def test_random_calls_and_the_environment_alone_are_no_detectable_meaning() -> None:
    for mode in ("random", "tic", "environment"):
        rules, arms = _arms((mode, mode, mode), benefit=0.0)
        rules.equivalence = 0.05  # the synthetic worlds are small: the default bound needs more rows
        res = _verdict(rules, arms)
        assert res["decision"]["verdict"] == "signalling, no detectable meaning", (
            mode,
            res["decision"]["detail"],
        )
        assert not res["contrast_decoy"]["class"]["positive"]


def test_a_contrast_too_noisy_to_bound_is_no_evidence() -> None:
    rules, arms = _arms(("random", "random", "random"), benefit=0.0)
    rules.equivalence = 1e-6  # nothing can be bounded that tightly
    d = _verdict(rules, arms)["decision"]
    assert d["verdict"] == "no evidence" and "not enough to say" in d["detail"]


def test_a_fork_off_the_design_makes_the_report_exploratory() -> None:
    rules, arms = _arms(("random", "random", "random"), benefit=0.0)
    rules.confirmatory = True  # as if a frozen pre-registration were loaded
    assert _verdict(rules, arms)["confirmatory"] is True
    arms["deaf"][2].designed = False  # a fork of the wrong length or a log of the wrong window or sampling
    res = _verdict(rules, arms)
    assert res[
        "confirmatory"
    ] is False and "pre-registered length, window or sampling" in meaning.to_markdown(res)


def test_the_arms_must_be_complete() -> None:
    rules, arms = _arms(("random", "random", "random"), benefit=0.0)
    del arms["decoy"][3]
    with pytest.raises(ValueError, match="seeds"):
        _verdict(rules, arms)
    del arms["decoy"]
    with pytest.raises(ValueError, match="missing"):
        _verdict(rules, arms)


# --- the pre-registration and the scorecard adapter ---


def test_the_preregistration_is_hashed_and_drives_the_rules(tmp_path: Path) -> None:
    d = spec.default_prereg(code_git="abc")
    d["frozen"] = True
    h = pr.write_prereg(tmp_path / "p.json", d)
    loaded = pr.load_prereg(tmp_path / "p.json")
    r = meaning.Rules(loaded)
    assert (
        r.confirmatory and r.hash == h and (r.min_effect, r.equivalence, r.use_thrust) == (0.005, 0.005, 0.01)
    )
    assert not meaning.Rules().confirmatory
    text = (tmp_path / "p.json").read_text().replace('"min_effect_bits": 0.005', '"min_effect_bits": 0.0001')
    (tmp_path / "p.json").write_text(text)
    with pytest.raises(pr.PreregError):
        pr.load_prereg(tmp_path / "p.json")


def test_the_scorecard_degrades_then_reads_a_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, worlds: dict[str, tables.Tables]
) -> None:
    card = comm_card.CommScorecard()
    assert card.available(SimpleNamespace()) == ["a run directory"]
    assert (
        scorecards.evaluate("comm", SimpleNamespace(), (0, 10), None, {}, tmp_path).label == "not evaluated"
    )
    cfg = SimpleNamespace(comm=SimpleNamespace(enabled=True))
    off = SimpleNamespace(
        config=lambda: SimpleNamespace(comm=SimpleNamespace(enabled=False)), path=Path("r1")
    )
    assert "comm.enabled is false" in card.available(off)[0]
    monkeypatch.setenv("ALIFE_VOICE_LOGS", str(tmp_path / "logs"))
    monkeypatch.chdir(tmp_path)  # the log root does not depend on the working directory
    run = SimpleNamespace(config=lambda: cfg, path=tmp_path / "runs" / "r1")
    assert "a voice log" in card.available(run)[0]

    def log(tab: tables.Tables, name: str, run_path: Path) -> None:
        tables.Tables(tab.senders, tab.hearers, {**tab.meta, "run": str(run_path)}).save(
            tmp_path / "logs" / name / "tables.npz"
        )

    log(worlds["planted"], "r1", run.path)
    assert card.available(run, WINDOW) == [] and "window" in card.available(run, (5, 80))[0]
    other = SimpleNamespace(config=lambda: cfg, path=tmp_path / "elsewhere" / "r1")  # same name, another run
    assert "a voice log" in card.available(other)[0]
    r = card.compute(run, WINDOW, None, {}, tmp_path)
    assert r.evaluated and r.feature == "comm" and r.label.startswith("signalling, meaning not tested")
    assert r.primary == r.endpoint["transfer_bits"] and r.primary > 0.1
    decoy = SimpleNamespace(config=lambda: cfg, path=tmp_path / "runs" / "d1")
    log(worlds["random"], "d1", decoy.path)
    r1 = card.compute(
        run, WINDOW, None, {"decoy": decoy}, tmp_path
    )  # the planted code against a random decoy
    assert r1.label == "signalling with trace meaning" and r1.endpoint["transfer_over_decoy"] > 0.05
    assert any(c.passed for c in r1.criteria)
    log(worlds["planted"], "d1", decoy.path)  # a decoy that carries the same behaviour: nothing call-specific
    r2 = card.compute(run, WINDOW, None, {"decoy": decoy}, tmp_path)
    assert abs(r2.endpoint["transfer_over_decoy"]) < 0.05 and r2.label != "signalling with trace meaning"


# --- lint ---


def test_mlx_only_in_replay_and_nothing_writes_into_a_run() -> None:
    for p in sorted(PKG.glob("*.py")):
        code = re.sub(r'""".*?"""', "", p.read_text(), flags=re.S)
        assert bool(re.search(r"^\s*(import|from)\s+mlx", code, flags=re.M)) == (p.name == "replay.py"), (
            p.name
        )
        assert not re.search(
            r"^\s*(import|from)\s+(scipy|sklearn|requests|urllib|http|socket)", code, flags=re.M
        )
        assert "lmstudio" not in code.lower()
        assert not re.search(r"\.(write_text|write_bytes|unlink|rmtree)\(", code) or p.name in {
            "__main__.py"
        }, p.name
