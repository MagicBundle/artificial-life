"""The voice analysis (DESIGN §39.14) on synthetic worlds with a planted code: the information measures and
their
shuffle nulls, the sampler, the arm contrast and the verdict, the scorecard adapter. Host numpy only."""

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
    assert b["transfer"]["z"] > spec.THRESH["transfer"]["z"] and b["reception"]["p_family"] <= 0.02


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


def test_the_pooled_test_is_calibrated_under_the_null() -> None:
    hits = {f: 0 for f in ("code", "reception", "transfer")}
    n = 24
    for seed in range(n):
        tab = vs.sample_tables(np.random.default_rng(100 + seed), "random", n=150, steps=40, rows=150)
        p = analyse.prepare(tab, (0, 40))
        for f in hits:
            r = analyse.run_family(p, f, 60, np.random.default_rng(seed), 30)
            hits[f] += int(r is not None and r.p_family <= 0.05)
    assert all(h <= 0.25 * n for h in hits.values()), hits  # nominal 5 %; a loose guard against a broken null


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
    assert "## Kin groups" in md and "Dialect drift" in md
    cmp = summary.compare_windows(s, s)
    assert cmp["population_js"] == pytest.approx(0.0, abs=1e-9) and cmp[
        "code_table_correlation"
    ] == pytest.approx(1.0)


# --- the arm contrast and the verdict ---


class FakeArm(meaning.ArmRun):
    def __init__(self, arm: str, seed: int, tab: tables.Tables, fit: dict[str, np.ndarray]):
        self.arm, self.seed, self.tab, self.fit = arm, seed, tab, fit
        self.window, self.prep, self.run = WINDOW, None, Path("x")
        self.fork_start, self.designed = 0, True

    def as_designed(self, rules: meaning.Rules) -> bool:
        return self.designed


def _fit(rng: np.random.Generator, level: float, due: float = 100.0) -> dict[str, np.ndarray]:
    return {
        "population": level + rng.normal(0, 1, 12),
        "births": 0.01 + 0.001 * rng.normal(0, 1, 12),
        "energy": 500 + rng.normal(0, 5, 12),
        "comm_due": np.full(12, due) + rng.normal(0, 1, 12),
        "last_step": np.array([30000]),
    }


_TABLES: dict[tuple[str, int], tables.Tables] = {}


def _table(mode: str, seed: int) -> tables.Tables:
    if (mode, seed) not in _TABLES:
        _TABLES[(mode, seed)] = vs.sample_tables(np.random.default_rng(seed), mode, n=200, steps=80, rows=200)
    return _TABLES[(mode, seed)]


def _arms(none_mode: str, ctrl_mode: str, benefit: float, due: dict[str, float] | None = None):
    rules = meaning.Rules()
    rules.n_perm, rules.n_perm_block = 60, 8  # the smallest p is 1/61, so the threshold p is relaxed to match
    rules.thresholds = {k: {**v, "p": 0.02} for k, v in rules.thresholds.items()}
    due = due or {}
    arms: dict[str, dict[int, meaning.ArmRun]] = {n: {} for n in spec.ARMS}
    for s in (1, 2, 3):
        rng = np.random.default_rng(50 + s)
        for i, name in enumerate(spec.ARMS):
            mode = none_mode if name == "none" else ctrl_mode
            tab = _table(mode, 10 * s + i)
            fit = _fit(rng, 1000 + (benefit if name == "none" else 0), due.get(name, 100.0))
            arms[name][s] = FakeArm(name, s, tab, fit)
    return rules, arms


@pytest.fixture(scope="module")
def language_case():
    return _arms("planted", "random", benefit=15.0)


def test_a_planted_language_gets_the_verdict_language(language_case) -> None:
    rules, arms = language_case
    res = meaning.run_meaning(arms, rules, log=lambda m: None)
    d = res["decision"]
    assert d["verdict"] == "language", json.dumps(jsonable(d["clauses"]), indent=1)[:1500]
    assert all(c["passed"] for c in d["clauses"].values())
    assert res["transfer_contrast"]["deaf"]["call"]["label"] == "supported"
    json.dumps(jsonable(res), allow_nan=False)
    assert "Verdict: language" in meaning.to_markdown(res)


def test_use_without_benefit_is_not_a_language() -> None:
    rules, arms = _arms("planted", "random", benefit=0.0)  # the same worlds with no fitness difference
    res = meaning.run_meaning(arms, rules, log=lambda m: None)
    d = res["decision"]
    assert d["verdict"] == "signalling without meaning" and "use without benefit" in d["detail"]
    assert d["clauses"]["transfer_contrast"]["passed"] and not d["clauses"]["fitness"]["passed"]


def test_a_benefit_of_hearing_that_the_decoy_shares_is_not_a_benefit_of_the_calls() -> None:
    rules, arms = _arms("planted", "random", benefit=15.0)
    for s, a in arms["decoy"].items():  # the decoy world is as fit as the real one
        a.fit = _fit(np.random.default_rng(90 + s), 1015)
    d = meaning.run_meaning(arms, rules, log=lambda m: None)["decision"]
    assert (
        d["verdict"] == "signalling without meaning" and "benefit of hearing but none specific" in d["detail"]
    )
    assert "none beats deaf on population" in d["detail"] and "none beats decoy on no readout" in d["detail"]


def test_unmatched_cost_withholds_the_language() -> None:
    rules, arms = _arms("planted", "random", benefit=15.0, due={"deaf": 60.0})
    res = meaning.run_meaning(arms, rules, log=lambda m: None)
    assert not res["cost"]["matched"] and res["decision"]["verdict"] == "signalling without meaning"
    assert "cost" in res["decision"]["detail"]


def test_a_tic_is_information_without_use_and_random_calls_are_no_evidence() -> None:
    rules, arms = _arms("tic", "tic", benefit=0.0)
    d = meaning.run_meaning(arms, rules, log=lambda m: None)["decision"]
    assert d["verdict"] == "signalling without meaning" and "information without use" in d["detail"]
    rules, arms = _arms("random", "random", benefit=0.0)
    d = meaning.run_meaning(arms, rules, log=lambda m: None)["decision"]
    assert d["verdict"] == "no evidence"


def test_the_environment_alone_is_not_a_language() -> None:
    """Neighbours' situations reach the hearer in every arm: the transfer test passes in none, but so it
    does in
    deaf and decoy, so the contrast (and the verdict) says no."""
    rules, arms = _arms("environment", "environment", benefit=0.0)
    res = meaning.run_meaning(arms, rules, log=lambda m: None)
    assert res["decision"]["clauses"]["transfer"]["passed"]
    assert (
        not res["decision"]["clauses"]["transfer_contrast"]["passed"]
        and res["decision"]["verdict"] != "language"
    )


def test_a_fork_off_the_design_makes_the_report_exploratory() -> None:
    rules, arms = _arms("random", "random", benefit=0.0)
    rules.confirmatory = True  # as if a frozen pre-registration were loaded
    assert meaning.run_meaning(arms, rules, log=lambda m: None)["confirmatory"] is True
    arms["deaf"][2].designed = False  # a fork of the wrong length or a log of the wrong window
    res = meaning.run_meaning(arms, rules, log=lambda m: None)
    assert res["confirmatory"] is False and "pre-registered length or window" in meaning.to_markdown(res)


def test_the_arms_must_be_complete() -> None:
    rules, arms = _arms("random", "random", benefit=0.0)
    del arms["decoy"][3]
    with pytest.raises(ValueError, match="seeds"):
        meaning.run_meaning(arms, rules, log=lambda m: None)
    del arms["decoy"]
    with pytest.raises(ValueError, match="missing"):
        meaning.run_meaning(arms, rules, log=lambda m: None)


# --- the pre-registration and the scorecard adapter ---


def test_the_preregistration_is_hashed_and_drives_the_rules(tmp_path: Path) -> None:
    d = spec.default_prereg(code_git="abc")
    d["frozen"] = True
    h = pr.write_prereg(tmp_path / "p.json", d)
    loaded = pr.load_prereg(tmp_path / "p.json")
    r = meaning.Rules(loaded)
    assert r.confirmatory and r.hash == h and r.thresholds["transfer"] == {"z": 4.0, "bits": 0.05, "p": 0.01}
    assert not meaning.Rules().confirmatory
    text = (tmp_path / "p.json").read_text().replace('"bits": 0.05', '"bits": 0.01')
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
    off = SimpleNamespace(
        config=lambda: SimpleNamespace(comm=SimpleNamespace(enabled=False)), path=Path("r1")
    )
    assert "comm.enabled is false" in card.available(off)[0]
    monkeypatch.chdir(tmp_path)
    run = SimpleNamespace(config=lambda: SimpleNamespace(comm=SimpleNamespace(enabled=True)), path=Path("r1"))
    assert "a voice log" in card.available(run)[0]
    worlds["planted"].save(tmp_path / "analysis_cache" / "voice" / "r1" / "tables.npz")
    assert card.available(run, WINDOW) == []
    assert "window" in card.available(run, (5, 80))[0]
    r = card.compute(run, WINDOW, None, {}, tmp_path)
    assert r.evaluated and r.label == "signalling" and r.feature == "comm"
    assert r.primary == r.endpoint["transfer_bits"] and r.primary > 0.1
    assert any(c.passed for c in r.criteria)
    r0 = card.compute(run, WINDOW, None, {"deaf": run}, tmp_path)  # a control with the same log: contrast ~ 0
    assert abs(r0.endpoint["transfer_over_deaf"]) < 0.05


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
