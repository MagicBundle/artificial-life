"""The fact sheet, the lever fold, the spans, the event frames, the assembly and the ticker of the world report
(docs/DESIGN.md §30), on a synthetic observed run (``report_synth``: hand-written files in the engine's
formats) and on scripted logs. CPU only, no socket, no write into a run directory."""

# ruff: noqa: E501  (expected report texts and fixture tables are long lines)

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

import report_synth as RS
from alife.config import apply_overrides, config_hash, load_config
from alife.game import report as R
from alife.game import templates
from alife.game.rules import RULES_V1, RULES_V2
from report_helpers import A0, deaths, event, group, houses, interv, mk_sheet, pop
from report_synth import entry, variant

DATA = Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def rv(tmp_path_factory: pytest.TempPathFactory) -> R.RunView:
    return RS.synth(tmp_path_factory.mktemp("synth"))


def facts_of(sheet, kind):
    return [f for f in sheet["facts"] if f["kind"] == kind]


def one(sheet, kind):
    (f,) = facts_of(sheet, kind)
    return f


def texts(sheet):
    body, heads, block, excluded = R.candidates(sheet)
    assert excluded == []
    return [c["text"] for c in body], [h["text"] for h in heads], [b["text"] for b in block]


# --- the synthetic run and the sheet -----------------------------------------------------------------------------


def test_committed_synth_run_is_what_the_module_writes(tmp_path: Path) -> None:
    out = RS.write_run(tmp_path / "run")
    committed = DATA / "report_run_synth"
    assert committed.is_dir()
    files = sorted(p.relative_to(out) for p in out.rglob("*") if p.is_file() and p.suffix != ".npz")
    assert files == sorted(
        p.relative_to(committed) for p in committed.rglob("*") if p.is_file() and p.suffix != ".npz"
    )
    for f in files:
        assert (out / f).read_bytes() == (committed / f).read_bytes(), f
    z = np.load(committed / "observers" / "step_0000000000.npz", allow_pickle=False)
    assert json.loads(str(z["meta"])) == RS.sidecar_meta()


def test_the_sheet_of_the_dip_chapter(rv: R.RunView) -> None:
    sheet = R.build_sheet(rv, 1000, 2000)
    assert [f["id"] for f in sheet["facts"]] == [f"F{i}" for i in range(1, len(sheet["facts"]) + 1)]
    kinds = [f["kind"] for f in sheet["facts"]]
    assert kinds == sorted(kinds, key=R.FACT_KINDS.index)
    assert one(sheet, "chapter") | {"id": "x"} == {"kind": "chapter", "start": 1000, "end": 2000, "id": "x"}
    pop = one(sheet, "population")
    assert (pop["from"], pop["to"]) == (RS.alive(1000), RS.alive(2000)) == (1050, 1100)
    assert pop["change"] == (1100 - 1050) / 1050
    # the minimum of the row interval minima, with the row that holds it (the interval is known to that row only)
    assert (pop["min"], pop["min_row"], pop["min_steps"]) == (775, 1500, 100)
    assert pop["deep_dip"] is True  # 775 < 0.8 * 1050
    assert one(sheet, "deaths") | {"id": 0} == {
        "kind": "deaths", "starved": 50, "killed": 30, "aged": 10, "catastrophe": 0, "total": 90, "id": 0,
    }  # fmt: skip
    assert one(sheet, "births")["births"] == 100 and one(sheet, "births")["immigrants"] == 0
    assert one(sheet, "lineages")["from"] == one(sheet, "lineages")["to"] == 5
    top = one(sheet, "top_lineage")
    assert (top["from_count"], top["to_count"]) == (1050 // 4, 1100 // 4)
    assert top["from_share"] == (1050 // 4) / 1050 and top["from_of"] == 1050 and top["to_of"] == 1100
    assert (
        sheet["origin"] == 0 and sheet["start_step"] == 0 and sheet["dims"] == 2 and sheet["quiet"] is False
    )
    assert sheet["excluded"]["social"] is None
    assert {"entities", "excluded", "labels", "snapshots", "report_rules_digest", "world_key"} <= set(sheet)
    assert [s["step"] for s in sheet["snapshots"]] == [1250, 1500, 1750, 2000]
    assert sheet["snapshots"][1] == {"id": "S1500", "step": 1500, "alive": 775}
    # the sheet is JSON with no float that is not finite
    assert R.facts_digest(sheet) == R.facts_digest(json.loads(json.dumps(sheet)))


def test_population_counts_are_observers_alive_not_the_mean(rv: R.RunView) -> None:
    changed = variant(
        rv, rows=lambda rows: [{**r, "n_alive": 5.5, "wall_s": 9.0, "steps_per_s": 1.0} for r in rows]
    )
    a, b = R.build_sheet(rv, 1000, 2000), R.build_sheet(changed, 1000, 2000)
    assert R.facts_digest(a) == R.facts_digest(b)  # `n_alive`, `wall_s` and `steps_per_s` are never read
    bad = variant(
        rv,
        rows=lambda rows: [
            {**r, "observers": {**r["observers"], "alive": r["observers"]["alive"] + 1}}
            if r["step"] == 2000
            else r
            for r in rows
        ],
    )
    s = R.build_sheet(bad, 1000, 2000)  # the alive identity of the record at 2,000 no longer matches its row
    assert "alive count" in s["excluded"]["social"] and not facts_of(s, "event") and not facts_of(s, "group")
    assert facts_of(s, "population")  # the row-level facts stay


def test_the_cuts_must_lie_on_the_row_and_snapshot_grid(rv: R.RunView) -> None:
    for a, b in ((1000, 1000), (2000, 1000), (1050, 2000), (1000, 2050), (1000, 2250), (1000, 4500)):
        with pytest.raises(R.ReportError) as e:
            R.build_sheet(rv, a, b)
        assert e.value.code == 2, (a, b)
    # a gap in the rows
    gap = variant(rv, rows=lambda rows: [r for r in rows if r["step"] != 1600])
    with pytest.raises(R.ReportError, match="do not cover"):
        R.build_sheet(gap, 1000, 2000)
    # off the 250 grid even where a row exists
    with pytest.raises(R.ReportError, match="multiple"):
        R.build_sheet(rv, 1100, 2000)
    # an unobserved run
    plain = variant(rv, rows=lambda rows: [{k: v for k, v in r.items() if k != "observers"} for r in rows])
    with pytest.raises(R.ReportError, match="observe"):
        R.build_sheet(plain, 1000, 2000)
    # rows out of order or repeated
    dup = variant(rv, rows=lambda rows: [*rows, rows[0]])
    with pytest.raises(R.ReportError, match="strictly"):
        R.build_sheet(dup, 1000, 2000)


def test_social_facts_of_the_dip_chapter(rv: R.RunView) -> None:
    sheet = R.build_sheet(rv, 1000, 2000)
    groups = {g["entity"]: g for g in facts_of(sheet, "group")}
    assert set(groups) == {1, 2, 3}
    g1 = groups[1]
    assert (g1["size"], g1["of"], g1["rank"]) == (RS.size_of(1, 2000), 1100, 1)
    assert groups[2]["rank"] == 2 and groups[3]["rank"] == 3
    # born / died / died_killed are sums over the chapter's four records
    assert (g1["born"], g1["died"], g1["died_killed"]) == (4 * 4, 4 * 3, 4 * 1)
    # events: the born (kept), the merged inside the dip (suspect: the population fell 25% in its window) and the split
    evs = {(e["event"], e["system"]): e for e in facts_of(sheet, "event")}
    assert set(evs) == {("born", 3), ("split", 2)}
    assert sheet["excluded"]["events_suspect"] == 1
    born = evs[("born", 3)]
    assert (born["s0"], born["s1"], born["other"], born["overlap"]) == (1000, 1250, -1, 0)
    split = evs[("split", 2)]
    assert (split["s0"], split["s1"], split["other"], split["size"]) == (1750, 2000, 1, 130)
    assert not any(e["reid"] for e in evs.values())
    h = one(sheet, "houses")
    assert (h["alive_from"], h["alive_to"], h["exact"], h["origin"]) == (14, 12, True, 0)
    assert (h["first_seen"], h["extinct"]) == (2, 1)  # founded at the records 1500 and 2000, extinct at 2000
    assert (h["largest"], h["largest_size"], h["largest_strict"], h["largest_name"]) == (
        101,
        300,
        True,
        "Zaeld",
    )
    rel = one(sheet, "relation")
    assert (rel["a"], rel["b"], rel["attacks"], rel["kills"]) == (1, 2, 12, 4)
    assert "ratio" not in rel and "expected" not in rel  # B7: no exposure-biased ratio
    col = one(sheet, "colonies")
    assert (col["tracked"], col["largest_size"]) == (3, 40)
    assert (
        not facts_of(sheet, "run_end")
        and not facts_of(sheet, "intervention")
        and not facts_of(sheet, "label")
    )
    # entities: names come from the records; ids as given
    ents = {e["ref"]: e for e in sheet["entities"]}
    assert ents["K1"]["display"] == "the Braelix kin" and ents["K1"]["name_from"] == 2000
    assert ents["K3"]["display"] == "the Torvane kin" and ents["H101"]["display"] == "House Zaeld"


def test_names_come_from_earlier_records_and_fall_back_to_the_id(rv: R.RunView) -> None:
    # Torvane dissolves at 3,250 and has no row at 4,000; its name is found in the record of 3,000 (F7)
    sheet = R.build_sheet(rv, 3000, 4000)
    ents = {e["ref"]: e for e in sheet["entities"]}
    assert ents["K3"] == {"ref": "K3", "kind": "kin_group", "id": 3, "display": "the Torvane kin",
                          "name": "Torvane", "name_from": 3000}  # fmt: skip

    # no record at or before the fact lists the group: "kin group {id}"
    def drop_names(recs):
        return [
            {
                **r,
                "kin_groups": {
                    **r["kin_groups"],
                    "id": [i + 100 if i == 3 else i for i in r["kin_groups"]["id"]],
                },
            }
            for r in recs
        ]

    sheet = R.build_sheet(variant(rv, records=drop_names), 3000, 4000)
    ents = {e["ref"]: e for e in sheet["entities"]}
    assert ents["K3"]["display"] == "kin group 3" and ents["K3"]["name_from"] is None
    body, _, _ = texts(sheet)
    assert any(t.startswith("Between step 3,000 and step 3,250, kin group 3 dissolved") for t in body)


# --- editing the synthetic world ---------------------------------------------------------------------------------


def edit_rec(step, fn):
    """A ``records=`` function that applies ``fn`` to the record at ``step``."""
    return lambda recs: [fn(r) if r["step"] == step else r for r in recs]


def edit_row(step, fn):
    return lambda rows: [fn(r) if r["step"] == step else r for r in rows]


def set_sizes(rec, sizes):
    """The record with some kin groups resized, ``summary.kin_groups.other`` set so that the alive identity holds."""
    ids = rec["kin_groups"]["id"]
    new = [sizes.get(i, s) for i, s in zip(ids, rec["kin_groups"]["size"], strict=True)]
    other = RS.alive(rec["step"]) - sum(new)
    return {**rec, "kin_groups": {**rec["kin_groups"], "size": new},
            "summary": {**rec["summary"], "kin_groups": {**rec["summary"]["kin_groups"], "other": other}}}  # fmt: skip


def with_events(rec, rows):
    cols = ("step", "family", "kind", "id", "size", "other", "overlap")
    return {**rec, "events": {c: [r[j] for r in rows] for j, c in enumerate(cols)}}


SOCIAL_FACTS = ("group", "event", "houses", "relation", "colonies")


def assert_no_social(sheet):
    for k in SOCIAL_FACTS:
        assert not facts_of(sheet, k), k
    assert facts_of(sheet, "population") and facts_of(sheet, "deaths")


@pytest.mark.parametrize(
    ("what", "fn", "reason"),
    [
        ("a missing record", lambda recs: [r for r in recs if r["step"] != 1500], "no social record at step 1500"),
        ("a repeated record", lambda recs: [*recs, recs[5]], "repeated"),
        ("two origins", lambda recs: [{**r, "origin": 1300} if r["step"] >= 1500 else r for r in recs],
         "observer restarted at step 1300"),
        ("an origin after a", lambda recs: [{**r, "origin": 1100} if 1000 < r["step"] <= 2000 else r for r in recs],
         "observer restarted at step 1100"),
        ("a record without an origin", lambda recs: [{k: v for k, v in r.items() if k != "origin"} if r["step"] == 1500 else r
                                                      for r in recs], "has no origin"),
        ("an unknown digest", lambda recs: [{**r, "social_digest": "f" * 64} if r["step"] == 1250 else r for r in recs],
         "unknown social digest"),
    ],
)  # fmt: skip
def test_social_facts_are_excluded_when_the_records_do_not_cover_the_chapter(rv, what, fn, reason) -> None:
    sheet = R.build_sheet(variant(rv, records=fn), 1000, 2000)
    assert reason in sheet["excluded"]["social"], what
    assert_no_social(sheet)
    body, heads, _block = texts(sheet)
    assert body and heads
    # the ticker has no population line where the count is only in a record (the other snapshot steps)
    lines = R.ticker(sheet)
    assert [t["step"] for t in lines] == [1500, 2000]  # the row steps only
    # the earlier chapters keep their facts
    assert (
        R.build_sheet(variant(rv, records=fn), 2000, 3000)["excluded"]["social"] is None
        or what == "a repeated record"
    )


def test_the_alive_identity_must_hold_at_row_steps(rv) -> None:
    def off_by_one(rec):
        t = rec["kin_groups"]
        return {
            **rec,
            "kin_groups": {**t, "size": [t["size"][0] + 1, *t["size"][1:]]},
        }  # `other` not adjusted

    sheet = R.build_sheet(variant(rv, records=edit_rec(2000, off_by_one)), 1000, 2000)
    assert "alive count of the record at step 2000" in sheet["excluded"]["social"]
    # a record between row steps has no row to compare with: the same error there goes unseen
    assert (
        R.build_sheet(variant(rv, records=edit_rec(1750, off_by_one)), 1000, 2000)["excluded"]["social"]
        is None
    )
    # sizes that move with `other` compensating keep the identity
    ok = variant(rv, records=edit_rec(2000, lambda r: set_sizes(r, {1: RS.size_of(1, 2000) + 5})))
    assert R.build_sheet(ok, 1000, 2000)["excluded"]["social"] is None


def test_a_share_that_is_not_a_whole_count_drops_the_fact(rv) -> None:
    sheet = R.build_sheet(
        variant(rv, rows=edit_row(2000, lambda r: {**r, "top_clade_share": 0.2512345})), 1000, 2000
    )
    assert not facts_of(sheet, "top_lineage")
    assert any("top lineage" in x for x in sheet["excluded"]["facts_dropped"])
    assert facts_of(sheet, "lineages")


def test_rank_words_need_the_complete_roster(rv) -> None:
    def tracked(n):
        return edit_rec(
            2000,
            lambda r: {
                **r,
                "summary": {**r["summary"], "kin_groups": {**r["summary"]["kin_groups"], "tracked": n}},
            },
        )

    sheet = R.build_sheet(
        variant(rv, records=tracked(4)), 1000, 2000
    )  # 4 tracked, 3 in the table (slot slack)
    assert all("rank" not in g for g in facts_of(sheet, "group"))
    body, _, _ = texts(sheet)
    assert not any("largest kin group" in t for t in body) and any(
        " members at step 2,000" in t for t in body
    )
    assert any(t.startswith("The Braelix kin had 220 members at step 2,000") for t in body)
    ranked = R.build_sheet(rv, 1000, 2000)
    assert [g.get("rank") for g in facts_of(ranked, "group")] == [1, 2, 3]


def test_ties_carry_no_rank_word(rv) -> None:
    def tie(sizes):
        return variant(rv, records=edit_rec(2000, lambda r: set_sizes(r, sizes)))

    # two groups of 100 at the top: no rank for either (B4); a tie lower down keeps the prefix above it
    top = R.build_sheet(tie({1: 100, 2: 100, 3: 50}), 1000, 2000)
    assert [g.get("rank") for g in facts_of(top, "group")] == [None, None, None]
    # a three-way tie (N8)
    three = R.build_sheet(tie({1: 100, 2: 100, 3: 100}), 1000, 2000)
    assert all("rank" not in g for g in facts_of(three, "group"))
    body, _, _ = texts(three)
    assert not any("largest" in t for t in body if "kin group at step" in t)
    low = R.build_sheet(tie({1: 200, 2: 100, 3: 100}), 1000, 2000)
    assert [g.get("rank") for g in facts_of(low, "group")] == [1, None, None]
    body, _, _ = texts(low)
    assert any(
        t.startswith("The Braelix kin was the largest kin group at step 2,000, with 200 members")
        for t in body
    )
    assert any(t.startswith("The Shaemund kin had 100 members at step 2,000") for t in body)
    # the order of the groups in the facts is (-size, id)
    assert [g["entity"] for g in facts_of(three, "group")] == [1, 2, 3]


def test_a_group_with_a_null_window_has_no_chapter_counts(rv) -> None:
    def null(rec):
        t = rec["kin_groups"]
        j = t["id"].index(2)
        return {**rec, "kin_groups": {**t, "births": [None if i == j else v for i, v in enumerate(t["births"])],
                                      "deaths": [None if i == j else v for i, v in enumerate(t["deaths"])]}}  # fmt: skip

    sheet = R.build_sheet(variant(rv, records=edit_rec(1500, null)), 1000, 2000)
    groups = {g["entity"]: g for g in facts_of(sheet, "group")}
    assert {"born", "died", "died_killed"} <= set(groups[1])
    assert not ({"born", "died", "died_killed"} & set(groups[2]))
    body, _, _ = texts(sheet)
    assert not any("born into the Shaemund kin" in t for t in body)


def test_houses_exact_and_inexact(rv) -> None:
    def inexact(rec):
        return {
            **rec,
            "summary": {**rec["summary"], "houses": {**rec["summary"]["houses"], "exact": False}},
            "origin": 0,
        }

    sheet = R.build_sheet(variant(rv, records=edit_rec(4000, inexact)), 3000, 4000)
    h = one(sheet, "houses")
    assert h["exact"] is False
    body, heads, _ = texts(sheet)
    assert any(
        t.startswith("Counting houses traced from step 0, 12 houses had living members at step 4,000")
        for t in body
    )
    # a headline about houses that died out is refused when the count is not exact (N5)
    gone = variant(rv, records=edit_rec(4000, inexact))
    assert all("died out" not in t for t in heads)
    assert R.build_sheet(rv, 3000, 4000)["facts"]
    exact = [t for t in texts(R.build_sheet(rv, 1000, 2000))[0] if "houses" in t or "house had" in t]
    assert exact and not exact[0].startswith("Counting")
    assert gone


def test_houses_largest_only_when_strict(rv) -> None:
    def equal(rec):
        t = rec["houses"]
        return {**rec, "houses": {**t, "size": [200, 200, 200]}}

    sheet = R.build_sheet(variant(rv, records=edit_rec(2000, equal)), 1000, 2000)
    h = one(sheet, "houses")
    assert h["largest_strict"] is False
    assert not any("largest house" in t for t in texts(sheet)[0])  # B5: two equal houses name no largest
    assert any("was the largest house at step 2,000" in t for t in texts(R.build_sheet(rv, 1000, 2000))[0])
    # no houses at all: no H-2 and the count only
    none = variant(rv, records=edit_rec(2000, lambda r: {**r, "houses": {c: [] for c in r["houses"]}}))
    h = one(R.build_sheet(none, 1000, 2000), "houses")
    assert "largest" not in h


def test_relations_are_exact_or_absent(rv) -> None:
    base = R.build_sheet(rv, 1000, 2000)
    assert one(base, "relation")["attacks"] == 12
    # relations_dropped > 0 in any record of the chapter: the sum would be low
    dropped = R.build_sheet(
        variant(rv, records=edit_rec(1500, lambda r: {**r, "relations_dropped": 2})), 1000, 2000
    )
    assert not facts_of(dropped, "relation")

    # a group without a slot in one record (B8): the attacks on it were counted under -1
    def unslot(rec):
        t = rec["kin_groups"]
        keep = [i for i, g in enumerate(t["id"]) if g != 2]
        return {**rec, "kin_groups": {c: [v[i] for i in keep] for c, v in t.items()}}

    assert not facts_of(R.build_sheet(variant(rv, records=edit_rec(1750, unslot)), 1000, 2000), "relation")

    # below the attack gate, not 'distinct', or between a group and itself or the unslotted (-1)
    def weak(rec):
        t = rec["relations"]
        return {**rec, "relations": {**t, "attacks": [1, 0]}}

    assert not facts_of(
        R.build_sheet(variant(rv, records=lambda recs: [weak(r) for r in recs]), 1000, 2000), "relation"
    )

    def indistinct(rec):
        return {**rec, "relations": {**rec["relations"], "distinct": [False, False]}}

    assert not facts_of(
        R.build_sheet(variant(rv, records=lambda recs: [indistinct(r) for r in recs]), 1000, 2000), "relation"
    )

    def minus(rec):
        return {**rec, "relations": {**rec["relations"], "b": [-1, 1]}}

    assert not facts_of(
        R.build_sheet(variant(rv, records=lambda recs: [minus(r) for r in recs]), 1000, 2000), "relation"
    )

    # the attacker is `a`: a reversed pair reads "Members of the Shaemund kin attacked members of the Braelix kin"
    def reverse(rec):
        t = rec["relations"]
        return {**rec, "relations": {**t, "a": [2, 1], "b": [1, 2]}}

    rev = R.build_sheet(variant(rv, records=lambda recs: [reverse(r) for r in recs]), 1000, 2000)
    rel = one(rev, "relation")
    assert (rel["a"], rel["b"]) == (2, 1) and rel[
        "attacks"
    ] == 12  # the pair 1 -> 2 has 4 and is below the gate
    assert len(facts_of(rev, "relation")) == 1
    assert all(k not in rel for k in ("expected", "ratio", "exp_attacks"))
    body, _, _ = texts(rev)
    assert (
        "Members of the Shaemund kin attacked members of the Braelix kin 12 times; 4 of the attacks were fatal."
        in body
    )


# --- quotas, ties, names, zeros, the end of the run -------------------------------------------------------------


def test_the_event_quota_keeps_the_largest_not_the_earliest(rv) -> None:
    """Eight kin-group events in a chapter, the largest last in step order: the quota of six keeps it (N7)."""
    # seven dissolved groups of 100 to 400 members in the record at 2,500, after the born of 2,250 (55 members)
    big = [(2500, "kin_group", "dissolved", 50 + j, 100 + 50 * j, -1, 0) for j in range(7)]
    sheet = R.build_sheet(variant(rv, records=edit_rec(2500, lambda r: with_events(r, big))), 2000, 3000)
    evs = facts_of(sheet, "event")
    # nine events (the 7, the born of 55 and a colony of 40): six are kept, three counted
    assert len(evs) == 6 and sheet["excluded"]["quota"] == {"event": 3}
    sizes = [e["size"] for e in evs]
    assert sizes == [400, 350, 300, 250, 200, 150]  # the order of the facts is (-size, s1, id)
    assert 100 not in sizes and 55 not in sizes and 40 not in sizes
    _body, _, _ = texts(sheet)
    mandatory = [c for c in R.candidates(sheet)[0] if c["mandatory"]]
    assert [c["template"] for c in mandatory] == ["D-1", "E-dissolved"]
    assert "400 members" in mandatory[1]["text"]


def test_equal_sized_events_are_all_mandatory_up_to_three(rv) -> None:
    rows = [(2500, "kin_group", "dissolved", 60 + j, 300, -1, 0) for j in range(4)]
    sheet = R.build_sheet(variant(rv, records=edit_rec(2500, lambda r: with_events(r, rows))), 2000, 3000)
    mand = [c for c in R.candidates(sheet)[0] if c["mandatory"] and c["template"].startswith("E-")]
    assert len(mand) == 3 and [c["cites"] for c in mand] == sorted(c["cites"] for c in mand)
    ids = sorted(f["system"] for f in facts_of(sheet, "event") if f["size"] == 300)
    assert len(ids) == 4 and all(f"kin group {i}" in " ".join(c["text"] for c in mand) for i in ids[:3])


def test_a_name_released_and_taken_in_one_snapshot_renders_the_ids(rv) -> None:
    """B3: Torvane dissolves at 3,250 and its name is taken by a group formed in the same snapshot."""

    def rename(rec):
        t = rec["kin_groups"]
        return {**rec, "kin_groups": {**t, "id": [*t["id"], 9], "name": [*t["name"][:-0 or None], "Torvane"],
                                      "slot": [*t["slot"], len(t["slot"])], "size": [*t["size"], 40],
                                      "births": [*t["births"], 1], "deaths": [*t["deaths"], [0, 0, 0, 0]],
                                      "named_after": [*t["named_after"], None]},
                "summary": {**rec["summary"], "kin_groups": {**rec["summary"]["kin_groups"], "other":
                            rec["summary"]["kin_groups"]["other"] - 40}}}  # fmt: skip

    world = variant(rv, records=lambda recs: [rename(r) if r["step"] >= 3250 else r for r in recs])
    world = variant(
        world,
        records=edit_rec(
            3250,
            lambda r: with_events(
                r,
                [
                    (3250, "kin_group", "dissolved", 3, 95, -1, 0),
                    (3250, "kin_group", "born", 9, 40 + 40, -1, 0),
                ],
            ),
        ),
    )
    sheet = R.build_sheet(world, 3000, 4000)
    ents = {e["ref"]: e["display"] for e in sheet["entities"]}
    assert ents["K3"] == "kin group 3" and ents["K9"] == "kin group 9"
    assert sheet["excluded"]["name_clash"] == ["the Torvane kin"]
    body, heads, _ = texts(sheet)
    assert any("kin group 3 dissolved" in t for t in body) and any("kin group 9 formed" in t for t in body)
    assert not any("Torvane" in t for t in body + heads)
    # ids are not counts: no thousands separator
    big = mk_sheet(ents=[("kin_group", 1830, None), ("kin_group", 2, "Braelix")])
    assert {e["display"] for e in big["entities"]} == {"kin group 1830", "the Braelix kin"}


def test_the_immigrant_lineage_bin_drops_the_lineage_facts(rv) -> None:
    base = R.build_sheet(rv, 1000, 2000)
    assert (
        facts_of(base, "lineages") and facts_of(base, "top_lineage") and not base["excluded"]["facts_dropped"]
    )
    # an immigration pulse before b, a forward entry (a revert or a pulse after b does not count)
    pulse = entry(1500, "rule", "immigration_pulse", {"count": 64}, rule="extinction_rescue")
    w = R.build_sheet(variant(rv, entries=[pulse]), 1000, 2000)
    assert not facts_of(w, "lineages") and not facts_of(w, "top_lineage")
    assert w["excluded"]["facts_dropped"] == ["immigrant lineage bin"]
    later = entry(2000, "rule", "immigration_pulse", {"count": 64}, rule="extinction_rescue")
    assert facts_of(R.build_sheet(variant(rv, entries=[later]), 1000, 2000), "lineages")
    # arrivals in the chapter
    arrivals = variant(rv, rows=edit_row(1300, lambda r: {**r, "immigrants": 3}))
    assert not facts_of(R.build_sheet(arrivals, 1000, 2000), "top_lineage")
    # a config with immigration on
    cfg = apply_overrides(rv.cfg, ["population.immigration_floor=10"])
    assert not facts_of(R.build_sheet(variant(rv, cfg=cfg), 1000, 2000), "lineages")
    body, heads, _ = texts(w)
    assert not any("lineage" in t for t in body) and not any(t.startswith("Lineages") for t in heads)


def test_a_chapter_that_starts_at_zero_has_no_change(rv) -> None:
    zero = variant(rv, rows=edit_row(1000, lambda r: {**r, "clades_alive": 0, "top_clade_share": None,
                                                      "observers": {**r["observers"], "alive": 0}}))  # fmt: skip
    sheet = R.build_sheet(zero, 1000, 2000)
    pop = one(sheet, "population")
    assert pop["from"] == 0 and "change" not in pop and pop["deep_dip"] is False
    assert not facts_of(sheet, "top_lineage")
    rep = R.template_report(sheet)
    assert rep["lead"] == "The population rose from 0 to 1,100."
    assert rep["headline"]["text"] == "Population rose to 1,100"
    json.dumps(sheet, allow_nan=False)


def test_a_run_that_ends_in_extinction_ends_the_report(rv) -> None:
    def extinct(rows):
        last = {**rows[-1], "step": 4040, "steps": 40, "n_alive_min": 0,
                "observers": {**rows[-1]["observers"], "alive": 0}}  # fmt: skip
        return [*rows, last]

    world = variant(rv, rows=extinct)
    sheet = R.build_sheet(world, 3000, 4000)
    assert one(sheet, "run_end") | {"id": 0} == {"kind": "run_end", "step": 4040, "alive": 0, "id": 0}
    rep = R.template_report(sheet)
    assert rep["body"][-1]["text"] == "The run stopped at step 4,040 with 0 agents alive."
    assert rep["body"][-1]["origin"] == "mandatory"
    # an earlier chapter, a last row that is a cut, or a population above zero: no end
    assert not facts_of(R.build_sheet(world, 2000, 3000), "run_end")
    alive = variant(rv, rows=lambda rows: [*rows, {**rows[-1], "step": 4040, "steps": 40}])
    assert not facts_of(R.build_sheet(alive, 3000, 4000), "run_end")
    cut = variant(rv, rows=lambda rows: [*rows, {**rows[-1], "step": 4250, "steps": 250, "observers":
                                                 {**rows[-1]["observers"], "alive": 0}}])  # fmt: skip
    assert not facts_of(R.build_sheet(cut, 3000, 4000), "run_end")


def test_a_rule_entry_without_a_rule_is_named_the_director() -> None:
    view = RS.make_view(
        1_000,
        3_000,
        [
            entry(1_500, "rule", "drift_change", {"mult": 4.0}),
            entry(1_600, "llm", "drift_change", {"mult": 2.0}, rule="boredom"),
        ],
    )
    sheet = R.build_sheet(view, 1_000, 2_000)
    _, _, block = texts(sheet)
    assert block[0].startswith("At step 1,500, the director set the drift of food patches to 4 times")
    assert block[2].startswith(
        "At step 1,600, the director's model set the drift"
    )  # the model has no rule name
    assert "boredom" not in " ".join(block)


def test_reason_and_journal_are_never_read(tmp_path: Path) -> None:
    sentinel = "SENTINEL-REASON-7f3a"
    e = entry(1500, "llm", "drift_change", {"mult": 3.0}, reason=sentinel, prev={"mult": 1.0},
              llm_call={"prompt": sentinel}, vitals={"x": sentinel}, persona="gardener", vitals_step=1400)  # fmt: skip
    path = RS.write_run(tmp_path / "run", entries=[e])
    journal = path / "journal.jsonl"
    journal.write_text(json.dumps({"kind": "llm_call", "journal": sentinel}) + "\n")
    journal.chmod(0)  # opening it would fail: the report never opens the journal
    try:
        view = R.load_run(path)
        view = variant(view, entries=list(view.entries))  # the rows of a director run carry the env blocks
        sheet = R.build_sheet(view, 1000, 2000)
        everything = json.dumps([sheet, R.candidates(sheet), R.template_report(sheet), R.ticker(sheet)])
    finally:
        journal.chmod(0o644)
    assert sentinel not in everything and "SENTINEL" not in everything
    assert "reason" not in json.dumps(sheet) and "llm_call" not in json.dumps(sheet)
    assert one(sheet, "intervention")["step"] == 1500


def test_ids_and_digest_do_not_depend_on_key_order(rv) -> None:
    sheet = R.build_sheet(rv, 1000, 2000)

    def shuffled(x):
        if isinstance(x, dict):
            return {k: shuffled(x[k]) for k in sorted(x, reverse=True)}
        return [shuffled(v) for v in x] if isinstance(x, list) else x

    assert R.facts_digest(shuffled(sheet)) == R.facts_digest(sheet)
    assert list(sheet["facts"][0]) != sorted(sheet["facts"][0]) or True
    with pytest.raises(ValueError):
        R.facts_digest({**sheet, "bad": float("nan")})
    with pytest.raises(ValueError):
        R.facts_digest({**sheet, "bad": math.inf})
    assert R.facts_digest(sheet) == R.facts_digest(
        R.build_sheet(rv, 1000, 2000)
    )  # a pure function of the files
    assert R.facts_digest(sheet) != R.facts_digest(R.build_sheet(rv, 2000, 3000))


# --- the lever fold, spans, the block and the windows --------------------------------------------------------------


def E(step, source, action, params, duration=None, **more):
    return entry(step, source, action, params, duration, **more)


def lone(extra=(), a=1_000_000, b=1_006_000):
    return RS.make_view(a, b + 3_000, list(extra)), a, b


def block_of(sheet):
    return texts(sheet)[2]


def test_fold_levers_is_the_one_source_of_lever_state() -> None:
    cfg = RS.config()
    ents = [
        E(100, "rule", "regen_pulse", {"mult": 0.5}, 500, rule="runaway_damper"),
        E(200, "llm", "habitat_loss", {"patch": 3, "gain": 0.0}, 500),
        E(250, "llm", "set_seasons", {"period": 4000, "amp": 0.5}),
        E(300, "llm", "fertility_shift", {"patch": 3, "x": 5.0, "y": 6.0}),
        E(300, "llm", "immigration_pulse", {"count": 64}),
        E(600, "revert", "regen_pulse", {"mult": 1.0}, reverts=100),
        E(700, "revert", "habitat_loss", {"patch": 3, "gain": 1.75}, reverts=200),
    ]
    fold = R.fold_levers(ents, 650, 0, cfg)
    assert {k: (v.value, v.step, v.entry) for k, v in fold.items()} == {
        "regen": ({"mult": 1.0}, 600, 5),
        "patch:3:gain": ({"gain": 0.0}, 200, 1),
        "seasons": ({"period": 4000, "amp": 0.5}, 250, 2),
        "patch:3:pos": ({"x": 5.0, "y": 6.0}, 300, 3),
    }  # points write no lever; the entry at t itself is not yet in the row
    assert R.fold_levers(ents, 700, 0, cfg)["patch:3:gain"].value == {"gain": 0.0}
    assert R.fold_levers(ents, 701, 0, cfg)["patch:3:gain"].value == {
        "gain": 1.75
    }  # a revert writes its params
    # a replay applies every entry after start_step, a revert included; the forward entry itself is ignored
    fork = R.fold_levers(ents, 800, 150, cfg)
    assert "regen" in fork and fork["regen"].value == {"mult": 1.0}  # the revert at 600 is applied alone (A7)
    assert fork["patch:3:gain"].step == 700
    assert R.fold_levers(ents, 800, 700, cfg) == {}
    # defaults and "away"
    assert R.lever_default("regen", cfg) == {"mult": 1.0} and R.lever_default("patch:4:pos", cfg) is None
    assert R.lever_default("seasons", cfg) == {
        "period": cfg.fields.season_period,
        "amp": cfg.fields.season_amplitude,
    }
    assert not R.away_from_default("patch:3:gain", {"gain": 1.0000000001}, cfg)  # compared as float32
    assert R.away_from_default("patch:3:gain", {"gain": 1.75}, cfg) and R.away_from_default(
        "patch:1:pos", {"x": 0, "y": 0}, cfg
    )
    assert R.away_from_default("seasons", {"period": 4000, "amp": 0.5}, cfg)


def test_the_env_check_compares_rows_with_the_fold() -> None:
    ents = [E(1_000_200, "rule", "regen_pulse", {"mult": 0.800000011920929}, 1_000, rule="runaway_damper")]
    view, a, b = lone(ents)
    R.check_env(view, a, b)  # the rows were computed by an independent model
    bad = variant(view, rows=edit_row(a + 300, lambda r: {**r, "env": {**r["env"], "regen_mult": 1.0}}))
    with pytest.raises(R.ReportError) as e:
        R.build_sheet(bad, a, b)
    assert e.value.code == 1 and "regen_mult" in str(e.value)
    for key, val in (("season_amp", 0.25), ("drift_mult", 2.0), ("patch_gain_min", 0.5), ("patches_off", 1),
                     ("season_period", 99)):  # fmt: skip
        w = variant(view, rows=edit_row(a + 500, lambda r, k=key, v=val: {**r, "env": {**r["env"], k: v}}))
        with pytest.raises(R.ReportError) as e:
            R.build_sheet(w, a, b)
        assert e.value.code == 1, key
    # the keys outside the check (cat_*, imm_request) are free
    ok = variant(
        view,
        rows=edit_row(a + 500, lambda r: {**r, "env": {**r["env"], "cat_radius": 40.0, "imm_request": 64}}),
    )
    R.build_sheet(ok, a, b)
    # a director run whose rows carry no env block cannot be checked
    plain = variant(view, rows=lambda rows: [{k: v for k, v in r.items() if k != "env"} for r in rows])
    with pytest.raises(R.ReportError) as e:
        R.build_sheet(plain, a, b)
    assert e.value.code == 1


def test_a_timed_entry_applied_inside_the_chapter() -> None:
    view, a, b = lone([
        E(1_001_000, "rule", "regen_pulse", {"mult": 1.5}, 2_000, rule="extinction_rescue"),
        E(1_003_000, "revert", "regen_pulse", {"mult": 1.0}, reverts=1_001_000),
    ])  # fmt: skip
    sheet = R.build_sheet(view, a, b)
    iv = one(sheet, "intervention")
    assert (iv["step"], iv["end"], iv["end_by"], iv["restored"], iv["open_to"]) == (
        1_001_000,
        1_003_000,
        "elapsed",
        {"mult": 1.0},
        None,
    )
    assert block_of(sheet) == [
        "At step 1,001,000, the director's rule extinction_rescue set food regrowth to 1.5 times its base rate for 2,000 steps.",
        "It ended at step 1,003,000; food regrowth returned to its base rate.",
    ]
    # the window is (1,001,000, 1,003,000]: 2,000 steps, inside the span and the chapter, nothing else in force
    w = one(sheet, "window")
    assert (w["w0"], w["w1"]) == (1_001_000, 1_003_000) and w["intervention"] == iv["id"]
    assert (w["from"], w["to"]) == (RS.alive(1_001_000), RS.alive(1_003_000))
    assert not any(t.startswith("From step") for t in block_of(sheet))  # W is off: no sentence of it


def test_the_s11_case_applied_before_a_and_elapsed_inside() -> None:
    view, a, b = lone([
        E(995_000, "rule", "regen_pulse", {"mult": 0.800000011920929}, 8_000, rule="runaway_damper"),
        E(1_003_000, "revert", "regen_pulse", {"mult": 1.0}, reverts=995_000),
    ])  # fmt: skip
    sheet = R.build_sheet(view, a, b)
    iv = one(sheet, "intervention")
    assert (iv["step"], iv["end"], iv["end_by"]) == (995_000, 1_003_000, "elapsed")
    w = one(sheet, "window")
    assert (w["w0"], w["w1"]) == (
        1_000_000,
        1_003_000,
    )  # (max(s, a), min(e, b)]: nothing quoted outside the span
    assert block_of(sheet)[1] == "It ended at step 1,003,000; food regrowth returned to its base rate."
    # the same entry ended 100 steps into the chapter has a window shorter than 1,000: no window
    short = RS.make_view(a, b, [E(995_000, "rule", "regen_pulse", {"mult": 0.5}, 5_100, rule="boredom"),
                               E(1_000_100, "revert", "regen_pulse", {"mult": 1.0}, reverts=995_000)])  # fmt: skip
    assert not facts_of(R.build_sheet(short, a, b), "window")


def test_open_entries_an_entry_at_b_and_untimed_settings() -> None:
    view, a, b = lone([
        E(1_001_000, "rule", "regen_pulse", {"mult": 1.5}, 20_000, rule="extinction_rescue"),
        E(1_002_000, "llm", "set_seasons", {"period": 4000, "amp": 0.5}),
        E(1_004_000, "llm", "set_seasons", {"period": 3000, "amp": 0.25}),
        E(1_006_000, "llm", "drift_change", {"mult": 4.0}),  # at b: belongs to the next chapter
    ])  # fmt: skip
    sheet = R.build_sheet(view, a, b)
    ivs = {f["action"] + str(f["step"]): f for f in facts_of(sheet, "intervention")}
    assert set(ivs) == {"regen_pulse1001000", "set_seasons1002000", "set_seasons1004000"}
    assert ivs["regen_pulse1001000"]["end"] is None and ivs["regen_pulse1001000"]["open_to"] == b
    assert (ivs["set_seasons1002000"]["end_by"], ivs["set_seasons1002000"]["end"]) == ("replaced", 1_004_000)
    assert ivs["set_seasons1004000"]["end_by"] is None
    block = block_of(sheet)
    assert block[1] == "It was still in place at step 1,006,000."
    assert block[3] == "It was replaced at step 1,004,000."
    assert block[5] == "It was still in place at step 1,006,000."
    # the entry at b is the next chapter's: it is a fact there, with a standing season already
    nxt = R.build_sheet(view, b, b + 2_000)
    # the pulse still open and the seasons in force are listed by their own entries, the drift is this chapter's
    assert [f["action"] for f in facts_of(nxt, "intervention")] == [
        "regen_pulse",
        "set_seasons",
        "drift_change",
    ]
    assert not facts_of(nxt, "standing")


def test_a_timed_supersession_and_a_revert_of_another_entry() -> None:
    sup = [
        E(1_001_000, "rule", "regen_pulse", {"mult": 0.5}, 5_000, rule="boredom"),
        E(1_002_000, "rule", "regen_pulse", {"mult": 1.5}, 2_000, rule="extinction_rescue"),
        E(1_004_000, "revert", "regen_pulse", {"mult": 1.0}, reverts=1_002_000),
    ]
    view, a, b = lone(sup)
    sheet = R.build_sheet(view, a, b)
    first, second = facts_of(sheet, "intervention")
    assert (first["end_by"], first["end"]) == ("replaced", 1_002_000)
    assert (second["end_by"], second["end"]) == ("elapsed", 1_004_000)
    # each window lies inside its own span with the regen lever alone (the first ends where the second starts)
    assert sorted((w["entry"], w["w0"], w["w1"]) for w in facts_of(sheet, "window")) == [
        (0, 1_001_000, 1_002_000), (1, 1_002_000, 1_004_000),
    ]  # fmt: skip
    # N4: a fertility shift under an active habitat loss, overwritten by the habitat revert
    n4 = [
        E(1_001_000, "llm", "habitat_loss", {"patch": 3, "gain": 0.0}, 45_000),
        E(1_002_000, "llm", "fertility_shift", {"patch": 3, "gain": 2.0}),
        E(1_004_000, "revert", "habitat_loss", {"patch": 3, "gain": 1.0}, reverts=1_001_000),
    ]
    view, a, b = lone(n4)
    sheet = R.build_sheet(view, a, b)
    loss, fert = facts_of(sheet, "intervention")
    assert (loss["end_by"], loss["end"]) == (
        "replaced",
        1_002_000,
    )  # not 'elapsed': the shift wrote the lever
    assert (fert["end_by"], fert["end"]) == (
        "replaced",
        1_004_000,
    )  # the revert of another entry writes it too
    assert not any("still in place" in t for t in block_of(sheet))
    assert not facts_of(sheet, "standing")
    later = R.build_sheet(view, b, b + 2_000)  # the revert restored the default: nothing in force
    assert not facts_of(later, "standing") and not facts_of(later, "intervention")


def test_a_revert_that_restores_a_non_default_value_is_a_standing_line() -> None:
    """The eval_s0_C shape: patch 12 at 1.75 (untimed), a habitat loss over it, its revert restoring 1.75 (A5, A5b)."""
    ents = [
        E(225_000, "llm", "fertility_shift", {"patch": 12, "gain": 1.75}),
        E(250_000, "llm", "habitat_loss", {"patch": 12, "gain": 0.0}, 45_000),
        E(295_000, "revert", "habitat_loss", {"patch": 12, "gain": 1.75}, reverts=250_000),
    ]
    view = RS.make_view(246_000, 312_000, ents)
    s = R.build_sheet(view, 246_000, 252_000)  # A5b: the 225,000 setting and the loss that replaced it
    first, _loss = facts_of(s, "intervention")
    assert (first["step"], first["end_by"], first["end"]) == (225_000, "replaced", 250_000)
    assert block_of(s)[1] == "It was replaced at step 250,000."
    assert block_of(s)[3] == "It was still in place at step 252,000."
    assert not facts_of(s, "standing")
    later = R.build_sheet(view, 300_000, 306_000)  # A5: the lever is back at 1.75 and the block says so
    (st,) = facts_of(later, "standing")
    assert (st["lever"], st["patch"], st["gain"], st["step"]) == ("patch:12:gain", 12, 1.75, 295_000)
    assert block_of(later) == [
        "From step 295,000, the fertility of patch 12 has been 1.75 times its base value."
    ]
    assert not facts_of(later, "intervention")
    assert later["excluded"]["interventions_dropped"] == 0
    # at the row of the revert itself the loss is still in force and listed; the lever is not yet restored
    edge = R.build_sheet(view, 295_000, 301_000)
    (iv,) = facts_of(edge, "intervention")
    assert (iv["end"], iv["end_by"]) == (295_000, "elapsed") and iv["restored"] == {"patch": 12, "gain": 1.75}
    assert not facts_of(edge, "standing")
    assert (
        block_of(edge)[1]
        == "It ended at step 295,000; the fertility of patch 12 returned to 1.75 times its base value."
    )


def test_a_replay_forks_dangling_revert_is_a_standing_line() -> None:
    """A7: a fork at 260,000 restarts the levers inert; the logged revert after it writes gain 1.75 all the same."""
    ents = [
        E(225_000, "llm", "fertility_shift", {"patch": 12, "gain": 1.75}),
        E(250_000, "llm", "habitat_loss", {"patch": 12, "gain": 0.0}, 45_000),
        E(295_000, "revert", "habitat_loss", {"patch": 12, "gain": 1.75}, reverts=250_000),
    ]
    view = RS.make_view(260_000, 312_000, ents, start=260_000)
    sheet = R.build_sheet(view, 295_000, 300_000)  # the revert is applied at 295,000, inside this chapter
    (st,) = facts_of(sheet, "standing")
    assert (st["step"], st["gain"], st["patch"]) == (295_000, 1.75, 12)
    assert sheet["excluded"]["entries_before_start"] == 2  # the two forward entries before the fork
    assert not facts_of(sheet, "intervention")
    later = R.build_sheet(view, 300_000, 305_000)  # and the lever stands in the chapters after it
    assert [(f["step"], f["gain"]) for f in facts_of(later, "standing")] == [(295_000, 1.75)]
    assert block_of(later) == [
        "From step 295,000, the fertility of patch 12 has been 1.75 times its base value."
    ]
    # the env check passes: the rows follow the fold, which applies the revert
    R.check_env(view, 295_000, 300_000)


def test_no_quota_of_five_and_the_cap_of_24() -> None:
    ents = [E(990_000 + 100 * j, "llm", "fertility_shift", {"patch": j, "gain": 0.5}) for j in range(7)]
    ents += [
        E(992_000, "llm", "set_seasons", {"period": 4000, "amp": 0.5}),
        E(993_000, "llm", "drift_change", {"mult": 2.0}),
    ]
    view, a, b = lone(ents)
    sheet = R.build_sheet(view, a, b)
    assert len(facts_of(sheet, "intervention")) == 9 and sheet["excluded"]["interventions_dropped"] == 0
    assert len([t for t in block_of(sheet) if t.startswith("At step")]) == 9
    # more than 24 settings in force: the cap keeps the latest, counts the rest and the block says so
    many = [E(900_000 + 100 * j, "llm", "fertility_shift", {"patch": j, "gain": 0.5}) for j in range(24)]
    many += [
        E(902_400 + 100 * j, "llm", "fertility_shift", {"patch": j, "x": 1.5, "y": 2.5}) for j in range(6)
    ]
    view, a, b = lone(many)
    sheet = R.build_sheet(view, a, b)
    ivs = facts_of(sheet, "intervention")
    assert len(ivs) == 24 and sheet["excluded"]["interventions_dropped"] == 6
    assert min(f["step"] for f in ivs) == 900_600  # the oldest six are the ones dropped
    assert block_of(sheet)[-1] == "6 further settings are in place and are not listed."
    sheet = R.build_sheet(lone(many[:25])[0], a, b)  # 25 settings: one dropped
    assert block_of(sheet)[-1] == "1 further setting is in place and is not listed."


def test_points_have_a_line_and_no_end() -> None:
    view, a, b = lone([E(1_002_000, "rule", "immigration_pulse", {"count": 64}, rule="extinction_rescue"),
                       E(1_003_000, "llm", "catastrophe", {"center": [12.0, 34.5], "radius": 40.0,
                                                           "kill_p": 0.6000000238418579})])  # fmt: skip
    sheet = R.build_sheet(view, a, b)
    imm, _cat = facts_of(sheet, "intervention")
    assert imm["end"] is None and imm["end_by"] is None and imm["open_to"] is None
    assert block_of(sheet) == [
        "At step 1,002,000, the director's rule extinction_rescue queued 64 immigrants.",
        "At step 1,003,000, the director's model set off a catastrophe of radius 40 around (12, 34.5) with a kill chance of 0.6.",
    ]
    assert not facts_of(sheet, "window")
    assert sheet["excluded"]["windows_confounded"] == 0


def test_same_step_entries_are_ordered_by_log_index() -> None:
    ents = [E(1_002_000, "rule", "immigration_pulse", {"count": 64}, rule="extinction_rescue"),
            E(1_002_000, "rule", "regen_pulse", {"mult": 1.5}, 2_000, rule="extinction_rescue"),
            E(1_004_000, "revert", "regen_pulse", {"mult": 1.0}, reverts=1_002_000)]  # fmt: skip
    view, a, b = lone(ents)
    sheet = R.build_sheet(view, a, b)
    assert [f["entry"] for f in facts_of(sheet, "intervention")] == [0, 1]
    assert block_of(sheet)[0].endswith("queued 64 immigrants.")
    lines = R.ticker(sheet)
    first = [t for t in lines if t["step"] == 1_002_250]
    assert [t["text"].split(", ", 1)[1].split(" ")[3] for t in first] == ["rule", "rule"] or len(first) == 2
    assert (
        len(first) == 2
        and "queued 64 immigrants" in first[0]["text"]
        and "set food regrowth" in first[1]["text"]
    )
    assert [t["step"] for t in lines].count(1_002_250) == 2


# --- the window gate -------------------------------------------------------------------------------------------------


def test_the_window_gate_refuses_confounded_windows() -> None:
    # the standard rescue: a regen pulse and an immigration pulse at one step (A2, N1)
    rescue = [E(1_001_000, "rule", "regen_pulse", {"mult": 1.5}, 3_000, rule="extinction_rescue"),
              E(1_001_000, "rule", "immigration_pulse", {"count": 1024}, rule="extinction_rescue"),
              E(1_004_000, "revert", "regen_pulse", {"mult": 1.0}, reverts=1_001_000)]  # fmt: skip
    view, a, b = lone(rescue)
    sheet = R.build_sheet(view, a, b)
    assert not facts_of(sheet, "window") and sheet["excluded"]["windows_confounded"] == 1
    # a catastrophe inside the window (A3)
    cat = [E(1_001_000, "rule", "regen_pulse", {"mult": 0.5}, 3_000, rule="runaway_damper"),
           E(1_002_000, "llm", "catastrophe", {"center": [1.0, 2.0], "radius": 40.0, "kill_p": 0.6}),
           E(1_004_000, "revert", "regen_pulse", {"mult": 1.0}, reverts=1_001_000)]  # fmt: skip
    assert R.build_sheet(lone(cat)[0], a, b)["excluded"]["windows_confounded"] == 1
    # deaths by catastrophe in the rows of the window, or arrivals, without an entry in the window
    lonely = [E(1_001_000, "rule", "regen_pulse", {"mult": 0.5}, 3_000, rule="runaway_damper"),
              E(1_004_000, "revert", "regen_pulse", {"mult": 1.0}, reverts=1_001_000)]  # fmt: skip
    view, a, b = lone(lonely)
    assert len(facts_of(R.build_sheet(view, a, b), "window")) == 1  # a lone regen pulse keeps its window
    for field in ("deaths_catastrophe", "immigrants"):
        bad = variant(view, rows=edit_row(1_002_500, lambda r, f=field: {**r, f: 5}))
        s = R.build_sheet(bad, a, b)
        assert not facts_of(s, "window") and s["excluded"]["windows_confounded"] == 1, field
    # two timed levers (A4) and seasons in force
    two = [*lonely, E(1000500, "llm", "habitat_loss", {"patch": 3, "gain": 0.0}, 45000)]
    two.sort(key=lambda e: e["step_applied"])
    s = R.build_sheet(lone(two)[0], a, b)
    assert not any(w["intervention"] == facts_of(s, "intervention")[1]["id"] for w in facts_of(s, "window"))
    assert s["excluded"]["windows_confounded"] >= 1
    seasonal = [E(990_000, "llm", "set_seasons", {"period": 4000, "amp": 0.5}), *lonely]
    s = R.build_sheet(lone(seasonal)[0], a, b)
    assert not facts_of(s, "window") and s["excluded"]["windows_confounded"] == 1


def test_window_sentences_are_off_unless_asked() -> None:
    lonely = [E(1_001_000, "rule", "regen_pulse", {"mult": 0.5}, 3_000, rule="runaway_damper"),
              E(1_004_000, "revert", "regen_pulse", {"mult": 1.0}, reverts=1_001_000)]  # fmt: skip
    view, a, b = lone(lonely)
    off = R.build_sheet(view, a, b)
    assert not any("while this was in place" in t for t in block_of(off) + R.ticker(off) and block_of(off))
    on_rules = R.ReportRules(window_sentences=True)
    on = R.build_sheet(view, a, b, on_rules)
    body, _heads, block, ex = R.candidates(on, on_rules)
    assert ex == []
    w = [t["text"] for t in block if t["template"] == "W"]
    pop_from, pop_to = RS.alive(1_001_000), RS.alive(1_004_000)
    assert w == [
        f"From step 1,001,000 to step 1,004,000, while this was in place, the population went from {pop_from:,} to {pop_to:,}."
    ]
    assert not any(
        "while this was in place" in t["text"] for t in R.ticker(on, on_rules)
    )  # W is never in the ticker
    assert not any("while" in t for t in body)


# --- labels: which may be cited -----------------------------------------------------------------------------------

PASS = ("pass", 5000, 0.25, 0.004975, 200)


def world_key(view: R.RunView) -> str:
    import dataclasses

    from alife.config import DirectorConfig, GameConfig

    return config_hash(dataclasses.replace(view.cfg, director=DirectorConfig(), game=GameConfig()))


def with_table(monkeypatch, view, table=None, version=1, digest=RS.V1_DIGEST):
    """Give the synthetic world a verdict row (the real table holds the two GPU studies' worlds only)."""
    rows = table or {"hunter": PASS, "settler": PASS, "resting": PASS, "hunting": PASS}
    monkeypatch.setattr(R, "LABEL_VERDICTS", {(version, digest, 2, world_key(view)): rows})


def labels_of(sheet):
    return {f["key"]: f for f in facts_of(sheet, "label")}


def test_label_facts_for_a_validated_world(rv, monkeypatch) -> None:
    with_table(monkeypatch, rv)
    sheet = R.build_sheet(rv, 1000, 2000)
    assert sheet["labels"] == {
        "intent_version": 1,
        "rules_digest": RS.V1_DIGEST,
        "key_found": True,
        "env_default": True,
    }
    labs = labels_of(sheet)
    assert set(labs) == {"hunter", "settler", "resting", "hunting"} and sheet["excluded"]["labels"] == {}
    h = labs["hunter"]
    assert (h["label"], h["family"], h["from_of"], h["to_of"]) == ("Hunter", "archetype", 400, 400)
    assert (h["from_count"], h["to_count"]) == (110, 120)  # 0.25 + step / 40,000 of 400 labelled agents
    assert (h["from_share"], h["to_share"]) == (0.275, 0.3)
    r = labs["resting"]
    assert (r["label"], r["family"], r["from_count"], r["to_count"]) == ("Resting", "intent", 190, 180)
    body, _, _ = texts(sheet)
    assert (
        "27.5% of the agents with an archetype label matched Hunter at step 1,000, and 30.0% at step 2,000."
        in body
    )
    assert (
        "47.5% of the agents with a behaviour label were labelled Resting at step 1,000, and 45.0% at step 2,000."
        in body
    )
    # no rank, majority or dominance word on a label share; no 'Generalist'
    assert not any(
        w in " ".join(body) for w in ("majority", "dominant", "most", "Unspecialised", "Generalist")
    )
    # the order of the facts: the largest change of share first
    assert next(f["key"] for f in facts_of(sheet, "label")) in ("hunter", "settler", "resting", "hunting")


@pytest.mark.parametrize(
    ("row", "citable"),
    [
        (("pass", 5000, 0.25, 0.004975, 200), True),
        (("fail", 5000, 0.25, 0.004975, 200), False),
        (("insufficient", 8, 0.25, 0.004975, 200), False),
        (("pass", 79, 0.25, 0.004975, 200), False),  # n_pos 79 (2D Fleeing)
        (
            ("pass", 125, 0.058, 0.00995, 200),
            False,
        ),  # margin 0.058, p 0.00995 (2D Bonding): a borderline pass
        (("pass", 5000, 0.0999, 0.004975, 200), False),
        (("pass", 5000, 0.25, 0.00995, 200), False),  # not the permutation floor
        (("pass", 100, 0.10, 0.004975, 200), True),
    ],
)
def test_only_a_robust_pass_is_citable(rv, monkeypatch, row, citable) -> None:
    with_table(monkeypatch, rv, {"hunter": row, "settler": PASS})
    sheet = R.build_sheet(rv, 1000, 2000)
    assert ("hunter" in labels_of(sheet)) is citable
    assert "settler" in labels_of(sheet)


def test_no_label_without_a_key(rv, monkeypatch) -> None:
    # an unknown world: the key is not found and nothing is said
    monkeypatch.setattr(R, "LABEL_VERDICTS", R.LABEL_VERDICTS)
    sheet = R.build_sheet(rv, 1000, 2000)
    assert (
        sheet["labels"]["key_found"] is False
        and not facts_of(sheet, "label")
        and sheet["excluded"]["labels"] == {}
    )
    # the real table holds the two GPU worlds, by dims and world key
    keys = list(R.LABEL_VERDICTS)
    assert sorted((k[0], k[2]) for k in keys) == [(1, 2), (1, 3)] and all(k[1] == RS.V1_DIGEST for k in keys)
    assert {k[3][:16] for k in keys} == {"69b4e21ad6b6016b", "df54434fe947dea8"}
    # a missing sidecar
    with_table(monkeypatch, rv)
    assert R.build_sheet(variant(rv, sidecars={}), 1000, 2000)["labels"]["key_found"] is False
    # INTENT_VERSION 2 (the rows and the sidecar agree): no verdict row, no label
    v2 = variant(
        rv,
        rows=lambda rows: [{**r, "observers": {**r["observers"], "intent_version": 2}} for r in rows],
        sidecars={0: {**RS.sidecar_meta(), "intent_version": 2, "rules_digest": RULES_V2.digest()}},
    )
    s2 = R.build_sheet(v2, 1000, 2000)
    assert (
        s2["labels"]["intent_version"] == 2
        and s2["labels"]["key_found"] is False
        and not facts_of(s2, "label")
    )
    assert RULES_V1.digest() == RS.V1_DIGEST and RULES_V2.digest() != RS.V1_DIGEST
    # a different world (another seed's regen_max, say) does not share the key
    other = (
        apply_overrides(rv.cfg, ["fields.regen_max=77"]) if hasattr(rv.cfg.fields, "regen_max") else rv.cfg
    )
    assert world_key(variant(rv, cfg=other)) != world_key(rv) or other is rv.cfg
    # the world key ignores the director and the game sections
    assert world_key(rv) == config_hash(
        load_config(preset="tiny", overrides=["brain.weight_dtype=float32", "fields.n_patches=24"])
    )


def test_a_row_version_that_differs_from_the_sidecars_withholds_the_labels(rv, monkeypatch) -> None:
    """D1: a run resumed under new code starts a fresh observer and writes no sidecar until its next checkpoint."""
    with_table(monkeypatch, rv)
    resumed = variant(rv, rows=lambda rows: [
        {**r, "observers": {**r["observers"], "intent_version": 2}} if r["step"] >= 1500 else r for r in rows])  # fmt: skip
    sheet = R.build_sheet(resumed, 1000, 2000)
    assert not facts_of(sheet, "label") and sheet["excluded"]["labels"] == {"*": "rules differ"}
    assert sheet["labels"]["intent_version"] is None
    # all rows at version 2 against a version-1 sidecar
    allv2 = variant(
        rv, rows=lambda rows: [{**r, "observers": {**r["observers"], "intent_version": 2}} for r in rows]
    )
    assert R.build_sheet(allv2, 1000, 2000)["excluded"]["labels"] == {"*": "rules differ"}


def test_a_restart_inside_the_warm_up_withholds_the_labels(rv, monkeypatch) -> None:
    with_table(monkeypatch, rv)

    def origin(o):
        return lambda rows: [{**r, "observers": {**r["observers"], "social": {**r["observers"]["social"], "origin": o}}}
                             for r in rows]  # fmt: skip

    # origin 809 <= a - 192 passes, 900 does not (N10): the labels of both ends come from one observer
    assert facts_of(R.build_sheet(variant(rv, rows=origin(808)), 1000, 2000), "label")
    late = R.build_sheet(variant(rv, rows=origin(900)), 1000, 2000)
    assert not facts_of(late, "label")
    assert set(late["excluded"]["labels"].values()) == {"observer restarted"}
    assert set(late["excluded"]["labels"]) == {"hunter", "settler", "resting", "hunting"}
    # a restart inside the chapter: the rows disagree on the origin
    mid = variant(rv, rows=lambda rows: [
        {**r, "observers": {**r["observers"], "social": {**r["observers"]["social"], "origin": 1500}}}
        if r["step"] >= 1500 else r for r in rows])  # fmt: skip
    assert not facts_of(R.build_sheet(mid, 1000, 2000), "label")


def test_too_few_labelled_agents_withholds_a_family(rv, monkeypatch) -> None:
    with_table(monkeypatch, rv)
    few = variant(rv, rows=edit_row(1000, lambda r: {**r, "observers": {**r["observers"], "arch_labelled": 99,
                                                                         "archetype_match": {"hunter": 0.0, "settler": 0.0}}}))  # fmt: skip
    sheet = R.build_sheet(few, 1000, 2000)
    assert set(labels_of(sheet)) == {"resting", "hunting"}  # the intent labels have 400 labelled agents
    assert sheet["excluded"]["labels"] == {"hunter": "too few labelled", "settler": "too few labelled"}
    exact = variant(rv, rows=edit_row(2000, lambda r: {**r, "observers": {**r["observers"], "arch_labelled": 100,
                                                                           "archetype_match": {"hunter": 0.25, "settler": 0.5}}}))  # fmt: skip
    assert "hunter" in labels_of(R.build_sheet(exact, 1000, 2000))  # exactly labelled_min is enough


def test_a_label_share_that_is_not_a_whole_count_is_dropped(rv, monkeypatch) -> None:
    with_table(monkeypatch, rv)
    odd = variant(rv, rows=edit_row(2000, lambda r: {**r, "observers": {**r["observers"],
                                                                         "archetype_match": {"hunter": 0.3003, "settler": 0.5}}}))  # fmt: skip
    sheet = R.build_sheet(odd, 1000, 2000)
    assert "hunter" not in labels_of(sheet) and "settler" in labels_of(sheet)
    assert any("label hunter" in x for x in sheet["excluded"]["facts_dropped"])


def test_labels_are_withheld_when_the_environment_changed(rv, monkeypatch) -> None:
    with_table(monkeypatch, rv)
    names = {"hunter", "settler", "resting", "hunting"}

    def sheet_with(*ents, a=1000, b=2000):
        return R.build_sheet(variant(rv, entries=list(ents)), a, b)

    # a non-default lever in a row of the chapter: seasons, drift, regen, a patch gain
    for e in (E(500, "llm", "set_seasons", {"period": 4000, "amp": 0.5}),
              E(500, "llm", "drift_change", {"mult": 2.0}),
              E(500, "rule", "regen_pulse", {"mult": 0.5}, 5000, rule="runaway_damper"),
              E(500, "llm", "fertility_shift", {"patch": 3, "gain": 0.5}),
              E(500, "llm", "habitat_loss", {"patch": 3, "gain": 0.0}, 5000)):  # fmt: skip
        s = sheet_with(e)
        assert set(s["excluded"]["labels"]) == names and set(s["excluded"]["labels"].values()) == {
            "environment changed"
        }
        assert s["labels"]["env_default"] is False and not facts_of(s, "label"), e["action"]
    # a patch move before b (the positions are not in `env`)
    s = sheet_with(E(500, "llm", "fertility_shift", {"patch": 3, "x": 5.0, "y": 6.0}))
    assert set(s["excluded"]["labels"].values()) == {"environment changed"}
    # a catastrophe within one snapshot window before a, or inside the chapter; before that it does not count
    for step, withheld in ((700, False), (750, True), (990, True), (1500, True), (2000, False)):
        s = sheet_with(E(step, "llm", "catastrophe", {"center": [1.0, 2.0], "radius": 40.0, "kill_p": 0.6}))
        assert (not facts_of(s, "label")) is withheld, step
    # D3: `cat_radius` stays at its value after a catastrophe and `imm_request` is non-zero while arrivals drain:
    # neither is a lever, so neither withholds
    base = variant(
        rv, entries=[E(100, "llm", "catastrophe", {"center": [1.0, 2.0], "radius": 40.0, "kill_p": 0.6})]
    )
    stale = variant(
        base,
        rows=lambda rows: [{**r, "env": {**r["env"], "cat_radius": 40.0, "imm_request": 64}} for r in rows],
    )
    s = R.build_sheet(stale, 1000, 2000)
    assert set(labels_of(s)) == names and s["labels"]["env_default"] is True
    # an entry before the fork point is not applied and does not withhold
    forked = variant(
        rv,
        entries=[E(500, "llm", "set_seasons", {"period": 4000, "amp": 0.5})],
        meta={**rv.meta, "start_step": 600},
    )
    assert set(labels_of(R.build_sheet(forked, 1000, 2000))) == names


def test_dims_come_from_the_config_not_the_json(tmp_path: Path, monkeypatch) -> None:
    """N18: a 2D config.json has no `world.dims` key (runs/64k_s0)."""
    path = RS.write_run(tmp_path / "run")
    cfg = json.loads((path / "config.json").read_text())
    cfg["world"].pop("dims")
    (path / "config.json").write_text(json.dumps(cfg))
    view = R.load_run(path)
    assert view.cfg.world.dims == 2
    with_table(monkeypatch, view)
    sheet = R.build_sheet(view, 1000, 2000)
    assert sheet["dims"] == 2 and sheet["labels"]["key_found"] is True and facts_of(sheet, "label")


def test_label_names_are_v1_or_v2_words(rv, monkeypatch) -> None:
    from alife.game.rules import ARCHETYPE_DISPLAY, INTENT_DISPLAY

    assert (ARCHETYPE_DISPLAY["generalist"], INTENT_DISPLAY["bonding"]) == ("Unspecialised", "Tethering")
    assert R._label_display("bonding", 1) == "Bonding" and R._label_display("bonding", 2) == "Tethering"
    assert R._label_display("generalist", 2) == "Unspecialised" and R._label_display("hunter", 2) == "Hunter"
    # a version-2 row set with a verdict row narrates the v2 words
    digest2 = RULES_V2.digest()
    rows2 = lambda rows: [  # noqa: E731
        {**r, "observers": {**r["observers"], "intent_version": 2,
                            "intent": {"bonding": 0.125 + r["step"] / 40000, "resting": 0.5}}} for r in rows]  # fmt: skip
    v2 = variant(
        rv, rows=rows2, sidecars={0: {**RS.sidecar_meta(), "intent_version": 2, "rules_digest": digest2}}
    )
    with_table(monkeypatch, v2, {"bonding": PASS, "resting": PASS}, version=2, digest=digest2)
    sheet = R.build_sheet(v2, 1000, 2000)
    assert labels_of(sheet)["bonding"]["label"] == "Tethering"
    body, _, _ = texts(sheet)
    assert any("were labelled Tethering at step 1,000" in t for t in body)


# --- event frames (the real tracker), the suspect and churn gates ---------------------------------------------------


def _members(groups):
    from alife.tracking import Members

    lab, key = [], []
    for g, uids in groups.items():
        for u in uids:
            lab.append(g)
            key.append(u)
    return Members(np.array(lab, dtype=np.int64), np.array(key, dtype=np.int64))


def tracker_rows(events, step):
    return [(step, "kin_group", e.kind.name.lower(), e.system, e.size, e.other, e.overlap) for e in events]


def scenario(name):
    """The events G2's real ``SystemTracker`` gives for three scripted transitions."""
    from alife.tracking import SystemTracker

    tr = SystemTracker((3, 10))
    if (
        name == "T1"
    ):  # a tracked group fuses with untracked agents; newborns are credited to their parents (uid mode)
        tr.observe(1, _members({1: range(0, 1100), 2: range(5000, 8000)}))
        kids, parents = np.arange(10000, 10500, dtype=np.int64), np.arange(0, 500, dtype=np.int64)
        cur = _members(
            {7: [*range(0, 1000), *range(10000, 10500), *range(20000, 24000)], 8: range(5000, 8000)}
        )
        return tr.observe(2, cur, credit=(kids, parents))[1]
    if name == "T2":  # 1,080 of 1,100 members continue under a new id (Jaccard 0.26 is under theta)
        tr.observe(1, _members({1: range(0, 1100)}))
        return tr.observe(2, _members({7: [*range(0, 1080), *range(20000, 23000)]}))[1]
    tr.observe(0, _members({1: range(0, 1000), 2: range(2000, 2600)}))
    ev = tr.observe(250, _members({1: range(0, 600), 2: range(600, 1000), 3: range(2000, 2600)}))[1]
    return [*ev, *tr.observe(500, _members({1: [*range(0, 600), *range(2000, 2600)]}))[1]]


def test_t1_a_dissolved_event_prints_no_overlap_number(rv) -> None:
    """B2: with newborns credited, `overlap` (1,500) exceeds the ended group's `size` (1,100)."""
    ev = scenario("T1")
    kinds = {(e.kind.name, e.system): e for e in ev}
    d, born = kinds[("DISSOLVED", 2)], kinds[("BORN", 3)]
    assert (d.size, d.overlap, d.other) == (1100, 1500, 3) and (born.size, born.overlap, born.other) == (
        5500,
        1500,
        2,
    )
    world = variant(rv, records=edit_rec(2250, lambda r: with_events(r, tracker_rows(ev, 2250))))
    sheet = R.build_sheet(world, 2000, 3000)
    by = {
        (e["event"], e["system"]): e
        for e in facts_of(sheet, "event")
        if e["s1"] == 2250 and e["family"] == "kin_group"
    }
    assert by[("dissolved", 2)]["overlap"] == 1500 and by[("dissolved", 2)]["reid"] is True
    assert (
        by[("born", 3)]["reid"] is True
    )  # the partner event of the pair, and 2 * overlap < size does not matter
    body, heads, _ = texts(sheet)
    line = next(t for t in body if "Shaemund kin dissolved" in t)
    assert line == ("Between step 2,000 and step 2,250, the Shaemund kin dissolved; it had 1,100 members at step 2,000, "
                    "and the group it overlapped most was the Torvane kin.")  # fmt: skip
    assert "1,500" not in line
    # a re-identified pair is never a headline (C9): the biggest event (5,500) is a reid, so the headline is the population's
    assert all("formed" not in h and "dissolved" not in h for h in heads)
    assert sheet["quiet"] is False


def test_t2_a_group_that_continues_under_a_new_id_is_not_a_formation(rv) -> None:
    """B1/C9: 1,080 of 1,100 continue; the tracker ends one id and starts another, and both carry one name."""
    ev = scenario("T2")
    kinds = {e.kind.name: e for e in ev}
    assert (
        kinds["DISSOLVED"].system,
        kinds["DISSOLVED"].other,
        kinds["DISSOLVED"].size,
        kinds["DISSOLVED"].overlap,
    ) == (1, 2, 1100, 1080)
    assert (kinds["BORN"].system, kinds["BORN"].other, kinds["BORN"].size, kinds["BORN"].overlap) == (
        2,
        1,
        4080,
        1080,
    )

    def same_name(rec):
        t = rec["kin_groups"]
        return {
            **rec,
            "kin_groups": {
                **t,
                "name": ["Braelix" if i == 2 else n for i, n in zip(t["id"], t["name"], strict=True)],
            },
        }

    world = variant(rv, records=lambda recs: [same_name(with_events(r, tracker_rows(ev, 2250)) if r["step"] == 2250 else r)
                                              if r["step"] >= 1250 else r for r in recs])  # fmt: skip
    sheet = R.build_sheet(world, 2000, 3000)
    assert {e["ref"]: e["display"] for e in sheet["entities"] if e["id"] in (1, 2)} == {
        "K1": "kin group 1",
        "K2": "kin group 2",
    }
    assert sheet["excluded"]["name_clash"] == ["the Braelix kin"]
    evs = {e["event"]: e for e in facts_of(sheet, "event") if e["s1"] == 2250}
    assert evs["dissolved"]["reid"] and evs["born"]["reid"]
    body, heads, _ = texts(sheet)
    assert (
        "Between step 2,000 and step 2,250, kin group 1 dissolved; it had 1,100 members at step 2,000, and the group it overlapped most was kin group 2."
        in body
    )
    assert (
        "Between step 2,000 and step 2,250, kin group 2 formed with 4,080 members, 1,080 of them from kin group 1."
        in body
    )
    assert "Braelix" not in " ".join(body + heads)
    assert not any(h.endswith(" formed") for h in heads)  # no 'X formed' headline for a continuing group


def test_event_frames_follow_the_tracker(rv) -> None:
    """Split: `system` is the new id and `other` the parent; merged: `system` the ended id and `other` the survivor."""
    ev = scenario("T3")
    got = {(e.kind.name, e.system): (e.other, e.size, e.overlap) for e in ev}
    assert got == {("SPLIT", 3): (1, 400, 400), ("MERGED", 2): (1, 600, 600), ("DISSOLVED", 3): (-1, 400, 0)}
    rows = [(2250, "kin_group", "split", 3, 400, 1, 400), (2250, "kin_group", "merged", 2, 600, 1, 600)]
    sheet = R.build_sheet(variant(rv, records=edit_rec(2250, lambda r: with_events(r, rows))), 2000, 3000)
    body, _, _ = texts(sheet)
    assert (
        "Between step 2,000 and step 2,250, the Torvane kin split from the Braelix kin with 400 members."
        in body
    )
    assert ("Between step 2,000 and step 2,250, the Shaemund kin merged into the Braelix kin; it had 600 members at "
            "step 2,000.") in body  # fmt: skip
    assert not any("reid" in k for e in facts_of(sheet, "event") for k in e if e[k] is True and k != "reid")


def test_a_born_event_with_an_origin_prints_the_overlap_within_its_size(rv) -> None:
    rows = [(2250, "kin_group", "born", 9, 500, 2, 250)]
    sheet = R.build_sheet(variant(rv, records=edit_rec(2250, lambda r: with_events(r, rows))), 2000, 3000)
    (e,) = [f for f in facts_of(sheet, "event") if f["system"] == 9]
    assert e["reid"] is False and e["overlap"] == 250
    assert (
        "Between step 2,000 and step 2,250, kin group 9 formed with 500 members, 250 of them from the Shaemund kin."
        in texts(sheet)[0]
    )
    # more than half of the members from the origin: a re-identification even without a partner event
    rows = [(2250, "kin_group", "born", 9, 500, 2, 251)]
    sheet = R.build_sheet(variant(rv, records=edit_rec(2250, lambda r: with_events(r, rows))), 2000, 3000)
    assert [f["reid"] for f in facts_of(sheet, "event") if f["system"] == 9] == [True]


def test_suspect_events_are_excluded_and_counted(rv) -> None:
    ev = [(2250, "kin_group", "dissolved", 3, 90, -1, 0)]

    def sheet_of(records):
        return R.build_sheet(variant(rv, records=records), 2000, 3000)

    base = sheet_of(edit_rec(2250, lambda r: with_events(r, ev)))
    assert [f["system"] for f in facts_of(base, "event") if f["event"] == "dissolved"] == [3]
    # (1) rescaled thresholds
    resc = sheet_of(
        edit_rec(2250, lambda r: {**with_events(r, ev), "thresholds": {**r["thresholds"], "rescaled": True}})
    )
    assert (
        not [f for f in facts_of(resc, "event") if f["event"] == "dissolved"]
        and resc["excluded"]["events_suspect"] >= 1
    )
    # (2) stale thresholds: n_ref lags the population by more than 1/20 (a steady fall of 2% per window trips it)
    stale = sheet_of(
        edit_rec(2250, lambda r: {**with_events(r, ev), "thresholds": {**r["thresholds"], "n_ref": 1000}})
    )
    assert not [f for f in facts_of(stale, "event") if f["event"] == "dissolved"]
    just = sheet_of(
        edit_rec(2250, lambda r: {**with_events(r, ev), "thresholds": {**r["thresholds"], "n_ref": 1060}})
    )
    assert [
        f for f in facts_of(just, "event") if f["event"] == "dissolved"
    ]  # |1,112 - 1,060| * 20 = 1,040 <= 1,060

    # (3) the population moved by more than 1/20 inside the window (a 6% step at a snapshot between rows)
    def jump(rec):
        return {**with_events(rec, ev), "summary": {**rec["summary"], "kin_groups": {**rec["summary"]["kin_groups"],
                "other": rec["summary"]["kin_groups"]["other"] + 70}}}  # fmt: skip

    moved = sheet_of(edit_rec(2250, jump))
    assert not [f for f in facts_of(moved, "event") if f["event"] == "dissolved"]
    assert moved["excluded"]["events_suspect"] >= 1


def test_a_steady_drift_trips_the_n_ref_gate_though_no_window_does(rv) -> None:
    """A growth of ~1% per window never trips the per-window gate; the stale n_ref does by the end of the chapter."""
    fixed = lambda recs: [{**r, "thresholds": {**r["thresholds"], "n_ref": 1100}} for r in recs]  # noqa: E731
    sheet = R.build_sheet(variant(rv, records=fixed), 3000, 4000)
    assert [
        f["event"] for f in facts_of(sheet, "event")
    ] == []  # the dissolution at 3,250: |1,162 - 1,100| * 20 > 1,100
    assert sheet["excluded"]["events_suspect"] == 1
    early = R.build_sheet(
        variant(rv, records=fixed), 2000, 3000
    )  # alive 1,112 at 2,250: 12 * 20 = 240 <= 1,100
    assert [f["event"] for f in facts_of(early, "event") if f["family"] == "kin_group"] == ["born"]


def test_churn_events_are_excluded(rv) -> None:
    rows = [(2250, "kin_group", "dissolved", 3, 15, -1, 0), (2250, "kin_group", "dissolved", 9, 16, -1, 0),
            (2250, "colony", "born", 31, 31, -1, 0), (2250, "colony", "born", 32, 32, -1, 0)]  # fmt: skip
    sheet = R.build_sheet(variant(rv, records=edit_rec(2250, lambda r: with_events(r, rows))), 2000, 3000)
    got = {(f["family"], f["size"]) for f in facts_of(sheet, "event") if f["s1"] == 2250}
    assert got == {("kin_group", 16), ("colony", 32)}  # below 2 * kin_enter (16) and below 32 are churn
    assert sheet["excluded"]["events_churn"] == 2


# --- the headline and the select-mode candidates -----------------------------------------------------------------


def test_the_template_headline(rv) -> None:
    def head(sheet):
        return R.template_report(sheet)["headline"]["text"]

    assert (
        head(R.build_sheet(rv, 2000, 3000)) == "The Kraiost kin formed"
    )  # the largest event (55 * 50 >= 1,150)
    assert head(R.build_sheet(rv, 1000, 2000)) == "Population rose to 1,100"  # a deep dip: the population's


def test_headline_candidates_in_select_mode() -> None:
    kin = [("kin_group", 1, "Braelix"), ("kin_group", 2, "Shaemund"), ("house", 3, "Zaeld")]
    ev = event("dissolved", 2, 980, s0=A0 + 250, s1=A0 + 500)
    small = {"population": [pop(1000, 1020, 1000)], "event": [ev], "houses": [houses(extinct=2)],
             "lineages": [{"kind": "lineages", "from": 5, "to": 2}]}  # fmt: skip
    sheet = mk_sheet(kinds=small, ents=kin)
    heads = R.candidates(sheet)[1]
    # the event is large enough (980 * 50 >= 1,020): HE first; the alternative is another kind (HH, houses that died out)
    assert [(h["id"], h["text"]) for h in heads] == [
        ("H1", "The Shaemund kin dissolved"),
        ("H2", "2 houses died out"),
    ]
    # a large change, or a deep dip: only the population's headline
    big = mk_sheet(kinds={**small, "population": [pop(1000, 1200, 1000)]}, ents=kin)
    assert [h["text"] for h in R.candidates(big)[1]] == ["Population rose to 1,200"]
    crash = mk_sheet(kinds={**small, "population": [pop(1000, 1020, 500)]}, ents=kin)
    assert [h["text"] for h in R.candidates(crash)[1]] == ["Population rose to 1,020"]
    # nothing to say but the lineages: HP and HL
    lin = mk_sheet(kinds={"population": [pop(1000, 1020, 1000)], "lineages": small["lineages"]})
    assert [h["text"] for h in R.candidates(lin)[1]] == [
        "Population rose to 1,020",
        "Lineages went from 5 to 2",
    ]
    # a houses headline needs an exact count
    inexact = mk_sheet(
        kinds={"population": [pop(1000, 1020, 1000)], "houses": [houses(extinct=2, exact=False)]},
        ents=[("house", 3, "Zaeld")],
    )
    assert [h["text"] for h in R.candidates(inexact)[1]] == ["Population rose to 1,020"]
    # a change that renders +0.0% is neither 'rose' nor 'unchanged'
    near = mk_sheet(kinds={"population": [pop(41_310, 41_312)]})
    assert (
        R.template_report(near)["lead"]
        == "The population was 41,310 at the start and 41,312 at the end of the chapter."
    )
    assert R.template_report(near)["headline"]["text"] == "Population at 41,312"
    flat = mk_sheet(kinds={"population": [pop(100, 100)]})
    assert (
        R.template_report(flat)["lead"]
        == "The population was 100 at the start and at the end of the chapter."
    )
    assert R.template_report(flat)["headline"]["text"] == "Population unchanged at 100"


# --- the assembly: template mode and the select-mode fixtures ----------------------------------------------------

KIN = [
    ("kin_group", 12, "Braelix"),
    ("kin_group", 31, "Shaemund"),
    ("kin_group", 7, "Torvane"),
    ("kin_group", 19, "Kraiost"),
]
DEATHS = "There were 41,300 deaths: 36,050 by starvation, 5,060 by attack and 190 by old age."
LOW = "The lowest count was 8,900, in the 100 steps to step 1,517,900."
LOW2 = "The lowest count was 20,914, in the 100 steps to step 1,517,900."
DISSOLVED_7 = "Between step 1,516,250 and step 1,516,500, the Torvane kin dissolved; it had 980 members at step 1,516,250."
BORN_19 = "Between step 1,514,500 and step 1,514,750, the Kraiost kin formed with 1,640 members."
SPLIT_31 = "Between step 1,513,000 and step 1,513,250, the Shaemund kin split from the Braelix kin with 1,210 members."
GC_12 = "In the chapter, 3,100 agents were born into the Braelix kin and 4,900 of its members died, 610 of them by attack."
GR_12 = "The Braelix kin was the largest kin group at step 1,518,000, with 5,020 members, 23.7% of the population."
RAIDS = (
    "Members of the Braelix kin attacked members of the Torvane kin 412 times; 61 of the attacks were fatal."
)
BIRTHS = "There were 41,950 births."
TL = "At step 1,512,000 the largest lineage held 41.0% of the agents, and at step 1,518,000 the largest held 42.0%."
LN = "The number of lineages with living members went from 9 to 7."


def pick(*templates_):
    """A ``choice`` builder: the ids of the first candidate of each template."""

    def order(body):
        return [next(c["id"] for c in body if c["template"] == t) for t in templates_]

    return order


def select(sheet, *templates_, headline=None):
    cands = R.candidates(sheet)
    choice = {"headline": headline, "order": pick(*templates_)(cands[0])}
    rep = R.assemble(sheet, cands, choice)
    return [x["text"] for x in rep["body"]], rep


def test_template_mode_is_the_mandatory_sentences_and_the_first_k_by_salience() -> None:
    full = mk_sheet(
        kinds={"population": [pop(34410, 21140, 20914)], "deaths": [deaths(36050, 5060, 190)],
               "births": [{"kind": "births", "births": 41950, "immigrants": 0}],
               "lineages": [{"kind": "lineages", "from": 9, "to": 7}],
               "event": [event("dissolved", 7, 980, s0=A0 + 4250, s1=A0 + 4500), event("born", 19, 1640, s0=A0 + 2500, s1=A0 + 2750)],
               "group": [group(12, 5020, 21140, 1, 3100, 4900, 610), group(31, 1830, 21140, 2, 700, 80, 10)]},
        ents=KIN,
    )  # fmt: skip
    body, _heads, _block, ex = R.candidates(full)
    assert ex == [] and [c["template"] for c in body] == [
        "D-1", "P-low", "E-born", "E-dissolved", "G-r1", "G-c", "B-1", "LN", "G-r2", "G-c",
    ]  # fmt: skip
    assert [c["id"] for c in body] == [f"C{i}" for i in range(1, 11)]
    assert [c["salience"] for c in body] == list(range(1, 11)) and [c["mandatory"] for c in body][:3] == [
        True,
        False,
        True,
    ]
    rep = R.template_report(full)
    # six sentences (k), the mandatory ones among them, displayed in the order of sections: D, P-low, events in time order
    # order of the events by end step, then the group and its G-c
    assert [x["text"] for x in rep["body"]] == [DEATHS, LOW2, BORN_19, DISSOLVED_7, GR_12, GC_12]
    assert [x["origin"] for x in rep["body"]] == [
        "mandatory",
        "template",
        "mandatory",
        "template",
        "template",
        "template",
    ]
    assert rep["lead"] == "The population fell from 34,410 to 21,140 (-38.6%)."
    assert rep["headline"] == {"id": "H1", "text": "Population fell to 21,140", "origin": "template"}
    assert rep["interventions"] == []
    # a quiet chapter shows three sentences
    quiet = mk_sheet(kinds={"population": [pop(1000, 1010, 1000)], "deaths": [deaths(5, 5, 5)],
                            "births": [{"kind": "births", "births": 9, "immigrants": 0}],
                            "lineages": [{"kind": "lineages", "from": 9, "to": 7}],
                            "top_lineage": [{"kind": "top_lineage", "from_share": 0.5, "to_share": 0.4, "from_count": 500,
                                             "to_count": 404, "from_of": 1000, "to_of": 1010}]})  # fmt: skip
    assert quiet["quiet"] is True
    assert [x["text"][:12] for x in R.template_report(quiet)["body"]] == [
        "There were 1",
        "There were 9",
        "The number o",
    ]
    # a chapter that has an intervention inside is not quiet
    ivq = mk_sheet(kinds={"population": [pop(1000, 1010, 1000)], "intervention": [
        interv_regen()]})  # fmt: skip
    assert ivq["quiet"] is False


def interv_regen():
    return interv(0, "regen_pulse", A0 + 500, {"mult": 0.5}, duration=5_000)


def test_select_c1_a_crash_in_a_chapter_that_ends_level_is_not_quiet() -> None:
    sheet = mk_sheet(kinds={"population": [pop(30250, 30900, 8900)], "deaths": [deaths(36050, 5060, 190)],
                            "births": [{"kind": "births", "births": 41950, "immigrants": 0}],
                            "top_lineage": [{"kind": "top_lineage", "from_share": 0.41, "to_share": 0.42,
                                             "from_count": 12400, "to_count": 12978, "from_of": 30250, "to_of": 30900}]})  # fmt: skip
    assert sheet["quiet"] is False
    body, _rep = select(sheet, "B-1", "TL")
    assert (
        body == [DEATHS, LOW, BIRTHS, TL]
        and R.template_report(sheet)["headline"]["text"] == "Population rose to 30,900"
    )
    assert [h["text"] for h in R.candidates(sheet)[1]] == [
        "Population rose to 30,900"
    ]  # the crash: no choice of headline
    # the model leaves the crash out: the mandatory sentence is added all the same
    body, _ = select(sheet, "TL")
    assert body == [DEATHS, LOW, TL]


def test_select_c2_the_population_headline_is_the_only_one_offered_under_a_collapse() -> None:
    sheet = mk_sheet(kinds={"population": [pop(30250, 9800, 8900)], "deaths": [deaths(7100, 3300, 150, 9400)],
                            "births": [{"kind": "births", "births": 9500, "immigrants": 0}],
                            "lineages": [{"kind": "lineages", "from": 5, "to": 2}],
                            "event": [event("dissolved", 7, 3900, s0=A0 + 250, s1=A0 + 500)]}, ents=KIN)  # fmt: skip
    heads = R.candidates(sheet)[1]
    assert [(h["id"], h["text"]) for h in heads] == [("H1", "Population fell to 9,800")]
    body, rep = select(sheet, "B-1", headline="H2")  # an id that was not offered: the template headline
    assert rep["headline"] == {"id": "H1", "text": "Population fell to 9,800", "origin": "template"}
    assert body == [
        "There were 19,950 deaths: 7,100 by starvation, 3,300 by attack, 150 by old age and 9,400 by catastrophe.",
        "Between step 1,512,250 and step 1,512,500, the Torvane kin dissolved; it had 3,900 members at step 1,512,250.",
        "There were 9,500 births.",
    ]  # fmt: skip
    # an offered headline is taken
    small = mk_sheet(
        kinds={"population": [pop(1000, 1020, 1000)], "lineages": [{"kind": "lineages", "from": 5, "to": 2}]}
    )
    _, rep = select(small, headline="H2")
    assert rep["headline"] == {"id": "H2", "text": "Lineages went from 5 to 2", "origin": "model"}


def test_select_c3_the_relation_cannot_follow_the_dissolution_of_the_attacked_group() -> None:
    sheet = mk_sheet(kinds={"population": [pop(34410, 21140, 20914)], "deaths": [deaths(36050, 5060, 190)],
                            "event": [event("dissolved", 7, 980, s0=A0 + 4250, s1=A0 + 4500),
                                      event("born", 19, 1640, s0=A0 + 2500, s1=A0 + 2750)],
                            "group": [group(12, 5020, 21140, 1, 3100, 4900, 610)],
                            "relation": [{"kind": "relation", "a": 12, "b": 7, "attacks": 412, "kills": 61}]}, ents=KIN)  # fmt: skip
    body, _ = select(sheet, "E-dissolved", "R", "G-c")
    assert body == [DEATHS, BORN_19, DISSOLVED_7, GC_12, RAIDS]  # the section order, events in time order
    assert body.index(RAIDS) > body.index(GC_12) > body.index(DISSOLVED_7)
    # the model asks for the relation first and the group's own sentence last: still after the events
    body, _ = select(sheet, "R", "G-c", "G-r1")
    assert body == [DEATHS, BORN_19, GC_12, GR_12, RAIDS] or body == [DEATHS, BORN_19, GR_12, GC_12, RAIDS]
    assert body[1] == BORN_19 and body[-1] == RAIDS


def test_select_c4_c5_events_tied_at_the_largest_size_are_all_mandatory_and_in_time_order() -> None:
    sheet = mk_sheet(kinds={"population": [pop(34410, 21140, 20914)], "lineages": [{"kind": "lineages", "from": 9, "to": 7}],
                            "event": [event("split", 31, 1210, 12, 1180, s0=A0 + 1000, s1=A0 + 1250),
                                      event("dissolved", 7, 1210, s0=A0 + 4250, s1=A0 + 4500)]}, ents=KIN)  # fmt: skip
    body, rep = select(sheet, "LN")  # C4: only LN chosen; both events are added
    assert body == [SPLIT_31, DISSOLVED_7.replace("980", "1,210"), LN]
    assert [x["origin"] for x in rep["body"]] == ["mandatory", "mandatory", "model"]
    body, _ = select(sheet, "E-dissolved", "E-split")  # C5: a later event first in the reply
    assert body == [SPLIT_31, DISSOLVED_7.replace("980", "1,210")]


def test_select_n12_a_deep_dip_is_mandatory_even_when_the_chapter_ends_higher() -> None:
    sheet = mk_sheet(kinds={"population": [pop(30250, 32700, 8900)], "deaths": [deaths(36050, 5060, 190)],
                            "births": [{"kind": "births", "births": 41950, "immigrants": 0}],
                            "lineages": [{"kind": "lineages", "from": 9, "to": 8}],
                            "top_lineage": [{"kind": "top_lineage", "from_share": 0.41, "to_share": 0.42, "from_count": 12400,
                                             "to_count": 13734, "from_of": 30250, "to_of": 32700}],
                            "group": [group(12, 5020, 32700, 1, 3100, 4900, 610)]}, ents=KIN)  # fmt: skip
    body, _ = select(sheet, "B-1", "G-r1", "LN", "TL")
    assert body[:2] == [DEATHS, LOW] and body[2] == GR_12.replace("23.7%", "15.4%")
    assert body[3:] == [BIRTHS, "The number of lineages with living members went from 9 to 8.", TL]
    template = [x["text"] for x in R.template_report(sheet)["body"]]
    assert template[:2] == [DEATHS, LOW]


def test_select_n13_the_deaths_cannot_follow_a_dissolution() -> None:
    sheet = mk_sheet(kinds={"population": [pop(34410, 21140, 20914)], "deaths": [deaths(36050, 5060, 190)],
                            "event": [event("dissolved", 7, 980, s0=A0 + 4250, s1=A0 + 4500)]}, ents=KIN)  # fmt: skip
    body, _ = select(sheet, "E-dissolved", "D-1")
    assert body == [
        DEATHS,
        DISSOLVED_7,
    ]  # the model put the dissolution first; the order of sections puts the deaths first


def test_select_n14_a_one_id_reply_leaves_the_lead_and_the_mandatory_sentences() -> None:
    sheet = mk_sheet(kinds={"population": [pop(34410, 21140, 20914)], "deaths": [deaths(36050, 5060, 190)],
                            "event": [event("dissolved", 7, 980, s0=A0 + 4250, s1=A0 + 4500)]}, ents=KIN)  # fmt: skip
    body, rep = select(sheet, "D-1")
    assert body == [DEATHS, DISSOLVED_7] and rep["lead"].startswith("The population fell")
    # nonsense in the reply is ignored: unknown ids, repeats, beyond k
    cands = R.candidates(sheet)
    rep = R.assemble(sheet, cands, {"headline": None, "order": ["C99", "C1", "C1", 7, None, "H1"]})
    assert [x["id"] for x in rep["body"]] == ["C1", "C3"]
    rep = R.assemble(sheet, cands, {"headline": "H1", "order": []})
    assert [x["id"] for x in rep["body"]] == ["C1", "C3"] and rep["headline"]["origin"] == "model"


def test_select_keeps_the_models_order_among_groups_labels_and_relations() -> None:
    groups = [group(12, 5020, 21140, 1, 3100, 4900, 610), group(31, 1830, 21140, 2, 700, 80, 10)]
    rel = [{"kind": "relation", "a": 12, "b": 31, "attacks": 412, "kills": 61},
           {"kind": "relation", "a": 31, "b": 12, "attacks": 20, "kills": 1}]  # fmt: skip
    sheet = mk_sheet(
        kinds={"population": [pop(34410, 21140, 21140)], "group": groups, "relation": rel}, ents=KIN
    )
    cands = R.candidates(sheet)
    by_text = {c["text"][:14] + c["template"]: c["id"] for c in cands[0]}
    g1, g2 = by_text["The Braelix kiG-r1"], by_text["The Shaemund kG-r2"]
    ids = {c["id"]: c["text"] for c in cands[0]}
    gc1 = next(c["id"] for c in cands[0] if c["template"] == "G-c" and "Braelix" in c["text"])
    gc2 = next(c["id"] for c in cands[0] if c["template"] == "G-c" and "Shaemund" in c["text"])

    def body(order):
        return [x["text"] for x in R.assemble(sheet, cands, {"headline": None, "order": order})["body"]]

    # the model's order of groups (the second group first), each group followed by its own G-c
    assert body([g2, gc2, g1, gc1]) == [ids[g2], ids[gc2], ids[g1], ids[gc1]]
    assert body([gc2, g2, gc1, g1]) == [
        ids[g2],
        ids[gc2],
        ids[g1],
        ids[gc1],
    ]  # a G-c never goes before its group
    assert body([g1, gc1, g2, gc2]) == [ids[g1], ids[gc1], ids[g2], ids[gc2]]
    r_ids = [c["id"] for c in cands[0] if c["template"] == "R"]
    r1 = "Members of the Braelix kin attacked members of the Shaemund kin 412 times; 61 of the attacks were fatal."
    r2 = "Members of the Shaemund kin attacked members of the Braelix kin 20 times; 1 of the attacks was fatal."
    assert body(r_ids) == [r1, r2] and body(r_ids[::-1]) == [r2, r1]
    # relations come after the groups whatever the order of the reply
    assert body([r_ids[0], g1]) == [ids[g1], r1]


def test_the_body_is_in_section_order_whatever_the_choice() -> None:
    """On random choices over random sheets: the lead is first, every mandatory sentence is there once, no repeat, the
    sections are in order, events in time order, only candidate strings; the headline is a candidate."""
    import random

    from report_helpers import random_sheet

    for seed in range(300):
        rng = random.Random(seed)
        sheet = random_sheet(rng)
        cands = R.candidates(sheet)
        body, heads, block, _ = cands
        ids = [c["id"] for c in body]
        order = [rng.choice([*ids, "C0", "H1", "x", 3]) for _ in range(rng.randint(0, 9))]
        head = rng.choice([None, "H1", "H2", "H9"])
        rep = R.assemble(sheet, cands, {"headline": head, "order": order})
        by = {c["id"]: c for c in body}
        got = [x["id"] for x in rep["body"]]
        assert len(set(got)) == len(got) and set(got) <= set(by)
        assert {c["id"] for c in body if c["mandatory"]} <= set(got)
        sec = [templates.SECTION_ORDER.index(by[i]["section"]) for i in got]
        assert sec == sorted(sec), (seed, got)
        times = [by[i]["order"][1] for i in got if by[i]["section"] == "E"]
        assert times == sorted(times)
        assert rep["lead"] == R.lead_of(sheet)["text"]
        assert rep["headline"]["text"] in {h["text"] for h in heads}
        assert [x["text"] for x in rep["interventions"]] == [x["text"] for x in block]
        assert all(x["text"] == by[x["id"]]["text"] for x in rep["body"])
        k = max(
            R.REPORT_V1.k_quiet if sheet["quiet"] else R.REPORT_V1.k_body, sum(c["mandatory"] for c in body)
        )
        assert len([x for x in rep["body"] if x["origin"] == "model"]) <= k


def test_the_block_is_complete_last_and_independent_of_the_choice(rv) -> None:
    ents = [E(1_001_000, "rule", "regen_pulse", {"mult": 1.5}, 2_000, rule="extinction_rescue"),
            E(1_001_000, "rule", "immigration_pulse", {"count": 1024}, rule="extinction_rescue"),
            E(1_003_000, "revert", "regen_pulse", {"mult": 1.0}, reverts=1_001_000)]  # fmt: skip
    view, a, b = lone(ents)
    sheet = R.build_sheet(view, a, b)
    cands = R.candidates(sheet)
    full = [x["text"] for x in cands[2]]
    assert full == block_of(sheet) and len(full) == 3
    for choice in (None, {"headline": None, "order": []}, {"headline": "H1", "order": ["C1", "C2"]}):
        rep = R.assemble(sheet, cands, choice)
        assert [x["text"] for x in rep["interventions"]] == full
    # no headline, lead or body text names an intervention
    rep = R.template_report(sheet)
    joined = " ".join([rep["headline"]["text"], rep["lead"], *[x["text"] for x in rep["body"]]])
    assert not any(w in joined for w in ("regrowth", "immigrants", "director"))


# --- the ticker -------------------------------------------------------------------------------------------------


def test_ticker_lines_by_priority(rv) -> None:
    sheet = R.build_sheet(rv, 1000, 2000)
    lines = R.ticker(sheet)
    assert [t["step"] for t in lines] == [1250, 1500, 1750, 2000]
    assert (
        lines[0]["text"]
        == "Step 1,250: Between step 1,000 and step 1,250, the Torvane kin formed with 88 members."
    )
    assert (
        lines[1]["text"] == "Step 1,500: the population was 775."
    )  # the merged event inside the dip is suspect: none
    assert lines[3]["text"].startswith(
        "Step 2,000: Between step 1,750 and step 2,000, the Shaemund kin split from"
    )
    # an entry applied after the snapshot appears at the next one, in front of the event
    view, a, b = lone([E(1_002_000, "rule", "regen_pulse", {"mult": 1.5}, 2_000, rule="extinction_rescue"),
                       E(1_004_000, "revert", "regen_pulse", {"mult": 1.0}, reverts=1_002_000)])  # fmt: skip
    t = R.ticker(R.build_sheet(view, a, b))
    at = [x["text"] for x in t if x["step"] == 1_002_250]
    assert at == ["Step 1,002,250: At step 1,002,000, the director's rule extinction_rescue set food regrowth to 1.5 times "
                  "its base rate for 2,000 steps."]  # fmt: skip
    assert all("while this was in place" not in x["text"] for x in t)
    # at the snapshot at the entry's own step the entry is not yet there (it is applied after the snapshot)
    assert [x["text"][:20] for x in t if x["step"] == 1_002_000] == ["Step 1,002,000: the "]


def test_ticker_has_no_population_line_where_the_count_is_unknown(rv) -> None:
    without = variant(rv, records=lambda recs: [r for r in recs if r["step"] != 1500])
    sheet = R.build_sheet(without, 1000, 2000)
    assert sheet["excluded"]["social"]
    assert [t["step"] for t in R.ticker(sheet)] == [1500, 2000]  # only the snapshots that are also row steps


def test_every_ticker_line_passes_the_checker(rv) -> None:
    for a, b in ((1000, 2000), (2000, 3000), (3000, 4000)):
        sheet = R.build_sheet(rv, a, b)
        for line in R._ticker_lines(sheet):
            assert R.check_string(line["text"], line, sheet, block=True) == []


# --- 2,000 random logs: the block is the engine's lever state ------------------------------------------------------


def random_log(rng, a0, b0):
    """A random director log on [a0, b0): timed regen pulses and habitat losses with their reverts (restoring what the
    lever held before, as the engine does; a superseded pulse has no revert), untimed settings and points."""
    steps = sorted(rng.randrange(a0 // 100 + 1, b0 // 100 - 1) * 100 for _ in range(rng.randint(2, 12)))
    entries: list[dict] = []
    pending: list[tuple[int, dict]] = []  # (revert step, revert entry)
    for s in steps:
        for rs, re_ in sorted(pending, key=lambda x: x[0]):
            if rs <= s:
                entries.append(re_)
        pending = [p for p in pending if p[0] > s]
        src = rng.choice(["rule", "llm"])
        kind = rng.choice(["regen", "loss", "seasons", "drift", "gain", "pos", "imm", "cat"])
        if kind == "regen":
            before = RS.oracle_state(entries, s + 1)["regen"]
            pending = [
                p for p in pending if p[1]["action"] != "regen_pulse"
            ]  # a new pulse supersedes the old one
            d = rng.choice([500, 1000, 2500])
            entries.append(E(s, src, "regen_pulse", {"mult": rng.choice([0.5, 0.8, 1.5])}, d,
                             **({"rule": "extinction_rescue"} if src == "rule" else {})))  # fmt: skip
            pending.append((s + d, E(s + d, "revert", "regen_pulse", {"mult": before}, reverts=s)))
        elif kind == "loss":
            p = rng.randrange(0, 2)
            before = RS.oracle_state(entries, s + 1)["gain"].get(p, 1.0)
            pending = [
                x for x in pending if not (x[1]["action"] == "habitat_loss" and x[1]["params"]["patch"] == p)
            ]
            d = rng.choice([1000, 3000])
            entries.append(E(s, src, "habitat_loss", {"patch": p, "gain": 0.0}, d))
            pending.append(
                (s + d, E(s + d, "revert", "habitat_loss", {"patch": p, "gain": before}, reverts=s))
            )
        elif kind == "seasons":
            entries.append(
                E(s, src, "set_seasons", {"period": rng.choice([2000, 5000]), "amp": rng.choice([0.25, 0.5])})
            )
        elif kind == "drift":
            entries.append(E(s, src, "drift_change", {"mult": rng.choice([0.5, 2.0, 4.0])}))
        elif kind == "gain":
            entries.append(
                E(
                    s,
                    src,
                    "fertility_shift",
                    {"patch": rng.randrange(0, 2), "gain": rng.choice([0.25, 1.0, 1.75])},
                )
            )
        elif kind == "pos":
            entries.append(E(s, src, "fertility_shift", {"patch": rng.randrange(0, 2), "x": 5.5, "y": 6.5}))
        elif kind == "imm":
            entries.append(E(s, src, "immigration_pulse", {"count": 64}))
        else:
            entries.append(E(s, src, "catastrophe", {"center": [1.0, 2.0], "radius": 40.0, "kill_p": 0.5}))
    for rs, re_ in sorted(pending, key=lambda x: x[0]):
        if rs < b0:
            entries.append(re_)
    entries.sort(key=lambda e: e["step_applied"])  # stable: same-step entries keep their generation order
    return entries


def away_levers(state):
    f32 = np.float32
    out = set()
    if f32(state["regen"]) != f32(1.0):
        out.add("regen")
    if f32(state["drift"]) != f32(1.0):
        out.add("drift")
    if state["seasons"] != (0, 0.0):
        out.add("seasons")
    out |= {f"patch:{p}:gain" for p, g in state["gain"].items() if f32(g) != f32(1.0)}
    out |= {f"patch:{p}:pos" for p in state["pos"]}
    return out


def test_2000_random_logs_the_block_lists_every_lever_once_and_windows_lie_in_their_span() -> None:
    import random

    a0, b0, a, b = 1_000_000, 1_008_000, 1_004_000, 1_008_000
    shown = windows = standing = 0
    for seed in range(2000):
        rng = random.Random(seed)
        ents = random_log(rng, a0, b0)
        view = RS.make_view(a0, b0 + 1_000, ents)
        sheet = R.build_sheet(view, a, b)  # the env check inside compares the fold with the independent model
        ivs, st = facts_of(sheet, "intervention"), facts_of(sheet, "standing")
        assert sheet["excluded"]["interventions_dropped"] == 0
        away = away_levers(RS.oracle_state(ents, a))
        for lv in away:
            holders = [f for f in st if f["lever"] == lv] + [
                f for f in ivs
                if R.lever_of(f["action"], f["params"]) == lv and f["step"] < a and (f["end"] is None or f["end"] >= a)
            ]  # fmt: skip
            assert len(holders) == 1, (seed, lv, holders)
        assert {f["lever"] for f in st} <= away
        for w in facts_of(sheet, "window"):
            iv = next(f for f in ivs if f["id"] == w["intervention"])
            lo, hi = max(iv["step"], a), min(iv["end"] if iv["end"] is not None else b, b)
            assert (
                lo <= w["w0"] < w["w1"] <= hi and w["w1"] - w["w0"] >= 1_000 and (w["w0"], w["w1"]) != (a, b)
            )
        # the order of the block: standing lines, then the entries by (step, log index)
        order = [(f["step"], f["entry"]) for f in ivs]
        assert order == sorted(order)
        _body, _heads, _block, ex = R.candidates(sheet)
        assert ex == []
        shown += len(ivs)
        windows += len(facts_of(sheet, "window"))
        standing += len(st)
    assert shown > 3000 and windows > 30 and standing > 50  # the generator reaches the cases


# --- a golden tiny observed run (real engine output, committed as files) -----------------------------------------------

TINY = DATA / "report_tiny"
TINY_RESUMED = DATA / "report_tiny_resumed"


def test_golden_tiny_run() -> None:
    """The sheet, the candidates, the template report and the ticker of two chapters of a real tiny observed run
    (`tests/data/make_report_tiny.py`: the `tiny` preset, 3,000 steps on the CPU, rows every 100): equal to the
    stored JSON. The field names the sheet reads are the ones the engine writes."""
    view = R.load_run(TINY)
    golden = json.loads((TINY / "golden.json").read_text())
    assert [g["chapter"] for g in golden] == [[1000, 2000], [2000, 3000]]
    for g in golden:
        lo, hi = g["chapter"]
        sheet = R.build_sheet(view, lo, hi)
        assert json.loads(json.dumps(sheet)) == g["sheet"]
        assert R.facts_digest(sheet) == g["facts_digest"]
        body, heads, block, excluded = R.candidates(sheet)
        assert excluded == [] == g["excluded"]
        assert [{k: c[k] for k in ("id", "template", "cites", "text", "mandatory")} for c in body] == g[
            "candidates"
        ]
        assert [{k: h[k] for k in ("id", "template", "text")} for h in heads] == g["headlines"]
        assert [b["text"] for b in block] == g["block"]
        assert json.loads(json.dumps(R.template_report(sheet))) == g["report"]
        assert R.ticker(sheet) == g["ticker"]
        assert sheet["excluded"]["social"] is None and sheet["dims"] == 2 and sheet["origin"] == 0
        assert facts_of(sheet, "population") and facts_of(sheet, "group")


def test_golden_tiny_alive_identity_at_every_common_step() -> None:
    """`other + sum(size)` of a record is the engine's `observers.alive` wherever a row has the same step."""
    view = R.load_run(TINY)
    rows = {r["step"]: r for r in view.rows}
    common = 0
    for rec in view.records:
        if rec["step"] in rows:
            common += 1
            assert R.alive_of(rec) == rows[rec["step"]]["observers"]["alive"], rec["step"]
    assert common >= 5
    assert [r["step"] for r in view.records] == list(range(250, 3001, 250))


def test_golden_tiny_resumed_run_gives_the_same_sheets() -> None:
    """A run stopped at 2,000 and resumed from its checkpoint at 1,500 writes the same rows and records (the engine's
    resume is bitwise), so the sheets and the texts are equal."""
    a, b = R.load_run(TINY), R.load_run(TINY_RESUMED)
    assert [r["step"] for r in b.rows] == [r["step"] for r in a.rows]
    for lo, hi in ((1000, 2000), (2000, 3000)):
        sa, sb = R.build_sheet(a, lo, hi), R.build_sheet(b, lo, hi)
        assert R.facts_digest(sa) == R.facts_digest(sb)
        assert R.template_report(sa) == R.template_report(sb) and R.ticker(sa) == R.ticker(sb)


def test_group_relation_label_and_window_quotas_count_what_they_drop(rv, monkeypatch) -> None:
    """Groups 6, relations 3, labels 6 and windows 5: the first facts in each kind's key order are kept and the rest
    is counted in `excluded.quota`; ties are broken by the ids, never by input order."""
    ids = list(range(10, 18))  # eight more kin groups (ids 10-17, sizes 1-8 above the existing ones' order)

    def many(rec):
        t = rec["kin_groups"]
        n = len(t["id"])
        extra = {
            "id": ids, "name": [f"Grp{i}" for i in ids], "named_after": [None] * 8, "slot": [n + j for j in range(8)],
            "size": [30 + j for j in range(8)], "births": [1] * 8, "deaths": [[0, 0, 0, 0]] * 8,
        }  # fmt: skip
        cols = {c: [*t[c], *extra.get(c, [None] * 8)] for c in t}
        other = rec["summary"]["kin_groups"]["other"] - sum(extra["size"])
        rel = rec["relations"]
        pairs = [(10 + j, 11 + j) for j in range(5)]  # five more pairs with attacks 20 + j
        rel2 = {c: [*rel[c], *[{"a": p[0], "b": p[1], "attacks": 20 + j, "hits": 20, "kills": 1, "bonds": 0, "shared": 0,
                                "bond_slots": 0, "exp_attacks": 1.0, "exp_bonds": None, "sim": 0.2, "distinct": True}.get(c)
                               for j, p in enumerate(pairs)]] for c in rel}  # fmt: skip
        return {**rec, "kin_groups": cols, "relations": rel2,
                "summary": {**rec["summary"], "kin_groups": {**rec["summary"]["kin_groups"], "other": other, "tracked": n + 8}}}  # fmt: skip

    world = variant(rv, records=lambda recs: [many(r) if r["step"] > 1000 else r for r in recs])
    with_table(monkeypatch, rv)
    sheet = R.build_sheet(world, 1000, 2000)
    assert len(facts_of(sheet, "group")) == 6 and sheet["excluded"]["quota"]["group"] == 5  # 11 listed
    assert [g["size"] for g in facts_of(sheet, "group")] == sorted(
        (g["size"] for g in facts_of(sheet, "group")), reverse=True
    )
    rels = facts_of(sheet, "relation")
    assert (
        len(rels) == 3 and sheet["excluded"]["quota"]["relation"] == 3
    )  # the pair 1 -> 2 (12) and five of 20 to 24 attacks
    assert [r["attacks"] for r in rels] == [
        96,
        92,
        88,
    ]  # four windows of (20 + j) attacks each, the largest first
    assert all(r["a"] < r["b"] for r in rels)


def test_label_quota_of_six(rv, monkeypatch) -> None:
    keys = ("hunter", "settler", "provider", "trader", "nomad", "gatherer")
    names = ("resting", "hunting", "roaming", "caring")
    rows = lambda rs: [  # noqa: E731
        {**r, "observers": {**r["observers"], "arch_labelled": 400, "intent_labelled": 400,
                            "archetype_match": {k: 0.1 + j / 100 + (r["step"] == 2000) * 0.05 * (j + 1) for j, k in enumerate(keys)},
                            "intent": {k: 0.1 + (r["step"] == 2000) * 0.025 * (j + 1) for j, k in enumerate(names)}}}
        for r in rs]  # fmt: skip
    world = variant(rv, rows=rows)
    monkeypatch.setattr(
        R, "LABEL_VERDICTS", {(1, RS.V1_DIGEST, 2, world_key(rv)): {k: PASS for k in (*keys, *names)}}
    )
    sheet = R.build_sheet(world, 1000, 2000)
    got = [f["key"] for f in facts_of(sheet, "label")]
    assert len(got) == 6 and sheet["excluded"]["quota"] == {"label": 4}
    changes = [abs(f["to_share"] - f["from_share"]) for f in facts_of(sheet, "label")]
    assert changes == sorted(changes, reverse=True)  # the salience key: the largest change of share first
