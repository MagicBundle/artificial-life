"""The world report, version 2 (docs/DESIGN.md §34.4; G3b spec §5): the three chapters of the Mac-day play of planet
``p0`` with their texts pinned, the kin threads and the storms in a sheet, the origin row of the first chapter, the
player's entries (rule P), a chapter read from the end of the logs, the quiet form and the version digests. CPU
only, no socket. The live job is in ``test_report_job.py``."""

# ruff: noqa: E501  (expected report texts are long lines)

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

import report_synth as RS
from alife.director import actions
from alife.game import kinthreads
from alife.game import report as R
from alife.game import reportjob as RJ
from report_synth import entry, variant

DATA = Path(__file__).parent / "data"
P0 = DATA / "report_p0"


@pytest.fixture(scope="module")
def p0() -> R.RunView:
    return R.load_run(P0)


def facts_of(sheet: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    return [f for f in sheet["facts"] if f["kind"] == kind]


# --- the three chapters of p0 (spec §5.2a: the v1 text is N1's, the v2 text is what this code writes) ----------

P0_CHAPTERS = {
    (450250, 456250): {
        "headline": "3 storms reached named strength",
        "lead": "The population fell from 44,860 to 44,799 (-0.1%).",
        "body": [
            "269,868 organisms were born and 269,860 died: a turnover of 6.0 times the population.",
            "6 storms were tracked at the end of the chapter, 4 of them named.",
            "3 storms formed and 0 dissolved during the chapter.",
            "Storm Phaitraern reached named strength at step 451,600.",
            "Pruashol merged into Teshash at step 454,000; it had up to 4,659 members.",
            "45 further kin group changes did not last long enough to be listed.",
        ],
        "interventions": [],
        "ticker": [
            (
                451250,
                "Step 451,250: Flauzaest has held together since step 450,750 after splitting from Leifloan: 332 members.",
            ),
            (451600, "Step 451,600: Storm Phaitraern reached named strength."),
            (451850, "Step 451,850: Storm Fuapriond reached named strength."),
            (
                453750,
                "Step 453,750: Guadouck merged into Leifloan at step 453,000; it had up to 537 members.",
            ),
            (
                454000,
                "Step 454,000: Daudrall IV has held together since step 453,500 after splitting from Leifloan: 423 members.",
            ),
            (
                454750,
                "Step 454,750: Pruashol merged into Teshash at step 454,000; it had up to 4,659 members.",
            ),
            (
                455000,
                "Step 455,000: Daudrall IV merged into Leifloan at step 454,250; it lasted 750 steps and had up to 477 members.",
            ),
            (455025, "Step 455,025: Storm Bruakaum reached named strength."),
            (
                455750,
                "Step 455,750: Stauthaend has held together since step 455,250 after splitting from Teshash: 4,462 members.",
            ),
        ],
        "untold": 45,
    },
    (456000, 462000): {
        "headline": "2 storms reached named strength",
        "lead": "The population fell from 44,910 to 44,002 (-2.0%).",
        "body": [
            "269,929 organisms were born and 270,753 died: a turnover of 6.1 times the population.",
            "6 storms were tracked at the end of the chapter, 5 of them named.",
            "3 storms formed and 1 dissolved during the chapter.",
            "Storm Phaitraern dissolved at step 458,825 after 8,725 steps.",
            "Prymar dissolved, up to 247 members.",
            "10 further kin group changes did not last long enough to be listed.",
        ],
        "interventions": [],
        "ticker": [
            (
                456250,
                "Step 456,250: Roabraeth merged into Leifloan at step 455,500; it lasted 1,000 steps and had up to 127 members.",
            ),
            (
                457250,
                "Step 457,250: Daudrall II has held together since step 456,750 after splitting from Leifloan: 163 members.",
            ),
            (
                457750,
                "Step 457,750: Doaglaill merged into Leifloan at step 457,000; it lasted 2,500 steps and had up to 236 members.",
            ),
            (458000, "Step 458,000: Prymar dissolved, up to 247 members."),
            (458225, "Step 458,225: Storm Gufleask reached named strength."),
            (458825, "Step 458,825: Storm Phaitraern dissolved after 8,725 steps."),
            (459750, "Step 459,750: Niodrond dissolved after 3,000 steps, up to 131 members."),
            (460500, "Step 460,500: Storm Raigeal reached named strength."),
            (462000, "Step 462,000: Rofoand dissolved after 7,750 steps, up to 170 members."),
        ],
        "untold": 10,
    },
    (462000, 468000): {
        "headline": "Population fell to 35,934",
        "lead": "The population fell from 44,002 to 35,934 (-18.3%).",
        "body": [
            "242,635 organisms were born and 250,656 died: a turnover of 6.3 times the population.",
            "8,249 starved in the 250 steps to step 466,250, against a median of 5,451.",
            "6 storms were tracked at the end of the chapter, 5 of them named.",
            "1 storm formed and 1 dissolved during the chapter.",
            "Stauthaend merged into Teshash at step 464,750; it lasted 9,500 steps and had up to 4,539 members.",
            "28 further kin group changes did not last long enough to be listed.",
        ],
        "interventions": [
            "At step 466,000, the director's rule runaway_damper set food regrowth to 0.8 times its base rate for 10,000 steps.",
            "It was still in place at step 468,000.",
            "From 42,873 at step 466,000 to 35,934 at step 468,000, with the change applied at step 466,000.",
        ],
        "ticker": [
            (464250, "Step 464,250: Flauzaest dissolved after 12,750 steps, up to 370 members."),
            (
                465500,
                "Step 465,500: Stauthaend merged into Teshash at step 464,750; it lasted 9,500 steps and had up to 4,539 members.",
            ),
            (465750, "Step 465,750: Baepuav dissolved after 4,250 steps, up to 155 members."),
            (
                466250,
                "Step 466,250: At step 466,000, the director's rule runaway_damper set food regrowth to 0.8 times its base rate for 10,000 steps.",
            ),
            (
                466750,
                "Step 466,750: Vaucliom has held together since step 466,250 after splitting from Floahand: 525 members.",
            ),
            (
                467250,
                "Step 467,250: Pruashol has held together since step 466,750 after splitting from Teshash: 3,243 members.",
            ),
            (
                468000,
                "Step 468,000: Lobrirn has held together since step 467,500 after splitting from Vaucliom: 223 members.",
            ),
        ],
        "untold": 28,
    },
}


@pytest.mark.parametrize("chapter", sorted(P0_CHAPTERS))
def test_the_p0_chapters_read_as_pinned(p0: R.RunView, chapter: tuple[int, int]) -> None:
    """The v2 report of the three chapters of the Mac-day play (spec §5.2a). The v1 text had three kin sentences
    about groups that split and merged inside 750 steps, "101 houses died out" and a deaths list; this has the
    turnover, the storms, the lineages that lasted, and says how many changes did not."""
    a, b = chapter
    want = P0_CHAPTERS[chapter]
    sheet = R.build_sheet(p0, a, b)
    rec = R.narrate(sheet, None)
    assert rec["excluded_candidates"] == [] and sheet["quiet"] is False
    rep = rec["report"]
    assert rep["headline"]["text"] == want["headline"] and rep["lead"] == want["lead"]
    assert [x["text"] for x in rep["body"]] == want["body"]
    assert [x["text"] for x in rep["interventions"]] == want["interventions"]
    assert [(t["step"], t["text"]) for t in rec["ticker"]] == want["ticker"]
    assert sum(f["n"] for f in facts_of(sheet, "untold")) == want["untold"]
    assert rep["body"][0]["origin"] == "mandatory" and rep["body"][-1]["origin"] == "mandatory"
    assert len(rep["body"]) == 6
    # no house headline: the 101 "died out" of version 1 were houses outside the 16 largest
    assert all("died out" not in t for t in [rep["headline"]["text"], *[x["text"] for x in rep["body"]]])
    # the replay of the stored record from its own facts gives the same report
    assert R.verify_record(json.loads(json.dumps(rec)), sheet)["status"] == "ok"
    # every ticker line and every sentence passes the checker
    for line in R._ticker_lines(sheet):
        assert R.check_string(line["text"], line, sheet, block=True) == []


def test_the_p0_storms_and_the_untold_counts_of_the_spec(p0: R.RunView) -> None:
    s1 = R.build_sheet(p0, 450250, 456250)
    (st,) = facts_of(s1, "storms")
    # 6 tracked and 4 named at the end; 3 formed (told as born or named) in the chapter, 3 named
    assert (st["tracked"], st["named"], st["formed"], st["dissolved"], st["named_new"]) == (6, 4, 3, 0, 3)
    named = [(f["name"], f["named_step"], f["hemisphere"]) for f in facts_of(s1, "storm")]
    assert ("Phaitraern", 451600, "southern") in named and len(named) == 3
    # the strongest first; a ratio is the peak vorticity in naming-threshold units
    ratios = [f["ratio"] for f in facts_of(s1, "storm")]
    assert ratios == sorted(ratios, reverse=True) and all(r >= 1.0 for r in ratios)
    s2 = R.build_sheet(p0, 456000, 462000)
    (st2,) = facts_of(s2, "storms")
    assert (st2["tracked"], st2["named"], st2["formed"], st2["dissolved"], st2["named_new"]) == (
        6,
        5,
        3,
        1,
        2,
    )
    end = [f for f in facts_of(s2, "storm") if f["end"] is not None]
    assert [(f["name"], f["end"], f["age"]) for f in end] == [("Phaitraern", 458825, 8725)]
    # the 2 storm tracks of the chapter 3 are 1 formed and 1 dissolved
    s3 = R.build_sheet(p0, 462000, 468000)
    (st3,) = facts_of(s3, "storms")
    assert (st3["formed"], st3["dissolved"], st3["named_new"]) == (1, 1, 0)
    assert [f["n"] for f in facts_of(s3, "starve")] == [8_249]
    assert [f["n"] for f in facts_of(s3, "stormdeaths")] == [12]


def test_the_p0_window_of_the_director_move_has_its_rows(p0: R.RunView) -> None:
    sheet = R.build_sheet(p0, 462000, 468000)
    (w,) = facts_of(sheet, "window")
    assert (w["w0"], w["w1"], w["r0"], w["r1"], w["from"], w["to"]) == (
        466000,
        468000,
        466000,
        468000,
        42_873,
        35_934,
    )
    (iv,) = facts_of(sheet, "intervention")
    assert (iv["source"], iv["rule"], iv["step"], iv["duration"]) == (
        "rule",
        "runaway_damper",
        466000,
        10_000,
    )


def test_the_chronicles_events_and_the_reports_are_the_same_events(p0: R.RunView) -> None:
    """The report tells the kin threads the chronicle tells (one fold): the told events of a chapter are the
    ``kin_formed`` and ``kin_ended`` events of the chronicle in the same steps, with the same names."""
    from alife.game.chronicle import Chronicler

    chron = Chronicler(0)
    told: list[dict[str, Any]] = []
    for rec in p0.records:
        evs = chron.from_record(rec)
        told += [e for e in evs if e["fam"] == "kin" and 456000 < e["step"] <= 462000]
    sheet = R.build_sheet(p0, 456000, 462000)
    # the sheet keeps the largest six; the chronicle's events of the chapter are all of them
    names = sorted(f["name"] for f in facts_of(sheet, "thread"))
    assert all(any(f["name"] in e["text"] for e in told) for f in facts_of(sheet, "thread"))
    assert names and len(told) >= len(names)


# --- the versions ------------------------------------------------------------------------------------------------


def test_the_versions_and_a_version_1_record(p0: R.RunView) -> None:
    sheet = R.build_sheet(p0, 456000, 462000)
    rec = R.narrate(sheet, None)
    v = R.record_versions()
    assert (R.REPORT_VERSION, R.CHECKER_VERSION) == (2, 1)
    from alife.game import lexicon, templates

    assert (templates.TEMPLATE_VERSION, lexicon.LEXICON_VERSION) == (2, 3)
    assert {k: rec[k] for k in v} == v
    assert v["kinthreads_digest"] == kinthreads.constants_digest()
    # a record of N1's first version is read as a record of other versions, never compared field by field
    old = {**rec, "report_version": 1, "template_version": 1, "lexicon_digest": "0" * 64}
    got = R.verify_record(old, sheet)
    assert got["status"] == "versions_differ"
    assert got["versions"]["report_version"] == [1, 2] and got["versions"]["template_version"] == [1, 2]
    assert got["versions"]["lexicon_digest"][0] == "0" * 64
    # the storm persistence rule and the thread rule are pinned in the record
    assert R.verify_record({**rec, "storm_rules_digest": "0" * 64}, sheet)["status"] == "versions_differ"
    assert R.verify_record({**rec, "kinthreads_digest": "0" * 64}, sheet)["status"] == "versions_differ"


def test_the_storm_rules_digest_follows_the_chronicles_constants(monkeypatch: pytest.MonkeyPatch) -> None:
    before = R.storm_rules_digest()
    import alife.storms as S

    monkeypatch.setattr(S, "STORM_MIN_DETECTIONS", S.STORM_MIN_DETECTIONS + 1)
    assert R.storm_rules_digest() != before


# --- the tail read: a chapter from the end of the logs -----------------------------------------------------------


def saved_run(tmp_path: Path, a: int) -> Path:
    """A copy of p0 with the kin thread table of the checkpoint at ``a`` saved, as a played world saves it."""
    run = tmp_path / "p0"
    shutil.copytree(P0, run)
    kt = kinthreads.KinThreads()
    full = R.load_run(P0)
    for rec in full.records:
        if rec["step"] <= a:
            kt.fold(rec)
    obs = run / "observers"
    (obs / f"kinthreads_{a:010d}.json").write_text(json.dumps(kt.state()) + "\n")
    return run


@pytest.mark.parametrize(("a", "b"), [(456000, 462000), (462000, 468000)])
def test_the_tail_read_sheet_equals_the_full_read_sheet(tmp_path: Path, a: int, b: int) -> None:
    run = saved_run(tmp_path, a)
    full = R.build_sheet(R.load_run(run), a, b)
    tail_view = R.load_run(run, tail_from=a)
    assert len(tail_view.rows) < len(R.load_run(run).rows) and len(tail_view.records) < len(
        R.load_run(run).records
    )
    assert min(r["step"] for r in tail_view.rows) >= a and min(r["step"] for r in tail_view.records) >= a
    tail = R.build_sheet(tail_view, a, b)
    assert R.facts_digest(tail) == R.facts_digest(full) and tail == full
    assert R.narrate(tail, None) == R.narrate(full, None)  # the record, byte for byte


def test_a_tail_read_without_the_saved_table_is_refused_and_the_full_read_does_not_need_it(
    tmp_path: Path,
) -> None:
    run = tmp_path / "p0"
    shutil.copytree(P0, run)
    with pytest.raises(R.TailError):
        R.build_sheet(R.load_run(run, tail_from=456000), 456000, 462000)
    assert R.TailError("x").code == 2 and issubclass(R.TailError, R.ReportError)
    # the whole log refolds to the table the checkpoint would have saved
    saved = saved_run(tmp_path / "saved", 456000)
    assert R.facts_digest(R.build_sheet(R.load_run(run), 456000, 462000)) == R.facts_digest(
        R.build_sheet(R.load_run(saved), 456000, 462000)
    )


def test_a_saved_table_of_another_epoch_or_step_is_ignored_and_the_log_refolded(tmp_path: Path) -> None:
    run = saved_run(tmp_path, 456000)
    want = R.facts_digest(R.build_sheet(R.load_run(P0), 456000, 462000))
    path = run / "observers" / "kinthreads_0000456000.json"
    state = json.loads(path.read_text())
    for bad in (
        {**state, "origin": 99},
        {**state, "step": 450250},
        {**state, "constants": "0" * 64},
        {"format": "x"},
    ):
        path.write_text(json.dumps(bad) + "\n")
        assert R.facts_digest(R.build_sheet(R.load_run(run), 456000, 462000)) == want
    path.write_text("not json")
    assert R.facts_digest(R.build_sheet(R.load_run(run), 456000, 462000)) == want  # unreadable: refolded


def test_a_saved_table_of_an_earlier_step_is_not_the_table_at_the_start_of_the_chapter(
    tmp_path: Path,
) -> None:
    run = saved_run(tmp_path, 456000)
    want = R.facts_digest(R.build_sheet(R.load_run(P0), 456000, 462000))
    early = kinthreads.KinThreads()
    for rec in R.load_run(P0).records:
        if rec["step"] <= 450250:
            early.fold(rec)
    path = run / "observers" / "kinthreads_0000456000.json"
    assert early.state()["step"] == 450250 and early.state() != json.loads(path.read_text())
    path.write_text(json.dumps(early.state()) + "\n")
    assert R.facts_digest(R.build_sheet(R.load_run(run), 456000, 462000)) == want


def test_the_records_of_an_earlier_observer_epoch_are_not_folded(p0: R.RunView) -> None:
    a, b = 456000, 462000
    old = tuple({**r, "origin": 7} if r["step"] <= 455000 else r for r in p0.records)
    only = tuple(r for r in p0.records if r["step"] > 455000)
    epoch = p0.records[-1]["origin"]
    got, want = (
        R.thread_fold(p0._replace(records=old), a, b, epoch),
        R.thread_fold(p0._replace(records=only), a, b, epoch),
    )
    assert got.told == want.told and got.untold == want.untold and got.largest_a == want.largest_a


def test_the_thread_fold_is_a_function_of_the_records_up_to_the_end_of_the_chapter(p0: R.RunView) -> None:
    """A world that played on gives the report the world had when the chapter ended: records after ``b`` are
    never read (live equals verify)."""
    a, b = 456000, 462000
    cut = R.RunView(
        p0.cfg, p0.meta, tuple(r for r in p0.rows if r["step"] <= b + 250),
        tuple(r for r in p0.records if r["step"] <= b), p0.entries, p0.sidecars, tuple(s for s in p0.storms if s["step"] <= b),
    )  # fmt: skip
    assert R.facts_digest(R.build_sheet(cut, a, b)) == R.facts_digest(R.build_sheet(p0, a, b))


# --- the origin row: the first chapter of a world ----------------------------------------------------------------


def origin_view(start: int = 1000, *, with_row: bool = True) -> R.RunView:
    """An in-memory world that starts at ``start`` with no row at it (a fresh world's first row is one step in)."""
    view = RS.make_view(start, start + 1_000, start=start)
    rows = tuple(r for r in view.rows if r["step"] > start)
    n = RS.alive(start)
    row = {"step": start, "steps": 0, "origin": True, "observers": {"alive": n}, "clades_alive": 5,
           "top_clade_share": (n // 4) / n}  # fmt: skip
    meta = {**view.meta, **({"origin_row": row} if with_row else {})}
    return R.RunView(view.cfg, meta, rows, view.records, view.entries, view.sidecars)


def test_the_first_chapter_of_a_world_is_cut_from_its_origin_row() -> None:
    view = origin_view()
    with pytest.raises(R.ReportError, match="not the step of a metrics row"):
        R.build_sheet(origin_view(with_row=False), 1000, 2000)  # N1 version 1: no report of the first chapter
    sheet = R.build_sheet(view, 1000, 2000)
    pop = facts_of(sheet, "population")[0]
    assert (pop["from"], pop["to"]) == (RS.alive(1000), RS.alive(2000))
    assert sheet["excluded"]["facts_dropped"] == ["no social baseline at the origin"]
    assert not facts_of(sheet, "houses") and not facts_of(
        sheet, "label"
    )  # they need the social head of the row at a
    assert facts_of(sheet, "lineages") and facts_of(sheet, "top_lineage")  # the origin row has these
    assert facts_of(sheet, "thread") or facts_of(sheet, "untold") or facts_of(sheet, "group")
    body, heads, _block, ex = R.candidates(sheet)
    assert ex == [] and heads and body
    # the origin row is the row at the cut only for the chapter that starts at the start step
    with pytest.raises(R.ReportError, match="not the step of a metrics row"):
        R.build_sheet(view, 1250, 2000)
    # read from the end of the logs the first chapter needs no saved table: nothing came before it
    tail = view._replace(tail_from=1000)
    assert R.facts_digest(R.build_sheet(tail, 1000, 2000)) == R.facts_digest(sheet)
    # a start off the 250 grid has no origin cut
    off = origin_view(start=1050)
    with pytest.raises(R.ReportError, match="multiple of the snapshot cadence"):
        R.build_sheet(off, 1050, 2000)


def test_the_origin_row_has_no_labels_even_for_a_world_whose_rules_are_known(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sheet0 = R.build_sheet(origin_view(), 1000, 2000)
    table = next(t for (v, _d, dims, _w), t in R.LABEL_VERDICTS.items() if v == 1 and dims == sheet0["dims"])
    monkeypatch.setitem(R.LABEL_VERDICTS, (1, RS.V1_DIGEST, sheet0["dims"], sheet0["world_key"]), table)
    sheet = R.build_sheet(origin_view(), 1000, 2000)
    assert sheet["labels"]["key_found"] is True and not facts_of(sheet, "label")
    assert sheet["excluded"]["facts_dropped"] == ["no social baseline at the origin"]


def test_the_origin_row_needs_a_count_of_the_living() -> None:
    view = origin_view()
    broken = R.RunView(view.cfg, {**view.meta, "origin_row": {"step": 1000, "origin": True, "observers": {}}},
                       view.rows, view.records, view.entries, view.sidecars)  # fmt: skip
    with pytest.raises(R.ReportError, match="no count of the living"):
        R.build_sheet(broken, 1000, 2000)


def test_the_thread_fold_of_a_fresh_world_starts_from_nothing() -> None:
    """The initial cast of the first record is silent whatever the world: the first chapter of a fresh world tells
    only what happens after its first snapshot."""
    sheet = R.build_sheet(origin_view(), 1000, 2000)
    assert {t["name"] for t in facts_of(sheet, "thread")} <= {
        "Torvane"
    }  # Braelix and Shaemund were there from the start


# --- the player's entries (rule P) -------------------------------------------------------------------------------


def power(
    step: int,
    action: str = "regen_pulse",
    params: dict | None = None,
    duration: int | None = 3_000,
    name: str = "food",
):
    return entry(
        step,
        "player",
        action,
        params or {"mult": 0.5},
        duration,
        power=name,
        specs="game-v0",
        cost=1,
        scope="player",
    )


def revert(step: int, reverts: int, params: dict | None = None):
    return entry(
        step, "revert", "regen_pulse", params or {"mult": 1.0}, None, reverts=reverts, scope="player"
    )


def test_is_player_agrees_with_the_directors_predicate() -> None:
    samples = [
        {"source": "player"},
        {"source": "revert", "scope": "player"},
        {"source": "revert", "scope": "rule"},
        {"source": "revert"},
        {"source": "rule"},
        {"source": "llm", "scope": "player"},
    ]
    for rec in samples:
        assert R.is_player(rec) == actions.is_player(rec), rec
    assert R.after_start({"source": "player", "step_applied": 100}, 100)
    assert not R.after_start({"source": "rule", "step_applied": 100}, 100)
    assert R.after_start({"source": "rule", "step_applied": 101}, 100)
    assert not R.after_start({"source": "player", "step_applied": 99}, 100)
    assert R.after_start({"source": "revert", "scope": "player", "step_applied": 100}, 100)
    assert not R.after_start({"source": "revert", "step_applied": 100}, 100)


def test_a_power_at_the_origin_step_is_in_force_for_the_whole_first_chapter() -> None:
    """A player's entry at ``s`` is applied after the checkpoint at ``s`` (rule P): the rows from the next one
    carry it, the check of the env block passes and the sheet lists it. Version 1 read it as before the start."""
    ents = [power(1000), revert(4000, 1000)]
    view = RS.make_view(1000, 3000, ents, start=1000)
    sheet = R.build_sheet(view, 1000, 2000)  # the checker's fold applies the entry at `start_step`
    (iv,) = facts_of(sheet, "intervention")
    assert (iv["source"], iv["src"], iv["step"], iv["duration"]) == ("player", "the player", 1000, 3_000)
    assert iv["end"] is None and iv["open_to"] == 2000
    assert sheet["excluded"]["entries_before_start"] == 0
    (line,) = [b for b in R.candidates(sheet)[2] if b["template"] == "I-regen"]
    assert (
        line["text"]
        == "At step 1,000, the player set food regrowth to 0.5 times its base rate for 3,000 steps."
    )
    # the same entry of a boundary source at the start step is the checkpoint's: ignored
    rule = entry(1000, "rule", "regen_pulse", {"mult": 0.5}, 3_000, rule="boredom")
    view2 = RS.make_view(1000, 3000, [rule], start=1000)
    assert facts_of(R.build_sheet(view2, 1000, 2000), "intervention") == []
    assert R.build_sheet(view2, 1000, 2000)["excluded"]["entries_before_start"] == 1


def test_the_env_check_sees_a_power_at_the_start_step_and_refuses_a_log_that_ignores_it() -> None:
    ents = [power(1000)]
    view = RS.make_view(1000, 2000, ents, start=1000)
    R.check_env(
        view, 1000, 2000
    )  # the rows were computed by the independent model: the entry applies after `start`
    bad = variant(view, rows=lambda rows: [{**r, "env": {**r["env"], "regen_mult": 1.0}} for r in rows])
    with pytest.raises(R.ReportError, match="regen_mult"):
        R.check_env(bad, 1000, 2000)
    # the entry's own step has not seen it: the row at 1,000 (if there were one) is before it
    assert R.fold_levers(ents, 1000, 1000) == {}
    assert R.fold_levers(ents, 1001, 1000)["regen"].value == {"mult": 0.5}


def test_a_timed_bloom_has_a_span_and_a_window() -> None:
    """``TIMED`` is "the entry has a duration" (a Bloom is a fertility gain with one), not a list of two names."""
    ents = [power(1_001_013, "fertility_shift", {"patch": 3, "gain": 1.8}, 3_000, name="bloom"),
            entry(1_004_013, "revert", "fertility_shift", {"patch": 3, "gain": 1.0}, None, reverts=1_001_013, scope="player")]  # fmt: skip
    view = RS.make_view(1_000_000, 1_009_000, ents)
    sheet = R.build_sheet(view, 1_000_000, 1_006_000)
    (iv,) = facts_of(sheet, "intervention")
    assert (iv["end"], iv["end_by"], iv["restored"]) == (1_004_013, "elapsed", {"patch": 3, "gain": 1.0})
    (w,) = facts_of(sheet, "window")
    assert (w["r0"], w["r1"]) == (1_001_000, 1_004_000)
    _, _, block, ex = R.candidates(sheet)
    assert ex == [] and [x["template"] for x in block] == ["I-fert-d", "X-gain-1", "W"]
    assert (
        block[0]["text"]
        == "At step 1,001,013, the player set the fertility of patch 3 to 1.8 times its base value for 3,000 steps."
    )
    # an untimed gain has no window
    untimed = [entry(1_001_013, "llm", "fertility_shift", {"patch": 3, "gain": 1.8})]
    s2 = R.build_sheet(RS.make_view(1_000_000, 1_009_000, untimed), 1_000_000, 1_006_000)
    assert not facts_of(s2, "window") and [x["template"] for x in R.candidates(s2)[2]] == ["I-fert", "X-open"]


def test_an_effect_that_spans_the_whole_chapter_has_no_window() -> None:
    """A window from the first row to the last row of the chapter is the chapter's own change, not a consequence."""
    ents = [power(1_000_500, duration=20_000)]
    view = RS.make_view(1_000_000, 1_009_000, ents, start=900_000)
    spans = R.build_sheet(view, 1_000_000, 1_006_000)
    assert [f["w0"] for f in facts_of(spans, "window")] == [1_000_500]  # starts 500 steps in: a window
    long_ago = [power(950_000, duration=200_000)]
    sheet = R.build_sheet(RS.make_view(1_000_000, 1_009_000, long_ago, start=900_000), 1_000_000, 1_006_000)
    (iv,) = facts_of(sheet, "intervention")
    assert iv["step"] == 950_000 and facts_of(sheet, "window") == []


def test_a_cancelled_power_ends_at_the_cancel() -> None:
    ents = [power(1_001_000, duration=12_000), revert(1_002_500, 1_001_000)]
    view = RS.make_view(1_000_000, 1_009_000, ents)
    sheet = R.build_sheet(view, 1_000_000, 1_006_000)
    (iv,) = facts_of(sheet, "intervention")
    assert (iv["end"], iv["end_by"]) == (1_002_500, "elapsed")
    block = [x["text"] for x in R.candidates(sheet)[2]]
    assert block[1] == "It ended at step 1,002,500; food regrowth returned to its base rate."


# --- the quiet form on a real run -------------------------------------------------------------------------------


def test_a_quiet_chapter_of_the_tiny_run_is_a_headline_a_lead_the_turnover_and_what_did_not_last() -> None:
    view = R.load_run(DATA / "report_tiny")
    sheet = R.build_sheet(view, 1000, 2000)
    assert sheet["quiet"] is True
    rep = R.template_report(sheet)
    assert rep["headline"]["text"] == "A quiet chapter"
    assert rep["lead"] == "The population was 198 at the start and at the end of the chapter."
    assert [x["text"][:20] for x in rep["body"]] == ["128 organisms were b", "2 further kin group "]
    assert rep["interventions"] == []
    # QUIET_POP is 3%: the same chapter with a population that moved by 4% is not quiet
    moved = variant_pop(view, 2000, 206)
    assert R.build_sheet(moved, 1000, 2000)["quiet"] is False


def variant_pop(view: R.RunView, step: int, alive: int) -> R.RunView:
    """The view with the count of the living at ``step`` changed in its row (and record: the alive identity)."""
    rows = [
        {**r, "observers": {**r["observers"], "alive": alive}} if r["step"] == step else r for r in view.rows
    ]
    recs = []
    for rec in view.records:
        if rec["step"] == step:
            other = rec["summary"]["kin_groups"]["other"] + alive - R.alive_of(rec)
            rec = {
                **rec,
                "summary": {**rec["summary"], "kin_groups": {**rec["summary"]["kin_groups"], "other": other}},
            }
        recs.append(rec)
    return R.RunView(
        view.cfg,
        view.meta,
        tuple(rows),
        tuple(recs),
        view.entries,
        view.sidecars,
        view.storms,
        view.kin_states,
    )


# --- the storms: the chronicle's persistence rule, in a sheet ----------------------------------------------------


def tick(step: int, tracks: dict[int, float], events: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "step": step,
        "storms": [{"id": i, "x": 10.0 + i, "y": 20.0, "r": 7.0, "zeta": z} for i, z in tracks.items()],
        "events": events or [],
    }


def ev(
    kind: str, step: int, ident: int, *, name: str | None = None, age: int = 0, initial: bool = False
) -> dict[str, Any]:
    return {"kind": kind, "step": step, "id": ident, "name": name, "other": 0, "other_name": None, "x": 10.0 + ident,
            "y": 20.0, "hemi": ident % 2, "age": age, "initial": initial}  # fmt: skip


def script(base: int = 2000) -> list[dict[str, Any]]:
    """A tracker log of ticks every 25 steps from ``base``. Track 1 is there at the first tick (initial: not
    told). Track 2 is born at base + 25 and lasts three ticks (silent). Track 3 is born at base + 50, has 8
    detections at base + 225 (told as born then), is named at base + 300 and dissolves at base + 500 (told).
    Track 4 is born at base + 100 and named at base + 125 with 2 detections (a naming is told at once)."""
    lines = []
    for step in range(base, base + 626, 25):
        k = step - base
        tracks = {1: 0.3}
        events: list[dict[str, Any]] = []
        if k == 0:
            events.append(ev("born", step, 1, initial=True))
        if 25 <= k <= 75:
            tracks.update({2: 0.2})
        if k == 25:
            events.append(ev("born", step, 2))
        if k == 75:
            events.append(ev("dissolved", step, 2, age=50))
        if 50 <= k <= 500:
            tracks.update({3: 0.4 if k < 300 else 0.55})
        if k == 50:
            events.append(ev("born", step, 3))
        if k == 300:
            events.append(ev("named", step, 3, name="Alder"))
        if k == 500:
            events.append(ev("dissolved", step, 3, name="Alder", age=450))
        if 100 <= k <= 625:
            tracks.update({4: 0.6})
        if k == 100:
            events.append(ev("born", step, 4))
        if k == 125:
            events.append(ev("named", step, 4, name="Birch"))
        lines.append(tick(step, tracks, events))
    return lines


def storm_view(lines: list[dict[str, Any]]) -> R.RunView:
    base = RS.make_view(1000, 4000)
    return R.RunView(base.cfg, base.meta, base.rows, base.records, base.entries, base.sidecars, tuple(lines))


def test_a_storm_is_told_after_eight_detections_or_when_it_is_named_and_an_initial_one_is_not() -> None:
    sf = R.storm_fold(script(), 2500)
    told = [(t, e["kind"], e["id"]) for t, e in sf.told]
    # track 3: born at 2,050, 8 detections at the tick of 2,225 (2,050 + 7 * 25); track 4: named at once at 2,125
    assert (2225, "born", 3) in told and (2125, "named", 4) in told and (2300, "named", 3) in told
    assert (2500, "dissolved", 3) in told
    assert not [x for x in told if x[2] in (1, 2)]  # the initial track and the ephemeral one are never told
    # the log up to b only: nothing after it is read
    assert [t for t, _ in R.storm_fold(script(), 2200).told] == [2125]


def test_the_storm_facts_of_a_chapter_count_what_the_chronicle_told() -> None:
    view = storm_view(script())
    sheet = R.build_sheet(view, 2000, 3000)
    # (2,000, 3,000]: the whole script. Told: track 4 (named, first at 2,125), track 3 (born at 2,225, named at
    # 2,300, dissolved at 2,500). Tracked at the end (the last tick, 2,625): tracks 1 and 4; one named.
    (st,) = facts_of(sheet, "storms")
    assert (st["tracked"], st["named"], st["formed"], st["dissolved"], st["named_new"]) == (2, 1, 2, 1, 2)
    assert st["longest"] == 625  # track 1, there from the first tick (2,000) to the last (2,625)
    storms = {f["name"]: f for f in facts_of(sheet, "storm")}
    assert set(storms) == {"Alder", "Birch"}
    alder = storms["Alder"]
    assert (alder["named_step"], alder["end"], alder["age"], alder["hemisphere"]) == (
        2300,
        2500,
        450,
        "southern",
    )
    assert alder["ratio"] == pytest.approx(0.55 / 0.5)  # the peak in naming-threshold units
    birch = storms["Birch"]
    assert (birch["named_step"], birch["end"], birch["age"]) == (2125, None, None)
    # strongest first
    assert [f["name"] for f in facts_of(sheet, "storm")] == ["Birch", "Alder"]
    # the sentences
    body, heads, _block, ex = R.candidates(sheet)
    assert ex == []
    texts = [c["text"] for c in body]
    assert "2 storms were tracked at the end of the chapter, 1 of them named." in texts
    assert "2 storms formed and 1 dissolved during the chapter." in texts
    assert "Storm Alder dissolved at step 2,500 after 450 steps." in texts
    assert heads[0]["text"] == "2 storms reached named strength"


def test_the_storm_facts_and_entities_of_a_sphere_carry_the_planet_frame_z() -> None:
    """Review L8: a sphere's tracker states the planet-frame ``z``; the storm facts and the entities that cite
    them keep it (a box's carry none), so that a place in a report can fly a camera."""
    lines = script()
    for line in lines:
        for item in (*line["storms"], *line["events"]):
            item["z"] = -3.5
    view = storm_view(lines)
    sheet = R.build_sheet(view, 2000, 3000)
    storms = facts_of(sheet, "storm")
    assert storms and all(f["z"] == -3.5 for f in storms)
    box = facts_of(R.build_sheet(storm_view(script()), 2000, 3000), "storm")
    assert box and all("z" not in f for f in box)
    ((key, ent),) = RJ._entity_of(sheet, storms[0])
    assert key.startswith("S") and ent["z"] == -3.5 and "z" not in RJ._entity_of(sheet, box[0])[0][1]


def test_a_thread_fact_states_z_on_a_sphere_only() -> None:
    told = {"kind": "formed", "first_id": 7, "name": "Oak", "parent": 0, "into": 0, "first_step": 10,
            "end_step": None, "steps": 5, "size": 9, "peak": 9, "step": 15, "imp": 1, "initial": False,
            "a": 7, "pos": (1.0, 2.0, 3.0)}  # fmt: skip
    tf = R.ThreadFold((told,), None, None, 0)  # type: ignore[arg-type]
    (sph,) = R.thread_facts(tf, R.REPORT_V1, {}, True)
    (box,) = R.thread_facts(tf, R.REPORT_V1, {})
    assert (sph["x"], sph["y"], sph["z"]) == (1.0, 2.0, 3.0)
    assert (box["x"], box["y"]) == (1.0, 2.0) and "z" not in box


def test_a_chapter_with_no_tracker_log_has_no_storm_facts() -> None:
    sheet = R.build_sheet(RS.make_view(1000, 2000), 1000, 2000)
    assert not facts_of(sheet, "storms") and not facts_of(sheet, "storm")
    assert not any(c["template"].startswith("ST-") for c in R.candidates(sheet)[0])


def test_only_the_chapters_own_events_are_in_its_storm_facts() -> None:
    view = storm_view(script())
    early = R.build_sheet(view, 2000, 2500)  # (2,000, 2,500] with the rest of the log after it
    # told up to 2,500: track 4 named at 2,125, track 3 born at 2,225, named at 2,300 and dissolved at 2,500
    (st,) = facts_of(early, "storms")
    assert (st["formed"], st["dissolved"], st["named_new"]) == (2, 1, 2)
    late = R.build_sheet(view, 2500, 3000)
    (st2,) = facts_of(late, "storms")
    assert (st2["formed"], st2["dissolved"], st2["named_new"]) == (0, 0, 0) and not facts_of(late, "storm")
    assert (st2["tracked"], st2["named"]) == (2, 1)  # the state at the end of the chapter
    # the sum of the two halves is the whole
    whole = facts_of(R.build_sheet(view, 2000, 3000), "storms")[0]
    assert (st["formed"] + st2["formed"], st["named_new"] + st2["named_new"]) == (
        whole["formed"],
        whole["named_new"],
    )


def test_a_storm_told_at_the_first_step_of_the_chapter_belongs_to_the_chapter_before() -> None:
    view = storm_view(script(2375))  # track 4 is named, so told, at 2,500 and track 3 is told at 2,600
    before = facts_of(R.build_sheet(view, 2000, 2500), "storms")[0]
    after = facts_of(R.build_sheet(view, 2500, 3000), "storms")[0]
    assert (before["formed"], after["formed"]) == (1, 1)


def test_a_storm_row_with_deaths_and_a_starved_row_are_told_at_their_thresholds() -> None:
    view = RS.make_view(1000, 2000)

    def rows(deaths_storm: int, starved: int, base: int = 10):
        def edit(rs: list[dict[str, Any]]) -> list[dict[str, Any]]:
            return [
                {**r, "deaths_storm": deaths_storm if r["step"] == 1500 else 0,
                 "deaths_starved": starved if r["step"] == 1500 else base}
                for r in rs
            ]  # fmt: skip

        return edit

    for n, told in ((9, False), (10, True), (14, True)):
        s = R.build_sheet(variant(view, rows=rows(n, 10)), 1000, 2000)
        assert bool(facts_of(s, "stormdeaths")) is told, n
        if told:
            (f,) = facts_of(s, "stormdeaths")
            assert (f["n"], f["s"], f["steps"]) == (n, 1500, 100)
    # a row of 1.35 times the median is told (the median of ten rows of 10 and one of x), below it is not
    for starved, told in ((13, False), (14, True), (100, True)):
        s = R.build_sheet(variant(view, rows=rows(0, starved)), 1000, 2000)
        assert bool(facts_of(s, "starve")) is told, starved
        if told:
            (f,) = facts_of(s, "starve")
            assert (f["n"], f["s"], f["median"]) == (starved, 1500, 10)
    # exactly 1.35 times the median is told (the median 20: 27), one below is not
    for starved, told in ((26, False), (27, True)):
        s = R.build_sheet(variant(view, rows=rows(0, starved, 20)), 1000, 2000)
        assert bool(facts_of(s, "starve")) is told, starved
    # a chapter whose median is zero tells nothing about starvation
    zero = variant(
        view, rows=lambda rs: [{**r, "deaths_starved": 0 if r["step"] != 1500 else 50} for r in rs]
    )
    assert not facts_of(R.build_sheet(zero, 1000, 2000), "starve")


def test_houses_that_died_out_are_headlined_only_if_one_had_been_among_the_largest() -> None:
    view = RS.make_view(500, 4000)

    def extinct(rec: dict[str, Any], ids: list[int]) -> dict[str, Any]:
        ev_ = rec["house_events"]
        return {**rec, "house_events": {**ev_, "extinct": {"id": ids, "step": [rec["step"]] * len(ids)}}}

    # the record at 1,750 lists the houses 101, 102, 103; the one at 2,000 says 102 died out (it was listed at 1,750)
    # and 999 (it was not among the largest)
    world = variant(
        view, records=lambda recs: [extinct(r, [102, 999]) if r["step"] == 2000 else r for r in recs]
    )
    (h,) = facts_of(R.build_sheet(world, 1000, 2000), "houses")
    assert h["extinct_top"] == 1 and h["extinct"] >= 2
    none = variant(
        view, records=lambda recs: [extinct(r, [888, 999]) if r["step"] == 2000 else r for r in recs]
    )
    (h2,) = facts_of(R.build_sheet(none, 1000, 2000), "houses")
    assert h2["extinct_top"] == 0
    # the first record of the chapter is compared with the record at a
    first = variant(view, records=lambda recs: [extinct(r, [101]) if r["step"] == 1250 else r for r in recs])
    assert facts_of(R.build_sheet(first, 1000, 2000), "houses")[0]["extinct_top"] == 1
