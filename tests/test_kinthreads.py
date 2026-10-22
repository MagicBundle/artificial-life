"""Kin threads (``game/kinthreads.py``, G3b spec §2.2, DESIGN §34.1): a kin group through its id changes.

Synthetic record sequences pin each clause of the rule; the committed trim of the Mac-day play of planet p0
(``tests/data/play20k_logs``, ``make_play20k_logs.py``) is the regression pin: 139 raw kin rows become 45 told
events. A changed constant changes a pinned number on purpose and needs a ``KINTHREADS_VERSION`` bump."""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from alife.game import kinthreads as kt
from alife.game.kinthreads import KinThreads, kin_untold, roman

DATA = Path(__file__).resolve().parent / "data" / "play20k_logs"
STEP = 250


# --- synthetic records --------------------------------------------------------------------------------------


def tag(x: float, y: float = 0.0) -> tuple[float, float, float, float]:
    return (x, y, 0.0, 0.0)


def rec(
    k: int,
    groups: list[tuple[int, str, int, tuple[float, ...]]] = (),
    rows: list[tuple[str, int, int]] = (),
    *,
    origin: int = 0,
    kin_enter: int | None = 87,
) -> dict[str, Any]:
    """The social record of snapshot ``k`` (step 250 k). ``groups``: (id, name, size, tag); ``rows``: kin rows
    (kind, id, other)."""
    out: dict[str, Any] = {
        "step": STEP * k,
        "origin": origin,
        "kin_groups": {
            "id": [g[0] for g in groups],
            "name": [g[1] for g in groups],
            "size": [g[2] for g in groups],
            "tag": [list(g[3]) for g in groups],
            "x": [10.0 + g[0] for g in groups],
            "y": [20.0 + g[0] for g in groups],
        },
        "events": {
            "family": ["kin_group" for _ in rows],
            "kind": [r[0] for r in rows],
            "id": [r[1] for r in rows],
            "other": [r[2] for r in rows],
        },
    }
    if kin_enter is not None:
        out["thresholds"] = {"kin_enter": kin_enter}
    return out


def run(records: list[dict[str, Any]]) -> tuple[KinThreads, list[dict[str, Any]]]:
    fold, told = KinThreads(), []
    for r in records:
        told += fold.fold(r)
    return fold, told


def kinds(told: list[dict[str, Any]]) -> list[tuple[int, str, str]]:
    return [(e["step"] // STEP, e["kind"], e["name"]) for e in told]


# a record 0 with one anchor group, so that the groups of the scenarios below are not the initial cast
ANCHOR = (1, "Anchor", 500, tag(0.9, 0.9))


def after_anchor(schedule: dict[int, list[tuple[int, str, int, tuple[float, ...]]]], last: int, **kw: Any):
    return [rec(k, [ANCHOR, *schedule.get(k, [])], **kw) for k in range(last + 1)]


G = (2, "Brenn", 60, tag(0.0))


def regroup(gid: int, name: str = "Brenn", size: int = 60, at: tuple[float, ...] = tag(0.0)):
    return (gid, name, size, at)


# --- the rule, clause by clause -----------------------------------------------------------------------------


def test_a_group_that_lives_p_minus_one_snapshots_tells_nothing() -> None:
    fold, told = run(after_anchor({1: [G], 2: [G]}, 12))
    assert told == [] and [th.id for th in fold.threads.values() if th.est] == [1]


def test_a_group_that_lives_p_snapshots_is_formed_exactly_once_at_the_third() -> None:
    _, told = run(after_anchor({1: [G], 2: [G], 3: [G], 4: [G]}, 5))
    assert kinds(told) == [(3, "formed", "Brenn")]
    (e,) = told
    assert e["first_step"] == STEP and e["size"] == 60 and e["steps"] is None and not e["initial"]
    assert e["imp"] == 1 and e["parent"] is None and e["consumed"] is None


def test_a_flicker_inside_r_snapshots_is_one_thread_and_tells_nothing() -> None:
    """Listed at 1 to 3 (formed), gone at 4, 5 and 6 (R = 3), back at 7 under a new id with the same tag: the
    same thread, a flap, and nothing is told about it."""
    fold, told = run(after_anchor({1: [G], 2: [G], 3: [G], 7: [regroup(9)]}, 9))
    assert kinds(told) == [(3, "formed", "Brenn")]
    assert fold.flaps == 1
    th = fold.thread_of(9)
    assert th is not None and th.first_id == 2 and th.ids == [9] and not th.ended


def test_a_thread_absent_r_plus_one_snapshots_ends_exactly_once() -> None:
    _, told = run(after_anchor({1: [G], 2: [G], 3: [G]}, 12))
    # last listing 3; absent at 4, 5, 6 and the fourth snapshot of absence, 7, ends it
    assert kinds(told) == [(3, "formed", "Brenn"), (7, "ended", "Brenn")]
    assert told[1]["steps"] == (4 * STEP - 1 * STEP) and told[1]["end_step"] == 4 * STEP
    assert told[1]["life"] == 3 and told[1]["into"] is None


def test_a_group_back_after_r_plus_one_snapshots_is_a_new_thread() -> None:
    fold, told = run(
        after_anchor({1: [G], 2: [G], 3: [G], 8: [regroup(9)], 9: [regroup(9)], 10: [regroup(9)]}, 11)
    )
    # a new thread, not a flap; its name collides with the one that ended three snapshots ago (NAME_GAP is 8)
    assert kinds(told) == [(3, "formed", "Brenn"), (7, "ended", "Brenn"), (10, "formed", "Brenn II")]
    assert fold.flaps == 0


def test_the_nearest_absent_thread_wins_and_the_other_new_id_starts_a_candidate() -> None:
    a, b = regroup(2, "Aaa", at=tag(0.0)), regroup(3, "Bbb", at=tag(0.30))
    near_a, near_b = regroup(8, "Ccc", at=tag(0.02)), regroup(7, "Ddd", at=tag(0.05))
    sched = {1: [a, b], 2: [a, b], 3: [a, b], 5: [near_a, near_b]}
    fold, _ = run(after_anchor(sched, 6))
    # both new ids are within DELTA of thread "Aaa" only: the nearer (8, distance 0.02) takes it
    assert fold.thread_of(8).name == "Aaa" and fold.thread_of(7).name == "Ddd"
    assert fold.thread_of(7).first == 5 and not fold.thread_of(7).est
    assert fold.flaps == 1


def test_two_distinct_groups_a_quarter_apart_never_bridge() -> None:
    a, b = regroup(2, "Aaa", at=tag(0.0)), regroup(3, "Bbb", at=tag(0.25))
    fold, _ = run(after_anchor({1: [a], 2: [a], 3: [a], 5: [b]}, 6))
    assert fold.flaps == 0 and fold.thread_of(3).name == "Bbb"


def test_delta_is_a_strict_bound_and_ties_are_deterministic() -> None:
    a = regroup(2, "Aaa", at=tag(0.0))
    for d, bridged in ((kt.DELTA - 1e-9, 1), (kt.DELTA, 0)):
        fold, _ = run(after_anchor({1: [a], 2: [a], 3: [a], 5: [regroup(8, "Zzz", at=tag(d))]}, 6))
        assert fold.flaps == bridged, d
    # two threads at the same distance from the new id: the lower thread id takes it, whatever the list order
    a2, b2 = regroup(2, "Aaa", at=tag(0.0, 0.0)), regroup(3, "Bbb", at=tag(0.0, 0.0))
    new = regroup(8, "Zzz", at=tag(0.0, 0.05))
    seen = set()
    for order in ([a2, b2], [b2, a2]):
        sched = {1: order, 2: order, 3: order, 5: [new]}
        fold, _ = run(after_anchor(sched, 5))
        seen.add(fold.thread_of(8).name)
    assert seen == {"Aaa"}


def test_the_tag_and_size_of_a_thread_are_the_size_weighted_mean_of_its_ids() -> None:
    big, small = regroup(2, "Aaa", size=300, at=tag(0.0)), regroup(3, "Bbb", size=100, at=tag(0.4))
    fold, _ = run([rec(0, [big, small])])
    th_big, th_small = fold.thread_of(2), fold.thread_of(3)
    assert (th_big.size, th_small.size) == (300, 100) and th_big.peak == 300
    # one thread owning two listed ids: the second joins by id ownership after a bridge
    fold, _ = run(
        [
            rec(0, [ANCHOR, regroup(2, "Aaa", 300, tag(0.0))]),
            rec(1, [ANCHOR, regroup(2, "Aaa", 300, tag(0.0))]),
            rec(2, [ANCHOR]),
            rec(3, [ANCHOR, regroup(5, "X", 300, tag(0.0)), regroup(6, "Y", 100, tag(0.1))]),
        ]
    )
    th = fold.thread_of(5)
    assert fold.thread_of(6) is not th, "only one id joins an absent thread; the other starts a candidate"
    assert th.size == 300 and th.tag == [0.0, 0.0, 0.0, 0.0]


def test_a_thread_holding_two_ids_sums_their_sizes_and_weights_their_tags() -> None:
    """Rule 3 in general: a thread whose state lists two ids (possible in a restored table of another
    version, never made by the fold itself, which gives an absent thread one id) is their sum and their
    size-weighted mean."""
    base, _ = run([rec(0, [(5, "Pair", 300, tag(0.0))])])
    state = base.state()
    state["threads"][0]["ids"] = [5, 6]
    pair = KinThreads.from_state(state)
    pair.fold(rec(1, [(5, "Pair", 300, tag(0.0)), (6, "Pair", 100, tag(0.4))]))
    (th,) = pair.threads.values()
    assert th.size == 400 and th.ids == [5, 6] and th.peak == 400
    assert th.tag == pytest.approx([0.1, 0.0, 0.0, 0.0]), (
        "(300 x 0 + 100 x 0.4) / 400, not the plain mean 0.2"
    )


def test_the_initial_cast_is_silent_but_its_end_is_told_without_a_lifetime() -> None:
    _, told = run([rec(0, [ANCHOR, G]), rec(1, [ANCHOR, G]), *[rec(k, [ANCHOR]) for k in range(2, 8)]])
    assert kinds(told) == [(5, "ended", "Brenn")]
    (e,) = told
    assert e["initial"] and e["steps"] is None and e["first_step"] == 0 and e["life"] == 2


def test_an_origin_change_closes_the_fold_and_the_next_record_is_a_new_initial_cast() -> None:
    first = [rec(k, [ANCHOR, G]) for k in range(3)]
    second = [
        rec(k, [regroup(1, "Anchor", 500, tag(0.9, 0.9)), regroup(2, "Other", 60, tag(0.0))], origin=1)
        for k in range(3, 6)
    ]
    fold, told = run(first + second)
    assert told == [], "nothing is told at the restart: the old threads end silently, the new cast is initial"
    open_threads = fold.open_threads()
    assert {th.name for th in open_threads} == {"Anchor", "Other"} and all(
        th.init and th.est for th in open_threads
    )
    assert fold.flaps == 0, "an id renumbered by the restart is never bridged to an old thread"


def test_a_merged_row_one_record_after_the_last_listing_gives_merged_into() -> None:
    other = (3, "Other", 400, tag(0.5, 0.5))
    recs = [
        rec(0, [ANCHOR, other]),
        rec(1, [ANCHOR, other, G]),
        rec(2, [ANCHOR, other, G]),
        rec(3, [ANCHOR, other, G]),
        rec(4, [ANCHOR, other], [("merged", 2, 3)]),
        *[rec(k, [ANCHOR, other]) for k in range(5, 9)],
    ]
    _, told = run(recs)
    ended = [e for e in told if e["kind"] == "ended"]
    assert [(e["name"], e["into"], e["end_step"], e["steps"]) for e in ended] == [
        ("Brenn", "Other", 4 * STEP, 3 * STEP)
    ]
    assert ended[0]["b"] == 3 and ended[0]["consumed"] == 4 * STEP
    # the same sequence with a dissolved row, and with none: dissolved
    for row in ([("dissolved", 2, -1)], []):
        recs[4] = rec(4, [ANCHOR, other], row)
        _, told = run(recs)
        (e,) = [e for e in told if e["kind"] == "ended"]
        assert e["into"] is None
        assert e["consumed"] == (4 * STEP if row else None)


def test_a_stray_exit_row_for_a_thread_gone_longer_is_not_its_exit() -> None:
    """Only the record right after the last listing carries the row that explains the end: a row two records
    later (it should not exist) is not taken for it."""
    other = (3, "Other", 400, tag(0.5, 0.5))
    recs = [
        rec(0, [ANCHOR, other]),
        rec(1, [ANCHOR, other, G]),
        rec(2, [ANCHOR, other, G]),
        rec(3, [ANCHOR, other, G]),
        rec(4, [ANCHOR, other]),
        rec(5, [ANCHOR, other], [("merged", 2, 3)]),
        *[rec(k, [ANCHOR, other]) for k in range(6, 9)],
    ]
    _, told = run(recs)
    (e,) = [e for e in told if e["kind"] == "ended"]
    assert e["into"] is None and e["consumed"] is None


def test_an_empty_table_restored_still_takes_its_first_record_as_the_initial_cast() -> None:
    restored = KinThreads.from_state(json.loads(json.dumps(KinThreads().state())))
    assert restored.first_record is True
    told: list[dict[str, Any]] = []
    for k in range(4):
        told += restored.fold(rec(k, [ANCHOR, G]))
    assert told == [], "no formed: both groups are the initial cast"
    assert all(th.init and th.est for th in restored.open_threads())


def test_a_group_that_merged_away_and_came_back_within_r_is_never_told_as_merged() -> None:
    other = (3, "Other", 400, tag(0.5, 0.5))
    recs = [
        rec(0, [ANCHOR, other]),
        rec(1, [ANCHOR, other, G]),
        rec(2, [ANCHOR, other, G]),
        rec(3, [ANCHOR, other, G]),
        rec(4, [ANCHOR, other], [("merged", 2, 3)]),
        rec(5, [ANCHOR, other, regroup(9)], [("split", 9, 3)]),
        *[rec(k, [ANCHOR, other, regroup(9)]) for k in range(6, 9)],
    ]
    fold, told = run(recs)
    assert kinds(told) == [(3, "formed", "Brenn")] and fold.flaps == 1
    assert fold.thread_of(9).exit_kind is None, "the exit row is forgotten when the thread comes back"


def test_the_parent_of_a_split_group_is_named_in_its_formed_event() -> None:
    parent = (3, "Mother", 800, tag(0.5, 0.5))
    child = (4, "Child", 90, tag(0.0, 0.7))
    recs = [
        rec(0, [ANCHOR, parent]),
        rec(1, [ANCHOR, parent, child], [("split", 4, 3)]),
        rec(2, [ANCHOR, parent, child]),
        rec(3, [ANCHOR, parent, child]),
    ]
    _, told = run(recs)
    (e,) = told
    assert (e["name"], e["parent"], e["b"], e["consumed"]) == ("Child", "Mother", 3, STEP)


def test_importance_two_is_a_toast_sized_group_or_a_long_lived_large_one() -> None:
    small, large = regroup(2, "Aaa", size=100, at=tag(0.0)), regroup(3, "Bbb", size=348, at=tag(0.5))
    sched = {k: [small, large] for k in range(1, 4)}
    _, told = run(after_anchor(sched, 3))
    assert {e["name"]: e["imp"] for e in told} == {"Aaa": 1, "Bbb": 2}, "4 x kin_enter = 348 members"
    # a record without thresholds uses the default toast size, 350
    _, told = run(after_anchor(sched, 3, kin_enter=None))
    assert {e["name"]: e["imp"] for e in told} == {"Aaa": 1, "Bbb": 1}
    # an end is important when the peak is large and the thread lived TOAST_LIFE snapshots
    for life, imp in ((kt.TOAST_LIFE - 1, 1), (kt.TOAST_LIFE, 2)):
        sched = {k: [large] for k in range(1, 1 + life)}
        _, told = run(after_anchor(sched, life + 6))
        (end,) = [e for e in told if e["kind"] == "ended"]
        assert end["life"] == life and end["imp"] == imp, life


# --- names --------------------------------------------------------------------------------------------------


def test_the_name_is_the_first_listed_ids_and_frozen_across_re_formations() -> None:
    sched = {
        1: [G],
        2: [G],
        3: [G],
        5: [regroup(9, "Renamed")],
        6: [regroup(10, "Again")],
        7: [regroup(10, "Again")],
    }
    fold, told = run(after_anchor(sched, 9))
    assert fold.flaps == 2
    assert {e["name"] for e in told} == {"Brenn"} and fold.thread_of(10).name == "Brenn"


def test_a_name_shared_by_two_live_threads_takes_the_lowest_unused_suffix() -> None:
    a, b, c = (
        regroup(2, "Brenn", at=tag(0.0)),
        regroup(3, "Brenn", at=tag(0.5)),
        regroup(4, "Brenn", at=tag(0.0, 0.8)),
    )
    sched = {k: [a, b, c] for k in range(1, 4)}
    _, told = run(after_anchor(sched, 3))
    assert [e["name"] for e in told] == ["Brenn", "Brenn II", "Brenn III"]
    assert roman(4) == "IV" and roman(9) == "IX" and roman(14) == "XIV" and roman(1994) == "MCMXCIV"
    with pytest.raises(ValueError):
        roman(0)


@pytest.mark.parametrize(("gap", "name"), [(kt.NAME_GAP - 1, "Brenn II"), (kt.NAME_GAP, "Brenn")])
def test_a_name_is_free_again_after_name_gap_snapshots(gap: int, name: str) -> None:
    """A thread established, ended, and a same-name thread established ``gap`` snapshots after it ended: a
    collision only inside ``NAME_GAP`` (a later thread of the same house keeps the bare name)."""
    first = regroup(2, "Brenn", at=tag(0.0))
    last_first = 3  # listed at 1, 2, 3, ended at 7
    ended_at = last_first + kt.R + 1
    start2 = ended_at + gap - 2  # established (third listing) `gap` snapshots after the end
    second = regroup(5, "Brenn", at=tag(0.5))
    sched = {k: [first] for k in range(1, last_first + 1)} | {k: [second] for k in range(start2, start2 + 3)}
    _, told = run(after_anchor(sched, start2 + 3))
    formed = [e for e in told if e["kind"] == "formed"]
    ended = [e for e in told if e["kind"] == "ended"]
    assert ended[0]["step"] // STEP == ended_at and formed[1]["step"] // STEP == ended_at + gap
    assert formed[1]["name"] == name


def test_the_suffix_is_frozen_when_the_other_thread_ends() -> None:
    a, b = regroup(2, "Brenn", at=tag(0.0)), regroup(3, "Brenn", at=tag(0.5))
    sched = {k: [a, b] for k in range(1, 4)} | {k: [b] for k in range(4, 12)}
    fold, told = run(after_anchor(sched, 11))
    assert [e["name"] for e in told if e["kind"] == "formed"] == ["Brenn", "Brenn II"]
    assert [e["name"] for e in told if e["kind"] == "ended"] == ["Brenn"]
    assert fold.thread_of(3).shown == "Brenn II"


# --- state, chunking, refold --------------------------------------------------------------------------------


def p0_records() -> list[dict[str, Any]]:
    """The 80 records of the Mac-day window, steps 450,250 to 470,000."""
    recs = [json.loads(line) for line in (DATA / "social.jsonl").read_text().splitlines()]
    return [r for r in recs if r["step"] <= 470_000]


@pytest.fixture(scope="module")
def p0() -> tuple[list[dict[str, Any]], KinThreads, list[dict[str, Any]]]:
    recs = p0_records()
    fold, told = run(recs)
    return recs, fold, told


def test_the_state_round_trips_through_json_and_the_fold_continues_exactly(p0) -> None:
    """Play to T, quit, resume, play to T2: the same events and the same table as an uninterrupted play, at
    every cut (including a candidate that establishes after the resume)."""
    recs, whole, told = p0
    expected = json.dumps(whole.state(), sort_keys=True)
    candidates_after_cut = 0
    for cut in range(1, len(recs)):
        head, head_told = run(recs[:cut])
        saved = json.loads(json.dumps(head.state()))
        if any(not th.est and not th.ended for th in head.threads.values()):
            candidates_after_cut += 1
        resumed = KinThreads.from_state(saved)
        tail = []
        for r in recs[cut:]:
            tail += resumed.fold(r)
        assert head_told + tail == told, f"cut at {cut}"
        assert json.dumps(resumed.state(), sort_keys=True) == expected, f"cut at {cut}"
    assert candidates_after_cut > 10, "the cuts include many with an unestablished candidate"


def test_a_refold_of_the_log_equals_the_saved_table(p0) -> None:
    recs, whole, _ = p0
    fresh, _ = run(recs)
    assert fresh.state() == whole.state()
    # chunking cannot matter: the fold sees one record at a time
    shuffled = KinThreads()
    for r in recs:
        shuffled.fold(json.loads(json.dumps(r)))
    assert shuffled.state() == whole.state()


def test_a_state_of_another_version_or_constants_is_refused(p0) -> None:
    _, whole, _ = p0
    good = whole.state()
    for bad in (
        {**good, "format": "x"},
        {**good, "version": 99},
        {**good, "constants": "0" * 64},
        {k: v for k, v in good.items() if k != "threads"},
        {**good, "threads": [{"id": 1}]},
    ):
        with pytest.raises(ValueError):
            KinThreads.from_state(bad)


def test_the_table_is_bounded_and_drops_the_oldest_absent_threads_first() -> None:
    fold = KinThreads()
    fold.fold(rec(0, [(i + 1, f"N{i}", 30, tag((i % 20) * 0.05, (i // 20) * 0.5)) for i in range(80)]))
    assert len(fold.threads) == 80, "listed threads are never dropped"
    fold.fold(rec(1, [(100, "Late", 30, tag(0.9, 0.9))]))
    assert len(fold.threads) == kt.MAX_THREADS
    assert sorted(fold.threads)[:2] == [17, 18] and fold.thread_of(100) is not None
    assert set(fold.owner.values()) <= set(fold.threads)
    # an unestablished absent candidate goes before an established absent thread
    fold = KinThreads()
    spread = [tag(i * 0.3) for i in range(90)]
    fold.fold(rec(0, [(i + 1, f"E{i}", 30, spread[i]) for i in range(30)]))
    fold.fold(rec(1, [(i + 1, f"C{i}", 30, spread[i]) for i in range(30, 60)]))
    assert len(fold.threads) == 60
    fold.fold(rec(2, [(i + 1, f"D{i}", 30, spread[i]) for i in range(60, 90)]))
    assert len(fold.threads) == kt.MAX_THREADS
    assert all(i in fold.threads for i in range(1, 31)), "the established threads stay"
    assert [i for i in range(31, 61) if i in fold.threads] == list(range(57, 61)), "the oldest candidates go"
    assert all(i in fold.threads for i in range(61, 91)), "and the listed ones are never dropped"


# --- the Mac-day window -------------------------------------------------------------------------------------


def test_the_constants_are_pinned() -> None:
    """A change to a constant of the rule is a new rule: bump ``KINTHREADS_VERSION`` and re-pin."""
    assert (kt.KINTHREADS_VERSION, kt.P, kt.R, kt.DELTA, kt.TOAST_FACTOR, kt.TOAST_LIFE, kt.NAME_GAP) == (
        1, 3, 3, 0.15, 4, 8, 8,
    )  # fmt: skip
    assert kt.constants_digest() == "897a74dd3e1078a59f9ae3e84e787d44846dcba5338aa83291537f2528bf3f4a"


def test_p0_139_raw_kin_rows_become_45_told_events(p0) -> None:
    recs, fold, told = p0
    raw = sum(f == "kin_group" for r in recs for f in r["events"]["family"])
    assert raw == 139 and len(recs) == 80
    assert Counter(e["kind"] for e in told) == {"formed": 22, "ended": 23}
    assert fold.flaps == 36
    assert sum(e["imp"] == 2 for e in told) == 9
    minutes = (recs[-1]["step"] - recs[0]["step"] + STEP) / 864  # a snapshot is 250 steps at 14.4 steps/s
    assert round(len(told) / minutes, 2) == 1.94


def test_p0_kin_untold_is_raw_rows_minus_the_rows_events_consumed(p0) -> None:
    recs, _, told = p0
    assert kin_untold(recs, told, 0, 10**9) == 94 == 139 - sum(e["consumed"] is not None for e in told)
    # a told event consumes at most one row, counted in the chapter of the row and not of the event
    assert sum(e["consumed"] is not None for e in told) == 45, "every told event of the window consumed a row"
    chapters = [(450_000, 456_000), (456_000, 462_000), (462_000, 468_000), (468_000, 470_000)]
    assert [kin_untold(recs, told, a, b) for a, b in chapters] == [48, 10, 28, 8]
    assert sum(kin_untold(recs, told, a, b) for a, b in chapters) == 94


def test_p0_a_thread_has_one_name_for_its_whole_life(p0) -> None:
    _, _, told = p0
    by_thread: dict[int, set[str]] = {}
    for e in told:
        by_thread.setdefault(e["thread"], set()).add(e["name"])
    assert all(len(names) == 1 for names in by_thread.values())
    base = Counter(e["name"].split(" ")[0] for e in told if e["kind"] == "formed")
    assert base["Daudrall"] >= 4, "a house whose name many groups carry"
    suffixed = {e["thread"] for e in told if e["kind"] == "formed" and e["name"].endswith((" II", " III"))}
    assert len(suffixed) == 3


def test_a_refold_that_starts_mid_log_is_not_the_table(p0) -> None:
    """The reason the live job saves the table at the chapter start (spec §5.1): folding only the tail of the
    log gives other threads (an initial cast where the earlier fold had candidates and names)."""
    recs, whole, _ = p0
    tail, _ = run(recs[40:])
    assert tail.state() != whole.state()
    saved, _ = run(recs[:40])
    resumed = KinThreads.from_state(json.loads(json.dumps(saved.state())))
    for r in recs[40:]:
        resumed.fold(r)
    assert resumed.state() == whole.state()


def test_random_sequences_keep_the_invariants() -> None:
    rnd = random.Random(7)
    for trial in range(40):
        fold, nid = KinThreads(), 1
        live: list[tuple[int, str, int, tuple[float, ...]]] = []
        for k in range(30):
            if rnd.random() < 0.4:
                live.append(
                    (
                        nid,
                        f"N{rnd.randrange(4)}",
                        rnd.randrange(8, 400),
                        tag(rnd.random() * 0.6, rnd.random() * 0.6),
                    )
                )
                nid += 1
            live = [g for g in live if rnd.random() > 0.15]
            told = fold.fold(rec(k, live))
            ids = [g[0] for g in live]
            # every listed id is owned by one open thread, and no id by two
            assert sorted(fold.owner) == sorted({i for th in fold.open_threads() for i in th.ids})
            assert all(i in fold.owner for i in ids), trial
            for e in told:
                assert e["kind"] in ("formed", "ended") and e["imp"] in (1, 2) and len(e["name"]) < 20
            assert len(fold.threads) <= kt.MAX_THREADS
            assert all(not th.ended or th.est for th in fold.threads.values())
