"""§29.3 ``tracking``: persistent ids by overlap matching, grid clusters, dominance and the small helpers.

Host numpy only (no device array), so these tests do not depend on the MLX device. Each tracker case is a
synthetic sequence of snapshots, and a set-based reference implementation of the same rules is compared with
the sorted-array implementation over random sequences.
"""

from __future__ import annotations

import itertools
from collections import Counter, deque
from collections.abc import Iterable

import numpy as np
import pytest

from alife import tracking
from alife.tracking import (
    Dominance,
    DominanceState,
    Event,
    EventKind,
    Members,
    SystemTracker,
    TrackerState,
    deadband,
    grid_clusters,
    sticky_slots,
)

THETA = (3, 10)
BORN, SPLIT, MERGED, DISSOLVED = EventKind.BORN, EventKind.SPLIT, EventKind.MERGED, EventKind.DISSOLVED


def mem(d: dict[int, Iterable[int]]) -> Members:
    """Members from ``{label: keys}``; the pairs are listed label by label."""
    items = [(lab, list(ks)) for lab, ks in d.items()]
    label = [lab for lab, ks in items for _ in ks]
    key = [k for _, ks in items for k in ks]
    return Members(np.array(label, dtype=np.int64), np.array(key, dtype=np.int64))


def rng_(lo: int, hi: int) -> range:
    return range(lo, hi)


def ids_of(assign: tracking.Assignment) -> dict[int, int]:
    return dict(zip(assign.label.tolist(), assign.system.tolist(), strict=True))


# --- the cases of the spec ---------------------------------------------------------------------------------


def test_identity_has_no_events() -> None:
    tr = SystemTracker(THETA)
    a, ev, _ = tr.observe(0, mem({7: rng_(0, 10), 3: rng_(20, 25)}))
    assert ev == []
    assert ids_of(a) == {3: 2, 7: 1}  # ids by size
    a2, ev2, d2 = tr.observe(1, mem({7: rng_(0, 10), 3: rng_(20, 25)}))
    assert ev2 == [] and ids_of(a2) == {3: 2, 7: 1}
    assert d2.n_prev == 2 and d2.n_cur == 2 and d2.n_matched == 2
    assert d2.matched_jaccard.tolist() == [1.0, 1.0]


def test_turnover_edge_is_inclusive() -> None:
    # |A| = 10; B shares 6 and adds 10 new keys: union 20, 10 * 6 == 3 * 20, a match
    tr = SystemTracker(THETA)
    tr.observe(0, mem({1: rng_(0, 10)}))
    a, ev, d = tr.observe(1, mem({1: [*rng_(0, 6), *rng_(100, 110)]}))
    assert ev == [] and ids_of(a) == {1: 1}
    assert d.matched_jaccard.tolist() == [pytest.approx(0.3)]
    # one more new key: union 21, 60 < 63, no match: the system ends and a new one is born with provenance
    tr = SystemTracker(THETA)
    tr.observe(0, mem({1: rng_(0, 10)}))
    a, ev, d = tr.observe(1, mem({1: [*rng_(0, 6), *rng_(100, 111)]}))
    assert ids_of(a) == {1: 2}
    assert ev == [Event(1, DISSOLVED, 1, 10, 2, 6), Event(1, BORN, 2, 17, 1, 6)]
    assert d.n_matched == 0 and d.unmatched_best_jaccard.tolist() == [pytest.approx(6 / 21)]


def test_born_without_overlap_has_no_provenance() -> None:
    tr = SystemTracker(THETA)
    tr.observe(0, mem({1: rng_(0, 10)}))
    a, ev, _ = tr.observe(1, mem({1: rng_(0, 10), 2: rng_(50, 54)}))
    assert ev == [Event(1, BORN, 2, 4, -1, 0)]
    assert ids_of(a) == {1: 1, 2: 2}


def test_split_70_30() -> None:
    tr = SystemTracker(THETA)
    tr.observe(0, mem({5: rng_(0, 100)}))
    # B keeps J = 0.7; C has J = 0.3 with A too, but A is taken by the larger overlap
    a, ev, d = tr.observe(1, mem({5: rng_(0, 70), 9: rng_(70, 100)}))
    assert ids_of(a) == {5: 1, 9: 2}
    assert ev == [Event(1, SPLIT, 2, 30, 1, 30)]
    assert d.n_matched == 1


def test_merged_into_the_larger_system() -> None:
    tr = SystemTracker(THETA)
    tr.observe(0, mem({1: rng_(0, 70), 2: rng_(70, 100)}))  # A (70) is id 1, B (30) is id 2
    a, ev, _ = tr.observe(1, mem({8: rng_(0, 100)}))
    assert ids_of(a) == {8: 1}
    assert ev == [Event(1, MERGED, 2, 30, 1, 30)]


def test_dissolved_when_it_vanishes() -> None:
    tr = SystemTracker(THETA)
    tr.observe(0, mem({1: rng_(0, 10), 2: rng_(20, 25)}))
    a, ev, d = tr.observe(1, mem({1: rng_(0, 10)}))
    assert ev == [Event(1, DISSOLVED, 2, 5, -1, 0)]
    assert ids_of(a) == {1: 1}
    assert d.unmatched_best_jaccard.tolist() == [0.0]


def test_dissolved_when_members_went_to_an_unmatched_new_system() -> None:
    tr = SystemTracker(THETA)
    tr.observe(0, mem({1: rng_(0, 10)}))
    a, ev, _ = tr.observe(1, mem({6: [0, 1, 2, *rng_(100, 120)]}))  # J = 3 / 30
    assert ids_of(a) == {6: 2}
    assert ev == [Event(1, DISSOLVED, 1, 10, 2, 3), Event(1, BORN, 2, 23, 1, 3)]


def test_fragmentation_into_two_halves_below_theta() -> None:
    tr = SystemTracker(THETA)
    tr.observe(0, mem({1: rng_(0, 100)}))
    h1 = [*rng_(0, 40), *rng_(1000, 1100)]  # 40 shared, J = 40 / 200
    h2 = [*rng_(40, 75), *rng_(2000, 2100)]  # 35 shared, J = 35 / 200
    a, ev, _ = tr.observe(1, mem({50: h1, 20: h2}))
    assert ids_of(a) == {50: 2, 20: 3}  # new ids by (-size, label): h1 has 140 members, h2 135
    assert ev == [
        Event(1, DISSOLVED, 1, 100, 2, 40),
        Event(1, BORN, 2, 140, 1, 40),
        Event(1, BORN, 3, 135, 1, 35),
    ]


def test_greedy_prefers_the_larger_overlap_over_a_higher_jaccard() -> None:
    tr = SystemTracker(THETA)
    tr.observe(0, mem({1: rng_(0, 50), 2: rng_(0, 20)}))  # id 1 = 50 members, id 2 = 20 members (shared keys)
    # L shares 25 with id 1 (J 0.5) and 20 with id 2 (J 0.8): the larger overlap, id 1, takes L
    a, ev, _ = tr.observe(1, mem({9: rng_(0, 25)}))
    assert ids_of(a) == {9: 1}
    assert ev == [Event(1, MERGED, 2, 20, 1, 20)]


def test_ties_go_to_the_smaller_id_and_the_smaller_label() -> None:
    tr = SystemTracker(THETA)
    tr.observe(0, mem({10: [0, 1, 2, 3], 20: [4, 5, 6, 7]}))  # ids 1 and 2 (equal sizes: by label)
    a, ev, _ = tr.observe(1, mem({5: [2, 3, 4, 5]}))  # c = 2 with both, J = 1/3
    assert ids_of(a) == {5: 1}
    assert ev == [Event(1, MERGED, 2, 4, 1, 2)]
    # one previous system, two labels with equal overlap: the smaller label keeps the id
    tr = SystemTracker(THETA)
    tr.observe(0, mem({1: rng_(0, 8)}))
    a, ev, _ = tr.observe(1, mem({40: [*rng_(0, 4), 100, 101, 102], 30: [*rng_(4, 8), 200, 201, 202]}))
    assert ids_of(a) == {30: 1, 40: 2}
    assert ev == [Event(1, SPLIT, 2, 7, 1, 4)]


def test_input_order_does_not_matter() -> None:
    rs = np.random.default_rng(5)
    first = {3: rng_(0, 30), 8: rng_(30, 50), 1: rng_(100, 112)}
    second = {7: [*rng_(0, 20), 300], 2: rng_(20, 50), 9: rng_(100, 104), 4: rng_(500, 520)}
    out = []
    for shuffle in (False, True):
        tr = SystemTracker(THETA)
        runs = []
        for step, d in enumerate((first, second)):
            m = mem(d)
            if shuffle:
                o = rs.permutation(m.label.size)
                m = Members(m.label[o], m.key[o])
            a, ev, _ = tr.observe(step, m)
            runs.append((a.label.tolist(), a.system.tolist(), a.size.tolist(), ev))
        out.append((runs, tr.state()))
    assert out[0][0] == out[1][0]
    assert all(np.array_equal(x, y) for x, y in zip(out[0][1][:2], out[1][1][:2], strict=True))


def test_credit_lifts_the_jaccard_over_theta() -> None:
    first = mem({1: rng_(0, 10)})
    # three old members and seven newborns (uids 10..16, children of members 0..6): J = 3/17 without credit
    cur = mem({1: [0, 1, 2, *rng_(10, 17)]})
    tr = SystemTracker(THETA)
    tr.observe(0, first)
    a, ev, _ = tr.observe(1, cur)
    assert ids_of(a) == {1: 2} and ev[0].kind == DISSOLVED
    children, parents = np.arange(10, 17), np.arange(0, 7)
    tr = SystemTracker(THETA)
    tr.observe(0, first)
    a, ev, d = tr.observe(1, cur, credit=(children, parents))
    assert ids_of(a) == {1: 1} and ev == []
    assert d.matched_jaccard.tolist() == [pytest.approx(10 / 17)]
    # a credit whose parent was not a member is ignored
    tr = SystemTracker(THETA)
    tr.observe(0, first)
    a, ev, _ = tr.observe(1, cur, credit=(children, parents + 50))
    assert ids_of(a) == {1: 2} and ev[0].kind == DISSOLVED


def test_credit_goes_to_every_system_of_the_parent() -> None:
    tr = SystemTracker(THETA)
    tr.observe(0, mem({1: rng_(0, 10), 2: rng_(0, 8)}))  # keys 0..7 are in both systems
    # one current system holds ten newborns, all children of key 0; it joins neither system without credit
    cur = mem({5: rng_(100, 110)})
    a, ev, _ = tr.observe(1, cur, credit=(np.arange(100, 110), np.zeros(10, np.int64)))
    # A'_1 = 0..9 plus the ten children, A'_2 = 0..7 plus the ten: c = 10 with both, J = 10/20 and 10/18
    assert ids_of(a) == {5: 1}  # ties in c: the smaller id
    assert ev == [Event(1, MERGED, 2, 8, 1, 10)]


def test_footprint_keys_shared_by_two_systems() -> None:
    tr = SystemTracker(THETA)
    first = {1: [1, 2, 3, 4], 2: [3, 4, 5, 6]}
    a, ev, _ = tr.observe(0, mem(first))
    assert ev == [] and ids_of(a) == {1: 1, 2: 2}
    assert tr.state().key.tolist() == [1, 2, 3, 3, 4, 4, 5, 6]
    assert tr.state().system.tolist() == [1, 1, 1, 2, 1, 2, 2, 2]
    a, ev, d = tr.observe(1, mem({9: [3, 4, 5, 6], 8: [1, 2, 3, 4]}))
    assert ids_of(a) == {8: 1, 9: 2} and ev == []
    assert d.matched_jaccard.tolist() == [1.0, 1.0]


def test_first_snapshot_gives_ids_to_the_admitted_by_size() -> None:
    tr = SystemTracker(THETA)
    cur = mem({4: rng_(0, 5), 2: rng_(10, 19), 6: rng_(30, 37), 1: rng_(40, 47)})
    a, ev, d = tr.observe(0, cur, admit=np.array([1, 4, 6]))  # label 2 (9 members) is not admitted
    assert ev == []
    assert a.label.tolist() == [1, 2, 4, 6]
    assert a.system.tolist() == [1, -1, 3, 2]  # 6 and 1 have 7 members: ties to the smaller label
    assert a.size.tolist() == [7, 9, 5, 7]
    assert d.n_admitted == 3 and d.n_cur == 4 and d.n_prev == 0
    assert tr.state().system.size == 19  # the unadmitted label's members are not stored
    assert tr.state().next_id == 4


def test_state_round_trip_then_identical_tracking() -> None:
    seq = [
        {1: rng_(0, 30), 2: rng_(40, 60)},
        {1: rng_(5, 35), 2: rng_(40, 60), 3: rng_(100, 110)},
        {1: rng_(0, 20), 3: rng_(100, 125), 4: rng_(200, 209)},
        {5: rng_(0, 20), 3: rng_(100, 110)},
    ]
    full = SystemTracker(THETA)
    outs = [full.observe(i, mem(d)) for i, d in enumerate(seq)]
    half = SystemTracker(THETA)
    for i, d in enumerate(seq[:2]):
        half.observe(i, mem(d))
    saved = half.state()
    again = SystemTracker(
        THETA, TrackerState(saved.system.copy(), saved.key.copy(), saved.next_id, saved.step)
    )
    for i in (2, 3):
        a, ev, _ = again.observe(i, mem(seq[i]))
        assert ev == outs[i][1]
        assert [x.tolist() for x in a] == [x.tolist() for x in outs[i][0]]
    assert [x.tolist() if hasattr(x, "tolist") else x for x in again.state()] == [
        x.tolist() if hasattr(x, "tolist") else x for x in full.state()
    ]
    # the state is a copy: changing the returned arrays does not touch the tracker
    s = full.state()
    s.key[:] = -1
    assert full.state().key.min() >= 0


def test_state_validation() -> None:
    ok = TrackerState(np.array([1, 2]), np.array([4, 4]), 3, 7)
    SystemTracker(THETA, ok)
    bad = [
        TrackerState(np.array([1, 2]), np.array([4, 4]), 2, 7),  # id 2 >= next_id
        TrackerState(np.array([2, 1]), np.array([4, 4]), 3, 7),  # not sorted by (key, system)
        TrackerState(np.array([1, 1]), np.array([4, 4]), 3, 7),  # duplicate pair
        TrackerState(np.array([0]), np.array([4]), 3, 7),  # id 0
        TrackerState(np.array([1]), np.array([4, 5]), 3, 7),
        TrackerState(np.array([1]), np.array([4]), 0, 7),  # next_id < 1
        TrackerState(np.array([1]), np.array([4]), 3, -1),  # memberships before the first snapshot
    ]
    for state in bad:
        with pytest.raises(ValueError):
            SystemTracker(THETA, state)
    for theta in ((0, 10), (11, 10), (-1, 3)):
        with pytest.raises(ValueError, match="theta"):
            SystemTracker(theta)
    with pytest.raises(ValueError, match="differ"):
        SystemTracker(THETA).observe(0, Members(np.array([1, 2]), np.array([1])))
    with pytest.raises(TypeError, match="integer"):
        SystemTracker(THETA).observe(0, Members(np.array([1.5]), np.array([1])))


# --- admission ---------------------------------------------------------------------------------------------


def test_unadmitted_candidates_get_no_id_no_event_and_are_not_stored() -> None:
    tr = SystemTracker(THETA)
    tr.observe(0, mem({1: rng_(0, 20)}))
    cur = mem({1: rng_(0, 20), 2: rng_(100, 106), 3: rng_(200, 210)})
    a, ev, d = tr.observe(1, cur, admit=np.array([1, 3]))
    assert ids_of(a) == {1: 1, 2: -1, 3: 2}
    assert ev == [Event(1, BORN, 2, 10, -1, 0)]
    assert d.n_cur == 3 and d.n_admitted == 2
    assert set(tr.state().key.tolist()) == {*range(20), *range(200, 210)}
    # label 2 is still a candidate next time but cannot found a system: nothing is stored or reported
    a, ev, _ = tr.observe(2, cur, admit=np.array([1, 3]))
    assert ev == [] and ids_of(a)[2] == -1


def test_a_matched_candidate_outside_admit_keeps_its_id() -> None:
    tr = SystemTracker(THETA)
    tr.observe(0, mem({1: rng_(0, 20)}))
    a, ev, _ = tr.observe(1, mem({1: rng_(0, 18)}), admit=np.array([], dtype=np.int64))
    assert ev == [] and ids_of(a) == {1: 1}
    assert tr.state().next_id == 2


def test_a_system_oscillating_around_enter_emits_nothing_after_entering() -> None:
    # enter at 9 members, leave below 4: the caller passes every group of >= 4 and admits those of >= 9
    tr = SystemTracker(THETA)
    sizes = [9, 8, 9, 8, 9, 8, 9, 8, 9, 8]
    events = []
    for step, n in enumerate(sizes):
        cur = mem({1: rng_(0, n)})
        admit = np.array([1]) if n >= 9 else np.array([], dtype=np.int64)
        a, ev, _ = tr.observe(step, cur, admit=admit)
        events += ev
        assert ids_of(a) == {1: 1}
    assert events == []


def test_a_system_below_enter_that_never_reaches_it_is_never_tracked() -> None:
    tr = SystemTracker(THETA)
    events = []
    for step, n in enumerate([8, 7, 8, 7, 8, 7, 8, 7, 8, 7]):
        a, ev, _ = tr.observe(step, mem({1: rng_(0, n)}), admit=np.array([], dtype=np.int64))
        events += ev
        assert ids_of(a) == {1: -1}
    assert events == [] and tr.state().system.size == 0 and tr.state().next_id == 1


def test_a_system_leaving_the_candidates_ends() -> None:
    # the candidate set no longer holds the system: DISSOLVED; an overlapping label without an id is ignored
    tr = SystemTracker(THETA)
    tr.observe(0, mem({1: rng_(0, 10)}))
    a, ev, _ = tr.observe(1, mem({2: [0, 1, 2, *rng_(100, 120)]}), admit=np.array([], dtype=np.int64))
    assert ids_of(a) == {2: -1}
    assert ev == [Event(1, DISSOLVED, 1, 10, -1, 0)]
    # the same, with a continuing system that shares two members: MERGED into it
    tr = SystemTracker(THETA)
    tr.observe(0, mem({1: rng_(0, 10), 2: rng_(8, 40)}))  # id 1 = 32 members (label 2), id 2 = 10 (label 1)
    a, ev, _ = tr.observe(1, mem({2: rng_(8, 40), 3: [0, 1, 2, *rng_(100, 120)]}), admit=np.array([2]))
    assert ids_of(a) == {2: 1, 3: -1}
    assert ev == [Event(1, MERGED, 2, 10, 1, 2)]


# --- empty inputs ------------------------------------------------------------------------------------------


def test_empty_inputs() -> None:
    none = Members(np.array([], dtype=np.int64), np.array([], dtype=np.int64))
    # nothing at all, at the first snapshot and after it
    tr = SystemTracker(THETA)
    a, ev, d = tr.observe(0, none)
    assert ev == [] and a.label.size == 0 and a.system.size == 0 and a.size.size == 0
    assert (d.n_prev, d.n_cur, d.n_admitted, d.n_matched) == (0, 0, 0, 0)
    a, ev, _ = tr.observe(1, none)
    assert ev == [] and tr.state().next_id == 1
    # no previous system (not the first snapshot): every admitted label is BORN with no provenance
    a, ev, _ = tr.observe(2, mem({4: rng_(0, 5), 9: rng_(10, 20)}))
    assert ev == [Event(2, BORN, 1, 10, -1, 0), Event(2, BORN, 2, 5, -1, 0)]
    # no current candidate: every previous system is DISSOLVED
    a, ev, d = tr.observe(3, none)
    assert ev == [Event(3, DISSOLVED, 1, 10, -1, 0), Event(3, DISSOLVED, 2, 5, -1, 0)]
    assert d.n_prev == 2 and d.unmatched_best_jaccard.tolist() == [0.0, 0.0]
    assert tr.state().system.size == 0 and tr.state().next_id == 3 and tr.state().step == 3
    # the empty state round-trips, before and after a first snapshot, and the next ids continue
    fresh = SystemTracker(THETA)
    again = SystemTracker(THETA, fresh.state())
    assert [x.tolist() if hasattr(x, "tolist") else x for x in again.state()] == [[], [], 1, -1]
    a, ev, _ = SystemTracker(THETA, tr.state()).observe(4, mem({1: rng_(0, 4)}))
    assert ev == [Event(4, BORN, 3, 4, -1, 0)]


# --- a reference implementation over sets, compared on random sequences -------------------------------------


def ref_step(prev, cur, theta, admit, next_id, step, credit):
    """The rules of §3.2 on Python sets: ``prev`` {id: keys}, ``cur`` {label: keys}.

    Returns ``(label -> id, events, next_id, {id: keys})``.
    """
    thn, thd = theta
    a_ext = {p: set(ks) for p, ks in prev.items()}
    for child, parent in credit:
        for p, ks in prev.items():
            if parent in ks:
                a_ext[p].add(child)
    ov = {}
    for p, a in a_ext.items():
        for lab, b in cur.items():
            n = len(a & b)
            if n:
                ov[p, lab] = n
    passing = [
        (-n, p, lab) for (p, lab), n in ov.items() if thd * n >= thn * (len(a_ext[p]) + len(cur[lab]) - n)
    ]
    mp, ml = {}, {}
    for _, p, lab in sorted(passing):
        if p not in mp and lab not in ml:
            mp[p], ml[lab] = lab, p
    ids = dict(ml)
    events, new_events = [], []
    for lab in sorted((x for x in cur if x not in ml and x in admit), key=lambda x: (-len(cur[x]), x)):
        over = [(n, -p) for (p, lb), n in ov.items() if lb == lab]
        n, negp = max(over) if over else (0, 0)
        p_star = -negp
        kind = SPLIT if n > 0 and p_star in mp else BORN
        ids[lab] = next_id
        new_events.append(Event(step, kind, next_id, len(cur[lab]), p_star if n > 0 else -1, n))
        next_id += 1
    for p in sorted(prev):
        if p in mp:
            continue
        over = [(n, -lb) for (q, lb), n in ov.items() if q == p and lb in ids]
        n, neglab = max(over) if over else (0, 0)
        lab = -neglab
        merged = n > 0 and lab in ml
        events.append(
            Event(step, MERGED if merged else DISSOLVED, p, len(prev[p]), ids[lab] if n > 0 else -1, n)
        )
    state = {i: set(cur[lab]) for lab, i in ids.items()}
    return ids, events + new_events, next_id, state


def _evolve(rs: np.random.Generator, systems: list[set[int]]) -> list[set[int]]:
    out: list[set[int]] = []
    for s in systems:
        r = rs.random()
        if r < 0.1:
            continue
        keep = {k for k in s if rs.random() > rs.uniform(0, 0.6)}
        t = keep | {int(k) for k in rs.integers(0, 400, int(rs.integers(0, 12)))}
        if r > 0.85 and len(t) > 3:
            ks = sorted(t)
            out += [set(ks[: len(ks) // 2]), set(ks[len(ks) // 2 :])]
        else:
            out.append(t)
    if rs.random() < 0.3 and len(out) > 1:
        out.append(out.pop() | out.pop())
    if rs.random() < 0.5:
        out.append({int(k) for k in rs.integers(0, 400, int(rs.integers(1, 25)))})
    return [s for s in out if s]


def _run_random(seed: int) -> Counter:
    """One random 14-snapshot sequence checked against ``ref_step``; returns what it exercised."""
    seen = Counter()
    rs = np.random.default_rng(seed)
    tr = SystemTracker(THETA)
    systems: list[set[int]] = []
    prev: dict[int, set[int]] = {}
    next_id, seen_ids = 1, set()
    for step in range(14):
        systems = _evolve(rs, systems) if step else _evolve(rs, [set(range(k, k + 30)) for k in (0, 60, 130)])
        labels = rs.choice(10**6, size=len(systems), replace=False).tolist()
        cur = dict(zip(labels, systems, strict=True))
        admit_set = {lab for lab, s in cur.items() if len(s) >= 6}
        credit = []
        if step and rs.random() < 0.5:
            known = sorted({k for ks in prev.values() for k in ks}) or [0]
            credit = [
                (1000 + 10 * step + j, int(rs.choice([*known, -5]))) for j in range(int(rs.integers(1, 15)))
            ]
        m = mem({lab: sorted(ks) for lab, ks in cur.items()})
        o = rs.permutation(m.label.size)
        m = Members(m.label[o], m.key[o])
        cr = (
            (np.array([c for c, _ in credit], np.int64), np.array([p for _, p in credit], np.int64))
            if credit
            else None
        )
        adm = np.array(sorted(admit_set), dtype=np.int64)
        assign, events, diag = tr.observe(step, m, credit=cr, admit=adm)
        if step == 0:
            order = sorted((lab for lab in cur if lab in admit_set), key=lambda x: (-len(cur[x]), x))
            ref_ids = {lab: 1 + i for i, lab in enumerate(order)}
            ref_events, ref_next = [], 1 + len(order)
            ref_state = {i: set(cur[lab]) for lab, i in ref_ids.items()}
        else:
            ref_ids, ref_events, ref_next, ref_state = ref_step(
                prev, cur, THETA, admit_set, next_id, step, credit
            )
        assert events == ref_events, (seed, step)
        assert ids_of(assign) == {lab: ref_ids.get(lab, -1) for lab in cur}, (seed, step)
        assert assign.label.tolist() == sorted(cur)
        assert assign.size.tolist() == [len(cur[lab]) for lab in sorted(cur)]
        st = tr.state()
        got: dict[int, set[int]] = {}
        for sy, k in zip(st.system.tolist(), st.key.tolist(), strict=True):
            got.setdefault(sy, set()).add(k)
        assert got == ref_state, (seed, step)
        assert st.next_id == ref_next and st.step == step
        # invariants: ids are never reused or revived
        new_ids = {e.system for e in events if e.kind in (BORN, SPLIT)}
        assert not new_ids & seen_ids
        seen_ids |= new_ids | set(ref_ids.values())
        assert diag.n_matched == len(ref_ids) - len(new_ids) if step else True
        prev, next_id = ref_state, ref_next
        seen.update(e.kind.name for e in events)
        seen["matched"] += diag.n_matched
        seen["credited"] += credit != []
        seen["unadmitted"] += sum(1 for lab in cur if lab not in admit_set)
    return seen


@pytest.mark.parametrize("seed", range(40))
def test_tracker_matches_the_set_reference_on_random_sequences(seed: int) -> None:
    _run_random(seed)


def test_the_random_sequences_exercise_every_rule() -> None:
    total = sum((_run_random(seed) for seed in range(40)), Counter())
    assert all(
        total[k] >= 25 for k in ("BORN", "SPLIT", "MERGED", "DISSOLVED", "matched", "credited", "unadmitted")
    )


def test_ids_are_never_resurrected() -> None:
    tr = SystemTracker(THETA)
    tr.observe(0, mem({1: rng_(0, 20)}))
    tr.observe(1, mem({}))  # the system ends
    a, ev, _ = tr.observe(2, mem({1: rng_(0, 20)}))  # the same members come back: a new id
    assert ids_of(a) == {1: 2} and ev == [Event(2, BORN, 2, 20, -1, 0)]


# --- sticky slots ------------------------------------------------------------------------------------------


def _slots(prev, ranked, slack):
    return sticky_slots(np.array(prev, dtype=np.int64), np.array(ranked, dtype=np.int64), slack).tolist()


def test_sticky_slots() -> None:
    # F = 3, slack 1: a slotted id keeps its slot inside ranked[:4], loses it beyond
    assert _slots([10, 11, 12], [12, 11, 10, 13, 14], 1) == [10, 11, 12]  # a reordering within the top F
    assert _slots([10, 11, 12], [12, 11, 13, 10, 14], 1) == [10, 11, 12]  # rank 3 = F + 0 is inside the slack
    assert _slots([10, 11, 12], [12, 11, 13, 14, 10], 1) == [13, 11, 12]  # rank 4 is out: 10 loses slot 0
    # freed slots are filled in ascending order by rank
    assert _slots([10, 11, 12], [20, 21, 22, 23, 11], 0) == [20, 21, 22]
    assert _slots([10, -1, 12, -1], [30, 12, 31, 32, 10], 2) == [10, 30, 12, 31]
    # fewer ids than slots leave -1; an untracked id frees its slot
    assert _slots([10, 11, 12], [12], 1) == [-1, -1, 12]
    assert _slots([-1, -1, -1], [5, 6], 4) == [5, 6, -1]
    assert _slots([-1, -1], [], 4) == [-1, -1]
    # at most `slack` slotted ids rank outside the top F
    rs = np.random.default_rng(3)
    prev = np.arange(16)
    for _ in range(200):
        ranked = rs.permutation(np.arange(40))
        out = sticky_slots(prev, ranked, 4)
        held = out[out >= 0]
        assert len(set(held.tolist())) == held.size
        rank = {int(v): i for i, v in enumerate(ranked)}
        assert sum(rank[int(v)] >= 16 for v in held) <= 4
        assert all(rank[int(v)] < 20 for v in held)
        prev = out


def test_sticky_slots_never_regives_a_dropped_slot() -> None:
    # an id beyond F + slack loses its slot and cannot take a freed one in the same call
    assert _slots([7, -1, -1], [1, 2, 3, 4, 7], 1) == [1, 2, 3]


# --- the dead-band -----------------------------------------------------------------------------------------


def test_deadband() -> None:
    assert deadband(-1, 5000, 3) == (5000, True)  # the first value is taken
    ref = 8 * 4096
    assert deadband(ref, ref + 1, 3) == (ref, False)
    assert deadband(ref, ref - 1, 3) == (ref, False)
    assert deadband(ref, ref + (ref >> 3), 3) == (ref, False)  # exactly 12.5% is inside
    assert deadband(ref, ref + (ref >> 3) + 1, 3) == (ref + (ref >> 3) + 1, True)
    assert deadband(ref, ref - (ref >> 3) - 1, 3) == (ref - (ref >> 3) - 1, True)
    assert deadband(0, 5, 3) == (5, True) and deadband(0, 0, 3) == (
        0,
        False,
    )  # ref 0 follows any positive value
    # ±1 around a multiple of 4,096 never moves the reference
    r = 32768
    for v in (32767, 32768, 32769, 32768, 32767):
        r, moved = deadband(r, v, 3)
        assert r == 32768 and not moved
    assert isinstance(deadband(np.int64(10), np.int64(30), 3)[0], int)


# --- helpers -----------------------------------------------------------------------------------------------


def test_lookup_table_and_rank_by_size() -> None:
    t = tracking.lookup_table(10, np.array([2, 5, 9]), np.array([7, 8, 9]), -1)
    assert t.dtype == np.int32 and t.tolist() == [-1, -1, 7, -1, -1, 8, -1, -1, -1, 9]
    assert (
        tracking.lookup_table(4, np.array([], dtype=np.int64), np.array([], dtype=np.int64), 3).tolist()
        == [3] * 4
    )
    assert tracking.lookup_table(4, np.array([1, 2]), 5, 0).tolist() == [0, 5, 5, 0]
    for bad in ([4], [-1]):
        with pytest.raises(ValueError, match="keys"):
            tracking.lookup_table(4, np.array(bad), 1, 0)
    ranked = tracking.rank_by_size(np.array([9, 3, 7, 5]), np.array([10, 40, 10, 40]))
    assert ranked.tolist() == [3, 5, 7, 9]  # by (-size, id)
    assert tracking.rank_by_size(np.array([], dtype=np.int64), np.array([], dtype=np.int64)).size == 0


def test_circular_mean() -> None:
    assert tracking.circular_mean(np.array([10.0]), 100.0) == pytest.approx(10.0)
    across = tracking.circular_mean(np.array([95.0, 5.0]), 100.0)  # the mean of two points across the seam
    assert min(across, 100.0 - across) < 1e-9
    assert tracking.circular_mean(np.array([90.0, 98.0, 6.0]), 100.0) == pytest.approx(98.0, abs=2.0)
    assert 0.0 <= tracking.circular_mean(np.array([99.999999, 99.999998]), 100.0) < 100.0
    assert tracking.circular_mean(np.array([20.0, 30.0, 40.0]), 100.0) == pytest.approx(30.0)
    with pytest.raises(ValueError, match="empty"):
        tracking.circular_mean(np.array([]), 100.0)


# --- grid clusters -----------------------------------------------------------------------------------------


def ref_clusters(counts, occ_min, periodic):
    """Breadth-first components of the occupied cells plus the attachment rule, on an n-d array."""
    shape = counts.shape
    out = np.full(shape, -1, dtype=np.int64)
    code = {idx: i for i, idx in enumerate(itertools.product(*(range(s) for s in shape)))}

    def faces(idx):
        for ax, step in itertools.product(range(len(shape)), (-1, 1)):
            j = list(idx)
            j[ax] += step
            if not 0 <= j[ax] < shape[ax]:
                if not periodic[ax]:
                    continue
                j[ax] %= shape[ax]
            yield tuple(j)

    occ = counts >= occ_min
    seen = set()
    for idx in itertools.product(*(range(s) for s in shape)):
        if not occ[idx] or idx in seen:
            continue
        comp, queue = [idx], deque([idx])
        seen.add(idx)
        while queue:
            for n in faces(queue.popleft()):
                if occ[n] and n not in seen:
                    seen.add(n)
                    comp.append(n)
                    queue.append(n)
        root = min(code[c] for c in comp)
        for c in comp:
            out[c] = root
    for idx in itertools.product(*(range(s) for s in shape)):
        if counts[idx] > 0 and not occ[idx]:
            best = max(((counts[n], -code[n], n) for n in faces(idx) if occ[n]), default=None)
            if best is not None:
                out[idx] = out[best[2]]
    return out.reshape(-1)


def test_grid_clusters_hand_cases() -> None:
    # diagonal bins are two clusters
    c = np.array([[4, 0], [0, 4]])
    assert grid_clusters(c, (2, 2), 2).tolist() == [0, -1, -1, 3]
    # a bridge bin below occ_min cuts a cluster and attaches to its most populous neighbour (ties: lower code)
    assert grid_clusters(np.array([3, 3, 1, 3, 3]), (5,), 2).tolist() == [0, 0, 0, 3, 3]
    assert grid_clusters(np.array([3, 3, 1, 4, 3]), (5,), 2).tolist() == [0, 0, 3, 3, 3]
    # an isolated sparse bin is -1, empty bins too; a sparse bin next to a sparse bin is not attached
    assert grid_clusters(np.array([0, 1, 0, 0, 5, 1, 1]), (7,), 2).tolist() == [-1, -1, -1, -1, 4, 4, -1]
    # a periodic axis wraps; labels are the component minima
    cnt = np.array([3, 0, 0, 0, 0, 3])
    assert grid_clusters(cnt, (6,), 2, periodic=(True,)).tolist() == [0, -1, -1, -1, -1, 0]
    assert grid_clusters(cnt, (6,), 2, periodic=(False,)).tolist() == [0, -1, -1, -1, -1, 5]
    # in 2-D only the periodic axis wraps: (0, 0) and (2, 0) touch across axis 0 of length 3
    g = np.array([[2, 0], [0, 0], [2, 0]])
    assert grid_clusters(g, (3, 2), 2, periodic=(False, True)).tolist() == [0, -1, -1, -1, 4, -1]
    assert grid_clusters(g, (3, 2), 2, periodic=(True, False)).tolist() == [0, -1, -1, -1, 0, -1]
    # a cluster's label is its smallest code
    c = np.zeros((4, 4), int)
    c[1, 2] = c[2, 2] = c[2, 3] = 5
    got = grid_clusters(c, (4, 4), 5).reshape(4, 4)
    assert set(got[got >= 0].tolist()) == {1 * 4 + 2}
    # nothing occupied: all -1, including non-empty sparse bins
    assert (grid_clusters(np.array([[1, 0], [1, 1]]), (2, 2), 2) == -1).all()
    assert (grid_clusters(np.zeros(16, int), (4, 4), 1) == -1).all()
    # input forms: flat or shaped, any integer dtype
    flat = np.array([5, 0, 3, 3], dtype=np.int32)
    assert grid_clusters(flat, (2, 2), 3).tolist() == grid_clusters(flat.reshape(2, 2), (2, 2), 3).tolist()
    with pytest.raises(ValueError, match="occ_min"):
        grid_clusters(flat, (2, 2), 0)
    with pytest.raises(ValueError, match="bins"):
        grid_clusters(flat, (3, 2), 1)
    with pytest.raises(ValueError, match="periodic"):
        grid_clusters(flat, (2, 2), 1, periodic=(True,))


@pytest.mark.parametrize(
    "shape,periodic",
    [
        ((12, 9), (False, False)),
        ((12, 9), (True, False)),
        ((7, 7), (True, True)),
        ((5, 5, 5, 5), (False, False, False, False)),
        ((4, 6, 3, 5), (True, False, True, False)),
        ((1, 6), (True, True)),
        ((2, 5), (True, True)),
    ],
)
def test_grid_clusters_match_a_breadth_first_reference(shape, periodic) -> None:
    rs = np.random.default_rng(int(np.prod(shape)) + len(shape))
    for occupancy in (0.15, 0.4, 0.7):
        counts = np.where(rs.random(shape) < occupancy, rs.integers(1, 7, shape), 0)
        for occ_min in (1, 2, 4):
            got = grid_clusters(counts, shape, occ_min, periodic)
            assert got.dtype == np.int64
            assert np.array_equal(got, ref_clusters(counts, occ_min, periodic)), (shape, occupancy, occ_min)


def test_grid_clusters_on_the_kin_grid_is_fast_enough_and_terminates() -> None:
    rs = np.random.default_rng(1)
    counts = np.zeros(16**4, np.int32)
    centres = rs.integers(2, 14, (30, 4))
    for c in centres:
        for _ in range(60):
            p = np.clip(c + rs.integers(-1, 2, 4), 0, 15)
            counts[np.ravel_multi_index(tuple(p), (16,) * 4)] += 1
    out = grid_clusters(counts, (16,) * 4, 2)
    assert out.shape == (16**4,) and (out >= 0).any()
    roots = np.unique(out[out >= 0])
    assert all(out[r] == r for r in roots)  # a root labels itself


# --- dominance ---------------------------------------------------------------------------------------------


def ref_dominance(shape, k, min_count, periodic, seq):
    """Python-int reference: yields (owner, purity, border) per update; ``seq`` = [(ids, counts (C, K+1))]."""
    nx, ny = shape
    cells = nx * ny
    prev_ids, ema = None, None
    for ids, counts in seq:
        ids = list(ids)
        new = [[0] * (len(ids) + 1) for _ in range(cells)]
        for c in range(cells):
            for j in range(len(ids) + 1):
                x = int(counts[c][j]) << 8
                tracked = ema is not None and (j == len(ids) or ids[j] in prev_ids)
                if tracked:
                    jo = len(prev_ids) if j == len(ids) else prev_ids.index(ids[j])
                    old = ema[c][jo]
                    new[c][j] = old + x - (old >> k)
                else:
                    new[c][j] = x << k
        prev_ids, ema = ids, new
        owner, purity = [], []
        for c in range(cells):
            row = new[c]
            total = sum(row)
            arg = max(range(len(row)), key=lambda j: (row[j], -j))
            if arg < len(ids) and total >= (min_count << 8) << k and total > 0:
                owner.append(ids[arg])
                purity.append(row[arg] * 65536 // total)
            else:
                owner.append(-1)
                purity.append(0)
        border = []
        for c in range(cells):
            x, y = c % nx, c // nx
            row = []
            for dx, dy in ((1, 0), (0, 1)):
                xx, yy = x + dx, y + dy
                if (xx >= nx and not periodic[0]) or (yy >= ny and not periodic[1]):
                    row.append(False)
                    continue
                n = (yy % ny) * nx + (xx % nx)
                row.append(owner[c] != owner[n] and (owner[c] >= 0 or owner[n] >= 0))
            border.append(row)
        yield owner, purity, border


def test_dominance_owner_other_and_ties() -> None:
    d = Dominance((2, 1), k=0, min_count=1)
    # cell 0: ids 7 and 9 have 5 each, other 1: a tie goes to the lower column (7); cell 1: other largest
    m = d.update(np.array([7, 9]), np.array([[5, 5, 1], [1, 0, 9]]))
    assert m.owner.tolist() == [7, -1]
    assert m.purity_q16.tolist() == [5 * 65536 // 11, 0]
    # a tie between an id and "other" goes to the lower column, which is the id's
    d = Dominance((2, 1), k=0, min_count=1)
    m = d.update(np.array([7]), np.array([[4, 4], [4, 4]]))
    assert m.owner.tolist() == [7, 7]
    # "other" never owns, even alone
    d = Dominance((2, 1), k=0, min_count=1)
    assert d.update(np.array([], dtype=np.int64), np.array([[9], [3]])).owner.tolist() == [-1, -1]


def test_dominance_min_count_and_the_smoothed_total() -> None:
    d = Dominance((3, 1), k=2, min_count=3)
    # a new column starts at x << k, so the totals are 2, 3 and 10 agents: the edge (3) is inclusive
    m = d.update(np.array([4]), np.array([[2, 0], [3, 0], [9, 1]]))
    assert m.owner.tolist() == [-1, 4, 4]
    # the average decays by E += x - (E >> k): a lone cell with 3 agents is released once E < 1 << 8 << k
    d = Dominance((1, 1), k=2, min_count=1)
    assert d.update(np.array([4]), np.array([[3, 0]])).owner.tolist() == [4]
    e, owned = 3 << 10, []
    for _ in range(6):
        e -= e >> 2
        owned.append(e >= 1 << 10)
        assert d.update(np.array([4]), np.array([[0, 0]])).owner.tolist() == [4 if owned[-1] else -1]
    assert owned == [True, True, True, False, False, False]
    # min_count 0 never owns an empty cell
    d = Dominance((2, 1), k=1, min_count=0)
    assert d.update(np.array([4]), np.array([[0, 0], [1, 0]])).owner.tolist() == [-1, 4]


def test_dominance_entering_and_leaving_ids() -> None:
    d = Dominance((1, 1), k=2, min_count=1)
    d.update(np.array([1]), np.array([[8, 0]]))
    s = d.state()
    assert s.ids.tolist() == [1] and s.ema.tolist() == [[(8 << 8) << 2, 0]]
    # id 2 enters (starts at x << k), id 1 continues (E + x - (E >> k)), other continues
    d.update(np.array([1, 2]), np.array([[4, 5, 1]]))
    e1 = (8 << 10) + (4 << 8) - ((8 << 10) >> 2)
    assert d.state().ema.tolist() == [[e1, (5 << 8) << 2, (1 << 8)]]
    # id 1 leaves: dropped; coming back later it starts afresh
    d.update(np.array([2]), np.array([[0, 3]]))
    assert d.state().ids.tolist() == [2]
    d.update(np.array([1, 2]), np.array([[6, 0, 0]]))
    assert d.state().ema[0, 0] == (6 << 8) << 2


def test_dominance_borders() -> None:
    # 3 x 2 grid (code y * 3 + x): row 0 is [1, 1, 2], row 1 is [2, unowned, 2]
    owner_by_cell = [1, 1, 2, 2, -1, 2]
    counts = np.array([[9, 0, 0] if o == 1 else [0, 9, 0] if o == 2 else [0, 0, 9] for o in owner_by_cell])
    east_np, north_np = [0, 1, 0, 1, 1, 0], [1, 1, 0, 0, 0, 0]  # no east past x = 2, no north past y = 1
    east_t, north_t = (
        [0, 1, 1, 1, 1, 0],
        [1, 1, 0, 1, 1, 0],
    )  # wrapping: 2 vs 1 east of cell 2, row 1 vs row 0
    for periodic, east, north in (
        ((False, False), east_np, north_np),
        ((True, True), east_t, north_t),
        ((True, False), east_t, north_np),
        ((False, True), east_np, north_t),
    ):
        d = Dominance((3, 2), k=0, min_count=1, periodic=periodic)
        m = d.update(np.array([1, 2]), counts)
        assert m.owner.tolist() == owner_by_cell
        assert m.border[:, 0].astype(int).tolist() == east, periodic
        assert m.border[:, 1].astype(int).tolist() == north, periodic
        ref = next(ref_dominance((3, 2), 0, 1, periodic, [([1, 2], counts)]))
        assert m.border.tolist() == ref[2], periodic
    # an axis of length 1 or 2 never makes a border against itself
    d = Dominance((1, 2), k=0, min_count=1, periodic=(True, True))
    m = d.update(np.array([1, 2]), np.array([[9, 0, 0], [0, 9, 0]]))
    assert m.border.tolist() == [[False, True], [False, True]]


def test_dominance_empty_ids_and_non_square() -> None:
    d = Dominance((4, 3), k=2, min_count=1)
    m = d.update(np.array([], dtype=np.int64), np.full((12, 1), 5))
    assert m.owner.tolist() == [-1] * 12 and m.purity_q16.tolist() == [0] * 12
    assert m.border.shape == (12, 2) and not m.border.any()
    # a non-square grid, code y * nx + x: one owned cell at x = 3, y = 1 (code 7)
    d = Dominance((4, 3), k=0, min_count=1, periodic=(False, False))
    counts = np.zeros((12, 2), int)
    counts[7] = (9, 0)
    m = d.update(np.array([5]), counts)
    assert np.flatnonzero(m.owner == 5).tolist() == [7]
    assert np.flatnonzero(m.border[:, 0]).tolist() == [
        6
    ]  # cell 6 has an east neighbour owned; cell 7 is the last column
    assert np.flatnonzero(m.border[:, 1]).tolist() == [
        3,
        7,
    ]  # cell 3 (below it, y = 0) and cell 7 itself (y = 2 above)
    for bad in (np.zeros((12, 3)), np.zeros((11, 2))):
        with pytest.raises(ValueError, match="counts"):
            d.update(np.array([5]), bad)
    with pytest.raises(ValueError, match="unique"):
        Dominance((1, 1), 0, 1).update(np.array([3, 3]), np.zeros((1, 3)))


@pytest.mark.parametrize(
    "shape,periodic", [((4, 3), (True, True)), ((5, 5), (False, True)), ((3, 6), (True, False))]
)
def test_dominance_matches_a_reference_on_random_counts(shape, periodic) -> None:
    rs = np.random.default_rng(shape[0] * 10 + shape[1])
    cells = shape[0] * shape[1]
    k, min_count = 2, 2
    ids_pool = [3, 8, 12, 20, 21]
    seq = []
    for _ in range(12):
        ids = [int(i) for i in rs.permutation(ids_pool)[: int(rs.integers(0, 5))]]
        counts = rs.integers(0, 6, (cells, len(ids) + 1)) * (rs.random((cells, len(ids) + 1)) < 0.6)
        seq.append((ids, counts))
    d = Dominance(shape, k, min_count, periodic)
    for (ids, counts), (owner, purity, border) in zip(
        seq, ref_dominance(shape, k, min_count, periodic, seq), strict=True
    ):
        m = d.update(np.array(ids, dtype=np.int64), counts)
        assert m.owner.tolist() == owner and m.purity_q16.tolist() == purity
        assert m.border.tolist() == border
        assert m.owner.dtype == np.int64 and m.purity_q16.dtype == np.int64 and m.border.dtype == np.bool_
    # the state round-trips: a copy continues identically
    s = d.state()
    twin = Dominance(shape, k, min_count, periodic, DominanceState(s.ids.copy(), s.ema.copy()))
    ids, counts = seq[3]
    a, b = d.update(np.array(ids, dtype=np.int64), counts), twin.update(np.array(ids, dtype=np.int64), counts)
    assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))
    with pytest.raises(ValueError, match="shape"):
        Dominance(
            shape, k, min_count, periodic, DominanceState(np.array([1]), np.zeros((cells, 5), np.int64))
        )


def test_duplicate_credit_pairs_count_once() -> None:
    # the same child credited twice (two parents in one system, one pair repeated): A' holds it once
    first = mem({1: rng_(0, 10)})
    cur = mem({1: [0, 1, 2, *rng_(10, 17)]})
    children = np.array([*range(10, 17)] * 2)
    parents = np.array([*range(0, 7), *range(1, 8)])
    tr = SystemTracker(THETA)
    tr.observe(0, first)
    a, ev, d = tr.observe(1, cur, credit=(children, parents))
    assert ids_of(a) == {1: 1} and ev == []
    assert d.matched_jaccard.tolist() == [pytest.approx(10 / 17)]  # |A'| = 17, not 24
