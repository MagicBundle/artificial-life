"""Focus, follow and the hand-off at death (DESIGN §33.2.7): ``FOCS`` against the state, the observer's slots
and labels, the four successor rules on a built scene, the death point from the cached row, a death in the
step of its child's birth, and a real run in which the followed organism starves."""

from __future__ import annotations

import io
import struct
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import mlx.core as mx
import numpy as np
import pytest

from alife import play
from alife.config import Config
from alife.game.chronicle import KIND, Chronicler
from alife.play import PlayPacer, _Followed
from alife.state import DeathCause, State, init_state, to_numpy
from play_decode import Rec, _focs
from play_helpers import FakeClock, ScriptControl, decoded, run_paced, snapshots, uids, world

RUN = play.PlayOptions(rate=float("inf"), paused=False, fresh=True)


# --- a built scene ------------------------------------------------------------------------------------------


def make_pacer(cfg: Config, state: State, kin: list[int] | None = None) -> PlayPacer:
    """A pacer with just enough of a bound world for the focus methods; ``kin`` is the kin slot per row (None:
    no observers)."""
    p = PlayPacer(io.BytesIO(), ScriptControl([]), play.PlayOptions(), clock=FakeClock(), note=lambda m: None)
    p.cfg, p.chron, p._t = cfg, Chronicler(0), 50
    if kin is not None:
        n = state.uid.shape[0]
        zeros = mx.zeros((n,), dtype=mx.int32)
        p.obs = SimpleNamespace(  # type: ignore[assignment]
            so=SimpleNamespace(
                kin=mx.array(np.array(kin, np.int32)),
                house=mx.full((n,), -1, dtype=mx.int64),
                colony=zeros - 1,
            ),
            o=SimpleNamespace(
                label_arch=mx.zeros((n, 2), dtype=mx.int32),
                label_intent=zeros,
                held=zeros,
                match_intent=zeros,
            ),
            social=SimpleNamespace(slots=16),
        )
    return p


def scene(cfg: Config, rows: dict[int, tuple[int, tuple[float, ...]]]) -> State:
    """The ``init_state`` of ``cfg`` with only ``rows`` alive: ``{row: (uid, position)}``."""
    s = init_state(cfg)
    n = s.uid.shape[0]
    uid = np.full(n, -1, np.int64)
    pos = np.array(to_numpy(s.pos))
    alive = np.zeros(n, bool)
    for row, (u, xy) in rows.items():
        uid[row], alive[row] = u, True
        pos = np.concatenate([pos[:row], np.array([xy], pos.dtype), pos[row + 1 :]])
    return s._replace(uid=mx.array(uid), alive=mx.array(alive), pos=mx.array(pos))


def out_of(cfg: Config, *, deaths: list[tuple[int, int]] = (), births: list[tuple[int, int]] = ()) -> Any:
    """A ``StepOut`` stand-in: ``deaths`` as ``(uid, cause)`` and ``births`` as ``(uid, parent_uid)``."""
    m = 8

    def lanes(items, cols):
        v = np.zeros(m, bool)
        a = [np.full(m, -1, np.int64) for _ in range(cols)]
        for i, item in enumerate(items):
            v[i] = True
            for col, val in zip(a, item, strict=True):
                col[i] = val
        return mx.array(v), [mx.array(x) for x in a]

    dv, (du, dc) = lanes(deaths, 2)
    bv, (bu, bp) = lanes(births, 2)
    return SimpleNamespace(
        deaths=SimpleNamespace(valid=dv, uid=du, cause=dc),
        births=SimpleNamespace(valid=bv, uid=bu, parent_uid=bp),
    )


def records(p: PlayPacer) -> list[Rec]:
    """The ``FOCS`` records written to a bare pacer's stream (no header)."""
    data, out, off = p.out.getvalue(), [], 0
    while off < len(data):
        tag, length = data[off : off + 4], struct.unpack_from("<I", data, off + 4)[0]
        payload = data[off + 8 : off + 8 + length]
        step, seq, version, flags = struct.unpack_from("<QIHH", payload, 0)
        r = Rec(tag.decode(), off, length, step, seq, version, flags)
        _focs(r, payload, 0)
        out.append(r)
        off += 8 + length
    return out


@pytest.fixture(scope="module")
def cfg2d() -> Config:
    return world("2d", observe=False)


def hand_off(p: PlayPacer, state: State, out: Any, dead: _Followed) -> None:
    p.focus, p.follow = dead.uid, True
    p._hand_off(state, out, 50, dead)


def test_rule_1_the_eldest_living_child(cfg2d: Config) -> None:
    state = scene(
        cfg2d,
        {0: (300, (10.0, 10.0)), 1: (205, (60.0, 60.0)), 2: (210, (11.0, 11.0)), 3: (150, (12.0, 12.0))},
    )
    p = make_pacer(cfg2d, state)
    dead = _Followed(uid=100, row=5, pos=(10.0, 10.0), kin=-1, children=frozenset({210, 205, 999}))
    hand_off(p, state, out_of(cfg2d, deaths=[(100, DeathCause.AGED.value)]), dead)
    assert p.focus == 205  # the smallest living child uid, though 210 is nearer; 999 is dead
    (r,) = records(p)
    assert r.f["uid"] == 205 and r.f["row"] == 1 and r.f["flags"] & 0b110 == 0b110  # alive is bit 0
    assert [e["text"] for e in p._events] == ["Organism 100 died (aged); follow passed to a child."]
    assert [(e["kind"], e["a"], e["b"]) for e in p._events] == [(KIND["focus_handoff"], 100, 205)]


def test_rule_2_the_nearest_of_the_same_kin_slot(cfg2d: Config) -> None:
    state = scene(cfg2d, {0: (301, (10.5, 10.0)), 1: (302, (20.0, 20.0)), 2: (303, (14.0, 10.0))})
    kin = [9, 4, 4, *[-1] * (state.uid.shape[0] - 3)]  # row 0 is the nearest, but of another kin group
    p = make_pacer(cfg2d, state, kin)
    dead = _Followed(uid=100, row=5, pos=(10.0, 10.0), kin=4, children=frozenset({777}))  # the child is gone
    hand_off(p, state, out_of(cfg2d, deaths=[(100, DeathCause.KILLED.value)]), dead)
    assert p.focus == 303  # row 0 is nearest overall, but it is of another kin group
    assert [e["text"] for e in p._events] == [
        "Organism 100 died (killed); follow passed to the nearest member of its kin group."
    ]


def test_rule_3_the_nearest_living_organism(cfg2d: Config) -> None:
    state = scene(cfg2d, {0: (301, (10.5, 10.0)), 1: (302, (20.0, 20.0))})
    kin = [7, 7] + [-1] * (state.uid.shape[0] - 2)
    p = make_pacer(cfg2d, state, kin)
    dead = _Followed(uid=100, row=5, pos=(10.0, 10.0), kin=4)  # nobody of kin 4 is alive
    hand_off(p, state, out_of(cfg2d, deaths=[(100, DeathCause.STARVED.value)]), dead)
    assert p.focus == 301
    assert [e["text"] for e in p._events] == [
        "Organism 100 died (starved); follow passed to the nearest organism."
    ]
    # without observers rule 2 does not exist: the same scene gives the same answer by rule 3
    q = make_pacer(cfg2d, state)
    hand_off(q, state, out_of(cfg2d, deaths=[(100, DeathCause.STARVED.value)]), dead)
    assert q.focus == 301 and [e["kind"] for e in q._events] == [KIND["focus_handoff"]]


def test_the_distance_is_the_torus_distance(cfg2d: Config) -> None:
    size = cfg2d.world.size
    state = scene(cfg2d, {0: (301, (size - 1.0, 5.0)), 1: (302, (size / 2, 5.0))})
    p = make_pacer(cfg2d, state)
    hand_off(p, state, out_of(cfg2d), _Followed(uid=100, row=5, pos=(1.0, 5.0), kin=-1))
    assert p.focus == 301  # 2 units across the seam, not size - 2


def test_rule_4_an_empty_world_ends_the_follow(cfg2d: Config) -> None:
    state = scene(cfg2d, {})
    p = make_pacer(cfg2d, state)
    hand_off(
        p, state, out_of(cfg2d, deaths=[(100, DeathCause.STARVED.value)]), _Followed(100, 5, (1.0, 1.0), -1)
    )
    assert p.focus is None and p._followed is None
    (r,) = records(p)
    assert r.f["uid"] == 100 and r.f["flags"] == 0  # one FOCS with the alive flag clear
    assert [e["text"] for e in p._events] == ["Organism 100 died (starved)."]


def test_a_death_the_lanes_lost_has_no_cause(cfg2d: Config) -> None:
    """If the death lanes overflowed, the followed uid is not among them: the cause is unknown and the
    hand-off goes on."""
    state = scene(cfg2d, {0: (301, (12.0, 10.0))})
    p = make_pacer(cfg2d, state)
    hand_off(
        p, state, out_of(cfg2d, deaths=[(55, DeathCause.AGED.value)]), _Followed(100, 5, (10.0, 10.0), -1)
    )
    assert [e["text"] for e in p._events] == ["Organism 100 died; follow passed to the nearest organism."]
    assert p.focus == 301


def test_the_death_point_is_the_cached_previous_position(cfg2d: Config) -> None:
    state = scene(cfg2d, {0: (301, (12.0, 10.0))})
    p = make_pacer(cfg2d, state)
    hand_off(p, state, out_of(cfg2d, deaths=[(100, 0)]), _Followed(100, 5, (33.25, 44.5), -1))
    assert [(e["x"], e["y"], e["z"]) for e in p._events] == [(33.25, 44.5, None)]


def test_a_child_born_in_the_step_its_parent_died_is_the_successor(
    cfg2d: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``follow`` + a death in the same step as the birth of the followed organism's child: the child is read
    from this step's birth lanes, which the cache has not seen, and takes rule 1."""
    state = scene(cfg2d, {0: (400, (50.0, 50.0)), 1: (301, (10.5, 10.0))})
    p = make_pacer(cfg2d, state)
    p.focus, p.follow, p._followed = 100, True, _Followed(100, 5, (10.0, 10.0), -1)
    out = out_of(cfg2d, deaths=[(100, DeathCause.AGED.value)], births=[(400, 100)])
    p._follow_step(state, out, 50)  # row 5 no longer holds uid 100
    assert p.focus == 400
    assert [e["text"] for e in p._events] == ["Organism 100 died (aged); follow passed to a child."]


def test_a_plain_focus_does_not_hand_off(cfg2d: Config) -> None:
    state = scene(cfg2d, {0: (301, (12.0, 10.0))})
    p = make_pacer(cfg2d, state)
    p.focus, p.follow, p._followed = 100, False, _Followed(100, 5, (10.0, 10.0), -1)
    p._follow_step(state, out_of(cfg2d), 50)
    assert p.focus is None
    (r,) = records(p)
    assert r.f["uid"] == 100 and r.f["flags"] == 0 and p._events == []


# --- real runs ----------------------------------------------------------------------------------------------


def test_focs_is_the_row_of_the_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    snapshots(monkeypatch, 10)
    cfg = world("2d")
    uid = uids(cfg)[3]
    _, res, _, out = run_paced(cfg, tmp_path / "w", 25, [["follow 0"], [f"follow {uid}"]], options=RUN)
    s = res.state
    row = int(np.argmax(to_numpy(s.uid) == uid))
    last = [r for r in decoded(out).records if r.tag == "FOCS"][-1]
    f = last.f
    assert last.step == 25 and f["uid"] == uid and f["row"] == row and f["flags"] == 0b101
    np.testing.assert_allclose([f["x"], f["y"]], to_numpy(s.pos)[row], rtol=0, atol=0)
    np.testing.assert_allclose([f["vx"], f["vy"]], to_numpy(s.vel)[row], rtol=0, atol=0)
    assert f["z"] == 0.0 and f["vz"] == 0.0  # a 2D world
    assert f["energy"] == int(to_numpy(s.energy)[row]) and f["generation"] == int(to_numpy(s.generation)[row])
    assert f["parent_uid"] == int(to_numpy(s.parent_uid)[row])
    assert f["age"] == 25 - int(to_numpy(s.birth_step)[row])
    assert f["n_bonds"] == int((to_numpy(s.partner)[row] >= 0).sum())


def test_focs_slots_and_labels_agree_with_idnt_and_intn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """At a step that also has an ``IDNT`` the focused row carries the same slot bytes, and at a step with an
    ``INTN`` the same labels: both read the observer, one for every row, one for a row."""
    snapshots(monkeypatch, 10)
    cfg = world("3d")
    pool = uids(cfg)
    checked = {"idnt": 0, "intn": 0}
    for uid in pool[:6]:
        _, _, _, out = run_paced(cfg, tmp_path / f"w{uid}", 60, [[f"follow {uid}"]], options=RUN)
        recs = decoded(out).records
        focs = {r.step: r.f for r in recs if r.tag == "FOCS"}
        for r in recs:
            if r.tag == "IDNT" and r.step in focs and uid in r.a["uid"]:
                i = int(np.flatnonzero(r.a["uid"] == uid)[0])
                f = focs[r.step]
                assert (f["house"], f["kin"], f["colony"]) == (
                    r.a["house"][i],
                    r.a["kin"][i],
                    r.a["colony"][i],
                )
                checked["idnt"] += 1
            if r.tag == "INTN" and r.step in focs and uid in r.a["uid"]:
                i = int(np.flatnonzero(r.a["uid"] == uid)[0])
                f = focs[r.step]
                assert (f["arch_primary"], f["arch_secondary"], f["intent"], f["label_flags"]) == (
                    r.a["arch_primary"][i], r.a["arch_secondary"][i], r.a["intent"][i], r.a["flags"][i]
                )  # fmt: skip
                checked["intn"] += 1
    assert checked["idnt"] >= 20 and checked["intn"] >= 8  # not vacuous


def test_a_newborn_is_focused_with_the_slots_the_observer_gives_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A row born after the last snapshot has the slots the observer gives it (its parent's): ``FOCS`` and the
    next ``IDNT`` row, where it is flagged as born since the last snapshot, agree."""
    snapshots(monkeypatch, 10)
    cfg = world("3d", "energy.e_init=60000")
    rd, *_ = run_paced(cfg, tmp_path / "probe", 40, [], options=RUN)  # the world is deterministic
    ev = rd.read_events("births")
    kids = [
        (int(u), int(s)) for u, s, p in zip(ev["uid"], ev["step"], ev["parent_uid"], strict=True) if p >= 0
    ]
    uid, born = next(k for k in kids if 12 <= k[1] <= 25)  # a birth between two snapshots
    _, _, _, out = run_paced(cfg, tmp_path / "w", 40, [*[[]] * (born + 1), [f"follow {uid}"]], options=RUN)
    recs = decoded(out).records
    focs = next(r for r in recs if r.tag == "FOCS")
    assert focs.f["uid"] == uid and focs.f["flags"] & 1 and focs.step > born
    assert focs.f["age"] == focs.step - born  # a newborn's age counts from its birth step
    idnt = next(r for r in recs if r.tag == "IDNT" and r.step % 10 == 0 and r.step >= focs.step)
    i = int(np.flatnonzero(idnt.a["uid"] == uid)[0])
    focs_later = next(r for r in recs if r.tag == "FOCS" and r.step == idnt.step)
    assert (focs_later.f["house"], focs_later.f["kin"], focs_later.f["colony"]) == (
        idnt.a["house"][i], idnt.a["kin"][i], idnt.a["colony"][i]
    )  # fmt: skip


def test_the_followed_organism_starves_and_the_follow_goes_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshots(monkeypatch, 10)
    cfg = world("2d", "energy.e_init=3000", "population.n_init=60")
    _, _, pacer, out = run_paced(cfg, tmp_path / "w", 80, [["follow 5"]], options=RUN)
    assert pacer.failure is None
    recs = decoded(out).records
    texts = [
        (r.step, e["fam"], e["text"])
        for r in recs
        if r.tag == "EVNT"
        for e in r.f["json"]
        if e["fam"] == "focus"
    ]
    assert texts[0][2].startswith("Organism 5 died (starved); follow passed to"), "one line, not two"
    step = texts[0][0]
    focs = [r for r in recs if r.tag == "FOCS"]
    assert [r.f["uid"] for r in focs if r.step == step - 1] == [5]
    after = next(r for r in focs if r.step == step)
    assert after.f["uid"] != 5 and after.f["flags"] == 0b111  # alive, hand-off this frame, followed
    later = [r.f["uid"] for r in focs if step < r.step <= step + 5]
    assert later and set(later) == {after.f["uid"]}  # and the follow stays on the new organism
