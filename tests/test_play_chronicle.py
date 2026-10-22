"""The chronicle (DESIGN §33.2 and §34.1, ``game/chronicle.py``, ``CHRONICLE_VERSION`` 2): events from
records (kin threads and houses), tracker ticks, the director and the play loop, in a closed table of
sentences; ids that do not depend on delivery order; a backlog rebuilt from the logs that equals what the live
chronicler said."""

from __future__ import annotations

import dataclasses
import itertools
import json
import re
from typing import Any

import mlx.core as mx
import pytest

from alife import streamrec
from alife.config import apply_overrides, load_config
from alife.game import chronicle, lexicon
from alife.game.chronicle import FAMILY_BASE, KIND, TEMPLATES, Chronicler, rebuild_backlog, render
from alife.game.observer import Observer
from alife.game.social import SOCIAL_V1
from alife.step import make_step
from game_helpers import planted_state

SALT = SOCIAL_V1.name_salt
NAMES = ("Aldric", "Brennar", "Cestria", "Merrow", "Houra", "Telan", "Nolen")
LEX_NEVER = (*lexicon.NEVER_LISTS, *lexicon.LABEL_LISTS)


# --- hand-built records ------------------------------------------------------------------------------------


def _table(columns: tuple[str, ...], rows: list[tuple[Any, ...]]) -> dict[str, list[Any]]:
    return {c: [r[j] for r in rows] for j, c in enumerate(columns)}


def record(
    step: int,
    kin: list[tuple[int, str, int, tuple[float, ...]]] = (),
    events: list[tuple[str, str, int, int, int, int]] = (),
    houses: list[tuple[int, str]] = (),
    founded: list[tuple[int, int | None]] = (),
    extinct: list[int] = (),
    three: bool = False,
    origin: int = 0,
) -> dict[str, Any]:
    """A social record with just the columns the chronicle reads. ``kin`` rows: id, name, size, tag;
    ``events`` rows: family, kind, id, size, other, overlap."""
    kin_cols = ("id", "name", "size", "tag", "x", "y") + (("z_mean",) if three else ())
    kin_rows = [
        (i, nm, size, list(tag), 10.0 + i, 20.0 + i) + ((5.5,) if three else ()) for i, nm, size, tag in kin
    ]
    return {
        "step": step,
        "origin": origin,
        "thresholds": {"kin_enter": 87},
        "kin_groups": _table(kin_cols, kin_rows),
        "events": _table(
            ("step", "family", "kind", "id", "size", "other", "overlap"),
            [(step, fam, k, i, size, other, ov) for fam, k, i, size, other, ov in events],
        ),
        "houses": _table(("id", "name", "size", "clade"), [(i, nm, 3, 0) for i, nm in houses]),
        "house_events": {
            "founded": _table(("id", "step", "parent_house"), [(i, step, p) for i, p in founded]),
            "extinct": _table(("id", "step"), [(i, step) for i in extinct]),
            "founded_dropped": 0,
            "extinct_dropped": 0,
        },
    }


def storm_event(kind: str, step: int, ident: int, **kw: Any) -> dict[str, Any]:
    return {
        "kind": kind,
        "step": step,
        "id": ident,
        "name": kw.get("name"),
        "other": kw.get("other", 0),
        "other_name": kw.get("other_name"),
        "x": 12.5,
        "y": 40.25,
        "hemi": kw.get("hemi", 0),
        "age": kw.get("age", 0),
    }


def director_entry(step: int, **kw: Any) -> dict[str, Any]:
    return {
        "step_applied": step,
        "source": "rule",
        "action": "bloom",
        "params": kw.get("params", {"x": 24.0, "y": 12.0}),
        "duration": 500,
        "reason": "population below target",
        "rule": "rescue",
    }


TA, TB, TC, TD, TE = (
    (0.0, 0.0, 0.0, 0.0),
    (0.5, 0.5, 0.0, 0.0),
    (0.0, 0.8, 0.0, 0.0),
    (0.9, 0.0, 0.5, 0.0),
    (0.4, 0.9, 0.9, 0.0),
)
ALD, BRE, EIR = (1, "Aldric", 400, TA), (2, "Brennar", 40, TB), (5, "Eirik", 25, TE)
CES, DOR = (3, "Cestria", 360, TC), (4, "Dorin", 33, TD)
STEPS = [250 * k for k in range(1, 14)]

SCRIPT = [
    # 250: the initial cast (silent: Aldric, Brennar, Eirik) and a house founded from a listed parent
    record(
        250,
        kin=[ALD, BRE, EIR],
        events=[("kin_group", "born", 1, 400, -1, 0)],
        houses=[(100, "Houra"), (200, "Telan")],
        founded=[(200, 100)],
    ),
    # 500: Cestria splits off Aldric (a candidate); a colony event is not narrated
    record(
        500,
        kin=[ALD, BRE, EIR, CES],
        events=[("kin_group", "split", 3, 360, 1, 5), ("colony", "born", 9, 4, -1, 0)],
        houses=[(100, "Houra"), (200, "Telan")],
    ),
    # 750: Brennar is gone and its merged row arrives; Eirik is gone with no row; house 200 goes
    record(
        750,
        kin=[ALD, CES],
        events=[("kin_group", "merged", 2, 40, 1, 25)],
        houses=[(100, "Houra")],
        extinct=[200, 999],
    ),
    # 1000: Cestria has held together for three snapshots (formed); Dorin is born
    record(1000, kin=[ALD, CES, DOR], events=[("kin_group", "born", 4, 33, -1, 0)], houses=[(100, "Houra")]),
    record(
        1250, kin=[ALD, CES, DOR], houses=[(100, "Houra"), (300, "Nolen")], founded=[(300, None), (301, None)]
    ),
    # 1500: Dorin has held together three snapshots (formed); Cestria is gone with a dissolved row; two end
    record(1500, kin=[ALD, DOR], events=[("kin_group", "dissolved", 3, 10, -1, 0)]),
    record(1750, kin=[ALD, DOR]),
    record(2000, kin=[ALD, DOR]),
    # 2250: Cestria has been absent for four snapshots (ended); Dorin is gone, its merged row arrives
    record(2250, kin=[ALD], events=[("kin_group", "merged", 4, 33, 1, 20)]),
    record(2500, kin=[ALD]),
    record(2750, kin=[ALD]),
    record(3000, kin=[ALD]),  # 3000: Dorin ended (merged into Aldric)
    record(3250, kin=[ALD]),
]


def storm_lines() -> list[dict[str, Any]]:
    """Tracker lines at the 25-step cadence from 750: storm 3 is born and named, 4 is born and held, 5 splits
    from 4, 3 merges into 4, 4 is named and dissolves, 5 dissolves, and a short-lived 9 comes and goes."""
    lines: list[dict[str, Any]] = []
    plan: dict[int, list[dict[str, Any]]] = {
        0: [storm_event("born", 750, 3), storm_event("born", 750, 4, hemi=1)],
        1: [storm_event("named", 775, 3, name="Aldara")],
        9: [storm_event("split", 950, 5, other=4)],
        10: [storm_event("born", 975, 9)],
        11: [storm_event("dissolved", 1000, 9, age=25)],
        14: [storm_event("merged", 1100, 3, name="Aldara", other=4)],
        15: [storm_event("named", 1125, 4, name="Merrow", hemi=1)],
        20: [storm_event("dissolved", 1250, 5, other=4, age=300)],
        24: [storm_event("dissolved", 1350, 4, name="Merrow", hemi=1, age=600)],
    }
    alive = {0: {3, 4}, 9: {3, 4, 5}, 10: {3, 4, 5, 9}, 11: {3, 4, 5}, 14: {4, 5}, 20: {4}, 24: set()}
    now: set[int] = set()
    for i in range(26):
        now = alive.get(i, now)
        events = [{**e, "step": 750 + 25 * i} for e in plan.get(i, ())]
        lines.append({"step": 750 + 25 * i, "storms": [{"id": j} for j in sorted(now)], "events": events})
    return lines


STORMS = [line for line in storm_lines() if line["storms"] or line["events"]]
DIRECTOR = [director_entry(500), director_entry(500, params={"patch": 3}), director_entry(750)]


def live(order: str = "rsd") -> list[dict[str, Any]]:
    """Feed the script to one chronicler step by step, in the order of ``order`` (r: records, s: storms,
    d: director) within a step."""
    chron = Chronicler(SALT)
    by_step: dict[int, dict[str, list[Any]]] = {}
    for r in SCRIPT:
        by_step.setdefault(r["step"], {}).setdefault("r", []).append(r)
    for s in STORMS:
        by_step.setdefault(s["step"], {}).setdefault("s", []).append(s)
    for d in DIRECTOR:
        by_step.setdefault(d["step_applied"], {}).setdefault("d", []).append(d)
    out: list[dict[str, Any]] = []
    for step in sorted(by_step):
        for source in order:
            for item in by_step[step].get(source, ()):
                if source == "r":
                    out += chron.from_record(item)
                elif source == "s":
                    out += chron.from_storms(item["step"], item["events"], [x["id"] for x in item["storms"]])
                else:
                    out += chron.from_director([item])
    return out


# --- the table of sentences ---------------------------------------------------------------------------------


def _masked(text: str) -> str:
    return re.sub("|".join(NAMES), "Name", text)


def _player(
    step: int, power: str, action: str, params: dict[str, Any], duration: int | None
) -> dict[str, Any]:
    return {
        "step_applied": step,
        "source": "player",
        "action": action,
        "params": params,
        "duration": duration,
        "command": 1,
        "power": power,
        "scope": "player",
    }


def _revert(
    step: int, action: str, params: dict[str, Any], origin: int, *, command: int | None = None
) -> dict[str, Any]:
    cancel = {"command": command} if command is not None else {}
    return {
        "step_applied": step,
        "source": "revert",
        "action": action,
        "params": params,
        "duration": None,
        "scope": "player",
        "reverts": origin,
        **cancel,
    }


PLAYER_ENTRIES = [
    _player(20, "food", "regen_pulse", {"mult": 1.2999999523162842}, 6000),
    _player(20, "bloom", "fertility_shift", {"patch": 3, "gain": 1.75}, 3000),
    _player(21, "blight", "habitat_loss", {"patch": 4, "gain": 0.0}, 3000),
    _player(22, "strike", "catastrophe", {"center": [10.0, 20.5], "radius": 15.625, "kill_p": 0.45}, None),
    _revert(6020, "regen_pulse", {"mult": 1.0}, 20),
    _revert(6021, "fertility_shift", {"patch": 3, "gain": 1.0}, 20, command=9),
    _revert(6022, "habitat_loss", {"patch": 4, "gain": 1.0}, 21),
    _revert(6023, "habitat_loss", {"patch": 4, "gain": 1.0}, 21, command=10),
    _revert(6024, "regen_pulse", {"mult": 1.0}, 20, command=11),
]


def play_loop_events(chron: Chronicler) -> list[dict[str, Any]]:
    events = chron.from_director(PLAYER_ENTRIES)
    events += chron.focus_death(10, 5_000_000_007, "starved", (1.0, 2.0, 3.0))
    events += chron.focus_death(10, 5_000_000_008, None, (None, None, None))
    for rule in (1, 2, 3):
        events += chron.focus_handoff(11, 7, 8, rule, (1.0, 2.0, None), "killed")
        events += chron.focus_handoff(12, 7, 8, rule, (1.0, 2.0, None))
    return events + chron.checkpoint(6000) + chron.renumbered(6000) + chron.extinct(6001)


def test_every_template_is_reachable_and_renders_within_the_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    used: set[str] = set()
    real = chronicle.render

    def spy(key: str, **slots: Any) -> str:
        used.add(key)
        return real(key, **slots)

    monkeypatch.setattr(chronicle, "render", spy)
    events = live() + play_loop_events(Chronicler(SALT))
    assert used == set(TEMPLATES), sorted(set(TEMPLATES) ^ used)
    produced = {e["kind"] for e in events}
    raw_kin = {KIND[k] for k in ("kin_born", "kin_split", "kin_merged", "kin_dissolved")}
    assert produced == set(KIND.values()) - raw_kin, "every kind but the raw kin"
    for e in events:
        streamrec.check_event(e)
        assert len(e["text"]) <= 120 and "because" not in e["text"].lower()


def test_the_world_reports_lexicon_finds_nothing_to_object_to() -> None:
    """The chronicle's sentences carry no causal, modal, evaluative, taxonomic or label word (display names
    masked), as the report's do; the director's line, which quotes the lever it pulled, is checked for the
    words the lever list does not cover."""
    events = live() + play_loop_events(Chronicler(SALT))
    for e in events:
        lists = LEX_NEVER if e["fam"] != "director" else tuple(x for x in LEX_NEVER if x != "V1")
        text = _masked(e["text"])
        if e["fam"] == "director":  # "population below target" is the director's own reason, not ours
            text = text.split("[")[0]
        assert lexicon.scan(text, lists) == [], (e["kind"], e["text"])


def test_a_name_that_would_overflow_is_cut_to_the_limit() -> None:
    text = render("kin_formed", name="N" * 300, s0=250, size=12)
    assert len(text) == 120 and text.endswith("...")
    assert (
        render("kin_formed", name="Aldric", s0=250, size=12)
        == "Aldric has held together since step 250: 12 members."
    )


# --- events: shape, ids, importance -------------------------------------------------------------------------


def test_texts_and_fields_of_the_record_events() -> None:
    chron = Chronicler(SALT)
    by_step = {r["step"]: chron.from_record(r) for r in SCRIPT}
    assert [e["text"] for e in by_step[250]] == ["House Telan was founded from House Houra."], (
        "the cast is silent"
    )
    assert by_step[500] == [] and by_step[1250][0]["text"] == "House Nolen was founded."
    assert [e["text"] for e in by_step[750]] == ["House Telan has no living members."]
    (formed,) = by_step[1000]
    assert (
        formed["text"] == "Cestria has held together since step 500 after splitting from Aldric: 360 members."
    )
    assert (formed["fam"], formed["kind"], formed["a"], formed["b"], formed["imp"]) == (
        "kin",
        KIND["kin_formed"],
        3,
        1,
        2,
    )
    assert (formed["x"], formed["y"], formed["z"]) == (13.0, 23.0, None) and formed["step"] == 1000
    assert [(e["kind"], e["text"], e["imp"]) for e in by_step[1500] if e["fam"] == "kin"] == [
        (KIND["kin_formed"], "Dorin has held together since step 1000: 33 members.", 1),
        (KIND["kin_ended"], "Brennar merged into Aldric at step 750; it had up to 40 members.", 1),
        (KIND["kin_ended"], "Eirik dissolved, up to 25 members.", 1),
    ], "formed first, then ended, each by thread; an initial thread has no lifetime"
    (a, b) = (by_step[2250][0], by_step[3000][0])
    assert a["text"] == "Cestria dissolved after 1000 steps, up to 360 members." and a["imp"] == 1
    assert (
        b["text"] == "Dorin merged into Aldric at step 2250; it lasted 1250 steps and had up to 33 members."
    )
    assert (b["a"], b["b"], b["fam"]) == (4, 1, "kin")
    assert [e["kind"] for e in by_step[1500]] == [KIND["kin_formed"], KIND["kin_ended"], KIND["kin_ended"]]


def test_3d_records_give_a_height_and_colony_events_are_not_narrated() -> None:
    chron = Chronicler(SALT)
    cast = [(1, "Aldric", 400, TA)]
    chron.from_record(record(250, kin=cast, three=True))
    got = []
    for step in (500, 750, 1000):
        got += chron.from_record(
            record(
                step,
                kin=[*cast, (7, "Newcomer", 60, TC)],
                three=True,
                events=[("colony", "born", 4, 4, -1, 0)],
            )
        )
    (e,) = got
    assert e["text"].startswith("Newcomer has held together") and (e["x"], e["y"], e["z"]) == (
        17.0,
        27.0,
        5.5,
    )


def test_storm_events_and_their_importance() -> None:
    chron = Chronicler(SALT)
    told = {}
    for line in STORMS:
        for e in chron.from_storms(line["step"], line["events"], [x["id"] for x in line["storms"]]):
            told.setdefault(e["kind"], []).append(e)
    assert [e["text"] for e in told[KIND["storm_born"]]] == [
        "Disturbance 4 formed in the southern hemisphere."
    ]
    assert [(e["text"], e["imp"]) for e in told[KIND["storm_named"]]] == [
        ("Storm Aldara reached named strength.", 2),
        ("Storm Merrow reached named strength.", 2),
    ]
    assert [e["text"] for e in told[KIND["storm_split"]]] == ["Disturbance 5 split from Disturbance 4."]
    assert [e["text"] for e in told[KIND["storm_merged"]]] == ["Storm Aldara merged into Disturbance 4."]
    assert [(e["text"], e["imp"]) for e in told[KIND["storm_dissolved"]]] == [
        ("Disturbance 5 dissolved after 300 steps.", 1),
        ("Storm Merrow dissolved after 600 steps.", 2),
    ], "the short-lived storm 9 is silent"
    born = told[KIND["storm_born"]][0]
    assert born["imp"] == 1 and (born["x"], born["y"]) == (12.5, 40.25) and born["a"] == 4
    assert (told[KIND["storm_split"]][0]["a"], told[KIND["storm_split"]][0]["b"]) == (5, 4)
    with pytest.raises(ValueError, match="unknown tracker event"):
        chron.from_storms(900, [storm_event("exploded", 900, 1)], ())


def test_director_events_carry_the_entry_and_its_position() -> None:
    (e,) = Chronicler(SALT).from_director([director_entry(500)])
    assert e["fam"] == "director" and e["step"] == 500 and (e["x"], e["y"]) == (24.0, 12.0)
    assert e["text"].startswith("Director: bloom(x=24.0, y=12.0) for 500 steps [rescue:")
    (e2,) = Chronicler(SALT).from_director([director_entry(500, params={"patch": 3})])
    assert (e2["x"], e2["y"]) == (None, None)
    long = director_entry(9, params={"note": "z" * 400})
    assert len(Chronicler(SALT).from_director([long])[0]["text"]) == 120


def test_play_loop_events() -> None:
    chron = Chronicler(SALT)
    (d,) = chron.focus_death(77, 123, "killed", (1.0, 2.0, 3.0))
    assert (d["text"], d["a"], d["imp"]) == ("Organism 123 died (killed).", 123, 1)
    (h,) = chron.focus_handoff(77, 123, 456, 1, (1.0, 2.0, 3.0), "starved")
    assert (h["text"], h["a"], h["b"], h["kind"]) == (
        "Organism 123 died (starved); follow passed to a child.",
        123,
        456,
        KIND["focus_handoff"],
    )
    (h2,) = chron.focus_handoff(78, 124, 457, 3, (1.0, 2.0, 3.0))
    assert h2["text"] == "Organism 124 died; follow passed to the nearest organism."
    assert chron.checkpoint(6000)[0]["imp"] == 0 and chron.extinct(5)[0]["imp"] == 2
    assert chron.renumbered(5)[0]["text"] == "Storm numbering restarted at step 5."
    with pytest.raises(KeyError):
        chron.focus_handoff(1, 2, 3, 4, (0, 0, 0))


def test_a_handoff_is_one_line_for_the_death_and_the_follow() -> None:
    """A followed organism that dies and whose follow passes on gave two lines for one fact in G3a (32 lines
    for 16 facts on the Mac day); a death with no successor keeps its own line."""
    chron = Chronicler(SALT)
    pair = chron.focus_handoff(5, 1, 2, 2, (0.0, 0.0, None), "aged")
    assert len(pair) == 1 and pair[0]["fam"] == "focus"
    assert len(chron.focus_death(6, 2, "aged", (0.0, 0.0, None))) == 1


def test_ids_are_step_times_a_thousand_plus_a_family_base_and_do_not_depend_on_the_order() -> None:
    a = live("rsd")
    for order in ("".join(p) for p in itertools.permutations("rsd")):
        assert sorted(live(order), key=lambda e: e["id"]) == sorted(a, key=lambda e: e["id"]), order
    assert FAMILY_BASE == {"kin": 0, "house": 100, "storm": 200, "director": 300, "focus": 400, "system": 500}
    assert len({e["id"] for e in a}) == len(a)
    for e in a:
        assert e["id"] // 1000 == e["step"] and e["id"] % 1000 >= FAMILY_BASE[e["fam"]]
        assert e["id"] % 1000 < FAMILY_BASE[e["fam"]] + 100
    kin1500 = [e["id"] for e in a if e["step"] == 1500 and e["fam"] == "kin"]
    assert kin1500 == [1500000, 1500001, 1500002], "indices count the family's events of the step"
    assert {e["step"] for e in a if e["fam"] == "storm"} & {
        e["step"] for e in a if e["fam"] == "director"
    } == set()


def test_a_family_keeps_its_first_hundred_events_of_a_step() -> None:
    """120 groups established in the same snapshot: 120 events, the first hundred ids kept."""
    groups = [(1000 + i, f"G{i}", 30, (i * 0.3, 0.0, 0.0, 0.0)) for i in range(120)]
    chron = Chronicler(SALT)
    chron.from_record(record(250, kin=[(1, "Anchor", 400, (-1.0, 0.0, 0.0, 0.0))]))
    events = []
    for step in (500, 750, 1000):
        events = chron.from_record(record(step, kin=[(1, "Anchor", 400, (-1.0, 0.0, 0.0, 0.0)), *groups]))
    assert len(events) == 100 and events[-1]["id"] == 1000099
    assert len({e["id"] for e in events}) == 100


# --- the backlog --------------------------------------------------------------------------------------------


def test_the_backlog_equals_what_the_live_chronicler_said() -> None:
    expected = sorted(live(), key=lambda e: (e["step"], e["id"]))
    rebuilt = rebuild_backlog(SCRIPT, STORMS, DIRECTOR, house_salt=SALT, limit=10_000)
    assert rebuilt == expected and len(expected) >= 15
    assert json.dumps(rebuilt) == json.dumps(expected)
    assert rebuild_backlog(SCRIPT, STORMS, DIRECTOR, house_salt=SALT, limit=5) == expected[-5:]
    assert rebuild_backlog(SCRIPT, STORMS, DIRECTOR, house_salt=SALT, limit=0) == []
    assert rebuild_backlog([], [], [], house_salt=SALT) == []


def test_a_live_chronicler_that_made_the_backlog_continues_it() -> None:
    """``chronicler=`` makes the backlog with the live chronicler: it then holds the thread table and the ids
    it handed out, so the next record of the world is told and numbered as a play that had never stopped."""
    for cut in range(1, len(SCRIPT)):
        chron = Chronicler(SALT)
        rebuild_backlog(SCRIPT[:cut], [], [], house_salt=SALT, chronicler=chron)
        whole = Chronicler(SALT)
        for rec in SCRIPT[:cut]:
            whole.from_record(rec)
        rest = [e for rec in SCRIPT[cut:] for e in chron.from_record(rec)]
        assert rest == [e for rec in SCRIPT[cut:] for e in whole.from_record(rec)], f"cut at {cut}"
    assert [e for rec in SCRIPT[:4] for e in Chronicler(SALT).from_record(rec)] != [], (
        "the first half said something"
    )


def test_a_saved_thread_table_gives_the_same_play_as_a_refold() -> None:
    """Play, quit at a checkpoint (the table is saved), resume: the saved table and a refold of the log are
    the same table, and either way the events told next are those of an uninterrupted play. Cut at every
    record, including those with a candidate that establishes after the resume (Cestria, Dorin)."""
    whole = Chronicler(SALT)
    told = [e for rec in SCRIPT for e in whole.from_record(rec)]
    for cut in range(1, len(SCRIPT)):
        first = Chronicler(SALT)
        head = [e for rec in SCRIPT[:cut] for e in first.from_record(rec)]
        saved = json.loads(json.dumps(first.threads.state()))
        refolded = Chronicler(SALT)
        rebuild_backlog(SCRIPT[:cut], [], [], house_salt=SALT, chronicler=refolded)
        assert refolded.threads.state() == saved, f"refold equals saved at {cut}"
        restored = Chronicler(SALT)
        rebuild_backlog(SCRIPT[:cut], [], [], house_salt=SALT, chronicler=restored)
        restored.restore_threads(saved)
        tail = [e for rec in SCRIPT[cut:] for e in restored.from_record(rec)]
        assert head + tail == told, f"cut at {cut}"
    with pytest.raises(ValueError):
        Chronicler(SALT).restore_threads({"format": "nope"})


def test_the_cast_shows_one_name_and_one_thread_across_a_re_formation() -> None:
    chron = Chronicler(SALT)
    groups = [(1, "Aldric", 400, TA), (2, "Brennar", 90, TB)]
    chron.from_record(record(250, kin=groups))
    assert chron.cast_names() == {1: (1, "Aldric"), 2: (2, "Brennar")}
    chron.from_record(record(500, kin=groups[:1]))
    chron.from_record(
        record(750, kin=[groups[0], (9, "Renamed", 80, TB)])
    )  # Brennar back under a new id and name
    assert chron.cast_names() == {1: (1, "Aldric"), 9: (2, "Brennar")}, "the thread keeps its id and its name"


def test_the_backlog_fits_one_evnt_record() -> None:
    many = []
    for i in range(200):
        gid = i // 4 + 10
        many.append(
            record(250 * (i + 1), kin=[(gid, f"G{gid}", 60, (0.3 * (gid % 3), 0.3 * (gid // 3), 0.0, 0.0))])
        )
    backlog = rebuild_backlog(many, [], [], house_salt=SALT)
    assert len(backlog) == 64 and backlog[-1]["step"] <= 50_000
    streamrec.encode_evnt(step=50_000, seq=0, events=backlog, backlog=True)


# --- real records -------------------------------------------------------------------------------------------


@pytest.mark.parametrize("preset", ["tiny", "tiny3d"])
def test_real_snapshot_records_give_valid_events_and_the_backlog_equals_them(preset: str) -> None:
    """The columns the chronicle reads are the columns the observers write: an observed run of 90 steps with
    a snapshot every 15 (2D and 3D, a high house rate). Colony events are in the records and not in the
    chronicle; the house events follow the filter of the spec. A world this small seldom has a kin group
    that holds together three snapshots, so the kin cases are the hand-built records above; this checks that
    the columns the chronicle reads (``id name size tag x y``, ``events``, ``thresholds``) are the ones
    written, and that the fold runs over them."""
    cfg = apply_overrides(
        load_config(preset=preset, overrides=["brain.weight_dtype=float32"]),
        ["game.observe=true", "game.house_rate=0.25"],
    )
    sr = dataclasses.replace(SOCIAL_V1, snapshot_every=15)
    s = planted_state(cfg)
    fn = make_step(cfg, observe=True)
    ob = Observer.fresh(s, cfg, social=sr)
    records: list[dict[str, Any]] = []
    for t in range(1, 91):
        s, out = fn(s)
        ob.update(s, out.acts, t)
        mx.eval(s, out.base(), *ob.arrays())
        records += ob.take_records()
    assert [r["step"] for r in records] == [15, 30, 45, 60, 75, 90]
    chron = Chronicler(sr.name_salt)
    events = [e for r in records for e in chron.from_record(r)]
    for e in events:
        streamrec.check_event(e)
    assert {e["fam"] for e in events} <= {"kin", "house"}
    assert {e["kind"] for e in events if e["fam"] == "kin"} <= {KIND["kin_formed"], KIND["kin_ended"]}
    colony = sum(f == "colony" for r in records for f in r["events"]["family"])
    assert colony > 0, "the records carry colony events, which the chronicle leaves out"
    want = 0
    previous: set[int] = set()
    for r in records:
        listed = set(r["houses"]["id"])
        want += sum(h in listed for h in r["house_events"]["founded"]["id"])
        want += sum(h in previous for h in r["house_events"]["extinct"]["id"])
        previous = listed
    assert sum(e["fam"] == "house" for e in events) == want
    assert chron.threads.k == len(records) - 1 and chron.threads.step == 90
    assert rebuild_backlog(records, [], [], house_salt=sr.name_salt, limit=1000) == sorted(
        events, key=lambda e: (e["step"], e["id"])
    )
