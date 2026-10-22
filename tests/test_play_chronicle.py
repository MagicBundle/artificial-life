"""The chronicle (DESIGN §33.2, ``game/chronicle.py``): events from records, tracker ticks, the director and
the play loop, in a closed table of sentences; ids that do not depend on delivery order; a backlog rebuilt
from the logs that equals what the live chronicler said."""

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
    kin: list[tuple[int, str]] = (),
    events: list[tuple[str, str, int, int, int, int]] = (),
    houses: list[tuple[int, str]] = (),
    founded: list[tuple[int, int | None]] = (),
    extinct: list[int] = (),
    three: bool = False,
) -> dict[str, Any]:
    """A social record with just the columns the chronicle reads. ``events`` rows: family, kind, id, size,
    other, overlap."""
    kin_cols = ("id", "name", "x", "y") + (("z_mean",) if three else ())
    kin_rows = [(i, nm, 10.0 + i, 20.0 + i) + ((5.5,) if three else ()) for i, nm in kin]
    return {
        "step": step,
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


SCRIPT = [
    # step 250: two kin groups appear, a house is founded from a listed parent
    record(
        250,
        kin=[(1, "Aldric"), (2, "Brennar")],
        events=[("kin_group", "born", 1, 40, -1, 0), ("kin_group", "born", 2, 33, -1, 0)],
        houses=[(100, "Houra"), (200, "Telan")],
        founded=[(200, 100)],
    ),
    # step 500: a split, a merge, a dissolution into a survivor, a colony event (not narrated), a house goes
    record(
        500,
        kin=[(1, "Aldric"), (3, "Cestria")],
        events=[
            ("kin_group", "split", 3, 12, 1, 5),
            ("kin_group", "merged", 2, 30, 1, 25),
            ("colony", "born", 9, 4, -1, 0),
            ("kin_group", "dissolved", 7, 9, 1, 6),
        ],
        houses=[(100, "Houra")],
        extinct=[200, 999],
    ),
    # step 750: a dissolution with no successor, a split and a merge with no related group, a founded house
    # with no parent (listed) and one outside the top 16 (not narrated)
    record(
        750,
        kin=[(1, "Aldric")],
        events=[
            ("kin_group", "dissolved", 3, 10, -1, 0),
            ("kin_group", "split", 8, 7, -1, 0),
            ("kin_group", "merged", 9, 5, -1, 0),
        ],
        houses=[(100, "Houra"), (300, "Nolen")],
        founded=[(300, None), (301, None)],
    ),
]
STORMS = [
    {"step": 750, "events": [storm_event("born", 750, 4, hemi=1)]},
    {"step": 800, "events": [storm_event("named", 800, 4, name="Merrow", hemi=1)]},
    {
        "step": 825,
        "events": [
            storm_event("split", 825, 5, other=4, other_name="Merrow"),
            storm_event("merged", 825, 3, other=4, other_name="Merrow"),
        ],
    },
    {"step": 900, "events": [storm_event("dissolved", 900, 4, name="Merrow", age=125)]},
    {"step": 925, "events": [storm_event("dissolved", 925, 5, age=100)]},
]
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
                    out += chron.from_storms(item["events"])
                else:
                    out += chron.from_director([item])
    return out


# --- the table of sentences ---------------------------------------------------------------------------------


def _masked(text: str) -> str:
    return re.sub("|".join(NAMES), "Name", text)


def test_every_template_is_reachable_and_renders_within_the_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    used: set[str] = set()
    real = chronicle.render

    def spy(key: str, **slots: Any) -> str:
        used.add(key)
        return real(key, **slots)

    monkeypatch.setattr(chronicle, "render", spy)
    events = live()
    chron = Chronicler(SALT)
    events += chron.focus_death(10, 5_000_000_007, "starved", (1.0, 2.0, 3.0))
    events += chron.focus_death(10, 5_000_000_008, None, (None, None, None))
    for rule in (1, 2, 3):
        events += chron.focus_handoff(11, 7, 8, rule, (1.0, 2.0, None))
    events += chron.checkpoint(6000) + chron.renumbered(6000) + chron.extinct(6001)
    assert used == set(TEMPLATES), sorted(set(TEMPLATES) ^ used)
    assert {e["kind"] for e in events} == set(KIND.values()), "every kind of the registry is produced"
    for e in events:
        streamrec.check_event(e)
        assert len(e["text"]) <= 120 and "because" not in e["text"].lower()


def test_the_world_reports_lexicon_finds_nothing_to_object_to() -> None:
    """The chronicle's sentences carry no causal, modal, evaluative, taxonomic or label word (display names
    masked), as the report's do; the director's line, which quotes the lever it pulled, is checked for the
    words the lever list does not cover."""
    chron = Chronicler(SALT)
    events = (
        live() + chron.focus_death(10, 5, "starved", (0, 0, 0)) + chron.focus_death(10, 6, None, (0, 0, 0))
    )
    for rule in (1, 2, 3):
        events += chron.focus_handoff(11, 7, 8, rule, (0, 0, 0))
    events += chron.checkpoint(6000) + chron.renumbered(6000) + chron.extinct(6001)
    for e in events:
        lists = LEX_NEVER if e["fam"] != "director" else tuple(x for x in LEX_NEVER if x != "V1")
        text = _masked(e["text"])
        if e["fam"] == "director":  # "population below target" is the director's own reason, not ours
            text = text.split("[")[0]
        assert lexicon.scan(text, lists) == [], (e["kind"], e["text"])


def test_a_name_that_would_overflow_is_cut_to_the_limit() -> None:
    text = render("kin_born", name="N" * 300, size=12)
    assert len(text) == 120 and text.endswith("...")
    assert render("kin_born", name="Aldric", size=12) == "Aldric formed with 12 members."


# --- events: shape, ids, importance -------------------------------------------------------------------------


def test_texts_and_fields_of_the_record_events() -> None:
    chron = Chronicler(SALT)
    e1 = chron.from_record(SCRIPT[0])
    assert [e["text"] for e in e1] == [
        "Aldric formed with 40 members.",
        "Brennar formed with 33 members.",
        "House Telan was founded from House Houra.",
    ]
    assert [(e["fam"], e["kind"], e["a"], e["b"], e["imp"]) for e in e1] == [
        ("kin", KIND["kin_born"], 1, 0, 1),
        ("kin", KIND["kin_born"], 2, 0, 1),
        ("house", KIND["house_founded"], 200, 100, 1),
    ]
    assert (e1[0]["x"], e1[0]["y"], e1[0]["z"]) == (11.0, 21.0, None)
    e2 = chron.from_record(SCRIPT[1])
    assert [e["text"] for e in e2] == [
        "Cestria split from Aldric with 12 members.",
        "Brennar merged into Aldric (30 members).",
        "Kin group 7 dissolved at 9 members; 6 of them are in Aldric.",
        "House Telan has no living members.",
    ]
    assert [e["imp"] for e in e2] == [2, 1, 2, 1], "a split and a dissolution raise a toast"
    assert e2[3]["a"] == 200, "house 999 was never listed, so its end is not narrated"
    e3 = chron.from_record(SCRIPT[2])
    assert [e["text"] for e in e3] == [
        "Cestria dissolved at 10 members.",
        "Kin group 8 split off with 7 members.",
        "Kin group 9 merged (5 members).",
        "House Nolen was founded.",
    ], "a house founded outside the top 16 is silent"
    assert (e3[0]["a"], e3[0]["b"]) == (3, 0) and (e3[3]["a"], e3[3]["b"]) == (300, 0)


def test_3d_records_give_a_height_and_colony_events_are_not_narrated() -> None:
    chron = Chronicler(SALT)
    rec = record(
        250,
        kin=[(1, "Aldric")],
        events=[("kin_group", "born", 1, 8, -1, 0), ("colony", "born", 4, 4, -1, 0)],
        three=True,
    )
    (e,) = chron.from_record(rec)
    assert (e["x"], e["y"], e["z"]) == (11.0, 21.0, 5.5)


def test_storm_events_and_their_importance() -> None:
    chron = Chronicler(SALT)
    got = {e["kind"]: e for s in STORMS for e in chron.from_storms(s["events"])}
    assert got[KIND["storm_born"]]["text"] == "Disturbance 4 formed in the southern hemisphere."
    assert got[KIND["storm_named"]]["text"] == "Storm Merrow reached named strength."
    assert got[KIND["storm_split"]]["text"] == "Disturbance 5 split from Storm Merrow."
    assert got[KIND["storm_merged"]]["text"] == "Disturbance 3 merged into Storm Merrow."
    dissolved = [e for s in STORMS for e in Chronicler(SALT).from_storms(s["events"]) if e["kind"] == 11]
    assert [(e["text"], e["imp"]) for e in dissolved] == [
        ("Storm Merrow dissolved after 125 steps.", 2),
        ("Disturbance 5 dissolved after 100 steps.", 1),
    ]
    assert got[KIND["storm_named"]]["imp"] == 2 and got[KIND["storm_born"]]["imp"] == 1
    assert (got[KIND["storm_born"]]["x"], got[KIND["storm_born"]]["y"]) == (12.5, 40.25)
    assert (got[KIND["storm_split"]]["a"], got[KIND["storm_split"]]["b"]) == (5, 4)
    with pytest.raises(ValueError, match="unknown tracker event"):
        chron.from_storms([storm_event("exploded", 900, 1)])


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
    (h,) = chron.focus_handoff(77, 123, 456, 1, (1.0, 2.0, 3.0))
    assert (h["text"], h["a"], h["b"]) == ("Follow passed to a child.", 123, 456)
    assert chron.checkpoint(6000)[0]["imp"] == 0 and chron.extinct(5)[0]["imp"] == 2
    assert chron.renumbered(5)[0]["text"] == "Storm numbering restarted at step 5."
    with pytest.raises(KeyError):
        chron.focus_handoff(1, 2, 3, 4, (0, 0, 0))


def test_ids_are_step_times_a_thousand_plus_a_family_base_and_do_not_depend_on_the_order() -> None:
    a = live("rsd")
    for order in ("".join(p) for p in itertools.permutations("rsd")):
        assert sorted(live(order), key=lambda e: e["id"]) == sorted(a, key=lambda e: e["id"]), order
    assert FAMILY_BASE == {"kin": 0, "house": 100, "storm": 200, "director": 300, "focus": 400, "system": 500}
    assert {e["step"] for e in a if e["fam"] == "storm"} & {e["step"] for e in a if e["fam"] == "house"}
    assert len({e["id"] for e in a}) == len(a)
    for e in a:
        assert e["id"] // 1000 == e["step"] and e["id"] % 1000 >= FAMILY_BASE[e["fam"]]
        assert e["id"] % 1000 < FAMILY_BASE[e["fam"]] + 100
    kin500 = [e["id"] for e in a if e["step"] == 500 and e["fam"] == "kin"]
    assert kin500 == [500000, 500001, 500002], "indices count the family's events of the step"


def test_a_family_keeps_its_first_hundred_events_of_a_step() -> None:
    rows = [("kin_group", "born", 1000 + i, 8, -1, 0) for i in range(120)]
    events = Chronicler(SALT).from_record(record(250, events=rows))
    assert len(events) == 100 and events[-1]["id"] == 250099
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


def test_the_backlog_fits_one_evnt_record() -> None:
    many = [
        record(250 * (i + 1), kin=[(1, "Aldric")], events=[("kin_group", "born", 1, 8, -1, 0)])
        for i in range(200)
    ]
    backlog = rebuild_backlog(many, [], [], house_salt=SALT)
    assert len(backlog) == 64 and backlog[-1]["step"] == 50_000
    streamrec.encode_evnt(step=50_000, seq=0, events=backlog, backlog=True)


# --- real records -------------------------------------------------------------------------------------------


@pytest.mark.parametrize("preset", ["tiny", "tiny3d"])
def test_real_snapshot_records_give_valid_events_and_the_backlog_equals_them(preset: str) -> None:
    """The columns the chronicle reads are the columns the observers write: an observed run of 90 steps with
    a snapshot every 15 (2D and 3D, a high house rate). Colony events are in the records and not in the
    chronicle; the count of the others follows the filter of the spec. A world this small seldom has a kin
    group or a top-16 house change, so the cases are the hand-built records above; this checks that the
    columns the chronicle reads are the ones written."""
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
    colony = sum(f == "colony" for r in records for f in r["events"]["family"])
    assert colony > 0, "the records carry colony events, which the chronicle leaves out"
    want = 0
    previous: set[int] = set()
    for r in records:
        listed = set(r["houses"]["id"])
        want += sum(f == "kin_group" for f in r["events"]["family"])
        want += sum(h in listed for h in r["house_events"]["founded"]["id"])
        want += sum(h in previous for h in r["house_events"]["extinct"]["id"])
        previous = listed
    assert len(events) == want
    assert rebuild_backlog(records, [], [], house_salt=sr.name_salt, limit=1000) == sorted(
        events, key=lambda e: (e["step"], e["id"])
    )
