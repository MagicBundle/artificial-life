"""Storm persistence and the initial storms (G3b spec §2.4, DESIGN §34.1): the chronicle tells a birth once
the track has ``STORM_MIN_DETECTIONS`` detections, a naming at once, an end only for a track it told, and
nothing of the first tick of a layer made by this run. The rule is a fold over the lines of
``observers/storms.jsonl`` (events and detected ids), so a backlog rebuilt from it is the live chronicle."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from alife import storms
from alife.game.chronicle import KIND, Chronicler, rebuild_backlog
from alife.storms import STORM_MIN_DETECTIONS, StormLayer
from test_play_storms import DC_REF, P40, scene

SALT = 0x686F757365
DATA = Path(__file__).resolve().parent / "data" / "play20k_logs"
MIN = STORM_MIN_DETECTIONS


def ev(kind: str, step: int, ident: int, **kw: Any) -> dict[str, Any]:
    out = {
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
    if kw.get("initial"):
        out["initial"] = True
    return out


class Feed:
    """A tracker's lines, fed to one chronicler: ``tick(events, ids)`` at the next step, 25 steps on."""

    def __init__(self) -> None:
        self.chron, self.step, self.told, self.lines = Chronicler(SALT), 0, [], []

    def tick(self, events: list[dict[str, Any]] = (), ids: tuple[int, ...] = ()) -> list[dict[str, Any]]:
        self.step += 25
        events = [{**e, "step": self.step} for e in events]
        self.lines.append({"step": self.step, "storms": [{"id": i} for i in ids], "events": events})
        got = self.chron.from_storms(self.step, events, ids)
        self.told += got
        return got

    def live(self, ident: int, n: int) -> None:
        for _ in range(n):
            self.tick([], (ident,))


def texts(events: list[dict[str, Any]]) -> list[str]:
    return [e["text"] for e in events]


# --- the rule -----------------------------------------------------------------------------------------------


def test_a_birth_is_told_at_the_eighth_detection_and_not_before() -> None:
    f = Feed()
    assert f.tick([ev("born", 0, 4, hemi=1)], (4,)) == []  # detection 1
    f.live(4, MIN - 2)  # detections 2 .. MIN - 1
    assert f.told == []
    (e,) = f.tick([], (4,))  # detection MIN
    assert e["text"] == "Disturbance 4 formed in the southern hemisphere." and e["step"] == f.step
    assert (e["kind"], e["fam"], e["a"], e["imp"]) == (KIND["storm_born"], "storm", 4, 1)
    assert (e["x"], e["y"]) == (12.5, 40.25), "at the point it was born"
    f.live(4, 5)
    assert len(f.told) == 1, "told once"


def test_a_track_that_ends_sooner_is_silent_all_the_way() -> None:
    f = Feed()
    f.tick([ev("born", 0, 7)], (7,))
    f.tick([], (7,))
    f.tick([ev("dissolved", 0, 7, age=50)], ())
    f.live(8, 3)
    assert f.told == [] and f.chron._sheld == {} and 7 not in f.chron._sdets
    # merged: same
    f.tick([ev("born", 0, 8)], (8,))
    f.tick([ev("merged", 0, 8, other=3, other_name="Merrow")], ())
    assert f.told == []


def test_a_split_is_a_birth_with_a_parent_and_waits_too() -> None:
    f = Feed()
    f.tick([ev("split", 0, 6, other=2, other_name="Merrow")], (6,))
    assert f.told == []
    f.live(6, MIN - 1)
    assert texts(f.told) == ["Disturbance 6 split from Storm Merrow."]
    assert (f.told[0]["a"], f.told[0]["b"], f.told[0]["kind"]) == (6, 2, KIND["storm_split"])


def test_a_naming_is_told_at_once_and_replaces_the_birth() -> None:
    f = Feed()
    f.tick([ev("born", 0, 4)], (4,))
    (named,) = f.tick([ev("named", 0, 4, name="Merrow")], (4,))
    assert named["text"] == "Storm Merrow reached named strength." and named["imp"] == 2
    f.live(4, MIN + 2)
    assert texts(f.told) == ["Storm Merrow reached named strength."], "the birth sentence is gone"


def test_a_birth_told_before_the_naming_is_followed_by_it() -> None:
    f = Feed()
    f.tick([ev("born", 0, 4)], (4,))
    f.live(4, MIN - 1)
    f.tick([ev("named", 0, 4, name="Merrow")], (4,))
    assert texts(f.told) == [
        "Disturbance 4 formed in the northern hemisphere.",
        "Storm Merrow reached named strength.",
    ]


def test_an_end_is_told_only_for_a_track_that_was_told() -> None:
    f = Feed()
    f.tick([ev("born", 0, 1)], (1,))  # never reaches MIN
    f.tick([ev("dissolved", 0, 1, age=75)], ())
    f.tick([ev("born", 0, 2)], (2,))  # told as born, then dissolves unnamed
    f.live(2, MIN)
    f.tick([ev("dissolved", 0, 2, age=300)], ())
    f.tick([ev("born", 0, 3)], (3,))  # named: told, then dissolves
    f.tick([ev("named", 0, 3, name="Merrow")], (3,))
    f.tick([ev("dissolved", 0, 3, name="Merrow", age=125)], ())
    assert texts(f.told) == [
        "Disturbance 2 formed in the northern hemisphere.",
        "Disturbance 2 dissolved after 300 steps.",
        "Storm Merrow reached named strength.",
        "Storm Merrow dissolved after 125 steps.",
    ]
    assert [e["imp"] for e in f.told] == [1, 1, 2, 2]


def test_the_first_tick_of_a_layer_is_not_told_but_its_storms_are_known() -> None:
    f = Feed()
    first = [ev("born", 0, i, initial=True) for i in (1, 2)] + [
        ev("named", 0, 2, name="Merrow", initial=True)
    ]
    assert f.tick(first, (1, 2)) == []
    f.live(1, MIN + 3)  # an initial track is not told as born when it reaches the count
    assert f.told == []
    f.tick([ev("dissolved", 0, 1, age=900)], ())  # never told, never named
    assert f.told == []
    (e,) = f.tick([ev("dissolved", 0, 2, name="Merrow", age=900)], ())
    assert e["text"] == "Storm Merrow dissolved after 900 steps." and e["imp"] == 2, (
        "a named storm's end is told"
    )


def test_an_initial_track_that_is_named_later_is_told_named() -> None:
    f = Feed()
    f.tick([ev("born", 0, 1, initial=True)], (1,))
    (e,) = f.tick([ev("named", 0, 1, name="Merrow")], (1,))
    assert e["text"] == "Storm Merrow reached named strength."


def test_a_restarted_tracker_reuses_ids_without_inheriting_their_history() -> None:
    f = Feed()
    f.tick([ev("born", 0, 1)], (1,))
    f.tick([ev("named", 0, 1, name="Merrow")], (1,))
    # the tracker restarts: id 1 is born again (initial), the old one was never ended in the log
    f.tick([ev("born", 0, 1, initial=True)], (1,))
    before = len(f.told)
    f.live(1, MIN + 2)
    assert len(f.told) == before, "the new track 1 is an initial one: not told"
    f.tick([ev("dissolved", 0, 1, age=10)], ())
    assert len(f.told) == before, "and it was not named, whatever its predecessor was"


def test_an_unknown_tracker_event_is_an_error() -> None:
    with pytest.raises(ValueError, match="unknown tracker event"):
        Chronicler(SALT).from_storms(1, [ev("exploded", 1, 1)], ())


def test_the_ids_of_told_events_are_those_of_the_tick_they_are_told_at() -> None:
    f = Feed()
    f.tick([ev("born", 0, 4)], (4,))
    f.live(4, MIN - 1)
    (e,) = f.told
    assert e["step"] == 25 * MIN and e["id"] == e["step"] * 1000 + 200


# --- the backlog is the live chronicle ----------------------------------------------------------------------


def test_the_backlog_rebuilt_from_the_lines_is_what_the_live_chronicler_said() -> None:
    f = Feed()
    f.tick([ev("born", 0, 1, initial=True), ev("born", 0, 2, initial=True)], (1, 2))
    f.tick([ev("born", 0, 3), ev("named", 0, 2, name="Aaa", initial=True)], (1, 2, 3))
    for _ in range(MIN):
        f.tick([], (1, 2, 3))
    f.tick([ev("named", 0, 3, name="Bbb"), ev("dissolved", 0, 1, age=400)], (2, 3))
    f.tick([ev("dissolved", 0, 2, name="Aaa", age=500), ev("dissolved", 0, 3, name="Bbb", age=300)], ())
    rebuilt = rebuild_backlog([], f.lines, [], house_salt=SALT, limit=1000)
    assert rebuilt == sorted(f.told, key=lambda e: (e["step"], e["id"])) and len(rebuilt) == 4
    # and it leaves the chronicler where the live one is: the same next tick tells the same
    chron = Chronicler(SALT)
    rebuild_backlog([], f.lines, [], house_salt=SALT, chronicler=chron)
    assert (chron._sdets, chron._sheld, chron._sknown) == (f.chron._sdets, f.chron._sheld, f.chron._sknown)


def test_a_real_layer_run_gives_the_same_chronicle_live_and_from_its_log() -> None:
    for seed in range(8):
        layer, live_chron = StormLayer(P40, DC_REF, 25), Chronicler(SALT)
        layer.initial_pending = True
        lines, live = [], []
        for d in scene(seed, 80):
            tick = layer.ingest(d)
            ids = [s["id"] for s in tick.line["storms"]] if tick.line else []
            live += live_chron.from_storms(tick.step, tick.events, ids)
            if tick.line is not None:
                lines.append(json.loads(json.dumps(tick.line)))
        assert rebuild_backlog([], lines, [], house_salt=SALT, limit=10_000) == live, seed
        assert all(e["imp"] in (1, 2) and e["step"] > 25 for e in live), "the first tick tells nothing"


# --- the layer: the initial flag ----------------------------------------------------------------------------


def test_the_first_tick_of_a_pending_layer_flags_its_events_and_clears_the_flag() -> None:
    layer = StormLayer(P40, DC_REF, 25)
    layer.initial_pending = True
    ticks = [layer.ingest(d) for d in scene(3, 12)]
    flagged = [bool(e.get("initial")) for t in ticks for e in t.events]
    first_with_events = next(i for i, t in enumerate(ticks) if t.events)
    assert first_with_events == 0, "the scene has storms at its first tick"
    assert all(e.get("initial") is True for e in ticks[0].events)
    assert not any(e.get("initial") for t in ticks[1:] for e in t.events) and any(flagged)
    assert layer.initial_pending is False
    # the logged line carries the flag as well
    assert all(e.get("initial") for e in ticks[0].line["events"])
    # a layer that is not pending never writes the key (existing logs stay byte-equal)
    plain = StormLayer(P40, DC_REF, 25)
    assert not any("initial" in e for t in [plain.ingest(d) for d in scene(3, 12)] for e in t.events)


def test_a_flag_set_and_not_yet_used_survives_a_stop() -> None:
    layer = StormLayer(P40, DC_REF, 25)
    layer.initial_pending = True
    state = json.loads(json.dumps(layer.state()))
    assert state["initial"] is True
    back = StormLayer.restore(P40, DC_REF, state)
    assert back.initial_pending is True
    ticks_a, ticks_b = [layer.ingest(d) for d in scene(5, 6)], [back.ingest(d) for d in scene(5, 6)]
    assert [t.line for t in ticks_a] == [t.line for t in ticks_b]
    # once used it is gone from the saved state, and a plain layer's state has no such key
    assert "initial" not in layer.state() and "initial" not in StormLayer(P40, DC_REF, 25).state()
    assert StormLayer.restore(P40, DC_REF, layer.state()).initial_pending is False


def test_a_layer_opened_without_state_at_a_later_step_is_pending(tmp_path: Path) -> None:
    from alife.config import load_config
    from alife.runio import RunDir

    rd = RunDir.create(tmp_path / "w", load_config(preset="tiny"))
    layer, reason = StormLayer.open(rd, P40, DC_REF, 450_000, 25)
    assert layer.restarted and layer.initial_pending and reason is not None
    layer0, _ = StormLayer.open(rd, P40, DC_REF, 0, 25)
    assert not layer0.restarted and not layer0.initial_pending, "a world at step 0 starts clean"


# --- the Mac-day storms -------------------------------------------------------------------------------------


def test_p0_25_tracker_events_become_12_told_ones() -> None:
    lines = [json.loads(line) for line in (DATA / "storms.jsonl").read_text().splitlines()]
    lines = [x for x in lines if x["step"] <= 470_000]
    raw = [e for x in lines for e in x["events"]]
    assert len(lines) == 800 and len(raw) == 25
    assert sum(1 for e in raw if e.get("initial")) == 5 and {e["step"] for e in raw if e.get("initial")} == {
        450_025
    }
    chron, told = Chronicler(SALT), []
    for x in lines:
        told += chron.from_storms(x["step"], x["events"], [s["id"] for s in x["storms"]])
    kinds = {
        k: sum(e["kind"] == KIND[k] for e in told) for k in ("storm_born", "storm_named", "storm_dissolved")
    }
    assert kinds == {"storm_born": 4, "storm_named": 6, "storm_dissolved": 2} and len(told) == 12
    minutes = 20_000 / 864  # 20,000 steps at 14.4 steps per second
    assert round(len(told) / minutes, 2) == 0.52 and round(len(raw) / minutes, 2) == 1.08
    # the four ephemeral tracks (ids 7, 8, 9, 12: 1, 3, 1, 1 detections) and the initial storms are silent
    assert not {e["a"] for e in told} & {7, 8, 9, 12}
    assert not {e["a"] for e in told if e["kind"] == KIND["storm_born"]} & {1, 2, 3, 4}
    assert all(e["imp"] == 2 for e in told if e["kind"] == KIND["storm_named"])
    # the backlog of the log is the live chronicle
    again = rebuild_backlog([], lines, [], house_salt=SALT, limit=1000)
    assert again == sorted(told, key=lambda e: (e["step"], e["id"]))


def test_the_constants_are_the_specs() -> None:
    assert storms.STORM_MIN_DETECTIONS == 8
