"""The alert engine (``game/alerts.py``, G3b spec §4, DESIGN §34.3).

Synthetic sequences pin each class, the dedupe and re-arm rules, the rate limit, the pause policy and the
provenance rule; the committed trim of the Mac-day play of planet p0 (``tests/data/play20k_logs``) is the
regression pin of the per-class counts of spec §4.4. A changed threshold changes a pinned number on purpose
and needs an ``ALERT_VERSION`` bump (``constants_digest``)."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from alife import streamrec
from alife.game import alerts as al
from alife.game import lexicon
from alife.game.alerts import AlertEngine, Taken
from alife.game.chronicle import Chronicler

DATA = Path(__file__).resolve().parent / "data" / "play20k_logs"
STEP = 250
ALL_ARMED = lambda cls: True  # noqa: E731
NONE_ARMED = lambda cls: False  # noqa: E731


# --- synthetic inputs ---------------------------------------------------------------------------------------


def row(step: int, alive: int, *, steps: int = 250, starved: int = 5000, storm: int = 1) -> dict[str, Any]:
    return {
        "step": step,
        "steps": steps,
        "n_alive": float(alive),
        "deaths_starved": starved,
        "deaths_storm": storm,
        "observers": {"alive": alive},
    }


def engine(armed: Any = NONE_ARMED, **kw: Any) -> AlertEngine:
    return AlertEngine(armed=armed, chapter=6000, **kw)


def feed(eng: AlertEngine, alives: list[int], start: int = 0, **kw: Any) -> list[Taken]:
    """One row per ``STEP`` from ``start``; returns what each row produced."""
    out = []
    for i, a in enumerate(alives):
        t = start + STEP * (i + 1)
        eng.on_row(row(t, a, **kw), t)
        out.append(eng.take())
    return out


def raised(takens: list[Taken], cls: str | None = None) -> list[dict[str, Any]]:
    return [a for t in takens for a in t.alerts if a["state"] == "raised" and cls in (None, a["class"])]


def states(takens: list[Taken], cls: str) -> list[str]:
    return [a["state"] for t in takens for a in t.alerts if a["class"] == cls]


def steady(eng: AlertEngine, n: int = 10, alive: int = 10_000, start: int = 0) -> int:
    """Settle the engine on a steady population; the first row is the (silent) first evaluation."""
    feed(eng, [alive] * n, start)
    return start + STEP * n


def check_all(takens: list[Taken]) -> None:
    """Every record the engine emitted is a valid ``ALRT`` alert of the registry."""
    for t in takens:
        for a in t.alerts:
            streamrec.check_alert(a, al.CLASSES)
        streamrec.encode_alrt(step=0, seq=0, alerts=t.alerts[: streamrec.MAX_ALERTS], registry=al.CLASSES)


# --- the registry -------------------------------------------------------------------------------------------


def test_the_registry_is_a_valid_header_block_and_lists_the_spec_classes() -> None:
    block = streamrec.alerts_block(al.registry())
    assert set(block["classes"]) == set(al.CLASSES)
    assert {
        "chapter", "pop_fall", "pop_low", "pop_slide", "pop_surge", "storm_named", "storm_strong",
        "storm_ended", "storm_toll", "lineage_formed", "lineage_ended", "lineage_collapse",
        "house_fading", "effect_ending", "director_acted", "storm_over_lineage", "engine", "summary",
    } == set(al.CLASSES)  # fmt: skip
    assert block["cap"] == {
        "n": al.CAP_N,
        "window": al.CAP_WINDOW,
        "exempt": ["imp3", "pause", "chapter", "engine"],
    }
    surfaces = {n: c.surface for n, c in al.CLASSES.items()}
    assert {n for n, s in surfaces.items() if s == "log"} == {
        "storm_strong", "storm_ended", "storm_toll", "lineage_formed", "lineage_ended", "house_fading",
        "storm_over_lineage",
    }  # fmt: skip


def test_the_default_pauses_are_pop_fall_pop_low_and_engine() -> None:
    d = al.default_autopause()
    assert {n for n, on in d.items() if on} == {"pop_fall", "pop_low", "engine"}
    assert "chapter" not in d and "summary" not in d  # the chapter has its own switch
    assert al.CLASSES["chapter"].pause is True
    assert al.CLASSES["pop_fall"].pause_min == 3 and al.CLASSES["engine"].pause_min == 3


def test_the_constants_are_pinned() -> None:
    """A changed threshold is a deliberate change: bump ``ALERT_VERSION`` and re-pin."""
    assert al.ALERT_VERSION == 1
    assert al.constants_digest() == PINNED_CONSTANTS


def test_every_text_and_title_template_renders_inside_the_caps_and_the_lexicon() -> None:
    names = (
        "step peak alive steps starved row median phrase since span pct ref n low name zeta age label size "
        "end cells radius storm sec achieved requested error"
    )
    slots = dict.fromkeys(names.split(), "12,345")
    for key, template in al.TEXTS.items():
        text = template.format(**slots)
        assert len(text) <= 160 or key in ("pop_fall_starved", "pop_fall_entry"), key
        never = [*lexicon.NEVER_LISTS, *lexicon.LABEL_LISTS]
        assert lexicon.scan(text, tuple(never)) == [], (key, text)
    for key, template in al.TITLES.items():
        title = template.format(**slots)
        assert len(title) <= 48, key
        assert lexicon.scan(title, tuple([*lexicon.NEVER_LISTS, *lexicon.LABEL_LISTS])) == [], key


# --- the population -----------------------------------------------------------------------------------------


def test_a_fall_of_eight_percent_raises_pop_fall_at_importance_two_and_a_deep_one_at_three() -> None:
    eng = engine()
    t0 = steady(eng)
    (a,) = raised(feed(eng, [9_100], t0))  # 9% below the maximum of the previous rows
    assert a["class"] == "pop_fall" and a["imp"] == 2 and a["surface"] == "alert" and a["version"] == 1
    assert a["text"] == "The population fell from 10,000 to 9,100 in 250 steps."
    assert a["expires"] == t0 + STEP + al.EXPIRE_STEPS and a["pause"] is False and a["ref"] is None
    eng2 = engine()
    t1 = steady(eng2)
    (b,) = raised(feed(eng2, [8_000], t1))  # 20%
    assert b["imp"] == 3 and b["expires"] is None


def test_a_one_row_fall_of_ten_percent_is_importance_three() -> None:
    eng = engine()
    t0 = steady(eng)
    (a,) = raised(feed(eng, [8_900], t0))  # 11% in one row
    assert a["imp"] == 3
    eng2 = engine()
    t1 = steady(eng2)
    feed(eng2, [9_600, 9_300], t1)  # slow slide: 7%, then 7% from the maximum
    (b,) = raised(feed(eng2, [9_150], t1 + 2 * STEP))  # 8.5% from the maximum, 1.6% in the row
    assert b["imp"] == 2


def test_pop_fall_fires_once_updates_a_deeper_fall_and_clears_below_four_percent() -> None:
    eng = engine()
    t0 = steady(eng)
    takens = feed(eng, [9_000, 9_000, 8_300, 8_300, 9_900, 9_900], t0)
    assert states(takens, "pop_fall") == ["raised", "updated", "cleared"]
    ids = {a["id"] for t in takens for a in t.alerts if a["class"] == "pop_fall"}
    assert len(ids) == 1 and next(iter(ids)) == (t0 + STEP) * 1000 + 600
    versions = [a["version"] for t in takens for a in t.alerts if a["class"] == "pop_fall"]
    assert versions == [1, 2, 3]
    check_all(takens)
    # the key is armed again: a new fall raises a new alert
    again = feed(eng, [8_000], t0 + 6 * STEP)
    (a,) = raised(again)
    assert a["id"] != next(iter(ids)) and a["version"] == 1


def test_pop_fall_that_outlasts_six_thousand_steps_is_cleared_and_rearmed() -> None:
    eng = engine()
    t0 = steady(eng, alive=100_000)
    falling = [round(100_000 * 0.985 ** (i + 1)) for i in range(30)]  # 1.5% a row: 8.7% over 6 rows, for ever
    takens = feed(eng, falling, t0)
    raises = [(a["step"], a["id"]) for t in takens for a in t.alerts if a["state"] == "raised"]
    assert len(raises) >= 2 and len({i for _, i in raises}) == len(raises)
    assert raises[1][0] - raises[0][0] == al.REARM_STEPS  # cleared and re-armed after 6,000 steps
    assert "cleared" in states(takens, "pop_fall")


def test_a_fall_after_a_regrowth_entry_is_importance_one_never_pauses_and_refers_to_the_entry() -> None:
    entry = {
        "step_applied": 2_000, "source": "rule", "action": "regen_pulse", "params": {"mult": 0.8},
        "duration": 10_000, "reason": "x",
    }  # fmt: skip
    far = engine(ALL_ARMED)
    steady(far, n=10)  # rows up to 2,500
    far.on_row(row(2_750, 8_000), 2_750, entries=[entry])  # 750 after the entry: outside the 500 steps
    (a,) = raised([far.take()])
    assert a["imp"] == 3 and a["pause"] is True and a["ref"] is None
    near = engine(ALL_ARMED)
    steady(near, n=9)  # rows up to 2,250
    near.on_row(
        row(2_500, 9_100), 2_500, entries=[entry]
    )  # 500 after the entry: inside; 9% under the maximum
    taken = near.take()
    (b,) = raised([taken])
    assert b["imp"] == 1 and b["pause"] is False and taken.pause is False
    assert b["ref"] == {"kind": "power", "id": 2_000}
    assert b["text"].endswith("A regrowth pulse of x0.80 for 10,000 steps has been active since step 2,000.")
    assert b["expires"] == 2_500 + al.EXPIRE_STEPS  # an importance-1 alert expires
    # later rows, the entry no longer recent and the fall deeper but not yet as deep as a hard one (15%): the
    # alert is stated, not changed
    near.on_row(row(2_750, 8_800), 2_750, entries=[entry])
    assert near.take().alerts == []
    # once it is as deep as a hard fall it is one whatever came before it: importance 3, it pauses, and it
    # still states the entry (third review, stage 8)
    near.on_row(row(3_000, 8_400), 3_000, entries=[entry])
    up = near.take()
    (u,) = [a for a in up.alerts if a["class"] == "pop_fall"]
    assert u["state"] == "updated" and u["imp"] == 3 and u["pause"] is True and up.pause is True
    assert (
        u["ref"] == {"kind": "power", "id": 2_000} and "x0.80 for 10,000 steps has been active" in u["text"]
    )
    check_all([taken, up])
    # a fall that is hard at its first row is not demoted: the entry is stated, the alert is importance 3
    hard = engine(ALL_ARMED)
    steady(hard, n=9)
    hard.on_row(row(2_500, 8_000), 2_500, entries=[entry])
    (h,) = raised([hard.take()])
    assert h["imp"] == 3 and h["pause"] is True and h["ref"] == {"kind": "power", "id": 2_000}


def test_a_regrowth_pulse_that_raises_food_is_no_provenance_for_a_fall() -> None:
    """A Food x1.3 is not the reason to expect a fall: the fall after it is an ordinary one (third review)."""

    def fall_after(mult: float) -> dict[str, Any]:
        eng = engine(ALL_ARMED)
        steady(eng, n=9)
        e = {"step_applied": 2_250, "source": "player", "action": "regen_pulse", "params": {"mult": mult},
             "duration": 6_000, "reason": "x"}  # fmt: skip
        eng.on_row(row(2_500, 9_100), 2_500, entries=[e])
        (a,) = raised([eng.take()])
        return a

    up, down = fall_after(1.3), fall_after(0.8)
    assert up["imp"] == 2 and up["ref"] is None and "Food" not in up["text"]
    assert down["imp"] == 1 and down["ref"] == {"kind": "power", "id": 2_250}
    assert fall_after(1.0)["imp"] == 2


def test_a_habitat_entry_and_a_player_entry_count_but_a_revert_and_other_actions_do_not() -> None:
    def fall_with(entry: dict[str, Any]) -> dict[str, Any]:
        eng = engine()
        steady(eng)
        eng.on_row(row(2_500, 9_100), 2_500, entries=[entry])  # 9% under the maximum: not a hard fall
        (a,) = raised([eng.take()])
        return a

    base = {"step_applied": 2_400, "params": {"patch": 0}, "duration": 6_000, "reason": "x"}
    assert fall_with({**base, "source": "player", "action": "habitat_loss"})["imp"] == 1
    player = fall_with({**base, "source": "player", "action": "regen_pulse", "params": {"mult": 0.7}})
    assert player["text"].endswith("Food x0.70 for 6,000 steps has been active since step 2,400.")
    assert fall_with({**base, "source": "revert", "action": "regen_pulse"})["imp"] == 2
    assert fall_with({**base, "source": "rule", "action": "immigration_pulse"})["imp"] == 2
    assert fall_with({**base, "source": "rule", "action": "regen_pulse", "step_applied": 1_000})["imp"] == 2


def test_the_starvation_clause_needs_a_ratio_of_one_point_three_five_over_the_median() -> None:
    eng = engine()
    t0 = steady(eng)
    eng.on_row(row(t0 + STEP, 8_000, starved=5_000 * 1.35), t0 + STEP)
    (a,) = raised([eng.take()])
    assert " 6,750 starved in the latest 250 steps, against a median of 5,000." in a["text"]
    eng2 = engine()
    t1 = steady(eng2)
    eng2.on_row(row(t1 + STEP, 8_000, starved=6_700), t1 + STEP)
    assert "starved" not in raised([eng2.take()])[0]["text"]


def test_pop_low_is_unconditional_pauses_and_clears_above_sixty_percent_of_the_maximum() -> None:
    entry = {"step_applied": 2_400, "source": "rule", "action": "regen_pulse", "params": {"mult": 0.5}}
    eng = engine(ALL_ARMED)
    t0 = steady(eng)
    eng.on_row(row(t0 + STEP, 4_900), t0 + STEP, entries=[entry])
    taken = eng.take()
    (low,) = raised([taken], "pop_low")
    assert low["imp"] == 3 and low["pause"] is True and taken.pause is True and low["ref"] is None
    assert low["text"] == "The population is 4,900, 49% of the 10,000 it held within 6,000 steps."
    eng.on_row(row(t0 + 2 * STEP, 5_900), t0 + 2 * STEP)
    assert states([eng.take()], "pop_low") == []  # 59%: still low, and one alert stays up
    eng.on_row(row(t0 + 3 * STEP, 6_100), t0 + 3 * STEP)
    assert states([eng.take()], "pop_low") == ["cleared"]


def test_the_first_evaluation_after_open_arms_the_level_classes_silently() -> None:
    eng = engine(ALL_ARMED)
    eng.prime_rows([row(250 * i, 10_000) for i in range(1, 25)])
    eng.on_row(row(6_250, 4_000), 6_250)  # the world opens in a collapse: nothing is raised
    assert eng.take().alerts == [] and eng.take().log == []
    eng.on_row(row(6_500, 4_100), 6_500)
    assert eng.take().alerts == []
    # recovering re-arms the classes (silently: they were never raised), then a new fall raises
    feed(eng, [9_000] * 8, 6_500)
    (a,) = raised(feed(eng, [7_000], 6_500 + 8 * STEP))
    assert a["class"] == "pop_fall"


def test_pop_slide_compares_the_chapter_end_with_the_three_previous_ends() -> None:
    eng = engine()
    for end, alive in ((6_000, 10_000), (12_000, 9_000), (18_000, 8_600)):
        eng.on_row(row(end, alive), end)
    eng.take()
    eng.on_row(row(24_000, 8_000), 24_000)  # exactly 0.80 of 10,000: raised (<=)
    (a,) = raised([eng.take()], "pop_slide")
    assert a["imp"] == 2 and a["surface"] == "alert" and a["pause"] is False
    assert (
        a["text"]
        == "The population is 8,000, 80% of the 10,000 it had at the highest of the 3 chapter ends before."
    )
    eng.on_row(row(24_250, 7_000), 24_250)  # off the chapter end: not evaluated
    assert eng.take().alerts == []
    eng.on_row(row(30_000, 9_500), 30_000)  # 9,500 against the max of (9,000, 8,600, 8,000): >= 0.90
    assert states([eng.take()], "pop_slide") == ["cleared"]


def test_pop_slide_does_not_fire_on_a_hair_above_the_threshold() -> None:
    """The p0 chapter ends are 44,910 / 44,002 / 35,934: 35,934 / 44,910 = 0.8002, a hair above 0.80."""
    eng = engine()
    for end, alive in ((456_000, 44_910), (462_000, 44_002)):
        eng.on_row(row(end, alive), end)
    eng.take()
    eng.on_row(row(468_000, 35_934), 468_000)
    assert raised([eng.take()], "pop_slide") == []
    # the next end is compared with the three before it, 44,910 among them: 35,900 / 44,910 is under 0.80
    eng.on_row(row(474_000, 35_900), 474_000)
    assert len(raised([eng.take()], "pop_slide")) == 1


def test_pop_surge_is_a_rise_of_ten_percent_over_the_minimum_of_the_previous_rows() -> None:
    eng = engine()
    t0 = steady(eng)
    (a,) = raised(feed(eng, [11_100], t0))
    assert a["class"] == "pop_surge" and a["imp"] == 1 and a["surface"] == "alert"
    assert a["text"] == "The population rose from 10,000 to 11,100 in 250 steps."


def test_a_partial_row_is_no_sample_of_the_population_and_no_rate() -> None:
    """The row a stop writes off the grid, and the first one after a resume, span fewer steps than the metrics
    interval: they add no sample to the windows, no rate and no alert, so a quit and a resume change nothing
    the engine says (third review, stage 8)."""
    eng = engine()
    t0 = steady(eng)
    before = (list(eng.rows), list(eng.starved), list(eng.storm_deaths))
    eng.on_row(row(t0 + 1, 10_000, steps=1, storm=500, starved=100_000), t0 + 1)
    assert raised([eng.take()], "storm_toll") == []
    assert (list(eng.rows), list(eng.starved), list(eng.storm_deaths)) == before


def test_a_dip_that_only_a_partial_row_sees_is_not_a_fall() -> None:
    """Uninterrupted, the rows at the steps after 65 hold 60 (a fall of 7.7%, under the 8% that fires); a quit
    adds a row of 101 steps at a trough of 57 that the grid never samples, and the row after the resume spans
    149 steps. The fall they show is not raised, and neither enters the windows."""
    straight, quit_ = engine(cadence=250), engine(cadence=250)
    t = steady(straight, 10, 65)
    assert steady(quit_, 10, 65) == t
    out_a = feed(straight, [60, 60], t)
    quit_.on_row(row(t + 101, 57, steps=101), t + 101)  # the stop's row
    out_b = [quit_.take()]
    quit_.on_row(row(t + 250, 60, steps=149), t + 250)  # the first row after the resume
    out_b.append(quit_.take())
    quit_.on_row(row(t + 500, 60), t + 500)
    out_b.append(quit_.take())
    assert raised(out_a) == raised(out_b) == []
    assert not any(s in (t + 101, t + 250) for s, _ in quit_.rows)
    # a world whose metrics interval is not 250 asks the same of its own rows
    small = engine(cadence=10)
    steady(small, 10, 100)
    small.on_row(row(2_511, 10, steps=4), 2_511)
    assert raised([small.take()]) == [] and small.rows[-1][0] != 2_511


def test_the_first_partial_row_after_an_open_does_not_end_the_silent_first_evaluation() -> None:
    """The level classes arm silently at the first evaluation after an open; a partial row is not one, so a
    world resumed at a low population is not told "low" by the full row that follows it."""
    eng = engine(cadence=250)
    eng.prime_rows([row(STEP * (i + 1), 10_000) for i in range(24)])
    eng.on_row(row(6_101, 3_000, steps=101), 6_101)
    assert raised([eng.take()]) == []
    eng.on_row(row(6_250, 3_000), 6_250)
    assert raised([eng.take()]) == [], "the first full row after the open arms silently"


# --- storms -------------------------------------------------------------------------------------------------


def test_a_storm_toll_needs_ten_deaths_and_three_times_the_median_and_dedupes_for_1500_steps() -> None:
    eng = engine()
    t0 = steady(eng)
    out = []
    for i, d in enumerate([1, 10, 1, 30, 1, 1, 1, 1, 1, 35, 36]):
        t = t0 + STEP * (i + 1)
        eng.on_row(row(t, 10_000, storm=d), t)
        out.append(eng.take())
    # 10 (median 1) fires at i=1; 30 at i=3 is inside the 1,500 steps; 35 at i=9 is 2,000 steps on and fires;
    # 36 at i=10 is a rise to a level the median has not caught up with, but inside the dedupe again
    assert [i for i, t in enumerate(out) if raised([t], "storm_toll")] == [1, 9]
    (a,) = raised([out[1]], "storm_toll")
    assert a["surface"] == "log" and a["imp"] == 0
    assert a["text"] == "10 organisms died in storms in the latest 250 steps, against a median of 1."
    eng2 = engine()
    t1 = steady(eng2)
    for i, d in enumerate([9, 12, 12, 12]):  # nine is under the minimum; twelve against a median of 1 fires
        t = t1 + STEP * (i + 1)
        eng2.on_row(row(t, 10_000, storm=d), t)
    assert len(raised([eng2.take()], "storm_toll")) == 1
    eng3 = engine()
    t2 = steady(eng3)
    for i, d in enumerate([20] * 9 + [30]):  # a sustained high level is its own median: no spike
        t = t2 + STEP * (i + 1)
        eng3.on_row(row(t, 10_000, storm=d), t)
    assert [a["step"] for a in raised([eng3.take()], "storm_toll")] == [t2 + STEP]


def storms_at(*zetas: tuple[int, float]) -> list[dict[str, Any]]:
    return [{"id": i, "x": 10.0 * i, "y": 5.0, "r": 7.0, "zeta": z} for i, z in zetas]


def named_event(ident: int, name: str, *, initial: bool = False) -> dict[str, Any]:
    return {
        "kind": "named", "step": 0, "id": ident, "name": name, "other": 0, "other_name": None,
        "x": 10.0 * ident, "y": 5.0, "hemi": 0, "age": 0, "initial": initial,
    }  # fmt: skip


def test_a_storm_named_alert_is_on_the_alert_surface_and_flies_to_the_storm() -> None:
    eng = engine()
    ev = named_event(4, "Phaitraern")
    eng.on_tick(451_600, storms_at((4, 0.52)), [ev], [ev])
    taken = eng.take()
    (a,) = raised([taken])
    assert a["class"] == "storm_named" and a["imp"] == 2 and a["surface"] == "alert"
    assert (a["x"], a["y"]) == (40.0, 5.0) and a["ref"] == {"kind": "storm", "id": 4}
    assert a["title"] == "Storm Phaitraern named"
    assert a["text"] == "Storm Phaitraern reached named strength at step 451,600."
    check_all([taken])


def test_an_initial_naming_is_not_told_so_it_raises_nothing_but_the_storm_is_known_as_named() -> None:
    eng = engine()
    ev = named_event(3, "Chugrox", initial=True)
    eng.on_tick(450_025, storms_at((3, 0.51)), [ev], [])  # the chronicle told nothing
    assert eng.take().alerts == []
    eng.on_tick(450_050, storms_at((3, 0.75)), [], [])
    (a,) = raised([eng.take()], "storm_strong")
    assert (
        a["text"] == "Storm Chugrox is at 0.75 f0 at step 450,050."
        and a["surface"] == "log"
        and a["imp"] == 0
    )


def test_storm_strong_is_for_named_storms_only_and_dedupes_for_6000_steps() -> None:
    eng = engine()
    eng.on_tick(100, storms_at((1, 0.9)), [], [])  # unnamed: nothing
    assert eng.take().alerts == []
    ev = named_event(1, "Aa")
    eng.on_tick(200, storms_at((1, 0.5)), [ev], [ev])
    eng.take()
    eng.on_tick(225, storms_at((1, 0.7)), [], [])  # 0.70 itself counts
    assert len(raised([eng.take()], "storm_strong")) == 1
    eng.on_tick(250, storms_at((1, 0.9)), [], [])
    assert eng.take().alerts == []
    eng.on_tick(225 + al.STRONG_REARM, storms_at((1, 0.8)), [], [])
    assert len(raised([eng.take()], "storm_strong")) == 1
    dis = {"kind": "dissolved", "step": 0, "id": 1, "name": "Aa", "age": 9_000, "x": 1.0, "y": 2.0}
    eng.on_tick(9_000, [], [dis], [dis])
    (end,) = raised([eng.take()], "storm_ended")
    assert end["text"] == "Storm Aa dissolved after 9,000 steps."
    eng.on_tick(12_300, storms_at((1, 0.9)), [], [])  # a dissolved storm is no longer named (past the dedupe)
    assert eng.take().alerts == []


def test_an_unnamed_disturbance_that_was_told_is_a_storm_ended_log_line() -> None:
    eng = engine()
    dis = {"kind": "dissolved", "step": 0, "id": 10, "name": None, "age": 1_900, "x": 1.0, "y": 2.0}
    eng.on_tick(462_575, [], [dis], [dis])
    (a,) = raised([eng.take()], "storm_ended")
    assert a["text"] == "Disturbance 10 dissolved after 1,900 steps." and a["title"] == "Disturbance 10 ended"


# --- lineages and houses ------------------------------------------------------------------------------------


def tag(x: float) -> list[float]:
    return [x, 0.0, 0.0, 0.0]


def srec(
    k: int,
    groups: list[tuple[int, str, int, float]],
    rows: list[tuple[str, int, int]] = (),
    houses: list[tuple[int, str, int]] = (),
    *,
    origin: int = 0,
) -> dict[str, Any]:
    return {
        "step": STEP * k,
        "origin": origin,
        "thresholds": {"kin_enter": 87},
        "kin_groups": {
            "id": [g[0] for g in groups], "name": [g[1] for g in groups], "size": [g[2] for g in groups],
            "tag": [tag(g[3]) for g in groups], "x": [10.0 * g[0] for g in groups],
            "y": [20.0 * g[0] for g in groups], "z_mean": [None for _ in groups],
        },
        "events": {
            "family": ["kin_group" for _ in rows], "kind": [r[0] for r in rows], "id": [r[1] for r in rows],
            "other": [r[2] for r in rows],
        },
        "houses": {
            "id": [h[0] for h in houses], "name": [h[1] for h in houses], "size": [h[2] for h in houses],
        },
        "house_events": {"founded": {"id": [], "parent_house": []}, "extinct": {"id": []}},
    }  # fmt: skip


class Scene:
    """A chronicler and an engine fed the way the pacer feeds them."""

    def __init__(self, armed: Any = NONE_ARMED) -> None:
        self.chron, self.eng = Chronicler(0), engine(armed)
        self.out: list[Taken] = []

    def record(self, rec: dict[str, Any]) -> Taken:
        self.chron.from_record(rec)
        self.eng.on_record(rec, self.chron.told_threads, self.chron.threads)
        taken = self.eng.take()
        self.out.append(taken)
        return taken

    def run(self, recs: list[dict[str, Any]]) -> None:
        for r in recs:
            self.record(r)


ANCHOR = (1, "Anchor", 600, 0.9)
BIG = (2, "Brenn", 400, 0.0)  # >= 4 x kin_enter = 348: a toast-size group


def test_a_toast_size_lineage_formed_is_a_log_line_with_a_fly_to_point() -> None:
    sc = Scene()
    sc.run([srec(0, [ANCHOR])] + [srec(k, [ANCHOR, BIG]) for k in range(1, 5)])
    (a,) = raised(sc.out, "lineage_formed")
    assert a["surface"] == "log" and a["imp"] == 0 and a["watched"] is False
    assert a["text"] == "Brenn has held together since step 250: 400 members."
    assert (a["x"], a["y"]) == (20.0, 40.0) and a["ref"] == {"kind": "kin", "id": 2}
    check_all(sc.out)


def test_a_small_group_forming_raises_no_alert() -> None:
    sc = Scene()
    small = (2, "Brenn", 100, 0.0)
    sc.run([srec(0, [ANCHOR])] + [srec(k, [ANCHOR, small]) for k in range(1, 6)])
    assert raised(sc.out) == []


def test_an_ended_lineage_is_importance_three_on_the_alert_surface_if_watched_else_a_log_line() -> None:
    recs = [srec(0, [ANCHOR])] + [srec(k, [ANCHOR, BIG] if 1 <= k <= 10 else [ANCHOR]) for k in range(1, 16)]
    # nothing set: the three largest threads are watched, and a lone group beside a bigger one is among them
    sc = Scene(ALL_ARMED)
    sc.run(recs)
    (a,) = raised(sc.out, "lineage_ended")
    assert a["watched"] is True and a["surface"] == "alert" and a["imp"] == 3 and a["pause"] is True
    # watching an unrelated thread only: the ended one is not watched and is a log line
    sc2 = Scene()
    sc2.run(recs[:4])
    sc2.eng.watch_threads = [sc2.chron.threads.thread_of(1).id]
    sc2.run(recs[4:])
    (b,) = raised(sc2.out, "lineage_ended")
    assert b["watched"] is False and b["surface"] == "log" and b["imp"] == 0
    # four anchors bigger than the thread: it is not among the three largest
    many = [(10 + i, f"Big{i}", 5_000 + i, 0.2 + 0.1 * i) for i in range(4)]
    sc3 = Scene()
    sc3.run([srec(0, [ANCHOR, *many])])
    sc3.run([srec(k, [ANCHOR, *many, BIG] if 1 <= k <= 10 else [ANCHOR, *many]) for k in range(1, 16)])
    (c,) = raised(sc3.out, "lineage_ended")
    assert c["watched"] is False and c["surface"] == "log"


def test_watch_maps_a_kin_id_to_its_thread_limits_to_three_and_survives_a_reformation() -> None:
    sc = Scene()
    groups = [(i, f"G{i}", 500 + i, 0.15 * i) for i in range(1, 6)]
    sc.run([srec(0, groups), srec(1, groups)])
    assert sc.eng.watch(1, sc.chron.threads) is None
    assert sc.eng.watch(1, sc.chron.threads) is None and len(sc.eng.watch_threads) == 1
    assert sc.eng.watch(2, sc.chron.threads) is None and sc.eng.watch(3, sc.chron.threads) is None
    assert "already watched" in sc.eng.watch(4, sc.chron.threads)
    assert "no kin group 99" in sc.eng.watch(99, sc.chron.threads)
    assert sc.eng.unwatch(2, sc.chron.threads) is None
    assert "not watched" in sc.eng.unwatch(2, sc.chron.threads)
    assert len(sc.eng.watch_threads) == 2
    sc.eng.watch_off()
    assert sc.eng.watch_threads == []
    # a watched thread that re-forms under a new id stays watched
    sc.eng.watch(1, sc.chron.threads)
    tid = sc.eng.watch_threads[0]
    sc.run([srec(2, groups[1:]), srec(3, [(9, "G1", 501, 0.15), *groups[1:]])])
    assert sc.eng.watch_threads == [tid] and sc.chron.threads.thread_of(9).id == tid


def test_collapse_is_measured_against_the_maximum_of_the_four_previous_snapshots() -> None:
    sc = Scene()
    sizes = [1000, 1000, 800, 700, 590]
    sc.run([srec(k, [(1, "Teshash", s, 0.9), (2, "Other", 300, 0.1)]) for k, s in enumerate(sizes)])
    (a,) = raised(sc.out, "lineage_collapse")
    assert a["surface"] == "alert" and a["imp"] == 2 and a["watched"] is True
    assert a["text"] == "Teshash has 590 members, 59% of the 1,000 it had within 1,000 steps."
    assert a["title"] == "Teshash shrinking" and a["step"] == 4 * STEP
    sc2 = Scene()
    sc2.run(
        [
            srec(k, [(1, "Teshash", s, 0.9), (2, "Other", 300, 0.1)])
            for k, s in enumerate([1000, 1000, 800, 700, 610])
        ]
    )
    assert raised(sc2.out, "lineage_collapse") == []  # 61%: not enough
    sc3 = Scene()  # a drop that is the maximum leaving the window of 4: 1000 five snapshots back is forgotten
    sc3.run(
        [
            srec(k, [(1, "Teshash", s, 0.9), (2, "Other", 300, 0.1)])
            for k, s in enumerate([1000, 700, 700, 700, 700, 590])
        ]
    )
    assert raised(sc3.out, "lineage_collapse") == []


def test_a_merge_changes_the_partition_not_the_people() -> None:
    """Thread T (kin id 1) merges into group 2 and re-forms a snapshot later, much smaller: the drop is the
    people who joined group 2, so no collapse is raised. The same re-formation with no merged row is one."""

    def scene(rows: list[tuple[str, int, int]]) -> Scene:
        sc = Scene()
        other = (2, "Other", 300, 0.1)
        sc.run([
            srec(0, [(1, "Teshash", 1000, 0.9), other]),
            srec(1, [(1, "Teshash", 1000, 0.9), other]),
            srec(2, [other], rows=rows),
            srec(3, [(9, "Teshash", 300, 0.9), other]),
        ])  # fmt: skip
        return sc

    assert raised(scene([("merged", 1, 2)]).out, "lineage_collapse") == []
    sc = scene([])
    (a,) = raised(sc.out, "lineage_collapse")
    assert a["step"] == 3 * STEP and a["text"].startswith("Teshash has 300 members, 30% of the 1,000")


def test_collapse_fires_once_per_6000_steps() -> None:
    sc = Scene()
    sizes = [1000, 1000, 800, 700, 590, 400, 300]
    sc.run([srec(k, [(1, "Teshash", s, 0.9), (2, "Other", 300, 0.1)]) for k, s in enumerate(sizes)])
    assert len(raised(sc.out, "lineage_collapse")) == 1


def test_a_house_fading_is_a_log_line_once_by_size_or_by_leaving_the_table() -> None:
    sc = Scene()
    house = lambda k, size: [(77, "Mycleav", size)] if size else []  # noqa: E731
    sizes = [300] * 9 + [200, 120, 70, 60, 0, 0]
    sc.run([srec(k, [ANCHOR], houses=house(k, s)) for k, s in enumerate(sizes)])
    (a,) = raised(sc.out, "house_fading")
    assert a["surface"] == "log" and a["text"] == "House Mycleav fell to 23% of its peak of 300."
    assert a["step"] == 11 * STEP and a["ref"] == {"kind": "house", "id": 77}
    sc2 = Scene()
    sc2.run([srec(k, [ANCHOR], houses=house(k, s)) for k, s in enumerate([300] * 9 + [250, 0, 0])])
    (b,) = raised(sc2.out, "house_fading")
    assert b["text"] == "House Mycleav is no longer among the 16 largest houses."
    sc3 = Scene()  # a young house (7 listings) or a small one (peak 199) is not tracked
    sc3.run([srec(k, [ANCHOR], houses=house(k, s)) for k, s in enumerate([300] * 7 + [0, 0])])
    sc3.run([srec(k, [ANCHOR], houses=[(5, "Tiny", 199)] if k < 20 else []) for k in range(9, 22)])
    assert raised(sc3.out, "house_fading") == []


def test_an_observer_restart_closes_the_threads_silently_and_drops_the_watch() -> None:
    sc = Scene(ALL_ARMED)
    sc.run([srec(0, [ANCHOR, BIG]), srec(1, [ANCHOR, BIG])])
    assert sc.eng.watch(2, sc.chron.threads) is None and len(sc.eng.watch_threads) == 1
    # the observer restarted: the ids are renumbered, every old thread ends without a word
    sc.run([srec(2, [ANCHOR], origin=1), srec(3, [ANCHOR], origin=1)])
    assert sc.eng.watch_threads == [] and raised(sc.out, "lineage_ended") == []
    assert "no kin group 2" in sc.eng.watch(2, sc.chron.threads)


# --- effects and the director -----------------------------------------------------------------------------


def test_an_effect_ending_within_500_steps_is_noticed_once_and_cleared_at_its_end() -> None:
    eng = engine()
    eff = {"action": "regen_pulse", "params": {"mult": 0.8}, "start": 1_000, "end": 4_000, "source": "player"}
    for t in (2_000, 3_250):
        eng.on_row(row(t, 10_000), t, effects=[eff])
    assert eng.take().alerts == []
    eng.on_row(row(3_500, 10_000), 3_500, effects=[eff])  # 500 before the end
    (a,) = raised([eng.take()], "effect_ending")
    assert a["text"] == "Food x0.80 ends at step 4,000." and a["ref"] == {"kind": "power", "id": 1_000}
    assert a["imp"] == 1 and a["surface"] == "alert"
    eng.on_row(row(3_750, 10_000), 3_750, effects=[eff])
    assert eng.take().alerts == []
    eng.on_row(row(4_000, 10_000), 4_000, effects=[])  # reverted
    assert states([eng.take()], "effect_ending") == ["cleared"]


def test_director_moves_by_the_rule_tier_or_the_model_are_alerts_but_reverts_and_the_players_are_not() -> (
    None
):
    eng = engine()
    base = {"step_applied": 466_000, "action": "regen_pulse", "params": {"mult": 0.8}, "duration": 10_000}
    eng.on_row(
        row(466_000, 42_000), 466_000, applied=[{**base, "source": "revert"}, {**base, "source": "player"}]
    )
    assert eng.take().alerts == []
    eng.on_row(row(466_250, 42_000), 466_250, applied=[{**base, "source": "rule"}])
    (a,) = raised([eng.take()], "director_acted")
    assert a["imp"] == 2 and a["surface"] == "alert"
    assert a["text"] == "The director applied a regrowth pulse of x0.80 for 10,000 steps at step 466,000."
    eng.on_row(
        row(466_500, 42_000), 466_500, applied=[{**base, "source": "llm"}]
    )  # the same entry again: once
    assert raised([eng.take()], "director_acted") == []


# --- the rate limit, the pauses, the identities -----------------------------------------------------------


def storm_named_at(eng: AlertEngine, t: int, ident: int) -> None:
    ev = named_event(ident, f"S{ident}")
    eng.on_tick(t, storms_at((ident, 0.52)), [ev], [ev])


def test_at_most_four_importance_two_alerts_per_1500_steps_the_rest_fold_into_a_summary() -> None:
    eng = engine()
    for i in range(6):
        storm_named_at(eng, 1_000 + 25 * i, i + 1)
    taken = eng.take()
    assert [a["class"] for a in taken.alerts] == ["storm_named"] * 4 + ["summary"]
    assert [m["state"] for m in taken.log if m["class"] == "summary"] == ["raised", "updated"]  # two steps
    assert [a["state"] for a in taken.alerts if a["class"] == "summary"] == ["updated"]  # one record, newest
    held = [m for m in taken.log if m["suppressed_by"] == "cap"]
    assert len(held) == 2 and {m["class"] for m in held} == {"storm_named"}
    summary = taken.alerts[-1]
    assert summary["text"] == "2 more alerts were raised within 1,500 steps." and summary["imp"] == 1
    check_all([taken])
    # after the window the summary clears and the cap is open again
    eng.on_row(row(3_000, 10_000), 3_000)
    assert states([eng.take()], "summary") == ["cleared"]
    storm_named_at(eng, 3_025, 7)
    assert [a["class"] for a in eng.take().alerts] == ["storm_named"]


def test_importance_three_armed_classes_the_chapter_and_the_engine_are_exempt_from_the_cap() -> None:
    eng = engine(lambda cls: cls == "storm_named")  # an armed class is exempt
    for i in range(6):
        storm_named_at(eng, 1_000 + 25 * i, i + 1)
    assert len(raised([eng.take()], "storm_named")) == 6
    eng2 = engine()
    for i in range(4):
        storm_named_at(eng2, 1_000 + 25 * i, i + 1)
    eng2.on_chapter(1_200)
    eng2.failure(1_200, "boom")
    eng2.engine_check(10.0, 1_200, late_steps=0, achieved=0.0, requested=0.0, running=False)
    out = eng2.take()
    assert {a["class"] for a in out.alerts} == {"storm_named", "chapter", "engine"}
    # a fifth importance-3 alert: low population
    eng2.prime_rows([row(250 * i, 10_000) for i in range(1, 8)])
    eng2.on_row(row(1_950, 10_000), 1_950)  # the first evaluation after open
    eng2.take()
    eng2.on_row(row(2_200, 3_000), 2_200)
    assert {a["class"] for a in eng2.take().alerts} >= {"pop_low", "pop_fall"}


def test_an_armed_class_pauses_from_its_pause_min_and_only_on_the_alert_surface() -> None:
    eng = engine(ALL_ARMED)
    t0 = steady(eng)
    taken = feed(eng, [9_100], t0)[0]  # pop_fall at importance 2: the class pauses from 3
    assert raised([taken])[0]["pause"] is False and taken.pause is False
    ev = named_event(4, "Aa")
    eng.on_tick(t0 + 300, storms_at((4, 0.6)), [ev], [ev])
    taken = eng.take()
    assert raised([taken], "storm_named")[0]["pause"] is True and taken.pause is True
    eng.on_tick(t0 + 325, storms_at((4, 0.9)), [], [])  # a log-surface class never pauses, armed or not
    taken = eng.take()
    assert raised([taken], "storm_strong")[0]["pause"] is False and taken.pause is False


def test_pause_ok_false_keeps_an_alert_from_asking_for_a_pause() -> None:
    eng = engine(ALL_ARMED)
    ev = named_event(4, "Aa")
    eng.on_tick(100, storms_at((4, 0.6)), [ev], [ev], pause_ok=False)
    taken = eng.take()
    assert raised([taken])[0]["pause"] is False and taken.pause is False


def test_the_chapter_alert_flags_the_chapter_pause_but_does_not_ask_for_an_alert_pause() -> None:
    eng = engine(lambda cls: cls == "chapter")
    eng.on_chapter(6_000)
    taken = eng.take()
    (a,) = taken.alerts
    assert a["class"] == "chapter" and a["imp"] == 3 and a["pause"] is True and a["expires"] is None
    assert a["text"] == "The chapter ended at step 6,000." and taken.pause is False
    eng.clear_chapter(6_010)
    (c,) = eng.take().alerts
    assert c["state"] == "cleared" and c["version"] == 2 and c["id"] == a["id"] and c["pause"] is False
    eng2 = engine(NONE_ARMED)
    eng2.on_chapter(6_000)
    assert eng2.take().alerts[0]["pause"] is False
    eng3 = engine(lambda cls: True)
    eng3.on_chapter(6_000, pause_ok=False)
    assert eng3.take().alerts[0]["pause"] is False


def test_an_upgrade_to_the_pausing_importance_asks_for_the_pause_then() -> None:
    eng = engine(ALL_ARMED)
    t0 = steady(eng)
    first = feed(eng, [9_000], t0)[0]
    assert first.pause is False
    second = feed(eng, [8_000], t0 + STEP)[0]
    assert states([second], "pop_fall") == ["updated"] and second.pause is True
    assert second.alerts[0]["pause"] is True and second.alerts[0]["imp"] == 3


def test_ids_are_the_first_step_times_a_thousand_plus_600_plus_a_per_step_index() -> None:
    eng = engine()
    for i in range(3):
        storm_named_at(eng, 777, i + 1)
    ids = [a["id"] for a in eng.take().alerts]
    assert ids == [777_600, 777_601, 777_602]
    eng.on_chapter(777)
    assert eng.take().alerts[0]["id"] == 777_603


def test_ack_logs_the_acknowledgement_and_notes_an_unknown_id() -> None:
    eng = engine()
    storm_named_at(eng, 100, 1)
    (a,) = eng.take().alerts
    assert eng.ack(a["id"], 120) is None
    (line,) = eng.take().log
    assert line == {"id": a["id"], "version": 0, "step": 120, "class": "storm_named", "state": "acknowledged"}
    assert "no alert 5" in eng.ack(5, 120)


def test_the_engine_class_watches_late_steps_and_the_achieved_rate_on_the_injected_clock() -> None:
    eng = engine()
    for i in range(5):
        eng.engine_check(float(i), 10 * i, late_steps=i, achieved=14.0, requested=14.4, running=True)
    (a,) = raised([eng.take()], "engine")
    assert a["imp"] == 1 and a["text"] == "The play ran late: 3 steps were late within 10 seconds."
    for i in range(5, 60):  # late steps keep rising, but the key is deduped for 300 s of the clock
        eng.engine_check(float(i), 10 * i, late_steps=i, achieved=14.0, requested=14.4, running=True)
    assert raised([eng.take()], "engine") == []
    eng2 = engine()
    out = []
    for i in range(40):  # 10 steps a second against 14.4: running 5 s, slow for 20 s
        eng2.engine_check(float(i), i, late_steps=0, achieved=10.0, requested=14.4, running=True)
        out += raised([eng2.take()], "engine")
    assert len(out) == 1 and out[0]["text"].startswith(
        "The play ran at 10.0 steps a second against 14.4 for 20"
    )
    eng3 = engine()
    for i in range(60):  # paused: nothing
        eng3.engine_check(float(i), i, late_steps=0, achieved=0.0, requested=14.4, running=False)
    assert eng3.take().alerts == []
    eng4 = engine()
    for i in range(60):  # at the maximum rate the request is infinite: nothing
        eng4.engine_check(float(i), i, late_steps=0, achieved=30.0, requested=float("inf"), running=True)
    assert eng4.take().alerts == []


def test_a_presenter_failure_is_an_importance_three_alert_that_asks_for_no_pause() -> None:
    eng = engine(ALL_ARMED)
    eng.failure(300, "OSError: [Errno 28] No space left on device")
    taken = eng.take()
    (a,) = taken.alerts
    assert a["imp"] == 3 and a["pause"] is False and taken.pause is False
    assert a["text"] == "The play stopped after an error: OSError: [Errno 28] No space left on device"


# --- the experimental class ---------------------------------------------------------------------------------


def territory_scene(armed: Any = NONE_ARMED, **kw: Any) -> tuple[Scene, np.ndarray]:
    sc = Scene(armed)
    sc.eng.experimental = kw.get("experimental", True)
    sc.run([srec(0, [ANCHOR, BIG]), srec(1, [ANCHOR, BIG])])
    sc.eng.watch(2, sc.chron.threads)
    sc.eng.named = {4}
    owner = np.full((16, 16), -1, np.int64)
    owner[2:6, 2:6] = 2  # sixteen cells of the watched thread around (40, 40) in a world of 256
    return sc, owner


def test_storm_over_lineage_needs_three_cells_and_a_quarter_of_the_territory_and_is_dark_by_default() -> None:
    sc, owner = territory_scene()
    storm = [{"id": 4, "x": 40.0, "y": 40.0, "r": 20.0, "zeta": 0.6}]
    sc.eng.names = {4: "Aa"}
    sc.eng.on_territory(1_000, owner, storm, sc.chron.threads, 256.0)
    (a,) = raised([sc.eng.take()], "storm_over_lineage")
    assert a["surface"] == "log" and a["calibrated"] is False and a["imp"] == 0
    assert a["text"] == "4 cells held by Brenn lie within 30.0 of Storm Aa."  # 4 of 16: exactly a quarter
    sc.eng.on_territory(1_250, owner, storm, sc.chron.threads, 256.0)  # deduped for 3,000 steps
    assert sc.eng.take().alerts == []
    far = [{"id": 4, "x": 200.0, "y": 200.0, "r": 20.0, "zeta": 0.6}]
    sc2, owner2 = territory_scene()
    sc2.eng.on_territory(1_000, owner2, far, sc2.chron.threads, 256.0)
    assert sc2.eng.take().alerts == []
    sc3, owner3 = territory_scene(experimental=False)  # not enabled: nothing is evaluated
    sc3.eng.on_territory(1_000, owner3, storm, sc3.chron.threads, 256.0)
    assert sc3.eng.take().alerts == []


def test_storm_over_lineage_sees_across_the_edge_of_the_world() -> None:
    sc, owner = territory_scene()
    owner[:] = -1
    owner[0:4, 12:16] = 2  # the territory touches the right edge of the grid
    storm = [{"id": 4, "x": 4.0, "y": 24.0, "r": 20.0, "zeta": 0.6}]  # centred just over the left edge
    sc.eng.on_territory(1_000, owner, storm, sc.chron.threads, 256.0)
    assert len(raised([sc.eng.take()], "storm_over_lineage")) == 1


# --- the backlog of a resumed world -------------------------------------------------------------------------


def line(**kw: Any) -> dict[str, Any]:
    base = {
        "id": 600, "version": 1, "step": 100, "class": "storm_named", "key": "1", "state": "raised",
        "surface": "alert", "imp": 2, "title": "t", "text": "x", "x": None, "y": None, "z": None, "ref": None,
        "pause": True, "expires": 3_100, "watched": False, "calibrated": True, "nominal": 2,
        "suppressed_by": None,
    }  # fmt: skip
    return {**base, **kw}


def test_the_backlog_is_the_newest_state_of_the_unexpired_unacknowledged_alerts_without_a_pause() -> None:
    lines = [
        line(id=600),
        line(id=601, expires=2_000),  # expired at 2,500
        line(id=602),
        line(id=602, version=2, state="cleared"),  # cleared
        line(id=603),
        {"id": 603, "version": 0, "step": 150, "class": "storm_named", "state": "acknowledged"},
        line(id=604, imp=3, expires=None),  # importance 3: the clearing belongs to a session that is over
        line(id=605, **{"class": "chapter", "imp": 3, "expires": None}),
        line(id=606, suppressed_by="cap"),
        line(id=607, version=2, state="updated", text="newer"),
        line(id=607, version=1),
    ]
    got = al.backlog(lines, 2_500)
    assert [a["id"] for a in got] == [600, 607]
    assert (
        got[1]["text"] == "newer" and got[1]["state"] == "updated" and all(a["pause"] is False for a in got)
    )
    assert set(got[0]) == streamrec.ALERT_KEYS
    for a in got:
        streamrec.check_alert(a, al.CLASSES)
    assert len(al.backlog([line(id=600 + i, step=i) for i in range(30)], 0, limit=16)) == 16


# --- the thresholds, at their boundaries ----------------------------------------------------------------


def test_the_starvation_clause_needs_four_rows_of_history() -> None:
    eng = engine()
    t0 = steady(eng, n=3)  # the first row is the silent one; the history holds 3 rows of starvation
    eng.on_row(row(t0 + STEP, 8_000, starved=50_000), t0 + STEP)
    assert "starved" not in raised([eng.take()])[0]["text"]


def test_pop_low_needs_a_population_strictly_under_half_of_the_maximum() -> None:
    eng = engine()
    t0 = steady(eng)
    eng.on_row(row(t0 + STEP, 5_000), t0 + STEP)  # exactly half
    assert raised([eng.take()], "pop_low") == []
    eng.on_row(row(t0 + 2 * STEP, 4_999), t0 + 2 * STEP)
    assert len(raised([eng.take()], "pop_low")) == 1


def test_pop_slide_stays_up_until_the_chapter_end_is_back_to_ninety_percent() -> None:
    eng = engine()
    for end, alive in ((6_000, 10_000), (12_000, 7_900)):
        eng.on_row(row(end, alive), end)
    assert len(raised([eng.take()], "pop_slide")) == 1
    eng.on_row(row(18_000, 8_500), 18_000)  # 85% of 10,000: between the thresholds
    assert states([eng.take()], "pop_slide") == []
    eng.on_row(row(24_000, 9_100), 24_000)  # 91% of the maximum of the three before: back
    assert states([eng.take()], "pop_slide") == ["cleared"]


def test_collapse_fires_at_exactly_sixty_percent() -> None:
    sc = Scene()
    sc.run(
        [
            srec(k, [(1, "Teshash", s, 0.9), (2, "Other", 300, 0.1)])
            for k, s in enumerate([1000, 1000, 800, 700, 600])
        ]
    )
    (a,) = raised(sc.out, "lineage_collapse")
    assert a["text"].startswith("Teshash has 600 members, 60% of the 1,000")


def test_a_house_with_eight_listings_that_falls_to_exactly_a_quarter_of_its_peak_is_fading() -> None:
    sc = Scene()
    sizes = [300] * 7 + [75]  # eight listings, the last at 75 = 25% of 300
    sc.run([srec(k, [ANCHOR], houses=[(77, "Mycleav", s)]) for k, s in enumerate(sizes)])
    (a,) = raised(sc.out, "house_fading")
    assert a["text"] == "House Mycleav fell to 25% of its peak of 300."
    sc2 = Scene()
    sc2.run([srec(k, [ANCHOR], houses=[(77, "Mycleav", s)]) for k, s in enumerate([300] * 6 + [75])])
    assert raised(sc2.out, "house_fading") == []  # seven listings: too young


def test_the_cap_window_is_open_again_exactly_1500_steps_after_the_alert_that_opened_it() -> None:
    eng = engine()
    for i in range(4):
        storm_named_at(eng, 1_000, i + 1)
    assert len(raised([eng.take()], "storm_named")) == 4
    storm_named_at(eng, 1_000 + al.CAP_WINDOW - 1, 5)
    assert raised([eng.take()], "storm_named") == []  # held
    storm_named_at(eng, 1_000 + al.CAP_WINDOW, 6)
    assert len(raised([eng.take()], "storm_named")) == 1


def test_log_surface_alerts_do_not_count_toward_the_cap() -> None:
    eng = engine()
    for i in range(6):  # six lineage_formed lines (nominal importance 2, importance 0 on the wire)
        eng._fire("lineage_formed", str(i), 1_000, title="t", text="x", pause_ok=True)
    assert len(eng.take().alerts) == 6
    for i in range(4):
        storm_named_at(eng, 1_000, 10 + i)
    assert len(raised([eng.take()], "storm_named")) == 4


def test_the_late_steps_window_is_ten_seconds() -> None:
    eng = engine()
    for i, now in enumerate((0.0, 4.0, 8.0, 12.0)):  # one late step every 4 s: 2 within any 10 s
        eng.engine_check(now, i, late_steps=i, achieved=14.4, requested=14.4, running=True)
    assert eng.take().alerts == []


def test_a_slow_world_is_given_five_seconds_to_settle_and_twenty_to_be_slow() -> None:
    eng = engine()
    fired = []
    for i in range(40):
        eng.engine_check(float(i), i, late_steps=0, achieved=10.0, requested=14.4, running=True)
        if raised([eng.take()], "engine"):
            fired.append(i)
    assert fired == [25]  # running since 0, slow since 5, 20 s later
    eng2 = engine()
    for i in range(10):
        eng2.engine_check(float(i), i, late_steps=0, achieved=14.0, requested=14.4, running=True)
    assert eng2.take().alerts == []  # 14.0 is 97% of the request: not slow


def test_the_backlog_leaves_out_an_alert_that_expires_at_the_step_it_is_opened_at() -> None:
    assert [a["id"] for a in al.backlog([line(id=600, expires=2_500)], 2_499)] == [600]
    assert al.backlog([line(id=600, expires=2_500)], 2_500) == []


def test_an_alert_the_session_never_raised_can_be_acknowledged_once_the_engine_knows_it() -> None:
    eng = engine()
    assert "no alert 600" in eng.ack(600, 10)
    eng.know(al.backlog([line(id=600)], 0))
    assert eng.ack(600, 10) is None and eng.take().log[0]["state"] == "acknowledged"


def test_storm_over_lineage_needs_both_the_three_cells_and_the_quarter() -> None:
    sc, owner = territory_scene()
    wide = [{"id": 4, "x": 40.0, "y": 40.0, "r": 20.0, "zeta": 0.6}]  # 4 of 16 cells: a quarter
    small = np.full((16, 16), -1, np.int64)
    small[2:4, 2:3] = 2  # two cells of territory, both inside: a share of 100% but fewer than three cells
    sc.eng.on_territory(1_000, small, wide, sc.chron.threads, 256.0)
    assert sc.eng.take().alerts == []
    big = np.full((16, 16), -1, np.int64)
    big[0:10, 0:10] = 2  # 100 cells, the storm covers 4 of them: three cells but 4%
    sc.eng.on_territory(1_000, big, wide, sc.chron.threads, 256.0)
    assert sc.eng.take().alerts == []
    sc.eng.on_territory(1_000, owner, wide, sc.chron.threads, 256.0)  # 4 of 16: both hold
    assert len(raised([sc.eng.take()], "storm_over_lineage")) == 1


# --- the Mac-day play of p0 ---------------------------------------------------------------------------------


def logs() -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    def read(name: str) -> list[dict]:
        return [json.loads(x) for x in (DATA / name).read_text().splitlines() if x.strip()]

    return read("metrics.jsonl"), read("social.jsonl"), read("storms.jsonl"), read("director.jsonl")


def p0_armed(cls: str) -> bool:
    """The server defaults: the chapter pause and ``default_autopause``."""
    return al.default_autopause().get(cls, cls == "chapter")


@pytest.fixture(scope="module")
def p0() -> tuple[list[dict[str, Any]], AlertEngine]:
    m, s, st, d = logs()
    return al.fold_logs(m, s, st, d, limit=470_000, armed=p0_armed)


def counts(log: list[dict[str, Any]]) -> Counter:
    return Counter(m["class"] for m in log if m["state"] == "raised" and m["suppressed_by"] is None)


def test_the_p0_window_raises_the_alerts_the_spec_measured(p0: Any) -> None:
    """Spec §4.4 on the 20,000 steps of the Mac-day play (82 rows, 23.7 min at 1x). Differences from the
    spec's table, all from its rules read exactly: ``house_fading`` is 4 (the table counted the houses that
    left the top 16, not those that fell to a quarter of their peak while listed) and ``storm_ended`` counts
    the told end of an unnamed disturbance too (2, as the table says)."""
    log, _ = p0
    assert counts(log) == Counter(
        {
            "chapter": 3,  # 456,000, 462,000, 468,000
            "pop_fall": 1,  # 466,250: the aftermath of the director's own move (provenance rule)
            "director_acted": 1,  # the regen_pulse at 466,000
            "storm_named": 6,
            "storm_strong": 3,
            "storm_toll": 3,
            "storm_ended": 2,
            "lineage_formed": 4,
            "lineage_ended": 5,
            "house_fading": 4,
        }
    )  # fmt: skip
    assert counts(log)["lineage_formed"] + counts(log)["lineage_ended"] == 9  # the 9 toast-size threads
    assert not [m for m in log if m["suppressed_by"]], "the cap held nothing back"


def test_the_only_population_alert_is_the_aftermath_of_the_directors_move(p0: Any) -> None:
    """Raised at 13% under the maximum as the aftermath of the move (importance 1, no pause), and raised to
    importance 3 at 466,500 when it deepens to 17.5% (a hard fall is one whatever came before it; third
    review, stage 8), the entry still stated."""
    log, _ = p0
    fall = [m for m in log if m["class"] == "pop_fall"]
    assert [m["state"] for m in fall] == ["raised", "updated", "cleared"]
    assert (fall[1]["step"], fall[1]["imp"], fall[1]["pause"]) == (466_500, 3, True)
    assert fall[1]["ref"] == {"kind": "power", "id": 466_000} and fall[1]["text"].endswith(
        "since step 466,000."
    )
    first = fall[0]
    assert (first["step"], first["imp"], first["pause"]) == (466_250, 1, False)
    assert first["ref"] == {"kind": "power", "id": 466_000}
    assert first["text"] == (
        "The population fell from 43,335 to 37,310 in 1,500 steps. "
        "A regrowth pulse of x0.80 for 10,000 steps has been active since step 466,000."
    )
    assert [m for m in log if m["class"] in ("pop_low", "pop_slide", "pop_surge")] == []


def test_with_the_defaults_the_p0_window_pauses_only_at_its_three_chapter_ends(p0: Any) -> None:
    log, _ = p0
    pausing = [x for x in log if x["state"] == "raised" and x["pause"]]
    assert [(x["class"], x["step"]) for x in pausing] == [
        ("chapter", 456_000), ("chapter", 462_000), ("chapter", 468_000),
    ]  # fmt: skip
    alert_surface = [x for x in log if x["state"] == "raised" and x["surface"] == "alert"]
    assert Counter(x["class"] for x in alert_surface) == Counter(
        {"chapter": 3, "pop_fall": 1, "director_acted": 1, "storm_named": 6}
    )
    loud = [x for x in alert_surface if x["imp"] >= 2]
    assert len(loud) == 10 and len(loud) / 23.7 <= 0.6  # 0.42 a minute


def test_the_p0_alerts_are_pinned_by_a_digest_of_their_texts(p0: Any) -> None:
    log, _ = p0
    blob = json.dumps(
        [
            [m["id"], m["version"], m["class"], m["key"], m["state"], m["surface"], m["imp"], m["text"]]
            for m in log
        ],
        separators=(",", ":"),
    )
    assert (
        len(log) == 36
    )  # 33 raised or cleared by their condition, one update (the fall), 2 chapter alerts the fold never `run`
    assert hashlib.sha256(blob.encode()).hexdigest() == PINNED_LOG


def test_every_p0_alert_is_a_valid_record_inside_the_caps_and_the_lexicon(p0: Any) -> None:
    log, _ = p0
    never = tuple([*lexicon.NEVER_LISTS, *lexicon.LABEL_LISTS])
    for m in log:
        a = {k: m[k] for k in streamrec.ALERT_KEYS}
        streamrec.check_alert(a, al.CLASSES)
        assert lexicon.scan(m["text"], never) == [], m["text"]
        assert lexicon.scan(m["title"], never) == [], m["title"]
    assert max(len(m["text"]) for m in log) <= 160


def test_a_watched_largest_thread_turns_its_lineage_changes_into_alert_surface_alerts() -> None:
    m, s, st, d = logs()
    plain, _ = al.fold_logs(m, s, st, d, limit=470_000)
    # the largest thread of the first record: watch it alone
    first = s[0]["kin_groups"]
    big = first["id"][int(np.argmax(first["size"]))]
    watched, _ = al.fold_logs(m, s, st, d, limit=470_000, watch=[big])
    ended = lambda log: [x for x in log if x["class"] == "lineage_ended"]  # noqa: E731
    assert len(ended(watched)) == len(ended(plain))
    assert sum(x["watched"] for x in ended(watched)) <= 1


def test_the_fold_is_a_pure_function_of_the_logs() -> None:
    m, s, st, d = logs()
    a, _ = al.fold_logs(m, s, st, d, limit=470_000)
    b, _ = al.fold_logs(m, s, st, d, limit=470_000)
    assert a == b


PINNED_CONSTANTS = "5efb0924bb3329c87296a68c889efbd20fae62fb44a0d29537a958d8bb2af253"
# (stage 8: "Daudrall IV", a name taken at creation, and the update of the fall at 466,500)
PINNED_LOG = "3e56c84973245218f1a90a6ec112db1645466340a0789e3a705704609e47fe05"
