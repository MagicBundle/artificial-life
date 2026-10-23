"""The player's powers as pure functions (G3b spec §6, DESIGN §34.5): the names and the game table, the
request a ``power`` line makes, influence and readiness as folds of the director log, the since card's rows,
the column count of a strike, the precedent table and the player's words. The pacer's use of them (``PLAN``
records, the queue, ``commands.jsonl``, replay) is ``test_play_powers.py``.

Every string a player reads is run through the report's lexicon here (spec §5.3): the causal, modal and
evaluative lists and the label names are forbidden; the lever words (regen, patch, pulse, director) are the
interventions' own and allowed.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np
import pytest

from alife import play, streamrec
from alife.config import load_config
from alife.director import PlayerRequest, actions, pacing
from alife.game import lexicon, powers
from alife.game.precedents import PRECEDENT_VERSION, TABLE, direction, precedent_object

CFG = load_config(preset="64k3d")  # the planet: 320 wide, 8 food patches
FORBIDDEN = (*lexicon.NEVER_LISTS, *lexicon.LABEL_LISTS)


def clean(text: str) -> None:
    assert lexicon.scan(text, FORBIDDEN) == [], text


def line(power: str, id: int = 1, verb: str = "apply", **params: float) -> play.PowerLine:
    return play.PowerLine(verb, power, id, tuple(params.items()))


def fwd(step: int, power: str, **over: Any) -> dict[str, Any]:
    """A player's forward entry as the director writes it."""
    action = actions.GAME_POWERS[power]
    spec = actions.GAME_SPECS[action]
    base = {
        "step_applied": step,
        "source": "player",
        "persona": "gardener",
        "action": action,
        "params": {
            "food": {"mult": 1.3},
            "bloom": {"patch": 1, "gain": 1.75},
            "blight": {"patch": 2, "gain": 0.0},
            "strike": {"center": [10.0, 20.0], "radius": 12.5, "kill_p": 0.45},
        }[power],
        "prev": {},
        "duration": 6000 if spec.duration else None,
        "reason": f"the player used {power}",
        "vitals_step": step,
        "vitals": {"n_alive": 1.0},
        "scope": "player",
        "command": 1,
        "power": power,
        "specs": "game-v0",
        "cost": spec.cost,
    }
    return {**base, **over}


def revert(step: int, action: str, origin: int, **over: Any) -> dict[str, Any]:
    params = {"regen_pulse": {"mult": 1.0}, "fertility_shift": {"patch": 1, "gain": 1.0}}.get(
        action, {"patch": 2, "gain": 1.0}
    )
    return {
        "step_applied": step,
        "source": "revert",
        "action": action,
        "params": params,
        "prev": {},
        "duration": None,
        "scope": "player",
        "reverts": origin,
        **over,
    }


# --- the table ----------------------------------------------------------------------------------------------


def test_the_power_specs_are_the_game_tables_powers() -> None:
    assert {n: s.action for n, s in powers.POWER_SPECS.items()} == dict(actions.GAME_POWERS)
    assert tuple(powers.POWER_SPECS) == play.POWER_NAMES
    assert set().union(*(s.keys for s in powers.POWER_SPECS.values())) <= set(play.POWER_KEYS)
    assert {a: n for n, a in actions.GAME_POWERS.items()} == powers.POWER_OF_ACTION
    assert {s.label for s in powers.POWER_SPECS.values()} == {"Food", "Bloom", "Blight", "Strike"}
    assert {s.scope for s in powers.POWER_SPECS.values()} == {"global", "patch", "column"}
    for spec in powers.POWER_SPECS.values():
        clean(spec.blurb)
        assert spec.blurb.endswith(".")


def test_the_game_table_is_reachable_from_the_powers_module_and_is_the_same_object() -> None:
    assert powers.GAME_SPECS is actions.GAME_SPECS and powers.GAME_SPEC_VERSION == "game-v0"


def test_the_header_block_states_each_power_with_the_tables_numbers() -> None:
    block = powers.header_block(CFG)
    assert list(block) == ["food", "bloom", "blight", "strike"]
    streamrec.powers_block(block)  # the header encoder accepts it
    food, bloom, blight, strike = (block[n] for n in block)
    assert food["params"] == {"mult": [[0.6, 0.9], [1.15, 1.5]]} and food["duration"] == [3000, 12000]
    assert (food["cost"], food["cooldown"], food["scope"]) == (1, 3000, "global")
    assert bloom["params"] == {"gain": [1.5, 2.0]} and bloom["scope"] == "patch" and bloom["cooldown"] == 3000
    assert blight["params"] == {} and (blight["cost"], blight["cooldown"]) == (2, 6000)
    assert strike["duration"] is None and strike["scope"] == "column"
    assert strike["params"] == {
        "kill_p": [0.3, 0.6],
        "radius": [6.25, 25.0],
    }  # 20 to 80, scaled by 320 / 1024
    assert all(
        set(b) == {"action", "label", "cost", "cooldown", "scope", "duration", "params"}
        for b in block.values()
    )
    small = powers.header_block(load_config(preset="tiny3d"))["strike"]["params"]["radius"]
    assert small == [2.5, 10.0]  # the same share of the floor on a world 128 wide


# --- the request --------------------------------------------------------------------------------------------

PATCHES = np.array([[10.0, 10.0], [300.0, 20.0], [160.0, 160.0]], np.float32)


def test_a_food_line_makes_a_request_with_the_default_duration() -> None:
    req, raw = powers.request_of(line("food", mult=1.3), CFG, PATCHES)
    assert req == PlayerRequest(1, "food", {"mult": 1.3}, powers.DEFAULT_STEPS)
    assert raw == {"mult": 1.3, "steps": 6000}
    req, raw = powers.request_of(line("food", 4, mult=0.8, steps=9000), CFG, PATCHES)
    assert req.duration == 9000 and raw == {"mult": 0.8, "steps": 9000}


def test_bloom_and_blight_take_a_patch_or_the_nearest_to_a_click() -> None:
    req, raw = powers.request_of(line("bloom", patch=2, gain=1.9, steps=3000), CFG, PATCHES)
    assert req.raw == {"patch": 2, "gain": 1.9} and req.duration == 3000
    assert raw == {"patch": 2, "gain": 1.9, "steps": 3000}
    req, raw = powers.request_of(line("blight", x=302, y=15), CFG, PATCHES)  # 5.4 from patch 1 at (300, 20)
    assert req.raw == {"patch": 1} and raw == {"patch": 1, "steps": 6000}
    req, _ = powers.request_of(line("blight", x=318, y=15), CFG, PATCHES)  # 13 from patch 0 across the seam
    assert req.raw == {"patch": 0}
    req, raw = powers.request_of(
        line("bloom", patch=0), CFG, PATCHES
    )  # the default gain is the middle of 1.5 to 2
    assert req.raw["gain"] == 1.75 and raw["gain"] == 1.75


def test_a_strike_takes_a_column_with_default_radius_and_kill_chance() -> None:
    req, raw = powers.request_of(line("strike", x=100, y=50), CFG, PATCHES)
    assert req == PlayerRequest(1, "strike", {"x": 100, "y": 50, "radius": 15.625, "kill_p": 0.45}, None)
    assert raw == {"x": 100, "y": 50, "radius": 15.625, "kill": 0.45}
    req, raw = powers.request_of(line("strike", x=1, y=2, radius=7, kill=0.3), CFG, PATCHES)
    assert req.raw["kill_p"] == 0.3 and raw["kill"] == 0.3


def test_a_cancel_line_asks_only_for_the_early_end() -> None:
    req, raw = powers.request_of(line("food", 5, "cancel"), CFG, PATCHES)
    assert req == PlayerRequest(5, "food", {}, None, cancel=True) and raw == {}


@pytest.mark.parametrize(
    ("ln", "message"),
    [
        (line("food", gain=1.2), "Food takes mult and steps, not gain."),
        (line("food", steps=6000), "Food is missing mult, the factor on food regrowth."),
        (line("bloom", gain=1.7), "Bloom is missing a patch: give patch, or x and y for the nearest."),
        (line("blight", x=3), "Blight is missing a patch: give patch, or x and y for the nearest."),
        (line("bloom", patch=3), "Patch 3 does not exist; patches are numbered 0 to 2."),
        (line("strike", x=3), "Strike is missing x and y, the centre of the column."),
        (line("strike", x=3, y=4, steps=100), "Strike takes x, y, radius and kill, not steps."),
        (line("blight", patch=1, mult=1.2), "Blight takes patch, x, y and steps, not mult."),
    ],
)
def test_a_request_the_power_cannot_make_is_refused_in_the_players_words(
    ln: play.PowerLine, message: str
) -> None:
    with pytest.raises(powers.PowerError) as e:
        powers.request_of(ln, CFG, PATCHES)
    assert str(e.value) == message
    clean(message)


def test_the_nearest_patch_is_the_torus_minimum_image_and_the_lowest_index_on_a_tie() -> None:
    p = np.array([[0.0, 0.0], [100.0, 0.0], [200.0, 0.0]], np.float32)
    assert powers.nearest_patch(p, 310.0, 0.0, 320.0) == 0  # 10 from patch 0 across the wrap
    assert (
        powers.nearest_patch(p, 50.0, 0.0, 100.0) == 0
    )  # a tie between patches 0 and 1 at size 100: the lowest
    assert powers.nearest_patch(p, 90.0, 0.0, 320.0) == 1


def test_the_engines_messages_are_put_in_the_players_words() -> None:
    dead = powers.player_words("regen_pulse: mult 1.0 lies in the range 0.9 to 1.15, which is refused")
    assert dead == "Food x1 lies between 0.9 and 1.15, where it makes no difference worth a cost."
    assert powers.player_words("catastrophe is cooling down") == "Strike is cooling down."
    assert powers.player_words("fertility_shift went wrong.") == "Bloom went wrong."
    for text in (dead, powers.player_words("habitat_loss is busy")):
        clean(text)
    # a message with a word the lexicon forbids is not shown
    assert powers.player_words("no effect of regen_pulse to cancel") == powers.REFUSED_ELSE
    clean(powers.REFUSED_ELSE)


# --- influence and readiness --------------------------------------------------------------------------------


def test_influence_counts_the_players_costs_in_the_window_and_says_when_each_unit_returns() -> None:
    entries = [fwd(100, "food"), fwd(500, "blight", command=2), fwd(900, "strike", command=3)]
    assert powers.influence([], 1000) == {"have": 6, "cap": 6, "window": 18_000, "free_at": []}
    got = powers.influence(entries, 1000)
    assert got == {"have": 2, "cap": 6, "window": 18_000, "free_at": [18_100, 18_500, 18_500, 18_900]}
    # an entry leaves the window at step + window, exactly (budget_used is the open end)
    assert powers.influence(entries, 18_099)["have"] == 2
    assert powers.influence(entries, 18_100)["have"] == 3
    assert powers.influence(entries, 18_900)["have"] == 6
    rule = {
        "step_applied": 200,
        "source": "rule",
        "action": "regen_pulse",
        "params": {"mult": 0.8},
        "cost": 1,
    }
    assert powers.influence([rule, *entries], 1000) == got  # the rule tier's spending is its own budget


def test_a_power_that_costs_exactly_what_is_left_is_not_busy() -> None:
    pulses = [fwd(100 + 10 * i, "food", command=2 + i) for i in range(6)]  # Strike has never been used
    assert powers.influence(pulses[:5], 200)["have"] == 1
    assert powers.blocked(pulses[:5], "strike", {}, 200) == (None, None)  # one unit left, one unit asked
    ready, why = powers.blocked(pulses, "strike", {}, 200)
    assert (
        why == "Strike costs 1 unit; 0 left until step 18,100." and ready == 18_100
    )  # the singular, and the step


def test_a_power_is_busy_while_it_cools_down_costs_too_much_or_its_lever_is_held() -> None:
    f = fwd(100, "food")
    ready, why = powers.blocked([f], "food", {}, 3000)
    # the effect runs 100..6100 and cools down 3000 more
    assert (ready, why) == (9100, "Food is cooling down until step 9,100.")
    ended = revert(6100, "regen_pulse", 100)  # the director ends the effect at 100 + 6000
    assert powers.blocked([f, ended], "food", {}, 9100) == (None, None)
    assert powers.blocked([f, ended], "food", {}, 9099)[1] == "Food is cooling down until step 9,100."
    spent = [fwd(100, "blight", command=2, duration=3000), fwd(110, "blight", command=3, duration=3000)]
    ready, why = powers.blocked(spent, "strike", {}, 200)
    assert why is None  # 4 of 6 used: a unit of Strike is there
    heavy = [fwd(100 + 10 * i, p, command=2 + i) for i, p in enumerate(["food", "bloom", *["strike"] * 4])]
    ready, why = powers.blocked(heavy, "blight", {"patch": 5, "gain": 0.0}, 200)
    assert why == "Blight costs 2 units; 0 left until step 18,110."  # the second unit to return is Bloom's
    assert ready == 18_110
    clean(why)
    held = [fwd(100, "bloom")]  # patch 1 until step 6100
    ready, why = powers.blocked(held, "blight", {"patch": 1, "gain": 0.0}, 300)
    assert why == "Blight is busy: patch 1 is held until step 6,100." and ready == 6100
    assert powers.blocked(held, "blight", {"patch": 2, "gain": 0.0}, 300) == (None, None)
    ready, why = powers.blocked([fwd(100, "food", params={"mult": 0.8})], "food", {}, 6200)
    assert why == "Food is cooling down until step 9,100." and ready == 9100
    other = {**fwd(0, "food"), "source": "rule", "scope": "rule", "duration": 10_000, "cost": 1}
    del other["power"], other["specs"], other["command"]
    ready, why = powers.blocked([other], "food", {}, 500)
    assert why == "Food is busy: the regen lever is held until step 10,000." and ready == 10_000
    assert "regen lever" in why


def test_the_priority_of_the_reasons_is_cooldown_then_units_then_lever() -> None:
    """Food runs 10..3010 and cools down until 6010; every unit is spent; a rule's pulse also holds the lever.
    The sentence is the cooldown's; the step it is ready is the latest of the three."""
    rule = {**fwd(0, "food"), "source": "rule", "scope": "rule", "duration": 20_000}
    entries = [
        fwd(10, "food", duration=3000),
        fwd(20, "bloom", command=2),
        fwd(30, "strike", command=3),
        fwd(40, "strike", command=4),
        fwd(50, "strike", command=5),
        fwd(60, "strike", command=6),
        rule,
    ]
    assert powers.influence(entries, 100)["have"] == 0
    ready, why = powers.blocked(entries, "food", {}, 100)
    assert why == "Food is cooling down until step 6,010."
    assert ready == 20_000  # the rule's pulse holds the regen lever the longest


def test_the_states_of_the_four_powers_for_the_environment_record() -> None:
    assert powers.power_states([], 100) == {n: {"ready": None, "busy": None} for n in play.POWER_NAMES}
    states = powers.power_states([fwd(100, "food")], 300)
    assert states["food"] == {"ready": 9100, "busy": "Food is cooling down until step 9,100."}
    assert states["bloom"] == {"ready": None, "busy": None}
    streamrec.check_envs({"powers": states})


def test_the_patches_block_lists_the_food_patches_with_their_gain() -> None:
    block = powers.patches_block(PATCHES, np.array([1.0, 0.0, 1.8], np.float32), 48.0)
    assert block[2] == {"i": 2, "x": 160.0, "y": 160.0, "gain": 1.8, "sigma": 48.0}
    streamrec.check_envs({"patches": block})


# --- the since card -----------------------------------------------------------------------------------------


def row(step: int, alive: int = 1000, **over: int) -> dict[str, int]:
    return {
        "step": step,
        "alive": alive,
        "starved": 1,
        "killed": 2,
        "born": 3,
        "storm": 4,
        "catastrophe": 5,
        **over,
    }


GRID = [row(s, alive=1000 + s) for s in range(0, 13_001, 250)]


def test_the_compact_row_reads_the_observers_count_else_the_mean() -> None:
    full = {"step": 250, "n_alive": 99.6, "observers": {"alive": 97}, "deaths_starved": 5, "births": 7}
    assert powers.compact_row(full) == {
        "step": 250, "alive": 97, "starved": 5, "killed": 0, "born": 7, "storm": 0, "catastrophe": 0,
    }  # fmt: skip
    assert powers.compact_row({"step": 10, "n_alive": 24.4})["alive"] == 24


def test_milestones_are_plus_250_plus_1000_and_the_end_of_a_timed_effect() -> None:
    f = fwd(466_013, "food", duration=6000)
    assert powers.milestones(f, None) == [466_263, 467_013, 472_013]
    assert powers.milestones(f, 468_000) == [466_263, 467_013, 468_000]  # cancelled early
    assert powers.milestones(fwd(100, "strike"), None) == [350, 1100]
    assert powers.milestones(fwd(100, "blight", duration=3000), None) == [350, 1100, 3100]


def test_a_card_has_the_row_before_the_power_and_the_first_row_at_each_milestone() -> None:
    f = fwd(1013, "food", duration=3000)  # off the grid: the row before is 1000
    rows, reached = powers.since_rows(f, None, GRID, 1000)
    assert rows == [GRID[4]] and reached == 0  # nothing is reached before +250
    rows, reached = powers.since_rows(f, None, GRID, 1250)
    # 1263 is not reached at 1250, but the row after the power is there: its interval holds the application
    assert [r["step"] for r in rows] == [1000, 1250] and reached == 0
    rows, reached = powers.since_rows(f, None, GRID, 1500)
    assert [r["step"] for r in rows] == [1000, 1250, 1500] and reached == 1
    rows, reached = powers.since_rows(f, None, GRID, 2250)
    assert [r["step"] for r in rows] == [1000, 1250, 1500, 2250] and reached == 2  # +1000 = 2013 -> row 2250
    rows, reached = powers.since_rows(f, None, GRID, 4250)
    assert [r["step"] for r in rows] == [
        1000,
        1250,
        1500,
        2250,
        4250,
    ] and reached == 3  # the end: 4013 -> 4250


def test_a_power_on_the_grid_takes_the_row_at_its_own_step_as_before() -> None:
    """The row at ``s`` is already written when a power is applied at ``s`` (rule P puts it after the row and
    the checkpoint): it is the "before" row, never a row that holds part of the effect."""
    f = fwd(2000, "strike")
    rows, reached = powers.since_rows(f, None, GRID, 2250)
    assert [r["step"] for r in rows] == [2000, 2250] and reached == 1


def test_milestones_that_share_a_row_give_one_row_and_count_both() -> None:
    f = fwd(1000, "food", duration=1250)
    coarse = [row(1000), row(2500), row(3000)]
    rows, reached = powers.since_rows(f, None, coarse, 3000)
    assert [r["step"] for r in rows] == [1000, 2500] and reached == 3


def test_a_cancelled_effect_ends_its_card_at_the_cancellation() -> None:
    f = fwd(1000, "food", duration=6000)
    rows, reached = powers.since_rows(f, 1100, GRID, 1250)
    assert [r["step"] for r in rows] == [
        1000,
        1250,
    ] and reached == 2  # +250 and the cancel at 1100 -> row 1250
    assert reached < len(powers.milestones(f, 1100))  # +1000 is still to come


def test_no_row_after_now_is_used_and_a_missing_before_row_is_left_out() -> None:
    f = fwd(0, "food", duration=3000)
    rows, reached = powers.since_rows(f, None, [row(10), row(20)], 20)
    assert [r["step"] for r in rows] == [10] and reached == 0  # a power at step 0: no row at or before it
    rows, reached = powers.since_rows(f, None, [row(0), row(300), row(5000)], 300)
    assert [r["step"] for r in rows] == [0, 300] and reached == 1


def test_a_card_is_rebuilt_from_an_entry_with_the_values_the_player_would_have_sent() -> None:
    assert powers.raw_of(fwd(10, "food")) == {"mult": 1.3, "steps": 6000}
    assert powers.raw_of(fwd(10, "bloom")) == {"patch": 1, "gain": 1.75, "steps": 6000}
    assert powers.raw_of(fwd(10, "blight")) == {"patch": 2, "steps": 6000}
    assert powers.raw_of(fwd(10, "strike")) == {"x": 10.0, "y": 20.0, "radius": 12.5, "kill": 0.45}


# --- the column ---------------------------------------------------------------------------------------------


def test_a_strikes_column_is_the_catastrophes_own_test_on_the_torus() -> None:
    pos = np.array([[10, 10, 5], [10, 10, 900], [315, 10, 3], [40, 10, 3], [10, 60, 3]], np.float32)
    alive = np.array([True, True, True, True, True])
    assert (
        powers.column_count(pos, alive, [10.0, 10.0], 20.0, 320.0) == 3
    )  # any height; (315,10) across the seam
    assert (
        powers.column_count(pos, alive, [10.0, 10.0], 30.0, 320.0) == 4
    )  # (40, 10) is exactly 30 away: inside
    assert powers.column_count(pos, alive, [10.0, 10.0], 29.9, 320.0) == 3
    assert powers.column_count(pos, np.array([True, False, True, True, True]), [10.0, 10.0], 20.0, 320.0) == 2
    assert powers.column_count(pos[:0], alive[:0], [0.0, 0.0], 5.0, 320.0) == 0


# --- the words ----------------------------------------------------------------------------------------------


def test_the_precedent_of_a_pulse_down_is_the_damper_runs_table_row() -> None:
    row_ = TABLE[("regen_pulse", "down")]
    assert (row_.n, row_.factor, row_.ln_change) == (14, 0.8, -0.136)
    assert (
        direction("regen_pulse", {"mult": 0.8}) == "down" and direction("regen_pulse", {"mult": 1.3}) == "up"
    )
    assert direction("regen_pulse", {"mult": 1.0}) == "up"  # "down" is strictly below 1
    assert direction("fertility_shift", {"patch": 0}) == "any"
    p = precedent_object("regen_pulse", {"mult": 0.8}, "Food")
    assert p["line"] == (
        "Earlier x0.8 pulses (14 in the 3D director runs): the population fell 13% (12% to 14%) within about "
        "1,000 steps and rose again within about 1,000 steps of the end."
    )
    assert (p["version"], p["n"], p["key"]) == (PRECEDENT_VERSION, 14, "regen_pulse:down")
    assert (p["step"], p["rows"]) == (None, [])
    assert math.exp(-0.136) == pytest.approx(0.8728, abs=1e-4)  # 13%, and the interval's 12% and 14%


def test_a_direction_with_no_row_says_no_earlier_case_and_never_shows_the_other_direction() -> None:
    up = precedent_object("regen_pulse", {"mult": 1.3}, "Food")
    assert up["line"] == "No earlier case for x1.3." and up["n"] == 0 and up["source"] is None
    assert "x0.8" not in up["line"]
    assert precedent_object("fertility_shift", {"patch": 1, "gain": 1.7}, "Bloom")["line"] == (
        "No earlier case for Bloom."
    )
    for obj in (
        up,
        precedent_object("regen_pulse", {"mult": 0.6}, "Food"),
        precedent_object("catastrophe", {}, "Strike"),
    ):
        clean(obj["line"])


def test_the_table_row_is_pinned() -> None:
    """A change of a number is a change of the table's version, on purpose."""
    assert dataclasses.astuple(TABLE[("regen_pulse", "down")]) == (
        "regen_pulse", "down", 0.8, 14, -0.136, -0.150, -0.124, 1000, 1000, "RESULTS_3D_RUNS section 5.4",
    )  # fmt: skip
    assert PRECEDENT_VERSION == 1


def test_the_powers_module_is_versioned_and_the_intent_refusal_is_the_specs() -> None:
    assert powers.POWERS_VERSION == 1 and powers.INTENT_REFUSAL == "intent is not available in this build"
    assert actions.GAME_SPEC_VERSION == "game-v0" and pacing.PLAYER_BUDGET_WINDOW == 18_000
