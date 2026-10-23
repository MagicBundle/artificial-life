"""The player channel's tables, validation, pacing scopes and log rules (G3b spec §6.1; DESIGN §26.8, §34.5).

No engine run here (``test_player_run.py`` has those): the game table against ``MENU``, ``clamp`` and
``validate_entry`` / ``validate_log`` for the player's records, ``pacing.fold`` by scope, the identity of the
research folds over the three research logs of the evaluation, the header key and rule P in
``RunDir.truncate_after``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import director_identity as di
from alife import runio
from alife.config import Config, apply_overrides, config_from_dict, config_to_dict
from alife.director import actions, log, pacing
from alife.director.actions import GAME_SPECS, MENU, ActionError, EnvView, clamp, env_view, validate_entry
from alife.director.log import LogError
from alife.runio import COMMANDS_FILE, DIRECTOR_LOG_FILE, JOURNAL_FILE, RunDir
from alife.state import State, init_state

# The folds of the three research logs before the player channel existed (computed on a8956b9 by
# ``director_identity.summary``); ``fold``, ``due_reverts``, the budget and the cooldowns must still give
# them.
RESEARCH_GOLDEN = {
    "eval_s0_C": "056b735fa627e3701ccd7a176dd1bef7797b1c106eeac40a80603e0706d97538",
    "eval_s1_C": "3da30179f5881080499465fa665ba89e64d1a3e3935df6240190124b9f3bb852",
    "eval_s0_RC": "967c609d2cbdd52935385bdb6fcaf6cf381bf7ced514cb5a3dbb7a61946b024c",
}
HEADER: dict[str, Any] = {"kind": "header", "step": 0, "rules_version": 2, "game_specs": "game-v0"}


@pytest.fixture(scope="module")
def dcfg(tiny_cfg: Config) -> Config:
    return apply_overrides(tiny_cfg, ["director.enabled=true", "director.rule_every=100"])


@pytest.fixture(scope="module")
def s0(dcfg: Config) -> State:
    return init_state(dcfg)


@pytest.fixture(scope="module")
def view(s0: State) -> EnvView:
    return env_view(s0)


def with_size(cfg: Config, size: float) -> Config:
    d = config_to_dict(cfg)
    d["world"] = {**d["world"], "size": size}
    return config_from_dict(d)


def power(
    cfg: Config,
    view: EnvView,
    name: str,
    raw: dict[str, Any],
    duration: int | None = None,
    step: int = 7_013,
    command: int = 1,
) -> dict[str, Any]:
    """The record of the player channel for ``name`` as ``Director`` makes it (game table clamp)."""
    action = actions.GAME_POWERS[name]
    c = clamp(action, raw, duration, view, cfg, specs=GAME_SPECS)
    return {
        "step_applied": step,
        "source": "player",
        "persona": "gardener",
        "action": action,
        "params": c.params,
        "prev": c.prev,
        "duration": c.duration,
        "reason": f"the player used {name}",
        "vitals_step": 7_000,
        "vitals": {"n_alive": 100.0},
        "scope": "player",
        "command": command,
        "power": name,
        "specs": "game-v0",
        "cost": GAME_SPECS[action].cost,
    }


def revert_of(e: dict[str, Any], step: int, scope: str | None = "player") -> dict[str, Any]:
    return {
        "step_applied": step,
        "source": "revert",
        "persona": "gardener",
        "action": e["action"],
        "params": dict(e["prev"]),
        "prev": {},
        "duration": None,
        "reason": "end of the effect",
        "reverts": e["step_applied"],
        **({"scope": scope} if scope else {}),
    }


# --- the tables ---


def test_menu_is_byte_identical_to_the_research_table() -> None:
    """The game table is an override of ``MENU``, not an edit of it (GAMES §5.5): ``MENU`` is what it was."""
    assert {k: (v.cost, v.cooldown, dict(v.bounds), v.duration) for k, v in MENU.items()} == {
        "set_seasons": (1, 50_000, {"period": (2_000, 100_000), "amp": (0.0, 0.9)}, None),
        "fertility_shift": (1, 20_000, {"gain": (0.25, 2.0)}, None),
        "habitat_loss": (2, 50_000, {}, (10_000, 200_000)),
        "regen_pulse": (1, 30_000, {"mult": (0.4, 1.8)}, (5_000, 50_000)),
        "immigration_pulse": (1, 20_000, {"count": (16, 1024)}, None),
        "catastrophe": (3, 100_000, {"kill_p": (0.2, 0.9)}, None),
        "drift_change": (1, 50_000, {"mult": (0.5, 10.0)}, None),
    }
    assert all(v.quantum is None and not v.excluded for v in MENU.values())  # the new fields are off
    assert set(actions.TIMED) == {
        "habitat_loss",
        "regen_pulse",
    }  # by name, as before; the fold uses the entry
    assert actions.SOURCES == ("rule", "llm", "revert") and actions.GAME_SOURCES[-1] == "player"


def test_the_game_table_overrides_the_menu_row_by_row() -> None:
    """Every difference from ``MENU`` is listed: durations below ``MENU``'s, a quantum of 250, Bloom timed,
    Strike at 1, the cooldowns, and the dead zone of Food."""
    assert set(GAME_SPECS) == {"regen_pulse", "fertility_shift", "habitat_loss", "catastrophe"}
    assert actions.GAME_POWERS == {
        "food": "regen_pulse",
        "bloom": "fertility_shift",
        "blight": "habitat_loss",
        "strike": "catastrophe",
    }
    assert {k: (v.cost, v.cooldown, v.duration, v.quantum) for k, v in GAME_SPECS.items()} == {
        "regen_pulse": (1, 3_000, (3_000, 12_000), 250),
        "fertility_shift": (1, 3_000, (3_000, 12_000), 250),
        "habitat_loss": (2, 6_000, (3_000, 12_000), 250),
        "catastrophe": (1, 3_000, None, None),
    }
    assert dict(GAME_SPECS["regen_pulse"].bounds) == {"mult": (0.6, 1.5)}
    assert dict(GAME_SPECS["regen_pulse"].excluded) == {"mult": (0.9, 1.15)}
    assert dict(GAME_SPECS["fertility_shift"].bounds) == {"gain": (1.5, 2.0)}
    assert dict(GAME_SPECS["catastrophe"].bounds) == {"kill_p": (0.3, 0.6), "radius": (20.0, 80.0)}
    assert (actions.PLAYER_BUDGET, actions.PLAYER_BUDGET_WINDOW, actions.PLAYER_QUANTUM) == (6, 18_000, 250)
    for name in ("regen_pulse", "habitat_loss"):  # below MENU's lower bounds: not a subset
        assert GAME_SPECS[name].duration[0] < MENU[name].duration[0]  # type: ignore[index]
    assert actions.GAME_SPEC_VERSION == "game-v0"


def test_the_strike_radius_is_the_research_range_scaled_by_the_width(
    tiny_cfg: Config, tiny3d_cfg: Config
) -> None:
    assert actions.radius_bounds(with_size(tiny3d_cfg, 320.0), GAME_SPECS) == (6.25, 25.0)  # the planet's
    assert actions.radius_bounds(tiny3d_cfg, GAME_SPECS) == (2.5, 10.0)  # L = 128
    assert actions.radius_bounds(with_size(tiny_cfg, 1024.0), GAME_SPECS) == (20.0, 80.0)  # 2D, as written
    assert actions.radius_bounds(tiny3d_cfg) == (2.5, 25.0)  # MENU's range, unchanged
    assert actions.radius_bounds(with_size(tiny3d_cfg, 320.0)) == (6.25, 62.5)


def test_is_player_and_is_timed() -> None:
    assert actions.is_player({"source": "player"})
    assert actions.is_player({"source": "revert", "scope": "player"})
    assert not actions.is_player({"source": "revert"}) and not actions.is_player(
        {"source": "revert", "scope": "x"}
    )
    assert not actions.is_player({"source": "rule"}) and not actions.is_player(
        {"source": "llm", "scope": "player"}
    )
    assert actions.is_timed({"duration": 3000}) and not actions.is_timed({"duration": None})
    assert not actions.is_timed({"action": "regen_pulse"})  # by the entry, not the name
    # the runio copy (it cannot import the director) agrees on every shape
    for rec in (
        {"source": "player"},
        {"source": "revert", "scope": "player"},
        {"source": "revert", "scope": "rule"},
        {"source": "revert"},
        {"source": "rule", "scope": "player"},
        {"source": "llm"},
        {},
    ):
        assert runio._is_player_record(rec) == actions.is_player(rec), rec


# --- clamp with the game table ---


@pytest.mark.parametrize(
    ("raw", "duration", "mult", "dur"),
    [
        ({"mult": 1.3}, 6_000, actions._f32(1.3), 6_000),
        ({"mult": 9}, 12_000, actions._f32(1.5), 12_000),
        ({"mult": 0.0}, 3_000, actions._f32(0.6), 3_000),
        ({"mult": 0.9}, 4_100, actions._f32(0.9), 4_250),  # the edge of the dead zone is allowed
        ({"mult": 1.15}, 1, actions._f32(1.15), 3_000),
        ({"mult": 1.2}, 10**9, actions._f32(1.2), 12_000),
    ],
)
def test_food_is_clamped_to_the_game_range_and_rounded_up_to_250(
    dcfg: Config, view: EnvView, raw: dict[str, Any], duration: int, mult: float, dur: int
) -> None:
    c = clamp("regen_pulse", raw, duration, view, dcfg, specs=GAME_SPECS)
    assert (c.params, c.duration) == ({"mult": mult}, dur)
    assert c.prev == {"mult": 1.0} and dur % 250 == 0


@pytest.mark.parametrize("mult", [0.91, 1.0, 1.1, 1.149])
def test_food_inside_the_dead_zone_is_refused_not_clamped(dcfg: Config, view: EnvView, mult: float) -> None:
    with pytest.raises(ActionError, match="refused"):
        clamp("regen_pulse", {"mult": mult}, 6_000, view, dcfg, specs=GAME_SPECS)
    clamp("regen_pulse", {"mult": mult}, 6_000, view, dcfg)  # MENU: no dead zone


def test_bloom_is_a_timed_gain_and_has_no_move_form(dcfg: Config, view: EnvView) -> None:
    c = clamp("fertility_shift", {"patch": 2, "gain": 1.0}, 5_100, view, dcfg, specs=GAME_SPECS)
    assert c.params == {"patch": 2, "gain": 1.5} and c.duration == 5_250
    assert c.prev == {"patch": 2, "gain": 1.0}
    assert (
        clamp("fertility_shift", {"patch": 2, "gain": 9}, 1, view, dcfg, specs=GAME_SPECS).params["gain"]
        == 2.0
    )
    for raw, duration in (({"patch": 2, "dx": 3.0, "dy": 4.0}, 5_000), ({"patch": 2, "gain": 1.7}, None)):
        with pytest.raises(ActionError):
            clamp("fertility_shift", raw, duration, view, dcfg, specs=GAME_SPECS)
    # the research form is untouched: a permanent gain (no duration) and the move
    m = clamp("fertility_shift", {"patch": 2, "gain": 1.7}, None, view, dcfg)
    assert m.duration is None and m.params == {"patch": 2, "gain": actions._f32(1.7)}
    assert set(clamp("fertility_shift", {"patch": 0, "dx": 3.0, "dy": 4.0}, None, view, dcfg).params) == {
        "patch",
        "x",
        "y",
    }


def test_blight_and_strike_with_the_game_table(tiny3d_cfg: Config) -> None:
    enabled = apply_overrides(tiny3d_cfg, ["director.enabled=true"])
    cfg, v = with_size(enabled, 320.0), env_view(init_state(enabled))
    b = clamp("habitat_loss", {"patch": 1}, 100, v, cfg, specs=GAME_SPECS)
    assert (b.params, b.duration) == ({"patch": 1, "gain": 0.0}, 3_000)
    lo = clamp("catastrophe", {"x": 5, "y": 6, "radius": 1, "kill_p": 0}, 99, v, cfg, specs=GAME_SPECS)
    hi = clamp("catastrophe", {"x": 5, "y": 6, "radius": 99, "kill_p": 1}, None, v, cfg, specs=GAME_SPECS)
    assert (lo.params["radius"], lo.params["kill_p"]) == (6.25, actions._f32(0.3))
    assert (hi.params["radius"], hi.params["kill_p"]) == (25.0, actions._f32(0.6))
    assert lo.duration is None and hi.duration is None  # one step: no duration, whatever was asked


@pytest.mark.parametrize("action", ["set_seasons", "immigration_pulse", "drift_change", "nonsense"])
def test_the_game_table_has_no_other_actions(dcfg: Config, view: EnvView, action: str) -> None:
    with pytest.raises(ActionError, match="unknown action"):
        clamp(
            action, {"period": 5000, "amp": 0.3, "count": 20, "mult": 2.0}, None, view, dcfg, specs=GAME_SPECS
        )


# --- validate_entry for the player's records ---


def all_powers(cfg: Config, view: EnvView) -> list[dict[str, Any]]:
    """Every power at both ends of its range (the lowest and highest value of each parameter and duration)."""
    out = []
    for mult, d in ((0.6, 3_000), (0.9, 3_250), (1.15, 11_999), (1.5, 12_000)):
        out.append(power(cfg, view, "food", {"mult": mult}, d))
    for gain, d in ((1.5, 3_000), (2.0, 12_000)):
        out.append(power(cfg, view, "bloom", {"patch": 3, "gain": gain}, d))
    for p, d in ((0, 3_000), (3, 12_000)):
        out.append(power(cfg, view, "blight", {"patch": p}, d))
    lo, hi = actions.radius_bounds(cfg, GAME_SPECS)
    for r, k, x in ((lo, 0.3, 0.0), (hi, 0.6, cfg.world.size * 0.999), ((lo + hi) / 2, 0.45, 12.5)):
        out.append(power(cfg, view, "strike", {"x": x, "y": 3.0, "radius": r, "kill_p": k}))
    return out


def test_every_power_at_both_ends_of_its_range_passes_validation(dcfg: Config, view: EnvView) -> None:
    for e in all_powers(dcfg, view):
        validate_entry(e, dcfg, dcfg.director.rule_every)  # off the 100-step grid: 7,013
    # and as a log, with each timed one's revert at its own off-grid end
    recs: list[dict[str, Any]] = [HEADER]
    for i, e in enumerate(all_powers(dcfg, view)):
        e = {**e, "step_applied": 7_013 + 20_000 * i, "command": i}
        recs.append(e)
        if e["duration"]:
            recs.append(revert_of(e, e["step_applied"] + e["duration"]))
    recs.sort(key=lambda r: r.get("step_applied", 0))
    log.validate_log(recs, dcfg)


def test_every_power_at_both_ends_on_the_planet_world(tiny3d_cfg: Config) -> None:
    enabled = apply_overrides(tiny3d_cfg, ["director.enabled=true", "director.rule_every=100"])
    cfg, v = with_size(enabled, 320.0), env_view(init_state(enabled))
    for e in all_powers(cfg, v):
        validate_entry(e, cfg, 100)


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"params": {"mult": 1.0}}, "refused"),  # the dead zone
        ({"params": {"mult": 1.6}}, "outside"),
        ({"params": {"mult": 0.5}}, "outside"),
        ({"duration": 2_750}, "outside"),
        ({"duration": 12_250}, "outside"),
        ({"duration": 3_100}, "multiple"),
        ({"duration": None}, "must be a finite"),
        ({"scope": None}, "scope"),
        ({"scope": "rule"}, "scope"),
        ({"power": "bloom"}, "is not a power of"),
        ({"power": None}, "is not a power of"),
        ({"command": None}, "request id"),
        ({"command": -1}, "non-negative"),
        ({"specs": "game-v1"}, "specs"),
        ({"cost": 2}, "cost"),
        ({"step_applied": -1}, "non-negative"),
        ({"step_applied": 7.5}, "non-negative"),
    ],
)
def test_a_malformed_player_record_is_refused(
    dcfg: Config, view: EnvView, change: dict[str, Any], match: str
) -> None:
    e = {**power(dcfg, view, "food", {"mult": 1.3}, 6_000), **change}
    with pytest.raises(ActionError, match=match):
        validate_entry(e, dcfg, 100)


def test_player_steps_are_any_non_negative_integer_the_rest_stay_on_the_grid(
    dcfg: Config, view: EnvView
) -> None:
    for step in (0, 1, 99, 100, 7_013):
        validate_entry(power(dcfg, view, "food", {"mult": 1.3}, 6_000, step=step), dcfg, 100)
    rule = {
        "step_applied": 100,
        "source": "rule",
        "action": "regen_pulse",
        "params": {"mult": 1.3},
        "prev": {"mult": 1.0},
        "duration": 5_000,
    }
    validate_entry(rule, dcfg, 100)
    for step in (0, 50, 1_234):
        with pytest.raises(ActionError, match="positive multiple of rule_every"):
            validate_entry({**rule, "step_applied": step}, dcfg, 100)
    with pytest.raises(ActionError, match="scope"):  # only the player's records carry a scope
        validate_entry({**rule, "scope": "rule"}, dcfg, 100)
    with pytest.raises(ActionError, match="outside"):  # a rule entry meets MENU, not the game table
        validate_entry({**rule, "params": {"mult": 0.5}, "duration": 3_000}, dcfg, 100)


def test_a_player_revert_is_checked_against_the_game_table(dcfg: Config, view: EnvView) -> None:
    for name, raw in (
        ("food", {"mult": 1.3}),
        ("bloom", {"patch": 1, "gain": 1.8}),
        ("blight", {"patch": 0}),
    ):
        e = power(dcfg, view, name, raw, 3_000)
        validate_entry(revert_of(e, 10_013), dcfg, 100)
        with pytest.raises(ActionError, match="scope"):  # a scope other than the player's is not a player's
            validate_entry({**revert_of(e, 10_000), "scope": "rule"}, dcfg, 100)
    strike = power(dcfg, view, "strike", {"x": 1.0, "y": 2.0, "radius": 20.0, "kill_p": 0.5})
    with pytest.raises(ActionError, match="no timed effect"):
        validate_entry(revert_of(strike, 7_100), dcfg, 100)
    bloom = power(dcfg, view, "bloom", {"patch": 1, "gain": 1.8}, 3_000)
    with pytest.raises(ActionError, match="exactly"):  # Bloom's revert restores a patch and a gain, no mult
        validate_entry({**revert_of(bloom, 10_013), "params": {"mult": 1.0}}, dcfg, 100)
    # a research revert of fertility_shift stays impossible: MENU's gain is permanent, nothing to revert
    with pytest.raises(ActionError, match="no timed effect"):
        validate_entry(revert_of(bloom, 10_000, scope=None), dcfg, 100)


# --- validate_log ---


def test_a_log_needs_the_game_specs_header_for_the_players_records(dcfg: Config, view: EnvView) -> None:
    e = power(dcfg, view, "food", {"mult": 1.3}, 3_000)
    log.validate_log([HEADER, e, revert_of(e, e["step_applied"] + 3_000)], dcfg)
    plain = {k: v for k, v in HEADER.items() if k != "game_specs"}
    with pytest.raises(LogError, match="no header names game_specs"):
        log.validate_log([plain, e], dcfg)
    with pytest.raises(LogError, match="no header names game_specs"):
        log.validate_log([e], dcfg)
    with pytest.raises(LogError, match="game_specs"):
        log.validate_log([{**HEADER, "game_specs": "game-v9"}, e], dcfg)
    # the header in force is the latest one before the record
    with pytest.raises(LogError, match="no header names game_specs"):
        log.validate_log([HEADER, plain, e], dcfg)
    log.validate_log([plain, HEADER, e], dcfg)


def test_the_log_checks_the_player_reverts_and_the_step_order(dcfg: Config, view: EnvView) -> None:
    e = power(dcfg, view, "food", {"mult": 1.3}, 3_000)
    r = revert_of(e, e["step_applied"] + 3_000)
    with pytest.raises(LogError, match="no active effect"):
        log.validate_log([HEADER, r], dcfg)
    with pytest.raises(LogError, match="comes after"):
        log.validate_log([HEADER, e, r, {**e, "step_applied": 8_000, "command": 2}], dcfg)
    log.validate_log([HEADER, e, r, {**e, "step_applied": r["step_applied"] + 1, "command": 2}], dcfg)
    # the same step in a boundary entry then a player's is the order the loop writes (end of t-1, top of t)
    rule = {
        "step_applied": 7_100,
        "source": "rule",
        "action": "drift_change",
        "params": {"mult": 2.0},
        "prev": {"mult": 1.0},
        "duration": None,
    }
    log.validate_log([HEADER, rule, power(dcfg, view, "food", {"mult": 1.3}, 3_000, step=7_100)], dcfg)


def test_a_timed_bloom_is_an_active_effect_in_the_log_but_a_research_gain_is_not(
    dcfg: Config, view: EnvView
) -> None:
    bloom = power(dcfg, view, "bloom", {"patch": 1, "gain": 1.8}, 3_000)
    again = power(dcfg, view, "bloom", {"patch": 1, "gain": 1.6}, 3_000, step=8_000, command=2)
    log.validate_log([HEADER, bloom, revert_of(bloom, 10_013)], dcfg)
    log.validate_log([HEADER, bloom, again, revert_of(again, 11_000)], dcfg)  # a newer one supersedes
    permanent = {  # the research form: a gain with no duration holds no lever
        "step_applied": 7_100,
        "source": "llm",
        "action": "fertility_shift",
        "params": {"patch": 1, "gain": 1.8},
        "prev": {"patch": 1, "gain": 1.0},
        "duration": None,
    }
    log.validate_log([{**HEADER, "game_specs": None}, permanent], dcfg)
    with pytest.raises(ActionError, match="no timed effect"):  # MENU's gain is permanent: nothing to revert
        log.validate_log(
            [{**HEADER, "game_specs": None}, permanent, revert_of(permanent, 7_200, scope=None)], dcfg
        )


def test_the_header_key_is_only_written_when_asked_and_a_resumed_research_log_gets_no_header(
    dcfg: Config,
) -> None:
    plain = log.make_header(dcfg, 0, 6)
    assert "game_specs" not in plain
    played = log.make_header(dcfg, 0, 6, game_specs=True)
    assert (
        played["game_specs"] == "game-v0" and {k: v for k, v in played.items() if k != "game_specs"} == plain
    )
    assert not log.needs_header([plain], log.make_header(dcfg, 500, 6))  # a research resume: nothing appended
    assert log.needs_header([plain], played) and log.needs_header([played], plain)
    assert not log.needs_header([played], log.make_header(dcfg, 500, 6, game_specs=True))


# --- pacing scopes ---


def fe(step: int, action: str, source: str = "rule", **kw: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "step_applied": step,
        "source": source,
        "action": action,
        "params": {},
        "prev": {},
    }
    if source == "player":
        base |= {"scope": "player", "cost": GAME_SPECS[action].cost}
    return {**base, **kw}


def player_food(step: int, mult: float = 1.3, duration: int = 3_000) -> dict[str, Any]:
    return fe(step, "regen_pulse", "player", params={"mult": mult}, prev={"mult": 1.0}, duration=duration)


def rule_food(step: int, mult: float = 0.8, duration: int = 10_000) -> dict[str, Any]:
    return fe(
        step,
        "regen_pulse",
        "rule",
        rule="runaway_damper",
        params={"mult": mult},
        prev={"mult": 1.0},
        duration=duration,
    )


def test_the_rule_scope_ignores_the_player_and_the_player_scope_ignores_the_rest() -> None:
    entries = [
        rule_food(1_000),
        player_food(2_013, duration=3_000),
        fe(3_000, "catastrophe", "player", params={"center": [1, 2], "radius": 9, "kill_p": 0.4}),
        fe(4_000, "drift_change", "rule", rule="stagnation_response", params={"mult": 2.0}),
    ]
    allp, rule, mine = (pacing.fold(entries, 20_000, s) for s in (None, "rule", "player"))
    assert [c for _, c in allp.spent] == [1, 1, 1, 1] and allp.last_step == 4_000
    assert rule.spent == ((1_000, 1), (4_000, 1)) and rule.last_action == "drift_change"
    assert mine.spent == ((2_013, 1), (3_000, 1)) and mine.last_action == "catastrophe"
    assert rule.ends == {"regen_pulse": 11_000, "drift_change": 4_000}
    assert mine.ends == {"regen_pulse": 5_013, "catastrophe": 3_000}
    assert rule.last_rule == {"runaway_damper": 1_000, "stagnation_response": 4_000}
    assert mine.last_rule == {} and set(allp.last_rule) == set(rule.last_rule)
    # the player's Food superseded the rule's on the one lever: shared, whatever the scope
    assert {s: pacing.fold(entries, 3_500, s).active["regen"].origin for s in (None, "rule", "player")} == {
        None: 2_013,
        "rule": 2_013,
        "player": 2_013,
    }
    assert pacing.drama(rule, 5_000, 20_000) < pacing.drama(allp, 5_000, 20_000)
    with pytest.raises(ValueError, match="scope"):
        pacing.fold(entries, 1, "nobody")


def test_cost_is_the_entrys_own_and_drops_to_the_menu_only_without_one() -> None:
    free = fe(1_000, "catastrophe", "rule", rule="x", params={"center": [1, 2], "radius": 9, "kill_p": 0.4})
    assert pacing.fold([free], 2_000).spent == ((1_000, 3),)  # MENU's cost
    assert pacing.fold([{**free, "cost": 1}], 2_000).spent == ((1_000, 1),)
    assert pacing.fold([fe(1_000, "catastrophe", "player", params={})], 2_000).spent == ((1_000, 1),)


def test_due_reverts_by_scope_and_a_cancel_removes_the_lever() -> None:
    entries = [
        player_food(1_013, duration=3_000),
        fe(
            1_500,
            "habitat_loss",
            "rule",
            rule="r",
            params={"patch": 2, "gain": 0.0},
            prev={"patch": 2, "gain": 1.0},
            duration=10_000,
        ),
    ]
    p = pacing.fold(entries, 20_000)
    assert [e.key for e in pacing.due_reverts(p, 20_000)] == ["regen", "patch:2"]
    assert [e.key for e in pacing.due_reverts(p, 20_000, "player")] == ["regen"]
    assert [e.key for e in pacing.due_reverts(p, 20_000, "rule")] == ["patch:2"]
    assert pacing.due_reverts(p, 4_012, "player") == [] and len(pacing.due_reverts(p, 4_013, "player")) == 1
    assert {e.scope for e in p.active.values()} == {"player", "rule"}
    cancelled = [*entries, {**revert_of(entries[0], 2_000), "action": "regen_pulse", "params": {"mult": 1.0}}]
    assert "regen" not in pacing.fold(cancelled, 2_000).active
    assert pacing.fold(cancelled, 2_000).ends == pacing.fold(entries, 2_000).ends  # the planned end stands


def six_units() -> list[dict[str, Any]]:
    """Six units inside one window: Food 1, Strike 1, Blight 2 and two Blooms."""
    return [
        player_food(1_000, 1.3, 3_000),
        fe(1_100, "catastrophe", "player", params={}),
        fe(
            4_000,
            "habitat_loss",
            "player",
            params={"patch": 1, "gain": 0.0},
            duration=3_000,
            prev={"patch": 1, "gain": 1.0},
        ),
        fe(
            4_100,
            "fertility_shift",
            "player",
            params={"patch": 2, "gain": 1.7},
            duration=3_000,
            prev={"patch": 2, "gain": 1.0},
        ),
        fe(
            4_200,
            "fertility_shift",
            "player",
            params={"patch": 3, "gain": 1.7},
            duration=3_000,
            prev={"patch": 3, "gain": 1.0},
        ),
    ]


def test_the_player_budget_is_a_rolling_18000_and_blocked_says_why() -> None:
    entries = six_units()[:2]
    mine = pacing.fold(entries, 5_000, "player")
    assert pacing.budget_used(mine, 5_000, 18_000) == 2
    assert pacing.budget_used(mine, 19_000, 18_000) == 1 and pacing.budget_used(mine, 19_100, 18_000) == 0
    assert pacing.budget_used(mine, 5_000) == 2  # the default window is the rule tier's
    shared = pacing.fold(entries, 5_000)
    assert pacing.player_blocked(mine, shared, "habitat_loss", {"patch": 1}, 5_000) is None
    cooling = pacing.player_blocked(mine, shared, "regen_pulse", {"mult": 1.3}, 5_999)
    assert (
        cooling == "regen_pulse is cooling down until step 7000"
    )  # from the planned end: 1,000 + 3,000 + 3,000
    ended = [*entries, revert_of(entries[0], 4_000)]  # the lever is free once the revert is logged
    mine, shared = pacing.fold(ended, 7_000, "player"), pacing.fold(ended, 7_000)
    assert pacing.player_blocked(mine, shared, "regen_pulse", {"mult": 1.3}, 7_000) is None
    used = pacing.fold(six_units(), 9_000, "player")
    assert pacing.budget_used(used, 9_000, 18_000) == 6
    why = pacing.player_blocked(used, pacing.fold(six_units(), 9_000), "catastrophe", {}, 9_000)
    assert why == "influence: 6 of 6 units are in use"
    held = pacing.fold([rule_food(1_000)], 2_000)  # a director's pulse holds the lever, whoever asks
    assert pacing.player_blocked(
        pacing.fold([], 2_000, "player"), held, "regen_pulse", {"mult": 1.3}, 2_000
    ) == ("regen already has an active effect")


# --- the research folds are unchanged ---


@pytest.mark.parametrize("name", di.LOGS)
def test_the_research_logs_fold_validate_and_revert_as_before(name: str) -> None:
    """``eval_s0_C``, ``eval_s1_C`` and ``eval_s0_RC`` hold permanent ``fertility_shift`` gains (4, 1 and 3
    entries with a null duration): timed by the entry, not by the action's name, they stay permanent."""
    records, cfg = di.load(name)
    entries = [r for r in records if r.get("kind") != "header"]
    assert any(e["action"] == "fertility_shift" and e["duration"] is None for e in entries)
    assert di.validates(records, cfg)
    assert di.summary(entries) == RESEARCH_GOLDEN[name]  # the unscoped fold
    rule = di.summary(
        entries, lambda es, t: pacing.fold(es, t, "rule"), lambda p, t: pacing.due_reverts(p, t, "rule")
    )
    assert rule == RESEARCH_GOLDEN[name]  # a research log has no player: the rule scope is the whole log
    assert not any(actions.is_player(e) for e in entries)
    assert all("game_specs" not in r for r in records)


def test_a_timed_bloom_holds_its_patch_in_the_fold() -> None:
    """Timed is the entry's property: the player's Bloom (a gain with a duration) is an active effect on its
    patch and ends with its revert, where the research gain of the same action holds nothing."""
    bloom = fe(
        1_013,
        "fertility_shift",
        "player",
        params={"patch": 1, "gain": 1.8},
        prev={"patch": 1, "gain": 1.0},
        duration=3_000,
    )
    p = pacing.fold([bloom], 2_000)
    assert sorted(p.active) == ["patch:1"] and (p.active["patch:1"].end, p.active["patch:1"].scope) == (
        4_013,
        "player",
    )
    assert p.active["patch:1"].restore == {"patch": 1, "gain": 1.0}
    assert pacing.fold([bloom, revert_of(bloom, 4_013)], 5_000).active == {}


def test_a_permanent_gain_holds_no_lever_in_the_fold() -> None:
    gain = fe(
        1_000,
        "fertility_shift",
        "llm",
        params={"patch": 1, "gain": 1.5},
        prev={"patch": 1, "gain": 1.0},
        duration=None,
    )
    assert pacing.fold([gain], 5_000).active == {}


# --- rule P in the log cut ---


def _rule(s: int) -> dict[str, Any]:
    return {"step_applied": s, "source": "rule", "action": "drift_change", "params": {}}


def _mine(s: int) -> dict[str, Any]:
    return {"step_applied": s, "source": "player", "scope": "player", "action": "regen_pulse"}


def _end(s: int, scope: str | None) -> dict[str, Any]:
    return {
        "step_applied": s,
        "source": "revert",
        "action": "regen_pulse",
        **({"scope": scope} if scope else {}),
    }


@pytest.fixture
def played_dir(tmp_path: Path, tiny_cfg: Config) -> RunDir:
    """A run directory whose director log, journal and commands hold records at 99, 100 and 101."""
    cfg = apply_overrides(tiny_cfg, ["director.enabled=true", "director.rule_every=100"])
    rd = RunDir.create(tmp_path / "w", cfg)
    rd.update_meta(start_step=0)
    lines = [{**HEADER, "step": 100}]
    for s in (99, 100, 101):
        lines += [_rule(s), _mine(s), _end(s, "player"), _end(s, None)]
    rd.append_jsonl(DIRECTOR_LOG_FILE, lines)
    rd.append_jsonl(JOURNAL_FILE, [{"step": s, "persona": "gardener", "text": "x"} for s in (99, 100, 101)])
    rd.append_jsonl(COMMANDS_FILE, [{"id": s, "t": s, "line": "power food id=1"} for s in (99, 100, 101)])
    return rd


def kept(rd: RunDir, name: str) -> list[tuple[Any, ...]]:
    """``(step, source, scope)`` of the records of a log, whatever the key the step is under."""
    return [
        (r.get("step_applied", r.get("step", r.get("t"))), r.get("source"), r.get("scope"))
        for r in rd.read_jsonl(name)
        if r.get("kind") != "header"
    ]


@pytest.mark.parametrize("step", [99, 100, 101])
def test_truncate_after_keeps_a_player_record_only_before_the_checkpoint_step(
    played_dir: RunDir, step: int
) -> None:
    """A boundary entry at ``s`` is in the checkpoint at ``s``; a player's at ``s`` is applied after it."""
    rd = played_dir
    rd.truncate_after(step)
    got = kept(rd, DIRECTOR_LOG_FILE)
    assert [g for g in got if g[1] == "rule"] == [(s, "rule", None) for s in (99, 100, 101) if s <= step]
    assert [g for g in got if g[1] == "player"] == [
        (s, "player", "player") for s in (99, 100, 101) if s < step
    ]
    assert [g for g in got if g[1] == "revert" and g[2] == "player"] == [
        (s, "revert", "player") for s in (99, 100, 101) if s < step
    ]
    assert [
        g for g in got if g[1] == "revert" and g[2] is None
    ] == [  # a rule's revert: at or before, as ever
        (s, "revert", None) for s in (99, 100, 101) if s <= step
    ]
    assert [g[0] for g in kept(rd, JOURNAL_FILE)] == [s for s in (99, 100, 101) if s <= step]
    assert [g[0] for g in kept(rd, COMMANDS_FILE)] == [s for s in (99, 100, 101) if s < step]
    has_header = any(r.get("kind") == "header" for r in rd.read_jsonl(DIRECTOR_LOG_FILE))
    assert has_header == (step >= 100)  # the header carries the step it was written at: kept as ever


def test_a_replay_cut_touches_none_of_the_inputs(played_dir: RunDir) -> None:
    before = {n: (played_dir.path / n).read_bytes() for n in (DIRECTOR_LOG_FILE, JOURNAL_FILE, COMMANDS_FILE)}
    played_dir.truncate_after(99, keep_director=True)
    assert before == {n: (played_dir.path / n).read_bytes() for n in before}


def test_truncate_after_without_the_new_files_is_the_old_cut(tmp_path: Path, tiny_cfg: Config) -> None:
    cfg = apply_overrides(tiny_cfg, ["director.enabled=true", "director.rule_every=100"])
    rd = RunDir.create(tmp_path / "r", cfg)
    rd.update_meta(start_step=0)
    rd.append_jsonl(
        DIRECTOR_LOG_FILE,
        [{"kind": "header", "step": 0}, {"step_applied": 100, "source": "rule", "action": "drift_change"}],
    )
    rd.truncate_after(100)
    assert len(rd.read_jsonl(DIRECTOR_LOG_FILE)) == 2 and not (rd.path / COMMANDS_FILE).exists()
