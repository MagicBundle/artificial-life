"""The player's powers in the play loop (G3b spec §6, DESIGN §34.5): ``power`` lines answered with ``PLAN``
records, the queue and the application point, ``commands.jsonl``, the ``ENVS`` additions, the since cards, the
chronicle rows, the ledger with powers, replay equality, rule P at a chapter end and the pacer's refusals.

Worlds are ``tiny3d`` with 24 organisms and a director in the ``player`` mode; the game table is shrunk by an
autouse patch (durations of 30 to 120 steps in tens, cooldowns of 20, a budget window of 400, since cards at
+20 and +50) so a 120-step run holds whole sessions. A control script is sparse: ``at({12: [...]})`` puts its
lines at the top of that iteration (one poll per step while the world runs). The scenes that need an
organism's place take it from the world itself, through a pacer that aims.
"""

from __future__ import annotations

import dataclasses
import io
import json
import shutil
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from alife import cli, play, runner, streamrec
from alife.analysis import world_report as W
from alife.config import Config, apply_overrides, load_config
from alife.director import PlayerRequest, actions, make_director, pacing
from alife.director import log as director_log
from alife.game import lexicon, powers
from alife.game import report as R
from alife.game.chronicle import KIND, rebuild_backlog
from alife.runio import COMMANDS_FILE, DIRECTOR_LOG_FILE, RunDir, load_checkpoint
from alife.state import init_state, state_digest, to_numpy
from play_decode import decode
from play_helpers import (
    FakeClock,
    ScriptControl,
    checkpoint_digests,
    decoded,
    rows,
    run_paced,
    run_plain,
    same_events,
    snapshots,
    spawn,
    world,
)

END = 120
FORBIDDEN = (*lexicon.NEVER_LISTS, *lexicon.LABEL_LISTS)
CADENCE = {"metrics_every": 10, "heavy_metrics_every": 20, "checkpoint_every": 30, "keep_checkpoints": 50}


@pytest.fixture(autouse=True)
def small_powers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Durations of 30 to 120 steps in tens, cooldowns of 20, a budget window of 400, cards at +20 and +50."""
    for name, spec in list(actions.GAME_SPECS.items()):
        small: dict[str, Any] = {"cooldown": 20}
        if spec.duration is not None:
            small |= {"duration": (30, 120), "quantum": 10}
        monkeypatch.setitem(actions.GAME_SPECS, name, dataclasses.replace(spec, **small))
    monkeypatch.setattr(pacing, "PLAYER_BUDGET_WINDOW", 400)
    monkeypatch.setattr(powers, "SINCE_AFTER", (20, 50))


def player_cfg(*extra: str) -> Config:
    return apply_overrides(world("3d", director=True), ["run.director=player", *extra])


def at(lines: Mapping[int, Sequence[str]], n: int = END) -> list[list[str]]:
    """A control script with ``lines[t]`` at the top of iteration ``t`` of a running world."""
    return [list(lines.get(t, ())) for t in range(n)]


def go(
    tmp_path: Path,
    lines: Mapping[int, Sequence[str]] | Sequence[Sequence[str]],
    *,
    name: str = "w",
    cfg: Config | None = None,
    steps: int = END,
    options: play.PlayOptions | None = None,
    pacer_class: type[play.PlayPacer] = play.PlayPacer,
    control: Any = None,
) -> tuple[RunDir, play.PlayPacer, io.BytesIO]:
    script = at(lines) if isinstance(lines, Mapping) else lines
    rd, res, pacer, out = run_paced(
        cfg or player_cfg(),
        tmp_path / name,
        steps,
        control or script,
        options=options,
        pacer_class=pacer_class,
        game_specs=True,
    )
    assert res.status == "done" and pacer.failure is None
    return rd, pacer, out


def plans(out: io.BytesIO) -> list[dict[str, Any]]:
    return [
        {"at": r.step, "reply": r.f["reply"], **r.f["json"]} for r in decoded(out).records if r.tag == "PLAN"
    ]


def brief(out: io.BytesIO) -> list[tuple[int, int, str, bool]]:
    return [(p["at"], p["id"], p["status"], p["reply"]) for p in plans(out)]


def commands(rd: RunDir) -> list[dict[str, Any]]:
    return rd.read_jsonl(COMMANDS_FILE)


def log_of(rd: RunDir) -> list[dict[str, Any]]:
    return [r for r in rd.read_jsonl(DIRECTOR_LOG_FILE) if r.get("kind") != "header"]


def clean(text: str) -> None:
    assert lexicon.scan(text, FORBIDDEN) == [], text


def clean_plan_text(p: Mapping[str, Any]) -> None:
    """A plan's strings through the lexicon. The refusal of ``intent`` quotes the name of the line the client
    sent (the lexicon lists "intent" as a motive word): its name is masked, as a display name is."""
    mask = (lambda t: t.replace("intent", "Name")) if p["kind"] == "intent" else (lambda t: t)
    for text in (p["text"], *p["caution"], p["refused"] or "", (p["precedent"] or {}).get("line", "")):
        clean(mask(text))


SESSION = {
    12: ["power preview food id=1 mult=1.3 steps=40", "power food id=2 mult=1.3 steps=40"],
    15: ["power strike id=3 x=10 y=10"],
    17: ["power preview bloom id=4 x=10 y=10"],
    18: ["power bloom id=5 patch=1 gain=1.8 steps=60"],
    19: ["power cancel food id=6"],
    20: ["power food id=7 mult=1.0"],
    21: ["intent make it rain"],
}


@pytest.fixture(scope="module")
def session(tmp_path_factory: pytest.TempPathFactory) -> tuple[RunDir, io.BytesIO]:
    with pytest.MonkeyPatch.context() as mp:
        for name, spec in list(actions.GAME_SPECS.items()):
            small: dict[str, Any] = {"cooldown": 20}
            if spec.duration is not None:
                small |= {"duration": (30, 120), "quantum": 10}
            mp.setitem(actions.GAME_SPECS, name, dataclasses.replace(spec, **small))
        mp.setattr(pacing, "PLAYER_BUDGET_WINDOW", 400)
        mp.setattr(powers, "SINCE_AFTER", (20, 50))
        rd, _, out = go(tmp_path_factory.mktemp("session"), SESSION)
    return rd, out


# --- the exchange -------------------------------------------------------------------------------------------


def test_each_line_of_a_session_is_answered_with_its_plan_and_the_id_is_echoed(
    session: tuple[RunDir, io.BytesIO],
) -> None:
    _rd, out = session
    assert brief(out) == [
        (12, 1, "preview", True),
        (12, 2, "queued", False),
        (13, 2, "applied", False),
        (15, 3, "queued", False),
        (16, 3, "applied", False),
        (17, 4, "preview", True),
        (18, 5, "queued", False),
        (19, 5, "applied", False),
        (19, 6, "queued", False),
        (20, 6, "cancelled", False),
        (20, 2, "since", False),
        (20, 7, "refused", False),
        (21, 0, "refused", True),
        (40, 2, "since", False),  # +20 after Food (at 12; its end was the cancel at 19: reached at row 20)
        (40, 3, "since", False),
        (40, 5, "since", False),
        (70, 2, "since", False),  # +50 after Food: 62 -> 70
        (70, 3, "since", False),
        (70, 5, "since", False),
        (80, 5, "since", False),  # the end of Bloom (18 + 60 = 78 -> 80)
    ]
    by = {(p["id"], p["status"]): p for p in plans(out)}
    assert by[2, "applied"]["step_applied"] == 12 and by[3, "applied"]["step_applied"] == 15
    assert by[1, "preview"]["step_applied"] is None and by[2, "queued"]["step_applied"] is None
    assert by[7, "refused"]["refused"] == "Food is cooling down until step 72."
    assert by[0, "refused"]["kind"] == "intent" and by[0, "refused"]["refused"] == powers.INTENT_REFUSAL
    for p in plans(out):  # the encoder checked the shape; the texts are the lexicon's
        streamrec.check_plan({k: v for k, v in p.items() if k not in ("at", "reply")})
        clean_plan_text(p)


def test_a_preview_and_the_power_after_it_say_the_same_thing(session: tuple[RunDir, io.BytesIO]) -> None:
    _, out = session
    by = {(p["id"], p["status"]): p for p in plans(out)}
    preview, queued, applied = by[1, "preview"], by[2, "queued"], by[2, "applied"]
    drop = ("at", "reply", "id", "status", "step_applied", "caution")
    assert {k: v for k, v in preview.items() if k not in drop} == {
        k: v for k, v in queued.items() if k not in drop
    }
    assert {k: v for k, v in queued.items() if k not in (*drop, "ends")} == {
        k: v for k, v in applied.items() if k not in (*drop, "ends")
    }
    assert (preview["power"], preview["action"], preview["params"], preview["duration"]) == (
        "food",
        "regen_pulse",
        {"mult": pytest.approx(1.3)},
        40,
    )
    assert preview["text"] == "Food x1.3 for 40 steps (0.0 min at 1x), cost 1, 5 of 6 units left after."
    assert preview["caution"][0] == "holds the regen lever until step 52"
    assert preview["holds"] == "regen" and preview["ends"] == 52 and preview["ready"] == 72
    assert preview["influence"] == {"have": 5, "cap": 6, "window": 400, "free_at": [412]}
    assert preview["precedent"]["line"] == "No earlier case for x1.3." and preview["precedent"]["n"] == 0
    assert applied["step_applied"] == 12 and applied["ends"] == 52


def test_the_nearest_patch_of_a_click_and_the_clamped_names(session: tuple[RunDir, io.BytesIO]) -> None:
    _, out = session
    by = {(p["id"], p["status"]): p for p in plans(out)}
    bloom = by[4, "preview"]
    assert bloom["raw"] == {"patch": bloom["params"]["patch"], "steps": 6000, "gain": 1.75}
    assert bloom["clamped"] == ["steps"] and bloom["duration"] == 120  # 6000 asked, 120 is the (shrunk) cap
    assert (
        bloom["text"].startswith("Bloom x1.75 on patch ")
        and bloom["holds"] == f"patch:{bloom['params']['patch']}"
    )
    assert by[5, "queued"]["params"]["patch"] == 1 and by[5, "queued"]["clamped"] == []
    assert by[3, "queued"]["text"].startswith("Strike at (10.0, 10.0), radius 6.25, kill chance 0.45")
    assert by[3, "queued"]["holds"] is None and by[3, "queued"]["duration"] is None


def test_a_cancel_is_queued_then_cancelled_with_no_refund_and_the_cooldown_of_the_planned_end(
    session: tuple[RunDir, io.BytesIO],
) -> None:
    rd, out = session
    by = {(p["id"], p["status"]): p for p in plans(out)}
    q, c = by[6, "queued"], by[6, "cancelled"]
    assert q["text"] == "Food stops at step 19. No refund; ready again at step 72."
    assert c["text"] == "Food was stopped at step 19. No refund; ready again at step 72."
    assert (c["step_applied"], c["ends"], c["ready"], c["power"]) == (19, 19, 72, "food")
    [revert] = [e for e in log_of(rd) if e["source"] == "revert" and e["action"] == "regen_pulse"]
    assert (
        revert["step_applied"] == 19
        and revert["command"] == 6
        and revert["reason"] == "cancelled by the player"
    )
    # 12 + 40 = 52 planned end, 20 cooldown: Food stays unavailable until 72, cancelled or not
    assert by[7, "refused"]["ready"] == 72
    assert by[2, "since"]["ends"] == 19  # the card of the cancelled power ends at the cancellation


def test_the_log_holds_what_was_applied_at_the_step_it_was_applied(
    session: tuple[RunDir, io.BytesIO],
) -> None:
    rd, _ = session
    steps = [(e["step_applied"], e["source"], e["action"], e.get("command")) for e in log_of(rd)]
    assert steps == [
        (12, "player", "regen_pulse", 2),
        (15, "player", "catastrophe", 3),
        (18, "player", "fertility_shift", 5),
        (19, "revert", "regen_pulse", 6),
        (78, "revert", "fertility_shift", None),
    ]
    director_log.validate_log(rd.read_jsonl(DIRECTOR_LOG_FILE), rd.config())
    header = rd.read_jsonl(DIRECTOR_LOG_FILE)[0]
    assert header["kind"] == "header" and header["game_specs"] == "game-v0"


def test_commands_jsonl_has_a_line_for_each_power_line_and_the_last_line_of_an_id_says_what_became_of_it(
    session: tuple[RunDir, io.BytesIO],
) -> None:
    rd, _ = session
    got = [(c["id"], c["t"], c["verb"], c["status"], c.get("entry_step")) for c in commands(rd)]
    assert got == [
        (1, 12, "preview", "preview", None),
        (2, 12, "power", "queued", 12),
        (2, 12, "power", "applied", 12),
        (3, 15, "power", "queued", 15),
        (3, 15, "power", "applied", 15),
        (4, 17, "preview", "preview", None),
        (5, 18, "power", "queued", 18),
        (5, 18, "power", "applied", 18),
        (6, 19, "cancel", "queued", 19),
        (6, 19, "cancel", "applied", 19),
        (7, 20, "power", "refused", None),
        (0, 21, "intent", "refused", None),
    ]
    # a line is kept as it was received on every input line that answers it; the later line of an id (the
    # application) has none
    received = [text for texts in SESSION.values() for text in texts]
    assert [c["line"] for c in commands(rd) if c["status"] != "applied"] == received
    assert all("line" not in c for c in commands(rd) if c["status"] == "applied")
    assert commands(rd)[10]["reason"] == "Food is cooling down until step 72."


def test_the_chronicle_rows_are_kinds_20_and_21_and_a_strike_has_a_place(
    session: tuple[RunDir, io.BytesIO],
) -> None:
    _, out = session
    rows_ = [e for r in decoded(out).records if r.tag == "EVNT" and not r.f["backlog"] for e in r.f["json"]]
    mine = [
        (e["step"], e["kind"], e["imp"], e["text"], e["x"], e["y"]) for e in rows_ if e["kind"] in (20, 21)
    ]
    assert mine == [
        (12, 20, 1, "Food x1.3 from step 12 for 40 steps.", None, None),
        (15, 20, 1, "Strike at (10, 10), radius 6.25, kill chance 0.45, step 15.", 10.0, 10.0),
        (18, 20, 1, "Bloom x1.8 on patch 1 from step 18 for 60 steps.", None, None),
        (19, 21, 0, "Food was cancelled at step 19.", None, None),
        (78, 21, 0, "Bloom on patch 1 ended at step 78.", None, None),
    ]
    assert all(e["fam"] == "director" for e in rows_ if e["kind"] in (20, 21))
    assert [
        e
        for r in decoded(out).records
        if r.tag == "ALRT"
        for e in r.f["json"]
        if e["class"] == "director_acted"
    ] == []


def test_the_chronicle_of_a_resumed_world_tells_the_same_rows_as_the_live_one(
    session: tuple[RunDir, io.BytesIO],
) -> None:
    """``rebuild_backlog`` over the logs and the live ``from_director`` are one function."""
    rd, out = session
    live = [e for r in decoded(out).records if r.tag == "EVNT" and not r.f["backlog"] for e in r.f["json"]]
    entries = director_log.entries_of(rd.read_jsonl(DIRECTOR_LOG_FILE))
    from alife.game.chronicle import Chronicler

    back = rebuild_backlog([], [], entries, house_salt=0, limit=500, chronicler=Chronicler(0))
    kinds = (KIND["power_applied"], KIND["power_ended"])
    assert [e for e in back if e["kind"] in kinds] == [e for e in live if e["kind"] in kinds]


# --- the environment record ---------------------------------------------------------------------------------


def test_the_environment_record_carries_influence_patches_and_the_state_of_each_power(
    session: tuple[RunDir, io.BytesIO],
) -> None:
    _rd, out = session
    envs = [(r.step, r.f["json"]) for r in decoded(out).records if r.tag == "ENVS"]
    first = envs[0][1]
    assert first["influence"] == {"have": 6, "cap": 6, "window": 400, "free_at": []}
    assert [p["i"] for p in first["patches"]] == [0, 1, 2, 3] and all(
        p["gain"] == 1.0 for p in first["patches"]
    )
    assert first["powers"] == {n: {"ready": None, "busy": None} for n in play.POWER_NAMES}
    by_step = {s: j for s, j in envs}
    assert by_step[12]["influence"]["have"] == 5  # sent right after the request was queued: the unit is held
    assert by_step[12]["effects"] == []  # ... and the effect is not yet in force
    assert by_step[13]["influence"]["have"] == 5
    assert by_step[13]["effects"][0]["source"] == "player" and by_step[13]["effects"][0]["end"] == 52
    assert by_step[13]["powers"]["food"] == {"ready": 72, "busy": "Food is cooling down until step 72."}
    assert by_step[20]["effects"] == [e for e in by_step[20]["effects"] if e["action"] != "regen_pulse"]
    last = envs[-1][1]
    assert last["influence"]["have"] == 3 and last["influence"]["free_at"] == [412, 415, 418]
    assert all(set(p) == {"i", "x", "y", "gain", "sigma"} for p in last["patches"])
    assert [p["gain"] for p in last["patches"]] == [1.0, 1.0, 1.0, 1.0]  # Bloom ended at 78
    mid = by_step[30]
    assert [p["gain"] for p in mid["patches"]][1] == pytest.approx(1.8)
    assert mid["powers"]["bloom"]["busy"] == "Bloom is cooling down until step 98."


def test_the_header_lists_the_powers_and_the_game_table_and_a_world_without_a_director_has_neither(
    tmp_path: Path, session: tuple[RunDir, io.BytesIO]
) -> None:
    rd, out = session
    play_hdr = decoded(out).header["play"]
    assert play_hdr["powers"] == powers.header_block(rd.config())
    assert (
        play_hdr["game_specs"] == "game-v0"
        and "PLAN" in play_hdr["records"]
        and play_hdr["versions"]["PLAN"] == 1
    )
    _, _, _, bare = run_paced(world("3d"), tmp_path / "bare", 5, [[]])
    hdr = decoded(bare).header["play"]
    assert "powers" not in hdr and "game_specs" not in hdr and "PLAN" in hdr["records"]
    assert all(
        r.f["json"]["influence"] is None and "powers" not in r.f["json"]
        for r in decoded(bare).records
        if r.tag == "ENVS"
    )


# --- the since card -----------------------------------------------------------------------------------------


def test_a_since_card_carries_the_exact_rows_of_the_metrics_log(session: tuple[RunDir, io.BytesIO]) -> None:
    rd, out = session
    metrics = {r["step"]: powers.compact_row(r) for r in rd.metrics_rows()}
    cards = [p for p in plans(out) if p["status"] == "since"]
    assert [(p["id"], p["at"]) for p in cards] == [
        (2, 20), (2, 40), (3, 40), (5, 40), (2, 70), (3, 70), (5, 70), (5, 80),
    ]  # fmt: skip
    for p in cards:
        assert p["since"]["from"] == p["step_applied"]
        assert all(r == metrics[r["step"]] for r in p["since"]["rows"])
        assert p["since"]["rows"][-1]["step"] == p["at"]  # a card is sent when its newest row is written
    by = {(p["id"], p["at"]): [r["step"] for r in p["since"]["rows"]] for p in cards}
    # the row before the power, the row after it (whose interval holds it), +20 and +50
    assert by[2, 20] == [10, 20] and by[2, 70] == [10, 20, 40, 70]  # the cancel at 19 is reached at row 20
    # Bloom at 18: before 10, during 20; +20 is 38 -> 40, +50 is 68 -> 70, the end at 78 -> 80
    assert by[5, 40] == [10, 20, 40] and by[5, 70] == [10, 20, 40, 70] and by[5, 80] == [10, 20, 40, 70, 80]
    assert by[3, 40] == [10, 20, 40] and by[3, 70] == [10, 20, 40, 70]  # Strike has no end: two milestones


def test_every_milestone_gets_its_plan_in_order_and_a_card_is_the_function_of_the_files(
    tmp_path: Path,
) -> None:
    rd, _, out = go(tmp_path, {12: ["power food id=2 mult=1.3 steps=40"]})
    cards = [p for p in plans(out) if p["status"] == "since"]
    assert [(p["at"], [r["step"] for r in p["since"]["rows"]]) for p in cards] == [
        (40, [10, 20, 40]),  # before 10, the row after the power 20, +20 = 32 -> 40
        (60, [10, 20, 40, 60]),  # the end, 12 + 40 = 52 -> 60
        (70, [10, 20, 40, 60, 70]),  # +50 = 62 -> 70
    ]
    metrics = [powers.compact_row(r) for r in rd.metrics_rows()]
    [entry] = [e for e in log_of(rd) if e["source"] == "player"]
    [end] = [e for e in log_of(rd) if e["source"] == "revert"]
    offline, reached = powers.since_rows(entry, end["step_applied"], metrics, END)
    assert reached == len(powers.milestones(entry, end["step_applied"])) == 3
    assert cards[-1]["since"]["rows"] == offline
    assert all(len(json.dumps(p)) < streamrec.PLAN_JSON_CAP for p in plans(out))


# --- previews and refusals change nothing -------------------------------------------------------------------


REFUSALS = {
    10: ["power preview food id=1 mult=1.0", "power food id=2 mult=1.0"],
    12: ["power bloom id=3 patch=9", "power food id=4 steps=40", "power strike id=5 x=1 y=2 gain=1.5"],
    14: [
        "power cancel food id=6",
        "power preview blight id=7 patch=2 steps=60",
        "power preview strike id=8 x=20 y=20",
    ],
    16: ["power cancel strike id=9", "intent do something", "power blight id=10 patch=7"],
    18: ["power preview bloom id=11 x=500 y=4 gain=9"],
}


def test_previews_and_refusals_leave_the_digest_and_the_logs_of_a_plain_run(tmp_path: Path) -> None:
    plain_rd, _ = run_plain(player_cfg(), tmp_path / "plain", END)
    rd, _, out = go(tmp_path, REFUSALS)
    assert checkpoint_digests(rd) == checkpoint_digests(plain_rd)
    assert rows(rd) == rows(plain_rd)
    assert same_events(rd, plain_rd)
    assert log_of(rd) == [] and len(rd.read_jsonl(DIRECTOR_LOG_FILE)) == 1  # a header
    statuses = [(p["id"], p["status"]) for p in plans(out)]
    assert statuses == [
        (1, "refused"), (2, "refused"), (3, "refused"), (4, "refused"), (5, "refused"), (6, "refused"),
        (7, "preview"), (8, "preview"), (9, "refused"), (0, "refused"), (10, "refused"), (11, "preview"),
    ]  # fmt: skip
    assert [e["status"] for e in commands(rd)] == [p[1] for p in statuses]


def test_each_refusal_says_why_in_the_players_words(tmp_path: Path) -> None:
    _, _, out = go(tmp_path, REFUSALS)
    why = {p["id"]: p["refused"] for p in plans(out)}
    assert why[1] == why[2] == "Food x1 lies between 0.9 and 1.15, where it makes no difference worth a cost."
    assert why[3] == "Patch 9 does not exist; patches are numbered 0 to 3."
    assert why[4] == "Food is missing mult, the factor on food regrowth."
    assert why[5] == "Strike takes x, y, radius and kill, not gain."
    assert (
        why[6] == "Nothing to cancel: Food is not running."
        and why[9] == "Nothing to cancel: Strike is not running."
    )
    assert why[10] == "Patch 7 does not exist; patches are numbered 0 to 3."
    for p in plans(out):
        clean_plan_text(p)


def test_a_preview_clamps_to_the_tables_bounds_and_names_what_it_moved(tmp_path: Path) -> None:
    _, _, out = go(tmp_path, {10: ["power preview bloom id=1 x=5000 y=4 gain=9 steps=1"]})
    [p] = [p for p in plans(out) if p["status"] == "preview"]
    assert p["clamped"] == ["gain", "steps"] and p["params"]["gain"] == 2.0 and p["duration"] == 30
    assert p["text"].startswith("Bloom x2 on patch ") and p["raw"]["gain"] == 9.0


def aimed(lines: Mapping[int, str]) -> type[play.PlayPacer]:
    """A pacer that, at the top of the given iterations, handles a line whose ``{x}`` and ``{y}`` are the
    place of the first living organism (a strike aimed at somebody; a tiny world cannot be told where)."""

    class Aimed(play.PlayPacer):
        def before_step(self, state: Any, t: int) -> None:
            if t in lines:
                self._t = t
                pos, alive = to_numpy(state.pos), to_numpy(state.alive)
                i = int(np.argmax(alive))
                self._apply([lines[t].format(x=f"{pos[i, 0]:.4f}", y=f"{pos[i, 1]:.4f}")], state)
            super().before_step(state, t)

    return Aimed


def test_a_strike_preview_counts_the_column_and_never_changes_the_world(tmp_path: Path) -> None:
    line = "power preview strike id=1 x={x} y={y} radius=10 kill=0.6"
    rd, _, out = go(tmp_path, [], pacer_class=aimed({15: line}), steps=20)
    [p] = [p for p in plans(out) if p["status"] == "preview"]
    n = int(p["caution"][-1].split(" ")[0])
    assert n >= 1 and p["caution"][-1].startswith(
        f"{n} organisms in the column; {n * 0.6:,.0f} deaths on average ("
    )
    clean(p["caution"][-1])
    assert log_of(rd) == []


# --- when a power lands -------------------------------------------------------------------------------------


def test_a_power_sent_while_the_world_is_paused_lands_at_the_first_step_after_it_resumes(
    tmp_path: Path,
) -> None:
    script: list[list[str]] = [[]] * 10 + [
        ["pause"],
        ["power food id=1 mult=1.3 steps=40"],
        [],
        [],
        [],
        ["run"],
    ]
    rd, _, out = go(tmp_path, script, steps=60)
    [entry] = [e for e in log_of(rd) if e["source"] == "player"]
    assert entry["step_applied"] == 10  # the world stood at step 10 the whole time it was paused
    queued = next(p for p in plans(out) if p["status"] == "queued")
    assert queued["caution"][-1] == "lands when the world resumes" and queued["at"] == 10
    applied = next(p for p in plans(out) if p["status"] == "applied")
    assert applied["at"] == 11 and applied["step_applied"] == 10
    [c] = [c for c in commands(rd) if c["status"] == "queued"]
    assert c["t"] == 10 and c["entry_step"] == 10


def test_a_queued_power_is_dropped_at_quit_and_never_logged(tmp_path: Path) -> None:
    script = at({20: ["power food id=1 mult=1.3 steps=40", "quit"]})
    rd, res, _pacer, out = run_paced(
        player_cfg(),
        tmp_path / "q",
        END,
        script,
        options=play.PlayOptions(rate=float("inf"), paused=False, unattended=True, fresh=True),
        game_specs=True,
    )
    assert res.status == "interrupted"
    assert log_of(rd) == []  # not applied, not logged; the world stopped one step later
    assert [c["status"] for c in commands(rd)] == ["queued"]  # ... and the input log says what was asked
    assert [p["status"] for p in plans(out)] == ["queued"]
    stop = int(res.state.step)
    assert stop == 21  # a quit lets one more step run
    _, plain = run_plain(player_cfg(), tmp_path / "plain", stop)
    assert state_digest(res.state) == state_digest(plain.state)  # the world the quit left has no power in it


def test_the_ledger_closes_with_every_power(tmp_path: Path) -> None:
    """Powers move energy only through the director's existing actions: the exact 16-term ledger closes at
    every row (the runner stops the run with an invariant violation otherwise), also across a Strike that
    kills (its deaths are the catastrophe's)."""
    script = {
        12: ["power food id=1 mult=1.5 steps=60"],
        14: ["power bloom id=2 patch=0 gain=2 steps=60"],
        16: ["power blight id=3 patch=1 steps=40"],
    }
    strike = "power strike id=9 x={x} y={y} radius=10 kill=0.6"
    rd, _, out = go(tmp_path, script, pacer_class=aimed({25: strike}))
    assert [(e["step_applied"], e["action"]) for e in log_of(rd) if e["source"] == "player"] == [
        (12, "regen_pulse"), (14, "fertility_shift"), (16, "habitat_loss"), (25, "catastrophe"),
    ]  # fmt: skip
    metrics = rd.metrics_rows()
    assert len(metrics) == END // 10 and all(r["ledger_error"] == 0 for r in metrics)
    killed = sum(r["deaths_catastrophe"] for r in metrics)
    assert killed >= 1, "the strike was aimed at an organism: it killed some"
    assert next(r for r in metrics if r["step"] == 30)["deaths_catastrophe"] == killed  # all in its interval
    # the units: Food 1 + Bloom 1 + Blight 2 + Strike 1 = 5 of 6, the last one is free
    assert [p for p in plans(out) if p["status"] == "applied"][-1]["influence"]["have"] == 1


def test_a_strikes_card_counts_the_organisms_in_its_column_before_and_after(tmp_path: Path) -> None:
    strike = "power strike id=9 x={x} y={y} radius=10 kill=0.6"
    rd, _, out = go(tmp_path, [], pacer_class=aimed({25: strike}), steps=60)
    card = next(p for p in plans(out) if p["status"] == "since")
    disc = card["since"]["disc"]
    metrics = {r["step"]: r for r in rd.metrics_rows()}
    assert disc["before"] >= 1 and disc["after"] <= disc["before"]
    held = next(r for r in card["since"]["rows"] if r["step"] == 30)  # the row whose interval holds step 25
    assert held["catastrophe"] == metrics[30]["deaths_catastrophe"] >= 1


# --- replay -------------------------------------------------------------------------------------------------


@pytest.fixture
def replayable(session: tuple[RunDir, io.BytesIO], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(play, "PLAY_CADENCE", CADENCE)
    snapshots(monkeypatch, 10)
    dst = tmp_path / "live"
    shutil.copytree(session[0].path, dst)
    return dst


def test_a_scripted_session_replays_to_the_live_digests_by_both_commands_and_leaves_the_inputs_alone(
    replayable: Path, tmp_path: Path, capfdbinary: pytest.CaptureFixture[bytes]
) -> None:
    live = RunDir.open(replayable)
    log, inputs = (replayable / DIRECTOR_LOG_FILE).read_bytes(), (replayable / COMMANDS_FILE).read_bytes()
    a, b = tmp_path / "a", tmp_path / "b"
    shutil.copytree(replayable, a)
    shutil.copytree(replayable, b)
    assert (
        cli.main(
            ["resume", str(a), "--checkpoint", "0", "--director", "replay", "--steps", str(END), "--quiet"]
        )
        == 0
    )
    code = cli.main(
        ["play", str(b), "--adopt", "--checkpoint", "0", "--truncate", "--replay", "--no-control",
         "--rate", "max", "--until", str(END)]
    )  # fmt: skip
    cap = capfdbinary.readouterr()
    assert code == 0, cap.err.decode(errors="replace")
    ra, rb = RunDir.open(a), RunDir.open(b)
    mine = checkpoint_digests(live)  # the live run kept 0, 60, 90 and 120; a replay keeps them all
    assert sorted(mine) == [0, 60, 90, 120]
    for r in (ra, rb):
        assert {s: d for s, d in checkpoint_digests(r).items() if s in mine} == mine
    for d in (a, b):
        assert (d / DIRECTOR_LOG_FILE).read_bytes() == log, "the log is the one they started from"
    assert (b / COMMANDS_FILE).read_bytes() == inputs, "a replay never touches the inputs"
    assert not (a / COMMANDS_FILE).exists() or (a / COMMANDS_FILE).read_bytes() == inputs
    # the replay shows what the log holds: applications, the cancel, the effects with their source
    d = decode(cap.out)
    pl = [(r.f["json"]["id"], r.f["json"]["status"]) for r in d.records if r.tag == "PLAN"]
    assert (2, "applied") in pl and (3, "applied") in pl and (5, "applied") in pl and (6, "cancelled") in pl
    envs = [r.f["json"] for r in d.records if r.tag == "ENVS" and r.f["json"]["effects"]]
    assert envs and {e["source"] for j in envs for e in j["effects"]} == {"player"}
    assert all(
        "influence" in j and j["influence"] is None
        for j in (r.f["json"] for r in d.records if r.tag == "ENVS")
    )


def test_a_replay_refuses_every_new_power_and_changes_nothing(
    session: tuple[RunDir, io.BytesIO], tmp_path: Path
) -> None:
    live, _ = session
    cfg = apply_overrides(live.config(), ["run.director=replay"])
    rd = RunDir.create(tmp_path / "r", cfg)
    rd.update_meta(start_step=0)
    shutil.copyfile(live.path / DIRECTOR_LOG_FILE, rd.path / DIRECTOR_LOG_FILE)
    shutil.copyfile(live.path / COMMANDS_FILE, rd.path / COMMANDS_FILE)
    inputs = (rd.path / COMMANDS_FILE).read_bytes()
    from alife.director import Director

    director = Director(rd, cfg, "replay", 0)
    out = io.BytesIO()
    pacer = play.PlayPacer(
        out,
        ScriptControl(
            at(
                {
                    40: [
                        "power food id=50 mult=1.3",
                        "power preview strike id=51 x=1 y=1",
                        "power cancel bloom id=52",
                    ]
                }
            )
        ),
        play.PlayOptions(rate=float("inf"), paused=False, fresh=True, unattended=True),
        clock=FakeClock(),
        note=lambda m: None,
    )
    res = runner.run(rd, cfg, init_state(cfg), END, quiet=True, director=director, pacer=pacer)
    pacer.close()
    assert res.status == "done"
    refused = [p for p in plans(out) if p["id"] >= 50]
    assert [(p["id"], p["status"], p["refused"]) for p in refused] == [
        (50, "refused", "A replay plays the log back; it takes no new power."),
        (51, "refused", "A replay plays the log back; it takes no new power."),
        (52, "refused", "A replay plays the log back; it takes no new power."),
    ]
    assert (rd.path / COMMANDS_FILE).read_bytes() == inputs  # a replay writes no line of its own
    assert checkpoint_digests(rd) == checkpoint_digests(live)
    assert rd.read_jsonl(DIRECTOR_LOG_FILE) == live.read_jsonl(DIRECTOR_LOG_FILE)


# --- rule P and resume --------------------------------------------------------------------------------------


def test_a_power_at_a_chapter_end_is_after_its_checkpoint_and_a_crash_resume_drops_it_with_its_input(
    tmp_path: Path,
) -> None:
    """The chapter pause sits on the checkpoint step, so the power the player sends there is applied at that
    step after the checkpoint (rule P): resuming from it cuts the record and its ``commands.jsonl`` lines, the
    world is what it was, and playing the same input again reproduces the log byte for byte."""
    opts = play.PlayOptions(rate=float("inf"), paused=False, fresh=True, chapter=30)
    script = [*at({}, 30), [], ["power food id=1 mult=1.3 steps=40"], [], ["run"]]
    cfg = player_cfg()
    rd, res, _pacer, _out = run_paced(cfg, tmp_path / "a", 60, script, options=opts, game_specs=True)
    assert res.status == "done"
    [entry] = [e for e in log_of(rd) if e["source"] == "player"]
    assert entry["step_applied"] == 30  # the chapter end: after the checkpoint at 30
    assert [(c["t"], c["status"]) for c in commands(rd)] == [(30, "queued"), (30, "applied")]
    digests = checkpoint_digests(rd)
    plain, _ = run_plain(cfg, tmp_path / "plain", 60)
    assert digests[30] == checkpoint_digests(plain)[30], "the checkpoint at 30 does not hold the power"
    log_bytes = (rd.path / DIRECTOR_LOG_FILE).read_bytes()
    # crash after the power: resume from the checkpoint at 30
    b = tmp_path / "b"
    shutil.copytree(rd.path, b)
    rb = RunDir.open(b)
    rb.truncate_after(30)
    assert [e for e in log_of(rb)] == [] and commands(rb) == []  # the record and its lines are gone
    state = load_checkpoint(rb.checkpoint_at(30), cfg)
    director = make_director(rb, cfg, 30, game_specs=True)
    pacer2 = play.PlayPacer(
        io.BytesIO(),
        ScriptControl([["power food id=1 mult=1.3 steps=40"]]),
        play.PlayOptions(rate=float("inf"), paused=False, unattended=True),
        clock=FakeClock(),
        note=lambda m: None,
    )
    runner.run(rb, cfg, state, 30, quiet=True, director=director, pacer=pacer2)
    pacer2.close()
    assert (rb.path / DIRECTOR_LOG_FILE).read_bytes() == log_bytes
    assert checkpoint_digests(rb)[60] == digests[60]


def test_a_resumed_world_rebuilds_the_open_cards_and_goes_on_as_the_uninterrupted_one(tmp_path: Path) -> None:
    lines = {12: ["power food id=2 mult=1.3 steps=100"]}
    rd_full, _, out_full = go(tmp_path, lines, name="full")
    # the same world, stopped at 41 (a quit checkpoints one step on) and resumed
    rd_a, res, _pacer, _out_a = run_paced(
        player_cfg(), tmp_path / "part", END, at({**lines, 40: ["quit"]}), game_specs=True,
        options=play.PlayOptions(rate=float("inf"), paused=False, unattended=True, fresh=True),
    )  # fmt: skip
    assert res.status == "interrupted"
    stop = max(s for s, _ in rd_a.checkpoints())
    assert 40 < stop <= 42
    cfg = rd_a.config()
    rd_a.truncate_after(stop)
    out_b = io.BytesIO()
    pacer_b = play.PlayPacer(
        out_b, ScriptControl([]), play.PlayOptions(rate=float("inf"), paused=False, unattended=True),
        clock=FakeClock(), note=lambda m: None,
    )  # fmt: skip
    director = make_director(rd_a, cfg, stop, game_specs=True)
    runner.run(
        rd_a,
        cfg,
        load_checkpoint(rd_a.checkpoint_at(stop), cfg),
        END - stop,
        quiet=True,
        director=director,
        pacer=pacer_b,
    )
    pacer_b.close()
    assert checkpoint_digests(rd_a)[END] == checkpoint_digests(rd_full)[END]
    assert (rd_a.path / DIRECTOR_LOG_FILE).read_bytes() == (rd_full.path / DIRECTOR_LOG_FILE).read_bytes()
    first = plans(out_b)[:2]
    assert [(p["id"], p["status"]) for p in first] == [(2, "applied"), (2, "since")]
    assert first[0]["step_applied"] == 12 and first[0]["text"] == plans(out_full)[2]["text"]
    # what comes after the open is what the uninterrupted world sent after the same step
    tail_b = [(p["at"], p["id"], p["status"], p["since"]) for p in plans(out_b)[2:]]
    tail_full = [(p["at"], p["id"], p["status"], p["since"]) for p in plans(out_full) if p["at"] > stop]
    assert tail_b == tail_full and tail_b
    back = [
        e["text"] for r in decoded(out_b).records if r.tag == "EVNT" and r.f["backlog"] for e in r.f["json"]
    ]
    assert "Food x1.3 from step 12 for 100 steps." in back


# --- the desk, with a stand-in director ---------------------------------------------------------------------


def test_a_request_refused_when_it_is_applied_is_reported_and_its_hold_released() -> None:
    cfg = player_cfg()
    written: list[Any] = []
    stand_in = SimpleNamespace(mode="player", entries=[])
    desk = powers.PowerDesk(cfg, stand_in, base_rate=14.4, append=written.extend)
    desk.reserved = {5: [{"step_applied": 3, "source": "player"}]}
    request = PlayerRequest(5, "food", {"mult": 1.3}, 40)
    outs, changed = desk.after_step([], [(request, "regen_pulse is cooling down until step 99")], 3)
    assert changed is True and desk.reserved == {}  # the environment record follows: the units are free
    [out] = outs
    assert out.plan["status"] == "refused" and out.plan["refused"] == "Food is cooling down until step 99."
    assert written == [
        {
            "id": 5,
            "t": 3,
            "verb": "power",
            "status": "refused",
            "entry_step": None,
            "reason": "Food is cooling down until step 99.",
        }
    ]
    streamrec.check_plan(out.plan)


def test_a_world_opened_at_a_powers_step_does_not_show_the_power_applied_after_that_step_s_checkpoint() -> (
    None
):
    """Rule P at the desk: a replay from the checkpoint at ``s`` has the player's record at ``s`` in the log
    (it is applied at the top of iteration ``s``), and it must not be on the screen before it happens."""
    entry = {
        "step_applied": 60, "source": "player", "action": "regen_pulse", "params": {"mult": 1.3}, "prev": {},
        "duration": 40, "scope": "player", "command": 4, "power": "food", "cost": 1, "specs": "game-v0",
    }  # fmt: skip
    stand_in = SimpleNamespace(mode="replay", entries=[entry])
    desk = powers.PowerDesk(player_cfg(), stand_in, base_rate=14.4, append=lambda lines: None)
    assert desk.open([{"step": 60, "n_alive": 24.0}], 60) == [] and desk.cards == {}
    outs = desk.open([{"step": 60, "n_alive": 24.0}], 61)  # one step later it has happened
    assert [o.plan["status"] for o in outs] == ["applied"] and list(desk.cards) == [4]


def test_a_world_with_a_rule_tier_director_takes_powers_too_and_keeps_the_scopes_apart(
    tmp_path: Path,
) -> None:
    cfg = apply_overrides(world("3d", director=True), ["run.director=live"])
    rd, _pacer, out = go(tmp_path, {12: ["power food id=1 mult=0.8 steps=60"]}, cfg=cfg)
    assert [p["status"] for p in plans(out)][:2] == ["queued", "applied"]
    assert any(e["source"] == "player" for e in log_of(rd))
    director_log.validate_log(rd.read_jsonl(DIRECTOR_LOG_FILE), cfg)
    assert decoded(out).header["play"]["powers"]["food"]["label"] == "Food"


# --- the command as a process ----------------------------------------------------------------------------


def test_a_power_over_the_pipe_of_a_real_play_process(tmp_path: Path) -> None:
    """``alife play`` on a world with a director launches in the player mode; a power sent on stdin is
    answered on stdout, applied, logged with its input, and the header says what the window may use. The real
    table is in force here (a 3,000-step Food), the run is 40 steps."""
    out = tmp_path / "w"
    args = [
        *(
            "play",
            "--preset",
            "tiny3d",
            "--set",
            "brain.weight_dtype=float32",
            "--set",
            "director.enabled=true",
        ),
        *("--set", "population.n_init=24", "--out", str(out), "--until", "40"),
    ]
    send = "".join(
        f"{line}\n"
        for line in (
            "power preview food id=1 mult=1.3 steps=3000",
            "power food id=2 mult=1.3 steps=3000",
            "intent hi",
            "rate max",
            "run",
        )
    )
    p = spawn(*args)
    watchdog = threading.Timer(240, p.kill)  # a hung process fails the test instead of hanging the suite
    watchdog.start()
    assert p.stdin is not None and p.stdout is not None and p.stderr is not None
    p.stdin.write(send.encode())  # stdin stays open: its EOF would be a quit, and a quit drops a queued power
    p.stdin.flush()
    data = p.stdout.read()  # the run ends at its 40 steps
    p.stdin.close()
    code, err = p.wait(timeout=120), p.stderr.read().decode(errors="replace")
    watchdog.cancel()
    assert code == 0, err
    d = decode(data)
    header = d.header["play"]
    assert header["game_specs"] == "game-v0" and list(header["powers"]) == list(play.POWER_NAMES)
    assert "PLAN" in header["records"] and d.skipped == []
    got = [(r.f["json"]["id"], r.f["json"]["status"]) for r in d.records if r.tag == "PLAN"]
    assert got == [(1, "preview"), (2, "queued"), (0, "refused"), (2, "applied")]
    [applied] = [r.f["json"] for r in d.records if r.tag == "PLAN" and r.f["json"]["status"] == "applied"]
    assert applied["step_applied"] == 0 and applied["duration"] == 3000 and applied["ends"] == 3000
    rd = RunDir.open(out)
    assert rd.config().run.director == "player"
    [entry] = [e for e in log_of(rd) if e["source"] == "player"]
    assert (entry["step_applied"], entry["command"], entry["power"], entry["duration"]) == (
        0,
        2,
        "food",
        3000,
    )
    assert [(c["id"], c["status"]) for c in commands(rd)] == [
        (1, "preview"), (2, "queued"), (0, "refused"), (2, "applied"),
    ]  # fmt: skip


# --- the report --------------------------------------------------------------------------------


def test_the_reports_of_a_world_with_powers_name_the_player_and_verify_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Three chapters of 500 steps with a Food at 300, a Strike and a Bloom at 640 (off every grid; the real
    table, so Food and Bloom outlast the run): the live reports are the offline ones (`verify` passes, nothing
    excluded) and the interventions block says "the player" with each power's numbers."""
    monkeypatch.undo()  # the real game table, not the shrunk one of the other tests
    cfg = load_config(
        preset="tiny3d",
        overrides=[
            *("brain.weight_dtype=float32", "run.metrics_every=250", "run.heavy_metrics_every=250"),
            *("run.checkpoint_every=500", "run.diversity_sample=64", "game.observe=true"),
            *("director.enabled=true", "director.rule_every=250", "population.n_init=24"),
            "run.director=player",
        ],
    )
    rd = RunDir.create(tmp_path / "w", cfg)
    state = init_state(cfg)
    rd.update_meta(origin_row=play.origin_row(state, cfg), start_step=0)
    script = at(
        {
            300: ["power food id=1 mult=1.3 steps=3000"],
            640: [
                "power strike id=2 x=30 y=30 radius=8 kill=0.5",
                "power bloom id=3 patch=1 gain=1.8 steps=3000",
            ],
        },
        1500,
    )
    pacer = play.PlayPacer(
        io.BytesIO(),
        ScriptControl(script),
        play.PlayOptions(rate=float("inf"), paused=False, fresh=True, unattended=True, chapter=500),
        clock=FakeClock(),
        note=lambda m: None,
    )
    director = make_director(rd, cfg, 0, game_specs=True)
    res = runner.run(rd, cfg, state, 1500, quiet=True, director=director, pacer=pacer)
    pacer.close()
    assert res.status == "done" and pacer.failure is None
    recs = rd.reports()
    assert [r["step"] for r in recs] == [500, 1000, 1500]
    code, summary = W.verify_reports(R.load_run(rd.path), recs)
    assert code == 0 and summary["counts"] == {"ok": 3}, summary
    assert all(r["excluded_candidates"] == [] for r in recs)
    texts = [[s["text"] for s in r["report"]["interventions"]] for r in recs]
    assert (
        texts[0][0] == "At step 300, the player set food regrowth to 1.3 times its base rate for 3,000 steps."
    )
    assert any(
        "At step 640, the player set off a catastrophe of radius 8 around (30, 30)" in t for t in texts[1]
    )
    assert any("the player set the fertility of patch 1 to 1.8 times" in t for t in texts[2])
