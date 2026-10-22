"""The findings of the third review of G3b (stage 8, docs/DESIGN.md I54): one test per defect, each of which
failed before its fix. Worlds are ``tiny3d`` on the CPU with a chapter of 500 steps (a whole number of social
snapshots) and the director in the ``player`` mode."""

# ruff: noqa: E501

from __future__ import annotations

import dataclasses
import io
from pathlib import Path
from typing import Any

import pytest

import play_helpers as H
from alife import play, runner
from alife.analysis import world_report as W
from alife.config import load_config
from alife.director import actions, make_director
from alife.game import report as R
from alife.game import reportjob
from alife.runio import RunDir
from alife.state import init_state

CHAPTER = 500


def player_world(
    path: Path, script: dict[int, list[str]], steps: int, *extra: str, director: str = "player"
) -> tuple[RunDir, io.BytesIO, play.PlayPacer]:
    """Play a tiny3d world with a director for ``steps`` steps, the lines of ``script`` at the top of their
    iteration, chapters of 500 steps and reports on."""
    cfg = load_config(
        preset="tiny3d",
        overrides=[
            *("brain.weight_dtype=float32", "run.metrics_every=250", "run.heavy_metrics_every=250"),
            *("run.checkpoint_every=500", "run.diversity_sample=64", "game.observe=true"),
            *("director.enabled=true", "director.rule_every=250", "population.n_init=24"),
            f"run.director={director}",
            *extra,
        ],
    )
    rd = RunDir.create(path, cfg)
    state = init_state(cfg)
    rd.update_meta(origin_row=play.origin_row(state, cfg), start_step=0)
    out = io.BytesIO()
    pacer = play.PlayPacer(
        out,
        H.ScriptControl([list(script.get(t, ())) for t in range(steps)]),
        play.PlayOptions(rate=float("inf"), paused=False, fresh=True, unattended=True, chapter=CHAPTER),
        clock=H.FakeClock(),
        note=lambda m: None,
    )
    director = make_director(rd, cfg, 0, game_specs=True)
    res = runner.run(rd, cfg, state, steps, quiet=True, director=director, pacer=pacer)
    pacer.close()
    assert res.status == "done" and pacer.failure is None
    return rd, out, pacer


def rprt(out: io.BytesIO) -> dict[int, dict[str, Any]]:
    return {int(r.f["json"]["b"]): r.f["json"] for r in H.decoded(out).records if r.tag == "RPRT"}


# --- 1 and 2: a player's record at the chapter end is outside the chapter ---------------------------------------


@pytest.fixture(scope="module")
def end_on_the_chapter(tmp_path_factory: pytest.TempPathFactory):
    """A Food cast at the chapter pause (step 500) that lasts one chapter: its revert is at 1,000, the chapter
    end, after the checkpoint there."""
    mp = pytest.MonkeyPatch()
    spec = actions.GAME_SPECS["regen_pulse"]
    mp.setitem(actions.GAME_SPECS, "regen_pulse", dataclasses.replace(spec, duration=(500, 3000)))
    try:
        rd, out, pacer = player_world(
            tmp_path_factory.mktemp("end") / "w",
            {
                300: ["power preview food id=7 mult=1.3 steps=500"],
                500: ["power food id=1 mult=1.3 steps=500"],
            },
            1600,
        )
    finally:
        mp.undo()
    return rd, out, pacer


def test_the_pacer_tells_the_alert_engine_the_metrics_interval_of_the_world(end_on_the_chapter) -> None:
    _, _, pacer = end_on_the_chapter
    assert pacer.alerts.cadence == 250  # run.metrics_every of the world; a row of fewer steps is no sample


def test_a_preview_caution_names_the_slots_not_a_capacity(end_on_the_chapter) -> None:
    """The population "0.55 of the world's capacity of 65,536" read as headroom where the world already sat at
    its carrying capacity: the caution counts organisms against the slots of the state and says so."""
    _, out, _ = end_on_the_chapter
    (preview,) = [
        r.f["json"] for r in H.decoded(out).records if r.tag == "PLAN" and r.f["json"]["status"] == "preview"
    ]
    pop = [c for c in preview["caution"] if c.startswith("population")]
    assert len(pop) == 1 and pop[0].endswith(" slots") and "capacity" not in pop[0], preview["caution"]


def test_the_live_report_of_an_effect_ending_on_the_chapter_end_is_the_offline_report(
    end_on_the_chapter,
) -> None:
    rd, _, _ = end_on_the_chapter
    entries = [e for e in rd.read_jsonl("director.jsonl") if e.get("kind") != "header"]
    assert [(e["step_applied"], e["source"]) for e in entries] == [(500, "player"), (1000, "revert")]
    recs = rd.reports()
    assert [r["step"] for r in recs] == [500, 1000, 1500]
    code, summary = W.verify_reports(R.load_run(rd.path), recs)
    assert code == 0 and summary["counts"] == {"ok": 3}, summary
    texts = [[s["text"] for s in r["report"]["interventions"]] for r in recs]
    assert "It was still in place at step 1,000." in texts[1]
    assert any("It ended at step 1,000" in t for t in texts[2]), texts[2]


def test_the_live_since_strip_is_the_backlog_the_files_give_when_the_world_is_opened(
    end_on_the_chapter,
) -> None:
    rd, out, _ = end_on_the_chapter
    live = rprt(out)
    backlog = dict(reportjob.stored_objects(rd.path, length=CHAPTER, n_origin=0))
    assert sorted(live) == sorted(backlog) == [500, 1000, 1500]
    for b in backlog:
        assert live[b]["since"] == backlog[b]["since"], b
    assert live[500]["since"]["influence"] == {"have": 6, "cap": 6}  # the power at 500 comes after the card
    assert live[1000]["since"]["ending"] == [{"action": "regen_pulse", "end": 1000}]


# --- 3: a rule's effect held at launch ends in the player mode -------------------------------------------------


def test_a_rule_effect_held_when_a_world_is_launched_in_the_player_mode_ends_at_its_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A world last played with the rule tier (or a research run adopted by ``alife play``) holds a timed rule
    effect at the checkpoint it is resumed from. The player mode decides nothing, but it ends what is due (a
    revert is not a decision): the same entries, at the same boundary, as the world resumed in ``live`` mode,
    and a replay of the log reproduces the state."""
    from alife.config import apply_overrides
    from alife.director import rules as drules
    from alife.runio import load_checkpoint
    from alife.state import state_digest, to_numpy

    monkeypatch.setitem(
        actions.MENU, "regen_pulse", dataclasses.replace(actions.MENU["regen_pulse"], duration=(20, 2000))
    )

    def prop(action: str, raw: dict[str, Any], duration: int) -> Any:
        return drules.Proposal(action, raw, duration, "scripted", "scripted")

    plan = {20: (prop("regen_pulse", {"mult": 1.4}, 60),)}  # applied at 20, its revert is due at 80
    monkeypatch.setattr(drules, "decide", lambda ctx, params=None: drules.Decision(plan.get(ctx.t, ()), ()))
    ov = [
        *("brain.weight_dtype=float32", "director.enabled=true", "director.rule_every=20"),
        *("run.metrics_every=10", "run.heavy_metrics_every=20", "run.checkpoint_every=40"),
        *("run.keep_checkpoints=20", "population.n_init=24"),
    ]
    cfg = load_config(preset="tiny", overrides=[*ov, "run.director=live"])
    rd = RunDir.create(tmp_path / "run", cfg)
    rd.update_meta(start_step=0)
    runner.run(rd, cfg, init_state(cfg), 40, quiet=True, director=make_director(rd, cfg, 0))
    monkeypatch.setattr(
        drules, "decide", lambda ctx, params=None: drules.Decision((), ())
    )  # nothing new is decided

    def resumed(mode: str) -> tuple[list[tuple[int, str, str]], float, str, RunDir]:
        import shutil

        d = tmp_path / f"resume_{mode}"
        shutil.copytree(tmp_path / "run", d)
        r = RunDir.open(d)
        c = apply_overrides(r.config(), [f"run.director={mode}"])
        st = load_checkpoint(dict(r.checkpoints())[40], c)
        r.truncate_after(40)
        res = runner.run(
            r, c, st, 160, quiet=True, director=make_director(r, c, 40, game_specs=mode == "player")
        )
        ents = [e for e in r.read_jsonl("director.jsonl") if e.get("kind") != "header"]
        return (
            [(e["step_applied"], e["source"], e["action"]) for e in ents],
            float(to_numpy(res.state.env_regen_mult)),
            state_digest(res.state),
            r,
        )

    live_entries, live_mult, live_digest, _ = resumed("live")
    assert live_entries == [(20, "rule", "regen_pulse"), (80, "revert", "regen_pulse")] and live_mult == 1.0
    entries, mult, digest, r = resumed("player")
    assert entries == live_entries and mult == 1.0 and digest == live_digest
    # the revert is a boundary record: a replay of the player-mode log applies it at its boundary
    import shutil

    cfg_r = apply_overrides(r.config(), ["run.director=replay"])
    rr = RunDir.create(tmp_path / "replay", cfg_r)
    rr.update_meta(start_step=0)
    shutil.copyfile(r.path / "director.jsonl", rr.path / "director.jsonl")
    res = runner.run(rr, cfg_r, init_state(cfg_r), 200, quiet=True, director=make_director(rr, cfg_r, 0))
    assert state_digest(res.state) == digest


# --- 4: partial rows leave the starved median ------------------------------------------------------------------


def test_the_starved_median_of_a_chapter_counts_only_the_rows_of_the_metrics_interval() -> None:
    """A quit at 7,601 leaves a row of 101 steps and one of 149: counted raw, the median fell from 7 to 6 and the
    sentence ("in the 250 steps") was about rows it did not describe."""
    rules = R.REPORT_V1

    def rows(counts: list[tuple[int, int]]) -> list[dict[str, Any]]:
        return [
            {"step": 7_000 + 250 * i, "steps": steps, "deaths_starved": n}
            for i, (steps, n) in enumerate(counts)
        ]

    full = [(250, n) for n in (7, 7, 7, 7, 11, 7, 7, 7)]
    (a,) = R._starve_fact(rows(full), rules)
    assert (a["n"], a["steps"], a["median"]) == (11, 250, 7)
    # the same trajectory cut by a stop: two short rows with few deaths, which the raw median would follow
    cut = [(250, 7), (250, 7), (101, 2), (149, 3), (50, 1), (250, 11), (30, 2), (250, 7), (60, 2)]
    (b,) = R._starve_fact(rows(cut), rules)  # the raw median of these nine rows is 3
    assert (b["n"], b["steps"], b["median"]) == (11, 250, 7)
    assert R._starve_fact(rows([(250, 7), (250, 7), (101, 2), (149, 20)]), rules) == []  # too few full rows


# --- 6: two effects of one action at one step ------------------------------------------------------------------


def test_two_effects_of_one_action_at_one_step_keep_their_own_patch_source_and_end() -> None:
    """A rule's habitat loss on patch 0 and the player's Blight on patch 1, both applied at a chapter end under
    ``--director rules``: ``ENVS.effects`` (and the alert engine reading it) told patch 1 and the player for
    both, with the other's end."""
    from alife.director import pacing

    rule = {"step_applied": 12_000, "source": "rule", "rule": "habitat", "action": "habitat_loss",
            "params": {"patch": 0}, "prev": {"gain": 1.0}, "duration": 10_000, "cost": 2}  # fmt: skip
    mine = {"step_applied": 12_000, "source": "player", "action": "habitat_loss", "params": {"patch": 1},
            "prev": {"gain": 1.0}, "duration": 4_000, "cost": 2, "power": "blight", "scope": "player"}  # fmt: skip
    for entries in ([rule, mine], [mine, rule]):
        got = play._effects(entries, pacing.fold(entries, 12_000))
        assert [(e["params"]["patch"], e["source"], e["start"], e["end"]) for e in got] == sorted(
            [(0, "rule", 12_000, 22_000), (1, "player", 12_000, 16_000)]
        ), entries


# --- 12: the arrival chapter -----------------------------------------------------------------------------------


def test_the_first_chapter_of_a_world_that_began_inside_it_has_a_notice_in_words_and_no_job(
    tmp_path: Path,
) -> None:
    """A fork or a planet that starts off the snapshot grid has an arrival chapter with no row to cut a report
    at. It got a withheld ``RPRT`` whose reason was an internal error string ("the cut 450000 is not the step of
    a metrics row"), which the card printed."""
    from alife import streamrec

    got: list[tuple[int, dict[str, Any]]] = []
    job = reportjob.ReportJob(
        tmp_path / "unused",
        length=6_000,
        n_origin=75,
        emit=lambda b, o, bl: got.append((b, o)),
        note=print,
        start_step=450_500,
    )
    job.start(456_000, None)  # the first chapter end of a world made at 450,500: the chapter began at 450,000
    job.close()
    assert job.built == [] and [b for b, _ in got] == [456_000]
    obj = got[0][1]
    assert obj["withheld"] == [reportjob.ARRIVAL_REASON] and "metrics row" not in obj["withheld"][0]
    assert (obj["a"], obj["b"], obj["chapter"]) == (450_000, 456_000, 1)
    streamrec.check_report(obj)


def test_the_pacer_gives_the_job_the_step_the_world_began_at(tmp_path: Path) -> None:
    cfg = load_config(
        preset="tiny3d",
        overrides=[
            *("brain.weight_dtype=float32", "run.metrics_every=250", "run.heavy_metrics_every=250"),
            *("run.checkpoint_every=500", "game.observe=true", "population.n_init=24"),
        ],
    )
    rd = RunDir.create(tmp_path / "w", cfg)
    rd.update_meta(start_step=130)  # the world "began" inside the first chapter of 500
    out = io.BytesIO()
    pacer = play.PlayPacer(
        out,
        H.ScriptControl([[] for _ in range(600)]),
        play.PlayOptions(rate=float("inf"), paused=False, fresh=True, unattended=True, chapter=CHAPTER),
        clock=H.FakeClock(),
        note=lambda m: None,
    )
    try:
        runner.run(rd, cfg, init_state(cfg), 600, quiet=True, pacer=pacer)
    finally:
        pacer.close()
    assert pacer.reports is not None and pacer.reports.start_step == 130 and pacer.reports.built == []
    (first,) = rprt(out).values()
    assert first["withheld"] == [reportjob.ARRIVAL_REASON] and (first["a"], first["b"]) == (0, 500)


# --- 9: the sentences of a report have unique ids --------------------------------------------------------------


def test_the_sentences_of_a_report_object_have_unique_ids_and_the_checker_refuses_a_repeat() -> None:
    """The chapter 462,000 to 468,000 of p0 tells the damper's pulse and that it "was still in place": two
    sentences of one fact. A client that lists sentences by id drew the first twice and lost the second."""
    from alife import streamrec

    view = R.load_run(Path(__file__).parent / "data" / "report_p0")
    since = {"influence": None, "alerts": {}, "ending": []}
    rec = R.narrate(R.build_sheet(view, 462000, 468000), None)
    assert [x["fact"] for x in rec["report"]["interventions"]] == [
        "F10",
        "F10",
        "F11",
    ]  # the record keeps facts
    obj = reportjob.report_object(rec, n_origin=75, length=6000, since=since)
    ids = [s["id"] for s in obj["interventions"]]
    assert ids == ["F10", "F10.2", "F11"]
    assert [s["text"] for s in obj["interventions"]][:2] == [
        x["text"] for x in rec["report"]["interventions"]
    ][:2]
    streamrec.check_report(obj)
    every = [obj["headline"]["id"], obj["lead"]["id"], *(s["id"] for s in obj["body"]), *ids]
    assert len(set(every)) == len(every)
    twice = {**obj, "interventions": [obj["interventions"][0], obj["interventions"][0]]}
    with pytest.raises(streamrec.RecordError, match="unique"):
        streamrec.check_report(twice)
