"""``director.jsonl`` / ``journal.jsonl``: append, fsync, torn lines, resume truncation, replay checks."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from alife import cli, runner
from alife.config import Config, apply_overrides
from alife.director import Director, actions, log, rules
from alife.director.actions import ActionError
from alife.director.rules import RULES_VERSION, Proposal
from alife.runio import DIRECTOR_ARCHIVE_FILE, DIRECTOR_LOG_FILE, JOURNAL_FILE, RunDir, read_jsonl
from alife.state import init_state
from helpers import scripted_decide

EVERY = 100


@pytest.fixture(scope="module")
def dcfg(tiny_cfg: Config) -> Config:
    return apply_overrides(
        tiny_cfg,
        [
            "director.enabled=true",
            "run.director=live",
            f"director.rule_every={EVERY}",
            "run.heavy_metrics_every=100",
            "run.checkpoint_every=100",
            "run.diversity_sample=64",
        ],
    )


def entry(step: int, action: str = "drift_change", params: dict | None = None, **kw: Any) -> dict:
    return {
        "step_applied": step,
        "source": "rule",
        "persona": "gardener",
        "action": action,
        "params": params or {"mult": 2.0},
        "prev": {"mult": 1.0},
        "duration": None,
        "reason": "test",
        **kw,
    }


def header(step: int = 0, **kw: Any) -> dict:
    return {"kind": "header", "step": step, "rules_version": RULES_VERSION, "persona": "gardener", **kw}


# --- generic JSONL helpers ---


def test_append_jsonl_appends_and_fsyncs(
    dcfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rd = RunDir.create(tmp_path / "r", dcfg)
    synced = []
    real = os.fsync
    monkeypatch.setattr(os, "fsync", lambda fd: (synced.append(fd), real(fd))[1])
    rd.append_jsonl("director.jsonl", [header(), entry(100)])
    rd.append_jsonl("director.jsonl", [entry(200)])
    rd.append_jsonl("director.jsonl", [])  # nothing to write, nothing to sync
    assert len(synced) == 2
    lines = (rd.path / "director.jsonl").read_text().splitlines()
    assert [json.loads(x).get("step_applied") for x in lines] == [None, 100, 200]
    assert rd.read_jsonl("director.jsonl") == [header(), entry(100), entry(200)]
    assert rd.read_jsonl("absent.jsonl") == []


def test_a_torn_last_line_is_ignored_but_a_torn_middle_line_is_an_error(dcfg: Config, tmp_path: Path) -> None:
    rd = RunDir.create(tmp_path / "r", dcfg)
    path = rd.path / DIRECTOR_LOG_FILE
    path.write_text(json.dumps(header()) + "\n" + json.dumps(entry(100)) + '\n{"step_applied": 2')
    assert [r.get("step_applied") for r in rd.read_jsonl(DIRECTOR_LOG_FILE)] == [None, 100]
    path.write_text(json.dumps(header()) + '\n{"step_app\n' + json.dumps(entry(100)) + "\n")
    with pytest.raises(json.JSONDecodeError):
        rd.read_jsonl(DIRECTOR_LOG_FILE)
    with pytest.raises(log.LogError, match="not valid JSON"):
        log.read_log(path)
    with pytest.raises(log.LogError, match="no director log"):
        log.read_log(tmp_path / "nope.jsonl")


# --- resume truncation ---


def write_director_files(rd: RunDir, as_of: int) -> dict[str, bytes]:
    recs = [header(0), entry(100), entry(200), header(250, persona="trickster"), entry(300), entry(400)]
    rd.append_jsonl(DIRECTOR_LOG_FILE, recs)
    rd.append_jsonl(JOURNAL_FILE, [{"step": s, "text": f"t{s}"} for s in (100, 200, 300, 400)])
    rd.write_json(DIRECTOR_ARCHIVE_FILE, {"as_of_step": as_of, "entries": []})
    return {n: (rd.path / n).read_bytes() for n in (DIRECTOR_LOG_FILE, JOURNAL_FILE, DIRECTOR_ARCHIVE_FILE)}


def test_live_resume_truncates_the_director_files_to_the_checkpoint(dcfg: Config, tmp_path: Path) -> None:
    rd = RunDir.create(tmp_path / "r", dcfg)
    write_director_files(rd, as_of=400)
    with open(rd.path / DIRECTOR_LOG_FILE, "a") as f:
        f.write('{"step_applied": 5')  # a torn last line
    rd.truncate_after(250)
    kept = rd.read_jsonl(DIRECTOR_LOG_FILE)
    assert [r.get("step_applied", r.get("step")) for r in kept] == [
        0,
        100,
        200,
        250,
    ]  # headers cut by step too
    assert kept[-1]["persona"] == "trickster"
    assert [r["step"] for r in rd.read_jsonl(JOURNAL_FILE)] == [100, 200]
    assert not (
        rd.path / DIRECTOR_ARCHIVE_FILE
    ).exists()  # taken after the checkpoint: analysis only, dropped
    assert (rd.path / DIRECTOR_LOG_FILE).read_text().endswith("\n")  # rewritten whole: ready to append to
    rd.truncate_after(200)
    assert [r.get("step_applied", r.get("step")) for r in rd.read_jsonl(DIRECTOR_LOG_FILE)] == [0, 100, 200]


def test_an_archive_taken_at_or_before_the_checkpoint_is_kept(dcfg: Config, tmp_path: Path) -> None:
    rd = RunDir.create(tmp_path / "r", dcfg)
    write_director_files(rd, as_of=200)
    rd.truncate_after(250)
    assert json.loads((rd.path / DIRECTOR_ARCHIVE_FILE).read_text())["as_of_step"] == 200
    (rd.path / DIRECTOR_ARCHIVE_FILE).write_text("{not json")
    rd.truncate_after(250)  # an unreadable archive is just dropped
    assert not (rd.path / DIRECTOR_ARCHIVE_FILE).exists()


def test_a_replay_never_truncates_the_director_files(dcfg: Config, tmp_path: Path) -> None:
    rd = RunDir.create(tmp_path / "r", dcfg)
    rd.append_metrics([{"step": s, "steps": 100} for s in (100, 200, 300)])
    before = write_director_files(rd, as_of=400)
    rd.truncate_after(150, keep_director=True)
    assert {n: (rd.path / n).read_bytes() for n in before} == before
    assert [r["step"] for r in rd.metrics_rows()] == [100]  # the rest of the run directory is still cut


def test_truncation_without_director_files_is_a_no_op(dcfg: Config, tmp_path: Path) -> None:
    rd = RunDir.create(tmp_path / "r", dcfg)
    rd.truncate_after(10)
    assert not any((rd.path / n).exists() for n in (DIRECTOR_LOG_FILE, JOURNAL_FILE, DIRECTOR_ARCHIVE_FILE))


# --- header ---


def test_headers_record_the_policy_and_are_not_repeated_unchanged(dcfg: Config) -> None:
    h = log.make_header(dcfg, 500, 2)
    assert h["kind"] == "header" and h["rules_version"] == RULES_VERSION and h["rule_every"] == EVERY
    assert (h["persona"], h["goal"], h["budget_per_100k"], h["seed"], h["step"]) == (
        "gardener",
        "",
        2,
        0,
        500,
    )
    assert not log.needs_header([h], log.make_header(dcfg, 900, 2))  # the start step does not count
    assert log.needs_header([], h)
    other = apply_overrides(dcfg, ["director.persona=trickster"])
    assert log.needs_header([h, entry(100)], log.make_header(other, 900, 6))
    assert log.needs_header([h], log.make_header(dcfg, 500, 3))  # another budget
    assert log.entries_of([h, entry(100), h]) == [entry(100)]


# --- validation ---


def revert_pair(step: int) -> list[dict]:
    hab = entry(
        step, "habitat_loss", {"patch": 1, "gain": 0.0}, prev={"patch": 1, "gain": 1.0}, duration=10_000
    )
    back = entry(
        step + 10_000, "habitat_loss", {"patch": 1, "gain": 1.0}, source="revert", prev={}, duration=None
    )
    return [hab, back]


def test_a_valid_log_passes(dcfg: Config) -> None:
    recs = [header(), entry(100), *revert_pair(200), entry(10_200, params={"mult": 5.0})]
    log.validate_log(recs, dcfg)
    log.validate_log([], dcfg)
    log.validate_log([header()], dcfg)


@pytest.mark.parametrize(
    ("records", "message"),
    [
        ([entry(100, action="fireball")], "unknown action"),
        ([entry(150)], "multiple of rule_every"),
        ([entry(100, params={"mult": 99.0})], "outside"),
        ([entry(200), entry(100)], "comes after"),
        ([{"kind": "footer"}], "unknown kind"),
        (["not a record"], "not an object"),
        ([entry(100, source="oracle")], "unknown source"),
        (
            [entry(100, "habitat_loss", {"patch": 1, "gain": 1.0}, source="revert", prev={}, duration=None)],
            "no active effect",
        ),
        ([*revert_pair(100), revert_pair(100)[1]], "no active effect"),  # reverted twice
    ],
)
def test_validation_rejects_a_bad_log(dcfg: Config, records: list, message: str) -> None:
    with pytest.raises((log.LogError, ActionError), match=message):
        log.validate_log(records, dcfg)


def test_a_log_made_for_another_cadence_is_rejected(dcfg: Config) -> None:
    slower = apply_overrides(dcfg, ["director.rule_every=300"])
    with pytest.raises(ActionError, match="multiple of rule_every 300"):
        log.validate_log([entry(100)], slower)


# --- the live writer ---


def test_entries_are_durable_before_the_checkpoint_and_the_header_leads(
    dcfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = {100: [Proposal("drift_change", {"mult": 3.0}, None, "test", "scripted")]}
    monkeypatch.setattr("alife.director.rules.decide", scripted_decide(plan))
    rd = RunDir.create(tmp_path / "r", dcfg)
    events: list[str] = []
    real_append, real_ckpt = rd.append_jsonl, rd.save_checkpoint
    rd.append_jsonl = lambda name, recs: (events.append(f"append {name}"), real_append(name, recs))[1]  # type: ignore[method-assign]
    rd.save_checkpoint = lambda s, c: (events.append("checkpoint"), real_ckpt(s, c))[1]  # type: ignore[method-assign]
    director = Director(rd, dcfg, "live", 0)
    res = runner.run(rd, dcfg, init_state(dcfg), 200, quiet=True, director=director)
    assert res.status == "done"
    recs = read_jsonl(rd.path / DIRECTOR_LOG_FILE)
    assert recs[0]["kind"] == "header" and [r["step_applied"] for r in recs[1:]] == [100]
    first_entry = events.index(f"append {DIRECTOR_LOG_FILE}", 1)  # after the header write
    ckpt_100 = [i for i, e in enumerate(events) if e == "checkpoint"][1]  # the step-0 one comes first
    assert first_entry < ckpt_100
    e = recs[1]
    assert (
        e["source"] == "rule" and e["rule"] == "test" and e["vitals_step"] == 100 and "novelty" in e["vitals"]
    )
    assert e["params"] == {"mult": 3.0} and e["prev"] == {"mult": 1.0} and e["persona"] == "gardener"
    j = read_jsonl(rd.path / JOURNAL_FILE)
    assert [x["step"] for x in j] == [100] and "drift_change" in j[0]["text"]
    assert json.loads((rd.path / DIRECTOR_ARCHIVE_FILE).read_text())["as_of_step"] == 200


def test_a_policy_change_on_resume_appends_a_header(dcfg: Config, tmp_path: Path) -> None:
    rd = RunDir.create(tmp_path / "r", dcfg)
    Director(rd, dcfg, "live", 0)
    Director(rd, dcfg, "live", 0)  # unchanged: no second header
    trick = apply_overrides(dcfg, ["director.persona=trickster"])
    Director(rd, trick, "live", 300)
    heads = [r for r in read_jsonl(rd.path / DIRECTOR_LOG_FILE) if r["kind"] == "header"]
    assert [(h["step"], h["persona"], h["budget_per_100k"]) for h in heads] == [
        (0, "gardener", 2),
        (300, "trickster", 6),
    ]


def test_a_replay_director_writes_nothing(dcfg: Config, tmp_path: Path) -> None:
    rd = RunDir.create(tmp_path / "r", dcfg)
    rd.append_jsonl(DIRECTOR_LOG_FILE, [header(), entry(100)])
    before = (rd.path / DIRECTOR_LOG_FILE).read_bytes()
    d = Director(rd, apply_overrides(dcfg, ["run.director=replay"]), "replay", 0)
    d.on_checkpoint(100)
    assert (rd.path / DIRECTOR_LOG_FILE).read_bytes() == before
    assert not (rd.path / JOURNAL_FILE).exists() and not (rd.path / DIRECTOR_ARCHIVE_FILE).exists()
    assert d.due(200) and not d.due(150)
    with pytest.raises(ValueError, match="mode"):
        Director(rd, dcfg, "off", 0)
    assert actions.MENU  # the menu the log is validated against


# --- the CLI refuses a bad log before touching anything (exit 2) ---


@pytest.fixture(scope="module")
def source(tmp_path_factory: pytest.TempPathFactory, dcfg: Config) -> Path:
    """A run directory holding only the step-0 checkpoint of a director-enabled run."""
    cfg = apply_overrides(dcfg, ["run.director=off"])
    rd = RunDir.create(tmp_path_factory.mktemp("src") / "src", cfg)
    rd.save_checkpoint(init_state(cfg), cfg)
    return rd.path


def write_log(path: Path, records: list) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return path


@pytest.mark.parametrize(
    "records",
    [
        [entry(100, action="fireball")],
        [entry(150)],
        [entry(100, params={"mult": 99.0})],
        [
            entry(
                100,
                "fertility_shift",
                {"patch": 0, "x": 5.0, "y": 5.0},
                prev={"patch": 0, "x": 90.0, "y": 90.0},
            )
        ],
    ],
    ids=["unknown action", "off the cadence", "out of bounds", "move beyond L/4"],
)
def test_fork_replay_with_a_bad_log_exits_2_and_creates_nothing(
    source: Path, tmp_path: Path, records: list
) -> None:
    bad = write_log(tmp_path / "bad.jsonl", [header(), *records])
    out = tmp_path / "out"
    argv = ["run", "--init-from", f"{source}:0", "--director", "replay", "--replay-log", str(bad)]
    argv += ["--set", f"director.rule_every={EVERY}", "--steps", "100", "--quiet", "--out", str(out)]
    assert cli.main(argv) == 2
    assert not out.exists()


def test_replay_flag_combinations_are_config_errors(source: Path, tmp_path: Path) -> None:
    ok = write_log(tmp_path / "ok.jsonl", [header(), entry(100)])
    base = [
        "run",
        "--init-from",
        f"{source}:0",
        "--set",
        f"director.rule_every={EVERY}",
        "--steps",
        "100",
        "--quiet",
    ]
    assert cli.main([*base, "--director", "replay", "--out", str(tmp_path / "a")]) == 2  # no --replay-log
    assert cli.main([*base, "--replay-log", str(ok), "--out", str(tmp_path / "b")]) == 2  # no replay mode
    assert cli.main([*base, "--director", "live", "--replay-log", str(ok), "--out", str(tmp_path / "c")]) == 2
    assert (
        cli.main(
            [*base, "--director", "replay", "--replay-log", str(tmp_path / "x"), "--out", str(tmp_path / "d")]
        )
        == 2
    )
    assert cli.main(["run", "--preset", "tiny", "--director", "replay", "--out", str(tmp_path / "e")]) == 2
    assert not any((tmp_path / n).exists() for n in "abcde")


def test_resume_replay_validates_before_it_truncates_and_never_edits_the_log(
    dcfg: Config, tmp_path: Path
) -> None:
    cfg = apply_overrides(dcfg, ["run.director=off"])
    rd = RunDir.create(tmp_path / "r", cfg)
    runner.run(rd, cfg, init_state(cfg), 100, quiet=True)
    bad = write_log(rd.path / DIRECTOR_LOG_FILE, [header(), entry(100), entry(150)])
    snapshot = {p: p.read_bytes() for p in (rd.path / "metrics.jsonl", rd.path / "config.json", bad)}
    assert cli.main(["resume", str(rd.path), "--director", "replay", "--checkpoint", "0", "--quiet"]) == 2
    assert {p: p.read_bytes() for p in snapshot} == snapshot  # nothing truncated, nothing rewritten
    (rd.path / DIRECTOR_LOG_FILE).unlink()
    assert cli.main(["resume", str(rd.path), "--director", "replay", "--quiet"]) == 2  # no log at all


# --- rules versions (Review-log row I43) ---


def relabel(rd: RunDir, version: int | None) -> None:
    """Rewrite the version of every header of the run's log, as if an older (or newer) alife had written it
    (``None`` drops the key: a header that predates it)."""
    recs = rd.read_jsonl(DIRECTOR_LOG_FILE)
    for r in recs:
        if r.get("kind") == log.HEADER_KIND:
            r.pop("rules_version", None)
            if version is not None:
                r["rules_version"] = version
    (rd.path / DIRECTOR_LOG_FILE).write_text("".join(json.dumps(r) + "\n" for r in recs))


def headers_of(rd: RunDir) -> list[dict]:
    return [r for r in rd.read_jsonl(DIRECTOR_LOG_FILE) if r.get("kind") == log.HEADER_KIND]


def test_a_new_log_takes_the_current_rules_version(dcfg: Config, tmp_path: Path) -> None:
    rd = RunDir.create(tmp_path / "r", dcfg)
    d = Director(rd, dcfg, "live", 0)
    (h,) = headers_of(rd)
    assert h["rules_version"] == RULES_VERSION == 2
    assert d.params.rules_version == 2 and d.params.season_aware


def test_a_resumed_run_keeps_the_rules_version_of_its_log(
    dcfg: Config, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A run begun under rules version 1 is not switched to 2 by a resume: its rules never change mid-run, no
    second header is appended for it, and stderr says which rules apply. A policy change on the same resume
    appends a header that still names version 1."""
    rd = RunDir.create(tmp_path / "r", dcfg)
    rd.append_jsonl(DIRECTOR_LOG_FILE, [log.make_header(dcfg, 0, 2, rules_version=1)])
    d = Director(rd, dcfg, "live", 100)
    assert d.params.rules_version == 1 and not d.params.season_aware
    assert [h["rules_version"] for h in headers_of(rd)] == [1]
    err = capsys.readouterr().err
    assert "rules version 1" in err and "current version is 2" in err
    trick = apply_overrides(dcfg, ["director.persona=trickster"])
    assert Director(rd, trick, "live", 200).params.rules_version == 1
    assert [(h["persona"], h["rules_version"]) for h in headers_of(rd)] == [("gardener", 1), ("trickster", 1)]
    # a log whose latest header says 2 (a fork of a version-1 log, say) lives by 2
    rd.append_jsonl(DIRECTOR_LOG_FILE, [log.make_header(trick, 300, 6, rules_version=2)])
    assert Director(rd, trick, "live", 300).params.rules_version == 2
    capsys.readouterr()
    # a header that predates the key is version 1 (no header is appended for it: it says what version 1 says);
    # a replay never changes anything and says nothing
    relabel(rd, None)
    n_headers = len(headers_of(rd))
    assert Director(rd, trick, "live", 300).params.rules_version == 1
    assert len(headers_of(rd)) == n_headers and all("rules_version" not in h for h in headers_of(rd))
    assert log.needs_header(headers_of(rd), log.make_header(trick, 300, 6, rules_version=2))  # 2 is a change
    assert Director(rd, apply_overrides(dcfg, ["run.director=replay"]), "replay", 0).params.rules_version == 2


def test_validation_accepts_every_implemented_version_and_refuses_the_rest(dcfg: Config) -> None:
    for v in (1, 2):
        log.validate_log([header(rules_version=v), entry(100)], dcfg)  # a replay never decides: all validate
    bare = {k: v for k, v in header().items() if k != "rules_version"}
    log.validate_log([bare, entry(100)], dcfg)  # written before the key: version 1
    assert log.rules_version_of([bare]) == 1 and log.rules_version_of([entry(100)]) is None
    assert log.rules_version_of([header(rules_version=1), entry(100), header(rules_version=2)]) == 2
    for bad in (0, 3, "2", None, True, 2.5):
        with pytest.raises(log.LogError, match="rules_version"):
            log.validate_log([header(rules_version=bad)], dcfg)
    with pytest.raises(log.LogError, match="record 2"):  # a later header is checked too
        log.validate_log([header(), entry(100), header(rules_version=3)], dcfg)


def test_a_live_director_refuses_a_log_of_rules_it_does_not_implement(dcfg: Config, tmp_path: Path) -> None:
    rd = RunDir.create(tmp_path / "r", dcfg)
    rd.append_jsonl(DIRECTOR_LOG_FILE, [log.make_header(dcfg, 0, 2, rules_version=3)])
    with pytest.raises(log.LogError, match="rules_version 3"):
        Director(rd, dcfg, "live", 100)


def run_tree(rd: RunDir) -> dict[str, bytes]:
    """Every file of a run directory (relative path -> bytes): checkpoints, logs, archive, config."""
    return {str(p.relative_to(rd.path)): p.read_bytes() for p in sorted(rd.path.rglob("*")) if p.is_file()}


def test_a_live_resume_refuses_unknown_rules_before_touching_anything_and_a_known_one_continues(
    dcfg: Config, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The refusal is the CLI's own pre-check, not the director's later one: the resume is from the checkpoint
    at 100 of a run that went on to 200, so a resume that got as far as truncating would cut the logs, delete
    the archive (``as_of_step`` 200) and drop the checkpoint at 200. Not one byte of the directory changes."""
    rd = RunDir.create(tmp_path / "r", dcfg)
    res = runner.run(rd, dcfg, init_state(dcfg), 200, quiet=True, director=Director(rd, dcfg, "live", 0))
    assert res.status == "done"
    relabel(rd, 3)
    before = run_tree(rd)
    assert DIRECTOR_ARCHIVE_FILE in before and len(rd.checkpoints()) >= 2
    resume = ["resume", str(rd.path), "--checkpoint", "100", "--until", "300", "--quiet"]
    assert cli.main(resume) == 2
    assert run_tree(rd) == before  # same files, same bytes: not truncated, not rewritten, nothing deleted
    assert "rules_version 3" in capsys.readouterr().err
    assert cli.main(["resume", str(rd.path), "--director", "replay", "--quiet"]) == 2  # nor does a replay
    assert run_tree(rd) == before
    # version 1 is known: the resume goes ahead under it, with no new header
    relabel(rd, 1)
    assert cli.main(resume) == 0
    assert [h["rules_version"] for h in headers_of(rd)] == [1]
    assert "rules version 1" in capsys.readouterr().err
    assert max(r["step"] for r in RunDir.open(rd.path).metrics_rows()) == 300


def test_a_resumed_live_run_notes_a_persistent_blocked_rule_once(
    dcfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End to end: a rule blocked at every boundary is journalled once, however the run is split by resumes
    (it was noted again at the first boundary after every resume, Review-log row I43). Both the journal and
    the archive file, which carries the set, equal the uninterrupted run's."""
    note = "extinction_rescue (n/K_est 0.1, fall 0.0): blocked [regen_pulse: regen_pulse is cooling down]"
    monkeypatch.setattr("alife.director.rules.decide", lambda ctx, params=None: rules.Decision((), (note,)))
    whole = RunDir.create(tmp_path / "whole", dcfg)
    runner.run(whole, dcfg, init_state(dcfg), 300, quiet=True, director=Director(whole, dcfg, "live", 0))
    split = RunDir.create(tmp_path / "split", dcfg)
    runner.run(split, dcfg, init_state(dcfg), 100, quiet=True, director=Director(split, dcfg, "live", 0))
    assert cli.main(["resume", str(split.path), "--until", "200", "--quiet"]) == 0
    assert cli.main(["resume", str(split.path), "--until", "300", "--quiet"]) == 0
    assert [ln["step"] for ln in read_jsonl(whole.path / JOURNAL_FILE)] == [100]
    for name in (JOURNAL_FILE, DIRECTOR_LOG_FILE, DIRECTOR_ARCHIVE_FILE):
        assert (whole.path / name).read_bytes() == (split.path / name).read_bytes(), name
    assert json.loads((split.path / DIRECTOR_ARCHIVE_FILE).read_text())["blocked"] == ["extinction_rescue"]
