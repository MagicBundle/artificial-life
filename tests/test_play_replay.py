"""``alife play RUN --replay`` (DESIGN §33.2.8): the director's log applied, never rewritten.

A short live run of the rule tier is replayed twice from its first checkpoint, once by ``resume --director
replay`` and once by ``play --replay``: both reach the same states, and the log is the one they started from.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from alife import cli, play
from alife.runio import DIRECTOR_LOG_FILE, RunDir
from play_decode import decode
from play_helpers import checkpoint_digests, rows, same_events, snapshots

CADENCE = {"metrics_every": 10, "heavy_metrics_every": 20, "checkpoint_every": 30, "keep_checkpoints": 3}
RUN = [
    "run", "--preset", "tiny", "--set", "brain.weight_dtype=float32", "--set", "population.n_init=24",
    "--set", "director.rule_every=20", "--set", "run.heavy_metrics_every=20",
    "--set", "run.diversity_sample=64", "--set", "run.checkpoint_every=30",
    "--director", "live", "--steps", "60", "--quiet",
]  # fmt: skip


@pytest.fixture
def live(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(play, "PLAY_CADENCE", CADENCE)
    snapshots(monkeypatch, 10)
    out = tmp_path / "live"
    assert cli.main([*RUN, "--out", str(out)]) == 0
    entries = [e for e in RunDir.open(out).read_jsonl(DIRECTOR_LOG_FILE) if e.get("kind") != "header"]
    assert {e["action"] for e in entries} >= {"regen_pulse", "immigration_pulse"}  # the director acted
    return out


def test_play_replay_equals_resume_replay_and_leaves_the_log_alone(
    live: Path, tmp_path: Path, capfdbinary: pytest.CaptureFixture[bytes]
) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    shutil.copytree(live, a)
    shutil.copytree(live, b)
    log = (live / DIRECTOR_LOG_FILE).read_bytes()
    code = cli.main(
        ["resume", str(a), "--checkpoint", "0", "--director", "replay", "--steps", "60", "--quiet"]
    )
    assert code == 0
    code = cli.main(
        ["play", str(b), "--adopt", "--checkpoint", "0", "--truncate", "--replay", "--no-control",
         "--rate", "max", "--until", "60"]
    )  # fmt: skip
    cap = capfdbinary.readouterr()
    assert code == 0, cap.err.decode(errors="replace")
    assert (a / DIRECTOR_LOG_FILE).read_bytes() == log and (b / DIRECTOR_LOG_FILE).read_bytes() == log
    ra, rb = RunDir.open(a), RunDir.open(b)
    assert (
        checkpoint_digests(ra)[60] == checkpoint_digests(rb)[60] == checkpoint_digests(RunDir.open(live))[60]
    )
    assert (
        [r["step"] for r in ra.metrics_rows()]
        == [r["step"] for r in rb.metrics_rows()]
        == list(range(10, 61, 10))
    )
    assert rows(ra) == [{k: v for k, v in r.items() if k != "observers"} for r in rows(rb)]  # play observes
    assert same_events(ra, rb)
    d = decode(cap.out)
    envs = [r for r in d.records if r.tag == "ENVS" and r.f["json"]["levers"]]
    assert envs, "the replayed levers reach the ENVS record"


def test_a_replay_with_a_log_that_does_not_validate_is_refused_before_anything_is_written(
    live: Path, tmp_path: Path, capfdbinary: pytest.CaptureFixture[bytes]
) -> None:
    b = tmp_path / "b"
    shutil.copytree(live, b)
    with (b / DIRECTOR_LOG_FILE).open("a") as f:
        f.write(
            '{"step": 7, "step_applied": 7, "action": "no_such_action", "params": {}, "source": "rule"}\n'
        )
    before = {p.name: p.read_bytes() for p in b.iterdir() if p.is_file()}
    code = cli.main(["play", str(b), "--adopt", "--replay", "--no-control", "--rate", "max", "--until", "60"])
    assert code == 2
    assert {
        p.name: p.read_bytes() for p in b.iterdir() if p.is_file()
    } == before  # not even the adoption backup
    assert capfdbinary.readouterr().out == b""


def test_a_live_rule_tier_plays_at_the_play_cadence_and_refuses_another(
    live: Path, tmp_path: Path, capfdbinary: pytest.CaptureFixture[bytes], monkeypatch: pytest.MonkeyPatch
) -> None:
    b = tmp_path / "b"
    shutil.copytree(live, b)
    code = cli.main(
        ["play", str(b), "--adopt", "--director", "rules", "--no-control", "--rate", "max", "--until", "60"]
    )
    assert code == 0, capfdbinary.readouterr().err.decode(errors="replace")
    assert RunDir.open(b).read_play_json()["cadence"] == CADENCE
    # the recorded cadence is CADENCE now; with another one the rule tier is refused and nothing changes
    monkeypatch.setattr(play, "PLAY_CADENCE", {**CADENCE, "metrics_every": 20})
    before = (b / "config.json").read_bytes()
    assert cli.main(["play", str(b), "--director", "rules", "--no-control"]) == 2
    assert b"cadence" in capfdbinary.readouterr().err and (b / "config.json").read_bytes() == before


def test_a_new_world_with_the_rule_tier(
    tmp_path: Path, capfdbinary: pytest.CaptureFixture[bytes], monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--director rules`` makes the world with the director enabled and the rule tier live (LLM off); its
    decisions are in ``director.jsonl`` and reach the stream's ``ENVS`` and chronicle."""
    monkeypatch.setattr(play, "PLAY_CADENCE", CADENCE)
    snapshots(monkeypatch, 10)
    out = tmp_path / "w"
    args = [
        "play", "--preset", "tiny", "--director", "rules", "--set", "brain.weight_dtype=float32",
        "--set", "population.n_init=24", "--set", "director.rule_every=20", "--out", str(out),
        "--no-control", "--rate", "max", "--until", "60",
    ]  # fmt: skip
    assert cli.main(args) == 0
    cap = capfdbinary.readouterr()
    rd = RunDir.open(out)
    cfg = rd.config()
    assert cfg.director.enabled and cfg.run.director == "live" and cfg.director.llm_every == 0
    entries = [e for e in rd.read_jsonl(DIRECTOR_LOG_FILE) if e.get("kind") != "header"]
    assert {e["action"] for e in entries} >= {"regen_pulse", "immigration_pulse"}
    d = decode(cap.out)
    texts = [e["text"] for r in d.records if r.tag == "EVNT" for e in r.f["json"] if e["fam"] == "director"]
    assert len(texts) == len(entries) and all(t.startswith("Director: ") for t in texts)
    assert any(r.f["json"]["effects"] for r in d.records if r.tag == "ENVS")
