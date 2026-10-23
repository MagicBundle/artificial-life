"""The pacer's side of the kin threads (DESIGN §34.1): the table is saved with every checkpoint the loop
writes, read back at ``bind`` (a missing or unusable file leaves the refold of the social log), and the cast
tables the client gets carry one name and one thread per group."""

from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from alife import play, runio
from alife.game.chronicle import Chronicler
from alife.game.kinthreads import KinThreads
from alife.runio import PLAY_STATE_NAMES
from play_helpers import FakeClock, ScriptControl, run_paced, snapshots, world
from test_play_chronicle import ALD, BRE, SALT, record

RUN = play.PlayOptions(rate=float("inf"), paused=False, fresh=True, unattended=True)


def bare_pacer(notes: list[str] | None = None) -> play.PlayPacer:
    p = play.PlayPacer(
        io.BytesIO(), ScriptControl([]), play.PlayOptions(), clock=FakeClock(),
        note=(notes if notes is not None else []).append,
    )  # fmt: skip
    p.chron = Chronicler(SALT)
    return p


def test_the_table_is_saved_with_every_checkpoint_the_loop_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshots(monkeypatch, 10)
    rd, _, pacer, _ = run_paced(world("2d"), tmp_path / "w", 60, [], options=RUN)
    saved = [st for st, _ in rd.play_state_files("kinthreads")]
    assert saved == [30, 60], (
        "with the checkpoints of the loop (not the initial one: its table is an empty fold)"
    )
    final = rd.read_play_state("kinthreads", 60)
    assert final == json.loads(json.dumps(pacer.chron.threads.state()))
    assert final["k"] == 5 and final["step"] == 60 and final["format"] == "alife-kinthreads"
    assert KinThreads.from_state(final).state() == final
    assert (rd.observers_dir / "kinthreads_0000000030.json").is_file()


def test_a_world_without_observers_saves_no_table(tmp_path: Path) -> None:
    rd, *_ = run_paced(world("2d", observe=False), tmp_path / "w", 40, [], options=RUN)
    assert rd.play_state_files("kinthreads") == []


def test_bind_takes_the_saved_table_when_it_is_usable() -> None:
    p = bare_pacer()
    first = Chronicler(SALT)
    for step in (250, 500):
        first.from_record(record(step, kin=[ALD, BRE]))
    saved = json.loads(json.dumps(first.threads.state()))
    p._t = 500
    seen: list[tuple[str, int]] = []

    def read(name: str, step: int) -> dict[str, Any]:
        seen.append((name, step))
        return saved

    p._restore_threads(SimpleNamespace(read_play_state=read))  # type: ignore[arg-type]
    assert seen == [("kinthreads", 500)] and p.chron.threads.state() == saved


@pytest.mark.parametrize("bad", ["missing", "unreadable", "foreign"])
def test_an_unusable_table_leaves_the_refold_and_says_so(bad: str) -> None:
    notes: list[str] = []
    p = bare_pacer(notes)
    p.chron.from_record(record(250, kin=[ALD]))
    before = p.chron.threads.state()
    p._t = 250

    def read(name: str, step: int) -> dict[str, Any] | None:
        if bad == "missing":
            return None
        if bad == "unreadable":
            raise json.JSONDecodeError("nope", "x", 0)
        return {"format": "alife-kinthreads", "version": 99}

    p._restore_threads(SimpleNamespace(read_play_state=read))  # type: ignore[arg-type]
    assert p.chron.threads.state() == before
    assert len(notes) == (0 if bad == "missing" else 1) and all("refolded" in n for n in notes)


def test_the_cast_tables_carry_the_thread_names() -> None:
    p = bare_pacer()
    p.chron.from_record(record(250, kin=[ALD, BRE]))
    p.chron.from_record(record(500, kin=[ALD]))
    p.chron.from_record(record(750, kin=[ALD, (9, "Renamed", 80, (0.5, 0.5, 0.0, 0.0))]))
    tables = {
        "houses": [],
        "kin": [
            {"id": 1, "name": "Aldric", "slot": 0},
            {"id": 9, "name": "Renamed", "slot": 1},
            {"id": 77, "name": "Stranger", "slot": 2},
        ],
        "colonies": [],
    }
    out = p._thread_names(tables)
    assert [(k["id"], k["name"], k.get("thread")) for k in out["kin"]] == [
        (1, "Aldric", 1),
        (9, "Brennar", 2),  # back under a new id and name: the thread's own
        (77, "Stranger", None),  # unknown to the fold: unchanged, no thread
    ]
    assert tables["kin"][1]["name"] == "Renamed", "the observer's table is not modified"
    assert out["houses"] is tables["houses"] and out["colonies"] is tables["colonies"]


def test_the_state_files_are_the_play_states_runio_knows() -> None:
    assert PLAY_STATE_NAMES == ("storms", "kinthreads") and runio.PLAY_STATE_NAMES is PLAY_STATE_NAMES
