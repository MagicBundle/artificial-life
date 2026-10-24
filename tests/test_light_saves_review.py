"""Light saves, review fixes (DESIGN §34.11, I64): the data-safety cases of the compressed save path.

Each test asserts the behaviour the review asked for; the ones that reproduce a finding fail on dbb62c7."""

from __future__ import annotations

import errno
import hashlib
import os
import threading
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest

from alife import play, runio
from alife.config import load_config
from alife.runio import LATEST_FILE, RunDir, checkpoint_metadata, load_checkpoint, state_file_is_complete
from alife.state import init_state

CFG = load_config(
    preset="tiny",
    overrides=[
        "population.n_max=64",
        "population.max_births_per_step=8",
        "population.n_init=8",
        "world.size=48",
        "brain.weight_dtype=float32",
        "run.keep_checkpoints=3",
    ],
)
S0 = init_state(CFG)


class Kill(BaseException):
    """A kill as the code sees it: not an Exception, so the save path's handlers do not take it."""


def at(step: int):
    return S0._replace(step=mx.array(step, dtype=mx.int32))


def world(tmp_path: Path, name: str = "w", compress: bool = True, mode: str = "all") -> RunDir:
    rd = RunDir.create(tmp_path / name, CFG)
    rd.compress = compress
    rd.set_save_mode(mode)
    return rd


def save(rd: RunDir, step: int) -> None:
    rd.save_checkpoint(at(step), CFG)
    rd.wait_saves()


def steps_of(rd: RunDir) -> list[int]:
    return [st for st, _ in rd.checkpoints()]


def test_F1_latest_never_moves_over_a_save_that_fails_its_check(tmp_path, monkeypatch):
    rd = world(tmp_path, mode="single")
    save(rd, 10)
    real = runio.state_file_is_complete
    monkeypatch.setattr(
        runio, "state_file_is_complete", lambda p, st: False if "20" in Path(p).name else real(p, st)
    )
    save(rd, 20)
    latest = (rd.checkpoint_dir / LATEST_FILE).read_text().strip()
    assert latest == "step_0000000010.safetensors.zst", (
        f"LATEST names {latest}, which failed the whole-file check"
    )
    assert not list(rd.checkpoint_dir.glob("*step_0000000020*"))


def test_M2_a_compressed_save_that_fails_its_check_never_gets_its_name(tmp_path, monkeypatch):
    rd = world(tmp_path, mode="all")
    save(rd, 10)
    real = runio.state_file_is_complete
    monkeypatch.setattr(
        runio,
        "state_file_is_complete",
        lambda p, st: False if Path(p).name.startswith("step_0000000020.safetensors.zst") else real(p, st),
    )
    save(rd, 20)
    assert (rd.checkpoint_dir / LATEST_FILE).read_text().strip() == "step_0000000020.safetensors"
    assert not (rd.checkpoint_dir / "step_0000000020.safetensors.zst").exists()
    assert not list(rd.checkpoint_dir.glob("*.tmp"))
    assert load_checkpoint(rd.latest_checkpoint(), CFG) is not None


def test_F2_no_partial_plain_tmp_after_disk_full_in_the_synchronous_write(tmp_path, monkeypatch):
    rd = world(tmp_path)
    save(rd, 10)

    def full(f, arrays, meta=None):
        f.write(b"x" * 100000)
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(mx, "save_safetensors", full)
    with pytest.raises(OSError):
        rd.save_checkpoint(at(20), CFG)
    leftovers = [p.name for p in rd.checkpoint_dir.iterdir() if p.name.endswith(".tmp")]
    assert not leftovers, f"partial file left on a full disk: {leftovers}"


def test_F3_the_compress_worker_puts_the_new_save_on_permanent_storage_before_deleting_older_ones(
    tmp_path, monkeypatch
):
    calls = []
    real = runio._full_sync
    monkeypatch.setattr(runio, "_full_sync", lambda p: (calls.append(Path(p).name), real(p)))
    rd = world(tmp_path, mode="single")
    save(rd, 10)
    save(rd, 20)
    assert any(n.endswith(".zst") for n in calls), f"_full_sync calls: {calls}"


def test_F4_truncate_after_removes_both_forms_of_a_later_step(tmp_path):
    rd = world(tmp_path, compress=False)
    save(rd, 10)
    save(rd, 20)
    rd.compress = True
    save(rd, 20)
    rd.truncate_after(10)
    assert steps_of(rd) == [10]


def test_F5_a_stale_hard_linked_tmp_is_never_written_into(tmp_path):
    src = world(tmp_path, "src", compress=False)
    save(src, 10)
    srcf = src.checkpoint_dir / "step_0000000010.safetensors"
    before = hashlib.sha256(srcf.read_bytes()).hexdigest()
    w = world(tmp_path, "dst", compress=False)
    w.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    os.link(
        srcf, w.checkpoint_dir / "step_0000000010.safetensors.tmp"
    )  # a kill between os.link and os.replace
    w.save_checkpoint(at(10), CFG)
    assert hashlib.sha256(srcf.read_bytes()).hexdigest() == before, (
        "the source world's save was rewritten in place"
    )


def test_F6_a_planet_whose_source_save_vanished_still_gets_its_origin(tmp_path):
    src = world(tmp_path, "src", compress=False)
    save(src, 10)
    srcf = src.checkpoint_dir / "step_0000000010.safetensors"
    w = world(tmp_path, "dst")
    w.origin_link = (10, srcf)
    srcf.unlink()
    w.save_checkpoint(at(10), CFG)
    w.wait_saves()
    assert w.latest_checkpoint() is not None
    assert load_checkpoint(w.latest_checkpoint(), CFG) is not None


def test_H1_a_new_world_has_a_loadable_save_while_its_first_compress_runs(tmp_path, monkeypatch):
    gate = threading.Event()
    real_compress = runio.compress_file

    def held(src, dst, **kw):
        gate.wait(20)
        return real_compress(src, dst, **kw)

    monkeypatch.setattr(runio, "compress_file", held)
    rd = world(tmp_path)
    rd.save_checkpoint(at(0), CFG)  # the origin of the world
    try:
        back = RunDir.open(rd.path)  # what a kill or a quit at this moment would leave
        newest = back.latest_checkpoint()
        assert newest is not None, "a new world without a loadable save while its origin compresses"
        assert state_file_is_complete(newest, 0)
        assert checkpoint_metadata(newest)["step"] == "0"
    finally:
        gate.set()
        rd.wait_saves()
    assert rd.latest_checkpoint().name == "step_0000000000.safetensors.zst"


def test_H2_a_signal_during_the_wait_for_a_compress_does_not_end_it_early(tmp_path, monkeypatch):
    """The signal lands inside the wait by construction (a worker proxy raises KeyboardInterrupt on the first
    join), not by a timer: the compress is held until the wait has gone on after the signal."""
    real_compress = runio.compress_file
    release = threading.Event()

    def held(src, dst, **kw):
        release.wait(20)  # a safety net only: the test releases it once the wait has resumed
        return real_compress(src, dst, **kw)

    rd = world(tmp_path, mode="single")
    save(rd, 10)  # a whole save first, with the real compress
    monkeypatch.setattr(runio, "compress_file", held)
    rd.save_checkpoint(at(20), CFG)

    class SignalDuringTheWait:
        """The save's worker as the wait sees it: the first join is the signal, later joins are real."""

        def __init__(self, thread: threading.Thread) -> None:
            self.thread = thread
            self.signalled = False

        def join(self, timeout: float | None = None) -> None:
            if not self.signalled:
                self.signalled = True
                raise KeyboardInterrupt  # the signal, delivered in the wait
            release.set()  # the wait went on after the signal: the save may finish
            self.thread.join(timeout)

        def is_alive(self) -> bool:
            return self.thread.is_alive()

    worker = rd._save_worker
    assert worker is not None and worker.is_alive(), "the compress is in flight"
    rd._save_worker = SignalDuringTheWait(worker)  # type: ignore[assignment]
    with pytest.raises(KeyboardInterrupt):  # raised once the save is whole, after the wait
        rd.wait_saves()
    assert not worker.is_alive(), "the wait ended only when the save did"
    assert rd.latest_checkpoint().name == "step_0000000020.safetensors.zst", "the quit save was lost"
    assert steps_of(rd) == [20]


def test_M1_a_kill_between_a_save_and_its_latest_is_adopted_on_resume(tmp_path, monkeypatch):
    rd = world(tmp_path, mode="single")
    save(rd, 10)
    real = runio._atomic_write_text

    def kill_at_20(path, text):
        if text.startswith("step_0000000020"):
            raise Kill()
        return real(path, text)

    monkeypatch.setattr(runio, "_atomic_write_text", kill_at_20)
    with pytest.raises(Kill):
        rd.save_checkpoint(at(20), CFG)
    monkeypatch.undo()
    back = RunDir.open(rd.path)
    assert int(checkpoint_metadata(back.latest_checkpoint())["step"]) == 10  # LATEST is behind
    assert back.adopt_newest_save() is not None  # what _play_resume does before it checks for --truncate
    assert int(checkpoint_metadata(back.latest_checkpoint())["step"]) == 20
    assert load_checkpoint(back.latest_checkpoint(), CFG) is not None


def test_F7_a_killed_worker_leaves_a_world_play_resumes_at_its_newest_save(tmp_path, monkeypatch):
    monkeypatch.setattr(
        threading, "excepthook", lambda args: None
    )  # the worker's death is the test, not a report
    rd = world(tmp_path, mode="single")
    save(rd, 10)

    def killed(src, dst, **kw):
        raise Kill()

    monkeypatch.setattr(runio, "compress_file", killed)
    rd.save_checkpoint(at(20), CFG)
    rd.wait_saves()  # returns: the worker is done, dead of the kill
    monkeypatch.undo()
    back = RunDir.open(rd.path)
    newest = max(steps_of(back))
    assert int(checkpoint_metadata(back.latest_checkpoint())["step"]) == newest == 20
    assert load_checkpoint(back.latest_checkpoint(), CFG) is not None


def test_M6a_a_tmp_that_is_a_hard_link_to_a_save_is_not_partly_rewritten(tmp_path):
    rd = world(tmp_path, compress=False)
    save(rd, 10)
    plain = rd.checkpoint_dir / "step_0000000010.safetensors"
    link = rd.checkpoint_dir / "step_0000000030.safetensors.tmp"
    os.link(plain, link)
    before = hashlib.sha256(plain.read_bytes()).hexdigest()
    rd.save_checkpoint(at(30), CFG)
    assert hashlib.sha256(plain.read_bytes()).hexdigest() == before


class _Rec:
    def __init__(self, fail: Exception | None = None) -> None:
        self.calls: list[tuple] = []
        self.fail = fail

    def wait_saves(self) -> None:
        if self.fail is not None:
            raise self.fail

    def save(self, rd, t) -> None:
        self.calls.append(("layer", t))

    def write_play_state(self, name, t, obj) -> None:
        self.calls.append((name, t))


def test_M6c_a_failed_save_still_writes_the_checkpoint_files_then_stops_the_play():
    rd = _Rec(fail=OSError("disk full"))
    layer = _Rec()
    fake = SimpleNamespace(
        rd=rd,
        _t=0,
        _saving=False,
        sched=SimpleNamespace(begin_save=lambda: None),
        layer=layer,
        obs=object(),
        chapters=SimpleNamespace(is_end=lambda t: False),
        _send_clock=lambda: None,
        on_chapter_end=None,
        _chapter_pending=False,
        chron=SimpleNamespace(threads=SimpleNamespace(state=lambda: {})),
    )
    with pytest.raises(OSError):  # the play stops through _guard with this error
        play.PlayPacer.before_checkpoint.__wrapped__(fake, None, 10)
    assert ("layer", 10) in layer.calls, "the storm layer's files of this checkpoint were not written"
    assert ("kinthreads", 10) in rd.calls
