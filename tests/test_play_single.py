"""The single save mode and the linked origin of a planet (DESIGN §34.11, row I64, stage B2).

``saves single`` keeps only the newest save: the origin and step 0 go too, at the next save, after the new
save is whole and ``LATEST`` names it. Every reader of an origin (a fork's founders, the houses, a missing
step) refuses with a message that says so, never a crash. A planet made from a save of its own world links
that save (``os.link``, else a copy) when the state it starts from is the source's byte for byte; a save never
writes into another save in place."""

from __future__ import annotations

import errno
import hashlib
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import pytest

from alife import cli, play, runio, runner
from alife.analysis import houses as house_rebuild
from alife.analysis import phylo, signals
from alife.config import Config, apply_overrides
from alife.runio import LATEST_FILE, CheckpointError, RunDir, load_checkpoint, state_file_is_complete
from alife.state import init_state, state_digest
from play_helpers import checkpoint_digests, rows, run_paced, run_plain, snapshots, world

END = 150


class Killed(BaseException):
    """Stands for a kill: not an Exception, so no ``except Exception`` of the code takes it."""


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def steps_of(rd: RunDir) -> list[int]:
    return [st for st, _ in rd.checkpoints()]


def save(rd: RunDir, cfg: Config, s: Any, step: int, *, obs: bool = False) -> None:
    rd.write_play_state("kinthreads", step, {"step": step})
    rd.save_checkpoint(
        s._replace(step=mx.array(step, dtype=mx.int32)),
        cfg,
        observers={"x": np.arange(3)} if obs else None,
    )


# --- the files ----------------------------------------------------------------------------------------


@pytest.mark.parametrize("compress", [False, True], ids=["plain", "compressed"])
def test_single_keeps_only_the_newest_save_and_its_sidecars(
    micro_cfg: Config, tmp_path: Path, compress: bool
) -> None:
    cfg = apply_overrides(micro_cfg, ["run.keep_checkpoints=3"])
    rd = RunDir.create(tmp_path / "fork", cfg, parent="src:20")
    rd.update_meta(start_step=20)
    rd.compress = compress
    rd.set_save_mode("single")
    s = init_state(cfg)
    for step in (20, 30, 40, 50):
        save(rd, cfg, s, step, obs=True)
        rd.wait_saves()
        assert steps_of(rd) == [step], "the origin and the older saves go at the next save"
    assert [st for st, _ in rd.observer_files()] == [50]
    assert [st for st, _ in rd.play_state_files("kinthreads")] == [50]
    suffix = ".zst" if compress else ""
    assert sorted(p.name for p in rd.checkpoint_dir.iterdir()) == sorted(
        [LATEST_FILE, f"step_0000000050.safetensors{suffix}"]
    ), "no older file and no debris is left"
    assert rd.latest_checkpoint() == rd.checkpoint_at(50)
    assert int(load_checkpoint(rd.checkpoint_at(50), cfg).step) == 50


def test_switching_to_single_deletes_nothing_and_the_next_save_applies_it(
    micro_cfg: Config, tmp_path: Path
) -> None:
    cfg = apply_overrides(micro_cfg, ["run.keep_checkpoints=3"])
    rd = RunDir.create(tmp_path / "w", cfg)
    rd.update_meta(start_step=0)
    s = init_state(cfg)
    for step in (0, 10, 20):
        save(rd, cfg, s, step)
    rd.set_save_mode("single")
    assert rd.save_mode == "single" and rd.keep_newest and rd.drop_origin
    assert steps_of(rd) == [0, 10, 20], "switching deletes nothing"
    save(rd, cfg, s, 30)
    assert steps_of(rd) == [30]
    rd.set_save_mode("all")
    assert rd.save_mode == "all" and not rd.keep_newest and not rd.drop_origin
    save(rd, cfg, s, 40)
    assert steps_of(rd) == [30, 40], "back to all: saves are kept from then on"
    with pytest.raises(ValueError, match="one of"):
        rd.set_save_mode("some")


@pytest.mark.parametrize("where", ["prune", "latest", "compress", "kill"])
@pytest.mark.parametrize("compress", [False, True], ids=["plain", "compressed"])
def test_a_save_that_fails_before_its_prune_leaves_a_loadable_world(
    micro_cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, where: str, compress: bool
) -> None:
    """Failure or kill between the write and the prune (or before ``LATEST`` moves): at every instant one
    complete save is loadable, and the world resumes from it."""
    if where == "compress" and not compress:
        pytest.skip("the compression is the compressed save's step")
    if where == "kill" and compress:
        pytest.skip("a kill is simulated in the synchronous save; a worker's failure is the next case")
    cfg = apply_overrides(micro_cfg, ["run.keep_checkpoints=3"])
    rd = RunDir.create(tmp_path / "w", cfg)
    rd.update_meta(start_step=20)
    rd.compress = compress
    s = init_state(cfg)
    save(rd, cfg, s, 20)
    save(rd, cfg, s, 30)  # all: the origin and the one before are kept
    rd.wait_saves()
    rd.set_save_mode("single")
    boom = RuntimeError("injected")
    if where == "prune":
        monkeypatch.setattr(rd, "_prune_saves", lambda *a, **k: (_ for _ in ()).throw(boom))
    elif where == "kill":
        monkeypatch.setattr(rd, "_prune_saves", lambda *a, **k: (_ for _ in ()).throw(Killed()))
    elif where == "latest":
        real = runio._atomic_write_text

        def latest_fails(path: Path, text: str) -> None:
            if path.name == LATEST_FILE:
                raise boom
            real(path, text)

        monkeypatch.setattr(runio, "_atomic_write_text", latest_fails)
    else:
        monkeypatch.setattr(
            runio, "compress_file", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full"))
        )
    with pytest.raises((RuntimeError, OSError, Killed)):
        save(rd, cfg, s, 40)
        rd.wait_saves()  # a compressed save's failure is raised here, in the play loop's next save or the end
    monkeypatch.undo()
    back = RunDir.open(rd.path)
    latest = back.latest_checkpoint()
    assert latest is not None and state_file_is_complete(latest, int(load_checkpoint(latest, cfg).step))
    assert int(load_checkpoint(latest, cfg).step) in (30, 40)
    if where in ("prune", "kill"):
        assert steps_of(back) == [20, 30, 40], "the failure came before any deletion"
    assert state_digest(load_checkpoint(latest, cfg)) == state_digest(
        load_checkpoint(back.checkpoint_at(int(load_checkpoint(latest, cfg).step)), cfg)
    )
    back.set_save_mode("single")  # a reopened world starts in the default mode (the play sets its own)
    save(back, cfg, s, 50)  # the world saves again and deletes what the single mode drops
    back.wait_saves()
    assert steps_of(back) == [50]


# --- the readers of an origin ------------------------------------------------------------------------


def test_the_readers_of_a_missing_origin_refuse_with_a_message(micro_cfg: Config, tmp_path: Path) -> None:
    cfg = micro_cfg
    rd = RunDir.create(tmp_path / "fork", cfg, parent="src:20")
    rd.update_meta(start_step=20)
    rd.set_save_mode("single")
    s = init_state(cfg)
    save(rd, cfg, s, 20)
    save(rd, cfg, s, 30)
    assert steps_of(rd) == [30]
    with pytest.raises(FileNotFoundError, match=r"no checkpoint at step 20 \(kept: \[30\]\)"):
        rd.checkpoint_at(20)
    with pytest.raises(CheckpointError, match="no checkpoint at step 20"):
        cli._select_checkpoint(rd, 20)
    assert cli._select_checkpoint(rd, None) == rd.checkpoint_at(30)
    with pytest.raises(ValueError, match="saves single"):
        phylo.load_tree(rd.path)  # a fork's founders are the agents alive in its origin save
    with pytest.raises(FileNotFoundError, match="kept"):
        signals.load_state(rd, cfg, None, 20)
    why = house_rebuild.source_problem(rd, 30)
    assert why is not None and "is itself a fork" in why, (
        "a fork's houses say so (before the step is looked at)"
    )


def test_the_houses_of_a_world_from_step_0_need_the_step_it_keeps(micro_cfg: Config, tmp_path: Path) -> None:
    cfg = micro_cfg
    rd = RunDir.create(tmp_path / "w", cfg)
    rd.update_meta(start_step=0)
    rd.set_save_mode("single")
    s = init_state(cfg)
    for step in (0, 10, 20):
        save(rd, cfg, s, step)
    why = house_rebuild.source_problem(rd, 10)
    assert why is not None and "has no checkpoint at step 10" in why and "[20]" in why
    with pytest.raises(house_rebuild.HouseError, match="has no checkpoint at step 10"):
        house_rebuild.rebuild(rd.path, 10, rate=0.5, seed=1)


def test_the_single_note_says_what_the_first_save_deletes(
    micro_cfg: Config, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rd = RunDir.create(tmp_path / "w", micro_cfg)
    rd.update_meta(start_step=0)
    s = init_state(micro_cfg)
    for step in (0, 10, 20):
        save(rd, micro_cfg, s, step)
    cli._newest_note(rd, None, "single", 20)
    err = capsys.readouterr().err
    assert "--saves single keeps only the newest save (the origin goes too)" in err
    assert "3 checkpoint(s)" in err and "steps 0 to 20" in err and "DELETED" in err
    cli._newest_note(rd, None, "newest", 20)
    err = capsys.readouterr().err
    assert "2 checkpoint(s)" in err and "steps 10 to 20" in err, "newest keeps step 0 and the origin"
    cli._newest_note(rd, {"saves": "single"}, "single", 20)
    assert capsys.readouterr().err == "", "a session already in the mode says nothing"


# --- the linked origin --------------------------------------------------------------------------------


@pytest.mark.parametrize("compress", [False, True], ids=["plain", "compressed"])
def test_a_linked_origin_is_the_source_save_and_outlives_its_source(
    micro_cfg: Config, tmp_path: Path, compress: bool
) -> None:
    s = init_state(micro_cfg)
    src = RunDir.create(tmp_path / "src", micro_cfg)
    src.compress = compress
    save(src, micro_cfg, s, 0)
    src.wait_saves()
    src_path = src.checkpoint_at(0)
    before = sha(src_path)
    dst = RunDir.create(tmp_path / "dst", micro_cfg, parent=f"{src.path}:0")
    dst.update_meta(start_step=0)
    dst.origin_link = (0, src_path)
    dst.save_checkpoint(s, micro_cfg)
    dst_path = dst.checkpoint_at(0)
    assert dst_path.name == src_path.name, "a compressed source stays compressed"
    assert dst_path.stat().st_ino == src_path.stat().st_ino and dst_path.stat().st_nlink == 2
    assert dst.origin_link is None, "the link is made once"
    assert dst.latest_checkpoint() == dst_path and sha(src_path) == before
    assert state_digest(load_checkpoint(dst_path, micro_cfg)) == state_digest(s)
    src.set_save_mode("single")  # the source deletes its origin at its next save
    save(src, micro_cfg, s, 10)
    src.wait_saves()
    assert not src_path.exists() and steps_of(src) == [10]
    assert dst_path.stat().st_nlink == 1 and sha(dst_path) == before, "the planet keeps the bytes"


def test_the_copy_fallback_writes_the_same_bytes(
    micro_cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    s = init_state(micro_cfg)
    src = RunDir.create(tmp_path / "src", micro_cfg)
    save(src, micro_cfg, s, 0)
    src_path = src.checkpoint_at(0)

    def cross_volume(a: Any, b: Any) -> None:
        raise OSError(errno.EXDEV, "cross-device link")

    monkeypatch.setattr(runio.os, "link", cross_volume)
    dst = RunDir.create(tmp_path / "dst", micro_cfg)
    dst.origin_link = (0, src_path)
    dst.save_checkpoint(s, micro_cfg)
    dst_path = dst.checkpoint_at(0)
    assert dst_path.stat().st_ino != src_path.stat().st_ino and dst_path.stat().st_nlink == 1
    assert sha(dst_path) == sha(src_path)
    assert not list(dst.checkpoint_dir.glob("*.tmp")), "no temporary file is left"


def test_a_planet_links_its_source_save_only_when_the_state_is_the_sources(
    micro_cfg: Config,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfdbinary: pytest.CaptureFixture[bytes],
) -> None:
    """``alife play --init-from RUN:STEP``: at step 0 the fork's state is the source's (its ledger and
    diagnostics are zero), so the origin is a link; at step 5 they are not, so the origin is written."""
    snapshots(monkeypatch, 10)
    src0 = RunDir.create(tmp_path / "src0", micro_cfg)
    runner_state = init_state(micro_cfg)
    save(src0, micro_cfg, runner_state, 0)
    src0_path = src0.checkpoint_at(0)
    code = cli.main(
        [
            "play",
            "--init-from",
            f"{src0.path}:0",
            "--out",
            str(tmp_path / "p0"),
            "--no-control",
            "--rate",
            "max",
            "--until",
            "4",
        ]
    )
    cap = capfdbinary.readouterr()
    assert code == 0, cap.err.decode(errors="replace")[-800:]
    planet0 = tmp_path / "p0" / "checkpoints" / src0_path.name
    assert planet0.stat().st_ino == src0_path.stat().st_ino, "the planet's origin is the source save"

    src5 = RunDir.create(tmp_path / "src5", micro_cfg)
    runner.run(
        src5, micro_cfg, init_state(micro_cfg), 5, quiet=True
    )  # a state that has run: ledger, diagnostics
    assert 5 in steps_of(src5)
    code = cli.main(
        [
            "play",
            "--init-from",
            f"{src5.path}:5",
            "--out",
            str(tmp_path / "p5"),
            "--no-control",
            "--rate",
            "max",
            "--until",
            "9",
        ]
    )
    cap = capfdbinary.readouterr()
    assert code == 0, cap.err.decode(errors="replace")[-800:]
    planet5 = (
        tmp_path / "p5" / "checkpoints" / "step_0000000005.safetensors.zst"
    )  # a play world saves compressed
    assert planet5.is_file() and planet5.stat().st_nlink == 1, "the origin is written, not linked"
    assert planet5.stat().st_ino != src5.checkpoint_at(5).stat().st_ino


# --- a single world through its chapters ---------------------------------------------------------------


def test_a_single_world_keeps_its_newest_save_and_equals_the_plain_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshots(monkeypatch, 10)
    cfg = world("2d")
    plain, res_plain = run_plain(cfg, tmp_path / "plain", END)
    opts = play.PlayOptions(rate=float("inf"), paused=False, fresh=True, unattended=True, saves="single")
    rd, res, pacer, _ = run_paced(cfg, tmp_path / "single", END, [[] for _ in range(END)], options=opts)
    assert pacer.failure is None and res.status == res_plain.status == "done"
    assert steps_of(rd) == [END], "five chapters, one save"
    assert checkpoint_digests(rd)[END] == checkpoint_digests(plain)[END]
    assert state_digest(res.state) == state_digest(res_plain.state) and rows(rd) == rows(plain)
