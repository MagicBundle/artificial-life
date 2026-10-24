"""Light saves (DESIGN §34.11): the compressed save format, its one loader, and the save path's ordering."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import pytest

from alife import runio, runner
from alife.config import Config, apply_overrides
from alife.runio import (
    CheckpointError,
    RunDir,
    checkpoint_metadata,
    compress_file,
    load_checkpoint,
    read_state_arrays,
    state_file,
    state_file_is_complete,
    write_state_file,
)
from alife.state import init_state, state_digest, to_numpy


@pytest.fixture
def cfg(micro_cfg: Config) -> Config:
    return apply_overrides(
        micro_cfg, ["run.metrics_every=5", "run.checkpoint_every=10", "run.heavy_metrics_every=5"]
    )


def _plain_and_zst(tmp: Path, cfg: Config, steps: int = 0) -> tuple[Path, Path, object]:
    s = init_state(cfg)
    if steps:
        s = runner.run(RunDir.create(tmp / "warm", cfg), cfg, s, steps, quiet=True).state
    plain = tmp / f"step_{int(to_numpy(s.step)):010d}.safetensors"
    write_state_file(plain, s, cfg)
    zst = plain.with_name(plain.name + ".zst")
    compress_file(plain, zst)
    return plain, zst, s


def test_round_trip_is_bit_identical(cfg: Config, tmp_path: Path) -> None:
    plain, zst, _ = _plain_and_zst(tmp_path, cfg, steps=3)
    assert zst.stat().st_size < plain.stat().st_size
    with state_file(zst) as decoded:
        assert Path(decoded).read_bytes() == plain.read_bytes()  # every tensor and the metadata
    assert checkpoint_metadata(zst) == checkpoint_metadata(plain)
    a = read_state_arrays(zst)
    b = read_state_arrays(plain)
    assert sorted(a) == sorted(b)
    for k in a:
        assert a[k].dtype == b[k].dtype and a[k].shape == b[k].shape
        assert bool(mx.array_equal(a[k], b[k]).item()), k
    assert not list(tmp_path.glob("alife-state-*"))  # the decoded copy is removed


def test_header_only_metadata_of_a_zst(tmp_path: Path) -> None:
    plain = tmp_path / "big.safetensors"
    mx.save_safetensors(str(plain), {"x": mx.random.normal((2_000_000,))}, {"step": "7", "config": "{}"})
    zst = tmp_path / "big.safetensors.zst"
    compress_file(plain, zst)
    meta = checkpoint_metadata(zst)
    assert meta == {"step": "7", "config": "{}"}
    cut = tmp_path / "cut.safetensors.zst"
    data = zst.read_bytes()
    cut.write_bytes(data[: len(data) // 2])  # the body is torn; the header is whole and is all that is read
    assert checkpoint_metadata(cut) == meta
    with pytest.raises(CheckpointError):
        read_state_arrays(cut)


def test_loaded_state_equals_the_plain_one(cfg: Config, tmp_path: Path) -> None:
    plain, zst, s = _plain_and_zst(tmp_path, cfg, steps=2)
    assert (
        state_digest(load_checkpoint(zst, cfg))
        == state_digest(load_checkpoint(plain, cfg))
        == state_digest(s)
    )


@pytest.mark.parametrize("damage", ["flip", "truncate", "trailing", "garbage"])
def test_corrupt_or_truncated_zst_raises(cfg: Config, tmp_path: Path, damage: str) -> None:
    _, zst, _ = _plain_and_zst(tmp_path, cfg)
    bad = tmp_path / "bad.safetensors.zst"
    data = bytearray(zst.read_bytes())
    if damage == "flip":
        data[len(data) // 2] ^= 0x10
    elif damage == "truncate":
        data = data[:-7]
    elif damage == "trailing":
        data += b"\x00\x01\x02"
    else:
        data = bytearray(b"not a zstd frame at all, just some bytes")
    bad.write_bytes(bytes(data))
    with pytest.raises(CheckpointError):
        load_checkpoint(bad, cfg)
    assert not state_file_is_complete(bad, int(zst.name[5:15]))
    with pytest.raises(CheckpointError):
        read_state_arrays(bad)


def test_old_and_new_names_are_both_checkpoints(cfg: Config, tmp_path: Path) -> None:
    _, zst, s = _plain_and_zst(tmp_path, cfg)
    rd = RunDir.create(tmp_path / "run", cfg)
    step = int(to_numpy(s.step))
    write_state_file(rd.checkpoint_dir / f"step_{step:010d}.safetensors", s, cfg)
    (rd.checkpoint_dir / f"step_{step + 6:010d}.safetensors.zst").write_bytes(zst.read_bytes())
    names = [p.name for _, p in rd.checkpoints()]
    assert names == [f"step_{step:010d}.safetensors", f"step_{step + 6:010d}.safetensors.zst"]
    assert rd.checkpoint_at(step + 6).name.endswith(".zst")
    assert rd.latest_checkpoint() is not None


def _deterministic(rows: list[dict]) -> list[dict]:
    return [{k: v for k, v in r.items() if not any(w in k for w in ("wall", "per_s", "mem"))} for r in rows]


def test_compressed_run_resumes_like_the_plain_one(cfg: Config, tmp_path: Path) -> None:
    def run_and_resume(name: str, compress: bool):
        rd = RunDir.create(tmp_path / name, cfg)
        rd.compress = compress
        runner.run(rd, cfg, init_state(cfg), 12, quiet=True)
        rd.wait_saves()
        s = load_checkpoint(rd.checkpoint_at(10), cfg)
        res = runner.run(rd, cfg, s, 14, quiet=True)
        rd.wait_saves()
        return rd, state_digest(res.state)

    plain, digest_plain = run_and_resume("plain", False)
    light, digest_light = run_and_resume("light", True)
    assert digest_light == digest_plain
    assert [st for st, _ in light.checkpoints()] == [0, 12, 20, 24] == [st for st, _ in plain.checkpoints()]
    for (st, p), (_, q) in zip(light.checkpoints(), plain.checkpoints(), strict=True):
        assert state_digest(load_checkpoint(p, cfg)) == state_digest(load_checkpoint(q, cfg)), st
    # the wall-clock columns differ between the two runs; every other column is the same
    assert _deterministic(light.metrics_rows()) == _deterministic(plain.metrics_rows())


def test_newest_only_compressed_saves_keep_the_newest_and_the_origin(cfg: Config, tmp_path: Path) -> None:
    rd = RunDir.create(tmp_path / "run", cfg)
    rd.compress = True
    rd.keep_newest = True
    runner.run(rd, cfg, init_state(cfg), 18, quiet=True)
    rd.wait_saves()
    steps = [st for st, _ in rd.checkpoints()]
    assert steps == [0, 18]
    assert all(p.name.endswith(".zst") for _, p in rd.checkpoints())
    assert not list(rd.checkpoint_dir.glob("*.tmp"))


def test_a_killed_save_leaves_the_previous_one_resumable(cfg: Config, tmp_path: Path, monkeypatch) -> None:
    rd = RunDir.create(tmp_path / "run", cfg)
    rd.compress = True
    runner.run(rd, cfg, init_state(cfg), 6, quiet=True)
    rd.wait_saves()
    before = rd.latest_checkpoint()

    def crash(src, dst, **kw):
        dst.with_name(dst.name + ".tmp").write_bytes(b"partial")  # what a kill leaves behind
        raise OSError("killed")

    monkeypatch.setattr(runio, "compress_file", crash)
    s = load_checkpoint(before, cfg)
    with pytest.raises(
        OSError
    ):  # the save of step 12 is killed in its thread; the next save raises its error
        runner.run(rd, cfg, s, 6, quiet=True)
    monkeypatch.undo()
    assert rd.latest_checkpoint() == before  # LATEST never moved onto a save that was not whole
    assert state_digest(load_checkpoint(rd.latest_checkpoint(), cfg)) == state_digest(s)
    assert [st for st, _ in rd.checkpoints()] == [0, 6]  # the older saves stay in place
    assert RunDir.open(tmp_path / "run").latest_checkpoint() == before
