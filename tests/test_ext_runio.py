"""The ``layout_features`` meta key and the extension layout guard of checkpoints (DESIGN §43.2), the silent
initialisation of tail blocks, and the config serialisation of the feature series."""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

import ext_stub as stub
from alife import brain, ext, rng
from alife.config import dims, load_config
from alife.runio import CheckpointError, RunDir, checkpoint_metadata, load_checkpoint, write_state_file
from alife.state import init_state, state_digest


@pytest.fixture(autouse=True)
def _registry(monkeypatch: pytest.MonkeyPatch) -> None:
    stub.install(monkeypatch)


def _cfg(nd: int, names=(), *over: str):
    cfg = load_config(preset="tiny" if nd == 2 else "tiny3d", overrides=["brain.weight_dtype=float32", *over])
    return stub.stub_cfg(cfg, *names) if names else cfg


@pytest.mark.parametrize("nd", [2, 3])
def test_feature_off_meta_is_todays(nd: int, tmp_path: Path) -> None:
    cfg = _cfg(nd)
    path = tmp_path / "s.safetensors"
    write_state_file(path, init_state(cfg), cfg)
    ck = checkpoint_metadata(path)
    assert "layout_features" not in ck and ck["genome_layout_version"] == str(nd - 1 if nd == 2 else 2)
    assert ck["rng_layout_version"] == "3" == str(rng.layout_version(cfg))
    meta = RunDir.create(tmp_path / "run", cfg).meta()
    assert "layout_features" not in meta and meta["rng_layout_version"] == 3
    assert "protect_floor" not in json.loads(ck["config"])["combat"]
    assert load_checkpoint(path, cfg).step.item() == 0  # and it loads


def test_feature_on_meta_and_the_round_trip(tmp_path: Path) -> None:
    cfg = _cfg(2, ("structures", "comm"))
    s = init_state(cfg)
    path = tmp_path / "s.safetensors"
    write_state_file(path, s, cfg)
    ck = checkpoint_metadata(path)
    assert ck["layout_features"] == "structures@1,comm4@1" and ck["genome_layout_version"] == str(
        1 + 4 * (2 + 8)
    )
    assert RunDir.create(tmp_path / "run", cfg).meta()["layout_features"] == "structures@1,comm4@1"
    back = load_checkpoint(path, cfg)
    assert type(back) is type(s) and state_digest(back) == state_digest(s)


def test_a_checkpoint_never_crosses_layout_features(tmp_path: Path) -> None:
    on, off, other = _cfg(2, ("structures",)), _cfg(2), _cfg(2, ("niche",))
    p_on, p_off = tmp_path / "on.safetensors", tmp_path / "off.safetensors"
    write_state_file(p_on, init_state(on), on)
    write_state_file(p_off, init_state(off), off)
    for path, cfg, both in (
        (p_on, off, ("structures@1", "''")),
        (p_off, on, ("''", "structures@1")),
        (p_on, other, ("structures@1", "niche.farm@1")),
    ):
        with pytest.raises(CheckpointError, match="layout_features") as e:
            load_checkpoint(path, cfg, check_hash=False)
        assert both[0] in str(e.value) and both[1] in str(e.value)  # names both strings
    # the same files load into their own config
    load_checkpoint(p_on, on, check_hash=False)
    load_checkpoint(p_off, off, check_hash=False)


def test_an_old_file_without_the_key_is_the_empty_string(tmp_path: Path) -> None:
    """A file written before the key existed (absent) has no features: it loads into a feature-off config."""
    cfg = _cfg(2)
    path = tmp_path / "s.safetensors"
    write_state_file(path, init_state(cfg), cfg)
    assert "layout_features" not in checkpoint_metadata(path)
    assert load_checkpoint(path, cfg, check_hash=False) is not None


def test_silent_blocks_start_at_zero_and_the_others_do_not() -> None:
    cfg = _cfg(2, ext.EXTENSIONS)
    d = dims(cfg)
    ik = rng.init_keys(0)
    w1, w2 = brain.init_weights(ik[rng.InitStream.W1], ik[rng.InitStream.W2], 16, cfg)
    lay = d.lay
    assert w1.shape == (16, d.w1_rows, d.h) and w2.shape == (16, d.w2_rows, lay.D_OUT)
    for _, name, start, width, silent in lay.EXT_IN:
        rows = np.array(w1[:, start : start + width])
        assert bool((rows == 0).all()) == silent, name
    for _, name, start, width, silent in lay.EXT_OUT:
        cols = np.array(w2[:, :, start : start + width])
        assert bool((cols == 0).all()) == silent, name
    # the base block is the same draw as without the features on the rows below the tail
    off = _cfg(2)
    o1, _ = brain.init_weights(ik[rng.InitStream.W1], ik[rng.InitStream.W2], 16, off)
    assert o1.shape[1] == dims(off).w1_rows == d.w1_rows - 28
    # a feature-off config draws exactly the weights it always did
    assert mx.array_equal(
        brain.init_weights(ik[rng.InitStream.W1], ik[rng.InitStream.W2], 16, off)[1], _legacy_w2(off, ik)
    )


def _legacy_w2(cfg, ik):
    d, bc = dims(cfg), cfg.brain
    import math

    w2 = mx.random.normal((16, d.w2_rows, d.lay.D_OUT), key=ik[rng.InitStream.W2]) * (
        bc.init_scale / math.sqrt(d.w2_rows)
    )
    return mx.clip(w2, -bc.weight_clip, bc.weight_clip).astype(d.wdtype)
