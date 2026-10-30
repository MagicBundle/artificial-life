"""The registry, the config plumbing and the small primitives of the extension mechanism (DESIGN §43)."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

import ext_stub as stub
from alife import config, ext, ops, rng
from alife.config import ConfigError, apply_overrides, config_hash, load_config
from alife.step import ActFlag


def test_with_every_feature_off_every_function_is_the_identity() -> None:
    """PR 0a shipped the machinery with an empty registry; A1 registers the arms row and S1 the structures
    row, B1 the bonds row, C1 the comm row and F1 the niche (farming) row. With every feature off (the
    default), nothing a feature-off run sees can change."""
    assert [s.name for s in ext.REGISTRY] == ["arms", "structures", "bonds", "comm", "niche"]
    cfg = load_config(preset="tiny")
    assert ext.active(cfg) == () and ext.layout_features(cfg) == "" and ext.mask(cfg) == 0
    assert (
        ext.layout_blocks(cfg, 2) == ()
        and ext.state_fields(cfg, 2) == ()
        and ext.step_inputs(None, cfg) == ()
    )
    assert ext.sum_due([None, None]) is None
    assert ext.registered_block_names() == frozenset(
        {
            "MATERIAL",
            "BUILD",
            "ROLE",
            "BOND_ROLE",
            "BOND_KIN",
            "BOND_FLOW",
            "ROLE_A",
            "ROLE_B",
            "VOICE",
            "VOICE_BEARING",
            "FARM",
            "FARM_PUT",
        }
    )
    n_feature_keys = sum(len(v) for v in config.FEATURE_KEYS.values())
    assert config.POST_V01_KEYS[-n_feature_keys - 1] == ("combat", "protect_floor")
    assert set(config.FEATURE_KEYS) == {"arms", "structures", "bonds", "comm", "niche"}
    assert config.POST_V01_KEYS[-1] == ("niche", "control")
    assert config.WORLD_OVERLAYS == (
        "storm",
        "storm160",
        "build",
        "bonds",
        "comm",
        "niche",
        "niche3d",
        "stormsph",
        "stormsph16",
        "stormsph22",
    )
    assert ext.check_registry() == []


def test_sum_due_adds_the_present_dues() -> None:
    a, b = mx.array([1, 2], dtype=mx.int32), mx.array([10, 20], dtype=mx.int32)
    assert ext.sum_due([a, None, b]).tolist() == [11, 22] and ext.sum_due([a]) is a


def test_act_flag_values_are_fixed_by_the_extension_index() -> None:
    expect = {"CONTESTED": 128, "BUILDING": 256, "ROLE_SWITCHED": 512, "SPOKE": 1024, "FARMING": 2048}
    for name, value in expect.items():
        assert ActFlag[name].value == value
    assert [ext.ACT_FLAG_BASE << i for i in range(5)] == [128, 256, 512, 1024, 2048]
    base = {f.value for f in ActFlag if f.name not in expect}
    assert base == {1, 2, 4, 8, 16, 32, 64} and all(v < 65536 for v in expect.values())
    assert [ext.EXTENSIONS.index(n) for n in ext.EXTENSIONS] == [0, 1, 2, 3, 4]


def test_the_stub_registry_is_sound_and_its_flags_match(monkeypatch: pytest.MonkeyPatch) -> None:
    reg = stub.install(monkeypatch)
    assert ext.check_registry() == []
    assert ext.flag_values() == {stub.FLAGS[s.name]: s.flag_value for s in reg}
    for name, flag in stub.FLAGS.items():
        assert ActFlag[flag].value == ext.ACT_FLAG_BASE << ext.EXTENSIONS.index(name)
    assert [s.index for s in reg] == [0, 1, 2, 3, 4] and [s.tail_rank for s in reg] == [4, 0, 1, 2, 3]


def test_collisions_are_reported() -> None:
    a = dataclasses.replace(stub.spec("arms"), dump_kinds=("x",), color_modes=("m",), stream_tags=("TAG1",))
    b = dataclasses.replace(stub.spec("comm"), dump_kinds=("x",), color_modes=("m",), stream_tags=("TAG1",))
    problems = ext.check_registry((a, b))
    for what in ("dump kind", "colour mode", "stream tag"):
        assert any(what in p for p in problems), what
    assert any("order" in p for p in ext.check_registry((b, a)))
    dup = dataclasses.replace(stub.spec("niche"), metric_prefix="arms_", act_flag="CONTESTED")
    assert any("metric prefix" in p for p in ext.check_registry((a, dup)))
    assert any("act flag" in p for p in ext.check_registry((a, dup)))


# --- hash32 ------------------------------------------------------------------------------------------------


def _hash32_ref(a, b, salt):
    m = 0xFFFFFFFF
    out = []
    for x, y in zip(a, b, strict=True):
        v = (x * 0x9E3779B1 + y * 0x85EBCA77 + ((salt * 0xC2B2AE3D + 0x27D4EB2F) & m)) & m
        v ^= v >> 15
        v = (v * 0x2C1B3C6D) & m
        v ^= v >> 12
        v = (v * 0x297A2D39) & m
        v ^= v >> 15
        out.append(v)
    return out


def test_hash32_pinned_vectors_and_reference() -> None:
    a = mx.arange(8, dtype=mx.int32)
    b = mx.array([0, 1, 2, 3, 100, 2**20, 5, 7], dtype=mx.int32)
    got = np.array(ops.hash32(a, b, 13))
    assert got.dtype == np.uint32
    assert got.tolist() == [
        3618484190,
        1293935347,
        2007747244,
        126437133,
        758401359,
        3746078765,
        3856809323,
        3827122364,
    ]
    assert got.tolist() == _hash32_ref(range(8), [0, 1, 2, 3, 100, 2**20, 5, 7], 13)
    # the salt separates the features; a scalar second argument works; negative ints wrap, never raise
    assert np.array(ops.hash32(a, 5, 14)).tolist() != np.array(ops.hash32(a, 5, 13)).tolist()
    assert np.array(ops.hash32(mx.array([-1], dtype=mx.int32), 0, 13)).dtype == np.uint32


def test_hash32_is_roughly_uniform() -> None:
    h = np.array(ops.hash32(mx.arange(1 << 16, dtype=mx.int32), 7, 14)) >> 28  # 16 buckets
    counts = np.bincount(h, minlength=16)
    assert counts.min() > 3500 and counts.max() < 4700


# --- rng ---------------------------------------------------------------------------------------------------


def test_rng_layout_version_is_unchanged_for_every_config() -> None:
    cfg = load_config(preset="tiny")
    assert (
        rng.layout_version(cfg) == rng.RNG_LAYOUT_VERSION == 3 and {1, 2, 3, 4} == rng.RNG_COMPATIBLE_VERSIONS
    )
    assert rng.layout_version(apply_overrides(cfg, ["arms.enabled=true"])) == 4  # only arms takes a stream
    assert rng.NUM_STREAMS == 16


def test_streams_zero_to_nine_keep_their_keys() -> None:
    """Frozen keys of the existing streams (no extension takes an index in PR 0a)."""
    ks = rng.step_keys(mx.random.key(0))[1]
    assert ks.shape == (16, 2)
    assert [s.name for s in rng.Stream] == [
        "NOISE", "BOND_PRIO", "BIRTH_PRIO", "PLACEMENT", "MUT_W1", "MUT_W2", "MUT_GENES", "IMMIGRANT",
        "CATASTROPHE", "WEATHER", "ARMS",
    ]  # fmt: skip
    assert max(s.value for s in rng.Stream) == 10 and max(s.value for s in rng.InitStream) == 8


# --- config ------------------------------------------------------------------------------------------------


def test_protect_floor_default_hash_and_serialisation_are_unchanged(tmp_path: Path) -> None:
    cfg = load_config(preset="tiny")
    on = apply_overrides(cfg, ["combat.protect_floor=0.1"])
    assert config_hash(on) != config_hash(cfg)
    # the key is hashed only off its default: setting the default back restores the hash
    assert config_hash(apply_overrides(on, ["combat.protect_floor=0.0"])) == config_hash(cfg)
    assert ("combat", "protect_floor") in config.POST_V01_KEYS
    # config.json and the checkpoint metadata: the key is left out at its default, written when set
    config.save_config(cfg, tmp_path / "off.json")
    config.save_config(on, tmp_path / "on.json")
    off, onj = (json.loads((tmp_path / n).read_text()) for n in ("off.json", "on.json"))
    assert "protect_floor" not in off["combat"] and onj["combat"]["protect_floor"] == 0.1
    assert config.config_from_dict(off) == cfg and config.config_from_dict(onj) == on
    assert "protect_floor" in config.config_to_dict(cfg)["combat"]  # the full dict (overrides, forks) has it


def test_protect_floor_validation() -> None:
    cfg = load_config(preset="tiny")
    for bad in ("-0.01", "0.51"):
        with pytest.raises(ConfigError, match="protect_floor"):
            apply_overrides(cfg, [f"combat.protect_floor={bad}"])
    for good in ("0.0", "0.1", "0.5"):
        apply_overrides(cfg, [f"combat.protect_floor={good}"])


def test_profit_warning_when_the_floor_makes_predation_unprofitable(caplog: pytest.LogCaptureFixture) -> None:
    cfg = load_config(preset="tiny")
    with caplog.at_level("WARNING", logger="alife.config"):
        apply_overrides(cfg, ["combat.protect_floor=0.10"])  # 0.7 * 4000 * 0.25 * 0.1 = 70 >= 50
        assert not caplog.records
        apply_overrides(cfg, ["combat.protect_floor=0.05"])  # 35 < 50
    assert any("unprofitable" in r.getMessage() for r in caplog.records)


def test_validate_all_runs_each_feature_rule(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(cfg: object) -> None:
        raise ConfigError("stub rule broken")

    monkeypatch.setattr(ext, "REGISTRY", (dataclasses.replace(stub.spec("comm"), validate=refuse),))
    with pytest.raises(ConfigError, match="stub rule broken"):
        apply_overrides(load_config(preset="tiny"), [])


def test_registered_modes_and_tags_are_in_streamrec(monkeypatch: pytest.MonkeyPatch) -> None:
    """streamrec is numpy-only (it may not import ``ext``), so its mode and tag lists stay literal: a feature
    that registers a colour mode or a stream tag must add it there too (DESIGN §43.5)."""
    from alife import streamrec

    assert all(m in streamrec.COLOR_MODES for s in ext.REGISTRY for m in s.color_modes)
    assert all(t in streamrec.EXT_TAGS for s in ext.REGISTRY for t in s.stream_tags)
    registered = tuple(t for s in ext.REGISTRY for t in s.stream_tags)
    assert tuple(t for t in streamrec.EXT_TAGS if t in registered) == registered  # the same order
    spec = dataclasses.replace(stub.spec("arms"), color_modes=("arms",), stream_tags=("FARS",))
    monkeypatch.setattr(ext, "REGISTRY", (spec,))
    assert not all(m in streamrec.COLOR_MODES for s in ext.REGISTRY for m in s.color_modes)  # the check bites
