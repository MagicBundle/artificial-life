"""Built structures: config, layout, state schema and the registry row (DESIGN §38.8, §38.13, §43)."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from alife import config, ext, fields, geom, layout, structures
from alife.config import (
    Config,
    ConfigError,
    StructuresConfig,
    apply_overrides,
    config_hash,
    config_to_dict,
    dims,
    load_config,
)
from alife.runio import load_checkpoint, write_state_file
from alife.state import (
    counter_blocks,
    counter_index,
    expected_layout,
    init_state,
    legacy_digest,
    n_counters,
    state_cls,
    state_from_arrays,
    state_to_arrays,
    to_numpy,
)

KEYS = [f.name for f in dataclasses.fields(StructuresConfig)]
OVR = ["brain.weight_dtype=float32"]


def _cfg(nd: int, *extra: str) -> Config:
    return load_config(
        preset="tiny" if nd == 2 else "tiny3d", overrides=[*OVR, "structures.enabled=true", *extra]
    )


def test_the_section_and_its_thirteen_keys_are_registered_and_default_to_off() -> None:
    assert KEYS[0] == "enabled" and len(KEYS) == 13
    s = Config().structures
    assert not s.enabled and not s.silent_build
    assert (s.cap, s.build_max, s.build_cost, s.decay_q12) == (4096, 16, 2, 16)
    assert (s.shelter_gain, s.barrier_gain, s.shield_gain) == (0.8, 3.0, 0.5)
    assert (s.reach, s.open_frac, s.aperture, s.exit_ease) == (0.5, 0.5, 0.25, 0.75)
    assert set(config.FEATURE_KEYS["structures"]) == {("structures", k) for k in KEYS}
    assert set(config.FEATURE_KEYS["structures"]) <= set(config.POST_V01_KEYS) & set(config.SERIAL_OMIT_KEYS)
    assert config.CLI_ONLY_FEATURE_KEYS["structures"] == (("structures", "silent_build"),)


def test_an_off_config_hashes_and_serialises_as_if_the_section_did_not_exist() -> None:
    for preset in config.PRESETS:
        cfg = load_config(preset=preset)
        assert "structures" not in config_to_dict(cfg, omit_defaults=True)
        assert config_hash(cfg) == config_hash(
            dataclasses.replace(cfg, structures=StructuresConfig())
        )  # a tautology guard: the default section is the absent one
    base = load_config(preset="tiny")
    on = apply_overrides(base, ["structures.enabled=true"])
    assert config_hash(on) != config_hash(base)
    # every key changes the hash once the feature is on, so none is silently inert
    for key, val in (
        ("cap", 2048),
        ("build_max", 8),
        ("build_cost", 1),
        ("decay_q12", 32),
        ("shelter_gain", 0.5),
        ("barrier_gain", 2.0),
        ("shield_gain", 0.25),
        ("reach", 0.25),
        ("open_frac", 0.6),
        ("aperture", 0.125),
        ("exit_ease", 0.5),
        ("silent_build", True),
    ):
        assert config_hash(apply_overrides(on, [f"structures.{key}={str(val).lower()}"])) != config_hash(
            on
        ), key


@pytest.mark.parametrize(
    "bad",
    [
        "structures.cap=100",
        "structures.cap=40000",
        "structures.build_max=0",
        "structures.build_max=2000",
        "structures.build_cost=-1",
        "structures.decay_q12=0",
        "structures.decay_q12=5000",
        "structures.shelter_gain=1.5",
        "structures.shield_gain=-0.1",
        "structures.reach=2",
        "structures.exit_ease=1.5",
        "structures.barrier_gain=3.1",
        "structures.barrier_gain=-1",
        "structures.open_frac=0",
        "structures.open_frac=1.2",
        "structures.aperture=1.5",
    ],
)
def test_validation_refuses_out_of_range_values(bad: str) -> None:
    with pytest.raises(ConfigError, match=bad.split("=")[0].split(".")[1]):
        _cfg(2, bad)


def test_keys_are_inert_while_the_feature_is_off() -> None:
    with pytest.raises(ConfigError, match=r"inert while structures\.enabled is false"):
        load_config(preset="tiny", overrides=["structures.cap=2048"])


def test_the_counter_cap_product_must_fit_int32() -> None:
    # 64k3d has 128 000 cells: cap 32767 would overflow the int32 MAT_TOTAL counter
    with pytest.raises(ConfigError, match="overflows"):
        load_config(preset="64k3d", overrides=["structures.enabled=true", "structures.cap=32767"])
    load_config(preset="64k3d", overrides=["structures.enabled=true", "structures.cap=16000"])


@pytest.mark.parametrize("nd", [2, 3])
def test_layout_totals_offsets_version_and_token(nd: int) -> None:
    cfg = _cfg(nd)
    lay = dims(cfg).lay
    d_in, d_out = (70, 36) if nd == 2 else (85, 40)
    assert (d_in, d_out) == (lay.D_IN, lay.D_OUT)
    assert (d_in - 1, d_out - 1) == (lay.IN_MATERIAL, lay.OUT_BUILD)
    assert ("material", d_in - 1, 1) in lay.INPUT_BLOCKS and ("build", d_out - 1, 1) in lay.OUTPUT_BLOCKS
    off = load_config(preset="tiny" if nd == 2 else "tiny3d")
    assert (dims(off).lay.IN_MATERIAL, dims(off).lay.OUT_BUILD) == (None, None)  # None, never -1
    assert ext.layout_features(cfg) == "structures@1" and ext.layout_features(off) == ""
    assert ext.mask(cfg) == 2
    assert layout.genome_layout_version(nd, ext.mask(cfg)) == (1 if nd == 2 else 2) + 8
    d = dims(cfg)
    assert (
        d.w1_rows == d_in + d.h
        and d.w2_rows == d.h + 1
        and d.params == (d_in + d.h) * d.h + (d.h + 1) * d_out
    )
    s = init_state(cfg)
    assert s.W1.shape[1:] == (d.w1_rows, d.h) and s.W2.shape[1:] == (d.w2_rows, d_out)


@pytest.mark.parametrize("nd", [2, 3])
def test_the_state_schema_with_the_field_on_and_off(nd: int) -> None:
    cfg = _cfg(nd)
    off = load_config(preset="tiny" if nd == 2 else "tiny3d", overrides=OVR)
    d = dims(cfg)
    want = expected_layout(cfg)["mat"]
    assert want == (d.field_shape, mx.int32) and "mat" not in expected_layout(off)
    s = init_state(cfg)
    assert type(s)._fields[-1] == "mat" and s.mat.shape == d.field_shape and s.mat.dtype == mx.int32
    assert not np.asarray(s.mat).any()
    assert state_cls(off).__name__ in ("State", "State3") and "mat" not in state_cls(off)._fields
    # the counter block sits after the base counters and lists the eleven names in order
    ((name, names),) = counter_blocks(cfg)
    assert name == "structures" and names == structures.COUNTERS
    assert counter_index(cfg, "BUILD_UNITS") == n_counters(nd, False) and n_counters(
        nd, False, counter_blocks(cfg)
    ) == (n_counters(nd) + 11)


@pytest.mark.parametrize("nd", [2, 3])
def test_a_file_without_mat_is_filled_with_zeros_and_the_reverse_is_refused(nd: int, tmp_path: Path) -> None:
    cfg = _cfg(nd)
    arrays = state_to_arrays(init_state(cfg))
    stripped = {k: v for k, v in arrays.items() if k != "mat"}
    filled = state_from_arrays(stripped, cfg)
    assert not np.asarray(filled.mat).any() and filled.mat.shape == dims(cfg).field_shape
    off = load_config(preset="tiny" if nd == 2 else "tiny3d", overrides=OVR)
    with pytest.raises(ValueError, match=r"extra=\['mat'\]"):
        state_from_arrays(arrays, off)
    # a checkpoint round trip keeps the material
    s = init_state(cfg)._replace(mat=mx.full(dims(cfg).field_shape, 77, mx.int32))
    path = tmp_path / "s.safetensors"
    write_state_file(path, s, cfg)
    back = load_checkpoint(path, cfg)
    assert np.array_equal(np.asarray(back.mat), np.asarray(s.mat))
    # the legacy digest does not see the new field
    assert legacy_digest(init_state(cfg)) == legacy_digest(init_state(cfg)._replace(mat=s.mat))


def test_silent_build_zeroes_the_build_column_of_founders_only() -> None:
    d_cfg = _cfg(2)
    silent = _cfg(2, "structures.silent_build=true")
    for cfg, is_zero in ((d_cfg, False), (silent, True)):
        d = dims(cfg)
        w2 = to_numpy(init_state(cfg).W2)[to_numpy(init_state(cfg).alive)]
        assert (not w2[:, :, d.lay.OUT_BUILD].any()) == is_zero
        assert w2[:, :, : d.lay.OUT_BUILD].any()  # the other columns are random
    # the material row of W1 is random either way (O6: only the build column is the control)
    d = dims(silent)
    assert to_numpy(init_state(silent).W1)[:, d.lay.IN_MATERIAL].any()


def test_the_build_overlay_validates_against_the_presets_that_can_be_coloured() -> None:
    assert "build" in config.WORLD_OVERLAYS and "build" not in config.OVERLAYS
    for preset in config.PRESETS:
        cfg = load_config(preset=preset, overlays=["build"])
        if preset == "tiny3s":  # a shell has no lattice side: its closing cycle is 3·n_colours phases
            n_phase = 3 * geom.min_colouring(cfg.world.nside)[1]
            assert cfg.structures.enabled and int(fields.mat_colour(cfg).max().item()) + 1 == n_phase
            continue
        assert cfg.structures.enabled and fields.mat_phases(dims(cfg).fg, dims(cfg).nd)[0] >= 3
    # an overlay on a world whose lattice is prime is refused with the reason
    with pytest.raises(ConfigError, match="colouring"):
        load_config(preset="tiny", overlays=["build"], overrides=["world.size=44"])


def test_validate_message_names_the_open_neighbour_limit_in_3d() -> None:
    with pytest.raises(ConfigError, match="17"):
        _cfg(3, "structures.aperture=0.8")


def test_shield_gain_one_needs_a_protection_floor() -> None:
    """cf289b4's policy: an immunity (demand 0 in a full cell) is refused unless ``combat.protect_floor``
    bounds it, as ``arms.key_floor = 0`` is."""
    with pytest.raises(ConfigError, match="protect_floor"):
        _cfg(2, "structures.shield_gain=1")
    cfg = _cfg(2, "structures.shield_gain=1", "combat.protect_floor=0.1")
    assert cfg.structures.shield_gain == 1.0
    _cfg(2, "structures.shield_gain=0.9")


def test_aperture_zero_with_a_barrier_is_allowed_as_a_control_and_says_so(caplog) -> None:
    import logging

    with caplog.at_level(logging.WARNING):
        cfg = _cfg(2, "structures.aperture=0")
    assert cfg.structures.aperture == 0.0
    assert any("control only" in r.message or "a control only" in r.message for r in caplog.records)
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        _cfg(2)
        _cfg(2, "structures.aperture=0", "structures.barrier_gain=0")  # no barrier: nothing to enclose with
    assert not [r for r in caplog.records if "aperture" in r.message]
