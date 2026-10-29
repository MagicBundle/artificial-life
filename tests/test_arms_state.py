"""Arms races, stages 1 and 2: config, state, checkpoints, RNG version, registry row (DESIGN §42, §43)."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from alife import arms, config, ext, rng
from alife.config import Config, ConfigError, apply_overrides, config_hash, dims, load_config
from alife.runio import CheckpointError, RunDir, load_checkpoint, write_state_file
from alife.state import (
    AGENT_FIELDS,
    AGENT_FIELDS_3D,
    State,
    State3,
    agent_fields,
    expected_layout,
    init_state,
    legacy_digest,
    state_cls,
    state_digest,
    state_from_arrays,
    state_to_arrays,
)

# config_hash of the engine before the arms section existed (main e64be03): feature-off runs keep their hash.
PRE_ARMS_HASH = {
    "default": "69b4e21ad6b6016b17c11d6a1d9b3371c03b4b9f590fdb4a04752515c13a01d8",
    "tiny": "d25e0eaee7349ae366e3e4e60ebab107cf7a89fcf49be88a12d81099ff2561d5",
    "tiny3d": "90e799beeba91e8cb0e2e021ad9595340efa9975eb53f963047c63646e61797a",
}
OVR = [
    "arms.enabled=true",
    "arms.key_floor=0.2",
    "arms.sigma=0.03",
    "arms.sigma_phase=0.05",
    "brain.weight_dtype=float32",
]


def _cfg(nd: int, *extra: str) -> Config:
    return load_config(preset="tiny" if nd == 2 else "tiny3d", overrides=[*OVR, *extra])


def test_the_registry_row_and_the_hash_of_every_off_config() -> None:
    assert [s.name for s in ext.REGISTRY] == ["arms", "structures"] and ext.check_registry() == []
    assert config_hash(Config()) == PRE_ARMS_HASH["default"]
    assert config_hash(load_config(preset="tiny")) == PRE_ARMS_HASH["tiny"]
    assert config_hash(load_config(preset="tiny3d")) == PRE_ARMS_HASH["tiny3d"]
    assert ("arms", "enabled") in config.POST_V01_KEYS and ("arms", "sigma_phase") in config.SERIAL_OMIT_KEYS
    assert all(sec == "arms" for sec, _ in config.FEATURE_KEYS["arms"])
    on = _cfg(2)
    assert config_hash(on) != PRE_ARMS_HASH["tiny"]
    assert config_hash(apply_overrides(on, ["arms.sigma=0.04"])) != config_hash(on)


def test_the_defaults_are_off() -> None:
    a = Config().arms
    assert not a.enabled and a.key_floor == 1.0 and a.speed_slow == 0.0
    assert (a.upkeep_atk, a.upkeep_arm, a.init_atk, a.init_arm, a.sigma, a.sigma_phase) == (0.0,) * 6


@pytest.mark.parametrize(
    "bad",
    [
        ["arms.kappa=-1"],
        ["arms.a0=0"],
        ["arms.kappa=20"],  # kappa/(kappa+a0) > 0.9
        ["arms.key_floor=1.5"],
        ["arms.speed_slow=0.95"],
        ["arms.upkeep_atk=40000"],
        ["arms.init_arm=1.2"],
        ["arms.sigma=0.6"],
    ],
)
def test_validate_refuses_out_of_range_keys(bad: list[str]) -> None:
    with pytest.raises(ConfigError):
        apply_overrides(load_config(preset="tiny"), ["arms.enabled=true", *bad])


def test_key_floor_zero_needs_a_protection_floor() -> None:
    base = load_config(preset="tiny")
    with pytest.raises(ConfigError, match=r"key_floor = 0 needs combat\.protect_floor"):
        apply_overrides(base, ["arms.enabled=true", "arms.key_floor=0"])
    apply_overrides(base, ["arms.enabled=true", "arms.key_floor=0", "combat.protect_floor=0.1"])
    apply_overrides(base, ["arms.enabled=true", "arms.key_floor=0.01"])


def test_a_key_is_inert_while_arms_is_off() -> None:
    with pytest.raises(ConfigError, match=r"inert while arms\.enabled is false"):
        apply_overrides(load_config(preset="tiny"), ["arms.sigma=0.03"])


@pytest.mark.parametrize("nd", [2, 3])
def test_state_type_fields_and_init(nd: int) -> None:
    off, on = (
        _cfg(nd, "arms.enabled=false", "arms.key_floor=1.0", "arms.sigma=0", "arms.sigma_phase=0"),
        _cfg(nd),
    )
    assert state_cls(off) is (State3 if nd == 3 else State)
    base = AGENT_FIELDS_3D if nd == 3 else AGENT_FIELDS
    assert agent_fields(on) == (*base, "arms") and agent_fields(off) == base
    s = init_state(_cfg(nd, "arms.init_atk=0.2", "arms.init_arm=0.4"))
    n = dims(on).n
    assert s.arms.shape == (n, 6) and s.arms.dtype == mx.float32
    x = np.array(s.arms)
    assert (x[:, arms.ATK] == np.float32(0.2)).all() and (x[:, arms.ARM] == np.float32(0.4)).all()
    assert not x[:, arms.TOX].any() and not x[:, arms.SIG].any()
    assert ((x[:, arms.KEY :] >= 0) & (x[:, arms.KEY :] < 1)).all()
    assert len(np.unique(x[:, arms.KEY])) > n // 2 and not np.array_equal(x[:, arms.KEY], x[:, arms.LOCK])
    assert (tuple(s.arms.shape), s.arms.dtype) == expected_layout(on)["arms"]
    # deterministic in the seed
    assert np.array_equal(np.array(init_state(_cfg(nd, "arms.init_atk=0.2", "arms.init_arm=0.4")).arms), x)
    assert not np.array_equal(np.array(init_state(_cfg(nd, "run.seed=5")).arms)[:, arms.KEY], x[:, arms.KEY])
    # the legacy digest (frozen field list) is unchanged by the new column
    assert legacy_digest(init_state(_cfg(nd))) == legacy_digest(init_state(_cfg(nd, "arms.sigma=0.01")))


@pytest.mark.parametrize("nd", [2, 3])
def test_a_pre_arms_checkpoint_loads_with_arms_on_and_the_reverse_is_refused(nd: int, tmp_path: Path) -> None:
    on = _cfg(nd)
    off = load_config(preset="tiny" if nd == 2 else "tiny3d", overrides=["brain.weight_dtype=float32"])
    s_off = init_state(off)
    path = tmp_path / "off.safetensors"
    write_state_file(path, s_off, off)
    filled = load_checkpoint(path, on, check_hash=False)  # a fork onto arms from an old file
    assert filled.arms.shape == (dims(on).n, 6)
    assert np.array_equal(np.array(filled.arms), np.array(arms.rest_array(on)))
    assert legacy_digest(filled) == legacy_digest(s_off)
    # an arms file into an arms-off config: the "extra" refusal of state_from_arrays
    path2 = tmp_path / "on.safetensors"
    write_state_file(path2, init_state(on), on)
    with pytest.raises((CheckpointError, ValueError)):
        load_checkpoint(path2, off, check_hash=False)
    arrays = state_to_arrays(init_state(on))
    with pytest.raises(ValueError, match=r"extra=\['arms'\]"):
        state_from_arrays(arrays, off)
    assert state_digest(state_from_arrays(arrays, on)) == state_digest(init_state(on))


def test_the_rng_version_and_the_streams(tmp_path: Path) -> None:
    on, off = _cfg(2), load_config(preset="tiny")
    assert rng.layout_version(off) == 3 and rng.layout_version(on) == 4 == rng.ARMS_LAYOUT_VERSION
    assert {1, 2, 3, 4} == rng.RNG_COMPATIBLE_VERSIONS and rng.RNG_LAYOUT_VERSION == 3
    assert rng.Stream.ARMS == 10 and rng.InitStream.ARMS == 8
    # taking a reserved stream moves no other key
    k = mx.random.key(3)
    _, ks = rng.step_keys(k)
    assert ks.shape == (rng.NUM_STREAMS, 2)
    assert RunDir.create(tmp_path / "a", on).meta()["rng_layout_version"] in (4, "4")
    assert RunDir.create(tmp_path / "b", off).meta()["rng_layout_version"] in (3, "3")
