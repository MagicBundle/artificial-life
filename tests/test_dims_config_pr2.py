"""§27 the 3D keys of docs/3D.md PR 2 (gravity, the floor, buoyancy, the buoyancy-cap trait, the vertical
food profile): off by default and hashed only off default, rejected in 2D; their 3D rules; the z bounds."""

from __future__ import annotations

import numpy as np
import pytest

from alife import config
from alife.config import (
    POST_V01_KEYS,
    THREE_D_KEYS,
    UNHASHED_KEYS,
    Config,
    ConfigError,
    config_from_dict,
    config_hash,
    config_to_dict,
    dims,
    load_config,
    validate,
)
from helpers import as_3d

# Pinned in test_features.py and test_director_config.py; re-asserted: the new keys must not move them.
TINY_HASH_V01 = "d25e0eaee7349ae366e3e4e60ebab107cf7a89fcf49be88a12d81099ff2561d5"
K64_HASH = "69b4e21ad6b6016b17c11d6a1d9b3371c03b4b9f590fdb4a04752515c13a01d8"

PR2_KEYS = (
    ("physics", "gravity"),
    ("physics", "floor_eps"),
    ("physics", "buoyancy_max"),
    ("mutation", "buoy_init"),
    ("mutation", "buoy_sigma"),
    ("fields", "benthic_lambda"),
    ("fields", "photic_amp"),
    ("fields", "photic_lambda"),
)

# A valid 3D tiny once 3D is enabled (test_dims_config.TINY3D).
TINY3D = ["world.dims=3", "world.height=24", "fields.chem_diffusion=0.16,0.1,0.16", "fields.benthic_lambda=8"]


@pytest.fixture
def enable_3d(monkeypatch: pytest.MonkeyPatch) -> None:
    """``validate`` accepts dims = 3 (as it will from docs/3D.md PR 3)."""
    monkeypatch.setattr(config, "DIMS_SUPPORTED", (2, 3))


def _problems(overrides: list[str]) -> str:
    with pytest.raises(ConfigError) as ei:
        load_config(preset="tiny", overrides=overrides)
    return str(ei.value)


# --- keys and hashes --------------------------------------------------------------------------------------


def test_pr2_keys_default_off_and_listed() -> None:
    assert THREE_D_KEYS[2:] == PR2_KEYS
    defaults = config_to_dict(Config())
    for sec, key in PR2_KEYS:
        assert defaults[sec][key] == 0.0 and isinstance(defaults[sec][key], float), (sec, key)
        assert (sec, key) in POST_V01_KEYS and (sec, key) not in UNHASHED_KEYS, (sec, key)


def test_pinned_hashes_unchanged() -> None:
    assert config_hash(load_config(preset="tiny")) == TINY_HASH_V01
    assert config_hash(load_config(preset="64k")) == K64_HASH


@pytest.mark.parametrize(("sec", "key"), PR2_KEYS)
def test_hash_moves_only_off_default(sec: str, key: str) -> None:
    base = config_to_dict(load_config(preset="tiny"))
    changed = config_from_dict({**base, sec: {**base[sec], key: 0.5}})
    assert config_hash(changed) != TINY_HASH_V01
    explicit = config_from_dict({**base, sec: {**base[sec], key: 0.0}})
    assert config_hash(explicit) == TINY_HASH_V01


def test_old_dict_loads_equal() -> None:
    cfg = load_config(preset="tiny")
    old = config_to_dict(cfg)
    for sec, key in PR2_KEYS:  # a config.json written before PR 2
        del old[sec][key]
    back = config_from_dict(old)
    validate(back)
    assert back == cfg and config_hash(back) == TINY_HASH_V01


@pytest.mark.parametrize(("sec", "key"), PR2_KEYS)
def test_pr2_keys_rejected_in_2d(sec: str, key: str) -> None:
    msg = _problems([f"{sec}.{key}=0.5"])
    assert msg.startswith("1 config problem(s)") and f"{sec}.{key} is a 3D key" in msg, msg


# --- the float32 z bounds (Dims) ----------------------------------------------------------------------------


def test_z_bounds_2d_are_zero() -> None:
    d = dims(load_config(preset="tiny"))
    assert (d.z_lo, d.z_hi, d.z_ground) == (0.0, 0.0, 0.0)


def test_z_bounds_tiny3d() -> None:
    d = dims(as_3d(load_config(preset="tiny"), physics={"floor_eps": 0.25}))
    assert (d.z_lo, d.z_hi, d.z_ground) == (1.0, 23.0, 1.25)


@pytest.mark.parametrize("r", [0.7, 0.3])  # float32 rounds 0.7 and 23.3 down, 0.3 and 23.7 up: every branch
def test_z_bounds_are_float32_exact(r: float) -> None:
    cfg = as_3d(load_config(preset="tiny"), physics={"radius": r, "floor_eps": 0.1})
    d, height = dims(cfg), cfg.world.height
    f32 = np.float32
    for v in (d.z_lo, d.z_hi, d.z_ground):
        assert float(f32(v)) == v  # representable: MLX's cast of the Python bound is exact
    assert d.z_lo >= r and float(np.nextafter(f32(d.z_lo), f32(-np.inf))) < r
    assert d.z_hi <= height - r and float(np.nextafter(f32(d.z_hi), f32(np.inf))) > height - r
    assert d.z_ground <= r + 0.1 and float(np.nextafter(f32(d.z_ground), f32(np.inf))) > r + 0.1  # round down


@pytest.mark.parametrize("r", [0.7, 0.9, 0.3, 1.0])  # float32 rounds 0.7 and 0.9 down, 0.3 up; 1.0 is exact
def test_z_ground_is_never_below_the_floor_rest(r: float) -> None:
    """With floor_eps = 0 the threshold is z_lo itself, so an agent resting on the floor is grounded even when
    the nearest float32 to r lies below z_lo (r = 0.7: 0.69999999 < 0.70000005)."""
    d = dims(as_3d(load_config(preset="tiny"), physics={"radius": r}))
    assert d.z_ground == d.z_lo


# --- the 3D rules (validate widened to dims = 3) ------------------------------------------------------------


@pytest.mark.usefixtures("enable_3d")
def test_pr2_3d_config_validates(micro_cfg: Config) -> None:
    load_config(
        preset="tiny",
        overrides=[
            *TINY3D,
            "physics.gravity=0.03",
            "physics.floor_eps=0.5",
            "physics.buoyancy_max=0.05",
            "mutation.buoy_init=0.5",
            "mutation.buoy_sigma=0.02",
            "fields.photic_amp=0.3",
            "fields.photic_lambda=8",
        ],
    )
    validate(as_3d(micro_cfg))  # the tests' 3D variant is a valid 3D config
    validate(as_3d(load_config(preset="tiny")))


# One crafted violation per PR 2 rule: (extra overrides on TINY3D, message fragment, the only problem).
BAD_PR2: list[tuple[list[str], str, bool]] = [
    (["world.height=8", "repro.offspring_distance=3.5"], "must be ≥ 2·repro.offspring_distance = 7.0", True),
    (
        ["physics.gravity=-0.01"],
        "physics.gravity -0.01 must be in [0, thrust_max·sin(PITCH_MAX) = 0.09848)",
        True,
    ),
    (["physics.thrust_max=0.05", "physics.gravity=0.05"], "physics.gravity 0.05 must be in [0, ", True),
    (["physics.gravity=nan"], "physics.gravity nan must be in", False),
    (["physics.floor_eps=-0.1"], "physics.floor_eps -0.1 must be ≥ 0", True),
    (["physics.floor_eps=3"], "radius + floor_eps < world.field_cell", True),
    (["physics.buoyancy_max=-0.1"], "physics.buoyancy_max must be ≥ 0 and finite", True),
    (["physics.buoyancy_max=inf"], "physics.buoyancy_max must be ≥ 0 and finite", False),
    (["physics.gravity=0.07"], "buoyancy_max − gravity) = 0.17 must be < 2·radius·k_repel = 0.16", True),
    (["physics.buoyancy_max=0.1"], "max(gravity, buoyancy_max − gravity) = 0.2 must be", True),
    (["physics.buoyancy_max=0.05", "mutation.buoy_init=1.5"], "mutation.buoy_init must be in [0, 1]", True),
    (["physics.buoyancy_max=0.05", "mutation.buoy_sigma=-0.01"], "mutation.buoy_sigma must be ≥ 0", True),
    (["mutation.buoy_init=0.5"], "must be 0 when physics.buoyancy_max = 0", True),
    (["mutation.buoy_sigma=0.02"], "must be 0 when physics.buoyancy_max = 0", True),
    (["fields.benthic_lambda=0"], "fields.benthic_lambda must be > 0 in 3D", True),
    (["fields.benthic_lambda=-8"], "fields.benthic_lambda must be > 0 in 3D", True),
    (["fields.photic_amp=1.5", "fields.photic_lambda=8"], "fields.photic_amp must be in [0, 1]", True),
    (["fields.photic_amp=0.3"], "fields.photic_lambda must be > 0 when fields.photic_amp > 0", True),
    (["director.enabled=true"], "director.enabled is not supported with world.dims = 3 yet", True),
]


@pytest.mark.usefixtures("enable_3d")
@pytest.mark.parametrize(("extra", "fragment", "alone"), BAD_PR2)
def test_each_pr2_rule_rejects(extra: list[str], fragment: str, alone: bool) -> None:
    msg = _problems([*TINY3D, *extra])
    assert fragment in msg, msg
    if alone:
        assert msg.startswith("1 config problem(s)"), msg


@pytest.mark.parametrize(("extra", "fragment", "alone"), BAD_PR2)
def test_pr2_rules_never_fire_in_2d(extra: list[str], fragment: str, alone: bool) -> None:
    """The settings that are not 3D keys are valid in a 2D world: the PR 2 rules apply in 3D only."""
    over = [o for o in extra if tuple(o.split("=")[0].split(".")) not in THREE_D_KEYS]
    load_config(preset="tiny", overrides=over)
