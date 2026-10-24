"""§27.1 the dims constant: ``world.dims`` / ``world.height``, the validation gate, the hash and the ``Dims``
fields per dims."""

from __future__ import annotations

import math

import pytest

from alife import config, layout
from alife.config import (
    OVERLAYS,
    POST_V01_KEYS,
    PRESETS,
    THREE_D_KEYS,
    UNHASHED_KEYS,
    Config,
    ConfigError,
    config_from_dict,
    config_hash,
    config_to_dict,
    dims,
    is_3d,
    load_config,
    validate,
)

# Pinned in test_features.py and test_director_config.py; re-asserted: the 3D keys must not move them.
TINY_HASH_V01 = "d25e0eaee7349ae366e3e4e60ebab107cf7a89fcf49be88a12d81099ff2561d5"
K64_HASH = "69b4e21ad6b6016b17c11d6a1d9b3371c03b4b9f590fdb4a04752515c13a01d8"

# A valid 3D variant of ``tiny`` once 3D is enabled: height 24 (FZ 6, G_z 4), chem_diffusion ≤ 1/6, and (PR 2,
# §27.5) a benthic food band.
TINY3D = ["world.dims=3", "world.height=24", "fields.chem_diffusion=0.16,0.1,0.16", "fields.benthic_lambda=8"]


@pytest.fixture
def enable_3d(monkeypatch: pytest.MonkeyPatch) -> None:
    """``validate`` accepts dims = 3 (as it will from docs/3D.md PR 3)."""
    monkeypatch.setattr(config, "DIMS_SUPPORTED", (2, 3))


def _problems(overrides: list[str], preset: str = "tiny") -> str:
    with pytest.raises(ConfigError) as ei:
        load_config(preset=preset, overrides=overrides)
    return str(ei.value)


# --- defaults, keys, hashes -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "preset",
    [p for p in PRESETS if p not in ("tiny3d", "64k3d", "mini3d")],  # the 3D presets: test_config.py
)
@pytest.mark.parametrize("overlays", [(), OVERLAYS])
def test_defaults_are_2d(preset: str, overlays: tuple[str, ...]) -> None:
    cfg = load_config(preset=preset, overlays=overlays)
    assert (cfg.world.dims, cfg.world.height) == (2, 0.0) and not is_3d(cfg)
    assert (Config().world.dims, Config().world.height) == (2, 0.0)


def test_pinned_hashes_unchanged() -> None:
    assert config_hash(load_config(preset="tiny")) == TINY_HASH_V01
    assert config_hash(load_config(preset="64k")) == K64_HASH


def test_3d_keys_are_post_v01_and_hashed() -> None:
    assert THREE_D_KEYS[:2] == (("world", "dims"), ("world", "height"))  # PR 2 appends its keys
    assert set(THREE_D_KEYS) <= set(POST_V01_KEYS) and not set(THREE_D_KEYS) & set(UNHASHED_KEYS)
    assert config.DIMS_SUPPORTED == (2, 3)  # enabled in PR 3


def test_hash_moves_off_default() -> None:
    base = config_to_dict(load_config(preset="tiny"))
    for world in ({"dims": 3}, {"height": 24.0}, {"dims": 3, "height": 24.0}):
        changed = config_from_dict({**base, "world": {**base["world"], **world}})
        assert config_hash(changed) != TINY_HASH_V01, world
    explicit = config_from_dict({**base, "world": {**base["world"], "dims": 2, "height": 0.0}})
    assert config_hash(explicit) == TINY_HASH_V01


def test_old_dict_loads_equal() -> None:
    cfg = load_config(preset="tiny")
    old = config_to_dict(cfg)
    del old["world"]["dims"], old["world"]["height"]  # a config.json written before 3D existed
    back = config_from_dict(old)
    validate(back)
    assert back == cfg and config_hash(back) == TINY_HASH_V01


def test_dims_is_an_int_key() -> None:
    with pytest.raises(ConfigError):
        config_from_dict({"world": {"dims": 3.0}})
    with pytest.raises(ConfigError):
        load_config(overrides=["world.dims=3.5"])


# --- the gate and the 2D rules ----------------------------------------------------------------------------


def test_dims3_rejected_until_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "DIMS_SUPPORTED", (2,))  # the gate as it stood before PR 3 enabled 3D
    msg = _problems(TINY3D)
    assert msg.startswith("1 config problem(s)") and "not enabled yet" in msg and "docs/3D.md" in msg
    assert "world.dims must be 2 or 3" in _problems(["world.dims=4"])
    assert "world.dims must be 2 or 3" in _problems(["world.dims=1"])
    # The 3D rules run while dims = 3 is gated: a non-finite size is reported, never an OverflowError.
    msg = _problems([*TINY3D, "world.size=inf"])
    assert msg.startswith("2 config problem(s)") and "not enabled yet" in msg and "positive and finite" in msg


def test_3d_keys_rejected_in_2d() -> None:
    msg = _problems(["world.height=10"])
    assert msg.startswith("1 config problem(s)") and "world.height is a 3D key" in msg


def test_2d_messages_unchanged() -> None:
    """The 2D neighbour rule keeps its exact text (test_config BAD_CASES match fragments of it)."""
    msg = _problems(["sensing.k_neighbors=145"])
    assert "sensing.k_neighbors 145 must be in [NS=3, 9·C=144]" in msg and "27·C" not in msg


def test_dims_2d_fields() -> None:
    for cfg in (Config(), load_config(preset="tiny")):
        d = dims(cfg)
        assert (d.nd, d.d3, d.fz, d.g_hash_z) == (2, False, 1, 1) and math.isinf(d.cell_hash_z)
        assert d.n_cells == d.fg * d.fg and d.field_shape == (d.fg, d.fg)
        assert d.lay is layout.for_dims(2) and d.q == 9 * d.c == 144
        assert (d.w1_rows, d.params) == (101, 4387)


# --- 3D (validate widened to dims = 3) ---------------------------------------------------------------------


@pytest.mark.usefixtures("enable_3d")
def test_dims_3d_fields() -> None:
    cfg = load_config(preset="tiny", overrides=TINY3D)
    assert is_3d(cfg)
    d = dims(cfg)
    assert (d.nd, d.d3, d.fz, d.g_hash_z, d.cell_hash_z) == (3, True, 6, 4, 6.0)
    assert (d.fg, d.n_cells, d.field_shape) == (32, 6 * 32 * 32, (6, 32, 32))
    assert d.lay is layout.for_dims(3) and d.q == 27 * 16 == 432
    assert (d.w1_rows, d.w2_rows, d.params) == (116, 33, 4999)
    # The 2D values are untouched by the dims switch.
    assert (d.n, d.m, d.g_hash, d.cell_hash, d.h) == (1024, 32, 21, 128 / 21, 32)


@pytest.mark.usefixtures("enable_3d")
def test_3d_presets_validate() -> None:
    load_config(preset="tiny", overrides=TINY3D)
    load_config(preset="tiny", overlays=["fast"], overrides=TINY3D)  # K 6 ≤ 27·12
    # The 2D neighbour bound does not apply in 3D: 145 > 9·C but ≤ 27·C (the stability rule still fires).
    assert "k_neighbors" not in _problems([*TINY3D, "sensing.k_neighbors=145"])


# One crafted violation per 3D world rule: (extra overrides on TINY3D, message fragment, the only problem).
BAD_3D: list[tuple[list[str], str, bool]] = [
    (["world.size=1024", "world.height=80"], "FZ·FG² = 1310720 must be ≤ 2^20", True),
    (["world.height=22"], "world.height / world.field_cell must be an integer", True),
    (["world.height=2"], "world.height must be finite and > 2·physics.radius", False),
    (["world.height=nan"], "world.height must be finite", False),
    (["world.height=inf"], "world.height must be finite", True),
    (["world.size=inf"], "world.size must be positive and finite, got inf", True),  # not an OverflowError
    (["world.height=4"], "field layers FZ = 1 must be ≥ 2", False),
    # G_z = floor(6 / 8) = 0, with height − 2r = 4 ≥ 2·offspring_distance (the PR 2 rule) and FZ = 3.
    (["world.field_cell=2", "world.height=6", "sensing.radius=8"], "hash layers G_z", True),
    (["world.height=68"], "world.height 68.0 must be ≤ world.size / 2 = 64.0", True),
    (["sensing.cell_capacity=1", "sensing.k_neighbors=28"], "27·C=27", False),
    (["fields.chem_diffusion=0.2,0.1,0.16"], "chem_diffusion must be ≤ 1/6 in 3D", True),
    (["fields.resource_diffusion_q8=43"], "resource_diffusion_q8 must be ≤ 42 in 3D", True),
]


@pytest.mark.usefixtures("enable_3d")
@pytest.mark.parametrize(("extra", "fragment", "alone"), BAD_3D)
def test_each_3d_rule_rejects(extra: list[str], fragment: str, alone: bool) -> None:
    msg = _problems([*TINY3D, *extra])
    assert fragment in msg, msg
    if alone:
        assert msg.startswith("1 config problem(s)"), msg


@pytest.mark.parametrize(("extra", "fragment", "alone"), BAD_3D)
def test_3d_rules_never_fire_in_2d(extra: list[str], fragment: str, alone: bool) -> None:
    """The same settings in a 2D world trip none of the 3D rules (those that are also 2D errors aside)."""
    over = [o for o in extra if not o.startswith("world.height")]
    try:
        load_config(preset="tiny", overrides=over)
    except ConfigError as e:
        assert "3D" not in str(e) and "27·C" not in str(e) and "world.height" not in str(e)
