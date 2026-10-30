"""The sphere world's configuration (DESIGN §46.9; sphere series S0, S2): five keys, all off by default and
invisible in a box world, and the validation rules (S2 enabled the sphere: ``GEOMETRY_SUPPORTED``)."""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import pytest

from alife import cli, config, geom
from alife.config import (
    Config,
    ConfigError,
    config_from_dict,
    config_hash,
    config_to_dict,
    dims,
    load_config,
    save_config,
    validate,
)
from sphere_helpers import SPHERE_SETS, sphere_cfg, sphere_enabled  # noqa: F401  (a fixture)

# config hashes of main at 5984583 (before the sphere keys existed); a box world must keep them
BASE_HASHES = {
    "tiny": "d25e0eaee7349ae366e3e4e60ebab107cf7a89fcf49be88a12d81099ff2561d5",
    "tiny3d": "90e799beeba91e8cb0e2e021ad9595340efa9975eb53f963047c63646e61797a",
    "64k3d": "df54434fe947dea8f36e5e23223f9676df1ee784959806c31eafb235012ee242",
    "mini3d": "f3c53cb87069bda5b43061d2d137992e2a780135733e6ca2ea3e01db69b30aa8",
}


def problems(which: str = "tiny3s", *overrides: str) -> str:
    """The text of the ConfigError of an invalid sphere config ('' when it is valid)."""
    try:
        sphere_cfg(which, *overrides)
    except ConfigError as e:
        return str(e)
    return ""


def test_keys_default_to_a_box_and_are_post_v01_keys() -> None:
    w, p = Config().world, Config().physics
    assert (w.geometry, w.radius, w.nside, w.layers, p.ceiling_band) == ("box", 0.0, 0, 0, 0.0)
    assert config.SPHERE_KEYS == (
        ("world", "geometry"),
        ("world", "radius"),
        ("world", "nside"),
        ("world", "layers"),
        ("physics", "ceiling_band"),
    )
    assert set(config.SPHERE_KEYS) <= set(config.POST_V01_KEYS) and set(config.SPHERE_KEYS) <= set(
        config.SERIAL_OMIT_KEYS
    )
    assert not set(config.SPHERE_KEYS) & (set(config.UNHASHED_KEYS) | set(config.OBSERVER_KEYS))
    assert config.GEOMETRY_SUPPORTED == ("box", "sphere")  # S2 enabled it
    # and gated the layers that are not converted: S3 converted the five features and the roles, S5 the
    # observers and the director; only the weather (S4) is left
    assert {
        "structures",
        "niche",
        "comm",
        "bonds",
        "roles",
        "arms",
        "observers",
        "director",
    } >= config.SPHERE_READY
    assert config.SPHERE_READY.isdisjoint({"weather"})


@pytest.mark.parametrize("preset", sorted(BASE_HASHES))
def test_a_box_world_keeps_its_hash_and_its_serialisation(preset: str) -> None:
    cfg = load_config(preset=preset)
    assert config_hash(cfg) == BASE_HASHES[preset]
    d = config_to_dict(cfg, omit_defaults=True)
    assert not any(key in d.get(sec, {}) for sec, key in config.SPHERE_KEYS)  # config.json is byte-identical


def test_the_dims_of_a_box_carry_box_neutral_sphere_fields() -> None:
    for preset in ("tiny", "tiny3d"):
        d = dims(load_config(preset=preset))
        assert (d.sph, d.radius, d.nside, d.npix, d.nside_h, d.npix_h, d.r_lo, d.r_hi) == (
            False,
            0.0,
            0,
            0,
            0,
            0,
            0.0,
            0.0,
        )
    d3 = dims(load_config(preset="tiny3d"))
    assert d3.q == 27 * d3.c and d3.fg > 0 and d3.g_hash > 0


def test_a_sphere_config_is_refused_while_the_geometry_is_not_supported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "GEOMETRY_SUPPORTED", ("box",))
    msg = problems("tiny3s")
    assert "world.geometry = 'sphere' is not enabled yet" in msg
    assert msg.count("config problem") == 1 and msg.startswith("1 config problem(s)")  # nothing else is wrong
    assert cli.main(["run", "--preset", "tiny3d", "--set", "world.geometry=sphere", "--quiet"]) == 2


def test_the_sphere_is_enabled_and_tiny3s_is_a_valid_preset() -> None:
    assert problems("tiny3s") == "" and config.GEOMETRY_SUPPORTED == ("box", "sphere")
    cfg = load_config(preset="tiny3s")
    d = dims(cfg)
    assert (cfg.world.geometry, cfg.world.radius, cfg.world.height, d.nside, d.fz) == (
        "sphere",
        36.0,
        16.0,
        8,
        8,
    )
    assert (d.nside_h, d.g_hash_z, d.c, d.q, d.k) == (8, 5, 32, 4000, 8)


@pytest.mark.usefixtures("sphere_enabled")
@pytest.mark.parametrize("which", sorted(SPHERE_SETS))
def test_the_presets_of_the_spec_validate(which: str) -> None:
    cfg = sphere_cfg(which)
    d = dims(cfg)
    assert d.sph and d.nd == 3 and d.d3 and d.fg == 0 and d.g_hash == 0 and d.cell_hash == float("inf")
    nside = int(SPHERE_SETS[which][2].split("=")[1])
    layers = int(SPHERE_SETS[which][3].split("=")[1])
    assert (d.nside, d.nside_h, d.fz) == (nside, nside, layers)  # Nh = Nf in every preset
    assert d.npix == 12 * nside**2 and d.n_cells == layers * d.npix and d.field_shape == (layers, d.npix)
    assert d.q == 125 * 32 and d.g_hash_z == {"tiny3s": 5, "mini3s": 7, "64k3s": 13}[which]
    assert (
        d.r_lo == pytest.approx(cfg.world.radius + 1.0, abs=1e-4)
        and d.r_hi < cfg.world.radius + cfg.world.height
    )
    assert d.z_ground == pytest.approx(1.05, abs=1e-6)


@pytest.mark.usefixtures("sphere_enabled")
def test_a_sphere_config_round_trips_through_json_and_overrides(tmp_path: Path) -> None:
    cfg = sphere_cfg("tiny3s", "physics.ceiling_band=2.0")
    d = config_to_dict(cfg, omit_defaults=True)
    assert (
        d["world"]["geometry"] == "sphere"
        and d["world"]["radius"] == 36.0
        and d["physics"]["ceiling_band"] == 2.0
    )
    assert config_from_dict(d) == cfg
    save_config(cfg, tmp_path / "c.json")
    assert config_from_dict(json.loads((tmp_path / "c.json").read_text())) == cfg
    assert config_hash(cfg) != BASE_HASHES["tiny3d"]  # a sphere is a different world
    assert config_hash(sphere_cfg("tiny3s")) != config_hash(sphere_cfg("tiny3s", "physics.ceiling_band=2.0"))


@pytest.mark.usefixtures("sphere_enabled")
@pytest.mark.parametrize(
    "overrides,expect",
    [
        (["world.geometry=torus"], "world.geometry must be one of"),
        (["world.dims=2"], "needs world.dims = 3"),
        (["world.radius=15"], "world.radius must be finite and in [20, "),
        (["world.height=2"], "world.height must be finite and > 2·physics.radius"),
        (["world.height=40"], "must be ≤ world.radius"),
        (["world.size=200"], "world.size 200.0 must be > 4·(radius + height) = 208"),
        (["world.nside=1"], "world.nside must be ≥ 2"),
        (["world.layers=1"], "world.layers must be ≥ 2"),
        (["physics.floor_eps=0.0"], "physics.floor_eps must be ≥ 0.01 on a sphere"),
        (["physics.floor_eps=2.5"], "must be < z_1"),
        (["world.nside=40"], "the cell volume"),
        (["world.nside=300", "world.layers=16"], "must be ≤ 2^20"),
        (["sensing.cell_capacity=8"], "sensing.cell_capacity 8 must be ≥ ceil(0.167·V_lid) + 4 = 28"),
        (["physics.ceiling_band=-1"], "physics.ceiling_band must be in [0, height"),
        (["physics.ceiling_band=14"], "physics.ceiling_band must be in [0, height"),
        (["sensing.k_neighbors=4001"], "must be in [NS="),
    ],
)
def test_every_sphere_rule_has_a_message(overrides: list[str], expect: str) -> None:
    assert expect in problems("tiny3s", *overrides)


@pytest.mark.usefixtures("sphere_enabled")
def test_the_hash_rules_need_a_planet_the_sensing_radius_can_cover(monkeypatch: pytest.MonkeyPatch) -> None:
    # a ball almost as wide as the planet: allowed by the planet-size rule, but no Nside >= 2 covers it
    assert "no hash Nside >= 2 covers" in problems("tiny3s", "sensing.radius=71")
    # a hash Nside beyond the cap is refused with the remedy
    with monkeypatch.context() as m:
        m.setattr(config.geom, "estimate_nh", lambda angle: config.NH_MAX + 1)
        assert "would exceed 64" in problems("tiny3s")
    # the lid cell holds more as the shell gets thick: the capacity rule follows V_lid
    assert "must be ≥ ceil(0.167·V_lid) + 4 = 30" in problems("mini3s", "sensing.cell_capacity=20")


def test_the_sphere_keys_are_inert_in_a_box_world() -> None:
    for sets, key in [
        (["world.radius=40"], "world.radius"),
        (["world.nside=8"], "world.nside"),
        (["world.layers=8"], "world.layers"),
        (["physics.ceiling_band=1.0"], "physics.ceiling_band"),
    ]:
        with pytest.raises(ConfigError, match=f"{key} is inert while world.geometry is box"):
            load_config(preset="tiny3d", overrides=sets)
    validate(load_config(preset="tiny3d"))  # and the box itself is valid


@pytest.mark.usefixtures("sphere_enabled")
@pytest.mark.parametrize(
    "overrides,expect",
    [
        (["world.layers=100000000000"], "must be ≤ 2^20"),  # a typo of a count must not allocate 800 GB
        (["world.layers=1000000000000"], "must be ≤ 2^20"),
        (["world.nside=100000000", "world.layers=2"], "must be ≤ 2^20"),
        (["world.radius=1e200", "world.height=1", "world.size=1e201"], "world.radius must be finite and in"),
        (["world.radius=1e308", "world.height=1", "world.size=1e308"], "world.radius must be finite and in"),
        (["world.radius=1e7", "world.height=10", "world.size=1e8"], "world.radius must be finite and in"),
        (["world.size=1e308"], ""),  # a huge image length is fine: it only has to exceed 4(Rp + H)
    ],
)
def test_absurd_sizes_raise_a_config_error_before_anything_is_allocated(
    overrides: list[str], expect: str
) -> None:
    msg = problems("tiny3s", *overrides)
    assert (expect in msg) if expect else msg == ""


@pytest.mark.usefixtures("sphere_enabled")
def test_nothing_is_computed_for_a_config_that_fails_a_cheap_rule(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ordering (M5): the exact hash derivation only runs when every cheap rule passed."""

    def boom(*a, **k):
        raise AssertionError("the hash derivation ran for a refused config")

    monkeypatch.setattr(config.geom, "derive_nh", boom)
    monkeypatch.setattr(config.geom, "estimate_nh", boom)
    assert "world.layers must be ≥ 2" in problems("tiny3s", "world.layers=1")
    assert "world.height 40.0 must be ≤ world.radius" in problems("tiny3s", "world.height=40")
    monkeypatch.setattr(config, "GEOMETRY_SUPPORTED", ("box",))  # the refusal is alone and immediate
    assert problems("64k3s").startswith("1 config problem(s): world.geometry = 'sphere' is not enabled yet")


@pytest.mark.usefixtures("sphere_enabled")
def test_a_planet_the_sensing_radius_cannot_cover_is_refused_and_dims_survives_it() -> None:
    over = ["world.radius=20", "world.height=16", "world.nside=4", "world.layers=6", "sensing.radius=45"]
    msg = problems("tiny3s", *over)
    assert "sensing.radius 45.0 must be < 2·(world.radius + physics.radius) = 42.0" in msg
    assert "math domain" not in msg
    # dims() is defined even for the config validate refused (no math domain error): no hash Nside, nh = 0
    cfg = dataclasses.replace(
        load_config(preset="tiny3d"),
        world=dataclasses.replace(
            Config().world,
            geometry="sphere",
            dims=3,
            size=1024.0,
            radius=20.0,
            height=16.0,
            nside=4,
            layers=6,
        ),
        sensing=dataclasses.replace(Config().sensing, radius=45.0, cell_capacity=32),
    )
    d = dims(cfg)
    assert d.sph and d.nside_h == 0 and d.npix_h == 0
    assert geom.rigorous_angle(20.0, 1.0, 45.0) == pytest.approx(math.pi)
    assert geom.derive_nh(20.0, 1.0, 45.0) == 0 and geom.derive_nh(1.0, 1.0, 1e9) == 0
