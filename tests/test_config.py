"""§17 config: defaults, presets, validation rules, loading precedence, overrides, round trip and hashing."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import mlx.core as mx
import pytest

from alife import layout
from alife.config import (
    OVERLAYS,
    PRESETS,
    R_HARD,
    THREE_D_KEYS,
    UNHASHED_KEYS,
    Config,
    ConfigError,
    apply_overrides,
    config_from_dict,
    config_hash,
    config_to_dict,
    dims,
    load_config,
    save_config,
    stability_number,
    validate,
)


def _with(cfg: Config, **sections: dict) -> Config:
    """Replace fields in sections without validation."""
    for name, vals in sections.items():
        cfg = dataclasses.replace(cfg, **{name: dataclasses.replace(getattr(cfg, name), **vals)})
    return cfg


def test_defaults_are_64k_and_valid() -> None:
    cfg = Config()
    validate(cfg)
    assert load_config() == cfg
    assert load_config(preset="64k") == cfg


@pytest.mark.parametrize("preset", PRESETS)
@pytest.mark.parametrize("overlays", [(), ("fast",)])
def test_presets_and_overlays_validate(preset: str, overlays: tuple[str, ...]) -> None:
    cfg = load_config(preset=preset, overlays=overlays)
    validate(cfg)
    assert OVERLAYS == ("fast",)


@pytest.mark.parametrize(
    ("preset", "n", "m", "n_init", "size", "g_hash", "fg"),
    [
        ("tiny", 1024, 32, 256, 128.0, 21, 32),
        ("dev", 16384, 256, 4096, 512.0, 85, 128),
        ("64k", 65536, 1024, 16384, 1024.0, 170, 256),
        ("256k", 262144, 4096, 65536, 2048.0, 341, 512),
    ],
)
def test_preset_table(preset: str, n: int, m: int, n_init: int, size: float, g_hash: int, fg: int) -> None:
    cfg = load_config(preset=preset)
    d = dims(cfg)
    assert (d.n, d.m, cfg.population.n_init, cfg.world.size) == (n, m, n_init, size)
    assert (d.g_hash, d.fg) == (g_hash, fg)
    assert d.n_cap == n - m and d.dl == 2 * m
    assert d.cell_hash >= cfg.sensing.radius


def test_preset_table_tiny3d() -> None:
    """``tiny3d`` is ``tiny`` plus every 3D key and a 3D-stable ``chem_diffusion`` (§27.8); with the fast
    overlay it validates too."""
    cfg, tiny = load_config(preset="tiny3d"), load_config(preset="tiny")
    d = dims(cfg)
    assert "tiny3d" in PRESETS and cfg.world.dims == 3
    assert (d.n, d.m, cfg.population.n_init, cfg.world.size, d.g_hash, d.fg) == (1024, 32, 256, 128.0, 21, 32)
    assert (d.fz, d.g_hash_z, d.n_cells, d.field_shape) == (6, 4, 6 * 32 * 32, (6, 32, 32))
    assert (d.lay.D_IN, d.lay.D_OUT, d.q, d.w1_rows, d.params) == (84, 39, 27 * 16, 116, 4999)
    a, b = config_to_dict(cfg), config_to_dict(tiny)
    changed = {(sec, k) for sec in a for k in a[sec] if a[sec][k] != b[sec][k]}
    assert changed == {*THREE_D_KEYS, ("fields", "chem_diffusion")}
    assert max(cfg.fields.chem_diffusion) <= 1 / 6
    ph, en, mu, fi = cfg.physics, cfg.energy, cfg.mutation, cfg.fields
    assert (cfg.world.height, ph.gravity, ph.floor_eps, ph.buoyancy_max) == (24.0, 0.03, 0.1, 0.04)
    assert (en.buoyancy_cost, en.buoyancy_cap_cost, mu.buoy_init, mu.buoy_sigma) == (16, 8, 0.5, 0.02)
    assert (fi.benthic_lambda, fi.photic_amp, fi.photic_lambda) == (8.0, 0.5, 8.0)
    fast = load_config(preset="tiny3d", overlays=["fast"])
    assert (fast.sensing.k_neighbors, dims(fast).q) == (6, 27 * 12)


def test_tiny_and_fast_overrides() -> None:
    tiny = load_config(preset="tiny")
    assert (tiny.fields.n_patches, tiny.fields.patch_sigma) == (4, 12.0)
    assert (tiny.run.steps, tiny.run.metrics_every, tiny.run.checkpoint_every) == (500, 10, 50)
    assert tiny.run.heavy_metrics_every == 10
    fast = load_config(preset="tiny", overlays=["fast"])
    assert (fast.sensing.k_neighbors, fast.sensing.cell_capacity, fast.brain.hidden) == (6, 12, 24)
    assert dims(fast).q == 108 and dims(fast).w1_rows == layout.D_IN + 24


def test_dims_defaults() -> None:
    d = dims(Config())
    assert (d.k, d.c, d.q, d.h, d.w1_rows, d.w2_rows) == (8, 16, 144, 32, 101, 33)
    assert d.params == 101 * 32 + 33 * 35 == 4387
    assert d.pp == 24 and d.wdtype == mx.float16
    assert d.occ_limit == min(2**15 - 1, (2**31 - 1 - R_HARD) // (2 * 1_000_000 + 10_000)) == 1060
    assert dims(load_config(preset="tiny", overrides=["brain.weight_dtype=bfloat16"])).wdtype == mx.bfloat16


def test_stability_number() -> None:
    assert stability_number(Config()) == pytest.approx(2 * (4 * 0.05 + 8 * 0.08) / 1.0)
    assert stability_number(Config()) == pytest.approx(1.68)


# One crafted violation per validate rule (§17): (section overrides, expected message fragment).
BAD_CASES: list[tuple[dict, str]] = [
    ({"world": {"field_cell": 5.0}}, "integer"),
    ({"world": {"size": 8.0, "field_cell": 4.0}}, "field grid"),
    ({"world": {"size": 4096.0 * 2, "field_cell": 4.0}}, "field grid"),
    ({"sensing": {"radius": 500.0}}, "G_h"),
    ({"bonds": {"form_range": 3.9}, "sensing": {"radius": 3.5}}, "sensing.radius"),
    ({"sensing": {"k_neighbors": 2}}, "k_neighbors"),
    ({"sensing": {"k_neighbors": 145}}, "k_neighbors"),
    ({"population": {"n_max": 32, "max_births_per_step": 4, "n_init": 4}}, "n_max"),
    ({"population": {"n_max": (1 << 20) + 4}}, "n_max"),
    ({"population": {"max_births_per_step": 0}}, "max_births_per_step"),
    ({"population": {"max_births_per_step": 16385}}, "max_births_per_step"),
    ({"population": {"n_init": 0}}, "n_init"),
    ({"population": {"n_init": 64513}}, "n_init"),
    ({"population": {"immigration_floor": 64513}}, "immigration_floor"),
    ({"population": {"immigration_floor": 4, "immigrants_per_step": 0}}, "immigrants_per_step"),
    ({"physics": {"k_repel": 0.2}}, "stability_number"),
    ({"physics": {"thrust_max": 0.16}}, "thrust_max"),
    ({"physics": {"v_max": 1.5}}, "v_max"),
    ({"physics": {"thrust_contact_limit": 0.0}}, "thrust_contact_limit"),
    ({"bonds": {"contract_gain": 1.0}}, "rest_length"),
    ({"bonds": {"form_range": 4.0}, "sensing": {"radius": 6.0}}, "form_range"),
    ({"repro": {"offspring_distance": 4.0}}, "offspring_distance"),
    ({"energy": {"e_max": 1 << 20}}, "e_max"),
    ({"energy": {"e_init": 1_000_001}}, "e_init"),
    ({"energy": {"structure": 1 << 20}}, "structure"),
    ({"energy": {"structure": -1}}, "structure"),
    ({"repro": {"min_energy": 12000}}, "min_energy"),
    ({"repro": {"min_energy": 12005}, "mutation": {"frac_min": 0.1}}, "frac_min"),
    # Quantised split (§10.7): frac_q8 rounds to 0 / 256 although the unquantised floor rule held.
    ({"mutation": {"frac_min": 0.001, "frac_init": 0.001}}, "frac_min"),
    ({"mutation": {"frac_max": 0.999, "frac_init": 0.999}}, "frac_max"),
    ({"repro": {"min_energy": 12200}, "mutation": {"frac_min": 0.00582}}, "frac_min"),  # q=1: 200·1>>8 = 0
    ({"energy": {"e_init": 0}}, "e_init"),
    ({"energy": {"harvest_max": 1 << 16}}, "harvest_max"),
    ({"combat": {"attack_max": 1 << 16}}, "attack_max"),
    ({"bonds": {"share_rate": 1 << 16}}, "share_rate"),
    ({"fields": {"resource_cap": 0}}, "resource_cap"),
    ({"fields": {"resource_cap": R_HARD}}, "resource_cap"),
    ({"fields": {"regen_max": 9000}}, "regen_max"),
    ({"fields": {"resource_diffusion_q8": 65}}, "resource_diffusion_q8"),
    ({"fields": {"chem_diffusion": (0.2, 0.3, 0.1)}}, "chem_diffusion"),
    ({"fields": {"chem_decay": (0.1, 1.0, 0.1)}}, "chem_decay"),
    ({"fields": {"chem_diffusion": (0.2, 0.1)}}, "chem_diffusion"),
    ({"fields": {"chem_decay": (0.1, 0.1, 0.1, 0.1)}}, "chem_decay"),
    ({"bonds": {"share_efficiency": 0.0}}, "share_efficiency"),
    ({"combat": {"efficiency": 1.5}}, "combat.efficiency"),
    ({"combat": {"defend_efficiency": 0.0}}, "defend_efficiency"),
    ({"mutation": {"frac_min": 0.6}}, "frac_min"),
    ({"mutation": {"frac_max": 1.0}}, "frac_min"),
    ({"mutation": {"sigma_min": 0.0}}, "sigma_min"),
    ({"mutation": {"sigma_init": 0.5}}, "sigma_min"),
    ({"mutation": {"rate_max": 1.0}}, "rate_min"),
    ({"mutation": {"rate_init": 0.0005}}, "rate_min"),
    ({"brain": {"weight_dtype": "float64"}}, "weight_dtype"),
    ({"brain": {"impl": "loop"}}, "impl"),
    ({"run": {"runtime_checks": "full"}}, "runtime_checks"),
    ({"run": {"checkpoint_every": 150}}, "checkpoint_every"),
    ({"run": {"heavy_metrics_every": 150}}, "heavy_metrics_every"),
    ({"brain": {"hidden": 0}}, "hidden"),
    ({"physics": {"drag": 0.0}}, "drag"),
    ({"fields": {"patch_sigma": 0.0}}, "patch_sigma"),
    ({"population": {"immigration_every": -1}}, "immigration_every"),
    ({"population": {"immigration_every": 10, "immigration_count": 0}}, "immigration_count"),
    ({"population": {"immigration_every": 10, "immigration_count": 1025}}, "immigration_count"),
    ({"population": {"immigration_count": -1}}, "immigration_count"),
    ({"fields": {"season_period": -1}}, "season_period"),
    ({"fields": {"season_amplitude": 1.5}}, "season_amplitude"),
    ({"fields": {"season_amplitude": -0.1}}, "season_amplitude"),
    ({"fields": {"season_amplitude": 0.5, "season_period": 0}}, "season_period"),
    ({"fields": {"season_mode": "sine"}}, "season_mode"),
]


@pytest.mark.parametrize(("sections", "fragment"), BAD_CASES)
def test_each_rule_rejects(sections: dict, fragment: str) -> None:
    with pytest.raises(ConfigError, match=fragment):
        validate(_with(Config(), **sections))


def test_all_problems_reported_at_once() -> None:
    bad = _with(
        Config(),
        energy={"e_max": 1 << 21},
        brain={"impl": "nope"},
        run={"checkpoint_every": 7},
        fields={"resource_diffusion_q8": 100},
    )
    with pytest.raises(ConfigError) as ei:
        validate(bad)
    msg = str(ei.value)
    for frag in ("e_max", "brain.impl", "checkpoint_every", "resource_diffusion_q8"):
        assert frag in msg
    assert msg.startswith("4 config problem(s)")


def test_toml_and_json_load(tmp_path: Path) -> None:
    toml = tmp_path / "c.toml"
    toml.write_text("[population]\nn_init = 100\n[fields]\nchem_decay = [0.1, 0.02, 0.001]\n")
    cfg = load_config(toml, preset="tiny")
    assert cfg.population.n_init == 100 and cfg.fields.chem_decay == (0.1, 0.02, 0.001)
    js = tmp_path / "c.json"
    js.write_text(json.dumps({"combat": {"attack_bonded": True}, "world": {"size": 96}}))
    cfg = load_config(js, preset="tiny")
    assert cfg.combat.attack_bonded is True and cfg.world.size == 96.0 and isinstance(cfg.world.size, float)
    with pytest.raises(ConfigError):
        load_config(tmp_path / "c.yaml")


def test_precedence(tmp_path: Path) -> None:
    # defaults < preset < overlay < file < --set
    assert load_config(preset="tiny").sensing.k_neighbors == 8
    assert load_config(preset="tiny", overlays=["fast"]).sensing.k_neighbors == 6
    f = tmp_path / "f.toml"
    f.write_text("[sensing]\nk_neighbors = 7\n[population]\nn_max = 2048\n")
    cfg = load_config(f, preset="tiny", overlays=["fast"])
    assert cfg.sensing.k_neighbors == 7 and cfg.population.n_max == 2048
    assert cfg.population.n_init == 256  # preset value survives where the file is silent
    assert cfg.brain.hidden == 24  # overlay survives where the file is silent
    cfg = load_config(f, preset="tiny", overlays=["fast"], overrides=["sensing.k_neighbors=5"])
    assert cfg.sensing.k_neighbors == 5


def test_typed_override_parsing() -> None:
    cfg = apply_overrides(
        Config(),
        [
            "population.n_init=1000",
            "physics.drag=2",
            "combat.attack_bonded=true",
            "brain.weight_dtype=float32",
            "fields.chem_diffusion=0.2,0.1,0.2",
            "run.compile=False",
        ],
    )
    assert cfg.population.n_init == 1000 and isinstance(cfg.population.n_init, int)
    assert cfg.physics.drag == 2.0 and isinstance(cfg.physics.drag, float)
    assert cfg.combat.attack_bonded is True and cfg.run.compile is False
    assert cfg.brain.weight_dtype == "float32"
    assert cfg.fields.chem_diffusion == (0.2, 0.1, 0.2)
    for bad in ("population.n_init=1.5", "combat.attack_bonded=yes", "physics.drag=fast", "noequals"):
        with pytest.raises(ConfigError):
            apply_overrides(Config(), [bad])
    with pytest.raises(ConfigError, match="k_neighbors"):  # overrides are validated
        apply_overrides(Config(), ["sensing.k_neighbors=1"])


def test_unknown_keys_rejected(tmp_path: Path) -> None:
    for bad in ("nosuch.key=1", "world.nosuch=1"):
        with pytest.raises(ConfigError, match="unknown"):
            load_config(overrides=[bad])
    with pytest.raises(ConfigError, match="unknown"):
        config_from_dict({"world": {"size": 64.0, "colour": 1}})
    with pytest.raises(ConfigError, match="unknown"):
        config_from_dict({"galaxy": {}})
    with pytest.raises(ConfigError):
        config_from_dict({"world": {"size": "big"}})
    with pytest.raises(ConfigError):
        config_from_dict({"population": {"n_max": True}})
    f = tmp_path / "bad.toml"
    f.write_text("[world]\nsize = 64.0\nwrap = true\n")
    with pytest.raises(ConfigError, match=r"world\.wrap"):
        load_config(f)
    with pytest.raises(ConfigError):
        load_config(preset="huge")
    with pytest.raises(ConfigError):
        load_config(overlays=["slow"])


def test_round_trip_and_save(tmp_path: Path) -> None:
    for cfg in (
        Config(),
        load_config(preset="tiny", overlays=["fast"], overrides=["combat.attack_bonded=true"]),
    ):
        assert config_from_dict(config_to_dict(cfg)) == cfg
        p = tmp_path / "config.json"
        save_config(cfg, p)
        text = p.read_text()
        data = json.loads(text)
        assert list(data) == sorted(data)
        assert isinstance(data["fields"]["chem_diffusion"], list)
        assert config_from_dict(data) == cfg
        assert load_config(p, preset="tiny") == cfg
        assert text == json.dumps(data, sort_keys=True, indent=2) + "\n"


def test_hash_ignores_run_and_tracks_dynamics() -> None:
    base = Config()
    h = config_hash(base)
    assert len(h) == 64 and h == config_hash(Config())
    assert config_hash(_with(base, run={"seed": 7, "steps": 3, "compile": False})) == h
    for sec, vals in config_to_dict(base).items():
        if sec == "run":
            continue
        for key, val in vals.items():
            if (sec, key) in UNHASHED_KEYS:
                continue  # the director's policy is read between steps only (see test_director_config)
            if isinstance(val, bool):
                new = not val
            elif isinstance(val, int):
                new = val + 1
            elif isinstance(val, float):
                new = val * 1.5 + 0.25
            elif isinstance(val, str):
                new = val + "x"
            else:
                new = (*val[:-1], val[-1] + 0.001)
            changed = config_from_dict({**config_to_dict(base), sec: {**vals, key: new}})
            assert config_hash(changed) != h, f"{sec}.{key} does not affect the hash"


def test_config_is_hashable_and_dims_cached() -> None:
    cfg = load_config(preset="tiny")
    assert dims(cfg) is dims(load_config(preset="tiny"))
