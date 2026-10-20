"""W1 build step 1 (DESIGN §30): the ``[weather]`` config section, its hash behaviour, ``validate_config``
(guards, W0 rules, world rules W-1/W-2, load rules R1-R4, coupling keys), the ``storm`` world overlay.

Pure Python on configs (CPU, fast). The 3D case is ``64k3d`` + ``storm`` (L = 320, 160-cell weather); the 2D
case is ``tiny`` at ``world.size = 320`` (``helpers``-free: nothing here steps the engine).
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import fields, replace

import pytest

from alife import cli
from alife import weather as W
from alife.config import (
    ALL_OVERLAYS,
    OBSERVER_KEYS,
    OVERLAYS,
    POST_V01_KEYS,
    PRESETS,
    UNHASHED_KEYS,
    WORLD_OVERLAYS,
    Config,
    ConfigError,
    WeatherConfig,
    apply_overrides,
    config_from_dict,
    config_hash,
    config_to_dict,
    dims,
    load_config,
    validate,
)

# config_hash of every preset (and preset + fast) at the commit before the weather section existed (4337484).
PRESET_HASHES = {
    "256k": "717498d6623af5d51716cb595f00f0954db8e718389f83f5c0891e458d2a42ad",
    "256k+fast": "15d636076a36ff1cad4eb64d212ee55561c949640d8718239db450bc3276d24f",
    "64k": "69b4e21ad6b6016b17c11d6a1d9b3371c03b4b9f590fdb4a04752515c13a01d8",
    "64k+fast": "8ea373cf9a6c0ca8d6029d8d96a6d37157e2962b37792da9cc25812e3511e132",
    "64k3d": "df54434fe947dea8f36e5e23223f9676df1ee784959806c31eafb235012ee242",
    "64k3d+fast": "d7ab917652afe79b01515487ef3a7138c188877ace91a1a8ace210ad485fbf6e",
    "dev": "3151fd847b14f7fa067ebee30a22af546dffbf2c75622eb6cd8f4a1798269a0d",
    "dev+fast": "08878178da3e0e42051f9d05a0206b03116d30e5f2e875b6b2d1981be010d85f",
    "tiny": "d25e0eaee7349ae366e3e4e60ebab107cf7a89fcf49be88a12d81099ff2561d5",
    "tiny+fast": "3c9d4894192bb83872309226539949c64b4cd16402e8c0499dfbd0c382c2d95c",
    "tiny3d": "90e799beeba91e8cb0e2e021ad9595340efa9975eb53f963047c63646e61797a",
    "tiny3d+fast": "fd56d2121a6f2bb680734dbdb134c03c73a2d10c350b27f1bbd024b8c966d1d2",
}


@pytest.fixture(scope="module")
def storm3d() -> Config:
    return load_config(preset="64k3d", overlays=["storm"])


@pytest.fixture(scope="module")
def storm2d() -> Config:
    return load_config(
        preset="tiny", overrides=["world.size=320", "weather.enabled=true", "weather.u_max=0.2"]
    )


def with_weather(cfg: Config, **kw) -> Config:
    return replace(cfg, weather=replace(cfg.weather, **kw))


def msgs(cfg: Config, **kw) -> list[str]:
    return W.validate_config(with_weather(cfg, **kw))


def has(ms: list[str], text: str) -> bool:
    return any(text in m for m in ms)


# --- the section, its defaults, hashing -----------------------------------------------------------


def test_the_section_has_the_33_keys_and_all_are_hashed() -> None:
    names = [f.name for f in fields(WeatherConfig)]
    assert len(names) == 33 and names[0] == "enabled"
    assert [(s, k) for s, k in POST_V01_KEYS if s == "weather"] == [("weather", n) for n in names]
    assert not [
        1 for sk in POST_V01_KEYS if sk[0] == "weather" and (sk in UNHASHED_KEYS or sk in OBSERVER_KEYS)
    ]
    assert [f.name for f in fields(Config)][-2:] == ["weather", "run"]


def test_defaults_agree_with_weather_params_by_field_name() -> None:
    """For every config key that has a ``WeatherParams`` field of the same name the defaults are equal (pools
    as regrouped 7-tuples), so a later change of a ``WeatherParams`` default cannot silently change a hashed
    config. ``params_from_config(Config())`` is not compared: the default world is 1024 wide (``dx`` 6.4)."""
    wc, wp = WeatherConfig(), W.WeatherParams()
    names = {f.name for f in fields(WeatherConfig)}
    own = names - {"enabled", "cells"} - {n for n in names if n.startswith(("rain_", "storm_"))}
    assert own <= {f.name for f in fields(W.WeatherParams)}
    for k in sorted(own - {"pools"}):
        assert getattr(wc, k) == getattr(wp, k), k
    assert tuple(wc.pools[i : i + 7] for i in range(0, len(wc.pools), 7)) == wp.pools == W.DEFAULT_POOLS
    assert len(wc.pools) == 42 and all(isinstance(x, float) for x in wc.pools)
    assert wc.cells == wp.n == 160
    # the second form: a 320-wide world builds the whole object, ENGINE_FIXED included
    c320 = replace(Config(), world=replace(Config().world, size=320.0))
    assert W.params_from_config(c320) == wp


def test_engine_fixed_equals_the_weather_params_defaults_it_stands_for() -> None:
    wp = W.WeatherParams()
    assert set(W.ENGINE_FIXED) <= {f.name for f in fields(W.WeatherParams)}
    for k, v in W.ENGINE_FIXED.items():
        assert getattr(wp, k) == v and type(getattr(wp, k)) is type(v), k
    # every WeatherParams field is a config key, derived from the world, or fixed
    derived = {"n", "dx", "height"}
    keys = {f.name for f in fields(WeatherConfig)}
    assert {f.name for f in fields(W.WeatherParams)} == (
        keys & {f.name for f in fields(W.WeatherParams)}
    ) | derived | set(W.ENGINE_FIXED)
    assert len(W.ENGINE_FIXED) == 19  # 12 numerics + the seven host detection fields


def test_params_from_config(storm3d: Config, storm2d: Config) -> None:
    p = W.params_from_config(storm3d)
    assert (p.n, p.dx, p.height, p.L) == (160, 2.0, 80.0, 320.0) and p == W.WeatherParams()
    q = W.params_from_config(storm2d)  # 2D: height = H1 + H2
    assert (q.n, q.dx, q.height) == (160, 2.0, 80.0) and q == replace(W.WeatherParams(), u_max=0.2)
    moved = with_weather(storm3d, gamma=20.0, H1=30.0, H2=50.0, cells=80)
    r = W.params_from_config(moved)
    assert (r.gamma, r.H1, r.H2, r.n, r.dx) == (20.0, 30.0, 50.0, 80, 4.0) and r.pools == p.pools
    assert W.params_from_config(storm3d) is W.params_from_config(storm3d)  # cached
    with pytest.raises(AttributeError):
        W.params_from_config(storm3d).storm_p  # noqa: B018 - the coupling keys are not WeatherParams fields


def test_the_pools_default_is_flat_and_regroups() -> None:
    assert WeatherConfig().pools == tuple(x for row in W.DEFAULT_POOLS for x in row)


@pytest.mark.parametrize("name", sorted(PRESET_HASHES))
def test_every_preset_hash_is_unchanged(name: str) -> None:
    preset, _, ov = name.partition("+")
    cfg = load_config(preset=preset, overlays=[ov] if ov else [])
    assert config_hash(cfg) == PRESET_HASHES[name]
    assert (
        config_to_dict(cfg)["weather"] == config_to_dict(Config())["weather"]
    )  # in config.json, hash-neutral


def test_the_weather_table_is_omitted_at_defaults_and_hashed_when_changed(storm3d: Config) -> None:
    base = load_config(preset="64k3d")
    d = config_to_dict(base)
    assert "weather" in d
    on = config_hash(storm3d)
    assert on != config_hash(base)
    # every key, changed, changes the hash (the compiled step reads them all)
    changes = {
        "cells": 80,
        "H1": 30.0,
        "H2": 50.0,
        "g_top": 0.009,
        "g_prime": 0.005,
        "f0": 0.02,
        "gamma": 20.0,
        "tau_c": 9.0,
        "k_E": 0.09,
        "u_gust": 0.01,
        "q_bg": 0.65,
        "q_pool": 1.1,
        "tau_rad": 2500.0,
        "r1": 3e-4,
        "c_d": 3e-3,
        "r2": 2e-4,
        "pools": storm3d.weather.pools[:35],
        "seed_rate": 0.03,
        "seed_v": 0.03,
        "seed_sigma": 14.0,
        "seed_jitter": 0.4,
        "alpha0": 0.2,
        "alpha1": 0.7,
        "u_max": 0.25,
        "w_edge": 0.02,
        "z_sl": 3.0,
        "rain_gain": 1e5,
        "rain_floor": 0.1,
        "storm_p": 0.01,
        "storm_u0": 0.05,
        "storm_span": 0.04,
        "storm_break_u": 0.07,
    }
    assert set(changes) | {"enabled"} == {f.name for f in fields(WeatherConfig)}
    for k, v in changes.items():
        assert config_hash(with_weather(storm3d, **{k: v})) != on, k


def test_a_weather_off_config_with_a_non_default_key_is_rejected() -> None:
    """W-0: every weather key is inert while ``enabled`` is false, so it must keep its default."""
    base = load_config(preset="tiny")
    for f in fields(WeatherConfig):
        if f.name == "enabled":
            continue
        default = getattr(WeatherConfig(), f.name)
        other = (
            default[:-7] if isinstance(default, tuple) else default + (1 if isinstance(default, int) else 0.5)
        )
        with pytest.raises(ConfigError, match=rf"weather\.{f.name} is inert while weather.enabled is false"):
            validate(with_weather(base, **{f.name: other}))
    validate(with_weather(base))  # the defaults pass
    with pytest.raises(ConfigError, match=r"weather\.storm_p is inert"):
        load_config(preset="tiny", overrides=["weather.storm_p=0.1"])


def test_the_dict_round_trip_keeps_pools(storm3d: Config) -> None:
    c = with_weather(storm3d, pools=(10.0, 20.0, 5.0, 6.0, 0.5, 0.001, 0.0))
    assert config_from_dict(json.loads(json.dumps(config_to_dict(c)))) == c
    assert isinstance(config_from_dict(config_to_dict(c)).weather.pools, tuple)
    assert config_from_dict(config_to_dict(storm3d)) == storm3d


def test_set_parses_pools(storm3d: Config) -> None:
    c = apply_overrides(storm3d, ["weather.pools=53.0,36.0,32.0,16.0,1.0,-0.009,0.0", "weather.gamma=20"])
    assert c.weather.pools == (53.0, 36.0, 32.0, 16.0, 1.0, -0.009, 0.0) and c.weather.gamma == 20.0
    with pytest.raises(ConfigError, match="7 numbers per pool"):
        apply_overrides(storm3d, ["weather.pools=1,2,3"])
    with pytest.raises(ConfigError, match="cannot parse"):
        apply_overrides(storm3d, ["weather.pools=a,b"])
    with pytest.raises(ConfigError, match=r"unknown config key weather\.nope"):
        apply_overrides(storm3d, ["weather.nope=1"])


# --- the storm overlay ----------------------------------------------------------------------------


def test_overlay_names() -> None:
    assert OVERLAYS == ("fast",) and WORLD_OVERLAYS == ("storm",) and ALL_OVERLAYS == ("fast", "storm")


def test_the_storm_overlay_composes_with_64k3d(storm3d: Config) -> None:
    assert (storm3d.world.size, storm3d.world.height) == (320.0, 80.0)
    wc = storm3d.weather
    assert wc.enabled and (wc.rain_gain, wc.storm_p, wc.storm_u0, wc.storm_span) == (
        50000.0,
        0.002,
        0.06,
        0.03,
    )
    assert wc.rain_floor == 0.0 and wc.storm_break_u == 0.0  # director runs need rain_floor = 0 (§5.3)
    # nothing else of the weather section moves
    assert dataclasses.replace(wc, enabled=False, rain_gain=0.0, storm_p=0.0) == WeatherConfig()
    # the rest of the config is 64k3d
    assert replace(storm3d, weather=WeatherConfig()) == load_config(preset="64k3d")
    assert W.validate_config(storm3d) == []


def test_the_storm_overlay_fails_on_a_world_that_cannot_hold_it() -> None:
    with pytest.raises(ConfigError, match="rule W-2"):  # tiny: size 128, the pools are L = 320 coordinates
        load_config(preset="tiny", overlays=["storm"])
    with pytest.raises(ConfigError, match="rule 4"):  # dx = 128/160 = 0.8: fast-mode Courant over 0.5
        load_config(preset="tiny", overlays=["storm"])
    with pytest.raises(ConfigError):
        load_config(preset="tiny3d", overlays=["storm"])


def test_unknown_overlays_name_all_of_them() -> None:
    with pytest.raises(ConfigError, match=r"\('fast', 'storm'\)"):
        load_config(overlays=["nope"])
    with pytest.raises(ConfigError, match="storm"):
        cli._layer(load_config(preset="tiny"), overlays=["nope"], path=None, overrides=())


def test_a_world_overlay_is_not_a_preset() -> None:
    with pytest.raises(ConfigError, match="unknown preset"):
        load_config(preset="storm")


def test_cli_lists_and_accepts_the_storm_overlay(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["run", "--help"]) == 0
    assert "storm" in capsys.readouterr().out
    base = load_config(preset="64k3d")
    assert cli._layer(base, overlays=["storm"], path=None, overrides=()) == load_config(
        preset="64k3d", overlays=["storm"]
    )


@pytest.mark.parametrize("preset", PRESETS)
@pytest.mark.parametrize("overlays", [(), OVERLAYS])
def test_presets_with_weather_off_validate(preset: str, overlays: tuple[str, ...]) -> None:
    cfg = load_config(preset=preset, overlays=overlays)
    assert not cfg.weather.enabled and WeatherConfig() == cfg.weather
    validate(cfg)


def test_dims_wn(storm3d: Config) -> None:
    assert dims(storm3d).wn == 160 == dims(load_config(preset="tiny")).wn
    assert dims(with_weather(storm3d, cells=80)).wn == 80


# --- validation: the guards return, they do not raise ---------------------------------------------


@pytest.mark.parametrize(
    ("override", "text"),
    [
        ({"cells": 0}, "weather.cells must be an integer in [8, 1024]"),
        ({"cells": 7}, "weather.cells must be an integer in [8, 1024]"),
        ({"cells": 100000}, "weather.cells must be an integer in [8, 1024]"),
        ({"cells": -160}, "weather.cells must be an integer in [8, 1024]"),
        ({"cells": 160.0}, "weather.cells must be an integer"),
        ({"cells": True}, "weather.cells must be an integer"),
        ({"pools": tuple(range(41))}, "7 numbers per pool"),
        ({"pools": (float("nan"),) + (1.0,) * 6}, "pools must be finite"),
        ({"pools": (float("inf"),) + (1.0,) * 6}, "pools must be finite"),
    ],
)
def test_the_guards_return_messages(storm3d: Config, override: dict, text: str) -> None:
    ms = msgs(storm3d, **override)
    assert has(ms, text) and len(ms) <= 3
    with pytest.raises(ConfigError, match=r"\d+ config problem\(s\)"):
        validate(with_weather(storm3d, **override))


def test_the_guards_hold_through_load_config_and_the_cli(tmp_path) -> None:
    for ov in ("weather.cells=0", "weather.cells=7", "weather.cells=100000"):
        with pytest.raises(ConfigError, match=r"weather\.cells"):
            load_config(preset="64k3d", overlays=["storm"], overrides=[ov])
        out = tmp_path / ov.replace("=", "_")
        args = ["run", "--preset", "64k3d", "--overlay", "storm", "--set", ov, "--out", str(out)]
        assert cli.main(args) == 2 and not out.exists()  # exit 2, never a traceback
    with pytest.raises(ConfigError):
        load_config(preset="64k3d", overlays=["storm"], overrides=["world.size=0"])
    zero = replace(load_config(preset="64k3d", overlays=["storm"]), world=replace(Config().world, size=0.0))
    assert has(W.validate_config(zero), "world.size > 0")  # returns, no ZeroDivisionError
    nan = with_weather(load_config(preset="64k3d", overlays=["storm"]), pools=(float("nan"),) * 7)
    with pytest.raises(ConfigError):
        validate(nan)


# --- validation: the W0 rules through the config --------------------------------------------------

# (rule tag, weather key overrides) for the rules the config keys can reach; rules 1, 14 and the fixed
# numerics are covered in tests/test_weather.py on ``WeatherParams``.
CONFIG_RULES = [
    ("rule 2", {"H1": 33.0}),
    ("rule 2", {"H2": 47.0}),
    ("rule 3", {"g_prime": 0.01}),
    ("rule 4", {"g_top": 0.03}),
    ("rule 5", {"f0": 0.6}),
    ("rule 6", {"f0": 0.06}),
    ("rule 7", {"tau_c": 6.0}),
    ("rule 8", {"gamma": 32.0}),
    ("rule 9", {"q_pool": 1.3}),
    ("rule 9", {"q_bg": 1.2}),
    ("rule 10", {"k_E": 4.0}),
    ("rule 11", {"tau_rad": 0.5}),
    ("rule 11", {"r1": -1e-4}),
    ("rule 12", {"seed_rate": 1.5}),
    ("rule 12", {"seed_sigma": 1.0}),
    ("rule 13", {"pools": (160.0, 36.0, 32.0, 0.0, 1.0, 0.0, 0.0)}),
    ("rule 13", {"pools": (160.0, 36.0, 32.0, 16.0, -1.0, 0.0, 0.0)}),
    ("rule 16", {"alpha1": 0.8}),
    ("rule 16", {"alpha0": -0.1}),
    ("rule 16", {"alpha1": -0.1}),
    ("rule 18", {"u_max": 0.0}),
    ("rule 18", {"w_edge": -0.01}),
    ("rule 18", {"z_sl": 0.0}),
]


@pytest.mark.parametrize(("tag", "override"), CONFIG_RULES, ids=lambda x: str(x)[:40])
def test_each_w0_rule_fails_on_a_violating_value(storm3d: Config, tag: str, override: dict) -> None:
    ms = msgs(storm3d, **override)
    assert any(m.startswith(tag + ":") or m.startswith(tag + " ") for m in ms), (tag, ms)
    with pytest.raises(ConfigError):
        validate(with_weather(storm3d, **override))


def test_the_world_rules(storm3d: Config, storm2d: Config) -> None:
    # W-1: the rain remap is a block mean onto the field lattice (FG = 80 in 64k3d)
    assert has(msgs(storm3d, cells=161), "rule W-1")
    assert has(msgs(storm3d, cells=100), "rule W-1")
    assert not has(msgs(storm3d, cells=240), "rule W-1")  # 240 % 80 == 0 (the rest fails other rules)
    # W-2: every pool centre lies in [0, size)
    p = list(storm3d.weather.pools)
    assert not has(msgs(storm3d), "W-2")
    for x, y in ((320.0, 36.0), (-1.0, 36.0), (10.0, 320.0), (10.0, -0.5)):
        q = tuple([x, y, *p[2:7], *p[7:]])
        assert has(msgs(storm3d, pools=q), "rule W-2"), (x, y)
    assert not has(msgs(storm3d, pools=tuple([0.0, 0.0, *p[2:]])), "W-2")
    assert msgs(storm3d, pools=tuple([319.999, 100.0, *p[2:]])) == []  # just inside the domain
    # 3D geometry (check_world): z_sl within a field cell, H1 + H2 = world.height
    assert has(msgs(storm3d, z_sl=5.0), "rule 15")
    assert has(msgs(storm3d, H1=30.0), "rule 2")
    # a world that is not 320 wide: the pools are L = 320 coordinates
    small = replace(storm3d, world=replace(storm3d.world, size=160.0))
    assert has(W.validate_config(small), "rule W-2")
    # 2D: no height rule, no z_sl rule
    assert W.validate_config(storm2d) == []
    assert W.validate_config(with_weather(storm2d, z_sl=99.0)) == []
    assert W.validate_config(with_weather(storm2d, H1=30.0, H2=50.0)) == []  # height = H1 + H2 in 2D


def test_the_3d_load_rules_at_the_confirmed_set(storm3d: Config) -> None:
    """R1-R3 from ``64k3d`` and ``PhysicsConfig``: 0.1 + 0.025 + 0.03 = 0.155 < 0.16 and 0.1 + 0.3 <= 0.5."""
    assert msgs(storm3d) == []
    assert msgs(storm3d, w_edge=0.0349) == []  # R2: w_edge < 0.035 at a = 1
    assert has(msgs(storm3d, w_edge=0.0351), "rule 16")
    assert msgs(storm3d, u_max=0.4) == []  # R3: (0.1 + 0.4) / 1 <= 0.5
    assert has(msgs(storm3d, u_max=0.4001), "<= v_max")
    assert has(msgs(storm3d, alpha1=0.7501), "alpha_max")  # R1: alpha0 + alpha1 <= 1 (1.0001 fails)
    assert (
        msgs(storm3d, alpha0=0.2, alpha1=0.7) == []
    )  # a = 0.9 leaves room: u_max 0.44 passes (0.1 + 0.396 <= 0.5)
    assert msgs(storm3d, alpha0=0.2, alpha1=0.7, u_max=0.44) == []


def test_the_2d_load_rules(storm2d: Config) -> None:
    """R4: ``thrust_max + alpha0·drag·u_max < 2r·k_repel`` strictly (0.1 + 0.25·0.24 is not below 0.16)."""
    default_u = with_weather(load_config(preset="tiny", overrides=["world.size=320"]), enabled=True)
    assert has(W.validate_config(default_u), "rule R4")  # u_max = 0.3: 0.175 > 0.16
    assert msgs(storm2d, u_max=0.2) == []
    assert msgs(storm2d, u_max=0.2399) == []
    assert has(msgs(storm2d, u_max=0.2401), "rule R4")
    assert has(msgs(storm2d, u_max=0.24), "rule R4")  # the boundary itself is not below
    assert msgs(storm2d, alpha0=0.0, alpha1=0.0, u_max=0.3) == []  # no wind coupling: nothing to load
    assert msgs(storm2d, alpha0=0.1, u_max=0.5) == []  # 0.1 + 0.05 < 0.16; alpha1 is inert in 2D
    assert has(msgs(storm2d, alpha0=0.5, alpha1=0.5, u_max=0.2), "rule R4")  # 0.1 + 0.1 = 0.2


# --- validation: the coupling keys ----------------------------------------------------------------


def test_coupling_key_ranges(storm3d: Config) -> None:
    assert has(msgs(storm3d, storm_p=1.5), "storm_p")
    assert has(msgs(storm3d, storm_p=-0.001), "storm_p")
    assert has(msgs(storm3d, storm_p=float("nan")), "storm_p")
    assert msgs(storm3d, storm_p=0.0) == [] and msgs(storm3d, storm_p=1.0) == []
    assert has(msgs(storm3d, storm_u0=-0.01), "storm_u0")
    assert msgs(storm3d, storm_u0=0.0) == []
    assert has(msgs(storm3d, storm_span=0.0), "storm_span")
    assert has(msgs(storm3d, storm_span=-1.0), "storm_span")
    assert has(msgs(storm3d, rain_floor=1.5), "rain_floor")
    assert has(msgs(storm3d, rain_floor=-0.1), "rain_floor")
    assert msgs(storm3d, rain_floor=0.5) == []
    assert has(msgs(storm3d, rain_gain=-1.0), "rain_gain")


def test_the_bond_break_threshold_rule(storm3d: Config) -> None:
    """``storm_break_u`` is 0 (off) or at least ``storm_u0``: the air speed includes the layer-2 jet."""
    assert has(msgs(storm3d, storm_break_u=0.03), "storm_break_u")  # below the default u0 = 0.06
    assert msgs(storm3d, storm_break_u=0.06) == []
    assert msgs(storm3d, storm_break_u=0.0) == []
    assert msgs(storm3d, storm_break_u=0.03, storm_u0=0.02) == []
    assert has(msgs(storm3d, storm_break_u=-0.1), "storm_break_u")
    assert msgs(storm3d, storm_break_u=0.03, storm_u0=0.02, storm_p=0.0) == []  # holds with the hazard off


def test_the_rain_gain_bound_is_the_int32_derivation(storm3d: Config) -> None:
    """``rain_gain·(H1 + H2 - 2·h_min)/gamma <= 2^30``: 3.05e8 at the confirmed set (not a policy cap)."""
    assert msgs(storm3d, rain_gain=1e6) == []
    assert msgs(storm3d, rain_gain=3.0e8) == []
    assert has(msgs(storm3d, rain_gain=3.1e8), "rain_gain")
    assert has(msgs(storm3d, rain_gain=1e9), "rain_gain")
    assert has(msgs(storm3d, rain_gain=float("inf")), "rain_gain")
    assert has(msgs(storm3d, rain_gain=float("nan")), "rain_gain")
    assert msgs(storm3d, rain_gain=1e12, gamma=0.0) == []  # gamma = 0: no moisture feedback, no rain bound


def test_rain_floor_is_zero_in_2d_and_with_the_director(storm3d: Config, storm2d: Config) -> None:
    assert has(msgs(storm2d, rain_floor=0.1), "rain_floor must be 0 in a 2D world")
    assert msgs(storm2d) == []
    # C17: the floor K and the seasonal table do not carry the rain, so a director run keeps rain_floor = 0
    director = replace(storm3d, director=replace(storm3d.director, enabled=True))
    assert W.validate_config(director) == []
    assert has(W.validate_config(with_weather(director, rain_floor=0.2)), "director.enabled")
    assert msgs(storm3d, rain_floor=0.2) == []  # without the director it is allowed
    with pytest.raises(ConfigError, match="rain_floor"):
        apply_overrides(storm3d, ["director.enabled=true", "weather.rain_floor=0.2"])
    apply_overrides(storm3d, ["director.enabled=true"])  # the overlay itself is director-safe
    # a director run with weather off has rain_floor at its default (W-0)
    tiny = load_config(preset="tiny", overrides=["director.enabled=true"])
    assert tiny.weather.rain_floor == 0.0


def test_validate_config_never_raises_on_broken_numbers(storm3d: Config) -> None:
    for kw in (
        {"gamma": float("nan")},
        {"H1": float("inf")},
        {"g_top": 0.0},
        {"tau_c": 0.0},
        {"seed_sigma": float("nan")},
        {"pools": ()},
        {"cells": 8},
        {"cells": 1024},
        {"f0": float("nan")},
    ):
        assert isinstance(msgs(storm3d, **kw), list)
    broken = replace(storm3d, world=replace(storm3d.world, field_cell=0.0))
    assert has(W.validate_config(broken), "field_cell")
