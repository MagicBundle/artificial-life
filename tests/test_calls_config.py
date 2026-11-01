"""The ``[calls]`` section (DESIGN §47.6): defaults off and inert, hashing, validation, the overlays, the
registry row. Step K1 builds the trigger; the far hearing and the controls are declared and refused until
K2."""

from __future__ import annotations

import dataclasses

import pytest

from alife import config, ext
from alife.config import CallsConfig, ConfigError, config_hash, config_to_dict, dims, load_config
from alife.ext.calls import CONTROLS, EVENTS, SPEC, event_names

BASE = ["comm.enabled=true", "comm.width=2", "calls.enabled=true", "calls.trigger=true", "calls.events=3"]
WX = ["weather.enabled=true", "world.size=320", "weather.u_max=0.2"]  # tiny with the weather (rule R4)


def load(*over: str, preset: str = "tiny"):
    return load_config(preset=preset, overrides=list(over))


def test_every_key_is_off_by_default_hashed_and_omitted_from_config_json() -> None:
    d = CallsConfig()
    assert (d.enabled, d.trigger, d.far_range, d.control) == (False, False, 0.0, "none")
    assert (d.events, d.hold, d.food_q, d.storm_frac, d.far_xref) == (15, 4, 200, 1.0, 1.0)
    keys = {("calls", f.name) for f in dataclasses.fields(CallsConfig)}
    assert keys == set(ext.lever_keys()["calls"]) == set(SPEC.lever_keys) and len(keys) == 9
    assert keys <= set(config.POST_V01_KEYS) and keys <= set(config.SERIAL_OMIT_KEYS)
    cfg = load("brain.weight_dtype=float32")
    assert "calls" not in config_to_dict(cfg, omit_defaults=True)
    on = load("brain.weight_dtype=float32", *BASE)
    assert config_to_dict(on, omit_defaults=True)["calls"]["enabled"] is True
    assert config_hash(on) != config_hash(load("brain.weight_dtype=float32", *BASE, "calls.hold=5"))
    assert config_hash(cfg) != config_hash(on)
    # every key moves the hash while the feature is on (none is unhashed or an observer key)
    assert not keys & (set(config.UNHASHED_KEYS) | set(config.OBSERVER_KEYS))


def test_the_registry_row() -> None:
    assert ext.check_registry() == []
    assert SPEC.name == "calls" and SPEC.index == 5 and SPEC.tail_rank == 3 and SPEC.metric_prefix == "calls_"
    assert (
        SPEC.overlay == "calls"
        and "calls" in config.WORLD_OVERLAYS
        and "stormsph22h" in config.WORLD_OVERLAYS
    )
    assert set(config.CLI_ONLY_FEATURE_KEYS["calls"]) == set(SPEC.lever_keys)  # no GUI levers yet (K4)
    assert [n for n, _ in EVENTS] == ["hurt", "food", "storm", "lost"] and [b for _, b in EVENTS] == [
        1,
        2,
        4,
        8,
    ]
    assert event_names(5) == ("hurt", "storm") and event_names(15) == ("hurt", "food", "storm", "lost")
    assert CONTROLS == ("none", "nofar", "reflex", "reflex_near")
    assert "calls" in config.SPHERE_READY and config.SPHERE_FEATURE_PR["calls"] == "K1"


@pytest.mark.parametrize(
    ("over", "msg"),
    [
        (["calls.trigger=true"], "inert while calls.enabled is false"),
        (["calls.events=3"], "inert while calls.enabled is false"),
        (["calls.hold=9"], "inert while calls.enabled is false"),
        (["calls.control=nofar"], "inert while calls.enabled is false"),
        (["calls.enabled=true"], "needs comm.enabled"),
        (["calls.enabled=true", "calls.events=3"], "inert while calls.trigger is false"),
        (["calls.enabled=true", "calls.hold=9"], "inert while calls.trigger is false"),
        (["calls.enabled=true", "calls.food_q=9"], "inert while calls.trigger is false"),
        (["calls.enabled=true", "calls.storm_frac=0.5"], "inert while calls.trigger is false"),
        (
            ["comm.enabled=true", "calls.enabled=true", "calls.trigger=true", "calls.events=3"],
            "comm.width == the number of events 2",
        ),  # comm.width defaults to 4
        (
            [
                "comm.enabled=true",
                "comm.width=2",
                "calls.enabled=true",
                "calls.trigger=true",
                "calls.events=11",
            ],
            "comm.width == the number of events 3",
        ),
        ([*BASE, "calls.events=0"], r"calls.events 0 must be an integer in \[1, 15\]"),
        ([*BASE, "calls.events=16"], r"calls.events 16 must be an integer in \[1, 15\]"),
        ([*BASE, "calls.events=5"], "storm bit \\(4\\) needs weather.enabled"),
        ([*BASE, "calls.hold=-1"], r"calls.hold -1 must be in \[0, 64\]"),
        ([*BASE, "calls.hold=65"], r"calls.hold 65 must be in \[0, 64\]"),
        ([*BASE, "calls.food_q=0"], "calls.food_q 0 must be in"),
        ([*BASE, "calls.food_q=99999999"], "calls.food_q 99999999 must be in"),
        (
            [
                "comm.enabled=true",
                "comm.width=2",
                "calls.enabled=true",
                "calls.trigger=true",
                "calls.events=9",
                "calls.food_q=100",
            ],
            "calls.food_q is inert without the food bit",
        ),
        ([*BASE, "calls.storm_frac=0.5"], "storm_frac is inert without the storm bit"),
        ([*BASE, "calls.far_range=18"], "built in step K2"),
        ([*BASE, "calls.far_xref=2"], "calls.far_xref is inert without calls.far_range"),
        ([*BASE, "calls.control=nofar"], "needs calls.far_range > 0"),
        ([*BASE, "calls.control=reflex_near"], "needs calls.trigger and the storm bit"),
        ([*BASE, "calls.control=loud"], "must be one of"),
    ],
)
def test_validation(over: list[str], msg: str) -> None:
    with pytest.raises(ConfigError, match=msg):
        load(*over)
    if over[:1] == ["calls.enabled=true"] or "calls.events=9" in over:
        return
    assert load(*BASE, "calls.food_q=100").calls.food_q == 100  # events 3 has the food bit: allowed there


def test_a_storm_event_needs_the_weather_and_a_hazard_onset() -> None:
    cfg = load(
        *WX, "comm.enabled=true", "comm.width=2", "calls.enabled=true", "calls.trigger=true", "calls.events=5"
    )
    assert event_names(cfg.calls.events) == ("hurt", "storm") and cfg.weather.storm_u0 > 0
    with pytest.raises(ConfigError, match="storm_u0 > 0"):
        load(*WX, "weather.storm_u0=0", "comm.enabled=true", "comm.width=2", "calls.enabled=true",
             "calls.trigger=true", "calls.events=5")  # fmt: skip
    with pytest.raises(ConfigError, match="storm_frac"):
        load(*WX, "comm.enabled=true", "comm.width=2", "calls.enabled=true", "calls.trigger=true",
             "calls.events=5", "calls.storm_frac=2.5")  # fmt: skip
    ok = load(*WX, "comm.enabled=true", "comm.width=2", "calls.enabled=true", "calls.trigger=true",
              "calls.events=5", "calls.storm_frac=0.75", "calls.hold=64")  # fmt: skip
    assert ok.calls.storm_frac == 0.75 and ok.calls.hold == 64


def test_enabled_alone_is_valid_and_inert() -> None:
    """The first form of the migration: comm and calls on, no trigger, no far hearing. No state, no counters,
    no brain footprint."""
    cfg = load("comm.enabled=true", "calls.enabled=true")
    assert ext.layout_features(cfg) == "comm4@1" and ext.state_fields(cfg, 2) == ()
    assert dims(cfg).features == ("comm", "calls")


def test_the_overlays_load_on_the_experiment_world() -> None:
    """`--preset mini3s --overlay stormsph22h --overlay comm --overlay calls`: width 2, cost 32, events 5, the
    weather of stormsph22 with only `storm_u0` different; the hazard-off control is a `--set`."""
    cfg = load_config(preset="mini3s", overlays=["stormsph22h", "comm", "calls"], overrides=[])
    assert (cfg.comm.enabled, cfg.comm.width, cfg.comm.cost) == (True, 2, 32.0)
    assert (cfg.calls.enabled, cfg.calls.trigger, cfg.calls.events, cfg.calls.hold) == (True, True, 5, 4)
    assert cfg.calls.far_range == 0.0 and cfg.calls.control == "none"
    base = load_config(preset="mini3s", overlays=["stormsph22"], overrides=[])
    wa, wb = (dataclasses.asdict(c.weather) for c in (cfg, base))
    assert {k for k in wa if wa[k] != wb[k]} == {"storm_u0"}  # the weather, hence the S4c statistics, as is
    assert wb["storm_u0"] == 0.095 and 0.0 < wa["storm_u0"] < wb["storm_u0"]
    off = load_config(
        preset="mini3s", overlays=["stormsph22h", "comm", "calls"], overrides=["weather.storm_p=0"]
    )
    assert off.weather.storm_p == 0.0
    assert "calls" in config.sphere_wanted(cfg) and dims(cfg).sph
    # the calls overlay brings its own comm: either order composes, and a dry world is refused (the storm bit)
    assert load_config(preset="mini3s", overlays=["calls", "stormsph22h"], overrides=[]).comm.width == 2
    with pytest.raises(ConfigError, match=r"needs weather\.enabled"):
        load_config(preset="tiny", overlays=["calls"], overrides=[])
    with pytest.raises(ConfigError):  # stormsph22h is for the mini3s planet only
        load_config(preset="tiny3s", overlays=["stormsph22h"], overrides=[])


def test_the_far_hearing_and_the_controls_are_refused_until_k2() -> None:
    for over in (["calls.far_range=18"], ["calls.control=reflex_near", "calls.events=5"]):
        extra = [*WX] if "calls.events=5" in over else []
        base = [*extra, "comm.enabled=true", "comm.width=2", "calls.enabled=true", "calls.trigger=true",
                *(["calls.events=3"] if not extra else [])]  # fmt: skip
        with pytest.raises(ConfigError, match="step K2"):
            load(*base, *over)
