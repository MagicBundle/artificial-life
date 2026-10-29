"""Farming, F1 (DESIGN §41): the layout and config of the soil layer, the off world, validation, the hash,
the counters and metrics (E10), the blocked-births alarm (E20) and its stress test."""

from __future__ import annotations

import itertools
import logging

import mlx.core as mx
import numpy as np
import pytest

from alife import config, ext, layout, metrics
from alife.config import ConfigError, config_hash, dims, load_config
from alife.state import (
    N_COUNTERS,
    CounterIdx,
    State,
    State3,
    counter_blocks,
    counter_index,
    init_state,
    n_counters,
    state_cls,
    to_numpy,
)
from alife.step import ActFlag, make_step
from niche_helpers import PRESET, cfg_for

F32 = "brain.weight_dtype=float32"


# --- layout, config, registry -----------------------------------------------------------------------------


@pytest.mark.parametrize("nd", [2, 3])
def test_the_footprint_is_one_input_and_one_output(nd: int) -> None:
    base = dims(load_config(preset=PRESET[nd])).lay
    lay = dims(cfg_for(nd)).lay
    assert (base.D_IN + 1, base.D_OUT + 1) == (lay.D_IN, lay.D_OUT)
    assert ((70, 36) if nd == 2 else (85, 40)) == (lay.D_IN, lay.D_OUT)
    assert (lay.IN_FARM, lay.OUT_FARM_PUT) == (base.D_IN, base.D_OUT)
    assert base.IN_FARM is None and base.OUT_FARM_PUT is None  # None when off, never -1
    spans = sorted(b[1:] for b in lay.INPUT_BLOCKS)
    assert spans[0][0] == 0 and all(a[0] + a[1] == b[0] for a, b in itertools.pairwise(spans))
    assert len(set(lay.INPUT_NAMES)) == lay.D_IN and len(set(lay.OUTPUT_NAMES)) == lay.D_OUT
    s = init_state(cfg_for(nd))
    assert s.soil.shape == dims(cfg_for(nd)).field_shape and s.W2.shape[2] == lay.D_OUT
    assert ext.layout_features(cfg_for(nd)) == "niche.farm@1"
    assert ActFlag.FARMING.value == 2048 == 128 << ext.EXTENSIONS.index("niche")


def test_silent_init_zeroes_the_put_column_and_can_be_switched_off() -> None:
    cfg = cfg_for(2)
    d = dims(cfg)
    s = init_state(cfg)
    alive = to_numpy(s.alive)
    assert not to_numpy(s.W2)[alive][:, :, d.lay.OUT_FARM_PUT].any()  # silent: nobody farms at first
    assert to_numpy(s.W1)[alive][:, d.lay.IN_FARM].any()  # the sensor row is random (as any other)
    loud = cfg_for(2, "niche.silent_init=false")
    assert to_numpy(init_state(loud).W2)[to_numpy(init_state(loud).alive)][:, :, d.lay.OUT_FARM_PUT].any()


@pytest.mark.parametrize("preset", ["tiny", "tiny3d"])
def test_an_off_world_has_no_niche_footprint(preset: str) -> None:
    cfg = load_config(preset=preset, overrides=[F32])
    d = dims(cfg)
    assert "niche" not in [sp.name for sp in ext.active(cfg)] and ext.layout_features(cfg) == ""
    assert (d.lay.IN_FARM, d.lay.OUT_FARM_PUT) == (None, None)
    assert state_cls(cfg) is (State if preset == "tiny" else State3)  # the exact class: no `soil`
    s = init_state(cfg)
    assert "soil" not in type(s)._fields
    s1, out = make_step(cfg)(s)
    mx.eval(s1, out)
    assert out.counters.shape == (n_counters(d.nd, cfg.weather.enabled),) and counter_blocks(cfg) == ()
    assert len(CounterIdx) == N_COUNTERS


def test_off_is_identical() -> None:
    """With ``niche.farm`` false nothing is allocated, hashed or serialised."""
    for preset in config.PRESETS:
        cfg = load_config(preset=preset)
        assert "niche" not in config.config_to_dict(cfg, omit_defaults=True)
    cfg = load_config(preset="tiny", overrides=[F32])
    assert config_hash(cfg) != config_hash(cfg_for(2))
    assert config_hash(load_config(preset="tiny")) == config_hash(load_config(preset="tiny", overrides=[]))


@pytest.mark.parametrize(
    ("over", "msg"),
    [
        (["niche.farm_rate=9"], "inert while niche.farm is false"),
        (["niche.farm_cost=31"], "inert while niche.farm is false"),
        (["niche.control=mute"], "inert while niche.farm is false"),
        (["niche.silent_init=false"], "inert while niche.farm is false"),
        (["niche.farm=true", "niche.farm_rate=0"], "farm_rate"),
        (["niche.farm=true", "niche.farm_rate=257"], "farm_rate"),
        (["niche.farm=true", "niche.farm_cap=0"], "farm_cap"),
        (["niche.farm=true", "niche.farm_cap=32768"], "farm_cap"),
        (["niche.farm=true", "niche.farm_cost=-1"], "farm_cost"),
        (["niche.farm=true", "niche.farm_cost=40000"], "farm_cost"),
        (["niche.farm=true", "niche.farm_cost=4200", "niche.farm_rate=256"], "farm_cost"),
        (["niche.farm=true", "niche.farm_decay_q16=0"], "farm_decay_q16"),
        (["niche.farm=true", "niche.farm_decay_q16=65536"], "farm_decay_q16"),
        (["niche.farm=true", "niche.farm_cap=32767", "niche.farm_decay_q16=40000"], "dithered loss"),
        (["niche.farm=true", "niche.farm_bonus_frac=1.5"], "farm_bonus_frac"),
        (["niche.farm=true", "niche.farm_bonus_frac=-0.1"], "farm_bonus_frac"),
        (["niche.farm=true", "niche.control=loud"], "niche.control"),
        (["niche.farm=true", "niche.farm_cap=32767", "world.size=2048"], "SOIL_TOTAL"),
    ],
)
def test_validation(over: list[str], msg: str) -> None:
    with pytest.raises(ConfigError, match=msg):
        load_config(preset="tiny", overrides=over)


def test_a_cheap_farm_in_a_rich_world_warns(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="alife.ext.niche"):
        load_config(preset="tiny", overrides=["niche.farm=true", "fields.regen_max=1700"])
    assert any("farm_cost" in r.message and "low" in r.message for r in caplog.records)
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="alife.ext.niche"):
        load_config(
            preset="tiny", overrides=["niche.farm=true", "fields.regen_max=1700", "niche.farm_cost=150"]
        )
        load_config(preset="tiny", overrides=["niche.farm=true"])
    assert not caplog.records


def test_every_key_changes_the_hash_and_the_overlay_composes() -> None:
    base = cfg_for(2)
    for k, v in (
        ("farm_rate", 4),
        ("farm_cap", 2000),
        ("farm_cost", 60),
        ("farm_decay_q16", 32),
        ("farm_bonus_frac", 0.25),
        ("silent_init", "false"),
        ("control", "ablate"),
    ):
        other = load_config(preset="tiny", overrides=[F32, "niche.farm=true", f"niche.{k}={v}"])
        assert config_hash(other) != config_hash(base), k
    keys = set(config.FEATURE_KEYS["niche"])
    assert keys == {
        ("niche", f)
        for f in (
            "farm",
            "farm_rate",
            "farm_cap",
            "farm_cost",
            "farm_decay_q16",
            "farm_bonus_frac",
            "silent_init",
            "control",
        )
    }
    assert keys <= set(config.POST_V01_KEYS) and keys <= set(config.SERIAL_OMIT_KEYS)
    assert set(config.CLI_ONLY_FEATURE_KEYS["niche"]) == {("niche", "control"), ("niche", "silent_init")}
    ov = load_config(preset="tiny3d", overlays=["niche"], overrides=["niche.farm_cost=150"])
    assert ov.niche.farm and "niche" in config.WORLD_OVERLAYS and ext.active(ov)[0].name == "niche"
    assert load_config(preset="tiny", overlays=["niche"]).niche.farm_cost == 30


def test_the_enable_hook_switches_farming_on_and_nothing_else() -> None:
    spec = next(sp for sp in ext.REGISTRY if sp.name == "niche")
    cfg = load_config(preset="tiny")
    on = spec.enable(cfg)
    assert on.niche.farm and on.niche.farm_cost == 30 and on.niche.silent_init and on.niche.control == "none"
    assert spec.enable(on) == on


# --- E10: counters and metrics ----------------------------------------------------------------------------


def test_e10_counters_and_metrics() -> None:
    cfg = cfg_for(2)
    names = ("FARMERS", "FARM_YIELD", "SOIL_PUT", "SOIL_WASTED", "SOIL_DECAYED", "SOIL_TOTAL", "SLOTS_FULL")
    assert counter_blocks(cfg) == (("niche", names),)
    assert n_counters(2, cfg.weather.enabled, counter_blocks(cfg)) == n_counters(2, cfg.weather.enabled) + 7
    assert len({counter_index(cfg, n) for n in names}) == 7
    from niche_helpers import farmers, run

    cs, ls = [], []
    for _, _, out, _ in run(cfg, farmers(cfg), 40):
        cs.append(to_numpy(out.counters))
        ls.append(to_numpy(out.ledger))
    row = metrics.interval_row(40, np.stack(cs), np.stack(ls), {}, {}, nd=2, blocks=counter_blocks(cfg))
    assert 0.0 < row["farm_farmer_frac"] <= 1.0 and row["farm_put"] > 0 and row["farm_soil_total"] > 0
    assert row["farm_yield"] > 0 and 0.0 < row["farm_yield_share"] < 1.0
    assert (
        row["farm_blocked_frac"] is not None
        and row["farm_full_frac"] == 0.0
        and not row["farm_blocked_alarm"]
    )
    # the per-step counters are int32; the interval sums are int64 on the host
    assert np.stack(cs).dtype == np.int32 and isinstance(row["farm_put"], int)
    # the validation bound keeps SOIL_TOTAL inside int32 at the largest lattice
    assert cfg.niche.farm_cap * dims(cfg).n_cells < 2**31


# --- E20: immortal colonies fill every slot -----------------------------------------------------------------


def test_e20_the_alarm_needs_full_rows_and_turned_away_births() -> None:
    """``farm_full_frac`` is the share of steps that ended with the rows full (``SLOTS_FULL``); the alarm is
    ``full_frac > niche.BLOCKED_ALARM_FRAC`` with some birth blocked. Synthetic counters pin the
    thresholds."""
    from alife import niche

    cfg = cfg_for(2)
    blocks = counter_blocks(cfg)
    w = n_counters(2, cfg.weather.enabled, blocks)
    at = n_counters(2, cfg.weather.enabled)
    full_col = at + dict(blocks)["niche"].index("SLOTS_FULL")
    ls = np.zeros((10, 16), np.int64)
    for n_full, blocked, alarm in ((0, 5, False), (5, 5, False), (6, 5, True), (6, 0, False), (10, 1, True)):
        c = np.zeros((10, w), np.int64)
        c[:, CounterIdx.N_ALIVE] = 100
        c[:, CounterIdx.BIRTH_REQUESTS] = 3
        c[0, CounterIdx.BIRTHS_BLOCKED] = blocked
        c[:n_full, full_col] = 1
        row = metrics.interval_row(10, c, ls, {}, {}, nd=2, blocks=blocks)
        assert row["farm_blocked_alarm"] is alarm, (n_full, blocked)
        assert row["farm_full_frac"] == n_full / 10
    assert niche.BLOCKED_ALARM_FRAC == 0.5


def test_e20_stress_a_packed_farming_world_raises_the_alarm() -> None:
    """A small world whose organisms all ask to breed fills every slot within a few steps and stays full:
    births are turned away, ``SLOTS_FULL`` counts the full steps and the alarm fires; the ledger still closes
    and no invariant trips (nothing wraps or overflows with the colony pinned at ``n_max``)."""
    import helpers
    from alife import invariants

    cfg = load_config(
        preset="tiny",
        overrides=[
            F32,
            "niche.farm=true",
            "population.n_max=64",
            "population.n_init=56",
            "population.max_births_per_step=8",
            "world.size=48",
            "repro.min_energy=30000",
            "repro.min_age=0",
            "population.max_age=0",
        ],
    )
    s = init_state(cfg)
    s = helpers.set_output_bias(s, cfg, {layout.OUT_REPRODUCE: 4.0})
    cs, ls = [], []
    fn = make_step(cfg)
    for t in range(40):
        prev = s
        s, out = fn(s)
        mx.eval(s, out)
        cs.append(to_numpy(out.counters))
        ls.append(to_numpy(out.ledger))
        assert helpers.ledger_net(ls[-1], cfg) == int(to_numpy(s.total) - to_numpy(prev.total)), t
    row = metrics.interval_row(40, np.stack(cs), np.stack(ls), {}, {}, nd=2, blocks=counter_blocks(cfg))
    assert row["births_blocked"] > 0 and row["farm_full_frac"] > 0.5, row
    assert row["farm_blocked_alarm"] is True
    assert int(to_numpy(s.diag)) & 0xFFFF == 0 and invariants.full_check(s, cfg) == []


# --- review fixes: the 3D overlay and the E10 bounds ------------------------------------------------------


def test_the_niche3d_overlay_is_the_3d_design_point_without_a_set(caplog: pytest.LogCaptureFixture) -> None:
    assert "niche3d" in config.WORLD_OVERLAYS
    with caplog.at_level(logging.WARNING, logger="alife.ext.niche"):
        c3 = load_config(preset="mini3d", overlays=["niche3d"])
    assert c3.niche.farm and c3.niche.farm_cost == 150 and c3.fields.regen_max == 1700
    assert not caplog.records  # at the design point: no "nearly free" warning
    assert load_config(preset="mini3d", overlays=["niche"]).niche.farm_cost == 30  # the 2D value, still warns
    with caplog.at_level(logging.WARNING, logger="alife.ext.niche"):
        load_config(preset="mini3d", overlays=["niche3d"], overrides=["niche.farm_cost=10"])
    assert any("farm_cost" in r.message for r in caplog.records)  # an explicit cheap cost still warns


@pytest.mark.parametrize(
    ("over", "msg"),
    [
        (["niche.farm_cost=0", "niche.farm_rate=256", "population.n_max=16777216"], "SOIL_PUT"),
        (
            ["niche.farm_bonus_frac=1.0", "fields.regen_max=8000", "world.size=4096", "world.field_cell=1"],
            "FARM_YIELD",
        ),
    ],
)
def test_e10_the_counters_cannot_overflow_even_at_zero_cost(over: list[str], msg: str) -> None:
    with pytest.raises(ConfigError, match=msg):
        load_config(preset="tiny", overrides=["niche.farm=true", *over])


def test_the_bench_sums_counters_in_int64() -> None:
    import inspect

    from alife import bench

    assert "astype(mx.int64)" in inspect.getsource(bench)
