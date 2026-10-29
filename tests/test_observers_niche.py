"""Farming, the observer PR F2 (DESIGN §41.14): ``Acts.ext.niche`` and the FARMING flag, the soil dump, the
``farm_`` snapshot keys and the ``FSOL`` record of the play loop."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from alife import runner
from alife import streamrec as sr
from alife.config import ConfigError, apply_overrides, config_hash, config_to_dict, dims
from alife.metrics import snapshot_metrics
from alife.runio import RunDir
from alife.state import counter_index, init_state, to_numpy
from alife.step import ActFlag, make_step
from niche_helpers import cfg_for, farmers, np_soil
from play_helpers import decoded, run_paced, snapshots, world


@pytest.mark.parametrize("nd", [2, 3])
def test_the_channel_carries_the_units_put_and_the_farming_flag(nd: int) -> None:
    cfg = cfg_for(nd, "niche.silent_init=false")
    s = farmers(cfg, bias=4.0, zero_weights=True)
    s1, out = make_step(cfg, observe=True)(s)
    mx.eval(s1, *out)
    put = to_numpy(out.acts.ext.niche.farm_put)
    alive = to_numpy(s.alive)
    assert put.dtype == np.int16 and put.shape == alive.shape
    assert (put[alive] > 0).all() and not put[~alive].any() and put.max() <= cfg.niche.farm_rate
    flags = to_numpy(out.acts.flags).astype(np.int64)
    farming = (flags & ActFlag.FARMING.value) != 0
    assert np.array_equal(farming, put > 0)
    c = to_numpy(out.counters)
    assert int(farming.sum()) == int(c[counter_index(cfg, "FARMERS")])
    assert int(put.astype(np.int64).sum()) == int(
        c[counter_index(cfg, "SOIL_PUT")] + c[counter_index(cfg, "SOIL_WASTED")]
    )


def test_a_silent_start_has_a_channel_with_no_farmer() -> None:
    cfg = cfg_for(2)  # silent_init: nobody farms at first
    s1, out = make_step(cfg, observe=True)(init_state(cfg))
    mx.eval(s1, *out)
    assert not to_numpy(out.acts.ext.niche.farm_put).any()
    assert not (to_numpy(out.acts.flags).astype(np.int64) & ActFlag.FARMING.value).any()


@pytest.mark.parametrize("nd", [2, 3])
def test_the_snapshot_keys_follow_the_soil(nd: int) -> None:
    cfg = cfg_for(nd, "niche.silent_init=false")
    s = farmers(cfg, bias=4.0, zero_weights=True)
    for _ in range(20):
        s, out = make_step(cfg)(s)
        mx.eval(s, out)
    snap = snapshot_metrics(s, cfg, heavy=False)
    soil = np_soil(s)
    assert snap["farm_cells"] == int((soil > 0).sum()) > 0
    assert snap["farm_soil_max"] == int(soil.max())
    assert snap["farm_soil_mean"] == pytest.approx(soil.sum() / soil.size)
    off = cfg_for(nd, "niche.farm=false")
    assert not [k for k in snapshot_metrics(init_state(off), off, heavy=False) if k.startswith("farm_")]


def test_the_soil_dump_is_the_grid_and_its_cadence_is_an_unhashed_resume_key(tmp_path) -> None:
    from alife.cli import RESUME_KEYS, _resume_allowed

    cfg = apply_overrides(
        cfg_for(2, "niche.silent_init=false"),
        ["run.niche_every=10", "run.metrics_every=10", "run.checkpoint_every=30"],
    )
    rd = RunDir.create(tmp_path / "run", cfg)
    res = runner.run(rd, cfg, farmers(cfg, bias=4.0), 30, quiet=True)
    assert res.status == "done"
    files = rd.dump_files("niche")
    assert [st for st, _ in files] == [10, 20, 30] and files[0][1].name == "niche_0000000010.npy"
    g = rd.read_grid("niche", 30)
    assert g.dtype == np.int32 and g.shape == dims(cfg).field_shape and np.array_equal(g, np_soil(res.state))
    assert g.sum() > 0
    rd.truncate_after(20)
    assert [st for st, _ in rd.dump_files("niche")] == [10, 20]
    assert "niche_every" in RESUME_KEYS and _resume_allowed("run.niche_every=50")
    base = cfg_for(2)
    assert "niche_every" not in config_to_dict(base, omit_defaults=True)["run"]
    assert config_hash(base) == config_hash(apply_overrides(base, ["run.niche_every=5"]))
    with pytest.raises(ConfigError, match="niche_every"):
        apply_overrides(cfg_for(2, "niche.farm=false"), ["run.niche_every=5"])


def test_fsol_follows_the_soil_in_the_play_loop(tmp_path, monkeypatch) -> None:
    snapshots(monkeypatch, 25)
    cfg = world("3d", "niche.farm=true", "niche.silent_init=false", "niche.farm_cost=0")
    _, res, _, out = run_paced(cfg, tmp_path / "run", 50, [])
    dec = decoded(out)
    d = dims(cfg)
    h = dec.header["play"]
    assert "FSOL" in h["records"] and "STRC" not in h["records"]
    assert h["niche"] == {
        "enabled": True,
        "farm": True,
        "cap": cfg.niche.farm_cap,
        "cells": [d.fz, d.fg, d.fg],
        "cell": cfg.world.field_cell,
    }
    fsol = [r for r in dec.records if r.tag == "FSOL"]
    assert [r.step for r in fsol] == [0, 25, 50]
    idx, lev, _ = sr.quantise_cells(np_soil(res.state), cfg.niche.farm_cap)
    assert np.array_equal(fsol[-1].a["idx"], idx) and np.array_equal(fsol[-1].a["level"], lev)
