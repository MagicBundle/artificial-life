"""Farming in the step (DESIGN §41): determinism, compiled equals eager, resume equals the uninterrupted run,
observer purity and the FARMING flag, the staged bench pipeline, the bench flag, and coexistence with the
other features."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

import helpers
from alife import invariants
from alife.config import ConfigError, load_config
from alife.runio import load_checkpoint, write_state_file
from alife.state import Diag, counter_index, state_digest, to_numpy
from alife.step import ActFlag, make_step
from niche_helpers import cfg_for, farmers, np_soil

STEPS = 40


def _final(cfg, s, steps, *, observe=False, compile=None):  # None: the config's own (eager on a sphere)
    fn = make_step(cfg, observe=observe, compile=compile)
    outs = []
    for _ in range(steps):
        s, out = fn(s)
        mx.eval(s, out)
        outs.append(out)
    return s, outs


@pytest.mark.parametrize("nd", [2, 3, "s"])
def test_determinism_resume_and_observer_purity(nd: int | str, tmp_path: Path) -> None:
    cfg = cfg_for(nd)
    a, _ = _final(cfg, farmers(cfg), STEPS)
    assert state_digest(a) == state_digest(_final(cfg, farmers(cfg), STEPS)[0])
    half, _ = _final(cfg, farmers(cfg), STEPS // 2)
    path = tmp_path / "half.safetensors"
    write_state_file(path, half, cfg)
    resumed, _ = _final(cfg, load_checkpoint(path, cfg), STEPS // 2)
    assert state_digest(resumed) == state_digest(a)  # resume equals the uninterrupted run, soil included
    assert np_soil(a).sum() > 0 and np.array_equal(np_soil(a), np_soil(resumed))
    plain, po = _final(cfg, farmers(cfg), 12)
    obs, oo = _final(cfg, farmers(cfg), 12, observe=True)
    assert state_digest(plain) == state_digest(obs)  # the observer channel never feeds back
    for o1, o2 in zip(po, oo, strict=True):
        assert np.array_equal(to_numpy(o1.counters), to_numpy(o2.counters))
    # the FARMING flag is the per-row form of FARMERS
    assert all(to_numpy(o.acts.flags).dtype == np.uint16 for o in oo)
    flagged = [int(((to_numpy(o.acts.flags) & ActFlag.FARMING.value) > 0).sum()) for o in oo]
    counted = [int(to_numpy(o.counters)[counter_index(cfg, "FARMERS")]) for o in oo]
    assert flagged == counted and sum(flagged) > 0


@pytest.mark.parametrize("nd", [2, 3])
def test_compiled_and_eager_steps_agree(nd: int) -> None:
    """One step agrees on every counter, the ledger and every integer field; floats to 1e-5 (bitwise equality
    is not promised, §14.2)."""
    cfg = cfg_for(nd)
    s = farmers(cfg)
    a, oa = make_step(cfg, compile=False)(s)
    b, ob = make_step(cfg, compile=True)(s)
    mx.eval(a, oa, b, ob)
    assert np.array_equal(to_numpy(oa.counters), to_numpy(ob.counters))
    assert np.array_equal(to_numpy(oa.ledger), to_numpy(ob.ledger))
    for name in type(s)._fields:
        x, y = to_numpy(getattr(a, name)), to_numpy(getattr(b, name))
        if x.dtype.kind == "f":
            np.testing.assert_allclose(x, y, rtol=1e-5, atol=1e-5, err_msg=name)
        else:
            assert np.array_equal(x, y), name
    assert to_numpy(oa.counters)[counter_index(cfg, "FARMERS")] > 0


@pytest.mark.parametrize("nd", [2, 3, "s"])
@pytest.mark.parametrize("control", ["none", "ablate"])
def test_the_staged_pipeline_equals_the_step_with_farming_on(nd: int | str, control: str) -> None:
    from mlx.utils import tree_flatten

    from alife import bench

    cfg = cfg_for(nd, f"niche.control={control}")
    cfg = replace(cfg, run=replace(cfg.run, compile=False))
    s0 = farmers(cfg)
    fused, staged = make_step(cfg), bench._StagedStep(cfg)
    s, holder = s0, [s0]
    for i in range(5):
        s, out = fused(s)
        out_staged = staged(holder)
        mx.eval(s, out, holder[0], out_staged)
        assert state_digest(holder[0]) == state_digest(s), i
        for a, b in zip(tree_flatten(out), tree_flatten(out_staged), strict=True):
            assert a[0] == b[0]
            np.testing.assert_array_equal(to_numpy(a[1]), to_numpy(b[1]), err_msg=a[0])
    assert np_soil(s).any()


def test_bench_niche_flag_needs_the_feature() -> None:
    from alife import bench

    with pytest.raises(ConfigError, match="bench --niche"):
        bench.run_bench(load_config(preset="tiny"), steps=1, warmup=1, niche_arm=True)


@pytest.mark.parametrize("nd", [2, 3, "s"])
def test_farming_coexists_with_every_other_feature(nd: int | str) -> None:
    """Structures, bonds, comm and arms on together with farming (DESIGN §41.7): 150 steps, the ledger closes,
    the soil and material identities hold, nothing trips."""
    extra = [
        "structures.enabled=true",
        "bonds.share_mode=2",
        "roles.enabled=true",
        "comm.enabled=true",
        "arms.enabled=true",
    ]
    cfg = cfg_for(nd, *extra)
    s = farmers(cfg)
    fn = make_step(cfg)
    tot = int(np_soil(s).sum())
    for t in range(60 if nd == "s" else 150):
        prev = s
        s, out = fn(s)
        mx.eval(s, out)
        led, c = to_numpy(out.ledger), to_numpy(out.counters)
        assert helpers.ledger_net(led, cfg) == int(to_numpy(s.total) - to_numpy(prev.total)), t
        put, dec = (int(c[counter_index(cfg, n)]) for n in ("SOIL_PUT", "SOIL_DECAYED"))
        tot += put - dec
        assert tot == int(np_soil(s).sum()), t
        assert int(to_numpy(s.diag)) & 0xFFFF == 0, (t, Diag(int(to_numpy(s.diag))))
    assert invariants.full_check(s, cfg, prev, out) == []
    assert tot > 0
