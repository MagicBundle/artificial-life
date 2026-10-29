"""Enabling arms on an evolved world with ``alife migrate --enable arms`` (DESIGN §43.8, §42)."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from alife import arms, cli, ext
from alife.config import load_config
from alife.runio import RunDir, load_checkpoint
from alife.state import CounterIdx, Diag, LedgerIdx, init_state, legacy_digest, to_numpy
from alife.step import make_step


@pytest.mark.parametrize("nd", [2, 3])
def test_migrating_an_evolved_world_onto_arms(nd: int, tmp_path: Path, capsys) -> None:
    cfg = load_config(preset="tiny" if nd == 2 else "tiny3d", overrides=["brain.weight_dtype=float32"])
    fn, s = make_step(cfg), init_state(cfg)
    for _ in range(40):  # an evolved world: births, deaths, attacks have happened
        s, _ = fn(s)
    mx.eval(s)
    rd = RunDir.create(tmp_path / "src", cfg)
    rd.save_checkpoint(s, cfg)
    rd.wait_saves()
    out = tmp_path / "out"
    code = cli.main(
        ["migrate", str(rd.path), "--enable", "arms", "--out", str(out), "--quiet",
         "--set", "arms.init_atk=0.2", "--set", "arms.init_arm=0.3", "--set", "arms.key_floor=0.2",
         "--set", "arms.sigma=0.03", "--set", "arms.sigma_phase=0.05", "--set", "arms.upkeep_arm=20"]
    )  # fmt: skip
    assert code == 0, capsys.readouterr().err
    run = RunDir.open(out)
    cfg2, meta = run.config(), run.meta()
    assert cfg2.arms.enabled and ext.layout_features(cfg2) == "" and "layout_features" not in meta
    assert int(meta["rng_layout_version"]) == 4 and meta["migrated_from"]["added"] == ["arms"]
    (_, ck) = run.checkpoints()[-1]
    s2 = load_checkpoint(ck, cfg2)
    x = np.array(s2.arms)
    # the founders' policy: init values on every row, keys and locks from the init stream
    assert np.array_equal(x, np.array(arms.rest_array(cfg2)))
    assert (x[:, arms.ATK] == np.float32(0.2)).all() and len(np.unique(x[:, arms.KEY])) > 8
    assert legacy_digest(s2) == legacy_digest(s)  # the evolved world itself is untouched
    # it steps with arms on and the ledger closes
    fn2, bits, attacks = make_step(cfg2), 0, 0
    for _ in range(60):
        s2, out2 = fn2(s2)
        mx.eval(s2, out2)
        bits |= int(to_numpy(s2.diag))
        led = to_numpy(out2.ledger)
        assert led[LedgerIdx.FLOW_STOLEN] == led[LedgerIdx.FLOW_ATTACK_GAIN] + led[LedgerIdx.FLOW_SPILL]
        attacks += int(to_numpy(out2.counters)[CounterIdx.ATTACKS])
    assert bits & int(Diag.LEDGER) == 0 and bits == 0 and attacks > 0
