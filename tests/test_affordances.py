"""Every affordance of §0.1 is exercised by the whole step (§22.5). Owned by WP4.

``tiny`` runs for 400 compiled steps with the pinned seed. Founders get a fixed output-bias row from a
test-only helper so that every action fires; this affects the test only, never the engine defaults. If a
parameter change makes an affordance unreachable, this test fails.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from alife import layout
from alife.config import Config, dims
from alife.state import (
    ERROR_MASK,
    N_COUNTERS,
    CounterIdx,
    DeathCause,
    LedgerIdx,
    init_state,
    to_numpy,
)
from alife.step import make_step
from helpers import make_scene, set_output_bias

STEPS = 400
C = CounterIdx

# Output biases for the founders: harvest, bond, share, reproduce (bonded), attack and emit.
BIAS = {
    layout.OUT_HARVEST: 2.0,
    layout.OUT_BOND: 2.0,
    **{layout.OUT_SHARE + b: 1.0 for b in range(layout.B)},
    layout.OUT_REPRODUCE: 2.0,
    layout.OUT_CHILD_BOND: 2.0,
    layout.OUT_ATTACK: 1.0,
    **{layout.OUT_EMIT + c: 1.0 for c in range(layout.NCH)},
}


def _np(a: mx.array) -> np.ndarray:
    return to_numpy(a)


@pytest.fixture(scope="module")
def affordance_run(tiny_cfg: Config) -> dict:
    cfg = tiny_cfg
    assert cfg.run.seed == 0
    s = set_output_bias(init_state(cfg), cfg, BIAS)
    fn = make_step(cfg)
    totals = np.zeros(N_COUNTERS, np.int64)
    ledger = np.zeros(len(LedgerIdx), np.int64)
    parent_of: dict[int, int] = {}
    moved = 0.0
    for _ in range(STEPS):
        prev_pos, prev_alive = _np(s.pos), _np(s.alive)
        s, out = fn(s)
        mx.eval(s, out)
        totals += _np(out.counters)
        ledger += _np(out.ledger)
        bv = _np(out.births.valid)
        uids, parents = _np(out.births.uid)[bv].tolist(), _np(out.births.parent_uid)[bv].tolist()
        parent_of.update(zip(uids, parents, strict=True))
        still = prev_alive & _np(s.alive)
        moved += float(np.abs(_np(s.pos)[still] - prev_pos[still]).sum())
    return {"state": s, "totals": totals, "ledger": ledger, "parent_of": parent_of, "moved": moved}


def test_every_affordance_is_exercised(tiny_cfg: Config, affordance_run: dict) -> None:
    t, led, s = affordance_run["totals"], affordance_run["ledger"], affordance_run["state"]
    assert t[C.BIRTHS] > 0  # REPRODUCE
    assert t[C.DEATHS_STARVED] + t[C.DEATHS_KILLED] > 0  # DIE
    assert t[C.ATTACKS] > 0 and t[C.ATTACK_HITS] > 0 and led[LedgerIdx.FLOW_STOLEN] > 0  # FIGHT
    assert t[C.BONDS_FORMED] > 0  # BOND (adult handshake)
    assert t[C.BONDS_BROKEN] > 0  # SPLIT (fission)
    assert t[C.CHILD_BONDS] > 0  # SPLIT (division, staying bonded)
    assert t[C.SHARERS] > 0 and led[LedgerIdx.FLOW_SHARED] > 0  # SHARE ENERGY
    assert t[C.EMITTERS] > 0  # SIGNAL (chemical deposits)
    assert led[LedgerIdx.FLOW_HARVEST] > 0  # HARVEST
    assert affordance_run["moved"] > 0.0  # MOVE
    assert int(_np(s.alive).sum()) > 0  # the population survives the run
    assert int(_np(s.diag)) & ERROR_MASK == 0


def test_signals_reach_the_world(affordance_run: dict) -> None:
    s = affordance_run["state"]
    alive = _np(s.alive)
    assert np.abs(_np(s.msg)[alive]).max() > 0.0  # broadcast channels
    assert _np(s.chem).sum(axis=(1, 2)).min() > 0.0  # every chemical channel holds a deposit


def test_reproduction_mutates_genes_and_weights(tiny_cfg: Config, affordance_run: dict) -> None:
    s, parent_of = affordance_run["state"], affordance_run["parent_of"]
    alive, uid = _np(s.alive), _np(s.uid)
    assert _np(s.generation)[alive].max() >= 1
    row_of = {int(u): int(r) for r, u in enumerate(uid) if alive[r]}
    step = int(_np(s.step))
    birth = _np(s.birth_step)
    w1, w2 = _np(s.W1), _np(s.W2)
    sigma = _np(s.mut_sigma)
    pairs = [
        (row_of[c], row_of[p])
        for c, p in parent_of.items()
        if c in row_of and p in row_of and birth[row_of[c]] < step - 1  # flushed child rows (§12.6)
    ]
    assert pairs, "no living parent-child pair to compare"
    changed = sum(int((w1[c] != w1[p]).sum() + (w2[c] != w2[p]).sum()) for c, p in pairs)
    assert changed > 0  # weight mutation
    assert any(sigma[c] != sigma[p] for c, p in pairs)  # self-adaptive σ mutates
    assert any(_np(s.clade)[c] == _np(s.clade)[p] for c, p in pairs)  # lineage is inherited


def test_attack_drains_a_poor_victim_to_death(micro_cfg: Config) -> None:
    """Three attackers surround a victim holding less than their demand: it dies KILLED this step (§10.7)."""
    cfg = micro_cfg
    d = dims(cfg)
    w2 = np.zeros((d.w2_rows, layout.D_OUT), np.float32)
    w2[d.h, layout.OUT_ATTACK] = 4.0  # zero brains except an attack bias: the victim is the only target
    centre = np.array([24.0, 24.0])
    attackers = [
        dict(pos=centre + 2.0 * np.array([np.cos(a), np.sin(a)]), energy=50_000, W2=w2, birth_step=-50)
        for a in (0.0, 2.0 * np.pi / 3, 4.0 * np.pi / 3)
    ]
    victim = dict(pos=centre, energy=1_000, birth_step=-50, tag=(0.5, 0.0, 0.0, 0.0))  # attackers: tag 0
    s = make_scene(cfg, [victim, *attackers], resource=0)
    s1, out = make_step(cfg, compile=False)(s)
    mx.eval(s1, out)
    c = _np(out.counters)
    assert c[C.ATTACKS] == 3 and c[C.ATTACK_HITS] == 3
    # Kin counters (§11): sim(attacker, victim) = e^−1; each attacker also senses the two others (sim 1).
    sim = float(np.exp(np.float32(-0.25 / cfg.sensing.tag_scale**2)))
    assert c[C.KIN_ATTACK_SUM] == 3 * round(256 * sim)
    assert c[C.KIN_ATTACK_BASE] == 3 * round(256 * (sim + 2.0) / 3.0)
    assert c[C.DEATHS_KILLED] == 1 and not bool(_np(s1.alive)[0])
    dv = _np(out.deaths.valid)
    assert _np(out.deaths.uid)[dv].tolist() == [0] and _np(out.deaths.cause)[dv].tolist() == [
        DeathCause.KILLED
    ]
    led = _np(out.ledger)
    assert led[LedgerIdx.FLOW_STOLEN] == led[LedgerIdx.FLOW_ATTACK_GAIN] + led[LedgerIdx.FLOW_SPILL] > 0
    assert int(_np(s1.total)) - int(_np(s.total)) == int(
        led[LedgerIdx.INJECT_REGEN] + led[LedgerIdx.INJECT_IMMIGRATION] - led[2:7].sum()
    )
    # The corpse (S + whatever was left, here nothing) went back into the field.
    assert led[LedgerIdx.FLOW_CORPSE] == cfg.energy.structure


def test_mutual_bond_counts_two_proposers(micro_cfg: Config) -> None:
    """Two willing agents in range target each other: one pair forms, both are proposers (§16.3)."""
    cfg = micro_cfg
    d = dims(cfg)
    w2 = np.zeros((d.w2_rows, layout.D_OUT), np.float32)
    w2[d.h, layout.OUT_BOND] = 4.0
    a = dict(pos=(20.0, 20.0), W2=w2, birth_step=-50, tag=(0.5, 0.0, 0.0, 0.0))
    b = dict(pos=(22.5, 20.0), W2=w2, birth_step=-50)
    s = make_scene(cfg, [a, b])
    s1, out = make_step(cfg, compile=False)(s)
    mx.eval(s1, out)
    c = _np(out.counters)
    sim = float(np.exp(np.float32(-0.25 / cfg.sensing.tag_scale**2)))
    assert c[C.BONDS_FORMED] == 1 and c[C.BONDS_TOTAL] == 1 and c[C.BOND_PROPOSERS] == 2
    assert c[C.KIN_BOND_SUM] == 2 * round(256 * sim) and c[C.KIN_BOND_BASE] == 2 * round(256 * sim)
    assert _np(s1.partner)[0, 0] == 1 and _np(s1.partner)[1, 0] == 0
