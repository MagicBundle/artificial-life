"""§28.2 the observer channel (G1): ``step(..., observe=True)`` returns ``StepOutG``, whose ``Acts`` agree
with the ledger, the counters and the post-S8 actions; ``energy.share`` is
``share_from_slots(share_slots(...))``."""

from __future__ import annotations

from typing import Any, NamedTuple

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from alife import actions, energy, layout, ops
from alife.actions import Actions
from alife.config import Config, apply_overrides, dims
from alife.state import CounterIdx as C
from alife.state import LedgerIdx as L
from alife.state import State, init_state, to_numpy
from alife.step import ActFlag, Acts, StepOut, StepOutG, make_step, step
from helpers import ON_GPU, set_output_bias

STEPS = 40

# Founders' output biases that make every act fire (the affordance tests' BIAS).
BIAS = {
    layout.OUT_HARVEST: 2.0,
    layout.OUT_BOND: 2.0,
    **{layout.OUT_SHARE + b: 1.0 for b in range(layout.B)},
    layout.OUT_REPRODUCE: 2.0,
    layout.OUT_CHILD_BOND: 2.0,
    layout.OUT_ATTACK: 1.0,
    **{layout.OUT_EMIT + c: 1.0 for c in range(layout.NCH)},
}
IMMIGRATION = ["population.immigration_every=5", "population.immigration_count=8"]


def _np(a: mx.array) -> np.ndarray:
    return to_numpy(a)


def _same(a: Any, b: Any) -> bool:
    fa, fb = tree_flatten(a), tree_flatten(b)
    return [k for k, _ in fa] == [k for k, _ in fb] and all(
        x.dtype == y.dtype and np.array_equal(_np(x), _np(y)) for (_, x), (_, y) in zip(fa, fb, strict=True)
    )


def test_step_returns_stepout_unless_observing(tiny_cfg: Config) -> None:
    s = init_state(tiny_cfg)
    for fn in (
        make_step(tiny_cfg),
        make_step(tiny_cfg, observe=False),
        lambda x: step(x, tiny_cfg),
        lambda x: step(x, tiny_cfg, observe=False),
    ):
        s1, out = fn(s)
        mx.eval(s1, out)
        assert type(out) is StepOut
    s_off, o_off = make_step(tiny_cfg)(s)
    s_on, o_on = make_step(tiny_cfg, observe=True)(s)
    mx.eval(s_off, o_off, s_on, o_on)
    assert type(o_on) is StepOutG and type(o_on.acts) is Acts and type(o_on.base()) is StepOut
    assert StepOutG._fields == (*StepOut._fields, "acts")
    assert _same(o_on.base(), o_off) and _same(s_on, s_off)
    # Eager gives the same types, and on the CPU (fused kernels round like eager ones) the same channel.
    _, o_eager = make_step(tiny_cfg, compile=False, observe=True)(s)
    assert type(o_eager) is StepOutG and type(o_eager.acts) is Acts
    assert ON_GPU or _same(o_eager, o_on)


def test_act_flag_bits_and_dtypes(tiny_cfg: Config) -> None:
    assert [f.value for f in ActFlag] == [1, 2, 4, 8, 16]
    assert [f.name for f in ActFlag] == ["ATTACKED", "HIT", "FORMED", "BUDDED", "EMITTED"]
    n, b = dims(tiny_cfg).n, layout.B
    _, out = make_step(tiny_cfg, observe=True)(init_state(tiny_cfg))
    mx.eval(out)
    a = out.acts
    assert (a.attack_gain.shape, a.share_out.shape, a.share_got.shape) == ((n,), (n, b), (n, b))
    assert (a.flags.shape, a.thrust.shape, a.defend.shape, a.emit.shape) == ((n,),) * 4
    assert (a.attack_gain.dtype, a.share_out.dtype, a.share_got.dtype) == (mx.int32,) * 3
    assert (a.flags.dtype, a.thrust.dtype, a.defend.dtype, a.emit.dtype) == (mx.uint8, *(mx.int16,) * 3)


class _Post(NamedTuple):
    thrust: np.ndarray
    defend: np.ndarray
    emit: np.ndarray


def _post_s8(prev: State, cfg: Config) -> _Post:
    """The quantised post-S8 actions of an eager step from ``prev``: ``actions.mask`` is spied on, and the
    last ``Actions`` it returns is the one S8 masks by solvency (``decode`` masks by ``alive`` before it)."""
    seen: list[Actions] = []
    real = actions.mask

    def spy(act, keep):
        out = real(act, keep)
        if isinstance(out, Actions):
            seen.append(out)
        return out

    with pytest.MonkeyPatch.context() as m:
        m.setattr(actions, "mask", spy)
        _, out = make_step(cfg, compile=False)(prev)
        mx.eval(out)
    act = seen[-1]
    return _Post(
        _np(mx.round(256.0 * act.thrust).astype(mx.int16)),
        _np(mx.round(256.0 * act.defend).astype(mx.int16)),
        _np(ops.floor_i32(act.emit * 256.0).sum(axis=1).astype(mx.int16)),
    )


def _arm(name: str, tiny_cfg: Config, tiny3d_cfg: Config) -> tuple[Config, dict[int, float]]:
    return {
        "2d": (tiny_cfg, {}),
        "3d": (tiny3d_cfg, {}),
        "biased": (tiny_cfg, BIAS),
        "immigration": (apply_overrides(tiny_cfg, IMMIGRATION), {}),
    }[name]


@pytest.mark.parametrize("arm", ["2d", "3d", "biased", "immigration"])
def test_acts_match_the_ledger_and_counters(arm: str, tiny_cfg: Config, tiny3d_cfg: Config) -> None:
    cfg, bias = _arm(arm, tiny_cfg, tiny3d_cfg)
    s = init_state(cfg)
    if bias:
        s = set_output_bias(s, cfg, bias)
    fn = make_step(cfg, observe=True)
    seen = dict.fromkeys(("attack", "hit", "formed", "bud", "imm", "both", "share", "emit"), 0)
    for t in range(STEPS):
        prev = s
        s, out = fn(prev)
        mx.eval(s, out)
        a, c, led = out.acts, _np(out.counters), _np(out.ledger)
        fl = _np(a.flags)
        has = {f: ((fl & f.value) > 0).sum() for f in ActFlag}
        where = f"{arm} step {t}"
        assert _np(a.attack_gain).sum() == led[L.FLOW_ATTACK_GAIN], where
        assert _np(a.share_out).sum() == led[L.FLOW_SHARED], where
        assert _np(a.share_out).sum() - _np(a.share_got).sum() == led[L.HEAT_SHARE], where
        assert has[ActFlag.ATTACKED] == c[C.ATTACKS] and has[ActFlag.HIT] == c[C.ATTACK_HITS], where
        assert has[ActFlag.FORMED] == 2 * c[C.BONDS_FORMED], where
        assert has[ActFlag.BUDDED] == c[C.BIRTHS], where  # also on steps with immigrants (distinct parents)
        assert has[ActFlag.EMITTED] == c[C.EMITTERS], where
        # Σ_b got is the inflow the survivors keep in State
        alive0, alive1 = _np(prev.alive), _np(s.alive)
        surv = alive0 & alive1 & (_np(s.uid) == _np(prev.uid))
        assert np.array_equal(_np(a.share_got).sum(axis=1)[surv], _np(s.last_inflow)[surv]), where
        # the quantised actions are those of the post-S8 actions, recomputed eagerly from the same state
        post = _post_s8(prev, cfg)
        assert np.array_equal(_np(a.thrust), post.thrust), where
        assert np.array_equal(_np(a.defend), post.defend), where
        assert np.array_equal(_np(a.emit), post.emit), where
        thrust, defend, emit = (_np(x).astype(np.int64) for x in (a.thrust, a.defend, a.emit))
        assert thrust.min() >= -256 and thrust.max() <= 256, where
        assert defend.min() >= 0 and defend.max() <= 256, where
        assert emit.min() >= 0 and emit.max() <= 765, where
        # every act is zero on rows dead at the step's start
        dead0 = ~alive0
        for x in (a.attack_gain, a.share_out, a.share_got, a.flags, a.thrust, a.defend, a.emit):
            assert not _np(x)[dead0].any(), where
        seen["attack"] += c[C.ATTACKS]
        seen["hit"] += c[C.ATTACK_HITS]
        seen["formed"] += c[C.BONDS_FORMED]
        seen["bud"] += c[C.BIRTHS]
        seen["imm"] += c[C.IMMIGRANTS]
        seen["both"] += int(c[C.BIRTHS] > 0 and c[C.IMMIGRANTS] > 0)
        seen["share"] += int(led[L.FLOW_SHARED] > 0)
        seen["emit"] += c[C.EMITTERS]
    # the arm exercises what it checks: every act fires, and the immigration arm has both lane kinds at once
    for k in ("attack", "hit", "formed", "bud", "share", "emit"):
        assert seen[k] > 0, (arm, k)
    assert (seen["imm"] > 0) == (arm == "immigration")
    if arm == "immigration":
        assert seen["both"] > 0


# --- share_slots / share_from_slots (S12) ------------------------------------------------------------------


def _share_actions(n: int, share: np.ndarray) -> Actions:
    z = np.zeros(n, np.float32)
    zb = np.zeros(n, bool)
    return Actions(
        thrust=mx.array(z),
        turn=mx.array(z),
        msg=mx.zeros((n, layout.MSG)),
        emit=mx.zeros((n, layout.NCH)),
        harvest=mx.array(z),
        grip=mx.array(z),
        contract=mx.array(z),
        bond_want=mx.array(zb),
        release=mx.zeros((n, layout.B), dtype=mx.bool_),
        share=mx.array(share),
        attack=mx.array(zb),
        defend=mx.array(z),
        reproduce=mx.array(zb),
        child_bond=mx.array(zb),
        bond_k=mx.full((n,), -1, mx.int32),
        bond_target=mx.full((n,), -1, mx.int32),
        attack_k=mx.full((n,), -1, mx.int32),
        attack_target=mx.full((n,), -1, mx.int32),
    )


def _random_bonds(rs: np.random.Generator, n: int, n_pairs: int) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric ``partner``/``rev`` of ``n_pairs`` random bonds on free slots (no self-bonds)."""
    partner = np.full((n, layout.B), -1, np.int32)
    rev = np.full((n, layout.B), -1, np.int32)
    for _ in range(n_pairs):
        i, j = (int(x) for x in rs.choice(n, 2, replace=False))
        fi, fj = np.flatnonzero(partner[i] < 0), np.flatnonzero(partner[j] < 0)
        if not len(fi) or not len(fj) or (partner[i] == j).any():
            continue
        si, sj = int(rs.choice(fi)), int(rs.choice(fj))
        partner[i, si], rev[i, si], partner[j, sj], rev[j, sj] = j, sj, i, si
    return partner, rev


def test_share_slots_sums_to_share(micro_cfg: Config) -> None:
    cfg, rs = micro_cfg, np.random.default_rng(11)
    n = dims(cfg).n
    q = ops.q8(cfg.bonds.share_efficiency)
    moved = 0
    for trial in range(12):
        partner, rev = _random_bonds(rs, n, 30)
        partner0 = partner.copy()
        if trial % 2:  # bonds that changed during the step are gated off (partner != partner0)
            drop = rs.random(partner.shape) < 0.3
            partner0 = np.where(drop, np.int32(-1), partner0)
        e = rs.integers(0, 6000, n).astype(np.int32)
        share = np.maximum(np.tanh(rs.normal(0, 1, (n, layout.B))), 0).astype(np.float32)
        solvent = rs.random(n) < 0.85
        args = (mx.array(e), mx.array(partner), mx.array(rev), mx.array(partner0))
        act = _share_actions(n, share)
        sl = energy.share_slots(*args, act, mx.array(solvent), cfg)
        sh = energy.share_from_slots(mx.array(e), sl)
        ref = energy.share(*args, act, mx.array(solvent), cfg)
        mx.eval(sl, sh, ref)
        assert type(sl) is energy.ShareSlots and type(sh) is energy.Share
        assert _same(sh, ref)
        out, inc, got = (_np(x).astype(np.int64) for x in sl)
        assert np.array_equal(_np(sh.out), out.sum(1)) and np.array_equal(_np(sh.inflow), got.sum(1))
        assert np.array_equal(_np(sh.heat), (inc - got).sum(1))  # the heat is Σ(in − g)
        assert np.array_equal(_np(sh.energy), e - out.sum(1) + got.sum(1))
        assert out.sum() - got.sum() == (inc - got).sum()  # Σ out = Σ inc: nothing is lost in transit
        # an independent per-slot reading of §10.6
        for i in range(n):
            for b in range(layout.B):
                p = partner[i, b]
                sends = p >= 0 and partner0[i, b] == p and solvent[i]
                want = min(
                    int(np.floor(np.float32(share[i, b]) * np.float32(cfg.bonds.share_rate))),
                    e[i] // layout.B,
                )
                assert out[i, b] == (want if sends else 0)
        for i in range(n):
            for b in range(layout.B):
                p = partner[i, b]
                want_inc = out[p, rev[i, b]] if p >= 0 else 0
                assert inc[i, b] == want_inc and got[i, b] == (want_inc * q) >> 8
        moved += int(out.sum())
        # compiled composition equals the eager one
        comp = mx.compile(lambda *a: energy.share_from_slots(a[0], energy.share_slots(*a, cfg)))(
            *args, act, mx.array(solvent)
        )
        mx.eval(comp)
        assert _same(comp, ref)
    assert moved > 0
