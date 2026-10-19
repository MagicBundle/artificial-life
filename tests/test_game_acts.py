"""§28.2 the observer channel (G1) and its social-tracking additions (G2, §29.2): ``step(..., observe=True)``
returns ``StepOutG``, whose ``Acts`` agree with the ledger, the counters, the post-S8 actions, the neighbour
rows, the brain output and the birth lanes; ``energy.share`` is ``share_from_slots(share_slots(...))``."""

from __future__ import annotations

from typing import Any, NamedTuple

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from alife import actions, brain, energy, layout, lifecycle, ops, spatial
from alife.actions import Actions
from alife.config import Config, apply_overrides, dims
from alife.game.rules import RULES_V1
from alife.state import CounterIdx as C
from alife.state import LedgerIdx as L
from alife.state import State, init_state, to_numpy
from alife.step import KIN_Q_CLIP, ActFlag, Acts, StepOut, StepOutG, make_step, step
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
    assert [f.value for f in ActFlag] == [1, 2, 4, 8, 16, 32, 64]
    names = ["ATTACKED", "HIT", "FORMED", "BUDDED", "EMITTED", "PROPOSED", "KILLED"]
    assert [f.name for f in ActFlag] == names
    assert Acts._fields == (
        *("attack_gain", "share_out", "share_got", "flags", "thrust", "defend", "emit"),  # G1
        *("attack_target", "bond_target", "q_kin", "nbr", "born_row", "born_parent", "born_uid"),  # G2
    )
    d = dims(tiny_cfg)
    n, m, k, b = d.n, d.m, d.k, layout.B
    _, out = make_step(tiny_cfg, observe=True)(init_state(tiny_cfg))
    mx.eval(out)
    a = out.acts
    assert (a.attack_gain.shape, a.share_out.shape, a.share_got.shape) == ((n,), (n, b), (n, b))
    assert (a.flags.shape, a.thrust.shape, a.defend.shape, a.emit.shape) == ((n,),) * 4
    assert (a.attack_gain.dtype, a.share_out.dtype, a.share_got.dtype) == (mx.int32,) * 3
    assert (a.flags.dtype, a.thrust.dtype, a.defend.dtype, a.emit.dtype) == (mx.uint8, *(mx.int16,) * 3)
    shapes = (a.attack_target.shape, a.bond_target.shape, a.q_kin.shape, a.nbr.shape)
    assert shapes == ((n,), (n,), (n, 2), (n, k))
    assert (a.born_row.shape, a.born_parent.shape, a.born_uid.shape) == ((m,),) * 3
    assert (a.attack_target.dtype, a.bond_target.dtype, a.q_kin.dtype, a.nbr.dtype) == (
        mx.int32, mx.int32, mx.int16, mx.int32,
    )  # fmt: skip
    assert (a.born_row.dtype, a.born_parent.dtype, a.born_uid.dtype) == (mx.int32, mx.int32, mx.int64)
    assert KIN_Q_CLIP == 127.0 and round(256 * KIN_Q_CLIP) <= np.iinfo(np.int16).max


class _Post(NamedTuple):
    thrust: np.ndarray
    defend: np.ndarray
    emit: np.ndarray
    attack_target: np.ndarray  # (N,) post-S8
    bond_target: np.ndarray
    y: np.ndarray  # (N, D_OUT) float32, the brain output
    nbr_idx: np.ndarray  # (N, K) the neighbour finder's idx and valid
    nbr_valid: np.ndarray
    lane_kind: np.ndarray  # (M,) the lane allocation
    lane_parent: np.ndarray


def _post_s8(prev: State, cfg: Config) -> _Post:
    """The quantised post-S8 actions, targets, brain output, neighbours and lanes of an eager step from
    ``prev``: ``actions.mask``, ``brain.forward``, ``spatial.find_neighbors`` and ``lifecycle.allocate_lanes``
    are spied on, and the last ``Actions`` ``mask`` returns is the one S8 masks by solvency (``decode`` masks
    by ``alive`` before it)."""
    seen: list[Actions] = []
    got: dict[str, Any] = {}
    real_mask, real_fwd = actions.mask, brain.forward
    real_nbr, real_lanes = spatial.find_neighbors, lifecycle.allocate_lanes

    def spy_mask(act, keep):
        out = real_mask(act, keep)
        if isinstance(out, Actions):
            seen.append(out)
        return out

    def once(name, fn):
        def wrapped(*a, **k):
            assert name not in got, f"{name} called twice"
            got[name] = fn(*a, **k)
            return got[name]

        return wrapped

    with pytest.MonkeyPatch.context() as m:
        m.setattr(actions, "mask", spy_mask)
        m.setattr(brain, "forward", once("fwd", real_fwd))
        m.setattr(spatial, "find_neighbors", once("nbr", real_nbr))
        m.setattr(lifecycle, "allocate_lanes", once("lanes", real_lanes))
        _, out = make_step(cfg, compile=False)(prev)
        mx.eval(out)
    act = seen[-1]
    nbr, lanes = got["nbr"], got["lanes"]
    return _Post(
        _np(mx.round(256.0 * act.thrust).astype(mx.int16)),
        _np(mx.round(256.0 * act.defend).astype(mx.int16)),
        _np(ops.floor_i32(act.emit * 256.0).sum(axis=1).astype(mx.int16)),
        _np(act.attack_target),
        _np(act.bond_target),
        _np(got["fwd"][1]),
        _np(nbr.idx),
        _np(nbr.valid),
        _np(lanes.kind),
        _np(lanes.parent),
    )


# The brain-output columns of the kin feature in the attack and bond queries (§29.2): written as literals, and
# tied to the layout names by ``test_q_kin_columns_are_the_kin_components``.
Q_KIN_COLS = (31, 25)


def _q_kin_ref(y: np.ndarray, alive0: np.ndarray) -> np.ndarray:
    """``round(256·clip(y_kin, ±127))`` of both queries as int16, 0 on rows dead at the step's start."""
    q = np.rint(np.float32(256.0) * np.clip(y[:, list(Q_KIN_COLS)], -KIN_Q_CLIP, KIN_Q_CLIP))
    return np.where(alive0[:, None], q, 0).astype(np.int16)


def _arm(name: str, tiny_cfg: Config, tiny3d_cfg: Config) -> tuple[Config, dict[int, float]]:
    return {
        "2d": (tiny_cfg, {}),
        "3d": (tiny3d_cfg, {}),
        "biased": (tiny_cfg, BIAS),
        "immigration": (apply_overrides(tiny_cfg, IMMIGRATION), {}),
    }[name]


STARVE_AT = (0, 20)  # steps at which a few rows are made insolvent (energy 0 cannot pay the basal cost)
N_STARVED = 4


def _starve(s: State) -> tuple[State, np.ndarray]:
    """``s`` with the energy of its first ``N_STARVED`` alive rows set to 0, and their indexes. The solvency
    mask of the thrust, defend and emit (S8) is then exercised on purpose in every arm, not by chance."""
    rows = np.flatnonzero(_np(s.alive))[:N_STARVED]
    energy_ = _np(s.energy).copy()
    energy_[rows] = 0
    return s._replace(energy=mx.array(energy_)), rows


def _check_social_acts(a: Acts, out: StepOutG, post: _Post, prev: State, s: State, where: str) -> None:
    """The additions of §29.2 against the spied sub-steps of an eager step from ``prev``."""
    n, k = post.nbr_idx.shape
    alive0 = _np(prev.alive)
    fl = _np(a.flags)
    c = _np(out.counters)
    # the targets are the post-S8 targets, the kin queries the quantised columns of the brain output
    assert np.array_equal(_np(a.attack_target), post.attack_target), where
    assert np.array_equal(_np(a.bond_target), post.bond_target), where
    assert np.array_equal(_np(a.q_kin), _q_kin_ref(post.y, alive0)), where
    assert np.abs(_np(a.q_kin).astype(np.int64)).max() <= round(256 * KIN_Q_CLIP), where
    # the neighbour rows: the finder's idx where valid; validity is exactly "idx is not the row's own index"
    rows = np.arange(n)[:, None]
    assert np.array_equal(post.nbr_valid, post.nbr_idx != rows), where
    assert np.array_equal(_np(a.nbr), np.where(post.nbr_valid, post.nbr_idx, -1)), where
    nb = _np(a.nbr)
    assert alive0[nb[nb >= 0]].all(), where  # sensed neighbours were alive at the step's start
    # a proposal formed a bond with its bond target; a mutual pair is two PROPOSED rows
    prop, formed = (fl & ActFlag.PROPOSED.value) > 0, (fl & ActFlag.FORMED.value) > 0
    bt = _np(a.bond_target)
    assert (bt[prop] >= 0).all() and formed[prop].all() and formed[bt[prop]].all(), where
    assert prop.sum() == c[C.BOND_PROPOSERS], where
    # the flag KILLED is the DEATHS_KILLED popcount, and a killed row was some attacker's target
    killed = (fl & ActFlag.KILLED.value) > 0
    assert killed.sum() == c[C.DEATHS_KILLED], where
    attacked = (fl & ActFlag.ATTACKED.value) > 0
    assert set(np.flatnonzero(killed)) <= set(_np(a.attack_target)[attacked]), where
    survivor = _np(s.alive) & (_np(s.uid) == _np(prev.uid))  # a reused row holds a newborn's new uid
    assert alive0[killed].all() and not survivor[killed].any(), where
    # the birth lanes: the records' rows and uids, the parent row on birth lanes only
    births = out.births
    born_row, born_uid, born_parent = (_np(x) for x in (a.born_row, a.born_uid, a.born_parent))
    assert np.array_equal(born_row, _np(births.slot)) and np.array_equal(born_uid, _np(births.uid)), where
    kind = post.lane_kind
    is_birth = kind == lifecycle.LaneKind.BIRTH.value
    assert np.array_equal(born_row >= 0, kind != lifecycle.LaneKind.PAD.value), where
    assert np.array_equal(born_parent >= 0, is_birth), where
    assert np.array_equal(born_parent[is_birth], post.lane_parent[is_birth]), where
    assert (born_parent[~is_birth] == -1).all(), where
    used = born_row >= 0
    assert np.array_equal(_np(s.uid)[born_row[used]], born_uid[used]), where  # the newborn row holds the uid
    assert np.array_equal(  # the parent row held the uid the record names, and it survived
        _np(prev.uid)[born_parent[is_birth]], _np(births.parent_uid)[is_birth]
    ), where
    assert _np(s.alive)[born_parent[is_birth]].all(), where
    assert k == nb.shape[1], where


@pytest.mark.parametrize("arm", ["2d", "3d", "biased", "immigration"])
def test_acts_match_the_ledger_and_counters(arm: str, tiny_cfg: Config, tiny3d_cfg: Config) -> None:
    cfg, bias = _arm(arm, tiny_cfg, tiny3d_cfg)
    s = init_state(cfg)
    if bias:
        s = set_output_bias(s, cfg, bias)
    fn = make_step(cfg, observe=True)
    seen = dict.fromkeys(
        ("attack", "hit", "formed", "bud", "imm", "both", "share", "emit", "insolvent"), 0
    ) | dict.fromkeys(("proposed", "nbr", "q_nonzero"), 0)
    for t in range(STEPS):
        starved = np.array([], np.int64)
        if t in STARVE_AT:
            s, starved = _starve(s)
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
        # the social-tracking additions (§29.2): targets, kin queries, neighbour rows, birth lanes, two flags
        _check_social_acts(a, out, post, prev, s, where)
        thrust, defend, emit = (_np(x).astype(np.int64) for x in (a.thrust, a.defend, a.emit))
        assert thrust.min() >= -256 and thrust.max() <= 256, where
        assert defend.min() >= 0 and defend.max() <= 256, where
        assert emit.min() >= 0 and emit.max() <= 768, where  # 3 channels × floor(256·1.0)
        # an insolvent row acts as nothing: its thrust, defend and emit are masked (S8), whatever it decoded
        assert c[C.INSOLVENT] >= len(starved), where
        for x in (a.thrust, a.defend, a.emit, a.attack_gain, a.share_out, a.flags):
            assert not _np(x)[starved].any(), where
        for x in (a.attack_target, a.bond_target):  # and the S8 mask cancels its targets
            assert (_np(x)[starved] == -1).all(), where
        # every act is zero on rows dead at the step's start
        dead0 = ~alive0
        for x in (a.attack_gain, a.share_out, a.share_got, a.flags, a.thrust, a.defend, a.emit, a.q_kin):
            assert not _np(x)[dead0].any(), where
        for x in (a.attack_target, a.bond_target, a.nbr):  # the new per-row acts are −1 there
            assert (_np(x)[dead0] == -1).all(), where
        seen["attack"] += c[C.ATTACKS]
        seen["hit"] += c[C.ATTACK_HITS]
        seen["formed"] += c[C.BONDS_FORMED]
        seen["bud"] += c[C.BIRTHS]
        seen["imm"] += c[C.IMMIGRANTS]
        seen["both"] += int(c[C.BIRTHS] > 0 and c[C.IMMIGRANTS] > 0)
        seen["share"] += int(led[L.FLOW_SHARED] > 0)
        seen["emit"] += c[C.EMITTERS]
        seen["insolvent"] += c[C.INSOLVENT]
        seen["proposed"] += has[ActFlag.PROPOSED]
        seen["nbr"] += int((_np(a.nbr) >= 0).any(axis=1).sum())
        seen["q_nonzero"] += int(_np(a.q_kin).any(axis=1).sum())
    # the arm exercises what it checks: every act fires, and the immigration arm has both lane kinds at once
    for k in ("attack", "hit", "formed", "bud", "share", "emit", "insolvent", "proposed", "nbr", "q_nonzero"):
        assert seen[k] > 0, (arm, k)
    assert (seen["imm"] > 0) == (arm == "immigration")
    if arm == "immigration":
        assert seen["both"] > 0


def test_killed_flag_under_forced_attacks(tiny_cfg: Config) -> None:
    """With every founder attacking and holding 800 quanta, victims are drained to 0 and die KILLED: the
    flag's popcount is DEATHS_KILLED and each killed row was an attack target (``_check_social_acts``)."""
    s = set_output_bias(init_state(tiny_cfg), tiny_cfg, BIAS)
    e = _np(s.energy).copy()
    e[_np(s.alive)] = 800
    s = s._replace(energy=mx.array(e))
    fn = make_step(tiny_cfg, observe=True)
    kills = 0
    for t in range(6):
        prev = s
        s, out = fn(prev)
        mx.eval(s, out)
        _check_social_acts(out.acts, out, _post_s8(prev, tiny_cfg), prev, s, f"step {t}")
        kills += int(((_np(out.acts.flags) & ActFlag.KILLED.value) > 0).sum())
    assert kills > 0


def test_q_kin_columns_are_the_kin_components() -> None:
    """The kin feature is φ[NBR_KIN − 1] (φ = desc[..., 1:]), so its query weights are column NBR_KIN − 1
    of the attack and bond query blocks: 29 + 2 = 31 and 23 + 2 = 25, in both dims."""
    assert layout.NBR_FEATURE_NAMES[1:][layout.NBR_KIN - 1] == "kin"
    assert [layout.OUTPUT_NAMES[c] for c in Q_KIN_COLS] == ["q_attack2", "q_bond2"]
    assert Q_KIN_COLS == (layout.OUT_Q_ATTACK + layout.NBR_KIN - 1, layout.OUT_Q_BOND + layout.NBR_KIN - 1)
    for nd in (2, 3):  # the 3D layout appends to the 2D one (the _DZ queries come last)
        lay = layout.for_dims(nd)
        assert [lay.OUTPUT_NAMES[c] for c in Q_KIN_COLS] == ["q_attack2", "q_bond2"]


@pytest.mark.parametrize("d3", [False, True], ids=["2d", "3d"])
def test_q_kin_clips_and_masks(d3: bool, tiny_cfg: Config, tiny3d_cfg: Config) -> None:
    """Saturated weights drive the kin outputs to about ±132, past KIN_Q_CLIP, so the quantised value is
    ±round(256·127) = ±32512 on the living rows; a dead row is 0 whatever the brain computed for it."""
    cfg = tiny3d_cfg if d3 else tiny_cfg
    d, clip = dims(cfg), cfg.brain.weight_clip
    s = init_state(cfg)
    alive = _np(s.alive)
    assert 0 < alive.sum() < d.n  # a dead row exists, and its brain is evaluated too
    w1, w2 = (_np(w).astype(np.float32) for w in (s.W1, s.W2))
    w1[:] = 0.0
    w1[:, layout.IN_BIAS, :] = clip  # every hidden unit is tanh(clip) = 0.9993, on dead rows too
    w2[:, :, Q_KIN_COLS[0]] = clip  # y_kin_attack ≈ +clip·(H·tanh(clip) + 1) ≈ +132
    w2[:, :, Q_KIN_COLS[1]] = -clip  # y_kin_bond ≈ −132
    W1, W2 = mx.array(w1).astype(d.wdtype), mx.array(w2).astype(d.wdtype)
    s = s._replace(W1=W1, W2=W2)
    assert clip * (d.h * np.tanh(clip) + 1) > KIN_Q_CLIP  # the premise: the unclipped value exceeds the bound
    s1, out = make_step(cfg, observe=True)(s)
    mx.eval(s1, out)
    q = _np(out.acts.q_kin)
    lim = round(256 * KIN_Q_CLIP)
    assert (q[alive] == [lim, -lim]).all() and not q[~alive].any()
    # the same state through the eager step: the clip is not a compilation artefact
    _, o_eager = make_step(cfg, compile=False, observe=True)(s)
    assert np.array_equal(_np(o_eager.acts.q_kin), q)


def test_q_kin_is_the_unclipped_quantisation_below_the_bound(tiny_cfg: Config) -> None:
    """Between the bounds the value is round(256·y), an exact int16 of the float32 output."""
    s = init_state(tiny_cfg)
    _, out = make_step(tiny_cfg, observe=True)(s)
    post = _post_s8(s, tiny_cfg)
    mx.eval(out)
    y = post.y[:, list(Q_KIN_COLS)]
    alive = _np(s.alive)
    assert np.abs(y[alive]).max() < KIN_Q_CLIP and np.abs(y[alive]).max() > 0
    assert np.array_equal(_np(out.acts.q_kin)[alive], np.rint(np.float32(256.0) * y[alive]).astype(np.int16))


@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_emit_reaches_its_upper_bound(dtype: str, tiny_cfg: Config) -> None:
    """``tanh`` reaches exactly 1.0 in float32 and ``floor(256·1.0) = 256``, so ``emit`` is bounded by 768,
    not 765: weights within ``weight_clip`` can drive every hidden unit and every emit output to saturation.
    The rule set's int32 check uses that bound."""
    cfg = apply_overrides(tiny_cfg, [f"brain.weight_dtype={dtype}"])
    d, clip = dims(cfg), cfg.brain.weight_clip
    s = init_state(cfg)
    alive = _np(s.alive)
    w1, w2 = (_np(w).astype(np.float32) for w in (s.W1, s.W2))
    w1[alive, layout.IN_BIAS, :] = clip  # every hidden unit gets +clip from the constant input
    for c in range(layout.NCH):
        w2[alive, :, layout.OUT_EMIT + c] = clip  # y_emit = clip·(Σ_j h_j + 1), far past tanh's saturation
    W1, W2 = mx.array(w1).astype(d.wdtype), mx.array(w2).astype(d.wdtype)
    cost = brain.l1_cost(W1, W2, cfg)
    s = s._replace(W1=W1, W2=W2, brain_cost=mx.where(s.alive, cost, s.brain_cost))
    s1, out = make_step(cfg, observe=True)(s)
    mx.eval(s1, out)
    emit = _np(out.acts.emit).astype(np.int64)
    assert emit.max() <= layout.NCH * 256
    assert ON_GPU or (emit[alive] == layout.NCH * 256).all()  # on the CPU every living row saturates
    assert emit.max() > layout.NCH * 255  # past the old bound 765
    RULES_V1.check(cfg)  # the accumulator bound of the emit column is NCH·256


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
