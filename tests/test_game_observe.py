"""§28.3 the observer state and update (G1): the fixed-point accumulator, the reset rule, per-slot bond
identity, the quantised inputs, and the invariants on real steps (immigration, bonds, births, a row reused in
the step it died)."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from alife import layout
from alife.config import Config, apply_overrides
from alife.game.observe import (
    OBS_CAP,
    VQ_MAX,
    ArchCol,
    IntentCol,
    ObsState,
    care_out,
    ema,
    evaluate,
    init_obs,
    n_arch_cols,
    obs_in,
    quantise_inputs,
    speed_unit,
    update,
)
from alife.game.rules import RULES_V1
from alife.state import CounterIdx as C
from alife.state import init_state, to_numpy
from alife.step import ActFlag, make_step
from game_helpers import Driver, blank_acts, living, make_acts, make_oi, np_state
from helpers import set_output_bias

B = layout.B
I32 = mx.int32


def np_(a: mx.array) -> np.ndarray:
    return to_numpy(a)


# --- the accumulator ----------------------------------------------------------------------------------------


@pytest.mark.parametrize("k", [3, 6])
def test_ema_constant_input_converges_exactly(k: int) -> None:
    x = np.array([1, 7, 400, 2796, 3600], np.int32)
    a = mx.zeros((len(x),), dtype=I32)
    history = []
    for _ in range(16 << k):
        a = ema(a, mx.array(x), k)
        mx.eval(a)
        history.append(np_(a))
    h = np.array(history)
    assert (np.diff(h, axis=0) >= 0).all()  # monotone from below
    assert (h[-1] == x << k).all()  # exactly x·2^k: the last increments are 1, no dead zone
    for _ in range(100):  # and it stays there
        a = ema(a, mx.array(x), k)
    assert (np_(a) == x << k).all() and (np_(a >> k) == x).all()


@pytest.mark.parametrize("k", [3, 6])
@pytest.mark.parametrize("spike", [256, 3600, VQ_MAX])
def test_ema_spike_decays_without_going_negative(k: int, spike: int) -> None:
    a = ema(mx.zeros((1,), dtype=I32), mx.array([spike], dtype=I32), k)
    assert int(np_(a)[0]) == spike
    last = spike
    for _ in range(10 << k):
        a = ema(a, mx.zeros((1,), dtype=I32), k)
        v = int(np_(a)[0])
        assert 0 <= v <= last  # never negative, never growing
        last = v
    assert 0 <= last < 1 << k and int(np_(a >> k)[0]) == 0  # it stops below 2^k, at mean 0


def test_ema_window_mean_identity() -> None:
    """Σ (A_t >> k) = Σ x_t − (A_end − A_start) exactly, so a windowed mean's bias is |A_end − A_start|/n."""
    k, n, series, warm = 6, 1 << 15, 64, 5 << 6
    rng = np.random.default_rng(7)
    x = mx.array(rng.integers(0, 401, size=(warm + n, series), dtype=np.int32))
    a = mx.zeros((series,), dtype=I32)
    start = mx.zeros((series,), dtype=I32)
    total_mean = mx.zeros((series,), dtype=I32)
    for t in range(warm + n):
        if t == warm:
            start = a
        if t >= warm:
            total_mean = total_mean + (a >> k)  # the pre-update A_t
        a = ema(a, x[t], k)
        if t % 1024 == 1023:
            mx.eval(a, start, total_mean)
    mx.eval(a, start, total_mean)
    xs = np_(x)[warm:].astype(np.int64).sum(axis=0)
    drift = np_(a).astype(np.int64) - np_(start).astype(np.int64)
    assert (np_(total_mean).astype(np.int64) == xs - drift).all()  # exact, in integers
    bias = np.abs(np_(total_mean) / n - xs / n)
    assert (bias == np.abs(drift) / n).all() or np.allclose(bias, np.abs(drift) / n, atol=1e-9)
    assert (np.abs(drift) / n <= 400 * 2**k / n).all() and 400 * 2**k / n < 0.79  # under one quantum


@pytest.mark.parametrize(
    ("k", "x"),
    [(6, VQ_MAX), (16, 6800)],  # the second is GAMES §4.3's bound: harvest + attack gain + shares in
)
def test_ema_headroom(k: int, x: int) -> None:
    """Under a constant maximal input A never exceeds ``x·2^k`` and ``A + x`` stays inside int32."""
    a = mx.zeros((1,), dtype=I32)
    bound, peak, inter = x << k, mx.zeros((1,), dtype=I32), mx.zeros((1,), dtype=I32)
    xv = mx.array([x], dtype=I32)
    for t in range(5 << k):
        inter = mx.maximum(inter, a + xv)  # the intermediate of the update
        a = ema(a, xv, k)
        peak = mx.maximum(peak, a)
        if t % 4096 == 4095:
            mx.eval(a, peak, inter)
    mx.eval(a, peak, inter)
    assert int(np_(peak)[0]) <= bound < 2**31
    assert int(np_(inter)[0]) <= x * ((1 << k) + 1) < 2**31
    assert int(np_(a)[0]) >= bound * 0.99  # five windows: within 1% of the bound, so the bound is tight


# --- reset, identity ----------------------------------------------------------------------------------------


def _flag(cfg: Config, rows: dict[int, int]) -> dict[str, np.ndarray]:
    acts = blank_acts(cfg)
    for r, f in rows.items():
        acts["flags"][r] = f
    return acts


def test_reset_of_died_and_new_rows(micro_cfg: Config) -> None:
    cfg, rs = micro_cfg, RULES_V1
    oi0 = living(cfg, 4, last_harvest=100)
    o = init_obs(make_oi(oi0), cfg)
    # labelled rows before the first update, to see which ones survive
    n = o.uid.shape[0]
    o = o._replace(
        label_arch=mx.full((n, 2), 3, dtype=I32),
        match_arch=mx.full((n,), 5, dtype=I32),
        label_intent=mx.full((n,), 6, dtype=I32),
        match_intent=mx.full((n,), 7, dtype=I32),
        cand=mx.full((n,), 8, dtype=I32),
        cand_n=mx.full((n,), 2, dtype=I32),
    )
    assert (np_(o.uid)[:4] == [0, 1, 2, 3]).all() and (np_(o.uid)[4:] == -1).all()
    assert (np_(o.since_bud) == OBS_CAP).all() and not np_(o.n_obs).any()

    # step 1: the same four occupants, row 0 buds
    o = update(o, make_oi(oi0), make_acts(_flag(cfg, {0: ActFlag.BUDDED.value})), cfg, rs)
    s = np_state(o)
    assert (s["n_obs"][:4] == 1).all() and not s["n_obs"][4:].any()
    assert (s["arch"][:4, ArchCol.HARVEST.value] == 100).all() and (
        s["intent"][:4, IntentCol.HARVEST.value] == 100
    ).all()
    assert (
        (s["label_arch"][:4] == 3).all()
        and (s["match_intent"][:4] == 7).all()
        and (s["cand_n"][:4] == 2).all()
    )
    assert not s["label_arch"][4:].any()  # rows that were not observed stay unlabelled
    assert s["since_bud"][0] == 0 and (s["since_bud"][1:4] == OBS_CAP).all()

    # step 2: row 1 dies, row 2 is reused by a new uid, row 3 stays, row 4 is a newborn; the acts and the
    # harvest of the reset rows belong to their previous occupants and are discarded
    oi2 = living(cfg, 4, last_harvest=100)
    oi2["alive"][1], oi2["uid"][1] = False, -1
    oi2["uid"][2] = 10
    oi2["alive"][4], oi2["uid"][4], oi2["last_harvest"][4] = True, 11, 100
    acts2 = _flag(cfg, {0: 0, 2: ActFlag.BUDDED.value, 4: ActFlag.BUDDED.value})
    o = update(o, make_oi(oi2), make_acts(acts2), cfg, rs)
    s = np_state(o)
    assert s["uid"][:5].tolist() == [0, -1, 10, 3, 11]
    assert s["n_obs"][:5].tolist() == [2, 0, 0, 2, 0]
    for r in (1, 2, 4):  # reset rows are zero everywhere
        for f in ("arch", "arch_slot", "intent", "label_arch", "match_arch", "label_intent", "match_intent"):
            assert not s[f][r].any(), (f, r)
        assert s["cand"][r] == 0 and s["cand_n"][r] == 0 and s["since_bud"][r] == OBS_CAP
    for r in (0, 3):  # kept rows keep labels and accumulate
        assert (s["label_arch"][r] == 3).all() and s["label_intent"][r] == 6 and s["cand_n"][r] == 2
        assert s["arch"][r, ArchCol.HARVEST.value] == 100 + 100 - (100 >> RULES_V1.arch_k)
    assert s["since_bud"][0] == 1 and s["since_bud"][3] == OBS_CAP

    # step 3: the new occupants of rows 2 and 4 are now observed; the dead row 1 stays reset
    oi3 = living(cfg, 4, last_harvest=100)
    oi3["alive"][1], oi3["uid"][1] = False, -1
    oi3["uid"][2] = 10
    oi3["alive"][4], oi3["uid"][4], oi3["last_harvest"][4] = True, 11, 100
    o = update(o, make_oi(oi3), make_acts(blank_acts(cfg)), cfg, rs)
    s = np_state(o)
    assert s["n_obs"][:5].tolist() == [3, 0, 1, 3, 1]
    assert s["arch"][2, ArchCol.HARVEST.value] == 100 and s["arch"][4, ArchCol.HARVEST.value] == 100
    assert s["uid"][1] == -1 and not s["arch"][1].any()


def _bond(oi: dict[str, np.ndarray], i: int, si: int, j: int, sj: int) -> None:
    oi["partner"][i, si], oi["partner"][j, sj] = j, i


def test_slot_identity_and_age(micro_cfg: Config) -> None:
    cfg, rs = micro_cfg, RULES_V1
    # rows 0..2 are founders (A, B, C), row 3 is A's child D
    base = living(cfg, 4)
    base["parent_uid"][3] = 0

    def inputs(bonds: list[tuple[int, int, int, int]]) -> dict[str, np.ndarray]:
        oi = {k: v.copy() for k, v in base.items()}
        for b in bonds:
            _bond(oi, *b)
        return oi

    ab_ad = inputs([(0, 0, 1, 0), (0, 1, 3, 0)])  # A–B in A's slot 0, A–D (parent–child) in A's slot 1
    d = Driver(cfg, make_oi(ab_ad))
    s = np_state(d.o)
    assert s["slot_uid"][0].tolist() == [1, 3, -1, -1] and s["slot_uid"][1].tolist() == [0, -1, -1, -1]
    assert s["slot_age"][0].tolist() == [OBS_CAP, OBS_CAP, 0, 0]  # bonds that predate the observation
    # kin in both directions; founders (parent_uid = −1) are not kin of each other
    assert s["slot_kin"][0].tolist() == [False, True, False, False]  # A–B no, A–D yes (A's child)
    assert s["slot_kin"][3].tolist() == [True, False, False, False]  # D–A: D's parent is A
    assert not s["slot_kin"][1:3].any() and not s["slot_kin"][4:].any()  # B, C and the empty rows

    # the bonds persist for a step, with flows on A's two slots
    acts = blank_acts(cfg)
    acts["share_out"][0, :2], acts["share_got"][0, :2] = [5, 7], [3, 0]
    s = np_state(d.step(make_oi(ab_ad), make_acts(acts)))
    assert s["arch_slot"][0, 0].tolist() == [5, 3] and s["arch_slot"][0, 1].tolist() == [7, 0]
    assert s["slot_age"][0].tolist() == [OBS_CAP, OBS_CAP, 0, 0]  # saturated
    # A's slot 0 swaps B for C in one step (flows on the old bond are discarded), B's slot is emptied
    ac_ad = inputs([(0, 0, 2, 0), (0, 1, 3, 0)])
    acts["share_out"][0, 0] = 9
    s = np_state(d.step(make_oi(ac_ad), make_acts(acts)))
    assert s["slot_uid"][0].tolist() == [2, 3, -1, -1] and s["slot_uid"][1].tolist() == [-1] * B
    assert s["arch_slot"][0, 0].tolist() == [0, 0]  # a replaced partner resets the slot ...
    assert s["slot_age"][0, 0] == 1 and s["slot_age"][2, 0] == 1  # ... and the new bond is 1 step old
    assert s["slot_age"][1, 0] == 0 and not s["arch_slot"][1].any()  # a broken bond: age 0
    assert s["slot_kin"][0].tolist() == [False, True, False, False]
    assert s["arch_slot"][0, 1].tolist() != [0, 0]  # the untouched A–D bond kept its accumulator
    # the new bond persists: it ages and accumulates from 0
    acts = blank_acts(cfg)
    acts["share_out"][0, 0] = 4
    s = np_state(d.step(make_oi(ac_ad), make_acts(acts)))
    assert s["slot_age"][0, 0] == 2 and s["arch_slot"][0, 0].tolist() == [4, 0]
    s = np_state(d.step(make_oi(ac_ad), make_acts(acts)))
    assert s["slot_age"][0, 0] == 3 and s["arch_slot"][0, 0, 0] == 4 + 4 - (4 >> rs.arch_k)
    # the bond breaks
    s = np_state(d.step(make_oi(inputs([(0, 1, 3, 0)])), make_acts(blank_acts(cfg))))
    assert s["slot_uid"][0].tolist() == [-1, 3, -1, -1] and s["slot_age"][0, 0] == 0
    assert not s["arch_slot"][0, 0].any() and not s["slot_kin"][0, 0]
    # a partner that dies leaves nothing behind either
    dead = inputs([])
    dead["alive"][3], dead["uid"][3] = False, -1
    s = np_state(d.step(make_oi(dead), make_acts(blank_acts(cfg))))
    assert s["slot_uid"][0].tolist() == [-1] * B and not s["slot_kin"].any() and s["n_obs"][3] == 0


def test_kin_needs_a_living_row(micro_cfg: Config) -> None:
    """Founders have ``parent_uid = −1`` and dead rows ``uid = −1``: a stale pointer from a dead row to a
    founder must not read as "the partner's parent is me"."""
    oi = living(micro_cfg, 2)
    oi["partner"][5, 0] = 0  # row 5 is dead (uid −1) but points at founder 0
    s = np_state(init_obs(make_oi(oi), micro_cfg))
    assert not s["slot_kin"].any() and s["slot_uid"][5, 0] == 0


def test_init_obs_shapes_and_bonds(micro_cfg: Config, micro3d_cfg: Config) -> None:
    for cfg, nd in ((micro_cfg, 2), (micro3d_cfg, 3)):
        oi = living(cfg, 3)
        _bond(oi, 0, 2, 1, 0)
        o = init_obs(make_oi(oi), cfg)
        n = len(oi["alive"])
        shapes = {
            **dict.fromkeys(("uid", "n_obs", "since_bud", "match_arch", "label_intent"), (n,)),
            **dict.fromkeys(("match_intent", "cand", "cand_n"), (n,)),
            **dict.fromkeys(("slot_uid", "slot_age", "slot_kin"), (n, B)),
            "arch": (n, n_arch_cols(nd)),
            "arch_slot": (n, B, 2),
            "intent": (n, 7),
            "label_arch": (n, 2),
        }
        assert {f: getattr(o, f).shape for f in ObsState._fields} == shapes
        dtypes = {f: getattr(o, f).dtype for f in ObsState._fields}
        assert dtypes["uid"] == dtypes["slot_uid"] == mx.int64 and dtypes["slot_kin"] == mx.bool_
        assert all(dtypes[f] == I32 for f in ObsState._fields if f not in ("uid", "slot_uid", "slot_kin"))
        s = np_state(o)
        assert s["uid"][:3].tolist() == [0, 1, 2] and (s["uid"][3:] == -1).all()
        assert (s["since_bud"] == OBS_CAP).all()
        assert (
            s["slot_age"][0, 2] == OBS_CAP
            and s["slot_age"][1, 0] == OBS_CAP
            and (s["slot_age"] > 0).sum() == 2
        )
        for f in (
            "n_obs",
            "arch",
            "arch_slot",
            "intent",
            "label_arch",
            "match_arch",
            "label_intent",
            "cand_n",
        ):
            assert not s[f].any(), f


# --- inputs -------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("dims_", ["2d", "3d"])
def test_inputs_quantisation(dims_: str, micro_cfg: Config, micro3d_cfg: Config) -> None:
    cfg, nd = (micro_cfg, 2) if dims_ == "2d" else (micro3d_cfg, 3)
    scale = np.float32(256.0 / speed_unit(cfg))
    assert speed_unit(cfg) == pytest.approx(cfg.physics.thrust_max / cfg.physics.drag)
    oi = living(cfg, 4, last_harvest=[10, 20, 30, 40], last_inflow=[1, 2, 3, 4], last_loss=[0, 5, 0, 0])
    cfg_n = len(oi["alive"])
    vel = np.zeros((cfg_n, nd), np.float32)
    vel[0] = [0.05, -0.02, 0.03][:nd]
    vel[1] = [100.0, -100.0, 100.0][:nd]  # far beyond the clip
    vel[3] = [-0.07, 0.0, 0.0][:nd]
    oi["vel"] = vel
    acts = blank_acts(cfg)
    acts["attack_gain"][:2] = [77, 5]
    acts["share_out"][0], acts["share_got"][0] = [3, 4, 5, 6], [1, 2, 3, 4]
    acts["share_out"][1], acts["share_got"][1] = [1, 0, 0, 0], [0, 0, 0, 9]
    acts["flags"][:3] = [
        ActFlag.ATTACKED.value | ActFlag.HIT.value | ActFlag.BUDDED.value,
        ActFlag.EMITTED.value | ActFlag.FORMED.value,
        ActFlag.ATTACKED.value,
    ]
    acts["thrust"][:4] = [-200, 256, 0, 31]
    acts["defend"][:4] = [130, 0, 256, 1]
    acts["emit"][:4] = [300, 0, 765, 128]
    kin_prev = np.zeros((cfg_n, B), bool)
    kin_prev[0] = [True, False, True, False]  # slots 0 and 2: care = 3 + 5
    kin_prev[1] = [False, False, False, True]  # slot 3 carries nothing sent
    x6, xs, x3 = quantise_inputs(make_oi(oi), make_acts(acts), mx.array(kin_prev), cfg)
    assert x6.dtype == xs.dtype == x3.dtype == I32
    assert x6.shape == (cfg_n, n_arch_cols(nd)) and xs.shape == (cfg_n, B, 2) and x3.shape == (cfg_n, 7)
    x6, xs, x3 = np_(x6), np_(xs), np_(x3)

    q = np.clip(np.round(vel * scale), -VQ_MAX, VQ_MAX).astype(np.int32)
    sp = np.clip(np.round(np.sqrt((vel * vel).sum(axis=1)) * scale), 0, VQ_MAX).astype(np.int32)
    assert (
        q[0, :2].tolist() == [128, -51] and q[1, 0] == VQ_MAX and q[1, 1] == -VQ_MAX
    )  # the clip is exercised
    assert sp[1] == VQ_MAX and sp[2] == 0
    assert (x6[:, ArchCol.HARVEST.value] == oi["last_harvest"]).all()
    assert (x6[:, ArchCol.GAIN.value] == acts["attack_gain"]).all()
    assert (x6[:, ArchCol.INFLOW.value] == oi["last_inflow"]).all()
    assert (x6[:, ArchCol.OUTFLOW.value] == acts["share_out"].sum(axis=1)).all()
    assert (x6[:, ArchCol.SPEED.value] == sp).all()
    assert (x6[:, 5 : 5 + nd] == np.maximum(q, 0)).all() and (x6[:, 5 + nd :] == np.maximum(-q, 0)).all()
    assert (
        x6.min() >= 0 and x6[3, 5] == 0 and x6[3, 5 + nd] == 179
    )  # the negative half of a negative component
    assert (xs[..., 0] == acts["share_out"]).all() and (xs[..., 1] == acts["share_got"]).all()
    col = IntentCol
    assert x3[:3, col.ATTACK.value].tolist() == [256, 0, 256] and not x3[3:, col.ATTACK.value].any()
    assert x3[:4, col.HURT.value].tolist() == [0, 256, 0, 0]  # last_loss > 0
    assert x3[:4, col.THRUST.value].tolist() == [200, 256, 0, 31]  # |thrust|
    assert x3[:4, col.DEFEND.value].tolist() == [130, 0, 256, 1]
    assert x3[:4, col.EMIT.value].tolist() == [300, 0, 765, 128]
    assert (x3[:, col.HARVEST.value] == oi["last_harvest"]).all()
    assert x3[:2, col.CARE.value].tolist() == [8, 0]  # the previous step's kin slots, not the current ones
    assert (x3 >= 0).all()
    assert np_(care_out(mx.array(kin_prev), make_acts(acts).share_out))[:2].tolist() == [8, 0]


def test_speed_unit_without_thrust(micro_cfg: Config) -> None:
    assert speed_unit(micro_cfg) == micro_cfg.physics.thrust_max / micro_cfg.physics.drag
    nothrust = apply_overrides(micro_cfg, ["physics.thrust_max=0.0"])
    assert speed_unit(nothrust) == nothrust.physics.v_max


def test_update_outputs_do_not_alias_inputs(micro_cfg: Config) -> None:
    cfg, rs = micro_cfg, RULES_V1
    oi, acts = make_oi(living(cfg, 6, last_harvest=50)), make_acts(blank_acts(cfg))
    o = init_obs(oi, cfg)
    mx.eval(*o)
    inputs = [*o, *oi, *acts]
    compiled = mx.compile(lambda o_, oi_, a_: update(o_, oi_, a_, cfg, rs))
    eager = update(o, oi, acts, cfg, rs)
    comp = compiled(o, oi, acts)
    mx.eval(*eager, *comp)
    for name, out in (("eager", eager), ("compiled", comp)):
        assert type(out) is ObsState
        for f, arr in zip(ObsState._fields, out, strict=True):
            assert all(arr is not x for x in inputs), (name, f)  # no output is an input (donation, F1)
            assert arr.dtype == getattr(o, f).dtype and arr.shape == getattr(o, f).shape, (name, f)
    for f in ObsState._fields:  # a compiled update (the runner's mode) agrees with the eager one
        assert np.array_equal(np_(getattr(eager, f)), np_(getattr(comp, f))), f
    # the same for the evaluation step, compiled
    ev = mx.compile(lambda o_, oi_, a_: evaluate(update(o_, oi_, a_, cfg, rs), cfg, rs))
    e1, e2 = ev(o, oi, acts), evaluate(update(o, oi, acts, cfg, rs), cfg, rs)
    mx.eval(*e1, *e2)
    for f in ObsState._fields:
        assert np.array_equal(np_(getattr(e1, f)), np_(getattr(e2, f))), f


# --- real steps ---------------------------------------------------------------------------------------------

IMMIGRATION = ["population.immigration_every=5", "population.immigration_count=8"]
# founders' output biases that make bonds, shares, births and child bonds happen (the affordance tests' BIAS)
BIAS = {
    layout.OUT_HARVEST: 2.0,
    layout.OUT_BOND: 2.0,
    **{layout.OUT_SHARE + b: 1.0 for b in range(layout.B)},
    layout.OUT_REPRODUCE: 2.0,
    layout.OUT_CHILD_BOND: 2.0,
}


@pytest.mark.parametrize("dims_", ["2d", "3d"])
def test_observer_invariants_on_real_steps(dims_: str, tiny_cfg: Config, tiny3d_cfg: Config) -> None:
    cfg = apply_overrides(tiny_cfg if dims_ == "2d" else tiny3d_cfg, IMMIGRATION)
    rs = RULES_V1
    s = set_output_bias(init_state(cfg), cfg, BIAS)
    fn = make_step(cfg, observe=True)
    o = init_obs(obs_in(s), cfg)
    mx.eval(*o)
    seen = dict.fromkeys(("reset_old", "replaced", "kin", "flows", "imm", "births", "both", "bonded"), 0)
    for t in range(80):
        prev_s, prev_o = s, o
        s, out = fn(prev_s)
        o = update(prev_o, obs_in(s), out.acts, cfg, rs)
        if int(np_(s.step)) % rs.every == 0:
            o = evaluate(o, cfg, rs)
        mx.eval(s, out.base(), *o)  # one eval per step, as the runner does
        where = f"{dims_} step {t}"
        alive, uid = np_(s.alive), np_(s.uid)
        cur, pre = np_state(o), np_state(prev_o)
        # (a) reset rows are zero; the others count one more step
        reset = ~alive | (uid != pre["uid"])
        assert (cur["n_obs"][reset] == 0).all(), where
        assert (cur["n_obs"][~reset] == np.minimum(pre["n_obs"][~reset] + 1, OBS_CAP)).all(), where
        for f in ("arch", "arch_slot", "intent", "label_arch", "match_arch", "label_intent", "match_intent"):
            assert not cur[f][reset].any(), (where, f)
        assert (cur["cand"][reset] == 0).all() and (cur["cand_n"][reset] == 0).all(), where
        assert (cur["since_bud"][reset] == OBS_CAP).all() and (cur["uid"] == np.where(alive, uid, -1)).all()
        seen["reset_old"] += int((reset & (pre["uid"] >= 0)).sum())
        # (b) a slot whose bond is not the one it had accumulates from scratch, with age 0 or 1
        su, psu = cur["slot_uid"], pre["slot_uid"]
        kept = (~reset)[:, None] & (su == psu) & (su >= 0)
        assert (cur["slot_age"][kept] == np.minimum(pre["slot_age"][kept] + 1, OBS_CAP)).all(), where
        assert not cur["arch_slot"][~kept].any(), where
        assert (cur["slot_age"][~kept] == (su[~kept] >= 0)).all(), where
        seen["replaced"] += int(((su >= 0) & (psu >= 0) & (su != psu)).sum())
        seen["bonded"] += int((su >= 0).sum())
        seen["kin"] += int(cur["slot_kin"].sum())
        # (c) flows occur only on slots whose bond existed at the start of the step, with that partner's uid
        p0, u0 = np_(prev_s.partner), np_(prev_s.uid)
        assert (psu == np.where(p0 >= 0, u0[np.maximum(p0, 0)], -1)).all(), where
        flows = (np_(out.acts.share_out) != 0) | (np_(out.acts.share_got) != 0)
        assert (psu[flows] >= 0).all(), where
        seen["flows"] += int(flows.sum())
        # (d) every birth marks exactly one distinct parent, immigrants none
        c = np_(out.counters)
        budded = int(((np_(out.acts.flags) & ActFlag.BUDDED.value) > 0).sum())
        assert budded == c[C.BIRTHS], where
        seen["imm"] += int(c[C.IMMIGRANTS] > 0)
        seen["births"] += int(c[C.BIRTHS] > 0)
        seen["both"] += int(c[C.IMMIGRANTS] > 0 and c[C.BIRTHS] > 0)
    # the scenario exercised what the invariants talk about
    assert all(seen[k] > 0 for k in seen), seen


def test_same_step_row_reuse_resets_the_observer(micro_cfg: Config) -> None:
    """Row 0 starves at S14 and is the first non-survivor row, so an immigrant lands on it at S16 in the same
    step (lanes fill ``~survivor`` rows in index order): alive both before and after, with a new uid."""
    cfg = apply_overrides(micro_cfg, ["population.immigration_every=1", "population.immigration_count=1"])
    rs = RULES_V1
    fn = make_step(cfg, observe=True)
    s = init_state(cfg)
    o = init_obs(obs_in(s), cfg)
    for _ in range(3):  # row 0 is observed for a while first
        s, out = fn(s)
        o = update(o, obs_in(s), out.acts, cfg, rs)
        mx.eval(s, out.base(), *o)
    assert bool(np_(s.alive)[0]) and int(np_(o.n_obs)[0]) == 3
    old_uid = int(np_(s.uid)[0])
    energy = np_(s.energy).copy()
    energy[0] = 0
    s = s._replace(energy=mx.array(energy))
    s, out = fn(s)
    o = update(o, obs_in(s), out.acts, cfg, rs)
    mx.eval(s, out.base(), *o)
    assert bool(np_(s.alive)[0]) and int(np_(s.uid)[0]) != old_uid  # reused in the step it died
    assert np_(out.counters)[C.DEATHS_STARVED] >= 1 and np_(out.counters)[C.IMMIGRANTS] >= 1
    cur = np_state(o)
    assert cur["uid"][0] == int(np_(s.uid)[0]) and cur["n_obs"][0] == 0
    assert not cur["arch"][0].any() and not cur["intent"][0].any() and cur["since_bud"][0] == OBS_CAP
    assert (cur["n_obs"][1:8] == 4).all()  # the other founders carry on
