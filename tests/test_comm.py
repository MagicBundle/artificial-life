"""Communication, C1 (DESIGN §39): the layout and config of the voice channel, ``hear``, the decode, the
dithered cost and the ledger (E17), the controls (E19), the hash salts (E18), the counters (E10), the step
(determinism, resume, observer purity) and the 2D and 3D worlds. Hand-built scenes on the CPU."""

from __future__ import annotations

import itertools
import re
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

import helpers
from alife import comm, config, ext, invariants, layout, metrics, rng, spatial
from alife import energy as en
from alife.config import ConfigError, config_hash, dims, load_config
from alife.runio import load_checkpoint, write_state_file
from alife.state import (
    CounterIdx,
    Diag,
    LedgerIdx,
    counter_blocks,
    counter_index,
    init_state,
    n_counters,
    state_digest,
    to_numpy,
)
from alife.step import ActFlag, make_step

PRESET = {2: "tiny", 3: "tiny3d"}
F32 = "brain.weight_dtype=float32"


def cfg_for(nd: int, *extra: str, width: int = 4):
    return load_config(preset=PRESET[nd], overrides=[F32, "comm.enabled=true", f"comm.width={width}", *extra])


def micro(nd: int, *extra: str, width: int = 4):
    base = (
        "population.n_max=64",
        "population.max_births_per_step=8",
        "population.n_init=8",
        "world.size=48",
        F32,
        "comm.enabled=true",
        f"comm.width={width}",
    )
    return load_config(preset=PRESET[nd], overrides=[*base, *extra])


# --- layout, config, registry -----------------------------------------------------------------------------


@pytest.mark.parametrize("nd", [2, 3])
@pytest.mark.parametrize("width", [1, 4, 8])
def test_the_footprint_is_w_plus_nd_inputs_and_w_outputs(nd: int, width: int) -> None:
    base = dims(load_config(preset=PRESET[nd])).lay
    d = dims(cfg_for(nd, width=width))
    lay = d.lay
    assert (base.D_IN + width + nd, base.D_OUT + width) == (lay.D_IN, lay.D_OUT)
    assert (
        base.D_IN,
        base.D_IN + width,
        base.D_OUT,
    ) == (lay.IN_VOICE, lay.IN_VOICE_BEARING, lay.OUT_VOICE)
    assert (d.voice_w, d.msg_w, d.ext("comm").hear_range) == (
        width,
        layout.MSG + width,
        cfg_for(nd).sensing.radius,
    )
    assert base.IN_VOICE is None and base.OUT_VOICE is None  # None when off, never -1
    # the blocks tile the vectors
    spans = sorted(b[1:] for b in lay.INPUT_BLOCKS)
    assert spans[0][0] == 0 and all(a[0] + a[1] == b[0] for a, b in itertools.pairwise(spans))
    assert spans[-1][0] + spans[-1][1] == lay.D_IN
    assert len(set(lay.INPUT_NAMES)) == lay.D_IN and len(set(lay.OUTPUT_NAMES)) == lay.D_OUT
    s = init_state(cfg_for(nd, width=width))
    assert s.msg.shape == (d.n, layout.MSG + width) and s.W1.shape[1] == d.lay.D_IN + d.h
    assert s.W2.shape[2] == lay.D_OUT


def test_the_footprint_of_the_spec() -> None:
    assert (dims(cfg_for(2)).lay.D_IN, dims(cfg_for(2)).lay.D_OUT) == (75, 39)
    assert (dims(cfg_for(3)).lay.D_IN, dims(cfg_for(3)).lay.D_OUT) == (91, 43)
    assert (
        ext.layout_features(cfg_for(2)) == "comm4@1" and ext.layout_features(cfg_for(2, width=2)) == "comm2@1"
    )
    assert ActFlag.SPOKE.value == 1024 == 128 << ext.EXTENSIONS.index("comm")


def test_off_is_identical() -> None:
    """With ``comm.enabled`` false nothing is allocated, hashed or serialised."""
    for preset in config.PRESETS:
        cfg = load_config(preset=preset)
        assert "comm" not in config.config_to_dict(cfg, omit_defaults=True)
        d = dims(cfg)
        assert (d.voice_w, d.msg_w, d.features) == (0, layout.MSG, ())
    cfg = load_config(preset="tiny", overrides=[F32])
    assert counter_blocks(cfg) == () and init_state(cfg).msg.shape[1] == layout.MSG
    _, out = make_step(cfg)(init_state(cfg))
    assert to_numpy(out.counters).shape == (n_counters(2, cfg.weather.enabled),)
    assert config_hash(cfg) != config_hash(cfg_for(2)) and ext.active(cfg) == ()


@pytest.mark.parametrize(
    ("over", "msg"),
    [
        (["comm.width=6"], "inert while comm.enabled is false"),
        (["comm.cost=5"], "inert while comm.enabled is false"),
        (["comm.control=deaf"], "inert while comm.enabled is false"),
        (["comm.kin_exp=1"], "inert while comm.enabled is false"),
        (["comm.enabled=true", "comm.width=0"], "comm.width"),
        (["comm.enabled=true", "comm.width=9"], "comm.width"),
        (["comm.enabled=true", "comm.range=9"], "comm.range"),
        (["comm.enabled=true", "comm.range=-1"], "comm.range"),
        (["comm.enabled=true", "comm.atten_pow=0"], "comm.atten_pow"),
        (["comm.enabled=true", "comm.kin_exp=-0.5"], "comm.kin_exp"),
        (["comm.enabled=true", "comm.cost=-1"], "comm.cost"),
        (["comm.enabled=true", "comm.cost=9000"], "per-row due"),
        (["comm.enabled=true", "comm.cost=4000", "comm.width=4", "population.n_max=262144"], "COMM_DUE"),
        (["comm.enabled=true", "comm.control=loud"], "comm.control"),
    ],
)
def test_validation(over: list[str], msg: str) -> None:
    with pytest.raises(ConfigError, match=msg):
        load_config(preset="tiny", overrides=over)


def test_every_key_changes_the_hash_and_the_overlay_composes() -> None:
    base = cfg_for(2)
    for k, v in (
        ("width", 3),
        ("range", 4.0),
        ("atten_pow", 2.0),
        ("kin_exp", 1.0),
        ("cost", 16.0),
        ("control", "deaf"),
    ):
        other = load_config(preset="tiny", overrides=[F32, "comm.enabled=true", f"comm.{k}={v}"])
        assert config_hash(other) != config_hash(base), k
    keys = set(config.FEATURE_KEYS["comm"])
    assert keys == {
        ("comm", f) for f in ("enabled", "width", "range", "atten_pow", "kin_exp", "cost", "control")
    }
    assert keys <= set(config.POST_V01_KEYS) and keys <= set(config.SERIAL_OMIT_KEYS)
    assert ("comm", "control") in config.CLI_ONLY_FEATURE_KEYS["comm"]
    ov = load_config(preset="tiny3d", overlays=["comm"])
    assert ov.comm.enabled and "comm" in config.WORLD_OVERLAYS and ext.active(ov)[0].name == "comm"


# --- hear -------------------------------------------------------------------------------------------------


def scene(cfg, spec: dict[int, list[tuple[int, tuple[float, ...], float]]]):
    """``(nbr, desc, n)`` with ``spec[i] = [(j, offset (nd,), kin), ...]``: neighbour ``j`` of row ``i`` at
    the body-frame offset ``offset`` (the body frame is the world frame here)."""
    d, R = dims(cfg), cfg.sensing.radius
    n, K = d.n, d.k
    idx = np.tile(np.arange(n, dtype=np.int32)[:, None], (1, K))
    valid = np.zeros((n, K), bool)
    off = np.zeros((n, K, d.nd), np.float32)
    dist = np.full((n, K), R, np.float32)
    desc = np.zeros((n, K, d.lay.NF), np.float32)
    for i, entries in spec.items():
        for k, (j, o, kin) in enumerate(entries):
            idx[i, k], valid[i, k], off[i, k] = j, True, o
            dist[i, k] = float(np.linalg.norm(o))
            desc[i, k, layout.NBR_PRESENT] = 1.0
            desc[i, k, layout.NBR_DX] = o[0] / R
            desc[i, k, layout.NBR_DY] = o[1] / R
            desc[i, k, layout.NBR_KIN] = kin
            if d.d3:
                desc[i, k, d.lay.NBR_DZ] = o[2] / R
    z = mx.zeros((), mx.int32)
    nbr = spatial.Neighbors(mx.array(idx), mx.array(valid), mx.array(off), mx.array(dist), z, z)
    return nbr, mx.array(desc), n


def voice_of(cfg, rows: dict[int, list[float]]) -> mx.array:
    d = dims(cfg)
    v = np.zeros((d.n, d.voice_w), np.float32)
    for r, vals in rows.items():
        v[r, : len(vals)] = vals
    return mx.array(v)


def off_(nd: int, *xyz: float) -> tuple[float, ...]:
    return (*xyz, 0.0, 0.0, 0.0)[:nd]


@pytest.mark.parametrize("nd", [2, 3])
def test_hear_range_validity_and_the_linear_falloff(nd: int) -> None:
    cfg = micro(nd)
    R = cfg.sensing.radius
    nbr, desc, _ = scene(
        cfg,
        {
            0: [(1, off_(nd, 0.25 * R), 1.0)],  # a quarter of the way: weight 0.75
            2: [(3, off_(nd, R), 1.0)],  # at the range: exactly 0
            4: [(5, off_(nd, 0.5 * R), 1.0), (6, off_(nd, 0.0, 0.5 * R), 1.0)],  # two sources add
        },
    )
    voice = voice_of(cfg, {1: [1.0, 0.4], 3: [1.0], 5: [0.4, 0.0, 0.2], 6: [0.2, 0.0, 0.2]})
    h = to_numpy(comm.hear(voice, nbr, desc, cfg))
    assert np.allclose(h[0, :2], [0.75, 0.3]) and not h[0, 2:4].any()
    assert not h[2].any()  # at dist >= range: exactly nothing, bearing included
    assert np.allclose(h[4, :4], [0.5 * 0.4 + 0.5 * 0.2, 0, 0.5 * 0.2 + 0.5 * 0.2, 0])
    assert not h[1].any() and not h[3].any()  # rows with no neighbour hear nothing
    # an invalid entry pointing at a loud voice contributes exactly 0
    nbr2, desc2, _ = scene(cfg, {})
    idx = np.array(nbr2.idx)
    idx[7, 0] = 1  # invalid slot, but its index names a loud row
    nbr2 = nbr2._replace(idx=mx.array(idx))
    assert not to_numpy(comm.hear(voice, nbr2, desc2, cfg))[7].any()


def test_hear_range_key_narrows_the_hearing() -> None:
    cfg = micro(2, "comm.range=4.0")
    nbr, desc, _ = scene(cfg, {0: [(1, (2.0, 0.0), 1.0)], 2: [(3, (5.0, 0.0), 1.0)]})
    voice = voice_of(cfg, {1: [1.0], 3: [1.0]})
    h = to_numpy(comm.hear(voice, nbr, desc, cfg))
    assert dims(cfg).ext("comm").hear_range == 4.0
    assert np.isclose(h[0, 0], 0.5) and not h[2].any()  # 5 > 4: out of range though inside the sensing radius


@pytest.mark.parametrize("kin_exp", [0.0, 1.0, 2.0])
def test_hear_kin_exponent(kin_exp: float) -> None:
    cfg = micro(2, f"comm.kin_exp={kin_exp}")
    R = cfg.sensing.radius
    nbr, desc, _ = scene(cfg, {0: [(1, (0.5 * R, 0.0), 0.5)]})
    h = to_numpy(comm.hear(voice_of(cfg, {1: [1.0]}), nbr, desc, cfg))
    assert np.isclose(h[0, 0], 0.5 * 0.5**kin_exp)  # distance weight 0.5, kin 0.5 ** kin_exp


def test_hear_atten_pow() -> None:
    R = micro(2).sensing.radius
    for p in (1.0, 2.0):
        cfg = micro(2, f"comm.atten_pow={p}")
        nbr, desc, _ = scene(cfg, {0: [(1, (0.5 * R, 0.0), 1.0)]})
        h = to_numpy(comm.hear(voice_of(cfg, {1: [1.0]}), nbr, desc, cfg))
        assert np.isclose(h[0, 0], 0.5**p)


@pytest.mark.parametrize("nd", [2, 3])
def test_hear_bearing_sign_ahead_left_above(nd: int) -> None:
    cfg = micro(nd)
    R = cfg.sensing.radius
    nbr, desc, _ = scene(
        cfg,
        {
            0: [(1, off_(nd, 0.5 * R), 1.0)],
            2: [(3, off_(nd, 0.0, 0.5 * R), 1.0)],
            4: [(5, off_(nd, -0.5 * R), 1.0)],
            **({6: [(7, (0.0, 0.0, 0.5 * R), 1.0)]} if nd == 3 else {}),
        },
    )
    voice = voice_of(cfg, {1: [1.0], 3: [1.0], 5: [1.0], 7: [1.0]})
    h = to_numpy(comm.hear(voice, nbr, desc, cfg))
    w = dims(cfg).voice_w
    assert np.allclose(h[0, w:], [0.5, 0, 0][:nd]) and np.allclose(h[2, w:], [0, 0.5, 0][:nd])
    assert h[4, w] < 0 and np.isclose(h[4, w], -0.5)
    if nd == 3:
        assert np.allclose(h[6, w:], [0, 0, 0.5])
    # the bearing is a weighted mean: a faint source on the other side moves it only by its loudness
    nbr, desc, _ = scene(cfg, {0: [(1, off_(nd, 0.5 * R), 1.0), (2, off_(nd, -0.5 * R), 1.0)]})
    h = to_numpy(comm.hear(voice_of(cfg, {1: [1.0], 2: [0.25]}), nbr, desc, cfg))
    assert np.isclose(h[0, w], (1.0 * 0.5 * 0.5 + 0.25 * 0.5 * -0.5) / (0.5 + 0.125))


@pytest.mark.parametrize("nd", [2, 3])
def test_hear_bounds_silence_and_row_equivariance(nd: int) -> None:
    cfg = micro(nd)
    d = dims(cfg)
    R = cfg.sensing.radius
    rs = np.random.default_rng(5)
    spec = {}
    for i in range(d.n):
        k = int(rs.integers(0, d.k + 1))
        js = rs.choice(d.n, k, replace=False)
        spec[i] = [(int(j), tuple(rs.uniform(-R, R, nd) * 0.9), float(rs.random())) for j in js if j != i]
    nbr, desc, _ = scene(cfg, spec)
    voice = mx.array(rs.random((d.n, d.voice_w)).astype(np.float32) * (rs.random((d.n, 1)) < 0.7))
    h = to_numpy(comm.hear(voice, nbr, desc, cfg))
    assert (
        (h[:, : d.voice_w] >= 0).all()
        and (h[:, : d.voice_w] <= 1).all()
        and (np.abs(h[:, d.voice_w :]) <= 1).all()
    )
    assert not to_numpy(comm.hear(mx.zeros_like(voice), nbr, desc, cfg)).any()  # silence: exact zeros
    perm = rs.permutation(d.n)
    inv = np.argsort(perm)
    nbr2 = nbr._replace(
        idx=mx.array(inv[np.array(nbr.idx)[perm]].astype(np.int32)),
        valid=mx.array(np.array(nbr.valid)[perm]),
        off=mx.array(np.array(nbr.off)[perm]),
        dist=mx.array(np.array(nbr.dist)[perm]),
    )
    h2 = to_numpy(comm.hear(mx.array(np.array(voice)[perm]), nbr2, mx.array(np.array(desc)[perm]), cfg))
    assert np.array_equal(h2, h[perm])


@pytest.mark.parametrize("kin_exp", [0.0, 1.0, 2.0])
def test_hear_is_a_saturating_superposition(kin_exp: float) -> None:
    cfg = micro(2, f"comm.kin_exp={kin_exp}")
    nbr, desc, _ = scene(cfg, {0: [(j, (0.01 * j, 0.0), 1.0) for j in range(1, 9)]})
    h = to_numpy(comm.hear(voice_of(cfg, {j: [1.0] for j in range(1, 9)}), nbr, desc, cfg))
    assert h[0, 0] == 1.0  # eight near sources saturate at 1


# --- decode, mask, controls -------------------------------------------------------------------------------


@pytest.mark.parametrize("nd", [2, 3])
def test_decode_range_zero_and_masks(nd: int) -> None:
    cfg = micro(nd)
    d, lay = dims(cfg), dims(cfg).lay
    y = np.zeros((d.n, lay.D_OUT), np.float32)
    y[:, lay.OUT_VOICE : lay.OUT_VOICE + 4] = [-5.0, 0.0, 0.3, 50.0]
    alive = np.zeros(d.n, bool)
    alive[:4] = True
    s = helpers.make_scene(cfg, [{} for _ in range(4)])
    dec = comm.decode(s, mx.array(y), None, cfg)
    sig = to_numpy(dec.sig)
    assert np.allclose(sig[0], [0.0, 0.0, np.tanh(0.3), comm.SIG_MAX]) and sig[0, 3] < 1.0
    assert (sig[0] == 0).sum() == 2 and not sig[4:].any()  # exact zero reachable; dead rows silent
    solvent = mx.array(np.array([True, False, True, True] + [False] * (d.n - 4)))
    m = comm.mask(dec, solvent, cfg)
    assert (
        not to_numpy(m.sig)[1].any()
        and to_numpy(m.due)[1] == 0
        and np.array_equal(to_numpy(m.sig)[0], sig[0])
    )


# --- E17: the dithered cost -------------------------------------------------------------------------------


def test_e17_a_sub_unit_call_costs_its_value_on_average() -> None:
    cfg = micro(2)
    n = 200_000
    uid = mx.arange(n, dtype=mx.int32)
    for per_channel, step in ((0.05, 7), (0.1, 1234), (0.011, 99999)):
        sig = mx.full((n, 4), per_channel, mx.float32)
        due = to_numpy(comm.due_of(uid, mx.array(step, dtype=mx.int32), sig, cfg)).astype(np.float64)
        want = 4.0 * 4 * per_channel
        assert abs(due.mean() - want) < 0.01 * max(want, 1) + 0.005, (per_channel, due.mean(), want)
        assert set(np.unique(due)) <= {np.floor(want), np.floor(want) + 1}  # floor or floor + 1
        assert (due > 0).any()  # a whisper is paid for on average, not free
    silent = comm.due_of(uid, mx.array(3, dtype=mx.int32), mx.zeros((n, 4), mx.float32), cfg)
    assert not to_numpy(silent).any()  # silence is exactly free
    free = micro(2, "comm.cost=0.0")
    assert not to_numpy(comm.due_of(uid, mx.array(3, dtype=mx.int32), mx.ones((n, 4)) * 0.9, free)).any()
    # no draw is taken: the same (uid, step) always gives the same due, another step another dither
    a = comm.dither(uid, mx.array(5, dtype=mx.int32))
    assert np.array_equal(to_numpy(a), to_numpy(comm.dither(uid, mx.array(5, dtype=mx.int32))))
    assert not np.array_equal(to_numpy(a), to_numpy(comm.dither(uid, mx.array(6, dtype=mx.int32))))


def test_e17_the_due_hook_is_the_decoded_due_for_a_solvent_row() -> None:
    cfg = micro(2)
    d, lay = dims(cfg), dims(cfg).lay
    rs = np.random.default_rng(1)
    y = mx.array(rs.normal(0, 1, (d.n, lay.D_OUT)).astype(np.float32))
    s = helpers.make_scene(cfg, [{} for _ in range(40)])
    dec = comm.decode(s, y, None, cfg)
    hook = comm.due(s, None, y, cfg)
    assert np.array_equal(to_numpy(hook), to_numpy(dec.due)) and to_numpy(hook)[40:].sum() == 0
    assert comm.due(s, None, y, micro(2, "comm.cost=0.0")) is None


@pytest.mark.parametrize("nd", [2, 3])
def test_pay_costs_books_the_voice_in_heat_action_and_solvency(nd: int) -> None:
    cfg = micro(nd)
    d = dims(cfg)
    from test_energy import _actions

    due = mx.array(np.arange(d.n, dtype=np.int32) % 9)
    e = mx.array(np.where(np.arange(d.n) % 5 == 0, 3, 1000).astype(np.int32))
    alive = mx.ones((d.n,), mx.bool_)
    z = mx.zeros((d.n,), mx.int32)
    c0 = en.pay_costs(e, _actions(d.n), z, z, alive, cfg)
    c1 = en.pay_costs(e, _actions(d.n), z, z, alive, cfg, due)
    ee = np.array(e)
    want = np.minimum(np.array(due), ee - np.array(c0.paid_basal))
    assert np.array_equal(np.array(c1.paid_action - c0.paid_action), want)
    # E_in - E_out = paid_basal + paid_action, and an unaffordable call is not made
    assert np.array_equal(np.array(c1.energy), ee - np.array(c1.paid_basal) - np.array(c1.paid_action))
    solvent = np.array(c1.solvent)
    assert not solvent[np.array(due) + np.array(c0.paid_basal) > ee].any()


# --- the step ---------------------------------------------------------------------------------------------


def _run(cfg, steps, s=None, *, observe=False):
    fn = make_step(cfg, observe=observe)
    s = init_state(cfg) if s is None else s
    for _ in range(steps):
        prev = s
        s, out = fn(s)
        mx.eval(s, out)
        yield prev, s, out


@pytest.mark.parametrize("nd", [2, 3])
@pytest.mark.parametrize("cost", [0.0, 4.0, 64.0])
def test_the_ledger_closes_and_the_counters_are_sound(nd: int, cost: float) -> None:
    """No 17th term: the voice is HEAT_ACTION. 200 steps conserve TOTAL; COMM_DUE <= HEAT_ACTION."""
    cfg = cfg_for(nd, f"comm.cost={cost}")
    ci = counter_index(cfg, "COMM_DUE")
    cs = counter_index(cfg, "COMM_SPEAKERS")
    spk = due = 0
    for t, (prev, s, out) in enumerate(_run(cfg, 200)):
        led, c = to_numpy(out.ledger), to_numpy(out.counters)
        assert led.shape == (16,) and helpers.ledger_net(led, cfg) == int(
            to_numpy(s.total) - to_numpy(prev.total)
        )
        assert c[ci] <= led[LedgerIdx.HEAT_ACTION] and 0 <= c[cs] <= int(to_numpy(s.alive).sum()) + 64
        assert int(to_numpy(s.diag)) & 0xFFFF == 0, (t, Diag(int(to_numpy(s.diag))))
        spk, due = spk + int(c[cs]), due + int(c[ci])
        if t % 50 == 0:
            assert invariants.full_check(s, cfg, prev, out) == [], t
    assert spk > 0  # the channel is alive at the founders' random call
    assert (due == 0) == (cost == 0.0)  # cost 0 pays nothing; otherwise the speakers pay


def test_comm_due_equals_the_action_due_in_a_one_term_scene() -> None:
    """Zero weights but one bias: every alive row says the same call and does nothing else, so the voice is
    the only action cost and COMM_DUE equals HEAT_ACTION exactly, with a bias that makes every row speak."""
    cfg = micro(2, "comm.cost=8.0")
    d, lay = dims(cfg), dims(cfg).lay
    w2 = np.zeros((d.w2_rows, lay.D_OUT), np.float32)
    w2[d.h, lay.OUT_VOICE : lay.OUT_VOICE + 4] = [0.5, 0.0, 0.0, 0.2]
    # act on nothing else: the other outputs' biases are 0 (thrust 0, harvest sigmoid(0) = 0.5 costs: set the
    # harvest/defend/grip costs aside by zeroing them in the config)
    cfg = micro(
        2,
        "comm.cost=8.0",
        "energy.harvest_cost=0",
        "energy.defend_cost=0",
        "energy.grip_cost=0",
        "energy.msg_cost=0",
        "energy.emit_cost=0",
        "energy.move_cost=0",
        "energy.turn_cost=0",
        "energy.contract_cost=0",
        "energy.attack_cost=0",
    )
    s = helpers.make_scene(
        cfg, [{"W2": w2, "energy": 5000, "pos": [4.0 + 5 * i, 4.0 + 5 * (i // 8) * 3]} for i in range(8)]
    )
    s1, out = make_step(cfg, compile=False)(s)
    mx.eval(s1, out)
    led, c = to_numpy(out.ledger), to_numpy(out.counters)
    assert c[counter_index(cfg, "COMM_SPEAKERS")] == 8
    assert c[counter_index(cfg, "COMM_DUE")] == led[LedgerIdx.HEAT_ACTION] > 0
    assert np.allclose(to_numpy(s1.msg)[:8, 2:], [np.tanh(0.5), 0, 0, np.tanh(0.2)])


@pytest.mark.parametrize("nd", [2, 3])
def test_an_insolvent_speaker_pays_what_it_holds_and_says_nothing(nd: int) -> None:
    cfg = micro(nd, "comm.cost=64.0")
    d, lay = dims(cfg), dims(cfg).lay
    w2 = np.zeros((d.w2_rows, lay.D_OUT), np.float32)
    w2[d.h, lay.OUT_VOICE : lay.OUT_VOICE + 4] = 3.0  # a full chorus: due ~ 256
    s = helpers.make_scene(
        cfg,
        [
            {"W2": w2, "energy": 150, "pos": [5, 5, 5][:nd]},
            {"W2": w2, "energy": 9000, "pos": [30, 30, 5][:nd]},
        ],
    )
    s1, out = make_step(cfg, compile=False)(s)
    mx.eval(s1, out)
    msg = to_numpy(s1.msg)
    assert not msg[0, 2:].any() and msg[1, 2:].all()  # the poor row cannot afford the call
    assert helpers.ledger_net(to_numpy(out.ledger), cfg) == int(to_numpy(s1.total) - to_numpy(s.total))
    assert to_numpy(s1.alive)[0] == (to_numpy(s1.energy)[0] > 0) or not to_numpy(s1.alive)[0]


def test_newborns_start_silent_and_dead_rows_hold_nothing() -> None:
    cfg = cfg_for(2)
    born = 0
    for _, s, out in _run(cfg, 60):
        msg, alive = to_numpy(s.msg), to_numpy(s.alive)
        assert not msg[~alive].any() and msg[:, 2:].max() < 1.0 and msg[:, 2:].min() >= 0.0
        slot, valid = np.asarray(out.births.slot), np.asarray(out.births.valid)
        born += int(valid.sum())
        assert not msg[slot[valid]].any()  # a newborn is silent: calls are not inherited
    assert born > 0


def test_the_invariant_checks_the_voice_columns() -> None:
    cfg = cfg_for(2)
    s = init_state(cfg)
    assert invariants.full_check(s, cfg) == []
    msg = np.array(s.msg)
    msg[0, 3] = 1.0  # a voice of exactly 1: outside [0, 1)
    bad = s._replace(msg=mx.array(msg))
    assert any("voice outside [0, 1)" in p for p in invariants.full_check(bad, cfg))
    msg[0, 3], msg[0, 0] = 0.5, 1.5  # the legacy range check still reads the first two columns
    assert any("msg: outside" in p for p in invariants.full_check(s._replace(msg=mx.array(msg)), cfg))


@pytest.mark.parametrize("nd", [2, 3])
@pytest.mark.parametrize("control", ["none", "deaf", "decoy"])
def test_determinism_resume_and_observer_purity(nd: int, control: str, tmp_path: Path) -> None:
    cfg = cfg_for(nd, f"comm.control={control}")
    last = lambda g: list(g)[-1][1]  # noqa: E731
    a = last(_run(cfg, 40, init_state(cfg)))
    assert state_digest(a) == state_digest(last(_run(cfg, 40, init_state(cfg))))
    half = last(_run(cfg, 20, init_state(cfg)))
    path = tmp_path / "half.safetensors"
    write_state_file(path, half, cfg)
    assert state_digest(last(_run(cfg, 20, load_checkpoint(path, cfg)))) == state_digest(a)
    plain, obs = list(_run(cfg, 12)), list(_run(cfg, 12, observe=True))
    assert state_digest(plain[-1][1]) == state_digest(obs[-1][1])
    for (_, _, o1), (_, _, o2) in zip(plain, obs, strict=True):
        assert np.array_equal(to_numpy(o1.counters), to_numpy(o2.counters))
    # the SPOKE flag is the per-row form of COMM_SPEAKERS
    assert all(to_numpy(o.acts.flags).dtype == np.uint16 for _, _, o in obs)
    spoken = [int(((to_numpy(o.acts.flags) & ActFlag.SPOKE.value) > 0).sum()) for _, _, o in obs]
    counted = [int(to_numpy(o.counters)[counter_index(cfg, "COMM_SPEAKERS")]) for _, _, o in obs]
    assert spoken == counted and sum(spoken) > 0


@pytest.mark.parametrize("nd", [2, 3])
def test_compiled_and_eager_steps_agree(nd: int) -> None:
    """As ``test_determinism``: one step agrees on every counter, the ledger and every integer field; floats
    to 1e-5 (bitwise equality is not promised, §14.2)."""
    cfg = cfg_for(nd)
    s = init_state(cfg)
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
    assert to_numpy(oa.counters)[counter_index(cfg, "COMM_SPEAKERS")] > 0


# --- E19: the controls leak nothing -----------------------------------------------------------------------


def test_e19_decoy_is_a_bijection_that_never_hears_itself_and_keeps_the_statistics() -> None:
    n = 64
    seen = set()
    for step in [*range(300), 10**6, 2**31 - 5]:
        idx = to_numpy(comm.decoy_index(mx.array(step, dtype=mx.int32), n))
        assert sorted(idx.tolist()) == list(range(n))  # a bijection: the marginal voice statistics are kept
        assert (idx != np.arange(n)).all()  # nobody is its own decoy
        seen.add(int(idx[0]))
    assert len(seen) > 20  # the shift moves from step to step
    v = np.random.default_rng(0).random((n, 4)).astype(np.float32)
    assert sorted(v[idx].ravel().tolist()) == sorted(v.ravel().tolist())


@pytest.mark.parametrize("nd", [2, 3])
def test_e19_deaf_and_decoy_change_only_the_heard_inputs(nd: int) -> None:
    """Zero the W1 rows of the voice inputs: the brain then ignores what it hears, so the three controls run
    bit-identical steps (same costs, same calls, same motion): the controls act only through the inputs."""
    base = cfg_for(nd, "repro.min_age=100000")  # no births: a child's mutated rows would wake the inputs
    d = dims(base)
    s = init_state(base)
    w1 = np.array(s.W1)
    lo = d.lay.IN_VOICE
    w1[:, lo : lo + d.voice_w + nd] = 0.0
    s = s._replace(W1=mx.array(w1).astype(s.W1.dtype))
    s = s._replace(brain_cost=__import__("alife.brain", fromlist=["brain"]).l1_cost(s.W1, s.W2, base))
    mx.eval(s)
    digests = {}
    for control in ("none", "deaf", "decoy"):
        cfg = cfg_for(nd, "repro.min_age=100000", f"comm.control={control}")
        cur, hist = s, []
        fn = make_step(cfg)
        for _ in range(25):
            cur, out = fn(cur)
            mx.eval(cur, out)
            hist.append(to_numpy(out.counters).tobytes() + to_numpy(out.ledger).tobytes())
        digests[control] = (state_digest(cur), tuple(hist))
    assert digests["none"] == digests["deaf"] == digests["decoy"]


@pytest.mark.parametrize("nd", [2, 3])
def test_e19_the_controls_differ_when_the_brain_listens(nd: int) -> None:
    out = {}
    for control in ("none", "deaf", "decoy"):
        cfg = cfg_for(nd, f"comm.control={control}")
        out[control] = state_digest(list(_run(cfg, 40))[-1][1])
    assert len(set(out.values())) == 3
    # deaf listens to exact zeros
    cfg = cfg_for(nd, "comm.control=deaf")
    s = init_state(cfg)
    nbr, desc, _ = scene(cfg, {})
    ctx = ext.ExtCtx(nbr, desc, None, None)
    heard, bearing = comm.ext_inputs(s, cfg, ctx)
    assert not to_numpy(heard).any() and not to_numpy(bearing).any()
    assert heard.shape == (dims(cfg).n, 4) and bearing.shape == (dims(cfg).n, nd)


# --- E18: the hash salt -----------------------------------------------------------------------------------


def test_e18_comm_uses_only_its_own_salt() -> None:
    src = Path(comm.__file__).read_text()
    used = re.findall(r"HASH_SALTS\[([^\]]*)\]", src)
    assert used and set(used) == {'"comm"'}
    assert comm.SALT == rng.HASH_SALTS["comm"] == 0x6E00A046
    # the dither of comm and arms, on the same keys, are independent uniforms
    from alife import arms

    n = 20000
    uid = mx.arange(n, dtype=mx.int32)
    a = to_numpy(comm.dither(uid, mx.array(11, dtype=mx.int32))).astype(np.float64)
    b = to_numpy(arms.dither(uid, mx.array(11, dtype=mx.int32))).astype(np.float64)
    assert abs(np.corrcoef(a, b)[0, 1]) * np.sqrt(n) < 4.5 and abs(a.mean() - 0.5) < 0.01
    # the decoy shift is a hash of the comm salt, not of the neighbours' salts
    shifts = {int(to_numpy(comm.decoy_index(mx.array(t, dtype=mx.int32), 1000))[0]) for t in range(200)}
    assert len(shifts) > 150


# --- E10: counters and metrics ----------------------------------------------------------------------------


def test_e10_counters_and_metrics() -> None:
    cfg = cfg_for(2)
    assert counter_blocks(cfg) == (("comm", ("COMM_SPEAKERS", "COMM_DUE")),)
    assert n_counters(2, cfg.weather.enabled, counter_blocks(cfg)) == n_counters(2, cfg.weather.enabled) + 2
    cs, ls = [], []
    for _, _, out in _run(cfg, 30):
        cs.append(to_numpy(out.counters))
        ls.append(to_numpy(out.ledger))
    row = metrics.interval_row(30, np.stack(cs), np.stack(ls), {}, {}, nd=2, blocks=counter_blocks(cfg))
    assert (
        0.0 < row["comm_speaker_frac"] <= 1.0 and row["comm_due"] > 0 and 0.0 < row["comm_due_share"] <= 1.0
    )
    # the per-step counter is int32; the interval sums are int64 on the host
    assert np.stack(cs).dtype == np.int32 and isinstance(row["comm_due"], int)
    # the validation bound keeps one step's COMM_DUE inside int32 at the largest population
    worst = config.CommConfig(enabled=True, cost=1.0, width=8).cost * 8 * 262144
    assert worst < 2**31


@pytest.mark.parametrize("nd", [2, 3])
@pytest.mark.parametrize("control", ["none", "decoy"])
def test_the_staged_pipeline_equals_the_step_with_comm_on(nd: int, control: str) -> None:
    from dataclasses import replace

    from mlx.utils import tree_flatten

    from alife import bench

    cfg = cfg_for(nd, f"comm.control={control}")
    cfg = replace(cfg, run=replace(cfg.run, compile=False))
    s0 = init_state(cfg)
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
    assert to_numpy(s.msg)[:, layout.MSG :].any()


def test_bench_comm_flag_needs_the_feature() -> None:
    from alife import bench

    with pytest.raises(ConfigError, match="bench --comm"):
        bench.run_bench(load_config(preset="tiny"), steps=1, warmup=1, comm_arm=True)


@pytest.mark.slow
@pytest.mark.parametrize("control", ["none", "decoy"])
def test_smoke_3000_steps_of_tiny3d_the_channel_is_neither_dead_nor_saturated(control: str) -> None:
    """DESIGN §39.11: no invariant error, the population survives, the ledger closes, and the share of
    speakers is neither identically zero nor all (the channel is alive at the default cost)."""
    cfg = cfg_for(3, f"comm.control={control}")
    cs, ci = counter_index(cfg, "COMM_SPEAKERS"), counter_index(cfg, "COMM_DUE")
    alive_i = CounterIdx.N_ALIVE
    fracs, due, heat = [], 0, 0
    for t, (prev, s, out) in enumerate(_run(cfg, 3000)):
        c, led = to_numpy(out.counters), to_numpy(out.ledger)
        fracs.append(c[cs] / max(int(c[alive_i]), 1))
        due, heat = due + int(c[ci]), heat + int(led[LedgerIdx.HEAT_ACTION])
        assert c[ci] <= led[LedgerIdx.HEAT_ACTION] and int(to_numpy(s.diag)) & 0xFFFF == 0, t
        if t % 500 == 499:
            assert invariants.full_check(s, cfg, prev, out) == [], t
    assert int(to_numpy(s.alive).sum()) > 0 and min(fracs) > 0.0 and max(fracs) < 1.0 and due > 0
