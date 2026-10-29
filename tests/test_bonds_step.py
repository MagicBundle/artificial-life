"""Bonds that pay, B1 (DESIGN §40): the layout and config of the feature, the conductive share inside the step
(ledger, sharing active, determinism, resume, observer purity), the state memory and its rows (E15), the
counters and metrics (E10)."""

from __future__ import annotations

import dataclasses
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

import helpers
from alife import bonds, config, ext, invariants, layout, metrics
from alife.config import ConfigError, dims, load_config
from alife.runio import load_checkpoint, write_state_file
from alife.state import (
    Diag,
    LedgerIdx,
    counter_blocks,
    counter_index,
    init_state,
    n_counters,
    state_cls,
    state_digest,
    to_numpy,
)
from alife.step import ActFlag, ExtActs, make_step

PRESET = {2: "tiny", 3: "tiny3d"}
NAMES = ("ROLE_SWITCHES", "N_FORAGERS", "N_GUARDS", "SHIELDED_BY_GUARD", "SHARE_SLOTS_ACTIVE")


def cfg_for(nd: int, mode: int = 2, *extra: str, roles: bool = True):
    over = ["brain.weight_dtype=float32", f"bonds.share_mode={mode}", *extra]
    if roles:
        over.append("roles.enabled=true")
    return load_config(preset=PRESET[nd], overrides=over)


# --- layout, config, registry -----------------------------------------------------------------------------


@pytest.mark.parametrize(("nd", "d_in", "d_out"), [(2, 82, 37), (3, 97, 41)])
@pytest.mark.parametrize("which", ["roles", "mode1", "mode2"])
def test_the_footprint_is_13_inputs_and_2_outputs(nd: int, d_in: int, d_out: int, which: str) -> None:
    cfg = {
        "roles": cfg_for(nd, 0),
        "mode1": cfg_for(nd, 1, roles=False),
        "mode2": cfg_for(nd, 2, roles=False),
    }[which]
    base = load_config(preset=PRESET[nd])
    d, d0 = dims(cfg), dims(base)
    lay = d.lay
    assert (lay.D_IN, lay.D_OUT) == (d_in, d_out) == (d0.lay.D_IN + 13, d0.lay.D_OUT + 2)
    assert tuple(d0.lay.D_IN + k for k in (0, 1, 5, 9)) == (
        lay.IN_ROLE,
        lay.IN_BOND_ROLE,
        lay.IN_BOND_KIN,
        lay.IN_BOND_FLOW,
    )
    assert (lay.OUT_ROLE_A, lay.OUT_ROLE_B) == (d0.lay.D_OUT, d0.lay.D_OUT + 1)
    assert ext.layout_features(cfg) == "bonds@1" and ext.mask(cfg) == 1 << 2
    assert layout.genome_layout_version(nd, ext.mask(cfg)) == (1 if nd == 2 else 2) + 4 * 4
    assert (d0.lay.IN_ROLE, d0.lay.OUT_ROLE_A) == (None, None)
    assert {"role", "role_dwell", "bond_flow"} <= set(state_cls(cfg)._fields)
    assert counter_blocks(cfg) == (("bonds", NAMES),)


def test_the_inert_twin_keeps_the_layout() -> None:
    """The measures' inert twin of bonds (O3): the roles switched off with the layout kept, i.e. the same
    brain, state and counters as the full configuration (sharing on, roles off)."""
    full, twin = cfg_for(2, 2), cfg_for(2, 2, roles=False)
    assert twin.roles.enabled is False and bonds.bonds_ext(twin)
    assert ext.layout_features(full) == ext.layout_features(twin) == "bonds@1"
    assert dims(full).lay.D_IN == dims(twin).lay.D_IN and state_cls(full)._fields == state_cls(twin)._fields


def test_validation() -> None:
    with pytest.raises(ConfigError, match="share_mode"):
        load_config(preset="tiny", overrides=["bonds.share_mode=3"])
    with pytest.raises(ConfigError, match=r"inert while bonds\.share_mode is 0"):
        load_config(preset="tiny", overrides=["bonds.share_kappa=0.1"])
    with pytest.raises(ConfigError, match=r"inert while bonds\.share_mode is 0"):
        load_config(preset="tiny", overrides=["bonds.share_kin_floor=0.5"])
    for bad in ("0", "0.2", "-0.1"):
        with pytest.raises(ConfigError, match="share_kappa"):
            load_config(preset="tiny", overrides=["bonds.share_mode=1", f"bonds.share_kappa={bad}"])
    with pytest.raises(ConfigError, match="share_kin_floor"):
        load_config(preset="tiny", overrides=["bonds.share_mode=2", "bonds.share_kin_floor=1.5"])
    with pytest.raises(ConfigError, match="share_rate"):
        load_config(preset="tiny", overrides=["bonds.share_mode=2", "bonds.share_rate=70000"])
    ok = load_config(preset="tiny", overrides=["bonds.share_mode=2", "bonds.share_kappa=0.125"])
    assert ok.bonds.share_mode == 2
    assert load_config(preset="tiny", overrides=["roles.enabled=true"]).roles.enabled


def test_defaults_keep_every_hash_and_the_keys_are_post_v01() -> None:
    for preset in config.PRESETS:
        cfg = load_config(preset=preset)
        d = config.config_to_dict(cfg, omit_defaults=True)
        assert "roles" not in d and not {"share_mode", "share_kappa", "share_kin_floor"} & set(d["bonds"])
    keys = {("bonds", k) for k in ("share_mode", "share_kappa", "share_kin_floor")}
    keys |= {("roles", f.name) for f in dataclasses.fields(config.RolesConfig)}
    assert keys <= set(config.POST_V01_KEYS) and keys <= set(config.SERIAL_OMIT_KEYS)
    assert keys == set(config.FEATURE_KEYS["bonds"])
    on = load_config(preset="tiny", overrides=["bonds.share_mode=2"])
    assert config.config_hash(on) != config.config_hash(load_config(preset="tiny"))


def test_the_overlay_composes() -> None:
    cfg = load_config(preset="tiny", overlays=["bonds"])
    assert cfg.bonds.share_mode == 2 and cfg.bonds.share_kin_floor == 0.5 and cfg.roles.enabled
    assert "bonds" in config.WORLD_OVERLAYS and bonds.bonds_ext(cfg)


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
@pytest.mark.parametrize("mode", [1, 2])
def test_the_ledger_closes_and_sharing_is_active(nd: int, mode: int) -> None:
    """No 17th term: sharing is FLOW_SHARED and HEAT_SHARE and the 16-term ledger closes every step."""
    cfg = cfg_for(nd, mode, "bonds.share_kin_floor=0.5")
    active = shared = heat = 0
    for t, (prev, s, out) in enumerate(_run(cfg, 120)):
        led = to_numpy(out.ledger)
        assert led.shape == (16,)
        assert helpers.ledger_net(led, cfg) == int(to_numpy(s.total) - to_numpy(prev.total)), t
        assert led[LedgerIdx.FLOW_STOLEN] == led[LedgerIdx.FLOW_ATTACK_GAIN] + led[LedgerIdx.FLOW_SPILL]
        assert int(to_numpy(s.diag)) & 0xFFFF == 0, (t, Diag(int(to_numpy(s.diag))))
        c = to_numpy(out.counters)
        active += int(c[counter_index(cfg, "SHARE_SLOTS_ACTIVE")])
        shared += int(led[LedgerIdx.FLOW_SHARED])
        heat += int(led[LedgerIdx.HEAT_SHARE])
        if t % 40 == 0:
            assert invariants.full_check(s, cfg, prev, out) == [], t
    assert active > 0 and shared > 0 and 0 < heat < shared  # sharing runs and loses some as heat


@pytest.mark.parametrize("nd", [2, 3])
def test_bond_flow_is_the_net_transfer_of_the_slot(nd: int) -> None:
    """``bond_flow`` = received − sent per slot at S12; over a bond the two ends sum to minus the heat, and
    only the richer end is a net giver; empty slots and dead rows hold 0."""
    cfg = cfg_for(nd, 2)
    seen = 0
    for _, s, _ in _run(cfg, 80):
        flow, partner, rev = (to_numpy(a).astype(np.int64) for a in (s.bond_flow, s.partner, s.rev))
        alive = to_numpy(s.alive)
        assert (flow[partner < 0] == 0).all() and (flow[~alive] == 0).all()
        i, b = np.nonzero(partner >= 0)
        j, r = partner[i, b], rev[i, b]
        pair = flow[i, b] + flow[j, r]
        assert (pair <= 0).all()  # the pair's net is the (non-positive) heat of the bond
        assert not ((flow[i, b] > 0) & (flow[j, r] > 0)).any()  # never two receivers on one bond
        seen += int((flow[i, b] != 0).sum())
    assert seen > 0


@pytest.mark.parametrize("nd", [2, 3])
def test_same_seed_same_digest_and_resume_equals_the_uninterrupted_run(nd: int, tmp_path: Path) -> None:
    cfg = cfg_for(nd, 1, "bonds.share_kin_floor=0.5")
    s0 = init_state(cfg)
    last = lambda g: list(g)[-1][1]  # noqa: E731
    a, b = last(_run(cfg, 50, s0)), last(_run(cfg, 50, init_state(cfg)))
    assert state_digest(a) == state_digest(b)
    half = last(_run(cfg, 25, s0))
    path = tmp_path / "half.safetensors"
    write_state_file(path, half, cfg)
    resumed = last(_run(cfg, 25, load_checkpoint(path, cfg)))
    assert state_digest(resumed) == state_digest(a)
    assert to_numpy(a.bond_flow).any()


@pytest.mark.parametrize("nd", [2, 3])
def test_observing_changes_nothing_and_the_channel_carries_the_bonds_tuple(nd: int) -> None:
    cfg = cfg_for(nd, 2)
    plain = list(_run(cfg, 20))
    obs = list(_run(cfg, 20, observe=True))
    assert state_digest(plain[-1][1]) == state_digest(obs[-1][1])
    for (_, _, o1), (_, s2, o2) in zip(plain, obs, strict=True):
        assert np.array_equal(to_numpy(o1.counters), to_numpy(o2.counters))
        x = o2.acts.ext
        assert isinstance(x, ExtActs) and isinstance(x.bonds, bonds.BondActs) and x.structures is None
        assert x.bonds.role.dtype == mx.int8 and x.bonds.bond_flow.dtype == mx.int16
        born = {int(r) for r in np.asarray(o2.acts.born_row) if r >= 0}
        keep = np.array([i not in born for i in range(len(to_numpy(s2.alive)))])
        assert np.array_equal(to_numpy(x.bonds.bond_flow)[keep], to_numpy(s2.bond_flow)[keep])
    assert ActFlag.ROLE_SWITCHED.value == 512


def test_off_has_no_bonds_state_and_the_counter_block_is_absent() -> None:
    cfg = load_config(preset="tiny", overrides=["brain.weight_dtype=float32"])
    s = init_state(cfg)
    assert not {"role", "role_dwell", "bond_flow"} & set(type(s)._fields)
    assert counter_blocks(cfg) == () and not bonds.bonds_ext(cfg)
    _, out = make_step(cfg)(s)
    assert to_numpy(out.counters).shape == (n_counters(2, cfg.weather.enabled),)


# --- E15: a partner dies and its row is reused in the same step -------------------------------------------


def test_e15_the_scripted_reuse_of_a_dead_partners_row() -> None:
    """Row 0 (energy 0, consent closed, so nothing rescues it) starves at S14 while bonded to row 1, which
    holds a flow memory of that bond; in the same step row 2 buds into the first free row, which is row 0, and
    bonds to its child. The survivor must lose the bond and the memory; the newborn must carry neither the
    bond, the role, the dwell nor the flow of its predecessor."""
    cfg = cfg_for(2, 1)
    n = dims(cfg).n
    bf, role, dwell = np.zeros((n, 4), np.int16), np.zeros(n, np.int8), np.zeros(n, np.int16)
    bf[1, 0], bf[0, 0], role[0], role[1], dwell[0], dwell[1] = 777, -777, 1, 2, 5, 9
    agents = [
        dict(pos=(10.0, 10.0), energy=0, birth_step=-100),
        dict(pos=(11.5, 10.0), energy=300_000, birth_step=-100),
        dict(pos=(30.0, 30.0), energy=400_000, birth_step=-100),
    ]
    s = helpers.make_scene(
        cfg, agents, bonds=[(0, 1)], role=mx.array(role), role_dwell=mx.array(dwell), bond_flow=mx.array(bf)
    )
    s = helpers.set_output_bias(s, cfg, {layout.OUT_REPRODUCE: 4.0, layout.OUT_CHILD_BOND: 4.0})
    w2 = np.array(s.W2)
    w2[:2, dims(cfg).h, layout.OUT_REPRODUCE] = -4.0  # only row 2 buds
    s = s._replace(W2=mx.array(w2))
    uid0 = int(to_numpy(s.uid)[0])
    s1, out = make_step(cfg)(s)
    mx.eval(s1, out)
    alive, uid = to_numpy(s1.alive), to_numpy(s1.uid)
    assert alive[:3].all() and uid[0] != uid0  # row 0 died and was reused in the same step
    assert np.asarray(out.births.slot)[np.asarray(out.births.valid)].tolist() == [0]
    partner, flow = to_numpy(s1.partner), to_numpy(s1.bond_flow)
    assert (partner[1] == -1).all() and not flow[1].any()  # the survivor lost the bond and its memory
    assert partner[0].tolist() == [2, -1, -1, -1] and partner[2, 0] == 0  # the newborn has only its parent
    assert not flow[0].any() and to_numpy(s1.role)[0] == 0 and to_numpy(s1.role_dwell)[0] == 0
    assert to_numpy(s1.role)[1] == 2 and to_numpy(s1.role_dwell)[1] == 10  # the survivor keeps its own role
    assert invariants.full_check(s1, cfg, s, out) == []


def test_e15_random_runs_keep_the_bond_invariants_across_deaths_and_births() -> None:
    """Over 400 steps of a tiny world (mode 2, kin floor 0.5) every row reused in a step has no partner but
    its parent and fresh fields, no survivor points at a reused row but its parent, and the host invariants
    hold."""
    cfg = cfg_for(2, 2, "bonds.share_kin_floor=0.5")
    reused = 0
    for prev, s, out in _run(cfg, 400):
        pa, na = to_numpy(prev.alive), to_numpy(s.alive)
        new = na & ((~pa) | (to_numpy(prev.uid) != to_numpy(s.uid)))
        reused += int((new & pa).sum())
        partner, rev, flow = to_numpy(s.partner), to_numpy(s.rev), to_numpy(s.bond_flow)
        uid, parent_uid = to_numpy(s.uid), to_numpy(s.parent_uid)
        for row in np.flatnonzero(new):
            assert to_numpy(s.role)[row] == 0 and to_numpy(s.role_dwell)[row] == 0 and not flow[row].any()
            for p in partner[row][partner[row] >= 0]:
                assert uid[p] == parent_uid[row]
        for i, b in zip(*np.nonzero(partner >= 0), strict=True):
            j = partner[i, b]
            assert na[j] and partner[j, rev[i, b]] == i and rev[j, rev[i, b]] == b
            if new[j] and not new[i]:
                assert uid[i] == parent_uid[j]
        assert invariants.full_check(s, cfg, prev, out) == []
    assert reused > 0


def test_state_after_clears_dead_rows_and_gone_slots() -> None:
    """The unit form of E15: a survivor whose partner died keeps no flow memory for that slot, and a dead row
    is cleared, whatever the S12 flows were."""
    n, b = 4, 4
    s = SimpleNamespace(
        role=mx.array([1, 2, 1, 0], dtype=mx.int8),
        role_dwell=mx.array([5, 6, 7, 8], dtype=mx.int16),
        alive=mx.array([True, True, True, True]),
    )
    out_, got_ = np.zeros((n, b), np.int32), np.zeros((n, b), np.int32)
    out_[0, 0], got_[1, 0] = 7, 5  # row 0 sends 7, row 1 receives 5 (the rest is heat); row 2 had a flow too
    got_[2, 0], got_[0, 1] = 40, 9  # a dying row and a gone slot (partner[0, 1] = -1) with nonzero flows
    sl = SimpleNamespace(out=mx.array(out_), got=mx.array(got_))
    sv = mx.array([True, True, False, True])
    partner = mx.array([[1, -1, -1, -1], [0, -1, -1, -1], [-1] * 4, [-1] * 4], dtype=mx.int32)
    upd, f = bonds.state_after(s, sv, partner, sl, None)
    flow = np.asarray(upd["bond_flow"])
    assert flow[0, 0] == -7 and flow[1, 0] == 5 and not flow[0, 1:].any()  # live slots keep their net
    assert not flow[2].any() and not flow[3].any()  # the dead row and the slot of a gone bond: 0
    assert np.asarray(upd["role"]).tolist() == [1, 2, 0, 0] and np.asarray(upd["role_dwell"]).tolist() == [
        5,
        6,
        0,
        8,
    ]
    assert int(f.switched.sum()) == 0  # clearing a dying row is not a switch
    big = SimpleNamespace(out=mx.zeros((n, b), dtype=mx.int32), got=mx.full((n, b), 10**6, dtype=mx.int32))
    upd, _ = bonds.state_after(s, sv, partner, big, None)
    assert np.asarray(upd["bond_flow"]).max() == 32767  # clipped to int16


# --- counters, metrics (E10) ------------------------------------------------------------------------------


def test_counters_and_metrics_row() -> None:
    cfg = cfg_for(2, 2, roles=False)  # sharing only: the role counters read 0
    fn, s = make_step(cfg), init_state(cfg)
    cs, ls = [], []
    for _ in range(30):
        s, out = fn(s)
        mx.eval(s, out)
        cs.append(to_numpy(out.counters))
        ls.append(to_numpy(out.ledger))
    c = np.stack(cs)
    i = counter_index(cfg, "SHARE_SLOTS_ACTIVE")
    assert c[:, i].sum() > 0 and c[:, counter_index(cfg, "N_FORAGERS")].sum() == 0
    blocks = counter_blocks(cfg)
    row = metrics.interval_row(30, c, np.stack(ls), {}, {}, nd=2, blocks=blocks)
    assert row["bond_flux"] == int(np.stack(ls)[:, LedgerIdx.FLOW_SHARED].sum()) > 0
    assert row["bond_share_slots"] == pytest.approx(c[:, i].sum() / 30)
    assert row["bond_role_switches"] == 0 and row["bond_forager_frac"] == 0.0


def test_e10_the_interval_sums_are_int64() -> None:
    """The per-step counters are int32 (each at most N); the interval row sums them in int64, so a long
    interval of a large world does not wrap (65 536 slots x 60 000 steps = 3.9e9 > 2^31)."""
    cfg = cfg_for(2, 2)
    width = n_counters(2, False, counter_blocks(cfg))
    t = 60_000
    c = np.zeros((t, width), np.int32)
    i = counter_index(cfg, "SHARE_SLOTS_ACTIVE")
    c[:, i] = 65_536
    c[:, 0] = 1
    lg = np.zeros((t, len(LedgerIdx)), np.int64)
    lg[:, LedgerIdx.FLOW_SHARED] = 2_000_000_000  # int32 would wrap after two steps
    row = metrics.interval_row(t, c, lg, {}, {}, nd=2, blocks=counter_blocks(cfg))
    assert row["bond_share_slots"] == 65_536.0
    assert row["bond_flux"] == t * 2_000_000_000 > 2**40


# --- the staged mirror of bench.py equals step() ----------------------------------------------------------


@pytest.mark.parametrize("nd", [2, 3])
@pytest.mark.parametrize("kin_floor", ["1.0", "0.5"])
def test_the_staged_pipeline_equals_the_step_with_bonds_on(nd: int, kin_floor: str) -> None:
    from mlx.utils import tree_flatten

    from alife import bench

    cfg = cfg_for(nd, 2, f"bonds.share_kin_floor={kin_floor}")
    cfg = replace(cfg, run=replace(cfg.run, compile=False))
    s0 = init_state(cfg)
    fused, staged = make_step(cfg), bench._StagedStep(cfg)
    s, holder = s0, [s0]
    for i in range(6):
        s, out = fused(s)
        out_staged = staged(holder)
        mx.eval(s, out, holder[0], out_staged)
        assert state_digest(holder[0]) == state_digest(s), i
        for a, b in zip(tree_flatten(out), tree_flatten(out_staged), strict=True):
            assert a[0] == b[0]
            np.testing.assert_array_equal(to_numpy(a[1]), to_numpy(b[1]), err_msg=a[0])
    assert to_numpy(s.bond_flow).any()


def test_bench_bonds_flag_needs_the_feature() -> None:
    from alife import bench

    with pytest.raises(ConfigError, match="bench --bonds"):
        bench.run_bench(load_config(preset="tiny"), steps=1, warmup=1, bonds_arm=True)
