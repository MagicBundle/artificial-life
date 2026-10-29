"""Bonds that pay, B2 (DESIGN §40.3 to §40.5): the role machine, its effects on closed flows, the payoff
matrix that fixes the starting parameters, the combat order (E16) and the ledger with roles on."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

import helpers
from alife import energy as en
from alife import invariants, layout, ops, roles
from alife.config import ConfigError, dims, load_config
from alife.layout import B
from alife.state import LedgerIdx, counter_index, to_numpy
from alife.step import ActFlag, make_step
from helpers import make_scene, set_output_bias
from test_bonds_step import cfg_for
from test_energy import _actions

ROLE_KEYS = [
    "roles.forage_gain=0.75",
    "roles.forage_defend_pen=0.5",
    "roles.guard_harvest_pen=0.5",
    "roles.guard_shield=0.5",
    "roles.synergy=0.25",
    "roles.ramp_steps=16",
    "roles.switch_cost=100",
]


def rcfg(nd: int = 2, *extra: str):
    return cfg_for(nd, 2, *ROLE_KEYS, *extra)


# --- config -----------------------------------------------------------------------------------------------


def test_validation() -> None:
    bad = {
        "roles.forage_gain=-0.1": "forage_gain",
        "roles.forage_defend_pen=1.0": "forage_defend_pen",
        "roles.guard_harvest_pen=1.0": "guard_harvest_pen",
        "roles.guard_shield=0.95": "guard_shield",
        "roles.synergy=-1": "synergy",
        "roles.ramp_steps=0": "ramp_steps",
        "roles.ramp_steps=40000": "ramp_steps",
        "roles.threshold=0": "threshold",
        "roles.threshold=1": "threshold",
        "roles.switch_cost=-1": "switch_cost",
        "roles.forage_gain=200": "65 536",
    }
    for kv, word in bad.items():
        with pytest.raises(ConfigError, match=word):
            load_config(preset="tiny", overrides=["roles.enabled=true", kv])
    with pytest.raises(ConfigError, match=r"inert while roles\.enabled is false"):
        load_config(preset="tiny", overrides=["roles.guard_shield=0.5"])
    ok = load_config(preset="tiny", overrides=["roles.enabled=true", *ROLE_KEYS])
    assert ok.roles.guard_shield == 0.5 and ok.roles.inherit_role is False


# --- the role machine -------------------------------------------------------------------------------------


def _y(a: float, b: float, cfg, n: int = 1):
    d = dims(cfg)
    y = np.zeros((n, d.lay.D_OUT), np.float32)
    y[:, d.lay.OUT_ROLE_A], y[:, d.lay.OUT_ROLE_B] = np.arctanh(a), np.arctanh(b)
    return mx.array(y)


def test_r1_decode_role_corners_and_the_dead_zone() -> None:
    cfg = rcfg()
    for role in (0, 1, 2):
        r = mx.array([role], dtype=mx.int8)
        got = lambda a, b, r=r: int(roles.decode_role(_y(a, b, cfg), r, cfg)[0])  # noqa: E731
        assert got(0.9, -0.9) == 1  # forager
        assert got(-0.9, 0.9) == 2  # guard
        assert got(-0.9, -0.9) == 0  # generalist
        assert got(0.9, 0.9) == role  # the germ corner is reserved: keep
        # The dead zone keeps the role (|x| <= tau = 0.5). Not tested at exactly tau: tanh(arctanh(0.5))
        # rounds to either side of 0.5 in float32 depending on the platform's libm (Linux CI: 0.50000006).
        for a, b in ((0.3, -0.9), (0.9, -0.3), (0.0, 0.0), (-0.3, 0.3), (0.49, -0.49)):
            assert got(a, b) == role, (a, b)
        assert got(0.51, -0.51) == 1


def _scene(cfg, n_agents: int = 3, **agent):
    return make_scene(cfg, [dict(pos=(10.0 + 6 * i, 10.0), **agent) for i in range(n_agents)])


def _scripted(cfg, a: float, b: float, n_agents: int = 3, **agent):
    d = dims(cfg)
    s = _scene(cfg, n_agents, **agent)
    return set_output_bias(s, cfg, {d.lay.OUT_ROLE_A: a, d.lay.OUT_ROLE_B: b})


def test_r1_a_switch_commits_resets_the_dwell_and_costs_energy() -> None:
    cfg = rcfg(2, "roles.ramp_steps=4")
    d = dims(cfg)
    fn = make_step(cfg)
    s = _scripted(cfg, 4.0, -4.0)  # everyone asks for forager
    e0 = to_numpy(s.energy)[:3].copy()
    s1, o1 = fn(s)
    mx.eval(s1, o1)
    role, dwell = to_numpy(s1.role)[:3], to_numpy(s1.role_dwell)[:3]
    assert role.tolist() == [1, 1, 1] and dwell.tolist() == [0, 0, 0]
    c = to_numpy(o1.counters)
    assert c[counter_index(cfg, "ROLE_SWITCHES")] == 3 and c[counter_index(cfg, "N_FORAGERS")] == 3
    # the cost is part of HEAT_ACTION: the same step without the switch cost has 3 * 100 less action heat
    free = dataclasses.replace(cfg, roles=dataclasses.replace(cfg.roles, switch_cost=0))
    _, of = make_step(free)(s)
    mx.eval(of)
    assert (
        to_numpy(o1.ledger)[LedgerIdx.HEAT_ACTION] - to_numpy(of.ledger)[LedgerIdx.HEAT_ACTION] == 300
    ) and e0.sum() > 0
    # staying asked for the same role: no new switch, the dwell grows one per step and saturates at ramp_steps
    seq, s_ = [], s1
    for _ in range(7):
        s_, o_ = fn(s_)
        mx.eval(s_, o_)
        seq.append(int(to_numpy(s_.role_dwell)[0]))
        assert to_numpy(o_.counters)[counter_index(cfg, "ROLE_SWITCHES")] == 0
    assert seq == [1, 2, 3, 4, 4, 4, 4] and to_numpy(s_.role)[:3].tolist() == [1, 1, 1]
    # a different request switches again and resets the dwell
    s_ = set_output_bias(s_, cfg, {d.lay.OUT_ROLE_A: -4.0, d.lay.OUT_ROLE_B: 4.0})
    s_, o_ = fn(s_)
    mx.eval(s_, o_)
    assert to_numpy(s_.role)[:3].tolist() == [2, 2, 2] and to_numpy(s_.role_dwell)[:3].tolist() == [0, 0, 0]
    assert to_numpy(o_.counters)[counter_index(cfg, "N_GUARDS")] == 3


def test_r1_the_dead_zone_keeps_a_committed_role_without_a_sustained_output() -> None:
    cfg = rcfg(2, "roles.switch_cost=0")
    d = dims(cfg)
    fn = make_step(cfg)
    s, _ = fn(_scripted(cfg, 4.0, -4.0))
    s = set_output_bias(s, cfg, {d.lay.OUT_ROLE_A: 0.0, d.lay.OUT_ROLE_B: 0.0})  # silence
    for _ in range(5):
        s, o = fn(s)
        mx.eval(s, o)
    assert to_numpy(s.role)[:3].tolist() == [1, 1, 1] and int(to_numpy(s.role_dwell)[0]) == 5


def test_r1_an_insolvent_row_cannot_switch() -> None:
    cfg = rcfg(2, "roles.switch_cost=100")
    s = _scripted(cfg, 4.0, -4.0, energy=60)  # basal alone is 100 q: insolvent before the switch is even due
    s1, o1 = make_step(cfg)(s)
    mx.eval(s1, o1)
    alive = to_numpy(s1.alive)
    assert not alive[:3].any()  # starved (no one switched into a role as a dead row either)
    assert (
        to_numpy(s1.role)[:3].tolist() == [0, 0, 0]
        and int(to_numpy(o1.counters)[counter_index(cfg, "ROLE_SWITCHES")]) == 0
    )
    # solvent for the basal cost but not for basal + switch: energy just above 100 + brain cost, below +100
    cfg2 = rcfg(2, "roles.switch_cost=100")
    s = _scripted(cfg2, 4.0, -4.0, energy=100 + 40)
    s = s._replace(brain_cost=mx.zeros_like(s.brain_cost))
    s1, _ = make_step(cfg2)(s)
    mx.eval(s1)
    assert to_numpy(s1.role)[:3].tolist() != [1, 1, 1] or not to_numpy(s1.alive)[:3].any()


def test_r1_newborn_role_and_inherit_role() -> None:
    """A newborn starts with role 0 and dwell 0; with ``inherit_role`` a bud takes its parent's committed role
    (the role at the end of the step it is born in), dwell 0 either way."""
    for inherit, want in ((False, 0), (True, 1)):
        cfg = rcfg(2, "roles.switch_cost=0", f"roles.inherit_role={'true' if inherit else 'false'}")
        s = _scripted(cfg, 4.0, -4.0, n_agents=2, energy=400_000, birth_step=-100)
        s = set_output_bias(s, cfg, {layout.OUT_REPRODUCE: 4.0})
        fn = make_step(cfg)
        seen = 0
        for _ in range(3):
            s, o = fn(s)
            mx.eval(s, o)
            valid = np.asarray(o.births.valid)
            for r in np.asarray(o.births.slot)[valid]:
                assert to_numpy(s.role_dwell)[int(r)] == 0
                assert to_numpy(s.role)[int(r)] == want, (inherit, int(r))
                seen += 1
        assert seen > 0, "the scene must give births"


# --- effects (R-3) ----------------------------------------------------------------------------------------


def _ns(role, dwell, n=None):
    role = np.asarray(role, np.int8)
    return SimpleNamespace(role=mx.array(role), role_dwell=mx.array(np.asarray(dwell, np.int16)))


def test_r3_ramp_and_the_integer_harvest_cap() -> None:
    cfg = rcfg(2, "roles.ramp_steps=64")
    H = cfg.energy.harvest_max
    assert np.asarray(roles.ramp_q8(mx.array([0, 1, 16, 32, 63, 64, 99], dtype=mx.int16), cfg)).tolist() == [
        0, 4, 64, 128, 252, 256, 256,
    ]  # fmt: skip
    # role: G(0) F(1) F D(2) F ; the forager beside a ramped guard gets the synergy
    s = _ns([0, 1, 1, 2, 1, 2], [0, 64, 32, 64, 64, 10])
    n = 6
    partner = np.full((n, B), -1, np.int32)
    partner[4, 0], partner[3, 0] = 3, 4  # forager 4 beside guard 3 (ramp 256)
    partner[1, 0], partner[5, 0] = 5, 1  # forager 1 beside guard 5 (ramp 40 < 128: no synergy)
    act = SimpleNamespace(
        attack=mx.zeros(n, dtype=mx.bool_),
        attack_target=mx.full((n,), -1, dtype=mx.int32),
        defend=mx.ones(n),
    )
    ef = roles.effects(s, act, mx.array(partner), cfg)
    q = ops.q8
    expect = [
        H,
        H + (H * q(0.75) * 256 >> 16),
        H + (H * q(0.75) * 128 >> 16),
        H - (H * q(0.5) * 256 >> 16),
        H + (H * q(0.75) * 256 >> 16) + (H * q(0.25) * 256 >> 16),
        H - (H * q(0.5) * 40 >> 16),
    ]
    expect[1] = H + (H * q(0.75) * 256 >> 16)  # beside the weak guard: no synergy term
    assert np.asarray(ef.harvest_cap).tolist() == expect
    assert max(expect) < 1 << 16
    # the own-defence multiplier is the forager's only
    dm = np.asarray(ef.defend_mult)
    assert dm.tolist() == pytest.approx([1.0, 0.5, 0.75, 1.0, 0.5, 1.0])


def test_r3_off_effects_are_absent() -> None:
    cfg = cfg_for(2, 2)  # roles.enabled with every effect key at its default
    s = _ns([1, 2], [64, 64])
    act = SimpleNamespace(
        attack=mx.zeros(2, dtype=mx.bool_), attack_target=mx.full((2,), -1, dtype=mx.int32), defend=mx.ones(2)
    )
    ef = roles.effects(s, act, mx.full((2, B), -1, dtype=mx.int32), cfg)
    assert ef.harvest_cap is None and ef.defend_mult is None and ef.guard is None
    assert roles.combat_act(act, None) is act


def test_r3_harvest_with_a_cap_obeys_the_cap_and_the_ledger() -> None:
    cfg = rcfg()
    d = dims(cfg)
    n = d.n
    rs = np.random.default_rng(1)
    energy = mx.array(rs.integers(0, 400_000, n).astype(np.int32))
    res = mx.array(rs.integers(0, 600, d.field_shape).astype(np.int32))
    fcell = mx.array(rs.integers(0, d.n_cells // 6, n).astype(np.int32))
    act = _actions(n, harvest=rs.random(n))
    solvent = mx.array(rs.random(n) < 0.9)
    cap = mx.array(rs.integers(40, 700, n).astype(np.int32))
    _, r2, got = en.harvest(energy, res, fcell, act, solvent, cfg, harvest_cap=cap)
    got_, cap_ = np.asarray(got), np.asarray(cap)
    assert (got_ <= cap_).all() and (got_ >= 0).all()
    assert int(got.sum()) == int(res.sum()) - int(r2.sum())  # field to agents, exactly
    # with a constant cap equal to harvest_max the result equals the legacy call
    base = en.harvest(energy, res, fcell, act, solvent, cfg)
    same = en.harvest(
        energy,
        res,
        fcell,
        act,
        solvent,
        cfg,
        harvest_cap=mx.full((n,), cfg.energy.harvest_max, dtype=mx.int32),
    )
    for a, b in zip(base, same, strict=True):
        assert np.array_equal(np.asarray(a), np.asarray(b))


def test_r3_a_guard_shields_itself_and_its_partner_not_a_third_party() -> None:
    cfg = rcfg()
    # rows: 0 guard, 1 its partner, 2 a third party (no bond), 3 a guard attacking
    s = _ns([2, 0, 0, 2], [64, 0, 0, 64])
    n = 4
    partner = np.full((n, B), -1, np.int32)
    partner[0, 0], partner[1, 0] = 1, 0
    act = SimpleNamespace(
        attack=mx.array([False, False, False, True]),
        attack_target=mx.array([-1, -1, -1, 2], dtype=mx.int32),
        defend=mx.array([1.0, 0.0, 0.0, 1.0]),
    )
    g = np.asarray(roles.effects(s, act, mx.array(partner), cfg).guard)
    gs = cfg.roles.guard_shield
    # row 0: guard, defend 1, ramp 256 -> itself gs; row 1: its partner -> gs; row 2: nobody guards -> 0
    assert g.tolist() == pytest.approx([gs, gs, 0.0, 0.0])  # row 3 attacks: it contributes no shield (E16)


def test_r3_the_guard_factor_keeps_the_stolen_identity_with_structures_and_arms_stacked() -> None:
    cfg = rcfg()
    d = dims(cfg)
    n = d.n
    rs = np.random.default_rng(3)
    for seed in range(4):
        e = mx.array(np.random.default_rng(seed).integers(0, 12_000, n).astype(np.int32))
        res = mx.array(rs.integers(0, 500, d.field_shape).astype(np.int32))
        fcell = mx.array(rs.integers(0, res.size, n).astype(np.int32))
        tgt = rs.integers(0, n, n).astype(np.int32)
        act = _actions(n, attack=rs.random(n) < 0.6, attack_target=tgt, defend=rs.random(n))
        partner = mx.array(np.full((n, B), -1, np.int32))
        guard = mx.array((0.5 * (rs.random(n) < 0.5)).astype(np.float32))
        shelter = mx.array((0.5 * rs.random(n)).astype(np.float32))
        cb = en.combat(e, res, fcell, act, partner, cfg, guard=guard, shelter=shelter)
        assert int(cb.lost.sum()) == int(cb.gain.sum()) + int(cb.spill.sum()) + int(cb.residue.sum())
        assert int(cb.shielded.sum()) >= 0
        plain = en.combat(e, res, fcell, act, partner, cfg)
        assert int(plain.lost.sum()) >= int(cb.lost.sum())  # protection only reduces the demand


# --- E16: a guard and its bonded attacker -----------------------------------------------------------------


def test_e16_a_bonded_attacker_is_not_softened_by_its_own_guard() -> None:
    """Order of S11: the guard potential is computed from the START-of-S11 actions and is zero for a row that
    attacks this step. With ``attack_bonded`` a guard that steals from its bonded partner gets the full demand
    (its own shield never covers its target), while a third guard bonded to the victim does reduce it."""
    cfg = rcfg(2, "combat.attack_bonded=true")
    d = dims(cfg)
    res = mx.array(np.zeros(d.field_shape, np.int32))
    fcell = mx.array(np.zeros(dims(cfg).n, np.int32))
    N = d.n
    role = np.zeros(N, np.int8)
    dwell = np.zeros(N, np.int16)
    role[[0, 2]], dwell[[0, 2]] = 2, 64  # 0 and 2 are guards
    partner = np.full((N, B), -1, np.int32)
    partner[0, 0], partner[1, 0] = 1, 0  # 0 (guard) -- 1 (victim)
    partner[2, 0], partner[1, 1] = 1, 2  # 2 (guard) -- 1
    attack = np.zeros(N, bool)
    attack[0] = True
    tgt = np.full(N, -1, np.int32)
    tgt[0] = 1
    defend = np.zeros(N, np.float32)
    defend[[0, 2]] = 1.0
    act = _actions(N, attack=attack, attack_target=tgt, defend=defend)
    e = np.zeros(N, np.int32)
    e[:3] = 500_000
    s = SimpleNamespace(role=mx.array(role), role_dwell=mx.array(dwell))
    ef = roles.effects(s, act, mx.array(partner), cfg)
    gs = cfg.roles.guard_shield
    g = np.asarray(ef.guard)
    assert g[0] == pytest.approx(0.0) and g[1] == pytest.approx(gs) and g[2] == pytest.approx(gs)
    cb = en.combat(mx.array(e), res, fcell, act, mx.array(partner), cfg, guard=ef.guard)
    # the victim (row 1, defend 0) is covered by the third guard only: demand = attack_max * (1 - gs)
    assert int(cb.lost[1]) == int(np.floor(cfg.combat.attack_max * (1 - gs)))
    # remove the third guard: no cover at all, the full demand
    partner2 = partner.copy()
    partner2[2], partner2[1, 1] = -1, -1
    ef2 = roles.effects(s, act, mx.array(partner2), cfg)
    cb2 = en.combat(mx.array(e), res, fcell, act, mx.array(partner2), cfg, guard=ef2.guard)
    assert int(cb2.lost[1]) == cfg.combat.attack_max


# --- the payoff matrix (R-2) ------------------------------------------------------------------------------


def _net(cfg, members: str, supply: int, outsiders: int, attack_rate: float) -> float:
    """Steady-state net of a group of one or two members (``"G"``, ``"F"``, ``"D"``; two are bonded), in q per
    step: harvest on a cell of ``supply`` shared with ``outsiders`` generalist rivals, minus the predation
    loss, minus the defend cost. Each flow is measured on the engine's own functions (``roles.effects``,
    ``en.harvest(harvest_cap=)``, one attack on each member through ``en.combat(guard=)``) and combined
    analytically as in the spec's ``probe_payoff.py`` (the attack rate scales the one-attack loss)."""
    d = dims(cfg)
    n = d.n
    code = {"G": 0, "F": 1, "D": 2}
    m = len(members)
    role = np.zeros(n, np.int8)
    role[:m] = [code[r] for r in members]
    s = SimpleNamespace(role=mx.array(role), role_dwell=mx.full((n,), cfg.roles.ramp_steps, dtype=mx.int16))
    partner = np.full((n, B), -1, np.int32)
    if m == 2:
        partner[0, 0], partner[1, 0] = 1, 0
    pred = [n - 1, n - 2]  # attackers, outside the cell group
    defend = np.zeros(n, np.float32)
    defend[:m] = 1.0
    ones = np.ones(n, np.float32)
    act0 = _actions(n, harvest=ones, defend=defend)
    ef = roles.effects(s, act0, mx.array(partner), cfg)
    grp = np.zeros(n, bool)
    grp[: m + outsiders] = True
    res = np.zeros(d.field_shape, np.int32)
    res.reshape(-1)[0] = supply
    fcell = np.zeros(n, np.int32)
    _, _, take = en.harvest(
        mx.array(np.where(grp, 1000, 0).astype(np.int32)),
        mx.array(res),
        mx.array(fcell),
        act0,
        mx.array(grp),
        cfg,
        harvest_cap=ef.harvest_cap,
    )
    income = int(np.asarray(take)[:m].sum())
    loss = 0
    for k in range(m):
        attack = np.zeros(n, bool)
        attack[pred[0]] = True
        tgt = np.full(n, -1, np.int32)
        tgt[pred[0]] = k
        act = _actions(n, attack=attack, attack_target=tgt, defend=defend)
        e = np.zeros(n, np.int32)
        e[:m], e[pred[0]] = 600_000, 600_000
        ef_k = roles.effects(s, act, mx.array(partner), cfg)
        cb = en.combat(
            mx.array(e), mx.array(res), mx.array(fcell), roles.combat_act(act, ef_k.defend_mult),
            mx.array(partner), cfg, guard=ef_k.guard,
        )  # fmt: skip
        loss += int(cb.lost[k])
    return income - attack_rate * loss - m * cfg.energy.defend_cost


def test_r2_the_payoff_matrix_at_the_overlay_values() -> None:
    """The pair F+D beats two generalists under moderate predation, and no lone specialist beats a lone
    generalist: the complementarity the roles are built on (spec 3.3), measured on the engine's own flows."""
    cfg = rcfg(2, "roles.ramp_steps=64")
    for supply, outs, rate in ((1500, 2, 0.25), (900, 4, 0.10)):
        pairs = {k: _net(cfg, k, supply, outs, rate) for k in ("GG", "FF", "DD", "FD", "FG", "GD")}
        lone = {k: _net(cfg, k, supply, outs, rate) for k in "GFD"}
        assert pairs["FD"] > pairs["GG"], (supply, pairs)
        assert pairs["FD"] > pairs["GD"] and pairs["FD"] > pairs["FF"] > -1e9, (supply, pairs)
        assert lone["G"] > lone["F"] and lone["G"] > lone["D"], (supply, lone)
    # no predation: the guard is a waste and the foragers win (the negative control of the spec's O4)
    calm = {k: _net(cfg, k, 900, 4, 0.0) for k in ("GG", "FF", "DD", "FD")}
    assert calm["FF"] > calm["FD"] > calm["GG"] > calm["DD"], calm
    # the synergy is what widens the pair's margin over a generalist beside a guard
    nosyn = rcfg(2, "roles.ramp_steps=64", "roles.synergy=0.0")
    fd, gd = _net(cfg, "FD", 1500, 2, 0.25), _net(cfg, "GD", 1500, 2, 0.25)
    assert fd - gd > _net(nosyn, "FD", 1500, 2, 0.25) - _net(nosyn, "GD", 1500, 2, 0.25)


# --- the step with roles ----------------------------------------------------------------------------------


def _run(cfg, steps, s=None, *, observe=False):
    from test_bonds_step import _run as run

    yield from run(cfg, steps, s, observe=observe)


@pytest.mark.parametrize("nd", [2, 3])
def test_the_ledger_closes_with_roles_and_roles_appear(nd: int) -> None:
    """No 17th term with every role effect on: the harvest cap, the guard shield, the switch cost and the
    conductive share all book into existing terms, and the specialists appear from the founders' random
    brains."""
    cfg = rcfg(nd)
    foragers = guards = switches = shielded = 0
    for t, (prev, s, out) in enumerate(_run(cfg, 250)):
        led = to_numpy(out.ledger)
        assert helpers.ledger_net(led, cfg) == int(to_numpy(s.total) - to_numpy(prev.total)), t
        assert led[LedgerIdx.FLOW_STOLEN] == led[LedgerIdx.FLOW_ATTACK_GAIN] + led[LedgerIdx.FLOW_SPILL], t
        assert int(to_numpy(s.diag)) & 0xFFFF == 0, t
        c = to_numpy(out.counters)
        foragers += int(c[counter_index(cfg, "N_FORAGERS")])
        guards += int(c[counter_index(cfg, "N_GUARDS")])
        switches += int(c[counter_index(cfg, "ROLE_SWITCHES")])
        shielded += int(c[counter_index(cfg, "SHIELDED_BY_GUARD")])
        dwell, role = to_numpy(s.role_dwell), to_numpy(s.role)
        assert dwell.max() <= cfg.roles.ramp_steps and set(np.unique(role)) <= {0, 1, 2}
        if t % 50 == 0:
            assert invariants.full_check(s, cfg, prev, out) == [], t
    assert foragers > 0 and guards > 0 and switches > 0
    assert shielded >= 0


@pytest.mark.parametrize("nd", [2, 3])
def test_determinism_and_observer_purity_with_roles(nd: int) -> None:
    from alife.state import state_digest

    cfg = rcfg(nd)
    a = list(_run(cfg, 40))[-1][1]
    b = list(_run(cfg, 40))[-1][1]
    obs = list(_run(cfg, 40, observe=True))
    assert state_digest(a) == state_digest(b) == state_digest(obs[-1][1])
    # the ROLE_SWITCHED bit marks exactly the rows whose role changed
    flagged = 0
    for prev, s, out in obs:
        bit = (np.asarray(out.acts.flags).astype(np.int64) & ActFlag.ROLE_SWITCHED.value) != 0
        pa, na = to_numpy(prev.alive), to_numpy(s.alive)
        changed = (
            pa & na & (to_numpy(prev.uid) == to_numpy(s.uid)) & (to_numpy(prev.role) != to_numpy(s.role))
        )
        assert np.array_equal(bit & changed, changed) and not (bit & ~pa).any()
        flagged += int(bit.sum())
    assert flagged > 0


def test_roles_without_effects_keys_change_only_the_state_not_the_energy_flows() -> None:
    """``roles.enabled`` with every effect key at its default: roles switch (cost 0) but harvest, combat and
    sharing are the conductive world's, so the ledger of a step equals that of the same world with the roles
    switched off (the inert twin) when no row has switched yet."""
    base = cfg_for(2, 2, roles=True)
    twin = cfg_for(2, 2, roles=False)
    s = _scene(base, 4, energy=300_000)
    s0 = make_scene(twin, [dict(pos=(10.0 + 6 * i, 10.0), energy=300_000) for i in range(4)])
    _, o1 = make_step(base)(s)
    _, o2 = make_step(twin)(s0)
    mx.eval(o1, o2)
    assert np.array_equal(to_numpy(o1.ledger), to_numpy(o2.ledger))


def test_the_inspector_shows_the_role_and_the_bond_slots() -> None:
    from alife import cli

    cfg = rcfg(2, "roles.switch_cost=0")
    s = make_scene(
        cfg,
        [dict(pos=(10.0 + 2 * i, 10.0), energy=300_000) for i in range(3)],
        bonds=[(0, 1)],
        role=mx.array(np.pad(np.array([1, 2, 0], np.int8), (0, dims(cfg).n - 3))),
        role_dwell=mx.array(np.pad(np.array([8, 16, 0], np.int16), (0, dims(cfg).n - 3))),
    )
    info = cli._agent_info(0, s, cfg, None)
    assert info["role"] == {"code": 1, "name": "forager", "ramp": 0.5}
    (slot,) = info["bonds"]
    assert slot["role"] == "guard" and slot["flow"] == 0 and 0.0 < slot["kin"] <= 1.0
    assert cli._agent_info(1, s, cfg, None)["shield_max"] == cfg.roles.guard_shield


def test_e16_a_dropped_attack_on_a_bonded_partner_does_not_cost_the_guard_its_shield() -> None:
    """Without ``attack_bonded`` combat drops an attack on a partner, so the guard that "attacks" its partner
    is not attacking and keeps shielding."""
    cfg = rcfg()
    n = dims(cfg).n
    role, dwell = np.zeros(n, np.int8), np.full(n, 64, np.int16)
    role[0] = 2
    partner = np.full((n, B), -1, np.int32)
    partner[0, 0], partner[1, 0] = 1, 0
    attack, tgt = np.zeros(n, bool), np.full(n, -1, np.int32)
    attack[0], tgt[0] = True, 1
    defend = np.zeros(n, np.float32)
    defend[0] = 1.0
    act = _actions(n, attack=attack, attack_target=tgt, defend=defend)
    s = SimpleNamespace(role=mx.array(role), role_dwell=mx.array(dwell))
    g = np.asarray(roles.effects(s, act, mx.array(partner), cfg).guard)
    assert g[0] == pytest.approx(cfg.roles.guard_shield) and g[1] == pytest.approx(cfg.roles.guard_shield)
    bonded = rcfg(2, "combat.attack_bonded=true")
    g = np.asarray(roles.effects(s, act, mx.array(partner), bonded).guard)
    assert g[0] == 0.0
