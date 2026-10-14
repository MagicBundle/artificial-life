"""§27.3–§27.8 3D affordances through one eager step on hand-built ``micro3d`` scenes: an attack drain and
its closure with ``HEAT_BUOYANCY``, a corpse sinking to the floor cell, harvest from the agent's own cell
(mid-water or grounded), a vertical bond, the pitch and its turn cost, buoyant lift and its cost against a
float32 reference, a zero cap and the upkeep without lift, the surface clamp without grounding; and, over 400
compiled ``tiny3d`` steps, every affordance of §0.1 with the 3D terms (the z clamps, ``HEAT_BUOYANCY``, pitch,
the heritable cap, signals in the volume, swimming apart from sinking)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import mlx.core as mx
import numpy as np
import pytest

from alife import brain, invariants, layout, physics
from alife.config import Config, apply_overrides, dims
from alife.state import (
    ERROR_MASK,
    Counter3D,
    CounterIdx,
    DeathCause,
    LedgerIdx,
    State3,
    init_state,
    n_counters,
    to_numpy,
)
from alife.step import StepOut, make_step
from helpers import ledger_net, make_scene, set_output_bias

C = CounterIdx
N3 = Counter3D.Z_FLOOR_CLAMPS  # the first Counter3D index
H15 = LedgerIdx.HEAT_BUOYANCY
f32 = np.float32


def _np(a: mx.array) -> np.ndarray:
    return to_numpy(a)


def _w2(cfg: Config, bias: Mapping[int, float]) -> np.ndarray:
    """A zero brain with output biases: ``y = bias`` whatever the inputs."""
    d = dims(cfg)
    w2 = np.zeros((d.w2_rows, d.lay.D_OUT), np.float32)
    for out, v in bias.items():
        w2[d.h, out] = v
    return w2


def _step(cfg: Config, s: State3) -> tuple[State3, StepOut]:
    s1, out = make_step(cfg, compile=False)(s)
    mx.eval(s1, out)
    total0, total1 = int(_np(s.total)), int(_np(s1.total))
    assert total1 - total0 == ledger_net(out.ledger, cfg)  # every scene closes with slot 15 as a sink
    return s1, out


def _upkeep(cfg: Config, cap: float) -> int:
    return int(np.floor(f32(cfg.energy.buoyancy_cap_cost) * f32(cap)))


def _idle_action_due(cfg: Config) -> int:
    """A zero brain's action due: grip, harvest and defend are ``sigmoid(0) = 0.5``, everything else 0."""
    ec = cfg.energy
    return sum(int(np.floor(f32(c) * f32(0.5))) for c in (ec.grip_cost, ec.harvest_cost, ec.defend_cost))


# --- combat and corpses -------------------------------------------------------------------------------------


def _attack_scene(cfg: Config, z: float) -> State3:
    """Three attackers on a horizontal ring of radius 2 around a poor victim at height ``z``."""
    w2 = _w2(cfg, {layout.OUT_ATTACK: 4.0})  # the victim is every attacker's only target
    centre = np.array([24.0, 24.0])
    ring = [centre + 2.0 * np.array([np.cos(a), np.sin(a)]) for a in (0.0, 2.0 * np.pi / 3, 4.0 * np.pi / 3)]
    attackers = [dict(pos=(*xy, z), energy=50_000, W2=w2, birth_step=-50) for xy in ring]
    victim = dict(pos=(*centre, z), energy=1_000, birth_step=-50, tag=(0.5, 0.0, 0.0, 0.0))
    return make_scene(cfg, [victim, *attackers], resource=0)


def test_attack_drains_a_poor_victim_in_3d(micro3d_cfg: Config) -> None:
    cfg = micro3d_cfg
    s = _attack_scene(cfg, 12.0)
    s1, out = _step(cfg, s)
    c, led = _np(out.counters), _np(out.ledger)
    assert c[C.ATTACKS] == 3 and c[C.ATTACK_HITS] == 3 and c[C.DEATHS_KILLED] == 1
    assert not bool(_np(s1.alive)[0])
    dv = _np(out.deaths.valid)
    assert _np(out.deaths.cause)[dv].tolist() == [DeathCause.KILLED]
    sim = float(np.exp(f32(-0.25 / cfg.sensing.tag_scale**2)))  # the 2D kin counters: a level ring
    assert c[C.KIN_ATTACK_SUM] == 3 * round(256 * sim)
    assert c[C.KIN_ATTACK_BASE] == 3 * round(256 * (sim + 2.0) / 3.0)
    assert led[LedgerIdx.FLOW_STOLEN] == led[LedgerIdx.FLOW_ATTACK_GAIN] + led[LedgerIdx.FLOW_SPILL] > 0
    assert led[LedgerIdx.FLOW_CORPSE] == cfg.energy.structure
    assert led[H15] == 4 * _upkeep(cfg, cfg.mutation.buoy_init)  # four caps at buoy_init, no lift


def test_a_corpse_sinks_to_the_floor_cell(micro3d_cfg: Config) -> None:
    """Without regeneration and diffusion the step's only field changes are the corpse in the floor cell below
    the victim and the combat spill and residue in the victim's own cell (§27.6, docs/3D.md §12 Q19)."""
    cfg = apply_overrides(micro3d_cfg, ["fields.regen_max=0", "fields.resource_diffusion_q8=0"])
    fc = cfg.world.field_cell
    s = _attack_scene(cfg, 13.0)
    s1, out = _step(cfg, s)
    led = _np(out.ledger)
    r = _np(s1.resource).astype(np.int64)
    cx = cy = int(24.0 // fc)
    cz = int(13.0 // fc)
    assert cz == 3 and led[LedgerIdx.INJECT_REGEN] == 0 and led[LedgerIdx.FLOW_HARVEST] == 0
    assert r[0, cy, cx] == led[LedgerIdx.FLOW_CORPSE] == cfg.energy.structure
    assert r[cz, cy, cx] == led[LedgerIdx.FLOW_SPILL] > 0
    assert r.sum() == r[0, cy, cx] + r[cz, cy, cx]


def test_harvest_takes_from_the_agents_own_cell(micro3d_cfg: Config) -> None:
    """A mid-water agent harvests its own 3D cell, not the floor cell below it, and a grounded agent its floor
    cell: harvest is not gated on grounding (§27.6, docs/3D.md §12)."""
    cfg = apply_overrides(micro3d_cfg, ["fields.regen_max=0", "fields.resource_diffusion_q8=0"])
    d = dims(cfg)
    w2 = _w2(cfg, {layout.OUT_HARVEST: 4.0})
    mid = dict(pos=(24.0, 24.0, 13.0), W2=w2, birth_step=-50)
    floor = dict(pos=(10.0, 10.0, d.z_lo), W2=w2, birth_step=-50)
    own, below, ground = (3, 6, 6), (0, 6, 6), (0, 2, 2)  # field cell 4: (24, 24, 13) and (10, 10, z_lo)
    r0 = np.zeros(d.field_shape, np.int32)
    for cell in (own, below, ground):
        r0[cell] = 5_000
    s = make_scene(cfg, [mid, floor], resource=r0)
    assert _np(physics.grounded(s.pos, cfg))[:2].tolist() == [False, True]
    s1, out = _step(cfg, s)
    led, r1 = _np(out.ledger), _np(s1.resource)
    take = int(np.floor(f32(cfg.energy.harvest_max) / (f32(1.0) + np.exp(f32(-4.0)))))  # sigmoid(4)·h_max
    assert take == 392 and led[LedgerIdx.INJECT_REGEN] == 0 and led[LedgerIdx.FLOW_HARVEST] == 2 * take
    assert r1[own] == r1[ground] == 5_000 - take and r1[below] == 5_000
    assert r1.astype(np.int64).sum() == 3 * 5_000 - 2 * take


# --- bonds and buoyancy -------------------------------------------------------------------------------------


def test_a_vertical_pair_bonds(micro3d_cfg: Config) -> None:
    """``b`` 2.5 directly above ``a``: they bond (one pair, two proposers), and the spring pulls along z."""
    cfg = micro3d_cfg
    w2 = _w2(cfg, {layout.OUT_BOND: 4.0})
    a = dict(pos=(20.0, 20.0, 10.0), W2=w2, birth_step=-50, tag=(0.5, 0.0, 0.0, 0.0))
    b = dict(pos=(20.0, 20.0, 12.5), W2=w2, birth_step=-50)
    s = make_scene(cfg, [a, b], resource=0)
    s1, out = _step(cfg, s)
    c = _np(out.counters)
    sim = float(np.exp(f32(-0.25 / cfg.sensing.tag_scale**2)))
    assert c[C.BONDS_FORMED] == 1 and c[C.BONDS_TOTAL] == 1 and c[C.BOND_PROPOSERS] == 2
    assert c[C.KIN_BOND_SUM] == 2 * round(256 * sim)
    assert _np(s1.partner)[0, 0] == 1 and _np(s1.partner)[1, 0] == 0
    p0, p1 = _np(s.pos), _np(s1.pos)
    np.testing.assert_array_equal(p1[:2, :2], p0[:2, :2])  # nothing acts on x and y
    pull = 2.0 * cfg.bonds.k_spring * (2.5 - cfg.bonds.rest_length) / cfg.physics.drag  # both ends, at rest
    np.testing.assert_allclose(p1[1, 2] - p1[0, 2], 2.5 - pull, atol=1e-5)


def test_pitch_steers_and_pays_the_turn_cost(micro3d_cfg: Config) -> None:
    """``OUT_PITCH`` changes the stored pitch by ``turn_max·tanh(y)``, and the turn term charges
    ``|turn| + |Δφ|`` (§27.7)."""
    cfg = micro3d_cfg
    ph, ec, lay = cfg.physics, cfg.energy, dims(cfg).lay
    w2 = _w2(cfg, {layout.OUT_TURN: 4.0, lay.OUT_PITCH: 4.0})
    s = make_scene(cfg, [dict(pos=(24.0, 24.0, 12.0), W2=w2, birth_step=-50)], resource=0)
    s1, out = _step(cfg, s)
    dphi = f32(ph.turn_max) * np.tanh(f32(4.0))
    np.testing.assert_allclose(_np(s1.pitch)[0], dphi, rtol=0, atol=1e-6)
    np.testing.assert_allclose(_np(s1.heading)[0], dphi, rtol=0, atol=1e-6)
    turn = int(np.floor(f32(ec.turn_cost) * (dphi + dphi) / f32(ph.turn_max)))
    assert turn == 19 and _np(out.ledger)[LedgerIdx.HEAT_ACTION] == _idle_action_due(cfg) + turn


def test_buoyant_lift_and_its_cost(micro3d_cfg: Config) -> None:
    """A full cap at full ``OUT_BUOYANCY`` and no thrust rises at ``(buoyancy_max·b − gravity)/drag`` and pays
    ``floor(buoyancy_cost·b²) + floor(buoyancy_cap_cost)`` into ``HEAT_BUOYANCY`` (float32 reference)."""
    cfg = micro3d_cfg
    ph, ec = cfg.physics, cfg.energy
    lay = dims(cfg).lay
    assert ph.buoyancy_max > ph.gravity
    agent = dict(pos=(24.0, 24.0, 12.0), buoy_cap=1.0, W2=_w2(cfg, {lay.OUT_BUOYANCY: 4.0}), birth_step=-50)
    s = make_scene(cfg, [agent], resource=0)
    s1, out = _step(cfg, s)
    b = np.maximum(np.tanh(f32(4.0)), f32(0.0))
    due = int(np.floor(f32(ec.buoyancy_cost) * (f32(1.0) * b) ** 2)) + _upkeep(cfg, 1.0)
    led = _np(out.ledger)
    assert led[H15] == due == 23
    e0, e1 = int(_np(s.energy)[0]), int(_np(s1.energy)[0])
    assert e0 - e1 == led[LedgerIdx.HEAT_BASAL] + led[LedgerIdx.HEAT_ACTION] + due
    vz = (f32(ph.buoyancy_max) * f32(1.0) * b - f32(ph.gravity)) / f32(ph.drag)
    assert vz > 0
    np.testing.assert_allclose(_np(s1.pos)[0, 2], f32(12.0) + vz, rtol=0, atol=1e-6)
    np.testing.assert_allclose(_np(s1.vel)[0], (0.0, 0.0, vz), rtol=0, atol=1e-6)


def test_a_zero_cap_lifts_nothing_and_the_upkeep_needs_no_lift(micro3d_cfg: Config) -> None:
    """Cap 0 at full ``OUT_BUOYANCY`` sinks like an agent without the actuator and pays nothing for it; cap
    0.5 without lift sinks the same way and pays only the upkeep."""
    cfg = micro3d_cfg
    lay, ph = dims(cfg).lay, cfg.physics
    lift = _w2(cfg, {lay.OUT_BUOYANCY: 4.0})
    zero_cap = dict(pos=(10.0, 10.0, 12.0), buoy_cap=0.0, W2=lift, birth_step=-50)
    no_lift = dict(pos=(30.0, 30.0, 12.0), buoy_cap=0.5, birth_step=-50)
    s = make_scene(cfg, [zero_cap, no_lift], resource=0)
    s1, out = _step(cfg, s)
    assert _np(out.ledger)[H15] == _upkeep(cfg, 0.5) == 4
    paid = _np(s.energy)[:2].astype(np.int64) - _np(s1.energy)[:2]
    basal = cfg.energy.basal_cost + _np(s.brain_cost)[:2]
    np.testing.assert_array_equal(paid, basal + _idle_action_due(cfg) + np.array([0, 4]))
    sink = f32(12.0) + (f32(0.0) - f32(ph.gravity)) / f32(ph.drag)
    assert (_np(s1.pos)[:2, 2] == sink).all()


def test_the_surface_clamps_without_grounding(micro3d_cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    """A buoyant agent at ``z_hi`` stays there with ``v_z = 0`` (a surface clamp) and is not grounded; an
    agent on the floor is (``IN_GROUNDED``, ``GROUNDED``), and the brain sees both heights (``IN_HEIGHT``)."""
    cfg = micro3d_cfg
    d = dims(cfg)
    seen: dict[str, Any] = {}
    forward = brain.forward

    def spy(W1: mx.array, W2: mx.array, x: mx.array, h: mx.array, c: Config) -> Any:
        seen["x"] = x
        return forward(W1, W2, x, h, c)

    monkeypatch.setattr(brain, "forward", spy)
    top = dict(pos=(24.0, 24.0, d.z_hi), buoy_cap=1.0, W2=_w2(cfg, {d.lay.OUT_BUOYANCY: 4.0}), birth_step=-50)
    floor = dict(pos=(10.0, 10.0, d.z_lo), birth_step=-50)
    s = make_scene(cfg, [top, floor], resource=0)
    assert _np(physics.grounded(s.pos, cfg))[:2].tolist() == [False, True]
    s1, out = _step(cfg, s)
    c = _np(out.counters)
    assert c[N3:].tolist() == [1, 1, 1]  # Z_FLOOR_CLAMPS, Z_SURFACE_CLAMPS, GROUNDED
    assert _np(s1.pos)[0, 2] == f32(d.z_hi) and _np(s1.vel)[0, 2] == 0.0
    assert _np(s1.pos)[1, 2] == f32(d.z_lo) and _np(s1.vel)[1, 2] == 0.0
    x = _np(seen["x"])
    assert x[:2, d.lay.IN_GROUNDED].tolist() == [0.0, 1.0]
    height = np.array([d.z_hi, d.z_lo]) / cfg.world.height
    np.testing.assert_allclose(x[:2, d.lay.IN_HEIGHT], height, atol=1e-7)


# --- the whole step: 400 compiled steps of tiny3d (§22.5 in 3D) --------------------------------------------

STEPS = 400

# Output biases for the founders, as ``test_affordances.BIAS``: harvest, bond, share, reproduce (bonded),
# attack and emit. Pitch, thrust and the buoyancy use stay with the founders' random brains.
BIAS = {
    layout.OUT_HARVEST: 2.0,
    layout.OUT_BOND: 2.0,
    **{layout.OUT_SHARE + b: 1.0 for b in range(layout.B)},
    layout.OUT_REPRODUCE: 2.0,
    layout.OUT_CHILD_BOND: 2.0,
    layout.OUT_ATTACK: 1.0,
    **{layout.OUT_EMIT + c: 1.0 for c in range(layout.NCH)},
}


@pytest.fixture(scope="module")
def run3d(tiny3d_cfg: Config) -> dict[str, Any]:
    """``tiny3d`` for ``STEPS`` compiled steps with the pinned seed, founders biased as in ``BIAS``.

    Besides the totals it records, step by step: the height and pitch ranges of the alive agents; the
    horizontal (x and y, minimum image) and the vertical distance moved by the agents that lived through the
    step; the horizontal distance of those that started it grounded (gravity cannot move them, and the floor
    holds them); the number of agent-steps in which a surviving agent's pitch changed; the chemicals per
    channel and layer after the second step (``chem_early``, (NCH, FZ)); and the parent of each child.
    """
    cfg = tiny3d_cfg
    assert cfg.run.seed == 0 and cfg.world.dims == 3
    L = cfg.world.size
    s = set_output_bias(init_state(cfg), cfg, BIAS)
    fn = make_step(cfg)
    totals = np.zeros(n_counters(3), np.int64)
    ledger = np.zeros(len(LedgerIdx), np.int64)
    parent_of: dict[int, int] = {}
    r = dict(
        xy=0.0,
        xy_grounded=0.0,
        z=0.0,
        sink_steps=0,
        pitch_moves=0,
        z_lo=np.inf,
        z_hi=-np.inf,
        p_lo=np.inf,
        p_hi=-np.inf,
    )
    chem_early = np.zeros((0, 0))
    for step in range(STEPS):
        prev_pos, prev_alive, prev_pitch, prev_uid = _np(s.pos), _np(s.alive), _np(s.pitch), _np(s.uid)
        was_grounded = _np(physics.grounded(s.pos, cfg))
        s, out = fn(s)
        mx.eval(s, out)
        totals += _np(out.counters)
        ledger += _np(out.ledger)
        bv = _np(out.births.valid)
        parent_of.update(
            zip(_np(out.births.uid)[bv].tolist(), _np(out.births.parent_uid)[bv].tolist(), strict=True)
        )
        pos, alive = _np(s.pos), _np(s.alive)
        still = prev_alive & alive
        dxy = pos[still, :2] - prev_pos[still, :2]
        dxy -= L * np.round(dxy / L)  # x and y are periodic
        r["xy"] += float(np.abs(dxy).sum())
        r["xy_grounded"] += float(np.abs(dxy[was_grounded[still]]).sum())
        dz = pos[still, 2] - prev_pos[still, 2]
        r["z"] += float(np.abs(dz).sum())
        r["sink_steps"] += int((dz < 0).sum())
        r["z_lo"], r["z_hi"] = min(r["z_lo"], pos[alive, 2].min()), max(r["z_hi"], pos[alive, 2].max())
        pitch_all = _np(s.pitch)
        same = still & (_np(s.uid) == prev_uid)  # the same agent, not a child that took a freed row
        r["pitch_moves"] += int((pitch_all[same] != prev_pitch[same]).sum())
        pitch = pitch_all[alive]
        r["p_lo"], r["p_hi"] = min(r["p_lo"], pitch.min()), max(r["p_hi"], pitch.max())
        if step == 1:  # two steps of diffusion reach two layers up: higher chemicals were deposited there
            chem_early = _np(s.chem).sum(axis=(2, 3))
    return {
        "state": s,
        "totals": totals,
        "ledger": ledger,
        "parent_of": parent_of,
        "chem_early": chem_early,
        **r,
    }


def test_every_3d_affordance_is_exercised(tiny3d_cfg: Config, run3d: dict[str, Any]) -> None:
    t, led, s = run3d["totals"], run3d["ledger"], run3d["state"]
    assert t[C.BIRTHS] > 0 and t[C.DEATHS_STARVED] + t[C.DEATHS_KILLED] > 0  # REPRODUCE, DIE
    assert t[C.ATTACK_HITS] > 0 and led[LedgerIdx.FLOW_STOLEN] > 0  # FIGHT
    assert t[C.BONDS_FORMED] > 0 and t[C.BONDS_BROKEN] > 0 and t[C.CHILD_BONDS] > 0  # BOND, SPLIT
    assert t[C.SHARERS] > 0 and led[LedgerIdx.FLOW_SHARED] > 0 and led[LedgerIdx.FLOW_HARVEST] > 0
    # the 3D terms: the floor and the surface hold agents, the buoyancy actuator is paid for
    assert t[Counter3D.Z_FLOOR_CLAMPS] > 0 and t[Counter3D.Z_SURFACE_CLAMPS] > 0 and t[Counter3D.GROUNDED] > 0
    assert led[H15] > 0 and led[H15] == led[LedgerIdx.RESERVED]  # HEAT_BUOYANCY is slot 15
    assert int(_np(s.alive).sum()) > 0
    assert int(_np(s.diag)) & ERROR_MASK == 0
    assert invariants.full_check(s, tiny3d_cfg) == []  # the ledger closes, z and pitch are in range


def test_a_3d_run_stays_between_the_floor_and_the_surface(tiny3d_cfg: Config, run3d: dict[str, Any]) -> None:
    d, s = dims(tiny3d_cfg), run3d["state"]
    assert d.z_lo <= run3d["z_lo"] and run3d["z_hi"] <= d.z_hi  # every alive height, at every step
    assert run3d["z_lo"] == d.z_lo and run3d["z_hi"] == d.z_hi  # and both bounds were reached
    alive, pitch = _np(s.alive), _np(s.pitch)
    assert (pitch[alive] != 0.0).any()  # (the founders start with a random pitch: this alone proves nothing)
    assert run3d["pitch_moves"] > 0  # pitch is used: surviving agents turn, step after step
    assert run3d["p_lo"] >= -f32(layout.PITCH_MAX) and run3d["p_hi"] <= f32(layout.PITCH_MAX)
    grounded = _np(physics.grounded(s.pos, tiny3d_cfg))[alive]
    assert 0 < grounded.sum() < alive.sum()  # some rest on the floor and some swim


def test_signals_reach_the_water(run3d: dict[str, Any]) -> None:
    """Every chemical channel holds a deposit at the end, made where the emitters are: two steps in, only a
    direct deposit can be in the layers three or more above the floor (the chemicals diffuse one layer per
    step), and every channel has some there."""
    s = run3d["state"]
    alive = _np(s.alive)
    assert np.abs(_np(s.msg)[alive]).max() > 0.0  # broadcast channels
    assert _np(s.chem).sum(axis=(1, 2, 3)).min() > 0.0  # every chemical channel holds a deposit in the volume
    early = run3d["chem_early"]
    assert early.shape == (_np(s.chem).shape[0], _np(s.chem).shape[1])
    assert (
        early[:, 3:].sum(axis=1) > 0.0
    ).all()  # deposits in the water above the floor, not on the floor only


def test_reproduction_mutates_the_buoyancy_cap(tiny3d_cfg: Config, run3d: dict[str, Any]) -> None:
    s, parent_of = run3d["state"], run3d["parent_of"]
    alive, uid, cap = _np(s.alive), _np(s.uid), _np(s.buoy_cap)
    row_of = {int(u): int(r) for r, u in enumerate(uid) if alive[r]}
    birth, step = _np(s.birth_step), int(_np(s.step))
    pairs = [
        (row_of[c], row_of[p])
        for c, p in parent_of.items()
        if c in row_of and p in row_of and birth[row_of[c]] < step - 1  # flushed child rows (§12.6)
    ]
    assert pairs, "no living parent-child pair to compare"
    assert any(cap[c] != cap[p] for c, p in pairs)  # the additive mutation of the cap
    assert cap[alive].min() >= 0.0 and cap[alive].max() <= 1.0 and cap[alive].std() > 0.0
    assert np.isclose(cap[alive].mean(), tiny3d_cfg.mutation.buoy_init, atol=0.1)  # no drift in 400 steps


def test_horizontal_movement_is_separate_from_sinking(micro3d_cfg: Config, run3d: dict[str, Any]) -> None:
    """Gravity acts on z alone, so movement in x and y is the agents' own, and so is all of the movement of an
    agent held by the floor. In the run both happen; a passive agent sinks and does not move sideways."""
    assert run3d["xy"] > 0.0 and run3d["xy_grounded"] > 0.0  # swimming, and walking on the floor
    assert run3d["z"] > 0.0 and run3d["sink_steps"] > 0  # vertical movement, downward steps included

    cfg = micro3d_cfg
    ph, d = cfg.physics, dims(cfg)
    passive = [dict(pos=(6.0 * (k + 1), 24.0, 20.0), birth_step=-50) for k in range(6)]  # zero brains
    s = make_scene(cfg, passive, resource=0)
    fn = make_step(cfg, compile=False)
    xy0, z0 = _np(s.pos)[:6, :2].copy(), _np(s.pos)[:6, 2].copy()
    for n in range(1, 41):
        s, out = fn(s)
        mx.eval(s, out)
        np.testing.assert_array_equal(_np(s.pos)[:6, :2], xy0)  # nothing acts on x and y
        np.testing.assert_allclose(_np(s.pos)[:6, 2], z0 - n * ph.gravity / ph.drag, rtol=0, atol=1e-4)
    assert (_np(s.pos)[:6, 2] > d.z_lo).all() and _np(out.counters)[N3] == 0  # still in the water: no clamp
