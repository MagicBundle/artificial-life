"""§27.3–§27.8 3D affordances through one eager step on hand-built ``micro3d`` scenes: an attack drain and
its closure with ``HEAT_BUOYANCY``, a corpse sinking to the floor cell, a vertical bond, the pitch and its
turn cost, buoyant lift and its cost against a float32 reference, a zero cap and the upkeep without lift, the
surface clamp without grounding."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import mlx.core as mx
import numpy as np
import pytest

from alife import brain, layout, physics
from alife.config import Config, apply_overrides, dims
from alife.state import Counter3D, CounterIdx, DeathCause, LedgerIdx, State3, to_numpy
from alife.step import StepOut, make_step
from helpers import ledger_net, make_scene

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
