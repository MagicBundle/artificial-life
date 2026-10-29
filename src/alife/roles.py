"""Committed roles on bonds (DESIGN §40.3 to §40.5): the role machine and its effects on flows the ledger
already books.

A role is a committed state, not an instantaneous output: ``decode_role`` reads two brain outputs through a
dead zone (keep-by-default gives hysteresis), the switch costs ``roles.switch_cost`` (booked in
``HEAT_ACTION`` through ``extra_due``) and the effects ramp up with ``role_dwell``. Every effect multiplies or
caps a flow of the 16-term ledger (the harvest demand, the predation demand, the action cost): there is no
17th term and no new source. The effects read the START-of-step ``role`` and ``role_dwell`` (a step input, no
ordering hazard); a switch becomes visible at the next step.

Pure MLX, RNG-free, gathers over ``(N, B)`` only.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import mlx.core as mx

from alife import layout, ops
from alife.config import Config

GENERALIST, FORAGER, GUARD = 0, 1, 2
_I32, _F32 = mx.int32, mx.float32
RAMP_GUARD_MIN = 128  # a guard's partner counts as guarded (synergy) once its ramp reaches half


class RoleDec(NamedTuple):
    """The decoded role request of a step (``ExtFlows.bonds`` until S12 replaces it with ``BondFlows``)."""

    req: mx.array  # (N,) int8: the requested role (keep-by-default: equals ``role0`` in the dead zone)
    role0: mx.array  # (N,) int8: the start-of-step role


def decode_role(y: mx.array, role: mx.array, cfg: Config, lay: layout.Layout | None = None) -> mx.array:
    """The requested role of every row, (N,) int8 (S7). ``a = tanh(y_a)``, ``b = tanh(y_b)``, dead zone
    ``τ = roles.threshold``:

    ``a > τ, b < −τ`` forager; ``a < −τ, b > τ`` guard; ``a < −τ, b < −τ`` generalist; anything else (the
    dead zone, and the ``a > τ, b > τ`` corner reserved for the deferred germ role) keeps ``role``.
    """
    from alife.config import dims

    lay = dims(cfg).lay if lay is None else lay
    tau = cfg.roles.threshold
    a, b = mx.tanh(y[:, lay.OUT_ROLE_A]), mx.tanh(y[:, lay.OUT_ROLE_B])
    req = mx.where(
        (a > tau) & (b < -tau),
        FORAGER,
        mx.where((a < -tau) & (b > tau), GUARD, mx.where((a < -tau) & (b < -tau), GENERALIST, role)),
    )
    return req.astype(mx.int8)


def decode(s: Any, y: mx.array, ctx: Any, cfg: Config) -> RoleDec:
    """The registry ``decode`` hook (S7). Without ``roles.enabled`` (sharing only, or the inert twin of the
    measures) the request is the current role: nothing ever switches."""
    if not cfg.roles.enabled:  # static
        return RoleDec(s.role, s.role)
    return RoleDec(decode_role(y, s.role, cfg), s.role)


def mask(dec: RoleDec, solvent: mx.array, cfg: Config) -> RoleDec:
    """The registry ``mask`` hook: an insolvent row cannot switch (its cost was due and is paid as far as it
    holds)."""
    return RoleDec(mx.where(solvent, dec.req, dec.role0).astype(mx.int8), dec.role0)


def due(s: Any, act: Any, y: mx.array, cfg: Config) -> mx.array | None:
    """The registry ``due`` hook: ``switch_cost`` quanta for a row that asks for another role, an int32
    (None when switching is free). Booked in ``HEAT_ACTION`` with the other action costs."""
    if cfg.roles.switch_cost == 0 or not cfg.roles.enabled:  # static
        return None
    req = decode_role(y, s.role, cfg)
    return (req != s.role).astype(_I32) * cfg.roles.switch_cost


def ramp_q8(dwell: mx.array, cfg: Config) -> mx.array:
    """``min(256, dwell·256 // ramp_steps)`` as int32: the share of a role's effect that has built up
    since the last switch (``role_dwell`` saturates at ``ramp_steps``)."""
    return mx.minimum(dwell.astype(_I32) * 256 // cfg.roles.ramp_steps, 256)


class Effects(NamedTuple):
    """The role effects of S10 and S11 for one step (all from START-of-step ``role`` and ``role_dwell``)."""

    harvest_cap: mx.array | None  # (N,) int32 per-agent harvest demand cap; None leaves ``harvest_max``
    defend_mult: mx.array | None  # (N,) f32 multiplier of the behavioural ``defend`` seen by ``combat``
    guard: mx.array | None  # (N,) f32 protection fraction of each victim in [0, guard_shield]


def effects(s: Any, act: Any, partner: mx.array, cfg: Config) -> Effects:
    """The harvest cap, the own-defence multiplier and the guard shield of every row.

    * Harvest cap ``hm = H + (H·gF·ramp·F >> 16) + (H·sy·ramp·F·has_guard >> 16) − (H·gp·ramp·G >> 16)``
      with ``H = harvest_max``, ``gF = q8(forage_gain)``, ``sy = q8(synergy)``, ``gp = q8(guard_harvest_pen)``
      (all integer): a forager harvests more, more again beside a guard (``has_guard``: a valid partner is a
      guard with ramp ≥ 128), a guard less. ``hm < 65 536`` (validated) keeps ``proportional_take`` exact.
    * ``defend_mult = 1 − forage_defend_pen·ramp/256`` for a forager (it is exposed), 1 otherwise; it scales
      the ``defend`` that ``combat`` sees, not the one that is paid for or observed.
    * Guard shield: a guard's potential is ``gp_i = ramp_i/256 · defend_i`` (it pays ``defend_cost`` for it),
      zero while the row makes a (gated) attack this step (E16: an attacker's own shield never covers its
      target, so attacking one's bonded partner is not softened by that partner's guard); the victim's fraction is
      ``guard_shield · max(gp_t, max over its valid partners of gp)``: a guard shields itself and each bonded
      partner. ``partner`` is the post-S9 partner.
    """
    rc, en = cfg.roles, cfg.energy
    ramp = ramp_q8(s.role_dwell, cfg)
    is_f, is_g = (s.role == FORAGER).astype(_I32), (s.role == GUARD).astype(_I32)
    hm = None
    if rc.forage_gain > 0 or rc.synergy > 0 or rc.guard_harvest_pen > 0:  # static
        hmax = en.harvest_max
        hm = mx.full(s.role.shape, hmax, dtype=_I32)
        if rc.forage_gain > 0:
            hm = hm + ((hmax * ops.q8(rc.forage_gain)) * (ramp * is_f) >> 16)
        if rc.synergy > 0:
            valid = partner >= 0
            pj = mx.maximum(partner, 0)
            pg = valid & (s.role[pj] == GUARD) & (ramp[pj] >= RAMP_GUARD_MIN)
            has_guard = pg.any(axis=1).astype(_I32)
            hm = hm + ((hmax * ops.q8(rc.synergy)) * (ramp * is_f * has_guard) >> 16)
        if rc.guard_harvest_pen > 0:
            hm = hm - ((hmax * ops.q8(rc.guard_harvest_pen)) * (ramp * is_g) >> 16)
    dm = None
    if rc.forage_defend_pen > 0:  # static
        dm = 1.0 - rc.forage_defend_pen * (ramp.astype(_F32) * (1.0 / 256.0)) * is_f.astype(_F32)
    guard = None
    if rc.guard_shield > 0:  # static
        attacking = act.attack & (act.attack_target >= 0)
        if not cfg.combat.attack_bonded:  # the gated attack: one on a bonded partner is dropped by combat
            attacking = attacking & ~((partner == act.attack_target[:, None]) & (partner >= 0)).any(axis=1)
        gp = mx.where((s.role == GUARD) & ~attacking, ramp.astype(_F32) * (1.0 / 256.0) * act.defend, 0.0)
        pot = mx.maximum(gp, mx.where(partner >= 0, gp[mx.maximum(partner, 0)], 0.0).max(axis=1))
        guard = rc.guard_shield * pot
    return Effects(hm, dm, guard)


def combat_act(act: Any, dm: mx.array | None) -> Any:
    """The actions ``combat`` sees: ``defend`` scaled by the own-defence multiplier (cost and observer channel
    keep the raw output). ``act`` itself when there is no multiplier (static)."""
    return act if dm is None else act._replace(defend=act.defend * dm)


def shielded_by_guard(guard: mx.array | None, attack: mx.array, target: mx.array) -> mx.array | None:
    """``SHIELDED_BY_GUARD``: the attacks (post-gate ``Combat.attack``) whose victim has a guard fraction > 0,
    () int32; None without a guard shield (static)."""
    if guard is None:
        return None
    return (attack & (guard[mx.maximum(target, 0)] > 0.0)).astype(_I32).sum()
