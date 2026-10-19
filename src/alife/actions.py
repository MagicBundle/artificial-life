"""Decoding brain outputs into actions, including query-based target selection (§7). Owned by WP2."""

from __future__ import annotations

from typing import NamedTuple

import mlx.core as mx

from alife import layout
from alife.config import Config, dims
from alife.spatial import Neighbors

_F32 = mx.float32
_I32 = mx.int32


class Actions(NamedTuple):
    """Decoded actions, all (N, ...) and already masked by ``alive``."""

    thrust: mx.array  # (N,) f32 in (−1, 1)
    turn: mx.array  # (N,) f32 radians
    msg: mx.array  # (N,MSG) f32
    emit: mx.array  # (N,NCH) f32 in [0, 1] (tanh reaches 1.0 in float32)
    harvest: mx.array  # (N,) f32 in (0, 1)
    grip: mx.array  # (N,) f32 in (0, 1)
    contract: mx.array  # (N,) f32 in (−1, 1)
    bond_want: mx.array  # (N,) bool
    release: mx.array  # (N,B) bool
    share: mx.array  # (N,B) f32 in [0, 1)
    attack: mx.array  # (N,) bool
    defend: mx.array  # (N,) f32 in (0, 1)
    reproduce: mx.array  # (N,) bool
    child_bond: mx.array  # (N,) bool
    bond_k: mx.array  # (N,) int32, −1 = none
    bond_target: mx.array  # (N,) int32, −1 = none
    attack_k: mx.array  # (N,) int32, −1 = none
    attack_target: mx.array  # (N,) int32, −1 = none


class Motion3(NamedTuple):
    """The 3D-only decoded actions (§27.2), (N,) each and masked like ``Actions``; a parallel type, so
    ``Actions`` keeps its fields."""

    pitch: mx.array  # (N,) f32 radians in (−turn_max, turn_max): pitch change (up is positive)
    buoy: mx.array  # (N,) f32 in [0, 1): buoyancy fraction


def decode(
    y: mx.array, desc: mx.array, nbr: Neighbors, partner: mx.array, alive: mx.array, cfg: Config
) -> Actions:
    """Decode raw logits ``y (N,D_OUT)`` and select bond/attack targets (§7).

    Targets are the argmax of ``φ_k · q`` over eligible neighbours (``φ = desc[..., 1:]``); ties go to the
    lowest k, i.e. the nearest (first-index argmax, F5). With nothing eligible, ``k = target = −1`` and
    ``attack`` is False. The result is masked by ``alive``. In 3D (§27.2) ``φ`` ends with ``NBR_DZ`` and each
    query is its 2D block of six followed by its ``*_DZ`` column, in the order of ``φ``; the 3D-only outputs
    are decoded by ``decode_motion3``.
    """
    d = dims(cfg)
    y = y.astype(_F32)
    pc, bc, cc = cfg.physics, cfg.bonds, cfg.combat

    def cols(start: int, width: int) -> mx.array:
        return y[:, start : start + width]

    def query(start: int, dz: int | None) -> mx.array:
        q = cols(start, layout.F)  # the 2D block (layout.F = 6 in both dims)
        return q if dz is None else mx.concatenate([q, cols(dz, 1)], axis=1)  # dz is None in 2D

    bond_want = y[:, layout.OUT_BOND] > 0
    has_free = (partner < 0).any(1)
    can_bond = alive & bond_want & has_free
    seen = nbr.valid & alive[nbr.idx]  # alive[idx] is a no-op under the §4 contract, kept as a guard
    bonded = desc[..., layout.NBR_BONDED] > 0.5
    phi = desc[..., 1:]
    elig_b = seen & (nbr.dist < bc.form_range) & ~bonded & can_bond[nbr.idx]
    elig_a = seen & (nbr.dist < cc.range)
    if not cc.attack_bonded:
        elig_a = elig_a & ~bonded
    q_bond = query(layout.OUT_Q_BOND, d.lay.OUT_Q_BOND_DZ)
    bond_k, bond_target = _select(elig_b, phi, q_bond, bond_want, nbr.idx)
    attack_k, attack_target = _select(
        elig_a,
        phi,
        query(layout.OUT_Q_ATTACK, d.lay.OUT_Q_ATTACK_DZ),
        y[:, layout.OUT_ATTACK] > cc.threshold,
        nbr.idx,
    )

    act = Actions(
        thrust=mx.tanh(y[:, layout.OUT_THRUST]),
        turn=pc.turn_max * mx.tanh(y[:, layout.OUT_TURN]),
        msg=mx.tanh(cols(layout.OUT_MSG, layout.MSG)),
        emit=mx.maximum(mx.tanh(cols(layout.OUT_EMIT, layout.NCH)), 0.0),
        harvest=mx.sigmoid(y[:, layout.OUT_HARVEST]),
        grip=mx.sigmoid(y[:, layout.OUT_GRIP]),
        contract=mx.tanh(y[:, layout.OUT_CONTRACT]),
        bond_want=bond_want,
        release=cols(layout.OUT_RELEASE, layout.B) > bc.release_threshold,
        share=mx.maximum(mx.tanh(cols(layout.OUT_SHARE, layout.B)), 0.0),
        attack=attack_k >= 0,
        defend=mx.sigmoid(y[:, layout.OUT_DEFEND]),
        reproduce=y[:, layout.OUT_REPRODUCE] > 0,
        child_bond=y[:, layout.OUT_CHILD_BOND] > 0,
        bond_k=bond_k,
        bond_target=bond_target,
        attack_k=attack_k,
        attack_target=attack_target,
    )
    return mask(act, alive)


def decode_motion3(y: mx.array, alive: mx.array, cfg: Config) -> Motion3:
    """Decode the 3D-only outputs of ``y (N,39)`` (§27.2), masked by ``alive``.

    ``pitch = turn_max·tanh(y[OUT_PITCH])`` (the change of φ; the clip to ±PITCH_MAX is the integrator's) and
    ``buoy = max(tanh(y[OUT_BUOYANCY]), 0)``, like EMIT and SHARE, so zero buoyancy is exactly reachable.
    """
    d = dims(cfg)
    assert d.d3, "decode_motion3 is 3D only"  # static: the 3D output indices are None in 2D
    y = y.astype(_F32)
    mo = Motion3(
        pitch=cfg.physics.turn_max * mx.tanh(y[:, d.lay.OUT_PITCH]),
        buoy=mx.maximum(mx.tanh(y[:, d.lay.OUT_BUOYANCY]), 0.0),
    )
    return mask(mo, alive)


def _select(
    elig: mx.array, phi: mx.array, q: mx.array, want: mx.array, idx: mx.array
) -> tuple[mx.array, mx.array]:
    """``(k, target)`` of the best eligible neighbour by score ``φ·q`` where ``want``; −1 where none."""
    score = mx.where(elig, (phi * q[:, None, :]).sum(-1), -mx.inf)
    k = mx.where(elig.any(1) & want, mx.argmax(score, axis=1).astype(_I32), -1)
    target = mx.take_along_axis(idx, mx.maximum(k, 0)[:, None], axis=1)[:, 0]
    return k, mx.where(k >= 0, target, -1)


def mask[A: (Actions, Motion3)](act: A, keep: mx.array) -> A:
    """Cancel actions where ``~keep``: bools → False, floats → 0, k/targets → −1."""

    def cancel(a: mx.array) -> mx.array:
        k = keep.reshape(keep.shape + (1,) * (a.ndim - keep.ndim))
        if a.dtype == mx.bool_:
            return a & k
        if mx.issubdtype(a.dtype, mx.integer):
            return mx.where(k, a, -1)
        return mx.where(k, a, 0.0)

    return type(act)(*(cancel(a) for a in act))
