"""Communication: the voice channel (DESIGN §39).

An optional channel beside the legacy 2-number ``msg``: W call intensities per organism, each in ``[0, 1)``
(zero is exactly reachable), stored in the extra columns of ``State.msg`` (``Dims.msg_w``). A listener hears
the K sensed neighbours it already has (no new search): a linear falloff to zero at ``comm.range``, optionally
weighted by kin similarity, superposed per channel (saturating at 1), plus the body-frame bearing of the
sound. Both are an appended brain tail (``ext.comm``); the next call is the W appended outputs.

Speaking costs ``floor(comm.cost · Σ_c sig_c + u)`` quanta, ``u`` a deterministic dither in ``[0, 1)`` of the
row's ``uid`` and the step (E17: a call below one quantum is not free, its mean cost is exact). The due joins
the action due through ``extra_due``, so it is booked in ``HEAT_ACTION`` and the 16-term ledger closes with no
new term. The two engine controls ``comm.control = deaf | decoy`` change only what the listeners hear (integer
arithmetic and the comm hash salt, no RNG stream): costs, speaking and motion stay on the same code path.

With ``calls.trigger`` (DESIGN §47) each channel belongs to one event and is gated by that event's window
(``gate_of``): a closed channel is silent and costs nothing. Nothing else of this module changes.

Only ``rng.HASH_SALTS["comm"]`` is used here (E18): the dither and the decoy shift are hashes of it.
"""

from __future__ import annotations

from typing import Any, Final, NamedTuple

import mlx.core as mx

from alife import layout, ops, rng
from alife.config import Config, dims

SALT: Final = rng.HASH_SALTS["comm"]  # ops.hash32 salt of the comm feature (DESIGN §43.3)
SIG_MAX: Final = 1.0 - 2.0**-24  # the largest f32 below 1: tanh saturates to exactly 1.0, a call stays < 1
BEARING_FLOOR: Final = 1e-3  # denominator floor of the bearing (no sound -> exactly 0)
ORACLE_INTENSITY: Final = 0.5  # the fixed call of an open window under calls.control = reflex | reflex_near
_I32 = mx.int32
_F32 = mx.float32


class Voice(NamedTuple):
    """The decoded voice of a step (``ExtFlows.comm``)."""

    sig: mx.array  # (N, W) f32 in [0, 1): what each row says this step (0 for dead or, after mask, insolvent)
    due: mx.array  # (N,) int32: the dithered voice due in quanta (0 for dead or, after mask, insolvent)
    # observer channel only (None in the unobserved step): what the row heard, the inputs of this step
    heard: mx.array | None = None  # (N, W) f32 in [0, 1]
    audible: mx.array | None = None  # (N,) int32: neighbours with a hearing weight > 0


class CommActs(NamedTuple):
    """``Acts.ext.comm`` (observer channel, DESIGN §39.7)."""

    voice: mx.array  # (N, W) uint8: min(255, floor(256·sig)), what was said this step
    heard: mx.array  # (N, W) uint8: round(255·heard) of the inputs that drove this step's decision
    audible: mx.array  # (N,) uint8: neighbours with a hearing weight > 0 (saturating at 255)


def dither(uid: mx.array, step: mx.array) -> mx.array:
    """A deterministic uniform in ``[0, 1)`` per row from the ``uid`` and the step (24 bits of the hash)."""
    h = ops.hash32(uid, step, SALT)
    return mx.right_shift(h, mx.array(8, dtype=mx.uint32)).astype(_F32) * (1.0 / (1 << 24))


def sig_of(y: mx.array, cfg: Config, gate: mx.array | None = None) -> mx.array:
    """``max(tanh(y[OUT_VOICE:+W]), 0)`` clipped below 1: the call intensities (N, W), not masked. With a
    ``gate`` ((N, W) bool, the open event windows of ``calls.trigger``, DESIGN §47.2) a closed channel is
    exactly 0: silent and free (edge case K1)."""
    d = dims(cfg)
    o = d.lay.OUT_VOICE
    if gate is not None and cfg.calls.control in (
        "reflex",
        "reflex_near",
    ):  # static: the oracle (DESIGN §47.7)
        return mx.where(gate, ORACLE_INTENSITY, 0.0).astype(_F32)  # every open window calls at 0.5, y ignored
    sig = mx.clip(mx.tanh(y[:, o : o + d.voice_w]), 0.0, SIG_MAX)
    return sig if gate is None else mx.where(gate, sig, 0.0)


def gate_of(s: Any, cfg: Config) -> mx.array | None:
    """The calls gate of the step that starts from ``s``: the open windows ``s.evt > 0`` (N, W) with
    ``calls.trigger``, else None (the traced graph is then exactly the ungated one)."""
    return (s.evt > 0) if cfg.calls.trigger else None  # static


def due_of(uid: mx.array, step: mx.array, sig: mx.array, cfg: Config) -> mx.array:
    """The integer voice due per row, ``floor(cost·Σ sig + u)``: exactly 0 for silence (``u < 1``), exact on
    average for a sub-unit call (E17). int32, not masked (``pay_costs`` masks dead rows)."""
    x = cfg.comm.cost * sig.sum(axis=1)
    return ops.floor_i32(x + dither(uid, step))


def decode(s: Any, y: mx.array, ctx: Any, cfg: Config) -> Voice:
    """The registry ``decode`` hook (S7): the calls of the alive rows and their dues."""
    alive = s.alive
    sig = mx.where(alive[:, None], sig_of(y, cfg, gate_of(s, cfg)), 0.0)
    due = mx.where(alive, due_of(s.uid, s.step, sig, cfg), 0).astype(_I32)
    return Voice(sig, due)


def mask(dec: Voice, solvent: mx.array, cfg: Config) -> Voice:
    """The registry ``mask`` hook (after S8): an insolvent row pays everything it holds and does not speak."""
    return dec._replace(
        sig=mx.where(solvent[:, None], dec.sig, 0.0), due=mx.where(solvent, dec.due, 0).astype(_I32)
    )


def due(s: Any, act: Any, y: mx.array, cfg: Config) -> mx.array | None:
    """The registry ``due`` hook (S8): the voice due, an int32 (None when speaking is free). Booked in
    ``HEAT_ACTION`` with the other action costs."""
    if cfg.comm.cost == 0.0:  # static
        return None
    sig = mx.where(s.alive[:, None], sig_of(y, cfg, gate_of(s, cfg)), 0.0)
    return mx.where(s.alive, due_of(s.uid, s.step, sig, cfg), 0).astype(_I32)


def decoy_index(step: mx.array, alive: mx.array) -> mx.array:
    """The row each row's voice is taken from under ``comm.control = decoy``, a cyclic shift in alive-rank
    space: the alive rows in row order are ranked, and the row of rank ``r`` hears the row of rank
    ``(r + shift) % n_alive`` with ``shift = 1 + hash(step) % (n_alive - 1)``. A bijection on the alive rows,
    never the identity for ``n_alive >= 2``, changing every step; a dead row maps to itself (it is silent and
    nobody hears it). Shifting over all N rows would deliver mostly silence when alive << N (a partial deaf).
    Integer arithmetic and the comm salt only: no RNG stream."""
    n = alive.shape[0]
    rows = mx.arange(n, dtype=_I32)
    a = alive.astype(_I32)
    n_alive = mx.maximum(a.sum(), 1)
    order = mx.argsort((1 - a) * n + rows)  # alive rows first, in row order
    rank = mx.cumsum(a) - 1
    shift = 1 + (ops.hash32(step, 0, SALT) % mx.maximum(n_alive - 1, 1).astype(mx.uint32)).astype(_I32)
    src = order[(rank + shift) % n_alive]
    return mx.where(alive, src, rows)


def weights(nbr: Any, desc: mx.array, cfg: Config) -> mx.array:
    """(N, K) f32: the hearing weight ``a = clip(1 - dist/range, 0, 1) ** atten_pow`` of each sensed neighbour
    (times ``kin ** kin_exp`` with ``kin_exp > 0``), 0 where invalid. Shared by ``hear``,
    ``audible_count`` and the near oracle of ``calls`` (``calls.control = reflex_near``)."""
    c, d = cfg.comm, dims(cfg)
    valid = desc[..., layout.NBR_PRESENT] > 0.5  # desc is zero where invalid or dead
    a = mx.clip(1.0 - nbr.dist / d.ext("comm").hear_range, 0.0, 1.0)
    if c.atten_pow != 1.0:  # static
        a = a**c.atten_pow
    a = mx.where(valid, a, 0.0)
    if c.kin_exp > 0.0:  # static
        kin = desc[..., layout.NBR_KIN]
        a = a * (kin if c.kin_exp == 1.0 else kin**c.kin_exp)
    return a


def hear(voice: mx.array, nbr: Any, desc: mx.array, cfg: Config) -> mx.array:
    """What each row hears: ``(N, W + nd)`` f32, the heard intensity per channel in ``[0, 1]`` then the
    body-frame bearing of the sound in ``[-1, 1]`` (0 with no sound).

    ``voice (N, W)`` is the stored call of every row (a pure argument, so the offline counterfactuals call
    this with a modified array). Neighbour ``j`` of ``i`` weighs
    ``a = clip(1 - dist/range, 0, 1) ** atten_pow`` (times ``kin ** kin_exp`` with ``kin_exp > 0``), 0 where
    invalid; the intensity is the saturating sum of ``a · voice_j``, the bearing the loudness-weighted mean of
    the neighbours' body-frame offsets (``desc``)."""
    d = dims(cfg)
    a = weights(nbr, desc, cfg)
    sig = voice[nbr.idx]  # (N, K, W): a gather, no scatter
    heard = mx.minimum((a[..., None] * sig).sum(axis=1), 1.0)
    w = a * sig.sum(axis=-1)  # loudness of each source at the listener
    rel = desc[..., layout.NBR_DX : layout.NBR_DY + 1]
    if d.d3:  # static: (fwd, lat, up)
        rel = mx.concatenate([rel, desc[..., d.lay.NBR_DZ : d.lay.NBR_DZ + 1]], axis=-1)
    bearing = (w[..., None] * rel).sum(axis=1) / mx.maximum(w.sum(axis=1), BEARING_FLOOR)[:, None]
    return mx.concatenate([heard, bearing], axis=1)


def ext_inputs(s: Any, cfg: Config, ctx: Any) -> tuple[mx.array, mx.array]:
    """The registry ``inputs`` hook (S5): ``(heard (N, W), bearing (N, nd))`` of the stored voice of the
    previous step. ``deaf`` hears nothing; ``decoy`` hears the voice of another row (``decoy_index``)."""
    d = dims(cfg)
    w = d.voice_w
    if cfg.comm.control == "deaf":  # static
        z = mx.zeros((d.n, w + d.nd), _F32)
        return z[:, :w], z[:, w:]
    voice = s.msg[:, layout.MSG :]
    if cfg.comm.control == "decoy":  # static
        voice = voice[decoy_index(s.step, s.alive)]
    h = hear(voice, ctx.nbr, ctx.desc, cfg)
    return h[:, :w], h[:, w:]


def audible_count(nbr: Any, desc: mx.array, cfg: Config) -> mx.array:
    """(N,) int32: the neighbours each row can hear, those with a hearing weight ``a > 0`` (the weight of
    ``hear``: valid, inside the range, and with ``kin_exp > 0`` a positive kin similarity). 0 under the
    ``deaf``
    control. Observer channel only."""
    if cfg.comm.control == "deaf":  # static
        return mx.zeros((dims(cfg).n,), dtype=_I32)
    return (weights(nbr, desc, cfg) > 0.0).astype(_I32).sum(axis=1)


def with_heard(v: Voice, heard: mx.array, audible: mx.array) -> Voice:
    """``v`` with the observer channel's inputs of the step (``heard (N, W)``, ``audible (N,)``)."""
    return v._replace(heard=heard, audible=audible)


def acts_of(v: Voice) -> CommActs:
    """The observer sub-tuple: quantised from the values the step already holds (``v.heard``/``v.audible`` are
    set by ``with_heard`` in the observed step only)."""
    assert v.heard is not None and v.audible is not None, "the observed step attaches heard and audible"
    voice = mx.minimum(ops.floor_i32(v.sig * 256.0), 255).astype(mx.uint8)
    heard = mx.clip(mx.round(v.heard * 255.0), 0, 255).astype(mx.uint8)
    return CommActs(voice, heard, mx.minimum(v.audible, 255).astype(mx.uint8))


def ext_count(s_new: Any, alive0: mx.array, flows: Any, cfg: Config) -> dict[str, mx.array]:
    """The counter block (names in ``ext.comm._COUNTERS``): the rows that spoke (some call quantum
    ``floor(256·sig) > 0``, the ``ActFlag.SPOKE`` form) and the voice quanta paid, both of the solvent
    rows."""
    f = flows.ext.comm
    return {"COMM_SPEAKERS": spoke(f).astype(_I32).sum(), "COMM_DUE": f.due.sum()}


def spoke(f: Voice) -> mx.array:
    """(N,) bool: the row said something this step (the ``ActFlag.SPOKE`` bit, the ``COMM_SPEAKERS`` rows)."""
    return (ops.floor_i32(f.sig * 256.0) > 0).any(axis=1)
