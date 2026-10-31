"""Event-triggered calls: the event windows and the gate (DESIGN §47, step K1).

With ``calls.trigger`` every voice channel of ``comm`` belongs to one event of ``calls.events`` (``hurt``,
``food``, ``storm``, ``lost``; the active events in bit order take channels 0, 1, ...). Each row carries a
window per event, ``State.evt (N, E) uint8`` = steps left (0 = closed). A channel can be called on only while
its window is open: ``comm.sig_of`` takes ``s.evt > 0`` as its gate, so a closed channel is exactly 0, silent
and free, and an open one is the brain's own ``clip(tanh(y), 0, 1 - 2^-24)`` (the intensity carries no
content in v1: it is a constant "call when open" scaled by whatever else the brain senses). The cost is the
existing dithered ``comm.cost`` through ``extra_due`` (``HEAT_ACTION``): no ledger term.

The levels that open a window are computed **on the survivor stage** (``windows``, called by ``step`` at S15
with ``s1``, before ``reproduce``), from the arrays of the table of DESIGN §47.2, and masked by the survivors
``sv``: a dead row has level 0, and a newborn on a row that was alive at the start of the step (the
``reproduce`` lanes take the lowest-index dead rows, which include rows that died this step) gets a closed
window from ``FieldSpec.on_birth = "rest"``, never the previous occupant's. The update is

    open  = evt > 0
    left  = where(open, evt - 1, 0)                   # uint8 arithmetic wraps: 0 - 1 = 255 in MLX
    evt'  = where(sv, where(level, hold + 1, left), 0)

so a window opens at t + 1 iff ``evt > 0`` after step t, a one-step level opens ``hold + 1`` steps (``hold`` 0
opens one), and ``evt = 0`` stays 0 (the v0 formula ``maximum(evt - 1, 0)`` re-opened every closed window).

The scalars that vary with the config (``hold + 1``, ``food_q``, the storm threshold) are passed to the
compiled step as runtime arrays (``runtime``), never as printed constants: MLX's CPU backend writes the
scalar constants of a fused kernel into its source and names the kernel after them, and a constant list whose
print concatenates like another's shares its kernel (DESIGN F21 item 6, §47.7).
"""

from __future__ import annotations

from typing import Any, Final, NamedTuple

import mlx.core as mx

from alife import ops
from alife.config import Config
from alife.ext.calls import event_names

_I32 = mx.int32
_U8 = mx.uint8
_F32 = mx.float32

# Indices into ``Runtime.ints`` / ``Runtime.floats``.
I_HOLD1: Final = 0
I_FOOD_Q: Final = 1
F_STORM: Final = 0


class Runtime(NamedTuple):
    """The config scalars of the window update as constant arrays of two elements. MLX inlines a scalar (0-d
    or one-element) constant into the fused kernel's source, where it is a printed constant; an array of two
    or more is a kernel input (``B[0]``), so two configurations that differ in these values share no
    kernel text. The second float is padding."""

    ints: mx.array  # (2,) int32: hold + 1, food_q
    floats: mx.array  # (2,) float32: storm_frac * storm_u0, and a pad


def runtime(cfg: Config) -> Runtime:
    c = cfg.calls
    thr = c.storm_frac * cfg.weather.storm_u0  # rounded to float32 once, here
    return Runtime(mx.array([c.hold + 1, c.food_q], _I32), mx.array([thr, thr], _F32))


class CallFlows(NamedTuple):
    """The windows of a step (``ExtFlows.calls``)."""

    open: mx.array  # (N, E) bool: the windows that gated this step's voice (start-of-step ``evt > 0``, alive)


class CallsActs(NamedTuple):
    """``Acts.ext.calls`` (observer channel, DESIGN §47.9)."""

    open: mx.array  # (N, E) uint8: 1 where the channel's window was open when the row decided


def levels(s: Any, s1: Any, sv: mx.array, wind: Any, cfg: Config, rt: Runtime) -> mx.array:
    """(N, E) bool: the level of each active event at the survivor stage of this step, masked by ``sv``.

    ``s`` is the start-of-step state, ``s1`` the survivor state (``alive = sv``, the post-deaths ``partner``,
    ``last_loss = where(sv, cb.lost, 0)``, ``last_harvest = where(sv, got, 0)``), ``wind`` the sheltered
    ``weather.AgentForce`` of the start-of-step positions (None without weather). Every term is read where
    DESIGN §47.2 says; none from ``s2`` (after ``reproduce``), where a newborn would inherit a dead
    occupant's value."""
    cols = []
    for name in event_names(cfg.calls.events):  # static: the active events, in channel order
        if name == "hurt":  # energy taken by an attacker this step
            lv = s1.last_loss > 0
        elif name == "food":  # a level: a rich harvest this step
            lv = s1.last_harvest >= rt.ints[I_FOOD_Q]
        elif name == "storm":  # the air speed (after shelter) of the start-of-step occupant of the row
            lv = wind.speed >= rt.floats[F_STORM]
        else:  # "lost": a bond of the start-of-step state is gone or was refilled by another partner
            lv = ((s.partner >= 0) & (s1.partner != s.partner)).any(axis=1)
        cols.append(lv & sv)
    return mx.stack(cols, axis=1)


def window_update(evt: mx.array, level: mx.array, sv: mx.array, rt: Runtime) -> mx.array:
    """The next windows: ``hold + 1`` where the level holds, else one step less (never below 0), and 0 on
    every row that did not survive. ``evt`` is the uint8 window of the start of the step."""
    left = mx.where(evt > 0, evt - 1, 0)  # not ``evt - 1`` alone: a uint8 0 would wrap to 255
    new = mx.where(level, rt.ints[I_HOLD1].astype(_U8), left)
    return mx.where(sv[:, None], new, 0).astype(_U8)


def windows(s: Any, s1: Any, sv: mx.array, wind: Any, cfg: Config) -> mx.array:
    """The ``evt`` of ``s1`` (S15): levels on the survivor stage, then the update."""
    rt = runtime(cfg)
    return window_update(s.evt, levels(s, s1, sv, wind, cfg, rt), sv, rt)


def decode(s: Any, y: mx.array, ctx: Any, cfg: Config) -> CallFlows | None:
    """The registry ``decode`` hook (S7): the windows that gate this step's voice. None without
    ``calls.trigger``. (The gate itself is applied in ``comm.sig_of``.)"""
    if not cfg.calls.trigger:  # static
        return None
    return CallFlows((s.evt > 0) & s.alive[:, None])


def ext_count(s_new: Any, alive0: mx.array, flows: Any, cfg: Config) -> dict[str, mx.array]:
    """The counter block (names in ``ext.calls._counters``): per active event the rows whose window was open
    (``CALLS_OPEN_<E>``) and those that spoke on it (``CALLS_SPOKE_<E>``: a call quantum ``floor(256·sig) >
    0`` of the solvent rows, the ``ActFlag.SPOKE`` form per channel)."""
    f, sig = flows.ext.calls, flows.ext.comm.sig
    spoke = ops.floor_i32(sig * 256.0) > 0
    return {  # static: the active events
        key: arr[:, c].astype(_I32).sum()
        for c, name in enumerate(event_names(cfg.calls.events))
        for key, arr in ((f"CALLS_OPEN_{name.upper()}", f.open), (f"CALLS_SPOKE_{name.upper()}", spoke))
    }


def acts_of(f: CallFlows) -> CallsActs:
    """The observer sub-tuple: the open windows as 0/1 bytes."""
    return CallsActs(f.open.astype(_U8))
