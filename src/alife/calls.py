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

The scalars that vary with the config (``hold + 1``, ``food_q``, the storm threshold, and in step K2 the
column count, the far intensity reference and ``physics.thrust_max``) are passed to the compiled step as
runtime arrays (``runtime``), never as printed constants: MLX's CPU backend writes the scalar constants of a
fused kernel into its source and names the kernel after them, and a constant list whose print concatenates
like another's shares its kernel (DESIGN F21 item 6, §47.6).

**Far hearing (step K2, ``calls.far_range > 0``; DESIGN §47.7).** Every row's stored call (said at t - 1) is
deposited as integer quanta ``min(255, floor(256·sig))`` into the angular column of its start-of-step field
cell (``ops.scatter_add_i32``: exact and independent of the order of the rows); ``calltab`` gives, for each
column, the columns at a centre distance in ``[hear_range, far_range]`` with a weight ``w``; the heard sum of
a column and channel is ``x = Σ_s w · deposit / 256`` over the fixed-order padded table row, the bearing the
loudness-weighted mean of the centre offsets; each hearer reads its own column: ``voice_far = 1 - exp(-x /
far_xref)`` and ``far_bearing`` per channel (forward, lateral) in its own frame (box: rotated by minus the
yaw; sphere: the tangent projection on the heading vector and ``up × hdg``). The vertical position is
dropped, and a column within ``hear_range`` of the hearer (its own included) is not heard by this block: the
near block of ``comm`` covers it. No matmul and no atomic float add, so the block is bitwise reproducible.

The controls (cost and the speaking path identical in every arm): ``nofar`` and ``reflex_near`` zero the far
block, ``comm.control = deaf`` zeroes both blocks, ``decoy`` deposits the donor's voice at the sender's
column. The oracle ``reflex`` / ``reflex_near`` calls every open window at the fixed intensity 0.5
(``comm.sig_of``) and
adds to every hearer whose storm channel is heard at ``>= 0.05`` a horizontal force of magnitude
``physics.thrust_max`` along minus the storm bearing (``oracle_force``: planet frame, through the ``wind=``
argument of the physics): what the information is worth to a perfect, free listener.
"""

from __future__ import annotations

import functools
from typing import Any, Final, NamedTuple

import mlx.core as mx

from alife import calltab, comm, layout, ops, sphere
from alife.config import Config, dims
from alife.ext.calls import ORACLES, event_names

_I32 = mx.int32
_U8 = mx.uint8
_F32 = mx.float32

# Indices into ``Runtime.ints`` / ``Runtime.floats``.
I_HOLD1: Final = 0
I_FOOD_Q: Final = 1
I_NCOL: Final = 2
F_STORM: Final = 0
F_XREF: Final = 1
F_THRUST: Final = 2  # physics.thrust_max x calls.oracle_gain
F_ORACLE_MIN: Final = 3
DEPOSIT_MAX: Final = 255  # a call quantum min(255, floor(256·sig)): sig < 1 gives at most 255
QUANTA: Final = 256.0
BEARING_FLOOR: Final = (
    1e-3  # denominator floor of the far bearing, in heard-intensity units (0 without sound)
)
ORACLE_HEARD_MIN: Final = 0.05  # the default of calls.oracle_min (the config key is the value in use)
ORACLE_DIR_FLOOR: Final = (
    1e-3  # |bearing| below this scales the oracle force down linearly (no garbage direction)
)


class Runtime(NamedTuple):
    """The config scalars of the calls layer as constant arrays of two or more elements. MLX inlines a scalar
    (0-d or one-element) constant into the fused kernel's source, where it is a printed constant; an array of
    two or more is a kernel input (``B[0]``), so two configurations that differ in these values share no
    kernel text. The last element of each array is padding."""

    ints: mx.array  # (4,) int32: hold + 1, food_q, the number of columns, pad
    floats: mx.array  # (4,) float32: storm_frac * storm_u0, far_xref, thrust_max * oracle_gain, oracle_min


def runtime(cfg: Config) -> Runtime:
    c = cfg.calls
    thr = c.storm_frac * cfg.weather.storm_u0  # rounded to float32 once, here
    kind, n, _ = calltab.geometry_key(cfg)
    return Runtime(
        mx.array([c.hold + 1, c.food_q, calltab.ncol_of(kind, n), 0], _I32),
        mx.array([thr, c.far_xref, cfg.physics.thrust_max * c.oracle_gain, c.oracle_min], _F32),
    )


class CallFlows(NamedTuple):
    """The windows and the far hearing of a step (``ExtFlows.calls``)."""

    # (N, E) bool: the windows that gated this step's voice (start-of-step ``evt > 0``, alive); None without
    # ``calls.trigger``
    open: mx.array | None
    # (N,) bool: the row heard some far sound this step (some ``voice_far`` channel > 0, alive at the start);
    # None without far hearing. Attached by ``with_far`` from the brain's own input block.
    far: mx.array | None = None
    # (N,) bool: the oracle's force acted on the row this step (non-zero force); None unless calls.control is
    # an oracle. Attached by ``with_oracle``.
    oracle: mx.array | None = None


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
    """The registry ``decode`` hook (S7): the windows that gate this step's voice (``open``). None without
    ``calls.trigger`` and without far hearing. (The gate itself is applied in ``comm.sig_of``; the far flag is
    attached by ``with_far``.)"""
    if not (cfg.calls.trigger or cfg.calls.far_range > 0.0):  # static
        return None
    return CallFlows((s.evt > 0) & s.alive[:, None] if cfg.calls.trigger else None)


def with_oracle(f: CallFlows | None, force: mx.array) -> CallFlows:
    """``f`` with the rows the oracle's force acted on: a non-zero force ((N, nd), 0 on dead rows)."""
    on = (force * force).sum(axis=-1) > 0.0
    return CallFlows(None, None, on) if f is None else f._replace(oracle=on)


def with_far(f: CallFlows | None, blocks: tuple[mx.array, ...], alive: mx.array) -> CallFlows:
    """``f`` with the far-heard flag of the step from the brain's own ``(voice_far, far_bearing)`` blocks (the
    values the row decided on): some channel heard above 0, on a row alive at the start."""
    far = (blocks[0] > 0.0).any(axis=1) & alive
    return CallFlows(None, far) if f is None else f._replace(far=far)


def ext_count(s_new: Any, alive0: mx.array, flows: Any, cfg: Config) -> dict[str, mx.array]:
    """The counter block (names in ``ext.calls._counters``): per active event the rows whose window was open
    (``CALLS_OPEN_<E>``) and those that spoke on it (``CALLS_SPOKE_<E>``: a call quantum ``floor(256·sig) >
    0`` of the solvent rows, the ``ActFlag.SPOKE`` form per channel); with far hearing
    ``CALLS_FAR_HEARD``, the rows (alive at the start) that heard some far sound."""
    f = flows.ext.calls
    out: dict[str, mx.array] = {}
    if cfg.calls.trigger:  # static
        sig = flows.ext.comm.sig
        spoke = ops.floor_i32(sig * 256.0) > 0
        out = {  # static: the active events
            key: arr[:, c].astype(_I32).sum()
            for c, name in enumerate(event_names(cfg.calls.events))
            for key, arr in ((f"CALLS_OPEN_{name.upper()}", f.open), (f"CALLS_SPOKE_{name.upper()}", spoke))
        }
    if cfg.calls.far_range > 0.0:  # static
        out = {**out, "CALLS_FAR_HEARD": f.far.astype(_I32).sum()}
    if cfg.calls.control in ORACLES:  # static
        out = {**out, "CALLS_ORACLE_ON": f.oracle.astype(_I32).sum()}
    return out


def acts_of(f: CallFlows) -> CallsActs | None:
    """The observer sub-tuple: the open windows as 0/1 bytes (None without the trigger)."""
    return None if f.open is None else CallsActs(f.open.astype(_U8))


# --- far hearing: the column voice grid (DESIGN §47.7) ----------------------------------------------------


class DeviceTab(NamedTuple):
    """The table of ``calltab`` as device arrays, laid out for the reductions over the entry axis."""

    idx: mx.array  # (ncol, S) int32
    w: mx.array  # (ncol, S) float32
    off_t: mx.array  # (nt, ncol, S) float32: the centre offsets, component first


@functools.cache
def _device_table(kind: str, n: int, scale: float, hear: float, far: float, atten: float) -> DeviceTab:
    tab = calltab.build(kind, n, scale, hear, far, atten)
    return DeviceTab(mx.array(tab.idx), mx.array(tab.w), mx.array(tab.off.transpose(2, 0, 1).copy()))


def device_table(cfg: Config) -> DeviceTab:
    """The table of ``cfg`` on the device (built once per geometry and ranges)."""
    kind, n, scale = calltab.geometry_key(cfg)
    return _device_table(
        kind, n, scale, float(cfg.sensing.radius), float(cfg.calls.far_range), float(cfg.comm.atten_pow)
    )


def column_of(fcell: mx.array, cfg: Config, rt: Runtime) -> mx.array:
    """The column of each start-of-step field cell: ``fcell % Ncol`` (a sphere's flat cell is ``layer·Npix +
    pixel``, a 3D box's ``(cz·FG + cy)·FG + cx``) with the column count a runtime array; a 2D box's cell is
    already its column."""
    d = dims(cfg)
    return fcell % rt.ints[I_NCOL] if d.sph or d.d3 else fcell  # static


def deposit(voice: mx.array, col: mx.array, alive: mx.array, rt: Runtime, ncol: int) -> mx.array:
    """(W, Ncol) int32: the integer quanta ``min(255, floor(256·voice))`` of every alive row summed into its
    column, per channel. One ``scatter_add_i32``: exact, and the same for any order of the rows."""
    w = voice.shape[1]
    q = mx.minimum(ops.floor_i32(voice * QUANTA), DEPOSIT_MAX)
    q = mx.where(alive[:, None], q, 0)
    flat = col[:, None] + rt.ints[I_NCOL] * mx.arange(w, dtype=_I32)[None, :]  # channel-major, runtime Ncol
    return ops.scatter_add_i32(ncol * w, flat, q).reshape(w, ncol)


def apply(dep: mx.array, tab: DeviceTab) -> tuple[mx.array, mx.array]:
    """``(x (W, Ncol), B (nt, W, Ncol))`` from the deposit: ``x[c, p] = Σ_s w[p, s]·dep[c, idx[p, s]] / 256``
    and ``B[:, c, p] = Σ_s w·dep·off / max(x, 1e-3)``, each sum over the fixed padded entry axis (a padded
    entry has ``w = 0`` and adds exactly 0; ``|B| <= 1``, exactly 0 without sound)."""
    d = dep.astype(_F32) * (1.0 / QUANTA)  # exact: a power of two
    wg = tab.w[None] * d[:, tab.idx]  # (W, ncol, S)
    x = wg.sum(axis=-1)
    num = (wg[None] * tab.off_t[:, None]).sum(axis=-1)  # (nt, W, ncol)
    return x, num / mx.maximum(x, BEARING_FLOOR)[None]


def _frame(s: Any, cfg: Config) -> tuple[mx.array, mx.array]:
    """The two horizontal unit axes ``(fwd, lat)`` of every row in the planet frame, (N, nd) each: the box's
    yaw ``(cos, sin)`` and ``(-sin, cos)`` (padded with a 0 z in a 3D box); a sphere's heading vector and
    ``up × hdg`` (the same ``lat`` as ``ops.body_axes_s``)."""
    d = dims(cfg)
    if d.sph:  # static
        up = s.pos / sphere.radius_of(s.pos)[:, None]
        return s.hdg, ops.cross3(up, s.hdg)
    c, sn = mx.cos(s.heading), mx.sin(s.heading)
    fwd, lat = mx.stack([c, sn], axis=-1), mx.stack([-sn, c], axis=-1)
    if d.d3:  # static
        z = mx.zeros((d.n, 1), _F32)
        return mx.concatenate([fwd, z], axis=-1), mx.concatenate([lat, z], axis=-1)
    return fwd, lat


def hear_far(s: Any, voice: mx.array, col: mx.array, rt: Runtime, cfg: Config) -> tuple[mx.array, mx.array]:
    """``(voice_far (N, W), far_bearing (N, 2W))`` of the stored ``voice (N, W)`` for the rows in the columns
    ``col``: the grid pipeline end to end (a pure function of the arguments, so the offline counterfactuals
    call it with a modified voice). ``voice_far = 1 - exp(-x / far_xref)`` of the hearer's column; the
    bearing is per channel (forward, lateral) in the hearer's frame, 0 without sound; dead rows hear 0."""
    d = dims(cfg)
    ncol = calltab.ncol_of(*calltab.geometry_key(cfg)[:2])
    dep = deposit(voice, col, s.alive, rt, ncol)
    x, b = apply(dep, device_table(cfg))
    xh = x.T[col]  # (N, W): each hearer reads its own column
    heard = 1.0 - mx.exp(
        -xh / rt.floats[F_XREF]
    )  # (not -expm1: its fused kernel differs by an ulp, DESIGN 47.7)
    bh = b[:, :, col]  # (nt, W, N)
    if d.sph:  # static: the tangent projection on (hdg, up × hdg) of the hearer
        up = s.pos / sphere.radius_of(s.pos)[:, None]
        lat_axis = ops.cross3(up, s.hdg)
        fwd = (bh * s.hdg.T[:, None, :]).sum(axis=0)  # (W, N)
        lat = (bh * lat_axis.T[:, None, :]).sum(axis=0)
        body = mx.stack([fwd.T, lat.T], axis=-1)  # (N, W, 2)
    else:
        v = mx.transpose(bh, (2, 1, 0))  # (N, W, 2)
        body = ops.to_body(v, mx.cos(s.heading)[:, None], mx.sin(s.heading)[:, None])
    bear = body.reshape(d.n, 2 * voice.shape[1])
    on = s.alive[:, None]
    return mx.where(on, heard, 0.0), mx.where(on, bear, 0.0)


def stored_voice(s: Any, cfg: Config) -> mx.array:
    """The stored call of every row as the far block hears it (what was said at t - 1): ``comm.control =
    decoy`` takes the donor's voice at the sender's column, as the near block does."""
    voice = s.msg[:, layout.MSG :]
    if cfg.comm.control == "decoy":  # static
        voice = voice[comm.decoy_index(s.step, s.alive)]
    return voice


def far_off(cfg: Config) -> bool:
    """The far block is exactly 0 in this run: ``comm.control = deaf`` (both blocks), or the controls
    ``nofar`` and ``reflex_near`` (near hearing only)."""
    return cfg.comm.control == "deaf" or cfg.calls.control in ("nofar", "reflex_near")


def ext_inputs(s: Any, cfg: Config, ctx: Any) -> tuple[mx.array, mx.array]:
    """The registry ``inputs`` hook (S5): ``(voice_far (N, W), far_bearing (N, 2W))`` of the stored voice."""
    d = dims(cfg)
    w = d.voice_w
    if far_off(cfg):  # static
        return mx.zeros((d.n, w), _F32), mx.zeros((d.n, 2 * w), _F32)
    rt = runtime(cfg)
    return hear_far(s, stored_voice(s, cfg), column_of(ctx.fcell, cfg, rt), rt, cfg)


# --- the oracle: what the information is worth to a perfect, free listener ---------------------------------


def storm_channel(cfg: Config) -> int:
    """The voice channel of the storm event (the active events in bit order take channels 0, 1, ...)."""
    return event_names(cfg.calls.events).index("storm")


def _unit_force(v: mx.array, heard: mx.array, alive: mx.array, rt: Runtime) -> mx.array:
    """``-thrust_max · v / max(|v|, 1e-3)`` where the storm channel is heard at ``>= 0.05`` (else 0): the
    force of magnitude ``thrust_max`` away from the storm bearing ``v`` ((N, nd) in the planet frame; a
    bearing shorter than 1e-3 scales the force down so that a cancelled bearing gives no random direction)."""
    n = mx.sqrt((v * v).sum(axis=-1, keepdims=True))
    on = (heard >= rt.floats[F_ORACLE_MIN]) & alive
    return mx.where(on[:, None], -rt.floats[F_THRUST] * v / mx.maximum(n, ORACLE_DIR_FLOOR), 0.0)


def oracle_force(s: Any, cfg: Config, ctx: Any, blocks: tuple[mx.array, ...] | None) -> mx.array | None:
    """The additive force (N, nd) of ``calls.control = reflex | reflex_near`` (None for any other control),
    added to the wind force term of the physics. ``blocks`` is the brain's ``(voice_far, far_bearing)``.

    * ``reflex``: the storm channel of the far block (heard intensity and its bearing, rotated back to the
      planet frame from the hearer's frame).
    * ``reflex_near``: the same law on the near hearing only: the storm channel of the stored voice of the K
      nearest sensed neighbours, weighted as ``comm.hear`` (``comm.weights``), the bearing the
      loudness-weighted mean of their planet-frame offsets (horizontal).
    """
    ctl = cfg.calls.control
    if ctl not in ORACLES:  # static
        return None
    rt = runtime(cfg)
    cs = storm_channel(cfg)
    if ctl == "reflex":  # static
        assert blocks is not None, "reflex needs the far blocks of the step"
        heard, fb = blocks[0][:, cs], blocks[1][:, 2 * cs : 2 * cs + 2]
        fwd, lat = _frame(s, cfg)
        v = fb[:, 0:1] * fwd + fb[:, 1:2] * lat
        return _unit_force(v, heard, s.alive, rt)
    nbr, desc = ctx.nbr, ctx.desc
    if cfg.comm.control == "deaf":  # static
        return _unit_force(mx.zeros_like(nbr.off[:, 0]), mx.zeros((dims(cfg).n,), _F32), s.alive, rt)
    voice = stored_voice(s, cfg)[:, cs]
    wgt = comm.weights(nbr, desc, cfg) * voice[nbr.idx]  # (N, K): loudness of each source at the listener
    heard = mx.minimum(wgt.sum(axis=1), 1.0)
    b = (wgt[..., None] * nbr.off).sum(axis=1) / mx.maximum(wgt.sum(axis=1), comm.BEARING_FLOOR)[:, None]
    d = dims(cfg)
    if d.sph:  # static: the tangent part
        up = s.pos / sphere.radius_of(s.pos)[:, None]
        b = b - (b * up).sum(axis=-1, keepdims=True) * up
    elif d.d3:  # static: horizontal
        b = mx.concatenate([b[:, :2], mx.zeros((d.n, 1), _F32)], axis=-1)
    return _unit_force(b, heard, s.alive, rt)
