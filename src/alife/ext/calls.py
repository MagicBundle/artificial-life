"""The declaration of the event-triggered calls feature (DESIGN §47, §43): a row of ``ext.REGISTRY``.

Step K1 builds the event windows and the gate: with ``calls.trigger`` each voice channel of ``comm`` belongs
to one event (``hurt``, ``food``, ``storm``, ``lost``) and a row can call on it only while its window is
open: one conditional per-agent state field ``evt`` (the window, steps left, ``uint8``, present only with
``trigger``) and two counters per active event. Step K2 adds the far hearing: with ``far_range > 0`` the
feature has a **brain footprint**, the input blocks ``voice_far`` (W) and ``far_bearing`` (2W) after comm's
tail (token ``calls{W}@1``), the counter ``CALLS_FAR_HEARD`` and the controls ``nofar``/``reflex``/
``reflex_near``; with ``far_range = 0`` there is none (the genome layout, the weights and the state of a run
are those of K1). The implementation lives in ``alife.calls`` and ``alife.calltab``; imports of them are made
at call time because ``config`` imports ``ext``.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from alife.ext.base import Block, ExtCtx, ExtSpec, FieldSpec

_KEYS = (
    ("calls", "enabled"),
    ("calls", "trigger"),
    ("calls", "events"),
    ("calls", "hold"),
    ("calls", "food_q"),
    ("calls", "storm_frac"),
    ("calls", "far_range"),
    ("calls", "far_xref"),
    ("calls", "control"),
    ("calls", "oracle_min"),
    ("calls", "oracle_gain"),
)
_KEYS_NAMES = tuple(k for _, k in _KEYS)
# bit -> event name, in channel order (the active events in bit order take voice channels 0, 1, ...)
EVENTS = (("hurt", 1), ("food", 2), ("storm", 4), ("lost", 8))
EVENTS_MAX = 15
HOLD_MAX = 64  # evt is uint8 and holds hold + 1 <= 65 steps left
CONTROLS = ("none", "nofar", "reflex", "reflex_near", "reflex_call")
ORACLES = ("reflex", "reflex_near")  # the controls that add a force (and count CALLS_ORACLE_ON)
CALLING = ("reflex", "reflex_near", "reflex_call")  # the controls that call every open window at 0.5
ORACLE_GAIN_MAX = 4.0
FAR_RANGE_RADII = 4.0  # far_range <= 4 sensing radii


def event_names(events: int) -> tuple[str, ...]:
    """The active events of the mask ``events``, in bit order (= voice channel order)."""
    return tuple(name for name, bit in EVENTS if events & bit)


def _enabled(cfg: Any) -> bool:
    return bool(cfg.calls.enabled)


def _state_fields(cfg: Any, nd: int) -> tuple[FieldSpec, ...]:
    if not cfg.calls.trigger:  # the window exists only with the trigger (checkpointed and digested with it)
        return ()

    def shape(d: Any) -> tuple[int, ...]:
        return (d.n, len(event_names(cfg.calls.events)))

    def rest(cfg: Any) -> Any:
        from alife.config import dims

        return mx.zeros(shape(dims(cfg)), mx.uint8)

    # on_birth "rest": a newborn (a birth or an immigrant, on any row, including one reused from a row that
    # died this step) starts with every window closed (DESIGN §47.2, edge case K3). Not conserved: a label.
    return (FieldSpec("evt", True, shape, mx.uint8, rest, "rest", conserved=False),)


def _far(cfg: Any) -> bool:
    return bool(cfg.calls.enabled) and cfg.calls.far_range > 0.0


def _in_blocks(cfg: Any, nd: int) -> tuple[Block, ...]:
    w = cfg.comm.width
    return (Block("voice_far", w, False), Block("far_bearing", 2 * w, False))


def _inputs(s: Any, cfg: Any, ctx: ExtCtx) -> Any:
    from alife import calls

    return calls.ext_inputs(s, cfg, ctx)


def _counters(cfg: Any) -> tuple[str, ...]:
    out: tuple[str, ...] = ()
    if cfg.calls.trigger:
        out = tuple(
            f"CALLS_{kind}_{name.upper()}"
            for name in event_names(cfg.calls.events)
            for kind in ("OPEN", "SPOKE")
        )
    if cfg.calls.far_range > 0.0:
        out = (*out, "CALLS_FAR_HEARD")
    return (*out, "CALLS_ORACLE_ON") if cfg.calls.control in ORACLES else out


def _enable(cfg: Any) -> Any:
    """``alife migrate --enable calls``: the config with ``calls.enabled`` on and nothing else (the trigger
    and the far hearing stay off, so the migrated world behaves as before; ``--set calls.trigger=true --set
    calls.events=5 --set calls.far_range=18``, in the same call, moves it onto the event-gated voice and the
    far block, DESIGN §47.5)."""
    import dataclasses

    return dataclasses.replace(cfg, calls=dataclasses.replace(cfg.calls, enabled=True))


def _decode(s: Any, y: Any, ctx: ExtCtx, cfg: Any) -> Any:
    from alife import calls

    return calls.decode(s, y, ctx, cfg)


def _count(s_new: Any, alive0: Any, flows: Any, cfg: Any) -> Any:
    from alife import calls

    return calls.ext_count(s_new, alive0, flows, cfg)


def _check(arrays: Any, cfg: Any) -> list[str]:
    """Host check of the numpy ``evt`` field: at most ``hold + 1`` steps left, and 0 on a dead row (a dead or
    reused row has level 0, K6). Here, not in ``alife.calls``: the host reads of numpy arrays are allowed
    outside the simulation modules (``tests/test_lint.py`` rule 5)."""
    import numpy as np

    if "evt" not in arrays:
        return []
    evt, alive = np.asarray(arrays["evt"]), np.asarray(arrays["alive"])
    bad = []
    if (evt > cfg.calls.hold + 1).any():
        bad.append(f"evt: a window longer than hold + 1 = {cfg.calls.hold + 1} (max {int(evt.max())})")
    if evt[~alive].any():
        bad.append("evt: non-zero on a dead row")
    return bad


def _snap_dev(s: Any, cfg: Any, heavy: bool) -> Any:
    from alife import featobs

    return featobs.calls_dev(s, cfg, heavy)


def _snap_row(h: Any, cfg: Any) -> Any:
    from alife import featobs

    return featobs.calls_row(h, cfg)


def _far_range_problems(cfg: Any) -> list[str]:
    """Why ``calls.far_range`` (nonzero) is not valid for this config (DESIGN §47.7); empty when sound. The
    inner edge of the far annulus is the near range, so far hearing needs ``comm.range == 0`` (the sensing
    radius); the kernel ramps up over one column side and the sparse-caller accuracy tables hold from four of
    them, so ``far_range >= 4`` columns; the table is bounded (S <= 512 entries per column, 64 MB). Reads the
    world keys only: ``dims`` is not safe to call before every validator has run."""
    import math

    from alife import calltab

    c, bad = cfg.calls, []
    r = cfg.sensing.radius
    kind, n, scale = calltab.geometry_key(cfg)
    cell = calltab.column_side(kind, n, scale)
    if not (math.isfinite(c.far_range) and c.far_range > 0.0):
        return [f"calls.far_range {c.far_range} must be 0 or a finite length > 0"]
    if cfg.comm.range != 0.0:
        bad.append(
            "calls.far_range > 0 needs comm.range == 0 (the near range is the inner edge of the far "
            f"annulus: the sensing radius {r}), got comm.range = {cfg.comm.range}"
        )
    if cfg.comm.kin_exp > 0.0:
        bad.append(
            "calls.far_range > 0 needs comm.kin_exp == 0 (a mean-tag kin weight over a column is wrong in "
            "magnitude and in direction: refused in v1)"
        )
    lo = max(calltab.CELLS_MIN * cell, r)
    # (strictly above the near range: a range equal to R is no annulus)
    if not (lo <= c.far_range <= FAR_RANGE_RADII * r and c.far_range > r):
        bad.append(
            f"calls.far_range {c.far_range} must be in [max(4 columns = {calltab.CELLS_MIN * cell:.3f}, "
            f"the sensing radius {r} exclusive), 4 sensing radii = {FAR_RANGE_RADII * r}] "
            f"(a column is {cell:.3f} wide)"
        )
    reach = 2.0 * scale if kind == "sphere" else scale / 2.0  # the planet diameter; half a torus side
    if c.far_range >= reach:
        bad.append(f"calls.far_range {c.far_range} must be below {reach} (the world's reach)")
    if not bad:
        tab = calltab.build(kind, n, scale, float(r), float(c.far_range), float(cfg.comm.atten_pow))
        if tab.s > calltab.S_MAX:
            bad.append(
                f"calls.far_range {c.far_range} gives {tab.s} table entries per column (max {calltab.S_MAX})"
            )
        if tab.nbytes > calltab.TABLE_MAX_BYTES:
            bad.append(
                f"calls.far_range {c.far_range} gives a {tab.nbytes / 2**20:.1f} MB table (max "
                f"{calltab.TABLE_MAX_BYTES >> 20} MB)"
            )
    return bad


def _validate(cfg: Any) -> None:
    import math

    from alife.config import CallsConfig, ConfigError

    c, d, bad = cfg.calls, CallsConfig(), []
    if not c.enabled:  # inert: every key must keep its default, so no hash changes in vain
        bad = [
            f"calls.{k} is inert while calls.enabled is false: it must keep its default"
            for k in _KEYS_NAMES
            if k != "enabled" and getattr(c, k) != getattr(d, k)
        ]
        if bad:
            raise ConfigError("; ".join(bad))
        return
    if not cfg.comm.enabled:
        bad.append("calls.enabled needs comm.enabled (the calls gate the voice of comm)")
    if not c.trigger:  # inert: the event keys are read only with the trigger
        bad += [
            f"calls.{k} is inert while calls.trigger is false: it must keep its default"
            for k in ("events", "hold", "food_q", "storm_frac")
            if getattr(c, k) != getattr(d, k)
        ]
    else:
        if not 1 <= c.events <= EVENTS_MAX:
            bad.append(f"calls.events {c.events} must be an integer in [1, {EVENTS_MAX}]")
        else:
            n = len(event_names(c.events))
            if cfg.comm.enabled and cfg.comm.width != n:
                bad.append(
                    f"calls.trigger needs comm.width == the number of events {n} "
                    f"(calls.events = {c.events}), got comm.width = {cfg.comm.width}: "
                    "one voice channel per event"
                )
            if c.events & 4:
                if not cfg.weather.enabled:
                    bad.append("calls.events with the storm bit (4) needs weather.enabled")
                elif not cfg.weather.storm_u0 > 0.0:
                    bad.append("calls.events with the storm bit (4) needs weather.storm_u0 > 0")
        if not 0 <= c.hold <= HOLD_MAX:
            bad.append(f"calls.hold {c.hold} must be in [0, {HOLD_MAX}] (the window is a uint8)")
        if (
            c.events & 2
        ):  # the range applies only where the key is read (the default 200 may exceed harvest_max)
            if not 1 <= c.food_q <= cfg.energy.harvest_max:
                bad.append(
                    f"calls.food_q {c.food_q} must be in [1, energy.harvest_max = {cfg.energy.harvest_max}]"
                )
        elif c.food_q != d.food_q:
            bad.append(
                "calls.food_q is inert without the food bit (2) in calls.events: it must keep its default"
            )
        if not (math.isfinite(c.storm_frac) and 0.0 < c.storm_frac <= 2.0):
            bad.append(f"calls.storm_frac {c.storm_frac} must be in (0, 2]")
        elif not c.events & 4 and c.storm_frac != d.storm_frac:
            bad.append(
                "calls.storm_frac is inert without the storm bit (4) in calls.events: "
                "it must keep its default"
            )
    # (enabled alone, with neither the trigger nor far hearing, is valid: it has no behaviour, the first form
    # of the migration, DESIGN §47.8)
    if c.far_range != 0.0:
        bad += _far_range_problems(cfg)
    if not (math.isfinite(c.far_xref) and c.far_xref > 0.0):
        bad.append(f"calls.far_xref {c.far_xref} must be > 0")
    elif c.far_range == 0.0 and c.far_xref != d.far_xref:
        bad.append("calls.far_xref is inert without calls.far_range > 0: it must keep its default")
    if c.control not in CONTROLS:
        bad.append(f"calls.control {c.control!r} must be one of {CONTROLS}")
    else:
        if c.control in ("nofar", "reflex", "reflex_call") and not c.far_range > 0.0:
            bad.append(f"calls.control {c.control!r} needs calls.far_range > 0")
        if c.control in CALLING and not (c.trigger and c.events & 4):
            bad.append(
                f"calls.control {c.control!r} needs calls.trigger and the storm bit (4) in calls.events"
            )
    if c.control in ORACLES:
        if not (math.isfinite(c.oracle_min) and 0.0 < c.oracle_min <= 1.0):
            bad.append(f"calls.oracle_min {c.oracle_min} must be in (0, 1]")
        if not (math.isfinite(c.oracle_gain) and 0.0 < c.oracle_gain <= ORACLE_GAIN_MAX):
            bad.append(f"calls.oracle_gain {c.oracle_gain} must be in (0, {ORACLE_GAIN_MAX}]")
    else:  # inert keys keep their default (the repository convention)
        bad += [
            f"calls.{k} is inert unless calls.control is an oracle {ORACLES}: it must keep its default"
            for k in ("oracle_min", "oracle_gain")
            if getattr(c, k) != getattr(d, k)
        ]
    if bad:
        raise ConfigError("; ".join(bad))


SPEC = ExtSpec(
    name="calls",
    enabled=_enabled,
    layout_on=_far,  # far_range > 0 adds voice_far and far_bearing (K2); the trigger alone has no footprint
    token=lambda cfg: f"calls{cfg.comm.width}@1" if _far(cfg) else "",
    in_blocks=_in_blocks,
    block_names=("voice_far", "far_bearing"),
    state_fields=_state_fields,
    counters=_counters,
    metric_prefix="calls_",
    lever_keys=_KEYS,
    cli_only_keys=_KEYS,  # no GUI lever file yet (step K4): every key is command-line only
    overlay="calls",
    enable=_enable,
    validate=_validate,
    inputs=_inputs,
    decode=_decode,
    count=_count,
    check=_check,
    snap_dev=_snap_dev,
    snap_row=_snap_row,
)
