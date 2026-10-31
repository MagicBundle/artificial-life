"""The declaration of the event-triggered calls feature (DESIGN §47, §43): a row of ``ext.REGISTRY``.

Step K1 builds the event windows and the gate: with ``calls.trigger`` each voice channel of ``comm`` belongs
to one event (``hurt``, ``food``, ``storm``, ``lost``) and a row can call on it only while its window is
open. The feature has **no brain footprint** in K1 (no input and no output is added; ``layout_on`` is never
true, so the genome layout version and the weights of a run do not change), one conditional per-agent state
field ``evt`` (the window, steps left, ``uint8``, present only with ``trigger``) and two counters per active
event. The far hearing of step K2 (``far_range > 0``, the ``voice_far`` and ``far_bearing`` blocks, the
controls ``nofar``/``reflex``/``reflex_near``) is declared in the config and refused by ``validate`` until
it is built. The implementation lives in ``alife.calls``; imports of it are made at call time because
``config`` imports ``ext``.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from alife.ext.base import ExtCtx, ExtSpec, FieldSpec

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
)
_KEYS_NAMES = tuple(k for _, k in _KEYS)
# bit -> event name, in channel order (the active events in bit order take voice channels 0, 1, ...)
EVENTS = (("hurt", 1), ("food", 2), ("storm", 4), ("lost", 8))
EVENTS_MAX = 15
HOLD_MAX = 64  # evt is uint8 and holds hold + 1 <= 65 steps left
CONTROLS = ("none", "nofar", "reflex", "reflex_near")
K2_NOTE = "is built in step K2 of the calls series (far hearing and the oracle controls)"


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


def _counters(cfg: Any) -> tuple[str, ...]:
    if not cfg.calls.trigger:
        return ()
    return tuple(
        f"CALLS_{kind}_{name.upper()}" for name in event_names(cfg.calls.events) for kind in ("OPEN", "SPOKE")
    )


def _enable(cfg: Any) -> Any:
    """``alife migrate --enable calls``: the config with ``calls.enabled`` on and nothing else (the trigger
    stays off, so the migrated world behaves as before; ``--set calls.trigger=true --set calls.events=5``
    moves it onto the event-gated voice, DESIGN §47.8)."""
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
        bad.append(f"calls.far_range {c.far_range} {K2_NOTE}: it must be 0 until then")
    if not (math.isfinite(c.far_xref) and c.far_xref > 0.0):
        bad.append(f"calls.far_xref {c.far_xref} must be > 0")
    elif c.far_range == 0.0 and c.far_xref != d.far_xref:
        bad.append("calls.far_xref is inert without calls.far_range > 0: it must keep its default")
    if c.control not in CONTROLS:
        bad.append(f"calls.control {c.control!r} must be one of {CONTROLS}")
    else:
        if c.control in ("nofar", "reflex") and not c.far_range > 0.0:
            bad.append(f"calls.control {c.control!r} needs calls.far_range > 0")
        if c.control in ("reflex", "reflex_near") and not (c.trigger and c.events & 4):
            bad.append(
                f"calls.control {c.control!r} needs calls.trigger and the storm bit (4) in calls.events"
            )
        if c.control != "none":
            bad.append(f"calls.control {c.control!r} {K2_NOTE}: it must be 'none' until then")
    if bad:
        raise ConfigError("; ".join(bad))


SPEC = ExtSpec(
    name="calls",
    enabled=_enabled,
    layout_on=lambda cfg: False,  # K1: no brain footprint (K2: far_range > 0 adds voice_far and far_bearing)
    token=lambda cfg: "",
    state_fields=_state_fields,
    counters=_counters,
    metric_prefix="calls_",
    lever_keys=_KEYS,
    cli_only_keys=_KEYS,  # no GUI lever file yet (step K4): every key is command-line only
    overlay="calls",
    enable=_enable,
    validate=_validate,
    decode=_decode,
    count=_count,
    check=_check,
    snap_dev=_snap_dev,
    snap_row=_snap_row,
)
