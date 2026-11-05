"""The declaration of the diel feature, the day-night light cycle on the photic food band (DESIGN §49): a
registry row.

D1: no brain footprint (``layout_on`` is false until D2 adds the ``light`` input), no counter, no stream tag;
one conditional state field ``diel_ctl`` (2,) float32, the (target, effective) day fraction of the Long Night
power (§49.8), and the config section ``[diel]``. D1b adds the oracle pilot (``diel.oracle > 0``): the
per-agent int8 field ``diel_orc``, 24 harvest counters and the per-stratum energy rows
(``alife.diel_oracle``). The light factor lives in ``alife.diel``; imports of it are made at call time because
``config`` imports ``ext``.
"""

from __future__ import annotations

import math
from typing import Any

import mlx.core as mx

from alife.ext.base import ExtSpec, FieldSpec

_KEYS = (
    "period",
    "mode",
    "day_frac",
    "twilight",
    "night_level",
    "control",
    "sense",
    "extinction",
    "sense_control",
    "oracle",
    "silent_init",
    "oracle_schedule",
)
MODES = ("global", "terminator")
CONTROLS = ("none", "mean")
SENSE_CONTROLS = ("none", "mute")
ORACLE_SCHEDULES = (1, 2)  # D1b's schedule (§49.12) and D1c's round trips (§49.14)
PERIOD_MIN = 400
PERIOD_MAX = 1 << 20
DAY_FRAC_MAX = 0.95
ROWS_PER_DAY_MIN = 12  # metrics rows per day: the D3 amplitude fit needs at least this many phases


def day_frac_min(twilight: float) -> float:
    """The least day fraction at which the equatorial noon still reaches full light: ``acos(1 - tw/2)/pi``
    (``h0 + tw/2 <= 1``, so the smoothstep reaches 1 at ``h = 1``)."""
    return math.acos(1.0 - twilight / 2.0) / math.pi


def _enabled(cfg: Any) -> bool:
    return cfg.diel.period > 0


def _ctl_shape(dims: Any) -> tuple[int, ...]:
    return (2,)


def rest_ctl(cfg: Any) -> mx.array:
    """``diel_ctl`` at rest, ``(day_frac, day_frac)``: the initial value, the fill of a file without the field
    and what a fork restarts from (``cli._prepare_fork``: Long Night's target is a lever, like the env)."""
    d = float(cfg.diel.day_frac)
    return mx.array([d, d], dtype=mx.float32)


def _orc_shape(dims: Any) -> tuple[int, ...]:
    return (dims.n,)


def _orc_rest(cfg: Any) -> mx.array:
    from alife.config import dims

    return mx.zeros((dims(cfg).n,), dtype=mx.int8)


def _state_fields(cfg: Any, nd: int) -> tuple[FieldSpec, ...]:
    # conserved=False: the day fraction and the oracle labels hold no energy, so a forced removal by `alife
    # migrate` leaks nothing. diel_orc (D1b) exists only while the oracle is on; a child inherits its
    # parent's label, an immigrant gets 0 (unlabelled).
    ctl = FieldSpec("diel_ctl", False, _ctl_shape, mx.float32, rest_ctl, conserved=False)
    if cfg.diel.oracle > 0:  # static: the oracle pilot (DESIGN §49.7)
        orc = FieldSpec("diel_orc", True, _orc_shape, mx.int8, _orc_rest, on_birth="inherit", conserved=False)
        return (ctl, orc)
    return (ctl,)


def _counters(cfg: Any) -> tuple[str, ...]:
    from alife import diel_oracle

    return diel_oracle.COUNTERS if cfg.diel.oracle > 0 else ()


def _count(s_new: Any, alive0: Any, flows: Any, cfg: Any) -> Any:
    from alife import diel_oracle

    return diel_oracle.count(s_new, alive0, flows, cfg)


def _snap_dev(s: Any, cfg: Any, heavy: bool) -> Any:
    from alife import diel_oracle

    return diel_oracle.snap_dev(s, cfg, heavy)


def _snap_row(h: Any, cfg: Any) -> Any:
    from alife import diel_oracle

    return diel_oracle.snap_row(h, cfg)


def _validate(cfg: Any) -> None:
    import dataclasses

    from alife.config import ConfigError, DielConfig

    c, dflt, bad = cfg.diel, DielConfig(), []
    if c.period == 0:  # inert: every key must keep its default, so no config changes its hash without effect
        bad = [
            f"diel.{f.name} is inert while diel.period is 0: it must keep its default"
            for f in dataclasses.fields(DielConfig)
            if f.name != "period" and getattr(c, f.name) != getattr(dflt, f.name)
        ]
    else:
        every = cfg.run.metrics_every
        if not PERIOD_MIN <= c.period <= PERIOD_MAX:
            bad.append(f"diel.period {c.period} must be 0 (off) or in [{PERIOD_MIN}, 2^20]")
        elif every <= 0 or c.period % every or c.period // every < ROWS_PER_DAY_MIN:
            bad.append(
                f"diel.period {c.period} must be a multiple of run.metrics_every {every} with at least "
                f"{ROWS_PER_DAY_MIN} metrics rows per day"
            )
        if cfg.world.dims != 3:
            bad.append("diel needs world.dims = 3 (the photic band is a 3D profile)")
        elif not cfg.fields.photic_amp > 0:
            bad.append("diel needs fields.photic_amp > 0 (the light acts on the photic term only)")
        if c.mode not in MODES:
            bad.append(f"diel.mode {c.mode!r} must be one of {MODES}")
        elif c.mode == "terminator" and cfg.world.geometry != "sphere":
            bad.append("diel.mode terminator needs world.geometry = sphere (the box has one clock)")
        if not 0.0 < c.twilight <= 1.0:
            bad.append(f"diel.twilight {c.twilight} must be in (0, 1]")
        elif not day_frac_min(c.twilight) <= c.day_frac < DAY_FRAC_MAX:
            bad.append(
                f"diel.day_frac {c.day_frac} must be in [{day_frac_min(c.twilight):.4f}, {DAY_FRAC_MAX}) "
                f"for diel.twilight {c.twilight} (noon must reach full light)"
            )
        if not 0.0 <= c.night_level < 1.0:
            bad.append(f"diel.night_level {c.night_level} must be in [0, 1)")
        if c.control not in CONTROLS:
            bad.append(f"diel.control {c.control!r} must be one of {CONTROLS}")
        if c.sense_control not in SENSE_CONTROLS:
            bad.append(f"diel.sense_control {c.sense_control!r} must be one of {SENSE_CONTROLS}")
        elif c.sense_control == "mute" and not c.sense:
            bad.append("diel.sense_control mute needs diel.sense")
        if not c.extinction >= 0.0:
            bad.append(f"diel.extinction {c.extinction} must be >= 0")
        if not 0 <= c.oracle <= 100:
            bad.append(f"diel.oracle {c.oracle} must be in [0, 100]")
        elif c.oracle > 0 and c.control != "none":
            bad.append("diel.oracle needs diel.control = none")
        if c.oracle_schedule not in ORACLE_SCHEDULES:
            bad.append(f"diel.oracle_schedule {c.oracle_schedule} must be one of {ORACLE_SCHEDULES}")
        elif c.oracle == 0 and c.oracle_schedule != dflt.oracle_schedule:
            bad.append("diel.oracle_schedule is inert while diel.oracle is 0: it must keep its default")
        # the steps that have not landed: declared so their hash and file format are fixed, refused until
        # built
        if c.sense:
            bad.append("diel.sense is refused until PR D2 (the light input) lands")
        if c.extinction != dflt.extinction:
            bad.append("diel.extinction acts on the light input only: refused until PR D2 lands")
        if not c.silent_init:
            bad.append("diel.silent_init acts on the light input only: refused until PR D2 lands")
        if c.oracle > 0:  # the oracle pilot (D1b): one clock, and a lift that can rise and hold
            if c.mode != "global":
                bad.append("diel.oracle needs diel.mode = global (the schedule reads one clock)")
            ph = cfg.physics
            if not ph.buoyancy_max > ph.gravity > 0:
                bad.append(
                    "diel.oracle needs physics.buoyancy_max > physics.gravity > 0 (the scripted rise and the "
                    "balancing lift)"
                )
            if c.oracle_schedule == 2:  # D1c (§49.14): the day target is a layer of the box's field
                from alife import diel_oracle, sphere

                if cfg.world.geometry == "sphere":
                    bad.append("diel.oracle_schedule 2 needs the box (its day target is a field layer)")
                elif diel_oracle.photic_target(cfg) >= sphere.geometry(cfg).height_max:
                    bad.append(
                        "diel.oracle_schedule 2 needs a field layer wholly in the photic band (raise "
                        "fields.photic_lambda or lower world.field_cell)"
                    )
        sp = cfg.fields.season_period
        if sp != 0 and sp < 8 * c.period:
            bad.append(f"fields.season_period {sp} must be 0 or >= 8 x diel.period ({8 * c.period})")
        if cfg.run.director == "live":
            bad.append(
                "a diel world refuses --director live until the rule tier has a diel exemption of its fall "
                "criterion (DESIGN §49.2); off, replay and the player mode are allowed"
            )
    if bad:
        raise ConfigError("; ".join(bad))


def _enable(cfg: Any) -> Any:
    """``alife migrate --enable diel``: the config with the cycle on at the claim world's period (4,000 steps,
    DESIGN §49.9) unless ``--set diel.period=...`` moved it; the light input stays off until D2."""
    import dataclasses

    period = cfg.diel.period or 4000
    return dataclasses.replace(cfg, diel=dataclasses.replace(cfg.diel, period=period))


def _check(arrays: Any, cfg: Any) -> list[str]:
    import numpy as np

    bad = []
    ctl = arrays.get("diel_ctl")
    if ctl is not None:
        ctl = np.asarray(ctl, dtype=np.float64)
        if ctl.shape != (2,) or not np.all(np.isfinite(ctl)) or not np.all((ctl > 0.0) & (ctl < 1.0)):
            bad.append(f"diel_ctl {ctl.tolist()} must be two finite day fractions in (0, 1)")
    orc = arrays.get("diel_orc")
    if orc is not None and np.abs(np.asarray(orc, dtype=np.int64)).max(initial=0) > 3:
        bad.append("diel_orc: labels outside [-3, 3]")
    return bad


SPEC = ExtSpec(
    name="diel",
    enabled=_enabled,
    state_fields=_state_fields,
    counters=_counters,
    metric_prefix="diel_",
    lever_keys=tuple(("diel", k) for k in _KEYS),
    cli_only_keys=tuple(("diel", k) for k in _KEYS),  # no GUI levers yet: DielLevers.swift lands with D6
    overlay="diel",
    enable=_enable,
    validate=_validate,
    count=_count,
    check=_check,
    snap_dev=_snap_dev,
    snap_row=_snap_row,
)
