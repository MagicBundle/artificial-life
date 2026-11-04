"""The declaration of the diel feature, the day-night light cycle on the photic food band (DESIGN §49): a
registry row.

D1: no brain footprint (``layout_on`` is false until D2 adds the ``light`` input), no counter, no stream tag;
one conditional state field ``diel_ctl`` (2,) float32, the (target, effective) day fraction of the Long Night
power (§49.8), and the config section ``[diel]``. The light factor lives in ``alife.diel``; imports of it are
made at call time because ``config`` imports ``ext``.
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
)
MODES = ("global", "terminator")
CONTROLS = ("none", "mean")
SENSE_CONTROLS = ("none", "mute")
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


def _state_fields(cfg: Any, nd: int) -> tuple[FieldSpec, ...]:
    # conserved=False: the day fraction holds no energy, so a forced removal by `alife migrate` leaks nothing
    return (FieldSpec("diel_ctl", False, _ctl_shape, mx.float32, rest_ctl, conserved=False),)


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
        # the steps that have not landed: declared so their hash and file format are fixed, refused until
        # built
        if c.sense:
            bad.append("diel.sense is refused until PR D2 (the light input) lands")
        if c.extinction != dflt.extinction:
            bad.append("diel.extinction acts on the light input only: refused until PR D2 lands")
        if not c.silent_init:
            bad.append("diel.silent_init acts on the light input only: refused until PR D2 lands")
        if c.oracle > 0:
            bad.append("diel.oracle is refused until PR D1b (the oracle pilot) lands")
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

    ctl = arrays.get("diel_ctl")
    if ctl is None:
        return []
    ctl = np.asarray(ctl, dtype=np.float64)
    if ctl.shape != (2,) or not np.all(np.isfinite(ctl)) or not np.all((ctl > 0.0) & (ctl < 1.0)):
        return [f"diel_ctl {ctl.tolist()} must be two finite day fractions in (0, 1)"]
    return []


SPEC = ExtSpec(
    name="diel",
    enabled=_enabled,
    state_fields=_state_fields,
    lever_keys=tuple(("diel", k) for k in _KEYS),
    cli_only_keys=tuple(("diel", k) for k in _KEYS),  # no GUI levers yet: DielLevers.swift lands with D6
    overlay="diel",
    enable=_enable,
    validate=_validate,
    check=_check,
)
