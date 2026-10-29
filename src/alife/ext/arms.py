"""The declaration of the arms feature (DESIGN §42, §43): a row of ``ext.REGISTRY``.

Stages 1 and 2 have no brain footprint (no layout block), no counters and no flag yet (those are the
observer PR); the row declares the conditional state field ``arms``, the config keys and the validation.
Imports of ``alife.arms`` and ``alife.config`` are made at call time: ``config`` imports ``ext``.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from alife.ext.base import ExtSpec, FieldSpec

_COUNTERS = ("ARMS_ATK_SUM", "ARMS_VIC_ARM_SUM", "ARMS_YIELD_SUM", "ARMS_MATCH_SUM", "ARMS_VIC_TOX_SUM")

_KEYS = (
    "enabled",
    "kappa",
    "a0",
    "key_floor",
    "speed_slow",
    "upkeep_atk",
    "upkeep_arm",
    "init_atk",
    "init_arm",
    "sigma",
    "sigma_phase",
    "dump_n",
)


def _enabled(cfg: Any) -> bool:
    return bool(cfg.arms.enabled)


def _state_fields(cfg: Any, nd: int) -> tuple[FieldSpec, ...]:
    from alife import arms

    return (
        FieldSpec(
            "arms",
            True,
            lambda d: (d.n, arms.N_TRAITS),
            mx.float32,
            arms.rest_array,
            arms.make_on_birth(cfg),
        ),
    )


def _count(s_new: Any, alive0: Any, flows: Any, cfg: Any) -> Any:
    from alife import arms

    return arms.count(s_new, alive0, flows, cfg)


def _check(arrays: Any, cfg: Any) -> list[str]:
    from alife import arms

    return arms.check(arrays, cfg)


def _validate(cfg: Any) -> None:
    from alife.config import ArmsConfig, ConfigError

    a, bad = cfg.arms, []
    if not a.enabled:  # inert: every key must keep its default, so no config changes its hash without effect
        import dataclasses

        d = ArmsConfig()
        bad = [
            f"arms.{f.name} is inert while arms.enabled is false: it must keep its default"
            for f in dataclasses.fields(ArmsConfig)
            if f.name != "enabled" and getattr(a, f.name) != getattr(d, f.name)
        ]
    else:
        if not a.kappa >= 0.0:
            bad.append(f"arms.kappa {a.kappa} must be >= 0")
        if not a.a0 > 0.0:
            bad.append(f"arms.a0 {a.a0} must be > 0")
        if a.kappa >= 0.0 and a.a0 > 0.0 and a.kappa / (a.kappa + a.a0) > 0.9:
            bad.append(
                f"arms: kappa/(kappa + a0) = {a.kappa / (a.kappa + a.a0):.3f} must be <= 0.9 (no immunity)"
            )
        if not 0.0 <= a.key_floor <= 1.0:
            bad.append(f"arms.key_floor {a.key_floor} must be in [0, 1]")
        if not 0.0 <= a.speed_slow <= 0.9:
            bad.append(f"arms.speed_slow {a.speed_slow} must be in [0, 0.9]")
        if a.key_floor == 0.0 and cfg.combat.protect_floor == 0.0:
            bad.append(
                "arms.key_floor = 0 needs combat.protect_floor > 0: a mismatched key and lock would give "
                "immunity (the demand floors to 0)"
            )
        for k in ("upkeep_atk", "upkeep_arm"):
            if not 0.0 <= getattr(a, k) < 2**15:
                bad.append(f"arms.{k} {getattr(a, k)} must be in [0, 2^15)")
        for k in ("init_atk", "init_arm"):
            if not 0.0 <= getattr(a, k) <= 1.0:
                bad.append(f"arms.{k} {getattr(a, k)} must be in [0, 1]")
        for k in ("sigma", "sigma_phase"):
            if not 0.0 <= getattr(a, k) <= 0.5:
                bad.append(f"arms.{k} {getattr(a, k)} must be in [0, 0.5]")
        if a.dump_n < 1:
            bad.append(f"arms.dump_n {a.dump_n} must be >= 1")
    if bad:
        raise ConfigError("; ".join(bad))


def _enable(cfg: Any) -> Any:
    """``alife migrate --enable arms``: the config with ``arms.enabled`` on (``--set`` moves the rest)."""
    import dataclasses

    return dataclasses.replace(cfg, arms=dataclasses.replace(cfg.arms, enabled=True))


def _snap_dev(s: Any, cfg: Any, heavy: bool) -> Any:
    from alife import featobs

    return featobs.arms_dev(s, cfg, heavy)


def _snap_row(h: Any, cfg: Any) -> Any:
    from alife import featobs

    return featobs.arms_row(h, cfg)


SPEC = ExtSpec(
    name="arms",
    enable=_enable,
    enabled=_enabled,
    state_fields=_state_fields,
    counters=lambda cfg: _COUNTERS,
    act_flag="CONTESTED",
    dump_kinds=("arms",),
    metric_prefix="arms_",
    lever_keys=tuple(("arms", k) for k in _KEYS),
    observer_keys=(("arms", "dump_n"),),
    cli_only_keys=(("arms", "dump_n"),),
    validate=_validate,
    count=_count,
    check=_check,
    snap_dev=_snap_dev,
    snap_row=_snap_row,
)
