"""The declaration of the built-structures feature (DESIGN §38, §43): a row of ``ext.REGISTRY``.

Footprint: one brain input ``material`` and one output ``build`` (2D 70/36, 3D 85/40 with only this
feature), the conditional state field ``mat`` on the food lattice, ten counters and the config section
``[structures]``. The implementation lives in ``alife.structures`` (hooks) and ``alife.fields``
(``update_mat``); imports of them are made at call time because ``config`` imports ``ext``.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from alife.ext.base import Block, ExtCtx, ExtSpec, FieldSpec

_KEYS = (
    "enabled",
    "cap",
    "build_max",
    "build_cost",
    "decay_q12",
    "shelter_gain",
    "barrier_gain",
    "shield_gain",
    "reach",
    "open_frac",
    "aperture",
    "exit_ease",
    "silent_build",
)
_COUNTERS = (
    "BUILD_UNITS",
    "MAT_WASTED",
    "MAT_DECAYED",
    "MAT_TOTAL",
    "MAT_CELLS",
    "SHIELDED_ATTACKS",
    "TRAPPED",
    "APERTURE_VIOLATIONS",
    "SLOWED",
    "MAT_CLOSED",
)


def _enabled(cfg: Any) -> bool:
    return bool(cfg.structures.enabled)


def _in_blocks(cfg: Any, nd: int) -> tuple[Block, ...]:
    return (Block("material", 1, False),)


def _out_blocks(cfg: Any, nd: int) -> tuple[Block, ...]:
    return (Block("build", 1, bool(cfg.structures.silent_build)),)


def _rest(cfg: Any) -> mx.array:
    from alife.config import dims

    return mx.zeros(dims(cfg).field_shape, dtype=mx.int32)


def _state_fields(cfg: Any, nd: int) -> tuple[FieldSpec, ...]:
    return (FieldSpec("mat", False, lambda d: d.field_shape, mx.int32, _rest),)


def _inputs(s: Any, cfg: Any, ctx: ExtCtx) -> Any:
    from alife import structures

    return structures.inputs(s, cfg, ctx)


def _decode(s: Any, y: Any, ctx: ExtCtx, cfg: Any) -> Any:
    from alife import structures

    return structures.decode(s, y, ctx, cfg)


def _due(s: Any, act: Any, y: Any, cfg: Any) -> Any:
    from alife import structures

    return structures.due(s, act, y, cfg)


def _mask(dec: Any, solvent: Any, cfg: Any) -> Any:
    from alife import structures

    return structures.mask(dec, solvent, cfg)


def _count(s_new: Any, alive0: Any, flows: Any, cfg: Any) -> Any:
    from alife import structures

    return structures.count(s_new, alive0, flows, cfg)


def _check(arrays: Any, cfg: Any) -> list[str]:
    from alife import structures

    return structures.check(arrays, cfg)


def _validate(cfg: Any) -> None:
    import dataclasses
    import math

    from alife import fields
    from alife.config import ConfigError, StructuresConfig

    sc, dflt, bad = cfg.structures, StructuresConfig(), []
    if not sc.enabled:  # inert: every key must keep its default, so no config changes its hash without effect
        bad = [
            f"structures.{f.name} is inert while structures.enabled is false: it must keep its default"
            for f in dataclasses.fields(StructuresConfig)
            if f.name != "enabled" and getattr(sc, f.name) != getattr(dflt, f.name)
        ]
    else:
        nd = cfg.world.dims
        fg = round(cfg.world.size / cfg.world.field_cell)
        n_cells = fg * fg * (round(cfg.world.height / cfg.world.field_cell) if nd == 3 else 1)
        if not 256 <= sc.cap <= 32767:
            bad.append(f"structures.cap {sc.cap} must be in [256, 32767]")
        elif sc.cap * n_cells > 2**31 - 1:
            bad.append(f"structures.cap {sc.cap} x {n_cells} cells overflows the int32 MAT_TOTAL counter")
        if not 1 <= sc.build_max <= 1024:
            bad.append(f"structures.build_max {sc.build_max} must be in [1, 1024]")
        if not 0 <= sc.build_cost <= 2**15 or sc.build_cost * sc.build_max >= 2**20:
            bad.append(
                f"structures.build_cost {sc.build_cost} must be in [0, 2^15] with build_cost*build_max < 2^20"
            )
        if not 1 <= sc.decay_q12 <= 4096:
            bad.append(f"structures.decay_q12 {sc.decay_q12} must be in [1, 4096] (material always decays)")
        for k in ("shelter_gain", "shield_gain", "reach", "exit_ease"):
            if not 0.0 <= getattr(sc, k) <= 1.0:
                bad.append(f"structures.{k} {getattr(sc, k)} must be in [0, 1]")
        if not 0.0 <= sc.barrier_gain <= 3.0:
            bad.append(
                f"structures.barrier_gain {sc.barrier_gain} must be in [0, 3] (R1: at most a 4x slowdown)"
            )
        if not 0.0 < sc.open_frac <= 1.0:
            bad.append(f"structures.open_frac {sc.open_frac} must be in (0, 1]")
        if not 0.0 <= sc.aperture <= 1.0:
            bad.append(f"structures.aperture {sc.aperture} must be in [0, 1]")
        elif sc.aperture > 0:
            k = math.ceil(sc.aperture * (26 if nd == 3 else 8) - 1e-9)
            if k > (17 if nd == 3 else 8):
                bad.append(
                    f"structures.aperture {sc.aperture} needs {k} open neighbours; at most "
                    f"{17 if nd == 3 else 8} exist (a boundary cell has 17 in 3D)"
                )
            try:
                fields.mat_phases(fg, nd)
            except ValueError as e:
                bad.append(f"structures.aperture > 0: {e}")
    if bad:
        raise ConfigError("; ".join(bad))


SPEC = ExtSpec(
    name="structures",
    enabled=_enabled,
    layout_on=_enabled,
    token=lambda cfg: "structures@1",
    in_blocks=_in_blocks,
    out_blocks=_out_blocks,
    block_names=("material", "build"),
    state_fields=_state_fields,
    counters=lambda cfg: _COUNTERS,
    act_flag="BUILDING",
    metric_prefix="build_",
    lever_keys=tuple(("structures", k) for k in _KEYS),
    cli_only_keys=(("structures", "silent_build"),),
    overlay="build",
    validate=_validate,
    inputs=_inputs,
    decode=_decode,
    due=_due,
    mask=_mask,
    count=_count,
    check=_check,
)
