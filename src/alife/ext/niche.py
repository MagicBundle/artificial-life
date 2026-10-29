"""The declaration of the niche-construction feature, farming only (DESIGN §41, §43): a registry row.

Footprint (``niche.farm``): one brain input ``farm`` (the soil level at the agent's cell) and one output
``farm_put`` (2D 70/36 and 3D 85/40 with only this feature), the conditional state field ``soil`` on the food
lattice, seven counters and the config section ``[niche]``. Marks and food stores are deferred (they would
join this feature; stores need a 17th ledger term). The implementation lives in ``alife.niche`` (hooks) and
``alife.fields`` (``update_resource_farm``); imports of them are made at call time because ``config`` imports
``ext``.
"""

from __future__ import annotations

import logging
from typing import Any

import mlx.core as mx

from alife.ext.base import Block, ExtCtx, ExtSpec, FieldSpec

log = logging.getLogger(__name__)

_KEYS = (
    "farm",
    "farm_rate",
    "farm_cap",
    "farm_cost",
    "farm_decay_q16",
    "farm_bonus_frac",
    "silent_init",
    "control",
)
_COUNTERS = ("FARMERS", "FARM_YIELD", "SOIL_PUT", "SOIL_WASTED", "SOIL_DECAYED", "SOIL_TOTAL", "SLOTS_FULL")
CONTROLS = ("none", "ablate", "mute")
RATE_MAX = 256
CAP_MAX = 32767
ROI_WARN = 12.0  # gross return per unit over its cost above which validate warns (the design point is 5.5)


def _enabled(cfg: Any) -> bool:
    return bool(cfg.niche.farm)


def _in_blocks(cfg: Any, nd: int) -> tuple[Block, ...]:
    return (Block("farm", 1, False),)


def _out_blocks(cfg: Any, nd: int) -> tuple[Block, ...]:
    return (Block("farm_put", 1, bool(cfg.niche.silent_init)),)


def _rest(cfg: Any) -> mx.array:
    from alife.config import dims

    return mx.zeros(dims(cfg).field_shape, dtype=mx.int32)


def _state_fields(cfg: Any, nd: int) -> tuple[FieldSpec, ...]:
    # conserved=False: the soil holds no energy (DESIGN 41.3), so a forced removal by `alife migrate` cannot
    # leak TOTAL
    return (FieldSpec("soil", False, lambda d: d.field_shape, mx.int32, _rest, conserved=False),)


def _enable(cfg: Any) -> Any:
    """``alife migrate --enable niche``: the config with ``niche.farm`` on (``--set`` moves the rest, for
    example ``niche.farm_cost=150``; the overlay ``niche3d`` does it); ``silent_init`` stays True, so nobody
    farms at first."""
    import dataclasses

    return dataclasses.replace(cfg, niche=dataclasses.replace(cfg.niche, farm=True))


def _inputs(s: Any, cfg: Any, ctx: ExtCtx) -> Any:
    from alife import niche

    return niche.inputs(s, cfg, ctx)


def _decode(s: Any, y: Any, ctx: ExtCtx, cfg: Any) -> Any:
    from alife import niche

    return niche.decode(s, y, ctx, cfg)


def _due(s: Any, act: Any, y: Any, cfg: Any) -> Any:
    from alife import niche

    return niche.due(s, act, y, cfg)


def _mask(dec: Any, solvent: Any, cfg: Any) -> Any:
    from alife import niche

    return niche.mask(dec, solvent, cfg)


def _count(s_new: Any, alive0: Any, flows: Any, cfg: Any) -> Any:
    from alife import niche

    return niche.count(s_new, alive0, flows, cfg)


def _check(arrays: Any, cfg: Any) -> list[str]:
    from alife import niche

    return niche.check(arrays, cfg)


def _validate(cfg: Any) -> None:
    import dataclasses

    from alife.config import ConfigError, NicheConfig

    nc, dflt, bad = cfg.niche, NicheConfig(), []
    if not nc.farm:  # inert: every key must keep its default, so no config changes its hash without effect
        bad = [
            f"niche.{f.name} is inert while niche.farm is false: it must keep its default"
            for f in dataclasses.fields(NicheConfig)
            if f.name != "farm" and getattr(nc, f.name) != getattr(dflt, f.name)
        ]
    else:
        nd = cfg.world.dims
        fg = round(cfg.world.size / cfg.world.field_cell)
        n_cells = fg * fg * (round(cfg.world.height / cfg.world.field_cell) if nd == 3 else 1)
        if not 1 <= nc.farm_rate <= RATE_MAX:
            bad.append(f"niche.farm_rate {nc.farm_rate} must be in [1, {RATE_MAX}]")
        if not 1 <= nc.farm_cap <= CAP_MAX:
            bad.append(f"niche.farm_cap {nc.farm_cap} must be in [1, {CAP_MAX}]")
        elif nc.farm_cap * n_cells > 2**31 - 1:
            bad.append(
                f"niche.farm_cap {nc.farm_cap} x {n_cells} cells overflows the int32 SOIL_TOTAL counter"
            )
        if not 1 <= nc.farm_decay_q16 <= 65535:
            bad.append(f"niche.farm_decay_q16 {nc.farm_decay_q16} must be in [1, 65535] (soil always decays)")
        elif 1 <= nc.farm_cap <= CAP_MAX and nc.farm_cap * nc.farm_decay_q16 > 2**30:
            bad.append("niche.farm_cap x niche.farm_decay_q16 must be <= 2^30 (the dithered loss is int32)")
        if not 0 <= nc.farm_cost <= 2**15 or nc.farm_cost * nc.farm_rate >= 2**20:
            bad.append(
                f"niche.farm_cost {nc.farm_cost} must be in [0, 2^15] with farm_cost x farm_rate < 2^20"
            )
        elif nc.farm_cost * nc.farm_rate * cfg.population.n_max >= 2**31:
            bad.append("niche.farm_cost x farm_rate x population.n_max must be < 2^31 (the per-step due)")
        if nc.farm_rate * cfg.population.n_max >= 2**31:
            bad.append(
                "niche.farm_rate x population.n_max must be < 2^31 (the SOIL_PUT / SOIL_WASTED counters)"
            )
        if not 0.0 <= nc.farm_bonus_frac <= 1.0:
            bad.append(f"niche.farm_bonus_frac {nc.farm_bonus_frac} must be in [0, 1]")
        elif (
            1 <= nc.farm_cap <= CAP_MAX
            and nc.farm_cap * round(nc.farm_bonus_frac * cfg.fields.regen_max / nc.farm_cap * 65536) >= 2**31
        ):
            bad.append("niche.farm_cap x bonus_q16 must be < 2^31 (the integer bonus)")
        elif n_cells * (round(nc.farm_bonus_frac * cfg.fields.regen_max) + 1) >= 2**31:
            bad.append("niche: cells x farm_bonus_frac x regen_max must be < 2^31 (the FARM_YIELD counter)")
        if nc.control not in CONTROLS:
            bad.append(f"niche.control {nc.control!r} must be one of {CONTROLS}")
        if not bad and nc.farm_cost > 0 and nc.control != "mute":
            gross = nc.farm_bonus_frac * cfg.fields.regen_max * 65536 / (nc.farm_cap * nc.farm_decay_q16)
            if gross / nc.farm_cost > ROI_WARN:
                log.warning(
                    "niche.farm_cost %s is low for fields.regen_max %s: the gross return per unit is about "
                    "%.1fx its cost (the design point is 5.5x; use about %.0f); farming will be nearly free",
                    nc.farm_cost,
                    cfg.fields.regen_max,
                    gross / nc.farm_cost,
                    gross / 5.5,
                )
    if bad:
        raise ConfigError("; ".join(bad))


def _snap_dev(s: Any, cfg: Any, heavy: bool) -> Any:
    from alife import featobs

    return featobs.niche_dev(s, cfg, heavy)


def _snap_row(h: Any, cfg: Any) -> Any:
    from alife import featobs

    return featobs.niche_row(h, cfg)


SPEC = ExtSpec(
    name="niche",
    enabled=_enabled,
    layout_on=_enabled,
    token=lambda cfg: "niche.farm@1",
    in_blocks=_in_blocks,
    out_blocks=_out_blocks,
    block_names=("farm", "farm_put"),
    state_fields=_state_fields,
    counters=lambda cfg: _COUNTERS,
    act_flag="FARMING",
    dump_kinds=("niche",),
    stream_tags=("FSOL",),
    metric_prefix="farm_",
    lever_keys=tuple(("niche", k) for k in _KEYS),
    cli_only_keys=(("niche", "control"), ("niche", "silent_init")),
    overlay="niche",
    enable=_enable,
    validate=_validate,
    inputs=_inputs,
    decode=_decode,
    due=_due,
    mask=_mask,
    count=_count,
    check=_check,
    snap_dev=_snap_dev,
    snap_row=_snap_row,
)
