"""The feature extension registry (DESIGN §43).

One ``ExtSpec`` row per feature that lands (``REGISTRY``, ordered by ``EXTENSIONS``); PR 0a ships the
machinery
with an empty registry, so every function here is the identity of today's engine while no feature is
registered. Every function reads ``REGISTRY`` at call time (a test installs a stub with ``monkeypatch``) and
depends on the config only, so a loop over ``active(cfg)`` inside the traced step is a static Python branch.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from alife.ext.base import (
    ACT_FLAG_BASE,
    EXTENSIONS,
    MAX_EXTENSIONS,
    TAIL_ORDER,
    BirthCtx,
    Block,
    ExtCtx,
    ExtSpec,
    FieldSpec,
)

__all__ = [
    "ACT_FLAG_BASE",
    "EXTENSIONS",
    "MAX_EXTENSIONS",
    "REGISTRY",
    "TAIL_ORDER",
    "BirthCtx",
    "Block",
    "ExtBlocks",
    "ExtCtx",
    "ExtSpec",
    "FieldSpec",
    "active",
    "check_registry",
    "decode_all",
    "flag_values",
    "layout_active",
    "layout_blocks",
    "layout_features",
    "lever_keys",
    "mask",
    "mask_all",
    "registered_block_names",
    "state_fields",
    "step_inputs",
    "sum_due",
    "validate_all",
]

from alife.ext.arms import SPEC as _ARMS
from alife.ext.bonds import SPEC as _BONDS
from alife.ext.comm import SPEC as _COMM
from alife.ext.niche import SPEC as _NICHE
from alife.ext.structures import SPEC as _STRUCTURES

REGISTRY: tuple[ExtSpec, ...] = (
    _ARMS,
    _STRUCTURES,
    _BONDS,
    _COMM,
    _NICHE,
)  # a feature PR adds ONE row (ordered by ExtSpec.index)

# (owner, ((in_name, width, silent), ...), ((out_name, width, silent), ...), nbr_features): hashable, so that
# ``layout.for_dims`` stays cached on it.
ExtBlocks = tuple[
    tuple[str, tuple[tuple[str, int, bool], ...], tuple[tuple[str, int, bool], ...], tuple[str, ...]], ...
]


def active(cfg: Any) -> tuple[ExtSpec, ...]:
    """The enabled features, in registry (``EXTENSIONS``) order."""
    return tuple(spec for spec in REGISTRY if spec.enabled(cfg))


def layout_active(cfg: Any) -> tuple[ExtSpec, ...]:
    """The features with a brain footprint, in registry order."""
    return tuple(spec for spec in REGISTRY if spec.layout_on(cfg))


def layout_features(cfg: Any) -> str:
    """The ``layout_features`` meta value: the tokens of the footprint features, comma-joined; ``""`` when
    none
    (absent from the meta, which then equals today's)."""
    return ",".join(spec.token(cfg) for spec in layout_active(cfg))


def mask(cfg: Any) -> int:
    """Bit ``i`` is set when extension ``i`` has a brain footprint; ``layout.genome_layout_version(nd,
    mask)``."""
    return sum(1 << spec.index for spec in layout_active(cfg))


def layout_blocks(cfg: Any, nd: int) -> ExtBlocks:
    """The brain-tail blocks of the footprint features in ``TAIL_ORDER``, as ``layout.for_dims`` takes
    them."""
    out = []
    for spec in sorted(layout_active(cfg), key=lambda sp: sp.tail_rank):
        out.append(
            (
                spec.name,
                tuple((b.name, b.width, b.silent) for b in spec.in_blocks(cfg, nd)),
                tuple((b.name, b.width, b.silent) for b in spec.out_blocks(cfg, nd)),
                tuple(spec.nbr_features(cfg, nd)),
            )
        )
    return tuple(out)


def registered_block_names() -> frozenset[str]:
    """Every block name any registered feature can declare, upper-cased (``Layout.IN_*``/``OUT_*`` resolve
    against it: the offset when the block is active, None when its feature is off)."""
    return frozenset(name.upper() for spec in REGISTRY for name in spec.block_names)


def state_fields(cfg: Any, nd: int) -> tuple[FieldSpec, ...]:
    """The extension fields of the state type of ``cfg``, in ``EXTENSIONS`` order."""
    return tuple(f for spec in active(cfg) for f in spec.state_fields(cfg, nd))


def decode_all(s: Any, y: Any, ctx: Any, cfg: Any) -> dict[str, Any]:
    """feature name -> its decoded brain outputs (the ``decode`` hooks of the enabled features; S7)."""
    return {spec.name: spec.decode(s, y, ctx, cfg) for spec in active(cfg) if spec.decode is not None}


def mask_all(dec: dict[str, Any], solvent: Any, cfg: Any) -> dict[str, Any]:
    """The decoded outputs masked by solvency (after S8); features without a ``mask`` hook keep theirs."""
    by = {spec.name: spec for spec in active(cfg)}
    return {n: (by[n].mask(d, solvent, cfg) if by[n].mask is not None else d) for n, d in dec.items()}


def step_inputs(
    s: Any, cfg: Any, ctx: Any = None, by_feature: dict[str, tuple[Any, ...]] | None = None
) -> tuple[Any, ...]:
    """The extension input blocks of one step, (N, width) each, in ``TAIL_ORDER`` then block order: what
    ``sensing.build_inputs(ext_in=)`` concatenates after the base vector. Empty while no footprint feature is
    on. A feature without an ``inputs`` hook contributes zero columns' worth only if it declares no blocks."""
    out: list[Any] = []
    for spec in sorted(layout_active(cfg), key=lambda sp: sp.tail_rank):
        if spec.inputs is not None:
            blocks = tuple(spec.inputs(s, cfg, ctx))
            out.extend(blocks)
            if by_feature is not None:  # the observer channel keeps what a feature's brain tail was fed
                by_feature[spec.name] = blocks
    return tuple(out)


def sum_due(parts: Iterable[Any]) -> Any | None:
    """The ``extra_due`` of ``energy.pay_costs``: the sum of the feature dues (each floored by its own
    module),
    or None when no feature has a due."""
    total = None
    for p in parts:
        if p is not None:
            total = p if total is None else total + p
    return total


def flag_values() -> Mapping[str, int]:
    """``ActFlag`` names of the registered features and their values (fixed by the extension index)."""
    return {spec.act_flag: spec.flag_value for spec in REGISTRY if spec.act_flag}


def lever_keys() -> Mapping[str, tuple[tuple[str, str], ...]]:
    """feature -> every (section, key) it adds to ``Config``."""
    return {spec.name: spec.lever_keys for spec in REGISTRY}


def validate_all(cfg: Any) -> None:
    """Each registered feature's own validation, in registry order (after the base rules of
    ``config.validate``)."""
    for spec in REGISTRY:
        spec.validate(cfg)


def check_registry(registry: tuple[ExtSpec, ...] | None = None) -> list[str]:
    """Structural problems of a registry (empty when sound): order, names, and the collisions of section 5.4
    (no two features claim the same dump kind, colour mode, stream tag, metric prefix, flag or block name)."""
    reg = REGISTRY if registry is None else registry
    problems: list[str] = []
    names = [spec.name for spec in reg]
    if len(set(names)) != len(names):
        problems.append(f"duplicate feature names: {names}")
    for spec in reg:
        if spec.name not in EXTENSIONS:
            problems.append(f"{spec.name!r} is not in EXTENSIONS {EXTENSIONS}")
    idx = [spec.index for spec in reg if spec.name in EXTENSIONS]
    if idx != sorted(idx):
        problems.append(f"registry is not in EXTENSIONS order: {names}")
    for what, claims in (
        ("dump kind", [(s.name, k) for s in reg for k in s.dump_kinds]),
        ("colour mode", [(s.name, k) for s in reg for k in s.color_modes]),
        ("stream tag", [(s.name, k) for s in reg for k in s.stream_tags]),
        ("metric prefix", [(s.name, s.metric_prefix) for s in reg if s.metric_prefix]),
        ("act flag", [(s.name, s.act_flag) for s in reg if s.act_flag]),
    ):
        seen: dict[str, str] = {}
        for owner, key in claims:
            if key in seen:
                problems.append(f"{what} {key!r} claimed by {seen[key]} and {owner}")
            seen[key] = owner
    return problems
