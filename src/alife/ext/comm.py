"""The declaration of the communication feature (DESIGN §39, §43): a row of ``ext.REGISTRY``.

Footprint (``comm.enabled``): ``width`` + ``nd`` brain inputs (the heard intensity per call channel, then the
body-frame bearing of the sound) and ``width`` outputs (the next call), appended after the bonds tail (2D
75/39 and 3D 91/43 with only this feature at ``width`` 4). No state field: the existing ``msg`` widens to
``(N, MSG + width)`` through ``Dims.msg_w``. Two counters (``COMM_SPEAKERS``, ``COMM_DUE``) and the keys of
``[comm]``. The implementation lives in ``alife.comm``; imports of it are made at call time because
``config`` imports ``ext``.
"""

from __future__ import annotations

from typing import Any

from alife.ext.base import Block, ExtCtx, ExtSpec

_KEYS = (
    ("comm", "enabled"),
    ("comm", "width"),
    ("comm", "range"),
    ("comm", "atten_pow"),
    ("comm", "kin_exp"),
    ("comm", "cost"),
    ("comm", "control"),
)
_COUNTERS = ("COMM_SPEAKERS", "COMM_DUE")
CONTROLS = ("none", "deaf", "decoy")
WIDTH_MAX = 8  # every channel adds a W1 row and a W2 column to every brain
DUE_ROW_MAX = 1 << 15  # the largest per-row voice due (cost x width): keeps the action due far inside int32
COUNTER_MAX = 1 << 31  # the per-step COMM_DUE counter (int32) is at most n_max x cost x width


def _enabled(cfg: Any) -> bool:
    return bool(cfg.comm.enabled)


def _in_blocks(cfg: Any, nd: int) -> tuple[Block, ...]:
    return (Block("voice", cfg.comm.width, False), Block("voice_bearing", nd, False))


def _out_blocks(cfg: Any, nd: int) -> tuple[Block, ...]:
    return (Block("voice", cfg.comm.width, False),)


def _dims(cfg: Any) -> dict[str, int | float]:
    c = cfg.comm
    return {"voice_w": c.width, "hear_range": c.range if c.range > 0.0 else cfg.sensing.radius}


def _enable(cfg: Any) -> Any:
    """``alife migrate --enable comm``: the config with ``comm.enabled`` on (the other keys keep their
    values; ``--set comm.cost=16`` and the like move them)."""
    import dataclasses

    return dataclasses.replace(cfg, comm=dataclasses.replace(cfg.comm, enabled=True))


def _inputs(s: Any, cfg: Any, ctx: ExtCtx) -> Any:
    from alife import comm

    return comm.ext_inputs(s, cfg, ctx)


def _decode(s: Any, y: Any, ctx: ExtCtx, cfg: Any) -> Any:
    from alife import comm

    return comm.decode(s, y, ctx, cfg)


def _mask(dec: Any, solvent: Any, cfg: Any) -> Any:
    from alife import comm

    return comm.mask(dec, solvent, cfg)


def _due(s: Any, act: Any, y: Any, cfg: Any) -> Any:
    from alife import comm

    return comm.due(s, act, y, cfg)


def _count(s_new: Any, alive0: Any, flows: Any, cfg: Any) -> Any:
    from alife import comm

    return comm.ext_count(s_new, alive0, flows, cfg)


def _validate(cfg: Any) -> None:
    from alife.config import CommConfig, ConfigError

    c, bad, d = cfg.comm, [], CommConfig()
    if not c.enabled:  # inert: every key must keep its default, so no hash changes in vain
        bad = [
            f"comm.{k} is inert while comm.enabled is false: it must keep its default"
            for k in _KEYS_NAMES
            if k != "enabled" and getattr(c, k) != getattr(d, k)
        ]
    else:
        if not 1 <= c.width <= WIDTH_MAX:
            bad.append(f"comm.width {c.width} must be in [1, {WIDTH_MAX}]")
        if c.range != 0.0 and not 0.0 < c.range <= cfg.sensing.radius:
            bad.append(
                f"comm.range {c.range} must be 0 (the sensing radius) or in (0, sensing.radius = "
                f"{cfg.sensing.radius}] (the neighbour list serves no farther)"
            )
        if not c.atten_pow > 0.0:
            bad.append(f"comm.atten_pow {c.atten_pow} must be > 0")
        if not c.kin_exp >= 0.0:
            bad.append(f"comm.kin_exp {c.kin_exp} must be >= 0")
        if not c.cost >= 0.0:
            bad.append(f"comm.cost {c.cost} must be >= 0")
        elif 1 <= c.width <= WIDTH_MAX and c.cost * c.width >= DUE_ROW_MAX:
            bad.append(f"comm.cost {c.cost} x comm.width {c.width} must be < {DUE_ROW_MAX} (the per-row due)")
        elif c.cost * min(max(c.width, 1), WIDTH_MAX) * cfg.population.n_max >= COUNTER_MAX:
            bad.append("comm.cost x comm.width x population.n_max must be < 2^31 (the COMM_DUE counter)")
        if c.control not in CONTROLS:
            bad.append(f"comm.control {c.control!r} must be one of {CONTROLS}")
    if bad:
        raise ConfigError("; ".join(bad))


_KEYS_NAMES = tuple(k for _, k in _KEYS)

SPEC = ExtSpec(
    name="comm",
    enabled=_enabled,
    layout_on=_enabled,
    token=lambda cfg: f"comm{cfg.comm.width}@1",
    in_blocks=_in_blocks,
    out_blocks=_out_blocks,
    block_names=("voice", "voice_bearing"),
    dims=_dims,
    counters=lambda cfg: _COUNTERS,
    act_flag="SPOKE",
    metric_prefix="comm_",
    stream_tags=("FSIG",),
    lever_keys=_KEYS,
    cli_only_keys=(("comm", "control"),),
    overlay="comm",
    enable=_enable,
    validate=_validate,
    inputs=_inputs,
    decode=_decode,
    mask=_mask,
    due=_due,
    count=_count,
)
