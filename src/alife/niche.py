"""Niche construction, farming only (DESIGN §41): the soil an organism builds up and the regrowth it adds.

``State.soil`` is an int32 level per food cell (the lattice of ``resource``), ``0 <= soil <= farm_cap``. An
organism with the brain output ``farm_put`` puts up to ``farm_rate`` units into the cell it occupies at
``farm_cost`` quanta per unit (booked in ``HEAT_ACTION`` through ``extra_due``). The put is dithered
(``floor(farm_rate·put + u)``, ``u`` a hash of the row and the step), so a small output buys a unit with the
matching probability and costs exactly its mean (E17). The soil decays by a dithered integer rule
(``ops.dithered_decay``, a hash of the cell and the step; no RNG stream, E18) and holds no energy.

The effect is one term inside the existing regrowth clip (``fields.update_resource_farm``):
``inc = clip(min(regen_rate + bonus, resource_cap − R), 0, ∞)`` with ``bonus = (soil·bonus_q16) >> 16``
read from the START-of-step soil. ``INJECT_REGEN = Σ inc`` therefore books the bonus (no new source term, no
17th ledger term) and the resource cap still binds. The soil has its own exact identity
``Σsoil' = Σsoil + SOIL_PUT − SOIL_DECAYED`` (``SOIL_PUT`` counts the units the cap accepted; the units a
full cell refused are ``SOIL_WASTED`` and were paid for all the same).

Farming is a public-goods game (the regrowth is for whoever stands there): the gross return per unit is about
``frac·regen_max·65536/(cap·q16)`` quanta over its life and the break-even kin share about 0.18 at the default
cost (DESIGN §41.2).

Pure MLX, no host sync; the one scatter is ``ops.scatter_add_i32``. ``check`` is the host check.
"""

from __future__ import annotations

from typing import Any, Final, NamedTuple

import mlx.core as mx
import numpy as np

from alife import fields, ops, rng
from alife.config import Config, dims

SALT: Final = rng.HASH_SALTS["niche"]  # the put dither: hash32(uid, step, SALT)
DECAY_SALT: Final = (
    SALT ^ 0x5BD1E995
)  # the decay dither: hash32(cell, step, DECAY_SALT), an independent stream
COUNTERS: Final[tuple[str, ...]] = (
    "FARMERS",
    "FARM_YIELD",
    "SOIL_PUT",
    "SOIL_WASTED",
    "SOIL_DECAYED",
    "SOIL_TOTAL",
    "SLOTS_FULL",
)
BLOCKED_ALARM_FRAC: Final = (
    0.5  # E20: more than this share of the steps full (with births blocked) is an alarm
)
_I32 = mx.int32
_F32 = mx.float32


class Put(NamedTuple):
    """The decoded put of a step (``ExtFlows.niche`` before S17)."""

    units: (
        mx.array
    )  # (N,) int32 in [0, farm_rate]: units put this step (0 for dead or, after mask, insolvent)


class Flows(NamedTuple):
    """The step's farming intermediates for the counters (``ExtFlows.niche`` after S17)."""

    units: mx.array  # (N,) int32: units put (paid for), masked by solvency
    put: mx.array  # () int32: units the cap accepted (Σ soil' − Σ soil before decay)
    wasted: mx.array  # () int32: units beyond the cap (paid, not accepted)
    decayed: mx.array  # () int32
    total: mx.array  # () int32: Σ soil after the step
    yield_: mx.array  # () int32: Σ (inc − inc0): the extra regrowth the soil produced after the cap clip


def bonus_q16(cfg: Config) -> int:
    """``round(farm_bonus_frac·regen_max/farm_cap·65536)``: the bonus per unit of soil, in 1/65536 quanta."""
    n = cfg.niche
    return round(n.farm_bonus_frac * cfg.fields.regen_max / n.farm_cap * 65536)


def bonus_of(soil: mx.array, cfg: Config) -> mx.array:
    """The integer regrowth bonus of the soil, ``(soil·bonus_q16) >> 16`` (``<= frac·regen_max``); zero under
    the ``ablate`` control."""
    if cfg.niche.control == "ablate":  # static: the soil is built and paid for, and gives nothing
        return mx.zeros_like(soil)
    return (soil * bonus_q16(cfg)) >> 16


def dither(uid: mx.array, step: mx.array) -> mx.array:
    """A deterministic uniform in ``[0, 1)`` per row from the ``uid`` and the step (24 bits of the hash)."""
    h = ops.hash32(uid, step, SALT)
    return mx.right_shift(h, mx.array(8, dtype=mx.uint32)).astype(_F32) * (1.0 / (1 << 24))


def units_of(y: mx.array, uid: mx.array, step: mx.array, cfg: Config) -> mx.array:
    """``floor(farm_rate·max(tanh(y), 0) + u)`` clipped to ``[0, farm_rate]``, (N,) int32, not masked.

    ``u < 1`` and ``tanh`` is exactly 0 for no output, so silence costs nothing; ``tanh`` reaches 1.0 in
    float32 so full output is exactly ``farm_rate``; in between the units are unbiased (E17)."""
    col = dims(cfg).lay.OUT_FARM_PUT
    assert col is not None, "farm_put needs niche.farm"  # static
    rate = cfg.niche.farm_rate
    put = mx.maximum(mx.tanh(y[:, col].astype(_F32)), 0.0)
    return mx.clip(ops.floor_i32(put * rate + dither(uid, step)), 0, rate)


# --- the hooks of the registry row (ext/niche.py) ---------------------------------------------------------


def inputs(s: Any, cfg: Config, ctx: Any) -> tuple[mx.array]:
    """The brain input ``farm``: ``soil/farm_cap`` at the agent's cell, (N, 1) f32 in [0, 1]."""
    here = s.soil.reshape(-1)[ctx.fcell].astype(_F32) * (1.0 / cfg.niche.farm_cap)
    return (here[:, None],)


def decode(s: Any, y: mx.array, ctx: Any, cfg: Config) -> Put:
    """S7: the units each alive row puts (none under the ``mute`` control)."""
    if cfg.niche.control == "mute":  # static
        return Put(mx.zeros((dims(cfg).n,), dtype=_I32))
    return Put(mx.where(s.alive, units_of(y, s.uid, s.step, cfg), 0).astype(_I32))


def due(s: Any, act: Any, y: mx.array, cfg: Config) -> mx.array | None:
    """S8: ``units·farm_cost`` quanta, an int32 product (None when farming is free or muted)."""
    if cfg.niche.farm_cost == 0 or cfg.niche.control == "mute":  # static
        return None
    u = mx.where(s.alive, units_of(y, s.uid, s.step, cfg), 0).astype(_I32)
    return u * cfg.niche.farm_cost


def mask(dec: Put, solvent: mx.array, cfg: Config) -> Put:
    """Solvency: an insolvent row puts nothing (its cost was due and is paid as far as it holds)."""
    return Put(mx.where(solvent, dec.units, 0).astype(_I32))


def update(
    resource: mx.array,
    patch_pos: mx.array,
    soil: mx.array,
    dec: Put,
    fcell: mx.array,
    cfg: Config,
    step: mx.array,
    env: Any = None,
    rain: Any = None,
) -> tuple[mx.array, mx.array, mx.array, Flows]:
    """S17: regrow the resource with the bonus of the START-of-step soil, then deposit, cap and decay the
    soil. Returns ``(resource', inc, soil', Flows)``; ``inc`` is what ``INJECT_REGEN`` books."""
    d = dims(cfg)
    n = cfg.niche
    resource2, inc, inc0 = fields.update_resource_farm(
        resource, patch_pos, bonus_of(soil, cfg), cfg, step, env, rain
    )
    dep = ops.scatter_add_i32(d.n_cells, fcell, dec.units).reshape(d.field_shape)
    f1 = mx.minimum(soil + dep, n.farm_cap)
    put = (f1 - soil).sum()
    cells = mx.arange(d.n_cells, dtype=_I32).reshape(d.field_shape)
    lost = ops.dithered_decay(f1, n.farm_decay_q16, cells, step, DECAY_SALT)
    soil2 = f1 - lost
    return (
        resource2,
        inc,
        soil2,
        Flows(
            units=dec.units,
            put=put,
            wasted=dec.units.sum() - put,
            decayed=lost.sum(),
            total=soil2.sum(),
            yield_=(inc - inc0).sum(),
        ),
    )


def count(s_new: Any, alive0: mx.array, flows: Any, cfg: Config) -> dict[str, mx.array]:
    """The counter block (names in ``COUNTERS``)."""
    f = flows.ext.niche
    return {
        "FARMERS": (f.units > 0).astype(_I32).sum(),
        "FARM_YIELD": f.yield_,
        "SOIL_PUT": f.put,
        "SOIL_WASTED": f.wasted,
        "SOIL_DECAYED": f.decayed,
        "SOIL_TOTAL": f.total,
        # E20: the population fills the slots (an immortal colony): 1 on a step that ends at
        # N_CAP = n_max - M, the bound the lane allocator never lets it exceed (further requests are refused)
        "SLOTS_FULL": (s_new.alive.astype(_I32).sum() >= dims(cfg).n_cap).astype(_I32),
    }


class FarmActs(NamedTuple):
    """``Acts.ext.niche`` (observer channel, DESIGN §41.14): what each row put into the soil this step."""

    farm_put: mx.array  # (N,) int16: units put (paid for), after the solvency mask; in [0, farm_rate]


def acts_of(f: Flows) -> FarmActs:
    """The observer sub-tuple; quantised from the units the step already holds."""
    return FarmActs(f.units.astype(mx.int16))


def farming(f: Flows) -> mx.array:
    """(N,) bool: the row put something this step (the ``ActFlag.FARMING`` bit, the ``FARMERS`` rows)."""
    return f.units > 0


# --- host check ---------------------------------------------------------------------------------------------


def check(arrays: Any, cfg: Config) -> list[str]:
    """Host check of the numpy ``soil``: in ``[0, farm_cap]``."""
    a = np.asarray(arrays["soil"])
    if (a < 0).any() or (a > cfg.niche.farm_cap).any():
        return [f"soil: values outside [0, {cfg.niche.farm_cap}] (min {a.min()}, max {a.max()})"]
    return []
