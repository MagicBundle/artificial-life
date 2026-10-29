"""Simulation state (§2), enums for the ledger, counters and diag bits, and host (de)serialisation helpers."""

from __future__ import annotations

import functools
import hashlib
import math
import typing
from collections.abc import Mapping
from enum import IntEnum, IntFlag
from typing import Final, NamedTuple

import mlx.core as mx
import numpy as np

from alife import brain, config, ext, fields, fields_sph, layout, ops, rng, weather
from alife.config import Config, dims, is_3d

R_HARD: Final = config.R_HARD  # (1 << 24) − 1: hard per-cell resource cap (§10.9)


class LedgerIdx(IntEnum):
    """Energy ledger entries (§10.8): int64, per step in ``StepOut.ledger``, cumulative in ``ledger_cum``."""

    INJECT_REGEN = 0
    INJECT_IMMIGRATION = 1
    HEAT_BASAL = 2
    HEAT_ACTION = 3
    HEAT_SHARE = 4
    HEAT_BIRTH = 5
    HEAT_OVERFLOW = 6
    FLOW_HARVEST = 7
    FLOW_STOLEN = 8
    FLOW_ATTACK_GAIN = 9
    FLOW_SPILL = 10
    FLOW_SHARED = 11
    FLOW_CORPSE = 12
    FLOW_ENDOWMENT = 13
    FLOW_STRUCTURE = 14
    RESERVED = 15  # canonical name; always 0 in 2D
    HEAT_BUOYANCY = 15  # 3D alias of slot 15 (§27.1): not iterated, not in len(); see ledger_names


N_LEDGER: Final = len(LedgerIdx)
SOURCES: Final = (LedgerIdx.INJECT_REGEN, LedgerIdx.INJECT_IMMIGRATION)
SINKS: Final = (
    LedgerIdx.HEAT_BASAL,
    LedgerIdx.HEAT_ACTION,
    LedgerIdx.HEAT_SHARE,
    LedgerIdx.HEAT_BIRTH,
    LedgerIdx.HEAT_OVERFLOW,
)


def sinks(nd: int) -> tuple[LedgerIdx, ...]:
    """The heat sinks of an ``nd``-dimensional world: ``SINKS``, plus slot 15 (``HEAT_BUOYANCY``) in 3D."""
    return SINKS + ((LedgerIdx.HEAT_BUOYANCY,) if nd == 3 else ())


def ledger_names(nd: int) -> tuple[str, ...]:
    """Ledger entry names in index order; slot 15 is ``RESERVED`` in 2D and ``HEAT_BUOYANCY`` in 3D.

    Use this rather than ``LedgerIdx.HEAT_BUOYANCY.name``, which is ``RESERVED`` (an alias has no own name).
    """
    return tuple("HEAT_BUOYANCY" if nd == 3 and i == LedgerIdx.HEAT_BUOYANCY else i.name for i in LedgerIdx)


class CounterIdx(IntEnum):
    """Per-step int32 counters (§19.1), ``StepOut.counters``."""

    N_ALIVE = 0
    BIRTHS = 1
    BIRTH_REQUESTS = 2
    BIRTHS_BLOCKED = 3
    IMMIGRANTS = 4
    DEATHS_STARVED = 5
    DEATHS_KILLED = 6
    DEATHS_AGED = 7
    DEATHS_UNLOGGED = 8
    INSOLVENT = 9
    ATTACKS = 10
    ATTACK_HITS = 11
    BONDS_FORMED = 12
    BONDS_BROKEN = 13
    CHILD_BONDS = 14
    BONDS_TOTAL = 15
    SHARERS = 16
    EMITTERS = 17
    CELL_OVERFLOW = 18
    MAX_OCCUPANCY = 19
    V_CLIPS = 20
    KIN_ATTACK_SUM = 21
    KIN_ATTACK_BASE = 22
    KIN_BOND_SUM = 23
    KIN_BOND_BASE = 24
    BOND_PROPOSERS = 25
    DEATHS_CATASTROPHE = 26


N_COUNTERS: Final = len(CounterIdx)


class Counter3D(IntEnum):
    """3D-only per-step int32 counters (§27.8), appended after ``CounterIdx`` in a 3D ``StepOut.counters``."""

    Z_FLOOR_CLAMPS = 27  # survivors stopped by the floor this step (pre-clamp z ≤ z_lo, falling)
    Z_SURFACE_CLAMPS = 28  # survivors stopped by the surface (pre-clamp z ≥ z_hi, rising)
    GROUNDED = 29  # alive agents grounded at the start of the step (physics.grounded)


CounterBlocks = tuple[tuple[str, tuple[str, ...]], ...]  # (block, counter names), in step order


def n_counters(nd: int, weather: bool = False, blocks: CounterBlocks = ()) -> int:
    """The length of ``StepOut.counters`` in an ``nd``-dimensional world: 27 in 2D, 30 in 3D, one more with
    the weather layer on (``DEATHS_STORM``, the last base entry, §30), then the extension ``blocks`` of the
    config (``counter_blocks(cfg)``, DESIGN §43.4; none for a feature-off config)."""
    return (
        N_COUNTERS
        + (len(Counter3D) if nd == 3 else 0)
        + (1 if weather else 0)
        + sum(len(names) for _, names in blocks)
    )


def counter_blocks(cfg: Config) -> CounterBlocks:
    """The extension counter blocks of ``cfg`` after ``DEATHS_STORM``, in order: ``protect`` (when
    ``combat.protect_floor > 0``: ``PROT_CAPPED``), then each enabled feature's block in ``EXTENSIONS`` order.
    Every base index stays fixed whatever is on (``storm_counter_index`` does not move)."""
    prot: CounterBlocks = (("protect", ("PROT_CAPPED",)),) if cfg.combat.protect_floor > 0 else ()
    return prot + tuple((spec.name, names) for spec in ext.active(cfg) if (names := spec.counters(cfg)))


def counter_index(cfg: Config, name: str) -> int:
    """The index of the counter ``name`` in ``StepOut.counters`` of ``cfg``: a ``CounterIdx``, ``Counter3D``
    or ``DEATHS_STORM`` name, or an extension counter (names are globally unique). Consumers use this, never
    offsets."""
    nd = dims(cfg).nd
    if name in CounterIdx.__members__:
        return int(CounterIdx[name])
    if nd == 3 and name in Counter3D.__members__:
        return int(Counter3D[name])
    if name == "DEATHS_STORM" and cfg.weather.enabled:
        return storm_counter_index(nd)
    at = n_counters(nd, cfg.weather.enabled)
    for _, names in counter_blocks(cfg):
        if name in names:
            return at + names.index(name)
        at += len(names)
    raise KeyError(f"no counter {name!r} in a {nd}D world with this config")


def storm_counter_index(nd: int) -> int:
    """The index of ``DEATHS_STORM`` in a weather-on ``StepOut.counters``: last, after ``Counter3D`` in 3D, so
    no existing index moves (27 in 2D, 30 in 3D)."""
    return N_COUNTERS + (len(Counter3D) if nd == 3 else 0)


class Diag(IntFlag):
    """Sticky invariant bits OR-ed into ``state.diag`` (§22.3). Bits below 1<<16 are errors."""

    LEDGER = 1
    ENERGY_RANGE = 2
    DEAD_DIRTY = 4
    BOND_ASYM = 8
    NONFINITE = 16
    FIELD_RANGE = 32
    INT_HEADROOM = 64
    CAPACITY = 128
    POS_RANGE = 256
    CELL_OVERFLOW_WARN = 1 << 16
    VCLIP_WARN = 1 << 17
    WEATHER_WARN = (
        1 << 18
    )  # the W0 health gate of the weather layer fired, or a wind sample was clipped (§30)


ERROR_MASK: Final = 0xFFFF


class DeathCause(IntEnum):
    STARVED = 0
    KILLED = 1
    AGED = 2
    CATASTROPHE = 3  # director catastrophe (env.cat_*), only for agents that would otherwise have survived
    STORM = 4  # weather hazard (weather.storm_*), only for agents that would otherwise have survived


class State(NamedTuple):
    """The whole simulation state; field order is canonical (§2.2)."""

    alive: mx.array
    uid: mx.array
    parent_uid: mx.array
    clade: mx.array
    generation: mx.array
    birth_step: mx.array
    energy: mx.array
    pos: mx.array
    vel: mx.array
    heading: mx.array
    h: mx.array
    msg: mx.array
    last_loss: mx.array
    last_harvest: mx.array
    last_inflow: mx.array
    partner: mx.array
    rev: mx.array
    tag: mx.array
    mut_sigma: mx.array
    mut_rate: mx.array
    offspring_frac: mx.array
    brain_cost: mx.array
    W1: mx.array
    W2: mx.array
    pend_slot: mx.array
    pend_W1: mx.array
    pend_W2: mx.array
    resource: mx.array
    chem: mx.array
    patch_pos: mx.array
    patch_vel: mx.array
    key: mx.array
    step: mx.array
    next_uid: mx.array
    total: mx.array
    total0: mx.array
    ledger_cum: mx.array
    diag: mx.array
    # Live environment levers (docs/DIRECTOR.md §3). Flat so checkpoints stay name-keyed; ``State.env`` groups
    # them. Read by the compiled step only when ``cfg.director.enabled``; inert values change nothing.
    env_regen_mult: mx.array  # () f32: multiplier on per-cell regeneration
    env_season_amp: mx.array  # () f32: replaces fields.season_amplitude
    env_season_period: mx.array  # () i32: replaces fields.season_period (0 = off)
    env_drift_mult: mx.array  # () f32: multiplier on patch drift velocity
    env_patch_gain: mx.array  # (PP,) f32: per-patch fertility gain (0 disables a patch)
    env_imm_request: mx.array  # () i32: immigrants still to admit (consumed as lanes allow)
    env_cat_center: mx.array  # (2,) f32: catastrophe centre
    env_cat_radius: mx.array  # () f32: catastrophe radius (0 = none)
    env_cat_kill_p: mx.array  # () f32: per-agent kill probability inside the radius (reset after one step)
    # The weather (docs/STORMS.md, DESIGN §30): the nine WeatherState arrays stacked, (N_WX, wn, wn) f32.
    # Stored at its rest state and carried unchanged while ``cfg.weather.enabled`` is false.
    wx: mx.array

    @property
    def env(self) -> Env:
        """Read-only grouped view of the ``env_*`` fields (the fields stay flat)."""
        return Env(*(getattr(self, name) for name in ENV_FIELDS))


class Env(NamedTuple):
    """The ``env_*`` fields of ``State`` without the prefix."""

    regen_mult: mx.array
    season_amp: mx.array
    season_period: mx.array
    drift_mult: mx.array
    patch_gain: mx.array
    imm_request: mx.array
    cat_center: mx.array
    cat_radius: mx.array
    cat_kill_p: mx.array


ENV_FIELDS: Final = tuple(f"env_{name}" for name in Env._fields)
# The 38 fields of the v0 layout, frozen: ``legacy_state_digest`` hashes exactly these, in this order.
V0_FIELDS: Final = State._fields[: State._fields.index("env_regen_mult")]


def default_env(cfg: Config) -> dict[str, mx.array]:
    """Inert ``env_*`` arrays: a director-enabled run with these values matches a director-off run (D1)."""
    f32, i32 = mx.float32, mx.int32
    return {
        "env_regen_mult": mx.array(1.0, dtype=f32),
        "env_season_amp": mx.array(cfg.fields.season_amplitude, dtype=f32),
        "env_season_period": mx.array(cfg.fields.season_period, dtype=i32),
        "env_drift_mult": mx.array(1.0, dtype=f32),
        "env_patch_gain": mx.ones((dims(cfg).pp,), dtype=f32),
        "env_imm_request": mx.array(0, dtype=i32),
        "env_cat_center": mx.zeros((2,), dtype=f32),
        "env_cat_radius": mx.array(0.0, dtype=f32),
        "env_cat_kill_p": mx.array(0.0, dtype=f32),
    }


# Per-agent fields (leading dim N), in canonical order.
AGENT_FIELDS: Final = State._fields[: State._fields.index("W2") + 1]


class State3(NamedTuple):
    """The state of a 3D world (§27.4): ``State``'s 48 fields in the same order, then ``EXTRA_3D_FIELDS``.

    ``pos`` and ``vel`` are (N,3), ``resource`` is (FZ,FG,FG) and ``chem`` (NCH,FZ,FG,FG); every other shared
    field is as in ``State``. A parallel type, so ``State``, its digests and 2D checkpoints never change.
    """

    alive: mx.array
    uid: mx.array
    parent_uid: mx.array
    clade: mx.array
    generation: mx.array
    birth_step: mx.array
    energy: mx.array
    pos: mx.array
    vel: mx.array
    heading: mx.array
    h: mx.array
    msg: mx.array
    last_loss: mx.array
    last_harvest: mx.array
    last_inflow: mx.array
    partner: mx.array
    rev: mx.array
    tag: mx.array
    mut_sigma: mx.array
    mut_rate: mx.array
    offspring_frac: mx.array
    brain_cost: mx.array
    W1: mx.array
    W2: mx.array
    pend_slot: mx.array
    pend_W1: mx.array
    pend_W2: mx.array
    resource: mx.array
    chem: mx.array
    patch_pos: mx.array
    patch_vel: mx.array
    key: mx.array
    step: mx.array
    next_uid: mx.array
    total: mx.array
    total0: mx.array
    ledger_cum: mx.array
    diag: mx.array
    env_regen_mult: mx.array
    env_season_amp: mx.array
    env_season_period: mx.array
    env_drift_mult: mx.array
    env_patch_gain: mx.array
    env_imm_request: mx.array
    env_cat_center: mx.array  # (2,): the catastrophe is a vertical column
    env_cat_radius: mx.array
    env_cat_kill_p: mx.array
    wx: mx.array  # (N_WX, wn, wn): the weather, as in State
    pitch: mx.array  # (N,) f32 in [−PITCH_MAX, PITCH_MAX]; stale on dead rows, like heading
    buoy_cap: mx.array  # (N,) f32 in [0, 1]: heritable buoyancy cap (effective cap buoy_cap·buoyancy_max)

    env = State.env  # the same read-only view (unannotated, so not a field)


EXTRA_3D_FIELDS: Final = ("pitch", "buoy_cap")  # appended after State's fields (per-agent, leading dim N)
AGENT_FIELDS_3D: Final = AGENT_FIELDS + EXTRA_3D_FIELDS

# The 49 fields of ``State3`` as they stood before ``wx`` (docs/STORMS.md §3.3, DESIGN §30), frozen as a
# literal: ``legacy_state_digest_3d`` hashes exactly these, in this order, so a field added in the middle of
# ``State3`` cannot change it. A trajectory-neutral field is frozen out of the legacy digests like this one.
V0_FIELDS_3D: Final = (
    "alive",
    "uid",
    "parent_uid",
    "clade",
    "generation",
    "birth_step",
    "energy",
    "pos",
    "vel",
    "heading",
    "h",
    "msg",
    "last_loss",
    "last_harvest",
    "last_inflow",
    "partner",
    "rev",
    "tag",
    "mut_sigma",
    "mut_rate",
    "offspring_frac",
    "brain_cost",
    "W1",
    "W2",
    "pend_slot",
    "pend_W1",
    "pend_W2",
    "resource",
    "chem",
    "patch_pos",
    "patch_vel",
    "key",
    "step",
    "next_uid",
    "total",
    "total0",
    "ledger_cum",
    "diag",
    "env_regen_mult",
    "env_season_amp",
    "env_season_period",
    "env_drift_mult",
    "env_patch_gain",
    "env_imm_request",
    "env_cat_center",
    "env_cat_radius",
    "env_cat_kill_p",
    "pitch",
    "buoy_cap",
)


# The sphere world (DESIGN §46.2): StateS is State3's fields in the same order, then ``hdg``, the tangent
# heading vector (the scalar ``heading`` stays, derived and output only). Built by the same NamedTuple call
# as ``_ext_cls`` so State3 and its 49-field digest are untouched. On a sphere ``pos`` and ``vel``
# are Cartesian planet-frame (N,3), ``resource`` is (FZ,Npix), ``chem`` (NCH,FZ,Npix), ``patch_pos`` and
# ``patch_vel`` are (PP,3) (unit vectors and angular velocities), ``env_cat_center`` is (lon, lat) and ``wx``
# is (N_WX_SPHERE, 12·Nw²).
StateS = typing.NamedTuple("StateS", [*State3.__annotations__.items(), ("hdg", mx.array)])
StateS.__module__ = __name__
StateS.__doc__ = (
    "The state of a sphere world (§46.2): ``State3``'s 50 fields in the same order, then ``hdg``."
)
StateS.env = State.env  # type: ignore[attr-defined]  # the same read-only view

EXTRA_SPHERE_FIELDS: Final = ("hdg",)  # appended after State3's fields (per-agent, leading dim N)
AGENT_FIELDS_S: Final = AGENT_FIELDS_3D + EXTRA_SPHERE_FIELDS
N_WX_SPHERE: Final = 11  # h1, h2, the two layers' tangent velocities (6 planes), Q, P, Dc (DESIGN §46.6)
WX_RATIO_SPHERE: Final = (
    2  # the placeholder's weather Nside is Nw = 2·Nf, as the presets (spec 7.3: 16, 22, 42)
)


def wx_shape_sphere(nside: int) -> tuple[int, int]:
    """The shape of ``wx`` on a sphere: ``(N_WX_SPHERE, 12·Nw²)`` with ``Nw = WX_RATIO_SPHERE·Nf``, from the
    field Nside and not from the box's ``weather.cells`` (0.13 to 0.93 MB for the three presets). A
    placeholder until
    the weather (S4) fixes ``Nw``; if S4 changes the ratio it is a planned re-pin of the sphere digests."""
    return N_WX_SPHERE, 12 * (WX_RATIO_SPHERE * nside) ** 2


def is_sphere_state(s: object) -> bool:
    """True for a sphere state of any extension set (``"hdg" in type(s)._fields``)."""
    return "hdg" in type(s)._fields


def is_3d_state(s: object) -> bool:
    """True for a 3D state of any extension set (``"pitch" in type(s)._fields``): the type test that replaces
    ``isinstance(s, State3)``, since a state with extension fields is a generated class (DESIGN §43.2)."""
    return "pitch" in type(s)._fields


@functools.cache
def _ext_cls(nd: int, names: tuple[str, ...], sph: bool = False) -> type:
    """The state type with the extension fields ``names`` after the whole base type (``State``/``State3``, or
    ``StateS`` on a sphere), built once and cached; ``env`` is attached (the read-only view)."""
    base = StateS if sph else State3 if nd == 3 else State
    cls = typing.NamedTuple(base.__name__, [*base.__annotations__.items(), *((n, mx.array) for n in names)])
    cls.__module__ = __name__
    cls.env = State.env  # type: ignore[attr-defined]
    return cls


def state_cls(cfg: Config) -> type[State] | type[State3] | type[StateS]:
    """The state type of a config: ``StateS`` for a sphere world, ``State3`` for a 3D world, else ``State``;
    with an enabled feature that has state fields, a generated class with those fields appended after the base
    fields (the exact class when no extension field is on, so feature-off states, digests and checkpoints are
    today's)."""
    nd = 3 if is_3d(cfg) else 2
    sph = dims(cfg).sph
    names = tuple(f.name for f in ext.state_fields(cfg, nd))
    if not names:
        return StateS if sph else State3 if nd == 3 else State
    return _ext_cls(nd, names, sph)  # type: ignore[return-value]


def agent_fields(cfg: Config) -> tuple[str, ...]:
    """The per-agent (leading dim N) fields of a config's state type, in canonical order."""
    nd = 3 if is_3d(cfg) else 2
    extra = tuple(f.name for f in ext.state_fields(cfg, nd) if f.per_agent)
    return (AGENT_FIELDS_S if dims(cfg).sph else AGENT_FIELDS_3D if nd == 3 else AGENT_FIELDS) + extra


def _pad_rows(a: mx.array, n: int) -> mx.array:
    """Append zero rows so the leading dim becomes ``n``."""
    pad = mx.zeros((n - a.shape[0], *a.shape[1:]), dtype=a.dtype)
    return mx.concatenate([a, pad], axis=0)


def volume_kinematics(
    k_pos: mx.array, k_head: mx.array, n: int, cfg: Config
) -> tuple[mx.array, mx.array, mx.array]:
    """``(pos (n,3), heading (n,), pitch (n,))`` uniform in the 3D volume (§27.4): founders and immigrants.

    Positions are uniform in ``[0, L)² × [z_lo, z_hi]`` from ``k_pos``; yaw is uniform in ``[−π, π)`` and
    pitch in ``[−PITCH_MAX, PITCH_MAX]`` from one ``(n, 2)`` draw of ``k_head``. The keys are the 2D ones
    (``InitStream.POS``/``HEADING``, or the ``IMMIGRANT`` split's) with 3D-only shapes: no new stream and no
    RNG layout bump.
    """
    d, L, pm = dims(cfg), cfg.world.size, layout.PITCH_MAX
    lo, hi = mx.array([0.0, 0.0, d.z_lo]), mx.array([L, L, d.z_hi])
    pos = mx.random.uniform(lo, hi, (n, 3), key=k_pos)
    angles = mx.random.uniform(mx.array([-math.pi, -pm]), mx.array([math.pi, pm]), (n, 2), key=k_head)
    heading, pitch = ops.wrap_angle(angles[:, 0]), mx.clip(angles[:, 1], -pm, pm)
    return ops.wrap_xy_clamp_z(pos, L, d.z_lo, d.z_hi), heading, pitch


def sphere_kinematics(
    k_pos: mx.array, k_head: mx.array, n: int, cfg: Config
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """``(pos (n,3), hdg (n,3), heading (n,), pitch (n,))`` uniform in the shell by volume (DESIGN §46.5):
    founders and immigrants. One ``(n, 3)`` uniform draw ``(a, b, c)`` of ``k_pos`` gives the direction
    (``cos θ = 2a − 1``, ``λ = 2πb``, equal area) and the distance ``r³ = r_lo³ + c·(r_hi³ − r_lo³)`` (equal
    volume), with the float32 bounds ``Dims.r_lo``/``r_hi``; one ``(n, 2)`` draw of ``k_head`` gives the yaw
    and the pitch exactly as in the box. The yaw is the bearing from local north: ``hdg = cos(yaw)·north +
    sin(yaw)·east``, so ``sphere.bearing(hdg, pos) = yaw``. The keys are the 2D ones with sphere shapes: no
    new stream and no RNG layout bump."""
    d, pm = dims(cfg), layout.PITCH_MAX
    u = mx.random.uniform(0.0, 1.0, (n, 3), key=k_pos)
    lo3, hi3 = d.r_lo**3, d.r_hi**3
    ua, ub, uc = mx.split(u, 3, axis=1)
    rho = mx.clip(mx.power(lo3 + uc[:, 0] * (hi3 - lo3), 1.0 / 3.0), d.r_lo, d.r_hi)
    up = fields_sph.sphere_point(ua[:, 0], ub[:, 0])
    angles = mx.random.uniform(mx.array([-math.pi, -pm]), mx.array([math.pi, pm]), (n, 2), key=k_head)
    yaw, pitch = ops.wrap_angle(angles[:, 0]), mx.clip(angles[:, 1], -pm, pm)
    east, north = ops.local_frame_s(up)
    hdg = mx.cos(yaw)[:, None] * north + mx.sin(yaw)[:, None] * east
    return up * rho[:, None], hdg, yaw, pitch


def rest_wx(cfg: Config, step: mx.array | int = 0) -> mx.array:
    """The rest ``State.wx`` of a config: ``weather.rest_array`` in a box; on a sphere an all-zero
    ``wx_shape_sphere`` placeholder (the weather of the sphere, and its rest state, are S4's)."""
    if dims(cfg).sph:
        return mx.zeros(wx_shape_sphere(dims(cfg).nside), dtype=mx.float32)
    return weather.rest_array(cfg, step)


def _initial_state(cfg: Config) -> State | State3 | StateS:
    """The (unevaluated) initial state of §2.2 (``State3`` in 3D, §27.4; ``StateS`` on a sphere, §46.2);
    ``init_state`` evaluates it."""
    d = dims(cfg)
    n, m, n0, L = d.n, d.m, cfg.population.n_init, cfg.world.size
    mu = cfg.mutation
    ik = rng.init_keys(cfg.run.seed)
    rows = mx.arange(n, dtype=mx.int32)
    alive = rows < n0

    if d.sph:  # static: uniform in the shell by volume, from the 2D keys with sphere shapes (§46.5)
        pos, hdg, heading, pitch = sphere_kinematics(
            ik[rng.InitStream.POS], ik[rng.InitStream.HEADING], n0, cfg
        )
    elif d.d3:  # static: uniform in the volume, from the 2D keys with 3D-only shapes (§27.4)
        pos, heading, pitch = volume_kinematics(ik[rng.InitStream.POS], ik[rng.InitStream.HEADING], n0, cfg)
    else:
        pos = ops.wrap(mx.random.uniform(0.0, L, (n0, 2), key=ik[rng.InitStream.POS]), L)
        heading = ops.wrap_angle(mx.random.uniform(-math.pi, math.pi, (n0,), key=ik[rng.InitStream.HEADING]))
    tag = mx.random.uniform(-1.0, 1.0, (n0, layout.T), key=ik[rng.InitStream.TAG])
    w1, w2 = brain.init_weights(ik[rng.InitStream.W1], ik[rng.InitStream.W2], n0, cfg)
    brain_cost = brain.l1_cost(w1, w2, cfg)
    patch_pos, patch_vel = fields.init_patches(
        ik[rng.InitStream.PATCH_POS], ik[rng.InitStream.PATCH_VEL], cfg
    )
    resource = ops.floor_i32(cfg.fields.resource_cap * fields.fertility(patch_pos, cfg))
    energy = mx.where(alive, cfg.energy.e_init, 0).astype(mx.int32)

    def full(shape: tuple[int, ...], value: float | int, dtype: mx.Dtype) -> mx.array:
        return mx.full(shape, value, dtype=dtype)

    s = State(
        alive=alive,
        uid=mx.where(alive, rows, -1).astype(mx.int64),
        parent_uid=full((n,), -1, mx.int64),
        clade=mx.where(alive, rows, 0),
        generation=full((n,), 0, mx.int32),
        birth_step=full((n,), -1, mx.int32),
        energy=energy,
        pos=_pad_rows(pos, n),
        vel=full((n, d.nd), 0.0, mx.float32),
        heading=_pad_rows(heading, n),
        h=full((n, d.h), 0.0, mx.float32),
        msg=full((n, d.msg_w), 0.0, mx.float32),
        last_loss=full((n,), 0, mx.int32),
        last_harvest=full((n,), 0, mx.int32),
        last_inflow=full((n,), 0, mx.int32),
        partner=full((n, layout.B), -1, mx.int32),
        rev=full((n, layout.B), -1, mx.int32),
        tag=_pad_rows(tag, n),
        mut_sigma=full((n,), mu.sigma_init, mx.float32),
        mut_rate=full((n,), mu.rate_init, mx.float32),
        offspring_frac=full((n,), mu.frac_init, mx.float32),
        brain_cost=_pad_rows(brain_cost, n),
        W1=_pad_rows(w1, n),
        W2=_pad_rows(w2, n),
        pend_slot=mx.arange(n - m, n, dtype=mx.int32),
        pend_W1=full((m, d.w1_rows, d.h), 0, d.wdtype),
        pend_W2=full((m, d.w2_rows, d.lay.D_OUT), 0, d.wdtype),
        resource=resource,
        chem=full((layout.NCH, *d.field_shape), 0.0, mx.float32),
        patch_pos=patch_pos,
        patch_vel=patch_vel,
        key=ik[rng.InitStream.STATE_KEY],
        step=mx.array(0, dtype=mx.int32),
        next_uid=mx.array(n0, dtype=mx.int64),
        total=mx.array(0, dtype=mx.int64),
        total0=mx.array(0, dtype=mx.int64),
        ledger_cum=full((N_LEDGER,), 0, mx.int64),
        diag=mx.array(0, dtype=mx.int32),
        **default_env(cfg),
        wx=rest_wx(cfg),
    )
    if d.sph:  # static: dead rows rest at (0, 0, r_lo) heading along x; every row's cap starts at buoy_init
        s = StateS(
            *s._replace(pos=mx.where(alive[:, None], s.pos, mx.array([0.0, 0.0, d.r_lo]))),
            pitch=_pad_rows(pitch, n),
            buoy_cap=full((n,), mu.buoy_init, mx.float32),
            hdg=mx.where(alive[:, None], _pad_rows(hdg, n), mx.array([1.0, 0.0, 0.0])),
        )
    elif d.d3:  # static: dead rows rest on the floor at (0, 0, z_lo); every row's cap starts at buoy_init
        s = State3(
            *s._replace(pos=mx.where(alive[:, None], s.pos, mx.array([0.0, 0.0, d.z_lo]))),
            pitch=_pad_rows(pitch, n),
            buoy_cap=full((n,), mu.buoy_init, mx.float32),
        )
    if ext_fields := ext.state_fields(cfg, d.nd):  # static: the conditional fields of the enabled features
        s = state_cls(cfg)(*s, *(f.rest(cfg) for f in ext_fields))
    total = total_energy(s.energy, s.alive, s.resource, cfg)
    return s._replace(total=total, total0=total)


def init_state(cfg: Config) -> State | State3 | StateS:
    """Founders in rows ``[0, n_init)``, canonical dead rows elsewhere, seeded by ``run.seed``; evaluated."""
    s = _initial_state(cfg)
    mx.eval(s)
    return s


def expected_layout(cfg: Config) -> dict[str, tuple[tuple[int, ...], mx.Dtype]]:
    """Shape and dtype of every State field under ``cfg``, in canonical order (§2.2, §27.1).

    The one schema: ``load_checkpoint`` and ``full_check`` both compare against it.
    """
    d = dims(cfg)
    n, m, wd = d.n, d.m, d.wdtype
    i32, i64, f32 = mx.int32, mx.int64, mx.float32
    return {
        "alive": ((n,), mx.bool_),
        "uid": ((n,), i64),
        "parent_uid": ((n,), i64),
        "clade": ((n,), i32),
        "generation": ((n,), i32),
        "birth_step": ((n,), i32),
        "energy": ((n,), i32),
        "pos": ((n, d.nd), f32),
        "vel": ((n, d.nd), f32),
        "heading": ((n,), f32),
        "h": ((n, d.h), f32),
        "msg": ((n, d.msg_w), f32),
        "last_loss": ((n,), i32),
        "last_harvest": ((n,), i32),
        "last_inflow": ((n,), i32),
        "partner": ((n, layout.B), i32),
        "rev": ((n, layout.B), i32),
        "tag": ((n, layout.T), f32),
        "mut_sigma": ((n,), f32),
        "mut_rate": ((n,), f32),
        "offspring_frac": ((n,), f32),
        "brain_cost": ((n,), i32),
        "W1": ((n, d.w1_rows, d.h), wd),
        "W2": ((n, d.w2_rows, d.lay.D_OUT), wd),
        "pend_slot": ((m,), i32),
        "pend_W1": ((m, d.w1_rows, d.h), wd),
        "pend_W2": ((m, d.w2_rows, d.lay.D_OUT), wd),
        "resource": (d.field_shape, i32),
        "chem": ((layout.NCH, *d.field_shape), f32),
        "patch_pos": ((d.pp, 3 if d.sph else 2), f32),  # a sphere: unit vectors
        "patch_vel": ((d.pp, 3 if d.sph else 2), f32),  # a sphere: angular velocities
        "key": ((2,), mx.uint32),
        "step": ((), i32),
        "next_uid": ((), i64),
        "total": ((), i64),
        "total0": ((), i64),
        "ledger_cum": ((N_LEDGER,), i64),
        "diag": ((), i32),
        "env_regen_mult": ((), f32),
        "env_season_amp": ((), f32),
        "env_season_period": ((), i32),
        "env_drift_mult": ((), f32),
        "env_patch_gain": ((d.pp,), f32),
        "env_imm_request": ((), i32),
        "env_cat_center": ((2,), f32),
        "env_cat_radius": ((), f32),
        "env_cat_kill_p": ((), f32),
        "wx": (wx_shape_sphere(d.nside) if d.sph else (weather.N_WX, d.wn, d.wn), f32),
        **({"pitch": ((n,), f32), "buoy_cap": ((n,), f32)} if d.d3 else {}),  # State3 (§27.4)
        **({"hdg": ((n, 3), f32)} if d.sph else {}),  # StateS (§46.2): after the State3 fields
        **{f.name: (f.shape(d), f.dtype) for f in ext.state_fields(cfg, d.nd)},  # extension fields (§43.2)
    }


def total_energy(energy: mx.array, alive: mx.array, resource: mx.array, cfg: Config) -> mx.array:
    """``TOTAL = Σ energy + S·n_alive + Σ resource`` as an exact int64 scalar (§10.1)."""
    n_alive = alive.astype(mx.int32).sum().astype(mx.int64)
    return ops.exact_sum(energy) + n_alive * cfg.energy.structure + ops.exact_sum(resource)


def age(step: mx.array, birth_step: mx.array) -> mx.array:
    """Age during step ``step``: ``step − 1 − birth_step`` (int32)."""
    return (step - 1 - birth_step).astype(mx.int32)


def state_to_arrays(s: State | State3) -> dict[str, mx.array]:
    """Field name → array, in canonical order."""
    return dict(zip(type(s)._fields, s, strict=True))


def state_from_arrays(d: Mapping[str, mx.array], cfg: Config | None = None) -> State | State3:
    """Rebuild a state from a name-keyed mapping; the key set must equal the state type's fields exactly.

    The type is ``state_cls(cfg)``, or without ``cfg`` ``State3`` when the keys include ``EXTRA_3D_FIELDS``.
    The one exception, when ``cfg`` is given: a key set that lacks only fillable fields is completed. A 2D
    file may lack the ``env_*`` fields (pre-director) and ``wx`` (pre-weather), a 3D file ``wx``. ``env_*``
    get the inert ``default_env(cfg)``; ``wx`` gets ``weather.rest_array(cfg, step)`` with the file's own
    ``step``, so a fork that turns weather on starts from the rest state of its own parameters under the
    pools of the step it resumes at (a calm start: the storms need their spin-up, DESIGN §30).
    """
    keys = set(d)
    cls = (
        state_cls(cfg)
        if cfg is not None
        else StateS
        if set(EXTRA_SPHERE_FIELDS) <= keys
        else State3
        if set(EXTRA_3D_FIELDS) <= keys
        else State
    )
    want = set(cls._fields)
    if cfg is not None and keys < want and "step" in keys:
        ext_names = {f.name for f in ext.state_fields(cfg, dims(cfg).nd)}
        fillable = ((set(ENV_FIELDS) | {"wx"}) if not is_3d(cfg) else {"wx"}) | ext_names
        if (want - keys) <= fillable:
            wx = {"wx": rest_wx(cfg, d["step"])} if "wx" in want - keys else {}
            env = default_env(cfg) if (want - keys) & set(ENV_FIELDS) else {}
            rest = {f.name: f.rest(cfg) for f in ext.state_fields(cfg, dims(cfg).nd) if f.name in want - keys}
            d = {**env, **wx, **rest, **d}
            keys = set(d)
    if keys != want:
        missing, extra = sorted(want - keys), sorted(keys - want)
        raise ValueError(f"state arrays do not match {cls.__name__} fields: missing={missing}, extra={extra}")
    return cls(*(d[name] for name in cls._fields))


# --- layout migration (DESIGN §43.8) ------------------------------------------------------------------------


class MigrationError(ValueError):
    """A layout migration that is refused (a removal, a width change, a different base layout)."""


class LayoutChange(NamedTuple):
    """Which features a migration adds, removes and keeps (by ``ext.active``: footprint or fields)."""

    added: tuple[str, ...]
    removed: tuple[str, ...]
    kept: tuple[str, ...]


_WEIGHT_FIELDS: Final = ("W1", "W2", "pend_W1", "pend_W2", "msg")


def layout_change(cfg_old: Config, cfg_new: Config, *, allow_removal: bool = False) -> LayoutChange:
    """The feature sets of two configs, after the checks every migration needs: the base layout (dims, N, M,
    H, every non-extension field's shape) is the same, a feature kept on has the same ``layout_features``
    token (a width or block-version change is never a migration), and a removal needs ``allow_removal``."""
    old = {sp.name for sp in ext.active(cfg_old)}
    new = {sp.name for sp in ext.active(cfg_new)}
    order = [sp.name for sp in ext.REGISTRY]
    pick = lambda names: tuple(n for n in order if n in names)  # noqa: E731
    change = LayoutChange(pick(new - old), pick(old - new), pick(old & new))
    d_old, d_new = dims(cfg_old), dims(cfg_new)
    for key in ("nd", "n", "m", "h"):
        if getattr(d_old, key) != getattr(d_new, key):
            raise MigrationError(
                f"the base layout differs: {key} {getattr(d_old, key)} -> {getattr(d_new, key)} "
                "(a migration changes features only)"
            )
    ext_old = {f.name for f in ext.state_fields(cfg_old, d_old.nd)}
    ext_new = {f.name for f in ext.state_fields(cfg_new, d_new.nd)}
    want_old, want_new = expected_layout(cfg_old), expected_layout(cfg_new)
    for name, spec_old in want_old.items():
        if name in _WEIGHT_FIELDS or name in ext_old or name in ext_new:
            continue
        if name not in want_new or want_new[name] != spec_old:
            raise MigrationError(f"the base layout differs at {name}: {spec_old} -> {want_new.get(name)}")
    by = {sp.name: sp for sp in ext.REGISTRY}
    for name in change.kept:
        if by[name].token(cfg_old) != by[name].token(cfg_new):
            raise MigrationError(
                f"{name}: layout {by[name].token(cfg_old)!r} -> {by[name].token(cfg_new)!r}: a width or "
                "block-version change is never a migration (start a new run)"
            )
        for fn in by[name].state_fields(cfg_new, d_new.nd):
            if want_old.get(fn.name) != want_new.get(fn.name):
                raise MigrationError(f"{name}: the shape of {fn.name} changes: a restart, not a migration")
    if change.removed and not allow_removal:
        raise MigrationError(
            f"this would turn {', '.join(change.removed)} off, which drops its evolved weights; a removal "
            "is refused unless forced (leave-one-out arms are fresh runs, not removals)"
        )
    return change


def _remap_blocks(
    a: mx.array,
    axis: int,
    base: int,
    old_ext: tuple[tuple[str, str, int, int, bool], ...],
    new_ext: tuple[tuple[str, str, int, int, bool], ...],
    tail_from: int | None,
) -> mx.array:
    """``a`` with its extension blocks moved to the new layout along ``axis``: the base columns, then each new
    block (the old slice, or zeros when the block is new; a block that is gone is dropped), then the columns
    from ``tail_from`` on (the recurrent rows of W1)."""
    old = {(owner, name): (start, width) for owner, name, start, width, _ in old_ext}
    take = lambda lo, hi: a[(slice(None),) * (axis % a.ndim) + (slice(lo, hi),)]  # noqa: E731
    parts = [take(0, base)]
    for owner, name, _, width, _ in new_ext:
        if (owner, name) in old:
            start, w_old = old[(owner, name)]
            if w_old != width:
                raise MigrationError(f"block {owner}.{name}: width {w_old} -> {width}")
            parts.append(take(start, start + width))
        else:
            ax = axis % a.ndim
            parts.append(mx.zeros((*a.shape[:ax], width, *a.shape[ax + 1 :]), a.dtype))
    if tail_from is not None:
        parts.append(take(tail_from, a.shape[axis % a.ndim]))
    return mx.concatenate(parts, axis=axis)


def migrate_layout(
    arrays: Mapping[str, mx.array], cfg_old: Config, cfg_new: Config, *, allow_removal: bool = False
) -> dict[str, mx.array]:
    """The arrays of a state of ``cfg_old`` as the arrays of a state of ``cfg_new`` (DESIGN §43.8): the same
    world on a new feature layout.

    For every extension block of ``cfg_new`` that ``cfg_old`` lacks, in ``TAIL_ORDER``: ``width`` zero rows
    in ``W1`` / ``pend_W1`` between the old inputs and the recurrent rows (which shift), ``width`` zero
    columns in ``W2`` / ``pend_W2``, and zero columns appended to ``msg`` (comm's voice). Blocks present in
    both stay (at their new offsets: enabling a feature whose tail ranks earlier moves the later ones), so the
    order of enabling does not matter. New state fields come from ``FieldSpec.rest(cfg_new)``. A zero row
    feeds nothing into the hidden state and a zero column outputs ``0``, so the migrated population behaves as
    before until mutation wakes the new channels. Refused: a width change, a different base layout, and
    (unless ``allow_removal``) a removal, which drops the weights of the feature and recomputes
    ``brain_cost``. Every other field is carried unchanged."""
    change = layout_change(cfg_old, cfg_new, allow_removal=allow_removal)
    lay_old, lay_new = dims(cfg_old).lay, dims(cfg_new).lay
    base_in = lay_old.D_IN - sum(w for *_, w, _ in lay_old.EXT_IN)
    base_out = lay_old.D_OUT - sum(w for *_, w, _ in lay_old.EXT_OUT)
    if (base_in, base_out) != (
        lay_new.D_IN - sum(w for *_, w, _ in lay_new.EXT_IN),
        lay_new.D_OUT - sum(w for *_, w, _ in lay_new.EXT_OUT),
    ):
        raise MigrationError("the base brain layout differs")
    moved = {
        **{
            name: _remap_blocks(arrays[name], -2, base_in, lay_old.EXT_IN, lay_new.EXT_IN, lay_old.D_IN)
            for name in ("W1", "pend_W1")
        },
        **{
            name: _remap_blocks(arrays[name], -1, base_out, lay_old.EXT_OUT, lay_new.EXT_OUT, None)
            for name in ("W2", "pend_W2")
        },
    }
    msg = arrays["msg"]
    w_old, w_new = msg.shape[1], dims(cfg_new).msg_w
    if w_new > w_old:
        moved = {**moved, "msg": mx.concatenate([msg, mx.zeros((msg.shape[0], w_new - w_old), msg.dtype)], 1)}
    elif w_new < w_old:
        moved = {**moved, "msg": msg[:, :w_new]}
    nd = dims(cfg_new).nd
    new_fields = ext.state_fields(cfg_new, nd)
    gone = {f.name for f in ext.state_fields(cfg_old, nd)} - {f.name for f in new_fields}
    for f in ext.state_fields(cfg_old, nd):
        if f.name in gone and f.conserved and not bool(mx.array_equal(arrays[f.name], f.rest(cfg_old))):
            raise MigrationError(
                f"{f.name} holds conserved quanta that are not at rest: dropping it would leak energy "
                "(settle it first); a removal never drops it silently"
            )
    out = {k: v for k, v in {**arrays, **moved}.items() if k not in gone}
    out = {**{f.name: f.rest(cfg_new) for f in new_fields if f.name not in out}, **out}
    if change.removed:  # the L1 upkeep of the weights that are left
        out = {**out, "brain_cost": brain.l1_cost(out["W1"], out["W2"], cfg_new)}
    mx.eval(list(out.values()))
    cls = state_cls(cfg_new)
    return {name: out[name] for name in cls._fields}


def migrate_state(
    s: State | State3, cfg_old: Config, cfg_new: Config, *, allow_removal: bool = False
) -> State | State3:
    """``s`` (a state of ``cfg_old``) as a state of ``cfg_new`` (``migrate_layout`` over the arrays)."""
    return state_from_arrays(
        migrate_layout(state_to_arrays(s), cfg_old, cfg_new, allow_removal=allow_removal), cfg_new
    )


def to_numpy(a: mx.array) -> np.ndarray:
    """The only host read path: an owned numpy copy; bfloat16 is widened to float32 on device first (F17)."""
    if a.dtype == mx.bfloat16:
        a = a.astype(mx.float32)
    return np.array(a, copy=True)


def _digest(s: State | State3, names: tuple[str, ...]) -> str:
    h = hashlib.sha256()
    for name in names:
        a = getattr(s, name)
        arr = np.ascontiguousarray(to_numpy(a))
        h.update(f"{name}|{a.dtype}|{tuple(a.shape)}|".encode())
        h.update(arr.tobytes())
    return h.hexdigest()


def state_digest(s: State | State3) -> str:
    """SHA-256 hex over every field (name, dtype, shape, bytes) in the state type's ``_fields`` order."""
    return _digest(s, type(s)._fields)


def legacy_state_digest(s: State) -> str:
    """``state_digest`` restricted to the 38 frozen v0 fields (``V0_FIELDS``, §14.2).

    It equals the digest the engine produced before the ``env_*`` fields existed, so the golden pin and
    bit-identity checks against old trajectories stay comparable. ``state_digest`` also covers ``env_*``.
    A 3D state has its own frozen digest (``legacy_state_digest_3d``; ``TypeError`` here).
    """
    if is_3d_state(s):
        raise TypeError("legacy_state_digest is defined for 2D states only (the v0 layout)")
    return _digest(s, V0_FIELDS)


def legacy_state_digest_3d(s: State3) -> str:
    """``state_digest`` restricted to the 49 frozen fields ``State3`` had before ``wx`` (``V0_FIELDS_3D``).

    For an unchanged trajectory it equals the ``state_digest`` the 3D engine produced before the weather layer
    existed, so ``GOLDEN_3D`` and the bit-identity checks against old trajectories stay comparable.
    """
    if not is_3d_state(s):
        raise TypeError("legacy_state_digest_3d is defined for 3D states only")
    return _digest(s, V0_FIELDS_3D)


def legacy_digest(s: State | State3) -> str:
    """The frozen digest of a state's own dimension: ``legacy_state_digest`` for ``State``,
    ``legacy_state_digest_3d`` for ``State3``."""
    return legacy_state_digest_3d(s) if is_3d_state(s) else legacy_state_digest(s)
