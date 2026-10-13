"""Simulation state (§2), enums for the ledger, counters and diag bits, and host (de)serialisation helpers."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from enum import IntEnum, IntFlag
from typing import Final, NamedTuple

import mlx.core as mx
import numpy as np

from alife import brain, config, fields, layout, ops, rng
from alife.config import Config, dims

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


ERROR_MASK: Final = 0xFFFF


class DeathCause(IntEnum):
    STARVED = 0
    KILLED = 1
    AGED = 2
    CATASTROPHE = 3  # director catastrophe (env.cat_*), only for agents that would otherwise have survived


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


def _pad_rows(a: mx.array, n: int) -> mx.array:
    """Append zero rows so the leading dim becomes ``n``."""
    pad = mx.zeros((n - a.shape[0], *a.shape[1:]), dtype=a.dtype)
    return mx.concatenate([a, pad], axis=0)


def _initial_state(cfg: Config) -> State:
    """The (unevaluated) initial state of §2.2; ``init_state`` evaluates it."""
    d = dims(cfg)
    n, m, n0, L = d.n, d.m, cfg.population.n_init, cfg.world.size
    mu = cfg.mutation
    ik = rng.init_keys(cfg.run.seed)
    rows = mx.arange(n, dtype=mx.int32)
    alive = rows < n0

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
        vel=full((n, 2), 0.0, mx.float32),
        heading=_pad_rows(heading, n),
        h=full((n, d.h), 0.0, mx.float32),
        msg=full((n, layout.MSG), 0.0, mx.float32),
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
        chem=full((layout.NCH, d.fg, d.fg), 0.0, mx.float32),
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
    )
    total = total_energy(s.energy, s.alive, s.resource, cfg)
    return s._replace(total=total, total0=total)


def init_state(cfg: Config) -> State:
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
        "msg": ((n, layout.MSG), f32),
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
        "patch_pos": ((d.pp, 2), f32),
        "patch_vel": ((d.pp, 2), f32),
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
    }


def total_energy(energy: mx.array, alive: mx.array, resource: mx.array, cfg: Config) -> mx.array:
    """``TOTAL = Σ energy + S·n_alive + Σ resource`` as an exact int64 scalar (§10.1)."""
    n_alive = alive.astype(mx.int32).sum().astype(mx.int64)
    return ops.exact_sum(energy) + n_alive * cfg.energy.structure + ops.exact_sum(resource)


def age(step: mx.array, birth_step: mx.array) -> mx.array:
    """Age during step ``step``: ``step − 1 − birth_step`` (int32)."""
    return (step - 1 - birth_step).astype(mx.int32)


def state_to_arrays(s: State) -> dict[str, mx.array]:
    """Field name → array, in canonical order."""
    return dict(zip(State._fields, s, strict=True))


def state_from_arrays(d: Mapping[str, mx.array], cfg: Config | None = None) -> State:
    """Rebuild a State from a name-keyed mapping; the key set must equal ``State._fields`` exactly.

    The one exception: when ``cfg`` is given and every missing key is an ``env_*`` field (a pre-director
    checkpoint), those fields are filled with the inert ``default_env(cfg)``.
    """
    keys, want = set(d), set(State._fields)
    if cfg is not None and keys < want and (want - keys) <= set(ENV_FIELDS):
        d = {**default_env(cfg), **d}
        keys = set(d)
    if keys != want:
        missing, extra = sorted(want - keys), sorted(keys - want)
        raise ValueError(f"state arrays do not match State fields: missing={missing}, extra={extra}")
    return State(*(d[name] for name in State._fields))


def to_numpy(a: mx.array) -> np.ndarray:
    """The only host read path: an owned numpy copy; bfloat16 is widened to float32 on device first (F17)."""
    if a.dtype == mx.bfloat16:
        a = a.astype(mx.float32)
    return np.array(a, copy=True)


def _digest(s: State, names: tuple[str, ...]) -> str:
    h = hashlib.sha256()
    for name in names:
        a = getattr(s, name)
        arr = np.ascontiguousarray(to_numpy(a))
        h.update(f"{name}|{a.dtype}|{tuple(a.shape)}|".encode())
        h.update(arr.tobytes())
    return h.hexdigest()


def state_digest(s: State) -> str:
    """SHA-256 hex over every field (name, dtype, shape, bytes) in ``State._fields`` order."""
    return _digest(s, State._fields)


def legacy_state_digest(s: State) -> str:
    """``state_digest`` restricted to the 38 frozen v0 fields (``V0_FIELDS``, §14.2).

    It equals the digest the engine produced before the ``env_*`` fields existed, so the golden pin and
    bit-identity checks against old trajectories stay comparable. ``state_digest`` also covers ``env_*``.
    """
    return _digest(s, V0_FIELDS)
