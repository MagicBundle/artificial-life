"""Configuration: frozen dataclasses, presets, load/save/override, validation and hashing (§17).

Defaults are the ``64k`` preset. Presets and overlays are sparse TOML overrides shipped as package data.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import importlib.resources
import json
import logging
import math
import tomllib
import typing
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Final
from urllib.parse import urlsplit

import mlx.core as mx
import numpy as np

from alife import ext, fieldtab, geom, layout, ops, weather
from alife.weather import DEFAULT_POOLS as _WEATHER_POOLS

log = logging.getLogger(__name__)

R_HARD: Final = (1 << 24) - 1  # hard resource cap per field cell (§10.9); re-exported as state.R_HARD


class ConfigError(ValueError):
    """Invalid configuration: unknown keys, bad types, or violated validation rules."""


@dataclass(frozen=True)
class WorldConfig:
    size: float = 1024.0  # torus side L
    field_cell: float = 4.0  # FG = size / field_cell (must be an integer ≤ 1024)
    dims: int = 2  # 2 | 3, static (docs/3D.md, §27.1); added after v0.1
    height: float = 0.0  # 3D world height (floor z = 0, surface z = height); 0 in 2D
    # The sphere world (DESIGN §46); added after v0.1 (``SPHERE_KEYS``): all inert while ``geometry = "box"``.
    geometry: str = "box"  # "box" | "sphere", static (changes the compiled step and the state)
    radius: float = 0.0  # planet radius Rp (sphere)
    nside: int = 0  # HEALPix Nside Nf of the field grid (sphere)
    layers: int = 0  # FZ equal-volume shell layers (sphere)


@dataclass(frozen=True)
class PopulationConfig:
    n_max: int = 65536  # N rows (≤ 2^20)
    n_init: int = 16384  # founders, rows [0, n_init)
    max_births_per_step: int = 1024  # M lanes; N_CAP = n_max − M
    max_age: int = 10000  # senescence death; 0 disables
    immigration_floor: int = 0  # 0 disables immigration (static)
    immigrants_per_step: int = 64
    immigration_every: int = 0  # 0 disables periodic immigration (static); added after v0.1
    immigration_count: int = 0  # founders requested on each step t with (t + 1) % every == 0


@dataclass(frozen=True)
class SensingConfig:
    radius: float = 6.0  # R; hash cell ≥ R
    k_neighbors: int = 8  # K
    cell_capacity: int = 16  # C
    tag_scale: float = 0.5
    clock_period: int = 32
    chem_half_sat: float = 1.0


@dataclass(frozen=True)
class BrainConfig:
    hidden: int = 32  # H
    weight_dtype: str = "float16"  # float16 | bfloat16 | float32
    impl: str = "matmul"  # matmul | einsum
    init_scale: float = 1.0
    recurrent_scale: float = 0.5
    weight_clip: float = 4.0


@dataclass(frozen=True)
class PhysicsConfig:
    radius: float = 1.0  # agent radius r
    thrust_max: float = 0.1
    turn_max: float = 0.3
    drag: float = 1.0
    grip_gain: float = 4.0
    v_max: float = 0.5
    k_repel: float = 0.08
    thrust_contact_limit: float = 4.0  # thrust × max(0, 1 − contacts/limit) (§9)
    gravity: float = 0.0  # 3D (§27.3): downward body force on every survivor; added after v0.1
    floor_eps: float = 0.0  # 3D: grounded iff z ≤ radius + floor_eps (the floor only)
    buoyancy_max: float = 0.0  # 3D: upward force at buoy_cap = 1 and full OUT_BUOYANCY; 0 = no actuator
    ceiling_band: float = 0.0  # sphere (§46): width of the soft band under the lid; 0 = off; inert in a box


@dataclass(frozen=True)
class BondConfig:
    rest_length: float = 2.0
    k_spring: float = 0.05
    contract_gain: float = 0.25
    form_range: float = 3.0
    break_length: float = 4.0
    release_threshold: float = 1.0
    share_rate: int = 1000  # q per slot per step at share = 1
    share_efficiency: float = 0.9
    # Bonds that pay (DESIGN §40), post v0.1 (``POST_V01_KEYS`` through ``ext.bonds.SPEC.lever_keys``): inert
    # at their defaults, and ``share_kappa`` / ``share_kin_floor`` only while ``share_mode`` is not 0.
    share_mode: int = 0  # 0 the legacy gift; 1 conductive, mutual consent; 2 conductive, obligate
    share_kappa: float = 0.125  # conductive: fraction of the energy gap sent per step, (0, 1/(2B)]
    share_kin_floor: float = 1.0  # conductive: efficiency scale of a non-kin bond, [0, 1]; 1 = kin-blind


@dataclass(frozen=True)
class EnergyConfig:
    e_init: int = 50000
    e_max: int = 1000000  # < 2^20
    structure: int = 10000  # S
    basal_cost: int = 100
    brain_cost_per_l1: float = 0.1
    bond_upkeep: int = 5
    move_cost: int = 100
    turn_cost: int = 10
    grip_cost: int = 20
    contract_cost: int = 5
    msg_cost: int = 5
    emit_cost: int = 20
    harvest_cost: int = 10
    defend_cost: int = 20
    attack_cost: int = 50
    harvest_max: int = 400
    buoyancy_cost: int = 0  # 3D (§27.7): q per step at the full buoyancy_max lift, ·(lift/buoyancy_max)²
    buoyancy_cap_cost: int = 0  # 3D: q per step per unit of buoy_cap, paid even without lift (upkeep)


@dataclass(frozen=True)
class CombatConfig:
    range: float = 3.0
    threshold: float = 0.5
    attack_max: int = 4000
    efficiency: float = 0.7
    defend_efficiency: float = 0.75
    attack_bonded: bool = False
    # Lower bound of the product of the protections (armour, shelter, guard) on an attack's demand, as a
    # fraction of ``attack_max`` before ``defend`` (DESIGN §43.6). 0 = off: no op traced. Added after v0.1.
    protect_floor: float = 0.0


@dataclass(frozen=True)
class ReproConfig:
    min_energy: int = 40000
    min_age: int = 20
    birth_cost: int = 2000
    offspring_distance: float = 2.0


@dataclass(frozen=True)
class MutationConfig:
    sigma_init: float = 0.03
    sigma_min: float = 0.002
    sigma_max: float = 0.3
    tau_sigma: float = 0.15
    rate_init: float = 0.05
    rate_min: float = 0.001
    rate_max: float = 0.5
    tau_rate: float = 0.15
    frac_init: float = 0.5
    frac_min: float = 0.1
    frac_max: float = 0.9
    tau_frac: float = 0.1
    tag_sigma: float = 0.02
    buoy_init: float = 0.0  # 3D (§27.4): buoy_cap ∈ [0, 1] of founders and immigrants; added after v0.1
    buoy_sigma: float = 0.0  # 3D: additive Gaussian mutation of buoy_cap, clipped to [0, 1] (like tag_sigma)


@dataclass(frozen=True)
class FieldConfig:
    resource_cap: int = 8000
    regen_max: int = 320
    fertility_base: float = 0.05
    n_patches: int = 24
    patch_sigma: float = 48.0
    patch_drift: float = 0.005
    resource_diffusion_q8: int = 4
    chem_diffusion: tuple[float, float, float] = (0.20, 0.10, 0.24)
    chem_decay: tuple[float, float, float] = (0.10, 0.03, 0.005)
    chem_deposit_scale: float = 1.0 / 256.0
    chem_max: float = 16.0
    season_period: int = 0  # steps per seasonal cycle; 0 disables seasons (static); added after v0.1
    season_amplitude: float = 0.0  # A in [0, 1]: patch weight 1 + A*sin(2*pi*t/P + phi_p)
    season_mode: str = "global"  # global (phi_p = 0) | per_patch (phi_p = 2*pi*p/PP)
    benthic_lambda: float = 0.0  # 3D (§27.5): e-folding height of the floor food band; added after v0.1
    photic_amp: float = 0.0  # 3D: amplitude in [0, 1] of the surface food band; 0 = off
    photic_lambda: float = 0.0  # 3D: e-folding depth of the surface band below the surface


@dataclass(frozen=True)
class DirectorConfig:  # ecosystem director (docs/DIRECTOR.md); added after v0.1
    enabled: bool = False  # static: the compiled step reads the live environment state (State.env)
    # Policy keys: read by the host-side director between steps, never by the compiled step. They are not
    # hashed (``UNHASHED_KEYS``) and may change on resume.
    persona: str = "gardener"  # gardener | trickster | architect | random (the yoked control, needs yoke_log)
    goal: str = ""  # the Architect's goal text (stored, not interpreted before the LLM tier)
    rule_every: int = 1000  # steps between rule-tier decisions; a multiple of the metrics cadences
    budget_per_100k: int = 0  # intervention cost units per rolling 100k steps; 0 = the persona's default
    yoke_log: str = ""  # the random persona only: the director.jsonl of the run whose moves it is yoked to
    # The LLM tier (build step 3): off at 0. Where it runs and which model answers are run.llm_* keys.
    llm_every: int = 0  # steps between LLM calls; 0 = off, else a multiple of rule_every
    llm_temperature: float = 0.4  # sampling temperature of the request
    llm_timeout_s: float = 60.0  # seconds to wait for a reply before the rule tier takes over
    llm_max_tokens: int = 400  # reply length limit of the request


@dataclass(frozen=True)
class GameConfig:  # the games vertical (docs/GAMES.md, §28); added after v0.1
    # Observers (§28): the step also returns the observer channel and the runner folds it into accumulators,
    # labels, the metrics block and the sidecar. Never changes the trajectory: unhashed (``OBSERVER_KEYS``)
    # and a resume key.
    observe: bool = False
    # Houses (§29.4), read only by the host observer while ``observe`` is on; unhashed and a resume key. A
    # working default, not "off": an observer-only key (CLAUDE.md). A birth founds a house iff
    # splitmix64(uid, run.seed) < house_rate·2^64; 1/2048 is exact in binary.
    house_rate: float = 1 / 2048
    # The tracer rig (W2, DESIGN §32): passive tracers moved by the wind, the null of "agents against passive
    # drift". Host-observer keys like ``house_rate``: unhashed, a resume key, inert while 0. ``tracers`` is
    # the count (0 = off; needs ``observe`` and ``weather.enabled``), ``tracer_period`` the steps between
    # reseeds at live agents.
    tracers: int = 0
    tracer_period: int = 500


# The default warm pools of ``[weather]`` (``weather.DEFAULT_POOLS``, flat: 7 floats per pool).
_POOLS_FLAT: Final[tuple[float, ...]] = tuple(float(x) for row in _WEATHER_POOLS for x in row)


@dataclass(frozen=True)
class WeatherConfig:  # the storm world's engine weather layer (docs/STORMS.md, DESIGN §30); added after v0.1
    """Every key changes the compiled step, so every key is hashed (``POST_V01_KEYS``) and is inert at its
    default while ``enabled`` is false (``validate``). Field names equal ``weather.WeatherParams``' where both
    exist; the numerics a tuning never touched are ``weather.ENGINE_FIXED``."""

    enabled: bool = False  # static: the compiled step carries the weather update, wind, rain and hazard
    cells: int = 160  # weather grid side n; dx = world.size / cells (a multiple of the field lattice FG)
    # layers: rest depths (H1 + H2 = world.height in 3D) and gravities, Coriolis amplitude
    H1: float = 32.0
    H2: float = 48.0
    g_top: float = 9.18256e-3
    g_prime: float = 5.16862e-3
    f0: float = 0.025
    # moisture
    gamma: float = 21.6
    tau_c: float = 8.0
    k_E: float = 0.1
    u_gust: float = 0.02
    q_bg: float = 0.70
    q_pool: float = 1.11
    # relaxation and damping
    tau_rad: float = 2000.0
    r1: float = 2e-4
    c_d: float = 2e-3
    r2: float = 1e-4
    # warm pools: flat, 7 per pool (x, y, sx, sy, amp, vx, vy)
    pools: tuple[float, ...] = _POOLS_FLAT
    # disturbances
    seed_rate: float = 0.02
    seed_v: float = 0.02
    seed_sigma: float = 12.0
    seed_jitter: float = 0.5
    # the wind the agents feel (STORMS §4)
    alpha0: float = 0.25
    alpha1: float = 0.75
    u_max: float = 0.3
    w_edge: float = 0.03
    z_sl: float = 4.0
    # coupling to the organisms
    rain_gain: float = 0.0  # nominal quanta per field column per step per unit of rain rate; 0 = no rain food
    rain_floor: float = 0.0  # extra share of the rain added to the floor layer (3D)
    storm_p: float = 0.0  # peak per-step storm kill probability; 0 = no hazard
    storm_u0: float = 0.06  # air speed where the hazard starts
    storm_span: float = 0.03  # air speed range over which it ramps to storm_p
    storm_break_u: float = 0.0  # air speed above which a bond slot breaks; 0 = off
    # W2 couplings (docs/STORMS.md §4, DESIGN §32); every one is off (or neutral) at its default
    seed_gain: float = (
        0.0  # rain seeding: the condensation threshold Q_s falls by at most this fraction; 0 = off
    )
    seed_c0: float = 2.0  # c2 excess (over the lattice mean) where the seeding response is half of seed_gain
    shade_gain: float = 0.0  # shade on the seas: T_s falls by at most this fraction under crowds; 0 = off
    shade_n0: float = 8.0  # agents per field column (3x3 mean) where the shade is half of shade_gain
    wind_scale: float = (
        1.0  # multiplies the wind force on the agents; 0 = still air for them, weather runs on
    )
    rain_zonal: bool = False  # the rain food is the x-mean of the rain: latitude only (a control)
    storm_eddy: bool = False  # the hazard and the bond break read the speed relative to the zonal mean
    wind_eddy_force: bool = False  # the wind force uses the eddy wind (u - zonal mean, w); a control


@dataclass(frozen=True)
class RunConfig:  # excluded from config_hash; on resume only observer/cadence keys may change (§18)
    seed: int = 0
    steps: int = 100_000
    metrics_every: int = 100
    checkpoint_every: int = 10_000  # multiple of metrics_every
    keep_checkpoints: int = 3  # plus the step-0 checkpoint, which is always kept
    compile: bool = True
    runtime_checks: str = "cheap"  # cheap | off
    heavy_metrics_every: int = 1000  # components + weight diversity; multiple of metrics_every
    components_iters: int = 32
    diversity_sample: int = 256
    director: str = "off"  # off | live | replay | player (needs director.enabled); resume may change it
    llm_url: str = "http://localhost:1234/v1/chat/completions"  # the LLM tier's endpoint (LM Studio)
    llm_model: str = "gemma-4-26b-a4b-it-mlx@8bit"  # the model named in each request
    # Run dumps (W2, DESIGN §32), cadence keys like ``metrics_every`` (``run`` is outside the hash; a resume
    # key): ``eco_every`` writes ``observers/eco_<step>.npz`` (per-agent ecology of the live rows; needs
    # ``game.observe``, ``weather.enabled`` and a 3D world), ``wx_every`` writes ``observers/wx_<step>.npy``
    # (the ten weather planes; needs ``weather.enabled``). 0 = off.
    eco_every: int = 0
    wx_every: int = 0
    # The feature dumps (DESIGN §38.9, §42.6, §41.14), the same kind of key: ``mat_every`` writes
    # ``observers/mat_<step>.npy`` (the built material; needs ``structures.enabled``), ``arms_every`` writes
    # ``observers/arms_<step>.npz`` (a sample of the arms traits; needs ``arms.enabled``). 0 = off.
    mat_every: int = 0
    arms_every: int = 0
    niche_every: int = 0  # ``observers/niche_<step>.npy`` (the soil; needs ``niche.farm``), DESIGN §41.14


@dataclass(frozen=True)
class ArmsConfig:  # the arms-race layer: heritable attack, armour, key and lock (DESIGN §42); post v0.1
    """Every key changes the compiled step, so every key is hashed (``POST_V01_KEYS`` through
    ``ext.arms.SPEC.lever_keys``) and is inert at its default while ``enabled`` is false (``validate``). All
    defaults are zeros or "off": ``enabled`` alone is an arms layer with inert traits (the null-equivalence
    configuration, DESIGN §42.2). The recommended working set lives in a preset overlay (A3), not here."""

    enabled: bool = False  # static: the contest, the upkeep, the speed cost and the heritable traits exist
    kappa: float = 2.0  # armour weight in the Tullock repulsion r = κA / (κA + a0 + ATK); 0 disables armour
    a0: float = 1.0  # baseline attacker strength in the denominator (> 0)
    key_floor: float = 1.0  # key/lock match floor; 1.0 disables the axis, lower makes a mismatch costlier
    speed_slow: float = 0.0  # thrust factor 1 - speed_slow * ARM in the physics (the cost of armour)
    upkeep_atk: float = 0.0  # q/step at ATK = 1, quadratic, dithered (HEAT_BASAL)
    upkeep_arm: float = 0.0  # q/step at ARM = 1, quadratic, dithered (HEAT_BASAL)
    init_atk: float = 0.0  # founders' and immigrants' attack
    init_arm: float = 0.0  # founders' and immigrants' armour
    sigma: float = 0.0  # fixed mutation sigma of the magnitudes (attack, armour)
    sigma_phase: float = 0.0  # fixed mutation sigma of the key and lock phases (turns)
    dump_n: int = 4096  # rows of the ``arms_<step>.npz`` sample (observer: not hashed, DESIGN §42.6)


@dataclass(frozen=True)
class StructuresConfig:  # built structures: material, shelter, barrier, predator shield (DESIGN §38)
    """Every key changes the compiled step, so every key is hashed (``POST_V01_KEYS`` through
    ``ext.structures.SPEC.lever_keys``) and is inert at its default while ``enabled`` is false (``validate``).
    The defaults are the working set of the spec (all provisional until the first runs); ``enabled`` is the
    one switch that adds the grid ``State.mat``, the brain input ``material`` and the output ``build``."""

    enabled: bool = False  # static: the grid, the sensor, the build action and the three effects exist
    cap: int = 4096  # material units a cell can hold, [256, 32767] (levels fit a byte on the wire)
    build_max: int = 16  # units deposited per step at build output 1, [1, 1024]
    build_cost: int = 2  # quanta per unit built (HEAT_ACTION); 0 = free building
    decay_q12: int = 16  # exponential decay per step in 1/4096 (ceil, so a floor of 1 unit/step), [1, 4096]
    shelter_gain: float = 0.8  # wind force and air speed multiplied by 1 - gain*cover, [0, 1]
    barrier_gain: float = 3.0  # drag multiplied by 1 + gain*here, [0, 3] (R1: at most a 4x slowdown)
    shield_gain: float = 0.5  # the victim's predation demand multiplied by 1 - gain*cover, [0, 1]
    reach: float = 0.5  # face neighbours count at this weight in cover, [0, 1]
    open_frac: float = 0.5  # a cell is OPEN below open_frac*cap (the threshold of R2), (0, 1]
    aperture: float = 0.25  # R2: every cell keeps ceil(aperture*n_nbr) open neighbours; 0 switches R2 off
    exit_ease: float = 0.75  # R3: leaving a wall costs (1 - exit_ease) of the entering drag, [0, 1]
    silent_build: bool = False  # start-of-run: zero the W2 column of build (no founder builds; the control)


@dataclass(frozen=True)
class RolesConfig:  # committed roles on bonds (DESIGN §40); post v0.1
    """Every key changes the compiled step, so every key is hashed (``POST_V01_KEYS`` through
    ``ext.bonds.SPEC.lever_keys``) and is inert at its default while ``enabled`` is false (``validate``).
    ``enabled`` allocates the bond brain block and the role state; the other keys are the role effects (DESIGN
    §40.3), all zero or "off" so ``enabled`` alone is a role machine with no effect on any flow."""

    enabled: bool = False  # static: the role state and the brain block exist (with bonds.share_mode != 0 too)
    forage_gain: float = 0.0  # a forager's harvest demand cap grows by this fraction of harvest_max (ramped)
    forage_defend_pen: float = 0.0  # a forager's own behavioural defence is multiplied by 1 - this (ramped)
    guard_harvest_pen: float = 0.0  # a guard's harvest demand cap shrinks by this fraction (ramped), [0, 1)
    guard_shield: float = 0.0  # protection a guard grants itself and each bonded partner (ramped), [0, 0.9]
    synergy: float = (
        0.0  # extra forager cap beside a guard whose ramp is at least half (fraction of harvest_max)
    )
    ramp_steps: int = 64  # steps after a switch for a role's effects to reach full strength, [1, 32767]
    threshold: float = 0.5  # dead zone of the two role outputs, (0, 1): inside it the role is kept
    switch_cost: int = 0  # quanta paid (HEAT_ACTION) by a row that asks for another role
    inherit_role: bool = False  # a bud starts with its parent's role (dwell 0); else generalist


@dataclass(frozen=True)
class CommConfig:  # the voice channel (DESIGN §39); post v0.1
    """Every key changes the compiled step, so every key is hashed (``POST_V01_KEYS`` through
    ``ext.comm.SPEC.lever_keys``) and is inert at its default while ``enabled`` is false (``validate``).
    ``enabled`` allocates the voice (W call intensities, a brain tail of W + nd inputs and W outputs)."""

    enabled: bool = False  # static: the voice channel, its brain tail and the widened ``msg`` exist
    width: int = 4  # W: call channels per organism, [1, 8]; a run constant
    range: float = 0.0  # hearing range; 0 = sensing.radius, else (0, sensing.radius] (the neighbour list)
    atten_pow: float = 1.0  # falloff exponent of the linear-to-zero distance weight
    kin_exp: float = 0.0  # listen to kin: weight x kin similarity ** kin_exp (0 = everyone is heard)
    cost: float = (
        4.0  # q per unit of call intensity (HEAT_ACTION), dithered so sub-unit calls cost on average
    )
    control: str = "none"  # receiver-side ablation: none | deaf (hear nothing) | decoy (hear a stranger)


@dataclass(frozen=True)
class NicheConfig:  # niche construction, farming only (DESIGN §41); marks and stores are deferred
    """Every key changes the compiled step, so every key is hashed (``POST_V01_KEYS`` through
    ``ext.niche.SPEC.lever_keys``) and is inert at its default while ``farm`` is false (``validate``).
    ``farm`` allocates the soil (``State.soil``, an int32 level per food cell), the brain input ``farm`` and
    the output ``farm_put``. The defaults are the 2D 64k working set (``regen_max`` 320); the 3D worlds
    (``regen_max`` 1700) need ``farm_cost`` about 150 (the gross return scales with ``regen_max``)."""

    farm: bool = False  # static: the soil, its sensor and the put action exist
    farm_rate: int = 8  # units put per step at output 1, [1, 256] (dithered rounding, E17)
    farm_cap: int = 4095  # soil level a cell can hold, [1, 32767]
    farm_cost: int = 30  # quanta per unit put (HEAT_ACTION); 0 = free farming
    farm_decay_q16: int = 16  # dithered exponential decay per step in 1/65536, [1, 65535]: mean life 4096
    farm_bonus_frac: float = 0.5  # bonus at full soil = frac * regen_max quanta/step on top of the regrowth
    silent_init: bool = True  # start-of-run: zero the W2 column of farm_put (no founder farms; the migration)
    control: str = (
        "none"  # fork-only: none | ablate (soil is put and paid but gives no bonus) | mute (no puts)
    )


@dataclass(frozen=True)
class Config:
    world: WorldConfig = field(default_factory=WorldConfig)
    population: PopulationConfig = field(default_factory=PopulationConfig)
    sensing: SensingConfig = field(default_factory=SensingConfig)
    brain: BrainConfig = field(default_factory=BrainConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    bonds: BondConfig = field(default_factory=BondConfig)
    energy: EnergyConfig = field(default_factory=EnergyConfig)
    combat: CombatConfig = field(default_factory=CombatConfig)
    repro: ReproConfig = field(default_factory=ReproConfig)
    mutation: MutationConfig = field(default_factory=MutationConfig)
    fields: FieldConfig = field(default_factory=FieldConfig)
    director: DirectorConfig = field(default_factory=DirectorConfig)
    game: GameConfig = field(default_factory=GameConfig)
    weather: WeatherConfig = field(default_factory=WeatherConfig)
    arms: ArmsConfig = field(default_factory=ArmsConfig)
    structures: StructuresConfig = field(default_factory=StructuresConfig)
    roles: RolesConfig = field(default_factory=RolesConfig)
    comm: CommConfig = field(default_factory=CommConfig)
    niche: NicheConfig = field(default_factory=NicheConfig)
    run: RunConfig = field(default_factory=RunConfig)


@dataclass(frozen=True)
class Dims:
    """Config-derived sizes (§2.1)."""

    n: int
    m: int
    n_cap: int
    dl: int
    k: int
    c: int
    q: int
    g_hash: int
    cell_hash: float
    fg: int
    h: int
    w1_rows: int
    w2_rows: int
    params: int
    pp: int
    wdtype: mx.Dtype
    occ_limit: int  # §10.9: min(2^15−1, (2^31−1−R_HARD)//(2·e_max+S))
    # World dimensions (§27.1). 2D values are the neutral ones (one layer, no z hashing).
    nd: int  # world.dims
    d3: bool  # nd == 3: the static switch of every 3D branch
    lay: layout.Layout  # the genome I/O layout for nd (layout.for_dims)
    fz: int  # field layers FZ = height / field_cell; 1 in 2D
    n_cells: int  # field cells FZ·FG²
    field_shape: tuple[int, ...]  # (FG, FG) in 2D, (FZ, FG, FG) in 3D
    g_hash_z: int  # hash layers floor(height / radius); 1 in 2D
    cell_hash_z: float  # height / g_hash_z; inf in 2D
    # The f32 z bounds every 3D clamp and check uses (§27.2); 0.0 in 2D (unused).
    z_lo: float  # the smallest float32 ≥ physics.radius
    z_hi: float  # the largest float32 ≤ height − physics.radius
    z_ground: float  # max(z_lo, the largest float32 ≤ radius + floor_eps): grounded iff z ≤ z_ground
    wn: int  # weather grid side (weather.cells): the packed State.wx is (N_WX, wn, wn)
    # Feature extensions (DESIGN §43). The neutral values are today's: no voice channel, msg width MSG.
    voice_w: int = 0  # comm: voice channel width W (0 = off)
    msg_w: int = layout.MSG  # State.msg width: MSG + voice_w
    features: tuple[str, ...] = ()  # names of the enabled features, in registry order
    # per enabled feature, the sizes its ``dims(cfg)`` hook declares (voice_w, hear_range, ...): read them
    # with ``d.ext("comm").hear_range``
    ext_dims: tuple[tuple[str, tuple[tuple[str, Any], ...]], ...] = ()
    # The sphere world (DESIGN §46). The neutral values are the box's. On a sphere ``fg``, ``g_hash`` and
    # ``cell_hash`` (the box lattice) are 0/0/inf, ``fz`` is ``world.layers``, ``n_cells = FZ·Npix``,
    # ``field_shape = (FZ, Npix)``, ``q = 125·C``, ``g_hash_z``/``cell_hash_z`` the radial hash layers.
    sph: bool = False
    radius: float = 0.0  # Rp
    nside: int = 0  # field Nside Nf
    npix: int = 0  # 12·Nf² angular field cells
    nside_h: int = 0  # hash Nside Nh (derived from the exact reach-2 coverage)
    npix_h: int = 0
    r_lo: float = 0.0  # Rp + z_lo, the f32 floor radius
    r_hi: float = 0.0  # Rp + z_hi, the f32 lid radius

    def ext(self, name: str) -> SimpleNamespace:
        """The derived sizes of the enabled feature ``name`` as attributes; ``KeyError`` if it is off."""
        for feature, items in self.ext_dims:
            if feature == name:
                return SimpleNamespace(**dict(items))
        raise KeyError(f"feature {name!r} is not enabled")


PRESETS: tuple[str, ...] = ("tiny", "dev", "64k", "256k", "tiny3d", "64k3d", "mini3d", "tiny3s")
OVERLAYS: tuple[str, ...] = ("fast",)  # compose with every preset
# Overlays that need a particular world (DESIGN §30): validated against whatever preset they land on, and not
# part of ``OVERLAYS`` (which means "composes with every preset").
WORLD_OVERLAYS: tuple[str, ...] = (
    "storm",
    "storm160",
    *(s.overlay for s in ext.REGISTRY if s.overlay),
    "niche3d",  # farming at the 3D design point (farm_cost 150): a variant of the niche overlay
)
ALL_OVERLAYS: tuple[str, ...] = OVERLAYS + WORLD_OVERLAYS  # what the loaders and the CLI accept
DEFAULT_PRESET: Final = "64k"

WEIGHT_DTYPES: Final[Mapping[str, mx.Dtype]] = {
    "float16": mx.float16,
    "bfloat16": mx.bfloat16,
    "float32": mx.float32,
}
BRAIN_IMPLS: Final = ("matmul", "einsum")
RUNTIME_CHECKS: Final = ("cheap", "off")


@functools.cache
def _section_types() -> dict[str, type]:
    hints = typing.get_type_hints(Config)
    return {f.name: hints[f.name] for f in dataclasses.fields(Config)}


@functools.cache
def _field_types(section_cls: type) -> dict[str, Any]:
    hints = typing.get_type_hints(section_cls)
    return {f.name: hints[f.name] for f in dataclasses.fields(section_cls)}


def _field_fg(cfg: Config) -> int:
    return round(cfg.world.size / cfg.world.field_cell)


def field_cells(cfg: Config) -> int:
    """The number of field cells (``Dims.n_cells``) from the world keys alone: the feature validators run
    before ``dims(cfg)`` is safe to call (on a sphere it derives the hash)."""
    w = cfg.world
    if w.geometry == "sphere":
        return w.layers * geom.npix(w.nside)
    fg = _field_fg(cfg)
    return fg * fg * (round(w.height / w.field_cell) if w.dims == 3 else 1)


def cell_grid_shape(d: Dims) -> tuple[int, ...]:
    """The shape of a per-cell field of the world: ``(FZ, FG, FG)`` in a 3D box, ``(FG, FG)`` in 2D,
    ``(FZ, Npix)`` on a shell (``Dims.field_shape``). The extension fields on the cell grid (the built
    material, the soil) declare their shape through this one function."""
    return d.field_shape


def aperture_neighbours(cfg: Config) -> tuple[int, int]:
    """``(n_nbr, n_min)`` of the structures' aperture rule R2: the neighbours of an interior cell (8, or 26 in
    3D and on a shell) and the fewest a cell of the boundary layer has (8, 17). ``ceil(aperture·n_nbr)`` must
    not exceed ``n_min`` in either geometry (a sphere's requirement is per cell, ``ceil(aperture·n_cell)``,
    so it is satisfiable wherever the box's is)."""
    return (26, 17) if cfg.world.dims == 3 else (8, 8)


def mat_colouring_error(cfg: Config) -> str | None:
    """Why the aperture rule's closing cycle does not exist for this world, or None. A box needs a field
    lattice side that is composite and at least 4 (``fields.mat_phases``); a shell always has the colouring of
    its pixel graph."""
    if cfg.world.geometry == "sphere":
        return None
    from alife import fields  # deferred: fields imports this module

    try:
        fields.mat_phases(_field_fg(cfg), cfg.world.dims)
    except ValueError as e:
        return str(e)
    return None


def _f32_at_least(x: float) -> float:
    """The smallest float32 ≥ ``x``, as a Python float (MLX casts Python bounds to float32)."""
    f = np.float32(x)  # nearest; compared in float64 (NumPy would compare a Python float in float32)
    return float(f) if float(f) >= x else float(np.nextafter(f, np.float32(np.inf)))


def _f32_at_most(x: float) -> float:
    """The largest float32 ≤ ``x``, as a Python float."""
    f = np.float32(x)
    return float(f) if float(f) <= x else float(np.nextafter(f, np.float32(-np.inf)))


@functools.cache
def dims(cfg: Config) -> Dims:
    """Derived sizes for a (validated) config. Cached per config."""
    n, m = cfg.population.n_max, cfg.population.max_births_per_step
    h = cfg.brain.hidden
    g_hash = math.floor(cfg.world.size / cfg.sensing.radius)
    nd = cfg.world.dims
    lay = layout.for_dims(nd, ext.layout_blocks(cfg, nd))
    ext_d = {spec.name: dict(spec.dims(cfg)) for spec in ext.active(cfg)}
    voice_w = max((dd.get("voice_w", 0) for dd in ext_d.values()), default=0)
    w1_rows, w2_rows = lay.D_IN + h, h + 1
    e_max, s = cfg.energy.e_max, cfg.energy.structure
    fg = _field_fg(cfg)
    r = cfg.physics.radius
    if nd == 3:
        fz = round(cfg.world.height / cfg.world.field_cell)
        g_hash_z = math.floor(cfg.world.height / cfg.sensing.radius)
        cell_hash_z = cfg.world.height / g_hash_z if g_hash_z > 0 else float("inf")
        z_lo, z_hi = _f32_at_least(r), _f32_at_most(cfg.world.height - r)
        z_ground = max(z_lo, _f32_at_most(r + cfg.physics.floor_eps))  # the floor rest z_lo always grounds
    else:
        fz, g_hash_z, cell_hash_z = 1, 1, float("inf")
        z_lo = z_hi = z_ground = 0.0
    sph = cfg.world.geometry == "sphere"
    sph_kw: dict[str, Any] = {}
    if sph:  # DESIGN §46: every sphere quantity from the exact host tables; the box lattice is zeroed
        wd = cfg.world
        nside = wd.nside
        nh = geom.derive_nh(wd.radius, r, cfg.sensing.radius)
        g_hash_z, cell_hash_z = geom.hash_layers(wd.height, cfg.sensing.radius)
        fz, fg, g_hash = wd.layers, 0, 0
        sph_kw = {
            "sph": True,
            "radius": wd.radius,
            "nside": nside,
            "npix": geom.npix(nside),
            "nside_h": nh,
            "npix_h": geom.npix(nh) if nh > 0 else 0,
            "r_lo": _f32_at_least(wd.radius + z_lo),
            "r_hi": _f32_at_most(wd.radius + z_hi),
        }
    return Dims(
        n=n,
        m=m,
        n_cap=n - m,
        dl=2 * m,
        k=cfg.sensing.k_neighbors,
        c=cfg.sensing.cell_capacity,
        q=(125 if sph else 3**nd) * cfg.sensing.cell_capacity,  # 9C in 2D, 27C in 3D, 125C on a sphere
        g_hash=g_hash,
        cell_hash=cfg.world.size / g_hash if g_hash > 0 else float("inf"),
        fg=fg,
        h=h,
        w1_rows=w1_rows,
        w2_rows=w2_rows,
        params=w1_rows * h + w2_rows * lay.D_OUT,
        pp=cfg.fields.n_patches,
        wdtype=WEIGHT_DTYPES.get(cfg.brain.weight_dtype, mx.float32),
        occ_limit=min((1 << 15) - 1, ((1 << 31) - 1 - R_HARD) // max(2 * e_max + s, 1)),
        nd=nd,
        d3=nd == 3,
        lay=lay,
        fz=fz,
        n_cells=fz * sph_kw["npix"] if sph else fz * fg * fg,
        field_shape=(fz, sph_kw["npix"]) if sph else (fz, fg, fg) if nd == 3 else (fg, fg),
        g_hash_z=g_hash_z,
        cell_hash_z=cell_hash_z,
        z_lo=z_lo,
        z_hi=z_hi,
        z_ground=z_ground,
        wn=cfg.weather.cells,
        voice_w=voice_w,
        msg_w=layout.MSG + voice_w,
        features=tuple(spec.name for spec in ext.active(cfg)),
        ext_dims=tuple((n, tuple(sorted(dd.items()))) for n, dd in ext_d.items()),
        **sph_kw,
    )


def stability_number(cfg: Config) -> float:
    """Gershgorin bound 2(B·k_s + K·k_r)/drag of the explicit overdamped step (§9)."""
    return (
        2.0
        * (layout.B * cfg.bonds.k_spring + cfg.sensing.k_neighbors * cfg.physics.k_repel)
        / cfg.physics.drag
    )


# --- dict conversion ------------------------------------------------------------------------------------


def config_to_dict(cfg: Config, *, omit_defaults: bool = False) -> dict[str, dict[str, Any]]:
    """Plain JSON-compatible dict (tuples become lists).

    With ``omit_defaults`` the keys of ``SERIAL_OMIT_KEYS`` (the feature series, DESIGN §43) are left out
    while
    they equal their default, and a section emptied by that is left out too: the form written to
    ``config.json``
    and to a checkpoint's metadata, so a run that does not use a feature serialises exactly as before the
    feature existed. ``config_from_dict`` completes the missing keys from the defaults.
    """
    out: dict[str, dict[str, Any]] = {}
    for sec in dataclasses.fields(Config):
        sub = getattr(cfg, sec.name)
        out[sec.name] = {
            f.name: list(v) if isinstance(v := getattr(sub, f.name), tuple) else v
            for f in dataclasses.fields(sub)
        }
    if omit_defaults:
        defaults = config_to_dict(Config())
        for sec, key in SERIAL_OMIT_KEYS:
            if out[sec][key] == defaults[sec][key]:
                del out[sec][key]
        for sec in {sec for sec, _ in SERIAL_OMIT_KEYS}:
            if not out[sec]:
                del out[sec]
    return out


def _coerce(value: Any, typ: Any, where: str) -> Any:
    """Coerce a TOML/JSON value to a field type; raise ConfigError on a mismatch."""
    if typ is bool:
        if isinstance(value, bool):
            return value
    elif typ is int:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    elif typ is float:
        if isinstance(value, int | float) and not isinstance(value, bool):
            return float(value)
    elif typ is str:
        if isinstance(value, str):
            return value
    elif typing.get_origin(typ) is tuple:
        if isinstance(value, list | tuple) and all(
            isinstance(v, int | float) and not isinstance(v, bool) for v in value
        ):
            return tuple(float(v) for v in value)
    else:  # pragma: no cover - every config field has one of the types above
        raise ConfigError(f"{where}: unsupported field type {typ!r}")
    raise ConfigError(f"{where}: expected {getattr(typ, '__name__', typ)}, got {value!r}")


def config_from_dict(d: Mapping[str, Any]) -> Config:
    """Build a Config from a (possibly sparse) nested mapping. Unknown sections/keys → ConfigError."""
    sections = _section_types()
    errors: list[str] = []
    kwargs: dict[str, Any] = {}
    for name, sub in d.items():
        if name not in sections:
            errors.append(f"unknown config section {name!r}")
            continue
        if not isinstance(sub, Mapping):
            errors.append(f"section {name!r} must be a table, got {type(sub).__name__}")
            continue
        types = _field_types(sections[name])
        vals: dict[str, Any] = {}
        for key, value in sub.items():
            if key not in types:
                errors.append(f"unknown config key {name}.{key}")
                continue
            try:
                vals[key] = _coerce(value, types[key], f"{name}.{key}")
            except ConfigError as e:
                errors.append(str(e))
        kwargs[name] = sections[name](**vals)
    if errors:
        raise ConfigError("; ".join(errors))
    return Config(**kwargs)


def _merge_layer(base: dict[str, dict[str, Any]], layer: Mapping[str, Any], source: str) -> None:
    """Deep-merge one sparse layer into ``base`` in place, rejecting unknown sections and keys."""
    errors: list[str] = []
    for name, sub in layer.items():
        if name not in base:
            errors.append(f"{source}: unknown config section {name!r}")
            continue
        if not isinstance(sub, Mapping):
            errors.append(f"{source}: section {name!r} must be a table")
            continue
        for key, value in sub.items():
            if key not in base[name]:
                errors.append(f"{source}: unknown config key {name}.{key}")
                continue
            base[name][key] = value
    if errors:
        raise ConfigError("; ".join(errors))


def _parse_override_value(raw: str, typ: Any, where: str) -> Any:
    raw = raw.strip()
    try:
        if typ is bool:
            low = raw.lower()
            if low not in ("true", "false"):
                raise ValueError(raw)
            return low == "true"
        if typ is int:
            return int(raw)
        if typ is float:
            return float(raw)
        if typ is str:
            return raw
        if typing.get_origin(typ) is tuple:
            return tuple(float(v) for v in raw.split(",") if v.strip() != "")
    except ValueError:
        pass
    raise ConfigError(f"--set {where}: cannot parse {raw!r} as {getattr(typ, '__name__', typ)}")


def _overrides_layer(overrides: Sequence[str]) -> dict[str, dict[str, Any]]:
    sections = _section_types()
    layer: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for item in overrides:
        lhs, sep, rhs = item.partition("=")
        name, dot, key = lhs.strip().partition(".")
        if not sep or not dot or not key:
            errors.append(f"--set {item!r}: expected section.key=value")
            continue
        if name not in sections:
            errors.append(f"--set {item!r}: unknown config section {name!r}")
            continue
        types = _field_types(sections[name])
        if key not in types:
            errors.append(f"--set {item!r}: unknown config key {name}.{key}")
            continue
        try:
            layer.setdefault(name, {})[key] = _parse_override_value(rhs, types[key], f"{name}.{key}")
        except ConfigError as e:
            errors.append(str(e))
    if errors:
        raise ConfigError("; ".join(errors))
    return layer


def apply_overrides(cfg: Config, overrides: Sequence[str]) -> Config:
    """Apply ``section.key=value`` strings, parsed with each field's type. The result is validated."""
    base = config_to_dict(cfg)
    _merge_layer(base, _overrides_layer(overrides), "--set")
    out = config_from_dict(base)
    validate(out)
    return out


# --- presets and files ----------------------------------------------------------------------------------


def _read_preset(name: str) -> dict[str, Any]:
    if name not in PRESETS and name not in ALL_OVERLAYS:
        raise ConfigError(
            f"unknown preset or overlay {name!r} (presets: {PRESETS}, overlays: {ALL_OVERLAYS})"
        )
    text = importlib.resources.files("alife.presets").joinpath(f"{name}.toml").read_text(encoding="utf-8")
    return tomllib.loads(text)


def _read_file(path: Path) -> dict[str, Any]:
    try:
        if path.suffix.lower() == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
        elif path.suffix.lower() == ".toml":
            data = tomllib.loads(path.read_text(encoding="utf-8"))
        else:
            raise ConfigError(f"{path}: config files must be .toml or .json")
    except (OSError, json.JSONDecodeError, tomllib.TOMLDecodeError) as e:
        raise ConfigError(f"{path}: {e}") from e
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a table")
    return data


def load_config(
    path: str | Path | None = None,
    *,
    preset: str | None = None,
    overlays: Sequence[str] = (),
    overrides: Sequence[str] = (),
) -> Config:
    """Resolve defaults < preset < overlays < file < overrides, then validate."""
    base = config_to_dict(Config())
    name = preset if preset is not None else DEFAULT_PRESET
    if name not in PRESETS:
        raise ConfigError(f"unknown preset {name!r} (choose from {PRESETS})")
    _merge_layer(base, _read_preset(name), f"preset {name}")
    for ov in overlays:
        if ov not in ALL_OVERLAYS:
            raise ConfigError(f"unknown overlay {ov!r} (choose from {ALL_OVERLAYS})")
        _merge_layer(base, _read_preset(ov), f"overlay {ov}")
    if path is not None:
        p = Path(path)
        _merge_layer(base, _read_file(p), str(p))
    _merge_layer(base, _overrides_layer(overrides), "--set")
    cfg = config_from_dict(base)
    validate(cfg)
    return cfg


def _canonical_json(d: Mapping[str, Any]) -> str:
    return json.dumps(d, sort_keys=True, indent=2, allow_nan=False) + "\n"


def save_config(cfg: Config, path: Path) -> None:
    """Write canonical JSON (sorted keys, indent 2, tuples as lists)."""
    Path(path).write_text(_canonical_json(config_to_dict(cfg, omit_defaults=True)), encoding="utf-8")


# Keys added after v0.1 (section, key). They are omitted from ``config_hash`` while they equal their defaults,
# so runs created before they existed keep their hash and stay resumable (§18).
_POST_V01_BASE: Final[tuple[tuple[str, str], ...]] = (
    ("population", "immigration_every"),
    ("population", "immigration_count"),
    ("fields", "season_period"),
    ("fields", "season_amplitude"),
    ("fields", "season_mode"),
    ("director", "enabled"),
    ("director", "persona"),
    ("director", "goal"),
    ("director", "rule_every"),
    ("director", "budget_per_100k"),
    ("director", "llm_every"),
    ("director", "llm_temperature"),
    ("director", "llm_timeout_s"),
    ("director", "llm_max_tokens"),
    ("director", "yoke_log"),
    ("world", "dims"),
    ("world", "height"),
    ("physics", "gravity"),
    ("physics", "floor_eps"),
    ("physics", "buoyancy_max"),
    ("mutation", "buoy_init"),
    ("mutation", "buoy_sigma"),
    ("fields", "benthic_lambda"),
    ("fields", "photic_amp"),
    ("fields", "photic_lambda"),
    ("energy", "buoyancy_cost"),
    ("energy", "buoyancy_cap_cost"),
    ("world", "geometry"),  # the sphere world (DESIGN §46): SPHERE_KEYS, defined below
    ("world", "radius"),
    ("world", "nside"),
    ("world", "layers"),
    ("physics", "ceiling_band"),
    ("game", "observe"),
    ("game", "house_rate"),
    ("game", "tracers"),
    ("game", "tracer_period"),
    ("weather", "enabled"),
    ("weather", "cells"),
    ("weather", "H1"),
    ("weather", "H2"),
    ("weather", "g_top"),
    ("weather", "g_prime"),
    ("weather", "f0"),
    ("weather", "gamma"),
    ("weather", "tau_c"),
    ("weather", "k_E"),
    ("weather", "u_gust"),
    ("weather", "q_bg"),
    ("weather", "q_pool"),
    ("weather", "tau_rad"),
    ("weather", "r1"),
    ("weather", "c_d"),
    ("weather", "r2"),
    ("weather", "pools"),
    ("weather", "seed_rate"),
    ("weather", "seed_v"),
    ("weather", "seed_sigma"),
    ("weather", "seed_jitter"),
    ("weather", "alpha0"),
    ("weather", "alpha1"),
    ("weather", "u_max"),
    ("weather", "w_edge"),
    ("weather", "z_sl"),
    ("weather", "rain_gain"),
    ("weather", "rain_floor"),
    ("weather", "storm_p"),
    ("weather", "storm_u0"),
    ("weather", "storm_span"),
    ("weather", "storm_break_u"),
    ("weather", "seed_gain"),
    ("weather", "seed_c0"),
    ("weather", "shade_gain"),
    ("weather", "shade_n0"),
    ("weather", "wind_scale"),
    ("weather", "rain_zonal"),
    ("weather", "storm_eddy"),
    ("weather", "wind_eddy_force"),
    ("combat", "protect_floor"),
)
# The keys the registered features add (``ext.ExtSpec.lever_keys``; none while the registry is empty).
POST_V01_KEYS: Final[tuple[tuple[str, str], ...]] = (
    *_POST_V01_BASE,
    *(key for spec in ext.REGISTRY for key in spec.lever_keys),
)
# The feature-series keys (DESIGN §43): omitted from ``config.json`` and the checkpoint metadata at their
# defaults, so an off run's files are byte-identical to the engine before the feature existed.
SERIAL_OMIT_KEYS: Final[tuple[tuple[str, str], ...]] = (
    ("combat", "protect_floor"),
    (
        "world",
        "geometry",
    ),  # the sphere world (DESIGN §46.9): a box run serialises as before the sphere existed
    ("world", "radius"),
    ("world", "nside"),
    ("world", "layers"),
    ("physics", "ceiling_band"),
    ("run", "mat_every"),  # the feature dumps (DESIGN §38.9, §42.6): unhashed run cadences
    ("run", "arms_every"),
    ("run", "niche_every"),
    *(key for spec in ext.REGISTRY for key in spec.lever_keys),
)
# feature -> every (section, key) it adds to Config, and the subset the GUI does not carry (``*.control``,
# observer dumps): ``tests/test_gui_sync.py`` compares the Swift lever files against the difference.
FEATURE_KEYS: Final[dict[str, tuple[tuple[str, str], ...]]] = dict(ext.lever_keys())
CLI_ONLY_FEATURE_KEYS: Final[dict[str, tuple[tuple[str, str], ...]]] = {
    spec.name: spec.cli_only_keys for spec in ext.REGISTRY
}

# The POST_V01 keys of the 3D world (docs/3D.md, §27.1). In a 2D world every one but ``world.dims`` must keep
# its default (``validate``), so no 2D config can change its hash without changing its behaviour.
THREE_D_KEYS: Final[tuple[tuple[str, str], ...]] = (
    ("world", "dims"),
    ("world", "height"),
    ("physics", "gravity"),
    ("physics", "floor_eps"),
    ("physics", "buoyancy_max"),
    ("mutation", "buoy_init"),
    ("mutation", "buoy_sigma"),
    ("fields", "benthic_lambda"),
    ("fields", "photic_amp"),
    ("fields", "photic_lambda"),
    ("energy", "buoyancy_cost"),
    ("energy", "buoyancy_cap_cost"),
)
# The POST_V01 keys of the sphere world (DESIGN §46.9). ``world.height`` and ``weather.cells``
# (reinterpreted on a
# sphere: the weather Nside) are existing keys and belong to THREE_D_KEYS and the weather. In a box world
# every
# one but ``world.geometry`` must keep its default (``validate``), so no box config changes hash without
# changing
# behaviour. All five are omitted from ``config.json`` and the checkpoint metadata at their defaults
# (``SERIAL_OMIT_KEYS``), so a box run serialises byte for byte as before the sphere existed.
SPHERE_KEYS: Final[tuple[tuple[str, str], ...]] = (
    ("world", "geometry"),
    ("world", "radius"),
    ("world", "nside"),
    ("world", "layers"),
    ("physics", "ceiling_band"),
)
# The world.geometry values validate accepts. Read at call time: S2 of the sphere series enabled "sphere"
# (a test can narrow it to check the refusal text).
GEOMETRY_SUPPORTED: tuple[str, ...] = ("box", "sphere")
GEOMETRIES: Final = ("box", "sphere")

# What a sphere run may enable beyond the dry core (DESIGN §46.12, rule 4 of the consumer guard). The
# consumers of these layers read the box geometry (lattice cells, torus distances, ``pos[:, 2]``), so a sphere
# run must not reach them before the PR that converts them: ``SPHERE_FEATURE_PR`` names each layer and the PR
# that will enable it, ``SPHERE_READY`` (read at call time, like ``DIMS_SUPPORTED``) the ones that are. Every
# PR of the series that converts a layer adds its name here.
SPHERE_FEATURE_PR: Final[Mapping[str, str]] = {
    "arms": "S3",
    "structures": "S3",
    "bonds": "S3",
    "comm": "S3",
    "niche": "S3",
    "roles": "S3",
    "weather": "S4",
    "director": "S5",
    "observers": "S5",
}
SPHERE_READY: frozenset[str] = frozenset({"structures", "niche"})


def sphere_wanted(cfg: Config) -> list[str]:
    """The not-dry layers a config enables, by the names of ``SPHERE_FEATURE_PR`` (registry order first)."""
    names = [spec.name for spec in ext.active(cfg)]
    if cfg.roles.enabled and "roles" not in names:
        names.append("roles")
    if cfg.weather.enabled:
        names.append("weather")
    if cfg.director.enabled or cfg.run.director != "off":
        names.append("director")
    if cfg.game.observe or cfg.game.tracers > 0:
        names.append("observers")
    return names


# The world.dims values validate accepts. Read at call time, so a test can narrow it (docs/3D.md §10: PR 3
# enabled 3).
DIMS_SUPPORTED: tuple[int, ...] = (2, 3)

# The POST_V01 keys that are omitted from ``config_hash`` at every value: the director's policy. Only
# ``director.enabled`` changes the compiled step, so only it is hashed.
UNHASHED_KEYS: Final[tuple[tuple[str, str], ...]] = (
    ("director", "persona"),
    ("director", "goal"),
    ("director", "rule_every"),
    ("director", "budget_per_100k"),
    ("director", "llm_every"),
    ("director", "llm_temperature"),
    ("director", "llm_timeout_s"),
    ("director", "llm_max_tokens"),
    ("director", "yoke_log"),
)
DIRECTOR_POLICY_KEYS: Final = tuple(key for _, key in UNHASHED_KEYS)
# POST_V01 keys read only by host-side observers (§28, §29): omitted from ``config_hash`` at every value and
# inert while ``game.observe`` is off, so they may default to a working value. Separate from
# ``UNHASHED_KEYS``, from which ``DIRECTOR_POLICY_KEYS`` (the director's resume keys) is derived.
_GAME_OBSERVER_KEYS: Final[tuple[tuple[str, str], ...]] = (
    ("game", "observe"),
    ("game", "house_rate"),
    ("game", "tracers"),
    ("game", "tracer_period"),
)
# ... and the observer keys of the registered features (``ExtSpec.observer_keys``, DESIGN §43.5)
OBSERVER_KEYS: Final[tuple[tuple[str, str], ...]] = (
    *_GAME_OBSERVER_KEYS,
    *(key for spec in ext.REGISTRY for key in spec.observer_keys),
)
TRACERS_MAX: Final = 1 << 20  # the largest ``game.tracers``
DIRECTOR_MODES: Final = ("off", "live", "replay")  # the research modes (``--director`` of run and resume)
PLAYER_MODE: Final = "player"  # ``alife play``: applies the player's powers, decides nothing (DESIGN §26.8)
RUN_DIRECTOR_MODES: Final = (*DIRECTOR_MODES, PLAYER_MODE)  # every value ``run.director`` takes
RANDOM_PERSONA: Final = "random"  # arm R of DIRECTOR.md §9: yoked to another run's discretionary moves
PERSONAS: Final = ("gardener", "trickster", "architect", RANDOM_PERSONA)


SEASON_MODES: Final = ("global", "per_patch")


def immigration_on(pop: PopulationConfig) -> bool:
    """Static: any immigration source (floor rule or periodic) is enabled."""
    return pop.immigration_floor > 0 or pop.immigration_every > 0


def immigration_possible(cfg: Config) -> bool:
    """Static: immigrants can appear, from a config rule or from the director's ``env.imm_request``."""
    return immigration_on(cfg.population) or cfg.director.enabled


def seasons_on(fi: FieldConfig) -> bool:
    """Static: seasonal modulation of the fertility patches is enabled."""
    return fi.season_period > 0 and fi.season_amplitude > 0.0


def is_3d(cfg: Config) -> bool:
    """Static: the world is 3D (``world.dims = 3``, §27.1)."""
    return cfg.world.dims == 3


def require_2d(cfg: Config, what: str) -> None:
    """Refuse a 3D config in a 2D-only tool: ``ConfigError`` (exit 2) naming ``what``."""
    if is_3d(cfg):
        raise ConfigError(
            f"{what} supports 2D worlds only (world.dims = {cfg.world.dims}); a 3D run is viewed in the "
            "AlifeControl 3D viewer through `alife stream` (docs/3D.md §8)"
        )


def require_box(cfg: Config, what: str, pr: str = "S5") -> None:
    """Refuse a sphere config in a tool that reads the box geometry: ``ConfigError`` (exit 2) naming ``what``
    and the PR of the sphere series that converts it (DESIGN §46.12: nothing runs through an unconverted
    consumer)."""
    if cfg.world.geometry == "sphere":
        raise ConfigError(
            f"{what} does not support world.geometry = 'sphere' yet (the true-sphere series, PR {pr}); "
            "`alife run`, `resume` and `bench` run a sphere world"
        )


def config_hash(cfg: Config) -> str:
    """SHA-256 hex of the canonical JSON of every section except ``run``.

    Keys added after v0.1 (``POST_V01_KEYS``) are omitted at their defaults (``UNHASHED_KEYS`` and
    ``OBSERVER_KEYS``: always), and a section left empty by that (``director``, ``game``) is omitted too, so
    pre-director and pre-game runs keep their hash.
    """
    d = config_to_dict(cfg)
    d.pop("run")
    defaults = config_to_dict(Config())
    for sec, key in POST_V01_KEYS:
        if (sec, key) in UNHASHED_KEYS or (sec, key) in OBSERVER_KEYS or d[sec][key] == defaults[sec][key]:
            del d[sec][key]
    for sec in {sec for sec, _ in POST_V01_KEYS}:
        if not d[sec]:
            del d[sec]
    return hashlib.sha256(_canonical_json(d).encode("utf-8")).hexdigest()


# --- validation -----------------------------------------------------------------------------------------


def is_http_url(url: str) -> bool:
    """True for an ``http://`` or ``https://`` URL with a host (the only schemes the LLM client contacts)."""
    try:
        parts = urlsplit(url)
        return parts.scheme in ("http", "https") and bool(parts.hostname)
    except ValueError:
        return False


def frac_q8(frac: float) -> int:
    """The engine's quantised offspring fraction ``floor(f32(frac)·256 + 0.5)`` (§10.7), in MLX f32."""
    return int(mx.floor(mx.array(frac, mx.float32) * 256.0 + 0.5).astype(mx.int32).item())


# The largest hash Nside validate will derive. The shipped table of exact coverage angles
# (geom.min_coverage_angle)
# covers Nside 2..72, so a derivation up to 64 (it probes Nh + 1) costs nothing; beyond the table one
# exact table of
# 12·Nside² cells is computed per probe (about 40 s at Nside 64) and cached on disk.
NH_MAX: Final = 64
SPHERE_CELL_VOLUME_BAND: Final = (0.4, 2.5)  # an (angular cell, layer) volume in units of field_cell³


SPHERE_RADIUS_MAX: Final = 1e6  # keeps radius³ and every derived float finite; far beyond any feasible planet


def _validate_sphere(cfg: Config) -> list[str]:
    """The rules of a sphere world (DESIGN §46.9), as messages. The layers a sphere run may enable are
    ``SPHERE_READY`` (S2: none; the weather is S4's); the fields' rules (S1) are ``_sphere_field_rules``.

    Ordered by cost, and total: the refusal comes first and alone (nothing else is computed for a config the
    engine cannot run); the cheap arithmetic rules come next and every quantity they derive is bounded
    before it
    is used (no allocation or overflow from a typo such as ``layers = 10**11`` or ``radius = 1e200``);
    the exact
    hash derivation (``geom.derive_nh``, seconds without the shipped table) only runs for a config that passed
    all of them."""
    w, ph, sen = cfg.world, cfg.physics, cfg.sensing
    if "sphere" not in GEOMETRY_SUPPORTED:
        return ["world.geometry = 'sphere' is not enabled yet (the true-sphere series, PR S2)"]
    errs: list[str] = []

    def req(ok: bool, msg: str) -> bool:
        if not ok:
            errs.append(msg)
        return ok

    req(w.dims == 3, f"world.geometry = 'sphere' needs world.dims = 3 (the shell is 3D), got {w.dims}")
    for name in sphere_wanted(
        cfg
    ):  # the readiness gates (DESIGN §46.12): nothing runs through an unconverted consumer
        if name not in SPHERE_READY:
            pr = SPHERE_FEATURE_PR.get(name, "S3")
            req(
                False,
                f"{name} is not available on a sphere yet (PR {pr}): a sphere runs the dry world",
            )
    rad, h, r = w.radius, w.height, ph.radius
    rad_ok = req(
        math.isfinite(rad) and 20.0 <= rad <= SPHERE_RADIUS_MAX,
        f"world.radius must be finite and in [20, {SPHERE_RADIUS_MAX:g}], got {rad}",
    )
    h_ok = req(
        math.isfinite(h) and h > 2 * r,
        f"world.height must be finite and > 2·physics.radius = {2 * r}, got {h}",
    )
    sane = rad_ok and h_ok and math.isfinite(w.size)
    req(
        not sane or h <= rad,
        f"world.height {h} must be ≤ world.radius {rad} (a thin shell bounds the area ratio by 4)",
    )
    req(
        not sane or w.size > 4.0 * (rad + h),
        f"world.size {w.size} must be > 4·(radius + height) = {4.0 * (rad + h)}: it is the image length of "
        "min_image, which must be the identity on pair offsets",
    )
    nside_ok = req(w.nside >= 2, f"world.nside must be ≥ 2, got {w.nside}")
    layers_ok = req(w.layers >= 2, f"world.layers must be ≥ 2, got {w.layers}")
    req(
        math.isfinite(ph.ceiling_band)
        and ph.ceiling_band >= 0.0
        and (not sane or ph.ceiling_band < h - 2 * r),
        f"physics.ceiling_band must be in [0, height − 2·radius), got {ph.ceiling_band}",
    )
    req(
        ph.floor_eps >= 0.01,
        f"physics.floor_eps must be ≥ 0.01 on a sphere (float32 radius), got {ph.floor_eps}",
    )
    if sane and nside_ok and layers_ok and w.field_cell > 0:
        n_cells = w.layers * geom.npix(w.nside)
        if req(n_cells <= 1 << 20, f"field cells FZ·12·Nf² = {n_cells} must be ≤ 2^20 (exact_sum)"):
            z1 = float(geom.layer_edges(rad, h, w.layers)[1])  # layers·npix <= 2^20: a small allocation
            req(
                r + ph.floor_eps < z1,
                f"physics.radius + floor_eps = {r + ph.floor_eps} must be < z_1 = {z1:.4g} "
                "(a grounded agent is in the floor layer)",
            )
            lo, hi = SPHERE_CELL_VOLUME_BAND
            vol = geom.layer_volume(rad, h, w.layers, w.nside)
            req(
                lo * w.field_cell**3 <= vol <= hi * w.field_cell**3,
                f"the cell volume {vol:.4g} must be in [{lo}, {hi}]·world.field_cell³ = "
                f"[{lo * w.field_cell**3:.4g}, {hi * w.field_cell**3:.4g}]",
            )
    # the fields (S1): the conductance tables need the shell that passed the rules above; before the hash
    # block so that a bad diffusion constant is reported without the expensive derivation
    if sane and nside_ok and layers_ok and w.field_cell > 0 and not errs:
        errs.extend(_sphere_field_rules(cfg))
    # the hash: Nh from the exact reach-2 coverage, then C from the lid cell. Only for a config that passed
    # everything above (the derivation is the expensive part), and only for inputs it is defined for.
    hash_ok = sane and not errs and sen.radius > 0 and math.isfinite(sen.radius) and r > 0
    if hash_ok:
        if not req(
            sen.radius < 2.0 * (rad + r),
            f"sensing.radius {sen.radius} must be < 2·(world.radius + physics.radius) = {2.0 * (rad + r)}: "
            "a hash cannot cover a planet smaller than the sensing ball",
        ):
            return errs
        angle = geom.rigorous_angle(rad, r, sen.radius)
        if geom.estimate_nh(angle) > NH_MAX:
            errs.append(
                f"the hash Nside for sensing.radius {sen.radius} on world.radius {rad} would exceed "
                f"{NH_MAX}: raise sensing.radius or lower world.radius"
            )
        else:
            nh = geom.derive_nh(rad, r, sen.radius)
            req(nh >= 2, f"no hash Nside >= 2 covers sensing.radius {sen.radius} on world.radius {rad}")
            gz, _ = geom.hash_layers(h, sen.radius)
            req(gz >= 1, f"hash layers G_z = floor(2·height / radius) = {gz} must be ≥ 1")
            if nh >= 2 and gz >= 1:
                c_min = geom.min_cell_capacity(rad, h, sen.radius, nh)
                req(
                    sen.cell_capacity >= c_min,
                    f"sensing.cell_capacity {sen.cell_capacity} must be ≥ ceil(0.167·V_lid) + 4 = {c_min} "
                    f"(the lid hash cell holds {geom.lid_cell_volume(rad, h, sen.radius, nh):.4g} units³)",
                )
    return errs


SPHERE_CHEM_SUBSTEPS_MAX: Final = 16  # the unrolled sub-steps of the chemical update (3 to 6 for the presets)


def _sphere_field_rules(cfg: Config) -> list[str]:
    """The rules of the sphere's fields (DESIGN §46.5, sphere series S1), from the conductance tables. Called
    by ``_validate_sphere`` before the hash block, only for a shell that passed its geometry rules (the tables
    need a sane shell).

    The integer resource diffusion needs ``Σ_j α_ij <= 0.9`` in every cell (``α_ij = d_q8·w_ij / 2^16``), i.e.
    ``d_q8 <= floor(0.9·2^16 / rowsum_max)`` (11, 11, 14 for the three presets), and then the cap bound
    ``R <= R_HARD`` is verified exhaustively on the actual tables (``ops.check_diffusion_tables``). The patch
    drift is a rotation by a series whose truncation is below 3e-11 for ``|ω| <= 0.05`` and 2e-9 for
    ``|ω| <= 0.1`` (``fields_sph.rotate``), under the float32 rounding; ``patch_drift <= 0.005·radius`` keeps
    ``|ω| <= 0.1`` at the largest device-valid ``drift_mult`` (20). The chemical update unrolls
    ``ceil(D·rowsum_max/256)`` sub-steps, at most ``SPHERE_CHEM_SUBSTEPS_MAX``."""
    w, fi = cfg.world, cfg.fields
    errs: list[str] = []
    cond = fieldtab.conductance(*fieldtab.shell(cfg))
    bound = cond.d_q8_max()
    if fi.resource_diffusion_q8 > bound:
        errs.append(
            f"fields.resource_diffusion_q8 {fi.resource_diffusion_q8} must be <= {bound} on this shell "
            f"(the largest row of pair weights sums to {cond.rowsum_max}/256; the content a cell sends away "
            "per step must stay below 0.9 of it)"
        )
    else:
        try:
            ops.check_diffusion_tables(
                cond.wt * fi.resource_diffusion_q8, cond.wz * fi.resource_diffusion_q8, R_HARD
            )
        except ValueError as e:
            errs.append(f"fields.resource_diffusion_q8 {fi.resource_diffusion_q8}: {e}")
    d_max = max((v for v in fi.chem_diffusion if math.isfinite(v)), default=0.0)
    n_sub = cond.chem_substeps(d_max)
    if n_sub > SPHERE_CHEM_SUBSTEPS_MAX:
        errs.append(
            f"the chemical update of this shell needs {n_sub} sub-steps (fields.chem_diffusion {d_max} times "
            f"the largest row sum {cond.rowsum_max}/256); the limit is {SPHERE_CHEM_SUBSTEPS_MAX}: lower "
            "fields.chem_diffusion or use a thicker layer spacing"
        )
    if fi.patch_drift * 20.0 > 0.1 * w.radius:
        errs.append(
            f"fields.patch_drift {fi.patch_drift} must be <= 0.005·world.radius = {0.005 * w.radius} "
            "(a patch rotates by |ω| = drift/radius per step; the rotation series must stay accurate, "
            "|ω| <= 0.1, at the largest device-valid drift_mult 20)"
        )
    return errs


def validate(cfg: Config) -> None:
    """Check every rule of §17 (plus basic sanity); raise one ConfigError listing all problems."""
    w, pop, sen, br = cfg.world, cfg.population, cfg.sensing, cfg.brain
    ph, bo, en, co = cfg.physics, cfg.bonds, cfg.energy, cfg.combat
    rp, mu, fi, ru = cfg.repro, cfg.mutation, cfg.fields, cfg.run
    di = cfg.director
    errs: list[str] = []

    def req(ok: bool, msg: str) -> None:
        if not ok:
            errs.append(msg)

    # Sanity: positive sizes and counts, so the derived rules below are well defined.
    req(w.size > 0 and math.isfinite(w.size), f"world.size must be positive and finite, got {w.size}")
    req(w.field_cell > 0, f"world.field_cell must be > 0, got {w.field_cell}")
    req(sen.radius > 0, f"sensing.radius must be > 0, got {sen.radius}")
    req(sen.cell_capacity >= 1, f"sensing.cell_capacity must be ≥ 1, got {sen.cell_capacity}")
    req(sen.tag_scale > 0, f"sensing.tag_scale must be > 0, got {sen.tag_scale}")
    req(sen.clock_period >= 1, f"sensing.clock_period must be ≥ 1, got {sen.clock_period}")
    req(sen.chem_half_sat > 0, f"sensing.chem_half_sat must be > 0, got {sen.chem_half_sat}")
    req(br.hidden >= 1, f"brain.hidden must be ≥ 1, got {br.hidden}")
    req(br.weight_clip > 0, f"brain.weight_clip must be > 0, got {br.weight_clip}")
    req(ph.radius > 0, f"physics.radius must be > 0, got {ph.radius}")
    req(ph.drag > 0, f"physics.drag must be > 0, got {ph.drag}")
    req(ph.turn_max > 0, f"physics.turn_max must be > 0, got {ph.turn_max}")
    req(ph.v_max > 0, f"physics.v_max must be > 0, got {ph.v_max}")
    req(pop.max_age >= 0, f"population.max_age must be ≥ 0, got {pop.max_age}")
    req(pop.immigration_floor >= 0, f"population.immigration_floor must be ≥ 0, got {pop.immigration_floor}")
    req(
        pop.immigrants_per_step >= 0,
        f"population.immigrants_per_step must be ≥ 0, got {pop.immigrants_per_step}",
    )
    req(fi.n_patches >= 1, f"fields.n_patches must be ≥ 1, got {fi.n_patches}")
    req(fi.patch_sigma > 0, f"fields.patch_sigma must be > 0, got {fi.patch_sigma}")
    req(fi.chem_max > 0, f"fields.chem_max must be > 0, got {fi.chem_max}")
    req(ru.metrics_every >= 1, f"run.metrics_every must be ≥ 1, got {ru.metrics_every}")
    req(ru.keep_checkpoints >= 1, f"run.keep_checkpoints must be ≥ 1, got {ru.keep_checkpoints}")
    req(ru.components_iters >= 1, f"run.components_iters must be ≥ 1, got {ru.components_iters}")
    req(ru.diversity_sample >= 2, f"run.diversity_sample must be ≥ 2, got {ru.diversity_sample}")
    req(ru.steps >= 0, f"run.steps must be ≥ 0, got {ru.steps}")
    req(0 <= ru.seed < 2**63, f"run.seed must be in [0, 2^63), got {ru.seed}")
    for name in (
        "e_init",
        "basal_cost",
        "bond_upkeep",
        "move_cost",
        "turn_cost",
        "grip_cost",
        "contract_cost",
        "msg_cost",
        "emit_cost",
        "harvest_cost",
        "defend_cost",
        "attack_cost",
        "harvest_max",
        "buoyancy_cost",
        "buoyancy_cap_cost",
    ):
        req(getattr(en, name) >= 0, f"energy.{name} must be ≥ 0, got {getattr(en, name)}")
    req(en.brain_cost_per_l1 >= 0, f"energy.brain_cost_per_l1 must be ≥ 0, got {en.brain_cost_per_l1}")
    req(rp.birth_cost >= 0 and rp.min_age >= 0, "repro.birth_cost and repro.min_age must be ≥ 0")

    # World geometry. A sphere skips the box lattice rules (G_hash, FG, FZ = height/field_cell, 2·height
    # <= size):
    # its cells come from the HEALPix tables, and ``world.size`` is only an image length (§46.9).
    sph = w.geometry == "sphere"
    if not sph and w.size > 0 and w.field_cell > 0 and sen.radius > 0 and math.isfinite(w.size):
        ratio = w.size / w.field_cell
        req(ratio == round(ratio), f"world.size / world.field_cell must be an integer, got {ratio}")
        req(3 <= round(ratio) <= 1024, f"field grid FG = {round(ratio)} must be in [3, 1024]")
        g_hash = math.floor(w.size / sen.radius)
        req(g_hash >= 3, f"hash grid G_h = floor(size / radius) = {g_hash} must be ≥ 3")

    # World dimensions (§27.1). A 2D config trips none of these: its 3D keys sit at their defaults.
    d3 = w.dims == 3
    req(w.dims in (2, 3), f"world.dims must be 2 or 3, got {w.dims}")
    req(
        w.dims not in (2, 3) or w.dims in DIMS_SUPPORTED,
        f"world.dims = {w.dims} is not enabled yet (docs/3D.md PR 3)",
    )
    if w.dims == 2:
        for sec, key in THREE_D_KEYS:
            if (sec, key) == ("world", "dims"):
                continue
            val, default = getattr(getattr(cfg, sec), key), getattr(getattr(Config(), sec), key)
            req(
                val == default,
                f"{sec}.{key} is a 3D key: it must keep its default {default!r} when world.dims = 2, "
                f"got {val!r}",
            )
    # The sphere world (DESIGN §46.9): its keys are inert in a box, and the sphere rules apply to a sphere.
    req(w.geometry in GEOMETRIES, f"world.geometry must be one of {list(GEOMETRIES)}, got {w.geometry!r}")
    if not sph:
        wdef, pdef = WorldConfig(), PhysicsConfig()
        for sec, key in SPHERE_KEYS:
            val, default = getattr(getattr(cfg, sec), key), getattr(wdef if sec == "world" else pdef, key)
            if (sec, key) != ("world", "geometry"):
                req(
                    val == default,
                    f"{sec}.{key} is inert while world.geometry is box: it must keep its default",
                )
    else:
        errs.extend(_validate_sphere(cfg))
    if d3:
        h = w.height
        req(
            math.isfinite(h) and h > 2 * ph.radius,
            f"world.height must be finite and > 2·physics.radius = {2 * ph.radius}, got {h}",
        )
        # Only finite ratios reach round() and floor(); the sanity rules above report the inputs that are not.
        sane = h > 0 and w.size > 0 and w.field_cell > 0 and sen.radius > 0
        fz, fg, hz = (h / w.field_cell, w.size / w.field_cell, h / sen.radius) if sane else (0.0, 0.0, 0.0)
        if not sph and sane and math.isfinite(fz) and math.isfinite(fg) and math.isfinite(hz):
            req(fz == round(fz), f"world.height / world.field_cell must be an integer, got {fz}")
            req(round(fz) >= 2, f"field layers FZ = {round(fz)} must be ≥ 2")
            cells = round(fz) * round(fg) ** 2
            req(cells <= 1 << 20, f"field cells FZ·FG² = {cells} must be ≤ 2^20 (exact_sum)")
            g_z = math.floor(hz)
            req(g_z >= 1, f"hash layers G_z = floor(height / radius) = {g_z} must be ≥ 1")
            # Then |dz| < L/2 for any two agents, so min_image is the identity on z (§27.1).
            req(2 * h <= w.size, f"world.height {h} must be ≤ world.size / 2 = {w.size / 2}")
        if math.isfinite(h):  # room for a child's vertical reflection at full offset (§27.4)
            req(
                h - 2 * ph.radius >= 2 * rp.offspring_distance,
                f"world.height − 2·physics.radius = {h - 2 * ph.radius} must be ≥ "
                f"2·repro.offspring_distance = {2 * rp.offspring_distance}",
            )

    # Interaction ranges.
    need = max(bo.form_range, co.range, 2 * ph.radius)
    req(
        sen.radius >= need,
        f"sensing.radius {sen.radius} must be ≥ max(form_range, combat.range, 2·r) = {need}",
    )
    if sph:  # the reach-2 stencil: 25 angular cells × 5 radial layers = 125·C candidates (§46.3)
        req(
            layout.NS <= sen.k_neighbors <= 125 * sen.cell_capacity,
            f"sensing.k_neighbors {sen.k_neighbors} must be in "
            f"[NS={layout.NS}, 125·C={125 * sen.cell_capacity}]",
        )
    elif d3:  # the 3×3×3 cell stencil scans 27·C candidates
        req(
            layout.NS <= sen.k_neighbors <= 27 * sen.cell_capacity,
            f"sensing.k_neighbors {sen.k_neighbors} must be in "
            f"[NS={layout.NS}, 27·C={27 * sen.cell_capacity}]",
        )
    else:
        req(
            layout.NS <= sen.k_neighbors <= 9 * sen.cell_capacity,
            f"sensing.k_neighbors {sen.k_neighbors} must be in [NS={layout.NS}, 9·C={9 * sen.cell_capacity}]",
        )

    # Population.
    n, m = pop.n_max, pop.max_births_per_step
    req(64 <= n <= 1 << 20, f"population.n_max {n} must be in [64, 2^20]")
    req(1 <= m <= n // 4, f"population.max_births_per_step {m} must be in [1, n_max // 4 = {n // 4}]")
    req(1 <= pop.n_init <= n - m, f"population.n_init {pop.n_init} must be in [1, N_CAP = {n - m}]")
    req(pop.immigration_floor <= n - m, f"population.immigration_floor must be ≤ N_CAP = {n - m}")
    req(
        pop.immigration_floor == 0 or pop.immigrants_per_step >= 1,
        "population.immigrants_per_step must be ≥ 1 when immigration_floor > 0 (immigration on)",
    )
    req(pop.immigration_every >= 0, f"population.immigration_every must be ≥ 0, got {pop.immigration_every}")
    req(
        pop.immigration_every == 0 or pop.immigration_count >= 1,
        "population.immigration_count must be ≥ 1 when immigration_every > 0 (periodic immigration on)",
    )
    req(
        0 <= pop.immigration_count <= m,
        f"population.immigration_count must be in [0, max_births_per_step = {m}]",
    )

    # Physics stability.
    if ph.drag > 0:
        sigma = stability_number(cfg)
        req(sigma <= 1.8, f"stability_number {sigma:.4g} must be ≤ 1.8")
    req(ph.thrust_max < 2 * ph.radius * ph.k_repel, "physics.thrust_max must be < 2·radius·k_repel")
    req(ph.v_max <= ph.radius, "physics.v_max must be ≤ physics.radius")
    req(ph.thrust_contact_limit > 0, "physics.thrust_contact_limit must be > 0")
    if d3:  # gravity, the floor and buoyancy (§27.3)
        g, bmax = ph.gravity, ph.buoyancy_max
        climb = ph.thrust_max * math.sin(layout.PITCH_MAX)
        req(0 <= g < climb, f"physics.gravity {g} must be in [0, thrust_max·sin(PITCH_MAX) = {climb:.4g})")
        if (
            not sph
        ):  # a sphere's floor-layer rule (radius + floor_eps < z_1, floor_eps >= 0.01) is in _validate_sphere
            req(
                ph.floor_eps >= 0 and ph.radius + ph.floor_eps < w.field_cell,
                f"physics.floor_eps {ph.floor_eps} must be ≥ 0 with radius + floor_eps < world.field_cell "
                "(a grounded agent is in the floor layer)",
            )
        req(0 <= bmax < math.inf, f"physics.buoyancy_max must be ≥ 0 and finite, got {bmax}")
        # The 3D form of thrust_max < 2·radius·k_repel: full thrust plus the net vertical load must not push a
        # resting agent through the one below it (or above it, when buoyant).
        load = ph.thrust_max + max(g, bmax - g)
        req(
            load < 2 * ph.radius * ph.k_repel,
            f"physics.thrust_max + max(gravity, buoyancy_max − gravity) = {load:.4g} must be "
            f"< 2·radius·k_repel = {2 * ph.radius * ph.k_repel:.4g}",
        )

    # Bond geometry.
    req(
        bo.rest_length * (1 + bo.contract_gain) < bo.break_length,
        "rest_length·(1 + contract_gain) must be < break_length",
    )
    req(bo.form_range < bo.break_length, "bonds.form_range must be < bonds.break_length")
    req(rp.offspring_distance < bo.break_length, "repro.offspring_distance must be < bonds.break_length")

    # Energy bounds.
    req(en.e_max < 1 << 20, f"energy.e_max {en.e_max} must be < 2^20")
    req(1 <= en.e_init <= en.e_max, "energy.e_init must be in [1, energy.e_max]")
    req(0 <= en.structure < 1 << 20, f"energy.structure {en.structure} must be in [0, 2^20)")
    req(rp.min_energy > rp.birth_cost + en.structure, "repro.min_energy must be > birth_cost + structure")
    # §10.7 splits Dp with the engine's quantised fraction, so the endow ≥ 1 / parent keeps ≥ 1 guarantees are
    # checked on frac_q8 itself: q_min ≥ 1, (Dp_min·q_min) >> 8 ≥ 1 and q_max ≤ 255 (then endow < Dp).
    if 0 < mu.frac_min <= mu.frac_max < 1:
        q_min, q_max = frac_q8(mu.frac_min), frac_q8(mu.frac_max)
        dp_min = rp.min_energy - rp.birth_cost - en.structure
        req(
            q_min >= 1 and (dp_min * q_min) >> 8 >= 1,
            f"frac_min: need frac_q8(frac_min) = {q_min} ≥ 1 and "
            f"((min_energy − birth_cost − structure)·frac_q8(frac_min)) >> 8 ≥ 1",
        )
        req(q_max <= 255, f"frac_max: need frac_q8(frac_max) = {q_max} ≤ 255")
    if d3:  # the buoyancy costs (§27.7): each term is at most its coefficient; inert without the actuator
        for name in ("buoyancy_cost", "buoyancy_cap_cost"):
            req(getattr(en, name) < 1 << 16, f"energy.{name} {getattr(en, name)} must be < 2^16")
        req(
            ph.buoyancy_max > 0 or (en.buoyancy_cost == 0 and en.buoyancy_cap_cost == 0),
            "energy.buoyancy_cost and energy.buoyancy_cap_cost must be 0 when physics.buoyancy_max = 0",
        )

    # Contention headroom.
    for name, val in (
        ("energy.harvest_max", en.harvest_max),
        ("combat.attack_max", co.attack_max),
        ("bonds.share_rate", bo.share_rate),
    ):
        req(0 <= val < 1 << 16, f"{name} {val} must be in [0, 2^16)")

    # Resource field.
    req(0 < fi.resource_cap < R_HARD, f"fields.resource_cap {fi.resource_cap} must be in (0, 2^24 − 1)")
    req(0 <= fi.regen_max <= fi.resource_cap, "fields.regen_max must be in [0, resource_cap]")
    req(0 <= fi.resource_diffusion_q8 <= 64, "fields.resource_diffusion_q8 must be in [0, 64]")
    if d3 and not sph:  # six neighbours: 6·((R·d) >> 8) ≤ R keeps every cell ≥ 0 (a sphere: S1's table rules)
        req(fi.resource_diffusion_q8 <= 42, "fields.resource_diffusion_q8 must be ≤ 42 in 3D")
    req(fi.season_period >= 0, f"fields.season_period must be ≥ 0, got {fi.season_period}")
    req(
        0.0 <= fi.season_amplitude <= 1.0,
        f"fields.season_amplitude must be in [0, 1], got {fi.season_amplitude}",
    )
    req(
        fi.season_amplitude == 0.0 or fi.season_period >= 1,
        "fields.season_period must be ≥ 1 when fields.season_amplitude > 0 (seasons on)",
    )
    req(
        fi.season_mode in SEASON_MODES,
        f"fields.season_mode must be one of {list(SEASON_MODES)}, got {fi.season_mode!r}",
    )

    # Chemicals.
    req(len(fi.chem_diffusion) == layout.NCH, f"fields.chem_diffusion must have {layout.NCH} entries")
    req(len(fi.chem_decay) == layout.NCH, f"fields.chem_decay must have {layout.NCH} entries")
    req(all(0 <= v <= 0.25 for v in fi.chem_diffusion), "every fields.chem_diffusion must be in [0, 0.25]")
    req(all(0 <= v < 1 for v in fi.chem_decay), "every fields.chem_decay must be in [0, 1)")
    if d3 and not sph:  # the explicit 7-point stencil is stable for diffusion ≤ 1/6 (a sphere sub-steps: S1)
        req(all(v <= 1 / 6 for v in fi.chem_diffusion), "every fields.chem_diffusion must be ≤ 1/6 in 3D")

    # Vertical food profile (3D, §27.5).
    if d3:
        req(fi.benthic_lambda > 0, f"fields.benthic_lambda must be > 0 in 3D, got {fi.benthic_lambda}")
        req(0 <= fi.photic_amp <= 1, f"fields.photic_amp must be in [0, 1], got {fi.photic_amp}")
        req(
            fi.photic_amp == 0 or fi.photic_lambda > 0,
            f"fields.photic_lambda must be > 0 when fields.photic_amp > 0, got {fi.photic_lambda}",
        )

    # Rates and gene bounds.
    for name, val in (
        ("bonds.share_efficiency", bo.share_efficiency),
        ("combat.efficiency", co.efficiency),
        ("combat.defend_efficiency", co.defend_efficiency),
    ):
        req(0 < val <= 1, f"{name} {val} must be in (0, 1]")
    req(0 < mu.frac_min <= mu.frac_init <= mu.frac_max < 1, "need 0 < frac_min ≤ frac_init ≤ frac_max < 1")
    req(0 < mu.sigma_min <= mu.sigma_init <= mu.sigma_max, "need 0 < sigma_min ≤ sigma_init ≤ sigma_max")
    req(0 < mu.rate_min <= mu.rate_init <= mu.rate_max < 1, "need 0 < rate_min ≤ rate_init ≤ rate_max < 1")
    if d3:  # the buoyancy cap (§27.4): inert unless the actuator exists
        req(0 <= mu.buoy_init <= 1, f"mutation.buoy_init must be in [0, 1], got {mu.buoy_init}")
        req(0 <= mu.buoy_sigma < math.inf, f"mutation.buoy_sigma must be ≥ 0 and finite, got {mu.buoy_sigma}")
        req(
            ph.buoyancy_max > 0 or (mu.buoy_init == 0 and mu.buoy_sigma == 0),
            "mutation.buoy_init and mutation.buoy_sigma must be 0 when physics.buoyancy_max = 0",
        )

    # Enumerations.
    req(
        br.weight_dtype in WEIGHT_DTYPES,
        f"brain.weight_dtype {br.weight_dtype!r} not in {tuple(WEIGHT_DTYPES)}",
    )
    req(br.impl in BRAIN_IMPLS, f"brain.impl {br.impl!r} not in {BRAIN_IMPLS}")
    req(
        ru.runtime_checks in RUNTIME_CHECKS,
        f"run.runtime_checks {ru.runtime_checks!r} not in {RUNTIME_CHECKS}",
    )

    # Cadences.
    if ru.metrics_every >= 1:
        req(
            ru.checkpoint_every >= 1 and ru.checkpoint_every % ru.metrics_every == 0,
            "run.checkpoint_every must be a positive multiple of run.metrics_every",
        )
        req(
            ru.heavy_metrics_every >= 1 and ru.heavy_metrics_every % ru.metrics_every == 0,
            "run.heavy_metrics_every must be a positive multiple of run.metrics_every",
        )

    # Weather (DESIGN §30). Off: every weather key is inert, so it must keep its default (no config can
    # change its hash without changing its behaviour). On: weather.validate_config returns the messages.
    wthr = cfg.weather
    if not wthr.enabled:
        wdef = WeatherConfig()
        for f in dataclasses.fields(WeatherConfig):
            if f.name != "enabled" and getattr(wthr, f.name) != getattr(wdef, f.name):
                errs.append(
                    f"weather.{f.name} is inert while weather.enabled is false: it must keep its default"
                )
    elif sph and "weather" not in SPHERE_READY:
        pass  # refused above by the readiness gate; the weather rules are the box lattice's
    else:
        errs.extend(weather.validate_config(cfg))

    # Director (valid in 3D too: the column catastrophe, scaled radius bounds and 3D prompt, §27.10).
    req(di.persona in PERSONAS, f"director.persona {di.persona!r} not in {PERSONAS}")
    req(di.rule_every >= 1, f"director.rule_every must be ≥ 1, got {di.rule_every}")
    req(di.budget_per_100k >= 0, f"director.budget_per_100k must be ≥ 0, got {di.budget_per_100k}")
    req(
        di.llm_every >= 0 and (di.rule_every < 1 or di.llm_every % di.rule_every == 0),
        f"director.llm_every must be 0 (off) or a positive multiple of director.rule_every "
        f"{di.rule_every}, got {di.llm_every}",
    )
    req(
        math.isfinite(di.llm_temperature) and 0.0 <= di.llm_temperature <= 2.0,
        f"director.llm_temperature must be in [0, 2], got {di.llm_temperature}",
    )
    req(
        math.isfinite(di.llm_timeout_s) and di.llm_timeout_s > 0,
        f"director.llm_timeout_s must be positive and finite, got {di.llm_timeout_s}",
    )
    req(di.llm_max_tokens >= 1, f"director.llm_max_tokens must be ≥ 1, got {di.llm_max_tokens}")
    if di.persona == RANDOM_PERSONA:  # the yoked control: moves come from another run's log, never a model
        req(di.yoke_log.strip() != "", "director.persona 'random' needs director.yoke_log (a director.jsonl)")
        req(
            di.llm_every == 0,
            f"director.persona 'random' has no LLM tier: llm_every must be 0, got {di.llm_every}",
        )
    else:
        req(di.yoke_log == "", f"director.yoke_log is only for the random persona, not {di.persona!r}")
    req(ru.llm_model.strip() != "", "run.llm_model must not be empty")
    if di.llm_every > 0:  # the endpoint is user config: only http(s) is ever contacted (DIRECTOR.md L11)
        req(is_http_url(ru.llm_url), f"run.llm_url must be an http(s) URL, got {ru.llm_url!r}")
    req(ru.director in RUN_DIRECTOR_MODES, f"run.director {ru.director!r} not in {RUN_DIRECTOR_MODES}")
    if ru.director != "off":
        req(
            di.enabled,
            "run.director live/replay/player needs director.enabled (use --director, which sets it)",
        )
        if di.rule_every >= 1 and ru.metrics_every >= 1 and ru.heavy_metrics_every >= 1:
            req(
                di.rule_every % ru.metrics_every == 0 and di.rule_every % ru.heavy_metrics_every == 0,
                f"director.rule_every {di.rule_every} must be a multiple of run.metrics_every "
                f"{ru.metrics_every} and run.heavy_metrics_every {ru.heavy_metrics_every}",
            )

    # Game (§29.4): observer-only; `game.observe` is a plain bool (no rule), the house rate is a probability.
    gm = cfg.game
    req(0.0 <= gm.house_rate <= 1.0, f"game.house_rate must be in [0, 1], got {gm.house_rate}")
    # The tracer rig and the run dumps (W2, DESIGN §32): host observers of the weather, so they need it.
    req(0 <= gm.tracers <= TRACERS_MAX, f"game.tracers must be in [0, {TRACERS_MAX}], got {gm.tracers}")
    req(gm.tracer_period >= 1, f"game.tracer_period must be ≥ 1, got {gm.tracer_period}")
    if gm.tracers > 0:
        req(gm.observe, "game.tracers > 0 needs game.observe (the tracers are an observer)")
        req(cfg.weather.enabled, "game.tracers > 0 needs weather.enabled (the tracers move with its wind)")
    req(ru.eco_every >= 0, f"run.eco_every must be ≥ 0, got {ru.eco_every}")
    req(ru.wx_every >= 0, f"run.wx_every must be ≥ 0, got {ru.wx_every}")
    req(ru.mat_every >= 0, f"run.mat_every must be ≥ 0, got {ru.mat_every}")
    req(ru.arms_every >= 0, f"run.arms_every must be ≥ 0, got {ru.arms_every}")
    if ru.mat_every > 0:
        req(cfg.structures.enabled, "run.mat_every > 0 needs structures.enabled")
    if ru.arms_every > 0:
        req(cfg.arms.enabled, "run.arms_every > 0 needs arms.enabled")
    req(ru.niche_every >= 0, f"run.niche_every must be ≥ 0, got {ru.niche_every}")
    if ru.niche_every > 0:
        req(cfg.niche.farm, "run.niche_every > 0 needs niche.farm")
    if ru.eco_every > 0:
        req(
            gm.observe,
            "run.eco_every > 0 needs game.observe (the dump holds the observer channel's emission)",
        )
        req(
            cfg.weather.enabled,
            "run.eco_every > 0 needs weather.enabled (the dump holds the wind at the agents)",
        )
        req(w.dims == 3, "run.eco_every > 0 needs a 3D world (the dump holds height and vertical wind)")
    if ru.wx_every > 0:
        req(cfg.weather.enabled, "run.wx_every > 0 needs weather.enabled")

    # Feature extensions (DESIGN §43): the protection floor, then each registered feature's own rules.
    req(0.0 <= co.protect_floor <= 0.5, f"combat.protect_floor {co.protect_floor} must be in [0, 0.5]")
    if co.protect_floor > 0:
        worst = co.efficiency * co.attack_max * (1.0 - co.defend_efficiency) * co.protect_floor
        if worst < en.attack_cost:
            log.warning(
                "combat.protect_floor %s: the worst-case attack gains %.0f < attack_cost %d, so predation "
                "against a fully defended, fully protected victim is unprofitable",
                co.protect_floor,
                worst,
                en.attack_cost,
            )
    try:
        ext.validate_all(cfg)
    except ConfigError as e:
        errs.append(str(e))

    if errs:
        raise ConfigError(f"{len(errs)} config problem(s): " + "; ".join(errs))
