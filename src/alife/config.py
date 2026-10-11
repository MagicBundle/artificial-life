"""Configuration: frozen dataclasses, presets, load/save/override, validation and hashing (§17).

Defaults are the ``64k`` preset. Presets and overlays are sparse TOML overrides shipped as package data.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import importlib.resources
import json
import math
import tomllib
import typing
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final
from urllib.parse import urlsplit

import mlx.core as mx

from alife import layout

R_HARD: Final = (1 << 24) - 1  # hard resource cap per field cell (§10.9); re-exported as state.R_HARD


class ConfigError(ValueError):
    """Invalid configuration: unknown keys, bad types, or violated validation rules."""


@dataclass(frozen=True)
class WorldConfig:
    size: float = 1024.0  # torus side L
    field_cell: float = 4.0  # FG = size / field_cell (must be an integer ≤ 1024)


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


@dataclass(frozen=True)
class CombatConfig:
    range: float = 3.0
    threshold: float = 0.5
    attack_max: int = 4000
    efficiency: float = 0.7
    defend_efficiency: float = 0.75
    attack_bonded: bool = False


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


@dataclass(frozen=True)
class DirectorConfig:  # ecosystem director (docs/DIRECTOR.md); added after v0.1
    enabled: bool = False  # static: the compiled step reads the live environment state (State.env)
    # Policy keys: read by the host-side director between steps, never by the compiled step. They are not
    # hashed (``UNHASHED_KEYS``) and may change on resume.
    persona: str = "gardener"  # gardener | trickster | architect
    goal: str = ""  # the Architect's goal text (stored, not interpreted before the LLM tier)
    rule_every: int = 1000  # steps between rule-tier decisions; a multiple of the metrics cadences
    budget_per_100k: int = 0  # intervention cost units per rolling 100k steps; 0 = the persona's default
    # The LLM tier (build step 3): off at 0. Where it runs and which model answers are run.llm_* keys.
    llm_every: int = 0  # steps between LLM calls; 0 = off, else a multiple of rule_every
    llm_temperature: float = 0.4  # sampling temperature of the request
    llm_timeout_s: float = 60.0  # seconds to wait for a reply before the rule tier takes over
    llm_max_tokens: int = 400  # reply length limit of the request


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
    director: str = "off"  # off | live | replay (needs director.enabled); resume may change it
    llm_url: str = "http://localhost:1234/v1/chat/completions"  # the LLM tier's endpoint (LM Studio)
    llm_model: str = "gemma-4-26b-a4b-it-mlx@8bit"  # the model named in each request


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


PRESETS: tuple[str, ...] = ("tiny", "dev", "64k", "256k")
OVERLAYS: tuple[str, ...] = ("fast",)
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


@functools.cache
def dims(cfg: Config) -> Dims:
    """Derived sizes for a (validated) config. Cached per config."""
    n, m = cfg.population.n_max, cfg.population.max_births_per_step
    h = cfg.brain.hidden
    g_hash = math.floor(cfg.world.size / cfg.sensing.radius)
    w1_rows, w2_rows = layout.D_IN + h, h + 1
    e_max, s = cfg.energy.e_max, cfg.energy.structure
    return Dims(
        n=n,
        m=m,
        n_cap=n - m,
        dl=2 * m,
        k=cfg.sensing.k_neighbors,
        c=cfg.sensing.cell_capacity,
        q=9 * cfg.sensing.cell_capacity,
        g_hash=g_hash,
        cell_hash=cfg.world.size / g_hash if g_hash > 0 else float("inf"),
        fg=_field_fg(cfg),
        h=h,
        w1_rows=w1_rows,
        w2_rows=w2_rows,
        params=w1_rows * h + w2_rows * layout.D_OUT,
        pp=cfg.fields.n_patches,
        wdtype=WEIGHT_DTYPES.get(cfg.brain.weight_dtype, mx.float32),
        occ_limit=min((1 << 15) - 1, ((1 << 31) - 1 - R_HARD) // max(2 * e_max + s, 1)),
    )


def stability_number(cfg: Config) -> float:
    """Gershgorin bound 2(B·k_s + K·k_r)/drag of the explicit overdamped step (§9)."""
    return (
        2.0
        * (layout.B * cfg.bonds.k_spring + cfg.sensing.k_neighbors * cfg.physics.k_repel)
        / cfg.physics.drag
    )


# --- dict conversion ------------------------------------------------------------------------------------


def config_to_dict(cfg: Config) -> dict[str, dict[str, Any]]:
    """Plain JSON-compatible dict (tuples become lists)."""
    out: dict[str, dict[str, Any]] = {}
    for sec in dataclasses.fields(Config):
        sub = getattr(cfg, sec.name)
        out[sec.name] = {
            f.name: list(v) if isinstance(v := getattr(sub, f.name), tuple) else v
            for f in dataclasses.fields(sub)
        }
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
    if name not in PRESETS and name not in OVERLAYS:
        raise ConfigError(f"unknown preset or overlay {name!r} (presets: {PRESETS}, overlays: {OVERLAYS})")
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
        if ov not in OVERLAYS:
            raise ConfigError(f"unknown overlay {ov!r} (choose from {OVERLAYS})")
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
    Path(path).write_text(_canonical_json(config_to_dict(cfg)), encoding="utf-8")


# Keys added after v0.1 (section, key). They are omitted from ``config_hash`` while they equal their defaults,
# so runs created before they existed keep their hash and stay resumable (§18).
POST_V01_KEYS: Final[tuple[tuple[str, str], ...]] = (
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
)

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
)
DIRECTOR_POLICY_KEYS: Final = tuple(key for _, key in UNHASHED_KEYS)
DIRECTOR_MODES: Final = ("off", "live", "replay")
PERSONAS: Final = ("gardener", "trickster", "architect")


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


def config_hash(cfg: Config) -> str:
    """SHA-256 hex of the canonical JSON of every section except ``run``.

    Keys added after v0.1 (``POST_V01_KEYS``) are omitted at their defaults (``UNHASHED_KEYS``: always), and a
    section left empty by that (``director``) is omitted too, so pre-director runs keep their hash.
    """
    d = config_to_dict(cfg)
    d.pop("run")
    defaults = config_to_dict(Config())
    for sec, key in POST_V01_KEYS:
        if (sec, key) in UNHASHED_KEYS or d[sec][key] == defaults[sec][key]:
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
    ):
        req(getattr(en, name) >= 0, f"energy.{name} must be ≥ 0, got {getattr(en, name)}")
    req(en.brain_cost_per_l1 >= 0, f"energy.brain_cost_per_l1 must be ≥ 0, got {en.brain_cost_per_l1}")
    req(rp.birth_cost >= 0 and rp.min_age >= 0, "repro.birth_cost and repro.min_age must be ≥ 0")

    # World geometry.
    if w.size > 0 and w.field_cell > 0 and sen.radius > 0 and math.isfinite(w.size):
        ratio = w.size / w.field_cell
        req(ratio == round(ratio), f"world.size / world.field_cell must be an integer, got {ratio}")
        req(3 <= round(ratio) <= 1024, f"field grid FG = {round(ratio)} must be in [3, 1024]")
        g_hash = math.floor(w.size / sen.radius)
        req(g_hash >= 3, f"hash grid G_h = floor(size / radius) = {g_hash} must be ≥ 3")

    # Interaction ranges.
    need = max(bo.form_range, co.range, 2 * ph.radius)
    req(
        sen.radius >= need,
        f"sensing.radius {sen.radius} must be ≥ max(form_range, combat.range, 2·r) = {need}",
    )
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

    # Director.
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
    req(ru.llm_model.strip() != "", "run.llm_model must not be empty")
    if di.llm_every > 0:  # the endpoint is user config: only http(s) is ever contacted (DIRECTOR.md L11)
        req(is_http_url(ru.llm_url), f"run.llm_url must be an http(s) URL, got {ru.llm_url!r}")
    req(ru.director in DIRECTOR_MODES, f"run.director {ru.director!r} not in {DIRECTOR_MODES}")
    if ru.director != "off":
        req(di.enabled, "run.director live/replay needs director.enabled (use --director, which sets it)")
        if di.rule_every >= 1 and ru.metrics_every >= 1 and ru.heavy_metrics_every >= 1:
            req(
                di.rule_every % ru.metrics_every == 0 and di.rule_every % ru.heavy_metrics_every == 0,
                f"director.rule_every {di.rule_every} must be a multiple of run.metrics_every "
                f"{ru.metrics_every} and run.heavy_metrics_every {ru.heavy_metrics_every}",
            )

    if errs:
        raise ConfigError(f"{len(errs)} config problem(s): " + "; ".join(errs))
