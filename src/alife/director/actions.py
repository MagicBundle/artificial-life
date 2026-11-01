"""The director's intervention menu: bounds, clamping, strict validation and application (DIRECTOR.md §5).

An *entry* is the log record of one intervention. Its ``params`` are explicit, absolute and already clamped (a
patch index and its new gain, absolute patch coordinates, the catastrophe centre), so applying an entry never
consults the rules or an RNG. ``clamp`` (live mode) turns a rule's raw request into ``params`` / ``prev`` /
``duration``; ``validate_entry`` (replay mode) rejects whatever is outside the menu instead of clamping it;
``apply`` writes a validated entry into ``State.env`` (and ``patch_pos`` for a patch move) with ``_replace``,
keeping every array's shape and dtype so the compiled step is not retraced.

Host-only: nothing here is traced, and it only reads the few ``env_*`` scalars and two small patch arrays.

On a sphere (DESIGN §46.7) a place is ``(x, y) = (longitude, latitude)`` in degrees, longitude in [-180, 180)
and latitude in [-90, 90]: the catastrophe's ``center`` and a patch move's absolute ``x``, ``y``. The engine's
state holds the catastrophe centre in radians and the patches as unit vectors; ``apply`` converts. Lengths
(the catastrophe's radius, a move's distance) are arc lengths at the floor, and the world's "width" in the
bounds below is the equator's circumference ``2·π·Rp``.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, NamedTuple

import mlx.core as mx
import numpy as np

from alife import geom, ops
from alife.config import Config, ConfigError, is_3d
from alife.invariants import (
    ENV_DRIFT_MULT,
    ENV_PATCH_GAIN,
    ENV_REGEN_MULT,
    ENV_SEASON_AMP,
    env_out_of_range,
)
from alife.state import State, to_numpy

SOURCES: Final = ("rule", "llm", "revert")
GAME_SOURCES: Final = (*SOURCES, "player")  # what a record of the player channel (``is_player``) may say
IMM_PENDING_CAP: Final = 1024  # the most immigrants the director may leave pending in ``env_imm_request``
MOVE_MAX_FRAC: Final = 0.25  # a patch moves by at most L/4
CATASTROPHE_RADIUS: Final = (20.0, 200.0)  # the upper end is also capped by L/2
CATASTROPHE_REF_SIZE: Final = 1024.0  # the world width ``CATASTROPHE_RADIUS`` is written for (3D scales it)
_MOVE_SLACK: Final = 1e-5  # relative slack for the f32 rounding of an absolute coordinate


class ActionError(ConfigError):
    """A malformed, unknown or out-of-bounds intervention (exit 2 when it reaches the CLI)."""


@dataclass(frozen=True)
class Spec:
    """One menu row: cost against the rolling budget, cooldown (from the effect's end), parameter bounds."""

    cost: int
    cooldown: int
    bounds: Mapping[str, tuple[float, float]]
    duration: tuple[int, int] | None = None  # timed effects only: reverted after this many steps
    quantum: int | None = None  # durations round up to a multiple of this (None: ``director.rule_every``)
    excluded: Mapping[str, tuple[float, float]] = field(default_factory=dict)  # open ranges inside a bound


# DIRECTOR.md §5. ``set_seasons`` has no mode (it stays ``fields.season_mode``); the mutation-floor nudge is
# not implemented. The geometric bounds (move distance, catastrophe radius) are ``MOVE_MAX_FRAC`` and
# ``CATASTROPHE_RADIUS``.
MENU: Final[Mapping[str, Spec]] = {
    "set_seasons": Spec(1, 50_000, {"period": (2_000, 100_000), "amp": (0.0, 0.9)}),
    "fertility_shift": Spec(1, 20_000, {"gain": (0.25, 2.0)}),
    "habitat_loss": Spec(2, 50_000, {}, duration=(10_000, 200_000)),
    "regen_pulse": Spec(1, 30_000, {"mult": (0.4, 1.8)}, duration=(5_000, 50_000)),
    "immigration_pulse": Spec(1, 20_000, {"count": (16, IMM_PENDING_CAP)}),
    "catastrophe": Spec(3, 100_000, {"kill_p": (0.2, 0.9)}),
    "drift_change": Spec(1, 50_000, {"mult": (0.5, 10.0)}),
}
TIMED: Final = tuple(name for name, spec in MENU.items() if spec.duration is not None)
_REVERT_PARAMS: Final = {
    "habitat_loss": ("patch", "gain"),
    "fertility_shift": ("patch", "gain"),
    "regen_pulse": ("mult",),
}

# The player's table (GAMES.md §5.5, G3b spec §6.1.4 and §6.2): an override of ``MENU`` for entries of the
# player channel (``is_player``), not a subset of it: its durations (3,000 to 12,000) lie below ``MENU``'s
# lower bounds, round up to 250 (not ``rule_every``), Strike costs 1, and Bloom is a *timed* gain (``MENU``'s
# gain is permanent). Bounds of a "radius" are written in the units of a 1024-wide world and scaled as the
# research range is (``radius_bounds``). Everything is provisional until the G3b Mac day; it is data under a
# version.
GAME_SPEC_VERSION: Final = "game-v0"
PLAYER_QUANTUM: Final = 250  # player durations are multiples of the metrics row
PLAYER_BUDGET: Final = 6  # influence: cost units per ``PLAYER_BUDGET_WINDOW`` steps
PLAYER_BUDGET_WINDOW: Final = 18_000
GAME_SPECS: Final[Mapping[str, Spec]] = {
    "regen_pulse": Spec(
        1,
        3_000,
        {"mult": (0.6, 1.5)},
        duration=(3_000, 12_000),
        quantum=PLAYER_QUANTUM,
        excluded={"mult": (0.9, 1.15)},  # a dead zone: no effect worth a cost, refused rather than clamped
    ),
    "fertility_shift": Spec(1, 3_000, {"gain": (1.5, 2.0)}, duration=(3_000, 12_000), quantum=PLAYER_QUANTUM),
    "habitat_loss": Spec(2, 6_000, {}, duration=(3_000, 12_000), quantum=PLAYER_QUANTUM),
    "catastrophe": Spec(1, 3_000, {"kill_p": (0.3, 0.6), "radius": (20.0, 80.0)}),
}
GAME_POWERS: Final = {
    "food": "regen_pulse",
    "bloom": "fertility_shift",
    "blight": "habitat_loss",
    "strike": "catastrophe",
}


def is_player(entry: Mapping[str, Any]) -> bool:
    """True for a record of the player channel: a power (``source`` ``player``) or a revert written for one
    (``source`` ``revert`` with ``scope`` ``player``). Rule P (G3b spec §6.1.2) applies to these records."""
    return entry.get("source") == "player" or (
        entry.get("source") == "revert" and entry.get("scope") == "player"
    )


def is_timed(entry: Mapping[str, Any]) -> bool:
    """True if the entry has a duration. "Timed" is a property of the entry, not of the action's name:
    ``fertility_shift`` is permanent in the research menu and timed in the player's."""
    return entry.get("duration") is not None


class EnvView(NamedTuple):
    """The host copy of the environment fields the menu reads."""

    regen_mult: float
    season_amp: float
    season_period: int
    drift_mult: float
    imm_request: int
    patch_gain: np.ndarray  # (PP,) f32
    patch_pos: np.ndarray  # (PP, 2) f32; (PP, 3) unit vectors on a sphere


class Clamped(NamedTuple):
    params: dict[str, Any]
    prev: dict[str, Any]
    duration: int | None


def env_view(s: State) -> EnvView:
    return EnvView(
        float(to_numpy(s.env_regen_mult)),
        float(to_numpy(s.env_season_amp)),
        int(to_numpy(s.env_season_period)),
        float(to_numpy(s.env_drift_mult)),
        int(to_numpy(s.env_imm_request)),
        to_numpy(s.env_patch_gain),
        to_numpy(s.patch_pos),
    )


def lever_keys(action: str, params: Mapping[str, Any]) -> tuple[str, ...]:
    """The levers an entry writes that another effect may also be holding: ``regen`` or ``patch:<p>``.

    A patch *move* does not touch the gain, so it takes no lever.
    """
    if action == "regen_pulse":
        return ("regen",)
    if action == "habitat_loss" or (action == "fertility_shift" and "gain" in params):
        return (f"patch:{params['patch']}",)
    return ()


def round_up(steps: int, multiple: int) -> int:
    return -(-steps // multiple) * multiple


def _f32(x: float) -> float:
    return float(np.float32(x))


def _clip(x: float, lo: float, hi: float) -> float:
    """``x`` clipped to ``[lo, hi]`` (both taken as float32) and rounded to float32, so it lies inside."""
    lo, hi = _f32(lo), _f32(hi)
    return min(max(_f32(x), lo), hi)


def _number(raw: Mapping[str, Any], name: str) -> float:
    v = raw.get(name)
    if isinstance(v, bool) or not isinstance(v, numbers.Real) or not math.isfinite(v):
        raise ActionError(f"parameter {name!r} must be a finite number, got {v!r}")
    return float(v)


def _patch(raw: Mapping[str, Any], view: EnvView) -> int:
    p = _number(raw, "patch")
    if p != int(p) or not 0 <= p < len(view.patch_gain):
        raise ActionError(f"patch {p!r} is not in [0, {len(view.patch_gain)})")
    return int(p)


def span(cfg: Config) -> float:
    """The world's width in the menu's length bounds: ``world.size`` in a box, the equator ``2·π·Rp`` on a
    sphere."""
    g = geom.geometry(cfg)
    return 2.0 * math.pi * g.radius if g.name == "sphere" else g.size


def area_width(cfg: Config) -> float:
    """The width of the square whose area is the planet's, ``sqrt(4π)·Rp``: the unit of the catastrophe's
    radius on a sphere (``radius_bounds``, the rule tier). A box column of radius ``f·L`` covers ``π f²`` of
    the floor whatever ``L``; a cap of arc radius ``f·sqrt(4π)·Rp`` covers ``(1 − cos(f·sqrt(4π)))/2``, which
    is ``π f²`` to the second order, so the menu's largest strike (``f = 200/1024``) is 11.5 % of the planet
    against the box's 12.0 % at any size (the equator ``2π·Rp`` would give 33 %)."""
    g = geom.geometry(cfg)
    return math.sqrt(4.0 * math.pi) * g.radius if g.name == "sphere" else g.size


def is_sphere(cfg: Config) -> bool:
    return cfg.world.geometry == "sphere"


LON_MAX: Final = 180.0
LAT_MAX: Final = 90.0


def wrap_lon(lon: float) -> float:
    """A longitude in degrees brought into [-180, 180) in float32 (strictly below 180 after the rounding)."""
    v = _f32(((lon + LON_MAX) % (2 * LON_MAX)) - LON_MAX)
    return -LON_MAX if v >= LON_MAX else v


def lonlat_of(unit: np.ndarray) -> list[float]:
    """``[lon, lat]`` in float32 degrees of a unit vector (the stored form of a sphere's place)."""
    lat, lon = geom.to_latlon(np.asarray(unit, dtype=np.float64))
    return [wrap_lon(float(lon)), _clip(float(lat), -LAT_MAX, LAT_MAX)]


def unit_of(lon: float, lat: float) -> np.ndarray:
    """The unit vector (float32) at longitude ``lon`` and latitude ``lat`` in degrees."""
    return geom.from_latlon(lat, lon).astype(np.float32)


def arc_between(a: Sequence[float], b: Sequence[float], radius: float) -> float:
    """The great-circle arc between two ``[lon, lat]`` places on a sphere of ``radius`` (degrees in)."""
    return float(
        geom.great_circle_distance(geom.from_latlon(a[1], a[0]), geom.from_latlon(b[1], b[0]), radius)
    )


def move_on_sphere(unit: np.ndarray, dx: float, dy: float, radius: float) -> np.ndarray:
    """The unit vector reached from ``unit`` by walking ``dx`` east and ``dy`` north (arc lengths at
    ``radius``) along the great circle of that direction."""
    u = np.asarray(unit, dtype=np.float64)
    e, n = geom.local_frame(u)
    step = math.hypot(dx, dy)
    if step == 0.0:
        return u
    direction = (dx * e + dy * n) / step
    ang = step / radius
    return math.cos(ang) * u + math.sin(ang) * direction


def _wrap(x: list[float], size: float) -> list[float]:
    """Wrap a coordinate pair into ``[0, size)`` with ``ops.wrap``'s float32 guard (strictly below)."""
    return [float(v) for v in to_numpy(ops.wrap(mx.array(x, dtype=mx.float32), size))]


def _device(value: float, bounds: tuple[float, float]) -> float:
    """Clip to a step-1 device range (``invariants.ENV_*``); the menu bounds already lie inside them."""
    return _clip(value, *bounds)


def radius_bounds(cfg: Config, specs: Mapping[str, Spec] | None = None) -> tuple[float, float]:
    """The catastrophe radius range: ``CATASTROPHE_RADIUS`` capped by L/2. In 3D the menu's range is scaled
    by ``L / 1024`` (``CATASTROPHE_REF_SIZE``, §27.10): the column covers the same share of the floor as the
    2D disc of a 1024-wide world, rather than most of a small 3D world. A table that lists a ``radius`` bound
    for ``catastrophe`` (the player's) gives the range, in the same 1024-world units, scaled the same way."""
    ref = (specs if specs is not None else MENU)["catastrophe"].bounds.get("radius", CATASTROPHE_RADIUS)
    if is_sphere(cfg):  # the area-equivalent width: the largest cap is the box column's 12 % of the surface
        width = area_width(cfg)
        k = width / CATASTROPHE_REF_SIZE
        return ref[0] * k, min(ref[1] * k, width / 2)
    if is_3d(cfg):
        k = cfg.world.size / CATASTROPHE_REF_SIZE
        return ref[0] * k, min(ref[1] * k, cfg.world.size / 2)
    hi = min(ref[1], cfg.world.size / 2)
    return min(ref[0], hi), hi


# --- live: clamp ------------------------------------------------------------------------------------------


def _bounds_of(action: str, specs: Mapping[str, Spec]) -> tuple[int, int]:
    bounds = specs[action].duration
    if bounds is None:
        raise ActionError(f"{action} is not a timed action")
    return bounds


def _quantum(action: str, specs: Mapping[str, Spec], cfg: Config) -> int:
    """The multiple a duration of ``action`` rounds up to: the table's, else ``director.rule_every``."""
    return specs[action].quantum or cfg.director.rule_every


def _clamp_set_seasons(
    raw: Mapping[str, Any],
    view: EnvView,
    cfg: Config,
    duration: int | float | None,
    specs: Mapping[str, Spec],
) -> Clamped:
    lo, hi = specs["set_seasons"].bounds["period"]
    period = int(min(max(round(_number(raw, "period")), lo), hi))
    amp = _device(_clip(_number(raw, "amp"), *specs["set_seasons"].bounds["amp"]), ENV_SEASON_AMP)
    return Clamped(
        {"period": period, "amp": amp}, {"period": view.season_period, "amp": view.season_amp}, None
    )


def _clamp_fertility_shift(
    raw: Mapping[str, Any],
    view: EnvView,
    cfg: Config,
    duration: int | float | None,
    specs: Mapping[str, Spec],
) -> Clamped:
    p = _patch(raw, view)
    spec = specs["fertility_shift"]
    if "gain" in raw or spec.duration is not None:  # a timed table (the player's) has the gain form only
        gain = _device(_clip(_number(raw, "gain"), *spec.bounds["gain"]), ENV_PATCH_GAIN)
        d = None
        if spec.duration is not None:
            d = _duration(
                _bounds_of("fertility_shift", specs), duration, _quantum("fertility_shift", specs, cfg)
            )
        return Clamped({"patch": p, "gain": gain}, {"patch": p, "gain": float(view.patch_gain[p])}, d)
    size = span(cfg)
    dx, dy = _number(raw, "dx"), _number(raw, "dy")
    cap = MOVE_MAX_FRAC * size
    norm = math.hypot(dx, dy)
    scale = cap / norm if norm > cap else 1.0
    if is_sphere(cfg):  # walk along the great circle; the places are stored as [lon, lat] degrees
        moved = move_on_sphere(view.patch_pos[p], dx * scale, dy * scale, cfg.world.radius)
        new, old = lonlat_of(moved), lonlat_of(view.patch_pos[p])
        return Clamped({"patch": p, "x": new[0], "y": new[1]}, {"patch": p, "x": old[0], "y": old[1]}, None)
    old = [float(v) for v in view.patch_pos[p]]
    new = _wrap([old[0] + dx * scale, old[1] + dy * scale], size)
    return Clamped({"patch": p, "x": new[0], "y": new[1]}, {"patch": p, "x": old[0], "y": old[1]}, None)


def _duration(bounds: tuple[int, int], raw: int | float | None, quantum: int) -> int:
    if raw is None or isinstance(raw, bool) or not math.isfinite(raw):
        raise ActionError(f"a timed action needs a finite duration, got {raw!r}")
    lo, hi = bounds
    return round_up(int(min(max(round(raw), lo), hi)), quantum)


def _clamp_habitat_loss(
    raw: Mapping[str, Any],
    view: EnvView,
    cfg: Config,
    duration: int | float | None,
    specs: Mapping[str, Spec],
) -> Clamped:
    p = _patch(raw, view)
    d = _duration(_bounds_of("habitat_loss", specs), duration, _quantum("habitat_loss", specs, cfg))
    return Clamped({"patch": p, "gain": 0.0}, {"patch": p, "gain": float(view.patch_gain[p])}, d)


def _outside(where: str, name: str, v: float, specs: Mapping[str, Spec], action: str) -> None:
    """``ActionError`` if ``v`` lies strictly inside the open range the table excludes for ``name``."""
    hole = specs[action].excluded.get(name)
    if hole is not None and _f32(hole[0]) < v < _f32(hole[1]):
        raise ActionError(f"{where}: {name} {v!r} lies in the range {hole[0]} to {hole[1]}, which is refused")


def _clamp_regen_pulse(
    raw: Mapping[str, Any],
    view: EnvView,
    cfg: Config,
    duration: int | float | None,
    specs: Mapping[str, Spec],
    restore: Mapping[str, Any] | None,
) -> Clamped:
    mult = _device(_clip(_number(raw, "mult"), *specs["regen_pulse"].bounds["mult"]), ENV_REGEN_MULT)
    _outside("regen_pulse", "mult", mult, specs, "regen_pulse")
    d = _duration(_bounds_of("regen_pulse", specs), duration, _quantum("regen_pulse", specs, cfg))
    prev = dict(restore) if restore is not None else {"mult": view.regen_mult}  # superseding keeps the base
    return Clamped({"mult": mult}, prev, d)


def _clamp_immigration_pulse(
    raw: Mapping[str, Any],
    view: EnvView,
    cfg: Config,
    duration: int | float | None,
    specs: Mapping[str, Spec],
) -> Clamped:
    lo, hi = specs["immigration_pulse"].bounds["count"]
    count = int(min(max(round(_number(raw, "count")), lo), hi, IMM_PENDING_CAP - view.imm_request))
    if count < lo:
        raise ActionError(f"{view.imm_request} immigrants are already pending (cap {IMM_PENDING_CAP})")
    return Clamped({"count": count}, {"pending": view.imm_request}, None)


def _clamp_catastrophe(
    raw: Mapping[str, Any],
    view: EnvView,
    cfg: Config,
    duration: int | float | None,
    specs: Mapping[str, Spec],
) -> Clamped:
    if is_sphere(cfg):
        centre = [wrap_lon(_number(raw, "x")), _clip(_number(raw, "y"), -LAT_MAX, LAT_MAX)]
    else:
        centre = _wrap([_number(raw, "x"), _number(raw, "y")], cfg.world.size)
    radius = _clip(_number(raw, "radius"), *radius_bounds(cfg, specs))
    kill_p = _clip(_number(raw, "kill_p"), *specs["catastrophe"].bounds["kill_p"])
    return Clamped({"center": centre, "radius": radius, "kill_p": kill_p}, {}, None)


def _clamp_drift_change(
    raw: Mapping[str, Any],
    view: EnvView,
    cfg: Config,
    duration: int | float | None,
    specs: Mapping[str, Spec],
) -> Clamped:
    mult = _device(_clip(_number(raw, "mult"), *specs["drift_change"].bounds["mult"]), ENV_DRIFT_MULT)
    return Clamped({"mult": mult}, {"mult": view.drift_mult}, None)


def clamp(
    action: str,
    raw: Mapping[str, Any],
    duration: int | float | None,
    view: EnvView,
    cfg: Config,
    restore: Mapping[str, Any] | None = None,
    specs: Mapping[str, Spec] | None = None,
) -> Clamped:
    """Clamp a rule's raw request to the menu bounds and the device ranges (live mode).

    ``raw`` carries the request in its natural form (a move as ``dx, dy``, a catastrophe as ``x, y``); the
    result holds the explicit values that are logged and applied. A timed action's ``duration`` is clamped and
    rounded up to a multiple of the table's ``quantum`` (``director.rule_every`` for ``MENU``). ``restore``
    (regen pulses only) is the value to return to when the new pulse supersedes an active one. ``specs`` is
    the table (``MENU``, or ``GAME_SPECS`` for the player's powers). Raises ``ActionError`` for an action not
    in the table or an unusable request (missing or non-finite values, a patch that does not exist, a full
    immigration queue, a value in a range the table refuses).
    """
    specs = MENU if specs is None else specs
    if action not in specs:
        raise ActionError(f"unknown action {action!r}")
    if action == "regen_pulse":
        return _clamp_regen_pulse(raw, view, cfg, duration, specs, restore)
    return _CLAMPERS[action](raw, view, cfg, duration, specs)


_CLAMPERS: Final = {
    "set_seasons": _clamp_set_seasons,
    "fertility_shift": _clamp_fertility_shift,
    "habitat_loss": _clamp_habitat_loss,
    "immigration_pulse": _clamp_immigration_pulse,
    "catastrophe": _clamp_catastrophe,
    "drift_change": _clamp_drift_change,
}


# --- replay: strict validation ----------------------------------------------------------------------


def _in_range(where: str, name: str, v: Any, lo: float, hi: float, *, integer: bool = False) -> float:
    ok = isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v)
    if ok and integer:
        ok = v == int(v)
    if not ok:
        raise ActionError(f"{where}: {name} must be a finite {'integer' if integer else 'number'}, got {v!r}")
    lo, hi = (lo, hi) if integer else (_f32(lo), _f32(hi))
    if not lo <= v <= hi:
        raise ActionError(f"{where}: {name} {v!r} is outside [{lo}, {hi}]")
    return float(v)


def _check_keys(where: str, params: Mapping[str, Any], allowed: set[str]) -> None:
    if set(params) != allowed:
        raise ActionError(f"{where}: params must be exactly {sorted(allowed)}, got {sorted(params)}")


def _check_patch(where: str, params: Mapping[str, Any], cfg: Config) -> int:
    return int(_in_range(where, "patch", params["patch"], 0, cfg.fields.n_patches - 1, integer=True))


def _check_lonlat(where: str, name: str, lon: Any, lat: Any) -> list[float]:
    """A sphere place [lon, lat] in degrees: lon in [-180, 180), lat in [-90, 90]."""
    x = _in_range(where, f"{name} x (longitude)", lon, -LON_MAX, LON_MAX)
    if x >= LON_MAX:
        raise ActionError(f"{where}: {name} x (longitude) {x!r} is not strictly below {LON_MAX:g}")
    return [x, _in_range(where, f"{name} y (latitude)", lat, -LAT_MAX, LAT_MAX)]


def _check_move(where: str, params: Mapping[str, Any], prev: Any, cfg: Config) -> None:
    if is_sphere(cfg):
        new = _check_lonlat(where, "", params["x"], params["y"])
        if not isinstance(prev, Mapping) or not {"x", "y"} <= set(prev):
            raise ActionError(f"{where}: a patch move must carry prev coordinates (x, y)")
        old = _check_lonlat(where, "prev", prev["x"], prev["y"])
        cap = MOVE_MAX_FRAC * span(cfg)
        if arc_between(old, new, cfg.world.radius) > cap * (1 + _MOVE_SLACK) + 1e-3:
            raise ActionError(f"{where}: the move exceeds a quarter of the equator = {cap:g}")
        return
    size = cfg.world.size
    for axis in ("x", "y"):
        v = _in_range(where, axis, params[axis], 0, size)
        if v >= size:
            raise ActionError(f"{where}: {axis} {v!r} is not strictly below the world size {size}")
    if not isinstance(prev, Mapping) or not {"x", "y"} <= set(prev):
        raise ActionError(f"{where}: a patch move must carry prev coordinates (x, y)")
    old = [_in_range(where, f"prev {a}", prev[a], 0, size) for a in ("x", "y")]
    d = [params[a] - o for a, o in zip(("x", "y"), old, strict=True)]
    d = [v - size * round(v / size) for v in d]  # torus minimum image
    if math.hypot(*d) > MOVE_MAX_FRAC * size * (1 + _MOVE_SLACK) + 1e-3:
        raise ActionError(f"{where}: the move exceeds L/4 = {MOVE_MAX_FRAC * size:g}")


def _check_forward(
    where: str,
    action: str,
    params: Mapping[str, Any],
    prev: Any,
    cfg: Config,
    specs: Mapping[str, Spec] | None = None,
) -> None:
    specs = MENU if specs is None else specs
    bounds = specs[action].bounds
    if action == "set_seasons":
        _check_keys(where, params, {"period", "amp"})
        _in_range(where, "period", params["period"], *bounds["period"], integer=True)
        _in_range(where, "amp", params["amp"], *bounds["amp"])
    elif action == "fertility_shift" and ("gain" in params or specs[action].duration is not None):
        _check_keys(where, params, {"patch", "gain"})  # a timed table has the gain form only
        _check_patch(where, params, cfg)
        _in_range(where, "gain", params["gain"], *bounds["gain"])
    elif action == "fertility_shift":
        _check_keys(where, params, {"patch", "x", "y"})
        _check_patch(where, params, cfg)
        _check_move(where, params, prev, cfg)
    elif action == "habitat_loss":
        _check_keys(where, params, {"patch", "gain"})
        _check_patch(where, params, cfg)
        _in_range(where, "gain", params["gain"], 0.0, 0.0)
    elif action == "regen_pulse":
        _check_keys(where, params, {"mult"})
        _outside(where, "mult", _in_range(where, "mult", params["mult"], *bounds["mult"]), specs, action)
    elif action == "immigration_pulse":
        _check_keys(where, params, {"count"})
        _in_range(where, "count", params["count"], *bounds["count"], integer=True)
    elif action == "catastrophe":
        _check_keys(where, params, {"center", "radius", "kill_p"})
        c = params["center"]
        if not isinstance(c, list | tuple) or len(c) != 2:
            raise ActionError(f"{where}: center must be [x, y], got {c!r}")
        if is_sphere(cfg):
            _check_lonlat(where, "center", c[0], c[1])
            c = ()
        for v in c:
            if _in_range(where, "center", v, 0, cfg.world.size) >= cfg.world.size:
                raise ActionError(f"{where}: center {c!r} is not strictly inside the world")
        _in_range(where, "radius", params["radius"], *radius_bounds(cfg, specs))
        _in_range(where, "kill_p", params["kill_p"], *bounds["kill_p"])
    elif action == "drift_change":
        _check_keys(where, params, {"mult"})
        _in_range(where, "mult", params["mult"], *bounds["mult"])


def _check_revert(
    where: str, action: str, params: Mapping[str, Any], cfg: Config, specs: Mapping[str, Spec] | None = None
) -> None:
    specs = MENU if specs is None else specs
    names = _REVERT_PARAMS.get(action) if action in specs and specs[action].duration is not None else None
    if names is None:
        raise ActionError(f"{where}: {action!r} has no timed effect to revert")
    _check_keys(where, params, set(names))
    if action in ("habitat_loss", "fertility_shift"):
        _check_patch(where, params, cfg)
        _in_range(where, "gain", params["gain"], *ENV_PATCH_GAIN)
    else:
        _in_range(where, "mult", params["mult"], *ENV_REGEN_MULT)


def _check_duration(
    where: str,
    action: str,
    entry: Mapping[str, Any],
    rule_every: int,
    specs: Mapping[str, Spec] | None = None,
) -> None:
    specs = MENU if specs is None else specs
    spec, d = specs[action], entry.get("duration")
    if spec.duration is None:
        if d is not None:
            raise ActionError(f"{where}: {action} takes no duration, got {d!r}")
        return
    quantum = spec.quantum or rule_every
    lo, hi = spec.duration
    _in_range(where, "duration", d, lo, round_up(hi, quantum), integer=True)
    if d % quantum:
        name = "rule_every" if spec.quantum is None else "the player's quantum"
        raise ActionError(f"{where}: duration {d} is not a multiple of {name} {quantum}")


def _check_player_fields(where: str, entry: Mapping[str, Any], specs: Mapping[str, Spec]) -> None:
    """The fields that make a record a player's: its scope, and for a power the command, the power's name
    (which must be the action's), the table's version and the cost the table gives."""
    if entry.get("scope") != "player":
        raise ActionError(f"{where}: a player record must carry scope 'player', got {entry.get('scope')!r}")
    command = entry.get("command")
    if command is not None and (isinstance(command, bool) or not isinstance(command, int) or command < 0):
        raise ActionError(f"{where}: command must be a non-negative integer, got {command!r}")
    if entry["source"] == "revert":
        return
    if isinstance(command, bool) or not isinstance(command, int):
        raise ActionError(f"{where}: a power must carry the request id as its command")
    if GAME_POWERS.get(entry.get("power")) != entry["action"]:
        raise ActionError(f"{where}: power {entry.get('power')!r} is not a power of {entry['action']}")
    if entry.get("specs") != GAME_SPEC_VERSION:
        raise ActionError(f"{where}: specs must be {GAME_SPEC_VERSION!r}, got {entry.get('specs')!r}")
    if entry.get("cost") != specs[entry["action"]].cost:
        raise ActionError(
            f"{where}: cost {entry.get('cost')!r} is not the table's {specs[entry['action']].cost}"
        )


def validate_entry(entry: Any, cfg: Config, rule_every: int) -> None:
    """Strict check of one log entry (replay never clamps): ``ActionError`` on anything outside the menu.

    Checks the structure, the source and action names, ``step_applied`` (a positive multiple of
    ``rule_every``; any non-negative integer for a record of the player channel, which is applied at the top
    of a step), the exact parameter set, every bound (menu and device ranges), the duration of a timed
    action, and that a patch move carries its previous coordinates and stays within L/4. A record of the
    player channel (``is_player``) is checked against ``GAME_SPECS`` instead of ``MENU``, and carries its
    scope, command, power, table version and cost.
    """
    if not isinstance(entry, Mapping):
        raise ActionError(f"a log entry must be an object, got {type(entry).__name__}")
    step = entry.get("step_applied")
    where = f"entry at step {step!r}"
    player = is_player(entry)
    specs = GAME_SPECS if player else MENU
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        bad = True
    else:
        bad = not player and (step <= 0 or step % rule_every != 0)
    if bad:
        raise ActionError(
            f"{where}: step_applied must be a non-negative integer"
            if player
            else f"{where}: step_applied must be a positive multiple of rule_every {rule_every}"
        )
    action, source, params = entry.get("action"), entry.get("source"), entry.get("params")
    if action not in specs:
        raise ActionError(f"{where}: unknown action {action!r}")
    if source not in (GAME_SOURCES if player else SOURCES):
        raise ActionError(f"{where}: unknown source {source!r}")
    if not isinstance(params, Mapping):
        raise ActionError(f"{where}: params must be an object")
    where = f"{where} ({action})"
    if entry.get("scope") is not None and not player:
        raise ActionError(f"{where}: only a record of the player channel carries a scope")
    if player:
        _check_player_fields(where, entry, specs)
    if source == "revert":
        if entry.get("duration") is not None:
            raise ActionError(f"{where}: a revert has no duration")
        _check_revert(where, action, params, cfg, specs)
        return
    _check_forward(where, action, params, entry.get("prev"), cfg, specs)
    _check_duration(where, action, entry, rule_every, specs)


# --- apply ------------------------------------------------------------------------------------------------


def _put(s: State, **named: Any) -> State:
    """``s`` with the named fields replaced by host values, converted to each field's dtype."""
    return s._replace(**{k: mx.array(np.asarray(v), dtype=getattr(s, k).dtype) for k, v in named.items()})


def _with_gain(s: State, p: int, gain: float) -> State:
    cur = to_numpy(s.env_patch_gain)
    return _put(s, env_patch_gain=np.where(np.arange(cur.size) == p, np.float32(gain), cur))


def _apply_move(s: State, params: Mapping[str, Any], size: float, sph: bool = False) -> State:
    cur = to_numpy(s.patch_pos)
    row = np.arange(len(cur)) == params["patch"]
    if sph:  # the place is [lon, lat] degrees; the state holds the unit vector
        return _put(s, patch_pos=np.where(row[:, None], unit_of(params["x"], params["y"]), cur))
    new = np.where(row[:, None], np.asarray([params["x"], params["y"]], np.float32), cur)
    if not (new >= 0).all() or not (new < size).all():
        raise ActionError(f"patch coordinates {[params['x'], params['y']]} are outside [0, {size})")
    return _put(s, patch_pos=new)


def _apply_immigration(s: State, params: Mapping[str, Any]) -> State:
    total = int(to_numpy(s.env_imm_request)) + int(params["count"])
    if total > IMM_PENDING_CAP:
        raise ActionError(f"{total} pending immigrants exceed the cap {IMM_PENDING_CAP}")
    return _put(s, env_imm_request=total)


def apply(s: State, entry: Mapping[str, Any], cfg: Config) -> State:
    """The state after ``entry`` (forward or revert), checked against the device ranges of step 1.

    The entry must have passed ``clamp`` (live) or ``validate_entry`` (replay). The result keeps the dtype
    and shape of every field; ``ActionError`` if it would leave the env out of range.
    """
    action, p = entry["action"], entry["params"]
    if action == "set_seasons":
        out = _put(s, env_season_amp=p["amp"], env_season_period=p["period"])
    elif action == "fertility_shift" and "gain" not in p:
        out = _apply_move(s, p, cfg.world.size, is_sphere(cfg))
    elif action in ("fertility_shift", "habitat_loss"):
        out = _with_gain(s, int(p["patch"]), p["gain"])
    elif action == "regen_pulse":
        out = _put(s, env_regen_mult=p["mult"])
    elif action == "immigration_pulse":
        out = _apply_immigration(s, p)
    elif action == "catastrophe":
        centre = np.radians(np.asarray(p["center"], np.float64)) if is_sphere(cfg) else p["center"]
        out = _put(s, env_cat_center=centre, env_cat_radius=p["radius"], env_cat_kill_p=p["kill_p"])
    elif action == "drift_change":
        out = _put(s, env_drift_mult=p["mult"])
    else:
        raise ActionError(f"unknown action {action!r}")
    if bool(to_numpy(env_out_of_range(out, cfg))):
        raise ActionError(f"{action} {p} leaves the environment outside its device ranges")
    return out
