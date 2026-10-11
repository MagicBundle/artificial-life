"""The director's intervention menu: bounds, clamping, strict validation and application (DIRECTOR.md §5).

An *entry* is the log record of one intervention. Its ``params`` are explicit, absolute and already clamped (a
patch index and its new gain, absolute patch coordinates, the catastrophe centre), so applying an entry never
consults the rules or an RNG. ``clamp`` (live mode) turns a rule's raw request into ``params`` / ``prev`` /
``duration``; ``validate_entry`` (replay mode) rejects whatever is outside the menu instead of clamping it;
``apply`` writes a validated entry into ``State.env`` (and ``patch_pos`` for a patch move) with ``_replace``,
keeping every array's shape and dtype so the compiled step is not retraced.

Host-only: nothing here is traced, and it only reads the few ``env_*`` scalars and two small patch arrays.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final, NamedTuple

import mlx.core as mx
import numpy as np

from alife import ops
from alife.config import Config, ConfigError
from alife.invariants import (
    ENV_DRIFT_MULT,
    ENV_PATCH_GAIN,
    ENV_REGEN_MULT,
    ENV_SEASON_AMP,
    env_out_of_range,
)
from alife.state import State, to_numpy

SOURCES: Final = ("rule", "llm", "revert")
IMM_PENDING_CAP: Final = 1024  # the most immigrants the director may leave pending in ``env_imm_request``
MOVE_MAX_FRAC: Final = 0.25  # a patch moves by at most L/4
CATASTROPHE_RADIUS: Final = (20.0, 200.0)  # the upper end is also capped by L/2
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
_REVERT_PARAMS: Final = {"habitat_loss": ("patch", "gain"), "regen_pulse": ("mult",)}


class EnvView(NamedTuple):
    """The host copy of the environment fields the menu reads."""

    regen_mult: float
    season_amp: float
    season_period: int
    drift_mult: float
    imm_request: int
    patch_gain: np.ndarray  # (PP,) f32
    patch_pos: np.ndarray  # (PP, 2) f32


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


def _wrap(x: list[float], size: float) -> list[float]:
    """Wrap a coordinate pair into ``[0, size)`` with ``ops.wrap``'s float32 guard (strictly below)."""
    return [float(v) for v in to_numpy(ops.wrap(mx.array(x, dtype=mx.float32), size))]


def _device(value: float, bounds: tuple[float, float]) -> float:
    """Clip to a step-1 device range (``invariants.ENV_*``); the menu bounds already lie inside them."""
    return _clip(value, *bounds)


def radius_bounds(cfg: Config) -> tuple[float, float]:
    hi = min(CATASTROPHE_RADIUS[1], cfg.world.size / 2)
    return min(CATASTROPHE_RADIUS[0], hi), hi


# --- live: clamp ------------------------------------------------------------------------------------------


def _bounds_of(action: str) -> tuple[int, int]:
    bounds = MENU[action].duration
    if bounds is None:
        raise ActionError(f"{action} is not a timed action")
    return bounds


def _clamp_set_seasons(raw: Mapping[str, Any], view: EnvView, cfg: Config) -> Clamped:
    lo, hi = MENU["set_seasons"].bounds["period"]
    period = int(min(max(round(_number(raw, "period")), lo), hi))
    amp = _device(_clip(_number(raw, "amp"), *MENU["set_seasons"].bounds["amp"]), ENV_SEASON_AMP)
    return Clamped(
        {"period": period, "amp": amp}, {"period": view.season_period, "amp": view.season_amp}, None
    )


def _clamp_fertility_shift(raw: Mapping[str, Any], view: EnvView, cfg: Config) -> Clamped:
    p = _patch(raw, view)
    if "gain" in raw:
        gain = _device(_clip(_number(raw, "gain"), *MENU["fertility_shift"].bounds["gain"]), ENV_PATCH_GAIN)
        return Clamped({"patch": p, "gain": gain}, {"patch": p, "gain": float(view.patch_gain[p])}, None)
    size = cfg.world.size
    dx, dy = _number(raw, "dx"), _number(raw, "dy")
    cap = MOVE_MAX_FRAC * size
    norm = math.hypot(dx, dy)
    scale = cap / norm if norm > cap else 1.0
    old = [float(v) for v in view.patch_pos[p]]
    new = _wrap([old[0] + dx * scale, old[1] + dy * scale], size)
    return Clamped({"patch": p, "x": new[0], "y": new[1]}, {"patch": p, "x": old[0], "y": old[1]}, None)


def _duration(bounds: tuple[int, int], raw: int | float | None, rule_every: int) -> int:
    if raw is None or isinstance(raw, bool) or not math.isfinite(raw):
        raise ActionError(f"a timed action needs a finite duration, got {raw!r}")
    lo, hi = bounds
    return round_up(int(min(max(round(raw), lo), hi)), rule_every)


def _clamp_habitat_loss(raw: Mapping[str, Any], view: EnvView, cfg: Config, duration: int | None) -> Clamped:
    p = _patch(raw, view)
    d = _duration(_bounds_of("habitat_loss"), duration, cfg.director.rule_every)
    return Clamped({"patch": p, "gain": 0.0}, {"patch": p, "gain": float(view.patch_gain[p])}, d)


def _clamp_regen_pulse(
    raw: Mapping[str, Any],
    view: EnvView,
    cfg: Config,
    duration: int | None,
    restore: Mapping[str, Any] | None,
) -> Clamped:
    mult = _device(_clip(_number(raw, "mult"), *MENU["regen_pulse"].bounds["mult"]), ENV_REGEN_MULT)
    d = _duration(_bounds_of("regen_pulse"), duration, cfg.director.rule_every)
    prev = dict(restore) if restore is not None else {"mult": view.regen_mult}  # superseding keeps the base
    return Clamped({"mult": mult}, prev, d)


def _clamp_immigration_pulse(raw: Mapping[str, Any], view: EnvView, cfg: Config) -> Clamped:
    lo, hi = MENU["immigration_pulse"].bounds["count"]
    count = int(min(max(round(_number(raw, "count")), lo), hi, IMM_PENDING_CAP - view.imm_request))
    if count < lo:
        raise ActionError(f"{view.imm_request} immigrants are already pending (cap {IMM_PENDING_CAP})")
    return Clamped({"count": count}, {"pending": view.imm_request}, None)


def _clamp_catastrophe(raw: Mapping[str, Any], view: EnvView, cfg: Config) -> Clamped:
    centre = _wrap([_number(raw, "x"), _number(raw, "y")], cfg.world.size)
    radius = _clip(_number(raw, "radius"), *radius_bounds(cfg))
    kill_p = _clip(_number(raw, "kill_p"), *MENU["catastrophe"].bounds["kill_p"])
    return Clamped({"center": centre, "radius": radius, "kill_p": kill_p}, {}, None)


def _clamp_drift_change(raw: Mapping[str, Any], view: EnvView, cfg: Config) -> Clamped:
    mult = _device(_clip(_number(raw, "mult"), *MENU["drift_change"].bounds["mult"]), ENV_DRIFT_MULT)
    return Clamped({"mult": mult}, {"mult": view.drift_mult}, None)


def clamp(
    action: str,
    raw: Mapping[str, Any],
    duration: int | float | None,
    view: EnvView,
    cfg: Config,
    restore: Mapping[str, Any] | None = None,
) -> Clamped:
    """Clamp a rule's raw request to the menu bounds and the device ranges (live mode).

    ``raw`` carries the request in its natural form (a move as ``dx, dy``, a catastrophe as ``x, y``); the
    result holds the explicit values that are logged and applied. A timed action's ``duration`` is clamped and
    rounded up to a multiple of ``director.rule_every``. ``restore`` (regen pulses only) is the value to
    return to when the new pulse supersedes an active one. Raises ``ActionError`` for an unknown action or an
    unusable request (missing or non-finite values, a patch that does not exist, a full immigration queue).
    """
    if action not in MENU:
        raise ActionError(f"unknown action {action!r}")
    if action == "habitat_loss":
        return _clamp_habitat_loss(raw, view, cfg, duration)
    if action == "regen_pulse":
        return _clamp_regen_pulse(raw, view, cfg, duration, restore)
    return _CLAMPERS[action](raw, view, cfg)


_CLAMPERS: Final = {
    "set_seasons": _clamp_set_seasons,
    "fertility_shift": _clamp_fertility_shift,
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


def _check_move(where: str, params: Mapping[str, Any], prev: Any, cfg: Config) -> None:
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


def _check_forward(where: str, action: str, params: Mapping[str, Any], prev: Any, cfg: Config) -> None:
    bounds = MENU[action].bounds
    if action == "set_seasons":
        _check_keys(where, params, {"period", "amp"})
        _in_range(where, "period", params["period"], *bounds["period"], integer=True)
        _in_range(where, "amp", params["amp"], *bounds["amp"])
    elif action == "fertility_shift" and "gain" in params:
        _check_keys(where, params, {"patch", "gain"})
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
        _in_range(where, "mult", params["mult"], *bounds["mult"])
    elif action == "immigration_pulse":
        _check_keys(where, params, {"count"})
        _in_range(where, "count", params["count"], *bounds["count"], integer=True)
    elif action == "catastrophe":
        _check_keys(where, params, {"center", "radius", "kill_p"})
        c = params["center"]
        if not isinstance(c, list | tuple) or len(c) != 2:
            raise ActionError(f"{where}: center must be [x, y], got {c!r}")
        for v in c:
            if _in_range(where, "center", v, 0, cfg.world.size) >= cfg.world.size:
                raise ActionError(f"{where}: center {c!r} is not strictly inside the world")
        _in_range(where, "radius", params["radius"], *radius_bounds(cfg))
        _in_range(where, "kill_p", params["kill_p"], *bounds["kill_p"])
    elif action == "drift_change":
        _check_keys(where, params, {"mult"})
        _in_range(where, "mult", params["mult"], *bounds["mult"])


def _check_revert(where: str, action: str, params: Mapping[str, Any], cfg: Config) -> None:
    names = _REVERT_PARAMS.get(action)
    if names is None:
        raise ActionError(f"{where}: {action!r} has no timed effect to revert")
    _check_keys(where, params, set(names))
    if action == "habitat_loss":
        _check_patch(where, params, cfg)
        _in_range(where, "gain", params["gain"], *ENV_PATCH_GAIN)
    else:
        _in_range(where, "mult", params["mult"], *ENV_REGEN_MULT)


def _check_duration(where: str, action: str, entry: Mapping[str, Any], rule_every: int) -> None:
    spec, d = MENU[action], entry.get("duration")
    if spec.duration is None:
        if d is not None:
            raise ActionError(f"{where}: {action} takes no duration, got {d!r}")
        return
    lo, hi = spec.duration
    _in_range(where, "duration", d, lo, round_up(hi, rule_every), integer=True)
    if d % rule_every:
        raise ActionError(f"{where}: duration {d} is not a multiple of rule_every {rule_every}")


def validate_entry(entry: Any, cfg: Config, rule_every: int) -> None:
    """Strict check of one log entry (replay never clamps): ``ActionError`` on anything outside the menu.

    Checks the structure, the source and action names, ``step_applied`` (a positive multiple of
    ``rule_every``), the exact parameter set, every bound (menu and device ranges), the duration of a timed
    action, and that a patch move carries its previous coordinates and stays within L/4.
    """
    if not isinstance(entry, Mapping):
        raise ActionError(f"a log entry must be an object, got {type(entry).__name__}")
    step = entry.get("step_applied")
    where = f"entry at step {step!r}"
    if isinstance(step, bool) or not isinstance(step, int) or step <= 0 or step % rule_every:
        raise ActionError(f"{where}: step_applied must be a positive multiple of rule_every {rule_every}")
    action, source, params = entry.get("action"), entry.get("source"), entry.get("params")
    if action not in MENU:
        raise ActionError(f"{where}: unknown action {action!r}")
    if source not in SOURCES:
        raise ActionError(f"{where}: unknown source {source!r}")
    if not isinstance(params, Mapping):
        raise ActionError(f"{where}: params must be an object")
    where = f"{where} ({action})"
    if source == "revert":
        if entry.get("duration") is not None:
            raise ActionError(f"{where}: a revert has no duration")
        _check_revert(where, action, params, cfg)
        return
    _check_forward(where, action, params, entry.get("prev"), cfg)
    _check_duration(where, action, entry, rule_every)


# --- apply ------------------------------------------------------------------------------------------------


def _put(s: State, **named: Any) -> State:
    """``s`` with the named fields replaced by host values, converted to each field's dtype."""
    return s._replace(**{k: mx.array(np.asarray(v), dtype=getattr(s, k).dtype) for k, v in named.items()})


def _with_gain(s: State, p: int, gain: float) -> State:
    cur = to_numpy(s.env_patch_gain)
    return _put(s, env_patch_gain=np.where(np.arange(cur.size) == p, np.float32(gain), cur))


def _apply_move(s: State, params: Mapping[str, Any], size: float) -> State:
    cur = to_numpy(s.patch_pos)
    row = np.arange(len(cur)) == params["patch"]
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
        out = _apply_move(s, p, cfg.world.size)
    elif action in ("fertility_shift", "habitat_loss"):
        out = _with_gain(s, int(p["patch"]), p["gain"])
    elif action == "regen_pulse":
        out = _put(s, env_regen_mult=p["mult"])
    elif action == "immigration_pulse":
        out = _apply_immigration(s, p)
    elif action == "catastrophe":
        out = _put(s, env_cat_center=p["center"], env_cat_radius=p["radius"], env_cat_kill_p=p["kill_p"])
    elif action == "drift_change":
        out = _put(s, env_drift_mult=p["mult"])
    else:
        raise ActionError(f"unknown action {action!r}")
    if bool(to_numpy(env_out_of_range(out, cfg))):
        raise ActionError(f"{action} {p} leaves the environment outside its device ranges")
    return out
