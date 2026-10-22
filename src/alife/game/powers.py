"""The player's direct powers: names, previews, plans, influence and the since cards (G3b spec §6, §34.5).

Host module, no mlx. A power is a director action under the game table (``director.actions.GAME_SPECS``,
"game-v0") with a name the player knows it by: Food ``regen_pulse``, Bloom ``fertility_shift``, Blight
``habitat_loss``, Strike ``catastrophe``. The engine side is the player channel of the director
(``Director.queue_player``, ``plan_player``, ``apply_pending``, ``take_player``: spec §6.1); this module is
everything the player sees and sends around it:

- ``request_of`` turns a ``power`` control line into the director's ``PlayerRequest`` (the line's keys, the
  defaults, the nearest patch of a click);
- ``PowerDesk`` is the pacer's helper. It answers ``power preview`` with a ``PLAN`` (the clamped parameters,
  cost, influence after, end step, a code-written sentence, cautions and the precedent line), queues a
  ``power`` after the same checks, reports what became of each request (``applied``, ``refused``,
  ``cancelled``), follows an applied power with its **since card** (one ``PLAN`` per milestone, +250 and
  +1,000 steps and the end of the effect, with the exact metrics rows), writes the player's input log
  (``commands.jsonl``) and answers the ``ENVS`` additions (influence, patches, per-power readiness);
- the **influence** is a fold over the director log (``pacing.fold`` with the player scope): the budget of
  ``PLAYER_BUDGET`` units per ``PLAYER_BUDGET_WINDOW`` steps minus the costs of the player's entries in the
  window, with the steps at which each spent unit comes back.

Every string the player reads is written by code from closed templates, from the exact quantities, and passes
the report's lexicon (``tests/test_powers.py``): no cause, credit, motive, forecast or valuation. The unit of
the budget is called "units" in these sentences ("influence" is in the lexicon's list of causal words).
"""

from __future__ import annotations

import re
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Protocol

import numpy as np

from alife.config import Config, dims
from alife.director import PlayerRequest, actions, pacing
from alife.game import lexicon
from alife.game.precedents import precedent_object

__all__ = [
    "DEFAULT_STEPS",
    "FORBIDDEN",
    "GAME_SPECS",
    "GAME_SPEC_VERSION",
    "INTENT_REFUSAL",
    "LABEL_OF_ACTION",
    "POWERS_VERSION",
    "POWER_OF_ACTION",
    "POWER_SPECS",
    "REFUSED_ELSE",
    "SINCE_AFTER",
    "Out",
    "PowerDesk",
    "PowerError",
    "PowerLineLike",
    "PowerSpec",
    "blocked",
    "column_count",
    "compact_row",
    "header_block",
    "influence",
    "milestones",
    "nearest_patch",
    "patches_block",
    "player_words",
    "power_states",
    "request_of",
    "since_rows",
]

POWERS_VERSION: Final = 1
# The game table lives in ``director.actions`` (the director package must not import ``game``); these are the
# same objects, so a test that patches a row of ``GAME_SPECS`` patches it for the desk as well.
GAME_SPECS: Final = actions.GAME_SPECS
GAME_SPEC_VERSION: Final = actions.GAME_SPEC_VERSION
DEFAULT_STEPS: Final = 6000  # the duration of a timed power whose line names none
SINCE_AFTER: Final = (250, 1000)  # the since card's milestones after the application, besides the end
RING: Final = 512  # metrics rows kept for the since cards (an effect lasts at most 12,000 steps = 48 rows)
PLAN_TEXT: Final = 160
PLAN_CAUTION: Final = 100
INTENT_REFUSAL: Final = "intent is not available in this build"
REFUSED_ELSE: Final = "The request cannot be made now."
FORBIDDEN: Final = (
    *lexicon.NEVER_LISTS,
    *lexicon.LABEL_LISTS,
)  # no report string may match these (spec §5.3)


class PowerError(ValueError):
    """A request the player's words cannot make into a power (the message is for the player)."""


class PowerLineLike(Protocol):
    """A parsed ``power`` line (``play.PowerLine``): ``verb`` ``preview``, ``apply`` or ``cancel``, the
    ``power`` name, the client's ``id`` and the ``params`` pairs in the order given."""

    verb: str
    power: str
    id: int
    params: tuple[tuple[str, int | float], ...]


@dataclass(frozen=True)
class PowerSpec:
    """What the player sees of a power besides the game table's numbers (cost, cooldown, bounds)."""

    name: str
    label: str
    action: str
    scope: str  # global | patch | column
    lever: str | None  # regen | patch | None: what it holds while it runs
    blurb: str
    keys: tuple[str, ...]  # the keys of a ``power`` line it takes (besides ``id``)


POWER_SPECS: Final[Mapping[str, PowerSpec]] = {
    "food": PowerSpec(
        "food", "Food", "regen_pulse", "global", "regen",
        "Raise or lower how fast food regrows everywhere.", ("mult", "steps"),
    ),
    "bloom": PowerSpec(
        "bloom", "Bloom", "fertility_shift", "patch", "patch",
        "Raise the fertility of a food patch.", ("patch", "x", "y", "gain", "steps"),
    ),
    "blight": PowerSpec(
        "blight", "Blight", "habitat_loss", "patch", "patch",
        "Turn a food patch off for a while.", ("patch", "x", "y", "steps"),
    ),
    "strike": PowerSpec(
        "strike", "Strike", "catastrophe", "column", None,
        "Kill part of the organisms in a column.", ("x", "y", "radius", "kill"),
    ),
}  # fmt: skip
LABEL_OF_ACTION: Final = {s.action: s.label for s in POWER_SPECS.values()}
POWER_OF_ACTION: Final = {s.action: s.name for s in POWER_SPECS.values()}
assert {n: s.action for n, s in POWER_SPECS.items()} == dict(actions.GAME_POWERS)


# --- the header -----------------------------------------------------------------------------------------


def header_block(cfg: Config) -> dict[str, Any]:
    """The ``play.powers`` header object: per power its action, label, cost and cooldown from the game table,
    its scope, the bounds of a timed power's duration and of its parameters (the sliders of the window; a
    list of ranges where the table excludes one)."""
    out: dict[str, Any] = {}
    for name, p in POWER_SPECS.items():
        spec = actions.GAME_SPECS[p.action]
        params: dict[str, Any] = {k: [float(lo), float(hi)] for k, (lo, hi) in spec.bounds.items()}
        if "radius" in spec.bounds:
            lo, hi = actions.radius_bounds(cfg, actions.GAME_SPECS)
            params = {**params, "radius": [float(lo), float(hi)]}
        for k, (h0, h1) in spec.excluded.items():
            lo, hi = spec.bounds[k]
            params = {**params, k: [[float(lo), float(h0)], [float(h1), float(hi)]]}
        out = {
            **out,
            name: {
                "action": p.action,
                "label": p.label,
                "cost": spec.cost,
                "cooldown": spec.cooldown,
                "scope": p.scope,
                "duration": list(spec.duration) if spec.duration is not None else None,
                "params": params,
            },
        }
    return out


# --- the request ----------------------------------------------------------------------------------------


def nearest_patch(patch_pos: np.ndarray, x: float, y: float, size: float) -> int:
    """The index of the food patch nearest to ``(x, y)`` on the torus (the lowest index on a tie)."""
    off = patch_pos[:, :2].astype(np.float64) - np.array([x, y])
    off = off - size * np.round(off / size)
    return int(np.argmin((off * off).sum(axis=1)))


def _list(words: Sequence[str]) -> str:
    return words[0] if len(words) == 1 else f"{', '.join(words[:-1])} and {words[-1]}"


def request_of(
    line: PowerLineLike, cfg: Config, patch_pos: np.ndarray
) -> tuple[PlayerRequest, dict[str, Any]]:
    """The director's request for a ``power`` line and the ``raw`` object of its ``PLAN`` (the player's values
    with the defaults filled in and the patch a click resolved to). ``PowerError`` says why not, in the
    player's words: a key the power does not take, a missing value, a patch that does not exist."""
    spec = POWER_SPECS[line.power]
    table = actions.GAME_SPECS[spec.action]
    given = dict(line.params)
    if line.verb == "cancel":
        return PlayerRequest(line.id, spec.name, {}, None, cancel=True), {}
    extra = [k for k in given if k not in spec.keys]
    if extra:
        raise PowerError(f"{spec.label} takes {_list(spec.keys)}, not {extra[0]}.")
    raw: dict[str, Any] = dict(given)
    timed = table.duration is not None
    if timed:
        raw = {**raw, "steps": int(given.get("steps", DEFAULT_STEPS))}
    clamp_raw: dict[str, Any] = {}
    if spec.action == "regen_pulse":
        if "mult" not in given:
            raise PowerError("Food is missing mult, the factor on food regrowth.")
        clamp_raw = {"mult": given["mult"]}
    elif spec.lever == "patch":
        n = len(patch_pos)
        if "patch" in given:
            p = int(given["patch"])
        elif "x" in given and "y" in given:
            p = nearest_patch(patch_pos, float(given["x"]), float(given["y"]), cfg.world.size)
        else:
            raise PowerError(f"{spec.label} is missing a patch: give patch, or x and y for the nearest.")
        if not 0 <= p < n:
            raise PowerError(f"Patch {p} does not exist; patches are numbered 0 to {n - 1}.")
        raw = {**{k: v for k, v in raw.items() if k not in ("x", "y")}, "patch": p}
        clamp_raw = {"patch": p}
        if spec.action == "fertility_shift":
            lo, hi = table.bounds["gain"]
            gain = float(given.get("gain", round((lo + hi) / 2, 4)))
            raw = {**raw, "gain": gain}
            clamp_raw = {**clamp_raw, "gain": gain}
    else:  # strike
        if "x" not in given or "y" not in given:
            raise PowerError("Strike is missing x and y, the centre of the column.")
        r_lo, r_hi = actions.radius_bounds(cfg, actions.GAME_SPECS)
        k_lo, k_hi = table.bounds["kill_p"]
        radius = float(given.get("radius", round((r_lo + r_hi) / 2, 4)))
        kill = float(given.get("kill", round((k_lo + k_hi) / 2, 4)))
        raw = {**raw, "radius": radius, "kill": kill}
        clamp_raw = {"x": given["x"], "y": given["y"], "radius": radius, "kill_p": kill}
    return PlayerRequest(line.id, spec.name, clamp_raw, raw["steps"] if timed else None), raw


def player_words(message: str) -> str:
    """A director error message in the player's words: action names become power names and the dead zone is
    said plainly. Whatever else the engine says stays as it is, with its identifiers replaced."""
    m = re.search(r"mult (\S+) lies in the range (\S+) to (\S+), which is refused", message)
    if m:
        return (
            f"Food x{float(m.group(1)):g} lies between {m.group(2)} and {m.group(3)}, "
            "where it makes no difference worth a cost."
        )
    for action, label in LABEL_OF_ACTION.items():
        message = message.replace(action, label)
    said = message[:1].upper() + message[1:] + ("" if message.endswith(".") else ".")
    # the lexicon is the last word on what a player may be told: an engine message with a causal, modal or
    # evaluative word in it is replaced rather than shown
    return said if not lexicon.scan(said, FORBIDDEN) else REFUSED_ELSE


# --- influence, readiness -------------------------------------------------------------------------------


def influence(entries: Sequence[Mapping[str, Any]], t: int) -> dict[str, Any]:
    """The player's budget at ``t``: units left, the cap, the window, and the steps at which each spent unit
    comes back (sorted)."""
    window, cap = pacing.PLAYER_BUDGET_WINDOW, actions.PLAYER_BUDGET
    mine = pacing.fold(entries, t, "player")
    live = [(s, c) for s, c in mine.spent if t - window < s <= t]
    used = sum(c for _, c in live)
    return {
        "have": max(cap - used, 0),
        "cap": cap,
        "window": window,
        "free_at": sorted(s + window for s, c in live for _ in range(c)),
    }


def _lever_words(key: str) -> str:
    return "the regen lever" if key == "regen" else f"patch {key.split(':', 1)[1]}"


def blocked(
    entries: Sequence[Mapping[str, Any]], power: str, params: Mapping[str, Any], t: int
) -> tuple[int | None, str | None]:
    """``(ready, why)`` for ``power`` with ``params`` at step ``t``: the step it can be used (None when it can
    be now) and the reason in the player's words (None when it can be). The cooldown runs from the end of the
    power's latest effect, the budget is the rolling window, and a lever is held by any active effect."""
    spec = POWER_SPECS[power]
    table = actions.GAME_SPECS[spec.action]
    mine, shared = pacing.fold(entries, t, "player"), pacing.fold(entries, t)
    info = influence(entries, t)
    due = [0]
    why: str | None = None
    end = mine.ends.get(spec.action)
    if end is not None and t < end + table.cooldown:
        due = [*due, end + table.cooldown]
        why = f"{spec.label} is cooling down until step {end + table.cooldown:,}."
    short = table.cost - info["have"]
    if short > 0:
        back = info["free_at"][short - 1]
        due = [*due, back]
        units = f"{table.cost} unit{'s' if table.cost != 1 else ''}"
        why = why or f"{spec.label} costs {units}; {info['have']} left until step {back:,}."
    for key in (
        () if spec.lever == "patch" and "patch" not in params else actions.lever_keys(spec.action, params)
    ):
        eff = shared.active.get(key)
        if eff is not None:
            due = [*due, eff.end]
            why = why or f"{spec.label} is busy: {_lever_words(key)} is held until step {eff.end:,}."
    ready = max(due)
    return (ready if ready > t else None), why


def power_states(entries: Sequence[Mapping[str, Any]], t: int) -> dict[str, dict[str, Any]]:
    """``ENVS.powers``: per power the step it is ready (None: now) and the reason it is not (None: usable).
    A patch power is checked without a target (its lever is the target's: shown when it is aimed)."""
    out: dict[str, dict[str, Any]] = {}
    for name in POWER_SPECS:
        ready, why = blocked(entries, name, {}, t)
        out = {**out, name: {"ready": ready, "busy": why}}
    return out


def patches_block(patch_pos: np.ndarray, gains: np.ndarray, sigma: float) -> list[dict[str, Any]]:
    """``ENVS.patches``: the food patches the powers target, with their current gain."""
    return [
        {
            "i": i,
            "x": round(float(patch_pos[i, 0]), 2),
            "y": round(float(patch_pos[i, 1]), 2),
            "gain": round(float(gains[i]), 4),
            "sigma": round(float(sigma), 4),
        }
        for i in range(len(patch_pos))
    ]


# --- the since card -------------------------------------------------------------------------------------


def compact_row(row: Mapping[str, Any]) -> dict[str, int]:
    """The columns of a since card from a metrics row: the living at the end of the row and the interval's
    starved, killed, born, storm and catastrophe deaths."""
    alive = (row.get("observers") or {}).get("alive")
    return {
        "step": int(row["step"]),
        "alive": int(alive if alive is not None else round(row.get("n_alive") or 0)),
        "starved": int(row.get("deaths_starved", 0)),
        "killed": int(row.get("deaths_killed", 0)),
        "born": int(row.get("births", 0)),
        "storm": int(row.get("deaths_storm", 0)),
        "catastrophe": int(row.get("deaths_catastrophe", 0)),
    }


def milestones(entry: Mapping[str, Any], ended_at: int | None) -> list[int]:
    """The steps a since card follows: +250 and +1,000 after the application and the end of a timed effect
    (``ended_at``, the step of its revert or its cancellation, else ``step_applied + duration``)."""
    o = int(entry["step_applied"])
    ms = {o + a for a in SINCE_AFTER}
    if entry.get("duration") is not None:
        ms = {*ms, ended_at if ended_at is not None else o + int(entry["duration"])}
    return sorted(ms)


def since_rows(
    entry: Mapping[str, Any], ended_at: int | None, rows: Sequence[Mapping[str, int]], t: int
) -> tuple[list[dict[str, int]], int]:
    """``(rows, reached)`` of the card of ``entry`` at step ``t`` from the compact metrics ``rows``
    (ascending): the last row at or before the application (the "before"; the row at its own step is, since
    a power is applied after that step's row), the first row after it (the row whose interval holds the
    application: a strike's catastrophe deaths are in it) and, for each milestone, the first row at or after
    it, none after ``t``. ``reached`` counts the milestones that have a row."""
    o = int(entry["step_applied"])
    before = [r for r in rows if r["step"] <= o][-1:]
    during = [r for r in rows if o < r["step"] <= t][:1]
    out = [dict(r) for r in (*before, *during)]
    seen = {r["step"] for r in out}
    reached = 0
    for m in milestones(entry, ended_at):
        row = next((r for r in rows if m <= r["step"] <= t), None)
        if row is None:
            continue
        reached += 1
        if row["step"] not in seen:
            out = [*out, dict(row)]
            seen = {*seen, row["step"]}
    return sorted(out, key=lambda r: r["step"]), reached


def column_count(
    pos: np.ndarray, alive: np.ndarray, centre: Sequence[float], radius: float, size: float
) -> int:
    """The living organisms inside a column of ``radius`` around ``centre`` (the torus minimum image in x and
    y, at any height): the catastrophe's own test, on the host."""
    off = pos[:, :2].astype(np.float32) - np.asarray(centre, np.float32)
    off = off - np.float32(size) * np.round(off / np.float32(size))
    inside = (off * off).sum(axis=1) <= np.float32(radius) * np.float32(radius)
    return int((alive.astype(bool) & inside).sum())


def raw_of(entry: Mapping[str, Any]) -> dict[str, Any]:
    """The line's values an applied entry stands for (the clamped ones: what a card rebuilt from the log
    has)."""
    p, power = entry["params"], entry["power"]
    steps = {"steps": entry["duration"]} if entry.get("duration") is not None else {}
    if power == "food":
        return {"mult": p["mult"], **steps}
    if power == "bloom":
        return {"patch": p["patch"], "gain": p["gain"], **steps}
    if power == "blight":
        return {"patch": p["patch"], **steps}
    return {"x": p["center"][0], "y": p["center"][1], "radius": p["radius"], "kill": p["kill_p"]}


# --- the desk -------------------------------------------------------------------------------------------


@dataclass
class Out:
    """A ``PLAN`` the pacer is to send (``reply``: the answer to a ``power preview``)."""

    plan: dict[str, Any]
    reply: bool = False


@dataclass
class Card:
    """A since card in progress: the applied entry, the step its effect ended at (a revert or a cancel), the
    strike's counts in the column and how many milestones were already sent."""

    entry: dict[str, Any]
    raw: dict[str, Any]
    ended_at: int | None = None
    disc: dict[str, int] | None = None
    sent: int = 0


class PowerDesk:
    """The pacer's power helper (module docstring). One per played world; ``mode`` is ``live`` (a director
    that takes requests), ``replay`` (the log is played back: powers are refused, the logged ones are
    shown) or ``off`` (no director). The desk reads the director's entries and the metrics rows, never the
    state: the pacer passes the arrays a request needs."""

    def __init__(
        self,
        cfg: Config,
        director: Any,
        *,
        base_rate: float,
        append: Callable[[list[dict[str, Any]]], None],
    ) -> None:
        self.cfg, self.director, self.base_rate, self._append = cfg, director, base_rate, append
        mode = getattr(director, "mode", None)
        self.mode = "live" if mode in ("live", "player") else "replay" if mode == "replay" else "off"
        self.reserved: dict[int, list[dict[str, Any]]] = {}  # command -> entries planned, not yet applied
        self.raws: dict[int, dict[str, Any]] = {}  # command -> the player's values of a queued request
        self.cards: dict[int, Card] = {}
        self.ring: deque[dict[str, int]] = deque(maxlen=RING)
        self.pending_disc: dict[int, int] = {}  # command -> organisms in the column when it was queued

    # --- the world -----------------------------------------------------------------------------------

    @property
    def available(self) -> bool:
        return self.mode == "live"

    def entries(self) -> list[Mapping[str, Any]]:
        """The director's log and the entries of the requests queued and not yet applied."""
        if self.director is None:
            return []
        reserved = [e for es in self.reserved.values() for e in es]
        return [*self.director.entries, *reserved]

    def alive(self) -> int | None:
        return self.ring[-1]["alive"] if self.ring else None

    def header(self) -> dict[str, Any] | None:
        return header_block(self.cfg) if self.available else None

    def env_block(self, t: int, patch_pos: np.ndarray, gains: np.ndarray) -> dict[str, Any]:
        """The G3b keys of ``ENVS`` (influence, patches, powers) at step ``t``; empty for a world that takes
        no power (a replay shows the logged effects only)."""
        if not self.available:
            return {}
        entries = self.entries()
        return {
            "influence": influence(entries, t),
            "patches": patches_block(patch_pos, gains, self.cfg.fields.patch_sigma),
            "powers": power_states(entries, t),
        }

    # --- texts ---------------------------------------------------------------------------------------

    def _minutes(self, steps: int) -> str:
        return f"{steps / self.base_rate / 60:.1f} min at 1x"

    def sentence(self, power: str, entry: Mapping[str, Any], have_after: int) -> str:
        """The code-written sentence of a power: what, how long, the cost and the units left."""
        spec, p = POWER_SPECS[power], entry["params"]
        left = f"cost {entry['cost']}, {have_after} of {actions.PLAYER_BUDGET} units left after."
        d = entry.get("duration")
        span = f" for {d:,} steps ({self._minutes(d)})" if d else ""
        if power == "food":
            what = f"{spec.label} x{p['mult']:g}{span}"
        elif power == "bloom":
            what = f"{spec.label} x{p['gain']:g} on patch {p['patch']}{span}"
        elif power == "blight":
            what = f"{spec.label} on patch {p['patch']}{span}"
        else:
            cx, cy = p["center"]
            what = (
                f"{spec.label} at ({cx:.1f}, {cy:.1f}), radius {p['radius']:g}, kill chance {p['kill_p']:g}"
            )
        return f"{what}, {left}"[:PLAN_TEXT]

    def cautions(
        self,
        power: str,
        entry: Mapping[str, Any],
        *,
        paused: bool,
        count: int | None = None,
        exact: bool = True,
    ) -> list[str]:
        """What the player should know before confirming, each a short code-written clause. A preview or a
        queued request on a world that is running is made at the step it is asked at, and the power lands a
        few steps later: its steps are said "if applied now" (``exact`` False). A paused world applies it at
        the very step, and an applied record is exact."""
        spec, p = POWER_SPECS[power], entry["params"]
        out: list[str] = []
        end = entry["step_applied"] + (entry.get("duration") or 0)
        lead = "" if exact else "if applied now, "
        if spec.lever == "regen":
            out = [*out, f"{lead}holds the regen lever until step {end:,}"]
        elif spec.lever == "patch":
            out = [*out, f"{lead}holds patch {p['patch']} until step {end:,}"]
        alive, slots = self.alive(), dims(self.cfg).n
        if power in ("food", "bloom") and alive:
            out = [
                *out,
                f"population {alive:,} of {slots:,} slots",
            ]  # the slots of the state, no carrying capacity
        if power == "strike" and count is not None:
            deaths = count * float(p["kill_p"])
            share = f" ({100 * deaths / alive:.1f}% of the population)" if alive else ""
            out = [*out, f"{count:,} organisms in the column; {deaths:,.0f} deaths on average{share}"]
        if paused:
            out = [*out, "lands when the world resumes"]
        return [c[:PLAN_CAUTION] for c in out]

    # --- the PLAN object ----------------------------------------------------------------------------

    @staticmethod
    def clamped(raw: Mapping[str, Any], entry: Mapping[str, Any]) -> list[str]:
        """The names (of the line's keys) whose values the engine moved: bounds, rounding of the duration to
        the quantum, the wrap of a coordinate."""
        p = entry["params"]
        mine = {
            "mult": p.get("mult"),
            "gain": p.get("gain"),
            "radius": p.get("radius"),
            "kill": p.get("kill_p"),
            "steps": entry.get("duration"),
        }
        if "center" in p:
            mine = {**mine, "x": p["center"][0], "y": p["center"][1]}
        return [
            key
            for key, value in raw.items()
            if mine.get(key) is not None
            and abs(float(mine[key]) - float(value)) > 1e-5 * max(1.0, abs(float(value)))
        ]

    def plan(
        self,
        command: int,
        status: str,
        power: str,
        *,
        raw: Mapping[str, Any],
        t: int,
        entry: Mapping[str, Any] | None = None,
        entries: Sequence[Mapping[str, Any]] | None = None,
        text: str = "",
        caution: Sequence[str] = (),
        reason: str | None = None,
        ready: int | None = None,
        step_applied: int | None = None,
        ends: int | None = None,
        since: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """A ``PLAN`` (``streamrec.PLAN_KEYS``). ``entry`` is the director entry the plan is about (planned,
        queued or applied), None for a refusal or a cancel; ``entries`` the log the influence is folded from
        at ``t``."""
        spec = POWER_SPECS[power]
        table = actions.GAME_SPECS[spec.action]
        log = self.entries() if entries is None else entries
        params = dict(entry["params"]) if entry is not None else {}
        duration = entry.get("duration") if entry is not None else None
        if ends is None and entry is not None and duration is not None:
            ends = entry["step_applied"] + duration
        if ready is None and entry is not None and status in ("preview", "queued", "applied"):
            ready = (ends if ends is not None else entry["step_applied"]) + table.cooldown
        holds = None
        if entry is not None and spec.lever == "regen":
            holds = "regen"
        elif entry is not None and spec.lever == "patch":
            holds = f"patch:{params['patch']}"
        with_entry = entry is not None and status in ("preview", "queued", "applied", "since")
        return {
            "id": command,
            "kind": "power",
            "status": status,
            "power": power,
            "action": spec.action,
            "raw": dict(raw),
            "params": params,
            "clamped": self.clamped(raw, entry) if entry is not None else [],
            "duration": duration,
            "step_applied": step_applied,
            "ends": ends,
            "cost": table.cost,
            "influence": influence(log, t),
            "ready": ready,
            "holds": holds,
            "text": (text or reason or "")[:PLAN_TEXT],
            "caution": [c[:PLAN_CAUTION] for c in caution],
            "precedent": precedent_object(spec.action, params, spec.label) if with_entry else None,
            "since": dict(since) if since is not None else None,
            "refused": reason if status == "refused" else None,
        }

    def _log(
        self, command: int, t: int, verb: str, status: str, line: str | None = None, **more: Any
    ) -> None:
        """One line of ``commands.jsonl`` (the player's input log; a replay writes none). The last line of an
        id says what became of it: ``preview``, ``queued`` then ``applied``, or ``refused``."""
        if self.mode == "replay":
            return
        extra = {"line": line} if line is not None else {}
        self._append([{"id": command, "t": t, "verb": verb, "status": status, **extra, **more}])

    # --- the lines -----------------------------------------------------------------------------------

    def intent(self, line: str, t: int) -> list[Out]:
        """``intent TEXT`` is reserved (G4): refused, with the shape of a plan row."""
        plan = {
            "id": 0,
            "kind": "intent",
            "status": "refused",
            "power": None,
            "action": None,
            "raw": {},
            "params": {},
            "clamped": [],
            "duration": None,
            "step_applied": None,
            "ends": None,
            "cost": 0,
            "influence": None,
            "ready": None,
            "holds": None,
            "text": INTENT_REFUSAL,
            "caution": [],
            "precedent": None,
            "since": None,
            "refused": INTENT_REFUSAL,
        }
        self._log(0, t, "intent", "refused", line, entry_step=None, reason=INTENT_REFUSAL)
        return [Out(plan, reply=True)]

    def _refused(
        self,
        line: PowerLineLike,
        text: str,
        raw: Mapping[str, Any],
        t: int,
        why: str,
        ready: int | None = None,
    ) -> list[Out]:
        verb = {"preview": "preview", "cancel": "cancel"}.get(line.verb, "power")
        self._log(line.id, t, verb, "refused", text, entry_step=None, reason=why)
        plan = self.plan(line.id, "refused", line.power, raw=raw, t=t, reason=why, ready=ready)
        return [Out(plan, reply=line.verb == "preview")]

    def handle(
        self,
        line: PowerLineLike,
        text: str,
        t: int,
        state: Any,
        *,
        paused: bool,
        patch_pos: np.ndarray,
        positions: Callable[[], tuple[np.ndarray, np.ndarray]],
    ) -> list[Out]:
        """A ``power`` line at the top of iteration ``t`` (the step the power would be applied at; ``text`` is
        the line as received). ``state`` is the borrowed state, which the director's dry run reads;
        ``positions`` reads the arrays of a strike's column. Previews change nothing; a request that passes
        the same checks is queued."""
        if not self.available:
            why = (
                "A replay plays the log back; it takes no new power."
                if self.mode == "replay"
                else "This world has no director, so it takes no power."
            )
            return self._refused(line, text, dict(line.params), t, why)
        spec = POWER_SPECS[line.power]
        try:
            request, raw = request_of(line, self.cfg, patch_pos)
        except PowerError as e:
            return self._refused(line, text, dict(line.params), t, str(e))
        if request.cancel:
            return self._cancel(line, request, text, t, state)
        entries = self.entries()
        probe = {"patch": request.raw["patch"], "gain": 0.0} if spec.lever == "patch" else {}
        ready, why = blocked(entries, line.power, probe, t)
        planned: list[dict[str, Any]] = []
        if why is None:
            try:
                planned = self.director.plan_player(state, request, t)
            except actions.ActionError as e:
                why = player_words(str(e))
        if why is not None:
            return self._refused(line, text, raw, t, why, ready)
        [entry] = planned
        count = None
        if line.power == "strike":
            pos, alive = positions()
            p = entry["params"]
            count = column_count(pos, alive, p["center"], p["radius"], self.cfg.world.size)
        after = [*entries, entry]
        preview = line.verb == "preview"
        plan = self.plan(
            line.id,
            "preview" if preview else "queued",
            line.power,
            raw=raw,
            t=t,
            entry=entry,
            entries=after,
            text=self.sentence(line.power, entry, influence(after, t)["have"]),
            caution=self.cautions(line.power, entry, paused=paused, count=count, exact=paused),
        )
        if preview:
            self._log(line.id, t, "preview", "preview", text, entry_step=None)
            return [Out(plan, reply=True)]
        self.director.queue_player(request)
        self.reserved = {**self.reserved, line.id: [entry]}
        self.raws = {**self.raws, line.id: dict(raw)}
        if count is not None:
            self.pending_disc = {**self.pending_disc, line.id: count}
        self._log(line.id, t, "power", "queued", text, entry_step=t)
        return [Out(plan)]

    def _cancel(
        self, line: PowerLineLike, request: PlayerRequest, text: str, t: int, state: Any
    ) -> list[Out]:
        spec = POWER_SPECS[line.power]
        entries = self.entries()
        running = [
            e
            for e in pacing.fold(entries, t).active.values()
            if e.scope == "player" and e.action == spec.action
        ]
        if not running:
            return self._refused(line, text, {}, t, f"Nothing to cancel: {spec.label} is not running.")
        try:
            planned = self.director.plan_player(state, request, t)
        except actions.ActionError as e:
            return self._refused(line, text, {}, t, player_words(str(e)))
        cooldown = actions.GAME_SPECS[spec.action].cooldown
        ready = max(e.end for e in running) + cooldown  # cancelling does not shorten the cooldown
        self.director.queue_player(request)
        self.reserved = {**self.reserved, line.id: planned}
        self._log(line.id, t, "cancel", "queued", text, entry_step=t)
        sentence = f"{spec.label} stops at step {t:,}. No refund; ready again at step {ready:,}."
        return [
            Out(self.plan(line.id, "queued", line.power, raw={}, t=t, text=sentence, ready=ready, ends=t))
        ]

    # --- after the step -------------------------------------------------------------------------------

    def after_step(
        self,
        applied: Sequence[Mapping[str, Any]],
        refused: Sequence[tuple[PlayerRequest, str]],
        at: int,
    ) -> tuple[list[Out], bool]:
        """What the step that has just run did to the player's requests: the entries applied at its top, step
        ``at`` (powers, and the reverts that ended or cancelled them) and the requests refused when they were
        applied. Returns the ``PLAN`` records to send and whether the environment changed."""
        outs: list[Out] = []
        changed = False
        cancels: dict[int, list[Mapping[str, Any]]] = {}
        for entry in applied:
            changed = True
            if entry["source"] == "player":
                outs = [*outs, self._applied(entry)]
            elif "command" in entry:
                cancels = {**cancels, entry["command"]: [*cancels.get(entry["command"], []), entry]}
                self._ended(entry)
            else:
                self._ended(entry)
        for command, reverts in cancels.items():
            outs = [*outs, self._cancelled(command, reverts)]
        for request, why in refused:
            changed = True  # the units it held are free again
            self.reserved = {k: v for k, v in self.reserved.items() if k != request.command}
            self.raws = {k: v for k, v in self.raws.items() if k != request.command}
            verb = "cancel" if request.cancel else "power"
            reason = player_words(why)
            self._log(request.command, at, verb, "refused", None, entry_step=None, reason=reason)
            plan = self.plan(
                request.command, "refused", request.power, raw=dict(request.raw), t=at, reason=reason
            )
            outs = [*outs, Out(plan)]
        return outs, changed

    def _applied(self, entry: Mapping[str, Any]) -> Out:
        command, power, at = entry["command"], entry["power"], entry["step_applied"]
        raw = self.raws.get(command) or raw_of(entry)
        self.reserved = {k: v for k, v in self.reserved.items() if k != command}
        self.raws = {k: v for k, v in self.raws.items() if k != command}
        disc = self.pending_disc.get(command)
        self.pending_disc = {k: v for k, v in self.pending_disc.items() if k != command}
        self.cards = {
            **self.cards,
            command: Card(dict(entry), dict(raw), disc={"before": disc} if disc is not None else None),
        }
        self._log(command, at, "power", "applied", None, entry_step=at)
        entries = self.entries()
        return Out(self.plan(command, "applied", power, raw=raw, t=at, entry=entry, entries=entries,
                             text=self.sentence(power, entry, influence(entries, at)["have"]),
                             caution=self.cautions(power, entry, paused=False), step_applied=at))  # fmt: skip

    def _ended(self, revert: Mapping[str, Any]) -> None:
        """A revert (the end of an effect, or its cancellation): its card follows to the step it ended at."""
        for card in self.cards.values():
            e = card.entry
            if e["action"] == revert["action"] and e["step_applied"] == revert["reverts"]:
                card.ended_at = int(revert["step_applied"])

    def _cancelled(self, command: int, reverts: Sequence[Mapping[str, Any]]) -> Out:
        at = int(reverts[0]["step_applied"])
        action = reverts[0]["action"]
        power = POWER_OF_ACTION[action]
        spec = POWER_SPECS[power]
        origins = [r["reverts"] for r in reverts]
        started = [
            e
            for e in self.director.entries
            if e["source"] == "player" and e["action"] == action and e["step_applied"] in origins
        ]
        end = max((e["step_applied"] + (e.get("duration") or 0) for e in started), default=at)
        ready = max(end, at) + actions.GAME_SPECS[action].cooldown
        self._log(command, at, "cancel", "applied", None, entry_step=at)
        sentence = f"{spec.label} was stopped at step {at:,}. No refund; ready again at step {ready:,}."
        return Out(
            self.plan(
                command,
                "cancelled",
                power,
                raw={},
                t=at,
                text=sentence,
                ready=ready,
                step_applied=at,
                ends=at,
            )
        )

    # --- the since cards -----------------------------------------------------------------------------

    def wants_disc(self) -> bool:
        """Whether a strike's after-count is wanted (one step after it was applied)."""
        return any(c.disc is not None and "after" not in c.disc for c in self.cards.values())

    def set_disc(self, pos: np.ndarray, alive: np.ndarray) -> None:
        """The organisms in each open strike's column now, one step after it (host arrays the pacer read)."""
        for card in self.cards.values():
            if card.disc is not None and "after" not in card.disc:
                p = card.entry["params"]
                n = column_count(pos, alive, p["center"], p["radius"], self.cfg.world.size)
                card.disc = {**card.disc, "after": n}

    def _since_plan(self, command: int, card: Card, rows: Sequence[Mapping[str, int]], t: int) -> Out:
        e = card.entry
        at = e["step_applied"]
        entries = self.entries()
        since: dict[str, Any] = {"from": at, "rows": [dict(r) for r in rows]}
        if card.disc is not None:
            since = {**since, "disc": dict(card.disc)}
        end = card.ended_at if card.ended_at is not None else None
        return Out(
            self.plan(
                command, "since", e["power"], raw=card.raw, t=t, entry=e, entries=entries, since=since,
                text=self.sentence(e["power"], e, influence(entries, at)["have"]), step_applied=at,
                ends=end if end is not None else None,
            )
        )  # fmt: skip

    def on_row(self, row: Mapping[str, Any]) -> list[Out]:
        """A metrics row has been written: the cards that have reached a milestone get their next ``PLAN``."""
        self.ring.append(compact_row(row))
        t = int(row["step"])
        outs: list[Out] = []
        done: list[int] = []
        for command, card in self.cards.items():
            rows, reached = since_rows(card.entry, card.ended_at, list(self.ring), t)
            if reached > card.sent:
                card.sent = reached
                outs = [*outs, self._since_plan(command, card, rows, t)]
            if reached == len(milestones(card.entry, card.ended_at)):
                done = [*done, command]
        self.cards = {k: v for k, v in self.cards.items() if k not in done}
        return outs

    def open(self, rows: Sequence[Mapping[str, Any]], t: int) -> list[Out]:
        """Open on the files of a world at step ``t`` (a resume, a replay): the ring of metrics rows, and a
        card for each of the player's powers whose since milestones are not all behind it. Returns the
        ``PLAN`` records that put those cards on the client's screen (the application and what the card has
        so far)."""
        self.ring.extend(compact_row(r) for r in rows if r["step"] <= t)
        if self.director is None:
            return []
        entries = [e for e in self.director.entries if e["step_applied"] < t]
        reverts = {
            (e["action"], e["reverts"]): e["step_applied"]
            for e in entries
            if e["source"] == "revert" and "reverts" in e
        }
        outs: list[Out] = []
        for e in entries:
            if e["source"] != "player":
                continue
            ended = reverts.get((e["action"], e["step_applied"]))
            rows_, reached = since_rows(e, ended, list(self.ring), t)
            if reached == len(milestones(e, ended)):
                continue
            card = Card(dict(e), raw_of(e), ended_at=ended, sent=reached)
            self.cards = {**self.cards, e["command"]: card}
            log = [x for x in self.director.entries if x["step_applied"] < t]
            at = e["step_applied"]
            text = self.sentence(e["power"], e, influence(log, at)["have"])
            plan = self.plan(
                e["command"], "applied", e["power"], raw=card.raw, t=at, entry=e, entries=log, text=text,
                step_applied=at,
            )  # fmt: skip
            outs = [*outs, Out(plan)]
            if reached:
                outs = [*outs, self._since_plan(e["command"], card, rows_, t)]
        return outs
