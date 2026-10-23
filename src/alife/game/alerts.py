"""Alerts: what the world wants the player to look at (G3b spec §4, DESIGN §34.3).

An **alert** is a signal with a class, a dedupe key, an importance, a surface, a fly-to point and an expiry;
an event (``EVNT``, ``game.chronicle``) is the log. ``AlertEngine`` is a pure function of the host quantities
it is given (a metrics row, a social record with its kin threads, a storm tick, the director's entries and the
effects it holds, the chapter clock, and, for the ``engine`` class, wall-clock samples handed in by the
caller) plus its own small state (armed flags, last-fired steps, short histories). It writes no ``State``, no
digest-relevant file and reads no clock: ``tests/test_alerts.py`` feeds it synthetic sequences, and
``fold_logs`` feeds it the files of a finished play in the order the live loop calls it.

**Classes** (``CLASSES``): each has a nominal importance, a surface (``alert``: the stack, a toast by
importance, an auto-pause when the class is armed; ``log``: a chronicle row with fly-to, sent as importance
0), the importance from which an armed class pauses (``pause_min``), and whether a measurement backs its
threshold (``calibrated``). Every threshold is a named constant here, pinned by ``constants_digest``,
provisional until the G3b Mac day.

**Identity.** An alert is ``(class, key)``; its ``id`` is ``first_step * 1000 + 600 + index`` and stays the
same for the life of the alert: a change of text or importance is re-sent with the next ``version`` and state
``updated``, the end of its condition sends ``cleared``. A key fires once and re-arms by its class's rule.

**Rate limits.** At most ``CAP_N`` importance-2 alerts of the alert surface per ``CAP_WINDOW`` steps; the rest
are folded into one ``summary`` alert. Exempt: importance 3, every class whose pause is armed, ``chapter`` and
``engine``.

**Provenance.** A population fall within ``PROVENANCE_STEPS`` after a forward habitat loss or a regrowth
pulse of a factor under 1 (the player's or the director's: an entry that takes food away) is importance 1,
never pauses and refers to that entry; ``pop_low`` stays unconditional, and so does a fall as deep as
``FALL_HARD`` (15% under the maximum of the window): at the raise it is importance 3, and a demoted fall that
deepens to it is raised to importance 3 and may pause, the entry still stated. The text states the fall and
the entry's time, never a cause.

**Open.** The engine's state is not saved: a resume re-arms everything, and the first evaluation of a level
class after ``prime_rows`` arms it silently (no spurious ``pop_low`` at open).

Host code: no mlx, no subscript assignment (``tests/test_lint.py``); texts are closed templates checked
against the report lexicon (``tests/test_alerts.py``)."""

from __future__ import annotations

import hashlib
import json
import statistics
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, NamedTuple

import numpy as np

from alife import streamrec

__all__ = [
    "ALERT_VERSION",
    "CAP_N",
    "CAP_WINDOW",
    "CLASSES",
    "EXPIRE_STEPS",
    "Alert",
    "AlertClass",
    "AlertEngine",
    "Taken",
    "backlog",
    "constants_digest",
    "default_autopause",
    "entry_phrase",
    "fold_logs",
    "registry",
]

ALERT_VERSION: Final = 1
EXPIRE_STEPS: Final = 3000  # an alert below importance 3 stays in the stack this long
CAP_N: Final = 4  # importance-2 alerts of the alert surface per window ...
CAP_WINDOW: Final = 1500  # ... of this many steps
MIN_ROW_STEPS: Final = 200  # a metrics row of fewer steps is a partial interval: no rate is read from it
INDEX_MAX: Final = 1000 - streamrec.ALERT_ID_BASE  # ids a step can give out (600 to 999)

# population (level classes: the metrics row's ``observers.alive``, the count at the end of the row)
FALL_ROWS: Final = 6  # the maximum is over these rows before the current one and the current one
FALL_FIRE: Final = 0.08
FALL_HARD: Final = 0.15  # importance 3 from this fall ...
FALL_HARD_ROW: Final = 0.10  # ... or from this fall in one row
FALL_CLEAR: Final = 0.04
REARM_STEPS: Final = 6000  # a fall or surge alert that outlasts this many steps is cleared and re-armed
STARVE_ROWS: Final = 8
STARVE_RATIO: Final = 1.35
LOW_ROWS: Final = 24
LOW_FIRE: Final = 0.5
LOW_CLEAR: Final = 0.6
SLIDE_REFS: Final = 3  # chapter ends compared with
SLIDE_FIRE: Final = 0.80
SLIDE_CLEAR: Final = 0.90
SURGE_FIRE: Final = 0.10
# storms
STRONG_ZETA: Final = 0.70  # zeta_now / f0 of a named storm
STRONG_REARM: Final = 6000
TOLL_MIN: Final = 10
TOLL_RATIO: Final = 3.0
TOLL_ROWS: Final = 8
TOLL_DEDUPE: Final = 1500
# lineages and houses
COLLAPSE_FRAC: Final = 0.6
COLLAPSE_SNAPSHOTS: Final = (
    4  # the maximum is over these snapshots before the current one and the current one
)
COLLAPSE_DEDUPE: Final = 6000
HOUSE_LISTINGS: Final = 8
HOUSE_PEAK: Final = 200
HOUSE_FADE: Final = 0.25
WATCH_MAX: Final = 3
# effects and the director
EFFECT_NOTICE: Final = 500
PROVENANCE_STEPS: Final = 500
PROVENANCE_ACTIONS: Final = ("regen_pulse", "habitat_loss")
# the experimental storm-over-lineage class
OVERLAP_CELLS: Final = 3
OVERLAP_SHARE: Final = 0.25
OVERLAP_RADIUS: Final = 1.5  # in storm radii
OVERLAP_DEDUPE: Final = 3000
# the engine class (seconds, on the injected clock)
LATE_WINDOW_S: Final = 10.0
LATE_STEPS: Final = 3
SLOW_RATIO: Final = 0.9
SLOW_FOR_S: Final = 20.0
SLOW_SETTLE_S: Final = 5.0  # the achieved rate needs its window filled after a (re)start
ENGINE_DEDUPE_S: Final = 300.0
KEEP_FIRED: Final = 4096  # dedupe memory
KEEP_IDS: Final = 512  # alert ids ``ack`` can name


@dataclass(frozen=True)
class AlertClass:
    """One class of the registry (the header's ``play.alerts.classes``): nominal importance, display label,
    default surface, whether its pause is armed by default, the importance from which an armed class pauses,
    and whether a measurement backs its threshold."""

    imp: int
    label: str
    surface: str
    pause: bool
    pause_min: int
    calibrated: bool


CLASSES: Final[Mapping[str, AlertClass]] = {
    "chapter": AlertClass(3, "Chapter end", "alert", True, 3, True),
    "pop_fall": AlertClass(3, "Population fall", "alert", True, 3, True),
    "pop_low": AlertClass(3, "Population low", "alert", True, 3, False),
    "pop_slide": AlertClass(2, "Population slide", "alert", False, 2, False),
    "pop_surge": AlertClass(1, "Population rise", "alert", False, 1, False),
    "storm_named": AlertClass(2, "Storm named", "alert", False, 2, True),
    "storm_strong": AlertClass(1, "Storm at high vorticity", "log", False, 1, True),
    "storm_ended": AlertClass(1, "Storm ended", "log", False, 1, True),
    "storm_toll": AlertClass(1, "Storm deaths", "log", False, 1, True),
    "lineage_formed": AlertClass(2, "Lineage formed", "log", False, 2, True),
    "lineage_ended": AlertClass(2, "Lineage ended", "log", False, 2, True),
    "lineage_collapse": AlertClass(2, "Lineage shrinking", "alert", False, 2, True),
    "house_fading": AlertClass(1, "House fading", "log", False, 1, True),
    "effect_ending": AlertClass(1, "Power ending", "alert", False, 1, True),
    "director_acted": AlertClass(2, "Director move", "alert", False, 2, True),
    "storm_over_lineage": AlertClass(1, "Storm over a lineage", "log", False, 1, False),
    "engine": AlertClass(1, "Play engine", "alert", True, 3, False),
    "summary": AlertClass(1, "More alerts", "alert", False, 4, True),
}
NO_ARMED_PAUSE: Final = ("chapter", "summary")  # classes the ``autopause alert`` table does not list
EXEMPT_FROM_CAP: Final = ("chapter", "engine", "summary")


def registry() -> dict[str, dict[str, Any]]:
    """The classes as ``streamrec.alerts_block`` takes them (the header's ``play.alerts.classes``)."""
    return {
        name: {
            "imp": c.imp,
            "pause": c.pause,
            "label": c.label,
            "calibrated": c.calibrated,
            "surface": c.surface,
        }
        for name, c in CLASSES.items()
    }


def default_autopause() -> dict[str, bool]:
    """The alert classes whose pause the player can switch (``play.autopause.alert``) and their defaults:
    ``pop_fall`` (at importance 3), ``pop_low`` and ``engine`` (at importance 3) on, the rest off. The
    chapter has its own switch."""
    return {name: c.pause for name, c in CLASSES.items() if name not in NO_ARMED_PAUSE}


def constants_digest() -> str:
    """SHA-256 of the version, the class table and every threshold (pinned by a test: a change is
    deliberate and needs an ``ALERT_VERSION`` bump)."""
    consts = [
        EXPIRE_STEPS, CAP_N, CAP_WINDOW, MIN_ROW_STEPS, FALL_ROWS, FALL_FIRE, FALL_HARD, FALL_HARD_ROW,
        FALL_CLEAR, REARM_STEPS, STARVE_ROWS, STARVE_RATIO, LOW_ROWS, LOW_FIRE, LOW_CLEAR, SLIDE_REFS,
        SLIDE_FIRE, SLIDE_CLEAR, SURGE_FIRE, STRONG_ZETA, STRONG_REARM, TOLL_MIN, TOLL_RATIO, TOLL_ROWS,
        TOLL_DEDUPE, COLLAPSE_FRAC, COLLAPSE_SNAPSHOTS, COLLAPSE_DEDUPE, HOUSE_LISTINGS, HOUSE_PEAK,
        HOUSE_FADE, WATCH_MAX, EFFECT_NOTICE, PROVENANCE_STEPS, list(PROVENANCE_ACTIONS), OVERLAP_CELLS,
        OVERLAP_SHARE, OVERLAP_RADIUS, OVERLAP_DEDUPE, LATE_WINDOW_S, LATE_STEPS, SLOW_RATIO, SLOW_FOR_S,
        SLOW_SETTLE_S, ENGINE_DEDUPE_S,
    ]  # fmt: skip
    table = {n: [c.imp, c.label, c.surface, c.pause, c.pause_min, c.calibrated] for n, c in CLASSES.items()}
    blob = json.dumps([ALERT_VERSION, consts, table], separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# --- texts ----------------------------------------------------------------------------------------------

# Closed templates (spec §4.6): the exact quantities of the source rows, "since" and never "because", checked
# against the report lexicon by ``tests/test_alerts.py``. A slot is a count or a step.
TEXTS: Final = {
    "chapter": "The chapter ended at step {step}.",
    "pop_fall": "The population fell from {peak} to {alive} in {steps} steps.",
    "pop_fall_starved": " {starved} starved in the latest {row} steps, against a median of {median}.",
    "pop_fall_entry": " {phrase} has been active since step {since}.",
    "pop_low": "The population is {alive}, {pct}% of the {peak} it held within {span} steps.",
    "pop_slide": (
        "The population is {alive}, {pct}% of the {ref} it had at the highest of the {n} chapter ends before."
    ),
    "pop_surge": "The population rose from {low} to {alive} in {steps} steps.",
    "storm_named": "Storm {name} reached named strength at step {step}.",
    "storm_strong": "Storm {name} is at {zeta} f0 at step {step}.",
    "storm_ended": "{label} dissolved after {age} steps.",
    "storm_toll": "{n} organisms died in storms in the latest {row} steps, against a median of {median}.",
    "house_fading": "House {name} fell to {pct}% of its peak of {peak}.",
    "house_out": "House {name} is no longer among the 16 largest houses.",
    "lineage_collapse": "{name} has {size} members, {pct}% of the {peak} it had within {span} steps.",
    "effect_ending": "{phrase} ends at step {end}.",
    "director_acted": "The director applied {phrase} at step {step}.",
    "storm_over_lineage": "{cells} cells held by {name} lie within {radius} of Storm {storm}.",
    "summary": "{n} more alerts were raised within {span} steps.",
    "engine_late": "The play ran late: {n} steps were late within {sec} seconds.",
    "engine_slow": "The play ran at {achieved} steps a second against {requested} for {sec} seconds.",
    "engine_failure": "The play stopped after an error: {error}",
}
TITLES: Final = {
    "chapter": "Chapter ended",
    "pop_fall": "Population falling",
    "pop_low": "Population low",
    "pop_slide": "Population sliding",
    "pop_surge": "Population rising",
    "storm_named": "Storm {name} named",
    "storm_strong": "Storm {name} at high vorticity",
    "storm_ended": "{label} ended",
    "storm_toll": "Storm deaths",
    "lineage_formed": "{name} formed",
    "lineage_ended": "{name} ended",
    "lineage_collapse": "{name} shrinking",
    "house_fading": "House {name} fading",
    "effect_ending": "Power ending",
    "director_acted": "Director move",
    "storm_over_lineage": "Storm over {name}",
    "summary": "More alerts",
    "engine_late": "Play running late",
    "engine_slow": "Play running slow",
    "engine_failure": "Play stopped",
}
ACTION_WORDS: Final = {
    "regen_pulse": "a regrowth pulse",
    "fertility_shift": "a fertility shift",
    "habitat_loss": "a habitat loss",
    "catastrophe": "a catastrophe",
    "immigration_pulse": "an immigration pulse",
    "set_seasons": "a season change",
    "drift_change": "a drift change",
}
POWER_WORDS: Final = {
    "regen_pulse": "Food",
    "fertility_shift": "Bloom",
    "habitat_loss": "Blight",
    "catastrophe": "Strike",
}  # the player's names for the director actions they use (``actions.GAME_POWERS``)


def _n(value: float) -> str:
    return f"{round(value):,}"


def entry_phrase(entry: Mapping[str, Any]) -> str:
    """A director-log entry as a phrase for an alert: the action (the player's own name for a power) with its
    factor or count, and its duration: ``a regrowth pulse of x0.80 for 10,000 steps``, ``Food x0.80 for 6,000
    steps``."""
    action, params = str(entry["action"]), entry.get("params") or {}
    player = entry.get("source") == "player"
    name = POWER_WORDS.get(action, ACTION_WORDS.get(action, action)) if player else ACTION_WORDS.get(action)
    name = name or action
    factor = params.get("mult", params.get("gain"))
    detail = ""
    if isinstance(factor, int | float) and not isinstance(factor, bool):
        detail = f" x{factor:.2f}" if player else f" of x{factor:.2f}"
    elif isinstance(params.get("count"), int | float):
        detail = f" of {_n(params['count'])}"
    dur = entry.get("duration")
    tail = f" for {_n(dur)} steps" if isinstance(dur, int | float) and dur else ""
    return f"{name}{detail}{tail}"


def _cut(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


# --- the alert ------------------------------------------------------------------------------------------


@dataclass
class Alert:
    """An alert as the engine holds it: the 18 keys of an ``ALRT`` alert (``record``) plus ``nominal`` (its
    importance before the surface rule: a log alert is sent as importance 0)."""

    id: int
    cls: str
    key: str
    first_step: int
    step: int
    version: int
    state: str
    surface: str
    imp: int
    nominal: int
    title: str
    text: str
    pos: tuple[float | None, float | None, float | None]
    ref: dict[str, int | str] | None
    pause: bool
    expires: int | None
    watched: bool
    calibrated: bool

    def record(self) -> dict[str, Any]:
        """The alert as an ``ALRT`` object (``streamrec.check_alert``): a log alert is importance 0."""
        return {
            "id": self.id, "version": self.version, "step": self.step, "class": self.cls, "key": self.key,
            "state": self.state, "surface": self.surface, "imp": 0 if self.surface == "log" else self.imp,
            "title": self.title, "text": self.text, "x": self.pos[0], "y": self.pos[1], "z": self.pos[2],
            "ref": self.ref, "pause": self.pause, "expires": self.expires, "watched": self.watched,
            "calibrated": self.calibrated,
        }  # fmt: skip


class Taken(NamedTuple):
    """What the engine produced since the last ``take``: the ``ALRT`` objects to send, the lines to append to
    ``alerts.jsonl`` (superset: the same objects with ``nominal`` and ``suppressed_by``, plus the
    acknowledgements and the alerts a rate limit held back) and whether an alert asks for an auto-pause."""

    alerts: list[dict[str, Any]]
    log: list[dict[str, Any]]
    pause: bool


def _pos(x: Any, y: Any, z: Any = None) -> tuple[float | None, float | None, float | None]:
    def f(v: Any) -> float | None:
        return None if v is None else round(float(v), 2)

    return (f(x), f(y), f(z))


# --- the engine -----------------------------------------------------------------------------------------


@dataclass
class _Level:
    """The arm/fire/clear state of one level class (key ``world``)."""

    armed: bool = True
    silent: bool = False  # the condition held at open: armed off without an alert
    since: int = 0


@dataclass
class AlertEngine:
    """The alert classes of §4.3 over the host quantities (module docstring). ``armed(cls)`` says whether a
    raised alert of the class pauses the world (the pacer's ``Autopause``); ``chapter`` is the chapter length;
    ``experimental`` enables ``storm_over_lineage``."""

    armed: Callable[[str], bool] = lambda cls: False
    chapter: int = 6000
    experimental: bool = False
    cadence: int = (
        0  # the metrics interval of the world (a full row has this many steps); 0: ``MIN_ROW_STEPS``
    )
    # --- state ---
    live: dict[tuple[str, str], Alert] = field(default_factory=dict)
    fired: dict[tuple[str, str], float] = field(default_factory=dict)  # last fired step (seconds for engine)
    rows: deque[tuple[int, int]] = field(default_factory=lambda: deque(maxlen=LOW_ROWS + 1))
    ends: deque[tuple[int, int]] = field(default_factory=lambda: deque(maxlen=SLIDE_REFS))
    starved: deque[int] = field(default_factory=lambda: deque(maxlen=STARVE_ROWS))
    storm_deaths: deque[int] = field(default_factory=lambda: deque(maxlen=TOLL_ROWS))
    levels: dict[str, _Level] = field(default_factory=dict)
    opened: bool = True  # the next row is the first evaluation after open: level classes arm silently
    named: set[int] = field(default_factory=set)
    watch_threads: list[int] = field(default_factory=list)
    k: int = -1  # snapshot counter
    sizes: dict[int, deque[tuple[int, int]]] = field(default_factory=dict)  # thread id -> (k, size)
    merged_at: dict[int, int] = field(default_factory=dict)  # thread id -> k of a merged row of it
    prev_owner: dict[int, int] = field(default_factory=dict)  # kin id -> thread id at the previous record
    houses: dict[int, dict[str, int]] = field(default_factory=dict)  # id -> {listings, peak, size, out}
    house_names: dict[int, str] = field(default_factory=dict)
    names: dict[int, str] = field(default_factory=dict)  # storm id -> name
    cap_steps: deque[int] = field(default_factory=deque)
    held_steps: deque[int] = field(default_factory=deque)  # steps at which an alert was folded into summary
    index_of: dict[int, int] = field(default_factory=dict)
    ids: dict[int, str] = field(default_factory=dict)
    late: deque[tuple[float, int]] = field(default_factory=deque)
    run_since: float | None = None
    slow_since: float | None = None
    _out: list[dict[str, Any]] = field(default_factory=list)
    _log: list[dict[str, Any]] = field(default_factory=list)
    _pause: bool = False
    step: int = 0

    # --- opening ----------------------------------------------------------------------------------------

    def _min_steps(self) -> int:
        """The steps a row must span to be a full interval: the world's metrics interval."""
        return self.cadence or MIN_ROW_STEPS

    def _whole(self, row: Mapping[str, Any]) -> bool:
        """Whether a row is a sample of the grid: a full interval, or the origin row of a new world (``steps``
        0). The row a stop writes off the grid, and the first one after a resume, span fewer steps: they add
        no sample, no rate and no window, so that a quit and a resume change nothing the engine says."""
        steps = int(row.get("steps", 0))
        return steps == 0 or steps >= self._min_steps()

    def prime_rows(self, rows: Iterable[Mapping[str, Any]]) -> None:
        """Fill the histories from the rows of a resumed world (``metrics.jsonl`` up to the start step),
        silently, and arm the level classes at the first evaluation after this call."""
        for r in rows:
            alive = _alive(r)
            if alive is None or not self._whole(r):
                continue
            step = int(r["step"])
            self.rows.append((step, alive))
            if step % self.chapter == 0 and step > 0:
                self.ends.append((step, alive))
            if int(r.get("steps", 0)) >= self._min_steps():
                self.starved.append(int(r.get("deaths_starved", 0)))
                self.storm_deaths.append(int(r.get("deaths_storm", 0)))
        self.opened = True

    def prime_records(self, records: Iterable[Mapping[str, Any]]) -> None:
        """Fill the house histories from the social records of a resumed world (silently)."""
        for rec in records:
            self._houses(rec, fire=False)

    def prime_storms(self, named: Iterable[int]) -> None:
        """The ids of the storms the tracker holds as named."""
        self.named = set(named)

    # --- taking what was produced -----------------------------------------------------------------------

    def take(self) -> Taken:
        """What was produced since the last call. ``alerts`` holds each id once, in the order of its first
        appearance, with its newest state (an alert raised and updated or cleared at one step is one record:
        the client keeps the newest ``version`` of an id); ``log`` keeps every transition."""
        newest: dict[int, dict[str, Any]] = {}
        for a in self._out:
            newest = {**newest, a["id"]: a}
        out, log, pause = list(newest.values()), self._log, self._pause
        self._out, self._log, self._pause = [], [], False
        return Taken(out, log, pause)

    # --- the machinery ----------------------------------------------------------------------------------

    def _is_armed(self, cls: str) -> bool:
        return bool(self.armed(cls))

    def _next_id(self, step: int) -> int | None:
        n = self.index_of.get(step, 0)
        if n >= INDEX_MAX:
            return None
        self.index_of = {**{s: c for s, c in self.index_of.items() if s >= step - 16}, step: n + 1}
        return step * 1000 + streamrec.ALERT_ID_BASE + n

    def _remember(self, alert_id: int, cls: str) -> None:
        self.ids = {**self.ids, alert_id: cls}
        for old in list(self.ids)[: max(0, len(self.ids) - KEEP_IDS)]:
            self.ids.pop(old)

    def _would_pause(self, cls: str, surface: str, imp: int, pause_ok: bool) -> bool:
        return pause_ok and surface == "alert" and self._is_armed(cls) and imp >= CLASSES[cls].pause_min

    def _line(self, a: Alert, suppressed_by: str | None = None) -> dict[str, Any]:
        return {**a.record(), "nominal": a.nominal, "suppressed_by": suppressed_by}

    @staticmethod
    def _title(cls: str, **slots: Any) -> str:
        return _cut(TITLES[cls].format(**slots), 48)

    def _fire(
        self,
        cls: str,
        key: str,
        step: int,
        *,
        title: str,
        text: str,
        imp: int | None = None,
        surface: str | None = None,
        pos: tuple[float | None, float | None, float | None] = (None, None, None),
        ref: dict[str, int | str] | None = None,
        watched: bool = False,
        pause_ok: bool = True,
        stamp: float | None = None,
    ) -> Alert | None:
        """Raise ``(cls, key)``: None if a rate limit or the id space held it back (it is logged)."""
        spec = CLASSES[cls]
        imp = spec.imp if imp is None else imp
        surface = spec.surface if surface is None else surface
        alert_id = self._next_id(step)
        self.fired = {**self.fired, (cls, key): step if stamp is None else stamp}
        for old in list(self.fired)[: max(0, len(self.fired) - KEEP_FIRED)]:
            self.fired.pop(old)
        if alert_id is None:
            return None
        a = Alert(
            id=alert_id, cls=cls, key=key, first_step=step, step=step, version=1, state="raised",
            surface=surface, imp=imp, nominal=imp, title=_cut(title, 48), text=_cut(text, 160),
            pos=pos, ref=ref, pause=False, expires=None, watched=watched, calibrated=spec.calibrated,
        )  # fmt: skip
        a.pause = self._would_pause(cls, surface, imp, pause_ok)
        if cls == "chapter":
            a.pause = pause_ok and self._chapter_armed()
        a.expires = None if (surface == "alert" and imp >= 3) else step + EXPIRE_STEPS
        # importance 3 never reaches the cap, which counts importance 2 only; the chapter, the engine and
        # every class whose pause is armed are exempt too
        exempt = cls in EXEMPT_FROM_CAP or self._is_armed(cls)
        if surface == "alert" and imp == 2 and not exempt and not self._room(step):
            self._hold(a, step)
            return None
        if surface == "alert" and imp == 2 and not exempt:
            self.cap_steps.append(step)
        self.live = {**self.live, (cls, key): a}
        self._remember(a.id, cls)
        self._out.append(a.record())
        self._log.append(self._line(a))
        if a.pause and cls != "chapter":
            self._pause = True
        return a

    def _chapter_armed(self) -> bool:
        return bool(self.armed("chapter"))

    def _room(self, step: int) -> bool:
        while self.cap_steps and step - self.cap_steps[0] >= CAP_WINDOW:
            self.cap_steps.popleft()
        return len(self.cap_steps) < CAP_N

    def _hold(self, a: Alert, step: int) -> None:
        """A rate limit held ``a`` back: it is logged as suppressed and counted in the ``summary`` alert."""
        self._log.append(self._line(a, "cap"))
        self.held_steps.append(step)
        self._summary(step)

    def _summary(self, step: int) -> None:
        while self.held_steps and step - self.held_steps[0] >= CAP_WINDOW:
            self.held_steps.popleft()
        n = len(self.held_steps)
        live = self.live.get(("summary", "cap"))
        text = TEXTS["summary"].format(n=n, span=_n(CAP_WINDOW))
        if n == 0:
            if live is not None:
                self._clear("summary", "cap", step)
            return
        if live is None:
            self._fire("summary", "cap", step, title=TITLES["summary"], text=text)
        elif live.text != text:
            self._update(live, step, text=text)

    def _update(
        self, a: Alert, step: int, *, imp: int | None = None, text: str | None = None, pause_ok: bool = True
    ) -> None:
        """Re-send ``a`` with the next version (a change of text or importance)."""
        a.version += 1
        a.state, a.step = "updated", step
        if imp is not None:
            a.imp = a.nominal = imp
        if text is not None:
            a.text = _cut(text, 160)
        if not a.pause and self._would_pause(a.cls, a.surface, a.imp, pause_ok):
            a.pause = True
            self._pause = True
        a.expires = None if (a.surface == "alert" and a.imp >= 3) else step + EXPIRE_STEPS
        self._out.append(a.record())
        self._log.append(self._line(a))

    def _clear(self, cls: str, key: str, step: int) -> None:
        """The end of an alert's condition: ``cleared`` (the client removes the row and any pause flag)."""
        a = self.live.get((cls, key))
        if a is None:
            return
        self.live = {k: v for k, v in self.live.items() if k != (cls, key)}
        a.version += 1
        a.state, a.step, a.pause = "cleared", step, False
        self._out.append(a.record())
        self._log.append(self._line(a))

    def _since(self, cls: str, key: str, step: int, within: float) -> bool:
        """Whether ``(cls, key)`` fired less than ``within`` ago (same unit as ``step``)."""
        last = self.fired.get((cls, key))
        return last is not None and step - last < within

    def _housekeeping(self, step: int) -> None:
        self._summary(step)
        old = [
            k
            for k, a in self.live.items()
            if a.expires is not None and step > a.expires + REARM_STEPS and a.cls != "summary"
        ]
        if old:
            self.live = {k: v for k, v in self.live.items() if k not in old}

    def know(self, alerts: Iterable[Mapping[str, Any]]) -> None:
        """Alerts the client was shown that this engine did not raise (the backlog of a resumed world): the
        player can acknowledge them."""
        for a in alerts:
            self._remember(int(a["id"]), str(a["class"]))

    def ack(self, alert_id: int, step: int) -> str | None:
        """The player acknowledged an alert: a line in ``alerts.jsonl`` (so a backlog leaves it out); a note
        for an id the engine does not know."""
        cls = self.ids.get(alert_id)
        if cls is None:
            return f"ack: no alert {alert_id}"
        self._log.append({"id": alert_id, "version": 0, "step": step, "class": cls, "state": "acknowledged"})
        return None

    # --- chapter ----------------------------------------------------------------------------------------

    def on_chapter(self, t: int, *, pause_ok: bool = True) -> None:
        """The chapter ended at ``t`` (the checkpoint hook): an importance-3 alert that the next ``run``
        clears. Its pause is the chapter pause (``autopause chapter``), not an alert pause."""
        self.step = t
        self.clear_chapter(t)  # one the player never cleared (the pause was off) goes with the new one
        self._fire(
            "chapter", str(t), t, title=TITLES["chapter"], text=TEXTS["chapter"].format(step=_n(t)),
            pause_ok=pause_ok,
        )  # fmt: skip

    def clear_chapter(self, t: int) -> None:
        """The world runs again: the chapter alert is acknowledged by the player's ``run``."""
        for cls, key in [k for k in self.live if k[0] == "chapter"]:
            self._clear(cls, key, t)

    # --- the metrics row --------------------------------------------------------------------------------

    def on_row(
        self,
        row: Mapping[str, Any],
        t: int,
        *,
        entries: Sequence[Mapping[str, Any]] = (),
        applied: Sequence[Mapping[str, Any]] = (),
        effects: Sequence[Mapping[str, Any]] = (),
        pause_ok: bool = True,
    ) -> None:
        """A metrics row at ``t``. ``entries`` are the director log's entries (the provenance rule reads the
        recent ones), ``applied`` the entries applied at this boundary, ``effects`` the timed effects held
        (``{action, start, end, source, params}``)."""
        self.step = t
        alive = _alive(row)
        full = int(row.get("steps", 0)) >= self._min_steps()
        if self._whole(row):
            if alive is not None:
                self.rows.append((t, alive))
                self._population(row, t, alive, full, entries, pause_ok)
            self.opened = False
        if full:
            self._toll(row, t, pause_ok)
            self.starved.append(int(row.get("deaths_starved", 0)))
            self.storm_deaths.append(int(row.get("deaths_storm", 0)))
        self._director(applied, t, pause_ok)
        self._effects(effects, t, pause_ok)
        self._housekeeping(t)

    def _level(self, name: str) -> _Level:
        lv = self.levels.get(name)
        if lv is None:
            lv = _Level()
            self.levels = {**self.levels, name: lv}
        return lv

    def _population(
        self,
        row: Mapping[str, Any],
        t: int,
        alive: int,
        full: bool,
        entries: Sequence[Mapping[str, Any]],
        pause_ok: bool,
    ) -> None:
        window = list(self.rows)
        # pop_fall
        recent = window[-(FALL_ROWS + 1) :]
        peak_step, peak = max(recent, key=lambda p: (p[1], p[0]))  # the latest row that held the maximum
        fall = 1.0 - alive / peak if peak > 0 else 0.0
        before = window[-2][1] if len(window) >= 2 else alive
        one_row = 1.0 - alive / before if before > 0 else 0.0
        lv = self._level("pop_fall")
        if not lv.armed and (fall < FALL_CLEAR or t - lv.since >= REARM_STEPS):
            self._clear("pop_fall", "world", t)
            lv.armed, lv.silent = True, False
        if lv.armed and fall >= FALL_FIRE:
            imp = 3 if (fall >= FALL_HARD or one_row >= FALL_HARD_ROW) else 2
            cause = _recent_entry(entries, t)
            if (
                cause is not None and fall < FALL_HARD
            ):  # a fall as deep as a hard one is one whatever the cause
                imp = 1
            lv.armed, lv.since = False, t
            if self.opened:
                lv.silent = True
            else:
                text = self._fall_text(row, t, alive, peak, peak_step, full, cause)
                self._fire(
                    "pop_fall", "world", t, title=TITLES["pop_fall"], text=text, imp=imp,
                    ref=None if cause is None else _power_ref(cause), pause_ok=pause_ok,
                )  # fmt: skip
        elif not lv.armed and not lv.silent:
            live = self.live.get(("pop_fall", "world"))
            if live is not None and live.ref is None:
                cause = _recent_entry(entries, t)
                hard = fall >= FALL_HARD or one_row >= FALL_HARD_ROW
                imp = 1 if cause is not None else 3 if hard else 2
                if imp > live.imp:
                    text = self._fall_text(row, t, alive, peak, peak_step, full, cause)
                    self._update(live, t, imp=imp, text=text, pause_ok=pause_ok)
            elif live is not None and fall >= FALL_HARD and live.imp < 3:
                # a fall that followed an entry is stated, not blamed, and once it is as deep as a hard fall
                # is, it is one whatever came before it: raised to importance 3, the entry still said
                cause = _entry_at(entries, int(live.ref["id"])) if live.ref is not None else None
                text = self._fall_text(row, t, alive, peak, peak_step, full, cause)
                self._update(live, t, imp=3, text=text, pause_ok=pause_ok)
        # pop_low
        low = max(a for _, a in window)
        lv = self._level("pop_low")
        if lv.armed and alive < LOW_FIRE * low:
            lv.armed, lv.since = False, t
            if self.opened:
                lv.silent = True
            else:
                text = TEXTS["pop_low"].format(
                    alive=_n(alive), pct=round(100 * alive / low), peak=_n(low), span=_n(LOW_ROWS * 250)
                )
                self._fire("pop_low", "world", t, title=TITLES["pop_low"], text=text, pause_ok=pause_ok)
        elif not lv.armed and alive >= LOW_CLEAR * low:
            self._clear("pop_low", "world", t)
            lv.armed, lv.silent = True, False
        # pop_surge
        floor = min(a for _, a in window[-(FALL_ROWS + 1) :])
        surge = alive / floor - 1.0 if floor > 0 else 0.0
        lv = self._level("pop_surge")
        if not lv.armed and (surge < FALL_CLEAR or t - lv.since >= REARM_STEPS):
            self._clear("pop_surge", "world", t)
            lv.armed, lv.silent = True, False
        if lv.armed and surge >= SURGE_FIRE:
            lv.armed, lv.since = False, t
            if self.opened:
                lv.silent = True
            else:
                low_step = max(s for s, a in window[-(FALL_ROWS + 1) :] if a == floor)
                text = TEXTS["pop_surge"].format(low=_n(floor), alive=_n(alive), steps=_n(t - low_step))
                self._fire("pop_surge", "world", t, title=TITLES["pop_surge"], text=text, pause_ok=pause_ok)
        # pop_slide: at chapter ends only
        if t > 0 and t % self.chapter == 0:
            self._slide(t, alive, pause_ok)

    def _fall_text(
        self,
        row: Mapping[str, Any],
        t: int,
        alive: int,
        peak: int,
        peak_step: int,
        full: bool,
        cause: Mapping[str, Any] | None,
    ) -> str:
        base = TEXTS["pop_fall"].format(peak=_n(peak), alive=_n(alive), steps=_n(t - peak_step))
        starve = ""
        spans = int(row.get("steps", 0))
        if full and len(self.starved) >= 4:
            median = statistics.median(self.starved)
            now = int(row.get("deaths_starved", 0))
            if median > 0 and now >= STARVE_RATIO * median:
                starve = TEXTS["pop_fall_starved"].format(starved=_n(now), row=_n(spans), median=_n(median))
        entry = ""
        if cause is not None:
            entry = TEXTS["pop_fall_entry"].format(
                phrase=entry_phrase(cause).capitalize(), since=_n(cause["step_applied"])
            )
        for parts in (base + starve + entry, base + entry, base + starve, base):
            if len(parts) <= 160:
                return parts
        return base

    def _slide(self, t: int, alive: int, pause_ok: bool) -> None:
        refs = [a for s, a in self.ends if s < t]
        lv = self._level("pop_slide")
        if refs:
            ref = max(refs)
            ratio = alive / ref if ref > 0 else 1.0
            if not lv.armed and ratio >= SLIDE_CLEAR:
                self._clear("pop_slide", "world", t)
                lv.armed, lv.silent = True, False
            if lv.armed and ratio <= SLIDE_FIRE:
                lv.armed, lv.since = False, t
                if self.opened:
                    lv.silent = True
                else:
                    text = TEXTS["pop_slide"].format(
                        alive=_n(alive), pct=round(100 * ratio), ref=_n(ref), n=len(refs)
                    )
                    self._fire(
                        "pop_slide", "world", t, title=TITLES["pop_slide"], text=text, pause_ok=pause_ok
                    )
        if not self.ends or self.ends[-1][0] != t:
            self.ends.append((t, alive))

    def _toll(self, row: Mapping[str, Any], t: int, pause_ok: bool) -> None:
        deaths = int(row.get("deaths_storm", 0))
        if deaths < TOLL_MIN or self._since("storm_toll", "world", t, TOLL_DEDUPE):
            return
        median = statistics.median(self.storm_deaths) if self.storm_deaths else 0
        if deaths < TOLL_RATIO * median:
            return
        text = TEXTS["storm_toll"].format(n=_n(deaths), row=_n(row.get("steps", 0)), median=_n(median))
        self._fire("storm_toll", "world", t, title=TITLES["storm_toll"], text=text, pause_ok=pause_ok)

    # --- the director and its effects -------------------------------------------------------------------

    def _director(self, applied: Sequence[Mapping[str, Any]], t: int, pause_ok: bool) -> None:
        for e in applied:
            if e.get("source") not in ("rule", "llm"):
                continue
            key = f"{e['step_applied']}:{e['action']}"
            if ("director_acted", key) in self.fired:
                continue
            text = TEXTS["director_acted"].format(phrase=entry_phrase(e), step=_n(e["step_applied"]))
            self._fire(
                "director_acted", key, t, title=TITLES["director_acted"], text=text,
                ref=_power_ref(e), pause_ok=pause_ok,
            )  # fmt: skip

    def _effects(self, effects: Sequence[Mapping[str, Any]], t: int, pause_ok: bool) -> None:
        now: set[str] = set()
        for eff in effects:
            end = int(eff["end"])
            key = f"{eff['action']}:{eff['start']}"
            if 0 < end - t <= EFFECT_NOTICE:
                now.add(key)
                if ("effect_ending", key) not in self.fired:
                    entry = {
                        "action": eff["action"],
                        "params": eff.get("params") or {},
                        "source": eff.get("source"),
                        "duration": None,
                    }
                    text = TEXTS["effect_ending"].format(phrase=entry_phrase(entry).capitalize(), end=_n(end))
                    self._fire(
                        "effect_ending", key, t, title=TITLES["effect_ending"], text=text,
                        ref={"kind": "power", "id": int(eff["start"])}, pause_ok=pause_ok,
                    )  # fmt: skip
        for cls, key in [k for k in self.live if k[0] == "effect_ending" and k[1] not in now]:
            self._clear(cls, key, t)

    # --- storms -----------------------------------------------------------------------------------------

    def on_tick(
        self,
        t: int,
        storms: Sequence[Mapping[str, Any]],
        events: Sequence[Mapping[str, Any]],
        told: Sequence[Mapping[str, Any]],
        *,
        pause_ok: bool = True,
    ) -> None:
        """A storm-tracker tick at ``t``. ``storms`` are the detected tracks (``id``, ``x``, ``y``, ``zeta``
        in f0), ``events`` every tracker event of the tick and ``told`` the ones the chronicle told (its
        persistence rule: a naming at once, an end only for a track that was told)."""
        self.step = t
        for e in events:
            if e["kind"] == "named":
                self.named = self.named | {int(e["id"])}
            if e.get("name"):
                self.names = {**self.names, int(e["id"]): str(e["name"])}
        for e in told:
            ident = int(e["id"])
            if e["kind"] == "named":
                name = str(e.get("name") or ident)
                text = TEXTS["storm_named"].format(name=name, step=_n(t))
                self._fire(
                    "storm_named", str(ident), t, title=self._title("storm_named", name=name), text=text,
                    pos=_pos(e.get("x"), e.get("y")), ref={"kind": "storm", "id": ident}, pause_ok=pause_ok,
                )  # fmt: skip
            elif e["kind"] == "dissolved":
                label = f"Storm {e['name']}" if e.get("name") else f"Disturbance {ident}"
                text = TEXTS["storm_ended"].format(label=label, age=_n(e.get("age", 0)))
                self._fire(
                    "storm_ended", str(ident), t, title=self._title("storm_ended", label=label), text=text,
                    pos=_pos(e.get("x"), e.get("y")), ref={"kind": "storm", "id": ident}, pause_ok=pause_ok,
                )  # fmt: skip
        for s in sorted(storms, key=lambda s: int(s["id"])):
            ident = int(s["id"])
            if ident not in self.named or float(s["zeta"]) < STRONG_ZETA:
                continue
            if self._since("storm_strong", str(ident), t, STRONG_REARM):
                continue
            name = self.names.get(ident, str(ident))
            text = TEXTS["storm_strong"].format(name=name, zeta=f"{float(s['zeta']):.2f}", step=_n(t))
            self._fire(
                "storm_strong", str(ident), t, title=self._title("storm_strong", name=name), text=text,
                pos=_pos(s.get("x"), s.get("y")), ref={"kind": "storm", "id": ident}, pause_ok=pause_ok,
            )  # fmt: skip
        ended = {int(e["id"]) for e in events if e["kind"] in ("dissolved", "merged")}
        self.named = self.named - ended
        self.names = {i: n for i, n in self.names.items() if i not in ended}

    # --- lineages and houses ----------------------------------------------------------------------------

    def watch(self, kin_id: int, threads: Any) -> str | None:
        """Watch the thread that owns kin id ``kin_id`` (at most ``WATCH_MAX``); a note if it cannot."""
        th = threads.thread_of(kin_id)
        if th is None:
            return f"watch: no kin group {kin_id} is listed"
        if th.id in self.watch_threads:
            return None
        if len(self.watch_threads) >= WATCH_MAX:
            return f"watch: {WATCH_MAX} groups are already watched; unwatch one first"
        self.watch_threads = [*self.watch_threads, th.id]
        return None

    def unwatch(self, kin_id: int, threads: Any) -> str | None:
        th = threads.thread_of(kin_id)
        if th is None or th.id not in self.watch_threads:
            return f"unwatch: kin group {kin_id} is not watched"
        self.watch_threads = [t for t in self.watch_threads if t != th.id]
        return None

    def watch_off(self) -> None:
        self.watch_threads = []

    def watched(self, threads: Any, extra: Any = None) -> set[int]:
        """The ids of the watched threads: the ones the player set, or, with none set, the three largest
        (``extra`` is a thread that has just ended and still ranks)."""
        if self.watch_threads:
            return set(self.watch_threads)
        pool = list(threads.open_threads())
        if extra is not None and all(extra.id != t.id for t in pool):
            pool.append(extra)
        ranked = sorted(pool, key=lambda th: (-th.size, th.id))
        return {th.id for th in ranked[:WATCH_MAX]}

    def on_record(
        self,
        rec: Mapping[str, Any],
        told: Sequence[tuple[Mapping[str, Any], str]],
        threads: Any,
        *,
        pause_ok: bool = True,
    ) -> None:
        """A social snapshot record, after the kin threads folded it. ``told`` are the thread events it told
        (``Chronicler.told_threads``: the event dict and its sentence), ``threads`` the ``KinThreads``."""
        t = int(rec["step"])
        self.step = t
        self.k += 1
        k = self.k
        # merged rows: the thread that owned the id at the previous record had it merged away
        events = rec.get("events") or {}
        for kind, fam, gid in zip(
            events.get("kind", ()), events.get("family", ()), events.get("id", ()), strict=False
        ):
            if fam == "kin_group" and kind == "merged" and int(gid) in self.prev_owner:
                self.merged_at = {**self.merged_at, self.prev_owner[int(gid)]: k}
        for th in threads.open_threads():
            if th.last == threads.k:
                hist = self.sizes.get(th.id)
                if hist is None:
                    hist = deque(maxlen=COLLAPSE_SNAPSHOTS + 1)
                    self.sizes = {**self.sizes, th.id: hist}
                hist.append((k, th.size))
        for event, text in told:
            self._lineage(event, text, threads, pause_ok)
        self._collapse(t, k, threads, pause_ok)
        self.prev_owner = dict(threads.owner)
        live_ids = {th.id for th in threads.open_threads()}
        self.sizes = {i: h for i, h in self.sizes.items() if i in live_ids}
        self.merged_at = {i: s for i, s in self.merged_at.items() if i in live_ids}
        self.watch_threads = [i for i in self.watch_threads if i in live_ids]
        self._houses(rec, fire=True, pause_ok=pause_ok)
        self._housekeeping(t)

    def _lineage(self, e: Mapping[str, Any], text: str, threads: Any, pause_ok: bool) -> None:
        t, ident = int(e["step"]), int(e["thread"])
        pos = _pos(*e["pos"])
        ref = {"kind": "kin", "id": int(e["a"])}
        if e["kind"] == "formed":
            if e["imp"] >= 2:
                self._fire(
                    "lineage_formed", str(ident), t, title=self._title("lineage_formed", name=e["name"]),
                    text=text, pos=pos, ref=ref, pause_ok=pause_ok,
                )  # fmt: skip
            return
        if e["imp"] < 2:
            return
        th = threads.threads.get(ident)
        is_watched = ident in self.watched(threads, th)
        self._fire(
            "lineage_ended", str(ident), t, title=self._title("lineage_ended", name=e["name"]), text=text,
            imp=3 if is_watched else 2, surface="alert" if is_watched else "log", pos=pos, ref=ref,
            watched=is_watched, pause_ok=pause_ok,
        )  # fmt: skip

    def _collapse(self, t: int, k: int, threads: Any, pause_ok: bool) -> None:
        for tid in sorted(self.watched(threads)):
            th = threads.threads.get(tid)
            hist = self.sizes.get(tid)
            if th is None or th.ended or hist is None or th.last != threads.k:
                continue
            window = [s for kk, s in hist if k - kk <= COLLAPSE_SNAPSHOTS]
            peak = max(window)
            merged = self.merged_at.get(tid)
            if len(window) < 2 or th.size > COLLAPSE_FRAC * peak:
                continue
            if merged is not None and k - merged <= COLLAPSE_SNAPSHOTS:
                continue  # a merge changes the partition, not the people
            if self._since("lineage_collapse", str(tid), t, COLLAPSE_DEDUPE):
                continue
            text = TEXTS["lineage_collapse"].format(
                name=th.shown, size=_n(th.size), pct=round(100 * th.size / peak), peak=_n(peak),
                span=_n(COLLAPSE_SNAPSHOTS * 250),
            )  # fmt: skip
            self._fire(
                "lineage_collapse", str(tid), t, title=self._title("lineage_collapse", name=th.shown),
                text=text, pos=_pos(*th.pos), ref={"kind": "kin", "id": int(th.ids[0]) if th.ids else 0},
                watched=True, pause_ok=pause_ok,
            )  # fmt: skip

    def _houses(self, rec: Mapping[str, Any], *, fire: bool, pause_ok: bool = True) -> None:
        t = int(rec["step"])
        table = rec.get("houses") or {}
        ids = [int(i) for i in table.get("id", ())]
        listed = dict(zip(ids, zip(table.get("name", ()), table.get("size", ()), strict=True), strict=True))
        book = dict(self.houses)
        for hid, (name, size) in listed.items():
            h = book.get(hid, {"listings": 0, "peak": 0, "size": 0, "out": 0})
            h = {
                "listings": h["listings"] + 1, "peak": max(h["peak"], int(size)), "size": int(size), "out": 0,
            }  # fmt: skip
            book = {**book, hid: h}
            veteran = h["listings"] >= HOUSE_LISTINGS and h["peak"] >= HOUSE_PEAK
            faded = h["size"] <= HOUSE_FADE * h["peak"] and ("house_fading", str(hid)) not in self.fired
            if fire and veteran and faded:
                text = TEXTS["house_fading"].format(
                    name=name, pct=round(100 * h["size"] / h["peak"]), peak=_n(h["peak"])
                )
                self._fire(
                    "house_fading", str(hid), t, title=self._title("house_fading", name=name),
                    text=text, ref={"kind": "house", "id": hid}, pause_ok=pause_ok,
                )  # fmt: skip
        for hid, h in list(book.items()):
            if hid in listed or h["out"]:
                continue
            book = {**book, hid: {**h, "out": 1}}
            if fire and h["listings"] >= HOUSE_LISTINGS and h["peak"] >= HOUSE_PEAK:
                if ("house_fading", str(hid)) in self.fired:
                    continue
                name = self.house_names.get(hid, str(hid))
                text = TEXTS["house_out"].format(name=name)
                self._fire(
                    "house_fading", str(hid), t, title=self._title("house_fading", name=name), text=text,
                    ref={"kind": "house", "id": hid}, pause_ok=pause_ok,
                )  # fmt: skip
        self.house_names = {**self.house_names, **{h: n for h, (n, _) in listed.items()}}
        self.houses = book
        if len(self.houses) > 512:  # the oldest forgotten houses go first
            gone = [h for h, v in self.houses.items() if v["out"]][: len(self.houses) - 512]
            self.houses = {h: v for h, v in self.houses.items() if h not in gone}

    # --- territory (experimental) -----------------------------------------------------------------------

    def on_territory(
        self,
        t: int,
        owner: np.ndarray,
        storms: Sequence[Mapping[str, Any]],
        threads: Any,
        size: float,
        *,
        pause_ok: bool = True,
    ) -> None:
        """``storm_over_lineage`` (experimental, ``calibrated`` false): at least ``OVERLAP_CELLS`` cells and
        ``OVERLAP_SHARE`` of a watched thread's territory cells inside ``OVERLAP_RADIUS`` radii of a named
        storm's centre. A current co-location, not an impact claim. ``owner`` is the ``(ny, nx)`` grid of kin
        ids (-1: none) over a square world of side ``size``."""
        if not self.experimental or not self.named:
            return
        ny, nx = owner.shape
        cx = (np.arange(nx) + 0.5) * size / nx
        cy = (np.arange(ny) + 0.5) * size / ny
        gx, gy = np.meshgrid(cx, cy)
        for tid in sorted(self.watched(threads)):
            th = threads.threads.get(tid)
            if th is None or th.ended or not th.ids:
                continue
            mine = np.isin(owner, np.asarray(th.ids, owner.dtype))
            total = int(mine.sum())
            if total == 0:
                continue
            for s in sorted(storms, key=lambda s: int(s["id"])):
                if int(s["id"]) not in self.named:
                    continue
                dx = np.abs(gx - float(s["x"]))
                dy = np.abs(gy - float(s["y"]))
                dx, dy = np.minimum(dx, size - dx), np.minimum(dy, size - dy)
                reach = OVERLAP_RADIUS * float(s["r"])
                cells = int((mine & (np.hypot(dx, dy) <= reach)).sum())
                key = f"{tid}:{int(s['id'])}"
                if cells >= OVERLAP_CELLS and cells >= OVERLAP_SHARE * total:
                    if self._since("storm_over_lineage", key, t, OVERLAP_DEDUPE):
                        continue
                    sname = self.names.get(int(s["id"]), str(int(s["id"])))
                    text = TEXTS["storm_over_lineage"].format(
                        cells=cells, name=th.shown, radius=f"{reach:.1f}", storm=sname
                    )
                    self._fire(
                        "storm_over_lineage", key, t, title=self._title("storm_over_lineage", name=th.shown),
                        text=text, pos=_pos(s["x"], s["y"]), ref={"kind": "storm", "id": int(s["id"])},
                        watched=True, pause_ok=pause_ok,
                    )  # fmt: skip

    # --- the play engine (wall clock) -------------------------------------------------------------------

    def engine_check(
        self,
        now: float,
        step: int,
        *,
        late_steps: int,
        achieved: float,
        requested: float,
        running: bool,
    ) -> None:
        """A sample of the play loop on the injected clock: ``late_steps`` rising by ``LATE_STEPS`` within
        ``LATE_WINDOW_S``, or the achieved rate below ``SLOW_RATIO`` of the request for ``SLOW_FOR_S`` while
        running. Wall-time based, so a replay writes these to ``alerts.replay.jsonl``."""
        self.step = step
        while self.late and now - self.late[0][0] > LATE_WINDOW_S:
            self.late.popleft()
        self.late.append((now, late_steps))
        rise = late_steps - min(n for _, n in self.late)
        if rise >= LATE_STEPS:
            self._engine_fire("late", now, step, TEXTS["engine_late"].format(n=rise, sec=_n(LATE_WINDOW_S)))
        finite = running and 0 < requested < float("inf")
        if not finite:
            self.run_since = self.slow_since = None
            return
        if self.run_since is None:
            self.run_since = now
        if now - self.run_since < SLOW_SETTLE_S or achieved >= SLOW_RATIO * requested:
            self.slow_since = None
            return
        if self.slow_since is None:
            self.slow_since = now
        elif now - self.slow_since >= SLOW_FOR_S:
            text = TEXTS["engine_slow"].format(
                achieved=f"{achieved:.1f}", requested=f"{requested:.1f}", sec=_n(SLOW_FOR_S)
            )
            self._engine_fire("slow", now, step, text)

    def _engine_fire(self, key: str, now: float, step: int, text: str, *, imp: int = 1) -> None:
        last = self.fired.get(("engine", key))
        if last is not None and now - last < ENGINE_DEDUPE_S:
            return
        title = TITLES[f"engine_{key}"]
        live = self.live.get(("engine", key))
        if live is not None:
            self._clear("engine", key, step)
        self._fire("engine", key, step, title=title, text=text, imp=imp, stamp=now)

    def failure(self, step: int, error: str) -> None:
        """The presenter failed (``_fail``): an importance-3 alert; no pause (the play is stopping)."""
        self.step = step
        text = TEXTS["engine_failure"].format(error=_cut(" ".join(error.split()), 100))
        self._fire(
            "engine", "failure", step, title=TITLES["engine_failure"], text=text, imp=3, pause_ok=False
        )


# --- helpers --------------------------------------------------------------------------------------------


def _alive(row: Mapping[str, Any]) -> int | None:
    """The population of a metrics row: ``observers.alive`` (the count at the end of the row; ``n_alive`` is
    the row's mean, a row late), else the rounded mean of a world without observers."""
    obs = row.get("observers")
    if isinstance(obs, Mapping) and isinstance(obs.get("alive"), int | float):
        return int(obs["alive"])
    n = row.get("n_alive")
    return round(n) if isinstance(n, int | float) else None


def _lowers_food(e: Mapping[str, Any]) -> bool:
    """Whether a forward regrowth or habitat entry takes food away: a habitat loss always, a regrowth pulse
    when its factor is under 1 (a pulse that raises it is no provenance for a fall)."""
    if e.get("action") == "habitat_loss":
        return True
    mult = (e.get("params") or {}).get("mult")
    return isinstance(mult, int | float) and not isinstance(mult, bool) and mult < 1.0


def _recent_entry(entries: Sequence[Mapping[str, Any]], t: int) -> Mapping[str, Any] | None:
    """The latest forward entry that takes food away (``_lowers_food``) applied within ``PROVENANCE_STEPS``
    before ``t``."""
    for e in reversed(entries):
        step = e.get("step_applied")
        if not isinstance(step, int):
            continue
        if step < t - PROVENANCE_STEPS:
            break
        if (
            step <= t
            and e.get("source") != "revert"
            and e.get("action") in PROVENANCE_ACTIONS
            and _lowers_food(e)
        ):
            return e
    return None


def _entry_at(entries: Sequence[Mapping[str, Any]], step: int) -> Mapping[str, Any] | None:
    """The forward provenance entry applied at ``step`` (the one a demoted fall refers to)."""
    return next(
        (
            e
            for e in reversed(entries)
            if e.get("step_applied") == step
            and e.get("source") != "revert"
            and e.get("action") in PROVENANCE_ACTIONS
        ),
        None,
    )


def _power_ref(entry: Mapping[str, Any]) -> dict[str, int | str]:
    return {"kind": "power", "id": int(entry["step_applied"])}


# --- the backlog of a resumed world ---------------------------------------------------------------------


def backlog(
    lines: Iterable[Mapping[str, Any]], t: int, *, limit: int = streamrec.MAX_ALERTS
) -> list[dict[str, Any]]:
    """The alerts a client opening a world at step ``t`` is shown (spec §4.5): the newest state of each alert
    of ``alerts.jsonl`` that is not cleared, acknowledged, held back, below importance 3 only and not
    expired, as ``ALRT`` objects with no pause, the newest ``limit`` by id. Importance-3 alerts (the chapter's
    among them: they do not expire) are left out: their clearing belongs to a session that is over."""
    newest: dict[int, Mapping[str, Any]] = {}
    done: set[int] = set()
    for line in lines:
        alert_id = line.get("id")
        if not isinstance(alert_id, int):
            continue
        if line.get("state") == "acknowledged":
            done.add(alert_id)
        elif line.get("suppressed_by") is None and line.get("version", 0) >= newest.get(alert_id, {}).get(
            "version", 0
        ):
            newest = {**newest, alert_id: line}
    out = []
    for alert_id in sorted(newest):
        a = newest[alert_id]
        expires = a.get("expires")
        if alert_id in done or a["state"] == "cleared" or expires is None or expires <= t:
            continue
        out.append({**{k: a[k] for k in streamrec.ALERT_KEYS}, "pause": False})
    return out[-limit:]


# --- the engine over the files of a finished play --------------------------------------------------------


def fold_logs(
    metrics: Sequence[Mapping[str, Any]],
    social: Sequence[Mapping[str, Any]],
    storms: Sequence[Mapping[str, Any]],
    director: Sequence[Mapping[str, Any]] = (),
    *,
    house_salt: int = 0,
    chapter: int = 6000,
    armed: Callable[[str], bool] = lambda cls: False,
    watch: Sequence[int] = (),
    experimental: bool = False,
    limit: int | None = None,
    cadence: int | None = None,
) -> tuple[list[dict[str, Any]], AlertEngine]:
    """The alerts a play of these logs raises, in the order the live loop calls the engine at one step:
    the social record, the storm tick, the chapter end, then the metrics row. Returns the ``alerts.jsonl``
    lines and the engine. The first row is the first evaluation (level classes arm silently); ``watch`` are
    kin ids to watch from the first record on. ``cadence`` is the metrics interval of the world (the live
    engine is told it at bind); by default the longest row of ``metrics``."""
    from alife.game.chronicle import Chronicler  # imported here: the chronicle needs this module's neighbours

    chron = Chronicler(house_salt)
    if cadence is None:
        cadence = max((int(r.get("steps", 0)) for r in metrics), default=0)
    eng = AlertEngine(armed=armed, chapter=chapter, experimental=experimental, cadence=cadence)
    by_step: dict[int, dict[str, list[Any]]] = {}
    for kind, items in (("social", social), ("storms", storms), ("rows", metrics)):
        for it in items:
            by_step.setdefault(int(it["step"]), {}).setdefault(kind, []).append(it)
    log: list[dict[str, Any]] = []
    entries: list[Mapping[str, Any]] = []
    watched_set = False
    for t in sorted(by_step):
        if limit is not None and t > limit:
            break
        at = by_step[t]
        for rec in at.get("social", ()):
            chron.from_record(rec)
            if watch and not watched_set:
                for gid in watch:
                    eng.watch(gid, chron.threads)
                watched_set = True
            eng.on_record(rec, chron.told_threads, chron.threads)
        for line in at.get("storms", ()):
            chron.from_storms(t, line.get("events", ()), [s["id"] for s in line.get("storms", ())])
            eng.on_tick(t, line.get("storms", ()), line.get("events", ()), chron.told_storm)
        if t > 0 and t % chapter == 0:
            eng.on_chapter(t)
        applied = [e for e in director if e.get("step_applied") == t]
        entries = [*entries, *applied]
        for row in at.get("rows", ()):
            eng.on_row(row, t, entries=entries, applied=applied)
        log += eng.take().log
    return log, eng
