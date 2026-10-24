"""``alife play``: the scheduler, the control grammar and the presenter of a played world (DESIGN §33.2).

Host module. A played world is an ordinary run (``runner.run``) with a ``PlayPacer`` hooked into its loop. The
pacer does two things and nothing else:

- **Schedule.** ``Scheduler`` decides when the next step may start (pause, ``step N`` batches, a rate of 0.25x
  to 2x of the base rate, or ``max``). It reads a ``Clock`` only to decide *when*; the step index, the order
  of the observer and director calls and every written log line are functions of the step count, so pacing
  never reaches the trajectory.
- **Present.** After a step it writes the records of §33.1 (``FRMA``, ``IDNT``, ``INTN``, ``TERR``, ``MAP1``,
  ``STRM``, ``EVNT``, ``ENVS``, ``CLCK``, ``FOCS``, and from G3b ``ALRT``, ``RPRT`` and ``PLAN``) to the
  protocol stream. The records are built from arrays it evaluates and converts to numpy before it returns,
  from the observer's accessors and from the storm layer; it holds no reference to the runner's state (the
  donation rule: the runner is the only holder when it steps).

The one thing a pacer never does on its own is change the trajectory. A ``power`` line (G3b, DESIGN §34.5) is
the exception, and it goes through the director: the pacer only *queues* the request
(``Director.queue_player``, after the dry run that answers a preview), the runner applies it at the top of the
next iteration, and what was applied is in ``director.jsonl`` before the state is handed back.
``game.powers.PowerDesk`` writes the ``PLAN`` records and the player's input log (``commands.jsonl``) around
that.

``parse_play_command`` is the control grammar (§33.2.6). The module reads no wall clock and sleeps only
through the injected ``Clock`` (``tests/test_lint.py``), so the scheduler tests run on a fake one. The pacer
never raises into the runner (which has no ``try/finally``): a closed reader or an internal failure sets
``stop_requested``, swaps the output for a null sink and lets the runner take its ordinary stop path
(checkpoint, event chunk, status), after which ``failure`` says what happened."""

from __future__ import annotations

import dataclasses
import functools
import math
import re
import statistics
import sys
import threading
import time
import traceback
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, BinaryIO, Final, Protocol

import mlx.core as mx
import numpy as np

from alife import layout, metrics, runner, streamrec
from alife import weather as W
from alife.config import Config, config_hash, dims, immigration_on
from alife.director import actions as director_actions
from alife.director import log as director_log
from alife.director import pacing, rules
from alife.game import alerts as alerts_mod
from alife.game import observer as observer_mod
from alife.game import powers as powers_mod
from alife.game import report as report_mod
from alife.game import reportjob
from alife.game.chronicle import Chronicler, rebuild_backlog
from alife.game.colors import OBSERVER_MODES, play_colors
from alife.game.observe import EVAL_BIT
from alife.game.observer import Observer, default_rules, house_bytes, slot_bytes
from alife.game.report import label_gate, label_names
from alife.game.rules import INTENT_VERSION
from alife.runio import ALERTS_FILE, ALERTS_REPLAY_FILE, COMMANDS_FILE, DIRECTOR_LOG_FILE, RunDir
from alife.state import ERROR_MASK, CounterIdx, DeathCause, State, default_env, to_numpy
from alife.step import StepOut
from alife.storms import StormLayer, dc_reference
from alife.stream import Control, ControlError, reader_closed
from alife.stream import _write_all as write_all  # the stream module's short-write handling, shared
from alife.weather import WeatherParams

__all__ = [
    "BASE_RATE",
    "PLAY_CADENCE",
    "PLAY_TICK",
    "POWER_KEYS",
    "POWER_NAMES",
    "SAVE_MODES",
    "Autopause",
    "AutopauseLine",
    "ChapterClock",
    "Clock",
    "PlayCommand",
    "PlayOptions",
    "PlayPacer",
    "PowerLine",
    "RealClock",
    "Scheduler",
    "check_tick",
    "director_effects",
    "origin_row",
    "parse_play_command",
    "planet_origin_step",
    "replaying",
    "serve_play",
    "world_tick",
]

BASE_RATE: Final = (
    14.4  # steps per second at 1x: a design rate (the cadences of the game are derived from it)
)
RATES: Final = (
    0.25,
    0.5,
    1.0,
    2.0,
)  # the rungs the window offers; ``rate`` accepts any M in [RATE_MIN, RATE_MAX]
RATE_MIN: Final = 0.05
RATE_MAX: Final = 16.0
MAX_RATE: Final = math.inf  # the multiplier of ``max``: unthrottled
PLAY_TICK: Final = (
    25  # the storm tracker's cadence in a world without weather (and the weather's detect_every)
)
SNAPSHOT_EVERY: Final = 250  # the social snapshot, the metrics row and the play tick all divide into this
CHAPTER: Final = 6000  # the save point and (G3b) the chapter length
FRAME_HZ: Final = 30
PLAY_CADENCE: Final = {
    "metrics_every": 250,
    "heavy_metrics_every": 250,
    "checkpoint_every": 6000,
    "keep_checkpoints": 3,
}  # run.* keys play applies on top of the world's config (resume-allowed ones; DESIGN §33.2.2)
WAIT_S: Final = 0.25  # the longest wait, the CLCK period and the pause-poll period
LAG_S: Final = 0.25  # a debt older than this is dropped (and counted late)
ACHIEVED_S: Final = 2.0  # the sliding window of the achieved rate
FOCS_PERIOD_S: Final = 0.25  # a focused organism is reported this often (every step while followed)
MAX_EVENTS_PER_RECORD: Final = streamrec.MAX_EVENTS
EPS: Final = 1e-9

COMMANDS: Final = (
    "pause",
    "run",
    "step",
    "rate",
    "color",
    "every",
    "focus",
    "follow",
    "idnt",
    "labels",
    "quit",
    "autopause",
    "watch",
    "unwatch",
    "ack",
    "power",
    "intent",
    "saves",
)  # the six after ``quit`` are G3b's (DESIGN §34.6.1), ``saves`` is §34.10's
# lines that are parsed but have no handler yet (each G3b line came with its stage; none remains)
UNBUILT_LINES: Final = frozenset[str]()
SAVE_MODES: Final = (
    streamrec.SAVE_MODES
)  # ``saves all|newest``: keep the last ``keep`` saves, or only the newest
POWER_NAMES: Final = ("food", "bloom", "blight", "strike")
POWER_KEYS: Final = ("mult", "steps", "patch", "gain", "x", "y", "radius", "kill")
POWER_INT_KEYS: Final = frozenset({"steps", "patch"})  # the others are decimals
_CLASS = re.compile(r"[a-z][a-z0-9_]*")
_UINT = re.compile(r"[0-9]+")
_DECIMAL = re.compile(r"[0-9]+(\.[0-9]+)?")
INT64_MAX: Final = 2**63 - 1


def check_tick(tick: int) -> None:
    """The tick must divide the snapshot cadence (the play records coincide at the boundary)."""
    if tick < 1 or SNAPSHOT_EVERY % tick:
        raise ValueError(f"the tick must divide {SNAPSHOT_EVERY}, got {tick}")


def world_tick(cfg: Config) -> int:
    """The storm tracker's cadence in a world: the weather's own ``detect_every`` (calibrated together with
    the tracker's ``miss_grace`` and predictor, so there is no flag for it) or ``PLAY_TICK`` without weather.
    """
    return W.params_from_config(cfg).detect_every if cfg.weather.enabled else PLAY_TICK


# --- the clock ------------------------------------------------------------------------------------------


class Clock(Protocol):
    """Time for the scheduler: ``monotonic()`` seconds, ``sleep(s)`` and ``wait(control, timeout)``, which
    blocks until a control line arrives or ``timeout`` seconds pass and returns the lines then available."""

    def monotonic(self) -> float: ...

    def sleep(self, seconds: float) -> None: ...

    def wait(self, control: Control, timeout: float) -> list[str]: ...


class RealClock:
    """The wall clock. The only place of this module that reads or waits on real time."""

    def monotonic(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)

    def wait(self, control: Control, timeout: float) -> list[str]:
        return control.poll(timeout)


# --- the grammar ----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class AutopauseLine:
    """``autopause`` parsed: ``scope`` ``all`` (every pause), ``chapter`` or ``alert`` (then ``cls`` names the
    alert class), and ``on``."""

    scope: str
    cls: str | None
    on: bool


@dataclass(frozen=True)
class PowerLine:
    """``power`` parsed: ``verb`` ``preview``, ``apply`` (``power NAME ...``) or ``cancel``; ``power`` one of
    ``POWER_NAMES``; ``id`` the client's request number; ``params`` the ``key=value`` pairs in the order
    given (ints for ``steps`` and ``patch``, floats for the rest). The grammar checks the shape of a value;
    its range is the power's to clamp or refuse."""

    verb: str
    power: str
    id: int
    params: tuple[tuple[str, int | float], ...] = ()


@dataclass(frozen=True)
class PlayCommand:
    """One control line, parsed. ``arg``: an int (``step``, ``every``, ``focus``, ``follow``, ``watch``,
    ``unwatch``, ``ack``), a float (``rate`` M, 0 pauses), a string (``rate`` ``max``, ``color`` MODE,
    ``every`` ``auto``, ``focus`` / ``follow`` / ``watch`` ``off``, ``labels`` ``on`` / ``off``, ``saves``
    ``all`` / ``newest`` / ``single``, the text of ``intent``), an ``AutopauseLine`` (``autopause``), a
    ``PowerLine``
    (``power``) or None."""

    name: str
    arg: int | float | str | AutopauseLine | PowerLine | None = None


def _uint(text: str, what: str) -> int:
    if not _UINT.fullmatch(text):
        raise ControlError(f"{what} must be a non-negative integer, got {text!r}")
    return int(text)


def parse_play_command(line: str, *, modes: Sequence[str] = streamrec.COLOR_MODES) -> PlayCommand | None:
    """One control line of ``alife play`` (None for a blank line or a ``#`` comment; ``ControlError`` if
    invalid).

    ``pause`` | ``run`` | ``step [N]`` | ``rate (M | max)`` | ``color MODE`` | ``every (K | auto)`` | ``focus
    (UID | off)`` | ``follow (UID | off)`` | ``idnt`` | ``labels (on | off)`` | ``quit``; and G3b's
    ``autopause (on | off | chapter (on | off) | alert CLASS (on | off))`` | ``watch (KIN_ID | off)`` |
    ``unwatch KIN_ID`` | ``ack ALERT_ID`` | ``power preview NAME id=N [k=v ...]`` |
    ``power NAME id=N [k=v ...]`` | ``power cancel NAME id=N`` | ``intent TEXT``; and §34.10's ``saves (all |
    newest | single)``; lower case (the text of
    ``intent`` excepted), whitespace separated. ``N`` >= 1; ``M`` a decimal, 0 for a pause or in [0.05, 16];
    ``K`` in 1..64; ``UID``, ``KIN_ID`` and ``ALERT_ID`` non-negative int64. ``fields`` is the Lab's and
    refused here."""
    tok = line.split()
    if not tok or tok[0].startswith("#"):
        return None
    name, args = tok[0], tok[1:]
    if name in ("fields", "field"):
        raise ControlError("fields belong to the Lab's viewer: the planet map replaces them in a play stream")
    if name not in COMMANDS:
        raise ControlError(f"unknown command {name!r}; expected one of {', '.join(COMMANDS)}")

    def arity(lo: int, hi: int) -> None:
        if not lo <= len(args) <= hi:
            want = str(lo) if lo == hi else f"{lo} to {hi}"
            raise ControlError(f"{name} takes {want} argument(s), got {len(args)}")

    if name in ("pause", "run", "idnt", "quit"):
        arity(0, 0)
        return PlayCommand(name)
    if name == "step":
        arity(0, 1)
        n = _uint(args[0], "N") if args else 1
        if n < 1:
            raise ControlError("step N needs N >= 1")
        return PlayCommand(name, n)
    if name == "rate":
        arity(1, 1)
        if args[0] == "max":
            return PlayCommand(name, "max")
        if not _DECIMAL.fullmatch(args[0]):
            raise ControlError(f"rate takes a decimal multiplier or 'max', got {args[0]!r}")
        m = float(args[0])
        if m != 0.0 and not RATE_MIN <= m <= RATE_MAX:
            raise ControlError(f"rate must be 0 or in [{RATE_MIN}, {RATE_MAX}], got {m}")
        return PlayCommand(name, m)
    if name == "color":
        arity(1, 1)
        if args[0] not in modes:
            raise ControlError(f"unknown colour mode {args[0]!r}; expected one of {tuple(modes)}")
        return PlayCommand(name, args[0])
    if name == "every":
        arity(1, 1)
        if args[0] == "auto":
            return PlayCommand(name, "auto")
        k = _uint(args[0], "K")
        if not 1 <= k <= 64:
            raise ControlError("every K needs K in 1..64")
        return PlayCommand(name, k)
    if name in ("focus", "follow"):
        arity(1, 1)
        if args[0] == "off":
            return PlayCommand(name, "off")
        uid = _uint(args[0], "UID")
        if uid > INT64_MAX:
            raise ControlError("UID does not fit an int64")
        return PlayCommand(name, uid)
    if name == "labels":
        arity(1, 1)
        if args[0] not in ("on", "off"):
            raise ControlError("labels takes 'on' or 'off'")
        return PlayCommand(name, args[0])
    return _parse_game_command(name, args, line)


def _parse_game_command(name: str, args: list[str], line: str) -> PlayCommand:
    """The G3b lines (``autopause watch unwatch ack power intent``) and ``saves``; ``args`` are the tokens
    after it."""
    if name == "intent":  # the text is free: everything after the name, whitespace kept as typed
        text = line.strip()[len("intent") :].strip()
        if not text:
            raise ControlError("intent takes a text")
        return PlayCommand(name, text)
    if name == "autopause":
        return PlayCommand(name, _parse_autopause(args))
    if name == "saves":
        if len(args) != 1 or args[0] not in SAVE_MODES:
            raise ControlError(f"saves takes one of {', '.join(SAVE_MODES)}")
        return PlayCommand(name, args[0])
    if name == "power":
        return PlayCommand(name, _parse_power(args))
    if len(args) != 1:
        raise ControlError(f"{name} takes 1 argument(s), got {len(args)}")
    if name == "watch" and args[0] == "off":
        return PlayCommand(name, "off")
    what = {"watch": "KIN_ID", "unwatch": "KIN_ID", "ack": "ALERT_ID"}[name]
    n = _uint(args[0], what)
    if n > INT64_MAX:
        raise ControlError(f"{what} does not fit an int64")
    return PlayCommand(name, n)


def _on_off(token: str, what: str) -> bool:
    if token not in ("on", "off"):
        raise ControlError(f"{what} takes 'on' or 'off'")
    return token == "on"


def _parse_autopause(args: list[str]) -> AutopauseLine:
    if len(args) == 1:
        return AutopauseLine("all", None, _on_off(args[0], "autopause"))
    if args and args[0] == "chapter" and len(args) == 2:
        return AutopauseLine("chapter", None, _on_off(args[1], "autopause chapter"))
    if args and args[0] == "alert" and len(args) == 3:
        if not _CLASS.fullmatch(args[1]):
            raise ControlError(f"autopause alert takes an alert class name, got {args[1]!r}")
        return AutopauseLine("alert", args[1], _on_off(args[2], "autopause alert"))
    raise ControlError("autopause takes on|off, chapter on|off or alert CLASS on|off")


def _parse_power(args: list[str]) -> PowerLine:
    verb = "apply"
    if args and args[0] in ("preview", "cancel"):
        verb, args = args[0], args[1:]
    if not args:
        raise ControlError("power takes a power name")
    power, rest = args[0], args[1:]
    if power not in POWER_NAMES:
        raise ControlError(f"unknown power {power!r}; expected one of {', '.join(POWER_NAMES)}")
    found: dict[str, int | float] = {}
    order: list[str] = []
    for token in rest:
        key, eq, value = token.partition("=")
        if not eq or key not in ("id", *POWER_KEYS):
            raise ControlError(f"power takes key=value pairs of id, {', '.join(POWER_KEYS)}, got {token!r}")
        if key in found:
            raise ControlError(f"power names {key} twice")
        if key == "id" or key in POWER_INT_KEYS:
            number: int | float = _uint(value, key)
            if number > INT64_MAX:
                raise ControlError(f"{key} does not fit an int64")
        else:
            if not _DECIMAL.fullmatch(value):
                raise ControlError(f"{key} must be a decimal, got {value!r}")
            number = float(value)
            if not math.isfinite(number):
                raise ControlError(f"{key} is too large")
        found = {**found, key: number}
        order.append(key)
    if "id" not in found:
        raise ControlError("power needs id=N, a non-negative integer chosen by the client")
    if verb == "cancel" and len(found) > 1:
        raise ControlError("power cancel takes only id=N")
    return PowerLine(verb, power, int(found["id"]), tuple((k, found[k]) for k in order if k != "id"))


# --- the scheduler --------------------------------------------------------------------------------------


class Scheduler:
    """When may the next step start? A fixed timestep with a bounded debt (DESIGN §33.2.5).

    The requested rate is ``rate · base_rate`` steps per second (``rate`` infinite for ``max``). On every run
    or rate change the schedule is anchored at ``now`` with nothing issued; step ``k`` (counted from the
    anchor) is due at ``anchor + k / (rate · base_rate)``. ``poll()`` returns ``None`` when a step may start
    now (it counts it as issued) and otherwise the seconds to wait, at most ``WAIT_S``.

    **Debt.** A step that overruns its slot leaves the schedule behind; the steps owed are repaid one per
    call. A debt older than ``LAG_S`` is dropped: the schedule is re-anchored on one step due now and the
    dropped steps are added to ``late_steps``, so the loop never spirals. A checkpoint is not lateness: the
    pacer brackets it with ``begin_save`` and ``end_save``, which re-anchor and add the steps the stall
    skipped to ``save_stall_steps``.

    **Batches.** ``step N`` pauses and queues a batch of N steps issued without throttling; ``batch_done`` is
    set when the last step of a batch is issued (the pacer writes one frame after it, then the world stays
    paused). ``pause`` and ``run`` clear the queue."""

    def __init__(
        self, clock: Clock, *, base_rate: float = BASE_RATE, rate: float = 1.0, paused: bool = True
    ) -> None:
        self.clock = clock
        self.base_rate = base_rate
        self.rate = rate
        self.paused = paused
        self.queued: deque[int] = deque()
        self.running = 0  # steps left of the batch in progress
        self.late_steps = 0
        self.save_stall_steps = 0
        self.batch_done = False
        self._anchor = clock.monotonic()
        self._issued = 0
        self._owed_at_save = 0

    # --- commands ---------------------------------------------------------------------------------------

    @property
    def busy(self) -> bool:
        """A ``step`` batch is running or queued (a paused world steps only while it is)."""
        return self.running > 0 or bool(self.queued)

    def _reanchor(self) -> None:
        self._anchor, self._issued = self.clock.monotonic(), 0

    def restart(self) -> None:
        """Start the schedule from now with nothing issued or owed. The pacer calls it when the world is
        bound and again when the first step is done: the start-up (observers, files, the first records) and
        the compile of the first step are not lateness."""
        self._reanchor()

    def pause(self) -> None:
        self.paused = True
        self.queued.clear()
        self.running = 0

    def run(self) -> None:
        self.paused = False
        self.queued.clear()
        self.running = 0
        self._reanchor()

    def set_rate(self, rate: float) -> None:
        """Run at ``rate`` times the base rate (``MAX_RATE`` for max); 0 is a pause."""
        if rate == 0:
            self.pause()
            return
        self.rate = rate
        self.run()

    def step(self, n: int) -> None:
        self.paused = True
        self.queued.append(n)

    # --- the schedule -----------------------------------------------------------------------------------

    @property
    def steps_per_s(self) -> float:
        return self.rate * self.base_rate

    def requested(self) -> float:
        """The ``CLCK.rate`` value: 0 paused, -1 max, else the multiplier."""
        if self.paused:
            return 0.0
        return -1.0 if math.isinf(self.rate) else self.rate

    def _owed(self, now: float) -> int:
        return math.floor((now - self._anchor) * self.steps_per_s + EPS) - self._issued

    def poll(self) -> float | None:
        """None: a step may start now. Otherwise the seconds to wait (at most ``WAIT_S``) before asking."""
        if self.running == 0 and self.queued:
            self.running = self.queued.popleft()
        if self.running > 0:
            self.running -= 1
            self.batch_done = self.running == 0
            return None
        if self.paused:
            return WAIT_S
        if math.isinf(self.rate):
            return None
        now = self.clock.monotonic()
        sps = self.steps_per_s
        owed = self._owed(now)
        if owed >= 1:
            if owed / sps > LAG_S:  # too far behind: drop what cannot be repaid, keep one step due now
                self.late_steps += owed - 1
                self._issued += owed - 1
            self._issued += 1
            return None
        return min(max(self._anchor + (self._issued + 1) / sps - now, 1e-4), WAIT_S)

    def begin_save(self) -> None:
        """A checkpoint starts: remember the debt, so the stall's own steps can be told from it."""
        self._owed_at_save = max(self._owed(self.clock.monotonic()), 0) if self._timed() else 0

    def end_save(self, *, count: bool = True) -> None:
        """The checkpoint is written: the steps the stall skipped are ``save_stall_steps`` (never late) and
        the schedule starts again from now. ``count=False`` is for a stall the player did not wait through: a
        chapter end whose pause follows, where the world is about to stop and the stall costs nothing."""
        if self._timed():
            if count:
                self.save_stall_steps += max(self._owed(self.clock.monotonic()) - self._owed_at_save, 0)
            self._reanchor()

    def _timed(self) -> bool:
        return not self.paused and not self.busy and not math.isinf(self.rate)


# --- chapters and autopause ------------------------------------------------------------------------------


@dataclass(frozen=True)
class ChapterClock:
    """Where a step is in the chapters (§34.2). Chapter ``n`` is the steps ``(length (n - 1), length n]``,
    so a chapter's last step belongs to it: at a boundary the bar reads 100% and the chapter changes at the
    next step. The first step of a world (``t = 0``) is chapter 1 at 0. A host-only counter: it reads no state
    and changes no file."""

    length: int = CHAPTER

    def chapter(self, t: int) -> int:
        """The absolute chapter containing step ``t`` (at least 1)."""
        return max(1, -(-t // self.length))

    def position(self, t: int) -> tuple[int, int]:
        """``(chapter, in_chapter)``: ``in_chapter`` is in ``1..length`` once ``t > 0`` and 0 at the start."""
        chapter = self.chapter(t)
        return chapter, t - self.length * (chapter - 1)

    def is_end(self, t: int) -> bool:
        """Whether ``t`` is the end of a chapter (a stop checkpoint at another step is not)."""
        return t > 0 and t % self.length == 0


@dataclass
class Autopause:
    """Which pauses the world takes by itself (``autopause`` lines, DESIGN §34.2): at a chapter end, and per
    alert class. ``alert`` holds the classes the world knows (the header's registry); presentation only."""

    chapter: bool = True
    alert: dict[str, bool] = dataclasses.field(default_factory=dict)

    def apply(self, line: AutopauseLine) -> str | None:
        """Sets what ``line`` says; a note (and no change) for an alert class the world does not have."""
        if line.scope == "all":
            self.chapter = line.on
            self.alert = dict.fromkeys(self.alert, line.on)
        elif line.scope == "chapter" or (line.scope == "alert" and line.cls == "chapter"):
            self.chapter = line.on  # the chapter end is an alert class with its own switch
        else:
            assert line.cls is not None
            if line.cls not in self.alert:
                return f"autopause: this world has no alert class {line.cls!r}"
            self.alert = {**self.alert, line.cls: line.on}
        return None

    @property
    def flags(self) -> int:
        """``CLCK.flags``: bit 0 the pause at a chapter end is armed, bit 1 at least one alert class is."""
        return (streamrec.CLCK_FLAG_CHAPTER_PAUSE if self.chapter else 0) | (
            streamrec.CLCK_FLAG_ALERT_PAUSE if any(self.alert.values()) else 0
        )

    def block(self) -> dict[str, Any]:
        """The header's ``play.autopause``: the defaults the client starts from."""
        return streamrec.autopause_block(chapter=self.chapter, alert=self.alert)


# --- the options ----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PlayOptions:
    """What a played world is told besides its config (``alife play`` flags and ``play.json``).

    ``rate`` is the multiplier of the base rate the world runs at (``MAX_RATE`` for max) and ``paused``
    whether it starts paused. ``tick`` is the storm tracker's cadence (None: the weather's own
    ``detect_every`` or ``PLAY_TICK``). ``fresh`` marks a world made by this call: its tracker starts empty at
    the start step without the ``renumbered`` event and the empty state is saved with the first checkpoint.
    ``dc_ref`` is the cloud-depth scale of ``MAP1`` (None: computed from the state at start). ``on_ready``
    runs once the facts of the world are known and before the header is written (the caller writes
    ``play.json`` there); ``until`` is a step at which the stream says "end of steps". ``chapter`` is the
    chapter length in steps (``CHAPTER``; a multiple of the checkpoint cadence, which the tests shorten).
    ``alerts_experimental`` switches on the alert classes marked experimental (``--alerts experimental``).
    ``unattended`` is a world nobody reads (the benchmark's arms, a test's scripted run): no pause is armed at
    the start, neither at a chapter end nor for an alert, since nobody would press ``run``.
    ``origin_step`` is the step the world's chapter numbering starts after (the planet's ``made_at_step`` of
    ``play.json``, ``planet_origin_step``; 0 for a preset or an adopted run): the report's chapter is
    ``step // chapter - origin_step // chapter``, as the client's bar and card count it (§34.2.1).
    ``planet`` is only the header's block (``source``, ``houses``) and does not hold it. ``saves`` is the save
    mode the session starts in (``play.json``'s ``saves``, ``--saves``; ``SAVE_MODES``, DESIGN §34.10)."""

    base_rate: float = BASE_RATE
    rate: float = 1.0
    paused: bool = True
    frame_hz: int = FRAME_HZ
    tick: int | None = None
    source: str = "preset"
    run: str | None = None
    planet: Mapping[str, Any] | None = None
    dc_ref: float | None = None
    fresh: bool = False
    until: int | None = None
    on_ready: Callable[[dict[str, Any]], None] | None = None
    chapter: int = CHAPTER
    alerts_experimental: bool = False
    origin_step: int = 0
    unattended: bool = False
    saves: str = "all"
    compress: bool = True  # the saves are compressed (DESIGN §34.11); a runtime setting, not in play.json


class _NullSink:
    """Where the stream goes after the reader has gone or the pacer has failed."""

    def write(self, data: Any) -> int:
        return len(data)

    def flush(self) -> None:
        pass


@dataclass
class _Followed:
    """The followed organism as the last step left it: the cache the hand-off reads (a dead row is cleared at
    its death and ``DeathRecords`` carry no position)."""

    uid: int
    row: int
    pos: tuple[float, ...]
    kin: int  # SocState.kin of the row (raw slot value; -1 without observers)
    children: frozenset[int] = frozenset()


CLOSED_READER: Final = (BrokenPipeError, ConnectionResetError)  # the reader is gone: a normal way to stop
HOST_WORK: Final = frozenset({"after_step", "before_checkpoint"})  # the callbacks that keep the files exact


def _guard(method: Callable[..., Any]) -> Callable[..., Any]:
    """Run a pacer callback so that it never raises into the runner. A closed reader (``BrokenPipeError``,
    ``ConnectionResetError``) stops the play quietly; any other exception, ``OSError`` included, stops it
    with the traceback on stderr and ``failure`` set.

    After a failure the stream is a ``_NullSink``, and the callbacks of ``HOST_WORK`` still run: the storm
    tick of the last step and the tracker state saved with the checkpoint belong to the files, not to the
    stream, and a resume must find them (DESIGN §33.3.2). The others return at once."""

    name = method.__name__
    idle = 0.0 if name == "take_paused_s" else None

    @functools.wraps(method)
    def wrapper(self: PlayPacer, *args: Any, **kwargs: Any) -> Any:
        if self._dead and name not in HOST_WORK:
            return idle
        try:
            return method(self, *args, **kwargs)
        except CLOSED_READER as e:
            self._fail(e, quiet=True)
        except Exception as e:
            self._fail(e, quiet=False)
        return idle

    return wrapper


# --- the pacer ------------------------------------------------------------------------------------------


class PlayPacer:
    """The ``runner.Pacer`` of ``alife play``: a ``Scheduler`` and the presenter of the stream (module
    docstring).

    ``out`` is the protocol stream (``stream.protocol_output()``), ``control`` the source of control lines.
    After the run ``failure`` is None, a ``BrokenPipeError`` or ``ConnectionResetError`` (the reader closed: a
    normal way to stop) or the exception that stopped the play (an ``OSError`` that is not a closed reader,
    a full disk say, is one of those)."""

    def __init__(
        self,
        out: BinaryIO,
        control: Control,
        options: PlayOptions | None = None,
        *,
        clock: Clock | None = None,
        note: Callable[[str], None] | None = None,
    ) -> None:
        self.opt = options or PlayOptions()
        self.clock: Clock = clock or RealClock()
        self.out: Any = out
        self.control = control
        self._note = note or (lambda m: print(f"play: {m}", file=sys.stderr, flush=True))
        self.sched = Scheduler(
            self.clock, base_rate=self.opt.base_rate, rate=self.opt.rate, paused=self.opt.paused
        )
        self.stop_requested = False
        self.failure: BaseException | None = None
        self._dead = False
        self._ended = False
        self.seq = streamrec.Sequencer()
        self.bytes_by_tag: dict[str, int] = {}
        # One lock for everything a worker thread can reach: the write of a whole record (a 1.4 MB FRMA is
        # several short writes; a lock around one write call would let a worker's record interleave with it),
        # the byte counters and the failure state. Re-entrant: a failed write calls ``_fail`` under it.
        self._lock = threading.RLock()
        # chapters and the pauses the world takes by itself (DESIGN §34.2)
        self.chapters = ChapterClock(self.opt.chapter)
        self.autopause = Autopause(alert=alerts_mod.default_autopause())
        if self.opt.saves not in SAVE_MODES:
            raise ValueError(f"saves must be one of {SAVE_MODES}, got {self.opt.saves!r}")
        self.saves = self.opt.saves  # the save mode in force (``saves`` lines, DESIGN §34.10)
        if self.opt.unattended:
            self.autopause.apply(AutopauseLine("all", None, False))
        # the alerts the world raises (DESIGN §34.3): a pure engine over the host quantities of each hook
        self.alerts = alerts_mod.AlertEngine(
            armed=self._armed, chapter=self.opt.chapter, experimental=self.opt.alerts_experimental
        )
        self._alert_pause_pending = False  # an alert asked for a pause: it starts at the next ``before_step``
        self._replaying = (
            False  # a replay of the director's log: its engine-class alerts go to their own file
        )
        self._terr: np.ndarray | None = None  # the latest territory: the (ny, nx) grid of owning kin ids
        self._storms_now: Sequence[Mapping[str, Any]] = ()  # the storms the latest tracker tick detected
        # the report job (DESIGN §34.4): a worker that builds the report of each chapter at its end; None for
        # a world with no observers or a chapter that is not a whole number of social snapshots
        self.reports: reportjob.ReportJob | None = None
        # the player's powers (DESIGN §34.5): previews, requests, the PLAN cards, ``commands.jsonl``
        self.desk: powers_mod.PowerDesk | None = None
        # called with the step of a chapter end, in ``before_checkpoint``, once the announcing ``CLCK`` is
        # sent
        self.on_chapter_end: Callable[[int], None] | None = None
        # a chapter ended: the pause (if armed) starts at the next ``before_step``
        self._chapter_pending = False
        # why the world paused itself: 3 chapter, 4 alert (0: not, or a pause of the player's own)
        self._pause_reason = 0
        # what the world is (set in bind)
        self.cfg: Config
        self.rd: RunDir
        self.obs: Observer | None = None
        self.director: Any = None
        self.layer: StormLayer | None = None
        self.chron: Chronicler
        self._stop_check: Callable[[], bool] = lambda: False
        self._tick = PLAY_TICK
        self._t = 0
        # presentation state
        self.color = "tag"
        self.labels_on = True
        self.fe_override: int | None = None
        self._fe = 1
        self._house_top = np.zeros(0, np.int64)
        self._tables: dict[str, Any] = {}
        self._events: list[dict[str, Any]] = []
        self._last_frame_step = -1
        self._frames_written = 0
        self._last_clck = -1e9
        self._saving = False
        # timing
        self._t_step = 0.0
        self._paused_s = 0.0
        self._step_ms: deque[float] = deque(maxlen=60)
        self._frame_ms: deque[float] = deque(maxlen=60)
        self._window: deque[tuple[float, int]] = deque()  # (time, steps taken) samples of the achieved rate
        self._steps_taken = 0
        # focus
        self.focus: int | None = None
        self.follow = False
        self._followed: _Followed | None = None
        self._last_focs = -1e9

    # --- failure ----------------------------------------------------------------------------------------

    def _fail(self, exc: BaseException, *, quiet: bool) -> None:
        """The first failure stops the play; a later one (the host work of the last step) replaces a closed
        reader as the reason, so that a full disk is not reported as a normal stop."""
        with self._lock:  # a worker thread can get here through ``post``
            first = not self._dead
            if first or (not quiet and isinstance(self.failure, CLOSED_READER)):
                self.failure = exc
            self._dead = True
            self.stop_requested = True
            self.out = _NullSink()
            if not quiet:
                try:  # the failure is an alert of the log (the stream is gone; the clock says "error")
                    self.alerts.failure(self._t, f"{type(exc).__name__}: {exc}")
                    self._emit_alerts()
                except Exception:  # a failure of the failure path changes nothing
                    pass
            if quiet:
                if first:
                    self._note(
                        f"the reader closed the stream ({type(exc).__name__}); stopping after a checkpoint"
                    )
            else:
                print(
                    "play: the presenter failed; stopping after a checkpoint:"
                    if first
                    else "play: a failure after the stop:",
                    file=sys.stderr,
                    flush=True,
                )
                traceback.print_exception(exc, file=sys.stderr)

    # --- output -----------------------------------------------------------------------------------------

    def _write(self, tag: str, data: bytes) -> None:
        """Write a record. A reader that has gone does not abort the callback: the stream becomes a null sink
        and the host work of the step (the storm tick, the logs) goes on, so the files stay exact."""
        with self._lock:
            try:
                write_all(self.out, data)
            except CLOSED_READER as e:
                self._fail(e, quiet=True)
                return
            self.bytes_by_tag = {**self.bytes_by_tag, tag: self.bytes_by_tag.get(tag, 0) + len(data)}

    def _send(self, tag: str, make: Callable[[int, int], bytes]) -> None:
        """Write the record ``make(step, seq)`` of ``tag`` at the current step."""
        self._write(tag, make(self._t, self.seq.take(tag)))

    def post(self, tag: str, make: Callable[[int, int], bytes], *, step: int) -> None:
        """Write the record ``make(step, seq)`` of ``tag`` from a worker thread (the report job, §34.2).

        The sequence number is taken inside the lock that spans the whole write, so that the records of one
        tag reach the stream in the order of their ``seq`` whichever thread writes them, and never between the
        short writes of another record. A reader that has gone is the pacer's failure (the loop takes its
        ordinary stop path at its next call); a record that cannot be made raises in the caller, which is
        the worker and must report it itself. Nothing is written after the pacer has failed."""
        with self._lock:
            if self._dead:
                return
            self._write(tag, make(step, self.seq.take(tag)))

    # --- the Pacer protocol -----------------------------------------------------------------------------

    @_guard
    def bind(
        self,
        state: State,
        *,
        rd: RunDir,
        cfg: Config,
        obs: Observer | None,
        director: Any,
        quiet: bool,
        stop_check: Callable[[], bool],
    ) -> None:
        self.rd, self.cfg, self.obs, self.director = rd, cfg, obs, director
        rd.set_save_mode(self.saves)
        self.alerts.cadence = (
            cfg.run.metrics_every
        )  # a row of fewer steps (a stop's, a resume's) is no sample
        self._replaying = replaying(director)
        self._stop_check = stop_check
        self._t = int(to_numpy(state.step))
        salt = obs.social.name_salt if obs is not None else 0
        self.chron = Chronicler(salt)
        d = dims(cfg)
        dc_ref = None
        if cfg.weather.enabled:
            p = W.params_from_config(cfg)
            self._tick = self.opt.tick or p.detect_every
            ws = W.unpack(state.wx)
            mx.eval(ws.Dc)
            dc_ref = self.opt.dc_ref if self.opt.dc_ref is not None else dc_reference(to_numpy(ws.Dc))
            layer, reason = StormLayer.open(rd, p, dc_ref, self._t, self._tick)
            self.layer = layer
            if self.opt.fresh:
                layer.initial_pending = True  # the storms its first tick finds were there before (not told)
                layer.save(rd, self._t)  # an empty tracker, so that a resume at the first checkpoint is exact
            elif layer.restarted:
                self._events += self.chron.renumbered(self._t)
                self._note(f"storm numbering restarts at step {self._t}: {reason}")
        else:
            self._tick = self.opt.tick or PLAY_TICK
        check_tick(self._tick)
        if self.opt.on_ready is not None:
            self.opt.on_ready({"dc_ref": dc_ref, "step": self._t, "tick": self._tick})
        self.desk = powers_mod.PowerDesk(
            cfg,
            director,
            base_rate=self.opt.base_rate,
            append=lambda lines: rd.append_jsonl(COMMANDS_FILE, lines),
        )
        records = [
            t
            for t in streamrec.TAGS
            if t in ("FRMA", "EVNT", "ENVS", "CLCK", "FOCS", "ALRT", "PLAN")
            or (t in ("IDNT", "INTN", "TERR") and obs is not None)
            or (t == "RPRT" and self._reports_wanted())
            or (t in ("MAP1", "STRM") and self.layer is not None)
        ]
        legacy = streamrec.legacy_header(
            dims=d.nd,
            size=cfg.world.size,
            height=cfg.world.height if d.d3 else 0.0,
            n=d.n,
            bond_slots=layout.B,
            radius=cfg.physics.radius,
            field_cell=cfg.world.field_cell,
            fg=d.fg,
            fz=d.fz,
            nch=layout.NCH,
            step=self._t,
            run=self.opt.run,
            config_hash=config_hash(cfg),
            source=self.opt.source,
        )
        powers = self.desk.header()
        play = streamrec.header_play(
            records=records,
            weather=streamrec.weather_block(enabled=self.layer is not None, dc_ref=dc_ref),
            labels=self._labels_block(cfg, obs),
            energy_cap=cfg.energy.e_max,
            planet=self.opt.planet,
            base_rate=self.opt.base_rate,
            rates=RATES,
            tick=self._tick,
            snapshot_every=obs.social.snapshot_every if obs is not None else SNAPSHOT_EVERY,
            frame_hz=self.opt.frame_hz,
            chapter=self.opt.chapter,
            autopause=self.autopause.block(),
            alerts=streamrec.alerts_block(alerts_mod.registry()),
            powers=powers,
            game_specs=director_actions.GAME_SPEC_VERSION if powers is not None else None,
            saves=streamrec.saves_block(mode=self.saves, keep=cfg.run.keep_checkpoints, adopted=rd.adopted),
        )
        self._write("HEAD", streamrec.play_header_bytes(legacy, play))
        self._backlog(rd, state)
        self._send_frame(state)
        self._send_tables(state)
        self._send_environment(state)
        self._send_weather(state)
        self._flush_events(backlog=True)
        self._send_alert_backlog()
        self._open_reports()
        self._send_plans(self.desk.open(rd.metrics_rows(), self._t))
        self._send_clock()
        self.sched.restart()
        self._t_step = self.clock.monotonic()

    @staticmethod
    def _labels_block(cfg: Config, obs: Observer | None) -> dict[str, Any]:
        """The header's ``labels`` object (DESIGN §33.1.1): the names of the rule set in force (a weather
        world's are the storm names, as N1's reports print them) and the held-out verdict gate of N1 for this
        world and rule set (``report.label_gate``)."""
        rules = obs.rules if obs is not None else default_rules(cfg)
        arch, intent = label_names(INTENT_VERSION, cfg.weather.enabled)
        gate = label_gate(cfg, INTENT_VERSION, rules.digest())
        return streamrec.labels_block(intent_version=INTENT_VERSION, arch=arch, intent=intent, gate=gate)

    def _backlog(self, rd: RunDir, state: State) -> None:
        """The newest 64 events of the world's logs, which also warm the chronicler's names (the events are
        sent after the first records, flagged as the backlog)."""
        entries = (
            director_log.entries_of(rd.read_jsonl(DIRECTOR_LOG_FILE)) if self.director is not None else []
        )
        records = rd.social_records()
        log = rebuild_backlog(
            records,
            rd.storms_lines(),
            entries,
            house_salt=self.chron.house_salt,
            chronicler=self.chron,
        )
        self._events = [*log, *self._events]
        self._restore_threads(rd)
        self._open_alerts(rd, records)

    def _open_alerts(self, rd: RunDir, records: Sequence[Mapping[str, Any]]) -> None:
        """Open the alert engine on the files of this world: the histories of the population from the rows up
        to the start step (the last 25 and the last chapter ends, the origin row among them), the houses of
        the social log and the storms the tracker holds as named. All silent: a resume re-arms everything and
        the first evaluation only arms the level classes."""
        rows = rd.metrics_rows()
        ends = [r for r in rows if r["step"] % self.opt.chapter == 0 and r["step"] > 0]
        origin = rd.meta().get("origin_row")
        if isinstance(origin, dict) and origin.get("step", 1) % self.opt.chapter == 0:
            ends = [origin, *ends]
        have = {r["step"]: r for r in [*ends[-alerts_mod.SLIDE_REFS :], *rows[-(alerts_mod.LOW_ROWS + 1) :]]}
        self.alerts.prime_rows([have[k] for k in sorted(have)])
        self.alerts.prime_records(records)
        if self.layer is not None:
            self.alerts.prime_storms(i for i, n in self.layer.notes.items() if n.name)

    def _restore_threads(self, rd: RunDir) -> None:
        """The kin thread table the checkpoint of this step was saved with, if it is usable: the refold of the
        social log that made the backlog is its twin (a pure function of the same records), so this only
        matters when the log was cut at its start. A missing or unusable file leaves the refold."""
        try:
            saved = rd.read_play_state("kinthreads", self._t)
            if saved is not None:
                self.chron.restore_threads(saved)
        except (OSError, ValueError) as e:
            self._note(
                f"the kin thread state at step {self._t} is unusable ({e}); the social log is refolded"
            )

    def _reports_wanted(self) -> bool:
        """Whether this world has live reports: it is observed, its social snapshots are at the cadence a
        report is cut at (``ReportRules.snapshot_every``) and its chapter is a whole number of them."""
        grid = report_mod.REPORT_V1.snapshot_every
        return (
            self.obs is not None and self.obs.social.snapshot_every == grid and self.opt.chapter % grid == 0
        )

    def _open_reports(self) -> None:
        """Open the report job of this world (an observed one whose chapter is a whole number of social
        snapshots), send the last reports of ``reports.jsonl`` as the backlog and queue the chapters of the
        last three that have none (a quit just after a chapter end can stop before the worker has written its
        record; a report is a pure function of the files)."""
        if not self._reports_wanted():
            return
        grid = report_mod.REPORT_V1.snapshot_every
        length = self.opt.chapter
        n_origin = reportjob.origin_chapters(self.opt.origin_step, length)
        start = int(self.rd.meta().get("start_step") or 0)
        self.reports = reportjob.ReportJob(
            self.rd.path,
            length=length,
            n_origin=n_origin,
            emit=self._emit_report,
            note=self._note,
            start_step=start,
        )
        self.on_chapter_end = self._start_report
        for step, obj in reportjob.stored_objects(self.rd.path, length=length, n_origin=n_origin):
            self._emit_report(step, obj, True)
        for b in reportjob.missing_chapters(self.rd.reports(), self._t, start, length, grid):
            self.reports.start(b, None, backlog=True)

    def _emit_report(self, step: int, obj: dict[str, Any], backlog: bool) -> None:
        """Write an ``RPRT`` (from the loop thread or the report worker: ``post`` takes the write lock)."""
        self.post(
            "RPRT",
            lambda st, seq: streamrec.encode_rprt(step=st, seq=seq, obj=obj, backlog=backlog),
            step=step,
        )

    def _start_report(self, t: int) -> None:
        """A chapter ended at ``t``: queue its report. The sidecar of ``t`` is written inside the checkpoint
        that has just started, so the worker is handed the ``meta`` it will hold."""
        if self.reports is not None and self.obs is not None:
            self.reports.start(t, self.obs.meta())

    def close(self) -> None:
        """Let the report worker finish what it has (five seconds at most) and join it. ``serve_play`` calls
        it when the run has ended; a chapter whose report was not finished is rebuilt when the world is
        opened."""
        if self.reports is not None and not self.reports.close():
            self._note("the report worker did not finish within 5 s; it is rebuilt when the world is opened")

    @_guard
    def before_step(self, state: State, t: int) -> None:
        self._t = t
        if self._chapter_pending:
            self._begin_chapter_pause(state)
        if self._alert_pause_pending:
            self._begin_alert_pause(state)
        pending = self.control.poll(0)
        while True:
            if pending and self._apply(pending, state):
                break
            pending = []
            if self.stop_requested or self._stop_check():
                break  # a stop lets exactly one more step run: the runner's unscheduled-stop path
            timeout = self.sched.poll()
            if timeout is None:
                break
            self._service()
            paused = self.sched.paused
            t0 = self.clock.monotonic()
            pending = self.clock.wait(self.control, timeout)
            if paused:
                self._paused_s += self.clock.monotonic() - t0
        self._t_step = self.clock.monotonic()

    @_guard
    def after_step(self, state: State, out: StepOut, t: int, social: list[dict[str, Any]]) -> None:
        now = self.clock.monotonic()
        self._t = t
        self._step_ms.append((now - self._t_step) * 1e3)
        self._steps_taken += 1
        self._window.append((now, self._steps_taken))
        if self._steps_taken == 1:
            self.sched.restart()  # the compile of the first step is not lateness
        for rec in social:
            self._events += self.chron.from_record(rec)
            self.alerts.on_record(rec, self.chron.told_threads, self.chron.threads, pause_ok=self._pause_ok())
        if social and self.obs is not None:
            self._send_tables(
                state, labels=False
            )  # a snapshot: identities and territory (INTN follows with the tick)
            self._territory_alerts(t)
        if t % self._tick == 0:
            if self.layer is not None:
                self._tick_storms(state, t)
            if self.obs is not None and self.labels_on:
                self._send_labels(state)
        self._follow_step(state, out, t)
        self._player_step(state, t)
        batch_end = self.sched.batch_done
        self.sched.batch_done = False
        if batch_end or (not self.sched.paused and t % self._effective_fe() == 0):
            self._send_frame(state)
        if self.opt.until is not None and t >= self.opt.until:
            self._ended = True
        if self._extinct(state, out):
            self._events += self.chron.extinct(t)
        self._emit_alerts()
        self._flush_events()
        self._service()

    @_guard
    def before_checkpoint(self, state: State, t: int) -> None:
        self.rd.wait_saves()  # before the sidecars and play-state files of this save are written (§34.11)
        self._t = t
        self._saving = True
        self.sched.begin_save()
        if self.layer is not None:
            self.layer.save(self.rd, t)  # before the state file, as the observer sidecar is
        if self.obs is not None:
            self.rd.write_play_state("kinthreads", t, self.chron.threads.state())
        ended = self.chapters.is_end(t)
        if ended:
            # a chapter ended: the pause (if armed) starts at the next ``before_step``, so no step is owed
            self._chapter_pending = True
        if ended:
            self.alerts.on_chapter(t, pause_ok=self._chapter_pause_follows())
            self._emit_alerts()
        self._send_clock()  # reason 3, saving: the chapter card slides in
        if ended and self.on_chapter_end is not None:
            # the report job starts here, off the loop thread and overlapped with the save; the announcement
            # comes first, so the card is there for the report to fill (a worker's record follows the clock)
            self.on_chapter_end(t)

    @_guard
    def after_checkpoint(self, t: int) -> None:
        self._t = t
        self._saving = False
        # the stall of a chapter end whose pause follows costs the player nothing: they are reading the card
        self.sched.end_save(count=not self._chapter_pause_follows())
        self._events += self.chron.checkpoint(t)
        self._flush_events()
        self._send_clock()

    @_guard
    def after_boundary(self, state: State, row: dict[str, Any], t: int) -> None:
        self._t = t
        applied = self.director.applied if self.director is not None else ()
        if applied:
            self._events += self.chron.from_director(applied)
        self.alerts.on_row(
            row,
            t,
            entries=self.director.entries if self.director is not None else (),
            applied=applied,
            effects=director_effects(self.director, t),
            pause_ok=self._pause_ok(),
        )
        self._emit_alerts()
        if self.desk is not None:
            self._send_plans(self.desk.on_row(row))
        self._send_environment(state)
        self._flush_events()

    @_guard
    def take_paused_s(self) -> float:
        paused, self._paused_s = self._paused_s, 0.0
        return paused

    # --- commands ---------------------------------------------------------------------------------------

    def _apply(self, lines: Sequence[str], state: State) -> bool:
        """Apply control lines one at a time, in order, writing what each asks for before the next is looked
        at (so the output does not depend on how the pipe chunked the lines). True after ``quit``, which drops
        the rest."""
        modes = streamrec.COLOR_MODES
        for line in lines:
            try:
                cmd = parse_play_command(line, modes=modes)
            except ControlError as e:
                self._note(f"ignored control line {line.strip()[:80]!r}: {e}")
                continue
            if cmd is None:
                continue
            if cmd.name == "quit":
                self.stop_requested = True
                return True
            self._command(cmd, state, line.strip())
        return False

    def _command(self, cmd: PlayCommand, state: State, text: str = "") -> None:
        sched, name, arg = self.sched, cmd.name, cmd.arg
        if name == "pause":
            was_running = not sched.paused
            sched.pause()
            if was_running:
                self._pause_reason = 0  # a pause of the player's own
                self._enter_pause(state)
            self._send_clock()
        elif name == "run":
            sched.run()
            self._resumed(acknowledge=True)
            self._send_clock()
        elif name == "step":
            assert isinstance(arg, int)
            sched.step(arg)
            self._resumed(acknowledge=True)
            self._send_clock()
        elif name == "rate":
            was_running = not sched.paused
            sched.set_rate(MAX_RATE if arg == "max" else float(arg))  # type: ignore[arg-type]
            if not sched.paused or was_running:
                self._resumed(acknowledge=self._pause_reason == 3)
            if sched.paused and was_running:
                self._enter_pause(state)
            self._send_clock()
        elif name == "color":
            assert isinstance(arg, str)
            if arg in OBSERVER_MODES and self.obs is None:
                self._note(f"colour mode {arg!r} needs the observers (game.observe)")
                return
            self.color = arg
            self._send_frame(state)
        elif name == "every":
            self.fe_override = None if arg == "auto" else int(arg)  # type: ignore[arg-type]
            self._send_clock()
        elif name == "idnt":
            self._send_frame(state)
            self._send_tables(state)
            self._send_environment(state)
            self._send_weather(state)
        elif name == "labels":
            self.labels_on = arg == "on"
        elif name == "saves":
            assert isinstance(arg, str)
            self._set_saves(arg)
            self._send_clock()
        elif name == "autopause":
            assert isinstance(arg, AutopauseLine)
            if (note := self.autopause.apply(arg)) is not None:
                self._note(note)
            self._send_clock()
        elif name == "watch":
            self._watch(arg)
        elif name == "unwatch":
            assert isinstance(arg, int)
            if (note := self.alerts.unwatch(arg, self.chron.threads)) is not None:
                self._note(note)
        elif name == "ack":
            assert isinstance(arg, int)
            if (note := self.alerts.ack(arg, self._t)) is not None:
                self._note(note)
            self._emit_alerts()
        elif name == "power":
            self._power(arg, text, state)
        elif name == "intent":  # reserved for G4: answered with a refusal
            if self.desk is not None:
                self._send_plans(self.desk.intent(text, self._t))
        elif name in UNBUILT_LINES:
            self._note(f"{name}: not available in this build yet")
        else:  # focus, follow
            self._command_focus(cmd, state)

    def _set_saves(self, mode: str) -> None:
        """``saves all|newest`` (DESIGN §34.10): from the next save on, the world keeps the last
        ``run.keep_checkpoints`` saves (``all``) or only the newest one (``newest``); the origin is kept
        either way. Nothing is deleted now: ``RunDir.save_checkpoint`` prunes, after it has written the new
        save, so one complete save exists at every instant. A change is kept in ``play.json`` (so a resume
        keeps it) and logged to ``commands.jsonl``; a line that changes nothing does neither. Storage only: it
        reaches no state, row or log of the trajectory.

        A research run played in place (``alife play RUN --adopt``, ``RunDir.adopted``) holds the research's
        checkpoints (``init-from RUN:STEP``, the houses, the activity analyses), which a click must not
        delete: a ``saves newest`` line that would change its mode is refused with a note (the header's
        ``play.saves.adopted`` tells the client, which does not offer it), and ``alife play RUN --saves
        newest`` is the explicit way, with a note that says what it deletes."""
        if mode != "all" and self.saves != mode and self.rd.adopted:
            self._note(
                "saves: refused: this world is a research run played in place, and keeping only the "
                f"{mode} save would delete its checkpoints; `alife play RUN --saves {mode}` does that on "
                "purpose"
            )
            return
        was, self.saves = self.saves, mode
        self.rd.set_save_mode(mode)
        if mode == was:
            return
        try:
            if not self._replaying:  # (a replay of the log writes no input log, like the powers)
                self.rd.append_jsonl(
                    COMMANDS_FILE,
                    [{"t": self._t, "verb": "saves", "status": "applied", "mode": mode, "was": was}],
                )
            self._persist_saves()
        except (OSError, ValueError) as e:  # the mode is in force; a resume will not know of it
            self._note(f"saves: could not keep the mode in the world's files: {e}")
        self._note(
            "saves: only the newest save (and the origin) is kept, from the next save on"
            if mode == "newest"
            else f"saves: the last {self.cfg.run.keep_checkpoints} saves are kept, from the next save on"
        )

    def _persist_saves(self) -> None:
        """``play.json``'s ``saves`` follows the mode (absent: ``all``). A directory with no ``play.json`` (a
        world driven without ``alife play``) has nothing to keep it in."""
        obj = self.rd.read_play_json()
        if obj is None:
            return
        kept = {k: v for k, v in obj.items() if k != "saves"}
        self.rd.write_play_json(kept if self.saves == "all" else {**kept, "saves": self.saves})

    def _watch(self, arg: Any) -> None:
        """``watch KIN_ID`` / ``watch off`` (DESIGN §34.3): the lineage alerts read the watched threads."""
        if arg == "off":
            self.alerts.watch_off()
        else:
            assert isinstance(arg, int)
            if (note := self.alerts.watch(arg, self.chron.threads)) is not None:
                self._note(note)

    # --- the player's powers ----------------------------------------------------------------------------

    def _power(self, arg: Any, text: str, state: State) -> None:
        """A ``power`` line (DESIGN §34.5): answered with a ``PLAN`` (a preview, a refusal or the request
        queued; the runner applies a queued one at the top of this iteration, or at the first step after the
        world resumes). Previews and refusals change nothing; a queued request is a trajectory change that is
        logged when it is applied."""
        assert isinstance(arg, PowerLine)
        if self.desk is None:
            return
        outs = self.desk.handle(
            arg,
            text,
            self._t,
            state,
            paused=self.sched.paused,
            patch_pos=to_numpy(state.patch_pos),
            positions=lambda: (to_numpy(state.pos), to_numpy(state.alive)),
        )
        self._send_plans(outs)
        if any(o.plan["status"] == "queued" for o in outs):
            self._send_environment(state)  # the units the request holds

    def _send_plans(self, outs: Sequence[powers_mod.Out]) -> None:
        for o in outs:
            self._send(
                "PLAN",
                lambda step, seq, o=o: streamrec.encode_plan(step=step, seq=seq, obj=o.plan, reply=o.reply),
            )

    def _player_step(self, state: State, t: int) -> None:
        """After a step: what the director applied or refused at its top (``Director.take_player``), the
        chronicle rows, the ``PLAN`` records, ``ENVS`` if the environment changed, and a strike's count in its
        column one step after it was applied."""
        if self.desk is None or self.desk.mode == "off":  # no director (or not one that has a player channel)
            return
        applied, refused = self.director.take_player()
        if applied or refused:
            outs, changed = self.desk.after_step(applied, refused, t - 1)
            if applied:
                self._events += self.chron.from_director(applied)
            self._send_plans(outs)
            if changed:
                self._send_environment(state)
        if self.desk.wants_disc():
            self.desk.set_disc(to_numpy(state.pos), to_numpy(state.alive))

    def _resumed(self, *, acknowledge: bool) -> None:
        """The world runs (or steps): the pause by a chapter or an alert is over. A ``run``, a ``step`` or a
        rate change out of the chapter pause is the player's acknowledgement of the chapter alert."""
        self._pause_reason = 0
        if acknowledge:
            self.alerts.clear_chapter(self._t)
            self._emit_alerts()

    # --- alerts -----------------------------------------------------------------------------------------

    def _armed(self, cls: str) -> bool:
        """Whether a raised alert of ``cls`` pauses the world (``autopause``)."""
        return self.autopause.chapter if cls == "chapter" else self.autopause.alert.get(cls, False)

    def _pause_ok(self) -> bool:
        """Whether an alert raised now can pause the world: not inside a ``step N`` batch, and not when the
        pause at a chapter end is certain to follow (the chapter pause wins and carries the other alerts)."""
        return not self.sched.busy and not (self.chapters.is_end(self._t) and self.autopause.chapter)

    def _emit_alerts(self) -> None:
        """Write what the engine produced: the lines to ``alerts.jsonl`` first (durable before the client
        sees them), then ``ALRT`` records of at most 16 alerts, and note a requested pause."""
        taken = self.alerts.take()
        if taken.log:
            self._log_alerts(taken.log)
        for i in range(0, len(taken.alerts), streamrec.MAX_ALERTS):
            batch = taken.alerts[i : i + streamrec.MAX_ALERTS]
            self._send(
                "ALRT",
                lambda step, seq, batch=batch: streamrec.encode_alrt(
                    step=step, seq=seq, alerts=batch, registry=alerts_mod.CLASSES
                ),
            )
        if taken.pause:
            self._alert_pause_pending = True

    def _log_alerts(self, lines: Sequence[Mapping[str, Any]]) -> None:
        """Append to ``alerts.jsonl``; in a replay of the director's log the lines of the wall-clock class go
        to ``alerts.replay.jsonl`` (they differ from the original play's), the rest are the same lines the
        play wrote."""
        mine = [m for m in lines if not (self._replaying and m.get("class") == "engine")]
        other = [m for m in lines if self._replaying and m.get("class") == "engine"]
        if mine:
            self.rd.append_jsonl(ALERTS_FILE, mine)
        if other:
            self.rd.append_jsonl(ALERTS_REPLAY_FILE, other)

    def _send_alert_backlog(self) -> None:
        """The alerts still on the stack of a world being opened (flagged as the backlog: no toast, no
        pause)."""
        backlog = alerts_mod.backlog(self.rd.read_jsonl(ALERTS_FILE), self._t)
        if backlog:
            self.alerts.know(backlog)  # the player can acknowledge what this session did not raise
            self._send(
                "ALRT",
                lambda step, seq: streamrec.encode_alrt(
                    step=step, seq=seq, alerts=backlog, backlog=True, registry=alerts_mod.CLASSES
                ),
            )

    def _begin_alert_pause(self, state: State) -> None:
        """The first ``before_step`` after an alert asked for a pause (DESIGN §34.3): pause with reason 4,
        unless the world is already paused (the chapter pause wins), runs a ``step N`` batch or stops."""
        self._alert_pause_pending = False
        if self.sched.busy or self.sched.paused or self.stop_requested or self._stop_check():
            return
        self.sched.pause()
        self._enter_pause(state)
        self._pause_reason = 4
        self._send_clock()

    def _territory_alerts(self, t: int) -> None:
        """``storm_over_lineage`` (experimental): the territory of the snapshot just sent against the named
        storms. Only evaluated with ``--alerts experimental``."""
        if not self.opt.alerts_experimental or self._terr is None or self.layer is None:
            return
        self.alerts.on_territory(
            t,
            self._terr,
            self._storms_now,
            self.chron.threads,
            self.cfg.world.size,
            pause_ok=self._pause_ok(),
        )

    def _enter_pause(self, state: State) -> None:
        if self._last_frame_step != self._t:
            self._send_frame(state)

    def _chapter_pause_follows(self) -> bool:
        """Whether the next ``before_step`` will pause the world for the chapter that just ended."""
        return (
            self._chapter_pending
            and self.autopause.chapter
            and not self.sched.busy
            and not self.stop_requested
            and not self._stop_check()
        )

    def _begin_chapter_pause(self, state: State) -> None:
        """The first ``before_step`` after a chapter end (DESIGN §34.2): pause if the chapter pause is armed.
        A ``step N`` batch never pauses by itself, and a pause the player already made only changes its
        reason; with the pause off the world runs on and the chapter card is a notification."""
        self._chapter_pending = False
        if self.sched.busy or not self.autopause.chapter or self.stop_requested or self._stop_check():
            return
        if not self.sched.paused:
            self.sched.pause()
            self._enter_pause(state)
        self._pause_reason = 3
        self._send_clock()

    # --- the achieved rate and the frame period ---------------------------------------------------------

    def _achieved(self, now: float) -> float:
        """Steps per second over the last ``ACHIEVED_S`` seconds."""
        while self._window and self._window[0][0] < now - ACHIEVED_S:
            self._window.popleft()
        if len(self._window) < 2:
            return 0.0
        (t0, n0), (t1, n1) = self._window[0], self._window[-1]
        span = max(t1 - t0, 1e-9)
        # the window ends now, not at the last step: a pause lets the rate decay
        return (n1 - n0) / max(span, min(now - t0, ACHIEVED_S))

    def _effective_fe(self) -> int:
        return self.fe_override if self.fe_override is not None else self._fe

    def _service(self) -> None:
        """Every ``WAIT_S``: a ``CLCK`` (and the frame period from the achieved rate), and a look at the
        reader."""
        now = self.clock.monotonic()
        if reader_closed(self.out):
            raise BrokenPipeError("the reader closed the protocol stream")
        if now - self._last_clck >= WAIT_S:
            achieved = self._achieved(now)
            requested = self.sched.steps_per_s
            r_eff = achieved if math.isinf(self.sched.rate) else min(requested, achieved)
            self._fe = max(1, math.ceil(r_eff / self.opt.frame_hz - EPS))
            self.alerts.engine_check(
                now,
                self._t,
                late_steps=self.sched.late_steps,
                achieved=achieved,
                requested=requested,
                running=self.sched._timed(),
            )
            self._emit_alerts()
            self._send_clock()

    # --- records ----------------------------------------------------------------------------------------

    def _send_clock(self) -> None:
        now = self.clock.monotonic()
        self._last_clck = now
        reason = (
            5
            if self._dead
            else 2
            if self._ended
            else 3
            if self._chapter_pending or self._pause_reason == 3
            else 4
            if self._pause_reason == 4
            else 1
            if (self.sched.paused and not self.sched.busy)
            else 0
        )
        chapter, in_chapter = self.chapters.position(self._t)
        steps_per_s = self._achieved(now)
        self._send(
            "CLCK",
            lambda step, seq: streamrec.encode_clck(
                step=step,
                seq=seq,
                rate=self.sched.requested(),
                achieved=steps_per_s,
                step_ms=statistics.median(self._step_ms) if self._step_ms else 0.0,
                frame_ms=statistics.median(self._frame_ms) if self._frame_ms else 0.0,
                late_steps=self.sched.late_steps,
                save_stall_steps=self.sched.save_stall_steps,
                reason=reason,
                saving=self._saving,
                fe=self._effective_fe(),
                chapter=chapter,
                in_chapter=in_chapter,
                chapter_len=self.opt.chapter,
                flags=self.autopause.flags
                | (streamrec.CLCK_FLAG_SAVES_NEWEST if self.saves != "all" else 0)
                | (streamrec.CLCK_FLAG_SAVES_SINGLE if self.saves == "single" else 0),
            ),
        )

    def _send_frame(self, state: State) -> None:
        """One ``FRMA`` of ``state`` in the current colour mode: one device evaluation, one host transfer."""
        t0 = self.clock.monotonic()
        cfg, mode = self.cfg, self.color
        inputs = self.obs.color_inputs(self._house_top) if self.obs is not None else None
        colors = play_colors(state, cfg, mode, inputs)
        pitch = state.pitch if hasattr(state, "pitch") else None
        dev = {
            "alive": state.alive,
            "pos": state.pos,
            "yaw": state.heading,
            "partner": state.partner,
            "uid": state.uid,
            "diag": state.diag,
            "rgb": colors,
            **({"pitch": pitch} if pitch is not None else {}),
        }
        mx.eval(*dev.values())
        h = {k: to_numpy(v) for k, v in dev.items()}
        alive = h["alive"].astype(bool)
        n = alive.shape[0]
        yaw = h["yaw"].astype(np.float32)
        pitch_a = h["pitch"].astype(np.float32) if "pitch" in h else np.zeros_like(yaw)
        partner = h["partner"]
        rows = np.arange(n)[:, None]
        drawn = (partner > rows) & alive[:, None] & alive[np.maximum(partner, 0)]
        ii, bb = np.nonzero(drawn)  # row-major: ascending row, then slot
        pairs = np.stack([ii, partner[ii, bb]], axis=1)
        span = 0 if self._last_frame_step < 0 else self._t - self._last_frame_step
        self._send(
            "FRMA",
            lambda step, seq: streamrec.encode_frma(
                step=step,
                seq=seq,
                alive=alive,
                pos=h["pos"],
                orient=np.stack([yaw, pitch_a], axis=1),
                rgb=h["rgb"],
                uid=h["uid"],
                bonds=pairs,
                color_index=streamrec.COLOR_MODES.index(mode),
                span=span,
                diag_error=bool(int(h["diag"]) & ERROR_MASK),
            ),
        )
        self._last_frame_step = self._t
        self._frames_written += 1
        self._frame_ms.append((self.clock.monotonic() - t0) * 1e3)

    def _send_tables(self, state: State, *, labels: bool = True) -> None:
        """``IDNT``, ``INTN`` (with ``labels``, unless switched off) and ``TERR`` now: the observer's
        identities, labels and territory."""
        obs = self.obs
        if obs is None:
            return
        ident = obs.identity_rows(state)
        self._tables = self._thread_names(ident.tables)
        self._house_top = np.array([h["id"] for h in ident.tables["houses"]], np.int64)
        self._send(
            "IDNT",
            lambda step, seq: streamrec.encode_idnt(
                step=step,
                seq=seq,
                snapshot_step=ident.snapshot_step,
                uid=ident.uid,
                clade=ident.clade,
                kin=ident.kin,
                house=ident.house,
                colony=ident.colony,
                flags=ident.flags,
                tables=self._tables,
            ),
        )
        if labels and self.labels_on:
            self._send_labels(state)
        dom, ids = obs.territory()
        if dom is not None:
            self._send_territory(dom, ids, ident.snapshot_step)

    def _thread_names(self, tables: Mapping[str, Any]) -> dict[str, Any]:
        """The ``IDNT`` tables with each kin group's name as its thread is shown (frozen at the first listing,
        with a suffix where two threads share a name) and ``thread``, the thread's id: the Cast keeps one name
        and one swatch across a re-formation under a new kin id. A group the thread table does not know
        (before the first record) keeps its name and has no ``thread``."""
        names = self.chron.cast_names()
        kin = [
            {**k, "name": names[k["id"]][1], "thread": names[k["id"]][0]} if k["id"] in names else k
            for k in tables["kin"]
        ]
        return {**tables, "kin": kin}

    def _send_territory(self, dom: Any, ids: np.ndarray, snapshot_step: int) -> None:
        nx = ny = self.obs.social.terr_cells  # type: ignore[union-attr]
        listed = [int(i) for i in ids]
        slot_of = {k: s for s, k in enumerate(listed) if k >= 0}
        self._terr = np.asarray(dom.owner, np.int64).reshape(ny, nx)
        owner = np.array([slot_of.get(int(o), streamrec.SLOT_NONE) for o in dom.owner], np.uint8)
        purity = np.rint(np.asarray(dom.purity_q16, np.float64) / 257.0).astype(np.uint8)
        self._send(
            "TERR",
            lambda step, seq: streamrec.encode_terr(
                step=step,
                seq=seq,
                snapshot_step=snapshot_step,
                ids=listed,
                owner=owner.reshape(ny, nx),
                purity=purity.reshape(ny, nx),
            ),
        )

    def _send_labels(self, state: State) -> None:
        obs = self.obs
        if obs is None:
            return
        lb, held = obs.labels(), obs.o.held
        arrays = (state.uid, lb.arch_primary, lb.arch_secondary, lb.intent, lb.intent_match, held)
        mx.eval(*arrays)
        uid, a1, a2, intent, match, held_a = (to_numpy(a) for a in arrays)
        rows = np.flatnonzero(uid >= 0)
        flags = ((match[rows] & EVAL_BIT) != 0).astype(np.uint8) | (
            (intent[rows] != held_a[rows]).astype(np.uint8) << 1
        )
        every = obs.rules.every
        self._send(
            "INTN",
            lambda step, seq: streamrec.encode_intn(
                step=step,
                seq=seq,
                labels_step=self._t - self._t % every,
                intent_version=INTENT_VERSION,
                uid=uid[rows],
                arch_primary=a1[rows],
                arch_secondary=a2[rows],
                intent=intent[rows],
                flags=flags,
            ),
        )

    def _shade(self, state: State) -> dict[str, mx.array]:
        """``{"t_mult": shade}`` for the ``MAP1`` sea-temperature plane when the world shades its warm pools
        (``weather.shade_gain > 0``, static), else no keyword. ``state`` is the state at step ``t``: the
        start-of-step state of the step that will force the weather with this ``T_s`` (DESIGN §33.1.7)."""
        if self.cfg.weather.shade_gain > 0:
            return {"t_mult": W.shade_mult(state.pos, state.alive, self.cfg)}
        return {}

    def _tick_storms(self, state: State, t: int) -> None:
        """The storm layer's tick at ``t``: detect, track, log, then the weather records."""
        layer = self.layer
        assert layer is not None
        tick = layer.tick_at(W.unpack(state.wx), t, **self._shade(state))
        layer.log(self.rd, tick)
        detected = [d["id"] for d in tick.line["storms"]] if tick.line is not None else []
        self._events += self.chron.from_storms(t, tick.events, detected)
        self._storms_now = tick.line["storms"] if tick.line is not None else ()
        self.alerts.on_tick(
            t,
            self._storms_now,
            tick.events,
            self.chron.told_storm,
            pause_ok=self._pause_ok(),
        )
        self._write_weather(tick)

    def _write_weather(self, tick: Any) -> None:
        dc_ref = self.layer.dc_ref  # type: ignore[union-attr]
        if tick.planes is not None:
            self._send(
                "MAP1",
                lambda step, seq: streamrec.encode_map1(
                    step=step, seq=seq, planes=tick.planes, dc_ref=dc_ref
                ),
            )
        self._send(
            "STRM",
            lambda step, seq: streamrec.encode_strm(step=step, seq=seq, tick=self._tick, obj=tick.strm),
        )

    def _send_weather(self, state: State) -> None:
        if self.layer is None:
            return
        self._write_weather(self.layer.current(W.unpack(state.wx), self._t, **self._shade(state)))

    def _send_environment(self, state: State) -> None:
        obj = environment_object(
            state, self.cfg, self.director, self._t, self.layer.p if self.layer else None, desk=self.desk
        )
        self._send("ENVS", lambda step, seq: streamrec.encode_envs(step=step, seq=seq, obj=obj))

    def _flush_events(self, *, backlog: bool = False) -> None:
        """Write the pending events, in batches of at most 64 (an ``EVNT`` per batch; the backlog flag on the
        first records of a stream)."""
        events, self._events = self._events, []
        if backlog:
            events = events[-MAX_EVENTS_PER_RECORD:]
            self._send(
                "EVNT",
                lambda step, seq: streamrec.encode_evnt(step=step, seq=seq, events=events, backlog=True),
            )
            return
        for i in range(0, len(events), MAX_EVENTS_PER_RECORD):
            batch = events[i : i + MAX_EVENTS_PER_RECORD]
            self._send(
                "EVNT", lambda step, seq, batch=batch: streamrec.encode_evnt(step=step, seq=seq, events=batch)
            )

    def _extinct(self, state: State, out: StepOut) -> bool:
        cfg = self.cfg
        if immigration_on(cfg.population) or int(to_numpy(out.counters)[CounterIdx.N_ALIVE]) != 0:
            return False
        return not (cfg.director.enabled and int(to_numpy(state.env_imm_request)) > 0)

    # --- focus and follow -------------------------------------------------------------------------------

    def _command_focus(self, cmd: PlayCommand, state: State) -> None:
        if cmd.arg == "off":
            if cmd.name == "follow":
                self.follow = False
            else:
                self.focus, self.follow, self._followed = None, False, None
            return
        assert isinstance(cmd.arg, int)
        uid = cmd.arg
        row = self._find_row(state, uid)
        if row is None:  # unknown or dead: one FOCS with the alive flag clear
            self._send_focus(None, uid)
            self.focus, self.follow, self._followed = None, False, None
            return
        self.focus, self.follow = uid, cmd.name == "follow"
        r = self._read_row(state, row)
        self._followed = self._cache(uid, row, r)
        self._send_focus(r, uid, row=row)
        self._last_focs = self.clock.monotonic()

    def _find_row(self, state: State, uid: int) -> int | None:
        """The row holding ``uid``, or None: one device compare."""
        match = state.uid == mx.array(uid, dtype=mx.int64)
        found, row = match.any(), mx.argmax(match)
        mx.eval(found, row)
        return int(to_numpy(row)) if bool(to_numpy(found)) else None

    def _read_row(self, state: State, row: int) -> dict[str, np.ndarray]:
        """One row of the state and of the observer: a handful of scalars in one small device read."""
        obs = self.obs
        items: dict[str, mx.array] = {
            "uid": state.uid[row],
            "pos": state.pos[row],
            "vel": state.vel[row],
            "energy": state.energy[row],
            "birth": state.birth_step[row],
            "generation": state.generation[row],
            "parent": state.parent_uid[row],
            "bonds": (state.partner[row] >= 0).sum(),
            **(
                {
                    "house": obs.so.house[row],
                    "kin": obs.so.kin[row],
                    "colony": obs.so.colony[row],
                    "arch": obs.o.label_arch[row],
                    "intent": obs.o.label_intent[row],
                    "held": obs.o.held[row],
                    "match": obs.o.match_intent[row],
                }
                if obs is not None
                else {}
            ),
        }
        mx.eval(*items.values())
        return {k: to_numpy(v) for k, v in items.items()}

    @staticmethod
    def _cache(
        uid: int, row: int, r: Mapping[str, np.ndarray], kids: frozenset[int] = frozenset()
    ) -> _Followed:
        return _Followed(
            uid,
            row,
            tuple(np.asarray(r["pos"], np.float64).ravel().tolist()),
            int(r["kin"]) if "kin" in r else -1,
            kids,
        )

    def _send_focus(
        self, r: Mapping[str, np.ndarray] | None, uid: int, *, handoff: bool = False, row: int = 0
    ) -> None:
        """``FOCS`` of the row read ``r`` (None: the organism is not alive)."""
        none = streamrec.SLOT_NONE
        if r is None:
            f: dict[str, Any] = {
                "flags": 0, "pos": (0.0, 0.0, 0.0), "vel": (0.0, 0.0, 0.0), "energy": 0, "age": 0,
                "generation": 0, "parent_uid": -1, "n_bonds": 0, "house": none, "kin": none, "colony": none,
                "arch_primary": 0, "arch_secondary": 0, "intent": 0, "label_flags": 0,
            }  # fmt: skip
            row = 0
        else:
            obs = self.obs
            slots = obs.social.slots if obs is not None else 0
            f = {
                "flags": streamrec.FOCS_ALIVE
                | (streamrec.FOCS_HANDOFF if handoff else 0)
                | (streamrec.FOCS_FOLLOWED if self.follow else 0),
                "pos": _pad3(r["pos"]), "vel": _pad3(r["vel"]), "energy": int(r["energy"]),
                "age": max(self._t - int(r["birth"]), 0), "generation": int(r["generation"]),
                "parent_uid": int(r["parent"]), "n_bonds": int(r["bonds"]), "house": none, "kin": none,
                "colony": none, "arch_primary": 0, "arch_secondary": 0, "intent": 0, "label_flags": 0,
            }  # fmt: skip
            if obs is not None:
                f = {
                    **f,
                    "house": int(house_bytes(np.array([int(r["house"])], np.int64), self._house_top)[0]),
                    "kin": int(slot_bytes(np.array([int(r["kin"])]), slots)[0]),
                    "colony": int(slot_bytes(np.array([int(r["colony"])]), slots)[0]),
                    "arch_primary": int(r["arch"][0]),
                    "arch_secondary": int(r["arch"][1]),
                    "intent": int(r["intent"]),
                    "label_flags": int((int(r["match"]) & EVAL_BIT) != 0)
                    | (int(r["intent"] != r["held"]) << 1),
                }
        self._send("FOCS", lambda step, seq: streamrec.encode_focs(step=step, seq=seq, uid=uid, row=row, **f))

    def _follow_step(self, state: State, out: StepOut, t: int) -> None:
        """After a step: report the focused organism (about 4 per second; every step while followed). While
        followed, keep the cache of its last known row and children and hand the follow on at its death."""
        if self.focus is None:
            return
        now = self.clock.monotonic()
        if not self.follow and now - self._last_focs < FOCS_PERIOD_S:
            return
        u, cached = self.focus, self._followed
        row = cached.row if cached is not None and cached.uid == u else self._find_row(state, u)
        r = self._read_row(state, row) if row is not None else None
        alive = r is not None and int(r["uid"]) == u  # the row may hold another organism, or none
        self._last_focs = now
        kids = (cached.children if cached is not None else frozenset()) | (
            self._children_of(u, out) if self.follow else frozenset()
        )
        if alive:
            assert r is not None and row is not None
            self._followed = self._cache(u, row, r, kids)
            self._send_focus(r, u, row=row)
        elif self.follow and cached is not None:
            self._hand_off(state, out, t, dataclasses.replace(cached, children=kids))
        else:
            self._send_focus(None, u)
            self.focus, self._followed = None, None

    def _children_of(self, uid: int, out: StepOut) -> frozenset[int]:
        """The uids born to ``uid`` at this step (the lanes of ``out.births``)."""
        b = out.births
        valid, parent, child = (to_numpy(a) for a in (b.valid, b.parent_uid, b.uid))
        return frozenset(int(c) for c in child[valid.astype(bool) & (parent == uid)])

    def _hand_off(self, state: State, out: StepOut, t: int, dead: _Followed) -> None:
        """The followed organism died at step ``t``: say so and pass the follow on (or end it)."""
        d = out.deaths
        valid, uids, causes = (to_numpy(a) for a in (d.valid, d.uid, d.cause))
        hit = np.flatnonzero(valid.astype(bool) & (uids == dead.uid))
        cause = DeathCause(int(causes[hit[0]])).name.lower() if hit.size else None  # None: lanes overflowed
        found = self._successor(state, dead)
        if found is None:
            self._events += self.chron.focus_death(t, dead.uid, cause, dead.pos)
            self._send_focus(None, dead.uid)
            self.focus, self._followed = None, None
            return
        new_uid, row, rule = found
        self._events += self.chron.focus_handoff(t, dead.uid, new_uid, rule, dead.pos, cause)
        r = self._read_row(state, row)
        self.focus, self._followed = new_uid, self._cache(new_uid, row, r)
        self._send_focus(r, new_uid, handoff=True, row=row)

    def _successor(self, state: State, dead: _Followed) -> tuple[int, int, int] | None:
        """The organism the follow passes to, ``(uid, row, rule)``: (1) the eldest living child; else (2) the
        nearest living one of the same kin slot to the death point; else (3) the nearest living one; else
        None."""
        if dead.children:
            kids = sorted(dead.children)
            alive = (state.uid[:, None] == mx.array(kids, dtype=mx.int64)[None, :]).any(axis=0)
            mx.eval(alive)
            hits = np.flatnonzero(to_numpy(alive))
            if hits.size:
                uid = kids[int(hits[0])]
                row = self._find_row(state, uid)
                if row is not None:
                    return uid, row, 1
        length = self.cfg.world.size
        delta = state.pos - mx.array(np.asarray(dead.pos, np.float32))
        xy = delta[:, :2] - length * mx.round(delta[:, :2] / length)
        dist2 = (xy * xy).sum(axis=1) + ((delta[:, 2] * delta[:, 2]) if delta.shape[1] > 2 else 0.0)
        living = state.uid >= 0
        same = living & (self.obs.so.kin == dead.kin) if self.obs is not None and dead.kin >= 0 else None
        big = mx.array(3.0e38, dtype=mx.float32)
        for rule, mask in ((2, same), (3, living)):
            if mask is None:
                continue
            row_a, any_a = mx.argmin(mx.where(mask, dist2, big)), mask.any()
            mx.eval(row_a, any_a, state.uid)
            if bool(to_numpy(any_a)):
                row = int(to_numpy(row_a))
                return int(to_numpy(state.uid)[row]), row, rule
        return None


def _pad3(a: Any) -> tuple[float, float, float]:
    """A 2- or 3-vector as three floats (z = 0 in 2D)."""
    v = np.asarray(a, np.float64).ravel().tolist()
    return (v[0], v[1], v[2] if len(v) > 2 else 0.0)


# --- the origin row ---------------------------------------------------------------------------------------


def planet_origin_step(planet: Mapping[str, Any] | None) -> int:
    """The step a world's chapter numbering starts after: ``made_at_step`` of the ``planet`` block of its
    ``play.json`` (written by ``alife play`` for a planet, §33.3.3), 0 for a world with no planet (a preset
    played from step 0, an adopted run) and for a ``play.json`` whose planet has no usable ``made_at_step``
    (not an integer, or not positive). The 0 is also what the client reads from such a file
    (``ChapterOrigin``, §34.7), so the two numberings agree whatever the file holds."""
    made = (planet or {}).get("made_at_step")
    return made if isinstance(made, int) and not isinstance(made, bool) and made > 0 else 0


def origin_row(state: State, cfg: Config) -> dict[str, Any] | None:
    """The row a new world has at its start step (``meta.json["origin_row"]``, §34.2): the first chapter of a
    world is cut from it, because N1 needs a metrics row at both ends of a chapter and a world has none at the
    step it starts from. ``{step, steps: 0, origin: true, observers: {alive}, clades_alive,
    top_clade_share}``, from one ``snapshot_metrics`` and a count of the live organisms; no social head (the
    observer has taken no snapshot at its start step), so the facts that need one are dropped from that
    report.

    None when there can be no such cut: the world has no observers, or its start step is off the snapshot grid
    (its first, partial chapter is an arrival chapter with no report)."""
    step = int(to_numpy(state.step))
    if not cfg.game.observe or step % observer_mod.DEFAULT_SOCIAL.snapshot_every:
        return None
    snap = metrics.snapshot_metrics(state, cfg, heavy=True)
    return {
        "step": step,
        "steps": 0,
        "origin": True,
        "observers": {"alive": int(to_numpy(state.alive).sum())},
        "clades_alive": snap["clades_alive"],
        "top_clade_share": snap["top_clade_share"],
    }


# --- ENVS -----------------------------------------------------------------------------------------------


def environment_object(
    state: State,
    cfg: Config,
    director: Any,
    t: int,
    params: WeatherParams | None,
    desk: powers_mod.PowerDesk | None = None,
) -> dict[str, Any]:
    """The ``ENVS`` object (DESIGN §33.1.9): the warm pools at step ``t``, the season in force, the levers the
    director has moved from their inert values, and the director's effects and cooldowns folded from its log
    (empty without a director). A world that takes the player's powers (``desk``) adds the influence, the food
    patches and the state of each power (DESIGN §34.6)."""
    pools = []
    if params is not None:
        length = params.L
        pools = [
            {
                "x": round((x0 + vx * t) % length, 2),
                "y": round((y0 + vy * t) % length, 2),
                "amp": amp,
                "sx": sx,
                "sy": sy,
                "vx": vx,
                "vy": vy,
            }
            for x0, y0, sx, sy, amp, vx, vy in params.pools
        ]
    inert = {k.removeprefix("env_"): np.asarray(to_numpy(v)) for k, v in default_env(cfg).items()}
    mx.eval(*(getattr(state, f"env_{k}") for k in inert))
    now = {k: np.asarray(to_numpy(getattr(state, f"env_{k}"))) for k in inert}
    levers = {
        k: (v.tolist() if v.ndim else v.item()) for k, v in now.items() if not np.array_equal(v, inert[k])
    }
    period, amp = int(now["season_period"]), float(now["season_amp"])
    season = (
        {"period": period, "amp": round(amp, 6), "phase": round((t % period) / period, 6)}
        if period > 0 and amp > 0.0
        else None
    )
    effects: list[dict[str, Any]] = []
    cooldowns: list[dict[str, Any]] = []
    if director is not None:
        entries = list(director.entries)
        # the rule tier's cooldowns (a player's use of an action cools only the player's); the effects held
        # are every scope's
        folded = pacing.fold(entries, t, "rule")
        effects = _effects(entries, folded)
        cooldowns = [
            {"action": a, "ready": end + rules.cooldown_steps(a, director.params, director.persona)}
            for a, end in sorted(folded.ends.items())
            if end + rules.cooldown_steps(a, director.params, director.persona) > t
        ]
    player: dict[str, Any] = (
        desk.env_block(t, to_numpy(state.patch_pos), now["patch_gain"]) if desk is not None else {}
    )  # influence, patches and powers: only a world that takes powers has them
    return {
        "pools": pools,
        "season": season,
        "levers": levers,
        "effects": effects,
        "cooldowns": cooldowns,
        "influence": player.get("influence"),
        **{k: v for k, v in player.items() if k != "influence"},
    }


def replaying(director: Any) -> bool:
    """Whether the world plays the director's log back (``play --replay``, ``Director.mode`` ``replay``)."""
    return getattr(director, "mode", None) == "replay"


def _effects(entries: Sequence[Mapping[str, Any]], folded: pacing.Pacing) -> list[dict[str, Any]]:
    """The timed effects a fold holds, as ``ENVS.effects`` objects (every scope's). An effect is its entry:
    the forward entry of its action at its step that writes its lever (two habitat losses on two patches at
    one step are two effects, each with its own patch and source)."""
    forward = [e for e in entries if e.get("source") != "revert"]

    def entry_of(eff: pacing.Effect) -> Mapping[str, Any] | None:
        return next(
            (
                e
                for e in forward
                if e["action"] == eff.action
                and e["step_applied"] == eff.origin
                and eff.key in director_actions.lever_keys(e["action"], e["params"])
            ),
            None,
        )

    out = []
    for eff in sorted(folded.active.values(), key=lambda e: (e.origin, e.key)):
        entry = entry_of(eff)
        if entry is not None:
            out.append(
                {
                    "action": eff.action,
                    "params": entry["params"],
                    "start": eff.origin,
                    "end": eff.end,
                    "source": _source(entry["source"]),
                }
            )
    return out


def director_effects(director: Any, t: int) -> list[dict[str, Any]]:
    """The timed effects the director's log holds at ``t`` (the alert engine's ``effect_ending`` reads
    them); none without a director."""
    if director is None:
        return []
    entries = list(director.entries)
    return _effects(entries, pacing.fold(entries, t, "rule"))


def _source(source: str) -> str:
    """The entry's source as the stream names it: ``rule``, ``llm`` or ``player`` (any other is a rule)."""
    return source if source in ("rule", "llm", "player") else "rule"


# --- the process ------------------------------------------------------------------------------------------


def serve_play(
    rd: RunDir,
    cfg: Config,
    box: list[State],
    *,
    steps: int,
    options: PlayOptions,
    control: Control,
    out: BinaryIO,
    director: Any = None,
    clock: Clock | None = None,
) -> int:
    """Play the world in ``box`` for ``steps`` steps under a ``PlayPacer`` and return the process exit code.

    The runner holds the only reference to the state (``box`` is popped here). Every way the play can end has
    taken the runner's ordinary stop path first (checkpoint, event chunk, status), so the exit code only says
    why: 0 for a ``quit``, an end of the control channel, a closed reader, the ``until`` step and an extinct
    world (which ends before the header, with a note on stderr); 130 for a SIGINT; 1 when the presenter failed
    (after the checkpoint); 3 for an invariant violation. ``control`` is closed on the way out.
    """
    pacer = PlayPacer(out, control, options, clock=clock)
    rd.set_save_mode(options.saves)  # (also for the starting checkpoint, which precedes bind)
    rd.compress = options.compress
    try:
        res = runner.run(rd, cfg, box.pop(), steps, quiet=True, director=director, pacer=pacer)
    finally:
        control.close()
        pacer.close()
        rd.wait_saves()  # the quit path waits for a compressed save in flight (DESIGN §34.11)
    if res.status == "extinct":
        print(
            f"play: the world is extinct at step {int(to_numpy(res.state.step))}", file=sys.stderr, flush=True
        )
        return int(runner.ExitCode.OK)
    if pacer.failure is not None and not isinstance(pacer.failure, CLOSED_READER):
        return int(runner.ExitCode.ERROR)
    if res.status == "violation":
        return int(res.exit_code)
    if res.status == "interrupted" and not pacer.stop_requested:
        return int(runner.ExitCode.INTERRUPTED)  # a SIGINT
    return int(runner.ExitCode.OK)
