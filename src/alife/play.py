"""``alife play``: the scheduler, the control grammar and the presenter of a played world (DESIGN §33.2).

Host module. A played world is an ordinary run (``runner.run``) with a ``PlayPacer`` hooked into its loop. The
pacer does two things and nothing else:

- **Schedule.** ``Scheduler`` decides when the next step may start (pause, ``step N`` batches, a rate of 0.25x
  to 2x of the base rate, or ``max``). It reads a ``Clock`` only to decide *when*; the step index, the order
  of the observer and director calls and every written log line are functions of the step count, so pacing
  never reaches the trajectory.
- **Present.** After a step it writes the records of §33.1 (``FRMA``, ``IDNT``, ``INTN``, ``TERR``, ``MAP1``,
  ``STRM``, ``EVNT``, ``ENVS``, ``CLCK``, ``FOCS``) to the protocol stream. The records are built from arrays
  it evaluates and converts to numpy before it returns, from the observer's accessors and from the storm
  layer; it holds no reference to the runner's state (the donation rule: the runner is the only holder when it
  steps).

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
import time
import traceback
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, BinaryIO, Final, Protocol

import mlx.core as mx
import numpy as np

from alife import layout, runner, streamrec
from alife import weather as W
from alife.config import Config, config_hash, dims, immigration_on
from alife.director import log as director_log
from alife.director import pacing, rules
from alife.game.chronicle import Chronicler, rebuild_backlog
from alife.game.colors import OBSERVER_MODES, play_colors
from alife.game.observe import EVAL_BIT
from alife.game.observer import Observer, default_rules, house_bytes, slot_bytes
from alife.game.report import label_gate, label_names
from alife.game.rules import INTENT_VERSION
from alife.runio import DIRECTOR_LOG_FILE, RunDir
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
    "Clock",
    "PlayCommand",
    "PlayOptions",
    "PlayPacer",
    "RealClock",
    "Scheduler",
    "check_tick",
    "parse_play_command",
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
)
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
class PlayCommand:
    """One control line, parsed. ``arg``: an int (``step``, ``every``, ``focus``, ``follow``), a float
    (``rate`` M, 0 pauses), a string (``rate`` ``max``, ``color`` MODE, ``every`` ``auto``, ``focus`` /
    ``follow`` ``off``, ``labels`` ``on`` / ``off``) or None."""

    name: str
    arg: int | float | str | None = None


def _uint(text: str, what: str) -> int:
    if not _UINT.fullmatch(text):
        raise ControlError(f"{what} must be a non-negative integer, got {text!r}")
    return int(text)


def parse_play_command(line: str, *, modes: Sequence[str] = streamrec.COLOR_MODES) -> PlayCommand | None:
    """One control line of ``alife play`` (None for a blank line or a ``#`` comment; ``ControlError`` if
    invalid).

    ``pause`` | ``run`` | ``step [N]`` | ``rate (M | max)`` | ``color MODE`` | ``every (K | auto)`` | ``focus
    (UID | off)`` | ``follow (UID | off)`` | ``idnt`` | ``labels (on | off)`` | ``quit``; lower case,
    whitespace separated. ``N`` >= 1; ``M`` a decimal, 0 for a pause or in [0.05, 16]; ``K`` in 1..64; ``UID``
    a non-negative int64. ``fields`` is the Lab's and refused here."""
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
    arity(1, 1)  # labels
    if args[0] not in ("on", "off"):
        raise ControlError("labels takes 'on' or 'off'")
    return PlayCommand(name, args[0])


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

    def end_save(self) -> None:
        """The checkpoint is written: the steps the stall skipped are ``save_stall_steps`` (never late) and
        the schedule starts again from now."""
        if self._timed():
            self.save_stall_steps += max(self._owed(self.clock.monotonic()) - self._owed_at_save, 0)
            self._reanchor()

    def _timed(self) -> bool:
        return not self.paused and not self.busy and not math.isinf(self.rate)


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
    ``play.json`` there); ``until`` is a step at which the stream says "end of steps"."""

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
        first = not self._dead
        if first or (not quiet and isinstance(self.failure, CLOSED_READER)):
            self.failure = exc
        self._dead = True
        self.stop_requested = True
        self.out = _NullSink()
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
        try:
            write_all(self.out, data)
        except CLOSED_READER as e:
            self._fail(e, quiet=True)
            return
        self.bytes_by_tag = {**self.bytes_by_tag, tag: self.bytes_by_tag.get(tag, 0) + len(data)}

    def _send(self, tag: str, make: Callable[[int, int], bytes]) -> None:
        """Write the record ``make(step, seq)`` of ``tag`` at the current step."""
        self._write(tag, make(self._t, self.seq.take(tag)))

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
        records = [
            t
            for t in streamrec.TAGS
            if t in ("FRMA", "EVNT", "ENVS", "CLCK", "FOCS")
            or (t in ("IDNT", "INTN", "TERR") and obs is not None)
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
            chapter=CHAPTER,
        )
        self._write("HEAD", streamrec.play_header_bytes(legacy, play))
        self._backlog(rd, state)
        self._send_frame(state)
        self._send_tables(state)
        self._send_environment(state)
        self._send_weather(state)
        self._flush_events(backlog=True)
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
        log = rebuild_backlog(
            rd.social_records(),
            rd.storms_lines(),
            entries,
            house_salt=self.chron.house_salt,
            chronicler=self.chron,
        )
        self._events = [*log, *self._events]
        self._restore_threads(rd)

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

    @_guard
    def before_step(self, state: State, t: int) -> None:
        self._t = t
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
        if social and self.obs is not None:
            self._send_tables(
                state, labels=False
            )  # a snapshot: identities and territory (INTN follows with the tick)
        if t % self._tick == 0:
            if self.layer is not None:
                self._tick_storms(state, t)
            if self.obs is not None and self.labels_on:
                self._send_labels(state)
        self._follow_step(state, out, t)
        batch_end = self.sched.batch_done
        self.sched.batch_done = False
        if batch_end or (not self.sched.paused and t % self._effective_fe() == 0):
            self._send_frame(state)
        if self.opt.until is not None and t >= self.opt.until:
            self._ended = True
        if self._extinct(state, out):
            self._events += self.chron.extinct(t)
        self._flush_events()
        self._service()

    @_guard
    def before_checkpoint(self, state: State, t: int) -> None:
        self._t = t
        self._saving = True
        self.sched.begin_save()
        if self.layer is not None:
            self.layer.save(self.rd, t)  # before the state file, as the observer sidecar is
        if self.obs is not None:
            self.rd.write_play_state("kinthreads", t, self.chron.threads.state())
        self._send_clock()

    @_guard
    def after_checkpoint(self, t: int) -> None:
        self._t = t
        self._saving = False
        self.sched.end_save()
        self._events += self.chron.checkpoint(t)
        self._flush_events()
        self._send_clock()

    @_guard
    def after_boundary(self, state: State, row: dict[str, Any], t: int) -> None:
        self._t = t
        if self.director is not None and self.director.applied:
            self._events += self.chron.from_director(self.director.applied)
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
            self._command(cmd, state)
        return False

    def _command(self, cmd: PlayCommand, state: State) -> None:
        sched, name, arg = self.sched, cmd.name, cmd.arg
        if name == "pause":
            was_running = not sched.paused
            sched.pause()
            if was_running:
                self._enter_pause(state)
            self._send_clock()
        elif name == "run":
            sched.run()
            self._send_clock()
        elif name == "step":
            assert isinstance(arg, int)
            sched.step(arg)
            self._send_clock()
        elif name == "rate":
            was_running = not sched.paused
            sched.set_rate(MAX_RATE if arg == "max" else float(arg))  # type: ignore[arg-type]
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
        else:  # focus, follow
            self._command_focus(cmd, state)

    def _enter_pause(self, state: State) -> None:
        if self._last_frame_step != self._t:
            self._send_frame(state)

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
            self._send_clock()

    # --- records ----------------------------------------------------------------------------------------

    def _send_clock(self) -> None:
        now = self.clock.monotonic()
        self._last_clck = now
        reason = (
            5 if self._dead else 2 if self._ended else 1 if (self.sched.paused and not self.sched.busy) else 0
        )
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
                chapter=self._t // CHAPTER,
                in_chapter=self._t % CHAPTER,
                chapter_len=CHAPTER,
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
            state, self.cfg, self.director, self._t, self.layer.p if self.layer else None
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


# --- ENVS -----------------------------------------------------------------------------------------------


def environment_object(
    state: State, cfg: Config, director: Any, t: int, params: WeatherParams | None
) -> dict[str, Any]:
    """The ``ENVS`` object (DESIGN §33.1.9): the warm pools at step ``t``, the season in force, the levers the
    director has moved from their inert values, and the director's effects and cooldowns folded from its log
    (empty without a director)."""
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
        by_origin = {(e["action"], e["step_applied"]): e for e in entries if e.get("source") != "revert"}
        effects = [
            {
                "action": eff.action,
                "params": by_origin[(eff.action, eff.origin)]["params"],
                "start": eff.origin,
                "end": eff.end,
                "source": _source(by_origin[(eff.action, eff.origin)]["source"]),
            }
            for eff in sorted(folded.active.values(), key=lambda e: (e.origin, e.key))
            if (eff.action, eff.origin) in by_origin
        ]
        cooldowns = [
            {"action": a, "ready": end + rules.cooldown_steps(a, director.params, director.persona)}
            for a, end in sorted(folded.ends.items())
            if end + rules.cooldown_steps(a, director.params, director.persona) > t
        ]
    return {
        "pools": pools,
        "season": season,
        "levers": levers,
        "effects": effects,
        "cooldowns": cooldowns,
        "influence": None,
    }


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
    try:
        res = runner.run(rd, cfg, box.pop(), steps, quiet=True, director=director, pacer=pacer)
    finally:
        control.close()
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
