"""``alife stream``: the binary frame protocol that feeds the 3D viewer (DESIGN §27.11, docs/3D.md §8).

Host module. The command re-simulates from a state like ``view`` (``run_simulation``: one blocking ``mx.eval``
per step, the loop holding the only reference to the state: the donation rule, §15.4) or, with ``run_follow``,
re-reads a running run's newest checkpoint without stepping. Either way it writes a header and then one frame
per streamed state to a private duplicate of fd 1 (``protocol_output``), with ``sys.stdout`` and fd 1 pointed
at stderr meanwhile, so no stray print can corrupt the framing. Colours are computed here, with the renderer's
palette (``render.agent_colors`` and ``render.field_colors``). The byte layout, the control grammar and the
exit codes are normative in DESIGN §27.11; ``tests/stream_decode.py`` is an independent decoder written from
that text.

Streaming never writes into a run directory and never touches the state's RNG key: the trajectory a stream
steps through is the one ``alife view`` shows (the director is not replayed, §26.8).
"""

from __future__ import annotations

import contextlib
import json
import os
import queue
import re
import struct
import sys
import threading
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, BinaryIO, Final, Protocol

import mlx.core as mx
import numpy as np

from alife import layout
from alife.config import Config, ConfigError, config_hash, dims
from alife.render import COLOR_MODES, agent_colors, field_colors
from alife.runio import CheckpointError, RunDir, load_checkpoint
from alife.state import ERROR_MASK, State, State3, to_numpy
from alife.step import make_step

MAGIC: Final = b"ALSTRM\x00\x01"  # "ALSTRM", a zero byte, the major protocol version
PROTOCOL: Final = 1
FRAME_TAG: Final = b"FRM1"
HEAD: Final = struct.Struct("<QIIIIQ")  # step, n_alive, flags, n_bonds, colour index, 8 reserved bytes
HEAD_SIZE: Final = 32
FLAG_FIELD: Final = 1 << 0  # a field section follows the bonds
FLAG_DIAG: Final = 1 << 1  # the state has an invariant (error) diag bit set
FIELD_LAYER_SHIFT: Final = 16  # flags >> 16: the lattice layer of the field section
MODES: Final = ("simulate", "follow")
SOURCES: Final = ("run", "file", "preset")
CONTROL_FD: Final = 0  # the descriptor control lines are read from
MAX_LINE: Final = 4096  # a longer control line is dropped
COMMANDS: Final = ("pause", "run", "step", "every", "color", "fields", "quit")

_UINT = re.compile(r"[0-9]+")

assert HEAD.size == HEAD_SIZE


def _warn(msg: str) -> None:
    print(f"stream: {msg}", file=sys.stderr, flush=True)


# --- encoding -------------------------------------------------------------------------------------------


def _pad4(b: bytes) -> bytes:
    return b + b"\x00" * (-len(b) % 4)


def header_bytes(info: Mapping[str, Any]) -> bytes:
    """``MAGIC``, the u32 length of the JSON text and the JSON, space-padded to a multiple of 4 bytes."""
    body = json.dumps(info, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")
    body += b" " * (-len(body) % 4)
    return MAGIC + struct.pack("<I", len(body)) + body


def encode_frame(
    *,
    step: int,
    n_alive: int,
    flags: int,
    color_index: int,
    alive: np.ndarray,
    pos: np.ndarray,
    orient: np.ndarray,
    rgb: np.ndarray,
    bonds: np.ndarray,
    field_rgb: np.ndarray | None,
) -> bytes:
    """One frame record: ``FRM1``, the u32 payload length, the 32-byte head and the sections (little endian).

    ``alive`` (N,) bool, ``pos`` (N,3) float32, ``orient`` (N,2) (yaw, pitch), ``rgb`` (N,) ``0x00RRGGBB``,
    ``bonds`` (n_bonds,2) row pairs, ``field_rgb`` (FG²,) or None (then ``FLAG_FIELD`` must be clear).
    """
    parts = [
        _pad4(alive.astype(np.uint8).tobytes()),
        np.ascontiguousarray(pos, dtype="<f4").tobytes(),
        np.ascontiguousarray(orient, dtype="<f2").tobytes(),
        np.ascontiguousarray(rgb, dtype="<u4").tobytes(),
        np.ascontiguousarray(bonds, dtype="<u4").tobytes(),
    ]
    if field_rgb is not None:
        parts.append(np.ascontiguousarray(field_rgb, dtype="<u4").tobytes())
    payload = HEAD.pack(step, n_alive, flags, len(bonds), color_index, 0) + b"".join(parts)
    return FRAME_TAG + struct.pack("<I", len(payload)) + payload


def _write_all(out: BinaryIO, data: bytes) -> None:
    """Write all of ``data`` (a raw pipe write may be short)."""
    view = memoryview(data)
    while view:
        n = out.write(view)
        if n is None or n <= 0:
            raise BrokenPipeError("short write to the protocol stream")
        view = view[n:]


# --- the view and the frame builder ---------------------------------------------------------------------


@dataclass
class View:
    """What a frame shows: the colour mode and the optional field slice."""

    color: str = COLOR_MODES[0]
    fields: bool = False
    field_z: int = 0

    def check(self, cfg: Config) -> None:
        if self.color not in COLOR_MODES:
            raise ConfigError(f"unknown colour mode {self.color!r}; expected one of {COLOR_MODES}")
        fz = dims(cfg).fz
        if not 0 <= self.field_z < fz:
            raise ConfigError(f"field layer {self.field_z} is outside [0, {fz})")


class Streamer:
    """Writes the header and the frames of one stream to ``out`` (a binary file object)."""

    def __init__(
        self,
        cfg: Config,
        out: BinaryIO,
        *,
        mode: str = "simulate",
        source: str = "preset",
        run: str | None = None,
        view: View | None = None,
        every: int = 1,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if source not in SOURCES:
            raise ValueError(f"source must be one of {SOURCES}, got {source!r}")
        self.cfg = cfg
        self.out = out
        self.mode, self.source, self.run, self.every = mode, source, run, every
        self.view = view or View()
        self.view.check(cfg)
        self.frames = 0

    def header_info(self, state: State | State3) -> dict[str, Any]:
        """The header JSON object, in the documented key order."""
        cfg, d = self.cfg, dims(self.cfg)
        return {
            "protocol": PROTOCOL,
            "dims": d.nd,
            "L": float(cfg.world.size),
            "height": float(cfg.world.height) if d.d3 else 0.0,
            "N": int(state.alive.shape[0]),
            "B": layout.B,
            "radius": float(cfg.physics.radius),
            "field_cell": float(cfg.world.field_cell),
            "FG": d.fg,
            "FZ": d.fz,
            "NCH": layout.NCH,
            "color_modes": list(COLOR_MODES),
            "color": self.view.color,
            "fields": self.view.fields,
            "field_z": self.view.field_z,
            "step": int(to_numpy(state.step)),
            "run": self.run,
            "config_hash": config_hash(cfg),
            "mode": self.mode,
            "every": self.every,
            "source": self.source,
        }

    def begin(self, state: State | State3) -> None:
        """Write the header (once, before the first frame)."""
        _write_all(self.out, header_bytes(self.header_info(state)))

    def frame_bytes(self, state: State | State3) -> bytes:
        """The frame record of ``state`` in the current view (one device evaluation, one host transfer)."""
        cfg, view = self.cfg, self.view
        d = dims(cfg)
        colors = agent_colors(state, cfg, view.color)
        dev: dict[str, mx.array] = {
            "alive": state.alive,
            "pos": state.pos,
            "yaw": state.heading,
            "partner": state.partner,
            "step": state.step,
            "diag": state.diag,
            "rgb": colors,
        }
        if d.d3:
            assert isinstance(state, State3)
            dev["pitch"] = state.pitch
        if view.fields:
            dev["field"] = field_colors(state, cfg, view.field_z)
        mx.eval(*dev.values())
        h = {k: to_numpy(v) for k, v in dev.items()}

        alive = h["alive"].astype(bool)
        n = alive.shape[0]
        pos = np.zeros((n, 3), np.float32)
        pos[:, : h["pos"].shape[1]] = h["pos"]
        orient = np.zeros((n, 2), np.float32)
        orient[:, 0] = h["yaw"]
        if d.d3:
            orient[:, 1] = h["pitch"]
        partner = h["partner"]
        rows = np.arange(n)[:, None]
        drawn = (partner > rows) & alive[:, None] & alive[np.maximum(partner, 0)]
        ii, bb = np.nonzero(drawn)  # row-major: ascending row, then slot
        pairs = np.stack([ii, partner[ii, bb]], axis=1).astype(np.uint32)

        flags = (FLAG_FIELD | view.field_z << FIELD_LAYER_SHIFT) if view.fields else 0
        if int(h["diag"]) & ERROR_MASK:
            flags |= FLAG_DIAG
        return encode_frame(
            step=int(h["step"]),
            n_alive=int(alive.sum()),
            flags=flags,
            color_index=COLOR_MODES.index(view.color),
            alive=alive,
            pos=pos,
            orient=orient,
            rgb=h["rgb"],
            bonds=pairs,
            field_rgb=h["field"] if view.fields else None,
        )

    def send(self, state: State | State3) -> None:
        _write_all(self.out, self.frame_bytes(state))
        self.frames += 1


# --- control lines --------------------------------------------------------------------------------------


class ControlError(ValueError):
    """A control line that is not valid (it is ignored, with a note on stderr)."""


@dataclass(frozen=True)
class Command:
    name: str
    arg: int | str | None = None
    sub: str | None = None  # "on", "off" or "z" for ``fields``


def _uint(text: str, what: str) -> int:
    if not _UINT.fullmatch(text):
        raise ControlError(f"{what} must be a non-negative integer, got {text!r}")
    return int(text)


def parse_command(line: str, *, fz: int = 1) -> Command | None:
    """One control line (``None`` for a blank line or a ``#`` comment); ``ControlError`` if it is invalid.

    Grammar: ``pause`` | ``run`` | ``step [N]`` | ``every K`` | ``color MODE`` | ``fields on`` |
    ``fields off`` | ``fields z K`` | ``quit``; whitespace-separated, lower case, ``N`` and ``K`` integers
    with ``N, K ≥ 1`` (``step``, ``every``) and ``0 ≤ K < FZ`` (``fields z``).
    """
    tok = line.split()
    if not tok or tok[0].startswith("#"):
        return None
    name, args = tok[0], tok[1:]
    if name not in COMMANDS:
        raise ControlError(f"unknown command {name!r}; expected one of {', '.join(COMMANDS)}")

    def arity(lo: int, hi: int) -> None:
        if not lo <= len(args) <= hi:
            want = str(lo) if lo == hi else f"{lo} to {hi}"
            raise ControlError(f"{name} takes {want} argument(s), got {len(args)}")

    if name in ("pause", "run", "quit"):
        arity(0, 0)
        return Command(name)
    if name == "step":
        arity(0, 1)
        n = _uint(args[0], "N") if args else 1
        if n < 1:
            raise ControlError("step N needs N >= 1")
        return Command(name, n)
    if name == "every":
        arity(1, 1)
        k = _uint(args[0], "K")
        if k < 1:
            raise ControlError("every K needs K >= 1")
        return Command(name, k)
    if name == "color":
        arity(1, 1)
        if args[0] not in COLOR_MODES:
            raise ControlError(f"unknown colour mode {args[0]!r}; expected one of {COLOR_MODES}")
        return Command(name, args[0])
    arity(1, 2)  # fields
    if args[0] in ("on", "off") and len(args) == 1:
        return Command(name, sub=args[0])
    if args[0] == "z" and len(args) == 2:
        layer = _uint(args[1], "K")
        if layer >= fz:
            raise ControlError(f"field layer {layer} is outside [0, {fz})")
        return Command(name, layer, sub="z")
    raise ControlError("fields takes 'on', 'off' or 'z K'")


class Control(Protocol):
    """A source of control lines. ``poll(0)`` never blocks, ``poll(t)`` waits up to ``t`` seconds for a line
    and ``poll(None)`` until one arrives; each returns every line then available. EOF reads as ``quit``."""

    def poll(self, timeout: float | None = 0) -> list[str]: ...

    def close(self) -> None: ...


class NoControl:
    """No control channel: ``poll`` only waits (``--no-control``; stdin is not read)."""

    def poll(self, timeout: float | None = 0) -> list[str]:
        if timeout:
            threading.Event().wait(timeout)
        return []

    def close(self) -> None:
        pass


class QueueControl:
    """Control lines from a thread-safe queue (``put``); the base of ``FdControl``."""

    def __init__(self) -> None:
        self._q: queue.Queue[str] = queue.Queue()

    def put(self, line: str) -> None:
        self._q.put(line)

    def poll(self, timeout: float | None = 0) -> list[str]:
        out: list[str] = []
        try:
            if timeout is None:
                out.append(self._q.get())
            elif timeout > 0:
                out.append(self._q.get(timeout=timeout))
            while True:
                out.append(self._q.get_nowait())
        except queue.Empty:
            pass
        return out

    def close(self) -> None:
        pass


class FdControl(QueueControl):
    """Control lines read from a file descriptor (stdin) by a daemon thread; EOF queues ``quit``.

    It reads with ``os.read`` rather than through ``sys.stdin``: a daemon thread blocked in a buffered reader
    can abort interpreter shutdown, and a raw read cannot. A line longer than ``MAX_LINE`` bytes is dropped.
    """

    def __init__(self, fd: int | None = None) -> None:
        super().__init__()
        self._fd = CONTROL_FD if fd is None else fd
        self._thread = threading.Thread(target=self._read, name="alife-stream-control", daemon=True)
        self._thread.start()

    def _read(self) -> None:
        buf, skipping = b"", False
        while True:
            try:
                chunk = os.read(self._fd, 4096)
            except OSError:
                chunk = b""
            if not chunk:  # EOF (or an unreadable descriptor): the client is gone
                if buf.strip() and not skipping:
                    self.put(buf.decode("utf-8", "replace"))
                self.put("quit")
                return
            buf += chunk
            *lines, buf = buf.split(b"\n")
            for raw in lines:
                if not skipping and len(raw) <= MAX_LINE:
                    self.put(raw.decode("utf-8", "replace"))
                skipping = False
            if len(buf) > MAX_LINE:
                buf, skipping = b"", True


# --- the loops ------------------------------------------------------------------------------------------


@dataclass
class Session:
    """The mutable control state of one stream (``apply`` turns control lines into it)."""

    streamer: Streamer
    mode: str = "simulate"
    every: int = 1
    paused: bool = False
    pending: int = 0  # steps left of a ``step N`` batch (simulate)
    quit: bool = False
    redraw: bool = False  # a frame of the current state is due now
    refresh: bool = False  # follow: look for a newer checkpoint now even when paused
    steps: int = 0

    def apply(self, line: str) -> None:
        view = self.streamer.view
        try:
            cmd = parse_command(line, fz=dims(self.streamer.cfg).fz)
        except ControlError as e:
            _warn(f"ignored control line {line.strip()[:80]!r}: {e}")
            return
        if cmd is None:
            return
        if cmd.name == "quit":
            self.quit = True
        elif cmd.name == "pause":
            self.paused, self.pending = True, 0
        elif cmd.name == "run":
            self.paused, self.pending = False, 0
        elif cmd.name == "step":
            if self.mode == "follow":
                self.refresh = True
            else:
                assert isinstance(cmd.arg, int)
                self.paused, self.pending = True, self.pending + cmd.arg
        elif cmd.name == "every":
            if self.mode == "follow":
                _warn("ignored 'every': a followed stream has a frame per checkpoint")
            else:
                assert isinstance(cmd.arg, int)
                self.every = cmd.arg
        elif cmd.name == "color":
            assert isinstance(cmd.arg, str)
            view.color = cmd.arg
            self.redraw = True
        elif cmd.sub in ("on", "off"):
            view.fields = cmd.sub == "on"
            self.redraw = True
        else:  # fields z K
            assert isinstance(cmd.arg, int)
            view.field_z = cmd.arg
            self.redraw = self.redraw or view.fields


@dataclass
class StreamStats:
    frames: int
    steps: int
    ended: str  # "steps" (the bound), "quit" (a command or EOF) or "pipe" (the reader closed)
    state: State | State3 | None = field(default=None, repr=False)


StepFn = Callable[[Any], tuple[Any, Any]]


def run_simulation(
    cfg: Config,
    state: State | State3,
    streamer: Streamer,
    control: Control,
    *,
    every: int = 1,
    max_steps: int | None = None,
    step_fn: StepFn | None = None,
    session: Session | None = None,
) -> StreamStats:
    """Write the header, a frame of ``state`` and then, stepping, a frame every ``every`` steps.

    The loop applies the control lines between steps (a paused loop with nothing to do blocks on the control
    channel) and stops after ``max_steps`` steps, on ``quit`` or at EOF of the control channel. A
    ``BrokenPipeError`` from the output propagates (``serve`` turns it into a quiet exit). Frames follow the
    step grid ``steps % every == 0`` counted from the first state, so a bound of ``S`` steps writes
    ``S // every + 1`` frames. The loop holds the only reference to the state while it steps (donation rule).
    """
    step_fn = step_fn or make_step(cfg)
    sess = session or Session(streamer)
    sess.mode, sess.every = "simulate", every
    streamer.begin(state)
    streamer.send(state)
    ended = "quit"
    while True:
        done = max_steps is not None and sess.steps >= max_steps
        idle = sess.paused and sess.pending == 0
        for line in control.poll(None if idle and not done else 0):
            sess.apply(line)
        if sess.quit:
            break
        if sess.redraw:
            streamer.send(state)
            sess.redraw = False
        if done:
            ended = "steps"
            break
        if sess.paused and sess.pending == 0:
            continue
        state, out = step_fn(state)
        mx.eval(state, out)
        del out
        sess.steps += 1
        if sess.pending > 0:
            sess.pending -= 1
            if sess.pending == 0:
                streamer.send(state)
        elif not sess.paused and sess.steps % sess.every == 0:
            streamer.send(state)
    return StreamStats(streamer.frames, sess.steps, ended, state)


def _newest(
    rd: RunDir, cfg: Config, since: tuple[Any, ...] | None
) -> tuple[tuple[Any, ...], State | State3] | None:
    """The run's newest checkpoint if it is not the one identified by ``since`` (name, mtime, size)."""
    path = rd.latest_checkpoint()
    if path is None:
        return None
    try:
        st = path.stat()
        ident = (path.name, st.st_mtime_ns, st.st_size)
        if ident == since:
            return None
        return ident, load_checkpoint(path, cfg)
    except FileNotFoundError:  # pruned between the listing and the read: the next poll finds the new one
        return None


def run_follow(
    cfg: Config,
    rd: RunDir,
    streamer: Streamer,
    control: Control,
    *,
    poll_s: float = 1.0,
    session: Session | None = None,
) -> StreamStats:
    """Write the header and a frame of the run's latest checkpoint, then a frame for every newer checkpoint.

    Nothing is stepped: the frames are the states the runner wrote, director actions included, so this is the
    faithful view of a director run. The directory is read, never written. ``pause`` holds the stream on its
    current checkpoint, ``step`` fetches the newest one once, ``color`` and ``fields`` redraw the current
    state, ``every`` is ignored. Ends on ``quit`` or at EOF of the control channel; ``CheckpointError`` if
    the run has no checkpoint.
    """
    first = _newest(rd, cfg, None)
    if first is None:
        raise CheckpointError(f"{rd.path}: no checkpoint")
    ident, state = first
    sess = session or Session(streamer)
    sess.mode, sess.every = "follow", 0
    streamer.begin(state)
    streamer.send(state)
    while True:
        for line in control.poll(poll_s):
            sess.apply(line)
        if sess.quit:
            break
        if not sess.paused or sess.refresh:
            sess.refresh = False
            if (newer := _newest(rd, cfg, ident)) is not None:
                ident, state = newer
                sess.redraw = True
        if sess.redraw:
            streamer.send(state)
            sess.redraw = False
    return StreamStats(streamer.frames, 0, "quit", state)


# --- the process boundary -------------------------------------------------------------------------------


@contextlib.contextmanager
def protocol_output() -> Iterator[BinaryIO]:
    """A private unbuffered binary duplicate of fd 1, with ``sys.stdout`` and fd 1 itself pointed at stderr.

    Inside the block nothing but the stream writes to the protocol descriptor: a ``print``, a C library's
    ``printf`` and a child process all land on stderr. On exit the original descriptor 1 and ``sys.stdout``
    are restored. Closing never raises (a broken pipe included).
    """
    for f in (sys.stdout, sys.stderr):
        with contextlib.suppress(OSError, ValueError):
            f.flush()
    out = os.fdopen(os.dup(1), "wb", buffering=0)
    saved = sys.stdout
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    try:
        yield out
    finally:
        sys.stdout = saved
        os.dup2(out.fileno(), 1)
        with contextlib.suppress(OSError):
            out.close()


def serve(
    cfg: Config,
    *,
    mode: str,
    source: str,
    run: str | None,
    state: list[State | State3] | None = None,
    rd: RunDir | None = None,
    view: View | None = None,
    every: int = 1,
    max_steps: int | None = None,
    poll_s: float = 1.0,
    control: Control | None = None,
    out: BinaryIO | None = None,
    step_fn: StepFn | None = None,
) -> StreamStats:
    """Stream to ``out`` (default: ``protocol_output()``) until the bound, ``quit``, EOF of the control
    channel or a closed reader (``ended = "pipe"``, no error).

    ``state`` is a one-element box holding the state to simulate (popped here, so the caller keeps no
    reference); ``rd`` the run directory of a followed stream. ``control`` defaults to no channel.
    """
    control = NoControl() if control is None else control
    streamer: Streamer | None = None
    session = None
    target = contextlib.nullcontext(out) if out is not None else protocol_output()
    with target as sink:
        try:
            every = 0 if mode == "follow" else every  # a followed stream has a frame per checkpoint
            streamer = Streamer(cfg, sink, mode=mode, source=source, run=run, view=view, every=every)
            session = Session(streamer)
            if mode == "follow":
                assert rd is not None
                return run_follow(cfg, rd, streamer, control, poll_s=poll_s, session=session)
            assert state
            return run_simulation(
                cfg, state.pop(), streamer, control, every=every, max_steps=max_steps, step_fn=step_fn,
                session=session,
            )  # fmt: skip
        except (BrokenPipeError, ConnectionResetError):  # the reader closed: a normal way to stop
            return StreamStats(streamer.frames if streamer else 0, session.steps if session else 0, "pipe")
        finally:
            control.close()
