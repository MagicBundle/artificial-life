"""Optional pygame-ce live viewer (§20.2); pygame is imported lazily. Owned by WP6.

``ViewerSession`` holds everything except the window: the current state (and only it, the donation rule of
§15.4), the compiled step, the viewport, layers, colour mode, selection, the HUD values and snapshots. It has
no pygame dependency, so it is fully testable headless. ``run_viewer`` maps pygame events onto it and blits
the frames ``render.render_frame`` composites.

Controls: Space pause, ``.`` single step, ``+``/``-`` steps per frame (×2 / ÷2), drag pan, wheel zoom
(about the cursor), ``1``–``4`` toggle the agents/bonds/resource/chem layers, ``c`` cycle the colour mode,
click select the nearest agent, ``s`` state snapshot, ``p`` PNG, Esc or ``q`` quit.

Viewing never changes the trajectory: the session only calls the compiled step, and every observer (render,
selection, snapshot) reads the state without touching ``state.key``. Snapshots go to
``media/snap_<step>.safetensors`` (in the run directory if one is attached, else the current directory)
via ``runio.write_state_file``; checkpoints, ``LATEST``, ``metrics.jsonl`` and ``events/`` are never touched.

The viewer re-simulates without the director (docs/DIRECTOR.md): it keeps the checkpoint's environment levers
but applies none of ``director.jsonl``, so after a director action it diverges from the run it was opened on.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, NamedTuple

import mlx.core as mx
import numpy as np

from alife import bonds, ops
from alife.config import Config, dims, require_2d
from alife.render import COLOR_MODES, DEFAULT_LAYERS, Viewport, render_frame, write_png
from alife.runio import write_state_file
from alife.state import ERROR_MASK, CounterIdx, Diag, State, age, n_counters, storm_counter_index, to_numpy
from alife.step import make_step

if TYPE_CHECKING:
    from alife.runio import RunDir

LAYER_KEYS: Final = ("agents", "bonds", "resource", "chem")  # keys 1–4
ZOOM_STEP: Final = 1.25  # per wheel notch
ZOOM_MIN: Final = 0.25
ZOOM_MAX: Final = 256.0
SPF_MAX: Final = 1024  # steps per frame
CLICK_SLOP: Final = 4  # pixels a press may move and still count as a click
RATE_WINDOW_S: Final = 0.5  # steps/s refresh period
FRAME_BUDGET_S: Final = 0.1  # wall time a frame may spend stepping before it renders and pumps events
_HUD_FONT_PX: Final = 18
_HUD_FG: Final = (235, 235, 235)
_HUD_ERR: Final = (255, 90, 90)
_HUD_BG: Final = (0, 0, 0, 150)
_SELECT_RGB: Final = (255, 230, 0)
_INSTALL_HINT: Final = (
    "the live viewer needs pygame-ce: install the 'viewer' extra (uv sync --extra viewer, "
    "or pip install 'alife[viewer]')"
)

_DEATH_COUNTERS: Final = (
    CounterIdx.DEATHS_STARVED,
    CounterIdx.DEATHS_KILLED,
    CounterIdx.DEATHS_AGED,
    CounterIdx.DEATHS_CATASTROPHE,
)


class AgentInfo(NamedTuple):
    """HUD values of the selected agent."""

    uid: int
    energy: int
    age: int
    generation: int
    clade: int
    bonds: int
    x: float
    y: float


class FrameInfo(NamedTuple):
    """Host values read together with a frame (one transfer per frame)."""

    step: int
    n_alive: int
    bonds: int
    diag: int
    births: int  # in the HUD interval
    deaths: int
    interval_steps: int  # steps that interval covers (0 before the first step)
    selected: AgentInfo | None  # None when nothing is selected or the selected agent has died


class ViewerSession:
    """Viewer state and actions without a window; ``run_viewer`` drives one of these."""

    def __init__(
        self,
        cfg: Config,
        state: State,
        *,
        rundir: RunDir | None = None,
        size: tuple[int, int] = (900, 900),
        steps_per_frame: int = 1,
    ) -> None:
        require_2d(cfg, "the live viewer")
        self.cfg = cfg
        self.rundir = rundir
        self._state = state  # the only long-lived reference; replaced on every step (donation rule)
        self._step = make_step(cfg)
        L = float(cfg.world.size)
        self.viewport = Viewport(L / 2.0, L / 2.0, 1.0)
        self.layers: frozenset[str] = DEFAULT_LAYERS
        self.color = COLOR_MODES[0]
        self.paused = False
        self.steps_per_frame = int(min(max(steps_per_frame, 1), SPF_MAX))
        self.size = self._check_size(size)
        self.selected_uid: int | None = None
        self.message = ""  # last action's feedback (snapshot path, ...)
        self.frame: np.ndarray | None = None  # last rendered (H,W,3) uint8 image
        self.dirty = True  # the view changed since the last frame
        self.steps_per_s = 0.0
        nd = dims(cfg).nd
        self._acc = mx.zeros(
            (n_counters(nd, cfg.weather.enabled),), dtype=mx.int32
        )  # counters, interval sums
        self._deaths_idx: tuple[int, ...] = _DEATH_COUNTERS + (
            (storm_counter_index(nd),) if cfg.weather.enabled else ()
        )
        self._acc_steps = 0
        self._last_interval: tuple[int, int, int] | None = None  # (births, deaths, steps)
        self._rate_t0 = time.perf_counter()
        self._rate_steps = 0

    # --- state and stepping -------------------------------------------------------------------------------

    @property
    def state(self) -> State:
        return self._state

    @staticmethod
    def _check_size(size: tuple[int, int]) -> tuple[int, int]:
        w, h = (int(v) for v in size)
        if w < 1 or h < 1:
            raise ValueError(f"size must be positive, got {size}")
        return w, h

    def resize(self, size: tuple[int, int]) -> None:
        self.size = self._check_size(size)
        self.dirty = True

    def advance(self, n: int, budget_s: float | None = None) -> int:
        """Run up to ``n`` compiled steps, one blocking eval each (§15.4); counters accumulate on the device.

        With ``budget_s``, stop after the first step that ends ``budget_s`` or more after the call started, so
        a large ``steps_per_frame`` cannot keep the window from pumping events for seconds. Returns the steps
        run. The trajectory does not depend on how steps are split into frames.
        """
        t0 = time.perf_counter()
        done = 0
        while done < n:
            self._state, out = self._step(self._state)
            self._acc = self._acc + out.counters
            mx.eval(self._state, self._acc)
            done += 1
            if budget_s is not None and time.perf_counter() - t0 >= budget_s:
                break
        n = done
        if n > 0:
            self._acc_steps += n
            self._rate_steps += n
            self.dirty = True
        return n

    def tick(self) -> bool:
        """One frame's worth of simulation unless paused: ``steps_per_frame`` steps, or fewer (at least one)
        once ``FRAME_BUDGET_S`` of wall time is spent. True if it stepped."""
        now = time.perf_counter()
        if now - self._rate_t0 >= RATE_WINDOW_S:
            self.steps_per_s = self._rate_steps / (now - self._rate_t0)
            self._rate_t0, self._rate_steps = now, 0
        if self.paused:
            return False
        self.advance(self.steps_per_frame, FRAME_BUDGET_S)
        return True

    # --- rendering ------------------------------------------------------------------------------------------

    def render(self) -> tuple[np.ndarray, FrameInfo]:
        """Composite the frame and gather the HUD values; everything is evaluated and read together."""
        s = self._state
        img = render_frame(s, self.cfg, self.size, self.viewport, self.layers, self.color)
        level = mx.stack(
            [
                s.step.astype(mx.int64),
                s.alive.astype(mx.int32).sum().astype(mx.int64),
                ((s.partner >= 0).astype(mx.int32).sum() // 2).astype(mx.int64),
                s.diag.astype(mx.int64),
            ]
        )
        dev: list[mx.array] = [img, level, self._acc]
        if self.selected_uid is not None:
            dev += list(self._agent_arrays(self.selected_uid))
        mx.eval(*dev)
        host = [to_numpy(a) for a in dev]
        frame, lv, acc = host[0], host[1], host[2]

        births = int(acc[CounterIdx.BIRTHS])
        deaths = int(sum(int(acc[i]) for i in self._deaths_idx))
        if self._acc_steps >= self.cfg.run.metrics_every:  # latch a completed HUD interval
            self._last_interval = (births, deaths, self._acc_steps)
            self._acc = mx.zeros_like(self._acc)
            self._acc_steps = 0
        b, d, k = self._last_interval or (births, deaths, self._acc_steps)

        selected = None
        if self.selected_uid is not None:
            ints, floats = host[3], host[4]
            if ints[0]:
                selected = AgentInfo(*(int(v) for v in ints[1:]), float(floats[0]), float(floats[1]))
        self.frame = frame
        self.dirty = False
        info = FrameInfo(int(lv[0]), int(lv[1]), int(lv[2]), int(lv[3]), b, d, k, selected)
        return frame, info

    def _agent_arrays(self, uid: int) -> tuple[mx.array, mx.array]:
        """``(ints (7,) int64 = [found, uid, E, age, generation, clade, bonds], pos (2,) f32)`` on device."""
        s = self._state
        hit = (s.uid == uid) & s.alive
        row = mx.argmax(hit).astype(mx.int32)
        a = age(s.step, s.birth_step)
        ints = mx.stack(
            [
                hit.any().astype(mx.int64),
                s.uid[row],
                s.energy[row].astype(mx.int64),
                a[row].astype(mx.int64),
                s.generation[row].astype(mx.int64),
                s.clade[row].astype(mx.int64),
                bonds.n_bonds(s.partner)[row].astype(mx.int64),
            ]
        )
        return ints, s.pos[row].astype(mx.float32)

    def hud_lines(self, info: FrameInfo) -> list[tuple[str, bool]]:
        """HUD text lines with an "is an error" flag each."""
        status = "  [paused]" if self.paused else ""
        extinct = "  EXTINCT" if info.n_alive == 0 else ""
        lines = [
            (
                f"step {info.step}  pop {info.n_alive}{extinct}  births/deaths {info.births}/{info.deaths}"
                f" per {info.interval_steps} steps  bonds {info.bonds}  {self.steps_per_s:.1f} steps/s"
                f"  x{self.steps_per_frame}{status}",
                False,
            ),
            (
                f"layers {','.join(k for k in LAYER_KEYS if k in self.layers) or '-'}  color {self.color}"
                f"  zoom {self.viewport.zoom:.2f}",
                False,
            ),
        ]
        if info.diag & ERROR_MASK:
            flags = ",".join(f.name for f in Diag if info.diag & f.value & ERROR_MASK)
            lines.append((f"INVARIANT VIOLATION: diag {flags}", True))
        if self.selected_uid is not None:
            a = info.selected
            lines.append(
                (
                    f"uid {a.uid}  energy {a.energy}  age {a.age}  gen {a.generation}  clade {a.clade}"
                    f"  bonds {a.bonds}"
                    if a is not None
                    else f"uid {self.selected_uid} has died",
                    False,
                )
            )
        if self.message:
            lines.append((self.message, False))
        return lines

    # --- viewport -------------------------------------------------------------------------------------------

    @property
    def scale(self) -> float:
        """Pixels per world unit (§20.1)."""
        return self.viewport.zoom * min(self.size) / float(self.cfg.world.size)

    def screen_to_world(self, px: float, py: float) -> tuple[float, float]:
        """World coordinates (in ``[0, L)``) of the centre of pixel ``(px, py)``."""
        L, s = float(self.cfg.world.size), self.scale
        w, h = self.size
        x = self.viewport.cx + (px + 0.5 - w / 2.0) / s
        y = self.viewport.cy + (py + 0.5 - h / 2.0) / s
        return x % L, y % L

    def world_to_screen(self, x: float, y: float) -> tuple[float, float]:
        """Pixel position of the nearest periodic image of a world point (as ``render`` places centres)."""
        L, s = float(self.cfg.world.size), self.scale
        w, h = self.size

        def axis(v: float, c: float, extent: int) -> float:
            d = v - c
            return extent / 2.0 + (d - L * round(d / L)) * s

        return axis(x, self.viewport.cx, w), axis(y, self.viewport.cy, h)

    def pan(self, dx: float, dy: float) -> None:
        """Move the view with a drag of ``(dx, dy)`` pixels (the world follows the cursor)."""
        L, s = float(self.cfg.world.size), self.scale
        vp = self.viewport
        self.viewport = Viewport((vp.cx - dx / s) % L, (vp.cy - dy / s) % L, vp.zoom)
        self.dirty = True

    def zoom_at(self, px: float, py: float, factor: float) -> None:
        """Zoom by ``factor`` (clamped to ``[ZOOM_MIN, ZOOM_MAX]``), keeping the point under the cursor."""
        vp = self.viewport
        zoom = min(max(vp.zoom * factor, ZOOM_MIN), ZOOM_MAX)
        if zoom == vp.zoom:
            return
        L, s0 = float(self.cfg.world.size), self.scale
        s1 = s0 * zoom / vp.zoom
        w, h = self.size
        ux, uy = px + 0.5 - w / 2.0, py + 0.5 - h / 2.0
        cx = (vp.cx + ux / s0 - ux / s1) % L
        cy = (vp.cy + uy / s0 - uy / s1) % L
        self.viewport = Viewport(cx, cy, zoom)
        self.dirty = True

    # --- selection and keys ---------------------------------------------------------------------------------

    def select_at(self, px: float, py: float) -> int | None:
        """Select the alive agent nearest (minimum image) to the point under the pixel; None if extinct."""
        s = self._state
        L = float(self.cfg.world.size)
        x, y = self.screen_to_world(px, py)
        d = ops.min_image(s.pos - mx.array([x, y], dtype=mx.float32), L)
        d2 = mx.where(s.alive, (d * d).sum(axis=1), math.inf)
        row = mx.argmin(d2).astype(mx.int32)
        found, uid = s.alive.any(), s.uid[row]
        mx.eval(found, uid)
        self.selected_uid = int(to_numpy(uid)) if bool(to_numpy(found)) else None
        self.dirty = True
        return self.selected_uid

    def handle_key(self, key: str) -> None:
        """Apply one control key (§20.2): `` `` ``.`` ``+`` ``-`` ``1``–``4`` ``c`` ``s`` ``p``."""
        if key == " ":
            self.paused = not self.paused
        elif key == ".":
            self.paused = True
            self.advance(1)
        elif key in ("+", "="):
            self.steps_per_frame = min(self.steps_per_frame * 2, SPF_MAX)
        elif key == "-":
            self.steps_per_frame = max(self.steps_per_frame // 2, 1)
        elif key in ("1", "2", "3", "4"):
            layer = LAYER_KEYS[int(key) - 1]
            self.layers = self.layers ^ {layer}
        elif key == "c":
            self.color = COLOR_MODES[(COLOR_MODES.index(self.color) + 1) % len(COLOR_MODES)]
        elif key == "s":
            self.message = f"snapshot: {self.snapshot()}"
        elif key == "p":
            self.message = f"png: {self.save_png()}"
        else:
            return
        self.dirty = True

    # --- files ----------------------------------------------------------------------------------------------

    def media_dir(self) -> Path:
        """``<run>/media`` when a run directory is attached, else ``./media``."""
        base = Path(self.rundir.path) if self.rundir is not None else Path.cwd()
        return base / "media"

    def snapshot(self) -> Path:
        """Write the current state to ``media/snap_<step>.safetensors`` (a forkable state file, §18)."""
        step = int(to_numpy(self._state.step))
        path = self.media_dir() / f"snap_{step}.safetensors"
        path.parent.mkdir(parents=True, exist_ok=True)
        write_state_file(path, self._state, self.cfg)
        return path

    def save_png(self) -> Path:
        """Write the current view to ``media/view_<step>.png``."""
        if self.frame is None or self.dirty:
            self.render()
        assert self.frame is not None
        step = int(to_numpy(self._state.step))
        path = self.media_dir() / f"view_{step}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        write_png(path, self.frame)
        return path


def _import_pygame() -> Any:
    try:
        import pygame
    except ImportError as e:
        raise ImportError(_INSTALL_HINT) from e
    return pygame


def _key_char(ev: Any, pg: Any) -> str:
    """The control character of a KEYDOWN event (keypad and US-layout ``=`` map to ``+``)."""
    special = {pg.K_SPACE: " ", pg.K_KP_PLUS: "+", pg.K_KP_MINUS: "-", pg.K_KP_PERIOD: ".", pg.K_EQUALS: "+"}
    if ev.key in special:
        return special[ev.key]
    return (getattr(ev, "unicode", "") or "").lower()


def _draw(
    screen: Any, font: Any, pg: Any, session: ViewerSession, frame: np.ndarray, info: FrameInfo
) -> None:
    h, w = frame.shape[:2]
    if screen.get_size() == (w, h):
        pg.surfarray.blit_array(screen, frame.swapaxes(0, 1))
    else:  # the window was resized after this frame was rendered
        screen.blit(pg.surfarray.make_surface(frame.swapaxes(0, 1)), (0, 0))
    if info.selected is not None:
        px, py = session.world_to_screen(info.selected.x, info.selected.y)
        r = max(4, round(session.cfg.physics.radius * session.scale) + 3)
        pg.draw.circle(screen, _SELECT_RGB, (int(px), int(py)), r, 1)
    lines = [
        (font.render(text, True, _HUD_ERR if err else _HUD_FG), err) for text, err in session.hud_lines(info)
    ]
    pad = 4
    box_w = max(t.get_width() for t, _ in lines) + 2 * pad
    box_h = sum(t.get_height() for t, _ in lines) + 2 * pad
    bg = pg.Surface((box_w, box_h), pg.SRCALPHA)
    bg.fill(_HUD_BG)
    screen.blit(bg, (0, 0))
    y = pad
    for t, _ in lines:
        screen.blit(t, (pad, y))
        y += t.get_height()
    pg.display.flip()


def run_viewer(
    cfg: Config,
    state: State,
    *,
    rundir: RunDir | None = None,
    size: tuple[int, int] = (900, 900),
    steps_per_frame: int = 1,
) -> None:
    """Interactive viewer loop; holds only the current state (donation rule).

    Each frame runs up to ``steps_per_frame`` compiled steps (one sync eval each; stepping stops early once
    ``FRAME_BUDGET_S`` of wall time is spent, so input stays responsive), then one ``render_frame`` and
    one blit. Returns when the window is closed (or on Esc / ``q``). Raises ``ImportError`` with an install
    hint when pygame-ce is missing. Pass the state without keeping another reference to it, so the first
    step can update the weights in place.
    """
    require_2d(cfg, "the live viewer")  # before pygame, so a 3D run never opens a window
    pg = _import_pygame()
    session = ViewerSession(cfg, state, rundir=rundir, size=size, steps_per_frame=steps_per_frame)
    del state
    pg.display.init()
    pg.font.init()
    try:
        screen = pg.display.set_mode(session.size, pg.RESIZABLE)
        pg.display.set_caption("alife")
        font = pg.font.Font(None, _HUD_FONT_PX)
        clock = pg.time.Clock()
        press: tuple[int, int] | None = None  # left-button press position while held
        last: tuple[int, int] = (0, 0)
        dragging = False
        running = True
        while running:
            for ev in pg.event.get():
                if ev.type == pg.QUIT:
                    running = False
                elif ev.type == pg.KEYDOWN:
                    if ev.key in (pg.K_ESCAPE, pg.K_q):
                        running = False
                    else:
                        session.handle_key(_key_char(ev, pg))
                elif ev.type == pg.MOUSEBUTTONDOWN and ev.button == 1:
                    press, last, dragging = ev.pos, ev.pos, False
                elif ev.type == pg.MOUSEMOTION and press is not None:
                    if (
                        not dragging
                        and max(abs(ev.pos[0] - press[0]), abs(ev.pos[1] - press[1])) > CLICK_SLOP
                    ):
                        dragging = True
                    if dragging:
                        session.pan(ev.pos[0] - last[0], ev.pos[1] - last[1])
                        last = ev.pos
                elif ev.type == pg.MOUSEBUTTONUP and ev.button == 1 and press is not None:
                    if not dragging:
                        session.select_at(*ev.pos)
                    press, dragging = None, False
                elif ev.type == pg.MOUSEWHEEL and ev.y:
                    mx_, my_ = pg.mouse.get_pos()
                    session.zoom_at(mx_, my_, ZOOM_STEP**ev.y)
                elif ev.type == pg.VIDEORESIZE:
                    session.resize(ev.size)
                    screen = pg.display.set_mode(session.size, pg.RESIZABLE)
            if not running:
                break
            if session.tick() or session.dirty:
                frame, info = session.render()
                _draw(screen, font, pg, session, frame, info)
            else:
                clock.tick(30)  # paused and nothing changed: idle without spinning
    finally:
        pg.font.quit()
        pg.display.quit()
