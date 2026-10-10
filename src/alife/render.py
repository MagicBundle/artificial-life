"""Headless MLX rasteriser, stdlib PNG writer and ffmpeg video writer (§20.1). Owned by WP6.

Screen convention: pixel column ``px`` grows with world ``x`` and pixel row ``py`` grows with world ``y`` (the
image's top-left is the world's low ``x``/``y`` corner of the view), matching the ``resource[y, x]`` layout.
The view is periodic: every sample is wrapped modulo the world period in pixel space, so panning across the
seam wraps and zooming out (``zoom < 1``) tiles the torus.
"""

from __future__ import annotations

import contextlib
import math
import os
import shutil
import struct
import subprocess
import tempfile
import warnings
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

import mlx.core as mx
import numpy as np

from alife import layout, ops
from alife.config import Config, dims
from alife.state import State, to_numpy


@dataclass(frozen=True)
class Viewport:
    cx: float
    cy: float
    zoom: float = 1.0


DEFAULT_LAYERS: frozenset[str] = frozenset({"resource", "chem", "bonds", "agents"})
COLOR_MODES: tuple[str, ...] = ("tag", "energy", "clade")

# Render colours and priorities (§20.1; local constants by design, see R25).
_PRIO_BOND = 1
_PRIO_AGENT_BASE = 128
_PRIO_AGENT_SPAN = 127
_PRIO_NOSE = 255
_BOND_RGB = (128, 128, 128)
_NOSE_RGB = (255, 255, 255)
_HURT_RGB = (255, 0, 0)
_BOND_SAMPLES = 8  # interior points t = (k+1)/9, k = 0..7, along the minimum-image segment
_NOSE_DIST = 0.7  # nose at pos + 0.7·r·fwd
_RES_G0, _RES_G_SPAN = 30.0, 150.0
_CHEM_GAIN = 90.0
_CHEM_TINT = ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.6, 0.0, 1.0))  # c0 red, c1 blue, c2 violet
# Energy colour ramp: (t, RGB) stops over t = log1p(E) / log1p(e_max).
_ENERGY_RAMP = ((0.0, (30, 60, 200)), (0.5, (60, 200, 120)), (1.0, (255, 220, 40)))
_CLADE_HASH = 2654435761  # Knuth multiplicative hash, wraps mod 2^32
_SAMPLE_BUDGET = 1 << 23  # max samples per scatter; larger splats are split into evaluated row chunks

_U32 = mx.uint32
_I32 = mx.int32
_F32 = mx.float32


@dataclass(frozen=True)
class _Axis:
    """Static (Python) mapping of one world axis onto ``extent`` pixels."""

    centre: float
    extent: int
    scale: float  # pixels per world unit
    period: float  # L·scale: the world period in pixels
    k_max: int  # periodic images needed on each side to cover the frame


def _axes(cfg: Config, size: tuple[int, int], viewport: Viewport | None) -> tuple[_Axis, _Axis]:
    w, h = size
    L = float(cfg.world.size)
    vp = viewport or Viewport(L / 2.0, L / 2.0, 1.0)
    if not (math.isfinite(vp.zoom) and vp.zoom > 0 and math.isfinite(vp.cx) and math.isfinite(vp.cy)):
        raise ValueError(f"viewport must be finite with zoom > 0, got {vp}")
    scale = vp.zoom * min(w, h) / L
    period = L * scale

    def axis(centre: float, extent: int) -> _Axis:
        # A wrapped sample v lies in [−P/2, P/2); image k is visible iff |v + kP| < extent/2.
        k_max = max(0, math.ceil((extent / 2 + period / 2) / period) - 1)
        return _Axis(centre % L, extent, scale, period, k_max)

    return axis(vp.cx, w), axis(vp.cy, h)


def _axis_images(coord: mx.array, offs: mx.array, ax: _Axis, L: float) -> list[mx.array]:
    """Pixel coordinates (f32, shape coord.shape + offs.shape) of every periodic image of each sample.

    The sample's centre pixel is ``floor(extent/2 + min_image(coord − centre)·scale)``; integer stencil
    offsets are added in pixel space and each pixel centre is then wrapped into ``[−P/2, P/2)`` around the
    frame centre (floor-based, so the interval is half-open and ``k_max`` images per side cover the frame).
    """
    half = ax.extent / 2.0
    c = mx.floor(ops.min_image(coord - ax.centre, L) * ax.scale + half)
    v = c[..., None] + offs + (0.5 - half)
    v = v - ax.period * mx.floor(v / ax.period + 0.5)
    return [mx.floor(v + (half + k * ax.period)) for k in range(-ax.k_max, ax.k_max + 1)]


def _splat(
    canvas: mx.array,
    x: mx.array,
    y: mx.array,
    valid: mx.array,
    packed: mx.array,
    stencil: tuple[mx.array, mx.array],
    axes: tuple[_Axis, _Axis],
    L: float,
) -> mx.array:
    """Scatter-max ``packed`` (n,) uint32 at every stencil pixel of every visible image of (x, y) (n,).

    Off-frame and invalid samples go to the sink pixel ``H·W`` (the last canvas entry). When the sample count
    (n × stencil × images) exceeds ``_SAMPLE_BUDGET`` (deep zoom or far zoom-out), the rows are processed in
    chunks with an eval in between to bound transient memory; integer max makes the result identical.
    """
    n = x.shape[0]
    per_row = stencil[0].shape[0] * (2 * axes[0].k_max + 1) * (2 * axes[1].k_max + 1)
    chunk = max(1, _SAMPLE_BUDGET // per_row)
    if chunk >= n:
        return _splat_once(canvas, x, y, valid, packed, stencil, axes, L)
    for lo in range(0, n, chunk):
        hi = min(n, lo + chunk)
        canvas = _splat_once(canvas, x[lo:hi], y[lo:hi], valid[lo:hi], packed[lo:hi], stencil, axes, L)
        mx.eval(canvas)
    return canvas


def _splat_once(
    canvas: mx.array,
    x: mx.array,
    y: mx.array,
    valid: mx.array,
    packed: mx.array,
    stencil: tuple[mx.array, mx.array],
    axes: tuple[_Axis, _Axis],
    L: float,
) -> mx.array:
    ax_x, ax_y = axes
    w, h = ax_x.extent, ax_y.extent
    sink = w * h
    ox, oy = stencil
    pxs = _axis_images(x, ox, ax_x, L)
    pys = _axis_images(y, oy, ax_y, L)
    idx, vals = [], []
    vmask = valid[:, None]
    pval = mx.broadcast_to(packed[:, None], (packed.shape[0], ox.shape[0]))
    for px in pxs:
        in_x = (px >= 0) & (px < w)
        for py in pys:
            vis = vmask & in_x & (py >= 0) & (py < h)
            pix = mx.where(vis, py, 0).astype(_I32) * w + mx.where(vis, px, 0).astype(_I32)
            idx.append(mx.where(vis, pix, sink).reshape(-1))
            vals.append(pval.reshape(-1))
    flat_idx = idx[0] if len(idx) == 1 else mx.concatenate(idx)
    flat_val = vals[0] if len(vals) == 1 else mx.concatenate(vals)
    return canvas.at[flat_idx].maximum(flat_val)


def _pack(prio: mx.array | int, rgb: mx.array) -> mx.array:
    """(prio << 24) | (R << 16) | (G << 8) | B as uint32; ``rgb`` (n,3) integer-valued in [0, 255]."""
    c = rgb.astype(_U32)
    p = prio.astype(_U32) if isinstance(prio, mx.array) else mx.array(prio, dtype=_U32)
    return (p << 24) | (c[:, 0] << 16) | (c[:, 1] << 8) | c[:, 2]


def _const_rgb(rgb: tuple[int, int, int], n: int) -> mx.array:
    return mx.broadcast_to(mx.array(rgb, dtype=_U32)[None, :], (n, 3))


def _disc_stencil(rho: int) -> tuple[mx.array, mx.array]:
    """Integer offsets (static per ρ) of the disc ``ox² + oy² ≤ ρ²``, as f32 (S,) pairs."""
    ox, oy = [], []
    for dy in range(-rho, rho + 1):
        for dx in range(-rho, rho + 1):
            if dx * dx + dy * dy <= rho * rho:
                ox.append(dx)
                oy.append(dy)
    return mx.array(ox, dtype=_F32), mx.array(oy, dtype=_F32)


def _agent_rgb(s: State, cfg: Config, color: str) -> mx.array:
    """(N,3) f32 integer-valued agent colours; agents hurt last step (``last_loss > 0``) are red."""
    if color == "tag":
        t = mx.clip(s.tag[:, :3].astype(_F32), -1.0, 1.0)
        rgb = mx.floor(64.0 + 191.0 * (t + 1.0) / 2.0 + 0.5)
    elif color == "energy":
        t = mx.log1p(mx.maximum(s.energy, 0).astype(_F32)) / math.log1p(cfg.energy.e_max)
        t = mx.clip(t, 0.0, 1.0)[:, None]
        (t0, c0), *rest = _ENERGY_RAMP
        rgb = mx.array(c0, dtype=_F32)[None, :]
        prev_t, prev_c = t0, c0
        for ti, ci in rest:
            frac = mx.clip((t - prev_t) / (ti - prev_t), 0.0, 1.0)
            rgb = rgb + frac * mx.array([b - a for a, b in zip(prev_c, ci, strict=True)], dtype=_F32)[None, :]
            prev_t, prev_c = ti, ci
        rgb = mx.floor(rgb + 0.5)
    else:  # clade
        hsh = (s.clade.astype(_U32) * mx.array(_CLADE_HASH, dtype=_U32)) & 0xFFFFFF
        rgb = mx.stack([(hsh >> 16) & 255, (hsh >> 8) & 255, hsh & 255], axis=1).astype(_F32)
    hurt = (s.last_loss > 0)[:, None]
    return mx.where(hurt, mx.array(_HURT_RGB, dtype=_F32)[None, :], rgb)


def _background(s: State, cfg: Config, axes: tuple[_Axis, _Axis], layers: frozenset[str]) -> mx.array:
    """(H,W) uint32 packed RGB background: colours are computed per field cell, then each pixel centre gathers
    its (wrapped) cell."""
    ax_x, ax_y = axes
    w, h = ax_x.extent, ax_y.extent
    if not layers & {"resource", "chem"}:
        return mx.zeros((h, w), dtype=_U32)
    L, fg = float(cfg.world.size), dims(cfg).fg

    rgb = mx.zeros((fg * fg, 3), dtype=_F32)
    if "resource" in layers:
        r = s.resource.reshape(-1).astype(_F32)
        g = _RES_G0 + _RES_G_SPAN * mx.minimum(r / float(cfg.fields.resource_cap), 1.0)
        rgb = rgb + g[:, None] * mx.array([0.0, 1.0, 0.0], dtype=_F32)
    if "chem" in layers:
        chem = mx.maximum(s.chem.reshape(layout.NCH, -1), 0.0)  # (NCH, FG²)
        tint = _CHEM_GAIN * chem / (chem + 1.0)
        for ch in range(layout.NCH):
            rgb = rgb + tint[ch][:, None] * mx.array(_CHEM_TINT[ch], dtype=_F32)
    cell_rgb = mx.clip(mx.floor(rgb + 0.5), 0.0, 255.0).astype(_U32)
    cell_packed = (cell_rgb[:, 0] << 16) | (cell_rgb[:, 1] << 8) | cell_rgb[:, 2]

    def cells(ax: _Axis) -> mx.array:
        p = mx.arange(ax.extent, dtype=_F32) + 0.5 - ax.extent / 2.0
        world = ops.wrap(ax.centre + p / ax.scale, L)
        return mx.clip(mx.floor(world / cfg.world.field_cell).astype(_I32), 0, fg - 1)

    return mx.take(cell_packed, cells(ax_y)[:, None] * fg + cells(ax_x)[None, :])


def render_frame(
    s: State,
    cfg: Config,
    size: tuple[int, int] = (1024, 1024),
    viewport: Viewport | None = None,
    layers: frozenset[str] = DEFAULT_LAYERS,
    color: str = "tag",
) -> mx.array:
    """Composite one frame, (H,W,3) uint8, with ``size = (W, H)``.

    Foreground samples (bonds, agent discs, heading noses) are packed as ``(prio << 24) | RGB`` and combined
    by uint32 scatter-max, so the frame is independent of row order and deterministic. Pixels that no sample
    covers show the background. Rendering never modifies ``s``.
    """
    w, h = (int(v) for v in size)
    if w < 1 or h < 1 or w * h + 1 >= 1 << 31:
        raise ValueError(f"size must be positive with W·H < 2^31 − 1, got {size}")
    layers = frozenset(layers)
    if unknown := layers - DEFAULT_LAYERS:
        raise ValueError(f"unknown layers {sorted(unknown)}; expected a subset of {sorted(DEFAULT_LAYERS)}")
    if color not in COLOR_MODES:
        raise ValueError(f"unknown color mode {color!r}; expected one of {COLOR_MODES}")
    axes = _axes(cfg, (w, h), viewport)
    L = float(cfg.world.size)
    n = s.alive.shape[0]

    canvas = mx.zeros((w * h + 1,), dtype=_U32)
    pos = s.pos.astype(_F32)
    point = (mx.zeros((1,), dtype=_F32), mx.zeros((1,), dtype=_F32))

    if "bonds" in layers:
        rows = mx.arange(n, dtype=_I32)[:, None]
        pidx = mx.maximum(s.partner, 0).reshape(-1)
        # Each bond once, from its lower row; both ends alive (always true for a consistent state).
        draw = s.alive[:, None] & (s.partner > rows) & mx.take(s.alive, pidx).reshape(n, layout.B)
        pj = mx.take(pos, pidx, axis=0).reshape(n, layout.B, 2)
        d = ops.min_image(pj - pos[:, None, :], L)  # (N,B,2)
        t = (mx.arange(_BOND_SAMPLES, dtype=_F32) + 1.0) / (_BOND_SAMPLES + 1)
        pts = pos[:, None, None, :] + t[None, None, :, None] * d[:, :, None, :]  # (N,B,S,2)
        nb = n * layout.B * _BOND_SAMPLES
        valid = mx.broadcast_to(draw[:, :, None], (n, layout.B, _BOND_SAMPLES)).reshape(-1)
        packed = mx.broadcast_to(_pack(_PRIO_BOND, _const_rgb(_BOND_RGB, 1)), (nb,))
        pts = pts.reshape(-1, 2)
        canvas = _splat(canvas, pts[:, 0], pts[:, 1], valid, packed, point, axes, L)

    if "agents" in layers:
        r = float(cfg.physics.radius)
        rho = max(1, round(r * axes[0].scale))
        stencil = _disc_stencil(rho)
        e = s.energy.astype(_F32)
        prio = _PRIO_AGENT_BASE + mx.floor(
            _PRIO_AGENT_SPAN * mx.clip(e / float(cfg.repro.min_energy), 0.0, 1.0)
        )
        packed = _pack(prio, _agent_rgb(s, cfg, color))
        canvas = _splat(canvas, pos[:, 0], pos[:, 1], s.alive, packed, stencil, axes, L)
        nose = pos + (_NOSE_DIST * r) * mx.stack([mx.cos(s.heading), mx.sin(s.heading)], axis=1)
        nose_packed = mx.broadcast_to(_pack(_PRIO_NOSE, _const_rgb(_NOSE_RGB, 1)), (n,))
        canvas = _splat(canvas, nose[:, 0], nose[:, 1], s.alive, nose_packed, point, axes, L)

    fg = canvas[: w * h].reshape(h, w)
    out = mx.where((fg >> 24) > 0, fg, _background(s, cfg, axes, layers))
    return mx.stack([(out >> 16) & 255, (out >> 8) & 255, out & 255], axis=-1).astype(mx.uint8)


# --- host-side image output ----------------------------------------------------------------------------


def _as_image(img: Any) -> np.ndarray:
    """A C-contiguous uint8 (H,W,C) or (H,W) numpy image; ``mx.array`` inputs go through ``to_numpy``."""
    a = to_numpy(img) if isinstance(img, mx.array) else np.asarray(img)
    if a.dtype != np.uint8:
        raise ValueError(f"image must be uint8, got {a.dtype}")
    if a.ndim == 3 and a.shape[2] == 1:
        a = a[:, :, 0]
    if not (a.ndim == 2 or (a.ndim == 3 and a.shape[2] in (3, 4))) or a.shape[0] < 1 or a.shape[1] < 1:
        raise ValueError(f"image must be (H,W), (H,W,3) or (H,W,4) with H, W ≥ 1, got {a.shape}")
    return np.ascontiguousarray(a)


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(kind + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", crc)


def _png_bytes(a: np.ndarray) -> bytes:
    h, w = a.shape[:2]
    colour_type = 0 if a.ndim == 2 else {3: 2, 4: 6}[a.shape[2]]  # greyscale, RGB, RGBA
    raw = np.empty((h, 1 + a[0].size), dtype=np.uint8)
    raw[:, 0] = 0  # filter type None on every scanline
    raw[:, 1:] = a.reshape(h, -1)
    ihdr = struct.pack(">IIBBBBB", w, h, 8, colour_type, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", zlib.compress(raw.tobytes(), 6))
        + _png_chunk(b"IEND", b"")
    )


def write_png(path: Path, img: np.ndarray) -> None:
    """Write an 8-bit image as PNG using only the standard library (``zlib``, ``struct``, CRC32).

    ``img`` is (H,W,3) RGB, (H,W,4) RGBA or (H,W) greyscale uint8. The write is atomic (temp file + replace).
    """
    path = Path(path)
    data = _png_bytes(_as_image(img))
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _find_ffmpeg() -> str | None:
    """The bundled ``imageio-ffmpeg`` binary (extra ``video``), else ``ffmpeg`` on PATH, else None."""
    try:
        import imageio_ffmpeg  # type: ignore[import-not-found]

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and Path(exe).exists():
            return str(exe)
    except Exception:  # absent extra, or a platform without a bundled binary
        pass
    return shutil.which("ffmpeg")


VIDEO_SUFFIXES: frozenset[str] = frozenset({".mp4", ".mov", ".mkv", ".m4v"})


class VideoWriter:
    """Pipe RGB24 frames to ffmpeg (libx264, yuv420p, crf 18); falls back to a PNG sequence without ffmpeg.

    - ``path`` with a video suffix (``.mp4``, ``.mov``, ``.mkv``, ``.m4v``) encodes a video. If no ffmpeg
      binary is found (or it cannot be started) it warns and writes ``frame_000000.png, ...`` into ``path``
      minus its suffix instead.
    - ``path`` without a suffix is a directory for a PNG sequence (no warning, ffmpeg not needed).
    ``size`` is ``(W, H)``; every frame must be (H,W,3) uint8. Usable as a context manager.
    """

    def __init__(self, path: Path, size: tuple[int, int], fps: int = 30) -> None:
        path = Path(path)
        w, h = (int(v) for v in size)
        if w < 1 or h < 1:
            raise ValueError(f"size must be positive, got {size}")
        if fps < 1:
            raise ValueError(f"fps must be ≥ 1, got {fps}")
        suffix = path.suffix.lower()
        if suffix and suffix not in VIDEO_SUFFIXES:
            raise ValueError(f"unsupported video suffix {path.suffix!r}; use one of {sorted(VIDEO_SUFFIXES)}")
        self.path = path
        self.size = (w, h)
        self.fps = int(fps)
        self.frames = 0
        self.out_dir: Path | None = None  # set when writing a PNG sequence
        self._proc: subprocess.Popen[bytes] | None = None
        self._log: IO[bytes] | None = None
        self._closed = False
        if suffix:
            exe = _find_ffmpeg()
            if exe is not None:
                try:
                    self._start(exe)
                except OSError as err:
                    warnings.warn(
                        f"could not start ffmpeg ({err}); writing a PNG sequence to {path.with_suffix('')}",
                        stacklevel=2,
                    )
            else:
                warnings.warn(
                    "ffmpeg not found (install the 'video' extra or put ffmpeg on PATH); "
                    f"writing a PNG sequence to {path.with_suffix('')}",
                    stacklevel=2,
                )
        if self._proc is None:
            self.out_dir = path.with_suffix("") if suffix else path
            self.out_dir.mkdir(parents=True, exist_ok=True)

    @property
    def fallback(self) -> bool:
        """True when frames go to a PNG sequence rather than ffmpeg."""
        return self._proc is None

    def _start(self, exe: str) -> None:
        w, h = self.size
        self.path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            exe, "-y", "-loglevel", "error",
            "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
            "-r", str(self.fps), "-i", "-", "-an",
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",  # yuv420p needs even dimensions
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
            str(self.path),
        ]  # fmt: skip
        # stderr goes to a file (closed in close()): a full stderr pipe could otherwise deadlock the writer.
        self._log = tempfile.TemporaryFile()  # noqa: SIM115
        try:
            self._proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self._log
            )
        except OSError:
            self._log.close()
            self._log = None
            raise

    def _ffmpeg_error(self, what: str) -> RuntimeError:
        msg = ""
        if self._log is not None:
            self._log.seek(0)
            msg = self._log.read().decode(errors="replace").strip()
        return RuntimeError(f"ffmpeg {what} while writing {self.path}: {msg or 'no output'}")

    def write(self, frame: np.ndarray) -> None:
        """Append one (H,W,3) uint8 frame."""
        if self._closed:
            raise ValueError("VideoWriter is closed")
        a = _as_image(frame)
        w, h = self.size
        if a.shape != (h, w, 3):
            raise ValueError(f"frame must be ({h}, {w}, 3) uint8, got {a.shape}")
        if self._proc is None:
            assert self.out_dir is not None
            write_png(self.out_dir / f"frame_{self.frames:06d}.png", a)
        else:
            assert self._proc.stdin is not None
            try:
                self._proc.stdin.write(a.tobytes())
            except OSError as err:  # ffmpeg died (BrokenPipeError): report its stderr and stop
                self._closed = True
                code = self._proc.wait()
                failure = self._ffmpeg_error(f"exited with code {code}")
                self._close_log()
                raise failure from err
        self.frames += 1

    def close(self) -> None:
        """Flush and finish the video (idempotent); raises RuntimeError if ffmpeg failed."""
        if self._closed:
            return
        self._closed = True
        if self._proc is None:
            return
        try:
            if self._proc.stdin is not None:
                with contextlib.suppress(OSError):  # a dead ffmpeg is reported through its exit code below
                    self._proc.stdin.close()
            code = self._proc.wait()
            if code != 0:
                raise self._ffmpeg_error(f"exited with code {code}")
        finally:
            self._close_log()

    def _close_log(self) -> None:
        if self._log is not None:
            self._log.close()
            self._log = None

    def __enter__(self) -> VideoWriter:
        return self

    def __exit__(self, exc_type: type[BaseException] | None, *_: object) -> None:
        if exc_type is None:
            self.close()
        else:  # do not mask the original error with an ffmpeg failure
            with contextlib.suppress(RuntimeError):
                self.close()
