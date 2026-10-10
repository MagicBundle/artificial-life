"""Headless MLX rasteriser, stdlib PNG writer and ffmpeg video writer (§20.1). Owned by WP6."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np

from alife.config import Config
from alife.state import State


@dataclass(frozen=True)
class Viewport:
    cx: float
    cy: float
    zoom: float = 1.0


DEFAULT_LAYERS: frozenset[str] = frozenset({"resource", "chem", "bonds", "agents"})


def render_frame(
    s: State,
    cfg: Config,
    size: tuple[int, int] = (1024, 1024),
    viewport: Viewport | None = None,
    layers: frozenset[str] = DEFAULT_LAYERS,
    color: str = "tag",
) -> mx.array:
    """Composite one frame, (H,W,3) uint8."""
    raise NotImplementedError


def write_png(path: Path, img: np.ndarray) -> None:
    """Write an RGB uint8 image as PNG using only the standard library."""
    raise NotImplementedError


class VideoWriter:
    """Pipe RGB24 frames to ffmpeg (libx264); falls back to a PNG sequence without ffmpeg."""

    def __init__(self, path: Path, size: tuple[int, int], fps: int = 30) -> None:
        raise NotImplementedError

    def write(self, frame: np.ndarray) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError
