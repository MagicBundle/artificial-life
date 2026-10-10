"""Run directories: config, meta, checkpoints, metrics and event logs (§18). Owned by WP5."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Final

import numpy as np

from alife.config import Config
from alife.state import State

FORMAT_VERSION: Final = 1


class CheckpointError(Exception):
    """Incompatible or corrupt checkpoint (version, shape or config-hash mismatch)."""


class RunDir:
    path: Path

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    @classmethod
    def create(cls, path: Path, cfg: Config, *, overwrite: bool = False, parent: str | None = None) -> RunDir:
        raise NotImplementedError

    @classmethod
    def open(cls, path: Path) -> RunDir:
        raise NotImplementedError

    def config(self) -> Config:
        raise NotImplementedError

    def meta(self) -> dict[str, Any]:
        raise NotImplementedError

    def update_meta(self, **fields: Any) -> None:
        raise NotImplementedError

    def save_checkpoint(self, s: State, cfg: Config) -> Path:
        """write_state_file + LATEST + prune."""
        raise NotImplementedError

    def checkpoints(self) -> list[tuple[int, Path]]:
        raise NotImplementedError

    def latest_checkpoint(self) -> Path | None:
        raise NotImplementedError

    def append_metrics(self, rows: Iterable[Mapping[str, Any]]) -> None:
        raise NotImplementedError

    def write_events(
        self, s0: int, s1: int, births: Mapping[str, np.ndarray], deaths: Mapping[str, np.ndarray]
    ) -> None:
        raise NotImplementedError

    def truncate_after(self, step: int) -> None:
        raise NotImplementedError

    def dump_diag(self, s: State, step: int) -> Path:
        raise NotImplementedError


def write_state_file(path: Path, s: State, cfg: Config) -> None:
    """Atomic safetensors write (``*.tmp`` + ``os.replace``) with checkpoint metadata (§18)."""
    raise NotImplementedError


def checkpoint_metadata(path: Path) -> dict[str, str]:
    raise NotImplementedError


def load_checkpoint(path: Path, cfg: Config, *, check_hash: bool = True) -> State:
    """Load a state file; ``CheckpointError`` on a version/shape mismatch, or a config-hash one if checked."""
    raise NotImplementedError
