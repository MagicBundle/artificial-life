"""The run loop: cadences, flushes, checkpoints, SIGINT handling and exit codes (§15.4, §18). Owned by WP5."""

from __future__ import annotations

from collections.abc import Callable
from enum import IntEnum
from typing import Any, NamedTuple

from alife.config import Config
from alife.runio import RunDir
from alife.state import State


class ExitCode(IntEnum):
    OK = 0
    ERROR = 1
    CONFIG = 2
    INVARIANT = 3
    CHECKPOINT = 4
    INTERRUPTED = 130


class RunResult(NamedTuple):
    state: State
    status: str  # done | extinct | violation | interrupted
    exit_code: ExitCode


def run(
    rd: RunDir,
    cfg: Config,
    state: State,
    steps: int,
    *,
    quiet: bool = False,
    on_interval: Callable[[State, dict[str, Any]], None] | None = None,
) -> RunResult:
    """Run ``steps`` steps with one blocking eval per step, writing metrics, events and checkpoints."""
    raise NotImplementedError
