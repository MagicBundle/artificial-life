"""The ``alife`` command line: run, resume, render, inspect, view, bench (§21). Owned by WP5."""

from __future__ import annotations

from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point; returns a ``runner.ExitCode`` value."""
    raise NotImplementedError
