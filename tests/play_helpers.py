"""Shared pieces of the play-loop tests (DESIGN §33.2): a fake clock, scripted control sources, a sink that
can fail, the world configs and a runner driver, and the comparison of two run directories."""

from __future__ import annotations

import dataclasses
import io
import json
import os
import random
import subprocess
import sys
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from alife import play, runner
from alife.config import Config, load_config
from alife.director import make_director
from alife.game import observer
from alife.game.social import SOCIAL_V1
from alife.runio import DIRECTOR_LOG_FILE, RunDir, load_checkpoint
from alife.state import State, init_state, state_digest, to_numpy
from play_decode import Decoded, decode
from weather_helpers import storm3d_config

PERF_KEYS = {"wall_s", "steps_per_s", "peak_mem_gb"}
CADENCE = [
    "brain.weight_dtype=float32",
    "run.metrics_every=10",
    "run.heavy_metrics_every=10",
    "run.checkpoint_every=30",
    "run.diversity_sample=64",
]


class FakeClock:
    """Time that passes only when the scheduler waits: ``wait`` returns the lines due now at once and
    otherwise advances by the timeout. ``max_waits`` guards a test against a loop that waits for ever."""

    def __init__(self, start: float = 1000.0, max_waits: int = 100_000) -> None:
        self.now = start
        self.waits = 0
        self.max_waits = max_waits

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds

    def wait(self, control: Any, timeout: float) -> list[str]:
        self.waits += 1
        assert self.waits <= self.max_waits, "the pacer waited for ever"
        lines = control.poll(0)
        if lines:
            return lines
        self.now += timeout
        return control.poll(0)


class ScriptControl:
    """Control lines by poll: the i-th call of ``poll`` returns ``script[i]`` (nothing once it is used up)."""

    def __init__(self, script: Sequence[Sequence[str]]) -> None:
        self.script = [list(s) for s in script]
        self.calls = 0

    def poll(self, timeout: float | None = 0) -> list[str]:
        i, self.calls = self.calls, self.calls + 1
        return self.script[i] if i < len(self.script) else []

    def close(self) -> None:
        pass


class TimedControl:
    """Control lines at fake times: a line is returned by the first ``poll`` at or after its time."""

    def __init__(self, clock: FakeClock, lines: Iterable[tuple[float, str]]) -> None:
        self.clock = clock
        self.lines = sorted(lines, key=lambda x: x[0])

    def poll(self, timeout: float | None = 0) -> list[str]:
        due = [text for at, text in self.lines if at <= self.clock.now]
        self.lines = [(at, text) for at, text in self.lines if at > self.clock.now]
        return due

    def close(self) -> None:
        pass


class FailingSink(io.BytesIO):
    """A stream that raises ``BrokenPipeError`` from the ``after``-th write on (the reader closed)."""

    def __init__(self, after: int) -> None:
        super().__init__()
        self.after = after
        self.writes = 0

    def write(self, data: Any) -> int:
        self.writes += 1
        if self.writes > self.after:
            raise BrokenPipeError("reader closed")
        return super().write(data)


class CostedPacer(play.PlayPacer):
    """A pacer whose steps take ``cost`` fake seconds (the fake clock does not advance while a step runs)."""

    cost = 0.0

    def after_step(self, state: State, out: Any, t: int, social: list[dict[str, Any]]) -> None:
        self.clock.now += self.cost  # type: ignore[attr-defined]
        super().after_step(state, out, t, social)


def snapshots(monkeypatch: pytest.MonkeyPatch, every: int) -> None:
    """The observers of this test snapshot every ``every`` steps (the module default the runner reads)."""
    monkeypatch.setattr(observer, "DEFAULT_SOCIAL", dataclasses.replace(SOCIAL_V1, snapshot_every=every))


def world(kind: str, *extra: str, observe: bool = True, director: bool = False) -> Config:
    """A test world: ``2d`` (tiny), ``3d`` (tiny3d) or ``storm`` (256 rows of 64k3d, storm overlay)."""
    ov = [*CADENCE, f"game.observe={'true' if observe else 'false'}", *extra]
    if director:
        ov += ["director.enabled=true", "director.rule_every=20", "population.n_init=24", "run.director=live"]
    if kind == "storm":
        base = [o for o in ov if not o.startswith("brain.weight_dtype")]
        return storm3d_config(*base)
    return load_config(preset="tiny" if kind == "2d" else "tiny3d", overrides=ov)


def uids(cfg: Config) -> list[int]:
    return [int(u) for u in to_numpy(init_state(cfg).uid) if u >= 0]


def run_plain(
    cfg: Config, path: Path, steps: int, *, start: State | None = None
) -> tuple[RunDir, runner.RunResult]:
    rd = RunDir.create(path, cfg)
    director = make_director(rd, cfg, 0) if cfg.run.director != "off" else None
    return rd, runner.run(rd, cfg, start or init_state(cfg), steps, quiet=True, director=director)


def run_paced(
    cfg: Config,
    path: Path,
    steps: int,
    script: Any,
    *,
    clock: FakeClock | None = None,
    out: io.BytesIO | None = None,
    options: play.PlayOptions | None = None,
    pacer_class: type[play.PlayPacer] = play.PlayPacer,
) -> tuple[RunDir, runner.RunResult, play.PlayPacer, io.BytesIO]:
    rd = RunDir.create(path, cfg)
    out = out if out is not None else io.BytesIO()
    pacer = pacer_class(
        out,
        script if hasattr(script, "poll") else ScriptControl(script),
        options or play.PlayOptions(rate=float("inf"), paused=False, fresh=True),
        clock=clock or FakeClock(),
        note=lambda m: None,
    )
    director = make_director(rd, cfg, 0) if cfg.run.director != "off" else None
    res = runner.run(rd, cfg, init_state(cfg), steps, quiet=True, director=director, pacer=pacer)
    return rd, res, pacer, out


def random_script(seed: int, n_polls: int, uid_pool: Sequence[int]) -> list[list[str]]:
    """A seeded control script: pauses, rates, ``step`` batches, colours, ``every``, ``focus`` and ``follow``,
    ``idnt`` and ``labels``, ending in ``rate max`` and ``run`` so that the run finishes."""
    rng = random.Random(seed)
    uid_choice = list(uid_pool)
    modes = ["tag", "energy", "clade", "house", "kin", "archetype", "intent"]
    makers = [
        lambda: "pause",
        lambda: "run",
        lambda: f"step {rng.randint(1, 4)}",
        lambda: f"rate {rng.choice(['0.25', '1', '2', '16', 'max', '0'])}",
        lambda: f"color {rng.choice(modes)}",
        lambda: f"every {rng.choice(['1', '2', '7', 'auto'])}",
        lambda: f"focus {rng.choice(uid_choice)}",
        lambda: f"follow {rng.choice(uid_choice)}",
        lambda: rng.choice(["focus off", "follow off"]),
        lambda: "idnt",
        lambda: f"labels {rng.choice(['on', 'off'])}",
        lambda: rng.choice(["bogus", "rate 99", "fields on", "color nope", "focus -1"]),  # refused, ignored
        # G3b's presentation lines (DESIGN §34.6.1): an armed pause, a watch, an acknowledgement, a preview
        lambda: rng.choice(
            ["autopause on", "autopause off", "autopause chapter on", "autopause chapter off"]
        ),
        lambda: rng.choice(["autopause alert pop_fall on", "autopause alert nope off"]),
        lambda: rng.choice(["watch 3", "watch off", "unwatch 3", "ack 466250600"]),
        lambda: rng.choice(
            ["power preview food id=1 mult=1.3", "power strike id=2", "power cancel food id=3"]
        ),
        lambda: "intent make it rain",
    ]
    script: list[list[str]] = []
    for _ in range(n_polls):
        script.append([rng.choice(makers)() for _ in range(rng.choice([0, 0, 1, 1, 2, 3]))])
    return [*script, ["rate max", "run"]]


# --- comparing run directories -------------------------------------------------------------------------


def rows(rd: RunDir) -> list[dict[str, Any]]:
    """The metrics rows without the wall-clock keys."""
    return [{k: v for k, v in r.items() if k not in PERF_KEYS} for r in rd.metrics_rows()]


def checkpoint_digests(rd: RunDir) -> dict[int, str]:
    cfg = rd.config()
    return {st: state_digest(load_checkpoint(path, cfg)) for st, path in rd.checkpoints()}


def events(rd: RunDir) -> dict[str, dict[str, np.ndarray]]:
    return {kind: rd.read_events(kind) for kind in ("births", "deaths")}


def same_events(a: RunDir, b: RunDir) -> bool:
    ea, eb = events(a), events(b)
    return all(
        [(s0, s1) for s0, s1, _ in a.event_files(kind)] == [(s0, s1) for s0, s1, _ in b.event_files(kind)]
        and ea[kind].keys() == eb[kind].keys()
        and all(np.array_equal(ea[kind][k], eb[kind][k]) for k in ea[kind])
        for kind in ("births", "deaths")
    )


def sidecar(rd: RunDir, step: int) -> dict[str, np.ndarray] | None:
    return rd.read_observers(step)


def same_sidecar(a: RunDir, b: RunDir, step: int) -> bool:
    x, y = sidecar(a, step), sidecar(b, step)
    if x is None or y is None:
        return x is None and y is None
    return x.keys() == y.keys() and all(np.array_equal(x[k], y[k]) for k in x if k != "meta")


def director_log(rd: RunDir) -> list[dict[str, Any]]:
    return rd.read_jsonl(DIRECTOR_LOG_FILE) if (rd.path / DIRECTOR_LOG_FILE).exists() else []


def social_text(rd: RunDir) -> str:
    return rd.social_path.read_text() if rd.social_path.exists() else ""


def decoded(out: io.BytesIO) -> Decoded:
    return decode(out.getvalue())


def json_dump(x: Any) -> str:
    return json.dumps(x, sort_keys=True)


# --- the command as a process ----------------------------------------------------------------------------

CODE = (
    "import sys; import mlx.core as mx; mx.set_default_device(mx.cpu); "
    "from alife.cli import main; sys.exit(main(sys.argv[1:]))"
)  # the CPU, set before alife is imported: the GPU belongs to the coordinator


def spawn(*args: str, stdout: int = subprocess.PIPE) -> subprocess.Popen[bytes]:
    """``alife <args>`` as a subprocess on the CPU with pipes for stdin, stdout and stderr."""
    env = {**os.environ, "TMPDIR": os.environ.get("TMPDIR", "/tmp")}
    return subprocess.Popen(
        [sys.executable, "-c", CODE, *args],
        stdin=subprocess.PIPE,
        stdout=stdout,
        stderr=subprocess.PIPE,
        env=env,
    )


def finish(p: subprocess.Popen[bytes], send: str = "", timeout: float = 240) -> tuple[int, bytes, str]:
    """Send ``send`` and close stdin (EOF is a quit), wait for the exit; ``(code, stdout, stderr)``."""
    out, err = p.communicate(send.encode(), timeout=timeout)
    return p.returncode, out, err.decode(errors="replace")


def wait_for(path: Path, timeout: float = 120) -> None:
    deadline = time.monotonic() + timeout
    while not path.exists():
        assert time.monotonic() < deadline, f"{path} did not appear"
        time.sleep(0.05)
