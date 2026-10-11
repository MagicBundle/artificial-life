"""The run loop: cadences, flushes, checkpoints, SIGINT handling and exit codes (§15.4, §18). Owned by WP5.

One compiled step per iteration with exactly one blocking ``mx.eval(state, out)``. The loop holds only the
current ``State`` (the donation rule, F1): each step's ``StepOut`` is kept as small evaluated arrays and
stacked on the host at the next metrics boundary, so no step's counters, ledger or event records are lost.

Cadences (all in units of ``state.step``):
- a **metrics boundary** is a multiple of ``run.metrics_every``: one ``metrics.jsonl`` row, the diag and
  host-ledger checks, ``on_interval``;
- a **checkpoint boundary** is a multiple of ``run.checkpoint_every`` (itself a metrics boundary): the event
  chunk ``[s0, step)`` is written first, then the checkpoint;
- an **unscheduled stop** (end of run, extinction, SIGINT) finishes the current step, writes a possibly short
  interval row, flushes events and checkpoints at that step (§18).
"""

from __future__ import annotations

import signal
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from enum import IntEnum
from typing import Any, NamedTuple

import mlx.core as mx
import numpy as np

from alife import metrics
from alife.config import Config, immigration_on
from alife.runio import RunDir
from alife.state import ERROR_MASK, CounterIdx, Diag, State, to_numpy
from alife.step import StepOut, make_step


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


def diag_names(diag: int) -> list[str]:
    """Names of the ``Diag`` bits set in ``diag`` (unknown bits as hex)."""
    names = [f.name for f in Diag if diag & f and f.name is not None]
    unknown = diag & ~sum(int(f) for f in Diag)
    if unknown:
        names.append(hex(unknown))
    return names


class _Interrupt:
    """SIGINT latch: the first Ctrl-C asks the loop to stop cleanly; a second one raises KeyboardInterrupt."""

    def __init__(self) -> None:
        self.requested = False

    def handler(self, signum: int, frame: Any) -> None:
        self.requested = True
        signal.signal(signal.SIGINT, signal.default_int_handler)
        print("\ninterrupt: finishing the current step, then flushing and checkpointing", file=sys.stderr)


@contextmanager
def _sigint(latch: _Interrupt) -> Iterator[None]:
    """Install the latch for SIGINT (main thread only) and restore the previous handler afterwards."""
    try:
        previous = signal.signal(signal.SIGINT, latch.handler)
    except ValueError:  # not the main thread: signals cannot be handled here
        yield
        return
    try:
        yield
    finally:  # a handler installed outside Python reads as None; fall back to Python's default
        signal.signal(signal.SIGINT, previous if previous is not None else signal.default_int_handler)


def _stack_outs(outs: list[StepOut], steps: list[int]) -> dict[str, np.ndarray]:
    """One host transfer of the interval's stacked counters, ledgers and event records."""
    dev: dict[str, mx.array] = {
        "counters": mx.stack([o.counters for o in outs]),
        "ledgers": mx.stack([o.ledger for o in outs]),
    }
    for kind in ("births", "deaths"):
        recs = [getattr(o, kind) for o in outs]
        for name in recs[0]._fields:
            dev[f"{kind}.{name}"] = mx.stack([getattr(r, name) for r in recs])
    mx.eval(*dev.values())
    host = {k: to_numpy(v) for k, v in dev.items()}
    t = np.asarray(steps, dtype=np.int32)[:, None]
    for kind in ("births", "deaths"):
        host[f"{kind}.step"] = np.broadcast_to(t, host[f"{kind}.valid"].shape)
    return host


def _records(host: dict[str, np.ndarray], kind: str) -> dict[str, np.ndarray]:
    """The valid ``kind`` records of an interval as flat columns (the §18 event columns)."""
    keep = host[f"{kind}.valid"].reshape(-1).astype(bool)
    prefix = f"{kind}."
    return {
        k[len(prefix) :]: v.reshape(-1)[keep]
        for k, v in host.items()
        if k.startswith(prefix) and k != f"{kind}.valid"
    }


def _concat(chunks: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}


def _status_line(row: dict[str, Any]) -> str:
    deaths = row["deaths_starved"] + row["deaths_killed"] + row["deaths_aged"]
    sps = row.get("steps_per_s")
    parts = [
        f"step {row['step']}",
        f"pop {row['n_alive']:.0f}" if row["n_alive"] is not None else "pop -",
        f"births {row['births']} deaths {deaths}",
        f"bonds {row['bonds_total']:.0f}" if row["bonds_total"] is not None else "bonds -",
    ]
    if "bodies_largest" in row:
        parts.append(f"largest body {row['bodies_largest']}")
    parts.append(f"{sps:.1f} steps/s" if sps else "- steps/s")
    parts.append("ledger OK" if row["ledger_error"] == 0 else f"LEDGER ERROR {row['ledger_error']}")
    return " | ".join(parts)


def run(
    rd: RunDir,
    cfg: Config,
    state: State,
    steps: int,
    *,
    quiet: bool = False,
    on_interval: Callable[[State, dict[str, Any]], None] | None = None,
) -> RunResult:
    """Run ``steps`` steps with one blocking eval per step, writing metrics, events and checkpoints.

    The trajectory depends only on ``state`` and the dynamics config: metrics, checkpoint and render cadences
    never touch it (observer purity). A checkpoint at the starting step is written if the run directory lacks
    one. Extinction (no agent alive with immigration off) is read from each step's already evaluated
    ``N_ALIVE`` counter, which adds no sync. On an error diag bit or a non-zero host ledger error the state
    is dumped to ``diag/`` and the run stops with ``violation`` (exit 3); an invalid state is never
    checkpointed.
    ``on_interval(state, row)`` runs after each metrics row is written and must not retain ``state``.
    """
    if steps < 0:
        raise ValueError(f"steps must be ≥ 0, got {steps}")
    run_cfg = cfg.run
    me, ce, he = run_cfg.metrics_every, run_cfg.checkpoint_every, run_cfg.heavy_metrics_every
    immigration = immigration_on(cfg.population)
    t = int(to_numpy(state.step))
    end = t + steps
    rd.update_meta(status="running")
    if not any(st == t for st, _ in rd.checkpoints()):
        rd.save_checkpoint(state, cfg)
    first_row = not rd.metrics_rows()
    if not immigration and not bool(to_numpy(state.alive).any()):
        rd.update_meta(status="extinct")
        return RunResult(state, "extinct", ExitCode.OK)

    step_fn = make_step(cfg)
    latch = _Interrupt()
    outs: list[StepOut] = []
    out_steps: list[int] = []
    births: list[dict[str, np.ndarray]] = []
    deaths: list[dict[str, np.ndarray]] = []
    chunk_s0 = ckpt_step = t
    status = "done"

    def checkpoint() -> None:
        """Write the event chunk ``[chunk_s0, t)``, then checkpoint at ``t`` (logs first, §18)."""
        nonlocal births, deaths, chunk_s0, ckpt_step
        rd.write_events(chunk_s0, t, _concat(births), _concat(deaths))
        births, deaths, chunk_s0 = [], [], t
        rd.save_checkpoint(state, cfg)
        ckpt_step = t

    mx.reset_peak_memory()
    t_wall = time.perf_counter()

    with _sigint(latch):
        while t < end:
            state, out = step_fn(state)  # rebinding drops the only reference to the previous state
            mx.eval(state, out)
            outs.append(out)
            out_steps.append(t)
            t += 1
            extinct = not immigration and int(to_numpy(out.counters)[CounterIdx.N_ALIVE]) == 0
            stop = extinct or latch.requested or t == end
            if t % me and not stop:
                continue

            wall = time.perf_counter() - t_wall
            host = _stack_outs(outs, out_steps)
            births.append(_records(host, "births"))
            deaths.append(_records(host, "deaths"))
            snap = metrics.snapshot_metrics(state, cfg, heavy=t % he == 0)
            perf = {"wall_s": wall, "peak_mem_gb": mx.get_peak_memory() / 1e9}
            row = metrics.interval_row(t, host["counters"], host["ledgers"], snap, perf)
            if first_row:
                row["carrying_capacity_est"] = metrics.carrying_capacity(state, cfg)
                first_row = False
            rd.append_metrics([row])
            outs, out_steps = [], []

            diag = int(snap["diag"])
            if diag & ERROR_MASK or snap["ledger_error"] != 0:
                rd.write_events(chunk_s0, t, _concat(births), _concat(deaths))
                path = rd.dump_diag(state, t)
                problems = diag_names(diag & ERROR_MASK)
                if snap["ledger_error"] != 0:
                    problems.append(f"ledger_error={snap['ledger_error']}")
                print(
                    f"invariant violation at step {t}: {', '.join(problems)}; state dumped to {path}",
                    file=sys.stderr,
                )
                rd.update_meta(status="violation")
                return RunResult(state, "violation", ExitCode.INVARIANT)

            if t % ce == 0 or stop:
                checkpoint()
            if not quiet:
                print(_status_line(row), flush=True)
            if on_interval is not None:
                on_interval(state, row)
            if stop or latch.requested:  # a SIGINT may also arrive during the flush or the callback
                if ckpt_step != t:
                    checkpoint()
                status = "extinct" if extinct else "done" if t == end else "interrupted"
                break
            mx.reset_peak_memory()
            t_wall = time.perf_counter()

    code = ExitCode.INTERRUPTED if status == "interrupted" else ExitCode.OK
    rd.update_meta(status=status)
    if not quiet and status != "done":
        print(f"{status} at step {t}", file=sys.stderr)
    return RunResult(state, status, code)
