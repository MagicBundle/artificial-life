"""Re-simulate a run in memory to steps it no longer has checkpoints for, verified against its metrics.

The run directory is read only. Starting from the newest checkpoint at or before the first target, the
compiled step is replayed exactly as ``runner.run`` does (one blocking eval per step, §15.4), so under the
§14.2 determinism contract (same device, MLX version and config) the trajectory is the run's own. Every
``metrics.jsonl`` row on the way is recomputed from the per-step counters and ledgers and compared with the
stored row (event sums, interval means, the ledger and ``E_total``); the first difference raises
``ResimMismatch``. States at the targets are written with ``runio.write_state_file`` to a cache directory
outside the run, where ``runio.load_checkpoint`` reads them like checkpoints.

Usage: ``python -m alife.analysis.resim RUN OUT_DIR STEP [STEP ...]``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import mlx.core as mx

from alife import metrics, runio, spatial
from alife.config import dims
from alife.runio import CheckpointError, RunDir
from alife.state import to_numpy
from alife.step import make_step

# Interval-row keys that depend on wall-clock time, not on the trajectory.
_PERF_KEYS = frozenset({"wall_s", "steps_per_s", "peak_mem_gb"})


class ResimMismatch(RuntimeError):
    """The re-simulated trajectory differs from the stored metrics row."""


def state_path(out_dir: Path, step: int) -> Path:
    return Path(out_dir) / f"step_{step:010d}.safetensors"


def cached_states(out_dir: Path) -> list[tuple[int, Path]]:
    """``(step, path)`` of the state files in ``out_dir``, ascending by step."""
    found = []
    for p in Path(out_dir).glob("step_*.safetensors"):
        try:
            found.append((int(p.stem.removeprefix("step_")), p))
        except ValueError:
            continue
    return sorted(found)


def compare_row(
    stored: dict[str, Any],
    counters: list[mx.array],
    ledgers: list[mx.array],
    total: int,
    *,
    nd: int = 2,
    weather: bool = False,
) -> list[str]:
    """Keys whose recomputed value differs from ``stored`` (interval keys plus ``E_total``); ``nd`` and
    ``weather`` are the run's dims and ``weather.enabled`` (``metrics.interval_row``)."""
    c = to_numpy(mx.stack(counters))
    lg = to_numpy(mx.stack(ledgers))
    row = metrics.interval_row(int(stored["step"]), c, lg, {}, {}, nd=nd, weather=weather)
    bad = [k for k, v in row.items() if k not in _PERF_KEYS and k in stored and stored[k] != v]
    if "E_total" in stored and stored["E_total"] != total:
        bad.append("E_total")
    return bad


def resimulate(
    run: Path,
    targets: Sequence[int],
    out_dir: Path,
    *,
    verify: bool = True,
    log: Callable[[str], None] = print,
) -> list[Path]:
    """Write the run's states at ``targets`` to ``out_dir``; returns their paths in ascending step order.

    Targets that already have a file in ``out_dir`` are not rewritten. The pass starts from the newest run
    checkpoint or cached state at or before the first missing target. ``verify`` compares every stored
    metrics row passed on the way and raises ``ResimMismatch`` on the first difference.
    """
    rd = RunDir.open(Path(run))
    cfg = rd.config()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = sorted({int(t) for t in targets})
    todo = [t for t in targets if not state_path(out_dir, t).is_file()]
    if not todo:
        return [state_path(out_dir, t) for t in targets]
    start = sorted((st, p) for st, p in [*rd.checkpoints(), *cached_states(out_dir)] if st <= todo[0])
    if not start:
        kept = sorted({st for st, _ in rd.checkpoints()})
        raise FileNotFoundError(
            f"{rd.path}: no checkpoint or cached state at or before step {todo[0]} "
            f"(checkpoints kept: {kept}; a world saved with saves single keeps only its newest save)"
        )
    t, path = start[-1]
    state = runio.load_checkpoint(path, cfg)
    rows = {int(r["step"]): r for r in rd.metrics_rows()} if verify else {}
    log(f"resim: {rd.path} from step {t} to {todo[-1]}, verify={verify} ({len(rows)} rows)")

    spatial.prepare_knn(cfg)  # the self-check of the fused search, once per process (§4)
    step_fn = make_step(cfg)
    counters: list[mx.array] = []
    ledgers: list[mx.array] = []
    checked = 0
    t_wall, t_log = time.perf_counter(), t
    end = todo[-1]
    if t in todo:  # a target that is itself a checkpoint
        runio.write_state_file(state_path(out_dir, t), state, cfg)
    while t < end:
        state, out = step_fn(state)
        mx.eval(state, out.counters, out.ledger)
        counters.append(out.counters)
        ledgers.append(out.ledger)
        t += 1
        stored = rows.get(t)
        if stored is not None:
            if stored["steps"] == len(counters):
                bad = compare_row(
                    stored,
                    counters,
                    ledgers,
                    int(to_numpy(state.total)),
                    nd=dims(cfg).nd,
                    weather=cfg.weather.enabled,
                )
                if bad:
                    raise ResimMismatch(f"step {t}: re-simulated values differ from metrics.jsonl in {bad}")
                checked += 1
            counters, ledgers = [], []
        if t in todo:
            p = state_path(out_dir, t)
            runio.write_state_file(p, state, cfg)
            log(f"resim: wrote {p}")
        if t - t_log >= 10_000:
            rate = (t - t_log) / (time.perf_counter() - t_wall)
            log(f"resim: step {t}, {rate:.1f} steps/s, {checked} rows verified")
            t_wall, t_log = time.perf_counter(), t
    _write_report(out_dir, rd, checked)
    return [state_path(out_dir, t) for t in targets]


def _write_report(out_dir: Path, rd: RunDir, checked: int) -> None:
    """A small JSON sidecar recording where the states came from and how much of them was verified."""
    path = out_dir / "resim.json"
    rec = {
        "source_run": str(rd.path),
        "source_config_hash": rd.meta().get("config_hash"),
        "mlx_version": mx.__version__,
        "device": str(mx.default_device()),
        "steps": [st for st, _ in cached_states(out_dir)],  # every cached state, from any pass
        "rows_verified_last_pass": checked,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    path.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m alife.analysis.resim", description=__doc__.split("\n")[0])
    ap.add_argument("run", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("steps", type=int, nargs="+")
    ap.add_argument("--no-verify", action="store_true")
    a = ap.parse_args(argv)
    try:
        resimulate(a.run, a.steps, a.out_dir, verify=not a.no_verify, log=lambda m: print(m, flush=True))
    except ResimMismatch as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except (
        FileNotFoundError,
        ValueError,
        CheckpointError,
    ) as e:  # no save to start from: a message, not a traceback
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
