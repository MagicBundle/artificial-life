"""Child process of the compiled sphere tests: one configuration per process, on an empty kernel cache.

MLX 0.32.3's compile cache can hand a compiled function another one's kernel when their scalar constants
print alike, and the CPU kernel cache is machine-wide (DESIGN F21 item 6, docs/SPHERE.md 10.1). The guard
branch ``fix/mlx-compile-collision`` is not merged, so the sphere tests follow the fallback of the spec: a
compiled sphere configuration runs in its own process whose ``TMPDIR`` is a fresh empty directory
(``golden_run.cold_env``), and a test that compares the compiled step with the eager one permutes their order
between processes. As a script: ``sphere_run.py KIND CONFIG_JSON DEVICE ARGS...``; prints one JSON line.

* ``trajectory STEPS CHECK_EVERY DIGEST_AT``: ``STEPS`` compiled steps; every ``CHECK_EVERY`` steps the full
  host check against the previous step's state and records (``invariants.full_check``) and the diag word;
  the ledger, the counters and the peak hash occupancy are accumulated; ``state_digest`` at the steps of the
  comma list ``DIGEST_AT``.
* ``compare STEPS ORDER``: from one state, each step is run by the eager and the compiled step in the order
  ``ORDER`` (``eager-first`` or ``compiled-first``), compared field by field (integers exactly, floats to
  1e-5), and the walk continues from the eager result.
* ``wx_compare STEPS ORDER``: the weather alone (``weather.advance``, the pools of step t, the engine's key
chain)
  from the rest state, eager and compiled in the order ``ORDER``, compared bit for bit (the walk continues
  from
  the eager result); the last line also reports the largest field value.
* ``lookups``: ``sphere_check.require_compile_safe`` for the configuration (a compiled position-to-cell
  lookup equals the eager one).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from alife.config import Config, config_from_dict, config_to_dict

HERE = Path(__file__).resolve().parent


def spawn(
    kind: str, cfg: Config, tmpdir: Path, *args: Any, device: str | None = None, timeout: float = 3000
) -> dict[str, Any]:
    """Run ``kind`` for ``cfg`` in a child whose ``TMPDIR`` is the empty directory ``tmpdir`` (created here);
    ``device`` ``"cpu"`` or ``"gpu"`` (default: the test process's default device)."""
    import golden_run  # tests/ is on sys.path under pytest

    if device is None:
        device = "cpu" if mx.default_device().type == mx.DeviceType.cpu else "gpu"
    argv = [sys.executable, str(HERE / "sphere_run.py"), kind, json.dumps(config_to_dict(cfg)), device]
    argv += [str(a) for a in args]
    done = subprocess.run(
        argv, env=golden_run.cold_env(tmpdir), capture_output=True, text=True, timeout=timeout, check=False
    )
    if done.returncode != 0:
        raise RuntimeError(f"sphere child failed ({done.returncode}):\n{done.stderr[-3000:]}")
    return json.loads(done.stdout.strip().splitlines()[-1])


def _trajectory(cfg, steps: int, every: int, digest_at: set[int]) -> dict:
    from alife.invariants import full_check
    from alife.state import CounterIdx, init_state, state_digest, to_numpy
    from alife.step import make_step

    fn = make_step(cfg, compile=True)
    s = init_state(cfg)
    prev = s
    problems: list[str] = []
    digests: dict[str, str] = {}
    ledger = np.zeros(16, np.int64)
    max_occ = overflow = 0
    for i in range(1, steps + 1):
        s, out = fn(s)
        mx.eval(s, out)
        ledger += to_numpy(out.ledger).astype(np.int64)
        c = to_numpy(out.counters)
        max_occ = max(max_occ, int(c[CounterIdx.MAX_OCCUPANCY]))
        overflow += int(c[CounterIdx.CELL_OVERFLOW])
        if i % every == 0 or i == steps:
            problems += [f"step {i}: {p}" for p in full_check(s, cfg, prev, out)]
        if i in digest_at:
            digests[str(i)] = state_digest(s)
        prev = s
    return {
        "problems": problems,
        "diag": int(to_numpy(s.diag)),
        "digests": digests,
        "ledger": ledger.tolist(),
        "max_occupancy": max_occ,
        "overflow": overflow,
        "alive": int(to_numpy(s.alive.sum())),
    }


def _compare(cfg, steps: int, order: str) -> dict:
    from alife.state import init_state, to_numpy
    from alife.step import make_step

    eager, compiled = make_step(cfg, compile=False), make_step(cfg, compile=True)
    s = init_state(cfg)
    bad: list[str] = []
    for i in range(steps):
        if order == "eager-first":
            a, out_a = eager(s)
            mx.eval(a, out_a)
            b, out_b = compiled(s)
        else:
            b, out_b = compiled(s)
            mx.eval(b, out_b)
            a, out_a = eager(s)
        mx.eval(a, out_a, b, out_b)
        for name in ("counters", "ledger"):
            if not np.array_equal(to_numpy(getattr(out_a, name)), to_numpy(getattr(out_b, name))):
                bad.append(f"step {i}: {name}")
        for name in type(a)._fields:
            x, y = to_numpy(getattr(a, name)), to_numpy(getattr(b, name))
            if x.dtype.kind == "f":
                if not np.allclose(x, y, rtol=1e-5, atol=1e-5):
                    bad.append(f"step {i}: {name} (max {np.abs(x - y).max():.3g})")
            elif not np.array_equal(x, y):
                bad.append(f"step {i}: {name}")
        s = a
    return {"mismatch": bad, "steps": steps}


def _wx_compare(cfg, steps: int, order: str) -> dict:
    from alife import weather
    from alife.rng import Stream, step_keys

    def one(wx, t, key):
        key_next, ks = step_keys(key)
        ws, _ = weather.advance(weather.unpack(wx), t, mx.random.split(ks[Stream.WEATHER], 2)[0], cfg)
        return weather.pack(ws), key_next

    eager, compiled = one, mx.compile(one)
    wx, key = weather.rest_array(cfg, 0), mx.random.key(7)
    bad: list[str] = []
    peak = 0.0
    for i in range(steps):
        t = mx.array(i, dtype=mx.int32)
        if order == "eager-first":
            a, ka = eager(wx, t, key)
            mx.eval(a, ka)
            b, kb = compiled(wx, t, key)
        else:
            b, kb = compiled(wx, t, key)
            mx.eval(b, kb)
            a, ka = eager(wx, t, key)
        mx.eval(a, ka, b, kb)
        if not np.array_equal(np.array(a), np.array(b)):
            bad.append(f"step {i}: max {np.abs(np.array(a) - np.array(b)).max():.3g}")
        peak = max(peak, float(np.abs(np.array(a)[2:8]).max()))
        wx, key = a, ka
    return {"mismatch": bad, "steps": steps, "peak_wind": peak}


def _lookups(cfg) -> dict:
    from alife import sphere_check

    sphere_check.require_compile_safe(cfg)
    return {"ok": True}


def main(argv: list[str]) -> int:
    kind, cfg_json, device, *rest = argv
    if device == "cpu":
        mx.set_default_device(mx.cpu)
    cfg = config_from_dict(json.loads(cfg_json))
    if kind == "trajectory":
        out = _trajectory(cfg, int(rest[0]), int(rest[1]), {int(x) for x in rest[2].split(",") if x})
    elif kind == "compare":
        out = _compare(cfg, int(rest[0]), rest[1])
    elif kind == "wx_compare":
        out = _wx_compare(cfg, int(rest[0]), rest[1])
    elif kind == "lookups":
        out = _lookups(cfg)
    else:
        raise SystemExit(f"unknown kind {kind!r}")
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
