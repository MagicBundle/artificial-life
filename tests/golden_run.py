"""The golden tests' trajectories, run in a child process on an empty MLX kernel cache (DESIGN §22.0, I58).

A golden digest depends on the compiler that built MLX's CPU kernels (``g++``: the Linux digests changed with
the CI image, DESIGN §22.0), and a test process's kernel cache is seeded from the shared one (the developer's,
or the one CI restores), which may hold libraries that an older compiler or another configuration built. So a
golden test never runs its trajectory in the test process: it calls ``run_cold``, which starts this module
in a child whose ``TMPDIR`` is a fresh empty directory, so that every kernel of the 50 steps is compiled
by the compiler of this machine and nothing else is loaded. The cost is one cold compile of the
configuration, and nothing is handed back to the shared cache.

As a script: ``golden_run.py KIND CONFIG_JSON DEVICE [STEPS]`` with KIND ``2d`` or ``3d`` (the trajectory of
``test_determinism`` or ``test_determinism_3d``, whose ``_run`` and ``STEPS`` it uses) or ``probe`` (no
trajectory: compile one function and report the cache); prints one JSON line.
"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import mlx.core as mx

from alife.config import Config, config_from_dict, config_to_dict

HERE = Path(__file__).resolve().parent
XDIST_VARS = ("PYTEST_XDIST_WORKER", "PYTEST_XDIST_WORKER_COUNT", "ALIFE_TEST_JIT_SHARED")


def cold_env(tmpdir: Path) -> dict[str, str]:
    """This process's environment with ``TMPDIR`` set to the empty directory ``tmpdir`` (created here).
    Whatever kernel cache the process itself uses, private and seeded or shared, the child sees none."""
    tmpdir.mkdir(parents=True, exist_ok=True)
    if any(tmpdir.iterdir()):
        raise ValueError(f"the cold cache directory is not empty: {tmpdir}")
    env = {k: v for k, v in os.environ.items() if k not in XDIST_VARS}
    env["TMPDIR"] = f"{tmpdir}{os.sep}"
    return env


def _spawn(kind: str, cfg: Config, tmpdir: Path, *extra: str, timeout: float) -> dict[str, Any]:
    device = "cpu" if mx.default_device().type == mx.DeviceType.cpu else "gpu"
    argv = [
        sys.executable,
        str(HERE / "golden_run.py"),
        kind,
        json.dumps(config_to_dict(cfg)),
        device,
        *extra,
    ]
    done = subprocess.run(
        argv, env=cold_env(tmpdir), capture_output=True, text=True, timeout=timeout, check=False
    )
    if done.returncode != 0:
        raise RuntimeError(f"golden child failed ({done.returncode}):\n{done.stderr[-3000:]}")
    return json.loads(done.stdout.strip().splitlines()[-1])


def run_cold(
    kind: str, cfg: Config, tmpdir: Path, steps: int | None = None, timeout: float = 3000
) -> dict[str, Any]:
    """The legacy digest of the ``kind`` (``"2d"`` or ``"3d"``) trajectory of ``cfg``, computed in a
    child whose kernel cache is the empty directory ``tmpdir``. Returns ``{"digest", "buoyancy" (3d),
    "libs_at_start", "libs_at_end", "tmpdir"}``."""
    return _spawn(kind, cfg, tmpdir, *([str(steps)] if steps is not None else []), timeout=timeout)


def probe(cfg: Config, tmpdir: Path, timeout: float = 300) -> dict[str, Any]:
    """What a cold child sees (``libs_at_start``) and what it holds after compiling one small function."""
    return _spawn("probe", cfg, tmpdir, timeout=timeout)


def _libs() -> list[str]:
    root = Path(os.environ.get("TMPDIR") or "/tmp")
    return sorted(p.name for p in (root / "mlx" / mx.__version__ / "cpu").glob("lib*.so"))


def main(argv: list[str]) -> int:
    kind, cfg_json, device, *rest = argv
    if device == "cpu":
        mx.set_default_device(mx.cpu)
    cfg = config_from_dict(json.loads(cfg_json))
    out: dict[str, Any] = {"tmpdir": os.environ.get("TMPDIR"), "libs_at_start": len(_libs())}
    if kind == "probe":
        f = mx.compile(lambda x: mx.exp(x * 1.25) + mx.sin(x) * 3.5 + mx.cos(x))
        mx.eval(f(mx.arange(10.0)))
    else:
        mod = importlib.import_module("test_determinism" if kind == "2d" else "test_determinism_3d")
        steps = int(rest[0]) if rest else mod.STEPS
        state = mod._run(cfg, steps)[0]
        if kind == "2d":
            from alife.state import legacy_state_digest

            out["digest"] = legacy_state_digest(state)
        else:
            from alife.state import LedgerIdx, legacy_state_digest_3d, to_numpy

            out["digest"] = legacy_state_digest_3d(state)
            out["buoyancy"] = int(to_numpy(state.ledger_cum)[LedgerIdx.HEAT_BUOYANCY])
    out["libs_at_end"] = len(_libs())
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(HERE))  # test_determinism and test_determinism_3d
    raise SystemExit(main(sys.argv[1:]))
