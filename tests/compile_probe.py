"""Probe the real compiled step for MLX's compile-cache constant collision (DESIGN F21 item 6).

Each configuration compiles ``make_step`` in its own process on the CPU device, with an empty private MLX
kernel cache, and checks ``steps`` compiled steps against eager ones from the same states. A collision shows
in one of two ways:
- across configurations: one kernel file name with two different sources (constant literals) in two caches.
  Those configurations would share a kernel in one process, or in two processes sharing a cache directory;
- inside one configuration: an integer or bool leaf of the compiled step that differs from the eager one.
Float leaves are reported as the largest absolute difference: fused and eager CPU code round some float ops
differently, so they are not compared exactly. Re-run it on every MLX bump and when a preset or a test
configuration is added (about two minutes per configuration on an M5 Pro, almost all of it ``g++``):

    uv run python tests/compile_probe.py OUT_DIR [--jobs 4] [--steps 3] [--only NAME ...]

It writes ``OUT_DIR/report.json`` and exits 1 on a collision or an integer mismatch.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

MICRO = [
    "population.n_max=64",
    "population.max_births_per_step=8",
    "population.n_init=8",
    "world.size=48",
    "brain.weight_dtype=float32",
]
F32 = ["brain.weight_dtype=float32"]
# name -> (preset, overrides, observe): every preset, the conftest configurations and the static variants. A
# preset "64k+fast" adds the ``fast`` overlay, "64k3d+storm" the storm world's.
CONFIGS: dict[str, tuple[str, list[str], bool]] = {
    "tiny_f32": ("tiny", F32, False),  # conftest tiny_cfg
    "tiny": ("tiny", [], False),  # tiny_cfg_f16, the preset
    "tiny_bf16": ("tiny", ["brain.weight_dtype=bfloat16"], False),
    "micro": ("tiny", MICRO, False),
    "tiny3d_f32": ("tiny3d", F32, False),
    "tiny3d": ("tiny3d", [], False),
    "micro3d": ("tiny3d", MICRO, False),
    "tiny_director": ("tiny", [*F32, "director.enabled=true"], False),
    "tiny_observe": ("tiny", F32, True),
    "tiny3d_observe": ("tiny3d", F32, True),
    "tiny_checks_off": ("tiny", [*F32, "run.runtime_checks=off"], False),
    "dev": ("dev", [], False),
    "64k": ("64k", [], False),
    "64k_director": ("64k", ["director.enabled=true"], False),
    "64k3d": ("64k3d", [], False),
    "256k": ("256k", [], False),
    "tiny_fast": ("tiny+fast", F32, False),
    "64k_fast": ("64k+fast", [], False),
    "64k3d_storm": ("64k3d+storm", [], False),  # the weather world overlay (added in the port, DESIGN I58)
}


def probe_one(name: str, steps: int, out: Path) -> None:
    """The child: compiled against eager ``make_step`` for ``CONFIGS[name]``; writes a JSON summary."""
    import mlx.core as mx

    mx.set_default_device(mx.cpu)
    import numpy as np
    from mlx.utils import tree_flatten

    from alife.config import load_config
    from alife.state import init_state
    from alife.step import make_step

    preset, overrides, observe = CONFIGS[name]
    preset, *overlays = preset.split("+")
    cfg = load_config(preset=preset, overlays=overlays, overrides=overrides)
    s = init_state(cfg)
    fc = make_step(cfg, compile=True, observe=observe)
    fe = make_step(cfg, compile=False, observe=observe)
    ints: list[dict[str, Any]] = []
    floats: dict[str, float] = {}
    for k in range(steps):
        c, e = fc(s), fe(s)
        mx.eval(c, e)
        for (leaf, a), (_, b) in zip(tree_flatten(c), tree_flatten(e), strict=True):
            if a is None or b is None:  # a field the observer channel leaves out: both None
                assert a is None and b is None, leaf
                continue
            x = np.array(a.astype(mx.float32) if a.dtype == mx.bfloat16 else a)
            y = np.array(b.astype(mx.float32) if b.dtype == mx.bfloat16 else b)
            if x.dtype.kind == "f":
                d = float(np.nanmax(np.abs(x.astype(np.float64) - y), initial=0.0))
                floats[leaf] = max(floats.get(leaf, 0.0), d)
            elif not np.array_equal(x, y):
                ints.append({"step": k, "leaf": leaf, "n_diff": int((x != y).sum())})
        s = c[0]
    nonzero = {k: v for k, v in floats.items() if v}
    out.write_text(json.dumps({"int_mismatches": ints, "float_max_abs": nonzero}))


def literals(source: str) -> list[str]:
    """The constant literals of a kernel source, in input order."""
    import kernel_ambiguity

    return [text for _, text in kernel_ambiguity.declarations(source) or []]


def scan(caches: dict[str, Path]) -> list[dict[str, Any]]:
    """Kernel files that hold different constants in different caches: one entry per colliding name."""
    seen: dict[str, dict[tuple[str, ...], list[str]]] = defaultdict(lambda: defaultdict(list))
    for label, d in caches.items():
        for cpp in d.glob("*.cpp"):
            seen[cpp.name][tuple(literals(cpp.read_text(errors="replace")))].append(label)
    return [
        {"file": f, "variants": [{"constants": list(c), "configs": v} for c, v in by.items()]}
        for f, by in sorted(seen.items())
        if len(by) > 1
    ]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path, help="output directory (one private kernel cache per configuration)")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--only", nargs="+", choices=sorted(CONFIGS), default=sorted(CONFIGS))
    ap.add_argument("--one", help=argparse.SUPPRESS)  # child mode: probe this configuration
    a = ap.parse_args(argv)
    if a.one:
        probe_one(a.one, a.steps, a.out)
        return 0
    a.out.mkdir(parents=True, exist_ok=True)

    def run(name: str) -> tuple[str, dict[str, Any]]:
        tmp = a.out / name / "tmp"
        tmp.mkdir(parents=True, exist_ok=False)  # an empty cache: never reuse one
        res = a.out / name / "result.json"
        env = {**os.environ, "TMPDIR": f"{tmp}{os.sep}"}
        argv_ = [sys.executable, __file__, str(res), "--one", name, "--steps", str(a.steps)]
        done = subprocess.run(argv_, env=env, capture_output=True, text=True)
        if done.returncode != 0:
            return name, {"error": done.stderr[-2000:]}
        return name, json.loads(res.read_text())

    with ThreadPoolExecutor(a.jobs) as ex:
        results = dict(ex.map(run, a.only))
    import worker_cache

    collisions = scan({n: worker_cache.kernel_dir(a.out / n / "tmp") for n in a.only})
    report = {"configs": results, "collisions": collisions}
    (a.out / "report.json").write_text(json.dumps(report, indent=1))
    bad = False
    for n, r in results.items():
        if "error" in r:
            print(f"{n}: ERROR {r['error'].splitlines()[-1:]}")
            bad = True
            continue
        worst = max(r["float_max_abs"].values(), default=0.0)
        print(f"{n}: integer mismatches {len(r['int_mismatches'])}, largest float difference {worst:.3g}")
        bad |= bool(r["int_mismatches"])
    for c in collisions:
        print("COLLISION", c["file"][:100], [(v["constants"], v["configs"]) for v in c["variants"]])
    print(f"{len(collisions)} colliding kernel names across {len(a.only)} configurations")
    return 1 if bad or collisions else 0


if __name__ == "__main__":
    sys.path.insert(
        0, str(Path(__file__).resolve().parent)
    )  # kernel_ambiguity and worker_cache, when run as a script
    raise SystemExit(main())
