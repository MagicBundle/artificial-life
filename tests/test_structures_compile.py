"""E4 (DESIGN F21 item 6, I50): every constant the structures layer puts into a compiled function.

MLX names a fused kernel after its op tape and the printed form of its scalar constants back to back, so two
configurations that differ only in a constant can share one kernel when the prints concatenate alike. For each
structures constant, configurations differing only in it are compiled one after the other in ONE process on
the CPU with a private kernel cache, and every compiled result must equal its eager one. Values are chosen so
that pairs print alike ("12" and "1" next to "21", "2" and "12", ...).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

SCRIPT = textwrap.dedent(
    """
    import dataclasses, json, sys
    import mlx.core as mx
    import numpy as np
    mx.set_default_device(mx.cpu)
    from alife import fields, physics, structures, weather
    from alife.config import load_config

    base = load_config(preset="tiny", overrides=["structures.enabled=true", "structures.aperture=0"])
    rs = np.random.default_rng(3)
    n, cells = 512, 32 * 32
    m = mx.array(rs.integers(0, 4097, (32, 32)).astype(np.int32))
    dep = mx.array((rs.integers(0, 40, (32, 32)) * (rs.random((32, 32)) < 0.5)).astype(np.int32))
    force = mx.array(rs.normal(0, 0.1, (n, 2)).astype(np.float32))
    fcell = mx.array(rs.integers(0, cells, n).astype(np.int32))
    step = mx.array(5, mx.int32)
    speed = mx.array(rs.random(n).astype(np.float32))

    def make(cfg):
        def f(m_, d_, t_, force_, fcell_, speed_):
            wind = weather.AgentForce(force_, speed_, speed_ > 0.5, mx.array(False))
            up = fields.update_mat(m_, d_, t_, cfg)
            sn = structures.sense(m_, fcell_, cfg)
            dm = structures.drag_mult(force_, sn, cfg)
            dm = mx.ones((n,)) if dm is None else dm
            sw = structures.apply_shelter(wind, sn, cfg) if sn.cover is not None else wind
            kw = structures.combat_kw(sn, cfg)
            sh = kw.get("shelter", mx.zeros((n,)))
            due = cfg.structures.build_cost * 7 + cfg.structures.build_max
            return up.mat + due, up.wasted, up.decayed, dm, sw.force, sw.speed, sh
        return f

    keep, out = [], []
    for kv in json.loads(sys.argv[1]):
        cfg = base
        cfg = dataclasses.replace(cfg, structures=dataclasses.replace(cfg.structures, **kv))
        if cfg.structures.shelter_gain > 0:
            cfg = dataclasses.replace(cfg, weather=dataclasses.replace(cfg.weather, enabled=True))
        f = make(cfg)
        cf = mx.compile(f)
        keep.append(cf)
        got, want = cf(m, dep, step, force, fcell, speed), f(m, dep, step, force, fcell, speed)
        out.append(all(bool(mx.array_equal(g, w).item()) for g, w in zip(got, want)))
    print(json.dumps(out))
    """
)
BASE = {
    "cap": 4096, "build_max": 16, "build_cost": 2, "decay_q12": 16, "shelter_gain": 0.8, "barrier_gain": 3.0,
    "shield_gain": 0.5, "reach": 0.5, "open_frac": 0.5, "exit_ease": 0.75,
}  # fmt: skip
VALUES = {
    "cap": [256, 1200, 1210, 2100, 4096, 32767],
    "build_max": [1, 2, 12, 21, 16, 1024],
    "build_cost": [0, 1, 2, 12, 21, 4],
    "decay_q12": [1, 2, 12, 21, 16, 4096],
    "shelter_gain": [0.1, 0.12, 0.21, 0.8, 1.0],
    "barrier_gain": [0.1, 0.12, 0.21, 1.0, 2.0, 3.0],
    "shield_gain": [0.1, 0.12, 0.21, 0.5, 1.0],
    "reach": [0.1, 0.12, 0.21, 0.5, 1.0],
    "open_frac": [0.1, 0.12, 0.21, 0.5, 1.0],
    "exit_ease": [0.1, 0.12, 0.21, 0.75, 1.0],
}


def _run(tmpdir: Path, cases: list[dict]) -> list[bool]:
    tmpdir.mkdir(parents=True, exist_ok=True)  # MLX needs TMPDIR to exist
    env = {**os.environ, "TMPDIR": f"{str(tmpdir).rstrip(os.sep)}{os.sep}"}
    done = subprocess.run(
        [sys.executable, "-c", SCRIPT, json.dumps(cases)],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


@pytest.mark.parametrize("key", sorted(VALUES))
def test_e4_configs_differing_only_in_one_structures_constant_do_not_share_a_kernel(
    key: str, tmp_path: Path
) -> None:
    cases = [{**BASE, key: v} for v in VALUES[key]]
    assert _run(tmp_path, cases) == [True] * len(cases)


def test_e4_the_aperture_thresholds_k_and_open_frac_together(tmp_path: Path) -> None:
    """``aperture`` (k = ceil(a * 8)) and ``open_frac`` (thr) are Python ints in the graph: pairs that print
    alike ((2, 1210) next to (21, 210)) are compiled in one process; the engine's gate is exact."""
    script = SCRIPT.replace('"structures.aperture=0"', '"structures.aperture=0.25"')
    tmp_path.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "TMPDIR": f"{str(tmp_path).rstrip(os.sep)}{os.sep}"}
    cases = [
        {**BASE, "aperture": a, "open_frac": o}
        for a, o in ((0.25, 0.295), (0.125, 0.5), (0.125, 0.21), (0.25, 0.5))
    ]
    done = subprocess.run(
        [sys.executable, "-c", script, json.dumps(cases)],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout) == [True] * len(cases)
