"""E4 (DESIGN F21 item 6, I50): every constant farming puts into a compiled function.

MLX names a fused kernel after its op tape and the printed form of its scalar constants back to back, so two
configurations that differ only in a constant can share one kernel when the prints concatenate alike. For each
constant of the farm layer (``farm_rate``, ``farm_cap``, ``farm_cost``, ``farm_decay_q16``,
``farm_bonus_frac``), configurations differing only in it are compiled one after the other in ONE process on
the CPU with a private kernel cache, and every compiled result must equal its eager one (and differ between
values that must differ).
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
    import dataclasses, hashlib, json, sys
    import mlx.core as mx
    import numpy as np
    mx.set_default_device(mx.cpu)
    from alife import niche, ops
    from alife.config import dims, load_config

    base = load_config(preset="tiny", overrides=["niche.farm=true", "brain.weight_dtype=float32"])
    d = dims(base)
    rs = np.random.default_rng(3)
    n = 512
    y = mx.array(rs.normal(0, 0.6, (n, d.lay.D_OUT)).astype(np.float32))
    uid = mx.arange(n, dtype=mx.int32)
    step = mx.array(123, dtype=mx.int32)
    soil = mx.array(rs.integers(0, 4096, d.field_shape).astype(np.int32))
    cells = mx.arange(d.n_cells, dtype=mx.int32).reshape(d.field_shape)

    def make(cfg):
        def f(yy, so):
            u = niche.units_of(yy, uid, step, cfg)
            b = niche.bonus_of(so, cfg)
            dec = so - ops.dithered_decay(so, cfg.niche.farm_decay_q16, cells, step, niche.DECAY_SALT)
            capped = mx.minimum(so + 300, cfg.niche.farm_cap)
            return u, b, dec, capped, u * cfg.niche.farm_cost
        return f

    keep, out, sigs = [], [], []
    for kv in json.loads(sys.argv[1]):
        cfg = dataclasses.replace(base, niche=dataclasses.replace(base.niche, **kv))
        dims.cache_clear()
        f = make(cfg)
        cf = mx.compile(f)
        keep.append(cf)
        got, want = cf(y, soil), f(y, soil)
        out.append(all(bool(mx.array_equal(g, w).item()) for g, w in zip(got, want)))
        sigs.append([hashlib.sha1(np.asarray(g).tobytes()).hexdigest() for g in got])
    print(json.dumps({"equal": out, "sig": sigs}))
    """
)
BASE = {"farm": True}
# value -> the outputs it must move: 0 units, 1 bonus, 2 decayed soil, 3 capped soil, 4 the cost
VALUES = {
    "farm_rate": ([8, 9, 16, 4, 5, 6, 7, 10, 80, 64, 3], (0,)),
    "farm_cap": ([4095, 4096, 1000, 2000, 3000, 4000, 512, 777, 20000], (3,)),
    "farm_cost": ([30, 31, 300, 3, 150, 1, 2, 10, 20, 4000], (4,)),
    "farm_decay_q16": ([16, 17, 160, 1600, 33, 64, 32, 1, 40, 4000], (2,)),
    "farm_bonus_frac": ([0.5, 0.25, 0.75, 0.05, 0.1, 0.9, 0.55, 0.45, 1.0, 0.0], (1,)),
}


def _run(tmpdir: Path, cases: list[dict]) -> dict:
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
def test_e4_configs_differing_only_in_one_niche_constant_do_not_share_a_kernel(
    key: str, tmp_path: Path
) -> None:
    values, moves = VALUES[key]
    cases = [{**BASE, key: v} for v in values]
    res = _run(tmp_path, cases)
    assert res["equal"] == [True] * len(cases)
    for out in moves:
        sigs = [s[out] for s in res["sig"]]
        assert len(set(sigs)) == len(sigs), f"two values of {key} gave one result: a shared kernel"
