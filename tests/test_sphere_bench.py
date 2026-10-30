"""The benchmark on a sphere (DESIGN §46.10; sphere series S2): the staged pipeline mirrors every sphere
branch of ``step.step`` bit for bit in eager mode, the report labels the planet, and ``--alive`` builds the
steady scene the per-alive-agent comparison needs (a fixed number of alive agents, births off).
"""

from __future__ import annotations

import json
from dataclasses import replace

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from alife import bench, cli
from alife.config import Config, dims, load_config
from alife.state import CounterIdx, init_state, state_digest, to_numpy
from alife.step import make_step


def tiny3s() -> Config:
    return load_config(preset="tiny3s", overrides=["brain.weight_dtype=float32"])


def test_the_staged_pipeline_equals_the_sphere_step() -> None:
    """Every sphere branch of ``step.step`` is mirrored by ``bench._stage_fns`` (the sense stage's frame, the
    physics stage's ``forces_s``/``integrate_s``, the reproduce stage's ``hdg``): bitwise equal states and
    StepOuts in eager mode, with births exercised."""
    cfg = replace(tiny3s(), run=replace(tiny3s().run, compile=False))
    fused, staged = make_step(cfg), bench._StagedStep(cfg)
    s0 = init_state(cfg)
    s0 = s0._replace(birth_step=mx.where(s0.alive, -50, s0.birth_step))  # founders old enough to breed
    mx.eval(s0)
    s, holder = s0, [s0]
    births = 0
    for _ in range(6):
        s, out = fused(s)
        out_staged = staged(holder)
        mx.eval(s, out, holder[0], out_staged)
        assert state_digest(holder[0]) == state_digest(s)
        for a, b in zip(tree_flatten(out), tree_flatten(out_staged), strict=True):
            assert a[0] == b[0]
            np.testing.assert_array_equal(to_numpy(a[1]), to_numpy(b[1]), err_msg=a[0])
        births += int(to_numpy(out.counters)[CounterIdx.BIRTHS])
    assert births > 0


def test_the_report_labels_the_planet_and_all_parts_run_on_tiny3s() -> None:
    r = bench.run_bench(tiny3s(), steps=3, warmup=2, stages=True, determinism=True)
    json.dumps(r, allow_nan=False)
    c = r["config"]
    assert (c["geometry"], c["planet_radius"], c["field_nside"], c["hash_nside"]) == ("sphere", 36.0, 8, 8)
    assert "world_size" not in c and "field_grid" not in c and c["dims"] == 3
    assert r["invariants_ok"] and r["diag_ok"] and r["determinism"]["match"]
    assert set(r["stages"]["ms"]) == {name for name, _ in bench.STAGES}
    assert "R=36 Nside=8" in bench.format_report(r)


def test_steady_overrides_make_a_scene_with_a_fixed_alive_count() -> None:
    sets = bench.steady_overrides(512)
    cfg = load_config(preset="tiny3s", overrides=["brain.weight_dtype=float32", *sets])
    d = dims(cfg)
    assert (cfg.population.n_init, d.n % 64, cfg.repro.min_energy) == (512, 0, 999_000)
    assert d.n_cap >= 512 * 1.5
    fn = make_step(replace(cfg, run=replace(cfg.run, compile=False)))
    s = init_state(cfg)
    for _ in range(25):
        s, out = fn(s)
    mx.eval(s)
    c = to_numpy(out.counters)
    assert int(c[CounterIdx.BIRTHS]) == 0 and 400 < int(c[CounterIdx.N_ALIVE]) <= 512  # nobody is born
    with pytest.raises(ValueError, match="at least 256"):
        bench.steady_overrides(100)


def test_the_bench_command_takes_alive(capsys: pytest.CaptureFixture) -> None:
    assert cli.main(["bench", "--preset", "tiny3s", "--alive", "512", "--steps", "3", "--warmup", "2"]) == 0
    r = json.loads(capsys.readouterr().out)
    assert r["config"]["n_init"] == 512 and r["n_alive_start"] <= 512
