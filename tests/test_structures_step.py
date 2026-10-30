"""Built structures in the whole step: the bench mirror, observer purity, the pair with arms (the protection
product), the Director's levers (E24), the metrics row and its 64-bit sums (E10) (DESIGN §38)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from alife import bench, config, director, invariants, metrics
from alife import energy as en
from alife import weather as W
from alife.config import dims
from alife.director import actions as dir_actions
from alife.state import CounterIdx, Env, LedgerIdx, counter_blocks, state_digest, to_numpy
from alife.step import make_step
from structures_helpers import NAMES, builders, cfg_for, np_mat, run
from weather_helpers import STRONG, spun_ws, storm2d_config

# --- the staged mirror of bench.py equals step() --------------------------------------------------


@pytest.mark.parametrize("which", ["2d", "3d", "sphere", "storm2d"])
def test_the_staged_pipeline_equals_the_step_with_structures_on(which: str) -> None:
    if which == "storm2d":
        cfg = storm2d_config("structures.enabled=true", *STRONG, "weather.storm_p=0.001")
        s0 = builders(cfg)._replace(wx=W.pack(spun_ws(cfg)))
    else:
        cfg = cfg_for({"2d": 2, "3d": 3, "sphere": "s"}[which])
        s0 = builders(cfg)
    cfg = replace(cfg, run=replace(cfg.run, compile=False))
    fused, staged = make_step(cfg), bench._StagedStep(cfg)
    s, holder = s0, [s0]
    for i in range(6):
        s, out = fused(s)
        out_staged = staged(holder)
        mx.eval(s, out, holder[0], out_staged)
        assert state_digest(holder[0]) == state_digest(s), i
        for a, b in zip(tree_flatten(out), tree_flatten(out_staged), strict=True):
            assert a[0] == b[0]
            np.testing.assert_array_equal(to_numpy(a[1]), to_numpy(b[1]), err_msg=a[0])
    assert to_numpy(s.mat).any()


# --- observing changes nothing --------------------------------------------------------------------


@pytest.mark.parametrize("nd", [2, 3, "s"])
def test_observing_does_not_change_the_trajectory_with_structures_on(nd: int | str) -> None:
    cfg = cfg_for(nd)
    s0 = builders(cfg)
    f_off, f_on = make_step(cfg), make_step(cfg, observe=True)
    a = b = s0
    for t in range(25):
        a, oa = f_off(a)
        b, ob = f_on(b)
        mx.eval(a, b, oa, ob)
        assert state_digest(a) == state_digest(b), t
        for x, y in zip(tree_flatten(oa), tree_flatten(ob.base()), strict=True):
            np.testing.assert_array_equal(to_numpy(x[1]), to_numpy(y[1]), err_msg=x[0])


# --- the pair with arms: shelter inside the protection product (DESIGN §43.6) ---------------------


@pytest.mark.parametrize("nd", [2, 3, "s"])
def test_structures_and_arms_together_close_the_ledger_and_keep_the_material_identity(nd: int | str) -> None:
    arms = [
        "arms.enabled=true",
        "arms.key_floor=0.2",
        "arms.init_atk=0.3",
        "arms.init_arm=0.4",
        "arms.sigma=0.05",
        "arms.sigma_phase=0.05",
        "combat.protect_floor=0.1",
    ]
    cfg = cfg_for(nd, *arms)
    s0 = builders(cfg)
    s0 = s0._replace(birth_step=mx.where(s0.alive, -50, s0.birth_step))
    shielded = attacks = 0
    for t, (prev, s, out, row) in enumerate(run(cfg, s0, 100 if nd == "s" else 250)):
        led = to_numpy(out.ledger)
        assert led[LedgerIdx.FLOW_STOLEN] == led[LedgerIdx.FLOW_ATTACK_GAIN] + led[LedgerIdx.FLOW_SPILL]
        assert (
            row["MAT_TOTAL"]
            == np_mat(prev).sum() + row["BUILD_UNITS"] - row["MAT_WASTED"] - row["MAT_DECAYED"]
        )
        shielded += row["SHIELDED_ATTACKS"]
        attacks += int(to_numpy(out.counters)[CounterIdx.ATTACKS])
        if t % 25 == 0:
            assert invariants.full_check(s, cfg, prev, out) == [], t
    assert attacks > 0 and shielded > 0  # attacks happened, some on victims standing in material


def test_the_shelter_factor_is_inside_the_protection_product_and_under_the_floor() -> None:
    """The structures fraction enters ``energy.protection`` after the armour factor; with a floor the product
    cannot go below it."""
    import mlx.core as mx

    t0 = mx.arange(3, dtype=mx.int32)
    arm = mx.array([0.5, 0.5, 0.5])
    p, capped = en.protection(t0, 0.0, arms_p=arm, shelter=mx.array([0.5, 0.0, 0.0]))
    assert np.allclose(np.asarray(p), [0.25, 0.5, 0.5]) and capped is None
    p, capped = en.protection(t0, 0.4, arms_p=arm, shelter=mx.array([0.5, 0.0, 0.0]))
    assert np.allclose(np.asarray(p), [0.4, 0.5, 0.5]) and np.asarray(capped).tolist() == [True, False, False]


# --- E24: the Director never reaches the structures keys ------------------------------------------


def _keys() -> set[str]:
    return {k for _, k in config.FEATURE_KEYS["structures"]} - {"enabled"}


def test_e24_no_director_lever_is_a_structures_key() -> None:
    params = {
        p
        for table in (dir_actions.MENU, dir_actions.GAME_SPECS)
        for spec in table.values()
        for p in spec.bounds
    }
    assert params and not (params & _keys())
    assert not (set(Env._fields) & _keys()) and not (set(config.DIRECTOR_POLICY_KEYS) & _keys())
    assert all(sec != "structures" for sec, _ in config.UNHASHED_KEYS)  # every structures key is hashed
    root = Path(director.__file__).parent
    needles = ("cfg.structures", ".structures.", "StructuresConfig", '"structures"', "alife.structures")
    assert [p.name for p in root.rglob("*.py") if any(n in p.read_text() for n in needles)] == []


# --- the metrics row and E10 ----------------------------------------------------------------------


def test_the_metrics_row_carries_the_build_keys_and_sums_in_64_bits() -> None:
    cfg = cfg_for(2)
    blocks = counter_blocks(cfg)
    width = metrics.n_counters(2, False, blocks)
    t = 5000
    c = np.zeros((t, width), np.int32)
    base = metrics.n_counters(2, False)
    c[:, CounterIdx.N_ALIVE] = 300
    # a per-step flow of 2^20 for 5000 steps sums to 5.2e9 > 2^31: int32 would wrap
    c[:, base + NAMES.index("BUILD_UNITS")] = 1 << 20
    c[:, base + NAMES.index("MAT_WASTED")] = 1 << 20
    c[:, base + NAMES.index("MAT_TOTAL")] = 2_000_000_000
    c[:, base + NAMES.index("MAT_CELLS")] = 500
    c[:, base + NAMES.index("SLOWED")] = 30
    c[:, base + NAMES.index("TRAPPED")] = 1
    lg = np.zeros((t, 16), np.int64)
    row = metrics.interval_row(10_000, c, lg, {}, {"wall_s": 1.0}, nd=2, blocks=blocks)
    assert row["build_units"] == t * (1 << 20) > 2**31
    assert row["build_wasted"] == t * (1 << 20)
    assert row["build_mat_total"] == 2_000_000_000 and row["build_mat_cells"] == 500
    assert row["build_slowed_frac"] == pytest.approx(30 / 300) and row["build_trapped"] == t
    assert (
        row["build_aperture_violations"] == 0 and row["build_mat_closed"] == 0 and row["build_decayed"] == 0
    )
    # a row of a structures-off config has none of the keys
    plain = metrics.interval_row(
        10, np.zeros((t, metrics.n_counters(2)), np.int32), lg, {}, {"wall_s": 1.0}, nd=2
    )
    assert not any(k.startswith("build_") for k in plain)


def test_a_run_reports_the_build_keys_in_its_rows(tmp_path: Path) -> None:
    from alife import cli
    from alife.runio import RunDir

    out = tmp_path / "run"
    code = cli.main(
        [
            *["run", "--preset", "tiny", "--overlay", "build", "--set", "brain.weight_dtype=float32"],
            *["--steps", "40", "--out", str(out), "--quiet", "--set", "run.diversity_sample=32"],
        ]
    )
    assert code == 0
    rows = RunDir.open(out).metrics_rows()
    assert rows and all(k in rows[-1] for k in ("build_units", "build_mat_total", "build_trapped"))
    assert all(r["ledger_error"] == 0 and r["diag"] & 0xFFFF == 0 for r in rows)
    assert all(r["build_aperture_violations"] == 0 and r["build_trapped"] == 0 for r in rows)
    assert sum(r["build_units"] for r in rows) > 0  # random founders: about half build a little from step 0
    assert dims(RunDir.open(out).config()).lay.OUT_BUILD == 35


# --- bench --structures --------------------------------------------------------------------------------


def test_bench_structures_needs_the_feature_and_reports_the_marginal_cost() -> None:
    from alife.config import ConfigError, load_config

    off = load_config(preset="tiny", overrides=["brain.weight_dtype=float32"])
    with pytest.raises(ConfigError, match=r"structures\.enabled"):
        bench.run_bench(off, steps=2, warmup=1, structures=True)
    r = bench.run_bench(cfg_for(2), steps=4, warmup=1, structures=True, weather_pairs=1)
    st = r["structures"]
    assert (
        st["structures_config"]["enabled"]
        and st["block_pairs"] == 1
        and st["on_ms_median"] > 0
        and "marginal_ms_median" in st
    )
    assert "structures (1 block pairs" in bench.format_report(r)
