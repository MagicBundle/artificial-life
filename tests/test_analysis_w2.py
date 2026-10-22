"""``alife.analysis.w2`` (DESIGN §32, spec §8): the pre-registered statistics on synthetic runs.

The definitions, windows and thresholds of the W2 results pipeline are pinned here before any W2b run is read:
the block layout and the t-interval, the paired block differences, the claim rule with the s1 noise floor, and
each of P1a-P6 against a number computed in the test. The synthetic runs shrink the windows (``TRANSIENT``,
``BLOCK``, ``EARLY``, ``ECO_WINDOW`` are module constants read at call time) and keep the structure: 50-step
metrics rows, eco dumps one window apart, wx dumps, birth and death logs.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from alife.analysis import w2
from alife.config import apply_overrides
from alife.runio import RunDir
from alife.state import DeathCause, Diag
from weather_helpers import storm3d_config

# --- pure functions -------------------------------------------------------------------------------


def test_t_interval_is_the_student_interval_of_the_values() -> None:
    iv = w2.t_interval([1, 2, 3, 4, 5, 6, 7, 8])
    sd = float(np.std([1, 2, 3, 4, 5, 6, 7, 8], ddof=1))
    half = 2.365 * sd / math.sqrt(8)  # 7 df
    assert iv["n"] == 8 and iv["mean"] == 4.5 and iv["sd"] == pytest.approx(sd)
    assert iv["lo"] == pytest.approx(4.5 - half) and iv["hi"] == pytest.approx(4.5 + half)
    assert (
        w2.t_crit(7) == 2.365 and w2.t_crit(1) == 12.706 and w2.t_crit(30) == 2.042 and w2.t_crit(31) == 1.96
    )


def test_t_interval_drops_non_finite_values_and_needs_two() -> None:
    iv = w2.t_interval([1.0, math.nan, 3.0])
    assert iv["n"] == 2 and iv["mean"] == 2.0 and iv["lo"] is not None
    one = w2.t_interval([5.0, math.nan])
    assert one["n"] == 1 and one["mean"] == 5.0 and one["lo"] is None and one["hi"] is None
    none = w2.t_interval([])
    assert none["n"] == 0 and none["mean"] is None


def test_the_blocks_are_eight_of_ten_thousand_after_the_transient(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.undo()  # the real windows, not the shortened ones of the synthetic runs
    blk = w2.Blocking(450_000, w2.TRANSIENT, w2.BLOCK, w2.N_BLOCKS)
    assert (w2.TRANSIENT, w2.BLOCK, w2.N_BLOCKS) == (20_000, 10_000, 8) and blk.lo == 470_000
    steps = np.array([450_100, 470_000, 470_001, 480_000, 480_001, 550_000, 550_001, 600_000])
    assert blk.index(steps).tolist() == [-1, -1, 0, 0, 1, 7, -1, -1]
    sens = blk.sensitivity()
    assert (sens.n, sens.width, sens.lo) == (16, 5_000, 470_000)
    assert sens.index(np.array([475_000, 475_001, 550_000])).tolist() == [0, 1, 15]


def test_block_mean_and_ratio() -> None:
    blk = w2.Blocking(0, 100, 100, 3)
    steps = np.array([50, 101, 150, 200, 201, 300, 301])
    v = np.array([9.0, 1.0, 3.0, 5.0, np.nan, 7.0, 99.0])
    bm = w2.block_mean(blk, steps, v)
    assert bm.tolist() == [
        3.0,
        7.0,
        99.0,
    ]  # (101, 150, 200), (201 is NaN, 300), (301); 50 is in the transient
    assert np.isnan(w2.block_mean(blk, np.array([50]), np.array([1.0]))).all()  # nothing in the blocks
    r = w2.block_ratio(blk, steps, np.array([0, 2, 4, 6, 8, 10, 99.0]), np.array([0, 1, 1, 2, 2, 2, 1.0]))
    assert r.tolist() == [3.0, 4.5, 99.0]
    z = w2.block_ratio(blk, np.array([101]), np.array([1.0]), np.array([0.0]))
    assert np.isnan(z[0])


def test_paired_differences_keep_nan() -> None:
    d = w2.paired(np.array([3.0, np.nan, 1.0]), np.array([1.0, 2.0, 1.5]))
    assert d[0] == 2.0 and np.isnan(d[1]) and d[2] == -0.5


def _iv(mean: float, half: float) -> dict[str, float | int]:
    return {"mean": mean, "lo": mean - half, "hi": mean + half, "n": 8}


@pytest.mark.parametrize(
    ("ivs", "floor", "want"),
    [
        ([_iv(1, 0.5), _iv(2, 0.5)], None, "supported"),
        ([_iv(-1, 0.5), _iv(-2, 0.5)], None, "supported"),
        ([_iv(1, 0.5), _iv(-2, 0.5)], None, "mixed"),  # both exclude 0 with opposite signs
        ([_iv(1, 0.5), _iv(0.2, 0.5)], None, "mixed"),
        ([_iv(0.1, 0.5), _iv(0.2, 0.5)], None, "not supported"),
        ([_iv(1, 0.5), None], None, "mixed"),  # a seed without the pair counts as not excluding
        ([None, None], None, "not supported"),
        ([_iv(1, 0.5), _iv(2, 0.5)], 1.5, "mixed"),  # s0 does not exceed the floor
        ([_iv(2, 0.5), _iv(2, 0.5)], 1.5, "supported"),
        ([_iv(0.2, 0.5), _iv(0.3, 0.5)], 0.1, "not supported"),
        ([{"mean": None, "lo": None, "hi": None, "n": 1}, _iv(1, 0.1)], None, "mixed"),
    ],
)
def test_the_claim_rule(ivs: list, floor: float | None, want: str) -> None:
    assert w2.classify(ivs, floor) == want


def test_a_single_seed_cannot_support_a_claim() -> None:
    assert w2.classify([_iv(1, 0.1)]) == "mixed"


def test_selection_gradient_is_the_covariance_of_relative_fitness_with_the_standardised_trait() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=1000)
    z = (x - x.mean()) / x.std()
    for b in (0.4, -0.2, 0.0):
        assert w2.selection_gradient(x * 3 + 7, 2.0 + b * 2.0 * z) == pytest.approx(b, abs=1e-12)  # w̄ = 2
    assert w2.selection_gradient(np.ones(10), np.arange(10)) is None  # constant trait
    assert w2.selection_gradient(np.arange(10.0), np.zeros(10)) is None  # nobody reproduces
    assert w2.selection_gradient(np.array([1.0]), np.array([1.0])) is None


def test_births_per_parent_and_match_uid() -> None:
    parents = np.array([5, 5, 9, 2, 5])
    assert w2.births_per_parent(parents, np.array([2, 5, 7, 9, 11])).tolist() == [1, 3, 0, 1, 0]
    assert w2.births_per_parent(np.zeros(0, np.int64), np.array([1, 2])).tolist() == [0, 0]
    j, hit = w2.match_uid(np.array([9, 3, 5]), np.array([5, 9, 1]))
    assert hit.tolist() == [True, False, True] and np.array([5, 9, 1])[j][hit].tolist() == [9, 5]
    assert w2.match_uid(np.array([1]), np.zeros(0, np.int64))[1].tolist() == [False]


def test_slope_and_infer_arm() -> None:
    x = np.arange(10.0)
    assert w2.slope(3.0 * x + 1.0, x) == pytest.approx(3.0)
    assert w2.slope(x, np.ones(10)) is None
    base = storm3d_config("weather.seed_gain=0.03")
    cases = {
        "A": [],
        "N": ["weather.seed_gain=0.0"],
        "WB": ["weather.wind_scale=0.0"],
        "S": ["weather.wind_scale=0.0", "weather.rain_zonal=true"],
        "A1": ["weather.alpha1=0.0"],
        "E": ["weather.wind_eddy_force=true"],
    }
    for arm, ov in cases.items():
        assert w2.infer_arm(apply_overrides(base, ov)) == arm


# --- synthetic runs -------------------------------------------------------------------------------

ARM_OVERRIDES = {
    "A": [],
    "N": ["weather.seed_gain=0.0"],
    "WB": ["weather.wind_scale=0.0", "weather.storm_p=0.0"],
    "S": ["weather.wind_scale=0.0", "weather.storm_p=0.0", "weather.rain_zonal=true"],
    "A1": ["weather.alpha1=0.0"],
    "E": ["weather.wind_eddy_force=true"],
}
UB = {"A": 0.16, "N": 0.16, "E": 0.14, "A1": 0.13, "WB": 0.11, "S": 0.10}
HOLD = {"A": 0.30, "N": 0.30, "E": 0.10, "A1": 0.20, "WB": 0.02, "S": 0.02}
STORM = {"A": 4, "N": 4, "E": 2, "A1": 4, "WB": 0, "S": 0}  # storm deaths per 50-step row
SELECT = {"A": 0.8, "N": 0.8, "E": 0.5, "A1": 0.3, "WB": 0.0, "S": 0.0}  # survival slope on the cap
ROW = 50
LAST = 9_000
START = 0


@pytest.fixture(autouse=True)
def short_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(w2, "TRANSIENT", 1_000)
    monkeypatch.setattr(w2, "BLOCK", 1_000)
    monkeypatch.setattr(w2, "N_BLOCKS", 8)
    monkeypatch.setattr(w2, "EARLY", 300)
    monkeypatch.setattr(w2, "ECO_WINDOW", 100)


def _rows(arm: str, seed: int, rng: np.random.Generator, **special: float) -> list[dict]:
    rows = []
    for step in range(ROW, LAST + 1, ROW):
        wave = 1.0 + 0.03 * math.sin(step / 700.0)  # the weather clock, shared by the arms
        gain = 0.0 if arm == "N" else (0.15 if step <= 300 else 0.30)
        ub = UB[arm] + special.get("ub", 0.0) + rng.normal(0.0, 0.004)
        deaths = {"deaths_starved": 20, "deaths_killed": 15, "deaths_aged": 1, "deaths_catastrophe": 0}
        rows.append(
            {
                "step": step, "steps": ROW, "n_alive": 1000.0 + 10 * seed,
                "speed_mean": 0.02 + 0.001 * (arm == "A"), "births": 40, **deaths,
                "deaths_storm": STORM[arm], "resource_mean": 2000.0, "height_mean": 30.0,
                "max_occupancy": 20, "cell_overflow": 0, "ledger_error": 0, "v_clips": 0, "diag": 0,
                "weather": {
                    "rain_total": 4.0 * wave * (1.0 + gain), "overlap_deep": 1, "hyper_max": 0.1,
                    "occ_hz0": 12, "occ_hz1_11": 10, "occ_hz12": 13,
                },
                "coupling": {
                    "ub_agents_all": ub, "hold": HOLD[arm] + special.get("hold", 0.0) + rng.normal(0.0, 0.01),
                },
            }
        )  # fmt: skip
    return rows


def _eco_family(
    arm: str, rng: np.random.Generator
) -> tuple[list[tuple[int, dict]], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Ecology dumps every 100 steps, the birth log and the storm death log of one synthetic run."""
    n = 300
    uid = np.arange(n, dtype=np.int64)
    cap = rng.uniform(0.0, 1.0, n)
    z = rng.uniform(0.0, 80.0, n)
    next_uid = n
    dumps = []
    b_step, b_uid, b_par = [], [], []
    d_step, d_uid, d_clade = [], [], []
    tracer_z = rng.uniform(0.0, 80.0, 50)
    for k in range(0, LAST // 100 + 1):
        t = 100 * k
        w_air = rng.normal(0.0, 0.01, uid.size)
        dumps.append(
            (
                t,
                {
                    "uid": uid.copy(), "clade": (uid % 3).astype(np.int32), "cap": cap.astype(np.float16),
                    "z": z.astype(np.float16), "vz_own": rng.normal(0.0, 0.01, uid.size).astype(np.float16),
                    "w_air": w_air.astype(np.float16),
                    "emit": rng.integers(0, 30, uid.size).astype(np.int16),
                    "tracer_z": tracer_z.astype(np.float16),
                    "tracer_age": np.full(50, 100 * (k % 2), np.int32),
                },
            )
        )  # fmt: skip
        # the next window: survival rises with the cap by SELECT[arm]; parents of the births also
        p_surv = np.clip(0.6 + SELECT[arm] * (cap - 0.5), 0.02, 0.98)
        keep = rng.random(uid.size) < p_surv
        w_par = 0.2 + SELECT[arm] * cap
        n_births = 60
        parents = rng.choice(uid, n_births, p=w_par / w_par.sum())
        child = np.arange(next_uid, next_uid + n_births, dtype=np.int64)
        b_step.extend(rng.integers(t + 1, t + 101, n_births).tolist())
        b_uid.extend(child.tolist())
        b_par.extend(parents.tolist())
        dead = uid[~keep]
        d_step.extend([t + 50] * dead.size)
        d_uid.extend(dead.tolist())
        d_clade.extend((dead % 3).tolist())
        uid, cap, z = uid[keep], cap[keep], np.clip(z[keep] + rng.normal(0.0, 1.0, keep.sum()), 0, 80)
        # newcomers: the children, with a fresh cap (the distribution stays stationary under selection)
        uid = np.concatenate([uid, child])
        cap = np.concatenate([cap, rng.uniform(0.0, 1.0, n_births)])
        z = np.concatenate([z, rng.uniform(0.0, 80.0, n_births)])
        next_uid += n_births
        tracer_z = np.clip(tracer_z + rng.normal(0.0, 0.5, 50), 0, 80)
    births = {
        "step": np.array(b_step, np.int32), "uid": np.array(b_uid, np.int64),
        "parent_uid": np.array(b_par, np.int64),
        "clade": np.zeros(len(b_step), np.int32), "generation": np.zeros(len(b_step), np.int32),
        "slot": np.zeros(len(b_step), np.int32),
    }  # fmt: skip
    deaths = {
        "step": np.array(d_step, np.int32), "uid": np.array(d_uid, np.int64),
        "age": np.zeros(len(d_step), np.int32), "generation": np.zeros(len(d_step), np.int32),
        "cause": np.full(len(d_step), DeathCause.STORM.value, np.uint8),
        "clade": np.array(d_clade, np.int32),
    }  # fmt: skip
    return dumps, births, deaths


def _make(root: Path, arm: str, seed: int, *, eco: bool = False, wx: bool = False, **special: float) -> Path:
    cfg = apply_overrides(storm3d_config("weather.seed_gain=0.03"), ARM_OVERRIDES[arm])
    rd = RunDir.create(root / f"{arm}_s{seed}", cfg, parent=f"runs/64k3d_s{seed}")
    rng = np.random.default_rng(1000 * seed + sum(map(ord, arm)))
    rd.append_metrics(_rows(arm, seed, rng, **special))
    if eco:
        dumps, births, deaths = _eco_family(arm, rng)
        rd.observers_dir.mkdir(exist_ok=True)
        for t, cols in dumps:
            np.savez(rd.dump_path("eco", t), **cols)
        # the event chunk [0, LAST + 1) holds the whole log
        deaths = {k: v[deaths["step"] <= LAST] for k, v in deaths.items()}
        births = {k: v[births["step"] <= LAST] for k, v in births.items()}
        rd.write_events(0, LAST + 1, births, deaths)
    if wx:
        rd.observers_dir.mkdir(exist_ok=True)
        rng_w = np.random.default_rng(7 + seed)
        for t in range(0, LAST + 1, 500):
            planes = rng_w.uniform(0.0, 1.0, (10, 8, 8)).astype(np.float32)
            planes[7] = rng_w.uniform(0.0, 1.0, (8, 8))
            planes[9] = rng_w.uniform(0.0, 0.05, (8, 8)) if arm == "A" else 0.0
            if arm == "N":  # the seeded run's rain is the unseeded run's plus 3 x the seeding strength
                a = np.load(root / f"A_s{seed}" / "observers" / f"wx_{t:010d}.npy")
                planes[7] = a[7] - 3.0 * a[9]
            np.save(rd.dump_path("wx", t), planes)
    return rd.path


@pytest.fixture(scope="module")
def family(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The six arms on seeds 0 and 1, A and N on seed 2; eco dumps for every run, wx dumps for A and N."""
    root = tmp_path_factory.mktemp("w2")
    mp = pytest.MonkeyPatch()
    mp.setattr(w2, "TRANSIENT", 1_000)
    mp.setattr(w2, "ECO_WINDOW", 100)
    try:
        for seed in (0, 1):
            for arm in ("A", "N", "WB", "S", "A1", "E"):
                _make(
                    root,
                    arm,
                    seed,
                    eco=True,
                    wx=arm in ("A", "N"),
                    **({"ub": 0.015} if (arm, seed) == ("N", 1) else {}),
                )
        for arm in ("A", "N"):
            _make(root, arm, 2, eco=False, wx=False)
    finally:
        mp.undo()
    return root


@pytest.fixture(scope="module")
def runs(family: Path) -> list[w2.Run]:
    return [w2.load_run(p) for p in sorted(family.iterdir())]


@pytest.fixture(scope="module")
def report(runs: list[w2.Run]) -> dict:
    mp = pytest.MonkeyPatch()
    for k, v in (
        ("TRANSIENT", 1_000),
        ("BLOCK", 1_000),
        ("N_BLOCKS", 8),
        ("EARLY", 300),
        ("ECO_WINDOW", 100),
    ):
        mp.setattr(w2, k, v)
    try:
        return w2._clean(w2.analyse(runs))
    finally:
        mp.undo()


def test_load_run_reads_the_arm_the_seed_and_the_fork_step(runs: list[w2.Run]) -> None:
    got = {(r.arm, r.seed) for r in runs}
    assert got == {(a, s) for a in ("A", "N", "WB", "S", "A1", "E") for s in (0, 1)} | {("A", 2), ("N", 2)}
    assert {r.start for r in runs} == {START} and all(len(r.rows) == LAST // ROW for r in runs)


def test_load_run_takes_explicit_labels_and_a_start(family: Path) -> None:
    r = w2.load_run(f"J@3={family / 'A_s0'}", start=123)
    assert (r.arm, r.seed, r.start) == ("J", 3, 123)
    with pytest.raises(FileNotFoundError):
        w2.load_run(family / "nope")


# --- the predictions ------------------------------------------------------------------------------


def test_p1a_is_the_log_rain_ratio_early_and_late(report: dict) -> None:
    for s in ("0", "1", "2"):
        d = report["P1a"][s]
        assert d["early_ratio"] == pytest.approx(1.15) and d["log_early"] == pytest.approx(math.log(1.15))
        assert d["late_ratio"] == pytest.approx(1.30) and d["log_late"] == pytest.approx(math.log(1.30))
        assert d["late_log_ratio_blocks"]["mean"] == pytest.approx(math.log(1.30))
        assert d["late_log_ratio_blocks"]["sd"] == pytest.approx(0.0, abs=1e-12)
        assert d["early_in_calibration_band"] is True and d["same_sign"] is True


def test_p1b_is_the_per_dump_slope_of_the_rain_difference_on_the_seeding_strength(report: dict) -> None:
    for s in ("0", "1"):
        d = report["P1b"][s]
        assert d["dumps"] == 16  # the dumps at 1,500 ... 9,000 (every 500 steps) are post-transient
        assert d["slope"]["mean"] == pytest.approx(3.0, abs=1e-4) and d["slope"]["sd"] == pytest.approx(
            0.0, abs=1e-4
        )
        assert 0.9 < (d["pair_correlation"]["mean"] or 0.0) <= 1.0  # P_A - P_N = 3 q is small against P_A


def test_p1c_reports_the_emission_above_the_free_level(runs: list[w2.Run], report: dict) -> None:
    r = next(x for x in runs if (x.arm, x.seed) == ("A", 0))
    monkey = pytest.MonkeyPatch()
    monkey.setattr(w2, "TRANSIENT", 1_000)
    try:
        blk = r.blocking()
        tot = paid = n = 0.0
        for st, cols in r.eco():
            if blk.index(np.array([st]))[0] < 0:
                continue
            e = cols["emit"].astype(np.float64)
            tot, paid, n = tot + e.sum(), paid + np.maximum(e - 12, 0).sum(), n + e.size
    finally:
        monkey.undo()
    d = report["P1c"]["A"]["0"]
    assert d["free_emit"] == 12 and d["paid_share_all"] == pytest.approx(paid / tot)
    assert sum(d["agents"]) == n and len(d["paid_share"]) == 4 and len(d["clades"]) == 3
    assert 0.0 < d["paid_share_all"] < 0.5


def test_p2_contrasts_and_thresholds(report: dict) -> None:
    d = report["P2"]
    assert d["ub"]["A"]["0"]["mean"] == pytest.approx(0.16, abs=0.003)
    wind = d["A_minus_WB"]
    assert wind["contrast"] == "A - WB" and wind["seeds"]["0"]["mean"] == pytest.approx(0.05, abs=0.004)
    assert wind["claim"] == "supported" and wind["seeds"]["0"]["sens"]["n"] == 16
    assert (
        d["ub_A_at_least_0.12"] == {"0": True, "1": True, "2": True} or d["ub_A_at_least_0.12"]["0"] is True
    )
    assert d["wind_contrast_at_least_0.02"] == {"0": True, "1": True}
    # WB - S is 0.01 and the s1 A - N contrast is 0.015 (that N run carries +0.015): the noise floor binds
    pat = d["WB_minus_S"]
    assert pat["noise_floor"] == pytest.approx(0.015, abs=0.004)
    assert pat["claim_unfloored"] == "supported" and pat["claim"] == "not supported"


def test_p3_hold_and_the_persistence_of_agents_and_tracers(report: dict) -> None:
    d = report["P3"]
    assert d["hold"]["A"]["0"]["mean"] == pytest.approx(0.30, abs=0.01)
    c = d["A_minus_WB"]
    assert c["seeds"]["0"]["mean"] == pytest.approx(0.28, abs=0.01) and c["claim"] == "supported"
    assert d["hold_A_positive"]["0"] is True and d["wind_contrast_at_least_0.05"] == {"0": True, "1": True}
    assert d["abs_dz_250_agents_in_drafts"]["A"]["0"]["mean"] > 0
    # tracers are matched only where they were not reseeded in between (age 0 -> 100 in the fixture)
    assert d["abs_dz_250_tracers"]["A"]["0"]["mean"] == pytest.approx(0.5 * math.sqrt(2 / math.pi), abs=0.08)


def test_the_persistence_series_are_the_mean_absolute_height_change(runs: list[w2.Run]) -> None:
    r = next(x for x in runs if (x.arm, x.seed) == ("A", 0))
    monkey = pytest.MonkeyPatch()
    monkey.setattr(w2, "ECO_WINDOW", 100)
    try:
        dumps = list(r.eco())
        steps, vals = w2.persistence_series(r, "agents")
        a, b = dumps[3][1], dumps[4][1]
        j, hit = w2.match_uid(a["uid"], b["uid"])
        m = hit & (np.abs(a["w_air"].astype(np.float64)) >= 0.002)
        want = np.abs(b["z"][j].astype(np.float64) - a["z"].astype(np.float64))[m].mean()
        assert steps[3] == dumps[4][0] and vals[3] == pytest.approx(want)
        tsteps, tvals = w2.persistence_series(r, "tracers")
        assert (np.diff(tsteps) == 200).all()  # every other pair: the tracers are reseeded between the rest
        ta, tb = dumps[0][1], dumps[1][1]
        assert tvals[0] == pytest.approx(
            np.abs(tb["tracer_z"].astype(float) - ta["tracer_z"].astype(float)).mean()
        )
    finally:
        monkey.undo()


def test_p4_gradients_follow_the_selection_that_the_runs_were_built_with(
    runs: list[w2.Run], report: dict
) -> None:
    r = next(x for x in runs if (x.arm, x.seed) == ("A", 0))
    monkey = pytest.MonkeyPatch()
    monkey.setattr(w2, "ECO_WINDOW", 100)
    try:
        dumps = list(r.eco())
        t, a, b = dumps[10][0], dumps[10][1], dumps[11][1]
        surv = np.isin(a["uid"], b["uid"]).astype(float)
        steps, vals = w2.gradient_series(r, "survival")
        assert steps[10] == t + 100 and vals[10] == pytest.approx(
            w2.selection_gradient(a["cap"].astype(float), surv)
        )
        ev = r.rd.read_events("births")
        w = w2.births_per_parent(ev["parent_uid"][(ev["step"] > t) & (ev["step"] <= t + 100)], a["uid"])
        _, bvals = w2.gradient_series(r, "births")
        assert bvals[10] == pytest.approx(w2.selection_gradient(a["cap"].astype(float), w.astype(float)))
    finally:
        monkey.undo()
    d = report["P4"]
    for what in ("beta_survival", "beta_births"):
        c = d[what]["contrasts"]
        assert c["A_minus_WB"]["seeds"]["0"]["mean"] > 0.1 and c["A_minus_WB"]["claim"] == "supported", what
        assert abs(c["WB_minus_S"]["seeds"]["0"]["mean"]) < 0.1
    c = d["beta_survival"]["contrasts"]  # the survival slopes are 0.8, 0.3 and 0.5 for A, A1 and E
    assert c["A_minus_A1"]["seeds"]["0"]["mean"] > 0.05 and c["A_minus_E"]["seeds"]["0"]["mean"] > 0.0
    assert d["beta_survival"]["per_arm"]["WB"]["0"]["mean"] == pytest.approx(0.0, abs=0.05)
    assert set(d["cap_trend"]) == {f"{a}.{s}" for a in ("A", "N", "WB", "S", "A1", "E") for s in (0, 1)}


def test_the_cap_trend_and_alpha(runs: list[w2.Run]) -> None:
    r = next(x for x in runs if (x.arm, x.seed) == ("WB", 0))
    monkey = pytest.MonkeyPatch()
    monkey.setattr(w2, "ECO_WINDOW", 100)
    monkey.setattr(w2, "TRANSIENT", 1_000)
    try:
        _, alpha = w2.cap_series(r, "alpha")
        cap = np.array([c["cap"].astype(float).mean() for _, c in r.eco()])
        np.testing.assert_allclose(alpha, 0.25 + 0.75 * cap)
        assert w2.cap_trend(r)["alpha"]["slope_per_10k"] is not None
        assert w2.cap_series(r, "q90")[1][0] > w2.cap_series(r, "median")[1][0]
    finally:
        monkey.undo()


def test_p5_biosphere_statistics_and_the_decomposition(report: dict) -> None:
    d = report["P5"]
    assert (
        d["population"]["per_arm"]["A"]["0"]["mean"] == 1000.0
        and d["population"]["per_arm"]["A"]["1"]["mean"] == 1010.0
    )
    share = d["storm_death_share"]["per_arm"]
    assert share["A"]["0"]["mean"] == pytest.approx(4 / 40) and share["E"]["0"]["mean"] == pytest.approx(
        2 / 38
    )
    assert share["WB"]["0"]["mean"] == 0.0
    assert d["births_per_step"]["per_arm"]["A"]["0"]["mean"] == pytest.approx(40 / 50)
    c = d["storm_deaths_per_step"]["contrasts"]
    assert c["A_minus_WB"]["seeds"]["0"]["mean"] == pytest.approx(
        4 / 50
    )  # the hazard and the rest of the wind
    assert c["A_minus_E"]["seeds"]["0"]["mean"] == pytest.approx(2 / 50)  # the jet
    assert c["E_minus_WB"]["seeds"]["0"]["mean"] == pytest.approx(2 / 50)  # the eddy wind
    assert c["A_minus_WB"]["seeds"]["0"]["mean"] == pytest.approx(
        c["A_minus_E"]["seeds"]["0"]["mean"] + c["E_minus_WB"]["seeds"]["0"]["mean"]
    )
    assert "cloud_band_share" in d and d["cloud_band_share"]["per_arm"]["A"]["0"]["n"] == 8
    sp = d["speed_mean"]["contrasts"]["A_minus_WB"]["seeds"]["0"]["mean"]
    assert sp == pytest.approx(0.001)


def test_storm_deaths_by_clade_and_z_band(runs: list[w2.Run]) -> None:
    r = next(x for x in runs if (x.arm, x.seed) == ("A", 0))
    monkey = pytest.MonkeyPatch()
    for k, v in (("ECO_WINDOW", 100), ("TRANSIENT", 1_000)):
        monkey.setattr(w2, k, v)
    try:
        sd = w2.storm_deaths(r)
        d = r.rd.read_events("deaths")
        m = d["step"] > 1_000
        assert sd["total"] == int(m.sum()) and sum(sd["by_clade"].values()) == sd["total"]
        assert sd["unattributed"] + sum(sd["by_z_band"]) == sd["total"]
        assert sum(sd["by_z_band"]) > 0.9 * sd["total"]  # every dying agent was in the dump a window before
    finally:
        monkey.undo()


def test_p6_flags_the_violations(tmp_path: Path) -> None:
    mp = pytest.MonkeyPatch()
    mp.setattr(w2, "TRANSIENT", 1_000)
    try:
        good = w2.load_run(_make(tmp_path, "A", 0))
        bad_path = _make(tmp_path, "WB", 0)
        rd = RunDir.open(bad_path)
        rows = rd.metrics_rows()
        for r in rows[100:]:
            r["max_occupancy"] = 44
            r["cell_overflow"] = 2
            r["diag"] = int(Diag.WEATHER_WARN)
            r["weather"]["hyper_max"] = 0.9
        (bad_path / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        out = w2.p6(w2.Runs([good, w2.load_run(bad_path)]))
    finally:
        mp.undo()
    g, b = out["A"]["0"], out["WB"]["0"]
    assert g["safe"] and g["max_occupancy"] == 20 and g["weather_warn_row_share"] == 0.0
    assert not b["safe"] and not b["max_occupancy_ok"] and b["cell_overflow_total"] > 0
    assert b["weather_warn_row_share"] == pytest.approx(80 / 180) and b[
        "hyper_gate_row_share"
    ] == pytest.approx(80 / 180)
    assert b["error_bits_set"] is False  # WEATHER_WARN is advisory, not an error bit
    assert g["overlap_deep_ok"] and g["overlap_deep_mean_share"] == pytest.approx(1 / 1000)


def test_the_report_is_json_safe_and_names_the_design(report: dict) -> None:
    text = json.dumps(report, allow_nan=False)
    assert json.loads(text)["design"]["blocks"] == 8
    assert report["design"]["sensitivity"] == [16, 500]  # the patched windows: 2 x 8 blocks of half the width
    assert {r["arm"] for r in report["runs"]} == {"A", "N", "WB", "S", "A1", "E"}


def test_the_cli_writes_the_report(family: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "w2.json"
    paths = [str(family / f"{a}_s{s}") for a in ("A", "N") for s in (0, 2)]
    assert w2.main([*paths, "--out", str(out)]) == 0
    data = json.loads(out.read_text())
    assert "P1a" in data and {"0", "2"} <= set(data["P1a"])


def test_the_cli_refuses_an_output_inside_a_run(family: Path) -> None:
    with pytest.raises(SystemExit):
        w2.main([str(family / "A_s0"), "--out", str(family / "A_s0" / "r.json")])


def test_the_run_directories_are_not_written(family: Path) -> None:
    before = sorted(p.name for p in (family / "A_s0").rglob("*"))
    w2.analyse([w2.load_run(family / "A_s0")])
    assert sorted(p.name for p in (family / "A_s0").rglob("*")) == before
