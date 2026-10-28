"""Block B of the emergence suite (DESIGN §44.3): the cross-fitted entropy-production estimator against chains
with known answers, the lattice and transition extraction, and the guards. CPU, numpy only."""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np
import pytest

import emergence_sim as sim
from alife.analysis.emergence import epr, lattice
from alife.analysis.emergence import frames as fm
from alife.analysis.emergence.__main__ import main

# --- chains with known answers ---


def ring_P(n: int, kp: float, km: float, dt: float) -> np.ndarray:
    q = np.zeros((n, n))
    for i in range(n):
        q[i, (i + 1) % n] += kp
        q[i, (i - 1) % n] += km
        q[i, i] -= kp + km
    w, v = np.linalg.eig(q)
    p = np.clip(np.real(v @ np.diag(np.exp(w * dt)) @ np.linalg.inv(v)), 0, None)
    return p / p.sum(1, keepdims=True)


def walk(
    rng: np.random.Generator, n: int, kp: float, km: float, dt: float, walkers: int, steps: int
) -> np.ndarray:
    """Sequences ``(walkers, steps + 1)`` of a ring chain started from the uniform law."""
    cum = ring_P(n, kp, km, dt).cumsum(1)
    x = np.empty((walkers, steps + 1), np.int64)
    x[:, 0] = rng.integers(0, n, walkers)
    for t in range(steps):
        u = rng.random(walkers)
        x[:, t + 1] = np.minimum((u[:, None] > cum[x[:, t]]).sum(1), n - 1)
    return x


def tab_of(x: np.ndarray, n: int, k: int = epr.K_BLOCKS) -> epr.EdgeTable:
    w = x.shape[0]
    uid = np.repeat(np.arange(w, dtype=np.int64), x.shape[1] - 1)
    return epr.edge_table(x[:, :-1].ravel(), x[:, 1:].ravel(), epr.block_ids(uid, k), n, k)


def sigma(tab: epr.EdgeTable, salt: int = 0, n_boot: int = 100) -> dict:
    # dt = 1000 makes the reported rate the cross-fitted value in nats per transition
    return epr.estimate(tab, 1000, salt=salt, n_boot=n_boot)


def test_diffusion_has_zero_ep_with_calibrated_error() -> None:
    rng = np.random.default_rng(1)
    n, reps = 12, 200
    x = walk(rng, n, 1.0, 1.0, 0.05, reps * 200, 500)  # 200 replicates of 200 walkers x 500 steps = 100k
    ests, zs, pls = [], [], []
    for r in range(reps):
        e = sigma(tab_of(x[r * 200 : (r + 1) * 200], n), salt=r, n_boot=50)
        ests.append(e["rate"])
        zs.append(e["z"])
        pls.append(e["plugin"])
    ests, zs = np.array(ests), np.array(zs, dtype=float)
    assert abs(ests.mean()) < 3 * ests.std(ddof=1) / math.sqrt(reps)
    assert (zs > 1.96).mean() <= 0.07
    assert np.mean(pls) > 0  # the plug-in is positively biased at this size; the cross-fit is not


def test_driven_cycle_recovers_the_known_value() -> None:
    rng = np.random.default_rng(2)
    n, kp, km, dt = 12, 1.0, 0.5, 0.05
    true = (kp - km) * math.log(kp / km) * dt  # nats per step, 0.017329
    x = walk(rng, n, kp, km, dt, 4000, 500)  # 2M transitions
    e = sigma(tab_of(x, n))
    assert e["rate"] == pytest.approx(true, rel=0.05)
    small = sigma(tab_of(x[:400, :500], n))  # 200k transitions
    assert abs(small["rate"] - true) < 3 * small["se"]


def test_weak_drive() -> None:
    rng = np.random.default_rng(3)
    kp, km, dt = 1.0, 0.9, 0.05
    true = (kp - km) * math.log(kp / km) * dt
    x = walk(rng, 12, kp, km, dt, 4000, 500)
    assert sigma(tab_of(x, 12))["rate"] == pytest.approx(true, rel=0.15)


def test_coarse_graining_is_monotone() -> None:
    rng = np.random.default_rng(4)
    x = walk(rng, 12, 1.0, 0.5, 0.05, 4000, 500)
    vals = []
    for m in (12, 6, 4, 3, 2):
        xc = x // (12 // m)
        vals.append(sigma(tab_of(xc, m))["rate"])
    assert all(a >= b - 1e-4 for a, b in itertools.pairwise(vals)), vals
    assert vals[0] > 0.015 and abs(vals[-1]) < 1e-3  # two cells: a two-state chain is reversible


def test_time_reversal_and_shuffle() -> None:
    rng = np.random.default_rng(5)
    x = walk(rng, 12, 1.0, 0.5, 0.05, 1000, 300)
    a = sigma(tab_of(x, 12))
    b = sigma(tab_of(x[:, ::-1], 12))
    assert a["rate"] == pytest.approx(b["rate"], rel=1e-9)  # the estimator is invariant under reversal
    # shuffling the frame order of every organism destroys the temporal dependence: zero EP
    sh = np.stack([rng.permutation(r) for r in x])
    s = sigma(tab_of(sh, 12))
    assert abs(s["z"]) < 3
    # swapping ci and cj negates the flux field
    uid = np.repeat(np.arange(1000), 300)
    lat = lattice.Lattice(12.0, 12)  # 144 cells, codes are not the ring but the field is still odd
    t = lattice.Transitions(
        lat.n_cells, 100, 300, 0, x[:, :-1].ravel().astype(np.int32), x[:, 1:].ravel().astype(np.int32),
        np.tile(np.arange(300, dtype=np.int32), 1000), uid, np.zeros(len(uid), np.int32),
        np.zeros(len(uid), np.int32), np.zeros(len(uid), np.float32),
    )  # fmt: skip
    f1, f2 = epr.flux_field(t, lat), epr.flux_field(t.reversed(), lat)
    assert np.abs(f1).max() > 0 and np.allclose(f1, -f2)


def test_pseudocount_edges_and_empty() -> None:
    tab = epr.edge_table(np.zeros(0), np.zeros(0), np.zeros(0), 4, 4)
    assert math.isnan(epr.plugin(tab))
    e = epr.estimate(tab, 100)
    assert e["rate"] is None and e["n"] == 0
    # one-way traffic only: finite, positive
    ci = np.tile([0, 1, 2], 400)
    cj = np.tile([1, 2, 0], 400)
    uid = np.arange(1200, dtype=np.int64)
    e = epr.estimate(epr.edge_table(ci, cj, epr.block_ids(uid), 3), 1000)
    assert e["rate"] > 1.0


# --- lattice, transitions and frames ---


def make_frames(
    rng: np.random.Generator,
    *,
    n: int = 3000,
    nf: int = 50,
    size: float = 160.0,
    sd: float = 1.5,
    swirl: float = 0.0,
    drift: float = 0.0,
    step: int = 100,
    uid_of=None,
    density=None,
) -> fm.Frames:
    """Organisms diffusing (``sd`` units per frame) on the periodic square; ``swirl`` adds a circulation about
    centre, ``drift`` a uniform wind in x (units per step, recorded)."""
    pos = rng.uniform(0, size, (n, 2)) if density is None else density(rng, n, size)
    uid = np.arange(n, dtype=np.int64)
    ps, us = [], []
    for _ in range(nf):
        ps.append(pos.copy())
        us.append(uid.copy() if uid_of is None else uid_of(len(ps) - 1, uid))
        v = rng.normal(0, sd, pos.shape)
        if swirl:
            r = pos - size / 2
            nr = np.maximum(np.linalg.norm(r, axis=1, keepdims=True), 1e-9)
            v += swirl * np.stack([-r[:, 1], r[:, 0]], 1) / nr
        v[:, 0] += drift * step
        pos = np.mod(pos + v, size)
    allp = np.concatenate(ps)
    off = np.arange(nf + 1, dtype=np.int64) * n
    dr = None
    if drift:
        dr = np.zeros((len(allp), 2), np.float16)
        dr[:, 0] = drift
    return fm.Frames(
        meta={"size": size, "height": 0.0},
        step=np.arange(nf, dtype=np.int64) * step,
        off=off,
        uid=np.concatenate(us),
        row=np.zeros(len(allp), np.int32),
        pos=fm.quantise(allp, size),
        clade=np.zeros(len(allp), np.int32),
        body=np.full(len(allp), -1, np.int32),
        drift=dr,
    )


def test_lattice_geometry() -> None:
    lat = lattice.lattice_for(160.0, 8.0)
    assert lat.ncell == 20 and lat.n_cells == 400 and lat.cell == 8.0
    xy = np.array([[0.0, 0.0], [7.99, 8.0], [159.99, 160.0], [-0.5, 168.0]])
    assert lat.index(xy).tolist() == [0, 1, 380, 381]
    with pytest.raises(ValueError):
        lat.merge(3)
    assert lat.merge(2).ncell == 10
    c = lat.centres()
    assert c.shape == (400, 2) and np.all(lat.index(c) == np.arange(400))


def test_uid_reuse_makes_no_transition() -> None:
    rng = np.random.default_rng(6)
    # every frame the same rows carry new uids: nobody is alive at two frames
    fr = make_frames(rng, n=200, nf=5, uid_of=lambda i, u: u + 1000 * i)
    t = lattice.extract_transitions(fr)
    assert len(t) == 0
    fr2 = make_frames(rng, n=200, nf=5, uid_of=lambda i, u: np.where(u < 50, u + 1000 * i, u))
    assert len(lattice.extract_transitions(fr2)) == 150 * 4


def test_uneven_spacing_skipped_and_stride() -> None:
    rng = np.random.default_rng(7)
    fr = make_frames(rng, n=100, nf=6)
    fr.step[3:] += 40
    t = lattice.extract_transitions(fr)
    assert t.skipped == 1 and t.n_pairs == 4
    fr = make_frames(rng, n=100, nf=7)
    t2 = lattice.extract_transitions(fr, stride=2)
    assert t2.dt == 200 and len(t2) == 100 * 5


def test_diffusion_on_the_lattice_has_zero_ep() -> None:
    rng = np.random.default_rng(8)
    r = epr.analyse_window(make_frames(rng), n_boot=50, scales=(8.0, 16.0))
    assert abs(r["own"]["z"]) < 3 and r["stationarity"]["nonstationary"] is False
    assert 0.1 < r["displacement"]["median_cells"] < 0.5 and r["ambient"] == 0.0
    assert r["vertical"] is None and len(r["blocks"]) == 16


def test_circulation_has_positive_ep_and_drift_is_removed_in_own_coordinates() -> None:
    rng = np.random.default_rng(9)
    sw = epr.analyse_window(make_frames(rng, swirl=1.2), n_boot=50, scales=(8.0,))
    assert sw["own"]["z"] > 3 and sw["own"]["rate"] > 0
    # a uniform wind alone: raw motion is irreversible, own motion is not
    wd = epr.analyse_window(make_frames(rng, drift=0.02), n_boot=50, scales=(8.0,))
    assert wd["raw"]["z"] > 10 and wd["raw"]["rate"] > 0
    assert abs(wd["own"]["z"]) < 3
    assert wd["ambient"] == pytest.approx(wd["raw"]["rate"] - wd["own"]["rate"])
    # both: own-motion keeps the circulation
    both = epr.analyse_window(make_frames(rng, swirl=1.2, drift=0.02), n_boot=50, scales=(8.0,))
    assert both["own"]["z"] > 3 and both["raw"]["rate"] > both["own"]["rate"]


def test_stationarity_guard_fires_on_a_relaxing_density() -> None:
    rng = np.random.default_rng(10)

    def packed(rng: np.random.Generator, n: int, size: float) -> np.ndarray:
        return rng.uniform(0, size / 4, (n, 2))  # a corner blob relaxing by diffusion

    fr = make_frames(rng, n=2000, nf=40, sd=4.0, density=packed)
    r = epr.analyse_window(fr, n_boot=20, scales=(8.0,))
    assert r["stationarity"]["nonstationary"] is True and r["stationarity"]["js_excess"] > 0.02


def test_subset_gate_returns_null_with_reason() -> None:
    rng = np.random.default_rng(11)
    fr = make_frames(rng, n=300, nf=8)
    t = lattice.extract_transitions(fr)
    r = epr.subset(t, t.uid < 20, t.dt)
    assert r["rate"] is None and "active pairs" in r["reason"]
    assert epr.subset(t, t.uid < 0, t.dt)["rate"] is None
    ok = epr.subset(t, np.ones(len(t), bool), t.dt, n_boot=10)
    assert ok["rate"] is not None or "active pairs" in ok.get("reason", "")


def test_vertical_bands_and_ambient_height_drift() -> None:
    rng = np.random.default_rng(12)
    n, nf, size, height = 1500, 40, 160.0, 80.0
    xy = rng.uniform(0, size, (n, 2))
    z = rng.uniform(0, height, n)
    ps, step = [], 100
    for _ in range(nf):  # buoyant lift: z climbs 3 units a frame, wraps from the top (a conveyor)
        ps.append(np.column_stack([xy, z]))
        xy = np.mod(xy + rng.normal(0, 1.5, xy.shape), size)
        z = np.mod(z + 3.0 + rng.normal(0, 1.0, n), height)
    allp = np.concatenate(ps)
    dr = np.zeros((len(allp), 3), np.float16)
    dr[:, 2] = 0.03
    fr = fm.Frames(
        meta={"size": size, "height": height},
        step=np.arange(nf, dtype=np.int64) * step,
        off=np.arange(nf + 1, dtype=np.int64) * n,
        uid=np.tile(np.arange(n, dtype=np.int64), nf),
        row=np.zeros(len(allp), np.int32),
        pos=fm.quantise(allp, size, height),
        clade=np.zeros(len(allp), np.int32),
        body=np.full(len(allp), -1, np.int32),
        drift=dr,
    )
    r = epr.analyse_window(fr, n_boot=30, scales=(8.0,))
    assert r["vertical"]["n_bands"] == 10 and r["vertical"]["z"] is not None
    assert r["has_drift"] is True


def test_time_blocks_are_unbiased_on_diffusion() -> None:
    rng = np.random.default_rng(13)
    t = lattice.extract_transitions(make_frames(rng, n=4000, nf=65))
    b = epr.time_blocks(t, 8)
    assert len(b) == 8 and all(v is not None for v in b)
    assert abs(float(np.mean(b))) < 3 * float(np.std(b, ddof=1)) / math.sqrt(8) + 1e-9


def test_js_divergence_known() -> None:
    assert epr.js_divergence(np.array([1, 1]), np.array([1, 1])) == 0.0
    assert epr.js_divergence(np.array([1, 0]), np.array([0, 1])) == pytest.approx(math.log(2))
    assert math.isnan(epr.js_divergence(np.zeros(2), np.ones(2)))


def test_determinism() -> None:
    rng = np.random.default_rng(14)
    fr = make_frames(rng, n=500, nf=10)
    a = epr.analyse_window(fr, n_boot=10, scales=(8.0,))
    b = epr.analyse_window(fr, n_boot=10, scales=(8.0,))
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_cli_epr_writes_json_csv(tmp_path: Path) -> None:
    rng = np.random.default_rng(15)
    fr = make_frames(rng, n=800, nf=20)
    fr.meta.update({"run": "toy", "window": [0, 1900], "frames_version": 1})
    path = fm.write_frames(tmp_path / "frames_0_1900.npz", fr)
    out = tmp_path / "out"
    assert main(["epr", str(path), "--out", str(out), "--n-boot", "10", "--no-plot"]) == 0
    d = json.loads((out / "epr.json").read_text())
    assert d["version"] == epr.EPR_VERSION and "own" in d and d["window"] == [0, 1900]
    rows = (out / "epr_scale.csv").read_text().splitlines()
    assert rows[0].startswith("cell,") and len(rows) == 1 + 4


# --- E18: independence of hash-salted streams ---


def test_hash_streams_with_distinct_salts_are_independent_and_equal_salts_fail() -> None:
    from alife.analysis.emergence import independence as ind

    keys = np.arange(20_000, dtype=np.uint64)
    res = ind.salts_independent(keys, [0x1234, 0xBEEF, 0x77])
    assert all(r["independent"] for r in res.values())
    same = ind.independence(ind.hash_stream(keys, 5), ind.hash_stream(keys, 5))
    assert not same["independent"] and same["r"] == pytest.approx(1.0)
    a = ind.hash_stream(keys, 9)
    rng = np.random.default_rng(0)
    # a stream sharing its low 40 bits with a (the same salt on part of the state): dependent bit positions
    partial = a ^ (rng.integers(0, 2**20, len(a)).astype(np.uint64) << np.uint64(40))
    part = ind.independence(a, partial)
    assert not part["independent"] and part["bit_positions_off_half"] >= 40
    af = a.astype(np.float64)
    noisy = ind.independence(af, af + rng.normal(0, 1e18, len(a)))
    assert noisy["z"] > 4 and not noisy["independent"]
    with pytest.raises(ValueError):
        ind.independence(a[:10], a[:10])


def test_plots_when_matplotlib_is_present(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from alife.analysis.emergence import epr_io

    fr = make_frames(np.random.default_rng(16), n=600, nf=12, swirl=1.0)
    res = epr.analyse_window(fr, n_boot=10, scales=(8.0, 16.0))
    paths = epr_io.write_outputs(tmp_path, res, fr)
    assert {p.name for p in paths} >= {"epr.json", "epr_scale.csv", "epr_scale.png", "epr_flux.png"}


def test_panel_command_fills_block_b_and_the_p4_endpoint(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "meta.json").write_text(json.dumps({"seed": 3, "config_hash": "abc", "start_step": 0}))
    (run / "metrics.jsonl").write_text("")
    rng = np.random.default_rng(17)
    paths = []
    for k, sw in enumerate((0.0, 1.2)):
        fr = make_frames(rng, n=2000, nf=33, swirl=sw)
        fr.meta.update({"run": "run", "window": [0, 3200], "frames_version": 1})
        paths.append(str(fm.write_frames(tmp_path / f"frames_{k}.npz", fr)))
    out = tmp_path / "panel"
    assert (
        main(["panel", str(run), "--frames", *paths, "--windows", "0:3200", "0:3200", "--out", str(out)]) == 0
    )
    p = json.loads((out / "panel.json").read_text())
    assert p["B"]["status"] == "ok" and len(p["B"]["windows"]) == 2
    e = p["endpoints"]["P4"]
    assert e["status"] == "ok" and len(e["blocks"]) == 32 and e["direction"] == 1
    assert (
        p["header"]["flags"]["nonstationary"] is False and p["header"]["versions"]["epr"] == epr.EPR_VERSION
    )
    assert p["A"]["status"] == "needs_missing"


# --- review fixes: source-sink currents, comoving groups, advection, block bias ---


def _reversible_window(seed: int, births: np.ndarray | None, n: int = 1200, burn: int = 50):
    rng = np.random.default_rng(seed)
    cum, pi = sim.chain(rng)
    law = pi if births is None else births
    us, cells = sim.simulate(rng, cum, law, law, n=n, nf=101 + burn)
    return rng, sim.to_frames(us[burn:], cells[burn:], rng)


def test_hot_spot_births_are_source_sink_current_not_circulation() -> None:
    """Reversible movement with births at a hot spot: the total current is large (a source-sink current),
    the circulating part, the headline, is about 0. The closed-chain cross-fit read it as EP (0.28, z 24)."""
    zs, circ, tot = [], [], []
    for seed in (100, 101):
        _, fr = _reversible_window(seed, sim.hot_spot())
        o = epr.analyse_window(fr, scales=(8.0,), n_boot=30)
        zs.append(o["own"]["z"])
        circ.append(o["own"]["rate"])
        tot.append(o["own"]["rate_total"])
        assert (
            o["gradient_null"]["rate_total"] > 0.5 * o["own"]["rate_total"]
        )  # the null reproduces the current
        assert abs(o["gradient_null"]["rate"]) < 0.03  # and reads no circulation on it
    assert min(tot) > 0.15 and max(abs(np.array(zs))) < 3.0
    assert max(circ) < 0.1 * min(tot)


def test_closed_reversible_population_has_no_source_sink_current() -> None:
    _, fr = _reversible_window(7, None)
    o = epr.analyse_window(fr, scales=(8.0,), n_boot=30)["own"]
    assert abs(o["z"]) < 3
    # the pooled total carries a small-sample bias (sparse pairs); it stays far below the hot-spot current
    assert o["rate_total"] < 0.5 * 0.28


def test_circulation_survives_the_source_sink_split() -> None:
    """A circulating population keeps (most of) its rate in the circulating part."""
    fr = make_frames(np.random.default_rng(20), n=2500, nf=60, swirl=1.2)
    o = epr.analyse_window(fr, scales=(8.0,), n_boot=30)["own"]
    assert o["z"] > 3 and o["rate"] > 0.6 * o["rate_total"]


def test_comoving_groups_have_an_honest_error() -> None:
    """Groups of 10 organisms with identical trajectories and EP 0: hashed-uid blocks put each group in both
    halves (se 3x too small, every run a false positive); blocks by the cell where organisms first appear
    keep a group in one block."""
    zs = []
    for seed in range(6):
        rng = np.random.default_rng(1000 + seed)
        cum, pi = sim.chain(rng)
        us, cells = sim.simulate(rng, cum, pi, pi, n=150, nf=101)
        us10 = [np.concatenate([u * 10 + g for g in range(10)]) for u in us]
        fr = sim.to_frames(us10, [np.tile(c, 10) for c in cells], rng)
        o = epr.analyse_window(fr, scales=(8.0,), n_boot=30)["own"]
        zs.append(o["z"])
    assert (np.abs(zs) > 1.96).mean() <= 0.2 and abs(np.mean(zs)) < 1.5


def test_recorded_drift_that_does_not_advect_is_not_subtracted() -> None:
    rng = np.random.default_rng(21)
    fr = make_frames(rng, n=2500, nf=40)  # diffusion, no wind
    fr.drift = np.zeros((len(fr.uid), 2), np.float16)
    fr.drift[:, 0] = 0.02  # a wind that was recorded but never applied to the motion
    r = epr.analyse_window(fr, scales=(8.0,), n_boot=20)
    assert r["drift_removal"]["applied"] is False and abs(r["drift_removal"]["slope"]) < 0.1
    assert abs(r["own"]["z"]) < 3 and r["ambient"] == 0.0  # not a spurious own-motion current
    adv = epr.analyse_window(make_frames(rng, n=2500, nf=40, drift=0.02), scales=(8.0,), n_boot=20)
    assert adv["drift_removal"]["applied"] is True and 0.8 < adv["drift_removal"]["slope"] < 1.2


def test_block_values_do_not_depend_on_the_block_sample_size() -> None:
    """The 16 block values of a circulating population average to the window value, with a half-size
    population too (fitting each block's own sparse table would bias them by its size)."""
    rates = {}
    for n in (1500, 3000):
        fr = make_frames(np.random.default_rng(22), n=n, nf=65, swirl=1.2)
        r = epr.analyse_window(fr, scales=(8.0,), n_boot=10)
        rates[n] = (np.mean(r["blocks"]), r["own"]["rate"])
        assert abs(rates[n][0] - rates[n][1]) < 0.35 * rates[n][1]
    assert abs(rates[1500][0] - rates[3000][0]) < 0.35 * rates[3000][0]


def test_gradient_null_keeps_starts_and_redraws_destinations() -> None:
    fr = make_frames(np.random.default_rng(23), n=1500, nf=20, swirl=1.2)
    t = lattice.extract_transitions(fr)
    n = epr.gradient_null(t, np.random.default_rng(0))
    assert len(n) == len(t) and np.array_equal(n.ci, t.ci) and not np.array_equal(n.cj, t.cj)
    assert epr.stationarity(n)["js"] == epr.stationarity(t)["js"]


def test_affinity_residual_removes_a_potential_and_keeps_a_cycle() -> None:
    # a 4-cycle 0-1-2-3-0: affinity along the cycle is circulation; add a potential to the same edges
    edges = np.array([[0, 1], [1, 2], [2, 3], [0, 3]])
    p = np.array([0.0, 1.0, -0.5, 0.3])
    grad = p[edges[:, 0]] - p[edges[:, 1]]
    n = 4000.0
    for aff, expect_cycle in ((grad, False), (grad + np.array([1.0, 1.0, 1.0, -1.0]), True)):
        f = n / (1 + np.exp(-aff))
        res = epr.affinity_residual(f[None], (n - f)[None], edges)[0]
        assert (np.abs(res).max() > 0.5) == expect_cycle


def test_independence_detects_a_rotated_copy_and_has_a_small_false_failure_rate() -> None:
    from alife.analysis.emergence import independence as ind

    keys = np.arange(6000, dtype=np.uint64)
    a = ind.hash_stream(keys, 11)
    rotl32 = (a << np.uint64(32)) | (a >> np.uint64(32))
    res = ind.independence(a, rotl32, n_perm=50)
    assert not res["independent"] and res["bit_rotation_pairs_off_half"] > 0
    rotl7 = (a << np.uint64(7)) | (a >> np.uint64(57))
    assert not ind.independence(a, rotl7, n_perm=50)["independent"]
    rng = np.random.default_rng(5)
    salts = rng.integers(0, 2**62, (60, 2))

    def failed(s: int, t: int) -> bool:
        res = ind.independence(ind.hash_stream(keys, s), ind.hash_stream(keys, t), n_perm=50)
        return not res["independent"]

    # well-separated 64-bit salts: about 1% false failures per pair is expected, a few of 60 at most
    assert sum(failed(int(s), int(t)) for s, t in salts) <= 3
    # a finding: salts that differ only in low bits give weakly dependent streams (single-bit agreement)
    assert sum(failed(i, i + 1) for i in range(40)) >= 3
