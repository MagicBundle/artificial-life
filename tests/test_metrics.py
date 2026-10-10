"""§19 metrics: bodies vs union-find, snapshot statistics, interval rows and carrying capacity (WP5)."""

from __future__ import annotations

import itertools
import json
import math

import mlx.core as mx
import numpy as np
import pytest

from alife import fields, layout, metrics, rng
from alife.config import Config, apply_overrides, dims
from alife.state import CounterIdx, LedgerIdx, init_state, state_digest, to_numpy
from helpers import make_scene

# --- helpers --------------------------------------------------------------------------------------------


def _bond(partner: np.ndarray, rev: np.ndarray, a: int, b: int) -> bool:
    """Add a symmetric bond a–b in the first free slots; False if either row is full or already bonded."""
    if a == b or b in partner[a]:
        return False
    fa, fb = np.flatnonzero(partner[a] < 0), np.flatnonzero(partner[b] < 0)
    if not fa.size or not fb.size:
        return False
    partner[a, fa[0]], rev[a, fa[0]] = b, fb[0]
    partner[b, fb[0]], rev[b, fb[0]] = a, fa[0]
    return True


def _union_find(partner: np.ndarray, alive: np.ndarray) -> np.ndarray:
    """Smallest row index of each row's component (bonds touching dead rows ignored)."""
    n = partner.shape[0]
    parent = np.arange(n)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in partner[i]:
            if j >= 0 and alive[i] and alive[j]:
                ri, rj = find(i), find(int(j))
                if ri != rj:
                    parent[max(ri, rj)] = min(ri, rj)
    return np.array([find(i) for i in range(n)])


def _chain(n_rows: int, order: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    partner = np.full((n_rows, layout.B), -1, np.int32)
    rev = np.full((n_rows, layout.B), -1, np.int32)
    for a, b in itertools.pairwise(order):
        assert _bond(partner, rev, int(a), int(b))
    return partner, rev


# --- components -----------------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(6))
def test_components_match_union_find(seed: int) -> None:
    rs = np.random.default_rng(seed)
    n = 400
    partner = np.full((n, layout.B), -1, np.int32)
    rev = np.full((n, layout.B), -1, np.int32)
    for _ in range(int(rs.integers(50, 700))):
        _bond(partner, rev, int(rs.integers(n)), int(rs.integers(n)))
    alive = rs.random(n) < 0.8
    if seed % 2 == 0:  # canonical dead rows hold no bonds; odd seeds keep dangling ones to test the mask
        dead = ~alive
        keep = ~dead[:, None] & ~(dead[np.maximum(partner, 0)] & (partner >= 0))
        partner, rev = np.where(keep, partner, -1), np.where(keep, rev, -1)
    lab, conv = metrics.components(mx.array(partner), mx.array(alive), 32)
    assert bool(conv)
    np.testing.assert_array_equal(np.array(lab), _union_find(partner, alive))
    assert np.array(lab).dtype == np.int32


@pytest.mark.parametrize("order", ["ordered", "reversed", "shuffled"])
def test_components_chain_2000(order: str) -> None:
    n = 2000
    rows = {
        "ordered": np.arange(n),
        "reversed": np.arange(n)[::-1],
        "shuffled": np.random.default_rng(7).permutation(n),
    }[order]
    partner, _ = _chain(n + 48, rows)  # a few unbonded rows after the chain
    alive = np.ones(n + 48, bool)
    lab, conv = metrics.components(mx.array(partner), mx.array(alive), 32)
    assert bool(conv)
    lab = np.array(lab)
    assert (lab[:n] == 0).all()
    np.testing.assert_array_equal(lab[n:], np.arange(n, n + 48))
    # Too few rounds: the flag reports it, and labels are still rows of the same component.
    lab2, conv2 = metrics.components(mx.array(partner), mx.array(alive), 2)
    assert not bool(conv2)
    assert (np.array(lab2)[:n] < n).all() and len(np.unique(np.array(lab2)[:n])) > 1
    assert not bool(metrics.components(mx.array(partner), mx.array(alive), 0)[1])


def test_components_without_bonds() -> None:
    partner = mx.full((10, layout.B), -1, dtype=mx.int32)
    lab, conv = metrics.components(partner, mx.ones((10,), dtype=mx.bool_), 0)
    assert bool(conv)
    np.testing.assert_array_equal(np.array(lab), np.arange(10))


# --- snapshot -------------------------------------------------------------------------------------------


def _agents(n: int, **cols: list) -> list[dict]:
    return [{k: v[i] for k, v in cols.items()} | {"pos": [2.0 + 3.0 * i, 5.0]} for i in range(n)]


def test_snapshot_bodies(micro_cfg: Config) -> None:
    # bodies: pair (0,1); 4-chain (2..5); 5-star (6 centre, 7..10); singletons 11, 12
    bonds = [(0, 1), (2, 3), (3, 4), (4, 5), (6, 7), (6, 8), (6, 9), (6, 10)]
    s = make_scene(micro_cfg, [{"pos": [2.0 + 3.0 * i, 5.0]} for i in range(13)], bonds=bonds)
    m = metrics.snapshot_metrics(s, micro_cfg, heavy=True)
    assert m["bodies_converged"] is True
    assert m["bodies_n"] == 3
    assert m["bodies_largest"] == 5
    assert m["bodies_mean_size"] == pytest.approx((4 + 16 + 25) / 11)
    hist = {"1": 2, "2": 1, "3-4": 1, "5-8": 1, "9-16": 0, "17-32": 0, "33-64": 0, "65+": 0}
    assert m["bodies_hist"] == hist
    assert m["frac_in_bodies_ge4"] == pytest.approx(9 / 13)


def test_snapshot_clades_life_history_and_genes(micro_cfg: Config) -> None:
    clades = [0, 0, 0, 0, 1, 1, 3, 8]  # 8 = n_init: immigrants
    energy = [100, 200, 300, 400, 500, 600, 700, 800]
    birth = [-1, -1, 3, 5, 7, 7, 8, 9]
    gen = [0, 0, 1, 2, 3, 3, 1, 0]
    sigma = np.linspace(0.01, 0.08, 8).tolist()
    tags = np.random.default_rng(1).uniform(-1, 1, (8, layout.T)).astype(np.float32)
    s = make_scene(
        micro_cfg,
        _agents(
            8,
            clade=clades,
            energy=energy,
            birth_step=birth,
            generation=gen,
            mut_sigma=sigma,
            tag=list(tags),
            brain_cost=[10, 20, 30, 40, 50, 60, 70, 80],
        ),
        step=mx.array(11, dtype=mx.int32),
    )
    m = metrics.snapshot_metrics(s, micro_cfg, heavy=False)
    p = np.array([4, 2, 1, 1]) / 8
    assert m["clades_alive"] == 4
    assert m["clade_entropy"] == pytest.approx(-(p * np.log(p)).sum())
    assert m["clade_simpson"] == pytest.approx(1 - (p * p).sum())
    assert m["top_clade_share"] == pytest.approx(0.5)
    assert m["energy_mean"] == pytest.approx(450.0)
    assert m["energy_median"] == pytest.approx(450.0)
    assert m["age_mean"] == pytest.approx(np.mean([10 - b for b in birth]))
    assert m["generation_max"] == 3 and m["generation_mean"] == pytest.approx(np.mean(gen))
    q = np.quantile(np.array(sigma, np.float32).astype(np.float64), [0.1, 0.5, 0.9])
    assert [m["sigma_q10"], m["sigma_q50"], m["sigma_q90"]] == pytest.approx(q.tolist())
    assert m["rate_q50"] == pytest.approx(micro_cfg.mutation.rate_init)
    assert m["tag_var"] == pytest.approx(np.trace(np.cov(tags.astype(np.float64).T, ddof=0)))
    assert m["brain_cost_mean"] == pytest.approx(45.0)
    assert "bodies_n" not in m and "weight_diversity" not in m
    json.dumps(m)


def test_snapshot_energy_ledger_and_fields(micro_cfg: Config) -> None:
    chem = np.zeros((layout.NCH, dims(micro_cfg).fg, dims(micro_cfg).fg), np.float32)
    chem[1] = 2.0
    s = make_scene(micro_cfg, _agents(3, energy=[5, 6, 7]), resource=11, chem=chem)
    m = metrics.snapshot_metrics(s, micro_cfg, heavy=False)
    fg2 = dims(micro_cfg).fg ** 2
    assert m["E_agents"] == 18
    assert m["E_structure"] == 3 * micro_cfg.energy.structure
    assert m["E_field"] == 11 * fg2
    assert m["E_total"] == int(to_numpy(s.total))
    assert m["ledger_error"] == 0
    assert m["resource_mean"] == pytest.approx(11.0)
    assert m["chem_mean"] == pytest.approx([0.0, 2.0, 0.0])
    assert m["diag"] == 0

    # A ledger that accounts for the change keeps the error at 0; a leaked quantum does not.
    cum = np.zeros(len(LedgerIdx), np.int64)
    cum[LedgerIdx.INJECT_REGEN], cum[LedgerIdx.HEAT_BASAL] = 1000, 400
    moved = s._replace(
        total=s.total + 600,
        ledger_cum=mx.array(cum),
        energy=mx.where(s.alive, s.energy + 200, 0),
    )
    assert metrics.snapshot_metrics(moved, micro_cfg, heavy=False)["ledger_error"] == 0
    one = np.zeros((dims(micro_cfg).fg,) * 2, np.int32)
    one[3, 4] = 1
    leaked = moved._replace(resource=moved.resource + mx.array(one))
    assert metrics.snapshot_metrics(leaked, micro_cfg, heavy=False)["ledger_error"] == 2


def test_snapshot_empty_population(micro_cfg: Config) -> None:
    s = make_scene(micro_cfg, [])
    m = metrics.snapshot_metrics(s, micro_cfg, heavy=True)
    assert m["E_agents"] == 0 and m["E_structure"] == 0 and m["ledger_error"] == 0
    for key in ("energy_mean", "age_mean", "generation_max", "clade_entropy", "tag_var", "sigma_q50"):
        assert m[key] is None, key
    assert m["clades_alive"] == 0
    assert m["bodies_n"] == 0 and m["bodies_largest"] == 0 and m["frac_in_bodies_ge4"] is None
    assert m["weight_diversity"] is None
    json.dumps(m)


def _pairwise_mean(x: np.ndarray) -> float:
    d = np.sqrt(((x[:, None, :] - x[None, :, :]) ** 2).sum(-1))
    iu = np.triu_indices(len(x), 1)
    return float(d[iu].mean())


def test_weight_diversity_excludes_unflushed_rows(micro_cfg: Config) -> None:
    d = dims(micro_cfg)
    rs = np.random.default_rng(3)
    w1 = rs.normal(0, 0.3, (6, d.w1_rows, d.h)).astype(np.float32)
    w2 = rs.normal(0, 0.3, (6, d.w2_rows, layout.D_OUT)).astype(np.float32)
    w1[4:] += 3.0  # rows 4, 5 were born during the last step: their W rows are stale garbage
    birth = [-1, 2, 3, 5, 9, 9]
    s = make_scene(
        micro_cfg,
        _agents(6, W1=list(w1), W2=list(w2), birth_step=birth),
        step=mx.array(10, dtype=mx.int32),
    )
    flat = np.concatenate([w1.reshape(6, -1), w2.reshape(6, -1)], axis=1).astype(np.float64)
    got = metrics.snapshot_metrics(s, micro_cfg, heavy=True)["weight_diversity"]
    assert got == pytest.approx(_pairwise_mean(flat[:4]), rel=1e-4)
    # At step 0 nothing is pending, so founders are sampled.
    s0 = make_scene(micro_cfg, _agents(6, W1=list(w1), W2=list(w2)))
    got0 = metrics.snapshot_metrics(s0, micro_cfg, heavy=True)["weight_diversity"]
    assert got0 == pytest.approx(_pairwise_mean(flat), rel=1e-4)
    # Fewer than two eligible rows: no pairs.
    s1 = make_scene(micro_cfg, _agents(2, birth_step=[-1, 9]), step=mx.array(10, dtype=mx.int32))
    assert metrics.snapshot_metrics(s1, micro_cfg, heavy=True)["weight_diversity"] is None


def test_weight_diversity_close_kin_is_accurate(micro_cfg: Config) -> None:
    """Near-identical genomes (a shared base plus tiny mutations) must not lose the distance to cancellation.

    Uncentred, ``|x|² ≈ 400`` against ``dist² ≈ 0.01``, so the Gram trick in float32 is off by percent.
    """
    d = dims(micro_cfg)
    rs = np.random.default_rng(11)
    base1 = rs.normal(0, 0.3, (d.w1_rows, d.h))
    base2 = rs.normal(0, 0.3, (d.w2_rows, layout.D_OUT))
    w1 = (base1 + rs.normal(0, 1e-3, (6, d.w1_rows, d.h))).astype(np.float32)
    w2 = (base2 + rs.normal(0, 1e-3, (6, d.w2_rows, layout.D_OUT))).astype(np.float32)
    s = make_scene(micro_cfg, _agents(6, W1=list(w1), W2=list(w2)))
    flat = np.concatenate([w1.reshape(6, -1), w2.reshape(6, -1)], axis=1).astype(np.float64)
    got = metrics.snapshot_metrics(s, micro_cfg, heavy=True)["weight_diversity"]
    assert got == pytest.approx(_pairwise_mean(flat), rel=1e-3)


def test_weight_diversity_sample_matches_numpy(tiny_cfg: Config) -> None:
    cfg = apply_overrides(tiny_cfg, ["run.diversity_sample=40"])
    s = init_state(cfg)
    s = s._replace(step=mx.array(3, dtype=mx.int32))
    perm = np.array(mx.random.permutation(dims(cfg).n, key=rng.observer_key(cfg.run.seed, 3)))
    alive = np.array(s.alive)
    rows = perm[alive[perm]][:40]
    w = np.concatenate(
        [np.array(s.W1)[rows].reshape(40, -1), np.array(s.W2)[rows].reshape(40, -1)], axis=1
    ).astype(np.float64)
    got = metrics.snapshot_metrics(s, cfg, heavy=True)["weight_diversity"]
    assert got == pytest.approx(_pairwise_mean(w), rel=1e-4)


@pytest.mark.parametrize("which", ["tiny_cfg", "tiny_cfg_bf16"])
def test_snapshot_is_pure_and_handles_weight_dtypes(which: str, request: pytest.FixtureRequest) -> None:
    cfg = request.getfixturevalue(which)
    s = init_state(cfg)
    before = state_digest(s)
    m = metrics.snapshot_metrics(s, cfg, heavy=True)
    assert state_digest(s) == before
    assert m["ledger_error"] == 0
    assert m["weight_diversity"] is not None and m["weight_diversity"] > 0
    assert m["clades_alive"] == cfg.population.n_init
    assert m["bodies_hist"]["1"] == cfg.population.n_init
    json.dumps(m)


# --- interval rows --------------------------------------------------------------------------------------


def _counters(rows: list[dict[CounterIdx, int]]) -> np.ndarray:
    c = np.zeros((len(rows), len(CounterIdx)), np.int32)
    for t, r in enumerate(rows):
        for k, v in r.items():
            c[t, k] = v
    return c


def test_interval_row_aggregation_and_kin_bias() -> None:
    C = CounterIdx
    counters = _counters(
        [
            {C.N_ALIVE: 100, C.BIRTHS: 3, C.BONDS_TOTAL: 10, C.SHARERS: 5, C.EMITTERS: 20, C.MAX_OCCUPANCY: 4,
             C.ATTACKS: 2, C.KIN_ATTACK_SUM: 300, C.KIN_ATTACK_BASE: 100, C.CELL_OVERFLOW: 1, C.V_CLIPS: 2},
            {C.N_ALIVE: 110, C.BIRTHS: 4, C.BONDS_TOTAL: 12, C.SHARERS: 6, C.EMITTERS: 22, C.MAX_OCCUPANCY: 9,
             C.ATTACKS: 1, C.KIN_ATTACK_SUM: 0, C.KIN_ATTACK_BASE: 200, C.DEATHS_KILLED: 2},
            {C.N_ALIVE: 90, C.BIRTHS: 0, C.BONDS_TOTAL: 8, C.SHARERS: 4, C.EMITTERS: 18, C.MAX_OCCUPANCY: 7,
             C.DEATHS_STARVED: 5, C.INSOLVENT: 3, C.BONDS_FORMED: 2, C.BONDS_BROKEN: 1},
        ]
    )  # fmt: skip
    ledgers = np.zeros((3, len(LedgerIdx)), np.int64)
    ledgers[:, LedgerIdx.FLOW_HARVEST] = [1000, 1000, 2000]
    ledgers[:, LedgerIdx.FLOW_STOLEN] = [100, 0, 300]
    ledgers[0, LedgerIdx.INJECT_REGEN] = 3 << 33  # int64 sums stay exact
    snap = {"E_total": 5, "diag": 0}
    row = metrics.interval_row(30, counters, ledgers, snap, {"wall_s": 2.0, "peak_mem_gb": 1.5})
    assert next(iter(row)) == "step"
    assert row["step"] == 30 and row["steps"] == 3
    assert row["wall_s"] == 2.0 and row["steps_per_s"] == pytest.approx(1.5) and row["peak_mem_gb"] == 1.5
    assert row["n_alive"] == pytest.approx(100.0)
    assert row["births"] == 7 and row["deaths_killed"] == 2 and row["deaths_starved"] == 5
    assert row["insolvent"] == 3 and row["bonds_formed"] == 2 and row["bonds_broken"] == 1
    assert row["bonds_total"] == pytest.approx(10.0)
    assert row["mean_bonds_per_agent"] == pytest.approx(2 * 30 / 300)
    assert row["sharers_frac"] == pytest.approx(15 / 300) and row["emitters_frac"] == pytest.approx(60 / 300)
    assert row["predation_ratio"] == pytest.approx(400 / 4000)
    assert row["kin_attack_bias"] == pytest.approx((300 - 300) / (256 * 3))
    assert row["kin_bond_bias"] is None  # no proposers
    assert row["max_occupancy"] == 9 and row["cell_overflow"] == 1 and row["v_clips"] == 2
    assert row["ledger"]["INJECT_REGEN"] == 3 << 33 and row["ledger"]["FLOW_STOLEN"] == 400
    assert list(row["ledger"]) == [i.name for i in LedgerIdx]
    assert row["E_total"] == 5 and row["diag"] == 0
    json.dumps(row)


def test_interval_row_kin_bias_signs_and_nulls() -> None:
    C = CounterIdx
    counters = _counters(
        [{C.N_ALIVE: 10, C.ATTACKS: 4, C.KIN_ATTACK_SUM: 0, C.KIN_ATTACK_BASE: 512,
          C.BOND_PROPOSERS: 2, C.KIN_BOND_SUM: 512, C.KIN_BOND_BASE: 0}]
    )  # fmt: skip
    row = metrics.interval_row(1, counters, np.zeros((1, 16), np.int64), {}, {"wall_s": 0.0})
    assert row["kin_attack_bias"] == pytest.approx(-0.5)  # kin spared
    assert row["kin_bond_bias"] == pytest.approx(1.0)  # kin preferred
    assert row["predation_ratio"] is None and row["steps_per_s"] is None and row["peak_mem_gb"] is None
    empty = metrics.interval_row(0, _counters([]), np.zeros((0, 16), np.int64), {}, {})
    assert empty["steps"] == 0 and empty["n_alive"] is None and empty["max_occupancy"] == 0


def test_interval_row_rejects_bad_input() -> None:
    c = _counters([{CounterIdx.N_ALIVE: 1}])
    with pytest.raises(ValueError, match="differ"):
        metrics.interval_row(1, c, np.zeros((2, 16), np.int64), {}, {})
    with pytest.raises(ValueError, match="collide"):
        metrics.interval_row(1, c, np.zeros((1, 16), np.int64), {"births": 3}, {})


# --- carrying capacity ----------------------------------------------------------------------------------


def test_carrying_capacity(tiny_cfg: Config) -> None:
    s = init_state(tiny_cfg)
    regen = int(np.array(fields.regen_rate(s.patch_pos, tiny_cfg)).astype(np.int64).sum())
    c_ref = tiny_cfg.energy.basal_cost + metrics.founder_brain_cost(tiny_cfg) + layout.C_REF_MARGIN
    k = metrics.carrying_capacity(s, tiny_cfg)
    assert k == pytest.approx(regen / c_ref, rel=1e-12)

    # Independent numpy reference of Σ regen_rate (§13.1, float64 torus Gaussians).
    L, fc, fg = tiny_cfg.world.size, tiny_cfg.world.field_cell, dims(tiny_cfg).fg
    pp = np.array(s.patch_pos, np.float64)
    c = (np.arange(fg) + 0.5) * fc

    def g(axis: int) -> np.ndarray:
        dd = c[None, :] - pp[:, axis : axis + 1]
        dd -= L * np.round(dd / L)
        return np.exp(-(dd**2) / (2 * tiny_cfg.fields.patch_sigma**2))

    fert = np.clip(tiny_cfg.fields.fertility_base + g(1).T @ g(0), 0, 1)
    ref = np.floor(tiny_cfg.fields.regen_max * fert).sum()
    assert regen == pytest.approx(ref, rel=1e-3)

    # The analytic founder cost matches the founders' stored costs (which are floored: −0.5 on average).
    founders = np.array(s.brain_cost)[: tiny_cfg.population.n_init]
    assert abs(metrics.founder_brain_cost(tiny_cfg) - 0.5 - founders.mean()) < 0.3


@pytest.mark.parametrize(("sigma", "clip"), [(0.1, 4.0), (1.0, 1.0), (2.0, 0.5)])
def test_mean_abs_clipped_normal(sigma: float, clip: float) -> None:
    x = np.clip(np.random.default_rng(0).normal(0, sigma, 2_000_000), -clip, clip)
    assert metrics._mean_abs_clipped_normal(sigma, clip) == pytest.approx(np.abs(x).mean(), rel=2e-3)
    assert metrics._mean_abs_clipped_normal(sigma, math.inf) == pytest.approx(sigma * math.sqrt(2 / math.pi))
