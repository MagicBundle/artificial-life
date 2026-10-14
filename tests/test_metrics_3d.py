"""§27.8 and §27.9 3D metrics: ``Counter3D`` and the per-dims counter width, interval rows with ``nd = 3`` (no
silent reshape, ``HEAT_BUOYANCY`` named in 3D and ``RESERVED`` kept in 2D, the 3D interval keys), the snapshot
over the volume (chemical means, the global closure with slot 15, the 3D keys against a host reference), the
carrying capacity (the occupancy-weighted regeneration, §27.10, and the cap's upkeep) and resim's
comparison."""

from __future__ import annotations

import json
import math
from collections import Counter

import mlx.core as mx
import numpy as np
import pytest

from alife import fields, layout, metrics
from alife.analysis import resim
from alife.config import Config, dims, load_config
from alife.state import (
    N_COUNTERS,
    Counter3D,
    CounterIdx,
    LedgerIdx,
    State3,
    _initial_state,
    init_state,
    ledger_names,
    n_counters,
    to_numpy,
)
from helpers import as_3d, make_scene

C = CounterIdx


@pytest.fixture(scope="module")
def cfg3(micro_cfg: Config) -> Config:
    """``micro`` in 3D (FZ 6, FG 12) with an active buoyancy cap and its costs."""
    return as_3d(
        micro_cfg,
        physics={"buoyancy_max": 0.05},
        mutation={"buoy_init": 0.25, "buoy_sigma": 0.02},
        energy={"buoyancy_cost": 120, "buoyancy_cap_cost": 30},
    )


def _counters3(t: int, rs: np.random.Generator) -> np.ndarray:
    """(t, 30) counters: random event counts, then the three Counter3D columns."""
    c = rs.integers(0, 50, (t, n_counters(3))).astype(np.int32)
    c[:, C.N_ALIVE] = rs.integers(100, 200, t)
    return c


# --- counters -----------------------------------------------------------------------------------------------


def test_counter3d_follows_counteridx() -> None:
    assert [int(i) for i in Counter3D] == list(range(N_COUNTERS, N_COUNTERS + 3))
    assert [i.name for i in Counter3D] == ["Z_FLOOR_CLAMPS", "Z_SURFACE_CLAMPS", "GROUNDED"]
    assert len(CounterIdx) == N_COUNTERS == n_counters(2) == 27 and n_counters(3) == 30


# --- interval rows ------------------------------------------------------------------------------------------


def test_interval_row_3d_sums_the_shared_counters_and_names_slot_15() -> None:
    """T = 9 is the trap: 9·30 = 10·27, so a reshape by 27 would succeed with scrambled columns."""
    rs = np.random.default_rng(4)
    t = 9
    c = _counters3(t, rs)
    lg = rs.integers(0, 1 << 40, (t, len(LedgerIdx))).astype(np.int64)
    row = metrics.interval_row(90, c, lg, {"E_total": 1}, {}, nd=3)
    assert row["steps"] == t and row["births"] == int(c[:, C.BIRTHS].sum())
    assert row["n_alive"] == pytest.approx(c[:, C.N_ALIVE].mean()) and row["n_alive_min"] == c[:, 0].min()
    assert row["deaths_catastrophe"] == int(c[:, C.DEATHS_CATASTROPHE].sum())
    assert row["max_occupancy"] == int(c[:, C.MAX_OCCUPANCY].max())
    assert list(row["ledger"]) == list(ledger_names(3)) and "RESERVED" not in row["ledger"]
    assert row["ledger"]["HEAT_BUOYANCY"] == int(lg[:, LedgerIdx.HEAT_BUOYANCY].sum())
    assert all(row["ledger"][n] == int(lg[:, i].sum()) for i, n in enumerate(ledger_names(3)))
    json.dumps(row)
    with pytest.raises(ValueError, match="30 columns, a 2D world has 27"):
        metrics.interval_row(90, c, lg, {}, {})  # the default dims refuse 3D counters
    with pytest.raises(ValueError, match="27 columns, a 3D world has 30"):
        metrics.interval_row(90, c[:, :N_COUNTERS], lg, {}, {}, nd=3)


def test_interval_row_2d_keeps_reserved() -> None:
    c = np.zeros((2, N_COUNTERS), np.int32)
    lg = np.zeros((2, len(LedgerIdx)), np.int64)
    row = metrics.interval_row(2, c, lg, {}, {})
    assert list(row["ledger"]) == [i.name for i in LedgerIdx] == list(ledger_names(2))
    assert row["ledger"]["RESERVED"] == 0 and "HEAT_BUOYANCY" not in row["ledger"]
    assert row == metrics.interval_row(2, c, lg, {}, {}, nd=2)


# --- snapshot -----------------------------------------------------------------------------------------------


def test_snapshot_3d_means_over_the_volume_and_closes_with_slot_15(cfg3: Config) -> None:
    d = dims(cfg3)
    rs = np.random.default_rng(8)
    agents = [
        dict(pos=(rs.uniform(0, 48), rs.uniform(0, 48), rs.uniform(d.z_lo, d.z_hi)), energy=30_000 + k)
        for k in range(10)
    ]
    chem = rs.uniform(0, 4, (layout.NCH, *d.field_shape)).astype(np.float32)
    s = make_scene(cfg3, agents, bonds=[(0, 1), (1, 2)], chem=chem)
    spent = 1234  # quanta paid for buoyancy since t = 0: TOTAL fell, slot 15 holds them
    cum = np.zeros(len(LedgerIdx), np.int64)
    cum[LedgerIdx.HEAT_BUOYANCY] = spent
    s = s._replace(ledger_cum=mx.array(cum), total0=s.total + spent)
    m = metrics.snapshot_metrics(s, cfg3, heavy=True)
    assert m["ledger_error"] == 0
    assert m["chem_mean"] == pytest.approx(chem.reshape(layout.NCH, -1).mean(axis=1).tolist(), rel=1e-5)
    assert m["resource_mean"] == m["E_field"] / d.n_cells
    assert m["bodies_largest"] == 3 and m["weight_diversity"] is not None
    json.dumps(m)
    s2 = s._replace(ledger_cum=mx.zeros_like(s.ledger_cum))  # the payment without its sink
    assert metrics.snapshot_metrics(s2, cfg3, heavy=False)["ledger_error"] == spent


# --- carrying capacity --------------------------------------------------------------------------------------


def test_carrying_capacity_3d_includes_the_cap_upkeep(cfg3: Config) -> None:
    s = make_scene(cfg3, [])
    rate = to_numpy(fields.regen_rate(s.patch_pos, cfg3)).astype(np.int64)
    assert rate.shape == dims(cfg3).field_shape
    regen = int(rate[0].sum())  # nobody alive: the floor layer alone (§27.10)
    upkeep = math.floor(30 * 0.25)  # floor(buoyancy_cap_cost·buoy_init) = 7
    c_ref = cfg3.energy.basal_cost + metrics.founder_brain_cost(cfg3) + layout.C_REF_MARGIN + upkeep
    assert metrics.carrying_capacity(s, cfg3) == pytest.approx(regen / c_ref, rel=1e-12)
    off = as_3d(cfg3, energy={"buoyancy_cap_cost": 0})
    assert metrics.carrying_capacity(s, off) == pytest.approx(regen / (c_ref - upkeep), rel=1e-12)


# --- resim --------------------------------------------------------------------------------------------------


def test_resim_compare_uses_the_3d_names() -> None:
    rs = np.random.default_rng(2)
    c = _counters3(5, rs)
    lg = rs.integers(0, 1000, (5, len(LedgerIdx))).astype(np.int64)
    stored = {**metrics.interval_row(50, c, lg, {}, {}, nd=3), "E_total": 77}
    stored = json.loads(json.dumps(stored))  # as read back from metrics.jsonl
    counters, ledgers = [mx.array(r) for r in c], [mx.array(r) for r in lg]
    assert resim._compare(stored, counters, ledgers, 77, nd=3) == []
    stored["ledger"]["HEAT_BUOYANCY"] += 1
    assert resim._compare(stored, counters, ledgers, 77, nd=3) == ["ledger"]
    with pytest.raises(ValueError, match="columns"):
        resim._compare(stored, counters, ledgers, 77)  # a 3D run compared as 2D is refused, not scrambled


# --- §27.9: the 3D interval keys ----------------------------------------------------------------------------

INTERVAL_3D = ("z_floor_clamps", "z_surface_clamps", "grounded_frac", "buoy_heat")


def test_interval_row_3d_keys_are_sums_and_a_weighted_share() -> None:
    rs = np.random.default_rng(11)
    t = 7
    c = _counters3(t, rs)
    lg = rs.integers(0, 1 << 30, (t, len(LedgerIdx))).astype(np.int64)
    row = metrics.interval_row(70, c, lg, {}, {}, nd=3)
    assert row["z_floor_clamps"] == int(c[:, Counter3D.Z_FLOOR_CLAMPS].sum())
    assert row["z_surface_clamps"] == int(c[:, Counter3D.Z_SURFACE_CLAMPS].sum())
    # GROUNDED over the alive agent-steps (the same weighting as every per-agent fraction of the row)
    assert row["grounded_frac"] == pytest.approx(
        c[:, Counter3D.GROUNDED].sum() / c[:, CounterIdx.N_ALIVE].sum(), rel=1e-15
    )
    assert row["buoy_heat"] == int(lg[:, 15].sum()) == row["ledger"]["HEAT_BUOYANCY"]
    keys = list(row)
    assert keys[keys.index("v_clips") + 1 :] == list(INTERVAL_3D)  # appended after the shared keys
    assert all(type(row[k]) is int for k in INTERVAL_3D if k != "grounded_frac")
    json.dumps(row)


def test_interval_row_3d_grounded_frac_without_alive_steps_is_none() -> None:
    c = np.zeros((3, n_counters(3)), np.int32)
    c[:, Counter3D.GROUNDED] = (
        5  # cannot happen (grounded counts alive agents), but the ratio must not divide
    )
    row = metrics.interval_row(3, c, np.zeros((3, len(LedgerIdx)), np.int64), {}, {}, nd=3)
    assert row["grounded_frac"] is None and row["buoy_heat"] == 0 and row["z_floor_clamps"] == 0


def test_interval_row_2d_has_no_3d_keys() -> None:
    row = metrics.interval_row(2, np.zeros((2, N_COUNTERS), np.int32), np.zeros((2, 16), np.int64), {}, {})
    assert not set(INTERVAL_3D) & set(row)
    assert list(row)[-1] == "v_clips"


def test_resim_compare_checks_the_3d_interval_keys() -> None:
    rs = np.random.default_rng(3)
    c = _counters3(5, rs)
    lg = rs.integers(0, 1000, (5, len(LedgerIdx))).astype(np.int64)
    stored = json.loads(json.dumps(metrics.interval_row(50, c, lg, {}, {}, nd=3)))
    counters, ledgers = [mx.array(r) for r in c], [mx.array(r) for r in lg]
    assert resim._compare(stored, counters, ledgers, 0, nd=3) == []
    for key in INTERVAL_3D:
        bad = {**stored, key: stored[key] + 1}
        assert resim._compare(bad, counters, ledgers, 0, nd=3) == [key]
    old = {
        k: v for k, v in stored.items() if k not in INTERVAL_3D
    }  # a row written before PR 4 still compares
    assert resim._compare(old, counters, ledgers, 0, nd=3) == []


# --- §27.9: the 3D snapshot keys ----------------------------------------------------------------------------


def _words(text: str) -> list[str]:
    return text.split()


SNAPSHOT_2D = _words(
    """E_agents E_structure E_field E_total ledger_error energy_mean energy_median age_mean generation_max
    generation_mean clades_alive clade_entropy clade_simpson top_clade_share tag_var sigma_q10 sigma_q50
    sigma_q90 rate_q10 rate_q50 rate_q90 frac_q10 frac_q50 frac_q90 brain_cost_mean resource_mean chem_mean
    diag"""
)
HEAVY_2D = _words(
    """bodies_n bodies_largest bodies_mean_size bodies_hist frac_in_bodies_ge4 bodies_converged
    weight_diversity"""
)
SNAPSHOT_3D = _words(
    """height_mean height_q25 height_q50 height_q75 surface_frac speed_mean pitch_abs_mean buoy_cap_q10
    buoy_cap_q50 buoy_cap_q90 band_frac clade_height_mi"""
)
HEAVY_3D = ["body_z_extent_mean", "body_z_extent_max"]

# Heights straddling the band edges 6, 12 and 18 of a height-24 world and the surface band (z_hi = 23,
# floor_eps = 0.1): 22.95 and 23.0 are within floor_eps of the surface bound, 22.0 is not.
Z = [1.0, 1.0, 2.5, 5.99, 6.0, 9.0, 12.0, 14.5, 17.99, 18.0, 22.95, 23.0]
CLADE = [0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 8, 8]  # 8 = n_init: immigrants share one bin
BONDS = [(0, 1), (2, 3), (3, 4), (9, 10), (10, 11)]  # bodies {0, 1}, {2, 3, 4} and {9, 10, 11}


def _scene3(cfg: Config) -> State3:
    rs = np.random.default_rng(5)
    n = len(Z)
    agents = [
        dict(
            pos=(2.0 + 3.0 * i, 5.0, Z[i]),
            vel=rs.normal(0.0, 0.01, 3),
            pitch=rs.uniform(-1.2, 1.2),
            buoy_cap=rs.uniform(0.0, 1.0),
            clade=CLADE[i],
        )
        for i in range(n)
    ]
    return make_scene(cfg, agents, bonds=BONDS)


def _band_ref(z: np.ndarray) -> np.ndarray:
    return (z >= 6.0).astype(int) + (z >= 12.0) + (z >= 18.0)


def test_snapshot_3d_keys_against_a_host_reference(micro3d_cfg: Config) -> None:
    cfg = micro3d_cfg
    assert dims(cfg).z_hi == 23.0 and cfg.physics.floor_eps == 0.1 and cfg.population.n_init == 8
    s = _scene3(cfg)
    m = metrics.snapshot_metrics(s, cfg, heavy=True)
    json.dumps(m)

    a = {k: to_numpy(getattr(s, k)) for k in ("alive", "pos", "vel", "pitch", "buoy_cap", "clade")}
    al = a["alive"]
    z = a["pos"][al, 2].astype(np.float64)
    n = z.size
    assert n == len(Z)
    q = np.quantile(z, [0.25, 0.5, 0.75])
    assert m["height_mean"] == pytest.approx(z.mean(), rel=1e-12)
    assert [m["height_q25"], m["height_q50"], m["height_q75"]] == pytest.approx(q.tolist(), rel=1e-12)
    assert m["surface_frac"] == pytest.approx(2 / n)
    speed = [math.sqrt(sum(float(c) ** 2 for c in v)) for v in a["vel"][al]]
    assert m["speed_mean"] == pytest.approx(sum(speed) / n, rel=1e-12)
    assert m["pitch_abs_mean"] == pytest.approx(np.abs(a["pitch"][al].astype(np.float64)).mean(), rel=1e-12)
    bq = np.quantile(a["buoy_cap"][al].astype(np.float64), [0.1, 0.5, 0.9])
    assert [m["buoy_cap_q10"], m["buoy_cap_q50"], m["buoy_cap_q90"]] == pytest.approx(bq.tolist(), rel=1e-12)

    band = _band_ref(z)
    assert m["band_frac"] == pytest.approx([float((band == b).sum()) / n for b in range(4)])
    # bands [0, 6): 1, 1, 2.5, 5.99; [6, 12): 6, 9; [12, 18): 12, 14.5, 17.99; [18, 24]: 18, 22.95, 23
    assert m["band_frac"] == pytest.approx([4 / n, 2 / n, 3 / n, 3 / n])
    pc, pb, pj = Counter(CLADE), Counter(band.tolist()), Counter(zip(CLADE, band.tolist(), strict=True))
    mi = sum(c / n * math.log((c / n) / ((pc[i] / n) * (pb[j] / n))) for (i, j), c in pj.items())
    assert mi > 0.1  # the heights are clade-structured, so the reference is not the trivial 0
    assert m["clade_height_mi"] == pytest.approx(mi, rel=1e-12)

    pz = a["pos"][:, 2].astype(np.float64)
    ext = [max(pz[g]) - min(pz[g]) for g in ([0, 1], [2, 3, 4], [9, 10, 11])]
    assert m["bodies_n"] == 3
    assert m["body_z_extent_mean"] == pytest.approx(sum(ext) / 3, rel=1e-12)
    assert m["body_z_extent_max"] == pytest.approx(max(ext), rel=1e-12)
    assert ext[0] == 0.0 and m["body_z_extent_max"] == pytest.approx(5.0)


def test_snapshot_3d_band_edges_belong_to_the_band_above() -> None:
    z = np.array([1.0, 5.999999, 6.0, 11.999999, 12.0, 17.999999, 18.0, 23.0, 24.0], np.float32)
    assert metrics.z_band(z, 24.0).tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 3]
    # the edges of a height-10 world (2.5, 5, 7.5) are exact in binary, so z·4 / height is exact too
    assert metrics.z_band(np.array([2.4999998, 2.5, 5.0, 7.5], np.float32), 10.0).tolist() == [0, 1, 2, 3]
    # height 49: z·(4/49) rounds to 0.9999999999999999 at the first edge 12.25, z·4/49 is exactly 1
    assert metrics.z_band(np.array([12.25, 24.5, 36.75], np.float32), 49.0).tolist() == [1, 2, 3]


def test_snapshot_3d_heavy_keys_only_when_heavy(micro3d_cfg: Config) -> None:
    s = _scene3(micro3d_cfg)
    light = metrics.snapshot_metrics(s, micro3d_cfg, heavy=False)
    assert not set(HEAVY_3D) & set(light) and set(SNAPSHOT_3D) <= set(light)


def test_snapshot_3d_key_order_and_the_2d_dict_is_unchanged(micro_cfg: Config, micro3d_cfg: Config) -> None:
    s2 = make_scene(micro_cfg, [{"pos": [2.0, 3.0]}, {"pos": [5.0, 3.0]}], bonds=[(0, 1)])
    assert list(metrics.snapshot_metrics(s2, micro_cfg, heavy=False)) == SNAPSHOT_2D
    assert list(metrics.snapshot_metrics(s2, micro_cfg, heavy=True)) == [*SNAPSHOT_2D, *HEAVY_2D]
    s3 = _scene3(micro3d_cfg)
    assert list(metrics.snapshot_metrics(s3, micro3d_cfg, heavy=False)) == [*SNAPSHOT_2D, *SNAPSHOT_3D]
    heavy3 = list(metrics.snapshot_metrics(s3, micro3d_cfg, heavy=True))
    assert heavy3 == [*SNAPSHOT_2D, *SNAPSHOT_3D, *HEAVY_2D[:-1], *HEAVY_3D, HEAVY_2D[-1]]
    assert "grounded_frac" not in heavy3  # the interval key of that name would make interval_row raise


def test_the_3d_snapshot_never_collides_with_the_interval_keys(micro3d_cfg: Config) -> None:
    snap = metrics.snapshot_metrics(_scene3(micro3d_cfg), micro3d_cfg, heavy=True)
    c = np.zeros((2, n_counters(3)), np.int32)
    row = metrics.interval_row(2, c, np.zeros((2, 16), np.int64), snap, {}, nd=3)
    assert set(INTERVAL_3D) <= set(row) and set(SNAPSHOT_3D) <= set(row)
    with pytest.raises(ValueError, match=r"collide.*grounded_frac"):
        metrics.interval_row(2, c, np.zeros((2, 16), np.int64), {"grounded_frac": 0.5}, {}, nd=3)


def test_snapshot_3d_single_clade_has_exactly_zero_mutual_information(micro3d_cfg: Config) -> None:
    agents = [dict(pos=(2.0 + 3.0 * i, 5.0, z), clade=0) for i, z in enumerate((1.0, 7.0, 13.0, 19.0, 23.0))]
    m = metrics.snapshot_metrics(make_scene(micro3d_cfg, agents), micro3d_cfg, heavy=False)
    assert m["clade_height_mi"] == 0.0 and m["band_frac"] == [0.2, 0.2, 0.2, 0.4]


def test_snapshot_3d_mutual_information_limits() -> None:
    mi = metrics._mutual_information
    clade = np.repeat(np.arange(4), 3)
    assert mi(clade, np.repeat(np.arange(4), 3), 4, 4) == pytest.approx(math.log(4), rel=1e-12)  # determined
    assert mi(clade, np.tile(np.arange(3), 4), 4, 4) == 0.0  # independent: every clade spans the bands evenly
    xor = mi(np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1]), 2, 2)
    assert xor == 0.0
    # 3 + 1 agents: p = [[.5, .25], [0, .25]] -> hand value
    got = mi(np.array([0, 0, 0, 1]), np.array([0, 0, 1, 1]), 2, 2)
    want = (
        0.5 * math.log(0.5 / (0.75 * 0.5))
        + 0.25 * math.log(0.25 / (0.75 * 0.5))
        + 0.25 * math.log(0.25 / (0.25 * 0.5))
    )
    assert got == pytest.approx(want, rel=1e-12)


def test_snapshot_3d_with_nobody_alive_is_all_none(micro3d_cfg: Config) -> None:
    m = metrics.snapshot_metrics(make_scene(micro3d_cfg, []), micro3d_cfg, heavy=True)
    for key in [*SNAPSHOT_3D, *HEAVY_3D]:
        assert m[key] is None, key
    json.dumps(m)


def test_snapshot_3d_bodies_without_bonds_have_no_extent(micro3d_cfg: Config) -> None:
    agents = [dict(pos=(2.0 + 3.0 * i, 5.0, 1.0 + 4.0 * i)) for i in range(4)]
    m = metrics.snapshot_metrics(make_scene(micro3d_cfg, agents), micro3d_cfg, heavy=True)
    assert m["bodies_n"] == 0 and m["body_z_extent_mean"] is None and m["body_z_extent_max"] is None


def test_snapshot_3d_is_pure_and_matches_an_eager_second_call(micro3d_cfg: Config) -> None:
    s = _scene3(micro3d_cfg)
    before = [to_numpy(a).tobytes() for a in s]
    first = metrics.snapshot_metrics(s, micro3d_cfg, heavy=True)
    assert [to_numpy(a).tobytes() for a in s] == before
    assert metrics.snapshot_metrics(s, micro3d_cfg, heavy=True) == first


# --- §27.10: with nobody alive the carrying capacity counts the floor layer ---------------------------------


def test_carrying_capacity_3d_is_the_2d_lattice_over_the_3d_cost(
    micro_cfg: Config, micro3d_cfg: Config
) -> None:
    """The floor layer regenerates as the 2D lattice (§27.5), so the 3D numerator of a world nobody lives in
    (the floor layer alone, §27.10) equals the 2D one for the same patches; only the divisor differs (the 3D
    founder's larger brain and its cap upkeep)."""
    s = make_scene(micro3d_cfg, [])
    regen2d = int(to_numpy(fields.regen_rate(s.patch_pos, micro_cfg)).astype(np.int64).sum())
    all_layers = int(to_numpy(fields.regen_rate(s.patch_pos, micro3d_cfg)).astype(np.int64).sum())
    assert all_layers > 3 * regen2d  # the upper layers hold food too: summing them would inflate K_est
    upkeep = math.floor(micro3d_cfg.energy.buoyancy_cap_cost * micro3d_cfg.mutation.buoy_init)
    assert upkeep > 0
    c_ref = (
        micro3d_cfg.energy.basal_cost + metrics.founder_brain_cost(micro3d_cfg) + layout.C_REF_MARGIN + upkeep
    )
    assert metrics.carrying_capacity(s, micro3d_cfg) == pytest.approx(regen2d / c_ref, rel=1e-12)


def test_carrying_capacity_3d_floor_layer_against_a_float64_host_reference(micro3d_cfg: Config) -> None:
    """Independent of ``fields``: the torus-Gaussian fertility of the floor layer in float64 (a world nobody
    lives in, so the floor layer alone counts, §27.10)."""
    cfg = micro3d_cfg
    s = make_scene(cfg, [])
    d, fi = dims(cfg), cfg.fields
    pp = to_numpy(s.patch_pos).astype(np.float64)
    centres = (np.arange(d.fg) + 0.5) * cfg.world.field_cell
    L = cfg.world.size

    def torus(dx: np.ndarray) -> np.ndarray:
        dx = dx - L * np.round(dx / L)
        return np.exp(-(dx**2) / (2.0 * fi.patch_sigma**2))

    fert = fi.fertility_base + sum(
        torus(centres[:, None] - p[1]) * torus(centres[None, :] - p[0]) for p in pp
    )
    host = int(np.floor(fi.regen_max * np.clip(fert, 0.0, 1.0)).sum())
    got = metrics.carrying_capacity(s, cfg) * (
        cfg.energy.basal_cost
        + metrics.founder_brain_cost(cfg)
        + layout.C_REF_MARGIN
        + math.floor(cfg.energy.buoyancy_cap_cost * cfg.mutation.buoy_init)
    )
    # the float32 and float64 floors of a cell differ by at most a quantum, a tiny part of the sum
    assert got == pytest.approx(host, abs=0.002 * host)


def test_carrying_capacity_3d_follows_the_live_environment_on_the_floor(
    micro_cfg: Config, micro3d_cfg: Config
) -> None:
    """The director's call (``step``, ``env``) on a world nobody lives in: the floor layer of the live
    regeneration, as the 2D function defines it for the same patches, levers and step."""
    s = make_scene(micro3d_cfg, [])
    boosted = s._replace(env_regen_mult=mx.array(2.0), env_season_amp=mx.array(0.5, dtype=mx.float32))
    boosted = boosted._replace(env_season_period=mx.array(40, dtype=mx.int32))
    step = mx.array(10, dtype=mx.int32)
    ref = int(
        to_numpy(fields.regen_rate(boosted.patch_pos, micro_cfg, step, boosted.env)).astype(np.int64).sum()
    )
    base = metrics.carrying_capacity(s, micro3d_cfg)
    live = metrics.carrying_capacity(boosted, micro3d_cfg, step, boosted.env)
    c_ref = (
        micro3d_cfg.energy.basal_cost
        + metrics.founder_brain_cost(micro3d_cfg)
        + layout.C_REF_MARGIN
        + math.floor(micro3d_cfg.energy.buoyancy_cap_cost * micro3d_cfg.mutation.buoy_init)
    )
    assert live == pytest.approx(ref / c_ref, rel=1e-12) and live > 1.5 * base
    assert metrics.carrying_capacity(boosted, micro3d_cfg) == base  # the plain call ignores the environment


def test_carrying_capacity_of_tiny3d_is_comparable_to_tiny(tiny_cfg: Config, tiny3d_cfg: Config) -> None:
    """Same seed, same patches, every founder on the floor (the benthic limit of the occupancy rule, §27.10;
    the uniform founders of ``init_state`` count the upper layers too): the numerators are equal, and K_est is
    within the cost difference of 2D's (1 456 when every layer is counted)."""
    s2, s3 = init_state(tiny_cfg), init_state(tiny3d_cfg)
    np.testing.assert_array_equal(to_numpy(s2.patch_pos), to_numpy(s3.patch_pos))
    z = mx.array(np.full(dims(tiny3d_cfg).n, dims(tiny3d_cfg).z_lo, np.float32))
    s3 = s3._replace(pos=mx.concatenate([s3.pos[:, :2], z[:, None]], axis=1))
    k2, k3 = metrics.carrying_capacity(s2, tiny_cfg), metrics.carrying_capacity(s3, tiny3d_cfg)
    assert 0.9 * k2 < k3 < k2


# --- §27.10: the occupancy-weighted carrying capacity ------------------------------------------------------

# Heights inside each of the six layers of ``micro3d_cfg`` (field cell 4, z in [1, 23]); up to ten agents per
# layer are spaced by 0.1, so none reaches a layer edge.
LAYER_Z = (2.0, 5.5, 9.5, 13.5, 17.5, 21.0)


def _c_ref(cfg: Config) -> float:
    """``c_ref`` of §10.10 and §27.7, written out."""
    return (
        cfg.energy.basal_cost
        + metrics.founder_brain_cost(cfg)
        + layout.C_REF_MARGIN
        + math.floor(cfg.energy.buoyancy_cap_cost * cfg.mutation.buoy_init)
    )


def _scene_by_layer(cfg: Config, counts: tuple[int, ...]) -> State3:
    """``counts[k]`` agents in layer ``k``, spread over the torus (rows in layer order)."""
    zs = [LAYER_Z[k] + 0.1 * j for k, c in enumerate(counts) for j in range(c)]
    return make_scene(
        cfg, [dict(pos=(2.0 + 3.1 * (i % 14), 2.0 + 3.1 * (i // 14), z)) for i, z in enumerate(zs)]
    )


def _host_layers(cfg: Config, z: np.ndarray) -> np.ndarray:
    """``clip(floor(z / field_cell), 0, FZ − 1)`` in float32: the layer harvest reads (§27.5)."""
    layer = np.floor(z.astype(np.float32) / np.float32(cfg.world.field_cell)).astype(np.int64)
    return np.clip(layer, 0, dims(cfg).fz - 1)


def _host_weights(cfg: Config, z: np.ndarray) -> np.ndarray:
    """``w_k = min(1, FZ·p_k)`` of the alive heights ``z``; the floor layer alone when there are none."""
    fz = dims(cfg).fz
    if z.size == 0:
        return np.eye(fz)[0]
    return np.minimum(1.0, fz * np.bincount(_host_layers(cfg, z), minlength=fz) / z.size)


def _host_regen_by_layer(cfg: Config, patch_pos: np.ndarray) -> np.ndarray:
    """(FZ,) float64: the sum of ``floor(regen_max·fertility·V_k)`` over layer ``k``, written independently
    of ``fields`` (torus-Gaussian patches, the vertical profile of §27.5, seasons off)."""
    d, fi, fc, L = dims(cfg), cfg.fields, cfg.world.field_cell, cfg.world.size
    centres = (np.arange(d.fg) + 0.5) * fc

    def torus(dx: np.ndarray) -> np.ndarray:
        dx = dx - L * np.round(dx / L)
        return np.exp(-(dx**2) / (2.0 * fi.patch_sigma**2))

    fert = fi.fertility_base + sum(
        torus(centres[:, None] - p[1]) * torus(centres[None, :] - p[0]) for p in patch_pos.astype(np.float64)
    )
    fert = np.clip(fert, 0.0, 1.0)
    k = np.arange(d.fz)
    v = np.exp(-k * fc / fi.benthic_lambda)
    if fi.photic_amp > 0:
        v = v + fi.photic_amp * np.exp(-(d.fz - 1 - k) * fc / fi.photic_lambda)
    v = np.minimum(1.0, v)
    return np.array([np.floor(fi.regen_max * fert * v[i]).sum() for i in k])


def _host_k(cfg: Config, s: State3) -> float:
    """``K_est`` from the state's alive heights and patches alone."""
    z = to_numpy(s.pos)[to_numpy(s.alive), 2]
    regen = _host_regen_by_layer(cfg, to_numpy(s.patch_pos))
    return float((regen * _host_weights(cfg, z)).sum()) / _c_ref(cfg)


def _engine_regen_by_layer(s: State3, cfg: Config, *args: object) -> np.ndarray:
    rate = to_numpy(fields.regen_rate(s.patch_pos, cfg, *args)).astype(np.int64)
    return rate.reshape(rate.shape[0], -1).sum(axis=1).astype(np.float64)


@pytest.mark.parametrize(
    "counts",
    [
        (5, 0, 0, 0, 0, 0),  # every agent on the floor
        (1, 0, 0, 0, 0, 0),
        (4, 4, 4, 4, 4, 4),  # the column used evenly
        (2, 2, 0, 0, 0, 0),
        (12, 4, 2, 0, 0, 2),  # mixed: 0.6, 0.2, 0.1, 0, 0, 0.1 of the agents
        (0, 0, 0, 0, 0, 6),  # all at the top: the floor layer does not count
        (1, 2, 3, 4, 5, 6),
    ],
)
def test_layer_weights_and_k_est_follow_the_occupancy(micro3d_cfg: Config, counts: tuple[int, ...]) -> None:
    cfg = micro3d_cfg
    s = _scene_by_layer(cfg, counts)
    n = sum(counts)
    want = np.minimum(1.0, 6 * np.array(counts) / n)  # min(1, FZ·p_k)
    got = metrics.layer_occupancy_weights(s, cfg)
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)
    assert got.shape == (6,) and got.dtype == np.float64
    z = to_numpy(s.pos)[to_numpy(s.alive), 2]
    np.testing.assert_array_equal(
        np.bincount(_host_layers(cfg, z), minlength=6), counts
    )  # the scene is as built
    # K_est is the weighted layer sums over c_ref: against the engine's own layer sums (to rounding) ...
    k = metrics.carrying_capacity(s, cfg)
    regen = _engine_regen_by_layer(s, cfg)
    assert k == pytest.approx(float((regen * want).sum()) / _c_ref(cfg), rel=1e-12)
    # ... and against the independent float64 host reference (the floors may differ by a quantum a cell).
    assert k == pytest.approx(_host_k(cfg, s), rel=2e-3)


def test_all_benthic_is_the_floor_layer_and_an_even_column_is_every_layer(micro3d_cfg: Config) -> None:
    cfg = micro3d_cfg
    empty = make_scene(cfg, [])
    benthic, even = _scene_by_layer(cfg, (6, 0, 0, 0, 0, 0)), _scene_by_layer(cfg, (4, 4, 4, 4, 4, 4))
    mixed = _scene_by_layer(cfg, (12, 4, 2, 0, 0, 2))
    regen = _engine_regen_by_layer(empty, cfg)
    k_floor = metrics.carrying_capacity(benthic, cfg)
    assert k_floor == metrics.carrying_capacity(
        empty, cfg
    )  # the benthic limit is the floor-only value, exactly
    assert k_floor == regen[0] / _c_ref(cfg)
    assert metrics.carrying_capacity(even, cfg) == pytest.approx(regen.sum() / _c_ref(cfg), rel=1e-12)
    k_mixed = metrics.carrying_capacity(mixed, cfg)
    assert k_floor < k_mixed < metrics.carrying_capacity(even, cfg)
    assert (
        metrics.carrying_capacity(even, cfg) > 3 * k_floor
    )  # the upper layers hold more than 2 floors' food
    # the top layer alone: the floor layer's food is not counted, only the top layer's
    top = metrics.carrying_capacity(_scene_by_layer(cfg, (0, 0, 0, 0, 0, 6)), cfg)
    assert top == pytest.approx(regen[5] / _c_ref(cfg), rel=1e-12) and top < k_floor


def test_a_few_survivors_saturate_the_weights_and_n_zero_is_a_discontinuity(micro3d_cfg: Config) -> None:
    """With ``n ≤ FZ`` alive, any occupied layer holds at least ``1/n ≥ 1/FZ`` of them, so ``w_k = 1`` there
    (and 0 elsewhere): one survivor per layer gives the plain sum over the layers, the largest ``K_est`` there
    is, and a lone survivor counts its own layer's food alone, the floor layer's included only if it lives
    there. Only with nobody alive does the floor layer count by default, so ``K_est`` is not continuous at
    ``n = 0`` (§27.10): it does not inflate after a total collapse, but it is at its maximum just before."""
    cfg = micro3d_cfg
    regen = _engine_regen_by_layer(make_scene(cfg, []), cfg)
    c_ref = _c_ref(cfg)
    one_per_layer = metrics.carrying_capacity(_scene_by_layer(cfg, (1, 1, 1, 1, 1, 1)), cfg)
    assert one_per_layer == pytest.approx(regen.sum() / c_ref, rel=1e-12)  # the plain sum
    for counts in [
        (1, 0, 0, 0, 0, 0),
        (0, 1, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 1),
        (1, 1, 1, 0, 0, 0),
        (0, 2, 0, 1, 0, 1),
    ]:
        assert sum(counts) <= 6
        occupied = np.array(counts) > 0
        w = metrics.layer_occupancy_weights(_scene_by_layer(cfg, counts), cfg)
        np.testing.assert_array_equal(w, occupied.astype(np.float64))  # 1 where anyone lives, else 0
        k = metrics.carrying_capacity(_scene_by_layer(cfg, counts), cfg)
        assert k == pytest.approx(float(regen[occupied].sum()) / c_ref, rel=1e-12)
    floor_only = metrics.carrying_capacity(make_scene(cfg, []), cfg)
    assert floor_only == pytest.approx(regen[0] / c_ref, rel=1e-12)
    assert (
        one_per_layer > 3 * floor_only
    )  # six survivors in six layers: the maximum, at the edge of extinction
    lone_top = metrics.carrying_capacity(_scene_by_layer(cfg, (0, 0, 0, 0, 0, 1)), cfg)
    assert lone_top < floor_only  # a lone survivor in the top layer: below the empty world's floor-only value


def test_with_nobody_alive_the_floor_layer_counts_and_dead_rows_do_not(micro3d_cfg: Config) -> None:
    cfg = micro3d_cfg
    floor_only = np.eye(6)[0]
    np.testing.assert_array_equal(metrics.layer_occupancy_weights(make_scene(cfg, []), cfg), floor_only)
    spread = _scene_by_layer(cfg, (0, 4, 4, 4, 4, 4))  # upper layers only ...
    dead = spread._replace(alive=mx.zeros_like(spread.alive))  # ... and all of them dead
    np.testing.assert_array_equal(metrics.layer_occupancy_weights(dead, cfg), floor_only)
    assert metrics.carrying_capacity(dead, cfg) == metrics.carrying_capacity(make_scene(cfg, []), cfg)
    # three living benthic agents and three dead rows left in the upper layers: only the living are counted
    s = _scene_by_layer(cfg, (3, 1, 1, 1, 0, 0))
    alive = np.array(to_numpy(s.alive))
    alive[3:6] = False
    s = s._replace(alive=mx.array(alive))
    np.testing.assert_array_equal(metrics.layer_occupancy_weights(s, cfg), floor_only)
    assert metrics.carrying_capacity(s, cfg) == metrics.carrying_capacity(make_scene(cfg, []), cfg)


def test_the_layer_of_a_height_is_the_one_harvest_reads(micro3d_cfg: Config) -> None:
    """Heights on and around the layer edges, the floor rest height and the surface bound: the weights use
    the layer of ``fields.cell_index`` (float32 ``floor(z / field_cell)``, clamped to the lattice)."""
    cfg = micro3d_cfg
    d = dims(cfg)
    below4 = float(np.nextafter(np.float32(4.0), np.float32(0.0)))  # the largest float32 under 4.0
    zs = [d.z_lo, below4, 4.0, 8.0, 11.999, 12.0, 20.0, d.z_hi]
    s = make_scene(cfg, [dict(pos=(3.0 * (i + 1), 5.0, z)) for i, z in enumerate(zs)])
    z32 = to_numpy(s.pos)[:, 2][: len(zs)]
    layers = to_numpy(fields.cell_index(s.pos, cfg))[: len(zs)] // (d.fg * d.fg)
    np.testing.assert_array_equal(layers, _host_layers(cfg, z32))
    assert layers.tolist() == [0, 0, 1, 2, 2, 3, 5, 5]  # 4.0 is in layer 1, the largest float32 below it in 0
    counts = np.bincount(layers, minlength=6)
    np.testing.assert_allclose(
        metrics.layer_occupancy_weights(s, cfg), np.minimum(1.0, 6 * counts / len(zs)), rtol=0, atol=1e-12
    )


def test_k_est_follows_the_live_environment_by_occupancy(micro3d_cfg: Config) -> None:
    """The director's call: every layer's live regeneration (seasonal at ``step``, scaled by the levers),
    weighted by the same occupancy."""
    cfg = micro3d_cfg
    s = _scene_by_layer(cfg, (12, 4, 2, 0, 0, 2))
    boosted = s._replace(env_regen_mult=mx.array(2.0), env_season_amp=mx.array(0.5, dtype=mx.float32))
    boosted = boosted._replace(env_season_period=mx.array(40, dtype=mx.int32))
    step = mx.array(10, dtype=mx.int32)
    w = metrics.layer_occupancy_weights(boosted, cfg)
    live = metrics.carrying_capacity(boosted, cfg, step, boosted.env)
    ref = float((_engine_regen_by_layer(boosted, cfg, step, boosted.env) * w).sum()) / _c_ref(cfg)
    assert live == pytest.approx(ref, rel=1e-12)
    base = metrics.carrying_capacity(s, cfg)
    assert live > 1.5 * base  # the levers double the regeneration (and the season adds to it at step 10)
    assert metrics.carrying_capacity(boosted, cfg) == base  # the plain call ignores the environment
    # the occupancy part is the same with and without the levers: the ratio is the ratio of the regenerations
    plain = _engine_regen_by_layer(s, cfg)
    assert live / base == pytest.approx(
        float((_engine_regen_by_layer(boosted, cfg, step, boosted.env) * w).sum()) / float((plain * w).sum()),
        rel=1e-12,
    )


def test_a_2d_state_is_unaffected_by_the_occupancy_rule(tiny_cfg: Config) -> None:
    """The rule is a static 3D branch: a 2D ``K_est`` is the lattice's regeneration over ``c_ref`` whatever
    the agents do (§10.10)."""
    s = init_state(tiny_cfg)
    rate = to_numpy(fields.regen_rate(s.patch_pos, tiny_cfg)).astype(np.int64)
    c_ref = tiny_cfg.energy.basal_cost + metrics.founder_brain_cost(tiny_cfg) + layout.C_REF_MARGIN
    assert metrics.carrying_capacity(s, tiny_cfg) == int(rate.sum()) / c_ref
    emptied = s._replace(alive=mx.zeros_like(s.alive))
    assert metrics.carrying_capacity(emptied, tiny_cfg) == metrics.carrying_capacity(s, tiny_cfg)


def _lazy_64k3d(seed: int) -> tuple[Config, State3]:
    """The 64k3d initial state as a lazy graph: only the alive mask, the positions and the patches are ever
    evaluated, so the (65 536, 116, 32) weights are never materialised (as ``test_state.test_schema_*``)."""
    cfg = load_config(preset="64k3d", overrides=[f"run.seed={seed}"])
    return cfg, _initial_state(cfg)


def test_k_est_band_of_the_64k3d_initial_state() -> None:
    """The sizing of ``regen_max`` (DESIGN §27.13): for the founders uniform in the volume, for all of them on
    the floor and for the mostly benthic steady state the preset targets (70% in the floor layer, the rest
    spread over the other layers), K_est stays in a band around 61 000, 17 800 and 33 000 for any seed."""
    for seed in range(3):
        cfg, s = _lazy_64k3d(seed)
        d = dims(cfg)
        n0 = cfg.population.n_init
        pos = np.array(to_numpy(s.pos))
        assert d.fz == 20 and pos.shape == (65536, 3) and int(to_numpy(s.alive).sum()) == n0
        k_uniform = metrics.carrying_capacity(s, cfg)
        assert 55_000 < k_uniform < 68_000, (seed, k_uniform)

        rng = np.random.default_rng(seed)
        n_floor = int(0.7 * n0)
        z = np.array(pos[:, 2])
        z[:n_floor] = d.z_lo  # 70% in the floor layer ...
        z[n_floor:n0] = rng.uniform(cfg.world.field_cell, d.z_hi, n0 - n_floor)  # ... the rest in the others
        mixed = s._replace(pos=mx.array(np.stack([pos[:, 0], pos[:, 1], z], axis=1)))
        k_mixed = metrics.carrying_capacity(mixed, cfg)
        assert 30_000 < k_mixed < 36_500, (seed, k_mixed)

        z[:n0] = d.z_lo
        benthic = s._replace(pos=mx.array(np.stack([pos[:, 0], pos[:, 1], z], axis=1)))
        k_floor = metrics.carrying_capacity(benthic, cfg)
        assert 16_000 < k_floor < 19_500, (seed, k_floor)
        assert k_floor < k_mixed < k_uniform
        assert 0.4 < n0 / k_mixed < 0.6  # n_init is about half of the benthic steady state's K_est
