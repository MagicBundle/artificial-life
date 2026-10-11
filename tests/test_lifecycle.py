"""§10.7, §12: deaths, lane allocation, reproduction, the pending flush and immigration (WP4 unit tests)."""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from alife import bonds, brain, lifecycle, ops, rng
from alife.config import Config, apply_overrides, dims
from alife.layout import T
from alife.lifecycle import LaneKind
from alife.rng import Stream
from alife.state import AGENT_FIELDS, R_HARD, DeathCause, State, age, to_numpy, total_energy
from helpers import ON_GPU, make_scene

STARVED, KILLED, AGED = DeathCause.STARVED, DeathCause.KILLED, DeathCause.AGED


def _np(a: mx.array) -> np.ndarray:
    return to_numpy(a)


def _total(s: State, cfg: Config) -> int:
    return int(_np(total_energy(s.energy, s.alive, s.resource, cfg)))


@pytest.fixture(scope="module")
def imm_cfg(micro_cfg: Config) -> Config:
    return apply_overrides(micro_cfg, ["population.immigration_floor=40"])


# --- flush_pending ------------------------------------------------------------------------------------------


def test_flush_pending_writes_rows_without_mutating_inputs(micro_cfg: Config) -> None:
    d = dims(micro_cfg)
    rs = np.random.default_rng(0)
    w1 = mx.array(rs.normal(size=(d.n, d.w1_rows, d.h)).astype(np.float32))
    w2 = mx.array(rs.normal(size=(d.n, d.w2_rows, 35)).astype(np.float32))
    slot = mx.array(rs.permutation(d.n)[: d.m].astype(np.int32))
    p1 = mx.array(rs.normal(size=(d.m, d.w1_rows, d.h)).astype(np.float32))
    p2 = mx.array(rs.normal(size=(d.m, d.w2_rows, 35)).astype(np.float32))
    w1_before, w2_before = _np(w1), _np(w2)
    o1, o2 = lifecycle.flush_pending(w1, w2, slot, p1, p2)
    e1, e2 = w1_before.copy(), w2_before.copy()
    e1[_np(slot)], e2[_np(slot)] = _np(p1), _np(p2)
    assert np.array_equal(_np(o1), e1) and np.array_equal(_np(o2), e2)
    assert np.array_equal(_np(w1), w1_before) and np.array_equal(_np(w2), w2_before)  # F2: inputs untouched


# --- deaths -------------------------------------------------------------------------------------------------


def _death_inputs(cfg: Config, rows: list[dict], bonds_: list[tuple[int, int]], resource: np.ndarray):
    """A scene plus per-row overrides of solvent/E/lost/age/fcell (dict keys), as numpy arrays."""
    d = dims(cfg)
    alive_rows = [r for r in rows if r.get("alive", True)]
    s = make_scene(cfg, [{"row": r["row"]} for r in alive_rows], bonds=bonds_, resource=resource)
    n = d.n
    solvent, e = np.zeros(n, bool), np.zeros(n, np.int32)
    lost, ag, fcell = np.zeros(n, np.int32), np.zeros(n, np.int32), np.zeros(n, np.int32)
    for r in rows:
        i = r["row"]
        solvent[i] = r.get("solvent", True) and r.get("alive", True)
        e[i] = r.get("E", 0)
        lost[i] = r.get("lost", 0)
        ag[i] = r.get("age", 0)
        fcell[i] = r.get("cell", 0)
    return s, solvent, e, lost, ag, fcell


def _run_deaths(cfg: Config, s: State, solvent, e, lost, ag, fcell, cat_mask=None) -> lifecycle.Deaths:
    return lifecycle.deaths(
        s.alive,
        mx.array(solvent),
        mx.array(e),
        mx.array(lost),
        mx.array(ag),
        s.uid,
        s.generation,
        s.clade,
        s.partner,
        s.rev,
        s.resource,
        mx.array(fcell),
        cfg,
        cat_mask=None if cat_mask is None else mx.array(cat_mask),
    )


def test_deaths_causes_corpses_cap_and_bonds(micro_cfg: Config) -> None:
    cfg, d = micro_cfg, dims(micro_cfg)
    S, max_age = cfg.energy.structure, cfg.population.max_age
    assert max_age > 0
    fg2 = d.fg * d.fg
    resource = np.random.default_rng(1).integers(0, 8000, (d.fg, d.fg)).astype(np.int32)
    resource.reshape(-1)[7] = R_HARD - 100  # a corpse here overflows the hard cap
    rows = [
        dict(row=0, E=500, age=5, cell=3),  # survives
        dict(row=1, solvent=False, E=300, cell=5),  # insolvent with bond inflow: STARVED, corpse 300 + S
        dict(row=2, solvent=False, E=0, lost=100, cell=5),  # insolvency wins over KILLED
        dict(row=3, E=0, lost=50, cell=7),  # KILLED, into the near-full cell
        dict(row=4, E=0, lost=0, cell=9),  # STARVED
        dict(row=5, E=700, age=max_age, cell=9),  # AGED
        dict(row=6, E=0, lost=10, age=max_age, cell=11),  # KILLED wins over AGED
        dict(row=7, E=900, lost=400, cell=3),  # drained victim rescued by inflow: survives
        dict(row=9, E=1, age=max_age - 1, cell=fg2 - 1),  # survives
        dict(row=8, alive=False),  # already dead: not a death this step
    ]
    bonds_ = [(0, 1), (0, 7), (7, 3), (9, 5), (2, 4)]
    s, solvent, e, lost, ag, fcell = _death_inputs(cfg, rows, bonds_, resource)
    before = (
        int(e.astype(np.int64).sum()) + S * int(_np(s.alive).sum()) + int(resource.astype(np.int64).sum())
    )
    dt = _run_deaths(cfg, s, solvent, e, lost, ag, fcell)

    dead = np.zeros(d.n, bool)
    dead[[1, 2, 3, 4, 5, 6]] = True
    assert np.array_equal(_np(dt.dead), dead)
    assert np.array_equal(_np(dt.survivor), _np(s.alive) & ~dead)
    cause = np.full(d.n, -1, np.int32)
    cause[[1, 2, 3, 4, 5, 6]] = [STARVED, STARVED, KILLED, STARVED, AGED, KILLED]
    assert np.array_equal(_np(dt.cause), cause)

    corpse = np.where(dead, e + S, 0)
    assert np.array_equal(_np(dt.corpse), corpse) and _np(dt.corpse)[1] == 300 + S
    r = resource.reshape(-1).astype(np.int64)
    np.add.at(r, fcell[dead], corpse[dead])
    over = np.maximum(r - R_HARD, 0)
    assert over[7] == S - 100 and over.sum() == S - 100
    assert np.array_equal(_np(dt.r_overflow).reshape(-1), over)
    assert np.array_equal(_np(dt.resource).reshape(-1), r - over)
    assert dt.resource.shape == (d.fg, d.fg) and dt.resource.dtype == mx.int32

    # Dead rows canonical; survivors untouched.
    assert np.array_equal(_np(dt.energy), np.where(dead, 0, e))
    assert np.array_equal(_np(dt.uid), np.where(dead, -1, _np(s.uid)))
    # Bonds with a dead end are dropped on both sides; 0–7 survives with its slots.
    partner, rev = _np(dt.partner), _np(dt.rev)
    assert np.all(partner[dead] == -1) and np.all(rev[dead] == -1)
    assert sorted(partner[0][partner[0] >= 0]) == [7] and sorted(partner[7][partner[7] >= 0]) == [0]
    assert np.all(partner[9] == -1)
    assert bool(_np(bonds.symmetric_ok(dt.partner, dt.rev, dt.survivor)))

    after = int(_np(total_energy(dt.energy, dt.survivor, dt.resource, cfg)))
    assert before == after + int(over.sum())  # corpse is internal; the cap is the only sink

    rec = dt.records
    assert all(a.shape == (d.dl,) for a in rec)
    valid = _np(rec.valid)
    assert valid.sum() == 6 and valid[:6].all()  # compacted to a prefix, in row order
    assert np.array_equal(_np(rec.uid)[:6], _np(s.uid)[[1, 2, 3, 4, 5, 6]])
    assert np.array_equal(_np(rec.cause)[:6], cause[[1, 2, 3, 4, 5, 6]])
    assert np.array_equal(_np(rec.age)[:6], ag[[1, 2, 3, 4, 5, 6]])
    assert np.all(_np(rec.uid)[6:] == -1)


def test_catastrophe_deaths_are_excess_only_and_keep_cause_precedence(micro_cfg: Config) -> None:
    cfg, d = micro_cfg, dims(micro_cfg)
    S, max_age = cfg.energy.structure, cfg.population.max_age
    rows = [
        dict(row=0, E=500, cell=3),  # in the mask, would survive: CATASTROPHE
        dict(row=1, E=500, cell=3),  # not in the mask: survives
        dict(row=2, solvent=False, E=300, cell=5),  # in the mask but insolvent: STARVED
        dict(row=3, E=0, lost=50, cell=7),  # in the mask but drained by an attack: KILLED
        dict(row=4, E=0, lost=0, cell=9),  # in the mask, E == 0: STARVED
        dict(row=5, E=700, age=max_age, cell=9),  # in the mask, aged: AGED
        dict(row=6, E=800, cell=11),  # in the mask: CATASTROPHE, corpse E + S
        dict(row=7, alive=False),  # in the mask but already dead: not a death this step
    ]
    resource = np.full((d.fg, d.fg), 100, np.int32)
    s, solvent, e, lost, ag, fcell = _death_inputs(cfg, rows, [(0, 6), (6, 1)], resource)
    mask = np.zeros(d.n, bool)
    mask[[0, 2, 3, 4, 5, 6, 7]] = True
    dt = _run_deaths(cfg, s, solvent, e, lost, ag, fcell, cat_mask=mask)
    CAT = DeathCause.CATASTROPHE
    cause = np.full(d.n, -1, np.int32)
    cause[[0, 2, 3, 4, 5, 6]] = [CAT, STARVED, KILLED, STARVED, AGED, CAT]
    assert np.array_equal(_np(dt.cause), cause)
    dead = cause >= 0
    assert np.array_equal(_np(dt.dead), dead) and np.array_equal(_np(dt.survivor), _np(s.alive) & ~dead)
    assert _np(dt.corpse)[6] == 800 + S and _np(dt.corpse)[0] == 500 + S  # the existing corpse path
    assert np.array_equal(_np(dt.energy), np.where(dead, 0, e)) and (_np(dt.uid)[dead] == -1).all()
    assert (_np(dt.partner)[1] == -1).all()  # bonds to the dead are dropped
    assert bool(_np(bonds.symmetric_ok(dt.partner, dt.rev, dt.survivor)))
    rec = dt.records
    assert np.array_equal(_np(rec.cause)[:6], cause[[0, 2, 3, 4, 5, 6]])
    # The same inputs without a mask: the cat_mask rows that survive on their own keep surviving.
    plain = _run_deaths(cfg, s, solvent, e, lost, ag, fcell)
    assert np.array_equal(_np(plain.cause), np.where(np.isin(np.arange(d.n), [0, 6]), -1, cause))
    # An all-False mask changes nothing.
    none = _run_deaths(cfg, s, solvent, e, lost, ag, fcell, cat_mask=np.zeros(d.n, bool))
    assert np.array_equal(_np(none.cause), _np(plain.cause))


def test_deaths_records_truncate_and_max_age_off(micro_cfg: Config) -> None:
    d = dims(micro_cfg)
    n_dead = d.dl + 4
    rows = [dict(row=i, solvent=False, E=i, age=i) for i in range(n_dead)]
    rows += [dict(row=n_dead, E=50, age=10**6)]  # very old
    s, solvent, e, lost, ag, fcell = _death_inputs(micro_cfg, rows, [], np.zeros((d.fg, d.fg), np.int32))
    s = s._replace(clade=mx.arange(d.n, dtype=mx.int32), generation=mx.arange(d.n, dtype=mx.int32) * 2)
    dt = _run_deaths(micro_cfg, s, solvent, e, lost, ag, fcell)
    assert int(_np(dt.dead).sum()) == n_dead + 1  # the old agent dies of age
    rec = dt.records
    assert _np(rec.valid).all()
    assert np.array_equal(_np(rec.uid), _np(s.uid)[: d.dl])
    assert np.array_equal(_np(rec.clade), np.arange(d.dl)) and np.array_equal(
        _np(rec.generation), 2 * np.arange(d.dl)
    )
    assert max(int(_np(dt.dead).sum()) - d.dl, 0) == 5  # DEATHS_UNLOGGED as step.finish computes it

    no_age = apply_overrides(micro_cfg, ["population.max_age=0"])
    dt = _run_deaths(no_age, s, solvent, e, lost, ag, fcell)
    assert bool(_np(dt.survivor)[n_dead])


# --- allocate_lanes -----------------------------------------------------------------------------------------


def _lanes_ref(req: np.ndarray, survivor: np.ndarray, key: mx.array, cfg: Config) -> dict[str, np.ndarray]:
    d = dims(cfg)
    perm = _np(mx.random.permutation(d.n, key=key)).astype(np.int64)
    order = [int(i) for i in perm if req[i]]
    slots = np.flatnonzero(~survivor)[: d.m]
    n_surv = int(survivor.sum())
    room = d.n_cap - n_surv
    n_birth = min(len(order), d.m, room)
    pop = cfg.population
    n_imm = 0
    if pop.immigration_floor > 0:
        n_imm = min(
            pop.immigration_floor - n_surv - n_birth, pop.immigrants_per_step, d.m - n_birth, room - n_birth
        )
        n_imm = min(max(n_imm, 0), d.m)
    kind = np.full(d.m, LaneKind.PAD, np.int32)
    kind[:n_birth] = LaneKind.BIRTH
    kind[n_birth : n_birth + n_imm] = LaneKind.IMMIGRANT
    parent = np.zeros(d.m, np.int32)
    parent[:n_birth] = order[:n_birth]
    return dict(parent=parent, slot=slots, kind=kind, n_req=len(order), n_birth=n_birth, n_imm=n_imm)


def test_allocate_lanes_properties(micro_cfg: Config, imm_cfg: Config) -> None:
    rs = np.random.default_rng(2)
    for trial in range(320):
        cfg = imm_cfg if trial % 4 == 3 else micro_cfg
        d = dims(cfg)
        regime = trial % 5
        n_surv = {0: d.n_cap, 1: 0, 2: d.n_cap - 1}.get(regime, int(rs.integers(0, d.n_cap + 1)))
        survivor = np.zeros(d.n, bool)
        survivor[rs.choice(d.n, n_surv, replace=False)] = True
        density = 0.0 if trial % 7 == 0 else rs.uniform(0, 1)
        req = survivor & (rs.uniform(size=d.n) < density)
        key = mx.random.key(trial)
        lanes = lifecycle.allocate_lanes(mx.array(req), mx.array(survivor), key, cfg)
        got = {k: _np(v) for k, v in lanes._asdict().items()}
        ref = _lanes_ref(req, survivor, key, cfg)
        for k, v in ref.items():
            assert np.array_equal(got[k], v), (trial, k)
        assert all(got[k].dtype == np.int32 for k in got)

        slot, kind, parent = got["slot"], got["kind"], got["parent"]
        nb, ni = int(got["n_birth"]), int(got["n_imm"])
        assert (
            np.unique(slot).size == d.m and not survivor[slot].any()
        )  # distinct dead rows, padding included
        births = parent[kind == LaneKind.BIRTH]
        assert np.unique(births).size == nb and req[births].all()  # unique eligible parents
        assert np.all(kind[:nb] == LaneKind.BIRTH) and not np.any(kind[nb:] == LaneKind.BIRTH)  # prefix
        assert np.all(parent[kind != LaneKind.BIRTH] == 0)  # padding uses row 0 whatever the requests (R20)
        assert n_surv + nb + ni <= d.n_cap
        if not req.any() or n_surv == d.n_cap:
            assert nb == 0
        if n_surv == d.n_cap:
            assert ni == 0 and np.all(kind == LaneKind.PAD)


def test_allocate_lanes_fairness_chi2(micro_cfg: Config) -> None:
    d = dims(micro_cfg)
    survivor = np.zeros(d.n, bool)
    survivor[:32] = True
    req = np.zeros(d.n, bool)
    req[:16] = True  # 16 requesters for M = 8 lanes: each wins with probability 1/2
    counts = np.zeros(d.n, np.int64)
    trials = 400
    for seed in range(trials):
        lanes = lifecycle.allocate_lanes(
            mx.array(req), mx.array(survivor), mx.random.key(1000 + seed), micro_cfg
        )
        np.add.at(counts, _np(lanes.parent), 1)
    c = counts[:16]
    assert c.sum() == trials * d.m and counts[16:].sum() == 0
    chi2 = float((((c - trials / 2) ** 2) / (trials / 4)).sum())  # Var = n·p·(1 − p) per requester
    assert chi2 < 37.7  # χ²(15), p = 0.001


# --- reproduce ----------------------------------------------------------------------------------------------


def _repro_scene(cfg: Config, rs: np.random.Generator) -> tuple[State, np.ndarray, np.ndarray]:
    d, mu = dims(cfg), cfg.mutation
    clip = cfg.brain.weight_clip

    def weights() -> dict:
        return dict(
            W1=rs.uniform(-clip, clip, (d.w1_rows, d.h)).astype(np.float32),
            W2=rs.uniform(-clip, clip, (d.w2_rows, 35)).astype(np.float32),
        )

    parents = [
        dict(
            pos=(0.5, 47.8),
            heading=3.0,
            energy=40_000,
            offspring_frac=mu.frac_min,
            mut_sigma=mu.sigma_max,
            mut_rate=mu.rate_max,
            tag=(1.0, -1.0, 0.99, -0.99),
            clade=3,
            generation=4,
            **weights(),
        ),
        dict(
            pos=(10.0, 20.0),
            heading=-3.1,
            energy=55_555,
            offspring_frac=0.37,
            mut_sigma=mu.sigma_min,
            mut_rate=mu.rate_min,
            clade=5,
            generation=0,
            **weights(),
        ),
        dict(
            pos=(47.9, 0.1),
            heading=0.0,
            energy=cfg.energy.e_max,
            offspring_frac=mu.frac_max,
            clade=1,
            **weights(),
        ),
        dict(
            pos=(30.0, 30.0),
            heading=1.0,
            energy=123_457,
            offspring_frac=0.5,
            clade=7,
            generation=9,
            **weights(),
        ),
    ]
    others = [dict(pos=(20.0 + i, 5.0), energy=1_000) for i in range(4)]
    agents = [dict(a, birth_step=-50) for a in parents + others]
    s = make_scene(cfg, agents, bonds=[(1, 4), (1, 5), (1, 6), (1, 7), (3, 4)], step=7, next_uid=100)
    req = np.zeros(d.n, bool)
    req[:4] = True
    child_bond = np.zeros(d.n, bool)
    child_bond[[0, 1, 3]] = True  # row 1 has no free slot; row 2 does not want one
    return s, req, child_bond


def _reproduce(s: State, req: np.ndarray, child_bond: np.ndarray, cfg: Config, seed: int = 3):
    _, ks = rng.step_keys(mx.random.key(seed))
    lanes = lifecycle.allocate_lanes(mx.array(req), s.alive, ks[Stream.BIRTH_PRIO], cfg)
    return ks, lanes, lifecycle.reproduce(s, lanes, mx.array(child_bond), ks, cfg)


def _gene_ref(s: State, p: np.ndarray, key: mx.array, cfg: Config) -> tuple[np.ndarray, ...]:
    mu, m, f32 = cfg.mutation, p.shape[0], np.float32
    k = mx.random.split(key, 4)
    n1, n2 = _np(mx.random.normal((m,), key=k[0])), _np(mx.random.normal((m,), key=k[1]))
    n3, n4 = _np(mx.random.normal((m, T), key=k[2])), _np(mx.random.normal((m,), key=k[3]))

    def logit(x: np.ndarray) -> np.ndarray:
        return np.log(x / (f32(1) - x))

    def sig(x: np.ndarray) -> np.ndarray:
        return f32(1) / (f32(1) + np.exp(-x))

    sigma = np.clip(
        _np(s.mut_sigma)[p] * np.exp(f32(mu.tau_sigma) * n1), f32(mu.sigma_min), f32(mu.sigma_max)
    )
    rate = np.clip(sig(logit(_np(s.mut_rate)[p]) + f32(mu.tau_rate) * n2), f32(mu.rate_min), f32(mu.rate_max))
    tag = np.clip(_np(s.tag)[p] + f32(mu.tag_sigma) * n3, -1, 1)
    frac = np.clip(
        sig(logit(_np(s.offspring_frac)[p]) + f32(mu.tau_frac) * n4), f32(mu.frac_min), f32(mu.frac_max)
    )
    return sigma, rate, tag, frac


def _min_image(v: np.ndarray, L: float) -> np.ndarray:
    return v - L * np.round(v / L)


def test_reproduce_rows_energy_lineage_placement_and_bonds(micro_cfg: Config) -> None:
    cfg, d = micro_cfg, dims(micro_cfg)
    rs = np.random.default_rng(4)
    s, req, child_bond = _repro_scene(cfg, rs)
    ks, lanes, rp = _reproduce(s, req, child_bond, cfg)
    ns = rp.state
    kind, parent, slot = _np(lanes.kind), _np(lanes.parent), _np(lanes.slot)
    assert np.array_equal(kind, [1, 1, 1, 1, 0, 0, 0, 0])
    assert np.array_equal(slot, np.arange(8, 16))  # lowest-index dead rows
    assert sorted(parent[:4]) == [0, 1, 2, 3] and np.all(parent[4:] == 0)
    p, c = parent[:4], slot[:4]
    S, bc = cfg.energy.structure, cfg.repro.birth_cost
    a0, a1 = {k: _np(v) for k, v in s._asdict().items()}, {k: _np(v) for k, v in ns._asdict().items()}

    # Energy (§10.7).
    dp = a0["energy"][p] - bc - S
    fq = np.floor(a0["offspring_frac"][p] * np.float32(256) + np.float32(0.5)).astype(np.int64)
    endow = (dp * fq) >> 8
    assert np.all((fq >= 26) & (fq <= 230))
    assert np.array_equal(_np(rp.endow), np.concatenate([endow, np.zeros(4, np.int64)]))
    assert np.array_equal(a1["energy"][p], dp - endow) and np.all(dp - endow >= 1)
    assert np.array_equal(a1["energy"][c], endow) and np.all(endow >= 1)
    assert _total(ns, cfg) - _total(s, cfg) == -bc * 4  # HEAT_BIRTH; structure and endowment are internal

    # uids and lineage (§12.3).
    assert np.array_equal(a1["uid"][c], 100 + np.arange(4)) and int(a1["next_uid"]) == 104
    assert np.array_equal(a1["parent_uid"][c], a0["uid"][p])
    assert np.array_equal(a1["clade"][c], a0["clade"][p])
    assert np.array_equal(a1["generation"][c], a0["generation"][p] + 1)
    assert np.all(a1["birth_step"][c] == 7) and a1["alive"][c].all()
    for name in ("vel", "h", "msg", "last_loss", "last_harvest", "last_inflow"):
        assert not a1[name][c].any(), name

    # Placement: distance 2.0 from the parent, facing away from it.
    off = _min_image(a1["pos"][c].astype(np.float64) - a0["pos"][p], cfg.world.size)
    assert np.allclose(np.linalg.norm(off, axis=1), cfg.repro.offspring_distance, atol=1e-4)
    hc = a1["heading"][c].astype(np.float64)
    assert np.allclose(np.stack([np.cos(hc), np.sin(hc)], 1) * 2.0, off, atol=1e-4)
    hp = a0["heading"][p].astype(np.float64)
    assert np.all((np.cos(hp) * off[:, 0] + np.sin(hp) * off[:, 1]) < -2.0 * math.cos(0.5) + 1e-4)
    assert np.all((a1["pos"][c] >= 0) & (a1["pos"][c] < cfg.world.size))
    assert np.all((hc >= -math.pi) & (hc < math.pi))

    # Genes (§12.4) and weights (§6), all M lanes.
    sigma, rate, tag, frac = _gene_ref(s, parent, ks[Stream.MUT_GENES], cfg)
    np.testing.assert_allclose(a1["mut_sigma"][c], sigma[:4], rtol=1e-5)
    np.testing.assert_allclose(a1["mut_rate"][c], rate[:4], rtol=1e-5)
    np.testing.assert_allclose(a1["tag"][c], tag[:4], atol=1e-6)
    np.testing.assert_allclose(a1["offspring_frac"][c], frac[:4], rtol=1e-5)
    w1, w2 = brain.mutate_rows(
        s.W1[lanes.parent], s.W2[lanes.parent], ns.mut_sigma[lanes.slot], ns.mut_rate[lanes.slot],
        ks[Stream.MUT_W1], ks[Stream.MUT_W2], cfg,
    )  # fmt: skip
    assert np.array_equal(a1["pend_W1"][:4], _np(w1)[:4]) and np.array_equal(a1["pend_W2"][:4], _np(w2)[:4])
    assert np.array_equal(a1["pend_slot"], slot)
    assert np.array_equal(a1["brain_cost"][c], _np(brain.l1_cost(ns.pend_W1, ns.pend_W2, cfg))[:4])
    clip = np.float32(cfg.brain.weight_clip)
    assert np.abs(a1["pend_W1"]).max() <= clip and np.abs(a1["pend_W2"]).max() <= clip
    assert np.array_equal(a1["W1"], a0["W1"]) and np.array_equal(a1["W2"], a0["W2"])  # written at the next S0

    # Child bonds (§8.4): rows 0 and 3 bond; row 1 is full; row 2 does not want one.
    child = dict(zip(p.tolist(), c.tolist(), strict=True))
    assert int(_np(rp.n_child_bonds)) == 2
    for par, ff in ((0, 0), (3, 1)):
        ch = child[par]
        assert a1["partner"][par, ff] == ch and a1["rev"][par, ff] == 0
        assert a1["partner"][ch, 0] == par and a1["rev"][ch, 0] == ff and np.all(a1["partner"][ch, 1:] == -1)
    for par in (1, 2):
        assert np.all(a1["partner"][child[par]] == -1)
        assert child[par] not in a1["partner"][par]
    assert bool(_np(bonds.symmetric_ok(ns.partner, ns.rev, ns.alive)))

    # Records.
    rec = {k: _np(v) for k, v in rp.records._asdict().items()}
    assert np.array_equal(rec["valid"], kind > 0)
    assert np.array_equal(rec["uid"], [100, 101, 102, 103, -1, -1, -1, -1])
    assert np.array_equal(rec["parent_uid"][:4], a0["uid"][p]) and np.all(rec["parent_uid"][4:] == -1)
    assert np.array_equal(rec["slot"], np.concatenate([c, [-1] * 4]))
    assert np.array_equal(rec["clade"][:4], a0["clade"][p]) and np.array_equal(
        rec["generation"][:4], a0["generation"][p] + 1
    )

    # Every other row is untouched, apart from the parents' energy and bonds.
    other = np.ones(d.n, bool)
    other[c] = False
    for name in AGENT_FIELDS:
        x0, x1 = a0[name], a1[name]
        rows = other.copy()
        if name in ("energy", "partner", "rev"):
            rows[p] = False
        assert np.array_equal(x0[rows], x1[rows]), name


def test_reproduce_gene_and_weight_bounds_under_extremes(micro_cfg: Config) -> None:
    for cfg in (micro_cfg, apply_overrides(micro_cfg, ["brain.weight_dtype=float16"])):
        d, mu, f32 = dims(cfg), cfg.mutation, np.float32
        rs = np.random.default_rng(5)
        clip = cfg.brain.weight_clip
        lo = dict(mut_sigma=mu.sigma_min, mut_rate=mu.rate_min, offspring_frac=mu.frac_min, tag=(-1.0,) * T)
        hi = dict(mut_sigma=mu.sigma_max, mut_rate=mu.rate_max, offspring_frac=mu.frac_max, tag=(1.0,) * T)
        agents = [
            dict(lo if i % 2 else hi, energy=cfg.energy.e_max, W1=np.full((d.w1_rows, d.h), clip * (-1) ** i))
            for i in range(d.m)
        ]
        s = make_scene(cfg, agents, step=3)
        req = np.asarray(_np(s.alive))
        for seed in range(12):
            _, _, rp = _reproduce(s, req, np.zeros(d.n, bool), cfg, seed=seed + int(rs.integers(1 << 20)))
            ns = rp.state
            c = _np(ns.pend_slot)
            assert ns.pend_W1.dtype == d.wdtype
            assert np.all(
                (_np(ns.mut_sigma)[c] >= f32(mu.sigma_min)) & (_np(ns.mut_sigma)[c] <= f32(mu.sigma_max))
            )
            assert np.all(
                (_np(ns.mut_rate)[c] >= f32(mu.rate_min)) & (_np(ns.mut_rate)[c] <= f32(mu.rate_max))
            )
            fr = _np(ns.offspring_frac)[c]
            assert np.all((fr >= f32(mu.frac_min)) & (fr <= f32(mu.frac_max)))
            assert np.all(np.abs(_np(ns.tag)[c]) <= 1.0)
            assert np.abs(_np(ns.pend_W1)).max() <= clip and np.abs(_np(ns.pend_W2)).max() <= clip
            stored = brain.l1_cost(ns.pend_W1, ns.pend_W2, cfg)
            assert np.array_equal(_np(ns.brain_cost)[c], _np(stored))


def test_full_population_pads_into_distinct_dead_rows(micro_cfg: Config) -> None:
    d = dims(micro_cfg)
    s = make_scene(micro_cfg, [dict(energy=500_000, birth_step=-50) for _ in range(d.n_cap)], step=1)
    req = np.asarray(_np(s.alive))
    _, lanes, rp = _reproduce(s, req, req, micro_cfg)
    assert int(_np(lanes.n_birth)) == 0 and np.all(_np(lanes.kind) == LaneKind.PAD)
    assert np.all(_np(lanes.parent) == 0)
    slot = _np(lanes.slot)
    assert np.array_equal(np.sort(slot), np.arange(d.n_cap, d.n))
    ns = rp.state
    assert np.array_equal(_np(ns.alive), _np(s.alive)) and np.array_equal(_np(ns.energy), _np(s.energy))
    assert int(_np(ns.next_uid)) == int(_np(s.next_uid)) and not _np(rp.records.valid).any()
    assert np.array_equal(_np(ns.pend_slot), slot)


def test_child_row_completeness(micro_cfg: Config) -> None:
    """A dead row full of sentinel garbage becomes a clean newborn after reproduce and the next S0 flush."""
    cfg, d = micro_cfg, dims(micro_cfg)
    s = make_scene(cfg, [dict(energy=90_000, pos=(5.0, 5.0), birth_step=-40)], step=2)
    target = 1  # the lowest dead row
    sentinel = {
        "uid": 987_654_321, "parent_uid": 55_555, "clade": 77, "generation": 999, "birth_step": 4242,
        "energy": 31_337, "pos": 999.0, "vel": 9.5, "heading": 99.0, "h": 9.0, "msg": 9.0,
        "last_loss": 31_337, "last_harvest": 31_337, "last_inflow": 31_337, "partner": 33, "rev": 3,
        "tag": 9.0, "mut_sigma": 9.0, "mut_rate": 9.0, "offspring_frac": 9.0, "brain_cost": 31_337,
        "W1": 3.25, "W2": 3.25,
    }  # fmt: skip
    assert set(sentinel) == set(AGENT_FIELDS) - {"alive"}
    garbage = {}
    for name, v in sentinel.items():
        a = _np(getattr(s, name)).copy()
        a[target] = v
        garbage[name] = mx.array(a)
    s = s._replace(**garbage)
    req = np.zeros(d.n, bool)
    req[0] = True
    _, lanes, rp = _reproduce(s, req, req, cfg)
    assert int(_np(lanes.slot)[0]) == target
    ns = rp.state
    W1, W2 = lifecycle.flush_pending(ns.W1, ns.W2, ns.pend_slot, ns.pend_W1, ns.pend_W2)  # next step's S0
    ns = ns._replace(W1=W1, W2=W2)
    for name, v in sentinel.items():
        row = _np(getattr(ns, name))[target]
        assert not np.any(row == v), name
    assert bool(_np(ns.alive)[target])


# --- immigration --------------------------------------------------------------------------------------------


def test_immigration_lanes_fields_and_ledger(imm_cfg: Config) -> None:
    cfg, d, mu = imm_cfg, dims(imm_cfg), imm_cfg.mutation
    s = make_scene(cfg, [dict(energy=80_000, birth_step=-40), dict(energy=5_000)], step=4)
    req = np.zeros(d.n, bool)
    req[0] = True
    ks, lanes, rp = _reproduce(s, req, np.zeros(d.n, bool), cfg)
    kind = _np(lanes.kind)
    assert np.array_equal(
        kind, [LaneKind.BIRTH] + [LaneKind.IMMIGRANT] * 7
    )  # min(40 − 2 − 1, 64, M − 1, room)
    assert int(_np(lanes.n_imm)) == 7 and np.all(_np(lanes.parent) == 0)
    ns = rp.state
    a1 = {k: _np(v) for k, v in ns._asdict().items()}
    im = _np(lanes.slot)[1:]
    S, e_init = cfg.energy.structure, cfg.energy.e_init
    assert _total(ns, cfg) - _total(s, cfg) == 7 * (e_init + S) - cfg.repro.birth_cost  # INJECT_IMMIGRATION
    assert np.array_equal(a1["uid"][im], 2 + np.arange(1, 8)) and int(a1["next_uid"]) == 10
    assert np.all(a1["parent_uid"][im] == -1) and np.all(a1["clade"][im] == cfg.population.n_init)
    assert np.all(a1["generation"][im] == 0) and np.all(a1["energy"][im] == e_init)
    assert np.all(a1["birth_step"][im] == 4)
    assert np.all(a1["mut_sigma"][im] == np.float32(mu.sigma_init))
    assert np.all(a1["mut_rate"][im] == np.float32(mu.rate_init))
    assert np.all(a1["offspring_frac"][im] == np.float32(mu.frac_init))
    assert np.all((a1["pos"][im] >= 0) & (a1["pos"][im] < cfg.world.size))
    assert np.all((a1["tag"][im] >= -1) & (a1["tag"][im] < 1))
    k = mx.random.split(ks[Stream.IMMIGRANT], 5)
    w1, w2 = brain.init_weights(k[3], k[4], d.m, cfg)
    assert np.array_equal(a1["pend_W1"][1:], _np(w1)[1:]) and np.array_equal(a1["pend_W2"][1:], _np(w2)[1:])
    assert np.array_equal(a1["brain_cost"][im], _np(brain.l1_cost(w1, w2, cfg))[1:])
    rec = rp.records
    assert _np(rec.valid).all() and np.all(_np(rec.parent_uid)[1:] == -1)

    capped = apply_overrides(cfg, ["population.immigrants_per_step=3"])
    _, lanes, _ = _reproduce(s, req, np.zeros(d.n, bool), capped)
    assert int(_np(lanes.n_imm)) == 3
    above = apply_overrides(cfg, ["population.immigration_floor=2"])  # n_surv already at the floor
    _, lanes, _ = _reproduce(s, req, np.zeros(d.n, bool), above)
    assert int(_np(lanes.n_imm)) == 0 and int(_np(lanes.n_birth)) == 1


def test_lane_maps_are_placements(micro_cfg: Config) -> None:
    """born_lane / child_of are exact inverses of the lane slots and parents (§12.3)."""
    d = dims(micro_cfg)
    s = make_scene(micro_cfg, [dict(energy=60_000, birth_step=-30) for _ in range(20)], step=1)
    req = np.zeros(d.n, bool)
    req[:20:2] = True  # 10 requesters for 8 lanes
    _, lanes, rp = _reproduce(s, req, np.zeros(d.n, bool), micro_cfg)
    parent, slot = _np(lanes.parent), _np(lanes.slot)
    assert int(_np(lanes.n_birth)) == d.m
    ns = rp.state
    assert np.array_equal(_np(ns.parent_uid)[slot], _np(s.uid)[parent])
    born = _np(ns.alive) & ~_np(s.alive)
    assert np.array_equal(np.flatnonzero(born), np.sort(slot))
    child_of = _np(ops.scatter_max_i32(d.n, lanes.parent, lanes.slot, lanes.kind == LaneKind.BIRTH))
    assert np.array_equal(child_of[parent], slot)
    assert np.all(
        _np(ns.energy)[np.flatnonzero(~req & _np(s.alive))] == 60_000
    )  # blocked/non-requesters pay 0
    blocked = np.setdiff1d(np.flatnonzero(req), parent)
    assert blocked.size == 2 and np.all(_np(ns.energy)[blocked] == 60_000)


def test_age_helper_matches_birth_step_convention() -> None:
    assert int(_np(age(mx.array(0, dtype=mx.int32), mx.array(-1, dtype=mx.int32)))) == 0
    assert int(_np(age(mx.array(8, dtype=mx.int32), mx.array(7, dtype=mx.int32)))) == 0


def test_lifecycle_compiles_and_matches_eager(micro_cfg: Config) -> None:
    """S0 + S14 + S16 are pure graph code: compiled output equals eager (no host reads or array branching).

    Bitwise on CPU; on GPU float outputs (the mutated child weights) agree to rounding, integers exactly.
    """
    cfg = apply_overrides(micro_cfg, ["population.immigration_floor=30"])
    d = dims(cfg)
    agents = [dict(energy=30_000 + 7_000 * i, pos=(2.0 * i, 3.0), birth_step=-25) for i in range(12)]
    s = make_scene(cfg, agents, bonds=[(0, 1), (2, 3)], step=4)

    def core(s: State) -> tuple:
        W1, W2 = lifecycle.flush_pending(s.W1, s.W2, s.pend_slot, s.pend_W1, s.pend_W2)
        s = s._replace(W1=W1, W2=W2)
        _, ks = rng.step_keys(s.key)
        a = age(s.step, s.birth_step)
        fcell = mx.zeros((d.n,), dtype=mx.int32)
        dt = lifecycle.deaths(
            s.alive, s.alive & (s.energy > 40_000), s.energy, mx.zeros_like(s.energy), a, s.uid,
            s.generation, s.clade, s.partner, s.rev, s.resource, fcell, cfg,
        )  # fmt: skip
        s1 = s._replace(alive=dt.survivor, energy=dt.energy, uid=dt.uid, partner=dt.partner, rev=dt.rev)
        lanes = lifecycle.allocate_lanes(dt.survivor & (dt.energy >= 80_000), dt.survivor, ks[2], cfg)
        rp = lifecycle.reproduce(s1._replace(resource=dt.resource), lanes, dt.survivor, ks, cfg)
        return rp.state, rp.records, dt.records, lanes, rp.endow

    eager, comp = core(s), mx.compile(core)(s)
    mx.eval(eager, comp)
    for a, b in zip(tree_flatten(eager), tree_flatten(comp), strict=True):
        x, y = _np(a[1]), _np(b[1])
        assert a[0] == b[0], a[0]
        if ON_GPU and x.dtype.kind == "f":
            np.testing.assert_allclose(x, y, rtol=1e-6, atol=1e-6, err_msg=a[0])
        else:
            assert np.array_equal(x, y), a[0]
    lanes = eager[3]
    assert int(_np(lanes.n_birth)) > 0 and int(_np(lanes.n_imm)) > 0 and int(_np(eager[2].valid.sum())) > 0
