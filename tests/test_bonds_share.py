"""The conductive share of S12 (DESIGN §40.2): exact conservation on any bond graph, antisymmetry, consent,
the kin-scaled efficiency, the legacy gift unchanged in mode 0, and the ring/star overdraw cases (E14)."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from alife import energy as en
from alife import ops
from alife.config import load_config
from alife.layout import B
from test_bonds import _random_graph


def _cfg(mode: int, *extra: str):
    over = [f"bonds.share_mode={mode}", "roles.enabled=true", *extra] if mode else list(extra)
    return load_config(preset="tiny", overrides=over)


def _graph(edges: list[tuple[int, int]], n: int):
    partner = np.full((n, B), -1, np.int32)
    rev = np.full((n, B), -1, np.int32)
    for i, j in edges:
        si, sj = int(np.argmax(partner[i] < 0)), int(np.argmax(partner[j] < 0))
        assert partner[i, si] < 0 and partner[j, sj] < 0
        partner[i, si], rev[i, si], partner[j, sj], rev[j, sj] = j, sj, i, si
    return partner, rev


def _slots(cfg, E, partner, rev, *, share=None, solvent=None, partner0=None, kin=None):
    n = len(E)
    share = np.ones((n, B), np.float32) * 0.999 if share is None else share
    solvent = np.ones(n, bool) if solvent is None else solvent
    act = SimpleNamespace(share=mx.array(share.astype(np.float32)))
    sl = en.share_slots(
        mx.array(E.astype(np.int32)),
        mx.array(partner),
        mx.array(rev),
        mx.array(partner if partner0 is None else partner0),
        act,
        mx.array(solvent),
        cfg,
        None if kin is None else mx.array(kin.astype(np.float32)),
    )
    mx.eval(sl)
    return [np.asarray(a).astype(np.int64) for a in sl]


def _after(E, out, got):
    return E - out.sum(1) + got.sum(1)


def _check_exact(E, partner, out, inc, got):
    """S-1: every sent quantum is received once; nobody below zero; the heat is Σ(in − got)."""
    assert out.sum() == inc.sum()
    new = _after(E, out, got)
    assert new.min() >= 0
    assert E.sum() - new.sum() == (inc - got).sum()
    assert (out >= 0).all() and (got <= inc).all() and (out[partner < 0] == 0).all()


@pytest.mark.parametrize("mode", [1, 2])
@pytest.mark.parametrize("kin_floor", [1.0, 0.5, 0.0])
def test_s1_conservation_on_random_graphs(mode: int, kin_floor: float) -> None:
    cfg = _cfg(mode, f"bonds.share_kin_floor={kin_floor}")
    rs = np.random.default_rng(1)
    for _ in range(12):
        n = 300
        partner, rev, _ = _random_graph(rs, n, 400)
        E = rs.integers(0, 1_000_000, n).astype(np.int64)
        E[rs.random(n) < 0.2] = 0
        E[rs.random(n) < 0.1] = rs.integers(0, 40)
        kin = rs.random((n, B))
        share = rs.random((n, B))
        solvent = rs.random(n) < 0.9
        out, inc, got = _slots(cfg, E, partner, rev, share=share, solvent=solvent, kin=kin)
        _check_exact(E, partner, out, inc, got)
        # S-2: antisymmetric, richer end only
        for i, b in zip(*np.nonzero(out > 0), strict=True):
            j = partner[i, b]
            assert E[i] > E[j] and solvent[i]
            assert out[j, rev[i, b]] == 0
        # no flow below the quantisation step of the gap (gap * kappa_q8 >> 8 == 0)
        gap = E[:, None] - E[np.maximum(partner, 0)]
        assert (out[(partner >= 0) & (gap <= 0)] == 0).all()


def test_s2_no_overshoot_and_the_spread_never_widens() -> None:
    """Iterate the share on a star (degree 4), a chain and a ring: the maximum never rises, the minimum never
    falls, and the total only decreases by the heat."""
    cfg = _cfg(2)
    n = 40
    graphs = {
        "star": [(0, k) for k in range(1, 5)],
        "chain": [(k, k + 1) for k in range(n - 1)],
        "ring": [(k, (k + 1) % n) for k in range(n)],
        "k5": [(a, b) for a in range(5) for b in range(a + 1, 5)],
    }
    rs = np.random.default_rng(2)
    for name, edges in graphs.items():
        partner, rev = _graph(edges, n)
        used = np.unique(np.array(edges).ravel())
        E = np.zeros(n, np.int64)
        E[used] = rs.integers(0, 1_000_000, len(used))
        E[used[0]] = 1_000_000  # one rich node
        for _ in range(60):
            lo, hi, tot = E[used].min(), E[used].max(), E.sum()
            out, inc, got = _slots(cfg, E, partner, rev)
            _check_exact(E, partner, out, inc, got)
            E = _after(E, out, got)
            assert E[used].min() >= lo and E[used].max() <= hi and E.sum() <= tot, name


def test_e14_adversarial_rings_and_stars_never_overdraw() -> None:
    """E14. Every row sends at most ``E // B`` per slot, so the sum over its slots is at most ``E``: nobody
    goes below zero whatever the energies, with ties, all-rich, all-poor, alternating and one-in-a-sea."""
    for mode in (1, 2):
        cfg = _cfg(mode)
        n = 64
        ring = [(k, (k + 1) % n) for k in range(n)]
        star = [(0, k) for k in range(1, 5)] + [(5, k) for k in range(6, 10)]
        dense = [(a, b) for a in range(5) for b in range(a + 1, 5)]  # K5: every row has degree 4
        rs = np.random.default_rng(5)
        energies = [
            np.zeros(n),
            np.full(n, 1_000_000),
            np.arange(n) % 2 * 1_000_000,
            np.where(np.arange(n) == 0, 1_000_000, 0),
            np.where(np.arange(n) == 0, 3, 1_000_000),
            np.arange(n) * 15_000,
            rs.integers(0, 4, n),
            rs.integers(0, 1_000_000, n),
        ]
        for edges in (ring, star, dense):
            partner, rev = _graph(edges, n)
            for E in energies:
                out, inc, got = _slots(cfg, E.astype(np.int64), partner, rev)
                _check_exact(E.astype(np.int64), partner, out, inc, got)
                assert (out.sum(1) <= E).all()
    # the belt: even with a kappa beyond the validated range and no rate cap (which validation refuses), the
    # per-slot ``E // B`` keeps every row's total send within its energy
    wild = dataclasses.replace(
        cfg, bonds=dataclasses.replace(cfg.bonds, share_kappa=1.0, share_rate=900_000, share_efficiency=1.0)
    )
    for edges in (ring, star, dense):
        partner, rev = _graph(edges, n)
        for E in energies:
            out, inc, got = _slots(wild, E.astype(np.int64), partner, rev)
            _check_exact(E.astype(np.int64), partner, out, inc, got)
            assert (out.sum(1) <= E).all()
    # the hub of a star at full gap: 4 slots, each at most E // 4, kappa 1/8 of the gap
    cfg = _cfg(2)
    partner, rev = _graph([(0, k) for k in range(1, 5)], 8)
    E = np.array([1_000_000, 0, 0, 0, 0, 0, 0, 0], np.int64)
    out, _, _ = _slots(cfg, E, partner, rev)
    assert (out[0, :4] == 1000).all()  # min((gap * 32) >> 8 = 125000, cap share_rate = 1000, E // 4)
    E[0] = 6  # a poor hub: E // B = 1 per slot
    out, _, _ = _slots(cfg, E, partner, rev)
    assert out[0].sum() <= 4 and out.sum() <= 6


def test_s3_consent_gates_mode_1_and_mode_2_ignores_the_outputs() -> None:
    n = 4
    partner, rev = _graph([(0, 1), (2, 3)], n)
    E = np.array([900_000, 100_000, 900_000, 100_000], np.int64)
    zero = np.zeros((n, B))
    one = np.full((n, B), 0.999)
    c1, c2 = _cfg(1), _cfg(2)
    # the poor end closes its gate (share 0): mode 1 moves nothing on that bond; the rich end's gate is open
    sh = one.copy()
    sh[1, :] = 0.0
    out, inc, _ = _slots(c1, E, partner, rev, share=sh)
    assert out[0].sum() == 0 and out[2].sum() > 0 and inc.sum() == out.sum()
    out, _, _ = _slots(c1, E, partner, rev, share=zero)
    assert out.sum() == 0
    # mode 1 moves the weaker of the two outputs: share 0.5 at both ends sends (128 * 1000) >> 8 = 500
    half = np.full((n, B), 0.5)
    out, _, _ = _slots(c1, E, partner, rev, share=half)
    assert out[0, 0] == 500
    out, _, _ = _slots(
        c1, E, partner, rev, share=np.broadcast_to(np.where(np.arange(n)[:, None] == 1, 0.25, 0.75), (n, B))
    )
    assert out[0, 0] == (64 * 1000) >> 8  # min(floor(256*0.75), floor(256*0.25)) = 64
    # mode 2 ignores the outputs entirely
    out_z, _, _ = _slots(c2, E, partner, rev, share=zero)
    out_o, _, _ = _slots(c2, E, partner, rev, share=one)
    assert np.array_equal(out_z, out_o) and out_z[0, 0] == 1000
    # an insolvent sender sends nothing, an insolvent poor partner still receives
    out, inc, _ = _slots(c2, E, partner, rev, solvent=np.array([True, False, False, True]))
    assert out[0, 0] == 1000 and inc[1, 0] == 1000 and out[2].sum() == 0 and inc[3].sum() == 0
    # a slot whose partner changed since the start of the step (partner != partner0) carries nothing
    p0 = partner.copy()
    p0[0, 0] = 3
    out, _, _ = _slots(c2, E, partner, rev, partner0=p0)
    assert out[0, 0] == 0 and out[2, 0] == 1000


def test_kin_scaled_efficiency_only_changes_the_heat() -> None:
    n = 4
    partner, rev = _graph([(0, 1), (2, 3)], n)
    E = np.array([900_000, 100_000, 900_000, 100_000], np.int64)
    kin = np.zeros((n, B))
    kin[0, 0] = kin[1, 0] = 1.0  # a kin bond; the other bond has kin 0
    full = _slots(_cfg(2), E, partner, rev)
    out, inc, got = _slots(_cfg(2, "bonds.share_kin_floor=0.5"), E, partner, rev, kin=kin)
    assert np.array_equal(out, full[0]) and np.array_equal(inc, full[1])  # the send is kin-blind
    assert got[1, 0] == full[2][1, 0] == (1000 * 230) >> 8  # kin 1: the base efficiency q8(0.9) = 230
    assert got[3, 0] == (1000 * int(np.floor(256 * 0.9 * 0.5))) >> 8  # kin 0: floor 0.5 of it
    assert got[3, 0] < got[1, 0]
    _check_exact(E, partner, out, inc, got)


def _legacy(energy, partner, rev, partner0, share, solvent, rate, eff):
    """The pre-§40 ``share_slots``, copied out: ``min(floor(share·rate), E // B)`` gated by validity,
    ``partner == partner0`` and solvency, received through ``rev`` at the q8 efficiency."""
    valid = partner >= 0
    gate = valid & (partner == partner0) & solvent[:, None]
    amount = mx.minimum(ops.floor_i32(share * rate), (energy // B)[:, None])
    out = mx.where(gate, amount, 0)
    b = partner.shape[1]
    inc = mx.where(valid, out.reshape(-1)[mx.maximum(partner, 0) * b + mx.maximum(rev, 0)], 0)
    return out, inc, ops.mul_q8(inc, ops.q8(eff))


def test_s4_mode_0_is_the_legacy_gift_bit_for_bit() -> None:
    cfg = _cfg(0)
    rs = np.random.default_rng(6)
    for _ in range(8):
        n = 200
        partner, rev, _ = _random_graph(rs, n, 250)
        p0 = np.where(rs.random((n, B)) < 0.1, -1, partner).astype(np.int32)
        E = rs.integers(0, 1_000_000, n).astype(np.int32)
        share = rs.random((n, B)).astype(np.float32)
        solvent = rs.random(n) < 0.9
        got = _slots(cfg, E, partner, rev, share=share, solvent=solvent, partner0=p0)
        want = _legacy(
            mx.array(E),
            mx.array(partner),
            mx.array(rev),
            mx.array(p0),
            mx.array(share),
            mx.array(solvent),
            cfg.bonds.share_rate,
            cfg.bonds.share_efficiency,
        )
        for a, b in zip(got, want, strict=True):
            assert np.array_equal(a, np.asarray(b).astype(np.int64))
