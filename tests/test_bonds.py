"""§8 bonds: geometry, OR-symmetric breaking, the formation handshake (incl. the F9 fuzz port), child bonds,
death cleanup, springs and the device symmetry check."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from alife import bonds
from alife.config import Config
from alife.layout import B


def _ref_symmetric(partner: np.ndarray, rev: np.ndarray, alive: np.ndarray) -> bool:
    """Plain-loop reference of the §8.1 invariant plus no duplicate partners per row."""
    n = partner.shape[0]
    for i in range(n):
        seen = set()
        for b in range(B):
            j, r = int(partner[i, b]), int(rev[i, b])
            if (j < 0) != (r < 0):
                return False
            if j < 0:
                continue
            if not (alive[i] and j < n and alive[j] and j != i and 0 <= r < B):
                return False
            if partner[j, r] != i or rev[j, r] != b or j in seen:
                return False
            seen.add(j)
    return True


def _random_graph(rs: np.random.Generator, n: int, n_bonds: int, p_alive: float = 0.8):
    alive = rs.random(n) < p_alive
    partner = np.full((n, B), -1, np.int32)
    rev = np.full((n, B), -1, np.int32)
    live = np.flatnonzero(alive)
    for _ in range(n_bonds):
        if len(live) < 2:
            break
        i, j = rs.choice(live, 2, replace=False)
        if j in partner[i] or not (partner[i] < 0).any() or not (partner[j] < 0).any():
            continue
        si, sj = int(np.argmax(partner[i] < 0)), int(np.argmax(partner[j] < 0))
        partner[i, si], rev[i, si], partner[j, sj], rev[j, sj] = j, sj, i, si
    return partner, rev, alive


def _ok(partner: np.ndarray, rev: np.ndarray, alive: np.ndarray) -> bool:
    out = bonds.symmetric_ok(mx.array(partner), mx.array(rev), mx.array(alive))
    assert out.shape == () and out.dtype == mx.bool_
    return bool(np.array(out))


def test_symmetric_ok_on_valid_graphs() -> None:
    rs = np.random.default_rng(0)
    for _ in range(30):
        partner, rev, alive = _random_graph(rs, 40, 60)
        assert _ref_symmetric(partner, rev, alive)
        assert _ok(partner, rev, alive)
    n = 16
    assert _ok(np.full((n, B), -1, np.int32), np.full((n, B), -1, np.int32), np.zeros(n, bool))


def test_symmetric_ok_fires_on_single_entry_corruption() -> None:
    rs = np.random.default_rng(1)
    partner, rev, alive = _random_graph(rs, 30, 50)
    occ = np.argwhere(partner >= 0)
    assert len(occ) > 20
    n = partner.shape[0]
    fired = 0
    for i, b in occ:
        for field, values in (
            ("partner", (-1, i, (partner[i, b] + 1) % n, n + 3)),
            ("rev", (-1, (rev[i, b] + 1) % B, B)),
        ):
            for v in values:
                p2, r2 = partner.copy(), rev.copy()
                (p2 if field == "partner" else r2)[i, b] = v
                if _ref_symmetric(p2, r2, alive):
                    continue  # (cannot happen for these edits, but stay exact)
                assert not _ok(p2, r2, alive), (field, i, b, v)
                fired += 1
    assert fired > 100
    # An empty slot with a stray back-pointer, a dead endpoint, and a duplicate partner.
    i = int(np.argwhere((partner < 0).any(1) & alive)[0, 0])
    b = int(np.argmax(partner[i] < 0))
    r2 = rev.copy()
    r2[i, b] = 0
    assert not _ok(partner, r2, alive)
    a2 = alive.copy()
    a2[occ[0][0]] = False
    assert not _ok(partner, rev, a2)


def test_symmetric_ok_rejects_duplicates_and_self() -> None:
    n = 6
    alive = np.ones(n, bool)
    partner = np.full((n, B), -1, np.int32)
    rev = np.full((n, B), -1, np.int32)
    # Two bonds between rows 0 and 1, each internally consistent: a duplicate partner.
    partner[0, 0], rev[0, 0], partner[1, 0], rev[1, 0] = 1, 0, 0, 0
    partner[0, 1], rev[0, 1], partner[1, 1], rev[1, 1] = 1, 1, 0, 1
    assert not _ref_symmetric(partner, rev, alive) and not _ok(partner, rev, alive)
    partner = np.full((n, B), -1, np.int32)
    rev = np.full((n, B), -1, np.int32)
    partner[2, 0], rev[2, 0], partner[2, 1], rev[2, 1] = 2, 1, 2, 0  # self-bond, consistent back-pointers
    assert not _ok(partner, rev, alive)


def test_drop_dead() -> None:
    rs = np.random.default_rng(2)
    for _ in range(20):
        partner, rev, _alive = _random_graph(rs, 40, 70, p_alive=1.0)
        survivor = rs.random(40) < 0.7
        p2, r2 = bonds.drop_dead(mx.array(partner), mx.array(rev), mx.array(survivor))
        p2, r2 = np.array(p2), np.array(r2)
        keep = (partner >= 0) & survivor[:, None] & survivor[np.maximum(partner, 0)]
        assert np.array_equal(p2, np.where(keep, partner, -1))
        assert np.array_equal(r2, np.where(keep, rev, -1))
        assert np.all(p2[~survivor] == -1) and np.all(r2[~survivor] == -1)
        assert _ref_symmetric(p2, r2, survivor) and _ok(p2, r2, survivor)


def _child_scene():
    """Rows: 0 parent (1 free slot after 3 bonds), 1 parent (empty), 2 parent (full), 3 parent (no want),
    4..8 bystanders bonded to the parents, 9–12 newborn rows (children of 0, 1, 2, 3), 13 an immigrant."""
    n = 16
    partner = np.full((n, B), -1, np.int32)
    rev = np.full((n, B), -1, np.int32)

    def bond(i: int, si: int, j: int, sj: int) -> None:
        partner[i, si], rev[i, si], partner[j, sj], rev[j, sj] = j, sj, i, si

    bond(0, 0, 4, 0)
    bond(0, 2, 5, 0)
    bond(0, 3, 6, 0)
    for s, j in enumerate((4, 5, 6, 7)):
        bond(2, s, j, 1 if j != 7 else 0)
    child_of = np.full(n, -1, np.int32)
    born_parent = np.full(n, -1, np.int32)
    for p, c in ((0, 9), (1, 10), (2, 11), (3, 12)):
        child_of[p], born_parent[c] = c, p
    want = np.zeros(n, bool)
    want[[0, 1, 2]] = True
    want[13] = True  # irrelevant: row 13 has no child
    return partner, rev, child_of, want, born_parent


def test_add_child_bonds() -> None:
    partner, rev, child_of, want, born_parent = _child_scene()
    alive = np.ones(partner.shape[0], bool)
    assert _ref_symmetric(partner, rev, alive)
    p2, r2, made = bonds.add_child_bonds(
        mx.array(partner), mx.array(rev), mx.array(child_of), mx.array(want), mx.array(born_parent)
    )
    p2, r2, made = np.array(p2), np.array(r2), np.array(made)
    assert made.dtype == bool and np.flatnonzero(made).tolist() == [0, 1]  # 2 is full, 3 did not want one
    # Parent 0: first free slot is 1; child 9 uses slot 0 and points back at slot 1.
    assert (p2[0, 1], r2[0, 1]) == (9, 0) and (p2[9, 0], r2[9, 0]) == (0, 1)
    assert (p2[1, 0], r2[1, 0]) == (10, 0) and (p2[10, 0], r2[10, 0]) == (1, 0)
    assert np.all(p2[11] == -1) and np.all(p2[12] == -1) and np.all(p2[13] == -1)
    assert np.array_equal(p2[2], partner[2]) and np.array_equal(p2[3], partner[3])
    changed = np.flatnonzero((p2 != partner).any(1) | (r2 != rev).any(1)).tolist()
    assert changed == [0, 1, 9, 10]
    assert _ref_symmetric(p2, r2, alive) and _ok(p2, r2, alive)


def test_add_child_bonds_none() -> None:
    partner, rev, child_of, want, born_parent = _child_scene()
    p2, r2, made = bonds.add_child_bonds(
        mx.array(partner),
        mx.array(rev),
        mx.array(child_of),
        mx.zeros(want.shape, dtype=mx.bool_),
        mx.array(born_parent),
    )
    assert np.array_equal(np.array(p2), partner) and np.array_equal(np.array(r2), rev)
    assert not np.array(made).any()


@pytest.mark.parametrize("seed", range(5))
def test_add_child_bonds_random(seed: int) -> None:
    """Random parents and occupancy: the invariant holds and exactly the eligible parents bond."""
    rs = np.random.default_rng(seed)
    n, m = 64, 8
    partner, rev, alive = _random_graph(rs, n - m, 90, p_alive=1.0)
    partner = np.concatenate([partner, np.full((m, B), -1, np.int32)])
    rev = np.concatenate([rev, np.full((m, B), -1, np.int32)])
    alive = np.concatenate([alive, np.ones(m, bool)])
    parents = rs.choice(n - m, m, replace=False)
    child_of = np.full(n, -1, np.int32)
    born_parent = np.full(n, -1, np.int32)
    for k, p in enumerate(parents):
        if k < m - 1:  # the last newborn lane is an immigrant (no parent)
            child_of[p], born_parent[n - m + k] = n - m + k, p
    want = rs.random(n) < 0.7
    p2, r2, made = bonds.add_child_bonds(
        mx.array(partner), mx.array(rev), mx.array(child_of), mx.array(want), mx.array(born_parent)
    )
    p2, r2, made = np.array(p2), np.array(r2), np.array(made)
    expect = (child_of >= 0) & want & (partner < 0).any(1)
    assert np.array_equal(made, expect)
    assert _ref_symmetric(p2, r2, alive) and _ok(p2, r2, alive)
    for p in np.flatnonzero(expect):
        c = child_of[p]
        assert p2[c, 0] == p and p2[p, r2[c, 0]] == c and r2[p, r2[c, 0]] == 0


# ---------------------------------------------------------------------------------------------------------
# WP3: geometry, breaking, formation, springs
# ---------------------------------------------------------------------------------------------------------


def _np_sym(partner: np.ndarray, rev: np.ndarray, alive: np.ndarray) -> bool:
    """Vectorised equivalent of ``_ref_symmetric`` (used by the fuzz, where loops would be slow)."""
    n = partner.shape[0]
    occ = partner >= 0
    if np.any(occ != (rev >= 0)):
        return False
    i, b = np.nonzero(occ)
    j, r = partner[i, b], rev[i, b]
    if np.any((j >= n) | (r >= B) | (j == i)) or not (alive[i].all() and alive[j].all()):
        return False
    if not (np.all(partner[j, r] == i) and np.all(rev[j, r] == b)):
        return False
    srt = np.sort(np.where(occ, partner, -1 - np.arange(B)[None, :]), axis=1)  # empties made distinct
    return not np.any(srt[:, 1:] == srt[:, :-1])


def _np_min_image(d: np.ndarray, L: float) -> np.ndarray:
    return d - L * np.round(d / L)


def test_bond_geometry_and_n_bonds(micro_cfg: Config) -> None:
    rs = np.random.default_rng(10)
    L = micro_cfg.world.size
    n = 40
    partner, _rev, _ = _random_graph(rs, n, 60, p_alive=1.0)
    pos = rs.uniform(0, L, (n, 2)).astype(np.float32)
    pos[0], pos[partner[0, 0] if partner[0, 0] >= 0 else 1] = (0.2, 47.9), (47.7, 0.1)  # across both seams
    g = bonds.bond_geometry(mx.array(pos), mx.array(partner), micro_cfg)
    occ = partner >= 0
    off = _np_min_image(pos.astype(np.float64)[np.maximum(partner, 0)] - pos[:, None, :], L)
    off = np.where(occ[..., None], off, 0.0)
    assert g.off.shape == (n, B, 2) and g.length.shape == (n, B) and g.valid.dtype == mx.bool_
    assert np.array_equal(np.array(g.valid), occ)
    assert np.allclose(np.array(g.off), off, atol=1e-5)
    assert np.allclose(np.array(g.length), np.sqrt((off * off).sum(-1)), atol=1e-5)
    assert np.all(np.array(g.length)[~occ] == 0) and np.all(np.array(g.off)[~occ] == 0)
    assert np.all(np.array(g.length) <= L * np.sqrt(0.5) + 1e-4)  # minimum image
    nb = bonds.n_bonds(mx.array(partner))
    assert nb.dtype == mx.int32 and np.array_equal(np.array(nb), occ.sum(1))


def test_break_one_side_clears_both() -> None:
    n = 8
    partner = np.full((n, B), -1, np.int32)
    rev = np.full((n, B), -1, np.int32)
    partner[0, 1], rev[0, 1], partner[3, 2], rev[3, 2] = 3, 2, 0, 1
    partner[0, 0], rev[0, 0], partner[5, 0], rev[5, 0] = 5, 0, 0, 0
    flag = np.zeros((n, B), bool)
    flag[3, 2] = True  # only row 3 releases its bond to 0
    p2, r2, brk = bonds.break_bonds(mx.array(partner), mx.array(rev), mx.array(flag))
    p2, r2, brk = np.array(p2), np.array(r2), np.array(brk)
    assert p2[0, 1] == -1 and r2[0, 1] == -1 and p2[3, 2] == -1 and r2[3, 2] == -1
    assert (p2[0, 0], r2[0, 0], p2[5, 0], r2[5, 0]) == (5, 0, 0, 0)
    assert np.flatnonzero(brk.reshape(-1)).tolist() == [0 * B + 1, 3 * B + 2]
    # Flags on empty slots are ignored.
    flag = partner < 0
    p3, _, brk3 = bonds.break_bonds(mx.array(partner), mx.array(rev), mx.array(flag))
    assert not np.array(brk3).any() and np.array_equal(np.array(p3), partner)
    flag[5, 0] = True  # now row 5 releases 0–5; 0–3 stays
    p3, _r3, brk3 = bonds.break_bonds(mx.array(partner), mx.array(rev), mx.array(flag))
    p3 = np.array(p3)
    assert np.flatnonzero(np.array(brk3).reshape(-1)).tolist() == [0 * B + 0, 5 * B + 0]
    assert p3[0, 0] == -1 and p3[5, 0] == -1 and p3[0, 1] == 3 and p3[3, 2] == 0


def test_break_random_matches_reference() -> None:
    rs = np.random.default_rng(11)
    for _ in range(20):
        partner, rev, alive = _random_graph(rs, 50, 90)
        flag = rs.random((50, B)) < 0.2
        p2, r2, brk = bonds.break_bonds(mx.array(partner), mx.array(rev), mx.array(flag))
        p2, r2, brk = np.array(p2), np.array(r2), np.array(brk)
        occ = partner >= 0
        other = flag[np.maximum(partner, 0), np.maximum(rev, 0)]
        expect = occ & (flag | other)
        assert np.array_equal(brk, expect)
        i, b = np.nonzero(expect)
        assert np.all(expect[partner[i, b], rev[i, b]])  # both ends report the break
        assert np.array_equal(p2, np.where(expect, -1, partner)) and np.array_equal(
            r2, np.where(expect, -1, rev)
        )
        assert _np_sym(p2, r2, alive) and _ok(p2, r2, alive)


def _form(partner, rev, can, target, seed: int = 0):
    out = bonds.form_bonds(
        mx.array(partner), mx.array(rev), mx.array(can), mx.array(target, dtype=mx.int32), mx.random.key(seed)
    )
    return tuple(np.array(a) for a in out)


def _empty_bonds(n: int) -> tuple[np.ndarray, np.ndarray]:
    return np.full((n, B), -1, np.int32), np.full((n, B), -1, np.int32)


def test_form_mutual_and_accept() -> None:
    n = 10
    partner, rev = _empty_bonds(n)
    can = np.zeros(n, bool)
    target = np.full(n, -1, np.int32)
    can[[0, 1, 2, 3]] = True
    target[0], target[1] = 1, 0  # mutual pair 0 <-> 1
    target[2] = 3  # 2 proposes, 3 accepts (willing, no target of its own)
    p2, r2, formed, as_prop = _form(partner, rev, can, target)
    assert (p2[0, 0], r2[0, 0], p2[1, 0], r2[1, 0]) == (1, 0, 0, 0)
    assert (p2[2, 0], r2[2, 0], p2[3, 0], r2[3, 0]) == (3, 0, 2, 0)
    assert formed.tolist() == [True] * 4 + [False] * 6
    assert as_prop.tolist() == [True, True, True] + [False] * 7
    assert _ref_symmetric(p2, r2, np.ones(n, bool))


def test_form_winning_proposer_never_accepts() -> None:
    n = 6
    partner, rev = _empty_bonds(n)
    can = np.ones(n, bool)
    target = np.array([1, 2, -1, -1, -1, -1], np.int32)  # 0 → 1 → 2; 1 is the sole (winning) proposer at 2
    p2, _r2, formed, as_prop = _form(partner, rev, can, target)
    assert p2[1, 0] == 2 and p2[2, 0] == 1 and as_prop[1] and not as_prop[2]
    assert np.all(p2[0] == -1) and not formed[0] and not as_prop[0]  # 1 did not accept 0
    assert formed.tolist() == [False, True, True, False, False, False]


def test_form_consent_full_rows_and_one_bond_per_agent() -> None:
    n = 12
    partner, rev = _empty_bonds(n)
    # Row 5 is full (bonded to 6..9), row 6 has room.
    for s_, j in enumerate((6, 7, 8, 9)):
        partner[5, s_], rev[5, s_], partner[j, 0], rev[j, 0] = j, 0, 5, s_
    can = np.ones(n, bool)
    can[4] = False  # unwilling
    target = np.full(n, -1, np.int32)
    target[0] = 4  # to an unwilling target: no bond
    target[1] = 5  # to a full target: no bond
    target[5] = 10  # a full proposer cannot propose
    target[[2, 3, 11]] = 10  # three proposers at 10: exactly one wins
    p2, r2, formed, as_prop = _form(partner, rev, can, target, seed=3)
    assert not formed[[0, 1, 4, 5]].any() and np.array_equal(p2[5], partner[5])
    assert formed[10] and formed[[2, 3, 11]].sum() == 1 and as_prop[[2, 3, 11]].sum() == 1
    w = int(p2[10, 0])
    assert w in (2, 3, 11) and p2[w, 0] == 10
    # The winner is the proposer with the highest permutation priority.
    prio = np.array(mx.random.permutation(n, key=mx.random.key(3)))
    assert w == max((2, 3, 11), key=lambda i: prio[i])
    new = (p2 >= 0) & (partner < 0)
    assert np.all(new.sum(1) <= 1) and np.array_equal(new.any(1), formed)
    assert _ref_symmetric(p2, r2, np.ones(n, bool))


def test_form_slot_placement_and_guards() -> None:
    n = 8
    partner, rev = _empty_bonds(n)
    # Row 0 has slots 0 and 2 occupied (to 6 and 7); its first free slot is 1. Row 3 has slot 0 occupied.
    partner[0, 0], rev[0, 0], partner[6, 0], rev[6, 0] = 6, 0, 0, 0
    partner[0, 2], rev[0, 2], partner[7, 0], rev[7, 0] = 7, 0, 0, 2
    partner[3, 0], rev[3, 0], partner[7, 1], rev[7, 1] = 7, 1, 3, 0
    can = np.ones(n, bool)
    target = np.full(n, -1, np.int32)
    target[0] = 3
    target[1] = 1  # self: dropped by the guard
    target[2] = 7  # 7 has two bonds and a free slot 2
    target[6] = 0  # current partner: dropped by the guard
    p2, r2, formed, as_prop = _form(partner, rev, can, target)
    assert (p2[0, 1], r2[0, 1]) == (3, 1) and (p2[3, 1], r2[3, 1]) == (0, 1)
    assert not formed[1] and not formed[6] and not as_prop[6]
    assert (p2[2, 0], r2[2, 0], p2[7, 2], r2[7, 2]) == (7, 2, 2, 0)
    assert _ref_symmetric(p2, r2, np.ones(n, bool))
    # Nobody willing: nothing changes.
    p3, r3, f3, a3 = _form(partner, rev, np.zeros(n, bool), target)
    assert np.array_equal(p3, partner) and np.array_equal(r3, rev) and not f3.any() and not a3.any()


def test_form_deterministic_and_compiles() -> None:
    rs = np.random.default_rng(12)
    n = 128
    partner, rev, alive = _random_graph(rs, n, 120)
    can = alive & (rs.random(n) < 0.8)
    target = np.where(rs.random(n) < 0.8, rs.integers(0, n, n), -1).astype(np.int32)
    target = np.where((target == np.arange(n)) | (partner == target[:, None]).any(1), -1, target)
    args = (mx.array(partner), mx.array(rev), mx.array(can), mx.array(target), mx.random.key(5))
    a = bonds.form_bonds(*args)
    b = bonds.form_bonds(*args)
    c = mx.compile(bonds.form_bonds)(*args)
    for x, y, z in zip(a, b, c, strict=True):
        assert np.array_equal(np.array(x), np.array(y)) and np.array_equal(np.array(x), np.array(z))
    assert np.array(a[2]).sum() > 10


def test_handshake_fuzz_symmetry_after_every_phase() -> None:
    """Port of F9: 300 rounds of breaks, formations, deaths and revivals on 256 rows."""
    rs = np.random.default_rng(2024)
    n = 256
    rows = np.arange(n)
    alive = rs.random(n) < 0.8
    partner, rev = _empty_bonds(n)
    break_f = mx.compile(bonds.break_bonds)
    form_f = mx.compile(bonds.form_bonds)
    drop_f = mx.compile(bonds.drop_dead)
    key = mx.random.key(7)
    formed_total = 0
    for _ in range(300):
        p0 = partner.copy()
        # Break: random release flags (on any slot; empty ones must be ignored).
        flag = rs.random((n, B)) < 0.08
        p, r, brk = break_f(mx.array(partner), mx.array(rev), mx.array(flag))
        partner, rev, brk = np.array(p), np.array(r), np.array(brk)
        assert _np_sym(partner, rev, alive) and _ok(partner, rev, alive)
        assert np.array_equal(brk, (p0 >= 0) & (partner < 0))
        # Form: willing alive agents propose to random alive non-partners (never self, as in §7 targeting).
        can = alive & (rs.random(n) < 0.7)
        target = np.where(rs.random(n) < 0.7, rs.integers(0, n, n), -1)
        bad = (target < 0) | (target == rows) | (p0 == target[:, None]).any(1) | ~alive[np.maximum(target, 0)]
        target = np.where(bad, -1, target).astype(np.int32)
        key, sub = mx.random.split(key)
        pb = partner.copy()
        p, r, formed, as_prop = form_f(mx.array(partner), mx.array(rev), mx.array(can), mx.array(target), sub)
        partner, rev, formed, as_prop = np.array(p), np.array(r), np.array(formed), np.array(as_prop)
        assert _np_sym(partner, rev, alive) and _ok(partner, rev, alive)
        new = (partner >= 0) & (pb < 0)
        assert np.array_equal(np.where(pb >= 0, partner, -1), pb)  # existing bonds untouched
        assert np.all(new.sum(1) <= 1) and np.array_equal(new.any(1), formed)
        assert np.all(~formed | (can & (pb < 0).any(1)))  # consent and a free slot
        newp = np.where(new, partner, -1).max(1)
        assert np.all(newp[as_prop] == target[as_prop]) and np.all(formed[as_prop])
        acc = formed & ~as_prop  # acceptors: paired with a proposer that targeted them
        assert np.all(as_prop[newp[acc]]) and np.all(target[newp[acc]] == rows[acc])
        formed_total += int(formed.sum())
        # Deaths, then revivals of dead rows (which are bond-free).
        survivor = alive & (rs.random(n) > 0.03)
        p, r = drop_f(mx.array(partner), mx.array(rev), mx.array(survivor))
        partner, rev = np.array(p), np.array(r)
        assert _np_sym(partner, rev, survivor) and _ok(partner, rev, survivor)
        alive = survivor | (~survivor & (rs.random(n) < 0.1))
        assert _np_sym(partner, rev, alive)
    assert formed_total > 10_000


def test_spring_forces_antisymmetric_bitwise(micro_cfg: Config) -> None:
    rs = np.random.default_rng(13)
    L = micro_cfg.world.size
    n = 16
    for trial in range(200):
        partner, rev = _empty_bonds(n)
        i, j = rs.choice(n, 2, replace=False)
        si, sj = rs.integers(0, B, 2)
        partner[i, si], rev[i, si], partner[j, sj], rev[j, sj] = j, sj, i, si
        pos = rs.uniform(0, L, (n, 2)).astype(np.float32)
        if trial % 3 == 0:  # straddle the seam
            pos[j] = np.mod(pos[i] + rs.uniform(-3, 3, 2), L).astype(np.float32)
        contract = rs.uniform(-1, 1, n).astype(np.float32)
        f = np.array(bonds.spring_forces(mx.array(pos), mx.array(partner), mx.array(contract), micro_cfg))
        assert np.array_equal(f[i], -f[j]), (trial, f[i], f[j])
        others = np.setdiff1d(np.arange(n), [i, j])
        assert np.all(f[others] == 0)


def test_spring_forces_zero_at_rest_and_reference(micro_cfg: Config) -> None:
    bc = micro_cfg.bonds
    n = 8
    partner, rev = _empty_bonds(n)
    partner[0, 0], rev[0, 0], partner[1, 2], rev[1, 2] = 1, 2, 0, 0
    partner[2, 1], rev[2, 1], partner[3, 0], rev[3, 0] = 3, 0, 2, 1
    pos = np.zeros((n, 2), np.float32)
    pos[0], pos[1] = (10.0, 10.0), (12.0, 10.0)  # relaxed: d = L0 = 2, contract 0
    pos[2], pos[3] = (20.0, 30.0), (20.0, 32.5)  # fully extended: d = 2.5, contract +1 on both
    contract = np.zeros(n, np.float32)
    contract[[2, 3]] = 1.0
    f = np.array(bonds.spring_forces(mx.array(pos), mx.array(partner), mx.array(contract), micro_cfg))
    assert np.all(f == 0)
    # Random graph against a float64 reference.
    rs = np.random.default_rng(14)
    n = 40
    partner, rev, _ = _random_graph(rs, n, 70, p_alive=1.0)
    pos = rs.uniform(0, micro_cfg.world.size, (n, 2)).astype(np.float32)
    contract = rs.uniform(-1, 1, n).astype(np.float32)
    f = np.array(bonds.spring_forces(mx.array(pos), mx.array(partner), mx.array(contract), micro_cfg))
    occ = partner >= 0
    off = _np_min_image(
        pos.astype(np.float64)[np.maximum(partner, 0)] - pos[:, None, :], micro_cfg.world.size
    )
    d = np.sqrt((off**2).sum(-1))
    rest = bc.rest_length * (
        1 + bc.contract_gain * (contract[:, None] + contract[np.maximum(partner, 0)]) / 2
    )
    ref = np.where(occ[..., None], (bc.k_spring * (d - rest) / np.maximum(d, 1e-6))[..., None] * off, 0).sum(
        1
    )
    assert np.allclose(f, ref, atol=1e-5)
    assert np.allclose(f.sum(0), 0, atol=1e-5)  # internal forces
