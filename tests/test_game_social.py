"""§29.4 the device half of the social layer (G2): houses, kin and colony slots, the relation-window matrices,
the snapshot arrays and their relabel, and ``SocialRules``."""

from __future__ import annotations

import dataclasses

import mlx.core as mx
import numpy as np
import pytest

from alife import groups, layout, metrics
from alife.analysis import phylo
from alife.config import Config, apply_overrides, dims
from alife.game import social
from alife.game.social import (
    HOUSE_UNKNOWN,
    SOCIAL_V1,
    Expo,
    Rel,
    Snap,
    SocialRules,
    SocIn,
    SocState,
    init_soc,
    relabel,
    snapshot,
    soc_in,
    update,
)
from alife.state import CounterIdx as C
from alife.state import DeathCause, State, init_state, to_numpy
from alife.state import LedgerIdx as L
from alife.step import ActFlag, make_step
from game_helpers import blank_acts, life_reference, living_si, make_acts, make_si, window_reference
from helpers import set_output_bias

B = layout.B
I32 = mx.int32


def np_(a: mx.array) -> np.ndarray:
    return to_numpy(a)


def eval_soc(so: SocState) -> SocState:
    mx.eval(*so)
    return so


# --- houses and slots on hand-built lanes ---------------------------------------------------------------


def _hit_misses(rate: float, seed: int, want: int) -> tuple[list[int], list[int]]:
    """``want`` uids whose hash is below the cut of ``rate`` and ``want`` whose hash is not (numpy)."""
    u = np.arange(1000, 1000 + 4000, dtype=np.uint64)
    below = phylo.splitmix64(u, seed) < np.uint64(min(int(rate * 2.0**64), 2**64 - 1))
    return u[below][:want].tolist(), u[~below][:want].tolist()


def _lane_scene(cfg: Config, births: list[tuple[int, int, int]], immigrants: list[tuple[int, int]]):
    """Eight founders (rows 0-7, uid = clade = row) before; after: the same plus one child per ``births``
    entry ``(row, parent row, uid)`` and one immigrant per ``immigrants`` entry ``(row, uid)``.

    Returns ``(si_before, si_after, acts)``; the lanes are the births first, then the immigrants, then
    padding.
    """
    n_init = cfg.population.n_init
    before = living_si(cfg, n_init)
    after = living_si(cfg, n_init)
    acts = blank_acts(cfg)
    lane = 0
    for row, parent, uid in births:
        after["alive"][row] = True
        after["uid"][row] = uid
        after["parent_uid"][row] = parent
        after["clade"][row] = after["clade"][parent]
        acts["born_row"][lane], acts["born_parent"][lane], acts["born_uid"][lane] = row, parent, uid
        lane += 1
    for row, uid in immigrants:
        after["alive"][row] = True
        after["uid"][row] = uid
        after["clade"][row] = n_init
        acts["born_row"][lane], acts["born_uid"][lane] = row, uid
        lane += 1
    after["next_uid"] = np.array(
        1 + max([u for *_, u in births] + [u for _, u in immigrants], default=0), np.int64
    )
    return before, after, make_acts(acts)


def test_houses_on_hand_lanes(micro_cfg: Config) -> None:
    """Founders keep their clade; a birth inherits the parent row's house; rate 1 founds every birth, rate 0
    none; immigrants found their own; dead rows hold -1; a row reused in its death step takes the
    newborn's."""
    sr = SOCIAL_V1
    seed = micro_cfg.run.seed
    # rate 0: births inherit; immigrants still found their own house (their uid)
    cfg0 = apply_overrides(micro_cfg, ["game.house_rate=0"])
    before, after, acts = _lane_scene(cfg0, [(8, 2, 100), (9, 3, 101)], [(10, 102)])
    after["alive"][7] = False  # row 7 dies
    after["alive"][5] = True  # row 5 dies and is reused by a newborn in the same step:
    after["uid"][5], after["parent_uid"][5], after["clade"][5] = 103, 1, 1
    acts = acts._replace(
        born_row=mx.array(np.array([8, 9, 10, 5, -1, -1, -1, -1], np.int32)),
        born_parent=mx.array(np.array([2, 3, -1, 1, -1, -1, -1, -1], np.int32)),
        born_uid=mx.array(np.array([100, 101, 102, 103, -1, -1, -1, -1], np.int64)),
    )
    so0 = init_soc(make_si(before), cfg0, sr)
    assert np_(so0.house).tolist()[:8] == list(range(8)) and (np_(so0.house)[8:] == -1).all()
    so = eval_soc(update(so0, make_si(after), acts, cfg0, sr))
    house = np_(so.house)
    assert house[:7].tolist() == [
        0,
        1,
        2,
        3,
        4,
        1,
        6,
    ]  # founders keep theirs; row 5 took its parent's (row 1)
    assert house[7] == -1 and (house[11:] == -1).all()  # the dead row and the free rows
    assert house[8] == 2 and house[9] == 3  # births inherit the parent row's house
    assert house[10] == 102  # the immigrant founds its own house, whatever the rate
    # rate 1: every birth founds its own house (its uid)
    cfg1 = apply_overrides(micro_cfg, ["game.house_rate=1.0"])
    so1 = eval_soc(update(init_soc(make_si(before), cfg1, sr), make_si(after), acts, cfg1, sr))
    h1 = np_(so1.house)
    assert h1[8] == 100 and h1[9] == 101 and h1[10] == 102 and h1[5] == 103
    # a fractional rate: exactly the births whose hash is below the cut found their house
    rate = 0.5
    cfgr = apply_overrides(micro_cfg, [f"game.house_rate={rate}"])
    hit, miss = _hit_misses(rate, seed, 2)
    before2, after2, acts2 = _lane_scene(
        cfgr, [(8, 2, hit[0]), (9, 3, miss[0]), (10, 4, hit[1]), (11, 5, miss[1])], []
    )
    so2 = eval_soc(update(init_soc(make_si(before2), cfgr, sr), make_si(after2), acts2, cfgr, sr))
    assert np_(so2.house)[8:12].tolist() == [hit[0], 3, hit[1], 5]
    # the identical comparison, in numpy, of every uid (the cut is above 2^63 at this rate)
    assert groups.hash_cut(rate) >= 2**63


def test_reset_rule(micro_cfg: Config) -> None:
    """A row keeps its slots, colony and body only while the same occupant (same uid) lives in it: a death, a
    row reused in the step it died and a changed uid without a birth lane all reset it (G1's rule)."""
    cfg, sr = micro_cfg, SOCIAL_V1
    before, after, acts = _lane_scene(cfg, [(5, 1, 103)], [])  # row 5 is reused: it is alive before and after
    after["alive"][7] = False  # a death
    after["uid"][6] = 600  # an occupant with another uid and no lane (a gap in the observation)
    so = init_soc(make_si(before), cfg, sr)
    so = so._replace(
        colony=mx.array(np.arange(64, dtype=np.int32) % 3),
        body=mx.array(np.full(64, 4, np.int32)),
        kin=mx.array(np.where(before["alive"], 3, -1).astype(np.int32)),
    )
    out = update(so, make_si(after), acts, cfg, sr)
    colony, body, kin, house = (np_(x) for x in (out.colony, out.body, out.kin, out.house))
    assert colony[:5].tolist() == [0, 1, 2, 0, 1] and body[:5].tolist() == [4] * 5  # survivors keep them
    assert colony[5] == -1 and body[5] == 0  # the reused row: reset, then the newborn's kin slot
    assert kin[5] == 3  # a birth inherits its parent's slot (row 1's: 3), reset otherwise
    assert colony[6] == -1 and body[6] == 0 and kin[6] == -1 and house[6] == -1  # changed uid, no lane
    assert colony[7] == -1 and body[7] == 0 and kin[7] == -1 and house[7] == -1  # dead
    assert np_(out.uid)[:8].tolist() == [0, 1, 2, 3, 4, 103, 600, -1]


def test_init_soc_seeds_immigrant_lineages_unknown(micro_cfg: Config) -> None:
    """Rows with clade ``n_init`` (immigrants and their descendants) get ``HOUSE_UNKNOWN``, founders their
    clade; a child of a ``HOUSE_UNKNOWN`` row inherits it unless it founds."""
    cfg, sr = micro_cfg, SOCIAL_V1
    n_init = cfg.population.n_init
    si = living_si(cfg, 12)
    si["clade"][8:12] = n_init  # four rows of immigrant lineage
    so = init_soc(make_si(si), cfg, sr)
    house = np_(so.house)
    assert house[:8].tolist() == list(range(8))
    assert (house[8:12] == HOUSE_UNKNOWN).all() and HOUSE_UNKNOWN == -2
    assert (house[12:] == -1).all()
    assert (np_(so.kin)[:12] == sr.slots).all() and (np_(so.kin)[12:] == -1).all()
    assert (np_(so.colony) == -1).all() and (np_(so.body) == 0).all()
    # children of an unknown lineage: inherited, or founded at rate 1
    for rate, want in ((0.0, HOUSE_UNKNOWN), (1.0, 500)):
        c = apply_overrides(cfg, [f"game.house_rate={rate}"])
        after = {
            **si,
            "alive": si["alive"].copy(),
            "uid": si["uid"].copy(),
            "parent_uid": si["parent_uid"].copy(),
        }
        after["alive"][20], after["uid"][20], after["parent_uid"][20] = True, 500, 9
        after["clade"] = si["clade"].copy()
        after["clade"][20] = n_init
        acts = blank_acts(c)
        acts["born_row"][0], acts["born_parent"][0], acts["born_uid"][0] = 20, 9, 500
        so2 = update(init_soc(make_si(si), c, sr), make_si(after), make_acts(acts), c, sr)
        assert np_(so2.house)[20] == want, rate


def test_kin_slot_inheritance(micro_cfg: Config) -> None:
    """A newborn takes its parent's kin slot, an immigrant OTHER; dead rows and free rows hold -1."""
    cfg, sr = micro_cfg, SOCIAL_V1
    before, after, acts = _lane_scene(cfg, [(8, 2, 100), (9, 3, 101)], [(10, 102)])
    after["alive"][6] = False
    so = init_soc(make_si(before), cfg, sr)
    so = so._replace(kin=mx.array(np.array([0, 1, 2, 3, 4, 5, 6, 7, *([-1] * 56)], np.int32)))  # slots = row
    out = np_(update(so, make_si(after), acts, cfg, sr).kin)
    assert out[:6].tolist() == [0, 1, 2, 3, 4, 5] and out[6] == -1 and out[7] == 7
    assert out[8] == 2 and out[9] == 3  # births: the parent's slot
    assert out[10] == sr.slots  # an immigrant: OTHER
    assert (out[11:] == -1).all()


@pytest.mark.parametrize("rate", [0.0, 0.125, 1.0])
def test_houses_equal_offline_marker_roots(rate: float, tiny_cfg: Config) -> None:
    """On real steps with births and immigrants, the live house of every alive row is the uid of the nearest
    root of its lineage under ``phylo.marker_roots`` at the same rate and seed (the offline definition,
    DESIGN §29.4): founders and immigrants are roots, a birth is a root iff its uid hashes under the cut."""
    cfg = apply_overrides(tiny_cfg, [*IMMIGRATION, f"game.house_rate={rate}", "run.seed=7"])
    sr = dataclasses.replace(SOCIAL_V1, snapshot_every=EVERY)
    s = set_output_bias(init_state(cfg), cfg, BIAS)
    fn = make_step(cfg, observe=True)
    so = init_soc(soc_in(s), cfg, sr)
    parent = np.full(dims(cfg).n_cap * 8, -1, np.int64)  # parent uid by uid (the tree's index is the uid)
    for t in range(1, 61):
        s, out = fn(s)
        so = eval_soc(update(so, soc_in(s), out.acts, cfg, sr))
        mx.eval(s, out)
        b = out.births
        uids, pars = np_(b.uid), np_(b.parent_uid)
        used = np_(out.acts.born_row) >= 0
        parent[uids[used]] = pars[used]
        if t % 10 == 0 or t == 60:
            n_uid = int(np_(s.next_uid))
            tree = phylo.Tree(
                parent[:n_uid].astype(np.int32), np.zeros(n_uid, np.int32), np.ones(n_uid, np.int32), 1
            )
            roots = phylo.marker_roots(tree, rate, cfg.run.seed)
            near = phylo.nearest_root(tree.parent, roots)  # uid of the nearest root, per uid
            alive = np_(s.alive)
            uid, house = np_(s.uid)[alive], np_(so.house)[alive]
            assert np.array_equal(house, near[uid].astype(np.int64)), (rate, t)
    n_births = int((parent >= 0).sum())
    n_hashed = int(roots[parent[:n_uid] >= 0].sum())
    n_init = cfg.population.n_init
    assert n_births > 20 and (parent[n_init:n_uid] < 0).sum() > 0  # births and immigrants both
    assert {0.0: n_hashed == 0, 0.125: 0 < n_hashed < n_births, 1.0: n_hashed == n_births}[rate]


# --- the relation window on real steps ------------------------------------------------------------------

BIAS = {
    layout.OUT_HARVEST: 2.0,
    layout.OUT_BOND: 2.0,
    **{layout.OUT_SHARE + b: 1.0 for b in range(B)},
    layout.OUT_REPRODUCE: 2.0,
    layout.OUT_CHILD_BOND: 2.0,
    layout.OUT_ATTACK: 1.0,
    **{layout.OUT_EMIT + c: 1.0 for c in range(layout.NCH)},
}
IMMIGRATION = ["population.immigration_every=5", "population.immigration_count=8"]
EVERY = 10  # snapshot cadence of the real-step tests
STEPS = 40


def _arm(name: str, tiny_cfg: Config, tiny3d_cfg: Config) -> tuple[Config, dict[int, float]]:
    return {
        "2d": (tiny_cfg, {}),
        "3d": (tiny3d_cfg, {}),
        "biased": (tiny_cfg, BIAS),
        "immigration": (apply_overrides(tiny_cfg, IMMIGRATION), BIAS),
    }[name]


def _luts(cfg: Config, sr: social.SocialRules, t: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic synthetic slot tables: bins and component labels spread over slots 0-2 and OTHER."""
    g = sr.slots
    b = np.arange(sr.kin_bins**layout.T)
    kin = np.where((b + t) % 5 == 4, g, (b + t) % 5 % 3).astype(np.int32)
    col = (np.arange(dims(cfg).n) % 4 - 1).astype(np.int32)  # -1, 0, 1, 2
    return kin, col


@pytest.mark.parametrize("arm", ["2d", "3d", "biased", "immigration"])
def test_relations_match_counters(arm: str, tiny_cfg: Config, tiny3d_cfg: Config) -> None:
    """40 real steps with a snapshot every 10: per step Σ ATTACKS = ``ATTACKS``, Σ HITS = ``ATTACK_HITS``,
    Σ BONDS = ``BOND_PROPOSERS``, Σ shared = ``FLOW_SHARED``, Σ BOND_SLOTS = the occupied slots of the
    start-of-step state, the distinct kill targets are the ``DEATHS_KILLED`` rows, the ``expo`` identity of
    §29.4 holds exactly in integers; and the window matrices equal an independent numpy reference built from
    the acts and the previous slots (the slots changing at every snapshot, so that several slot pairs are
    exercised)."""
    cfg, bias = _arm(arm, tiny_cfg, tiny3d_cfg)
    sr = dataclasses.replace(SOCIAL_V1, snapshot_every=EVERY)
    g, k = sr.slots + 1, dims(cfg).k
    s = init_state(cfg)
    if bias:
        s = set_output_bias(s, cfg, bias)
    fn = make_step(cfg, observe=True)
    so = init_soc(soc_in(s), cfg, sr)
    # spread the initial rows over the slots too, so the first window is not all OTHER
    first = np_(s.alive)
    so = so._replace(kin=mx.array(np.where(first, np.arange(len(first)) % 4 % g, -1).astype(np.int32)))
    so = so._replace(slot_group=social._slot_groups(soc_in(s).partner, so.kin))
    seen = dict.fromkeys(("attacks", "hits", "bonds", "shared", "kills", "bslots", "pairs"), 0)
    win_rel, win_expo, win_shared = (np.zeros(x, np.int64) for x in ((5, g, g), (2, g, g, k), (g, g)))
    for t in range(1, STEPS + 1):
        prev_kin, prev_sg = np_(so.kin), np_(so.slot_group)
        prev_alive = np_(s.alive)
        s, out = fn(s)
        a = out.acts
        si = soc_in(s)
        so = eval_soc(update(so, si, a, cfg, sr))
        mx.eval(s, out)
        an = {f: np_(getattr(a, f)) for f in a._fields}
        c, led = np_(out.counters), np_(out.ledger)
        r, e, sh = window_reference(prev_kin, prev_sg, an, g, k)
        win_rel, win_expo, win_shared = win_rel + r, win_expo + e, win_shared + sh
        where = f"{arm} step {t}"
        # the per-step identities with the engine's counters
        assert r[Rel.ATTACKS.value].sum() == c[C.ATTACKS], where
        assert r[Rel.HITS.value].sum() == c[C.ATTACK_HITS], where
        assert r[Rel.BONDS.value].sum() == c[C.BOND_PROPOSERS], where
        assert sh.sum() == led[L.FLOW_SHARED], where
        assert r[Rel.BOND_SLOTS.value].sum() == (prev_sg >= 0)[prev_alive].sum(), where
        killed = np.flatnonzero((an["flags"] & ActFlag.KILLED.value) > 0)
        att = (an["flags"] & ActFlag.ATTACKED.value) > 0
        targets = set(an["attack_target"][att & np.isin(an["attack_target"], killed)].tolist())
        assert targets == set(killed.tolist()), where  # every killed row was a target of a counted attack
        assert len(killed) == c[C.DEATHS_KILLED], where
        # exposure: per actor slot and n_i, Σ_b expo[., a, b, j] = (j + 1) · (actors with n_i = j + 1)
        for which, flag in (
            (Expo.ATTACK.value, ActFlag.ATTACKED.value),
            (Expo.BOND.value, ActFlag.PROPOSED.value),
        ):
            ni = (an["nbr"] >= 0).sum(axis=1)
            actors = (an["flags"] & flag) > 0
            for slot in range(g):
                for j in range(k):
                    n_act = int((actors & (prev_kin == slot) & (ni == j + 1)).sum())
                    assert e[which, slot, :, j].sum() == (j + 1) * n_act, (where, which, slot, j)
        # the device window equals the reference, exactly
        assert np.array_equal(np_(so.rel), win_rel), where
        assert np.array_equal(np_(so.expo), win_expo), where
        assert np.array_equal(np_(so.shared), win_shared), where
        seen["attacks"] += int(r[0].sum())
        seen["hits"] += int(r[1].sum())
        seen["kills"] += int(r[2].sum())
        seen["bonds"] += int(r[3].sum())
        seen["bslots"] += int(r[4].sum())
        seen["shared"] += int(sh.sum())
        seen["pairs"] += int(((r[0] > 0).sum() > 1) or ((r[4] > 0).sum() > 1))
        if t % EVERY == 0:  # a snapshot: the matrices travel in it, then relabel resets the window
            snap = snapshot(mx.array(prev_kin), so, si, a, cfg, sr)
            mx.eval(*snap)
            assert np.array_equal(np_(snap.rel), win_rel) and np.array_equal(np_(snap.expo), win_expo), where
            assert np.array_equal(np_(snap.shared), win_shared), where
            kin_lut, col_lut = _luts(cfg, sr, t)
            so = eval_soc(relabel(so, snap, mx.array(kin_lut), mx.array(col_lut)))
            win_rel, win_expo, win_shared = (np.zeros_like(x) for x in (win_rel, win_expo, win_shared))
            assert not np_(so.rel).any() and not np_(so.expo).any() and not np_(so.shared).any(), where
    # the arm exercises what it checks
    assert seen["bslots"] > 0 and seen["pairs"] > 0
    if arm in ("biased", "immigration"):
        for key in ("attacks", "hits", "bonds", "shared"):
            assert seen[key] > 0, (arm, key)


def test_kills_are_counted_on_the_victim_flag(tiny_cfg: Config) -> None:
    """With every founder attacking and holding 800 quanta, victims are drained to 0 and die KILLED: the
    KILLS row counts the attacks whose target died KILLED, and its distinct targets are the
    ``DEATHS_KILLED`` rows."""
    cfg = tiny_cfg
    sr = dataclasses.replace(SOCIAL_V1, snapshot_every=EVERY)
    s = set_output_bias(init_state(cfg), cfg, BIAS)
    e = np_(s.energy).copy()
    e[np_(s.alive)] = 800
    s = s._replace(energy=mx.array(e))
    fn = make_step(cfg, observe=True)
    so = init_soc(soc_in(s), cfg, sr)
    so = so._replace(
        kin=mx.array(np.where(np_(s.alive), np.arange(cfg.population.n_max) % 3, -1).astype(np.int32))
    )
    kills, total = 0, np.zeros((sr.slots + 1, sr.slots + 1), np.int64)
    for _ in range(6):
        prev_kin = np_(so.kin)
        s, out = fn(s)
        mx.eval(s, out)
        a = out.acts
        so = eval_soc(update(so, soc_in(s), a, cfg, sr))
        fl, tgt = np_(a.flags), np_(a.attack_target)
        att = (fl & ActFlag.ATTACKED.value) > 0
        dead_by_kill = (fl & ActFlag.KILLED.value) > 0
        for i in np.flatnonzero(att & dead_by_kill[np.maximum(tgt, 0)]):
            total[prev_kin[i], prev_kin[tgt[i]]] += 1
            kills += 1
        assert np_(out.counters)[C.DEATHS_KILLED] == dead_by_kill.sum()
        assert set(tgt[att & dead_by_kill[np.maximum(tgt, 0)]].tolist()) == set(
            np.flatnonzero(dead_by_kill).tolist()
        )
        assert np.array_equal(np_(so.rel)[Rel.KILLS.value], total)  # the window since the start (no snapshot)
    assert kills > 0 and (total.sum(axis=1) > 0).sum() > 1


# --- the roster's device counts (stage 7, DESIGN section 29.14) --------------------------------------------


def _grow(si: dict[str, np.ndarray], births: list[tuple[int, int, int]]) -> dict[str, np.ndarray]:
    """A copy of ``si`` with one more alive child per ``(row, parent row, uid)``."""
    out = {k: v.copy() for k, v in si.items()}
    for row, parent, uid in births:
        out["alive"][row], out["uid"][row], out["parent_uid"][row] = True, uid, out["uid"][parent]
        out["clade"][row] = out["clade"][parent]
    out["next_uid"] = np.array(1 + max(uid for *_, uid in births), np.int64)
    return out


def _lanes(acts: dict[str, np.ndarray], births: list[tuple[int, int, int]]) -> dict[str, np.ndarray]:
    for lane, (row, parent, uid) in enumerate(births):
        acts["born_row"][lane], acts["born_parent"][lane], acts["born_uid"][lane] = row, parent, uid
    return acts


def test_house_from_on_hand_lanes(micro_cfg: Config) -> None:
    """``house_from`` is the house of the founder's parent, inherited with the house: a house founded by a
    birth takes its parent's house; a child that does not found takes its parent's ``house_from``; an
    immigrant and the founders of the run have none (-1); a parent of unknown house gives
    ``HOUSE_UNKNOWN``; dead rows hold -1, and the snapshot carries the array."""
    sr = SOCIAL_V1
    cfg1 = apply_overrides(micro_cfg, ["game.house_rate=1.0"])
    cfg0 = apply_overrides(micro_cfg, ["game.house_rate=0"])
    before = living_si(cfg1, 8)
    so0 = init_soc(make_si(before), cfg1, sr)
    assert (np_(so0.house_from) == -1).all()
    # step 1, rate 1: row 8 is a child of row 2 and founds house 100 from house 2; row 10 is an immigrant
    s1 = _grow(before, [(8, 2, 100)])
    s1["alive"][10], s1["uid"][10], s1["clade"][10] = True, 102, cfg1.population.n_init
    acts1 = _lanes(blank_acts(cfg1), [(8, 2, 100)])
    acts1["born_row"][1], acts1["born_uid"][1] = 10, 102
    s1["alive"][7] = False
    so1 = eval_soc(update(so0, make_si(s1), make_acts(acts1), cfg1, sr))
    house, frm = np_(so1.house), np_(so1.house_from)
    assert house[8] == 100 and frm[8] == 2  # the founder's parent's house
    assert house[10] == 102 and frm[10] == -1  # an immigrant has no parent
    assert (frm[:8] == -1).all() and (frm[11:] == -1).all()  # the founders of the run, dead and free rows
    # step 2, rate 0: row 9 is a child of row 8 and keeps house 100 and its origin, house 2
    s2 = _grow(s1, [(9, 8, 101)])
    so2 = eval_soc(update(so1, make_si(s2), make_acts(_lanes(blank_acts(cfg0), [(9, 8, 101)])), cfg0, sr))
    assert np_(so2.house)[9] == 100 and np_(so2.house_from)[9] == 2 and np_(so2.house_from)[8] == 2
    # step 3, rate 1: row 11 is a child of row 9 and founds house 103 from house 100
    s3 = _grow(s2, [(11, 9, 103)])
    so3 = eval_soc(update(so2, make_si(s3), make_acts(_lanes(blank_acts(cfg1), [(11, 9, 103)])), cfg1, sr))
    assert np_(so3.house)[11] == 103 and np_(so3.house_from)[11] == 100
    assert np_(so3.house_from)[8] == 2 and np_(so3.house_from)[9] == 2  # the others keep theirs
    # a row that dies and is reused by a newborn in the same step takes the newborn's, not the old row's
    s4 = _grow(s3, [(8, 11, 104)])  # row 8 (house 100, from 2) is reused by a child of row 11
    acts4 = _lanes(blank_acts(cfg1), [(8, 11, 104)])
    so4 = eval_soc(update(so3, make_si(s4), make_acts(acts4), cfg1, sr))
    assert np_(so4.house)[8] == 104 and np_(so4.house_from)[8] == 103
    # a parent of unknown house (an observer attached mid-run): HOUSE_UNKNOWN, whatever the rate
    unk = living_si(cfg1, 12)
    unk["clade"][8:12] = cfg1.population.n_init
    child = _grow(unk, [(20, 9, 500)])
    so_u = update(
        init_soc(make_si(unk), cfg1, sr),
        make_si(child),
        make_acts(_lanes(blank_acts(cfg1), [(20, 9, 500)])),
        cfg1,
        sr,
    )
    assert np_(so_u.house)[20] == 500 and np_(so_u.house_from)[20] == HOUSE_UNKNOWN
    # the snapshot carries the array, dead rows masked
    snap = snapshot(so3.kin, so3, make_si(s3), None, micro_cfg, sr)
    mx.eval(*snap)
    alive = s3["alive"]
    assert np.array_equal(np_(snap.house_from), np.where(alive, np_(so3.house_from), -1))
    assert np_(snap.house_from).dtype == np.int64


def test_window_life_on_hand_lanes(micro_cfg: Config) -> None:
    """Births are counted for the slot the newborn inherits (its parent's at the start of the step) and
    deaths for the dead row's slot, by cause; immigrants are not births; a cause on a row that was dead at the
    start and a birth from a dead parent are not counted; the window sums over steps, the snapshot carries it
    and ``relabel`` zeroes it."""
    cfg, sr = micro_cfg, SOCIAL_V1
    g, other = sr.slots + 1, sr.slots
    n = dims(cfg).n
    before = living_si(cfg, 12)
    so = init_soc(make_si(before), cfg, sr)
    slots = [0, 0, 1, 1, 2, 2, 3, 3, other, other, 1, 2, *([-1] * (n - 12))]
    so = so._replace(kin=mx.array(np.array(slots, np.int32)))
    assert np_(so.births).shape == (g,) and np_(so.deaths).shape == (len(DeathCause), g)
    assert not np_(so.births).any() and not np_(so.deaths).any()
    # step 1: births from rows 0, 2, 8 (OTHER) and 0; an immigrant; six deaths of all four causes
    births = [(20, 0, 100), (21, 2, 101), (22, 8, 102), (23, 0, 103)]
    after = _grow(before, births)
    after["alive"][24], after["uid"][24] = True, 104
    acts = _lanes(blank_acts(cfg), births)
    acts["born_row"][4], acts["born_uid"][4] = 24, 104  # an immigrant lane: no parent
    acts["born_row"][5], acts["born_parent"][5], acts["born_uid"][5] = 25, 50, 105  # a dead parent (stray)
    deaths = {1: DeathCause.STARVED, 3: DeathCause.KILLED, 4: DeathCause.AGED, 5: DeathCause.CATASTROPHE}
    deaths |= {9: DeathCause.STARVED, 6: DeathCause.KILLED}
    for row, cause in deaths.items():
        acts["death_cause"][row] = cause.value
        after["alive"][row] = False
    acts["death_cause"][40] = DeathCause.AGED.value  # inconsistent: row 40 was dead at the start
    ref_b, ref_d = life_reference(np.array(slots, np.int32), acts, g, len(DeathCause))
    so1 = eval_soc(update(so, make_si(after), make_acts(acts), cfg, sr))
    want_b = np.zeros(g, np.int64)
    want_b[[0, 1, other]] = [2, 1, 1]
    want_d = np.zeros((len(DeathCause), g), np.int64)
    for row, cause in deaths.items():
        want_d[cause.value, slots[row]] += 1
    assert np.array_equal(np_(so1.births), want_b) and np.array_equal(ref_b, want_b)
    assert np.array_equal(np_(so1.deaths), want_d) and np.array_equal(ref_d, want_d)
    assert np_(so1.births).dtype == np.int64 and np_(so1.deaths).dtype == np.int64
    assert want_d.sum() == 6 and want_b.sum() == 4  # the immigrant and the stray cause are not counted
    # step 2 adds to the window: row 20 (slot 0, born in step 1) starves, row 21 (slot 1) has a child
    kin1 = np_(so1.kin)
    assert kin1[20] == 0 and kin1[21] == 1
    births2 = [(30, 21, 105)]
    after2 = _grow(after, births2)
    acts2 = _lanes(blank_acts(cfg), births2)
    acts2["death_cause"][20] = DeathCause.STARVED.value
    after2["alive"][20] = False
    b2, d2 = life_reference(kin1, acts2, g, len(DeathCause))
    so2 = eval_soc(update(so1, make_si(after2), make_acts(acts2), cfg, sr))
    assert np.array_equal(np_(so2.births), want_b + b2) and np.array_equal(np_(so2.deaths), want_d + d2)
    assert (want_b + b2)[1] == 2 and (want_d + d2)[DeathCause.STARVED.value, 0] == 2
    # the snapshot carries the window as it stands, then relabel resets it
    snap = snapshot(so1.kin, so2, make_si(after2), make_acts(acts2), cfg, sr)
    mx.eval(*snap)
    assert np.array_equal(np_(snap.births), np_(so2.births)) and np.array_equal(
        np_(snap.deaths), np_(so2.deaths)
    )
    kin_lut, col_lut = np.zeros(16**4, np.int32), np.zeros(n, np.int32)
    out = eval_soc(relabel(so2, snap, mx.array(kin_lut), mx.array(col_lut)))
    assert not np_(out.births).any() and not np_(out.deaths).any()
    assert out.births.shape == so2.births.shape and out.deaths.shape == so2.deaths.shape


def test_window_life_matches_counters(tiny_cfg: Config) -> None:
    """60 real steps with immigrants, senescence (``max_age`` 40: the founders die at once, beyond the death
    records' cap) and a snapshot every 10: per step the births over the slots are the ``BIRTHS`` counter,
    the deaths of each cause are the ``DEATHS_*`` counter, and the window equals an independent numpy
    reference built from the channel and the previous slots (slots reshuffled at every snapshot)."""
    cfg = apply_overrides(tiny_cfg, [*IMMIGRATION, "population.max_age=40"])
    sr = dataclasses.replace(SOCIAL_V1, snapshot_every=EVERY)
    g, nc = sr.slots + 1, len(DeathCause)
    s = set_output_bias(init_state(cfg), cfg, BIAS)
    fn = make_step(cfg, observe=True)
    so = init_soc(soc_in(s), cfg, sr)
    first = np_(s.alive)
    so = so._replace(kin=mx.array(np.where(first, np.arange(len(first)) % 4 % g, -1).astype(np.int32)))
    counters = {
        DeathCause.STARVED: C.DEATHS_STARVED,
        DeathCause.KILLED: C.DEATHS_KILLED,
        DeathCause.AGED: C.DEATHS_AGED,
        DeathCause.CATASTROPHE: C.DEATHS_CATASTROPHE,
    }
    win_b, win_d = np.zeros(g, np.int64), np.zeros((nc, g), np.int64)
    seen_b, seen_d, slots_used = 0, np.zeros(nc, np.int64), set()
    for t in range(1, 61):
        prev_kin = np_(so.kin)
        s, out = fn(s)
        a = out.acts
        si = soc_in(s)
        so = eval_soc(update(so, si, a, cfg, sr))
        mx.eval(s, out)
        an = {f: np_(getattr(a, f)) for f in a._fields}
        c = np_(out.counters)
        b, d = life_reference(prev_kin, an, g, nc)
        where = f"step {t}"
        assert b.sum() == c[C.BIRTHS], where
        for cause, counter in counters.items():
            assert d[cause.value].sum() == c[counter], (where, cause)
        win_b, win_d = win_b + b, win_d + d
        assert np.array_equal(np_(so.births), win_b) and np.array_equal(np_(so.deaths), win_d), where
        seen_b += int(b.sum())
        seen_d += d.sum(axis=1)
        slots_used |= set(np.flatnonzero(win_b).tolist()) | set(np.flatnonzero(win_d.sum(axis=0)).tolist())
        if t % EVERY == 0:  # a snapshot carries the window, then the relabel resets it
            snap = snapshot(mx.array(prev_kin), so, si, a, cfg, sr)
            mx.eval(*snap)
            assert np.array_equal(np_(snap.births), win_b) and np.array_equal(np_(snap.deaths), win_d), where
            kin_lut, col_lut = _luts(cfg, sr, t)
            so = eval_soc(relabel(so, snap, mx.array(kin_lut), mx.array(col_lut)))
            win_b, win_d = np.zeros_like(win_b), np.zeros_like(win_d)
            assert not np_(so.births).any() and not np_(so.deaths).any(), where
    assert seen_b > 20 and seen_d[DeathCause.STARVED.value] > 0 and seen_d[DeathCause.AGED.value] > 0
    assert len(slots_used) > 2  # several slots hold counts, not only OTHER


def test_snapshot_carries_the_archetype_labels(micro_cfg: Config) -> None:
    """``Snap.arch`` is the labels passed in, masked by alive, as int32; without labels it is zero; compiled
    equals eager and no output aliases an input."""
    cfg, sr = micro_cfg, SOCIAL_V1
    si = make_si(living_si(cfg, 6, tag=0.1))
    so = init_soc(si, cfg, sr)
    arch = mx.array(
        np.array([0, 1, 2, 7, 3, 6, 5, 5, *([4] * (dims(cfg).n - 8))], np.int32)
    )  # dead rows labelled
    mx.eval(*so, arch)
    eager = snapshot(so.kin, so, si, None, cfg, sr, arch)
    compiled = mx.compile(lambda k, so_, si_, ar: snapshot(k, so_, si_, None, cfg, sr, ar))(
        so.kin, so, si, arch
    )
    bare = snapshot(so.kin, so, si, None, cfg, sr)
    mx.eval(*eager, *compiled, *bare)
    want = np.where(np_(si.alive), np_(arch), 0)
    assert np.array_equal(np_(eager.arch), want) and want.tolist()[:8] == [0, 1, 2, 7, 3, 6, 0, 0]
    assert np.array_equal(np_(compiled.arch), want) and np_(eager.arch).dtype == np.int32
    assert not np_(bare.arch).any() and np_(bare.arch).shape == want.shape
    assert all(x is not arch for x in compiled) and all(x is not arch for x in eager)


# --- territory and the snapshot arrays ------------------------------------------------------------------


def _random_scene(cfg: Config, rows: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    d = dims(cfg)
    si = living_si(cfg, rows)
    si["tag"][:rows] = rng.uniform(-1.0, 1.0, (rows, layout.T))
    size = cfg.world.size
    si["pos"][:rows] = rng.uniform(0.0, size, (rows, d.nd))
    if d.nd == 3:
        si["pos"][:rows, 2] = rng.uniform(0.0, cfg.world.height, rows)
    si["pos"][0, :2] = [0.0, size - 1e-3]  # on the edges of the cell grid
    si["pos"][1, :2] = [size - 1e-3, 0.0]
    partner = np.full((rows, B), -1, np.int32)
    for r in range(0, rows - 1, 3):  # pairs and short chains
        partner[r, 0], partner[r + 1, 1] = r + 1, r
    si["partner"][:rows] = partner
    si["partner"][rows:] = -1
    return si


@pytest.mark.parametrize("dims_", ["2d", "3d"])
def test_territory_counts_match_numpy(dims_: str, tiny_cfg: Config, tiny3d_cfg: Config) -> None:
    """``Snap.terr`` is ``np.add.at`` over ``cell * G + kin`` of the alive rows with a slot, with the window's
    slots (``so.kin``)."""
    cfg, sr = (tiny_cfg if dims_ == "2d" else tiny3d_cfg), SOCIAL_V1
    g = sr.slots + 1
    si = _random_scene(cfg, 700, 3)
    so = init_soc(make_si(si), cfg, sr)
    rng = np.random.default_rng(5)
    kin = np.where(si["alive"], rng.integers(0, g, dims(cfg).n), -1).astype(np.int32)
    kin[10:20] = -1  # alive rows without a slot are not counted
    so = so._replace(kin=mx.array(kin))
    snap = snapshot(so.kin, so, make_si(si), None, cfg, sr)
    mx.eval(*snap)
    cells = sr.terr_cells
    scale = np.float32(cells / cfg.world.size)
    px = np.clip(np.floor(si["pos"][:, 0] * scale), 0, cells - 1).astype(np.int64)
    py = np.clip(np.floor(si["pos"][:, 1] * scale), 0, cells - 1).astype(np.int64)
    ref = np.zeros(cells * cells * g, np.int64)
    ok = si["alive"] & (kin >= 0)
    np.add.at(ref, (py * cells + px)[ok] * g + kin[ok], 1)
    assert np.array_equal(np_(snap.terr).astype(np.int64), ref)
    assert np_(snap.cell).tolist() == np.where(si["alive"], py * cells + px, 0).tolist()
    assert np_(snap.terr).dtype == np.int32 and np_(snap.terr).sum() == ok.sum()


def test_snapshot_arrays(tiny_cfg: Config) -> None:
    """``comp`` is ``metrics.components`` with ``colony_iters`` (the run key ``components_iters`` is not
    read); ``body = size[comp]``; ``bins`` and ``bin_counts`` are the numpy grid code and bincount; the
    temperament sums and rows are the numpy ones; dead rows are masked."""
    cfg, sr = tiny_cfg, SOCIAL_V1
    d = dims(cfg)
    g = sr.slots + 1
    si = _random_scene(cfg, 600, 11)
    si["partner"][:] = -1
    for r in range(0, 600, 7):  # chains of up to 5 rows
        for j in range(4):
            if r + j + 1 < 600:
                si["partner"][r + j, 0], si["partner"][r + j + 1, 1] = r + j + 1, r + j
    si["alive"][33] = False  # a hole in a chain: the bond to a dead row is ignored
    si["uid"][33] = -1
    s = make_si(si)
    so = init_soc(s, cfg, sr)
    rng = np.random.default_rng(2)
    kin0 = np.where(si["alive"], rng.integers(0, g, d.n), -1).astype(np.int32)
    acts = blank_acts(cfg)
    acts["q_kin"] = np.where(si["alive"][:, None], rng.integers(-32512, 32513, (d.n, 2)), 0).astype(np.int16)
    a = make_acts(acts)
    snap = snapshot(mx.array(kin0), so, s, a, cfg, sr)
    mx.eval(*snap)
    h = {f: np_(x) for f, x in zip(Snap._fields, snap, strict=True)}
    alive = si["alive"]
    comp, conv = metrics.components(s.partner, s.alive, sr.colony_iters)
    assert np.array_equal(h["comp"], np_(comp)) and bool(h["converged"]) == bool(np_(conv))
    # a config with another components_iters reads the same labelling
    other = apply_overrides(cfg, ["run.components_iters=1"])
    snap2 = snapshot(mx.array(kin0), so, s, a, other, sr)
    assert np.array_equal(np_(snap2.comp), h["comp"])
    size = np.bincount(h["comp"][alive], minlength=d.n)
    assert np.array_equal(h["body"], np.where(alive, size[h["comp"]], 0))
    assert h["body"].max() == 5 and h["body"][33] == 0
    tag = np.where(alive[:, None], si["tag"], 0.0).astype(np.float32)
    code = np.clip(np.floor((tag + 1.0) * np.float32(8.0)), 0, 15).astype(np.int64)
    bins = np.ravel_multi_index(code.T, (16,) * 4)
    assert np.array_equal(h["bins"], bins)
    assert np.array_equal(h["bin_counts"], np.bincount(bins[alive], minlength=16**4))
    assert h["bin_counts"].sum() == alive.sum() and h["bin_counts"].dtype == np.int32
    assert np.array_equal(h["uid"], np.where(alive, si["uid"], -1))
    assert np.array_equal(h["tag"], tag) and (h["pos"][~alive] == 0).all()
    assert int(h["next_uid"]) == 600
    # temperament: the positive and negative parts of both queries per start-of-step slot
    q = acts["q_kin"].astype(np.int64)
    parts = np.stack(
        [np.maximum(q[:, 0], 0), np.maximum(-q[:, 0], 0), np.maximum(q[:, 1], 0), np.maximum(-q[:, 1], 0)]
    )
    ref = np.zeros((4, g), np.int64)
    rows = np.zeros(g, np.int64)
    for i in np.flatnonzero(kin0 >= 0):
        ref[:, kin0[i]] += parts[:, i]
        rows[kin0[i]] += 1
    assert np.array_equal(h["q_sums"], ref) and np.array_equal(h["q_rows"], rows)
    assert h["q_sums"].dtype == np.int64 and h["q_rows"].dtype == np.int32
    # the territory counts the window's slots (``so.kin``: OTHER everywhere here), not the start-of-step ones
    cells = sr.terr_cells
    scale = np.float32(cells / cfg.world.size)
    cell = np.clip(np.floor(si["pos"][:, 1] * scale), 0, cells - 1).astype(np.int64) * cells + np.clip(
        np.floor(si["pos"][:, 0] * scale), 0, cells - 1
    ).astype(np.int64)
    terr = np.zeros(cells * cells * g, np.int64)
    np.add.at(terr, cell[alive] * g + sr.slots, 1)
    assert np.array_equal(h["terr"].astype(np.int64), terr)
    # the initial snapshot of a fresh observer has no acts: zero temperament
    snap0 = snapshot(so.kin, so, s, None, cfg, sr)
    mx.eval(*snap0)
    assert not np_(snap0.q_sums).any() and not np_(snap0.q_rows).any()
    assert np_(snap0.q_sums).shape == (4, g) and np_(snap0.q_rows).shape == (g,)


def test_relabel_resets_the_window(tiny_cfg: Config) -> None:
    """``relabel`` writes the luts' slots to alive rows, derives the partner slots, copies the body and zeroes
    ``rel``, ``expo`` and ``shared``; dead rows stay -1."""
    cfg, sr = tiny_cfg, SOCIAL_V1
    d = dims(cfg)
    g = sr.slots + 1
    si = _random_scene(cfg, 300, 4)
    s = make_si(si)
    so = init_soc(s, cfg, sr)
    so = so._replace(
        rel=mx.ones(so.rel.shape, dtype=mx.int64),
        expo=mx.ones(so.expo.shape, dtype=mx.int64),
        shared=mx.ones(so.shared.shape, dtype=mx.int64),
    )
    snap = snapshot(so.kin, so, s, None, cfg, sr)
    mx.eval(*snap)
    kin_lut = ((np.arange(16**4) * 7) % (g + 1) - 0).astype(np.int32) % g
    col_lut = (np.arange(d.n) % 5 - 1).astype(np.int32)
    out = relabel(so, snap, mx.array(kin_lut), mx.array(col_lut))
    mx.eval(*out)
    h = {f: np_(x) for f, x in zip(Snap._fields, snap, strict=True)}
    alive = si["alive"]
    kin = np_(out.kin)
    assert np.array_equal(kin, np.where(alive, kin_lut[h["bins"]], -1))
    assert np.array_equal(np_(out.colony), np.where(alive, col_lut[h["comp"]], -1))
    assert np.array_equal(np_(out.body), h["body"])
    want_sg = np.where(si["partner"] >= 0, kin[np.maximum(si["partner"], 0)], -1)
    assert np.array_equal(np_(out.slot_group), want_sg)
    assert not np_(out.rel).any() and not np_(out.expo).any() and not np_(out.shared).any()
    assert np.array_equal(np_(out.house), np_(so.house)) and np.array_equal(np_(out.uid), np_(so.uid))
    assert (kin[~alive] == -1).all()


def test_update_outputs_do_not_alias_inputs(micro_cfg: Config) -> None:
    """No output of ``update`` (nor of ``snapshot`` and ``relabel``) is an input array, eager or compiled, and
    the compiled results equal the eager ones; ``SocIn`` has no key and no weight field."""
    cfg, sr = micro_cfg, SOCIAL_V1
    assert "key" not in SocIn._fields and not {"W1", "W2", "pend_W1", "pend_W2"} & set(SocIn._fields)
    assert set(SocIn._fields) <= set(State._fields)
    si = make_si(living_si(cfg, 6, tag=0.1))
    acts = make_acts(blank_acts(cfg))
    so = init_soc(si, cfg, sr)
    mx.eval(*so)
    inputs = [*so, *si, *acts]
    compiled = mx.compile(lambda so_, si_, a_: update(so_, si_, a_, cfg, sr))
    eager, comp = update(so, si, acts, cfg, sr), compiled(so, si, acts)
    mx.eval(*eager, *comp)
    for name, out in (("eager", eager), ("compiled", comp)):
        assert type(out) is SocState
        for f, arr in zip(SocState._fields, out, strict=True):
            assert all(arr is not x for x in inputs), (name, f)
            assert arr.dtype == getattr(so, f).dtype and arr.shape == getattr(so, f).shape, (name, f)
    for f in SocState._fields:
        assert np.array_equal(np_(getattr(eager, f)), np_(getattr(comp, f))), f
    # the snapshot and the relabel, compiled, equal the eager ones and alias nothing
    snap_c = mx.compile(lambda kin0, so_, si_, a_: snapshot(kin0, so_, si_, a_, cfg, sr))
    s_e, s_c = snapshot(so.kin, so, si, acts, cfg, sr), snap_c(so.kin, so, si, acts)
    mx.eval(*s_e, *s_c)
    for f in Snap._fields:
        assert np.array_equal(np_(getattr(s_e, f)), np_(getattr(s_c, f))), f
    ins = [*so, *si, *acts]
    assert all(arr is not x for arr in s_c for x in ins)
    lut_k, lut_c = mx.zeros((16**4,), dtype=I32), mx.zeros((dims(cfg).n,), dtype=I32)
    rl_c = mx.compile(lambda so_, sn, k, c: relabel(so_, sn, k, c))(so, s_c, lut_k, lut_c)
    rl_e = relabel(so, s_e, lut_k, lut_c)
    mx.eval(*rl_c, *rl_e)
    assert all(arr is not x for arr in rl_c for x in [*so, *s_c, lut_k, lut_c])
    for f in SocState._fields:
        assert np.array_equal(np_(getattr(rl_e, f)), np_(getattr(rl_c, f))), f


# --- SocialRules ----------------------------------------------------------------------------------------

# re-pinned at stage 7 (the roster adds ``kin_name_salt``); G2 is not merged: no record has the old digest
SOCIAL_V1_DIGEST = "df6ba5e483e9f2fcf23508deeaee142b7e20ace317509d3881df6d8815483d7c"


def test_social_rules(micro_cfg: Config) -> None:
    assert social.SOCIAL_VERSION == 1 and SocialRules() == SOCIAL_V1
    bad = {
        "kin_enter": dict(kin_enter=(3, 9)),  # leave (4, 10) would exceed it at small n
        "kin_enter_shift": dict(kin_enter=(8, 11)),  # leave (shift 10) would exceed it at large n
        "colony_leave_low": dict(colony_leave=1),
        "colony_leave_high": dict(colony_leave=5),
        "snapshot_every": dict(snapshot_every=0),
        "slots": dict(slots=0),
        "slack": dict(slot_slack=-1),
        "iters": dict(colony_iters=0),
        "theta": dict(theta=(0, 10)),
        "theta_over": dict(theta=(11, 10)),
        "events": dict(events_max=-1),
        "relations": dict(relations_max=-1),
        "bins": dict(kin_bins=1),
    }
    for kw in bad.values():
        with pytest.raises(ValueError, match="SocialRules"):
            SocialRules(**kw)
    # every field enters the digest, and the V1 digest is pinned
    changed = {
        "snapshot_every": 251, "slots": 15, "slot_slack": 5, "kin_bins": 8, "kin_occ": (3, 12),
        "kin_enter": (9, 9), "kin_leave": (5, 10), "n_band": 4, "colony_enter": 5, "colony_leave": 4,
        "colony_iters": 33, "theta": (1, 4), "uid_spacing": 301, "foot_cells": 64, "terr_cells": 16,
        "terr_k": 3, "terr_min": 2, "sim_sample": 32, "sim_distinct": (1, 5), "events_max": 31,
        "relations_max": 31, "sim_salt": 1, "name_salt": 2, "kin_name_salt": 3,
    }  # fmt: skip
    assert set(changed) == {f.name for f in dataclasses.fields(SocialRules)}
    assert SOCIAL_V1.digest() == SOCIAL_V1_DIGEST
    seen = {SOCIAL_V1.digest()}
    for name, value in changed.items():
        digest = dataclasses.replace(SOCIAL_V1, **{name: value}).digest()
        assert digest not in seen, name
        seen.add(digest)
    assert len(SOCIAL_V1.digest()) == 64 and SOCIAL_V1.digest() == SocialRules().digest()
    # the static bounds of the device code
    SOCIAL_V1.check(micro_cfg)
    with pytest.raises(ValueError, match=r"SocialRules\.check"):
        SocialRules(kin_bins=256).check(micro_cfg)  # 256^4 bins overflow int32
    with pytest.raises(ValueError, match=r"SocialRules\.check"):
        SocialRules(terr_cells=30000).check(micro_cfg)
    big = apply_overrides(micro_cfg, ["population.n_max=1048576", "population.n_init=1024"])
    SOCIAL_V1.check(big)  # n_max = 2^20 still fits: 24·2^20 lanes at most
