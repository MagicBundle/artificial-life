"""§29.14 the roster (G2 stage 7): kin-group names, house events and the per-group window counts, centroid and
archetype mix of the snapshot record.

Scenes are planted as in ``test_game_society.py`` (``Scene``, ``cloud``) and fed to a ``SocDriver``, which
runs the device update, the snapshot, ``Society.settle`` and ``relabel`` as the observer does; the last test
runs the whole ``Observer`` on real steps and compares the record with an independent numpy reference."""

from __future__ import annotations

import dataclasses
import itertools
import json
import math
from typing import Any

import mlx.core as mx
import numpy as np
import pytest

from alife.config import Config, apply_overrides
from alife.game import society
from alife.game.observer import Observer
from alife.game.rules import ARCHETYPE_NAMES, RULES_V1
from alife.game.social import DEATH_CAUSES, KIN_NAME_SALT, SOCIAL_V1, SocialRules
from alife.game.society import Society, raw_house_names
from alife.state import CounterIdx as C
from alife.state import DeathCause, to_numpy
from alife.step import make_step
from game_helpers import SocDriver, blank_acts, life_reference, living_si, make_acts, planted_state
from helpers import set_output_bias
from test_game_social import BIAS
from test_game_society import Scene, cloud

A_BIN, B_BIN, C_BIN = (2, 2, 2, 2), (12, 12, 12, 12), (6, 13, 6, 13)
OTHER_ARCH_NAMES = (
    *("gatherer", "hunter", "provider", "trader", "nomad", "settler", "generalist"),
    *("drifter", "floater", "sinker"),  # version 3 (DESIGN §32): the kin table has ten shares
)


def lattice(start: int, n: int) -> np.ndarray:
    """Tags of ``n`` lone rows, one in each of the bins whose four digits are all odd, from the ``start``-th:
    no two of them, and none of the planted clouds above, share a face, so they form no group."""
    bins = list(itertools.islice(itertools.product(range(1, 16, 2), repeat=4), start, start + n))
    return np.array([-1.0 + (np.asarray(b) + 0.5) / 8.0 for b in bins], np.float32)


def name_of(house: int, sr: SocialRules = SOCIAL_V1) -> str:
    """The name of a house on its own (no collision with another among the top houses of these scenes)."""
    return raw_house_names(np.array([house]), sr.name_salt)[0]


def by_size(rec: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """The rows of the record's kin-group table by size (the scenes give every group its own size)."""
    kg = rec["kin_groups"]
    assert len(set(kg["size"])) == len(kg["size"])
    return {kg["size"][i]: {c: kg[c][i] for c in kg} for i in range(len(kg["id"]))}


# --- names ----------------------------------------------------------------------------------------------


def _naming_scene(cfg: Config) -> tuple[Scene, dict[str, slice]]:
    """A: 12 rows of house 8 and 8 of house 7 in one cloud (house 8 leads, house 7 has the smaller id); B: 16
    of house 9; C: 5 rows of house 10 and 4 of house 6, scattered on the lattice (they join no group yet)."""
    rng = np.random.default_rng(1)
    sc = Scene(cfg)
    rows = {
        "a8": sc.add(cloud(rng, A_BIN, 12), clade=8),
        "a7": sc.add(cloud(rng, A_BIN, 8), clade=7),
        "b": sc.add(cloud(rng, B_BIN, 16), clade=9),
        "c10": sc.add(lattice(0, 5), clade=10),
        "c6": sc.add(lattice(5, 4), clade=6),
    }
    return sc, rows


def _flip(sc: Scene, rows: dict[str, slice], cfg: Config) -> dict[str, np.ndarray]:
    """The second snapshot of ``_naming_scene``: six of A's house-8 rows leave for the lattice (house 7 now
    leads A, 8 rows to 6) and C's nine rows gather in a cloud of their own (admitted now)."""
    rng = np.random.default_rng(2)
    si = sc.copy_si()
    si["tag"][rows["a8"].start : rows["a8"].start + 6] = lattice(20, 6)
    si["tag"][rows["c10"].start : rows["c6"].stop] = cloud(rng, C_BIN, 9)
    return si


def test_a_group_is_named_after_its_dominant_house_once(tiny_cfg: Config) -> None:
    """A group admitted with house 8 in the lead (12 members to 8 of house 7, which has the smaller id) takes
    house 8's name and ``named_after`` 8, and keeps both when house 7 later leads it; a group admitted later
    is named from its own members (house 10, with 5 to the 4 of house 6); the names are the houses' own."""
    cfg = tiny_cfg
    sc, rows = _naming_scene(cfg)
    drv = SocDriver(cfg, sc.si)
    names0 = dict(drv.society.kin_names)
    assert sorted(n for n, _ in names0.values()) == sorted([name_of(8), name_of(9)])
    assert sorted(a for _, a in names0.values()) == [8, 9]
    rec = drv.advance(_flip(sc, rows, cfg), 250)
    table = by_size(rec)
    a, b, c = table[14], table[16], table[9]
    assert (a["name"], a["named_after"]) == (name_of(8), 8)  # kept, although house 7 leads it now (8 to 6)
    assert (b["name"], b["named_after"]) == (name_of(9), 9)
    assert (c["name"], c["named_after"]) == (name_of(10), 10)  # admitted now: named from its members
    assert {a["id"], b["id"]} == set(names0) and c["id"] not in names0
    assert len({a["name"], b["name"], c["name"]}) == 3
    houses = dict(zip(rec["houses"]["id"], rec["houses"]["name"], strict=True))
    assert (houses[7], houses[8], houses[9]) == (name_of(7), name_of(8), name_of(9))  # the houses' own names
    # still the same names at the next snapshot, and only live groups are kept
    rec2 = drv.advance(_flip(sc, rows, cfg), 500)
    assert by_size(rec2)[14]["name"] == name_of(8) and by_size(rec2)[9]["name"] == name_of(10)
    assert dict(drv.society.kin_names) == {
        r["id"]: (r["name"], r["named_after"]) for r in by_size(rec2).values()
    }


def test_a_tie_between_houses_goes_to_the_smaller_id(tiny_cfg: Config) -> None:
    """Six members of house 21 and six of house 20: the group is named after house 20."""
    cfg = tiny_cfg
    rng = np.random.default_rng(8)
    sc = Scene(cfg)
    sc.add(cloud(rng, A_BIN, 6), clade=21)
    sc.add(cloud(rng, A_BIN, 6), clade=20)
    drv = SocDriver(cfg, sc.si)
    ((_, (name, after)),) = drv.society.kin_names.items()
    assert (name, after) == (name_of(20), 20)


def test_the_names_survive_a_sidecar_round_trip(tiny_cfg: Config) -> None:
    """A society restored from its arrays right after the first snapshot continues exactly like the original:
    the same names (A keeps house 8's name although a fresh naming would now say house 7's), the same
    record, the same arrays."""
    cfg = tiny_cfg
    sc, rows = _naming_scene(cfg)
    drv = SocDriver(cfg, sc.si)
    arrays = drv.society.arrays()
    restored, why = Society.from_arrays(arrays, cfg, SOCIAL_V1, step=0)
    assert why == "" and restored is not None and restored.kin_names == drv.society.kin_names
    assert name_of(7) != name_of(8)  # a renaming at the flip would show
    twin = SocDriver.__new__(SocDriver)
    twin.__dict__.update({**drv.__dict__, "society": restored})
    flipped = _flip(sc, rows, cfg)
    r1, r2 = drv.advance(flipped, 250), twin.advance(flipped, 250)
    assert json.dumps(r1) == json.dumps(r2)
    assert by_size(r2)[14]["name"] == name_of(8)
    a, b = drv.society.arrays(), twin.society.arrays()
    assert a.keys() == b.keys() and all(np.array_equal(a[k], b[k]) for k in a)
    # the names are in the sidecar as sorted [id, name, house or null] items
    items = json.loads(str(a["host.kin.names"]))
    assert [i[0] for i in items] == sorted(i[0] for i in items) and {i[2] for i in items} == {8, 9, 10}


def test_a_group_with_no_member_in_a_top_house_gets_a_generated_name(tiny_cfg: Config) -> None:
    """With the 16 largest houses elsewhere, a group of nine one-row houses is named by the house name
    generator on ``splitmix64(id, kin_name_salt)`` (not ``name_salt``) and is named after no house; another
    salt gives another name."""
    cfg = tiny_cfg
    rng = np.random.default_rng(4)
    sc = Scene(cfg)
    for i in range(16):  # sixteen houses of 20 lone rows
        sc.add(lattice(20 * i, 20), clade=100 + i)
    sc.add(cloud(rng, (4, 4, 4, 4), 9))  # nine rows, nine houses of one row each
    assert KIN_NAME_SALT == SOCIAL_V1.kin_name_salt != SOCIAL_V1.name_salt
    for sr in (SOCIAL_V1, dataclasses.replace(SOCIAL_V1, kin_name_salt=123)):
        drv = SocDriver(cfg, sc.si, sr=sr)
        ((gid, (name, after)),) = drv.society.kin_names.items()
        assert after == -1
        assert name == raw_house_names(np.array([gid]), sr.kin_name_salt)[0]
        assert name != raw_house_names(np.array([gid]), sr.name_salt)[0]
        rec = drv.advance(sc.si, 250)
        assert len(rec["houses"]["id"]) == 16 and set(rec["houses"]["size"]) == {20}
        assert rec["kin_groups"]["name"] == [name] and rec["kin_groups"]["named_after"] == [None]
        if sr is SOCIAL_V1:
            first = name
        else:
            assert name != first


def gen_name(gid: int, sr: SocialRules = SOCIAL_V1) -> str:
    """The generated name of a kin group (``splitmix64(id, kin_name_salt)``)."""
    return raw_house_names(np.array([gid]), sr.kin_name_salt)[0]


def test_a_house_name_is_given_to_one_live_group_and_free_when_its_holder_ends(tiny_cfg: Config) -> None:
    """Two groups admitted together with the same dominant house: the first (by id) has the house's name, the
    second a name generated from its own id (named after no house). When the holder of the house's name ends,
    a group admitted later takes the house's name; one admitted while it is held is generated too."""
    cfg = tiny_cfg
    rng = np.random.default_rng(5)
    sc = Scene(cfg)
    a = sc.add(cloud(rng, A_BIN, 10), clade=5)
    b = sc.add(cloud(rng, B_BIN, 10), clade=5)
    d = sc.add(lattice(0, 10), clade=5)
    e = sc.add(lattice(10, 10), clade=5)
    drv = SocDriver(cfg, sc.si)
    base = name_of(5)
    names = drv.society.kin_names
    lo, hi = sorted(names)
    assert names[lo] == (base, 5) and names[hi] == (gen_name(hi), -1)
    assert gen_name(hi) != base and " " not in gen_name(hi)  # no id appended: it is a name of its own
    holder = lo
    a_id = int(drv.society.kin_slots[drv.kin_lut[_bin(A_BIN)]])
    holder_rows = a if a_id == lo else b
    # the holder of the house's name ends; D gathers and is admitted: it takes the house's name
    si1 = sc.copy_si()
    si1["tag"][holder_rows] = lattice(40, 10)
    si1["tag"][d] = cloud(rng, C_BIN, 10)
    rec1 = drv.advance(si1, 250)
    live = dict(drv.society.kin_names)
    assert holder not in live and live[hi] == (gen_name(hi), -1)
    (d_id,) = set(live) - {hi}
    assert live[d_id] == (base, 5)
    assert sorted(rec1["kin_groups"]["name"]) == sorted([base, gen_name(hi)])
    # E gathers while D holds the house's name: its own generated name
    si2 = {k: v.copy() for k, v in si1.items()}
    si2["tag"][e] = cloud(rng, (6, 6, 6, 6), 10)
    drv.advance(si2, 500)
    live = dict(drv.society.kin_names)
    (e_id,) = set(live) - {hi, d_id}
    assert live[e_id] == (gen_name(e_id), -1) and len({nm for nm, _ in live.values()}) == 3


def test_several_groups_of_one_clade_get_distinct_names(tiny_cfg: Config) -> None:
    """The main case of an observer attached mid-run (a resume with ``game.observe`` on, a fork): houses are
    seeded from clades, so six kin groups of one clade have the same one house. The first has its name, the
    others generated ones, and no name has an id appended; a group of another clade has its own house's."""
    cfg = tiny_cfg
    rng = np.random.default_rng(0)
    sc = Scene(cfg)
    for i in range(6):
        sc.add(cloud(rng, (2 + 2 * i, 3, 3, 3), 12), clade=2)
    sc.add(cloud(rng, (12, 12, 12, 12), 12), clade=1)
    drv = SocDriver(cfg, sc.si, step=1_490_000)
    names = drv.society.kin_names
    assert len(names) == 7 and len({nm for nm, _ in names.values()}) == 7
    assert all(" " not in nm for nm, _ in names.values())
    named = {a: (i, nm) for i, (nm, a) in names.items() if a >= 0}  # house -> (group, name)
    assert sorted(named) == [1, 2] and named[1][1] == name_of(1) and named[2][1] == name_of(2)
    clade2 = sorted(set(names) - {named[1][0]})
    assert len(clade2) == 6 and named[2][0] == clade2[0]  # the first group of the clade has the house's name
    assert all(names[i] == (gen_name(i), -1) for i in clade2[1:])  # the others, generated from their ids


def test_a_generated_name_that_is_held_takes_the_id(
    tiny_cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The suffix is the last resort: two groups with no house name whose generated names are the same (the
    generator is made to return one name for the kin salt) get the name and ``f"{name} {id}"``."""
    cfg = tiny_cfg
    rng = np.random.default_rng(4)
    sc = Scene(cfg)
    for i in range(16):  # sixteen houses of 20 lone rows
        sc.add(lattice(20 * i, 20), clade=100 + i)
    sc.add(cloud(rng, (4, 4, 4, 4), 9))
    sc.add(cloud(rng, (9, 9, 9, 9), 9))  # two groups of one-row houses
    real = society.house_names
    monkeypatch.setattr(
        society,
        "house_names",
        lambda ids, salt: ["Same"] * len(ids) if salt == SOCIAL_V1.kin_name_salt else real(ids, salt),
    )
    drv = SocDriver(cfg, sc.si)
    lo, hi = sorted(drv.society.kin_names)
    assert drv.society.kin_names[lo] == ("Same", -1) and drv.society.kin_names[hi] == (f"Same {hi}", -1)


def _bin(b: tuple[int, int, int, int]) -> int:
    return int(np.ravel_multi_index(b, (16,) * 4))


# --- house events ---------------------------------------------------------------------------------------


def _next(
    cfg: Config,
    si: dict[str, np.ndarray],
    *,
    births: tuple[tuple[int, int, int], ...] = (),
    immigrants: tuple[tuple[int, int], ...] = (),
    deaths: tuple[int, ...] = (),
    cause: DeathCause = DeathCause.STARVED,
) -> tuple[dict[str, np.ndarray], Any]:
    """The state after a step with ``births`` (row, parent row, uid), ``immigrants`` (row, uid) and ``deaths``
    (rows), and the channel that says so (lanes in the order given)."""
    out = {k: v.copy() for k, v in si.items()}
    acts = blank_acts(cfg)
    lane = 0
    for row, parent, uid in births:
        out["alive"][row], out["uid"][row], out["parent_uid"][row] = True, uid, out["uid"][parent]
        out["clade"][row] = out["clade"][parent]
        acts["born_row"][lane], acts["born_parent"][lane], acts["born_uid"][lane] = row, parent, uid
        lane += 1
    for row, uid in immigrants:
        out["alive"][row], out["uid"][row], out["clade"][row] = True, uid, 99_999
        acts["born_row"][lane], acts["born_uid"][lane] = row, uid
        lane += 1
    for row in deaths:
        out["alive"][row], out["uid"][row] = False, -1
        acts["death_cause"][row] = cause.value
    out["next_uid"] = np.array(
        max([out["next_uid"], *[u for *_, u in births], *[u for _, u in immigrants]]) + 1
    )
    return out, make_acts(acts)


def _lineage_cfg(tiny_cfg: Config) -> Config:
    return apply_overrides(tiny_cfg, ["game.house_rate=1.0"])


def test_house_events_on_a_scripted_lineage(tiny_cfg: Config) -> None:
    """Rate 1 founds a house at every birth. Twelve founders (houses 0-11): window 1 has two births (houses
    100 and 101, from houses 3 and 4) and two deaths (houses 5 and 6); window 2 has a grandchild (102, from
    100), an immigrant (103, no parent), a child of a parent of unknown house (105) and a house founded and
    extinct inside the window (104, seen by neither snapshot). The initial snapshot has no events."""
    cfg = _lineage_cfg(tiny_cfg)
    si = living_si(cfg, 12)
    si["clade"][10] = cfg.population.n_init  # row 10 is of an unknown house (an immigrant lineage)
    drv = SocDriver(cfg, si)
    assert drv.society.records == [] and drv.society.houses.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11]
    si1, a1 = _next(cfg, si, births=((20, 3, 100), (21, 4, 101)), deaths=(5, 6))
    r1 = drv.advance(si1, 250, a1)
    ev = r1["house_events"]
    assert ev["founded"] == {"id": [100, 101], "step": [250, 250], "parent_house": [3, 4]}
    assert ev["extinct"] == {"id": [5, 6], "step": [250, 250]}
    assert (ev["founded_dropped"], ev["extinct_dropped"]) == (0, 0)
    assert drv.society.houses.tolist() == [0, 1, 2, 3, 4, 7, 8, 9, 11, 100, 101]
    # window 2: a birth that dies before the snapshot leaves no trace (tick, then advance)
    si2a, a2a = _next(cfg, si1, births=((22, 21, 104),))
    drv.tick(si2a, a2a)
    si2, a2 = _next(cfg, si2a, births=((23, 20, 102), (24, 10, 105)), immigrants=((25, 103),), deaths=(22,))
    r2 = drv.advance(si2, 500, a2)
    ev = r2["house_events"]
    assert ev["founded"]["id"] == [102, 103, 105] and ev["founded"]["step"] == [500] * 3
    assert ev["founded"]["parent_house"] == [100, None, None]  # a grandchild; an immigrant; an unknown parent
    assert ev["extinct"] == {"id": [], "step": []}
    assert (ev["founded_dropped"], ev["extinct_dropped"]) == (0, 0)
    assert 104 not in drv.society.houses.tolist()
    assert r2["summary"]["houses"]["unknown_immigrant"] == 1  # row 10; its child founded a house of its own
    # nothing changes while nothing is born or dies
    r3 = drv.advance(si2, 750)
    assert r3["house_events"]["founded"]["id"] == [] and r3["house_events"]["extinct"]["id"] == []


def test_house_events_cap_and_overflow_count(tiny_cfg: Config) -> None:
    """At most ``events_max`` of each list, the lowest ids first, and the rest counted."""
    cfg = _lineage_cfg(tiny_cfg)
    sr = dataclasses.replace(SOCIAL_V1, events_max=2)
    si = living_si(cfg, 12)
    drv = SocDriver(cfg, si, sr=sr)
    si1, a1 = _next(cfg, si, births=tuple((20 + i, i, 100 + i) for i in range(5)), deaths=(5, 6, 7, 8))
    ev = drv.advance(si1, 250, a1)["house_events"]
    assert ev["founded"]["id"] == [100, 101] and ev["founded"]["parent_house"] == [0, 1]
    assert ev["extinct"]["id"] == [5, 6]
    assert (ev["founded_dropped"], ev["extinct_dropped"]) == (3, 2)
    off = dataclasses.replace(SOCIAL_V1, events_max=0)
    drv0 = SocDriver(cfg, si, sr=off)
    ev0 = drv0.advance(si1, 250, a1)["house_events"]
    assert ev0["founded"]["id"] == [] and (ev0["founded_dropped"], ev0["extinct_dropped"]) == (5, 4)


def test_house_events_continue_identically_after_a_restore(tiny_cfg: Config) -> None:
    """The set of houses alive at the last snapshot is in the sidecar: a restored society reports the same
    founded and extinct houses as the original."""
    cfg = _lineage_cfg(tiny_cfg)
    si = living_si(cfg, 12)
    drv = SocDriver(cfg, si)
    si1, a1 = _next(cfg, si, births=((20, 3, 100),), deaths=(5,))
    drv.advance(si1, 250, a1)
    restored, why = Society.from_arrays(drv.society.arrays(), cfg, SOCIAL_V1, step=250)
    assert why == "" and restored is not None
    assert restored.houses.tolist() == drv.society.houses.tolist()
    twin = SocDriver.__new__(SocDriver)
    twin.__dict__.update({**drv.__dict__, "society": restored})
    si2, a2 = _next(cfg, si1, births=((21, 20, 101),), deaths=(6, 20))
    r1, r2 = drv.advance(si2, 500, a2), twin.advance(si2, 500, a2)
    assert json.dumps(r1) == json.dumps(r2)
    assert r1["house_events"]["founded"]["id"] == [101] and r1["house_events"]["extinct"]["id"] == [6, 100]


# --- window counts, centroid, archetype mix -----------------------------------------------------------------


def test_window_counts_belong_to_the_group_not_the_slot(tiny_cfg: Config) -> None:
    """A's births (3) and deaths by cause are the window's counts of the slot it held; B ends at the snapshot
    and C is admitted into the slot B held: C has no window counts (None), not B's."""
    cfg = tiny_cfg
    rng = np.random.default_rng(6)
    sc = Scene(cfg)
    sc.add(cloud(rng, A_BIN, 12), clade=1)
    b = sc.add(cloud(rng, B_BIN, 13), clade=2)
    c = sc.add(lattice(0, 9), clade=3)
    lone = sc.add(lattice(20, 3), clade=4)
    drv = SocDriver(cfg, sc.si)
    slots0 = {int(i): s for s, i in enumerate(drv.society.kin_slots) if i >= 0}
    assert len(slots0) == 2
    # the window: three births to A (tags of the parent), one to B; deaths in A (2 starved, 1 killed), in B
    # (1 aged, 1 catastrophe) and of a lone row (starved)
    nb = sc.next
    si, acts = _next(
        cfg, sc.si, births=((nb, 0, 500), (nb + 1, 1, 501), (nb + 2, 2, 502), (nb + 3, b.start, 503))
    )
    si["tag"][nb : nb + 3] = sc.si["tag"][0:3]
    si["tag"][nb + 3] = sc.si["tag"][b.start]
    acts = _with_deaths(
        acts,
        {3: DeathCause.STARVED, 4: DeathCause.STARVED, 5: DeathCause.KILLED, b.start + 1: DeathCause.AGED}
        | {b.start + 2: DeathCause.CATASTROPHE, lone.start: DeathCause.STARVED},
    )
    for row in (3, 4, 5, b.start + 1, b.start + 2, lone.start):
        si["alive"][row], si["uid"][row] = False, -1
    si["tag"][b] = lattice(40, 13)  # B's rows scatter: B ends
    si["tag"][c] = cloud(rng, C_BIN, 9)  # C gathers
    rec = drv.advance(si, 250, acts)
    table = by_size(rec)
    assert sorted(table) == [9, 12]  # C and A (12 - 3 deaths + 3 births); B is gone
    row_a, row_c = table[12], table[9]
    assert row_a["births"] == 3 and row_a["deaths"] == [2, 1, 0, 0]
    assert row_c["births"] is None and row_c["deaths"] is None
    assert row_a["slot"] == slots0[row_a["id"]] and row_c["slot"] == next(
        s for s in slots0.values() if s != row_a["slot"]
    )
    assert DEATH_CAUSES == ("starved", "killed", "aged", "catastrophe")
    # the next window starts from zero
    r2 = drv.advance(si, 500)
    assert by_size(r2)[12]["births"] == 0 and by_size(r2)[12]["deaths"] == [0, 0, 0, 0]
    assert by_size(r2)[9]["births"] == 0  # C holds a slot in this window


def _with_deaths(acts: Any, causes: dict[int, DeathCause]) -> Any:
    dc = to_numpy(acts.death_cause).copy()
    for row, cause in causes.items():
        dc[row] = cause.value
    return acts._replace(death_cause=mx.array(dc))


@pytest.mark.parametrize("dim", ["2d", "3d"])
def test_centroid_and_archetype_mix(dim: str, tiny_cfg: Config, tiny3d_cfg: Config) -> None:
    """The centroid is the circular mean of the members' floor positions (a group on both sides of the seam
    sits at the seam, not mid-world), and ``arch`` the shares of the labelled members' primary archetypes in
    ``ARCHETYPE_NAMES`` order: unlabelled members are left out, a group with none has None."""
    cfg = tiny_cfg if dim == "2d" else tiny3d_cfg
    size = cfg.world.size
    rng = np.random.default_rng(7)
    nd = 3 if dim == "3d" else 2

    def pos(xs: list[float], y: float) -> np.ndarray:
        p = np.zeros((len(xs), nd))
        p[:, 0], p[:, 1] = xs, y
        if nd == 3:
            p[:, 2] = 5.0
        return p

    xs = [1.0] * 4 + [size - 3.0] * 4 + [10.0] * 4
    sc = Scene(cfg)
    a = sc.add(cloud(rng, A_BIN, 12), pos=pos(xs, 40.0))
    sc.add(cloud(rng, B_BIN, 9), pos=pos([64.0] * 9, 64.0))
    c = sc.add(cloud(rng, C_BIN, 10), pos=pos([20.0] * 10, 100.0))
    arch = np.zeros(1024, np.int32)
    arch[a] = [1] * 6 + [2] * 3 + [7] * 1 + [0] * 2  # 10 labelled: 0.6, 0.3, ..., 0.1
    arch[c] = [3] * 7 + [4] * 3  # 0.7 provider, 0.3 trader
    drv = SocDriver(cfg, sc.si)
    rec = drv.advance(sc.si, 250, arch=arch)
    table = by_size(rec)
    ra, rb, rc = table[12], table[9], table[10]
    assert len(ARCHETYPE_NAMES) == 10 and ARCHETYPE_NAMES == OTHER_ARCH_NAMES
    assert ra["arch"] == [0.6, 0.3, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0]
    assert rb["arch"] is None
    assert rc["arch"] == [0.0, 0.0, 0.7, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # the centroid: the unit-vector mean of the angles, not the arithmetic mean (which would be ~ 33 here)
    ang = np.array(xs) * (2 * math.pi / size)
    want = (math.atan2(np.sin(ang).sum(), np.cos(ang).sum()) % (2 * math.pi)) * size / (2 * math.pi)
    assert abs(ra["x"] - want) < 0.006 and (ra["x"] < 10 or ra["x"] > size - 10), (ra["x"], want)
    assert ra["y"] == 40.0 and rb["x"] == 64.0 and rb["y"] == 64.0 and rc["x"] == 20.0 and rc["y"] == 100.0
    # a record without labels (the labels tool, direct callers): every mix is None
    drv2 = SocDriver(cfg, sc.si)
    rec2 = drv2.advance(sc.si, 250)
    assert rec2["kin_groups"]["arch"] == [None] * 3
    assert rec2["kin_groups"]["x"] == rec["kin_groups"]["x"]


# --- the observer, on real steps -----------------------------------------------------------------------


@pytest.mark.parametrize("dim", ["2d", "3d"])
def test_the_roster_of_a_real_run_matches_a_numpy_reference(
    dim: str, tiny_cfg: Config, tiny3d_cfg: Config
) -> None:
    """Planted tag clouds track three kin groups through 60 steps with a snapshot every 15 and labels every
    5 steps. In every record, per listed group: the births and deaths by cause equal the sums over the window
    of an independent reference from the channel and the observer's slots at the start of each step (None for
    a group with no slot in the window); the archetype mix is the numpy mix of the labels of the group's rows
    at the snapshot; the centroid is their circular mean; the names never change for an id and are distinct;
    over the run the reference's counts are the engine's counters."""
    cfg = apply_overrides(tiny_cfg if dim == "2d" else tiny3d_cfg, ["game.observe=true"])
    sr = dataclasses.replace(SOCIAL_V1, snapshot_every=15)
    rules = dataclasses.replace(RULES_V1, every=5, arch_warm=10, intent_warm=10)
    state = set_output_bias(planted_state(cfg), cfg, BIAS)
    fn = make_step(cfg, observe=True)
    obs = Observer.fresh(state, cfg, rules=rules, social=sr)
    g, nc = sr.slots + 1, len(DEATH_CAUSES)
    win_b, win_d = np.zeros(g, np.int64), np.zeros((nc, g), np.int64)
    names: dict[int, str] = {}
    totals = {"births": 0, "deaths": 0, "records": 0, "labelled": 0, "tracked": 0}
    engine, ref_total = np.zeros(5, np.int64), np.zeros(5, np.int64)  # births, then deaths by cause
    for t in range(1, 61):
        window_ids = obs.society.kin_slots.copy()  # the slot -> id map of the window that this step is in
        prev_kin = to_numpy(obs.so.kin)
        state, out = fn(state)
        obs.update(state, out.acts, t)
        mx.eval(state, out.base(), *obs.arrays())
        an = {f: to_numpy(getattr(out.acts, f)) for f in out.acts._fields if getattr(out.acts, f) is not None}
        b, d = life_reference(prev_kin, an, g, nc)
        win_b, win_d = win_b + b, win_d + d
        c = to_numpy(out.counters)
        engine += c[[C.BIRTHS, C.DEATHS_STARVED, C.DEATHS_KILLED, C.DEATHS_AGED, C.DEATHS_CATASTROPHE]]
        ref_total += np.array([b.sum(), *d.sum(axis=1)])
        records = obs.take_records()
        if not records:
            continue
        (rec,) = records
        assert rec["step"] == t and t % 15 == 0
        kg = rec["kin_groups"]
        lab = obs.host_labels()["arch_primary"]
        slot_of_row = to_numpy(obs.social_labels().kin_slot)
        pos = to_numpy(state.pos)
        win = {int(i): s for s, i in enumerate(window_ids[: sr.slots]) if i >= 0}
        totals["records"] += 1
        totals["tracked"] += len(kg["id"])
        for i, gid in enumerate(kg["id"]):
            assert names.setdefault(gid, kg["name"][i]) == kg["name"][i]
            rows = np.flatnonzero(slot_of_row == kg["slot"][i])
            assert len(rows) == kg["size"][i]
            labelled = lab[rows][lab[rows] > 0]
            want_arch = (
                [
                    round(float((labelled == a).sum()) / len(labelled), 3)
                    for a in range(1, len(ARCHETYPE_NAMES) + 1)
                ]
                if len(labelled)
                else None
            )
            assert kg["arch"][i] == want_arch, (t, gid)
            totals["labelled"] += int(len(labelled) > 0)
            ang = pos[rows, 0].astype(np.float64) * (2 * math.pi / cfg.world.size)
            cx = (math.atan2(np.sin(ang).sum(), np.cos(ang).sum()) % (2 * math.pi)) * cfg.world.size
            cx = cx / (2 * math.pi)
            assert abs(kg["x"][i] - cx) < 0.006 or abs(abs(kg["x"][i] - cx) - cfg.world.size) < 0.006, (
                t,
                gid,
            )
            if gid in win:
                assert kg["births"][i] == win_b[win[gid]] and kg["deaths"][i] == win_d[:, win[gid]].tolist()
                totals["births"] += int(kg["births"][i])
                totals["deaths"] += sum(kg["deaths"][i])
            else:
                assert kg["births"][i] is None and kg["deaths"][i] is None
        assert len(set(kg["name"])) == len(kg["name"])
        win_b, win_d = np.zeros_like(win_b), np.zeros_like(win_d)
    assert totals["records"] == 4 and totals["tracked"] == 12  # three groups at each of four snapshots
    assert totals["labelled"] > 0  # some group had labelled members
    assert totals["births"] > 0 and totals["deaths"] > 0  # and some window counts are not zero
    assert np.array_equal(engine, ref_total) and engine[0] > 0 and len(names) == 3
