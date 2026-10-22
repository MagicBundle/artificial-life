"""§29.4-§29.8 the host half of the social layer (G2): kin groups, colonies, houses, territory, relations,
temperament and depth bands from planted scenes, the head and the record, and the sidecar form."""

from __future__ import annotations

import dataclasses
import itertools
import json
from typing import Any

import mlx.core as mx
import numpy as np
import pytest

from alife import layout, metrics, runio
from alife.analysis import phylo
from alife.config import Config, apply_overrides, dims, load_config
from alife.game.rules import INTENT_VERSION
from alife.game.social import HOUSE_UNKNOWN, SOCIAL_V1, Expo, Rel, SocialRules, soc_in
from alife.game.society import (
    CODA,
    NUCLEUS,
    ONSET,
    Society,
    house_names,
    raw_house_names,
)
from alife.state import init_state, to_numpy
from alife.step import ActFlag, make_step
from game_helpers import SocDriver, blank_acts, living_si, make_acts, window_reference
from helpers import set_output_bias
from test_game_social import BIAS

T = layout.T
F = SOCIAL_V1.slots


def center(*b: int) -> np.ndarray:
    """The centre of a tag bin (width 1/8 over [-1, 1]) from its four digits."""
    return -1.0 + (np.asarray(b, np.float64) + 0.5) / 8.0


def cloud(rng: np.random.Generator, bin_: tuple[int, int, int, int], n: int) -> np.ndarray:
    """``n`` tags inside one bin."""
    return (center(*bin_) + rng.uniform(-0.04, 0.04, (n, T))).astype(np.float32)


def np_(a: mx.array) -> np.ndarray:
    return to_numpy(a)


class Scene:
    """A planted state: rows are added in groups with their tags; positions are random on the floor."""

    def __init__(self, cfg: Config, seed: int = 0) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.si = living_si(cfg, 0)
        self.next = 0

    def add(self, tags: np.ndarray, *, pos: np.ndarray | None = None, clade: int | None = None) -> slice:
        n = len(tags)
        lo, hi = self.next, self.next + n
        assert hi <= dims(self.cfg).n
        self.si["alive"][lo:hi] = True
        self.si["uid"][lo:hi] = np.arange(lo, hi)
        self.si["clade"][lo:hi] = np.arange(lo, hi) if clade is None else clade
        self.si["tag"][lo:hi] = tags
        size = self.cfg.world.size
        d = dims(self.cfg)
        p = self.rng.uniform(0.0, size, (n, d.nd)) if pos is None else pos
        self.si["pos"][lo:hi] = p
        self.next = hi
        self.si["next_uid"] = np.array(hi, np.int64)
        return slice(lo, hi)

    def bond(self, rows: list[int]) -> None:
        """A chain of bonds along ``rows``."""
        for a, b in itertools.pairwise(rows):
            self.si["partner"][a, 0], self.si["partner"][b, 1] = b, a

    def copy_si(self) -> dict[str, np.ndarray]:
        return {k: v.copy() for k, v in self.si.items()}


@pytest.fixture(scope="module")
def big_cfg() -> Config:
    """A 2-D config with 40,960 rows, for the thresholds that scale with the population (``n_ref >> 12``)."""
    return load_config(
        preset="tiny",
        overrides=[
            "population.n_max=40960",
            "population.n_init=39000",
            "population.max_births_per_step=1024",
            "brain.weight_dtype=float32",
        ],
    )


# --- kin groups on planted tags -----------------------------------------------------------------------

CLOUDS = {  # name -> (bin, rows)
    "A": ((2, 2, 2, 2), 400),
    "B": ((12, 12, 12, 12), 150),
    "C": ((2, 12, 2, 12), 100),
    "E": ((13, 2, 13, 2), 40),  # a candidate between the leave (32) and enter (64) thresholds at n = 33,000
    "F": ((6, 13, 6, 13), 20),  # under the leave threshold
}
CHAIN = [
    (b, 6, 6, 6) for b in range(4, 10)
]  # six adjacent bins of 12 rows: one group of 72 by single linkage


def _planted_kin_scene(cfg: Config, n_alive: int, seed: int = 0) -> tuple[Scene, dict[str, slice]]:
    """The planted clouds, the chain, 100 lone rows and fillers (one per bin, away from every planted bin and
    its face neighbours) up to ``n_alive`` rows."""
    rng = np.random.default_rng(seed)
    sc = Scene(cfg, seed)
    rows: dict[str, slice] = {}
    for name, (bin_, n) in CLOUDS.items():
        rows[name] = sc.add(cloud(rng, bin_, n))
    rows["D"] = sc.add(np.concatenate([cloud(rng, b, 12) for b in CHAIN]))
    planted = {np.ravel_multi_index(b, (16,) * 4) for b, _ in CLOUDS.values()}
    planted |= {np.ravel_multi_index(b, (16,) * 4) for b in CHAIN}
    forbidden = set(planted)
    for code in planted:  # the face neighbours of the planted bins (a sparse bin there would attach)
        idx = np.unravel_index(code, (16,) * 4)
        for ax in range(4):
            for s in (-1, 1):
                nb = list(idx)
                nb[ax] += s
                if 0 <= nb[ax] < 16:
                    forbidden.add(int(np.ravel_multi_index(tuple(nb), (16,) * 4)))
    free = np.array([c for c in rng.permutation(16**4) if c not in forbidden])
    n_lone, n_fill = 100, n_alive - sc.next - 100
    codes = free[: n_lone + n_fill]
    digits = np.stack(np.unravel_index(codes, (16,) * 4), axis=1)
    tags = (-1.0 + (digits + rng.uniform(0.1, 0.9, digits.shape)) / 8.0).astype(np.float32)
    rows["lone"] = sc.add(tags[:n_lone])
    rows["fill"] = sc.add(tags[n_lone:])
    return sc, rows


def test_kin_groups_on_planted_tags(big_cfg: Config) -> None:
    """Three tight clouds, a chain of bins, a candidate under the enter threshold, one under the leave
    threshold, lone rows and fillers: groups, the ``n_ref >> 12/9/10`` thresholds, slots by size, OTHER for
    the rest; the device ``kin`` after ``relabel`` equals the host mapping row by row."""
    cfg, sr = big_cfg, SOCIAL_V1
    sc, rows = _planted_kin_scene(cfg, 33_000)
    drv = SocDriver(cfg, sc.si)
    head = drv.society.head()
    assert (
        head["kin_groups"]["tracked"] == 4
    )  # A (400), B (150), C (100), D (72); E (40) and F (20) get no id
    ntot = int(sc.si["alive"].sum())
    assert ntot == 33_000 and head["kin_groups"]["other"] == ntot - (400 + 150 + 100 + 72)
    want = np.full(dims(cfg).n, -1, np.int32)
    want[:ntot] = sr.slots  # OTHER
    for slot, name in enumerate(["A", "B", "C", "D"]):  # by size: 400, 150, 100, 72
        want[rows[name]] = slot
    assert np.array_equal(drv.kin_slot_rows(), want)
    assert drv.kin_lut[np.ravel_multi_index(CLOUDS["A"][0], (16,) * 4)] == 0
    rec = drv.advance(sc.si, 250)
    assert rec["thresholds"] == {
        "n_ref": 33_000,
        "rescaled": False,
        "occ_min": 8,
        "kin_enter": 64,
        "kin_leave": 32,
    }
    kg = rec["kin_groups"]
    assert kg["slot"] == [0, 1, 2, 3] and kg["size"] == [400, 150, 100, 72] and kg["id"] == [1, 2, 3, 4]
    assert [len(v) for v in kg.values()] == [4] * len(kg)
    assert rec["events"]["id"] == [] and rec["event_counts"]["kin_group"] == {
        "born": 0, "split": 0, "merged": 0, "dissolved": 0,
    }  # fmt: skip
    assert np.array_equal(drv.kin_slot_rows(), want)  # unchanged by the second snapshot (sticky slots)
    # the thresholds at other populations (n_ref >> 12, >> 9, >> 10)
    for n_ref, (occ, enter, leave) in {5_000: (2, 9, 4), 12_000: (2, 23, 11), 40_000: (9, 78, 39)}.items():
        o = max(sr.kin_occ[0], n_ref >> sr.kin_occ[1])
        e = max(sr.kin_enter[0], n_ref >> sr.kin_enter[1])
        lv = max(sr.kin_leave[0], n_ref >> sr.kin_leave[1])
        assert (o, e, lv) == (occ, enter, leave), n_ref


def test_thresholds_follow_the_reference_population(big_cfg: Config) -> None:
    """The planted tags with 32,767 and then 32,768 alive rows (a row added far from the clouds) change no
    threshold, assignment, slot or event; a 20% change rescales once, with ``thresholds.rescaled`` true."""
    cfg = big_cfg
    sc, _ = _planted_kin_scene(cfg, 32_767)
    drv = SocDriver(cfg, sc.si)
    ref = drv.society.n_ref
    assert ref == 32_767
    first = drv.advance(sc.si, 250)
    assert first["thresholds"]["n_ref"] == 32_767 and first["thresholds"]["occ_min"] == 7  # 32,767 >> 12
    slots0 = drv.kin_slot_rows()[:32_767].copy()
    # add one lone row in a free bin: 32,768 alive rows, where n >> 12 steps from 7 to 8
    si2 = sc.copy_si()
    used = {
        int(c)
        for c in np.ravel_multi_index(
            np.clip(np.floor((si2["tag"][:32_767] + 1) * 8), 0, 15).astype(int).T, (16,) * 4
        )
    }
    free = next(
        c
        for c in range(16**4)
        if c not in used and c not in {u + d for u in used for d in (-1, 1, -16, 16, -256, 256, -4096, 4096)}
    )
    digits = np.unravel_index(free, (16,) * 4)
    si2["alive"][32_767], si2["uid"][32_767], si2["clade"][32_767] = True, 32_767, 32_767
    si2["tag"][32_767] = -1.0 + (np.array(digits) + 0.5) / 8.0
    si2["next_uid"] = np.array(32_768, np.int64)
    second = drv.advance(si2, 500)
    assert second["thresholds"] == first["thresholds"] | {"n_ref": 32_767}  # the reference stays at 32,767
    assert second["thresholds"]["rescaled"] is False and second["thresholds"]["occ_min"] == 7
    assert second["events"]["id"] == [] and second["kin_groups"]["id"] == first["kin_groups"]["id"]
    assert second["kin_groups"]["slot"] == first["kin_groups"]["slot"]
    assert np.array_equal(drv.kin_slot_rows()[:32_767], slots0)
    # and back to 32,767: nothing changes either
    third = drv.advance(sc.si, 750)
    assert third["thresholds"]["n_ref"] == 32_767 and third["events"]["id"] == []
    # a 20% change of the population rescales once: more fillers (one per free bin)
    more = _planted_kin_scene(cfg, 39_320)[0]
    fourth = drv.advance(more.si, 1000)
    th = fourth["thresholds"]
    assert th["rescaled"] is True and th["n_ref"] == 39_320
    assert (th["occ_min"], th["kin_enter"], th["kin_leave"]) == (9, 76, 38)  # n >> 12, >> 9, >> 10
    fifth = drv.advance(more.si, 1250)
    assert fifth["thresholds"]["rescaled"] is False and fifth["thresholds"]["n_ref"] == 39_320
    assert fifth["events"]["id"] == [] and fifth["kin_groups"]["id"] == fourth["kin_groups"]["id"]
    # the edge of the deadband: a change of exactly n_ref >> 3 = 4,915 keeps the reference, one more moves it
    edge = _planted_kin_scene(cfg, 39_320 - 4_915)[0]
    assert drv.advance(edge.si, 1500)["thresholds"] == fifth["thresholds"]
    edge = _planted_kin_scene(cfg, 39_320 - 4_916)[0]
    sixth = drv.advance(edge.si, 1750)["thresholds"]
    assert sixth["rescaled"] is True and sixth["n_ref"] == 39_320 - 4_916


def _events(rec: dict[str, Any]) -> list[tuple[str, str, int, int]]:
    ev = rec["events"]
    return list(zip(ev["family"], ev["kind"], ev["id"], ev["other"], strict=True))


def _cloud_scene(cfg: Config, sizes: dict[str, int], seed: int = 0) -> dict[str, np.ndarray]:
    """Rows of named clouds at fixed row ranges; ``sizes`` rows of each are alive (the first ones)."""
    layout_ = {"A": (0, (2, 2, 2, 2)), "B": (20, (12, 12, 12, 12)), "D": (40, (2, 12, 2, 12))}
    rng = np.random.default_rng(seed)
    si = living_si(cfg, 0)
    for name, n in sizes.items():
        lo, bin_ = layout_[name]
        si["alive"][lo : lo + n] = True
        si["uid"][lo : lo + n] = np.arange(lo, lo + n)
        si["clade"][lo : lo + n] = np.arange(lo, lo + n)
        si["tag"][lo : lo + n] = cloud(np.random.default_rng(lo), bin_, 12)[:n]
        si["pos"][lo : lo + n] = rng.uniform(0.0, cfg.world.size, (n, dims(cfg).nd))
    si["next_uid"] = np.array(100, np.int64)
    return si


def test_kin_group_hysteresis(tiny_cfg: Config) -> None:
    """A planted cloud resized across the enter (8) and leave (4) thresholds over 8 snapshots is BORN once at
    enter, raises no event while at or above leave, is DISSOLVED once below leave, and comes back under a new
    id; a second cloud oscillating +-1 around enter after entering emits nothing, and a third oscillating
    around leave without reaching enter is never tracked; sticky slots keep the device ``kin`` values."""
    cfg = tiny_cfg
    plan = [  # (A, B, D) alive rows per snapshot; step 0 is the initial snapshot
        (0, 9, 0), (6, 7, 5), (9, 9, 3), (6, 7, 5), (4, 9, 3), (3, 7, 5), (3, 9, 3), (9, 7, 5),
    ]  # fmt: skip
    drv = SocDriver(cfg, _cloud_scene(cfg, {"A": plan[0][0], "B": plan[0][1], "D": plan[0][2]}))
    assert drv.society.head()["kin_groups"]["tracked"] == 1
    events = []
    for i, (a, b, d) in enumerate(plan[1:], start=1):
        rec = drv.advance(_cloud_scene(cfg, {"A": a, "B": b, "D": d}), 250 * i)
        events.append(_events(rec))
        kin = drv.kin_slot_rows()
        assert (kin[20 : 20 + b] == 0).all(), i  # B keeps its slot throughout
        assert 1 in rec["kin_groups"]["id"]  # B's id is never lost
    # A: nothing at 6, BORN at 9 (snapshot 2), nothing at 6 and 4, DISSOLVED at 3 (snapshot 5), nothing at 3,
    # BORN again at 9 (snapshot 7) under a new id; B and D: no event at any snapshot
    assert events[0] == [] and events[2] == [] and events[3] == []
    assert events[1] == [("kin_group", "born", 2, -1)]
    assert events[4] == [("kin_group", "dissolved", 2, -1)]
    assert events[5] == []
    assert events[6] == [("kin_group", "born", 3, -1)]  # a new id: an ended id is never revived
    assert drv.society.head()["kin_groups"]["tracked"] == 2  # B and the new A
    # D never reaches the enter threshold: no id, its rows are OTHER
    assert (drv.kin_slot_rows()[40:45] == SOCIAL_V1.slots).all()
    # A's second life took the lowest free slot (its first slot was freed when it ended)
    assert (drv.kin_slot_rows()[0:9] == 1).all()


# --- colonies and bodies ------------------------------------------------------------------------------


def _chain_scene(cfg: Config, lengths: dict[int, int], cuts: tuple[int, ...] = ()) -> dict[str, np.ndarray]:
    """Bond chains of ``lengths[start_row]`` rows from each start row (one per 10 rows), 100 rows alive; the
    bonds *to* the rows in ``cuts`` are removed. Rows 0-99 are in the tag bin P (a kin group), except the
    last three rows of the chain of 9 (and ten more rows, 90-99), which are in the bin Q."""
    si = living_si(cfg, 100)
    rng = np.random.default_rng(1)
    si["tag"][:100] = cloud(rng, (2, 2, 2, 2), 100)
    si["tag"][90:100] = cloud(rng, (12, 12, 12, 12), 10)
    si["tag"][36:39] = cloud(rng, (12, 12, 12, 12), 3)  # three of the nine members belong to Q
    for k, (start, n) in enumerate(lengths.items()):
        for j in range(n):
            si["pos"][start + j] = [20.0 * k + 5.0 + 0.5 * j, 30.0]
        for a, b in zip(range(start, start + n - 1), range(start + 1, start + n), strict=True):
            if b not in cuts:
                si["partner"][a, 0], si["partner"][b, 1] = b, a
    return si


def test_sticky_slots_through_the_society(tiny_cfg: Config) -> None:
    """With all 16 slots taken, a bigger group born later gets no slot while the old groups still rank in
    the top 20 (the slack), so no slotted group is displaced; when an old group ends its slot goes to the
    newcomer at the next snapshot."""
    cfg = tiny_cfg
    sizes = [20 + 2 * i for i in range(16)]  # 20..50 rows, distinct

    def scene(newcomer: int, dropped: int = -1) -> dict[str, np.ndarray]:
        sc = Scene(cfg)
        for i, n in enumerate(sizes):
            sc.add(cloud(rng2(i), (3 * (i % 5), 3 * (i // 5), 2, 2), n))
        if dropped >= 0:  # that group's rows are gone
            lo = sum(sizes[:dropped])
            sc.si["alive"][lo : lo + sizes[dropped]] = False
            sc.si["uid"][lo : lo + sizes[dropped]] = -1
        if newcomer:
            sc.next = 600
            sc.add(cloud(rng2(99), (14, 14, 14, 14), newcomer))
        return sc.si

    def rng2(i: int) -> np.random.Generator:
        return np.random.default_rng(100 + i)

    drv = SocDriver(cfg, scene(0))
    assert drv.society.head()["kin_groups"]["tracked"] == 16
    slots0 = drv.kin_slot_rows().copy()
    assert sorted(set(slots0[slots0 >= 0].tolist())) == list(range(16))  # all 16 slots taken
    assert slots0[sum(sizes[:15])] == 0 and slots0[0] == 15
    rec = drv.advance(scene(120), 250)  # a group of 120 appears: it ranks first, but every slot is taken
    assert _events(rec) == [("kin_group", "born", 17, -1)]
    kin = drv.kin_slot_rows()
    assert np.array_equal(kin[:600], slots0[:600])  # nobody lost a slot to it (the slack of 4)
    assert (kin[600:720] == F).all()  # the newcomer's rows are OTHER
    assert rec["kin_groups"]["id"] == list(range(1, 17))  # the slotted ids by slot: unchanged
    assert drv.society.head()["kin_groups"]["tracked"] == 17
    # the smallest old group (slot 15) ends: the newcomer takes the free slot at the next snapshot
    drv.advance(scene(120, dropped=0), 500)
    kin2 = drv.kin_slot_rows()
    assert (kin2[600:720] == 15).all() and (kin2[:20] == -1).all()


def test_colonies_and_body(tiny_cfg: Config) -> None:
    """Bond chains of 2, 3, 4 and 9: the 4 and the 9 get an id at entry; a tracked 4 that loses a member stays
    tracked at 3 and ends at 2; the chain of 3 is a candidate that never gets an id; slots, ``body`` and
    ``col_lut`` follow; a colony's kin group is the plurality of its members' groups."""
    cfg = tiny_cfg
    lengths = {0: 2, 10: 3, 20: 4, 30: 9}
    drv = SocDriver(cfg, _chain_scene(cfg, lengths))
    head = drv.society.head()["colonies"]
    assert head == {"tracked": 2, "members": 13, "largest": 9, "converged": True, "match": "uid"}
    # slots by size: the 9 (root row 30) is slot 0, the 4 (root row 20) slot 1; the 3 and the 2 have none
    assert drv.col_lut[30] == 0 and drv.col_lut[20] == 1 and drv.col_lut[10] == -1 and drv.col_lut[0] == -1
    colony = np_(drv.so.colony)
    assert (colony[30:39] == 0).all() and (colony[20:24] == 1).all()
    assert (colony[:20] == -1).all() and (colony[39:100] == -1).all() and (colony[100:] == -1).all()
    body = np_(drv.so.body)
    want = np.where(np.arange(len(body)) < 100, 1, 0)
    for start, n in lengths.items():
        want[start : start + n] = n
    assert np.array_equal(body, want)
    rec = drv.advance(_chain_scene(cfg, lengths), 250)
    col = rec["colonies"]
    assert col["id"] == [1, 2] and col["slot"] == [0, 1] and col["size"] == [9, 4]
    # the kin plurality: six of the nine members and all four of the others are in the big group P (id 1)
    kin = {g: i for g, i in zip(rec["kin_groups"]["slot"], rec["kin_groups"]["id"], strict=True)}
    assert col["kin_group"] == [kin[0], kin[0]] and set(rec["kin_groups"]["id"]) == {1, 2}
    assert col["x"][1] == pytest.approx(20 * 2 + 5.0 + 0.75, abs=0.01) and col["y"] == [30.0, 30.0]
    assert rec["events"]["id"] == []
    # one member leaves the tracked 4: three alive bonded rows, the colony stays tracked
    rec = drv.advance(_chain_scene(cfg, lengths, cuts=(23,)), 500)
    assert rec["events"]["id"] == [] and rec["colonies"]["size"] == [9, 3]
    assert drv.col_lut[20] == 1 and (np_(drv.so.body)[20:23] == 3).all() and np_(drv.so.body)[23] == 1
    # then another: two rows are no candidate, the colony ends (DISSOLVED), its slot frees
    rec = drv.advance(_chain_scene(cfg, lengths, cuts=(22, 23)), 750)
    assert _events(rec) == [("colony", "dissolved", 2, -1)] and rec["colonies"]["id"] == [1]
    assert drv.col_lut[20] == -1 and (np_(drv.so.colony)[20:24] == -1).all()
    # the chain of 3 grows to 4 rows: BORN, under a new id, in the freed slot
    grown = _chain_scene(cfg, {**lengths, 10: 4}, cuts=(22, 23))
    rec = drv.advance(grown, 1000)
    assert _events(rec) == [("colony", "born", 3, -1)]
    assert drv.col_lut[10] == 1 and rec["colonies"]["id"] == [1, 3]


def test_match_mode_follows_the_spacing(tiny_cfg: Config) -> None:
    """``snapshot_every`` 250 matches by uid (with the newborns credited to their parents), 301 by footprint
    (the tag bins of a group, the floor cells of a colony); both run end to end, and a full turnover of the
    uids at the same places ends the systems in uid mode and keeps them in footprint mode."""
    cfg = tiny_cfg

    def scene(uid_shift: int, parent: bool) -> dict[str, np.ndarray]:
        si = living_si(cfg, 40)
        rng = np.random.default_rng(3)
        si["tag"][:20] = cloud(rng, (2, 2, 2, 2), 20)
        si["tag"][20:40] = cloud(rng, (12, 12, 12, 12), 20)
        si["pos"][:40] = np.stack([np.arange(40) % 20 + 0.5, np.arange(40) // 20 * 10 + 0.5], axis=1)
        for a in range(0, 5):
            si["partner"][a, 0], si["partner"][a + 1, 1] = a + 1, a
        si["uid"][:40] += uid_shift
        if parent:
            si["parent_uid"][:40] = np.arange(40)
        si["next_uid"] = np.array(uid_shift + 40, np.int64)
        return si

    for every, mode in ((250, "uid"), (300, "uid"), (301, "footprint")):
        sr = dataclasses.replace(SOCIAL_V1, snapshot_every=every)
        assert (every <= sr.uid_spacing) == (mode == "uid")
        drv = SocDriver(cfg, scene(0, False), sr=sr)
        assert drv.society.head()["colonies"]["match"] == mode
        assert drv.society.head()["kin_groups"]["tracked"] == 2
        same = drv.advance(scene(0, False), every)
        assert same["events"]["id"] == [] and same["summary"]["colonies"]["match"] == mode
        # every agent replaced by a child (new uid, parent uid = the old one) in the same place
        child = drv.advance(scene(1000, True), 2 * every)
        assert child["events"]["id"] == [], mode  # credit (uid mode) or the same footprint: the ids continue
        # every agent replaced by a stranger (new uid, no parent) in the same place
        stranger = drv.advance(scene(2000, False), 3 * every)
        kinds = sorted({(f, k) for f, k, *_ in _events(stranger)})
        if mode == "uid":  # nothing links the new uids to the old: all three systems end and are born again
            assert kinds == [
                ("colony", "born"),
                ("colony", "dissolved"),
                ("kin_group", "born"),
                ("kin_group", "dissolved"),
            ]
            assert len(_events(stranger)) == 6
        else:  # the places are the same
            assert kinds == [] and stranger["colonies"]["id"] == same["colonies"]["id"]
    # the credit is what keeps the ids across the turnover in uid mode: without parents they end
    sr = SOCIAL_V1
    drv = SocDriver(cfg, scene(0, False), sr=sr)
    assert len(_events(drv.advance(scene(1000, False), 250))) == 6


def test_the_first_newborn_of_a_window_is_credited(tiny_cfg: Config) -> None:
    """Credit goes to every newborn with ``uid >= next_uid`` of the previous snapshot, the first one included:
    a group of 20 that keeps 6 members and gains one child (uid = the previous ``next_uid``) is J = 7/21 with
    the child credited to its parent and 6/21 without, which is under theta = 0.3."""
    cfg = tiny_cfg
    rng = np.random.default_rng(5)
    tags = cloud(rng, (2, 2, 2, 2), 21)

    def scene(alive_old: int, with_child: bool, child_parent: int = -1) -> dict[str, np.ndarray]:
        si = living_si(cfg, 0)
        si["alive"][:alive_old] = True
        si["uid"][:alive_old] = np.arange(alive_old)
        si["tag"][:alive_old] = tags[:alive_old]
        si["clade"][:21] = np.arange(21)
        si["next_uid"] = np.array(20, np.int64)
        if with_child:  # row 20, uid 20: the first uid issued after the previous snapshot
            si["alive"][20], si["uid"][20], si["parent_uid"][20], si["tag"][20] = (
                True,
                20,
                child_parent,
                tags[20],
            )
            si["next_uid"] = np.array(21, np.int64)
        return si

    # the child of agent 5 is credited to agent 5's group; a stranger with the same uid is not
    for parent, events in ((5, []), (-1, [("kin_group", "dissolved", 1, -1)])):
        drv = SocDriver(cfg, scene(20, False))
        assert drv.society.prev_next_uid == 20
        assert _events(drv.advance(scene(6, True, parent), 250)) == events, parent
    # without any newborn the same six members are J = 6/20 = 0.3, on the inclusive edge of theta
    drv = SocDriver(cfg, scene(20, False))
    assert _events(drv.advance(scene(6, False), 250)) == []


# --- similarity, relations and temperament ------------------------------------------------------------


def _sim_scene(cfg: Config, extra_a: int = 0) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Group A (10 rows, a tight cloud), B (12 rows) and C (a chain of six bins, 3 rows each); a bond A0-B0.
    Returns ``(si, tags per group)``; A has ``extra_a`` more rows (the subsample test)."""
    rng = np.random.default_rng(7)
    sc = Scene(cfg, 7)
    a = sc.add(cloud(rng, (2, 2, 2, 2), 10 + extra_a))
    b = sc.add(cloud(rng, (12, 12, 12, 12), 12))
    c = sc.add(np.concatenate([cloud(rng, (4 + j, 6, 6, 6), 3) for j in range(6)]))
    sc.si["partner"][a.start, 0], sc.si["partner"][b.start, 1] = b.start, a.start
    return sc.si, {"A": sc.si["tag"][a].copy(), "B": sc.si["tag"][b].copy(), "C": sc.si["tag"][c].copy()}


def _kernel(cfg: Config, ta: np.ndarray, tb: np.ndarray, same: bool) -> float:
    d2 = ((ta[:, None, :].astype(np.float64) - tb[None, :, :].astype(np.float64)) ** 2).sum(-1)
    k = np.exp(-d2 / cfg.sensing.tag_scale**2)
    if same:
        n = len(ta)
        return round((k.sum() - n) / (n * (n - 1)), 4)
    return round(float(k.mean()), 4)


def _attack_acts(cfg: Config, si: dict[str, np.ndarray], extra_a: int = 0):
    """A0 attacks B0 (seeing B0, C0 and A1), A1 attacks C0 (seeing C0 and B1), B1 attacks A2 (seeing A2); the
    bond A0-B0 carries 7 quanta from A0 to B0; every row of A prefers kin to attack (q +1.0) and not to bond
    (q -0.5)."""
    a0, a1, a2 = 0, 1, 2
    b0, b1 = 10 + extra_a, 11 + extra_a
    c0 = 22 + extra_a
    acts = blank_acts(cfg)
    f = acts["flags"]
    f[a0] = f[a1] = f[b1] = ActFlag.ATTACKED.value
    acts["attack_target"][[a0, a1, b1]] = [b0, c0, a2]
    acts["nbr"][a0, :3] = [b0, c0, a1]
    acts["nbr"][a1, :2] = [c0, b1]
    acts["nbr"][b1, :1] = [a2]
    acts["share_out"][a0, 0] = 7
    acts["q_kin"][: 10 + extra_a] = [256, -128]
    return make_acts(acts)


def test_similarity_relations_and_temperament(tiny_cfg: Config) -> None:
    """The record's relations, ``sim_self`` and temperament against brute-force numpy on a planted scene."""
    cfg = tiny_cfg
    si, tags = _sim_scene(cfg)
    drv = SocDriver(cfg, si)
    # sizes: C 18 (id 1, slot 0), B 12 (id 2, slot 1), A 10 (id 3, slot 2)
    assert drv.society.kin_slots[:3].tolist() == [1, 2, 3]
    rec = drv.advance(si, 250, _attack_acts(cfg, si))
    kg = rec["kin_groups"]
    assert kg["id"] == [1, 2, 3] and kg["size"] == [18, 12, 10]
    sim_self = {"C": _kernel(cfg, tags["C"], tags["C"], True), "B": _kernel(cfg, tags["B"], tags["B"], True),
                "A": _kernel(cfg, tags["A"], tags["A"], True)}  # fmt: skip
    assert kg["sim_self"] == pytest.approx([sim_self["C"], sim_self["B"], sim_self["A"]], abs=1e-4)
    assert sim_self["C"] + 0.2 < min(sim_self["A"], sim_self["B"])  # a chained group is less self-similar
    # relations: B->A, A->C, A->B (A has id 3, B id 2, C id 1), listed by (a, b)
    r = rec["relations"]
    assert list(zip(r["a"], r["b"], strict=True)) == [(2, 3), (3, 1), (3, 2)]
    assert r["attacks"] == [1, 1, 1] and r["hits"] == [0, 0, 0] and r["kills"] == [0, 0, 0]
    assert r["bonds"] == [0, 0, 0] and r["shared"] == [0, 0, 7]
    assert r["bond_slots"] == [1, 0, 1]  # B->A has the bond's other end, A->B its first; (3, 1) has none
    # the exposure baseline: A0 sees 3 neighbours (one in each of B, C and A), A1 sees 2 (C and B)
    assert r["exp_attacks"] == pytest.approx([1.0, 1 / 3 + 1 / 2, 1 / 3 + 1 / 2], abs=1e-3)
    assert r["exp_bonds"] == [None, None, None]  # nobody proposed
    ab = _kernel(cfg, tags["A"], tags["B"], False)
    ac = _kernel(cfg, tags["A"], tags["C"], False)
    bc = _kernel(cfg, tags["B"], tags["C"], False)
    assert r["sim"] == pytest.approx([ab, ac, ab], abs=1e-4) and bc < 0.01
    assert r["distinct"] == [True, True, True]  # mean similarities are far under 0.25
    assert len(r["a"]) == 3 and rec["relations_dropped"] == 0  # (A, A) is exposure only: not listed
    # temperament: A's queries are +1.0 for attacks and -0.5 for bonds; B and C ask nothing; no rows in OTHER
    t = rec["temperament"]
    assert t["id"] == [1, 2, 3] and t["rows"] == [18, 12, 10]
    assert t["q_attack_kin"] == [0.0, 0.0, 1.0] and t["q_bond_kin"] == [0.0, 0.0, -0.5]
    # sampled similarity: a group over 64 rows uses the 64 smallest hashes of splitmix64(uid, sim_salt)
    big_si, _ = _sim_scene(cfg, extra_a=100)
    big = SocDriver(cfg, big_si)
    rec2 = big.advance(big_si, 250, _attack_acts(cfg, big_si, extra_a=100))
    a_rows = np.arange(110)
    h = phylo.splitmix64(a_rows.astype(np.uint64), SOCIAL_V1.sim_salt)
    pick = a_rows[np.lexsort((a_rows, h))][:64]
    want_a = _kernel(cfg, big_si["tag"][pick], big_si["tag"][pick], True)
    k2 = rec2["kin_groups"]
    assert k2["size"][0] == 110 and k2["sim_self"][0] == pytest.approx(want_a, abs=1e-4)
    # the same scene gives the same record twice
    again = SocDriver(cfg, si)
    assert json.dumps(again.advance(si, 250, _attack_acts(cfg, si))) == json.dumps(rec)


def test_pair_similarity_survives_an_inversion_of_ids_and_slots(tiny_cfg: Config) -> None:
    """Slots follow ids only loosely (a freed slot goes to the next newcomer): with the ids 4 and 3 in slots 0
    and 1, the similarity of the pair is still found, whatever the order of the two ids."""
    cfg = tiny_cfg
    bins = {
        "A": (0, (2, 2, 2, 2)),
        "B": (20, (12, 12, 12, 12)),
        "C": (40, (2, 12, 2, 12)),
        "D": (60, (12, 2, 12, 2)),
    }

    def scene(*names: str) -> dict[str, np.ndarray]:
        si = living_si(cfg, 0)
        for name in names:
            lo, bin_ = bins[name]
            si["alive"][lo : lo + 12] = True
            si["uid"][lo : lo + 12] = np.arange(lo, lo + 12)
            si["clade"][lo : lo + 12] = np.arange(lo, lo + 12)
            si["tag"][lo : lo + 12] = cloud(np.random.default_rng(lo), bin_, 12)
            si["next_uid"] = np.array(100, np.int64)
        return si

    drv = SocDriver(cfg, scene("A", "B"))
    assert drv.society.kin_slots[:2].tolist() == [1, 2]
    drv.advance(scene("A", "C"), 250)  # B ends, C is born (id 3) in the slot B freed
    assert drv.society.kin_slots[:2].tolist() == [1, 3]
    drv.advance(scene("C", "D"), 500)  # A ends, D is born (id 4) in the slot A freed
    assert drv.society.kin_slots[:2].tolist() == [4, 3]  # the ids are in descending order of slots
    acts = blank_acts(cfg)
    acts["flags"][60] = ActFlag.ATTACKED.value  # D0 attacks C0
    acts["attack_target"][60] = 40
    acts["nbr"][60, 0] = 40
    rec = drv.advance(scene("C", "D"), 750, make_acts(acts))
    r = rec["relations"]
    assert list(zip(r["a"], r["b"], strict=True)) == [(4, 3)] and r["attacks"] == [1]
    want = _kernel(cfg, scene("D")["tag"][60:72], scene("C")["tag"][40:52], False)
    assert r["sim"] == pytest.approx([want], abs=1e-4) and r["distinct"] == [True]


def test_relations_and_baselines_on_real_steps(tiny_cfg: Config) -> None:
    """Sixty real steps with three planted kin groups and a snapshot every 15 steps, through the whole chain
    (update, snapshot, settle, relabel): every listed relation equals an independent numpy reference of the
    window, ``exp_attacks`` and ``exp_bonds`` equal ``sum_i n_ib / n_i`` over the actors (relative 1e-12) and
    sum over ``b`` to the actor slot's attacks (resp. bonds) within 1e-9, and the temperament is the mean
    kin query of the rows alive at the snapshot step."""
    cfg = tiny_cfg
    sr = dataclasses.replace(SOCIAL_V1, snapshot_every=15)
    g, k = F + 1, dims(cfg).k
    s = set_output_bias(init_state(cfg), cfg, BIAS)
    rng = np.random.default_rng(0)
    tag = np_(s.tag).copy()
    for r in np.flatnonzero(np_(s.alive)):  # three tight clouds, founders by row modulo 3
        tag[r] = center(*[(2, 2, 2, 2), (12, 12, 12, 12), (2, 12, 2, 12)][r % 3]) + rng.uniform(
            -0.04, 0.04, T
        )
    s = s._replace(tag=mx.array(tag.astype(np.float32)))
    drv = SocDriver(cfg, soc_in(s), sr=sr)
    assert drv.society.head()["kin_groups"]["tracked"] == 3
    fn = make_step(cfg, observe=True)
    zeros = (np.zeros((5, g, g), np.int64), np.zeros((2, g, g, k), np.int64), np.zeros((g, g), np.int64))
    win, direct = list(zeros), np.zeros((2, g, g))
    window_ids = np.append(drv.society.kin_slots, -1)
    q_att = np.zeros(g)
    seen = {"pairs": 0, "cross": 0, "bonds": 0, "shared": 0, "snapshots": 0}
    for t in range(1, 61):
        prev_kin, prev_sg = drv.kin_slot_rows(), np_(drv.so.slot_group)
        s, out = fn(s)
        mx.eval(s, out)
        an = {f: np_(getattr(out.acts, f)) for f in out.acts._fields if getattr(out.acts, f) is not None}
        r, e, sh = window_reference(prev_kin, prev_sg, an, g, k)
        win = [win[0] + r, win[1] + e, win[2] + sh]
        for which, flag in ((0, ActFlag.ATTACKED.value), (1, ActFlag.PROPOSED.value)):  # sum_i n_ib / n_i
            for i in np.flatnonzero((an["flags"] & flag) > 0):
                nb = an["nbr"][i][an["nbr"][i] >= 0]
                for b in range(g):
                    direct[which, prev_kin[i], b] += (prev_kin[nb] == b).sum() / len(nb)
        alive0 = prev_kin >= 0
        rows = np.bincount(prev_kin[alive0], minlength=g)
        q_att = np.bincount(prev_kin[alive0], weights=an["q_kin"][alive0, 0].astype(np.float64), minlength=g)
        if t % 15:
            drv.tick(soc_in(s), out.acts)
            continue
        slot_of = {int(i): j for j, i in enumerate(window_ids)}
        rec = drv.advance(soc_in(s), t, out.acts)
        rel = rec["relations"]
        want_pairs = {(a, b) for a, b in zip(*np.nonzero(win[0][0] + win[0][3] + win[2] > 0), strict=True)}
        got = {(slot_of[x], slot_of[y]) for x, y in zip(rel["a"], rel["b"], strict=True)}
        assert got == want_pairs and rec["relations_dropped"] == 0, t
        names = ("attacks", "hits", "kills", "bonds", "bond_slots")
        att_a, bond_a = win[0][0].sum(axis=1), win[0][3].sum(axis=1)
        for j, (x, y) in enumerate(zip(rel["a"], rel["b"], strict=True)):
            a, b = slot_of[x], slot_of[y]
            for col, row in zip(names, (0, 1, 2, 3, 4), strict=True):
                assert rel[col][j] == win[0][row, a, b], (t, col, a, b)
            assert rel["shared"][j] == win[2][a, b]
            w = 1.0 / np.arange(1, k + 1)
            for col, which, tot in (("exp_attacks", 0, att_a), ("exp_bonds", 1, bond_a)):
                want = None if tot[a] == 0 else direct[which, a, b]
                if want is None:
                    assert rel[col][j] is None
                else:
                    assert rel[col][j] == pytest.approx(want, rel=1e-3, abs=6e-4)  # 3 decimals
                    assert (win[1][which, a, b] * w).sum() == pytest.approx(want, rel=1e-12)
            seen["cross"] += int(a != b and x >= 0 and y >= 0 and rel["attacks"][j] > 0)
        # the baselines sum to the actor slot's counts over every partner slot, listed or not
        h = drv.h
        for which, tot in ((0, att_a), (1, bond_a)):
            full = (h["expo"][which].astype(np.float64) / np.arange(1, k + 1)).sum(axis=-1)
            assert np.allclose(full.sum(axis=1), tot, rtol=1e-9, atol=1e-9), (t, which)
            assert np.allclose(full, direct[which], rtol=1e-12, atol=1e-12), (t, which)
        # temperament: the mean kin attack query of the rows alive at the start of the step
        tem = rec["temperament"]
        for j, ident in enumerate(tem["id"]):
            sl = slot_of[ident]
            assert tem["rows"][j] == rows[sl]
            want_q = None if rows[sl] == 0 else round(q_att[sl] / (256 * rows[sl]), 4)
            assert tem["q_attack_kin"][j] == (want_q if want_q is None else pytest.approx(want_q, abs=1e-4))
        seen["pairs"] += len(rel["a"])
        seen["bonds"] += sum(rel["bonds"])
        seen["shared"] += sum(rel["shared"])
        seen["snapshots"] += 1
        window_ids = np.append(drv.society.kin_slots, -1)
        win, direct = list(zeros), np.zeros((2, g, g))
    assert seen["snapshots"] == 4 and seen["pairs"] > 8 and seen["cross"] > 2 and seen["bonds"] > 0
    assert seen["shared"] > 0


def test_a_pair_listed_for_its_energy_alone_has_no_baseline(tiny_cfg: Config) -> None:
    """A pair that only shares energy (nobody attacked or proposed) is listed with ``exp_attacks`` and
    ``exp_bonds`` None: the baselines are defined per actor that acted."""
    cfg = tiny_cfg
    si, _ = _sim_scene(cfg)
    drv = SocDriver(cfg, si)
    acts = blank_acts(cfg)
    acts["share_out"][0, 0] = 5  # A0 sends 5 quanta to B0 along their bond
    r = drv.advance(si, 250, make_acts(acts))["relations"]
    assert (
        list(zip(r["a"], r["b"], strict=True)) == [(3, 2)] and r["shared"] == [5] and r["bond_slots"] == [1]
    )
    assert r["attacks"] == [0] and r["exp_attacks"] == [None] and r["exp_bonds"] == [None]
    assert r["sim"] != [None] and r["distinct"] == [True]


def test_houses_are_inexact_after_a_mid_run_start(tiny_cfg: Config) -> None:
    """Houses are exact iff the observer started at step 0: a fresh observer at a later step says so, in its
    head and in every record, whose ``origin`` and ``window_start`` are that step."""
    cfg = tiny_cfg
    si, _ = _sim_scene(cfg)
    si["clade"][:5] = cfg.population.n_init  # immigrant lineages, seeded as HOUSE_UNKNOWN
    drv = SocDriver(cfg, si, step=30)
    head = drv.society.head()
    assert head["origin"] == 30 and head["snapshot_step"] == 30
    assert head["houses"] == {"alive": 35, "exact": False, "unknown_immigrant": 5}
    rec = drv.advance(si, 280)
    assert rec["origin"] == 30 and rec["window_start"] == 30 and rec["summary"]["houses"]["exact"] is False
    assert HOUSE_UNKNOWN not in rec["houses"]["id"] and rec["summary"]["houses"]["unknown_immigrant"] == 5
    assert SocDriver(cfg, si, step=0).society.head()["houses"]["exact"] is True


def test_relations_cap_and_ranking(tiny_cfg: Config) -> None:
    """More pairs than ``relations_max``: the first by (-(attacks + bonds), -shared, a, b), counted in
    ``relations_dropped``, listed by (a, b); exposure-only pairs never listed."""
    cfg = tiny_cfg
    sr = dataclasses.replace(SOCIAL_V1, relations_max=2)
    si, _ = _sim_scene(cfg)
    drv = SocDriver(cfg, si, sr=sr)
    rec = drv.advance(si, 250, _attack_acts(cfg, si))
    r = rec["relations"]
    # A->B (shared 7) outranks the two pairs with one attack each; of those (2, 3) precedes (3, 1)
    assert list(zip(r["a"], r["b"], strict=True)) == [(2, 3), (3, 2)] and r["shared"] == [0, 7]
    assert rec["relations_dropped"] == 1
    # attacks and bonds rank before energy shared: a pair with two attacks beats one with a lot shared
    one = dataclasses.replace(SOCIAL_V1, relations_max=1)
    acts = blank_acts(cfg)
    acts["flags"][[0, 1]] = ActFlag.ATTACKED.value  # A0 and A1 both attack B0 (then C0 is untouched)
    acts["attack_target"][[0, 1]] = 10
    acts["nbr"][[0, 1], 0] = 10
    acts["share_out"][12, 0] = 90  # B2 sends 90 quanta along the bond to C0 (added below)
    si2 = {k: v.copy() for k, v in si.items()}
    si2["partner"][12, 0], si2["partner"][22, 1] = 22, 12
    sx = SocDriver(cfg, si2, sr=one)
    recx = sx.advance(si2, 250, make_acts(acts))
    assert list(zip(recx["relations"]["a"], recx["relations"]["b"], strict=True)) == [
        (3, 2)
    ]  # A -> B, 2 attacks
    assert recx["relations"]["attacks"] == [2] and recx["relations_dropped"] == 1
    none = dataclasses.replace(SOCIAL_V1, relations_max=0)
    rec0 = SocDriver(cfg, si, sr=none).advance(si, 250, _attack_acts(cfg, si))
    assert rec0["relations"]["a"] == [] and rec0["relations_dropped"] == 3


# --- houses -------------------------------------------------------------------------------------------


def test_house_names(tiny_cfg: Config) -> None:
    """Names are a deterministic function of the id and the salt; a name shared within the listed houses gets
    the id appended for every house after the first (by id); the top 16 come by (-size, id);
    ``HOUSE_UNKNOWN`` is never named."""
    ids = np.array([0, 1, 12345, 2**31 - 1, 7, 7])
    names = house_names(ids[:5], SOCIAL_V1.name_salt)
    assert names == house_names(ids[:5], SOCIAL_V1.name_salt) and len(set(names)) == 5
    h = phylo.splitmix64(ids[:5].astype(np.uint64), SOCIAL_V1.name_salt)
    for x, nm in zip(h.tolist(), names, strict=True):
        want = (
            ONSET[x & 31]
            + NUCLEUS[(x >> 5) & 15]
            + ONSET[(x >> 9) & 31]
            + NUCLEUS[(x >> 14) & 15]
            + CODA[(x >> 18) & 15]
        )
        assert nm == want.capitalize()
    assert (len(ONSET), len(NUCLEUS), len(CODA)) == (32, 16, 16)
    assert house_names(ids[:5], 1) != names  # another salt, other names
    assert house_names(np.array([], np.int64), 0) == []
    # a name shared by two ids: the smaller id keeps the bare name
    base = raw_house_names(np.arange(40000), SOCIAL_V1.name_salt)
    seen: dict[str, int] = {}
    pair = None
    for i, nm in enumerate(base):
        if nm in seen:
            pair = (seen[nm], i, nm)
            break
        seen[nm] = i
    assert pair is not None
    lo, hi, nm = pair
    assert house_names(np.array([hi, lo]), SOCIAL_V1.name_salt) == [f"{nm} {hi}", nm]
    # through a record: house ids are the founders' clades; sizes order the top 16, ties by id
    cfg = tiny_cfg
    rng = np.random.default_rng(0)
    sc = Scene(cfg)
    sizes = {7: 5, 3: 5, 100: 9, **{200 + i: 1 + (i % 3) for i in range(20)}}
    for house, n in sizes.items():
        sc.add(rng.uniform(-1, 1, (n, T)).astype(np.float32), clade=house)
    rec = SocDriver(cfg, sc.si).advance(sc.si, 250)
    hs = rec["houses"]
    assert len(hs["id"]) == 16 and hs["size"][0] == 9 and hs["id"][0] == 100
    assert hs["id"][1:3] == [3, 7] and hs["size"][1:3] == [5, 5]  # tied sizes by id
    assert hs["size"] == sorted(hs["size"], reverse=True) and hs["size"][3:6] == [3, 3, 3]
    assert hs["id"][3:6] == [202, 205, 208]  # the size-3 houses, by id
    assert rec["summary"]["houses"] == {"alive": len(sizes), "exact": True, "unknown_immigrant": 0}
    assert hs["clade"] == hs["id"]  # a founder's house is its clade
    # two listed houses with the same name: the one with the larger id gets its id appended
    sc5 = Scene(cfg)
    sc5.add(rng.uniform(-1, 1, (4, T)).astype(np.float32), clade=lo)
    sc5.add(rng.uniform(-1, 1, (3, T)).astype(np.float32), clade=hi)
    r5 = SocDriver(cfg, sc5.si).advance(sc5.si, 250)
    assert r5["houses"]["id"] == [lo, hi] and r5["houses"]["name"] == [nm, f"{nm} {hi}"]
    # HOUSE_UNKNOWN (clade n_init) is counted apart and never listed
    sc2 = Scene(cfg)
    sc2.add(rng.uniform(-1, 1, (4, T)).astype(np.float32), clade=cfg.population.n_init)
    sc2.add(rng.uniform(-1, 1, (3, T)).astype(np.float32), clade=9)
    d2 = SocDriver(cfg, sc2.si)
    head = d2.society.head()["houses"]
    assert head == {"alive": 1, "exact": True, "unknown_immigrant": 4}
    r2 = d2.advance(sc2.si, 250)
    assert r2["houses"]["id"] == [9] and HOUSE_UNKNOWN not in r2["houses"]["id"]


# --- depth bands, territory, the window's slots -------------------------------------------------------


def test_depth_bands_3d(tiny_cfg: Config, tiny3d_cfg: Config) -> None:
    """``z_mean`` and ``band_frac`` of a planted 3-D group against ``metrics.z_band``; absent in 2-D."""
    cfg = tiny3d_cfg
    height = cfg.world.height
    z = np.array([3.0] * 20 + [8.0] * 10 + [13.0] * 10 + [23.5] * 4)
    rng = np.random.default_rng(0)
    pos = np.concatenate([rng.uniform(0, cfg.world.size, (44, 2)), z[:, None]], axis=1)
    sc = Scene(cfg)
    sc.add(cloud(rng, (2, 2, 2, 2), 44), pos=pos)
    rec = SocDriver(cfg, sc.si).advance(sc.si, 250)
    kg = rec["kin_groups"]
    assert list(kg) == [*EMPTY_TABLES["kin_groups"], "z_mean", "band_frac"]
    band = metrics.z_band(z, height)
    frac = np.bincount(band, minlength=metrics.N_BANDS) / 44
    assert kg["z_mean"] == [round(float(z.mean()), 2)] and kg["band_frac"] == [
        [round(float(f), 4) for f in frac]
    ]
    assert kg["band_frac"][0] == [round(20 / 44, 4), round(10 / 44, 4), round(10 / 44, 4), round(4 / 44, 4)]
    # a height on a band edge belongs to the band above it, and the top band holds the surface
    z2 = np.array([6.0, 6.0, 24.0, 0.0] * 3)
    pos2 = np.concatenate([rng.uniform(0, cfg.world.size, (12, 2)), z2[:, None]], axis=1)
    sc2 = Scene(cfg)
    sc2.add(cloud(rng, (2, 2, 2, 2), 12), pos=pos2)
    rec2 = SocDriver(cfg, sc2.si).advance(sc2.si, 250)
    assert rec2["kin_groups"]["band_frac"] == [[0.25, 0.5, 0.0, 0.25]]
    # a colony in 3-D reports z too
    sc3 = Scene(cfg)
    sc3.add(
        cloud(rng, (2, 2, 2, 2), 5), pos=np.concatenate([np.full((5, 2), 40.0), np.full((5, 1), 5.0)], axis=1)
    )
    sc3.bond([0, 1, 2, 3, 4])
    rec3 = SocDriver(cfg, sc3.si).advance(sc3.si, 250)
    assert list(rec3["colonies"]) == ["id", "slot", "size", "kin_group", "x", "y", "z"]
    assert rec3["colonies"]["z"] == [5.0] and rec3["colonies"]["x"] == [40.0]
    # 2-D: neither kin-group nor colony rows have a height
    c2 = tiny_cfg
    sc4 = Scene(c2)
    sc4.add(cloud(rng, (2, 2, 2, 2), 12))
    sc4.bond([0, 1, 2, 3])
    rec4 = SocDriver(c2, sc4.si).advance(sc4.si, 250)
    assert "z_mean" not in rec4["kin_groups"] and "band_frac" not in rec4["kin_groups"]
    assert "z" not in rec4["colonies"]


def test_territory_report(tiny_cfg: Config) -> None:
    """Two groups on separate halves of the floor own their cells with purity 1; a cell shared by both goes to
    the larger share; the report uses the window's slots (the previous snapshot's map), so a group born at
    this snapshot appears in the territory only at the next one."""
    cfg = tiny_cfg
    rng = np.random.default_rng(2)
    cells = 32
    cs = cfg.world.size / cells  # 4 units
    sc = Scene(cfg)
    # group A: 12 rows in 12 left cells, one per cell; group B: 9 rows in 9 right cells
    left = np.array([[(2 * i + 0.5) * cs % 60, (i // 4 * 4 + 0.5) * cs] for i in range(12)])
    right = np.array([[70 + (i % 3) * cs * 2, (i // 3 * 4 + 0.5) * cs] for i in range(9)])
    a = sc.add(cloud(rng, (2, 2, 2, 2), 12), pos=left)
    b = sc.add(cloud(rng, (12, 12, 12, 12), 9), pos=right)
    drv = SocDriver(cfg, sc.si)
    ida, idb = int(drv.society.kin_slots[0]), int(drv.society.kin_slots[1])
    assert (ida, idb) == (1, 2)  # A is the larger
    rec = drv.advance(sc.si, 250)
    terr = rec["territory"]
    assert terr["owned_cells"] == 21 and terr["purity_mean"] == 1.0
    assert terr["cells"] == {"id": [1, 2], "cells": [12, 9]}
    dmap, slots = drv.society.territory()
    assert slots[:2].tolist() == [1, 2] and (dmap.owner >= 0).sum() == 21
    assert (
        set(dmap.owner[dmap.owner >= 0].tolist()) == {1, 2}
        and (dmap.purity_q16[dmap.owner >= 0] == 65536).all()
    )
    assert dmap.border.any()
    # a new group appears at the next snapshot: its rows are OTHER in this window, so it owns no cell yet
    c = sc.add(cloud(rng, (6, 13, 6, 13), 10), pos=np.full((10, 2), 100.0))
    rec2 = drv.advance(sc.si, 500)
    assert rec2["kin_groups"]["id"] == [1, 2, 3] and rec2["territory"]["cells"] == {
        "id": [1, 2],
        "cells": [12, 9],
    }
    # (the temperament reads the rows alive at the start of the snapshot step: the new rows are not yet)
    assert rec2["temperament"]["id"] == [1, 2] and rec2["temperament"]["rows"] == [12, 9]
    rec3 = drv.advance(sc.si, 750)
    assert rec3["territory"]["cells"] == {"id": [1, 2, 3], "cells": [12, 9, 1]}
    assert rec3["temperament"]["id"] == [1, 2, 3] and rec3["temperament"]["rows"] == [12, 9, 10]
    assert b.stop - b.start == 9 and a.stop == 12 and c.stop == 31
    # a shared cell goes to the larger share, with its purity: 3 agents of A and 1 of B in one cell
    mixed = np.concatenate([np.full((3, 2), 10.0), left[3:12], np.full((1, 2), 10.0), right[:8] + 5.0])
    sm = Scene(cfg)
    sm.add(cloud(rng, (2, 2, 2, 2), 12), pos=mixed[:12])
    sm.add(cloud(rng, (12, 12, 12, 12), 9), pos=mixed[12:])
    dm = SocDriver(cfg, sm.si)
    t2 = dm.advance(sm.si, 250)["territory"]
    assert t2["cells"] == {"id": [1, 2], "cells": [10, 8]} and t2["owned_cells"] == 18
    assert t2["purity_mean"] == round((0.75 + 17) / 18, 4)
    cell = int(10 // cs) * cells + int(10 // cs)
    mp = dm.society.territory()[0]
    assert mp.owner[cell] == 1 and mp.purity_q16[cell] == 3 * 65536 // 4


def test_kin_group_summary_columns(tiny_cfg: Config) -> None:
    """Mean tag, dominant clade (ties to the smaller clade) and its share, per slotted group."""
    cfg = tiny_cfg
    rng = np.random.default_rng(3)
    sc = Scene(cfg)
    a_tags = cloud(rng, (2, 2, 2, 2), 12)
    sc.add(a_tags, clade=7)
    sc.si["clade"][9:12] = 8  # nine rows of clade 7, three of clade 8
    b_tags = cloud(rng, (12, 12, 12, 12), 8)
    sc.add(b_tags, clade=3)
    sc.si["clade"][16:20] = 2  # a tie between clades 3 and 2: the smaller clade, 2
    rec = SocDriver(cfg, sc.si).advance(sc.si, 250)
    kg = rec["kin_groups"]
    assert kg["size"] == [12, 8] and kg["clade"] == [7, 2] and kg["clade_share"] == [0.75, 0.5]
    assert kg["tag"] == [
        [round(float(v), 3) for v in t.astype(np.float64).mean(axis=0)] for t in (a_tags, b_tags)
    ]


def test_snap_info_for_the_calibration_tool(tiny_cfg: Config) -> None:
    """``Society.info`` carries what ``analysis.social`` reads: counts, Jaccard values, thresholds, the
    occupied bins, the unaffiliated rows, the tracked sizes and ``sim_self``; it is not in the record."""
    cfg = tiny_cfg
    si, _ = _sim_scene(cfg)
    drv = SocDriver(cfg, si)
    assert (
        drv.society.info.step == 0 and drv.society.info.sim_self == {} and drv.society.info.kin_events == []
    )
    moved = {k: v.copy() for k, v in si.items()}
    moved["alive"][:3] = False  # three agents of A die: Jaccard 7/10
    moved["uid"][:3] = -1
    drv.advance(moved, 250)
    info = drv.society.info
    assert info.step == 250 and info.n_alive == 37 and info.thresholds["occ_min"] == 2
    d = info.kin_diag
    assert (d.n_prev, d.n_cur, d.n_matched) == (
        3,
        3,
        3,
    ) and d.n_admitted == 2  # A, down to 7 rows, is no longer admissible
    assert (
        sorted(d.matched_jaccard.tolist()) == pytest.approx([0.7, 1.0, 1.0])
        and d.unmatched_best_jaccard.size == 0
    )
    assert info.kin_sizes == {1: 18, 2: 12, 3: 7} and info.col_sizes == {}
    assert info.occupied_bins == 8 and info.unaffiliated == 0  # two clouds and a chain of six bins
    assert set(info.sim_self) == {1, 2, 3} and all(v is not None for v in info.sim_self.values())
    assert info.kin_events == [] and info.col_events == []


def test_society_api_errors(tiny_cfg: Config) -> None:
    sr = SOCIAL_V1
    soc = Society(tiny_cfg, sr)
    assert not soc.started
    with pytest.raises(RuntimeError, match="not initialised"):
        soc.head()
    with pytest.raises(RuntimeError, match="not initialised"):
        soc.arrays()
    si, _ = _sim_scene(tiny_cfg)
    drv = SocDriver(tiny_cfg, si, society=soc)
    assert soc.started and soc.snapshot_step == 0 and soc.house_origin == 0 and soc.records == []
    with pytest.raises(RuntimeError, match="already initialised"):
        soc.initialise(drv.h, 0)
    drv.advance(si, 250)
    with pytest.raises(ValueError, match="does not follow"):
        soc.settle(drv.h, 250)  # the same step again
    recs = soc.take_records()
    assert [r["step"] for r in recs] == [250] and soc.records == [] and soc.take_records() == []
    # a restored society continues from the same host state
    other, why = Society.from_arrays(soc.arrays(), tiny_cfg, sr)
    assert why == "" and other is not None and other.head() == soc.head()


# --- empty cases --------------------------------------------------------------------------------------

EMPTY_TABLES = {
    "houses": ["id", "name", "size", "clade"],
    "kin_groups": ["id", "name", "named_after", "slot", "size", "tag", "clade", "clade_share", "sim_self",
                   "x", "y", "arch", "births", "deaths"],
    "colonies": ["id", "slot", "size", "kin_group", "x", "y"],
    "events": ["step", "family", "kind", "id", "size", "other", "overlap"],
    "relations": ["a", "b", "attacks", "hits", "kills", "bonds", "shared", "bond_slots", "exp_attacks",
                  "exp_bonds", "sim", "distinct"],
    "temperament": ["id", "q_attack_kin", "q_bond_kin", "rows"],
}  # fmt: skip


def _valid_json(rec: dict[str, Any], head: dict[str, Any]) -> None:
    assert json.loads(runio._dumps(rec)) == rec and json.loads(runio._dumps(head)) == head


def test_empty_society(tiny_cfg: Config) -> None:
    """``initialise`` and ``settle`` on a state with no alive row, on one with alive rows but no occupied bin
    and no colony, and on an extinct state with immigration on: every table is empty with its columns, every
    ratio None, ``owned_cells`` 0, no exception; the head and the record are valid JSON."""
    # no alive row at all
    cfg = tiny_cfg
    drv = SocDriver(cfg, living_si(cfg, 0))
    assert drv.society.head()["houses"] == {"alive": 0, "exact": True, "unknown_immigrant": 0}
    assert drv.society.head()["kin_groups"] == {"tracked": 0, "other": 0}
    assert drv.society.head()["colonies"] == {
        "tracked": 0, "members": 0, "largest": 0, "converged": True, "match": "uid",
    }  # fmt: skip
    rec = drv.advance(living_si(cfg, 0), 250)
    for name, cols in EMPTY_TABLES.items():
        assert rec[name] == {c: [] for c in cols}, name
    assert rec["house_events"] == {
        "founded": {"id": [], "step": [], "parent_house": []},
        "extinct": {"id": [], "step": []},
        "founded_dropped": 0,
        "extinct_dropped": 0,
    }
    assert rec["territory"] == {"owned_cells": 0, "purity_mean": None, "cells": {"id": [], "cells": []}}
    assert rec["events_dropped"] == 0 and rec["relations_dropped"] == 0
    assert rec["thresholds"]["n_ref"] == 0 and rec["thresholds"]["occ_min"] == 2
    _valid_json(rec, drv.society.head())
    assert (drv.kin_slot_rows() == -1).all() and (np_(drv.so.colony) == -1).all()
    # alive rows, but one per bin and no bond: no occupied bin, no group, no colony
    rng = np.random.default_rng(0)
    sc = Scene(cfg)
    codes = rng.choice(16**4, 50, replace=False) * 1  # lone rows; none shares a face with another
    digits = np.stack(np.unravel_index(np.sort(codes), (16,) * 4), axis=1)
    keep = [0]
    for i in range(1, 50):  # drop rows adjacent to a kept one so that no sparse bin attaches
        if all(np.abs(digits[i] - digits[j]).sum() > 1 for j in keep):
            keep.append(i)
    sc.add((-1.0 + (digits[keep] + 0.5) / 8.0).astype(np.float32))
    drv2 = SocDriver(cfg, sc.si)
    head = drv2.society.head()
    assert head["kin_groups"] == {"tracked": 0, "other": len(keep)} and head["colonies"]["tracked"] == 0
    rec2 = drv2.advance(sc.si, 250)
    assert rec2["kin_groups"] == {c: [] for c in EMPTY_TABLES["kin_groups"]}
    assert rec2["colonies"] == {c: [] for c in EMPTY_TABLES["colonies"]}
    assert rec2["houses"]["size"] == [1] * min(16, len(keep)) and rec2["summary"]["houses"]["alive"] == len(
        keep
    )
    assert rec2["temperament"]["id"] == [-1] and rec2["temperament"]["q_attack_kin"] == [0.0]
    _valid_json(rec2, drv2.society.head())
    # a population that dies out with immigration on: groups and colonies end, tables empty, ratios None
    imm = apply_overrides(cfg, ["population.immigration_every=5", "population.immigration_count=8"])
    si, _ = _sim_scene(imm)
    drv3 = SocDriver(imm, si)
    tracked = drv3.society.head()
    assert tracked["kin_groups"]["tracked"] == 3
    dead = living_si(imm, 0)
    drv3.tick(dead)  # everybody dies one step before the snapshot
    rec3 = drv3.advance(dead, 250)
    kinds = sorted((f, k) for f, k, *_ in _events(rec3))
    assert kinds == [("kin_group", "dissolved")] * 3 and rec3["kin_groups"]["id"] == []
    assert rec3["temperament"]["id"] == [1, 2, 3] and rec3["temperament"]["q_attack_kin"] == [None] * 3
    assert rec3["temperament"]["rows"] == [0, 0, 0]  # nobody was alive at the start of the snapshot step
    assert rec3["territory"]["owned_cells"] == 0 and rec3["territory"]["purity_mean"] is None
    assert rec3["thresholds"]["rescaled"] is True and rec3["thresholds"]["n_ref"] == 0
    _valid_json(rec3, drv3.society.head())
    rec4 = drv3.advance(dead, 500)  # and nothing is left to end
    assert (
        rec4["events"]["id"] == []
        and rec4["temperament"]["id"] == []
        and rec4["territory"]["cells"]["id"] == []
    )
    assert rec4["thresholds"]["rescaled"] is False
    # a population that comes back is tracked afresh
    rec5 = drv3.advance(si, 750)
    assert sorted(rec5["kin_groups"]["id"]) == [4, 5, 6]


# --- the head and the record --------------------------------------------------------------------------

HEAD_KEYS = ["social_version", "snapshot_step", "origin", "houses", "kin_groups", "colonies"]
RECORD_KEYS = [
    "step", "social_version", "social_digest", "intent_version", "origin", "window_start", "summary",
    "thresholds", "houses", "house_events", "kin_groups", "colonies", "events", "events_dropped",
    "event_counts", "relations", "relations_dropped", "temperament", "territory",
]  # fmt: skip


def _keys_in_order(a: Any, b: Any) -> bool:
    """Equal as dicts and in key order at every level (json.dumps without sort_keys is equal)."""
    return json.dumps(a) == json.dumps(b)


def _two_snapshots(
    cfg: Config, sr: SocialRules = SOCIAL_V1
) -> tuple[SocDriver, dict[str, Any], dict[str, Any]]:
    si, _ = _sim_scene(cfg)
    drv = SocDriver(cfg, si, sr=sr)
    first = drv.advance(si, 250, _attack_acts(cfg, si))
    moved = {k: v.copy() for k, v in si.items()}
    moved["uid"][:40] += 500  # every agent is replaced by a stranger in the same place (events in uid mode)
    moved["next_uid"] = np.array(600, np.int64)
    second = drv.advance(moved, 500)
    return drv, first, second


def test_head_and_record_keys(tiny_cfg: Config, tiny3d_cfg: Config) -> None:
    """The exact keys, in order, of the head, the record and every table; ``window_start`` is the previous
    snapshot (the origin for the first record), ``origin`` the fresh start; the head equals ``{social_version,
    snapshot_step: record.step, origin: record.origin, **record.summary}`` in that key order at every level,
    also after a ``Society.from_arrays(society.arrays())`` round trip; no wall-clock field."""
    for cfg, z in ((tiny_cfg, False), (tiny3d_cfg, True)):
        sr = dataclasses.replace(SOCIAL_V1, events_max=2)
        drv, first, second = _two_snapshots(cfg, sr)
        for rec, step, start in ((first, 250, 0), (second, 500, 250)):
            assert list(rec) == RECORD_KEYS and rec["step"] == step and rec["window_start"] == start
            assert rec["origin"] == 0 and rec["social_version"] == 1 and rec["social_digest"] == sr.digest()
            assert rec["intent_version"] == INTENT_VERSION == 3  # the width key of ``kin_groups.arch``
            assert list(rec["summary"]) == ["houses", "kin_groups", "colonies"]
            assert list(rec["summary"]["houses"]) == ["alive", "exact", "unknown_immigrant"]
            assert list(rec["summary"]["kin_groups"]) == ["tracked", "other"]
            assert list(rec["summary"]["colonies"]) == ["tracked", "members", "largest", "converged", "match"]
            assert list(rec["thresholds"]) == ["n_ref", "rescaled", "occ_min", "kin_enter", "kin_leave"]
            assert list(rec["event_counts"]) == ["kin_group", "colony"]
            assert all(
                list(c) == ["born", "split", "merged", "dissolved"] for c in rec["event_counts"].values()
            )
            assert list(rec["territory"]) == ["owned_cells", "purity_mean", "cells"]
            assert list(rec["house_events"]) == ["founded", "extinct", "founded_dropped", "extinct_dropped"]
            assert list(rec["house_events"]["founded"]) == ["id", "step", "parent_house"]
            assert list(rec["house_events"]["extinct"]) == ["id", "step"]
            for name, cols in EMPTY_TABLES.items():
                want = cols + (["z_mean", "band_frac"] if z and name == "kin_groups" else [])
                want = want + (["z"] if z and name == "colonies" else [])
                assert list(rec[name]) == want, name
                assert len({len(v) for v in rec[name].values()}) == 1, name  # equal-length columns
            assert list(rec["territory"]["cells"]) == ["id", "cells"]
            assert not {"alife_version", "time", "wall", "host", "timestamp"} & set(rec)
        # the head names its record
        head = drv.society.head()
        assert list(head) == HEAD_KEYS and head["origin"] == 0 and head["snapshot_step"] == 500
        want = {
            "social_version": 1,
            "snapshot_step": second["step"],
            "origin": second["origin"],
            **second["summary"],
        }
        assert _keys_in_order(head, want)
        again, why = Society.from_arrays(drv.society.arrays(), cfg, sr)
        assert why == "" and _keys_in_order(again.head(), head)
        assert json.dumps(again.head()) == json.dumps(head)
        # caps: events_max 2 lists the largest two of the six events of the second snapshot, the rest counted
        ev = second["events"]
        total = sum(sum(c.values()) for c in second["event_counts"].values())
        assert len(ev["id"]) == 2 and second["events_dropped"] == total - 2 and total == 6
        # the largest first (by size, then kind: born before dissolved), here the two events of size 18
        assert (
            ev["size"] == [18, 18]
            and ev["kind"] == ["born", "dissolved"]
            and ev["family"] == ["kin_group"] * 2
        )
        # relations: A->B, A->C and B->A were listed in the first record; the pair (A, A) is exposure only
        assert len(first["relations"]["a"]) == 3 and (3, 3) not in zip(
            first["relations"]["a"], first["relations"]["b"], strict=True
        )
        # the first record's window starts at the origin, the second's at the first record's step
        assert first["window_start"] == drv.society.records[0]["window_start"] == 0


# --- the budget ---------------------------------------------------------------------------------------


def _cap_scene(cfg: Config, generation: int) -> dict[str, np.ndarray]:
    """A 3-D scene that reaches every cap of a record: 17 kin groups (one more than the slots), 40 colonies of
    4, 300 houses with 8-digit ids, lone rows. ``generation`` 1 replaces every agent but those of the kin
    groups (new uids for the colonies and the lone rows: colony, house founded and house extinct events); the
    groups stay (same uids, houses, bins), so each keeps its slot and its window counts are listed."""
    rng = np.random.default_rng(10 + generation)
    sc = Scene(cfg, 10 + generation)
    size = cfg.world.size
    uid0 = 10_000_000 * (generation + 1)
    for i in range(17):  # kin groups in well separated bins; sizes 9..11 so that the ranks are strict
        bin_ = (3 * (i % 5), 3 * (i // 5), (7 * i) % 16, (5 * i) % 16)
        n = 9 + i % 3
        sc.add(cloud(rng, bin_, n), clade=12_345_000 + i)
    n_kin = sc.next
    for k in range(40):  # colonies of 4 (lone tags: OTHER for the kin groups)
        lo = sc.next
        tags = rng.uniform(-1, 1, (4, T)).astype(np.float32)
        pos = np.concatenate(
            [
                np.stack([np.full(4, (k % 8) * 15.0 + 4), np.full(4, (k // 8) * 20.0 + 4)], 1),
                np.full((4, 1), 3.0 + k % 18),
            ],
            1,
        )
        sc.add(tags, pos=pos)
        sc.bond(list(range(lo, lo + 4)))
    lone = sc.add(
        rng.uniform(-1, 1, (80, T)).astype(np.float32),
        pos=np.concatenate([rng.uniform(0, size, (80, 2)), rng.uniform(1, 20, (80, 1))], 1),
    )
    n = sc.next
    sc.si["uid"][:n_kin] = 10_000_000 + np.arange(n_kin)  # the kin groups' rows keep their uids
    sc.si["uid"][n_kin:n] = uid0 + np.arange(n_kin, n)
    sc.si["next_uid"] = np.array(uid0 + n, np.int64)
    sc.si["clade"][:n] = np.where(
        sc.si["clade"][:n] >= 12_000_000, sc.si["clade"][:n], 20_000_000 + np.arange(n) % 300
    )
    assert lone.stop == n
    return sc.si


RECORD_BUDGET = 16 * 1024  # DESIGN §29.11: the widest plausible record of the scene below, with headroom
HEAD_BUDGET = 512


def _renumbered(soc: Society, cfg: Config, kin_offset: int, col_offset: int, step: int) -> Society:
    """``soc`` as it would be late in a long run: every kin-group id raised by ``kin_offset`` and every colony
    id by ``col_offset`` (tracker ids are never reused, so they grow with the run), and each kin group named
    with the longest generated name (10 letters) and its id, the form of a name that was held when it was
    given."""
    a = soc.arrays()
    for fam, off in (("kin", kin_offset), ("col", col_offset)):
        a[f"host.{fam}.system"] = a[f"host.{fam}.system"] + off
        a[f"host.{fam}.slots"] = np.where(a[f"host.{fam}.slots"] >= 0, a[f"host.{fam}.slots"] + off, -1)
        a[f"host.{fam}.next_id"] = np.array(int(a[f"host.{fam}.next_id"]) + off, np.int64)
    a["host.terr.ids"] = np.where(a["host.terr.ids"] >= 0, a["host.terr.ids"] + kin_offset, -1)
    names = [
        [i + kin_offset, f"{'B' * 10} {i + kin_offset}", after]
        for i, _, after in json.loads(str(a["host.kin.names"]))
    ]
    a["host.kin.names"] = np.array(runio._dumps(names))
    out, why = Society.from_arrays(a, cfg, SOCIAL_V1, step=step)
    assert out is not None, why
    return out


def test_head_and_record_size_budget(tiny3d_cfg: Config) -> None:
    """A synthetic 64k-like 3-D snapshot with every cap reached (17 window ids, 16 rows in each top table, 32
    events, 32 relations, 32 house events of each kind, 7-digit steps, 8-digit house ids) and every column at
    the widest value a long 64k run plausibly gives it: six-digit kin-group ids and seven-digit colony ids
    (tracker ids are never reused), names of 10 letters and the id, window counts of six to eight digits
    (the 250-step window of ``runs/64k_s0`` at 1.5M steps: 115k attacks, 780k proposals, 22.6M quanta shared,
    6.8M bond-steps, spread over the listed pairs, each at its largest), exposures of ten digits with three
    decimals, and the roster's columns (six-digit births and deaths by cause, seven archetype shares). The
    record is at most 16 KB, the head at most 0.5 KB. (A scene with one-attack pairs and two-digit ids gave
    10,951 B; the 64k_s0 window with long-run ids 12.5 KB; this one 13.3 KB.)"""
    cfg = tiny3d_cfg
    base = 1_490_000
    first = _cap_scene(cfg, 0)
    drv = SocDriver(cfg, first, step=base)
    assert (
        drv.society.head()["kin_groups"]["tracked"] == 17 and drv.society.head()["colonies"]["tracked"] == 40
    )
    drv.society = _renumbered(drv.society, cfg, 100_000, 1_000_000, base)
    # 16 kin groups hold a slot and one is OTHER; 16 colonies hold a slot and 24 are "other tracked" (F)
    kin_rows = drv.kin_slot_rows()
    in_group = kin_rows[first["alive"]]
    assert (
        set(range(F)) <= set(in_group.tolist()) and (in_group == F).sum() == 9 + 160 + 80
    )  # smallest + rest
    roots = np.flatnonzero(drv.col_lut >= 0)
    assert len(roots) == 40 and (drv.col_lut[roots] < F).sum() == 16 and (drv.col_lut[roots] == F).sum() == 24
    assert np.array_equal(
        np.sort(np_(drv.so.colony)[np_(drv.so.colony) >= 0]), np.sort(np.repeat(drv.col_lut[roots], 4))
    )
    # attacks between the window's groups: 40 distinct ordered pairs, each by one agent of the first group
    acts = blank_acts(cfg)
    rows = {}
    for slot in range(F):  # first row of every slotted group
        rows[slot] = int(np.flatnonzero(drv.kin_slot_rows() == slot)[0])
    pairs = [(a, b) for a in range(F) for b in range(F) if a != b][::6][:40]
    assert len(pairs) == 40
    used = set()
    for a, b in pairs:
        attacker = rows[a] + (1 if rows[a] in used else 0)
        while attacker in used:
            attacker += 1
        used.add(attacker)
        acts["flags"][attacker] = ActFlag.ATTACKED.value
        acts["attack_target"][attacker] = rows[b]
        acts["nbr"][attacker, 0] = rows[b]
    acts["q_kin"][:] = np.where(first["alive"][:, None], [300, -200], 0)
    # the window's counts: each listed pair at the width of the whole window's total (a bound, not a sum), the
    # exposure sums at ten digits with a third (j = 2) or a sixth (j = 5) so that three decimals are written
    g, k = SOCIAL_V1.slots + 1, drv.so.expo.shape[-1]
    rel = np.zeros((5, g, g), np.int64)
    expo = np.zeros((2, g, g, k), np.int64)
    shared = np.zeros((g, g), np.int64)
    for i, (a, b) in enumerate(pairs):
        rel[Rel.ATTACKS, a, b], rel[Rel.HITS, a, b], rel[Rel.KILLS, a, b] = (
            999_000 + i,
            998_000 + i,
            99_000 + i,
        )
        rel[Rel.BONDS, a, b], rel[Rel.BOND_SLOTS, a, b] = 997_000 + i, 9_990_000 + i
        shared[a, b] = 99_900_000 + i
        expo[Expo.ATTACK, a, b, 2], expo[Expo.BOND, a, b, 5] = 2_999_998 - i, 5_999_999 - i
    drv.so = drv.so._replace(
        births=mx.array(900_000 + np.arange(g), dtype=mx.int64),
        deaths=mx.array(800_000 + 17 * np.arange(4 * g).reshape(4, g), dtype=mx.int64),
        rel=drv.so.rel + mx.array(rel),
        expo=drv.so.expo + mx.array(expo),
        shared=drv.so.shared + mx.array(shared),
    )
    labels = 1 + np.random.default_rng(3).integers(0, 7, size=first["alive"].shape).astype(np.int32)
    rec = drv.advance(_cap_scene(cfg, 1), base + 250, make_acts(acts), arch=labels)
    head = drv.society.head()
    # every cap is reached
    assert (
        len(rec["houses"]["id"]) == 16
        and len(rec["kin_groups"]["id"]) == 16
        and len(rec["colonies"]["id"]) == 16
    )
    assert len(rec["events"]["id"]) == SOCIAL_V1.events_max and rec["events_dropped"] > 0
    assert len(rec["relations"]["a"]) == SOCIAL_V1.relations_max and rec["relations_dropped"] > 0
    assert len(rec["temperament"]["id"]) == F + 1 and len(rec["territory"]["cells"]["id"]) == F
    assert rec["step"] == base + 250 and rec["origin"] == base and len(str(rec["houses"]["id"][0])) == 8
    he = rec["house_events"]
    assert len(he["founded"]["id"]) == SOCIAL_V1.events_max and he["founded_dropped"] > 0
    assert len(he["extinct"]["id"]) == SOCIAL_V1.events_max and he["extinct_dropped"] > 0
    assert len(str(he["founded"]["id"][0])) == 8 and len(str(he["extinct"]["id"][0])) == 8
    kg = rec["kin_groups"]
    assert None not in kg["births"] and None not in kg["deaths"] and None not in kg["arch"]
    assert None not in kg["x"] and len(kg["name"]) == 16 and all(len(a) == 10 for a in kg["arch"])
    # ... and every column is at its width
    assert {len(str(i)) for i in kg["id"]} == {6} and {len(n) for n in kg["name"]} == {17}
    assert {len(str(i)) for i in rec["colonies"]["id"]} == {7}
    rl = rec["relations"]
    assert {len(str(i)) for i in (*rl["a"], *rl["b"])} == {6}
    for col, digits in (
        ("attacks", 6),
        ("hits", 6),
        ("kills", 5),
        ("bonds", 6),
        ("bond_slots", 7),
        ("shared", 8),
    ):
        assert {len(str(v)) for v in rl[col]} == {digits}, col
    for col in ("exp_attacks", "exp_bonds"):
        assert all(len(f"{v:.3f}") >= 10 for v in rl[col]), col
    # the budgets. House names are regenerated at every snapshot: add what the longest possible name, a
    # 10-letter one with " <8-digit id>" (a collision among the top houses), would add to each of them.
    widest_house = len("B" * 10) + len(" 99999999")
    allowance = sum(widest_house - len(n) for n in rec["houses"]["name"])
    size, hsize = len(runio._dumps(rec)), len(runio._dumps(head))
    assert size + allowance <= RECORD_BUDGET, (size, allowance)
    assert hsize <= HEAD_BUDGET, hsize
    _valid_json(rec, head)


# --- the sidecar form ---------------------------------------------------------------------------------


def _same_arrays(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> bool:
    return a.keys() == b.keys() and all(a[k].dtype == b[k].dtype and np.array_equal(a[k], b[k]) for k in a)


def test_from_arrays_round_trip_and_continuation(tiny_cfg: Config) -> None:
    """``from_arrays(arrays())`` gives a society with the same arrays, and the two continue identically over
    snapshots with births, endings, relations and a colony: the records and the arrays stay equal."""
    cfg = tiny_cfg
    plan = [(10, 12, 18), (10, 12, 18), (6, 12, 18), (9, 12, 18), (9, 3, 18), (9, 12, 12), (10, 12, 18)]

    rng0 = np.random.default_rng(7)
    tags_a, tags_b = cloud(rng0, (2, 2, 2, 2), 10), cloud(rng0, (12, 12, 12, 12), 12)
    tags_c = np.concatenate([cloud(rng0, (4 + j, 6, 6, 6), 3) for j in range(6)])
    pos = rng0.uniform(0.0, cfg.world.size, (64, 2)).astype(np.float32)

    def scene(a: int, b: int, c: int) -> dict[str, np.ndarray]:
        """Rows 0-9 (A), 20-31 (B) and 40-57 (C) with the first a, b, c of them alive; A and C are bonded."""
        si = living_si(cfg, 0)
        for lo, tags, n in ((0, tags_a, a), (20, tags_b, b), (40, tags_c, c)):
            sl = slice(lo, lo + n)
            si["alive"][sl] = True
            si["uid"][sl] = np.arange(lo, lo + n)
            si["clade"][sl] = np.arange(lo, lo + n)
            si["tag"][sl] = tags[:n]
            si["pos"][sl] = pos[sl]
        for lo, n in ((0, min(a, 6)), (40, min(c, 6))):
            for r in range(lo, lo + n - 1):
                si["partner"][r, 0], si["partner"][r + 1, 1] = r + 1, r
        si["next_uid"] = np.array(100, np.int64)
        return si

    drv = SocDriver(cfg, scene(*plan[0]))
    for i, p in enumerate(plan[1:4], start=1):
        drv.advance(scene(*p), 250 * i)
    arrays = drv.society.arrays()
    assert set(arrays) == {
        "host.kin.system", "host.kin.key", "host.col.system", "host.col.key", "host.kin.next_id",
        "host.col.next_id", "host.prev_step", "host.prev_next_uid", "host.window_start", "host.house_origin",
        "host.n_ref", "host.kin.slots", "host.col.slots", "host.terr.ids", "host.terr.ema", "host.houses",
        "host.kin.names", "host.summary",
    }  # fmt: skip
    restored, why = Society.from_arrays(arrays, cfg, SOCIAL_V1, step=750)
    assert why == "" and restored is not None and _same_arrays(restored.arrays(), arrays)
    assert arrays["host.terr.ema"].shape == (len(arrays["host.terr.ids"]) + 1, 32 * 32)
    assert restored.head() == drv.society.head() and not restored.records
    # the same device state; the two continue in lockstep
    twin = SocDriver.__new__(SocDriver)
    twin.__dict__.update({**drv.__dict__, "society": restored})
    kinds = []
    for i, p in enumerate(plan[4:], start=4):
        r1 = drv.advance(scene(*p), 250 * i)
        r2 = twin.advance(scene(*p), 250 * i)
        assert json.dumps(r1) == json.dumps(r2), i
        kinds += [(f, k) for f, k, *_ in _events(r1)]
        assert _same_arrays(drv.society.arrays(), twin.society.arrays()), i
        assert json.dumps(drv.society.head()) == json.dumps(twin.society.head())
    assert ("kin_group", "dissolved") in kinds and ("kin_group", "born") in kinds  # events after the restore
    ma, sa = drv.society.territory()
    mb, sb = twin.society.territory()
    assert np.array_equal(ma.owner, mb.owner) and np.array_equal(sa, sb)
    # a society restored before any settle has no dominance state and starts it at the first settle
    fresh = SocDriver(cfg, scene(*plan[0]))
    again, why = Society.from_arrays(fresh.society.arrays(), cfg, SOCIAL_V1, step=0)
    assert why == "" and again is not None and again.territory()[0] is None
    assert (
        fresh.society.arrays()["host.terr.ema"].shape == (1, 1024)
        and not fresh.society.arrays()["host.terr.ema"].any()
    )


def _case(name: str, arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """``arrays`` with one thing wrong, named by ``name``."""
    a = {k: v.copy() for k, v in arrays.items()}
    kin_sys, kin_key = a["host.kin.system"], a["host.kin.key"]
    if name == "missing":
        a.pop("host.kin.key")
    elif name == "dtype":
        a["host.n_ref"] = a["host.n_ref"].astype(np.int32)
    elif name == "rank":
        a["host.prev_step"] = a["host.prev_step"].reshape(1)
    elif name == "summary_dtype":
        a["host.summary"] = np.array(5)
    elif name == "lengths":
        a["host.kin.key"] = kin_key[:-1]
    elif name == "next_id":
        a["host.col.next_id"] = np.array(0, np.int64)
    elif name == "system_range":
        kin_sys[0] = a["host.kin.next_id"]
    elif name == "system_zero":
        kin_sys[0] = 0
    elif name == "unsorted":
        a["host.kin.system"], a["host.kin.key"] = kin_sys[::-1].copy(), kin_key[::-1].copy()
    elif name == "duplicate_pair":
        a["host.kin.system"] = np.concatenate([kin_sys[:1], kin_sys])
        a["host.kin.key"] = np.concatenate([kin_key[:1], kin_key])
    elif name == "slots_shape":
        a["host.kin.slots"] = a["host.kin.slots"][:-1]
    elif name == "slot_unknown_id":
        a["host.kin.slots"][5] = 10_000
    elif name == "slot_twice":
        a["host.kin.slots"][5] = a["host.kin.slots"][0]
    elif name == "slot_below":
        a["host.col.slots"][4] = -5
    elif name == "terr_ids_twice":
        a["host.terr.ids"] = np.concatenate([a["host.terr.ids"], a["host.terr.ids"][:1]])
    elif name == "terr_ema_shape":
        a["host.terr.ema"] = a["host.terr.ema"][:-1]
    elif name == "summary_not_json":
        a["host.summary"] = np.array("{not json")
    elif name == "summary_not_object":
        a["host.summary"] = np.array("[1, 2]")
    elif name == "summary_keys":
        a["host.summary"] = np.array('{"houses": {}, "kin_groups": {}}')
    elif name == "prev_step_late":
        a["host.prev_step"] = np.array(751, np.int64)
    elif name == "window_start_late":
        a["host.window_start"] = np.array(800, np.int64)
    elif name == "origin_late":
        a["host.house_origin"] = np.array(751, np.int64)
    elif name == "origin_negative":
        a["host.house_origin"] = np.array(-3, np.int64)
    elif name == "n_ref":
        a["host.n_ref"] = np.array(-2, np.int64)
    elif name == "houses_dtype":
        a["host.houses"] = a["host.houses"].astype(np.int32)
    elif name == "houses_rank":
        a["host.houses"] = a["host.houses"].reshape(1, -1)
    elif name == "houses_unsorted":
        a["host.houses"] = a["host.houses"][::-1].copy()
    elif name == "houses_duplicate":
        a["host.houses"] = np.concatenate([a["host.houses"][:1], a["host.houses"]])
    elif name == "houses_negative":
        a["host.houses"][0] = -1
    elif name == "names_missing":
        a.pop("host.kin.names")
    elif name == "names_dtype":
        a["host.kin.names"] = np.array(5)
    elif name == "names_rank":
        a["host.kin.names"] = a["host.kin.names"].reshape(1)
    elif name == "names_not_json":
        a["host.kin.names"] = np.array("[[1,")
    elif name == "names_not_list":
        a["host.kin.names"] = np.array('{"1": "A"}')
    elif name == "names_item_short":
        a["host.kin.names"] = np.array('[[1, "A"]]')
    elif name == "names_empty_name":
        a["host.kin.names"] = np.array(_names_json(lambda items: [[items[0][0], "", None], *items[1:]], a))
    elif name == "names_bad_house":
        a["host.kin.names"] = np.array(_names_json(lambda items: [[items[0][0], "A", -4], *items[1:]], a))
    elif name == "names_bool_id":
        a["host.kin.names"] = np.array(_names_json(lambda items: [[True, "A", None], *items[1:]], a))
    elif name == "names_unsorted":
        a["host.kin.names"] = np.array(_names_json(lambda items: items[::-1], a))
    elif name == "names_duplicate":
        a["host.kin.names"] = np.array(_names_json(lambda items: [items[0], *items], a))
    elif name == "names_group_missing":
        a["host.kin.names"] = np.array(_names_json(lambda items: items[:-1], a))
    elif name == "names_group_extra":
        a["host.kin.names"] = np.array(_names_json(lambda items: [*items, [10_000, "Zzz", None]], a))
    elif name == "names_id_over_int64":  # a JSON int that no int64 array can hold
        a["host.kin.names"] = np.array('[[100000000000000000000, "X", null]]')
    elif name == "names_id_int64_max":  # a well-formed item, but not a tracked group
        a["host.kin.names"] = np.array(_names_json(lambda items: [*items, [2**63 - 1, "Zzz", None]], a))
    elif name == "names_house_over_int64":
        a["host.kin.names"] = np.array(_names_json(lambda items: [[items[0][0], "A", 2**63], *items[1:]], a))
    elif name == "names_too_deep":  # nested past the parser's recursion limit
        a["host.kin.names"] = np.array("[" * 100_000 + "]" * 100_000)
    elif name == "summary_too_deep":
        a["host.summary"] = np.array('{"houses": ' + "[" * 100_000 + "]" * 100_000 + "}")
    else:
        raise AssertionError(name)
    return a


def _names_json(edit: Any, arrays: dict[str, np.ndarray]) -> str:
    """The sidecar's kin names after ``edit`` (a function of the list of ``[id, name, house]`` items)."""
    return runio._dumps(edit(json.loads(str(arrays["host.kin.names"]))))


CASES = [
    ("missing", "host.kin.key"), ("dtype", "host.n_ref"), ("rank", "host.prev_step"),
    ("summary_dtype", "host.summary"), ("lengths", "differ in length"), ("next_id", "host.col.next_id"),
    ("system_range", "outside"), ("system_zero", "outside"), ("unsorted", "sorted"),
    ("duplicate_pair", "sorted"), ("slots_shape", "host.kin.slots"), ("slot_unknown_id", "host.kin.slots"),
    ("slot_twice", "twice"), ("slot_below", "host.col.slots"), ("terr_ids_twice", "host.terr.ids"),
    ("terr_ema_shape", "host.terr.ema"),
    ("summary_not_json", "JSON"), ("summary_not_object", "keys"), ("summary_keys", "keys"),
    ("prev_step_late", "host.prev_step"), ("window_start_late", "host.window_start"),
    ("origin_late", "host.house_origin"), ("origin_negative", "host.house_origin"), ("n_ref", "host.n_ref"),
    ("houses_dtype", "host.houses is int32"), ("houses_rank", "host.houses is int64 with 2"),
    ("houses_unsorted", "host.houses is not a strictly increasing"),
    ("houses_duplicate", "host.houses is not a strictly increasing"),
    ("houses_negative", "host.houses is not a strictly increasing"),
    ("names_missing", "member 'host.kin.names' is missing"), ("names_dtype", "host.kin.names is"),
    ("names_rank", "host.kin.names is"), ("names_not_json", "host.kin.names is not JSON"),
    ("names_not_list", "host.kin.names is not a list"), ("names_item_short", "holds an item"),
    ("names_empty_name", "holds an item"), ("names_bad_house", "holds an item"),
    ("names_bool_id", "holds an item"), ("names_unsorted", "sorted by id"),
    ("names_duplicate", "sorted by id"), ("names_group_missing", "exactly the tracked"),
    ("names_group_extra", "exactly the tracked"), ("names_id_over_int64", "holds an item"),
    ("names_id_int64_max", "exactly the tracked"), ("names_house_over_int64", "holds an item"),
    ("names_too_deep", "host.kin.names is not JSON"),
    ("summary_too_deep", "host.summary is not JSON"),
]  # fmt: skip


@pytest.mark.parametrize(("name", "needle"), CASES, ids=[c[0] for c in CASES])
def test_from_arrays_rejects(name: str, needle: str, tiny_cfg: Config) -> None:
    """One case per check of the load order: ``(None, reason)`` naming it, never an exception."""
    cfg = tiny_cfg
    si, _ = _sim_scene(cfg)
    drv = SocDriver(cfg, si)
    drv.advance(si, 250)
    drv.advance(si, 500)
    good = drv.society.arrays()
    assert Society.from_arrays(good, cfg, SOCIAL_V1, step=750)[0] is not None
    # a long enough membership list for the cases that edit its first entries
    assert good["host.kin.system"].size > 2 and (good["host.kin.slots"] >= 0).sum() >= 3
    assert good["host.houses"].size >= 2 and len(json.loads(str(good["host.kin.names"]))) >= 3
    soc, reason = Society.from_arrays(_case(name, good), cfg, SOCIAL_V1, step=750)
    assert soc is None and needle in reason, (name, reason)


def test_from_arrays_accepts_a_house_id_up_to_int64_max(tiny_cfg: Config) -> None:
    """The bound on the ids of a sidecar's names is the int64 range, inclusive: a house id of 2^63 - 1 is a
    well-formed item (2^63 is not, see ``CASES``)."""
    cfg = tiny_cfg
    si, _ = _sim_scene(cfg)
    drv = SocDriver(cfg, si)
    drv.advance(si, 250)
    arrays = drv.society.arrays()
    first = json.loads(str(arrays["host.kin.names"]))[0][0]
    arrays["host.kin.names"] = np.array(
        _names_json(lambda items: [[first, items[0][1], 2**63 - 1], *items[1:]], arrays)
    )
    soc, why = Society.from_arrays(arrays, cfg, SOCIAL_V1, step=250)
    assert soc is not None, why
    assert soc.kin_names[first][1] == 2**63 - 1


def test_from_arrays_rejects_a_foreign_society(tiny_cfg: Config) -> None:
    """Slots for another ``SocialRules.slots`` and a dominance grid of another size are refused."""
    cfg = tiny_cfg
    si, _ = _sim_scene(cfg)
    drv = SocDriver(cfg, si)
    drv.advance(si, 250)
    good = drv.society.arrays()
    for sr in (dataclasses.replace(SOCIAL_V1, slots=12), dataclasses.replace(SOCIAL_V1, terr_cells=16)):
        soc, reason = Society.from_arrays(good, cfg, sr)
        assert soc is None and reason, sr
