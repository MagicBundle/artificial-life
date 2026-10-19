"""The host half of the social layer (docs/GAMES.md §6, DESIGN §29.4–§29.8): ``Society`` tracks houses, kin
groups, colonies, territory, relations and temperament from the host copy of a snapshot (``social.Snap``).

``settle(h, step)`` turns one snapshot into the slots the device writes back (``kin_lut``, ``col_lut``) and
into the snapshot's record (a JSON-able dict, appended to ``Society.records``); ``initialise(h, step)`` is the
first snapshot of a fresh observer: ids and slots, no events, relations, territory or record. Everything is
deterministic: sorted arrays, ``np.unique``, ``np.lexsort``, float64 numpy and no RNG, so the same trajectory
under the same rules gives the same records, byte for byte.

``Society`` keeps two ``SystemTracker``s (kin groups and colonies), one ``Dominance`` map (territory), the
slot maps (slot -> id, sticky) and a few counters; ``arrays()`` and ``from_arrays`` are its sidecar form
(DESIGN §29.10). Every step of ``settle`` handles the empty cases (no alive row, no occupied bin, no tracked
group, no colony, no window ids): tables are empty with their columns, ratios None, no exception.

The roster (stage 7, DESIGN §29.14) adds three things to the record: each kin group is named once, at its
admission, and keeps the name for life (``Society.kin_names``, saved in the sidecar); the houses founded and
gone extinct since the last snapshot are listed (``house_events``, from the set of houses with live members
at consecutive snapshots, ``Society.houses``); and each listed kin group carries its window births and
deaths by cause (the device's window counts, by the window's slots), its centroid and its archetype mix.

Lint (rule 2, ``tests/test_lint.py``): no subscript assignment anywhere in this module, dict keys included.
Arrays are built with ``tracking`` helpers (``lookup_table``) and ``np.where``, dicts and tables by literals
and comprehensions.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Final, NamedTuple

import numpy as np

from alife import layout, metrics
from alife.analysis import phylo
from alife.config import Config, dims
from alife.game import social
from alife.game.rules import ARCHETYPE_NAMES
from alife.game.social import SOCIAL_VERSION, Expo, Rel, SocialRules
from alife.runio import _dumps
from alife.state import to_numpy
from alife.tracking import (
    Diagnostics,
    Dominance,
    DominanceMap,
    DominanceState,
    Event,
    EventKind,
    Members,
    SystemTracker,
    TrackerState,
    circular_mean,
    deadband,
    grid_clusters,
    lookup_table,
    rank_by_size,
    sticky_slots,
)

__all__ = [
    "CODA",
    "NUCLEUS",
    "ONSET",
    "TOP_HOUSES",
    "SnapInfo",
    "Society",
    "SocietyState",
    "host_snap",
    "house_names",
    "raw_house_names",
]

TOP_HOUSES: Final = 16  # houses listed in a record, by (-size, id)
HOST_KEYS: Final = (
    "host.kin.system",
    "host.kin.key",
    "host.col.system",
    "host.col.key",
    "host.kin.next_id",
    "host.col.next_id",
    "host.prev_step",
    "host.prev_next_uid",
    "host.window_start",
    "host.house_origin",
    "host.n_ref",
    "host.kin.slots",
    "host.col.slots",
    "host.terr.ids",
    "host.terr.ema",
    "host.houses",
    "host.kin.names",
    "host.summary",
)
SUMMARY_KEYS: Final = ("houses", "kin_groups", "colonies")
_SCALARS: Final = (
    "host.kin.next_id",
    "host.col.next_id",
    "host.prev_step",
    "host.prev_next_uid",
    "host.window_start",
    "host.house_origin",
    "host.n_ref",
)

ONSET: Final = (
    "b", "c", "d", "f", "g", "h", "k", "l", "m", "n", "p", "r", "s", "t", "v", "z",
    "br", "cr", "dr", "gr", "kr", "pr", "tr", "st", "sh", "th", "ch", "ph", "bl", "cl", "fl", "gl",
)  # fmt: skip
NUCLEUS: Final = ("a", "e", "i", "o", "u", "y", "ae", "ai", "au", "ea", "ei", "ia", "io", "oa", "ou", "ua")
CODA: Final = ("n", "r", "s", "l", "th", "nd", "rn", "st", "x", "m", "sk", "rd", "ll", "v", "ck", "sh")

_EVENT_NAMES: Final = tuple(k.name.lower() for k in EventKind)


class SocietyState(NamedTuple):
    """Everything a ``Society`` keeps between snapshots (its sidecar form, ``arrays()``)."""

    kin: TrackerState
    col: TrackerState
    kin_slots: np.ndarray  # (F,) int64 slot -> kin group id, -1 empty
    col_slots: np.ndarray  # (F,) int64 slot -> colony id, -1 empty
    terr: DominanceState
    prev_step: int  # the last snapshot's step (the origin before the first settle)
    prev_next_uid: int  # ``next_uid`` of the last snapshot: uids at or above it are new in the next window
    window_start: int  # the window of the next record is (window_start, step]
    house_origin: int  # the step the observer started fresh: houses are exact iff 0
    n_ref: int  # the reference population of the thresholds, -1 before the first snapshot
    summary: dict[str, Any]  # the three summary objects of the last settled snapshot
    houses: np.ndarray  # (m,) int64 ids of the houses with live members at the last snapshot, ascending
    kin_names: tuple[tuple[int, str, int], ...]  # (kin group id, name, named_after: house id or -1), by id


class SnapInfo(NamedTuple):
    """What the calibration tool reads about the last snapshot (not persisted, not in the record)."""

    step: int
    n_alive: int
    thresholds: dict[str, int]
    kin_diag: Diagnostics
    col_diag: Diagnostics
    kin_events: list[Event]
    col_events: list[Event]
    occupied_bins: int  # tag bins at or above ``occ_min``
    unaffiliated: int  # alive rows in no bin cluster
    kin_sizes: dict[int, int]  # tracked kin group id -> alive rows
    col_sizes: dict[int, int]  # tracked colony id -> alive rows
    sim_self: dict[int, float | None]  # slotted kin group id -> within-group similarity (empty at initialise)
    house_ids: np.ndarray  # (m,) int64 every house with live members, ascending (not only the top 16)
    house_sizes: np.ndarray  # (m,) int64 their live members


def host_snap(snap: social.Snap) -> dict[str, np.ndarray]:
    """The host copy of a ``Snap`` (``h`` of ``Society.settle``): one numpy array per field."""
    return {f: to_numpy(a) for f, a in zip(social.Snap._fields, snap, strict=True)}


# --- small helpers ----------------------------------------------------------------------------------------


def _r(x: Any, nd: int) -> float:
    return round(float(x), nd)


def _table(columns: tuple[str, ...], rows: list[tuple[Any, ...]]) -> dict[str, list[Any]]:
    """Equal-length column lists, rows in the given order; an empty table keeps its columns."""
    return {c: [row[j] for row in rows] for j, c in enumerate(columns)}


def _first_of_run(sorted_group: np.ndarray) -> np.ndarray:
    """Bool mask of the first element of each run of equal values in a sorted vector."""
    if sorted_group.size == 0:
        return np.zeros(0, dtype=bool)
    return np.concatenate([np.ones(1, dtype=bool), sorted_group[1:] != sorted_group[:-1]])


def raw_house_names(ids: np.ndarray, salt: int) -> list[str]:
    """The name of each house id before collisions are resolved: ``h = splitmix64(id, salt)``, five
    syllables of ``ONSET``/``NUCLEUS``/``CODA`` picked by bit fields of ``h``, capitalised."""
    h = phylo.splitmix64(np.asarray(ids, dtype=np.int64).reshape(-1).astype(np.uint64), salt)
    on1, nu1 = (h & np.uint64(31)).tolist(), ((h >> np.uint64(5)) & np.uint64(15)).tolist()
    on2, nu2 = ((h >> np.uint64(9)) & np.uint64(31)).tolist(), ((h >> np.uint64(14)) & np.uint64(15)).tolist()
    co = ((h >> np.uint64(18)) & np.uint64(15)).tolist()
    return [
        (ONSET[a] + NUCLEUS[b] + ONSET[c] + NUCLEUS[d] + CODA[e]).capitalize()
        for a, b, c, d, e in zip(on1, nu1, on2, nu2, co, strict=True)
    ]


def house_names(ids: np.ndarray, salt: int) -> list[str]:
    """Names of houses from their ids (the founder's uid), see ``raw_house_names``. A name shared within
    ``ids`` gets ``f" {id}"`` appended for every house after the first (by id)."""
    ident = np.asarray(ids, dtype=np.int64).reshape(-1).tolist()
    names = raw_house_names(np.asarray(ids), salt)
    first = {nm: min(i for n2, i in zip(names, ident, strict=True) if n2 == nm) for nm in set(names)}
    return [nm if i == first[nm] else f"{nm} {i}" for nm, i in zip(names, ident, strict=True)]


def _kernel_mean(ta: np.ndarray, tb: np.ndarray, scale2: float, same: bool) -> float | None:
    """Mean of ``exp(-|t_a - t_b|^2 / tag_scale^2)`` over the ordered pairs of two subsamples; for the same
    group (``same``) over the pairs ``a != b`` only, None below two members."""
    if same:
        n = ta.shape[0]
        if n < 2:
            return None
    d2 = ((ta[:, None, :] - tb[None, :, :]) ** 2).sum(axis=-1)
    k = np.exp(-d2 / scale2)
    if same:
        return _r(np.where(np.eye(n, dtype=bool), 0.0, k).sum() / (n * (n - 1)), 4)
    return _r(k.mean(), 4)


# --- the society ------------------------------------------------------------------------------------------


class Society:
    """Houses, kin groups, colonies, territory, relations and temperament of one run, from its snapshots."""

    def __init__(self, cfg: Config, sr: SocialRules, state: SocietyState | None = None) -> None:
        self.cfg, self.sr = cfg, sr
        self.d = dims(cfg)
        self._digest = sr.digest()
        self._cells = sr.terr_cells**2
        self.records: list[dict[str, Any]] = []  # settled snapshots whose record is not taken yet
        self.kin = SystemTracker(sr.theta, None if state is None else state.kin)
        self.col = SystemTracker(sr.theta, None if state is None else state.col)
        self.dom = Dominance(
            (sr.terr_cells, sr.terr_cells),
            sr.terr_k,
            sr.terr_min,
            periodic=(True, True),
            state=None if state is None else state.terr,
        )
        empty = np.full(sr.slots, -1, dtype=np.int64)
        self.kin_slots = empty.copy() if state is None else state.kin_slots.copy()
        self.col_slots = empty.copy() if state is None else state.col_slots.copy()
        self.prev_step = -1 if state is None else state.prev_step
        self.prev_next_uid = 0 if state is None else state.prev_next_uid
        self.window_start = -1 if state is None else state.window_start
        self.house_origin = -1 if state is None else state.house_origin
        self.n_ref = -1 if state is None else state.n_ref
        self.summary: dict[str, Any] | None = None if state is None else state.summary
        self.houses = np.zeros(0, dtype=np.int64) if state is None else state.houses.copy()
        # the live tracked kin groups' names: id -> (name, named_after: house id or -1), pruned per snapshot
        self.kin_names: dict[int, tuple[str, int]] = (
            {} if state is None else {i: (nm, after) for i, nm, after in state.kin_names}
        )
        self.info: SnapInfo | None = None  # the last snapshot's diagnostics (not persisted)

    # -- public state -------------------------------------------------------------------------------------

    @property
    def started(self) -> bool:
        """True once ``initialise`` has run (or the society was restored)."""
        return self.summary is not None

    @property
    def snapshot_step(self) -> int:
        """The step of the last settled snapshot (the origin before the first settle)."""
        return self.prev_step

    def state(self) -> SocietyState:
        if self.summary is None:
            raise RuntimeError("Society.state: not initialised")
        return SocietyState(
            self.kin.state(),
            self.col.state(),
            self.kin_slots.copy(),
            self.col_slots.copy(),
            self.dom.state(),
            self.prev_step,
            self.prev_next_uid,
            self.window_start,
            self.house_origin,
            self.n_ref,
            self.summary,
            self.houses.copy(),
            tuple((i, nm, after) for i, (nm, after) in sorted(self.kin_names.items())),
        )

    def head(self) -> dict[str, Any]:
        """The ``social`` head of a metrics row: the last settled snapshot's summary, keyed to its record."""
        if self.summary is None:
            raise RuntimeError("Society.head: not initialised")
        return {
            "social_version": SOCIAL_VERSION,
            "snapshot_step": self.prev_step,
            "origin": self.house_origin,
            **self.summary,
        }

    def take_records(self) -> list[dict[str, Any]]:
        """The records of the snapshots settled since the last call, in snapshot order; forgets them."""
        out, self.records = self.records, []
        return out

    def territory(self) -> tuple[DominanceMap | None, np.ndarray]:
        """The dominance map of the last settle (None before the first) and the kin slot -> id map."""
        return self.dom.current(), self.kin_slots.copy()

    # -- the two entry points -----------------------------------------------------------------------------

    def initialise(self, h: Mapping[str, np.ndarray], step: int) -> tuple[np.ndarray, np.ndarray]:
        """The first snapshot of a fresh observer, at ``step``: ids and slots, no events, relations,
        territory, temperament or record. Returns ``(kin_lut, col_lut)``."""
        if self.summary is not None:
            raise RuntimeError("Society.initialise: already initialised")
        self.house_origin = int(step)
        return self._snapshot(h, int(step), first=True)

    def settle(self, h: Mapping[str, np.ndarray], step: int) -> tuple[np.ndarray, np.ndarray]:
        """The snapshot at ``step``: updates every tracker and appends the snapshot's record to ``records``.
        Returns ``(kin_lut, col_lut)`` for ``social.relabel``."""
        if self.summary is None:
            raise RuntimeError("Society.settle: not initialised")
        if step <= self.prev_step:
            raise ValueError(f"snapshot step {step} does not follow the last one ({self.prev_step})")
        return self._snapshot(h, int(step), first=False)

    # -- one snapshot -------------------------------------------------------------------------------------

    def _snapshot(
        self, h: Mapping[str, np.ndarray], step: int, *, first: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        sr = self.sr
        uid = h["uid"]
        alive = uid >= 0
        n_alive = int(alive.sum())
        window_ids = np.append(self.kin_slots, -1)  # (G,) the window's slot -> id map, OTHER last
        n_ref, rescaled = deadband(self.n_ref, n_alive, sr.n_band)
        th = {
            "occ_min": max(sr.kin_occ[0], n_ref >> sr.kin_occ[1]),
            "kin_enter": max(sr.kin_enter[0], n_ref >> sr.kin_enter[1]),
            "kin_leave": max(sr.kin_leave[0], n_ref >> sr.kin_leave[1]),
        }
        uid_mode = sr.snapshot_every <= sr.uid_spacing
        credit = None
        if uid_mode and not first:  # newborns of the window, credited to their parent's system
            born = alive & (uid >= self.prev_next_uid) & (h["parent_uid"] >= 0)
            credit = (uid[born], h["parent_uid"][born])

        houses_summary, houses_table, house_ids, house_parent, house_sizes = self._houses(h, alive)
        house_events = None if first else self._house_events(step, house_ids, house_parent)
        kin = self._kin(h, step, alive, th, uid_mode, credit)
        names = self._name_groups(h, kin, houses_table)
        col = self._colonies(h, step, alive, uid_mode, credit, kin["row_id"])
        summary = {
            "houses": houses_summary,
            "kin_groups": {"tracked": kin["tracked"], "other": n_alive - kin["slotted_rows"]},
            "colonies": col["summary"],
        }
        sim_pair, sim_self = ({}, {}) if first else self._similarity(h, kin)
        self.info = SnapInfo(
            step,
            n_alive,
            {"n_ref": n_ref, "rescaled": rescaled, **th},
            kin["diag"],
            col["diag"],
            kin["events"],
            col["events"],
            kin["occupied"],
            kin["unaffiliated"],
            kin["sizes"],
            col["sizes"],
            sim_self,
            house_ids,
            house_sizes,
        )
        if not first:
            self.records.append(
                self._record(
                    h,
                    step,
                    summary,
                    (n_ref, rescaled, th),
                    houses_table,
                    house_events,
                    kin,
                    col,
                    window_ids,
                    sim_pair,
                    sim_self,
                    names,
                )
            )
        self.houses, self.kin_names = house_ids, names
        self.kin_slots, self.col_slots = kin["slots"], col["slots"]
        self.window_start = self.prev_step = step
        self.prev_next_uid = int(h["next_uid"])
        self.n_ref, self.summary = n_ref, summary
        return kin["lut"], col["lut"]

    # -- houses -------------------------------------------------------------------------------------------

    def _houses(
        self, h: Mapping[str, np.ndarray], alive: np.ndarray
    ) -> tuple[dict[str, Any], dict[str, list[Any]], np.ndarray, np.ndarray, np.ndarray]:
        """``(summary, top table, ids, parents, sizes)``: ``ids`` are all the houses with live members
        (ascending), ``parents`` the house each was founded from (``house_from`` of its first alive row, which
        every member shares; -1 where none or unknown) and ``sizes`` their live members."""
        house = h["house"]
        sel = alive & (house >= 0)
        ids, first, counts = np.unique(house[sel], return_index=True, return_counts=True)
        clade = h["clade"][sel][first]  # the clade of the house's first alive row
        parents = h["house_from"][sel][first]
        top = np.lexsort((ids, -counts))[:TOP_HOUSES]
        names = house_names(ids[top], self.sr.name_salt)
        table = _table(
            ("id", "name", "size", "clade"),
            [
                (int(i), nm, int(c), int(cl))
                for i, nm, c, cl in zip(ids[top], names, counts[top], clade[top], strict=True)
            ],
        )
        summary = {
            "alive": int(ids.size),
            "exact": self.house_origin == 0,
            "unknown_immigrant": int((alive & (house == social.HOUSE_UNKNOWN)).sum()),
        }
        return summary, table, ids, parents, counts

    def _house_events(self, step: int, ids: np.ndarray, parents: np.ndarray) -> dict[str, Any]:
        """The houses founded (live members now, none at the last snapshot) and gone extinct (the reverse)
        since the last snapshot, each by ascending id and listed up to ``events_max`` with the rest counted.
        A house founded and extinct inside one window is seen by neither snapshot. ``step`` is the
        snapshot's: the step at which the change was observed."""
        cap = self.sr.events_max
        born = np.setdiff1d(ids, self.houses, assume_unique=True)
        gone = np.setdiff1d(self.houses, ids, assume_unique=True)
        listed = born[:cap]
        parent = parents[np.searchsorted(ids, listed)]
        founded = _table(
            ("id", "step", "parent_house"),
            [(int(i), step, int(pa) if pa >= 0 else None) for i, pa in zip(listed, parent, strict=True)],
        )
        extinct = _table(("id", "step"), [(int(i), step) for i in gone[:cap]])
        return {
            "founded": founded,
            "extinct": extinct,
            "founded_dropped": int(born.size - listed.size),
            "extinct_dropped": int(gone.size - min(gone.size, cap)),
        }

    # -- kin groups ---------------------------------------------------------------------------------------

    def _kin(
        self,
        h: Mapping[str, np.ndarray],
        step: int,
        alive: np.ndarray,
        th: dict[str, int],
        uid_mode: bool,
        credit: tuple[np.ndarray, np.ndarray] | None,
    ) -> dict[str, Any]:
        sr = self.sr
        other = sr.slots
        nbins = sr.kin_bins**layout.T
        bins = h["bins"].astype(np.int64)
        root = grid_clusters(h["bin_counts"], (sr.kin_bins,) * layout.T, th["occ_min"])
        g = np.where(alive, root[bins], -1)
        labs, rows = np.unique(g[g >= 0], return_counts=True)
        cand, adm = labs[rows >= th["kin_leave"]], labs[rows >= th["kin_enter"]]
        if uid_mode:
            sel = alive & np.isin(g, cand)
            members = Members(g[sel], h["uid"][sel])
        else:  # footprint: the occupied and attached bins of the candidates
            cells = np.flatnonzero(np.isin(root, cand))
            members = Members(root[cells], cells)
        asg, events, diag = self.kin.observe(step, members, credit, adm)

        held = asg.system >= 0
        ids, tlabels = asg.system[held], asg.label[held]
        trows = rows[np.searchsorted(labs, tlabels)]
        slots = sticky_slots(self.kin_slots, rank_by_size(ids, trows), sr.slot_slack)
        slot_of = {int(i): s for s, i in enumerate(slots.tolist()) if i >= 0}
        tslot = np.array([slot_of.get(i, other) for i in ids.tolist()], dtype=np.int64)
        lab_slot = lookup_table(nbins, tlabels, tslot, other)
        lut = np.where(root >= 0, lab_slot[np.maximum(root, 0)], other).astype(np.int32)
        lab_id = lookup_table(nbins, tlabels, ids, -1)
        row_slot = np.where(alive, lut[bins], -1)
        return {
            "events": events,
            "diag": diag,
            "tracked": int(ids.size),
            "slots": slots,
            "named": np.flatnonzero(slots >= 0),  # the slots that hold an id
            "lut": lut,
            "row_id": np.where(g >= 0, lab_id[np.maximum(g, 0)], -1),  # the kin group id of every row
            "row_slot": row_slot,
            "slotted_rows": int(((row_slot >= 0) & (row_slot < other)).sum()),
            "sizes": dict(zip(ids.tolist(), trows.tolist(), strict=True)),
            "occupied": int((h["bin_counts"] >= th["occ_min"]).sum()),
            "unaffiliated": int((alive & (g < 0)).sum()),
        }

    def _name_groups(
        self, h: Mapping[str, np.ndarray], kin: dict[str, Any], houses_table: dict[str, list[Any]]
    ) -> dict[int, tuple[str, int]]:
        """The names of the tracked kin groups after this snapshot: ``{id: (name, named_after)}``.

        A group keeps the name it was admitted with. A group admitted now (no entry yet), in id order, takes
        the name of the top-16 house that has the most of its members (ties to the smaller house id) and
        ``named_after`` is that house's id; with no member in a top-16 house its name is generated from the
        house name generator on ``splitmix64(id, kin_name_salt)`` and ``named_after`` is -1. A name already
        held by a live tracked group (kept, or given earlier in this loop) gets ``f" {id}"`` appended, again
        if that name is held too. Entries of groups no longer tracked are dropped, so a name can be held
        again by a later group.
        """
        top = dict(zip(houses_table["id"], houses_table["name"], strict=True))
        top_ids = np.array(sorted(top), dtype=np.int64)
        live = sorted(kin["sizes"])
        names = {i: self.kin_names[i] for i in live if i in self.kin_names}
        held = {nm for nm, _ in names.values()}
        house, row_id = h["house"], kin["row_id"]
        for gid in live:
            if gid in names:
                continue
            members = house[row_id == gid]
            members = members[np.isin(members, top_ids)]
            if members.size:
                cand, count = np.unique(members, return_counts=True)
                best = int(cand[np.lexsort((cand, -count))[0]])
                name, after = top[best], best
            else:
                name, after = house_names(np.array([gid]), self.sr.kin_name_salt)[0], -1
            while name in held:
                name = f"{name} {gid}"
            held.add(name)
            names = {**names, gid: (name, after)}
        return names

    def _kin_table(
        self,
        h: Mapping[str, np.ndarray],
        kin: dict[str, Any],
        sim_self: dict[int, float | None],
        window_ids: np.ndarray,
        names: dict[int, tuple[str, int]],
    ) -> dict[str, list[Any]]:
        """The ``kin_groups`` table: one row per slotted group, by slot.

        ``name`` and ``named_after`` are the group's roster entry. ``x`` and ``y`` are the centroid of its
        members (``circular_mean`` over the torus; ``z_mean`` is the height), ``arch`` the shares of the
        primary archetypes (``rules.ARCHETYPE_NAMES`` order) among its members that have a label (None if
        none has). ``births`` and the ``deaths`` by cause (``social.DEATH_CAUSES`` order) are the window's,
        counted for the slot the group had in the window, so they are None for a group that held none then.
        """
        sr, d = self.sr, self.d
        f = sr.slots
        slots, named, row_slot = kin["slots"], kin["named"], kin["row_slot"]
        sel = (row_slot >= 0) & (row_slot < f)
        rs = row_slot[sel].astype(np.int64)
        size = np.bincount(rs, minlength=f)[:f]
        tag = h["tag"][sel].astype(np.float64)
        denom = np.maximum(size, 1)
        mean_tag = np.stack(
            [np.bincount(rs, weights=tag[:, j], minlength=f)[:f] / denom for j in range(layout.T)], axis=1
        )
        code = (rs << 32) | h["clade"][sel].astype(np.int64)
        n_arch = len(ARCHETYPE_NAMES)
        arch = np.bincount(rs * (n_arch + 1) + h["arch"][sel], minlength=f * (n_arch + 1))
        arch = arch.reshape(f, n_arch + 1)[:, 1:]  # members per primary archetype, unlabelled left out
        order = np.argsort(rs, kind="stable")
        pos = h["pos"][sel][order]
        start = np.searchsorted(rs[order], np.arange(f + 1))  # the rows of slot s: pos[start[s]:start[s + 1]]
        win = {int(i): s for s, i in enumerate(window_ids[:f].tolist()) if i >= 0}
        u, c = np.unique(code, return_counts=True)
        su, cu = u >> 32, u & 0xFFFFFFFF
        o = np.lexsort((cu, -c, su))
        top = o[_first_of_run(su[o])]  # per slot: the clade with the most rows, ties to the smaller clade
        dom_clade = lookup_table(f, su[top], cu[top], -1)
        dom_n = lookup_table(f, su[top], c[top], 0)
        columns = [
            "id", "name", "named_after", "slot", "size", "tag", "clade", "clade_share", "sim_self",
            "x", "y", "arch", "births", "deaths",
        ]  # fmt: skip
        if d.d3:
            z = h["pos"][sel][:, 2].astype(np.float64)
            z_mean = np.bincount(rs, weights=z, minlength=f)[:f] / denom
            band = metrics.z_band(z, self.cfg.world.height)
            frac = np.bincount(rs * metrics.N_BANDS + band, minlength=f * metrics.N_BANDS)
            frac = frac.reshape(f, metrics.N_BANDS) / denom[:, None]
            columns += ["z_mean", "band_frac"]
        rows = []
        for s in named.tolist():
            gid = int(slots[s])
            name, after = names[gid]
            at = pos[start[s] : start[s + 1]]
            centre = (
                [_r(circular_mean(at[:, j], self.cfg.world.size), 2) for j in (0, 1)]
                if at.shape[0]
                else [None, None]
            )
            labelled = int(arch[s].sum())
            w = win.get(gid)
            row = (
                gid,
                name,
                after if after >= 0 else None,
                s,
                int(size[s]),
                [_r(v, 3) for v in mean_tag[s]],
                int(dom_clade[s]),
                _r(dom_n[s] / max(size[s], 1), 4),
                sim_self.get(gid),
                *centre,
                [_r(v / labelled, 3) for v in arch[s]] if labelled else None,
                None if w is None else int(h["births"][w]),
                None if w is None else [int(v) for v in h["deaths"][:, w]],
            )
            if d.d3:
                row += (_r(z_mean[s], 2), [_r(v, 4) for v in frac[s]])
            rows.append(row)
        return _table(tuple(columns), rows)

    def _similarity(
        self, h: Mapping[str, np.ndarray], kin: dict[str, Any]
    ) -> tuple[dict[tuple[int, int], float | None], dict[int, float | None]]:
        """Mean tag similarity between every pair of slotted groups (keyed ``(smaller id, larger id)``) and
        within each (``sim_self``, None below two members), over subsamples of the ``sim_sample`` members
        with the smallest ``splitmix64(uid, sim_salt)``."""
        sr = self.sr
        f, m = sr.slots, sr.sim_sample
        row_slot = kin["row_slot"]
        sel = np.flatnonzero((row_slot >= 0) & (row_slot < f))
        rs, uid = row_slot[sel].astype(np.int64), h["uid"][sel]
        hv = phylo.splitmix64(uid.astype(np.uint64), sr.sim_salt)
        order = np.lexsort((uid, hv, rs))  # by slot, then the hash, then the uid
        sel, rs = sel[order], rs[order]
        start = np.searchsorted(rs, np.arange(f))
        count = np.bincount(rs, minlength=f)[:f]
        named = kin["named"].tolist()
        tags = {
            s: h["tag"][sel[start[s] : start[s] + min(int(count[s]), m)]].astype(np.float64) for s in named
        }
        ids = {s: int(kin["slots"][s]) for s in named}
        scale2 = self.cfg.sensing.tag_scale**2
        pair = {
            (min(ids[a], ids[b]), max(ids[a], ids[b])): _kernel_mean(tags[a], tags[b], scale2, False)
            for i, a in enumerate(named)
            for b in named[i + 1 :]
        }
        diag = {ids[a]: _kernel_mean(tags[a], tags[a], scale2, True) for a in named}
        return pair, diag

    # -- colonies -----------------------------------------------------------------------------------------

    def _colonies(
        self,
        h: Mapping[str, np.ndarray],
        step: int,
        alive: np.ndarray,
        uid_mode: bool,
        credit: tuple[np.ndarray, np.ndarray] | None,
        row_kin: np.ndarray,
    ) -> dict[str, Any]:
        sr = self.sr
        n = alive.shape[0]
        comp = h["comp"].astype(np.int64)
        labs, size = np.unique(comp[alive], return_counts=True)
        cand, adm = labs[size >= sr.colony_leave], labs[size >= sr.colony_enter]
        rows = alive & np.isin(comp, cand)
        if uid_mode:
            members = Members(comp[rows], h["uid"][rows])
        else:  # footprint: the distinct floor cells of a colony's members
            code = np.unique((comp[rows] << 32) | h["foot"][rows].astype(np.int64))
            members = Members(code >> 32, code & 0xFFFFFFFF)
        asg, events, diag = self.col.observe(step, members, credit, adm)

        held = asg.system >= 0
        ids, tlabels = asg.system[held], asg.label[held]
        trows = size[np.searchsorted(labs, tlabels)]
        slots = sticky_slots(self.col_slots, rank_by_size(ids, trows), sr.slot_slack)
        slot_of = {int(i): s for s, i in enumerate(slots.tolist()) if i >= 0}
        # a tracked colony without a slot is "other tracked" (F); an untracked component is -1
        tslot = np.array([slot_of.get(i, sr.slots) for i in ids.tolist()], dtype=np.int64)
        lut = lookup_table(n, tlabels, tslot, -1)

        label_of = dict(zip(ids.tolist(), tlabels.tolist(), strict=True))
        table_rows = []
        for s in np.flatnonzero(slots >= 0).tolist():
            mem = alive & (comp == label_of[int(slots[s])])
            kin_ids, kin_n = np.unique(row_kin[mem][row_kin[mem] >= 0], return_counts=True)
            kin_id = int(kin_ids[np.argmax(kin_n)]) if kin_ids.size else -1  # the plurality, ties: smaller id
            pos = h["pos"][mem]
            row = (
                int(slots[s]),
                s,
                int(mem.sum()),
                kin_id,
                _r(circular_mean(pos[:, 0], self.cfg.world.size), 2),
                _r(circular_mean(pos[:, 1], self.cfg.world.size), 2),
            )
            if self.d.d3:
                row += (_r(pos[:, 2].astype(np.float64).mean(), 2),)
            table_rows.append(row)
        return {
            "events": events,
            "diag": diag,
            "slots": slots,
            "lut": lut,
            "table_rows": table_rows,
            "sizes": dict(zip(ids.tolist(), trows.tolist(), strict=True)),
            "summary": {
                "tracked": int(ids.size),
                "members": int(trows.sum()),
                "largest": int(trows.max()) if trows.size else 0,
                "converged": bool(h["converged"]),
                "match": "uid" if uid_mode else "footprint",
            },
        }

    # -- the record ---------------------------------------------------------------------------------------

    def _record(
        self,
        h: Mapping[str, np.ndarray],
        step: int,
        summary: dict[str, Any],
        thresholds: tuple[int, bool, dict[str, int]],
        houses_table: dict[str, list[Any]],
        house_events: dict[str, Any] | None,
        kin: dict[str, Any],
        col: dict[str, Any],
        window_ids: np.ndarray,
        sim_pair: dict[tuple[int, int], float | None],
        sim_self: dict[int, float | None],
        names: dict[int, tuple[str, int]],
    ) -> dict[str, Any]:
        sr = self.sr
        n_ref, rescaled, th = thresholds
        col_columns = ("id", "slot", "size", "kin_group", "x", "y") + (("z",) if self.d.d3 else ())
        events = [("kin_group", e) for e in kin["events"]] + [("colony", e) for e in col["events"]]
        events.sort(key=lambda fe: (-fe[1].size, fe[1].kind.value, fe[1].system, fe[0]))
        listed = events[: sr.events_max]
        event_table = _table(
            ("step", "family", "kind", "id", "size", "other", "overlap"),
            [(e.step, fam, e.kind.name.lower(), e.system, e.size, e.other, e.overlap) for fam, e in listed],
        )
        counts = {
            fam: {nm: sum(1 for e in evs if e.kind.name.lower() == nm) for nm in _EVENT_NAMES}
            for fam, evs in (("kin_group", kin["events"]), ("colony", col["events"]))
        }
        relations, dropped = self._relations(h, window_ids, sim_pair, sim_self)
        return {
            "step": step,
            "social_version": SOCIAL_VERSION,
            "social_digest": self._digest,
            "origin": self.house_origin,
            "window_start": self.window_start,
            "summary": summary,
            "thresholds": {
                "n_ref": n_ref,
                "rescaled": rescaled,
                "occ_min": th["occ_min"],
                "kin_enter": th["kin_enter"],
                "kin_leave": th["kin_leave"],
            },
            "houses": houses_table,
            "house_events": house_events,
            "kin_groups": self._kin_table(h, kin, sim_self, window_ids, names),
            "colonies": _table(col_columns, col["table_rows"]),
            "events": event_table,
            "events_dropped": len(events) - len(listed),
            "event_counts": counts,
            "relations": relations,
            "relations_dropped": dropped,
            "temperament": self._temperament(h, window_ids),
            "territory": self._territory(h, window_ids),
        }

    # -- relations ----------------------------------------------------------------------------------------

    def _relations(
        self,
        h: Mapping[str, np.ndarray],
        window_ids: np.ndarray,
        sim_pair: dict[tuple[int, int], float | None],
        sim_self: dict[int, float | None],
    ) -> tuple[dict[str, list[Any]], int]:
        sr = self.sr
        rel, expo, shared = h["rel"], h["expo"], h["shared"]
        att, hit, kill = rel[Rel.ATTACKS.value], rel[Rel.HITS.value], rel[Rel.KILLS.value]
        bond, bslot = rel[Rel.BONDS.value], rel[Rel.BOND_SLOTS.value]
        k = expo.shape[-1]
        w = 1.0 / np.arange(1, k + 1, dtype=np.float64)  # 1 / (j + 1) for the n_i - 1 axis
        exp_att = (expo[Expo.ATTACK.value].astype(np.float64) * w).sum(axis=-1)
        exp_bond = (expo[Expo.BOND.value].astype(np.float64) * w).sum(axis=-1)
        a_att, a_bond = att.sum(axis=1), bond.sum(axis=1)  # the actor slot's totals
        ia, ib = np.nonzero((att + bond + shared) > 0)
        id_a, id_b = window_ids[ia], window_ids[ib]
        order = np.lexsort((id_b, id_a, -shared[ia, ib], -(att[ia, ib] + bond[ia, ib])))
        kept = order[: sr.relations_max]
        kept = kept[np.lexsort((id_b[kept], id_a[kept]))]  # listed by (a, b)
        rows = []
        for j in kept.tolist():
            a, b = int(ia[j]), int(ib[j])
            ida, idb = int(id_a[j]), int(id_b[j])
            if ida >= 0 and idb >= 0 and ida != idb:
                sim = sim_pair.get((min(ida, idb), max(ida, idb)))
            elif ida >= 0 and ida == idb:
                sim = sim_self.get(ida)
            else:
                sim = None
            distinct = (
                None if sim is None or ida == idb else bool(sr.sim_distinct[1] * sim < sr.sim_distinct[0])
            )
            rows.append(
                (
                    ida,
                    idb,
                    int(att[a, b]),
                    int(hit[a, b]),
                    int(kill[a, b]),
                    int(bond[a, b]),
                    int(shared[a, b]),
                    int(bslot[a, b]),
                    _r(exp_att[a, b], 3) if a_att[a] > 0 else None,
                    _r(exp_bond[a, b], 3) if a_bond[a] > 0 else None,
                    sim,
                    distinct,
                )
            )
        columns = (
            "a",
            "b",
            "attacks",
            "hits",
            "kills",
            "bonds",
            "shared",
            "bond_slots",
            "exp_attacks",
            "exp_bonds",
            "sim",
            "distinct",
        )
        return _table(columns, rows), int(order.size - kept.size)

    # -- temperament and territory (the window's slots) ---------------------------------------------------

    def _temperament(self, h: Mapping[str, np.ndarray], window_ids: np.ndarray) -> dict[str, list[Any]]:
        q, rows = h["q_sums"], h["q_rows"]
        f = self.sr.slots
        listed = [s for s in range(f + 1) if window_ids[s] >= 0 or (s == f and rows[s] > 0)]

        def ratio(pos: int, neg: int, s: int) -> float | None:
            return _r((int(q[pos, s]) - int(q[neg, s])) / (256 * int(rows[s])), 4) if rows[s] > 0 else None

        return _table(
            ("id", "q_attack_kin", "q_bond_kin", "rows"),
            [(int(window_ids[s]), ratio(0, 1, s), ratio(2, 3, s), int(rows[s])) for s in listed],
        )

    def _territory(self, h: Mapping[str, np.ndarray], window_ids: np.ndarray) -> dict[str, Any]:
        f = self.sr.slots
        named = np.flatnonzero(window_ids[:f] >= 0)
        counts = h["terr"].reshape(self._cells, f + 1)[:, np.append(named, f)].astype(np.int64)
        ids = window_ids[named]
        dmap = self.dom.update(ids, counts)
        owned = dmap.owner >= 0
        n_owned = int(owned.sum())
        purity = _r(int(dmap.purity_q16[owned].sum()) / (n_owned * 65536), 4) if n_owned else None
        return {
            "owned_cells": n_owned,
            "purity_mean": purity,
            "cells": _table(("id", "cells"), [(int(i), int((dmap.owner == i).sum())) for i in ids]),
        }

    # -- the sidecar form ---------------------------------------------------------------------------------

    def arrays(self) -> dict[str, np.ndarray]:
        """The host members of the sidecar (DESIGN §29.10): numpy arrays keyed ``host.*``."""
        st = self.state()
        terr = st.terr
        ema = (
            np.zeros((1, self._cells), dtype=np.int64)
            if terr.ema is None
            else np.ascontiguousarray(terr.ema.T)
        )
        scalars = {
            "host.kin.next_id": st.kin.next_id,
            "host.col.next_id": st.col.next_id,
            "host.prev_step": st.prev_step,
            "host.prev_next_uid": st.prev_next_uid,
            "host.window_start": st.window_start,
            "host.house_origin": st.house_origin,
            "host.n_ref": st.n_ref,
        }
        return {
            "host.kin.system": st.kin.system,
            "host.kin.key": st.kin.key,
            "host.col.system": st.col.system,
            "host.col.key": st.col.key,
            **{name: np.array(v, dtype=np.int64) for name, v in scalars.items()},
            "host.kin.slots": st.kin_slots,
            "host.col.slots": st.col_slots,
            "host.terr.ids": terr.ids,
            "host.terr.ema": ema,
            "host.houses": st.houses,
            # the roster's names, by id: [[id, name, house id or null], ...]
            "host.kin.names": np.array(
                _dumps([[i, nm, after if after >= 0 else None] for i, nm, after in st.kin_names])
            ),
            # insertion order, never sort_keys: a resumed head must keep the key order of an uninterrupted run
            "host.summary": np.array(_dumps(st.summary)),
        }

    @classmethod
    def from_arrays(
        cls, arrays: Mapping[str, np.ndarray], cfg: Config, sr: SocialRules, *, step: int | None = None
    ) -> tuple[Society | None, str]:
        """``(society, "")`` from a sidecar's ``host.*`` members if they are valid for ``cfg`` and ``sr``,
        else ``(None, reason)``; never raises. ``step`` (the sidecar's step), when given, bounds
        ``prev_step``, ``window_start`` and ``house_origin``."""
        parsed = _parse_host(arrays, sr, sr.terr_cells**2, step)
        if isinstance(parsed, str):
            return None, parsed
        try:
            return cls(cfg, sr, parsed), ""
        except ValueError as e:  # a tracker or the dominance map rejecting its state
            return None, f"the host state is invalid ({e})"


def _parse_host(
    arrays: Mapping[str, np.ndarray], sr: SocialRules, cells: int, step: int | None
) -> SocietyState | str:
    """The ``SocietyState`` of a sidecar's host members, or the reason it is not valid (checks in order)."""
    for name in HOST_KEYS:
        if name not in arrays:
            return f"member {name!r} is missing"
        if not isinstance(arrays[name], np.ndarray):
            return f"member {name!r} is not an array"
    ranks = {
        **{name: 0 for name in _SCALARS},
        **{
            name: 1
            for name in (*HOST_KEYS[:4], "host.kin.slots", "host.col.slots", "host.terr.ids", "host.houses")
        },
        "host.terr.ema": 2,
        "host.kin.names": 0,
        "host.summary": 0,
    }
    for name, rank in ranks.items():
        a = arrays[name]
        if name in ("host.summary", "host.kin.names"):
            if a.dtype.kind != "U" or a.ndim != 0:
                return f"{name} is {a.dtype} with {a.ndim} dimensions, expected a 0-d string"
        elif a.dtype != np.int64 or a.ndim != rank:
            return f"{name} is {a.dtype} with {a.ndim} dimensions, expected int64 with {rank}"
    out: dict[str, TrackerState] = {}
    for fam in ("kin", "col"):
        system, key = arrays[f"host.{fam}.system"], arrays[f"host.{fam}.key"]
        if system.shape != key.shape:
            return f"host.{fam}.system and host.{fam}.key differ in length"
        next_id = int(arrays[f"host.{fam}.next_id"])
        if next_id < 1:
            return f"host.{fam}.next_id is {next_id}, expected >= 1"
        if system.size and (system.min() < 1 or system.max() >= next_id):
            return f"host.{fam}.system holds an id outside [1, next_id)"
        later = (key[1:] > key[:-1]) | ((key[1:] == key[:-1]) & (system[1:] > system[:-1]))
        if not later.all():
            return f"host.{fam}.* memberships are not sorted by (key, system) without duplicates"
        slots = arrays[f"host.{fam}.slots"]
        if slots.shape != (sr.slots,):
            return f"host.{fam}.slots has shape {slots.shape}, expected {(sr.slots,)}"
        held = slots[slots >= 0]
        if (slots < -1).any() or not np.isin(held, system).all():
            return f"host.{fam}.slots holds an id that is not in the memberships"
        if np.unique(held).size != held.size:
            return f"host.{fam}.slots holds an id twice"
        out = {**out, fam: TrackerState(system.copy(), key.copy(), next_id, int(arrays["host.prev_step"]))}
    ids, ema = arrays["host.terr.ids"], arrays["host.terr.ema"]
    if np.unique(ids).size != ids.size:
        return "host.terr.ids holds an id twice"
    if ema.shape != (ids.size + 1, cells):
        return f"host.terr.ema has shape {ema.shape}, expected {(ids.size + 1, cells)}"
    houses = arrays["host.houses"]
    if (houses < 0).any() or (houses[1:] <= houses[:-1]).any():
        return "host.houses is not a strictly increasing array of ids >= 0"
    names = _parse_names(arrays["host.kin.names"], np.unique(out["kin"].system))
    if isinstance(names, str):
        return names
    try:
        summary = json.loads(str(arrays["host.summary"]))
    except ValueError:
        return "host.summary is not JSON"
    if not isinstance(summary, dict) or not set(SUMMARY_KEYS) <= set(summary):
        return f"host.summary is not an object with the keys {list(SUMMARY_KEYS)}"
    prev, start, origin, n_ref = (
        int(arrays[f"host.{k}"]) for k in ("prev_step", "window_start", "house_origin", "n_ref")
    )
    if n_ref < -1:
        return f"host.n_ref is {n_ref}, expected >= -1"
    for name, v in (("prev_step", prev), ("window_start", start), ("house_origin", origin)):
        if v < 0 or (step is not None and v > step):
            return (
                f"host.{name} is {v}, expected in [0, {step}]" if step is not None else f"host.{name} is {v}"
            )
    terr = DominanceState(ids.copy(), ema.T.copy() if prev > origin else None)
    return SocietyState(
        out["kin"],
        out["col"],
        arrays["host.kin.slots"].copy(),
        arrays["host.col.slots"].copy(),
        terr,
        prev,
        int(arrays["host.prev_next_uid"]),
        start,
        origin,
        n_ref,
        summary,
        houses.copy(),
        names,
    )


def _parse_names(raw: np.ndarray, tracked: np.ndarray) -> tuple[tuple[int, str, int], ...] | str:
    """The kin names of a sidecar (``[[id, name, house id or null], ...]`` by id), or the reason they are not
    valid: well-formed items, ascending ids, and exactly the ids of the tracked kin groups."""
    try:
        items = json.loads(str(raw))
    except ValueError:
        return "host.kin.names is not JSON"
    if not isinstance(items, list):
        return "host.kin.names is not a list"
    out: list[tuple[int, str, int]] = []
    for item in items:
        ok = (
            isinstance(item, list)
            and len(item) == 3
            and type(item[0]) is int
            and item[0] >= 1
            and isinstance(item[1], str)
            and item[1] != ""
            and (item[2] is None or (type(item[2]) is int and item[2] >= 0))
        )
        if not ok:
            return "host.kin.names holds an item that is not [id >= 1, name, house id or null]"
        out.append((item[0], item[1], -1 if item[2] is None else item[2]))
    ids = np.array([i for i, _, _ in out], dtype=np.int64)
    if (ids[1:] <= ids[:-1]).any():
        return "host.kin.names is not sorted by id without duplicates"
    if not np.array_equal(ids, tracked):
        return "host.kin.names does not name exactly the tracked kin groups"
    return tuple(out)
