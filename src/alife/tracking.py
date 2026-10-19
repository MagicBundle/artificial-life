"""Persistent ids for tracked systems, on the host (DESIGN §29.3): numpy, deterministic, no game names.

A *system* is a set of *members* identified by int64 keys (agent uids, grid cells, bins). ``SystemTracker``
gives each system an id that survives from snapshot to snapshot by one-to-one greedy matching on the overlap
of member sets, and reports what happened to the others as events (born, split, merged, dissolved). The other
parts are the host halves of a spatial tracker: ``grid_clusters`` (clusters of occupied bins on an n-d grid),
``Dominance`` (owner, purity and borders of a smoothed 2-D count map) and small helpers (sticky slots, a
dead-band for a reference population, lookup tables).

Everything is sorted-array numpy (``np.unique``, ``np.lexsort``, ``searchsorted``) over int64, so the output
depends only on the inputs: no RNG, no hash order, no float comparison in a decision. The module is not one of
the lint-scoped simulation modules, so it uses host numpy and subscript assignment freely.
"""

from __future__ import annotations

import math
from enum import IntEnum
from typing import Final, NamedTuple

import numpy as np

TRACK_VERSION: Final = 1

_ID_LIMIT: Final = 2**31  # ids stay below this so that the pair code ``p << 32 | l`` fits int64
_EMA_SCALE: Final = 8  # Dominance counts enter the EMA as ``count << 8`` (G1's accumulator scale)
_Q16: Final = 65536


# --- types ------------------------------------------------------------------------------------------------


class EventKind(IntEnum):
    BORN = 1
    SPLIT = 2
    MERGED = 3
    DISSOLVED = 4


class Event(NamedTuple):
    """What happened to a system at a snapshot.

    ``system`` is the id the event is about (the new id for BORN and SPLIT, the ended id for MERGED and
    DISSOLVED), ``other`` the related id (provenance of a new system, the survivor of a merge, the larger
    successor of a dissolution; -1 if none), ``size`` the member count (current for new systems, previous for
    ended ones) and ``overlap`` the shared member count with ``other`` (0 if none).
    """

    step: int
    kind: EventKind
    system: int
    size: int
    other: int
    overlap: int


class Members(NamedTuple):
    """Candidate systems of a snapshot: ``(label[i], key[i])`` means key ``key[i]`` is a member of the
    candidate ``label[i]``. Equal-length int64 vectors; the pairs must be unique (a key may be in several
    candidates)."""

    label: np.ndarray
    key: np.ndarray


class Assignment(NamedTuple):
    """The candidates of a snapshot, sorted by label: the id each holds (-1 for none) and its member count."""

    label: np.ndarray
    system: np.ndarray
    size: np.ndarray


class TrackerState(NamedTuple):
    """The last snapshot's memberships of the id-holding systems, sorted by ``(key, system)``; ``next_id``
    starts at 1 and an id is never reused; ``step`` is -1 before the first snapshot."""

    system: np.ndarray
    key: np.ndarray
    next_id: int
    step: int


class Diagnostics(NamedTuple):
    """Counts and Jaccard values of one ``observe`` (for the calibration tool only; no decision reads them).

    ``matched_jaccard`` has one value per matched pair, by previous id; ``unmatched_best_jaccard`` one per
    unmatched previous system, by id: its best Jaccard with any current candidate (0.0 without overlap).
    """

    n_prev: int
    n_cur: int
    n_admitted: int
    n_matched: int
    matched_jaccard: np.ndarray
    unmatched_best_jaccard: np.ndarray


# --- array helpers ----------------------------------------------------------------------------------------


def _vec(a: object, name: str) -> np.ndarray:
    v = np.asarray(a)
    if v.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array, got shape {v.shape}")
    if v.size and not np.issubdtype(v.dtype, np.integer):
        raise TypeError(f"{name} must be an integer array, got {v.dtype}")
    return v.astype(np.int64, copy=False)


def _empty() -> np.ndarray:
    return np.empty(0, dtype=np.int64)


def _first_of_run(sorted_group: np.ndarray) -> np.ndarray:
    """Bool mask of the first element of each run of equal values in a sorted vector."""
    first = np.ones(sorted_group.size, dtype=bool)
    first[1:] = sorted_group[1:] != sorted_group[:-1]
    return first


def _join(sorted_keys: np.ndarray, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Many-to-many equi-join: ``(qi, si)`` of every pair with ``query[qi] == sorted_keys[si]``.

    ``sorted_keys`` ascending (duplicates allowed); the pairs are ordered by ``qi``, then ``si``.
    """
    lo = np.searchsorted(sorted_keys, query, side="left")
    hi = np.searchsorted(sorted_keys, query, side="right")
    cnt = hi - lo
    total = int(cnt.sum())
    qi = np.repeat(np.arange(query.size, dtype=np.int64), cnt)
    start = np.cumsum(cnt) - cnt
    si = np.repeat(lo, cnt) + (np.arange(total, dtype=np.int64) - np.repeat(start, cnt))
    return qi, si


def _check_state(state: TrackerState) -> TrackerState:
    system, key = _vec(state.system, "state.system"), _vec(state.key, "state.key")
    next_id, step = int(state.next_id), int(state.step)
    if system.size != key.size:
        raise ValueError("state.system and state.key differ in length")
    if not 1 <= next_id < _ID_LIMIT:
        raise ValueError(f"state.next_id {next_id} must be in [1, 2^31)")
    if step < -1:
        raise ValueError(f"state.step {step} must be >= -1")
    if system.size:
        if system.min() < 1 or system.max() >= next_id:
            raise ValueError("state.system holds an id outside [1, next_id)")
        later = (key[1:] > key[:-1]) | ((key[1:] == key[:-1]) & (system[1:] > system[:-1]))
        if not later.all():
            raise ValueError("state memberships must be unique and sorted by (key, system)")
    if step < 0 and (system.size or next_id != 1):
        raise ValueError("a state before the first snapshot has no memberships and next_id 1")
    return TrackerState(system.copy(), key.copy(), next_id, step)


# --- the tracker ------------------------------------------------------------------------------------------


class SystemTracker:
    """Persistent system ids by one-to-one greedy overlap matching, with admission hysteresis.

    ``theta = (num, den)``: a candidate pair matches iff its Jaccard index is at least ``num / den``, tested
    in integers. The tracker keeps the last snapshot's memberships of the systems that hold an id
    (``state()``) and nothing else; construct it from a saved state to continue.
    """

    def __init__(self, theta: tuple[int, int], state: TrackerState | None = None) -> None:
        num, den = int(theta[0]), int(theta[1])
        if not 0 < num <= den:
            raise ValueError(f"theta {theta} must satisfy 0 < num <= den")
        self._theta = (num, den)
        self._state = TrackerState(_empty(), _empty(), 1, -1) if state is None else _check_state(state)

    def state(self) -> TrackerState:
        s = self._state
        return TrackerState(s.system.copy(), s.key.copy(), s.next_id, s.step)

    def observe(
        self,
        step: int,
        cur: Members,
        credit: tuple[np.ndarray, np.ndarray] | None = None,
        admit: np.ndarray | None = None,
    ) -> tuple[Assignment, list[Event], Diagnostics]:
        """Match the candidates ``cur`` of ``step`` against the tracked systems and update the state.

        ``cur`` holds the candidates: every current system that may continue a tracked one (callers pass the
        systems at or above a *leave* threshold). ``admit`` (int64 labels, a subset of ``cur``'s) are the
        candidates that may found a new system (at or above an *enter* threshold); ``None`` admits every
        candidate. With leave < enter this is hysteresis: a system gets an id only once it reaches the enter
        threshold and keeps it while it stays a candidate. ``credit = (child, parent)``: keys new since the
        last snapshot inherit the previous memberships of their parent key.
        """
        label, key = _vec(cur.label, "cur.label"), _vec(cur.key, "cur.key")
        if label.size != key.size:
            raise ValueError("cur.label and cur.key differ in length")
        st = self._state
        labs, inv, bsize = np.unique(label, return_inverse=True, return_counts=True)
        inv, bsize = inv.reshape(-1).astype(np.int64), bsize.astype(np.int64)
        admitted = np.ones(labs.size, dtype=bool) if admit is None else np.isin(labs, _vec(admit, "admit"))
        if st.step < 0:
            sysof, next_id, events, diag = self._first(labs, bsize, admitted)
        else:
            sysof, next_id, events, diag = self._match(step, labs, inv, bsize, admitted, key, credit)
        if next_id >= _ID_LIMIT:
            raise OverflowError("tracker ids exhausted (2^31)")
        held = sysof[inv] >= 0
        new_sys, new_key = sysof[inv][held], key[held]
        order = np.lexsort((new_sys, new_key))
        self._state = TrackerState(new_sys[order], new_key[order], next_id, int(step))
        return Assignment(labs, sysof, bsize), events, diag

    # -- first snapshot ------------------------------------------------------------------------------------

    def _first(
        self, labs: np.ndarray, bsize: np.ndarray, admitted: np.ndarray
    ) -> tuple[np.ndarray, int, list[Event], Diagnostics]:
        """Step 1: ids to the admitted labels in the order (-size, label); no events."""
        sysof = np.full(labs.size, -1, dtype=np.int64)
        adm = np.flatnonzero(admitted)
        adm = adm[np.lexsort((labs[adm], -bsize[adm]))]
        next_id = self._state.next_id
        sysof[adm] = next_id + np.arange(adm.size, dtype=np.int64)
        diag = Diagnostics(0, labs.size, adm.size, 0, np.empty(0), np.empty(0))
        return sysof, next_id + adm.size, [], diag

    # -- later snapshots -----------------------------------------------------------------------------------

    def _match(
        self,
        step: int,
        labs: np.ndarray,
        inv: np.ndarray,
        bsize: np.ndarray,
        admitted: np.ndarray,
        key: np.ndarray,
        credit: tuple[np.ndarray, np.ndarray] | None,
    ) -> tuple[np.ndarray, int, list[Event], Diagnostics]:
        thn, thd = self._theta
        st = self._state
        n_cur = labs.size
        step = int(step)

        # 2. Credit: A' = the previous memberships plus (system(parent), child) per credited pair.
        asys, akey = st.system, st.key
        if credit is not None:
            child, parent = _vec(credit[0], "credit child"), _vec(credit[1], "credit parent")
            if child.size != parent.size:
                raise ValueError("credit child and parent differ in length")
            qi, si = _join(st.key, parent)
            if qi.size:
                asys = np.concatenate([st.system, st.system[si]])
                akey = np.concatenate([st.key, child[qi]])
                o = np.lexsort((asys, akey))
                asys, akey = asys[o], akey[o]
                keep = np.ones(asys.size, dtype=bool)
                keep[1:] = (akey[1:] != akey[:-1]) | (asys[1:] != asys[:-1])
                asys, akey = asys[keep], akey[keep]
        ids, asize = np.unique(asys, return_counts=True)  # the previous systems and |A'|
        asize = asize.astype(np.int64)
        n_prev = ids.size
        usize = np.bincount(np.searchsorted(ids, st.system), minlength=n_prev).astype(np.int64)  # |A|

        # 3. Overlap c(p, l) over every candidate: a many-to-many join on key, counted per (p, l) pair.
        qi, ai = _join(akey, key)
        code = (asys[ai] << 32) | inv[qi]
        ucode, c = np.unique(code, return_counts=True)
        c = c.astype(np.int64)
        p, lr = ucode >> 32, ucode & 0xFFFFFFFF
        pi = np.searchsorted(ids, p)
        union = asize[pi] + bsize[lr] - c

        # 4-5. Accept at Jaccard >= theta (integers), then greedy one-to-one by (-c, p, l).
        ok = thd * c >= thn * union
        order = np.lexsort((lr, p, -c))
        cand = order[ok[order]]
        used_p = np.zeros(n_prev, dtype=bool)
        match_l = np.full(n_cur, -1, dtype=np.int64)  # the previous id a current label continues
        taken: list[int] = []
        for k, a, b in zip(cand.tolist(), pi[cand].tolist(), lr[cand].tolist(), strict=True):
            if used_p[a] or match_l[b] >= 0:
                continue
            used_p[a] = True
            match_l[b] = ids[a]
            taken.append(k)

        # 6. Unmatched admitted labels found new systems, in the order (-size, label).
        best_p = np.full(n_cur, -1, dtype=np.int64)
        best_c = np.zeros(n_cur, dtype=np.int64)
        by_label = np.lexsort((p, -c, lr))  # per label: the largest overlap, ties to the smaller p
        first = by_label[_first_of_run(lr[by_label])]
        best_p[lr[first]], best_c[lr[first]] = p[first], c[first]
        new = np.flatnonzero((match_l < 0) & admitted)
        new = new[np.lexsort((labs[new], -bsize[new]))]
        sysof = match_l.copy()
        next_id = st.next_id
        new_events: list[Event] = []
        for lab in new.tolist():
            sysof[lab] = next_id
            bp, bc = int(best_p[lab]), int(best_c[lab])
            if bc > 0:
                split = bool(used_p[np.searchsorted(ids, bp)])
                kind, other, shared = (EventKind.SPLIT if split else EventKind.BORN), bp, bc
            else:
                kind, other, shared = EventKind.BORN, -1, 0
            new_events.append(Event(step, kind, next_id, int(bsize[lab]), other, shared))
            next_id += 1

        # 7. Unmatched previous systems end: MERGED into a continuing system, else DISSOLVED. Only labels that
        # hold an id count, the best overlap wins, ties to the smaller label.
        ended_events: list[Event] = []
        rows = np.flatnonzero(~used_p[pi] & (sysof[lr] >= 0))
        rows = rows[np.lexsort((lr[rows], -c[rows], p[rows]))]
        best = rows[_first_of_run(p[rows])]
        star = {int(p[k]): (int(lr[k]), int(c[k])) for k in best.tolist()}
        for a in np.flatnonzero(~used_p).tolist():
            pid = int(ids[a])
            lstar, shared = star.get(pid, (-1, 0))
            merged = shared > 0 and match_l[lstar] >= 0
            kind = EventKind.MERGED if merged else EventKind.DISSOLVED
            other = int(sysof[lstar]) if shared > 0 else -1
            ended_events.append(Event(step, kind, pid, int(usize[a]), other, shared))

        # Diagnostics: the Jaccard of each matched pair, the best Jaccard of each unmatched previous system.
        jac = c / union
        matched_j = jac[np.array(sorted(taken, key=lambda k: p[k]), dtype=np.int64)]
        best_j = np.zeros(n_prev)
        np.maximum.at(best_j, pi, jac)
        diag = Diagnostics(n_prev, n_cur, int(admitted.sum()), len(taken), matched_j, best_j[~used_p])
        return sysof, next_id, ended_events + new_events, diag


# --- grid clusters ----------------------------------------------------------------------------------------


def _face_neighbours(
    coords: np.ndarray, shape: tuple[int, ...], periodic: tuple[bool, ...]
) -> list[tuple[np.ndarray, np.ndarray]]:
    """For each of the 2D faces: (code of the neighbour, whether it exists) per row of ``coords``."""
    out = []
    for ax in range(len(shape)):
        for s in (-1, 1):
            c = coords.copy()
            c[:, ax] += s
            if periodic[ax]:
                c[:, ax] %= shape[ax]
                ok = np.ones(len(c), dtype=bool)
            else:
                ok = (c[:, ax] >= 0) & (c[:, ax] < shape[ax])
                c[:, ax] = np.clip(c[:, ax], 0, shape[ax] - 1)
            out.append((np.ravel_multi_index(tuple(c.T), shape).astype(np.int64), ok))
    return out


def grid_clusters(
    counts: np.ndarray, shape: tuple[int, ...], occ_min: int, periodic: tuple[bool, ...] | None = None
) -> np.ndarray:
    """Clusters of occupied bins on an n-d grid: ``(prod(shape),)`` int64, the root code of each bin or -1.

    A bin is *occupied* iff ``counts >= occ_min`` (``occ_min >= 1``). Occupied bins are joined by face
    adjacency (+-1 along one axis; an axis wraps only if ``periodic``); labels are propagated as the minimum
    over the sparse occupied set with one pointer jump per round until a fixed point, so the label of a
    component is its smallest code. A non-empty bin below ``occ_min`` is *attached* to the label of its most
    populous occupied face neighbour (ties to the smaller code), else -1; attached bins never join components.
    All -1 when no bin is occupied. ``counts`` is flat or of shape ``shape``; codes are C order (last axis
    fastest).
    """
    shape = tuple(int(s) for s in shape)
    per = (False,) * len(shape) if periodic is None else tuple(bool(b) for b in periodic)
    if len(per) != len(shape) or not shape or min(shape) < 1:
        raise ValueError(f"shape {shape} and periodic {per} do not describe a grid")
    if occ_min < 1:
        raise ValueError(f"occ_min {occ_min} must be >= 1")
    cnt = np.asarray(counts).reshape(-1)
    if cnt.size != math.prod(shape):
        raise ValueError(f"counts has {cnt.size} bins, shape {shape} has {math.prod(shape)}")
    out = np.full(cnt.size, -1, dtype=np.int64)
    occ = cnt >= occ_min
    codes = np.flatnonzero(occ)
    n = codes.size
    if n == 0:
        return out
    coords = np.stack(np.unravel_index(codes, shape), axis=1)
    cols = []
    for code, ok in _face_neighbours(coords, shape, per):
        j = np.searchsorted(codes, code)
        hit = ok & (j < n) & (codes[np.minimum(j, n - 1)] == code)
        cols.append(np.where(hit, j, -1))
    nb = np.stack(cols, axis=1)
    lab = np.arange(n, dtype=np.int64)
    while True:
        low = np.where(nb >= 0, lab[np.maximum(nb, 0)], n).min(axis=1)
        new = np.minimum(lab, low)
        new = new[new]  # one pointer jump
        if np.array_equal(new, lab):
            break
        lab = new
    out[codes] = codes[lab]
    sparse = np.flatnonzero((cnt > 0) & ~occ)
    if sparse.size:
        sc = np.stack(np.unravel_index(sparse, shape), axis=1)
        best_cnt = np.full(sparse.size, -1, dtype=np.int64)
        best_code = np.full(sparse.size, -1, dtype=np.int64)
        for code, ok in _face_neighbours(sc, shape, per):
            cc = np.where(ok & occ[code], cnt[code], -1).astype(np.int64)
            better = (cc > best_cnt) | ((cc == best_cnt) & (cc >= 0) & (code < best_code))
            best_cnt, best_code = np.where(better, cc, best_cnt), np.where(better, code, best_code)
        out[sparse] = np.where(best_cnt >= 0, out[np.maximum(best_code, 0)], -1)
    return out


# --- dominance --------------------------------------------------------------------------------------------


class DominanceMap(NamedTuple):
    """Per cell (code ``y * nx + x``): the owner id (-1 none), the owner's share of the smoothed total in Q16
    (0 if unowned) and the borders to the east (``[:, 0]``) and north (``[:, 1]``) neighbours."""

    owner: np.ndarray
    purity_q16: np.ndarray
    border: np.ndarray


class DominanceState(NamedTuple):
    """The smoothed counts: ``ids`` (K,) the tracked ids (sorted as given) and ``ema`` (C, K + 1) int64, the
    last column being "other"; ``ema`` is None before the first update."""

    ids: np.ndarray
    ema: np.ndarray | None


class Dominance:
    """Owner, purity and borders on a 2-D grid of ``C = nx * ny`` cells: ``shape = (nx, ny)``, ``y*nx+x``.

    Each update takes the per-cell counts of the current ids plus a last column "other" and smooths them with
    the exponential average of G1's accumulator, per column, in int64: ``x = count << 8``; a column whose id
    was not tracked at the previous update starts at ``E = x << k``, else ``E += x - (E >> k)``; ids no longer
    present are dropped; "other" is always tracked (and the first update starts every column at ``x << k``).
    A cell is owned by the id of the largest column of ``E`` (ties to the lower column) unless that is
    "other" or the smoothed total is below ``min_count << 8 << k`` (or zero). ``periodic = (px, py)`` says
    which axes wrap; on a non-periodic axis the last column (row) has no east (north) border.
    """

    def __init__(
        self,
        shape: tuple[int, int],
        k: int,
        min_count: int,
        periodic: tuple[bool, bool] = (True, True),
        state: DominanceState | None = None,
    ) -> None:
        nx, ny = int(shape[0]), int(shape[1])
        if nx < 1 or ny < 1:
            raise ValueError(f"shape {shape} must be positive")
        if not 0 <= int(k) <= 20:
            raise ValueError(f"k {k} must be in [0, 20]")
        if min_count < 0:
            raise ValueError(f"min_count {min_count} must be >= 0")
        self._nx, self._ny = nx, ny
        self._k, self._min = int(k), int(min_count)
        self._periodic = (bool(periodic[0]), bool(periodic[1]))
        self._ids = _empty()
        self._ema: np.ndarray | None = None
        if state is not None:
            ids = _vec(state.ids, "state.ids")
            ema = None if state.ema is None else np.asarray(state.ema, dtype=np.int64)
            if ema is not None and ema.shape != (nx * ny, ids.size + 1):
                raise ValueError(f"state.ema must have shape {(nx * ny, ids.size + 1)}, got {ema.shape}")
            if ema is None and ids.size:
                raise ValueError("state.ids without state.ema")
            self._ids, self._ema = ids.copy(), None if ema is None else ema.copy()

    def state(self) -> DominanceState:
        return DominanceState(self._ids.copy(), None if self._ema is None else self._ema.copy())

    def update(self, ids: np.ndarray, counts: np.ndarray) -> DominanceMap:
        ids = _vec(ids, "ids")
        cells = self._nx * self._ny
        x = np.asarray(counts, dtype=np.int64)
        if x.shape != (cells, ids.size + 1):
            raise ValueError(f"counts must have shape {(cells, ids.size + 1)}, got {x.shape}")
        if np.unique(ids).size != ids.size:
            raise ValueError("ids must be unique")
        x = x << _EMA_SCALE
        known = np.zeros(ids.size + 1, dtype=bool)
        prior = np.zeros_like(x)
        if self._ema is not None:  # the ids tracked at the previous update continue their average
            old = self._ids
            order = np.argsort(old, kind="stable")
            pos = np.minimum(np.searchsorted(old[order], ids), max(old.size - 1, 0))
            cols = order[pos] if old.size else np.zeros(ids.size, dtype=np.int64)
            known[: ids.size] = (old[order][pos] == ids) if old.size else False
            known[ids.size] = True  # "other" is always tracked
            prior[:, : ids.size] = self._ema[:, cols] if old.size else 0
            prior[:, ids.size] = self._ema[:, -1]
        # a column that was not tracked starts at its steady state ``x << k``
        e = np.where(known[None, :], prior + x - (prior >> self._k), x << self._k)
        self._ids, self._ema = ids.copy(), e
        return self._map(ids, e)

    def _map(self, ids: np.ndarray, e: np.ndarray) -> DominanceMap:
        total = e.sum(axis=1)
        arg = e.argmax(axis=1)  # numpy takes the first maximum: ties to the lower column
        owned = (arg < ids.size) & (total >= (self._min << _EMA_SCALE << self._k)) & (total > 0)
        owner = np.where(owned, np.append(ids, -1)[arg], -1).astype(np.int64)
        top = e[np.arange(e.shape[0]), arg]
        purity = np.where(owned, top * _Q16 // np.maximum(total, 1), 0).astype(np.int64)
        grid = owner.reshape(self._ny, self._nx)
        border = np.zeros((self._ny, self._nx, 2), dtype=bool)
        for axis, wrap in ((1, self._periodic[0]), (0, self._periodic[1])):
            nxt = np.roll(grid, -1, axis=axis)
            diff = (grid != nxt) & ((grid >= 0) | (nxt >= 0))
            if not wrap:
                diff[(slice(None), -1) if axis == 1 else (-1, slice(None))] = False
            border[..., 0 if axis == 1 else 1] = diff
        return DominanceMap(owner, purity, border.reshape(-1, 2))


# --- helpers ----------------------------------------------------------------------------------------------


def lookup_table(size: int, keys: np.ndarray, values: np.ndarray | int, fill: int) -> np.ndarray:
    """(size,) int32 table: ``fill`` everywhere, ``values`` at ``keys`` (later duplicates win)."""
    k = _vec(keys, "keys")
    if k.size and (k.min() < 0 or k.max() >= size):
        raise ValueError(f"keys must lie in [0, {size})")
    out = np.full(size, fill, dtype=np.int32)
    out[k] = np.asarray(values).astype(np.int32)
    return out


def rank_by_size(ids: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    """The ids ordered by (-size, id)."""
    i, s = _vec(ids, "ids"), _vec(sizes, "sizes")
    if i.size != s.size:
        raise ValueError("ids and sizes differ in length")
    return i[np.lexsort((i, -s))]


def circular_mean(x: np.ndarray, size: float) -> float:
    """Mean of positions on a circle of circumference ``size``, in [0, size) (float64)."""
    v = np.asarray(x, dtype=np.float64).reshape(-1)
    if v.size == 0:
        raise ValueError("circular_mean of an empty array")
    ang = v * (2.0 * math.pi / size)
    mean = math.atan2(float(np.sin(ang).sum()), float(np.cos(ang).sum()))
    return float((mean % (2.0 * math.pi)) * size / (2.0 * math.pi) % size)


def sticky_slots(prev: np.ndarray, ranked: np.ndarray, slack: int) -> np.ndarray:
    """Slots that follow their id: ``prev`` the (F,) int64 slot -> id map of the last snapshot (-1 empty),
    ``ranked`` the tracked ids by rank (best first).

    A slotted id keeps its slot iff it is in ``ranked[:F + slack]``; the free slots, in ascending order, take
    the unslotted ids of ``ranked`` in rank order; -1 where none is left. At most ``slack`` slotted ids rank
    outside the top F.
    """
    p, r = _vec(prev, "prev"), _vec(ranked, "ranked")
    f = p.size
    keep = (p >= 0) & np.isin(p, r[: f + slack])
    out = np.where(keep, p, -1)
    free = np.flatnonzero(~keep)
    waiting = r[~np.isin(r, out[keep])]
    n = min(free.size, waiting.size)
    out[free[:n]] = waiting[:n]
    return out


def deadband(ref: int, value: int, shift: int) -> tuple[int, bool]:
    """A reference that follows ``value`` only when it has moved by more than ``ref >> shift``: ``(value,
    True)`` if ``ref < 0`` (no reference yet) or ``|value - ref| > ref >> shift``, else ``(ref, False)``."""
    if ref < 0 or abs(value - ref) > ref >> shift:
        return int(value), True
    return int(ref), False
