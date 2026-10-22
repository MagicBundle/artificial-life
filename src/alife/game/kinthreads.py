"""Kin threads: the kin groups of a played world seen through id changes (G3b spec §2.2, DESIGN §34.1).

G2 gives a kin group that ends and comes back a new id (DESIGN §29.14), and on the Mac-day play 49% of the
groups that formed and ended inside the window lived two snapshots (500 steps) or less, so a chronicle that
told every ``events`` row of ``social.jsonl`` said "splits" and "merges" of the same cluster dozens of
times. A **thread** is a kin group through those id changes: the group's *tag centroid* (``kin_groups.tag``)
is what the tag-space clustering finds again, so a new id whose tag is within ``DELTA`` of a thread that was
absent a few snapshots is that thread coming back (a **flap**, told as nothing). The chronicle narrates
threads, not ids.

``KinThreads.fold(record)`` is called once per social record, in order, and is a pure function of the record
sequence: chunking, a stop and a resume cannot change what is told (the state round-trips through JSON,
``state``/``from_state``, and a refold of ``social.jsonl`` gives the same table). The report job of the live
play (spec §5.1) and N1's ``_event_facts`` use this same fold.

The rule, with the constants below (``KINTHREADS_VERSION`` 1, every one provisional until the G3b Mac day):

1. A listed id already owned by a thread refreshes it.
2. A newly listed id (not in the first record of a fold) looks for an *absent* thread (no id listed now, not
   matched by another id of this record, ``1 <= k - last <= R + 1``). The assignment is global: all (new id,
   absent thread) pairs with tag distance ``< DELTA``, sorted by (distance, thread id, kin id), taken
   greedily. A taken pair is a flap and tells nothing; an id left over starts a *candidate*.
3. A thread's size is the sum of its listed ids' sizes, its tag their size-weighted mean, its peak the
   largest size seen. Its name is the name of the first id listed in it, frozen; two established threads
   that overlap in time (or end less than ``NAME_GAP`` snapshots apart) and share a name are shown with the
   lowest unused suffix ("Daudrall II"), frozen at establishment.
4. A candidate is **established** at the first snapshot at which it is listed and ``k - first + 1 >= P``;
   ``formed`` is told then.
5. A thread not listed for ``R + 1`` consecutive snapshots **ends**: told if it was established, silent if
   not. It ended *merged into X* iff the record after its last listing carries a ``merged`` row for one of
   its last-listed ids (X is the thread owning the row's ``other``); otherwise it dissolved.
6. The threads of the fold's first record are the initial cast: established and silent.
7. A record whose ``origin`` differs from the previous one (an observer restart renumbers the ids) closes
   the fold: every thread ends silently and the record's groups are a new initial cast.

Host code: numpy only, no mlx, no subscript assignment (``tests/test_lint.py``).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np

__all__ = [
    "DELTA",
    "KINTHREADS_FORMAT",
    "KINTHREADS_VERSION",
    "MAX_THREADS",
    "NAME_GAP",
    "TOAST_FACTOR",
    "TOAST_LIFE",
    "KinThreads",
    "P",
    "R",
    "Thread",
    "constants_digest",
    "kin_untold",
    "roman",
]

KINTHREADS_VERSION: Final = 1
KINTHREADS_FORMAT: Final = "alife-kinthreads"  # the ``format`` of ``observers/kinthreads_<t>.json``
P: Final = 3  # snapshots to establish
R: Final = 3  # snapshots of absence tolerated
DELTA: Final = 0.15  # tag distance (1.2 kin-grid bins of 1/8)
TOAST_FACTOR: Final = 4  # TOAST_SIZE = TOAST_FACTOR * kin_enter of the record's thresholds
TOAST_SIZE_DEFAULT: Final = 350  # a record without thresholds (hand-built ones)
TOAST_LIFE: Final = 8  # snapshots a thread lived to raise an importance-2 ``ended``
NAME_GAP: Final = 8  # snapshots after which an ended thread's name is free again
MAX_THREADS: Final = 64  # threads kept in the table
KIN_FAMILY: Final = "kin_group"

_BORN = ("born", "split")
_EXIT = ("merged", "dissolved")


def constants_digest() -> str:
    """SHA-256 of the version and every constant of the rule (pinned by a test: a change of a constant is a
    deliberate one and needs a ``KINTHREADS_VERSION`` bump)."""
    blob = json.dumps(
        [
            KINTHREADS_VERSION,
            P,
            R,
            DELTA,
            TOAST_FACTOR,
            TOAST_SIZE_DEFAULT,
            TOAST_LIFE,
            NAME_GAP,
            MAX_THREADS,
        ],
        separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


_ROMAN: Final = (
    (1000, "M"),
    (900, "CM"),
    (500, "D"),
    (400, "CD"),
    (100, "C"),
    (90, "XC"),
    (50, "L"),
    (40, "XL"),
    (10, "X"),
    (9, "IX"),
    (5, "V"),
    (4, "IV"),
    (1, "I"),
)


def roman(n: int) -> str:
    """The Roman numeral of ``1 <= n < 4000``."""
    if not 1 <= n < 4000:
        raise ValueError(f"no Roman numeral for {n}")
    out = ""
    for value, sym in _ROMAN:
        while n >= value:
            out += sym
            n -= value
    return out


@dataclass
class Thread:
    """One thread. ``name`` is the frozen base name, ``suffix`` 1 (bare) or the number of the suffix taken at
    establishment (``shown`` is the name as displayed)."""

    id: int
    name: str
    first: int  # snapshot index of the first listing
    first_step: int
    last: int  # snapshot index of the last listing
    last_step: int
    first_id: int
    tag: list[float]
    size: int = 0
    peak: int = 0
    ids: list[int] = field(default_factory=list)  # the kin ids listed at the last listing, largest first
    pos: list[float | None] = field(default_factory=lambda: [None, None, None])
    est: bool = False
    init: bool = False
    ended: bool = False
    end_k: int = -1
    est_k: int = -1
    suffix: int = 1
    parent: int | None = None  # thread id of the group this one split from
    birth_row: bool = False  # the first listing carried a born/split row for the first id
    gone_step: int | None = None  # step of the record after the last listing, once seen
    exit_kind: str | None = None  # "merged" | "dissolved" row of the record after the last listing
    exit_into: int | None = None  # thread id the merged row names
    exit_into_name: str | None = None

    @property
    def shown(self) -> str:
        return self.name if self.suffix == 1 else f"{self.name} {roman(self.suffix)}"

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id, "name": self.name, "first": self.first, "first_step": self.first_step,
            "last": self.last, "last_step": self.last_step, "first_id": self.first_id,
            "tag": list(self.tag), "size": self.size, "peak": self.peak, "ids": list(self.ids),
            "pos": list(self.pos), "est": self.est, "init": self.init, "ended": self.ended,
            "end_k": self.end_k, "est_k": self.est_k, "suffix": self.suffix, "parent": self.parent,
            "birth_row": self.birth_row, "gone_step": self.gone_step, "exit_kind": self.exit_kind,
            "exit_into": self.exit_into, "exit_into_name": self.exit_into_name,
        }  # fmt: skip

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> Thread:
        return cls(
            id=int(d["id"]), name=str(d["name"]), first=int(d["first"]), first_step=int(d["first_step"]),
            last=int(d["last"]), last_step=int(d["last_step"]), first_id=int(d["first_id"]),
            tag=[float(v) for v in d["tag"]], size=int(d["size"]), peak=int(d["peak"]),
            ids=[int(v) for v in d["ids"]], pos=[None if v is None else float(v) for v in d["pos"]],
            est=bool(d["est"]), init=bool(d["init"]), ended=bool(d["ended"]), end_k=int(d["end_k"]),
            est_k=int(d["est_k"]), suffix=int(d["suffix"]),
            parent=None if d["parent"] is None else int(d["parent"]),
            birth_row=bool(d["birth_row"]),
            gone_step=None if d["gone_step"] is None else int(d["gone_step"]),
            exit_kind=d["exit_kind"],
            exit_into=None if d["exit_into"] is None else int(d["exit_into"]),
            exit_into_name=d["exit_into_name"],
        )  # fmt: skip


def _pad3(v: Sequence[Any]) -> list[float | None]:
    return [*[None if x is None else float(x) for x in v], None, None, None][:3]


class KinThreads:
    """The thread table and the fold over social records (module docstring)."""

    def __init__(self) -> None:
        self.threads: dict[int, Thread] = {}
        self.owner: dict[int, int] = {}  # kin id -> thread id, for the ids of threads still open
        self.k = -1  # snapshot index of the last record folded (records of every origin count)
        self.step = -1  # its step
        self.origin: int | None = None
        self.next_id = 1
        self.flaps = 0  # re-formations bridged (a diagnostic, kept in the state)
        self.first_record = True  # the next record starts a fold (the initial cast)

    # --- the fold ---------------------------------------------------------------------------------------

    def fold(self, rec: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Fold the next social record; returns the events it tells, in order (formed by thread id, then
        ended by thread id). An event is a dict: ``kind`` ("formed" or "ended"), ``step`` (the record's),
        ``thread``, ``name`` (as shown), ``a`` (the thread's current kin id), ``b`` (the kin id of the
        parent or of the thread it merged into, 0 if none), ``pos``, ``size``, ``peak``, ``first_step``,
        ``end_step``, ``steps`` (the lifetime in steps, None for an initial thread), ``life`` (snapshots),
        ``parent``/``into`` (the other thread as shown, or None), ``initial``, ``imp`` and ``consumed`` (the
        step of the record whose raw kin row the event consumes, None if it consumes none)."""
        step = int(rec["step"])
        origin = int(rec.get("origin", 0))
        if self.origin is not None and origin != self.origin:
            self._close_epoch()
        self.origin = origin
        self.k += 1
        k, first = self.k, self.first_record
        self.first_record = False
        self.step = step
        kin = rec["kin_groups"]
        ids = [int(i) for i in kin["id"]]
        listed: dict[int, dict[str, Any]] = {}
        for j, gid in enumerate(ids):
            z = kin["z_mean"][j] if "z_mean" in kin else None
            listed.update(
                {
                    gid: {
                        "size": int(kin["size"][j]),
                        "tag": np.asarray(kin["tag"][j], np.float64),
                        "name": str(kin["name"][j]),
                        "pos": [kin["x"][j], kin["y"][j], z],
                    }
                }
            )
        present = {self.owner[i] for i in ids if i in self.owner}
        new = sorted(i for i in ids if i not in self.owner)
        matched: dict[int, int] = {}
        if not first:
            matched = self._bridge(new, listed, present, k)
        for i in new:
            if i in matched:
                self.owner.update({i: matched[i]})  # joined: the listing below rebuilds ``ids``
                self.flaps += 1
            else:
                self._start(i, listed[i], k, step, init=first)
        rows = self._rows(rec, step)
        # the ids' owners are now complete: parents and exit rows
        self._born_rows(rows, k, step)
        present = {self.owner[i] for i in ids}
        self._exit_rows(rows, present, k, step)
        toast = self._toast_size(rec)
        told: list[dict[str, Any]] = []
        for tid in sorted(present):
            th = self.threads[tid]
            mine = sorted((i for i in ids if self.owner[i] == tid), key=lambda i: (-listed[i]["size"], i))
            sizes = np.array([listed[i]["size"] for i in mine], np.float64)
            th.size = int(sizes.sum())
            th.peak = max(th.peak, th.size)
            th.last, th.last_step = k, step
            th.gone_step, th.exit_kind, th.exit_into, th.exit_into_name = None, None, None, None
            if sizes.sum() > 0:
                tags = np.stack([listed[i]["tag"] for i in mine])
                th.tag = [float(v) for v in (tags * sizes[:, None]).sum(axis=0) / sizes.sum()]
            else:
                th.tag = [float(v) for v in listed[mine[0]]["tag"]]
            th.ids = mine
            th.pos = _pad3(listed[mine[0]]["pos"])
            if not th.est and k - th.first + 1 >= P:
                self._establish(th, k)  # an initial thread is established from the start: never here
                told.append(self._formed_event(th, step, toast))
        for tid in sorted(t for t, th in self.threads.items() if not th.ended and t not in present):
            th = self.threads[tid]
            if k - th.last >= R + 1:
                th.ended, th.end_k = True, k
                if th.est:
                    told.append(self._ended_event(th, step, toast))
        self._prune(k)
        return told

    # --- steps of the fold ------------------------------------------------------------------------------

    def _bridge(
        self, new: Sequence[int], listed: Mapping[int, Mapping[str, Any]], present: set[int], k: int
    ) -> dict[int, int]:
        """The global assignment of new ids to absent threads (rule 2): ``{kin id: thread id}``."""
        absent = [
            th
            for th in self.threads.values()
            if not th.ended and th.id not in present and 1 <= k - th.last <= R + 1
        ]
        pairs: list[tuple[float, int, int]] = []
        for i in new:
            for th in absent:
                d = float(np.linalg.norm(np.asarray(th.tag, np.float64) - listed[i]["tag"]))
                if d < DELTA:
                    pairs.append((d, th.id, i))
        pairs.sort()
        got: dict[int, int] = {}
        taken: set[int] = set()
        for _, tid, i in pairs:
            if tid in taken or i in got:
                continue
            taken.add(tid)
            got.update({i: tid})
        return got

    def _start(self, gid: int, info: Mapping[str, Any], k: int, step: int, *, init: bool) -> None:
        th = Thread(
            id=self.next_id,
            name=info["name"],
            first=k,
            first_step=step,
            last=k,
            last_step=step,
            first_id=gid,
            tag=[float(v) for v in info["tag"]],
            ids=[gid],
            init=init,
        )
        self.next_id += 1
        self.threads.update({th.id: th})
        self.owner.update({gid: th.id})
        if init:
            th.est, th.est_k = True, k
            self._name(th, k)

    def _rows(self, rec: Mapping[str, Any], step: int) -> list[tuple[str, int, int]]:
        """The raw kin rows of the record: ``(kind, id, other)``."""
        ev = rec.get("events")
        if not ev:
            return []
        return [
            (str(kind), int(gid), int(other))
            for fam, kind, gid, other in zip(ev["family"], ev["kind"], ev["id"], ev["other"], strict=True)
            if fam == KIN_FAMILY
        ]

    def _born_rows(self, rows: Iterable[tuple[str, int, int]], k: int, step: int) -> None:
        """The born/split row of a thread's first listing: its parent."""
        for kind, gid, other in rows:
            tid = self.owner.get(gid)
            if kind not in _BORN or tid is None:
                continue
            th = self.threads[tid]
            if th.first != k or th.first_id != gid:
                continue
            th.birth_row = True
            parent = self.owner.get(other) if other >= 0 else None
            if parent is not None and parent != tid:
                th.parent = parent

    def _exit_rows(self, rows: Iterable[tuple[str, int, int]], present: set[int], k: int, step: int) -> None:
        """The merged/dissolved row of a thread that was listed at the previous record and is not now."""
        for th in self.threads.values():
            if not th.ended and th.id not in present and th.last == k - 1 and th.gone_step is None:
                th.gone_step = step
        for kind, gid, other in rows:
            tid = self.owner.get(gid)
            if kind not in _EXIT or tid is None or tid in present:
                continue
            th = self.threads[tid]
            if th.last != k - 1 or gid not in th.ids:
                continue
            if th.exit_kind is not None and not (kind == "merged" and th.exit_kind != "merged"):
                continue
            into = self.owner.get(other) if (kind == "merged" and other >= 0) else None
            th.exit_kind = kind
            th.exit_into = into if into != tid else None
            th.exit_into_name = None if th.exit_into is None else self.threads[th.exit_into].shown

    def _name(self, th: Thread, k: int) -> None:
        """Establish the displayed name: the lowest unused suffix among the established threads of the same
        name that are open or ended less than ``NAME_GAP`` snapshots ago."""
        used = {
            o.suffix
            for o in self.threads.values()
            if o.id != th.id and o.est and o.name == th.name and (not o.ended or k - o.end_k < NAME_GAP)
        }
        n = 1
        while n in used:
            n += 1
        th.suffix = n

    def _establish(self, th: Thread, k: int) -> None:
        th.est, th.est_k = True, k
        self._name(th, k)

    def _toast_size(self, rec: Mapping[str, Any]) -> int:
        enter = (rec.get("thresholds") or {}).get("kin_enter")
        return TOAST_FACTOR * int(enter) if enter is not None else TOAST_SIZE_DEFAULT

    def _formed_event(self, th: Thread, step: int, toast: int) -> dict[str, Any]:
        parent = self.threads.get(th.parent) if th.parent is not None else None
        return {
            "kind": "formed",
            "step": step,
            "thread": th.id,
            "name": th.shown,
            "a": th.ids[0],
            "b": parent.ids[0] if parent is not None and parent.ids else 0,
            "pos": list(th.pos),
            "size": th.size,
            "peak": th.peak,
            "first_step": th.first_step,
            "end_step": None,
            "steps": None,
            "life": th.last - th.first + 1,
            "parent": None if parent is None else parent.shown,
            "into": None,
            "initial": False,
            "imp": 2 if th.size >= toast else 1,
            "consumed": th.first_step if th.birth_row else None,
        }

    def _ended_event(self, th: Thread, step: int, toast: int) -> dict[str, Any]:
        life = th.last - th.first + 1
        end_step = th.gone_step if th.gone_step is not None else th.last_step
        into = self.threads.get(th.exit_into) if th.exit_into is not None else None
        merged = th.exit_kind == "merged" and th.exit_into is not None
        return {
            "kind": "ended",
            "step": step,
            "thread": th.id,
            "name": th.shown,
            "a": th.ids[0] if th.ids else 0,
            "b": into.ids[0] if merged and into is not None and into.ids else 0,
            "pos": list(th.pos),
            "size": th.size,
            "peak": th.peak,
            "first_step": th.first_step,
            "end_step": end_step,
            "steps": None if th.init else end_step - th.first_step,
            "life": life,
            "parent": None,
            "into": (into.shown if into is not None else th.exit_into_name) if merged else None,
            "initial": th.init,
            "imp": 2 if (th.peak >= toast and life >= TOAST_LIFE) else 1,
            "consumed": end_step if th.exit_kind is not None else None,
        }

    def _close_epoch(self) -> None:
        """An observer restart renumbers the kin ids: every thread ends silently and the next record is a new
        initial cast."""
        for th in self.threads.values():
            if not th.ended:
                th.ended, th.end_k = True, self.k
        self.owner = {}
        self.first_record = True

    def _prune(self, k: int) -> None:
        """Drop the threads nothing refers to any more, then the oldest ones above ``MAX_THREADS``."""
        keep: dict[int, Thread] = {}
        for tid, th in self.threads.items():
            if th.ended and (not th.est or k - th.end_k >= NAME_GAP):
                continue
            keep.update({tid: th})
        self.threads = keep
        if len(self.threads) > MAX_THREADS:
            order = sorted(
                (th for th in self.threads.values() if th.last != k),
                key=lambda th: (0 if th.ended else 1 if not th.est else 2, th.last, th.id),
            )
            drop = {th.id for th in order[: len(self.threads) - MAX_THREADS]}
            self.threads = {tid: th for tid, th in self.threads.items() if tid not in drop}
        # only the ids of the last listing of an open thread stay owned (an id is never listed again once
        # gone, and the state restores exactly this table)
        self.owner = {i: th.id for th in self.threads.values() if not th.ended for i in th.ids}

    # --- reading ----------------------------------------------------------------------------------------

    def thread_of(self, gid: int) -> Thread | None:
        """The open thread that owns kin id ``gid`` (None if no thread does)."""
        tid = self.owner.get(gid)
        return self.threads.get(tid) if tid is not None else None

    def open_threads(self) -> list[Thread]:
        """The threads not ended, by id."""
        return [th for _, th in sorted(self.threads.items()) if not th.ended]

    # --- state ------------------------------------------------------------------------------------------

    def state(self) -> dict[str, Any]:
        """The complete state as plain JSON values (``observers/kinthreads_<t>.json``)."""
        return {
            "format": KINTHREADS_FORMAT,
            "version": KINTHREADS_VERSION,
            "constants": constants_digest(),
            "k": self.k,
            "step": self.step,
            "origin": self.origin,
            "next_id": self.next_id,
            "flaps": self.flaps,
            "first_record": self.first_record,
            "threads": [th.to_json() for _, th in sorted(self.threads.items())],
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> KinThreads:
        """The table of ``state`` (``ValueError`` if not a state of this version and these constants)."""
        if state.get("format") != KINTHREADS_FORMAT or state.get("version") != KINTHREADS_VERSION:
            raise ValueError("not a kin thread state of this version")
        if state.get("constants") != constants_digest():
            raise ValueError("the kin thread state was made with other constants")
        try:
            kt = cls()
            kt.k, kt.step = int(state["k"]), int(state["step"])
            kt.origin = None if state["origin"] is None else int(state["origin"])
            kt.next_id, kt.flaps = int(state["next_id"]), int(state["flaps"])
            kt.first_record = bool(state["first_record"])
            kt.threads = {th["id"]: Thread.from_json(th) for th in state["threads"]}
        except (KeyError, TypeError, ValueError, IndexError) as e:
            raise ValueError(f"unusable kin thread state: {e}") from e
        for th in kt.threads.values():
            if not th.ended:
                kt.owner.update(dict.fromkeys(th.ids, th.id))
        return kt


def kin_untold(
    records: Iterable[Mapping[str, Any]], told: Iterable[Mapping[str, Any]], a: int, b: int
) -> int:
    """The number of raw kin rows of the records in ``(a, b]`` that no told event consumed (spec §2.2): a
    told event consumes at most one row, the ``born``/``split`` row of its thread's first listing or the
    ``merged``/``dissolved`` row of the record after its last listing, counted in the record the row is in
    (an event can be told in a chapter later than its row)."""
    raw = sum(
        sum(f == KIN_FAMILY for f in r["events"]["family"])
        for r in records
        if a < int(r["step"]) <= b and r.get("events")
    )
    consumed = sum(1 for e in told if e.get("consumed") is not None and a < int(e["consumed"]) <= b)
    return raw - consumed
