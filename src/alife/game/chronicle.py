"""The chronicle of a played world: ``EVNT`` events from the observers, the storm tracker, the director and
the play loop (DESIGN §33.2, ``streamrec`` §33.1.8).

An event is the 11-key object of ``streamrec.EVENT_KEYS``. Its text comes from a **closed table of
sentences** (``TEMPLATES``, ``CHRONICLE_VERSION``): one fixed sentence per case with slots filled with names,
exact counts of the source record and steps. No model writes a word. The sentences state what a record says
and nothing about why: no causal word, no label shares, no archetype claim (the labels are provisional;
GAMES §10 C8 to C10), "since" and never "because". ``tests/test_play_chronicle.py`` renders every template and
runs the world report's lexicon (``game.lexicon``, DESIGN §31) over the texts.

**Ids.** ``id = step * 1000 + FAMILY_BASE[fam] + index``: unique per run, ordered by step, and *independent of
the order in which two sources deliver events of one step* (kin, house, storm, director, focus and system have
a hundred ids each), so the backlog rebuilt from the logs (``rebuild_backlog``) gives the ids the live
``Chronicler`` gave and a resumed play continues without a clash. A family that has more than a hundred
events at one step keeps its first hundred.

**Sources.** ``Chronicler.from_record(rec)`` reads a social snapshot record (``observers/social.jsonl``): the
``events`` table (kin groups; colony events are not narrated, 80% of colony ids end within three snapshots)
and ``house_events`` (founded in the record's top-16 houses, extinct among the previous record's top 16). The
chronicler remembers the names of the kin groups and houses it has been shown, so that a dissolved group is
named by the name it had. ``from_storms`` takes the tracker events of ``storms.StormLayer`` (plain dicts),
``from_director`` the entries of ``Director.applied``, and ``checkpoint`` ``renumbered`` ``extinct``
``focus_death`` ``focus_handoff`` are the play loop's own.

Host code: no mlx, no array, no subscript assignment (``tests/test_lint.py``).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Final

import numpy as np

from alife.game.society import raw_house_names
from alife.streamrec import EVENT_KINDS, MAX_TEXT, check_event

__all__ = [
    "CHRONICLE_VERSION",
    "FAMILY_BASE",
    "KIND",
    "TEMPLATES",
    "Chronicler",
    "event_id",
    "rebuild_backlog",
    "render",
]

CHRONICLE_VERSION: Final = 1

KIND: Final = {name: number for number, name in EVENT_KINDS.items()}
FAMILY_BASE: Final = {"kin": 0, "house": 100, "storm": 200, "director": 300, "focus": 400, "system": 500}
PER_FAMILY: Final = 100  # ids a family has at one step
KIN_FAMILY: Final = "kin_group"  # the family name of kin events in the record's ``events`` table
NAME_MEMORY: Final = 4096  # names remembered; the oldest go first
ID_MEMORY: Final = 4096  # (step, family) counters remembered

# The closed table. A slot is filled from the event; ``{n}`` is a count, ``{step}`` a step number.
TEMPLATES: Final = {
    "kin_born": "{name} formed with {size} members.",
    "kin_split": "{name} split from {other} with {size} members.",
    "kin_split_alone": "{name} split off with {size} members.",
    "kin_merged": "{name} merged into {other} ({size} members).",
    "kin_merged_alone": "{name} merged ({size} members).",
    "kin_dissolved": "{name} dissolved at {size} members.",
    "kin_dissolved_into": "{name} dissolved at {size} members; {overlap} of them are in {other}.",
    "house_founded": "House {name} was founded.",
    "house_founded_from": "House {name} was founded from House {parent}.",
    "house_extinct": "House {name} has no living members.",
    "storm_born": "{label} formed in the {hemisphere} hemisphere.",
    "storm_named": "Storm {name} reached named strength.",
    "storm_merged": "{label} merged into {other}.",
    "storm_split": "{label} split from {other}.",
    "storm_dissolved": "{label} dissolved after {age} steps.",
    "director_entry": "Director: {entry}",
    "focus_death": "Organism {uid} died ({cause}).",
    "focus_death_unknown": "Organism {uid} died.",
    "focus_handoff_child": "Follow passed to a child.",
    "focus_handoff_kin": "Follow passed to the nearest member of its kin group.",
    "focus_handoff_near": "Follow passed to the nearest organism.",
    "checkpoint": "Checkpoint saved at step {step}.",
    "renumbered": "Storm numbering restarted at step {step}.",
    "extinct": "No organism is alive at step {step}.",
}
HEMISPHERES: Final = ("northern", "southern")
HANDOFF_RULES: Final = {1: "focus_handoff_child", 2: "focus_handoff_kin", 3: "focus_handoff_near"}


def render(key: str, **slots: Any) -> str:
    """The sentence of template ``key`` with its slots, cut to the 120 characters an event may carry."""
    text = TEMPLATES[key].format(**slots)
    return text if len(text) <= MAX_TEXT else text[: MAX_TEXT - 3].rstrip() + "..."


def event_id(step: int, fam: str, index: int) -> int:
    return step * 1000 + FAMILY_BASE[fam] + index


def _event(
    step: int,
    kind: str,
    fam: str,
    text: str,
    imp: int,
    *,
    a: int = 0,
    b: int = 0,
    at: Sequence[float | None] = (None, None, None),
) -> dict[str, Any]:
    """An event without its id (``Chronicler._stamp`` numbers a step's events of a family)."""
    x, y, z = [*at, None, None, None][:3]
    return {
        "id": 0,
        "step": int(step),
        "kind": KIND[kind],
        "fam": fam,
        "a": int(a),
        "b": int(b),
        "x": x,
        "y": y,
        "z": z,
        "imp": imp,
        "text": text,
    }


def _label(name: str | None, ident: int) -> str:
    return f"Storm {name}" if name else f"Disturbance {ident}"


class Chronicler:
    """Turns the sources of a played world into events and keeps what the sentences need to name things."""

    def __init__(self, house_salt: int) -> None:
        self.house_salt = house_salt
        self._kin: dict[int, str] = {}
        self._top: dict[int, str] = {}  # the previous record's listed houses, id -> name
        self._used: dict[tuple[int, str], int] = {}  # (step, family) -> ids handed out

    # --- ids --------------------------------------------------------------------------------------------

    def _stamp(self, events: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        """The events with ids: the next free index of their (step, family), at most ``PER_FAMILY`` each."""
        out: list[dict[str, Any]] = []
        for e in events:
            key = (e["step"], e["fam"])
            n = self._used.get(key, 0)
            if n >= PER_FAMILY:
                continue
            self._used.update({key: n + 1})
            out.append({**e, "id": event_id(e["step"], e["fam"], n)})
            check_event(out[-1])
        for old in list(self._used)[: max(0, len(self._used) - ID_MEMORY)]:
            self._used.pop(old)
        return out

    # --- kin groups and houses: a snapshot record -------------------------------------------------------

    def from_record(self, rec: Mapping[str, Any]) -> list[dict[str, Any]]:
        """The kin-group and house events of one social record, kin first (table order), then houses."""
        step = int(rec["step"])
        kin = rec["kin_groups"]
        where = {
            gid: (kin["x"][i], kin["y"][i], kin["z_mean"][i] if "z_mean" in kin else None)
            for i, gid in enumerate(kin["id"])
        }
        self._remember(self._kin, dict(zip(kin["id"], kin["name"], strict=True)))
        events = [self._kin_event(step, rec["events"], i, where) for i in range(len(rec["events"]["id"]))]
        events = [e for e in events if e is not None]
        houses = rec["houses"]
        listed = dict(zip(houses["id"], houses["name"], strict=True))
        events += self._house_events(step, rec.get("house_events"), listed)
        self._top = listed
        return self._stamp(events)

    def _kin_name(self, gid: int) -> str:
        return self._kin.get(gid, f"Kin group {gid}")

    def _kin_event(
        self, step: int, table: Mapping[str, list[Any]], i: int, where: Mapping[int, Sequence[Any]]
    ) -> dict[str, Any] | None:
        if table["family"][i] != KIN_FAMILY:
            return None
        kind, gid, size = table["kind"][i], table["id"][i], table["size"][i]
        other, overlap = table["other"][i], table["overlap"][i]
        name, near = self._kin_name(gid), (self._kin_name(other) if other >= 0 else None)
        pos = where.get(gid, (None, None, None))
        if kind == "born":
            return _event(step, "kin_born", "kin", render("kin_born", name=name, size=size), 1, a=gid, at=pos)
        if kind == "split":
            key = "kin_split" if near else "kin_split_alone"
            text = render(key, name=name, other=near, size=size)
            return _event(step, "kin_split", "kin", text, 2, a=gid, b=max(other, 0), at=pos)
        if kind == "merged":
            key = "kin_merged" if near else "kin_merged_alone"
            text = render(key, name=name, other=near, size=size)
            return _event(
                step, "kin_merged", "kin", text, 1, a=gid, b=max(other, 0), at=where.get(other, pos)
            )
        if kind == "dissolved":
            into = near is not None and overlap > 0
            key = "kin_dissolved_into" if into else "kin_dissolved"
            text = render(key, name=name, other=near, size=size, overlap=overlap)
            return _event(step, "kin_dissolved", "kin", text, 2, a=gid, b=max(other, 0) if into else 0)
        return None

    def _house_events(
        self, step: int, events: Mapping[str, Any] | None, listed: Mapping[int, str]
    ) -> list[dict[str, Any]]:
        if not events:
            return []
        out = []
        founded = events["founded"]
        for hid, parent in zip(founded["id"], founded["parent_house"], strict=True):
            if hid not in listed:  # only the top-16 houses are narrated
                continue
            if parent is None:
                text = render("house_founded", name=listed[hid])
            else:
                other = (
                    listed.get(parent)
                    or self._top.get(parent)
                    or raw_house_names(np.array([parent]), self.house_salt)[0]
                )
                text = render("house_founded_from", name=listed[hid], parent=other)
            out.append(_event(step, "house_founded", "house", text, 1, a=hid, b=parent or 0))
        for hid in events["extinct"]["id"]:
            if hid in self._top:  # it was listed at the previous record
                out.append(
                    _event(
                        step, "house_extinct", "house", render("house_extinct", name=self._top[hid]), 1, a=hid
                    )
                )
        return out

    @staticmethod
    def _remember(memory: dict[int, str], names: Mapping[int, str]) -> None:
        memory.update(names)  # a dict update, not a subscript assignment
        for old in list(memory)[: max(0, len(memory) - NAME_MEMORY)]:
            memory.pop(old)

    # --- storms ------------------------------------------------------------------------------------------

    def from_storms(self, events: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """The events of a tracker tick. Each input is the dict ``StormLayer`` writes into ``storms.jsonl``:
        ``kind`` (born, named, merged, split, dissolved), ``step``, ``id``, ``name`` (None while a
        disturbance), ``other`` and ``other_name`` (the storm it merged into or split from, 0 and None if
        none), ``x`` ``y``, ``hemi``, ``age`` (steps, for a dissolved one)."""
        out = []
        for e in events:
            step, ident, name = int(e["step"]), int(e["id"]), e.get("name")
            label, other = _label(name, ident), _label(e.get("other_name"), int(e.get("other") or 0))
            kind = e["kind"]
            if kind == "born":
                text, imp = render("storm_born", label=label, hemisphere=HEMISPHERES[int(e["hemi"])]), 1
            elif kind == "named":
                text, imp = render("storm_named", name=name or ident), 2
            elif kind == "merged":
                text, imp = render("storm_merged", label=label, other=other), 1
            elif kind == "split":
                text, imp = render("storm_split", label=label, other=other), 1
            elif kind == "dissolved":
                text, imp = render("storm_dissolved", label=label, age=int(e["age"])), 2 if name else 1
            else:
                raise ValueError(f"unknown tracker event {kind!r}")
            out.append(
                _event(
                    step,
                    f"storm_{kind}",
                    "storm",
                    text,
                    imp,
                    a=ident,
                    b=int(e.get("other") or 0),
                    at=(e["x"], e["y"]),
                )
            )
        return self._stamp(out)

    # --- the director, the focus, the play loop ----------------------------------------------------------

    def from_director(self, entries: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """One event per entry (``Director.applied`` or a log entry), at its ``step_applied``; the position is
        the entry's ``x`` and ``y`` parameters when it has them."""
        from alife.director import describe  # imported here: the director package is heavy and rarely needed

        out = []
        for entry in entries:
            params = entry.get("params") or {}
            at = (params.get("x"), params.get("y"))
            text = render("director_entry", entry=describe(entry))
            out.append(_event(int(entry["step_applied"]), "director_entry", "director", text, 1, at=at))
        return self._stamp(out)

    def focus_death(
        self, step: int, uid: int, cause: str | None, at: Sequence[float | None]
    ) -> list[dict[str, Any]]:
        """The death of the followed organism; ``cause`` is a cause name (None if the lanes overflowed)."""
        key = "focus_death" if cause else "focus_death_unknown"
        text = render(key, uid=uid, cause=cause)
        return self._stamp([_event(step, "focus_death", "focus", text, 1, a=uid, at=at)])

    def focus_handoff(
        self, step: int, from_uid: int, to_uid: int, rule: int, at: Sequence[float | None]
    ) -> list[dict[str, Any]]:
        """The follow passing to ``to_uid`` by rule 1 (a child), 2 (nearest of its kin) or 3 (nearest)."""
        text = render(HANDOFF_RULES[rule])
        return self._stamp([_event(step, "focus_handoff", "focus", text, 1, a=from_uid, b=to_uid, at=at)])

    def checkpoint(self, step: int) -> list[dict[str, Any]]:
        return self._stamp([_event(step, "checkpoint", "system", render("checkpoint", step=step), 0)])

    def renumbered(self, step: int) -> list[dict[str, Any]]:
        return self._stamp([_event(step, "renumbered", "system", render("renumbered", step=step), 1)])

    def extinct(self, step: int) -> list[dict[str, Any]]:
        return self._stamp([_event(step, "extinct", "system", render("extinct", step=step), 2)])


def rebuild_backlog(
    social: Iterable[Mapping[str, Any]],
    storms: Iterable[Mapping[str, Any]],
    director: Iterable[Mapping[str, Any]],
    *,
    house_salt: int,
    limit: int = 64,
) -> list[dict[str, Any]]:
    """The newest ``limit`` events a play of this run would have produced from its logs: the records of
    ``observers/social.jsonl``, the lines of ``observers/storms.jsonl`` (each with an ``events`` list) and the
    entries of ``director.jsonl`` (headers excluded by the caller), in order of ``(step, id)``. Names are
    remembered from the records given, so pass the whole log when exact names of dissolved groups matter."""
    chron = Chronicler(house_salt)
    events: list[dict[str, Any]] = []
    for rec in social:
        events += chron.from_record(rec)
    for line in storms:
        events += chron.from_storms(line.get("events", ()))
    events += chron.from_director(director)
    events.sort(key=lambda e: (e["step"], e["id"]))
    return events[-limit:] if limit > 0 else []
