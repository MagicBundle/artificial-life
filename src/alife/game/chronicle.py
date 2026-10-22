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

**Sources.** ``Chronicler.from_record(rec)`` reads a social snapshot record (``observers/social.jsonl``):
the kin groups, folded into **threads** (``game.kinthreads``: a group through its id changes, told when it
has held together for three snapshots and when it has been gone for four, so the flapping of the raw
``events`` table is not narrated, DESIGN §34.1), and ``house_events`` (founded in the record's top-16
houses, extinct among the previous record's top 16). Colony events are not narrated (80% of colony ids end
within three snapshots). The chronicler remembers the names of the houses it has been shown, so that an
extinct one is named by the name it had. ``from_storms`` takes a tick of ``storms.StormLayer`` (its tracker
events as plain dicts and the ids it detected) and applies the **persistence rule**: a birth is told once
the track has ``STORM_MIN_DETECTIONS`` detections, a naming at once, an end only for a track that was told,
and the events of the first tick of a layer made by this run (``initial``) are not told. ``from_director``
takes the entries of ``Director.applied``, and ``checkpoint`` ``renumbered`` ``extinct`` ``focus_death``
``focus_handoff`` are the play loop's own (a hand-off is one line: the death and the follow passing on).

Host code: no mlx, no array, no subscript assignment (``tests/test_lint.py``).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Final

import numpy as np

from alife.game.kinthreads import KinThreads
from alife.game.society import raw_house_names
from alife.storms import STORM_MIN_DETECTIONS
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

CHRONICLE_VERSION: Final = 2

KIND: Final = {name: number for number, name in EVENT_KINDS.items()}
FAMILY_BASE: Final = {"kin": 0, "house": 100, "storm": 200, "director": 300, "focus": 400, "system": 500}
PER_FAMILY: Final = 100  # ids a family has at one step
NAME_MEMORY: Final = 4096  # names remembered; the oldest go first
ID_MEMORY: Final = 4096  # (step, family) counters remembered

# The closed table. A slot is filled from the event; ``{n}`` is a count, ``{step}`` a step number.
TEMPLATES: Final = {
    "kin_formed": "{name} has held together since step {s0}: {size} members.",
    "kin_formed_from": (
        "{name} has held together since step {s0} after splitting from {other}: {size} members."
    ),
    "kin_ended_into": (
        "{name} merged into {other} at step {s}; it lasted {steps} steps and had up to {peak} members."
    ),
    "kin_ended_into_initial": "{name} merged into {other} at step {s}; it had up to {peak} members.",
    "kin_ended": "{name} dissolved after {steps} steps, up to {peak} members.",
    "kin_ended_initial": "{name} dissolved, up to {peak} members.",
    "house_founded": "House {name} was founded.",
    "house_founded_from": "House {name} was founded from House {parent}.",
    "house_extinct": "House {name} has no living members.",
    "storm_born": "{label} formed in the {hemisphere} hemisphere.",
    "storm_named": "Storm {name} reached named strength.",
    "storm_merged": "{label} merged into {other}.",
    "storm_split": "{label} split from {other}.",
    "storm_dissolved": "{label} dissolved after {age} steps.",
    "director_entry": "Director: {entry}",
    "power_food": "Food x{mult:g} from step {step:,} for {duration:,} steps.",
    "power_bloom": "Bloom x{gain:g} on patch {patch} from step {step:,} for {duration:,} steps.",
    "power_blight": "Blight on patch {patch} from step {step:,} for {duration:,} steps.",
    "power_strike": "Strike at ({x:.0f}, {y:.0f}), radius {radius:g}, kill chance {kill_p:g}, step {step:,}.",
    "power_ended": "{label} ended at step {step:,}.",
    "power_ended_patch": "{label} on patch {patch} ended at step {step:,}.",
    "power_cancelled": "{label} was cancelled at step {step:,}.",
    "power_cancelled_patch": "{label} on patch {patch} was cancelled at step {step:,}.",
    "focus_death": "Organism {uid} died ({cause}).",
    "focus_death_unknown": "Organism {uid} died.",
    "focus_handoff_child": "Organism {uid} died ({cause}); follow passed to a child.",
    "focus_handoff_kin": (
        "Organism {uid} died ({cause}); follow passed to the nearest member of its kin group."
    ),
    "focus_handoff_near": "Organism {uid} died ({cause}); follow passed to the nearest organism.",
    "focus_handoff_child_unknown": "Organism {uid} died; follow passed to a child.",
    "focus_handoff_kin_unknown": "Organism {uid} died; follow passed to the nearest member of its kin group.",
    "focus_handoff_near_unknown": "Organism {uid} died; follow passed to the nearest organism.",
    "checkpoint": "Checkpoint saved at step {step}.",
    "renumbered": "Storm numbering restarted at step {step}.",
    "extinct": "No organism is alive at step {step}.",
}
_POWER_LABEL: Final = {
    "regen_pulse": "Food",
    "fertility_shift": "Bloom",
    "habitat_loss": "Blight",
    "catastrophe": "Strike",
}  # the player's names for the director actions (director.actions.GAME_POWERS; this module may not import it)
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
        self.threads = KinThreads()  # the kin groups through their id changes (game.kinthreads)
        self._top: dict[int, str] = {}  # the previous record's listed houses, id -> name
        self._used: dict[tuple[int, str], int] = {}  # (step, family) -> ids handed out
        # the storm persistence rule: per track id, its detections so far, its born event while it waits for
        # ``STORM_MIN_DETECTIONS``, and whether the chronicle has told it (born or named) or it was named
        self._sdets: dict[int, int] = {}
        self._sheld: dict[int, dict[str, Any]] = {}
        self._sknown: set[int] = set()
        # what the latest ``from_record`` / ``from_storms`` told, for the alert engine (alerts): the thread
        # events with their sentences, and the tracker events (raw dicts) that became chronicle events
        self.told_threads: list[tuple[dict[str, Any], str]] = []
        self.told_storm: list[dict[str, Any]] = []

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
        """The kin-thread and house events of one social record, kin first (formed, then ended, by thread),
        then houses."""
        step = int(rec["step"])
        told = self.threads.fold(rec)
        events = [self._thread_event(t) for t in told]
        self.told_threads = [(t, e["text"]) for t, e in zip(told, events, strict=True)]
        houses = rec["houses"]
        listed = dict(zip(houses["id"], houses["name"], strict=True))
        events += self._house_events(step, rec.get("house_events"), listed)
        self._top = listed
        return self._stamp(events)

    def restore_threads(self, state: Mapping[str, Any]) -> None:
        """Take the thread table of a saved ``kinthreads_<t>.json`` (``ValueError`` if it is not usable): the
        table a stopped play had, with its flags, so a candidate that has not established yet is told when it
        does."""
        self.threads = KinThreads.from_state(state)

    def cast_names(self) -> dict[int, tuple[int, str]]:
        """``{kin id: (thread id, name as shown)}`` for the kin ids the latest record listed: the Cast keeps
        one name and one swatch per thread across a re-formation."""
        return {i: (th.id, th.shown) for th in self.threads.open_threads() for i in th.ids}

    @staticmethod
    def _thread_event(t: Mapping[str, Any]) -> dict[str, Any]:
        """The event of a told thread change (``KinThreads.fold``)."""
        slots: dict[str, Any] = {"name": t["name"], "size": t["size"], "peak": t["peak"]}
        if t["kind"] == "formed":
            key = "kin_formed_from" if t["parent"] else "kin_formed"
            text = render(key, s0=t["first_step"], other=t["parent"], **slots)
            kind = "kin_formed"
        else:
            tail = "_initial" if t["initial"] else ""
            key = ("kin_ended_into" if t["into"] else "kin_ended") + tail
            text = render(key, s=t["end_step"], steps=t["steps"], other=t["into"], **slots)
            kind = "kin_ended"
        x, y, z = t["pos"]
        return _event(t["step"], kind, "kin", text, t["imp"], a=t["a"], b=t["b"], at=(x, y, z))

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

    # --- storms ------------------------------------------------------------------------------------------

    def from_storms(
        self, step: int, events: Iterable[Mapping[str, Any]], detected: Iterable[int] = ()
    ) -> list[dict[str, Any]]:
        """The events of the tracker tick at ``step``, with the persistence rule (DESIGN §34.1). Each event is
        the dict ``StormLayer`` writes into ``storms.jsonl``: ``kind`` (born, named, merged, split,
        dissolved), ``step``, ``id``, ``name`` (None while a disturbance), ``other`` and ``other_name`` (the
        storm it merged into or split from, 0 and None if none), ``x`` ``y``, ``hemi``, ``age`` (steps, for a
        dissolved one) and ``initial`` (the first tick of a layer made by this run). ``detected`` is the ids
        the tick detected (the ``storms`` list of the log line).

        A ``born`` or ``split`` track waits until it has ``STORM_MIN_DETECTIONS`` detections and is told
        then, at this tick; a ``named`` event is told at once and replaces the birth sentence; ``merged`` and
        ``dissolved`` are told only for a track that was told as born or named (or was named at the first
        tick); the events of a first tick are not told."""
        events = list(events)
        self.told_storm = []
        for e in events:
            if e["kind"] in ("born", "split"):  # an id is born once; a restarted tracker reuses them
                ident = int(e["id"])
                self._sdets.update({ident: 0})
                self._sheld.pop(ident, None)
                self._sknown.discard(ident)
        for ident in detected:
            self._sdets.update({int(ident): self._sdets.get(int(ident), 0) + 1})
        out = []
        for e in events:
            ident, kind = int(e["id"]), e["kind"]
            if kind not in ("born", "named", "merged", "split", "dissolved"):
                raise ValueError(f"unknown tracker event {kind!r}")
            if kind in ("born", "split"):
                if not e.get("initial"):
                    self._sheld.update({ident: e})
                continue
            if kind == "named":
                self._sheld.pop(ident, None)
                self._sknown.add(ident)
                if not e.get("initial"):
                    out.append(self._storm_event(step, e))
                    self.told_storm.append(dict(e))
                continue
            self._sheld.pop(ident, None)
            told = ident in self._sknown
            self._sdets.pop(ident, None)
            self._sknown.discard(ident)
            if told:
                out.append(self._storm_event(step, e))
                self.told_storm.append(dict(e))
        for ident, e in list(self._sheld.items()):
            if self._sdets.get(ident, 0) >= STORM_MIN_DETECTIONS:
                self._sheld.pop(ident)
                self._sknown.add(ident)
                out.append(self._storm_event(step, e))
                self.told_storm.append(dict(e))
        for memory in (self._sdets, self._sheld):  # ids of tracks that vanished without an end (a restart)
            for old in list(memory)[: max(0, len(memory) - NAME_MEMORY)]:
                memory.pop(old)
        return self._stamp(out)

    @staticmethod
    def _storm_event(step: int, e: Mapping[str, Any]) -> dict[str, Any]:
        """The event of a told tracker event, dated ``step`` (the tick at which it is told)."""
        ident, name, kind = int(e["id"]), e.get("name"), e["kind"]
        label, other = _label(name, ident), _label(e.get("other_name"), int(e.get("other") or 0))
        if kind == "born":
            text, imp = render("storm_born", label=label, hemisphere=HEMISPHERES[int(e["hemi"])]), 1
        elif kind == "named":
            text, imp = render("storm_named", name=name or ident), 2
        elif kind == "merged":
            text, imp = render("storm_merged", label=label, other=other), 1
        elif kind == "split":
            text, imp = render("storm_split", label=label, other=other), 1
        else:
            text, imp = render("storm_dissolved", label=label, age=int(e["age"])), 2 if name else 1
        return _event(
            step,
            f"storm_{kind}",
            "storm",
            text,
            imp,
            a=ident,
            b=int(e.get("other") or 0),
            at=(e["x"], e["y"]),
        )

    # --- the director, the focus, the play loop ----------------------------------------------------------

    def from_director(self, entries: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """One event per entry (``Director.applied`` or a log entry), at its ``step_applied``; the position is
        the entry's ``x`` and ``y`` parameters when it has them."""
        from alife.director import actions, describe  # imported here: the director package is heavy

        out = []
        for entry in entries:
            if actions.is_player(entry):
                out.append(self._power_event(entry))
                continue
            params = entry.get("params") or {}
            at = (params.get("x"), params.get("y"))
            text = render("director_entry", entry=describe(entry))
            out.append(_event(int(entry["step_applied"]), "director_entry", "director", text, 1, at=at))
        return self._stamp(out)

    @staticmethod
    def _power_event(entry: Mapping[str, Any]) -> dict[str, Any]:
        """The chronicle row of a record of the player channel (DESIGN §34.5): kind ``power_applied``
        (importance 1) for a power, ``power_ended`` (importance 0) for the revert that ends or cancels one; a
        strike has a position. No alert (the surfaces of one power are its ``PLAN`` cards, §34.3)."""
        step, params = int(entry["step_applied"]), entry.get("params") or {}
        patch = params.get("patch")
        if entry["source"] == "player":
            power = entry["power"]
            slots = {
                "food": {"mult": params.get("mult"), "duration": entry.get("duration")},
                "bloom": {"gain": params.get("gain"), "patch": patch, "duration": entry.get("duration")},
                "blight": {"patch": patch, "duration": entry.get("duration")},
                "strike": {
                    "x": (params.get("center") or [0, 0])[0],
                    "y": (params.get("center") or [0, 0])[1],
                    "radius": params.get("radius"),
                    "kill_p": params.get("kill_p"),
                },
            }[power]
            at = tuple(params["center"]) if power == "strike" else (None, None)
            text = render(f"power_{power}", step=step, **slots)
            return _event(step, "power_applied", "director", text, 1, at=at)
        label = _POWER_LABEL[entry["action"]]
        key = "power_cancelled" if "command" in entry else "power_ended"
        key += "_patch" if patch is not None else ""
        text = render(key, label=label, patch=patch, step=step)
        return _event(step, "power_ended", "director", text, 0)

    def focus_death(
        self, step: int, uid: int, cause: str | None, at: Sequence[float | None]
    ) -> list[dict[str, Any]]:
        """The death of the followed organism when the follow cannot pass on; ``cause`` is a cause name (None
        if the lanes overflowed)."""
        key = "focus_death" if cause else "focus_death_unknown"
        text = render(key, uid=uid, cause=cause)
        return self._stamp([_event(step, "focus_death", "focus", text, 1, a=uid, at=at)])

    def focus_handoff(
        self,
        step: int,
        from_uid: int,
        to_uid: int,
        rule: int,
        at: Sequence[float | None],
        cause: str | None = None,
    ) -> list[dict[str, Any]]:
        """The followed organism died and the follow passes to ``to_uid`` by rule 1 (a child), 2 (nearest of
        its kin) or 3 (nearest): one line for the death and the hand-off."""
        key = HANDOFF_RULES[rule] + ("" if cause else "_unknown")
        text = render(key, uid=from_uid, cause=cause)
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
    chronicler: Chronicler | None = None,
) -> list[dict[str, Any]]:
    """The newest ``limit`` events a play of this run would have produced from its logs: the records of
    ``observers/social.jsonl``, the lines of ``observers/storms.jsonl`` (each with an ``events`` list) and the
    entries of ``director.jsonl`` (headers excluded by the caller), in order of ``(step, id)``. The kin
    threads and the storm persistence are folds over the whole log, so pass all of it. With ``chronicler`` the
    events are made by that one (a live play's, which then holds the thread table, the storm counters and the
    ids handed out, so it continues the log it was started from); otherwise by a new one."""
    chron = Chronicler(house_salt) if chronicler is None else chronicler
    events: list[dict[str, Any]] = []
    for rec in social:
        events += chron.from_record(rec)
    for line in storms:
        events += chron.from_storms(
            int(line["step"]), line.get("events", ()), [s["id"] for s in line.get("storms", ())]
        )
    events += chron.from_director(director)
    events.sort(key=lambda e: (e["step"], e["id"]))
    return events[-limit:] if limit > 0 else []
