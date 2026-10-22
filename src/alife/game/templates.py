"""The templates of the world report (docs/DESIGN.md §31, ``TEMPLATE_VERSION``): every sentence a report
shows is one of these strings with its slots filled from the fact sheet (``game.report``), nothing else.

Version 2 (G3b, DESIGN §34.4): the kin sentences are the chronicle's own (a *thread* through its id changes,
``game.kinthreads``), the storm sentences ``ST-*`` and their headline are new, the deaths list gives way to a
turnover sentence, the sentence on the changes too short to list (``KS``) and a quiet chapter's headline are
new, and a player's entry is "the player".

A template is data: an id, a role (``lead``, ``body``, ``headline``, ``block``, ``ticker``), a text with
``{slot}`` and ``{marker}`` tokens, the slots (each reads one field of one kind of fact and has a format)
and the
markers (agreement words that follow a count slot: "was"/"were", "death"/"deaths"). The module also holds the
vocabulary derived from them (``TEMPLATE_VOCAB``), the words for the director's sources (``SOURCE_WORDS``),
the
section order of a body and the salience order of its candidates, and the number formats.
``TEMPLATE_DIGEST`` pins
all of it; a stored record carries it and ``verify`` compares it.

Import-time checks (a violation raises ``ValueError``): a template has no literal digit (numbers come from
slots, so the count of number tokens in a rendered text is the count of slots); a slot or marker token is
preceded by the start, a space or "(" and followed by a space, ",", ".", ";", ":", ")" or the end (so two
slots
never touch and no suffix such as "k" or "st" can be glued to a number); every token is declared; a marker
names a declared count slot.

Lint (rule 2, ``tests/test_lint.py``): no subscript assignment anywhere in this module, dict keys included.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any, Final, NamedTuple

__all__ = [
    "AGREE_WORDS",
    "BLOCK_ROLES",
    "DISPLAY_WORDS",
    "HEMISPHERE_WORDS",
    "RANK_WORDS",
    "RULE_NAMES",
    "SALIENCE_ORDER",
    "SECTION_OF",
    "SECTION_ORDER",
    "SOURCE_WORDS",
    "TEMPLATES",
    "TEMPLATE_DIGEST",
    "TEMPLATE_VERSION",
    "TEMPLATE_VOCAB",
    "Slot",
    "Template",
    "fmt_change",
    "fmt_dec",
    "fmt_int",
    "fmt_lever",
    "fmt_pct",
    "format_value",
    "render",
    "source_words",
    "template_words",
    "tokens_of",
    "validate",
]

TEMPLATE_VERSION: Final = 2


class Slot(NamedTuple):
    """One filled position of a template: the kind of fact it reads, the (dotted) field and the format."""

    kind: str  # a fact kind, or "chapter", "excluded", "snapshot"
    field: str  # a key of the fact; "params.mult" or "params.center.0" reach into it
    fmt: str  # int | id | pct | change | lever | dec1 | dec2 | kin | house | label | name | rule | text


class Template(NamedTuple):
    id: str
    role: str  # lead | body | headline | block | ticker
    text: str
    slots: Mapping[str, Slot]
    agree: Mapping[str, tuple[str, str]]  # marker -> (count slot, key of AGREE_WORDS)


# --- agreement words ---------------------------------------------------------------------------------------

# key -> (singular, plural). The first form is used at exactly 1, the second at every other count (0, 2, ...).
AGREE_WORDS: Final[Mapping[str, tuple[str, str]]] = {
    "be": ("was", "were"),
    "is": ("is", "are"),
    "death": ("death", "deaths"),
    "birth": ("birth", "births"),
    "arrival": ("arrival", "arrivals"),
    "member": ("member", "members"),
    "agent": ("agent", "agents"),
    "house": ("house", "houses"),
    "colony": ("colony", "colonies"),
    "step": ("step", "steps"),
    "setting": ("setting", "settings"),
    "immigrant": ("immigrant", "immigrants"),
    "organism": ("organism", "organisms"),
    "storm": ("storm", "storms"),
    "change": ("change", "changes"),
}
RANK_WORDS: Final = ("largest", "second largest", "third largest")  # the rank words of G-r1, G-r2, G-r3
HEMISPHERE_WORDS: Final = ("northern", "southern")  # the hemisphere slot of ST-peak (a "text" slot)
# Words an entity or label display adds around its name ("the Braelix kin", "House Zaeld", "kin group 12").
DISPLAY_WORDS: Final = ("the", "kin", "house", "group")

# The director's rule identifiers as they appear in an entry's ``rule`` field (``director.rules._TAGS`` and
# the
# yoked persona's ``random``). A test asserts that this is exactly the set of the rules' names.
RULE_NAMES: Final = frozenset(
    {"extinction_rescue", "runaway_damper", "monoculture_breaker", "stagnation_response", "boredom", "random"}
)
# How a report names the author of an entry: the model, a rule by its own identifier (an opaque name, masked
# by
# the checker; the log does not say what triggered a rule), or the director when the entry has no ``rule``.
SOURCE_WORDS: Final[Mapping[str, str]] = {
    "llm": "the director's model",
    "rule": "the director's rule {rule}",
    "none": "the director",
    "player": "the player",
}


def source_words(source: str, rule: str | None) -> str:
    """The words for the author of an entry (``source`` ``rule``, ``llm``, ``player`` or ``revert``; ``rule``
    its rule name)."""
    if source == "llm":
        return SOURCE_WORDS["llm"]
    if source == "player":
        return SOURCE_WORDS["player"]
    if rule is not None and rule in RULE_NAMES:
        return SOURCE_WORDS["rule"].format(rule=rule)
    return SOURCE_WORDS["none"]


# --- the catalogue -----------------------------------------------------------------------------------------

_CH_A, _CH_B = Slot("chapter", "start", "int"), Slot("chapter", "end", "int")


def _s(kind: str, field: str, fmt: str = "int") -> Slot:
    return Slot(kind, field, fmt)


def _t(
    tid: str,
    role: str,
    text: str,
    slots: Mapping[str, Slot],
    agree: Mapping[str, tuple[str, str]] | None = None,
):
    return Template(tid, role, text, dict(slots), dict(agree or {}))


_POP = "population"
_EV = "event"
_HOUSE_CORE = "{alive_to} {house_n} had living members at step {b}, against {alive_from} at step {a}"
_HOUSE_SLOTS = {
    "alive_to": _s("houses", "alive_to"),
    "alive_from": _s("houses", "alive_from"),
    "a": _CH_A,
    "b": _CH_B,
}
_HOUSE_AGREE = {"house_n": ("alive_to", "house")}
_INEXACT = "Counting houses traced from step {origin}, "


def _house_rows() -> list[Template]:
    clauses = {
        "": ("", {}, {}),
        "n": (
            "; {first_seen} {be} new",
            {"first_seen": _s("houses", "first_seen")},
            {"be": ("first_seen", "be")},
        ),
        "e": ("; {extinct} died out", {"extinct": _s("houses", "extinct")}, {}),
        "ne": (
            "; {first_seen} {be} new and {extinct} died out",
            {"first_seen": _s("houses", "first_seen"), "extinct": _s("houses", "extinct")},
            {"be": ("first_seen", "be")},
        ),
    }
    rows = []
    for exact in (True, False):
        for key, (clause, extra, agree) in clauses.items():
            tid = "H-1" + ("" if exact else "x") + key
            text = ("" if exact else _INEXACT) + _HOUSE_CORE + clause + "."
            slots = {**_HOUSE_SLOTS, **extra, **({} if exact else {"origin": _s("houses", "origin")})}
            rows.append(_t(tid, "body", text, slots, {**_HOUSE_AGREE, **agree}))
    return rows


def _ev_slots(*more: str) -> dict[str, Slot]:
    base = {
        "s0": _s(_EV, "s0"),
        "s1": _s(_EV, "s1"),
        "size": _s(_EV, "size"),
        "sys": _s(_EV, "system", "kin"),
        "other": _s(_EV, "other", "kin"),
        "overlap": _s(_EV, "overlap"),
    }
    return {k: v for k, v in base.items() if k in ("s0", "s1", "size", *more)}


_MEMBER = {"member_n": ("size", "member")}
_KT_FIELDS = {
    "name": ("name", "name"),
    "other": ("other", "name"),
    "s0": ("first_step", "int"),
    "s": ("end", "int"),
    "size": ("size", "int"),
    "peak": ("peak", "int"),
    "steps": ("steps", "int"),
}


def _kt_slots(*names: str) -> dict[str, Slot]:
    """The slots of a kin-thread sentence: ``name`` and ``other`` are thread names (a ``name`` slot, masked by
    the checker), the rest counts and steps of the ``thread`` fact."""
    return {n: _s("thread", _KT_FIELDS[n][0], _KT_FIELDS[n][1]) for n in names}


_GROUP_SLOTS = {
    "K": _s("group", "entity", "kin"),
    "b": _CH_B,
    "size": _s("group", "size"),
    "share": _s("group", "share", "pct"),
}


def _i_rows() -> list[Template]:
    i = "intervention"
    base = {"s": _s(i, "step"), "src": _s(i, "src", "text")}
    return [
        _t(
            "I-regen",
            "block",
            "At step {s}, {src} set food regrowth to {mult} times its base rate for {duration} steps.",
            {**base, "mult": _s(i, "params.mult", "lever"), "duration": _s(i, "duration")},
        ),
        _t(
            "I-loss",
            "block",
            "At step {s}, {src} switched off food regrowth on patch {patch} for {duration} steps.",
            {**base, "patch": _s(i, "params.patch", "id"), "duration": _s(i, "duration")},
        ),
        _t(
            "I-seasons",
            "block",
            "At step {s}, {src} set the seasons to a period of {period} steps and an amplitude of {amp}.",
            {**base, "period": _s(i, "params.period"), "amp": _s(i, "params.amp", "lever")},
        ),
        _t(
            "I-drift",
            "block",
            "At step {s}, {src} set the drift of food patches to {mult} times its base speed.",
            {**base, "mult": _s(i, "params.mult", "lever")},
        ),
        _t(
            "I-fert",
            "block",
            "At step {s}, {src} set the fertility of patch {patch} to {gain} times its base value.",
            {**base, "patch": _s(i, "params.patch", "id"), "gain": _s(i, "params.gain", "lever")},
        ),
        _t(
            "I-fert-d",
            "block",
            "At step {s}, {src} set the fertility of patch {patch} to {gain} times its base value for "
            "{duration} steps.",
            {
                **base,
                "patch": _s(i, "params.patch", "id"),
                "gain": _s(i, "params.gain", "lever"),
                "duration": _s(i, "duration"),
            },
        ),
        _t(
            "I-move",
            "block",
            "At step {s}, {src} moved patch {patch} to ({x}, {y}).",
            {
                **base,
                "patch": _s(i, "params.patch", "id"),
                "x": _s(i, "params.x", "lever"),
                "y": _s(i, "params.y", "lever"),
            },
        ),
        _t(
            "I-imm",
            "block",
            "At step {s}, {src} queued {count} {immigrant_n}.",
            {**base, "count": _s(i, "params.count")},
            {"immigrant_n": ("count", "immigrant")},
        ),
        _t(
            "I-cat",
            "block",
            "At step {s}, {src} set off a catastrophe of radius {radius} around ({x}, {y}) with a "
            "kill chance of "
            "{kill_p}.",
            {
                **base,
                "radius": _s(i, "params.radius", "lever"),
                "x": _s(i, "params.center.0", "lever"),
                "y": _s(i, "params.center.1", "lever"),
                "kill_p": _s(i, "params.kill_p", "lever"),
            },
        ),
    ]


def _s_rows() -> list[Template]:
    k = "standing"
    s = _s(k, "step")
    return [
        _t(
            "S-regen",
            "block",
            "From step {s}, food regrowth has been {mult} times its base rate.",
            {"s": s, "mult": _s(k, "mult", "lever")},
        ),
        _t(
            "S-gain",
            "block",
            "From step {s}, the fertility of patch {patch} has been {gain} times its base value.",
            {"s": s, "patch": _s(k, "patch", "id"), "gain": _s(k, "gain", "lever")},
        ),
        _t(
            "S-seasons",
            "block",
            "From step {s}, the seasons have had a period of {period} steps and an amplitude of {amp}.",
            {"s": s, "period": _s(k, "period"), "amp": _s(k, "amp", "lever")},
        ),
        _t(
            "S-drift",
            "block",
            "From step {s}, the drift of food patches has been {mult} times its base speed.",
            {"s": s, "mult": _s(k, "mult", "lever")},
        ),
    ]


def _x_rows() -> list[Template]:
    i = "intervention"
    end = _s(i, "end")
    return [
        _t(
            "X-regen",
            "block",
            "It ended at step {end}; food regrowth returned to {mult} times its base rate.",
            {"end": end, "mult": _s(i, "restored.mult", "lever")},
        ),
        _t(
            "X-regen-1",
            "block",
            "It ended at step {end}; food regrowth returned to its base rate.",
            {"end": end},
        ),
        _t(
            "X-gain",
            "block",
            "It ended at step {end}; the fertility of patch {patch} returned to {gain} times its base value.",
            {"end": end, "patch": _s(i, "restored.patch", "id"), "gain": _s(i, "restored.gain", "lever")},
        ),
        _t(
            "X-gain-1",
            "block",
            "It ended at step {end}; the fertility of patch {patch} returned to its base value.",
            {"end": end, "patch": _s(i, "restored.patch", "id")},
        ),
        _t("X-replaced", "block", "It was replaced at step {end}.", {"end": end}),
        _t("X-open", "block", "It was still in place at step {b}.", {"b": _s(i, "open_to")}),
    ]


def _ticker_rows(rows: list[Template]) -> list[Template]:
    """``T:<id>`` for each intervention line and each thread sentence: "Step {t}: " + the sentence, with ``t``
    the step of the snapshot the line belongs to; and for the two storm sentences that carry their own step
    (``T:ST-named``, ``T:ST-end``) a line dated by the storm's step."""
    out = []
    for r in rows:
        if (r.id.startswith("I-") and r.id != "I-more") or r.id.startswith("KT-"):
            out.append(
                Template(
                    "T:" + r.id,
                    "ticker",
                    "Step {t}: " + r.text,
                    {"t": _s("snapshot", "step"), **r.slots},
                    r.agree,
                )
            )
    st = "storm"
    out.append(
        Template(
            "T:ST-named",
            "ticker",
            "Step {t}: Storm {name} reached named strength.",
            {"t": _s(st, "named_step"), "name": _s(st, "name", "name")},
            {},
        )
    )
    out.append(
        Template(
            "T:ST-end",
            "ticker",
            "Step {t}: Storm {name} dissolved after {age} {step_n}.",
            {"t": _s(st, "end"), "name": _s(st, "name", "name"), "age": _s(st, "age")},
            {"step_n": ("age", "step")},
        )
    )
    return out


def _catalogue() -> dict[str, Template]:
    pop = _POP
    rows = [
        # --- the lead
        _t(
            "P-up",
            "lead",
            "The population rose from {from} to {to} ({change}).",
            {"from": _s(pop, "from"), "to": _s(pop, "to"), "change": _s(pop, "change", "change")},
        ),
        _t(
            "P-down",
            "lead",
            "The population fell from {from} to {to} ({change}).",
            {"from": _s(pop, "from"), "to": _s(pop, "to"), "change": _s(pop, "change", "change")},
        ),
        _t(
            "P-flat",
            "lead",
            "The population was {to} at the start and at the end of the chapter.",
            {"to": _s(pop, "to")},
        ),
        _t(
            "P-near",
            "lead",
            "The population was {from} at the start and {to} at the end of the chapter.",
            {"from": _s(pop, "from"), "to": _s(pop, "to")},
        ),
        _t(
            "P-zero",
            "lead",
            "The population rose from {from} to {to}.",
            {"from": _s(pop, "from"), "to": _s(pop, "to")},
        ),
        # --- the body
        _t(
            "P-low",
            "body",
            "The lowest count was {min}, in the {min_steps} {step_n} to step {min_row}.",
            {"min": _s(pop, "min"), "min_steps": _s(pop, "min_steps"), "min_row": _s(pop, "min_row")},
            {"step_n": ("min_steps", "step")},
        ),
        _t(
            "TO",
            "body",
            "{born} {organism_n} {be} born and {died} died: a turnover of {k} times the population.",
            {
                "born": _s("turnover", "born"),
                "died": _s("turnover", "died"),
                "k": _s("turnover", "k", "dec1"),
            },
            {"organism_n": ("born", "organism"), "be": ("born", "be")},
        ),
        _t(
            "DC",
            "body",
            "{n} {organism_n} died by catastrophe.",
            {"n": _s("deaths", "catastrophe")},
            {"organism_n": ("n", "organism")},
        ),
        _t(
            "DS",
            "body",
            "{n} starved in the {steps} {step_n} to step {s}, against a median of {median}.",
            {
                "n": _s("starve", "n"),
                "steps": _s("starve", "steps"),
                "s": _s("starve", "s"),
                "median": _s("starve", "median"),
            },
            {"step_n": ("steps", "step")},
        ),
        _t(
            "ST-d",
            "body",
            "{n} {organism_n} died in storms in the {steps} {step_n} to step {s}.",
            {
                "n": _s("stormdeaths", "n"),
                "steps": _s("stormdeaths", "steps"),
                "s": _s("stormdeaths", "s"),
            },
            {"organism_n": ("n", "organism"), "step_n": ("steps", "step")},
        ),
        _t(
            "ST-n",
            "body",
            "{tracked} {storm_n} {be} tracked at the end of the chapter, {named} of them named.",
            {"tracked": _s("storms", "tracked"), "named": _s("storms", "named")},
            {"storm_n": ("tracked", "storm"), "be": ("tracked", "be")},
        ),
        _t(
            "ST-b",
            "body",
            "{formed} {storm_n} formed and {dissolved} dissolved during the chapter.",
            {"formed": _s("storms", "formed"), "dissolved": _s("storms", "dissolved")},
            {"storm_n": ("formed", "storm")},
        ),
        _t(
            "ST-named",
            "body",
            "Storm {name} reached named strength at step {s}.",
            {"name": _s("storm", "name", "name"), "s": _s("storm", "named_step")},
        ),
        _t(
            "ST-peak",
            "body",
            "Storm {name}, in the {hemisphere} hemisphere, peaked at {ratio} times the naming threshold.",
            {
                "name": _s("storm", "name", "name"),
                "hemisphere": _s("storm", "hemisphere", "text"),
                "ratio": _s("storm", "ratio", "dec2"),
            },
        ),
        _t(
            "ST-end",
            "body",
            "Storm {name} dissolved at step {s} after {age} {step_n}.",
            {"name": _s("storm", "name", "name"), "s": _s("storm", "end"), "age": _s("storm", "age")},
            {"step_n": ("age", "step")},
        ),
        _t(
            "B-1",
            "body",
            "There {be} {births} {birth_n}.",
            {"births": _s("births", "births")},
            {"be": ("births", "be"), "birth_n": ("births", "birth")},
        ),
        _t(
            "B-2",
            "body",
            "There {be} {births} {birth_n} and {immigrants} {arrival_n} from outside.",
            {"births": _s("births", "births"), "immigrants": _s("births", "immigrants")},
            {"be": ("births", "be"), "birth_n": ("births", "birth"), "arrival_n": ("immigrants", "arrival")},
        ),
        _t(
            "LN",
            "body",
            "The number of lineages with living members went from {from} to {to}.",
            {"from": _s("lineages", "from"), "to": _s("lineages", "to")},
        ),
        _t(
            "TL",
            "body",
            "At step {a} the largest lineage held {from_share} of the agents, and at step {b} the "
            "largest held "
            "{to_share}.",
            {
                "a": _CH_A,
                "b": _CH_B,
                "from_share": _s("top_lineage", "from_share", "pct"),
                "to_share": _s("top_lineage", "to_share", "pct"),
            },
        ),
        *[
            _t(
                f"G-r{n}",
                "body",
                "{K} was the "
                + word
                + " kin group at step {b}, with {size} {member_n}, {share} of the population.",
                _GROUP_SLOTS,
                _MEMBER,
            )
            for n, word in enumerate(RANK_WORDS, start=1)
        ],
        _t(
            "G-s",
            "body",
            "{K} had {size} {member_n} at step {b}, {share} of the population.",
            _GROUP_SLOTS,
            _MEMBER,
        ),
        _t(
            "G-c",
            "body",
            "In the chapter, {born} {agent_n} {be} born into {K} and {died} of its members died, "
            "{died_killed} of "
            "them by attack.",
            {
                "K": _s("group", "entity", "kin"),
                "born": _s("group", "born"),
                "died": _s("group", "died"),
                "died_killed": _s("group", "died_killed"),
            },
            {"agent_n": ("born", "agent"), "be": ("born", "be")},
        ),
        _t(
            "KT-formed",
            "body",
            "{name} has held together since step {s0}: {size} {member_n}.",
            _kt_slots("name", "s0", "size"),
            _MEMBER,
        ),
        _t(
            "KT-formed-from",
            "body",
            "{name} has held together since step {s0} after splitting from {other}: {size} {member_n}.",
            _kt_slots("name", "s0", "other", "size"),
            _MEMBER,
        ),
        _t(
            "KT-ended-into",
            "body",
            "{name} merged into {other} at step {s}; it lasted {steps} {step_n} and had up to {peak} "
            "{member_n}.",
            _kt_slots("name", "other", "s", "steps", "peak"),
            {"step_n": ("steps", "step"), "member_n": ("peak", "member")},
        ),
        _t(
            "KT-ended-into-i",
            "body",
            "{name} merged into {other} at step {s}; it had up to {peak} {member_n}.",
            _kt_slots("name", "other", "s", "peak"),
            {"member_n": ("peak", "member")},
        ),
        _t(
            "KT-ended",
            "body",
            "{name} dissolved after {steps} {step_n}, up to {peak} {member_n}.",
            _kt_slots("name", "steps", "peak"),
            {"step_n": ("steps", "step"), "member_n": ("peak", "member")},
        ),
        _t(
            "KT-ended-i",
            "body",
            "{name} dissolved, up to {peak} {member_n}.",
            _kt_slots("name", "peak"),
            {"member_n": ("peak", "member")},
        ),
        _t(
            "KS",
            "body",
            "{n} further kin group {change_n} did not last long enough to be listed.",
            {"n": _s("untold", "n")},
            {"change_n": ("n", "change")},
        ),
        _t(
            "CE-born",
            "body",
            "Between step {s0} and step {s1}, a colony with {size} {member_n} formed.",
            _ev_slots(),
            _MEMBER,
        ),
        _t(
            "CE-split",
            "body",
            "Between step {s0} and step {s1}, a colony with {size} {member_n} split off another.",
            _ev_slots(),
            _MEMBER,
        ),
        _t(
            "CE-merged",
            "body",
            "Between step {s0} and step {s1}, a colony that had {size} {member_n} at step {s0} "
            "merged into another.",
            _ev_slots(),
            _MEMBER,
        ),
        _t(
            "CE-dissolved",
            "body",
            "Between step {s0} and step {s1}, a colony that had {size} {member_n} at step {s0} dissolved.",
            _ev_slots(),
            _MEMBER,
        ),
        *_house_rows(),
        _t(
            "H-2",
            "body",
            "{H} was the largest house at step {b}, with {largest_size} {member_n}.",
            {"H": _s("houses", "largest", "house"), "b": _CH_B, "largest_size": _s("houses", "largest_size")},
            {"member_n": ("largest_size", "member")},
        ),
        _t(
            "LB-a",
            "body",
            "{from_share} of the agents with an archetype label matched {label} at step {a}, and "
            "{to_share} at "
            "step {b}.",
            {
                "from_share": _s("label", "from_share", "pct"),
                "to_share": _s("label", "to_share", "pct"),
                "label": _s("label", "label", "label"),
                "a": _CH_A,
                "b": _CH_B,
            },
        ),
        _t(
            "LB-i",
            "body",
            "{from_share} of the agents with a behaviour label were labelled {label} at step {a}, "
            "and {to_share} "
            "at step {b}.",
            {
                "from_share": _s("label", "from_share", "pct"),
                "to_share": _s("label", "to_share", "pct"),
                "label": _s("label", "label", "label"),
                "a": _CH_A,
                "b": _CH_B,
            },
        ),
        _t(
            "R",
            "body",
            "Members of {A} attacked members of {B} {attacks} times; {kills} of the attacks {be} on a victim "
            "that died in the same step.",
            {
                "A": _s("relation", "a", "kin"),
                "B": _s("relation", "b", "kin"),
                "attacks": _s("relation", "attacks"),
                "kills": _s("relation", "kills"),
            },
            {"be": ("kills", "be")},
        ),
        _t(
            "C",
            "body",
            "{tracked} {colony_n} {be} tracked at step {b}; the largest had {largest_size} {member_n}.",
            {
                "tracked": _s("colonies", "tracked"),
                "b": _CH_B,
                "largest_size": _s("colonies", "largest_size"),
            },
            {
                "colony_n": ("tracked", "colony"),
                "be": ("tracked", "be"),
                "member_n": ("largest_size", "member"),
            },
        ),
        _t(
            "C-1",
            "body",
            "{tracked} {colony_n} {be} tracked at step {b}; it had {largest_size} {member_n}.",
            {
                "tracked": _s("colonies", "tracked"),
                "b": _CH_B,
                "largest_size": _s("colonies", "largest_size"),
            },
            {
                "colony_n": ("tracked", "colony"),
                "be": ("tracked", "be"),
                "member_n": ("largest_size", "member"),
            },
        ),
        _t(
            "END",
            "body",
            "The run stopped at step {step} with {alive} {agent_n} alive.",
            {"step": _s("run_end", "step"), "alive": _s("run_end", "alive")},
            {"agent_n": ("alive", "agent")},
        ),
        # --- the headlines
        _t("HP-up", "headline", "Population rose to {to}", {"to": _s(pop, "to")}),
        _t("HP-down", "headline", "Population fell to {to}", {"to": _s(pop, "to")}),
        _t("HP-flat", "headline", "Population unchanged at {to}", {"to": _s(pop, "to")}),
        _t("HP-near", "headline", "Population at {to}", {"to": _s(pop, "to")}),
        _t("HP-dip", "headline", "Population dipped to {min}", {"min": _s(pop, "min")}),
        _t("HK-formed", "headline", "{name} formed", _kt_slots("name")),
        _t("HK-ended-into", "headline", "{name} merged into {other}", _kt_slots("name", "other")),
        _t("HK-ended", "headline", "{name} dissolved", _kt_slots("name")),
        _t(
            "HS",
            "headline",
            "{n} {storm_n} reached named strength",
            {"n": _s("storms", "named_new")},
            {"storm_n": ("n", "storm")},
        ),
        _t("HQ", "headline", "A quiet chapter", {}),
        _t(
            "HH",
            "headline",
            "{extinct} {house_n} died out",
            {"extinct": _s("houses", "extinct")},
            {"house_n": ("extinct", "house")},
        ),
        _t(
            "HL",
            "headline",
            "Lineages went from {from} to {to}",
            {"from": _s("lineages", "from"), "to": _s("lineages", "to")},
        ),
        # --- the interventions block
        *_s_rows(),
        *_i_rows(),
        _t(
            "I-more",
            "block",
            "{n} further {setting_n} {is_n} in place and {is_n} not listed.",
            {"n": _s("excluded", "interventions_dropped")},
            {"setting_n": ("n", "setting"), "is_n": ("n", "is")},
        ),
        *_x_rows(),
        _t(
            "W",
            "block",
            "From {from} at step {r0} to {to} at step {r1}, with the change applied at step {w0}.",
            {
                "from": _s("window", "from"),
                "r0": _s("window", "r0"),
                "to": _s("window", "to"),
                "r1": _s("window", "r1"),
                "w0": _s("window", "w0"),
            },
        ),
    ]
    rows += _ticker_rows(rows)
    table = {r.id: r for r in rows}
    if len(table) != len(rows):
        raise ValueError("templates: duplicate id")
    return table


TEMPLATES: Final[Mapping[str, Template]] = _catalogue()
BLOCK_ROLES: Final = ("block", "ticker")  # the roles in which lever words and rule identifiers are allowed

# --- section order and salience order -----------------------------------------------------------------------

# The sections of a body, in the order they are displayed whatever the mode (docs/DESIGN.md §31, rev 3 F5):
# the order between sections is code's, so no sentence can be placed next to another by a model. Inside a
# section: the turnover before the catastrophe deaths; the storm sentences by (kind, step, id) with the counts
# first; thread sentences by (told step, -size, id); groups by (rank or -size, id), each followed by its G-c;
# labels by (-|change in share|, label); relations by (-attacks, a, b); colony events by (end step, -size).
SECTION_ORDER: Final = (
    "D",
    "P-low",
    "DS",
    "ST",
    "E",
    "KS",
    "G",
    "B",
    "LN",
    "TL",
    "H",
    "LB",
    "R",
    "CE",
    "C",
    "END",
)
SECTION_OF: Final[Mapping[str, str]] = {
    **{t: "D" for t in ("TO", "DC")},
    "P-low": "P-low",
    "DS": "DS",
    **{t: "ST" for t in ("ST-n", "ST-b", "ST-named", "ST-peak", "ST-end", "ST-d")},
    **{t: "E" for t in TEMPLATES if t.startswith("KT-")},
    "KS": "KS",
    **{t: "G" for t in ("G-r1", "G-r2", "G-r3", "G-s", "G-c")},
    **{t: "B" for t in ("B-1", "B-2")},
    "LN": "LN",
    "TL": "TL",
    **{t: "H" for t in TEMPLATES if t.startswith("H-")},
    **{t: "LB" for t in ("LB-a", "LB-i")},
    "R": "R",
    **{t: "CE" for t in TEMPLATES if t.startswith("CE-")},
    **{t: "C" for t in ("C", "C-1")},
    "END": "END",
}
# The order in which candidates are taken for a body (template mode takes the first k; ids C1..Cn follow it).
# "EV" is the thread sentences and the storm sentences of naming and ending, taken in turn (the largest
# thread event, the strongest storm, the next thread event, ...), so a chapter with many of one kind still
# says the other.
SALIENCE_ORDER: Final = (
    "D",
    "P-low",
    "DS",
    "ST-n",
    "ST-b",
    "EV",
    "ST-peak",
    "ST-d",
    "G-top",
    "B",
    "LN",
    "TL",
    "H",
    "LB",
    "R",
    "G-rest",
    "CE",
    "KS",
    "C",
    "END",
)

# --- rendering ---------------------------------------------------------------------------------------------


def fmt_int(n: int) -> str:
    return f"{int(n):,}"


def fmt_pct(x: float) -> str:
    return f"{float(x) * 100:.1f}%"


def fmt_change(x: float) -> str:
    return f"{float(x) * 100:+.1f}%"


def fmt_lever(x: float) -> str:
    """At most two decimals, trailing zeros stripped; a float32 such as 0.6000000238 renders "0.6"."""
    s = f"{float(x):.2f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def fmt_dec(x: float, places: int) -> str:
    """``x`` with exactly ``places`` decimals (``6.0``, ``1.01``)."""
    return f"{float(x):.{places}f}"


_TOKEN = re.compile(r"\{([A-Za-z_0-9]+)\}")
_WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")


def tokens_of(text: str) -> list[str]:
    """The ``{token}`` names of a template text, in order (repeats included)."""
    return _TOKEN.findall(text)


def format_value(slot: Slot, value: Any) -> str:
    """The text of ``value`` in the format of ``slot`` (a display string passes through)."""
    f = slot.fmt
    if f == "int":
        return fmt_int(value)
    if f == "id":
        return str(int(value))
    if f == "pct":
        return fmt_pct(value)
    if f == "change":
        return fmt_change(value)
    if f == "lever":
        return fmt_lever(value)
    if f == "dec1":
        return fmt_dec(value, 1)
    if f == "dec2":
        return fmt_dec(value, 2)
    return str(value)  # kin | house | label | name | rule | text: already a display string


def render(tid: str, values: Mapping[str, Any]) -> str:
    """The text of template ``tid`` for the raw slot ``values`` (a number for the numeric formats, a display
    string for ``kin``, ``house``, ``label`` and ``text``). Agreement markers follow their count slot (the
    singular form at exactly 1); the first letter is capitalised."""
    t = TEMPLATES[tid]

    def sub(m: re.Match[str]) -> str:
        name = m.group(1)
        if name in t.slots:
            return format_value(t.slots[name], values[name])
        count, key = t.agree[name]
        one, many = AGREE_WORDS[key]
        return one if int(values[count]) == 1 else many

    out = _TOKEN.sub(sub, t.text)
    return out[:1].upper() + out[1:]


def template_words(text: str) -> set[str]:
    """The lower-cased words of a template text with its tokens removed."""
    return set(_WORD.findall(_TOKEN.sub(" ", text).lower()))


def _vocab() -> frozenset[str]:
    words: set[str] = set(DISPLAY_WORDS)
    for t in TEMPLATES.values():
        words |= template_words(t.text)
    for one, many in AGREE_WORDS.values():
        words |= {one, many}
    for r in RANK_WORDS:
        words |= set(r.split())
    words |= set(HEMISPHERE_WORDS)
    for text in SOURCE_WORDS.values():
        words |= template_words(text)
    return frozenset(words)


TEMPLATE_VOCAB: Final = _vocab()  # the closed vocabulary of R3

# --- import-time checks -------------------------------------------------------------------------------------

_BEFORE_OK = " ("
_AFTER_OK = " ,.;:)"


def validate(tpls: Mapping[str, Template]) -> None:
    """The import-time checks on a catalogue (see the module doc); ``ValueError`` on the first violation."""
    roles = {"lead", "body", "headline", "block", "ticker"}
    for t in tpls.values():
        if t.role not in roles:
            raise ValueError(f"template {t.id}: unknown role {t.role!r}")
        if any(c.isdigit() for c in _TOKEN.sub("", t.text)):
            raise ValueError(f"template {t.id}: a literal digit")
        for m in _TOKEN.finditer(t.text):
            before = t.text[m.start() - 1] if m.start() else None
            after = t.text[m.end()] if m.end() < len(t.text) else None
            if before is not None and before not in _BEFORE_OK:
                raise ValueError(f"template {t.id}: {m.group(0)} is glued to {before!r}")
            if after is not None and after not in _AFTER_OK:
                raise ValueError(f"template {t.id}: {m.group(0)} is glued to {after!r}")
            if m.group(1) not in t.slots and m.group(1) not in t.agree:
                raise ValueError(f"template {t.id}: undeclared token {m.group(0)}")
        used = set(tokens_of(t.text))
        if used != set(t.slots) | set(t.agree):
            raise ValueError(f"template {t.id}: declared and used tokens differ")
        for marker, (count, key) in t.agree.items():
            if count not in t.slots or key not in AGREE_WORDS or t.slots[count].fmt != "int":
                raise ValueError(f"template {t.id}: marker {marker} names {count!r}/{key!r}")
    for text in (*SOURCE_WORDS.values(), *RANK_WORDS):
        if any(c.isdigit() for c in text):
            raise ValueError("source words or rank words with a digit")


validate(TEMPLATES)


def _digest() -> str:
    body = {
        "version": TEMPLATE_VERSION,
        "templates": {
            tid: {
                "role": t.role,
                "text": t.text,
                "slots": {k: list(v) for k, v in t.slots.items()},
                "agree": {k: list(v) for k, v in t.agree.items()},
            }
            for tid, t in TEMPLATES.items()
        },
        "agree_words": {k: list(v) for k, v in AGREE_WORDS.items()},
        "rank_words": list(RANK_WORDS),
        "hemisphere_words": list(HEMISPHERE_WORDS),
        "display_words": list(DISPLAY_WORDS),
        "rule_names": sorted(RULE_NAMES),
        "source_words": dict(SOURCE_WORDS),
        "section_order": list(SECTION_ORDER),
        "section_of": dict(SECTION_OF),
        "salience_order": list(SALIENCE_ORDER),
        "formats": ["thousands", "pct1", "signed_pct1", "decimals2_stripped", "dec1", "dec2"],
    }
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


TEMPLATE_DIGEST: Final = _digest()
