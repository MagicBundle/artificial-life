"""Scripted fact sheets for the world report's tests (docs/DESIGN.md §31): ``mk_sheet`` builds a sheet from the
parts a test cares about with ``report.compose_sheet`` (the function ``build_sheet`` ends in), and
``random_sheet`` draws one from a seeded ``random.Random`` that satisfies the invariants of the real sheets
(rank prefixes, event frames, the lever state of the block), so every template is reached."""

# ruff: noqa: E501  (expected report texts and fixture tables are long lines)

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any

from alife.game import report as R

A0, B0 = 1_512_000, 1_518_000
V1_DIGEST = "156ad4963b84ffe9636e9ba4fdd7592632c669bf6e490687d8eb2b38b9aec009"
W2D = "69b4e21ad6b6016b17c11d6a1d9b3371c03b4b9f590fdb4a04752515c13a01d8"
W3D = "df54434fe947dea8f36e5e23223f9676df1ee784959806c31eafb235012ee242"
NAMES = [
    "Braelix",
    "Shaemund",
    "Torvane",
    "Kraiost",
    "Flauth",
    "Clevask",
    "Gloarn",
    "Steiax",
    "Phouth",
    "Dreanor",
]

EXCLUDED = {
    "labels": {},
    "social": None,
    "events_churn": 0,
    "interventions_dropped": 0,
    "windows_confounded": 0,
    "quota": {},
    "entries_before_start": 0,
    "facts_dropped": [],
}


def pop(
    frm: int, to: int, mn: int | None = None, mn_row: int | None = None, rules: R.ReportRules = R.REPORT_V1
):
    f: dict[str, Any] = {"kind": "population", "from": frm, "to": to}
    if frm > 0:
        f["change"] = (to - frm) / frm
    if mn is not None:
        f.update({"min": mn, "min_row": mn_row or B0 - 100, "min_steps": 100})
    f["deep_dip"] = mn is not None and mn < rules.dip_mandatory * min(frm, to)
    return f


def deaths(starved=100, killed=20, aged=5, catastrophe=0):
    return {
        "kind": "deaths",
        "starved": starved,
        "killed": killed,
        "aged": aged,
        "catastrophe": catastrophe,
        "total": starved + killed + aged + catastrophe,
    }


def group(
    entity,
    size,
    of,
    rank=None,
    born=None,
    died=None,
    killed=None,
    thread=None,
    prev_thread=None,
    prev_share=None,
):
    f: dict[str, Any] = {"kind": "group", "entity": entity, "size": size, "of": of, "share": size / of}
    if rank is not None:
        f["rank"] = rank
    if rank == 1:
        f.update({"thread": thread, "prev_thread": prev_thread, "prev_share": prev_share})
    if born is not None:
        f.update({"born": born, "died": died, "died_killed": killed})
    return f


def event(kind, system, size, other=-1, overlap=0, s0=A0 + 250, s1=A0 + 500, family="colony", reid=False):
    return {
        "kind": "event",
        "family": family,
        "event": kind,
        "system": system,
        "other": other,
        "size": size,
        "overlap": overlap,
        "s0": s0,
        "s1": s1,
        "reid": reid,
    }


def thread(
    event_kind,
    tid,
    name,
    size,
    *,
    other=None,
    first=A0 + 250,
    told=A0 + 500,
    end=None,
    steps=None,
    peak=None,
    imp=1,
    initial=False,
    kin=None,
):
    """A thread fact (a kin group through its id changes): ``formed`` (from ``other``, a name) or ``ended`` (into
    ``other``, or dissolved). An ended thread that was not there from the start has its lifetime."""
    ended = event_kind == "ended"
    if ended and end is None:
        end = told - 250
    if ended and steps is None and not initial:
        steps = end - first
    return {
        "kind": "thread",
        "event": event_kind,
        "thread": tid,
        "name": name,
        "other": other,
        "first_step": first,
        "end": end if ended else None,
        "steps": steps if ended else None,
        "size": size,
        "peak": size if peak is None else peak,
        "told": told,
        "imp": imp,
        "initial": initial,
        "kin": tid if kin is None else kin,
        "x": 100.0,
        "y": 50.0,
    }


def storms(tracked=6, named=4, formed=3, dissolved=1, named_new=2, longest=9000):
    return {
        "kind": "storms",
        "tracked": tracked,
        "named": named,
        "formed": formed,
        "dissolved": dissolved,
        "named_new": named_new,
        "longest": longest,
    }


def storm(sid, name, *, named=A0 + 1600, end=None, age=None, ratio=1.2, hemisphere="northern"):
    return {
        "kind": "storm",
        "storm": sid,
        "name": name,
        "hemisphere": hemisphere,
        "ratio": ratio,
        "named_step": named,
        "end": end,
        "age": age,
        "x": 12.5,
        "y": 80.25,
    }


def untold(n=45):
    return {"kind": "untold", "n": n}


def turnover(born, died, frm, to):
    return {"kind": "turnover", "born": born, "died": died, "k": died / ((frm + to) / 2)}


def houses(
    alive_from=14,
    alive_to=12,
    first_seen=0,
    extinct=2,
    exact=True,
    origin=0,
    largest=3,
    size=6700,
    strict=True,
    extinct_top=None,
):
    f = {
        "kind": "houses",
        "alive_from": alive_from,
        "alive_to": alive_to,
        "first_seen": first_seen,
        "extinct": extinct,
        "extinct_top": extinct if extinct_top is None else extinct_top,
        "exact": exact,
        "origin": origin,
    }
    if largest is not None:
        f.update(
            {"largest": largest, "largest_size": size, "largest_strict": strict, "largest_name": "Zaeld"}
        )
    return f


def interv(
    entry,
    action,
    step,
    params,
    source="rule",
    rule="runaway_damper",
    duration=None,
    end=None,
    end_by=None,
    restored=None,
):
    from alife.game import templates

    return {
        "kind": "intervention",
        "entry": entry,
        "action": action,
        "source": source,
        "rule": rule,
        "src": templates.source_words(source, rule),
        "step": step,
        "params": params,
        "duration": duration,
        "end": end,
        "end_by": end_by,
        "restored": restored,
        "open_to": B0 if end is None and action not in R.POINTS else None,
    }


def mk_sheet(
    *,
    a: int = A0,
    b: int = B0,
    kinds: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    ents: Sequence[tuple[str, int, str | None]] = (),
    excluded: Mapping[str, Any] | None = None,
    snapshots: Sequence[Mapping[str, Any]] | None = None,
    dims: int = 2,
    labels: Mapping[str, Any] | None = None,
    rules: R.ReportRules = R.REPORT_V1,
) -> dict[str, Any]:
    """A sheet of the chapter ``(a, b]`` from ``kinds`` (fact dicts by kind; the chapter fact is added) and the
    entities ``(kind, id, name)`` (``kin_group`` or ``house``)."""
    k = {"chapter": [{"kind": "chapter", "start": a, "end": b}], **(kinds or {})}
    header = {
        "run": {"seed": 0},
        "config_hash": "c" * 64,
        "world_key": W2D if dims == 2 else W3D,
        "dims": dims,
        "chapter": [a, b],
        "origin": 0,
        "start_step": 0,
        "labels": dict(
            labels or {"intent_version": 1, "rules_digest": V1_DIGEST, "key_found": True, "env_default": True}
        ),
    }
    entities = [{"kind": kd, "id": i, "name": nm, "name_from": b if nm else None} for kd, i, nm in ents]
    if snapshots is None:
        snapshots = [
            {"id": f"S{s}", "step": s, "alive": 1000 + (s % 1000)} for s in range(a + 250, b + 1, 250)
        ]
    return R.compose_sheet(
        header=header,
        kinds=k,
        entities=entities,
        excluded={**EXCLUDED, **(excluded or {})},
        snapshots=snapshots,
        rules=rules,
    )


def full_sheet() -> dict[str, Any]:
    """A sheet that reaches most templates: a falling population with a deep dip, a catastrophe, arrivals,
    lineages, groups with ranks, events of every kind, houses, a label, a relation, colonies, the end of the run,
    an interventions block with every action and a standing lever."""
    kin = [(1, "Braelix"), (2, "Shaemund"), (3, "Torvane"), (4, None), (5, "Flauth")]
    return mk_sheet(
        kinds={
            "population": [pop(34_410, 21_140, 8_900)],
            "deaths": [deaths(36_050, 5_060, 190, 9_400)],
            "births": [{"kind": "births", "births": 28_030, "immigrants": 7}],
            "turnover": [turnover(28_030, 50_700, 34_410, 21_140)],
            "starve": [{"kind": "starve", "n": 8_249, "s": B0 - 1_500, "steps": 250, "median": 5_676}],
            "stormdeaths": [{"kind": "stormdeaths", "n": 14, "s": B0 - 750, "steps": 250}],
            "lineages": [{"kind": "lineages", "from": 9, "to": 7}],
            "top_lineage": [
                {
                    "kind": "top_lineage",
                    "from_share": 0.418,
                    "to_share": 0.471,
                    "from_count": 14_383,
                    "to_count": 9_957,
                    "from_of": 34_410,
                    "to_of": 21_140,
                }
            ],
            "intervention": [
                interv(
                    0,
                    "regen_pulse",
                    A0 + 500,
                    {"mult": 0.800000011920929},
                    duration=10_000,
                    end=A0 + 1500,
                    end_by="elapsed",
                    restored={"mult": 1.0},
                ),
                interv(
                    1,
                    "habitat_loss",
                    A0 + 600,
                    {"patch": 3, "gain": 0.0},
                    source="llm",
                    rule=None,
                    duration=10_000,
                    end=A0 + 2000,
                    end_by="elapsed",
                    restored={"patch": 3, "gain": 1.75},
                ),
                interv(
                    2,
                    "set_seasons",
                    A0 + 700,
                    {"period": 4000, "amp": 0.6000000238418579},
                    source="llm",
                    rule=None,
                    end=A0 + 2500,
                    end_by="replaced",
                ),
                interv(3, "drift_change", A0 + 800, {"mult": 4.0}, source="llm", rule=None),
                interv(4, "fertility_shift", A0 + 900, {"patch": 5, "gain": 0.25}, source="llm", rule=None),
                interv(
                    5,
                    "fertility_shift",
                    A0 + 1000,
                    {"patch": 6, "x": 300.5, "y": 120.25},
                    source="llm",
                    rule=None,
                ),
                interv(6, "immigration_pulse", A0 + 1100, {"count": 64}, rule="extinction_rescue"),
                interv(
                    7,
                    "catastrophe",
                    A0 + 1200,
                    {"center": [120.0, 88.0], "radius": 40.0, "kill_p": 0.6000000238418579},
                    source="llm",
                    rule=None,
                ),
            ],
            "standing": [
                {
                    "kind": "standing",
                    "lever": "patch:12:gain",
                    "step": 295_000,
                    "entry": 9,
                    "patch": 12,
                    "gain": 1.75,
                }
            ],
            "group": [
                group(1, 5020, 21_140, 1, 3100, 4900, 610),
                group(2, 1830, 21_140, 2, 700, 80, 10),
                group(5, 900, 21_140),
            ],
            "thread": [
                thread("formed", 2, "Shaemund", 1210, other="Braelix", imp=2),
                thread("ended", 3, "Torvane", 980, other="Braelix", peak=1100, imp=2),
                thread("formed", 4, "Kraiost", 640),
                thread("ended", 5, "Flauth", 520, other="Braelix", initial=True),
                thread("ended", 7, "Dreanor", 300, initial=True),
                thread(
                    "ended", 6, "Stiorn", 400, steps=3_000, first=A0 - 3_000, end=A0 + 2_000, told=A0 + 2_250
                ),
            ],
            "untold": [untold(45)],
            "storms": [storms()],
            "storm": [
                storm(3, "Phaitraern", named=A0 + 1_600, end=A0 + 5_000, age=8_725, ratio=1.4),
                storm(4, "Guadouck", named=A0 + 2_000, ratio=1.1, hemisphere="southern"),
            ],
            "event": [
                event("born", 6, 300, 2, 200, family="colony"),
                event("born", 7, 300, family="colony"),
            ],
            "houses": [houses(first_seen=3)],
            "label": [
                {
                    "kind": "label",
                    "label": "Hunter",
                    "key": "hunter",
                    "family": "archetype",
                    "from_share": 0.085,
                    "to_share": 0.12,
                    "from_count": 2925,
                    "to_count": 2558,
                    "from_of": 34_410,
                    "to_of": 21_140,
                }
            ],
            "relation": [{"kind": "relation", "a": 1, "b": 2, "attacks": 412, "kills": 61}],
            "colonies": [{"kind": "colonies", "tracked": 7, "largest_size": 40}],
            "run_end": [{"kind": "run_end", "step": B0 + 230, "alive": 0}],
        },
        ents=[("kin_group", i, nm) for i, nm in kin] + [("house", 3, "Zaeld")],
    )


# --- random sheets ---------------------------------------------------------------------------------------------


def _random_interventions(rng: random.Random, a: int, b: int) -> tuple[list[dict], list[dict]]:
    ivs, standing = [], []
    for j in range(rng.randint(0, 6)):
        s = rng.choice([a - 2000, a - 500, a, a + 250, a + 500, a + 1500])
        action = rng.choice(
            [
                "regen_pulse",
                "habitat_loss",
                "set_seasons",
                "drift_change",
                "fertility_shift",
                "fertility_shift",
                "immigration_pulse",
                "catastrophe",
            ]
        )
        source = rng.choice(["rule", "llm"])
        rule = rng.choice([*sorted(R.templates.RULE_NAMES), None]) if source == "rule" else None
        bloom = action == "fertility_shift" and rng.random() < 0.4  # a fertility gain with a duration
        timed = action in ("regen_pulse", "habitat_loss") or bloom
        end = end_by = restored = None
        if action not in R.POINTS:
            end_by = rng.choice(["elapsed", "replaced", None]) if timed else rng.choice(["replaced", None])
            if end_by:
                end = s + rng.choice([500, 1000, 2000])
            if end_by == "elapsed":
                restored = (
                    {"mult": rng.choice([1.0, 1.25])}
                    if action == "regen_pulse"
                    else {"patch": 3, "gain": rng.choice([1.0, 1.75])}
                )
        params = {
            "regen_pulse": {"mult": rng.choice([0.4, 0.8, 1.5, 1.0])},
            "habitat_loss": {"patch": 3, "gain": 0.0},
            "set_seasons": {"period": rng.choice([2000, 45_000]), "amp": rng.choice([0.0, 0.6])},
            "drift_change": {"mult": rng.choice([0.5, 4.0])},
            "immigration_pulse": {"count": rng.choice([16, 64, 1024])},
            "catastrophe": {
                "center": [rng.uniform(0, 1000), rng.uniform(0, 1000)],
                "radius": 40.0,
                "kill_p": rng.choice([0.2, 0.6]),
            },
            "fertility_shift": (
                {"patch": 5, "gain": rng.choice([1.5, 2.0])}
                if bloom
                else rng.choice(
                    [{"patch": 5, "gain": rng.choice([0.25, 1.75])}, {"patch": 6, "x": 300.5, "y": 120.25}]
                )
            ),
        }[action]
        ivs.append(
            interv(j, action, s, params, source, rule, 10_000 if timed else None, end, end_by, restored)
        )
    if rng.random() < 0.4:
        standing.append(
            {
                "kind": "standing",
                "lever": "patch:12:gain",
                "step": a - 3000,
                "entry": 20,
                "patch": 12,
                "gain": rng.choice([1.75, 0.25]),
            }
        )
    if rng.random() < 0.2:
        standing.append(
            {
                "kind": "standing",
                "lever": "seasons",
                "step": a - 3000,
                "entry": 21,
                "period": 4000,
                "amp": 0.6,
            }
        )
    levers = (
        ("regen", {"mult": 0.5}),
        ("drift", {"mult": 4.0}),
    )
    for lever, extra in levers:
        if rng.random() < 0.15:
            standing.append({"kind": "standing", "lever": lever, "step": a - 2000, "entry": 22, **extra})
    ivs.sort(key=lambda f: (f["step"], f["entry"]))
    return ivs, standing


def random_sheet(rng: random.Random, rules: R.ReportRules = R.REPORT_V1) -> dict[str, Any]:
    a = 1000 * rng.randint(1, 5)
    b = a + rng.choice([500, 1000, 2000, 6000])
    frm = rng.choice([0, 1, 40, rng.randint(1, 60_000)])
    to = rng.choice([frm, frm + 1, max(0, frm - 1), rng.randint(0, 60_000)])
    mn = rng.choice([None, to + 5, max(0, min(frm, to) - 1), int(0.5 * min(frm, to))])
    kinds: dict[str, list[dict]] = {"population": [pop(frm, to, mn, b - 200, rules)]}
    if rng.random() < 0.9:
        kinds["deaths"] = [
            deaths(*(rng.choice([0, 1, 7, 12_345]) for _ in range(3)), rng.choice([0, 0, 1, 900]))
        ]
    if rng.random() < 0.9:
        kinds["births"] = [
            {"kind": "births", "births": rng.choice([0, 1, 33_000]), "immigrants": rng.choice([0, 0, 1, 64])}
        ]
    if rng.random() < 0.6:
        kinds["lineages"] = [{"kind": "lineages", "from": rng.randint(1, 20), "to": rng.randint(1, 20)}]
        kinds["top_lineage"] = [
            {
                "kind": "top_lineage",
                "from_share": 0.25,
                "to_share": 0.5,
                "from_count": 100,
                "to_count": 200,
                "from_of": 400,
                "to_of": 400,
            }
        ]
    ids = rng.sample(range(1, 40), 8)
    kin = [(i, rng.choice([*NAMES, None])) for i in ids]
    alive_b = max(to, 1)
    sizes = sorted((rng.choice([1, 1, 50, 900, 2000]) for _ in range(rng.randint(0, 6))), reverse=True)
    groups, rank = [], True
    for j, size in enumerate(sizes):
        nxt = sizes[j + 1] if j + 1 < len(sizes) else None
        rank = rank and (nxt is None or size > nxt)
        groups.append(
            group(
                ids[j],
                size,
                alive_b,
                j + 1 if rank else None,
                *(rng.choice([(None, None, None), (rng.randint(0, 9), 5, 2), (1, 1, 1)])),
            )
        )
    if groups:
        kinds["group"] = groups
    evs = []
    for _ in range(rng.randint(0, 4)):  # colony event frames
        kind = rng.choice(["born", "split", "merged", "dissolved"])
        other = -1 if kind in ("born", "dissolved") and rng.random() < 0.5 else rng.choice(ids)
        size = rng.choice([40, 700, 5000])
        evs.append(
            event(
                kind,
                rng.choice(ids),
                size,
                other,
                rng.randint(0, size) if kind == "born" else 0,
                a + 250 * rng.randint(0, 3),
                a + 250 * rng.randint(4, 8),
                "colony",
                rng.random() < 0.15,
            )
        )
    evs.sort(key=lambda f: (-f["size"], f["s1"], f["system"]))
    if evs:
        kinds["event"] = evs[:6]
    names = rng.sample(NAMES, 6)
    ths = []
    for j in range(
        rng.randint(0, 6)
    ):  # the kin threads: formed (from another), merged into another, dissolved
        what = rng.choice(["formed", "formed_from", "into", "dissolved", "into_initial", "dissolved_initial"])
        size = rng.choice([1, 40, 700, 5000])
        told = a + 250 * rng.randint(1, 4)
        first = a - rng.choice([0, 750, 3000])
        other = names[(j + 1) % 6] if what in ("formed_from", "into", "into_initial") else None
        init = what.endswith("initial")
        ths.append(
            thread(
                "formed" if what.startswith("formed") else "ended",
                j + 1,
                names[j],
                size,
                other=other,
                first=first,
                told=told,
                end=told - 250,
                steps=None if init else told - 250 - first,
                peak=size + rng.choice([0, 300]),
                imp=rng.choice([1, 2]),
                initial=init,
            )
        )
    if ths:
        kinds["thread"] = ths
    if rng.random() < 0.5:
        kinds["untold"] = [untold(rng.choice([1, 7, 94]))]
    if "deaths" in kinds and "births" in kinds and frm + to > 0:
        kinds["turnover"] = [turnover(kinds["births"][0]["births"], kinds["deaths"][0]["total"], frm, to)]
    if rng.random() < 0.3:
        kinds["starve"] = [{"kind": "starve", "n": 1500, "s": b - 250, "steps": 250, "median": 1000}]
    if rng.random() < 0.3:
        kinds["stormdeaths"] = [
            {"kind": "stormdeaths", "n": rng.choice([10, 14]), "s": b - 500, "steps": 250}
        ]
    if rng.random() < 0.6:
        kinds["storms"] = [
            storms(rng.randint(0, 8), 0, rng.randint(0, 3), rng.randint(0, 2), rng.randint(0, 3))
        ]
        kinds["storms"][0]["named"] = min(kinds["storms"][0]["tracked"], rng.randint(0, 5))
        sts = []
        for j in range(rng.randint(0, 3)):
            ended = rng.random() < 0.5
            sts.append(
                storm(
                    j + 1,
                    names[5 - j],
                    named=a + 100 * rng.randint(1, 5) if (not ended or rng.random() < 0.5) else None,
                    end=a + 100 * rng.randint(6, 9) if ended else None,
                    age=rng.choice([0, 300, 8725]) if ended else None,
                    ratio=rng.choice([1.0, 1.01, 2.5]),
                    hemisphere=rng.choice(["northern", "southern"]),
                )
            )
        if sts:
            kinds["storm"] = sts
    if rng.random() < 0.7:
        kinds["houses"] = [
            houses(
                rng.randint(0, 20),
                rng.randint(0, 20),
                rng.choice([0, 1, 4]),
                rng.choice([0, 1, 5]),
                rng.random() < 0.7,
                0,
                rng.choice([None, 3]),
                rng.randint(1, 9000),
                rng.random() < 0.6,
            )
        ]
    if rng.random() < 0.5:
        kinds["label"] = [
            {
                "kind": "label",
                "label": lb.capitalize(),
                "key": lb,
                "family": "archetype" if lb in R.ARCHETYPE_NAMES else "intent",
                "from_share": 0.1,
                "to_share": 0.3,
                "from_count": 40,
                "to_count": 120,
                "from_of": 400,
                "to_of": 400,
            }
            for lb in rng.sample(["hunter", "provider", "hunting", "resting"], 2)
        ]
    if len(ids) > 3 and rng.random() < 0.5:
        kinds["relation"] = [
            {
                "kind": "relation",
                "a": ids[0],
                "b": ids[1],
                "attacks": rng.randint(10, 900),
                "kills": rng.choice([0, 1, 7]),
            }
        ]
        kinds["relation"][0]["kills"] = min(kinds["relation"][0]["kills"], kinds["relation"][0]["attacks"])
    if rng.random() < 0.6:
        kinds["colonies"] = [
            {"kind": "colonies", "tracked": rng.choice([1, 2, 7]), "largest_size": rng.choice([1, 40])}
        ]
    if to == 0 and rng.random() < 0.7:
        kinds["run_end"] = [{"kind": "run_end", "step": b + 17, "alive": 0}]
    ivs, standing = _random_interventions(rng, a, b)
    kinds["intervention"], kinds["standing"] = ivs, standing
    timed = [
        f
        for f in ivs
        if f["duration"] is not None and f["step"] <= a and (f["end"] is None or f["end"] >= b - 250)
    ]
    if timed and rng.random() < 0.5:  # a window lies inside the span of a timed entry
        kinds["window"] = [
            {
                "kind": "window",
                "entry": timed[0]["entry"],
                "w0": a,
                "w1": b - 250,
                "r0": a,
                "r1": b - 250,
                "from": 100,
                "to": 90,
            }
        ]
    ents = [("kin_group", i, nm) for i, nm in kin] + [("house", 3, "Zaeld")]
    dropped = rng.choice([0, 0, 1, 5])
    snaps = [
        {"id": f"S{s}", "step": s, "alive": rng.choice([None, 1, 1000])} for s in range(a + 250, b + 1, 250)
    ]
    return mk_sheet(
        a=a,
        b=b,
        kinds=kinds,
        ents=ents,
        excluded={"interventions_dropped": dropped},
        snapshots=snaps,
        rules=rules,
    )
