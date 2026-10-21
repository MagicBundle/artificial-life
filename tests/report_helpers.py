"""Scripted fact sheets for the world report's tests (docs/DESIGN.md §30): ``mk_sheet`` builds a sheet from the
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
    "events_suspect": 0,
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


def group(entity, size, of, rank=None, born=None, died=None, killed=None):
    f: dict[str, Any] = {"kind": "group", "entity": entity, "size": size, "of": of, "share": size / of}
    if rank is not None:
        f["rank"] = rank
    if born is not None:
        f.update({"born": born, "died": died, "died_killed": killed})
    return f


def event(kind, system, size, other=-1, overlap=0, s0=A0 + 250, s1=A0 + 500, family="kin_group", reid=False):
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
):
    f = {
        "kind": "houses",
        "alive_from": alive_from,
        "alive_to": alive_to,
        "first_seen": first_seen,
        "extinct": extinct,
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
            "event": [
                event("split", 2, 1210, 1, 1180),
                event("dissolved", 3, 980),
                event("born", 4, 640),
                event("merged", 5, 520, 1, 505),
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
        ents=[("kin_group", i, nm) for i, nm in kin]
        + [("kin_group", 6, None), ("kin_group", 7, None), ("house", 3, "Zaeld")],
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
        timed = action in R.TIMED
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
            "fertility_shift": rng.choice(
                [{"patch": 5, "gain": rng.choice([0.25, 1.75])}, {"patch": 6, "x": 300.5, "y": 120.25}]
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
    for _ in range(rng.randint(0, 7)):
        kind = rng.choice(["born", "split", "merged", "dissolved"])
        fam = rng.choice(["kin_group", "kin_group", "colony"])
        other = -1 if kind in ("born", "dissolved") and rng.random() < 0.5 else rng.choice(ids)
        size = rng.choice([1, 40, 700, 700, 5000])
        evs.append(
            event(
                kind,
                rng.choice(ids),
                size,
                other,
                rng.randint(0, size) if kind == "born" else 0,
                a + 250 * rng.randint(0, 3),
                a + 250 * rng.randint(4, 8),
                fam,
                rng.random() < 0.15,
            )
        )
    evs.sort(key=lambda f: (-f["size"], f["s1"], f["family"], f["system"]))
    if evs:
        kinds["event"] = evs[:6]
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
    if ivs and rng.random() < 0.5:
        kinds["window"] = [
            {"kind": "window", "entry": ivs[0]["entry"], "w0": a, "w1": b - 250, "from": 100, "to": 90}
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
