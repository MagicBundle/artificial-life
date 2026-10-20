"""A synthetic observed run for the world report's tests (docs/DESIGN.md §30): hand-written files in the exact
formats the engine writes (``metrics.jsonl`` rows with the ``observers`` block, ``observers/social.jsonl``
records with column tables, ``director.jsonl``, ``config.json``, ``meta.json`` and a sidecar holding only its
``meta``), no simulation. ``tests/data/report_run_synth/`` is ``write_run`` with its defaults (a test pins that
the committed files are what this module writes).

The world: rows every 100 steps to 4,000, snapshots every 250, the population ``alive(t)`` of ``alive``
(growing by 1 per 20 steps from 1,000, with a dip of 300 at step 1,500), four kin groups (Braelix, Shaemund,
Torvane, Kraiost; Torvane forms at 1,250 and dissolves at 3,250, Kraiost forms at 2,250) with window births and
deaths, a split at 2,000 and a merged event at 1,500 inside the dip (suspect, so excluded), houses, one
relation, colonies and no director entries. Chapters are cut at 1,000, 2,000, 3,000 and 4,000. ``variant`` and
``env_rows`` build the other worlds the tests need from these files in memory.
"""

# ruff: noqa: E501  (expected report texts and fixture tables are long lines)

from __future__ import annotations

import copy
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from alife.config import apply_overrides, load_config, save_config
from alife.game.report import RunView, load_run
from alife.game.rules import RULES_V1
from alife.game.social import SOCIAL_V1

END = 4000
NAMES = {1: "Braelix", 2: "Shaemund", 3: "Torvane", 4: "Kraiost"}
HOUSES = {101: "Zaeld", 102: "Kreoth", 103: "Mivask"}
V1_DIGEST = RULES_V1.digest()


def alive(t: int) -> int:
    """The population at step ``t`` (an integer for every multiple of 50)."""
    return 1000 + t // 20 - max(0, 300 - 3 * abs(t - 1500))


def n_alive_min(step: int) -> int:
    """The minimum of the row ending at ``step`` (its interval is (step - 100, step])."""
    return min(alive(step - 100), alive(step))


def groups_at(t: int) -> list[int]:
    """The ids of the slotted kin groups at snapshot ``t``."""
    ids = [1, 2]
    if 1250 <= t < 3250:
        ids.append(3)
    if t >= 2250:
        ids.append(4)
    return ids


def size_of(gid: int, t: int) -> int:
    return alive(t) // {1: 5, 2: 8, 3: 12, 4: 20}[gid]


def _table(cols: Sequence[str], rows: Sequence[Sequence[Any]]) -> dict[str, list[Any]]:
    return {c: [r[j] for r in rows] for j, c in enumerate(cols)}


def head_at(step: int) -> dict[str, Any]:
    snap = step - step % 250
    return {
        "social_version": 1,
        "snapshot_step": snap,
        "origin": 0,
        "houses": {"alive": 14 if snap < 2000 else 12, "exact": True, "unknown_immigrant": 0},
        "kin_groups": {
            "tracked": len(groups_at(snap)),
            "other": alive(snap) - sum(size_of(g, snap) for g in groups_at(snap)),
        },
        "colonies": {"tracked": 3, "members": 90, "largest": 40, "converged": True, "match": "uid"},
    }


def row_at(step: int) -> dict[str, Any]:
    n = alive(step)
    return {
        "step": step,
        "steps": 100,
        "wall_s": 1.0,
        "n_alive": float(n) + 0.5,
        "n_alive_min": n_alive_min(step),
        "births": 10,
        "immigrants": 0,
        "deaths_starved": 5,
        "deaths_killed": 3,
        "deaths_aged": 1,
        "deaths_catastrophe": 0,
        "clades_alive": 5,
        "top_clade_share": (n // 4) / n,
        "observers": {
            "intent_version": 1,
            "labels_step": step - step % 8,
            "alive": n,
            "arch_labelled": 400,
            "intent_evaluated": n,
            "intent_labelled": 400,
            "archetype": None,
            "archetype_match": {"hunter": 0.25 + step / 40000, "settler": 0.5},
            "intent": {"resting": 0.5 - step / 40000, "hunting": 0.125},
            "intent_match": None,
            "social": head_at(step),
        },
    }


def record_at(step: int) -> dict[str, Any]:
    ids = groups_at(step)
    sizes = {g: size_of(g, step) for g in ids}
    kin = _table(
        ("id", "name", "named_after", "slot", "size", "births", "deaths"),
        [(g, NAMES[g], None, j, sizes[g], 5 - g % 2, [2, 1, 0, 0]) for j, g in enumerate(ids)],
    )
    events: list[tuple[Any, ...]] = []
    if step == 1250:
        events.append((step, "kin_group", "born", 3, sizes[3], -1, 0))
    if step == 1500:
        events.append((step, "kin_group", "merged", 9, 120, 1, 118))
    if step == 2000:
        events.append((step, "kin_group", "split", 2, 130, 1, 120))
    if step == 2250:
        events.append((step, "kin_group", "born", 4, sizes[4], -1, 0))
        events.append((step, "colony", "born", 21, 40, -1, 0))
        events.append((step, "colony", "born", 22, 20, -1, 0))
    if step == 3250:
        events.append((step, "kin_group", "dissolved", 3, size_of(3, 3000), -1, 0))
    n_ref = alive(step)
    house_events = {
        "founded": _table(
            ("id", "step", "parent_house"), [(500 + step, step, None)] if step % 500 == 0 else []
        ),
        "extinct": _table(("id", "step"), [(400 + step, step)] if step % 1000 == 0 else []),
        "founded_dropped": 0,
        "extinct_dropped": 0,
    }
    return {
        "step": step,
        "social_version": 1,
        "social_digest": SOCIAL_V1.digest(),
        "origin": 0,
        "window_start": step - 250,
        "summary": {
            "houses": {"alive": 14 if step < 2000 else 12, "exact": True, "unknown_immigrant": 0},
            "kin_groups": {"tracked": len(ids), "other": alive(step) - sum(sizes.values())},
            "colonies": {"tracked": 3, "members": 90, "largest": 40, "converged": True, "match": "uid"},
        },
        "thresholds": {"n_ref": n_ref, "rescaled": False, "occ_min": 2, "kin_enter": 8, "kin_leave": 4},
        "houses": _table(
            ("id", "name", "size", "clade"),
            [(101, "Zaeld", 300, 1), (102, "Kreoth", 200, 2), (103, "Mivask", 200, 3)],
        ),
        "house_events": house_events,
        "kin_groups": kin,
        "colonies": _table(("id", "slot", "size", "kin_group", "x", "y"), []),
        "events": _table(("step", "family", "kind", "id", "size", "other", "overlap"), events),
        "events_dropped": 0,
        "event_counts": {},
        "relations": _table(
            (
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
            ),
            [(1, 2, 3, 3, 1, 0, 0, 0, 1.5, None, 0.2, True), (2, 1, 1, 1, 0, 0, 0, 0, 0.5, None, 0.2, True)],
        ),
        "relations_dropped": 0,
    }


def sidecar_meta(step: int = 0) -> dict[str, Any]:
    return {"format": "alife-observers", "intent_version": 1, "rules_digest": V1_DIGEST, "step": step}


def config():
    cfg = load_config(preset="tiny", overrides=["brain.weight_dtype=float32"])
    return apply_overrides(cfg, ["game.observe=true", "director.enabled=true", "fields.n_patches=24"])


def write_run(path: Path, *, entries: Sequence[Mapping[str, Any]] = ()) -> Path:
    """Write the synthetic run to ``path`` (created) and return it."""
    path = Path(path)
    (path / "observers").mkdir(parents=True, exist_ok=True)
    save_config(config(), path / "config.json")
    (path / "meta.json").write_text(
        json.dumps({"format_version": 1, "seed": 0, "start_step": 0, "status": "done"}, indent=2) + "\n",
        encoding="utf-8",
    )
    with open(path / "metrics.jsonl", "w", encoding="utf-8") as f:
        for s in range(100, END + 1, 100):
            f.write(json.dumps(row_at(s)) + "\n")
    with open(path / "observers" / "social.jsonl", "w", encoding="utf-8") as f:
        for s in range(250, END + 1, 250):
            f.write(json.dumps(record_at(s)) + "\n")
    np.savez(
        path / "observers" / "step_0000000000.npz", meta=np.array(json.dumps(sidecar_meta(), sort_keys=True))
    )
    if entries:
        with open(path / "director.jsonl", "w", encoding="utf-8") as f:
            f.write(
                json.dumps({"kind": "header", "step": 0, "rules_version": 2, "persona": "gardener"}) + "\n"
            )
            for e in entries:
                f.write(json.dumps(e) + "\n")
    return path


# --- variants ------------------------------------------------------------------------------------------------


def entry(
    step: int, source: str, action: str, params: Mapping[str, Any], duration: int | None = None, **more
):
    e: dict[str, Any] = {
        "step_applied": step,
        "source": source,
        "action": action,
        "params": dict(params),
        "duration": duration,
    }
    e.update(more)
    return e


def variant(
    rv: RunView,
    *,
    rows: Callable[[list[dict[str, Any]]], list[dict[str, Any]]] | None = None,
    records: Callable[[list[dict[str, Any]]], list[dict[str, Any]]] | None = None,
    entries: Sequence[Mapping[str, Any]] | None = None,
    sidecars: Mapping[int, Any] | None = None,
    cfg: Any = None,
    meta: Mapping[str, Any] | None = None,
) -> RunView:
    """A deep copy of ``rv`` with the given parts replaced (``rows`` and ``records`` are functions of the copied
    list). When ``entries`` is given the rows' ``env`` blocks are recomputed from them (``env_rows``)."""
    new_rows = copy.deepcopy(list(rv.rows))
    new_records = copy.deepcopy(list(rv.records))
    if rows is not None:
        new_rows = rows(new_rows)
    if records is not None:
        new_records = records(new_records)
    ents = copy.deepcopy(list(rv.entries if entries is None else entries))
    start = int((rv.meta if meta is None else meta).get("start_step") or 0)
    out_cfg = rv.cfg if cfg is None else cfg
    if entries is not None:
        new_rows = env_rows(new_rows, ents, start)
    return RunView(
        out_cfg,
        dict(rv.meta if meta is None else meta),
        tuple(new_rows),
        tuple(new_records),
        tuple(ents),
        dict(rv.sidecars if sidecars is None else sidecars),
    )


def env_rows(
    rows: list[dict[str, Any]], entries: Sequence[Mapping[str, Any]], start: int
) -> list[dict[str, Any]]:
    """``rows`` with an ``env`` block per row, computed by an independent mini-model of the director's levers:
    the last write of each lever by the entries applied strictly between ``start`` and the row (a revert writes
    its params), patches 24 at gain 1. The model does not import the report's fold."""
    out = []
    for r in rows:
        regen, amp, period, drift = 1.0, 0.0, 0, 1.0
        gain = [1.0] * 24
        for e in sorted(
            (e for e in entries if start < e["step_applied"] < r["step"]), key=lambda e: e["step_applied"]
        ):
            p, a = e["params"], e["action"]
            if a == "regen_pulse":
                regen = p["mult"]
            elif a == "set_seasons":
                amp, period = p["amp"], p["period"]
            elif a == "drift_change":
                drift = p["mult"]
            elif a in ("habitat_loss", "fertility_shift") and "gain" in p:
                gain = [p["gain"] if j == p["patch"] else g for j, g in enumerate(gain)]
        g32 = np.array(gain, dtype=np.float32)
        out.append(
            {
                **r,
                "env": {
                    "regen_mult": float(np.float32(regen)),
                    "season_amp": float(np.float32(amp)),
                    "season_period": period,
                    "drift_mult": float(np.float32(drift)),
                    "imm_request": 0,
                    "cat_radius": 0.0,
                    "cat_kill_p": 0.0,
                    "patch_gain_min": float(g32.min()),
                    "patch_gain_max": float(g32.max()),
                    "patches_off": int((g32 == 0).sum()),
                },
            }
        )
    return out


def synth(tmp_path: Path, **kw: Any) -> RunView:
    return load_run(write_run(tmp_path / "run", **kw))


def make_view(a0: int, b0: int, entries: Sequence[Mapping[str, Any]] = (), *, start: int = 0) -> RunView:
    """An in-memory run covering the steps ``a0`` to ``b0`` (rows every 100 including the row at ``a0``, records
    every 250 after it) with the ``entries`` and consistent ``env`` blocks: for chapters at any step, the sizes of
    the world follow ``alive``/``groups_at`` as in the committed run."""
    rows = [row_at(s) for s in range(a0, b0 + 1, 100)]
    recs = [record_at(s) for s in range(a0 + 250, b0 + 1, 250)]
    meta = {"format_version": 1, "seed": 0, "start_step": start, "status": "done"}
    rows = env_rows(rows, entries, start) if entries else rows
    return RunView(
        config(), meta, tuple(rows), tuple(recs), tuple(copy.deepcopy(list(entries))), {0: sidecar_meta()}
    )


def oracle_state(entries: Sequence[Mapping[str, Any]], t: int, start: int = 0) -> dict[str, Any]:
    """The levers at the row ``t`` by a model that does not import the report's fold: ``regen``, ``drift``,
    ``seasons`` (period, amp), ``gain`` per patch and the patches' moved positions."""
    st: dict[str, Any] = {"regen": 1.0, "drift": 1.0, "seasons": (0, 0.0), "gain": {}, "pos": {}}
    for e in sorted((e for e in entries if start < e["step_applied"] < t), key=lambda e: e["step_applied"]):
        p, a = e["params"], e["action"]
        if a == "regen_pulse":
            st.update({"regen": p["mult"]})
        elif a == "drift_change":
            st.update({"drift": p["mult"]})
        elif a == "set_seasons":
            st.update({"seasons": (p["period"], p["amp"])})
        elif a in ("habitat_loss", "fertility_shift") and "gain" in p:
            st.update({"gain": {**st["gain"], p["patch"]: p["gain"]}})
        elif a == "fertility_shift":
            st.update({"pos": {**st["pos"], p["patch"]: (p["x"], p["y"])}})
    return st
