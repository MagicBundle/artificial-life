"""The emergence panel of one run (DESIGN §44.7): a JSON/markdown record with a fixed section order.

``PANEL_VERSION`` 1. Sections, in order: ``header`` (run, arm, seed, config hash, fork step, windows,
versions, prereg hash, health gates, flags), ``A`` open-endedness, ``B`` irreversibility, ``C`` braids and
winding, ``D`` scorecards, ``endpoints`` (what the comparison uses: P1 to P5 and the scorecard primaries,
each with its block values) and ``not_claimed``. A block that has not been computed (its module has not
landed, or its data is absent) is ``{"status": "needs_missing", ...}``, printed as "not evaluated", never as
zero. No composite index exists: a weighted sum would hide which hallmark moved.

Reading a run uses its JSON files only (``config.json``, ``meta.json``, ``metrics.jsonl``); nothing is written
inside the run directory.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from alife.analysis.emergence.frames import FRAMES_VERSION
from alife.analysis.emergence.prereg import ENDPOINTS
from alife.analysis.emergence.scorecards import FEATURES, SCORECARD_VERSION, ScorecardResult, not_evaluated

PANEL_VERSION: Final = 1
SECTIONS: Final = ("header", "A", "B", "C", "D", "endpoints", "not_claimed")
HEADER_KEYS: Final = (
    "panel_version",
    "run",
    "arm",
    "seed",
    "config_hash",
    "fork_step",
    "windows",
    "versions",
    "prereg_hash",
    "gates",
    "flags",
)
GATES: Final = ("population", "ledger", "resim")
BLOCK_NAMES: Final = {"A": "open-endedness", "B": "irreversibility", "C": "braids and winding"}

NOT_CLAIMED: Final = (
    "Entropy production is a lower bound (coarse-graining and lag only lose irreversibility).",
    "Braid entropy is a lower-bound estimate (Burau t = -1) and a coordination index, not a score of more "
    "braiding.",
    "No endpoint is a composite emergence score.",
    "All thresholds are provisional until the calibration run.",
    "With 3 seeds forked from one evolved population nothing generalises to other evolutionary histories.",
)


class PanelError(ValueError):
    """A panel that does not match the schema, or panels of mixed versions."""


def needs_block(reason: str, *needs: str) -> dict[str, Any]:
    """A block that was not computed."""
    return {"status": "needs_missing", "reason": reason, "needs_missing": list(needs) or [reason]}


def endpoint_entry(
    name: str,
    *,
    value: float | None = None,
    blocks: Sequence[float | None] | None = None,
    needs: Sequence[str] = (),
    twin_spread: float | None = None,
) -> dict[str, Any]:
    """One row of the endpoints table. ``blocks`` are the block values the comparison pairs across arms."""
    meta = ENDPOINTS.get(name, {})
    ok = blocks is not None and not needs
    return {
        "status": "ok" if ok else "needs_missing",
        "value": value,
        "blocks": None if blocks is None else [None if b is None else float(b) for b in blocks],
        "direction": meta.get("direction", 1),
        "confirmatory": meta.get("confirmatory", False),
        "twin_spread": twin_spread,
        "needs_missing": list(needs),
    }


def gate(ok: bool | None, detail: str) -> dict[str, Any]:
    return {"ok": ok, "detail": detail}


def health_gates(
    rows: Sequence[Mapping[str, Any]],
    window: tuple[int, int] | None = None,
    *,
    population_ref: float | None = None,
    resim_ok: bool | None = None,
) -> dict[str, dict[str, Any]]:
    """The gates that block inference when they fail (None = not evaluated, which does not block)."""
    sel = [r for r in rows if window is None or window[0] < int(r["step"]) <= window[1]]
    out: dict[str, dict[str, Any]] = {}
    if sel and population_ref:
        pops = sorted(float(r["n_alive"]) for r in sel if r.get("n_alive") is not None)
        med = pops[len(pops) // 2] if pops else None
        ok = med is not None and 0.5 <= med / population_ref <= 2.0
        out["population"] = gate(
            ok, f"median alive {med} against reference {population_ref} (band x0.5 to x2)"
        )
    else:
        out["population"] = gate(None, "no reference population or no metrics rows")
    errs = [r.get("ledger_error") for r in sel if r.get("ledger_error") is not None]
    out["ledger"] = gate(
        (not any(errs)) if errs else None,
        f"{len(errs)} rows, ledger_error nonzero in {sum(bool(e) for e in errs)}",
    )
    out["resim"] = gate(resim_ok, "frames certified against the run" if resim_ok else "no verified frames")
    return out


def new_panel(
    *,
    run: str,
    arm: str,
    seed: int | None,
    config_hash: str | None,
    fork_step: int | None,
    windows: Sequence[tuple[int, int]],
    prereg_hash: str | None = None,
    gates: Mapping[str, Mapping[str, Any]] | None = None,
    blocks: Mapping[str, Mapping[str, Any]] | None = None,
    scorecards: Sequence[ScorecardResult] = (),
    endpoints: Mapping[str, Mapping[str, Any]] | None = None,
    flags: Mapping[str, Any] | None = None,
    versions: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a panel; every block or endpoint not given is ``needs_missing``."""
    blk = {
        k: dict(blocks[k]) if blocks and k in blocks else needs_block(f"{BLOCK_NAMES[k]} not computed")
        for k in BLOCK_NAMES
    }
    cards = {c.feature: c.to_json() for c in scorecards}
    for f in FEATURES:
        cards.setdefault(f, not_evaluated(f, ["not requested"]).to_json())
    ep = {
        k: dict(endpoints[k])
        if endpoints and k in endpoints
        else endpoint_entry(k, needs=["block not computed"])
        for k in ENDPOINTS
    }
    for f in FEATURES:  # S_k: the scorecard primaries
        c = cards[f]
        pr = c["endpoint"].get("primary")
        ep[f"S_{f}"] = endpoint_entry(
            f"S_{f}", value=pr, blocks=None if pr is None else [pr], needs=c["needs_missing"]
        )
    g = {k: dict(gates[k]) if gates and k in gates else gate(None, "not evaluated") for k in GATES}
    missing = sorted({n for b in blk.values() for n in b.get("needs_missing", [])})
    fl = {
        "nonstationary": None,
        "low_n": None,
        "needs_missing": missing,
        "blocked": any(v["ok"] is False for v in g.values()),
    }
    fl.update(flags or {})
    ver = {
        "panel": PANEL_VERSION,
        "frames": FRAMES_VERSION,
        "scorecards": SCORECARD_VERSION,
        **(versions or {}),
    }
    return {
        "header": {
            "panel_version": PANEL_VERSION,
            "run": run,
            "arm": arm,
            "seed": seed,
            "config_hash": config_hash,
            "fork_step": fork_step,
            "windows": [list(w) for w in windows],
            "versions": ver,
            "prereg_hash": prereg_hash,
            "gates": g,
            "flags": fl,
        },
        "A": blk["A"],
        "B": blk["B"],
        "C": blk["C"],
        "D": {"cards": cards},
        "endpoints": ep,
        "not_claimed": list(NOT_CLAIMED),
    }


def validate(panel: Mapping[str, Any]) -> None:
    """Raise ``PanelError`` unless ``panel`` has the sections and header keys of this ``PANEL_VERSION``."""
    if tuple(panel.keys()) != SECTIONS:
        raise PanelError(f"panel sections {tuple(panel.keys())}, expected {SECTIONS}")
    h = panel["header"]
    if h.get("panel_version") != PANEL_VERSION:
        raise PanelError(f"panel version {h.get('panel_version')}, this code reads {PANEL_VERSION}")
    if tuple(h.keys()) != HEADER_KEYS:
        raise PanelError(f"panel header keys {tuple(h.keys())}, expected {HEADER_KEYS}")


def check_same_version(panels: Sequence[Mapping[str, Any]]) -> None:
    """The comparison refuses panels of mixed versions."""
    vs = {p["header"]["panel_version"] for p in panels}
    if len(vs) > 1:
        raise PanelError(f"panels of mixed versions {sorted(vs)}")
    for p in panels:
        validate(p)


def read_run_header(run: Path) -> dict[str, Any]:
    """``{run, seed, config_hash, start_step, rows}`` from a run directory's JSON files (read only)."""
    run = Path(run)
    meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    mp = run / "metrics.jsonl"
    if mp.is_file():
        for line in mp.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    break  # a torn last line
    return {
        "run": run.name,
        "seed": meta.get("seed"),
        "config_hash": meta.get("config_hash"),
        "start_step": meta.get("start_step"),
        "rows": rows,
    }


def build_panel(
    run: Path,
    windows: Sequence[tuple[int, int]],
    *,
    arm: str = "unlabelled",
    prereg_hash: str | None = None,
    population_ref: float | None = None,
    frames_meta: Sequence[Mapping[str, Any]] = (),
    scorecards: Sequence[ScorecardResult] = (),
) -> dict[str, Any]:
    """The panel of a run with the blocks that exist (header, gates, scorecards; A, B, C: E3 to E5)."""
    h = read_run_header(run)
    resim_ok = all(m.get("verification", {}).get("verified") for m in frames_meta) if frames_meta else None
    gates = health_gates(
        h["rows"],
        (min(w[0] for w in windows), max(w[1] for w in windows)) if windows else None,
        population_ref=population_ref,
        resim_ok=resim_ok,
    )
    return new_panel(
        run=h["run"],
        arm=arm,
        seed=h["seed"],
        config_hash=h["config_hash"],
        fork_step=h["start_step"],
        windows=windows,
        prereg_hash=prereg_hash,
        gates=gates,
        scorecards=scorecards,
    )


def _fmt(v: Any) -> str:
    return "not evaluated" if v is None else (f"{v:.5g}" if isinstance(v, float) else str(v))


def to_markdown(panel: Mapping[str, Any]) -> str:
    h = panel["header"]
    L = [f"# Emergence panel: {h['run']} (arm {h['arm']}, seed {h['seed']})", ""]
    L += [f"- config hash `{h['config_hash']}`, fork step {h['fork_step']}, windows {h['windows']}"]
    L += [
        f"- panel version {h['panel_version']}, prereg hash `{h['prereg_hash']}`",
        f"- flags: {h['flags']}",
        "",
    ]
    L += (
        ["## Health gates", ""]
        + [f"- {k}: {_fmt(v['ok'])} ({v['detail']})" for k, v in h["gates"].items()]
        + [""]
    )
    for k, title in (("A", "A open-endedness"), ("B", "B irreversibility"), ("C", "C braids and winding")):
        b = panel[k]
        if b.get("status") == "needs_missing":
            body = "not evaluated: " + "; ".join(b["needs_missing"])
        else:
            body = json.dumps(b, indent=1)[:2000]
        L += [f"## {title}", "", body, ""]
    L += ["## D scorecards", ""]
    for f, c in panel["D"]["cards"].items():
        if c["needs_missing"]:
            L.append(f"- {f}: not evaluated ({'; '.join(c['needs_missing'])})")
        else:
            pr = _fmt(c["endpoint"].get("primary"))
            L.append(
                f"- {f}: {c['label']} (level {_fmt(c['level'])}, primary {pr}, "
                f"provisional {c['provisional']})"
            )
            for x in c["criteria"]:
                L.append(
                    f"    - {x['name']}: {_fmt(x['value'])} vs {_fmt(x['threshold'])} -> "
                    f"{_fmt(x['passed'])} {x['note']}"
                )
    L += ["", "## Endpoints", "", "| endpoint | status | value | blocks |", "|---|---|---|---|"]
    for k, e in panel["endpoints"].items():
        L.append(f"| {k} | {e['status']} | {_fmt(e['value'])} | {len(e['blocks']) if e['blocks'] else 0} |")
    L += ["", "## Not claimed", ""] + [f"- {s}" for s in panel["not_claimed"]]
    return "\n".join(L) + "\n"


def write_panel(out_dir: Path, panel: Mapping[str, Any]) -> tuple[Path, Path]:
    """``panel.json`` and ``panel.md`` under ``out_dir`` (atomic writes)."""
    validate(panel)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, text in (("panel.json", json.dumps(panel, indent=2) + "\n"), ("panel.md", to_markdown(panel))):
        p = out_dir / name
        tmp = out_dir / f".{name}.tmp{os.getpid()}"
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, p)
        paths.append(p)
    return paths[0], paths[1]


def read_panel(path: Path) -> dict[str, Any]:
    p = json.loads(Path(path).read_text(encoding="utf-8"))
    validate(p)
    return p
