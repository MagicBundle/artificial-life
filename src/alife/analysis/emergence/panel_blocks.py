"""Fill the analysis blocks of a built panel (DESIGN §44.7): B here, C and A with their stages.

Each ``fill_*`` mutates a panel made by ``panel.build_panel`` / ``panel.new_panel``: it sets the block, the
endpoint rows it owns and the header flags, leaving everything else as it was. Absent data stays
``needs_missing``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from alife.analysis.emergence import activity_adapter as aa
from alife.analysis.emergence import braid as braid_mod
from alife.analysis.emergence import modes as modes_mod
from alife.analysis.emergence.epr import CAVEATS, EPR_VERSION
from alife.analysis.emergence.panel import endpoint_entry, needs_block

B_KEYS = (
    "cell",
    "tau",
    "own",
    "raw",
    "ambient",
    "gradient_null",
    "drift_removal",
    "activity",
    "stationarity",
    "vertical",
)


def fill_b(panel: dict[str, Any], results: Sequence[dict[str, Any]]) -> None:
    """Block B from the ``epr.analyse_window`` results of the recorded windows (in window order).

    The P4 endpoint is the own-motion rate; its block values are the 16 time-block rates of every window, in
    order, with a nonstationary window's blocks set to ``None`` (not used in an inference, DESIGN §44.3)."""
    if not results:
        return
    blocks: list[float | None] = []
    flags = []
    for r in results:
        bad = bool(r["stationarity"]["nonstationary"])
        flags.append(bad)
        blocks += [None if bad else b for b in r["blocks"]]
    vals = [b for b in blocks if b is not None]
    panel["B"] = {
        "status": "ok",
        "version": EPR_VERSION,
        "windows": [
            {k: r[k] for k in B_KEYS}
            | {"scale_curve": r["scale_curve"], "clades": r["clades"], "bodies": r["bodies"]}
            for r in results
        ],
        "caveats": list(CAVEATS),
        "needs_missing": [],
    }
    panel["endpoints"]["P4"] = endpoint_entry(
        "P4",
        value=float(np.mean(vals)) if vals else None,
        blocks=blocks if vals else None,
        needs=[] if vals else ["no stationary window"],
    )
    panel["header"]["flags"]["nonstationary"] = any(flags)
    panel["header"]["versions"]["epr"] = EPR_VERSION
    panel["header"]["flags"]["needs_missing"] = sorted(
        {n for k in ("A", "B", "C") for n in panel[k].get("needs_missing", [])}
    )


def fill_c(
    panel: dict[str, Any], results: Sequence[dict[str, Any]], winding: dict[str, Any] | None = None
) -> None:
    """Block C from the ``braid.analyse_blocks`` results of the windows (in window order) and, optionally, the
    winding table (descriptive). The P5 block values are the per-step braid contrast of every window's
    blocks, in order (``None`` where a block had no usable group)."""
    if not results:
        return
    blocks = [b for r in results for b in r["blocks"]]
    vals = [b for b in blocks if b is not None]
    panel["C"] = {
        "status": "ok",
        "version": braid_mod.BRAID_VERSION,
        "braid": {
            "windows": [{k: v for k, v in r.items() if k != "caveats"} for r in results],
            "groups_total": int(sum(r["groups_total"] for r in results)),
        },
        "winding": winding if winding is not None else needs_block("no storm tracks given", "storms.jsonl"),
        "caveats": list(braid_mod.CAVEATS),
        "needs_missing": [],
    }
    panel["endpoints"]["P5"] = endpoint_entry(
        "P5",
        value=float(np.mean(vals)) if vals else None,
        blocks=blocks if vals else None,
        needs=[] if vals else ["no group of 8 or more survived a sub-window"],
    )
    panel["header"]["versions"]["braid"] = braid_mod.BRAID_VERSION
    panel["header"]["flags"]["needs_missing"] = sorted(
        {n for k in ("A", "B", "C") for n in panel[k].get("needs_missing", [])}
    )


def fill_a(
    panel: dict[str, Any],
    modes: Sequence[dict[str, Any]] = (),
    activity: dict[str, Any] | None = None,
) -> None:
    """Block A: the activity adapter's result (P1, run-level blocks) and the MODES results of the windows
    (P2 novelty per block, P3 p90 channel count per block; both exploratory, outside Holm)."""
    if not modes and activity is None:
        return
    blk: dict[str, Any] = {"status": "ok", "needs_missing": []}
    miss: list[str] = []
    if activity is not None:
        ep = activity["endpoint"]
        blk["activity"] = {k: v for k, v in activity.items() if k != "caveats"}
        vals = [b for b in ep["blocks"] if b is not None]
        panel["endpoints"]["P1"] = endpoint_entry(
            "P1",
            value=float(np.mean(vals)) if vals else None,
            blocks=ep["blocks"] if vals else None,
            needs=[] if vals else ["no valid activity window"],
        )
        panel["endpoints"]["P1"]["space"] = ep["space"]
        panel["header"]["versions"]["activity"] = aa.ACTIVITY_VERSION
        blk["caveats"] = list(aa.CAVEATS)
    else:
        miss.append("activity (run events)")
    if modes:
        blk["modes"] = {"windows": [{k: v for k, v in m.items() if k != "caveats"} for m in modes]}
        blk["modes_caveats"] = list(modes_mod.CAVEATS)
        for name, getter in (
            ("P2", lambda m: m["novelty"]["per_block"]),
            ("P3", lambda m: m["complexity"]["p90"]),
        ):
            blocks = [b for m in modes for b in getter(m)]
            vals = [b for b in blocks if b is not None]
            panel["endpoints"][name] = endpoint_entry(
                name,
                value=float(np.mean(vals)) if vals else None,
                blocks=blocks if vals else None,
                needs=[] if vals else ["no block"],
            )
        panel["header"]["versions"]["modes"] = modes_mod.MODES_VERSION
    else:
        miss.append("MODES (frames and a reference)")
    blk["needs_missing"] = miss
    panel["A"] = blk
    panel["header"]["flags"]["needs_missing"] = sorted(
        {n for k in ("A", "B", "C") for n in panel[k].get("needs_missing", [])}
    )


class SpaceMismatch(ValueError):
    """P1 values of panels computed in different activity spaces."""


def check_p1_spaces(panels: Sequence[dict[str, Any]]) -> None:
    """Refuse to compare P1 across activity spaces (the cohort and marker spaces are different quantities):
    every panel with a P1 value must carry the same ``space``."""
    spaces = {
        p["endpoints"]["P1"].get("space")
        for p in panels
        if p["endpoints"].get("P1", {}).get("status") == "ok"
    }
    if len(spaces) > 1:
        raise SpaceMismatch(f"P1 computed in different activity spaces: {sorted(map(str, spaces))}")
