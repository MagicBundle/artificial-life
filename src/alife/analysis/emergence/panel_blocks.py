"""Fill the analysis blocks of a built panel (DESIGN §44.7): B here, C and A with their stages.

Each ``fill_*`` mutates a panel made by ``panel.build_panel`` / ``panel.new_panel``: it sets the block, the
endpoint rows it owns and the header flags, leaving everything else as it was. Absent data stays
``needs_missing``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from alife.analysis.emergence import braid as braid_mod
from alife.analysis.emergence.epr import CAVEATS, EPR_VERSION
from alife.analysis.emergence.panel import endpoint_entry, needs_block


def fill_b(panel: dict[str, Any], results: Sequence[dict[str, Any]]) -> None:
    """Block B from the ``epr.analyse_window`` results of the recorded windows (in window order).

    The P4 endpoint is the own-motion rate; its block values are the 8 time-block rates of every window, in
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
            {
                k: r[k]
                for k in ("cell", "tau", "own", "raw", "ambient", "activity", "stationarity", "vertical")
            }
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
    winding table (descriptive). The P5 block values are the braid contrast per crossing of every window's
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
