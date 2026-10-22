"""Write ``tests/data/play20k_logs/``: the logs of the Mac-day play of planet ``p0`` (G3a, 20,000 steps from
450,000 on a fork of ``64k3d_s0``), reduced to the keys the G3b chronicle, alerts and report read. They are
the regression pin of the kin-thread rule and the storm persistence rule (G3b spec §2.3, §2.4, §2.6).

    python tests/data/make_play20k_logs.py PLANET_DIR [OUT_DIR]

``PLANET_DIR`` holds ``observers/social.jsonl``, ``observers/storms.jsonl`` and ``metrics.jsonl`` of the
played world. The reduction drops what no G3b reader looks at (colony rows and tables, relations, temperament,
territory, the ledgers and most metrics) and rounds nothing: every kept value is the logged one. The one
addition is the ``initial`` flag on the tracker events of the first tick of the layer (step 450,025), which a
fresh layer of G3b writes itself (``StormLayer.initial_pending``); the Mac-day files predate it.

Files: ``social.jsonl`` (82 records, steps 450,250 to 470,500; the pins use the 80 of steps up to 470,000),
``storms.jsonl`` (820 ticks), ``metrics.jsonl`` (85 rows), ``director.jsonl`` (the one move of the window).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
KEEP_METRICS = (
    "step", "steps", "n_alive", "n_alive_min", "births", "deaths_starved", "deaths_killed", "deaths_aged",
    "deaths_catastrophe", "deaths_storm", "clades_alive", "top_clade_share", "env",
)  # fmt: skip


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def social(rec: dict) -> dict:
    kin = rec["kin_groups"]
    ev = rec["events"]
    rows = [i for i, f in enumerate(ev["family"]) if f == "kin_group"]
    houses = rec["houses"]
    return {
        "step": rec["step"],
        "origin": rec["origin"],
        "thresholds": rec["thresholds"],
        "summary": {"houses": rec["summary"]["houses"], "kin_groups": rec["summary"]["kin_groups"]},
        "kin_groups": {k: kin[k] for k in ("id", "name", "size", "tag", "x", "y", "z_mean")},
        "events": {
            k: [ev[k][i] for i in rows] for k in ("step", "family", "kind", "id", "size", "other", "overlap")
        },
        "houses": {k: houses[k] for k in ("id", "name", "size")},
        "house_events": {
            "founded": rec["house_events"]["founded"],
            "extinct": rec["house_events"]["extinct"],
        },
    }


def storms(line: dict, first: bool) -> dict:
    events = [{**e, "initial": True} if first else e for e in line["events"]]
    return {
        "step": line["step"],
        "storms": [{k: s[k] for k in ("id", "x", "y", "r", "zeta")} for s in line["storms"]],
        "events": events,
    }


def metrics(row: dict) -> dict:
    out = {k: row[k] for k in KEEP_METRICS if k in row}
    out["observers"] = {"alive": row["observers"]["alive"]}
    return out


def dump(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows), encoding="utf-8")


def main(argv: list[str]) -> int:
    src = Path(argv[0])
    out = Path(argv[1]) if len(argv) > 1 else HERE / "play20k_logs"
    out.mkdir(parents=True, exist_ok=True)
    dump(out / "social.jsonl", [social(r) for r in read(src / "observers" / "social.jsonl")])
    ticks = read(src / "observers" / "storms.jsonl")
    dump(out / "storms.jsonl", [storms(t, i == 0) for i, t in enumerate(ticks)])
    dump(out / "metrics.jsonl", [metrics(r) for r in read(src / "metrics.jsonl")])
    entries = [r for r in read(src / "director.jsonl") if "step_applied" in r]
    dump(out / "director.jsonl", entries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
