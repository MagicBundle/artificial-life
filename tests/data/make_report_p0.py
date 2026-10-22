"""Write ``tests/data/report_p0/``: the Mac-day play of planet ``p0`` (G3a, 20,000 steps from 450,000 on a
fork of ``64k3d_s0``) as a run directory the world report can read (G3b spec §5.2a, DESIGN §34.4), with every
file reduced to the keys ``game.report`` reads.

    python tests/data/make_report_p0.py PLANET_DIR [OUT_DIR]

``PLANET_DIR`` holds ``config.json``, ``meta.json``, ``metrics.jsonl``, ``director.jsonl`` and
``observers/{social,storms}.jsonl`` of the played world. Nothing is rounded: every kept value is the logged
one. The additions are the ``initial`` flag on the tracker events of the first tick of the layer (step
450,025), which a fresh layer of G3b writes itself (``StormLayer.initial_pending``; the Mac-day files predate
it), the sidecar files of the checkpoints, reduced to their ``meta`` (the only member the report reads) and no
``origin_row`` in ``meta.json`` (the files predate it: the chapters of the tests start at 450,250 and later).
The social record keeps the columns of its tables the report reads and every row of ``events``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
KEEP_METRICS = (
    "step", "steps", "n_alive", "n_alive_min", "births", "immigrants", "deaths_starved", "deaths_killed",
    "deaths_aged", "deaths_catastrophe", "deaths_storm", "clades_alive", "top_clade_share", "env",
)  # fmt: skip
KIN_COLUMNS = ("id", "name", "size", "tag", "x", "y", "z_mean", "births", "deaths")
EVENT_COLUMNS = ("step", "family", "kind", "id", "size", "other", "overlap")
KEEP_META = ("format_version", "seed", "config_hash", "dims", "start_step", "weather_spinup")


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def dump(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows), encoding="utf-8")


def cols(table: dict, names: tuple[str, ...]) -> dict:
    return {k: table[k] for k in names}


def social(rec: dict) -> dict:
    return {
        "step": rec["step"],
        "social_version": rec["social_version"],
        "social_digest": rec["social_digest"],
        "origin": rec["origin"],
        "window_start": rec["window_start"],
        "thresholds": rec["thresholds"],
        "summary": rec["summary"],
        "houses": cols(rec["houses"], ("id", "name", "size")),
        "house_events": rec["house_events"],
        "kin_groups": cols(rec["kin_groups"], KIN_COLUMNS),
        "events": cols(rec["events"], EVENT_COLUMNS),
        "relations": cols(rec["relations"], ("a", "b", "attacks", "kills", "distinct")),
        "relations_dropped": rec["relations_dropped"],
    }


def metrics(row: dict) -> dict:
    out = {k: row[k] for k in KEEP_METRICS if k in row}
    obs = row["observers"]
    keep = {"alive": obs["alive"], "intent_version": obs["intent_version"], "social": obs["social"]}
    out["observers"] = keep
    return out


def storms(line: dict, first: bool) -> dict:
    events = [{**e, "initial": True} if first else e for e in line["events"]]
    return {
        "step": line["step"],
        "storms": [{k: s[k] for k in ("id", "x", "y", "r", "zeta")} for s in line["storms"]],
        "events": events,
    }


def main(argv: list[str]) -> int:
    src = Path(argv[0])
    out = Path(argv[1]) if len(argv) > 1 else HERE / "report_p0"
    (out / "observers").mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text((src / "config.json").read_text(encoding="utf-8"), encoding="utf-8")
    meta = json.loads((src / "meta.json").read_text(encoding="utf-8"))
    kept = {k: meta[k] for k in KEEP_META if k in meta}
    (out / "meta.json").write_text(json.dumps({**kept, "status": "done"}, indent=1) + "\n", encoding="utf-8")
    dump(out / "metrics.jsonl", [metrics(r) for r in read(src / "metrics.jsonl")])
    dump(out / "observers" / "social.jsonl", [social(r) for r in read(src / "observers" / "social.jsonl")])
    ticks = read(src / "observers" / "storms.jsonl")
    dump(out / "observers" / "storms.jsonl", [storms(t, i == 0) for i, t in enumerate(ticks)])
    log = (src / "director.jsonl").read_text(encoding="utf-8")
    (out / "director.jsonl").write_text(log, encoding="utf-8")
    for step in (450000, 468000):
        z = np.load(src / "observers" / f"step_{step:010d}.npz", allow_pickle=False)
        np.savez(out / "observers" / f"step_{step:010d}.npz", meta=z["meta"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
