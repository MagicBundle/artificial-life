"""Build ``tests/data/report_corpus.json``, the regression corpus of the world report's lexicon (docs/DESIGN.md §30,
R6): the sentences of the director's 131 model journals (three runs, read-only) with the owner-side hand labels of
the sentences that credit an intervention (``credit``), presuppose that it works or was needed (``presume``), use a
temporal connective (``temporal``) or another causal word (``causal_other``), and the 61 ``reason`` strings of the
model's log entries. Nothing here is shown to a player: the test requires that the checker rejects the credit and
presupposition sentences and every reason string, and pins the per-list counts.

    python tests/data/make_report_corpus.py /path/to/runs tests/data/report_corpus.json

Run once; the tests read only the JSON (never ``runs/``). The sentence ids are ``<journal index>.<sentence index>``
in the order of the runs ``eval_s0_C``, ``eval_s1_C``, ``64k3d_s0_dir``."""

# ruff: noqa: E501  (expected report texts and fixture tables are long lines)

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

RUNS = ("eval_s0_C", "eval_s1_C", "64k3d_s0_dir")
LABELS = {
    "credit": "5.0 9.0 11.0 13.0 16.0 18.0 20.0 22.0 26.1 30.1 32.0 38.0 50.0 54.0 60.0 69.0 80.1 83.0 84.0 84.1 102.0 104.0 123.0 127.2",
    "presume": "16.1 24.0 27.0 28.1 37.0 40.0 41.1 46.0 47.1 48.0 52.0 64.1 71.0 71.1 76.1 81.0 87.0 89.0 91.1 96.1 105.1 109.0 119.1 124.1",
    "temporal": "7.0 21.0 26.0 42.0 44.0 75.0 77.0 85.0 93.0 107.0",
    "causal_other": "42.1 78.0 111.0 114.0 28.0",
}  # fmt: skip


def main(runs: Path, out: Path) -> None:
    journals = []
    reasons = []
    for run in RUNS:
        for line in (runs / run / "journal.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("kind") == "llm_call" and r.get("journal"):
                journals.append(r["journal"])
        for line in (runs / run / "director.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r.get("source") == "llm" and r.get("reason"):
                reasons.append(r["reason"])
    label_of = {i: c for c, ids in LABELS.items() for i in ids.split()}
    sentences = []
    for j, text in enumerate(journals):
        parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z(])", text) if s.strip()]
        for k, s in enumerate(parts):
            sentences.append({"id": f"{j}.{k}", "class": label_of.get(f"{j}.{k}", "other"), "text": s})
    out.write_text(
        json.dumps(
            {"journals": len(journals), "sentences": sentences, "reasons": reasons},
            indent=1,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"{len(journals)} journals, {len(sentences)} sentences, {len(reasons)} reasons -> {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
