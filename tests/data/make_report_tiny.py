"""Build ``tests/data/report_tiny/`` and ``report_tiny_resumed/``: a tiny observed run (the ``tiny`` preset with
float32 weights, ``game.observe``, 3,000 steps, rows every 100, snapshots every 250, checkpoints every 1,500) and
the same run stopped at 2,000 and resumed from its middle checkpoint, as the files the world report reads
(config, ``meta.json`` without the host fields, ``metrics.jsonl``, ``observers/social.jsonl`` and the sidecar's
``meta`` member only). ``golden.json`` holds the sheet, the candidates, the template report and the ticker of the
chapters (1,000, 2,000] and (2,000, 3,000] of the first run.

    PYTHONPATH=tests python tests/data/make_report_tiny.py [OUT_DIR]

Slow (two simulations on the CPU, about a minute each); run it once, after a change to the report's output, and
read the diff. The tests read the committed files and never simulate."""

# ruff: noqa: E501  (expected report texts and fixture tables are long lines)

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np

mx.set_default_device(mx.cpu)

from alife import cli  # noqa: E402
from alife.game import report  # noqa: E402

CHAPTERS = ((1000, 2000), (2000, 3000))


def run(args: list[str]) -> None:
    code = cli.main([*args, "--quiet"])
    if code != 0:
        raise SystemExit(f"alife {' '.join(args)} exited {code}")


def keep(src: Path, dst: Path) -> None:
    """The files the report reads, without host fields and arrays."""
    if dst.exists():
        shutil.rmtree(dst)
    (dst / "observers").mkdir(parents=True)
    shutil.copy(src / "config.json", dst / "config.json")
    meta = json.loads((src / "meta.json").read_text())
    (dst / "meta.json").write_text(
        json.dumps(
            {k: meta[k] for k in ("format_version", "seed", "config_hash", "start_step", "status")}, indent=2
        )
        + "\n"
    )
    shutil.copy(src / "metrics.jsonl", dst / "metrics.jsonl")
    shutil.copy(src / "observers" / "social.jsonl", dst / "observers" / "social.jsonl")
    for p in sorted((src / "observers").glob("step_*.npz")):
        with np.load(p, allow_pickle=False) as z:
            np.savez(dst / "observers" / p.name, meta=z["meta"])


def main(out: Path) -> None:
    base = ["--preset", "tiny", "--set", "brain.weight_dtype=float32", "--set", "game.observe=true",
            "--metrics-every", "100", "--checkpoint-every", "1500", "--set", "run.heavy_metrics_every=500"]  # fmt: skip
    with tempfile.TemporaryDirectory() as tmp:
        a, b = Path(tmp) / "a", Path(tmp) / "b"
        run(["run", *base, "--steps", "3000", "--out", str(a)])
        run(["run", *base, "--steps", "2000", "--out", str(b)])
        run(["resume", str(b), "--checkpoint", "1500", "--until", "3000"])
        keep(a, out / "report_tiny")
        keep(b, out / "report_tiny_resumed")
    view = report.load_run(out / "report_tiny")
    golden = []
    for lo, hi in CHAPTERS:
        sheet = report.build_sheet(view, lo, hi)
        body, heads, block, excluded = report.candidates(sheet)
        golden.append(
            {
                "chapter": [lo, hi],
                "facts_digest": report.facts_digest(sheet),
                "sheet": sheet,
                "candidates": [
                    {k: c[k] for k in ("id", "template", "cites", "text", "mandatory")} for c in body
                ],
                "headlines": [{k: h[k] for k in ("id", "template", "text")} for h in heads],
                "block": [b_["text"] for b_ in block],
                "excluded": excluded,
                "report": report.template_report(sheet),
                "ticker": report.ticker(sheet),
            }
        )
    (out / "report_tiny" / "golden.json").write_text(
        json.dumps(golden, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {out}/report_tiny and report_tiny_resumed")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent)
