"""Build ``tests/data/report_e2e/``: a real observed director fork, the files the world report reads, and the
reports the template build writes for it (``golden.json``).

A tiny source run (``tiny`` preset, float32 weights, 2,000 steps) is forked at step 2,000 with
``--director replay`` and the hand-written ``report_e2e/director.jsonl`` (a rule's regeneration pulse with its
revert, an immigration pulse, seasons, a fertility shift and a drift change: every lever kind the block lists)
and ``game.observe=true``, for 8,000 more steps on the CPU. The engine writes the rows' ``env`` blocks, so the
test that compares them with the report's fold compares the engine's own output. The kept files are the config,
``meta.json`` without host fields, ``metrics.jsonl`` pruned to the fields the report reads, the director log,
``observers/social.jsonl`` and each sidecar's ``meta`` member only.

    PYTHONPATH=tests python tests/data/make_report_e2e.py [OUT_DIR] [--fork RUN_DIR] [--golden-only]

Slow (two simulations on the CPU, a few minutes) unless ``--fork`` names a fork you already made with the same
arguments, or ``--golden-only`` rebuilds ``golden.json`` from the committed files (the right thing after a change
of the report, which does not change the engine's output). Run it once, after a change to the report's output,
and read the diff. The tests read the committed files and never simulate."""

# ruff: noqa: E501  (expected report texts are long lines)

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
from alife.analysis import world_report  # noqa: E402

FROM, TO, CHAPTER = 2500, 10000, 2000
ROW_KEYS = (
    "step", "steps", "n_alive_min", "births", "immigrants", "deaths_starved", "deaths_killed", "deaths_aged",
    "deaths_catastrophe", "clades_alive", "top_clade_share", "env", "observers",
)  # fmt: skip
COMMON = ["--set", "brain.weight_dtype=float32", "--metrics-every", "100", "--checkpoint-every", "2000",
          "--set", "run.heavy_metrics_every=500"]  # fmt: skip


def run(args: list[str]) -> None:
    code = cli.main([*args, "--quiet"])
    if code != 0:
        raise SystemExit(f"alife {' '.join(args)} exited {code}")


def simulate(tmp: Path, log: Path) -> Path:
    src, fork = tmp / "src", tmp / "fork"
    run(["run", "--preset", "tiny", *COMMON, "--steps", "2000", "--out", str(src)])
    run(["run", "--init-from", f"{src}:2000", "--director", "replay", "--replay-log", str(log),
         "--set", "game.observe=true", *COMMON, "--steps", "8000", "--out", str(fork)])  # fmt: skip
    return fork


def keep(fork: Path, dst: Path, log: Path) -> None:
    (dst / "observers").mkdir(parents=True, exist_ok=True)
    shutil.copy(fork / "config.json", dst / "config.json")
    meta = json.loads((fork / "meta.json").read_text())
    (dst / "meta.json").write_text(
        json.dumps(
            {k: meta[k] for k in ("format_version", "seed", "config_hash", "start_step", "status")}, indent=2
        )
        + "\n"
    )
    rows = [json.loads(x) for x in (fork / "metrics.jsonl").read_text().splitlines() if x.strip()]
    with (dst / "metrics.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({k: r[k] for k in ROW_KEYS if k in r}, separators=(",", ":")) + "\n")
    if log.resolve() != (dst / "director.jsonl").resolve():
        shutil.copy(log, dst / "director.jsonl")
    shutil.copy(fork / "observers" / "social.jsonl", dst / "observers" / "social.jsonl")
    for p in sorted((fork / "observers").glob("step_*.npz")):
        with np.load(p, allow_pickle=False) as z:
            np.savez(dst / "observers" / p.name, meta=z["meta"])


def golden(dst: Path) -> list[dict]:
    """The template build over the kept files, as the CLI writes it, reduced to what the test pins."""
    with tempfile.TemporaryDirectory() as t:
        out = Path(t) / "reports.jsonl"
        code = world_report.main(
            [
                "build",
                str(dst),
                "--from",
                str(FROM),
                "--to",
                str(TO),
                "--chapter",
                str(CHAPTER),
                "--out",
                str(out),
            ]
        )
        if code != 0:
            raise SystemExit(f"build exited {code}")
        recs = [json.loads(x) for x in out.read_text().splitlines()]
    return [
        {
            "chapter": [r["chapter_start"], r["step"]],
            "facts_digest": r["facts_digest"],
            "candidate_set_digest": r["candidate_set_digest"],
            "headline": r["report"]["headline"]["text"],
            "lead": r["report"]["lead"],
            "body": [b["text"] for b in r["report"]["body"]],
            "interventions": [i["text"] for i in r["report"]["interventions"]],
            "ticker": r["ticker"],
            "excluded": {k: v for k, v in r["facts"]["excluded"].items() if v},
        }
        for r in recs
    ]


def main(out: Path, fork: Path | None = None, golden_only: bool = False) -> None:
    dst = out / "report_e2e"
    log = dst / "director.jsonl"
    if not golden_only:
        with tempfile.TemporaryDirectory() as tmp:
            keep(fork or simulate(Path(tmp), log), dst, log)
    (dst / "golden.json").write_text(
        json.dumps(golden(dst), indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {dst}")


if __name__ == "__main__":
    args = sys.argv[1:]
    fork_dir = Path(args[args.index("--fork") + 1]) if "--fork" in args else None
    rest = [
        a
        for i, a in enumerate(args)
        if a not in ("--fork", "--golden-only") and (i == 0 or args[i - 1] != "--fork")
    ]
    main(Path(rest[0]) if rest else Path(__file__).parent, fork_dir, "--golden-only" in args)
