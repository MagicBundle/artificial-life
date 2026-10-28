"""``python -m alife.analysis.emergence <command>``: the CLI shell of the emergence suite (DESIGN §44.9).

Rules: the run directory is read only, every output goes under ``--out`` (default
``analysis_cache/emergence/<run>/``), writes are atomic, and a command that needs a block that has not landed
says so and exits 2. ``record`` is the only command that simulates (it imports MLX lazily).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from alife.analysis.emergence import (
    braid,
    cli_blocks,
    compare,
    epr,
    frames,
    panel,
    panel_blocks,
    prereg,
    scorecards,
    synth,
    winding,
)

LATER = {
    "epr": "block B (E3)",
    "braid": "block C (E4)",
    "winding": "block C (E4)",
    "modes": "block A (E5)",
    "activity": "block A (E5)",
}


def _window(text: str) -> tuple[int, int]:
    a, _, b = text.partition(":")
    if not (a.isdigit() and b.isdigit()) or int(a) >= int(b):
        raise argparse.ArgumentTypeError(f"window {text!r} is not A:B with A < B")
    return int(a), int(b)


def _out(a: argparse.Namespace, run: Path | None) -> Path:
    return Path(a.out) if a.out else Path("analysis_cache/emergence") / (run.name if run else "synth")


def _git_rev() -> str | None:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10, check=False
        )
        return r.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def cmd_record(a: argparse.Namespace) -> int:
    from alife.analysis.emergence import record  # lazy: the only MLX import of the package

    t0, t1 = a.window
    try:
        record.record_window(
            a.run,
            t0,
            t1,
            _out(a, a.run),
            tau=a.tau,
            fine_tau=a.fine_tau,
            fine_groups=a.fine_groups,
            fine_max=a.fine_max,
            verify=not a.no_verify,
            max_lead=a.max_lead,
            log=lambda m: print(m, flush=True),
        )
    except record.RecordMismatch as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except (record.RecordError, FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


def cmd_panel(a: argparse.Namespace) -> int:
    wins = a.windows or []
    frames_list = [frames.read_frames(f) for f in a.frames or []]
    fmeta = [fr.meta for fr in frames_list]
    h = prereg.load_prereg(a.prereg)["sha256"] if a.prereg else None
    cards = []
    feats = [x for x in (a.scorecards or "").split(",") if x]
    if feats:
        from alife.runio import RunDir

        rd = RunDir.open(a.run)
        cards = [
            scorecards.evaluate(f, rd, wins[0] if wins else (0, 0), None, {}, _out(a, a.run)) for f in feats
        ]
    p = panel.build_panel(
        a.run,
        wins,
        arm=a.arm,
        prereg_hash=h,
        population_ref=a.population_ref,
        frames_meta=fmeta,
        scorecards=cards,
    )
    if frames_list:
        panel_blocks.fill_b(p, [epr.analyse_window(fr) for fr in frames_list])
        wind = None
        if a.storms:
            wind = winding.analyse(winding.read_storms_jsonl(a.storms), frames_list[0].size)
        panel_blocks.fill_c(p, [braid.analyse_blocks(fr) for fr in frames_list], wind)
    pj, pm = panel.write_panel(_out(a, a.run), p)
    print(f"wrote {pj} and {pm}")
    return 0


def cmd_compare(a: argparse.Namespace) -> int:
    pr = prereg.load_prereg(a.prereg)
    panels = [panel.read_panel(p) for p in a.panels]
    try:
        cmp = compare.compare_panels(panels, pr, exploratory=a.exploratory)
    except (prereg.PreregError, compare.CompareError, panel.PanelError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    cj, cm = compare.write_compare(Path(a.out or "analysis_cache/emergence/compare"), cmp)
    for f in cmp["order"]:
        print(f"{f}: {cmp['features'][f]['call']}")
    print(f"wrote {cj} and {cm}")
    return 0


def cmd_prereg(a: argparse.Namespace) -> int:
    d = prereg.default_prereg(code_git=_git_rev())
    d["frozen"] = bool(a.freeze)
    print(f"wrote {a.out} sha256 {prereg.write_prereg(Path(a.out), d)} frozen={d['frozen']}")
    return 0


def cmd_scorecards(a: argparse.Namespace) -> int:
    from alife.runio import RunDir

    rd = RunDir.open(a.run)
    for feat in [x for x in a.features.split(",") if x]:
        card = scorecards.SCORECARDS[feat]() if feat in scorecards.SCORECARDS else None
        need = card.available(rd) if card else [f"unknown feature {feat!r}"]
        print(f"{feat}: {'available' if not need else 'not evaluated: ' + '; '.join(need)}")
    return 0


def cmd_synth(a: argparse.Namespace) -> int:
    eff = {}
    for item in a.effect or []:
        k, _, v = item.partition("=")
        f, _, e = k.partition(":")
        eff[(f, e)] = float(v)
    rng = np.random.default_rng(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ps = synth.synth_panels(rng, effects=eff, prereg_hash=a.prereg_hash)
    for p in ps:
        h = p["header"]
        (out / f"panel_{h['arm']}_s{h['seed']}.json").write_text(
            json.dumps(p, indent=1) + "\n", encoding="utf-8"
        )
    print(f"wrote {len(ps)} synthetic panels to {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m alife.analysis.emergence", description=__doc__.split("\n")[0]
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser(
        "record", help="record a window of dense frames (re-simulates, certified against the run)"
    )
    p.add_argument("run", type=Path)
    p.add_argument("--window", type=_window, required=True, metavar="T0:T1")
    p.add_argument("--out", type=Path)
    p.add_argument("--tau", type=int, default=100)
    p.add_argument("--fine-tau", type=int, default=5)
    p.add_argument("--fine-groups", type=int, default=8)
    p.add_argument("--fine-max", type=int, default=250)
    p.add_argument("--max-lead", type=int, default=50_000)
    p.add_argument(
        "--no-verify", action="store_true", help="skip the certification against the run (uncertified frames)"
    )
    p.set_defaults(fn=cmd_record)
    p = sub.add_parser("panel", help="write the panel of a run")
    p.add_argument("run", type=Path)
    p.add_argument("--windows", type=_window, nargs="*")
    p.add_argument("--frames", type=Path, nargs="*")
    p.add_argument("--arm", default="unlabelled")
    p.add_argument("--prereg", type=Path)
    p.add_argument("--scorecards", default="")
    p.add_argument("--population-ref", type=float)
    p.add_argument("--storms", type=Path, help="observers/storms.jsonl for the (exploratory) winding table")
    p.add_argument("--out", type=Path)
    p.set_defaults(fn=cmd_panel)
    p = sub.add_parser("compare", help="compare arms from panels under a pre-registration")
    p.add_argument("panels", type=Path, nargs="+")
    p.add_argument("--prereg", type=Path, required=True)
    p.add_argument("--exploratory", action="store_true")
    p.add_argument("--out", type=Path)
    p.set_defaults(fn=cmd_compare)
    p = sub.add_parser("prereg", help="write the default pre-registration (unfrozen unless --freeze)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--freeze", action="store_true")
    p.set_defaults(fn=cmd_prereg)
    p = sub.add_parser("scorecards", help="which scorecards can be computed for a run")
    p.add_argument("run", type=Path)
    p.add_argument("--features", default=",".join(scorecards.FEATURES))
    p.set_defaults(fn=cmd_scorecards)
    p = sub.add_parser("synth", help="write synthetic panels with planted effects")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--effect", action="append", metavar="FEATURE:ENDPOINT=D")
    p.add_argument("--prereg-hash")
    p.set_defaults(fn=cmd_synth)
    for name, (add, fn) in cli_blocks.COMMANDS.items():
        q = sub.add_parser(name, help=f"block command {name} (DESIGN §44)")
        add(q)
        q.set_defaults(fn=fn)
    for name, where in LATER.items():
        if name in cli_blocks.COMMANDS:
            continue
        q = sub.add_parser(name, help=f"not yet implemented: {where}")
        q.add_argument("args", nargs="*")
        q.set_defaults(
            fn=lambda a, w=where, n=name: (print(f"error: {n} lands with {w}", file=sys.stderr), 2)[1]
        )
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    return int(a.fn(a))


if __name__ == "__main__":
    raise SystemExit(main())
