"""Command handlers of the analysis blocks (E3 to E5) for ``python -m alife.analysis.emergence``.

``COMMANDS`` maps a command name to ``(add_arguments, handler)``; ``__main__`` registers every entry and keeps
the "lands with" stub for the names not yet here.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np

from alife.analysis.emergence import activity_adapter, braid, epr, epr_io, frames, modes, winding
from alife.analysis.emergence.epr_io import _atomic


def _out_dir(a: argparse.Namespace, src: Path) -> Path:
    return Path(a.out) if a.out else Path("analysis_cache/emergence") / src.parent.name


def add_epr(p: argparse.ArgumentParser) -> None:
    p.add_argument("frames", type=Path)
    p.add_argument("--cell", type=float, default=epr.HEADLINE_CELL)
    p.add_argument(
        "--tau", type=int, help="lag in steps (a multiple of the recorded tau; default the recorded)"
    )
    p.add_argument("--n-boot", type=int, default=epr.N_BOOT)
    p.add_argument("--seed", type=int, default=epr.SEED)
    p.add_argument("--out", type=Path)
    p.add_argument("--no-plot", action="store_true")


def cmd_epr(a: argparse.Namespace) -> int:
    try:
        fr = frames.read_frames(a.frames)
    except (frames.FramesError, FileNotFoundError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    rec_tau = round(float(fr.step[1] - fr.step[0])) if fr.n_frames > 1 else 0
    stride = 1
    if a.tau:
        if rec_tau <= 0 or a.tau % rec_tau:
            print(f"error: --tau {a.tau} is not a multiple of the recorded tau {rec_tau}", file=sys.stderr)
            return 2
        stride = a.tau // rec_tau
    res = epr.analyse_window(fr, cell=a.cell, stride=stride, n_boot=a.n_boot, seed=a.seed)
    res["run"] = fr.meta.get("run")
    res["window"] = fr.meta.get("window")
    for p in epr_io.write_outputs(_out_dir(a, a.frames), res, fr, plot=not a.no_plot):
        print(f"wrote {p}")
    o = res["own"]
    print(
        f"own EP {o['rate']} +- {o['se']} nats/organism/1000 steps (z {o['z']}); "
        "a lower bound, the scale curve is in epr_scale.csv"
    )
    return 0


def add_braid(p: argparse.ArgumentParser) -> None:
    p.add_argument("frames", type=Path)
    p.add_argument("--blocks", type=int, default=epr.WINDOW_BLOCKS)
    p.add_argument("--n-surr", type=int, default=braid.N_SURR)
    p.add_argument("--surrogate", choices=("increment", "position"), default="increment")
    p.add_argument("--seed", type=int, default=braid.SEED)
    p.add_argument("--out", type=Path)


def cmd_braid(a: argparse.Namespace) -> int:
    try:
        fr = frames.read_frames(a.frames)
        res = braid.analyse_blocks(fr, n_blocks=a.blocks, n_surr=a.n_surr, kind=a.surrogate, seed=a.seed)
    except (frames.FramesError, FileNotFoundError, OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    res["run"], res["window"] = fr.meta.get("run"), fr.meta.get("window")
    p = _out_dir(a, a.frames) / "braid.json"
    _atomic(p, json.dumps(res, indent=1, sort_keys=True) + "\n")
    print(
        f"wrote {p}: {res['groups_total']} groups, skipped {res['skipped']}; a lower-bound coordination index"
    )
    return 0


def add_winding(p: argparse.ArgumentParser) -> None:
    p.add_argument("storms", type=Path, help="observers/storms.jsonl (or a copy)")
    p.add_argument("--size", type=float, required=True, help="world side L")
    p.add_argument("--frames", type=Path, help="frames file for the organism winding")
    p.add_argument("--out", type=Path)


def cmd_winding(a: argparse.Namespace) -> int:
    try:
        tracks = winding.read_storms_jsonl(a.storms)
        fr = frames.read_frames(a.frames) if a.frames else None
    except (frames.FramesError, FileNotFoundError, OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    res = winding.analyse(tracks, a.size, fr)
    p = (
        Path(a.out) if a.out else Path("analysis_cache/emergence") / a.storms.parent.parent.name
    ) / "winding.json"
    _atomic(p, json.dumps(res, indent=1, sort_keys=True) + "\n")
    print(f"wrote {p}: {res['storms']} storms, {res['co_rotating_pairs']} co-rotating pairs (exploratory)")
    return 0


def add_modes(p: argparse.ArgumentParser) -> None:
    p.add_argument("frames", type=Path, nargs="+")
    p.add_argument("--reference", type=Path, nargs="*", default=[], help="frames files defining the codebook")
    p.add_argument("--codebook", type=Path, help="reuse (or write, with --reference) this codebook.json")
    p.add_argument("--births", type=Path, help="run directory with events/births_*.npz for the edge change")
    p.add_argument("--out", type=Path)


def cmd_modes(a: argparse.Namespace) -> int:
    try:
        out = _out_dir(a, a.frames[0])
        cb_path = a.codebook or out / "codebook.json"
        if a.reference:
            refs = [frames.read_frames(r) for r in a.reference]
            cb = modes.build_codebook(refs)
            modes.write_codebook(cb_path, cb)
            seen = np.zeros(cb.n_codes, bool)
            for r in refs:
                seen |= modes.novelty(
                    modes.persistent(modes.class_counts(r, modes.skeletons(r, cb), cb.n_codes)), seen, 1
                )[1]
        elif cb_path.is_file():
            cb = modes.read_codebook(cb_path)
            seen = np.zeros(cb.n_codes, bool)
        else:
            print("error: give --reference frames or an existing --codebook", file=sys.stderr)
            return 2
        res = []
        for path in a.frames:
            fr = frames.read_frames(path)
            win = fr.meta.get("window") or [int(fr.step[0]), int(fr.step[-1])]
            births = modes.load_births(a.births, win[0], win[1]) if a.births else None
            r = modes.analyse(fr, cb, seen=seen, births=births)
            r["window"] = win
            seen = seen | modes.persistent_classes(fr, cb)
            res.append(r)
    except (frames.FramesError, FileNotFoundError, OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    p = out / "modes.json"
    _atomic(p, json.dumps({"version": modes.MODES_VERSION, "windows": res}, indent=1, sort_keys=True) + "\n")
    print(
        f"wrote {p} and {cb_path}: {len(res)} windows, {cb.n_codes} possible classes (adapted, exploratory)"
    )
    return 0


def add_activity(p: argparse.ArgumentParser) -> None:
    p.add_argument("run", type=Path)
    p.add_argument("--start", type=int, help="fork step (default: the run's first step)")
    p.add_argument("--space", help="activity space (default: cohort_h200k for long arms, else marker_0.01)")
    p.add_argument("--shadows", type=int, default=2)
    p.add_argument("--no-reuse", action="store_true")
    p.add_argument("--out", type=Path)


def cmd_activity(a: argparse.Namespace) -> int:
    out = Path(a.out) if a.out else Path("analysis_cache/emergence") / a.run.name
    try:
        res = activity_adapter.analyse_arm(
            a.run,
            start=a.start,
            st=activity_adapter.settings_for(shadows=a.shadows),
            space=a.space,
            out=out,
            reuse=not a.no_reuse,
            log=lambda m: print(m, flush=True),
        )
    except (FileNotFoundError, OSError, ValueError, KeyError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    p = out / "activity.json"
    _atomic(p, json.dumps(res, indent=1, sort_keys=True, default=str) + "\n")
    ep = res["endpoint"]
    print(f"wrote {p}: class {ep['channon_class']}, space {ep['space']}, {res['timing_s']['total']:.1f} s")
    return 0


COMMANDS: dict[str, tuple[Callable[[argparse.ArgumentParser], None], Callable[[argparse.Namespace], int]]] = {
    "epr": (add_epr, cmd_epr),
    "braid": (add_braid, cmd_braid),
    "winding": (add_winding, cmd_winding),
    "modes": (add_modes, cmd_modes),
    "activity": (add_activity, cmd_activity),
}
