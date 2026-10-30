"""``python -m alife.analysis.voice <command>``: the voice channel's log and meaning test (DESIGN §39.14).

* ``log RUN --from STEP --steps N --out DIR``: replay a window with the observers on, write ``tables.npz``,
  ``summary.json`` and ``summary.md`` (the run is read only; the replay is verified against its metrics).
* ``summarize DIR``: rebuild the summary of a log from its tables, without simulating.
* ``drift DIR_A DIR_B --out FILE``: the dialect drift between two logged windows of one world.
* ``prereg --out FILE [--freeze]``: write the pre-registration of the meaning test.
* ``meaning --arm ARM:SEED=RUN[,LOGDIR] ... --out DIR [--prereg FILE]``: the verdict of section 39.9.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from alife.analysis.emergence import prereg as pr
from alife.analysis.voice import meaning, spec, summary
from alife.analysis.voice.tables import Tables


def jsonable(x: Any) -> Any:
    """Plain JSON: numpy scalars and arrays to Python, non-finite floats to None, keys to strings."""
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, float | np.floating):
        return float(x) if math.isfinite(float(x)) else None
    return x


def _write(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(jsonable(obj), indent=1, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    tmp.replace(path)


def _git_rev() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def cmd_log(a: argparse.Namespace) -> int:
    from alife.analysis.voice import replay  # MLX is imported here only (the other commands are numpy)

    out = Path(a.out)
    tab = replay.record(
        a.run,
        start=a.start,
        steps=a.steps,
        out=out,
        states_dir=a.states,
        every=a.every,
        rows=a.rows,
        seed=a.seed,
        cf_every=a.cf_every,
        groups=a.groups,
        verify=not a.no_verify,
        log=lambda m: print(m, flush=True),
    )
    return _summarize(tab, out, a)


def _summarize(tab: Tables, out: Path, a: argparse.Namespace) -> int:
    window = tuple(tab.meta["window"])
    s = summary.build(
        tab,
        window,  # type: ignore[arg-type]
        n_perm=a.perms,
        n_perm_block=spec.N_PERM_BLOCK,
        top=a.top,
        seed=a.seed,
        with_tests=not a.no_tests,
        log=lambda m: print(m, flush=True),
    )
    _write(out / "summary.json", s)
    (out / "summary.md").write_text(summary.to_markdown(s), encoding="utf-8")
    print(f"wrote {out / 'summary.json'} and {out / 'summary.md'}")
    return 0


def cmd_summarize(a: argparse.Namespace) -> int:
    out = Path(a.dir)
    return _summarize(Tables.load(out / "tables.npz"), out, a)


def cmd_drift(a: argparse.Namespace) -> int:
    sa = json.loads((Path(a.a) / "summary.json").read_text(encoding="utf-8"))
    sb = json.loads((Path(a.b) / "summary.json").read_text(encoding="utf-8"))
    res = summary.compare_windows(sa, sb)
    _write(Path(a.out), res)
    print(json.dumps(jsonable(res), indent=1))
    return 0


def cmd_prereg(a: argparse.Namespace) -> int:
    d = spec.default_prereg(code_git=_git_rev())
    d["frozen"] = bool(a.freeze)
    print(f"wrote {a.out} sha256 {pr.write_prereg(Path(a.out), d)} frozen={d['frozen']}")
    return 0


def _parse_arm(text: str) -> tuple[str, int, Path, Path | None]:
    head, _, tail = text.partition("=")
    arm, _, seed = head.partition(":")
    run, _, logdir = tail.partition(",")
    if arm not in spec.ARMS or not seed.isdigit() or not run:
        raise argparse.ArgumentTypeError(
            f"--arm wants ARM:SEED=RUN[,LOGDIR] with ARM in {spec.ARMS}, got {text!r}"
        )
    return arm, int(seed), Path(run), Path(logdir) if logdir else None


def cmd_meaning(a: argparse.Namespace) -> int:
    prereg = None
    if a.prereg:
        try:
            prereg = pr.load_prereg(Path(a.prereg))
        except pr.PreregError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
    rules = meaning.Rules(prereg)
    if not rules.confirmatory and not a.exploratory:
        print(
            "error: no frozen pre-registration (--prereg FILE); pass --exploratory to run without",
            file=sys.stderr,
        )
        return 2
    arms: dict[str, dict[int, meaning.ArmRun]] = {}
    tests: dict[tuple[str, int], dict[str, Any]] = {}
    for arm, seed, run, logdir in a.arm:
        logdir = logdir or Path(a.logs) / run.name
        arms.setdefault(arm, {})[seed] = meaning.ArmRun(arm, seed, run, logdir / "tables.npz", rules)
        sj = logdir / "summary.json"
        if sj.is_file():
            t = json.loads(sj.read_text(encoding="utf-8")).get("tests")
            if t:
                tests[(arm, seed)] = t
    try:
        res = meaning.run_meaning(arms, rules, tests, log=lambda m: print(m, flush=True))
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    out = Path(a.out)
    _write(out / "meaning.json", res)
    (out / "meaning.md").write_text(meaning.to_markdown(res), encoding="utf-8")
    print(f"{res['decision']['verdict']}: {res['decision']['detail']}")
    print(f"wrote {out / 'meaning.json'} and {out / 'meaning.md'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m alife.analysis.voice", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--perms", type=int, default=spec.N_PERM, help="permutations of a pooled test")
        p.add_argument("--top", type=int, default=spec.TOP_LINEAGES, help="lineages reported one by one")
        p.add_argument("--seed", type=int, default=spec.SEED)
        p.add_argument("--no-tests", action="store_true", help="skip the information tests (tables only)")

    p = sub.add_parser("log", help="replay a window with the observers on and write the tables and summary")
    p.add_argument("run", type=Path)
    p.add_argument("--from", dest="start", type=int, required=True, help="checkpoint step to start from")
    p.add_argument("--steps", type=int, default=spec.WINDOW)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--states", type=Path, help="directory of cached states (analysis.resim)")
    p.add_argument("--every", type=int, default=spec.SAMPLE_EVERY)
    p.add_argument("--rows", type=int, default=spec.SAMPLE_ROWS)
    p.add_argument("--cf-every", type=int, default=4, help="counterfactual every Nth sample (0 = never)")
    p.add_argument("--groups", type=int, default=spec.LINEAGE_GROUPS, help="kin groups (k-means of the tags)")
    p.add_argument(
        "--no-verify", action="store_true", help="skip the check against metrics.jsonl and the checkpoint"
    )
    common(p)
    p.set_defaults(fn=cmd_log)

    p = sub.add_parser("summarize", help="rebuild the summary of a log from its tables")
    p.add_argument("dir", type=Path)
    common(p)
    p.set_defaults(fn=cmd_summarize)

    p = sub.add_parser("drift", help="dialect drift between two logged windows")
    p.add_argument("a", type=Path)
    p.add_argument("b", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(fn=cmd_drift)

    p = sub.add_parser("prereg", help="write the pre-registration of the meaning test")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--freeze", action="store_true")
    p.set_defaults(fn=cmd_prereg)

    p = sub.add_parser("meaning", help="the verdict of DESIGN 39.9 from the none, deaf and decoy arms")
    p.add_argument("--arm", type=_parse_arm, action="append", required=True, metavar="ARM:SEED=RUN[,LOGDIR]")
    p.add_argument(
        "--logs", type=Path, default=Path("analysis_cache/voice"), help="parent of the log directories"
    )
    p.add_argument("--prereg", type=Path)
    p.add_argument("--exploratory", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(fn=cmd_meaning)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    try:
        return int(a.fn(a))
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
