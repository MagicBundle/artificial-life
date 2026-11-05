"""Driver of the diel oracle pilot (DESIGN 49.7, 49.12, 49.14): warm-up, pilot fork, analysis, per seed.

    uv run python scripts/diel_pilot.py --out ROOT [--seeds 21 22 23] [--schedule 2] [--cap one|own]
        [--warm-from DIR] [--dry-run]

Per seed S: ``alife run --preset mini3d --overlay diel --seed S --steps 40000 --out ROOT/warm_S`` then
``alife run --init-from ROOT/warm_S --set diel.oracle=20 [--set diel.oracle_schedule=2] --steps 160000
--out ROOT/pilot_S``. On a fork, ``--steps`` is the number of ADDITIONAL steps (cli._announce: ``step -> step
+ steps``), so the pilot runs 160,000 steps after the warm-up's 40,000. ``--warm-from DIR`` forks the
existing warm-ups ``DIR/warm_S`` instead of running new ones (the warm-up does not depend on the oracle's
schedule). ``--cap own`` (schedule 2 only) keeps the oracle rows' own cap (D1d, 49.15).
Refuses to write into an existing directory. The analysis runs over the pilot directories at the
end: ``alife.analysis.diel_pilot`` (D1b's rule) for schedule 1, ``alife.analysis.diel_pilot2`` (the lineage
fitness rule of 49.14) for schedule 2.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

WARM_STEPS = 40_000
PILOT_STEPS = 160_000
ORACLE_PCT = 20
RULES = {
    1: ("alife.analysis.diel_pilot", "diel_pilot.json"),
    2: ("alife.analysis.diel_pilot2", "diel_pilot2.json"),
}


def commands(
    root: Path, seeds: list[int], schedule: int = 2, warm_from: Path | None = None, cap: str = "one"
) -> list[list[str]]:
    cmds: list[list[str]] = []
    base = ["uv", "run", "alife", "run"]
    for s in seeds:
        warm, pilot = (warm_from or root) / f"warm_{s}", root / f"pilot_{s}"
        if warm_from is None:
            cmds.append(
                [
                    *base,
                    *("--preset", "mini3d", "--overlay", "diel", "--seed", str(s)),
                    *("--steps", str(WARM_STEPS), "--out", str(warm), "--quiet"),
                ]
            )
        init = ["--init-from", str(warm), "--set", f"diel.oracle={ORACLE_PCT}"]
        if schedule != 1:
            init += ["--set", f"diel.oracle_schedule={schedule}"]
        if cap != "one":
            init += ["--set", f"diel.oracle_cap={cap}"]
        cmds.append([*base, *init, "--steps", str(PILOT_STEPS), "--out", str(pilot), "--quiet"])
    module, out = RULES[schedule]
    cmds.append(
        ["uv", "run", "python", "-m", module]
        + [str(root / f"pilot_{s}") for s in seeds]
        + ["--json", str(root / out)]
    )
    return cmds


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, help="output root")
    ap.add_argument("--seeds", nargs="+", type=int, default=[21, 22, 23])
    ap.add_argument("--schedule", type=int, choices=sorted(RULES), default=2, help="diel.oracle_schedule")
    ap.add_argument("--cap", choices=("one", "own"), default="one", help="diel.oracle_cap")
    ap.add_argument("--warm-from", default=None, help="fork the existing warm-ups DIR/warm_S")
    ap.add_argument("--dry-run", action="store_true", help="print the commands only")
    a = ap.parse_args(argv)
    root = Path(a.out)
    warm_from = Path(a.warm_from) if a.warm_from else None
    if a.cap != "one" and a.schedule != 2:
        ap.error("--cap own needs --schedule 2")
    cmds = commands(root, a.seeds, a.schedule, warm_from, a.cap)
    if a.dry_run:
        for c in cmds:
            print(shlex.join(c))
        return 0
    if warm_from is not None:
        missing = [str(warm_from / f"warm_{s}") for s in a.seeds if not (warm_from / f"warm_{s}").is_dir()]
        if missing:
            print(f"missing warm-ups: {', '.join(missing)}", file=sys.stderr)
            return 2
    mine = [root / f"pilot_{s}" for s in a.seeds] + (
        [] if warm_from else [root / f"warm_{s}" for s in a.seeds]
    )
    clash = [p for p in mine if p.exists()]
    if clash:
        print(f"refusing to write into existing directories: {', '.join(map(str, clash))}", file=sys.stderr)
        return 2
    root.mkdir(parents=True, exist_ok=True)
    for c in cmds:
        print("+", shlex.join(c), flush=True)
        rc = subprocess.run(c).returncode
        if rc != 0:
            print(f"command failed with exit code {rc}", file=sys.stderr)
            return rc
    return 0


if __name__ == "__main__":
    sys.exit(main())
