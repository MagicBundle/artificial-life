"""Driver of the diel oracle pilot (docs/DIEL.md 7.0): warm-up, pilot fork, analysis, per seed.

    uv run python scripts/diel_pilot.py --out ROOT [--seeds 21 22 23] [--dry-run]

Per seed S: ``alife run --preset mini3d --overlay diel --seed S --steps 40000 --out ROOT/warm_S`` then
``alife run --init-from ROOT/warm_S --set diel.oracle=20 --steps 160000 --out ROOT/pilot_S``. On a fork,
``--steps`` is the number of ADDITIONAL steps (cli._announce: ``step -> step + steps``), so the pilot runs
160,000 steps after the warm-up's 40,000. Refuses to write into an existing directory. The analysis
(``alife.analysis.diel_pilot``) runs over the pilot directories at the end.
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


def commands(root: Path, seeds: list[int]) -> list[list[str]]:
    cmds: list[list[str]] = []
    for s in seeds:
        warm, pilot = root / f"warm_{s}", root / f"pilot_{s}"
        base = ["uv", "run", "alife", "run"]
        tail = ["--out", str(warm), "--quiet"]
        cmds.append(
            [
                *base,
                "--preset",
                "mini3d",
                "--overlay",
                "diel",
                "--seed",
                str(s),
                "--steps",
                str(WARM_STEPS),
                *tail,
            ]
        )
        tail = ["--out", str(pilot), "--quiet"]
        init = ["--init-from", str(warm), "--set", f"diel.oracle={ORACLE_PCT}"]
        cmds.append([*base, *init, "--steps", str(PILOT_STEPS), *tail])
    cmds.append(
        ["uv", "run", "python", "-m", "alife.analysis.diel_pilot"]
        + [str(root / f"pilot_{s}") for s in seeds]
        + ["--json", str(root / "diel_pilot.json")]
    )
    return cmds


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, help="output root")
    ap.add_argument("--seeds", nargs="+", type=int, default=[21, 22, 23])
    ap.add_argument("--dry-run", action="store_true", help="print the commands only")
    a = ap.parse_args(argv)
    root = Path(a.out)
    cmds = commands(root, a.seeds)
    if a.dry_run:
        for c in cmds:
            print(shlex.join(c))
        return 0
    clash = [p for s in a.seeds for p in (root / f"warm_{s}", root / f"pilot_{s}") if p.exists()]
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
