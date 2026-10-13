"""Re-run digest_matrix.py on the current worktree and diff it against the 2D baseline (CPU device only).

Usage (from the worktree under test, so `uv run` resolves its alife):
    nice -n 10 uv run python <scratch>/digest_compare.py [--baseline B.json] [--out NEW.json] [--current NEW.json]

Exit status: 0 = bit-identical, 1 = any difference (or a case error), 2 = usage/run failure.

Policy:
  - strict everywhere, except these subtrees, where the baseline must be a subset (keys the new code ADDS are
    listed as INFO, not failures; changed or removed keys still fail):
      cases.*.dims                                (new Dims fields: nd, d3, lay, ...)
      layout                                      (new layout constants)
      runs.*.config_json                          (new config keys at their 2D defaults)
      config_hashes                               (new presets; existing preset hashes must still match)
      runs.*.checkpoints.*.metadata.config        (same, embedded in each checkpoint)
      config_hashes                               (a new preset, e.g. tiny3d of 3D PR 3; every 2D preset's
                                                   hash stays strict)
  - env (MLX/Python/macOS/device) differences are WARNINGS: digests are only meaningful on the same stack.
  - info, timing and descriptions are ignored.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
MATRIX = HERE / "digest_matrix.py"
SUBSET = (
    ("config_hashes",),  # new presets (e.g. tiny3d) are additions; existing preset hashes must still match
    ("cases", "*", "dims"),
    ("layout",),
    ("runs", "*", "config_json"),
    ("runs", "*", "checkpoints", "*", "metadata", "config"),
    ("config_hashes",),
)
IGNORED = {"info", "timing", "descriptions"}


def _match(path: tuple[str, ...], pattern: tuple[str, ...]) -> bool:
    return len(path) == len(pattern) and all(p == "*" or p == x for x, p in zip(path, pattern, strict=True))


def _short(v: Any, n: int = 100) -> str:
    s = json.dumps(v, sort_keys=True) if not isinstance(v, str) else v
    return s if len(s) <= n else s[: n - 3] + "..."


def _eq_scalar(a: Any, b: Any) -> bool:
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return type(a) is type(b) and a == b


class Diff:
    def __init__(self) -> None:
        self.diffs: list[str] = []
        self.infos: list[str] = []

    def walk(self, a: Any, b: Any, path: tuple[str, ...], subset: bool = False) -> None:
        subset = subset or any(_match(path, p) for p in SUBSET)
        where = ".".join(path) or "<root>"
        if isinstance(a, dict) and isinstance(b, dict):
            for k in a:
                if k not in b:
                    self.diffs.append(f"{where}.{k}: missing in current (baseline {_short(a[k])})")
                else:
                    self.walk(a[k], b[k], (*path, str(k)), subset)
            for k in b:
                if k not in a:
                    msg = f"{where}.{k}: added in current = {_short(b[k])}"
                    (self.infos if subset else self.diffs).append(msg)
        elif isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                self.diffs.append(f"{where}: length {len(a)} -> {len(b)}")
            for i, (x, y) in enumerate(zip(a, b)):
                self.walk(x, y, (*path, str(i)), subset)
        elif not _eq_scalar(a, b):
            self.diffs.append(f"{where}: {_short(a)} -> {_short(b)}")


def _first_divergence(base: dict[str, Any], cur: dict[str, Any]) -> str | None:
    """For a trajectory case: the first step whose state/StepOut digest differs (1-based step number)."""
    notes = []
    for key, label in (("digests", "state"), ("out_digests", "StepOut")):
        a, b = base.get(key), cur.get(key)
        if isinstance(a, list) and isinstance(b, list):
            i = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
            if i is not None:
                notes.append(f"{label} digest first differs after step {i + 1}")
    if base.get("digest_init") != cur.get("digest_init"):
        notes.insert(0, "init_state digest differs")
    ff_a, ff_b = base.get("final_field_digests") or {}, cur.get("final_field_digests") or {}
    fields = sorted(k for k in set(ff_a) | set(ff_b) if ff_a.get(k) != ff_b.get(k))
    if fields:
        notes.append(f"final fields differing: {', '.join(fields)}")
    return "; ".join(notes) or None


def compare(base: dict[str, Any], cur: dict[str, Any], max_per_section: int) -> int:
    print(f"baseline: git {base['info'].get('git_head')} (dirty={base['info'].get('git_dirty')})")
    print(f"current : git {cur['info'].get('git_head')} (dirty={cur['info'].get('git_dirty')}) "
          f"alife={cur['info'].get('alife_file')}")
    env = Diff()
    env.walk(base.get("env"), cur.get("env"), ("env",))
    for line in env.diffs + env.infos:
        print(f"WARNING env changed, digests may legitimately differ: {line}")

    n_diff = 0
    sections = [k for k in base if k not in IGNORED | {"env"}]
    sections += [k for k in cur if k not in base and k not in IGNORED | {"env"}]
    for sec in sections:
        if sec not in cur:
            print(f"DIFF {sec}: missing in current")
            n_diff += 1
            continue
        if sec not in base:
            print(f"DIFF {sec}: new section in current")
            n_diff += 1
            continue
        if sec in ("cases", "runs"):
            names = list(base[sec]) + [n for n in cur[sec] if n not in base[sec]]
            for name in names:
                b, c = base[sec].get(name), cur[sec].get(name)
                if b is None or c is None:
                    print(f"DIFF {sec}.{name}: {'missing in current' if c is None else 'not in baseline'}")
                    n_diff += 1
                    continue
                if "error" in c:
                    print(f"DIFF {sec}.{name}: case error in current: {c['error']}")
                    print("     " + c.get("traceback", "").replace("\n", "\n     "))
                    n_diff += 1
                    continue
                d = Diff()
                d.walk(b, c, (sec, name))
                status = "ok" if not d.diffs else f"{len(d.diffs)} difference(s)"
                print(f"{sec}.{name:30s} {status}")
                if d.diffs:
                    n_diff += len(d.diffs)
                    if sec == "cases":
                        note = _first_divergence(b, c)
                        if note:
                            print(f"     -> {note}")
                    for line in d.diffs[:max_per_section]:
                        print(f"     DIFF {line}")
                    if len(d.diffs) > max_per_section:
                        print(f"     ... {len(d.diffs) - max_per_section} more")
                for line in d.infos:
                    print(f"     INFO {line}")
        else:
            d = Diff()
            d.walk(base[sec], cur[sec], (sec,))
            print(f"{sec:36s} {'ok' if not d.diffs else f'{len(d.diffs)} difference(s)'}")
            n_diff += len(d.diffs)
            for line in d.diffs[:max_per_section]:
                print(f"     DIFF {line}")
            for line in d.infos:
                print(f"     INFO {line}")
    if n_diff:
        print(f"\nFAIL: {n_diff} difference(s) against the 2D baseline")
        return 1
    print("\nOK: bit-identical to the 2D baseline (CPU device)")
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--baseline", type=Path, default=HERE / "digest_baseline.json")
    ap.add_argument("--out", type=Path, default=HERE / "digest_current.json", help="where the new run is written")
    ap.add_argument("--current", type=Path, default=None, help="compare this file instead of re-running")
    ap.add_argument("--max-diffs", type=int, default=20, help="differences printed per case")
    a = ap.parse_args(argv)
    if not a.baseline.is_file():
        print(f"no baseline at {a.baseline}", file=sys.stderr)
        return 2
    cur_path = a.current
    if cur_path is None:
        env = dict(os.environ)
        env.setdefault("TMPDIR", str(HERE / "mlxjit"))
        Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
        a.out.unlink(missing_ok=True)  # never diff a stale file left by an earlier run
        r = subprocess.run([sys.executable, str(MATRIX), str(a.out)], env=env, check=False)
        if r.returncode not in (0, 1) or not a.out.is_file():  # 1 = some case errored: still diff it
            print(f"digest_matrix.py failed (exit {r.returncode})", file=sys.stderr)
            return 2
        cur_path = a.out
    base = json.loads(a.baseline.read_text(encoding="utf-8"))
    cur = json.loads(cur_path.read_text(encoding="utf-8"))
    return compare(base, cur, a.max_diffs)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
