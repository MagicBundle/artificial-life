"""The CPU identity matrix: a tree against a fresh ``git archive`` of its base (DESIGN §43.9, §22.2).

Every PR of the feature series (and any engine PR that claims "off = bit-identical") runs this as a script::

    PYTHONPATH= uv run python tests/identity_matrix.py --base e64be03

It extracts ``src`` of the base commit into a scratch directory, runs the same case list once under
``PYTHONPATH=<base>/src`` and once under the tree's ``src`` (each a fresh subprocess on the CPU device), and
compares everything the cases record. It exits 0 and prints ``OK: bit-identical`` or exits 1 listing every
``DIFF``. No ``INFO`` class exists: a difference of any kind fails the gate, including a new key in
``config.json`` or in the checkpoint metadata.

The cases:

* 15 trajectory cases (2D and 3D; compiled, eager, ``observe=True``): the full ``state_digest``, the legacy
  digest, the counters and the ledger of EVERY step, and for ``observe=True`` the observer channel.
* 3 runner cases (``alife run`` into a directory): the file listing, ``config.json``, ``meta.json`` without
its
  volatile keys, the checkpoint bytes, the metrics rows without wall-clock keys, the events and observer
  files.
* the config hashes and the layout of every preset and overlay combination, the genome layout version, the
  counter and ledger name lists, ``State._fields``, ``expected_layout``.
* the ``tiny3d`` state digest after 50 compiled float32 steps (``1fe07a9f299526c0...``) and its legacy digest
  (``556de6c6216dd530...``), pinned as literals as well, so the script cannot pass by running a different
  case.

The scratch directory is ``--work`` (default: a new directory under the system temp root, removed on success
only when the script created it).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

# The sphere presets have no counterpart in the base (it has no sphere, docs/SPHERE.md 7.4): their digests are
# pinned as literals in tests/test_sphere_trajectory.py, not compared here.
SPHERE_PRESETS = frozenset({"tiny3s"})

TINY3D_STATE = "1fe07a9f299526c0"
TINY3D_LEGACY = "556de6c6216dd530"

# volatile keys: they differ between two runs of the very same code
META_VOLATILE = {"created_utc", "git_rev", "python", "platform", "device"}
ROW_VOLATILE = {"wall_s", "steps_per_s", "peak_mem_gb"}


# --- the dump side: runs inside a subprocess under one PYTHONPATH -----------------------------------------


def _sha(*parts: bytes) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p)
    return h.hexdigest()[:20]


def _arr_digest(a: Any) -> str:
    from alife.state import to_numpy

    n = to_numpy(a)
    return _sha(f"{n.dtype}|{n.shape}|".encode(), n.tobytes())


def _acts_digest(acts: Any) -> dict[str, str]:
    """Per field of the observer channel; ``flags`` as int64 (uint8 on the base, uint16 later:
    observer-only)."""
    import numpy as np

    from alife.state import to_numpy

    out: dict[str, str] = {}
    for name in acts._fields:
        v = getattr(acts, name)
        if v is None or name == "ext":  # ``ext`` exists only after §43; all None while the features are off
            continue
        if name == "flags":
            n = to_numpy(v).astype(np.int64)
            out[name] = _sha(f"flags|{n.shape}|".encode(), n.tobytes())
        else:
            out[name] = _arr_digest(v)
    return out


def _trajectory(
    cfg: Any, steps: int, *, compile: bool, observe: bool = False, prep: Any = None
) -> dict[str, Any]:
    import mlx.core as mx

    from alife.state import init_state, legacy_digest, state_digest, to_numpy
    from alife.step import make_step

    s = init_state(cfg)
    if prep is not None:
        s = prep(s)
        mx.eval(s)
    fn = make_step(cfg, compile=compile, observe=observe)
    per_step: list[list[str]] = []
    for _ in range(steps):
        s, out = fn(s)
        mx.eval(s, out)
        row = [
            state_digest(s),
            legacy_digest(s),
            _sha(to_numpy(out.counters).tobytes()),
            _sha(to_numpy(out.ledger).tobytes()),
        ]
        if observe:
            row.append(json.dumps(_acts_digest(out.acts), sort_keys=True))
        per_step.append(row)
    return {
        "final_state": state_digest(s),
        "final_legacy": legacy_digest(s),
        "steps": per_step,
        "n_alive": int(to_numpy(s.alive).sum()),
        "attacks": int(to_numpy(out.counters)[10]),
    }


def _cases() -> dict[str, dict[str, Any]]:
    """name -> {preset, overlays, overrides, steps, compile, observe, prep}. 15 trajectory cases."""
    import mlx.core as mx

    f32 = "brain.weight_dtype=float32"

    def cat(s: Any) -> Any:  # a catastrophe lever set on the director's env (consumed after one step)
        return s._replace(
            env_cat_center=mx.array([60.0, 60.0]), env_cat_radius=mx.array(40.0), env_cat_kill_p=mx.array(0.3)
        )

    def imm(s: Any) -> Any:  # an immigration request
        return s._replace(env_imm_request=mx.array(6, dtype=mx.int32))

    return {
        "2d_tiny_c": dict(preset="tiny", overrides=[f32], steps=40, compile=True),
        "2d_tiny_eager": dict(preset="tiny", overrides=[f32, "run.seed=1"], steps=25, compile=False),
        "2d_tiny_default_dtype": dict(preset="tiny", overrides=[], steps=30, compile=True),
        "2d_tiny_observe": dict(preset="tiny", overrides=[f32], steps=30, compile=True, observe=True),
        "2d_dev": dict(preset="dev", overrides=[f32, "run.seed=2"], steps=12, compile=True),
        "2d_seasons_immigration": dict(
            preset="tiny",
            overrides=[
                f32,
                "fields.season_period=40",
                "fields.season_amplitude=0.5",
                "population.immigration_every=7",
                "population.immigration_count=3",
            ],
            steps=40,
            compile=True,
        ),
        "2d_director_inert": dict(
            preset="tiny", overrides=[f32, "director.enabled=true"], steps=30, compile=True
        ),
        "2d_director_catastrophe": dict(
            preset="tiny", overrides=[f32, "director.enabled=true"], steps=20, compile=True, prep=cat
        ),
        "2d_director_immigration": dict(
            preset="tiny", overrides=[f32, "director.enabled=true"], steps=20, compile=True, prep=imm
        ),
        "2d_weather": dict(
            preset="tiny",
            overrides=["world.size=320", "weather.enabled=true", "weather.u_max=0.2", f32],
            steps=25,
            compile=True,
        ),
        "2d_bonds_combat": dict(
            preset="tiny",
            overrides=[
                f32,
                "combat.attack_bonded=true",
                "bonds.form_range=6.0",
                "bonds.break_length=12.0",
                "combat.range=6.0",
            ],
            steps=40,
            compile=True,
        ),
        "2d_runtime_checks_off": dict(
            preset="tiny", overrides=[f32, "run.runtime_checks=off"], steps=25, compile=True
        ),
        "3d_tiny3d": dict(preset="tiny3d", overrides=[f32], steps=50, compile=True),
        "3d_tiny3d_observe": dict(
            preset="tiny3d", overrides=[f32, "run.seed=3"], steps=20, compile=True, observe=True
        ),
        "3d_tiny3d_eager": dict(preset="tiny3d", overrides=[f32, "run.seed=4"], steps=15, compile=False),
    }


def _runner(kind: str, out: Path) -> None:
    """Run ``alife run`` in-process for one of the 3 runner cases into ``out`` (an empty directory)."""
    from alife.cli import main

    common = ["--seed", "5", "--out", str(out), "--quiet"]
    if kind == "tiny":
        argv = [
            "run",
            "--preset",
            "tiny",
            "--steps",
            "120",
            *common,
            "--set",
            "run.checkpoint_every=40",
            "--set",
            "run.metrics_every=10",
        ]
    elif kind == "tiny3d":
        argv = [
            "run",
            "--preset",
            "tiny3d",
            "--steps",
            "60",
            *common,
            "--set",
            "run.checkpoint_every=30",
            "--set",
            "run.metrics_every=10",
        ]
    elif kind == "tiny_director_observe":
        argv = [
            "run", "--preset", "tiny", "--steps", "100", *common,
            "--set", "run.checkpoint_every=50", "--set", "run.metrics_every=10",
            "--set", "director.enabled=true", "--set", "game.observe=true",
        ]  # fmt: skip
    else:
        raise ValueError(kind)
    rc = main(argv)
    if rc not in (0, None):
        raise SystemExit(f"alife {' '.join(argv)} exited {rc}")


RUNNER_CASES = ("tiny", "tiny3d", "tiny_director_observe")


def _runner_snapshot(root: Path) -> dict[str, Any]:
    files: dict[str, str] = {}
    texts: dict[str, Any] = {}
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = str(p.relative_to(root))
        data = p.read_bytes()
        if rel.endswith("meta.json") and p.parent == root:
            meta = json.loads(data)
            texts[rel] = {k: v for k, v in meta.items() if k not in META_VOLATILE}
            files[rel] = "meta"
        elif rel.endswith("metrics.jsonl"):
            rows = [
                {k: v for k, v in json.loads(line).items() if k not in ROW_VOLATILE}
                for line in data.decode().splitlines()
            ]
            texts[rel] = rows
            files[rel] = "metrics"
        elif rel.endswith(".json") and "checkpoint" not in rel:
            files[rel] = _sha(data)
        else:
            files[rel] = _sha(data)
    return {"files": files, "texts": texts}


def _checkpoint_meta(root: Path) -> dict[str, Any]:
    """The metadata of the newest checkpoint, volatile keys removed (it holds the config JSON)."""
    from alife.runio import checkpoint_metadata

    cks = sorted((root / "checkpoints").glob("*"))
    out: dict[str, Any] = {}
    for p in cks:
        if p.is_file() and p.name != "LATEST":
            try:
                meta = dict(checkpoint_metadata(p))
            except Exception as e:
                out[p.name] = f"unreadable: {type(e).__name__}"
                continue
            out[p.name] = {k: v for k, v in meta.items() if k not in META_VOLATILE | {"created_utc"}}
    return out


def _config_and_layout(overlays: list[str] | None = None) -> dict[str, Any]:
    from alife import config, layout, state
    from alife.config import ConfigError, dims, load_config

    out: dict[str, Any] = {}
    combos: list[tuple[str, tuple[str, ...]]] = []
    for preset in config.PRESETS:
        if preset in SPHERE_PRESETS:
            continue
        combos.append((preset, ()))
        # The overlay list is the BASE tree's ``ALL_OVERLAYS`` (dumped by the driver): a feature overlay
        # (``build``) switches its feature ON, so it has no counterpart in the base and is not an off-identity
        # case; the feature's own tests cover it.
        for ov in overlays if overlays is not None else config.ALL_OVERLAYS:
            combos.append((preset, (ov,)))
    combos.append(("tiny", ("fast", "storm")))
    for preset, overlays in combos:
        key = f"{preset}+{'+'.join(overlays) or '-'}"
        try:
            cfg = load_config(preset=preset, overlays=overlays)
        except ConfigError as e:
            out[key] = f"ConfigError: {e}"
            continue
        d = dims(cfg)
        out[key] = {
            "hash": config.config_hash(cfg),
            "D_IN": d.lay.D_IN,
            "D_OUT": d.lay.D_OUT,
            "NF": d.lay.NF,
            "params": d.params,
            "w1_rows": d.w1_rows,
            "genome_layout_version": layout.genome_layout_version(cfg.world.dims),
            "n_counters": state.n_counters(d.nd, cfg.weather.enabled),
            "state_fields": len(state.expected_layout(cfg)),
        }
    out["static"] = {
        "state_fields": list(state.State._fields),
        "state3_fields": list(state.State3._fields),
        "v0": list(state.V0_FIELDS),
        "v0_3d": list(state.V0_FIELDS_3D),
        "ledger": [list(state.ledger_names(2)), list(state.ledger_names(3))],
        "counters": [e.name for e in state.CounterIdx],
        "counters3d": [e.name for e in state.Counter3D],
        "input_names": [list(layout.for_dims(2).INPUT_NAMES), list(layout.for_dims(3).INPUT_NAMES)],
        "output_names": [list(layout.for_dims(2).OUTPUT_NAMES), list(layout.for_dims(3).OUTPUT_NAMES)],
        "rng_layout_version": __import__("alife.rng", fromlist=["x"]).RNG_LAYOUT_VERSION,
    }
    return out


def dump(out_json: Path, work: Path, only: str | None = None, overlays: list[str] | None = None) -> None:
    """Run every case in this process (CPU device) and write the record to ``out_json``."""
    import mlx.core as mx

    mx.set_default_device(mx.cpu)
    from alife.config import load_config
    from alife.state import init_state, legacy_digest, state_digest
    from alife.step import make_step

    result: dict[str, Any] = {"traj": {}, "runner": {}, "config": {}}
    for name, spec in _cases().items():
        if only and only not in name:
            continue
        cfg = load_config(preset=spec["preset"], overrides=spec["overrides"])
        result["traj"][name] = _trajectory(
            cfg,
            spec["steps"],
            compile=spec["compile"],
            observe=spec.get("observe", False),
            prep=spec.get("prep"),
        )
        print(f"  trajectory {name}", file=sys.stderr, flush=True)
    for kind in RUNNER_CASES:
        if only and only not in kind:
            continue
        d = work / f"run_{kind}"
        d.mkdir(parents=True, exist_ok=False)
        _runner(kind, d)
        snap = _runner_snapshot(d)
        snap["checkpoints_meta"] = _checkpoint_meta(d)
        result["runner"][kind] = snap
        print(f"  runner {kind}", file=sys.stderr, flush=True)
    result["config"] = _config_and_layout(overlays)
    # the pinned tiny3d digests, recomputed apart from the trajectory cases
    cfg3 = load_config(preset="tiny3d", overrides=["brain.weight_dtype=float32"])
    s = init_state(cfg3)
    fn = make_step(cfg3, compile=True)
    for _ in range(50):
        s, out = fn(s)
        mx.eval(s, out)
    result["tiny3d"] = {"state": state_digest(s), "legacy": legacy_digest(s)}
    out_json.write_text(json.dumps(result, sort_keys=True))


# --- the driver side ---------------------------------------------------------------------------------------


def _diff(a: Any, b: Any, path: str, out: list[str]) -> None:
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a:
                out.append(f"DIFF {path}/{k}: only in head")
            elif k not in b:
                out.append(f"DIFF {path}/{k}: only in base")
            else:
                _diff(a[k], b[k], f"{path}/{k}", out)
    elif isinstance(a, list) and isinstance(b, list) and path.endswith("/steps"):
        for i, (x, y) in enumerate(zip(a, b, strict=False)):
            if x != y:
                which = [
                    n
                    for n, (p, q) in zip(
                        ("state", "legacy", "counters", "ledger", "acts"),
                        zip(x, y, strict=False),
                        strict=False,
                    )
                    if p != q
                ]
                out.append(f"DIFF {path}: first at step {i + 1} in {which}")
                return
        if len(a) != len(b):
            out.append(f"DIFF {path}: {len(a)} vs {len(b)} steps")
    elif a != b:
        out.append(f"DIFF {path}: base {str(a)[:100]!r} != head {str(b)[:100]!r}")


def _base_overlays(src: Path) -> list[str]:
    """The base tree's ``config.ALL_OVERLAYS`` (a child process under the base's ``PYTHONPATH``)."""
    env = {**os.environ, "PYTHONPATH": str(src)}
    code = "import json; from alife import config; print(json.dumps(list(config.ALL_OVERLAYS)))"
    r = subprocess.run([sys.executable, "-c", code], env=env, check=True, capture_output=True, text=True)
    return json.loads(r.stdout.strip().splitlines()[-1])


def _run_side(
    label: str, src: Path, work: Path, only: str | None, overlays: list[str] | None = None
) -> dict[str, Any]:
    side = work / label
    side.mkdir(parents=True, exist_ok=False)
    env = {**os.environ, "PYTHONPATH": str(src), "PYTHONHASHSEED": "0"}
    out_json = side / "dump.json"
    # the child puts ``--src`` first on ``sys.path`` itself (its own directory, ``tests/``, must not shadow
    # it)
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--dump",
        str(out_json),
        "--work",
        str(side),
        "--src",
        str(src),
    ]
    if only:
        cmd += ["--only", only]
    if overlays is not None:
        cmd += ["--overlays", json.dumps(overlays)]
    print(f"[{label}] {src}", file=sys.stderr, flush=True)
    r = subprocess.run(cmd, env=env, check=False)
    if r.returncode != 0:
        raise SystemExit(f"the {label} dump failed (exit {r.returncode})")
    return json.loads(out_json.read_text())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", default="e64be03", help="the base commit (default: main at PR 0a)")
    ap.add_argument("--repo", default=".", help="the repository (default: the current directory)")
    ap.add_argument("--work", default=None, help="the scratch directory (default: a new temp directory)")
    ap.add_argument(
        "--base-src", default=None, help="a src directory to use as the base instead of git archive"
    )
    ap.add_argument("--only", default=None, help="run only the cases whose name contains this")
    ap.add_argument("--dump", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--src", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--overlays", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    if args.dump:  # the child: one side of the comparison
        if args.src:
            sys.path.insert(0, args.src)
        dump(
            Path(args.dump), Path(args.work), args.only, json.loads(args.overlays) if args.overlays else None
        )
        return 0

    repo = Path(args.repo).resolve()
    own_work = args.work is None
    work = Path(tempfile.mkdtemp(prefix="identity-matrix-")) if own_work else Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    if args.base_src:  # a prepared base (the mutation test of the matrix itself)
        base_src = Path(args.base_src).resolve()
    else:
        base_dir = work / "base-archive"
        base_dir.mkdir(parents=True, exist_ok=False)
        tar = subprocess.run(
            ["git", "-C", str(repo), "archive", args.base, "src"], capture_output=True, check=True
        )
        subprocess.run(["tar", "-x", "-C", str(base_dir)], input=tar.stdout, check=True)
        base_src = base_dir / "src"
    head_rev = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    print(f"base {args.base} (git archive), head {head_rev}+tree, work {work}", file=sys.stderr)

    overlays = _base_overlays(base_src)
    base = _run_side("base", base_src, work, args.only, overlays)
    head = _run_side("head", repo / "src", work, args.only, overlays)

    diffs: list[str] = []
    _diff(base, head, "", diffs)
    # the pinned tiny3d digests, against the literals (the script must not pass by comparing two wrong runs)
    if not args.only:
        for side, rec in (("base", base), ("head", head)):
            if not rec["tiny3d"]["state"].startswith(TINY3D_STATE) or not rec["tiny3d"]["legacy"].startswith(
                TINY3D_LEGACY
            ):
                pinned = f"{TINY3D_STATE}.../{TINY3D_LEGACY}..."
                diffs.append(f"DIFF /tiny3d: {side} digests {rec['tiny3d']} != pinned {pinned}")
    n_traj, n_run = len(head["traj"]), len(head["runner"])
    for line in diffs:
        print(line)
    if diffs:
        print(f"FAIL: {len(diffs)} DIFF ({n_traj} trajectory, {n_run} runner cases; work kept in {work})")
        return 1
    steps = sum(len(v["steps"]) for v in head["traj"].values())
    t3 = f"{head['tiny3d']['state'][:8]}/{head['tiny3d']['legacy'][:8]}"
    print(
        f"OK: bit-identical, 0 DIFF, 0 INFO ({n_traj} trajectory cases = {steps} steps x 4 digests, "
        f"{n_run} runner cases, {len(head['config'])} config/layout entries, tiny3d {t3})"
    )
    if own_work and "--keep" not in (argv or sys.argv):
        import shutil

        shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
