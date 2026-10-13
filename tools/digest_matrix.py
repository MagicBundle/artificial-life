"""2D determinism digest matrix on the MLX CPU device (baseline for the 3D PRs, docs/3D.md, map §0.3).

Usage (from the worktree, so `uv run` resolves the worktree's alife):
    TMPDIR=<scratch>/mlxjit nice -n 10 uv run python digest_matrix.py OUT.json

Never touches the GPU: the default device is set to the CPU before alife is imported, and asserted again at
the end. Writes run directories only under <this script's dir>/digest_runs/ (removed afterwards unless
--keep-runs).

Recorded per trajectory case (compiled `make_step` unless noted):
  config_hash, Dims, init digests, state_digest after EVERY step (a superset of every-10), a digest of each
  step's full StepOut (counters, ledger, birth/death records), the per-step counters and ledger, the final
  counters/ledger, legacy_state_digest at the end, and per-field digests of the final state.
Recorded per in-process runner case (runner.run, the code path behind `alife run`, never the CLI):
  metrics.jsonl rows without wall_s/steps_per_s/peak_mem_gb, meta.json (volatile values masked), config.json,
  every checkpoint (metadata, per-tensor digests, state_digest after load_checkpoint), event chunks
  (per-column digests), director.jsonl / journal.jsonl / director_archive.json, and the file listing.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if sys.path and os.path.abspath(sys.path[0] or ".") == _HERE:  # never import from the scratch dir
    sys.path.pop(0)

import mlx.core as mx  # noqa: E402

mx.set_default_device(mx.cpu)  # before any alife import or array work: the GPU belongs to the eval arms

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from collections.abc import Callable, Iterator, Mapping  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402

import alife  # noqa: E402
from alife import bench, layout, runner  # noqa: E402
from alife.config import OVERLAYS, PRESETS, Config, apply_overrides, config_hash, dims, load_config  # noqa: E402
from alife.director import make_director  # noqa: E402
from alife.director import actions as director_actions  # noqa: E402
from alife.director import rules as director_rules  # noqa: E402
from alife.runio import RunDir, checkpoint_metadata, load_checkpoint  # noqa: E402
from alife.state import init_state, legacy_state_digest, state_digest, to_numpy  # noqa: E402
from alife.step import make_step  # noqa: E402

SCHEMA = 1
PERF_KEYS = ("wall_s", "steps_per_s", "peak_mem_gb")
META_VOLATILE = ("created_utc", "git_rev")
F32 = "brain.weight_dtype=float32"
MICRO = [
    "population.n_max=64",
    "population.max_births_per_step=8",
    "population.n_init=8",
    "world.size=48",
    F32,
]


# --- hashing helpers --------------------------------------------------------------------------------------


def _array_bytes(a: mx.array) -> tuple[str, tuple[int, ...], bytes]:
    arr = np.ascontiguousarray(to_numpy(a))
    return str(a.dtype), tuple(a.shape), arr.tobytes()


def array_digest(a: mx.array) -> str:
    dt, shape, raw = _array_bytes(a)
    h = hashlib.sha256(f"{dt}|{shape}|".encode())
    h.update(raw)
    return h.hexdigest()


def flatten(tree: Any, prefix: str = "") -> Iterator[tuple[str, Any]]:
    """(path, leaf) over NamedTuples, tuples, lists and dicts; leaves are mx.arrays or Python scalars."""
    if isinstance(tree, mx.array):
        yield prefix, tree
    elif hasattr(tree, "_fields"):
        for f in tree._fields:
            yield from flatten(getattr(tree, f), f"{prefix}.{f}" if prefix else f)
    elif isinstance(tree, (list, tuple)):
        for i, x in enumerate(tree):
            yield from flatten(x, f"{prefix}[{i}]")
    elif isinstance(tree, Mapping):
        for k in sorted(tree):
            yield from flatten(tree[k], f"{prefix}.{k}" if prefix else str(k))
    elif tree is not None:
        yield prefix, tree


def tree_digest(tree: Any) -> str:
    h = hashlib.sha256()
    for path, leaf in flatten(tree):
        if isinstance(leaf, mx.array):
            dt, shape, raw = _array_bytes(leaf)
            h.update(f"{path}|{dt}|{shape}|".encode())
            h.update(raw)
        else:
            h.update(f"{path}|py|{leaf!r}|".encode())
    return h.hexdigest()


def _jsonable_scalar(v: Any) -> Any:
    if isinstance(v, (bool, int, float, str)) or v is None:
        return v
    if isinstance(v, mx.Dtype):
        return str(v)
    if isinstance(v, (tuple, list)):
        return [_jsonable_scalar(x) for x in v]
    if dataclasses.is_dataclass(v) and not isinstance(v, type):
        return {f.name: _jsonable_scalar(getattr(v, f.name)) for f in dataclasses.fields(v)}
    return repr(v)


def dims_record(cfg: Config) -> dict[str, Any]:
    d = dims(cfg)
    return {f.name: _jsonable_scalar(getattr(d, f.name)) for f in dataclasses.fields(d)}


def layout_record() -> dict[str, Any]:
    """Public upper-case int/str/tuple constants of alife.layout (compared as a subset: additions are info)."""
    out: dict[str, Any] = {}
    for name in sorted(vars(layout)):
        if not name.isupper() or name.startswith("_"):
            continue
        v = getattr(layout, name)
        if isinstance(v, (int, str)) or (
            isinstance(v, tuple) and all(isinstance(x, (int, str, float)) for x in v)
        ):
            out[name] = _jsonable_scalar(v)
    return out


# --- trajectory cases -------------------------------------------------------------------------------------


def _set_env(s: Any, **kw: Any) -> Any:
    """``s`` with ``env_<name>`` replaced (dtype of the field), the host-side lever write the director does."""
    upd = {f"env_{k}": mx.array(np.asarray(v), dtype=getattr(s, f"env_{k}").dtype) for k, v in kw.items()}
    return s._replace(**upd)


def _lever_plan(cfg: Config) -> dict[int, dict[str, Any]]:
    L = cfg.world.size
    return {
        4: dict(cat_center=[L / 2, L / 2], cat_radius=L / 3, cat_kill_p=0.6),
        6: dict(imm_request=70),
        12: dict(cat_center=[10.0, 100.0], cat_radius=L / 2, cat_kill_p=1.0, imm_request=5),
        13: dict(regen_mult=0.0, drift_mult=7.0),
        20: dict(regen_mult=3.0, patch_gain=[0.0, 2.0, 1.0, 0.5], season_amp=0.8, season_period=9),
        30: dict(cat_center=[0.0, 0.0], cat_radius=L / 2, cat_kill_p=0.9, imm_request=40),
        100: dict(
            regen_mult=1.0, drift_mult=1.0, patch_gain=[1.0, 1.0, 1.0, 1.0], season_amp=0.3, season_period=50,
            imm_request=100,
        ),
        150: dict(cat_center=[100.0, 30.0], cat_radius=20.0, cat_kill_p=0.5),
    }


def _record_final(s: Any, rec: dict[str, Any]) -> None:
    rec["final_step"] = int(to_numpy(s.step))
    rec["final_diag"] = int(to_numpy(s.diag))
    rec["final_n_alive"] = int(to_numpy(s.alive).sum())
    rec["final_total"] = int(to_numpy(s.total))
    rec["final_total0"] = int(to_numpy(s.total0))
    rec["final_ledger_cum"] = to_numpy(s.ledger_cum).tolist()
    rec["legacy_digest_final"] = legacy_state_digest(s)
    rec["digest_final"] = state_digest(s)
    rec["final_field_digests"] = {name: array_digest(getattr(s, name)) for name in type(s)._fields}


def _record_step(rec: dict[str, Any], s: Any, out: Any) -> None:
    rec["digests"].append(state_digest(s))
    rec["out_digests"].append(tree_digest(out))
    rec["counters"].append(to_numpy(out.counters).tolist())
    rec["ledger"].append(to_numpy(out.ledger).tolist())


def _new_rec(cfg: Config, s0: Any, steps: int, kind: str) -> dict[str, Any]:
    return {
        "kind": kind,
        "steps": steps,
        "config_hash": config_hash(cfg),
        "dims": dims_record(cfg),
        "digest_init": state_digest(s0),
        "legacy_digest_init": legacy_state_digest(s0),
        "digests": [],
        "out_digests": [],
        "counters": [],
        "ledger": [],
    }


def trajectory(
    cfg: Config, steps: int, *, pre: Callable[[int, Any], Any] | None = None, compile: bool | None = None
) -> dict[str, Any]:
    fn = make_step(cfg, compile=compile)
    s = init_state(cfg)
    rec = _new_rec(cfg, s, steps, "step" if compile is None else f"step(compile={compile})")
    for i in range(steps):
        if pre is not None:
            s = pre(i, s)
        s, out = fn(s)
        mx.eval(s, out)
        _record_step(rec, s, out)
    rec["final_counters"] = rec["counters"][-1]
    rec["final_ledger"] = rec["ledger"][-1]
    _record_final(s, rec)
    return rec


def staged(cfg: Config, steps: int, *, compile: bool) -> dict[str, Any]:
    """bench._StagedStep (mirrors every step branch, map R6)."""
    st = bench._StagedStep(cfg, compile=compile)
    s0 = init_state(cfg)
    rec = _new_rec(cfg, s0, steps, f"bench._StagedStep(compile={compile})")
    holder = [s0]
    del s0
    for _ in range(steps):
        out = st(holder)
        mx.eval(holder[0], out)
        _record_step(rec, holder[0], out)
    rec["final_counters"] = rec["counters"][-1]
    rec["final_ledger"] = rec["ledger"][-1]
    _record_final(holder[0], rec)
    return rec


# --- runner cases -----------------------------------------------------------------------------------------


def _scripted_decide(plan: Mapping[int, list[Any]]) -> Callable[..., Any]:
    def decide(ctx: Any, params: Any = None) -> Any:
        return director_rules.Decision(tuple(plan.get(ctx.t, ())), ())

    return decide


def _prop(action: str, raw: dict[str, Any], duration: int | None = None) -> Any:
    return director_rules.Proposal(action, raw, duration, "scripted", f"scripted {action}")


DIRECTOR_PLAN = {
    20: [
        _prop("set_seasons", {"period": 5000, "amp": 0.5}),
        _prop("regen_pulse", {"mult": 1.4}, 60),  # reverts at 80
        _prop("fertility_shift", {"patch": 1, "gain": 1.5}),
    ],
    40: [_prop("habitat_loss", {"patch": 2}, 100), _prop("immigration_pulse", {"count": 60})],  # reverts 140
    60: [_prop("fertility_shift", {"patch": 0, "dx": 12.0, "dy": -9.0})],
    100: [
        _prop("catastrophe", {"x": 40.0, "y": 70.0, "radius": 30.0, "kill_p": 0.6}),
        _prop("drift_change", {"mult": 3.0}),
    ],
    120: [_prop("regen_pulse", {"mult": 0.6}, 40)],  # reverts at 160
    180: [_prop("immigration_pulse", {"count": 30}), _prop("set_seasons", {"period": 3000, "amp": 0.2})],
}


def _read_json_lines(path: Path) -> list[Any]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def collect_run(rd: RunDir, cfg: Config, res: Any) -> dict[str, Any]:
    root = rd.path
    rec: dict[str, Any] = {
        "result": {"status": res.status, "exit_code": int(res.exit_code)},
        "final_digest": state_digest(res.state),
        "final_legacy_digest": legacy_state_digest(res.state),
        "config_hash": config_hash(cfg),
        "files": sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()),
    }
    meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
    rec["meta"] = {k: ("<volatile>" if k in META_VOLATILE else v) for k, v in meta.items()}
    rec["config_json"] = json.loads((root / "config.json").read_text(encoding="utf-8"))
    rec["metrics"] = [
        {k: v for k, v in row.items() if k not in PERF_KEYS} for row in _read_json_lines(root / "metrics.jsonl")
    ]
    ckpts: dict[str, Any] = {}
    for step, path in rd.checkpoints():
        md = checkpoint_metadata(path)
        md = {k: (json.loads(v) if k == "config" else v) for k, v in md.items()}
        arrays = mx.load(str(path), format="safetensors")
        tensors = {k: [str(a.dtype), list(a.shape), array_digest(a)] for k, a in sorted(arrays.items())}
        entry: dict[str, Any] = {"step": step, "metadata": md, "tensors": tensors}
        try:
            entry["state_digest"] = state_digest(load_checkpoint(path, cfg))
        except Exception as e:  # recorded, so a later change in loading shows up as a difference
            entry["state_digest"] = f"ERROR {type(e).__name__}: {e}"
        ckpts[path.name] = entry
    rec["checkpoints"] = ckpts
    latest = rd.checkpoint_dir / "LATEST"
    rec["latest"] = latest.read_text(encoding="utf-8").strip() if latest.is_file() else None
    events: dict[str, Any] = {}
    for p in sorted(rd.events_dir.glob("*.npz")):
        with np.load(p) as z:
            events[p.name] = {
                k: [str(z[k].dtype), list(z[k].shape), hashlib.sha256(np.ascontiguousarray(z[k]).tobytes()).hexdigest()]
                for k in sorted(z.files)
            }
    rec["events"] = events
    other: dict[str, Any] = {}
    for p in sorted(root.glob("*.json*")):
        if p.name in ("meta.json", "config.json", "metrics.jsonl"):
            continue
        other[p.name] = _read_json_lines(p) if p.suffix == ".jsonl" else json.loads(p.read_text(encoding="utf-8"))
    rec["other_json"] = other
    return rec


def run_case(
    runs_dir: Path,
    name: str,
    cfg: Config,
    steps: int,
    *,
    plan: Mapping[int, list[Any]] | None = None,
    keep: bool = False,
) -> dict[str, Any]:
    tmp = Path(tempfile.mkdtemp(prefix=f"{name}_", dir=runs_dir))
    saved_decide = director_rules.decide
    saved_menu: dict[str, Any] = {}
    try:
        if plan is not None:
            director_rules.decide = _scripted_decide(plan)
            for k in ("regen_pulse", "habitat_loss"):  # reverts inside a 200-step run
                saved_menu[k] = director_actions.MENU[k]
                director_actions.MENU[k] = dataclasses.replace(saved_menu[k], duration=(20, 2000))  # type: ignore[index]
        rd = RunDir.create(tmp / "run", cfg)
        rd.update_meta(start_step=0)  # as `alife run` does
        director = make_director(rd, cfg, 0)
        res = runner.run(rd, cfg, init_state(cfg), steps, quiet=True, director=director)
        rec = collect_run(rd, cfg, res)
        rec["steps"] = steps
        return rec
    finally:
        director_rules.decide = saved_decide
        for k, v in saved_menu.items():
            director_actions.MENU[k] = v  # type: ignore[index]
        if not keep:
            shutil.rmtree(tmp, ignore_errors=True)


# --- the matrix -------------------------------------------------------------------------------------------


def trajectory_cases() -> list[tuple[str, Callable[[], dict[str, Any]], str]]:
    tiny_f32 = load_config(preset="tiny", overrides=[F32])  # conftest tiny_cfg (the golden config)

    def with_(*ov: str) -> Config:
        return apply_overrides(tiny_f32, list(ov))

    director_cfg = with_("director.enabled=true")
    plan = _lever_plan(director_cfg)

    def levers(i: int, s: Any) -> Any:
        return _set_env(s, **plan[i]) if i in plan else s

    return [
        ("tiny_f32", lambda: trajectory(tiny_f32, 200), "conftest tiny_cfg: tiny + float32 weights"),
        ("tiny_f16", lambda: trajectory(load_config(preset="tiny"), 200), "tiny preset, default (float16) weights"),
        (
            "tiny_bf16",
            lambda: trajectory(load_config(preset="tiny", overrides=["brain.weight_dtype=bfloat16"]), 200),
            "tiny + bfloat16 weights",
        ),
        ("micro", lambda: trajectory(load_config(preset="tiny", overrides=MICRO), 200), "conftest micro_cfg"),
        (
            "tiny_director_inert",
            lambda: trajectory(director_cfg, 200),
            "tiny_f32 + director.enabled=true (compiled step reads State.env; inert levers)",
        ),
        (
            "tiny_director_levers",
            lambda: trajectory(director_cfg, 200, pre=levers),
            "tiny_f32 + director.enabled=true, host lever writes at steps 4,6,12,13,20,30,100,150 "
            "(catastrophes, immigration requests, regen/drift/patch gain/seasons)",
        ),
        (
            "tiny_imm_floor",
            lambda: trajectory(with_("population.immigration_floor=400", "population.immigrants_per_step=16"), 200),
            "tiny_f32 + immigration_floor=400, immigrants_per_step=16",
        ),
        (
            "tiny_imm_every",
            lambda: trajectory(with_("population.immigration_every=10", "population.immigration_count=8"), 200),
            "tiny_f32 + immigration_every=10, immigration_count=8",
        ),
        (
            "tiny_seasons_global",
            lambda: trajectory(with_("fields.season_period=50", "fields.season_amplitude=0.5"), 200),
            "tiny_f32 + season_period=50, season_amplitude=0.5 (global)",
        ),
        (
            "tiny_seasons_per_patch",
            lambda: trajectory(
                with_("fields.season_period=40", "fields.season_amplitude=0.8", "fields.season_mode=per_patch"), 200
            ),
            "tiny_f32 + season_period=40, season_amplitude=0.8, per_patch",
        ),
        (
            "tiny_fast",
            lambda: trajectory(load_config(preset="tiny", overlays=["fast"], overrides=[F32]), 200),
            "tiny + fast overlay (H=24, K=6, C=12) + float32 weights",
        ),
        ("dev_20", lambda: trajectory(load_config(preset="dev"), 20), "dev preset, default weights, 20 steps"),
        (
            "tiny_f32_eager",
            lambda: trajectory(tiny_f32, 30, compile=False),
            "tiny_f32, eager step (compile=False), 30 steps",
        ),
        (
            "bench_staged_compiled",
            lambda: staged(tiny_f32, 50, compile=True),
            "tiny_f32 through bench._StagedStep, compiled stages, 50 steps",
        ),
        (
            "bench_staged_eager",
            lambda: staged(tiny_f32, 10, compile=False),
            "tiny_f32 through bench._StagedStep, eager stages, 10 steps",
        ),
    ]


def runner_cases(runs_dir: Path, keep: bool) -> list[tuple[str, Callable[[], dict[str, Any]], str]]:
    cli_tiny = load_config(preset="tiny", overrides=["run.steps=100"])  # == `alife run --preset tiny --steps 100`
    live = [
        F32,
        "director.enabled=true",
        "run.director=live",
        "director.rule_every=20",
        "run.metrics_every=10",
        "run.heavy_metrics_every=20",
        "run.checkpoint_every=100",
        "run.keep_checkpoints=20",
        "run.steps=200",
    ]
    scripted_cfg = load_config(preset="tiny", overrides=live)
    gardener_cfg = load_config(preset="tiny", overrides=live)
    return [
        (
            "run_tiny_cli",
            lambda: run_case(runs_dir, "run_tiny_cli", cli_tiny, 100, keep=keep),
            "runner.run of `alife run --preset tiny --steps 100` (in-process, director off)",
        ),
        (
            "run_tiny_director_scripted",
            lambda: run_case(runs_dir, "run_tiny_director_scripted", scripted_cfg, 200, plan=DIRECTOR_PLAN, keep=keep),
            "tiny_f32 + live director, rule_every=20, scripted rules.decide (all 7 actions + 3 reverts), 200 steps",
        ),
        (
            "run_tiny_director_gardener",
            lambda: run_case(runs_dir, "run_tiny_director_gardener", gardener_cfg, 200, keep=keep),
            "tiny_f32 + live director (gardener rule tier, unscripted), rule_every=20, 200 steps",
        ),
    ]


def config_hashes() -> dict[str, str]:
    out = {}
    for p in PRESETS:
        out[p] = config_hash(load_config(preset=p))
        for ov in OVERLAYS:
            out[f"{p}+{ov}"] = config_hash(load_config(preset=p, overlays=[ov]))
    return out


def _git(*args: str) -> str | None:
    try:
        r = subprocess.run(["git", *args], cwd=Path(alife.__file__).resolve().parent, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--only", nargs="*", default=None, help="case names to run (default: all)")
    ap.add_argument("--keep-runs", action="store_true", help="keep the runner case directories")
    a = ap.parse_args(argv)

    runs_dir = Path(_HERE) / "digest_runs"
    runs_dir.mkdir(exist_ok=True)
    t_all = time.perf_counter()
    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "env": {
            "mlx": mx.__version__,
            "python": platform.python_version(),
            "system": platform.system(),
            "machine": platform.machine(),
            "macos": platform.mac_ver()[0],
            "device": str(mx.default_device()),
        },
        "info": {
            "git_head": _git("rev-parse", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain")),
            "alife_file": alife.__file__,
            "alife_version": alife.__version__,
            "created_unix": time.time(),
        },
        "timing": {},
        "descriptions": {},
        "config_hashes": config_hashes(),
        "layout": layout_record(),
        "cases": {},
        "runs": {},
    }
    failures = 0
    for section, cases in (("cases", trajectory_cases()), ("runs", runner_cases(runs_dir, a.keep_runs))):
        for name, fn, desc in cases:
            if a.only is not None and name not in a.only:
                continue
            t0 = time.perf_counter()
            try:
                rec = fn()
            except Exception as e:
                failures += 1
                rec = {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}
            dt = time.perf_counter() - t0
            doc[section][name] = rec
            doc["descriptions"][name] = desc
            doc["timing"][name] = round(dt, 2)
            summary = rec.get("error") or (
                f"diag={rec.get('final_diag', '-')} n_alive={rec.get('final_n_alive', '-')} "
                f"digest={str(rec.get('digest_final', rec.get('final_digest')))[:12]}"
            )
            print(f"[{section}] {name:30s} {dt:7.1f}s  {summary}", flush=True)
    assert mx.default_device() == mx.cpu, "default device changed away from the CPU"
    doc["timing"]["total"] = round(time.perf_counter() - t_all, 2)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_suffix(a.out.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=False) + "\n", encoding="utf-8")
    os.replace(tmp, a.out)
    print(f"wrote {a.out} in {doc['timing']['total']:.1f}s ({failures} case error(s))", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
