"""The ``alife`` command line (§21): run, resume, render, inspect, view, stream, bench, llm-check. WP5.

``main(argv)`` never raises: errors map to ``runner.ExitCode`` values (2 config, 4 checkpoint, 130 SIGINT,
1 anything else; argparse usage errors are 2). ``view`` and ``bench`` delegate to WP6 (``viewer.run_viewer``,
``bench.run_bench``), imported lazily so that the optional pygame dependency is only needed by ``view``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import shutil
import sys
import time
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from alife import config as config_mod
from alife import metrics, physics, runner, stream
from alife.config import (
    DEFAULT_PRESET,
    DIRECTOR_MODES,
    DIRECTOR_POLICY_KEYS,
    OVERLAYS,
    PERSONAS,
    PRESETS,
    RANDOM_PERSONA,
    Config,
    ConfigError,
    apply_overrides,
    config_from_dict,
    config_hash,
    config_to_dict,
    dims,
    load_config,
)
from alife.director import llm as director_llm
from alife.director import log as director_log
from alife.director import make_director, yoked
from alife.invariants import full_check
from alife.render import (
    COLOR_MODES,
    DEFAULT_LAYERS,
    VIDEO_SUFFIXES,
    VideoWriter,
    Viewport,
    render_frame,
    write_png,
)
from alife.rng import InitStream, init_keys
from alife.runio import DIRECTOR_LOG_FILE, CheckpointError, RunDir, checkpoint_metadata, load_checkpoint
from alife.runner import ExitCode
from alife.state import DeathCause, State, State3, agent_fields, default_env, init_state, to_numpy
from alife.step import make_step

# run.* keys that resume may change: observers, cadences and the director mode (§18). Everything else goes
# through a fork. The director's policy keys (director.persona, ...) may change too: they never reach the
# compiled step.
RESUME_KEYS: frozenset[str] = frozenset(
    {
        "steps",
        "metrics_every",
        "checkpoint_every",
        "keep_checkpoints",
        "heavy_metrics_every",
        "components_iters",
        "diversity_sample",
        "director",
        "llm_url",
        "llm_model",
    }
)
RESUME_DIRECTOR_KEYS: frozenset[str] = frozenset(DIRECTOR_POLICY_KEYS)

LLM_DEFAULT_EVERY = 25_000  # steps between LLM calls with a bare --llm

_STREAM_HELP = (
    "write the world as binary frames on stdout for an external viewer (DESIGN §27.11). It re-simulates from "
    "RUN, a state FILE or --preset without the director, so after an intervention it differs from a director "
    "run; --follow streams the run's newest checkpoints instead and is the faithful view of a director run. "
    "It never writes into the run directory. Control lines on stdin: pause, run, step [N], every K, "
    "color MODE, fields on|off|z K, quit (stdin EOF = quit)."
)

_RUN_STEP_RE = re.compile(r"^(?P<run>.+):(?P<step>\d+)$")


# --- small parsers --------------------------------------------------------------------------------------


def _parse_size(text: str) -> tuple[int, int]:
    m = re.fullmatch(r"\s*(\d+)\s*[xX]\s*(\d+)\s*", text)
    if not m or int(m.group(1)) < 1 or int(m.group(2)) < 1:
        raise ConfigError(f"--size {text!r}: expected WxH with positive integers")
    return int(m.group(1)), int(m.group(2))


def _parse_view(text: str | None, cfg: Config) -> Viewport | None:
    if text is None:
        return None
    try:
        cx, cy, zoom = (float(v) for v in text.split(","))
    except ValueError:
        raise ConfigError(f"--view {text!r}: expected CX,CY,ZOOM") from None
    if not zoom > 0:
        raise ConfigError(f"--view {text!r}: ZOOM must be > 0")
    return Viewport(cx % cfg.world.size, cy % cfg.world.size, zoom)


def _parse_layers(text: str) -> frozenset[str]:
    layers = frozenset(v.strip() for v in text.split(",") if v.strip())
    if unknown := layers - DEFAULT_LAYERS:
        raise ConfigError(f"--layers: unknown {sorted(unknown)}; choose from {sorted(DEFAULT_LAYERS)}")
    return layers


def _checkpoint_arg(text: str) -> int | None:
    """``latest`` → None, else a non-negative step."""
    if text == "latest":
        return None
    if not text.isdigit():
        raise argparse.ArgumentTypeError(f"expected a step or 'latest', got {text!r}")
    return int(text)


def _non_negative(text: str) -> int:
    v = int(text)
    if v < 0:
        raise argparse.ArgumentTypeError(f"expected an integer ≥ 0, got {text!r}")
    return v


def _positive(text: str) -> int:
    v = int(text)
    if v < 1:
        raise argparse.ArgumentTypeError(f"expected an integer ≥ 1, got {text!r}")
    return v


def _positive_seconds(text: str) -> float:
    v = float(text)
    if not 0 < v < float("inf"):
        raise argparse.ArgumentTypeError(f"expected a number of seconds > 0, got {text!r}")
    return v


def _json_default(x: Any) -> Any:
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    raise TypeError(f"not JSON serialisable: {type(x).__name__}")


def _dumps(x: Any) -> str:
    return json.dumps(x, indent=2, default=_json_default)


def _default_out(prefix: str, seed: int) -> Path:
    return Path("runs") / f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}-s{seed}"


# --- loading --------------------------------------------------------------------------------------------


def _select_checkpoint(rd: RunDir, step: int | None) -> Path:
    if step is None:
        path = rd.latest_checkpoint()
        if path is None:
            raise CheckpointError(f"{rd.path}: no checkpoint")
        return path
    try:
        return rd.checkpoint_at(step)
    except FileNotFoundError as e:
        have = [st for st, _ in rd.checkpoints()]
        raise CheckpointError(f"{e} (available: {have})") from None


def _embedded_config(path: Path) -> Config:
    """The config stored in a state file's metadata (§18), validated."""
    meta = checkpoint_metadata(path)
    try:
        cfg = config_from_dict(json.loads(meta["config"]))
    except (KeyError, json.JSONDecodeError) as e:
        raise CheckpointError(f"{path}: no readable embedded config ({e})") from None
    config_mod.validate(cfg)
    return cfg


def _target_config(target: str, step: int | None) -> tuple[Config, RunDir | None, Path]:
    """``(cfg, rundir, path)`` of a run directory (``--checkpoint``) or a standalone state file, unloaded."""
    p = Path(target)
    if p.is_file():
        if step is not None:
            raise ConfigError("--checkpoint applies to a run directory, not a state file")
        return _embedded_config(p), None, p
    rd = RunDir.open(p)
    cfg = rd.config()
    return cfg, rd, _select_checkpoint(rd, step)


def _load_target(target: str, step: int | None) -> tuple[Config, State, RunDir | None, Path]:
    """``(cfg, state, rundir, path)`` from a run directory (``--checkpoint``) or a standalone state file."""
    cfg, rd, path = _target_config(target, step)
    return cfg, load_checkpoint(path, cfg), rd, path


def _layer(base: Config, *, overlays: Sequence[str], path: str | None, overrides: Sequence[str]) -> Config:
    """``base`` < overlays < file < ``--set``, with ``load_config``'s rules (a fork's config, §18)."""
    d = config_to_dict(base)
    for ov in overlays:
        if ov not in OVERLAYS:
            raise ConfigError(f"unknown overlay {ov!r} (choose from {OVERLAYS})")
        config_mod._merge_layer(d, config_mod._read_preset(ov), f"overlay {ov}")
    if path is not None:
        config_mod._merge_layer(d, config_mod._read_file(Path(path)), str(path))
    return apply_overrides(config_from_dict(d), overrides)


def _run_overrides(a: argparse.Namespace) -> list[str]:
    """``--set`` plus the shortcut flags, which take precedence (they come last)."""
    out = list(a.set)
    for flag, key in (
        ("seed", "seed"),
        ("steps", "steps"),
        ("metrics_every", "metrics_every"),
        ("checkpoint_every", "checkpoint_every"),
        ("checks", "runtime_checks"),
    ):
        if getattr(a, flag) is not None:
            out.append(f"run.{key}={getattr(a, flag)}")
    if a.no_compile:
        out.append("run.compile=false")
    if a.director is not None:  # live and replay imply an enabled director (the hash-relevant key)
        out.append(f"run.director={a.director}")
        if a.director != "off":
            out.append("director.enabled=true")
    return out + _llm_overrides(a)


def _llm_overrides(a: argparse.Namespace) -> list[str]:
    """The LLM, persona and yoke-log shortcut flags as ``--set`` items (run, fork and resume)."""
    out: list[str] = []
    if a.llm:
        out.append(f"director.llm_every={LLM_DEFAULT_EVERY}")
    for flag, key in (
        ("llm_every", "director.llm_every"),
        ("llm_url", "run.llm_url"),
        ("llm_model", "run.llm_model"),
        ("persona", "director.persona"),
        ("goal", "director.goal"),
        ("yoke_log", "director.yoke_log"),
    ):
        if getattr(a, flag) is not None:
            out.append(f"{key}={getattr(a, flag)}")
    return out


def _check_llm_flags(a: argparse.Namespace, cfg: Config) -> None:
    """The LLM tier only exists in live mode: asking for it in any other mode is a mistake, not a no-op."""
    if (a.llm or (a.llm_every or 0) > 0) and cfg.run.director != "live":
        raise ConfigError("--llm / --llm-every need --director live (a replay never calls the model)")


def _check_yoke(cfg: Config) -> None:
    """A live random persona follows another run's log: check it (exit 2) before any run directory is
    created or touched. A replay never reads it."""
    if cfg.run.director == "live" and cfg.director.persona == RANDOM_PERSONA:
        yoked.load(cfg)


def _summary(cfg: Config) -> dict[str, Any]:
    """The config block of ``inspect`` and the run announcement. A 3D world adds ``dims``, ``world_height``
    and ``field_grid_z`` (the layer count FZ) after ``field_grid``; a 2D summary is unchanged (§27.11)."""
    d = dims(cfg)
    return {
        "n_max": d.n,
        "n_init": cfg.population.n_init,
        "max_births_per_step": d.m,
        "n_cap": d.n_cap,
        "world_size": cfg.world.size,
        "field_grid": d.fg,
        **({"dims": d.nd, "world_height": cfg.world.height, "field_grid_z": d.fz} if d.d3 else {}),
        "hidden": d.h,
        "k_neighbors": d.k,
        "weight_dtype": cfg.brain.weight_dtype,
        "immigration_floor": cfg.population.immigration_floor,
        "seed": cfg.run.seed,
        "steps": cfg.run.steps,
        "metrics_every": cfg.run.metrics_every,
        "checkpoint_every": cfg.run.checkpoint_every,
        "compile": cfg.run.compile,
        "runtime_checks": cfg.run.runtime_checks,
        "config_hash": config_hash(cfg),
    }


def _announce(rd: RunDir, cfg: Config, step: int, steps: int, quiet: bool) -> None:
    if not quiet:
        s = _summary(cfg)
        mode = f" | director {cfg.run.director}" if cfg.run.director != "off" else ""
        if cfg.run.director == "live" and cfg.director.llm_every:
            mode += f" + llm every {cfg.director.llm_every} ({cfg.run.llm_model})"
        if cfg.run.director == "live" and cfg.director.persona == RANDOM_PERSONA:
            mode += f" + random, yoked to {cfg.director.yoke_log}"
        world = f"L={s['world_size']:g}" + (f" height={s['world_height']:g}" if "world_height" in s else "")
        print(
            f"run {rd.path}: N={s['n_max']} n_init={s['n_init']} {world} "
            f"weights={s['weight_dtype']} seed={s['seed']} | steps {step} → {step + steps}{mode}",
            flush=True,
        )


def _launch(rd: RunDir, cfg: Config, box: list[State], steps: int, quiet: bool) -> int:
    """Run ``steps`` steps from the state in ``box``, with the director ``cfg.run.director`` names (if any).

    The runner must hold the only reference to the state (donation rule, F1), so it is popped here.
    """
    director = make_director(rd, cfg, int(to_numpy(box[0].step)))
    return int(runner.run(rd, cfg, box.pop(), steps, quiet=quiet, director=director).exit_code)


# --- run / fork / resume --------------------------------------------------------------------------------


def _cmd_run(a: argparse.Namespace) -> int:
    overrides = _run_overrides(a)
    if a.init_from is not None:
        return _fork(a, overrides)
    cfg = load_config(a.config, preset=a.preset, overlays=a.overlay, overrides=overrides)
    _check_llm_flags(a, cfg)
    _check_yoke(cfg)
    if cfg.run.director == "replay" or a.replay_log is not None:
        raise ConfigError(
            "a replay continues an existing state: use `run --init-from SRC --director replay --replay-log "
            "LOG` or `resume RUN --director replay`"
        )
    out = Path(a.out) if a.out else _default_out(a.preset or DEFAULT_PRESET, cfg.run.seed)
    rd = RunDir.create(out, cfg, overwrite=a.overwrite)
    rd.update_meta(start_step=0)
    _announce(rd, cfg, 0, cfg.run.steps, a.quiet)
    return _launch(rd, cfg, [init_state(cfg)], cfg.run.steps, a.quiet)


def _fork_source(spec: str) -> tuple[Path, str]:
    """``(state file, origin)`` for ``--init-from RUN[:STEP]|FILE.safetensors``."""
    p = Path(spec)
    if p.suffix == ".safetensors" or p.is_file():
        if not p.is_file():
            raise CheckpointError(f"--init-from: no state file {p}")
        return p, str(p)
    m = _RUN_STEP_RE.match(spec)
    if m and not p.is_dir() and Path(m["run"]).is_dir():
        rd, step = RunDir.open(Path(m["run"])), int(m["step"])
    else:
        rd, step = RunDir.open(p), None
    path = _select_checkpoint(rd, step)
    return path, f"{rd.path}:{int(checkpoint_metadata(path)['step'])}"


def _fork(a: argparse.Namespace, overrides: list[str]) -> int:
    """Start a new run directory from another run's state with a changed config (§18 Fork)."""
    if a.preset is not None:
        raise ConfigError(
            "--preset cannot be combined with --init-from (the source state's config is the base)"
        )
    src, origin = _fork_source(a.init_from)
    src_cfg = _embedded_config(src)
    # A fork runs no director unless it is asked to (--director), whatever mode the source ran in.
    src_cfg = dataclasses.replace(src_cfg, run=dataclasses.replace(src_cfg.run, director="off"))
    cfg = _layer(src_cfg, overlays=a.overlay, path=a.config, overrides=overrides)
    _check_llm_flags(a, cfg)
    _check_yoke(cfg)
    replay = cfg.run.director == "replay"
    if replay != (a.replay_log is not None):
        raise ConfigError("--director replay and --replay-log LOG go together")
    if replay:  # the whole log is checked before anything is written (exit 2)
        director_log.validate_log(director_log.read_log(Path(a.replay_log)), cfg)
    for key in ("dims", "height"):  # a state never crosses dimensions or world heights (§27.1)
        if getattr(cfg.world, key) != getattr(src_cfg.world, key):
            raise CheckpointError(
                f"world.{key} must equal the source's {getattr(src_cfg.world, key)} "
                "(a fork cannot change the world's dimensions or height)"
            )
    if dims(cfg).d3 and cfg.physics.radius != src_cfg.physics.radius:  # the radius sets z_lo, z_hi (§27.1)
        raise CheckpointError(
            f"physics.radius must equal the source's {src_cfg.physics.radius} in a 3D world "
            "(it sets the floor and surface bounds z_lo, z_hi)"
        )
    if cfg.population.n_init != src_cfg.population.n_init:
        raise CheckpointError(
            f"population.n_init must equal the source's {src_cfg.population.n_init} (clade bins depend on it)"
        )
    s = load_checkpoint(src, cfg, check_hash=False)
    # Structure energy is not in any E, so the conserved TOTAL moves with S: rebase by (S_new − S_old)·n_alive
    # (a stored TOTAL that was already wrong stays wrong and is refused by the admission check below).
    n_alive = s.alive.astype(mx.int32).sum().astype(mx.int64)
    total = s.total + n_alive * (cfg.energy.structure - src_cfg.energy.structure)
    # The environment levers restart inert for the new config (the source's director history does not carry
    # over; a replay fork brings its own log).
    env = default_env(cfg)
    env_reset = any(not np.array_equal(to_numpy(getattr(s, k)), to_numpy(v)) for k, v in env.items())
    s = s._replace(
        diag=mx.zeros_like(s.diag), ledger_cum=mx.zeros_like(s.ledger_cum), total=total, total0=total, **env
    )
    if a.seed is not None:
        s = s._replace(key=init_keys(a.seed)[InitStream.STATE_KEY])
    mx.eval(s)
    problems = full_check(s, cfg)
    if problems:
        print(f"--init-from {a.init_from}: the state violates the new config:", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        return int(ExitCode.CHECKPOINT)
    step = int(to_numpy(s.step))
    out = Path(a.out) if a.out else _default_out("fork", cfg.run.seed)
    rd = RunDir.create(out, cfg, overwrite=a.overwrite, parent=origin)
    rd.update_meta(start_step=step)
    if replay:
        shutil.copyfile(a.replay_log, rd.path / DIRECTOR_LOG_FILE)
    _announce(rd, cfg, step, cfg.run.steps, a.quiet)
    if env_reset and not a.quiet:
        print(
            "note: the source's environment levers were reset to inert values for the new config", flush=True
        )
    box = [s]
    del s  # the runner must hold the only reference to the state (donation rule, F1)
    return _launch(rd, cfg, box, cfg.run.steps, a.quiet)


def _resume_allowed(item: str) -> bool:
    section, _, key = item.partition("=")[0].strip().partition(".")
    return (section == "run" and key in RESUME_KEYS) or (
        section == "director" and key in RESUME_DIRECTOR_KEYS
    )


def _cmd_resume(a: argparse.Namespace) -> int:
    rd = RunDir.open(Path(a.run))
    items = list(a.set)
    if a.director is not None:  # a shortcut for --set run.director=MODE, translated before the key check
        items.append(f"run.director={a.director}")
    items += _llm_overrides(a)
    bad = [item for item in items if not _resume_allowed(item)]
    if bad:
        raise ConfigError(
            f"resume accepts only --set run.<key> for {sorted(RESUME_KEYS)} and director.<key> for "
            f"{sorted(RESUME_DIRECTOR_KEYS)}; got {bad} "
            "(seed, compile, runtime_checks and dynamics changes need `alife run --init-from`)"
        )
    base = rd.config()
    if a.director not in (None, "off") and not base.director.enabled:
        raise ConfigError(
            f"{rd.path} was started with director.enabled=false, so its compiled step cannot take director "
            "actions; fork it with `alife run --init-from RUN --director live` instead"
        )
    cfg = apply_overrides(base, items)
    _check_llm_flags(a, cfg)
    _check_yoke(cfg)
    path = _select_checkpoint(rd, a.checkpoint)
    state = load_checkpoint(path, cfg)
    step = int(to_numpy(state.step))
    if a.steps is not None:
        steps = a.steps
    elif a.until is not None:
        if a.until < step:
            raise ConfigError(f"--until {a.until} is before the checkpoint step {step}")
        steps = a.until - step
    else:
        steps = max(0, int(rd.meta().get("start_step", 0)) + cfg.run.steps - step)
    replay = cfg.run.director == "replay"
    if replay:  # the whole log is checked before anything is written (exit 2)
        director_log.validate_log(director_log.read_log(rd.path / DIRECTOR_LOG_FILE), cfg)
    # Everything is validated; only now touch the run directory.
    if items:
        rd.save_config(cfg)
    rd.truncate_after(step, keep_director=replay)  # a replay never touches the log it replays
    _announce(rd, cfg, step, steps, a.quiet)
    box = [state]
    del state  # the runner must hold the only reference to the state (donation rule, F1)
    return _launch(rd, cfg, box, steps, a.quiet)


# --- render ---------------------------------------------------------------------------------------------


def _cmd_render(a: argparse.Namespace) -> int:
    cfg, rd, path = _target_config(a.run, a.checkpoint)
    config_mod.require_2d(cfg, "alife render")  # before the state is loaded
    state = load_checkpoint(path, cfg)
    size = _parse_size(a.size)
    view = _parse_view(a.view, cfg)
    layers = _parse_layers(a.layers)
    media = rd.media_dir if rd is not None else Path("media")
    step = int(to_numpy(state.step))

    def frame(s: State) -> np.ndarray:
        return to_numpy(render_frame(s, cfg, size, view, layers, a.color))

    if a.steps == 0:
        out = Path(a.out) if a.out else media / f"frame_{step:010d}.png"
        if not out.suffix:
            out = out / f"frame_{step:010d}.png"
        elif out.suffix.lower() != ".png":
            raise ConfigError(f"--out {out}: a single frame (--steps 0) is written as .png")
        out.parent.mkdir(parents=True, exist_ok=True)
        write_png(out, frame(state))
        print(out)
        return int(ExitCode.OK)

    out = Path(a.out) if a.out else media / f"render_{step:010d}_{step + a.steps:010d}.mp4"
    if out.suffix and out.suffix.lower() not in VIDEO_SUFFIXES:
        raise ConfigError(f"--out {out}: use a directory (PNG sequence) or one of {sorted(VIDEO_SUFFIXES)}")
    step_fn = make_step(cfg)
    with VideoWriter(out, size, a.fps) as vw:
        vw.write(frame(state))
        for i in range(1, a.steps + 1):
            state, out_i = step_fn(state)
            mx.eval(state, out_i)
            if i % a.every == 0:
                vw.write(frame(state))
        n = vw.frames
        where = vw.out_dir if vw.fallback else out
    print(f"{where} ({n} frames)")
    return int(ExitCode.OK)


# --- inspect --------------------------------------------------------------------------------------------


def _agent_info(uid: int, s: State | State3, cfg: Config, rd: RunDir | None) -> dict[str, Any] | None:
    """Row, bonds, birth/death records and ancestor chain of ``uid`` as of the state's step (None if unknown).

    Ancestors come from the run's birth events; the chain is complete when it ends at a founder or immigrant.
    In 3D ``fields`` also holds the (x, y, z) ``pos``, ``pitch`` and ``buoy_cap``, and the info gains
    ``height`` (z) and ``grounded`` (``z ≤ Dims.z_ground``, as ``physics.grounded``) (§27.11).
    """
    step = int(to_numpy(s.step))
    n_init = cfg.population.n_init
    host = {name: to_numpy(getattr(s, name)) for name in ("alive", "uid")}
    rows = np.flatnonzero(host["alive"] & (host["uid"] == uid))
    info: dict[str, Any] = {"uid": uid, "alive": bool(rows.size)}
    parent: int | None = -1 if 0 <= uid < n_init else None  # founders have no birth record
    if rows.size:
        r = int(rows[0])
        a = {name: to_numpy(getattr(s, name)[r]) for name in agent_fields(cfg) if name not in ("W1", "W2")}
        info["row"] = r
        info["age"] = step - 1 - int(a["birth_step"])
        info["fields"] = {k: v.tolist() for k, v in a.items()}
        if dims(cfg).d3:
            info["height"] = float(a["pos"][2])
            info["grounded"] = float(a["pos"][2]) <= dims(cfg).z_ground
        info["bonds"] = [
            {"slot": b, "row": p, "uid": int(host["uid"][p])}
            for b, p in enumerate(a["partner"].tolist())
            if p >= 0
        ]
        parent = int(a["parent_uid"])
    parents: dict[int, int] = {}
    if rd is not None:
        births, deaths = rd.read_events("births"), rd.read_events("deaths")
        bk, dk = births["step"] < step, deaths["step"] < step  # records up to this checkpoint only
        parents = dict(zip(births["uid"][bk].tolist(), births["parent_uid"][bk].tolist(), strict=True))
        if (hit := np.flatnonzero(bk & (births["uid"] == uid))).size:
            info["birth"] = {k: v[hit[0]].item() for k, v in births.items()}
            parent = int(births["parent_uid"][hit[0]])
        if (hit := np.flatnonzero(dk & (deaths["uid"] == uid))).size:
            rec = {k: v[hit[0]].item() for k, v in deaths.items()}
            rec["cause"] = DeathCause(rec["cause"]).name
            info["death"] = rec
    if parent is None and "death" not in info:
        return None
    chain: list[int] = []
    while parent is not None and parent >= 0 and parent not in chain:
        chain.append(parent)
        parent = -1 if 0 <= parent < n_init else parents.get(parent)
    info["ancestors"] = chain
    info["ancestors_complete"] = parent == -1
    return info


def _grounded_frac(state: State | State3, cfg: Config) -> float | None:
    """The share of the alive agents of a 3D state that are grounded (``physics.grounded``), None if none."""
    alive, on_floor = to_numpy(state.alive), to_numpy(physics.grounded(state.pos, cfg))
    n = int(alive.sum())
    return int((alive & on_floor).sum()) / n if n else None


def _cmd_inspect(a: argparse.Namespace) -> int:
    cfg, state, rd, path = _load_target(a.run, a.checkpoint)
    snap = metrics.snapshot_metrics(state, cfg, heavy=True)
    info: dict[str, Any] = {
        "target": str(rd.path if rd is not None else path),
        "checkpoint": str(path),
        "step": int(to_numpy(state.step)),
        "config": _summary(cfg),
        "carrying_capacity_est": metrics.carrying_capacity(state, cfg),
        "ledger_error": snap["ledger_error"],
        "diag": snap["diag"],
        "diag_flags": runner.diag_names(snap["diag"]),
        "n_alive": int(to_numpy(state.alive).sum()),
        "next_uid": int(to_numpy(state.next_uid)),
        "state": snap,
    }
    if dims(cfg).d3:  # the fraction of the alive agents on the floor now (the rows hold the interval means)
        info["grounded_frac"] = _grounded_frac(state, cfg)
    if rd is not None:
        meta = rd.meta()
        info["status"] = meta.get("status")
        info["parent"] = meta.get("parent")
        info["checkpoints"] = [st for st, _ in rd.checkpoints()]
        rows = rd.metrics_rows()
        info["last_metrics"] = rows[-1] if rows else None
    problems: list[str] = []
    if a.check:
        problems = full_check(state, cfg)
        info["check"] = {"ok": not problems, "problems": problems}
    code = ExitCode.INVARIANT if problems else ExitCode.OK
    if a.agent is not None:
        agent = _agent_info(a.agent, state, cfg, rd)
        info["agent"] = agent
        if agent is None:
            print(
                f"no agent with uid {a.agent} in {info['target']} up to step {info['step']}", file=sys.stderr
            )
            code = max(code, ExitCode.ERROR)
    if a.json:
        print(_dumps(info))
    else:
        _print_inspect(info)
    return int(code)


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def _height_line(info: Mapping[str, Any]) -> str:
    """The 3D line of ``inspect``: the height distribution, grounded and surface shares, and the bands."""
    s, c = info["state"], info["config"]
    if s["height_mean"] is None:
        return f"height (world {c['world_height']:g}): nobody alive"
    quart = "/".join(_fmt(s[f"height_q{p}"]) for p in (25, 50, 75))
    bands = " ".join(f"{x:.2f}" for x in s["band_frac"])
    return (
        f"height (world {c['world_height']:g}): mean {_fmt(s['height_mean'])}, quartiles {quart}, "
        f"grounded {_fmt(info['grounded_frac'])}, surface {_fmt(s['surface_frac'])}, bands [{bands}]; "
        f"speed {_fmt(s['speed_mean'])}, |pitch| {_fmt(s['pitch_abs_mean'])}, buoyancy cap median "
        f"{_fmt(s['buoy_cap_q50'])}"
    )


def _print_inspect(info: Mapping[str, Any]) -> None:
    s, c = info["state"], info["config"]
    ledger = "OK" if info["ledger_error"] == 0 else f"ERROR {info['ledger_error']}"
    print(f"{info['target']}  (checkpoint step {info['step']}, status {info.get('status', '-')})")
    print("config: " + ", ".join(f"{k}={_fmt(v)}" for k, v in c.items()))
    print(f"carrying capacity estimate: {info['carrying_capacity_est']:.0f}")
    if "checkpoints" in info:
        print(f"checkpoints: {info['checkpoints']}")
    print(
        f"population: {info['n_alive']} alive, next_uid {info['next_uid']}, "
        f"generation max {_fmt(s['generation_max'])}, mean age {_fmt(s['age_mean'])}"
    )
    print(
        f"energy: agents {s['E_agents']}, structure {s['E_structure']}, field {s['E_field']}, "
        f"total {s['E_total']}; ledger {ledger}"
    )
    print(f"diag: {info['diag']} {info['diag_flags']}")
    print(
        f"bodies: {s['bodies_n']} (largest {s['bodies_largest']}, converged {s['bodies_converged']}), "
        f"hist {s['bodies_hist']}"
    )
    print(
        f"clades: {s['clades_alive']} alive, top share {_fmt(s['top_clade_share'])}, "
        f"entropy {_fmt(s['clade_entropy'])}, simpson {_fmt(s['clade_simpson'])}"
    )
    if "height_mean" in s:  # a 3D world
        print(_height_line(info))
    if info.get("last_metrics") is not None:
        print("last metrics row: " + json.dumps(info["last_metrics"], default=_json_default))
    if "check" in info:
        chk = info["check"]
        print("full_check: OK" if chk["ok"] else f"full_check: {len(chk['problems'])} problem(s)")
        for p in chk["problems"]:
            print(f"  {p}")
    if info.get("agent") is not None:
        print("agent: " + json.dumps(info["agent"], default=_json_default))


# --- view / bench (WP6) ---------------------------------------------------------------------------------


def _cmd_view(a: argparse.Namespace) -> int:
    if a.run is not None and (a.preset is not None or a.set):
        raise ConfigError("view takes either RUN or --preset/--set, not both")
    if a.run is not None:
        cfg, rd, path = _target_config(a.run, a.checkpoint)
        config_mod.require_2d(cfg, "alife view")
        state = load_checkpoint(path, cfg)
    else:
        if a.checkpoint is not None:
            raise ConfigError("--checkpoint needs RUN")
        cfg = load_config(preset=a.preset, overrides=a.set)
        config_mod.require_2d(cfg, "alife view")
        state, rd = init_state(cfg), None
    box = [state]
    del state
    from alife.viewer import run_viewer

    size = _parse_size(a.size)
    try:
        run_viewer(cfg, box.pop(), rundir=rd, size=size, steps_per_frame=a.steps_per_frame)
    except ImportError as e:  # pygame-ce (the optional 'viewer' extra) is missing
        print(f"error: {e}", file=sys.stderr)
        return int(ExitCode.ERROR)
    return int(ExitCode.OK)


def _cmd_stream(a: argparse.Namespace) -> int:
    """``alife stream`` (§27.11): the binary frame protocol on stdout, for an external viewer (2D or 3D)."""
    if a.run is not None and (a.preset is not None or a.set):
        raise ConfigError("stream takes either RUN (or a state FILE) or --preset/--set, not both")
    if a.run is None and a.preset is None:
        raise ConfigError("stream needs a source: a run directory, a state file or --preset")
    view = stream.View(color=a.color, fields=a.fields or a.field_z is not None, field_z=a.field_z or 0)
    box: list[State | State3] = []
    rd: RunDir | None = None
    if a.follow:
        for flag, value in (("--checkpoint", a.checkpoint), ("--steps", a.steps), ("--every", a.every)):
            if value is not None:
                raise ConfigError(f"{flag} does not apply to --follow, which streams each new checkpoint")
        if a.run is None or not Path(a.run).is_dir():
            raise ConfigError("--follow needs a run directory")
        rd = RunDir.open(Path(a.run))
        cfg, source = rd.config(), "run"
        view.check(cfg)
    elif a.run is not None:
        cfg, rd, path = _target_config(a.run, a.checkpoint)
        source = "file" if rd is None else "run"
        view.check(cfg)
        box.append(load_checkpoint(path, cfg))
    else:
        if a.checkpoint is not None:
            raise ConfigError("--checkpoint needs RUN")
        cfg, source = load_config(preset=a.preset, overrides=a.set), "preset"
        view.check(cfg)
        box.append(init_state(cfg))
    control = stream.NoControl() if a.no_control else stream.FdControl()
    stream.serve(
        cfg,
        mode="follow" if a.follow else "simulate",
        source=source,
        run=a.run,
        state=box,
        rd=rd if a.follow else None,
        view=view,
        every=a.every or 1,
        max_steps=a.steps,
        poll_s=a.poll,
        control=control,
    )
    return int(ExitCode.OK)


def _floats(text: str) -> tuple[float, ...]:
    try:
        return tuple(float(x) for x in text.split(",") if x.strip())
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"expected comma-separated numbers, got {text!r}") from e


def _ints(text: str) -> tuple[int, ...]:
    try:
        return tuple(int(x) for x in text.split(",") if x.strip())
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"expected comma-separated integers, got {text!r}") from e


def _windows(text: str) -> tuple[tuple[int, int | None], ...]:
    """``A:B,C:D,E:end``: classification windows in steps (``end`` = the end of the run)."""
    out = []
    for part in text.split(","):
        a, sep, b = part.partition(":")
        try:
            if not sep:
                raise ValueError
            out.append((int(a), None if b.strip() == "end" else int(b)))
        except ValueError as e:
            raise argparse.ArgumentTypeError(
                f"expected windows like 0:120000,120000:end, got {text!r}"
            ) from e
    return tuple(out)


def _cmd_activity(a: argparse.Namespace) -> int:
    from alife.analysis import report  # local: host-only numpy analysis, kept off the simulation import path

    rd = RunDir.open(Path(a.run))
    out = Path(a.out) if a.out else rd.path / "activity"
    st = report.Settings(
        grid_every=a.grid,
        marker_rates=a.marker_rates,
        cohort_horizons=a.horizons,
        shadows=a.shadows,
        seed=a.seed,
        windows=a.windows,
    )
    plots = not a.no_plots
    if plots:
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            print("matplotlib is missing (uv sync --extra analysis): writing tables only", file=sys.stderr)
            plots = False
    report.run_report(
        rd.path, out, st, reuse=a.reuse, plots=plots, log=lambda m: None if a.quiet else print(m)
    )
    if not a.quiet:
        print(f"wrote {out}")
    return ExitCode.OK


def _cmd_bench(a: argparse.Namespace) -> int:
    from alife.bench import run_bench

    cfg = load_config(preset=a.preset, overlays=a.overlay, overrides=a.set)
    report = run_bench(
        cfg,
        steps=a.steps,
        warmup=a.warmup,
        stages=a.stages,
        determinism=a.determinism,
        brain_variants=a.brain_variants,
    )
    text = _dumps(report)
    if a.json is not None:
        out = Path(a.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    print(text)
    det = report.get("determinism")
    if isinstance(det, Mapping) and det.get("match") is False:
        print("determinism check FAILED: the two runs' state digests differ", file=sys.stderr)
        return int(ExitCode.ERROR)
    return int(ExitCode.OK)


# --- llm-check ------------------------------------------------------------------------------------------

_LLM_CHECK_MESSAGES: tuple[dict[str, str], ...] = (
    {"role": "system", "content": "You are a connectivity check for a simulation. Reply only with JSON."},
    {
        "role": "user",
        "content": 'Choose the action "none" with empty params and the expected_effect "nothing", and '
        'put "ok" in the journal.',
    },
)


def _cmd_llm_check(a: argparse.Namespace) -> int:
    """One tiny structured-output request under the director's own client and schema; one line out (§21)."""
    cfg = load_config(preset=a.preset)
    url = a.llm_url if a.llm_url is not None else cfg.run.llm_url
    model = a.llm_model if a.llm_model is not None else cfg.run.llm_model
    if model.strip() == "":
        raise ConfigError("--llm-model must not be empty")
    d = cfg.director
    client = director_llm.Client(url, model, a.timeout, d.llm_max_tokens, d.llm_temperature)
    ex = client.chat(client.body(_LLM_CHECK_MESSAGES, cfg, 0))
    failure = ex.error
    if ex.text is not None:
        try:
            director_llm.parse_reply(ex.text, cfg)
        except director_llm.ReplyError as e:
            failure = f"invalid: {e}"
    if failure is not None:
        print(f"FAIL {failure}", flush=True)
        return int(ExitCode.ERROR)
    print(f"OK model={model} latency={ex.latency_s:.2f}s", flush=True)
    return int(ExitCode.OK)


# --- parser ---------------------------------------------------------------------------------------------


def _config_args(p: argparse.ArgumentParser, *, overlays: bool = True) -> None:
    p.add_argument("--preset", choices=PRESETS, default=None, help=f"default {DEFAULT_PRESET}")
    if overlays:
        p.add_argument("--overlay", action="append", default=[], choices=OVERLAYS)
    p.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE")


def _llm_args(p: argparse.ArgumentParser) -> None:
    """The LLM tier and persona shortcuts (DIRECTOR.md §8); each is a shortcut for a ``--set`` item."""
    g = p.add_mutually_exclusive_group()
    g.add_argument(
        "--llm",
        action="store_true",
        help=f"enable the LLM tier: director.llm_every={LLM_DEFAULT_EVERY} (needs --director live)",
    )
    g.add_argument(
        "--llm-every", type=_non_negative, default=None, metavar="N", help="director.llm_every (0 = off)"
    )
    p.add_argument(
        "--llm-url", default=None, metavar="URL", help="run.llm_url (an OpenAI-compatible endpoint)"
    )
    p.add_argument("--llm-model", default=None, metavar="NAME", help="run.llm_model")
    p.add_argument("--persona", choices=PERSONAS, default=None, help="director.persona")
    p.add_argument("--goal", default=None, metavar="TEXT", help="director.goal (the Architect's goal)")
    p.add_argument(
        "--yoke-log",
        default=None,
        metavar="PATH",
        help="director.yoke_log: the director.jsonl the random persona follows (needs --persona random)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="alife", description="Neural artificial life on Apple MLX.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run", help="start a run (or fork one with --init-from)")
    _config_args(p)
    p.add_argument("--config", default=None, metavar="PATH", help="TOML or JSON overrides")
    p.add_argument("--seed", type=_non_negative, default=None)
    p.add_argument("--steps", type=_non_negative, default=None)
    p.add_argument("--out", default=None, metavar="DIR")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--init-from", default=None, metavar="RUN[:STEP]|FILE.safetensors")
    p.add_argument("--metrics-every", type=_positive, default=None)
    p.add_argument("--checkpoint-every", type=_positive, default=None)
    p.add_argument("--checks", choices=("cheap", "off"), default=None)
    p.add_argument("--no-compile", action="store_true")
    p.add_argument(
        "--director",
        choices=DIRECTOR_MODES,
        default=None,
        help="director mode: live decides and logs; replay applies --replay-log (with --init-from)",
    )
    _llm_args(p)
    p.add_argument(
        "--replay-log", default=None, metavar="PATH", help="director.jsonl to replay (with --init-from)"
    )
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=_cmd_run)

    p = sub.add_parser("resume", help="continue a run from a checkpoint (bitwise identical)")
    p.add_argument("run", metavar="RUN")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--steps", type=_non_negative, default=None)
    g.add_argument("--until", type=_non_negative, default=None, metavar="STEP")
    p.add_argument("--checkpoint", type=_checkpoint_arg, default=None, metavar="STEP|latest")
    p.add_argument("--set", action="append", default=[], metavar="run.KEY=VALUE|director.KEY=VALUE")
    p.add_argument(
        "--director",
        choices=DIRECTOR_MODES,
        default=None,
        help="shortcut for --set run.director=MODE (live or replay need a run started with director.enabled)",
    )
    _llm_args(p)
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=_cmd_resume)

    p = sub.add_parser(
        "render",
        help="re-simulate from a checkpoint and write a PNG or a video "
        "(the director is not replayed: after an intervention it differs from the run)",
    )
    p.add_argument("run", metavar="RUN")
    p.add_argument("--checkpoint", type=_checkpoint_arg, default=None, metavar="STEP|latest")
    p.add_argument("--steps", type=_non_negative, default=0)
    p.add_argument("--every", type=_positive, default=1, metavar="K")
    p.add_argument("--out", default=None, metavar="PATH")
    p.add_argument("--size", default="1024x1024", metavar="WxH")
    p.add_argument("--fps", type=_positive, default=30)
    p.add_argument("--layers", default=",".join(sorted(DEFAULT_LAYERS)))
    p.add_argument("--color", choices=COLOR_MODES, default="tag")
    p.add_argument("--view", default=None, metavar="CX,CY,ZOOM")
    p.set_defaults(func=_cmd_render)

    p = sub.add_parser("inspect", help="summarise a checkpoint")
    p.add_argument("run", metavar="RUN")
    p.add_argument("--checkpoint", type=_checkpoint_arg, default=None, metavar="STEP|latest")
    p.add_argument("--json", action="store_true")
    p.add_argument("--check", action="store_true", help="run invariants.full_check; exit 3 on a violation")
    p.add_argument("--agent", type=int, default=None, metavar="UID")
    p.set_defaults(func=_cmd_inspect)

    p = sub.add_parser(
        "view",
        help="live viewer (needs the 'viewer' extra); it re-simulates without the director, so after an "
        "intervention it differs from a director run",
    )
    p.add_argument("run", nargs="?", default=None, metavar="RUN")
    _config_args(p, overlays=False)
    p.add_argument("--checkpoint", type=_checkpoint_arg, default=None, metavar="STEP|latest")
    p.add_argument("--steps-per-frame", type=_positive, default=1)
    p.add_argument("--size", default="900x900", metavar="WxH")
    p.set_defaults(func=_cmd_view)

    p = sub.add_parser("stream", help=_STREAM_HELP, description=_STREAM_HELP)
    p.add_argument("run", nargs="?", default=None, metavar="RUN|FILE")
    _config_args(p, overlays=False)
    p.add_argument("--checkpoint", type=_checkpoint_arg, default=None, metavar="STEP|latest")
    p.add_argument(
        "--every", type=_positive, default=None, metavar="K", help="a frame every K steps (default 1)"
    )
    p.add_argument(
        "--steps",
        type=_non_negative,
        default=None,
        metavar="N",
        help="stop after N steps (default: run until the reader closes or stdin ends)",
    )
    p.add_argument(
        "--follow",
        action="store_true",
        help="no stepping: stream the latest checkpoint of RUN and every newer one as the runner writes it",
    )
    p.add_argument(
        "--poll",
        type=_positive_seconds,
        default=1.0,
        metavar="SECONDS",
        help="--follow: checkpoint poll period",
    )
    p.add_argument("--color", choices=COLOR_MODES, default=COLOR_MODES[0], help="initial colour mode")
    p.add_argument("--fields", action="store_true", help="include the field slice in each frame")
    p.add_argument(
        "--field-z",
        type=_non_negative,
        default=None,
        metavar="K",
        help="field slice layer (implies --fields; default 0)",
    )
    p.add_argument(
        "--no-control",
        action="store_true",
        help="do not read stdin (by default it carries control lines, and its EOF ends the stream)",
    )
    p.set_defaults(func=_cmd_stream)

    p = sub.add_parser("bench", help="throughput, donation health, determinism")
    _config_args(p)
    p.add_argument("--steps", type=_positive, default=200)
    p.add_argument("--warmup", type=_non_negative, default=20)
    p.add_argument("--stages", action="store_true")
    p.add_argument("--determinism", action="store_true")
    p.add_argument("--brain-variants", action="store_true")
    p.add_argument("--json", default=None, metavar="PATH")
    p.set_defaults(func=_cmd_bench)

    p = sub.add_parser(
        "activity",
        help="Bedau–Packard evolutionary activity of the lineages against null shadows (docs/ACTIVITY.md)",
    )
    p.add_argument("run", metavar="RUN")
    p.add_argument("--out", default=None, metavar="DIR", help="output directory (default RUN/activity)")
    p.add_argument(
        "--grid", type=_positive, default=10_000, metavar="N", help="grid and cohort spacing (steps)"
    )
    p.add_argument("--marker-rates", type=_floats, default=(0.003, 0.01, 0.03, 0.1), metavar="R,...")
    p.add_argument(
        "--horizons",
        type=_ints,
        default=(100_000, 200_000, 500_000),
        metavar="H,...",
        help="cohort-pool horizons (steps)",
    )
    p.add_argument("--shadows", type=_positive, default=2, metavar="N", help="replicates of each shadow")
    p.add_argument("--seed", type=_non_negative, default=0)
    p.add_argument(
        "--windows",
        type=_windows,
        default=((0, 120_000), (120_000, 1_000_000), (1_000_000, None)),
        metavar="A:B,...",
    )
    p.add_argument("--reuse", action="store_true", help="reuse OUT/activity.npz instead of recomputing")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=_cmd_activity)

    p = sub.add_parser(
        "llm-check",
        help="send one tiny structured-output request to the director's LLM endpoint; prints OK or FAIL",
    )
    p.add_argument("--llm-url", default=None, metavar="URL", help="default run.llm_url")
    p.add_argument("--llm-model", default=None, metavar="NAME", help="default run.llm_model")
    p.add_argument("--timeout", type=_positive_seconds, default=60.0, metavar="SECONDS")
    p.add_argument(
        "--preset", choices=PRESETS, default=None, help=f"world size of the schema; default {DEFAULT_PRESET}"
    )
    p.set_defaults(func=_cmd_llm_check)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point; returns a ``runner.ExitCode`` value."""
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as e:  # usage errors (2) and --help (0)
        return e.code if isinstance(e.code, int) else int(ExitCode.CONFIG)
    try:
        return int(args.func(args))
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return int(ExitCode.CONFIG)
    except CheckpointError as e:
        print(f"checkpoint error: {e}", file=sys.stderr)
        return int(ExitCode.CHECKPOINT)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return int(ExitCode.INTERRUPTED)
    except (FileNotFoundError, FileExistsError, NotADirectoryError) as e:
        print(f"error: {e}", file=sys.stderr)
        return int(ExitCode.ERROR)
    except Exception:
        traceback.print_exc()
        return int(ExitCode.ERROR)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
