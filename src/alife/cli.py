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
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from alife import config as config_mod
from alife import metrics, physics, play, runner, spatial, stream, weather_spinup
from alife.analysis import houses as house_rebuild
from alife.config import (
    ALL_OVERLAYS,
    DEFAULT_PRESET,
    DIRECTOR_MODES,
    DIRECTOR_POLICY_KEYS,
    OBSERVER_KEYS,
    PERSONAS,
    PLAYER_MODE,
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
from alife.game import housefile
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
from alife.runio import (
    DIRECTOR_LOG_FILE,
    PRE_PLAY_CONFIG_FILE,
    CheckpointError,
    PlayLock,
    PlayLockError,
    RunDir,
    checkpoint_metadata,
    load_checkpoint,
    play_lock_holder,
    write_state_file,
)
from alife.runner import ExitCode
from alife.state import (
    DeathCause,
    State,
    State3,
    agent_fields,
    default_env,
    init_state,
    state_digest,
    to_numpy,
)
from alife.step import make_step

# run.* keys that resume may change: observers, cadences and the director mode (§18). Everything else goes
# through a fork. The director's policy keys (director.persona, ...) and the observer keys game.observe and
# game.house_rate, game.tracers, game.tracer_period (RESUME_GAME_KEYS) may change too: they never reach the
# trajectory.
RESUME_KEYS: frozenset[str] = frozenset(
    {
        "steps",
        "metrics_every",
        "checkpoint_every",
        "keep_checkpoints",
        "heavy_metrics_every",
        "components_iters",
        "diversity_sample",
        "eco_every",
        "wx_every",
        "director",
        "llm_url",
        "llm_model",
    }
)
RESUME_DIRECTOR_KEYS: frozenset[str] = frozenset(DIRECTOR_POLICY_KEYS)
# The observer keys (game.observe, house_rate, tracers, tracer_period) are read by host-side observers only
# and are not hashed, so resume may change them too (§28.6, §29).
RESUME_GAME_KEYS: frozenset[str] = frozenset(key for _, key in OBSERVER_KEYS)

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
        raise CheckpointError(str(e)) from None  # (the message lists the steps that are kept)


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
        if ov not in ALL_OVERLAYS:
            raise ConfigError(f"unknown overlay {ov!r} (choose from {ALL_OVERLAYS})")
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


def _knn_guard(cfg: Config) -> None:
    """Before any step is traced: self-check the fused Metal search once and settle the search ``cfg``
    will use (``spatial.prepare_knn``, §4). An ``ALIFE_KNN`` that cannot be honoured is a config error."""
    try:
        spatial.prepare_knn(cfg)
    except (RuntimeError, ValueError) as e:
        raise ConfigError(str(e)) from e


def _launch(rd: RunDir, cfg: Config, box: list[State], steps: int, quiet: bool) -> int:
    """Run ``steps`` steps from the state in ``box``, with the director ``cfg.run.director`` names (if any).

    The runner must hold the only reference to the state (donation rule, F1), so it is popped here. The search
    this process uses is recorded in ``meta.json`` (``knn_impl``), like ``mlx_version`` and ``device``.
    """
    _knn_guard(cfg)
    rd.update_meta(knn_impl=spatial.knn_impl(cfg))
    director = make_director(rd, cfg, int(to_numpy(box[0].step)))
    return int(runner.run(rd, cfg, box.pop(), steps, quiet=quiet, director=director).exit_code)


# --- run / fork / resume --------------------------------------------------------------------------------


def _check_spinup(a: argparse.Namespace, cfg: Config, src_cfg: Config | None = None) -> None:
    """``--weather-spinup N`` needs the weather on, and a fork's source must not have carried a weather
    state of its own (the spin-up starts from the rest state and replaces it). Checked before any write."""
    if a.weather_spinup is None:
        return
    if not cfg.weather.enabled:
        raise ConfigError("--weather-spinup needs weather.enabled = true (for example --overlay storm)")
    if src_cfg is not None and src_cfg.weather.enabled:
        raise ConfigError(
            "--weather-spinup starts from the rest state: the source run already has an evolving weather "
            "state, so fork it without --weather-spinup"
        )


def _spin_weather(a: argparse.Namespace, rd: RunDir, cfg: Config, box: list[State]) -> int | None:
    """Replace the weather state of ``box[0]`` by the spin-up of ``--weather-spinup`` steps (a no-op without
    the flag). Returns an exit code when interrupted, else None. Records the steps in the run's meta."""
    n = a.weather_spinup
    if n is None:
        return None
    s = box.pop()
    step = int(to_numpy(s.step))
    last = 0

    def progress(done: int, total: int) -> None:
        nonlocal last
        if not a.quiet and (done == total or done - last >= 1000):
            print(f"weather spin-up: {done}/{total} steps", flush=True)
            last = done

    try:
        wx = weather_spinup.spinup(cfg, n, step=step, key=s.key, compile=cfg.run.compile, progress=progress)
    except KeyboardInterrupt:
        rd.update_meta(status="interrupted")
        print("interrupted during the weather spin-up", file=sys.stderr)
        return int(ExitCode.INTERRUPTED)
    box.append(s._replace(wx=wx))
    rd.update_meta(weather_spinup=n)
    return None


# called with a new run's directory and config after every check and before the first write
BeforeCreate = Callable[[Path, Config], None]


def _prepare_run(
    a: argparse.Namespace,
    overrides: list[str],
    *,
    before_create: BeforeCreate | None = None,
    extra_meta: Mapping[str, Any] | None = None,
) -> tuple[RunDir, Config, list[State], int | None]:
    """The new-run path of ``run``, up to ``_launch``: ``(rundir, cfg, [state], exit code or None)``. The exit
    code is set when the weather spin-up was interrupted. ``before_create(out, cfg)`` runs after every check
    and before the run directory is created (``play`` takes its lock there); ``extra_meta`` goes into the
    first ``meta.json`` (``play`` marks the world it builds)."""
    cfg = load_config(a.config, preset=a.preset, overlays=a.overlay, overrides=overrides)
    _check_llm_flags(a, cfg)
    _check_yoke(cfg)
    if cfg.run.director == "replay" or a.replay_log is not None:
        raise ConfigError(
            "a replay continues an existing state: use `run --init-from SRC --director replay --replay-log "
            "LOG` or `resume RUN --director replay`"
        )
    _knn_guard(cfg)  # before RunDir.create, which records the search in meta.json
    _check_spinup(a, cfg)
    out = Path(a.out) if a.out else _default_out(a.preset or DEFAULT_PRESET, cfg.run.seed)
    if before_create is not None:
        before_create(out, cfg)
    rd = RunDir.create(out, cfg, overwrite=a.overwrite, extra_meta=extra_meta)
    rd.update_meta(start_step=0)
    _announce(rd, cfg, 0, cfg.run.steps, a.quiet)
    box = [init_state(cfg)]
    return rd, cfg, box, _spin_weather(a, rd, cfg, box)


def _cmd_run(a: argparse.Namespace) -> int:
    overrides = _run_overrides(a)
    if a.init_from is not None:
        return _fork(a, overrides)
    rd, cfg, box, code = _prepare_run(a, overrides)
    return code if code is not None else _launch(rd, cfg, box, cfg.run.steps, a.quiet)


def _fork_lookup(spec: str) -> tuple[Path, str, RunDir | None, int | None]:
    """``(state file, origin, run directory, step)`` for ``--init-from RUN[:STEP]|FILE.safetensors``: the
    run directory and the checkpoint's step are None for a bare state file."""
    p = Path(spec)
    if p.suffix == ".safetensors" or p.is_file():
        if not p.is_file():
            raise CheckpointError(f"--init-from: no state file {p}")
        return p, str(p), None, None
    m = _RUN_STEP_RE.match(spec)
    if m and not p.is_dir() and Path(m["run"]).is_dir():
        rd, step = RunDir.open(Path(m["run"])), int(m["step"])
    else:
        rd, step = RunDir.open(p), None
    path = _select_checkpoint(rd, step)
    at = int(checkpoint_metadata(path)["step"])
    return path, f"{rd.path}:{at}", rd, at


def _fork_source(spec: str) -> tuple[Path, str]:
    """``(state file, origin)`` for ``--init-from RUN[:STEP]|FILE.safetensors``."""
    path, origin, _, _ = _fork_lookup(spec)
    return path, origin


def _prepare_fork(
    a: argparse.Namespace,
    overrides: list[str],
    *,
    before_create: BeforeCreate | None = None,
    require_spinup: bool = False,
    extra_meta: Mapping[str, Any] | None = None,
    link_origin: bool = False,
) -> tuple[RunDir | None, Config, list[State], int | None]:
    """A new run directory from another run's state with a changed config (§18 Fork), up to ``_launch``:
    ``(rundir, cfg, [state], exit code or None)``: the code is set when the weather spin-up was interrupted
    (``rundir`` exists) or the state violates the new config (``rundir`` is None: nothing was written).

    ``before_create(out, cfg)`` runs after every check and before the directory is created (``play`` takes its
    lock there). ``require_spinup`` refuses a fork that turns the weather on without ``--weather-spinup N`` of
    at least 1 (the rest state has no storms for about 15,000 steps; ``alife play``'s rule). ``extra_meta``
    goes into the first ``meta.json``."""
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
    _check_spinup(a, cfg, src_cfg)
    if require_spinup and cfg.weather.enabled and not src_cfg.weather.enabled and not a.weather_spinup:
        raise ConfigError(
            "the fork turns the weather on: it needs --weather-spinup N with N >= 1 (the rest state has no "
            "storms for about 15,000 steps)"
        )
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
    # The planet's first save links the source save instead of writing the state again, when the state is
    # the source's byte for byte (checked below) under the source's own config hash (DESIGN 34.11).
    src_digest = (
        state_digest(s)
        if link_origin
        and a.weather_spinup is None
        and checkpoint_metadata(src)["config_hash"] == config_hash(cfg)
        else None
    )
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
        return None, cfg, [], int(ExitCode.CHECKPOINT)  # (nothing was written)
    step = int(to_numpy(s.step))
    linked = src_digest is not None and state_digest(s) == src_digest
    _knn_guard(cfg)
    out = Path(a.out) if a.out else _default_out("fork", cfg.run.seed)
    if before_create is not None:
        before_create(out, cfg)
    rd = RunDir.create(out, cfg, overwrite=a.overwrite, parent=origin, extra_meta=extra_meta)
    rd.update_meta(start_step=step)
    if linked:
        rd.origin_link = (step, src)
    if replay:
        shutil.copyfile(a.replay_log, rd.path / DIRECTOR_LOG_FILE)
    _announce(rd, cfg, step, cfg.run.steps, a.quiet)
    if env_reset and not a.quiet:
        print(
            "note: the source's environment levers were reset to inert values for the new config", flush=True
        )
    box = [s]
    del s  # the runner must hold the only reference to the state (donation rule, F1)
    return rd, cfg, box, _spin_weather(a, rd, cfg, box)


def _fork(a: argparse.Namespace, overrides: list[str]) -> int:
    """Start a new run directory from another run's state with a changed config (§18 Fork)."""
    rd, cfg, box, code = _prepare_fork(a, overrides)
    if code is not None:
        return code
    assert rd is not None
    return _launch(rd, cfg, box, cfg.run.steps, a.quiet)


def _resume_allowed(item: str) -> bool:
    section, _, key = item.partition("=")[0].strip().partition(".")
    return (
        (section == "run" and key in RESUME_KEYS)
        or (section == "director" and key in RESUME_DIRECTOR_KEYS)
        or (section == "game" and key in RESUME_GAME_KEYS)
    )


def _refuse_if_played(rd: RunDir) -> None:
    """``resume`` (and any other command that would cut or rewrite a run) refuses a directory that an
    ``alife play`` process holds (DESIGN §33.3.3): the play would be changed under it."""
    holder = play_lock_holder(rd.path)
    if holder is not None:
        raise ConfigError(f"{rd.path} is being played by process {holder} (play.lock): stop that play first")


def _cmd_resume(a: argparse.Namespace) -> int:
    rd = RunDir.open(Path(a.run))
    _refuse_if_played(rd)
    items = list(a.set)
    if a.director is not None:  # a shortcut for --set run.director=MODE, translated before the key check
        items.append(f"run.director={a.director}")
    items += _llm_overrides(a)
    bad = [item for item in items if not _resume_allowed(item)]
    if bad:
        raise ConfigError(
            f"resume accepts only --set run.<key> for {sorted(RESUME_KEYS)}, director.<key> for "
            f"{sorted(RESUME_DIRECTOR_KEYS)} and game.<key> for {sorted(RESUME_GAME_KEYS)}; got {bad} "
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
    _knn_guard(cfg)
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
    elif cfg.run.director == "live":  # a log of rules this code does not implement is refused (exit 2) too
        director_log.rules_version_of(rd.read_jsonl(DIRECTOR_LOG_FILE))
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
    _knn_guard(cfg)
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
    _knn_guard(cfg)
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
    _knn_guard(cfg)
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


# --- play -----------------------------------------------------------------------------------------------

_PLAY_HELP = (
    "play a world: the engine under a fixed-timestep scheduler (pause, 0.25x to 2x of 14.4 steps/s, max) "
    "with the game observers on, writing the play records of DESIGN 33.1 after an `alife stream` header on "
    "stdout and reading control lines on stdin: pause, run, step [N], rate M|max, color MODE, every K|auto, "
    "focus UID|off, follow UID|off, idnt, labels on|off, quit (stdin EOF = quit). RUN is a play world (a "
    "directory with play.json); research runs are forked (--init-from), or played in place with --adopt. A "
    "world is saved every 6000 steps and on quit."
)


def _rate_arg(text: str) -> float:
    """``--rate``: ``max`` or a multiplier (0 starts paused)."""
    if text == "max":
        return play.MAX_RATE
    try:
        v = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected 'max' or a multiplier, got {text!r}") from None
    if v != 0 and not play.RATE_MIN <= v <= play.RATE_MAX:
        raise argparse.ArgumentTypeError(f"expected 0 or a multiplier in [{play.RATE_MIN}, {play.RATE_MAX}]")
    return v


def _play_namespace(a: argparse.Namespace) -> argparse.Namespace:
    """The namespace of ``run`` that ``_prepare_run`` and ``_prepare_fork`` read, from ``play``'s:
    ``--director rules`` is ``--director live``, the LLM stays off, nothing is quiet (the window reads the
    progress lines)."""
    run_defaults: dict[str, Any] = dict(
        config=None, seed=None, steps=None, metrics_every=None, checkpoint_every=None, checks=None,
        no_compile=False, director=None, llm=False, llm_every=None, llm_url=None, llm_model=None,
        persona=None, goal=None, yoke_log=None, replay_log=None, overwrite=False, quiet=False,
        weather_spinup=None, init_from=None, preset=None, overlay=[], set=[], out=None,
    )  # fmt: skip
    merged = {**run_defaults, **{k: v for k, v in vars(a).items() if k in run_defaults}}
    merged["director"] = "live" if a.director == "rules" else None
    return argparse.Namespace(**merged)


def _play_items(a: argparse.Namespace) -> list[str]:
    """What play applies on top of a world's config: the cadence and ``game.observe`` (``--set`` items that
    resume allows), and the director mode."""
    mode = "replay" if a.replay else "live" if a.director == "rules" else "off"
    return [
        *(f"run.{k}={v}" for k, v in play.PLAY_CADENCE.items()),
        "game.observe=true",
        f"run.director={mode}",
        *(["director.llm_every=0"] if mode == "live" else []),
    ]


def _check_play_tick(cfg: Config) -> None:
    try:
        play.check_tick(play.world_tick(cfg))
    except ValueError as e:
        raise ConfigError(str(e)) from None


def _lock(directory: Path) -> PlayLock:
    try:
        return PlayLock(directory).acquire()
    except PlayLockError as e:
        raise ConfigError(str(e)) from None


def _cmd_play(a: argparse.Namespace) -> int:
    """``alife play`` (DESIGN §33.2.8): ``protocol_output`` is the first act, so nothing that is printed while
    a world is prepared (the announcement, the spin-up progress) can reach the protocol stream."""
    if sum(x is not None for x in (a.run, a.init_from, a.preset)) != 1:
        raise ConfigError("play needs exactly one of RUN, --init-from SRC and --preset P")
    if a.run is None:
        for flag, given in (
            ("--checkpoint", a.checkpoint is not None),
            ("--truncate", a.truncate),
            ("--adopt", a.adopt),
            ("--replay", a.replay),
        ):
            if given:
                raise ConfigError(f"{flag} applies to a world directory (play RUN)")
        if a.houses is not None and a.init_from is None:
            raise ConfigError("--houses applies to a planet (--init-from): a preset starts at step 0")
        fixed = sorted(
            k
            for k in (item.partition("=")[0] for item in a.set)
            if k.removeprefix("run.") in play.PLAY_CADENCE
        )
        if fixed:
            raise ConfigError(
                f"play fixes the cadence of a world ({', '.join(f'run.{k}' for k in play.PLAY_CADENCE)}); "
                f"--set {', '.join(fixed)} would be overridden: remove it"
            )
    elif a.set or a.overlay or a.seed is not None or a.out or a.weather_spinup is not None or a.houses:
        raise ConfigError(
            "--set, --overlay, --seed, --out, --weather-spinup and --houses make a new world (--init-from "
            "or --preset); a world directory is played as it is"
        )
    if a.replay and a.director == "rules":
        raise ConfigError("--replay applies the director's log; it cannot also decide (--director rules)")
    control = stream.NoControl() if a.no_control else stream.FdControl()
    with stream.protocol_output() as sink:
        try:
            return _play_world(a, sink, control) if a.run is None else _play_resume(a, sink, control)
        finally:
            control.close()


def _houses_source(a: argparse.Namespace, mode: str) -> tuple[RunDir | None, int | None]:
    """The source run and checkpoint step whose exact houses a planet gets, or ``(None, None)`` for clade
    houses (``--houses clade``, or ``auto`` with a source that cannot give them: a note on stderr).
    ``--houses exact`` refuses such a source (``HouseError``, exit 4). A directory listing only (DESIGN
    §33.3.5): before the source's state is loaded and before anything is written."""
    if mode == "clade":
        return None, None
    _, _, run, step = _fork_lookup(a.init_from)
    why = (
        "the source is a state file, not a run directory (it has no event chunks)"
        if run is None or step is None
        else house_rebuild.source_problem(run, step)
    )
    if why is None:
        return run, step
    if mode == "exact":
        raise house_rebuild.HouseError(why)
    print(f"houses: {why}; the houses are the clades (approximate)", file=sys.stderr, flush=True)
    return None, None


def _rebuild_houses(mode: str, run: RunDir, step: int, cfg: Config) -> housefile.HouseSeed | None:
    """The exact houses of ``run``'s checkpoint at ``step`` under the planet's ``house_rate`` and seed. A
    rebuild that fails is an error under ``--houses exact`` and a note, with clade houses, under ``auto``."""
    print(f"houses: rebuilding the exact houses of {run.path} at step {step}", file=sys.stderr, flush=True)
    try:
        return house_rebuild.rebuild(run.path, step, rate=cfg.game.house_rate, seed=cfg.run.seed)
    except house_rebuild.HouseError as e:
        if mode == "exact":
            raise
        print(f"houses: {e}; the houses are the clades (approximate)", file=sys.stderr, flush=True)
        return None


def _play_world(a: argparse.Namespace, sink: Any, control: stream.Control) -> int:
    """A new world: a fork of a checkpoint (a planet) or a preset, locked before anything is written."""
    ra = _play_namespace(a)
    overrides = [*_run_overrides(ra), *_play_items(a)]
    lock: PlayLock | None = None
    mode = a.houses or "auto"
    src_run, src_step = _houses_source(a, mode) if a.init_from is not None else (None, None)
    seed: housefile.HouseSeed | None = None

    def before_create(out: Path, cfg: Config) -> None:
        nonlocal lock, seed
        _check_play_tick(cfg)
        RunDir.require_new(
            out
        )  # before the lock file: nothing is written into a directory create would refuse
        lock = _lock(out)
        if src_run is not None and src_step is not None:
            seed = _rebuild_houses(mode, src_run, src_step, cfg)

    try:
        # The mark of a world under construction (DESIGN §33.3.3): written with the first meta.json, so that a
        # build cancelled or failed at any later point can be told from a research run.
        built = {"play_build": "planet" if a.init_from is not None else "preset"}
        if a.init_from is not None:
            rd, cfg, box, code = _prepare_fork(
                ra,
                overrides,
                before_create=before_create,
                require_spinup=True,
                extra_meta=built,
                link_origin=True,
            )
        else:
            rd, cfg, box, code = _prepare_run(ra, overrides, before_create=before_create, extra_meta=built)
        if code is not None:
            return code
        assert rd is not None
        step = int(to_numpy(box[0].step))
        if (origin := play.origin_row(box[0], cfg)) is not None:  # the row Chapter 1 is cut from (§34.2)
            rd.update_meta(origin_row=origin)
        if seed is not None:  # before the first checkpoint: the observer opens on it (Observer.open)
            housefile.write(rd, step, seed, rate=cfg.game.house_rate, run_seed=cfg.run.seed)
        planet = (
            {
                "source": rd.meta().get("parent"),
                "step": step,
                "weather_spinup": a.weather_spinup,
                "houses": "exact" if seed is not None else "clade",
                "made_at_step": step,
            }
            if a.init_from is not None
            else None
        )
        return _play_serve(
            a, rd, cfg, box, fresh=True, planet=planet, saved=None, control=control, sink=sink,
            source="run" if a.init_from is not None else "preset",
        )  # fmt: skip
    finally:
        if lock is not None:
            lock.release()


def _cut_listing(rd: RunDir, step: int) -> str:
    """What playing the checkpoint at ``step`` would delete: the later checkpoints, metrics rows and event
    chunks."""
    ckpts = [st for st, _ in rd.checkpoints() if st > step]
    rows = [r["step"] for r in rd.metrics_rows() if r["step"] > step]
    chunks = sorted(
        {(s0, s1) for kind in ("births", "deaths") for s0, s1, _ in rd.event_files(kind) if s1 > step}
    )
    return (
        f"checkpoints {ckpts}, {len(rows)} metrics rows (steps {rows[0]} to {rows[-1]})"
        if rows
        else f"checkpoints {ckpts}"
    ) + f" and {len(chunks)} event chunk(s) {chunks}"


def _save_mode(a: argparse.Namespace, saved: dict[str, Any] | None) -> str:
    """The save mode of this session (DESIGN 34.10): ``--saves``, else play.json's ``saves``, else ``all``."""
    mode = getattr(a, "saves", None) or (saved or {}).get("saves") or "all"
    if mode not in play.SAVE_MODES:
        raise ConfigError(f"play.json's saves is one of {play.SAVE_MODES}, not {mode!r}")
    return str(mode)


def _keep_checkpoints(rd: RunDir, saved: dict[str, Any] | None, play_keep: int, saves: str = "all") -> int:
    """``run.keep_checkpoints`` of a world directory played from here on: never below what the world already
    holds, so that playing it prunes nothing it did not prune before. A play world has ``play_keep``
    already; a research run adopted with ``--adopt`` keeps its own value, and room for the checkpoint the
    first save adds (the newest ``keep`` survive, so a lower value would delete the older ones). The note
    says so only in the ``all`` save mode: in ``newest`` the first save deletes them (``_newest_note``)."""
    own = rd.config().run.keep_checkpoints
    held = len(rd.checkpoints()) + 1 if saved is None else 0
    keep = max(own, play_keep, held)
    if saved is None and keep != play_keep and saves == "all":
        print(
            f"play: --adopt keeps run.keep_checkpoints at {keep} (the run's own {own}, its "
            f"{held - 1} checkpoints and the next one): nothing is pruned at the first save",
            file=sys.stderr,
            flush=True,
        )
    return keep


def _newest_note(rd: RunDir, saved: dict[str, Any] | None, saves: str, step: int) -> None:
    """When ``--saves newest`` or ``--saves single`` turns this session to that save mode (DESIGN 34.10,
    34.11), say on stderr, before anything is touched, which checkpoints its first save deletes. ``newest``
    deletes every one up to ``step`` but step 0 and the origin (the later ones are what ``--truncate``
    deletes); ``single`` deletes every one up to ``step``, the origin and step 0 included. Those of a research
    run played in place (``--adopt``) are the research's: the flag is the explicit way to delete them (a
    control line is refused)."""
    if saves not in ("newest", "single") or (saved or {}).get("saves") == saves:
        return
    origin = rd._start_step()
    keep_origin = saves == "newest"
    doomed = [st for st, _ in rd.checkpoints() if st <= step and not (keep_origin and st in (0, origin))]
    if not doomed:
        return
    whose = "of the research run " if saved is None or rd.adopted else ""
    steps = f"step {doomed[0]}" if len(doomed) == 1 else f"steps {doomed[0]} to {doomed[-1]}"
    what = "the newest save and the origin" if keep_origin else "only the newest save (the origin goes too)"
    print(
        f"play: --saves {saves} keeps {what}: {len(doomed)} checkpoint(s) {whose}({steps}) are DELETED "
        "at the "
        "first save",
        file=sys.stderr,
        flush=True,
    )


def _play_resume(a: argparse.Namespace, sink: Any, control: stream.Control) -> int:
    """A world directory: ``play.json`` (or ``--adopt``), the chosen checkpoint (``--truncate`` if older than
    the newest), the logs cut to it as ``resume`` does, played under the pacer."""
    rd = RunDir.open(Path(a.run))
    with _lock(rd.path):
        try:
            saved = rd.read_play_json()
        except (ValueError, json.JSONDecodeError) as e:
            raise ConfigError(str(e)) from None
        if saved is None and not a.adopt:
            raise ConfigError(
                f"{rd.path} has no play.json: it is not a play world. Research runs are forked "
                "(`alife play --init-from RUN:STEP ...`); to play this directory in place pass --adopt "
                "(config.json is kept as config.pre_play.json)"
            )
        saves = _save_mode(a, saved)  # refuse a bad play.json before anything is touched
        items = _play_items(a)
        keep = _keep_checkpoints(rd, saved, play.PLAY_CADENCE["keep_checkpoints"], saves)
        items = [
            *(i for i in items if not i.startswith("run.keep_checkpoints=")),
            f"run.keep_checkpoints={keep}",
        ]
        bad = [item for item in items if not _resume_allowed(item)]
        if bad:
            raise ConfigError(f"play cannot apply {bad} to a run on resume")
        base = rd.config()
        if (a.replay or a.director == "rules") and not base.director.enabled:
            raise ConfigError(
                f"{rd.path} was started with director.enabled=false, so its compiled step cannot take "
                "director actions; fork it with `alife play --init-from RUN --director rules` instead"
            )
        if a.director == "rules" and saved is not None and saved.get("cadence") != dict(play.PLAY_CADENCE):
            raise ConfigError(
                f"{rd.path} was played at the cadence {saved.get('cadence')}, not "
                f"{dict(play.PLAY_CADENCE)}: the rule tier decides from rows written at that cadence, so "
                "another cadence is another trajectory of decisions"
            )
        cfg = apply_overrides(base, items)
        _check_play_tick(cfg)
        _check_yoke(cfg)
        _knn_guard(cfg)
        newest = max((st for st, _ in rd.checkpoints()), default=None)
        path = _select_checkpoint(rd, a.checkpoint)
        step = int(checkpoint_metadata(path)["step"])
        if newest is not None and step < newest and not a.truncate:
            raise ConfigError(
                f"--checkpoint {step} is older than the newest ({newest}): playing it would delete "
                f"{_cut_listing(rd, step)}; pass --truncate to do that"
            )
        state = load_checkpoint(path, cfg)
        if a.until is not None and a.until < step:
            raise ConfigError(f"--until {a.until} is before the checkpoint step {step}")
        replay = cfg.run.director == "replay"
        if replay:  # the whole log is checked before anything is written (exit 2)
            director_log.validate_log(director_log.read_log(rd.path / DIRECTOR_LOG_FILE), cfg)
        elif cfg.run.director == "live":
            director_log.rules_version_of(rd.read_jsonl(DIRECTOR_LOG_FILE))
        # Everything is validated; only now touch the directory.
        _newest_note(rd, saved, saves, step)
        if saved is None:
            backup = rd.path / PRE_PLAY_CONFIG_FILE
            if not backup.exists():  # never overwritten: the first adoption's config is the one worth keeping
                shutil.copyfile(rd.path / "config.json", backup)
        rd.save_config(cfg)
        rd.truncate_after(step, keep_director=replay)
        print(f"play {rd.path}: step {step}", file=sys.stderr, flush=True)
        box = [state]
        del state  # the runner must hold the only reference to the state (donation rule, F1)
        planet = (saved or {}).get("planet")
        return _play_serve(
            a, rd, cfg, box, fresh=False, planet=planet, saved=saved, control=control, sink=sink, source="run"
        )


def _play_serve(
    a: argparse.Namespace,
    rd: RunDir,
    cfg: Config,
    box: list[State],
    *,
    fresh: bool,
    planet: dict[str, Any] | None,
    saved: dict[str, Any] | None,
    control: stream.Control,
    sink: Any,
    source: str,
) -> int:
    """Run the world in ``box`` under the pacer. ``play.json`` is written in ``on_ready``: after the first
    checkpoint exists and the cloud-depth scale is known, before the header (so a directory without one is
    unfinished)."""
    step = int(to_numpy(box[0].step))
    if a.until is not None and a.until < step:
        raise ConfigError(f"--until {a.until} is before the start step {step}")
    saved = saved or {}
    saves = _save_mode(a, saved)
    base_rate = a.base_rate or saved.get("base_rate") or play.BASE_RATE
    rate = 1.0 if a.rate is None or a.rate == 0 else a.rate
    paused = a.rate is None or a.rate == 0 or a.paused

    def ready(info: dict[str, Any]) -> None:
        dc_ref = info["dc_ref"]
        try:
            rd.write_play_json(
                {
                    "base_rate": base_rate,
                    "cadence": dict(play.PLAY_CADENCE),
                    "tick": info["tick"],
                    "weather": None if dc_ref is None else {"dc_ref": dc_ref},
                    "planet": planet,
                    **({"saves": saves} if saves != "all" else {}),  # absent: all (DESIGN 34.10)
                }
            )
        except OSError as e:  # not a broken pipe: say what failed
            raise RuntimeError(f"cannot write play.json: {e}") from e

    options = play.PlayOptions(
        base_rate=base_rate,
        rate=rate,
        paused=paused,
        frame_hz=a.frame_hz,
        source=source,
        run=str(rd.path),
        planet={"source": planet["source"], "houses": planet["houses"]} if planet else None,
        origin_step=play.planet_origin_step(planet),
        dc_ref=((saved.get("weather") or {}).get("dc_ref")),
        fresh=fresh,
        until=a.until,
        on_ready=ready,
        alerts_experimental=getattr(a, "alerts", None) == "experimental",
        saves=saves,
    )
    rd.update_meta(knn_impl=spatial.knn_impl(cfg))
    if cfg.run.director == "off" and cfg.director.enabled:
        # The default launch: a world that can take director actions (a planet can) is played in the player
        # mode, which decides nothing and applies the player's powers. With none applied it is digest-equal to
        # `off` (inert env, DESIGN §26 D1). `--director rules` is `live` (and takes powers too).
        cfg = dataclasses.replace(cfg, run=dataclasses.replace(cfg.run, director=PLAYER_MODE))
        rd.save_config(cfg)
    director = make_director(rd, cfg, step, game_specs=True)  # the log of a played world may hold powers
    steps = 2**40 if a.until is None else a.until - step
    return play.serve_play(
        rd, cfg, box, steps=steps, options=options, control=control, out=sink, director=director
    )


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
        rd.path,
        out,
        st,
        reuse=a.reuse,
        plots=plots,
        start=a.start,
        log=lambda m: None if a.quiet else print(m),
    )
    if not a.quiet:
        print(f"wrote {out}")
    return ExitCode.OK


def _bench_play_world(a: argparse.Namespace, workdir: Path) -> tuple[Config, Path]:
    """The world ``bench --play`` plays, saved as a state file in ``workdir``: a preset (a founder population)
    or a planet (``--play-from SRC[:STEP]``, forked as ``alife play --init-from`` forks it), with the weather
    spun up by ``--play-spinup N``. Every arm of the bench loads its own copy."""
    items = argparse.Namespace(replay=False, director="off")
    template = workdir / "template.safetensors"
    if a.play_resume is not None:
        if a.play_from is not None:
            raise ConfigError("--play-resume and --play-from are two ways to name the world: give one")
        # a played world's checkpoint as it is (a planet some chapters in); the arms resume it
        src, _ = _fork_source(a.play_resume)
        cfg = _layer(
            _embedded_config(src), overlays=a.overlay, path=None, overrides=[*a.set, *_play_items(items)]
        )
        _knn_guard(cfg)
        return cfg, src
    if a.play_from is not None:
        ns = argparse.Namespace(
            init_from=a.play_from, overlay=a.overlay, set=a.set, weather_spinup=a.play_spinup,
            out=str(workdir / "template"), overwrite=True, quiet=True, director="off", preset=None,
        )  # fmt: skip
        ra = _play_namespace(ns)
        rd, cfg, box, code = _prepare_fork(
            ra, [*_run_overrides(ra), *_play_items(items)], require_spinup=True
        )
        if code is not None or rd is None:
            raise CheckpointError(f"bench --play: the planet could not be made (exit {code})")
        write_state_file(template, box[0], cfg)
        return cfg, template
    cfg = load_config(preset=a.preset, overlays=a.overlay, overrides=[*a.set, *_play_items(items)])
    _knn_guard(cfg)
    state = init_state(cfg)
    if a.play_spinup:
        if not cfg.weather.enabled:
            raise ConfigError("--play-spinup spins the weather up: it needs the weather on (--overlay storm)")
        wx = weather_spinup.spinup(cfg, a.play_spinup, step=0, key=state.key, compile=cfg.run.compile)
        state = state._replace(wx=wx)
    write_state_file(template, state, cfg)
    return cfg, template


def _bench_play(a: argparse.Namespace) -> dict[str, Any]:
    """``bench --play``: the play loop of ``alife play`` run headless (DESIGN §33.5), in its own report."""
    import tempfile

    from alife import bench

    workdir = Path(a.play_out) if a.play_out else Path(tempfile.mkdtemp(prefix="alife-bench-play-"))
    workdir.mkdir(parents=True, exist_ok=True)
    cfg, template = _bench_play_world(a, workdir)
    rates = tuple(r.strip() for r in a.play_rates.split(",") if r.strip())
    wanted = a.play_directors or (
        "off,rules,player" if cfg.director.enabled else "off"
    )  # the rule tier and the player need the levers
    directors = tuple(d.strip() for d in wanted.split(",") if d.strip())
    bad = [d for d in directors if d not in bench.PLAY_DIRECTORS]
    if bad:
        raise ConfigError(f"--play-directors takes {bench.PLAY_DIRECTORS}, not {bad}")
    try:
        for r in rates:
            bench._rate_value(r)
    except ValueError as e:
        raise ConfigError(str(e)) from None
    report = bench._environment(cfg)
    report["play"] = bench.run_play_bench(
        cfg, template, workdir, steps=a.steps or bench.PLAY_STEPS, warmup=a.play_warmup, rates=rates,
        directors=directors, chapter=a.play_chapter, powers=a.play_powers,
    )  # fmt: skip
    report["play"]["workdir"] = str(workdir)
    return report


def _cmd_bench(a: argparse.Namespace) -> int:
    from alife.bench import run_bench

    if a.play:
        report = _bench_play(a)
        text = _dumps(report)
        if a.json is not None:
            out = Path(a.json)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text + "\n", encoding="utf-8")
        print(text)
        if report["play"]["digests_equal"] is False:
            print("play bench: a played world's final state differs from the plain run's", file=sys.stderr)
            return int(ExitCode.ERROR)
        return int(ExitCode.OK)
    cfg = load_config(preset=a.preset, overlays=a.overlay, overrides=a.set)
    _knn_guard(cfg)
    report = run_bench(
        cfg,
        steps=a.steps or 200,
        warmup=a.warmup,
        stages=a.stages,
        determinism=a.determinism,
        brain_variants=a.brain_variants,
        observe=a.observe,
        weather=a.weather,
        weather_pairs=a.weather_pairs,
        weather_org=a.weather_org,
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
    obs = report.get("observe")
    if isinstance(obs, Mapping) and obs.get("trajectory_match") is False:
        print("observer purity check FAILED: observing changed the trajectory", file=sys.stderr)
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


def _config_args(
    p: argparse.ArgumentParser, *, overlays: bool = True, preset_help: str | None = None
) -> None:
    p.add_argument("--preset", choices=PRESETS, default=None, help=preset_help or f"default {DEFAULT_PRESET}")
    if overlays:
        p.add_argument("--overlay", action="append", default=[], choices=ALL_OVERLAYS)
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
    p.add_argument(
        "--weather-spinup",
        type=_non_negative,
        default=None,
        metavar="N",
        help="advance the weather alone for N steps before the agents feel it (needs weather.enabled; "
        "storms mature in about 15000 steps; a fork's source must not have the weather on)",
    )
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=_cmd_run)

    p = sub.add_parser("resume", help="continue a run from a checkpoint (bitwise identical)")
    p.add_argument("run", metavar="RUN")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--steps", type=_non_negative, default=None)
    g.add_argument("--until", type=_non_negative, default=None, metavar="STEP")
    p.add_argument("--checkpoint", type=_checkpoint_arg, default=None, metavar="STEP|latest")
    p.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="run.KEY=VALUE|director.KEY=VALUE|game.KEY=VALUE",
    )
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
    _config_args(
        p,
        overlays=False,
        preset_help="simulate from this preset's founders (no default: needs RUN or FILE otherwise)",
    )
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
        help="no stepping: stream the latest checkpoint of RUN, then at each poll the newest one if it is "
        "newer (checkpoints written between two polls are skipped)",
    )
    p.add_argument(
        "--poll",
        type=_positive_seconds,
        default=1.0,
        metavar="SECONDS",
        help="--follow: checkpoint poll period (also how often a closed reader is noticed)",
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

    p = sub.add_parser("play", help=_PLAY_HELP, description=_PLAY_HELP)
    p.add_argument(
        "run", nargs="?", default=None, metavar="RUN", help="a play world (a directory with play.json)"
    )
    p.add_argument("--checkpoint", type=_checkpoint_arg, default=None, metavar="STEP|latest")
    p.add_argument(
        "--truncate",
        action="store_true",
        help="allow --checkpoint older than the newest: later checkpoints, rows and chunks are deleted",
    )
    p.add_argument(
        "--adopt",
        action="store_true",
        help="play a directory without play.json in place (config.json is kept as config.pre_play.json)",
    )
    p.add_argument(
        "--replay", action="store_true", help="apply the world's director.jsonl instead of deciding"
    )
    p.add_argument("--until", type=_non_negative, default=None, metavar="STEP", help="stop at this step")
    p.add_argument(
        "--init-from",
        default=None,
        metavar="SRC[:STEP]",
        help="make a new world from a checkpoint (a planet), as `run --init-from`, saved in --out",
    )
    p.add_argument("--preset", choices=PRESETS, default=None, help="make a new world from a preset (tests)")
    p.add_argument("--overlay", action="append", default=[], choices=ALL_OVERLAYS)
    p.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE")
    p.add_argument("--seed", type=_non_negative, default=None)
    p.add_argument("--out", default=None, metavar="DIR")
    p.add_argument(
        "--weather-spinup",
        type=_non_negative,
        default=None,
        metavar="N",
        help="advance the weather alone for N steps (needed, with N >= 1, when a fork turns the weather on)",
    )
    p.add_argument(
        "--rate",
        type=_rate_arg,
        default=None,
        metavar="0|0.25|0.5|1|2|max",
        help="start running at this multiple of the base rate (default: start paused)",
    )
    p.add_argument("--paused", action="store_true", help="start paused even with --rate")
    p.add_argument(
        "--base-rate", type=_positive_seconds, default=None, metavar="SPS", help="steps/s at 1x (14.4)"
    )
    p.add_argument(
        "--frame-hz", type=_positive, default=play.FRAME_HZ, metavar="HZ", help="frames per second"
    )
    p.add_argument(
        "--director",
        choices=("off", "rules"),
        default="off",
        help="off (default): no rule tier; a world with director.enabled applies the player's powers "
        "(DESIGN 34.5). rules: the director's rule tier decides (the world needs director.enabled; no LLM)",
    )
    p.add_argument(
        "--houses",
        choices=("exact", "clade", "auto"),
        default=None,
        help="houses of a planet (--init-from): exact rebuilds them from the source's event chunks (exit 4 "
        "if they are missing), clade takes the clades, auto (default) tries exact and falls back with a note",
    )
    p.add_argument(
        "--alerts",
        choices=("experimental",),
        default=None,
        help="also evaluate the alert classes whose threshold no measurement backs (storm over a lineage; "
        "they are logged to alerts.jsonl as the log surface, DESIGN 34.3)",
    )
    p.add_argument(
        "--saves",
        choices=play.SAVE_MODES,
        default=None,
        help="all (default): keep the last 3 saves and the origin; newest: only the newest save and the "
        "origin; single: only the newest save (kept in play.json; the `saves` control line changes it while "
        "playing, DESIGN 34.10, 34.11)",
    )
    p.add_argument("--no-control", action="store_true", help="do not read stdin")
    p.set_defaults(func=_cmd_play)

    p = sub.add_parser("bench", help="throughput, donation health, determinism")
    _config_args(p)
    p.add_argument(
        "--steps", type=_positive, default=None, help="timed steps (default 200; 2000 with --play)"
    )
    p.add_argument("--warmup", type=_non_negative, default=20)
    p.add_argument("--stages", action="store_true")
    p.add_argument("--determinism", action="store_true")
    p.add_argument("--brain-variants", action="store_true")
    p.add_argument(
        "--observe",
        action="store_true",
        help="also time the step with the game observers (social snapshots included) against the plain step "
        "(interleaved blocks), and the parts of one snapshot",
    )
    p.add_argument(
        "--weather",
        action="store_true",
        help="also time the step with the weather layer on against the same config with it off "
        "(interleaved blocks; needs weather.enabled, for example --overlay storm)",
    )
    p.add_argument(
        "--weather-org",
        action="store_true",
        help="also time the step with the organism couplings (weather.seed_gain, weather.shade_gain) on "
        "against the same config with them 0 (interleaved blocks; needs weather.enabled and one of them "
        "above 0)",
    )
    p.add_argument(
        "--weather-pairs",
        type=_positive,
        default=2,
        metavar="N",
        help="off/on block pairs of the --weather and --weather-org measurements (default 2: off on on off)",
    )
    p.add_argument(
        "--play",
        action="store_true",
        help="the play loop of `alife play` run headless (DESIGN §33.5): per director mode and rate, the "
        "achieved rate, step_ms p50/p99/max, the cost of its parts, late steps, bytes by record and the "
        "digest against a plain run; a report of its own (the other bench parts are skipped). Use the "
        "planet's config: --preset 64k3d --overlay storm --set director.enabled=true",
    )
    p.add_argument(
        "--play-from",
        default=None,
        metavar="SRC[:STEP]",
        help="--play on a planet: fork this run's checkpoint as `alife play --init-from` does (an evolved "
        "population, not founders)",
    )
    p.add_argument(
        "--play-resume",
        default=None,
        metavar="SRC[:STEP]",
        help="--play on a played world: resume this run's checkpoint as it is (the gate of DESIGN §34.8 "
        "resumes a planet some steps before a chapter's end, so that the arm holds the end, its report and "
        "the scripted powers)",
    )
    p.add_argument(
        "--play-powers",
        action="store_true",
        help="--play: the player arm sends two scripted powers (a Food in the middle of the measured part "
        "and a Strike at the chapter-end step itself) and reports the click-to-applied latency; its digest "
        "is then checked against the replay of its own log, not the plain run's (needs the player mode)",
    )
    p.add_argument(
        "--play-chapter",
        type=_positive,
        default=None,
        metavar="N",
        help="--play: the chapter length in steps (default 6,000; a multiple of the checkpoint cadence; "
        "the tests shorten it)",
    )
    p.add_argument(
        "--play-spinup",
        type=_positive,
        default=None,
        metavar="N",
        help="--play: spin the weather up for N steps first (a storm world has no storms for about 15,000 "
        "steps)",
    )
    p.add_argument(
        "--play-rates", default="max,1", metavar="LIST", help="--play: rates, comma separated (default max,1)"
    )
    p.add_argument(
        "--play-directors",
        default=None,
        metavar="LIST",
        help="--play: director modes (off, rules, player; all three when director.enabled is set, else off)",
    )
    p.add_argument(
        "--play-warmup",
        type=_non_negative,
        default=300,
        metavar="N",
        help="--play: warm-up steps (default 300)",
    )
    p.add_argument(
        "--play-out",
        default=None,
        metavar="DIR",
        help="--play: where the arms' run directories go (default a temp dir)",
    )
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
    p.add_argument(
        "--start",
        type=_non_negative,
        default=None,
        metavar="STEP",
        help="analyse from STEP: the agents alive then are the founders (default: the run's first step, "
        "the fork step for a fork)",
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
