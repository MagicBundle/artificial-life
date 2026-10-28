"""The layout migration tool (DESIGN §43.8): a world moved onto a new feature layout, as a NEW run.

``migrate_checkpoint(src, out, enable=..., sets=...)`` loads a checkpoint (plain or compressed) with the
config embedded in it, builds the target config (``ExtSpec.enable`` hooks, then ``--set`` of feature keys
only), moves the state with ``state.migrate_layout`` (zero rows and columns for every new brain block, new
fields from their ``rest`` value) and writes ``out`` with the layout keys of the new config and the
provenance in ``meta.json`` (``migrated_from``). The source is never modified. The save is written in the
form of the source's (a compressed source gives a compressed output). The numbers of the world carry over
(step, energy, ledger, keys): the migrated population behaves as before up to floating-point reduction
order until evolution uses the new channels (the extra zero terms change the order in which the GEMV
accumulates: the CPU kernel sums in groups of 4, so a pad that is a multiple of 4 is exact and others are not;
Metal differs anyway; about 1% of the agents of a float16 world round a hidden state differently at the first
step), and a birth draws its mutation noise with the new weight shapes (the trajectory parts from the
unmigrated one at the first birth). Both are documented in §43.8.

The output is atomic: it is built in a temporary directory beside ``out`` and renamed at the end; ``out`` may
not be, contain or lie inside the source, and a run that is being played is never replaced.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import json
import os
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import mlx.core as mx

from alife import config as config_mod
from alife import ext
from alife.config import Config, config_from_dict, config_hash, config_to_dict
from alife.invariants import full_check
from alife.runio import (
    CheckpointError,
    RunDir,
    checkpoint_metadata,
    is_compressed,
    load_checkpoint,
    play_lock_holder,
)
from alife.state import (
    LayoutChange,
    MigrationError,
    layout_change,
    migrate_state,
    to_numpy,
)

LEAVE_ONE_OUT_NOTE = (
    "Leave-one-out arms are FRESH runs from the shared evolved source, never removals from an evolved "
    "world (edge case E23): to compare a world with and without a feature, migrate the same source with the "
    "other features and leave the feature out. Removal exists only with --force-removal and drops the "
    "evolved weights of the feature."
)


class MigrationReport(NamedTuple):
    """What a migration did: the run directory written, the step, the feature change, the layout strings."""

    out: Path
    step: int
    change: LayoutChange
    layout_features_old: str
    layout_features_new: str
    genome_layout_version_old: int
    genome_layout_version_new: int
    config_hash_old: str
    config_hash_new: str
    compressed: bool


def embedded_config(path: Path) -> Config:
    """The config stored in a state file's metadata, validated (what a fork starts from)."""
    meta = checkpoint_metadata(path)
    try:
        cfg = config_from_dict(json.loads(meta["config"]))
    except (KeyError, json.JSONDecodeError) as e:
        raise CheckpointError(f"{path}: no readable embedded config ({e})") from None
    config_mod.validate(cfg)
    return cfg


def _flat(cfg: Config) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for section, body in config_to_dict(cfg).items():
        if isinstance(body, dict):
            out.update({f"{section}.{k}": v for k, v in body.items()})
        else:
            out[section] = body
    return out


def _changed(a: Config, b: Config) -> set[str]:
    fa, fb = _flat(a), _flat(b)
    return {k for k in fa.keys() | fb.keys() if fa.get(k) != fb.get(k)}


def target_config(src_cfg: Config, enable: Sequence[str] = (), sets: Sequence[str] = ()) -> Config:
    """The config of the migrated world: ``src_cfg`` with each ``--enable`` feature switched on by its
    ``ExtSpec.enable`` hook, then the ``--set`` items, which may only name a key a feature adds
    (``ExtSpec.lever_keys``) or one the hooks changed: a migration changes features, nothing else (fork the
    migrated run for any other change). The director is off, as in a fork."""
    by = {sp.name: sp for sp in ext.REGISTRY}
    cfg = dataclasses.replace(src_cfg, run=dataclasses.replace(src_cfg.run, director="off"))
    for name in enable:
        if name not in by:
            raise MigrationError(f"--enable {name}: no such feature (registered: {sorted(by) or 'none'})")
        hook = by[name].enable
        if hook is None:
            raise MigrationError(f"--enable {name}: the feature has no switch of its own; use --set")
        cfg = hook(cfg)
    allowed = _changed(src_cfg, cfg) | {f"{s}.{k}" for keys in ext.lever_keys().values() for s, k in keys}
    for item in sets:
        key = item.partition("=")[0].strip()
        if key not in allowed:
            raise MigrationError(
                f"--set {key}: a migration changes feature keys only (lever keys of a registered feature); "
                "fork the migrated run for any other change"
            )
    cfg = config_mod.apply_overrides(cfg, list(sets))
    for name in enable:
        if not by[name].enabled(cfg):
            raise MigrationError(f"--enable {name}: the feature is not enabled in the resulting config")
    return cfg


def _source_root(src: Path) -> Path:
    """The tree that holds the source: its run directory for a save of a run, else the file itself."""
    src = src.resolve()
    if src.parent.name == "checkpoints" and (src.parent.parent / "meta.json").is_file():
        return src.parent.parent
    return src


def _check_output(src: Path, out: Path, overwrite: bool) -> None:
    """Refuse an ``out`` that could destroy or pollute the source, or a run that is being played, before
    anything is written (``MigrationError``; ``FileExistsError`` for an existing ``out`` without
    ``overwrite``)."""
    if out.is_symlink():
        raise MigrationError(f"--out {out} is a symbolic link: give the real directory")
    root, out_r = _source_root(src), out.resolve()
    if out_r == root or root.is_relative_to(out_r) or out_r.is_relative_to(root):
        raise MigrationError(
            f"--out {out} is, contains or lies inside the source ({root}): a migration writes a new, "
            "separate run directory and never touches its source"
        )
    if not out.exists():
        return
    if not overwrite:
        RunDir.require_new(out)
        return
    if not out.is_dir():
        raise FileExistsError(f"{out} exists and is not a directory")
    pid = play_lock_holder(out)
    if pid is not None:
        raise MigrationError(f"{out} is being played by process {pid} (play.lock): stop that play first")
    if any(p.name != "play.lock" for p in out.iterdir()) and not (out / "meta.json").is_file():
        raise FileExistsError(f"{out} is not a run directory; refusing to overwrite it")


def migrate_checkpoint(
    src: Path,
    out: Path,
    *,
    enable: Sequence[str] = (),
    sets: Sequence[str] = (),
    force_removal: bool = False,
    overwrite: bool = False,
    origin: str | None = None,
    cfg_new: Config | None = None,
    extra_provenance: Mapping[str, Any] | None = None,
) -> MigrationReport:
    """Migrate the state file ``src`` to a new run directory ``out`` (see the module docstring). Everything is
    checked before ``out`` is created; ``MigrationError`` (a refusal), ``CheckpointError`` (an unreadable or
    mismatched source), ``FileExistsError`` (``out`` exists)."""
    src, out = Path(src), Path(out)
    _check_output(src, out, overwrite)
    cfg_old = embedded_config(src)
    cfg = cfg_new if cfg_new is not None else target_config(cfg_old, enable, sets)
    change = layout_change(cfg_old, cfg, allow_removal=force_removal)
    if not change.added and not change.removed:
        raise MigrationError(
            "nothing to migrate: the target config adds and removes no feature "
            f"(already on: {', '.join(change.kept) or 'none'})"
        )
    if force_removal and not change.removed:
        raise MigrationError("--force-removal was given but no feature is removed: drop the flag")
    s_old = load_checkpoint(src, cfg_old, check_hash=False)
    s_new = migrate_state(s_old, cfg_old, cfg, allow_removal=force_removal)
    mx.eval(s_new)
    problems = full_check(s_new, cfg)
    if problems:
        raise MigrationError("the migrated state violates the new config: " + "; ".join(problems))
    step = int(to_numpy(s_new.step))
    meta_old = checkpoint_metadata(src)
    prov = {
        "source": origin or str(src.resolve()),
        "source_step": step,
        "source_config_hash": meta_old.get("config_hash"),
        "source_layout_features": meta_old.get("layout_features", ""),
        "source_genome_layout_version": int(meta_old["genome_layout_version"]),
        "added": list(change.added),
        "removed": list(change.removed),
        "kept": list(change.kept),
        "forced_removal": bool(force_removal and change.removed),
        "migrated_utc": _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        **(dict(extra_provenance) if extra_provenance else {}),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.parent / f".{out.name}.migrating-{os.getpid()}"
    if tmp.exists():
        raise FileExistsError(f"{tmp} exists (a migration of this pid left it): remove it")
    old = out.parent / f".{out.name}.replaced-{os.getpid()}"
    try:
        rd = RunDir.create(tmp, cfg, parent=origin or str(src.resolve()), extra_meta={"migrated_from": prov})
        rd.update_meta(start_step=step, status="interrupted")  # not yet run: resume continues it
        rd.compress = is_compressed(src)
        rd.save_checkpoint(s_new, cfg)
        rd.wait_saves()
        meta_new = rd.meta()
        replaced = out.exists()
        if replaced:  # --overwrite of a run (checked above): moved aside, deleted only after the swap
            os.rename(out, old)
        try:
            os.rename(tmp, out)
        except OSError:
            if replaced:
                os.rename(old, out)
            raise
        if replaced:
            shutil.rmtree(old, ignore_errors=True)
    except OSError as e:
        shutil.rmtree(tmp, ignore_errors=True)
        raise MigrationError(f"cannot write {out}: {e}") from e
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return MigrationReport(
        out=out,
        step=step,
        change=change,
        layout_features_old=meta_old.get("layout_features", ""),
        layout_features_new=meta_new.get("layout_features", ""),
        genome_layout_version_old=int(meta_old["genome_layout_version"]),
        genome_layout_version_new=int(meta_new["genome_layout_version"]),
        config_hash_old=str(meta_old.get("config_hash")),
        config_hash_new=config_hash(cfg),
        compressed=rd.compress,
    )
