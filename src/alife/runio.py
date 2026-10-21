"""Run directories: config, meta, checkpoints, metrics and event logs (§18). Owned by WP5.

State files, ``LATEST``, ``meta.json``, ``config.json`` and event chunks are written atomically (``*.tmp`` +
``os.replace``), so a crash leaves either the old or the new complete file; ``metrics.jsonl`` is appended and
fsynced, and a torn last line is ignored on read.

The director's ``director.jsonl`` and ``journal.jsonl`` use the same append/fsync/torn-line rules as
``metrics.jsonl`` (the generic ``append_jsonl`` / ``read_jsonl``); ``director_archive.json`` is atomic.

Event chunks ``births_<s0>_<s1>.npz`` / ``deaths_<s0>_<s1>.npz`` hold the records of steps ``s0 ≤ step < s1``,
where a record's ``step`` is the index of the step during which it happened; a checkpoint at
``state.step == s1`` therefore contains exactly the effects of every chunk ending at ``s1``.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
import zipfile
import zlib
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Final

import mlx.core as mx
import numpy as np

from alife import layout, rng, spatial
from alife.config import Config, config_from_dict, config_hash, config_to_dict, is_3d, save_config, validate
from alife.state import State, expected_layout, state_from_arrays, state_to_arrays, to_numpy

FORMAT_VERSION: Final = 1

CONFIG_FILE: Final = "config.json"
META_FILE: Final = "meta.json"
METRICS_FILE: Final = "metrics.jsonl"
LATEST_FILE: Final = "LATEST"
DIRECTOR_LOG_FILE: Final = "director.jsonl"
JOURNAL_FILE: Final = "journal.jsonl"
DIRECTOR_ARCHIVE_FILE: Final = "director_archive.json"
OBSERVERS_DIR: Final = "observers"  # the observer sidecars of an observing run (DESIGN §28.6)
STATUSES: Final = ("running", "done", "extinct", "violation", "interrupted")

BIRTH_COLUMNS: Final[Mapping[str, Any]] = {
    "step": np.int32,
    "uid": np.int64,
    "parent_uid": np.int64,
    "clade": np.int32,
    "generation": np.int32,
    "slot": np.int32,
}
DEATH_COLUMNS: Final[Mapping[str, Any]] = {
    "step": np.int32,
    "uid": np.int64,
    "age": np.int32,
    "cause": np.uint8,
    "generation": np.int32,
    "clade": np.int32,
}

_CKPT_RE: Final = re.compile(r"^step_(\d{10})\.safetensors$")
_EVENT_RE: Final = re.compile(r"^(births|deaths)_(\d{10})_(\d{10})\.npz$")
_OBSERVER_RE: Final = re.compile(r"^step_(\d{10})\.npz$")
_MAX_HEADER: Final = 1 << 26  # sanity bound on a safetensors JSON header


class CheckpointError(Exception):
    """Incompatible or corrupt checkpoint (version, shape or config-hash mismatch)."""


# --- atomic file helpers --------------------------------------------------------------------------------


def _tmp_path(path: Path) -> Path:
    return path.with_name(path.name + ".tmp")


def _fsync_dir(path: Path) -> None:
    """Best-effort fsync of a directory so a completed rename survives a crash."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _atomic_write(path: Path, write: Any) -> None:
    """Call ``write(f)`` on ``path.tmp`` (binary), fsync it, and move it over ``path``; never leaves a tmp."""
    path = Path(path)
    tmp = _tmp_path(path)
    try:
        with open(tmp, "wb") as f:
            write(f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    _fsync_dir(path.parent)


def _atomic_write_text(path: Path, text: str) -> None:
    _atomic_write(path, lambda f: f.write(text.encode("utf-8")))


def _jsonable(x: Any) -> Any:
    """Plain JSON values: numpy scalars/arrays → Python, non-finite floats → None, tuples → lists."""
    if isinstance(x, Mapping):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, list | tuple | np.ndarray):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def _dumps(x: Any) -> str:
    return json.dumps(_jsonable(x), allow_nan=False)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """The records of a JSONL file; a torn last line (a crash mid-append) is ignored."""
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                break
            raise
    return rows


def _append_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    """Append one JSON object per line and fsync."""
    lines = [_dumps(r) for r in records]
    if not lines:
        return
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _record_step(rec: Mapping[str, Any]) -> int | None:
    """The step a log record belongs to: ``step_applied`` (director entries) or ``step`` (rows, headers)."""
    step = rec.get("step_applied", rec.get("step"))
    return step if isinstance(step, int) else None


# --- state files ----------------------------------------------------------------------------------------


# The one State schema lives in ``state`` (§2.2, §27.1); this name is kept for tests/test_runio.py.
_expected_layout = expected_layout


def _alife_version() -> str:
    import alife  # local: the package __init__ imports the step graph, which never imports runio

    return alife.__version__


def write_state_file(path: Path, s: State, cfg: Config) -> None:
    """Atomic safetensors write (``*.tmp`` + ``os.replace``) with checkpoint metadata (§18).

    A 3D state also records ``dims`` (§27.1); a file without the key is 2D.
    """
    meta = {
        "format_version": str(FORMAT_VERSION),
        "rng_layout_version": str(rng.RNG_LAYOUT_VERSION),
        "genome_layout_version": str(layout.genome_layout_version(cfg.world.dims)),
        **({"dims": str(cfg.world.dims)} if is_3d(cfg) else {}),
        "step": str(int(to_numpy(s.step))),
        "config_hash": config_hash(cfg),
        "alife_version": _alife_version(),
        "mlx_version": mx.__version__,
        "weight_dtype": cfg.brain.weight_dtype,
        "config": json.dumps(config_to_dict(cfg), sort_keys=True, indent=2),
    }
    arrays = state_to_arrays(s)
    _atomic_write(Path(path), lambda f: mx.save_safetensors(f, arrays, meta))


def checkpoint_metadata(path: Path) -> dict[str, str]:
    """The string metadata of a state file, read from the safetensors header only."""
    try:
        with open(path, "rb") as f:
            raw = f.read(8)
            if len(raw) != 8:
                raise CheckpointError(f"{path}: truncated safetensors header")
            (size,) = struct.unpack("<Q", raw)
            if size > _MAX_HEADER:
                raise CheckpointError(f"{path}: implausible safetensors header size {size}")
            body = f.read(size)
        header = json.loads(body)
    except OSError as e:
        raise CheckpointError(f"{path}: cannot read state file: {e}") from e
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise CheckpointError(f"{path}: corrupt safetensors header: {e}") from e
    meta = header.get("__metadata__", {}) if isinstance(header, dict) else None
    if not isinstance(meta, dict):
        raise CheckpointError(f"{path}: corrupt safetensors metadata")
    return {str(k): str(v) for k, v in meta.items()}


def load_checkpoint(path: Path, cfg: Config, *, check_hash: bool = True) -> State:
    """Load a state file; ``CheckpointError`` on a version/shape mismatch, or a config-hash one if checked.

    Keys are matched by name (their order in the file is arbitrary, F15). The file's ``dims`` (absent = 2)
    must equal ``world.dims``: a state never crosses dimensions (§27.1). The RNG layout version must be in
    ``rng.RNG_COMPATIBLE_VERSIONS``. A pre-director file (no ``env_*`` arrays) loads with inert env values.
    The result is evaluated.
    """
    path = Path(path)
    meta = checkpoint_metadata(path)
    nd = str(cfg.world.dims)
    bad = []
    if meta.get("dims", "2") != nd:
        bad.append(
            f"dims={meta.get('dims', '2')!r} (expected {nd!r}): cross-dimension resume/fork is not supported"
        )
    want = {
        "format_version": str(FORMAT_VERSION),
        "genome_layout_version": str(layout.genome_layout_version(cfg.world.dims)),
    }
    if check_hash:
        want["config_hash"] = config_hash(cfg)
    bad += [f"{k}={meta.get(k)!r} (expected {v!r})" for k, v in want.items() if meta.get(k) != v]
    if meta.get("rng_layout_version") not in {str(v) for v in rng.RNG_COMPATIBLE_VERSIONS}:
        compatible = sorted(rng.RNG_COMPATIBLE_VERSIONS)
        bad.append(f"rng_layout_version={meta.get('rng_layout_version')!r} (expected one of {compatible})")
    if bad:
        raise CheckpointError(f"{path}: incompatible checkpoint: {', '.join(bad)}")
    try:
        arrays = mx.load(str(path), format="safetensors")
        s = state_from_arrays(arrays, cfg)
        mx.eval(s)  # loads are lazy; read now so a later replace of the file cannot affect this state
    except (ValueError, RuntimeError, OSError) as e:
        raise CheckpointError(f"{path}: {e}") from e
    spec = expected_layout(cfg)
    problems = [
        f"{name}: {tuple(a.shape)} {a.dtype} (expected {spec[name][0]} {spec[name][1]})"
        for name, a in zip(type(s)._fields, s, strict=True)
        if tuple(a.shape) != spec[name][0] or a.dtype != spec[name][1]
    ]
    if problems:
        raise CheckpointError(f"{path}: state does not match the config: {'; '.join(problems)}")
    return s


# --- run directory --------------------------------------------------------------------------------------


def _git_rev() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    rev = out.stdout.strip()
    return rev if out.returncode == 0 and rev else None


def _read_events(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def _write_npz(path: Path, cols: Mapping[str, np.ndarray]) -> None:
    _atomic_write(path, lambda f: np.savez_compressed(f, **cols))


def _event_columns(kind: str, rec: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Flatten, filter by ``valid`` (if present) and cast one record mapping to the §18 column dtypes."""
    spec = BIRTH_COLUMNS if kind == "births" else DEATH_COLUMNS
    missing = sorted(set(spec) - set(rec))
    if missing:
        raise ValueError(f"{kind} records are missing columns {missing}")
    keep = None if "valid" not in rec else np.asarray(rec["valid"], dtype=bool).reshape(-1)
    out = {}
    for name, dtype in spec.items():
        col = np.asarray(rec[name]).reshape(-1)
        if keep is not None:
            if col.shape != keep.shape:
                raise ValueError(f"{kind}.{name} has {col.size} entries, valid has {keep.size}")
            col = col[keep]
        out[name] = col.astype(dtype)
    sizes = {c.size for c in out.values()}
    if len(sizes) > 1:
        raise ValueError(f"{kind} columns differ in length: {sorted(sizes)}")
    return out


class RunDir:
    """A run directory (§18): config, meta, metrics, checkpoints, events, diag dumps and media."""

    path: Path

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    # paths
    @property
    def checkpoint_dir(self) -> Path:
        return self.path / "checkpoints"

    @property
    def events_dir(self) -> Path:
        return self.path / "events"

    @property
    def diag_dir(self) -> Path:
        return self.path / "diag"

    @property
    def observers_dir(self) -> Path:
        return self.path / OBSERVERS_DIR

    @property
    def media_dir(self) -> Path:
        return self.path / "media"

    @classmethod
    def create(cls, path: Path, cfg: Config, *, overwrite: bool = False, parent: str | None = None) -> RunDir:
        """Create a fresh run directory holding ``cfg`` and a ``running`` meta.

        An existing non-empty directory is an error unless ``overwrite`` is set and it is a run directory
        (has ``meta.json``); anything else is never deleted.
        """
        path = Path(path)
        if path.exists():
            if not path.is_dir():
                raise FileExistsError(f"{path} exists and is not a directory")
            if any(path.iterdir()):
                if not overwrite:
                    raise FileExistsError(f"{path} is not empty (use overwrite to replace a run directory)")
                if not (path / META_FILE).is_file():
                    raise FileExistsError(f"{path} is not a run directory; refusing to overwrite it")
                shutil.rmtree(path)
        rd = cls(path)
        rd.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        rd.events_dir.mkdir(exist_ok=True)
        tmp = _tmp_path(path / CONFIG_FILE)
        save_config(cfg, tmp)
        os.replace(tmp, path / CONFIG_FILE)
        meta = {
            "format_version": FORMAT_VERSION,
            "alife_version": _alife_version(),
            "mlx_version": mx.__version__,
            "python": platform.python_version(),
            "platform": f"{sys.platform}-{platform.machine()}",
            "device": str(mx.default_device()),
            "knn_impl": spatial.knn_impl(cfg),  # the neighbour search this process traces (§4)
            "git_rev": _git_rev(),
            "created_utc": _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "seed": cfg.run.seed,
            "config_hash": config_hash(cfg),
            "rng_layout_version": rng.RNG_LAYOUT_VERSION,
            "genome_layout_version": layout.genome_layout_version(cfg.world.dims),
            **({"dims": cfg.world.dims} if is_3d(cfg) else {}),  # absent = 2D (§27.1)
            "parent": parent,
            "status": "running",
        }
        rd._write_meta(meta)
        return rd

    @classmethod
    def open(cls, path: Path) -> RunDir:
        """Open an existing run directory; ``CheckpointError`` if its format version differs."""
        rd = cls(path)
        for name in (CONFIG_FILE, META_FILE):
            if not (rd.path / name).is_file():
                raise FileNotFoundError(f"{rd.path} is not a run directory (missing {name})")
        version = rd.meta().get("format_version")
        if version != FORMAT_VERSION:
            raise CheckpointError(f"{rd.path}: run format_version {version!r}, expected {FORMAT_VERSION}")
        return rd

    def config(self) -> Config:
        cfg = config_from_dict(json.loads((self.path / CONFIG_FILE).read_text(encoding="utf-8")))
        validate(cfg)
        return cfg

    def save_config(self, cfg: Config) -> None:
        """Rewrite ``config.json`` atomically (resume writes accepted ``run.*`` overrides back, §18)."""
        tmp = _tmp_path(self.path / CONFIG_FILE)
        save_config(cfg, tmp)
        os.replace(tmp, self.path / CONFIG_FILE)

    def meta(self) -> dict[str, Any]:
        return json.loads((self.path / META_FILE).read_text(encoding="utf-8"))

    def _write_meta(self, meta: Mapping[str, Any]) -> None:
        _atomic_write_text(
            self.path / META_FILE, json.dumps(_jsonable(meta), indent=2, allow_nan=False) + "\n"
        )

    def update_meta(self, **fields: Any) -> None:
        if "status" in fields and fields["status"] not in STATUSES:
            raise ValueError(f"unknown run status {fields['status']!r}; expected one of {STATUSES}")
        meta = self.meta()
        meta.update(fields)
        self._write_meta(meta)

    # checkpoints
    def save_checkpoint(
        self, s: State, cfg: Config, *, observers: Mapping[str, np.ndarray] | None = None
    ) -> Path:
        """sidecar + write_state_file + LATEST + prune.

        ``observers`` (the arrays of ``Observer.sidecar()``) is written to ``observers/step_<t>.npz`` before
        the state file, so every complete checkpoint of an observing run has its sidecar (a crash in between
        leaves an orphan sidecar, which pruning removes). Without ``observers`` a sidecar of this step is
        removed, so a checkpoint and its sidecar always come from the same save (§28.6).
        """
        step = int(to_numpy(s.step))
        path = self.checkpoint_dir / f"step_{step:010d}.safetensors"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if observers:
            self.observers_dir.mkdir(exist_ok=True)
            _write_npz(self.observer_path(step), observers)
        else:
            self.observer_path(step).unlink(missing_ok=True)
        write_state_file(path, s, cfg)
        _atomic_write_text(self.checkpoint_dir / LATEST_FILE, path.name + "\n")
        self._prune(cfg.run.keep_checkpoints)
        return path

    def _prune(self, keep: int) -> None:
        """Keep the newest ``keep`` checkpoints, step 0 and whatever ``LATEST`` names; an observer sidecar
        lives exactly as long as its checkpoint."""
        ckpts = self.checkpoints()
        latest = self.latest_checkpoint()
        protected = {p for _, p in ckpts[-keep:]} | {p for st, p in ckpts if st == 0}
        for _, p in ckpts:
            if p not in protected and p != latest:
                p.unlink(missing_ok=True)
        remaining = {st for st, _ in self.checkpoints()}
        for st, p in self.observer_files():
            if st not in remaining:
                p.unlink(missing_ok=True)

    # observer sidecars (§28.6)
    def observer_path(self, step: int) -> Path:
        return self.observers_dir / f"step_{step:010d}.npz"

    def observer_files(self) -> list[tuple[int, Path]]:
        """``(step, path)`` of every observer sidecar, ascending by step."""
        if not self.observers_dir.is_dir():
            return []
        found = [
            (int(m.group(1)), p)
            for p in self.observers_dir.iterdir()
            if (m := _OBSERVER_RE.match(p.name)) and p.is_file()
        ]
        return sorted(found)

    def read_observers(self, step: int) -> dict[str, np.ndarray] | None:
        """The arrays of the sidecar of ``step`` (every ``ObsState`` field and ``meta``), None if there is no
        such file; ``CheckpointError`` if it exists but cannot be read."""
        path = self.observer_path(step)
        if not path.is_file():
            return None
        try:
            z = np.load(path, allow_pickle=False)
            if not isinstance(z, np.lib.npyio.NpzFile):  # a bare .npy (an ndarray), not an archive
                raise ValueError("not an npz archive")
            with z:
                arrays = {k: z[k] for k in z.files}  # a member that is not an .npy comes back as bytes
        except (OSError, ValueError, EOFError, zipfile.BadZipFile, zlib.error) as e:
            raise CheckpointError(f"{path}: unreadable observer sidecar: {e}") from e
        bad = sorted(k for k, a in arrays.items() if not isinstance(a, np.ndarray))
        if bad:
            raise CheckpointError(f"{path}: unreadable observer sidecar: members {bad} are not arrays")
        return arrays

    def checkpoints(self) -> list[tuple[int, Path]]:
        """Complete checkpoints as ``(step, path)``, ascending by step."""
        if not self.checkpoint_dir.is_dir():
            return []
        found = [
            (int(m.group(1)), p)
            for p in self.checkpoint_dir.iterdir()
            if (m := _CKPT_RE.match(p.name)) and p.is_file()
        ]
        return sorted(found)

    def latest_checkpoint(self) -> Path | None:
        """The checkpoint named by ``LATEST``, else the newest complete one, else None."""
        latest = self.checkpoint_dir / LATEST_FILE
        if latest.is_file():
            name = latest.read_text(encoding="utf-8").strip()
            if _CKPT_RE.match(name) and (self.checkpoint_dir / name).is_file():
                return self.checkpoint_dir / name
        ckpts = self.checkpoints()
        return ckpts[-1][1] if ckpts else None

    def checkpoint_at(self, step: int) -> Path:
        """The checkpoint of ``step``; ``FileNotFoundError`` if there is none."""
        path = self.checkpoint_dir / f"step_{step:010d}.safetensors"
        if not path.is_file():
            raise FileNotFoundError(f"{self.path}: no checkpoint at step {step}")
        return path

    # logs
    def append_metrics(self, rows: Iterable[Mapping[str, Any]]) -> None:
        """Append rows to ``metrics.jsonl`` (one JSON object per line, key "step" first) and fsync."""
        ordered = []
        for row in rows:
            if "step" not in row:
                raise ValueError("a metrics row needs a 'step' key")
            ordered.append({"step": row["step"], **{k: v for k, v in row.items() if k != "step"}})
        _append_jsonl(self.path / METRICS_FILE, ordered)

    def metrics_rows(self) -> list[dict[str, Any]]:
        """All rows of ``metrics.jsonl``; a torn last line (a crash mid-append) is ignored."""
        return read_jsonl(self.path / METRICS_FILE)

    def append_jsonl(self, name: str, records: Iterable[Mapping[str, Any]]) -> None:
        """Append records to the run's JSONL file ``name`` (e.g. ``director.jsonl``) and fsync."""
        _append_jsonl(self.path / name, records)

    def read_jsonl(self, name: str) -> list[dict[str, Any]]:
        """All records of the run's JSONL file ``name`` (empty if missing); a torn last line is ignored."""
        return read_jsonl(self.path / name)

    def write_json(self, name: str, obj: Any) -> None:
        """Atomically (re)write the run's JSON file ``name`` (``director_archive.json``)."""
        _atomic_write_text(self.path / name, _dumps(obj) + "\n")

    def write_events(
        self, s0: int, s1: int, births: Mapping[str, np.ndarray], deaths: Mapping[str, np.ndarray]
    ) -> None:
        """Write the records of steps ``s0 ≤ step < s1`` as ``births_*.npz`` and ``deaths_*.npz`` (§18).

        Each mapping holds the §18 columns (any shape, flattened); an optional bool ``valid`` of the same
        size selects the rows to keep, so stacked ``(T, lanes)`` records can be passed directly.
        """
        if not 0 <= s0 <= s1:
            raise ValueError(f"bad event interval [{s0}, {s1})")
        self.events_dir.mkdir(parents=True, exist_ok=True)
        for kind, rec in (("births", births), ("deaths", deaths)):
            cols = _event_columns(kind, rec)
            st = cols["step"]
            if st.size and (st.min() < s0 or st.max() >= s1):
                raise ValueError(f"{kind} record steps outside [{s0}, {s1})")
            _write_npz(self.events_dir / f"{kind}_{s0:010d}_{s1:010d}.npz", cols)

    def event_files(self, kind: str) -> list[tuple[int, int, Path]]:
        """``(s0, s1, path)`` of every ``kind`` (births|deaths) chunk, ascending."""
        if not self.events_dir.is_dir():
            return []
        found = []
        for p in self.events_dir.iterdir():
            m = _EVENT_RE.match(p.name)
            if m and m.group(1) == kind:
                found.append((int(m.group(2)), int(m.group(3)), p))
        return sorted(found)

    def read_events(self, kind: str) -> dict[str, np.ndarray]:
        """Every ``kind`` (births|deaths) record of the run, concatenated over its chunks in step order."""
        if kind not in ("births", "deaths"):
            raise ValueError(f"unknown event kind {kind!r}")
        spec = BIRTH_COLUMNS if kind == "births" else DEATH_COLUMNS
        chunks = [_read_events(path) for _, _, path in self.event_files(kind)]
        return {
            name: np.concatenate([c[name] for c in chunks]).astype(dtype)
            if chunks
            else np.zeros(0, dtype=dtype)
            for name, dtype in spec.items()
        }

    def truncate_after(self, step: int, *, keep_director: bool = False) -> None:
        """Make the logs consistent with the checkpoint at ``step`` (§18 resume, step 4).

        Drops metrics rows with ``step > step``, deletes event chunks starting at or after ``step``, trims a
        chunk spanning it to the records with ``step < step`` (renamed ``*_<s0>_<step>.npz``), and deletes
        checkpoints and observer sidecars after ``step`` so they cannot be confused with the resumed
        trajectory (the sidecars always, whatever ``keep_director`` is). The director files
        are cut the same way (``director.jsonl`` and ``journal.jsonl`` to records with step ≤ ``step``; a
        ``director_archive.json`` taken after it is deleted), except with ``keep_director`` (a replay, which
        must never touch the log it replays).
        """
        if (self.path / METRICS_FILE).exists():
            kept = [r for r in self.metrics_rows() if r["step"] <= step]
            _atomic_write_text(self.path / METRICS_FILE, "".join(_dumps(r) + "\n" for r in kept))
        if not keep_director:
            self._truncate_director(step)

        for kind in ("births", "deaths"):
            for s0, s1, path in self.event_files(kind):
                if s1 <= step:
                    continue
                if s0 >= step:
                    path.unlink()
                    continue
                cols = _read_events(path)
                keep = cols["step"] < step
                trimmed = {k: v[keep] for k, v in cols.items()}
                _write_npz(path.with_name(f"{kind}_{s0:010d}_{step:010d}.npz"), trimmed)
                path.unlink()

        for st, path in self.checkpoints():
            if st > step:
                path.unlink()
        for st, path in self.observer_files():  # whatever keep_director is: a sidecar belongs to its step
            if st > step:
                path.unlink(missing_ok=True)
        latest = self.latest_checkpoint()
        if latest is not None:
            _atomic_write_text(self.checkpoint_dir / LATEST_FILE, latest.name + "\n")
        else:
            (self.checkpoint_dir / LATEST_FILE).unlink(missing_ok=True)

    def _truncate_director(self, step: int) -> None:
        for name in (DIRECTOR_LOG_FILE, JOURNAL_FILE):
            if (self.path / name).exists():
                kept = [r for r in self.read_jsonl(name) if (at := _record_step(r)) is None or at <= step]
                _atomic_write_text(self.path / name, "".join(_dumps(r) + "\n" for r in kept))
        archive = self.path / DIRECTOR_ARCHIVE_FILE
        if archive.exists():
            try:
                as_of = json.loads(archive.read_text(encoding="utf-8")).get("as_of_step")
            except (OSError, json.JSONDecodeError):
                as_of = None
            if not isinstance(as_of, int) or as_of > step:  # rewritten at the next checkpoint
                archive.unlink()

    def dump_diag(self, s: State, step: int) -> Path:
        """Write ``diag/step_<step>.safetensors`` (a full state file) after an invariant violation."""
        self.diag_dir.mkdir(parents=True, exist_ok=True)
        path = self.diag_dir / f"step_{step}.safetensors"
        write_state_file(path, s, self.config())
        return path
