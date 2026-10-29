"""Run directories: config, meta, checkpoints, metrics and event logs (§18). Owned by WP5.

State files, ``LATEST``, ``meta.json``, ``config.json`` and event chunks are written atomically (``*.tmp`` +
``os.replace``), so a crash leaves either the old or the new complete file; ``metrics.jsonl`` is appended and
fsynced, and a torn last line is ignored on read.

The director's ``director.jsonl`` and ``journal.jsonl`` use the same append/fsync/torn-line rules as
``metrics.jsonl`` (the generic ``append_jsonl`` / ``read_jsonl``); ``director_archive.json`` is atomic.

The run dumps ``observers/eco_<step>.npz`` and ``observers/wx_<step>.npy`` (W2, DESIGN §32) are written
atomically at their own cadence (``run.eco_every``, ``run.wx_every``); ``truncate_after`` removes those
after the resume step, as it does the sidecars.

The social log ``observers/social.jsonl`` (an observing run's snapshot records, DESIGN §29.10) has the same
append/fsync/torn-line rules. It belongs to the trajectory like the observer sidecars: ``truncate_after`` cuts
it to the checkpoint step, always (a replay included), after parsing it whole and before changing anything.

Event chunks ``births_<s0>_<s1>.npz`` / ``deaths_<s0>_<s1>.npz`` hold the records of steps ``s0 ≤ step < s1``,
where a record's ``step`` is the index of the step during which it happened; a checkpoint at
``state.step == s1`` therefore contains exactly the effects of every chunk ending at ``s1``.
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import fcntl
import json
import math
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import zipfile
import zlib
from collections.abc import Callable, Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any, Final

import mlx.core as mx
import numpy as np
import zstandard

from alife import ext, layout, rng, spatial
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
COMMANDS_FILE: Final = "commands.jsonl"  # alife play: one line per power request (G3b spec §6.6)
ALERTS_FILE: Final = "alerts.jsonl"  # alife play: every alert raised, updated, cleared or held back (§34.3)
ALERTS_REPLAY_FILE: Final = "alerts.replay.jsonl"  # the wall-clock class's alerts of a replay of the log
REPORTS_FILE: Final = "reports.jsonl"  # alife play: the world report of every chapter, as N1 records (§34.4)
OBSERVERS_DIR: Final = "observers"  # the observer sidecars of an observing run (DESIGN §28.6)
SOCIAL_FILE: Final = "social.jsonl"  # observers/social.jsonl: one record per social snapshot (DESIGN §29.10)
STORMS_FILE: Final = "storms.jsonl"  # observers/storms.jsonl: one line per tracker tick with storms (§33.3)
PLAY_FILE: Final = (
    "play.json"  # a played world's settings; its absence marks a directory as not a play world (§33.3.3)
)
PLAY_LOCK_FILE: Final = "play.lock"  # the advisory lock of the one process that plays a directory (§33.3.3)
PRE_PLAY_CONFIG_FILE: Final = "config.pre_play.json"  # a research run played in place: its config (§33.3.3)
PLAY_VERSION: Final = 1
# observers/eco_<step>.npz and observers/wx_<step>.npy (DESIGN §32), plus the dumps the registered features
# add
DUMP_KINDS: Final = ("eco", "wx", *(k for spec in ext.REGISTRY for k in spec.dump_kinds))
# the file extension of each kind: ``npz`` holds named columns, ``npy`` one array
DUMP_NPZ: Final = frozenset({"eco", "arms"})
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

_CKPT_RE: Final = re.compile(r"^step_(\d{10})\.safetensors(?:\.zst)?$")
SAVE_MODE_NAMES: Final = ("all", "newest", "single")  # DESIGN §34.10, §34.11 (streamrec.SAVE_MODES)
_EVENT_RE: Final = re.compile(r"^(births|deaths)_(\d{10})_(\d{10})\.npz$")
_OBSERVER_RE: Final = re.compile(r"^step_(\d{10})\.npz$")
PLAY_STATE_NAMES: Final = (
    "storms",
    "kinthreads",
)  # observers/<name>_<step>.json: a played world's host state
_PLAY_STATE_RE: Final = re.compile(r"^(storms|kinthreads)_(\d{10})\.json$")
_DUMP_RE: Final = re.compile(rf"^({'|'.join(map(re.escape, DUMP_KINDS))})_(\d{{10}})\.(?:npz|npy)$")
# The temporary names of what a save writes (``_tmp_path``): the state file and ``LATEST`` in
# ``checkpoints/``, the observer sidecar and the play-state files in ``observers/``.
_SAVE_TMP_RE: Final = re.compile(
    rf"^(?:step_\d{{10}}\.(?:safetensors(?:\.zst)?|npz)|{LATEST_FILE}|(?:{'|'.join(PLAY_STATE_NAMES)})_\d{{10}}\.json)\.tmp$"
)
_MAX_HEADER: Final = 1 << 26  # sanity bound on a safetensors JSON header
# Light saves (DESIGN 34.11): a compressed save is the zstd frame of the exact safetensors bytes, named
# ``step_<t>.safetensors.zst``. Every reader goes through this module, which reads both names.
ZST_SUFFIX: Final = ".zst"
ZST_LEVEL: Final = (
    15  # a full save compresses in about 4 s on 18 threads (level 17: 11 s, 8 threads: 15 s), +7% bytes
)
ZST_WINDOW_LOG: Final = 27  # long-distance matching window, as ``zstd --long=27``
_ZST_MAX_WINDOW: Final = 1 << 28  # the largest window a compressed save may name
_CHUNK: Final = 1 << 22


class CheckpointError(Exception):
    """Incompatible or corrupt checkpoint (version, shape or config-hash mismatch)."""


class PlayLockError(Exception):
    """The directory is being played by another process (``pid``, 0 if unknown)."""

    def __init__(self, path: Path, pid: int) -> None:
        super().__init__(f"{path} is being played by process {pid} (play.lock): stop that play first")
        self.pid = pid


def _lock_pid(path: Path) -> int:
    try:
        return int(path.read_text(encoding="utf-8").split()[0])
    except (OSError, ValueError, IndexError):
        return 0


class PlayLock:
    """The advisory ``flock`` of ``<dir>/play.lock``, held by the one process that plays a directory.

    ``acquire`` creates the directory if need be and the lock file, takes an exclusive non-blocking lock and
    writes the pid into it (``PlayLockError`` if another process holds it); ``release`` removes the file and
    drops the lock. The lock goes with the process, so a crash leaves a stale file that locks nothing.
    ``RunDir.create`` ignores a lock file in an otherwise empty directory, so a new world is locked before
    anything else is written to it.
    """

    def __init__(self, directory: Path) -> None:
        self.dir = Path(directory)
        self.path = self.dir / PLAY_LOCK_FILE
        self._fd: int | None = None

    def acquire(self) -> PlayLock:
        self.dir.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            raise PlayLockError(self.dir, _lock_pid(self.path)) from None
        os.ftruncate(fd, 0)
        os.write(fd, f"{os.getpid()}\n".encode())
        self._fd = fd
        return self

    def release(self) -> None:
        if self._fd is None:
            return
        with contextlib.suppress(OSError):
            self.path.unlink()
        with contextlib.suppress(OSError):
            os.close(self._fd)  # closing drops the flock
        self._fd = None

    def __enter__(self) -> PlayLock:
        return self.acquire() if self._fd is None else self

    def __exit__(self, *exc: object) -> None:
        self.release()


def play_lock_holder(directory: Path) -> int | None:
    """The pid of the process playing ``directory`` (0 if its pid is unreadable), else None."""
    path = Path(directory) / PLAY_LOCK_FILE
    try:
        fd = os.open(path, os.O_RDWR)
    except OSError:
        return None
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return _lock_pid(path)
    finally:
        os.close(fd)  # (a lock taken here goes with the descriptor)
    return None


# --- atomic file helpers --------------------------------------------------------------------------------


_TMP_SUFFIX: Final = ".tmp"


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


def _full_sync(path: Path) -> None:
    """Best effort: put ``path`` on permanent storage, past the drive's own cache (``F_FULLFSYNC``; on macOS
    ``os.fsync`` stops at the drive, which may still lose the data in a power cut). ``os.fsync`` where the
    call is missing or the file system refuses it; nothing if the file cannot be opened."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        full = getattr(fcntl, "F_FULLFSYNC", None)
        if full is not None:
            try:
                fcntl.fcntl(fd, full)
                return
            except OSError:
                pass
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _atomic_write(path: Path, write: Any) -> None:
    """Call ``write(f)`` on ``path.tmp`` (binary), fsync it, and move it over ``path``; never leaves a tmp."""
    path = Path(path)
    tmp = _tmp_path(path)
    tmp.unlink(
        missing_ok=True
    )  # a stale tmp may be a hard link to another save: it is replaced, never written
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


def read_jsonl_from(path: Path, min_step: int, chunk: int = 1 << 20) -> list[dict[str, Any]]:
    """The records of a step-ordered JSONL file (metrics rows, social records) with ``step >= min_step``, read
    from the end of the file: a long play's log is not parsed from its first line to report on its last
    chapter. Equal to ``[r for r in read_jsonl(path) if r["step"] >= min_step]`` (a torn last line is ignored;
    a corrupt line inside the range raises ``json.JSONDecodeError``); the file must be in increasing step
    order, which the runner writes it in. ``chunk`` is the size of a read."""
    if not path.is_file():
        return []
    with open(path, "rb") as f:
        pos = f.seek(0, os.SEEK_END)
        buf = b""
        while True:
            start = max(0, pos - chunk)
            f.seek(start)
            buf = f.read(pos - start) + buf
            pos = start
            complete = buf.split(b"\n")
            if pos > 0:
                complete = complete[1:]  # the first piece may be the end of a line that began before
            first = next((ln for ln in complete if ln.strip()), None)
            if pos == 0 or first is None:
                if pos == 0:
                    break
                continue
            try:
                step = json.loads(first).get("step")
            except json.JSONDecodeError:
                step = None  # only the torn last line can fail: it is the first piece of a tiny file
            if isinstance(step, int) and step < min_step:
                break
    rows: list[dict[str, Any]] = []
    for i, line in enumerate(complete):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            if i == len(complete) - 1:
                break
            raise
        if isinstance(rec, dict) and isinstance(rec.get("step"), int) and rec["step"] >= min_step:
            rows.append(rec)
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


def _is_report(rec: Any) -> bool:
    """A line of ``reports.jsonl`` that is a report record: an object with int ``chapter_start``, ``step``."""
    return (
        isinstance(rec, dict)
        and isinstance(rec.get("chapter_start"), int)
        and isinstance(rec.get("step"), int)
        and not isinstance(rec["chapter_start"], bool)
        and not isinstance(rec["step"], bool)
    )


def _is_player_record(rec: Mapping[str, Any]) -> bool:
    """A record of the player channel: a power, or the revert written for one (``actions.is_player``, which
    this repeats because ``director`` imports this module)."""
    return rec.get("source") == "player" or (rec.get("source") == "revert" and rec.get("scope") == "player")


# --- state files ----------------------------------------------------------------------------------------


# The one State schema lives in ``state`` (§2.2, §27.1); this name is kept for tests/test_runio.py.
_expected_layout = expected_layout


def _alife_version() -> str:
    import alife  # local: the package __init__ imports the step graph, which never imports runio

    return alife.__version__


def _state_arrays_and_meta(s: State, cfg: Config) -> tuple[dict[str, Any], dict[str, str]]:
    """The arrays of a state file and its checkpoint metadata (§18)."""
    meta = {
        "format_version": str(FORMAT_VERSION),
        "rng_layout_version": str(rng.layout_version(cfg)),
        "genome_layout_version": str(layout.genome_layout_version(cfg.world.dims, ext.mask(cfg))),
        **(
            {"layout_features": ext.layout_features(cfg)} if ext.layout_features(cfg) else {}
        ),  # absent = none
        **({"dims": str(cfg.world.dims)} if is_3d(cfg) else {}),
        **({"geometry": cfg.world.geometry} if cfg.world.geometry != "box" else {}),  # absent = box (§46.9)
        "step": str(int(to_numpy(s.step))),
        "config_hash": config_hash(cfg),
        "alife_version": _alife_version(),
        "mlx_version": mx.__version__,
        "weight_dtype": cfg.brain.weight_dtype,
        "config": json.dumps(config_to_dict(cfg, omit_defaults=True), sort_keys=True, indent=2),
    }
    return state_to_arrays(s), meta


def write_state_file(path: Path, s: State, cfg: Config) -> None:
    """Atomic safetensors write (``*.tmp`` + ``os.replace``) with checkpoint metadata (§18).

    A 3D state also records ``dims`` (§27.1); a file without the key is 2D. A sphere state records
    ``geometry`` (§46.9); a file without the key is a box.
    """
    arrays, meta = _state_arrays_and_meta(s, cfg)
    _atomic_write(Path(path), lambda f: mx.save_safetensors(f, arrays, meta))


def compress_file(
    src: Path, dst: Path, *, level: int = ZST_LEVEL, check: Callable[[Path], bool] | None = None
) -> bool:
    """Write the zstd frame of the bytes of ``src`` to ``dst``: through ``dst.tmp``, fsynced, then renamed.

    The frame names its size and carries a checksum, so a damaged copy fails to decode (DESIGN §34.11). With
    ``check`` (called with the temporary file), the rename happens only when it returns True: the caller's
    whole-file check goes before the name moves. Returns whether ``dst`` was written."""
    size = os.path.getsize(src)
    params = zstandard.ZstdCompressionParameters.from_level(
        level,
        source_size=size,
        enable_ldm=True,
        window_log=ZST_WINDOW_LOG,
        threads=-1,
        write_checksum=True,
        write_content_size=True,
    )
    cctx = zstandard.ZstdCompressor(compression_params=params)
    dst = Path(dst)
    tmp = _tmp_path(dst)
    tmp.unlink(missing_ok=True)
    try:
        with open(tmp, "wb") as out:
            with open(src, "rb") as ifh:
                cctx.copy_stream(ifh, out, size=size, read_size=1 << 24, write_size=1 << 22)
            out.flush()
            os.fsync(out.fileno())
        if check is not None and not check(tmp):
            tmp.unlink(missing_ok=True)
            return False
        os.replace(tmp, dst)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    _fsync_dir(dst.parent)
    return True


class _ZstStream:
    """The plain bytes of a compressed save, decoded as they are read. A save that does not end (truncated),
    that
    is followed by bytes, or whose frame fails its checksum raises ``CheckpointError``; nothing is taken for
    whole
    before its frame ends."""

    def __init__(self, f: Any, path: Path) -> None:
        self._f = f
        self._path = path
        self._d = zstandard.ZstdDecompressor(max_window_size=_ZST_MAX_WINDOW).decompressobj()
        self._buf = b""
        self._gen = self._pieces()

    def _pieces(self) -> Iterator[bytes]:
        try:
            while not self._d.eof:
                data = self._f.read(_CHUNK)
                if not data:
                    raise CheckpointError(
                        f"{self._path}: truncated compressed save (its zstd frame does not end)"
                    )
                out = self._d.decompress(data)
                if self._d.unused_data:
                    raise CheckpointError(f"{self._path}: bytes after the end of the compressed save")
                if out:
                    yield out
        except zstandard.ZstdError as e:
            raise CheckpointError(f"{self._path}: corrupt compressed save: {e}") from e

    def read(self, n: int) -> bytes:
        """Up to ``n`` plain bytes (fewer only at the end of the file); decodes no more than they need."""
        while len(self._buf) < n:
            piece = next(self._gen, None)
            if piece is None:
                break
            self._buf += piece
        out, self._buf = self._buf[:n], self._buf[n:]
        return out

    def drain(self) -> Iterator[bytes]:
        """The rest of the plain bytes, in pieces."""
        if self._buf:
            buf, self._buf = self._buf, b""
            yield buf
        yield from self._gen


def is_compressed(path: Path) -> bool:
    """True for a compressed save, ``step_<t>.safetensors.zst`` (or its name while it is being written)."""
    name = Path(path).name
    if name.endswith(_TMP_SUFFIX):
        name = name[: -len(_TMP_SUFFIX)]
    return name.endswith(".safetensors" + ZST_SUFFIX)


@contextlib.contextmanager
def _plain_reader(path: Path) -> Iterator[Any]:
    """A reader of the plain safetensors bytes of ``path`` (a compressed save is decoded as it is read)."""
    with open(path, "rb") as f:
        yield _ZstStream(f, Path(path)) if is_compressed(path) else f


def checkpoint_metadata(path: Path) -> dict[str, str]:
    """The string metadata of a state file, read from the safetensors header only (for a compressed save, only
    the first pieces of its frame are decoded)."""
    try:
        with _plain_reader(path) as f:
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


def _compressed_is_whole(path: Path, step: int) -> bool:
    """The whole-file check of a compressed save: its frame decodes to the end (the checksum is verified), its
    header names ``step``, and the decoded bytes reach the end of the last tensor."""
    try:
        with open(path, "rb") as f:
            src = _ZstStream(f, Path(path))
            raw = src.read(8)
            if len(raw) != 8:
                return False
            (size,) = struct.unpack("<Q", raw)
            if size > _MAX_HEADER:
                return False
            header = json.loads(src.read(size))
            meta = header.get("__metadata__", {})
            ends = [
                int(v["data_offsets"][1])
                for k, v in header.items()
                if k != "__metadata__" and isinstance(v, dict)
            ]
            total = 8 + size + sum(len(piece) for piece in src.drain())
        return str(meta.get("step")) == str(step) and total >= 8 + size + max(ends, default=0)
    except (OSError, ValueError, KeyError, IndexError, TypeError, AttributeError, CheckpointError):
        return False


def state_file_is_complete(path: Path, step: int) -> bool:
    """True when ``path`` is a whole state file of ``step``: a readable header that names ``step`` and a file
    at least as long as its last tensor ends (a short or torn write fails; the bytes themselves are not
    read). A compressed save is decoded whole instead, which checks its checksum too (DESIGN §34.11). The
    check
    ``save_checkpoint`` makes before the newest-only mode deletes the older saves."""
    if is_compressed(path):
        return _compressed_is_whole(path, step)
    try:
        with open(path, "rb") as f:
            raw = f.read(8)
            if len(raw) != 8:
                return False
            (size,) = struct.unpack("<Q", raw)
            if size > _MAX_HEADER:
                return False
            header = json.loads(f.read(size))
        meta = header.get("__metadata__", {})
        ends = [
            int(v["data_offsets"][1])
            for k, v in header.items()
            if k != "__metadata__" and isinstance(v, dict)
        ]
        return str(meta.get("step")) == str(step) and path.stat().st_size >= 8 + size + max(ends, default=0)
    except (OSError, ValueError, KeyError, IndexError, TypeError, AttributeError):
        return False


def _file_weather_off(meta: dict[str, str]) -> bool:
    """True when the state file's own config (its metadata) had the weather off, so its ``wx`` is the constant
    weather-off array of the default parameters, not a weather state to keep."""
    try:
        return not bool(json.loads(meta["config"])["weather"]["enabled"])
    except (KeyError, TypeError, ValueError):
        return False


def _decode_to(path: Path, dst: Path) -> None:
    with open(path, "rb") as f, open(dst, "wb") as out:
        for piece in _ZstStream(f, Path(path)).drain():
            out.write(piece)


@contextlib.contextmanager
def state_file(path: Path) -> Iterator[Path]:
    """The path of a plain safetensors file with the bytes of the state file ``path``: the file itself, or for
    a
    compressed save a decoded temporary copy, removed on exit. ``CheckpointError`` if the save is unreadable,
    corrupt or truncated (DESIGN §34.11)."""
    path = Path(path)
    if not is_compressed(path):
        yield path
        return
    fd, name = tempfile.mkstemp(prefix="alife-state-", suffix=".safetensors")
    os.close(fd)
    tmp = Path(name)
    try:
        try:
            _decode_to(path, tmp)
        except OSError as e:
            raise CheckpointError(f"{path}: cannot read state file: {e}") from e
        yield tmp
    finally:
        tmp.unlink(missing_ok=True)


def read_state_arrays(path: Path, keys: Iterable[str] | None = None) -> dict[str, Any]:
    """The arrays of a state file, read (``keys``: only these, all by default); ``CheckpointError`` if
    unreadable."""
    try:
        with state_file(path) as plain:
            arrays = mx.load(str(plain), format="safetensors")
            names = list(arrays) if keys is None else [k for k in keys if k in arrays]
            mx.eval([arrays[k] for k in names])
    except (ValueError, RuntimeError, OSError) as e:
        raise CheckpointError(f"{path}: {e}") from e
    return {k: arrays[k] for k in names}


def load_checkpoint(path: Path, cfg: Config, *, check_hash: bool = True) -> State:
    """Load a state file; ``CheckpointError`` on a version/shape mismatch, or a config-hash one if checked.

    Keys are matched by name (their order in the file is arbitrary, F15). The file's ``dims`` (absent = 2)
    must equal ``world.dims`` and its ``geometry`` (absent = box) ``world.geometry``: a state never crosses
    dimensions (§27.1) or geometries (§46.9). The RNG layout version must be in
    ``rng.RNG_COMPATIBLE_VERSIONS``. A pre-director file (no ``env_*`` arrays) loads with inert env values.
    A file whose own config had the weather off loads into a weather-on ``cfg`` with ``wx`` replaced by the
    rest array of ``cfg`` (its shape follows ``weather.cells``, which the file's constant array does not;
    DESIGN §30.3). The result is evaluated.
    """
    path = Path(path)
    meta = checkpoint_metadata(path)
    nd = str(cfg.world.dims)
    bad = []
    if meta.get("dims", "2") != nd:
        bad.append(
            f"dims={meta.get('dims', '2')!r} (expected {nd!r}): cross-dimension resume/fork is not supported"
        )
    if meta.get("geometry", "box") != cfg.world.geometry:
        bad.append(
            f"geometry={meta.get('geometry', 'box')!r} (expected {cfg.world.geometry!r}): "
            "cross-geometry resume/fork is not supported"
        )
    want = {
        "format_version": str(FORMAT_VERSION),
        "genome_layout_version": str(layout.genome_layout_version(cfg.world.dims, ext.mask(cfg))),
        "layout_features": ext.layout_features(cfg),  # a file without the key has none (today's layout)
    }
    if check_hash:
        want["config_hash"] = config_hash(cfg)
    bad += [
        f"{k}={meta.get(k, '' if k == 'layout_features' else None)!r} (expected {v!r})"
        for k, v in want.items()
        if meta.get(k, "" if k == "layout_features" else None) != v
    ]
    if meta.get("rng_layout_version") not in {str(v) for v in rng.RNG_COMPATIBLE_VERSIONS}:
        compatible = sorted(rng.RNG_COMPATIBLE_VERSIONS)
        bad.append(f"rng_layout_version={meta.get('rng_layout_version')!r} (expected one of {compatible})")
    if bad:
        hint = ""
        if any(b.startswith(("layout_features", "genome_layout_version")) for b in bad):
            hint = (
                "; the config and the checkpoint disagree on the feature layout (which extension "
                "features are on): a checkpoint never crosses layout_features and nothing is guessed. "
                "Fix the config, or migrate the checkpoint with `alife migrate` (DESIGN §43.8)"
            )
        raise CheckpointError(f"{path}: incompatible checkpoint: {', '.join(bad)}{hint}")
    try:
        with state_file(path) as plain:
            arrays = mx.load(str(plain), format="safetensors")
            if cfg.weather.enabled and "wx" in arrays and _file_weather_off(meta):
                arrays = {k: v for k, v in arrays.items() if k != "wx"}  # refilled from rest_array(cfg, step)
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
    keep_newest: bool
    drop_origin: bool
    compress: bool
    """Whether a save is written compressed (DESIGN §34.11): play worlds, set by the play pacer. A runtime
    setting, not
    a config key, like ``keep_newest``."""
    _save_worker: threading.Thread | None
    _save_error: BaseException | None
    _save_done: threading.Event
    """A played world's save mode (DESIGN §34.10): True keeps only the newest checkpoint and the origin at
    each save, whatever ``run.keep_checkpoints`` says. Set by the play pacer: a runtime setting of this
    object, not a config key, so it never reaches the config hash or a trajectory."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.keep_newest = False
        self.drop_origin = False
        self.origin_link: tuple[int, Path] | None = None
        self.compress = False
        self._save_worker: threading.Thread | None = None
        self._save_error: BaseException | None = None
        self._save_done = threading.Event()

    def set_save_mode(self, mode: str) -> None:
        """The save mode (DESIGN §34.10, §34.11): ``all`` keeps ``run.keep_checkpoints`` saves and the origin;
        ``newest`` keeps the newest save and the origin; ``single`` keeps only the newest save (the origin and
        step 0 go too). It takes effect at the next save: nothing is deleted here."""
        if mode not in SAVE_MODE_NAMES:
            raise ValueError(f"the save mode is one of {SAVE_MODE_NAMES}, got {mode!r}")
        self.keep_newest = mode != "all"
        self.drop_origin = mode == "single"

    @property
    def save_mode(self) -> str:
        return "single" if self.drop_origin else ("newest" if self.keep_newest else "all")

    @property
    def adopted(self) -> bool:
        """A research run played in place (``alife play RUN --adopt``): its first adoption kept
        ``config.json`` as ``config.pre_play.json``, which nothing removes. Its older checkpoints are the
        research's, which a ``saves newest`` line does not delete (DESIGN 34.10)."""
        return (self.path / PRE_PLAY_CONFIG_FILE).is_file()

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
    def social_path(self) -> Path:
        """``observers/social.jsonl``, the social log of an observing run (absent for any other run)."""
        return self.observers_dir / SOCIAL_FILE

    @property
    def storms_path(self) -> Path:
        """``observers/storms.jsonl``, the storm log of a played weather world (absent for any other run)."""
        return self.observers_dir / STORMS_FILE

    @property
    def media_dir(self) -> Path:
        return self.path / "media"

    @staticmethod
    def require_new(path: Path) -> None:
        """``FileExistsError`` unless ``path`` is absent or an empty directory (a ``play.lock`` file is not
        content): what ``create`` checks, for a caller that must know before it writes anything."""
        path = Path(path)
        if path.exists():
            if not path.is_dir():
                raise FileExistsError(f"{path} exists and is not a directory")
            if any(p.name != PLAY_LOCK_FILE for p in path.iterdir()):
                raise FileExistsError(f"{path} is not empty (use overwrite to replace a run directory)")

    @classmethod
    def create(
        cls,
        path: Path,
        cfg: Config,
        *,
        overwrite: bool = False,
        parent: str | None = None,
        extra_meta: Mapping[str, Any] | None = None,
    ) -> RunDir:
        """Create a fresh run directory holding ``cfg`` and a ``running`` meta.

        An existing non-empty directory is an error unless ``overwrite`` is set and it is a run directory
        (has ``meta.json``); anything else is never deleted. ``extra_meta`` keys are written into the first
        ``meta.json`` (``alife play`` marks a world it builds, §33.3.3); it cannot replace a key of the meta.
        """
        path = Path(path)
        if not overwrite:
            cls.require_new(path)
        elif path.exists():
            if not path.is_dir():
                raise FileExistsError(f"{path} exists and is not a directory")
            if any(p.name != PLAY_LOCK_FILE for p in path.iterdir()):  # a lock file is not content
                holder = play_lock_holder(path)
                if holder is not None:
                    raise FileExistsError(
                        f"{path} is being played by process {holder} (play.lock): refusing to overwrite it"
                    )
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
            "rng_layout_version": rng.layout_version(cfg),
            "genome_layout_version": layout.genome_layout_version(cfg.world.dims, ext.mask(cfg)),
            **(
                {"layout_features": ext.layout_features(cfg)} if ext.layout_features(cfg) else {}
            ),  # absent = none
            **({"dims": cfg.world.dims} if is_3d(cfg) else {}),  # absent = 2D (§27.1)
            **({"geometry": cfg.world.geometry} if cfg.world.geometry != "box" else {}),  # absent = box
            "parent": parent,
            "status": "running",
        }
        if extra_meta:
            clash = sorted(set(extra_meta) & set(meta))
            if clash:
                raise ValueError(f"extra_meta would replace the meta keys {clash}")
            meta.update(extra_meta)
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

    # play.json (§33.3.3)
    @property
    def play_path(self) -> Path:
        return self.path / PLAY_FILE

    def read_play_json(self) -> dict[str, Any] | None:
        """The world's ``play.json``, None if the directory has none (it is not a play world, or unfinished);
        ``ValueError`` if it is not a version-1 object."""
        if not self.play_path.is_file():
            return None
        obj = json.loads(self.play_path.read_text(encoding="utf-8"))
        if not isinstance(obj, dict) or obj.get("play_version") != PLAY_VERSION:
            raise ValueError(f"{self.play_path}: not a play.json of version {PLAY_VERSION}")
        return obj

    def write_play_json(self, obj: Mapping[str, Any]) -> None:
        """Atomically write ``play.json`` (version 1)."""
        _atomic_write_text(self.play_path, _dumps({**obj, "play_version": PLAY_VERSION}) + "\n")

    # checkpoints
    def save_checkpoint(
        self, s: State, cfg: Config, *, observers: Mapping[str, np.ndarray] | None = None
    ) -> Path:
        """sidecar + the state file + LATEST + prune (the save mode decides what the prune keeps).

        ``observers`` (the arrays of ``Observer.sidecar()``) is written to ``observers/step_<t>.npz`` before
        the state file, so every complete checkpoint of an observing run has its sidecar (a crash in between
        leaves an orphan sidecar, which pruning removes). Without ``observers`` a sidecar of this step is
        removed, so a checkpoint and its sidecar always come from the same save (§28.6).

        ``origin_link`` (DESIGN §34.11), when it names this step, makes the state file a link to the source
        save instead of a write: the caller has checked that the state is the source's, byte for byte.
        """
        self.wait_saves()  # a compressed save in flight finishes first (its prune must not meet this one)
        step = int(to_numpy(s.step))
        path = self.checkpoint_dir / f"step_{step:010d}.safetensors"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if observers:
            self.observers_dir.mkdir(exist_ok=True)
            _write_npz(self.observer_path(step), observers)
        else:
            self.observer_path(step).unlink(missing_ok=True)
        link: Path | None = None
        if self.origin_link is not None and self.origin_link[0] == step:
            link = self.origin_link[1]
            self.origin_link = None
        if link is not None:
            linked = self._link_origin(path, link)
            if linked is not None:
                return self._commit(linked, step, cfg.run.keep_checkpoints)
            # the source save is gone (pruned while the source world still plays): the state is in memory,
            # so it is saved as any state is (DESIGN §34.11)
        if self.compress:
            return self._save_compressed(path, s, cfg, step)
        write_state_file(path, s, cfg)
        return self._commit(path, step, cfg.run.keep_checkpoints)

    def _link_origin(self, path: Path, link: Path) -> Path | None:
        """The state file of a planet's origin as a link to its source save (``_link_state``), or None when
        the source cannot be linked or copied (it was pruned): the caller then saves the state in memory."""
        dst = path.with_name(path.name + (ZST_SUFFIX if is_compressed(link) else ""))
        try:
            self._link_state(dst, link)
        except OSError as e:
            print(
                f"save: the origin of this world is not linked ({e}); the state is saved as it is",
                file=sys.stderr,
                flush=True,
            )
            return None
        return dst

    def _commit(self, path: Path, step: int, keep: int) -> Path:
        """LATEST moves to the complete save ``path`` and the older saves are pruned (DESIGN §34.10, §34.11).

        In a newest-only mode the new save must read back whole first (on the drive, not only in its cache):
        a failing one is removed, LATEST stays on the previous save and nothing is pruned."""
        origin = True
        if self.keep_newest:
            _full_sync(path)
            if not state_file_is_complete(path, step):
                path.unlink(missing_ok=True)
                print(
                    f"save: {path} does not read back whole; LATEST stays on the previous save and the older "
                    "saves are kept",
                    file=sys.stderr,
                    flush=True,
                )
                return path
            keep, origin = 1, not self.drop_origin
        _atomic_write_text(self.checkpoint_dir / LATEST_FILE, path.name + "\n")
        self._prune(keep, origin=origin)
        return path

    def _link_state(self, dst: Path, src: Path) -> None:
        """Makes ``dst``, a state file name of this run, the save ``src`` of another run (the same bytes): a
        hard link when the volume allows it (``os.link``), else a copy. Both are made under a temporary name
        and moved
        with ``os.replace``, so nothing is written into ``src`` and a kill leaves no file under ``dst``'s name
        (DESIGN §34.11)."""
        tmp = _tmp_path(dst)
        tmp.unlink(missing_ok=True)
        try:
            try:
                os.link(src, tmp)
            except OSError:
                shutil.copyfile(src, tmp)
                with open(tmp, "rb+") as f:
                    os.fsync(f.fileno())
            os.replace(tmp, dst)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        _fsync_dir(dst.parent)

    def _save_compressed(self, path: Path, s: State, cfg: Config, step: int) -> Path:
        """A compressed save (DESIGN §34.11). The plain save is written as an ordinary state file at ``path``
        and ``LATEST`` names it at once, so the world always has a complete save to resume. A worker thread
        then compresses it to ``<name>.zst`` (its whole-file check runs before the name moves), moves
        ``LATEST`` over that file, deletes the plain copy and prunes. A kill at any point leaves a resumable
        world: before ``LATEST`` moves, the plain save is the newest; after, the compressed one is."""
        self._remove_debris()  # no worker runs now: wait_saves() came first
        write_state_file(path, s, cfg)
        if not state_file_is_complete(path, step):  # LATEST never names a save that fails its check
            path.unlink(missing_ok=True)
            print(
                f"save: {path} does not read back whole; LATEST stays on the previous save",
                file=sys.stderr,
                flush=True,
            )
            return path
        _atomic_write_text(self.checkpoint_dir / LATEST_FILE, path.name + "\n")
        zst = path.with_name(path.name + ZST_SUFFIX)
        keep_newest = self.keep_newest
        drop_origin = self.drop_origin
        keep = cfg.run.keep_checkpoints
        done = threading.Event()
        self._save_done = done

        def work() -> None:
            try:
                whole = compress_file(path, zst, check=lambda tmp: state_file_is_complete(tmp, step))
                if not whole:
                    print(
                        f"save: {zst} does not decode whole; LATEST stays on {path.name}",
                        file=sys.stderr,
                        flush=True,
                    )
                    return
                _full_sync(zst)  # on the drive before LATEST names it and the older saves go
                _atomic_write_text(self.checkpoint_dir / LATEST_FILE, zst.name + "\n")
                path.unlink(missing_ok=True)
                self._prune_saves(1 if keep_newest else keep, origin=not (drop_origin and keep_newest))
            except Exception as e:  # raised by wait_saves in the thread that saves
                self._save_error = e
            finally:
                done.set()

        self._save_worker = threading.Thread(target=work, name="alife-save")
        self._save_worker.start()
        return zst

    def wait_saves(self) -> None:
        """Wait for the compressed save in flight, if there is one (DESIGN §34.11): the play loop calls this
        before each save and when it ends. An error of that save is raised here, once; the save before it
        stays the newest. A KeyboardInterrupt during the wait does not end it: the wait goes on until the save
        is whole (or failed), and the interrupt is raised after that."""
        worker = self._save_worker
        interrupted = False
        if worker is not None:
            done = self._save_done
            while True:
                try:  # the whole wait is in the try: an interrupt anywhere in it is remembered, never lost
                    if done.is_set() and not worker.is_alive():
                        break
                    worker.join(0.25)
                except KeyboardInterrupt:
                    interrupted = True
            self._save_worker = None
        err, self._save_error = self._save_error, None
        if err is not None:
            raise err
        if interrupted:
            raise KeyboardInterrupt

    def _start_step(self) -> int:
        """``meta.json`` ``start_step`` (0 for a run from step 0 and for a directory without the key)."""
        try:
            return int(self.meta().get("start_step") or 0)
        except (OSError, ValueError, AttributeError):
            return 0

    def _prune(self, keep: int, *, origin: bool = True) -> None:
        """Keep the newest ``keep`` checkpoints, step 0, the fork's origin checkpoint (``meta.json``
        ``start_step``) and whatever ``LATEST`` names; an observer sidecar lives exactly as long as its
        checkpoint. ``origin`` False (the single save mode, DESIGN §34.11) keeps neither step 0 nor the
        origin.

        The origin is what a fork's phylogeny, activity analysis and recap builders read for their founders
        (``checkpoint_at(start_step)``); without the protection the third save of a fork deletes it (§18).

        The temporary file of a save (``step_<t>.safetensors.tmp``, ``LATEST.tmp``, a sidecar's or a
        play-state file's ``.tmp``) that is still there is the debris of a save a kill interrupted (a state
        file's is up to the size of a save, DESIGN 34.10): the one process that saves the run writes these
        names only inside ``save_checkpoint`` and the play loop's ``before_checkpoint``, which are done by
        now, so it is removed too. Other ``*.tmp`` files (a houses file an analysis may be writing) are not.
        """
        self._remove_debris()
        self._prune_saves(keep, origin=origin)

    def _remove_debris(self) -> None:
        """Removes the temporary files of saves (see ``_prune``); only the one process that saves writes
        them."""
        for d in (self.checkpoint_dir, self.observers_dir):
            if d.is_dir():
                for tmp in d.iterdir():
                    if _SAVE_TMP_RE.match(tmp.name):
                        tmp.unlink(missing_ok=True)

    def _prune_saves(self, keep: int, *, origin: bool = True) -> None:
        """What ``_prune`` keeps and deletes, without the debris removal (the worker of a compressed save
        calls
        it). Keeps the newest ``keep`` steps, ``LATEST`` and, with ``origin``, step 0 and the origin. Deletes
        every other save file of either name (a step may have a plain and a compressed file while a save is
        made), then the sidecars and play-state files of the steps that are gone."""
        latest = self.latest_checkpoint()
        ckpts = self.checkpoints()
        starts = {0, self._start_step()} if origin else set()
        kept = {st for st, _ in ckpts[-keep:]} | {st for st, _ in ckpts if st in starts}
        if self.checkpoint_dir.is_dir():
            for p in list(self.checkpoint_dir.iterdir()):
                m = _CKPT_RE.match(p.name)
                if m and p.is_file() and p != latest and int(m.group(1)) not in kept:
                    p.unlink(missing_ok=True)
        remaining = {st for st, _ in self.checkpoints()}
        for st, p in self.observer_files():
            if st not in remaining:
                p.unlink(missing_ok=True)
        for name in PLAY_STATE_NAMES:  # so do the tracker state and the kin threads of a played world
            for st, p in self.play_state_files(name):
                if st not in remaining:
                    p.unlink(missing_ok=True)

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

    # a played world's host state (§33.3, G3b §34.1): one JSON file per checkpoint and name
    def play_state_path(self, name: str, step: int) -> Path:
        """``observers/<name>_<step>.json`` (``name`` is one of ``PLAY_STATE_NAMES``)."""
        if name not in PLAY_STATE_NAMES:
            raise ValueError(f"unknown play state {name!r}")
        return self.observers_dir / f"{name}_{step:010d}.json"

    def play_state_files(self, name: str) -> list[tuple[int, Path]]:
        """``(step, path)`` of every state file of ``name``, ascending by step."""
        if name not in PLAY_STATE_NAMES:
            raise ValueError(f"unknown play state {name!r}")
        if not self.observers_dir.is_dir():
            return []
        found = [
            (int(m.group(2)), p)
            for p in self.observers_dir.iterdir()
            if (m := _PLAY_STATE_RE.match(p.name)) and m.group(1) == name and p.is_file()
        ]
        return sorted(found)

    def write_play_state(self, name: str, step: int, obj: Any) -> None:
        """Atomically write ``observers/<name>_<step>.json``, the host state of the played world for the
        checkpoint of ``step``. The play loop writes it before the state file, as the sidecar is."""
        self.observers_dir.mkdir(exist_ok=True)
        _atomic_write_text(self.play_state_path(name, step), _dumps(obj) + "\n")

    def read_play_state(self, name: str, step: int) -> dict[str, Any] | None:
        """The state of ``name`` saved for ``step``, None if there is none; ``ValueError`` if it is not a JSON
        object (``OSError`` if it cannot be read)."""
        path = self.play_state_path(name, step)
        if not path.is_file():
            return None
        obj = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(obj, dict):
            raise ValueError(f"{path}: the {name} state is not a JSON object")
        return obj

    # storm tracker state and log (§33.3)
    def storms_state_path(self, step: int) -> Path:
        return self.play_state_path("storms", step)

    def storms_state_files(self) -> list[tuple[int, Path]]:
        """``(step, path)`` of every storm tracker state file, ascending by step."""
        return self.play_state_files("storms")

    def write_storms_state(self, step: int, obj: Any) -> None:
        """The storm tracker's complete state for the checkpoint of ``step`` (``write_play_state``)."""
        self.write_play_state("storms", step, obj)

    def read_storms_state(self, step: int) -> dict[str, Any] | None:
        """The tracker state saved for ``step`` (``read_play_state``)."""
        return self.read_play_state("storms", step)

    def append_storms(self, lines: Iterable[Mapping[str, Any]]) -> None:
        """Append tick lines to ``observers/storms.jsonl`` (key ``step`` first) and fsync. Every line needs
        an int ``step`` and the steps must strictly increase within the call (``ValueError``); an empty
        iterable is a no-op that creates nothing."""
        recs = list(lines)
        if not recs:
            return
        last: int | None = None
        for rec in recs:
            step = rec.get("step") if isinstance(rec, Mapping) else None
            if not isinstance(step, int | np.integer) or isinstance(step, bool):
                raise ValueError("a storm log line needs an int 'step' key")
            if last is not None and int(step) <= last:
                raise ValueError(f"storm log steps must increase within a call ({int(step)} after {last})")
            last = int(step)
        self.observers_dir.mkdir(exist_ok=True)
        _append_jsonl(
            self.storms_path,
            [{"step": r["step"], **{k: v for k, v in r.items() if k != "step"}} for r in recs],
        )

    def storms_lines(self) -> list[dict[str, Any]]:
        """All lines of the storm log, in step order; ``[]`` if there is none. A torn last line is ignored."""
        return read_jsonl(self.storms_path)

    def _parse_alerts(self, name: str) -> list[dict[str, Any]] | None:
        """An alert log (``alerts.jsonl``, ``alerts.replay.jsonl``) parsed whole, with the rule of
        ``_parse_storms`` (None if absent, a torn last line dropped, a line without an int ``step`` raises
        ``ValueError``)."""
        path = self.path / name
        if not path.is_file():
            return None
        lines = read_jsonl(path)
        for i, rec in enumerate(lines, start=1):
            step = rec.get("step") if isinstance(rec, dict) else None
            if not isinstance(step, int) or isinstance(step, bool):
                raise ValueError(f"{path}: line {i} is not an object with an int 'step'")
        return lines

    def _parse_storms(self) -> list[dict[str, Any]] | None:
        """The storm log parsed whole, with the rule of ``_parse_social`` (None if absent; a torn last line is
        dropped; a corrupt line elsewhere raises ``json.JSONDecodeError``, a line without an int ``step``
        ``ValueError``)."""
        if not self.storms_path.is_file():
            return None
        records = read_jsonl(self.storms_path)
        for i, rec in enumerate(records, start=1):
            step = rec.get("step") if isinstance(rec, dict) else None
            if not isinstance(step, int) or isinstance(step, bool):
                raise ValueError(f"{self.storms_path}: line {i} is not an object with an int 'step'")
        return records

    # run dumps (W2, DESIGN §32): trajectory-bound like the sidecars, but not tied to a checkpoint
    def dump_path(self, kind: str, step: int) -> Path:
        """``observers/eco_<step>.npz`` or ``observers/wx_<step>.npy``."""
        if kind not in DUMP_KINDS:
            raise ValueError(f"unknown dump kind {kind!r}")
        return self.observers_dir / f"{kind}_{step:010d}.{'npz' if kind in DUMP_NPZ else 'npy'}"

    def write_eco(self, step: int, cols: Mapping[str, np.ndarray]) -> Path:
        """Atomically write the ecology dump of ``step`` (the live rows' columns)."""
        self.observers_dir.mkdir(exist_ok=True)
        path = self.dump_path("eco", step)
        _write_npz(path, cols)
        return path

    def write_wx(self, step: int, planes: np.ndarray) -> Path:
        """Atomically write the weather dump of ``step``: ``(10, n, n)`` float32 (9 planes and ``1 - qs``)."""
        self.observers_dir.mkdir(exist_ok=True)
        path = self.dump_path("wx", step)
        _atomic_write(path, lambda f: np.save(f, np.ascontiguousarray(planes, dtype=np.float32)))
        return path

    def write_grid(self, kind: str, step: int, grid: np.ndarray) -> Path:
        """Atomically write the grid dump of ``step``: ``observers/mat_<step>.npy`` (the built material,
        DESIGN §38.9) or ``observers/niche_<step>.npy`` (the soil, §41.14), int32 of the field shape."""
        if kind not in ("mat", "niche"):
            raise ValueError(f"{kind!r} is not a grid dump")
        self.observers_dir.mkdir(exist_ok=True)
        path = self.dump_path(kind, step)
        _atomic_write(path, lambda f: np.save(f, np.ascontiguousarray(grid, dtype=np.int32)))
        return path

    def read_grid(self, kind: str, step: int) -> np.ndarray:
        """The int32 grid of the ``mat`` or ``niche`` dump of ``step``."""
        if kind not in ("mat", "niche"):
            raise ValueError(f"{kind!r} is not a grid dump")
        return np.load(self.dump_path(kind, step), allow_pickle=False)

    def write_arms(self, step: int, cols: Mapping[str, np.ndarray]) -> Path:
        """Atomically write the arms trait sample of ``step`` (DESIGN §42.6): the alive rows' columns."""
        self.observers_dir.mkdir(exist_ok=True)
        path = self.dump_path("arms", step)
        _write_npz(path, cols)
        return path

    def read_arms(self, step: int) -> dict[str, np.ndarray]:
        """The columns of the arms sample of ``step``."""
        with np.load(self.dump_path("arms", step), allow_pickle=False) as z:
            return {k: z[k] for k in z.files}

    def dump_files(self, kind: str) -> list[tuple[int, Path]]:
        """``(step, path)`` of every ``kind`` dump, ascending by step."""
        if kind not in DUMP_KINDS:
            raise ValueError(f"unknown dump kind {kind!r}")
        if not self.observers_dir.is_dir():
            return []
        found = [
            (int(m.group(2)), p)
            for p in self.observers_dir.iterdir()
            if (m := _DUMP_RE.match(p.name)) and m.group(1) == kind and p.is_file()
        ]
        return sorted(found)

    def read_eco(self, step: int) -> dict[str, np.ndarray]:
        """The columns of the ecology dump of ``step``."""
        with np.load(self.dump_path("eco", step), allow_pickle=False) as z:
            return {k: z[k] for k in z.files}

    def read_wx(self, step: int) -> np.ndarray:
        """The ``(10, n, n)`` weather dump of ``step``."""
        return np.load(self.dump_path("wx", step), allow_pickle=False)

    def _checkpoint_files(self) -> list[tuple[int, Path]]:
        """Every state file of ``checkpoints/``, both forms of a step included, ascending by step."""
        if not self.checkpoint_dir.is_dir():
            return []
        found = [
            (int(m.group(1)), p)
            for p in self.checkpoint_dir.iterdir()
            if (m := _CKPT_RE.match(p.name)) and p.is_file()
        ]
        return sorted(found, key=lambda sp: (sp[0], is_compressed(sp[1])))

    def adopt_newest_save(self) -> Path | None:
        """Moves ``LATEST`` to the newest checkpoint when that one is complete and newer than the one
        ``LATEST`` names (DESIGN §34.11). A kill between a save's state file and its ``LATEST`` leaves exactly
        that: a whole save a resume must take without ``--truncate``. The check is the save's own
        (``state_file_is_complete``). Returns the adopted save, else None."""
        ckpts = self.checkpoints()
        if not ckpts:
            return None
        newest_step, newest = ckpts[-1]
        latest_step = -1
        latest = self.checkpoint_dir / LATEST_FILE
        if latest.is_file():
            name = latest.read_text(encoding="utf-8").strip()
            if (m := _CKPT_RE.match(name)) and (self.checkpoint_dir / name).is_file():
                latest_step = int(m.group(1))
        if latest_step >= newest_step or not state_file_is_complete(newest, newest_step):
            return None
        _atomic_write_text(latest, newest.name + "\n")
        return newest

    def checkpoints(self) -> list[tuple[int, Path]]:
        """Complete checkpoints as ``(step, path)``, ascending by step."""
        if not self.checkpoint_dir.is_dir():
            return []
        found: dict[int, Path] = {}
        for p in self.checkpoint_dir.iterdir():
            if (m := _CKPT_RE.match(p.name)) and p.is_file():
                st = int(m.group(1))
                if st not in found or is_compressed(
                    p
                ):  # both forms of a step: the compressed one is the later save
                    found[st] = p
        return sorted(found.items())

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
        for name in (f"step_{step:010d}.safetensors{ZST_SUFFIX}", f"step_{step:010d}.safetensors"):
            path = self.checkpoint_dir / name
            if path.is_file():
                return path
        kept = [st for st, _ in self.checkpoints()]
        raise FileNotFoundError(f"{self.path}: no checkpoint at step {step} (kept: {kept})")

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

    def metrics_rows_from(self, step: int) -> list[dict[str, Any]]:
        """The rows of ``metrics.jsonl`` with ``step >= step``, read from the end (``read_jsonl_from``)."""
        return read_jsonl_from(self.path / METRICS_FILE, step)

    def social_records_from(self, step: int) -> list[dict[str, Any]]:
        """The records of the social log with ``step >= step``, read from the end (``read_jsonl_from``)."""
        return read_jsonl_from(self.social_path, step)

    def reports(self) -> list[dict[str, Any]]:
        """The report records of ``reports.jsonl`` (``[]`` if there is none), in the file's order: by step,
        then chapter start. Lines that are not report records (``put_report`` keeps them) are not in it."""
        return [r for r in self.read_jsonl(REPORTS_FILE) if _is_report(r)]

    def put_report(self, rec: Mapping[str, Any]) -> None:
        """Put the report record of a chapter into ``reports.jsonl``, replacing the one of the same
        ``(chapter_start, step)``: an atomic rewrite, the records sorted by ``(step, chapter_start)`` and any
        line that is not a report record after them (what ``world_report build`` does, so the live file and an
        offline build are the same file). Called by one thread at a time (the report job)."""
        old = self.read_jsonl(REPORTS_FILE)
        keyed = {(r["chapter_start"], r["step"]): r for r in old if _is_report(r)}
        other = [r for r in old if not _is_report(r)]
        keyed.update({(rec["chapter_start"], rec["step"]): dict(rec)})
        ordered = [keyed[k] for k in sorted(keyed, key=lambda k: (k[1], k[0]))]
        text = "".join(json.dumps(r, allow_nan=False, ensure_ascii=False) + "\n" for r in [*ordered, *other])
        _atomic_write_text(self.path / REPORTS_FILE, text)

    def append_social(self, records: Iterable[Mapping[str, Any]]) -> None:
        """Append snapshot records to ``observers/social.jsonl`` (key ``step`` first) and fsync.

        An empty iterable is a no-op that creates neither ``observers/`` nor the file. Otherwise every record
        needs an int ``step`` and the steps must strictly increase within the call (``ValueError``).
        """
        recs = list(records)
        if not recs:
            return
        last: int | None = None
        for rec in recs:
            step = rec.get("step") if isinstance(rec, Mapping) else None
            if not isinstance(step, int | np.integer) or isinstance(step, bool):
                raise ValueError("a social record needs an int 'step' key")
            step = int(step)
            if last is not None and step <= last:
                raise ValueError(f"social record steps must increase within a call ({step} after {last})")
            last = step
        self.observers_dir.mkdir(exist_ok=True)
        _append_jsonl(
            self.social_path,
            [{"step": r["step"], **{k: v for k, v in r.items() if k != "step"}} for r in recs],
        )

    def social_records(self) -> list[dict[str, Any]]:
        """All records of the social log, in step order; ``[]`` if there is none. A torn last line (a crash
        mid-append) is ignored; a corrupt line elsewhere raises ``json.JSONDecodeError``."""
        return read_jsonl(self.social_path)

    def _parse_social(self) -> list[dict[str, Any]] | None:
        """The records of the social log, parsed whole (None if the file is absent) with the rule of
        ``read_jsonl``: a torn last line is dropped and a corrupt line elsewhere raises
        ``json.JSONDecodeError``; a complete line that is not an object with an int ``step`` raises
        ``ValueError`` naming the line. Only ``truncate_after`` needs the stricter second rule: it must not
        guess the place of a record."""
        if not self.social_path.is_file():
            return None
        records = read_jsonl(self.social_path)
        for i, rec in enumerate(records, start=1):
            step = rec.get("step") if isinstance(rec, dict) else None
            if not isinstance(step, int) or isinstance(step, bool):
                raise ValueError(f"{self.social_path}: record {i} is not an object with an int 'step'")
        return records

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
        are cut the same way (``director.jsonl`` and ``journal.jsonl`` to records with step ≤ ``step``, except
        that a record of the player channel at ``step`` itself is dropped, and ``commands.jsonl`` to its lines
        with ``t < step``: rule P, see ``_director_cut``; a ``director_archive.json`` taken after it is
        deleted), except with ``keep_director`` (a replay, which must never touch the log it replays, nor the
        player's inputs).

        The social log ``observers/social.jsonl`` is cut to its records with step ≤ ``step``, always, whatever
        ``keep_director`` is (it belongs to the trajectory, like the sidecars, and a replay regenerates it).
        Every log that is cut is read and parsed whole before anything changes, so a corrupt line before the
        last (``json.JSONDecodeError``) or a social record without an int ``step`` (``ValueError``) fails the
        call with the directory untouched. The kept records are re-serialised as the metrics rows are, which
        gives back the bytes the runner wrote, and a log is rewritten atomically even when no record is
        removed, because that also drops a torn last line (left in place, it would have the resumed run's
        first record glued onto it). An absent social log stays absent.

        The storm log ``observers/storms.jsonl`` and the tracker states ``observers/storms_<t>.json`` of a
        played weather world (§33.3) are cut the same way: the log to its lines with step ≤ ``step`` (parsed
        whole first, rewritten atomically, absent stays absent), the states after ``step`` deleted with the
        sidecars; the kin thread states ``observers/kinthreads_<t>.json`` (G3b) go with them.
        The alert logs ``alerts.jsonl`` and ``alerts.replay.jsonl`` (G3b, DESIGN §34.3) are cut to their lines
        with step ≤ ``step``, always, whatever ``keep_director`` is: they belong to the trajectory, and a
        replay regenerates the lines after the checkpoint. So is ``reports.jsonl`` (§34.4): its records of the
        chapters that ended after ``step`` go.

        **Order, and a stop in the middle.** The later checkpoints and sidecars go first (the newest
        checkpoint, which ``LATEST`` may name, last) and ``LATEST`` is rewritten; only then are the logs and
        the event chunks cut. A stop (a crash, a kill) before the first deletion leaves the directory as it
        was; after the checkpoint at ``step`` is the newest, a plain ``resume`` takes it and calls this again,
        and every cut is idempotent, so the logs end up cut exactly as an uninterrupted call would have cut
        them. In the other order a stop after a log was cut left ``LATEST`` on a later checkpoint, and a
        plain ``resume`` would continue from it with the records of the steps between lost for good.
        """
        social = self._parse_social()  # first: a log that cannot be parsed fails before any change
        storms = self._parse_storms()
        alerts = {name: self._parse_alerts(name) for name in (ALERTS_FILE, ALERTS_REPLAY_FILE)}
        reports = self.read_jsonl(REPORTS_FILE) if (self.path / REPORTS_FILE).exists() else None
        metrics = None
        if (self.path / METRICS_FILE).exists():
            metrics = "".join(_dumps(r) + "\n" for r in self.metrics_rows() if r["step"] <= step)
        director = [] if keep_director else self._director_cut(step)
        # the commit: from here the newest checkpoint is the one at (or before) ``step``
        for st, path in self._checkpoint_files():  # ascending: the newest, which LATEST may name, goes last
            if st > step:
                path.unlink(missing_ok=True)
        for st, path in self.observer_files():  # whatever keep_director is: a sidecar belongs to its step
            if st > step:
                path.unlink(missing_ok=True)
        for name in PLAY_STATE_NAMES:  # and so do the storm tracker's state and the kin threads
            for st, path in self.play_state_files(name):
                if st > step:
                    path.unlink(missing_ok=True)
        latest = self.latest_checkpoint()
        if latest is not None:
            _atomic_write_text(self.checkpoint_dir / LATEST_FILE, latest.name + "\n")
        else:
            (self.checkpoint_dir / LATEST_FILE).unlink(missing_ok=True)
        for kind in DUMP_KINDS:  # the dumps of the steps after it are regenerated by the resumed run
            for st, path in self.dump_files(kind):
                if st > step:
                    path.unlink(missing_ok=True)
        # the cuts, each of which gives the same files again if it is repeated
        if metrics is not None:
            _atomic_write_text(self.path / METRICS_FILE, metrics)
        for path, text in director:
            if text is None:
                path.unlink(missing_ok=True)
            else:
                _atomic_write_text(path, text)
        if social is not None:
            kept_social = [r for r in social if r["step"] <= step]
            _atomic_write_text(self.social_path, "".join(_dumps(r) + "\n" for r in kept_social))
        if storms is not None:
            kept_storms = [r for r in storms if r["step"] <= step]
            _atomic_write_text(self.storms_path, "".join(_dumps(r) + "\n" for r in kept_storms))
        for name, lines in alerts.items():  # the alert logs: trajectory-bound, cut whatever keep_director is
            if lines is not None:
                kept = [r for r in lines if r["step"] <= step]
                _atomic_write_text(self.path / name, "".join(_dumps(r) + "\n" for r in kept))
        if reports is not None:  # a chapter that ended after ``step`` has a report the replay regenerates
            kept_reports = [r for r in reports if not _is_report(r) or r["step"] <= step]
            _atomic_write_text(
                self.path / REPORTS_FILE,
                "".join(json.dumps(r, allow_nan=False, ensure_ascii=False) + "\n" for r in kept_reports),
            )
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

    def _director_cut(self, step: int) -> list[tuple[Path, str | None]]:
        """What ``truncate_after`` does to the director files, read now and applied later: ``(path, new
        text)``, or ``(path, None)`` for the archive taken after ``step``, which is deleted.

        **Rule P** (G3b spec §6.1.2): a boundary entry at ``step`` is applied before the checkpoint at
        ``step`` and so is in it, but a record of the player channel at ``step`` is applied *after* it (at the
        top of the iteration that starts there) and is not. So an entry is kept if ``step_applied <= step``,
        and a record of the player channel only if ``step_applied < step``; likewise a line of
        ``commands.jsonl`` (the player's inputs, whose ``t`` is the step they were handled at) only if
        ``t < step``."""
        cuts: list[tuple[Path, str | None]] = []
        for name in (DIRECTOR_LOG_FILE, JOURNAL_FILE):
            if (self.path / name).exists():
                kept = [
                    r
                    for r in self.read_jsonl(name)
                    if (at := _record_step(r)) is None or at < step + (0 if _is_player_record(r) else 1)
                ]
                cuts.append((self.path / name, "".join(_dumps(r) + "\n" for r in kept)))
        if (self.path / COMMANDS_FILE).exists():
            kept = [
                r
                for r in self.read_jsonl(COMMANDS_FILE)
                if not isinstance(at := r.get("t"), int) or at < step
            ]
            cuts.append((self.path / COMMANDS_FILE, "".join(_dumps(r) + "\n" for r in kept)))
        archive = self.path / DIRECTOR_ARCHIVE_FILE
        if archive.exists():
            try:
                as_of = json.loads(archive.read_text(encoding="utf-8")).get("as_of_step")
            except (OSError, json.JSONDecodeError):
                as_of = None
            if not isinstance(as_of, int) or as_of > step:  # rewritten at the next checkpoint
                cuts.append((archive, None))
        return cuts

    def dump_diag(self, s: State, step: int) -> Path:
        """Write ``diag/step_<step>.safetensors`` (a full state file) after an invariant violation."""
        self.diag_dir.mkdir(parents=True, exist_ok=True)
        path = self.diag_dir / f"step_{step}.safetensors"
        write_state_file(path, s, self.config())
        return path
