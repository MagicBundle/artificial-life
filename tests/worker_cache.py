"""One private MLX CPU kernel cache per test process (pytest-xdist worker or serial session; DESIGN §22.0).

MLX's CPU backend compiles every fused kernel with ``g++`` into ``$TMPDIR/mlx/<version>/cpu/lib<name>.so``
and loads an existing library by name. Several pytest processes sharing one ``TMPDIR`` therefore race on the
same file (two compiles of one name: a linker "File exists" error, or a half-written library loaded; 5 of 6
trials of four processes failed that way in the probe of DESIGN §22.0), and the name alone can stand for two
different kernels (DESIGN F21 item 6). Under ``pytest -n`` every worker would hit that.

MLX reads ``TMPDIR`` at **every** kernel compile (``std::filesystem::temp_directory_path()``; probed, DESIGN
§22.0), not once at import, so a process can repoint it at any time before its next compile; the xdist
controller and ``-p cpu_device`` importing MLX first do not matter. ``isolate`` does so for the life of the
process: a fresh private directory, seeded with hard links to the complete libraries already in the shared
cache (the developer's, or the one CI restores), so that a warm cache still avoids the ``g++`` runs.
``close`` hands the libraries the process compiled back to the shared cache, one atomic hard link each,
source first, so that CI's cache step and the next run find them, and restores ``TMPDIR``. Child processes
of a test (the CLI tests) inherit the private ``TMPDIR`` while the session runs.

Kernels whose names are ambiguous (two constant lists whose printed forms concatenate alike share a name;
F21 item 6) are **not** filtered here: they pass through as the shared cache always passed them. The branch
``fix/mlx-compile-collision`` (``tests/jit_cache.py``) filters them (``ambiguous``) in an ``isolate`` of the
same shape (it returns an object with ``close()``); the two combine by letting that ``isolate`` replace this
one and keeping ``is_collector`` and ``isolate_for`` below, which decide where it runs.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import pytest

# A library younger than this may still be being written by a process that shares the directory (MLX's
# compiler writes it in place); an older one is complete.
MIN_AGE_S = 60.0

# A private cache directory older than this belongs to a run that was killed (SIGTERM, SIGKILL, a power cut)
# before ``Session.close``; ``isolate`` removes it. No pytest run lasts a day.
STALE_S = 24 * 3600.0

PREFIX = "alife-jit-"

# The default cap on the buffers MLX keeps after freeing them (``limit_mlx_buffer_cache``).
CACHE_MB = 256


def temp_root() -> Path:
    """The directory MLX's CPU backend caches under: libc++'s ``temp_directory_path()``."""
    for var in ("TMPDIR", "TMP", "TEMP", "TEMPDIR"):
        if os.environ.get(var):
            return Path(os.environ[var])
    return Path("/tmp")


def kernel_dir(root: Path) -> Path:
    return root / "mlx" / mx.__version__ / "cpu"


def worker_id() -> str | None:
    """The xdist worker id (``gw0``, ...) in a worker process, else ``None``."""
    return os.environ.get("PYTEST_XDIST_WORKER")


def is_collector(config: pytest.Config) -> bool:
    """True in the xdist controller (``-n N`` with N > 0, or ``--tx``): it runs no test and compiles nothing.

    ``--dist loadfile`` alone sets ``dist`` but names no worker: xdist then runs the tests in this very
    process (its own ``_is_distribution_mode`` also asks for ``tx``), so that process is a serial session
    and compiles.
    """
    opt = config.option
    return worker_id() is None and getattr(opt, "dist", "no") != "no" and bool(getattr(opt, "tx", None))


def _link(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except FileExistsError:
        pass
    except OSError:  # another file system
        shutil.copy2(src, dst)


def _pairs(d: Path) -> list[tuple[Path, Path]]:
    """``(source, library)`` of every kernel in ``d``: ``lib<file>.so`` and its ``<file>.cpp``."""
    return [(d / f"{so.name[3:-3]}.cpp", so) for so in d.glob("lib*.so")]


def sweep_stale(parent: Path, max_age_s: float = STALE_S) -> int:
    """Remove the private cache directories (``alife-jit-*``, which only ``isolate`` creates) of ``parent``
    that are older than ``max_age_s``: a run that is killed never reaches ``Session.close``, and each leaves a
    farm of about 2,200 hard links. Returns the number removed."""
    n = 0
    now = time.time()
    for d in parent.glob(f"{PREFIX}*"):
        try:
            if d.is_symlink() or not d.is_dir() or now - d.stat().st_mtime < max_age_s:
                continue
        except OSError:
            continue
        shutil.rmtree(d, ignore_errors=True)
        n += 1
    return n


@dataclass
class Session:
    shared: Path
    root: Path  # the private TMPDIR
    old_tmpdir: str | None
    seeded: int

    @property
    def private(self) -> Path:
        return kernel_dir(self.root)

    def close(self) -> int:
        """Restore ``TMPDIR``, hand the new libraries to the shared cache, drop the private one. Returns the
        number handed back."""
        if self.old_tmpdir is None:
            os.environ.pop("TMPDIR", None)
        else:
            os.environ["TMPDIR"] = self.old_tmpdir
        n = 0
        try:
            self.shared.mkdir(parents=True, exist_ok=True)
            for cpp, so in _pairs(self.private):
                if (self.shared / so.name).exists() or not cpp.exists():
                    continue
                _link(cpp, self.shared / cpp.name)  # the source first: a library is never shared without it
                _link(so, self.shared / so.name)
                n += 1
        except OSError:
            pass
        shutil.rmtree(self.root, ignore_errors=True)
        return n


def isolate(parent: Path | None = None, min_age_s: float = MIN_AGE_S) -> Session:
    """Point MLX's CPU kernel cache at a fresh private directory under ``parent`` (default: the shared
    ``TMPDIR``), seeded with the shared cache's complete libraries."""
    tempfile.gettempdir()  # pin Python's own temp dir first: pytest's tmp_path stays where it was
    shared_root = temp_root()
    shared = kernel_dir(shared_root)
    parent = parent or shared_root
    parent.mkdir(parents=True, exist_ok=True)
    sweep_stale(parent)
    private_root = Path(tempfile.mkdtemp(prefix=PREFIX, dir=parent))
    private = kernel_dir(private_root)
    private.mkdir(parents=True)
    seeded = 0
    now = time.time()
    for cpp, so in _pairs(shared) if shared.is_dir() else []:
        try:
            if now - so.stat().st_mtime < min_age_s or not cpp.exists():
                continue
            _link(cpp, private / cpp.name)
            _link(so, private / so.name)
            seeded += 1
        except OSError:
            continue
    old = os.environ.get("TMPDIR")
    os.environ["TMPDIR"] = f"{private_root}{os.sep}"  # read at every MLX kernel compile; children inherit it
    return Session(shared, private_root, old, seeded)


def isolate_for(config: pytest.Config) -> Session | None:
    """The session's isolation, or ``None`` where it does not apply: the xdist controller, and
    ``ALIFE_TEST_JIT_SHARED=1`` (the old behaviour: every process on the shared cache, for measurements).

    A worker's directory goes beside its ``--basetemp`` (``<run basetemp>/popen-gwN``; pytest removes a given
    basetemp that exists, so not inside it), where pytest's retention of old runs also collects it should the
    worker die before ``close``; a serial session's goes under the shared ``TMPDIR``.
    """
    if os.environ.get("ALIFE_TEST_JIT_SHARED") == "1" or is_collector(config):
        return None
    base = getattr(config.option, "basetemp", None)
    parent = Path(base).parent / "jit" if worker_id() and base else None
    return isolate(parent)


def cap_applies() -> bool:
    """Where the buffer cap is applied: on the CPU device and in every xdist worker. A serial run on the GPU
    keeps MLX's own limit: the cap's effect on Metal speed was not measured (the Metal runs are the purity
    gate and the owner's checks, DESIGN §28.8), and its memory gain was measured on the CPU device only."""
    return worker_id() is not None or mx.default_device().type == mx.DeviceType.cpu


def limit_mlx_buffer_cache() -> None:
    """Cap the buffers MLX keeps after freeing them at ``ALIFE_TEST_MLX_CACHE_MB`` MB (default ``CACHE_MB``;
    a negative value leaves MLX's own limit, which is the machine's memory size).

    Without the cap a worker that once ran a large test keeps gigabytes of freed buffers for the rest of the
    run (6.6 GB resident for 90 MB live in a measured worker, 12.6 GB for four workers). With 256 MB the
    same four workers peak at 6.8 GB in the same time (DESIGN §22.0, I58). The cap changes no result: it only
    decides how many freed buffers are kept for reuse. Where MLX has no buffer cache to limit (a build
    without Metal: its allocator may ignore the call) it does nothing.
    """
    mb = int(os.environ.get("ALIFE_TEST_MLX_CACHE_MB", CACHE_MB))
    if mb >= 0:
        mx.set_cache_limit(mb << 20)
