"""The per-process MLX CPU kernel cache of the parallel test runs (tests/worker_cache.py; DESIGN §22.0, I58).

CPU only, no socket. The two tests that compile run Python children with their own ``TMPDIR``, so that they
neither touch the session's kernel cache nor depend on it.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest

import worker_cache

CHILD_ENV_DROP = ("PYTEST_XDIST_WORKER", "PYTEST_XDIST_WORKER_COUNT", "ALIFE_TEST_JIT_SHARED")


def child_env(tmpdir: Path) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in CHILD_ENV_DROP}
    env["TMPDIR"] = f"{tmpdir}{os.sep}"
    return env


def libs(root: Path) -> list[Path]:
    return sorted((root / "mlx" / mx.__version__ / "cpu").glob("lib*.so"))


def run_child(code: str, tmpdir: Path, *args: str, timeout: float = 240) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code, *args],
        env=child_env(tmpdir),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


# A fused kernel per call: the constants differ, so the generated source and the library name do.
COMPILE = """
import sys
import mlx.core as mx
mx.set_default_device(mx.cpu)
for k in sys.argv[1:]:
    f = mx.compile(lambda x, k=float(k): mx.exp(x * (k + 1.25)) + mx.sin(x) * (k + 3.5) + mx.cos(x))
    mx.eval(f(mx.arange(10.0)))
"""


def test_mlx_reads_tmpdir_at_every_compile(tmp_path: Path) -> None:
    """The assumption the whole module rests on: the CPU backend takes ``$TMPDIR`` at each compile, not once
    at import or first use, so a process can repoint its kernel cache after MLX is loaded and after it has
    compiled something (the xdist workers do both)."""
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    code = """
import os, sys
import mlx.core as mx
mx.set_default_device(mx.cpu)
a, b = sys.argv[1:3]
def go(k):
    f = mx.compile(lambda x: mx.exp(x * (k + 1.25)) + mx.sin(x) * (k + 3.5) + mx.cos(x))
    mx.eval(f(mx.arange(10.0)))
os.environ["TMPDIR"] = a + os.sep
go(1.0)
os.environ["TMPDIR"] = b + os.sep   # after the import and after a first compile
go(2.0)
"""
    r = run_child(code, tmp_path, str(a), str(b))
    assert r.returncode == 0, r.stderr[-2000:]
    assert len(libs(a)) == 1 and len(libs(b)) == 1 and libs(a) != libs(b)


def test_concurrent_processes_with_private_caches_do_not_collide(tmp_path: Path) -> None:
    """Four processes compile the same kernels at the same time, each in a ``worker_cache.isolate`` of one
    shared directory: all succeed, and what they hand back is complete (a fifth process loads it without
    compiling). Without the isolation, processes sharing one directory fail with a linker "File exists" or a
    missing library (5 of 6 trials in the probe of I58)."""
    shared = tmp_path / "shared"
    shared.mkdir()
    code = (
        "import sys\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "import worker_cache\n"
        "ses = worker_cache.isolate()\n"
        + COMPILE.replace("for k in sys.argv[1:]", "for k in sys.argv[2:]")
        + "ses.close()\n"
    )
    here = str(Path(__file__).parent)
    ks = [str(k) for k in range(5)]
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", code, here, *ks],
            env=child_env(shared),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for _ in range(4)
    ]
    outs = [(p.communicate(timeout=480)[0], p.returncode) for p in procs]
    assert all(rc == 0 for _, rc in outs), [o[-1500:] for o, rc in outs if rc != 0]
    assert len(libs(shared)) == 5
    assert not [p for p in shared.iterdir() if p.name.startswith("alife-jit-")]  # every private cache dropped
    before = {p: p.stat().st_mtime_ns for p in libs(shared)}
    r = run_child(COMPILE, shared, *ks)  # on the shared cache directly: every library must already be there
    assert r.returncode == 0, r.stderr[-2000:]
    assert {p: p.stat().st_mtime_ns for p in libs(shared)} == before


def kernel_source(name: str, consts: tuple[str, ...] = ()) -> str:
    """The shape of MLX's generated CPU source, enough for ``kernel_ambiguity``: the kernel function (its name
    is ``tape_kinds_dtypes_hash``, a ``C`` in the kinds per constant) and one ``int32_t`` declaration each."""
    kinds = "C" * len(consts) or "V"
    decls = "".join(f"  int32_t tmp_{chr(65 + i)} = {c};\n" for i, c in enumerate(consts))
    head = f"void Tape{name}_{kinds}_i4_1234_contiguous(int* shape, int64_t** strides, void** args)"
    return f"{head} {{\n{decls}}}\n"


def fake_kernel(d: Path, name: str, age_s: float, consts: tuple[str, ...] = ()) -> None:
    d.mkdir(parents=True, exist_ok=True)
    t = time.time() - age_s
    (d / f"{name}.cpp").write_text(kernel_source(name, consts))
    (d / f"lib{name}.so").write_text(name)
    for f in (d / f"{name}.cpp", d / f"lib{name}.so"):
        os.utime(f, (t, t))


AMBIGUOUS = ("12", "1")  # also (1, 21): the "121" of F21 item 6


def test_isolate_seeds_complete_libraries_and_hands_new_ones_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    shared = worker_cache.kernel_dir(tmp_path)
    fake_kernel(shared, "old", age_s=3600)
    fake_kernel(shared, "young", age_s=1)  # may still be being written by another process
    (shared / "libnosource.so").write_text("x")  # a library without its source is not shared on
    os.utime(shared / "libnosource.so", (1, 1))
    ses = worker_cache.isolate()
    try:
        private = ses.private
        assert ses.seeded == 1
        assert sorted(p.name for p in private.iterdir()) == ["libold.so", "old.cpp"]
        assert os.environ["TMPDIR"] == f"{ses.root}{os.sep}" and ses.root.parent == tmp_path
        assert os.stat(private / "libold.so").st_ino == os.stat(shared / "libold.so").st_ino  # a link
        fake_kernel(private, "new", age_s=0)
    finally:
        handed = ses.close()
    assert handed == 1
    assert os.environ["TMPDIR"] == str(tmp_path)  # restored
    assert not ses.root.exists()
    assert (shared / "libnew.so").read_text() == "new" and (shared / "new.cpp").exists()
    assert (shared / "libold.so").read_text() == "old"  # untouched


def test_isolate_does_not_seed_an_ambiguous_kernel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The shared cache may hold a library compiled by another configuration under a name that two constant
    lists share (F21 item 6); a process must compile its own, so it is not linked in. A source that cannot be
    read, and one that is no kernel, count as ambiguous."""
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    shared = worker_cache.kernel_dir(tmp_path)
    fake_kernel(shared, "plain", age_s=3600)
    fake_kernel(shared, "plain2", age_s=3600, consts=("7",))  # one constant pins itself
    fake_kernel(shared, "pinned", age_s=3600, consts=("0", "48"))  # "048" is no int32 print
    fake_kernel(shared, "amb", age_s=3600, consts=AMBIGUOUS)
    fake_kernel(shared, "binary", age_s=3600)
    (shared / "binary.cpp").write_bytes(b"\xff\xfe\x00")
    fake_kernel(shared, "nokernel", age_s=3600)
    (shared / "nokernel.cpp").write_text("int x;")
    ses = worker_cache.isolate()
    try:
        assert ses.seeded == 3
        assert sorted(p.name for p in ses.private.glob("lib*.so")) == [
            "libpinned.so",
            "libplain.so",
            "libplain2.so",
        ]
        assert not (ses.private / "amb.cpp").exists()
    finally:
        ses.close()
    assert (shared / "libamb.so").exists()  # the shared cache itself is untouched


def test_close_does_not_hand_back_an_ambiguous_kernel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    ses = worker_cache.isolate()
    try:
        fake_kernel(ses.private, "fresh", age_s=0)
        fake_kernel(ses.private, "freshamb", age_s=0, consts=AMBIGUOUS)
        fake_kernel(ses.private, "freshbin", age_s=0)
        (ses.private / "freshbin.cpp").write_bytes(b"\xff\xfe\x00")
    finally:
        handed = ses.close()
    shared = worker_cache.kernel_dir(tmp_path)
    assert handed == 1
    assert sorted(p.name for p in shared.iterdir()) == ["fresh.cpp", "libfresh.so"]


def test_isolate_in_a_given_parent_and_without_a_shared_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TMPDIR", str(tmp_path / "t"))
    (tmp_path / "t").mkdir()
    ses = worker_cache.isolate(tmp_path / "base" / "jit")
    try:
        assert ses.seeded == 0 and ses.root.parent == tmp_path / "base" / "jit"
    finally:
        ses.close()
    assert not ses.root.exists()


def test_isolate_sweeps_the_directories_of_runs_that_were_killed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A killed run never reaches ``close``; its private cache stays in ``TMPDIR`` until the next ``isolate``
    finds it older than a day. Younger ones (a run that is going on), other names and symlinks stay."""
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    old = tmp_path / "alife-jit-old"
    fake_kernel(worker_cache.kernel_dir(old), "k", age_s=0)
    young = tmp_path / "alife-jit-young"
    young.mkdir()
    other = tmp_path / "other-dir"
    other.mkdir()
    link = tmp_path / "alife-jit-link"
    link.symlink_to(other, target_is_directory=True)
    t = time.time() - 2 * 24 * 3600
    for d in (old, other):
        os.utime(d, (t, t))
    ses = worker_cache.isolate()
    try:
        assert not old.exists()
        assert young.exists() and other.exists() and link.is_symlink() and ses.root.exists()
    finally:
        ses.close()
    assert worker_cache.sweep_stale(tmp_path, max_age_s=0.0) == 1  # `young`, now "older" than zero seconds
    assert not young.exists() and other.exists()


def cfg(dist: str, tx: tuple[str, ...] | None = None) -> pytest.Config:
    """The options xdist leaves: ``-n N`` is ``dist`` ("load", ...) with N ``tx`` entries; ``--dist X`` alone
    is ``dist`` with none (and runs in this process)."""
    tx = tx if tx is not None else (("popen",) if dist != "no" else ())
    return SimpleNamespace(option=SimpleNamespace(dist=dist, tx=list(tx), basetemp=None))  # type: ignore[return-value]


def test_the_xdist_controller_does_not_isolate_but_workers_and_serial_runs_do(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)
    monkeypatch.delenv("ALIFE_TEST_JIT_SHARED", raising=False)
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    assert worker_cache.is_collector(cfg("load")) and worker_cache.is_collector(cfg("loadfile"))
    assert worker_cache.isolate_for(cfg("loadfile")) is None
    assert not worker_cache.is_collector(cfg("no"))  # serial, or -n 0
    assert not worker_cache.is_collector(cfg("loadfile", tx=()))  # --dist alone: xdist runs in this process
    alone = worker_cache.isolate_for(cfg("loadfile", tx=()))
    assert alone is not None
    alone.close()
    monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw3")  # a worker reports dist "no" to its own pytest
    assert not worker_cache.is_collector(cfg("no"))
    ses = worker_cache.isolate_for(cfg("no"))
    assert ses is not None
    ses.close()
    monkeypatch.setenv("ALIFE_TEST_JIT_SHARED", "1")  # the opt-out, for measurements
    assert worker_cache.isolate_for(cfg("no")) is None
    assert os.environ["TMPDIR"] == str(tmp_path)


def test_a_workers_directory_goes_beside_its_basetemp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Not inside it: pytest deletes a given basetemp that exists when its temp-path factory first starts."""
    monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw0")
    monkeypatch.delenv("ALIFE_TEST_JIT_SHARED", raising=False)
    monkeypatch.setenv("TMPDIR", str(tmp_path / "t"))
    (tmp_path / "t").mkdir()
    c = SimpleNamespace(option=SimpleNamespace(dist="no", basetemp=str(tmp_path / "run" / "popen-gw0")))
    ses = worker_cache.isolate_for(c)  # type: ignore[arg-type]
    assert ses is not None
    try:
        assert ses.root.parent == tmp_path / "run" / "jit"
    finally:
        ses.close()


def test_the_buffer_cache_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    """The cap is read from the environment and applied through ``mx.set_cache_limit``. Only the Metal
    allocator is known to keep and return the limit (``set_cache_limit`` returns the one it replaces); a
    build without Metal (the Linux CPU wheel of CI) may ignore the call, so there it is only called."""
    prev = mx.set_cache_limit(0)  # returns the limit it replaces
    try:
        monkeypatch.delenv("ALIFE_TEST_MLX_CACHE_MB", raising=False)
        worker_cache.limit_mlx_buffer_cache()
        monkeypatch.setenv("ALIFE_TEST_MLX_CACHE_MB", "17")
        worker_cache.limit_mlx_buffer_cache()
        monkeypatch.setenv("ALIFE_TEST_MLX_CACHE_MB", "-1")  # leave MLX's own limit alone
        worker_cache.limit_mlx_buffer_cache()
        if not mx.metal.is_available():
            return
        monkeypatch.delenv("ALIFE_TEST_MLX_CACHE_MB")
        worker_cache.limit_mlx_buffer_cache()
        assert mx.set_cache_limit(0) == worker_cache.CACHE_MB << 20
        monkeypatch.setenv("ALIFE_TEST_MLX_CACHE_MB", "17")
        worker_cache.limit_mlx_buffer_cache()
        assert mx.set_cache_limit(5 << 20) == 17 << 20
        monkeypatch.setenv("ALIFE_TEST_MLX_CACHE_MB", "-1")
        worker_cache.limit_mlx_buffer_cache()
        assert mx.set_cache_limit(0) == 5 << 20
    finally:
        mx.set_cache_limit(prev)


def test_the_cap_applies_to_workers_and_the_cpu_device_only(monkeypatch: pytest.MonkeyPatch) -> None:
    prev = mx.default_device()
    try:
        monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)
        mx.set_default_device(mx.cpu)
        assert worker_cache.cap_applies()
        if mx.metal.is_available():
            mx.set_default_device(mx.gpu)
            assert not worker_cache.cap_applies()  # a serial run on the GPU keeps MLX's own limit
            monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw0")
            assert worker_cache.cap_applies()
    finally:
        mx.set_default_device(prev)
