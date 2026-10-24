"""MLX's compile-cache constant collision (DESIGN F21 item 6) and the test session's guard (``worker_cache``,
which filters with ``kernel_ambiguity``; ported from ``tests/test_mlx_compile_collision.py`` of the branch
``fix/mlx-compile-collision``).

``mx.compile`` names a fused kernel after its op tape and ``std::hash`` of its scalar constants printed back
to back, so ``floor(p / c) mod g`` (the hash-cell line of ``find_neighbors``) with ``(c, g) = (12.0, 1)``
and with ``(1.0, 21)`` (both "121") get one kernel. Every MLX case runs in a subprocess on the CPU device
with its own ``TMPDIR``, so neither this process's in-memory kernel cache nor a shared on-disk one ever holds
the colliding kernel.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import mlx.core as mx
import pytest

import kernel_ambiguity
import worker_cache

# MLX versions on which (12.0, 1) and (1.0, 21) share a kernel. A version outside the list must not collide,
# so a bump fails here until F21 item 6 and the guard are re-checked (as ``knn_metal.VERIFIED_MLX`` is).
COLLIDING_MLX = ("0.32.3",)
COLLIDES = mx.__version__ in COLLIDING_MLX

# argv[1]: JSON ``[[mode, c, g], ...]`` compiled in this order in one process, ``c`` and ``g`` closed over as
# constants ("const") or passed as array inputs ("input"); prints whether each compiled result equals the
# eager one. The compiled functions are kept alive, so MLX never sees a recycled function id.
SCRIPT = textwrap.dedent(
    """
    import json, sys
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    p = mx.array([[0.5, 3.25], [7.75, 12.5], [13.0, 20.75], [1.0, 2.0]])

    def closed(c, g):
        return lambda q: mx.floor(q / c).astype(mx.int32) % g

    def given(q, c, g):
        return mx.floor(q / c).astype(mx.int32) % g

    keep, out = [], []
    for mode, c, g in json.loads(sys.argv[1]):
        f, args = (closed(c, g), (p,)) if mode == "const" else (given, (p, mx.array(c), mx.array(g)))
        cf = mx.compile(f)
        keep.append(cf)
        out.append(bool(mx.array_equal(cf(*args), f(*args)).item()))
    print(json.dumps(out))
    """
)
CELL = 128 / 21  # tiny's hash cell (L = 128, G = 21): prints 6.09523821, which no other (float, int) prints


def _run(tmpdir: Path | str, cases: list[list[Any]]) -> list[bool]:
    Path(tmpdir).mkdir(parents=True, exist_ok=True)  # MLX needs TMPDIR to exist
    env = {**os.environ, "TMPDIR": f"{str(tmpdir).rstrip(os.sep)}{os.sep}"}
    argv = [sys.executable, "-c", SCRIPT, json.dumps(cases)]
    done = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def _kernels(root: Path) -> dict[tuple[str, ...], Path]:
    """The cached kernel sources under ``root``, keyed by their constant literals."""
    out = {}
    for cpp in worker_cache.kernel_dir(root).glob("*.cpp"):
        out[tuple(text for _, text in kernel_ambiguity.declarations(cpp.read_text()) or [])] = cpp
    return out


def test_constants_that_print_alike_share_a_kernel(tmp_path: Path) -> None:
    """In one process the function compiled second runs the first one's kernel; passing the scalars as array
    inputs (they are then not constants) keeps both right."""
    got = _run(tmp_path, [["const", 12.0, 1], ["const", 1.0, 21], ["input", 12.0, 1], ["input", 1.0, 21]])
    assert got == [True, not COLLIDES, True, True]


def test_the_cpu_kernel_cache_carries_the_collision_across_processes(tmp_path: Path) -> None:
    """MLX's CPU backend reuses ``$TMPDIR/mlx/<version>/cpu/lib<name>.so`` from any earlier process, so one
    process per configuration is not enough on the CPU: the cache directory must not be shared either."""
    shared, fresh = tmp_path / "shared", tmp_path / "fresh"
    assert _run(shared, [["const", 1.0, 21]]) == [True]
    assert _run(shared, [["const", 12.0, 1]]) == [not COLLIDES]  # a new process, the same cache directory
    assert _run(fresh, [["const", 12.0, 1]]) == [True]


@pytest.mark.parametrize(
    ("decls", "want"),
    [
        ([("int32_t", "12"), ("int32_t", "1")], True),  # also (1, 21)
        ([("int32_t", "-1"), ("int32_t", "24")], True),  # also (-12, 4)
        ([("int32_t", "0"), ("int32_t", "48")], False),  # "04" is no int32 print
        ([("float", "12"), ("int32_t", "1")], True),  # (12.0, 1) and (1.0, 21)
        ([("float", "2"), ("int32_t", "13")], True),  # (21.0, 3)
        ([("float", "6.09523821"), ("int32_t", "21")], False),  # "6.0952382" is no float32 print
        ([("float", "0.5"), ("float", "0.25")], False),
        ([("bool", "true"), ("int32_t", "1")], False),
        ([("uint8_t", "30"), ("uint8_t", "12")], False),  # "301" is out of uint8 range
        ([("bfloat16_t", "0.5"), ("int32_t", "1")], True),  # bfloat16 prints are not modelled: conservative
        ([("int32_t", "7")], False),  # one constant pins itself
    ],
)
def test_constants_ambiguous(decls: list[tuple[str, str]], want: bool) -> None:
    assert kernel_ambiguity.constants_ambiguous(decls) is want


def test_ambiguous_reads_mlx_kernel_sources(tmp_path: Path) -> None:
    """On real MLX output (this platform's source format, preamble included): the (12.0, 1) kernel is
    ambiguous, tiny's hash-cell kernel and a kernel without constants are not, and unreadable text is."""
    assert _run(tmp_path, [["const", 12.0, 1], ["const", CELL, 21], ["input", 12.0, 1]]) == [True] * 3
    k = _kernels(tmp_path)
    assert set(k) == {("12", "1"), ("6.09523821", "21"), ()}
    assert {c: kernel_ambiguity.ambiguous(p.read_text()) for c, p in k.items()} == {
        ("12", "1"): True,
        ("6.09523821", "21"): False,
        (): False,
    }
    assert kernel_ambiguity.ambiguous("not a kernel")


def test_session_shares_only_unambiguous_kernels(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A guarded session sees only the shared cache's unambiguous kernels, so its (1.0, 21) is right although
    the shared cache holds (12.0, 1) under that name, and it hands back only its unambiguous new kernels."""
    shared = tmp_path / "shared"
    assert _run(shared, [["const", 12.0, 1], ["const", CELL, 21]]) == [True, True]
    monkeypatch.setenv("TMPDIR", f"{shared}{os.sep}")
    s = worker_cache.isolate(min_age_s=0.0)
    try:
        assert s.shared == worker_cache.kernel_dir(shared) and s.seeded == 1
        assert os.environ["TMPDIR"] != f"{shared}{os.sep}" and s.private.is_dir()
        assert set(_kernels(s.root)) == {("6.09523821", "21")}
        # A child of the session inherits its TMPDIR. (3.5, 7) prints "3.57", which nothing else prints.
        assert _run(os.environ["TMPDIR"], [["const", 1.0, 21], ["const", 3.5, 7]]) == [True, True]
    finally:
        handed_back = s.close()
    assert handed_back == 1 and os.environ["TMPDIR"] == f"{shared}{os.sep}" and not s.root.exists()
    assert set(_kernels(shared)) == {("12", "1"), ("6.09523821", "21"), ("3.5", "7")}
    assert _run(shared, [["const", 1.0, 21]]) == [not COLLIDES]  # unguarded, the shared cache still collides


@pytest.mark.slow
def test_probe_finds_no_collision_in_the_test_configurations(tmp_path: Path) -> None:
    """``compile_probe`` on the conftest configurations and the director and observer variants: no kernel
    name holds two sources across them, and the compiled step's integers equal the eager ones (2 min each)."""
    only = ["tiny_f32", "micro", "tiny3d_f32", "micro3d", "tiny_director", "tiny_observe"]
    probe = Path(__file__).with_name("compile_probe.py")
    argv = [sys.executable, str(probe), str(tmp_path), "--steps", "2", "--only", *only]
    done = subprocess.run(argv, capture_output=True, text=True, timeout=3600)
    assert done.returncode == 0, done.stdout + done.stderr
