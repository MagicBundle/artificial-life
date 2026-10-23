"""The golden tests run on an empty kernel cache (tests/golden_run.py; DESIGN §22.0, I58).

The golden tests themselves are pinned to Linux, so on the Mac these tests check the mechanism: that a child
started by ``golden_run`` sees no kernel library although the shared cache and this process's private one hold
some, that it compiles into its own directory, and that ``test_golden`` and ``test_golden_3d`` use it.
"""

from __future__ import annotations

import os
from pathlib import Path

import mlx.core as mx
import pytest

import golden_run
import test_determinism
import test_determinism_3d
import worker_cache
from alife.config import Config
from test_worker_cache import fake_kernel


def test_a_cold_child_sees_no_kernel_cache_although_the_shared_and_private_ones_are_warm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tiny_cfg: Config
) -> None:
    """The shared cache holds three old libraries and this process's private cache is seeded with them (as a
    test process's is); the child gets an empty one, compiles into it, and leaves the other two alone."""
    monkeypatch.setenv("TMPDIR", str(tmp_path / "t"))
    shared = worker_cache.kernel_dir(tmp_path / "t")
    for name in ("a", "b", "c"):
        fake_kernel(shared, name, age_s=3600)
    ses = worker_cache.isolate()
    try:
        assert ses.seeded == 3 and len(list(ses.private.glob("lib*.so"))) == 3
        got = golden_run.probe(tiny_cfg, tmp_path / "cold")
        assert got["libs_at_start"] == 0 and got["libs_at_end"] >= 1
        assert got["tmpdir"] == f"{tmp_path / 'cold'}{os.sep}"
        assert len(list(worker_cache.kernel_dir(tmp_path / "cold").glob("lib*.so"))) == got["libs_at_end"]
        assert len(list(ses.private.glob("lib*.so"))) == 3  # nothing leaked into the test process's cache
    finally:
        ses.close()
    assert len(list(shared.glob("lib*.so"))) == 3  # nor into the shared one


def test_cold_env_wants_an_empty_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw1")
    env = golden_run.cold_env(tmp_path / "new")
    assert env["TMPDIR"] == f"{tmp_path / 'new'}{os.sep}" and "PYTEST_XDIST_WORKER" not in env
    (tmp_path / "new" / "x").write_text("")
    with pytest.raises(ValueError, match="not empty"):
        golden_run.cold_env(tmp_path / "new")


@pytest.mark.parametrize(
    ("module", "name", "pin", "kind"),
    [
        (test_determinism, "test_golden", test_determinism.GOLDEN, "2d"),
        (test_determinism_3d, "test_golden_3d", test_determinism_3d.GOLDEN_3D, "3d"),
    ],
)
def test_the_golden_tests_run_their_trajectory_in_a_cold_child(
    module,
    name: str,
    pin: dict,
    kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tiny_cfg: Config,
    tiny3d_cfg: Config,
) -> None:
    """On the pinned platform (faked here) ``test_golden*`` asks ``run_cold`` for its trajectory, with a fresh
    directory under its ``tmp_path``, requires the child to have started on an empty cache, and compares
    the child's digest, not that of the module's in-process ``base_run``."""
    monkeypatch.setattr(module, "_platform", lambda: pin["platform"])
    monkeypatch.setattr(mx, "__version__", pin["mlx"], raising=False)
    calls = []

    def fake(k, cfg, tmpdir, steps=None, timeout=0):
        calls.append((k, tmpdir))
        return {"digest": pin["digest"] or "x", "libs_at_start": fake.libs, "buoyancy": 5}

    fake.libs = 0
    monkeypatch.setattr(golden_run, "run_cold", fake)
    test = getattr(module, name)
    cfg = tiny_cfg if kind == "2d" else tiny3d_cfg
    test(cfg, tmp_path)
    assert calls == [(kind, tmp_path / "cold")]
    fake.libs = 7  # a child that found libraries in its cache
    with pytest.raises(AssertionError):
        test(cfg, tmp_path)
    fake.libs = 0
    monkeypatch.setitem(pin, "digest", "0" * 64)  # a digest that differs
    monkeypatch.setattr(
        golden_run, "run_cold", lambda *a, **k: {"digest": "1" * 64, "libs_at_start": 0, "buoyancy": 5}
    )
    with pytest.raises(AssertionError):
        test(cfg, tmp_path)


@pytest.mark.slow
@pytest.mark.parametrize("kind", ["2d", "3d"])
def test_the_cold_trajectory_runs_end_to_end(
    kind: str, tmp_path: Path, tiny_cfg: Config, tiny3d_cfg: Config
) -> None:
    """One step of the real trajectory in a cold child (a cold compile of the whole step: minutes)."""
    got = golden_run.run_cold(kind, tiny_cfg if kind == "2d" else tiny3d_cfg, tmp_path / "cold", steps=1)
    assert got["libs_at_start"] == 0 and got["libs_at_end"] > 0
    assert len(got["digest"]) == 64 and int(got["digest"], 16) >= 0
