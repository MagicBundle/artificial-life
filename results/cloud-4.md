PR3 Linux-CPU: 3D golden digest recorded (PARTIAL REPORT; fast suite, slow suite, digest compare and ecology look still running)

# cloud-4: 3D PR 3 (`3d/pr3-fields`) on Linux CPU

| item | value |
|---|---|
| commit tested | `3bbcab0f36d93b2159046387d04f9dcd277f82c4` (origin/3d/pr3-fields, detached, clean) |
| `uname -m` | x86_64 (Linux, cloud container) |
| Python / mlx | 3.12.3 / 0.32.3; CPU device |
| env | `uv sync --locked --group dev` (exit 0), no extras |

## 3D golden digest (blocking)

**Digest:** `ffb171300f5377b0397b24c2783d177365ff4006c30c7c291ac8aeb92148b3a1`

**Command:** `uv run pytest tests/test_determinism_3d.py -rs -v`. Result: 4 passed, 1 skipped, in 35.8 s.

| test | result |
|---|---|
| `test_same_seed_is_identical` | PASSED |
| `test_different_seeds_differ` | PASSED |
| `test_runtime_checks_only_touch_diag` | PASSED |
| `test_eager_step_matches_compiled` | PASSED |
| `test_golden_3d` | SKIPPED (digest not pinned yet) |

**Skip message, verbatim:**

```
SKIPPED [1] tests/test_determinism_3d.py:105: 3D golden digest not pinned yet: record ffb171300f5377b0397b24c2783d177365ff4006c30c7c291ac8aeb92148b3a1 in GOLDEN_3D (DESIGN §27.8)
```

**Stability check.** A second, separate process
(`uv run pytest tests/test_determinism_3d.py::test_golden_3d -rs -q -p no:cacheprovider`) printed the identical
digest.
