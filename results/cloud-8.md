cloud-8 (PARTIAL: step 2 done): **test_ecology_3d PASSES at C = 48.** The 6,000-step DEV3D run, the fast suite and the digest compare are still running.

# cloud-8: 3D PR 6 recheck (`3d/pr6-64k3d` at `a3d46eb`) on Linux CPU

| item | value |
|---|---|
| commit tested | `a3d46ebf389aee7ab1db4aa3e34cde79516e9a98` (origin/3d/pr6-64k3d, detached, clean) |
| `uname -m` | x86_64, 4 cores |
| Python / mlx | 3.12.3 / 0.32.3; CPU device |
| env | `uv sync --locked --group dev` (exit 0) |

## Step 2 (key): `tests/test_ecology_3d.py` at C = 48

**Command:** `uv run pytest tests/test_ecology_3d.py -m slow -v -s -rA --durations=0`

**Result:** **PASSED** in 3706 s (1 h 1 m 46 s). That is slower than the expected 50–55 min because it shared the
4-core CPU with the long DEV3D run in step 3.

```
tests/test_ecology_3d.py::test_dev3d_ecology_is_alive_and_unsaturated PASSED
3706.08s call     tests/test_ecology_3d.py::test_dev3d_ecology_is_alive_and_unsaturated
======================== 1 passed in 3706.32s (1:01:46) ========================
```

The test prints no diagnostics. The per-window numbers for the same trajectory come from the step 3 run below,
whose first 2,500 steps are the test's run.
