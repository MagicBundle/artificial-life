cloud-8 (PARTIAL: steps 2, 4 and 5 done): **test_ecology_3d PASSES at C = 48.** Fast suite green (1467 passed, 2 expected xfails), both goldens pass, digest compare bit-identical. The 6,000-step DEV3D run is at step 3,000 with no overflow so far, but peak occupancy is rising (36 in steps 2,501–3,000).

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


## Step 4: fast suite

**Command:** `uv run pytest -m "not slow" -q -rsx`, under `nice -n 5` and in parallel with the step 3 run.

**Result:** **1467 passed, 2 skipped, 2 xfailed**, 24 deselected, 0 failed, in 27 m 1 s.

- **Skips:** the usual two.
  - `test_director_llm.py:994`: no `ALIFE_LLM_URL`.
  - `test_viewer.py:237`: no pygame.
- **xfails:** exactly the two expected strict ones, both
  `tests/test_director_3d.py::test_a_blocked_note_is_journalled_once_across_a_resume`, with
  `[tiny_cfg]` and `[tiny3d_cfg]`.
- **Goldens:** rerun verbosely, both pass.
  ```
  tests/test_determinism.py::test_golden PASSED
  tests/test_determinism_3d.py::test_golden_3d PASSED
  ```

## Step 5: digest compare against the cloud-1 main baseline

**Result:** exit **0**: "OK: bit-identical to the 2D baseline (CPU device)". It took 2 m 59 s.

- 314 INFO lines, all additions at defaults.
- No DIFF lines.
