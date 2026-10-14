cloud-7 (PARTIAL: step 2 done; the 3D ecology test, the rest of the slow suite, the digest compare and the 64k3d probe are still running): fast suite green, both goldens pass, exactly the 2 expected strict xfails.

# cloud-7: 3D PR 6 (`3d/pr6-64k3d`) on Linux CPU

| item | value |
|---|---|
| commit tested | `401399fbf099064925653c007232321664805593` (origin/3d/pr6-64k3d, detached, clean) |
| baseline | cloud-1, main `afe52f5`; slow suite also compared against cloud-5 |
| `uname -m` | x86_64, 4 cores |
| Python / mlx | 3.12.3 / 0.32.3; CPU device |
| env | `uv sync --locked --group dev` (exit 0) |

## Step 2: fast suite

**Command:** `uv run pytest -m "not slow" -q -rs -rx`, run under `nice -n 5` in parallel with step 3.

**Result:** **1451 passed, 2 skipped, 2 xfailed**, 24 deselected, 0 failed, in 24 m 28 s.

- **xfails.** Exactly the two expected strict xfails:
  - `tests/test_director_3d.py::test_a_blocked_note_is_journalled_once_across_a_resume[tiny_cfg]`
  - `tests/test_director_3d.py::test_a_blocked_note_is_journalled_once_across_a_resume[tiny3d_cfg]`

  The reason given for both: "Director._blocked is per process and not rebuilt on a live resume (DESIGN §27.10)
  … Remove the marker with the fix."
- **Skips.** The 2 skips were not itemised, because `-rx` replaced `-rs` in the summary. The count matches the
  usual two in cloud-5 (no `ALIFE_LLM_URL`, no pygame).
- **Goldens.** Rerun verbosely with
  `uv run pytest tests/test_determinism.py::test_golden tests/test_determinism_3d.py::test_golden_3d -v -rs`:

  ```
  tests/test_determinism.py::test_golden PASSED
  tests/test_determinism_3d.py::test_golden_3d PASSED
  ```
