PR1 Linux-CPU: identical, slow suite: same

# cloud-2: 3D PR 1 (PR #15, `3d/pr1-dims`) on Linux CPU

| item | value |
|---|---|
| commit tested | `c4845abe9baa7fab1878d2ac61665ce9ae859226` (origin/3d/pr1-dims, detached, clean) |
| baseline | cloud-1, main `afe52f5` (`results/cloud-1-main-digests.json`) |
| `uname -m` | x86_64 (Linux, cloud container) |
| Python / mlx | 3.12.3 / 0.32.3; CPU device |
| env | `uv sync --locked --group dev` (exit 0), no extras |

## Results

| step | command | result | wall |
|---|---|---|---|
| fast suite | `uv run pytest -m "not slow" -q -rs` | **1007 passed, 2 skipped**, 19 deselected, 0 failed (main: 933 passed; this branch adds 74 tests) | 24 m 9 s |
| slow suite | `uv run pytest -m slow -q -rs --durations=25` | **19 passed**, 0 failed | 35 m 50 s |
| digest compare | `uv run python /tmp/tools/digest_compare.py --baseline <cloud-1 main digests> --out results/cloud-2-pr1-digests.json` | **exit 0: "OK: bit-identical to the 2D baseline (CPU device)"** | 7 m 3 s |

**`test_golden`.** It ran and passed: it is not among the skips, which are the same two as on main (no
`ALIFE_LLM_URL`, no pygame).

**Slow suite, test by test.** This branch collects the same 19 slow test IDs as main, and all 19 pass on both.
No test passes on main and fails here. Wall times match main within about 5%:
- `test_dev_ecology_is_alive_and_unsaturated`: 1061 s vs 1036 s on main;
- `test_replay_with_the_real_timed_bounds`: 898 s vs 831 s.

**Digest compare.**
- Every section is `ok`: schema, layout, `config_hashes`, all 15 trajectory cases (every-step state digests,
  StepOut, counters and ledger) and all 3 runner cases.
- The 145 INFO lines are all additions, which is expected:
  - `world.dims = 2` and `world.height = 0.0` in `config.json` and in the checkpoint metadata;
  - the new `Dims` fields (`nd`, `d3`, `lay`, `fz`, `n_cells`, `field_shape`, `g_hash_z`, `cell_hash_z`).
- No value differs, and the config hashes are unchanged.

**Files.**
- `results/cloud-2-pr1-digests.json`.
- `results/cloud-2.log`: the full logs, including every INFO line.
