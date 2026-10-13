# cloud-1: Linux-CPU baselines on main

**Verdict: main `afe52f5` is green on Linux CPU. Fast 933 passed, 2 skipped. `test_golden` ran and passed.
Slow 19/19 passed. The digest matrix completed with 0 case errors.**

| item | value |
|---|---|
| commit tested | `afe52f534aee1a2b54ebf4001129ac3468f6271c` (origin/main, detached) |
| `uname -m` | x86_64 (Linux, cloud container) |
| Python / mlx | 3.12.3 / 0.32.3; default device `Device(cpu, 0)` |
| env | `uv sync --locked --group dev`, no extras (pygame not installed) |

## Results

| step | command | result | wall |
|---|---|---|---|
| fast suite | `uv run pytest -m "not slow" -q -rs` | **933 passed, 2 skipped**, 19 deselected, 0 failed | 25 m 17 s |
| slow suite | `uv run pytest -m slow -q -rs --durations=25` | **19 passed**, 0 failed | 34 m 8 s |
| digest matrix | `uv run python /tmp/tools/digest_matrix.py results/cloud-1-main-digests.json` (from the checkout) | **18 cases, 0 errors**; JSON written | 3 m 7 s |

**Fast-suite skips (both expected):**
- `tests/test_director_llm.py:994`: needs `ALIFE_LLM_URL`.
- `tests/test_viewer.py:237`: pygame is not installed (no `viewer` extra).

**`tests/test_determinism.py::test_golden` ran and passed.** It was collected and is not among the skips.
This is the Linux-x86_64-cpu, mlx 0.32.3 configuration that the golden digest is pinned for.

**Slowest slow tests:**
- `test_ecology.py::test_dev_ecology_is_alive_and_unsaturated`: 1036 s.
- `test_director_replay_slow.py::test_replay_with_the_real_timed_bounds`: 831 s.
- Everything else: under 25 s each.

The 19 slow-test IDs are listed in `cloud-1.log`, for the test-by-test comparison in cloud-2.

**Digest summary.** `diag` is the invariant-violation code; 0 means no violation.

| case | digest (first 12 hex) | n_alive | diag |
|---|---|---|---|
| tiny_f32 | 662b8f111f8a | 286 | 0 |
| tiny_f16 | 739f7f39df38 | 288 | 0 |
| tiny_bf16 | 7bbeb3f249ec | 291 | 0 |
| micro | 381854e13d89 | 14 | 0 |
| tiny_director_inert | 662b8f111f8a (= tiny_f32, as expected) | 286 | 0 |
| tiny_director_levers | c59017809268 | 267 | 0 |
| tiny_imm_floor | 845cb4c7768e | 400 | 0 |
| tiny_imm_every | 600625dd0cc8 | 485 | 0 |
| tiny_seasons_global | e22e1134b60f | 282 | 0 |
| tiny_seasons_per_patch | da795651d3ab | 276 | 0 |
| tiny_fast | 6f9507a41d33 | 287 | 0 |
| dev_20 | 38f1fe7ad714 | 4095 | 0 |
| tiny_f32_eager | e60e3be92a4d | 416 | 0 |
| bench_staged_compiled | c8c644e8bd48 | 426 | 0 |
| bench_staged_eager | ba6560179a4d | 256 | 0 |
| run_tiny_cli (runner) | 082a9054858a | – | – |
| run_tiny_director_scripted (runner) | 25b3674ef5f0 | – | – |
| run_tiny_director_gardener (runner) | c3890f0e0ea0 | – | – |

**Files.**
- `results/cloud-1-main-digests.json`: the baseline for `digest_compare.py`.
- `results/cloud-1.log`: the full logs.
