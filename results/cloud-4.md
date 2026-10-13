PR3 Linux-CPU: 2D identical (only new 3D preset hashes added), slow suite: same; 3D golden ffb171300f5377b0397b24c2783d177365ff4006c30c7c291ac8aeb92148b3a1; tiny3d 3000 steps ledger exact, full_check OK

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

## Fast suite

**Command:** `uv run pytest -m "not slow" -q -rs`

**Result:** **1278 passed, 3 skipped**, 23 deselected, 0 failed, in 25 m 14 s (main: 933 passed).

**`test_golden` (2D).** It ran and passed: it is not among the skips.

**Skips:**
- `test_determinism_3d.py:105` (`test_golden_3d`): the 3D digest is not pinned yet. Expected; it prints the
  same digest as above.
- `test_director_llm.py:994`: no `ALIFE_LLM_URL`, as on main.
- `test_viewer.py:237`: no pygame, as on main.

## Slow suite

**Command:** `uv run pytest -m slow -q -rs --durations=30`

**Result:** **23 passed**, 0 failed, in 35 m 42 s.

- The 19 slow tests from main are all present and all pass.
- The other 4 are the `test_crowd_3d` tests from PR 2, also passing.
- No regression.
- Main's two long tests: 1045 s and 862 s (main: 1036 s and 831 s).

## Digest compare against the cloud-1 main baseline

**Command:**
`uv run python /tmp/tools/digest_compare.py --baseline <cloud-1 main digests> --out results/cloud-4-pr3-digests.json`

**Result:** exit **1**, with 2 differences. Both are **additions, not changes**:

```
config_hashes                        2 difference(s)
     DIFF config_hashes.tiny3d: added in current = 90e799beeba91e8cb0e2e021ad9595340efa9975eb53f963047c63646e61797a
     DIFF config_hashes.tiny3d+fast: added in current = fd56d2121a6f2bb680734dbdb134c03c73a2d10c350b27f1bbd024b8c966d1d2
```

The tool counts new presets in its `config_hashes` table as DIFF, not INFO. Every 2D preset hash, all
15 trajectory cases and all 3 runner cases are `ok` (20 sections ok). The 2D trajectories are therefore
bit-identical to main. The INFO lines are additions at defaults, as in cloud-3. The coordinator may want
`digest_compare` to treat newly added presets as INFO.

## First 3D ecology look

**Commands:**
- `uv run alife run --preset tiny3d --steps 3000 --out <scratch>/t3d`: exit 0, 4 m 38 s.
- `uv run alife inspect <scratch>/t3d --check`: exit 0, **full_check OK**, ledger OK, diag 0.

**Setup:**
- world: dims 3, size 128, height 24, field_cell 4;
- physics: radius 1, gravity 0.03, floor_eps 0.1, buoyancy_max 0.04;
- buoy_init 0.5;
- n_max 1024, n_init 256, k_neighbors 8, f16 weights, seed 0;
- config_hash `90e799be…797a`.

**Metrics every 300 steps.** Counts are summed over the preceding 300 steps; `n_alive` and steps/s come from
the row at that step.

| step | n_alive | births | starved | killed | aged | catastrophe | unlogged | ledger_error | HEAT_BUOYANCY | steps/s |
|---|---|---|---|---|---|---|---|---|---|---|
| 300 | 270.9 | 210 | 186 | 11 | 0 | 0 | 0 | 0 | 393,003 | 16.2 |
| 600 | 238.0 | 30 | 55 | 6 | 0 | 0 | 0 | 0 | 282,032 | 16.7 |
| 900 | 228.0 | 25 | 34 | 1 | 0 | 0 | 0 | 0 | 262,087 | 17.8 |
| 1200 | 228.3 | 39 | 38 | 1 | 0 | 0 | 0 | 0 | 251,877 | 15.7 |
| 1500 | 238.7 | 36 | 24 | 2 | 0 | 0 | 0 | 0 | 261,633 | 15.9 |
| 1800 | 241.3 | 43 | 37 | 2 | 0 | 0 | 0 | 0 | 266,305 | 18.6 |
| 2100 | 247.7 | 45 | 32 | 6 | 0 | 0 | 0 | 0 | 263,271 | 16.9 |
| 2400 | 237.6 | 36 | 44 | 4 | 0 | 0 | 0 | 0 | 264,612 | 15.9 |
| 2700 | 246.9 | 48 | 35 | 3 | 0 | 0 | 0 | 0 | 258,135 | 14.6 |
| 3000 | 254.3 | 52 | 40 | 4 | 0 | 0 | 0 | 0 | 270,171 | 11.1 |

**Ledger and energy:**
- `ledger_error` is 0 on every row (max |error| = 0).
- `HEAT_BUOYANCY` totals 2,773,126 over the run, about 260k per 300 steps after the initial die-off.
- For scale, the last 10-step row has `HEAT_BASAL` 369,872 and `HEAT_BUOYANCY` 9,173, so buoyancy is about 2.5%
  of basal heat.

**Final state (step 3000):**
- 255 alive, generation max 8, 128 clades (top share 9%).
- 64 bodies, largest 6, with 13.7% of agents in bodies of 4 or more.
- `max_occupancy` 9, `cell_overflow` 0.

**Final checkpoint, agent z (alive, n = 255).** The floor is at z = 1 (= r) and the surface at z = 23 (= H − r).

| band | fraction |
|---|---|
| floor, z ≤ 1.2 | **0.800** |
| mid-water | 0.165 |
| surface, z ≥ 22.8 | 0.035 |

- z quantiles: min 1.000, q25 1.000, median 1.000, q75 1.006, max 23.000; mean 2.654.
- Histogram of z in bins of 2 from 0 to 24: `[210, 20, 3, 0, 3, 3, 2, 1, 0, 0, 2, 11]`.
- Mean `buoy_cap` of alive agents is **0.4966** (median 0.5000, q10 0.4682, q90 0.5281). It has barely moved
  from `buoy_init = 0.5` in 3000 steps and 8 generations.

**Throughput.** tiny3d on this CPU runs at a median of **16.9 steps/s** (mean 16.6, range 11.1–20.2, steps
> 100).

**Reading.**
- Life settles on the floor: 80% are grounded, and a small cluster sits at the surface (11 agents at z ≥ 22).
- The population is stable at about 230–255 after an initial die-off from 256.
- Buoyancy costs a small but non-zero share of energy.
- There has been no selection on `buoy_cap` yet over this horizon.

**Files.**
- `results/cloud-4-pr3-digests.json`.
- `results/cloud-4.log`: the golden, fast, slow, compare, inspect and summary output, plus the tail of the run
  log.
