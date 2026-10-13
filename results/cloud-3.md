PR2 Linux-CPU: identical, slow suite: same

# cloud-3: 3D PR 2 (`3d/pr2-geometry`) on Linux CPU

| item | value |
|---|---|
| commit tested | `b3199f3071a824b7efa51bad0d726a67dc0e4c8e` (origin/3d/pr2-geometry, detached, clean) |
| baseline | cloud-1, main `afe52f5` (`results/cloud-1-main-digests.json`) |
| `uname -m` | x86_64 (Linux, cloud container) |
| Python / mlx | 3.12.3 / 0.32.3; CPU device |
| env | `uv sync --locked --group dev` (exit 0), no extras |

## Results

| step | command | result | wall |
|---|---|---|---|
| fast suite | `uv run pytest -m "not slow" -q -rs` | **1157 passed, 2 skipped**, 23 deselected, 0 failed (main: 933 passed) | 24 m 30 s |
| slow suite | `uv run pytest -m slow -q -rs --durations=25` | **23 passed**, 0 failed | 36 m 5 s |
| crowd test, verbose | `uv run pytest tests/test_crowd_3d.py -m slow -s -v -rA` | 4 passed | 46 s |
| digest compare | `uv run python /tmp/tools/digest_compare.py --baseline <cloud-1 main digests> --out results/cloud-3-pr2-digests.json` | **exit 0: "OK: bit-identical to the 2D baseline (CPU device)"** | 3 m 9 s |

**`test_golden`.** It ran and passed. The skips are the same two as on main (no `ALIFE_LLM_URL`, no pygame).
No failures anywhere, including the new 3D test files.

**Slow suite, test by test.**
- The 19 slow tests from main are all present and all pass.
- 4 new slow tests, all passing:
  - `test_crowd_3d.py::test_floor_pile_under_gravity[C16|C64]`;
  - `test_crowd_3d.py::test_converging_sphere[C16|C64]`.
- No regression.
- Wall times match main within about 9%:
  - `test_dev_ecology_is_alive_and_unsaturated`: 1034 s vs 1036 s on main;
  - `test_replay_with_the_real_timed_bounds`: 902 s vs 831 s.

## test_crowd_3d output (sizes the 3D preset)

Printed by the tests (`-s`), verbatim:

```
pile C=16: max occupancy 38 (C = 16; 11 at the start), overflow on 716 steps; contacts max 17 true / 8 listed (K = 8), up to 9 agents with a K list full of contacts; min centre distance 0.000 (2r = 2.0)
pile C=64: max occupancy 28 (C = 64; 11 at the start), overflow on 0 steps; contacts max 10 true / 8 listed (K = 8), up to 43 agents with a K list full of contacts; min centre distance 1.571 (2r = 2.0)
sphere C=16: max occupancy 116 (C = 16; 12 at the start), overflow on 582 steps; contacts max 53 true / 8 listed (K = 8), up to 40 agents with a K list full of contacts; min centre distance 0.000 (2r = 2.0)
sphere C=64: max occupancy 36 (C = 64; 12 at the start), overflow on 0 steps; contacts max 10 true / 8 listed (K = 8), up to 32 agents with a K list full of contacts; min centre distance 1.818 (2r = 2.0)
```

| scenario | C | max cell occupancy | steps with overflow | max true contacts (K = 8 listed) | agents with a full K list | min centre distance (2r = 2) |
|---|---|---|---|---|---|---|
| floor pile, gravity | 16 | 38 | 716 | 17 | 9 | **0.000** |
| floor pile, gravity | 64 | 28 | 0 | 10 | 43 | 1.571 |
| converging sphere | 16 | 116 | 582 | 53 | 40 | **0.000** |
| converging sphere | 64 | 36 | 0 | 10 | 32 | 1.818 |

**Reading.**
- With C = 16, both scenarios overflow hash cells for hundreds of steps. Agents then become invisible to each
  other and collapse onto the same point (min distance 0.000).
- With C = 64 there is no overflow: peak occupancy is 28–36, so C = 32 would leave little headroom.
- Even with C = 64, true contacts reach 10 against K = 8 listed neighbours, and 32–43 agents have a K list
  made entirely of contacts. In 3D, packing allows up to 12 touching neighbours, so K = 8 truncates repulsion
  in dense piles. Repulsion still holds: min distance 1.57–1.82 against 2r = 2.0.
- These are test scenarios, not a tuned preset. Whether K needs raising to about 12 for `64k3d` is the
  coordinator's call.

**Digest compare.**
- Every section is `ok`: config hashes, layout, all 15 trajectory cases and all 3 runner cases.
- The 286 INFO lines are all "added in current", at defaults:
  - the `Dims` fields `z_lo`, `z_hi`, `z_ground`, `nd`, `d3`, `lay`, `fz`, `n_cells`, `field_shape`,
    `g_hash_z`, `cell_hash_z`;
  - the new config keys `world.dims`, `world.height`, `physics.gravity`, `physics.floor_eps`,
    `physics.buoyancy_max`, `mutation.buoy_init`, `mutation.buoy_sigma`, `fields.benthic_lambda`,
    `fields.photic_amp` and `fields.photic_lambda`, in `config.json` and in the checkpoint metadata.
- No changed values.

**Files.**
- `results/cloud-3-pr2-digests.json`.
- `results/cloud-3.log`: the full fast, slow, crowd and compare logs.
