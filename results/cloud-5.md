PR4 Linux-CPU: identical, slow suite: same; test_golden and test_golden_3d both run and pass; stream fixtures pass; stream and director smokes OK

# cloud-5: 3D PR 4 (`3d/pr4-metrics-stream`) on Linux CPU

| item | value |
|---|---|
| commit tested | `24f4526481e8e51cc5e4e0fd77aa1586f4cd33ff` (origin/3d/pr4-metrics-stream, detached, clean) |
| tools | `claude/cloud-tools` at `150fc5f` (digest_compare treats new presets as INFO) |
| baseline | cloud-1, main `afe52f5` |
| `uname -m` | x86_64 (Linux, cloud container) |
| Python / mlx | 3.12.3 / 0.32.3; CPU device |
| env | `uv sync --locked --group dev` (exit 0), no extras |

## Tests

| step | command | result | wall |
|---|---|---|---|
| fast suite | `uv run pytest -m "not slow" -q -rs` | **1417 passed, 2 skipped**, 23 deselected, 0 failed | 27 m 31 s |
| focus rerun, verbose | `uv run pytest tests/test_stream.py tests/test_determinism.py::test_golden tests/test_determinism_3d.py::test_golden_3d -v -rs` | **92 passed**, 0 skipped | 15 s |
| slow suite | `uv run pytest -m slow -q -rs --durations=25` | **23 passed**, 0 failed | 34 m 4 s |
| digest compare | `uv run python /tmp/tools/digest_compare.py --baseline <cloud-1 main digests> --out results/cloud-5-pr4-digests.json` | **exit 0: "OK: bit-identical to the 2D baseline (CPU device)"** | 6 m 10 s |

**Golden digests.** Both run and pass on this platform:
- `tests/test_determinism.py::test_golden` (2D);
- `tests/test_determinism_3d.py::test_golden_3d`, which is now pinned.

**Skips.** Only the usual two: no `ALIFE_LLM_URL`, and no pygame.

**`tests/test_stream.py`.** All tests pass, including the committed-fixture checks:
- `test_committed_fixtures_decode_as_their_sidecars_say[2d|3d]`;
- `test_fixtures_match_the_code[2d|3d]`, the byte comparison against the code;
- `test_fixture_configs_are_the_test_micro_configs`;
- `test_stdout_holds_only_protocol_bytes`;
- `test_streaming_does_not_change_the_trajectory[2d|3d]`.

**Slow suite.** All 19 tests from main are present and pass, plus the 4 `test_crowd_3d` tests. No regression.
Main's two long tests took 1019 s and 836 s (main: 1036 s and 831 s).

**Digest compare.** Exit 0. The new presets and the new keys and `Dims` fields at their defaults appear as
INFO lines only.

## Stream smoke (CPU)

**Commands:**
- `uv run alife run --preset tiny3d --steps 600 --out <scratch>/t4`: exit 0, 56 s.
- `uv run alife stream <scratch>/t4 --every 50 --steps 600 --no-control > s.bin`: exit 0, 35 s, 287,268 bytes,
  stderr empty.

**Deviation from the task.** I added `--steps 600`. With `--no-control` and no `--steps`, the stream only stops
when its reader closes, so redirecting to a file would never end. `--help` confirms this: "default: run until
the reader closes or stdin ends".

**Decoding.** `tests/stream_decode.py` has no CLI, so I used its API: `stream_decode.decode(bytes) ->
(header, frames)`.

**Header:**
```json
{"B": 4, "FG": 32, "FZ": 6, "L": 128.0, "N": 1024, "NCH": 3, "color": "tag", "color_modes": ["tag", "energy", "clade"], "config_hash": "90e799beeba91e8cb0e2e021ad9595340efa9975eb53f963047c63646e61797a", "dims": 3, "every": 50, "field_cell": 4.0, "field_z": 0, "fields": false, "height": 24.0, "mode": "simulate", "protocol": 1, "radius": 1.0, "run": "<scratch>/t4", "source": "run", "step": 600}
```

**Frames.** 13 frames: the start state at step 600, then one every 50 steps up to 1200. In every frame the
header `n_alive` equals the alive-mask sum, flags is 0 and the colour index is 0 (tag).

| step | n_alive | n_bonds | mean z | fraction z ≤ 1.2 |
|---|---|---|---|---|
| 600 | 238 | 71 | 3.71 | 0.58 |
| 650 | 234 | 66 | 3.56 | 0.61 |
| 700 | 232 | 64 | 3.42 | 0.64 |
| 750 | 233 | 70 | 3.33 | 0.66 |
| 800 | 234 | 67 | 3.33 | 0.66 |
| 850 | 230 | 63 | 3.30 | 0.69 |
| 900 | 228 | 64 | 3.25 | 0.67 |
| 950 | 225 | 59 | 3.25 | 0.68 |
| 1000 | 221 | 59 | 3.21 | 0.69 |
| 1050 | 222 | 63 | 3.23 | 0.69 |
| 1100 | 225 | 67 | 3.31 | 0.70 |
| 1150 | 225 | 65 | 3.29 | 0.71 |
| 1200 | 228 | 65 | 3.19 | 0.70 |

**Run directory untouched.** The `t4` file listing (32 files, with path, size and mtime to ns) is identical
before and after streaming; it is in `cloud-5.log`.

**Consistency.** The streamed frames agree with the independent director run below. That run uses the same
preset and seed, and its director made no interventions. For example, at step 600 both have n = 238, and
grounded fraction is 0.576 there vs 0.58 here.

## Director smoke in 3D (CPU)

**Commands:**
- `uv run alife run --preset tiny3d --steps 2000 --director live --persona gardener --out <scratch>/t5`:
  exit 0, 3 m 10 s, about 16 steps/s.
- `uv run alife inspect <scratch>/t5 --check`: **full_check OK**, ledger OK, diag 0, 241 alive, generation max
  6, 143 clades, 53 bodies (largest 5).

**director.jsonl.** It contains only the header line, so there were **no director entries** (no rule, LLM or
revert actions):
```
{"kind": "header", "step": 0, "rules_version": 1, "persona": "gardener", "goal": "", "rule_every": 1000, "budget_per_100k": 2, "seed": 0, "dims": 3}
```

- With `rule_every = 1000`, the rule tier evaluated only at steps 1000 and 2000, and the Gardener found no
  trigger.
- `env` stayed at its defaults on every metrics row.
- `director_archive.json` has 2 entries (at steps 1000 and 2000) and coverage 1.
- To exercise interventions in a short 3D run, a lower `rule_every` or a scripted director would be needed.

**n/K_est every 200 steps.** `K_est` is the `inspect` carrying-capacity estimate, **424**; there is no
`carrying_capacity_est` key in the metrics rows. Births and deaths are for the 10-step metrics row at that step.

| step | n_alive | n/K_est | grounded_frac | surface_frac | height_mean | speed_mean | births | starved | killed | ledger_error |
|---|---|---|---|---|---|---|---|---|---|---|
| 200 | 296.8 | 0.700 | 0.283 | 0.000 | 6.73 | 0.0203 | 0 | 8 | 0 | 0 |
| 400 | 252.6 | 0.596 | 0.443 | 0.004 | 4.72 | 0.0129 | 2 | 1 | 0 | 0 |
| 600 | 238.0 | 0.561 | 0.576 | 0.008 | 3.71 | 0.0096 | 0 | 1 | 0 | 0 |
| 800 | 234.5 | 0.553 | 0.651 | 0.009 | 3.33 | 0.0077 | 1 | 2 | 0 | 0 |
| 1000 | 222.7 | 0.525 | 0.667 | 0.014 | 3.21 | 0.0065 | 0 | 3 | 0 | 0 |
| 1200 | 228.3 | 0.538 | 0.696 | 0.018 | 3.19 | 0.0061 | 1 | 2 | 0 | 0 |
| 1400 | 237.7 | 0.561 | 0.706 | 0.021 | 3.06 | 0.0066 | 1 | 0 | 0 | 0 |
| 1600 | 237.8 | 0.561 | 0.741 | 0.017 | 3.01 | 0.0055 | 1 | 0 | 0 | 0 |
| 1800 | 241.3 | 0.569 | 0.734 | 0.021 | 2.84 | 0.0059 | 3 | 2 | 0 | 0 |
| 2000 | 242.3 | 0.571 | 0.766 | 0.017 | 2.87 | 0.0053 | 1 | 2 | 0 | 0 |

**Ledger.** `ledger_error` is 0 on every row.

**Other 3D metrics in the final row:**
- `band_frac` [0.884, 0.046, 0.021, 0.050];
- `clade_height_mi` 0.466;
- `buoy_cap` q10/q50/q90: 0.470 / 0.500 / 0.521;
- `z_floor_clamps` 1751, `z_surface_clamps` 17 (per 10-step row);
- `body_z_extent_mean` 0.58, max 3.59.

**Note for the coordinator.** On this branch `inspect` estimates K at 424 for tiny3d; the PR 3 build (cloud-4)
gave 1544 for the same preset. The estimator presumably became 3D-aware in PR 4. Worth confirming that this is
intended, since the director's vital signs use n/K.

**Files.**
- `results/cloud-5-pr4-digests.json`.
- `results/cloud-5.log`: the focus, fast, slow, compare and smoke summaries, the inspect output and the t4
  listing.
