PR6 (a3d46eb) Linux-CPU: test_ecology_3d PASSES at C = 48; fast suite green (2 expected xfails); goldens pass; digest compare identical. 6,000-step DEV3D: **0 overflow steps; peak occupancy levelled off at 30–37 (max 37 at step 3,610) against C = 48.** The population is still growing linearly (5,981 at 6,000; n/floor-K 1.29, so the floor-layer K is now below the population).

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


## Step 3: long DEV3D run (paused, incomplete)

**What was run.** The same loop as `tests/test_ecology_3d.run_dev3d(dev3d_cfg(), …)`, in chunks of 1,500 steps.
Each chunk saves the state with `runio.write_state_file` and the next resumes with `runio.load_checkpoint`.
- Every 500 steps the chunks recorded `metrics.carrying_capacity` (K_est), `metrics.floor_carrying_capacity`
  (floor-layer K) and n_alive.
- About 0.66 steps/s on this CPU, with the step 2 test running at the same time.

**Status when stopped.**
- Chunks 0–1500 and 1500–3000 completed; their counters are in `results/cloud-8-dev3d-counters_*.npz`.
- Chunk 3000–4500 was **interrupted** on the owner's pause instruction. Its per-step counters were not written,
  because each chunk saves them at its end.
- Its K_est, floor-K and n_alive records at steps 3,500 and 4,000 were already written and are included. They
  are in `results/cloud-8-dev3d-k_est.json`.
- Steps 4,500–6,000 were not run.

**Per 500-step window.**
- "grounded share" is the window mean of `GROUNDED / N_ALIVE`.
- The K columns are evaluated on the state at the window's end.
- The last two rows have only the K records.

| steps | n_alive (end) | grounded share (mean) | steps with overflow | max cell_overflow | peak occupancy (C = 48) | floor clamps | HEAT_BUOYANCY | K_est (end) | floor-layer K (end) | n / floor K |
|---|---|---|---|---|---|---|---|---|---|---|
| 1–500 | 3664 | 0.086 | 0 | 0 | 12 | 166063 | 9110855 | 13644 | 4208 | 0.87 |
| 501–1000 | 3192 | 0.202 | 0 | 0 | 11 | 309454 | 6400622 | 13259 | 4236 | 0.75 |
| 1001–1500 | 3043 | 0.259 | 0 | 0 | 16 | 363383 | 5810667 | 13012 | 4271 | 0.71 |
| 1501–2000 | 3013 | 0.285 | 0 | 0 | 18 | 390170 | 5631982 | 12763 | 4310 | 0.70 |
| 2001–2500 | 3170 | 0.301 | 0 | 0 | 32 | 415089 | 5669953 | 12812 | 4352 | 0.73 |
| 2501–3000 | 3388 | 0.305 | 0 | 0 | 36 | 446541 | 6000192 | 12717 | 4398 | 0.77 |
| 3001–3500 | 3660 | – | – | – | – | – | – | 12906 | 4446 | 0.82 |
| 3501–4000 | 4030 | – | – | – | – | – | – | 13089 | 4491 | 0.90 |

step 0: K_est 14190, floor-layer K 4185, n 4096
steps 1–3000: overflow on 0 steps; peak occupancy 36 at step 2896


**Reading so far.**
- **No overflow at C = 48 over steps 1–3,000.** Peak occupancy rises steadily: 12 → 11 → 16 → 18 → 32 → 36, with
  36 at step 2,896. This is the same growing floor pile as cloud-7, which reached 31 by step 2,500 at C = 24.
- At this rate, 48 could be reached within a few thousand more steps. That is unverified, because the run was
  paused before it could show whether the pile saturates.
- The first 2,151 steps reproduce cloud-7's C = 24 trajectory exactly, for example n 3,664 / 3,192 / 3,043 at
  500 / 1,000 / 1,500. That is expected while no cell overflows.
- **The floor-layer K tracks the population far better than K_est.** n / floor-K is 0.70–0.90; n / K_est is
  about 0.24–0.31.
- After a low of about 3,010 near step 2,000, the population is growing again: 3,170 → 3,388 → 3,660 → 4,030 by
  step 4,000. Combined with the rising peak occupancy, the floor is getting more crowded, not settling.


## Step 3, resumed (owner instruction, 2026-10-15 20:12)

**Resume.**
- Resumed from the saved 3,000-step state via `runio.load_checkpoint`, with the same chunk script.
- Chunk 3,000–4,500 ran alone on the CPU: 2,776 s, 0.54 steps/s.
- **Determinism check:** the n_alive, K_est and floor-K values it recorded at steps 3,500 and 4,000 are
  identical to the ones written by the interrupted attempt.

**Window table.** Rows 3001–3500 onward replace the K-only rows above.

| steps | n_alive (end) | grounded share (mean) | steps with overflow | max cell_overflow | peak occupancy (C = 48) | floor clamps | HEAT_BUOYANCY | K_est (end) | floor-layer K (end) | n / floor K |
|---|---|---|---|---|---|---|---|---|---|---|
| 1–500 | 3664 | 0.086 | 0 | 0 | 12 | 166063 | 9110855 | 13644 | 4208 | 0.87 |
| 501–1000 | 3192 | 0.202 | 0 | 0 | 11 | 309454 | 6400622 | 13259 | 4236 | 0.75 |
| 1001–1500 | 3043 | 0.259 | 0 | 0 | 16 | 363383 | 5810667 | 13012 | 4271 | 0.71 |
| 1501–2000 | 3013 | 0.285 | 0 | 0 | 18 | 390170 | 5631982 | 12763 | 4310 | 0.70 |
| 2001–2500 | 3170 | 0.301 | 0 | 0 | 32 | 415089 | 5669953 | 12812 | 4352 | 0.73 |
| 2501–3000 | 3388 | 0.305 | 0 | 0 | 36 | 446541 | 6000192 | 12717 | 4398 | 0.77 |
| 3001–3500 | 3660 | 0.298 | 0 | 0 | 32 | 471210 | 6460076 | 12906 | 4446 | 0.82 |
| 3501–4000 | 4030 | 0.282 | 0 | 0 | 37 | 489205 | 7053910 | 13089 | 4491 | 0.90 |
| 4001–4500 | 4465 | 0.271 | 0 | 0 | 32 | 517333 | 7756446 | 13320 | 4535 | 0.98 |

steps 1–4500: overflow on 0 steps; peak occupancy 37 at step 3610
peak occupancy per 100 steps since 2500: [31, 33, 32, 36, 36, 32, 31, 30, 31, 32, 35, 37, 34, 34, 31, 31, 31, 31, 32, 32]


**Reading at 4,500.**
- **Peak occupancy is levelling off, not climbing to 48.** Per-100-step peaks since step 2,500 are
  31, 33, 32, 36, 36, 32, 31, 30, 31, 32, 35, 37, 34, 34, 31, 31, 31, 31, 32, 32. The all-time maximum is 37,
  at step 3,610.
- That leaves a margin of 11 or more against C = 48. **No overflow on any of the 4,500 steps.**
- The population keeps growing: 3,388 → 3,660 → 4,030 → 4,465. n / floor-K reaches 0.98.
- The grounded share eases from 0.305 to 0.271 as the population grows: the extra agents live off the floor.
  The floor pile is not getting denser.


## Step 3 complete: 6,000 steps

**Last chunk.** Chunk 4,500–6,000 ran in 2,434 s, at 0.62 steps/s. All four chunk counter files are in
`results/cloud-8-dev3d-counters_*.npz`, and the K records in `results/cloud-8-dev3d-k_est.json`.

**Full window table.**

| steps | n_alive (end) | grounded share (mean) | steps with overflow | max cell_overflow | peak occupancy (C = 48) | floor clamps | HEAT_BUOYANCY | K_est (end) | floor-layer K (end) | n / floor K |
|---|---|---|---|---|---|---|---|---|---|---|
| 1–500 | 3664 | 0.086 | 0 | 0 | 12 | 166063 | 9110855 | 13644 | 4208 | 0.87 |
| 501–1000 | 3192 | 0.202 | 0 | 0 | 11 | 309454 | 6400622 | 13259 | 4236 | 0.75 |
| 1001–1500 | 3043 | 0.259 | 0 | 0 | 16 | 363383 | 5810667 | 13012 | 4271 | 0.71 |
| 1501–2000 | 3013 | 0.285 | 0 | 0 | 18 | 390170 | 5631982 | 12763 | 4310 | 0.70 |
| 2001–2500 | 3170 | 0.301 | 0 | 0 | 32 | 415089 | 5669953 | 12812 | 4352 | 0.73 |
| 2501–3000 | 3388 | 0.305 | 0 | 0 | 36 | 446541 | 6000192 | 12717 | 4398 | 0.77 |
| 3001–3500 | 3660 | 0.298 | 0 | 0 | 32 | 471210 | 6460076 | 12906 | 4446 | 0.82 |
| 3501–4000 | 4030 | 0.282 | 0 | 0 | 37 | 489205 | 7053910 | 13089 | 4491 | 0.90 |
| 4001–4500 | 4465 | 0.271 | 0 | 0 | 32 | 517333 | 7756446 | 13320 | 4535 | 0.98 |
| 4501–5000 | 4996 | 0.255 | 0 | 0 | 34 | 542557 | 8703274 | 13653 | 4575 | 1.09 |
| 5001–5500 | 5468 | 0.253 | 0 | 0 | 33 | 604368 | 9694612 | 13723 | 4613 | 1.19 |
| 5501–6000 | 5981 | 0.248 | 0 | 0 | 32 | 645317 | 10559988 | 13760 | 4648 | 1.29 |

steps 1–6000: overflow on 0 steps; peak occupancy 37 at step 3610
peak occupancy per 100 steps since 2500: [31, 33, 32, 36, 36, 32, 31, 30, 31, 32, 35, 37, 34, 34, 31, 31, 31, 31, 32, 32, 32, 32, 34, 34, 32, 31, 31, 31, 33, 32, 30, 30, 32, 32, 31]


**Answer to the key question: peak occupancy levels off.**
- After the rise to about 32–36 between steps 2,000 and 3,000, the per-100-step peak stays between 30 and 37
  for the remaining 3,000 steps, with no upward trend. The last 1,500 steps peak at 30–34.
- The all-time maximum is 37, at step 3,610.
- **No step of the 6,000 had a cell over C = 48**, so there was no overflow step to report.
- C = 48 has a margin of 11 or more on this run. C = 40 would also have passed it, but with only 3 to spare.

**Other observations.**
- **The population has not equilibrated.** It grows by about 450–530 every 500 steps from step 2,000 to 6,000:
  3,013 → 5,981.
- The growth is off the floor. The grounded share falls from 0.305 to 0.248, and `HEAT_BUOYANCY` per window
  nearly doubles (5.6M to 10.6M). The water column and the surface band are filling.
- This is why the floor pile stopped densifying while n kept rising.
- **n / floor-K passes 1 at about step 4,600 and reaches 1.29 at step 6,000.** The floor-layer K (4,648) now
  understates the realised population, because the newcomers do not live on the floor.
- K_est (13,760) is still about 2.3× n. Both estimates bracket the population, and the true 3D steady state is
  above 6,000 and not yet reached.
- For the director's extinction-risk test, which compares n against the floor-layer K, this errs on the safe
  side: n/K reads high, not low. Any rule that treats n/K > 1 as overcrowding would fire here, though.
- Longer runs, beyond 6,000 steps, would be needed to see where DEV3D's population and occupancy actually
  settle.
