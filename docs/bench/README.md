# Bench records

`alife bench --stages` reports (JSON, as written by the command) kept with the repository, so a later change to a
step's cost or memory can be compared with a measured baseline. The analysis of the 3D set is DESIGN §27.13.

## 2026-10-16, M5 Pro (3D world PR 6)

- **Machine:** MacBook Pro, Apple M5 Pro, 48 GB, Metal GPU (`Device(gpu, 0)`), darwin-arm64.
- **Software:** commit `a3d46eb` (branch `3d/pr6-64k3d`), alife 0.1.0, mlx 0.32.3, Python 3.12.11.
- **Command:** `alife bench --preset <preset> --stages`: 200 timed steps after 20 warm-up steps, seed 0, the preset's own
  config (float16 weights, `brain.impl matmul`, `compile` on, `runtime_checks cheap`). The `64k3d` files at C = 32
  and C = 24 differ from the shipped preset only in `sensing.cell_capacity` (an override; each file's `config` and
  `config_hash` record it).
- **Conditions:** the five runs were made one after the other in one GPU phase (`runs/pr6_gpu/gpu_phase.log`, not
  kept), with the GPU otherwise free. One run per file, no repeats: the spread within a run is in the file
  (`ms_per_step_min`, `_median`, `_p90`, `_max`), the spread between runs was not measured. The `stages` times come
  from evaluating each stage on its own, so they sum to `total_ms`, which differs from the fused `ms_per_step` by
  `inflation` (0.99 to 1.14 for the 64k-scale files, 2.4 for `tiny3d`); the steps/s figures are the fused ones.

| file | what | steps/s | ms/step | neighbours (ms, share of staged total) | peak memory |
|---|---|---|---|---|---|
| `2026-10-16_m5pro_tiny3d.json` | `tiny3d` (N = 1 024, C = 16) | 411 | 2.43 | 1.16 (20%) | 0.03 GB |
| `2026-10-16_m5pro_64k3d_C48.json` | `64k3d` as shipped, C = 48 (Q = 1 296) | 14.4 | 69.2 | 56.2 (79%) | 3.08 GB |
| `2026-10-16_m5pro_64k3d_C32.json` | `64k3d` at C = 32 (Q = 864) | 22.0 | 45.5 | 31.0 (69%) | 2.32 GB |
| `2026-10-16_m5pro_64k3d_C24.json` | `64k3d` at C = 24 (Q = 648) | 24.9 | 40.2 | 27.7 (65%) | 2.03 GB |
| `2026-10-16_m5pro_64k.json` | 2D `64k` reference (C = 16, Q = 144) | 59.3 | 16.9 | 6.3 (33%) | 1.11 GB |

Two things in the files to read with care:

- `donation` is `ok: false` at C = 48 and C = 32 (`saved_over_weights` 0.0, the limit is 0.25). At C = 48 the peak
  with the input state kept alive and the peak of the normal steps are the same number (2.4459 GB), so the peak is
  set by the neighbour search in both and the check has no information at that size; it was `ok` at C = 24 (0.74),
  on `tiny3d` and in 2D. It has not been investigated further (DESIGN §27.13, what remains).
- `n_alive_start`, `n_alive_end`, births and deaths of the three `64k3d` files are identical (26 178, 15 804, 4 033,
  14 407). C changes a trajectory only where a hash cell overflows, so no cell did in these 220 steps.
