# Bench records

`alife bench --stages` reports (JSON, as written by the command) kept with the repository, so a later change to a
step's cost or memory can be compared with a measured baseline. The analysis of the 3D set is DESIGN §27.13.

## 2026-10-16, M5 Pro (3D world PR 6)

- **Machine:** MacBook Pro, Apple M5 Pro, 48 GB, Metal GPU (`Device(gpu, 0)`), darwin-arm64.
- **Software:** commit `a3d46eb` (branch `3d/pr6-64k3d`), alife 0.1.0, mlx 0.32.3, Python 3.12.11.
- **Command:** `alife bench --preset <preset> --stages`: 20 warm-up steps, then 11 donation-check steps (10 normal and
  one that keeps its input state alive), then 200 timed steps, seed 0, the preset's own
  config (float16 weights, `brain.impl matmul`, `compile` on, `runtime_checks cheap`). The `64k3d` files at C = 32
  and C = 24 differ from the shipped preset only in `sensing.cell_capacity` (an override; each file's `config` and
  `config_hash` record it).
- **Conditions:** the five runs were made one after the other in one GPU phase (`runs/pr6_gpu/gpu_phase.log`, not
  kept), with the GPU otherwise free. One run per file, no repeats: the spread within a run is in the file
  (`ms_per_step_min`, `_median`, `_p90`, `_max`), the spread between runs was not measured. The `stages` times come
  from evaluating each stage on its own, so they sum to `total_ms`, which differs from the fused `ms_per_step` by
  `inflation` (0.99 to 1.14 for the 64k-scale files, 2.4 for `tiny3d`); the steps/s figures are the fused ones.

| file | what | steps/s | ms/step | neighbours (ms, share of staged total) | peak memory, GB (GiB in the file) |
|---|---|---|---|---|---|
| `2026-10-16_m5pro_tiny3d.json` | `tiny3d` (N = 1 024, C = 16) | 411 | 2.43 | 1.16 (20%) | 0.03 (0.03) |
| `2026-10-16_m5pro_64k3d_C48.json` | `64k3d` as shipped, C = 48 (Q = 1 296) | 14.4 | 69.2 | 56.2 (79%) | 3.31 (3.08) |
| `2026-10-16_m5pro_64k3d_C32.json` | `64k3d` at C = 32 (Q = 864) | 22.0 | 45.5 | 31.0 (69%) | 2.49 (2.32) |
| `2026-10-16_m5pro_64k3d_C24.json` | `64k3d` at C = 24 (Q = 648) | 24.9 | 40.2 | 27.7 (65%) | 2.18 (2.03) |
| `2026-10-16_m5pro_64k.json` | 2D `64k` reference (C = 16, Q = 144) | 59.3 | 16.9 | 6.3 (33%) | 1.19 (1.11) |

Three things in the files to read with care:

- **Units.** The `*_gb` fields (`peak_mem_gb`, `state_gb`, `excess_gb`, ...) are GiB (2^30 bytes, `bench.py` `_GB`). The
  table gives the peak in GB (10^9 bytes), the unit of DESIGN §15.5 and of the `peak_mem_gb` in a run's `metrics.jsonl`,
  and the file's value in brackets. The GB figure is the file's value times 1.0737.
- `donation` is `ok: false` at C = 48 and C = 32 (`saved_over_weights` 0.0, the limit is 0.25). At C = 48 the peak
  above the resident baseline is the same for the normal steps and for the step that keeps its input state alive
  (`excess_gb` = `excess_copy_gb` = 2.4459 GiB, 2.63 GB), so the peak is set by the neighbour search in both and the
  check has no information at that size; it was `ok` at C = 24 (0.74), on `tiny3d` and in 2D. It has not been
  investigated further (DESIGN §27.13, what remains).
- `n_alive_start`, `n_alive_end`, births and deaths of the three `64k3d` files are identical (26 178, 15 804, 4 033,
  14 407). C changes a trajectory only where a hash cell overflows, so no cell did in these 231 steps
  (20 warm-up, 11 donation-check and 200 timed).

## 2026-10-20, M5 Pro (the fused Metal neighbour search, Review-log row I48)

- **Machine and software:** as above (M5 Pro, 48 GB, mlx 0.32.3, Python 3.12.11), on the tree of this change (branch
  `perf/fused-knn`, origin/main 1ad339c plus the change). The machine was shared with other, lighter jobs (load average
  about 2.5 to 3, CPU only: the GPU was reserved for these runs); one process at a time.
- **Command:** `alife bench --preset <preset> --stages`, as above, once with `ALIFE_KNN=mlx` (the pure-MLX search, the
  reference) and once with the default (`auto`: the kernel in 3D on Metal). The 2D `64k` kernel file was made with
  `ALIFE_KNN=metal`, because `auto` kept the MLX path in 2D at that time (since 2026-10-21, row I52, `auto` takes the
  kernel in 2D too: see the section below). The `config` block of each file is unchanged; the
  environment block gains `knn_impl` (`"metal"` or `"mlx"`), which says which search the step was traced with. `64k3d`
  at `C` = 48 was run twice per search (`_r2` files, in the order metal, mlx after the first pair) to show the spread.
- **Equal trajectories:** each pair has the same `n_alive_end`, births and deaths (15 804, 4 033, 14 407 for `64k3d`;
  19 038, 4 844, 11 867 for `64k`; 283, 41, 176 for `tiny3d`), as the bitwise equality of the two searches requires.

| file | what | steps/s | ms/step | neighbours (ms, staged) | peak memory, GB (GiB in the file) |
|---|---|---|---|---|---|
| `2026-10-20_m5pro_64k3d_C48_mlx.json` (`_r2`) | `64k3d` C = 48, MLX search | 14.83 (14.77) | 67.44 (67.73) | 54.5 (54.8) | 3.31 (3.085) |
| `2026-10-20_m5pro_64k3d_C48_metal.json` (`_r2`) | `64k3d` C = 48, **fused kernel** | **75.19 (75.06)** | **13.30 (13.32)** | **0.42 (0.42)** | **1.07 (0.993)** |
| `2026-10-20_m5pro_64k3d_C32_metal.json` | `64k3d` C = 32, kernel | 75.16 | 13.31 | 0.43 | 1.07 (0.993) |
| `2026-10-20_m5pro_64k3d_C24_metal.json` | `64k3d` C = 24, kernel | 74.95 | 13.34 | 0.42 | 1.07 (0.993) |
| `2026-10-20_m5pro_64k_mlx.json` | 2D `64k` C = 16, MLX search | 61.34 | 16.30 | 6.21 | 1.19 (1.112) |
| `2026-10-20_m5pro_64k_metal.json` | 2D `64k` C = 16, kernel (`ALIFE_KNN=metal`) | 92.96 | 10.76 | 0.36 | 0.91 (0.848) |
| `2026-10-20_m5pro_tiny3d_metal.json` | `tiny3d` C = 16, kernel | 416.2 | 2.40 | 0.38 | 0.017 (0.016) |

The MLX-search comparison at `C` = 32 and 24 and for `tiny3d` uses the 2026-10-16 files above (21.96, 24.89 and 411.1
steps/s; 45.54, 40.18 and 2.43 ms), which were not re-measured: a re-measurement made in the probe (44.40 and 40.01 ms,
2.55 ms) is not kept as a file and is not used. The two MLX-path files that were re-measured and kept (`64k_mlx`,
`64k3d_C48_mlx`) are within 5% of their 2026-10-16 counterparts (61.34 against 59.30 steps/s; 14.83 against 14.45).

Read with care:

- The kernel step is 5.07 times faster at `C` = 48 (67.4 to 13.3 ms) and the time no longer depends on `C` (13.30, 13.31
  and 13.34 ms at `C` = 48, 32 and 24): the kernel's work follows the occupied candidates, not `Q`, so the
  capacity that the floor pile needs costs nothing here (the cost of a cell over `C` is still the candidates lost, §4). In 2D, where the MLX search was 6.2 ms of 18.7, the
  gain is 1.52 times (16.3 to 10.76 ms).
- "Neighbours" is the whole staged search stage (cell hash, sort, `searchsorted`, kernel and dispatch), not the kernel
  alone; the stages sum to more than the fused step (`inflation` 1.11 to 1.19 for the 64k and 64k3d kernel files,
  2.1 for `tiny3d` where 0.38 ms of search is in a 2.4 ms step, against 1.02 to 1.15 for the MLX-path files), because each staged function is evaluated on its own.
- The `donation` check is informative with the kernel: at `C` = 48 the normal steps peak 0.335 GiB above the resident
  baseline and the step that keeps its input state alive 0.924 GiB (`saved_over_weights` 0.97 of the weight bytes, limit
  0.25), `ok: true`, "weights updated in place". With the MLX search both excesses were 2.446 GiB, the search's own peak,
  and the check could not say (the note above).
- `n_alive_end`, births and deaths equal those of the 2026-10-16 files for `64k3d`, `64k` and `tiny3d` (same seed, same
  config hash), so the search did not change the trajectories (§4, F21).

## 2026-10-21, M5 Pro (the kernel is the default in 2D, Review-log row I52)

- **Software and tree:** mlx 0.32.3, Python 3.12.11, branch `perf/knn-2d-default` (origin/main b905e34 plus the change).
- **Command:** `alife bench --preset 64k --stages`, once with the default (`auto`, which now selects the kernel in 2D:
  `knn_impl` is `"metal"` in the file) and once with `ALIFE_KNN=mlx`. One run per file, one process at a time.
- **Conditions:** the machine was shared with other sessions' CPU jobs (load average 3 to 6), so both files are
  slower than the 2026-10-20 ones (about 8% for the kernel, 6% for the MLX path); the ratio is what carries over.

| file | what | steps/s | ms/step | neighbours (ms, staged) | peak memory, GB (GiB in the file) |
|---|---|---|---|---|---|
| `2026-10-21_m5pro_64k_mlx.json` | 2D `64k` C = 16, `ALIFE_KNN=mlx` | 57.64 | 17.35 | 6.49 | 1.19 (1.112) |
| `2026-10-21_m5pro_64k_default.json` | 2D `64k` C = 16, default (kernel) | **86.00** | **11.63** | **0.48** | **0.91 (0.848)** |

The default step is 1.49 times faster (1.52 times on 2026-10-20). Both files have the same config hash, `n_alive_end`
(19 038), births (4 844) and deaths (11 867), as the bitwise equality requires, the donation check `ok` and clean
invariants. The staged total is 16.7 ms against an 11.63 ms fused step in the kernel file (`inflation` 1.43, against
1.15 for the MLX file), which is the same effect as the `64k3d` kernel files (the stages are timed on their own).

### Probe figures not reproducible from the repo

The kernel was developed against a probe whose scripts and output are not in the tree. These figures, quoted in
DESIGN §4 and Review-log row I48, are labelled accordingly there: the threadgroup width sweep (32 to 256 within
±3%), the K sweep behind `K_MAX` (search alone, kernel against MLX path: K = 64 11.9 against 59 ms; K = 128 43
against 65 ms; K = 192 88 against 78 ms, so a crossover near K = 170), the worst case (every scanned cell near `C`:
30 to 140 times faster than the MLX path), the 2 500 random geometries and 169 adversarial scenes, and the 73
steps/s of a real `alife run` with metrics and checkpoints. What the repo does reproduce is
`tests/test_spatial_metal.py` and `alife bench` with `ALIFE_KNN=mlx` against the default.
