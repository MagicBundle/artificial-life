cloud-6 (PARTIAL: Parts A and B done; Part C running): tiny3d 20k-step baselines. Ledger exact and full_check OK on both seeds. buoy_cap widens but does not shift on seed 0, and drifts down on seed 1. Seed 1 colonises the water column and hits the N cap (992).

# cloud-6: 3D tuning sweeps (CPU, main)

| item | value |
|---|---|
| commit | `e1f718e68867fde71e5e3ae74107540dec4ef177` (origin/main, detached, clean) |
| `uname -m` | x86_64, 4 cores, 15 GB RAM |
| Python / mlx | 3.12.3 / 0.32.3; CPU device |
| env | `uv sync --locked --group dev` (exit 0) |
| runner | each job runs `nice -n 10 uv run alife run ... --overwrite`, then `alife inspect --check`; at most 3 concurrent processes (4 cores) |

**Raw data.** In `results/cloud-6/`:
- `partA_*_metrics.jsonl.gz` and `partB_*_metrics.jsonl.gz`: every 10-step row, gzipped;
- `partA_*_inspect.txt`;
- `partA_summary.json` and `partB_summary.json`.

## Part A: tiny3d baselines, 20,000 steps

**Commands:**
- `alife run --preset tiny3d --steps 20000 --seed {0,1}`;
- `alife inspect --check` on each.

**Wall time:** 28 m 5 s (seed 0) and 28 m 30 s (seed 1), run 2 at a time next to Part C. Median throughput
17.9 and 17.7 steps/s.

**Table conventions.**
- Births and deaths are summed over the preceding 1000 steps.
- The heavy keys (bodies and body z-extent) come from the latest row that has them, at or before the step.
- The overflow column gives the maximum `cell_overflow` in the window, with the number of rows > 0 in brackets.
- The occupancy column is the maximum `max_occupancy` in the window.
- tiny3d settings: C = 16, K = 8, `n_max` 1024 (so `n_cap` = 992), height 24, gravity 0.03,
  `buoyancy_cost` 16, `buoyancy_cap_cost` 8, `max_age` 10,000.

### A_s0

| step | n_alive | births | starved | killed | aged | cat | unlogged | height_mean | grounded | surface | buoy q10/q50/q90 | bodies_largest | frac_bodies≥4 | body_z_ext mean/max | clades | overflow max (rows>0) | occ max | ledger_err | steps/s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1000 | 222.700 | 277 | 294 | 18 | 0 | 0 | 0 | 3.213 | 0.667 | 0.014 | 0.477/0.500/0.519 | 4 | 0.036 | 0.502/2.399 | 159 | 0 (0) | 6 | 0 | 18.720 |
| 2000 | 242.300 | 133 | 105 | 8 | 0 | 0 | 0 | 2.872 | 0.766 | 0.017 | 0.470/0.500/0.521 | 5 | 0.054 | 0.581/3.593 | 143 | 0 (0) | 10 | 0 | 18.351 |
| 3000 | 254.300 | 154 | 126 | 14 | 0 | 0 | 0 | 2.654 | 0.787 | 0.024 | 0.468/0.500/0.528 | 6 | 0.137 | 0.443/2.153 | 128 | 0 (0) | 11 | 0 | 18.235 |
| 4000 | 274.200 | 153 | 127 | 7 | 0 | 0 | 0 | 2.972 | 0.776 | 0.036 | 0.462/0.500/0.532 | 8 | 0.157 | 0.565/4.382 | 123 | 0 (0) | 9 | 0 | 17.460 |
| 5000 | 277.800 | 165 | 148 | 14 | 0 | 0 | 0 | 2.506 | 0.787 | 0.029 | 0.455/0.500/0.541 | 10 | 0.170 | 0.518/6.174 | 116 | 0 (0) | 8 | 0 | 16.557 |
| 6000 | 280.300 | 175 | 158 | 13 | 0 | 0 | 0 | 2.370 | 0.829 | 0.032 | 0.454/0.500/0.547 | 12 | 0.189 | 0.417/2.308 | 112 | 0 (0) | 8 | 0 | 17.789 |
| 7000 | 278.900 | 188 | 167 | 22 | 0 | 0 | 0 | 2.324 | 0.821 | 0.025 | 0.456/0.500/0.556 | 14 | 0.186 | 0.470/5.664 | 105 | 0 (0) | 8 | 0 | 16.528 |
| 8000 | 285.800 | 160 | 126 | 27 | 0 | 0 | 0 | 2.720 | 0.781 | 0.031 | 0.449/0.500/0.560 | 8 | 0.192 | 0.666/5.313 | 105 | 0 (0) | 7 | 0 | 18.509 |
| 9000 | 285.300 | 155 | 136 | 21 | 0 | 0 | 0 | 2.579 | 0.816 | 0.028 | 0.455/0.500/0.554 | 9 | 0.116 | 0.523/4.118 | 102 | 0 (0) | 8 | 0 | 18.588 |
| 10000 | 291.300 | 142 | 115 | 22 | 0 | 0 | 0 | 2.811 | 0.816 | 0.048 | 0.452/0.500/0.552 | 19 | 0.134 | 0.428/4.439 | 97 | 0 (0) | 9 | 0 | 16.052 |
| 11000 | 278.200 | 191 | 115 | 35 | 52 | 0 | 0 | 2.271 | 0.831 | 0.018 | 0.444/0.503/0.565 | 10 | 0.143 | 0.361/3.471 | 64 | 0 (0) | 8 | 0 | 17.614 |
| 12000 | 301.900 | 171 | 121 | 23 | 4 | 0 | 0 | 2.359 | 0.831 | 0.020 | 0.443/0.503/0.564 | 13 | 0.142 | 0.475/4.197 | 62 | 0 (0) | 9 | 0 | 17.609 |
| 13000 | 337.700 | 191 | 131 | 21 | 5 | 0 | 0 | 2.184 | 0.838 | 0.015 | 0.437/0.504/0.578 | 11 | 0.137 | 0.379/3.448 | 59 | 0 (0) | 10 | 0 | 17.950 |
| 14000 | 355.500 | 214 | 164 | 26 | 2 | 0 | 0 | 2.414 | 0.822 | 0.022 | 0.427/0.501/0.583 | 10 | 0.151 | 0.366/2.730 | 59 | 0 (0) | 9 | 0 | 17.804 |
| 15000 | 385.700 | 249 | 188 | 24 | 8 | 0 | 0 | 2.536 | 0.815 | 0.023 | 0.423/0.502/0.584 | 9 | 0.186 | 0.422/4.131 | 56 | 0 (0) | 10 | 0 | 16.454 |
| 16000 | 399.200 | 240 | 203 | 19 | 6 | 0 | 0 | 2.381 | 0.833 | 0.018 | 0.419/0.501/0.580 | 8 | 0.150 | 0.367/3.208 | 55 | 0 (0) | 10 | 0 | 17.462 |
| 17000 | 415.000 | 254 | 211 | 19 | 8 | 0 | 0 | 2.631 | 0.804 | 0.007 | 0.415/0.499/0.586 | 9 | 0.152 | 0.510/3.975 | 52 | 0 (0) | 8 | 0 | 19.319 |
| 18000 | 434.600 | 278 | 231 | 21 | 6 | 0 | 0 | 2.595 | 0.829 | 0.014 | 0.410/0.498/0.586 | 13 | 0.147 | 0.375/3.274 | 52 | 0 (0) | 8 | 0 | 16.749 |
| 19000 | 445.200 | 269 | 227 | 23 | 8 | 0 | 0 | 2.429 | 0.792 | 0.013 | 0.404/0.497/0.581 | 9 | 0.152 | 0.487/7.124 | 48 | 0 (0) | 8 | 0 | 18.686 |
| 20000 | 486.200 | 316 | 249 | 24 | 4 | 0 | 0 | 2.262 | 0.817 | 0.014 | 0.402/0.496/0.585 | 7 | 0.128 | 0.406/3.889 | 46 | 0 (0) | 9 | 0 | 16.379 |

totals: {"rows": 2000, "last_step": 20000, "ledger_err_max": 0, "overflow_rows": 0, "overflow_max": 0, "occ_max": 11, "sps_median": 17.86645718364769, "peak_mem_gb_max": 0.039893182}

### A_s1

| step | n_alive | births | starved | killed | aged | cat | unlogged | height_mean | grounded | surface | buoy q10/q50/q90 | bodies_largest | frac_bodies≥4 | body_z_ext mean/max | clades | overflow max (rows>0) | occ max | ledger_err | steps/s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1000 | 246.500 | 310 | 292 | 27 | 0 | 0 | 0 | 3.002 | 0.719 | 0.028 | 0.472/0.500/0.523 | 6 | 0.061 | 0.771/3.637 | 164 | 0 (0) | 6 | 0 | 16.728 |
| 2000 | 249.400 | 151 | 130 | 20 | 0 | 0 | 0 | 3.046 | 0.706 | 0.036 | 0.468/0.500/0.529 | 6 | 0.093 | 0.560/2.822 | 143 | 0 (0) | 11 | 0 | 19.749 |
| 3000 | 265.600 | 174 | 131 | 25 | 0 | 0 | 0 | 3.138 | 0.739 | 0.053 | 0.468/0.500/0.535 | 7 | 0.090 | 0.583/3.050 | 134 | 0 (0) | 13 | 0 | 17.355 |
| 4000 | 270.100 | 220 | 186 | 31 | 0 | 0 | 0 | 3.459 | 0.679 | 0.045 | 0.459/0.500/0.542 | 7 | 0.175 | 0.951/4.191 | 127 | 0 (0) | 11 | 0 | 17.974 |
| 5000 | 295.500 | 236 | 186 | 22 | 0 | 0 | 0 | 3.136 | 0.709 | 0.034 | 0.455/0.500/0.538 | 7 | 0.094 | 0.765/3.777 | 121 | 0 (0) | 9 | 0 | 16.503 |
| 6000 | 322.800 | 233 | 192 | 15 | 0 | 0 | 0 | 2.931 | 0.687 | 0.034 | 0.450/0.500/0.540 | 10 | 0.152 | 0.853/4.891 | 116 | 0 (0) | 10 | 0 | 18.533 |
| 7000 | 364.400 | 301 | 233 | 27 | 0 | 0 | 0 | 2.842 | 0.680 | 0.025 | 0.435/0.499/0.535 | 22 | 0.192 | 0.818/6.401 | 107 | 0 (0) | 9 | 0 | 16.636 |
| 8000 | 395.600 | 327 | 267 | 27 | 0 | 0 | 0 | 2.731 | 0.671 | 0.030 | 0.427/0.493/0.531 | 12 | 0.174 | 0.895/3.919 | 102 | 0 (0) | 9 | 0 | 18.023 |
| 9000 | 440.100 | 369 | 296 | 27 | 0 | 0 | 0 | 3.129 | 0.669 | 0.034 | 0.425/0.491/0.536 | 15 | 0.172 | 0.875/5.720 | 97 | 0 (0) | 8 | 0 | 19.609 |
| 10000 | 500.200 | 439 | 352 | 31 | 0 | 0 | 0 | 3.589 | 0.614 | 0.048 | 0.417/0.488/0.538 | 6 | 0.158 | 1.010/7.198 | 92 | 0 (0) | 8 | 0 | 17.947 |
| 11000 | 525.700 | 496 | 395 | 30 | 46 | 0 | 0 | 4.081 | 0.543 | 0.040 | 0.413/0.479/0.541 | 10 | 0.214 | 1.339/6.856 | 61 | 0 (0) | 10 | 0 | 19.596 |
| 12000 | 598.700 | 591 | 473 | 43 | 1 | 0 | 0 | 4.788 | 0.505 | 0.023 | 0.407/0.476/0.541 | 12 | 0.269 | 1.634/10.138 | 56 | 0 (0) | 10 | 0 | 19.544 |
| 13000 | 701.600 | 717 | 563 | 49 | 0 | 0 | 0 | 5.593 | 0.460 | 0.033 | 0.402/0.473/0.540 | 14 | 0.219 | 1.516/14.451 | 54 | 0 (0) | 10 | 0 | 16.991 |
| 14000 | 773.600 | 726 | 602 | 44 | 4 | 0 | 0 | 6.856 | 0.417 | 0.045 | 0.404/0.472/0.541 | 8 | 0.220 | 1.482/10.672 | 51 | 0 (0) | 9 | 0 | 15.145 |
| 15000 | 901.800 | 835 | 657 | 53 | 4 | 0 | 0 | 7.214 | 0.387 | 0.044 | 0.392/0.475/0.542 | 8 | 0.153 | 1.460/6.052 | 46 | 0 (0) | 11 | 0 | 18.372 |
| 16000 | 992.000 | 893 | 724 | 74 | 3 | 0 | 0 | 7.807 | 0.354 | 0.058 | 0.381/0.469/0.542 | 10 | 0.143 | 1.619/11.473 | 42 | 0 (0) | 10 | 0 | 17.093 |
| 17000 | 992.000 | 692 | 638 | 50 | 4 | 0 | 0 | 8.374 | 0.321 | 0.083 | 0.379/0.464/0.540 | 10 | 0.150 | 1.687/8.262 | 37 | 0 (0) | 11 | 0 | 18.588 |
| 18000 | 992.000 | 481 | 436 | 42 | 3 | 0 | 0 | 8.785 | 0.317 | 0.091 | 0.375/0.461/0.544 | 12 | 0.111 | 1.620/10.996 | 32 | 0 (0) | 8 | 0 | 18.355 |
| 19000 | 992.000 | 381 | 343 | 33 | 5 | 0 | 0 | 9.145 | 0.293 | 0.095 | 0.373/0.459/0.544 | 10 | 0.131 | 1.659/9.620 | 29 | 0 (0) | 8 | 0 | 16.242 |
| 20000 | 992.000 | 338 | 316 | 20 | 2 | 0 | 0 | 9.332 | 0.279 | 0.109 | 0.366/0.459/0.544 | 14 | 0.144 | 1.862/18.930 | 25 | 0 (0) | 8 | 0 | 17.074 |

totals: {"rows": 2000, "last_step": 20000, "ledger_err_max": 0, "overflow_rows": 0, "overflow_max": 0, "occ_max": 13, "sps_median": 17.66403523943915, "peak_mem_gb_max": 0.039897626}

### `inspect --check`

| | seed 0 | seed 1 |
|---|---|---|
| full_check | **OK** | **OK** |
| ledger | OK; `ledger_error` 0 on all 2000 rows | OK; 0 on all 2000 rows |
| alive at 20k | 485 | **992 (= n_cap; births_blocked 3781 of 3785 requests in the last row)** |
| generation max | 26 | 81 |
| clades (top share) | 46 (0.20) | 25 (0.65) |
| bodies (largest) | 107 (7) | 185 (14) |
| carrying-capacity estimate (`inspect`) | 450 | 451 |
| `cell_overflow` | 0 on every row; max occupancy 11 | 0 on every row; max occupancy 13 |
| peak memory | 0.040 GB | 0.040 GB |

**Final-row depth metrics, seed 0.**
- `band_frac` [0.926, 0.023, 0.012, 0.039]: floor, lower water, upper water, surface.
- Grounded 0.82, surface 0.014.
- `clade_height_mi` 0.20.

**Final-row depth metrics, seed 1.**
- `band_frac` [0.475, 0.158, 0.140, 0.227].
- Grounded 0.28, surface 0.11, height median 6.7, q75 17.3.
- `clade_height_mi` 0.11.
- `z_surface_clamps` 953 per 10 steps.
- `body_z_extent_max` 18.9, i.e. a body spanning most of the 24-unit column.

### Reading (Part A)

**Does `buoy_cap` evolve away from 0.5?**
- The median does not move on seed 0 (0.500 → 0.496). The spread grows steadily: q10 falls from 0.477 to 0.402,
  and q90 rises from 0.519 to 0.585. Variation is accumulating with no directional selection over 20k steps.
- On seed 1 the whole distribution drifts down: median 0.500 → 0.459, q10 → 0.366. Cheaper cap upkeep is
  favoured once the population is rich (mean energy about 100k). Notably, that happens on the seed where agents
  actually use the water column.

**Do vertical niches form?** It depends on the seed, which argues against reading too much into one run.
- Seed 0 stays benthic: 82% grounded, 93% in the floor band.
- Seed 1 colonises the water column after about 10k steps: grounded falls from 0.71 to 0.28, mean height rises
  from 3.0 to 9.3, and 23% of agents are at the surface band.
- Clade–height mutual information is low on both seeds (0.20 and 0.11), so depth is not yet a clade-specific
  niche. The surface population is not a separate lineage.

**Capacity limit.**
- Seed 1 reaches `n_cap` = 992 at about step 16k. From then on almost every birth is blocked, so dynamics after
  16k are capacity-limited, not ecological.
- The `inspect` carrying-capacity estimate (about 450) is well below the realised 992. For 3D it underestimates
  K about 2×, probably because it ignores the photic band and the water column. This matters because the
  director uses n/K.
- tiny3d's `n_max` = 1024 is too small for runs longer than about 15k steps.

**Hash grid.** Never stressed at tiny3d: max occupancy 13 against C = 16, and zero overflow.


## Part B: gravity × buoyancy-cost grid (tiny3d, 10,000 steps, seed 0)

**Command:** `alife run --preset tiny3d --steps 10000 --seed 0 --set physics.gravity=G --set energy.buoyancy_cost=B --set energy.buoyancy_cap_cost=CC`

**Grid.** G ∈ {0.01, 0.03}, B ∈ {4, 16}, CC ∈ {2, 8}.
- All 8 combinations passed validation and ran to completion: exit 0, `inspect --check` **full_check OK**,
  `ledger_error` 0 on every row, `cell_overflow` 0 on every row (max occupancy ≤ 13).
- Each run took about 14 min of wall time, run 2 at a time.

**Determinism check.** G = 0.03, B = 16, CC = 8 equals the tiny3d defaults, and its rows at 5000 and 10000 match
Part A seed 0 exactly: n 277.8 and 291.3, same buoy quantiles. That confirms the runs are reproducible across
processes.

| G | B | CC | n@5k | n@10k | grounded 5k → 10k | surface 5k → 10k | height_mean @10k | buoy q10/q50/q90 @10k | band_frac @10k (floor, low, high, surface) | clade_height_mi @10k | HEAT_BUOYANCY / HEAT_BASAL (last 1k steps) | bodies_largest @10k | clades @10k | steps/s (median) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.01 | 4 | 2 | 293.3 | 335.4 | 0.436 → 0.397 | 0.160 → 0.134 | 7.57 | 0.435/0.499/0.542 | 0.624/0.110/0.054/0.212 | **0.695** | 0.003 | 8 | 109 | 17.9 |
| 0.01 | 4 | 8 | 261.4 | 285.9 | 0.457 → 0.442 | 0.172 → 0.129 | 7.23 | 0.445/0.500/0.543 | 0.654/0.077/0.063/0.206 | **0.716** | 0.024 | 5 | 102 | 17.7 |
| 0.01 | 16 | 2 | 289.1 | 289.7 | 0.434 → 0.407 | 0.208 → 0.190 | 7.70 | 0.442/0.500/0.540 | 0.638/0.090/0.041/0.231 | **0.717** | 0.004 | 8 | 108 | 18.2 |
| 0.01 | 16 | 8 | 279.0 | 314.4 | 0.419 → 0.396 | 0.172 → 0.182 | 8.21 | 0.448/0.500/0.556 | 0.589/0.111/0.051/0.248 | **0.714** | 0.026 | 8 | 110 | 18.2 |
| 0.03 | 4 | 2 | 281.0 | 281.8 | 0.810 → 0.832 | 0.025 → 0.025 | 2.44 | 0.424/0.493/0.549 | 0.922/0.018/0.018/0.043 | 0.301 | 0.003 | 11 | 105 | 18.4 |
| 0.03 | 4 | 8 | 250.7 | 238.9 | 0.780 → 0.797 | 0.036 → 0.042 | 3.19 | 0.450/0.500/0.546 | 0.887/0.013/0.025/0.076 | 0.369 | 0.025 | 7 | 96 | 18.3 |
| 0.03 | 16 | 2 | 294.5 | 292.2 | 0.764 → 0.756 | 0.027 → 0.041 | 3.14 | 0.444/0.496/0.547 | 0.881/0.027/0.031/0.061 | 0.413 | 0.004 | 11 | 96 | 18.3 |
| 0.03 | 16 | 8 | 277.8 | 291.3 | 0.787 → 0.816 | 0.029 → 0.048 | 2.81 | 0.452/0.500/0.552 | 0.903/0.017/0.017/0.062 | 0.378 | 0.025 | 19 | 97 | 18.2 |

The `inspect` carrying-capacity estimate is 451–458 on every run.

**Per-run tables.** Births and deaths by cause per 1000 steps, plus overflow, occupancy and body z-extent, at
steps 5000 and 10000:


### g0.01_b16_cc2

| step | n_alive | births | starved | killed | aged | cat | unlogged | height_mean | grounded | surface | buoy q10/q50/q90 | bodies_largest | frac_bodies≥4 | body_z_ext mean/max | clades | overflow max (rows>0) | occ max | ledger_err | steps/s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5000 | 289.100 | 171 | 149 | 21 | 0 | 0 | 0 | 7.821 | 0.434 | 0.208 | 0.451/0.500/0.527 | 7 | 0.138 | 0.836/3.285 | 123 | 0 (0) | 8 | 0 | 19.355 |
| 10000 | 289.700 | 192 | 146 | 38 | 0 | 0 | 0 | 7.703 | 0.407 | 0.190 | 0.442/0.500/0.540 | 8 | 0.117 | 1.174/4.675 | 108 | 0 (0) | 7 | 0 | 17.777 |

totals: {"rows": 1000, "last_step": 10000, "ledger_err_max": 0, "overflow_rows": 0, "overflow_max": 0, "occ_max": 10, "sps_median": 18.177921364297788, "peak_mem_gb_max": 0.039893182}

### g0.01_b16_cc8

| step | n_alive | births | starved | killed | aged | cat | unlogged | height_mean | grounded | surface | buoy q10/q50/q90 | bodies_largest | frac_bodies≥4 | body_z_ext mean/max | clades | overflow max (rows>0) | occ max | ledger_err | steps/s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5000 | 279.000 | 169 | 137 | 28 | 0 | 0 | 0 | 7.594 | 0.419 | 0.172 | 0.461/0.500/0.529 | 13 | 0.133 | 1.135/7.337 | 125 | 0 (0) | 10 | 0 | 17.781 |
| 10000 | 314.400 | 244 | 163 | 83 | 0 | 0 | 0 | 8.210 | 0.396 | 0.182 | 0.448/0.500/0.556 | 8 | 0.086 | 0.968/4.067 | 110 | 0 (0) | 6 | 0 | 18.598 |

totals: {"rows": 1000, "last_step": 10000, "ledger_err_max": 0, "overflow_rows": 0, "overflow_max": 0, "occ_max": 11, "sps_median": 18.197863237822464, "peak_mem_gb_max": 0.039901634}

### g0.01_b4_cc2

| step | n_alive | births | starved | killed | aged | cat | unlogged | height_mean | grounded | surface | buoy q10/q50/q90 | bodies_largest | frac_bodies≥4 | body_z_ext mean/max | clades | overflow max (rows>0) | occ max | ledger_err | steps/s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5000 | 293.300 | 182 | 162 | 26 | 0 | 0 | 0 | 7.712 | 0.436 | 0.160 | 0.461/0.500/0.531 | 10 | 0.129 | 0.880/3.271 | 125 | 0 (0) | 8 | 0 | 18.542 |
| 10000 | 335.400 | 199 | 160 | 26 | 0 | 0 | 0 | 7.574 | 0.397 | 0.134 | 0.435/0.499/0.542 | 8 | 0.116 | 1.091/6.330 | 109 | 0 (0) | 8 | 0 | 18.512 |

totals: {"rows": 1000, "last_step": 10000, "ledger_err_max": 0, "overflow_rows": 0, "overflow_max": 0, "occ_max": 13, "sps_median": 17.890468102963336, "peak_mem_gb_max": 0.039898074}

### g0.01_b4_cc8

| step | n_alive | births | starved | killed | aged | cat | unlogged | height_mean | grounded | surface | buoy q10/q50/q90 | bodies_largest | frac_bodies≥4 | body_z_ext mean/max | clades | overflow max (rows>0) | occ max | ledger_err | steps/s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5000 | 261.400 | 150 | 147 | 8 | 0 | 0 | 0 | 7.584 | 0.457 | 0.172 | 0.461/0.500/0.531 | 7 | 0.115 | 1.199/11.890 | 121 | 0 (0) | 10 | 0 | 15.573 |
| 10000 | 285.900 | 156 | 121 | 29 | 0 | 0 | 0 | 7.226 | 0.442 | 0.129 | 0.445/0.500/0.543 | 5 | 0.105 | 1.111/5.554 | 102 | 0 (0) | 8 | 0 | 15.829 |

totals: {"rows": 1000, "last_step": 10000, "ledger_err_max": 0, "overflow_rows": 0, "overflow_max": 0, "occ_max": 10, "sps_median": 17.74424922234686, "peak_mem_gb_max": 0.039893622}

### g0.03_b16_cc2

| step | n_alive | births | starved | killed | aged | cat | unlogged | height_mean | grounded | surface | buoy q10/q50/q90 | bodies_largest | frac_bodies≥4 | body_z_ext mean/max | clades | overflow max (rows>0) | occ max | ledger_err | steps/s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5000 | 294.500 | 167 | 156 | 10 | 0 | 0 | 0 | 2.666 | 0.764 | 0.027 | 0.464/0.500/0.538 | 13 | 0.156 | 0.528/4.438 | 118 | 0 (0) | 9 | 0 | 17.447 |
| 10000 | 292.200 | 185 | 146 | 38 | 0 | 0 | 0 | 3.141 | 0.756 | 0.041 | 0.444/0.496/0.547 | 11 | 0.116 | 0.298/3.547 | 96 | 0 (0) | 9 | 0 | 19.654 |

totals: {"rows": 1000, "last_step": 10000, "ledger_err_max": 0, "overflow_rows": 0, "overflow_max": 0, "occ_max": 10, "sps_median": 18.343049551915406, "peak_mem_gb_max": 0.03989775}

### g0.03_b16_cc8

| step | n_alive | births | starved | killed | aged | cat | unlogged | height_mean | grounded | surface | buoy q10/q50/q90 | bodies_largest | frac_bodies≥4 | body_z_ext mean/max | clades | overflow max (rows>0) | occ max | ledger_err | steps/s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5000 | 277.800 | 165 | 148 | 14 | 0 | 0 | 0 | 2.506 | 0.787 | 0.029 | 0.455/0.500/0.541 | 10 | 0.170 | 0.518/6.174 | 116 | 0 (0) | 8 | 0 | 18.488 |
| 10000 | 291.300 | 142 | 115 | 22 | 0 | 0 | 0 | 2.811 | 0.816 | 0.048 | 0.452/0.500/0.552 | 19 | 0.134 | 0.428/4.439 | 97 | 0 (0) | 9 | 0 | 19.472 |

totals: {"rows": 1000, "last_step": 10000, "ledger_err_max": 0, "overflow_rows": 0, "overflow_max": 0, "occ_max": 11, "sps_median": 18.209490183890853, "peak_mem_gb_max": 0.039892746}

### g0.03_b4_cc2

| step | n_alive | births | starved | killed | aged | cat | unlogged | height_mean | grounded | surface | buoy q10/q50/q90 | bodies_largest | frac_bodies≥4 | body_z_ext mean/max | clades | overflow max (rows>0) | occ max | ledger_err | steps/s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5000 | 281.000 | 171 | 134 | 15 | 0 | 0 | 0 | 2.503 | 0.810 | 0.025 | 0.451/0.500/0.531 | 14 | 0.163 | 0.366/2.240 | 116 | 0 (0) | 8 | 0 | 17.627 |
| 10000 | 281.800 | 158 | 109 | 32 | 0 | 0 | 0 | 2.443 | 0.832 | 0.025 | 0.424/0.493/0.549 | 11 | 0.160 | 0.307/2.226 | 105 | 0 (0) | 7 | 0 | 18.962 |

totals: {"rows": 1000, "last_step": 10000, "ledger_err_max": 0, "overflow_rows": 0, "overflow_max": 0, "occ_max": 10, "sps_median": 18.400100446883734, "peak_mem_gb_max": 0.039898142}

### g0.03_b4_cc8

| step | n_alive | births | starved | killed | aged | cat | unlogged | height_mean | grounded | surface | buoy q10/q50/q90 | bodies_largest | frac_bodies≥4 | body_z_ext mean/max | clades | overflow max (rows>0) | occ max | ledger_err | steps/s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5000 | 250.700 | 138 | 109 | 14 | 0 | 0 | 0 | 3.085 | 0.780 | 0.036 | 0.458/0.500/0.537 | 14 | 0.168 | 0.623/5.684 | 116 | 0 (0) | 8 | 0 | 18.806 |
| 10000 | 238.900 | 131 | 115 | 24 | 0 | 0 | 0 | 3.190 | 0.797 | 0.042 | 0.450/0.500/0.546 | 7 | 0.130 | 0.426/4.862 | 96 | 0 (0) | 6 | 0 | 19.027 |

totals: {"rows": 1000, "last_step": 10000, "ledger_err_max": 0, "overflow_rows": 0, "overflow_max": 0, "occ_max": 11, "sps_median": 18.342453182708006, "peak_mem_gb_max": 0.039892442}


### Reading (Part B)

**Gravity is the dominant knob.**
- At G = 0.01, about 40% of agents are grounded, about 20% sit at the surface, and mean height is about 7.5
  (of 24).
- At G = 0.03 the population is 76–83% benthic, with about 4% at the surface.
- Clade–height mutual information is about 0.70 bits at G = 0.01 versus 0.30–0.41 at G = 0.03. **Clade-specific
  depth niches form at low gravity within 10k steps.** At the default gravity they do not, at least on seed 0.

**The buoyancy costs barely matter at 10k steps.**
- `buoyancy_cost` 4 vs 16 changes nothing systematic: lift is rarely used at full strength.
- `buoyancy_cap_cost` sets the energy share. Upkeep is about 2.5% of basal heat at CC = 8 versus 0.3–0.4% at
  CC = 2. It does not yet move the `buoy_cap` median, which stays about 0.50 everywhere. The q10 tail spreads a
  little more at CC = 2 (0.424–0.435) than at CC = 8.

**Population.** About 240–335 at 10k in every cell. None hit `n_cap` within 10k steps.
