cloud-6 (PARTIAL: Part A done; Parts B and C running): tiny3d 20k-step baselines. Ledger exact and full_check OK on both seeds. buoy_cap widens but does not shift on seed 0, and drifts down on seed 1. Seed 1 colonises the water column and hits the N cap (992).

# cloud-6: 3D tuning sweeps (CPU, main)

| item | value |
|---|---|
| commit | `e1f718e68867fde71e5e3ae74107540dec4ef177` (origin/main, detached, clean) |
| `uname -m` | x86_64, 4 cores, 15 GB RAM |
| Python / mlx | 3.12.3 / 0.32.3; CPU device |
| env | `uv sync --locked --group dev` (exit 0) |
| runner | each job runs `nice -n 10 uv run alife run ... --overwrite`, then `alife inspect --check`; at most 3 concurrent processes (4 cores) |

**Raw data.** In `results/cloud-6/`:
- `partA_*_metrics.jsonl`: every 10-step row;
- `partA_*_inspect.txt`;
- `partA_summary.json`.

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
