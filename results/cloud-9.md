cloud-9 (PARTIAL, at step 8,000 of 20,000): **the population has not levelled off. It is accelerating**: 5,981 → 7,117 → 8,542, about +1,280 per 1,000 steps. n/floor-K is 1.81 and n/K_est 0.59. No overflow (peak occupancy 31 of 48).

# cloud-9: DEV3D continuation to 20,000 steps (`3d/pr6-64k3d` at `a3d46eb`, C = 48)

**Setup:**
- Resumes cloud-8's saved 6,000-step state (`runio.load_checkpoint`).
- The step loop is the one in `tests/test_ecology_3d.run_dev3d`, with `dev3d_cfg()` (C = 48, K = 8), run in
  2,000-step chunks.
- Each chunk saves `state_<end>.safetensors`, so the run can resume after a pause.

**Recording:**
- Every 500 steps the script records a state snapshot to `results/cloud-9-snapshots.json`:
  - n_alive;
  - `metrics.carrying_capacity` (K_est, occupancy-weighted) and `metrics.floor_carrying_capacity` (floor K);
  - height quartiles and mean, `surface_frac`, and the 4-band `band_frac`, all with `metrics._snapshot_3d`'s
    definitions;
  - buoy_cap q10/q50/q90;
  - clades alive and top clade share.
- Per-step counters and ledgers go to `results/cloud-9-counters_<a>_<b>.npz`.

**Platform:** x86_64 Linux, 4 cores, Python 3.12.3, mlx 0.32.3 on the CPU.

**Table columns.**
- Births and deaths are per 1,000-step window.
- "grounded" is the window mean of `GROUNDED / N_ALIVE`.
- Snapshot columns are at the window's end.
- Bands run floor → surface, each 10 units of the 40-unit column.

| steps | n_alive (end) | births | starved / killed / aged | grounded (mean) | band_frac (4 bands, end) | height q25/q50/q75 (end) | surface (end) | overflow steps | peak occ (C=48) | K_est | floor K | n/K_est | n/floor K | HEAT_BUOYANCY | buoy_cap q10/q50/q90 | clades | top clade |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 6001–7000 | 7117 | 7612 | 3900 / 2576 / 0 | 0.233 | 0.53/0.14/0.11/0.22 | 1.6/8.4/27.8 | 0.078 | 0 | 30 | 14065 | 4703 | 0.51 | 1.51 | 24048473 | 0.436/0.504/0.571 | 648 | 0.159 |
| 7001–8000 | 8542 | 9227 | 4553 / 3249 / 0 | 0.207 | 0.49/0.15/0.12/0.24 | 2.2/10.4/29.1 | 0.076 | 0 | 31 | 14593 | 4722 | 0.59 | 1.81 | 28975975 | 0.433/0.507/0.579 | 578 | 0.213 |

n_alive every 500 steps: [(6000, 5981), (6500, 6521), (7000, 7117), (7500, 7857), (8000, 8542)]
steps 6001–8000: overflow on 0 steps; peak occupancy 31 at step 7088

**Throughput:** 0.635 steps/s (3,148 s per 2,000 steps), alone on the CPU.

**Reading at 8,000.**
- **No plateau.** Growth is accelerating: +540 per 500 steps around step 6,000, up to +685 per 500 by
  7,500–8,000.
- Killing deaths are rising fast: 2,576 → 3,249 per 1,000 steps, now about 40% of deaths. Predation is
  scaling with density.
- The column keeps filling:
  - the floor band falls from 0.53 to 0.49;
  - the surface band rises from 0.22 to 0.24;
  - median height climbs from 8.4 to 10.4.
- **K_est rises with the population** (14,065 → 14,593), because it is occupancy-weighted and the newcomers live
  in productive upper layers. n/K_est climbs (0.51 → 0.59) more slowly than n/floor-K (1.51 → 1.81).
- Peak occupancy stays at 30–31, far below C = 48.
- buoy_cap is slowly widening (q10 0.433, q90 0.579) with the median about 0.50.
- One clade is spreading: its top share went from 0.159 to 0.213, and clades fell from 648 to 578.
