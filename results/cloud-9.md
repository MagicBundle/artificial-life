cloud-9 (PARTIAL, at step 16,000 of 20,000): **the population peaked at about 12,500 (steps 12,500–14,000), then declined**: 12,145 / 11,786 / 11,612 / 11,404 at steps 14,500–16,000. Kills now outpace births minus starvation. This is not a stable plateau, so the run continues to 20,000. n/K_est is 0.71; no overflow; peak occupancy 43 of 48.

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
| 8001–9000 | 10170 | 11454 | 5425 / 4401 / 0 | 0.185 | 0.46/0.15/0.13/0.25 | 2.5/12.1/30.1 | 0.074 | 0 | 33 | 15103 | 4725 | 0.67 | 2.15 | 34624283 | 0.426/0.509/0.586 | 497 | 0.271 |
| 9001–10000 | 11546 | 13814 | 6479 / 5959 / 0 | 0.158 | 0.45/0.16/0.14/0.25 | 3.1/12.8/30.2 | 0.072 | 0 | 37 | 15396 | 4726 | 0.75 | 2.44 | 40103338 | 0.421/0.508/0.587 | 430 | 0.295 |
| 10001–11000 | 11972 | 15184 | 7107 / 7570 / 81 | 0.144 | 0.43/0.17/0.14/0.26 | 3.4/13.5/30.6 | 0.071 | 0 | 43 | 15624 | 4727 | 0.77 | 2.53 | 43435329 | 0.416/0.508/0.590 | 339 | 0.294 |
| 11001–12000 | 12301 | 16253 | 7256 / 8658 / 10 | 0.130 | 0.41/0.18/0.14/0.27 | 3.9/14.2/31.0 | 0.074 | 0 | 42 | 15841 | 4727 | 0.78 | 2.60 | 44628324 | 0.411/0.507/0.593 | 294 | 0.290 |
| 12001–13000 | 12500 | 17507 | 7523 / 9777 / 8 | 0.127 | 0.41/0.18/0.15/0.26 | 4.1/14.5/30.8 | 0.072 | 0 | 32 | 15873 | 4727 | 0.79 | 2.64 | 46112301 | 0.412/0.509/0.597 | 260 | 0.297 |
| 13001–14000 | 12490 | 18438 | 7450 / 10985 / 13 | 0.122 | 0.41/0.18/0.15/0.26 | 4.2/14.5/30.8 | 0.066 | 0 | 37 | 15998 | 4727 | 0.78 | 2.64 | 46423502 | 0.411/0.510/0.608 | 224 | 0.315 |
| 14001–15000 | 11786 | 18530 | 7252 / 11973 / 9 | 0.117 | 0.39/0.19/0.15/0.26 | 4.9/14.8/30.7 | 0.069 | 0 | 31 | 16158 | 4727 | 0.73 | 2.49 | 45512207 | 0.408/0.512/0.615 | 205 | 0.336 |
| 15001–16000 | 11404 | 18733 | 6846 / 12262 / 7 | 0.112 | 0.40/0.19/0.15/0.26 | 4.9/14.6/30.3 | 0.066 | 0 | 31 | 16117 | 4727 | 0.71 | 2.41 | 44029617 | 0.405/0.514/0.621 | 189 | 0.338 |

n_alive every 500 steps: [(6000, 5981), (6500, 6521), (7000, 7117), (7500, 7857), (8000, 8542), (8500, 9338), (9000, 10170), (9500, 10869), (10000, 11546), (10500, 11867), (11000, 11972), (11500, 12115), (12000, 12301), (12500, 12489), (13000, 12500), (13500, 12509), (14000, 12490), (14500, 12145), (15000, 11786), (15500, 11612), (16000, 11404)]
steps 6001–16000: overflow on 0 steps; peak occupancy 43 at step 10705

**Throughput:** 0.635 steps/s (6,000–8,000) 0.662 steps/s (8,000–10,000) 0.703 steps/s (10,000–12,000) 0.729 steps/s (12,000–14,000), and 0.56–0.62 steps/s for 14,000–16,000, alone on the CPU.

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


**Reading at 10,000.**
- **Growth has started to slow but has not stopped.** Per 500 steps: +540, +596, +740, +685, +796, +832, +699,
  +677.
- **Predation is the brake.**
  - Killed deaths per 1,000 steps: 2,576 → 3,249 → 4,401 → 5,959, about 48% of deaths.
  - Starvation deaths per 1,000 steps: 3,900 → 6,479.
  - Births per 1,000 steps: 13,814.
- **Lineages are consolidating.** One clade now holds 29.5%, and clades alive fell from 648 to 430.
- **K_est tracks the population upward** (15,396). n/K_est rises steadily: 0.51, 0.59, 0.67, 0.75. On the
  current trend n reaches K_est around 13,000–15,000, if the slowdown continues.
- **The floor-layer K is flat at about 4,725**, so it says nothing about where this settles.
- Peak occupancy is 37, at step 9,878, still under C = 48 with no overflow.


**Reading at 12,000.**
- **The population is approaching a plateau, at about 12,000–12,500 so far.**
  - Per 500 steps after 10,000: +321, +105, +143, +186.
  - Over 10,000–12,000 it grew only 755 in total, against 2,004 over 8,000–10,000.
  - It is not yet flat for 3,000 steps, so the run continues.
- **Predation is now the main cause of death.**
  - Killed per 1,000 steps: 7,570, then 8,658.
  - Starved per 1,000 steps: 7,107, then 7,256.
  - The first aged deaths appear (81, then 10). Founder-era agents are reaching `max_age`.
- **n/K_est has flattened at 0.77–0.78** (11,972/15,624 and 12,301/15,841). If n settles here, the
  occupancy-weighted K_est overestimates the 3D equilibrium by about 1.25–1.3×. That is far better than the
  about 4× at step 2,500, because K_est only became accurate once the column filled.
- **n/floor-K is about 2.5–2.6.** The floor layer alone supports about 40% of the equilibrium.
- **Occupancy warning.** Peak occupancy jumped to **43** (step 10,705) and 42 in the next window, against
  C = 48. The rise with density is real. 64k3d at higher density may need C > 48, or a confirmation run.
- One clade holds about 29%, and clades alive fell to 294.


**Reading at 14,000.**
- **Plateau.** n_alive every 500 steps from 12,500 is 12,489, 12,500, 12,509, 12,490: within ±0.1% over
  1,500 steps.
- **The balance at the plateau.**
  - Births per 1,000 steps are still rising (17,507 → 18,438).
  - They are matched by kills (9,777 → 10,985), which now outnumber starvation deaths (about 7,450 per 1,000)
    by about 1.5×.
  - The equilibrium is set by predation as much as by food.
- **Calibration numbers at the plateau.**
  - **n / K_est = 0.78–0.79** (12,500 / 15,873; 12,490 / 15,998).
  - **n / floor-K = 2.64** (floor-K 4,727).
  - The column structure is stable:
    - band_frac 0.41 / 0.18 / 0.15 / 0.26;
    - height median 14.5;
    - surface share about 0.07;
    - grounded share about 0.12.
- **Occupancy.** Peaks of 32 and 37 in the last two windows. The all-time maximum is still 43, at step 10,705.
  No overflow on any of the 8,000 steps since 6,000.
- buoy_cap's upper tail is creeping up (q90 0.608) while the median holds at 0.51.
- Clades are consolidating: 224 alive, top share 0.315.


**Interruptions.**
- The container restarted twice during the first 14,000–16,000 attempts, and both were lost.
- From step 14,000 the run uses 500-step chunks, each saving its state. A restart now loses at most about
  15 minutes of compute.
- 14,000–16,000 completed in four 500-step chunks, with counters merged into
  `results/cloud-9-counters_14000_16000.npz`.

**Reading at 16,000.**
- **The plateau did not hold.** After 1,500 steps at about 12,500 (12,489 / 12,500 / 12,509 / 12,490), n
  fell steadily: 12,145, 11,786, 11,612, 11,404. That is a loss of about 1,100, or 9%, over 2,000 steps.
- **Predation is still rising** while births have flattened:
  - killed per 1,000 steps: 10,985 → 11,973 → 12,262;
  - births per 1,000 steps: 18,438 → 18,530 → 18,733;
  - starvation is falling (7,450 → 6,846).

  This looks like a predator–prey overshoot or an oscillation, not yet an equilibrium.
- **K_est has stopped rising** (about 16,100–16,160), so n/K_est falls from 0.78 to 0.71. Floor-K is unchanged
  at 4,727, and n/floor-K is 2.41.
- The column structure is unchanged: band_frac about 0.40 / 0.19 / 0.15 / 0.26, height median about 14.7,
  grounded about 0.11.
- **The top clade keeps growing** (0.315 → 0.338) while clades alive keep falling (224 → 189). The rising kill
  rate coincides with one lineage's spread. It may be a predatory lineage; this was not measured.
- Occupancy peaks at 31 in both windows. The all-time maximum is still 43, and there is no overflow on any step.
- **For calibration so far:** the realised population ranges from 0.71 to 0.79 of K_est over steps 11,000–16,000.
  The long-run level is not yet known. Steps 16,000–20,000 are running.
