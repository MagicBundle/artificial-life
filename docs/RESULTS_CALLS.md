# Results of the calls pilot (K5)

Pilot of the event-triggered calls with far hearing (DESIGN section 47; spec Revision 2, sections 7.2 to 7.4). It ran on `engine/calls-k5` from `f960ceb` (the K2 head), on `mini3s` with `stormsph22h`, `comm` and `calls`,
fresh founders, weather spun up 8 000 steps. The engine source is unchanged; the pilot's outputs on this branch are the overlay values (`stormsph22h.storm_u0`, `calls.far_range`, `calls.far_xref`), their test, DESIGN 47.9 and this file.
Every launch asserted from the run's config that `D_IN` is 95 and the token `calls2@1` is present (57 jobs). Seed 0 for the dial runs, `run.seed` 1 to 6 for the forks, metrics every 500 steps. The runs themselves (about 34 GB),
the scripts and the logs are in the session scratchpad (`SP/calls_pilot/`) and are not in the repository; the draft of the pre-registration is `SP/calls_pilot/prereg_calls_v4_draft.json` (not frozen).

## 0. Summary

| gate | verdict | number |
|---|---|---|
| G1 hazard: storm deaths 3 to 10 % of all deaths of evolved F agents, last 20 000 steps of P1 | **PASS, at the lower edge** | 0.05 gives 4.28 % (far_xref 0.7), **3.68 % (far_xref 3.0, the frozen run)**, 3.05 % (far_xref 4.0); 0.06 gives 1.82 %, 0.075 0.55 %, 0.095 0.07 %. Only 0.05 is in the band. In the chosen run the share is 4.5, 2.6, 2.0, 2.4, 3.2, 3.7, 3.6, 3.7 % per 10 000 steps: it is below 3 % for steps 10 000 to 40 000 |
| G2 event rates: each primary `calls_open_<e>` 0.5 to 25 % of alive rows | **PASS** | chosen run, last 20 000 steps: hurt 10.4 %, storm 6.8 % |
| G3 cost: `comm_due_share` 0.5 to 5 % | **PASS at 32** (8 fails) | 1.23 % at 32 (far_xref 3.0). Sensitivity at far_xref 4.0: cost 8 0.32 % (FAIL), 32 0.81 %, 128 2.80 % |
| G4 closure and purity | **PASS** | `ledger_error` 0 in every metrics row of all 57 runs, `ledger OK` in every log line, `alife inspect --check` (`full_check`) OK on the P1 checkpoint and on a `reflex` fork, no error bit of `diag`; the advisory bit `WEATHER_WARN` (1<<18) is set in every weather run (also in the K2 review's runs), as DESIGN 12.4 and 30.8 describe |
| G5 the world can reward a warning (Revision 2) | **PASS on its first clause; the net-fitness clause is reported and the oracle's net fitness FALLS** | `reflex`: storm deaths per agent-step -75.8 % against `none` (2h = 24 % from 4 null forks, 15 % from 6), but `n_alive` -59 %; `reflex_near`: -28.4 %, `n_alive` -2.4 %, births per agent-step +8.3 %. Sweeps in section 6 |

**Chosen inputs (frozen for P2 and proposed for M1):** `stormsph22h.storm_u0 = 0.05`; `comm.cost = 32` (unchanged); `calls.far_range = 18`; `calls.far_xref = 3.0`
(not 0.7); `calls.hold = 4`, `calls.events = 5` unchanged. The overlays `stormsph22h` and `calls` on the branch now carry them.

**Things the numbers say that the plan did not expect** (details in the sections named):
1. The 5 % effect of interest is 0.18 % of all deaths (storm deaths are 3.7 % of deaths): a bound on the selection pressure for any warning (section 8).
2. The pilot's evolved F (80 000 steps, one seed) shows a far-block effect of **+1.9 % storm deaths in `nofar` (95 % CI +0.3 to +3.5, six seeds)**, below 5 % (section 8).
3. The spec's rule "5 % valid only if h <= 2.5 %" fails for the spec's null forks (h = 12 % for 2 against 2 forks of 20 000 steps) and holds for the seed-paired design (h = 1.8 % for 3 pairs) (section 7).
4. Which call channel an evolved population uses is not stable: at cost 32 the storm channel is called at 0.84 and the hurt channel at 0.03; at cost 8 or 128 (far_xref 4.0) the reverse (section 4).
5. Every oracle reduction of storm deaths of 40 % or more costs 20 to 59 % of the population (section 6).
6. `far_xref` cannot be calibrated on the founders: the statistic depends on the world evolved at that value (section 3).

## 1. Runs, budget and speed

| stage | runs | steps each |
|---|---|---|
| smoke | F fresh 1 000 steps; forks `none nofar deaf decoy reflex reflex_near reflex_call reflex(gain 0.5, min 0.1) hazard-off` 500 steps | all passed, ledger OK, `calls_oracle_on_frac` present for `reflex*` |
| P1 dial | `P1_p095 P1_p075 P1_p06 P1_p05` (far_xref 0.7, cost 32) and `P1_off` (`storm_p = 0`, `storm_u0` 0.06, far_xref 0.7) | 80 000 |
| P1 reruns | `P1_u05_x4` (0.05, far_xref 4.0), `P1_u05_x3` (0.05, far_xref 3.0) | 80 000 |
| P1b cost | `P1b_c8`, `P1b_c128` (0.05, far_xref 4.0) | 80 000 |
| P2 | forks of `P1_u05_x3:80000`: `none` seeds 1 to 6, `nofar` and `decoy` seeds 1 to 6, `deaf reflex reflex_near reflex_call` seeds 1 and 2, `pnull` seeds 1 to 6 (a `none` fork with `comm.cost = 32.001`, see section 7) | 20 000 |
| G5 sweep | `reflex` and `reflex_near` with `oracle_gain` 0.5, 0.25, 0.1 and `reflex` with `oracle_min` 0.2, 0.5, seeds 1 and 2 | 20 000 |

Speed (Metal, M5 Pro, about 10 000 to 12 000 alive): one process 153 steps/s; two processes at once 101 steps/s each, 202 aggregate (32 % faster overall), so all work ran two at a time.
Memory 0.41 GB peak per process. Total 1.68 M steps in 57 jobs.

**GPU-hours.** The GPU was busy 2.38 h of wall time (15:57 to 18:21, two processes at once) plus about 0.1 h for smoke and the speed test: **2.5 GPU-h wall**.
In process-hours (the unit of the spec's 100 to 115 steps/s planning) it is 4.6 h. Plan 2.5, ceiling 5: under both ways of counting. The excess over the plan's 880 000 steps
is the two `far_xref` reruns (160 000), the oracle sweeps (16 forks) and the extra seeds for the noise (16 forks).

The run reproduces bit for bit: forks with the same seed share the weather exactly (`wind_max_l1`, `rain_total` identical in all controls of a seed), and a fork repeated in the speed test gave the same `n_alive`.

## 2. P1: the dial (gate G1) and the negative controls

| run | storm_u0 | far_xref | cost | storm deaths % of deaths (last 20k) | storm deaths / agent-step | exposure | n_alive | births / agent-step | open hurt / storm | call rate hurt / storm | due share % | median heard storm ch., audible rows 60k / 80k | same, rows in a storm 60k / 80k | steps/s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P1_p095 | 0.095 | 0.7 | 32 | 0.07 | 8.84e-07 | 0.0004 | 10918 | 1.33e-03 | 0.134 / 0.002 | 0.02 / 0.43 | 0.05 | 0.482 / 0.242 | 0.350 / 0.202 | 97 |
| P1_p075 | 0.075 | 0.7 | 32 | 0.55 | 9.05e-06 | 0.0045 | 10536 | 1.64e-03 | 0.151 / 0.016 | 0.39 / 0.46 | 0.96 | 0.371 / 0.606 | 0.579 / 0.834 | 103 |
| P1_p06 | 0.06 | 0.7 | 32 | 1.82 | 3.46e-05 | 0.0173 | 10868 | 1.91e-03 | 0.099 / 0.045 | 0.08 / 0.80 | 0.73 | 0.862 / 0.932 | 0.993 / 0.999 | 103 |
| P1_p05 | 0.05 | 0.7 | 32 | 4.28 | 5.31e-05 | 0.0268 | 8759 | 1.25e-03 | 0.115 / 0.066 | 0.60 / 0.77 | 2.31 | 0.912 / 0.996 | 0.993 / 1.000 | 97 |
| P1_off | 0.06, storm_p=0 | 0.7 | 32 | 0.00 | 0.00e+00 | 0.0183 | 10677 | 1.92e-03 | 0.091 / 0.045 | 0.14 / 0.43 | 0.54 | 0.587 / 0.754 | 0.841 / 0.953 | 98 |
| P1_u05_x4 | 0.05 | 4.0 | 32 | 3.05 | 6.34e-05 | 0.0319 | 10523 | 2.09e-03 | 0.082 / 0.076 | 0.08 / 0.56 | 0.81 | 0.225 / 0.303 | 0.334 / 0.433 | 99 |
| P1_u05_x3 | 0.05 | 3.0 | 32 | 3.68 | 5.66e-05 | 0.0285 | 10661 | 1.54e-03 | 0.104 / 0.068 | 0.03 / 0.84 | 1.23 | 0.493 / 0.650 | 0.697 / 0.822 | 104 |
| P1b_c8 | 0.05 | 4.0 | 8 | 3.00 | 6.22e-05 | 0.0312 | 11786 | 2.07e-03 | 0.058 / 0.074 | 0.88 / 0.05 | 0.32 | 0.035 / 0.030 | 0.036 / 0.034 | 108 |
| P1b_c128 | 0.05 | 4.0 | 128 | 3.06 | 6.46e-05 | 0.0324 | 11499 | 2.11e-03 | 0.073 / 0.077 | 0.55 / 0.06 | 2.80 | 0.033 / 0.033 | 0.035 / 0.040 | 109 |

Columns: share and rate are over the last 20 000 steps; "median heard" is the median over alive rows of `voice_far` (storm channel, rows with heard >= 0.02) at the checkpoints 60 000 and 80 000 for the run's own far_xref
(the "in a storm" columns restrict to rows whose own storm window is open); steps/s is the whole run with two processes at once.

Shares of all deaths by 10 000 steps (storm deaths / (starved + killed + storm)):
`P1_p095` 0.01 0.07 0.02 0.03 0.05 0.08 0.08 0.05 %; `P1_p075` 0.56 0.51 0.34 0.48 0.51 0.64 0.58 0.52 %; `P1_p06` 2.19 1.76 1.36 1.79 1.95 2.10 1.87 1.77 %;
`P1_p05` 4.19 2.74 2.40 2.89 2.86 3.93 4.42 4.16 %; `P1_u05_x4` 5.27 3.32 2.09 2.66 2.89 2.96 2.99 3.11 %; `P1_u05_x3` 4.47 2.59 1.99 2.39 3.24 3.70 3.64 3.72 %.

* **The ladder.** The tracer model of the spec (5.2) said 0.06 would give 10 to 13 % of baseline mortality for a well-mixed population and probably 0.3 to 2 % for agents; the measured value is 1.8 %, and 0.05 gives 4.3 %. Agent exposure (the mean ramp over alive rows) is 0.0173 at 0.06 and 0.027 to 0.032 at 0.05, against 0.068 to 0.093 for the well-mixed model at 0.06: agents are 4 to 5 times less exposed. G1 is met by one ladder value only. No extra dial value was added.
* **The shipped 0.095** is the negative control: 0.07 % of deaths, 9e-7 storm deaths per agent-step; `calls_open_storm` 0.2 % (it would fail G2). **Hazard-off** (`P1_off`): 0 storm deaths. In `P1_off` the storm window still opens (it is tied to `storm_u0`, here 0.06) and the storm call rate is 0.43 in the last 20 000 steps (0.35 to 0.46 over the run; the founders' first windows in every run are 0.42 to 0.50).
* **Exposure over time** does not fall in the chosen run (windows of 10 000 steps): 0.027, 0.023, 0.020, 0.026, 0.029, 0.033, 0.029, 0.029. Evolution has not found an own-wind escape that lowers it in 80 000 steps.
* **Is the call rate a response to the hazard?** Storm call rate in the last 20 000 steps: hazard-off 0.43, shipped 0.095 0.43, 0.075 0.46, 0.06 0.80, 0.05 0.77 (all far_xref 0.7), 0.05 at far_xref 3.0 0.84, at 4.0 0.56. One seed each. At far_xref 0.7 the rate is higher where the storm is lethal (0.80, 0.78) than where it is not (0.42 to 0.46); at far_xref 4.0 the difference is smaller (0.56 against 0.45). The hurt-channel call rate in the same window varies from run to run: 0.02 (0.095), 0.39 (0.075), 0.08 (0.06), 0.60 (0.05 at far_xref 0.7), 0.14 (hazard-off), 0.08 (far_xref 4.0), 0.03 (far_xref 3.0); it is below 0.15 in five of the seven and is not tied to the hazard. A fall from the founders' 0.43 to 0.5 is what a selection against the "dinner bell" (spec R10) would give; nothing here separates it from drift.
* **Population and clonality.** The chosen run at 80 000 steps has 4 clades alive, the top clade holds 90 % of the agents, `generation_max` 179. Every fork in P2 starts from this single-lineage state.

`P1_u05_x3`, per 10 000 steps (`storm%dth` is the share, `exposure` the mean hazard ramp, `open_*` the share of alive rows with an open window, `rate_*` the call rate when open, `due%` `comm_due_share`):

```
=== P1_u05_x3: to step 80000
          win  n_alive storm%dth  stormrate     birth  exposure  open_h  open_s  rate_h  rate_s   due%  E_mean   sps
    0-10   k     6723      4.47   5.56e-05  1.36e-03    0.0268  0.0208  0.0669   0.462   0.451   0.69   24330   104
   10-20   k    12246      2.59   4.56e-05  1.77e-03    0.0229  0.0463  0.0551   0.340   0.398   0.74   22334   103
   20-30   k    11953      1.99   3.96e-05  1.97e-03    0.0200  0.0651  0.0536   0.451   0.426   1.18   22324   104
   30-40   k     7846      2.39   5.13e-05  2.11e-03    0.0260  0.0703  0.0652   0.466   0.461   1.44   22824   104
   40-50   k     7726      3.24   5.70e-05  1.79e-03    0.0290  0.0798  0.0710   0.246   0.645   1.37   21748   104
   50-60   k    10409      3.70   6.40e-05  1.74e-03    0.0325  0.1078  0.0760   0.054   0.796   1.21   20519   104
   60-70   k    10200      3.64   5.85e-05  1.60e-03    0.0285  0.1017  0.0674   0.038   0.840   1.23   20228   107
   70-80   k    11122      3.72   5.48e-05  1.49e-03    0.0285  0.1061  0.0692   0.030   0.835   1.22   20394   104
  LAST20k: storm share 3.68% (deaths_storm 12063 of 327826), ledger_err_max 0, diag_or 262144
```

## 3. `far_xref`

The K2 report predicted 0.5 to 0.8 from founders. The evolved worlds need more: the storm channel is called by 360 to 640 of 8 500 to 12 000 agents and the far sums add up.
At the 80 000-step checkpoints, median heard of the audible rows (heard >= 0.02), storm channel:

| world (evolved at) | evaluated at 0.7 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| `P1_p05`, 60 000 | 0.91 | 0.58 | 0.44 | 0.36 | 0.31 | 0.27 |
| `P1_p05`, 80 000 | 1.00 | 0.86 | 0.73 | 0.63 | 0.55 | 0.49 |
| `P1_u05_x4`, 60 000 | | 0.39 | 0.28 | 0.23 | 0.19 | 0.16 |
| `P1_u05_x4`, 80 000 | | 0.51 | 0.38 | 0.30 | 0.25 | 0.22 |
| `P1_u05_x3`, 60 000 | | 0.64 | 0.49 | 0.40 | | |
| `P1_u05_x3`, 80 000 | | 0.79 | 0.65 | 0.55 | | |

At 0.7 the storm channel saturates (median 0.86 to 1.00 in the 0.06 and 0.05 worlds). The first calibration from the 0.7 worlds pointed at 4.0 (0.36 and 0.63 at the two checkpoints of `P1_p05`).
Rerunning the dial at 4.0 changed the world: the median at its own far_xref is 0.23 (60 000) and 0.30 (80 000), below the target at the first checkpoint. Rerunning at 3.0 gives
**0.49 and 0.65** at its own far_xref, inside [0.3, 0.7] at both checkpoints, with G1 3.68 % against 3.05 %. **Frozen: 3.0.** Caveats: (a) the statistic moves by 0.1 to 0.2 between two checkpoints of the same run;
(b) read for rows in a storm (own storm window open) the same run gives 0.70 and 0.82 at 3.0, above 0.7 at the second checkpoint, and 0.33 / 0.43 at 4.0; the literal "audible rows" reading was used;
(c) the hurt channel median at 3.0 in the chosen world is 0.06 (about 40 hurt callers), far below the target: only the storm channel is calibrated; (d) one seed per value: the choice between 3.0 and 4.0 rests on one run each. 3.0 is also the value the K2 builder left as provisional.

## 4. P1b: cost (gates G3 and G2)

`comm_due_share` over the last 20 000 steps: cost 8: 0.32 % (FAIL, below 0.5 %), cost 32: 1.23 % at far_xref 3.0 (0.81 % at 4.0), cost 128: 2.80 % (both far_xref 4.0). G3 holds at 32 and 128; 32 is kept.
The cost runs were made at far_xref 4.0 (the value being tested when they were launched); the chosen run at 3.0 has cost 32 only.

What cost does to the evolved channels (storm / hurt call rate when the window is open, last 20 000 steps): cost 8: 0.05 / 0.88; cost 32 (far_xref 4.0): 0.56 / 0.08; cost 32 (far_xref 3.0): 0.84 / 0.03; cost 128: 0.06 / 0.55.
At cost 8 and 128 the population calls on the hurt channel and almost not on the storm channel, and the median far intensity of the storm channel is 0.03 (28 and 64 storm callers): the far block carries almost nothing about storms.
That is one seed per cost and a different far_xref from the frozen one, so it is a warning and not a finding: the channel an evolving population uses can flip between runs. M1's O1 (storm call rate in each F seed above the D reference by 0.05) may be missed by a seed for this reason alone.

G2 at the chosen run: hurt 10.4 %, storm 6.8 % of alive rows open (last 20 000 steps); over the run 2.1 to 10.8 % and 5.4 to 7.6 %. PASS. (At the shipped 0.095 the storm window is open 0.2 % of the time.)

## 5. G4

57 runs (smoke and speed runs excluded): `ledger_error` 0 in every row, every log line `ledger OK`, no run exited with a violation. `diag` is 262 144 (bit 18, `WEATHER_WARN`) in every run, error mask clear.
`alife inspect --check`: `full_check: OK` for `P1_u05_x3` at step 80 000 and for `P2_reflex_s1` at 100 000. No `Traceback` in any log.

## 6. P2: oracle forks, G5, R and Rn curves

All forks start from `P1_u05_x3` at step 80 000, run 20 000 steps, `calls.far_xref` 3.0, `weather.storm_u0` 0.05. Storm deaths per agent-step is deaths_storm over the sum of alive agents over the steps.

| fork | n_alive | storm deaths | storm deaths / agent-step | births / agent-step | mean energy | exposure | oracle on frac | due share % | call rate storm / hurt |
|---|---|---|---|---|---|---|---|---|---|
| none_s1 | 11534 | 14424 | 6.253e-05 | 1.403e-03 | 20252 | 0.0310 | - | 1.21 | 0.84 / 0.03 |
| none_s2 | 11566 | 13698 | 5.922e-05 | 1.382e-03 | 20259 | 0.0300 | - | 1.17 | 0.84 / 0.03 |
| none_s3 | 11539 | 13554 | 5.873e-05 | 1.399e-03 | 20241 | 0.0297 | - | 1.14 | 0.84 / 0.03 |
| none_s4 | 11556 | 13205 | 5.714e-05 | 1.416e-03 | 20319 | 0.0288 | - | 1.12 | 0.85 / 0.03 |
| none_s5 | 11577 | 13723 | 5.927e-05 | 1.393e-03 | 20242 | 0.0289 | - | 1.15 | 0.86 / 0.03 |
| none_s6 | 11566 | 13810 | 5.970e-05 | 1.392e-03 | 20273 | 0.0296 | - | 1.16 | 0.85 / 0.03 |
| nofar_s1 | 11596 | 14732 | 6.352e-05 | 1.459e-03 | 20133 | 0.0318 | - | 1.33 | 0.86 / 0.04 |
| nofar_s2 | 11608 | 14297 | 6.159e-05 | 1.444e-03 | 20185 | 0.0306 | - | 1.27 | 0.86 / 0.04 |
| nofar_s3 | 11672 | 14136 | 6.056e-05 | 1.448e-03 | 20244 | 0.0299 | - | 1.24 | 0.86 / 0.05 |
| nofar_s4 | 11680 | 13455 | 5.760e-05 | 1.402e-03 | 20201 | 0.0294 | - | 1.21 | 0.86 / 0.04 |
| nofar_s5 | 11639 | 14100 | 6.057e-05 | 1.421e-03 | 20144 | 0.0295 | - | 1.22 | 0.86 / 0.04 |
| nofar_s6 | 11634 | 13855 | 5.954e-05 | 1.427e-03 | 20186 | 0.0296 | - | 1.22 | 0.86 / 0.03 |
| deaf_s1 | 11579 | 14955 | 6.458e-05 | 1.424e-03 | 20214 | 0.0324 | - | 1.41 | 0.86 / 0.05 |
| deaf_s2 | 11601 | 14315 | 6.170e-05 | 1.417e-03 | 20155 | 0.0306 | - | 1.30 | 0.85 / 0.03 |
| decoy_s1 | 11703 | 14604 | 6.239e-05 | 1.377e-03 | 20240 | 0.0313 | - | 1.33 | 0.86 / 0.03 |
| decoy_s2 | 11601 | 13832 | 5.962e-05 | 1.374e-03 | 20294 | 0.0304 | - | 1.24 | 0.84 / 0.03 |
| decoy_s3 | 11662 | 14065 | 6.030e-05 | 1.367e-03 | 20256 | 0.0299 | - | 1.27 | 0.86 / 0.03 |
| decoy_s4 | 11629 | 13063 | 5.617e-05 | 1.353e-03 | 20242 | 0.0288 | - | 1.20 | 0.86 / 0.03 |
| decoy_s5 | 11654 | 13903 | 5.965e-05 | 1.361e-03 | 20246 | 0.0292 | - | 1.26 | 0.87 / 0.03 |
| decoy_s6 | 11701 | 14006 | 5.985e-05 | 1.355e-03 | 20234 | 0.0300 | - | 1.30 | 0.87 / 0.03 |
| reflex_s1 | 4775 | 1491 | 1.561e-05 | 1.449e-03 | 20299 | 0.0082 | 0.214 | 2.08 | 1.00 / 1.00 |
| reflex_s2 | 4655 | 1325 | 1.423e-05 | 1.394e-03 | 20243 | 0.0073 | 0.202 | 1.98 | 1.00 / 1.00 |
| reflex_near_s1 | 11260 | 10192 | 4.526e-05 | 1.510e-03 | 20100 | 0.0223 | 0.070 | 2.57 | 1.00 / 1.00 |
| reflex_near_s2 | 11294 | 9463 | 4.189e-05 | 1.507e-03 | 20154 | 0.0211 | 0.065 | 2.50 | 1.00 / 1.00 |
| reflex_call_s1 | 11482 | 14618 | 6.366e-05 | 1.384e-03 | 20192 | 0.0311 | - | 2.85 | 1.00 / 1.00 |
| reflex_call_s2 | 11391 | 13549 | 5.947e-05 | 1.400e-03 | 20062 | 0.0298 | - | 2.78 | 1.00 / 1.00 |
| pnull_s1 | 11496 | 14537 | 6.322e-05 | 1.382e-03 | 20301 | 0.0310 | - | 1.19 | 0.83 / 0.02 |
| pnull_s2 | 11545 | 13663 | 5.917e-05 | 1.374e-03 | 20252 | 0.0299 | - | 1.16 | 0.85 / 0.03 |
| pnull_s3 | 11557 | 13634 | 5.898e-05 | 1.406e-03 | 20177 | 0.0296 | - | 1.16 | 0.86 / 0.03 |
| pnull_s4 | 11556 | 13200 | 5.711e-05 | 1.367e-03 | 20158 | 0.0290 | - | 1.14 | 0.85 / 0.04 |
| pnull_s5 | 11546 | 13753 | 5.956e-05 | 1.366e-03 | 20162 | 0.0293 | - | 1.14 | 0.84 / 0.03 |
| pnull_s6 | 11557 | 13787 | 5.965e-05 | 1.381e-03 | 20193 | 0.0292 | - | 1.17 | 0.85 / 0.03 |
| reflex_g0.1_s1 | 10874 | 11501 | 5.288e-05 | 1.400e-03 | 20113 | 0.0259 | 0.769 | 2.70 | 1.00 / 1.00 |
| reflex_g0.1_s2 | 10913 | 10790 | 4.944e-05 | 1.489e-03 | 20221 | 0.0248 | 0.765 | 2.72 | 1.00 / 1.00 |
| reflex_g0.25_s1 | 9294 | 6969 | 3.749e-05 | 1.347e-03 | 20103 | 0.0187 | 0.697 | 2.36 | 1.00 / 1.00 |
| reflex_g0.25_s2 | 9279 | 6161 | 3.320e-05 | 1.477e-03 | 20188 | 0.0169 | 0.683 | 2.41 | 1.00 / 1.00 |
| reflex_g0.5_s1 | 6900 | 2611 | 1.892e-05 | 1.583e-03 | 20442 | 0.0091 | 0.480 | 2.23 | 1.00 / 1.00 |
| reflex_g0.5_s2 | 6471 | 2312 | 1.786e-05 | 1.570e-03 | 20377 | 0.0088 | 0.438 | 2.06 | 1.00 / 1.00 |
| reflex_m0.2_s1 | 6352 | 2878 | 2.266e-05 | 1.466e-03 | 20210 | 0.0111 | 0.168 | 2.24 | 1.00 / 1.00 |
| reflex_m0.2_s2 | 6260 | 2636 | 2.105e-05 | 1.423e-03 | 20315 | 0.0106 | 0.144 | 2.11 | 1.00 / 1.00 |
| reflex_m0.5_s1 | 8202 | 5934 | 3.617e-05 | 1.275e-03 | 20022 | 0.0180 | 0.095 | 2.39 | 1.00 / 1.00 |
| reflex_m0.5_s2 | 8355 | 5681 | 3.400e-05 | 1.306e-03 | 20060 | 0.0169 | 0.089 | 2.36 | 1.00 / 1.00 |
| reflex_near_g0.1_s1 | 11748 | 15572 | 6.628e-05 | 1.415e-03 | 20154 | 0.0332 | 0.129 | 2.99 | 1.00 / 1.00 |
| reflex_near_g0.1_s2 | 11776 | 14667 | 6.227e-05 | 1.412e-03 | 20156 | 0.0317 | 0.124 | 2.90 | 1.00 / 1.00 |
| reflex_near_g0.25_s1 | 11701 | 14956 | 6.391e-05 | 1.427e-03 | 20086 | 0.0320 | 0.119 | 2.88 | 1.00 / 1.00 |
| reflex_near_g0.25_s2 | 11727 | 14288 | 6.092e-05 | 1.427e-03 | 20157 | 0.0306 | 0.114 | 2.85 | 1.00 / 1.00 |
| reflex_near_g0.5_s1 | 11538 | 12953 | 5.613e-05 | 1.468e-03 | 20147 | 0.0278 | 0.099 | 2.76 | 1.00 / 1.00 |
| reflex_near_g0.5_s2 | 11530 | 11945 | 5.180e-05 | 1.447e-03 | 20200 | 0.0261 | 0.093 | 2.66 | 1.00 / 1.00 |

G5 per Revision 2. Relative change (%) of the mean of seeds 1 and 2 against the same seed of `none`, then against `reflex_call`; per seed in brackets; `*` marks |change| > 2h with
h the 95 % half-width of the null forks for 2 forks against 2 forks (from the four null forks of seeds 1 to 4; the h values are in the first line: storm deaths 12.1 %, births 3.2 %, `n_alive` 0.4 %, energy 0.6 %).
These h are the unpaired ones (different seeds, different weather), a conservative choice for same-seed comparisons (section 7):

```
### {arm} against `none` (relative change in %, mean of seeds 1,2 [per seed]; flag: |change| > 2h with h = null half-width 2v2)
arm                                    storm deaths/agent-step                 births/agent-step                           n_alive                       mean energy
nofar                                        +2.8  [+1.6,+4.0]                 +4.2  [+4.0,+4.5]                 +0.4  [+0.5,+0.4]                 -0.5  [-0.6,-0.4]
deaf                                         +3.7  [+3.3,+4.2]                 +2.0  [+1.5,+2.5]                 +0.3  [+0.4,+0.3]                 -0.3  [-0.2,-0.5]
decoy                                        +0.2  [-0.2,+0.7]                 -1.2  [-1.8,-0.6]                 +0.9* [+1.5,+0.3]                 +0.1  [-0.1,+0.2]
reflex                                    -75.5* [-75.0,-76.0]                 +2.1  [+3.3,+0.9]              -59.2* [-58.6,-59.8]                 +0.1  [+0.2,-0.1]
reflex_near                               -28.4* [-27.6,-29.3]                 +8.3* [+7.6,+9.1]                 -2.4* [-2.4,-2.4]                 -0.6  [-0.8,-0.5]
reflex_call                                  +1.1  [+1.8,+0.4]                 -0.0  [-1.3,+1.3]                 -1.0* [-0.5,-1.5]                 -0.6  [-0.3,-1.0]

### {arm} against `reflex_call` (relative change in %, mean of seeds 1,2 [per seed]; flag: |change| > 2h with h = null half-width 2v2)
arm                                    storm deaths/agent-step                 births/agent-step                           n_alive                       mean energy
reflex                                    -75.8* [-75.5,-76.1]                 +2.1  [+4.7,-0.4]              -58.8* [-58.4,-59.1]                 +0.7  [+0.5,+0.9]
reflex_near                               -29.2* [-28.9,-29.6]                 +8.4* [+9.1,+7.7]                 -1.4* [-1.9,-0.9]                 +0.0  [-0.5,+0.5]

### {arm} against `none` (relative change in %, mean of seeds 1,2 [per seed]; flag: |change| > 2h with h = null half-width 2v2)
arm                                    storm deaths/agent-step                 births/agent-step                           n_alive                       mean energy
reflex_g0.5                               -69.8* [-69.7,-69.8]              +13.2* [+12.8,+13.6]              -42.1* [-40.2,-44.1]                 +0.8  [+0.9,+0.6]
reflex_g0.25                              -41.9* [-40.0,-43.9]                 +1.4  [-3.9,+6.8]              -19.6* [-19.4,-19.8]                 -0.5  [-0.7,-0.3]
reflex_g0.1                               -16.0  [-15.4,-16.5]                 +3.8  [-0.2,+7.8]                 -5.7* [-5.7,-5.7]                 -0.4  [-0.7,-0.2]
reflex_near_g0.5                          -11.3  [-10.2,-12.5]                 +4.7  [+4.7,+4.7]                 -0.1  [+0.0,-0.3]                 -0.4  [-0.5,-0.3]
reflex_near_g0.25                            +2.5  [+2.2,+2.9]                 +2.5  [+1.7,+3.3]                 +1.4* [+1.4,+1.4]                 -0.7  [-0.8,-0.5]
reflex_near_g0.1                             +5.6  [+6.0,+5.2]                 +1.5  [+0.9,+2.2]                 +1.8* [+1.8,+1.8]                 -0.5  [-0.5,-0.5]
reflex_m0.2                               -64.1* [-63.8,-64.4]                 +3.8  [+4.5,+2.9]              -45.4* [-44.9,-45.9]                 +0.0  [-0.2,+0.3]
reflex_m0.5                               -42.4* [-42.1,-42.6]                 -7.3* [-9.1,-5.5]              -28.3* [-28.9,-27.8]                 -1.1  [-1.1,-1.0]

### {arm} against `reflex_call` (relative change in %, mean of seeds 1,2 [per seed]; flag: |change| > 2h with h = null half-width 2v2)
arm                                    storm deaths/agent-step                 births/agent-step                           n_alive                       mean energy
reflex_g0.5                               -70.1* [-70.3,-70.0]              +13.2* [+14.4,+12.1]              -41.5* [-39.9,-43.2]                 +1.4* [+1.2,+1.6]
reflex_g0.25                              -42.6* [-41.1,-44.2]                 +1.4  [-2.6,+5.5]              -18.8* [-19.1,-18.5]                 +0.1  [-0.4,+0.6]
reflex_g0.1                               -16.9  [-16.9,-16.9]                 +3.8  [+1.2,+6.4]                 -4.7* [-5.3,-4.2]                 +0.2  [-0.4,+0.8]
reflex_near_g0.5                          -12.3  [-11.8,-12.9]                 +4.7  [+6.1,+3.4]                 +0.9* [+0.5,+1.2]                 +0.2  [-0.2,+0.7]
reflex_near_g0.25                            +1.4  [+0.4,+2.4]                 +2.5  [+3.1,+1.9]                 +2.4* [+1.9,+2.9]                 -0.0  [-0.5,+0.5]
reflex_near_g0.1                             +4.4  [+4.1,+4.7]                 +1.6  [+2.2,+0.9]                 +2.8* [+2.3,+3.4]                 +0.1  [-0.2,+0.5]
reflex_m0.2                               -64.5* [-64.4,-64.6]                 +3.8  [+6.0,+1.6]              -44.9* [-44.7,-45.0]                 +0.7  [+0.1,+1.3]
reflex_m0.5                               -43.0* [-43.2,-42.8]                 -7.3* [-7.9,-6.7]              -27.6* [-28.6,-26.6]                 -0.4  [-0.8,-0.0]

on_frac (calls_oracle_on_frac) by arm:
  reflex                 [0.214, 0.202]
  reflex_near            [0.07, 0.065]
  reflex_g0.5            [0.48, 0.438]
  reflex_g0.25           [0.697, 0.683]
  reflex_g0.1            [0.769, 0.765]
  reflex_near_g0.5       [0.099, 0.093]
  reflex_near_g0.25      [0.119, 0.114]
  reflex_near_g0.1       [0.129, 0.124]
  reflex_m0.2            [0.168, 0.144]
  reflex_m0.5            [0.095, 0.089]
```

Reading:
* **First clause.** `reflex` lowers storm deaths per agent-step by 75.8 % (both seeds -75.5, -76.1 against `reflex_call`, -75.0, -76.0 against `none`). 2h = 24 % (four null forks) or 15 % (six). **PASS.**
* **Net fitness falls.** Under `reflex` the population is 59 % smaller (4 775 and 4 655 against 11 534 and 11 566), births per agent-step +2 % (not different), mean energy unchanged. `reflex` pushes 21 % of the alive agent-steps (`calls_oracle_on_frac` 0.21) at `thrust_max` and the carrying capacity collapses, as the K2 review found on founders (85 % and 55 %).
* `reflex_call` (calls at 0.5, no force) changes nothing measurable: storm deaths +1.1 % (+1.8, +0.4), `n_alive` -1.0 %. The reduction is the force, not the calling.
* **`reflex_near`** (near block only, 7 % of agent-steps pushed): storm deaths -28.4 %, `n_alive` -2.4 %, births per agent-step +8.3 %, energy -0.6 %. It is the only oracle setting with a large storm reduction and a small population cost. The K2 review saw -50 % on founders.
* **The curve in the gain** (`oracle_gain`, storm deaths / `n_alive`, against `none`): `reflex` 1.0: -75.5 % / -59.2 %; 0.5: -69.8 % / -42.1 %; 0.25: -41.9 % / -19.6 %; 0.1: -16.0 % / -5.7 %.
  `reflex_near` 1.0: -28.4 % / -2.4 %; 0.5: -11.3 % / -0.1 %; 0.25: +2.5 % / +1.4 %; 0.1: +5.6 % / +1.8 %. `calls_oracle_on_frac` rises as the gain falls (0.21, 0.46, 0.69, 0.77 for `reflex`), because a weaker push keeps the agent inside the zone longer. No gain removes the population cost of the large reductions; the ratio of population lost to storm deaths avoided is 0.78 (gain 1), 0.60, 0.47, 0.36 (gain 0.1).
* **`oracle_min`** (hearers set at heard >= m, gain 1): 0.05: -75.5 % / -59.2 %; 0.2: -64.1 % / -45.4 %; 0.5: -42.4 % / -28.3 % (births per agent-step -7.3 %). The same trade.
* **What R licenses.** R is an upper bound on exposure reduction. Because the population falls with every large reduction, the Revision 2 clause "R >= 4 x 5 % with net fitness not falling" is met only by `reflex_near` at gain 1 (28 %, and `n_alive` -2.4 % is itself flagged against the unpaired 2h = 0.8 %; with births +8 %). The refutation clause of 7.4 is therefore available at most through `Rn`, not `R`.
* R minus Rn is not F minus N (R hears the far annulus only, Rn the near block only, and F both). `reflex` and `reflex_call` also force the `hurt` channel to 0.5.

## 7. Null forks, noise, and the 5 % rule

**Unpaired null (the spec's: same checkpoint, `run.seed` differs).** Storm deaths in 20 000 steps for seeds 1 to 6: 14 424, 13 698, 13 554, 13 205, 13 723, 13 810 (rate 5.7e-5 to 6.3e-5 per agent-step).
Coefficient of variation of the rate: 3.8 % (seeds 1 to 4) and 3.0 % (seeds 1 to 6). Half-widths (t with 3 and 5 df): for the contrast of m forks against m forks of 20 000 steps, h = 17.1 / 12.1 / 9.9 / 8.6 % for m = 1 to 4 (four forks) and 10.8 / 7.6 / 6.2 / 5.4 % (six forks; 4.5 % for 6 against 6).
Block structure: the autocorrelation of the across-fork deviation of storm deaths per 500 steps (six forks) is 0.53, 0.25, 0.15, 0.10, -0.01, -0.11 at lags 1 to 6 (500 to 3 000 steps): it is gone after 2 500 steps. Blocks of 2 500 steps are nearly independent (6 blocks of 5 000 steps are a conservative choice, 12 blocks of 2 500 the draft).

```
null forks: ['P2_none_s1', 'P2_none_s2', 'P2_none_s3', 'P2_none_s4']
storm deaths per fork: [14424 13698 13554 13205]  rate/agent-step: [6.253e-05 5.922e-05 5.873e-05 5.714e-05]
CV of fork rate (20k steps): 3.8%  (sd of log: 3.8%)
pairwise 1 - r_i/r_j (%): [(1, 2, np.float64(-5.6)), (1, 3, np.float64(-6.5)), (1, 4, np.float64(-9.4)), (2, 3, np.float64(-0.8)), (2, 4, np.float64(-3.6)), (3, 4, np.float64(-2.8))]
  contrast of 1 vs 1 forks (20k steps): sd 5.4%, 95% half-width h = 17.1%
  contrast of 2 vs 2 forks (20k steps): sd 3.8%, 95% half-width h = 12.1%
  contrast of 3 vs 3 forks (20k steps): sd 3.1%, 95% half-width h = 9.9%
  contrast of 4 vs 4 forks (20k steps): sd 2.7%, 95% half-width h = 8.6%
  block   500: 40 blocks; mean over blocks of across-fork relative sd 14.7%; relative sd of the fork-mean over blocks 8.8%
  block  1000: 20 blocks; mean over blocks of across-fork relative sd 13.4%; relative sd of the fork-mean over blocks 8.2%
  block  2500: 8 blocks; mean over blocks of across-fork relative sd 11.2%; relative sd of the fork-mean over blocks 7.3%
  block  5000: 4 blocks; mean over blocks of across-fork relative sd 9.5%; relative sd of the fork-mean over blocks 5.1%
  block 10000: 2 blocks; mean over blocks of across-fork relative sd 5.0%; relative sd of the fork-mean over blocks 3.5%
  ACF of across-fork deviation series (per 500 steps), lags 1..6, mean over forks: [0.58, 0.27, 0.21, 0.16, 0.01, -0.15]
  ACF of a fork's own rate series (per 500), lags 1..6: [0.59, 0.29, 0.23, 0.16, -0.03, -0.17]

six none forks (seeds 1-6):
null forks: ['P2_none_s1', 'P2_none_s2', 'P2_none_s3', 'P2_none_s4', 'P2_none_s5', 'P2_none_s6']
storm deaths per fork: [14424 13698 13554 13205 13723 13810]  rate/agent-step: [6.253e-05 5.922e-05 5.873e-05 5.714e-05 5.927e-05 5.970e-05]
CV of fork rate (20k steps): 3.0%  (sd of log: 2.9%)
pairwise 1 - r_i/r_j (%): [(1, 2, np.float64(-5.6)), (1, 3, np.float64(-6.5)), (1, 4, np.float64(-9.4)), (1, 5, np.float64(-5.5)), (1, 6, np.float64(-4.7)), (2, 3, np.float64(-0.8)), (2, 4, np.float64(-3.6)), (2, 5, np.float64(0.1)), (2, 6, np.float64(0.8)), (3, 4, np.float64(-2.8)), (3, 5, np.float64(0.9)), (3, 6, np.float64(1.6)), (4, 5, np.float64(3.6)), (4, 6, np.float64(4.3)), (5, 6, np.float64(0.7))]
  contrast of 1 vs 1 forks (20k steps): sd 4.2%, 95% half-width h = 10.8%
  contrast of 2 vs 2 forks (20k steps): sd 3.0%, 95% half-width h = 7.6%
  contrast of 3 vs 3 forks (20k steps): sd 2.4%, 95% half-width h = 6.2%
  contrast of 4 vs 4 forks (20k steps): sd 2.1%, 95% half-width h = 5.4%
  block   500: 40 blocks; mean over blocks of across-fork relative sd 14.0%; relative sd of the fork-mean over blocks 7.3%
  block  1000: 20 blocks; mean over blocks of across-fork relative sd 12.5%; relative sd of the fork-mean over blocks 6.6%
  block  2500: 8 blocks; mean over blocks of across-fork relative sd 10.2%; relative sd of the fork-mean over blocks 5.9%
  block  5000: 4 blocks; mean over blocks of across-fork relative sd 7.9%; relative sd of the fork-mean over blocks 4.7%
  block 10000: 2 blocks; mean over blocks of across-fork relative sd 4.0%; relative sd of the fork-mean over blocks 4.1%
  ACF of across-fork deviation series (per 500 steps), lags 1..6, mean over forks: [0.53, 0.25, 0.15, 0.1, -0.01, -0.11]
  ACF of a fork's own rate series (per 500), lags 1..6: [0.52, 0.27, 0.17, 0.12, 0.0, -0.09]
```

**The 5 % rule: "valid only if h <= 2.5 %".** For the unpaired null it **fails**: h = 12 % (2 against 2, 20 000 steps), 5 to 8 % for 3 against 3 forks of 30 000 steps (six or four null forks; scaling as T^-1/2, not measured). Reaching 2.5 % needs about 19 forks of 20 000 steps (12.7 of 30 000) per arm, i.e. about 380 000 fork-steps per arm, one hour of GPU per arm pair: not run (outside the plan; the paired noise below is a better description of the M1 contrasts).

**Same-seed noise.** The knock-out contrasts of M1 are paired by seed, and forks with the same `run.seed` share the weather exactly (verified: `wind_max_l1` and `rain_total` identical to all digits across controls). So the relevant noise is smaller. Two estimates:
(1) a **lower bound**: `pnull` = a `none` fork with `comm.cost = 32.001` (the call cost dither flips a few payments, the trajectories separate, the weather and the random stream stay the same): paired difference 0.47 % sd over six seeds, h 1.2 % for one pair; this is a perturbation under common random numbers and underestimates the noise of a real treatment;
(2) the seed-to-seed spread of the paired contrast of the four real treatments, pooled (df 12): **sd 1.40 %**, h (95 %) for the mean of m seed pairs of 20 000 steps: 3.0 / 2.2 / 1.8 / 1.5 / 1.2 % for m = 1 / 2 / 3 / 4 / 6. This includes any treatment effect that differs between seeds, so it is if anything high. **h <= 2.5 % holds from 2 pairs on.**

```
paired nulls (same seed => same weather; the pnull has comm.cost 32.001): ['s1', 's2', 's3', 's4', 's5', 's6']
storm deaths/agent-step    pair diffs % [ 1.12 -0.08  0.43 -0.04  0.48 -0.09]  mean +0.30  sd 0.47  sd of one-vs-one contrast; rms 0.53
births/agent-step          pair diffs % [-1.48 -0.61  0.5  -3.49 -1.98 -0.78]  mean -1.31  sd 1.36  sd of one-vs-one contrast; rms 1.80
n_alive                    pair diffs % [-0.33 -0.18  0.16  0.   -0.26 -0.08]  mean -0.12  sd 0.18  sd of one-vs-one contrast; rms 0.20
mean energy                pair diffs % [ 0.24 -0.04 -0.31 -0.79 -0.39 -0.39]  mean -0.28  sd 0.35  sd of one-vs-one contrast; rms 0.43

storm-rate paired null (n=6 pairs, df=5, t=2.571): sd of a single paired contrast (20k steps) = 0.47%
  mean of 1 paired contrasts: sd 0.47%  95% h = 1.22%
  mean of 2 paired contrasts: sd 0.34%  95% h = 0.86%
  mean of 3 paired contrasts: sd 0.27%  95% h = 0.70%
  mean of 4 paired contrasts: sd 0.24%  95% h = 0.61%
paired log-ratio per-500 interval: sd 7.1%; ACF lags 1..8: [0.05, -0.14, 0.04, -0.0, -0.05, -0.0, -0.1, -0.03]
  block   500 steps: 40 blocks/fork; sd of block log-ratio 7.1%; lag-1 ACF 0.05; sd of the 20k mean log-ratio 0.45%
  block  1000 steps: 20 blocks/fork; sd of block log-ratio 5.3%; lag-1 ACF -0.16; sd of the 20k mean log-ratio 0.44%
  block  2500 steps:  8 blocks/fork; sd of block log-ratio 3.0%; lag-1 ACF -0.13; sd of the 20k mean log-ratio 0.44%
  block  5000 steps:  4 blocks/fork; sd of block log-ratio 2.2%; lag-1 ACF -0.36; sd of the 20k mean log-ratio 0.48%
  block 10000 steps:  2 blocks/fork; sd of block log-ratio 1.1%; lag-1 ACF nan; sd of the 20k mean log-ratio 0.49%
```

```
paired by seed against `none` (same weather); relative change %, per seed, mean, sd, 95% t-interval of the mean
nofar        storm deaths/agent-step  n=6 [ 1.6  4.   3.1  0.8  2.2 -0.3]  mean +1.91  sd 1.54  95% CI [+0.29, +3.53]
nofar        births/agent-step        n=6 [ 4.   4.5  3.5 -1.   2.   2.5]  mean +2.57  sd 1.98  95% CI [+0.48, +4.65]
nofar        n_alive                  n=6 [0.5 0.4 1.2 1.1 0.5 0.6]  mean +0.71  sd 0.33  95% CI [+0.36, +1.05]
nofar        mean energy              n=6 [-0.6 -0.4  0.  -0.6 -0.5 -0.4]  mean -0.40  sd 0.22  95% CI [-0.64, -0.17]
deaf         storm deaths/agent-step  n=2 [3.3 4.2]  mean +3.74  sd 0.64  95% CI [-2.03, +9.50]
decoy        storm deaths/agent-step  n=6 [-0.2  0.7  2.7 -1.7  0.6  0.3]  mean +0.39  sd 1.42  95% CI [-1.10, +1.88]
decoy        births/agent-step        n=6 [-1.8 -0.6 -2.3 -4.4 -2.3 -2.7]  mean -2.35  sd 1.24  95% CI [-3.65, -1.05]
decoy        n_alive                  n=6 [1.5 0.3 1.1 0.6 0.7 1.2]  mean +0.88  sd 0.42  95% CI [+0.44, +1.33]
decoy        mean energy              n=6 [-0.1  0.2  0.1 -0.4  0.  -0.2]  mean -0.06  sd 0.20  95% CI [-0.27, +0.15]
reflex_call  storm deaths/agent-step  n=2 [1.8 0.4]  mean +1.12  sd 0.97  95% CI [-7.60, +9.84]
pnull        storm deaths/agent-step  n=6 [ 1.1 -0.1  0.4 -0.   0.5 -0.1]  mean +0.30  sd 0.47  95% CI [-0.19, +0.80]

pooled within-arm seed sd of the paired storm-rate contrast over nofar, deaf, decoy, reflex_call: 1.40% (df 12)
  mean of 1 seed pairs (20k-step forks): h = t_12 * sd / sqrt(m) = 3.04%
  mean of 2 seed pairs (20k-step forks): h = t_12 * sd / sqrt(m) = 2.15%
  mean of 3 seed pairs (20k-step forks): h = t_12 * sd / sqrt(m) = 1.76%
  mean of 4 seed pairs (20k-step forks): h = t_12 * sd / sqrt(m) = 1.52%
  mean of 6 seed pairs (20k-step forks): h = t_12 * sd / sqrt(m) = 1.24%
```

**Decision for the freeze:** the pre-registration has to say which noise its 5 % rule uses. The draft states the paired design (at least 3 seed pairs) and records that the spec's unpaired null does not support 5 %.

## 8. What the pilot says about the contrasts M1 will test

Paired by seed against `none`, 20 000 steps, six seeds (relative change %, positive = more storm deaths without the feature):

| contrast | storm deaths per agent-step | births per agent-step | `n_alive` | mean energy |
|---|---|---|---|---|
| `nofar` (far block zeroed) | +1.91, 95 % CI [+0.29, +3.53] (per seed +1.6 +4.0 +3.1 +0.8 +2.2 -0.3) | +2.57 [+0.48, +4.65] | +0.71 [+0.36, +1.05] | -0.40 [-0.64, -0.17] |
| `decoy` | +0.39 [-1.10, +1.88] | -2.35 [-3.65, -1.05] | +0.88 [+0.44, +1.33] | -0.06 [-0.27, +0.15] |
| `deaf` (seeds 1, 2) | +3.74 (+3.3, +4.2) | +2.0 | +0.3 | -0.3 |
| `reflex_call` (seeds 1, 2) | +1.12 (+1.8, +0.4) | 0.0 | -1.0 | -0.6 |

* F against `nofar`: the far block of this evolved population lowers storm deaths by **1.9 %** (the relative reduction `1 - 1/1.0191`), 0.3 to 3.4 % at 95 %. That is below the 5 % smallest effect of interest ("no effect of interest" band) but not zero. F against `decoy` is 0.4 % (not distinguishable from 0). The F in question is one seed after 80 000 steps (a single lineage, section 2); M1 runs 300 000 steps and three seeds.
* **Bound on the selection pressure.** Storm deaths are 3.7 % of all deaths in the chosen world. Eliminating every storm death would lower the death rate by 3.7 %; the 5 % effect of interest is 0.18 % of deaths, and the pilot's F-over-`nofar` difference is 0.07 %. This follows from G1's own band (3 to 10 %): at the top of the band the bound is 10 %. The bound is on any evolved response, whatever the mechanism.
* The hearing contrasts have the same sign for `nofar` and `deaf` (more storm deaths without hearing), for two to six seeds. The sign is consistent; the size is small.

## 9. M3: the altitude stretch (the far table is built on the floor sphere)

`P1_u05_x3` at step 80 000 (11 876 alive); `P1_u05_x4` gives the same picture (`m3_P1_u05_x4_80000.txt`). Altitude of alive agents: mean 13.4 of 24 (p10 2.6, p90 22.9), so the horizontal distances of the 3D shell are stretched against the floor sphere by (48 + z) / 48 = **1.28 on average** (1.05 at z = 2, 1.5 at the lid). The storm callers are low (mean altitude 5.5, stretch 1.11); the hurt callers 12.3.
Grid (floor distance) against the exact sum with the physical 3D distance, storm channel, audible = heard >= 0.02:

| | exact on the floor sphere | exact with 3D distances |
|---|---|---|
| hearers audible, grid | 0.859 | 0.859 |
| hearers audible, exact | 0.864 | **0.625** |
| grid inaudible but exact audible / phantom in grid, share of the union | 2.0 % / 1.5 % | 0.3 % / **27.5 %** |
| median symmetric error of the intensity (union) | 11 % | **74 %** |
| median error with 3 to 10 callers in the annulus | 28 % | 87 % |
| median error with 10 to 30 callers | 14 % | 48 % |
| median error with 30 or more callers | 7 % | 19 % |

The same for the hurt channel: grid audible 0.483, floor exact 0.479, 3D exact **0.267**; phantoms 48.8 % of the union in 3D terms. The grid's accuracy against its own definition (the floor sphere) is as the K2 tables say (11 % median for the storm channel; 21 % for the hurt channel). In physical terms the far block is audible to 38 % more hearers than a 3D reference would make it (0.859 against 0.625): a hearer 6 to 8 horizontal units away on the floor but at a different altitude is "far-audible" without being in the near block. `far_range = 18` is about 14 in the shell on average (12 to 25 % less, as the K2 review said). It does not bias the arm contrasts (every arm has the same geometry); it changes what "far" means in the pre-registration. The table was not rebuilt (the coordinator decides at the freeze).

## 10. What surprised me

1. The stable-looking `far_xref` is not a property of the sound field but of the evolved population (3.0 and 4.0 give different worlds, with different storm-channel medians and different shares of storm deaths).
2. The cost sweep flips the used channel (section 4).
3. Common random numbers: a cost change of 0.001 on 32 leaves the storm deaths within 0.5 % (sd) of the unperturbed fork, while a seed change moves them by 3 to 4 %.
4. `reflex_near` alone is a decent oracle with almost no population cost, and `reflex` is not: the "free full-thrust reflex" of the spec empties the planet.
5. Storm deaths are a small part of mortality at the dial G1 allows (3.7 %), which bounds the benefit of any response (section 8).
6. The storm share of deaths per 10 000 steps in the chosen run dips to 2.0 % in steps 20 000 to 30 000; G1 at a different window of the same run would have failed.

## 11. M1: cost, and the pilot's recommendation (the coordinator decides)

Measured speed with two processes: 101 steps/s each, 202 aggregate; alone 153. M1 as planned (spec 7.2): F x 3 seeds, D x 1, hazard-off x 1, each 300 000 steps = 1.5 M steps; knock-out forks {none, nofar, deaf, decoy} x 3 seeds x 30 000 = 0.36 M; total 1.86 M steps. At 202 steps/s aggregate: **2.6 GPU-h wall** (about 5.1 process-hours); alone at 153: 3.4 h; at the spec's 100 steps/s planning rate 5.2 h. Voice logs and the K3 analysis are not in these numbers (the spec adds 1.5 h for logs). If the paired design asks for more than three seed pairs, each extra seed pair of 30 000 steps costs 5 minutes.

**Recommendation: GO for M1, conditional on four decisions at the freeze**, because the pilot gates G1 to G5 pass and M1 is the experiment that can answer the question; the pilot does not predict a positive result.
1. Pre-register the **paired (same-seed) design** for the fitness clause and record that the spec's unpaired null does not support the 5 % rule (section 7). Block length 2 500 steps.
2. Accept that the **expected effect is below the 5 % line** (section 8: 1.9 % after 80 000 steps) and that the selection pressure is bounded by 3.7 % (section 8). If the owner wants a larger bound, the dial would have to go down to the top of the G1 band, which the ladder did not reach (0.05 is the lowest value run; 0.045 is the natural next one).
3. Decide what to do with the **channel flip** (section 4): O1 may fail in a seed because the population picks the hurt channel. Nothing in the engine needs to change for M1; the D reference and the hazard-off arm will show it.
4. Decide on the **M3 altitude stretch** (section 9): record it in the pre-registration, or rebuild the table at a mid-shell radius (`far_range` >= 22.4 by the 4-column rule on `mini3s`, a design change, not done here).

No-go conditions I looked for and did not find: a gate failing, a collapsed world, a ledger violation, a launch error, a noise level that makes the 5 % unreadable in the paired design.

## 12. Not done, and limits

* One seed per dial and per cost; the evolved F of the forks is a single-lineage state.
* P1b at far_xref 4.0, not 3.0. The hazard-off P1 run used far_xref 0.7 and `storm_u0` 0.06 (the storm event's threshold) and was not rerun at 3.0.
* `deaf` and `reflex_call` have two seeds, `reflex`, `reflex_near` two seeds each, not six.
* The 0.045 dial was not run (G1 passed). No extension of the null forks to h <= 2.5 % (unpaired) was run: about 380 000 fork-steps per arm.
* Not measured: the scaling of the paired noise with fork length (30 000 against 20 000 steps); the `calls_open_<e>` statistics for the D reference; `far_audible` with the 0.02 floor in the analysis (K3).


## 13. Tests

Changed: `tests/test_calls_config.py::test_the_overlays_load_on_the_experiment_world` now pins `storm_u0 == 0.05`, `far_range == 18.0`, `far_xref == 3.0`, `D_IN == 95` and `layout_features == "comm2@1,calls2@1"` for
`--preset mini3s --overlay stormsph22h --overlay comm --overlay calls`. `ruff check` and `ruff format --check` clean. `pytest -m "not slow" -n 6` (Metal, 10 min 44 s): 6 766 passed, 38 skipped, 1 xfailed, 2 failed.
The two failures are outside this change: `tests/test_stream.py::test_reader_closing_early_exits_zero_without_a_traceback` passes when rerun alone, and
`tests/test_golden_run.py::test_a_cold_child_sees_no_kernel_cache_although_the_shared_and_private_ones_are_warm` fails alone too (the probe of a child process on the `tiny` preset finds no new kernel library; it loads no overlay);
it was not run on the base commit, so whether it is pre-existing is not established here. The K2 report's known failure (`test_lifecycle_compiles_and_matches_eager`) passed in this run.

## 15. Coordinator's check and extension (supersedes the chosen inputs of section 0)

Sections 0 to 14 are the pilot driver's report on `storm_u0` 0.05 and `far_xref` 3.0 and stay as written. The numbers of `nofar` and `decoy` were recomputed from the raw metrics and agree. The check found two gaps.

**Gap 1, the noise.** The paired sd of 1.4 % comes from forks of one evolved lineage. M1 has one fork per F seed, so its pairs come from different lineages, and the variation of the effect between lineages was not measured. Section 4 already hinted at it (the channel flip). Blocks are also too narrow: the within-seed block SE of a 20 000-step contrast is 0.9 to 1.1 % against a seed-to-seed sd of 1.4 to 1.5 %.

**Gap 2, the dial.** 3.7 % of deaths bounds any benefit, and the share dipped below the band for 30 000 steps. A rough sweep estimate: lifespan about 650 steps, so about 460 to 670 generations in 300 000 steps; a variant that removes 5 % of storm deaths has s of about 0.2 %, and needs thousands of generations.

**Ladder at far_xref 3.0 (seed 0, 80 000 steps), then far_xref at 0.04.**

| run | storm_u0 | far_xref | storm % of deaths, last 20k | open hurt / storm | call rate hurt / storm | due % | n_alive |
|---|---|---|---|---|---|---|---|
| `P1_u045_x3` | 0.045 | 3 | 4.55 | 0.087 / 0.105 | 0.25 / 0.58 | 1.9 | 9 895 |
| `P1_u04_x3` | 0.04 | 3 | 5.85 | 0.097 / 0.150 | 0.47 / 0.98 | 4.6 | 9 374 |
| `P1_u04_x6` | 0.04 | 6 | 7.23 | 0.125 / 0.157 | 0.49 / 0.39 | 1.8 | 10 073 |
| `P1_u04_x10` | 0.04 | 10 | 4.48 | 0.125 / 0.132 | 0.72 / 0.88 | 3.6 | 8 403 |

At 0.04 and far_xref 3 the storm channel saturates (median heard 0.92). Median heard at 6: 0.12 / 0.38; at 10: 0.28 / 0.32. Chosen without a further run: 8.0, between the two. **The overlays now carry 0.04 and 8.0.**

**Three independent lineages at 0.04 and 8.0** (`L10`, `L11`, `L12`, run seeds 10 to 12, 80 000 steps, last 20 000):

| lineage | storm % of deaths | open hurt / storm | call rate hurt / storm | due % | n_alive | median heard 60k / 80k |
|---|---|---|---|---|---|---|
| L10 | 6.13 | 0.092 / 0.147 | 0.69 / 0.71 | 3.63 | 12 002 | 0.41 / 0.44 |
| L11 | 6.46 | 0.090 / 0.146 | 0.64 / 0.60 | 2.65 | 12 220 | 0.40 / 0.35 |
| L12 | 6.32 | 0.151 / 0.146 | 0.84 / 0.45 | 3.39 | 12 062 | 0.20 / 0.22 |

G1, G2, G3 pass in all three; the storm share is inside the band in every 10 000-step window after the first. Ledger error 0. L12 is below the median-heard target. No lineage lost the storm channel.

**Forks** (20 000 steps from step 80 000, seeds 1 and 2 of each lineage; log ratio against `none` of the same lineage and seed, %):

| contrast | L10 | L11 | L12 | mean | sd between lineages | 95 % h, 3 lineages |
|---|---|---|---|---|---|---|
| `nofar` storm deaths / agent-step | -1.5 (-0.3, -2.6) | +5.0 (+5.6, +4.4) | +1.3 (+2.2, +0.4) | +1.6 | 3.2 | 8.0 |
| `decoy` storm deaths / agent-step | +4.1 (+4.5, +3.8) | +12.3 (+12.4, +12.2) | +4.9 (+5.8, +3.9) | +7.1 | 4.5 | 11.2 |
| `nofar` births / agent-step | -4.2 | +1.1 | +1.4 | -0.5 | 3.2 | 7.9 |
| `decoy` births / agent-step | -3.0 | -1.1 | +2.9 | -0.4 | 3.0 | 7.5 |
| `nofar` `n_alive` | -0.9 | +2.2 | -1.3 | 0.0 | 1.9 | 4.7 |
| `decoy` `n_alive` | +0.7 | +4.5 | +0.1 | +1.8 | 2.4 | 5.9 |

The sd of one paired contrast within a lineage is 1.1 %; between lineages it is 3.2 to 4.5 %. The h for n lineages with sd 3.2 is 8.0 % (3), 3.4 % (6), 2.5 % (9). An sd from three lineages is itself rough (its 95 % interval runs from about half to about six times the value), and it was measured at 80 000 steps, not 300 000.

Reading: `decoy` raises storm deaths in all six pairs, by 4 to 12 %. `decoy` replaces both hearing blocks, and its calls land where no event is, so this may be misinformation as much as the loss of a warning (spec 7.3). The population under `decoy` is not lower, so it is not a net gain to F. Over lineages the one-sided lower bound is -0.5 %, narrowly below 0. `nofar` is not distinguishable from 0 and changes sign between lineages.

**Oracle at 0.04 and 8.0** (fork seed 1, per lineage, against `none`, storm deaths / `n_alive`): `reflex` -30.7 / -80.1, -27.2 / -79.3, -53.1 / -69.2 (`calls_oracle_on_frac` 0.31 to 0.42); `reflex_near` -32.3 / -4.2, -27.4 / -2.9, -25.9 / -10.2 (0.14 to 0.15); `reflex_call` -3.0 / -0.2, -6.9 / -1.4, +1.0 / -1.2. G5's first clause holds in each lineage against the pilot's 2h of 15 to 24 %; net fitness falls, so R licenses no refutation.

**Budget of the extension:** 1.1 M steps, 1.62 h of GPU wall time. In all 4.1 GPU-h of wall time against the ceiling of 5, and about 7.6 process-hours.

Not done: `deaf` at the new values, null forks on the same design as the pilot's section 7, hazard-off and the shipped dial at the new values (M1 has hazard-off), one more far_xref check at 8 in other seeds, runs past 80 000 steps. `tests/test_golden_run.py`'s cold-child test is the known failure on the GPU default device.
