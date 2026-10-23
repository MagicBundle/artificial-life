# G3b on the M5 Pro: Metal purity, the bench gate with powers, 2x, an 18,000-step played planet and the rendered decide half

The Mac day of chapters, alerts, the live report and direct powers (`docs/DESIGN.md` §34, Review-log row I54; `docs/GAMES.md` §9 G3b;
specification `g3b/spec.md` revision 2, §10 and §11), run 2026-10-23 (02:45 to 05:11 local) on the M5 Pro (MLX 0.32.3, Python 3.12.11,
Metal kNN kernel) on the branch `games/g3b` at `019fdde` (G3b merged with `origin/main` 3a14aa6, the G3a squash 9f84f7c plus #36, and the
sync review fixes). **No code changed and nothing was committed during the Mac day.** This note is the record: what was run, what the numbers
are, three defects the run found, and what is left to the owner. Nobody has clicked through the window by hand and no person has
played a world in the `player` mode (§10).

## Answer

- **The gate of DESIGN §34.8 passes on every item the bench can measure, in every arm.** On the planet configuration (33,796 alive at the end
  of the arms, 32,354 with powers), at 1x: achieved 0.99810x (`rules`), 0.99805x (`player`) and 0.99753x (`player` with the two scripted
  powers) over 313.0 to 313.2 s, `late_steps` 0, `step_ms` p99 42.1, 42.1 and 43.1 against the limit of 66; at `max` 3.822x, 3.821x and
  3.751x against 2.5. Click-to-applied p95 is 0.102 s at 1x (limit 0.5). Every arm without powers ends at the plain run's digest
  (`6bbe3bcb33b3...`); the arms with powers end at the replay of their own log (`bc69570f5f5d...`), which differs from the plain run.
- **The report-time item (at most 500 ms) was not measured by the bench, and the number it printed is not a report time (§5, finding 2).** Its 2.5 to
  2.7 ms is the time of an arrival notice, because an arm has no history before it. The report was measured in the played session instead: the
  `RPRT` arrived 0.047, 0.029 and 0.030 s after the chapter card opened, and the save finished 0.68 to 1.01 s after it, so every report was
  readable inside the stall.
- **2x holds.** `rules`, `player` and `player` with powers: 1.9923x, 1.9923x and 1.98994x, `late_steps` 0, `step_ms` p99 28.7, 28.9 and 29.0
  against a 34.7 ms slot, digests as at 1x, both powers applied (0.020 and 0.070 s click to applied). The spec names no 2x limit; the
  reading used here is in §6.
- **Metal purity: 34 test files, 1,393 passed, 1 skipped, 0 failed.** The skip needs a loaded model and no model call was made. That includes
  `tests/test_bench.py` (46), which G3a left open. Swift: `swift build`, `swift test` (AlifeCore 469 tests, AlifeControl 252), `GPURenderTests` 8,
  `GPUWorldTests` 14, `WorldRenderBench` 5; at 42,000 rows the draw GPU p50 is at most 2.72 ms (region) with 0 dropped.
- **The chronicle and the alert surface are inside the bands of spec §11 on one 18,001-step window of p0.** Chronicle 3.28 events a minute
  (G3a: 8.66; band 2 to 5), kin 2.04 (G3a: 5.99; band 1 to 3), alerts of importance 2 or more on the alert surface 0.38 a minute (band at most 0.6).
  Kin events are 43 of 69 live events and only 4 of the 13 events of importance 2 (G3a: 69 of 77).
- **Auto-pauses: 4 in 18,001 steps, at the limit, but not the intended four.** Three were chapter ends; the fourth was a population fall that
  followed the player's own Blight (finding 3). The spec's band says that must not pause; the stage-8 rule says a fall as deep as 15% pauses whatever
  the cause. The two texts disagree.
- **Three defects, none fixed here (§9).** (1) **The server numbers reports from step 0, not from the planet's origin:** the Report tab reads
  "Chapter 79" where the card and the bar read "Chapter 4". (2) The bench's report time is an arrival notice. (3) A deep fall after the player's own
  power pauses the world. Two more observations: Food was refused at the start of p0 by lever exclusivity (by design), and the chapter card is translucent over the map.
- **Not done:** the checkpoint on the external drive (the volume was not mounted), the second window of the play (a fork of `64k3d_s1`) and the 20,000th step
  (the play is 18,001 steps), the client's draw time beside a running engine, the 2D digest matrix on the GPU, `test_weather_w2_instruments` on Metal, and the
  first play-test by a person (D9). The precondition of spec §11 (the owner's click-through of the G3a window) is still open.

## 1. What was run

| step | what | where the result is |
|---|---|---|
| 1 | 34 test files separately on `Device(gpu, 0)`, `-m ""`, no `-p cpu_device` | §2 |
| 2 | `swift build`, `swift test`, `GPURenderTests`, `GPUWorldTests`, `WorldRenderBench` alone | §3 |
| 3 | The planet: a `cp -c` clone of G3a's `p0` (origin 450,000, 470,504 at the copy), an origin row, and a second clone advanced to 472,000 for the bench | §4 |
| 4 | `alife bench --play` at 1x, `max` and 2x, three director modes, with and without scripted powers | §5, §6 |
| 5 | `alife play p0 --rate 1 --director rules` for 18,001 steps with a strict scripted reader, then a quit and a resume | §7 |
| 6 | The decide half rendered offscreen from reader snapshots by the production `Renderer` and `NSHostingView` | §8 |

- **The planet.** `MD/planets/p0` is a `cp -c` clone of G3a's `p0` (the original was only read). G3a's `meta.json` has no `origin_row`
  (G3b writes one when it makes a planet), so one was written from the 450,000 checkpoint with the production `play.origin_row`: alive 44,912, clades 4,
  top clade share 0.4296 (`MD/add_origin_row.py`, `MD/logs/step3_origin_row.log`). This is a one-off edit of `meta.json`, not a fresh planet. With it the first chapter (456,000) builds, and its backlog report reads "4 storms reached
  named strength".
- **The bench source.** `MD/planets/p0_bench` is another clone advanced with `alife play --director rules --until 472000` (31 s). Its 472,000
  checkpoint puts the chapter end at 474,000 2,000 steps into the arm (the spec's "2,000 steps before a boundary"), 1,700 steps into the measured part.
- **Bench arms.** All arms resume at 472,000 with 300 warm-up steps and 4,500 measured (4,499 counted), cadence 250 / 250 / 6,000, `game.observe` on,
  a plain `runner.run` of the same world as the digest reference. Files in `MD/bench_{A,B,C,D}`; logs `MD/logs/step4_{A,B,C,D}_bench.{json,log}`, summarised by `MD/summ_bench.py`.
  Arm A: `rules` and `player` at 1x and `max`; B: `player` with `--play-powers` at 1x and `max`; C: `rules` and `player` at 2x; D: `player` with powers at 2x.
- **Files** (this note commits the small ones in `docs/results_g3b/`; the PNGs of §8, the logs, the snapshots and the planets stay in the scratch directory
  `scratchpad/g3b/macday`, called MD below): `bench_play_rules_player.json`, `bench_play_powers.json`, `bench_play_2x.json` and `bench_play_powers_2x.json` (MD/logs/`step4_{A,B,C,D}_bench.json`),
  `metal_purity_summary.txt` (MD/logs/`step1_summary.txt`), `world_render_bench_alone.json` (MD/render/`gpu_tests_out_bench_alone/world-render-bench.json`),
  `play18k_run1.json`, `play18k_resume500.json`, `play18k_events.jsonl`, `play18k_alerts.jsonl` and `play18k_plans.jsonl` (MD/play18k/), `play_reader_g3b.py`
  (the reader; needs the environment variables `WT`, `ALIFE_BIN` and `SNAPDIR`; a file-level `noqa` keeps it as it ran) and `summ_bench.py`.
  The 4,545 `CLCK` rows (396 KB, `run1.json.clck.jsonl`) are not committed. The JSON records carry the scratch paths they ran in.

## 2. Metal purity

Each file was run separately with `uv run pytest tests/FILE.py -m "" -q` on the default device (driver `MD/step1.sh`, summary `MD/logs/step1_summary.txt`, one log per file
in MD/logs/`step1_<file>.log`). 1,393 passed, 1 skipped, 0 failed. The skip is `test_analysis_world_report.py:1069`, which needs `--model` or `ALIFE_LLM_URL`; no model call was
made and the model loaded in LM Studio was left alone.

| file | passed | file | passed |
|---|---|---|---|
| `test_alerts` | 68 | `test_play_hook` | 22 |
| `test_analysis_world_report` | 71 (+1 skipped) | `test_play_planet` | 15 |
| `test_bench` | 46 | `test_play_powers` | 29 |
| `test_g3b_review` | 10 | `test_play_purity` | 15 |
| `test_game_purity` | 55 | `test_play_replay` | 6 |
| `test_game_report` | 94 | `test_play_sched` | 23 |
| `test_game_report_select` | 89 | `test_play_storm_persist` | 18 |
| `test_game_report_templates` | 103 | `test_play_storms` | 34 |
| `test_gui_sync` | 21 | `test_play_threads` | 8 |
| `test_kinthreads` | 35 | `test_player_channel` | 59 |
| `test_play_alerts` | 31 | `test_player_run` | 28 |
| `test_play_chapters` | 44 | `test_powers` | 36 |
| `test_play_chronicle` | 18 | `test_report_job` | 29 |
| `test_play_cli` | 19 | `test_report_v2` | 33 |
| `test_play_control` | 141 | `test_runner` | 16 |
| `test_play_envs` | 3 | `test_spatial_metal` | 106 |
| `test_play_focus` | 13 | `test_stream_records` | 55 |

- **What this closes of G3a's open list:** `tests/test_bench.py` on Metal (46 passed, including the player arm, the scripted powers and the replay-equality checks).
- **What it does not run:** `test_weather_w2_instruments.py`, and the G3a-only files `test_play_houses`, `test_play_identity`, `test_play_labels_v3`, `test_play_colors` and
  `test_game_tracers`. The 2D digest matrix as a script on the GPU was not run either
  (it was run on the CPU against 3a14aa6 at the sync, bit-identical).
- The subprocess tests of `test_play_cli` spawn the CPU on purpose, so only their in-process parts ran on Metal.

## 3. Swift

Logs `MD/logs/step2_*.log` and `step2_marks.txt`.

- `swift build` passed. `swift test` passed: AlifeCore 469 tests in 46 suites, AlifeControl 252 tests in 27 suites (G3a: 338 and 200).
- With `ALIFE_GPU_TESTS=1`: `GPURenderTests` 8 passed, `GPUWorldTests` 14 passed, `WorldRenderBench` 5 passed in 40.5 s.
- `WorldRenderBench` alone, 300 draws per scenario at 60 Hz, a debug build, offscreen, 0 dropped draws. Draw GPU time in ms:

| rows | scenario | GPU p50 | GPU p95 | GPU max | CPU encode p50 | CPU encode p95 | CPU encode max |
|---|---|---|---|---|---|---|---|
| 64 | planet | 0.65 | 0.66 | 0.66 | 0.07 | 0.15 | 2.60 |
| 64 | planet with points | 0.64 | 0.65 | 0.66 | 0.07 | 0.16 | 2.54 |
| 64 | region | 0.62 | 0.63 | 0.65 | 0.07 | 0.16 | 2.58 |
| 64 | organism | 0.62 | 0.63 | 0.70 | 0.07 | 0.16 | 2.56 |
| 42,000 | planet | 0.65 | 0.65 | 0.66 | 0.11 | 0.20 | 2.59 |
| 42,000 | planet with points | 1.01 | 1.02 | 1.02 | 0.11 | 0.20 | 2.61 |
| 42,000 | region | 2.72 | 2.76 | 2.77 | 0.11 | 0.23 | 2.64 |
| 42,000 | organism | 1.15 | 1.19 | 1.19 | 0.11 | 0.21 | 2.63 |

Source: `docs/results_g3b/world_render_bench_alone.json`, MD/logs/`step2_WorldRenderBench_alone.log`. The p50 values equal G3a's to within 0.01 ms (the same bench, the same render code); some p95 and maximum values are lower.
The region level at 42,000 rows is the largest, 2.76 ms at p95 against the gate of 25. The client's share beside a running engine (G3a §6) was not repeated.
The CPU encode maximum is again 2.5 to 2.6 ms in every scenario, a harness outlier as in G3a.

## 4. The planet and the cost of the loop

All numbers are from the arms of step 4 (`docs/results_g3b/bench_play_*.json`). The planet is G3a's `p0`, which has moved on: the arms end at 33,796 alive (32,354 in the arms
whose powers include a Strike at the chapter end), against 44,955 at the end of G3a's arms, so the observer, settle and frame costs are lower for a smaller population
and the two sets of numbers are not comparable at the decimal.

### 4.1 The arms at 1x and `max`

| arm | achieved | seconds | `step_ms` p50 | p99 | max | `late_steps` | `save_stall_steps` | digest |
|---|---|---|---|---|---|---|---|---|
| `rules`, 1x | 0.99810x | 313.0 | 30.0 | 42.1 | 595.6 | 0 | 15 | = plain |
| `player`, 1x | 0.99805x | 313.0 | 30.1 | 42.1 | 609.5 | 0 | 15 | = plain |
| `rules`, `max` | 3.822x | 81.7 | 18.5 | 23.7 | 590.6 | 0 | 0 | = plain |
| `player`, `max` | 3.821x | 81.8 | 18.5 | 23.6 | 593.7 | 0 | 0 | = plain |
| `player` + powers, 1x | 0.99753x | 313.2 | 30.7 | 43.1 | 766.5 | 0 | 20 | = replay |
| `player` + powers, `max` | 3.751x | 83.3 | 18.7 | 24.4 | 789.3 | 0 | 0 | = replay |

- The plain digest of both director modes is `6bbe3bcb33b366a1806d533e7e638a9b03a3254f4f99b6fe7dc0de7f517297bb` (33,796 alive); the replay digest of the arms with
  powers is `bc69570f5f5d00893d77e710a18f5ffb5db6e0359664ac2563fc99eff40f3d04` (32,354 alive), identical at 1x, `max` and 2x. The player channel unused changes nothing
  the bench can see, and a played world replays bit for bit on Metal.
- **The maximum step (590 to 800 ms) is the chapter-end step.** It carries the periodic save at the 474,000 mark; the bench's own save timer reads 522 to 686 ms there
  (`components_ms.save`). G3a's maximum was 141 to 152 ms and its `step_ms` did not include the checkpoint (`RESULTS_G3A` §4.1). The excess of this step over the typical boundary step (about 490 to 670 ms) is of the size of the save
  timer; whether the save falls inside the step because of the chapter hooks or because of where the bench's timer sits was not separated.
  The bench has no pause (`unattended`), so the save shows as `save_stall_steps` (15 at 1x, 20 with powers) and none of them is late. In a played world the
  chapter-end save is hidden behind the card (§7.3).
- The arms with powers are higher in the maximum (766 to 789 against 591 to 610 ms) and in the save timer (651 to 686 against 522 to 550 ms); the checkpoint file has the same size (686,866,342 bytes).
  Not investigated.

### 4.2 Components

Milliseconds, medians over the calls of each part (`components_ms`), G3a's rate-1 `rules` arm beside them.

| component | G3b `rules`, 1x | `player`, 1x | `player` + powers, 1x | `rules`, `max` | G3a `rules`, 1x |
|---|---|---|---|---|---|
| engine (reference, plain run) | 14.21 | 14.21 | 14.28 | 14.21 | 14.93 |
| runner step, typical (`loop_typical`) | 24.23 | 24.37 | 24.27 | 16.37 | 26.93 |
| observer (derived: loop minus engine) | 10.02 | 10.16 | 9.99 | 2.17 | 12.00 |
| `FRMA` | 5.08 | 5.09 | 5.67 | 2.69 | 6.24 |
| storm tick | 7.13 | 7.20 | 7.16 | 3.67 | 7.50 |
| labels | 1.03 | 1.02 | 1.20 | 0.54 | 1.32 |
| tables (`IDNT`, `INTN`, `TERR`) | 3.09 | 3.18 | 3.52 | 2.72 | 3.93 |
| write (per step) | 0.29 | 0.30 | 0.36 | 0.12 | 0.39 |
| runner step with settle (`loop_snapshot`) | 67.08 | 65.52 | 75.01 | 45.95 | 83.58 |
| settle (derived) | 42.85 | 41.16 | 50.74 | 29.57 | 56.65 |
| runner boundary work (without the checkpoint) | 24.59 | 24.72 | 32.35 | 22.34 | 30.45 |
| checkpoint (`save`) | 521.6 | 528.6 | 650.6 | 542.9 | 934.5 |

| step kind | `rules`, 1x | `player`, 1x | `player` + powers, 1x | `rules`, `max` | G3a `rules`, 1x |
|---|---|---|---|---|---|
| typical step (`step_ms_typical`) | 29.9 | 30.0 | 30.5 | 18.4 | 33.4 |
| boundary step (`step_ms_boundary`) | 105.5 | 102.2 | 120.1 | 79.6 | 127.6 |

- **The alert engine is not a cost.** 1,342 calls (`on_record`, `on_tick`, `on_row`, `on_chapter`, `on_territory`, `engine_check`) took 28.0 ms in all in the `rules` arm at 1x
  (27.9 ms `player`, 27.9 ms with powers); at `max` 531 calls took 4.75 ms; at 2x 782 calls took 9.6 ms. That is about 21 microseconds a call, 0.009% of the arm at 1x.
- **The observer fold and the settle are lower than G3a's** (10.0 against 12.0 ms; 42.9 against 56.7), as expected for 33,796 alive against 44,955; the observer is still about 4.6 times
  its `max` value at 1x (10.0 against 2.2), the low-duty-cycle effect G3a saw (12.0 against 1.7) and did not explain; this run did not test it either.
- **The boundary step at 1x is 102 to 120 ms**, 1.5 to 1.7 slots of 69.4 ms; G3a's was 127.6 to 130.0. The kin-thread fold, the alert engine and the report job (a worker) add
  nothing the medians show.

## 5. The gate, item by item

The limits of DESIGN §34.8 (spec §10) read from the arms of §4 and §6.

| item | limit | `rules`, 1x | `player`, 1x | `player` + powers, 1x | verdict |
|---|---|---|---|---|---|
| achieved at 1x over 5 minutes (`five_minutes` true: 313.0 s) | at least 0.98x | 0.99810x | 0.99805x | 0.99753x | pass |
| `late_steps` at 1x | 0 | 0 | 0 | 0 | pass |
| `step_ms` p99 at 1x | at most 66 ms | 42.1 | 42.1 | 43.1 | pass |
| `step_ms` maximum at 1x | reported | 595.6 | 609.5 | 766.5 | reported; the chapter-end step with the periodic save (§4.1) |
| boundary step at 1x | reported | 105.5 | 102.2 | 120.1 | reported; below G3a's 127.6 |
| achieved at `max` | at least 2.5x | 3.822x | 3.821x | 3.751x | pass |
| click to `PLAN` applied, p95 over the arm's clicks | at most 0.5 s (C6: 2 s) | n/a | n/a | 0.102 s at 1x (0.043 and 0.105), 0.024 at `max` (0.019 and 0.025) | pass |
| report job | at most 500 ms | **not measured** (2.7 ms is an arrival notice) | not measured (2.7 ms) | not measured (2.5 ms) | **open in the bench; 29 to 47 ms in the played session (§7.3)** |
| digest, arms without powers | equal to the plain run | equal | equal | n/a | pass (also at `max` and 2x) |
| digest, arm with powers | equal to the replay of its own log, and different from the plain run | n/a | n/a | equal; different | pass |
| both scripted powers applied | `all_applied`, none refused | n/a | n/a | Food at 473,150, Strike at 474,000, none refused | pass |

- **The two scripted powers** (`power_script`): Food x1.3 for 3,000 steps queued at 473,150 and a Strike (radius 6.25, kill 0.45, at (160, 160)) at 474,000, the chapter-end step
  itself (rule P); each shows the `queued`, `applied` and `since` `PLAN` records (9 in all). Both arms applied both.
- **The report item.** The bench's `report_ms` list is `[2.70]`, `[2.52]`, `[2.66]`, `[2.49]`, `[2.47]` and `[2.57]` for the arms of §4.1: the time between `before_checkpoint` of the
  chapter end and the `RPRT` on the stream, 2.5 to 2.7 ms, an order of magnitude below a built report. The reason is in the arm, not the report job (finding 2); the gate's
  verdict for that item in the JSON (`report_ms: true`) is therefore vacuous. The real time is in §7.3.
- **Presentation is not part of the trajectory, on Metal too.** Every arm without powers has the plain run's digest, in `rules` and `player` mode, at three rates.
- The client's draw p95 (the other G3a gate item) is §3's 2.76 ms offscreen with nothing else on the GPU; it was not measured with a running engine this time.

## 6. 2x and the external drive

**2x** is the rate G3a could not bench (its arms were 1x and `max`) and asked about: a boundary step at 2x (34.7 ms slot) would be about 3.7 slots. The spec names no 2x limit;
the reading used here is the gate's own with the slot halved: achieved at least 0.98 of 2x (1.96x), `late_steps` 0, `step_ms` p99 below one slot (34.7 ms).

| arm | achieved | seconds | `step_ms` p50 | p99 | max | `late_steps` | `save_stall_steps` | boundary step | digest |
|---|---|---|---|---|---|---|---|---|---|
| `rules`, 2x | 1.9923x | 156.8 | 21.2 | 28.7 | 613.6 | 0 | 32 | 90.7 | = plain |
| `player`, 2x | 1.9923x | 156.8 | 21.2 | 28.9 | 605.5 | 0 | 31 | 92.3 | = plain |
| `player` + powers, 2x | 1.98994x | 157.0 | 21.5 | 29.0 | 796.7 | 0 | 38 | 104.3 | = replay |

- All three meet that reading. The boundary step is 90.7 to 104.3 ms, 2.6 to 3.0 slots of 34.7 ms (derived), not 3.7: it is absorbed by the scheduler's bounded debt and nothing is late.
  Its components are close to the `max` row's (observer 2.7 ms, settle 38 to 48 ms, `save` 543 to 683 ms).
- Click to applied at 2x: 0.020 and 0.070 s (Food and Strike). Alert engine: 782 calls, 9.2 to 9.8 ms.
- The 31 to 38 save-stall steps at 2x (15 to 20 at 1x) are the periodic save counted in half-length slots; none is late.
- Not done: **the checkpoint on the external drive.** `/Volumes/M5 PRO external` was not mounted (`ls /Volumes` and `mount`); nothing was written there. The estimate of 1 to 3 s for 686 MB is untested
  and the card has the same shape for any stall length (spec §3.4). The internal-disk checkpoint is 0.52 to 0.69 s in the bench and 0.68 to 1.01 s from the card's opening in the play (§7.3).

## 7. An 18,001-step play of the planet

`alife play p0 --rate 1 --director rules`, 470,504 to 488,505, with a strict scripted reader (`play_reader_g3b.py`): every record decoded with `tests/play_decode.py`, chapter pauses answered
with `ack` and `run`, alerts acknowledged, one power of each kind and one cancel sent. **The reader is a script, not a person** (§7.4). The final run is `docs/results_g3b/play18k_run1.json`
(MD/play18k/`run1.json` and `run1.json.*`, MD/logs/`step5_reader.log`). Three earlier attempts were aborted by the reader (§9, finding 5) and are kept in MD.

### 7.1 The wire

| quantity | value |
|---|---|
| steps | 470,504 to 488,505 (18,001) |
| wall time | 1,263.2 s (stream 1,262.7 s), of which paused 8.6 s |
| bytes through the pipe | 22.87 GB (22,874,754,724) |
| decode errors | 0 (strict decode of every record) |
| decode time | at most 1.4 ms a record, 8.3 s in all |
| reader queue | never above 2 records |
| end | exit 0; `meta.status` "interrupted"; a checkpoint at 488,505 (the quit save is a 9-step stall) |

| tag | records | bytes |
|---|---|---|
| `FRMA` | 18,002 | 22,410,135,548 |
| `INTN` | 721 | 336,187,768 |
| `MAP1` | 721 | 78,485,176 |
| `IDNT` | 73 | 45,745,492 |
| `STRM` | 721 | 3,523,808 |
| `CLCK` | 4,545 | 309,060 |
| `TERR` | 73 | 156,720 |
| `ENVS` | 86 | 127,412 |
| `EVNT` | 57 | 26,044 |
| `PLAN` | 26 | 22,948 |
| `RPRT` | 6 | 16,196 |
| `ALRT` | 31 | 13,468 |

- **No `FOCS`.** The reader sent no `follow` (G3a's did), so there are no `FOCS` records and no focus events in this window.
- **The new records are small:** `ALRT`, `RPRT` and `PLAN` are 52.6 KB in all against 22.9 GB (the spec's estimate was under 70 KB a chapter).
- **Step times while running** (`CLCK.step_ms`, the median of the last 60 steps, so its p99 and maximum are of medians): p50 24.8 ms, p99 26.6, maximum 176.7 ms (the start-up); `late_steps` 0 to the end.
  The achieved rate at 1x was 14.36 steps a second (0.997x).
- **Saves.** The three periodic saves are the three chapter ends, behind the card (`end_save(count=False)`): none is counted as a stall. The only `save_stall_steps` is the quit save's 9.

### 7.2 Pauses

4 pauses, 30 `CLCK` rows with reason 3 and 10 with reason 4:

| step | wall s | reason | what |
|---|---|---|---|
| 474,000 | 243.8 | 3 (chapter) | chapter end; answered with `ack` and `run` |
| 477,750 | 506.3 | 4 (alert) | `pop_fall` raised to importance 3 (finding 3); `run` 2.5 s later |
| 480,000 | 666.2 | 3 | chapter end |
| 486,000 | 1,086.0 | 3 | chapter end |

At most 4 auto-pauses in 20,000 steps was the band; 4 in 18,001 are at its limit, but the spec's reading was "3 chapter ends and a population pause only if the world itself falls by 15% or
more, never as the aftermath of a power or a director move". The fourth pause is the aftermath of the player's Blight (below). The reader never let the world wait for long (paused wall time 8.6 s),
so what a player does at a card is not measured.

### 7.3 The chapter ends and the report

| chapter end | report arrived after the card opened | save finished after it | report's own `latency_s` | headline |
|---|---|---|---|---|
| 474,000 | 0.047 s | 0.682 s | 0.0466 | 2 storms reached named strength |
| 480,000 | 0.029 s | 0.960 s | 0.0283 | Population rose to 39,982 |
| 486,000 | 0.030 s | 1.006 s | 0.0294 | Toutraull formed |

- **Every report was readable inside the stall:** the `RPRT` arrives 29 to 47 ms after the `CLCK` that opens the card and the checkpoint is not written until 0.68 to 1.01 s. This is the
  limit's 500 ms with a factor of 10 or more to spare, on the real chapter (6,000 steps of a world with its history since the origin), where the bench could not measure it.
- The backlog of the last three reports (456,000, 462,000, 468,000) was rebuilt at `bind` (23, 33 and 47 ms) and sent first: "4 storms reached named strength", "2 storms reached named strength", "Population fell to 35,934" (the fall of 18.3% in chapter 78).
  The live reports (474,000, 480,000, 486,000) have 6 body lines each and 5, 8 and 0 interventions lines (3,300, 3,440 and 2,048 bytes).
- None of the six reports has a fallback or a withheld object.

### 7.4 Powers

Four powers, each previewed, queued and applied once, plus one cancel; the first Food was refused. The reader's script was Bloom, Strike, Food, Blight, then a cancel of Food.

| power | clicked at | applied at step | click to applied | preview reply | cost |
|---|---|---|---|---|---|
| Food (first try) | 471,004 | refused | n/a | 1 ms | n/a |
| Bloom x1.75 on patch 2, 3,000 steps | 472,004 | 472,009 | 0.054 s | 3 ms | 1 |
| Strike at (160, 160), radius 6.25, kill 0.45 | 473,504 | 473,519 | 0.056 s | 3 ms | 1 |
| Food x1.3, 6,000 steps | 476,010 | 476,015 | 0.075 s | 2 ms | 1 |
| Blight on patch 0, 3,000 steps | 476,504 | 476,509 | 0.056 s | 2 ms | 2 |
| Cancel Food | 477,515 | 477,515 (stopped) | n/a | n/a | no refund |

- 26 `PLAN` records: 4 previews, 5 queued, 4 applied, 1 cancelled, 1 refused and 11 `since` cards (Bloom 3, Food 3, Blight 3, Strike 2). The influence went from 6 to 1 (Bloom 1, Strike 1, Food 1, Blight 2);
  the cancel is not refunded and Food is ready again at step 485,015, the last `PLAN` text. `commands.jsonl` has 15 lines. The chronicle tells the powers: 4 "applied" lines (kind 20) and 3 "ended" lines (kind 21).
- **The first Food was refused: "Food is busy: the regen lever is held until step 476,000."** The rule director's `runaway_damper` had set food regrowth to 0.8 for 10,000 steps from 466,000, and the player's Food uses
  the same lever (lever exclusivity, spec §6.3). This is by design, but a player meets it at the start of p0 and the Powers tab greys Food with the same reason (§8). The reader retried at 476,010 and Food applied.
- **The Strike:** its preview said 12 organisms in the column and 5 deaths on average (the snapshot of §8, at 473,500); the report of the chapter says "10 organisms died by catastrophe" and its interventions block lists the Strike as the only catastrophe entry. The column was read 15 steps before the Strike was applied; the counts were not compared further.
- **The chapter card's interventions block** told the director's rule, the player's Bloom and the Strike, each with "still in place at step 474,000" or its date.

### 7.5 The chronicle and the alerts against G3a

The window is 21.05 minutes of wall time (1,262.7 s including 8.6 s paused); G3a's was 23.2. 69 live events (63 backlog events at the start are not counted), 3.28 a wall minute (3.30 per run minute).

| family | G3b events a minute | G3a | kinds (G3b count) |
|---|---|---|---|
| kin | 2.04 | 5.99 | thread formed 24, thread ended 19 (G3a: split 56, merge 56, formed 14, dissolved 13) |
| storm | 0.67 | 1.08 | formed 5, named 4, dissolved 5 |
| director | 0.38 | 0.04 | a rule's end 1, a power applied 4, a power ended 3 |
| system | 0.19 | 0.17 | checkpoint 4 |
| focus | 0 | 1.38 | no `follow` was sent |
| total | **3.28** | **8.66** | |

- **Both totals are inside the bands of spec §11** (2 to 5 total, 1 to 3 kin), but the total is nearer the bottom of its band and its director events are the scripted powers (G3a's total without its 1.38 focus events was 7.28). The kin thread rule did what the spec wanted: kin events fell about 2.9 times (5.99 to 2.04 a minute) on the same planet, 20,000 steps later.
- **By importance:** 0 for 7 events, 1 for 49, 2 for 13 (0.62 a minute, one every 97 s; G3a: 3.3 a minute, one every 18 s). The 13 are 5 storm dissolutions, 4 storm namings and 4 kin events (2 formed, 2 ended).
  Kin events are 4 of the 13 importance-2 events (G3a: 69 of 77). Peak: 7 events in one wall minute.
- **The kin threads still bud.** From the event file by hand: Viopeiv is the parent in 5 of the 24 "formed" lines, a thread called Pruashol forms, merges back into Viopeiv or Teshash and forms again four times (Pruashol 474,750, Pruashol 481,000,
  Pruashol II 483,750 and Pruashol III 486,750), and 4 of the 19 ended threads lasted 1,500 steps or less (Shaizur 750, Vuzen 1,250, Pruashol II 1,250, Pheben 1,500). This is owner item 15 (satellite budding, constant `P`), still open
  and still a story that reads as news at a rate of 2 kin events a minute.
- **Alerts: 26 raised**, 14 on the alert surface and 12 on the log surface. Alert surface: chapter 3 (importance 3), storm_named 4 and pop_slide 1 (importance 2), effect_ending 3, pop_surge 2 and pop_fall 1 (importance 1;
  the fall was raised to importance 3 by an update). Log surface: storm_ended 5, house_fading 3, lineage_ended 2, lineage_formed 2. 31 `ALRT` records (updates and clears included), 14 acknowledgements sent.
- **Alert rate:** alerts of importance 2 or more on the alert surface were 8 in 21.05 minutes, **0.38 a minute** (band at most 0.6; the earlier count on G3a's play, folded offline, was 10 alerts of importance 2 or more, 0.42 a minute). The 8 are the 3 chapter ends, 4 storm namings and the population-sliding alert
  (`calibrated` false). **Storm namings are half of them** (4 of 8; 6 of 10 on the earlier window): owner item 16 (whether storm namings belong on the alert surface); without them the rate is 0.19 a minute and the stack would hold only what a player can act on.
- 10 alerts were cleared and 1 updated. The spec's "about 28 alerts" prediction, which it called a number from one window, is 26 here.

### 7.6 Quit and resume

The quit save is a 9-step stall and `meta.json` reads "interrupted". The resume (`--rate max`, 500 steps, `docs/results_g3b/play18k_resume500.json`) ran 489,008 to 489,510: 361 MB, 255 `FRMA`, 0 decode errors, 0 late steps, and it sent
the last three reports (474,000, 480,000 and 486,000, with `latency_s` null: backlog) as the backlog plus 64 backlog events; its checkpoints now are 450,000, 488,505, 489,008 and 489,510. A first resume attempt (488,505 to 489,008; the checkpoint at 489,008 is its quit) ran the engine fine but the reader crashed on a rate of -1
(the `max` rate) and was fixed (MD/logs/`step5_resume_reader_attempt1_crash.log`).

## 8. The rendered decide half

Rendered offscreen from reader snapshots (MD/render/snaps/) with the production `Renderer` and `NSHostingView` (`MD/MacDayProbeG3b.swift.txt`, a probe removed from the worktree; log `MD/logs/step6_probe.log`; 5 tests passed in 3.3 s), at 1280 x 800 points
(2x) and 1100 x 700 for the dock. `play.json` was supplied so chapters count from the planet's origin. The PNGs are in MD/render/ and are not committed. Each was looked at.

| image | what it shows |
|---|---|
| `1_chapter_card` (step 474,000) | The card "Chapter 4 complete": the headline, the lead, six body lines and the Interventions block (the director's rule, the player's Bloom and Strike). Under it a strip: Units 4 of 6 with the next back at step 490,009, Ready Blight, Ending "Bloom at step 475,009; Food at step 476,000", and two alert rows. Buttons Continue, Continue at 2x, Powers, Show on map. The dock shows Resume, "Chapter 4" and the Cast tab. |
| `2_alert_stack` (475,773) | Three rows (Storm Stiogreix named, Population sliding with "provisional", Storm Fautheast named) and "+1 more", each with fly-to and acknowledge buttons; the bar reads "Chapter 5, next report in 4:54" with 1x highlighted. Fine. |
| `3_report_tab` (488,505) | Headline, lead, the body, Units, Ready, Alerts, a Ticker of five lines and "Earlier chapters". The label reads "Chapter 81 . steps 480,000 to 486,000", against "Chapter 7" on the bar (finding 1). |
| `4_powers_tab_strike_preview` (473,500) | Five of six unit pips filled. Food is greyed ("regen lever held until step 476,000"), Bloom cooling down until step 478,009, Blight and Strike usable. Strike is selected with radius 6.25 and kill chance 45%; the preview reads "Strike at (160.0, 160.0), radius 6.25, kill chance 0.45, cost 1, 4 of 6 units left after", with a caution of 12 organisms in the column and 5 deaths on average and "No earlier case for Strike"; Confirm and Cancel; then "Running: Bloom until step 475,009" with End early. On the map a red aim disc at the centre and the patch labels 0 to 7. Fine. |
| `5_dock_700` (1100 x 700) | Two alert rows plus "+2 more", the Report tab, and a storm inspector card (Gufleask) pinned under the tabs with its Follow button visible. Everything fits inside 700 points; the report list is cut mid-line above the card but scrolls. The tab header again reads "Chapter 79". |

- **The chapter card is translucent, so the map shows through its text.** Storm labels and rings sit on the lead line (the Chugrox label crosses "The population fell from 35,934 ...") and behind the buttons (Gufleask and Fautheast rings
  behind Continue and Show on map). This is a finding about the view (a material or a backing), not a logic fault; the text stays readable but not comfortably so.
- **The two alerts show twice** with the card up, once in the card's strip and once in the dock's stack (minor; stage 8 removed only the chapter alert from the stack while the card is up).
- **The window title** shows "Planet r forked from /private/..." (the probe's run name and the full source path), the G3a cosmetic finding (c) in another form; the launcher's name is not tested here.
- **A limit of the probe, not of the app:** the PNGs are a render of snapshot data; no click was made, and the aiming, `Esc`'s levels, the Continue keys and the badge are as tested by the Swift tests, not seen moving.

## 9. Findings

1. **Defect (G3b): the server numbers reports from step 0, not from the planet's origin.** `cli.py:1270` builds `PlayOptions(planet={"source": ..., "houses": ...})` and drops `made_at_step`; `play.py:1016`
   reads `planet.get("made_at_step", 0) // chapter` as `n_origin`, which is 0 for every world, so `RPRT.chapter` is the absolute chapter (DESIGN §34.4 says it is `n_abs - n_origin`, the number the player sees). `CLCK.chapter` is absolute by design and the client subtracts the origin chapter it reads from `play.json`
   (`made_at_step`, 450,000 on p0), so the card and the bar say "Chapter 4" and the Report tab "Chapter 79"; the first report of a fresh planet would read "Chapter 75". Not fixed here. The fix is to pass `made_at_step` through (one key at `cli.py` line 1270, which a new planet and a resume share) and a test that a planet made at 450,000 reports chapter 1 for its first chapter on the stream; the tests use worlds that start at 0 and cannot see it.
2. **The bench cannot read the report-time item** (§5). `bench --play-resume` gives each arm a fresh run directory with no history, so the chapter (468,000, 474,000] starts before the arm and the report job answers with an arrival notice
   (spec §3.5, the arrival chapter) after 2.5 to 2.7 ms. Offline, `load_run` on the arm's directory raises "the cut 468000 is not the step of a metrics row" (MD/logs/`step4_bench_report_is_withheld.log`). The gate item should either copy the
   source's metrics rows and origin row into the arm or refuse a chapter it cannot report; today it passes vacuously. The real time is §7.3's 29 to 47 ms.
3. **A population pause that followed the player's own Blight.** At 477,750 `pop_fall` (raised at importance 1 at 477,000, the provenance rule having found Blight in force) was updated to importance 3 and paused: "fell from 48,112 to 37,596 in 1,250 steps. Blight x0.00 for
   3,000 steps has been active since step 476,509." The fall is 21.9%, past the stage-8 rule's 15%, which makes it importance 3 whatever the cause. The spec's band says a population pause is for a world that falls by itself, never as the aftermath of a power. The two
   disagree; the choices are to keep the rule (a player is told what their Blight did; one pause), to exempt falls with a provenance at every depth, or to raise the depth for a fall that a provenance explains. A decision for the owner; not changed.
4. **Food is refused at the start of p0** by the rule director's `runaway_damper` (§7.4). By design (lever exclusivity); the refusal text says when the lever is back. A player meets it first thing.
5. **Three aborted sessions, all from the reader, none from the engine.** The reader reset its `run` timer on every repeated `CLCK` while paused, so it never sent `run` at the 477,750 alert pause and the world waited indefinitely. The reader was fixed and a watchdog added (it did not fire in the final run).
   The aborted artefacts are kept in MD (§11).
6. **Cosmetic and view findings** (§8): the translucent card, the doubled alert rows, the title with the source path, the report list cut mid-line above the pinned inspector card at 700 points (it scrolls).

## 10. What is still open

1. **The click-through by hand is the owner's, and it is the precondition of everything below.** The spec's precondition (the owner's click-through of the G3a window, `RESULTS_G3A` §9 item 1) was open when this Mac day ran. The G3b list is in `docs/HANDOVER.md`: the chapter card (saving,
   ready, folded, the notice with the pause off), the alert stack (acknowledge, fly, watch), the Report and Powers tabs, aiming a Strike and a Bloom on the map, `Esc`'s levels, the dock at 700 points with the inspector open and three alerts, and the card's translucency.
2. **D9: whether anyone acts without an objective** cannot be read from this run. The reader is a script that acted on a schedule; `commands.jsonl` has 15 lines, all the reader's. The measure the spec asked for (inputs per chapter by a person) needs a person to play `player` mode for a few chapters. The decision stands as taken (observe-and-poke, no score).
3. **Owner item 15, satellite budding of kin threads (constant `P`)**, and **owner item 16, storm namings on the alert surface** (§7.5): both measured here and both open. Neither is a defect of the build.
4. **The defects of §9**: 1 (the chapter numbering) and 2 (the bench's report time) are fixes for the next change; 3 is a decision.
5. **The second window of the play** (a fork of `64k3d_s1`) and the 20,000th step: only p0, 18,001 steps. The bands were measured on one planet and one window of 21 minutes.
6. **The checkpoint on the external drive**, **the client's draw beside a running engine**, **the 2D digest matrix on the GPU against 3a14aa6** and **`test_weather_w2_instruments` on Metal**.
7. **Labels stay provisional** until W2b adds `LABEL_VERDICTS` rows; W2b waits for the owner's go. The Strike cooldown (D10), the grace for a fresh preset world's founding die-off (stage 4) and the lineage alert classes for a watched thread remain the owner's calls.

## 11. Caveats

- One planet (G3a's `p0` at 470,504 to 488,505), one seed, one 21-minute play and one bench repeat per arm. The 1x arms agree with each other to 0.1% in the achieved rate and in the median; the maximum steps come from one chapter-end step and its save.
- `p0` has moved on since G3a (33,796 alive at 472,000 to 476,500 against 44,955): the costs scale with the population, and the planet's origin row is a one-off edit, not a fresh planet.
- The reader is a script: it never read a card and answered each pause within about two seconds, so the pauses cost 8.6 s of 1,263 s; the stall behind the card and the report time are measured, what a person does is not.
- The bench measures the loop with a pipe drained by a thread in the same process and unattended (no pause); the client's own copying and decode are not in it.
- The WorldRenderBench is a debug build offscreen; drawable acquisition and the compositor are not in it, and the present-to-present interval has not been measured.
- Housekeeping: nothing outside MD was written and nothing was deleted. MD holds the aborted artefacts (`planets/p0_abort1..3`, `planets/p0_repro_done`, `play18k_abort1..3`, `render/snaps_abort1..3`, `repro/`, `dbg/`) and is about 51 GB because of the planet clones. An orphaned `alife-deaf` test process (pid 54955) from earlier Swift runs was
  still running at the end; it is not this run's and was left.
- Nothing here validates a label or tunes a constant; the bands of spec §11 are read, not fitted.
