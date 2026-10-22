# G3a on the M5 Pro: Metal purity, the bench gate, a 20,000-step planet play and the rendered window

The Mac day of the play loop and the World window (`docs/DESIGN.md` §33, Review-log row I51; `docs/GAMES.md` §9 G3a), run
2026-10-22 on the M5 Pro (MLX 0.32.3, Python 3.12.11, Metal kNN kernel) on the branch `games/g3a` at `b7020a6` (the
merge with `origin/main` 39301ba and the sync review fixes). No code changed and nothing was committed during the Mac day. This
note is the record: what was run, what the numbers are, and what is left to the owner. The click-through of the window by hand
has not been done (§7).

## Answer

- **The gate of DESIGN §33.5.4 passes, in both director modes.** At 1x on the 64k3d storm planet (44,955 alive at the end of
  the window): achieved 1.00006x (director off) and 1.00004x (rules) over 312.4 s, `late_steps` 0, `step_ms` p99 47.6 and 50.4
  against the limit of 66, the client's draw p95 at most 2.41 ms against 25, and at `max` 3.68x (53.0 steps/s) and 3.69x against
  2.5. The maximum step (141 to 152 ms) is reported, not gated.
- **Presentation is not part of the trajectory, on Metal too.** Every arm's final state digest equals the plain `runner.run`'s
  (`d39b433b…dad9b`, four arms, two plain runs).
- **The estimates of §33.5.1 were right for the typical step and for fast-forward, and too low for the boundary step.** The
  boundary step (every 250 steps) takes 127 to 130 ms at 1x, not 80 to 90: 1.84 to 1.87 slots of 69.4 ms. The scheduler's
  bounded debt absorbs it (no `late_step`), but it is the largest cost of the loop, which makes the first remedy named in §33.5.1
  (moving the host-only kin matching and colony tracker after the step) more relevant; it was not tried. At `max` the same step is 95 ms. A
  checkpoint (686 MB) stalls the loop 0.93 to 0.98 s on the internal disk, 13 steps at 1x, not the 0.05 s of the CPU probe.
- **The loop is slower per step at 1x than at `max`** (typical 33.4 ms against 19.1), almost all of it in the observer fold
  (12.0 ms against 1.7). The likely cause is the GPU's clock at low duty (§5.1 of the specification); it was not tested. It
  does not miss the gate.
- **Metal purity: 19 test files, 0 failures.** That covers the play, stream-record, game-purity, tracer and spatial-Metal files
  (§1), and the Swift GPU tests (8 and 14 tests), the Swift unit tests (338 and 200) and `swift build` pass.
- **The 20,000-step play of the planet is clean on the wire and shows one finding about content.** 0 decode errors in 20,002
  frames and 27.85 GB; the process quit and resumed correctly. The chronicle runs at 8.66 events a minute, of which kin splits
  and merges are 112 of 201, many of them one group flapping (a group merges and splits again 250 steps later), and kin events are
  69 of the 77 events that would raise a toast. The chronicle as built is mostly kin churn (§5.3).
- **The planet of this Mac day is a fork of `64k3d_s0` at 450,000, not at 500,000 as decision D2 says** (the 500,000
  checkpoint exists in `runs/64k3d_s0`; the W2a pilot used 450,000). The reason was not recorded; the coordinator should
  confirm which step the first planet is to be (§8).
- **Not done:** the 2D digest matrix on the GPU, `tests/test_bench.py` and `test_weather_w2_instruments.py` on Metal, the
  checkpoint stall on the external drive, a present-to-present draw time in a real window, and the click-through (§7).

## 1. What was run

| step | what | where the result is |
|---|---|---|
| 1 | Nineteen test files separately on the default device (`Device(gpu, 0)`), `-m ""`, no filter | §2 |
| 2 | `swift build`, `swift test`, `GPURenderTests`, `GPUWorldTests`, `WorldRenderBench` alone | §3 |
| 3 | The planet: a fork of a copy of `64k3d_s0` at 450,000 with the storm overlay, a 20,000-step spin-up and exact houses | §4 |
| 4 | `alife bench --play`, both director modes at rates 1 and `max`, 4,500 steps per arm after 300 of warm-up; then the client's share (`WorldRenderBench` beside a `rules` arm at 1x) | §5, §6 |
| 5 | `alife play p0 --rate 1 --director rules` for 20,000 steps with a strict reader attached, then a resume | §7 |
| 6 | The world pass and the dock rendered offscreen from a stream snapshot | §8 |

- **Source of the planet.** The 450,000 checkpoint, `config.json`, `meta.json`, `metrics.jsonl` and the event chunks of
  `runs/64k3d_s0` were copied into a scratch directory with `LATEST` pointing at 450,000, and the planet was forked from the copy;
  the runs directory was not written. The planet configuration is `--preset 64k3d --overlay storm --set director.enabled=true`.
- **Spin-up.** 20,000 steps (the value of the specification §2.8, DESIGN §33.6 and D2), 21 s in total.
- **Starting step.** The planet was made with `--paused` and stdin at EOF, so the quit ran one step: `p0` starts at 450,001,
  with checkpoints 450,000 and 450,001. `--houses auto` chose `exact` (the header says `exact: true`), and the first stream
  decoded strictly from byte 0 with 0 skipped records.
- **Bench arm.** `alife bench --play` runs each arm under `PlayPacer` with the full record set, `stdout` into a drained pipe,
  cadence 250 / 250 / 6,000 (metrics, snapshot, checkpoint), `game.observe` on. 4,499 steps were measured per arm (4,500 requested after
  300 of warm-up), 18 of them boundary steps. A plain `runner.run` of the same world is the
  digest reference.
- **Files** (this note commits the small ones in `docs/results_g3a/`; the PNGs of §8, the logs and the planet stay in the
  scratch directory `scratchpad/g3a/macday`, called MD below): `bench_play.json` (MD/logs/`step4_bench_play.json`),
  `bench_play_share.json` (MD/logs/`step4b_bench_share.json`), `world_render_bench_alone.json` and `world_render_bench_share.json`
  (MD/render/`gpu_tests_out_bench_alone/` and `gpu_tests_out_share/`), `metal_purity_summary.txt` (MD/logs/`step1_summary.txt`),
  `play20k_run1.json`, `play20k_resume500.json` and `play20k_events.jsonl` (MD/play20k/), `play_reader.py` (the reader; needs the
  environment variables `WT` and `ALIFE_BIN`).

## 2. Metal purity

Each file was run separately with `uv run pytest tests/FILE.py -m "" -q` on the default device. None failed or was skipped. The
subprocess tests of `test_play_cli` spawn the CPU on purpose, so only their in-process parts ran on Metal.

| file | passed | file | passed |
|---|---|---|---|
| `test_play_purity` | 15 | `test_play_focus` | 13 |
| `test_play_hook` | 22 | `test_play_houses` | 50 |
| `test_play_storms` | 33 | `test_play_identity` | 14 |
| `test_play_cli` | 16 | `test_play_labels_v3` | 13 |
| `test_stream_records` | 40 | `test_play_planet` | 15 |
| `test_game_purity` | 55 | `test_play_replay` | 4 |
| `test_game_tracers` | 29 | `test_play_sched` | 23 |
| `test_spatial_metal` | 106 | `test_play_chronicle` | 15 |
| `test_play_colors` | 10 | `test_play_control` | 79 |
| `test_play_envs` | 3 | | |

Source: `docs/results_g3a/metal_purity_summary.txt`; one log per file in MD/logs/`step1_<file>.log`.

- **W2a's own Metal re-run at the merged head** (`I50` said the Metal gate was run on the stage-3 tree only, and asked for
  `test_game_purity.py`, `test_game_tracers.py` and `test_weather_w2_instruments.py`): the first two pass, 55 and 29 tests, with
  `Acts.buoy` and the tracers on. `test_weather_w2_instruments.py` was not in this run. The fused kNN kernel's own file
  (`test_spatial_metal`, 106) passes.
- **Not in this run:** `tests/test_bench.py` (§33.6 item 1 names it; the bench itself ran on Metal in §5 with its digests
  equal) and the 2D digest matrix as a script on the GPU against 39301ba.

## 3. Swift

- `swift build` passed. `swift test` passed: AlifeCore 338 tests in 32 suites, AlifeControl 200 tests in 23 suites
  (MD/logs/`step2_build.log`, `step2_swift_test.log`).
- With `ALIFE_GPU_TESTS=1`: `GPURenderTests` 8 passed and `GPUWorldTests` 14 passed (MD/logs/`step2_GPURenderTests.log`,
  `step2_GPUWorldTests.log`).
- `WorldRenderBench` alone at 2200 x 1400 pixels, 300 draws per scenario at 60 Hz, a debug build, offscreen. Nothing was dropped.
  Draw GPU time in ms (`MTLCommandBuffer.gpuEndTime - gpuStartTime`):

| rows | scenario | GPU p50 | GPU p95 | GPU max | CPU encode p50 | CPU encode p95 | CPU encode max |
|---|---|---|---|---|---|---|---|
| 64 | planet | 0.65 | 0.66 | 1.06 | 0.08 | 0.17 | 2.56 |
| 64 | planet with points | 0.64 | 0.65 | 0.66 | 0.10 | 0.17 | 2.57 |
| 64 | region | 0.62 | 0.63 | 0.72 | 0.10 | 0.19 | 2.62 |
| 64 | organism | 0.63 | 0.63 | 0.64 | 0.11 | 0.19 | 2.65 |
| 42,000 | planet | 0.65 | 0.76 | 0.99 | 0.14 | 0.23 | 2.69 |
| 42,000 | planet with points | 1.01 | 1.02 | 1.03 | 0.15 | 0.25 | 2.69 |
| 42,000 | region | 2.72 | 2.76 | 2.83 | 0.15 | 0.39 | 2.63 |
| 42,000 | organism | 1.16 | 1.19 | 1.30 | 0.14 | 0.22 | 2.65 |

Source: `docs/results_g3a/world_render_bench_alone.json`, MD/logs/`step2_WorldRenderBench_alone.log`.

The draw times with nothing else on the GPU are well inside a 16.7 ms frame at every level, 2.8 ms at the worst (42,000 rows,
region). The CPU maximum of 2.6 to 2.7 ms is the same in every scenario, including the 64-row ones, so it looks like a one-off outlier of
the harness (the first draw, say) and not a cost of the world; it was not investigated.

## 4. The planet and the cost of the loop

All numbers in this section are from the four-arm bench of step 4 (`docs/results_g3a/bench_play.json`; MD/logs/`step4_bench_play.json`
and `.log`). The engine reference is the same compiled step without observers, from the same state: **14.93 ms per step** (the
plain run).

### 4.1 The arms

| arm | achieved | seconds | `step_ms` p50 | p99 | max | `late_steps` | `save_stall_steps` | digest = plain |
|---|---|---|---|---|---|---|---|---|
| off, rate 1 | 1.00006x | 312.4 | 33.5 | 47.6 | 143.8 | 0 | 13 | yes |
| rules, rate 1 | 1.00004x | 312.4 | 33.4 | 50.4 | 141.1 | 0 | 13 | yes |
| off, rate `max` | 3.678x (52.96 steps/s) | 84.9 | 19.1 | 26.0 | 98.1 | 0 | 0 | yes |
| rules, rate `max` | 3.686x (53.08 steps/s) | 84.8 | 19.0 | 26.0 | 98.6 | 0 | 0 | yes |

- Every arm and both plain runs end at the digest `d39b433b69e234e088fc94121b0fa7cf375b32ad19474c745c8d77cc916dad9b`
  (44,955 alive). The rule tier changes nothing the bench can see.
- At `max` there is no slot, so a checkpoint is not a stall (`save_stall_steps` 0). At 1x the checkpoint costs 13 slots.
- `step_ms` is the time from the exit of `before_step` to the next entry, boundary work included, the checkpoint not.

### 4.2 Components

Milliseconds, medians over the calls of each part (`components_ms`).

| component | rate 1, off | rate 1, rules | `max`, off | `max`, rules | estimate (§33.5.1) |
|---|---|---|---|---|---|
| engine (reference) | 14.93 | 14.93 | 14.93 | 14.93 | 14.3 |
| runner step, typical (`loop_typical`) | 26.98 | 26.93 | 16.62 | 16.58 | 15.7 |
| observer (derived: loop minus engine) | 12.04 | 12.00 | 1.69 | 1.65 | 1.4 |
| `FRMA` | 6.19 | 6.24 | 3.69 | 3.70 | 3.6 |
| storm tick (`tick`) | 7.54 | 7.50 | 4.30 | 4.34 | 3.4 |
| labels | 1.32 | 1.32 | 0.83 | 0.83 | 0.7 |
| tables (`IDNT`, `INTN`, `TERR`) | 3.86 | 3.93 | 3.23 | 3.26 | (in 40) |
| write (the pipe, per step) | 0.38 | 0.39 | 0.17 | 0.18 | 0.5 |
| runner step with settle (`loop_snapshot`) | 85.74 | 83.58 | 59.30 | 59.35 | |
| settle (derived: `loop_snapshot - loop_typical`) | 58.76 | 56.65 | 42.68 | 42.77 | 38 to 40 |
| runner boundary work (without the checkpoint) | 30.26 | 30.45 | 24.30 | 24.33 | 15 to 25 |
| checkpoint (`save`, 686 MB) | 933.6 | 934.5 | 977.5 | 981.0 | 50 to 60 (CPU probe) |

| step kind | rate 1, off | rate 1, rules | `max`, off | `max`, rules | estimate |
|---|---|---|---|---|---|
| typical step (`step_ms_typical`) | 33.45 | 33.40 | 19.11 | 18.99 | 15.7 to 22.7 |
| boundary step (`step_ms_boundary`, 18 per arm) | 130.05 | 127.60 | 95.13 | 95.31 | 80 to 90 |

- **The typical step at `max` and the fast-forward rate are as estimated** (19.1 ms against 19.3 to 22.7; 3.68x against 3.5 to
  3.9x). The ladder 0.25x to 2x plus `max` stands; 2x was not benched (the arms are 1 and `max`).
- **The boundary step is 40% over the estimate at 1x.** The settle (58.8 ms against 38 to 40) and the runner's boundary work (30.3
  against 15 to 25) account for it; at `max` the same parts cost 42.7 and 24.3. A boundary step is 1.84 to 1.87 slots at 1x
  (69.4 ms); at 2x (34.7 ms) it would be 3.7 to 3.8 slots, absorbed only if the debt bound allows it (not measured: the bench has
  rates 1 and `max`).
- **The observer fold is 12.0 ms at 1x and 1.7 at `max`.** The fold is derived (the loop minus the reference engine from a plain
  run at full load), so it also absorbs any slowdown of the engine step itself at the low duty cycle of 1x. A low GPU clock at light load
  is the likely cause (§5.1 of the specification); a run with the GPU clock held was not made. Settle and boundary work are
  also higher at 1x (by 16 ms and 6 ms), which points the same way.
- **The checkpoint is 0.93 to 0.98 s on the internal disk** (the CPU probe on the same disk gave 0.05 to 0.06 s; the estimate for
  the external disk, 1 to 3 s, is untested). At 1x that is 13 skipped slots, reported as `save_stall_steps`.
- The 20,000-step play (§7) saw stalls of 16, 16 and 15 steps for the three periodic saves and 7 for the quit save, 54 in
  total (`CLCK` field), all of them `save_stall_steps`, none `late`.

### 4.3 The stream

Bytes a second by tag (`bytes_per_s`).

| tag | rate 1, off | `max`, off |
|---|---|---|
| `FRMA` | 20,751,354 | 38,160,554 |
| `INTN` | 311,696 | 1,146,359 |
| `MAP1` | 62,719 | 230,668 |
| `IDNT` | 41,817 | 153,795 |
| `STRM` | 2,740 | 10,078 |
| `CLCK` | 245 | 261 |
| `TERR` | 123 | 454 |
| `EVNT` | 48 | 176 |
| `ENVS` | 33 | 123 |
| total | 21.17 MB/s | 39.70 MB/s |

`FRMA` is 98.0% of the stream at 1x and 96.1% at `max`; a frame is 1.44 MB at 44,955 alive (1.36 MB averaged over the 20,000-step
play). The `max` arm writes a frame every other step (`frame_hz` 30), so its byte rate is below twice the 1x rate.

## 5. The gate, item by item

The numbers of `PLAY_GATE_1X` and `PLAY_GATE_MAX` (DESIGN §33.5.4), against the arms of §4.1 and the client's share of §6.

| item | limit | director off | rules | verdict |
|---|---|---|---|---|
| achieved at 1x over 5 minutes (`five_minutes` true: 312.4 s) | at least 0.98x | 1.00006x | 1.00004x | pass |
| `late_steps` at 1x | 0 | 0 | 0 | pass |
| `step_ms` p99 at 1x | at most 66 ms | 47.6 | 50.4 | pass |
| `step_ms` maximum at 1x | reported (80 to 90 expected) | 143.8 | 141.1 | reported; **above the expectation** |
| boundary step at 1x | (80 to 90 expected) | 130.0 | 127.6 | **above the expectation**; absorbed |
| client draw p95 | at most 25 ms | 2.41 (region, with the engine running) | | pass |
| achieved at `max` | at least 2.5x | 3.678x | 3.686x | pass |

The `max` arms ran 84.8 and 84.9 s, so `five_minutes` is false for them, as it should be (the limit there is a rate).

The two items above the expectation are findings, not misses: the gate names the maximum as reported. Per the gate's rule no physics or
parameter was changed. The component table (§4.2) is the report to the coordinator.

## 6. The client's share

The `rules` arm at rate 1 and `WorldRenderBench` at 42,000 rows, 4,200 draws per scenario, 2200 x 1400 pixels, ran together: the
Swift bench started 25 s into the arm (after its 300 warm-up steps) and ran 282 s inside it
(MD/logs/`step4b_marks.txt`: arm started 11:30:42, Swift bench done 11:35:49, arm done 11:36:16).

The arm under the client's load: 1.00002x, `late_steps` 0, `step_ms` p50 33.0, p99 49.5, max 151.5, 13 save-stall steps, digest equal
to the plain run (`docs/results_g3a/bench_play_share.json`).

Draw GPU time in ms with the engine running, zero draws dropped in 4,200 per scenario:

| scenario (42,000 rows) | GPU p50 | GPU p95 | GPU max | CPU encode p50 | CPU encode p95 |
|---|---|---|---|---|---|
| planet | 0.28 | 0.78 | 2.64 | 0.11 | 0.29 |
| planet with points | 0.54 | 1.19 | 3.78 | 0.12 | 0.28 |
| region | 1.28 | 2.41 | 5.38 | 0.12 | 0.28 |
| organism | 0.58 | 1.24 | 4.85 | 0.11 | 0.29 |

Source: `docs/results_g3a/world_render_bench_share.json`, MD/logs/`step4b_WorldRenderBench_share.log`.

- **The engine costs the client nothing visible.** The p95 is 2.41 ms at worst (the gate is 25), the engine's p99 moved from 47.6 and 50.4 to
  49.5, and the maximum from 141 to 152 ms, within the run-to-run spread of the other arms.
- **The medians here are lower than those of §3** (0.28 against 0.65 at the planet level). This run is longer and its draws
  alternate with the engine's work; the GPU's clock state is the likely difference (not tested). The two sets of numbers are not comparable draw for draw;
  both are far inside the gate.
- **Two limits of the measurement.** The Swift test is a debug build and draws offscreen: drawable acquisition and the compositor are
  not in it. The present-to-present interval (`Renderer.presentedHandler`) needs a real window and was not measured. A first
  attempt at this run failed because the probe file did not compile; it was killed and rerun (the failed logs are
  `step4b_*_failed_attempt.*` in MD/logs).

## 7. A 20,000-step play of the planet

`alife play p0 --rate 1 --director rules` with a reader attached (`play_reader.py`): every record decoded with the strict
`tests/play_decode.py`, `follow <uid>` sent at the start (`docs/results_g3a/play20k_run1.json`; MD/play20k/`run1.json` and
`run1.json.*`, MD/logs/`step5_reader.log`).

### 7.1 The wire

| quantity | value |
|---|---|
| steps | 450,001 to 470,002 |
| wall time | 1,393.7 s (stream 1,393.3 s), 14.36 steps/s on average (0.997x of 1x) |
| bytes through the pipe | 27.85 GB (19.99 MB/s on average) |
| decode errors, trailing bytes | 0, 0 |
| decode time | at most 1.5 ms per record, 10.6 s in total |
| reader queue | never above 2 records |
| end | exit 0; `meta.status` `interrupted`; the lock released; a checkpoint at 470,002 |

| tag | records | bytes |
|---|---|---|
| `FRMA` | 20,002 | 27,293,890,188 |
| `FOCS` | 20,002 | 1,840,184 |
| `CLCK` | 5,006 | 340,408 |
| `INTN` | 801 | 410,497,272 |
| `MAP1` | 801 | 87,193,656 |
| `STRM` | 801 | 3,791,692 |
| `EVNT` | 103 | 36,624 |
| `IDNT` | 81 | 55,669,792 |
| `TERR` | 81 | 173,600 |
| `ENVS` | 82 | 48,176 |

- **Saves.** Stalls of 16, 16 and 15 steps for the periodic saves at the 6,000-step marks and 7 for the quit save, 54
  `save_stall_steps` in all, `late_steps` 0 to the end (the last `CLCK`). `CLCK.saving` was set in 4 records.
- **Director.** One rule decision, `regen_pulse` at step 466,000.
- **The `CLCK` step times** (one record every 4th step): `step_ms` is the median of the last 60 steps, so its p50 of 26.0 ms is
  comparable to the bench's `loop_typical` at 1x (27.0), not to its p99 or maximum. The sample maximum of 191.8 ms is the second
  record of the run (step 450,002), the start-up.

### 7.2 The chronicle

201 events from the live feed, none of them backlog: 8.66 a minute over the 23.2 minutes of the play, with a peak of 25 in one wall
minute (`docs/results_g3a/play20k_events.jsonl`, one line per event).

| family | events a minute | kinds (count) |
|---|---|---|
| kin | 5.99 | split 56, merge 56, formed 14, dissolved 13 |
| focus | 1.38 | focus death 16, focus hand-off 16 |
| storm | 1.08 | born 12, named 7, dissolved 6 |
| system | 0.17 | checkpoint 4 |
| director | 0.04 | 1 |

By importance: 0 for 4 events, 1 for 120, 2 for 77 (3.3 a minute). The importance-2 events are 56 kin splits, 13 kin
dissolutions, 7 storm namings and 1 storm dissolution.

- **The ring and the toast.** The ring of 500 holds about 58 minutes at this rate. A toast on every new importance-2 event is one every
  18 s, and 69 of the 77 (90%) are kin events.
- **Kin churn.** 112 of the 201 events are kin splits and merges, and many are one group flapping. For example "Pruashol merged
  into Teshash" at step 451,000 is followed by "Pruashol split from Teshash" at 451,250, and Daudrall splits from
  Leifloan 6 times and merges into it 6 times between steps 450,750 and 454,250 (the event file lists every one). The chronicle as
  built would be mostly kin flapping, and the toasts would say little. This is the biggest finding on content; the remedies (a
  hysteresis or a minimum age before a split or merge is told, importance 2 for a split only when it persists) are G3b
  or G3a-follow-up decisions and none was tried.
- **Storms and focus.** 12 storm ids were seen, 7 named (Bruakaum, Chugrox, Flouflian, Fuapriond, Gufleask, Phaitraern, Raigeal),
  6 dissolved (the end reason is `dissipated` in each). 17 distinct focus uids: the first, plus 16 hand-offs at the focus's death
  (16 deaths).

### 7.3 Resume

The world was resumed at `--rate max` for 500 steps (not at 1x, to save time): it started at 470,002 and ended at 470,504, with
254 `FRMA` and 0 decode errors (`docs/results_g3a/play20k_resume500.json`). The houses were still `exact`, the origin checkpoint
450,000 survived, and the checkpoints are now 450,000, 468,000, 470,002 and 470,504. 64 backlog events were delivered at the start,
flagged as backlog (3 new events).

## 8. The rendered window

The window was not driven by hand. The stream snapshot at step 460,001 (44,162 alive) was rendered by the production `Renderer`
offscreen for the world pass, and by SwiftUI for the overlay and the dock at 1140 x 900 points plus a 360-point dock on the right, at 2x.
The PNGs are in MD/render/ (`planet_mid_*.png`, with a `_worldpass.png` for each) and are not committed. Each was looked at.

| image | what it shows |
|---|---|
| `1_planet` | The map with no organisms, by design at this level. The world is a square tile with its wrap strip repeated beyond the edges and a line across the middle (the hemispheres). Storm rings and names (Fuapriond, Chugrox, Bruakaum, Gufleask, "disturbance 1") and the dashed influence circles sit on the clouds. Territory cells are blocky with outlines. No seams, no wrong scale. |
| `2_region` | About 44,000 instanced spheres, dense and correct, with the storm rings and labels over them. |
| `3_organism` | Large shaded spheres with their bonds. The organism at the centre is at z = 79, near the top of the world. |
| `4_organism_selected` | A white selection ring centred on the organism, and the inspector card at the bottom of the dock: "Organism 20619802, Leifloan, House Leifloan", with Energy, Clade, Archetype "Unspecialised", Intent "Roaming" and their "provisional" tags, Age, Generation, Bonds, Parent, Follow, and the version-3 provisional note. |
| `5_region_selected` | A small ring at the same organism. |

- **The dock** is 360 points wide, with the clock strip, the Cast tab (Storms and Lineages) and the inspector card. At 900 points
  it fits without overflowing; the Follow button sits about 10 pixels above the bottom edge. At the 700-point minimum window height the
  card would crowd the Cast list; that was not rendered.
- **Cosmetic findings.** (a) Lineage detail lines truncate ("died: 3091 starved, 3099 killed, 3..."). (b) "6.216 born" uses
  a locale thousands separator that reads as a decimal point. (c) The window title showed "r", the name of the probe's run.
- **A limit of the probe, not of the app.** The first dock render showed yellow "prohibited" placeholders: `ImageRenderer` cannot
  draw AppKit-backed controls. The dock was then rendered through an `NSHostingView` in a hidden window.

## 9. What is still open

1. **The click-through by hand is the owner's.** Start the app (`cd gui && swift run`), open the World window from the Lab
   ("Open World"), or at launch with `--world RUN`. The launcher lists `runs/`, and the Mac day's planet `p0` is in the scratch
   directory (MD/planets/p0), so either copy it into `runs/` or make a new planet from the launcher (a fork of `64k3d_s0`, with the
   20,000-step spin-up, about 21 s). Try, and note anything that looks wrong:
   - **Launcher.** The list shows `p0` as a world with its step, source and houses mode; Resume starts it at 1x; "New planet" offers
     a source, a checkpoint, the storm overlay with its spin-up, the director and the houses mode; a build can be cancelled and
     is then listed as unfinished; a world being saved is listed as "saving..." and Resume is refused.
   - **Clock strip.** Pause, 0.25x, 0.5x, 1x, 2x and Max change the rate and the readout settles ("0.98x of 1x"); Resume does not change the rate;
     `+1` and, when paused, `+25`; "late N" and "saving..." appear at a checkpoint. Does 2x keep up at the boundary steps (a boundary
     step is 3.7 slots at 2x, §4.2)?
   - **Picking at each level.** Planet (no organisms; a storm, a warm pool, a territory cell), region (an organism, shift-click selects
     only, double click flies in), organism. Hover tooltips. The levels change at the thresholds of §33.4.7 and the map fades over 0.4 s.
   - **Follow and the hand-off.** Click an organism (it follows), wait for its death: the selection moves to the successor and
     the camera holds the point of death until the successor shows; follow a storm until it dissipates; follow a lineage.
   - **Cast and Chronicle.** The Cast lists the live storms and the eight largest kin groups, with hover highlights; the Chronicle's
     groups by minute, the five filters, a click flies to the event. Judge the kin churn of §7.2 in the window: is the
     Lineages filter usable, and would you want kin events off by default?
   - **Toasts.** Importance-2 events raise a 3 s toast, at most three at a time; at the measured rate one comes every 18 s,
     nine in ten of them kin.
   - **Keys.** `space`, `.`, `c` (the View menu's archetype and intent colour modes carry "(provisional)"), `1 2 3`, `f`, `[` `]`,
     `r`, `Esc`; `[` and `]` do nothing while paused.
   - **The dock on each side** (right, left, bottom), the divider drag, collapsing it (the one-line clock over the map), and the
     window at its 700-point minimum height (does the inspector card crowd the Cast list?).
   - **Close and quit while saving.** Close the World window during a save and quit the app during one: the app waits for the world
     (up to 40 s), the world finishes its checkpoint, and no `play.lock` is left behind. Reopen the launcher: the world is listed
     as "saving..." until it exits.
   - **A resume** of `p0` from the launcher, and the same world opened again after the app was quit.
   - **The three cosmetic findings of §8**, and whether the window judders (the present-to-present interval,
     `Renderer.presentedHandler`, has not been measured).
2. **Checkpoint stall on the external drive.** 686 MB on `/Volumes/M5 PRO external` (estimated 1 to 3 s; not measured).
3. **The 2D digest matrix on the GPU against 39301ba**, `tests/test_bench.py` and `tests/test_weather_w2_instruments.py` on Metal.
4. **The release build of the client** and a present-to-present draw time (§6); the bench is a debug build, offscreen.
5. **The first planet's source step** (450,000 here, 500,000 in D2; see the Answer).
6. **Labels stay provisional** until W2b adds `LABEL_VERDICTS` rows; the window and the chronicle say nothing of them as valid.

## 10. Caveats

- One planet, one seed, one 20,000-step play and one bench repeat per arm: the 1x arms agree with each other to 1% in the median,
  but the maximum steps (141 to 152 ms) come from 18 boundary steps and a checkpoint, and the kin event rate comes from a single
  window of 23 minutes.
- The planet is a fork at 450,000 with a 20,000-step spin-up; the numbers are for this population (44,955 alive at the end of the bench,
  44,162 at step 460,001), and the observer, settle and frame costs scale with it.
- The bench measures the loop with a pipe drained by a thread in the same process; the real client adds its own copying (not
  in the bench), which the share run of §6 covers for the draw, not for the decode.
- The cost of each part is a median over its calls; the sum of the medians is not the step.
- Nothing here validates a label; the render check is a look at five pictures.
