# Handover

Status of the `alife` project as of 2026-10-23 (the storm world's engine couplings, observers and instruments are built, W2a, `docs/STORMS.md`, DESIGN §32; the games vertical's observers are built, G1 and G2, and so is its offline World Report, N1, and its first playable steps, G3a, `alife play` and the World window, `docs/GAMES.md`, DESIGN §33; after the director evaluation, `docs/EVALUATION.md`; the 3D world is built, PRs #15 to #20, `docs/3D.md`; its first long run is analysed in `docs/RESULTS_3D.md`, and a second seed, a half-food control and a director run in `docs/RESULTS_3D_RUNS.md`), written for the next coding
session (a local Claude Code session on the owner's MacBook Pro, M5 Pro, 48 GB). Read this, then `CLAUDE.md`, then the parts of
`docs/DESIGN.md` you need.

## What this is

A neural artificial-life engine in Python on Apple MLX. It replicates the agent affordances from a Fraser Price
X thread: agents with per-agent recurrent neural controllers sense, signal, move, bond, split, exert forces,
harvest, share energy, fight, reproduce with mutation and die. There is no backprop and no fitness function.
`docs/DESIGN.md` is the normative spec. Its Review log at the end records every deviation and fix.

## Repository map

| path | what |
|---|---|
| `src/alife/` | engine (step, physics, bonds, energy, lifecycle, fields, brain, ...), run tooling (runner, runio, metrics), render, viewer, bench, CLI, the fused Metal neighbour kernel (`knn_metal.py`, DESIGN §4), the offline W0 weather prototype (`weather.py`, `analysis/w0.py`, `docs/STORMS_W0.md`), `director/` (the host-side ecosystem director, see below) and `game/` (the game observers of build steps G1 and G2, `docs/DESIGN.md` §28 and §29; G2 also adds the generic `groups.py` and `tracking.py` and `analysis/social.py`; W2a adds `game/tracers.py` and `analysis/w2.py`, DESIGN §32; N1 adds `game/report.py`, `templates.py`, `lexicon.py` and `analysis/world_report.py`, `docs/DESIGN.md` §31) |
| `tests/` | pytest suite. 4,252 fast tests passed, 89 were skipped (the GPU-only kernel tests, N1's `llm` test and the like) and 1 xfailed on the CPU device at the head of G3b (`-m "not slow"`, 37 deselected, `test_lifecycle_compiles_and_matches_eager` among them; 32 min 18 s in one process at the second merge-readiness check, 29 min 34 s at the first, about 9 minutes as six parallel shards during the build, which shared one kernel cache; since I58 `-n 6 --dist loadfile` takes 7 minutes with a private cache per worker) plus the slow ones |
| `docs/DESIGN.md` | the spec: state schema, step order, ledger, RNG streams, config, tests, decision and review logs (the director is §26, the LLM tier §26.10) |
| `docs/DIRECTOR.md` | the design of the ecosystem director; steps 1 to 4 are built, §12 to §14 record how steps 2 to 4 refine it |
| `docs/RESEARCH.md` | research directions (emergence, open-endedness, signalling) assessed against prior work, with a recommended order |
| `docs/ACTIVITY.md` | evolutionary-activity analysis of `runs/64k_s0` (method, results, sensitivity); figures in `docs/activity/` |
| `docs/SIGNALS.md` | information carried by the emit and msg signals of `runs/64k_s0` at eight states; full tables in `docs/signals/` |
| `docs/EVALUATION.md` | the director evaluation (`DIRECTOR.md` §9): arms A, B, C, R and RC on two seeds, results and verdict; outputs and figures in `docs/evaluation/` (`python -m alife.analysis.evaluation`) |
| `docs/3D.md` | the 3D world with gravity and a floor (owner decisions, layout, fields, Swift viewer, PR plan); built in PRs #15 to #20, normative spec `docs/DESIGN.md` §27 |
| `docs/RESULTS_3D.md` | the first long 3D run, `runs/64k3d_s0` (500k steps): population and `K`, vertical niches and buoyancy, bodies, evolutionary activity, comparison with 2D; figures and tables in `docs/results_3d/`, scripts in `analysis_cache/64k3d_s0/results3d/` (git-ignored) |
| `docs/RESULTS_3D_RUNS.md` | the four 500k-step `64k3d` runs (seed 0, seed 1, half food, director live): replication, the density control, the director in 3D, bodies and activity across runs; figures and tables in `docs/results_3d_runs/`, scripts in `analysis_cache/64k3d_runs/` (git-ignored) |
| `docs/RESULTS_G3A.md` | G3a's Mac day on the M5 Pro (2026-10-22): Metal purity, the bench gate (passed) and its component table, the client's share, a 20,000-step planet play (records, chronicle rates, stalls, resume), the rendered window, and what is open (the click-through by hand); small data files in `docs/results_g3a/` |
| `docs/RESULTS_G3B.md` | G3b's Mac day on the M5 Pro (2026-10-23): Metal purity (34 files, 1,393 passed), the bench gate with scripted powers at 1x, `max` and 2x (passed, except the report-time item the bench cannot read), an 18,001-step scripted play of a planet (records, pauses, powers, chronicle and alert rates against G3a), the rendered decide half, three defects found, and what is open; small data files and the reader in `docs/results_g3b/` |
| `docs/GAMES.md` | the gaming vertical (design and owner decisions): G1, the observer layer, is built (`game.observe`, `src/alife/game/`, `python -m alife.analysis.labels`, normative spec `docs/DESIGN.md` §28, Review-log row I44) with `INTENT_VERSION` 2 label rules (`RULES_V2`, calibrated offline and validated on three held-out windows on the GPU, 2026-10-21, DESIGN §28.4, §28.8) and, since W2a, `INTENT_VERSION` 3 (`RULES_V3`, the default: wind-relative Nomad and Settler plus Drifter, Floater and Sinker, calibrated on one storm pilot window, DESIGN §32.7, §32.8, row I50); G2, social tracking (houses, kin groups, colonies, territory, relations; `src/alife/groups.py`, `tracking.py`, `game/social.py`, `game/society.py`, `python -m alife.analysis.social`; the reports are `observers/social.jsonl` in an observing run's directory; normative spec `docs/DESIGN.md` §29, Review-log row I45) is built, run on the GPU (2026-10-21) and has provisional thresholds (`SOCIAL_VERSION` 1, measured, none changed); N1, the offline World Report (`python -m alife.analysis.world_report`; code-written template sentences with a checker, and an optional local model that only selects and orders them; normative spec `docs/DESIGN.md` §31, Review-log row I49) is built with provisional thresholds (`REPORT_VERSION` 1); G3a, the play loop and the World window (`docs/DESIGN.md` §33, row I51, `docs/RESULTS_G3A.md`), is built; G3b, chapters, alerts, the live report and direct powers (`docs/DESIGN.md` §34, row I54), is built and its Mac day ran 2026-10-23 (`docs/RESULTS_G3B.md`); G3c to G5 are not built |
| `docs/STORMS.md` | the storm world (an alien-planet weather game on the 3D world): the weather model, the coupling, the plan W0 to W5; W0 (the offline model, `weather.py`, `docs/STORMS_W0.md`), W1 (the engine weather layer, normative spec `docs/DESIGN.md` §30) and W2a (rain seeding, shade, the controls, the observers' version 3 and the instruments, `docs/DESIGN.md` §32, row I50) are built and off by default; W2b (calibration and the 14 runs) is not run |
| `docs/DESIGN.md` §37 | the mini world: preset `mini3d` (a quarter of `64k3d`'s area, about 10,500 alive), overlay `storm160`, the launcher's overlay choice by source size, the two host-side edits (stream header `cells`/`dx` from the layer, storm vorticity in units of the reference f0 0.025); Review-log row I62 |
| `docs/bench/` | `alife bench --stages` reports for `64k`, `64k3d` (`C` = 24, 32, 48) and `tiny3d` (`docs/DESIGN.md` §27.13), before (2026-10-16) and with (2026-10-20) the fused Metal neighbour kernel |
| `gui/` | AlifeControl, a SwiftUI macOS app that drives the CLI (see `gui/README.md`) |
| `runs/` | run directories (git-ignored, local to the Mac) |
| `.github/workflows/ci.yml` | Linux CPU CI: ruff, ruff format check, fast suite; nightly slow suite |

## How to run

```bash
uv sync --extra viewer --extra video          # Python 3.12 pinned in .python-version
uv run pytest -m "not slow" -q                # fast suite, one process (about 28 min on the Mac)
uv run pytest -m "not slow" -q -n 6 --dist loadfile   # the same in parallel (about 7 min; DESIGN section 22.0, I58)
uv run alife run --preset 64k --steps 20000 --out runs/NAME
uv run alife resume runs/NAME --steps 100000
uv run alife view runs/NAME                   # pygame viewer (re-simulates from the latest checkpoint)
uv run alife inspect runs/NAME --check        # summary plus full invariant check
uv run alife bench --preset 64k --determinism --stages
uv run alife activity runs/NAME                # evolutionary activity (needs --extra analysis for plots)
uv run python -m alife.analysis.signals RUN --steps ... --out FILE.json   # signal information, docs/SIGNALS.md
uv run alife resume runs/NAME --set game.observe=true   # attach the game observers (labels and a social head in the rows, observers/ sidecars, observers/social.jsonl)
uv run python -m alife.analysis.labels RUN --from STEP --steps N --out FILE.json   # do the game labels predict behaviour? docs/DESIGN.md §28.8
uv run python -m alife.analysis.social RUN --from STEP --steps N --out FILE.json   # social trackers: thresholds, churn, snapshot cost; docs/DESIGN.md §29.12
uv run python -m alife.analysis.w2 [A@0=]RUN... --out FILE.json   # the W2 storm-experiment statistics (P1a to P6) from W2b-style runs; docs/DESIGN.md §32.9
cd gui && swift run                           # the macOS control app (Lab) ; its "Open World" opens the World window (G3a)
gui/scripts/make-app.sh --app both --install  # the Lab and Storm Worlds (the game, no terminal) in /Applications (row I59; quit them first)
# stdout of `alife play` is the binary stream (the window reads it): redirect it when you run one by hand. The window's launcher makes planets.
uv run alife play --init-from runs/64k3d_s0:500000 --overlay storm --set director.enabled=true --weather-spinup 20000 --out runs/planet_a --no-control --until 500100 > /dev/null   # make a planet (DESIGN §33.3.4)
uv run alife play runs/planet_a --rate 1 --no-control --until 501000 > /dev/null       # play a world directory (to step 501000)
uv run alife bench --play --preset 64k3d --overlay storm --set director.enabled=true --steps 2000   # the play loop headless (DESIGN §33.5)
uv run alife run --preset mini3d --steps 500000 --seed 0 --out runs/mini3d_s0   # the mini world's evolved source (31 min idle; DESIGN §37)
uv run alife play --init-from runs/mini3d_s0:450000 --overlay storm160 --set director.enabled=true --weather-spinup 36000 --out runs/planet_m1 --no-control --until 450100 > /dev/null   # a mini planet
uv run alife bench --play --play-from runs/mini3d_s0:450000 --overlay storm160 --set director.enabled=true --play-spinup 20000   # its play loop (the gate of DESIGN §33.5.4)
```

Overnight runs: plug in, lid open, and either use the GUI's timer or
`caffeinate -is uv run alife resume ...`. Stop cleanly with exactly one SIGINT (Ctrl+C or `kill -INT <pid>`
of the Python process). A second SIGINT is a hard exit. Do not use `--quiet` if you want progress lines.

## Parallel tests (I58)

`pytest-xdist` is in the dev group. The fast suite runs as `uv run pytest -m "not slow" -q -n 6 --dist loadfile`
(7 min 0 s warm on the M5 Pro against 27 min 17 s serial; 7.8 GB in all, `-n 8` is 5 min 56 s and 9.1 GB; with `-p cpu_device` and `PYTHONPATH` as
before for the CPU device). Why those numbers and the whole measurement table are in DESIGN §22.0 "Parallel runs".
What to know:
- **Use `-n 6` while LM Studio holds its model; `-n 8` only with more than about 15 GB free** (`vm_stat`, free plus
  inactive; the worst case of 8 workers is 12.7 GB). **Never `-n auto`**: it is 18 workers here, 435 s against 362 s
  at `-n 8`, and 16.3 GB. 12 workers took 4 min 41 s and peaked at 11.7 to 12.2 GB; try that only with the model
  unloaded.
- **Every pytest process now compiles MLX CPU kernels into a private cache** (`tests/worker_cache.py`, hooked in
  `tests/conftest.py`), seeded from the shared `$TMPDIR/mlx` and merged back into it. Concurrent runs and parallel
  workers no longer collide (before: "File exists" linker errors, aborted workers, once a hang). Keep giving each
  agent its own `TMPDIR` as before: that is still the persistent, warm cache. `ALIFE_TEST_JIT_SHARED=1` switches the
  isolation off. A cold cache costs a parallel run 4 to 5 minutes (799 s against 540 s at `-n 4`).
- **Disk:** a fast-suite run used to leave about 38 GB under pytest's basetemp, three runs kept per `TMPDIR`; a dozen
  measured runs filled the Mac's disk (ENOSPC, the tests then failed). `tmp_path_retention_policy = "failed"` is now set
  (`pyproject.toml`, the owner's approval of 2026-10-23): a passing run leaves no basetemp, a run with a failure about
  2.4 GB (three such runs kept). Delete the `pytest-of-*` directory of a `TMPDIR` after reading a failed run. One test, `test_alife_bench_play_sends_scripted_powers_from_the_command_line`, also leaves
  a 655 MB `alife-bench-play-*` directory in `TMPDIR` per run (its last `cli.main` call drops `--play-out`; DESIGN §22.0).
- **Ambiguous kernels and the goldens:** kernels whose library names are ambiguous across configurations (F21 item 6)
  are never seeded from or handed back to the shared cache (`tests/kernel_ambiguity.py`; it costs a parallel run about
  90 s at `-n 6`, since every process recompiles about a quarter of its kernels), and the two Linux golden tests run
  their 50 steps in a child on an empty cache (`tests/golden_run.py`), so a restored or warm cache cannot change a digest.
- MLX buffers are capped at 256 MB per test process on the CPU device and in workers (`ALIFE_TEST_MLX_CACHE_MB`):
  without it four workers held 12.6 GB. A serial run on the GPU keeps MLX's limit (its effect on Metal speed is
  unmeasured).
- A killed serial run leaves `alife-jit-*` in its `TMPDIR` (hard links, no data); the next run sweeps those older
  than a day. `--dist loadfile` is the fast form, not a correctness condition (the order dependence of
  `test_director_llm.py` is fixed). CI saves its MLX cache only after a green job, so one red test keeps the next
  run cold; an `if: always()` save would pin a partial cache under the same key (owner's decision, not applied).
- **CI** runs `-n 4 --dist loadfile` on a pinned `ubuntu-24.04` (the goldens depend on the image's g++; `ImageOS` is in the
  MLX cache key); the Linux time is not measured yet (read it off the first runs, which also pay the goldens' cold
  compile and the ambiguous kernels, and lower the 120-minute timeout). **Open:** `fix/mlx-compile-collision` is unmerged
  and now only needs its docs (F21 item 6, §15.4, the CLAUDE.md line and a row): its test code is ported into
  `tests/kernel_ambiguity.py`, `worker_cache.py`, `compile_probe.py` and `test_mlx_compile_collision.py` here (DESIGN §22.0
  item 5); drop its `tests/` files and `conftest.py` hunk and rebase it after this PR.

## Verified state

- `main` holds the v0 engine (PR #1), the GUI, periodic immigration and seasons (PR #2), the GUI follow-ups (PR #3),
  the director design (PR #4) and director build step 1, the engine's live environment state (PR #5), all
  squash-merged, as are director build step 2 (the host-side director core, PR #6) and step 3 (the LLM tier, PR #7).
  Director build step 4 (the GUI Director panel and `alife llm-check`) is the PR this handover ships with.
  The yoked `random` persona for evaluation arm R (`--persona random --yoke-log runs/B/director.jsonl`; `DIRECTOR.md`
  §15, `DESIGN.md` §26.11) is built on its own branch; the GUI offers it (Persona menu, "Yoked to run" picker) and the
  click-through below has an item for it, not yet run by the owner.
- Mac (M5 Pro, Metal), this PR: 871 tests pass (852 fast, 19 slow), 2 skipped (the Linux-only golden digest and the
  `llm` test that needs a real server; it passed once with LM Studio), ruff and the format check clean. Linux CI was
  not run in this session. A real LM Studio smoke test (`gemma-4-26b-a4b-it-mlx@8bit`, tiny preset, 3000 steps, a call
  every 500 steps, Trickster and Architect): 12 calls, all valid, median latency 3.2 s (first call after idle 17 s);
  the logs replayed bitwise.
- Director off is bit-identical to `main`: the baseline digests (`legacy_state_digest`, counters and ledgers of the
  tiny cases on CPU and GPU, 200 steps) are unchanged, and no existing config hash moved (`runs/64k_s0` still hashes
  to `69b4e21a…`; `alife inspect runs/64k_s0 --check` passes).
- Mac, v0 code: `bench --preset 64k`: 61.6 steps/s (16 ms/step), 1.1 GB peak, weights updated in place,
  deterministic. The neighbour search is the biggest stage (33%).
- Metal quirk (DESIGN F20): float32 GPU matmul is reduced precision (about 8e-4 relative). Anything that must
  be accurate avoids GPU matmul.
- GUI on the Mac: it compiled first time (Swift 6.4 toolchain, macOS 27 SDK). The levers and CLI flags were
  checked against `config.py` and `alife --help`. The owner clicked through the tiny-preset checklist below
  before and after the Swift 6 migration, and every step passed. Step 4 added the Director panel (below); it was built
  and its pure logic and tail parsers were checked with a throwaway script and real runs, but the panel itself has not
  yet been clicked through by the owner (checklist below).

## New in PR #2

- **Periodic immigration:** `population.immigration_every` (N steps, 0 = off) and
  `population.immigration_count` (founders per event). These combine with the older floor rule
  (`immigration_floor`, `immigrants_per_step`), which only tops the population up while it is below the floor.
- **Seasons:** `fields.season_period` (steps, 0 = off), `fields.season_amplitude` ([0, 1]) and
  `fields.season_mode` (`global` or `per_patch`). Each patch's fertility Gaussian is weighted by
  `1 + A·sin(2πt/P + φ_p)`. Regeneration stays integer and ledger-accounted.
- **Compatibility:** keys added after v0.1 are omitted from `config_hash` while they equal their defaults
  (`config.POST_V01_KEYS`), so existing runs resume. With defaults, trajectories are bit-identical to before.
- **GUI:** new run, resume and fork; levers; stop button and stop-after timer; keep-awake; live status from
  `metrics.jsonl`; viewer, inspect and Finder buttons. Fork hides the fast overlay because it changes the brain
  size.

## Ecosystem director (docs/DIRECTOR.md)

Goal: keep the world novel but viable by acting only on the environment, never on agents or genomes.

- **Step 1 (PR #5, merged):** the engine's live environment state, `State.env_*` (regeneration multiplier, seasons,
  drift, per-patch gain, immigration request, catastrophe), read by the compiled step behind `director.enabled`.
- **Step 2 (PR #6, merged, `DESIGN.md` §26.1 to §26.9):** the host-side core, no LLM. `alife.director` computes vital signs from
  `metrics.jsonl`, keeps a behaviour archive (novelty and a coverage grid), and a rule tier (extinction rescue, runaway
  damper, monoculture breaker, stagnation response, three personas) decides at `director.rule_every` boundaries
  between compiled steps, within budgets and cooldowns. Every action is logged with explicit clamped values to
  `runs/NAME/director.jsonl` (plus `journal.jsonl` and `director_archive.json`), and the log replays bitwise.
- **Use:** `alife run --preset 64k --director live --set director.persona=trickster ...`; the same for `resume RUN
  --director live`. `--director replay` applies a log: `resume RUN --director replay` in place, or `run --init-from
  SRC:STEP --director replay --replay-log LOG --out NEW` for a fork. A run must be started with the director enabled
  (`--director live|replay` on `run` sets `director.enabled`); to add it to an older run, fork it.
- **Step 3 (PR #7, merged, `DESIGN.md` §26.10):** the LLM tier. With `--llm` / `--llm-every N` (live mode), a local LM Studio
  model (`gemma-4-26b-a4b-it-mlx@8bit`) is asked at every multiple of `director.llm_every` for up to two menu actions
  and a journal, with a JSON-schema reply generated from `MENU`. The call is synchronous in the director hook (60 s
  timeout). The rule tier keeps reverts and the safety rules at every boundary; the model makes the discretionary
  moves; a failed, invalid or fully refused call falls back to the rule tier's discretionary rules for that boundary.
  Accepted actions are `source: "llm"` entries in `director.jsonl` (they replay bitwise, and a replay never calls
  the model); every call has a full record (raw prompt and reply, outcome, clamps, fallback) in `journal.jsonl`.
  With the tier off (the default) behaviour is exactly step 2's and nothing touches the network.
- **LLM caveats:** the model's text is untrusted (only menu selection survives; no text is fed back into a prompt).
  `fertility_shift` is offered to the model in its gain form only. A reasoning model with thinking on returns an empty
  reply (fallback every time). A call takes about 3 to 8 s on the M5 Pro; the first call after a cold load can take
  longer than the timeout, so load the model first. A resumed live run re-asks the model for boundaries after the
  checkpoint, so it need not repeat the original decisions (the log is the truth). The prompt and the thresholds are
  first values, not tuned.
- **Caveats:** a fork resets the environment levers and starts with the director off. `view` and `render` re-simulate
  without the director, so after an action they diverge from the run. A resume that interrupts a rule window may
  decide differently from an uninterrupted run; the log is the truth. The thresholds in `RuleParams` (rules version
  1) are first values calibrated on the `64k_s0` rows (§26.7), not tuned results. On the tiny preset the extinction
  rescue fires at step 100 (a start-up boom and bust of more than 30% inside one window), and the damper sees a body
  above 5% of a 250-agent population; neither is expected on a 64k run.

## Results so far (run `runs/64k_s0`, seed 0, 64k preset, about 1.5M steps)

Analysis dashboard: https://claude.ai/artifact/KpVPBk7ceGcah9BGdsmQ7K. It is private to the owner and holds
verified findings for the whole run.

- The population stays near 32–34k throughout, below the carrying-capacity estimate of about 40k. The energy
  ledger was exact on every step and `full_check` passed at 120k and at 1.504M.
- Founder lineages collapsed: 16,384 founders, 88 clades at 120k, 5 at 942k, 4 at the end with one holding
  84%.
- Mutation scale was selected steadily down (median 0.022 to 0.006). The mutation rate fell to about 0.006 by
  500k, then rebounded to 0.014.
- Signalling became rarer, not emptier: emitters fell from 81% to 26%, flat since about 1M, but the remaining
  signals carry more information than the early ones (signal-information bullet below). Energy sharing nearly
  vanished (6% to 1.6%). Bonds became brief contacts: the lifetime proxy fell from about 135 steps to about 12.
- Predation: attack effort rose 4–5x early, then lethality per attack halved. Kill share of deaths is about
  19% at the end.
- Reading: the ecology largely settled after about 1M steps while evolution continues slowly. This is the
  classic open-endedness stall in a stable world without newcomers. The new levers are meant to test it.
- Caveats: `weight_diversity` before step 120k in this run is unreliable (pre-fix metric). The kill share
  undercounts delayed combat deaths, which are recorded as starvation. Kin biases are partly a proximity
  artefact.
- Evolutionary activity (`docs/ACTIVITY.md`, `uv run alife activity runs/64k_s0`): lineage-based Bedau–Packard
  statistics from the complete event logs (`deaths_unlogged` = 0), against a neutral shadow and a heritability
  null. Channon's component-normalised test gives **bounded** activity in 0–120k, 120k–1M and 1M–1.5M, for every
  component definition, grid and seed: new adaptive lineages keep arising at a constant rate, nothing
  accumulates. Lineage sorting relative to the heritability null is slightly stronger after 1M, so the "stall" is a
  steady state, not a decline of selection. Bedau's original normalisation is threshold-unstable on this run (see
  the note). The tool reads runs that start at step 0; forks (founders from a checkpoint) are not supported yet.
- Signal information (`docs/SIGNALS.md`, `python -m alife.analysis.signals`): mutual information of each emit
  channel and message component with the sender's resource, harvest, energy, crowding, kin and next action, at
  0, 120k, 250k, 500k, 750k, 1M, 1.49M and 1.504M, against shuffle nulls. The intermediate states were
  re-simulated with every `metrics.jsonl` row checked exactly (0–120k needs the run's original commit `f09b32a`)
  and are cached in `analysis_cache/64k_s0/` (git-ignored). Founders: no information. From 250k on, `emit0`,
  `emit1`, `msg0` and `msg1` robustly report local crowding and the coming stop (up to 0.34 bits, `msg0` at the
  end), and a little resource (≤ 0.11 bits); energy ≤ 0.05; kin composition beyond crowding ≈ 0; attack
  intent never; `emit2` almost nothing. At 120k, lineages and patches had their own codes; one code dominates
  as one clade takes over. Whether receivers use the signals is open (receiver-side test in the note).

## Results so far: the director evaluation (`docs/EVALUATION.md`)

Arms forked at step 120,000 and run to 1.5M on two seeds: A (off; seed 0 is `runs/64k_s0`), B (rule tier, Gardener),
C (rule tier + LLM, Trickster), R (random, yoked to B) and RC (random, yoked to C at C's budget). Runs are
`runs/eval_s<seed>_<arm>`; activity output is in `analysis_cache/eval/` (git-ignored).

- **Criterion 1 fails:** the Gardener acted once and twice in 1.38M steps, so B is almost A and beats neither A nor R
  on coverage and persistent novelty. No collapse.
- **Criterion 2 passes as specified:** C beats B on both seeds (coverage 87 and 69 against 24 and 30). But one random
  seasons move (R, seed 0) gave nearly the same coverage and persistent novelty, and RC (random moves at C's steps
  and budget) gave more coverage than C on both seeds (228 and 119 against 87 and 69) and slightly more persistent
  novelty. The archive's coverage and novelty mostly measure forced population oscillation. The owner rated C's
  journal legible (mean 4.3 of 5 over 10 entries, none below 4): the LLM tier explains its moves well, but the moves
  do no better than random ones on these metrics.
- **Evolutionary activity is bounded in every arm** (one non-robust window in RC, seed 0, during a sweep to
  monoculture). C's rate of new adaptive activity is 2.4–4 times A's in the cohort space (2.2–3 in the marker space)
  and 1.3–3.9 times RC's, in every window and space on both seeds: the one place where the LLM's choices beat random
  ones.
- **C's cost:** repeated crashes to about a quarter of the reference population, 34 and 38 extinction rescues, and
  about 60% of rule windows at risk. The extinction-risk flag reads every seasonal trough as a crash.
- **Immigrants rarely establish** (1 of 26,349 agents at the end of C on seed 0), so immigration pulses add no lasting
  genetic novelty.
- **The LLM** (gemma-4-26b-a4b-it-mlx@8bit) replied with valid JSON every time (one transport failure in 112 calls),
  with no clamps, at a median 3.6–4.4 s; it chose "none" in about two thirds of its calls, mostly for lack of budget.
- Analysis: `python -m alife.analysis.evaluation` recomputes the director's vital signs for every arm (equal to the
  logged ones); `alife activity` reads forks (`--start`).

## Results so far: the first long 3D run (`docs/RESULTS_3D.md`)

The run is `runs/64k3d_s0`: preset `64k3d`, seed 0, 500,000 steps, director off. It used code 358de29, whose tree
equals `main` 99098f4. It has 11 checkpoints, 50k apart, and complete event logs. The 2D reference is `runs/64k_s0`:
the metrics-row comparisons cover the same steps, but its bodies' structure is read from its checkpoints at 1.49M to
1.504M steps (the only ones that survive). This is one seed: no result below has a between-seed estimate. Seed 1 and
two controls are in the next section; several of these results did not replicate.

- **Stability.**
  - 9.5 h at 14.6 steps/s, peak 3.32 GB.
  - The ledger was exact on all 5,000 rows, with no error flags.
  - No hash cell overflowed. The maximum occupancy was 37, at step 4,500, and at most 18 after step 50k, against
    `C` = 48.
- **Population.**
  - The founders boomed to 25,719 at step 100 and crashed to 11,585 at step 1,200.
  - The population then overshot to 42,730 at step 13,100 and fell to 25,111 at step 36,600.
  - From step 100k it is stationary around 41,923 (sd of the 100-step means 3.8%, no trend). That is 1.25× the
    sizing aim of 33,600 and 1.32× the 2D run.
  - n/`K_est` is 0.63 to 0.68 and n/`K_floor` is 2.2 to 2.4 from step 100k.
  - Only 19 to 21% of the agents live in the floor layer, against the 70% that the sizing assumed. 53% live in the
    bottom quarter of the column and 35% in the top quarter.
  - n/`K_floor` is lowest at the founder crash, about 0.62, which is 3.1× above `risk_k_frac` 0.2.
- **Vertical niches.**
  - From step 177,100 four founder clades remain, in two depth zones of two clades each. Floor zone: clade 5165 (40%
    of agents, median height 5.8) and 8695 (20%, 17.0). Top zone: 4616 (11%, 68.0) and 13526 (28%, 74.4).
  - The clade label carries 0.60 of the height-band entropy (MI 0.63 nats against a shuffle baseline of 0.0001), and
    0.73 within an xy column (baseline 0.010 nats). Most of the band MI separates the two zones: the zone label alone
    carries 0.515 nats.
  - Every column holds at least two of the clades. The two clades of a zone are also partly separated in xy.
  - Each layer's share of the agents follows its share of the food. This is a correlation; food as the cause was not
    tested.
- **Buoyancy cap.**
  - The heritable cap evolves down: the median falls from 0.50 to 0.19, and `HEAT_BUOYANCY` from 1.75% to 0.62% of
    all heat.
  - The surface clade keeps 0.44. The upper-water clade has 0.08 and climbs by pitching up.
  - The change is far beyond neutral drift; it cannot be separated from hitchhiking.
- **Bodies.**
  - Bodies are about 35× larger than in 2D: a median largest body of 2,582 from step 100k, against 74.
  - They are thin single-clade sheets on the surface or the floor. They are adhesion clusters, not organisms. Agents
    have 1.25 to 1.59 contacts within 3 units against 0.81 to 0.89 in 2D. The crowd is 12 to 13 times denser per unit
    of floor area summed over the column, but 1.7 to 4.7 times denser in the slabs where the sheets sit.
  - Crowding is not separated from bonding as the cause of body size, because the bonds hold the contacts together
    (erasing them dissolves the contact clusters). The 2D side of the contact comparison is from steps 1.49M to
    1.504M.
  - After every bond is cut they re-form within about 10 steps. Members do not move together and do not share
    energy.
  - Erasing bonds at every step kills about 15% of the population within 100 steps (attacks 5×). This fits bonds
    acting as a ceasefire.
  - `docs/3D.md`'s goal of bodies with real 3D shapes, `RESEARCH.md` P2, is not met.
- **Evolutionary activity.**
  - Activity is bounded in all 22 valid cells (Channon), robust to horizon, grid and shadow seed, as in 2D.
  - Lineage sorting is much stronger early and weakens over the run. It reaches the 2D ratio at cohort ages 100k and
    200k (late cohorts only) but not at 50k, and it is still trending up at every age.
  - The survivors' excess share of births is not above the heritability null in 3D (age 100k: −0.003 ± 0.002, 2D
    +0.021).
- **Against 2D.** This is preset against preset, not a controlled dimension test.
  - Clades fall to 10 by step 99,500, against 392,800 in 2D (1.8 to 4.4 times faster, depending on the threshold).
  - Predation stays high: kill share 0.50 against 0.13.
  - Emitters stay at 0.88 against 0.53. Whether the emissions carry information in 3D is untested.
- **Outputs.** The figures and tables are in `docs/results_3d/`. The scripts and full outputs, with the activity
  runs, are in `analysis_cache/64k3d_s0/` (git-ignored).

## Results so far: the four 64k3d runs (`docs/RESULTS_3D_RUNS.md`)

Four runs of 500k steps each, preset `64k3d`: `runs/64k3d_s0` (above), `runs/64k3d_s1` (seed 1),
`runs/64k3d_s0_half` (`fields.regen_max` = 850) and `runs/64k3d_s0_dir` (director live, Trickster, LLM every 25k on
`google/gemma-4-26b-a4b-qat`, rules version 2, code aef79c2). All four are clean: ledger exact on every row, no
overflow, 14.7 to 15.0 steps/s (median row; 14.3 to 14.8 overall). The half-food and director runs were each paused
and resumed once. There is one run per condition apart from the seed pair.

- **Replication.** Seed 1 reproduces the structure but not the details: of 105 checks, the claim holds in 56, in part in
  32 and not in 14 (verdicts by hand; 33 of the 56 also have every listed value within about 10% of seed 0's).
  - Reproduced: the capacity ratios, a floor-layer share of 19 to 23%, the two-ended depth profile, four founder
    clades (the split into two depth zones of two holds only in part), thin single-clade bodies that never wrap,
    bounded activity.
  - Not reproduced: seed 1 rose in 100k to 300k and has been level since (1.35× the aim over 400k–500k against 1.26×),
    turns over 1.4× faster, keeps a higher cap (median 0.285 against 0.190, after a rebound; its floor clades keep lift,
    not its surface clade) and has a giant floor sheet (median largest body 11,655 against 2,582).
- **Half food.** 0.63× the population, not 0.5×. `K` halves, but the population fills more of it (n/`K_est` 0.86 against
  0.67): the resource cap clips the full-food field, and agents cost 9% less per step. Predation thins (attacks per
  agent-step and predation ratio; the kill-share change is inside the seed gap) and the cap's erosion is delayed. Two
  clades survive, both spanning floor and top (clade–depth MI 0.09), so the two-ended profile does not need
  depth-partitioned clades. The control cannot say whether density sets body size: the largest body is 1.11× s0's, far
  inside the 4.8× gap between the two seeds.
- **Director.** Identical to s0 in every metrics row through step 25,000 and every event record before it
  (`DESIGN.md` §26.8, director enabled with inert levers equals director off, holds in 3D; state equality is inferred,
  the only checkpoint before 50k is step 0), unpaired after.
  - 30 interventions at its full budget: 12 LLM immigration pulses (the model never chose another lever although
    others were affordable), 15 runaway-damper pulses and 3 rescue entries during the founder overshoot.
  - The damper's trigger (largest body above 5% of the population) is met at most boundaries of every 64k3d run, so
    it fires every 25k steps. Each pulse cuts the population by 13% and the largest body by about a quarter, both
    temporarily.
  - Immigrants leave no lineage (2 descendants alive at the end). The end state (3 founder clades, top clade 0.66)
    cannot be attributed to the director. Coverage is higher (28 against 21 and 20), but it mostly tracks forced
    population oscillation (`EVALUATION.md`), which the damper's pulses also impose, and the half-food run reaches 26
    without a director; persistent novelty is below both director-free trajectories, and the cohort activity rate equals
    s0's.
- **Activity.** Channon's test is bounded in all 22 valid cells of all four runs; Bedau's class is unstable between
  runs.
- **Outputs.** The figures and tables are in `docs/results_3d_runs/`, the scripts and full outputs in
  `analysis_cache/64k3d_runs/` (git-ignored).

## Storm world: W0 and W1 (`docs/STORMS.md`, DESIGN §30)

- **Current state.** W0 (the offline weather model, `weather.py`, `docs/STORMS_W0.md`; row I46, PR #28) and W1 (the engine
  weather layer, DESIGN §30; row I47; branch `games/w1-weather`, built on `games/w0-storms`) are implemented and **off by
  default**. With `weather.enabled = false` the engine is bit-identical to its parent: the 2D and 3D digest matrix, the
  goldens (`GOLDEN`, and `GOLDEN_3D` over the new `legacy_state_digest_3d`) and every config hash are unchanged. A
  weather-on run is `--preset 64k3d --overlay storm`. What exists: a hashed `[weather]` section (33 keys), `State.wx`,
  wind as a drag force, rain into the regrowth (and so `K_est`), the storm hazard (`DeathCause.STORM`, `DEATHS_STORM`),
  explicit wind bond breaking, the `weather` metrics block, `alife bench --weather`, and the weather-only spin-up
  `alife run --weather-spinup N` (storms need about 15,000 steps from a calm start; 80 s on the CPU).
- **Results (CPU smoke; the GPU results are in the next bullet).** The 200-step `64k3d` + `storm` run at 1,024 rows closes the ledger and agrees compiled
  against eager on `wx`; the CPU marginal of the weather is +6.75 ms per step at that size (`bench --weather`). The
  hazard at the overlay is numerically inert (no storm death in a short run), the rain is a local source and a 0.09%
  global perturbation, and **the layer-2 jet of about 0.05 is on whenever the weather is on and is the dominant
  coupling**, not the storms (DESIGN §30.1).
- **GPU results (M5 Pro, 2026-10-21, W1 at 80b50d0; DESIGN §30.8, §30.9, §30.11, row I47 (4b)).**
  - *Metal gate: passed.* `pytest -m ""` on the default device: 797 passed, 2 skipped, 1 xfailed (the MLX duplicate-roll
    compile bug, still in 0.32.3), 0 failed. The fused weather-on step compiles on Metal, so the separate-advance fallback
    is not needed. The base-versus-head digest matrix, run as a script on the GPU, was bit-identical (0 DIFF, the same 116
    INFO lines as on the CPU; its pytest forms passed too). The whole fast suite on Metal (`-m "not slow"`, the I26
    failure deselected) gave 2,298 passed, 0 failed.
  - *Cost: target met.* `bench --weather` at `64k3d` + `storm` with the fused kNN kernel: marginal about 0.5 ms (0.47 to
    0.64 ms over five runs; single pairs up to 1.1 ms; in the two published runs 0.47 to 0.50 ms, 3.4 to 3.6% of the 13.8 ms
    weather-off step, 3.3 to 3.5% of the on step), 69.6 steps/s with the weather on. The unpaired `--stages` delta (1.02 ms)
    is noise-affected.
  - *Occupancy go/no-go: no-go on `C` = 64.* A fork of `64k3d_s0` at 450,000 with the `storm` overlay and a 15,000-step
    weather spin-up, 20,000 coupled steps at 68.6 steps/s: `max_occupancy` 18 (trigger 43), `cell_overflow` 0, no health
    bit, ledger closed, `deaths_storm` 144 (0.016% of the 883,002 deaths in the window, nearly inert as a cause of death at
    the provisional values; the population followed the weather-off source within 0.1% at 470,000, but `speed_mean` was 5%
    higher with the storms on). `cell_capacity` stays 48. `C` = 64 is
    free of step time under the fused kernel, so it is available if W2's repeat (calibrated `storm_*`, a young
    population, 30,000 steps; this run was 20,000 from a spread population) fires the trigger; the other response is a
    lower `u_max` or `alpha` (C17). Neither was applied.
  - W1 is merge-ready on the §30.11 criteria; the merge is the owner's go (the G2 items are applied, see "Merge order").
- **Next (W2).** *(W2a, the code, is built: see "Storm world: W2a" below; the calibration and the runs are W2b.)* Calibrate `rain_gain` and `storm_*` on a still-air control of each seed (`storm_u0` in 0.03 to 0.05,
  `storm_p` from the `weather` block's `storm_exposure` and `air_speed_p99`); **a jet-removed (`weather.r2` raised) or
  tracer arm before any ecological effect is attributed to storms**; rain seeding and shade with the same band weights as
  the food; wind-relative observer fields (`INTENT_VERSION` 2). Director runs need `rain_floor = 0` (enforced).
- **Merge order (done on the branch).** G2 (`games/g2-social`) was merged into W1's branch before the W1 merge, and the G2
  follow-ups of the W1 build notes (`g2_followups.md`) were applied there: `DEATH_CAUSES` stays the pre-storm tuple and
  `death_causes(cfg)` adds `storm` under weather (a `STORM` row would otherwise index past the cause scatter),
  `compare_row(weather=...)` in `analysis/social.py`, regenerated GUI fixtures, and the weather-off identity of the
  observer sidecars and `social.jsonl` (DESIGN §30.7).

## Storm world: W2a (`docs/STORMS.md`, DESIGN §32, row I50)

- **Current state.** W2a is built on `games/w2a` from `origin/main` 3799009 and merged with `origin/main` after N1 and the 2D kNN default (not merged into main; the PR is the owner's go). Off by default,
  and with every new key at its default the engine is the W1 engine bit for bit (digest matrix, eight weather-on trajectory pins and the tiny3d
  legacy digest `556de6c6…` against a fresh archive of 3799009). The observer layer changes by design (`INTENT_VERSION` 3, `OBSERVER_FORMAT` 4,
  `Acts.buoy`). What exists:
  - **Eight `weather` keys** (the section has 41): `seed_gain`, `seed_c0` (rain seeding by the c2 excess, with the food's layer weights),
    `shade_gain`, `shade_n0` (the shade of the agents on `T_s`), `wind_scale`, `rain_zonal`, `storm_eddy` (the hazard and bond rule on the speed
    relative to the zonal mean) and `wind_eddy_force` (the agents feel the eddy wind only). The step changes only at S17. No ledger term.
  - **Observers, `INTENT_VERSION` 3** (`RULES_V3`, digest `3515d96e…380a17c`): Nomad and Settler on the velocity relative to the wind (`weather.agent_drift`),
    Drifter, Floater and Sinker (enum 8 to 10), `Acts.buoy` (a 21st tap in 3D; Metal purity gate passed on the stage-3 tree, to be re-run at the final head before W2b, DESIGN §32.11), `ARCHETYPE_STORM_DISPLAY`
    (Gatherer is "Grazer" in a storm world), `labels --rules {v2,v3} --dump DIR`.
  - **Instruments:** `game.tracers` (passive tracers, its own key), `run.eco_every` and `run.wx_every` (the ecology dump with `emit` and the ten weather planes),
    the `coupling` block (`ub_agents`, `hold`), `bench --weather-org`, and `python -m alife.analysis.w2` (P1a to P6 with block t-intervals and the s1 A - N noise floor).
- **Pilot (26,000 steps from `64k3d_s0` at 450,000, the prior coupling set).** Ledger closed in all rows, `max_occupancy` 16, storm deaths 3.2% of
  deaths; on the window 470,000 to 476,000 all ten intents and nine archetypes pass (Drifter 0.880, Floater 0.591, Sinker 0.637 against nulls of
  0.51 to 0.53; the Floater is scored off the lid, and by band it is evidenced only at z 56 to 78.9, DESIGN §32.8); the physics-derived thresholds were kept (so
  nothing was tuned; the keep rule is not a validation, the held-out windows are). Cost on Metal: observers 0.727 ms, organism couplings 0.69 ms.
- **Not done (W2b; needs the owner's go, about 10.7 GPU hours and 119 GB):** the occupancy gate G0 (before the calibration; remedy `cell_capacity` 64 is hashed),
  the calibration of `rain_gain`, `storm_*`, `seed_gain`, `shade_gain` (`presets/storm.toml` still holds W1's provisional values), the 14 runs (arms A, N, S, WB, A1, E on
  s0 and s1, A and N on s2, 100,000 steps), the held-out label windows and `LABEL_VERDICTS` rows, `docs/RESULTS_W2.md`. The commands need
  `--set game.observe=true --set game.tracers=16384 --set run.eco_every=250 --set run.wx_every=500 --set run.keep_checkpoints=5` (DESIGN §32.12).
- **Open for the owner** (the eight decisions of the W2 spec were taken as recommended by the coordinator, to revisit; DESIGN §32.12): which seeds (s2 added for A and N),
  the eddy hazard, `RULES_V3` as the default for every world, run length, the arms (WB and E added), the PR split and the GPU go, **seeding is density-driven**
  (crowds seed whatever their genes: STORMS §4's "sub-threshold emission alone does not trigger rain-out" is not met; P1c reports the paid share and a hinge is a later variant), and the bond break is not calibrated.
- **Finding for W2b.** `Diag.WEATHER_WARN` is sticky (set from step 456,100 on in the pilot): a stop rule "WEATHER_WARN in more than 1% of rows" would fire on every run that
  warns once; read the rate from the weather block (`analysis.w2` reports the `hyper_max` share).
- **N1 consequences are applied** (commit `W2a: N1 follow-ups`, after `origin/main` with N1 #31 and the 2D kNN default #33 was merged into the branch; DESIGN §32.13): `report._label_display(key, version, storm)`
  prints Grazer, Drifter, Floater and Sinker in a weather-on V3 run (Gatherer when weather is off); the lexicon's L2 list has the four words and the L1 `drift` stem no longer matches "drifter"
  (`LEXICON_VERSION` 2, new `LEXICON_DIGEST`); `LABEL_VERDICTS` has **no V3 row**, so N1 narrates no label for a V3 run until W2b's held-out windows add rows (the Floater one by z band, §32.8).
- Checks of the build: fast suite 2,944 passed (2,697 at the W1 head), `ruff` clean, `swift test --filter AlifeCoreTests` 93 passed. The CPU design numbers (seeding and shade dose-response,
  who is exposed to the jet, the drift of agents) are in DESIGN §32.1.

## Games G3a: the play loop and the World window (`docs/GAMES.md` §9, DESIGN §33, row I51)

- **Current state.** Built on branch `games/g3a` (15 commits on `origin/main` 2d1ab0c: G2, W0/W1 weather, N1, the fused kNN kernel), then
  merged with `origin/main` 39301ba (W2a #34, the 2D fused kernel #33; DESIGN §33.7), the sync review fixes (`b7020a6`) and the
  **Mac day, done 2026-10-22** (`docs/RESULTS_G3A.md`, commit "G3a: Mac day results" on top of `b7020a6`). **The PR is to be opened; it is not merged.**
  The labels of the stream are `INTENT_VERSION` 3's and carry
  N1's validity gate (no V3 verdict row exists, so no label is shown as valid; a lever away from its default also withdraws them).
  `alife play` runs one world under a fixed-timestep scheduler and writes ALSTRM records (`FRMA IDNT INTN TERR EVNT STRM MAP1 ENVS CLCK
  FOCS`) to stdout; the **World** window of the Swift app (Lab menu, "Open World", or `--world RUN` at launch) shows them: the planet
  map and organisms in a raw-Metal view (minimum image, alive-only frames, three semantic levels), a docked control center (clock
  strip, Cast, Chronicle, the inspector pinned under them), picking and follow with the hand-off at death, toasts and keys. Presentation
  never changes a trajectory (digest equality with a plain run is tested on the CPU and, on the Mac day, on Metal in four bench arms).
  **Mac day: the §33.5.4 gate passes; Metal purity (19 files) and the Swift tests pass; the click-through by hand is the owner's and not done.**
- **What to know before touching it.**
  - Planets: `alife play --init-from SRC:STEP --overlay storm --set director.enabled=true --weather-spinup N --out DIR` (the weather
    needs a spin-up; houses are rebuilt exactly from the source's event chunks, `--houses exact|clade|auto`). `play.json` is written
    last (a directory without it is unfinished), `play.lock` is held through the quit checkpoint, a research run is forked, never
    played in place (`--adopt` only with a backup of `config.json`).
  - `runner.run(pacer=)` is the one engine-side hook and is inert when `None`; a pacer callback never raises into the loop (the
    runner has no `try/finally`). `RunDir._prune` now keeps a fork's origin checkpoint.
  - The Swift package has no CI job: run `cd gui && swift build -Xswiftc -warnings-as-errors && swift test` before a push (AlifeCoreTests
    and AlifeControlTests; the GPU tests are opt-in with `ALIFE_GPU_TESTS=1`).
  - Measured on the Mac day (64k3d storm planet, 44,955 alive): 1x holds (1.00006x, `late_steps` 0, `step_ms` p99 47.6 ms), `max` is
    3.68x (53 steps/s), the boundary step is 127.6 to 130 ms at 1x (about 40% over the top of the estimate; the scheduler's debt absorbs it), a
    checkpoint stalls 0.93 to 0.98 s (13 slots) on the internal disk, the client's draw p95 is 2.41 ms. The chronicle runs at 8.66
    events a minute (ring of 500 = 58 minutes; a toast about every 18 s), **mostly kin splits and merges that flap** (112 of 201 events;
    69 of the 77 importance-2 events). Still unmeasured: the external-disk checkpoint stall, the present-to-present draw time in a real
    window, a release build, 2x. The planet of the Mac day is a fork at **450,000**, where D2 says 500,000. The labels shown are `INTENT_VERSION` 3's (Drifter,
    Floater, Sinker; Grazer in a storm world) and are tagged provisional unless the header's gate (`LABEL_VERDICTS`, as N1) lists them and `ENVS.levers`
    is empty; today none is listed, so all are provisional. The chronicle never states one.
- **The click-through, the owner's (not done; the other Mac-day results are in `docs/RESULTS_G3A.md` and DESIGN §33.6).**
  Start `cd gui && swift run`, Lab menu "Open World" (or `--world RUN`). The launcher lists `runs/`: make a new planet from the launcher
  (a fork of `64k3d_s0` with the 20,000-step spin-up, about 21 s; the Mac day's `p0` is in scratch, outside `runs/`). Look for:
  1. **Launcher:** a world listed with its step, source and houses; Resume starts at 1x; cancel a build (it becomes "unfinished"); a
     world that is saving is "saving..." and Resume is refused.
  2. **Clock strip:** Pause, 0.25x, 0.5x, 1x, 2x, Max, Resume (rate unchanged), `+1`, `+25` when paused; the readout settles ("0.98x of 1x");
     "late N" and "saving..." at a checkpoint. Does 2x stutter at the boundary steps (3.7 slots there, not benched)?
  3. **Picking at each level:** planet (storm, warm pool, territory cell; no organisms), region (organism: click follows, shift-click
     selects only, double click flies in), organism; hover tooltips; the map fade (0.4 s).
  4. **Follow and the hand-off:** follow an organism until it dies (the selection moves to its successor, the camera holds the
     death point meanwhile), a storm until it dissipates, a lineage.
  5. **Cast and Chronicle:** the eight largest kin groups and live storms; hover highlights; the Chronicle's groups, filters and
     click-to-fly. **Judge the kin churn**: is the Lineages filter usable, should kin events be off by default?
  6. **Toasts:** 3 s, at most 3; at the measured rate one every 18 s, nine in ten of them kin.
  7. **Keys:** `space . c 1 2 3 f [ ] r Esc`; the View menu's archetype and intent colour modes say "(provisional)"; `[` `]` do nothing paused.
  8. **The dock:** each side, the divider, collapsed (one-line clock), and the window at its 700-point minimum height (the inspector card
     may crowd the Cast list; not rendered).
  9. **Close and quit while saving:** the app waits for the world (up to 40 s), the checkpoint completes, no `play.lock` is left.
  10. **Resume** a world from the launcher, after quitting the app.
  11. **Cosmetics seen offscreen:** lineage detail lines truncate ("died: 3091 starved, 3099 killed, 3..."), "6.216 born" reads as a decimal
      under a locale thousands separator, the window title showed a run named `r`.
  Also open: the 2D digest matrix on the GPU against `39301ba`, `tests/test_bench.py` and `tests/test_weather_w2_instruments.py`
  on Metal (the external-disk checkpoint stall was measured afterwards: 5.6 to 6.4 s against 0.4 to 0.5 s, `RESULTS_G3B` §6.1; the G3b Mac day ran `tests/test_bench.py` on Metal, 46 passed; the other items are still open, `RESULTS_G3B` §10).
- **Next.** (1) G3a was squash-merged to `main` as 9f84f7c (the owner's click-through of its window is still open); (2) The kin-churn finding was answered by G3b's kin threads (kin events 5.99 to 2.04 a minute on the Mac day's window; satellite budding stays, the owner's decision of 2026-10-23 on item 15: kin groups that bud off a large group count as new groups, `P` stays 3). (3) **W2b waits for the owner's go** (the occupancy gate G0, the calibration, the 14 runs, the held-out
  label windows, `LABEL_VERDICTS` rows; about 10.7 GPU hours and 119 GB); until then every label shown is provisional. (4) G3b is built (the next section).
- **G3b.** Built: see the next section. Not yet: N1's select mode, canonicalised saves, recap files and rewind are G3c. W2 (the coupling values and the labels of storm worlds) only
  changes values the records already carry.

## Games G3b: chapters, alerts, the live report and direct powers (`docs/GAMES.md` §9, DESIGN §34, row I54)

- **Current state.** Built on branch `games/g3b` (nine commits of the spec, in seven stages, then the stage 8 review fixes, the merge with `origin/main` 3a14aa6 (G3a squash-merged as 9f84f7c, then #36), a CI timeout of 120 minutes, an injected clock for one flaky Swift test, and the sync review fixes; `019fdde`); **no PR is open for it and nothing is pushed**. The specification is `g3b/spec.md` revision 2 after two
  reviews; every owner decision was taken as the coordinator's recommendation (choice **C20**, to revisit). **The Mac day ran 2026-10-23 with no code change** (`docs/RESULTS_G3B.md`, DESIGN §34.8, §34.9, row I54): Metal purity 34 files / 1,393 passed; the bench gate passes at 1x, `max` and 2x except the report-time item, which the bench cannot read (an arm has no history), and which the played session measured at 29 to 47 ms;
  an 18,001-step scripted play of `p0` was clean on the wire (0 decode errors, 0 late steps) with the chronicle, kin and alert rates inside the spec's bands. **Not done:** the second window (`64k3d_s1`), the click-through, and any play by a person: **the window has been rendered offscreen and looked at, and no person has used it.**
  **Two defects and a decision came from the Mac day**: the report's chapter number counted from step 0 on the server (the Mac-day render of the Report tab showed it; **fixed afterwards in `G3b: report chapters count from the planet origin`**, `PlayOptions.origin_step`, DESIGN §34.2.1); the bench's report time is an arrival notice (not fixed, see "Next, in order"); and a population pause followed the player's own Blight, which **the owner decided on 2026-10-23 is the intended behaviour**: a fall of 15% or more pauses whatever the cause (DESIGN §34.3.2). The owner also decided items 15 and 16 that day (see "Next, in order").
  G3b is **observe-and-poke**: no score, no objective, no win state (D9). What it adds to G3a:
  - **The player channel** (engine-side, the one risky edit; it landed alone): `Director.apply_pending(state, t, stopping=)` at the top of every runner iteration, `run.director = "player"`, rule P (a player's record at `s` is after the checkpoint at `s`: `RunDir._director_cut`, the replay's
    pending list, `_replay`), `GAME_SPECS` "game-v0" in `director/actions.py` (an override of `MENU`, not a subset), "timed" = the entry has a duration, the Bloom revert, `power cancel`. Inert when unused: the 2D digest matrix against `a8956b9`, the tiny3d digests and the nine research director logs are bit-identical (`tests/director_identity.py`).
  - **Chronicle quality** (`game/kinthreads.py`, storm persistence, `CHRONICLE_VERSION` 2): 139 kin rows to 45 told events on the Mac-day window; **chapters** (`play.ChapterClock`, reason 3 and a pause at the next `before_step`, `autopause`, the write lock); **alerts** (`game/alerts.py`, 18 classes, the provenance rule, `alerts.jsonl`);
    **report v2 and the report job** (`game/report.py`, `reportjob.py`: a worker thread started in `before_checkpoint`, `reports.jsonl`, `RPRT`, byte-equal to `world_report verify`); **powers** (`game/powers.py`, `precedents.py`: `PLAN`, since cards, influence, `commands.jsonl`); the records `ALRT RPRT PLAN`; and the lines `autopause watch unwatch ack power intent`.
  - **The window** (DESIGN §34.7): chapter bar, alert stack, chapter card, Report and Powers tabs, aiming on the map, keys `g p a k Return` and a staged `Esc` (it never resumes or acknowledges), ticker, dock icon and badge, the pauses the player chose, and the three cosmetic fixes of the G3a Mac day.
  - **The bench** (DESIGN §34.8): `alife bench --play` gains the `player` director mode, `--play-powers`, `--play-resume SRC[:STEP]`, `--play-chapter N` and a chapter, report and alert line.
- **What to know before touching it.**
  - **A power is the only line that changes a trajectory.** Everything else (`autopause`, `watch`, `ack`, `power preview`, a refused power, the chapter pause, alerts, reports, the ticker) is presentation and is tested by digest equality with a plain run. A change to `play.py`, `powers.py` or the runner's hooks must keep `tests/test_play_purity.py`, `test_player_channel.py` and `test_play_powers.py` green.
  - `alife play` launches a world with `director.enabled` in `player` mode (`run.director = "player"` is written to its `config.json`); a world without it stays `off` and answers a `power` line with a refusal.
  - A fresh **preset** world raises `pop_fall` at importance 3 in its founding die-off and stops there (the tiny preset at step 500); planets, which are forks of a settled world, do not. A grace of a few rows is the fix if presets are played (the owner's call).
  - The Swift records, reducers and views: `PlayRecords.swift` (the new records), `WorldStore.swift` (`AlertBoard`, `ReportShelf`, `PlanLog`), `AlertStack.swift`, `ChapterBar.swift`, `ChapterCard.swift`, `PowersModel.swift`, `WorldPresentation.swift` (all pure, in `AlifeCore`), and in `AlifeControl` `WorldModel+Decide.swift` and the `World*Views.swift`.
    The side dock's fit at 700 points is budgeted by `DockBudget` from numbers measured through `NSHostingView` (`theSideDockFitsTheMinimumWindow` fails if a part outgrows its budget).
  - `tests/test_gui_sync.py` pins the key table, the power names and line keys, the header's parameter names, and every Swift record type's JSON keys against the encoders (`disc` is optional in Swift: only a Strike's card carries it).
- **Checks of the build.** Fast suite (CPU, `-m "not slow"`, `test_lifecycle_compiles_and_matches_eager` deselected; six shards of the test files in parallel): 4,219 passed, 89 skipped, 37 deselected, 1 xfailed (G3a's head: 3,774 at the first G3b commit); whole-repo `ruff check` and `ruff format --check` clean; Swift `swift build`, `swift build --build-tests` and `swift test` with `-Xswiftc -warnings-as-errors`: AlifeCoreTests 465 tests in 46 suites, AlifeControlTests 240 in 26. The decide half was also played against a real `alife play` process (CPU, tiny3d) from a throwaway Swift test: the chapter card, a Food and a Strike aimed on the map.
- **Checks since the build (merge and sync, CPU).** Fast suite 4,252 passed (4,234 before the chapter-number fix), 89 skipped, 37 deselected, 1 xfailed; `ruff` clean on the whole repo; the 2D digest matrix against a fresh archive of 3a14aa6 and the tiny3d digests bit-identical; `swift build` and `swift test` with warnings as errors: AlifeCoreTests 469, AlifeControlTests 252.
- **Click-through list for G3b** (the owner; run the G3a list of `RESULTS_G3A` §9 first; use `p0` (the Mac day's clone is in the session scratch directory `scratchpad/g3b/macday/planets/p0`, which is temporary: copy it into `runs/` or make a new planet from the launcher); `alife play` launches the `player` mode):
  1. **The chapter card** at a chapter end (every 6,000 steps): "saving" first, then the report text and the buttons (Continue, Continue at 2x, Powers, Show on map); fold it with `Esc` (which never resumes); with `autopause` off, the notice. The card is translucent over the map in the offscreen render (storm labels and rings cross the text): judge it.
  2. **The alert stack**: importance first, fly-to, acknowledge (the check button and `k`), the watch star, "+N more"; a pausing alert is never hidden; the same two alerts show in the card's strip and in the dock while the card is up.
  3. **The Report tab**: the headline, the body, the Units/Ready/Alerts block, the ticker, "Earlier chapters". It read "Chapter 79" on `p0` where the bar says "Chapter 4" on the Mac day (`RESULTS_G3B` finding 1); fixed since, so it should read "Chapter 4 · steps 468,000 to 474,000" there (a world made before the fix is numbered by its `play.json`, so nothing is to be migrated).
  4. **The Powers tab and aiming**: Food, Bloom, Blight, Strike with their units, cooldowns and busy reasons (Food is busy at the start of `p0` until step 476,000 because of the rule director's pulse), aim a Strike and a Bloom on the map, preview, confirm and cancel, "End early", the since cards in the ticker.
  5. **The keys** `g p a k Return` and the staged `Esc`; the dock icon and its badge; the pauses the player chose (`AutopausePrefs`).
  6. **The dock at 700 points** with the inspector open and three alerts (the report list is cut mid-line above the pinned card in the render; it scrolls).
  7. **Whether you act at all.** Play `player` mode for a few chapters and look at `commands.jsonl` (D9): the Mac day's inputs are a script's.
  8. **The window title** shows the run's name and the full source path of a fork ("Planet r forked from /private/..." in the render).
- **Next, in order.**
  (1) **Fix the one defect of the Mac day that remains** (the report's chapter number counting from step 0 is fixed in `G3b: report chapters count from the planet origin`: `PlayOptions.origin_step`, tested with a planet made at a non-zero step): make the bench's report-time item meaningful (copy the source's metrics rows and origin row into the arm, or refuse a chapter the arm cannot report). The fourth pause is decided and needs no work: the owner decided on 2026-10-23 that a population drop of 15% or more pauses the game whatever the cause, including right after the player's own power (the stage-8 rule as built; the spec's band is superseded for such falls).
  (2) **The owner's click-through** of the G3a window (`RESULTS_G3A` §9 item 1) and then of this one (the list above), and a few chapters played by a person for D9.
  (3) **Owner items 15 and 16**, measured on the Mac day and **decided by the owner on 2026-10-23, no behaviour change**: kin groups that bud off a large group count as new groups in the chronicle (`P` stays 3, `KINTHREADS_VERSION` unchanged; a thread Pruashol formed four times and merged back three times between steps 474,750 and 487,250, 4 of 19 ended threads lived 1,500 steps or less), and storm namings stay on the alert surface (4 of the 8 alerts of importance 2 or more). Nothing to do; revisit only if the first play-through says the stack is noisy.
  (4) **The leftovers of the Mac day** (for the next one; the checkpoint on the external drive was measured afterwards, 5.6 to 6.4 s, `RESULTS_G3B` §6.1): a fork of `64k3d_s1` as the second window, the client's draw beside a running engine, the 2D digest matrix and `test_weather_w2_instruments` on Metal.
  (5) **G3c spec after the owner plays G3b** (DESIGN §35, row I55 reserved): recorded replay files, rewind, offline advance, canonical saves behind a measurement, the select-and-order report mode. **W2b waits for the owner's go** (the occupancy gate G0, the calibration, the 14 runs, the held-out label windows; about 10.7 GPU hours and 119 GB); W4 as before. The decisions to revisit: C20 (every recommendation of review round 2), the fresh-preset grace, the objective (D9), the Strike cooldown (D10).

## Games storage: the size monitor and the save mode (DESIGN §34.10, row I56, `docs/GAMES.md` §10 item 22)

The owner's first play asked for a live size of the run on disk and an easy way to stop "recording". A played world's disk is about 99% saves
(686 MB per checkpoint on the planet, the last three and the origin kept), the rest logs. **Built:** `saves all|newest` (a control line,
`alife play --saves`, `play.json` `saves`, `RunDir.keep_newest`): `newest` keeps only the newest save and the origin, prunes at the next
save (never when the line is sent), keeps the logs; echoed by the header's `play.saves` and `CLCK.flags` bit 2; logged to `commands.jsonl`.
The window's **Storage strip** under the clock strip shows the world's size (saves and logs, growth since the start and this chapter, free
space, a warning under three saves of room), measured by the client from the directory, and carries the **Recording** switch (on keeps the last saves, off keeps the newest only).
No trajectory change (`test_play_saves` toggles the mode at random against a plain run). `DockBudget` was re-measured: two alert rows still fit
beside the inspector at 700 points, three need 760; the bottom dock's least height is 192. Review fixes: a research run played in place
(`--adopt`, `config.pre_play.json`) refuses a `saves newest` line and the window does not offer the switch there (`play.saves.adopted`);
`--saves newest` says on stderr which checkpoints its first save deletes; in `newest` the new save is flushed past the drive cache
(`F_FULLFSYNC`) and read back before the old one goes; a killed save's temporary files are removed at the next save. Accepted limit: after a
quit and a resume in `newest` mode the first chapter report refolds the social log of the play (output identical). **Not run:** the strip
beside a real planet on the Mac (a person pressing the switch during a chapter end), and the cost of the `F_FULLFSYNC` of a 686 MB save
(one drive flush a chapter, in `newest` mode only).

## The mini world (DESIGN §37, row I62, `docs/GAMES.md` §10 item 23)

The owner asked (2026-10-24) for a smaller test world for quick play-tests and lighter streams, launchable from the World launcher, and said yes to the specification's go/no-go
(`scratchpad/smallworld/spec.md`, the coordinator's choices C22, to revisit). **Built (branch `games/mini-world`, one PR "M1", no GPU needed to review):** the preset `mini3d` (`64k3d` with
five values changed: a 160 x 160 x 80 column, `n_max` 16,384, about 10,500 alive, 4.5 ms an engine step with the weather on against 14.9, a 4.6 MB/s stream against 21.2), the world overlay
`storm160` (80 weather cells, two 24 x 16 pools per hemisphere, `f0` 0.0165), the launcher choosing the overlay and the spin-up (36,000 for 160 worlds) from the source's `world.size`
(`PlaySource.worldSize`, `PlanetSpec.overlay(forWorldSize:)`), `Preset.mini3d` in the Lab, and two host-side edits (the stream header's `weather.cells` and `dx` come from the layer;
`storms.detect` measures vorticity in units of the reference f0 0.025 in every world, so a storm is named at an absolute 0.0125 and `storm_strong` fires at 0.0175). **No simulation change**: no
hash, golden or digest moved (`tests/test_weather_config.py` pins the twelve old preset hashes and the new ones). The W0 study driver shares `storms.detect`, so its `zeta_max_f0` columns follow the reference
unit (§37.4).

- **Run it:** `uv run alife run --preset mini3d --steps 500000 --seed 0 --out runs/mini3d_s0` (31 minutes idle, up to about 45 shared), then make a planet from a checkpoint with
  `--overlay storm160 --weather-spinup 36000` (or from the launcher, which passes both). `runs/mini3d_e0` is a `cp -c` copy of the 80,000-step probe run (9,688 alive, one dominant kin group) for a first play
  before the evolved source exists.
- **The evolved source** `runs/mini3d_s0` (seed 0, dry, started 2026-10-24 on the code of this branch; `config.json` reproduces it): **finished, 500,000 steps, `status` done** (about 1 h 10 of wall on a shared GPU, 150 steps/s on average, peak 0.45 GiB; checkpoints 480k, 490k and 500k kept; metrics rows every 100 steps).
  Ledger error 0 in all 5,000 rows, no hash overflow (peak occupancy 40 of 48 at the founder boom, 13 later), no diag flag; alive 9.25k at 100 to 200k, 11.13k at 300 to 400k, **11.16k at 400 to 500k** (0.436 per unit squared; 64k3d 0.409
  to 0.412), lowest 3,051 (the first crash); clades alive 5 at 80k, 3 at 250k and 500k; the largest body 359 / 373 / 2,303 rows at 80k / 250k / 500k. **Kin acceptance at 250k: FAIL** (a single-snapshot reading of
  the checkpoint with the G2 rules, `scratchpad/mini/accept.py`, an upper envelope for the kin counts and the territory column alike; `analysis.social` over 1,500 steps tracks 3 to 4 groups in six snapshots, the top group at 0.53 in five and 0.98 in one): 3 kin groups of at least `enter` (sizes 5,284 / 4,442 / 38; the criterion wants 5), top share 0.534
  (wants below 0.6: met), 2 territory owners (wants 3; the owner counts here are an upper envelope: no EMA, owners of 7 to 14 cells included). It is two large factions: 300k 3 groups (top share 0.452, 3 owners), 400k 3 (0.545, 2), 450k 4 (0.430, 4), 500k 3 (0.533, 3). The spec says to build the crop
  tool (a 64k3d checkpoint cropped to 160 x 160: 6 to 10 kin groups in the probe) when this fails; it is not in this PR, and **its go is the coordinator's**. `runs/mini3d_s0` is usable now for a planet with two
  large factions (a fork of the 450,000 or 500,000 checkpoint); seeds 1 and 2 are not run.
- **Not done (a Mac day, `docs/RESULTS_MINI.md` to come):** the bench gate on a 450,000-step fork (the five-minute arm at 1x), `GPUWorldTests` and `WorldRenderBench` at L = 160, the rendered camera poses at
  L = 160 (the region fly-to falls inside the 80-tall slab on the unmerged `gui/viewer2-a1` branch), the dry-versus-weather fork pair, the slow tests and the W0 golden of the mini weather (DESIGN §37.7), seeds 1 and 2 to
  100,000 steps, the crop tool (only if the kin acceptance fails), a launcher entry that makes a world from nothing. The equatorial band stays a design track (`scratchpad/smallworld/spec.md` section 7).

## GUI follow-ups (PR #3, merged)

- **Fork pre-fill:** in Fork mode the levers are loaded from `runs/SOURCE/config.json`, where a missing
  post-v0.1 key means the engine default. Only levers that differ from the source are passed as `--set`, so a
  fork can also set a lever back to its default. An unreadable config falls back to the engine defaults and
  shows an inline warning. Leaving Fork mode resets the levers. The Steps field reads "Additional steps" for
  fork and resume.
- **Timer:** hours and minutes can be typed, and minutes step by 1 instead of 5.
- **Swift 6:** the package builds in Swift 6 language mode, and `RunController` is `@MainActor`. Pipe output
  reaches the main actor through one ordered AsyncStream per stream. The exit status is reported only after
  the process has exited and both pipes have reached EOF. The 1 s ticker is a Task loop that skips missed
  ticks after sleep. No `DispatchQueue.main` hops remain.
- **App bundles** (row I59, DESIGN §33.4.10): `gui/scripts/make-app.sh [--app lab|storm|both] [--install]` builds, signs
  (hardened runtime; Developer ID, else Apple Development, else ad-hoc; `--sign -` is ad-hoc and never reaches the keychain) and
  installs `AlifeControl.app` (the Lab, the default) and/or `Storm Worlds.app` (opens straight into the World window; the Lab
  is Window > Lab, Command-2; quits with its last window), each with a generated icon. The two share the repository folder
  (`RepoPathStore`, the preferences domain `io.github.magicbundle.alife`, migrated from the Lab's own on first start) and
  Storm Worlds asks for it itself when it is missing. `gui/scripts/check-apps.sh` is the smoke test of the bundles (macOS,
  ad-hoc, a release build; also `ALIFE_APP_SMOKE=1 pytest tests/test_app_bundles.py`). For this Mac only: **no
  notarization and no in-app updater yet** (the owner's "later"; he has a Developer ID); the App Store would need the engine
  embedded and sandboxed, which is a separate project. Not shared between the two apps: the dock side and sizes, window frames.
- **Click-through checklist** (tiny preset), to repeat after any GUI change:
  1. startup;
  2. a new run with the fast overlay and a `--set` lever, then Stop once: exit 130 and a checkpoint;
  3. resume for 2000 steps: exit 0;
  4. fork pre-fill from a run with seasons on, including setting a lever back to its default;
  5. the stop-after timer at 1 minute;
  6. Inspect, the viewer and Reveal in Finder;
  7. ⌘Q during a run: the engine still writes its checkpoint.
- **3D viewer click-through checklist** (3D PR 5, to add to the checklist above; `gui/README.md`, DESIGN §27.12; not yet
  run by the owner; use the `tiny3d` preset, and `tiny` for the 2D items; run the GPU tests first with the GPU free:
  `cd gui && ALIFE_GPU_TESTS=1 swift test --filter GPURenderTests`, which prints where the PNGs went):
  1. pick the `tiny3d` preset: a "3D world levers" box appears, pre-filled from the preset (height 24, gravity 0.03, ...),
     and the Command box shows no `--set` for the 3D levers until one is changed; change the gravity and check the
     command shows only `--set physics.gravity=...`; set Buoyancy max to 0 and check it shows `buoy_init` and
     `buoy_sigma` as 0 and flags a nonzero buoyancy cost; switch to `tiny`: the box disappears; fork a run made with
     Buoyancy max 0 and set it back to 0.04 (and the costs to 16 and 8): Start is blocked with a message about
     `buoy_init` and `buoy_sigma` (decision for the owner: make those two keys levers instead);
  2. start `tiny3d` (200 steps or more, `checkpoint_every` is 50): the status box adds the height mean, grounded and
     surface shares; "Open viewer" (pygame) is disabled for it with a pointer to the 3D viewer;
  3. **Open 3D viewer** while it runs: the window opens on Follow, "Starting alife stream" then the world: the box, a
     translucent floor and surface, spheres with a pale nose; the status line shows the step moving at the checkpoint
     pace; Pause and Run, "Newest" fetches the newest checkpoint once and the status line does not say "paused" (the run
     is still followed); the log shows `$ .../alife stream runs/NAME --follow ...` and the stream's messages as
     `[3d] ...`;
  4. when the run has ended, **Open 3D viewer** again: it re-simulates (Source "Re-simulate", the hint says it ignores the
     director); Pause, Step, `+` and `-` change the steps per frame, `c` cycles tag, energy, clade, `f` shows the field
     slice and `[` `]` move it up and down; bonds across the box edge are two short stubs, not a line across the world;
  5. mouse and keys: left drag orbits, right or Option drag pans, the wheel and a pinch zoom, `o` switches to Fly (the
     arrow keys move, Shift is faster), `r` resets the view, `1` `2` `3` toggle agents, bonds and frame; resize the
     window: the picture refits until the camera has been moved; zoom out with the wheel to many times the home
     distance: the box stays in view and is never cut off;
  6. `s` (or Snapshot): `runs/NAME/media/view3d_STEP.png` exists and shows the same picture as the window;
  7. a `tiny` (2D) run in the 3D viewer: the same picture as `alife render` (x to the right, y down); press `f`: the
     field is the background, as in `alife render`; with Field slice on at a high layer in a `tiny3d` run, open the 2D
     run: it opens (no exit code 2) with the slice on at layer 0;
  8. close the 3D window while it streams: no `alife stream` process is left (`pgrep -fl "alife stream"`); close the main
     window with the 3D window open: the app quits; with two main windows (File > New Window), closing one leaves the
     3D window and the app, and closing the last one quits; Cmd-Q with a stream and a run active: the run still
     checkpoints and no stream is left (also after switching Source or opening another run within two seconds);
  9. Open 3D viewer on a run directory with no checkpoint (a copy of a run with `checkpoints/` removed, `config.json`
     and `meta.json` kept; or Follow within a second or two of Start, before the step-0 checkpoint is written): the
     window shows the failure (exit code 4) and the next press works; a new run has its step-0 checkpoint from the
     start, so Follow shows it at once; GPU time while the picture is still (Activity Monitor): none, also while the
     run it follows is busy.

## Director panel (step 4, `docs/DIRECTOR.md` §14)

- **Settings (left column):** a Director box with Mode (New run and Fork: Off or Live; Resume: Off, Live or Replay),
  Persona, Goal (Architect), Rule every, an LLM toggle with LLM every, URL and Model, and **Test connection**
  (`alife llm-check`, one line in green or red). Baselines and the "only changed values are passed" rule work as for the
  levers; the command preview shows the flags. A resume of a run without `director.enabled` shows a note and passes
  nothing. Validation mirrors `config.validate` (rule_every multiples, llm_every, URL).
- **Live views (right column):** Interventions (last 10 of `director.jsonl`) and Journal (last 20 of `journal.jsonl`,
  LLM calls with the raw prompt and reply under a disclosure), tailed every 2 s from the last 256 KiB.
- **`alife llm-check`:** `OK model=... latency=...s` (exit 0) or `FAIL <kind>: <detail>` (exit 1); `DESIGN.md` §21.
- **Director click-through checklist** (tiny preset; add to the checklist above):
  1. New run, Director Live with the LLM toggle on: the Command box shows `--director live --llm-every N`, **Test
     connection** turns green with LM Studio running and red (`FAIL connection`) with it stopped;
  2. start it with Rule every 50 and LLM every 100: Interventions and Journal fill in while it runs, the disclosure shows
     the raw prompt and reply;
  3. Fork that run: the Director box is pre-filled and Live; change only the persona and check the command shows only
     `--director live --persona ...`;
  4. Resume it with Replay, then Resume a run made without a director: the note appears and no director flag is shown;
  5. a run without `director.jsonl` shows "Director off for this run";
  6. pick Persona Random: the LLM controls disappear, Start is disabled until a run is chosen under "Yoked to run" (only
     runs with a `director.jsonl` are listed), and the Command box then shows `--persona random
     --yoke-log=runs/NAME/director.jsonl`; switch back to Gardener and, for a Resume of a random run, check the command
     adds `--yoke-log=` to clear the key.

## Suggested next steps

1. **The §9 evaluation is done** (`docs/EVALUATION.md`). Follow-ups it found, in order:
   - **Season-aware extinction risk** (owner decision: a PR after the evaluation, with the `Director._blocked` journal
     fix; PR #23, merged as aef79c2; first used in `runs/64k3d_s0_dir`, `docs/RESULTS_3D_RUNS.md` §5.3): the risk flag fires on every seasonal trough deeper than 30%, so a strong season triggers a rescue every
     period, and the rescues use up the budget.
   - **A better outcome criterion.** The archive's coverage and novelty move with any forced population oscillation;
     lead with Bedau–Packard activity, or give the archive a heritable descriptor (genotypes or per-lineage behaviour).
   - **The Gardener's triggers** (stagnation: novelty below 0.02 and no new cell for 200k steps; monoculture: top clade
     above 0.9 for 100k steps) almost never fire on the 64k preset.
   - **Archive immigrants** (`DIRECTOR.md` §5, deferred): random founders do not establish.
   - Arm D (Architect) is deferred by the owner; if run, with a goal above the baseline (largest body ≥ 200). A
     seasons-only control arm is deferred until after 3D tuning.
2. **3D world** (`docs/3D.md`, DESIGN §27). It is built: PRs #15 to #20 are merged. Four 500k-step runs are
   analysed: `runs/64k3d_s0` in `docs/RESULTS_3D.md`, and the second seed, the half-food control and the director run
   in `docs/RESULTS_3D_RUNS.md`. Next, in order:
   - **A third seed** of `64k3d` (director off, about 9.5 h of GPU). Turnover, the cap's trajectory, which clade keeps
     lift, the steps to the final clade count and body size differed between the two seeds by 1.4× or more.
   - **The director's rules in 3D** (owner). The damper's 5% body threshold is met at most boundaries of every 64k3d
     run (median largest-body share 0.063 in s0, 0.27 in s1), so it acts as a 25k-step clock and takes half the
     budget. The rescue's 30% `fall` criterion reads the founder overshoot as risk (a warm-up exemption or the `K`
     criterion alone would avoid that). The season-aware path of rules version 2 is unexercised in 3D.
   - **The LLM tier in 3D.** The model chose `immigration_pulse` in 12 of 12 actions with other levers affordable, and
     immigrants never establish; archive immigrants (`DIRECTOR.md` §5) would make the pulse meaningful. A 3D director
     evaluation, if wanted, would use paired forks (`DIRECTOR.md` §9) after the damper is recalibrated.
   - **Bodies.** The half-food control cannot say whether density sets body size (its largest body is 1.11× s0's against
     a seed gap of 4.8×), and crowding and bonding are still not separated. Next: the bond reset and erase continuations
     (`RESULTS_3D.md` §5.5) on s1 and the half-food run, and an arm with `combat.attack_bonded = true` to test the
     ceasefire reading. A bodies goal needs measures other than size: persistence, coordination and sharing.
   - **`regen_max` and `n_init`** (owner). Full food gives 1.25 to 1.35× the sizing aim at n = 2; the population's
     elasticity to regeneration is about 0.65. No result requires a change; the sign-off of `n_init` is still open.
   - **Signals.** The signal-information tools are 2D-only (`require_2d`), and emitters stay at 0.86 to 0.88 in 3D at
     full food, so they need a 3D port.
   - **Speed.** Done: the fused Metal neighbour kernel (item 6, DESIGN §4, row I48) takes `64k3d` from about 14.8 to
     75 steps/s (a 100 000-step run in about 22 minutes), with bitwise the same trajectory.
3. **Games G1** (merged as PR #27, `1ad339c`; DESIGN §28, Review-log row I44, `docs/GAMES.md` §9). Built and tested on
   the CPU and on Metal. Status and what follows:
   - **The Metal purity gate: passed** on the M5 Pro on 2026-10-19. `tests/test_game_purity.py` and
     `tests/test_game_acts.py` gave 26 passed on the GPU, so observing leaves the compiled step's rounding alone on Metal
     too (§28.1, §28.8; row I44).
   - **Post-merge work: the 64k benches and the calibration** (procedures in §28.8, GPU needed): `alife bench --observe`, the stream rate
     and the step rate during an LLM call at 64k 2D; `python -m alife.analysis.labels` on `64k_s0` and `64k3d_s0`, which
     was the calibration that led to `INTENT_VERSION` 2 (item 4, DESIGN §28.8).
   - **Owner decisions** (`docs/GAMES.md` §10, items 11 and 12): thresholds as code constants, and the verdict criterion.
   - G2 (next item) is built on top of it.
4. **Games G2** (branch `games/g2-social`, DESIGN §29, Review-log row I45, `docs/GAMES.md` §9). Built on top of G1,
   tested on the CPU, then on Metal on 2026-10-21 (below). Status and what follows:
   - **GPU day, 2026-10-21** (Metal, M5 Pro, MLX 0.32.3; full figures in DESIGN §29.12 and §28.8, row I45 item 13):
     - **The Metal gate passed:** the twelve game and analysis test files with `-m ""` on the default device gave 509 passed,
       0 failed, in 202.7 s (the purity file with the compiled-kernel comparison, `test_groups.py` with the `splitmix64`,
       `hash_below` and exact-sum equality tests). The merge condition of G2 is met.
     - **Benches** (`alife bench --preset 64k --observe --steps 600`): 54.8 observed steps/s at 64k 2D against the 30 target
       (59.7 plain; G1 61.0), overhead 1.38 ms a step (G1 0.48), a snapshot step 9.4 ms slower than the others; 64k3d 13.9
       against 14.3, overhead 2.07 ms, hitch 12.1 ms (the 64k3d figures predate the fused kNN kernel, I48: with it 64k3d runs at about
       75 steps/s, so the same observer cost is about 15% of a step; a re-measurement on the GPU is pending). The 3D bench shows `donation.ok = false` in the plain run too: not an
       observer effect, not investigated. The stream rate and the step rate with LLM calls in flight were not run.
     - **Calibration** (`python -m alife.analysis.social`, `runs/64k_s0` from 1,490,000 and 1,500,000, `runs/64k3d_s0` and
       `64k3d_s1` from 450,000, 2,500 steps): kin groups track well (matched Jaccard p50 0.75 to 0.87; the kin-group
       matching costs 5.0 and 7.5 ms per snapshot, the plan said 2). **Colonies are the weak part at the 250 cadence:**
       80 to 83% of ids end within 3 snapshots and 21 to 26% of current components match a tracked one, so the 250 cadence
       churns; whether 1000 is better is not established (the `--every 1000` studies had only 2 snapshots, so their 22%
       (2D) and 31% (3D) are not comparable with the 10-snapshot 250 studies). The host settle costs 32 ms (2D) and 38 ms (3D) per
       snapshot on evolved states, 0.13 and 0.16 ms a step amortised. **No threshold was changed** (`SOCIAL_VERSION` stays 1);
       whether to change θ or the matching rule, or to use the 1000 cadence for colonies, is open for the coordinator and
       G3's chronicle.
     - **Houses:** `houses.founded_share` after 2,500 steps is 0.00068 (2D) and 0.00091 (3D), under 0.1% of members, so
       `game.house_rate` 1/2048 yields almost no new houses in that time; confirmed by the owner on 2026-10-21: keep 1/2048 for now (GAMES
       §10 item 13).
   - **`INTENT_VERSION` 2 is built** (the last stage of G2, DESIGN §28.4, §28.8, row I45): one change redesigned the rules
     of the labels that failed G1's validity study (Gatherer, Defending, Budding, Feeding, and in 3D Bonding, Signalling,
     Roaming and Caring), defined `RULES_V2` with the Settler clause at `settler_body` 4, made it the default and pinned
     its digest (`dc78f771…b4f7`; `RULES_V1` keeps `156ad496…c009`). It was calibrated offline on dumps of the two G1
     study windows, and the implementation is checked exact against the offline predicates. **Validated on the GPU on
     2026-10-21:** step A on the calibration windows passes (the seven exact predicates bit-equal to the offline values); on the
     three held-out windows (`64k_s0` from 1,500,000, `64k3d_s1` and `64k3d_s2` from 450,000) every archetype and every
     intent's displayed label passes the provisional criterion (30 of 30 intent cells and 18 of 18 archetype cells; thinnest
     margins 3D Resting +0.112, 3D Roaming +0.124, 2D Settler +0.218). Nothing was re-tuned and no label is marked as not
     validated. Caveats: the 3D Signalling predicate covers 0.674 of rows on `64k3d_s1` (over the 0.6 cap; the displayed
     label covers 0.171 and passes), and the 2D held-out window is the calibration run 4,000 steps later, not another run.
     The Metal purity gate and the v2 bench are in the GPU day above. **Open for the
     owner** (coordinator choices of 2026-10-19 under the delegation, to revisit): the Budding overlay, the display
     names Tethering (Bonding) and Unspecialised (Generalist), the meaning changes of Gatherer, Defending and Budding
     (GAMES §4.4), the Settler clause at 4, and the stability of the displayed label (measured on the GPU windows:
     7.05 and 7.07 changes per 1,000 steps in 2D and 9.2 to 9.6 in 3D, 3.6 and 2.2 to 2.3 times G1's 1.98 and 4.12; the replay said
     5.3 and 6.5). The verdict criterion (GAMES §10 item 12) is still provisional: the owner has not confirmed it; §28.8 applies it to the held-out windows as written, and it was fixed before their reports were read.
   - **The roster** (stage 7 of G2, DESIGN §29.14; stage 8 applied the review's findings): kin groups are named once at admission, the record lists the
     houses founded and extinct, and each listed kin group has its window births and deaths by cause, centroid and
     archetype mix. The GPU verification reported `houses.founded_share` from `python -m alife.analysis.social` on
     `64k3d_s1` and `64k_s0` (see the GPU day above), and the owner confirmed `game.house_rate` 1/2048 on 2026-10-21 (GAMES §10
     item 13).
   - **Deferred to G3**: the stream tags and colour modes, moving the host part of a snapshot off the step thread, the
     Family row (a host lineage store and an ancestry panel), and sharing the snapshot's components with the heavy
     metrics. Copying the sidecar and the social log into a fork is G3 or G4.
   - Then G3: G3a (play loop and view) is built on the storm world (item 7 below, DESIGN §33); G3b is built and its Mac day is done (`docs/RESULTS_G3B.md`); G3c waits for the owner to play G3b.
5. **Storm world** (`docs/STORMS.md`; W0 merged as PR #28, row I46; W1 is built on `games/w1-weather`, DESIGN §30, row I47, PR #32,
   and its GPU gate, cost and occupancy results are recorded in the section above). W2a (seeding, shade, the controls, the
   observers' version 3 and the instruments; DESIGN §32, row I50) is built on `games/w2a` (section "Storm world: W2a"). Next: W2b
   (the occupancy gate G0, the calibration, the 14 runs, the held-out label windows, `docs/RESULTS_W2.md`), then the V3 labels' held-out verdicts for N1 (its consequences are applied, `W2a: N1 follow-ups`). G3 and
   the later steps on the storm world wait for W2b.
6. **Games N1, the offline World Report** (branch `games/n1-report`, on top of `games/g2-social`; DESIGN §31, Review-log row I49,
   `docs/GAMES.md` §9). Built and tested on the CPU only; no engine change, no config key and no LLM call was made in the build. Status and what follows:
   - **What it is.** `python -m alife.analysis.world_report build RUN --from S --to S --out F.jsonl [--llm --model ID]`, `verify` and
     `probe`. Code computes a fact sheet from an **observed** run and writes every sentence from 88 templates; a checker (R1 to R7)
     re-derives each string; `--llm` lets a local model only select and order candidate ids (coordinator choices C12 to C14 under the
     owner's delegation, to revisit); free prose is off. The block of interventions is the engine's lever state at the start of the chapter plus
     the chapter's entries, and the `env` check ties it to the rows (exit 1 on a disagreement).
   - **Checked on real data (CPU).** The lever fold equals the `env` block of every row of `eval_s0_C`, `eval_s1_C` and `64k3d_s0_dir`; the block lists every
     lever the rows show away from default in all 540 chapters examined, with 0 checker exclusions; a real tiny observed director fork
     (`tests/data/report_e2e/`) builds and verifies with exit 0 (DESIGN §31.13).
   - **Review fixes (stage 4, `N1 stage 4: review fixes`).** 15 findings of the third review, 14 applied and one narrowed (DESIGN §31.14 item 15;
     the disposition is `n1/build/disposition.md`): a patch move is a point event with no persistence claim (S-pos deleted); a span that ends at the
     chapter's first row is not listed; HP-dip; the relation sentence; `--out` merges instead of truncating; a request goes only to a chat URL;
     malformed input is exit 2. `tests/data/report_e2e/golden.json` and `TEMPLATE_DIGEST` were re-pinned.
   - **Next, the coordinator (GPU, when no run or timing is going; `lms ps` first):** the validation of DESIGN §31.13: fork A (`64k_s0:1500000`, no
     director) and fork B (`eval_s0_C:120000` with `--director replay`), both `--set game.observe=true`; then `build`, `build --llm --model
     google/gemma-4-26b-a4b-qat`, `verify` and `probe --n 24`; then `pytest -m llm --model google/gemma-4-26b-a4b-qat tests/test_analysis_world_report.py`.
     Report facts per chapter, exclusions by reason, checker exclusions (target 0), the measures of §31.11 and latency.
   - **Owner read** (20 chapters, template against select, blind): decides whether the select mode is kept, whether the window sentences W are shown,
     and whether the template reports are fit for G3 to show. Proposed targets are in DESIGN §31.12.
   - **Open:** the owner's confirmation of C12 to C14 (the one read-only `GET /api/v0/models`); no label is narrated for a `RULES_V2` run until its verdict
     rows are added by the change that validates `RULES_V2`; the provisional thresholds of DESIGN §31.14.
   - **G3** consumes the Python API (`build_sheet`, `narrate`, `ticker`) and appends `reports.jsonl` at the run root; the report never writes into a run directory.
7. **Games G3a** (branch `games/g3a`, DESIGN §33, row I51; the section "Games G3a" above). Built and tested on the CPU and, on the Mac day
   (2026-10-22), on Metal; the gate passes (`docs/RESULTS_G3A.md`). Open: the PR, the owner's click-through (checklist in that section).
8. Optional engine work, deferred in DESIGN §23.2:
   - done: the fused Metal kernel for the neighbour search (`src/alife/knn_metal.py`, DESIGN §4 and F21, row I48;
     `ALIFE_KNN=auto|mlx|metal`, the kernel is the default on Metal in 3D and, since row I52 (owner decision 2026-10-21), in 2D too:
     64k 2D 16.3 to 10.8 ms per step, bitwise the same trajectory; `ALIFE_KNN=mlx` forces the reference path).
     It is exact only on the MLX versions in `knn_metal.VERIFIED_MLX` and a startup self-check guards the rest. **On
     every MLX bump and macOS major update**, run `uv run pytest tests/test_spatial_metal.py -m "" -q` on the Mac (the
     GPU equality tests; they run on an unlisted version), then add the version to `VERIFIED_MLX`, or leave it out;
     the remaining custom kernels of DESIGN §23.2 item 1 (tiled GEMV, splat renderer) are still deferred;
   - a mean-speed metric, so "sessile vs mobile" is measurable;
   - `inspect` showing the resumed total instead of the original `steps`.
9. If a change alters trajectories, re-pin the golden digest deliberately (tests/test_determinism.py) and note
   it in the DESIGN Review log. A trajectory-neutral state field is instead frozen out of the legacy digests
   (`V0_FIELDS`, `V0_FIELDS_3D`).
