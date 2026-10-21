# Handover

Status of the `alife` project as of 2026-10-21 (the games vertical's observers are built, G1 and G2, `docs/GAMES.md`; after the director evaluation, `docs/EVALUATION.md`; the 3D world is built, PRs #15 to #20, `docs/3D.md`; its first long run is analysed in `docs/RESULTS_3D.md`, and a second seed, a half-food control and a director run in `docs/RESULTS_3D_RUNS.md`), written for the next coding
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
| `src/alife/` | engine (step, physics, bonds, energy, lifecycle, fields, brain, ...), run tooling (runner, runio, metrics), render, viewer, bench, CLI, the fused Metal neighbour kernel (`knn_metal.py`, DESIGN §4), the offline W0 weather prototype (`weather.py`, `analysis/w0.py`, `docs/STORMS_W0.md`), `director/` (the host-side ecosystem director, see below) and `game/` (the game observers of build steps G1 and G2, `docs/DESIGN.md` §28 and §29; G2 also adds the generic `groups.py` and `tracking.py` and `analysis/social.py`) |
| `tests/` | pytest suite. 2,434 fast tests passed, 83 were skipped (the GPU-only kernel tests among them) and 1 xfailed on the CPU device after merging main (#28, #29) into G2 (`-m "not slow"`, `test_lifecycle_compiles_and_matches_eager` deselected; about 11 minutes) plus the slow ones |
| `docs/DESIGN.md` | the spec: state schema, step order, ledger, RNG streams, config, tests, decision and review logs (the director is §26, the LLM tier §26.10) |
| `docs/DIRECTOR.md` | the design of the ecosystem director; steps 1 to 4 are built, §12 to §14 record how steps 2 to 4 refine it |
| `docs/RESEARCH.md` | research directions (emergence, open-endedness, signalling) assessed against prior work, with a recommended order |
| `docs/ACTIVITY.md` | evolutionary-activity analysis of `runs/64k_s0` (method, results, sensitivity); figures in `docs/activity/` |
| `docs/SIGNALS.md` | information carried by the emit and msg signals of `runs/64k_s0` at eight states; full tables in `docs/signals/` |
| `docs/EVALUATION.md` | the director evaluation (`DIRECTOR.md` §9): arms A, B, C, R and RC on two seeds, results and verdict; outputs and figures in `docs/evaluation/` (`python -m alife.analysis.evaluation`) |
| `docs/3D.md` | the 3D world with gravity and a floor (owner decisions, layout, fields, Swift viewer, PR plan); built in PRs #15 to #20, normative spec `docs/DESIGN.md` §27 |
| `docs/RESULTS_3D.md` | the first long 3D run, `runs/64k3d_s0` (500k steps): population and `K`, vertical niches and buoyancy, bodies, evolutionary activity, comparison with 2D; figures and tables in `docs/results_3d/`, scripts in `analysis_cache/64k3d_s0/results3d/` (git-ignored) |
| `docs/RESULTS_3D_RUNS.md` | the four 500k-step `64k3d` runs (seed 0, seed 1, half food, director live): replication, the density control, the director in 3D, bodies and activity across runs; figures and tables in `docs/results_3d_runs/`, scripts in `analysis_cache/64k3d_runs/` (git-ignored) |
| `docs/GAMES.md` | the gaming vertical (design and owner decisions): G1, the observer layer, is built (`game.observe`, `src/alife/game/`, `python -m alife.analysis.labels`, normative spec `docs/DESIGN.md` §28, Review-log row I44) with `INTENT_VERSION` 2 label rules (`RULES_V2`, calibrated offline and validated on three held-out windows on the GPU, 2026-10-21, DESIGN §28.4, §28.8); G2, social tracking (houses, kin groups, colonies, territory, relations; `src/alife/groups.py`, `tracking.py`, `game/social.py`, `game/society.py`, `python -m alife.analysis.social`; the reports are `observers/social.jsonl` in an observing run's directory; normative spec `docs/DESIGN.md` §29, Review-log row I45) is built, run on the GPU (2026-10-21) and has provisional thresholds (`SOCIAL_VERSION` 1, measured, none changed); G3 to G5 are not built |
| `docs/STORMS.md`, `docs/STORMS_W0.md` | the storm world (an alien-planet weather game on the engine; G3 onwards target it) and the results of its offline weather prototype W0 (DESIGN Review-log row I46): every exit criterion but the cost (about 1.16 ms a step against 1.0) passes |
| `docs/bench/` | `alife bench --stages` reports for `64k`, `64k3d` (`C` = 24, 32, 48) and `tiny3d` (`docs/DESIGN.md` §27.13), before (2026-10-16) and with (2026-10-20) the fused Metal neighbour kernel |
| `gui/` | AlifeControl, a SwiftUI macOS app that drives the CLI (see `gui/README.md`) |
| `runs/` | run directories (git-ignored, local to the Mac) |
| `.github/workflows/ci.yml` | Linux CPU CI: ruff, ruff format check, fast suite; nightly slow suite |

## How to run

```bash
uv sync --extra viewer --extra video          # Python 3.12 pinned in .python-version
uv run pytest -m "not slow" -q                # fast suite
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
cd gui && swift run                           # the macOS control app
```

Overnight runs: plug in, lid open, and either use the GUI's timer or
`caffeinate -is uv run alife resume ...`. Stop cleanly with exactly one SIGINT (Ctrl+C or `kill -INT <pid>`
of the Python process). A second SIGINT is a hard exit. Do not use `--quiet` if you want progress lines.

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
- **App bundle:** `gui/scripts/make-app.sh [--install]` builds, signs (hardened runtime; Developer ID, else Apple
  Development, else ad-hoc) and installs `AlifeControl.app`, with a generated icon. It is for this Mac only: no
  Developer ID certificate or notarization yet. The App Store would need the engine embedded and sandboxed,
  which is a separate project and is not done.
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
   - Then G3 (play loop and view), on the storm world after W0 to W2 (`docs/STORMS.md` §8).
5. **Storm world W0** (merged as PR #28; `docs/STORMS.md` §8, `docs/STORMS_W0.md`, DESIGN Review-log row I46). The
   offline weather prototype (`src/alife/weather.py`, `python -m alife.analysis.w0`; no engine change) passes every
   exit criterion except the cost one: 1.163 ms a step for the weather step plus the per-agent wind sampler against the
   1.0 ms budget. Next: the owner's decision on the cost (accept, trim the model or the grid), then W1 (wiring the
   weather into the step); G3 and the later steps on the storm world wait for it.
6. Optional engine work, deferred in DESIGN §23.2:
   - done: the fused Metal kernel for the neighbour search (`src/alife/knn_metal.py`, DESIGN §4 and F21, row I48;
     `ALIFE_KNN=auto|mlx|metal`, the kernel is the default in 3D on Metal and `ALIFE_KNN=metal` selects it in 2D).
     It is exact only on the MLX versions in `knn_metal.VERIFIED_MLX` and a startup self-check guards the rest. **On
     every MLX bump and macOS major update**, run `uv run pytest tests/test_spatial_metal.py -m "" -q` on the Mac (the
     GPU equality tests; they run on an unlisted version), then add the version to `VERIFIED_MLX`, or leave it out;
     the remaining custom kernels of DESIGN §23.2 item 1 (tiled GEMV, splat renderer) are still deferred;
   - a mean-speed metric, so "sessile vs mobile" is measurable;
   - `inspect` showing the resumed total instead of the original `steps`.
7. If a change alters trajectories, re-pin the golden digest deliberately (tests/test_determinism.py) and note
   it in the DESIGN Review log.
