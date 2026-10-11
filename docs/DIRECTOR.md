# Ecosystem director: design

Status: proposal (2026-10-11). Build step 1 (§3, the engine's live environment state) is implemented and is now
normative in `docs/DESIGN.md` (§2.2, §12.7, §12.9, §13.1, §14, §17–§19; Review-log row I28). **Build step 2 (the
director core: vital signs, archive, rule tier, decision log, replay, CLI) is implemented** and is normative in
`docs/DESIGN.md` §26 (Review-log row I29); §12 below records where it refines or contradicts this proposal. **Build
step 3 (the LLM tier) is implemented** and is normative in `docs/DESIGN.md` §26.10 (Review-log row I30); §13 records
where it refines §2.1, §7 and §8. **Build step 4 (the GUI Director panel and `alife llm-check`) is implemented**
(`gui/`, Review-log row I31); §14 records how it refines §8. Step 5 (the evaluation) is not implemented yet. This
document extends `docs/DESIGN.md`.
As each step lands, its normative parts (state fields, config keys, ledger terms, tests) move into DESIGN.md with
Review-log rows, per `CLAUDE.md`.

## 1. Goal and motivation

Run `runs/64k_s0` (1.5M steps, see `docs/HANDOVER.md`) settled into a low-novelty regime after about 1M
steps. One founder lineage reached 84%, signalling fell from 81% to 26%, sharing nearly vanished, and the
mutation scale was selected down. A stable world with no newcomers favours faithful copying and pruning.

The **director** watches the ecosystem's vital signs and acts on the environment (and immigration) to keep the
world **novel but viable**. That is the "novelty + learnability" criterion from open-endedness research.

Constraints:
- The director never edits agents, their genomes or their brains. All change flows through the environment,
  so outcomes stay emergent.
- It has two tiers. The rule tier is deterministic and always on. The LLM tier is optional, runs locally via
  LM Studio, and is slow-cadence.
- It must preserve determinism, checkpoint/resume and the exact energy ledger.
- It runs locally on the owner's Mac, with no paid API.

## 2. Architecture

```text
            every K_rule steps (default 1,000)              every K_llm steps (default 25,000–50,000)
 metrics ─► vital signs ─► RULE TIER ─┐            summary ─► LLM TIER (LM Studio) ─┐
                                      ├─► validate + clamp + budget ─► decision log ─► env state ─► step
                                      │                                  (director.jsonl)
                                      └──────────────────────────────── journal.jsonl (rationales)
```

- **Rule tier** (Python, no LLM). It computes vital signs from the metrics row and applies safety and pacing
  rules. It is deterministic in (state history, config).
- **LLM tier.** A synchronous request to LM Studio's OpenAI-compatible server, made at the `K_rule` boundary that is
  a multiple of `K_llm` (decision L1, §13). The reply is constrained to the intervention JSON schema, then
  validated, clamped and budgeted like rule actions, and applied at that same boundary.
- **Decision log** (`runs/NAME/director.jsonl`). One line per applied intervention: `{step_applied, source:
  rule|llm|revert, persona, action, params, prev, duration, reason, vitals_step, vitals}`, after a header line
  (§12). It is the single source of truth for replay.
- **Journal** (`runs/NAME/journal.jsonl`). LLM rationales and narrative entries, plus rule-tier one-liners.
  This is for legibility only and never feeds back into the simulation.

### 2.1 Determinism and replay

- Ordering in the runner: interventions apply only at `K_rule` step boundaries, between compiled steps, as
  pure updates of `State.env` (§3). The step function stays pure.
- **Replay.** `alife resume RUN --director replay` (and any rerun from a checkpoint) applies `director.jsonl`
  entries at their recorded steps and never calls the LLM. Same seed + same config + same decision log gives a
  bitwise-identical trajectory on the same backend.
- **Live.** `--director live` calls the rule tier and, if enabled, the LLM, and appends to the log.
- **Off.** `--director off` is the default and gives the current behaviour exactly.
- **The LLM call is synchronous (L1).** The first proposal had the call run asynchronously, with the reply applied
  at the next `K_rule` boundary. Built instead: at an LLM boundary the runner waits for the reply inside the
  director hook, so the reply is applied and logged at that boundary. Reason: a landing step that depends on
  wall-clock latency would make the log (and so any replay-by-rerun) depend on machine speed, and the cost of
  waiting is small: a call takes about 2 to 8 seconds against a boundary every 25,000 steps (about 7 minutes at 62
  steps/s), roughly 0.5% of wall time. A failure or a timeout (`director.llm_timeout_s`, default 60) falls back to
  the rule tier for that boundary. No wall-clock time enters the simulation.
- On resume in `live` mode, log entries after the checkpoint step are truncated, like events today.

## 3. Engine change: live environment state

**Status: implemented (build step 1).** DESIGN.md is the spec; this section is the design rationale, and the table
below is the original proposal. Where the implementation differs, the step-1 decisions apply:
- **Flat fields.** The record is nine flat `State` fields named `env_<name>` (`env_regen_mult`, `env_patch_gain`, ...),
  appended after `diag`, so checkpoints stay name-keyed. `State.env` is a read-only `Env` NamedTuple view
  (`env.regen_mult`, ...), and `state.default_env(cfg)` gives the inert values.
- **Digests.** `state_digest` hashes every field, `env_*` included. `legacy_state_digest` hashes exactly the 38 v0
  fields with the old encoding, and the golden pin uses it, so the pin did not change.
- **RNG.** `Stream.CATASTROPHE = 8` takes a reserved index and moves no other key. `RNG_LAYOUT_VERSION` is 2 and
  `RNG_COMPATIBLE_VERSIONS = {1, 2}`, so version-1 checkpoints still load.
- **Excess deaths.** Catastrophe deaths are only agents that would otherwise have survived the step. Every other
  cause keeps precedence, so `INSOLVENT ≤ DEATHS_STARVED` still holds. The corpse takes the ordinary path.
- **Carry-over request.** `env_imm_request` is a third immigration demand, combined with the floor and periodic rules
  by `max`. The step writes back `max(request − n_imm, 0)`, so a large pulse arrives over several steps. It is
  capped by lanes (births first) and capacity, and the remainder stays pending. The runner's extinction stop waits
  for a pending request.
- **One-step catastrophe.** `env_cat_kill_p` resets to 0 after the step it applied in. Centre and radius stay set.
- **Ranges.** `season_amp` is checked in `[0, 1]` (inclusive), matching `validate`. With the director on, an
  out-of-range value, including one that the step consumes, sets `Diag.FIELD_RANGE`.
- **Gating.** Everything is behind the static `cfg.director.enabled` (`DirectorConfig`, default off, in
  `POST_V01_KEYS`; `config_hash` also drops a section the POST_V01 deletions empty). Enabled with inert values is
  bit-identical to off.
- **Not in step 1.** The mutation-floor nudge (`env.rate_min` / `sigma_min`), the rule tier, CLI flags,
  `director.jsonl`, the LLM and the GUI.

Today the levers are `Config` constants closed over by the compiled step. Changing them mid-run needs a fork.
Add a small **environment state** record to `State`, read by the compiled step as arrays. Its fields are
checkpointed, cost nothing to change and need no recompile.

| field | shape, dtype | meaning | default (inert) |
|---|---|---|---|
| `env.regen_mult` | (), f32 | multiplier on per-cell regeneration | 1.0 |
| `env.season_amp` | (), f32 | replaces `fields.season_amplitude` at runtime | config value |
| `env.season_period` | (), i32 | replaces `fields.season_period` (0 = off) | config value |
| `env.drift_mult` | (), f32 | multiplier on patch drift velocity | 1.0 |
| `env.patch_gain` | (PP,), f32 | per-patch fertility gain (add or remove habitat; 0 disables a patch) | 1.0 |
| `env.imm_request` | (), i32 | immigrants to admit at the next lane allocation (consumed, reset to 0) | 0 |
| `env.cat_center` | (2,), f32 | catastrophe centre | 0 |
| `env.cat_radius` | (), f32 | catastrophe radius (0 = none) | 0 |
| `env.cat_kill_p` | (), f32 | per-agent kill probability inside the radius (consumed, reset) | 0 |

Rules:
- **Inert defaults are bit-identical.** With `--director off` and inert defaults, trajectories must equal
  today's, so the golden digest must not change. Read `env.*` only where the config already branches. Where a
  multiplier is exactly 1.0, keep the integer path identical: apply `floor(regen * mult)` only when the director
  is enabled, behind a static config branch `director.enabled`.
- **Energy and ledger.**
  - Regeneration stays `INJECT_REGEN`.
  - Immigration stays `INJECT_IMMIGRATION`.
  - Catastrophe deaths are a new death cause, `CATASTROPHE`. The corpse goes through the existing corpse path
    (`FLOW_CORPSE`, structure back to the field), so no new ledger term is needed. This needs a death-cause
    enum entry, counters and an events column.
  - A catastrophe draws randomness from a **new RNG stream** (`Stream.CATASTROPHE`), which bumps
    `RNG_LAYOUT_VERSION`. The other streams stay unchanged, so existing trajectories are unaffected.
- **Season and patch gain.** `fields.season_weights` uses `env.season_*`. Fertility weights each patch by
  `env.patch_gain[p] * w_p(t)`.
- **Checkpoints.** `env` is part of `State`, so it is saved and resumed. Old checkpoints without `env` load
  with inert defaults (`state_from_arrays` fills missing `env.*`).
- **Invariants.** Every `env.*` must be finite and in range: `regen_mult` in [0, 4], `drift_mult` in [0, 20],
  `patch_gain` in [0, 4], radius ≤ L/2, `kill_p` in [0, 1]. Add these to `device_check` and `full_check`.
- **Tests.**
  - Inert defaults give an unchanged digest.
  - Each knob has the intended effect.
  - The ledger closes with catastrophes.
  - Checkpoint round-trip.
  - Old-checkpoint load.
  - Replay of a decision log is bitwise-identical to the live run that produced it, on CPU.

## 4. Vital signs

Computed by the rule tier every `K_rule` steps from metrics already logged (§19 of DESIGN.md), plus a few
cheap additions.

| group | signals |
|---|---|
| Diversity | `clades_alive`, `clade_entropy`, `top_clade_share`, `weight_diversity` (reliable since the F20 fix), spread of gene quantiles (q90/q10 of sigma, rate, frac) |
| Behaviour descriptor `b` | emitters_frac, sharers_frac, mean_bonds_per_agent, frac_in_bodies_ge4, singleton fraction, kill share, predation_ratio, gene medians. Each is normalised to [0, 1] with fixed ranges. |
| Novelty | `novelty = mean k-NN distance of b to the descriptor archive` (k = 5). A MAP-Elites-style grid over 2–3 chosen descriptor axes records which cells have ever been visited. **Coverage** = visited cells. |
| Stagnation | `‖b(t) − b(t − 100k)‖`, and the number of new archive cells in the last 200k steps |
| Viability | n_alive / K_est, births/deaths balance, a min-population trend, an extinction-risk flag (`n_alive < 0.2 × K_est` or falling more than 30% in 50k) |
| Pacing | steps since the last intervention, a "drama" level (recent intervention intensity), and the persona's target curve |

The archive (descriptor vectors and grid cells) is written to `runs/NAME/director_archive.json` at each checkpoint for
analysis. It is never read back: a live run rebuilds it deterministically from `metrics.jsonl` on start or resume,
and a replay needs no archive.

## 5. Intervention menu

Every action has parameter bounds, a cost against a rolling **budget** (per 100k steps), a **cooldown**, and an
expected-effect note for the journal.

| action | params (bounds) | effect | cost | cooldown |
|---|---|---|---|---|
| `set_seasons` | period 2k–100k, amp 0–0.9 (the mode stays `fields.season_mode`; it is not a run-time lever) | climate regime; with `season_mode = per_patch` it forces migration | 1 | 50k |
| `fertility_shift` | move patch p by ≤ L/4 (a host `patch_pos` edit), or gain 0.25–2.0 | frontier or habitat change | 1 | 20k |
| `habitat_loss` | patch p, gain → 0 for D steps (10k–200k) | fragmentation, refugia | 2 | 50k |
| `regen_pulse` | mult 0.4–1.8 for D steps (5k–50k) | famine or boom | 1 | 30k |
| `immigration_pulse` | count 16–1,024 (≤ lanes over several steps; at most 1,024 pending in total), source random (no archive source yet) | invasion, new founders | 1 | 20k |
| `catastrophe` | centre, radius 20–200, kill_p 0.2–0.9 | founder effects, recolonisation | 3 | 100k |
| `drift_change` | mult 0.5–10 | environmental tempo | 1 | 50k |
| `mutation_floor_nudge` | rate_min ×(0.5–4), sigma_min ×(0.5–4) | exploration pressure (global; use sparingly) | 3 | 200k |

The mutation-floor nudge changes config-level bounds. It needs `env.rate_min` and `env.sigma_min` fields too.
It is optional for v1 and **not implemented in step 2**: it is not in the action schema.

**Archive immigrants** (`source: archive`) need a genome archive: every heavy-metrics step, snapshot a few
sampled genomes from distinct clades into `runs/NAME/genome_archive/`. An immigrant pulse can then reintroduce
an extinct lineage ("invasive species"). This is optional for v1; random founders work first.

### Safety rules (rule tier, always on)

- **Extinction rescue.** If the extinction-risk flag is set: a `regen_pulse` boost plus a random
  `immigration_pulse`. This overrides cooldowns and budget once per 200k steps.
- **Monoculture breaker.** If `top_clade_share > 0.9` for 100k steps: an immigration pulse, or per-patch
  seasons, or habitat loss, chosen by persona weights.
- **Stagnation response.** If novelty is below threshold and there are no new archive cells for 200k steps: the
  persona selects from the menu (the rule tier picks by weights when the LLM is off).
- **Runaway damper.** If the population exceeds 0.95 × capacity (`N_CAP`, the hard row capacity) or a single body
  exceeds 5% of the population: a mild `regen_pulse` < 1.
- **Hard caps.** At most one catastrophe per 100k steps. The total budget per 100k is bounded by the persona.

## 6. Personas

Each persona is a weight vector over objectives plus a system-prompt paragraph for the LLM tier.

| persona | objective | style |
|---|---|---|
| **Gardener** | maximise viability and diversity; minimal intervention | rescues, gentle seasons, rare pulses |
| **Trickster** | maximise persistent novelty within viability | irregular, varied shocks; avoids repeating an action |
| **Architect** | pursue a user goal (e.g. "bodies ≥ 100", "make signalling pay", "three trophic levels") | targeted, multi-step plans |

The objective is `J = w_n·novelty + w_c·coverage_gain + w_d·diversity − w_r·extinction_risk −
w_x·intervention_cost`. The persona sets the `w`.

## 7. LLM tier (LM Studio)

- **Endpoint.** Configurable; the default is `http://localhost:1234/v1/chat/completions` (OpenAI-compatible).
  Use stdlib `urllib`, with no new runtime dependency.
- **Structured output.** Request `response_format: {type: "json_schema", json_schema: <intervention schema>}`:
  ```json
  {"actions": [{"action": "<menu name>", "params": {...}, "expected_effect": "..."}],
   "journal": "2-3 sentences citing the vital signs that motivated the choice"}
  ```
  The schema allows at most two actions per call, including `"none"`. Every reply is validated against the
  schema, clamped to the bounds, checked against budget and cooldown, then logged. An invalid or late reply
  falls back to the rule tier.
- **Prompt.** About 2k tokens (built: 1.7k to 2.5k for the `tiny` preset, `prompt.py`):
  - persona paragraph;
  - menu with an effect description per action (bounds alone make a model choose `immigration_pulse` for
    everything), bounds, costs and cooldowns remaining;
  - current vital signs and their trends over the last 5–10 windows;
  - the last five interventions with their measured effect (vital-sign deltas 50k steps later);
  - budget remaining;
  - the user's goal (Architect).
- **Cadence.** Every `K_llm` steps (default 25,000, about 7 minutes at 62 steps/s), synchronous (§2.1, L1).
- **Model.** Any local instruct model with reliable JSON-schema output. On 48 GB, a 7–14B model in LM Studio's
  MLX format fits beside the simulation (about 1.2 GB). Shortlist two or three by validity rate and decision
  quality (§9). Latency of a few seconds per call is fine. **Chosen (L5): `gemma-4-26b-a4b-it-mlx@8bit`**, the
  default of `run.llm_model`. A local probe on 2026-10-11 (24 requests, two personas, four synthetic scenarios) gave
  24/24 valid, sensible replies at about 2.3 s median. The 4-bit QAT build degenerated, and the qwen3.6 reasoning
  models return an empty `content` unless thinking is disabled in LM Studio, which the client reports as an
  `empty` failure and answers with the rule tier. Run the model with LM Studio's default settings otherwise; the
  26B A4B mixture of experts has 4B active parameters, hence the speed.
- **Containment.** The LLM output is data and only selects menu actions. It never edits files, code or config,
  and it is never `eval`'d. Store the raw prompt and response in `journal.jsonl` for audit.

## 8. Interfaces

**CLI**
- `alife run|resume ... --director off|live|replay`.
- `--persona gardener|trickster|architect`, `--goal "..."` (Architect).
- `--llm` (every 25,000 steps), `--llm-every N`, `--llm-url URL`, `--llm-model NAME`. Built as in §13: there is no
  `--no-llm`, the tier is off unless asked for (`--llm-every 0` switches it off on a resume).

**Config** (new keys, defaults off, added to `config.POST_V01_KEYS`):
- `director.enabled`, `director.rule_every`, `director.llm_every` (step 3);
- `director.persona`, `director.budget_per_100k`, `director.goal`;
- `run.director` = `off|live|replay`.

Only `director.enabled` is hashed: the policy keys (`persona`, `goal`, `rule_every`, `budget_per_100k`, and from step 3
`llm_every`, `llm_temperature`, `llm_timeout_s`, `llm_max_tokens`) are omitted from `config_hash` at every value and may
change on resume. `--persona` / `--goal` and the LLM flags came with step 3 (§13).

Endpoint and model are run-level keys (`run.*`), so they can change on resume.

**GUI** (AlifeControl; built as in §14): a Director panel with:
- mode, persona and goal;
- LLM URL and model, with a "Test connection" button that sends a tiny chat request;
- cadence;
- a live journal view that tails `journal.jsonl`;
- the last ten interventions.

**Dashboard / analysis:** intervention markers on time-series charts, and novelty and coverage curves.

## 9. Evaluation

Fork `runs/64k_s0` at step 120,000 into arms that run to step 1.5M:

| arm | director |
|---|---|
| A | off (control; behaviour already known) |
| B | rule tier only, Gardener |
| C | rule + LLM, Trickster |
| D (optional) | rule + LLM, Architect with goal "bodies ≥ 100" |
| R | random director: same menu, budgets, cooldowns and safety rules as B, with actions drawn uniformly from the eligible menu (seeded, so it replays) |

**Why arm R.** Without it, "the director helps" cannot be separated from "any perturbation helps".
Prior LLM-in-the-loop ALife work (ASAL, ASAL++) reports no such matched control (`docs/RESEARCH.md` §1).
B and C count as an improvement only if they beat R as well as A. The random director needs a new
`random` persona: not built yet, see §10.

**Seeds.** Run every arm on at least two seeds:
- seed 0: the fork of `runs/64k_s0` described above;
- seed 1: the same fork of a second base run made with `--seed 1` and run to step 120,000.

Within a seed, all arms share the base checkpoint, so the director is the only difference.

Report each metric per seed, with the spread, and draw no conclusion from a single trajectory.

**Metrics.**
- Archive coverage over time.
- Bedau–Packard evolutionary activity, normalised against a neutral shadow (`docs/RESEARCH.md` §4).
- Persistent novelty: novelty that is still above the baseline 100k steps after an intervention.
- Clades and lineage depth.
- Behavioural turnover.
- Number of extinctions and rescues.
- Intervention diversity and cost.
- For the LLM: JSON validity rate, clamp rate and latency.

**Success criteria.**
- B beats both A and R on coverage and persistent novelty, on every seed, with no population collapse.
- C beats B on those metrics, or matches B with better journal legibility (a human rates 10 sampled journal
  entries).

**Cost.** About 6 hours per arm and seed at 64k on the M5 Pro, so 4 arms × 2 seeds is about 48 hours, plus D if
run. Run sequentially; parallel arms share the GPU.

## 10. Build order

1. **Engine (done):** `State.env`, the `CATASTROPHE` death cause and RNG stream, patch gain, regen multiplier,
   invariants, tests and a DESIGN.md update. This is the highest-risk step because it touches the compiled
   step; land it alone.
2. **Director core (done, §12):**
   - vital signs, archive and rule tier, with budgets, cooldowns and safety rules;
   - `director.jsonl`, and replay with a bitwise replay test;
   - CLI flags.
3. **LLM tier (done, §13):** LM Studio client, schema, personas, prompt builder, fallback and journal, with a
   mock-server test (`http.server` in tests) for valid, invalid and slow replies.
4. **GUI (done, §14):** the Director panel and `alife llm-check`.
5. **Evaluation:**
   - a `random` persona for arm R: uniform over the eligible menu, with the same budget, cooldown and safety
     rules, seeded per decision like the Trickster, plus a replay test;
   - arms A–D and R on two seeds;
   - dashboard markers.

## 11. Open decisions (owner)

**Decided 2026-10-11:** all four recommended options below were accepted by the owner.

1. **Live levers as simulation state** (§3), as opposed to forks. *Recommended:* yes. It is the only way to
   intervene without recompiles or new run directories.
2. **LLM role.** It picks from the menu (*recommended*), or it free-forms actions (rejected: unsafe, hard to
   validate).
3. **Initial objective.** Research open-endedness (Gardener and Trickster, evaluated by §9) or game-style pacing
   and drama. *Recommended:* the research objective first. Pacing personas reuse the same machinery later.
4. **Genome archive for immigrants.** Defer to v2 (*recommended*) or include in v1.

## 12. Build step 2 as built: where it refines this proposal

Decisions A1–A20 of the step-2 spec; `docs/DESIGN.md` §26 is normative.

- **Host-only (A1).** No change to the compiled step; the director edits `env_*` and `patch_pos` with `_replace`
  between steps, keeping shapes and dtypes, so nothing retraces.
- **Hook (A2).** After the metrics row and the violation check, before the checkpoint, at `t % rule_every == 0`; the
  checkpoint holds the applied env and a resume never re-applies anything. Not on an extinct world.
- **Modes (A4, A9).** `run.director` is `off|live|replay`, a resume key; `--director` on `run` (fork included) and
  `resume`. `off` with `director.enabled` leaves the env as it is. Replay is in place (`resume RUN --director
  replay`) or by fork (`run --init-from SRC[:STEP] --director replay --replay-log LOG`); the log is
  pre-validated (exit 2) and replay never clamps, truncates or writes.
- **Hash scope (A5).** Only `director.enabled` is hashed. The policy keys are excluded from the hash entirely.
- **Director state is folds (A6, A7).** Budget, cooldowns, active effects, drama and the pacing are folds of
  `director.jsonl`; the vital-sign tracker and archive are folds of `metrics.jsonl`, rebuilt on start or resume by the
  same code path. `director_archive.json` is analysis output and never read back.
- **Log (A8).** A header line (rules version, persona, goal, cadence, budget, seed), then entries with explicit,
  clamped, absolute `params`, `prev`, `duration`, `reason`, `vitals_step` and a compact `vitals`. A timed effect's
  end is **its own `revert` entry** at `step_applied + duration` (duration rounded up to the cadence); `prev` holds
  the values to restore. `journal.jsonl` holds the one-liners.
- **Menu (A11).** Seasons change period and amplitude only (the mode is `fields.season_mode`); a patch move is a host
  `patch_pos` edit wrapped like `ops.wrap`; immigration is random and capped at 1,024 pending; the mutation-floor
  nudge is not implemented.
- **Safety rules (A14).** "Capacity" is `N_CAP`. The rescue overrides cooldown and budget at most once per 200k, and
  never the one-catastrophe-per-100k cap.
- **Personas (A15).** Gardener: argmax of its weights; Trickster: seeded sampling that never repeats its last action,
  plus a boredom rule; Architect: the safety rules only (the goal is stored, not interpreted). Persona weights scale
  the windows, cooldowns and budget. The objective J is not optimised (no forward model).
- **Vitals and archive (A17, A18).** The descriptor is ten components in [0, 1] with fixed ranges; novelty is the mean
  RMS distance to the 5 nearest archive members excluding the 3 most recent; archive membership is by a novelty
  threshold; coverage is a 10×10×10 grid over emitters × bodies≥4 × predation. `K_est` is recomputed at each
  boundary from the live environment.
- **Fork (A19).** A fork resets `env_*` to `default_env(new cfg)` and starts with the director off.
- **Viewer (A20).** `view` and `render` re-simulate without the director; replay in the viewer is not implemented.

## 13. Build step 3 as built: where it refines this proposal

Decisions L1–L11 of the step-3 spec; `docs/DESIGN.md` §26.10 is normative.

- **Synchronous (L1).** The call blocks inside the existing hook (after the row and the violation check, before the
  checkpoint), at `t % director.llm_every == 0`. No thread, no pending-request state. Timeout `director.llm_timeout_s`
  (60). A timeout, a connection or HTTP error, unparseable JSON, a schema violation or a reply of which every
  proposal was refused falls back to the rule tier for that boundary (§2.1 has the reason).
- **Roles (L2).** The rule tier keeps the safety moves at every boundary (reverts, extinction rescue, runaway damper,
  the hard caps) and they take precedence. The discretionary moves (stagnation response, monoculture breaker, the
  persona-driven moves, the Trickster's boredom) come only from the model at an LLM boundary; the rule tier makes them
  only as the fallback of a failed call there, and none between LLM boundaries. The order at a boundary is: reverts,
  safety rules, then the model or the fallback; the prompt shows what the first two did. With the tier off the
  behaviour and the log are exactly step 2's.
- **Off by default (L3).** `director.llm_every = 0`; `--llm` sets 25,000; `validate` demands 0 or a multiple of
  `rule_every`. With the tier off, or in replay, nothing touches the network.
- **Config and flags (L4).** Policy keys `director.llm_every|llm_temperature|llm_timeout_s|llm_max_tokens` (unhashed,
  resumable) and run keys `run.llm_url|llm_model` (resumable); flags `--llm`, `--llm-every`, `--llm-url`,
  `--llm-model`, `--persona`, `--goal` on `run`, fork and `resume`. A bare `--llm` needs `--director live` (exit 2
  otherwise).
- **Model (L5).** `gemma-4-26b-a4b-it-mlx@8bit` (§7).
- **Request (L6).** The schema is generated from `actions.MENU`; every number has `minimum`/`maximum` and every string
  `maxLength` (240 for `expected_effect`, 600 for `journal`), which probes showed are needed to stop runaway integers
  and looping strings. `fertility_shift` is offered in its gain form (`patch`, `gain`) only; the patch move stays a
  rule-tier action, so that every menu action is one schema variant. Catastrophes are given as absolute `x`, `y`.
- **Prompt (L7).** Persona; the Architect's `director.goal` as delimited data; the menu with effect descriptions,
  bounds, cost, cooldown and cooldown remaining; active effects; vital signs and their trends over the last 8 rule
  windows; the safety actions and reverts of this boundary; the last five interventions with their measured effect;
  the budget left. No text the model wrote is ever put back into a prompt.
- **Validation (L8).** Structure and types are the schema's (a violation is an `invalid` reply and a fallback);
  numeric ranges are clamped by the step-2 clamp and each change is journalled; what the clamp refuses (a patch that
  does not exist) and what the rule tier's eligibility refuses (budget, cooldown, one effect per lever, the
  catastrophe cap) is a rejected proposal with its reason. Repeated actions are dropped; `none` beside another action
  is ignored; `none` alone is a decision. If nothing survives and the reply was not `none`, the rule tier decides.
- **Logging (L9).** `source: "llm"` entries with the persona, clamped params, `reason` = `expected_effect` and an
  `llm_call` id; one `llm_call` record per call in `journal.jsonl` with the raw prompt and response.
- **Replay (L10).** `llm` entries replay like rule entries; replay never calls the model; `RULES_VERSION` is unchanged.
- **Containment (L11).** The client POSTs to the configured http(s) URL only: no proxies, no redirects, a size cap and a
  total deadline.
- **Not built.** `--no-llm` (use `--llm-every 0`); the move form of `fertility_shift` for the model; a streaming or
  concurrent call; any use of the model for text that feeds the simulation or a prompt.

## 14. Build step 4 as built: where it refines this proposal

Decisions G1 and G2 of the step-4 spec. The panel is `gui/Sources/AlifeControl/DirectorPanel.swift`; its pure logic
(settings, baselines, validation, argument construction) is `DirectorSettings.swift`, and the file tailers are
`DirectorTailer.swift`. `gui/README.md` describes the use.

- **`alife llm-check` (G1).** A new CLI command (`DESIGN.md` §21, Review-log row I31): one tiny request under the
  director's own client and schema, one output line, `OK model=... latency=...s` (exit 0) or `FAIL <kind>: <detail>`
  (exit 1). The "Test connection" button runs it with the panel's URL and model and shows the line in green or red.
  No Python is duplicated in the GUI: the check goes through the same code a run uses.
- **Modes (G2).** A new run and a fork offer Off and Live; a resume offers Off, Live and Replay (in place, the run's own
  `director.jsonl`). A replay fork with `--replay-log` stays CLI-only. A resume of a run whose `config.json` has
  `director.enabled = false` offers nothing and passes nothing ("fork it to add one"), since its compiled step cannot
  take director actions; Replay is also refused in the panel when the run has no `director.jsonl`.
- **Baselines.** As with the levers, only values that differ from a baseline are passed: the engine defaults for a new
  run; for a fork and a resume the source's `config.json` (`director.*`, `run.llm_url`, `run.llm_model`, `run.director`).
  The engine starts a fork with the director off whatever the source ran, so a fork's baseline mode is Off: the picker is
  pre-filled with Live when the source ran live, which passes `--director live`.
- **Mapping.** `--director live|replay` (only when it differs from the baseline mode), then, in Live mode only,
  `--persona`, `--goal=TEXT` (Architect only), `--llm-every N` (0 switches the tier off; the toggle's default is 25000),
  `--llm-url=URL` and `--llm-model=NAME` (only with the tier on), and `--set director.rule_every=N`. Free text is passed as
  `--flag=value` so that a leading "-" cannot be taken for an option.
- **Validation hints** mirror `config.validate` and disable Start: `rule_every` is a positive integer and a multiple of
  `run.metrics_every` and `run.heavy_metrics_every` (read from the preset file for a new run, from `config.json` for a
  fork and a resume; if they cannot be read a note says the engine will check, exit 2); `llm_every` is a positive
  multiple of `rule_every` while the tier is on; the URL is http(s) and the model non-empty while the tier is on.
- **Live views.** The Interventions view (the last 10 entries of `director.jsonl`, newest first) and the Journal view (the
  last 20 records of `journal.jsonl`: rule one-liners, and LLM calls with step, latency, outcome, the model's journal
  text, accepted and rejected counts, the fallback, and the raw prompt and reply in a disclosure group) tail the files
  every 2 s, reading only the last 256 KiB; a missing file, a torn last line and malformed lines are skipped, and unknown
  keys are ignored. A run without `director.jsonl` shows "Director off for this run".
- **Not built.** Starting a replay fork from the GUI; editing `llm_temperature`, `llm_timeout_s`, `llm_max_tokens` or
  `budget_per_100k` (CLI `--set` only); intervention markers on the metrics (step 5).
