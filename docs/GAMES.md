# Emergent Worlds: a gaming vertical for alife (design proposal)

Status: **design; G1 and G2 built (DESIGN §28, §29), N1 built (§31), G3a built (§33; the Mac day ran 2026-10-22 and the gate passes, `docs/RESULTS_G3A.md`; the click-through by hand is open), G3b built, its Mac day ran 2026-10-23 and the gate passes except the report-time item the bench cannot read (§34, `docs/RESULTS_G3B.md`; branch `games/g3b`; the click-through and the first play by a person are open); G3c–G5 not built.** The owner decided items 1–4, 7 and 8 on 2026-10-18
and items 13–19 on 2026-10-19 (§10). This document maps the owner's brief ("Project: Emergent Worlds", 2026-10-18) onto the
engine at `aef79c2`. The brief's three pillars are:
- a motivation-to-action loop instead of scripted state machines;
- an LLM architect director;
- social lineage.

`docs/DESIGN.md` stays normative. Every feature starts off by default, so existing runs, their `config_hash` and the 2D
golden digest stay as they are. The one new live lever chosen for G4, `attack_mult` (§10.4), adds a `State` field; the
3D golden digest does not move, because W1 froze the 3D field list in `legacy_state_digest_3d` (§5.4).

## 1. Summary

- **Much of the brief already has a basis in the code.**
  - **Behaviour.** The agents already work the brief's needs-based way. A per-agent recurrent net maps sensed energy,
    hurt, crowding, kin, chemicals and messages to actions. There are no hand-written behaviour rules: only the output
    decode and the reproduction gate use fixed thresholds.
  - **Director.** It has an `architect` persona that takes a natural-language `director.goal`. It answers through a
    strict JSON schema, chooses only from a bounded menu, and logs every decision for bit-exact replay (DESIGN §26).
  - **Lineage.** Clades, heritable tags, the bond graph and the birth and death logs cover family, kinship and colonies.
- **The main gaps:**
  - Per agent, only the last step's harvest, share inflow and loss are stored. Other actions, attack gains and all
    longer-run counters are missing.
  - There is no player command channel and no live lever for violence, cooperation or harshness.
  - Director pacing is tuned for research runs: at game speed it would allow one or two commands an hour.
  - There is no faction or territory tracking, the viewers carry no per-agent identity, and nothing runs at a fixed
    step rate.
- **The runs warn about the content.**
  - Sharing is rare, bodies are uncoordinated "adhesion sheets", and founder clades collapse to about 4.
  - In the director evaluation, random moves at the LLM's steps matched or beat it on coverage and novelty. The LLM's
    moves gave 1.3–3.9× the new adaptive activity, and its journal was rated legible (4.3 of 5).
  - A game on today's dynamics shows mostly foragers, predators and clumps (§8.4).
- **Proposed design:**
  - **Observation.** A gated, observer-only per-agent channel (§4). It is not a per-frame decode of the step, which would
    cost about three quarters of a step.
  - **Director target.** The director acts on live environment state (`State.env`), not on `config.json`. Config keys are
    hashed, can change only by fork, and force a recompile; config changes become explicit "era changes" (§5.2).
  - **Player intent.** It reaches the world in stages: the LLM parses text into an intent vector, a deterministic,
    versioned compiler turns that into actions, and the player confirms a preview generated from the compiled plan
    (§5.3, §7).
  - **Pacing.** Player commands get their own pacing in game time. Once applied, the safety rules cannot supersede them
    (§5.5, §5.6).
- **Scale.** At 64k capacity (`n_max` 65,536: about 33k agents in 2D, about 42k in 3D) the step runs at 59 steps/s in 2D
  and 14 in 3D. The owner chose a 64k-capacity 2D world at 30 steps/s for the first slice (§8.3, §10.7).
- **Platform.** The engine runs on Apple Silicon at speed (MLX on Metal) and slowly on Linux CPU (`mlx[cpu]`, the CI
  platform). A Mac-first prototype is realistic. A Windows or console product needs the step ported.

## 2. Starting point

| asset | where | evidence |
|---|---|---|
| Needs-driven agents | 69 inputs (84 in 3D) feed an Elman RNN (H = 32; 4,387 weights, or 4,999 in 3D), which drives 35 outputs (39 in 3D): `layout.py`, `brain.py`, `actions.py` | DESIGN §5–§7 |
| Exact energy economy | 16-term integer ledger | DESIGN §10.8; `ledger_error` is 0 in all four 3D runs |
| Bounded, logged interventions | `director/actions.py` `MENU` (7 actions with bounds, costs, cooldowns); `director.jsonl` | DESIGN §26.4 |
| Natural-language goal | persona `architect`; `director.goal`, in `<goal>` tags in the system prompt (`director/prompt.py`) | DIRECTOR.md §6, §13 |
| Bit-exact replay on one backend | `run.director=replay` | DESIGN §14.2, §26.8 |
| Visible kinship | heritable 4-dimensional `tag` drifting at a fixed rate; render mode `tag` | DESIGN §12.4; `render.py` `COLOR_MODES` |
| Live 3D view | `alife stream` (ALSTRM) into the Swift Metal viewer | DESIGN §27.11; `gui/README.md` |

**Prior art** (sources in §11):
- **Already shipped:**
  - creatures evolved with neural nets (Creatures, 1996; The Bibites; Ecosystem);
  - needs-based utility AI (The Sims, Dwarf Fortress);
  - event directors (RimWorld's storytellers, the Left 4 Dead AI Director).
- **Found in no shipped game:** an LLM that turns player intent into bounded, replayable parameter changes on an evolving
  neural ecology. That is the defensible novelty.
- **What prior art shows is hard:**
  - designers cannot read or control evolved nets (Risi & Togelius 2015);
  - players want agency more than they want to watch selection (Spore dropped mutation-driven play);
  - LLM text must never become game state unchecked (Where Winds Meet, 2025).

## 3. Part A: feature mapping

Cost combines engineering complexity with runtime cost. "est." marks estimates. Step labels follow DESIGN §3: S7 decode,
S8 costs, S10 harvest, S11 combat, S12 share.

### Pillar I: the motivation-action loop

| gameplay concept | `alife` code component | implementation strategy | complexity / performance cost |
|---|---|---|---|
| Hunger / survival | `State.energy`; `IN_ENERGY` (`sensing.py`); dues and insolvency (`energy.pay_costs`, S8); `repro.min_energy` | Two bars: *runway* (steps of upkeep left) and *satiety* (`E / min_energy`). "Starving" when the runway is under 50 steps. | Low. Reads `State` only. |
| Threat / safety | `State.last_loss` (`IN_HURT`); `OUT_DEFEND`; `combat.*` | Hurt bar from `last_loss`. The Fleeing and Defending intents use the observer channel. | Low (hurt); Medium (intents). |
| Social need | `State.partner` (B = 4 slots); `IN_KIN_MEAN`; `IN_CROWD` | Social and crowding bars. Family ties are bonds where `parent_uid` matches the partner's `uid`. | Low. |
| Mating drive | `OUT_REPRODUCE`; `repro.min_energy` 40,000; `repro.min_age` 20; `StepOut.births` | "Ready to bud" when E ≥ min_energy and age ≥ min_age. | Low. |
| Calling / signals | `IN_CHEM`, `IN_CHEM_GRAD`, `IN_MSG_MEAN`; `OUT_EMIT`, `OUT_MSG`; chemical fields (`fields.py`); `analysis/signals.py` | A chemical-field overlay. A signal gets a label only for what it is measured to predict in that world: in 2D, msg0 and emit1 carry 0.2–0.38 bits about crowding and the next-step stop (SIGNALS.md). 3D is untested, so G1 runs the analysis in 3D. | Low (overlay); Medium (analysis). |
| Momentary intent | the decoded `Actions` (S7) inside `step.step` | Observer channel `StepOutG.acts` (§4.2), then threshold rules with step-clock hysteresis (§4.4). | Medium. A static branch, off by default; est. under 0.5 ms per step at 64k. |
| Archetype: Gatherer / Hunter | `State.last_harvest` (S10, `energy.harvest`); attack gain `Combat.gain` (S11, `energy.combat`) | Labels by income source, from fixed-point accumulators (§4.3, §4.4). | Medium. Harvest is in `State`; attack gain needs the channel. |
| Archetype: Provider / Trader | `State.last_inflow`; per-slot `out` and `in` of S12 `energy.share` | Provider gives a large share of its income. Trader is reciprocal flow on one bond, and is expected to be rare: there is no exchange mechanism. | Medium. Two (N, B) arrays in the channel. |
| Archetype: Nomad / Settler | `State.vel`; grip (`physics.py`); bonds | Directional persistence `‖EMA(v)‖ / EMA(‖v‖)` and speed, both relative to the thrust speed `u = thrust_max / drag` (0.1). | Low. |
| Temperament (kin attitude) | the kin component of `OUT_Q_ATTACK` and `OUT_Q_BOND` (`actions.py` φ·q) | Mean kin weight of each query, either spares kin or targets kin. Validated against `kin_attack_bias` per kin group. | Medium. Uses the channel. |
| Archetype mix | metrics row | Archetype shares go into the metrics row of game worlds. They are kept out of `DESCRIPTOR_NAMES`, so existing personas do not change. | Low. |

### Pillar II: the Architect Director

| gameplay concept | `alife` code component | implementation strategy | complexity / performance cost |
|---|---|---|---|
| Natural-language command | persona `architect` and `director.goal` (the precedent); `director/llm.py` strict `json_schema`; `llm.clean_text` | A new intent schema (§7.1). The LLM parses the player's text into an intent vector and never chooses parameters. | Medium. |
| Intent → parameters | `director/actions.py` `clamp`, `validate_entry`, `apply`; `rules.eligibility` | A deterministic compiler (`COMPILER_VERSION`). Its proposals are eligibility-checked and clamped at landing (§5.3). | Medium. Pure Python, unit-tested. |
| New live levers | `GAME_MENU` and `GAME_SPECS` (used only in game worlds); new `env_*` fields read under the static `cfg.director.enabled` (`step.py`); `invariants.ENV_*` | `env_attack_mult`, `env_share_q8`, `env_basal_mult` and `env_mut_mult`, plus a `regen_set` action (§5.4). `MENU`, the yoked sampler and the existing prompts stay byte-identical. | High. Dynamics, ledger tests, State3 loading (the 3D golden is frozen by W1, §5.4; `death_causes(cfg)` keeps the observer's cause rows apart from `STORM`, DESIGN §30.7). |
| Era change | fork (`run --init-from RUN:STEP --set …`); `config_hash` | A config-only change is planned as a fork at a chapter boundary. The fork recompiles (4.5–5.9 s) and must re-apply the env (§8.2). | Low (exists); Medium (UX). |
| Influence | `director/pacing.py` (`BUDGET_WINDOW`, cooldowns, `drama`); `MENU` `Spec` rows; `rules.PersonaParams` | Player entries get their own pacing scope in game time, from a `GAME_SPECS` table (§5.5). | Medium. Clamp, validation, fold and cooldowns take the spec table as a parameter. |
| Tension curve | counters `DEATHS_KILLED`, `ATTACKS`, `BIRTHS`; `Vitals.fall` | A new `intensity` vital (an EMA of the kill and death rate, as in Left 4 Dead), steered by `game_master` within the system budget. | Medium. |
| Player-triggered calls | `Director._decide`; stdin control (`stream.py`) | An `intent` control line and `commands.jsonl`. The call is parsed off the step thread, and its proposals land at a rule boundary (§5.7). | Medium. |
| Safety against player effects | `rules.py` rescue and runaway damper; lever exclusivity (`actions.lever_keys`) | Precedence and a risk preview (§5.6). | Medium. |

### Pillar III: procedural social complexity

| gameplay concept | `alife` code component | implementation strategy | complexity / performance cost |
|---|---|---|---|
| Family | `uid`, `parent_uid`, `generation`; `StepOut.births`; `events/births_*.npz`; `analysis/phylo.py`; `alife inspect --agent` | A host lineage store fed by births. An ancestry panel and a "kin nearby" list. | Low–Medium. G3: it needs picking and the HUD (DESIGN §29.1). |
| House (genealogical name) | `clade` (the founder dynasty, which never splits); `phylo.splitmix64`, `phylo.marker_roots` | Live sub-lineages: houses found themselves by `splitmix64(uid, seed)`, are inherited, and use no RNG (§6.2). | Medium. Built in G2 (DESIGN §29.4), with a parent-row channel in place of `parent_slot`; founded and extinct houses are listed (§29.14). |
| Kin group (perceived faction) | `State.tag` (N, 4); kin `exp(−‖Δtag‖²/0.25)` (`sensing.py`) | Clusters in tag space, tracked over time. Agents perceive this grouping, so relations are measured between kin groups (§6.3). | Medium. Built in G2 (DESIGN §29.5), with names and per-group counts (the roster, §29.14). |
| Kinship colour | `render.agent_colors` (mode `tag`), which `stream` uses | Use as is: similar colour means recent common ancestry. | Low (exists). |
| Colony | `metrics.components` (min-label propagation) | A one-to-one tracker on uids, giving born, merged, split and dissolved events. Presented as reefs or mats, not organisms (§6.4). | Medium. Built in G2 (DESIGN §29.6); `alife bench --observe` and `analysis.social` measure it. |
| Territory | `ops.scatter_add_i32`; coarse grid over `world.size` | Counts per cell for the top-F groups plus "other". The owner is the argmax and the purity is shown with EMA smoothing (§6.5). | Low–Medium. Built in G2 (DESIGN §29.7). |
| War, alliance, exchange | attack and bond targets (S7–S11); per-slot shares (S12) | F × F matrices between kin groups: attacks, kills, bonds formed, energy shared (§6.5). | Medium. Built in G2 (DESIGN §29.8); the targets are in the channel. |
| Depth niches (3D) | `clade_height_mi`, `band_frac` | The depth band of each group. In `64k3d_s0`, 4 clades held two zones, with clade–band MI of 0.63 nats. | Low. Built in G2 (DESIGN §29.5): each kin group's mean height and height bands. |

### Infrastructure

| gameplay concept | `alife` code component | implementation strategy | complexity / performance cost |
|---|---|---|---|
| Live view with picking and HUD | `stream.py` `FRM1` (alive, pos, orient, rgb, bonds, field slice); Swift `StreamReader`, `Renderer` | Add record tags within the major version (DESIGN §27.11): an alive-only frame record, `IDNT` (uid, house, kin group, colony; every k frames, alive rows only), `INTN` (archetype and intent bytes), `TERR` (overlay), `EVNT` (births, kills, director entries). Add picking and a HUD to the 3D viewer. | Medium. A 64k frame is 21 B per row (1.38 MB for all 65,536 rows) plus 8 B per bond, about 1.5 MB. IDNT at 16 B per alive row adds about 0.7 MB at 42k agents (est.). |
| Director in the live view | `stream --follow` (faithful only at checkpoint cadence); simulate mode ignores the director | `alife play`: one process with the runner, director, observers, command inbox and stream (§3.1). | Medium. |
| Real-time pacing | `runner.py` (no fixed rate) | A fixed-timestep scheduler with a target rate and fast-forward, plus interpolation in the viewer. | Low. |
| Save / load | checkpoints (`runio.py`, safetensors) | Saves are checkpoints plus logs plus the observer sidecar (§6.6). They shrink if dead rows are canonicalised before saving (§8.2). | Medium. 686 MB per 64k3d checkpoint. At 16k the weights alone are about 144 MB in 2D and 164 MB in 3D. |
| Shared replays | ALSTRM | Recorded, decimated stream files, independent of the backend (§8.2). | Medium. |

### 3.1 Process architecture

```
 Swift client ──stdin: intent, confirm, pick, speed, colour──▶ alife play (one process)
              ◀── ALSTRM header + FRMA / IDNT / INTN / TERR / STRM / MAP1 / ENVS / EVNT / CLCK / FOCS ───
 alife play:  scheduler ─ step (compiled; StepOutG when game.observe) ─ observers (accumulators, houses, kin groups,
              colonies, territory) ─ director hook at rule boundaries (reverts, safety, player proposals, game_master)
              ─ checkpoint + observer sidecar
              off the step thread: LLM intent parse → compiler → preview to the client
              logs: director.jsonl (decisions), commands.jsonl (text, exchange, intent, plan, confirmation, outcome)
```

**As built in G3a** (DESIGN §33): `alife play` is that process without the intent, confirm and power lines (G3b, G4); the scheduler
is `runner.run(pacer=)` with a `PlayPacer` (the hook is inert when None, and a pacer never changes a trajectory: digest equality is
tested), the records are the ten above (`ALRT`, `RPRT` and `PLAN` are G3b), and the control lines are `pause run step rate color every
focus follow idnt labels quit`. The Swift side is one **World window** (a raw-Metal map and organism view with a docked control
center) beside the Lab, fed by `PlayController` over a pipe.

Principles carried over from the director:
- **Environment only.** No intervention edits an agent, genome or brain.
- **Observers never write `State`.** Their values reach the dynamics only through logged director entries.
- **The log is the source of truth** for replay.
- **Hashing follows the trajectory.** `game.*` keys default to off and go into `POST_V01_KEYS`.
  - Observer-only keys (`game.observe` in G1, `game.house_rate` in G2) form a new `OBSERVER_KEYS` tuple (they may default
    to a working value: `house_rate` is 1/2048, §10 item 13). It is excluded
    from `config_hash` next to `UNHASHED_KEYS`, but kept out of `DIRECTOR_POLICY_KEYS`, which is derived from
    `UNHASHED_KEYS`. A new `game` branch in `cli._resume_allowed` accepts it, so observers can be attached to an
    existing world. The label thresholds are not config keys: they are versioned code constants of
    `rules.RuleSet`, identified by `INTENT_VERSION` and a pinned `RuleSet.digest()`, so a label means the same in
    every world (G1's choice, DESIGN §28.4, pending the owner's confirmation: §10 item 11).
  - This is justified by G1's purity test: the trajectory is identical with observers on and off.
  - Only keys that change dynamics are hashed.

## 4. Pillar I in detail: the behavioural mapping layer

### 4.1 What the layer may claim

The layer may show what the agent **senses** (its inputs, shown as needs), what it **does** (its decoded actions, shown
as intent), and what it **lives on** (its income sources over a window, shown as archetype). It may not claim motives.
The brief's `If (Energy < Threshold) → Search(Resource_Field)` describes the kind of policy that evolves, since energy is
input 1. But the brain has no such threshold to display, so the UI presents intents as observations, not rules.

**The same limit binds the words about the world (N1, DESIGN §31).** A chapter report may state what the files record: counts at
a step, a window between two steps, a name taken from a roster, a label only where its validity study passed robustly. It may
not credit an intervention with an outcome, imply an order between two facts, or give a motive. The sentences are templates
rendered by code from a fact sheet and re-checked by a second line (a closed vocabulary, a lexicon of causal, presupposing and
evaluative words, and per-kind gates); a local model, if one is used, only selects and orders code-written candidates, and
the shown order across kinds of sentence is code's. The intervention block is the engine's lever state, not a list of recent entries.

### 4.2 Data path

- **From `State`, free.** Needs, harvest and received shares come from `State` (`last_harvest`, `last_inflow`,
  `last_loss`, `energy`, `vel`, `partner`), so they work with observers off.
- **Why not decode per frame.** Re-running the decode (`analysis/signals.decode`, S0–S8, 2D-only, no RNG advance) costs
  about 14.4 ms (the staged S0–S7) plus part of S8, out of a 19.2 ms staged step at 64k in 2D (`docs/bench`). That is
  about three quarters of a step. It stays the offline validation tool, generalised to 3D in G1.
- **The observer channel.** With `game.observe` on (a static branch, so the compiled graph is unchanged when off),
  `step.step` returns a parallel type `StepOutG`: the StepOut fields plus `acts`.
  - `runner.run` keeps each step's StepOut (about 80 KB at 64k) until the metrics boundary.
  - `acts` is about 4.5 MB per step at 64k, so the play loop folds it into the accumulators every step and keeps only
    the StepOut part.

| `acts` field | shape, dtype | source |
|---|---|---|
| `attack_gain` | (N,) i32 | S11 `energy.combat`, `Combat.gain` |
| `attack_target`, `bond_target` | (N,) i32 | S7 targets (−1 when none), with flags for hit and formed |
| `share_out`, `share_in` | (N, B) i32 | S12 per-slot `out`, and the net received `mul_q8(inc, q8(share_efficiency))`, with the S12 `partner` (N, B) |
| `flags` | (N,) u8 | attacked, hit, bond formed, budded, emitted |
| `thrust`, `defend` | (N,) i16 | S7 decoded values, quantised as `round(256·v)` |
| `q_kin` | (N, 2) i16 | the kin columns of the raw `OUT_Q_ATTACK` and `OUT_Q_BOND` outputs, clipped to ±127 before `round(256·v)` |

`BirthRecords` gains a `parent_slot` field (the lane's parent row, −1 for immigrants) under the same flag. `runio`
ignores extra event columns.

**As built in G1** (DESIGN §28.2): `attack_gain`, `share_out`, the net received as `share_got`, `flags`, `thrust`,
`defend` and `emit` (the sum of the S17 deposit quanta, in [0, 765], not a float). The targets, `q_kin`, the S12
partner and `BirthRecords.parent_slot` serve only the relations, the per-kin-group temperament and the colonies, so
they move to G2.

**As built in G2** (DESIGN §29.2): `attack_target`, `bond_target`, `q_kin`, the neighbour rows `nbr` (validity from the
own-index contract), the birth lanes `born_row`, `born_parent` and `born_uid`, and the flag bits `PROPOSED` and `KILLED`.
The parent row travels in `Acts` (`born_parent`), not in `BirthRecords.parent_slot`: a new `BirthRecords` field would
change the event files of every run. The S12 partner is still not in the channel. The roster (§6.2, §6.3) adds one more
field, `death_cause` (the S14 cause of every row that died, which the capped death records cannot give). With the storm
weather (W1, DESIGN §30.7) a fifth cause, `storm`, exists: `social.death_causes(cfg)` adds its row to the window counts only
when `weather.enabled`, and a weather-off run's sidecar and social log keep their G2 bytes.

### 4.3 Accumulators

Each signal uses a fixed-point accumulator: `A += x − (A >> k)`, with `A ≥ 0`, mean `A >> k`. It has no dead zone, its
bias is under one quantum, and it never shifts a negative number.
- **Flags** enter as 256 per set bit, and floats enter quantised as above.
- **Windows.** k = 6 for archetypes (about 3·64 steps to reach 95%), and k = 3 for intents.
- **Headroom.** The largest per-step input is about 6,800 quanta (harvest 400, attack gain up to about 2,800, shares in
  up to about 3,600), so A stays under 4.5e8 even at k = 16.
- **Exactness.** Integer accumulators make the label arithmetic exact. On one backend a replay reproduces the labels
  because the trajectory is bitwise. Across backends the inputs drift with the trajectory, and so may the labels.
- **Persistence.** Accumulators are restored from the observer sidecar (§6.6). Without it they warm up again after a
  resume.

Cost (est.): about 15 elementwise int ops per step on N rows, under 0.5 ms at 64k. Nothing goes to the host until a UI
frame needs it.

### 4.4 Rules

**Archetypes.** Windowed income is `I = harvest + attack_gain + share_in`. Each agent gets a primary label and, where a
second rule also matches, a secondary one.

| archetype | rule (windowed means) | prevalence |
|---|---|---|
| Gatherer | harvest ≥ 0.75·`harvest_max` (a harvest level, no longer a share of the income) | 14% (2D) and 46% (3D) of labelled agents at the calibration windows |
| Hunter | attack_gain / I ≥ 0.35 | 0.9% / 4.0% |
| Provider | Σ share_out / I ≥ 0.2 | 0.7% / 0.2% |
| Trader | Provider, with some slot where `min(out, in) / max(out, in) ≥ 0.5` | 0.1% / 0.01% |
| Nomad | persistence ≥ 0.6 and speed ≥ 0.4·u | 0.2% / 1.3% |
| Settler / Colonial | speed < 0.1·u, or in a bonded body of 4 or more (at the last social snapshot) | 49% / 34% (the rule's prevalence) |
| Unspecialised (the Generalist) | none of the above: no income or movement rule matched | 47% / 46% (primary): mostly agents that harvest below 3/4 of the maximum and match no other rule |

The prevalences are those of the calibration windows (DESIGN §28.8). The Gatherer rule of version 1 (harvest ≥ 0.7 of the
income) matched 98% (2D) and 93% (3D) of agents, because nearly all income is harvest; it failed the validity test.
Version 2's Gatherer is the rate of the best harvesters, so most agents are now Unspecialised: moderate harvesters (a
median of 0.57 / 0.53 of `harvest_max`), not agents with mixed income. The enum and the report key stay `generalist`;
`rules.ARCHETYPE_DISPLAY` is the table of the words a player sees (the coordinator's choice C9 of 2026-10-19, to revisit).

**Version 3: the storm world (W2a, DESIGN §32.7; `RULES_V3`, `INTENT_VERSION` 3, the default for every world).** With the weather on, `vel`
includes the wind drift, so Nomad and Settler read the **velocity relative to the wind** (`vel` minus the velocity the wind alone would give
a free agent, `weather.agent_drift`); the numeric thresholds are V2's. With the weather off the drift is 0 and the V2 labels are unchanged.
Three archetypes are added after Settler in the priority order (the order keeps the older labels first; in a weather world Nomad and Settler still move, because they read the velocity relative to the wind: on the pilot window Nomad's primary share fell from 2.5% to 1.0% and Settler's rose from 5.7% to 6.1%; Unspecialised shrinks by the rows the new labels take, 8 points),
for the aerial biosphere of STORMS §5:

| archetype | rule (windowed means; `u` = `thrust_max/drag` = 0.1) | prevalence (pilot window) |
|---|---|---|
| Drifter | wind drift ≥ 0.015 per step and at least half the agent's own relative speed | 8.8% |
| Floater | lift ≥ 0.30 of `buoyancy_max`, height ≥ 8 and \|vertical velocity relative to the wind\| ≤ 0.0074 (half the free fall) | 6.8% |
| Sinker | lift < 0.10 of `buoyancy_max` and descending relative to the wind at ≥ 0.0074 per step | 10.1% |

Floater and Sinker are 3D rules written for the `64k3d` physics (gravity 0.015, `buoyancy_max` 0.04, H = 80) and exist only in 3D worlds. The words a
player sees in a storm world (`rules.ARCHETYPE_STORM_DISPLAY`): Gatherer is shown as **Grazer**, Hunter, Drifter, Floater and Sinker as they are;
a weather-off world keeps Gatherer. The Floater rule uses the buoyancy *output* the agent chose (the new `Acts.buoy` channel field), not its
capacity. The prevalences above are **match** rates (the validated bit); the label a player sees is the primary one, and Gatherer comes first in the
priority, so the displayed rates are lower (pilot: Drifter 3.1%, Floater 0.8%, Sinker 5.1% of the rows; about a third of the Drifter matches, an eighth of the
Floater matches and half of the Sinker matches show as the new label).

**Intents.** Intents are evaluated by the observer every s = 8 steps. Event predicates such as "harvested > 0", "emitted"
or "formed a bond" hold for nearly every agent (harvest effort is a sigmoid, 87% of 3D agents emit, about 3,700 bonds
form per step). With 0.06 attacks and 0.19 bond formations per agent-step, "any attack in 8 steps" would cover about
37% of agents and "any new bond" about 78%. Intents therefore use magnitude thresholds on the k = 3 means:
- the order puts the rarer, more specific states first, with one deliberate exception: Feeding precedes Signalling,
  because in 3D one clade both harvests at the cap and emits, and the label placed first takes those agents. The first
  match wins;
- a label changes only after the new rule has held at 4 consecutive evaluations, on the step clock, so labels do not
  depend on playback speed. Budding, a forecast of an event, is the exception: it is shown at each evaluation where it
  holds, over the intent that the hold keeps underneath (the coordinator's choice C10 of 2026-10-19, to revisit).

The order:
1. Budding: about to bud. Energy is just below the budding threshold and, at the agent's recent net energy flow, reaches
   it within about 10 steps. It is a forecast of energy and income, not of the decision to bud (in 3D 37% of budders wait
   above the threshold with their reproduce output off, which the observer cannot see).
2. Hunting: attack-flag mean ≥ 0.25, about two attempts per 8 steps.
3. Fleeing: hurt, with |thrust| ≥ 0.5.
4. Caring: sharing at least `harvest_max`/10 per step with a parent or child partner, by `parent_uid`.
5. Defending: guarding after a recent hit: hurt within about the last 31 steps, with the guard at 0.75 or more. The guard
   does not react to hits, so the UI must not say that the agent raised it because of the attack.
6. Tethering (the Bonding intent): forming and dropping bonds again and again, in 3D mostly with the same neighbours:
   a formation-flag mean of at least 0.25 over about the last 8 steps (a bond formed every 3 steps is always on, every 4
   steps it flickers across the threshold). In 3D that is bond flicker (94% of new bonds re-form a pair seen earlier in
   the window, RESULTS_3D §5.4; the partners' identity is not measured in 2D), not a new lasting partner; version 1's
   meaning failed the validity test in both worlds. Show the bond count beside it. The enum and the report key stay `bonding`; `rules.INTENT_DISPLAY` is the table
   of the words a player sees (the coordinator's choice C8 of 2026-10-19, to revisit).
7. Feeding: harvest ≥ 0.75·`harvest_max`.
8. Signalling: Σ emit ≥ 0.75.
9. Roaming: |thrust| ≥ 0.3.
10. Resting.

In 3D a Gatherer is usually Feeding (the two use the same level on different windows). For about a fifth of agents the
displayed (held) intent is more than 56 steps old.

**Thresholds are absolute and versioned** (`INTENT_VERSION`). They are not per-world quantiles, which would label 10% of
agents Hunters in a peaceful world. Version 2 was calibrated on dumps of the G1 study windows of `64k_s0` and `64k3d_s0`
and validated on held-out windows (`64k_s0` from 1,500,000, `64k3d_s1`, `64k3d_s2`); DESIGN §28.4 pins it.

### 4.5 Validity test (G1, before any UI work)

Run on `resim` replays of the reference runs:
- **Archetypes.** Does "Hunter" predict attacks over the next 1,000 steps, "Nomad" the area covered, and "Provider" the
  outflow? Each is scored as an AUC against a matched random label.
- **Intents.**
  - Each intent must predict its own action over the next 10 steps.
  - No intent may cover more than about 60% of alive agents.
- **Stability.** Label flips per agent per 1,000 steps (about 33 s at 30 steps/s).

`analysis/mi.py` supplies the MI estimators and permutation nulls; the AUC and the matched random label are new code.
Labels that fail are dropped from the UI.

**Outcome.** G1's GPU study (version 1 rules) failed Gatherer, Defending (the 0.6 coverage cap), Budding and Feeding in
both worlds and Bonding, Signalling, Roaming and Caring (too few positives) in 3D. Version 2 passes every archetype and
every intent on the calibration windows, offline, under the rule, the first-match display and the replayed hold, in both
worlds (thinnest displayed margins: 3D Roaming +0.082, 3D Resting +0.101, 2D Hunting +0.102). The validation on the three
held-out windows was run on the GPU on 2026-10-21 (DESIGN §28.8): every archetype and every intent's displayed label passes the
provisional criterion of §10 item 12 on all three (`64k_s0` from 1,500,000, `64k3d_s1` and `64k3d_s2` from 450,000; the thinnest
margins are 3D Resting +0.112, 3D Roaming +0.124 and 2D Settler +0.218), nothing was re-tuned, and no label is marked as not
validated. The 2D window is the calibration run 4,000 steps later, so it is held out in time, not in population. One caveat: the
3D Signalling predicate covers 0.674 of rows on `64k3d_s1`, over the 0.6 cap, though the displayed label covers 0.171 and passes.
Stability: the displayed label changes 7.05 and 7.07 (2D) and 9.2 to 9.6 (3D) times per 1,000 steps on the GPU windows (the replay
said about 5.3 and 6.5; the Budding overlay adds about 1.7 and 2.3), against G1's 1.98 and 4.12. The hold (4 × 8 = 32 steps) equals the
agents' 32-step clock, which makes thresholds near a clock-driven mean flicker; a hold or period that does not resonate
would help every intent, and is flagged for G3. No stability criterion is set.

**Version 3 (storm world, W2a; DESIGN §32.8).** Calibrated on one pilot window (`64k3d_s0` forked at 450,000 with the storm overlay and the prior
coupling set, 470,000 to 476,000; the Metal re-simulation matches the stored metrics): all ten intents and all nine archetypes pass. The new
labels score Drifter AUC 0.880 against a matched random null of 0.533 (outcome: the wind share of the displacement), Floater 0.591 against 0.514
(altitude held, scored within the agents aloft that are not pinned at the lid, with z-band strata; the first pass, 0.727 against 0.546, was driven by
agents pinned at the lid, where the surface clamp makes the outcome trivial; **by band the Floater is evidenced only at z 56 to 78.9**, with no signal at
z 4 to 32 and an inverted score at 32 to 56) and Sinker 0.637 against 0.508 (descent over the available room); Nomad and
Settler are scored on the displacement with the wind drift subtracted (`rg2_rel`; Nomad matches 1.7% of rows against 4.9% on the raw
velocity). The physics-derived starting thresholds passed a keep rule that is the same bar as the held-out criterion, so none was tuned and keeping them validates
nothing; the held-out windows do. Floater's first outcome (the share of steps with z ≥ 8) was saturated and replaced by the altitude held, chosen among four
on the pilot dump (so it is fitted to that window). **Not yet validated:** the held-out storm windows, a still-air window
(where Drifter must have no positives) and the weather-off windows are W2b's; until verdict rows exist the report module narrates no V3 label in a
storm world. Display of a V3 label in the UI follows the verdict.

## 5. Pillar II in detail: the Architect Director

### 5.1 What exists

The Architect persona, `director.goal`, the seven-action `MENU`, the strict reply schema, eligibility, clamping, `apply`
to `env_*` with no retrace, and `director.jsonl` (DESIGN §26.4–§26.10). The LLM runs on a timer (`llm_every`; 25,000
with `--llm`) and synchronously. Director calls took a median 3.6–4.4 s (p90 5.5–7.2 s) and up to 18 s cold
(EVALUATION.md §3.6), with a 60 s timeout.

### 5.2 Why the target is `State.env`, not `config.json`

The brief's example keys, `fields.season_amplitude` and `population.immigration_count`, are real (`config.py`), but
neither can change in a live world:
- Config keys outside `run` are hashed. Post-v0.1 keys are hashed once they differ from their default, except the
  director's policy keys (`UNHASHED_KEYS`).
- Resume accepts only `RESUME_KEYS` and those policy keys. Anything else is a fork, which recompiles, opens a new run
  directory, resets `env_*` and starts the director off.

Live equivalents exist for some keys: `set_seasons` writes `env_season_amp` (and the period), and `immigration_pulse`
writes `env_imm_request`. Live intent therefore targets `State.env`. A key with no live equivalent becomes an era change
(§7.4).

### 5.3 The pipeline: parse, compile, preview, confirm, land

1. **Parse (LLM, off the step thread).** The player's text is cleaned (`llm.clean_text`), capped at 280 characters, and
   has `<` and `>` escaped before it goes between delimiters. The model returns the intent vector of §7.1:
   - seven signed dimensions in [−1, 1] (scarcity, harshness, volatility, competition, cooperation, migration, tempo)
     and an unsigned disaster in [0, 1];
   - a horizon, and a target given as an index into lists shown in the prompt;
   - a confidence, and a `clarify` question when the command is ambiguous.

   Probes for the current schema showed that local models need `maxLength` on every string and bounds on every number
   (DESIGN §26.10). With both, the menu schema gave 100% valid replies in the evaluation. The intent schema keeps both.
2. **Compile (deterministic).** `game/compile.py` is a pure function of the intent, vitals, env and `COMPILER_VERSION`.
   It produces raw proposals (§7.3). Any random choice, such as which patch, uses `rng_for(seed, t, tag)` as `rules.py`
   does, and is logged.
3. **Preview.** The proposals are dry-run through eligibility and clamp against the current env. The preview also
   predicts the risk flag: the expected `n_over_k` and fall (§5.6). The player sees text generated from the plan itself,
   covering every action, permanent ones marked. The LLM's narration makes no promise about outcomes. One measured
   reason: halving the food lowered the kill share in 3D (0.48 to 0.44), the opposite of "hunting will pay more".
4. **Confirm**, then **land.** The proposals are queued. At the next rule boundary `Director._decide` runs in this order:
   - due reverts;
   - the confirmed player proposals: eligibility and clamp against the env actually being edited, so `prev` and the
     revert values are current, then apply and log with `source:"player"`;
   - the safety rules, which may not supersede a lever a player effect holds (§5.6);
   - the discretionary rules.

   The client is told about anything clamped or refused.

The LLM does not choose actions directly, unlike today's Architect, for three reasons:
- the intent-to-parameter mapping becomes testable and stable across models;
- the player can veto the interpretation;
- the evaluation's case for the LLM rests on its language rather than its choices.

The current Architect path stays as the baseline in G5.

**Calibration.** The compiler's gains need lever-response curves: the effect of each lever over 10k–50k steps on
population, kill share and archetype mix. These are short sweeps on the 64k 2D preset. A related long-run datum: a 3D run
with `regen_max` halved from step 0 (one seed, `runs/64k3d_s0_half`) settled at 0.63× the population and 2 clades against
4. Its median buoy_cap over the last 50k steps was 0.16 against 0.19. That is an evolutionary end state, not a pulse
response.

### 5.4 New live levers (`GAME_MENU`)

The new rows live in a separate `GAME_MENU`, used only in game worlds. `MENU`, the yoked sampler (arm R),
`llm.param_specs` and the pinned prompt digests of the existing personas therefore stay byte-identical. Each new field is
inert by default, read only under the static `cfg.director.enabled`, and range-checked in `invariants.ENV_*`. Each
energy lever scales an existing ledger term, so no energy path is added; `env_mut_mult` touches no energy term. Each
lever comes with ledger-closure tests and a DESIGN row.

The owner chose `attack_mult` for G4 (§10.4). The other rows are designed but deferred.

| lever | effect | implementation | bounds (proposed) | cost |
|---|---|---|---|---|
| `env_attack_mult` (action `attack_mult`; **G4**) | predation lethality | scales the `combat.attack_max` demand (S11), integer-exact | 0.5–2.0 | 1 |
| `env_share_q8` (action `share_eff`) | how well cooperation pays | absolute Q8 efficiency used as `mul_q8(inc, env_share_q8)` (S12); inert value `ops.q8(bonds.share_efficiency)` reproduces today's result bit for bit | 128–256 (0.5–1.0) | 1 |
| `env_basal_mult` (action `harshness`) | upkeep | scales the basal due (S8). `metrics.carrying_capacity` scales `c_ref` with it, so `K_est` and the safety rules stay calibrated | 2/3–1.5 | 1 |
| `env_mut_mult` (action `tempo`) | evolutionary tempo | a Q8 multiplier on the `mut_sigma` and `mut_rate` used for a child's weights at birth (S16); the inherited genes are unchanged. Replaces the unbuilt `mutation_floor_nudge` (DIRECTOR.md §5): a floor below the existing clip at `rate_min` would do nothing | 0.5–2.0 | 2 |
| `regen_set` (action only, no new field) | permanent food level | writes `env_regen_mult`, takes the `regen` lever, refused while a pulse holds it | 0.4–1.8 | 2 |

**Compatibility.**
- `legacy_state_digest` covers only the 38 frozen v0 fields, so the 2D golden does not move.
- `state_digest` hashes every field name, so any new env field moves it. **W1 (storms, DESIGN §30.3) already froze the 3D
  field list**: `state.V0_FIELDS_3D` is the 49 names `State3` had before W1's `wx`, `legacy_state_digest_3d` hashes them,
  and `GOLDEN_3D` is compared over it and is not re-pinned. So the G4 PR appends its four `env_*` fields to `State3`
  **without a re-pin**: `V0_FIELDS_3D` stays as it is, and G4 only extends the fill set and its tests. (If G4 merges
  before W1, it freezes the list itself and W1 reuses it.) `regen_set` alone adds no field.
- `state_from_arrays` fills missing env fields with inert values only for 2D `State`; since W1 a missing `wx` is filled
  for both dims (the fill set of `State3` is `{"wx"}`). G4 extends the `State3` set with its env fields, with a test that
  loads a pre-game 3D checkpoint (`runs/64k3d_s0`). Otherwise every existing 3D run fails to load.

### 5.5 Game pacing in real time

*As built in G3b (DESIGN §34.5): the table below is `GAME_SPECS` "game-v0", an override of `MENU` (not a subset) with the player scope, `Spec.quantum` and `Spec.excluded`; Strike's cooldown is 3,000 (D10) and the player's budget is 6 units per 18,000 steps.*

Research pacing does not fit a game. At 30 steps/s the 100k-step budget window (`pacing.BUDGET_WINDOW`) is 56 minutes,
and the Architect gets 4 units in it. A 50k-step regen pulse plus its 30k cooldown locks the lever for 44 minutes, and
catastrophes are floored at one per 100k steps.

**Proposed pacing.**
- **Separate scopes.** Player entries are folded in their own pacing scope, apart from the director's own moves. They
  have their own budget ("influence"), and lever exclusivity stays shared.
- **Game cadence.** The settings are:
  - `director.rule_every` = 250 (8.3 s);
  - `run.metrics_every` = 250 (or 50 or 125);
  - `run.heavy_metrics_every` = 250;
  - `run.checkpoint_every` a multiple of 250.

  These are needed because validation requires `heavy_metrics_every` to be a multiple of `metrics_every`, and
  `rule_every` a multiple of both. G2 benches components and weight diversity at that cadence.
- **A game spec table.** Durations, cooldowns and costs are fields of `MENU`'s `Spec` rows, and `MENU` must stay
  byte-identical, so game-time values cannot live in `RuleParams`.
  - A `GAME_SPECS` table (cost, cooldown and duration bounds per action, existing actions included) is selected for
    persona `game_master` and for `source:"player"`.
  - Its id is recorded in the `director.jsonl` header.
  - It is passed to `clamp`, `validate_entry`, `pacing.fold` / `budget_used` and `cooldown_steps`, so replay validates
    3,000-step entries.
  - The safety rules' own entries in a game world use it too, so the rescue pulse fits its bounds.
  - Per-persona constants go in `game_master`'s `PersonaParams`, not in a global rules version, which would also apply
    to research runs.

  The proposed values are multiples of 250 steps:

| action or item | duration (steps) | cooldown (steps) | at 30 steps/s |
|---|---|---|---|
| influence window / budget | 18,000 / 6 units | — | 10 min |
| `regen_pulse`, `attack_mult`, `harshness` | 3,000–36,000 | 3,000 | 1.7–20 min; cooldown 1.7 min |
| `share_eff`, `tempo` (slow) | 18,000–54,000, or permanent | 6,000 | 10–30 min (about 13–38 generations) |
| `habitat_loss` | 3,000–36,000 | 6,000 | 1.7–20 min; 3.3 min |
| `set_seasons` (permanent; period 2,000–18,000), `drift_change` (permanent) | — | 6,000 | a season of 1.1–10 min |
| `regen_set` (permanent) | — | 6,000 | 3.3 min |
| `immigration_pulse` | instant | 3,000 | 1.7 min |
| `catastrophe` | one step | 9,000, which replaces the `catastrophe_gap` floor for `game_master` only | 5 min |
| rescue `regen_pulse` (safety) | 6,000 | as the rule tier | 3.3 min |

**Two timescales.**
- Ecological responses show within one or two lifetimes. The 3D mean lifetime is about 954 steps, about 32 s.
- Evolutionary responses are slow. There are about 13 generations per 10 minutes at 30 steps/s (3D). An agent's tag
  similarity to its ancestor halves after about 108 generations (about 85 minutes). Two groups that split now reach
  similarity 0.5 after about 54 generations each (about 42 minutes).
- Goals should therefore be ecological ("make the reef recover"). The cooperation and tempo dimensions are labelled
  "slow (evolutionary)" in the UI, and a time-lapse mode exists for the rest.

### 5.6 Player effects against the safety rules

The safety rules can cancel a player's plan. A strong scarcity plan can set the risk flag (a fall above 0.3 within 50k
steps). The first `extinction_rescue` in any 200k window then waives budget, cooldown and the regen lever. It replaces
the player's pulse with ×1.5 for 20k plus immigrants equal to 10% of `K_est`, and its revert restores the pre-player
base.

Rules for game mode:
- Confirmed player proposals land before the safety rules at their boundary (§5.3), with the player-chosen risk floor as
  the guard.
- While a player effect holds a lever, rescues and the runaway damper may not supersede it. They act on other levers or
  wait.
- Safety moves draw from a system budget that is not shown as influence.
- The preview shows when a plan would set the risk flag. `game_master` takes its risk floor from a world setting chosen
  by the player (for example "allow a fall to 40% of K").

### 5.7 Interactivity and latency

- **Commands.** The client sends `intent <json>` and `confirm <id>` lines (the stdin protocol caps lines at 4,096 bytes).
  `alife play` appends every step of the pipeline to `commands.jsonl`.
- **Parsing does not block the step.** The world keeps stepping while the model parses (median about 4 s). Without that,
  the world freezes for the call (median 4 s, p90 about 7 s, 18 s cold).
  - Proposals land at a step that depends on wall-clock time either way, since confirmation is a human action. Replay
    is unaffected because it reads `step_applied` from the log and never calls the model or the compiler.
  - The real choice is responsiveness against a pause while the player reads the preview (§10.5).
- **Shared GPU.** LM Studio and the MLX step share one GPU and 48 GB of unified memory. The step rate during an
  inference is unmeasured, so G1 benches it. If the rate drops, the scheduler slows or pauses during calls. The game uses
  whatever model LM Studio has loaded, checked with `llm-check`, and never loads or unloads models.
- **Direct powers.** The UI keeps a few explicit actions, such as "catastrophe here" and "bloom this patch", and sliders
  over the intent dimensions. They give precision and still work when the LLM fails.

## 6. Pillar III in detail: social lineage

### 6.1 Units and evidence

| unit | source | what the runs show |
|---|---|---|
| Family | `parent_uid`; bonds to a parent or child | in the largest 3D body at step 500k, 25% of bonds are parent–child and 48% of members have their parent in the body (RESULTS_3D §5.2) |
| House | sub-lineages of a clade | clades alone are too coarse: 16,384 founder clades fall to 4 by step 177k (3D) and 1.13M (2D) |
| Kin group | tag similarity, which agents sense | 98% of sensed neighbours are same-clade in 2D (SIGNALS.md) |
| Colony | bond-graph components | 3D: the largest body has a median of 2,510 agents and is a single-clade sheet with velocity coherence 0.04–0.10. 2D: the median largest body is 73 |
| Territory | spatial dominance | 3D, step 500k: in 41% of floor-zone columns (52% of top-zone columns), one of the zone's two clades holds at least 90% of that pair's agents (columns with 20 or more). That may be kin clustering after birth rather than defence (RESULTS_3D §4.3) |

### 6.2 Houses: genealogical names

A house is a named sub-lineage within a clade.
- **Founding rule.**
  - Founders' house is their clade.
  - A birth inherits `house[parent_slot]`, unless `splitmix64(uid, seed) < r·2^64`. In that case the child founds a
    house with id = uid. This is the function and cut of `phylo.marker_roots`.
  - For a world played from step 0, offline analysis reproduces the live houses.
  - After a fork (an era change), `marker_roots` hashes tree indices rather than uids and roots every agent alive at
    the fork. Houses then come from the observer sidecar, or `marker_roots` must hash uids and take the sidecar's houses
    as roots.
  - Immigrants found their own house.
- **Properties.** The rule is deterministic, uses no RNG, and replays exactly.
- **Rate.** At 64k there are 29 (2D) to 45 (3D) births per step, so r = 1/2,000 founds a house every 44–69 steps. At 16k
  expect about 4× fewer births. r (`game.house_rate`) is set from the target number of named houses.
- **Storage.**
  - The label is `house (N,) i32` on the device, outside `State`.
  - It is updated between steps, in step order, from `BirthRecords.parent_slot`, using an `ops.py` scatter.
  - Only the top 16 houses by size are named. Houses get a distinct palette, not the founder's tag, which sibling houses
    share.
- **Not perceived by agents.** Houses are labels for the story (dynasties, family trees). They are not what agents
  respond to.
- **As built in G2** (DESIGN §29.4). The label is an int64 (the founder's uid, which `i32` would not hold), `−1` for a
  dead row and `HOUSE_UNKNOWN = −2`. It is inherited from the parent row in the channel (`born_parent`), not from
  `BirthRecords.parent_slot`. For a run observed from step 0 the live houses equal `nearest_root(marker_roots)` of the
  phylo tree (tested). An observer attached mid-run seeds houses from clades, labels immigrant lineages `HOUSE_UNKNOWN`
  and reports `exact: false`; there is no rebuild tool (§10 item 17). The rate is `game.house_rate`, 1/2048 by default
  (§10 item 13). The top 16 houses are named.
- **House events in G2** (the roster, DESIGN §29.14). Each snapshot record lists the houses **founded** (id, step, and the
  parent house: the house of the founder's parent, known from the birth lanes) and **extinct** (id, step) since the
  last snapshot, from the set of houses with live members at consecutive snapshots, up to `events_max` of each with
  the rest counted. The house id is its founder's uid, so no separate founder field. A house founded and extinct
  inside one window is seen by neither snapshot. The study (`analysis.social`) reports the share of members in houses
  founded since its start against those seeded from clades, for the owner's decision on the rate (§10 item 13,
  confirmed 2026-10-21: keep 1/2048) from `64k3d_s1` and `64k_s0`.

### 6.3 Kin groups: factions the agents perceive

Agents discriminate kin only through tag similarity: `NBR_KIN`, `IN_KIN_MEAN`, and the kin column of the bond and attack
queries. Factions that behave as factions must therefore be groups in tag space.
- **Clustering.**
  - Each snapshot quantises the 4-D tag on a fixed grid. The cell width is a parameter; 0.25 gives 8⁴ bins.
  - Occupied neighbouring bins are merged into groups.
  - Groups are matched across snapshots by member overlap, with the same tracker as colonies (§6.4).
- **Drift.** The tag is a neutral clock (similarity 0.5 at about 108 generations apart), so groups drift slowly and
  split as lineages diverge.
- **Relations.** War, alliance and exchange (§6.5) are reported only between kin groups whose mean tag similarity is
  below a threshold. They are validated against a cross-group `kin_attack_bias` corrected for proximity, as the
  existing metric is.
- **As built in G2** (DESIGN §29.5, provisional). The grid is 16 bins per tag dimension (width 1/8, 16⁴ = 65,536 bins),
  not 0.25: width 1/4 chains two 3-D clades into one group and leaves 4–5 groups in 2D, where 1/8 gives 15 and 8 tracked
  groups on the two reference checkpoints (§10 item 14). Occupied bins join by face adjacency (single linkage), and
  entering and leaving thresholds that scale with the population give each group hysteresis. Chaining within a clade is
  reported, not fixed: `sim_self`, the mean tag similarity within a group, is in every record. The validation against a
  cross-group `kin_attack_bias` stays G4/G5's.
- **Names and the roster in G2** (DESIGN §29.14). A chronicle needs characters, so a kin group is **named once, when it is
  admitted**, and keeps the name for as long as it is tracked: the name of its dominant house (the named top-16 house
  with the most of its members; `named_after` says which), else a generated name from the house name generator on its
  id; a house's name goes to one live group at a time (the next group of the same house gets a generated name), and a
  generated name that is held gets the id appended. The names are saved in the sidecar. Each listed kin group also
  carries, in its table row: its **births and deaths by cause** over the window (counted on the device for the slot it
  held, from the channel's per-row death cause, so a mass die-off beyond the 2,048 death records of a step is still
  counted whole; null for a group admitted at this snapshot), its **centroid** (a circular mean on the torus) and its
  **archetype mix** (shares of the primary archetypes among its labelled members). Per-group counts are not valid for a
  group that ends in the window: it has no row.

### 6.4 Colonies: persistent identity for bodies

`metrics.components` labels components by row index, with no identity across snapshots. The tracker:
- **Overlap on uids.** Overlaps are computed on uids, with newborns counted under their parent's colony.
- **Matching.** Old and new components are matched one-to-one, greedily by descending |A∩B|. A match is accepted when the
  Jaccard index is at least θ.
- **Events.**
  - An unmatched new component is `born`, or `split` if its largest overlap is with an already matched colony.
  - An unmatched old component is `merged` into the component that holds most of it, or `dissolved`.
- **Calibration.** θ is calibrated at the actual snapshot spacing. In 3D the Jaccard index with the starting body is 0.59
  at 100 steps and 0.41 at 300 (RESULTS_3D §5.4). Snapshots should therefore be at most about 300 steps apart. The
  tracker uses the 250-step game heavy cadence (§5.5), with footprint overlap (occupied coarse cells) as the cheaper
  fallback if components cost too much there.
- **Naming.** Colonies are presented as reefs or mats. "Super-organism" would promise coordination that does not exist.
- **As built in G2** (DESIGN §29.3, §29.6, provisional). One generic `tracking.SystemTracker` serves colonies and kin
  groups. θ is 0.3, a body gets an id at 4 members and keeps it while it has 3 or more, and an id that ends is never
  revived (§10 item 15). Matching is by uid up to a 300-step snapshot spacing (newborns credited to their parent's
  colony) and by footprint beyond it. The game cadence is 250 steps, a field of `SocialRules` rather than a config key.

### 6.5 Territory and relations

- **Territory.**
  - Each snapshot does one `ops.scatter_add_i32` of alive agents into a dense (cells × (F + 1)) array: a coarse grid such
    as 32 × 32 (the floor projection in 3D), times the top F = 16 kin groups plus "other".
  - The owner is the argmax and the purity its share, both EMA-smoothed.
  - Borders are drawn between cells with different owners.
  - The overlay is labelled "dominance", not "control", until tests show defence.
- **Relations.** F × F matrices per window, from the channel's targets and per-slot shares:
  - attacks and kills (a kill is a victim dying `KILLED` at the step it was a target);
  - bonds formed;
  - energy shared.

  Relation events are narrated in the journal ("Kin group 3 and kin group 7 are at war on the north reef") by the LLM at
  narrative cadence, as an observer, never in the loop.
- **As built in G2** (DESIGN §29.7, §29.8, provisional). Territory is a 32 × 32 floor grid (torus in both axes), the
  EMA is a shift of 2, and the owner of a cell is the argmax of the smoothed counts with a purity and borders. It is
  called dominance. Relations are counted per window for ordered pairs of kin-group slots: attacks, hits, kills,
  formed proposals, energy shared and bond-steps, each with the expectation if every actor had picked a uniformly random
  sensed neighbour (the per-step neighbour-exposure baseline, §10 item 18), the sampled tag similarity of the pair and
  whether it is distinct (below 0.25). Temperament is the mean kin preference of the attack and bond queries per
  slot, at the snapshot step. Narration of relation events stays G4/G5's.

### 6.6 Observer persistence

Accumulators, houses, kin-group and colony trackers live outside `State`, so they are written to a sidecar with each
checkpoint (`observers/step_<t>.npz`). The sidecar is restored on resume, rewind, load and replay, and cut by
`truncate_after`. Without a sidecar, houses can only be rebuilt by folding the birth log from step 0 (43.6M records in
`64k_s0`). Until the sidecar exists, observer values stay out of metrics rows and director vitals.

**As built in G2** (DESIGN §29.10, §29.11). The sidecar has format 2 (the social state and the host trackers). The
reports are not in the sidecar or in the rows: each snapshot's record is one line of the append-only
`observers/social.jsonl`, which `truncate_after` cuts on resume (§10 item 19), and a metrics row keeps a head of at most
0.5 KB. A fork starts its observers fresh and its social log absent.

### 6.7 Visual encoding

- **Colour modes.** Existing: `tag` (kinship gradient), `clade` (dynasty hash), `energy`. New: `house`, `kin_group`,
  `colony`, `archetype`, `intent`. All are computed on the server in `render.agent_colors`, so the 2D render and the 3D
  stream agree.
- **Glyphs.** Attacks are red sparks, keeping today's hurt-is-red rule. Shares are gold pulses along bonds, harvest a
  green inflow from the field, and emission a plume in the channel's tint.
- **Flows come from ledger quantities**, so the picture cannot show energy that does not exist.

## 7. Part B: the intent-to-parameter schema

### 7.1 Intent schema (LLM output, strict `json_schema`)

The wrapper is the one `director/llm.py` uses: `{"type": "json_schema", "json_schema": {"name": "intent", "strict":
true, "schema": S}}`. `target.index` refers to lists shown in the prompt: the top-16 kin groups or houses, the patches,
and a 32 × 32 region grid. The compiler resolves the index to an id or coordinates and logs the resolved value.

```json
{
  "type": "object", "additionalProperties": false,
  "required": ["dimensions", "horizon", "target", "confidence", "clarify"],
  "properties": {
    "dimensions": {
      "type": "object", "additionalProperties": false,
      "required": ["scarcity", "harshness", "volatility", "competition", "cooperation", "migration", "tempo", "disaster"],
      "properties": {
        "scarcity":    {"type": "number", "minimum": -1, "maximum": 1},
        "harshness":   {"type": "number", "minimum": -1, "maximum": 1},
        "volatility":  {"type": "number", "minimum": -1, "maximum": 1},
        "competition": {"type": "number", "minimum": -1, "maximum": 1},
        "cooperation": {"type": "number", "minimum": -1, "maximum": 1},
        "migration":   {"type": "number", "minimum": -1, "maximum": 1},
        "tempo":       {"type": "number", "minimum": -1, "maximum": 1},
        "disaster":    {"type": "number", "minimum": 0,  "maximum": 1}
      }
    },
    "horizon": {"type": "string", "enum": ["short", "medium", "long"]},
    "target": {
      "type": "object", "additionalProperties": false, "required": ["kind", "index"],
      "properties": {
        "kind":  {"type": "string", "enum": ["world", "patch", "region", "kin_group", "house"]},
        "index": {"type": "integer", "minimum": -1, "maximum": 1023}
      }
    },
    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    "clarify":    {"type": "string", "maxLength": 200}
  }
}
```

### 7.2 The brief's example: "Make the world more competitive and scarce"

The system prompt gives the eight dimension definitions, the rule "0 means no change", the target lists, and the current
vitals (`n_over_k`, kill share, archetype mix). The model replies:

```json
{
  "dimensions": {"scarcity": 0.7, "harshness": 0.0, "volatility": 0.0, "competition": 0.6,
                 "cooperation": 0.0, "migration": 0.0, "tempo": 0.0, "disaster": 0.0},
  "horizon": "medium",
  "target": {"kind": "world", "index": -1},
  "confidence": 0.85,
  "clarify": ""
}
```

The brief's own reading was `fields.season_amplitude=0.8, population.immigration_count=0`. Version 1 maps it this way:
- **Seasons are volatility, not scarcity.** They modulate fertility around its mean (`fields.season_weights`:
  `w = 1 + A·sin(…)`), so they bring alternating dearth and plenty around nearly the same mean. The clip at fertility 1
  trims a few percent at A = 0.8. "Scarce" is therefore read as
  slower regrowth. A player who asks for harsh seasons gets `set_seasons` with amp 0.8, live, since 0.8 is inside the
  menu's 0–0.9. No fork is needed.
- **Immigration is already off.** `population.immigration_count` is 0 in every preset, so "stop immigration" is a
  no-op, and the preview says so.

### 7.3 Compiler, version 1

The gains are placeholders until the lever-response curves exist (§5.3).
- **Dead zone.** |x| < 0.05 compiles to no action, and so does disaster below 0.1.
- **Quantisation.** Values are quantised to 0.01 before clamping. The log stores the float32 value, so 0.58 is logged as
  0.5799999833106995.
- **Durations by horizon** (short / medium / long):
  - `regen_pulse`, `attack_mult`, `harshness` and `habitat_loss`: 3,000 / 18,000 / 36,000;
  - the slow levers `share_eff` and `tempo`: 18,000 / 36,000 / permanent.
- **v1 scope.** Version 1 builds only `attack_mult` among the new levers (§10.4). Long scarcity therefore uses a
  36,000-step `regen_pulse` until `regen_set` exists. Harshness, cooperation and tempo compile to no-ops, which the
  preview states.
- **Always permanent.** `set_seasons` and `drift_change` are always permanent, and the
  preview marks them. Their season period by horizon is 2,000 / 6,000 / 18,000.

| dimension | lever | formula | bounds |
|---|---|---|---|
| scarcity s | `regen_pulse` (all horizons in v1; `regen_set` for long once built) | mult = 1 − 0.6·s | 0.4–1.8 |
| harshness h | `harshness` (not in v1: no-op) | mult = 1.5^h | 2/3–1.5 |
| volatility v | `set_seasons`; also `drift_change` when v > 0.5 | v > 0: amp = 0.9·v. v < 0: amp = current·(1 + v). Drift mult = 1 + 8·(v − 0.5) | amp 0–0.9; drift 1–5 |
| competition c | `attack_mult`; at c ≥ 0.5 also `habitat_loss` on the poorest patch (or the targeted one), which concentrates food | mult = 2^c | 0.5–2.0 |
| cooperation k | `share_eff` (slow; not in v1: no-op) | q8 = round(256·(0.9 + 0.1·k)) | 128–256 |
| migration m | `immigration_pulse` | m ≥ 1/64: count = max(16, round(1,024·m)). Otherwise a no-op, stated in the preview | 16–1,024 |
| tempo τ | `tempo` (slow; not in v1: no-op) | mult = 2^τ | 0.5–2.0 |
| disaster d | `catastrophe` at the target (a region centre, or the densest cell of a kin group) | kill_p = 0.2 + 0.7·d; radius = min + d·(max − min) | menu bounds |

The compiled plan for §7.2, as `director.jsonl` entries in the existing format. There is a new `source:"player"`, two new
optional keys (`command`, `compiler_version`), and `vitals` is abbreviated and illustrative here:

```json
{"step_applied": 412000, "source": "player", "persona": "game_master", "action": "regen_pulse",
 "params": {"mult": 0.5799999833106995}, "prev": {"mult": 1.0}, "duration": 18000,
 "reason": "intent scarcity +0.70 (medium)", "command": 17, "compiler_version": 1,
 "vitals_step": 412000, "vitals": {"n_alive": 7900, "k_est": 10400.0, "n_over_k": 0.76}}
{"step_applied": 412000, "source": "player", "persona": "game_master", "action": "attack_mult",
 "params": {"mult": 1.5199999809265137}, "prev": {"mult": 1.0}, "duration": 18000,
 "reason": "intent competition +0.60 (medium)", "command": 17, "compiler_version": 1,
 "vitals_step": 412000, "vitals": {"n_alive": 7900, "k_est": 10400.0, "n_over_k": 0.76}}
{"step_applied": 412000, "source": "player", "persona": "game_master", "action": "habitat_loss",
 "params": {"patch": 7, "gain": 0.0}, "prev": {"patch": 7, "gain": 1.0}, "duration": 18000,
 "reason": "intent competition +0.60: concentrate food", "command": 17, "compiler_version": 1,
 "vitals_step": 412000, "vitals": {"n_alive": 7900, "k_est": 10400.0, "n_over_k": 0.76}}
```

The preview text is generated from these entries. The costs are `GAME_SPECS` costs: `regen_pulse` 1, `attack_mult` 1,
`habitat_loss` 2.

> Food regrowth ×0.58, attacks 1.52× as damaging, and patch 7 barren, all for 18,000 steps (10 min). Cost 4 influence.
> Risk: the population may fall about 30% (preview estimate). Apply?

### 7.4 Era change: a key with no live lever

Turning on periodic immigration has no env equivalent, so it becomes an era change. The play loop first writes a
checkpoint at the chapter boundary, then forks from it:

```json
{"command": 23, "era_change": {"checkpoint_at": 430000, "init_from": "runs/world:430000",
  "set": {"population.immigration_every": 5000, "population.immigration_count": 256},
  "reapply_env": true, "copy_director_log": true}}
```

`reapply_env` and `copy_director_log` are the game's fork semantics (§8.2). Without them a fork starts with inert env
and no director history.

## 8. Part C: technical risks and bottlenecks

### 8.1 Visual feedback

- **Weights are not legible, so do not show them.** 4,387 fp16 weights per agent mean nothing to a player. Show inputs as
  needs, decoded actions as intents and glyphs, and income sources as archetypes. The one brain-derived trait worth
  showing is the kin attitude (§4.4), which is a measured output, not a weight.
- **Labels can mislead.**
  - Every label must pass the validity test (§4.5).
  - Labels are dimmed when borderline and hidden until their window is warm.
  - Labels that fail the test are dropped.
- **Individuals are short-lived.** At about 954 steps, a lifetime is about 32 s at 30 steps/s. The narrative units are
  therefore houses, kin groups and colonies. An ancestry camera can follow a lineage from a dying agent to its eldest
  living descendant.
- **The stream has no identity.**
  - It carries no uid, energy or labels, and the 3D viewer has no picking. The record tags in §3 fix this.
  - Today's frames carry all `n_max` rows, dead ones included. At 64k and 60 fps that is about 90 MB/s (est.).
  - An alive-only frame record (rows filtered by alive, each with a row index) brings a frame to about 1.06 MB in 3D
    and 0.83 MB in 2D. Identity sections, sent every k frames, add to that.
- **The faithful view and the live view are different paths.** `stream --follow` is faithful only at checkpoint cadence,
  and simulate-mode views ignore the director. `alife play` puts the director inside the viewed loop.

### 8.2 Deterministic replay

What already holds:
- **Trajectories.** They are bitwise identical for the same seed, resolved config, code, MLX version, backend,
  `run.compile` and `run.runtime_checks`, through checkpoint and resume (DESIGN §14.2, `test_resume_purity.py`).
- **Director replay.** It is bitwise, never calls the model, and pre-validates its logs, exiting with code 2 on a bad
  log (DESIGN §26.8).
- **Same-Mac reproducibility.** A re-simulation of `runs/64k_s0` matched all 1,200 metrics rows (EVALUATION.md §1.1).

What a game adds:
- **Player commands** are director entries with `step_applied`. Replay applies the logged entries and calls neither the
  LLM nor the compiler. `COMPILER_VERSION` is logged for audit.
- **Rewind.** A plain fork resets `env_*`, starts the director off, and has no `director.jsonl`. Re-applying the env alone
  would make an active timed effect permanent, because reverts come only from the log fold. The two options:
  - Option 1 (*recommended*): in-place resume from a kept checkpoint into a copy of the run directory. The checkpoint
    holds the applied env.
    - `truncate_after` cuts the logs to the step, extended to cover `commands.jsonl` and observer sidecars.
    - The copy hard-links the events and copies only the target checkpoint.
    - Rewind points are the kept checkpoints, so game mode raises `keep_checkpoints` and checkpoints more often.
  - Option 2: a fork that re-applies the env and copies the director entries up to the fork step.
- **Shared replays across Macs.**
  - Equality across MLX versions and backends is not promised (DESIGN §14.2).
  - The goldens are pinned per platform, because kernels can round differently across architectures (DESIGN §22.0,
    F20).
  - Equality across Mac GPU generations is untested.
  - Shared replays are therefore recorded, decimated ALSTRM files (alive rows plus IDNT and EVNT). They are independent
    of the backend and need no re-simulation.
  - Re-simulating replays from checkpoints is kept for the same machine.
- **Save size.**
  - A 64k3d checkpoint is 686 MB, almost all of it fp16 weights. At 16k the weights are about 144 MB in 2D and 164 MB in 3D.
  - Writing only alive rows breaks the digest, because dead rows keep stale genes and weights and `state_digest`
    hashes every row. Dead rows must be written in canonical form before saving, or checked with a new digest over alive
    rows only, with a save-load-step test.
- **Energy.** The ledger closes exactly on every backend. Per-agent energy still follows the float trajectory.

### 8.3 Scale against granularity

- **Cost follows `n_max`.** The step evaluates all rows and has a floor: `tiny3d` (1,024 rows) takes 2.43 ms (411
  steps/s; `docs/bench`), consistent with the 2.1–3.6 ms dispatch and host overhead estimated in DESIGN §15.3. A
  100-agent world hits that floor.

  | world (capacity) | steps/s | ms per step | basis |
  |---|---|---|---|
  | 64k 2D (about 33k agents) | 59.3 | 16.9 | measured. Fits 30 steps/s; no headroom at 60 |
  | 64k3d, C = 48 (about 42k) | 14.4 | 69.2 | measured; not real time |
  | 64k3d, C = 24 | 24.9 | 40.2 | measured; neighbour search dominates |
  | 16k 2D or 3D (about 8k, est.) | well above 30 (est.) | — | not benchmarked; G1 benches it |

- **Uniform brains.** `brain.hidden` is a compile-time constant, so no agent can have a bigger brain. The options:
  - (a) A second population with its own shapes. High complexity: neighbour search and the ledger across two state
    types.
  - (b) **Focus, not capacity** (*recommended*). The picked agent gets full per-step telemetry: inputs and decoded actions
    from a one-row decode. The protagonists at scale are groups.
  - (c) LLM narration of a few focal agents at narrative cadence, as an observer (*recommended*). Never one LLM per agent:
    Generative Agents cost thousands of dollars for 25 agents over two game days.
- **First world: 64k 2D at 30 steps/s** (the owner's decision, §10.7) for G1 and G2. G3 onwards target the 3D storm
  world (`docs/STORMS.md`, owner, 2026-10-19), whose step rate is open (STORMS §9.8).
  - **Budget.** The step takes 16.9 ms of the 33 ms frame. The remaining 16 ms must hold the observers (est. under
    0.5 ms), the stream and an LLM parse sharing the GPU.
  - **Stream.** At 30 fps it carries about 25 MB/s with alive-only frames (about 0.83 MB at 33k agents), or 45 MB/s with
    full frames.
  - **Fast-forward** is capped near 59 steps/s.
  - **G1 benches** this whole loop before any UI work.
  - **Content.** 2D has the signal tools, the pygame viewer, and the Swift viewer's face-on 2D view. It also has the
    thinnest content: kill share 0.13, a median largest body of 73, and a predation ratio of 0.078. The chosen lever,
    `attack_mult`, is aimed at that (§10.4).
  - **3D next.** 3D has more to see (depth zones, reefs, kill share 0.50) and no longer needs the fused Metal kNN kernel
    (built, DESIGN §4, row I48) to reach 30 steps/s at 64k: `64k3d` runs at about 75 steps/s on the M5 Pro.
  - **The play loop's budget (G3a, DESIGN §33.5; measured on the Mac day, 2026-10-22, `docs/RESULTS_G3A.md` §4).** 14.4 steps/s is the
    **design rate (1x)**, not a ceiling: the cadences of C5 and C6 were derived with it, and the engine step takes 14.9 ms of the 69.4 ms
    slot. Measured on the 64k3d storm planet (44,955 alive): a typical step with observers and a frame is 33.4 ms at 1x and 19.1 at `max`
    (estimate: 15.7 to 22.7), and the **boundary step** every 250 steps (the tick, the social settle of 58.8 ms at 1x, the metrics row) is
    127.6 to 130 ms at 1x and 95 at `max` (estimate 80 to 90), which is 1.84 to 1.87 slots; the bounded debt of the scheduler absorbs it
    without a `late_step` (0 in every arm and in a 20,000-step play). Fast-forward (`max`) is **52.96 steps/s (3.68x)**, 53.08 (3.69x) with
    the rule tier, so the ladder is 0.25x, 0.5x, 1x, 2x and `max` (2x was not benched). The stream at 1x is 21.17 MB/s (a frame is
    1.44 MB; 39.70 MB/s at `max`). A checkpoint (686 MB) stalls the loop 0.93 to 0.98 s on the internal disk (13 slots at 1x). The gate
    of DESIGN §33.5.4 passes (achieved 1.00006x, `step_ms` p99 47.6 ms, the client's draw p95 at most 2.41 ms with the engine running).
    The instrument is `alife bench --play` with `WorldRenderBench` beside it.

### 8.4 Further risks

| risk | evidence | mitigation |
|---|---|---|
| Thin social content | sharers fell from 6% to 1.6% (2D) and are 1.1% in 3D. In 2D the bond-lifetime proxy fell from about 135 to 12 steps (HANDOVER). In 3D the proxy (about 5 steps) is dominated by flicker, and 60% of bonds survive 100 steps (RESULTS_3D §5.4). Whether flicker also dominates the 2D proxy was not tested. Bodies are uncoordinated sheets. Clades collapse to 4 | Houses and kin groups restore display diversity. `share_eff` and research P2 (multicellularity, currently blocked) are the route to real colonies. Name things honestly ("reefs", "dominance"). |
| The director's choices are not shown to beat random | random moves matched or beat it on coverage and novelty; the LLM led on new adaptive activity by 1.3–3.9× (EVALUATION.md §3.7) | Sell faithful, explained execution of the player's intent, and measure it (G5). |
| LLM latency and GPU sharing | median 3.6–4.4 s, 18 s cold; one GPU for the model and the step | Parse off the step thread (§5.7), bench contention in G1, direct powers as a fallback. |
| Platform | MLX: Metal on Apple Silicon at speed, Linux CPU slowly. Python host loop | Mac-first. A Windows or console port needs a new backend, a non-Python host and per-backend goldens. |
| Evolution is slow in wall time | about 13 generations per 10 min (3D, 30 steps/s) | Ecological goals, time-lapse, chapters across sessions. |
| Engine regressions | CLAUDE.md conventions | `game.*` keys off and in `POST_V01_KEYS`. Scatters only in `ops.py`; static branches only. 2D digest matrix and CI goldens on every PR. The 3D golden is frozen (`legacy_state_digest_3d`, W1), so G4 needs no re-pin. |

## 9. Build plan (proposed; one PR per step, each merged before the next)

- **G1. Observers.** **Built** (DESIGN §28, Review-log row I44). Its rules are `INTENT_VERSION` 1 as built; G2's last
  stage replaced them with `INTENT_VERSION` 2 (`RULES_V2`, calibrated offline, validated on three held-out windows on
  the GPU on 2026-10-21: DESIGN §28.4, §28.8, row I45).
  - `game.observe` (unhashed), `StepOutG.acts`, and `BirthRecords.parent_slot` (moved to G2, see below).
  - The accumulators, archetype and intent rules (`INTENT_VERSION`), and the observer sidecar.
  - `signals.decode` and the SIGNALS analysis generalised to 3D (needed only when a game world is 3D).
  - The §4.5 validity analysis.
  - Benches of the 64k 2D world at 30 steps/s: the step with observers on, the stream, and the step rate with an LLM
    call in flight.
  - Tests:
    - the trajectory and digests are unchanged with observers on and off;
    - the accumulator arithmetic is checked on spike and constant inputs;
    - the sidecar round-trips.
  - **As built, and deferred.**
    - `BirthRecords.parent_slot` moved to G2, with the houses, kin groups, colonies, territory and relations, and the
      acts that serve only them (`attack_target`, `bond_target`, `q_kin`). The "in a colony of 4 or more" clause of
      Settler is G2's too: G1's Settler is the speed rule alone.
    - `signals.decode` and the SIGNALS analysis in 3D are deferred until a game world is 3D.
    - **Merge condition, met 2026-10-19:** `tests/test_game_purity.py` and `tests/test_game_acts.py` pass on Metal
      (26 passed on the M5 Pro; DESIGN §28.1, §28.8). The CPU tests pin the channel's read set structurally, and the
      Metal run shows that observing leaves the compiled float kernels' rounding alone.
    - The benches and the calibration run wait for the GPU, with the procedures of DESIGN §28.8. `alife bench
      --observe` and `python -m alife.analysis.labels` are built; the stream rate and the step rate with an LLM call
      in flight are measured by hand on the Mac (DESIGN §28.8), and the combined play-loop bench is G3's.
    - Copying the source's sidecar into a fork is G3 (rewind and replay) or G4 (era change). In G1 every fork,
      a replay fork included, starts its observers fresh, so its labels agree with the source's only approximately
      after the warm-up.
    - The stream tags and the colour modes are G3's, observer values in the director's vitals are G4's.
- **G2. Social tracking.** **Built, and run on the GPU on 2026-10-21** (DESIGN §29, Review-log row I45; every threshold is
  provisional, `SOCIAL_VERSION` 1: the calibration run measured them and changed none). Houses, kin groups, the colony tracker, territory, the relation
  matrices and temperament, on two generic modules (`groups.py`, pure MLX, and `tracking.py`, host numpy) with the
  game layer in `game/social.py` and `game/society.py`.
  - **As built, and deferred.**
    - The channel carries a parent row (`born_parent`) in place of `BirthRecords.parent_slot` (§4.2).
    - Houses are int64 rather than `i32`, with `HOUSE_UNKNOWN` for immigrant lineages of an observer attached mid-run,
      seeded from clades and flagged inexact; the rate is the config key `game.house_rate`, 1/2048.
    - The kin grid has width 1/8 (the evidence is in DESIGN §29.5), with hysteresis on the enter and leave sizes;
      `sim_self` is the chaining report; temperament is read at the snapshot step.
    - Depth bands (each kin group's mean height and height-band fractions in 3D) are built.
    - The **roster** (the last stage, DESIGN §29.14) is built: kin-group names, house events, and each listed kin
      group's window births and deaths by cause, centroid and archetype mix (§6.2, §6.3), for the chronicle of G4.
      `game.house_rate` stays 1/2048; the GPU verification reports, on `64k3d_s1` and `64k_s0`, the share of members in
      houses founded after the observer's start against those seeded from clades (`analysis.social`); the owner confirmed
      1/2048 on 2026-10-21 (§10 item 13).
    - **`INTENT_VERSION` 2** (the last stage, one change after a data study, DESIGN §28.4, §28.8): `RULES_V2` is the
      default and carries the Settler colony clause at `settler_body` 4 with the calibration fixes of G1's failing labels
      (Gatherer a harvest level, Defending after a recent hit, Budding a forecast shown outside the hold, Bonding shown as
      Tethering, Feeding and Signalling at 3/4 and 0.75, Caring ahead of Defending and Feeding ahead of Signalling).
      `RULES_V1` stays defined and keeps its G1 digest. The calibration was offline; the validation on held-out windows ran on
      the GPU on 2026-10-21 and every label passed the provisional criterion (item 12; DESIGN §28.8). The Budding overlay, the renamed labels and the clause at 4 are the coordinator's choices
      of 2026-10-19 under the owner's delegation, to revisit.
    - **`INTENT_VERSION` 3** (storm world, W2a, DESIGN §32.7, row I50): `RULES_V3` is the default; Nomad and Settler on the wind-relative velocity, and the
      Drifter, Floater and Sinker archetypes (§4.4). Calibrated on one pilot window and frozen; the held-out validation is W2b's.
    - The snapshot cadence is a field of `SocialRules`, not a config key. The reports go to `observers/social.jsonl`,
      not into the rows.
    - The **Family** row of §3 (a host lineage store, an ancestry panel, a "kin nearby" list) is deferred to G3, with
      the stream tags and colour modes.
    - **Merge condition, met 2026-10-21:** `tests/test_game_purity.py` and `tests/test_groups.py` on Metal, without a `-m`
      filter (the twelve game and analysis files gave 509 passed, 202.7 s). The 64k bench, the calibration, the churn rate,
      the house-rate report and the costs were measured the same day (DESIGN §29.12): observed 54.8 steps/s in 2D (target 30)
      and 13.9 in 3D, overhead 1.38 and 2.07 ms a step (the 3D figures were measured before the fused kNN kernel, I48: with it 64k3d runs at
      about 75 steps/s, so the same observer cost is about 15% of a step; a re-measurement on the GPU is pending); kin groups track well; **colonies churn 80 to 83% at the 250 cadence**
      (whether 1000 is better is not established: those studies had 2 snapshots), so a colony id at 250 is not a stable identity; the host settle is 32 and 38 ms per snapshot;
      under 0.1% of members are in houses founded within 2,500 steps. The stream rate and the step rate with an LLM call in
      flight were not run.
- **N1. World Report, offline.** **Built** (DESIGN §31, Review-log row I49; `REPORT_VERSION` 1, every threshold provisional until the
  owner read). `python -m alife.analysis.world_report build | verify | probe` (the module is not `analysis.report`, which is the activity
  report) turns an **observed** run's files into a chapter report: headline, lead, a body of at most six sentences, a complete
  interventions block and a ticker. It is offline and host-side, with no engine change, no config key and no live UI.
  - **The template path is the product** (coordinator choice C12, 2026-10-19, under the owner's delegation, to revisit): code computes the
    fact sheet (`game/report.py`) and writes every sentence from 88 templates over a closed vocabulary (`game/templates.py`); a checker
    (rules R1 to R7, `game/lexicon.py`) re-derives each string from the facts. **Select-and-order** (`--llm --model ID`) lets a local model
    choose and order code-written candidates by id and nothing else; the model's text is never shown, quoted or put into a prompt.
    The section order of the body is code's (C13) and the mandatory sentences cannot be dropped. **Free prose is off** (DESIGN §31.12).
    The preflight `GET /api/v0/models` is read-only and allowed (C14); the tool never loads or unloads a model.
  - **Replay:** records hold the facts, the candidates, the model's per-attempt choices and the rendered report; `verify` recomputes the
    sheet from the run and replays the choice, with no model call (exit 1 on a mismatch, 3 when the run changed under equal versions).
  - **As built, and deferred.**
    - A run must be observed (`game.observe`); the unobserved `eval_*` runs are reached through observed replay forks.
    - Labels are narrated only on a key in the version-1 verdict table; `INTENT_VERSION` 2 is the default, so a new run narrates no label
      until the change that validates `RULES_V2` adds its rows.
    - The window sentence W is off and its facts are gated against confounders; whether any are shown is the owner read's decision.
    - **Review fixes (stage 4).** A patch move is a point event (the patches drift every step, so "still in place" was false); an
      intervention whose span ended at the chapter's first row is not listed (the lever it restored is a standing line); a deep dip is
      headlined "Population dipped to {min}"; the relation sentence says "on a victim that died in the same step"; a build of one range
      keeps the records of another; a request goes only to a `/chat/completions` URL (DESIGN §31.1, §31.14 item 15).
    - **Not run:** the GPU validation (the observed forks A and B of DESIGN §31.13), `probe` and `build --llm` against a real model, and the owner read
      (20 chapters, template against select, blind). The build made no LLM call.
    - **Not in N1:** `alife play`, a report panel, a worker and `reports.jsonl` in a run directory (G3); player commands (G4); storm facts
      (W3; the storm-world report built on the same sheet, `docs/STORMS.md` §6.5, which records what N1 built).
- **G3a. The play loop and the World window** (built; DESIGN §33, Review-log row I51; the Mac day ran 2026-10-22, the gate passes, `docs/RESULTS_G3A.md`; the click-through by hand is open). On the 3D storm
  world, on W1's overlay values (W2 changes coupling values and labels, not records; after the merge with W2a the stream carries the
  version-3 labels with N1's validity gate, DESIGN §33.7). One process per world, `alife play`:
  - the fixed-timestep scheduler (pause, 0.25x, 0.5x, 1x, 2x, max) as `runner.run(pacer=)`, with `play.lock`, `play.json`, planets
    (a fork with the weather on and spun up), exact houses at forks, `--adopt`, `--truncate`, `--replay`;
  - the stream records `FRMA IDNT INTN TERR EVNT STRM MAP1 ENVS CLCK FOCS`, each with a Swift decoder, an independent Python
    decoder and shared fixtures; the chronicle's closed template table; the observer colour modes (`house kin archetype
    intent`); the host storm tracker moved out of `analysis/w0.py` with its persistence;
  - the World window: the planet map and organisms (minimum image, alive-only frames, semantic levels), picking and follow, the
    Cast, the inspector, the Chronicle and toasts, the keys; `alife bench --play` and `WorldRenderBench`.
  - Not in G3a: chapters (a checkpoint every 6,000 steps is a save point only), auto-pause, alerts, the ticker and the live World
    Report, powers, dead-row canonicalisation, recorded replay files, rewind, offline advance, any LLM.
- **G3b. Chapters, alerts, the live report and direct powers** (built; the Mac day ran 2026-10-23, `docs/RESULTS_G3B.md`; DESIGN §34, Review-log row I54; the specification is `g3b/spec.md` revision 2, every owner decision a coordinator's recommendation, choice C20, to revisit).
  **Observe-and-poke** (D9): no score, no objective, no win state; storm and lineage events are information, not decisions. What is in it:
  - the **chronicle quality** first (kin threads: 139 kin rows to 45 told events on the 20,000-step play; storm persistence), then chapters of 6,000 steps with a pause and a card at their end, `autopause`, **alerts** (18 classes, a stack of
    the `alert` surface only, a provenance rule so that the aftermath of a power never pauses), the **live World Report** (report version 2, template path, built on a worker thread inside the checkpoint stall and replayable byte for byte);
  - **direct powers** through a player channel of the director (`Director.apply_pending`, rule P: a player's record at step `s` is after the checkpoint at `s`): Food, Bloom, Blight and Strike under `GAME_SPECS` "game-v0" (an override of `MENU`,
    durations 3,000 to 12,000, influence 6 units per 18,000 steps, `power cancel` with no refund), previews with a versioned precedent, a since card per power; `intent` is answered "not in this build" (G4);
  - the records `ALRT RPRT PLAN`, the lines `autopause watch unwatch ack power intent`, and the window: chapter bar, alert stack, chapter card, Report and Powers tabs, aiming on the map, keys, ticker, dock icon;
  - **Not in G3b:** the storm verbs and any weather power (W4: new hashed `env_*` fields), `intent` (G4), the select-and-order LLM mode (G3c), recorded replay files, rewind, offline advance, canonical saves (G3c).
  - **Measured on the Mac day (2026-10-23, M5 Pro, one planet; `docs/RESULTS_G3B.md`, DESIGN §34.8):** the gate passes at 1x (0.99810x `rules`, 0.99805x `player`, 0.99753x with two scripted powers; `late_steps` 0; `step_ms` p99 42.1 to 43.1 ms), `max` (3.75x to 3.82x) and 2x (1.99x, p99 at most 29.0 ms); click to applied p95 0.102 s;
    digests equal to the plain run (no powers) and to the replay of the arm's log (powers). The report-time item is not readable in the bench (an arm has no history: 2.5 to 2.7 ms is an arrival notice); in an 18,001-step scripted play the report came 29 to 47 ms after the card opened, inside a 0.68 to 1.01 s save.
    Bands of the design: chronicle 3.28 events a minute (G3a 8.66; band 2 to 5), kin 2.04 (5.99; band 1 to 3), alerts of importance 2 or more on the alert surface 0.38 a minute (band at most 0.6); 4 auto-pauses in 18,001 steps, one of them a fall after the player's own Blight (the band says it must not pause: an open decision).
    Measured afterwards: the external drive (5.6 to 6.4 s for the checkpoint against 0.4 to 0.5 s internal, `RESULTS_G3B` §6.1). Not measured: a second window, a person's inputs per chapter (D9: the Mac day's reader is a script).
- **G3c. Time** (not built; DESIGN §35 and Review-log row I55 are reserved for it; spec `g3b/spec.md` §9). After the G3b Mac day:
  - recorded ALSTRM replay files and the recap player;
  - rewind as a branch (a clone into a copied run directory, its cost stated), offline advance under the rule tier with per-launch GPU consent;
  - dead-row canonicalisation of saves only if the save breakdown shows serialisation is at least 40% of the stall, and then with a separate `canonical_digest`, never a change of `state_digest`;
  - the select-and-order report mode (N1 §31.11, default off, 60 s timeout, a reasoning budget in the client, a `late_steps` gate with a request in flight).
- **G4. Player intent.**
  - The intent schema and prompt, the compiler, `commands.jsonl`, the preview, `source:"player"`, and the separate pacing
    scope.
  - `GAME_MENU` with `attack_mult` (§10.4), and inert-fill for `State3` (no 3D golden re-pin: W1 froze the list, §5.4).
  - `GAME_SPECS`, and the `game_master` persona with its `PersonaParams` and the intensity vital.
  - Direct powers beyond G3's, and the `game_master` safety re-scope.
  - Era changes: a checkpoint at the boundary, then a fork with `reapply_env` and `copy_director_log`.
  - Lever-response sweeps to calibrate the compiler.
- **G5. Evaluation.**
  - Intent fidelity: for scripted commands, does the targeted vital move as asked within the horizon, compared with a
    yoked random plan with the same budget?
  - Label validity on new worlds.
  - The owner's legibility ratings on the 1–5 scale used for the journal.

## 10. Decisions (owner)

**Decided 2026-10-18:** items 1, 2, 3, 4, 7 and 8; **2026-10-19:** items 13–19 (G2). Items 5, 6, 9 and 10 are open; their
recommendations stand until the owner decides. Items 11 and 12 came up in G1 (below) and are open too.

1. **Scope.** *Decided:* a Mac-first prototype in this repository.
2. **Code location.** *Decided:* a `src/alife/game/` package with a `game` config section, everything off by default.
3. **Intent translation.** *Decided:* the staged pipeline of LLM, intent vector, compiler, preview and confirm (§5.3).
   The current Architect, with the LLM choosing menu actions, stays only as the G5 baseline.
4. **New levers.** *Decided:* one lever, chosen for its effect on play: **`attack_mult`**. The owner asked for one that
   makes a difference for gameplay. The reasons for this one:
   - It acts within one or two lifetimes, so the player sees the response in under a minute.
   - The chosen 2D world has the thinnest predation (kill share 0.13, predation ratio 0.078), so it is where a lethality
     lever changes most.
   - It gives the competition dimension a direct lever.
   - It feeds the Hunter archetype (§4.4) and the kin-group war relations (§6.5).

   The alternatives make less difference to play:
   - `share_eff` and `tempo` act on evolutionary timescales;
   - `harshness` overlaps with the food levers;
   - `regen_set` only lifts the time limit of an existing lever.

   Consequences for v1:
   - `attack_mult` adds a `State` field; the 3D golden stays (W1's `V0_FIELDS_3D`, §5.4).
   - In compiler version 1 the harshness, cooperation and tempo dimensions compile to no-ops, which the preview states.
   - Long scarcity uses a 36,000-step `regen_pulse`.
   - The schema keeps all eight dimensions, so later levers do not change it.

  *Reopened 2026-10-19* for the storm world (`docs/STORMS.md` §9.1): the case above rests on 2D's thin predation, and
  3D's kill share is 0.50. Revisit before G4.
5. **During a parse.** Keep stepping (*recommended*), or pause until the player has confirmed the preview (§5.7).
   *Coordinator's choice under the owner's delegation of 2026-10-19, to revisit:* pause while composing in 3D, where a
   parse slows the step anyway; keep stepping in 2D (`docs/STORMS.md` §6.5).
6. **Rewind.** Era changes need the fork with `reapply_env` and `copy_director_log` either way (§7.4). The question is
   only what backs rewind: in-place resume into a copied run directory (*recommended*), or the same fork (§8.2).
7. **First world.** *Decided:* 64k 2D at 30 steps/s (§8.3). *Amended 2026-10-19:* G1 and G2 stay on it; G3 onwards
   target the 3D storm world (`docs/STORMS.md` §9.1), with the step rate open (STORMS §9.8).
8. **Factions.** *Decided:* kin groups (tag clusters, which agents perceive) are the factions, for relations and
   territory (§6.3, §6.5). Houses stay as genealogical names only (§6.2).
9. **"Trader".** Reserve it for reciprocal sharing and call one-way givers "Providers" (*recommended*), or add an exchange
   mechanism, which is new dynamics with its own ledger term.
10. **The Architect discrepancy.** DESIGN §26.6 and DIRECTOR.md §12 (A15) say "safety rules only". DESIGN §26.7,
    `rules.ARCHITECT` and `test_director_rules.py` give the Architect discretionary moves when the LLM tier is off.
    Correct the two sentences (*recommended*), or change the code under a new rules version.
11. **Label thresholds as code (G1).** G1 builds the thresholds of the archetype and intent rules as versioned code
    constants (`rules.RuleSet`, `INTENT_VERSION` 1 as G1 built it and 2 as G2 changed it, a pinned digest each), not
    as config keys, so a label means the same in every world and the sidecar can tell which rules wrote it (DESIGN §28.4). *Recommended:* keep. The alternative,
    per-world config keys, needs a decision on whether they enter `config_hash` (they would move every hash that sets
    them) and a sweep interface for the calibration.
12. **The verdict criterion (G1, provisional).** A label passes the validity test (§4.5) when its AUC against the
    matched random label has `p ≤ 0.01` and `auc − null_mean ≥ 0.05`, with at least 30 positives and 30 negatives, at
    least 99 permutations, and (intents) a coverage maximum of at most 0.6 (DESIGN §28.8). The numbers are G1's
    placeholders and still provisional: the owner has not confirmed them. They were applied as written to the three held-out
    windows of `INTENT_VERSION` 2 (DESIGN §28.8), fixed before those reports were read. *Result, 2026-10-21:* every archetype
    and every intent's displayed label passes on all three windows (30 of 30 intent cells and 18 of 18 archetype cells; every
    p at its 0.005 floor; thinnest margins 3D Resting +0.112, 3D Roaming +0.124, 2D Settler +0.218; fewest positives 53, for 3D
    Trader). No label failed, so none is marked as not validated and no rule was changed. Reported, not blocking: the 3D
    Signalling predicate covers 0.674 of rows on `64k3d_s1` (its displayed label, 0.171). The evidence is limited: the 2D
    window is the calibration run 4,000 steps later. A different criterion would have to be applied to new windows, because
    these are read. Confirm or change the criterion.
13. **The house rate (G2).** *Decided 2026-10-19:* `game.house_rate` defaults to 1/2048, not "off". It is an
    observer-only key, unhashed at every value and inert while `game.observe` is off, so a working default breaks
    nothing. CLAUDE.md records the exception to the "new keys default to off" convention. Confirmed by the owner 2026-10-21, after the GPU verification (founded_share 0.00068 in 2D and 0.00091 in 3D after 2,500 steps): keep 1/2048 for now.
14. **Kin-group grid width (G2).** *Decided 2026-10-19:* width 1/8 with single linkage. Chaining within a clade is
    reported (`sim_self`), not fixed by a mode-seeking rule. Width 1/4 merges two 3-D clades and leaves 4–5 groups in 2D
    (DESIGN §29.5).
15. **Matching threshold and hysteresis (G2).** *Decided 2026-10-19:* θ = 0.3 with the hysteresis as designed: enter at
    `max(8, n_ref >> 9)`, leave below `max(4, n_ref >> 10)`, `n_ref` with a 12.5% deadband, sticky slots within the top
    20, colonies entering at 4 and leaving below 3. A fragmentation is `dissolved` plus `born` with provenance, and a
    system that ends and comes back gets a new id. The values stay provisional until the calibration.
16. **`INTENT_VERSION` 2 (G2).** *Decided 2026-10-19:* version 2 carries the Settler clause and the calibration fixes of
    G1's GPU study (Gatherer, Defending, Budding, Feeding, and in 3D Bonding, Signalling, Roaming and Caring's too few
    positives, because G3 onwards targets the 3-D storm world). The redesigned rules come from a data study whose one
    change defines `RULES_V2`, makes it the default under version 2 and pins its digest (DESIGN §28.4); G2's earlier
    stages built the clause inert at its version-1 value. *Built* in G2's last stage. The coordinator's choices under the
    owner's delegation of 2026-10-19 (to revisit): the Budding overlay (§4.4's hold sentence), the display names
    Tethering (Bonding) and Unspecialised (Generalist), the meaning changes of Gatherer, Defending and Budding, and the
    colony clause at 4 (its body-only part does not predict a low radius of gyration in 2D; against the slow rows it
    carries information, DESIGN §28.4).
17. **Houses of observers attached mid-run (G2).** *Decided 2026-10-19:* seeded from clades, immigrant lineages as
    `HOUSE_UNKNOWN`, flagged inexact. No offline rebuild tool.
18. **Relation baseline (G2).** *Decided 2026-10-19:* the per-step sensed-neighbour exposure by actor slot, neighbour
    slot and neighbour count, with the sampled tag similarity at snapshots and "distinct" below 0.25; not the territory
    co-occupancy (DESIGN §29.8).
19. **Where the reports live (G2).** *Decided 2026-10-19:* the reports move out of `metrics.jsonl` into the append-only
    `observers/social.jsonl`, which `runio.truncate_after` cuts on resume. Rows keep a head of at most 0.5 KB (DESIGN
    §29.10, §29.11).

20. **G3a: the play loop and the World window (coordinator's choices C19 under the owner's delegation of 2026-10-19, each the
    spec's recommendation, to revisit; DESIGN §33).**
    - *D1* 14.4 steps/s is 1x (a play setting, `--base-rate`, one number to change after play-testing); the ladder is 0.25x to 2x
      plus `max`.
    - *D2* the first planet is a fork of `64k3d_s0` at 500,000 steps (its events are present, so the houses are exact) with the
      storm overlay, a 20,000-step spin-up and `director.enabled`; a source without event chunks gets clade houses with a note.
    - *D3* the dock is on the right, 360 pt, collapsible; the bottom (240 pt) and the left are options.
    - *D4* `runner.run(pacer=)` instead of a second loop, and `EngineSession` extracted from `StreamController`.
    - *D5* a checkpoint every 6,000 steps and on quit, the last three and the fork's origin kept, written synchronously with a
      "saving" state.
    - *D6* the chronicle says only what the observers support (kin groups, houses, storms, director entries, the focus), in code-written
      sentences; the labels appear in the inspector and the cast tagged provisional.
    - *D7* two small engine edits: `RunDir._prune` keeps a fork's origin checkpoint, and `phylo.load_tree(end=)`.
    - *D8* a planet opens running at 1x. *D9* no Swift job in CI (accepted risk; the Mac checklist has the Swift line).
      *D10* a single click selects and follows; shift-click selects only. *D11* organisms are not instanced at the planet level.

21. **G3b: chapters, alerts, the live report and direct powers (coordinator's choices C20 under the owner's delegation, each the
    spec's recommendation after review round 2, to revisit; DESIGN §34).**
    - *D1* G3b now, G3c after the G3b Mac day. *D2* the kin-thread rule (P = 3, R = 3, DELTA = 0.15; storms 8 detections; frozen thread names with
      a suffix on a collision). *D3* the chapter end pauses; alerts pause only for `pop_fall` at importance 3, `pop_low` and engine failures; the narrative classes are log
      rows; a fall shallower than 15% that follows a regen or habitat entry by at most 500 steps never pauses (a fall of 15% or more pauses whatever the cause: the owner, 2026-10-23, below); the default launch is `player`, not `rules`.
    - *D4* Food, Bloom, Blight, Strike under "game-v0", 6 units per 18,000 steps, `power cancel`; no weather verb (W4). *D5* `intent` is G4 (the line and the `PLAN` shape are reserved).
      *D6* the select-and-order mode is G3c, default off. *D7* rewind, storage, digests: G3c. *D8* offline advance: G3c, never default.
    - *D9* the play contract is observe-and-poke; the Mac day was to measure inputs per chapter and the objective is decided on that (it could not: the Mac day's reader is a script, so D9 needs a person to play G3b; `RESULTS_G3B` §10). *D10* Strike costs 1 (radius 6.25 to 25, cooldown 3,000, not GAMES §5.5's 9,000). *D11* Chapter 1 is reported from an origin row in `meta.json`.
    - **Decided by the owner, 2026-10-23, after the Mac day (the owner's decisions, not the coordinator's; none changed behaviour):**
      (1) *the fourth pause:* a population fall of 15% or more pauses the game whatever the cause, including right after the player's own power
      (the stage-8 behaviour; D3's "a fall that follows a power never pauses" is superseded for falls of 15% or more; on the Mac day's window it was the 21.9% fall after the player's Blight);
      (2) *item 15, satellite budding of kin threads:* kin groups that bud off a large group count as new groups in the chronicle; `P` stays 3 and `KINTHREADS_VERSION` is unchanged
      (the measure: a thread named Pruashol formed four times and merged back three times between steps 474,750 and 487,250 and 4 of 19 ended threads lived 1,500 steps or less);
      (3) *item 16, storm namings on the alert surface:* they stay there, as D3 has it (4 of the 8 alerts of importance 2 or more, 0.38 a minute in all, 0.19 without them).
    - **Still open:** D9 (the objective), which waits for the owner's first play-through of G3b (inputs per chapter), and D10 (the Strike cooldown).

22. **Storage: the size monitor and the save mode (the owner, 2026-10-23, from his first play of the World window; DESIGN §34.10, row I56).**
    The request: "an important UI element that we should include is live monitoring of run file size, if a run is being recorded. Users should
    also be able to switch off recording easily from the side panel." A played world records no video; about 99% of its disk is the saves
    (a 686 MB checkpoint at every chapter end, every 6,000 steps and on quit; the last three and the origin are kept, so a world plateaus near
    3 GB), the rest the logs (a few MB a chapter, needed by the reports and the replays).
    - *Decided by the owner:* **switching recording off means keep only the newest save** (plus the origin, which is always kept); saves still
      happen at chapter ends and on quit, each replacing the previous one; the logs are still written. D5's "the last three" is the default
      (`all`), `newest` is the switch.
    - *Built from it (the coordinator's choices, to revisit):* the switch deletes nothing at once, the next save prunes (and the window says
      so); the mode is kept in `play.json` and logged to `commands.jsonl`; the size is measured by the client (stat only, every 5 seconds and
      after each save) and shown in a one-line Storage strip under the clock strip with the switch, a warning under three saves of room; a
      700-point dock keeps its two alert rows beside the inspector, but three need 760 points now.
    - *After the review (also to revisit):* the switch is a **Recording** toggle (on keeps the last saves, off the newest only), not
      confirmed (the tooltip says that older saves go at the next save); a research run played in place (`--adopt`) keeps Recording on
      from the window, since off would delete the research's checkpoints, and only `alife play RUN --saves newest` turns it off there,
      with a note of what it deletes.
23. **The test world (the owner, 2026-10-24; DESIGN §37, row I62).** The request: a smaller test world of about 8 to 16 thousand organisms for
    quick play-tests and lighter streams, launchable from the World launcher. The owner said yes to the go/no-go the same day.
    - *Built:* the preset `mini3d` (a quarter of `64k3d`'s area at the same density, about 10,500 alive; 4.5 ms an engine step with the weather on
      against 14.9, a 4.6 MB/s stream against 21.2) and its weather overlay `storm160`; no simulation change. It sits beside the 64k3d planets in the
      build order as the world for play-tests, not as the game's world: the label-validity gate of N1 and G3 has no verdict row for it.
    - *Decided by the coordinator under the owner's "yes" (C22, to revisit):* the pool geometry (24 x 16, two per hemisphere, `f0` 0.0165) with a
      36,000-step spin-up for 160 worlds; a fresh dry 500,000-step source judged at 250,000 steps on kin groups (at least 5 tracked, top share below
      0.6, at least 3 territory owners), the crop of a 64k3d checkpoint only if it fails; the launcher picks the overlay from the source's size;
      no launcher entry that makes a world from nothing (a source is a run in `runs/`).

23. **Light saves: a planet's state in about 230 MB, not 687 MB (the owner, 2026-10-24; DESIGN §34.11, row I64).**
    The request: "a lightweight option to resume a planet's achieved state for a later run, without having to record large files on disk."
    - *Decided by the owner:* (1) lossless **zstd** saves with an exact resume (the dependency `zstandard`, over the stdlib `lzma`); old uncompressed saves keep loading everywhere; (2) a **one-save mode**: keep only the newest save and drop the origin, after auditing what reads the origin (each such reader degrades with a message, never a crash); (3) **a planet made from an own world links the source save** instead of copying it (`os.link` on the same volume, else a copy), and the launcher makes **Resume** the obvious way to continue a world. Data safety first: at every instant one complete, loadable save exists, and a crash or kill at any point leaves a resumable world.
    - *Built:* zstd level 15 (0.365 of the bytes, about 4 s to compress on 18 threads, 0.41 s to decode; the review of stage B4 moved it from 17 for the quit budget); the save is written plain, compressed on a worker thread, and the older saves go only after the new one is whole; `saves single` as the third choice of the Storage strip (All, Start, Newest); the launcher's hint under a played world's source (Resume stays first); the planet's origin is a link only when the state is the source's byte for byte.
    - *Open, the owner's decision:* a planet made from a played world is not linked. The fork's transformation resets the diagnostics, the ledger and the levers, so its state is not the source's, and the planet_a chain's "Make planet" writes a new compressed origin (about 230 MB) instead of sharing the 687 MB save. Linking those needs (a) the fork to keep the source's accounting (a change of the planet's energy accounting, §33), or (b) the transformation stored with the origin and applied on load (a larger file).


## 11. Prior-art sources

The research pass opened each page, except where marked.

- Creatures: https://en.wikipedia.org/wiki/Creatures_(video_game_series). Grand, Cliff & Malhotra (abstract only):
  https://www.sussex.ac.uk/informatics/cogslib/reports/csrp/csrp434.pdf
- Black & White: https://en.wikipedia.org/wiki/Black_%26_White_(video_game)
- The Bibites: https://store.steampowered.com/app/2736860/
- Species: https://store.steampowered.com/app/774541/
- Ecosystem: https://store.steampowered.com/app/1133120/
- Thrive auto-evo: https://www.revolutionarygamesstudio.com/devblog-20-lets-try-to-solve-evolution
- Spore: https://en.wikipedia.org/wiki/Spore_(2008_video_game)
- The Sims and needs-based AI (Zubek): https://robert.zubek.net/publications/Needs-based-AI-draft.pdf
- Dwarf Fortress needs: https://dwarffortresswiki.org/index.php/Need
- RimWorld storytellers: https://rimworldwiki.com/wiki/AI_Storytellers
- Left 4 Dead AI Director (Booth, GDC 2009): https://steamcdn-a.akamaihd.net/apps/valve/2009/ai_systems_of_l4d_mike_booth.pdf
- Generative Agents (Park et al. 2023): https://arxiv.org/abs/2304.03442
- Project Sid: https://arxiv.org/abs/2411.00114
- Where Winds Meet NPC exploits:
  https://www.gosugamers.net/entertainment/news/77655-where-winds-meet-players-outsmart-ai-npcs-to-easily-get-sidequest-rewards
- Risi & Togelius, neuroevolution in games: https://arxiv.org/abs/1410.7326
