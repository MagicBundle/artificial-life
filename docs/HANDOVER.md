# Handover

Status of the `alife` project as of 2026-10-11 (after director build step 4 and the yoked `random` persona), written for the next coding
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
| `src/alife/` | engine (step, physics, bonds, energy, lifecycle, fields, brain, ...), run tooling (runner, runio, metrics), render, viewer, bench, CLI, and `director/` (the host-side ecosystem director, see below) |
| `tests/` | pytest suite. 730 fast tests (`-m "not slow"`, about 50 s on the Mac) plus 19 slow ones |
| `docs/DESIGN.md` | the spec: state schema, step order, ledger, RNG streams, config, tests, decision and review logs (the director is §26, the LLM tier §26.10) |
| `docs/DIRECTOR.md` | the design of the ecosystem director; steps 1 to 4 are built, §12 to §14 record how steps 2 to 4 refine it |
| `docs/RESEARCH.md` | research directions (emergence, open-endedness, signalling) assessed against prior work, with a recommended order |
| `docs/ACTIVITY.md` | evolutionary-activity analysis of `runs/64k_s0` (method, results, sensitivity); figures in `docs/activity/` |
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
- Signalling was abandoned: emitters fell from 81% to 26%, flat since about 1M. Energy sharing nearly vanished
  (6% to 1.6%). Bonds became brief contacts: the lifetime proxy fell from about 135 steps to about 12.
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

1. **The §9 evaluation:** fork `runs/64k_s0` at step 120,000 into the arms (director off; rule tier, Gardener; rule +
   LLM, Trickster; optional Architect) with the same seed, and compare archive coverage, persistent novelty, clades,
   extinctions and rescues, and intervention cost, plus the LLM's validity, clamp rate and latency. About 6 hours per
   arm.
   §9 now also requires a matched random-director arm R (the `random` persona is built: run B first, then R with
   `--yoke-log runs/B/director.jsonl` and the same fork point, seed and `rule_every`), at least two
   seeds per arm, and Bedau–Packard activity as an outcome metric (`docs/RESEARCH.md` §4).
2. Optional engine work, deferred in DESIGN §23.2:
   - a fused Metal kernel for the neighbour search, the biggest stage;
   - a mean-speed metric, so "sessile vs mobile" is measurable;
   - `inspect` showing the resumed total instead of the original `steps`.
3. If a change alters trajectories, re-pin the golden digest deliberately (tests/test_determinism.py) and note
   it in the DESIGN Review log.
