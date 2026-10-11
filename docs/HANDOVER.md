# Handover

Status of the `alife` project as of 2026-10-11, written for the next coding session (a local Claude Code
session on the owner's MacBook Pro, M5 Pro, 48 GB). Read this, then `CLAUDE.md`, then the parts of
`docs/DESIGN.md` you need.

## What this is

A neural artificial-life engine in Python on Apple MLX. It replicates the agent affordances from a Fraser Price
X thread: agents with per-agent recurrent neural controllers sense, signal, move, bond, split, exert forces,
harvest, share energy, fight, reproduce with mutation and die. There is no backprop and no fitness function.
`docs/DESIGN.md` is the normative spec. Its Review log at the end records every deviation and fix.

## Repository map

| path | what |
|---|---|
| `src/alife/` | engine (step, physics, bonds, energy, lifecycle, fields, brain, ...), run tooling (runner, runio, metrics), render, viewer, bench, CLI |
| `tests/` | pytest suite. About 530 fast tests (`-m "not slow"`, about 2 min on Linux CPU, 15 s on the Mac) plus 18 slow ones |
| `docs/DESIGN.md` | the spec: state schema, step order, ledger, RNG streams, config, tests, decision and review logs |
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
cd gui && swift run                           # the macOS control app
```

Overnight runs: plug in, lid open, and either use the GUI's timer or
`caffeinate -is uv run alife resume ...`. Stop cleanly with exactly one SIGINT (Ctrl+C or `kill -INT <pid>`
of the Python process). A second SIGINT is a hard exit. Do not use `--quiet` if you want progress lines.

## Verified state

- `main` holds the v0 engine (PR #1) plus the GUI, periodic immigration and seasons (PR #2), both
  squash-merged. Branch `gui-fork-prefill-swift6` adds the GUI follow-ups below.
- Linux CPU (container and CI): 530 fast tests pass, ruff clean. The golden digest is pinned for MLX 0.32.3 on
  Linux-x86_64-cpu, and the test skips elsewhere.
- Mac (M5 Pro, Metal), v0 code: 505 fast and 18 slow tests pass. `bench --preset 64k`: 61.6 steps/s
  (16 ms/step), 1.1 GB peak, weights updated in place, deterministic. The neighbour search is the biggest stage
  (33%).
- Mac, current `main` (immigration and seasons included): 529 fast tests pass, 1 skipped (the Linux-only golden
  digest), ruff clean, about 18 s.
- Metal quirk (DESIGN F20): float32 GPU matmul is reduced precision (about 8e-4 relative). Anything that must
  be accurate avoids GPU matmul.
- GUI on the Mac: it compiled first time (Swift 6.4 toolchain, macOS 27 SDK). The levers and CLI flags were
  checked against `config.py` and `alife --help`. The owner clicked through the tiny-preset checklist below
  before and after the Swift 6 migration, and every step passed.

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

## GUI follow-ups (branch `gui-fork-prefill-swift6`)

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

## Suggested next steps

1. Experiments with the new levers. Fork `runs/64k_s0` from its latest checkpoint, for example with
   `immigration_every=5000 immigration_count=64`, then `season_period=20000 season_amplitude=0.6
   season_mode=per_patch`, then a higher `mutation.rate_min`. Compare against an unforked control and a second
   seed (`--seed 1`).
2. Optional engine work, deferred in DESIGN §23.2:
   - a fused Metal kernel for the neighbour search, the biggest stage;
   - a mean-speed metric, so "sessile vs mobile" is measurable;
   - `inspect` showing the resumed total instead of the original `steps`.
3. If a change alters trajectories, re-pin the golden digest deliberately (tests/test_determinism.py) and note
   it in the DESIGN Review log.
