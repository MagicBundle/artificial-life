# Alife Control

A small native macOS app (SwiftUI) for starting, stopping, resuming and forking runs of the `alife` engine, with
live status and a log. It launches `<repo>/.venv/bin/alife` directly as a subprocess; it has no third-party
dependencies and no Python code of its own.

## Requirements

- macOS 14 or later
- Xcode command line tools (`xcode-select --install`) or Xcode 15+ (Swift 5.9)
- The engine environment in the repository: `uv sync --extra viewer --extra video`
  (the app can run this for you with **Set up environment** if `.venv/bin/alife` is missing; it looks for `uv`
  in `/opt/homebrew/bin`, `/usr/local/bin`, `~/.local/bin` and `PATH`)

## Build and run

```bash
cd gui
swift run            # debug build, opens the window
swift build -c release   # optional; binary in .build/release/AlifeControl
```

Or open it in Xcode: `cd gui && open Package.swift`, select the `AlifeControl` scheme and run.

On first launch pick the repository folder (default `~/artificial-life`); the choice is remembered.

## Using it

- **Mode**: *New run* (`alife run --preset ... --out runs/NAME`), *Resume* (`alife resume runs/NAME`, optionally
  `--steps N` more steps) or *Fork* (`alife run --init-from runs/SOURCE --out runs/NAME`, starting from the source
  run's latest checkpoint). The exact command line is shown before you start.
- **Stop** sends a single SIGINT: the engine finishes the step, writes a checkpoint and exits (exit code 130).
  **Force stop** (shown after Stop) sends SIGTERM and ends it immediately, without a checkpoint.
- **Timer**: "Stop after HH:MM" sends the same single SIGINT when the time is up; the remaining time is shown.
- The Mac is kept awake (no idle sleep) while a run is active.
- **Status** is read from the last line of `runs/NAME/metrics.jsonl` every ~2 seconds: step, population,
  steps/s, clades, maximum generation and ledger error.
- **Open viewer** runs `alife view runs/NAME` as a separate process. It shares the GPU with a running
  simulation and slows it down. **Inspect** runs `alife inspect runs/NAME --check` into the log.
  **Reveal in Finder** shows the run directory.
- Only one run (or inspect / environment setup) can be active at a time from the app.

## The "keep inventing" levers

Each lever is passed as `--set section.key=value` only when it differs from the engine default. Resume cannot
change the dynamics (it accepts only `run.*` settings); use a new run or a fork to apply them.

| lever | key | default | effect |
|---|---|---|---|
| Mutation rate floor | `mutation.rate_min` | 0.001 | Lowest value the evolvable per-gene mutation rate can reach, in (0, 1). Raising it keeps lineages from freezing into near-clones. Must not exceed `rate_init` (0.05). |
| Mutation scale floor | `mutation.sigma_min` | 0.002 | Lowest value the evolvable mutation step size can reach, > 0. Must not exceed `sigma_init` (0.03). |
| Patch drift | `fields.patch_drift` | 0.005 | Drift speed of the resource patches (world units per step), >= 0. Faster drift is a proxy for a less predictable environment that keeps selecting for adaptability. |
| Immigration floor | `population.immigration_floor` | 0 (off) | When the population falls below this number, random founders ("immigrants") are added. Immigrants are added only while the population is below the floor. |
| Immigrants per step | `population.immigrants_per_step` | 64 | Maximum number of immigrants added per step while below the floor; must be >= 1 when the floor is above 0. |

The engine performs the final validation; any rejection (exit code 2) appears in the log.

## Files

- `AlifeControlApp.swift`: app entry point and app delegate
- `ContentView.swift`: the UI
- `RunController.swift`: subprocess management, log streaming, timer, keep-awake
- `Levers.swift`: run settings, levers, validation and command-line construction
- `MetricsTailer.swift`: reads the last row of `metrics.jsonl`
