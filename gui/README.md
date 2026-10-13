# Alife Control

A small native macOS app (SwiftUI) for starting, stopping, resuming and forking runs of the `alife` engine, with
live status and a log. It launches `<repo>/.venv/bin/alife` directly as a subprocess; it has no third-party
dependencies and no Python code of its own.

## Requirements

- macOS 14 or later
- A Swift 6 toolchain: Xcode 16 or later, or the command line tools that ship with it
  (`xcode-select --install`). The package builds in Swift 6 language mode (strict concurrency).
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

## Install as an app

```bash
gui/scripts/make-app.sh --install     # builds, bundles, signs and copies to /Applications/AlifeControl.app
gui/scripts/make-app.sh               # bundle only: gui/.build/AlifeControl.app
gui/scripts/make-app.sh --help
```

The script builds the release binary, generates the icon (`scripts/make-icon.swift`), writes `Info.plist`
(bundle ID `io.github.magicbundle.alifecontrol`, overridable with `BUNDLE_ID`; version from `VERSION`, build
number from the commit count) and signs the bundle with the hardened runtime. The signing identity is, in order:
`--sign "<identity>"`, the first "Developer ID Application" identity, the first "Apple Development" identity,
then ad-hoc. `--install-dir DIR` installs somewhere else (for example `~/Applications`). The script never uses
sudo and refuses to install over a running copy, so quit the app first (a run may be active).

Ad-hoc and Apple Development signing are for this Mac only. Sharing the app with others needs a Developer ID
certificate plus notarization (`notarytool` and `stapler`), which is not done yet. Re-run the script after any
GUI change to refresh the installed copy.

The installed app is only the front end: it still uses the repository's `.venv` at the configured repo path
(default `~/artificial-life`). The setting is stored under the bundle's own preferences domain, so it is
separate from the one used by `swift run`.

## Using it

- **Mode**: *New run* (`alife run --preset ... --out runs/NAME`), *Resume* (`alife resume runs/NAME`, optionally
  `--steps N` more steps) or *Fork* (`alife run --init-from runs/SOURCE --out runs/NAME`, starting from the source
  run's latest checkpoint; *Steps* is the number of steps to run from that checkpoint). In Fork mode the levers are
  pre-filled from the source run's `config.json` and only the levers you change are passed, so a lever can also be
  set back to its engine default. If the source config cannot be read, the levers show the engine defaults and a
  warning appears. The exact command line is shown before you start.
- **Run name** (new run, fork): if `runs/NAME` already exists, the next free name is used instead of blocking Start:
  a short trailing number counts up (`trickster2` → `trickster3`), otherwise `-2`, `-3`, ... is appended. The
  command preview shows the name that will be used, and on Start the name field switches to it.
- **Stop** sends a single SIGINT: the engine finishes the step, writes a checkpoint and exits (exit code 130).
  **Force stop** (shown after Stop) sends SIGTERM and ends it immediately, without a checkpoint.
- **Timer**: "Stop after HH:MM" sends the same single SIGINT when the time is up; the remaining time is shown.
- The Mac is kept awake (no idle sleep) while a run is active.
- **Director**: see the section below. New run and Fork can start a live director (optionally with the local LLM tier);
  Resume can switch it off, live or to an in-place replay.
- **Status** is read from the last line of `runs/NAME/metrics.jsonl` every ~2 seconds: step, population,
  steps/s, clades, maximum generation and ledger error.
- **Open viewer** runs `alife view runs/NAME` as a separate process. It shares the GPU with a running
  simulation and slows it down. **Inspect** runs `alife inspect runs/NAME --check` into the log.
  **Reveal in Finder** shows the run directory.
- Only one run (or inspect / environment setup) can be active at a time from the app.

## The "keep inventing" levers

Each lever is passed as `--set section.key=value` only when it differs from the baseline: the engine default for a
new run, the source run's value for a fork. Resume cannot change the dynamics (it accepts only `run.*` settings and the director's policy keys);
use a new run or a fork to apply them.

| lever | key | default | effect |
|---|---|---|---|
| Mutation rate floor | `mutation.rate_min` | 0.001 | Lowest value the evolvable per-gene mutation rate can reach, in (0, 1). Raising it keeps lineages from freezing into near-clones. Must not exceed `rate_init` (0.05). |
| Mutation scale floor | `mutation.sigma_min` | 0.002 | Lowest value the evolvable mutation step size can reach, > 0. Must not exceed `sigma_init` (0.03). |
| Patch drift | `fields.patch_drift` | 0.005 | Drift speed of the resource patches (world units per step), >= 0. Faster drift is a proxy for a less predictable environment that keeps selecting for adaptability. |
| Immigration floor | `population.immigration_floor` | 0 (off) | When the population falls below this number, random founders ("immigrants") are added. Immigrants are added only while the population is below the floor. |
| Immigrants per step | `population.immigrants_per_step` | 64 | Maximum number of immigrants added per step while below the floor; must be >= 1 when the floor is above 0. |
| Immigration every | `population.immigration_every` | 0 (off) | Add random founders every N steps, whatever the population size. |
| Immigrants per event | `population.immigration_count` | 0 | Founders added per periodic event (1 to max births per step); required when the period is above 0. |
| Season period | `fields.season_period` | 0 (off) | Length of one resource-regrowth season cycle, in steps. |
| Season amplitude | `fields.season_amplitude` | 0 | Seasonal swing in regrowth, 0 to 1 (1 = regrowth reaches zero at the trough). |
| Season mode | `fields.season_mode` | global | `per_patch` gives each resource patch its own phase, so good seasons move across the map. |

The engine performs the final validation; any rejection (exit code 2) appears in the log.

## The Director panel

The **Director** box (settings column) runs the ecosystem director (`docs/DIRECTOR.md`, DESIGN §26) from the app; the
**Director** views (right column) follow it. As with the levers, only values that differ from a baseline are passed:
the engine defaults for a new run, the source run's `config.json` (`director.*`, `run.llm_url`, `run.llm_model`,
`run.director`) for a fork and a resume. The Command box shows the flags.

| setting | CLI | notes |
|---|---|---|
| Mode | `--director live\|replay` | New run and Fork: Off or Live. Resume: Off, Live or Replay (in place, the run's own `director.jsonl`; the model is never called). Replay forks with `--replay-log` stay CLI-only. |
| Persona | `--persona gardener\|trickster\|architect\|random` | Live only. A menu. Random is arm R of the evaluation (`docs/DIRECTOR.md` §15): it hides the LLM controls (forced off, `--llm-every 0` if the source had it on). |
| Yoked to run | `--yoke-log=runs/NAME/director.jsonl` | Random only, and required: a picker of the runs that have a `director.jsonl` (the run being resumed is left out). The random director makes as many moves as that run's director did, at the same steps; run that one first. Leaving Random passes `--yoke-log=` to clear it. |
| Goal | `--goal=TEXT` | Architect only: the rule tier applies the safety rules only; the goal reaches the model as data. |
| Rule every | `--set director.rule_every=N` | Default 1000. A positive multiple of `run.metrics_every` and `run.heavy_metrics_every` (read from the preset file, or from the source's config). |
| LLM tier | `--llm-every N` | Off is `0`. When on, N defaults to 25000 and must be a positive multiple of Rule every. A failed call falls back to the rule tier. |
| URL, Model | `--llm-url=URL`, `--llm-model=NAME` | Only with the tier on. URL must be http(s); default `http://localhost:1234/v1/chat/completions` and `gemma-4-26b-a4b-it-mlx@8bit` (LM Studio). |

- **Fork** does not inherit the director: the engine starts it off and resets the environment levers, so the Mode
  picker is pre-filled with Live when the source ran live and `--director live` is passed. The other settings are
  pre-filled from the source and only changed ones are passed.
- **Resume** pre-fills the run's own mode and settings. If the run's `config.json` has `director.enabled = false` (it
  was created without the director), the box says "This run was created without the director; fork it to add one." and
  nothing is passed. Replay is refused when the run has no `director.jsonl`.
- **Test connection** runs `alife llm-check --llm-url=... --llm-model=...` (one tiny structured-output request under the
  director's own client and schema) and shows its line in green (`OK model=... latency=...s`) or red
  (`FAIL connection: ...`, `FAIL timeout: ...`, `FAIL empty: ...` for a reasoning model with thinking on, `FAIL invalid:
  ...`). It can take up to the 60 s timeout, and works while no run is active. LM Studio must be running with the model
  loaded; it may answer for another model if the name is unknown, so the check shows reachability and a valid reply.
- **Start** is disabled while a director setting is invalid (the messages appear under the command, like the levers).
  A note, not an error, appears if the metrics cadence could not be read: the engine then does the check (exit code 2).
- **Interventions** shows the last 10 entries of `director.jsonl`, newest first: step, source (gear = rule, sparkles =
  llm, arrow = revert), action, its params (floats at three significant digits) and the reason (full text on hover).
- **Journal** shows the last 20 records of `journal.jsonl`, newest first: rule one-liners, and for each LLM call the
  step, outcome (ok, none, rejected, failed, invalid), latency, accepted and rejected counts, any fallback to the rule
  tier, the model's own journal text and, under "Raw prompt and response", the full request and reply as selectable text.
- Both views follow the same run as the status box (the running run, else the selected one), poll every 2 s, read only
  the last 256 KiB of each file, and skip a missing file, a half-written last line and malformed lines. A run without a
  `director.jsonl` shows "Director off for this run".

## Files

`Package.swift` has three targets: **`AlifeCore`**, a library of pure logic with no AppKit and no Metal (its declarations
have `package` access: visible to the app and the tests, not an API); **`AlifeControl`**, the app; and **`AlifeCoreTests`**,
the tests (Swift Testing).

`Sources/AlifeCore/`:
- `Levers.swift`: run settings, levers, validation and command-line construction
- `World3DLevers.swift`: the 3D world's levers (height, gravity, buoyancy, the vertical food profile); a new run diffs them
  against the preset's values, a fork against its source's, so the 2D command lines do not change
- `PresetFile.swift`: the scalar (integer and number) settings of a preset file `src/alife/presets/NAME.toml`
- `DirectorSettings.swift`: the director's settings, baselines from `config.json`, validation, argument construction
  and the `alife llm-check` runner (pure logic apart from that runner)
- `MetricsTailer.swift`: reads the last row of `metrics.jsonl`
- `StreamProtocol.swift`: the incremental decoder of the `alife stream` binary protocol (DESIGN §27.11)
- `StreamControl.swift`: the encoder of the control lines written to the stream's stdin
- `Camera.swift`: the orbit and fly camera (view and projection matrices)

`Sources/AlifeControl/`:
- `AlifeControlApp.swift`: app entry point and app delegate
- `ContentView.swift`: the UI
- `RunController.swift`: subprocess management (main-actor isolated; pipe output arrives through ordered
  AsyncStreams), log streaming, timer, keep-awake
- `DirectorPanel.swift`: the Director settings box and the Interventions / Journal views
- `DirectorTailer.swift`: tails `director.jsonl` and `journal.jsonl` (last 256 KiB) into interventions and journal rows

Also `scripts/make-app.sh` and `scripts/make-icon.swift` (app bundle, signing, install and the generated icon).

## Tests

```bash
cd gui
swift build -Xswiftc -warnings-as-errors     # the build the checks use
swift test
```

The tests need no GPU and start no process. They cover the command lines the GUI builds for 2D runs (a table captured
before the 3D work, so that a 2D run, fork or resume keeps its arguments byte for byte), the 3D lever rules, the preset
reader, the stream decoder (the committed fixtures `tests/data/stream_tiny_{2d,3d}.bin` against the sidecars written by
the independent Python decoder, any chunking of the input, every refusal), the control-line encoder and the camera.
`tests/test_gui_sync.py` (Python, in the engine's suite) checks that every `--set` key and preset name in the Swift
sources exists in the engine's `Config`.
