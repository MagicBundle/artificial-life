# alife

A neural artificial-life engine in Python on [Apple MLX](https://github.com/ml-explore/mlx).

Agents ("dots") live on a periodic 2-D torus. Each one has its own mutable recurrent neural controller. Agents
sense, signal, move, bond into bodies, split, exert forces, harvest a regenerating resource field, share energy,
fight, reproduce with mutation, and die when their energy runs out. There is no backpropagation and no fitness
function: selection is ecological only. Energy is tracked in exact integer quanta, and a ledger closes bit-exactly
every step.

The whole simulation step is a pure, compiled MLX function on fixed shapes. It targets one Apple-silicon Mac
(Metal GPU, 64k agents at ≥ 30 steps/s), and CI runs on Linux with CPU-only MLX.

The single source of truth for the design (state schema, step pipeline, numeric defaults, RNG streams, tests and
work packages) is [`docs/DESIGN.md`](docs/DESIGN.md).

![dev preset, step 600: the whole 512-unit world](docs/media/dev_step600_world.png)
![dev preset, step 600: bonded bodies of kin, zoom 12](docs/media/dev_step600_bodies.png)

Frames from `alife render` of a `dev` run at step 600 (CPU, seed 0, float32 weights). Agents are coloured by tag, so
kin share colours. The green resource field is grazed down and tinted by the agents' chemical signals where they
gather (the dark cells). The zoomed frame shows small bonded bodies of kin.

## Quickstart (macOS, Apple silicon)

Requires Python 3.12–3.14 (mlx 0.32 ships no wheels for newer versions) and [uv](https://docs.astral.sh/uv/).
The repository pins 3.12 in `.python-version`, so `uv sync` picks (or downloads) a supported interpreter.

```bash
git clone <this repo> artificial-life && cd artificial-life
uv sync                                   # creates .venv with mlx (Metal) and numpy
uv sync --extra viewer --extra video      # optional: pygame-ce live viewer, bundled ffmpeg for videos
                                          # (a later plain `uv sync` removes the extras again)

uv run alife run --preset tiny --steps 500 --out runs/first
uv run alife resume runs/first --steps 500   # continue bitwise-identically from the latest checkpoint
uv run alife inspect runs/first --check
uv run alife render runs/first --steps 200 --every 2 --out runs/first/media/movie.mp4
uv run alife view --preset dev            # live viewer (needs the `viewer` extra)
```

## CLI overview

| command | what it does |
|---|---|
| `alife run` | create a run directory and simulate (`--preset tiny\|dev\|64k\|256k`, `--overlay fast`, `--set section.key=value`, `--init-from` to fork) |
| `alife resume RUN` | continue a run bitwise-identically from its latest (or a chosen) checkpoint |
| `alife render RUN` | re-simulate from a checkpoint and write a PNG, a video, or a PNG sequence |
| `alife inspect RUN` | config summary, carrying capacity, checkpoints, ledger status, bodies and clades; `--check` runs the full invariant check |
| `alife view` | interactive pygame-ce viewer (pan, zoom, layers, agent picking, snapshots) |
| `alife bench` | throughput, donation health, determinism and brain-variant benchmarks |

Run directories (`runs/<name>/`) hold the resolved `config.json`, `meta.json`, `metrics.jsonl`, safetensors
checkpoints, and birth/death event logs (DESIGN §18).

## Presets

| preset | agents (`n_max`) | world side | use |
|---|---|---|---|
| `tiny` | 1 024 | 128 | tests and CI |
| `dev` | 16 384 | 512 | development |
| `64k` (defaults) | 65 536 | 1 024 | the target configuration |
| `256k` | 262 144 | 2 048 | large runs |

The `fast` overlay (`K = 6`, `C = 12`, `H = 24`) combines with any preset.

## Optional environment levers

Both are off by default and leave old runs (and their config hashes) untouched:

- **Periodic immigration:** `population.immigration_every=P`, `population.immigration_count=c` add up to `c`
  random founders on every step `t` with `(t + 1) % P == 0` (combines with `immigration_floor` by `max`).
- **Seasons:** `fields.season_period=P`, `fields.season_amplitude=A` (0 to 1) and
  `fields.season_mode=global|per_patch` weight each fertility patch by `1 + A·sin(2πt/P + φ)`, so resource
  regeneration pulses in time (DESIGN §13.1).

```bash
uv run alife run --preset tiny --set population.immigration_every=100 --set population.immigration_count=8 \
    --set fields.season_period=400 --set fields.season_amplitude=0.6 --set fields.season_mode=per_patch
```

## Ecosystem director

`alife run ... --director live` (or `resume RUN --director live`) lets a rule-based director watch the vital signs
and act on the environment between steps: seasons, patch fertility, regeneration pulses, immigration, catastrophes.
It never edits agents or genomes. Its decisions go to `runs/<name>/director.jsonl` (with `journal.jsonl`), and
`--director replay` re-applies a log bitwise: `resume RUN --director replay`, or
`run --init-from SRC:STEP --director replay --replay-log LOG`. Tune it with `--set director.persona=gardener|trickster|architect`,
`director.rule_every` and `director.budget_per_100k`. The viewer and `render` ignore the director. See
`docs/DIRECTOR.md` and DESIGN §26.

### LLM tier (optional, LM Studio)

With `--llm` (every 25,000 steps) or `--llm-every N`, a local language model makes the discretionary interventions
at those boundaries; the rule tier keeps the safety moves and takes over whenever a call fails or its reply is
invalid. It needs `--director live`, and it is off unless asked for (nothing then touches the network). A replay
never calls the model.

```bash
uv run alife run --preset 64k --director live --llm --persona architect --goal "bodies of at least 100 agents" \
    --llm-model gemma-4-26b-a4b-it-mlx@8bit
```

LM Studio setup: start the local server (default `http://localhost:1234`; `--llm-url` changes it) and load the model,
ideally before the run (`lms load gemma-4-26b-a4b-it-mlx@8bit`), so the first call does not wait for the load. The
default, **`gemma-4-26b-a4b-it-mlx@8bit`**, gave valid replies in 24 of 24 probes at about 2.3 s; the 4-bit QAT build
degenerated. Reasoning models (the qwen3.6 family) return an empty reply unless **thinking is switched off** in LM
Studio, and the run then falls back to the rule tier at every boundary. Every call, with its prompt, reply, outcome and
fallback, is written to `journal.jsonl`. The model's text is data: it can only select menu actions, which pass the same
bounds, budget and cooldown checks as the rules. `ALIFE_LLM_URL=http://localhost:1234/v1/chat/completions uv run pytest
-m llm` runs the one test against a real server.

## Development and testing on Linux (CPU)

On Linux the `mlx[cpu]` wheel is installed automatically, so the full test suite runs without a GPU:

```bash
uv sync                                   # includes the dev group (pytest, ruff)
uv run ruff check && uv run ruff format --check
uv run pytest -m "not slow"               # the fast suite (< 3 min)
uv run pytest -m slow                     # nightly: long invariant runs and the ecology test
```

Trajectories are deterministic per backend (CPU and Metal differ), but every invariant, including exact ledger
closure, holds on both.

## Layout

```text
src/alife/      config, state, rng, ops (the only scatters), spatial, fields, sensing, brain, actions,
                bonds, physics, energy, lifecycle, step, invariants, metrics, runio, runner, render,
                viewer, bench, cli
src/alife/presets/   tiny, dev, 64k, 256k and the fast overlay (TOML)
tests/          pytest suite (DESIGN §22)
docs/DESIGN.md  the design document
```
