# alife: notes for Claude

Start with `docs/HANDOVER.md` (current state, results, next steps). `docs/DESIGN.md` is the normative spec.
`docs/DIRECTOR.md` is the design for the ecosystem director; build steps 1 (the engine's live environment
state, `State.env`), 2 (the host-side core: vital signs, archive, rule tier, `director.jsonl`, replay, CLI;
`DESIGN.md` §26), 3 (the optional LLM tier on a local LM Studio server, `DESIGN.md` §26.10) and 4 (the GUI Director
panel and `alife llm-check`, `DIRECTOR.md` §14) are implemented, and so is the yoked `random` persona for the evaluation's arm R
(`DIRECTOR.md` §15, `director.yoke_log`); the evaluation has been run and written up in `docs/EVALUATION.md`.
`docs/RESEARCH.md`
assesses research directions against prior work. `docs/3D.md` is the 3D world (gravity and a floor), built in PRs
#15 to #20 (normative spec: `DESIGN.md` §27); its first long run is described in `docs/RESULTS_3D.md`, and a second
seed, a half-food control and a director run in `docs/RESULTS_3D_RUNS.md`.
`docs/DESIGN.md` §37 is the mini world (preset `mini3d`, overlay `storm160`, a 160-wide 3D world of about 10,500 organisms for quick
play-tests; no simulation change); the launcher passes `storm160` for a 160-wide source and `storm` otherwise.
`docs/GAMES.md` is the design of the gaming vertical (owner decisions in its §10); its build step G1, the observer
layer (`game.observe`, `src/alife/game/`, `python -m alife.analysis.labels`; normative spec `DESIGN.md` §28), is
implemented, with `INTENT_VERSION` 2 (`RULES_V2`, the default until W2a made `RULES_V3` the default, calibrated offline on the G1 study windows; it
passed the provisional criterion on three held-out windows on the GPU, 2026-10-21, DESIGN §28.4, §28.8; `RULES_V1` stays defined with its G1 digest); its build step G2, social
tracking (houses, kin groups, colonies, territory, relations, and a roster of kin-group names, house events and
per-group window counts; generic `groups.py` and `tracking.py`; normative spec `DESIGN.md` §29), is implemented, with
provisional thresholds (`SOCIAL_VERSION` 1, measured on the GPU, none changed); the Settler colony clause is on in `RULES_V2` (`settler_body` 4,
DESIGN §28.4). The fused Metal neighbour
search (`src/alife/knn_metal.py`, `DESIGN.md` §4, row I48) is built: 75 steps/s at `64k3d` instead of 15, bitwise the
same trajectory. It is the default on Metal in 2D and 3D (row I52); `ALIFE_KNN=mlx` forces the pure-MLX path.
`docs/STORMS.md` is the storm world; W0 (the model, `weather.py`, `docs/STORMS_W0.md`) and W1 (`weather.*` config,
`State.wx`, wind/rain/hazard in the step; `DESIGN.md` §30) are implemented and off by default.
The World window's viewer v2 (legibility, the one-world flat map, the pose and input, the minimap, compass, scale bar and height colour; `gui/`, client only) is `docs/DESIGN.md` §36 and Review-log row I60 (A0 and A1; the globe is PR B); the keys and gestures are in `gui/README.md`.
`docs/GAMES.md` §9 N1, the offline World Report (`python -m alife.analysis.world_report build | verify | probe`,
`src/alife/game/report.py`, `templates.py`, `lexicon.py`; normative spec `DESIGN.md` §31), is implemented: code writes every
sentence from templates and checks it, a local model (`--llm --model ID`) only selects and orders candidate ids, and no
model-written text is ever shown or put into a prompt. Its thresholds are provisional until the owner read.
W2a (`DESIGN.md` §32: eight `weather` keys `seed_gain`, `seed_c0`, `shade_gain`, `shade_n0`, `wind_scale`, `rain_zonal`, `storm_eddy`,
`wind_eddy_force`; the tracer rig `game.tracers`; `run.eco_every`, `run.wx_every`; `RULES_V3` with `INTENT_VERSION` 3, the default
observer rules; `python -m alife.analysis.w2`) is implemented and off by default; the calibration and the 14 runs (W2b) are not run, and
`presets/storm.toml` still holds W1's provisional values.
`alife play` and the Swift **World** window (the play loop, the stream records, planets, the chronicle, picking and follow; GAMES §9 G3a;
normative spec `DESIGN.md` §33, row I51) are built; their Mac day ran 2026-10-22 (purity on Metal, the bench gate: passed; `docs/RESULTS_G3A.md`); the click-through by hand is the owner's.
Presentation-only control lines (`focus follow idnt labels color every`, pacing, frames, records) never change a trajectory; a change to
`play.py`, `streamrec.py` or the runner's `Pacer` hook must keep `tests/test_play_purity.py` (digest equality with a plain run) green.
**G3b** (chapters, alerts, the live World Report and direct powers; GAMES §9; normative spec `DESIGN.md` §34, row I54; built; the Mac day ran 2026-10-23, the gate passes except the report-time item the bench cannot read, `docs/RESULTS_G3B.md`; the click-through and the first play by a person are open; it found two defects: the server's report chapter number counted from step 0, fixed in the commit `G3b: report chapters count from the planet origin`, and the bench's report time, not fixed yet; the owner decided on 2026-10-23 that a population fall of 15% or more pauses whatever the cause, including after the player's own power, that kin groups budding off a large group count as new groups, and that storm namings stay on the alert surface; none changed behaviour): 6,000-step chapters that end in a pause with a report, alerts
the world raises (a stack of what the player can act on), a template report built on a worker thread, and four direct powers (Food, Bloom, Blight, Strike) through a player channel in the runner (`Director.apply_pending`, rule P: a player's record at step `s` is after the checkpoint at `s`).
**A power is the only line that changes a trajectory**: it is a logged `source:"player"` entry of `director.jsonl`, so a played world replays bitwise by `resume --director replay` and `play --replay`; `autopause watch unwatch ack`, `power preview`, a refused `power` and every window element are
presentation and must leave a plain run's digest unchanged (`tests/test_play_purity.py`, `test_player_channel.py`). The engine edit is inert when the channel is unused (`tests/director_identity.py`, the 2D digest matrix against the previous commit). Keep the Swift keys, alert classes and power names in step with
`tests/test_gui_sync.py`; `cd gui && swift build -Xswiftc -warnings-as-errors && swift build --build-tests -Xswiftc -warnings-as-errors && swift test`.
The stream's labels are `INTENT_VERSION` 3's (ten archetypes, the storm names) and carry N1's held-out verdict gate: a label is shown as valid only
with a `LABEL_VERDICTS` row for the world and rule set (none for V2 or V3 yet) and the environment at its defaults (`DESIGN.md` §33.7).

## Conventions

- Change DESIGN.md in the same change as any behaviour, signature or default it pins, and add a Review-log row.
- Pure-MLX simulation code: no host syncs, no Python branching on array values, and fixed shapes inside the
  compiled step. Branch only on config constants.
- Scatters and setitem live only in `src/alife/ops.py` (plus uint32 compositing in `render.py`). `test_lint`
  enforces this.
- Energy is integer quanta with an exact 16-term ledger. Every new energy path needs a ledger term and a test
  that the ledger closes.
- New config keys must default to "off" and be added to `config.POST_V01_KEYS`, so existing runs keep their
  config hash and still resume.
- Observer-only keys (`config.OBSERVER_KEYS`), which are unhashed at every value and inert while `game.observe` is off,
  may default to a working value.
- Avoid GPU float32 matmul for anything that must be accurate (Metal reduced precision, DESIGN F20).
- A trajectory-neutral state field is frozen out of the legacy digests (`V0_FIELDS`, `V0_FIELDS_3D`), so the goldens
  do not move when such a field is added.
- Custom Metal kernels live only in `src/alife/knn_metal.py` (`mx.fast.metal_kernel`, lint rule 6). Each must be
  bitwise equal to the pure-MLX path it replaces (`tests/test_spatial_metal.py`, run with `-m ""` on the Mac and on
  every MLX bump, which is then added to `knn_metal.VERIFIED_MLX`), fall back to that path off Metal, and write no
  contested outputs. `ALIFE_KNN=auto|mlx|metal` selects the neighbour search (DESIGN §4, I48).
- Checks before pushing: `uv run ruff check`, `uv run ruff format --check` (the whole repo, as CI runs them; scripts
  under `docs/` count), `uv run pytest -m "not slow" -q -n 6 --dist loadfile` (the fast suite in parallel: about 7
  minutes on the M5 Pro against 27 serial; with `-p cpu_device` and `PYTHONPATH` for the CPU device as before; plain
  serial `uv run pytest -m "not slow" -q` still works). `-n 6` while LM Studio holds its model, `-n 8` only with more
  than 15 GB free (`vm_stat`), never `-n auto` (18 workers: slower and 16 GB; DESIGN §22.0, I58). `--dist loadfile`
  is the fast form, no longer needed for correctness. Every pytest process compiles MLX CPU kernels into a private cache
  (`tests/worker_cache.py`), so parallel runs and concurrent sessions do not collide on `$TMPDIR/mlx`; keep one
  `TMPDIR` per agent as the warm cache. Ambiguous kernels (F21 item 6) are never seeded from or handed back to it
  (`tests/kernel_ambiguity.py`), and the two Linux golden tests run in a child on an empty cache
  (`tests/golden_run.py`). A test that fails only in parallel is a defect of the test (relative bounds, injected
  clocks, no shared paths). `tmp_path_retention_policy = "failed"` keeps only failed tests' directories: a passing
  run leaves nothing in `pytest-of-*` under its `TMPDIR` and a run with a failure about 2.4 GB (it was 38 GB a run, and
  the disk filled once); delete a failed run's directory after reading it. A trajectory change needs a deliberate golden-digest re-pin.
- The world report (DESIGN §31) never calls an LM Studio management endpoint (no load, unload or switch) and sends `model`
  explicitly; its only network calls are the chat POST and the read-only `GET /api/v0/models` preflight. Run `--llm` and
  `probe` only when no GPU job or timing runs. A change to a template, the lexicon, the label table or the select prompt
  changes a digest that `verify` compares: bump the version and re-pin.
- The GUI (`gui/`, SwiftUI, macOS 14+) shells out to `.venv/bin/alife`. Keep its lever list in sync with
  config keys, and build it with `cd gui && swift build`. Swift is not in CI: before a push run `swift build -Xswiftc
  -warnings-as-errors && swift test` there (`tests/test_gui_sync.py` ties its flags, palettes, event kinds and levels to the engine).

## Owner preferences

Concise and technical; no emojis or flattery. Ask when a request is ambiguous. Confirm tech choices before
committing to them. Treat the code as production-grade and explain risky changes before making them.
