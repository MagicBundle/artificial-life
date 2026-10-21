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
`docs/GAMES.md` is the design of the gaming vertical (owner decisions in its §10); its build step G1, the observer
layer (`game.observe`, `src/alife/game/`, `python -m alife.analysis.labels`; normative spec `DESIGN.md` §28), is
implemented, with `INTENT_VERSION` 2 (`RULES_V2`, the default, calibrated offline on the G1 study windows; it
passed the provisional criterion on three held-out windows on the GPU, 2026-10-21, DESIGN §28.4, §28.8; `RULES_V1` stays defined with its G1 digest); its build step G2, social
tracking (houses, kin groups, colonies, territory, relations, and a roster of kin-group names, house events and
per-group window counts; generic `groups.py` and `tracking.py`; normative spec `DESIGN.md` §29), is implemented, with
provisional thresholds (`SOCIAL_VERSION` 1, measured on the GPU, none changed); the Settler colony clause is on in `RULES_V2` (`settler_body` 4,
DESIGN §28.4). The fused Metal neighbour
search (`src/alife/knn_metal.py`, `DESIGN.md` §4, row I48) is built: 75 steps/s at `64k3d` instead of 15, bitwise the
same trajectory.
`docs/STORMS.md` is the storm world; W0 (the model, `weather.py`, `docs/STORMS_W0.md`) and W1 (`weather.*` config,
`State.wx`, wind/rain/hazard in the step; `DESIGN.md` §30) are implemented and off by default.

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
- Checks before pushing: `uv run ruff check src tests`, `uv run ruff format --check src tests`,
  `uv run pytest -m "not slow" -q`. A trajectory change needs a deliberate golden-digest re-pin.
- The GUI (`gui/`, SwiftUI, macOS 14+) shells out to `.venv/bin/alife`. Keep its lever list in sync with
  config keys, and build it with `cd gui && swift build`.

## Owner preferences

Concise and technical; no emojis or flattery. Ask when a request is ambiguous. Confirm tech choices before
committing to them. Treat the code as production-grade and explain risky changes before making them.
