# alife: notes for Claude

Start with `docs/HANDOVER.md` (current state, results, next steps). `docs/DESIGN.md` is the normative spec.
`docs/DIRECTOR.md` is the design for the ecosystem director; build steps 1 (the engine's live environment
state, `State.env`), 2 (the host-side core: vital signs, archive, rule tier, `director.jsonl`, replay, CLI;
`DESIGN.md` §26), 3 (the optional LLM tier on a local LM Studio server, `DESIGN.md` §26.10) and 4 (the GUI Director
panel and `alife llm-check`, `DIRECTOR.md` §14) are implemented, and so is the yoked `random` persona for the evaluation's arm R
(`DIRECTOR.md` §15, `director.yoke_log`); the evaluation itself has not been run. `docs/RESEARCH.md`
assesses research directions against prior work. `docs/3D.md` is the proposed 3D world (gravity and a floor;
not implemented; it starts after the 2D evaluation).

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
- Avoid GPU float32 matmul for anything that must be accurate (Metal reduced precision, DESIGN F20).
- Checks before pushing: `uv run ruff check src tests`, `uv run ruff format --check src tests`,
  `uv run pytest -m "not slow" -q`. A trajectory change needs a deliberate golden-digest re-pin.
- The GUI (`gui/`, SwiftUI, macOS 14+) shells out to `.venv/bin/alife`. Keep its lever list in sync with
  config keys, and build it with `cd gui && swift build`.

## Owner preferences

Concise and technical; no emojis or flattery. Ask when a request is ambiguous. Confirm tech choices before
committing to them. Treat the code as production-grade and explain risky changes before making them.
