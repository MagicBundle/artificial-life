"""The voice channel's analysis (DESIGN §39.14, PR C3): a log of what is said and heard, and the meaning test.

``python -m alife.analysis.voice log RUN`` replays a window of a run with the observers on and writes compact
tables (what each lineage says, in what situation, and what hearers did next) plus a readable summary.
``python -m alife.analysis.voice meaning`` decides, from forks of one checkpoint with ``comm.control`` set to
``none``, ``deaf`` and ``decoy``, whether the calls carry a language in the sense of §39.9. Both are
host-side and observer-only: no engine file, state, key or default changes.

Modules: ``spec`` (the constants and the pre-registration), ``tables`` (snapshots to tables, numpy),
``measures`` (information measures and their nulls, numpy), ``replay`` (the only module that touches MLX),
``summary`` (the log's report), ``meaning`` (the arms, the verdict, the report) and ``__main__`` (the CLI).
"""

from __future__ import annotations

VOICE_ANALYSIS_VERSION = 1
