"""The communication scorecard of the emergence suite: the language-score adapter (DESIGN §39.14 and §44.6,
E6).

The measurement lives in ``alife.analysis.voice`` (the voice log and the meaning test). This module is the
~60 lines that map a logged window of one run onto ``scorecards.ScorecardResult``: it does not simulate and
does
not run the arms. ``available`` asks for a voice log (``python -m alife.analysis.voice log RUN --from T0
--steps
N --out analysis_cache/voice/<run>``) of the requested window; ``compute`` reads
its tables and returns the pre-registered primary ``transfer_bits``, the excess over the shuffle null, in
bits, of the best (caller context, hearer response) pair, with the code and reception readings beside it.

The label is ``signalling`` when the calls carry information (the code or the transfer test passes its
thresholds), else ``no signal``. ``language`` is never given from one run: it needs the arm contrast and the
fitness clause, which are the verdict of ``python -m alife.analysis.voice meaning`` over forks with
``comm.control`` ``none``, ``deaf`` and ``decoy``. When the panel is given control arms that have a voice log,
``transfer_over_deaf`` and ``transfer_over_decoy`` report the same pair in them (same cut points) and the
difference.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from alife.analysis.emergence.scorecards import Criterion, ScorecardResult, not_evaluated

if TYPE_CHECKING:
    from alife.analysis.emergence.frames import Frames
    from alife.runio import RunDir

COMM_SCORECARD_VERSION: Final = "1"


LOG_ROOT: Final = Path("analysis_cache/voice")


def _log_dir(run: Any) -> Path | None:
    """``analysis_cache/voice/<run directory name>``, where ``voice log --out`` is pointed, if it has
    tables."""
    name = Path(getattr(run, "path", "")).name
    if name and (LOG_ROOT / name / "tables.npz").is_file():
        return LOG_ROOT / name
    return None


class CommScorecard:
    name = "comm"

    def available(self, run: RunDir, window: tuple[int, int] | None = None) -> list[str]:
        cfg_of = getattr(run, "config", None)
        if cfg_of is None:
            return ["a run directory"]
        if not cfg_of().comm.enabled:
            return ["comm.enabled is false in this run (there is no voice to score)"]
        d = _log_dir(run)
        if d is None:
            return [
                "a voice log of this run (python -m alife.analysis.voice log RUN --from T0 --steps N --out "
                "analysis_cache/voice/<run>)"
            ]
        if window is not None and window != (0, 0):
            from alife.analysis.voice.tables import Tables

            have = tuple(Tables.load(d / "tables.npz").meta["window"])
            if have != tuple(window):
                return [f"a voice log of the window {tuple(window)} (the logged window is {have})"]
        return []

    def compute(
        self,
        run: RunDir,
        window: tuple[int, int],
        frames: Frames | None,
        controls: Mapping[str, RunDir],
        out: Path,
    ) -> ScorecardResult:
        need = self.available(run, window)
        if need:
            return not_evaluated("comm", need, COMM_SCORECARD_VERSION)
        from alife.analysis.voice import analyse, spec
        from alife.analysis.voice.tables import Tables

        d = _log_dir(run)
        assert d is not None
        tab = Tables.load(d / "tables.npz")
        win = tuple(tab.meta["window"])
        p = analyse.prepare(tab, win)  # type: ignore[arg-type]
        rng = np.random.default_rng(spec.SEED)
        fam = {f: analyse.run_family(p, f, spec.N_PERM, rng) for f in ("code", "reception", "transfer")}
        best = {f: (r.best_summary() if r else None) for f, r in fam.items()}

        def val(f: str, key: str = "excess") -> float | None:
            return None if best[f] is None else best[f][key]

        endpoint: dict[str, float | None] = {
            "primary": val("transfer"),
            "transfer_bits": val("transfer"),
            "code_bits": val("code"),
            "reception_bits": val("reception"),
        }
        pair = best["transfer"]["pair"] if best["transfer"] else None
        for ctrl in ("deaf", "decoy"):
            cd = _log_dir(controls[ctrl]) if ctrl in controls else None
            if pair is None or cd is None:
                continue
            ct = Tables.load(cd / "tables.npz")
            pc = analyse.prepare(
                ct,
                tuple(ct.meta["window"]),  # type: ignore[arg-type]
                edges=p.edges,
                sym_thr=p.sym_thr,
                heard_thr=p.heard_thr,
                dist_edges=p.dist_edges,
            )
            r = analyse.run_family(pc, "transfer", spec.N_PERM, np.random.default_rng(spec.SEED))
            names = r.names if r else []
            e = float(r.excess[names.index(pair)]) if r and pair in names else None
            endpoint[f"transfer_{ctrl}_bits"] = e
            endpoint[f"transfer_over_{ctrl}"] = (
                None if e is None or endpoint["transfer_bits"] is None else endpoint["transfer_bits"] - e
            )  # type: ignore[operator]
        crit = []
        for f, key in (("code", "code"), ("transfer", "transfer"), ("reception", "reception")):
            thr = spec.THRESH[key]
            b = best[f]
            ok = (
                None
                if b is None
                else bool(
                    b["z"] is not None
                    and b["z"] >= thr["z"]
                    and b["excess"] >= thr["bits"]
                    and b["p_family"] <= thr["p"]
                )
            )
            crit.append(
                Criterion(
                    f"{f} excess >= {thr['bits']} bits, z >= {thr['z']}",
                    None if b is None else b["excess"],
                    thr["bits"],
                    ok,
                    "" if b is None else b["pair"],
                )
            )
        signalling = any(c.passed for c in crit[:2])
        return ScorecardResult(
            feature="comm",
            version=COMM_SCORECARD_VERSION,
            level=None,
            label="signalling" if signalling else "no signal",
            endpoint=endpoint,
            criteria=crit,
            provisional=True,
            needs_missing=[],
        )
