"""The communication scorecard of the emergence suite: the language-score adapter (DESIGN §39.14 and §44.6,
E6).

The measurement lives in ``alife.analysis.voice`` (the voice log and the meaning test). This module maps a
logged
window of one run onto ``scorecards.ScorecardResult``; it does not simulate and does not run the arms.
``available`` asks for a voice log of the requested window (``python -m alife.analysis.voice log RUN --from T0
--steps N --out <root>/<run>``) under the log root: ``$ALIFE_VOICE_LOGS`` if set, else
``analysis_cache/voice``
in the repository, whatever the working directory. The log must belong to the run (its meta names the run's
path).

``transfer_bits`` is the pre-registered primary: the excess over the shuffle null, in bits, of the transfer
pair
that has the largest raw information on the **even** blocks, estimated on the **odd** blocks (the pair is not
chosen on the data that estimate it). It contains the shared environment as well as the calls, so it is a
number
to compare with the inert twin's, not a verdict. ``code_bits`` (what the call says about the caller) and
``reception_bits`` are reported beside it.

The label is ``signalling, meaning not tested`` unless the panel gives a ``decoy`` control arm that has a
log of
the same window. Then the none-minus-decoy contrast of the pair chosen on the even blocks by that contrast is
estimated on the odd blocks, and the one-sided 95% bounds over the blocks of this one seed give ``signalling
with
trace meaning`` (lower bound above 0), ``signalling, no detectable meaning (bound X)`` (upper bound below the
equivalence bound) or ``no evidence``. ``language`` is never given from one run: it needs the seeds, the
fitness
clause and the cost gate, which are the verdict of ``python -m alife.analysis.voice meaning``.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from alife.analysis.emergence.scorecards import Criterion, ScorecardResult, not_evaluated

if TYPE_CHECKING:
    from alife.analysis.emergence.frames import Frames
    from alife.runio import RunDir

COMM_SCORECARD_VERSION: Final = "3"
REPO_ROOT: Final = Path(__file__).resolve().parents[3]


def log_root() -> Path:
    """The parent of the voice logs: ``$ALIFE_VOICE_LOGS``, else ``<repository>/analysis_cache/voice``."""
    env = os.environ.get("ALIFE_VOICE_LOGS")
    return Path(env) if env else REPO_ROOT / "analysis_cache" / "voice"


def _log_dir(run: Any) -> Path | None:
    """``<log root>/<run directory name>`` if it holds tables of exactly this run (the meta names its
    path)."""
    from alife.analysis.voice.tables import Tables

    path = Path(getattr(run, "path", ""))
    if not path.name:
        return None
    d = log_root() / path.name
    if not (d / "tables.npz").is_file():
        return None
    try:
        same = Path(Tables.load_meta(d / "tables.npz").get("run", "")).resolve() == path.resolve()
    except (OSError, ValueError, KeyError):
        return None
    return d if same else None


def _odd_mean_and_bounds(values: np.ndarray, alpha: float) -> dict[str, float | None]:
    from alife.analysis.voice import measures as ms

    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return {"mean": float(v.mean()) if v.size else None, "lower": None, "upper": None}
    half = ms.t_upper(alpha, v.size - 1) * float(v.std(ddof=1)) / math.sqrt(v.size)
    return {"mean": float(v.mean()), "lower": float(v.mean() - half), "upper": float(v.mean() + half)}


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
                f"a voice log of this run under {log_root()}/<run directory name> (python -m "
                "alife.analysis.voice log RUN --from T0 --steps N --out ...)"
            ]
        if window is not None and window != (0, 0):
            from alife.analysis.voice.tables import Tables

            have = tuple(Tables.load_meta(d / "tables.npz")["window"])
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
        speaking = float((p.s_sym > 0).mean())
        fam = {f: analyse.run_family(p, f, spec.N_PERM, rng) for f in ("code", "reception")}
        endpoint: dict[str, float | None] = {
            "primary": None,
            "transfer_bits": None,
            "code_bits": None if fam["code"] is None else float(fam["code"].excess[fam["code"].best]),
            "reception_bits": None
            if fam["reception"] is None
            else float(fam["reception"].excess[fam["reception"].best]),
            "speaking_share": speaking,
        }
        # the transfer pair: chosen on the even blocks (largest raw information), estimated on the odd blocks
        info = analyse.block_information(p, "transfer", parity=0)
        score = {k: float(np.nanmean(v)) for k, v in info.items() if np.isfinite(v).any()}
        pair = max(score, key=lambda k: score[k]) if score else None
        if pair is not None:
            odd = analyse.pair_blocks(p, "transfer", pair, spec.N_PERM_BLOCK, rng, parity=1)
            endpoint["transfer_bits"] = endpoint["primary"] = float(np.nanmean(odd))
        # the call-specific part, when the panel brings a decoy arm with a log of the same window
        label = "signalling, meaning not tested (no decoy log)"
        crit = [
            Criterion(
                f"speaking (>= {spec.SPEAKING_MIN} of the rows)",
                speaking,
                spec.SPEAKING_MIN,
                speaking >= spec.SPEAKING_MIN,
            )
        ]
        cd = _log_dir(controls["decoy"]) if "decoy" in controls else None
        note = ""
        if cd is not None and tuple(Tables.load_meta(cd / "tables.npz")["window"]) != win:
            cd, note = None, " (the decoy log is of another window)"
        if cd is not None:
            ct = Tables.load(cd / "tables.npz")
            pc = analyse.prepare(
                ct,
                win,  # type: ignore[arg-type]
                edges=p.edges,
                sym_thr=p.sym_thr,
                heard_thr=p.heard_thr,
                dist_edges=p.dist_edges,
            )
            a = analyse.block_information(p, "transfer", parity=0)
            b = analyse.block_information(pc, "transfer", parity=0)
            sc = {
                k: float(np.nanmean(a[k] - b[k]))
                for k in a.keys() & b.keys()
                if np.isfinite(a[k] - b[k]).any()
            }
            cpair = max(sc, key=lambda k: sc[k]) if sc else None
            if cpair is not None:
                x = analyse.pair_blocks(p, "transfer", cpair, spec.N_PERM_BLOCK, rng, parity=1)
                y = analyse.pair_blocks(pc, "transfer", cpair, spec.N_PERM_BLOCK, rng, parity=1)
                bd = _odd_mean_and_bounds((x - y)[1::2], spec.ALPHA)
                endpoint["transfer_over_decoy"] = bd["mean"]
                if bd["lower"] is not None and bd["upper"] is not None:
                    positive = bd["lower"] > 0
                    if speaking < spec.SPEAKING_MIN:
                        label = "no evidence"
                    elif positive:
                        label = "signalling with trace meaning"
                    elif bd["upper"] < spec.EQUIVALENCE:
                        label = f"signalling, no detectable meaning (bound {bd['upper']:.4f} bits)"
                    else:
                        label = "no evidence"
                    crit.append(
                        Criterion(
                            f"call-specific transfer ({cpair}) >= {spec.MIN_EFFECT} bits",
                            bd["mean"],
                            spec.MIN_EFFECT,
                            bool(positive and bd["mean"] >= spec.MIN_EFFECT),
                        )
                    )
                    crit.append(
                        Criterion(
                            f"upper 95% bound below {spec.EQUIVALENCE} bits",
                            bd["upper"],
                            spec.EQUIVALENCE,
                            bool(bd["upper"] < spec.EQUIVALENCE),
                        )
                    )
        elif note:
            label += note
        return ScorecardResult(
            feature="comm",
            version=COMM_SCORECARD_VERSION,
            level=None,
            label=label,
            endpoint=endpoint,
            criteria=crit,
            provisional=True,
            needs_missing=[],
        )
