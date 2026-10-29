"""Social tracking on a real run: calibration, churn and snapshot cost (docs/GAMES.md §6, DESIGN §29.12).

The G2 thresholds (the kin grid width and occupancy, the enter and leave sizes, theta, the deadband, the slot
slack) are provisional (``SOCIAL_VERSION`` 1) until they have been seen on a long sequence of snapshots of a
real run. This tool re-simulates a run from a checkpoint with the observers on (``game.observe`` need not be
set in the run) and reports, snapshot after snapshot, what the trackers did:

* per snapshot, the thresholds in force (``n_ref``, ``rescaled``, ``occ_min``, enter and leave); per family
  (kin groups, colonies) the tracker's counts (``n_prev``, ``n_cur``, ``n_admitted``, ``n_matched``), the
  10/50/90 quantiles of the Jaccard of the matched pairs and of the best Jaccard of the unmatched previous
  systems, and the event counts; for kin groups the occupied bins, the unaffiliated fraction, the tracked
  sizes and the quantiles of ``sim_self`` over the slotted groups (the chaining report: single linkage chains
  within a clade, DESIGN §29.5); for colonies ``members``, ``largest`` and ``converged``; the houses alive and
  the share of their members in houses founded since the study's start (``next_uid`` of the start state:
  houses below it were seeded from clades) against the seeded ones, which is the check of ``game.house_rate``
  (DESIGN §29.14);
* per study and family, the **churn**: the ids born in the study that ended within 3 snapshots, as a count and
  as a share of the ids born (a tracker that flickers shows up here);
* ``timing`` (kept apart: the rest of the report is deterministic for a given run, parameters and device): per
  snapshot the evaluation time of the snapshot step against the median of the 8 steps before it, the
  ``settle`` time (the ``take_records()`` call at the snapshot step: one host transfer and ``Society.settle``)
  and, with ``--time-components``, the median of 3 timed evaluations of ``metrics.components`` alone on that
  state. The first step and the first snapshot of the study include their compile.

**Stored records.** The study takes the records after each evaluation as the runner does (nothing is
appended anywhere). Unless ``--no-verify``, a study record is compared with the run's own record of the same
step in ``observers/social.jsonl`` when both have the same ``origin`` and ``social_digest`` (the study loaded
the run's sidecar at ``--from`` and runs the run's rules): ``records_verified`` counts the equal ones and a
difference is a mismatch (exit 1), like a metrics row. A study that starts fresh or sweeps ``--every`` has no
comparable record: ``records_verified`` is 0, ``records_comparable`` false, and ``records_note`` says why.
``--no-verify`` also skips the check of every stored ``metrics.jsonl`` row on the way (``resim.compare_row``).

The study's rules are ``dataclasses.replace(SOCIAL_V1, snapshot_every=S)`` for ``--every S`` (the output
records their digest) and ``observer.DEFAULT_SOCIAL`` without it. The run directory is read only. Usage::

    python -m alife.analysis.social RUN --from STEP --steps N --out FILE.json
        [--every S] [--states DIR] [--no-verify] [--time-components]
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import statistics
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import mlx.core as mx
import numpy as np

from alife import metrics, spatial
from alife.analysis.resim import ResimMismatch, compare_row
from alife.analysis.signals import load_state
from alife.config import config_hash, dims
from alife.game import observer
from alife.game.observer import Observer
from alife.game.social import SOCIAL_V1, SOCIAL_VERSION, SocialRules
from alife.game.society import SnapInfo
from alife.runio import RunDir, _dumps
from alife.state import counter_blocks, to_numpy
from alife.step import make_step
from alife.tracking import TRACK_VERSION, Diagnostics, Event, EventKind

__all__ = ["churn", "main", "study"]

CHURN_WITHIN: Final = 3  # an id that ends within this many snapshots of its birth counts as churn
BEFORE_STEPS: Final = 8  # the steps whose median time the snapshot step is compared with
COMPONENT_REPS: Final = 3  # timed evaluations of metrics.components per snapshot (--time-components)
_BORN: Final = (EventKind.BORN, EventKind.SPLIT)  # the kinds that give a system a new id
_EVENT_NAMES: Final = tuple(k.name.lower() for k in EventKind)


def _q(values: Sequence[float] | np.ndarray) -> dict[str, Any]:
    """``{n, p10, p50, p90}`` of a sample (the quantiles None when it is empty), 6 decimals."""
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return {"n": 0, "p10": None, "p50": None, "p90": None}
    p10, p50, p90 = (round(float(x), 6) for x in np.quantile(v, [0.1, 0.5, 0.9]))
    return {"n": int(v.size), "p10": p10, "p50": p50, "p90": p90}


def _event_counts(events: Sequence[Event]) -> dict[str, int]:
    return {nm: sum(1 for e in events if e.kind.name.lower() == nm) for nm in _EVENT_NAMES}


def churn(events_by_snapshot: Sequence[Sequence[Event]], within: int = CHURN_WITHIN) -> dict[str, Any]:
    """The churn of one family over a study: ``events_by_snapshot[i]`` are the events of its ``i``-th
    snapshot. ``born`` counts the ids that appeared (``BORN`` or ``SPLIT``), ``ended_within`` those of them
    that ended (``MERGED`` or ``DISSOLVED``) at most ``within`` snapshots later, ``share`` their ratio (None
    without a birth). An id is never revived, so each id is counted once."""
    born_at: dict[int, int] = {}
    ended_at: dict[int, int] = {}
    for i, events in enumerate(events_by_snapshot):
        for e in events:
            if e.kind in _BORN:
                born_at[e.system] = i
            else:
                ended_at[e.system] = i
    short = sum(1 for sid, i in born_at.items() if sid in ended_at and ended_at[sid] - i <= within)
    born = len(born_at)
    return {
        "born": born,
        "ended_within": short,
        "share": round(short / born, 6) if born else None,
    }


def _family(diag: Diagnostics, events: Sequence[Event]) -> dict[str, Any]:
    return {
        "n_prev": diag.n_prev,
        "n_cur": diag.n_cur,
        "n_admitted": diag.n_admitted,
        "n_matched": diag.n_matched,
        "jaccard_matched": _q(diag.matched_jaccard),
        "jaccard_unmatched_best": _q(diag.unmatched_best_jaccard),
        "events": _event_counts(events),
    }


def _houses_entry(info: SnapInfo, summary: Mapping[str, Any], first_uid: int) -> dict[str, Any]:
    """The houses of one snapshot: how many are alive, and the members of those seeded at the study's start
    against those of the houses founded since (DESIGN §29.14, the check of ``game.house_rate``). The id of a
    house is its founder's uid, so a house with an id below ``first_uid``, the ``next_uid`` of the start,
    existed then."""
    members = int(info.house_sizes.sum())
    founded = info.house_ids >= first_uid
    founded_members = int(info.house_sizes[founded].sum())
    return {
        "alive": summary["houses"]["alive"],
        "unknown_immigrant": summary["houses"]["unknown_immigrant"],
        "members": members,
        "seeded_members": members - founded_members,
        "founded_members": founded_members,
        "founded_houses": int(founded.sum()),
        "founded_share": round(founded_members / members, 6) if members else None,
    }


def _snapshot_entry(info: SnapInfo, record: Mapping[str, Any], first_uid: int) -> dict[str, Any]:
    """One snapshot of the report, from the society's last diagnostics and the snapshot's record;
    ``first_uid`` is the ``next_uid`` of the study's start (see ``_houses_entry``)."""
    summary = record["summary"]
    n = info.n_alive
    sim_self = [v for v in info.sim_self.values() if v is not None]
    kin = _family(info.kin_diag, info.kin_events)
    col = _family(info.col_diag, info.col_events)
    return {
        "step": info.step,
        "n_alive": n,
        "thresholds": dict(info.thresholds),
        "kin_groups": {
            **kin,
            "occupied_bins": info.occupied_bins,
            "unaffiliated_frac": round(info.unaffiliated / n, 6) if n else None,
            "tracked": len(info.kin_sizes),
            "sizes": sorted(info.kin_sizes.values(), reverse=True),
            "sim_self": _q(sim_self),
        },
        "colonies": {
            **col,
            "tracked": summary["colonies"]["tracked"],
            "members": summary["colonies"]["members"],
            "largest": summary["colonies"]["largest"],
            "converged": summary["colonies"]["converged"],
            "match": summary["colonies"]["match"],
        },
        "houses": _houses_entry(info, summary, first_uid),
    }


def _check_params(steps: int, every: int | None) -> None:
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if every is not None and every < 1:
        raise ValueError("every must be >= 1")


def _records_note(
    verify: bool,
    stored: Sequence[Mapping[str, Any]],
    study_records: Sequence[Mapping[str, Any]],
    comparable: int,
) -> str | None:
    """Why no study record was comparable with a stored one (None when at least one was)."""
    if not verify:
        return "verification is off (--no-verify)"
    if not stored:
        return "the run has no social log"
    if comparable:
        return None
    by_step = {r["step"]: r for r in stored}
    shared = [(r, by_step[r["step"]]) for r in study_records if r["step"] in by_step]
    if not shared:
        return (
            f"no snapshot of the study falls on a stored record (the log's steps end at {stored[-1]['step']})"
        )
    if any(a["social_digest"] != b["social_digest"] for a, b in shared):
        return "the study's social rules (cadence or thresholds) differ from the stored records' digest"
    return (
        f"the study's observer started fresh (origin {shared[0][0]['origin']}), "
        f"the stored records have origin {shared[0][1]['origin']}"
    )


def _differs(a: Mapping[str, Any], b: Mapping[str, Any]) -> list[str]:
    """The top-level keys in which two records differ (their serialisation is the comparison)."""
    return [k for k in dict.fromkeys([*a, *b]) if _dumps(a.get(k)) != _dumps(b.get(k))]


def study(
    run: Path | str,
    *,
    start: int,
    steps: int,
    every: int | None = None,
    states_dir: Path | None = None,
    verify: bool = True,
    time_components: bool = False,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """The social study of a run from step ``start`` for ``steps`` steps (see the module docstring).

    Needs a checkpoint of the run (or a cached state in ``states_dir``) at ``start``. Returns the JSON-safe
    report; ``ResimMismatch`` if a stored row or a comparable stored record differs from the re-simulation.
    """
    _check_params(steps, every)
    sr: SocialRules = (
        observer.DEFAULT_SOCIAL if every is None else dataclasses.replace(SOCIAL_V1, snapshot_every=every)
    )
    rd = RunDir.open(Path(run))
    cfg = rd.config()
    cfg = dataclasses.replace(cfg, game=dataclasses.replace(cfg.game, observe=True))
    end = start + steps
    nd = dims(cfg).nd

    state = load_state(rd, cfg, states_dir, start)
    first_uid = int(to_numpy(state.next_uid))
    # ValueError for rules this world cannot hold
    obs, note = Observer.open(rd, state, cfg, social=sr, tracers=False)
    resumed = any(st == start for st, _ in rd.checkpoints())
    source = "sidecar" if resumed and note is None else "fresh"
    rows = {int(r["step"]): r for r in rd.metrics_rows()} if verify else {}
    stored = rd.social_records() if verify else []
    stored_by_step = {int(r["step"]): r for r in stored}
    last_row = max(rows, default=-1)
    spatial.prepare_knn(cfg)  # the self-check of the fused search, once per process (§4)
    step_fn = make_step(cfg, observe=True)
    log(
        f"social: {rd.path} from step {start} to {end}, a snapshot every {sr.snapshot_every} steps, "
        f"observers {source}, verify={verify}"
    )

    comp_fn: Callable[[mx.array, mx.array], tuple[mx.array, mx.array]] | None = None
    comp_warm = False
    if time_components:

        def comp_raw(partner: mx.array, alive: mx.array) -> tuple[mx.array, mx.array]:
            return metrics.components(partner, alive, sr.colony_iters)

        comp_fn = mx.compile(comp_raw) if cfg.run.compile else comp_raw

    def components_ms(s: Any) -> float:
        nonlocal comp_warm
        assert comp_fn is not None
        if not comp_warm:  # the first evaluation compiles
            mx.eval(comp_fn(s.partner, s.alive))
            comp_warm = True
        times = []
        for _ in range(COMPONENT_REPS):
            t0 = time.perf_counter()
            mx.eval(comp_fn(s.partner, s.alive))
            times.append(time.perf_counter() - t0)
        return 1e3 * statistics.median(times)

    snapshots: list[dict[str, Any]] = []
    kin_events: list[list[Event]] = []
    col_events: list[list[Event]] = []
    study_records: list[dict[str, Any]] = []
    timing: list[dict[str, Any]] = []
    recent: collections.deque[float] = collections.deque(maxlen=BEFORE_STEPS)
    counters: list[mx.array] = []
    ledgers: list[mx.array] = []
    checked = comparable = verified = 0
    t = start
    t_wall, t_log = time.perf_counter(), t

    while t < end:
        t0 = time.perf_counter()
        state, out = step_fn(state)
        obs.update(state, out.acts, t + 1)
        mx.eval(state, out.base(), *obs.arrays())
        dt = time.perf_counter() - t0
        if t < last_row:  # only the rows still to be compared need the step's counters
            counters.append(out.counters)
            ledgers.append(out.ledger)
        t += 1

        stored_row = rows.get(t)
        if stored_row is not None:
            if stored_row["steps"] == len(counters):
                bad = compare_row(
                    stored_row,
                    counters,
                    ledgers,
                    int(to_numpy(state.total)),
                    nd=nd,
                    weather=cfg.weather.enabled,
                    blocks=counter_blocks(cfg),
                )
                if bad:
                    raise ResimMismatch(
                        f"step {t}: observed re-simulation differs from metrics.jsonl in {bad}"
                    )
                checked += 1
            counters, ledgers = [], []

        if t % sr.snapshot_every == 0:
            before = statistics.median(recent) if recent else None
            comp_ms = components_ms(state) if time_components else None
            t1 = time.perf_counter()
            records = obs.take_records()  # the host transfer and Society.settle of this snapshot
            settle = time.perf_counter() - t1
            if len(records) != 1 or records[0]["step"] != t or obs.society.info is None:
                raise RuntimeError(
                    f"step {t}: expected exactly the record of this snapshot, got {len(records)}"
                )
            record, info = records[0], obs.society.info
            snapshots.append(_snapshot_entry(info, record, first_uid))
            kin_events.append(info.kin_events)
            col_events.append(info.col_events)
            study_records.append(record)
            old = stored_by_step.get(t)
            if (
                old is not None
                and old["origin"] == record["origin"]
                and old["social_digest"] == record["social_digest"]
            ):
                comparable += 1
                if _dumps(old) != _dumps(record):
                    raise ResimMismatch(
                        f"step {t}: the study's social record differs from observers/social.jsonl "
                        f"in {_differs(old, record)}"
                    )
                verified += 1
            timing.append(
                {
                    "step": t,
                    "eval_ms": round(1e3 * dt, 3),
                    "before_median_ms": None if before is None else round(1e3 * before, 3),
                    "settle_ms": round(1e3 * settle, 3),
                    **({} if comp_ms is None else {"components_ms": round(comp_ms, 3)}),
                }
            )
        recent.append(dt)

        if t - t_log >= 1000:
            rate = (t - t_log) / (time.perf_counter() - t_wall)
            log(
                f"social: step {t}, {rate:.1f} steps/s, {len(snapshots)} snapshots, {checked} rows and "
                f"{verified} records verified"
            )
            t_wall, t_log = time.perf_counter(), t

    return {
        "run": str(rd.path),
        "config_hash": config_hash(cfg),
        "start": start,
        "steps": steps,
        "params": {
            "every": sr.snapshot_every,
            "match": "uid" if sr.snapshot_every <= sr.uid_spacing else "footprint",
            "verify": verify,
            "time_components": time_components,
            "churn_within": CHURN_WITHIN,
            "social": dataclasses.asdict(sr),
        },
        "social_version": SOCIAL_VERSION,
        "social_digest": sr.digest(),
        "track_version": TRACK_VERSION,
        "device": str(mx.default_device()),
        "mlx_version": mx.__version__,
        "observer_start": {"step": start, "source": source, "note": note, "next_uid": first_uid},
        "rows_verified": checked,
        "records_verified": verified,
        "records_comparable": comparable > 0,
        "records_note": _records_note(verify, stored, study_records, comparable),
        "snapshots": snapshots,
        "churn": {
            "within": CHURN_WITHIN,
            "kin_group": churn(kin_events),
            "colony": churn(col_events),
        },
        "timing": {
            "steps": steps,
            "snapshots": timing,
        },
    }


# --- command line ---------------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m alife.analysis.social", description=__doc__.split("\n")[0])
    ap.add_argument("run", type=Path, help="run directory (read only)")
    ap.add_argument("--from", dest="start", type=int, required=True, help="step to start from (a checkpoint)")
    ap.add_argument("--steps", type=int, required=True, help="steps to simulate from there")
    ap.add_argument("--out", type=Path, required=True, help="report JSON (outside the run directory)")
    ap.add_argument(
        "--every", type=int, default=None, help="steps between snapshots (default: the run's social rules)"
    )
    ap.add_argument("--states", type=Path, default=None, help="directory of re-simulated states (resim)")
    ap.add_argument(
        "--no-verify",
        action="store_true",
        help="skip the check of the stored metrics rows and social records",
    )
    ap.add_argument(
        "--time-components",
        action="store_true",
        help="also time metrics.components alone on each snapshot's state",
    )
    a = ap.parse_args(argv)
    if a.out.resolve().is_relative_to(a.run.resolve()):
        ap.error("--out must not be inside the run directory")
    try:
        report = study(
            a.run,
            start=a.start,
            steps=a.steps,
            every=a.every,
            states_dir=a.states,
            verify=not a.no_verify,
            time_components=a.time_components,
            log=lambda m: print(m, flush=True),
        )
    except ResimMismatch as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except (ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
