"""The ecosystem director (docs/DIRECTOR.md, build step 2): vital signs, a behaviour archive, the rule tier,
the decision log and replay. Host-only: it runs in ``runner.run`` between compiled steps and edits the state
with ``_replace`` of the ``env_*`` fields (and ``patch_pos`` for a patch move); the compiled step is
unchanged.

Modes (``run.director``): ``live`` decides at every rule boundary and appends to ``director.jsonl``;
``replay`` applies the entries already in a log and never decides; ``off`` builds no ``Director`` at all.

Everything the director remembers is a fold: the vital-sign tracker and archive of the ``metrics.jsonl`` rows,
and budget, cooldowns, active effects and drama of the ``director.jsonl`` entries (``pacing``). A resumed live
run rebuilds both from disk, so no sidecar state exists.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from alife import metrics
from alife.config import Config, dims
from alife.director import actions, log, pacing, rules
from alife.director.rules import RULE_PARAMS, RuleParams
from alife.director.vitals import Tracker, Vitals
from alife.runio import DIRECTOR_ARCHIVE_FILE, DIRECTOR_LOG_FILE, JOURNAL_FILE, RunDir
from alife.state import State

MODES = ("live", "replay")


def describe(entry: Mapping[str, Any]) -> str:
    """A one-line account of an entry for the status output and the journal."""
    p = ", ".join(f"{k}={v}" for k, v in entry["params"].items())
    dur = f" for {entry['duration']} steps" if entry.get("duration") else ""
    src = "revert" if entry["source"] == "revert" else entry.get("rule") or entry["source"]
    return f"{entry['action']}({p}){dur} [{src}: {entry['reason']}]"


class Director:
    """The controller the runner calls: ``due(t)``, ``observe(row)``, ``on_boundary(state, row, t)``,
    ``on_checkpoint(t)``. ``applied`` holds the entries of the latest boundary."""

    def __init__(
        self, rd: RunDir, cfg: Config, mode: str, start_step: int, params: RuleParams = RULE_PARAMS
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"director mode must be one of {MODES}, got {mode!r}")
        self.rd, self.cfg, self.mode, self.params = rd, cfg, mode, params
        self.rule_every = cfg.director.rule_every
        self.persona = params.persona(cfg.director.persona)
        self.budget = rules.effective_budget(cfg.director.budget_per_100k, self.persona)
        self.applied: list[dict[str, Any]] = []
        self._blocked: frozenset[str] = frozenset()  # rules the previous boundary journalled as blocked
        records = rd.read_jsonl(DIRECTOR_LOG_FILE)
        self.entries = log.entries_of(records)
        if mode == "replay":
            log.validate_log(records, cfg)
            self.pending: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
            for e in self.entries:  # entries up to the start step are already in the loaded state
                if e["step_applied"] > start_step:
                    self.pending[e["step_applied"]].append(e)
        else:
            self.tracker = Tracker(params)
            self._rebuild(rd.metrics_rows(), start_step)
            header = log.make_header(cfg, start_step, self.budget)
            if log.needs_header(records, header):
                rd.append_jsonl(DIRECTOR_LOG_FILE, [header])

    def _rebuild(self, rows: Sequence[Mapping[str, Any]], start_step: int) -> None:
        """Fold the stored rows up to the start step into the tracker, as live mode did (A7)."""
        for row in rows:
            if row["step"] <= start_step:
                self.observe(row)
                if self.due(row["step"]):
                    self.tracker.close(row["step"])

    # --- the runner's hooks -------------------------------------------------------------------------------

    def due(self, t: int) -> bool:
        return t % self.rule_every == 0

    def observe(self, row: Mapping[str, Any]) -> None:
        """Feed one metrics row (live mode; a replay needs no vital signs). Starts a new row: the entries of
        an earlier boundary are no longer reported by ``status_lines``."""
        self.applied = []
        if self.mode == "live":
            self.tracker.observe(row)

    def on_boundary(self, state: State, row: Mapping[str, Any], t: int) -> State:
        """The state after this boundary's interventions: decided from ``row`` (live) or read from the log
        (replay). In live mode the entries are durable in ``director.jsonl`` before the state is returned."""
        if row["step"] != t:
            raise ValueError(f"the boundary at step {t} was given the row of step {row['step']}")
        self.applied = []
        if self.mode == "replay":
            return self._replay(state, t)
        return self._decide(state, t)

    def status_lines(self) -> list[str]:
        """One line per entry applied at the latest boundary, for the run's console output."""
        return [f"director @ {e['step_applied']}: {describe(e)}" for e in self.applied]

    def on_checkpoint(self, t: int) -> None:
        """Write ``director_archive.json`` (analysis only; never read back). Replays write nothing."""
        if self.mode == "live":
            self.rd.write_json(DIRECTOR_ARCHIVE_FILE, self.tracker.archive.to_json(t))

    # --- replay -------------------------------------------------------------------------------------------

    def _replay(self, state: State, t: int) -> State:
        self.applied = list(self.pending.pop(t, []))
        for entry in self.applied:
            state = actions.apply(state, entry, self.cfg)
        return state

    # --- live ---------------------------------------------------------------------------------------------

    def _vitals(self, state: State, t: int) -> Vitals:
        bd = self.tracker.close(t)
        k_est = metrics.carrying_capacity(state, self.cfg, state.step, state.env)
        return self.tracker.vitals(bd, k_est, dims(self.cfg).n_cap)

    def _entry(
        self,
        v: Vitals,
        source: str,
        action: str,
        clamped: actions.Clamped,
        reason: str,
        rule: str | None = None,
        reverts: int | None = None,
    ) -> dict[str, Any]:
        """A log entry in its canonical JSON form (what a replay reads back is exactly what is applied)."""
        entry = {
            "step_applied": v.step,
            "source": source,
            "persona": self.persona.name,
            "action": action,
            "params": clamped.params,
            "prev": clamped.prev,
            "duration": clamped.duration,
            "reason": reason,
            "vitals_step": v.step,
            "vitals": v.compact(),
            **({"rule": rule} if rule is not None else {}),
            **({"reverts": reverts} if reverts is not None else {}),
        }
        return json.loads(json.dumps(entry, allow_nan=False))

    def _decide(self, state: State, t: int) -> State:
        v = self._vitals(state, t)
        new: list[dict[str, Any]] = []
        notes: list[str] = []
        work = state
        for eff in pacing.due_reverts(pacing.fold(self.entries, t), t):
            clamped = actions.Clamped(dict(eff.restore), {}, None)
            reason = f"end of the effect from step {eff.origin}"
            entry = self._entry(v, "revert", eff.action, clamped, reason, reverts=eff.origin)
            work = self._apply(work, entry)
            new.append(entry)
        ctx = rules.Context(
            t,
            self.cfg.run.seed,
            self.persona.name,
            self.budget,
            self.cfg.world.size,
            v,
            pacing.fold([*self.entries, *new], t),
            actions.env_view(work),
        )
        decision = rules.decide(ctx, self.params)
        notes.extend(decision.notes)
        for prop in decision.proposals:
            try:
                work, entry = self._take(work, v, prop, [*self.entries, *new])
            except actions.ActionError as e:
                notes.append(f"{prop.rule}: {prop.action} dropped ({e})")
                continue
            new.append(entry)
        if new:
            self.rd.append_jsonl(DIRECTOR_LOG_FILE, new)
        self.entries = [*self.entries, *new]
        self._journal(t, new, notes)
        self.applied = new
        return work

    def _take(
        self, work: State, v: Vitals, prop: rules.Proposal, entries: Sequence[Mapping[str, Any]]
    ) -> tuple[State, dict[str, Any]]:
        """Clamp and apply one proposal to ``work``; the entry that results."""
        active = pacing.fold(entries, v.step).active
        restore = active["regen"].restore if prop.action == "regen_pulse" and "regen" in active else None
        clamped = actions.clamp(
            prop.action, prop.raw, prop.duration, actions.env_view(work), self.cfg, restore
        )
        entry = self._entry(v, "rule", prop.action, clamped, prop.reason, rule=prop.rule)
        return self._apply(work, entry), entry

    def _apply(self, work: State, entry: Mapping[str, Any]) -> State:
        """``work`` after ``entry``, which first passes the replay validation: the director never acts on
        something a replay would reject."""
        actions.validate_entry(entry, self.cfg, self.rule_every)
        return actions.apply(work, entry, self.cfg)

    def _journal(self, t: int, entries: Sequence[Mapping[str, Any]], notes: Sequence[str]) -> None:
        """One line per entry, and one per rule that newly triggered but was blocked (a persisting condition
        is noted once, not at every boundary)."""
        rules_blocked = frozenset(n.split(" (", 1)[0] for n in notes)
        fresh = [n for n in notes if n.split(" (", 1)[0] not in self._blocked]
        self._blocked = rules_blocked
        lines = [{"step": t, "persona": self.persona.name, "text": describe(e)} for e in entries]
        lines += [{"step": t, "persona": self.persona.name, "text": n} for n in fresh]
        if lines:
            self.rd.append_jsonl(JOURNAL_FILE, lines)


def make_director(
    rd: RunDir, cfg: Config, start_step: int, params: RuleParams = RULE_PARAMS
) -> Director | None:
    """The director for ``cfg.run.director`` (None when ``off``), built from the run directory's files."""
    mode = cfg.run.director
    return None if mode == "off" else Director(rd, cfg, mode, start_step, params)
