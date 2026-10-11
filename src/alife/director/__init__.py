"""The ecosystem director (docs/DIRECTOR.md, build steps 2 and 3): vital signs, a behaviour archive, the rule
tier, the optional LLM tier, the decision log and replay. Host-only: it runs in ``runner.run`` between
compiled steps and edits the state with ``_replace`` of the ``env_*`` fields (and ``patch_pos`` for a patch
move); the compiled step is unchanged.

Modes (``run.director``): ``live`` decides at every rule boundary and appends to ``director.jsonl``;
``replay`` applies the entries already in a log and never decides; ``off`` builds no ``Director`` at all.

The LLM tier (``director.llm_every > 0``, live mode only) splits the roles at every boundary (L2): the rule
tier keeps the safety moves (reverts, extinction rescue, runaway damper) and, at a multiple of
``llm_every``, the model makes the discretionary ones, waiting for its reply inside the hook (L1). A failed
call, an invalid reply or a reply of which nothing could be applied falls back to the rule tier's
discretionary rules for that boundary. With the tier off the behaviour is exactly that of step 2.

Everything the director remembers is a fold: the vital-sign tracker and archive of the ``metrics.jsonl`` rows,
and budget, cooldowns, active effects and drama of the ``director.jsonl`` entries (``pacing``). A resumed live
run rebuilds both from disk, so no sidecar state exists.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

from alife import metrics
from alife.config import Config, dims
from alife.director import actions, llm, log, pacing, prompt, rules
from alife.director.rules import RULE_PARAMS, RuleParams
from alife.director.vitals import Tracker, Vitals
from alife.runio import DIRECTOR_ARCHIVE_FILE, DIRECTOR_LOG_FILE, JOURNAL_FILE, RunDir
from alife.state import State

MODES = ("live", "replay")


class _Taken(NamedTuple):
    """What became of a reply's candidates: the applied ones, the refused ones with a reason, the clamps."""

    accepted: list[dict[str, Any]]
    rejected: list[dict[str, Any]]
    clamps: list[dict[str, Any]]


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
        self.llm_line: str | None = None  # the console line of this boundary's LLM call (if it made one)
        # The LLM tier exists in live mode only: a replay never builds a client, so it cannot reach a network.
        self.llm = llm.Client.of(cfg) if mode == "live" and cfg.director.llm_every > 0 else None
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
        self.applied, self.llm_line = [], None
        if self.mode == "live":
            self.tracker.observe(row)

    def on_boundary(self, state: State, row: Mapping[str, Any], t: int) -> State:
        """The state after this boundary's interventions: decided from ``row`` (live) or read from the log
        (replay). In live mode the entries are durable in ``director.jsonl`` before the state is returned."""
        if row["step"] != t:
            raise ValueError(f"the boundary at step {t} was given the row of step {row['step']}")
        self.applied, self.llm_line = [], None
        if self.mode == "replay":
            return self._replay(state, t)
        return self._decide(state, t)

    def status_lines(self) -> list[str]:
        """One line per entry applied at the latest boundary (and one for its LLM call), for the console."""
        lines = [f"director @ {e['step_applied']}: {describe(e)}" for e in self.applied]
        return [self.llm_line, *lines] if self.llm_line else lines

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
        llm_call: str | None = None,
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
            **({"llm_call": llm_call} if llm_call is not None else {}),
        }
        return json.loads(json.dumps(entry, allow_nan=False))

    def _context(self, work: State, v: Vitals, t: int, new: Sequence[Mapping[str, Any]]) -> rules.Context:
        """What a decision sees: the vital signs, the pacing folded over the log plus this boundary's entries
        so far, and the environment as ``work`` now has it."""
        return rules.Context(
            t,
            self.cfg.run.seed,
            self.persona.name,
            self.budget,
            self.cfg.world.size,
            v,
            pacing.fold([*self.entries, *new], t),
            actions.env_view(work),
        )

    def _decide(self, state: State, t: int) -> State:
        """Reverts, then the safety rules, then (at an LLM boundary) the model or, failing that, the rule
        tier's discretionary rules. With the LLM tier off the rule tier alone decides, as in step 2."""
        v = self._vitals(state, t)
        new: list[dict[str, Any]] = []
        notes: list[str] = []
        records: list[dict[str, Any]] = []
        work = state
        for eff in pacing.due_reverts(pacing.fold(self.entries, t), t):
            clamped = actions.Clamped(dict(eff.restore), {}, None)
            reason = f"end of the effect from step {eff.origin}"
            entry = self._entry(v, "revert", eff.action, clamped, reason, reverts=eff.origin)
            work = self._apply(work, entry)
            new.append(entry)
        ctx = self._context(work, v, t, new)
        # The LLM tier takes the discretionary moves from the rule tier (L2): only the safety rules remain.
        decision = (rules.decide if self.llm is None else rules.decide_safety)(ctx, self.params)
        notes.extend(decision.notes)
        work = self._take_all(work, v, decision.proposals, new, notes)
        if self.llm is not None and t % self.cfg.director.llm_every == 0:
            work = self._consult(work, v, t, new, notes, records)
        if new:
            self.rd.append_jsonl(DIRECTOR_LOG_FILE, new)
        self.entries = [*self.entries, *new]
        self._journal(t, new, notes, records)
        self.applied = new
        return work

    def _take_all(
        self,
        work: State,
        v: Vitals,
        proposals: Sequence[rules.Proposal],
        new: list[dict[str, Any]],
        notes: list[str],
    ) -> State:
        """Take the rule tier's proposals in order: each entry goes to ``new``, a refused one to ``notes``."""
        for prop in proposals:
            try:
                work, entry = self._take(work, v, prop, [*self.entries, *new])
            except actions.ActionError as e:
                notes.append(f"{prop.rule}: {prop.action} dropped ({e})")
                continue
            new.append(entry)
        return work

    def _take(
        self,
        work: State,
        v: Vitals,
        prop: rules.Proposal,
        entries: Sequence[Mapping[str, Any]],
        llm_call: str | None = None,
    ) -> tuple[State, dict[str, Any]]:
        """Clamp and apply one proposal to ``work``; the entry that results (``source`` is ``llm`` for a
        proposal of the model, whose entry carries the call id and no rule)."""
        active = pacing.fold(entries, v.step).active
        restore = active["regen"].restore if prop.action == "regen_pulse" and "regen" in active else None
        clamped = actions.clamp(
            prop.action, prop.raw, prop.duration, actions.env_view(work), self.cfg, restore
        )
        if llm_call is None:
            entry = self._entry(v, "rule", prop.action, clamped, prop.reason, rule=prop.rule)
        else:
            entry = self._entry(v, "llm", prop.action, clamped, prop.reason, llm_call=llm_call)
        return self._apply(work, entry), entry

    def _apply(self, work: State, entry: Mapping[str, Any]) -> State:
        """``work`` after ``entry``, which first passes the replay validation: the director never acts on
        something a replay would reject."""
        actions.validate_entry(entry, self.cfg, self.rule_every)
        return actions.apply(work, entry, self.cfg)

    # --- the LLM tier -------------------------------------------------------------------------------------

    def _consult(
        self,
        work: State,
        v: Vitals,
        t: int,
        new: list[dict[str, Any]],
        notes: list[str],
        records: list[dict[str, Any]],
    ) -> State:
        """Ask the model for this boundary's discretionary moves and take what survives (L1, L8, L9).

        The prompt shows the reverts and safety actions already in ``new``. The reply is untrusted data: it
        selects menu actions, each of which then passes the rule tier's eligibility checks (budget, cooldown,
        lever, catastrophe cap) and the step-2 clamp, exactly like a rule's proposal. A call that fails, a
        reply that is invalid, or a reply of which nothing could be applied (and that was not ``none``) hands
        the boundary to the rule tier's discretionary rules. A valid ``none`` is a decision: no fallback.
        """
        assert self.llm is not None
        call = f"llm-{t}"
        messages = prompt.build_messages(
            self._context(work, v, t, new),
            self.params,
            self.cfg,
            self.tracker.history,
            self.entries,
            list(new),
            notes,
        )
        ex = self.llm.chat(self.llm.body(messages, self.cfg, t))
        outcome, error = ("failed" if ex.error else "pending"), ex.error
        reply: llm.Reply | None = None
        taken = _Taken([], [], [])
        if ex.text is not None:
            try:
                reply = llm.parse_reply(ex.text, self.cfg)
            except llm.ReplyError as e:
                outcome, error = "invalid", llm.clean_text(str(e), 300)
            else:
                work = self._take_reply(work, v, reply, call, new, taken)
                outcome = "none" if reply.explicit_none else "ok" if taken.accepted else "rejected"
        fallback: list[str] | None = None
        if outcome in ("failed", "invalid", "rejected"):
            decision = rules.decide_discretionary(self._context(work, v, t, new), self.params)
            notes.extend(decision.notes)
            before = len(new)
            work = self._take_all(work, v, decision.proposals, new, notes)
            fallback = [e["action"] for e in new[before:]]
        records.append(
            {
                "kind": "llm_call",
                "step": t,
                "persona": self.persona.name,
                "llm_call": call,
                "model": self.llm.model,
                "url": self.llm.url,
                "latency_s": round(ex.latency_s, 3),
                "messages": messages,
                "response": ex.text,
                "finish_reason": ex.finish_reason,
                "usage": ex.usage,
                "outcome": outcome,
                "error": error,
                "validation": list(reply.notes) if reply else [],
                "accepted": taken.accepted,
                "rejected": taken.rejected,
                "clamps": taken.clamps,
                "fallback": None if fallback is None else {"tier": "rule", "actions": fallback},
                "journal": reply.journal if reply else None,
            }
        )
        self.llm_line = self._llm_line(records[-1])
        return work

    def _take_reply(
        self,
        work: State,
        v: Vitals,
        reply: llm.Reply,
        call: str,
        new: list[dict[str, Any]],
        taken: _Taken,
    ) -> State:
        """Take the reply's candidates in order; each accepted one goes to ``new`` and ``taken.accepted``,
        each refused one to ``taken.rejected`` with the reason."""
        for cand in reply.candidates:
            prop = rules.Proposal(cand.action, cand.raw, cand.duration, "llm", cand.effect or "(none given)")
            why = rules.eligibility(self._context(work, v, v.step, new), self.params, cand.action, cand.raw)
            entry = None
            if why is None:
                try:
                    work, entry = self._take(work, v, prop, [*self.entries, *new], llm_call=call)
                except actions.ActionError as e:
                    why = llm.clean_text(str(e), 300)
            if entry is None:
                taken.rejected.append({"action": cand.action, "why": why})
                continue
            new.append(entry)
            taken.accepted.append(
                {"action": cand.action, "params": entry["params"], "duration": entry["duration"]}
            )
            taken.clamps.extend({"action": cand.action, **c} for c in llm.clamps(cand, entry))
        return work

    def _llm_line(self, rec: Mapping[str, Any]) -> str:
        what = {
            "ok": f"{len(rec['accepted'])} action(s) applied",
            "none": "chose to do nothing",
            "rejected": "every proposal refused",
            "invalid": "invalid reply",
            "failed": str(rec["error"]),
        }[rec["outcome"]]
        fallback = "; rule-tier fallback" if rec["fallback"] else ""
        return f"director @ {rec['step']}: llm {rec['latency_s']:.1f}s: {what}{fallback}"

    # --- journal ------------------------------------------------------------------------------------------

    def _journal(
        self,
        t: int,
        entries: Sequence[Mapping[str, Any]],
        notes: Sequence[str],
        records: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        """One line per entry, and one per rule that newly triggered but was blocked (a persisting condition
        is noted once, not at every boundary); then the record of the LLM call, if one was made."""
        rules_blocked = frozenset(n.split(" (", 1)[0] for n in notes)
        fresh = [n for n in notes if n.split(" (", 1)[0] not in self._blocked]
        self._blocked = rules_blocked
        lines = [{"step": t, "persona": self.persona.name, "text": describe(e)} for e in entries]
        lines += [{"step": t, "persona": self.persona.name, "text": n} for n in fresh]
        lines += records
        if lines:
            self.rd.append_jsonl(JOURNAL_FILE, lines)


def make_director(
    rd: RunDir, cfg: Config, start_step: int, params: RuleParams = RULE_PARAMS
) -> Director | None:
    """The director for ``cfg.run.director`` (None when ``off``), built from the run directory's files."""
    mode = cfg.run.director
    return None if mode == "off" else Director(rd, cfg, mode, start_step, params)
