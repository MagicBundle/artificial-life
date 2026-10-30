"""The ecosystem director (docs/DIRECTOR.md, build steps 2 and 3): vital signs, a behaviour archive, the rule
tier, the optional LLM tier, the decision log and replay. Host-only: it runs in ``runner.run`` between
compiled steps and edits the state with ``_replace`` of the ``env_*`` fields (and ``patch_pos`` for a patch
move); the compiled step is unchanged.

Modes (``run.director``): ``live`` decides at every rule boundary and appends to ``director.jsonl``;
``replay`` applies the entries already in a log and never decides; ``off`` builds no ``Director`` at all;
``player`` (``alife play``, G3b) decides nothing and applies only what the player queues, and ends (at the
boundary, as ``live`` writes it) a timed effect of the rule tier or the model that the world holds at launch.

**The player channel** (G3b spec §6.1). A player's power cannot wait for a rule boundary (a multiple of
``rule_every``, at a metrics row), so the runner calls ``Director.apply_pending(state, t)`` at the *top* of
loop iteration ``t`` (between steps ``t - 1`` and ``t``, after ``pacer.before_step``): in live and player
mode it applies what the pacer queued with ``queue_player`` and the end of the player's own timed effects, in
replay mode the logged records of the player channel (``actions.is_player``) with ``step_applied == t``.
**Rule P:** such a record at ``s`` is applied after the checkpoint at ``s`` (a boundary entry at ``s`` is
applied before it), so a checkpoint never contains a record of the player channel at its own step, and every
path that keeps or drops records by step (``RunDir.truncate_after``, a replay's ``pending``) treats them as
``step_applied < s`` to be inside the checkpoint at ``s``. Without a player (``pacer=None``, nothing queued,
no player's effect running) ``apply_pending`` is a set lookup: the research loop is the old one line for line.

The LLM tier (``director.llm_every > 0``, live mode only) splits the roles at every boundary (L2): the rule
tier keeps the safety moves (reverts, extinction rescue, runaway damper) and, at a multiple of
``llm_every``, the model makes the discretionary ones, waiting for its reply inside the hook (L1). A failed
call, an invalid reply or a reply of which nothing could be applied falls back to the rule tier's
discretionary rules for that boundary. With the tier off the behaviour is exactly that of step 2.

The ``random`` persona (arm R of the evaluation, ``yoked``) has no discretionary rules of its own. In live
mode it keeps the reverts and the safety rules, under the budget, cooldowns and thresholds of the persona that
made the log named by ``director.yoke_log`` (its first header's), and at the steps where that run made
discretionary moves it makes as many uniformly random eligible ones. A replay of it reads only its own log,
whose header records the persona it inherited (``params_persona``) and the budget.

Everything the director remembers is a fold: the vital-sign tracker and archive of the ``metrics.jsonl`` rows,
and budget, cooldowns, active effects and drama of the ``director.jsonl`` entries (``pacing``). A resumed live
run rebuilds both from disk. The one thing that is not a fold is the set of rules whose blocked note the
journal has already written (a persisting condition is noted once): it is saved with the archive at each
checkpoint and restored by a resume from that checkpoint, so that the journal of a resumed run equals the
uninterrupted one's. The rules version is the log's: a resumed run continues under the version of its latest
header (``rules.params_for``), so its rules never change mid-run; a new run or a fork takes the current one,
except a new random-persona run, which takes the version of its yoke log's first header like the rest of the
parameters it inherits from that log (``yoked.inherited_rules_version``).
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any, Final, NamedTuple

from alife import metrics
from alife.config import RANDOM_PERSONA, Config, dims, is_3d
from alife.director import actions, llm, log, pacing, prompt, rules, seasons, yoked
from alife.director.rules import RULE_PARAMS, RuleParams
from alife.director.vitals import Tracker, Vitals
from alife.runio import DIRECTOR_ARCHIVE_FILE, DIRECTOR_LOG_FILE, JOURNAL_FILE, RunDir
from alife.state import State

MODES = ("live", "replay", "player")

# The closed templates of a player record's ``reason`` (the lexicon of the report checks them, G3b spec §5.3).
PLAYER_REASON: Final = {
    "power": "the player used {power}",
    "cancel": "cancelled by the player",
    "end": "end of the effect from step {origin}",
}


class PlayerRequest(NamedTuple):
    """What the pacer queues for a power or its early end (``queue_player``): the client's request id
    (``command``), the power's name (``actions.GAME_POWERS``), its parameters in their natural form as for
    ``actions.clamp`` (``raw``; ``patch`` also selects which effect a cancel ends), and for a timed power its
    ``duration``. ``cancel`` ends the running effects of that power instead."""

    command: int
    power: str
    raw: Mapping[str, Any] = MappingProxyType({})
    duration: int | None = None
    cancel: bool = False


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


def _recorded(records: Sequence[Mapping[str, Any]]) -> tuple[str, int | None]:
    """The persona and budget a random director's own log says it inherited (its first header's; the
    Gardener and no budget for a log that predates them)."""
    first = next((r for r in records if r.get("kind") == log.HEADER_KIND), {})
    return first.get("params_persona") or yoked.FALLBACK_PERSONA, first.get("budget_per_100k")


class Director:
    """The controller the runner calls: ``due(t)``, ``observe(row)``, ``on_boundary(state, row, t)``,
    ``apply_pending(state, t)``, ``on_checkpoint(t)``. ``applied`` holds the entries of the latest boundary.

    ``game_specs`` (``alife play`` only; always on in ``player`` mode) writes ``game_specs`` into the log's
    header, which is what lets the log carry the records of the player channel (``log.validate_log``).
    """

    def __init__(
        self,
        rd: RunDir,
        cfg: Config,
        mode: str,
        start_step: int,
        params: RuleParams = RULE_PARAMS,
        *,
        game_specs: bool = False,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"director mode must be one of {MODES}, got {mode!r}")
        records = rd.read_jsonl(DIRECTOR_LOG_FILE)
        if mode != "replay":
            params = self._resume_params(rd, records, params)
        self.rd, self.cfg, self.mode, self.params = rd, cfg, mode, params
        self.game_specs = game_specs or mode == "player"
        self.rule_every = cfg.director.rule_every
        self.name = cfg.director.persona  # what entries and the journal call this director
        # The random persona has no parameters of its own: it lives by those of the persona that made its yoke
        # log (R3). A live run reads them from that log (before anything is written); a replay needs none of
        # them, and takes what its own header recorded.
        self.yoke: yoked.Yoke | None = None  # the random persona's yoke log (live mode only)
        if self.name != RANDOM_PERSONA:
            self.persona = params.persona(self.name)
            self.budget = rules.effective_budget(cfg.director.budget_per_100k, self.persona)
        elif mode == "live":
            self.yoke = yoked.load(cfg, params)
            if log.rules_version_of(records) is None:  # a new run follows the rules of the log it is yoked to
                params = self.params = self._yoked_params(self.yoke, params)
            self.persona = params.persona(self.yoke.persona)
            self.budget = yoked.effective_budget(cfg.director.budget_per_100k, self.yoke, self.persona)
        else:
            name, budget = _recorded(records)
            self.persona = params.persona(name)
            self.budget = rules.effective_budget(budget or cfg.director.budget_per_100k, self.persona)
        self.applied: list[dict[str, Any]] = []
        self.llm_line: str | None = None  # the console line of this boundary's LLM call (if it made one)
        # The LLM tier exists in live mode only: a replay never builds a client, so it cannot reach a network.
        self.llm = llm.Client.of(cfg) if mode == "live" and cfg.director.llm_every > 0 else None
        # Rules noted (blocked, or seasonal_exemption) at the previous boundary; saved at each checkpoint,
        # restored by a resume.
        self._blocked: frozenset[str] = self._saved_blocked(start_step) if mode == "live" else frozenset()
        self.entries = log.entries_of(records)
        # The player channel (G3b): requests queued by the pacer, what became of the latest ones, and the
        # earliest end of a running player's effect (the step at which its revert is due; None when none
        # runs).
        self._queue: list[PlayerRequest] = []
        self._applied_player: list[dict[str, Any]] = []  # applied since the last ``take_player``
        self._refused: list[tuple[PlayerRequest, str]] = []
        self._player_row: list[dict[str, Any]] = []  # applied during the interval of the latest row
        self._player_new: list[dict[str, Any]] = []  # applied since the latest row
        self._last_row: tuple[int, Any] = (start_step, None)  # (step, n_alive) of the latest metrics row
        self._player_due: int | None = None
        self._held_due: int | None = None  # the earliest end of a rule's or the model's effect (player mode)
        if mode == "replay":
            log.validate_log(records, cfg)
            self.pending: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
            self.pending_player: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
            # Entries up to the start step are already in the loaded state. Rule P: the checkpoint at the
            # start step holds a boundary entry at it but not a player's record (applied after it).
            for e in self.entries:
                if actions.is_player(e):
                    if e["step_applied"] >= start_step:
                        self.pending_player[e["step_applied"]].append(e)
                elif e["step_applied"] > start_step:
                    self.pending[e["step_applied"]].append(e)
        else:
            rows = rd.metrics_rows()
            self._last_row = next(
                ((r["step"], r.get("n_alive")) for r in reversed(rows) if r["step"] <= start_step),
                (start_step, None),
            )
            if mode == "live":
                self.tracker = Tracker(params)
                self._rebuild(rows, start_step)
                self._check_yoke(records)
            self._refresh_player_due(start_step)
            header = log.make_header(
                cfg,
                start_step,
                self.budget,
                self.yoke.sha256 if self.yoke else None,
                self.persona.name if self.yoke else None,
                self.params.rules_version,
                game_specs=self.game_specs,
            )
            if log.needs_header(records, header):
                rd.append_jsonl(DIRECTOR_LOG_FILE, [header])

    @staticmethod
    def _resume_params(rd: RunDir, records: Sequence[Mapping[str, Any]], params: RuleParams) -> RuleParams:
        """The rule parameters a live director runs under: those of the version of the log's latest header,
        so a resumed run keeps the rules it began with (a log of a newer, unknown version is ``LogError``);
        a new log takes ``params`` as given. A note on stderr says when that is not the current version."""
        logged = log.rules_version_of(records)
        if logged is None or logged == params.rules_version:
            return params
        print(
            f"note: {rd.path / DIRECTOR_LOG_FILE} was written under rules version {logged}; this run "
            f"continues under it (the current version is {params.rules_version}, which a new run or a "
            "fork takes)",
            file=sys.stderr,
            flush=True,
        )
        return rules.params_for(params, logged)

    @staticmethod
    def _yoked_params(yoke: yoked.Yoke, params: RuleParams) -> RuleParams:
        """The rule parameters a *new* random-persona run lives by: those of the rules version of its yoke
        log's first header (R3), so that the safety rules R runs are the ones the yoked run ran and an arm R
        started against a version-1 log does not run version 2's season-aware rescue. A note on stderr says
        when that is not the current version. A resumed R run keeps its own log's version instead
        (``_resume_params``)."""
        if yoke.rules_version == params.rules_version:
            return params
        print(
            f"note: {yoke.path} was written under rules version {yoke.rules_version}; this run follows it "
            f"under the same rules (the current version is {params.rules_version}, which a run not yoked to "
            "an older log takes)",
            file=sys.stderr,
            flush=True,
        )
        return rules.params_for(params, yoke.rules_version)

    def _saved_blocked(self, start_step: int) -> frozenset[str]:
        """The blocked-rule set saved with the archive of the checkpoint at ``start_step``; empty if the
        archive is missing, unreadable, of another step (stale: written at an earlier checkpoint, or by a run
        that went on past this one) or older than the field. The run then notes a still-blocked rule again."""
        try:
            saved = json.loads((self.rd.path / DIRECTOR_ARCHIVE_FILE).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return frozenset()
        blocked = (
            saved.get("blocked")
            if isinstance(saved, dict) and saved.get("as_of_step") == start_step
            else None
        )
        ok = isinstance(blocked, list) and all(isinstance(b, str) for b in blocked)
        return frozenset(blocked) if ok else frozenset()

    def _check_yoke(self, records: Sequence[Mapping[str, Any]]) -> None:
        """Warn if the yoke log is not the file this run was started with (its header says which): the
        run goes on, since the header records provenance and a new header records the file now in use. Also
        warn if the yoked run changed persona on a resume: R keeps the persona it started with."""
        if self.yoke is None:
            return
        if len(self.yoke.personas) > 1:
            print(
                f"warning: {self.yoke.path} has headers of several personas {list(self.yoke.personas)}; "
                f"using the parameters of {self.yoke.persona!r}, the first",
                file=sys.stderr,
                flush=True,
            )
        then = next((r["yoke_sha256"] for r in reversed(records) if r.get("yoke_sha256")), None)
        if then is not None and then != self.yoke.sha256:
            print(
                f"warning: {self.yoke.path} has changed since this run started following it "
                f"(sha256 {then[:12]}... then, {self.yoke.sha256[:12]}... now); continuing",
                file=sys.stderr,
                flush=True,
            )

    def _rebuild(self, rows: Sequence[Mapping[str, Any]], start_step: int) -> None:
        """Fold the stored rows up to the start step into the tracker, as live mode did (A7)."""
        for row in rows:
            if row["step"] <= start_step:
                self.observe(row)
                if self.due(row["step"]):
                    self.tracker.close(row["step"])

    # --- the runner's hooks -------------------------------------------------------------------------------

    def due(self, t: int) -> bool:
        """Whether ``t`` is a boundary. The player mode decides nothing, but a world launched in it may hold a
        timed effect of the rule tier or the model (last played under the director, or an adopted research
        run): the boundary at or after the effect's end is one, so that its revert is written as it would be
        in ``live`` mode."""
        if self.mode == "player":
            return self._held_due is not None and t >= self._held_due and t % self.rule_every == 0
        return t % self.rule_every == 0

    def observe(self, row: Mapping[str, Any]) -> None:
        """Feed one metrics row (live mode; a replay needs no vital signs). Starts a new row: the entries of
        an earlier boundary are no longer reported by ``status_lines``."""
        self.applied, self.llm_line = [], None
        self._player_row, self._player_new = self._player_new, []
        if self.mode == "live":
            self.tracker.observe(row)
        if self.mode != "replay":
            self._last_row = (row["step"], row.get("n_alive"))

    def on_boundary(self, state: State, row: Mapping[str, Any], t: int) -> State:
        """The state after this boundary's interventions: decided from ``row`` (live) or read from the log
        (replay). In live mode the entries are durable in ``director.jsonl`` before the state is returned."""
        if row["step"] != t:
            raise ValueError(f"the boundary at step {t} was given the row of step {row['step']}")
        self.applied, self.llm_line = [], None
        if self.mode == "replay":
            return self._replay(state, t)
        if self.mode == "player":  # decides nothing; ends what the director began
            return self._end_held(state, t)
        return self._decide(state, t)

    def status_lines(self) -> list[str]:
        """One line per entry applied at the latest boundary (and one for its LLM call), after the player's
        applied during the interval of the latest row, for the console."""
        lines = [f"director @ {e['step_applied']}: {describe(e)}" for e in [*self._player_row, *self.applied]]
        return [self.llm_line, *lines] if self.llm_line else lines

    def on_checkpoint(self, t: int) -> None:
        """Write ``director_archive.json``: the archive for analysis (never read back) and ``blocked``, the
        rules that were noted (blocked, or ``seasonal_exemption``) at the latest boundary, whether or not that
        boundary journalled the note (a persisting one is journalled only when it first appears), which a
        resume from this checkpoint restores. Replays and the player mode (no tracker) write nothing."""
        if self.mode == "live":
            self.rd.write_json(
                DIRECTOR_ARCHIVE_FILE, {**self.tracker.archive.to_json(t), "blocked": sorted(self._blocked)}
            )

    # --- the player channel -------------------------------------------------------------------------------

    def queue_player(self, request: PlayerRequest) -> None:
        """Queue a power (or its cancellation) for the next ``apply_pending`` (pure Python: the pacer
        enqueues, the runner applies). The caller has asked ``plan_player`` first; the request is planned
        again when it is applied, and a refusal then is reported by ``take_player``, never an exception in
        the loop."""
        if self.mode == "replay":
            raise actions.ActionError("a replay applies the log: it takes no new power")
        if request.power not in actions.GAME_POWERS:
            raise actions.ActionError(f"unknown power {request.power!r}")
        self._queue.append(request)

    def has_queued(self) -> bool:
        return bool(self._queue)

    def take_player(self) -> tuple[list[dict[str, Any]], list[tuple[PlayerRequest, str]]]:
        """The player's entries applied (powers, and the reverts that ended or cancelled them) and the
        requests refused since the last call, which empties both."""
        out = (self._applied_player, self._refused)
        self._applied_player, self._refused = [], []
        return out

    def plan_player(self, state: State, request: PlayerRequest, t: int) -> list[dict[str, Any]]:
        """The entries ``request`` would make if it were applied at step ``t`` to ``state`` (a dry run that
        changes nothing, for the preview and the checks before queueing): the queue is folded in first, in
        order, so influence and levers already promised are not promised twice. ``ActionError`` says why
        not."""
        if self.mode == "replay":
            raise actions.ActionError("a replay applies the log: it takes no new power")
        work, new = state, []
        for queued in self._queue:
            try:
                work, entries = self._player_step(work, queued, t, new)
            except actions.ActionError:
                continue  # it will be refused when it is applied; it holds nothing in the meantime
            new.extend(entries)
        return self._player_step(work, request, t, new)[1]

    def apply_pending(self, state: State, t: int, *, stopping: bool = False) -> State:
        """The state at the top of loop iteration ``t`` after the player channel: in live and player mode the
        reverts that end the player's effects (those due at ``t``) and then the queued requests, in order,
        each planned again against the state as it is; in replay mode the logged records of the channel at
        ``t``. What is applied is durable in ``director.jsonl`` before the state is returned. A refused
        request is left for ``take_player``. With ``stopping`` (the pacer asked for a stop, or SIGINT is
        latched; the loop still runs one more step) the queue is dropped unlogged and no new power is
        applied, so a quit never logs a power for a world that is going away; the ends of running effects
        and a replay's records still apply, so a stopped and resumed world equals an uninterrupted one.
        """
        if self.mode == "replay":
            batch = self.pending_player.pop(t, None)
            if not batch:
                return state
            for entry in batch:
                state = actions.apply(state, entry, self.cfg)
            self._note_player(batch)
            return state
        if stopping:
            self._queue.clear()
        if not self._queue and (self._player_due is None or t < self._player_due):
            return state
        queue, self._queue = self._queue, []
        work, new = state, []
        for eff in pacing.due_reverts(pacing.fold(self.entries, t), t, "player"):
            clamped = actions.Clamped(dict(eff.restore), {}, None)
            reason = PLAYER_REASON["end"].format(origin=eff.origin)
            entry = self._player_entry(t, "revert", eff.action, clamped, reason, reverts=eff.origin)
            work = self._apply(work, entry)
            new.append(entry)
        for request in queue:
            try:
                work, entries = self._player_step(work, request, t, new)
            except actions.ActionError as e:
                self._refused.append((request, str(e)))
                continue
            new.extend(entries)
        if new:
            self.rd.append_jsonl(DIRECTOR_LOG_FILE, new)  # durable before the state is handed back
            self.entries = [*self.entries, *new]
            self._note_player(new)
        self._refresh_player_due(t)
        return work

    def _note_player(self, entries: Sequence[dict[str, Any]]) -> None:
        self._applied_player.extend(entries)
        self._player_new.extend(entries)

    def _refresh_player_due(self, t: int) -> None:
        """The earliest end of an effect the player holds at ``t``, and (player mode) of one the rule tier or
        the model holds."""
        active = pacing.fold(self.entries, t).active.values()
        self._player_due = min((e.end for e in active if e.scope == "player"), default=None)
        self._held_due = min((e.end for e in active if e.scope != "player"), default=None)

    def _end_held(self, state: State, t: int) -> State:
        """The player mode's boundary: the reverts of the rule tier's and the model's timed effects that are
        due, written and applied as ``live`` mode writes them (same step, same entry; the vitals are the
        latest row's, there is no tracker). Nothing else is decided."""
        work, new = state, []
        for eff in pacing.due_reverts(pacing.fold(self.entries, t), t, "rule"):
            clamped = actions.Clamped(dict(eff.restore), {}, None)
            step, n_alive = self._last_row
            entry = {
                "step_applied": t,
                "source": "revert",
                "persona": self.name,
                "action": eff.action,
                "params": clamped.params,
                "prev": clamped.prev,
                "duration": None,
                "reason": f"end of the effect from step {eff.origin}",
                "vitals_step": step,
                "vitals": {"n_alive": n_alive},
                "reverts": eff.origin,
            }
            entry = json.loads(json.dumps(entry, allow_nan=False))
            work = self._apply(work, entry)
            new.append(entry)
        if new:
            self.rd.append_jsonl(DIRECTOR_LOG_FILE, new)  # durable before the state is handed back
            self.entries = [*self.entries, *new]
            self.applied = new
        self._refresh_player_due(t)
        return work

    def _player_step(
        self, work: State, request: PlayerRequest, t: int, new: Sequence[Mapping[str, Any]]
    ) -> tuple[State, list[dict[str, Any]]]:
        """``request`` planned and applied to ``work`` at ``t``, given the entries ``new`` of this call so
        far: ``(the new state, the entries made)``, nothing if ``ActionError`` is raised (all or none of a
        cancel's reverts)."""
        entries = self._player_entries(work, request, t, new)
        for entry in entries:
            work = self._apply(work, entry)
        return work, entries

    def _player_entries(
        self, work: State, request: PlayerRequest, t: int, new: Sequence[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        action = actions.GAME_POWERS.get(request.power)
        if action is None:
            raise actions.ActionError(f"unknown power {request.power!r}")
        so_far = [*self.entries, *new]
        if request.cancel:
            raw_patch = request.raw.get("patch")
            ending = sorted(
                (
                    eff
                    for eff in pacing.fold(so_far, t).active.values()
                    if eff.scope == "player"
                    and eff.action == action
                    and (raw_patch is None or eff.key == f"patch:{raw_patch}")
                ),
                key=lambda eff: (eff.end, eff.key),
            )
            if not ending:
                raise actions.ActionError(f"no effect of {request.power} to cancel")
            return [
                self._player_entry(
                    t,
                    "revert",
                    eff.action,
                    actions.Clamped(dict(eff.restore), {}, None),
                    PLAYER_REASON["cancel"],
                    command=request.command,
                    reverts=eff.origin,
                )
                for eff in ending
            ]
        clamped = actions.clamp(
            action, request.raw, request.duration, actions.env_view(work), self.cfg, specs=actions.GAME_SPECS
        )
        why = pacing.player_blocked(
            pacing.fold(so_far, t, "player"), pacing.fold(so_far, t), action, clamped.params, t
        )
        if why is not None:
            raise actions.ActionError(why)
        reason = PLAYER_REASON["power"].format(power=request.power)
        return [
            self._player_entry(
                t, "player", action, clamped, reason, command=request.command, power=request.power
            )
        ]

    def _player_entry(
        self,
        t: int,
        source: str,
        action: str,
        clamped: actions.Clamped,
        reason: str,
        *,
        command: int | None = None,
        power: str | None = None,
        reverts: int | None = None,
    ) -> dict[str, Any]:
        """A record of the player channel in its canonical JSON form. The vitals are the latest row's (there
        is no tracker in the player mode)."""
        step, n_alive = self._last_row
        entry = {
            "step_applied": t,
            "source": source,
            "persona": self.name,
            "action": action,
            "params": clamped.params,
            "prev": clamped.prev,
            "duration": clamped.duration,
            "reason": reason,
            "vitals_step": step,
            "vitals": {"n_alive": n_alive},
            "scope": "player",
            **({"command": command} if command is not None else {}),
            **(
                {"power": power, "specs": actions.GAME_SPEC_VERSION, "cost": actions.GAME_SPECS[action].cost}
                if power is not None
                else {}
            ),
            **({"reverts": reverts} if reverts is not None else {}),
        }
        return json.loads(json.dumps(entry, allow_nan=False))

    # --- replay -------------------------------------------------------------------------------------------

    def _replay(self, state: State, t: int) -> State:
        self.applied = list(self.pending.pop(t, []))  # the boundary's: the player's are ``apply_pending``'s
        for entry in self.applied:
            state = actions.apply(state, entry, self.cfg)
        return state

    # --- live ---------------------------------------------------------------------------------------------

    def _vitals(self, state: State, t: int) -> Vitals:
        bd = self.tracker.close(t)
        k_est = metrics.carrying_capacity(state, self.cfg, state.step, state.env)
        # 3D: the extinction-risk test reads the floor layer's K, every other consumer the occupancy-weighted
        # one (§27.10). 2D has one K, and its vitals carry no second.
        k_floor = None
        if is_3d(self.cfg):
            k_floor = metrics.floor_carrying_capacity(state, self.cfg, state.step, state.env)
        # Rules version 2: with seasons in force in the fall window, the capacity over one cycle is sampled
        # through the engine so that the vitals can tell the fall the seasons force from the rest
        # (seasons.py). Without seasons, or under version 1, nothing is computed (the vitals of version 1).
        shape = None
        if self.params.season_aware and (amp := self.tracker.season_amp(bd)) > 0:
            table = metrics.seasonal_capacity_table(state, self.cfg, state.env, amp, seasons.TABLE_SIZE)
            shape = seasons.CapacityShape(amp, tuple(float(x) for x in table))
        return self.tracker.vitals(bd, k_est, dims(self.cfg).n_cap, k_floor, shape)

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
        yoked_to: int | None = None,
    ) -> dict[str, Any]:
        """A log entry in its canonical JSON form (what a replay reads back is exactly what is applied)."""
        entry = {
            "step_applied": v.step,
            "source": source,
            "persona": self.name,
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
            **({"yoked_to": yoked_to} if yoked_to is not None else {}),
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
            actions.span(self.cfg),
            v,
            pacing.fold([*self.entries, *new], t, "rule"),
            actions.env_view(work),
            actions.is_sphere(self.cfg),
        )

    def _decide(self, state: State, t: int) -> State:
        """Reverts, then the safety rules, then (at an LLM boundary) the model or, failing that, the rule
        tier's discretionary rules; or, for the random persona, its yoked draws. With the LLM tier off the
        rule tier alone decides, as in step 2."""
        v = self._vitals(state, t)
        new: list[dict[str, Any]] = []
        notes: list[str] = []
        records: list[dict[str, Any]] = []
        work = state
        for eff in pacing.due_reverts(pacing.fold(self.entries, t, "rule"), t, "rule"):
            clamped = actions.Clamped(dict(eff.restore), {}, None)
            reason = f"end of the effect from step {eff.origin}"
            entry = self._entry(v, "revert", eff.action, clamped, reason, reverts=eff.origin)
            work = self._apply(work, entry)
            new.append(entry)
        ctx = self._context(work, v, t, new)
        # The LLM tier takes the discretionary moves from the rule tier (L2), and the random persona has none:
        # only the safety rules remain.
        safety_only = self.llm is not None or self.yoke is not None
        decision = (rules.decide_safety if safety_only else rules.decide)(ctx, self.params)
        notes.extend(decision.notes)
        work = self._take_all(work, v, decision.proposals, new, notes)
        if self.llm is not None and t % self.cfg.director.llm_every == 0:
            work = self._consult(work, v, t, new, notes, records)
        if self.yoke is not None and t in self.yoke.steps:
            work = self._yoked_moves(work, v, t, new, notes)
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
        yoked_to: int | None = None,
    ) -> tuple[State, dict[str, Any]]:
        """Clamp and apply one proposal to ``work``; the entry that results (``source`` is ``llm`` for a
        proposal of the model, whose entry carries the call id and no rule; a yoked move of the random
        persona carries the step it is yoked to)."""
        active = pacing.fold(entries, v.step).active
        restore = active["regen"].restore if prop.action == "regen_pulse" and "regen" in active else None
        clamped = actions.clamp(
            prop.action, prop.raw, prop.duration, actions.env_view(work), self.cfg, restore
        )
        if llm_call is None:
            entry = self._entry(
                v, "rule", prop.action, clamped, prop.reason, rule=prop.rule, yoked_to=yoked_to
            )
        else:
            entry = self._entry(v, "llm", prop.action, clamped, prop.reason, llm_call=llm_call)
        return self._apply(work, entry), entry

    def _apply(self, work: State, entry: Mapping[str, Any]) -> State:
        """``work`` after ``entry``, which first passes the replay validation: the director never acts on
        something a replay would reject."""
        actions.validate_entry(entry, self.cfg, self.rule_every)
        return actions.apply(work, entry, self.cfg)

    # --- the random persona -------------------------------------------------------------------------------

    def _yoked_moves(
        self, work: State, v: Vitals, t: int, new: list[dict[str, Any]], notes: list[str]
    ) -> State:
        """Make the moves the yoke log made at this step (R3): each is a uniformly random action among those
        eligible now, with random parameters, taken through the normal path (clamp, validate, apply) and
        followed by a refold of the pacing. When nothing is eligible a note is journalled and the remaining
        moves of the step are not made."""
        assert self.yoke is not None
        k = self.yoke.steps[t]
        for i in range(k):
            ctx = self._context(work, v, t, new)
            options, pools, skipped = yoked.candidates(ctx, self.params, self.cfg)
            rng = yoked.rng_for(self.cfg.run.seed, t, i)
            while options:
                action = options[int(rng.integers(len(options)))]
                raw, duration = yoked.sample_params(action, rng, self.cfg, pools)
                prop = rules.Proposal(action, raw, duration, yoked.RULE, f"yoked draw {i + 1} of {k}")
                try:
                    work, entry = self._take(work, v, prop, [*self.entries, *new], yoked_to=t)
                except actions.ActionError as e:  # the clamp refused it: nothing was applied; try another
                    notes.append(f"yoked step {t}: {action} dropped ({e})")
                    options.remove(action)
                    continue
                new.append(entry)
                break
            else:
                why = f" [{'; '.join(skipped)}]" if skipped else ""
                notes.append(f"yoked step {t}: nothing eligible ({k - i} of {k} moves not made){why}")
                break
        return work

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
                "persona": self.name,
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
        lines = [{"step": t, "persona": self.name, "text": describe(e)} for e in entries]
        lines += [{"step": t, "persona": self.name, "text": n} for n in fresh]
        lines += records
        if lines:
            self.rd.append_jsonl(JOURNAL_FILE, lines)


def make_director(
    rd: RunDir, cfg: Config, start_step: int, params: RuleParams = RULE_PARAMS, *, game_specs: bool = False
) -> Director | None:
    """The director for ``cfg.run.director`` (None when ``off``), built from the run directory's files.
    ``game_specs`` is ``alife play``'s: its log may carry the player's records."""
    mode = cfg.run.director
    return None if mode == "off" else Director(rd, cfg, mode, start_step, params, game_specs=game_specs)
