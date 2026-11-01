"""The LLM tier's request, reply and transport (DIRECTOR.md §7; build step 3, decisions L1-L11).

Three small pieces, all host-only and independent of the simulation:

- **The schema** (``intervention_schema``) is generated from ``actions.MENU`` and the config (patch count,
  world size), so the bounds the model is shown are the bounds the clamp enforces. Each action is one
  ``anyOf`` variant with exact parameter keys, numeric ``minimum`` / ``maximum`` and a ``maxLength`` on every
  string: probes of local models found that without ``maxLength`` they loop inside a string until
  ``max_tokens``, and without numeric bounds they emit runaway integers.
- **The reply parser** (``parse_reply``) treats every byte of the reply as untrusted data. It checks the
  structure the server may not enforce (exact keys, types, finite numbers, string lengths, duplicates) and
  yields menu selections only; nothing is evaluated and nothing but a menu name and its numbers survives.
  Numeric *ranges* are not rejected here: the step-2 clamp brings a number inside its bounds (and the journal
  records it), and refuses what it cannot make sense of (a patch that does not exist).
- **The client** (``Client``) POSTs one chat request to the configured URL with stdlib ``urllib``. It only
  speaks http(s), ignores proxy variables, refuses every redirect, caps the reply size and the wall time, and
  never raises: any failure is returned as an ``Exchange`` whose ``error`` makes the director fall back to the
  rule tier. With the tier off (``director.llm_every = 0``) no ``Client`` is ever built.
"""

from __future__ import annotations

import json
import math
import time
import unicodedata
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from http.client import HTTPException
from typing import Any, Final, NamedTuple

import numpy as np

from alife.config import Config, ConfigError, is_http_url
from alife.director import actions
from alife.director.actions import MENU

NONE: Final = "none"
MAX_ACTIONS: Final = 2
EFFECT_MAX: Final = 240  # characters of an action's ``expected_effect``
JOURNAL_MAX: Final = 600  # characters of the reply's ``journal``
MAX_REPLY_BYTES: Final = 1 << 20  # the largest HTTP response body that is read
SCHEMA_NAME: Final = "intervention"
_SEED_TAG: Final = 16  # the stream tag of the request seed (``rules._TAGS`` use 0-4 and 1-9)


class Param(NamedTuple):
    """One parameter of a menu action: its name, whether it is an integer, and its inclusive bounds."""

    name: str
    integer: bool
    lo: float
    hi: float


# --- the menu as the model sees it -----------------------------------------------------------------------


def param_specs(cfg: Config) -> dict[str, tuple[Param, ...]]:
    """The parameters of every menu action (not ``none``), with bounds taken from ``actions.MENU``.

    ``fertility_shift`` is offered in its gain form only (``patch``, ``gain``): the patch *move* stays a
    rule-tier action, so that every menu action is exactly one schema variant. ``habitat_loss`` and
    ``regen_pulse`` carry their ``duration`` as a parameter; the others carry none.
    """

    def bound(action: str, name: str, integer: bool = False) -> Param:
        lo, hi = MENU[action].bounds[name]
        return Param(name, integer, lo, hi)

    def duration(action: str) -> Param:
        lo, hi = MENU[action].duration or (0, 0)
        return Param("duration", True, lo, hi)

    size = actions.span(cfg)
    # a place: the torus' [0, size) twice, or a sphere's longitude and latitude in degrees (DESIGN §46.7)
    where = (
        (Param("x", False, -180.0, 180.0), Param("y", False, -90.0, 90.0))
        if actions.is_sphere(cfg)
        else (Param("x", False, 0.0, size), Param("y", False, 0.0, size))
    )
    patch = Param("patch", True, 0, cfg.fields.n_patches - 1)
    radius = Param("radius", False, *actions.radius_bounds(cfg))
    return {
        "set_seasons": (bound("set_seasons", "period", True), bound("set_seasons", "amp")),
        "fertility_shift": (patch, bound("fertility_shift", "gain")),
        "habitat_loss": (patch, duration("habitat_loss")),
        "regen_pulse": (bound("regen_pulse", "mult"), duration("regen_pulse")),
        "immigration_pulse": (bound("immigration_pulse", "count", True),),
        "catastrophe": (
            *where,
            radius,
            bound("catastrophe", "kill_p"),
        ),
        "drift_change": (bound("drift_change", "mult"),),
    }


def _number(x: float, integer: bool) -> float | int:
    return int(x) if integer else float(x)


def _variant(name: str, params: Sequence[Param]) -> dict[str, Any]:
    props = {
        p.name: {
            "type": "integer" if p.integer else "number",
            "minimum": _number(p.lo, p.integer),
            "maximum": _number(p.hi, p.integer),
        }
        for p in params
    }
    return {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": [name]},
            "params": {
                "type": "object",
                "properties": props,
                "required": [p.name for p in params],
                "additionalProperties": False,
            },
            "expected_effect": {"type": "string", "maxLength": EFFECT_MAX},
        },
        "required": ["action", "params", "expected_effect"],
        "additionalProperties": False,
    }


def intervention_schema(cfg: Config) -> dict[str, Any]:
    """The JSON schema of the reply: one or two actions (``anyOf`` one variant per menu action plus
    ``none``) and a journal."""
    variants = [_variant(name, params) for name, params in param_specs(cfg).items()]
    variants.append(_variant(NONE, ()))
    return {
        "type": "object",
        "properties": {
            "actions": {
                "type": "array",
                "minItems": 1,
                "maxItems": MAX_ACTIONS,
                "items": {"anyOf": variants},
            },
            "journal": {"type": "string", "maxLength": JOURNAL_MAX},
        },
        "required": ["actions", "journal"],
        "additionalProperties": False,
    }


def response_format(cfg: Config) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {"name": SCHEMA_NAME, "strict": True, "schema": intervention_schema(cfg)},
    }


def request_seed(seed: int, t: int) -> int:
    """The ``seed`` request parameter: a deterministic non-negative int32 of ``(run.seed, step)``. A server
    may ignore it; it never feeds the simulation."""
    return int(np.random.SeedSequence([seed, t, _SEED_TAG]).generate_state(1)[0]) & 0x7FFFFFFF


def clean_text(text: str, limit: int) -> str:
    """``text`` with control and format characters dropped and whitespace collapsed, cut to ``limit``
    characters. Model text is data: this keeps escape sequences out of the console and of logs a GUI shows."""
    kept = "".join(
        " " if ch.isspace() else ch for ch in text if ch.isspace() or unicodedata.category(ch)[0] != "C"
    )
    return " ".join(kept.split())[:limit]


# --- the reply ---------------------------------------------------------------------------------------------


class ReplyError(ValueError):
    """A reply that is not valid against the schema (the director then falls back to the rule tier)."""


class Candidate(NamedTuple):
    """One requested menu action in the form of a rule's ``Proposal`` (``duration`` apart from ``raw``)."""

    action: str
    raw: dict[str, Any]
    duration: int | None
    effect: str  # the model's ``expected_effect``, cleaned


class Reply(NamedTuple):
    candidates: tuple[Candidate, ...]  # in the model's order, duplicates and ``none`` removed
    explicit_none: bool  # the model chose ``none`` (alone): a decision to do nothing, not a failure
    journal: str  # the model's account, cleaned
    notes: tuple[str, ...]  # what validation dropped or ignored


def _no_constant(name: str) -> Any:
    raise ReplyError(f"the JSON constant {name} is not allowed")


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    keys = [key for key, _ in pairs]
    if len(set(keys)) != len(keys):
        raise ReplyError(f"duplicate key {next(k for k in keys if keys.count(k) > 1)!r}")
    return dict(pairs)


def _object(where: str, value: Any, keys: set[str]) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ReplyError(f"{where} must be an object, got {type(value).__name__}")
    if set(value) != keys:
        raise ReplyError(f"{where} must have exactly the keys {sorted(keys)}, got {sorted(value)}")
    return value


def _string(where: str, value: Any, limit: int) -> str:
    if not isinstance(value, str):
        raise ReplyError(f"{where} must be a string, got {type(value).__name__}")
    if len(value) > limit:
        raise ReplyError(f"{where} has {len(value)} characters (limit {limit})")
    return value


def _finite(value: float) -> bool:
    try:
        return math.isfinite(value)
    except OverflowError:  # an integer too large for a float
        return False


def _param_value(where: str, p: Param, value: Any) -> float | int:
    if isinstance(value, bool) or not isinstance(value, int | float) or not _finite(value):
        raise ReplyError(f"{where}.{p.name} must be a finite number, got {value!r}")
    if p.integer:
        if value != int(value):
            raise ReplyError(f"{where}.{p.name} must be an integer, got {value!r}")
        return int(value)
    return float(value)


def _candidate(i: int, item: Any, specs: Mapping[str, Sequence[Param]]) -> tuple[str, dict[str, Any], str]:
    where = f"actions[{i}]"
    obj = _object(where, item, {"action", "params", "expected_effect"})
    name = obj["action"]
    if not isinstance(name, str) or (name != NONE and name not in specs):
        raise ReplyError(f"{where}.action {name!r} is not a menu action")
    effect = clean_text(_string(f"{where}.expected_effect", obj["expected_effect"], EFFECT_MAX), EFFECT_MAX)
    params = specs.get(name, ())
    given = _object(f"{where}.params", obj["params"], {p.name for p in params})
    return name, {p.name: _param_value(where, p, given[p.name]) for p in params}, effect


def parse_reply(text: str, cfg: Config) -> Reply:
    """The reply's menu selections, or ``ReplyError``.

    - The text must be one JSON object ``{"actions": [...], "journal": "..."}`` with exactly those keys, one
      or two actions (``params`` with exactly the action's keys, finite numbers, integers where the menu has
      them, strings within the schema's ``maxLength``), no duplicate keys and no ``NaN`` / ``Infinity``.
    - A repeated action name is dropped (the first wins). ``none`` next to another action is ignored and the
      other action kept; ``none`` alone is a valid decision to do nothing.
    - Numeric ranges are *not* checked: ``actions.clamp`` brings them inside the menu.
    """
    if len(text.encode("utf-8", "replace")) > MAX_REPLY_BYTES:
        raise ReplyError(f"the reply is larger than {MAX_REPLY_BYTES} bytes")
    try:
        data = json.loads(text, parse_constant=_no_constant, object_pairs_hook=_unique)
    except (ValueError, RecursionError) as e:
        raise ReplyError(f"not valid JSON ({e})") from e
    root = _object("the reply", data, {"actions", "journal"})
    journal = clean_text(_string("journal", root["journal"], JOURNAL_MAX), JOURNAL_MAX)
    items = root["actions"]
    if not isinstance(items, list) or not 1 <= len(items) <= MAX_ACTIONS:
        raise ReplyError(f"actions must be a list of 1 to {MAX_ACTIONS} items")
    specs = param_specs(cfg)
    parsed = [_candidate(i, item, specs) for i, item in enumerate(items)]
    notes: list[str] = []
    seen: set[str] = set()
    chosen: list[Candidate] = []
    for name, raw, effect in parsed:
        if name in seen:
            notes.append(f"{name} repeated: the duplicate was dropped")
            continue
        seen.add(name)
        if name != NONE:
            duration = raw.pop("duration", None)
            chosen.append(Candidate(name, raw, duration, effect))
    if NONE in seen and chosen:
        notes.append("none was combined with another action: none was ignored")
    return Reply(tuple(chosen), NONE in seen and not chosen, journal, tuple(notes))


def clamps(candidate: Candidate, entry: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The requested numbers that the clamp changed (rounded to float32 or a rule boundary included), for the
    journal: ``[{"param", "requested", "applied"}]``."""
    applied = dict(entry["params"])
    if "center" in applied:
        applied = {**applied, "x": applied["center"][0], "y": applied["center"][1]}
    if entry.get("duration") is not None:
        applied = {**applied, "duration": entry["duration"]}
    requested = {
        **candidate.raw,
        **({"duration": candidate.duration} if candidate.duration is not None else {}),
    }
    return [
        {"param": name, "requested": value, "applied": applied[name]}
        for name, value in requested.items()
        if name in applied and not math.isclose(value, applied[name], rel_tol=1e-6, abs_tol=1e-9)
    ]


# --- the transport -----------------------------------------------------------------------------------------


class Exchange(NamedTuple):
    """The outcome of one request: the reply text, or an ``error`` of the form ``"<kind>: <detail>"`` with
    ``kind`` one of ``timeout``, ``connection``, ``http``, ``protocol``, ``oversize`` or ``empty``."""

    text: str | None
    error: str | None
    latency_s: float
    finish_reason: str | None
    usage: dict[str, int] | None


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """The URL is user config: the request goes to that URL only, never to a host a response names."""

    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None  # urllib then raises the 3xx response as an HTTPError


class _Failure(Exception):
    def __init__(self, kind: str, detail: str) -> None:
        super().__init__(f"{kind}: {detail}")


def _read_capped(resp: Any, deadline: float) -> bytes:
    """The response body, read in chunks so that a server that trickles bytes still meets the deadline."""
    chunks: list[bytes] = []
    size = 0
    while True:
        if time.monotonic() > deadline:
            raise _Failure("timeout", "the reply did not complete in time")
        chunk = resp.read1(65536)
        if not chunk:
            return b"".join(chunks)
        size += len(chunk)
        if size > MAX_REPLY_BYTES:
            raise _Failure("oversize", f"the response is larger than {MAX_REPLY_BYTES} bytes")
        chunks.append(chunk)


def _usage(value: Any) -> dict[str, int] | None:
    if not isinstance(value, dict):
        return None
    out = {
        k: v
        for k in ("prompt_tokens", "completion_tokens", "total_tokens")
        if isinstance(v := value.get(k), int) and not isinstance(v, bool)
    }
    return out or None


def _envelope(body: bytes) -> tuple[str, str | None, dict[str, int] | None]:
    """``(content, finish_reason, usage)`` of an OpenAI-compatible chat response; ``_Failure`` otherwise."""
    try:
        data = json.loads(body.decode("utf-8"), parse_constant=_no_constant)
        choice = data["choices"][0]
        content = choice["message"]["content"]
    except (ValueError, RecursionError, KeyError, IndexError, TypeError) as e:
        raise _Failure("protocol", f"not a chat completion ({type(e).__name__})") from e
    if content is None or (isinstance(content, str) and not content.strip()):
        raise _Failure("empty", "the reply has no content (a reasoning model with thinking on?)")
    if not isinstance(content, str):
        raise _Failure("protocol", "the reply content is not a string")
    finish = choice.get("finish_reason")
    return content, finish if isinstance(finish, str) else None, _usage(data.get("usage"))


class Client:
    """One chat-completions endpoint. ``chat`` blocks for at most about ``timeout_s`` seconds (L1)."""

    def __init__(self, url: str, model: str, timeout_s: float, max_tokens: int, temperature: float) -> None:
        if not is_http_url(url):
            raise ConfigError(f"run.llm_url must be an http(s) URL, got {url!r}")
        self.url, self.model = url, model
        self.timeout_s, self.max_tokens, self.temperature = timeout_s, max_tokens, temperature
        # No proxies (the request must go to the configured host, whatever the environment says) and no
        # redirects. The other default handlers are never reached: the URL scheme is http(s).
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())

    @classmethod
    def of(cls, cfg: Config) -> Client:
        """The client for ``cfg``'s ``run.llm_*`` and ``director.llm_*`` keys."""
        d = cfg.director
        return cls(cfg.run.llm_url, cfg.run.llm_model, d.llm_timeout_s, d.llm_max_tokens, d.llm_temperature)

    def body(self, messages: Sequence[Mapping[str, str]], cfg: Config, t: int) -> dict[str, Any]:
        """The request body for the boundary at step ``t``."""
        return {
            "model": self.model,
            "messages": list(messages),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "seed": request_seed(cfg.run.seed, t),
            "stream": False,
            "response_format": response_format(cfg),
        }

    def chat(self, body: Mapping[str, Any]) -> Exchange:
        """POST ``body`` to the URL. Never raises: a failure is an ``Exchange`` with ``error`` set."""
        t0 = time.monotonic()
        req = urllib.request.Request(
            self.url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with self._opener.open(req, timeout=self.timeout_s) as resp:
                raw = _read_capped(resp, t0 + self.timeout_s)
            text, finish, usage = _envelope(raw)
        except _Failure as e:
            return Exchange(None, str(e), time.monotonic() - t0, None, None)
        except urllib.error.HTTPError as e:
            return Exchange(None, f"http: status {e.code}", time.monotonic() - t0, None, None)
        except (OSError, HTTPException, ValueError) as e:  # URLError, timeouts, refused or reset connections
            reason = e.reason if isinstance(e, urllib.error.URLError) else e
            kind = "timeout" if isinstance(reason, TimeoutError) else "connection"
            what = type(reason).__name__ if isinstance(reason, BaseException) else "error"
            return Exchange(None, f"{kind}: {what}", time.monotonic() - t0, None, None)
        return Exchange(text, None, time.monotonic() - t0, finish, usage)
