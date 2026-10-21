"""The world report of an observed run (docs/DESIGN.md §30; the N1 step of docs/GAMES.md §9).

::

    python -m alife.analysis.world_report build RUN --from STEP --to STEP --out FILE.jsonl
           [--chapter 6000] [--llm --model ID [--url URL] [--timeout 120] [--attempts 2] [--no-preflight]]
    python -m alife.analysis.world_report verify RUN --reports FILE.jsonl --out FILE.json
    python -m alife.analysis.world_report probe --sheets DIR --model ID [--url URL] --n 24 --out FILE.json

``build`` cuts the run into chapters ``(a, b]`` at the multiples of ``--chapter`` inside
``[--from, --to]`` and writes one record per chapter: the fact sheet, the candidate sentences (all written
by code from the templates of ``alife.game.templates`` and checked again), the report and the ticker.
Without ``--llm`` the report is the template report and nothing touches the network. With ``--llm`` one
request per chapter asks the model to *select and order* the candidates (it writes nothing that is shown);
a reply that is not usable falls back to the template report. ``verify`` replays the stored records without
ever calling a model. ``probe`` measures the select mode on stored sheets (the coordinator runs it, with no
GPU job running).

The run directory is read only and ``--out`` must be outside it. **LLM rule:** the model is named
explicitly in every request (``--model`` is required with ``--llm``, no default); the tool never loads,
unloads or switches a model. The only requests are the ``POST`` to the chat-completions URL and, before any
of them, one read-only ``GET <server>/api/v0/models`` that lists the models and their state (the REST
counterpart of ``lms ps``): the request is refused unless the model is listed as loaded, so a request never
makes the server load one.

Exit codes: 0 ok (or records of other versions, reported); 1 a mismatch under equal versions or a run whose
``env`` rows disagree with the director's log; 2 bad parameters, missing rows, ``--out`` inside the run, a
refused preflight; 3 a stale record (the recomputed sheet differs while every version and table digest is
equal).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import urllib.request
from collections import Counter
from collections.abc import Mapping, Sequence
from http.client import HTTPException
from itertools import pairwise
from pathlib import Path
from typing import Any, NamedTuple
from urllib.parse import urlsplit

from alife.config import ConfigError, RunConfig, is_http_url
from alife.game import report as R
from alife.runio import CheckpointError

PREFLIGHT_PATH = "/api/v0/models"
PREFLIGHT_TIMEOUT_S = 10.0
PREFLIGHT_MAX_BYTES = 1 << 20
DEFAULT_URL = RunConfig.llm_url
OK, MISMATCH, USAGE, STALE = 0, 1, 2, 3


# --- the preflight: is the model loaded? (the owner's LLM rule) ------------------------------------------


class Preflight(NamedTuple):
    """The server's answer about one model: ``status`` is ``loaded``, ``not_loaded``, ``unlisted`` (the model
    is not in the list) or ``no_answer`` (the endpoint did not give a models list)."""

    status: str
    detail: str


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None


def models_url(chat_url: str) -> str:
    """The read-only models listing of the server that serves ``chat_url``."""
    parts = urlsplit(chat_url)
    return f"{parts.scheme}://{parts.netloc}{PREFLIGHT_PATH}"


def preflight(chat_url: str, model: str, timeout_s: float = PREFLIGHT_TIMEOUT_S) -> Preflight:
    """``GET <server>/api/v0/models``: the one request besides the chat ``POST``. It lists models and their
    state and changes nothing. Never raises: a failure is ``no_answer``."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
    req = urllib.request.Request(models_url(chat_url), headers={"Accept": "application/json"}, method="GET")
    try:
        with opener.open(req, timeout=timeout_s) as resp:
            raw = resp.read(PREFLIGHT_MAX_BYTES + 1)
        if len(raw) > PREFLIGHT_MAX_BYTES:
            return Preflight("no_answer", "the models list is larger than 1 MiB")
        listed = json.loads(raw.decode("utf-8"))["data"]
        if not isinstance(listed, list):
            raise TypeError("data is not a list")
    except (OSError, HTTPException, ValueError, KeyError, TypeError, RecursionError) as e:
        return Preflight("no_answer", f"{type(e).__name__}: {str(e)[:100]}")
    entries = [m for m in listed if isinstance(m, dict) and m.get("id") == model]
    if not entries:
        return Preflight("unlisted", f"{model!r} is not in the server's model list")
    state = entries[0].get("state")
    if state == "loaded":
        return Preflight("loaded", "loaded")
    return Preflight("not_loaded", f"{model!r} has state {state!r}, not 'loaded'")


def preflight_refusal(pf: Preflight, model: str, no_preflight: bool) -> str | None:
    """Why the requests may not start (exit 2), or None. The model must be listed as loaded;
    ``--no-preflight`` is accepted only when the endpoint does not answer (another server), else it is
    refused."""
    if pf.status == "no_answer":
        if no_preflight:
            return None
        return (
            f"the server does not answer {PREFLIGHT_PATH} ({pf.detail}); "
            "if it is not LM Studio, pass --no-preflight"
        )
    if no_preflight:
        return f"--no-preflight is refused: the server answers {PREFLIGHT_PATH} ({pf.status})"
    if pf.status != "loaded":
        return f"model {model!r} is not loaded ({pf.detail}); load it by hand (lms ps)"
    return None


# --- the record file -------------------------------------------------------------------------------------


def read_records(path: Path) -> list[Any]:
    """The JSON lines of ``path`` (``[]`` if it does not exist); ``ValueError`` if a line is not JSON."""
    if not path.is_file():
        return []
    out: list[Any] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}: line {i} is not valid JSON ({e})") from e
    return out


def _write_atomic(path: Path, text: str) -> None:
    """Atomic rewrite: a temporary file in the same directory, fsync, then ``os.replace``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def write_records(path: Path, records: Sequence[Any]) -> None:
    lines = (json.dumps(r, allow_nan=False, ensure_ascii=False) + "\n" for r in records)
    _write_atomic(path, "".join(lines))


def write_json(path: Path, obj: Any) -> None:
    _write_atomic(path, json.dumps(obj, indent=1, allow_nan=False, ensure_ascii=False) + "\n")


# --- build -----------------------------------------------------------------------------------------------


def chapter_cuts(start: int, end: int, size: int) -> list[tuple[int, int]]:
    """The chapters ``(a, b]`` of ``[start, end]`` cut at the multiples of ``size`` on the global step
    clock; the first and the last may be partial."""
    if size <= 0:
        raise R.ReportError("--chapter must be positive")
    if not 0 <= start < end:
        raise R.ReportError("--from must be non-negative and below --to")
    cuts = [start, *range((start // size + 1) * size, end, size), end]
    return list(pairwise(cuts))


def keeps(old: Any, facts_digest: str, mode: str, model: str | None, versions: Mapping[str, Any]) -> bool:
    """Whether a stored record is kept by a resumed build: the same chapter and facts, the same mode and
    model and every version and table digest of the code. A template record is rebuilt under ``--llm`` (the
    mode differs) and a select record that never got a model answer (``fallback`` ``invalid``, ``failed``
    or ``too_few``) is rebuilt so that a later ``--llm`` build retries it."""
    if not isinstance(old, dict):
        return False
    if any(old.get(k) != v for k, v in versions.items()):
        return False
    if old.get("facts_digest") != facts_digest or old.get("mode") != mode or old.get("model") != model:
        return False
    return not (mode == "select" and old.get("fallback") in R.RETRY_OUTCOMES)


def _say(msg: str) -> None:
    print(msg, flush=True)


def _warn(msg: str) -> None:
    print(f"warning: {msg}", file=sys.stderr, flush=True)


def _chapter_line(a: int, b: int, rec: Mapping[str, Any], how: str) -> str:
    parts = [f"({a}, {b}] {rec['mode']} {how}"]
    if rec["mode"] == "select":
        parts.append("attempts " + (",".join(x["outcome"] for x in rec["attempts"]) or "none"))
        if rec["fallback"]:
            parts.append(f"fallback {rec['fallback']}")
    return "; ".join(parts)


def _split_records(path: Path) -> tuple[dict[tuple[int, int], dict[str, Any]], list[Any]]:
    """The file's records by ``(chapter_start, step)`` (the last one of a repeated key), and its other lines
    (anything that is not a report record) in file order."""
    keyed: dict[tuple[int, int], dict[str, Any]] = {}
    other: list[Any] = []
    for r in read_records(path):
        ok = (
            isinstance(r, dict) and isinstance(r.get("chapter_start"), int) and isinstance(r.get("step"), int)
        )
        if ok and not isinstance(r["chapter_start"], bool) and not isinstance(r["step"], bool):
            keyed[(r["chapter_start"], r["step"])] = r
        else:
            other.append(r)
    return keyed, other


def pending_chapters(
    sheets: Sequence[Mapping[str, Any]],
    out: Path,
    *,
    mode: str,
    model: str | None,
    rules: R.ReportRules = R.REPORT_V1,
) -> int:
    """How many of ``sheets`` a build over ``out`` would build anew (the others are kept as they are)."""
    old, _other = _split_records(out)
    versions = R.record_versions(rules)
    return sum(
        not keeps(old.get(tuple(s["chapter"])), R.facts_digest(s), mode, model, versions) for s in sheets
    )


def build_reports(
    sheets: Sequence[Mapping[str, Any]],
    out: Path,
    *,
    client: Any | None,
    attempts: int,
    rules: R.ReportRules = R.REPORT_V1,
) -> int:
    """Build (or resume) the records of ``sheets`` (the chapters in order) into ``out``; the exit code.
    ``client`` None is the template mode. The file is rewritten atomically after every chapter, and a rewrite
    keeps everything it does not replace: the records of other chapters (a build of a range never deletes the
    records of another one) and any line that is not a report record, the records sorted by ``(step,
    chapter_start)`` and the other lines after them. Only a record of a chapter of this build is replaced;
    when that drops a select record (a template build over model records) the warning comes first."""
    mode = "template" if client is None else "select"
    model = None if client is None else client.model
    versions = R.record_versions(rules)
    old, other = _split_records(out)
    keys = [tuple(s["chapter"]) for s in sheets]
    if mode == "template":
        n_sel = sum(1 for k in keys if old.get(k, {}).get("mode") == "select")
        if n_sel:
            _warn(
                f"{n_sel} select records of {out} (model replies, not regenerable) will be replaced by "
                "template records"
            )
    done: list[dict[str, Any]] = []
    kept = built = 0
    all_old = dict(old)
    for n, sheet in enumerate(sheets):
        a, b = keys[n]
        prior = old.get((a, b))
        if keeps(prior, R.facts_digest(sheet), mode, model, versions):
            rec, how = prior, "kept"
            kept += 1
        else:
            rec, how = R.narrate(sheet, client, rules=rules, attempts=attempts), "built"
            built += 1
        done.append(rec)
        all_old[(a, b)] = rec
        write_records(out, [*(all_old[k] for k in sorted(all_old, key=lambda k: (k[1], k[0]))), *other])
        _say(_chapter_line(a, b, rec, how))
        if rec["excluded_candidates"]:
            n_bad = len(rec["excluded_candidates"])
            _warn(f"({a}, {b}]: {n_bad} candidate strings failed the checker (a bug in a real build)")
        dropped = sheet["excluded"]["interventions_dropped"]
        if dropped > 0:
            _warn(f"({a}, {b}]: {dropped} settings are not listed (the block cap)")
    asked = [r for r in done if r["mode"] == "select" and r["attempts"]]
    if asked and all(r["fallback"] in R.RETRY_OUTCOMES for r in asked):
        _warn(
            f"every one of the {len(asked)} chapters that asked the model fell back to the template report "
            "(the server may have stopped answering): read the attempts, and build again to retry them"
        )
    _say(f"wrote {out}: {len(done)} chapters ({kept} kept, {built} built)")
    return OK


# --- verify ----------------------------------------------------------------------------------------------


def verify_reports(
    rd: R.RunView, records: Sequence[Any], rules: R.ReportRules = R.REPORT_V1
) -> tuple[int, dict[str, Any]]:
    """Replay every record (never a model): ``(exit code, summary)``. The exit code is the worst of the
    records': 2 a chapter that cannot be cut from the run any more, 1 a mismatch (or an ``env`` check that
    fails), 3 a stale record, else 0 (``versions_differ`` is reported, not an error)."""
    results: list[dict[str, Any]] = []
    for i, rec in enumerate(records):
        try:
            a, b = rec["chapter_start"], rec["step"]
            if not (isinstance(a, int) and isinstance(b, int)):
                raise TypeError("the chapter is not made of steps")
        except (KeyError, TypeError) as e:
            err = f"not a report record ({type(e).__name__})"
            results.append({"index": i, "chapter": None, "status": "error", "code": USAGE, "reasons": [err]})
            continue
        try:
            sheet_now = R.build_sheet(rd, a, b, rules)
        except R.ReportError as e:
            results.append(
                {"index": i, "chapter": [a, b], "status": "error", "code": e.code, "reasons": [str(e)]}
            )
            continue
        results.append({"index": i, "chapter": [a, b], **R.verify_record(rec, sheet_now, rules)})
    counts = dict(Counter(r["status"] for r in results))
    codes = {r["code"] for r in results if r["status"] == "error"}
    if USAGE in codes:
        code = USAGE
    elif MISMATCH in codes or counts.get("mismatch"):
        code = MISMATCH
    elif counts.get("stale"):
        code = STALE
    else:
        code = OK
    return code, {"records": len(results), "counts": counts, "exit": code, "results": results}


# --- probe -----------------------------------------------------------------------------------------------


def _tau_b(x: Sequence[float], y: Sequence[float], pairs: Sequence[tuple[int, int]]) -> float | None:
    """Kendall's tau-b of ``x`` and ``y`` over the index ``pairs`` (None if there is no pair or one of the
    sequences is constant over them)."""
    conc = disc = tx = ty = 0
    for i, j in pairs:
        dx, dy = x[i] - x[j], y[i] - y[j]
        tx += dx == 0
        ty += dy == 0
        conc += dx * dy > 0
        disc += dx * dy < 0
    n0 = len(pairs)
    denom = math.sqrt((n0 - tx) * (n0 - ty))
    return None if denom == 0 else (conc - disc) / denom


def kendall_tau_b(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Kendall's tau-b of two equally long sequences (None for fewer than two items or a constant one)."""
    n = len(x)
    return _tau_b(x, y, [(i, j) for i in range(n) for j in range(i + 1, n)])


def jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa, sb = set(a), set(b)
    return 1.0 if not sa and not sb else len(sa & sb) / len(sa | sb)


def nearest_rank(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    return s[max(0, math.ceil(q * len(s)) - 1)]


def mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def call_metrics(rec: Mapping[str, Any], rules: R.ReportRules = R.REPORT_V1) -> dict[str, Any]:
    """The select-mode measures of one record (§6.4): the outcome of each attempt, the entry drops by
    reason, the fallback, the latency and, for the attempt that was used, the agreement of the model's raw
    order with the salience order (Kendall tau-b over all pairs and over the pairs inside one section), the
    Jaccard index of its set and the template's first k, the mandatory sentences it left out and whether its
    headline is the template's."""
    sheet = rec["facts"]
    body = R.candidates(sheet, rules)[0]
    k = R.body_k(sheet, body, rules)
    atts = rec["attempts"]
    used, fallback = R.settle(atts)
    drops = Counter(d["reason"] for a in atts if a["choice"] for d in a["choice"]["dropped"])
    out: dict[str, Any] = {
        "outcomes": [a["outcome"] for a in atts],
        "fallback": fallback,
        "drops": dict(drops),
        "latency_s": [a["latency_s"] for a in atts],
        "tau_b": None,
        "tau_b_within_sections": None,
        "jaccard": None,
        "mandatory_omitted": None,
        "headline_agrees": None,
    }
    if used is None:
        return out
    choice = atts[used]["choice"]
    order = choice["order"]
    salience = {c["id"]: c["salience"] for c in body}
    section = {c["id"]: c["section"] for c in body}
    pos, sal = list(range(len(order))), [salience[i] for i in order]
    n = len(order)
    every = [(i, j) for i in range(n) for j in range(i + 1, n)]
    inside = [(i, j) for i, j in every if section[order[i]] == section[order[j]]]
    out.update(
        {
            "tau_b": _tau_b(pos, sal, every),
            "tau_b_within_sections": _tau_b(pos, sal, inside),
            "jaccard": jaccard(order, [c["id"] for c in body[:k]]),
            "mandatory_omitted": [c["id"] for c in body if c["mandatory"] and c["id"] not in order],
            "headline_agrees": choice["headline"] == "H1",
        }
    )
    return out


def summarize_probe(calls: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The aggregate of ``call_metrics`` over the calls of a probe."""
    atts = [o for c in calls for o in c["outcomes"]]
    lat = [x for c in calls for x in c["latency_s"]]
    n = len(calls)
    usable = [c for c in calls if c["fallback"] in (None, "headline")]

    def avg(key: str) -> float | None:
        return mean([c[key] for c in usable if c[key] is not None])

    return {
        "calls": n,
        "attempts": len(atts),
        "valid_attempt_rate": (atts.count("ok") / len(atts)) if atts else None,
        "first_attempt_ok_rate": (sum(1 for c in calls if c["outcomes"][:1] == ["ok"]) / n) if n else None,
        "outcomes": dict(Counter(atts)),
        "fallbacks": dict(Counter(c["fallback"] for c in calls if c["fallback"])),
        "fallback_rate": (sum(1 for c in calls if c["fallback"] in R.RETRY_OUTCOMES) / n) if n else None,
        "entry_drops": dict(sum((Counter(c["drops"]) for c in calls), Counter())),
        "latency_median_s": nearest_rank(lat, 0.5),
        "latency_p90_s": nearest_rank(lat, 0.9),
        "latency_max_s": max(lat) if lat else None,
        "tau_b_mean": avg("tau_b"),
        "tau_b_within_sections_mean": avg("tau_b_within_sections"),
        "jaccard_mean": avg("jaccard"),
        "mandatory_omissions": sum(len(c["mandatory_omitted"] or ()) for c in usable),
        "calls_with_mandatory_omission": sum(1 for c in usable if c["mandatory_omitted"]),
        "headline_agreement_rate": mean(
            [float(c["headline_agrees"]) for c in usable if c["headline_agrees"] is not None]
        ),
    }


def probe_reports(
    sheets: Sequence[tuple[str, Mapping[str, Any]]],
    client: Any,
    n: int,
    *,
    attempts: int,
    rules: R.ReportRules = R.REPORT_V1,
) -> dict[str, Any]:
    """``n`` select requests cycling through ``sheets``: the records, the per-call measures, the summary."""
    records, calls = [], []
    for i in range(n):
        name, sheet = sheets[i % len(sheets)]
        rec = R.narrate(sheet, client, rules=rules, attempts=attempts)
        records.append(rec)
        calls.append({"sheet": name, **call_metrics(rec, rules)})
        _say(f"probe {i + 1}/{n} {name}: {','.join(calls[-1]['outcomes']) or 'no request'}")
    return {
        "model": client.model,
        "url": client.url,
        "n": n,
        "attempts": attempts,
        "sheets": [name for name, _ in sheets],
        "summary": summarize_probe(calls),
        "calls": calls,
        "records": records,
    }


# --- the command line ------------------------------------------------------------------------------------


def _client(url: str, model: str, timeout_s: float) -> Any:
    from alife.director.llm import Client  # lazy: only the --llm paths build a client

    return Client(url, model, timeout_s, R.SELECT_MAX_TOKENS, R.SELECT_TEMPERATURE)


def _check_url(url: str) -> int | None:
    """The exit code (2) if ``url`` is not an http(s) chat-completions URL, else None. A request is only ever
    sent to ``.../chat/completions``: never to a model-management endpoint."""
    if not is_http_url(url):
        print(f"error: --url must be an http(s) URL, got {url!r}", file=sys.stderr)
        return USAGE
    if not R.is_chat_url(url):
        print(
            f"error: --url must end in {R.CHAT_PATH} (the only endpoint a request is sent to), got {url!r}",
            file=sys.stderr,
        )
        return USAGE
    return None


def _check_preflight(url: str, model: str, no_preflight: bool) -> int | None:
    """Run the preflight; the exit code to stop with (2), or None to go on."""
    stop = _check_url(url)
    if stop is not None:
        return stop
    pf = preflight(url, model)
    why = preflight_refusal(pf, model, no_preflight)
    if why is not None:
        print(f"error: {why}", file=sys.stderr)
        return USAGE
    _say(f"preflight: {pf.status}" + (" (skipped: the endpoint does not answer)" if no_preflight else ""))
    return None


def _inside(out: Path, run: Path) -> bool:
    return out.resolve().is_relative_to(run.resolve())


def _same_file(x: Path, y: Path) -> bool:
    """Whether two paths name the same file (resolved, or one file through two links)."""
    try:
        if x.resolve() == y.resolve():
            return True
        return x.exists() and y.exists() and os.path.samefile(x, y)
    except OSError:
        return False


def _cmd_build(a: argparse.Namespace) -> int:
    if a.llm and not a.model:
        print("error: --llm needs --model (there is no default model)", file=sys.stderr)
        return USAGE
    if not a.llm and (a.model or a.no_preflight):
        print("error: --model and --no-preflight need --llm", file=sys.stderr)
        return USAGE
    if a.attempts < 1 or a.timeout <= 0:
        print("error: --attempts must be at least 1 and --timeout positive", file=sys.stderr)
        return USAGE
    if _inside(a.out, a.run):
        print("error: --out must not be inside the run directory", file=sys.stderr)
        return USAGE
    cuts = chapter_cuts(a.start, a.end, a.chapter)
    rd = R.load_run(a.run)
    # every chapter is cut and checked before the first request: a bad cut is exit 2 with nothing written
    sheets = [R.build_sheet(rd, lo, hi) for lo, hi in cuts]
    client = None
    if a.llm:
        url = a.url or DEFAULT_URL
        stop = _check_url(url)  # also when the preflight is skipped below
        if stop is not None:
            return stop
        if pending_chapters(sheets, a.out, mode="select", model=a.model) > 0:
            stop = _check_preflight(url, a.model, a.no_preflight)
            if stop is not None:
                return stop
        else:
            _say("every chapter is kept: no request will be sent, no preflight")
        client = _client(url, a.model, a.timeout)
    return build_reports(sheets, a.out, client=client, attempts=a.attempts)


def _cmd_verify(a: argparse.Namespace) -> int:
    if _inside(a.out, a.run):
        print("error: --out must not be inside the run directory", file=sys.stderr)
        return USAGE
    if not a.reports.is_file():
        print(f"error: {a.reports} is not a file", file=sys.stderr)
        return USAGE
    if _same_file(a.out, a.reports):
        print("error: --out must not be the --reports file (it would overwrite the records)", file=sys.stderr)
        return USAGE
    records = read_records(a.reports)
    if not records:
        print(f"error: {a.reports} holds no records: there is nothing to verify", file=sys.stderr)
        return USAGE
    code, summary = verify_reports(R.load_run(a.run), records)
    head = {"run": str(a.run), "reports": str(a.reports), "versions": R.record_versions()}
    write_json(a.out, {**head, **summary})
    _say(f"verified {summary['records']} records: {summary['counts']}; wrote {a.out}")
    return code


def _cmd_probe(a: argparse.Namespace) -> int:
    if a.n < 1 or a.attempts < 1 or a.timeout <= 0:
        print("error: --n and --attempts must be at least 1 and --timeout positive", file=sys.stderr)
        return USAGE
    files = sorted(a.sheets.glob("*.json")) if a.sheets.is_dir() else []
    if not files:
        print(f"error: no *.json sheets in {a.sheets}", file=sys.stderr)
        return USAGE
    if any(_same_file(a.out, p) for p in files):
        print("error: --out must not be one of the sheet files", file=sys.stderr)
        return USAGE
    sheets = [(p.stem, json.loads(p.read_text(encoding="utf-8"))) for p in files]
    bad = [name for name, s in sheets if s.get("report_rules_digest") != R.REPORT_V1.digest()]
    if bad:
        print(f"error: sheets built under other report rules: {bad}", file=sys.stderr)
        return USAGE
    url = a.url or DEFAULT_URL
    stop = _check_preflight(url, a.model, False)  # a probe always preflights
    if stop is not None:
        return stop
    result = probe_reports(sheets, _client(url, a.model, a.timeout), a.n, attempts=a.attempts)
    write_json(a.out, result)
    _say(f"wrote {a.out}: {json.dumps(result['summary'], sort_keys=True)}")
    return OK


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m alife.analysis.world_report", description=(__doc__ or "").split("\n")[0]
    )
    sub = ap.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="build the chapter reports of an observed run")
    b.add_argument("run", type=Path, help="run directory (read only)")
    b.add_argument("--from", dest="start", type=int, required=True, help="first cut (an observed row)")
    b.add_argument("--to", dest="end", type=int, required=True, help="last cut")
    b.add_argument("--out", type=Path, required=True, help="JSON lines of records (outside the run)")
    b.add_argument("--chapter", type=int, default=6000, help="chapter length: cuts at its multiples")
    b.add_argument("--llm", action="store_true", help="select and order the candidates with a model")
    b.add_argument("--model", default=None, help="the model id sent in every request (needs --llm)")
    b.add_argument("--url", default=None, help=f"chat-completions URL (default {DEFAULT_URL})")
    b.add_argument("--timeout", type=float, default=120.0, help="seconds per request (default 120)")
    b.add_argument("--attempts", type=int, default=2, help="requests per chapter at most (default 2)")
    b.add_argument(
        "--no-preflight",
        action="store_true",
        help="accepted only if the server does not answer the preflight",
    )
    b.set_defaults(func=_cmd_build)

    v = sub.add_parser("verify", help="replay stored records (never calls a model)")
    v.add_argument("run", type=Path, help="run directory (read only)")
    v.add_argument("--reports", type=Path, required=True, help="the JSON lines written by build")
    v.add_argument("--out", type=Path, required=True, help="verification JSON (outside the run)")
    v.set_defaults(func=_cmd_verify)

    p = sub.add_parser("probe", help="measure the select mode on stored sheets (coordinator only)")
    p.add_argument("--sheets", type=Path, required=True, help="directory of sheet JSON files")
    p.add_argument("--model", required=True, help="the model id sent in every request")
    p.add_argument("--url", default=None, help=f"chat-completions URL (default {DEFAULT_URL})")
    p.add_argument("--n", type=int, default=24, help="number of select calls, cycling through the sheets")
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--attempts", type=int, default=2)
    p.add_argument("--out", type=Path, required=True, help="probe JSON")
    p.set_defaults(func=_cmd_probe)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    try:
        a = _parser().parse_args(argv)
    except SystemExit as e:  # argparse: 0 for --help, 2 for a usage error
        return int(e.code) if isinstance(e.code, int) else USAGE
    try:
        return int(a.func(a))
    except R.ReportError as e:
        print(f"error: {e}", file=sys.stderr)
        return e.code
    except (ValueError, OSError, ConfigError, CheckpointError) as e:  # LogError is a ConfigError
        print(f"error: {e}", file=sys.stderr)
        return USAGE
    except (KeyError, TypeError, AttributeError, IndexError) as e:
        # a run, log or sheet file of the wrong shape: bad input (2), not a mismatch with the run (1)
        print(f"error: malformed input ({type(e).__name__}: {str(e)[:200]})", file=sys.stderr)
        return USAGE


if __name__ == "__main__":
    raise SystemExit(main())
