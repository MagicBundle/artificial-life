"""The world report's command line (docs/DESIGN.md §30): ``build``, ``verify`` and ``probe`` of
``python -m alife.analysis.world_report``. The template build and ``verify`` run with the network patched out; the
select mode talks to a local mock server (``tests/report_server.py``) and no other host. The one test that needs a
real server is marked ``llm`` and runs only with ``--model`` (or ``ALIFE_LLM_MODEL``)."""

# ruff: noqa: E501

from __future__ import annotations

import json
import os
import shutil
import socket
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from alife.analysis import world_report as W
from alife.game import report as R
from report_server import Server, valid_reply

DATA = Path(__file__).parent / "data"
TINY = DATA / "report_tiny"
PROBE = DATA / "report_probe"
ARGS = ["--from", "1000", "--to", "3000", "--chapter", "1000"]


def run(*argv: str | Path) -> int:
    return W.main([str(a) for a in argv])


@pytest.fixture
def server() -> Iterator[Server]:
    s = Server(models={"gem": "loaded", "cold": "not-loaded"})
    yield s
    s.close()


@pytest.fixture
def no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("the network was used")

    monkeypatch.setattr(socket, "socket", boom)
    monkeypatch.setattr(urllib.request, "urlopen", boom)


def lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]


def llm_args(server: Server, *extra: str, model: str = "gem") -> list[str]:
    return ["--llm", "--model", model, "--url", server.url, *extra]


# --- the chapters and the helper functions -----------------------------------------------------------------------


def test_chapter_cuts_on_the_global_clock() -> None:
    assert W.chapter_cuts(1000, 3000, 1000) == [(1000, 2000), (2000, 3000)]
    assert W.chapter_cuts(1500, 5500, 2000) == [(1500, 2000), (2000, 4000), (4000, 5500)]
    assert W.chapter_cuts(0, 6000, 6000) == [(0, 6000)]
    assert W.chapter_cuts(6000, 18000, 6000) == [(6000, 12000), (12000, 18000)]
    for bad in ((10, 10, 5), (10, 5, 5), (-1, 5, 5), (0, 10, 0)):
        with pytest.raises(R.ReportError):
            W.chapter_cuts(*bad)


def test_kendall_jaccard_and_percentiles() -> None:
    assert W.kendall_tau_b([0, 1, 2, 3], [1, 2, 3, 4]) == 1.0
    assert W.kendall_tau_b([0, 1, 2, 3], [4, 3, 2, 1]) == -1.0
    assert W.kendall_tau_b([0, 1, 2], [1, 3, 2]) == pytest.approx(1 / 3)
    assert W.kendall_tau_b([0], [1]) is None and W.kendall_tau_b([], []) is None
    assert W.kendall_tau_b([0, 1, 2], [5, 5, 5]) is None
    # with ties: tau-b = (C - D) / sqrt((n0 - n1)(n0 - n2))
    assert W.kendall_tau_b([0, 1, 2], [1, 1, 2]) == pytest.approx(2 / (3**0.5 * 2**0.5))
    assert W.jaccard(["a", "b"], ["b", "c"]) == pytest.approx(1 / 3) and W.jaccard([], []) == 1.0
    assert W.nearest_rank([3, 1, 2], 0.5) == 2 and W.nearest_rank([1, 2, 3, 4], 0.9) == 4
    assert W.nearest_rank([], 0.5) is None and W.mean([]) is None and W.mean([1, 2]) == 1.5


def test_models_url_is_the_read_only_listing_of_the_same_server() -> None:
    assert W.models_url("http://localhost:1234/v1/chat/completions") == "http://localhost:1234/api/v0/models"
    assert W.models_url("https://h.example:8443/x/y") == "https://h.example:8443/api/v0/models"


# --- build (template) --------------------------------------------------------------------------------------------------


def test_build_template_reports_equal_the_golden_and_use_no_network(tmp_path, capsys, no_network) -> None:
    out = tmp_path / "r.jsonl"
    assert run("build", TINY, *ARGS, "--out", out) == 0
    recs = lines(out)
    golden = json.loads((TINY / "golden.json").read_text())
    assert [(r["chapter_start"], r["step"]) for r in recs] == [(1000, 2000), (2000, 3000)]
    for rec, g in zip(recs, golden, strict=True):
        assert (
            rec["mode"] == "template"
            and rec["model"] is None
            and rec["attempts"] == []
            and rec["fallback"] is None
        )
        assert rec["report"] == g["report"] and rec["ticker"] == g["ticker"]
        assert rec["facts_digest"] == g["facts_digest"] and rec["facts"] == g["sheet"]
        assert [c["text"] for c in rec["candidates"]] == [c["text"] for c in g["candidates"]]
        assert rec["excluded_candidates"] == []
    cap = capsys.readouterr()
    assert "2 chapters (0 kept, 2 built)" in cap.out and cap.err == ""


def test_build_resumes_and_keeps_what_matches(tmp_path, capsys) -> None:
    out = tmp_path / "r.jsonl"
    assert run("build", TINY, *ARGS, "--out", out) == 0
    first = out.read_bytes()
    capsys.readouterr()
    assert run("build", TINY, *ARGS, "--out", out) == 0
    assert out.read_bytes() == first
    assert "(2000, 3000] template kept" in capsys.readouterr().out
    # another cut rebuilds: the file then holds the chapters of this build only
    assert run("build", TINY, "--from", "1000", "--to", "3000", "--chapter", "2000", "--out", out) == 0
    assert [(r["chapter_start"], r["step"]) for r in lines(out)] == [(1000, 2000), (2000, 3000)]
    assert run("build", TINY, "--from", "1000", "--to", "2000", "--chapter", "1000", "--out", out) == 0
    assert [(r["chapter_start"], r["step"]) for r in lines(out)] == [(1000, 2000)]


def test_build_rebuilds_a_record_whose_version_or_facts_differ(tmp_path, capsys) -> None:
    out = tmp_path / "r.jsonl"
    run("build", TINY, *ARGS, "--out", out)
    recs = lines(out)
    recs[0].update({"template_version": 0})
    recs[1].update({"facts_digest": "0" * 64})
    out.write_text("".join(json.dumps(r) + "\n" for r in recs))
    capsys.readouterr()
    assert run("build", TINY, *ARGS, "--out", out) == 0
    assert "2 chapters (0 kept, 2 built)" in capsys.readouterr().out
    assert lines(out)[0]["template_version"] == 1 and lines(out)[1]["facts_digest"] != "0" * 64


def test_build_checks_every_cut_before_writing_or_asking_anything(tmp_path, server, capsys) -> None:
    """A bad cut (here the last, (3000, 3500]) is exit 2 with nothing written and no request sent."""
    out = tmp_path / "r.jsonl"
    argv = ["build", TINY, "--from", "1000", "--to", "3500", "--chapter", "1000", "--out", out]
    assert run(*argv) == 2 and not out.exists()
    assert run(*argv, *llm_args(server)) == 2 and not out.exists()
    assert server.gets == [] and server.requests == []
    assert "not the step of a metrics row" in capsys.readouterr().err


def test_build_writes_after_every_chapter(tmp_path, monkeypatch) -> None:
    """The file is rewritten atomically after each chapter: when the run is interrupted the chapters done are on disk
    and the stored records of the others are still there."""
    out = tmp_path / "r.jsonl"
    assert run("build", TINY, *ARGS, "--out", out) == 0
    full = out.read_bytes()
    real, calls = R.narrate, []

    def interrupt(sheet, client, **kw):
        calls.append(sheet["chapter"])
        if len(calls) == 2:
            raise KeyboardInterrupt
        return real(sheet, client, **kw)

    recs = lines(out)
    recs[0].update({"template_version": 0})
    recs[1].update({"template_version": 0})
    out.write_text("".join(json.dumps(r) + "\n" for r in recs))
    monkeypatch.setattr(R, "narrate", interrupt)
    with pytest.raises(KeyboardInterrupt):
        run("build", TINY, *ARGS, "--out", out)
    got = lines(out)
    assert len(got) == 2 and got[0]["template_version"] == 1 and got[1]["template_version"] == 0
    assert not list(tmp_path.glob(".*"))  # no temporary file is left
    monkeypatch.undo()
    assert run("build", TINY, *ARGS, "--out", out) == 0 and out.read_bytes() == full


@pytest.mark.parametrize(
    ("argv", "needle"),
    [
        (["--llm", *ARGS], "--llm needs --model"),
        (["--model", "x", *ARGS], "need --llm"),
        (["--no-preflight", *ARGS], "need --llm"),
        (["--from", "1000", "--to", "3000", "--chapter", "0"], "chapter"),
        (["--from", "3000", "--to", "1000"], "below"),
        (["--from", "1050", "--to", "3000", "--chapter", "1000"], "not a multiple"),
        (["--from", "1000", "--to", "9000", "--chapter", "1000"], "not the step of a metrics row"),
        (["--from", "1000", "--to", "3000", "--chapter", "1000", "--attempts", "0"], "attempts"),
    ],
)
def test_build_bad_parameters_exit_2(tmp_path, capsys, argv, needle) -> None:
    out = tmp_path / "r.jsonl"
    assert run("build", TINY, *argv, "--out", out) == 2
    assert needle in capsys.readouterr().err


def test_build_refuses_out_inside_the_run_and_a_missing_run_and_a_broken_out(tmp_path, capsys) -> None:
    copy = tmp_path / "run"
    shutil.copytree(TINY, copy)
    before = sorted(p.name for p in copy.iterdir())
    assert run("build", copy, *ARGS, "--out", copy / "reports.jsonl") == 2
    assert run("build", copy, *ARGS, "--out", copy / "sub" / "r.jsonl") == 2
    assert sorted(p.name for p in copy.iterdir()) == before
    assert "inside the run" in capsys.readouterr().err
    assert run("build", tmp_path / "nope", *ARGS, "--out", tmp_path / "r.jsonl") == 2
    broken = tmp_path / "broken.jsonl"
    broken.write_text("not json\n")
    assert run("build", TINY, *ARGS, "--out", broken) == 2 and broken.read_text() == "not json\n"
    assert run("build", TINY, *ARGS) == 2  # --out is required (a usage error)


def test_build_never_writes_into_the_run_directory(tmp_path) -> None:
    copy = tmp_path / "run"
    shutil.copytree(TINY, copy)

    def snapshot() -> dict[str, bytes]:
        return {str(p.relative_to(copy)): p.read_bytes() for p in sorted(copy.rglob("*")) if p.is_file()}

    before = snapshot()
    assert run("build", copy, *ARGS, "--out", tmp_path / "r.jsonl") == 0
    assert run("verify", copy, "--reports", tmp_path / "r.jsonl", "--out", tmp_path / "v.json") == 0
    assert snapshot() == before


# --- build --llm: the preflight, the requests, the policy, the resume ------------------------------------------


def test_llm_build_preflights_once_then_sends_the_model_explicitly(tmp_path, server, capsys) -> None:
    server.respond = lambda body: valid_reply(body)
    out = tmp_path / "r.jsonl"
    assert run("build", TINY, *ARGS, "--out", out, *llm_args(server)) == 0
    assert server.gets == ["/api/v0/models"]  # the one read-only request besides the POSTs
    assert len(server.requests) == 2
    for body in server.requests:
        assert body["model"] == "gem" and body["max_tokens"] == 200 and body["temperature"] == 0.4
        assert body["response_format"]["json_schema"]["name"] == "world_report_select"
    recs = lines(out)
    assert [r["mode"] for r in recs] == ["select", "select"] and {r["model"] for r in recs} == {"gem"}
    assert all(r["fallback"] is None and [a["outcome"] for a in r["attempts"]] == ["ok"] for r in recs)
    assert all(r["url"] == server.url and r["messages"] and r["request"] for r in recs)
    assert "preflight: loaded" in capsys.readouterr().out


def test_llm_build_then_verify_without_a_network(tmp_path, server, monkeypatch) -> None:
    server.respond = lambda body: valid_reply(body, n=5)
    out = tmp_path / "r.jsonl"
    assert run("build", TINY, *ARGS, "--out", out, *llm_args(server)) == 0
    server.close()

    def boom(*a, **k):
        raise AssertionError("verify used the network")

    monkeypatch.setattr(socket, "socket", boom)
    monkeypatch.setattr(urllib.request, "urlopen", boom)
    v = tmp_path / "v.json"
    assert run("verify", TINY, "--reports", out, "--out", v) == 0
    res = json.loads(v.read_text())
    assert res["counts"] == {"ok": 2} and res["exit"] == 0 and res["records"] == 2
    assert res["versions"] == R.record_versions()


@pytest.mark.parametrize(
    ("models", "model", "flag", "needle"),
    [
        ({"gem": "loaded"}, "other", [], "not loaded"),
        ({"gem": "not-loaded"}, "gem", [], "not loaded"),
        ({"gem": "loaded"}, "gem", ["--no-preflight"], "--no-preflight is refused"),
        ({"gem": "not-loaded"}, "gem", ["--no-preflight"], "--no-preflight is refused"),
        (None, "gem", [], "does not answer"),
    ],
)
def test_a_refused_preflight_exits_2_before_any_request(
    tmp_path, capsys, models, model, flag, needle
) -> None:
    s = Server(models=models)
    try:
        out = tmp_path / "r.jsonl"
        assert run("build", TINY, *ARGS, "--out", out, *llm_args(s, *flag, model=model)) == 2
        assert s.requests == [] and not out.exists()
        assert needle in capsys.readouterr().err
    finally:
        s.close()


def test_no_preflight_is_accepted_only_when_the_endpoint_does_not_answer(tmp_path, capsys) -> None:
    s = Server(models=None)  # another server: no /api/v0/models
    s.respond = lambda body: valid_reply(body)
    try:
        out = tmp_path / "r.jsonl"
        assert run("build", TINY, *ARGS, "--out", out, *llm_args(s, "--no-preflight")) == 0
        assert (
            s.gets == ["/api/v0/models"] and len(s.requests) == 2
        )  # the GET is still issued, the answer ignored
        assert "skipped" in capsys.readouterr().out
    finally:
        s.close()


def test_an_unreachable_server_is_a_refused_preflight(tmp_path, server, capsys) -> None:
    url = server.url
    server.close()
    out = tmp_path / "r.jsonl"
    assert run("build", TINY, *ARGS, "--out", out, "--llm", "--model", "gem", "--url", url) == 2
    assert "does not answer" in capsys.readouterr().err
    assert run("build", TINY, *ARGS, "--out", out, "--llm", "--model", "gem", "--url", "ftp://x/y") == 2


def test_preflight_function_states(server) -> None:
    assert W.preflight(server.url, "gem").status == "loaded"
    assert W.preflight(server.url, "cold").status == "not_loaded"
    assert W.preflight(server.url, "nobody").status == "unlisted"
    other = Server(models=None)
    try:
        assert W.preflight(other.url, "gem").status == "no_answer"
        assert W.preflight_refusal(W.Preflight("no_answer", "x"), "gem", True) is None
        assert W.preflight_refusal(W.Preflight("loaded", "loaded"), "gem", False) is None
    finally:
        other.close()
    assert set(server.gets) == {"/api/v0/models"}


def test_a_garbled_reply_is_retried_then_falls_back_and_is_retried_by_a_later_build(
    tmp_path, server, capsys
) -> None:
    out = tmp_path / "r.jsonl"
    # the first four requests (two attempts for each chapter) get garbage
    server.respond = lambda body: "garbage" if len(server.requests) <= 4 else valid_reply(body)
    assert run("build", TINY, *ARGS, "--out", out, *llm_args(server)) == 0
    recs = lines(out)
    assert [a["outcome"] for a in recs[0]["attempts"]] == ["invalid", "invalid"] and recs[0][
        "fallback"
    ] == "invalid"
    assert recs[0]["report"] == json.loads(json.dumps(R.template_report(recs[0]["facts"])))
    assert recs[1]["fallback"] == "invalid" and len(server.requests) == 4
    assert "fallback invalid" in capsys.readouterr().out
    n_before = len(server.requests)
    # the server now answers: a later --llm build retries the chapters that never got a model answer
    server.respond = lambda body: valid_reply(body)
    assert run("build", TINY, *ARGS, "--out", out, *llm_args(server)) == 0
    recs = lines(out)
    assert all(r["fallback"] is None and r["attempts"][-1]["outcome"] == "ok" for r in recs)
    assert len(server.requests) > n_before
    # and now both are kept: no request
    n_after = len(server.requests)
    capsys.readouterr()
    assert run("build", TINY, *ARGS, "--out", out, *llm_args(server)) == 0
    assert len(server.requests) == n_after and "2 kept" in capsys.readouterr().out


def test_a_failed_http_call_is_retried_and_a_headline_fallback_is_kept(tmp_path, server) -> None:
    out = tmp_path / "r.jsonl"
    server.say(None, 500)
    assert (
        run(
            "build",
            TINY,
            "--from",
            "1000",
            "--to",
            "2000",
            "--chapter",
            "1000",
            "--out",
            out,
            *llm_args(server),
        )
        == 0
    )
    rec = lines(out)[0]
    assert [a["outcome"] for a in rec["attempts"]] == ["failed", "failed"] and rec["fallback"] == "failed"
    assert rec["attempts"][0]["error"].startswith("empty") and rec["attempts"][1]["error"].startswith(
        "http: status 500"
    )
    server.queue.clear()
    server.respond = lambda body: valid_reply(
        body, headline="H9"
    )  # unknown headline: usable, fallback "headline"
    assert (
        run(
            "build",
            TINY,
            "--from",
            "1000",
            "--to",
            "2000",
            "--chapter",
            "1000",
            "--out",
            out,
            *llm_args(server),
        )
        == 0
    )
    rec = lines(out)[0]
    assert rec["fallback"] == "headline" and rec["report"]["headline"]["origin"] == "template"
    n = len(server.requests)
    assert (
        run(
            "build",
            TINY,
            "--from",
            "1000",
            "--to",
            "2000",
            "--chapter",
            "1000",
            "--out",
            out,
            *llm_args(server),
        )
        == 0
    )
    assert len(server.requests) == n  # the headline fallback is a usable record: kept


def test_the_resume_key_has_the_mode_and_the_model(tmp_path, server, capsys) -> None:
    server.respond = lambda body: valid_reply(body)
    server.models = {"gem": "loaded", "gem2": "loaded"}
    out = tmp_path / "r.jsonl"
    one = ["--from", "1000", "--to", "2000", "--chapter", "1000", "--out", out]
    assert run("build", TINY, *one) == 0 and lines(out)[0]["mode"] == "template"
    # a template record is rebuilt under --llm (F13)
    assert run("build", TINY, *one, *llm_args(server)) == 0
    assert lines(out)[0]["mode"] == "select" and len(server.requests) == 1
    # kept under the same model
    assert run("build", TINY, *one, *llm_args(server)) == 0 and len(server.requests) == 1
    # another model rebuilds
    assert run("build", TINY, *one, *llm_args(server, model="gem2")) == 0
    assert (
        len(server.requests) == 2
        and lines(out)[0]["model"] == "gem2"
        and server.requests[1]["model"] == "gem2"
    )
    # a template build replaces the select record and says so
    capsys.readouterr()
    assert run("build", TINY, *one) == 0
    assert lines(out)[0]["mode"] == "template" and "replaced by template records" in capsys.readouterr().err


def test_build_prints_a_warning_for_candidate_strings_that_failed_the_checker(
    tmp_path, capsys, monkeypatch
) -> None:
    real = R.narrate

    def spoil(sheet, client, **kw):
        rec = real(sheet, client, **kw)
        return {**rec, "excluded_candidates": [{"template": "X", "cites": [], "reasons": ["R3"]}]}

    monkeypatch.setattr(R, "narrate", spoil)
    assert (
        run(
            "build",
            TINY,
            "--from",
            "1000",
            "--to",
            "2000",
            "--chapter",
            "1000",
            "--out",
            tmp_path / "r.jsonl",
        )
        == 0
    )
    assert "failed the checker" in capsys.readouterr().err


# --- verify ------------------------------------------------------------------------------------------------------------


@pytest.fixture
def built(tmp_path) -> tuple[Path, Path]:
    out = tmp_path / "r.jsonl"
    assert run("build", TINY, *ARGS, "--out", out) == 0
    return out, tmp_path


def edit_metrics(run_dir: Path, step: int, **changes: int) -> None:
    path = run_dir / "metrics.jsonl"
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    for r in rows:
        if r["step"] == step:
            r.update({k: r[k] + v for k, v in changes.items()})
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_verify_ok_writes_a_summary(built, no_network, capsys) -> None:
    out, tmp = built
    v = tmp / "v.json"
    assert run("verify", TINY, "--reports", out, "--out", v) == 0
    res = json.loads(v.read_text())
    assert res["counts"] == {"ok": 2} and [r["chapter"] for r in res["results"]] == [
        [1000, 2000],
        [2000, 3000],
    ]
    assert all(r["status"] == "ok" and r["reasons"] == [] and not r["sheet_differs"] for r in res["results"])
    assert "verified 2 records" in capsys.readouterr().out


def test_verify_stale_when_the_run_changed_under_equal_versions_exits_3(built, capsys) -> None:
    out, tmp = built
    copy = tmp / "run"
    shutil.copytree(TINY, copy)
    edit_metrics(copy, 1500, deaths_starved=1)  # inside the first chapter only
    v = tmp / "v.json"
    assert run("verify", copy, "--reports", out, "--out", v) == 3
    res = json.loads(v.read_text())
    assert (
        res["counts"] == {"stale": 1, "ok": 1}
        and res["results"][0]["sheet_differs"] is True
        and res["exit"] == 3
    )


def test_verify_versions_differ_is_reported_and_exits_0_even_with_a_changed_run(built) -> None:
    out, tmp = built
    recs = lines(out)
    recs[0].update({"checker_version": 7})
    out.write_text("".join(json.dumps(r) + "\n" for r in recs))
    copy = tmp / "run"
    shutil.copytree(TINY, copy)
    edit_metrics(copy, 1500, deaths_starved=1)
    v = tmp / "v.json"
    assert run("verify", copy, "--reports", out, "--out", v) == 0
    res = json.loads(v.read_text())
    first = res["results"][0]
    assert first["status"] == "versions_differ" and first["sheet_differs"] is True  # N17: both are reported
    assert first["versions"] == {"checker_version": [7, 1]}
    assert res["counts"] == {"versions_differ": 1, "ok": 1}


def test_verify_a_changed_label_table_with_equal_versions_is_versions_differ_not_stale(
    built, monkeypatch
) -> None:
    out, tmp = built
    monkeypatch.setitem(R.KNOWN_SOCIAL, "f" * 64, (250, 16))
    v = tmp / "v.json"
    assert run("verify", TINY, "--reports", out, "--out", v) == 0
    res = json.loads(v.read_text())
    assert res["counts"] == {"versions_differ": 2}
    assert res["results"][0]["versions"].keys() == {"known_social_digest"}


def test_verify_mismatch_exits_1(built) -> None:
    out, tmp = built
    recs = lines(out)
    recs[1]["report"]["body"][0]["text"] = "There were no deaths."
    out.write_text("".join(json.dumps(r) + "\n" for r in recs))
    v = tmp / "v.json"
    assert run("verify", TINY, "--reports", out, "--out", v) == 1
    res = json.loads(v.read_text())
    assert res["counts"] == {"ok": 1, "mismatch": 1} and "report" in " ".join(res["results"][1]["reasons"])


def test_verify_a_mismatch_beats_a_stale_record(built) -> None:
    out, tmp = built
    recs = lines(out)
    recs[1]["ticker"][0]["text"] = "Step 2,250: x."
    out.write_text("".join(json.dumps(r) + "\n" for r in recs))
    copy = tmp / "run"
    shutil.copytree(TINY, copy)
    edit_metrics(copy, 1500, deaths_starved=1)
    assert run("verify", copy, "--reports", out, "--out", tmp / "v.json") == 1


def test_verify_a_chapter_the_run_cannot_cut_exits_2(built) -> None:
    out, tmp = built
    recs = lines(out)
    recs[0].update({"chapter_start": 1050})
    out.write_text("".join(json.dumps(r) + "\n" for r in recs))
    v = tmp / "v.json"
    assert run("verify", TINY, "--reports", out, "--out", v) == 2
    assert json.loads(v.read_text())["results"][0]["status"] == "error"
    out.write_text('{"nothing": 1}\n[1]\n')
    assert run("verify", TINY, "--reports", out, "--out", v) == 2


def test_verify_bad_parameters(built, capsys) -> None:
    out, tmp = built
    copy = tmp / "run"
    shutil.copytree(TINY, copy)
    assert run("verify", copy, "--reports", out, "--out", copy / "v.json") == 2
    assert run("verify", TINY, "--reports", tmp / "missing.jsonl", "--out", tmp / "v.json") == 2
    assert run("verify", tmp / "nope", "--reports", out, "--out", tmp / "v.json") == 2
    assert run("verify", TINY) == 2


def test_verify_is_for_the_run_that_built_the_report_resumed_copy_agrees(built) -> None:
    out, tmp = built
    resumed = DATA / "report_tiny_resumed"
    assert run("verify", resumed, "--reports", out, "--out", tmp / "v.json") == 0


# --- probe ---------------------------------------------------------------------------------------------------------------


def test_probe_measures_the_select_mode_on_the_stored_sheets(tmp_path, server, capsys) -> None:
    server.respond = lambda body: valid_reply(body)
    out = tmp_path / "p.json"
    code = run("probe", "--sheets", PROBE, "--model", "gem", "--url", server.url, "--n", "13", "--out", out)
    assert code == 0 and server.gets == ["/api/v0/models"] and len(server.requests) == 13
    res = json.loads(out.read_text())
    assert (
        res["n"] == 13 and len(res["records"]) == 13 and len(res["calls"]) == 13 and len(res["sheets"]) == 12
    )
    assert res["calls"][12]["sheet"] == res["calls"][0]["sheet"]  # the sheets are cycled
    s = res["summary"]
    assert s["calls"] == 13 and s["valid_attempt_rate"] == 1.0 and s["first_attempt_ok_rate"] == 1.0
    assert s["fallback_rate"] == 0.0 and s["entry_drops"] == {} and s["latency_max_s"] is not None
    assert s["jaccard_mean"] == 1.0 and s["headline_agreement_rate"] == 1.0  # the reply is the salience order
    assert s["tau_b_mean"] == pytest.approx(1.0)
    for rec in res["records"]:
        assert R.verify_record(rec, rec["facts"])["status"] == "ok"
    assert "wrote" in capsys.readouterr().out


def test_probe_counts_drops_fallbacks_and_mandatory_omissions(tmp_path, server) -> None:
    def respond(body: dict[str, Any]) -> str:
        n = len(server.requests)
        if n % 4 == 1:
            return valid_reply(body, skip=1)  # leaves the first (mandatory) sentence out
        if n % 4 == 2:
            return json.dumps(
                {"headline": "H1", "order": ["C99", "C99", *json.loads(valid_reply(body))["order"]]}
            )
        if n % 4 == 3:
            return "not json"
        return valid_reply(body, headline="H1")

    server.respond = respond
    out = tmp_path / "p.json"
    sheets = tmp_path / "sheets"
    sheets.mkdir()
    shutil.copy(PROBE / "s02_2d_eventful.json", sheets / "a.json")
    assert (
        run(
            "probe",
            "--sheets",
            sheets,
            "--model",
            "gem",
            "--url",
            server.url,
            "--n",
            "8",
            "--attempts",
            "1",
            "--out",
            out,
        )
        == 0
    )
    s = json.loads(out.read_text())["summary"]
    assert (
        s["outcomes"] == {"ok": 6, "invalid": 2}
        and s["fallbacks"] == {"invalid": 2}
        and s["fallback_rate"] == 0.25
    )
    assert s["entry_drops"].get("unknown_id", 0) >= 2 and s["calls_with_mandatory_omission"] >= 1
    assert s["mandatory_omissions"] >= 1


def test_call_metrics_of_one_record() -> None:
    sheet = json.loads((PROBE / "s02_2d_eventful.json").read_text())
    body = R.candidates(sheet)[0]

    class One:
        url, model = "http://127.0.0.1:9/x", "m"

        def __init__(self, order: list[str], headline: str = "H1") -> None:
            self.text = json.dumps({"headline": headline, "order": order})

        def chat(self, b):
            from alife.director.llm import Exchange

            return Exchange(self.text, None, 0.1, "stop", None)

    ids = [c["id"] for c in body]
    perfect = W.call_metrics(R.narrate(sheet, One(ids[:6])))
    assert perfect["tau_b"] == 1.0 and perfect["tau_b_within_sections"] == 1.0 and perfect["jaccard"] == 1.0
    assert (
        perfect["mandatory_omitted"] == []
        and perfect["headline_agrees"] is True
        and perfect["fallback"] is None
    )
    backwards = W.call_metrics(R.narrate(sheet, One(ids[5::-1])))
    assert backwards["tau_b"] == -1.0 and backwards["jaccard"] == 1.0
    skewed = W.call_metrics(R.narrate(sheet, One(ids[6:11])))
    assert skewed["jaccard"] == 0.0 and skewed["mandatory_omitted"] == ["C1", "C3"]
    failed = W.call_metrics(R.narrate(sheet, One(["C1"])))
    assert failed["fallback"] == "too_few" and failed["tau_b"] is None and failed["mandatory_omitted"] is None


def test_probe_refuses_bad_parameters_and_a_model_that_is_not_loaded(tmp_path, server, capsys) -> None:
    base = ["probe", "--sheets", PROBE, "--url", server.url, "--out", tmp_path / "p.json"]
    assert run(*base, "--model", "cold") == 2 and server.requests == []
    assert run(*base, "--model", "gem", "--n", "0") == 2
    assert (
        run(
            "probe", "--sheets", tmp_path, "--model", "gem", "--url", server.url, "--out", tmp_path / "p.json"
        )
        == 2
    )
    assert run(*base) == 2  # --model is required
    bad = tmp_path / "bad"
    bad.mkdir()
    sheet = json.loads((PROBE / "s02_2d_eventful.json").read_text())
    (bad / "x.json").write_text(json.dumps({**sheet, "report_rules_digest": "0" * 64}))
    assert (
        run("probe", "--sheets", bad, "--model", "gem", "--url", server.url, "--out", tmp_path / "p.json")
        == 2
    )
    assert server.requests == [] and not (tmp_path / "p.json").exists()
    assert "other report rules" in capsys.readouterr().err
    # a probe has no --no-preflight
    assert run(*base, "--model", "gem", "--no-preflight") == 2


# --- an observed director fork of the real engine, end to end ----------------------------------------------------------

E2E = DATA / "report_e2e"
E2E_ARGS = ["--from", "2500", "--to", "10000", "--chapter", "2000"]


@pytest.fixture(scope="module")
def e2e_view() -> R.RunView:
    return R.load_run(E2E)


def e2e_copy(tmp_path: Path) -> Path:
    copy = tmp_path / "run"
    shutil.copytree(E2E, copy)
    return copy


def test_e2e_the_engines_env_blocks_are_the_fold_of_the_director_log(e2e_view) -> None:
    """`tests/data/make_report_e2e.py`: a tiny CPU fork replayed with a hand-written director log. The `env` blocks are
    the engine's own output, and the report's lever fold reproduces every one of them (the MUST of DESIGN §30)."""
    view = e2e_view
    R.check_env(view, 2500, 10000)
    env = {r["step"]: r["env"] for r in view.rows}
    assert (
        env[3000]["regen_mult"] == 1.0 and env[3100]["regen_mult"] == 1.5
    )  # the pulse acts after the row at 3000
    assert (
        env[8000]["regen_mult"] == 1.5 and env[8100]["regen_mult"] == 1.0
    )  # and its revert after the row at 8000
    assert env[4100]["season_amp"] == pytest.approx(0.3) and env[4100]["season_period"] == 2000
    assert env[5000]["patch_gain_max"] == 1.0 and env[5100]["patch_gain_max"] == 1.5
    assert env[9100]["drift_mult"] == 2.0 and env[2500]["drift_mult"] == 1.0
    assert [e["step_applied"] for e in view.entries] == [3000, 3000, 4000, 5000, 8000, 9000]
    assert view.start_step == 2000


def test_e2e_the_block_lists_every_lever_the_engine_holds_away_from_default(e2e_view) -> None:
    """The engine's levers away from default at the first row of a chapter, read from that row's `env` block alone,
    are exactly the levers that have a line in the chapter's block (one holder each)."""
    view = e2e_view
    cfg, rows = view.cfg, {r["step"]: r for r in view.rows}
    held = {}
    for a, b in ((2500, 4000), (4000, 6000), (6000, 8000), (8000, 10000)):
        env = rows[a]["env"]
        engine = set()
        if env["regen_mult"] != 1.0:
            engine.add("regen")
        if env["drift_mult"] != 1.0:
            engine.add("drift")
        if (
            env["season_amp"] != cfg.fields.season_amplitude
            or env["season_period"] != cfg.fields.season_period
        ):
            engine.add("seasons")
        if env["patch_gain_min"] != 1.0 or env["patch_gain_max"] != 1.0 or env["patches_off"]:
            engine.add("patch:0:gain")
        sheet = R.build_sheet(view, a, b)
        holders = [f["lever"] for f in sheet["facts"] if f["kind"] == "standing"] + [
            R.lever_of(f["action"], f["params"])
            for f in sheet["facts"]
            if f["kind"] == "intervention" and f["step"] < a and (f["end"] is None or f["end"] >= a)
        ]
        assert sorted(holders) == sorted(engine), (a, holders, engine)
        held[a] = sorted(engine)
    assert held == {
        2500: [],
        4000: ["regen"],
        6000: ["patch:0:gain", "regen", "seasons"],
        8000: ["patch:0:gain", "regen", "seasons"],
    }


def test_e2e_build_then_verify_exits_0_and_reports_equal_the_golden(tmp_path, no_network) -> None:
    out, v = tmp_path / "r.jsonl", tmp_path / "v.json"
    assert run("build", E2E, *E2E_ARGS, "--out", out) == 0
    recs = lines(out)
    golden = json.loads((E2E / "golden.json").read_text())
    assert len(recs) == len(golden) == 4
    for rec, g in zip(recs, golden, strict=True):
        assert [rec["chapter_start"], rec["step"]] == g["chapter"]
        assert (
            rec["facts_digest"] == g["facts_digest"]
            and rec["candidate_set_digest"] == g["candidate_set_digest"]
        )
        rep = rec["report"]
        assert rep["headline"]["text"] == g["headline"] and rep["lead"] == g["lead"]
        assert [b["text"] for b in rep["body"]] == g["body"]
        assert [i["text"] for i in rep["interventions"]] == g["interventions"] and rec["ticker"] == g[
            "ticker"
        ]
        assert {k: x for k, x in rec["facts"]["excluded"].items() if x} == g["excluded"]
        assert rec["excluded_candidates"] == [] and rec["mode"] == "template"
    assert run("verify", E2E, "--reports", out, "--out", v) == 0
    res = json.loads(v.read_text())
    assert res["counts"] == {"ok": 4} and res["exit"] == 0
    # a second build over the same file keeps every record
    assert run("build", E2E, *E2E_ARGS, "--out", out) == 0 and lines(out) == recs


def test_e2e_the_block_of_each_chapter_is_the_lever_state_not_the_recent_entries(tmp_path) -> None:
    out = tmp_path / "r.jsonl"
    assert run("build", E2E, *E2E_ARGS, "--out", out) == 0
    blocks = [[i["text"] for i in r["report"]["interventions"]] for r in lines(out)]
    assert [len(b) for b in blocks] == [3, 6, 6, 8]
    # chapter 3: the pulse ended at 8,000 (a revert inside the chapter's last row), the other two are still in place
    assert "It ended at step 8,000; food regrowth returned to its base rate." in blocks[2]
    assert blocks[2].count("It was still in place at step 8,000.") == 2
    # the 4,000 chapter lists the seasons and the fertility shift as in force; the first lists neither (not yet applied)
    assert not any("seasons" in x for x in blocks[0]) and any("seasons" in x for x in blocks[1])


def test_e2e_a_log_that_disagrees_with_the_rows_exits_1_before_anything_is_written(tmp_path, capsys) -> None:
    copy = e2e_copy(tmp_path)
    log = copy / "director.jsonl"
    log.write_text(log.read_text().replace('"mult": 1.5}', '"mult": 1.4}'))
    out = tmp_path / "r.jsonl"
    assert run("build", copy, *E2E_ARGS, "--out", out) == 1
    assert "regen_mult" in capsys.readouterr().err and not out.exists()
    # a report built from the real run does not verify against the doctored log: an env failure is exit 1
    good = tmp_path / "good.jsonl"
    assert run("build", E2E, *E2E_ARGS, "--out", good) == 0
    assert run("verify", copy, "--reports", good, "--out", tmp_path / "v.json") == 1
    res = json.loads((tmp_path / "v.json").read_text())
    assert {r["status"] for r in res["results"] if r["chapter"] != [2500, 4000]} == {"error"}


def test_e2e_a_log_with_a_missing_entry_cannot_be_checked(tmp_path) -> None:
    copy = e2e_copy(tmp_path)
    log = copy / "director.jsonl"
    rows = [x for x in log.read_text().splitlines() if '"step_applied": 5000' not in x]
    log.write_text("\n".join(rows) + "\n")
    assert run("build", copy, *E2E_ARGS, "--out", tmp_path / "r.jsonl") == 1


def test_e2e_an_unobserved_run_exits_2_with_the_hint_and_writes_nothing(tmp_path, capsys) -> None:
    copy = e2e_copy(tmp_path)
    rows = [json.loads(x) for x in (copy / "metrics.jsonl").read_text().splitlines()]
    (copy / "metrics.jsonl").write_text(
        "".join(json.dumps({k: v for k, v in r.items() if k != "observers"}) + "\n" for r in rows)
    )
    out = tmp_path / "r.jsonl"
    assert run("build", copy, *E2E_ARGS, "--out", out) == 2
    assert "--set game.observe=true" in capsys.readouterr().err and not out.exists()


# --- the one test that needs a real server -----------------------------------------------------------------------


@pytest.mark.llm
def test_real_server_one_narration_on_the_tiny_golden_sheet(request) -> None:
    """Structure only: one ``narrate`` of the tiny golden sheet against a real server, with the preflight. Runs only
    with ``--model ID`` (and optionally ``--llm-url``), or ``ALIFE_LLM_MODEL`` / ``ALIFE_LLM_URL``; the model must
    already be loaded (this test never loads one)."""
    model = request.config.getoption("--model") or os.environ.get("ALIFE_LLM_MODEL")
    if not model:
        pytest.skip("pass --model ID (a model that is already loaded) to run against a real server")
    url = request.config.getoption("--llm-url") or os.environ.get("ALIFE_LLM_URL") or W.DEFAULT_URL
    pf = W.preflight(url, model)
    assert W.preflight_refusal(pf, model, False) is None, pf
    from alife.director.llm import Client

    golden = json.loads((TINY / "golden.json").read_text())[0]
    sheet = golden["sheet"]
    rec = R.narrate(sheet, Client(url, model, 180.0, R.SELECT_MAX_TOKENS, R.SELECT_TEMPERATURE))
    assert rec["mode"] == "select" and rec["model"] == model and 1 <= len(rec["attempts"]) <= 2
    assert rec["fallback"] in (None, "headline", "invalid", "failed", "too_few")
    assert all(a["outcome"] in R.OUTCOMES for a in rec["attempts"])
    texts = {c["text"] for c in R.candidates(sheet)[0]}
    assert {x["text"] for x in rec["report"]["body"]} <= texts
    assert R.verify_record(json.loads(json.dumps(rec)), sheet)["status"] == "ok"
