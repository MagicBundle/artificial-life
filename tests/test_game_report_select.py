"""The select-and-order mode of the world report (docs/DESIGN.md §30; SELECT_PROMPT_VERSION 1): the prompt, the
schema, the validation of a reply, the policy over the attempts, the record, and its replay. No socket: the client
is a fake with scripted exchanges (the HTTP side is in ``test_analysis_world_report.py``)."""

# ruff: noqa: E501  (expected prompts and fixture tables are long lines)

from __future__ import annotations

import json
import random
import socket
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from alife.director.llm import Exchange
from alife.game import report as R
from alife.game import templates
from report_helpers import deaths, mk_sheet, pop, random_sheet

DATA = Path(__file__).parent / "data"


def probe_sheet(name: str) -> dict[str, Any]:
    return json.loads((DATA / "report_probe" / f"{name}.json").read_text(encoding="utf-8"))


EVENTFUL = "s02_2d_eventful"
QUIET = "s01_2d_quiet"
TWO_HEADS = "s03_2d_lineage_collapse"
SMALL = "s12_2d_unobserved_llm"
INTERVENED = "s11_2d_two_interventions"


class Fake:
    """A client with scripted exchanges: a string is the reply text, ``None`` a failed call, a callable builds the
    text from the request body. The last item repeats."""

    url = "http://127.0.0.1:9/v1/chat/completions"

    def __init__(self, *script: Any, model: str = "model-a") -> None:
        self.script, self.model, self.bodies = list(script), model, []

    def chat(self, body: dict[str, Any]) -> Exchange:
        self.bodies.append(json.loads(json.dumps(body)))
        item = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if callable(item):
            item = item(body)
        if item is None:
            return Exchange(None, "timeout: the reply did not complete in time", 0.25, None, None)
        usage = {"prompt_tokens": 800, "completion_tokens": 30, "total_tokens": 830}
        return Exchange(item, None, 0.5, "stop", usage)


def reply(headline: str | None = "H1", order: Any = ()) -> str:
    return json.dumps({"headline": headline, "order": list(order)})


def ids_of(sheet: dict[str, Any], *templates_: str) -> list[str]:
    return [c["id"] for c in R.candidates(sheet)[0] if c["template"] in templates_]


def first_n(n: int, skip: int = 0):
    def make(body: dict[str, Any]) -> str:
        enum = body["response_format"]["json_schema"]["schema"]["properties"]["order"]["items"]["enum"]
        return reply("H1", enum[skip : skip + n])

    return make


# --- the prompt and the schema -------------------------------------------------------------------------------------


def test_the_system_message_is_the_specified_text_and_the_prompt_is_pinned() -> None:
    assert R.SELECT_SYSTEM.startswith(
        "You are the editor of a chapter report about a simulated world. Code wrote"
    )
    assert R.SELECT_SYSTEM.endswith("Reply with the ids only, as JSON.")
    sheet = probe_sheet(SMALL)
    msgs = R.select_messages(sheet, R.candidates(sheet))
    assert [m["role"] for m in msgs] == ["system", "user"] and msgs[0]["content"] == R.SELECT_SYSTEM
    assert msgs[1]["content"] == (
        "Chapter: steps 792,000 to 798,000 of a 2D world.\n"
        "Lead (always first): The population fell from 21,400 to 18,900 (-11.7%).\n\n"
        "Body candidates (choose 3):\n"
        "C1 [always]: There were 16,260 deaths: 14,100 by starvation, 1,900 by attack and 260 by old age.\n"
        "C2: The lowest count was 18,640, in the 100 steps to step 1,517,900.\n"
        "C3: There were 13,660 births and 200 arrivals from outside.\n\n"
        "Headline candidates (choose 1):\n"
        "H1: Population fell to 18,900"
    )
    assert R.select_prompt_digest() == "7800edadecea2996e4e3b80d229b2d8e66c9f29512e9ebccdd32a9e456d938cb"


def test_the_user_message_lists_candidates_in_salience_order_with_the_mandatory_ones_marked() -> None:
    sheet = probe_sheet(EVENTFUL)
    cands = R.candidates(sheet)
    user = R.select_messages(sheet, cands)[1]["content"].split("\n")
    assert user[0] == "Chapter: steps 1,512,000 to 1,518,000 of a 2D world."
    assert user[2] == "" and user[3] == "Body candidates (choose 5 to 6):"
    lines = user[4 : 4 + len(cands[0])]
    for line, c in zip(lines, cands[0], strict=True):
        assert line == f"{c['id']}{' [always]' if c['mandatory'] else ''}: {c['text']}"
    assert user[-2:] == ["Headline candidates (choose 1):", "H1: " + cands[1][0]["text"]]
    # a 3D world and two headline candidates
    two = probe_sheet(TWO_HEADS)
    text = R.select_messages(two, R.candidates(two))[1]["content"]
    assert text.splitlines()[-3:-1][0] == "Headline candidates (choose 1):" and "\nH2: " in text
    three = probe_sheet("s07_3d_quiet")
    assert "of a 3D world." in R.select_messages(three, R.candidates(three))[1]["content"].splitlines()[0]


def test_the_prompt_holds_no_intervention_window_setting_or_model_text() -> None:
    """Only the lead, the body and headline candidates: not the block, not a window, no source word; a sentinel placed
    in fields the report never reads (`reason`, a journal) does not reach it."""
    sheet = probe_sheet(INTERVENED)
    cands = R.candidates(sheet)
    assert len(cands[2]) == 4  # the block exists
    user = R.select_messages(sheet, cands)[1]["content"]
    for item in cands[2]:
        assert item["text"] not in user
    for word in ("director", "regrowth", "seasons", "queued", "still in place", "ended", "While this was"):
        assert word not in user
    sentinel = json.loads(json.dumps(sheet))
    sentinel.update({"reason": "SENTINEL-REASON", "journal": "SENTINEL-JOURNAL", "llm_call": "SENTINEL-CALL"})
    assert "SENTINEL" not in json.dumps(R.select_messages(sentinel, R.candidates(sentinel)))
    rec = R.narrate(sheet, Fake(first_n(5)))
    assert "SENTINEL" not in json.dumps({k: v for k, v in rec.items() if k != "facts"})


def test_select_schema_shape_k_and_n_min() -> None:
    sheet = probe_sheet(EVENTFUL)
    cands = R.candidates(sheet)
    k = R.body_k(sheet, cands[0])
    assert k == 6
    schema = R.select_schema(cands, k)
    ids = [c["id"] for c in cands[0]]
    assert schema == {
        "type": "object",
        "additionalProperties": False,
        "required": ["headline", "order"],
        "properties": {
            "headline": {"type": "string", "maxLength": 4, "enum": ["H1"]},
            "order": {
                "type": "array",
                "minItems": 5,
                "maxItems": 6,
                "uniqueItems": True,
                "items": {"type": "string", "maxLength": 4, "enum": ids},
            },
        },
    }
    quiet = probe_sheet(QUIET)
    qc = R.candidates(quiet)
    qs = R.select_schema(qc, R.body_k(quiet, qc[0]))["properties"]["order"]
    assert (qs["minItems"], qs["maxItems"]) == (2, 3)  # k is 3 in a quiet chapter
    small = probe_sheet(SMALL)
    sc = R.candidates(small)
    ss = R.select_schema(sc, R.body_k(small, sc[0]))["properties"]["order"]
    assert (ss["minItems"], ss["maxItems"]) == (3, 6)  # three candidates: all of them
    two = probe_sheet(TWO_HEADS)
    assert R.select_schema(R.candidates(two), 6)["properties"]["headline"]["enum"] == ["H1", "H2"]
    assert (
        R.select_n_min(6, [{}] * 20) == 5
        and R.select_n_min(3, [{}] * 20) == 2
        and R.select_n_min(6, [{}] * 2) == 2
    )


def test_body_k_is_raised_to_the_number_of_mandatory_sentences() -> None:
    body = [{"mandatory": True}] * 8
    assert R.body_k({"quiet": False}, body) == 8 and R.body_k({"quiet": True}, body) == 8
    assert R.body_k({"quiet": False}, [{"mandatory": False}]) == 6 and R.body_k({"quiet": True}, []) == 3


def test_the_seed_is_a_function_of_the_run_the_chapter_and_the_attempt() -> None:
    s = [R.select_seed(0, 6000, i) for i in range(3)]
    assert len(set(s)) == 3 and all(0 <= x < 2**31 for x in s)
    assert R.select_seed(0, 6000, 1) == s[1] != R.select_seed(1, 6000, 1) != R.select_seed(0, 12000, 1)
    # the stream tag is 17 (the director's is 16)
    from alife.director.llm import request_seed

    assert R.select_seed(0, 6000, 0) != request_seed(0, 6000)


# --- parse_choice --------------------------------------------------------------------------------------------------

BAD_REPLIES = [
    ("", "not valid JSON"),
    ("sure!", "not valid JSON"),
    ("[]", "list, not an object"),
    ('"H1"', "str, not an object"),
    ("null", "NoneType"),
    ('{"headline": "H1"}', "keys"),
    ('{"order": []}', "keys"),
    ('{"headline": "H1", "order": [], "why": "x"}', "keys"),
    ('{"headline": "H1", "headline": "H1", "order": []}', "duplicate key"),
    ('{"headline": "H1", "order": [], "order": []}', "duplicate key"),
    ('{"headline": "H1", "order": [NaN]}', "not allowed"),
    ('{"headline": "H1", "order": [Infinity]}', "not allowed"),
    ('{"headline": "H1", "order": [-Infinity]}', "not allowed"),
    ('{"headline": "H1", "order": "C1"}', "not a list"),
    ('{"headline": "H1", "order": null}', "not a list"),
    ('{"headline": "H1", "order": {"a": 1}}', "not a list"),
    ('{"headline": "H1", "order": ["C1"]} trailing', "not valid JSON"),
    ('{"headline": "H1", "order": ["C1"]', "not valid JSON"),
    ("[" * 3000, "not valid JSON"),
    (reply("H1", ["C1"] * 2000), "longer than"),
]


@pytest.mark.parametrize(("text", "why"), BAD_REPLIES, ids=[str(i) for i in range(len(BAD_REPLIES))])
def test_reply_level_failures_are_invalid(text: str, why: str) -> None:
    sheet = probe_sheet(EVENTFUL)
    got = R.parse_choice(text, R.candidates(sheet), 6)
    assert got["outcome"] == "invalid" and got["choice"] is None and why in got["error"]


def test_entry_level_drops_keep_the_reply_and_record_the_reason() -> None:
    sheet = probe_sheet(EVENTFUL)
    cands = R.candidates(sheet)
    text = '{"headline": "H1", "order": ["C2", 7, "C99", "C2", null, "C5", ["C1"], {"a": 1}, true, 1e999, "C1", "c3", "C4", "C6", "C7", "C8"]}'
    got = R.parse_choice(text, cands, 6)
    assert got["outcome"] == "ok" and got["error"] is None
    ch = got["choice"]
    assert ch["order"][:3] == ["C2", "C5", "C1"] and len(ch["order"]) == 6
    assert ch["headline"] == "H1"
    reasons = [(d["entry"], d["reason"]) for d in ch["dropped"]]
    assert reasons == [
        ("7", "not_a_string"),
        ("C99", "unknown_id"),
        ("C2", "repeat"),
        ("None", "not_a_string"),
        ("<list>", "not_a_string"),
        ("<dict>", "not_a_string"),
        ("True", "not_a_string"),
        ("inf", "not_a_string"),
        ("c3", "unknown_id"),
        ("C8", "beyond_k"),
    ]
    json.dumps(got, allow_nan=False)  # nothing in a dropped entry is non-JSON
    assert {d["reason"] for d in ch["dropped"]} <= set(R.DROP_REASONS)


def test_a_long_or_odd_dropped_entry_is_cut_and_safe() -> None:
    sheet = probe_sheet(EVENTFUL)
    cands = R.candidates(sheet)
    long = "C" + "9" * 100 + "\x1b[31m\n"
    got = R.parse_choice(
        json.dumps({"headline": "H1", "order": [long, "C1", "C2", "C3", "C4", "C5"]}), cands, 6
    )
    entry = got["choice"]["dropped"][0]["entry"]
    assert len(entry) <= R.ENTRY_CHARS and "\x1b" not in entry and "\n" not in entry


def test_the_headline_is_none_when_unknown_or_not_a_string() -> None:
    sheet = probe_sheet(EVENTFUL)
    cands = R.candidates(sheet)
    five = ["C1", "C2", "C3", "C4", "C5"]
    for bad in ("H2", "H9", "h1", "", 1, None, ["H1"], {"id": "H1"}, True):
        got = R.parse_choice(json.dumps({"headline": bad, "order": five}), cands, 6)
        assert got["outcome"] == "ok" and got["choice"]["headline"] is None, bad
    assert R.parse_choice(reply("H1", five), cands, 6)["choice"]["headline"] == "H1"
    two = probe_sheet(TWO_HEADS)
    assert R.parse_choice(reply("H2", five), R.candidates(two), 6)["choice"]["headline"] == "H2"


def test_fewer_entries_than_n_min_is_too_few_and_keeps_the_choice() -> None:
    sheet = probe_sheet(EVENTFUL)
    cands = R.candidates(sheet)
    for order in ([], ["C1"], ["C1", "C2", "C3", "C4"], ["C1", "C1", "C1", "C1", "C1", "C2"], ["C99"] * 6):
        got = R.parse_choice(reply("H1", order), cands, 6)
        assert got["outcome"] == "too_few", order
        assert got["choice"]["order"] == list(dict.fromkeys(i for i in order if i != "C99"))
        assert "needed" in got["error"]
    assert R.parse_choice(reply("H1", ["C1", "C2", "C3", "C4", "C5"]), cands, 6)["outcome"] == "ok"
    # a quiet chapter needs two, a chapter with three candidates needs all three
    q = probe_sheet(QUIET)
    assert R.parse_choice(reply("H1", ["C1"]), R.candidates(q), 3)["outcome"] == "too_few"
    assert R.parse_choice(reply("H1", ["C1", "C2"]), R.candidates(q), 3)["outcome"] == "ok"
    s = probe_sheet(SMALL)
    assert R.parse_choice(reply("H1", ["C1", "C2"]), R.candidates(s), 6)["outcome"] == "too_few"
    assert R.parse_choice(reply("H1", ["C3", "C1", "C2"]), R.candidates(s), 6)["choice"]["order"] == [
        "C3",
        "C1",
        "C2",
    ]


def test_the_reply_may_name_the_mandatory_sentences_and_k_is_a_hard_cap() -> None:
    sheet = probe_sheet(EVENTFUL)
    cands = R.candidates(sheet)
    got = R.parse_choice(reply("H1", [c["id"] for c in cands[0]]), cands, 6)
    assert len(got["choice"]["order"]) == 6
    assert [d["reason"] for d in got["choice"]["dropped"]] == ["beyond_k"] * 14


def test_judge_attempt_rederives_the_outcome_from_the_stored_response() -> None:
    sheet = probe_sheet(EVENTFUL)
    cands = R.candidates(sheet)
    text = reply("H1", ["C1", "C2", "C3", "C4", "C5"])
    assert R.judge_attempt({"response": text, "error": None}, cands, 6) == R.parse_choice(text, cands, 6)
    failed = R.judge_attempt({"response": None, "error": "timeout: x"}, cands, 6)
    assert failed == {"outcome": "failed", "error": "timeout: x", "choice": None}


def test_settle_picks_the_first_usable_attempt() -> None:
    ok = {"outcome": "ok", "choice": {"headline": "H1"}}
    ok_nohead = {"outcome": "ok", "choice": {"headline": None}}
    bad = lambda o: {"outcome": o, "choice": None}  # noqa: E731
    assert R.settle([]) == (None, None)
    assert R.settle([ok]) == (0, None)
    assert R.settle([ok_nohead]) == (0, "headline")
    assert R.settle([bad("invalid"), ok]) == (1, None)
    assert R.settle([bad("failed"), bad("too_few")]) == (None, "too_few")
    assert R.settle([bad("too_few"), bad("failed")]) == (None, "failed")
    assert R.settle([bad("invalid"), ok_nohead, ok]) == (1, "headline")


# --- narrate: the request and the policy -----------------------------------------------------------------------------


def test_the_request_body_names_the_model_and_carries_the_schema_and_the_seed() -> None:
    sheet = probe_sheet(EVENTFUL)
    client = Fake(first_n(6), model="google/gemma-4-26b-a4b-qat")
    rec = R.narrate(sheet, client)
    assert len(client.bodies) == 1
    body = client.bodies[0]
    assert set(body) == {
        "model", "messages", "temperature", "max_tokens", "seed", "stream", "response_format",
    }  # fmt: skip
    assert body["model"] == "google/gemma-4-26b-a4b-qat" and body["stream"] is False
    assert body["temperature"] == 0.4 and body["max_tokens"] == 200
    assert body["seed"] == R.select_seed(sheet["run"]["seed"], sheet["chapter"][1], 0)
    assert body["messages"] == rec["messages"] == R.select_messages(sheet, R.candidates(sheet))
    fmt = body["response_format"]
    assert fmt["type"] == "json_schema" and fmt["json_schema"]["name"] == "world_report_select"
    assert fmt["json_schema"]["strict"] is True
    assert fmt["json_schema"]["schema"] == R.select_schema(R.candidates(sheet), 6)
    assert rec["model"] == "google/gemma-4-26b-a4b-qat" and rec["url"] == Fake.url
    assert rec["request"] == {"temperature": 0.4, "max_tokens": 200} and rec["mode"] == "select"


def test_a_good_first_reply_is_used_and_orders_only_inside_sections() -> None:
    sheet = probe_sheet(EVENTFUL)
    cands = R.candidates(sheet)
    pick = [c["id"] for c in cands[0] if not c["mandatory"]][2:7]
    rec = R.narrate(sheet, Fake(lambda b: reply("H1", pick)))
    assert rec["fallback"] is None and [a["outcome"] for a in rec["attempts"]] == ["ok"]
    origins = {x["id"]: x["origin"] for x in rec["report"]["body"]}
    assert set(origins) == set(pick) | {c["id"] for c in cands[0] if c["mandatory"]}
    assert all(origins[i] == "model" for i in pick) and {origins[i] for i in origins if i not in pick} == {
        "mandatory"
    }
    sections = [
        templates.SECTION_ORDER.index(next(c["section"] for c in cands[0] if c["id"] == x["id"]))
        for x in rec["report"]["body"]
    ]
    assert sections == sorted(sections)
    assert rec["report"]["headline"]["origin"] == "model" and rec["report"]["headline"]["id"] == "H1"


def test_a_chosen_headline_alternative_is_shown() -> None:
    sheet = probe_sheet(TWO_HEADS)
    rec = R.narrate(sheet, Fake(lambda b: reply("H2", json.loads(first_n(5)(b))["order"])))
    cands = R.candidates(sheet)
    assert rec["report"]["headline"] == {"id": "H2", "text": cands[1][1]["text"], "origin": "model"}


def test_retry_after_invalid_failed_and_too_few_uses_the_next_seed_and_the_same_messages() -> None:
    sheet = probe_sheet(EVENTFUL)
    for first in ("not json", None, reply("H1", ["C1"]), '{"headline": "H1"}', ""):
        client = Fake(first, first_n(5))
        rec = R.narrate(sheet, client)
        assert [a["outcome"] for a in rec["attempts"]][-1] == "ok" and len(rec["attempts"]) == 2, first
        assert rec["fallback"] is None
        assert client.bodies[0]["messages"] == client.bodies[1]["messages"]
        assert client.bodies[0]["seed"] != client.bodies[1]["seed"]
        assert [a["seed"] for a in rec["attempts"]] == [client.bodies[0]["seed"], client.bodies[1]["seed"]]
        assert rec["report"]["body"][0]["origin"] in ("model", "mandatory")
        assert R.verify_record(json.loads(json.dumps(rec)), sheet)["status"] == "ok"


def test_no_retry_after_a_usable_reply_and_attempts_bound_the_requests() -> None:
    sheet = probe_sheet(EVENTFUL)
    client = Fake(first_n(5), "never asked")
    R.narrate(sheet, client, attempts=2)
    assert len(client.bodies) == 1
    for n in (1, 2, 3):
        client = Fake("bad")
        rec = R.narrate(sheet, client, attempts=n)
        assert len(client.bodies) == n and len(rec["attempts"]) == n and rec["fallback"] == "invalid"
    with pytest.raises(ValueError):
        R.narrate(sheet, Fake("x"), attempts=0)


@pytest.mark.parametrize(
    ("script", "fallback"),
    [
        (["bad", "worse"], "invalid"),
        ([None, None], "failed"),
        (["bad", None], "failed"),
        ([None, reply("H1", ["C1"])], "too_few"),
        ([reply("H1", ["C1"]), "bad"], "invalid"),
    ],
)
def test_after_the_attempts_the_template_report_is_used(script: list[Any], fallback: str) -> None:
    sheet = probe_sheet(EVENTFUL)
    rec = R.narrate(sheet, Fake(*script))
    assert rec["fallback"] == fallback and rec["mode"] == "select"
    assert rec["report"] == json.loads(json.dumps(R.template_report(sheet)))
    assert {x["origin"] for x in rec["report"]["body"]} <= {"mandatory", "template"}
    assert rec["report"]["headline"]["origin"] == "template"
    assert R.verify_record(json.loads(json.dumps(rec)), sheet)["status"] == "ok"


def test_an_unknown_headline_falls_back_to_the_template_headline_only() -> None:
    sheet = probe_sheet(EVENTFUL)
    rec = R.narrate(sheet, Fake(lambda b: reply("H7", json.loads(first_n(5)(b))["order"])))
    assert rec["fallback"] == "headline" and len(rec["attempts"]) == 1
    assert rec["report"]["headline"]["origin"] == "template"
    assert any(x["origin"] == "model" for x in rec["report"]["body"])  # the order was still used
    assert R.verify_record(json.loads(json.dumps(rec)), sheet)["status"] == "ok"


def test_attempt_entries_hold_latency_finish_reason_usage_and_the_cleaned_reply() -> None:
    sheet = probe_sheet(EVENTFUL)
    rec = R.narrate(sheet, Fake(lambda b: first_n(5)(b).replace(", ", ",\n\t\x07 ")))
    att = rec["attempts"][0]
    assert att["latency_s"] == 0.5 and att["finish_reason"] == "stop"
    assert att["usage"] == {"prompt_tokens": 800, "completion_tokens": 30, "total_tokens": 830}
    assert "\x1b" not in att["response"] and "\n" not in att["response"]
    assert att["outcome"] == "ok" and att["error"] is None and att["choice"]["dropped"] == []
    failed = R.narrate(sheet, Fake(None), attempts=1)["attempts"][0]
    assert failed["outcome"] == "failed" and failed["response"] is None and failed["choice"] is None
    assert (
        failed["error"].startswith("timeout") and failed["usage"] is None and failed["finish_reason"] is None
    )


def test_an_empty_or_control_only_reply_is_a_failed_attempt() -> None:
    sheet = probe_sheet(EVENTFUL)
    rec = R.narrate(sheet, Fake("\x1b\x07", "   \n"))
    assert [a["outcome"] for a in rec["attempts"]] == ["failed", "failed"] and rec["fallback"] == "failed"
    assert all(a["response"] is None for a in rec["attempts"])
    assert R.verify_record(json.loads(json.dumps(rec)), sheet)["status"] == "ok"


def test_an_overlong_reply_is_invalid_and_replays_the_same() -> None:
    sheet = probe_sheet(EVENTFUL)
    rec = R.narrate(sheet, Fake(reply("H1", ["C1"] * 2000)), attempts=1)
    att = rec["attempts"][0]
    assert att["outcome"] == "invalid" and len(att["response"]) == R.MAX_REPLY_CHARS + 1
    assert R.verify_record(json.loads(json.dumps(rec)), sheet)["status"] == "ok"


def test_nothing_to_choose_means_no_request() -> None:
    sheet = mk_sheet(kinds={"population": [pop(100, 120)]})
    cands = R.candidates(sheet)
    assert cands[0] == []  # no body candidate
    client = Fake("never")
    rec = R.narrate(sheet, client)
    assert (
        client.bodies == [] and rec["mode"] == "select" and rec["attempts"] == [] and rec["fallback"] is None
    )
    assert rec["messages"] is None and rec["request"] is None
    assert rec["report"] == json.loads(json.dumps(R.template_report(sheet)))
    assert R.verify_record(json.loads(json.dumps(rec)), sheet)["status"] == "ok"


def test_the_template_record_has_no_model_fields() -> None:
    sheet = probe_sheet(EVENTFUL)
    rec = R.narrate(sheet, None)
    assert rec["mode"] == "template" and rec["model"] is None and rec["url"] is None
    assert (
        rec["messages"] is None
        and rec["request"] is None
        and rec["attempts"] == []
        and rec["fallback"] is None
    )
    assert rec["report"] == json.loads(json.dumps(R.template_report(sheet)))
    assert rec["ticker"] == json.loads(json.dumps(R.ticker(sheet)))


# --- the record -------------------------------------------------------------------------------------------------------


def test_the_record_has_the_specified_keys_in_order_and_is_plain_json() -> None:
    sheet = probe_sheet(EVENTFUL)
    rec = R.narrate(sheet, Fake(first_n(5)))
    assert list(rec) == [
        "step", "chapter_start", "report_version", "template_version", "checker_version", "select_prompt_version",
        "report_rules_digest", "template_digest", "lexicon_digest", "label_table_digest", "known_social_digest",
        "source_words_digest", "select_prompt_digest", "facts_digest", "candidate_set_digest", "facts", "candidates",
        "headlines", "excluded_candidates", "mode", "model", "url", "messages", "request", "attempts", "fallback",
        "report", "ticker",
    ]  # fmt: skip
    assert rec["step"] == sheet["chapter"][1] and rec["chapter_start"] == sheet["chapter"][0]
    text = json.dumps(rec, allow_nan=False)
    assert json.loads(text) == rec  # no tuple, no non-finite number
    assert set(rec["candidates"][0]) == {"id", "template", "cites", "text", "mandatory", "salience"}
    assert set(rec["headlines"][0]) == {"id", "template", "cites", "text"}
    assert set(rec["attempts"][0]) == {
        "seed",
        "latency_s",
        "finish_reason",
        "usage",
        "outcome",
        "error",
        "response",
        "choice",
    }
    assert set(rec["attempts"][0]["choice"]) == {"headline", "order", "dropped"}
    assert rec["facts"] == sheet and rec["facts_digest"] == R.facts_digest(sheet)
    assert rec["excluded_candidates"] == []
    assert {k: rec[k] for k in R.record_versions()} == R.record_versions()


def test_the_record_is_deterministic_given_the_same_exchanges() -> None:
    sheet = probe_sheet(INTERVENED)
    a = R.narrate(sheet, Fake(first_n(5)))
    b = R.narrate(sheet, Fake(first_n(5)))
    assert json.dumps(a) == json.dumps(b)
    assert R.narrate(sheet, None)["candidate_set_digest"] == a["candidate_set_digest"]


def test_record_versions_and_digests_are_pinned() -> None:
    v = R.record_versions()
    assert (v["report_version"], v["template_version"], v["checker_version"], v["select_prompt_version"]) == (
        1,
        1,
        1,
        1,
    )
    assert (
        v["report_rules_digest"] == R.REPORT_V1.digest() and v["template_digest"] == templates.TEMPLATE_DIGEST
    )
    assert v["source_words_digest"] == "218eec1a2629d43dbcead4cb7eb08535dd18796879ff71d666c1c98bff2ffeb4"
    assert v["select_prompt_digest"] == "7800edadecea2996e4e3b80d229b2d8e66c9f29512e9ebccdd32a9e456d938cb"
    assert R.source_words_digest() == v["source_words_digest"]


# --- random replies, whatever the sheet -------------------------------------------------------------------------


def _random_reply(rng: random.Random, body_ids: list[str]) -> str:
    pool = [*body_ids, *body_ids, "C0", "C99", "H1", "x", 3, None, 1.5, ["C1"], {"a": 1}, True]
    kind = rng.random()
    if kind < 0.04:
        return rng.choice(["", "[]", "null", "{", '{"headline": "H1"}', "NaN"])
    order = [rng.choice(pool) for _ in range(rng.choice([0, 1, 2, 4, 5, 6, 7, 12, 40]))]
    return json.dumps({"headline": rng.choice([None, "H1", "H1", "H2", "H9", 4]), "order": order})


def test_two_thousand_random_replies_never_break_the_report() -> None:
    """Any reply, on any sheet: lead first, every mandatory sentence once, no repeat, the body in the section order,
    events in time order, the complete block last, a headline candidate, only candidate strings, at most k chosen."""
    seen = {"ok": 0, "too_few": 0, "invalid": 0}
    for seed in range(2000):
        rng = random.Random(seed)
        sheet = random_sheet(rng)
        cands = R.candidates(sheet)
        body, heads, block, _ = cands
        if not body or not heads:
            continue
        k = R.body_k(sheet, body)
        got = R.parse_choice(_random_reply(rng, [c["id"] for c in body]), cands, k)
        seen[got["outcome"]] += 1
        assert got["outcome"] in ("ok", "too_few", "invalid")
        choice = None
        if got["outcome"] == "ok":
            ch = got["choice"]
            assert len(ch["order"]) == len(set(ch["order"])) <= k and len(ch["order"]) >= R.select_n_min(
                k, body
            )
            assert ch["headline"] in (None, *[h["id"] for h in heads])
            choice = {"headline": ch["headline"], "order": ch["order"]}
        rep = R.assemble(sheet, cands, choice)
        by = {c["id"]: c for c in body}
        ids = [x["id"] for x in rep["body"]]
        assert len(set(ids)) == len(ids) and set(ids) <= set(by)
        assert {c["id"] for c in body if c["mandatory"]} <= set(ids)
        sec = [templates.SECTION_ORDER.index(by[i]["section"]) for i in ids]
        assert sec == sorted(sec), (seed, ids)
        times = [by[i]["order"][1] for i in ids if by[i]["section"] == "E"]
        assert times == sorted(times)
        assert rep["lead"] == R.lead_of(sheet)["text"]
        assert rep["headline"]["text"] in {h["text"] for h in heads}
        assert [x["text"] for x in rep["interventions"]] == [x["text"] for x in block]
        assert all(x["text"] == by[x["id"]]["text"] for x in rep["body"])
        assert sum(1 for x in rep["body"] if x["origin"] == "model") <= k
        if choice is not None:
            assert {x["id"] for x in rep["body"] if x["origin"] == "model"} == set(choice["order"])
    assert all(n > 50 for n in seen.values()), seen


def test_two_hundred_random_narrations_replay_to_ok() -> None:
    for seed in range(200):
        rng = random.Random(10_000 + seed)
        sheet = random_sheet(rng)
        body = R.candidates(sheet)[0]
        script = [_random_reply(rng, [c["id"] for c in body]) for _ in range(2)]
        rec = json.loads(json.dumps(R.narrate(sheet, Fake(*script))))
        res = R.verify_record(rec, sheet)
        assert res["status"] == "ok" and res["reasons"] == [], (seed, res)


# --- replay ------------------------------------------------------------------------------------------------------------


@pytest.fixture
def select_record() -> tuple[dict[str, Any], dict[str, Any]]:
    sheet = probe_sheet(INTERVENED)
    rec = json.loads(json.dumps(R.narrate(sheet, Fake("not json", first_n(5)))))
    assert [a["outcome"] for a in rec["attempts"]] == ["invalid", "ok"]
    return sheet, rec


def status(rec: dict[str, Any], sheet: dict[str, Any] | None) -> dict[str, Any]:
    return R.verify_record(json.loads(json.dumps(rec)), sheet)


def test_a_clean_record_verifies_for_both_modes(select_record) -> None:
    sheet, rec = select_record
    assert status(rec, sheet) == {"status": "ok", "sheet_differs": False, "versions": {}, "reasons": []}
    assert status(json.loads(json.dumps(R.narrate(sheet, None))), sheet)["status"] == "ok"
    assert status(rec, None)["status"] == "ok"  # no recomputed sheet to compare with


def test_verify_never_opens_a_socket_or_calls_urlopen(select_record, monkeypatch) -> None:
    sheet, rec = select_record

    def boom(*a, **k):
        raise AssertionError("network used")

    monkeypatch.setattr(socket, "socket", boom)
    monkeypatch.setattr(urllib.request, "urlopen", boom)
    assert status(rec, sheet)["status"] == "ok"


def mutate(rec: dict[str, Any], path: list[Any], value: Any) -> dict[str, Any]:
    out = json.loads(json.dumps(rec))
    cur = out
    for p in path[:-1]:
        cur = cur[p]
    cur[path[-1]] = value
    return out


@pytest.mark.parametrize(
    ("path", "value", "needle"),
    [
        (["messages", 1, "content"], "an edited prompt", "messages"),
        (["messages", 0, "content"], "an edited system message", "messages"),
        (["request", "temperature"], 0.9, "request"),
        (["report", "lead"], "The population rose.", "report"),
        (["report", "body", 0, "text"], "There were no deaths.", "report"),
        (["report", "body", 0, "origin"], "zzz", "report"),
        (["report", "headline", "text"], "Population rose to 9", "report"),
        (["report", "interventions"], [], "report"),
        (["ticker", 0, "text"], "Step 1: x.", "ticker"),
        (["candidates", 0, "text"], "x", "candidates"),
        (["candidates", 1, "mandatory"], True, "candidates"),
        (["headlines", 0, "text"], "x", "headlines"),
        (["candidate_set_digest"], "0" * 64, "candidate_set_digest"),
        (["facts", "labels", "env_default"], False, "facts_digest"),
        (["facts_digest"], "0" * 64, "facts_digest"),
        (
            ["attempts", 1, "response"],
            '{"headline": "H1", "order": ["C1", "C2", "C3", "C4", "C6"]}',
            "attempts[1]",
        ),
        (["attempts", 1, "outcome"], "invalid", "attempts[1]"),
        (["attempts", 0, "outcome"], "ok", "attempts[0]"),
        (["attempts", 0, "seed"], 7, "seed"),
        (["attempts", 1, "choice", "order"], ["C1"], "attempts[1]"),
        (["fallback"], "headline", "fallback"),
        (["excluded_candidates"], [{"template": "X", "cites": [], "reasons": ["r"]}], "excluded_candidates"),
        (["step"], 1, "step"),
    ],
)
def test_any_edit_of_a_stored_record_is_a_mismatch(select_record, path, value, needle) -> None:
    sheet, rec = select_record
    got = status(mutate(rec, path, value), sheet)
    assert got["status"] == "mismatch", got
    assert any(needle in r for r in got["reasons"]), got["reasons"]


def test_dropping_or_adding_an_attempt_or_a_choice_changes_the_replay(select_record) -> None:
    sheet, rec = select_record
    extra = json.loads(json.dumps(rec))
    extra["attempts"].append(extra["attempts"][1])
    assert "follows" in " ".join(status(extra, sheet)["reasons"])
    template = json.loads(json.dumps(rec))
    template.update({"mode": "template"})
    assert status(template, sheet)["status"] == "mismatch"
    ghost = mutate(R.narrate(sheet, None), ["model"], "ghost-model")
    assert status(ghost, sheet)["status"] == "mismatch"
    cut = mutate(rec, ["mode"], "weird")
    assert status(cut, sheet)["status"] == "mismatch"


def test_a_record_that_is_not_a_record_is_a_mismatch_not_a_crash() -> None:
    sheet = probe_sheet(EVENTFUL)
    rec = json.loads(json.dumps(R.narrate(sheet, None)))
    broken = {k: v for k, v in rec.items() if k != "facts"}
    got = R.verify_record(broken, sheet)
    assert got["status"] == "mismatch" and "cannot be replayed" in got["reasons"][0]
    assert R.verify_record({}, sheet)["status"] == "versions_differ"


def test_stale_when_the_run_gives_another_sheet_under_equal_versions(select_record) -> None:
    sheet, rec = select_record
    other = json.loads(json.dumps(sheet))
    other["facts"][1]["to"] += 1
    got = status(rec, other)
    assert got["status"] == "stale" and got["sheet_differs"] is True and got["reasons"] == []


def test_a_stale_sheet_does_not_hide_a_mismatch(select_record) -> None:
    sheet, rec = select_record
    other = json.loads(json.dumps(sheet))
    other["facts"][1]["to"] += 1
    got = status(mutate(rec, ["report", "lead"], "x"), other)
    assert got["status"] == "mismatch" and got["sheet_differs"] is True


def test_other_versions_are_versions_differ_with_both_comparisons_reported(select_record) -> None:
    sheet, rec = select_record
    for key in ("report_version", "template_version", "checker_version", "select_prompt_version"):
        got = status(mutate(rec, [key], 99), sheet)
        assert (
            got["status"] == "versions_differ"
            and got["versions"] == {key: [99, 1]}
            and not got["sheet_differs"]
        )
    for key in (
        "report_rules_digest",
        "template_digest",
        "lexicon_digest",
        "label_table_digest",
        "known_social_digest",
        "source_words_digest",
        "select_prompt_digest",
    ):
        assert status(mutate(rec, [key], "0" * 64), sheet)["status"] == "versions_differ", key
    # N17: other versions and another sheet are reported together (the run or the code changed: not told apart)
    other = json.loads(json.dumps(sheet))
    other["facts"][1]["to"] += 1
    got = status(mutate(rec, ["template_version"], 0), other)
    assert got["status"] == "versions_differ" and got["sheet_differs"] is True
    # and a record with other versions is not replayed from its facts (a stale text is not a mismatch)
    got = status(mutate(mutate(rec, ["template_version"], 0), ["report", "lead"], "x"), sheet)
    assert got["status"] == "versions_differ" and got["reasons"] == []


def test_a_changed_label_table_known_social_or_source_words_is_versions_differ_not_stale(
    select_record, monkeypatch
) -> None:
    """E1: the sheet depends on the content of these tables, which the version numbers do not pin."""
    sheet, rec = select_record
    row = next(iter(R.LABEL_VERDICTS.items()))
    monkeypatch.setitem(R.LABEL_VERDICTS, (2, "d" * 64, 2, "w" * 64), row[1])
    got = status(rec, sheet)
    assert got["status"] == "versions_differ" and list(got["versions"]) == ["label_table_digest"]
    monkeypatch.undo()
    monkeypatch.setitem(R.KNOWN_SOCIAL, "e" * 64, (250, 16))
    got = status(rec, sheet)
    assert got["status"] == "versions_differ" and list(got["versions"]) == ["known_social_digest"]
    monkeypatch.undo()
    monkeypatch.setitem(templates.SOURCE_WORDS, "none", "the system")
    got = status(rec, sheet)
    assert got["status"] == "versions_differ" and list(got["versions"]) == ["source_words_digest"]
    monkeypatch.undo()
    assert status(rec, sheet)["status"] == "ok"
    # a changed prompt with the same version number is also versions_differ
    monkeypatch.setattr(R, "SELECT_SYSTEM", R.SELECT_SYSTEM + " Be brief.")
    assert list(status(rec, sheet)["versions"]) == ["select_prompt_digest"]


def test_ticker_and_report_replay_from_the_stored_facts_even_when_the_run_changed(select_record) -> None:
    sheet, rec = select_record
    assert status(rec, {**sheet, "run": {"seed": 1}})["status"] == "stale"


def test_deaths_only_sheet_round_trip() -> None:
    sheet = mk_sheet(kinds={"population": [pop(1000, 900, 850, 1_517_900)], "deaths": [deaths(5, 1, 0)]})
    rec = json.loads(json.dumps(R.narrate(sheet, Fake(first_n(1)))))
    assert rec["fallback"] in (None, "too_few", "headline")
    assert status(rec, sheet)["status"] == "ok"
