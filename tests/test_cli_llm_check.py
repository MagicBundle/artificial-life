"""``alife llm-check`` (DESIGN §21): one structured-output request under the director's own client and schema.

A mock OpenAI-compatible server stands in for LM Studio (the helpers of ``test_director_llm``); no real model
is ever needed.
"""

from __future__ import annotations

import socket
from collections.abc import Iterator

import pytest

from alife import cli
from alife.config import load_config
from alife.director import llm
from test_director_llm import NONE, Mock, Spec, chat, reply


@pytest.fixture
def mock() -> Iterator[Mock]:
    m = Mock()
    yield m
    m.close()


def check(
    capsys: pytest.CaptureFixture[str], url: str, *extra: str, model: str = "test-model"
) -> tuple[int, str]:
    code = cli.main(["llm-check", "--llm-url", url, "--llm-model", model, *extra])
    return code, capsys.readouterr().out


def test_ok_prints_one_line_and_exits_zero(mock: Mock, capsys: pytest.CaptureFixture[str]) -> None:
    mock.say(reply(NONE, journal="ok"))
    code, out = check(capsys, mock.url, "--timeout", "5")
    assert code == 0
    assert out.startswith("OK model=test-model latency=") and out.endswith("s\n") and out.count("\n") == 1


def test_the_request_is_the_directors_own(mock: Mock, capsys: pytest.CaptureFixture[str]) -> None:
    mock.say(reply(NONE))
    check(capsys, mock.url)
    (body,) = mock.requests
    cfg = load_config()
    assert body["model"] == "test-model" and body["stream"] is False
    assert body["response_format"] == llm.response_format(cfg)
    assert [m["role"] for m in body["messages"]] == ["system", "user"]


@pytest.mark.parametrize(
    ("spec", "kind"),
    [
        (Spec(b"{}", status=500), "http: status 500"),
        (Spec(b"{}", status=404), "http: status 404"),
        (chat(None), "empty:"),
        (chat("   "), "empty:"),
        (chat("this is not json"), "invalid: not valid JSON"),
        (chat('{"actions": [], "journal": "x"}'), "invalid: actions must be a list"),
        (
            chat(reply({"action": "dance", "params": {}, "expected_effect": "x"})),
            "invalid: actions[0].action",
        ),
        (Spec(b"not an envelope"), "protocol:"),
    ],
)
def test_a_bad_reply_fails_with_its_kind(
    mock: Mock, capsys: pytest.CaptureFixture[str], spec: Spec, kind: str
) -> None:
    mock.say(spec)
    code, out = check(capsys, mock.url)
    assert code == 1 and out.startswith(f"FAIL {kind}") and out.count("\n") == 1


def test_a_slow_server_times_out(mock: Mock, capsys: pytest.CaptureFixture[str]) -> None:
    mock.say(chat(reply(NONE), delay=1.5))
    code, out = check(capsys, mock.url, "--timeout", "0.2")
    assert code == 1 and out.startswith("FAIL timeout:")


def test_a_closed_port_is_a_connection_failure(capsys: pytest.CaptureFixture[str]) -> None:
    with socket.socket() as s:  # a port nobody listens on
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    code, out = check(capsys, f"http://127.0.0.1:{port}/v1/chat/completions")
    assert code == 1 and out.startswith("FAIL connection:")


@pytest.mark.parametrize(
    "extra", [["--llm-url", "ftp://host/x"], ["--llm-url", "localhost:1234"], ["--llm-model", " "]]
)
def test_a_bad_url_or_model_is_a_config_error(extra: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["llm-check", *extra]) == 2
    captured = capsys.readouterr()
    assert captured.out == "" and "config error" in captured.err


@pytest.mark.parametrize("timeout", ["0", "-1", "nan", "inf", "x"])
def test_the_timeout_must_be_positive_and_finite(timeout: str) -> None:
    assert cli.main(["llm-check", "--timeout", timeout]) == 2
