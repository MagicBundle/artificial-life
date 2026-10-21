"""A local mock of an OpenAI-compatible chat server with LM Studio's read-only ``/api/v0/models`` listing, for the
world report's tests (docs/DESIGN.md §31). It binds ``127.0.0.1`` on a free port; no test reaches any other host.

``Server(models={"m": "loaded"})`` answers ``GET /api/v0/models`` with that list (``models=None``: the endpoint
does not exist, a 404, like another server). ``POST`` bodies are recorded in ``requests``; the replies are the
queued texts in order (the last one repeats), or ``respond(body)`` when set."""

# ruff: noqa: E501

from __future__ import annotations

import json
import threading
from collections import deque
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


def envelope(content: str | None, finish: str = "stop") -> bytes:
    return json.dumps(
        {
            "choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": finish}],
            "usage": {"prompt_tokens": 900, "completion_tokens": 40, "total_tokens": 940},
        }
    ).encode()


class Server:
    def __init__(self, models: dict[str, str] | None = None) -> None:
        self.models = models
        self.requests: list[dict[str, Any]] = []
        self.gets: list[str] = []
        self.queue: deque[bytes | int] = deque()  # bytes: a 200 response body; int: an HTTP error status
        self.last: bytes | int = 500
        self.respond: Callable[[dict[str, Any]], str] | None = None
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def _send(self, status: int, body: bytes = b"") -> None:
                self.send_response(status)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                outer.gets.append(self.path)
                if outer.models is None or self.path != "/api/v0/models":
                    self._send(404)
                    return
                data = [{"id": k, "object": "model", "state": v} for k, v in outer.models.items()]
                self._send(200, json.dumps({"data": data, "object": "list"}).encode())

            def do_POST(self) -> None:
                size = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(size))
                outer.requests.append(body)
                if outer.respond is not None:
                    self._send(200, envelope(outer.respond(body)))
                    return
                spec = outer.queue.popleft() if outer.queue else outer.last
                outer.last = spec
                if isinstance(spec, int):
                    self._send(spec)
                else:
                    self._send(200, spec)

            def log_message(self, *args: Any) -> None:
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.thread = threading.Thread(
            target=self.httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        )
        self.thread.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_address[1]}/v1/chat/completions"

    def say(self, *texts: str | int | None) -> None:
        """Queue replies: a string is the assistant's content, ``None`` an empty content, an int an HTTP status."""
        self.queue.extend(t if isinstance(t, int) else envelope(t) for t in texts)

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


def schema_of(body: dict[str, Any]) -> dict[str, Any]:
    return body["response_format"]["json_schema"]["schema"]


def valid_reply(
    body: dict[str, Any], n: int | None = None, headline: str | None = None, skip: int = 0
) -> str:
    """A reply that the schema of ``body`` allows: the headline ``headline`` (default: the first) and ``n`` body ids
    (default: the schema's ``maxItems``) starting at the ``skip``-th candidate."""
    s = schema_of(body)
    ids = s["properties"]["order"]["items"]["enum"]
    size = s["properties"]["order"]["maxItems"] if n is None else n
    head = headline or s["properties"]["headline"]["enum"][0]
    return json.dumps({"headline": head, "order": ids[skip : skip + size]})
