"""Fixtures for the harness tests. Nothing here touches the network.

The stub endpoint is a real HTTP server speaking the OpenAI protocol, not a
mocked client. What we most need to know about this harness is whether it copes
with what a provider actually sends back -- a reasoning field under either of two
names, a truncated choice, a 500 -- and a mock of our own client would only tell
us what we already believe.
"""
from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, List, Optional

import pytest

import arggym
from arggym.core.freeze import PROMPT_VERSION, SCORING_VERSION, taskset_hash

#: Every task, not a sample of two. The four answer shapes are spread across
#: the twelve, and the end-to-end test exists to catch an extraction path that
#: mishandles one; two tasks cover two shapes. Building all twelve takes about
#: a second and a half.
TASKS = tuple(sorted(arggym.task_names()))
#: One exhibit row per task, at the cheapest cell the grid names. This suite tests the
#: harness -- what it sends, what it does with what comes back -- and a harder item would
#: change the price of the fixture and nothing it asserts.
LEVEL, ORDERING, SEED = 3, "last_link_elitist", 0


@pytest.fixture(scope="session")
def rows() -> List[Dict[str, Any]]:
    out = []
    for task in TASKS:
        ds = arggym.TaskDataset(task, LEVEL, ORDERING, size=1, seed=SEED)
        row = ds[0]
        row["metadata"]["source_index"] = len(out)
        out.append(row)
    return out


@pytest.fixture()
def taskset_file(tmp_path, rows) -> str:
    """A frozen taskset on disk, manifest line and all."""
    path = os.fspath(tmp_path / "taskset.jsonl")
    with open(path, "w") as f:
        f.write(json.dumps({"__manifest__": {
            "taskset_hash": taskset_hash(rows), "n_items": len(rows),
            "versions": {"arggym": arggym.__version__, "prompt_version": PROMPT_VERSION,
                         "scoring_version": SCORING_VERSION}}}) + "\n")
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    return path


class StubProvider:
    """An OpenAI-compatible endpoint whose replies the test dictates."""

    def __init__(self, handler: Callable[[Dict[str, Any]], Dict[str, Any]]) -> None:
        self.requests: List[Dict[str, Any]] = []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a: Any) -> None:
                pass

            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append({"body": body, "headers": dict(self.headers)})
                reply = handler(body)
                status = reply.pop("__status__", 200)
                data = json.dumps(reply).encode()
                try:
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError):
                    # The client gave up and went away, which is what a handler
                    # deliberately slower than the timeout is arranged to make
                    # happen. Answering nobody is the expected end of that
                    # request, not a failure worth a stack trace per item.
                    pass

        # Threading, so a handler that is deliberately slow does not hold the
        # next request behind it. A client that gives up on a slow request and
        # retries is exactly what the timeout tests exercise, and on a
        # single-threaded server the retry queues behind the sleep it was
        # trying to escape and times out too.
        self._srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self._srv.serve_forever, daemon=True).start()

    @property
    def url(self) -> str:
        host, port = self._srv.server_address
        return f"http://{host}:{port}/v1"

    def stop(self) -> None:
        self._srv.shutdown()


def completion(content: str, finish_reason: str = "stop",
               reasoning_key: Optional[str] = None,
               reasoning: str = "") -> Dict[str, Any]:
    message: Dict[str, Any] = {"role": "assistant", "content": content}
    if reasoning_key:
        message[reasoning_key] = reasoning
    return {"id": "stub", "object": "chat.completion", "created": 0, "model": "stub",
            "choices": [{"index": 0, "message": message,
                         "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}}


@pytest.fixture()
def provider():
    made: List[StubProvider] = []

    def make(handler: Callable[[Dict[str, Any]], Dict[str, Any]]) -> StubProvider:
        p = StubProvider(handler)
        made.append(p)
        return p

    yield make
    for p in made:
        p.stop()


def answering(rows: List[Dict[str, Any]], wrap: str = "<answer>\n{}\n</answer>",
              **kw: Any) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """A handler that replies to each row with that row's own reference answer.

    Keyed by the whole question, and the longest match wins. `counter_argument` and
    `counter_argument_strict` are an ablation pair over one theory, so their questions
    share a long prefix: at this fixture's cell they first differ 368 characters in, and
    where the theory is identical they differ only in the permitted-forms block at the
    end. A 300-character key put both rows in one dict entry, so the plain row was
    answered with the strict row's reference -- a strict rule its own scorer refuses --
    and the end-to-end test read that as the harness losing an answer (#37).
    """
    keyed = {r["question"]: r["reference_answer"] for r in rows}
    assert len(keyed) == len(rows), (
        "two rows share a question, so one would be answered with the other's reference")

    def handler(body: Dict[str, Any]) -> Dict[str, Any]:
        user = [m for m in body["messages"] if m["role"] == "user"][-1]["content"]
        hit = max((k for k in keyed if k in user), key=len, default=None)
        return completion(wrap.format(keyed[hit]) if hit else "no answer here", **kw)

    return handler
