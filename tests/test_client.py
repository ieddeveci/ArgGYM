import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.client import build_payload, complete, run_batch


def _serve(message: dict, port: int):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a, **k):
            pass

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            self.rfile.read(n)
            body = json.dumps({
                "choices": [{"message": message, "finish_reason": "stop"}],
                "usage": {"completion_tokens": 7},
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    srv = HTTPServer(("127.0.0.1", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def test_reasoning_field_new_vllm_name_is_captured():
    """Newer vLLM returns chain-of-thought as `reasoning`. Reading only
    `reasoning_content` silently discards every CoT."""
    srv = _serve({"content": "ans", "reasoning": "long thoughts"}, 8981)
    try:
        r = complete("http://127.0.0.1:8981", "m", "p", {}, 16)
        assert r["reasoning_content"] == "long thoughts"
        assert r["content"] == "ans"
    finally:
        srv.shutdown()


def test_reasoning_field_old_vllm_name_is_captured():
    srv = _serve({"content": "ans", "reasoning_content": "old style"}, 8982)
    try:
        r = complete("http://127.0.0.1:8982", "m", "p", {}, 16)
        assert r["reasoning_content"] == "old style"
    finally:
        srv.shutdown()


def test_connection_failure_returns_error_not_raises():
    r = complete("http://127.0.0.1:9", "m", "p", {}, 16, timeout_s=2, retries=0)
    assert r["error"] and r["content"] == ""


def test_payload_omits_unset_sampling_params():
    b = build_payload("m", "p", {"temperature": 0.6, "top_k": None}, 128)
    assert b["temperature"] == 0.6
    assert "top_k" not in b
    assert b["max_tokens"] == 128


def test_run_batch_streams_every_result():
    seen = []
    out = run_batch([{"sample_id": i} for i in range(10)],
                    lambda it: {"sample_id": it["sample_id"]}, 4, seen.append)
    assert len(out) == 10 and len(seen) == 10
    assert {r["sample_id"] for r in out} == set(range(10))


def test_run_batch_survives_a_crashing_worker():
    def boom(it):
        if it["sample_id"] == 2:
            raise ValueError("nope")
        return {"sample_id": it["sample_id"]}

    out = run_batch([{"sample_id": i} for i in range(4)], boom, 2)
    errs = [r for r in out if r.get("error")]
    assert len(out) == 4 and len(errs) == 1


def test_enable_thinking_sent_only_when_declared():
    """Templates disagree on the default -- Qwen3.x thinks unless disabled,
    Gemma-4 stays silent unless enabled -- so the flag must be explicit."""
    on = build_payload("m", "p", {}, 16, enable_thinking=True)
    assert on["chat_template_kwargs"] == {"enable_thinking": True}

    off = build_payload("m", "p", {}, 16, enable_thinking=False)
    assert off["chat_template_kwargs"] == {"enable_thinking": False}

    # Non-thinking models (Llama-3.1) must not receive the kwarg at all.
    unset = build_payload("m", "p", {}, 16, enable_thinking=None)
    assert "chat_template_kwargs" not in unset
