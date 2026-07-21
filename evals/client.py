"""Chat client for the vLLM OpenAI-compatible server.

Uses stdlib urllib and a thread pool rather than an async stack: the repo's
existing client (generate_argumentations.py) is urllib-based, and blocking HTTP
against a local server does not justify a new dependency.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Iterable, List, Optional


def build_payload(model: str, prompt: str, sampling: dict, max_tokens: int,
                  enable_thinking: Optional[bool] = None,
                  system: Optional[str] = None) -> dict:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "max_tokens": int(max_tokens),
    }
    for key in ("temperature", "top_p", "top_k"):
        val = sampling.get(key)
        if val is not None:
            body[key] = val
    if enable_thinking is not None:
        # Chat templates disagree on the default: Qwen3.x thinks unless told
        # otherwise, Gemma-4 stays silent unless told to think. Sending the flag
        # explicitly makes the roster comparable instead of template-dependent.
        body["chat_template_kwargs"] = {"enable_thinking": bool(enable_thinking)}
    return body


def complete(base_url: str, model: str, prompt: str, sampling: dict,
             max_tokens: int, timeout_s: int = 1800, retries: int = 2,
             enable_thinking: Optional[bool] = None,
             system: Optional[str] = None) -> dict:
    """One chat completion. Never raises -- failures come back as `error`."""
    url = base_url.rstrip("/") + "/v1/chat/completions"
    body = build_payload(model, prompt, sampling, max_tokens, enable_thinking,
                         system)
    data = json.dumps(body).encode("utf-8")

    last_err = None
    t0 = time.time()
    for attempt in range(1, retries + 2):
        try:
            req = urllib.request.Request(
                url, data=data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                out = json.loads(resp.read().decode("utf-8"))
            choice = (out.get("choices") or [{}])[0]
            msg = choice.get("message") or {}
            return {
                "content": msg.get("content") or "",
                # vLLM has used both names for the reasoning-parser output:
                # `reasoning_content` in older builds, `reasoning` in newer ones.
                # Reading only one silently discards every chain-of-thought,
                # which scores would not reveal because the answer lives in
                # `content` either way.
                "reasoning_content": (msg.get("reasoning_content")
                                      or msg.get("reasoning") or ""),
                "finish_reason": choice.get("finish_reason"),
                "usage": out.get("usage") or {},
                "latency_s": round(time.time() - t0, 3),
                "attempts": attempt,
                "error": None,
            }
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8", "replace")[:300]
            except Exception:
                pass
            last_err = f"HTTP {e.code}: {detail}"
        except Exception as e:  # timeouts, connection resets, malformed JSON
            last_err = f"{type(e).__name__}: {e}"
        if attempt <= retries:
            time.sleep(min(5 * attempt, 30))

    return {
        "content": "",
        "reasoning_content": "",
        "finish_reason": None,
        "usage": {},
        "latency_s": round(time.time() - t0, 3),
        "attempts": retries + 1,
        "error": last_err,
    }


def run_batch(items: Iterable[dict], fn: Callable[[dict], dict], concurrency: int,
              on_result: Optional[Callable[[dict], None]] = None) -> List[dict]:
    """Map `fn` over `items` in a thread pool, streaming results to `on_result`.

    Results are handed over as they land rather than at the end, so an
    interrupted run keeps everything already generated.
    """
    items = list(items)
    results: List[dict] = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(fn, it): it for it in items}
        for fut in as_completed(futures):
            try:
                res = fut.result()
            except Exception as e:
                item = futures[fut]
                res = {"sample_id": item.get("sample_id"),
                       "error": f"worker crashed: {type(e).__name__}: {e}"}
            results.append(res)
            if on_result:
                on_result(res)
    return results
