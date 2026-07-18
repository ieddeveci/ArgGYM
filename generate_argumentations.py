
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request


_TIMEOUT = 300
_RETRIES = 2

_API_CACHE: dict = {}


def _detect_api(host):
    """Return 'ollama' or 'openai' for the server at host.

    Override with LLM_API=ollama|openai; otherwise probe once per host
    (Ollama answers /api/version; OpenAI-compatible servers like vLLM don't).
    """
    if host not in _API_CACHE:
        api = os.environ.get("LLM_API", "").strip().lower()
        if api not in ("ollama", "openai"):
            try:
                with urllib.request.urlopen(host.rstrip("/") + "/api/version", timeout=5):
                    api = "ollama"
            except Exception:
                api = "openai"
        _API_CACHE[host] = api
    return _API_CACHE[host]


def _chat(model, system, user, host, temperature, api_key=None, schema=None):
    api = _detect_api(host)
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
    if api == "openai":
        url = host.rstrip("/") + "/v1/chat/completions"
        body = {"model": model, "messages": messages, "stream": False,
                "temperature": float(temperature)}
        schema_key = "response_format"
        if schema is not None:
            body[schema_key] = {"type": "json_schema",
                                "json_schema": {"name": "output", "schema": schema}}
    else:
        url = host.rstrip("/") + "/api/chat"
        body = {"model": model, "messages": messages, "stream": False,
                "options": {"temperature": float(temperature)}}
        schema_key = "format"
        if schema is not None:
            body[schema_key] = schema

    last = None
    for attempt in range(_RETRIES + 1):
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=data,
                                     headers={"Content-Type": "application/json"})
        if api_key:
            req.add_header("Authorization", f"Bearer {api_key}")
        try:
            with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
                out = json.loads(resp.read().decode("utf-8"))
            if api == "openai":
                choices = out.get("choices") or [{}]
                msg = (choices[0].get("message") or {}).get("content") or ""
            else:
                msg = (out.get("message") or {}).get("content", "")
            if msg:
                return msg
            last = RuntimeError(f"empty response: {str(out)[:200]}")
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8", "replace")[:300]
            except Exception:
                pass
            if e.code == 400 and schema_key in body:
                body.pop(schema_key)
                last = RuntimeError(f"HTTP 400 with schema format ({detail}); "
                                    f"retrying without structured output")
                continue
            last = RuntimeError(f"HTTP {e.code}: {detail}")
        except Exception as e:
            last = e
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"{api} chat failed after {_RETRIES + 1} attempts: {last}")


_ollama_chat = _chat

def _parse_json(txt):
    if not txt:
        return None
    t = txt.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.S)
    try:
        return json.loads(t)
    except Exception:
        pass
    start = t.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(t)):
            if t[i] == "{":
                depth += 1
            elif t[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(t[start:i + 1])
                    except Exception:
                        break
        start = t.find("{", start + 1)
    return None


def _san(s):
    return re.sub(r"\s+", " ", (s or "")).strip()

_CLAUSE_CONJ = re.compile(r",\s+(and|but|or|so|yet)\b|;\s", re.I)
_MARKERS = re.compile(r"\b(if|then|because|unless|whenever|provided that|however|"
                      r"although|whereas|therefore|hence|thus)\b", re.I)


def _atomic_problems(text):
    t = (text or "").strip()
    probs = []
    if not t:
        return ["empty"]
    if len(re.findall(r"[.!?](?:\s|$)", t)) > 1:
        probs.append("multiple sentences")
    if _CLAUSE_CONJ.search(t):
        probs.append("clause conjunction")
    if _MARKERS.search(t):
        probs.append("conditional/causal/concessive marker")
    if len(t.split()) > 24:
        probs.append("too long")
    if "?" in t:
        probs.append("question")
    return probs

def read_claims(path):
    if str(path).lower().endswith(".json"):
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            data = data.get("claims", [])
        return [str(c).strip() for c in data if str(c).strip()]
    with open(path, encoding="utf-8") as fh:
        return [ln.strip() for ln in fh
                if ln.strip() and not ln.strip().startswith("#")]
