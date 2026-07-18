
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request


_TIMEOUT = 300    
_RETRIES = 2          


def _ollama_chat(model, system, user, host, temperature, api_key=None, schema=None):
    url = host.rstrip("/") + "/api/chat"
    body = {
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "stream": False,
        "options": {"temperature": float(temperature)},
    }
    if schema is not None:
        body["format"] = schema        

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
            if e.code == 400 and "format" in body:
                body.pop("format")
                last = RuntimeError(f"HTTP 400 with schema format ({detail}); "
                                    f"retrying without structured output")
                continue
            last = RuntimeError(f"HTTP {e.code}: {detail}")
        except Exception as e:                   
            last = e
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"ollama chat failed after {_RETRIES + 1} attempts: {last}")

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
