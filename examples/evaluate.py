"""Score a model against a frozen ArgGYM taskset.

The seam this demonstrates is the point of it: ArgGYM never calls a model, and
this file never reaches into ArgGYM's internals. It reads a JSONL, sends each
`question` to an OpenAI-compatible endpoint however it likes, and hands the
answer body back to `arggym.score_row`.

The question already asks for the answer between `<answer>` and `</answer>`, so
this file reads that region back with `arggym.extract_answer` and adds nothing.
Another convention is a re-render, not an edit here: freeze with a different
`AnswerTemplate` and read its region instead. Scoring is unaffected either way.

    uv run arggym freeze -c tasksets/standard.yaml -o data/taskset.jsonl
    python examples/evaluate.py data/taskset.jsonl \
        --base-url http://localhost:8000/v1 --model my-model

Standard library only, so it runs anywhere the package does.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import arggym

# The question states the submission contract; this only tells the model it may
# think first, which the question does not say either way.
SYSTEM = "Work the problem out, then answer in the format the question asks for."


def complete(base_url: str, model: str, key: str, question: str,
             max_tokens: int, timeout: int) -> dict:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": question}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions", data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            payload = json.load(r)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        # An API failure and a wrong answer are different events. Returning
        # score 0 here would report infrastructure trouble as a reasoning
        # result, which is the mistake this field makes routinely.
        return {"error": f"{type(e).__name__}: {e}"}
    choice = payload["choices"][0]
    return {"text": choice["message"].get("content") or "",
            "finish_reason": choice.get("finish_reason")}


def rows(path: str):
    with open(path) as f:
        first = json.loads(f.readline())
        manifest = first.get("__manifest__")
        if manifest is None:
            yield first
        for line in f:
            yield json.loads(line)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("taskset")
    p.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL",
                                                        "http://localhost:8000/v1"))
    p.add_argument("--model", required=True)
    p.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "none"))
    p.add_argument("--out", default="scored.jsonl")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--limit", type=int, default=None, help="First N rows only.")
    a = p.parse_args()

    items = list(rows(a.taskset))[:a.limit]
    print(f"{len(items)} items from {a.taskset}", file=sys.stderr)

    def one(row: dict) -> dict:
        gen = complete(a.base_url, a.model, a.api_key, row["question"],
                       a.max_tokens, a.timeout)
        out = {"id": row["id"], "task": row["task"],
               "level": row["metadata"]["level"],
               "ordering": row["metadata"]["ordering"],
               "error": gen.get("error"),
               "truncated": gen.get("finish_reason") == "length",
               "completion": gen.get("text", "")}
        if gen.get("error"):
            return out
        result = arggym.score_row(arggym.extract_answer(gen["text"]), row)
        out.update(score=result.score, success=result.success, reason=result.reason,
                   diagnostics=result.diagnostics)
        return out

    with ThreadPoolExecutor(max_workers=a.concurrency) as pool:
        scored = list(pool.map(one, items))

    with open(a.out, "w") as f:
        for s in scored:
            f.write(json.dumps(s, default=str) + "\n")

    report(scored)
    print(f"\nwrote {len(scored)} -> {a.out}", file=sys.stderr)
    return 0


def report(scored: list) -> None:
    """Per task, never as one mean.

    The twelve metrics are of four kinds and their chance floors differ by an
    order of magnitude, so an unweighted average over them moves mostly with
    which tasks are in the basket. `docs/dataset-card.md` says why.
    """
    by_task = defaultdict(list)
    for s in scored:
        by_task[s["task"]].append(s)

    print(f"\n{'task':26s} {'n':>4s} {'mean':>7s} {'success':>8s} "
          f"{'errors':>7s} {'trunc':>6s}")
    for task in sorted(by_task):
        got = by_task[task]
        ok = [g for g in got if not g["error"]]
        mean = sum(g["score"] for g in ok) / len(ok) if ok else float("nan")
        succ = sum(1 for g in ok if g["success"]) / len(ok) if ok else float("nan")
        print(f"{task:26s} {len(got):4d} {mean:7.3f} {succ:8.3f} "
              f"{sum(1 for g in got if g['error']):7d} "
              f"{sum(1 for g in got if g['truncated']):6d}")
    print("\nNo overall mean: the per-task metrics are not the same quantity.")


if __name__ == "__main__":
    raise SystemExit(main())
