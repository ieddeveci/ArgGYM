#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import argparse
import json
import re
import time
import urllib.error
import urllib.request

import reasoning_gym

VALID_RELATIONSHIPS = [
    "mother-in-law", "father-in-law", "grandmother", "grandfather",
    "daughter", "brother", "husband", "mother", "father", "sister",
    "wife", "uncle", "niece", "nephew", "aunt", "son",
]
VALID_KK_ROLES = [
    "knight", "knave", "pioneer", "laggard", "saint", "sinner",
    "hero", "villain", "angel", "devil", "altruist", "egoist", "sage", "fool",
]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def append_jsonl(path: Path, obj: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()


def final_segment(text: str) -> str:
    if not text:
        return ""
    matches = list(re.finditer(r"(?i)\bfinal\s+answer\b\s*[:\-]?", text))
    if matches:
        return text[matches[-1].end():].strip()
    matches = list(re.finditer(r"(?i)(?:^|\n)\s*answer\s*[:\-]", text))
    if matches:
        return text[matches[-1].end():].strip()
    return text.strip()


def unbox(text: str) -> str:
    # Common Qwen final forms: \boxed{3}, \boxed{\text{Alice}}
    boxes = re.findall(r"\\boxed\{([^{}]*(?:\\text\{[^{}]*\}[^{}]*)?)\}", text)
    if boxes:
        text = boxes[-1]
    texts = re.findall(r"\\text\{([^{}]+)\}", text)
    if texts:
        text = texts[-1]
    text = text.replace("$$", " ").replace("$", " ")
    text = text.replace("**", " ").replace("`", " ")
    return text.strip()


def extract_answer(task: str, completion: str, entry: dict) -> str:
    seg = unbox(final_segment(completion))

    if task == "arc_1d":
        source = seg if seg else completion
        # Prefer the last line/list that consists only of integer grid cells.
        candidates = re.findall(
            r"(?m)^\s*[\[\(]?\s*(?:\d+\s*[,\s]\s*){2,}\d+\s*[\]\)]?\s*$",
            source,
        )
        if not candidates:
            return seg.strip()
        nums = re.findall(r"\d+", candidates[-1])
        return " ".join(nums)

    if task == "family_relationships":
        low = seg.lower()
        found = []
        for rel in VALID_RELATIONSHIPS:
            for m in re.finditer(rf"(?<![\w-]){re.escape(rel)}(?![\w-])", low):
                found.append((m.start(), rel))
        return max(found)[1] if found else seg.strip().lower()

    if task == "course_schedule":
        found = list(re.finditer(r"(?i)\b(true|false)\b", seg))
        if not found:
            found = list(re.finditer(r"(?i)\b(true|false)\b", completion))
        return found[-1].group(1).capitalize() if found else seg.strip()

    if task == "knights_knaves":
        names = list(entry.get("metadata", {}).get("names") or [])
        assignments = []
        for name in names:
            patt = re.compile(
                rf"(?i)\b{re.escape(name)}\b\s+is\s+(?:a|an)?\s*"
                rf"({'|'.join(map(re.escape, VALID_KK_ROLES))})\b"
            )
            matches = list(patt.finditer(completion))
            if matches:
                assignments.append(f"{name} is a {matches[-1].group(1).lower()}")
        return ", ".join(assignments) if assignments else seg.strip()

    if task == "propositional_logic":
        source = seg if seg else completion
        # Pull a final standalone symbolic expression. Do not consult the gold answer.
        lines = [x.strip() for x in source.splitlines() if x.strip()]
        allowed = re.compile(r"^[A-Z¬∧∨→↔()\s]+$")
        candidates = [x.strip(" .") for x in lines if allowed.fullmatch(x.strip(" ."))]
        if candidates:
            return candidates[-1]
        return seg.strip(" .")

    if task == "self_reference":
        source = seg if seg else completion
        boxed = re.findall(r"\\boxed\{(\d+)\}", completion)
        if boxed:
            return boxed[-1]
        ints = re.findall(r"(?<!\d)(\d+)(?!\d)", source)
        return ints[-1] if ints else seg.strip()

    if task == "syllogism":
        found = list(re.finditer(r"(?i)\b(yes|no)\b", seg))
        if not found:
            found = list(re.finditer(r"(?i)\b(yes|no)\b", completion))
        return found[-1].group(1).capitalize() if found else seg.strip()

    if task == "zebra_puzzles":
        source = seg if seg else completion
        source = unbox(source)
        # Final requested value is a name/entity. Take last alphabetic token
        # from the final-answer segment, independent of the gold answer.
        words = re.findall(r"[A-Za-z][A-Za-z'-]*", source)
        return words[-1].lower() if words else source.strip().lower()

    return seg.strip()


def request_one(base_url: str, question: str, timeout_s: int = 5400) -> dict:
    payload = {
        "model": "Qwen/Qwen3-14B",
        "messages": [{"role": "user", "content": question}],
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "max_tokens": 16384,
        "seed": 1234,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer none",
        },
        method="POST",
    )

    last_exc = None
    for attempt in range(1, 4):
        started = time.time()
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                obj = json.loads(resp.read().decode("utf-8"))
            latency = time.time() - started
            choice = obj["choices"][0]
            msg = choice["message"]
            return {
                "completion": msg.get("content") or "",
                "reasoning": msg.get("reasoning_content") or "",
                "finish_reason": choice.get("finish_reason") or "",
                "usage": obj.get("usage") or {},
                "latency_s": latency,
                "attempts": attempt,
                "error": "",
            }
        except Exception as exc:
            last_exc = exc
            if attempt < 3:
                time.sleep(5)
    return {
        "completion": "",
        "reasoning": "",
        "finish_reason": "",
        "usage": {},
        "latency_s": None,
        "attempts": 3,
        "error": f"{type(last_exc).__name__}: {last_exc}",
    }


def summarize(rows: list[dict], generations: dict[str, dict]) -> dict:
    out = {
        "rows": len(rows),
        "completed": len(generations),
        "by_task": {},
        "by_difficulty": {},
    }
    for group_name, key in (("by_task", "task"), ("by_difficulty", "difficulty")):
        groups = defaultdict(list)
        for row in rows:
            g = generations.get(row["id"])
            if g is not None:
                groups[row[key]].append(g)
        for k, xs in sorted(groups.items()):
            raw = [x["raw_native_score"] for x in xs if x.get("raw_native_score") is not None]
            adapted = [x["adapted_native_score"] for x in xs if x.get("adapted_native_score") is not None]
            out[group_name][k] = {
                "n": len(xs),
                "raw_native_mean": sum(raw) / len(raw) if raw else None,
                "adapted_native_mean": sum(adapted) / len(adapted) if adapted else None,
                "truncated": sum(bool(x.get("truncated")) for x in xs),
                "errors": sum(bool(x.get("error")) for x in xs),
                "mean_latency_s": (
                    sum(x["latency_s"] for x in xs if x.get("latency_s") is not None)
                    / max(1, sum(x.get("latency_s") is not None for x in xs))
                ),
            }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-jsonl", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--base-url", required=True)
    args = ap.parse_args()

    data_path = Path(args.data_jsonl)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(data_path)
    gen_path = out_dir / "generations.jsonl"

    existing = {}
    if gen_path.exists():
        for x in read_jsonl(gen_path):
            existing[x["id"]] = x

    print(f"rows={len(rows)} existing={len(existing)}", flush=True)

    for i, row in enumerate(rows, 1):
        item_id = row["id"]
        if item_id in existing:
            print(f"[{i}/{len(rows)}] SKIP {item_id}", flush=True)
            continue

        task = row["task"]
        entry = row["entry"]
        print(f"[{i}/{len(rows)}] {item_id}", flush=True)

        result = request_one(args.base_url, entry["question"])
        completion = result["completion"]

        raw_score = None
        adapted_score = None
        adapted_answer = ""
        scorer_error = ""

        if not result["error"]:
            scorer = reasoning_gym.get_score_answer_fn(task)
            try:
                raw_score = float(scorer(answer=completion, entry=entry))
                adapted_answer = extract_answer(task, completion, entry)
                adapted_score = float(scorer(answer=adapted_answer, entry=entry))
            except Exception as exc:
                scorer_error = f"{type(exc).__name__}: {exc}"

        record = {
            "id": item_id,
            "benchmark": row["benchmark"],
            "task": task,
            "difficulty": row["difficulty"],
            "generator_index": row["generator_index"],
            "generator_seed": row["generator_seed"],
            "completion": completion,
            "reasoning": result["reasoning"],
            "adapted_answer": adapted_answer,
            "raw_native_score": raw_score,
            "adapted_native_score": adapted_score,
            "finish_reason": result["finish_reason"],
            "truncated": result["finish_reason"] == "length",
            "usage": result["usage"],
            "latency_s": result["latency_s"],
            "attempts": result["attempts"],
            "error": result["error"],
            "scorer_error": scorer_error,
        }
        append_jsonl(gen_path, record)
        existing[item_id] = record

    summary = summarize(rows, existing)
    (out_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # Human-readable qualitative report.
    by_id = {r["id"]: r for r in rows}
    with (out_dir / "qualitative_report.txt").open("w", encoding="utf-8") as f:
        for item_id in [r["id"] for r in rows]:
            row = by_id[item_id]
            g = existing.get(item_id, {})
            f.write("=" * 100 + "\n")
            f.write(f"{item_id}\n")
            f.write(
                f"task={row['task']} difficulty={row['difficulty']} "
                f"raw={g.get('raw_native_score')} adapted={g.get('adapted_native_score')} "
                f"finish={g.get('finish_reason')} latency={g.get('latency_s')}\n"
            )
            f.write("metadata=" + json.dumps(row["entry"]["metadata"], ensure_ascii=False, sort_keys=True) + "\n\n")
            f.write("QUESTION\n" + row["entry"]["question"] + "\n\n")
            f.write("GOLD\n" + str(row["entry"]["answer"]) + "\n\n")
            f.write("ADAPTED ANSWER\n" + str(g.get("adapted_answer", "")) + "\n\n")
            f.write("MODEL COMPLETION\n" + str(g.get("completion", "")) + "\n\n")

    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
