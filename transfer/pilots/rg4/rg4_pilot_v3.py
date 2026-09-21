#!/usr/bin/env python3
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import argparse
import json
import re
import time
import urllib.request

import reasoning_gym

SELECTED = {
    "propositional_logic",
    "knights_knaves",
    "course_schedule",
    "zebra_puzzles",
}

VALID_KK_ROLES = [
    "knight", "knave", "pioneer", "laggard", "saint", "sinner",
    "hero", "villain", "angel", "devil", "altruist", "egoist",
    "sage", "fool",
]


def extract_last_braced_command(text: str, command: str) -> str | None:
    token = "\\" + command + "{"
    starts = []
    pos = 0
    while True:
        idx = text.find(token, pos)
        if idx < 0:
            break
        starts.append(idx)
        pos = idx + 1

    for idx in reversed(starts):
        i = idx + len(token)
        depth = 1
        chars = []
        while i < len(text):
            ch = text[i]
            if ch == "{":
                depth += 1
                chars.append(ch)
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return "".join(chars)
                chars.append(ch)
            else:
                chars.append(ch)
            i += 1
    return None


def final_segment(text: str) -> str:
    if not text:
        return ""

    patt = re.compile(
        r"(?is)(?:^|\n)\s*(?:#{1,6}\s*)?(?:\*\*)?"
        r"(?:final\s+answer|conclusion|answer)"
        r"(?:\*\*)?\s*[:\-]?\s*"
    )
    matches = list(patt.finditer(text))
    if matches:
        return text[matches[-1].end():].strip()
    return text.strip()


def strip_markdown_math(text: str) -> str:
    text = text.strip()
    text = text.replace("```latex", "").replace("```tex", "").replace("```", "")
    text = text.replace("$$", " ").replace("$", " ")
    text = text.replace("**", " ").replace("__", " ").replace("`", " ")
    text = text.replace("\\left", "").replace("\\right", "")
    return " ".join(text.split())


def normalize_logic_notation(text: str) -> str:
    # IMPORTANT: normalize logical TeX commands BEFORE stripping
    # \left / \right. Otherwise:
    #   \leftrightarrow -> rightarrow
    #   \rightarrow     -> arrow
    replacements = [
        (r"\longleftrightarrow", "↔"),
        (r"\leftrightarrow", "↔"),
        (r"\Leftrightarrow", "↔"),
        (r"\iff", "↔"),
        (r"\longrightarrow", "→"),
        (r"\rightarrow", "→"),
        (r"\Rightarrow", "→"),
        (r"\implies", "→"),
        (r"\to", "→"),
        (r"\land", "∧"),
        (r"\wedge", "∧"),
        (r"\lor", "∨"),
        (r"\vee", "∨"),
        (r"\neg", "¬"),
        (r"\lnot", "¬"),
    ]

    for old, new in replacements:
        text = text.replace(old, new)

    text = strip_markdown_math(text)

    text = text.replace(r"\,", " ").replace(r"\;", " ").replace(r"\!", "")
    return " ".join(text.split()).strip(" .,:;")


def unwrap_text_command(text: str) -> str:
    inner = extract_last_braced_command(text, "text")
    return inner.strip() if inner is not None else text


def is_logic_expression(text: str) -> bool:
    return bool(
        text
        and re.fullmatch(r"[A-Z¬∧∨→↔()\s]+", text)
        and re.search(r"[A-Z]", text)
    )


def extract_propositional_logic(completion: str) -> str:
    if not completion:
        return ""

    boxed = extract_last_braced_command(completion, "boxed")
    if boxed is not None:
        candidate = normalize_logic_notation(unwrap_text_command(boxed))
        if is_logic_expression(candidate):
            return candidate

    seg = final_segment(completion)

    math_chunks = re.findall(r"\${1,2}(.+?)\${1,2}", seg, flags=re.S)
    for chunk in reversed(math_chunks):
        boxed = extract_last_braced_command(chunk, "boxed")
        if boxed is not None:
            chunk = boxed
        candidate = normalize_logic_notation(unwrap_text_command(chunk))
        if is_logic_expression(candidate):
            return candidate

    lines = [x for x in seg.splitlines() if x.strip()]
    for line in reversed(lines):
        line = re.sub(
            r"(?i)^\s*(?:#{1,6}\s*)?(?:\*\*)?"
            r"(?:final\s+answer|conclusion|answer)"
            r"(?:\*\*)?\s*[:\-]?\s*",
            "",
            line,
        )
        boxed = extract_last_braced_command(line, "boxed")
        if boxed is not None:
            line = boxed
        candidate = normalize_logic_notation(unwrap_text_command(line))
        if is_logic_expression(candidate):
            return candidate

    candidate = normalize_logic_notation(seg)
    if is_logic_expression(candidate):
        return candidate

    return candidate


def extract_course_schedule(completion: str) -> str:
    seg = final_segment(completion)
    found = list(re.finditer(r"(?i)\b(true|false)\b", seg))
    if not found:
        found = list(re.finditer(r"(?i)\b(true|false)\b", completion))
    return found[-1].group(1).capitalize() if found else ""


def extract_knights_knaves(completion: str, entry: dict) -> str:
    names = list(entry.get("metadata", {}).get("names") or [])
    if not names:
        return ""

    for source in (final_segment(completion), completion):
        assignments = []
        for name in names:
            patt = re.compile(
                rf"(?i)\b{re.escape(name)}\b\s+is\s+(?:a|an)?\s*"
                rf"({'|'.join(map(re.escape, VALID_KK_ROLES))})\b"
            )
            matches = list(patt.finditer(source))
            if matches:
                assignments.append(
                    f"{name} is a {matches[-1].group(1).lower()}"
                )
        if len(assignments) == len(names):
            return ", ".join(assignments)
    return ""


def extract_zebra(completion: str) -> str:
    if not completion:
        return ""

    boxed = extract_last_braced_command(completion, "boxed")
    if boxed is not None:
        boxed = unwrap_text_command(boxed)
        words = re.findall(r"[A-Za-z][A-Za-z'-]*", boxed)
        if words:
            return words[-1].lower()

    seg = unwrap_text_command(strip_markdown_math(final_segment(completion)))
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", seg)
    return words[-1].lower() if words else ""


def extract_answer(task: str, completion: str, entry: dict) -> str:
    if task == "propositional_logic":
        return extract_propositional_logic(completion)
    if task == "course_schedule":
        return extract_course_schedule(completion)
    if task == "knights_knaves":
        return extract_knights_knaves(completion, entry)
    if task == "zebra_puzzles":
        return extract_zebra(completion)
    raise ValueError(f"Unsupported task: {task}")


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(x)
        for x in path.read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]


def append_jsonl(path: Path, obj: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()


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

    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
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

            choice = obj["choices"][0]
            msg = choice["message"]
            return {
                "completion": msg.get("content") or "",
                "reasoning": msg.get("reasoning_content") or "",
                "finish_reason": choice.get("finish_reason") or "",
                "usage": obj.get("usage") or {},
                "latency_s": time.time() - started,
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
    result = {
        "rows": len(rows),
        "completed": len(generations),
        "parser_version": "rg4_parser_v3",
        "by_task": {},
        "by_difficulty": {},
        "by_task_difficulty": {},
    }

    specs = [
        ("by_task", lambda r: r["task"]),
        ("by_difficulty", lambda r: r["difficulty"]),
        ("by_task_difficulty", lambda r: f"{r['task']}::{r['difficulty']}"),
    ]

    for out_key, key_fn in specs:
        groups = defaultdict(list)
        for row in rows:
            g = generations.get(row["id"])
            if g is not None:
                groups[key_fn(row)].append(g)

        for key, xs in sorted(groups.items()):
            scores = [
                x["adapted_native_score"]
                for x in xs
                if x.get("adapted_native_score") is not None
            ]
            lats = [
                x["latency_s"]
                for x in xs
                if x.get("latency_s") is not None
            ]
            result[out_key][key] = {
                "n": len(xs),
                "adapted_native_mean": (
                    sum(scores) / len(scores) if scores else None
                ),
                "full_credit_rate": (
                    sum(s >= 0.999999 for s in scores) / len(scores)
                    if scores else None
                ),
                "truncated": sum(bool(x.get("truncated")) for x in xs),
                "errors": sum(bool(x.get("error")) for x in xs),
                "mean_latency_s": (
                    sum(lats) / len(lats) if lats else None
                ),
            }

    return result


def command_rescore(args: argparse.Namespace) -> None:
    data = {x["id"]: x for x in read_jsonl(Path(args.data))}
    gens = {x["id"]: x for x in read_jsonl(Path(args.generations))}

    records = []
    groups = defaultdict(list)

    for item_id, row in data.items():
        if row["task"] not in SELECTED:
            continue
        if item_id not in gens:
            raise RuntimeError(f"Missing generation: {item_id}")

        g = gens[item_id]
        answer = extract_answer(
            row["task"],
            g.get("completion") or "",
            row["entry"],
        )
        scorer = reasoning_gym.get_score_answer_fn(row["task"])
        score = float(scorer(answer=answer, entry=row["entry"]))

        rec = {
            "id": item_id,
            "task": row["task"],
            "difficulty": row["difficulty"],
            "parser_version": "rg4_parser_v3",
            "old_adapted_answer": g.get("adapted_answer"),
            "old_adapted_native_score": g.get("adapted_native_score"),
            "adapted_answer_v2": answer,
            "adapted_native_score_v2": score,
            "full_credit_v2": score >= 0.999999,
            "finish_reason": g.get("finish_reason"),
            "truncated": bool(g.get("truncated")),
        }
        records.append(rec)
        groups[row["task"]].append(rec)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")

    print("=" * 100)
    print("RG4 PARSER-V2 RETROSPECTIVE RESCORE")
    print("=" * 100)
    for task in sorted(groups):
        xs = groups[task]
        mean = sum(x["adapted_native_score_v2"] for x in xs) / len(xs)
        full = sum(x["full_credit_v2"] for x in xs)
        trunc = sum(x["truncated"] for x in xs)
        print(
            f"{task}: n={len(xs)} mean={mean:.6f} "
            f"full_credit={full}/{len(xs)} trunc={trunc}"
        )
        for x in sorted(
            xs,
            key=lambda z: {"easy": 0, "medium": 1, "hard": 2}[z["difficulty"]],
        ):
            print(
                f"  {x['difficulty']:6s} "
                f"old={x['old_adapted_native_score']} "
                f"new={x['adapted_native_score_v2']} "
                f"answer={x['adapted_answer_v2']!r} "
                f"finish={x['finish_reason']}"
            )
    print("output:", out)


def command_run(args: argparse.Namespace) -> None:
    rows = read_jsonl(Path(args.data_jsonl))
    if len(rows) != 120:
        raise RuntimeError(f"Expected 120 rows, got {len(rows)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gen_path = out_dir / "generations.jsonl"

    existing = {}
    if gen_path.exists():
        existing = {x["id"]: x for x in read_jsonl(gen_path)}

    print(f"rows={len(rows)} existing={len(existing)}", flush=True)

    for i, row in enumerate(rows, 1):
        item_id = row["id"]
        if item_id in existing:
            print(f"[{i}/120] SKIP {item_id}", flush=True)
            continue

        print(f"[{i}/120] {item_id}", flush=True)
        result = request_one(args.base_url, row["entry"]["question"])

        raw_score = None
        adapted_score = None
        adapted_answer = ""
        scorer_error = ""

        if not result["error"]:
            scorer = reasoning_gym.get_score_answer_fn(row["task"])
            try:
                raw_score = float(
                    scorer(answer=result["completion"], entry=row["entry"])
                )
                adapted_answer = extract_answer(
                    row["task"], result["completion"], row["entry"]
                )
                adapted_score = float(
                    scorer(answer=adapted_answer, entry=row["entry"])
                )
            except Exception as exc:
                scorer_error = f"{type(exc).__name__}: {exc}"

        record = {
            "id": item_id,
            "benchmark": row["benchmark"],
            "task": row["task"],
            "difficulty": row["difficulty"],
            "generator_index": row["generator_index"],
            "generator_seed": row["generator_seed"],
            "parser_version": "rg4_parser_v3",
            "completion": result["completion"],
            "reasoning": result["reasoning"],
            "adapted_answer": adapted_answer,
            "raw_native_score": raw_score,
            "adapted_native_score": adapted_score,
            "full_credit": (
                adapted_score is not None and adapted_score >= 0.999999
            ),
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

    metrics = summarize(rows, existing)
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(metrics, indent=2, ensure_ascii=False, sort_keys=True),
        flush=True,
    )


def command_selftest(_: argparse.Namespace) -> None:
    prop_tests = [
        (
            "The conclusion follows.\n\n**Conclusion:** P ∧ Q",
            "P ∧ Q",
        ),
        (
            "**Final Answer:**\n$$\n\\boxed{S \\land P \\land T \\land Q}\n$$",
            "S ∧ P ∧ T ∧ Q",
        ),
        (
            "### Final Answer:\n¬Q",
            "¬Q",
        ),
    ]
    for text, expected in prop_tests:
        got = extract_propositional_logic(text)
        if got != expected:
            raise RuntimeError(
                f"prop parser self-test failed: got={got!r} expected={expected!r}"
            )

    if extract_course_schedule("Reasoning...\nFinal Answer: False") != "False":
        raise RuntimeError("course parser self-test failed")

    if extract_zebra("Final Answer: \\boxed{\\text{Alice}}") != "alice":
        raise RuntimeError("zebra parser self-test failed")

    kk_entry = {"metadata": {"names": ["Ada", "Bob"]}}
    kk_text = "Final Answer: Ada is a hero, and Bob is a villain."
    kk = extract_knights_knaves(kk_text, kk_entry)
    if kk != "Ada is a hero, Bob is a villain":
        raise RuntimeError(f"kk parser self-test failed: {kk!r}")

    print("RG4 PARSER SELF-TEST: PASS")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("selftest")
    p.set_defaults(func=command_selftest)

    p = sub.add_parser("rescore")
    p.add_argument(
        "--data",
        default="/arf/scratch/futan/ArgGYM/data/transfer/pilots/rg8/rg8_qual24_v4.jsonl",
    )
    p.add_argument(
        "--generations",
        default="/arf/scratch/futan/ArgGYM/outputs/transfer/pilots/rg8_qual24_v4/generations.jsonl",
    )
    p.add_argument(
        "--out",
        default="/arf/scratch/futan/ArgGYM/outputs/transfer/pilots/rg8_qual24_v4/parser_v2_rescore.jsonl",
    )
    p.set_defaults(func=command_rescore)

    p = sub.add_parser("run")
    p.add_argument("--data-jsonl", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--base-url", required=True)
    p.set_defaults(func=command_run)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
