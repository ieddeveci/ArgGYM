#!/usr/bin/env python3

from collections import defaultdict
from pathlib import Path
import importlib.util
import json
import sys

import reasoning_gym


ROOT = Path("/arf/scratch/futan/ArgGYM")

DATA = (
    ROOT
    / "data/transfer/pilots/zebra_puzzles/"
      "zebra_chars20_v1.jsonl"
)

OUT = (
    ROOT
    / "outputs/transfer/pilots/"
      "zebra_chars20_v1"
)

PARSER = (
    ROOT
    / "transfer/pilots/rg4/"
      "rg4_pilot_v4.py"
)


def load_rg4():
    spec = importlib.util.spec_from_file_location(
        "rg4v4",
        PARSER,
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    if len(sys.argv) != 2:
        raise SystemExit(
            "usage: run_zebra_chars20_v1.py BASE_URL"
        )

    base_url = sys.argv[1]
    m = load_rg4()

    rows = m.read_jsonl(DATA)

    if len(rows) != 20:
        raise RuntimeError(
            f"expected 20 rows, got {len(rows)}"
        )

    if any(
        r["task"] != "zebra_puzzles"
        for r in rows
    ):
        raise RuntimeError(
            "non-zebra_puzzles row present"
        )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    gen_path = OUT / "generations.jsonl"

    existing = {}

    if gen_path.exists():
        existing = {
            x["id"]: x
            for x in m.read_jsonl(gen_path)
        }

    print(
        f"rows=20 existing={len(existing)}",
        flush=True,
    )

    for i, row in enumerate(rows, 1):
        item_id = row["id"]

        if item_id in existing:
            print(
                f"[{i}/20] SKIP {item_id}",
                flush=True,
            )
            continue

        print(
            f"[{i}/20] {item_id}",
            flush=True,
        )

        result = m.request_one(
            base_url,
            row["entry"]["question"],
        )

        raw_score = None
        adapted_score = None
        adapted_answer = ""
        scorer_error = ""

        if not result["error"]:
            scorer = reasoning_gym.get_score_answer_fn(
                "zebra_puzzles"
            )

            try:
                raw_score = float(
                    scorer(
                        answer=result["completion"],
                        entry=row["entry"],
                    )
                )

                adapted_answer = m.extract_answer(
                    "zebra_puzzles",
                    result["completion"],
                    row["entry"],
                )

                adapted_score = float(
                    scorer(
                        answer=adapted_answer,
                        entry=row["entry"],
                    )
                )

            except Exception as exc:
                scorer_error = (
                    f"{type(exc).__name__}: {exc}"
                )

        record = {
            "id": item_id,
            "benchmark": row["benchmark"],
            "task": row["task"],
            "difficulty": row["difficulty"],
            "generator_index": row["generator_index"],
            "generator_seed": row["generator_seed"],
            "parser_version": "rg4_parser_v4",
            "completion": result["completion"],
            "reasoning": result["reasoning"],
            "adapted_answer": adapted_answer,
            "raw_native_score": raw_score,
            "adapted_native_score": adapted_score,
            "full_credit": (
                adapted_score is not None
                and adapted_score >= 0.999999
            ),
            "finish_reason": result["finish_reason"],
            "truncated": (
                result["finish_reason"] == "length"
            ),
            "usage": result["usage"],
            "latency_s": result["latency_s"],
            "attempts": result["attempts"],
            "error": result["error"],
            "scorer_error": scorer_error,
        }

        m.append_jsonl(
            gen_path,
            record,
        )

        existing[item_id] = record

    groups = defaultdict(list)

    for g in existing.values():
        groups[g["difficulty"]].append(g)

    metrics = {
        "benchmark": "zebra_chars20_v1",
        "rows": 20,
        "completed": len(existing),
        "parser_version": "rg4_parser_v4",
        "by_level": {},
    }

    for level, xs in sorted(groups.items()):
        scores = [
            x["adapted_native_score"]
            for x in xs
            if x["adapted_native_score"] is not None
        ]

        toks = [
            (x.get("usage") or {}).get(
                "completion_tokens"
            )
            for x in xs
            if (x.get("usage") or {}).get(
                "completion_tokens"
            ) is not None
        ]

        lats = [
            x["latency_s"]
            for x in xs
            if x.get("latency_s") is not None
        ]

        metrics["by_level"][level] = {
            "n": len(xs),
            "correct": sum(
                s >= 0.999999
                for s in scores
            ),
            "accuracy": (
                sum(
                    s >= 0.999999
                    for s in scores
                ) / len(scores)
                if scores
                else None
            ),
            "truncated": sum(
                bool(x["truncated"])
                for x in xs
            ),
            "mean_completion_tokens": (
                sum(toks) / len(toks)
                if toks
                else None
            ),
            "max_completion_tokens": (
                max(toks)
                if toks
                else None
            ),
            "mean_latency_s": (
                sum(lats) / len(lats)
                if lats
                else None
            ),
        }

    metrics_path = OUT / "metrics.json"

    metrics_path.write_text(
        json.dumps(
            metrics,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            metrics,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
