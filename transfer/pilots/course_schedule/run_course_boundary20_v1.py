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
    / "data/transfer/pilots/course_schedule/"
      "course_boundary20_v1.jsonl"
)

OUT = (
    ROOT
    / "outputs/transfer/pilots/"
      "course_boundary20_v1"
)

PARSER = (
    ROOT
    / "transfer/pilots/rg4/"
      "rg4_pilot_v4.py"
)


def load_parser():
    spec = importlib.util.spec_from_file_location(
        "rg4v4",
        PARSER,
    )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(module)

    return module


def main():
    if len(sys.argv) != 2:
        raise SystemExit(
            "usage: run_course_boundary20_v1.py BASE_URL"
        )

    base_url = sys.argv[1]

    m = load_parser()

    rows = m.read_jsonl(DATA)

    if len(rows) != 20:
        raise RuntimeError(
            f"expected 20 rows, got {len(rows)}"
        )

    if any(
        row["task"] != "course_schedule"
        for row in rows
    ):
        raise RuntimeError(
            "non-course_schedule row present"
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
            scorer = (
                reasoning_gym
                .get_score_answer_fn(
                    "course_schedule"
                )
            )

            try:
                raw_score = float(
                    scorer(
                        answer=result["completion"],
                        entry=row["entry"],
                    )
                )

                adapted_answer = m.extract_answer(
                    "course_schedule",
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

    # ----------------------------------------------------------
    # Final metrics: level × gold label
    # ----------------------------------------------------------

    data_by_id = {
        row["id"]: row
        for row in rows
    }

    groups = defaultdict(list)

    for item_id, generation in existing.items():
        row = data_by_id[item_id]

        gold = str(
            row["entry"]["answer"]
        )

        groups[
            (row["difficulty"], gold)
        ].append(generation)

    metrics = {
        "benchmark": "course_boundary20_v1",
        "rows": 20,
        "completed": len(existing),
        "parser_version": "rg4_parser_v4",
        "by_level_gold": {},
    }

    for key in sorted(groups):
        level, gold = key
        xs = groups[key]

        scores = [
            x["adapted_native_score"]
            for x in xs
            if x["adapted_native_score"]
            is not None
        ]

        tokens = [
            (x.get("usage") or {}).get(
                "completion_tokens"
            )
            for x in xs
            if (x.get("usage") or {}).get(
                "completion_tokens"
            )
            is not None
        ]

        latencies = [
            x["latency_s"]
            for x in xs
            if x.get("latency_s")
            is not None
        ]

        metrics["by_level_gold"][
            f"{level}::{gold}"
        ] = {
            "n": len(xs),
            "correct": sum(
                score >= 0.999999
                for score in scores
            ),
            "accuracy": (
                sum(
                    score >= 0.999999
                    for score in scores
                )
                / len(scores)
                if scores
                else None
            ),
            "truncated": sum(
                bool(x["truncated"])
                for x in xs
            ),
            "mean_completion_tokens": (
                sum(tokens) / len(tokens)
                if tokens
                else None
            ),
            "max_completion_tokens": (
                max(tokens)
                if tokens
                else None
            ),
            "mean_latency_s": (
                sum(latencies)
                / len(latencies)
                if latencies
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
