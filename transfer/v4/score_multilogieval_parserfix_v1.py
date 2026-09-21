from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from transfer.v4.parser_fixes.multilogieval_explicit_answer_v1 import (
    PARSER_VERSION,
    parse_multilogieval_answer,
)
from transfer.v4.registry import get_adapter
from transfer.v4.score_custom_transfer import best_per_id
from transfer.v4.verify_custom_freeze import verify_custom_freeze


def read_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(
            obj,
            f,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        f.write("\n")


def write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)

    verify_custom_freeze()

    run_obj = read_json(run_dir / "run.json")

    if run_obj.get("status") != "completed":
        raise RuntimeError(
            "Refusing to score incomplete run: "
            f"{run_obj.get('status')!r}"
        )

    identity = run_obj.get("identity") or {}

    if identity.get("limit") is not None:
        raise RuntimeError(
            "Refusing to score a limited/pilot run"
        )

    adapter = get_adapter("multilogieval")

    rows = list(adapter.rows())
    row_by_id = {
        row.row_id: row
        for row in rows
    }

    expected_total = len(rows)

    generations = best_per_id(
        str(run_dir)
    )

    if len(generations) != expected_total:
        raise RuntimeError(
            "Generation count mismatch: "
            f"expected={expected_total} "
            f"actual={len(generations)}"
        )

    prediction_counts = Counter()
    gold_counts = Counter()

    confusion = {
        gold: {
            pred: 0
            for pred in ("yes", "no", "None")
        }
        for gold in ("yes", "no")
    }

    samples = []

    correct = 0
    unparsed = 0
    infrastructure_errors = 0
    truncated = 0

    seen_row_ids = set()

    for item_id, generation in generations.items():
        row_id = generation.get("row_id")

        if row_id not in row_by_id:
            raise RuntimeError(
                f"Unknown row_id: {row_id!r}"
            )

        if row_id in seen_row_ids:
            raise RuntimeError(
                f"Duplicate row_id: {row_id}"
            )

        seen_row_ids.add(row_id)
        row = row_by_id[row_id]

        generation_error = (
            generation.get("error") or ""
        )

        if generation_error:
            infrastructure_errors += 1

        is_truncated = bool(
            generation.get("truncated")
        )

        if is_truncated:
            truncated += 1

        prediction = (
            parse_multilogieval_answer(
                generation.get("completion", "")
            )
        )

        if prediction is None:
            unparsed += 1

        score = float(
            adapter.score(
                prediction,
                row,
            )
        )

        correct += int(score == 1.0)

        gold = row.gold
        pred_key = (
            prediction
            if prediction is not None
            else "None"
        )

        gold_counts[gold] += 1
        prediction_counts[pred_key] += 1
        confusion[gold][pred_key] += 1

        samples.append(
            {
                "id": item_id,
                "row_id": row_id,
                "gold": gold,
                "prediction": prediction,
                "score": score,
                "parser_version": PARSER_VERSION,
                "generation_error":
                    generation_error,
                "truncated":
                    is_truncated,
            }
        )

    if len(seen_row_ids) != expected_total:
        raise RuntimeError(
            "Not every frozen row was scored"
        )

    accuracy = correct / expected_total

    metrics = {
        "status": "complete",
        "benchmark": "multilogieval",
        "parser_version": PARSER_VERSION,
        "protocol_note":
            "Formatting-only extraction repair; "
            "last explicit Answer: field is treated "
            "as the model's final declared answer; "
            "original generations unchanged.",
        "total": expected_total,
        "scored": expected_total,
        "correct": correct,
        "missing": 0,
        "infrastructure_errors":
            infrastructure_errors,
        "unparsed": unparsed,
        "truncated": truncated,
        "accuracy": accuracy,
        "gold_counts": dict(gold_counts),
        "prediction_counts":
            dict(prediction_counts),
        "confusion_matrix": confusion,
    }

    samples_path = (
        run_dir
        / "samples_parserfix_v1.jsonl"
    )

    metrics_path = (
        run_dir
        / "metrics_parserfix_v1.json"
    )

    write_jsonl(samples_path, samples)
    write_json(metrics_path, metrics)

    print(f"parser_version: {PARSER_VERSION}")
    print(f"total: {expected_total}")
    print(f"correct: {correct}")
    print(f"unparsed: {unparsed}")
    print(
        "infrastructure_errors:",
        infrastructure_errors,
    )
    print(f"truncated: {truncated}")
    print(f"accuracy: {accuracy}")
    print(f"metrics: {metrics_path}")
    print(f"samples: {samples_path}")


if __name__ == "__main__":
    main()
