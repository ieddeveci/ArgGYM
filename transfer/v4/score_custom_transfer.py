from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from evals.artifacts import (
    METRICS,
    PROMPTS,
    RUN,
    SAMPLES,
    best_per_id,
    read_jsonl,
    write_json,
    write_jsonl,
)
from transfer.v4.registry import ADAPTERS, get_adapter
from transfer.v4.verify_custom_freeze import (
    EXPECTED_MANIFEST_SHA256,
    verify_custom_freeze,
)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--benchmark",
        required=True,
        choices=sorted(ADAPTERS),
    )

    parser.add_argument(
        "--run-dir",
        required=True,
    )

    args = parser.parse_args()

    run_dir = Path(args.run_dir)

    # Revalidate the complete frozen scientific input contract.
    manifest = verify_custom_freeze()

    expected_count = int(
        manifest["expected_counts"][args.benchmark]
    )

    run_path = run_dir / RUN

    if not run_path.is_file():
        raise RuntimeError(
            f"Missing completed run manifest: {run_path}"
        )

    run_obj = json.loads(
        run_path.read_text(encoding="utf-8")
    )

    if run_obj.get("status") != "completed":
        raise RuntimeError(
            "Refusing to score a run whose status is not completed: "
            f"{run_obj.get('status')!r}"
        )

    identity = run_obj.get("identity") or {}

    if identity.get("benchmark") != args.benchmark:
        raise RuntimeError(
            "Run benchmark identity mismatch"
        )

    if (
        identity.get("custom_v4_manifest_sha256")
        != EXPECTED_MANIFEST_SHA256
    ):
        raise RuntimeError(
            "Run was not generated under the frozen v4 manifest"
        )

    # Official PRE/POST runs must contain the complete benchmark,
    # never a --limit subset.
    if identity.get("limit") is not None:
        raise RuntimeError(
            "Refusing to score a limited/pilot run as official v4"
        )

    if identity.get("prompt_count") != expected_count:
        raise RuntimeError(
            "Run prompt-count identity mismatch: "
            f"expected={expected_count} "
            f"actual={identity.get('prompt_count')}"
        )

    prompts = list(
        read_jsonl(
            str(run_dir / PROMPTS)
        )
    )

    if not prompts:
        raise RuntimeError(
            "No prompts.jsonl found"
        )

    if len(prompts) != expected_count:
        raise RuntimeError(
            "Frozen prompt count mismatch: "
            f"expected={expected_count} "
            f"actual={len(prompts)}"
        )

    adapter = get_adapter(
        args.benchmark
    )

    row_by_id = {
        row.row_id: row
        for row in adapter.rows()
    }

    generations = best_per_id(
        str(run_dir)
    )

    samples = []

    missing = 0
    infrastructure_errors = 0
    parser_errors = 0
    truncated = 0
    scored = 0
    score_sum = 0.0

    for prompt in prompts:
        item_id = prompt["id"]
        row_id = prompt["row_id"]

        if row_id not in row_by_id:
            raise RuntimeError(
                f"Frozen row disappeared: {row_id}"
            )

        row = row_by_id[row_id]

        expected_id = (
            f"{args.benchmark}::{row.row_id}"
        )

        if item_id != expected_id:
            raise RuntimeError(
                f"Prompt ID drift: {item_id}"
            )

        if prompt.get("benchmark") != args.benchmark:
            raise RuntimeError(
                f"Prompt benchmark drift: {item_id}"
            )

        expected_messages = adapter.messages(row)

        if prompt.get("messages") != expected_messages:
            raise RuntimeError(
                f"Frozen prompt drift: {item_id}"
            )

        if prompt.get("metadata") != row.metadata:
            raise RuntimeError(
                f"Frozen prompt metadata drift: {item_id}"
            )

        generation = generations.get(
            item_id
        )

        sample = {
            "id":
                item_id,
            "benchmark":
                args.benchmark,
            "row_id":
                row_id,
            "metadata":
                row.metadata,
            "prediction":
                None,
            "score":
                None,
            "parser_error":
                "",
            "generation_error":
                "",
            "truncated":
                False,
            "finish_reason":
                "",
        }

        if generation is None:
            missing += 1
            sample["generation_error"] = (
                "missing generation"
            )
            samples.append(sample)
            continue

        if generation.get("error"):
            infrastructure_errors += 1
            sample["generation_error"] = (
                generation["error"]
            )
            sample["finish_reason"] = (
                generation.get(
                    "finish_reason",
                    "",
                )
            )
            samples.append(sample)
            continue

        completion = generation.get(
            "completion",
            "",
        )

        sample["truncated"] = bool(
            generation.get(
                "truncated",
                False,
            )
        )

        sample["finish_reason"] = (
            generation.get(
                "finish_reason",
                "",
            )
        )

        if sample["truncated"]:
            truncated += 1

        try:
            prediction = adapter.parse(
                completion,
                row,
            )

            raw_score = adapter.score(
                prediction,
                row,
            )

            score = float(raw_score)

            if not math.isfinite(score):
                raise ValueError(
                    f"non-finite score: {score}"
                )

        except Exception as exc:
            parser_errors += 1

            sample["parser_error"] = (
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            samples.append(sample)
            continue

        sample["prediction"] = prediction
        sample["score"] = score

        score_sum += score
        scored += 1

        samples.append(sample)

    write_jsonl(
        str(run_dir / SAMPLES),
        samples,
    )

    complete = (
        missing == 0
        and infrastructure_errors == 0
        and parser_errors == 0
        and scored == len(prompts)
    )

    metrics = {
        "status":
            (
                "complete"
                if complete
                else "incomplete"
            ),
        "benchmark":
            args.benchmark,
        "total":
            len(prompts),
        "scored":
            scored,
        "missing":
            missing,
        "infrastructure_errors":
            infrastructure_errors,
        "parser_errors":
            parser_errors,
        "truncated":
            truncated,
        "accuracy":
            (
                score_sum / scored
                if complete
                else None
            ),
    }

    write_json(
        str(run_dir / METRICS),
        metrics,
    )

    print(
        json.dumps(
            metrics,
            indent=2,
        )
    )

    if not complete:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
