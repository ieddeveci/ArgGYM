from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import json
import math

import reasoning_gym

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

from transfer.v4.verify_reasoning_gym_freeze import (
    EXPECTED_DATA_SHA256,
    EXPECTED_FREEZE_MANIFEST_SHA256,
    EXPECTED_RG_REVISION,
    verify_reasoning_gym_freeze,
)


ROOT = Path("/arf/scratch/futan/ArgGYM")

EXPECTED_BASE_ENDPOINT_SHA256 = (
    "162f98bc6f65fecc5f914075d0d65fbabaa716f72b47a8d087e6c82fda0ee37d"
)


def main():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run-dir",
        required=True,
    )

    args = parser.parse_args()

    rows, _ = verify_reasoning_gym_freeze()

    run_dir = Path(args.run_dir)

    run_path = run_dir / RUN

    if not run_path.is_file():
        raise RuntimeError(
            f"Missing run manifest: {run_path}"
        )

    run = json.loads(
        run_path.read_text(
            encoding="utf-8"
        )
    )

    if run.get("status") != "completed":
        raise RuntimeError(
            "Refusing to score an incomplete "
            "Reasoning Gym run"
        )

    identity = run.get("identity") or {}

    expected_identity = {
        "benchmark":
            "reasoning_gym",
        "prompt_count":
            1000,
        "frozen_data_sha256":
            EXPECTED_DATA_SHA256,
        "freeze_manifest_sha256":
            EXPECTED_FREEZE_MANIFEST_SHA256,
        "reasoning_gym_revision":
            EXPECTED_RG_REVISION,
        "base_endpoint_sha256":
            EXPECTED_BASE_ENDPOINT_SHA256,
    }

    for key, expected in (
        expected_identity.items()
    ):
        if identity.get(key) != expected:
            raise RuntimeError(
                f"Run identity drift: {key}\n"
                f"expected={expected!r}\n"
                f"actual={identity.get(key)!r}"
            )

    row_by_id = {
        row["id"]: row
        for row in rows
    }

    if len(row_by_id) != 1000:
        raise RuntimeError(
            "Frozen RG ID inventory invalid"
        )

    prompts = list(
        read_jsonl(
            str(run_dir / PROMPTS)
        )
    )

    if len(prompts) != 1000:
        raise RuntimeError(
            f"Expected 1000 prompts, "
            f"got {len(prompts)}"
        )

    generations = best_per_id(
        str(run_dir)
    )

    scorers = {}

    samples = []

    missing = 0
    infrastructure_errors = 0
    scorer_errors = 0
    truncated = 0
    scored = 0

    all_scores = []
    task_scores = defaultdict(list)

    for prompt in prompts:
        item_id = prompt["id"]

        if item_id not in row_by_id:
            raise RuntimeError(
                f"Frozen row disappeared: "
                f"{item_id}"
            )

        row = row_by_id[item_id]
        entry = row["entry"]

        source_dataset = (
            entry["metadata"][
                "source_dataset"
            ]
        )

        expected_prompt = {
            "id":
                item_id,
            "benchmark":
                "reasoning_gym",
            "task":
                row["task"],
            "source_dataset":
                source_dataset,
            "generator_index":
                row["generator_index"],
            "messages": [
                {
                    "role": "user",
                    "content":
                        entry["question"],
                }
            ],
        }

        if prompt != expected_prompt:
            raise RuntimeError(
                f"Frozen prompt drift: "
                f"{item_id}"
            )

        generation = generations.get(
            item_id
        )

        sample = {
            "id":
                item_id,
            "task":
                row["task"],
            "source_dataset":
                source_dataset,
            "generator_index":
                row["generator_index"],
            "score":
                None,
            "generation_error":
                "",
            "scorer_error":
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
            if (
                source_dataset
                not in scorers
            ):
                scorers[
                    source_dataset
                ] = (
                    reasoning_gym
                    .get_score_answer_fn(
                        source_dataset
                    )
                )

            raw_score = scorers[
                source_dataset
            ](
                answer=completion,
                entry=entry,
            )

            score = float(raw_score)

            if not math.isfinite(score):
                raise ValueError(
                    f"non-finite score: "
                    f"{score}"
                )

        except Exception as exc:
            scorer_errors += 1

            sample["scorer_error"] = (
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            samples.append(sample)
            continue

        sample["score"] = score

        all_scores.append(score)

        task_scores[
            row["task"]
        ].append(score)

        scored += 1
        samples.append(sample)

    write_jsonl(
        str(run_dir / SAMPLES),
        samples,
    )

    complete = (
        missing == 0
        and infrastructure_errors == 0
        and scorer_errors == 0
        and scored == 1000
    )

    per_task = {}

    for task in sorted(task_scores):
        values = task_scores[task]

        per_task[task] = {
            "n":
                len(values),
            "mean_native_score":
                sum(values)
                / len(values),
            "full_credit_rate":
                sum(
                    score == 1.0
                    for score in values
                )
                / len(values),
        }

    metrics = {
        "status":
            (
                "complete"
                if complete
                else "incomplete"
            ),

        "benchmark":
            "reasoning_gym",

        "total":
            1000,

        "scored":
            scored,

        "missing":
            missing,

        "infrastructure_errors":
            infrastructure_errors,

        "scorer_errors":
            scorer_errors,

        "truncated":
            truncated,

        "overall_mean_native_score":
            (
                sum(all_scores)
                / len(all_scores)
                if complete
                else None
            ),

        "overall_full_credit_rate":
            (
                sum(
                    score == 1.0
                    for score in all_scores
                )
                / len(all_scores)
                if complete
                else None
            ),

        "per_task":
            per_task,
    }

    write_json(
        str(run_dir / METRICS),
        metrics,
    )

    print(
        json.dumps(
            metrics,
            indent=2,
            sort_keys=True,
        )
    )

    if not complete:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
