from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import re
from pathlib import Path

from evals.artifacts import (
    Appender,
    GENERATIONS,
    PROMPTS,
    RUN,
    completed_ids,
    exclusive,
    read_jsonl,
    write_json,
    write_jsonl,
)
from evals.client import ChatClient, Endpoint

from transfer.v4.verify_reasoning_gym_freeze import (
    EXPECTED_DATA_SHA256,
    EXPECTED_FREEZE_MANIFEST_SHA256,
    EXPECTED_RG_REVISION,
    verify_reasoning_gym_freeze,
)


ROOT = Path("/arf/scratch/futan/ArgGYM")

BASE_ENDPOINT = (
    ROOT
    / "configs/transfer/v4/"
    "qwen3_14b_pre.json"
)

EXPECTED_BASE_ENDPOINT_SHA256 = (
    "162f98bc6f65fecc5f914075d0d65fbabaa716f72b47a8d087e6c82fda0ee37d"
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def scientific_endpoint(obj: dict) -> dict:
    """
    Localhost TCP port is execution plumbing, not a
    scientific generation parameter.
    """
    out = dict(obj)
    out.pop("base_url", None)
    return out


def load_endpoint(path: Path) -> Endpoint:
    if not path.is_file():
        raise RuntimeError(
            f"Missing endpoint config: {path}"
        )

    if (
        sha256_file(BASE_ENDPOINT)
        != EXPECTED_BASE_ENDPOINT_SHA256
    ):
        raise RuntimeError(
            "Frozen base endpoint config drift"
        )

    frozen = json.loads(
        BASE_ENDPOINT.read_text(
            encoding="utf-8"
        )
    )

    runtime = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if (
        scientific_endpoint(runtime)
        != scientific_endpoint(frozen)
    ):
        raise RuntimeError(
            "Runtime endpoint differs scientifically "
            "from frozen v4 endpoint config"
        )

    url = runtime.get("base_url")

    if not re.fullmatch(
        r"http://127\.0\.0\.1:\d+/v1",
        str(url),
    ):
        raise RuntimeError(
            f"Unexpected runtime base_url: {url!r}"
        )

    endpoint = Endpoint(**runtime)
    endpoint.check()

    return endpoint


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run-dir",
        required=True,
    )

    parser.add_argument(
        "--endpoint-config",
        required=True,
    )

    parser.add_argument(
        "--prepare-only",
        action="store_true",
    )

    args = parser.parse_args()

    rows, manifest_path = (
        verify_reasoning_gym_freeze()
    )

    run_dir = Path(args.run_dir)
    endpoint_path = Path(
        args.endpoint_config
    )

    endpoint = load_endpoint(
        endpoint_path
    )

    prompts = []
    row_by_id = {}

    for row in rows:
        item_id = row["id"]

        if item_id in row_by_id:
            raise RuntimeError(
                f"Duplicate RG id: {item_id}"
            )

        row_by_id[item_id] = row

        # EXACT prompt protocol used by the prior
        # 1,000-row tokenizer audit:
        # one raw question as one user message;
        # no system instruction.
        messages = [
            {
                "role": "user",
                "content":
                    row["entry"]["question"],
            }
        ]

        prompts.append({
            "id":
                item_id,
            "benchmark":
                "reasoning_gym",
            "task":
                row["task"],
            "source_dataset":
                row["entry"]["metadata"][
                    "source_dataset"
                ],
            "generator_index":
                row["generator_index"],
            "messages":
                messages,
        })

    if len(prompts) != 1000:
        raise RuntimeError(
            f"Expected 1000 prompts, got "
            f"{len(prompts)}"
        )

    frozen_endpoint = json.loads(
        BASE_ENDPOINT.read_text(
            encoding="utf-8"
        )
    )

    identity = {
        "benchmark":
            "reasoning_gym",
        "prompt_count":
            1000,
        "frozen_data_sha256":
            EXPECTED_DATA_SHA256,
        "freeze_manifest_sha256":
            EXPECTED_FREEZE_MANIFEST_SHA256,
        "freeze_manifest_path":
            str(
                manifest_path.relative_to(ROOT)
            ),
        "reasoning_gym_revision":
            EXPECTED_RG_REVISION,
        "base_endpoint_sha256":
            EXPECTED_BASE_ENDPOINT_SHA256,
        "scientific_endpoint":
            scientific_endpoint(
                frozen_endpoint
            ),
        "runner_sha256":
            sha256_file(Path(__file__)),
    }

    run_path = run_dir / RUN
    prompt_path = run_dir / PROMPTS
    generation_path = (
        run_dir / GENERATIONS
    )

    with exclusive(str(run_dir)):
        if run_path.exists():
            old = json.loads(
                run_path.read_text(
                    encoding="utf-8"
                )
            )

            if old.get("identity") != identity:
                raise RuntimeError(
                    "Existing RG run directory has "
                    "a different scientific identity; "
                    "refusing to mix generations"
                )

        if prompt_path.exists():
            old_prompts = list(
                read_jsonl(
                    str(prompt_path)
                )
            )

            if old_prompts != prompts:
                raise RuntimeError(
                    "Existing RG prompts differ from "
                    "the frozen materialization"
                )
        else:
            write_jsonl(
                str(prompt_path),
                prompts,
            )

        write_json(
            str(run_path),
            {
                "status":
                    (
                        "prepared"
                        if args.prepare_only
                        else "running"
                    ),
                "identity":
                    identity,
            },
        )

        if args.prepare_only:
            print(
                "REASONING GYM PREPARE: PASS"
            )
            print("prompts:", len(prompts))
            return

        done = completed_ids(
            str(run_dir)
        )

        client = ChatClient(endpoint)

        with Appender(
            str(generation_path)
        ) as out:
            for i, prompt in enumerate(
                prompts,
                start=1,
            ):
                item_id = prompt["id"]

                if item_id in done:
                    continue

                row = row_by_id[item_id]

                expected_messages = [
                    {
                        "role": "user",
                        "content":
                            row["entry"][
                                "question"
                            ],
                    }
                ]

                if (
                    prompt["messages"]
                    != expected_messages
                ):
                    raise RuntimeError(
                        f"Prompt drift: {item_id}"
                    )

                print(
                    f"[{i}/1000] {item_id}",
                    flush=True,
                )

                attempt = client.complete(
                    None,
                    row["entry"]["question"],
                )

                out.write({
                    "id":
                        item_id,
                    "benchmark":
                        "reasoning_gym",
                    "task":
                        row["task"],
                    "source_dataset":
                        row["entry"][
                            "metadata"
                        ]["source_dataset"],
                    "generator_index":
                        row["generator_index"],
                    **asdict(attempt),
                })

        final_done = completed_ids(
            str(run_dir)
        )

        expected_ids = {
            prompt["id"]
            for prompt in prompts
        }

        successful = len(
            expected_ids & final_done
        )

        remaining = len(
            expected_ids - final_done
        )

        status = (
            "completed"
            if remaining == 0
            else "incomplete"
        )

        write_json(
            str(run_path),
            {
                "status":
                    status,
                "identity":
                    identity,
                "successful":
                    successful,
                "remaining":
                    remaining,
            },
        )

        print(
            "REASONING GYM GENERATION:",
            status.upper(),
        )
        print("successful:", successful)
        print("remaining:", remaining)

        if remaining:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
