from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
from typing import Any

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
from transfer.v4.registry import ADAPTERS, get_adapter
from transfer.v4.verify_custom_freeze import (
    EXPECTED_MANIFEST_SHA256,
    MANIFEST as FREEZE_MANIFEST,
    verify_custom_freeze,
)


ROOT = Path("/arf/scratch/futan/ArgGYM")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)
    return h.hexdigest()


def protocol_guard() -> dict[str, str]:
    manifest = verify_custom_freeze()

    actual_manifest_sha256 = sha256(
        FREEZE_MANIFEST
    )

    if (
        actual_manifest_sha256
        != EXPECTED_MANIFEST_SHA256
    ):
        raise RuntimeError(
            "Custom v4 freeze manifest hash drift: "
            f"expected={EXPECTED_MANIFEST_SHA256} "
            f"actual={actual_manifest_sha256}"
        )

    return {
        "custom_v4_manifest_sha256":
            actual_manifest_sha256,
        "custom_v4_freeze_name":
            manifest["freeze_name"],
    }


def normalize_messages(
    messages: list[dict[str, str]],
) -> tuple[str | None, str]:
    roles = tuple(
        m["role"] for m in messages
    )

    if roles == ("user",):
        return None, messages[0]["content"]

    if roles == ("system", "user"):
        return (
            messages[0]["content"],
            messages[1]["content"],
        )

    raise RuntimeError(
        f"Unsupported frozen message shape: {roles}"
    )


def prompt_records(
    benchmark: str,
    adapter: Any,
    limit: int | None,
):
    rows = list(adapter.rows())

    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be positive")
        rows = rows[:limit]

    records = []

    for row in rows:
        messages = adapter.messages(row)

        # Revalidate the exact interface contract at runtime.
        normalize_messages(messages)

        records.append(
            {
                "id":
                    f"{benchmark}::{row.row_id}",
                "benchmark":
                    benchmark,
                "row_id":
                    row.row_id,
                "messages":
                    messages,
                "metadata":
                    row.metadata,
            }
        )

    return rows, records


def load_endpoint(path: Path) -> Endpoint:
    obj = json.loads(path.read_text())

    allowed = {
        "model",
        "base_url",
        "api_key_env",
        "sampling",
        "extra_body",
        "timeout_s",
        "retries",
    }

    unknown = set(obj) - allowed

    if unknown:
        raise ValueError(
            f"Unknown endpoint config keys: "
            f"{sorted(unknown)}"
        )

    endpoint = Endpoint(**obj)
    endpoint.check()

    return endpoint


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

    parser.add_argument(
        "--endpoint-config",
        required=True,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--prepare-only",
        action="store_true",
    )

    args = parser.parse_args()

    freeze = protocol_guard()

    run_dir = Path(args.run_dir)
    endpoint_path = Path(args.endpoint_config)

    if not endpoint_path.exists():
        raise RuntimeError(
            f"Missing endpoint config: {endpoint_path}"
        )

    endpoint = load_endpoint(endpoint_path)
    adapter = get_adapter(args.benchmark)

    rows, prompts = prompt_records(
        args.benchmark,
        adapter,
        args.limit,
    )

    row_by_id = {
        row.row_id: row
        for row in rows
    }

    identity = {
        "benchmark":
            args.benchmark,
        "prompt_count":
            len(prompts),
        "limit":
            args.limit,
        "endpoint":
            endpoint.redacted(),
        "endpoint_config_sha256":
            sha256(endpoint_path),
        "custom_v4_manifest_sha256":
            freeze["custom_v4_manifest_sha256"],
        "custom_v4_freeze_name":
            freeze["custom_v4_freeze_name"],
        "runner_sha256":
            sha256(Path(__file__)),
    }

    run_path = run_dir / RUN
    prompt_path = run_dir / PROMPTS
    generation_path = run_dir / GENERATIONS

    with exclusive(str(run_dir)):
        # Resume is allowed only if the run identity is
        # byte-for-byte scientifically equivalent.
        if run_path.exists():
            old = json.loads(
                run_path.read_text()
            )

            if old.get("identity") != identity:
                raise RuntimeError(
                    "Existing run directory has a "
                    "different run identity; refusing "
                    "to mix generations."
                )

        # Prompts are written before inference. On resume,
        # they must match exactly rather than being replaced.
        if prompt_path.exists():
            old_prompts = list(
                read_jsonl(str(prompt_path))
            )

            if old_prompts != prompts:
                raise RuntimeError(
                    "Existing prompts.jsonl differs "
                    "from frozen prompt materialization"
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
                "CUSTOM TRANSFER PREPARE: PASS"
            )
            print(
                "benchmark:",
                args.benchmark,
            )
            print(
                "prompts:",
                len(prompts),
            )
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

                row_id = prompt["row_id"]
                row = row_by_id[row_id]

                # Regenerate messages from the frozen adapter
                # and make sure they still equal prompts.jsonl.
                messages = adapter.messages(row)

                if messages != prompt["messages"]:
                    raise RuntimeError(
                        f"Prompt drift at {item_id}"
                    )

                system, user = normalize_messages(
                    messages
                )

                print(
                    f"[{i}/{len(prompts)}] "
                    f"{item_id}",
                    flush=True,
                )

                attempt = client.complete(
                    system,
                    user,
                )

                record = {
                    "id":
                        item_id,
                    "benchmark":
                        args.benchmark,
                    "row_id":
                        row_id,
                    **asdict(attempt),
                }

                out.write(record)

        final_done = completed_ids(
            str(run_dir)
        )

        expected_ids = {
            p["id"]
            for p in prompts
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
            "CUSTOM TRANSFER GENERATION:",
            status.upper(),
        )
        print(
            "successful:",
            successful,
        )
        print(
            "remaining:",
            remaining,
        )

        if remaining:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
