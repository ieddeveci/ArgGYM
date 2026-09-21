from __future__ import annotations

import hashlib
import json

from collections import Counter

from transfer.integrity import (
    ROOT,
    verify_frozen,
    sha256_file,
)

from transfer.registry import ADAPTERS
from transfer.vendor_utils import (
    VENDOR,
    verify_vendor,
)


OUT = (
    ROOT
    / "outputs"
    / "transfer"
    / "protocol_freeze"
)

MANIFEST = (
    OUT
    / "custom_eval_inputs_manifest.jsonl"
)

SUMMARY = (
    OUT
    / "custom_eval_inputs_summary.json"
)

EXPECTED_COUNTS = {
    "multilogieval": 1556,
    "rulearena": 816,
    "logiqa2": 1572,
    # 2000 frozen states evaluated under
    # both official FineReason components.
    "finereason": 4000,
}


def canonical(obj):
    return json.dumps(
        obj,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def main():
    verify_frozen(
        full_file_check=True
    )

    verify_vendor()

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    if (
        MANIFEST.exists()
        or SUMMARY.exists()
    ):
        raise SystemExit(
            "REFUSING TO OVERWRITE existing "
            "custom eval-input freeze artifacts"
        )

    seen = set()
    counts = Counter()
    rows_out = []

    for name, cls in ADAPTERS.items():
        print(
            f"preflight: {name}"
        )

        adapter = cls()

        for row in adapter.rows():
            if row.row_id in seen:
                raise RuntimeError(
                    "duplicate row_id: "
                    f"{row.row_id}"
                )

            seen.add(
                row.row_id
            )

            messages = (
                adapter.messages(row)
            )

            # Prompt construction must be
            # deterministic.
            if (
                messages
                != adapter.messages(row)
            ):
                raise RuntimeError(
                    "nondeterministic prompt: "
                    f"{row.row_id}"
                )

            if (
                not messages
                or any(
                    message.get("role")
                    not in {
                        "system",
                        "user",
                    }
                    for message
                    in messages
                )
            ):
                raise RuntimeError(
                    "unsupported messages: "
                    f"{row.row_id}"
                )

            rows_out.append({
                "benchmark":
                    name,
                "row_id":
                    row.row_id,
                "messages_sha256":
                    hashlib.sha256(
                        canonical(messages)
                    ).hexdigest(),
                "gold_sha256":
                    hashlib.sha256(
                        canonical(row.gold)
                    ).hexdigest(),
                "metadata":
                    row.metadata,
            })

            counts[name] += 1

    if dict(counts) != EXPECTED_COUNTS:
        raise RuntimeError(
            f"count mismatch: "
            f"{dict(counts)} != "
            f"{EXPECTED_COUNTS}"
        )

    with MANIFEST.open(
        "w",
        encoding="utf-8",
    ) as f:
        for rec in rows_out:
            f.write(
                json.dumps(
                    rec,
                    sort_keys=True,
                    ensure_ascii=False,
                )
                + "\n"
            )

    code_files = (
        sorted(
            (ROOT / "transfer").glob(
                "*.py"
            )
        )
        + sorted(
            (
                ROOT
                / "transfer"
                / "adapters"
            ).glob("*.py")
        )
    )

    summary = {
        "status":
            "CUSTOM_TRANSFER_INPUTS_"
            "FROZEN_PRE_EVAL",
        "counts":
            dict(counts),
        "total":
            sum(counts.values()),
        "frozen_data_manifest_sha256":
            sha256_file(
                ROOT
                / "data"
                / "transfer"
                / "frozen"
                / "manifest.json"
            ),
        "transfer_protocol_sha256":
            sha256_file(
                ROOT
                / "configs"
                / "transfer"
                / "benchmarks.yaml"
            ),
        "vendor_manifest_sha256":
            sha256_file(
                VENDOR
                / "manifest.json"
            ),
        "eval_inputs_manifest_sha256":
            sha256_file(
                MANIFEST
            ),
        "code_sha256": {
            p.relative_to(
                ROOT
            ).as_posix():
                sha256_file(p)
            for p in code_files
        },
    }

    SUMMARY.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "CUSTOM TRANSFER PREFLIGHT: PASS"
    )

    print(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
