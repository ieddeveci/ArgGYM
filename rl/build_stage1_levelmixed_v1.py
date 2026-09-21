from __future__ import annotations

from collections import Counter, defaultdict, deque
from pathlib import Path
import hashlib
import json


SOURCE = Path("data/rl/curriculum/stage1.jsonl")
OUTPUT = Path("data/rl/curriculum/stage1_levelmixed_v1.jsonl")
MANIFEST = Path("data/rl/curriculum/stage1_levelmixed_v1.manifest.json")

EXPECTED_SOURCE_SHA256 = (
    "1e08153b108369121e2f3f1d0b052af8179fe87e8e7fd8897b6cfc13be49a235"
)

BASE_SEED = "arggym-rl-article-v1"
ORDERING_VERSION = "stage1-levelmixed-v1"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def deterministic_key(*parts: object) -> str:
    text = "|".join(str(x) for x in parts)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


source_sha = sha256_file(SOURCE)

if source_sha != EXPECTED_SOURCE_SHA256:
    raise SystemExit(
        "REFUSING TO CONTINUE: source Stage1 hash changed\n"
        f"expected: {EXPECTED_SOURCE_SHA256}\n"
        f"actual:   {source_sha}"
    )


records = []

with SOURCE.open("r", encoding="utf-8") as f:
    for idx, line in enumerate(f):
        payload = line.rstrip("\n")

        row = json.loads(payload)

        for required in (
            "task",
            "level",
            "ordering",
            "difficulty_band",
        ):
            if required not in row:
                raise SystemExit(
                    f"row {idx} missing required field {required!r}"
                )

        level = int(row["level"])

        if not 1 <= level <= 15:
            raise SystemExit(
                f"row {idx} has invalid level {level}"
            )

        records.append(
            {
                "source_index": idx,
                "payload": payload,
                "payload_sha256": sha256_bytes(
                    payload.encode("utf-8")
                ),
                "task": row["task"],
                "level": level,
                "ordering": row["ordering"],
                "difficulty_band": row["difficulty_band"],
            }
        )


if len(records) != 1440:
    raise SystemExit(
        f"expected 1440 Stage1 rows, found {len(records)}"
    )


# ------------------------------------------------------------
# Build task queues independently within every level.
#
# Within each task:
#   - ordering buckets are deterministically permuted
#   - examples from orderings are round-robin interleaved
#
# Across tasks:
#   - tasks are deterministically permuted for each level
#   - task queues are round-robin interleaved
#
# Therefore:
#   level is strictly curriculum ordered,
#   but task identity is mixed within each level.
# ------------------------------------------------------------

by_level = defaultdict(
    lambda: defaultdict(lambda: defaultdict(list))
)

for rec in records:
    by_level[rec["level"]][rec["task"]][rec["ordering"]].append(rec)


ordered_records = []


for level in range(1, 16):
    task_map = by_level[level]

    if not task_map:
        raise SystemExit(f"level {level} is empty")

    task_names = sorted(
        task_map,
        key=lambda task: deterministic_key(
            BASE_SEED,
            ORDERING_VERSION,
            "task-order",
            level,
            task,
        ),
    )

    task_queues = {}

    for task in task_names:
        ordering_map = task_map[task]

        ordering_names = sorted(
            ordering_map,
            key=lambda ordering: deterministic_key(
                BASE_SEED,
                ORDERING_VERSION,
                "ordering-order",
                level,
                task,
                ordering,
            ),
        )

        ordering_queues = {}

        for ordering in ordering_names:
            rows = sorted(
                ordering_map[ordering],
                key=lambda rec: deterministic_key(
                    BASE_SEED,
                    ORDERING_VERSION,
                    "row-order",
                    level,
                    task,
                    ordering,
                    rec["payload_sha256"],
                    rec["source_index"],
                ),
            )

            ordering_queues[ordering] = deque(rows)

        task_sequence = []

        while any(ordering_queues[o] for o in ordering_names):
            for ordering in ordering_names:
                q = ordering_queues[ordering]
                if q:
                    task_sequence.append(q.popleft())

        task_queues[task] = deque(task_sequence)

    # Round-robin across tasks.
    while any(task_queues[t] for t in task_names):
        for task in task_names:
            q = task_queues[task]
            if q:
                ordered_records.append(q.popleft())


if len(ordered_records) != len(records):
    raise SystemExit(
        "reordering changed row count: "
        f"{len(records)} -> {len(ordered_records)}"
    )


# ------------------------------------------------------------
# Strong equality check:
# The multiset of original JSONL payload bytes must be identical.
# Nothing may be generated, deleted, duplicated, or modified.
# ------------------------------------------------------------

source_payloads = Counter(
    rec["payload_sha256"]
    for rec in records
)

output_payloads = Counter(
    rec["payload_sha256"]
    for rec in ordered_records
)

if source_payloads != output_payloads:
    raise SystemExit(
        "REFUSING TO WRITE: output payload multiset differs from source"
    )


# Level monotonicity.
levels = [
    rec["level"]
    for rec in ordered_records
]

if levels != sorted(levels):
    raise SystemExit(
        "REFUSING TO WRITE: levels are not monotonically ordered"
    )


# Ensure task mixing.
max_task_streak = 0
current_task = None
current_streak = 0

for rec in ordered_records:
    if rec["task"] == current_task:
        current_streak += 1
    else:
        current_task = rec["task"]
        current_streak = 1

    max_task_streak = max(
        max_task_streak,
        current_streak,
    )


# Diagnostics for global 3-prompt optimizer groups.
three_row_groups = 0
three_row_groups_all_unique_tasks = 0
cross_level_groups = 0

for start in range(0, len(ordered_records), 3):
    chunk = ordered_records[start:start + 3]

    if len(chunk) != 3:
        continue

    three_row_groups += 1

    tasks = {
        rec["task"]
        for rec in chunk
    }

    levels_in_chunk = {
        rec["level"]
        for rec in chunk
    }

    if len(tasks) == 3:
        three_row_groups_all_unique_tasks += 1

    if len(levels_in_chunk) > 1:
        cross_level_groups += 1


# Counts before/after.
def count_signature(rows):
    return Counter(
        (
            rec["level"],
            rec["difficulty_band"],
            rec["task"],
            rec["ordering"],
        )
        for rec in rows
    )


if count_signature(records) != count_signature(ordered_records):
    raise SystemExit(
        "REFUSING TO WRITE: structural counts changed"
    )


# Write only the new artifact.
with OUTPUT.open("w", encoding="utf-8") as f:
    for rec in ordered_records:
        f.write(rec["payload"])
        f.write("\n")


output_sha = sha256_file(OUTPUT)


level_counts = Counter(
    rec["level"]
    for rec in ordered_records
)

band_counts = Counter(
    rec["difficulty_band"]
    for rec in ordered_records
)

task_counts = Counter(
    rec["task"]
    for rec in ordered_records
)


manifest = {
    "artifact": str(OUTPUT),
    "ordering_version": ORDERING_VERSION,
    "base_seed": BASE_SEED,

    "source": {
        "path": str(SOURCE),
        "sha256": source_sha,
        "rows": len(records),
    },

    "output": {
        "path": str(OUTPUT),
        "sha256": output_sha,
        "rows": len(ordered_records),
    },

    "invariants": {
        "same_payload_multiset": True,
        "same_structural_counts": True,
        "levels_monotonic_1_to_15": True,
        "new_instances_generated": False,
        "frozen_arggym_taskset_used": False,
        "source_stage1_modified": False,
    },

    "mixing": {
        "max_consecutive_same_task": max_task_streak,
        "three_row_groups": three_row_groups,
        "three_row_groups_all_unique_tasks":
            three_row_groups_all_unique_tasks,
        "three_row_groups_crossing_level_boundary":
            cross_level_groups,
    },

    "level_counts": {
        str(k): level_counts[k]
        for k in sorted(level_counts)
    },

    "difficulty_band_counts": dict(
        sorted(band_counts.items())
    ),

    "task_counts": dict(
        sorted(task_counts.items())
    ),
}


with MANIFEST.open("w", encoding="utf-8") as f:
    json.dump(
        manifest,
        f,
        indent=2,
        sort_keys=True,
    )
    f.write("\n")


print("=" * 72)
print("STAGE1 LEVEL-MIXED V1")
print("=" * 72)

print("source:")
print(" ", SOURCE)
print(" ", source_sha)

print()
print("output:")
print(" ", OUTPUT)
print(" ", output_sha)

print()
print("manifest:")
print(" ", MANIFEST)
print(" ", sha256_file(MANIFEST))

print()
print("rows:", len(ordered_records))
print("same payload multiset: PASS")
print("same structural counts: PASS")
print("levels monotonic: PASS")

print()
print("band counts:")
for k in sorted(band_counts):
    print(f"  {k}: {band_counts[k]}")

print()
print("level counts:")
for level in range(1, 16):
    print(
        f"  L{level:02d}: "
        f"{level_counts[level]}"
    )

print()
print("task counts:")
for task in sorted(task_counts):
    print(
        f"  {task}: "
        f"{task_counts[task]}"
    )

print()
print(
    "max consecutive same task:",
    max_task_streak,
)

print(
    "3-row groups with 3 unique tasks:",
    f"{three_row_groups_all_unique_tasks}/"
    f"{three_row_groups}",
)

print(
    "3-row groups crossing level boundary:",
    cross_level_groups,
)
