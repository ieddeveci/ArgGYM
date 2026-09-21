from __future__ import annotations

from collections import Counter
from pathlib import Path
import hashlib
import json


CANONICAL = Path("data/rl/curriculum/stage1.jsonl")
V1 = Path("data/rl/curriculum/stage1_levelmixed_v1.jsonl")

OUTPUT = Path(
    "data/rl/curriculum/stage1_levelmixed_v2.jsonl"
)
MANIFEST = Path(
    "data/rl/curriculum/stage1_levelmixed_v2.manifest.json"
)

EXPECTED_CANONICAL_SHA256 = (
    "1e08153b108369121e2f3f1d0b052af8179fe87e8e7fd8897b6cfc13be49a235"
)

EXPECTED_V1_SHA256 = (
    "6c47a451e62b45844e47f90953dcf019a8f015d7f3acd97d673a0c03822cf896"
)

ORDERING_VERSION = "stage1-levelmixed-v2"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def load(path: Path):
    rows = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for idx, line in enumerate(f):
            payload = line.rstrip("\n")
            obj = json.loads(payload)

            rows.append(
                {
                    "payload": payload,
                    "payload_sha256":
                        sha256_text(payload),
                    "task": obj["task"],
                    "level": int(obj["level"]),
                    "ordering": obj["ordering"],
                    "difficulty_band":
                        obj["difficulty_band"],
                    "source_position": idx,
                }
            )

    return rows


def group_bounds(group_index: int):
    start = group_index * 3
    return start, start + 3


def group_tasks(rows, group_index: int):
    a, b = group_bounds(group_index)
    return [
        r["task"]
        for r in rows[a:b]
    ]


def group_is_unique(rows, group_index: int):
    tasks = group_tasks(
        rows,
        group_index,
    )

    return len(tasks) == len(set(tasks))


def bad_groups(rows):
    return [
        g
        for g in range(len(rows) // 3)
        if not group_is_unique(rows, g)
    ]


canonical_sha = sha256_file(CANONICAL)
v1_sha = sha256_file(V1)

if canonical_sha != EXPECTED_CANONICAL_SHA256:
    raise SystemExit(
        "REFUSING: canonical Stage1 hash changed\n"
        f"expected={EXPECTED_CANONICAL_SHA256}\n"
        f"actual={canonical_sha}"
    )

if v1_sha != EXPECTED_V1_SHA256:
    raise SystemExit(
        "REFUSING: levelmixed v1 hash changed\n"
        f"expected={EXPECTED_V1_SHA256}\n"
        f"actual={v1_sha}"
    )


canonical_rows = load(CANONICAL)
rows = load(V1)

if len(canonical_rows) != 1440:
    raise SystemExit(
        "canonical Stage1 is not 1440 rows"
    )

if len(rows) != 1440:
    raise SystemExit(
        "v1 Stage1 is not 1440 rows"
    )


# ------------------------------------------------------------
# Verify v1 is exactly the canonical payload multiset.
# ------------------------------------------------------------

canonical_payloads = Counter(
    r["payload_sha256"]
    for r in canonical_rows
)

v1_payloads = Counter(
    r["payload_sha256"]
    for r in rows
)

if canonical_payloads != v1_payloads:
    raise SystemExit(
        "REFUSING: v1 payload multiset differs "
        "from canonical Stage1"
    )


# Strict level monotonicity before repair.
v1_levels = [
    r["level"]
    for r in rows
]

if v1_levels != sorted(v1_levels):
    raise SystemExit(
        "REFUSING: v1 is not level-monotonic"
    )


initial_bad = bad_groups(rows)

if initial_bad != [374, 436, 451]:
    raise SystemExit(
        "Unexpected v1 duplicate-task groups.\n"
        f"Expected zero-based groups "
        f"[374, 436, 451], got {initial_bad}"
    )


repairs = []


# ------------------------------------------------------------
# Repair one bad optimizer group at a time.
#
# Constraints:
#   1. swap only rows from the SAME level
#   2. never move an item across a level boundary
#   3. both affected optimizer groups must be task-unique
#   4. choose the nearest valid swap deterministically
#
# Therefore the curriculum level sequence remains byte-for-byte
# identical as a sequence of level labels.
# ------------------------------------------------------------

while True:
    bad = bad_groups(rows)

    if not bad:
        break

    target_group = bad[0]

    a, b = group_bounds(target_group)

    group_positions = list(
        range(a, b)
    )

    target_tasks = [
        rows[i]["task"]
        for i in group_positions
    ]

    task_counts = Counter(target_tasks)

    # Prefer moving one of the duplicated-task rows.
    candidate_i = [
        i
        for i in group_positions
        if task_counts[
            rows[i]["task"]
        ] > 1
    ]

    if not candidate_i:
        candidate_i = group_positions

    possibilities = []

    for i in candidate_i:
        level = rows[i]["level"]

        for j in range(len(rows)):
            if j == i:
                continue

            if rows[j]["level"] != level:
                continue

            other_group = j // 3

            if other_group == target_group:
                continue

            # Try swap.
            rows[i], rows[j] = rows[j], rows[i]

            target_ok = group_is_unique(
                rows,
                target_group,
            )

            other_ok = group_is_unique(
                rows,
                other_group,
            )

            # Restore.
            rows[i], rows[j] = rows[j], rows[i]

            if not (
                target_ok
                and other_ok
            ):
                continue

            possibilities.append(
                (
                    abs(j - i),
                    j,
                    i,
                    other_group,
                )
            )

    if not possibilities:
        raise SystemExit(
            "No same-level task-safe swap found "
            f"for optimizer group "
            f"{target_group + 1}"
        )

    possibilities.sort()

    _, j, i, other_group = (
        possibilities[0]
    )

    before_target = {
        "group":
            target_group + 1,
        "positions":
            list(range(a, b)),
        "tasks":
            group_tasks(
                rows,
                target_group,
            ),
    }

    oa, ob = group_bounds(
        other_group
    )

    before_other = {
        "group":
            other_group + 1,
        "positions":
            list(range(oa, ob)),
        "tasks":
            group_tasks(
                rows,
                other_group,
            ),
    }

    swap_a = {
        "position": i,
        "level": rows[i]["level"],
        "task": rows[i]["task"],
        "ordering": rows[i]["ordering"],
        "payload_sha256":
            rows[i]["payload_sha256"],
    }

    swap_b = {
        "position": j,
        "level": rows[j]["level"],
        "task": rows[j]["task"],
        "ordering": rows[j]["ordering"],
        "payload_sha256":
            rows[j]["payload_sha256"],
    }

    rows[i], rows[j] = rows[j], rows[i]

    repairs.append(
        {
            "target_group_before":
                before_target,
            "other_group_before":
                before_other,
            "swap_a":
                swap_a,
            "swap_b":
                swap_b,
            "target_group_after": {
                "group":
                    target_group + 1,
                "tasks":
                    group_tasks(
                        rows,
                        target_group,
                    ),
            },
            "other_group_after": {
                "group":
                    other_group + 1,
                "tasks":
                    group_tasks(
                        rows,
                        other_group,
                    ),
            },
        }
    )


# ------------------------------------------------------------
# Final invariants.
# ------------------------------------------------------------

if bad_groups(rows):
    raise SystemExit(
        "REFUSING TO WRITE: duplicate-task "
        "optimizer groups remain"
    )


final_levels = [
    r["level"]
    for r in rows
]

if final_levels != v1_levels:
    raise SystemExit(
        "REFUSING TO WRITE: level sequence changed"
    )


output_payloads = Counter(
    r["payload_sha256"]
    for r in rows
)

if output_payloads != canonical_payloads:
    raise SystemExit(
        "REFUSING TO WRITE: payload multiset changed"
    )


def structural_signature(items):
    return Counter(
        (
            r["task"],
            r["level"],
            r["ordering"],
            r["difficulty_band"],
        )
        for r in items
    )


if (
    structural_signature(rows)
    != structural_signature(
        canonical_rows
    )
):
    raise SystemExit(
        "REFUSING TO WRITE: structural "
        "signature changed"
    )


# Count level transitions / boundary groups.
cross_level_groups = []

for g in range(len(rows) // 3):
    a, b = group_bounds(g)

    levels = {
        r["level"]
        for r in rows[a:b]
    }

    if len(levels) > 1:
        cross_level_groups.append(
            g + 1
        )


# Maximum consecutive identical task.
max_task_streak = 0
current_task = None
streak = 0

for r in rows:
    if r["task"] == current_task:
        streak += 1
    else:
        current_task = r["task"]
        streak = 1

    max_task_streak = max(
        max_task_streak,
        streak,
    )


level_counts = Counter(
    r["level"]
    for r in rows
)

band_counts = Counter(
    r["difficulty_band"]
    for r in rows
)

task_counts = Counter(
    r["task"]
    for r in rows
)


# Write v2 only now, after all checks.
with OUTPUT.open(
    "w",
    encoding="utf-8",
) as f:
    for r in rows:
        f.write(r["payload"])
        f.write("\n")


output_sha = sha256_file(OUTPUT)


manifest = {
    "ordering_version":
        ORDERING_VERSION,

    "canonical_stage1": {
        "path":
            str(CANONICAL),
        "sha256":
            canonical_sha,
        "rows":
            len(canonical_rows),
    },

    "input_v1": {
        "path":
            str(V1),
        "sha256":
            v1_sha,
    },

    "output_v2": {
        "path":
            str(OUTPUT),
        "sha256":
            output_sha,
        "rows":
            len(rows),
    },

    "repair_policy":
        (
            "Deterministic same-level swaps "
            "only. No row may cross a level "
            "boundary. Every global 3-row "
            "optimizer group must contain "
            "three distinct task identities."
        ),

    "invariants": {
        "same_canonical_payload_multiset":
            True,
        "same_structural_signature":
            True,
        "identical_level_sequence_to_v1":
            True,
        "levels_monotonic_1_to_15":
            True,
        "all_480_optimizer_groups_have_3_unique_tasks":
            True,
        "new_instances_generated":
            False,
        "canonical_stage1_modified":
            False,
        "frozen_arggym_taskset_modified":
            False,
    },

    "repairs":
        repairs,

    "diagnostics": {
        "repairs_performed":
            len(repairs),
        "optimizer_groups":
            480,
        "optimizer_groups_with_3_unique_tasks":
            480,
        "cross_level_optimizer_groups":
            cross_level_groups,
        "cross_level_optimizer_group_count":
            len(cross_level_groups),
        "max_consecutive_same_task":
            max_task_streak,
    },

    "level_counts": {
        str(k): level_counts[k]
        for k in sorted(level_counts)
    },

    "difficulty_band_counts":
        dict(sorted(
            band_counts.items()
        )),

    "task_counts":
        dict(sorted(
            task_counts.items()
        )),
}


with MANIFEST.open(
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        manifest,
        f,
        indent=2,
        sort_keys=True,
    )
    f.write("\n")


manifest_sha = sha256_file(
    MANIFEST
)


print("=" * 72)
print("STAGE1 LEVEL-MIXED V2")
print("=" * 72)

print("canonical Stage1:")
print(" ", canonical_sha)

print()
print("input v1:")
print(" ", v1_sha)

print()
print("output v2:")
print(" ", OUTPUT)
print(" ", output_sha)

print()
print("manifest:")
print(" ", MANIFEST)
print(" ", manifest_sha)

print()
print("rows:", len(rows))
print(
    "same canonical payload multiset: PASS"
)
print(
    "same structural signature: PASS"
)
print(
    "identical level sequence to v1: PASS"
)
print(
    "levels monotonic: PASS"
)
print(
    "optimizer groups with 3 unique tasks: "
    "480/480"
)

print(
    "cross-level optimizer groups:",
    len(cross_level_groups),
    cross_level_groups,
)

print(
    "max consecutive same task:",
    max_task_streak,
)

print()
print("repairs performed:", len(repairs))

for n, repair in enumerate(
    repairs,
    start=1,
):
    print()
    print(f"repair {n}:")
    print(
        "  target before:",
        repair[
            "target_group_before"
        ],
    )
    print(
        "  other before:",
        repair[
            "other_group_before"
        ],
    )
    print(
        "  swap:",
        repair["swap_a"],
        "<->",
        repair["swap_b"],
    )
    print(
        "  target after:",
        repair[
            "target_group_after"
        ],
    )
    print(
        "  other after:",
        repair[
            "other_group_after"
        ],
    )
