from __future__ import annotations

from collections import Counter, defaultdict, deque
from copy import deepcopy
from pathlib import Path
import hashlib
import json


ROOT = Path("/arf/scratch/futan/ArgGYM")
DATA = ROOT / "data/rl/curriculum"

BASE_SEED = "arggym-rl-article-v1"
FROZEN_HASH = "121f452f2744ef0a6022c4731097b6b1"

EXPECTED = {
    "stage2": {
        "canonical_sha256":
            "9622ea1ebaa5d74f6493867cd22c0724f922ae2e7f1808bf2da53faad05e9c79",
        "namespace": "RL_TRAIN_STAGE_2",
        "bands": {
            "easy": 576,
            "medium": 432,
            "hard": 432,
        },
    },
    "stage3": {
        "canonical_sha256":
            "546b2374bc2922a42bf07bc5ff9f50aa9d70274d80e3b48b5ad21d4ee4449de9",
        "namespace": "RL_TRAIN_STAGE_3",
        "bands": {
            "easy": 288,
            "medium": 432,
            "hard": 720,
        },
    },
}


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


def deterministic_key(*parts: object) -> str:
    text = "|".join(str(x) for x in parts)
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def load_records(path: Path, stage: str):
    records = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for idx, line in enumerate(f):
            payload = line.rstrip("\n")
            row = json.loads(payload)

            for required in (
                "task",
                "level",
                "ordering",
                "difficulty_band",
                "curriculum_stage",
                "split_namespace",
            ):
                if required not in row:
                    raise SystemExit(
                        f"{path}: row {idx} missing "
                        f"{required!r}"
                    )

            if row["curriculum_stage"] != stage:
                raise SystemExit(
                    f"{path}: row {idx} curriculum_stage="
                    f"{row['curriculum_stage']!r}, expected "
                    f"{stage!r}"
                )

            level = int(row["level"])

            if not 1 <= level <= 15:
                raise SystemExit(
                    f"{path}: row {idx} invalid level={level}"
                )

            records.append(
                {
                    "source_index": idx,
                    "payload": payload,
                    "payload_sha256":
                        sha256_text(payload),
                    "task": row["task"],
                    "level": level,
                    "ordering": row["ordering"],
                    "difficulty_band":
                        row["difficulty_band"],
                }
            )

    if len(records) != 1440:
        raise SystemExit(
            f"{path}: expected 1440 rows, "
            f"found {len(records)}"
        )

    return records


def payload_multiset(rows):
    return Counter(
        r["payload_sha256"]
        for r in rows
    )


def structural_signature(rows):
    return Counter(
        (
            r["task"],
            r["level"],
            r["ordering"],
            r["difficulty_band"],
        )
        for r in rows
    )


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


def diagnostics(rows):
    cross_level = []

    for g in range(len(rows) // 3):
        a, b = group_bounds(g)

        levels = {
            r["level"]
            for r in rows[a:b]
        }

        if len(levels) > 1:
            cross_level.append(g + 1)

    max_streak = 0
    current_task = None
    streak = 0

    for r in rows:
        if r["task"] == current_task:
            streak += 1
        else:
            current_task = r["task"]
            streak = 1

        max_streak = max(
            max_streak,
            streak,
        )

    unique_groups = sum(
        group_is_unique(rows, g)
        for g in range(len(rows) // 3)
    )

    return {
        "optimizer_groups":
            len(rows) // 3,
        "optimizer_groups_with_3_unique_tasks":
            unique_groups,
        "cross_level_optimizer_groups":
            cross_level,
        "cross_level_optimizer_group_count":
            len(cross_level),
        "max_consecutive_same_task":
            max_streak,
    }


def counts(rows):
    return {
        "level": Counter(
            r["level"]
            for r in rows
        ),
        "band": Counter(
            r["difficulty_band"]
            for r in rows
        ),
        "task": Counter(
            r["task"]
            for r in rows
        ),
        "ordering": Counter(
            r["ordering"]
            for r in rows
        ),
    }


def verify_canonical(
    stage: str,
    canonical: Path,
):
    expected = EXPECTED[stage]

    actual_sha = sha256_file(canonical)

    if actual_sha != expected["canonical_sha256"]:
        raise SystemExit(
            f"{stage}: canonical hash drift\n"
            f"expected={expected['canonical_sha256']}\n"
            f"actual={actual_sha}"
        )

    sidecar = canonical.with_suffix(
        canonical.suffix + ".manifest.json"
    )

    if not sidecar.exists():
        raise SystemExit(
            f"{stage}: missing canonical sidecar "
            f"{sidecar}"
        )

    m = json.loads(
        sidecar.read_text(
            encoding="utf-8",
        )
    )

    if m.get("data_sha256") != actual_sha:
        raise SystemExit(
            f"{stage}: canonical sidecar "
            f"data_sha256 mismatch"
        )

    if (
        m.get("split_namespace")
        != expected["namespace"]
    ):
        raise SystemExit(
            f"{stage}: unexpected namespace "
            f"{m.get('split_namespace')!r}"
        )

    if m.get("curriculum_stage") != stage:
        raise SystemExit(
            f"{stage}: sidecar curriculum_stage "
            f"mismatch"
        )

    if (
        m.get("master_seed_namespace")
        != BASE_SEED
    ):
        raise SystemExit(
            f"{stage}: master seed namespace "
            f"mismatch"
        )

    if (
        m.get("frozen_taskset_hash")
        != FROZEN_HASH
    ):
        raise SystemExit(
            f"{stage}: frozen taskset hash "
            f"mismatch"
        )

    if int(m.get("n_rows", -1)) != 1440:
        raise SystemExit(
            f"{stage}: sidecar n_rows != 1440"
        )

    return m


def build_v1(stage: str):
    canonical = DATA / f"{stage}.jsonl"

    output = (
        DATA
        / f"{stage}_levelmixed_v1.jsonl"
    )

    manifest_path = (
        DATA
        / f"{stage}_levelmixed_v1.manifest.json"
    )

    if output.exists() or manifest_path.exists():
        raise SystemExit(
            f"{stage}: refusing to overwrite "
            f"existing V1 artifacts"
        )

    canonical_manifest = verify_canonical(
        stage,
        canonical,
    )

    records = load_records(
        canonical,
        stage,
    )

    expected_ns = EXPECTED[stage]["namespace"]

    for rec in records:
        obj = json.loads(rec["payload"])

        if (
            obj["split_namespace"]
            != expected_ns
        ):
            raise SystemExit(
                f"{stage}: row namespace mismatch"
            )

    version = f"{stage}-levelmixed-v1"

    by_level = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(list)
        )
    )

    for rec in records:
        by_level[
            rec["level"]
        ][
            rec["task"]
        ][
            rec["ordering"]
        ].append(rec)

    ordered = []

    for level in range(1, 16):
        task_map = by_level[level]

        if not task_map:
            raise SystemExit(
                f"{stage}: level {level} empty"
            )

        task_names = sorted(
            task_map,
            key=lambda task:
                deterministic_key(
                    BASE_SEED,
                    version,
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
                key=lambda ordering:
                    deterministic_key(
                        BASE_SEED,
                        version,
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
                    key=lambda rec:
                        deterministic_key(
                            BASE_SEED,
                            version,
                            "row-order",
                            level,
                            task,
                            ordering,
                            rec[
                                "payload_sha256"
                            ],
                            rec[
                                "source_index"
                            ],
                        ),
                )

                ordering_queues[
                    ordering
                ] = deque(rows)

            task_sequence = []

            while any(
                ordering_queues[o]
                for o in ordering_names
            ):
                for ordering in ordering_names:
                    q = ordering_queues[
                        ordering
                    ]

                    if q:
                        task_sequence.append(
                            q.popleft()
                        )

            task_queues[
                task
            ] = deque(task_sequence)

        while any(
            task_queues[t]
            for t in task_names
        ):
            for task in task_names:
                q = task_queues[task]

                if q:
                    ordered.append(
                        q.popleft()
                    )

    if len(ordered) != 1440:
        raise SystemExit(
            f"{stage}: V1 row count changed"
        )

    if (
        payload_multiset(records)
        != payload_multiset(ordered)
    ):
        raise SystemExit(
            f"{stage}: V1 payload multiset "
            f"changed"
        )

    levels = [
        r["level"]
        for r in ordered
    ]

    if levels != sorted(levels):
        raise SystemExit(
            f"{stage}: V1 levels not monotonic"
        )

    if (
        structural_signature(records)
        != structural_signature(ordered)
    ):
        raise SystemExit(
            f"{stage}: V1 structural "
            f"signature changed"
        )

    c = counts(ordered)
    d = diagnostics(ordered)

    expected_bands = Counter(
        EXPECTED[stage]["bands"]
    )

    if c["band"] != expected_bands:
        raise SystemExit(
            f"{stage}: V1 band counts mismatch "
            f"{dict(c['band'])}"
        )

    if (
        len(c["task"]) != 12
        or set(c["task"].values())
        != {120}
    ):
        raise SystemExit(
            f"{stage}: V1 task counts invalid"
        )

    if (
        len(c["ordering"]) != 4
        or set(c["ordering"].values())
        != {360}
    ):
        raise SystemExit(
            f"{stage}: V1 ordering counts invalid"
        )

    with output.open(
        "w",
        encoding="utf-8",
    ) as f:
        for rec in ordered:
            f.write(rec["payload"])
            f.write("\n")

    output_sha = sha256_file(output)

    manifest = {
        "artifact": str(
            output.relative_to(ROOT)
        ),
        "ordering_version": version,
        "base_seed": BASE_SEED,
        "source": {
            "path": str(
                canonical.relative_to(ROOT)
            ),
            "sha256":
                sha256_file(canonical),
            "rows": 1440,
        },
        "output": {
            "path": str(
                output.relative_to(ROOT)
            ),
            "sha256":
                output_sha,
            "rows": 1440,
        },
        "invariants": {
            "same_payload_multiset":
                True,
            "same_structural_counts":
                True,
            "levels_monotonic_1_to_15":
                True,
            "new_instances_generated":
                False,
            "frozen_arggym_taskset_used":
                False,
            "canonical_stage_modified":
                False,
        },
        "mixing": d,
        "level_counts": {
            str(k): c["level"][k]
            for k in sorted(c["level"])
        },
        "difficulty_band_counts":
            dict(sorted(
                c["band"].items()
            )),
        "task_counts":
            dict(sorted(
                c["task"].items()
            )),
    }

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "canonical":
            canonical,
        "canonical_manifest":
            canonical_manifest,
        "canonical_rows":
            records,
        "v1":
            output,
        "v1_manifest":
            manifest_path,
        "v1_rows":
            ordered,
        "v1_sha":
            output_sha,
        "initial_bad":
            bad_groups(ordered),
    }


def build_v2(stage: str, state):
    canonical = state["canonical"]
    canonical_rows = state[
        "canonical_rows"
    ]

    v1 = state["v1"]

    output = (
        DATA
        / f"{stage}_levelmixed_v2.jsonl"
    )

    ordering_manifest_path = (
        DATA
        / f"{stage}_levelmixed_v2.manifest.json"
    )

    sidecar_path = output.with_suffix(
        output.suffix + ".manifest.json"
    )

    for p in (
        output,
        ordering_manifest_path,
        sidecar_path,
    ):
        if p.exists():
            raise SystemExit(
                f"{stage}: refusing to overwrite "
                f"{p}"
            )

    rows = load_records(
        v1,
        stage,
    )

    if (
        payload_multiset(rows)
        != payload_multiset(
            canonical_rows
        )
    ):
        raise SystemExit(
            f"{stage}: V1 payload differs "
            f"from canonical"
        )

    v1_levels = [
        r["level"]
        for r in rows
    ]

    if v1_levels != sorted(v1_levels):
        raise SystemExit(
            f"{stage}: V1 not level-monotonic"
        )

    initial_bad = bad_groups(rows)

    repairs = []

    while True:
        bad = bad_groups(rows)

        if not bad:
            break

        target_group = bad[0]

        a, b = group_bounds(
            target_group
        )

        group_positions = list(
            range(a, b)
        )

        target_tasks = [
            rows[i]["task"]
            for i in group_positions
        ]

        task_counts = Counter(
            target_tasks
        )

        candidate_i = [
            i
            for i in group_positions
            if task_counts[
                rows[i]["task"]
            ] > 1
        ]

        if not candidate_i:
            candidate_i = (
                group_positions
            )

        possibilities = []

        for i in candidate_i:
            level = rows[i]["level"]

            for j in range(len(rows)):
                if j == i:
                    continue

                if (
                    rows[j]["level"]
                    != level
                ):
                    continue

                other_group = j // 3

                if (
                    other_group
                    == target_group
                ):
                    continue

                rows[i], rows[j] = (
                    rows[j],
                    rows[i],
                )

                target_ok = (
                    group_is_unique(
                        rows,
                        target_group,
                    )
                )

                other_ok = (
                    group_is_unique(
                        rows,
                        other_group,
                    )
                )

                rows[i], rows[j] = (
                    rows[j],
                    rows[i],
                )

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
                f"{stage}: no same-level "
                f"task-safe swap found for "
                f"optimizer group "
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
            "level":
                rows[i]["level"],
            "task":
                rows[i]["task"],
            "ordering":
                rows[i]["ordering"],
            "payload_sha256":
                rows[i][
                    "payload_sha256"
                ],
        }

        swap_b = {
            "position": j,
            "level":
                rows[j]["level"],
            "task":
                rows[j]["task"],
            "ordering":
                rows[j]["ordering"],
            "payload_sha256":
                rows[j][
                    "payload_sha256"
                ],
        }

        rows[i], rows[j] = (
            rows[j],
            rows[i],
        )

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

    if bad_groups(rows):
        raise SystemExit(
            f"{stage}: duplicate-task "
            f"optimizer groups remain"
        )

    final_levels = [
        r["level"]
        for r in rows
    ]

    if final_levels != v1_levels:
        raise SystemExit(
            f"{stage}: level sequence changed "
            f"during V2 repairs"
        )

    if (
        payload_multiset(rows)
        != payload_multiset(
            canonical_rows
        )
    ):
        raise SystemExit(
            f"{stage}: V2 payload multiset "
            f"changed"
        )

    if (
        structural_signature(rows)
        != structural_signature(
            canonical_rows
        )
    ):
        raise SystemExit(
            f"{stage}: V2 structural "
            f"signature changed"
        )

    c = counts(rows)
    d = diagnostics(rows)

    if (
        d[
            "optimizer_groups_with_3_unique_tasks"
        ]
        != 480
    ):
        raise SystemExit(
            f"{stage}: not all optimizer "
            f"groups have 3 unique tasks"
        )

    with output.open(
        "w",
        encoding="utf-8",
    ) as f:
        for rec in rows:
            f.write(rec["payload"])
            f.write("\n")

    output_sha = sha256_file(output)

    version = (
        f"{stage}-levelmixed-v2"
    )

    ordering_manifest = {
        "ordering_version":
            version,
        "canonical_stage": {
            "path": str(
                canonical.relative_to(ROOT)
            ),
            "sha256":
                sha256_file(canonical),
            "rows": 1440,
        },
        "input_v1": {
            "path": str(
                v1.relative_to(ROOT)
            ),
            "sha256":
                state["v1_sha"],
        },
        "output_v2": {
            "path": str(
                output.relative_to(ROOT)
            ),
            "sha256":
                output_sha,
            "rows": 1440,
        },
        "repair_policy":
            (
                "Deterministic same-level "
                "swaps only. No row may cross "
                "a level boundary. Every "
                "global 3-row optimizer group "
                "must contain three distinct "
                "task identities."
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
            "canonical_stage_modified":
                False,
            "frozen_arggym_taskset_modified":
                False,
        },
        "initial_bad_optimizer_groups":
            [
                g + 1
                for g in initial_bad
            ],
        "repairs":
            repairs,
        "diagnostics": {
            **d,
            "repairs_performed":
                len(repairs),
        },
        "level_counts": {
            str(k): c["level"][k]
            for k in sorted(c["level"])
        },
        "difficulty_band_counts":
            dict(sorted(
                c["band"].items()
            )),
        "task_counts":
            dict(sorted(
                c["task"].items()
            )),
        "ordering_counts":
            dict(sorted(
                c["ordering"].items()
            )),
    }

    ordering_manifest_path.write_text(
        json.dumps(
            ordering_manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    # Mirror Stage-1 V2 sidecar policy:
    # retain all canonical provenance/cells,
    # change only data_sha256.
    sidecar = deepcopy(
        state[
            "canonical_manifest"
        ]
    )

    canonical_sidecar = (
        canonical.with_suffix(
            canonical.suffix
            + ".manifest.json"
        )
    )

    original_sidecar = json.loads(
        canonical_sidecar.read_text(
            encoding="utf-8",
        )
    )

    if sidecar != original_sidecar:
        raise SystemExit(
            f"{stage}: unexpected sidecar "
            f"mutation before write"
        )

    sidecar["data_sha256"] = (
        output_sha
    )

    sidecar_path.write_text(
        json.dumps(
            sidecar,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    written_sidecar = json.loads(
        sidecar_path.read_text(
            encoding="utf-8",
        )
    )

    changed = [
        k
        for k in sorted(
            set(original_sidecar)
            | set(written_sidecar)
        )
        if (
            original_sidecar.get(k)
            != written_sidecar.get(k)
        )
    ]

    if changed != ["data_sha256"]:
        raise SystemExit(
            f"{stage}: V2 sidecar changed "
            f"unexpected fields: {changed}"
        )

    if (
        original_sidecar.get("cells")
        != written_sidecar.get("cells")
    ):
        raise SystemExit(
            f"{stage}: sidecar cells changed"
        )

    return {
        "stage":
            stage,
        "canonical_sha":
            sha256_file(canonical),
        "v1_sha":
            state["v1_sha"],
        "v2_sha":
            output_sha,
        "v1_manifest_sha":
            sha256_file(
                state["v1_manifest"]
            ),
        "v2_manifest_sha":
            sha256_file(
                ordering_manifest_path
            ),
        "v2_sidecar_sha":
            sha256_file(
                sidecar_path
            ),
        "initial_bad_groups":
            [
                g + 1
                for g in initial_bad
            ],
        "repairs":
            len(repairs),
        "diagnostics":
            d,
        "level_counts":
            dict(sorted(
                c["level"].items()
            )),
        "band_counts":
            dict(sorted(
                c["band"].items()
            )),
    }


def main():
    results = []

    for stage in (
        "stage2",
        "stage3",
    ):
        print()
        print("=" * 100)
        print(
            f"{stage.upper()} "
            f"LEVEL-MIXED BUILD"
        )
        print("=" * 100)

        state = build_v1(stage)
        result = build_v2(
            stage,
            state,
        )

        results.append(result)

        print(
            "canonical sha256:",
            result["canonical_sha"],
        )
        print(
            "v1 sha256:",
            result["v1_sha"],
        )
        print(
            "v2 sha256:",
            result["v2_sha"],
        )
        print(
            "initial bad optimizer groups:",
            result[
                "initial_bad_groups"
            ],
        )
        print(
            "repairs performed:",
            result["repairs"],
        )
        print(
            "optimizer groups unique:",
            f"{result['diagnostics']['optimizer_groups_with_3_unique_tasks']}/480",
        )
        print(
            "cross-level groups:",
            result[
                "diagnostics"
            ][
                "cross_level_optimizer_groups"
            ],
        )
        print(
            "max same-task streak:",
            result[
                "diagnostics"
            ][
                "max_consecutive_same_task"
            ],
        )
        print(
            "band counts:",
            result["band_counts"],
        )
        print(
            "level counts:",
            result["level_counts"],
        )
        print(
            "v1 manifest sha256:",
            result[
                "v1_manifest_sha"
            ],
        )
        print(
            "v2 ordering manifest sha256:",
            result[
                "v2_manifest_sha"
            ],
        )
        print(
            "v2 sidecar sha256:",
            result[
                "v2_sidecar_sha"
            ],
        )

    print()
    print("=" * 100)
    print(
        "STAGE2 + STAGE3 "
        "LEVEL-MIXED V2 BUILD: PASS"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()
