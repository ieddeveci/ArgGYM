#!/usr/bin/env python3

from collections import Counter, defaultdict
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import json
import os

import reasoning_gym


ROOT = Path("/arf/scratch/futan/ArgGYM")

EXPECTED_RG_REV = "49b07130b3fcd12f2d064bba7c43869543a0e7e7"

NAMESPACE = "course_boundary20_v1"
TASK = "course_schedule"

LEVELS = ("n13", "n14")
PER_LEVEL = 10

CONFIGS = {
    "n13": {
        "min_num_courses": 13,
        "max_num_courses": 13,
        "min_num_prerequisites": 2,
        "max_num_prerequisites": 3,
        "min_cycle_length": 3,
        "max_cycle_length": 5,
        "p_solvable": 0.5,
    },
    "n14": {
        "min_num_courses": 14,
        "max_num_courses": 14,
        "min_num_prerequisites": 2,
        "max_num_prerequisites": 3,
        "min_cycle_length": 3,
        "max_cycle_length": 5,
        "p_solvable": 0.5,
    },
}


def stable_seed(level):
    raw = f"{NAMESPACE}|{TASK}|{level}".encode()
    return int(sha256(raw).hexdigest()[:8], 16) & 0x7FFFFFFF


def sha256_file(path):
    h = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def gold_self_score(entry):
    scorer = reasoning_gym.get_score_answer_fn(TASK)
    return float(
        scorer(
            answer=entry["answer"],
            entry=entry,
        )
    )


def build_level(level):
    cfg = deepcopy(CONFIGS[level])
    seed = stable_seed(level)

    # Large deterministic candidate stream.
    ds = reasoning_gym.create_dataset(
        TASK,
        seed=seed,
        size=1000,
        **cfg,
    )

    selected = []
    counts = Counter()

    for source_idx in range(1000):
        if len(selected) == PER_LEVEL:
            break

        entry = deepcopy(ds[source_idx])

        metadata = entry.get("metadata") or {}

        if "solvable" not in metadata:
            raise RuntimeError(
                f"{level} candidate {source_idx}: "
                "missing metadata.solvable"
            )

        solvable = bool(metadata["solvable"])
        label = "true" if solvable else "false"

        if counts[label] >= 5:
            continue

        expected_answer = "True" if solvable else "False"

        if str(entry.get("answer")) != expected_answer:
            raise RuntimeError(
                f"{level} candidate {source_idx}: "
                f"answer={entry.get('answer')!r}, "
                f"solvable={solvable}"
            )

        self_score = gold_self_score(entry)

        if self_score != 1.0:
            raise RuntimeError(
                f"{level} candidate {source_idx}: "
                f"native gold self-score={self_score}"
            )

        counts[label] += 1

        entry["metadata"] = deepcopy(metadata)
        entry["metadata"]["pilot_difficulty_label"] = level
        entry["metadata"]["pilot_generation_config"] = deepcopy(cfg)
        entry["metadata"]["pilot_seed"] = seed

        selected.append(
            {
                "id": (
                    f"{NAMESPACE}::{TASK}::"
                    f"{level}::{len(selected):03d}"
                ),
                "benchmark": NAMESPACE,
                "task": TASK,
                "difficulty": level,
                "generator_index": source_idx,
                "generator_seed": seed,
                "generation_config": deepcopy(cfg),
                "entry": entry,
            }
        )

    if len(selected) != 10:
        raise RuntimeError(
            f"{level}: selected {len(selected)} rows, expected 10"
        )

    if counts != Counter({"true": 5, "false": 5}):
        raise RuntimeError(
            f"{level}: label balance failure {dict(counts)}"
        )

    return selected


def main():
    rg_rev = os.environ.get("ARGGYM_RG_REVISION", "").strip()

    if rg_rev != EXPECTED_RG_REV:
        raise RuntimeError(
            "Reasoning Gym revision mismatch: "
            f"expected={EXPECTED_RG_REV}, actual={rg_rev!r}"
        )

    rows = []

    for level in LEVELS:
        rows.extend(build_level(level))

    if len(rows) != 20:
        raise RuntimeError(
            f"expected 20 rows, got {len(rows)}"
        )

    if len({r["id"] for r in rows}) != 20:
        raise RuntimeError("duplicate row IDs")

    if len({r["entry"]["question"] for r in rows}) != 20:
        raise RuntimeError("duplicate questions")

    level_counts = Counter(
        r["difficulty"]
        for r in rows
    )

    expected_levels = Counter({
        "n13": 10,
        "n14": 10,
    })

    if level_counts != expected_levels:
        raise RuntimeError(
            f"bad level counts: {dict(level_counts)}"
        )

    balance = defaultdict(Counter)

    for row in rows:
        solvable = bool(
            row["entry"]["metadata"]["solvable"]
        )
        label = "true" if solvable else "false"

        balance[row["difficulty"]][label] += 1

    for level in LEVELS:
        expected = Counter({
            "true": 5,
            "false": 5,
        })

        if balance[level] != expected:
            raise RuntimeError(
                f"{level}: balance={dict(balance[level])}"
            )

    out_dir = (
        ROOT
        / "data/transfer/pilots/course_schedule"
    )
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    data_path = (
        out_dir
        / "course_boundary20_v1.jsonl"
    )

    manifest_path = (
        out_dir
        / "course_boundary20_v1.manifest.json"
    )

    preview_path = (
        out_dir
        / "course_boundary20_v1.preview.txt"
    )

    with data_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        for row in rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

    with preview_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        for row in rows:
            entry = row["entry"]

            f.write("=" * 100 + "\n")
            f.write(
                f"{row['id']} "
                f"source_idx={row['generator_index']} "
                f"gold={entry['answer']} "
                f"solvable="
                f"{entry['metadata']['solvable']}\n"
            )
            f.write("=" * 100 + "\n")
            f.write(entry["question"])
            f.write("\n\n")

    data_sha = sha256_file(data_path)

    manifest = {
        "benchmark": NAMESPACE,
        "status": "course_schedule_boundary_pilot_frozen",
        "reasoning_gym_revision": rg_rev,
        "rows": 20,
        "task": TASK,
        "levels": list(LEVELS),
        "configuration": CONFIGS,
        "level_counts": dict(level_counts),
        "label_balance": {
            level: dict(balance[level])
            for level in LEVELS
        },
        "selection_policy": (
            "Deterministic generator stream; "
            "first five native-valid solvable and "
            "first five native-valid unsolvable examples "
            "per level; no model outputs used for selection."
        ),
        "data_path": str(data_path),
        "data_sha256": data_sha,
        "preview_path": str(preview_path),
    }

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print("=" * 100)
    print("COURSE BOUNDARY20 BUILD: PASS")
    print("=" * 100)
    print("rows:", len(rows))
    print("data:", data_path)
    print("sha256:", data_sha)
    print(
        "level counts:",
        dict(level_counts),
    )
    print(
        "label balance:",
        {
            level: dict(balance[level])
            for level in LEVELS
        },
    )
    print("manifest:", manifest_path)
    print("preview:", preview_path)


if __name__ == "__main__":
    main()
