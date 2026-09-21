#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import argparse
import json
import os

import reasoning_gym

EXPECTED_RG_REV = "49b07130b3fcd12f2d064bba7c43869543a0e7e7"
NAMESPACE = "course_cal30_v1"
TASK = "course_schedule"
LEVELS = ("easy", "medium", "hard")
PER_CELL = 10

TASK_LEVELS = {
    "easy": {
        "min_num_courses": 5,
        "max_num_courses": 5,
        "min_num_prerequisites": 1,
        "max_num_prerequisites": 2,
        "min_cycle_length": 3,
        "max_cycle_length": 3,
        "p_solvable": 0.5,
    },
    "medium": {
        "min_num_courses": 8,
        "max_num_courses": 8,
        "min_num_prerequisites": 1,
        "max_num_prerequisites": 3,
        "min_cycle_length": 3,
        "max_cycle_length": 4,
        "p_solvable": 0.5,
    },
    "hard": {
        "min_num_courses": 12,
        "max_num_courses": 12,
        "min_num_prerequisites": 2,
        "max_num_prerequisites": 3,
        "min_cycle_length": 3,
        "max_cycle_length": 5,
        "p_solvable": 0.5,
    },
}


def file_sha(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_seed(level: str) -> int:
    raw = f"{NAMESPACE}|{TASK}|{level}".encode("utf-8")
    return int(sha256(raw).hexdigest()[:8], 16) & 0x7FFFFFFF


def gold_self_score(entry: dict) -> float:
    scorer = reasoning_gym.get_score_answer_fn(TASK)
    return float(scorer(answer=entry["answer"], entry=entry))


def select_level(level: str) -> list[dict]:
    cfg = deepcopy(TASK_LEVELS[level])
    seed = stable_seed(level)
    ds = reasoning_gym.create_dataset(TASK, seed=seed, size=1000, **cfg)

    rows = []
    labels = Counter()
    idx = 0

    while len(rows) < PER_CELL:
        if idx >= 1000:
            raise RuntimeError(
                f"candidate exhaustion level={level}; labels={dict(labels)}"
            )

        entry = deepcopy(ds[idx])
        source_idx = idx
        idx += 1

        meta = entry.get("metadata") or {}
        if "solvable" not in meta:
            raise RuntimeError(
                f"missing metadata.solvable level={level} idx={source_idx}"
            )

        solvable = bool(meta["solvable"])
        label = "true" if solvable else "false"
        if labels[label] >= 5:
            continue

        expected_answer = "True" if solvable else "False"
        if str(entry.get("answer")) != expected_answer:
            raise RuntimeError(
                f"answer/metadata mismatch level={level} idx={source_idx}: "
                f"answer={entry.get('answer')!r} solvable={solvable}"
            )

        score = gold_self_score(entry)
        if score != 1.0:
            raise RuntimeError(
                f"native gold self-score failure level={level} "
                f"idx={source_idx} score={score}"
            )

        labels[label] += 1

        entry["metadata"] = deepcopy(meta)
        entry["metadata"]["pilot_difficulty_label"] = level
        entry["metadata"]["pilot_generation_config"] = deepcopy(cfg)
        entry["metadata"]["pilot_seed"] = seed

        rows.append({
            "id": f"{NAMESPACE}::{TASK}::{level}::{len(rows):03d}",
            "benchmark": NAMESPACE,
            "task": TASK,
            "difficulty": level,
            "generator_index": source_idx,
            "generator_seed": seed,
            "generation_config": deepcopy(cfg),
            "entry": entry,
        })

    expected = Counter({"true": 5, "false": 5})
    if labels != expected:
        raise RuntimeError(
            f"label balance failure level={level}: {dict(labels)}"
        )

    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/arf/scratch/futan/ArgGYM")
    args = ap.parse_args()

    rg_rev = os.environ.get("ARGGYM_RG_REVISION", "").strip()
    if rg_rev != EXPECTED_RG_REV:
        raise RuntimeError(
            f"RG revision mismatch expected={EXPECTED_RG_REV} actual={rg_rev!r}"
        )

    rows = []
    for level in LEVELS:
        rows.extend(select_level(level))

    if len(rows) != 30:
        raise RuntimeError(f"expected 30 rows, got {len(rows)}")
    if len({r["id"] for r in rows}) != 30:
        raise RuntimeError("duplicate IDs")
    if len({r["entry"]["question"] for r in rows}) != 30:
        raise RuntimeError("duplicate questions")

    level_counts = Counter(r["difficulty"] for r in rows)
    if level_counts != Counter({"easy": 10, "medium": 10, "hard": 10}):
        raise RuntimeError(f"difficulty counts wrong: {dict(level_counts)}")

    balance = defaultdict(Counter)
    for r in rows:
        label = "true" if bool(r["entry"]["metadata"]["solvable"]) else "false"
        balance[r["difficulty"]][label] += 1

    for level in LEVELS:
        if balance[level] != Counter({"true": 5, "false": 5}):
            raise RuntimeError(
                f"balance wrong level={level}: {dict(balance[level])}"
            )

    root = Path(args.root)
    out_dir = root / "data/transfer/pilots/course_schedule"
    out_dir.mkdir(parents=True, exist_ok=True)

    data_path = out_dir / f"{NAMESPACE}.jsonl"
    manifest_path = out_dir / f"{NAMESPACE}.manifest.json"
    preview_path = out_dir / f"{NAMESPACE}.preview.txt"

    with data_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    with preview_path.open("w", encoding="utf-8") as f:
        for row in rows:
            e = row["entry"]
            f.write("=" * 100 + "\n")
            f.write(
                f"{row['id']} | idx={row['generator_index']} | "
                f"gold={e['answer']} | solvable={e['metadata']['solvable']}\n"
            )
            f.write("=" * 100 + "\n")
            f.write(e["question"] + "\n\n")

    manifest = {
        "benchmark": NAMESPACE,
        "status": "course_schedule_recalibration_pilot_frozen",
        "reasoning_gym_revision": rg_rev,
        "rows": len(rows),
        "task": TASK,
        "difficulty_policy": TASK_LEVELS,
        "per_difficulty": dict(sorted(level_counts.items())),
        "label_balance": {level: dict(balance[level]) for level in LEVELS},
        "selection_policy": (
            "deterministic generator order; exact 5 solvable + 5 unsolvable "
            "per difficulty; native gold self-score must equal 1.0; "
            "no model outputs used in selection"
        ),
        "data_path": str(data_path),
        "data_sha256": file_sha(data_path),
        "preview_path": str(preview_path),
    }

    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("=" * 100)
    print("COURSE CAL30 BUILD: PASS")
    print("=" * 100)
    print("rows:", len(rows))
    print("data:", data_path)
    print("sha256:", manifest["data_sha256"])
    print("difficulty counts:", dict(sorted(level_counts.items())))
    print("label balance:", {level: dict(balance[level]) for level in LEVELS})
    print("manifest:", manifest_path)
    print("preview:", preview_path)


if __name__ == "__main__":
    main()
