#!/usr/bin/env python3

from collections import Counter
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import json
import os

import reasoning_gym


ROOT = Path("/arf/scratch/futan/ArgGYM")
EXPECTED_RG_REV = "49b07130b3fcd12f2d064bba7c43869543a0e7e7"

NAMESPACE = "zebra_chars20_v1"
TASK = "zebra_puzzles"
LEVELS = ("c6", "c7")
PER_LEVEL = 10

CONFIGS = {
    "c6": {
        "num_people": 5,
        "num_characteristics": 6,
    },
    "c7": {
        "num_people": 5,
        "num_characteristics": 7,
    },
}


def stable_seed(level):
    raw = f"{NAMESPACE}|{TASK}|{level}".encode("utf-8")
    return int(sha256(raw).hexdigest()[:8], 16) & 0x7FFFFFFF


def file_sha256(path):
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

    ds = reasoning_gym.create_dataset(
        TASK,
        seed=seed,
        size=1000,
        **cfg,
    )

    selected = []
    seen_questions = set()

    for source_idx in range(1000):
        if len(selected) == PER_LEVEL:
            break

        entry = deepcopy(ds[source_idx])
        question = entry.get("question", "")

        if not question:
            raise RuntimeError(
                f"{level} idx={source_idx}: empty question"
            )

        if question in seen_questions:
            continue

        answer = entry.get("answer")

        if not isinstance(answer, str) or not answer.strip():
            raise RuntimeError(
                f"{level} idx={source_idx}: bad gold={answer!r}"
            )

        score = gold_self_score(entry)

        if score != 1.0:
            raise RuntimeError(
                f"{level} idx={source_idx}: gold self-score={score}"
            )

        seen_questions.add(question)

        meta = deepcopy(entry.get("metadata") or {})
        meta["pilot_difficulty_label"] = level
        meta["pilot_generation_config"] = deepcopy(cfg)
        meta["pilot_seed"] = seed
        entry["metadata"] = meta

        selected.append({
            "id": (
                f"{NAMESPACE}::{TASK}::{level}::"
                f"{len(selected):03d}"
            ),
            "benchmark": NAMESPACE,
            "task": TASK,
            "difficulty": level,
            "generator_index": source_idx,
            "generator_seed": seed,
            "generation_config": deepcopy(cfg),
            "entry": entry,
        })

    if len(selected) != 10:
        raise RuntimeError(
            f"{level}: selected {len(selected)}, expected 10"
        )

    return selected


def main():
    rg_rev = os.environ.get("ARGGYM_RG_REVISION", "").strip()

    if rg_rev != EXPECTED_RG_REV:
        raise RuntimeError(
            f"RG revision mismatch: "
            f"expected={EXPECTED_RG_REV} actual={rg_rev!r}"
        )

    rows = []

    for level in LEVELS:
        rows.extend(build_level(level))

    if len(rows) != 20:
        raise RuntimeError(f"expected 20 rows, got {len(rows)}")

    if len({r["id"] for r in rows}) != 20:
        raise RuntimeError("duplicate IDs")

    if len({r["entry"]["question"] for r in rows}) != 20:
        raise RuntimeError("duplicate questions")

    counts = Counter(r["difficulty"] for r in rows)

    if counts != Counter({"c6": 10, "c7": 10}):
        raise RuntimeError(f"bad counts: {dict(counts)}")

    out_dir = ROOT / "data/transfer/pilots/zebra_puzzles"
    out_dir.mkdir(parents=True, exist_ok=True)

    data_path = out_dir / f"{NAMESPACE}.jsonl"
    manifest_path = out_dir / f"{NAMESPACE}.manifest.json"
    preview_path = out_dir / f"{NAMESPACE}.preview.txt"

    with data_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

    with preview_path.open("w", encoding="utf-8") as f:
        for row in rows:
            e = row["entry"]
            f.write("=" * 100 + "\n")
            f.write(
                f"{row['id']} "
                f"idx={row['generator_index']} "
                f"gold={e['answer']!r}\n"
            )
            f.write("=" * 100 + "\n")
            f.write(e["question"] + "\n\n")

    data_sha = file_sha256(data_path)

    manifest = {
        "benchmark": NAMESPACE,
        "status": "zebra_characteristics_calibration_frozen",
        "reasoning_gym_revision": rg_rev,
        "rows": 20,
        "task": TASK,
        "configs": CONFIGS,
        "level_counts": dict(counts),
        "selection_policy": (
            "Deterministic generator stream; first ten unique "
            "native-gold-valid examples per cell; no model outcomes used."
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
    print("ZEBRA CHARS20 BUILD: PASS")
    print("=" * 100)
    print("rows:", len(rows))
    print("data:", data_path)
    print("sha256:", data_sha)
    print("level counts:", dict(counts))
    print("manifest:", manifest_path)
    print("preview:", preview_path)


if __name__ == "__main__":
    main()
