"""Run-directory layout.

The single place run paths are constructed, so the runner and the scorer cannot
disagree about where a sample's artifacts live.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Optional


def samples_dir(run_dir: Path) -> Path:
    return Path(run_dir) / "samples"


def sample_dir(run_dir: Path, sample_id: str) -> Path:
    return samples_dir(run_dir) / sample_id


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)


def read_json(path: Path) -> Optional[dict]:
    try:
        with open(path) as fh:
            return json.load(fh)
    except FileNotFoundError:
        return None


# Row fields that differ between the two benchmarks. v1 nests gold under
# `entry` and identifies an item by (mode, idx); ArgGYM_v2 stores `reference`
# at the top level and identifies by (ordering, seed). Everything else in the
# harness is shape-agnostic, so this is the only place that needs to know.
_V1_ONLY = ("mode", "idx")
_V2_ONLY = ("ordering", "seed", "min_directives")


def write_sample_input(run_dir: Path, row: dict) -> None:
    """Prompt and gold, written before inference so they exist even if it dies."""
    d = sample_dir(run_dir, row["sample_id"])
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "input.txt", "w") as fh:
        fh.write(row["prompt"])

    gold = {"sample_id": row["sample_id"], "task": row["task"],
            "level": row["level"]}
    for k in _V1_ONLY + _V2_ONLY:
        if k in row:
            gold[k] = row[k]

    entry = row.get("entry")
    if entry is not None:                      # v1
        gold["gold_answer"] = entry["answer"]
        gold["metadata"] = entry["metadata"]
    else:                                      # ArgGYM_v2
        gold["gold_answer"] = row["reference"]
        gold["metadata"] = row.get("metadata")
    write_json(d / "gold.json", gold)


def write_generation(run_dir: Path, sample_id: str, gen: dict) -> None:
    write_json(sample_dir(run_dir, sample_id) / "generation.json", gen)


def read_generation(run_dir: Path, sample_id: str) -> Optional[dict]:
    return read_json(sample_dir(run_dir, sample_id) / "generation.json")


def write_score(run_dir: Path, sample_id: str, score: dict) -> None:
    write_json(sample_dir(run_dir, sample_id) / "score.json", score)


def append_jsonl(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> Iterator[dict]:
    if not Path(path).exists():
        return
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def completed_sample_ids(run_dir: Path) -> set:
    """Sample ids already generated, so a resumed run skips them."""
    done = set()
    for rec in read_jsonl(Path(run_dir) / "generations.jsonl"):
        if rec.get("sample_id") and not rec.get("error"):
            done.add(rec["sample_id"])
    return done
