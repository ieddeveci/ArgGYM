#!/usr/bin/env python3
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from itertools import product
from pathlib import Path
import json
import os
import re

import reasoning_gym

ROOT = Path("/arf/scratch/futan/ArgGYM")
RG8 = ROOT / "data/transfer/pilots/rg8"

PARENT = RG8 / "rg8_qual24_v3.jsonl"
PARENT_EXPECTED_SHA256 = "e24090d07d17f99bddf016a5e67165dda492842f3218f18bdc83fe1df9802f16"

OUT = RG8 / "rg8_qual24_v4.jsonl"
MANIFEST = RG8 / "rg8_qual24_v4.manifest.json"
PREVIEW = RG8 / "rg8_qual24_v4.preview.txt"

EXPECTED_RG_REV = "49b07130b3fcd12f2d064bba7c43869543a0e7e7"

BUGGY_FAMILY_RELATIONS = {
    "mother-in-law",
    "father-in-law",
    "niece",
    "nephew",
}
SAFE_FAMILY_HARD = {"aunt", "uncle"}


def file_sha(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(x)
        for x in path.read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )


def syllogism_parse(statement: str) -> tuple[str, str, str]:
    statement = statement.strip()

    m = re.fullmatch(r"Some (.+) are not (.+)", statement)
    if m:
        return ("some_not", m.group(1), m.group(2))

    m = re.fullmatch(r"(All|No|Some) (.+) are (.+)", statement)
    if not m:
        raise ValueError(f"Cannot parse syllogism statement: {statement!r}")

    q = {
        "All": "all",
        "No": "no",
        "Some": "some",
    }[m.group(1)]

    return (q, m.group(2), m.group(3))


def statement_true(
    parsed: tuple[str, str, str],
    present_types: list[dict[str, bool]],
) -> bool:
    q, subj, pred = parsed

    if q == "all":
        return all(
            (not t[subj]) or t[pred]
            for t in present_types
        )

    if q == "no":
        return all(
            not (t[subj] and t[pred])
            for t in present_types
        )

    if q == "some":
        return any(
            t[subj] and t[pred]
            for t in present_types
        )

    if q == "some_not":
        return any(
            t[subj] and (not t[pred])
            for t in present_types
        )

    raise ValueError(q)


def independent_syllogism_label(entry: dict) -> tuple[bool, bool]:
    meta = entry["metadata"]

    p1 = syllogism_parse(meta["premise1"])
    p2 = syllogism_parse(meta["premise2"])
    c = syllogism_parse(meta["conclusion"])

    terms = sorted({
        p1[1], p1[2],
        p2[1], p2[2],
        c[1], c[2],
    })

    if len(terms) != 3:
        raise ValueError(
            f"Expected exactly 3 terms, got {terms}"
        )

    types = [
        dict(zip(terms, bits))
        for bits in product((False, True), repeat=len(terms))
    ]

    satisfiable = False
    counterexample = False

    # A model is represented by the set of unary-membership types
    # containing at least one individual. Counts >1 do not matter for
    # these categorical sentences. Exclude the empty domain.
    for mask in range(1, 1 << len(types)):
        present = [
            types[i]
            for i in range(len(types))
            if mask & (1 << i)
        ]

        if (
            statement_true(p1, present)
            and statement_true(p2, present)
        ):
            satisfiable = True
            if not statement_true(c, present):
                counterexample = True
                break

    entailed = satisfiable and not counterexample
    return satisfiable, entailed


def syllogism_level_ok(entry: dict, level: str) -> bool:
    meta = entry["metadata"]
    statements = [
        meta["premise1"],
        meta["premise2"],
        meta["conclusion"],
    ]

    has_some_not = any(
        re.fullmatch(r"Some .+ are not .+", s)
        for s in statements
    )

    if level == "easy":
        return all(s.startswith("All ") for s in statements)

    if level == "medium":
        return (
            not has_some_not
            and any(
                s.startswith("No ")
                or s.startswith("Some ")
                for s in statements
            )
        )

    if level == "hard":
        return has_some_not

    raise ValueError(level)


def syllogism_independently_valid(entry: dict) -> bool:
    satisfiable, entailed = independent_syllogism_label(entry)
    native_label = bool(entry["metadata"]["is_valid"])
    return satisfiable and entailed == native_label


def native_gold_score(task: str, entry: dict) -> float:
    scorer = reasoning_gym.get_score_answer_fn(task)

    if task == "propositional_logic":
        answer = entry["metadata"]["example_answer"]
    else:
        answer = entry["answer"]

    return float(
        scorer(
            answer=answer,
            entry=entry,
        )
    )


def add_pilot_metadata(
    entry: dict,
    level: str,
    cfg: dict,
    seed: int,
) -> dict:
    entry = deepcopy(entry)
    entry["metadata"] = deepcopy(entry.get("metadata") or {})
    entry["metadata"]["pilot_difficulty_label"] = level
    entry["metadata"]["pilot_generation_config"] = deepcopy(cfg)
    entry["metadata"]["pilot_seed"] = seed
    return entry


def first_safe_family_hard(row: dict) -> tuple[int, dict]:
    cfg = deepcopy(row["generation_config"])
    seed = int(row["generator_seed"])

    ds = reasoning_gym.create_dataset(
        "family_relationships",
        seed=seed,
        size=1000,
        **cfg,
    )

    for idx in range(1000):
        entry = ds[idx]
        rel = str(entry["metadata"]["relationship"]).lower()

        if rel not in SAFE_FAMILY_HARD:
            continue

        if native_gold_score("family_relationships", entry) != 1.0:
            continue

        return idx, add_pilot_metadata(
            entry,
            "hard",
            cfg,
            seed,
        )

    raise RuntimeError(
        "No safe aunt/uncle family_relationships hard row "
        "found in first 1000 candidates."
    )


def first_safe_syllogism(row: dict) -> tuple[int, dict]:
    level = row["difficulty"]
    cfg = deepcopy(row["generation_config"])
    seed = int(row["generator_seed"])

    ds = reasoning_gym.create_dataset(
        "syllogism",
        seed=seed,
        size=1000,
        **cfg,
    )

    for idx in range(1000):
        entry = ds[idx]

        if not syllogism_level_ok(entry, level):
            continue

        if not syllogism_independently_valid(entry):
            continue

        if native_gold_score("syllogism", entry) != 1.0:
            continue

        return idx, add_pilot_metadata(
            entry,
            level,
            cfg,
            seed,
        )

    raise RuntimeError(
        f"No independently validated syllogism row "
        f"found for level={level} in first 1000 candidates."
    )


def main() -> None:
    rg_rev = os.environ.get("ARGGYM_RG_REVISION", "").strip()

    if rg_rev != EXPECTED_RG_REV:
        raise RuntimeError(
            "Reasoning Gym revision mismatch: "
            f"expected={EXPECTED_RG_REV} actual={rg_rev!r}"
        )

    parent_sha = file_sha(PARENT)
    if parent_sha != PARENT_EXPECTED_SHA256:
        raise RuntimeError(
            "v3 parent hash mismatch: "
            f"expected={PARENT_EXPECTED_SHA256} actual={parent_sha}"
        )

    rows = read_jsonl(PARENT)

    if len(rows) != 24:
        raise RuntimeError(f"Expected 24 v3 rows, found {len(rows)}")

    repaired = []
    replacements = []

    for source_row in rows:
        row = deepcopy(source_row)

        old_id = row["id"]
        row["benchmark"] = "rg8_qual24_v4"
        row["id"] = old_id.replace(
            "rg8_qual24_v3::",
            "rg8_qual24_v4::",
            1,
        )

        row["entry"]["metadata"]["pilot_parent_id"] = old_id

        task = row["task"]
        level = row["difficulty"]

        if task == "family_relationships":
            rel = str(
                row["entry"]["metadata"]["relationship"]
            ).lower()

            if rel in BUGGY_FAMILY_RELATIONS:
                idx, entry = first_safe_family_hard(row)
                row["generator_index"] = idx
                row["entry"] = entry
                row["entry"]["metadata"]["pilot_parent_id"] = old_id
                replacements.append({
                    "id": row["id"],
                    "reason": f"excluded known direction-bug relation {rel}",
                    "new_generator_index": idx,
                    "new_relationship": entry["metadata"]["relationship"],
                })

        if task == "syllogism":
            if (
                not syllogism_level_ok(row["entry"], level)
                or not syllogism_independently_valid(row["entry"])
            ):
                old = {
                    "premise1": row["entry"]["metadata"]["premise1"],
                    "premise2": row["entry"]["metadata"]["premise2"],
                    "conclusion": row["entry"]["metadata"]["conclusion"],
                    "native_is_valid": row["entry"]["metadata"]["is_valid"],
                }

                idx, entry = first_safe_syllogism(row)
                row["generator_index"] = idx
                row["entry"] = entry
                row["entry"]["metadata"]["pilot_parent_id"] = old_id

                replacements.append({
                    "id": row["id"],
                    "reason": "native syllogism label disagreed with independent model-theoretic validation or level filter",
                    "old": old,
                    "new_generator_index": idx,
                    "new": {
                        "premise1": entry["metadata"]["premise1"],
                        "premise2": entry["metadata"]["premise2"],
                        "conclusion": entry["metadata"]["conclusion"],
                        "native_is_valid": entry["metadata"]["is_valid"],
                    },
                })

        repaired.append(row)

    # Structural assertions.
    if len(repaired) != 24:
        raise RuntimeError("v4 row count changed")

    if len({r["id"] for r in repaired}) != 24:
        raise RuntimeError("Duplicate v4 IDs")

    # Family rows: exclude known direction-bug categories.
    for row in repaired:
        if row["task"] == "family_relationships":
            rel = str(
                row["entry"]["metadata"]["relationship"]
            ).lower()

            if rel in BUGGY_FAMILY_RELATIONS:
                raise RuntimeError(
                    f"Unsafe family relationship survived: {rel}"
                )

            if row["difficulty"] == "hard" and rel not in SAFE_FAMILY_HARD:
                raise RuntimeError(
                    f"Hard family relation is not aunt/uncle: {rel}"
                )

    # Every syllogism must independently agree with its native label.
    for row in repaired:
        if row["task"] == "syllogism":
            if not syllogism_independently_valid(row["entry"]):
                raise RuntimeError(
                    f"Unsafe syllogism survived: {row['id']}"
                )

    # Native gold/witness score must remain 1.0 for every row.
    for row in repaired:
        score = native_gold_score(
            row["task"],
            row["entry"],
        )
        if score != 1.0:
            raise RuntimeError(
                f"Native gold self-score failure "
                f"{row['id']}: {score}"
            )

    write_jsonl(OUT, repaired)

    with PREVIEW.open("w", encoding="utf-8") as f:
        for row in repaired:
            e = row["entry"]
            f.write("=" * 100 + "\n")
            f.write(
                f"{row['task']} | {row['difficulty']} | "
                f"idx={row['generator_index']}\n"
            )
            f.write("QUESTION\n")
            f.write(e["question"] + "\n\n")

            if row["task"] == "propositional_logic":
                f.write(
                    "VALID EXAMPLE CONCLUSION\n"
                    + str(e["metadata"]["example_answer"])
                    + "\n\n"
                )
            else:
                f.write(
                    "GOLD\n"
                    + str(e["answer"])
                    + "\n\n"
                )

    manifest = {
        "benchmark": "rg8_qual24_v4",
        "status": "pilot_frozen",
        "reasoning_gym_revision": rg_rev,
        "rows": len(repaired),
        "parent_path": str(PARENT),
        "parent_sha256": parent_sha,
        "data_path": str(OUT),
        "data_sha256": file_sha(OUT),
        "preview_path": str(PREVIEW),
        "repairs": replacements,
        "family_filter": {
            "excluded": sorted(BUGGY_FAMILY_RELATIONS),
            "hard_allowed": sorted(SAFE_FAMILY_HARD),
            "reason": "pinned generator has directionally incorrect in-law and niece/nephew labeling branches",
        },
        "syllogism_filter": {
            "method": "independent finite-model entailment over the three unary predicates",
            "requires_premise_satisfiable": True,
            "requires_native_label_agreement": True,
        },
    }

    MANIFEST.write_text(
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
    print("RG8 QUAL24 V4 REPAIR: PASS")
    print("=" * 100)
    print("rows:", len(repaired))
    print("parent sha256:", parent_sha)
    print("v4 sha256:", manifest["data_sha256"])
    print("replacements:", len(replacements))
    for item in replacements:
        print(json.dumps(item, ensure_ascii=False, sort_keys=True))
    print("data:", OUT)
    print("manifest:", MANIFEST)
    print("preview:", PREVIEW)


if __name__ == "__main__":
    main()
