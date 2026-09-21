from __future__ import annotations

from collections import Counter
from pathlib import Path
import hashlib
import json


ROOT = Path("/arf/scratch/futan/ArgGYM")

DATA = (
    ROOT
    / "data/transfer/v4/frozen/reasoning_gym/"
    "reasoning_gym_v4_1000.jsonl"
)

FREEZE_DIR = (
    ROOT
    / "outputs/transfer/protocol_freeze/v4/"
    "reasoning_gym"
)

EXPECTED_DATA_SHA256 = (
    "c005eca94a9bd9c8ba1c2a3660349e300daddd101cf427bb61f573f5ccd31ca6"
)

EXPECTED_FREEZE_MANIFEST_SHA256 = (
    "edb05e3736b632f1384aa6996ce1fae972adce04a35e9a8a1a8434e2a8062a97"
)

EXPECTED_RG_REVISION = (
    "49b07130b3fcd12f2d064bba7c43869543a0e7e7"
)

EXPECTED_TASKS = {
    "reasoning_gym.logic.knights_knaves":
        "knights_knaves",
    "reasoning_gym.logic.zebra_puzzles":
        "zebra_puzzles",
    "reasoning_gym.logic.self_reference":
        "self_reference",
    "reasoning_gym.logic.circuit_logic":
        "circuit_logic",
    "reasoning_gym.graphs.shortest_path":
        "shortest_path",
    "reasoning_gym.graphs.family_relationships":
        "family_relationships",
    "reasoning_gym.algorithmic.cryptarithm":
        "cryptarithm",
    "reasoning_gym.games.sudoku":
        "sudoku",
    "reasoning_gym.games.n_queens":
        "n_queens",
    "reasoning_gym.algebra.polynomial_equations":
        "polynomial_equations",
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


def fail(message: str):
    raise RuntimeError(
        "REASONING GYM V4 FREEZE VERIFICATION FAILED:\n"
        + message
    )


def verify_reasoning_gym_freeze():
    if not DATA.is_file():
        fail(f"missing frozen data: {DATA}")

    data_sha = sha256_file(DATA)

    if data_sha != EXPECTED_DATA_SHA256:
        fail(
            "frozen JSONL hash drift\n"
            f"expected={EXPECTED_DATA_SHA256}\n"
            f"actual  ={data_sha}"
        )

    rows = [
        json.loads(line)
        for line in DATA.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    if len(rows) != 1000:
        fail(
            f"expected 1000 rows, got {len(rows)}"
        )

    ids = set()
    task_counts = Counter()
    source_counts = Counter()

    for row in rows:
        required = {
            "id",
            "task",
            "entry",
            "config",
            "generator_index",
            "seed",
            "curriculum_global_level",
            "range_mode",
        }

        missing = required - set(row)

        if missing:
            fail(
                f"{row.get('id')}: missing keys "
                f"{sorted(missing)}"
            )

        item_id = row["id"]

        if item_id in ids:
            fail(f"duplicate id: {item_id}")

        ids.add(item_id)

        task = row["task"]

        if task not in EXPECTED_TASKS:
            fail(f"unexpected task: {task}")

        entry = row["entry"]

        if not {
            "question",
            "answer",
            "metadata",
        }.issubset(entry):
            fail(
                f"{item_id}: malformed frozen entry"
            )

        source = (
            entry["metadata"]
            .get("source_dataset")
        )

        expected_source = EXPECTED_TASKS[task]

        if source != expected_source:
            fail(
                f"{item_id}: source_dataset drift\n"
                f"expected={expected_source!r}\n"
                f"actual  ={source!r}"
            )

        if row["seed"] != 1234:
            fail(
                f"{item_id}: seed drift"
            )

        if row["curriculum_global_level"] != 2:
            fail(
                f"{item_id}: curriculum level drift"
            )

        if row["range_mode"] != "upper_bound":
            fail(
                f"{item_id}: range mode drift"
            )

        task_counts[task] += 1
        source_counts[source] += 1

    if set(task_counts) != set(EXPECTED_TASKS):
        fail("task inventory drift")

    if set(task_counts.values()) != {100}:
        fail(
            f"expected 100 rows/task: {task_counts}"
        )

    if set(source_counts.values()) != {100}:
        fail(
            f"expected 100 rows/source: "
            f"{source_counts}"
        )

    # Locate the already-frozen RG manifest by its
    # previously recorded immutable SHA rather than
    # guessing its filename.
    matches = []

    if not FREEZE_DIR.is_dir():
        fail(
            f"missing freeze directory: {FREEZE_DIR}"
        )

    for path in FREEZE_DIR.rglob("*"):
        if not path.is_file():
            continue

        try:
            got = sha256_file(path)
        except OSError:
            continue

        if got == EXPECTED_FREEZE_MANIFEST_SHA256:
            matches.append(path)

    if len(matches) != 1:
        fail(
            "could not uniquely identify frozen RG "
            f"manifest: matches={matches}"
        )

    manifest_path = matches[0]

    print("=" * 72)
    print("REASONING GYM V4 FREEZE VERIFICATION")
    print("=" * 72)
    print("rows:", len(rows))
    print("tasks:", len(task_counts))
    print("data sha256:", data_sha)
    print(
        "freeze manifest:",
        manifest_path.relative_to(ROOT),
    )
    print(
        "freeze manifest sha256:",
        EXPECTED_FREEZE_MANIFEST_SHA256,
    )
    print()
    print(
        "REASONING GYM V4 FREEZE VERIFICATION: PASS"
    )

    return rows, manifest_path


if __name__ == "__main__":
    verify_reasoning_gym_freeze()
