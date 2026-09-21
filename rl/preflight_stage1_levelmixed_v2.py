from __future__ import annotations

from collections import Counter
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]

SRC = ROOT / "data/rl/curriculum/stage1.jsonl"
V2 = ROOT / "data/rl/curriculum/stage1_levelmixed_v2.jsonl"

V2_ORDER_MANIFEST = (
    ROOT / "data/rl/curriculum/stage1_levelmixed_v2.manifest.json"
)

V2_POOL_MANIFEST = (
    ROOT / "data/rl/curriculum/stage1_levelmixed_v2.jsonl.manifest.json"
)

EXPECTED_SRC = (
    "1e08153b108369121e2f3f1d0b052af8179fe87e8e7fd8897b6cfc13be49a235"
)

EXPECTED_V2 = (
    "ddabdb9ff221899234100a93d07a2a6eb5c809423f6df803270fc3111c0d789c"
)

EXPECTED_ORDER_MANIFEST = (
    "310bcea949e00ffd25ff5330777a244f2320ab4dbdedc01c1673dc33fc590638"
)

EXPECTED_FROZEN_HASH = (
    "121f452f2744ef0a6022c4731097b6b1"
)

EXPECTED_LEVEL_COUNTS = {
    1: 144,
    2: 144,
    3: 144,
    4: 144,
    5: 144,
    6: 100,
    7: 101,
    8: 101,
    9: 101,
    10: 101,
    11: 43,
    12: 43,
    13: 44,
    14: 43,
    15: 43,
}

EXPECTED_BANDS = {
    "easy": 720,
    "medium": 504,
    "hard": 216,
}

EXPECTED_TASKS = {
    "attack": 120,
    "attack_defense": 120,
    "claim_chain": 120,
    "counter_argument": 120,
    "counter_argument_strict": 120,
    "defeat_diagnosis": 120,
    "defence": 120,
    "formalization": 120,
    "perturbation": 120,
    "preference_construction": 120,
    "semantics_query": 120,
    "status_query": 120,
}

EXPECTED_CROSS_LEVEL_GROUPS = [
    274,
    341,
    375,
    423,
    437,
    452,
    466,
]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path: Path):
    rows = []
    payload_hashes = []

    with path.open(encoding="utf-8") as f:
        for line in f:
            payload = line.rstrip("\n")
            payload_hashes.append(
                hashlib.sha256(payload.encode("utf-8")).hexdigest()
            )
            rows.append(json.loads(payload))

    return rows, payload_hashes


def fail(msg):
    raise SystemExit("LEVELMIXED V2 PREFLIGHT FAILED: " + msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--skip-token-audit", action="store_true")
    args = ap.parse_args()

    config_path = Path(args.config)

    # --------------------------------------------------------
    # Immutable artifact hashes.
    # --------------------------------------------------------

    if sha(SRC) != EXPECTED_SRC:
        fail("canonical Stage1 hash changed")

    if sha(V2) != EXPECTED_V2:
        fail("levelmixed v2 hash changed")

    if sha(V2_ORDER_MANIFEST) != EXPECTED_ORDER_MANIFEST:
        fail("levelmixed ordering manifest hash changed")

    # --------------------------------------------------------
    # Exact dataset equivalence.
    # --------------------------------------------------------

    src, src_hashes = load(SRC)
    v2, v2_hashes = load(V2)

    if len(src) != 1440 or len(v2) != 1440:
        fail("Stage1 row count is not exactly 1440")

    if Counter(src_hashes) != Counter(v2_hashes):
        fail("v2 is not the exact canonical Stage1 payload multiset")

    sig = lambda r: (
        r["task"],
        int(r["level"]),
        r["ordering"],
        r["difficulty_band"],
    )

    if Counter(map(sig, src)) != Counter(map(sig, v2)):
        fail("structural signature differs from canonical Stage1")

    # --------------------------------------------------------
    # Curriculum order.
    # --------------------------------------------------------

    levels = [int(r["level"]) for r in v2]

    if levels != sorted(levels):
        fail("levels are not monotonically ordered")

    if dict(sorted(Counter(levels).items())) != EXPECTED_LEVEL_COUNTS:
        fail(
            f"unexpected level counts: "
            f"{dict(sorted(Counter(levels).items()))}"
        )

    band_counts = Counter(
        r["difficulty_band"] for r in v2
    )

    if dict(band_counts) != EXPECTED_BANDS:
        fail(f"unexpected band counts: {dict(band_counts)}")

    task_counts = Counter(
        r["task"] for r in v2
    )

    if dict(task_counts) != EXPECTED_TASKS:
        fail(f"unexpected task counts: {dict(task_counts)}")

    # --------------------------------------------------------
    # Exact optimizer-group structure:
    # 3 unique prompt groups per optimizer update.
    # --------------------------------------------------------

    bad_groups = []
    cross_groups = []

    for group, start in enumerate(
        range(0, 1440, 3),
        start=1,
    ):
        chunk = v2[start:start + 3]

        tasks = [r["task"] for r in chunk]
        ls = [int(r["level"]) for r in chunk]

        if len(set(tasks)) != 3:
            bad_groups.append((group, tasks))

        if len(set(ls)) > 1:
            cross_groups.append(group)

    if bad_groups:
        fail(f"duplicate-task optimizer groups: {bad_groups}")

    if cross_groups != EXPECTED_CROSS_LEVEL_GROUPS:
        fail(
            f"unexpected cross-level groups: {cross_groups}"
        )

    # --------------------------------------------------------
    # Pool sidecar provenance.
    # --------------------------------------------------------

    pm = json.loads(
        V2_POOL_MANIFEST.read_text(encoding="utf-8")
    )

    if pm.get("data_sha256") != EXPECTED_V2:
        fail("v2 pool sidecar has wrong data_sha256")

    if pm.get("n_rows") != 1440:
        fail("v2 pool sidecar has wrong row count")

    if pm.get("split_namespace") != "RL_TRAIN_STAGE_1":
        fail("wrong split namespace")

    if pm.get("curriculum_stage") != "stage1":
        fail("wrong curriculum stage")

    if pm.get("frozen_taskset_hash") != EXPECTED_FROZEN_HASH:
        fail("wrong frozen-taskset provenance")

    # --------------------------------------------------------
    # Config invariants.
    # --------------------------------------------------------

    cfg = yaml.safe_load(
        config_path.read_text(encoding="utf-8")
    )

    if cfg["data"]["train"] != (
        "data/rl/curriculum/stage1_levelmixed_v2.jsonl"
    ):
        fail("config does not point to levelmixed v2")

    tr = cfg["training"]

    expected = {
        "shuffle_dataset": False,
        "max_steps": 480,
        "warmup_steps": 40,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": 5.0e-6,
    }

    for key, value in expected.items():
        if tr.get(key) != value:
            fail(
                f"config {key}={tr.get(key)!r}, "
                f"expected {value!r}"
            )

    if cfg["generation"]["num_generations"] != 8:
        fail("num_generations must remain 8")

    if tr["output_dir"] != (
        "outputs/rl/qwen3_14b_stage1_levelmixed_v2"
    ):
        fail("replacement output_dir is not isolated")

    print("=" * 72)
    print("LEVELMIXED V2 SPECIFIC PREFLIGHT: PASS")
    print("=" * 72)
    print("v2 sha:", EXPECTED_V2)
    print("rows: 1440")
    print("levels: monotonic L1 -> L15")
    print("optimizer groups: 480")
    print("task-distinct groups: 480/480")
    print("cross-level groups:", cross_groups)
    print("canonical payload multiset: identical")
    print("frozen taskset provenance:", EXPECTED_FROZEN_HASH)
    print("replacement output isolated:", tr["output_dir"])

    # --------------------------------------------------------
    # Then invoke the existing repository preflight unchanged.
    # --------------------------------------------------------

    cmd = [
        sys.executable,
        "-m",
        "rl.preflight",
        "--config",
        str(config_path),
        "--world-size",
        "3",
    ]

    if args.skip_token_audit:
        cmd.append("--skip-token-audit")

    print()
    print("Running existing ArgGYM preflight...")
    print(" ".join(cmd))

    subprocess.run(
        cmd,
        cwd=ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
