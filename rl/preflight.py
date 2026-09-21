"""Preflight checks for ArgGYM GRPO runs.

The goal is to fail before allocating hours of H200 time if any provenance,
data, curriculum, context-window, or batch invariant is wrong.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import yaml

from evals import taskset as frozen_taskset
from rl.common import iter_jsonl, public_row_hash, question_hash, theory_hash

CANONICAL_FROZEN_HASH = "121f452f2744ef0a6022c4731097b6b1"
SUPPORTED_BASELINE_REVISIONS = {
    "Qwen/Qwen3-14B":
        "40c069824f4251a91eefaf281ebe4c544efd3e18",
    "Qwen/Qwen3-8B":
        "b968826d9c46dd6066d109eabc6255188de91218",
    "Qwen/Qwen3-4B":
        "1cfa9a7208912126459214e8b04321603b3df60c",
    "Qwen/Qwen3-1.7B":
        "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
}
BANDS = {
    "easy": set(range(1, 6)),
    "medium": set(range(6, 11)),
    "hard": set(range(11, 16)),
}
EXPECTED_STAGE_BANDS = {
    "stage1": {"easy": 720, "medium": 504, "hard": 216},
    "stage2": {"easy": 576, "medium": 432, "hard": 432},
    "stage3": {"easy": 288, "medium": 432, "hard": 720},
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _pool_manifest(data_path: Path) -> Dict[str, Any]:
    mp = data_path.with_suffix(data_path.suffix + ".manifest.json")
    if not data_path.is_file():
        raise SystemExit(f"missing RL pool: {data_path}")
    if not mp.is_file():
        raise SystemExit(f"missing RL pool manifest: {mp}")
    m = json.loads(mp.read_text(encoding="utf-8"))
    got = _sha256(data_path)
    if got != m.get("data_sha256"):
        raise SystemExit(
            f"{data_path} SHA-256 is {got}, manifest records {m.get('data_sha256')}; "
            "the generated pool changed after contamination checking"
        )
    n = sum(1 for line in data_path.open(encoding="utf-8") if line.strip())
    if n != int(m.get("n_rows", -1)):
        raise SystemExit(f"{data_path} has {n} rows, manifest records {m.get('n_rows')}")
    return m


def _verify_local_model(model_path: str, revision: str) -> None:
    p = Path(model_path)
    if not p.exists():
        raise SystemExit(f"ARGGYM_MODEL_PATH does not exist: {p}")
    resolved = p.resolve()
    parts = resolved.parts
    if "snapshots" in parts:
        i = parts.index("snapshots")
        if i + 1 >= len(parts):
            raise SystemExit(f"malformed Hugging Face snapshot path: {resolved}")
        snapshot = parts[i + 1]
        if snapshot != revision:
            raise SystemExit(
                f"local model snapshot is {snapshot}, config pins {revision}; "
                "training and baseline would not use the same Qwen revision"
            )
    if not (p / "config.json").is_file():
        raise SystemExit(f"ARGGYM_MODEL_PATH lacks config.json: {p}")


def _band(level: int) -> str:
    for name, levels in BANDS.items():
        if level in levels:
            return name
    raise SystemExit(f"unexpected level {level}")


def _verify_main_stage(path: Path, stage_name: str) -> None:
    rows = list(iter_jsonl(path))
    if len(rows) != 1440:
        raise SystemExit(f"{path}: main stage must have 1440 rows, got {len(rows)}")
    if any(r.get("curriculum_stage") != stage_name for r in rows):
        raise SystemExit(f"{path}: curriculum_stage field is not uniformly {stage_name!r}")

    tasks = Counter(r["task"] for r in rows)
    orderings = Counter(r["ordering"] for r in rows)
    task_order = Counter((r["task"], r["ordering"]) for r in rows)
    levels = Counter(int(r["level"]) for r in rows)
    bands = Counter(_band(int(r["level"])) for r in rows)

    if len(tasks) != 12 or set(tasks.values()) != {120}:
        raise SystemExit(f"{path}: each of 12 tasks must have exactly 120 rows: {dict(tasks)}")
    if len(orderings) != 4 or set(orderings.values()) != {360}:
        raise SystemExit(f"{path}: each of 4 orderings must have exactly 360 rows: {dict(orderings)}")
    if len(task_order) != 48 or set(task_order.values()) != {30}:
        raise SystemExit(f"{path}: each task x ordering stratum must have 30 rows")
    if dict(bands) != EXPECTED_STAGE_BANDS[stage_name]:
        raise SystemExit(
            f"{path}: difficulty-band counts {dict(bands)} != {EXPECTED_STAGE_BANDS[stage_name]}"
        )
    for band_name, band_levels in BANDS.items():
        counts = [levels[l] for l in sorted(band_levels)]
        if max(counts) - min(counts) > 1:
            raise SystemExit(f"{path}: {band_name} level imbalance {counts}")

    # Within every task x ordering stratum, counts among the five levels in a
    # difficulty band may differ only by one.
    grouped: Dict[tuple[str, str, str], Counter[int]] = defaultdict(Counter)
    for r in rows:
        grouped[(r["task"], r["ordering"], _band(int(r["level"])))][int(r["level"])] += 1
    for (task, ordering, band_name), c in grouped.items():
        counts = [c[l] for l in sorted(BANDS[band_name])]
        if max(counts) - min(counts) > 1:
            raise SystemExit(
                f"{path}: within-stratum imbalance {task}/{ordering}/{band_name}: {counts}"
            )


def _verify_pairwise_isolation(paths: Sequence[Path], frozen_rows: Sequence[Dict[str, Any]]) -> None:
    seen_question = {question_hash(r): "FROZEN" for r in frozen_rows}
    seen_public = {public_row_hash(r): "FROZEN" for r in frozen_rows}
    seen_theory = {h: "FROZEN" for r in frozen_rows if (h := theory_hash(r)) is not None}
    seen_coord = {
        (r["task"], int(r["metadata"]["level"]), r["metadata"]["ordering"], int(r["metadata"]["seed"])): "FROZEN"
        for r in frozen_rows
    }

    for path in paths:
        label = str(path)
        # Theory reuse within one split is allowed for task variants sharing a
        # generator, but theory reuse across different splits is forbidden.
        local_theory = set()
        for r in iter_jsonl(path):
            for field, seen in (("question_hash", seen_question), ("public_row_hash", seen_public)):
                h = r[field]
                if h in seen:
                    raise SystemExit(f"{label}: {field} collision with {seen[h]}")
                seen[h] = label
            h = r.get("theory_hash")
            if h is not None and h not in local_theory:
                if h in seen_theory:
                    raise SystemExit(f"{label}: theory_hash collision with {seen_theory[h]}")
                local_theory.add(h)
            coord = (r["task"], int(r["level"]), r["ordering"], int(r["seed"]))
            if coord in seen_coord:
                raise SystemExit(f"{label}: seed-coordinate collision with {seen_coord[coord]}: {coord}")
            seen_coord[coord] = label
        for h in local_theory:
            seen_theory[h] = label


def _token_audit(paths: Sequence[Path], model_path: str, max_prompt_length: int, enable_thinking: bool) -> None:
    try:
        from transformers import AutoTokenizer
    except Exception as exc:
        print(f"  WARNING: tokenizer audit skipped (transformers import failed: {exc})")
        return

    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    max_seen = 0
    max_id = None
    n = 0
    for path in paths:
        for r in iter_jsonl(path):
            encoded = tok.apply_chat_template(
                r["prompt"],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )

            # Transformers versions may return either a plain token-id list
            # or a BatchEncoding/dict-like object. Count actual input_ids,
            # never the number of container fields.
            if hasattr(encoded, "input_ids"):
                ids = encoded.input_ids
            elif isinstance(encoded, dict):
                ids = encoded["input_ids"]
            else:
                ids = encoded

            # Handle an optional singleton batch dimension defensively.
            if ids and isinstance(ids[0], (list, tuple)):
                if len(ids) != 1:
                    raise SystemExit(
                        f"unexpected batched tokenizer output for prompt {r.get('id')}: "
                        f"batch size={len(ids)}"
                    )
                ids = ids[0]

            length = len(ids)
            n += 1
            if length > max_seen:
                max_seen = length
                max_id = r.get("id")
            if length > max_prompt_length:
                raise SystemExit(
                    f"prompt {r.get('id')} has {length} tokens > max_prompt_length={max_prompt_length}; "
                    "ArgGYM prompts must never be silently truncated"
                )
    print(f"  tokenizer audit: {n} prompts; max={max_seen} tokens ({max_id})")


def _validate_config_fields(cfg: Dict[str, Any]) -> None:
    exp = cfg["experiment"]
    model_name = str(cfg["model"].get("name") or "")
    revision = str(cfg["model"].get("revision") or "")
    baseline_revision = str(exp.get("baseline_model_revision") or "")

    expected_revision = SUPPORTED_BASELINE_REVISIONS.get(model_name)

    if expected_revision is None:
        raise SystemExit(
            f"unsupported RL base model {model_name!r}; "
            "add an explicitly reviewed model/revision pair before training"
        )

    if (
        revision != expected_revision
        or baseline_revision != expected_revision
    ):
        raise SystemExit(
            f"RL must start from the frozen baseline revision for "
            f"{model_name}: {expected_revision}; "
            f"got model={revision}, baseline={baseline_revision}"
        )
    if exp.get("frozen_taskset_hash") != CANONICAL_FROZEN_HASH:
        raise SystemExit("config does not pin the canonical frozen ArgGYM taskset hash")
    if exp.get("kind") in {"main", "main_stage1"} and not bool(exp.get("main_run_unlocked", False)):
        raise SystemExit(
            "MAIN RUN LOCKED: review smoke/pilot first, freeze completion cap/LR, then set "
            "experiment.main_run_unlocked: true"
        )

    gen = cfg["generation"]
    if int(gen["max_model_length"]) < int(gen["max_prompt_length"]) + int(gen["max_completion_length"]):
        raise SystemExit(
            "max_model_length must be >= max_prompt_length + max_completion_length "
            f"({gen['max_model_length']} < {gen['max_prompt_length']} + {gen['max_completion_length']})"
        )
    tr = cfg["training"]
    if not bool(tr.get("bf16", True)) or bool(tr.get("fp16", False)):
        raise SystemExit("Qwen3 ArgGYM RL protocol requires BF16 and fp16=false")
    if tr.get("loss_type") != "dapo":
        raise SystemExit("article protocol currently fixes loss_type=dapo")
    if str(tr.get("scale_rewards")) not in {"none", "False", "false"}:
        raise SystemExit("article protocol fixes scale_rewards=none")
    if bool(tr.get("shuffle_dataset", True)) and exp.get("kind") in {"main", "main_stage1"}:
        raise SystemExit("main curriculum requires shuffle_dataset=false to preserve stage order")


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--frozen", default="data/taskset.jsonl")
    p.add_argument("--world-size", type=int, default=3)
    p.add_argument("--skip-token-audit", action="store_true")
    a = p.parse_args(argv)

    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    _validate_config_fields(cfg)

    frozen_manifest, frozen_rows = frozen_taskset.load(a.frozen, verify=True)
    if len(frozen_rows) != 1440 or frozen_manifest.get("taskset_hash") != CANONICAL_FROZEN_HASH:
        raise SystemExit(
            f"{a.frozen} is not canonical frozen ArgGYM: rows={len(frozen_rows)}, "
            f"hash={frozen_manifest.get('taskset_hash')}"
        )

    exp_kind = cfg["experiment"]["kind"]
    data = cfg["data"]
    manifests: list[tuple[Path, Dict[str, Any]]] = []
    train_paths: list[Path] = []
    dev_path = None

    if exp_kind == "smoke":
        tp = Path(data["train"])
        dp = Path(data["dev"])
        tm = _pool_manifest(tp)
        dm = _pool_manifest(dp)
        if tm.get("split_namespace") != "RL_SMOKE_TRAIN":
            raise SystemExit(f"{tp}: expected RL_SMOKE_TRAIN")
        if dm.get("split_namespace") != "RL_SMOKE_DEV":
            raise SystemExit(f"{dp}: expected RL_SMOKE_DEV")
        if int(tm["n_rows"]) != 24 or int(dm["n_rows"]) != 12:
            raise SystemExit("smoke pool must be exactly 24 train / 12 dev")
        train_paths = [tp]
        dev_path = dp
        manifests = [(tp, tm), (dp, dm)]
    elif exp_kind == "main":
        train_paths = [Path(x) for x in data["train_stages"]]
        if len(train_paths) != 3:
            raise SystemExit("main run requires exactly three training stages")
        for pth, stage_name, namespace in zip(
            train_paths,
            ("stage1", "stage2", "stage3"),
            ("RL_TRAIN_STAGE_1", "RL_TRAIN_STAGE_2", "RL_TRAIN_STAGE_3"),
        ):
            m = _pool_manifest(pth)
            if m.get("split_namespace") != namespace:
                raise SystemExit(f"{pth}: expected namespace {namespace}")
            _verify_main_stage(pth, stage_name)
            manifests.append((pth, m))
        dev_path = Path(data["dev"])
        dm = _pool_manifest(dev_path)
        if dm.get("split_namespace") != "RL_DEV" or int(dm.get("n_rows", -1)) != 720:
            raise SystemExit(f"{dev_path}: expected 720-row RL_DEV")
        manifests.append((dev_path, dm))
    elif exp_kind == "main_stage1":
        tp = Path(data["train"])
        tm = _pool_manifest(tp)
        if tm.get("split_namespace") != "RL_TRAIN_STAGE_1":
            raise SystemExit(f"{tp}: expected namespace RL_TRAIN_STAGE_1")
        if int(tm.get("n_rows", -1)) != 1440:
            raise SystemExit(f"{tp}: Stage 1 must contain exactly 1440 rows")
        _verify_main_stage(tp, "stage1")

        dev_path = Path(data["dev"])
        dm = _pool_manifest(dev_path)
        if dm.get("split_namespace") != "RL_DEV":
            raise SystemExit(f"{dev_path}: expected namespace RL_DEV")
        if int(dm.get("n_rows", -1)) != 720:
            raise SystemExit(f"{dev_path}: expected 720-row RL_DEV")

        train_paths = [tp]
        manifests = [(tp, tm), (dev_path, dm)]
    else:
        raise SystemExit(f"unknown experiment.kind={exp_kind!r}")

    for path, m in manifests:
        if m.get("frozen_taskset_hash") != CANONICAL_FROZEN_HASH:
            raise SystemExit(
                f"{path} was checked against frozen hash {m.get('frozen_taskset_hash')}, "
                f"expected {CANONICAL_FROZEN_HASH}"
            )

    isolation_paths = [*train_paths]
    if dev_path is not None:
        isolation_paths.append(dev_path)
    _verify_pairwise_isolation(isolation_paths, frozen_rows)

    tr = cfg["training"]
    gen = cfg["generation"]
    effective = a.world_size * int(tr["per_device_train_batch_size"]) * int(tr["gradient_accumulation_steps"])
    g = int(gen["num_generations"])
    if effective % g:
        raise SystemExit(f"effective GRPO batch {effective} is not divisible by num_generations={g}")
    prompt_groups_per_step = effective // g

    total_train = sum(int(_pool_manifest(p)["n_rows"]) for p in train_paths)
    max_steps = int(tr["max_steps"])
    nominal_prompt_groups = prompt_groups_per_step * max_steps
    if exp_kind in {"main", "main_stage1"} and nominal_prompt_groups != total_train:
        raise SystemExit(
            f"production run must expose each materialized training prompt exactly once: "
            f"{prompt_groups_per_step} groups/step * {max_steps} steps = {nominal_prompt_groups}, "
            f"but train pool has {total_train} rows"
        )
    if exp_kind == "main":
        for n in (1440, 2880, 4320):
            if n % prompt_groups_per_step:
                raise SystemExit("stage boundary does not align to optimizer-step prompt grouping")
    elif exp_kind == "main_stage1":
        if max_steps != 480:
            raise SystemExit(
                f"Stage-1 production run requires exactly 480 optimizer steps; got {max_steps}"
            )

    revision = cfg["model"]["revision"]
    local = os.environ.get("ARGGYM_MODEL_PATH")
    if local:
        _verify_local_model(local, revision)
        if not a.skip_token_audit:
            audit_paths = [*train_paths]
            if dev_path is not None:
                audit_paths.append(dev_path)
            _token_audit(
                audit_paths,
                local,
                int(gen["max_prompt_length"]),
                bool(gen.get("enable_thinking", True)),
            )
    else:
        print("  WARNING: ARGGYM_MODEL_PATH is unset; local snapshot and tokenizer checks skipped")

    print("ArgGYM GRPO preflight: PASS")
    print(f"  frozen taskset: {CANONICAL_FROZEN_HASH} ({len(frozen_rows)} rows)")
    for path, m in manifests:
        print(f"  pool: {path} -> {m['n_rows']} rows ({m['data_sha256'][:12]}...)")
    print(f"  model revision: {revision}")
    print(f"  precision: bf16={tr.get('bf16')} fp16={tr.get('fp16')}")
    print(f"  context: prompt<={gen['max_prompt_length']} completion<={gen['max_completion_length']} model={gen['max_model_length']}")
    print(f"  effective completion batch: {effective}; generations/group: {g}")
    print(f"  unique prompt groups/optimizer step: {prompt_groups_per_step}")
    print(f"  max_steps={max_steps}; nominal prompt-group exposures={nominal_prompt_groups}")
    if exp_kind == "main":
        print(f"  stage boundaries: step {1440 // prompt_groups_per_step}, {2880 // prompt_groups_per_step}, {4320 // prompt_groups_per_step}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
