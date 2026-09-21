"""Generate deterministic, contamination-checked ArgGYM RL datasets.

Two modes are supported:

1. ``smoke``: a tiny all-task set for plumbing validation.
2. ``curriculum``: the article training design with three disjoint 1,440-row
   stages plus a 720-row dev set.

The generator never reads model outputs and never writes to ``data/taskset.jsonl``.
Every accepted example is checked against the canonical frozen suite and all
previously generated splits by seed coordinate and problem hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import yaml

import arggym
from arggym.core import spec as spec_module
from evals import taskset as frozen_taskset
from evals.prompt import Elicitation, compose

from rl.common import (
    canonical_json,
    public_row_hash,
    question_hash,
    row_record_hash,
    sha256_text,
    theory_hash,
    write_jsonl,
)

TEMPLATE = "xml_tags"
ELICITATION = Elicitation(
    name="cot",
    system="Work the problem out step by step, then answer in the format the question asks for.",
)
PROMPT_PROTOCOL = "arggym-evals-xml-cot-v1"
POOL_SCHEMA = 2
CANONICAL_FROZEN_HASH = "121f452f2744ef0a6022c4731097b6b1"
MAX_SEED = 2_147_483_647


def git_sha() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def prompt_messages(row: Dict[str, Any]) -> List[Dict[str, str]]:
    system, user = compose(row, TEMPLATE, ELICITATION)
    out: List[Dict[str, str]] = []
    if system:
        out.append({"role": "system", "content": system})
    out.append({"role": "user", "content": user})
    return out


def _prompt_hash(messages: Sequence[Dict[str, str]]) -> str:
    return sha256_text(canonical_json(list(messages)))


def _frozen_index(path: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, set]]:
    manifest, rows = frozen_taskset.load(path, verify=True)
    if len(rows) != 1440:
        raise RuntimeError(f"{path} has {len(rows)} rows; expected canonical 1,440-row suite")
    if manifest.get("taskset_hash") != CANONICAL_FROZEN_HASH:
        raise RuntimeError(
            f"{path} taskset_hash={manifest.get('taskset_hash')}; "
            f"expected {CANONICAL_FROZEN_HASH}"
        )
    index = {
        "question": {question_hash(r) for r in rows},
        "public": {public_row_hash(r) for r in rows},
        "theory": {h for r in rows if (h := theory_hash(r)) is not None},
        "coordinate": {
            (
                r["task"],
                int(r["metadata"]["level"]),
                r["metadata"]["ordering"],
                int(r["metadata"]["seed"]),
            )
            for r in rows
        },
    }
    return manifest, rows, index


def _pool_row(
    row: Dict[str, Any],
    split: str,
    report: Any,
    *,
    curriculum_stage: Optional[str] = None,
    difficulty_band: Optional[str] = None,
) -> Dict[str, Any]:
    messages = prompt_messages(row)
    return {
        "pool_schema": POOL_SCHEMA,
        "split_namespace": split,
        "curriculum_stage": curriculum_stage,
        "difficulty_band": difficulty_band,
        "id": row["id"],
        "task": row["task"],
        "level": int(row["metadata"]["level"]),
        "ordering": row["metadata"]["ordering"],
        "seed": int(row["metadata"]["seed"]),
        "prompt": messages,
        # Hidden trainer metadata; only prompt is rendered to the policy.
        "row_json": canonical_json(row),
        "template_name": TEMPLATE,
        "prompt_protocol": PROMPT_PROTOCOL,
        "prompt_hash": _prompt_hash(messages),
        "question_hash": question_hash(row),
        "public_row_hash": public_row_hash(row),
        "theory_hash": theory_hash(row),
        "row_hash": row_record_hash(row),
        "build_calls": int(report.calls),
        "build_rejections": dict(report.reasons),
    }


def _domain_seed(
    master: str,
    namespace: str,
    task: str,
    level: int,
    ordering: str,
    candidate_index: int,
) -> int:
    material = f"{master}|{namespace}|{task}|{level}|{ordering}|{candidate_index}"
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    # Keep the seed in a conservative signed-32-bit-compatible range.
    return 1 + (int.from_bytes(digest[:8], "big") % (MAX_SEED - 1))


def _add_pool_to_forbidden(forbidden: Dict[str, set], rows: Iterable[Dict[str, Any]]) -> None:
    for r in rows:
        forbidden["question"].add(r["question_hash"])
        forbidden["public"].add(r["public_row_hash"])
        if r.get("theory_hash") is not None:
            forbidden["theory"].add(r["theory_hash"])
        forbidden["coordinate"].add((r["task"], r["level"], r["ordering"], r["seed"]))


def _band_for_level(level: int, bands: Mapping[str, Sequence[int]]) -> Optional[str]:
    for name, levels in bands.items():
        if level in levels:
            return name
    return None


def _build_quota_pool(
    *,
    quotas: Mapping[Tuple[str, int, str], int],
    split: str,
    master_seed_namespace: str,
    profile: str,
    forbidden: Dict[str, set],
    bands: Mapping[str, Sequence[int]],
    max_candidates_per_item: int,
    curriculum_stage: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    own = {"question": set(), "public": set(), "theory": set(), "coordinate": set()}
    cells: Dict[str, Any] = {}

    for (task, level, ordering), target in sorted(quotas.items()):
        if target <= 0:
            continue
        accepted = 0
        candidate_index = 0
        skipped: Counter[str] = Counter()
        build_rejections: Counter[str] = Counter()
        first_seed: Optional[int] = None
        last_seed: Optional[int] = None
        hard_limit = max(target * max_candidates_per_item, target)

        while accepted < target and candidate_index < hard_limit:
            seed = _domain_seed(
                master_seed_namespace,
                split,
                task,
                level,
                ordering,
                candidate_index,
            )
            candidate_index += 1
            if first_seed is None:
                first_seed = seed
            last_seed = seed
            coordinate = (task, level, ordering, seed)
            if coordinate in forbidden["coordinate"] or coordinate in own["coordinate"]:
                skipped["seed_coordinate_collision"] += 1
                continue

            ds = arggym.create(task, level, ordering, size=1, seed=seed, profile=profile)
            entry, report = ds.build_at(0)
            build_rejections.update(report.reasons)
            if entry is None:
                skipped["builder_rejected"] += 1
                continue

            band = _band_for_level(level, bands)
            record = _pool_row(
                entry,
                split,
                report,
                curriculum_stage=curriculum_stage,
                difficulty_band=band,
            )
            hashes = {
                "question": record["question_hash"],
                "public": record["public_row_hash"],
                "theory": record["theory_hash"],
            }

            collision = None
            for kind in ("question", "public"):
                if hashes[kind] in forbidden[kind]:
                    collision = f"forbidden_{kind}_hash"
                    break
                if hashes[kind] in own[kind]:
                    collision = f"duplicate_{kind}_hash"
                    break
            if collision is None and hashes["theory"] is not None:
                # Theory reuse inside one split is allowed: the two counter-
                # argument task variants intentionally share a generator. Theory
                # reuse across frozen/train/dev splits is not allowed.
                if hashes["theory"] in forbidden["theory"]:
                    collision = "forbidden_theory_hash"
            if collision is not None:
                skipped[collision] += 1
                continue

            own["question"].add(hashes["question"])
            own["public"].add(hashes["public"])
            if hashes["theory"] is not None:
                own["theory"].add(hashes["theory"])
            own["coordinate"].add(coordinate)
            rows.append(record)
            accepted += 1

        key = f"{task}|L{level}|{ordering}"
        cells[key] = {
            "target": target,
            "accepted": accepted,
            "candidates_examined": candidate_index,
            "first_seed": first_seed,
            "last_seed": last_seed,
            "skipped": dict(skipped),
            "build_rejections": dict(build_rejections),
        }
        if accepted != target:
            raise RuntimeError(
                f"{split}: {key} produced {accepted}/{target} accepted rows after "
                f"{candidate_index} deterministic candidates; refusing a short cell"
            )

    counts = {
        "task": dict(Counter(r["task"] for r in rows)),
        "ordering": dict(Counter(r["ordering"] for r in rows)),
        "level": {str(k): v for k, v in sorted(Counter(r["level"] for r in rows).items())},
        "band": dict(Counter(r.get("difficulty_band") for r in rows)),
    }
    manifest = {
        "pool_schema": POOL_SCHEMA,
        "split_namespace": split,
        "curriculum_stage": curriculum_stage,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "arggym": arggym.__version__,
        "git_sha": git_sha(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "profile": profile,
        "master_seed_namespace": master_seed_namespace,
        "seed_scheme": "sha256-domain-separated-v1",
        "prompt_protocol": PROMPT_PROTOCOL,
        "template": TEMPLATE,
        "elicitation": asdict(ELICITATION),
        "n_rows": len(rows),
        "counts": counts,
        "cells": cells,
    }
    return rows, manifest


def _write_pool(
    path: Path,
    rows: List[Dict[str, Any]],
    manifest: Dict[str, Any],
    frozen_manifest: Dict[str, Any],
) -> None:
    write_jsonl(path, rows)
    data_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    final = {
        **manifest,
        "data_sha256": data_sha,
        "frozen_taskset_hash": frozen_manifest.get("taskset_hash"),
    }
    mp = path.with_suffix(path.suffix + ".manifest.json")
    mp.write_text(json.dumps(final, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_standard_spec(path: str) -> arggym.TasksetSpec:
    spec = arggym.load_spec(path)
    spec_module.check_versions(spec)
    if len(spec.tasks) != 12 or len(spec.levels) != 15 or len(spec.orderings) != 4:
        raise RuntimeError(
            "article curriculum assumes the canonical 12 tasks x 15 levels x 4 orderings grid"
        )
    if sorted(spec.levels) != list(range(1, 16)):
        raise RuntimeError("article curriculum requires levels 1..15")
    if spec.profile != "FULL":
        raise RuntimeError("article curriculum requires profile FULL")
    return spec


def _stratum_band_counts(stage_index: int, task_index: int, ordering_index: int) -> Dict[str, int]:
    if stage_index == 0:
        # 24 checkerboard strata get (15,11,4), the other 24 get (15,10,5).
        # Every task has 2/2 and every ordering has 6/6, so task/order marginals
        # remain exact despite the 35/15 integer split.
        a = (task_index + ordering_index) % 2 == 0
        return {"easy": 15, "medium": 11 if a else 10, "hard": 4 if a else 5}
    if stage_index == 1:
        return {"easy": 12, "medium": 9, "hard": 9}
    if stage_index == 2:
        return {"easy": 6, "medium": 9, "hard": 15}
    raise ValueError(stage_index)


def _stage_quotas(
    spec: arggym.TasksetSpec,
    stage_index: int,
    bands: Mapping[str, Sequence[int]],
) -> Dict[Tuple[str, int, str], int]:
    quotas: Dict[Tuple[str, int, str], int] = defaultdict(int)
    # A rolling remainder cursor makes each band's global level counts differ by
    # at most one while every task x ordering stratum keeps exactly 30 rows.
    cursors = {name: (stage_index + i) % len(levels) for i, (name, levels) in enumerate(bands.items())}

    for ti, task in enumerate(spec.tasks):
        for oi, ordering in enumerate(spec.orderings):
            band_counts = _stratum_band_counts(stage_index, ti, oi)
            if sum(band_counts.values()) != 30:
                raise AssertionError("task-ordering stratum must total 30")
            for band_name, q in band_counts.items():
                levels = list(bands[band_name])
                base, rem = divmod(q, len(levels))
                for level in levels:
                    quotas[(task, int(level), ordering)] += base
                cursor = cursors[band_name]
                for k in range(rem):
                    level = levels[(cursor + k) % len(levels)]
                    quotas[(task, int(level), ordering)] += 1
                cursors[band_name] = (cursor + rem) % len(levels)
    return dict(quotas)


def _verify_stage_plan(
    spec: arggym.TasksetSpec,
    quotas: Mapping[Tuple[str, int, str], int],
    expected_band_counts: Mapping[str, int],
    bands: Mapping[str, Sequence[int]],
) -> None:
    total = sum(quotas.values())
    if total != 1440:
        raise AssertionError(f"stage plan has {total} rows, expected 1440")

    for task in spec.tasks:
        got = sum(v for (t, _l, _o), v in quotas.items() if t == task)
        if got != 120:
            raise AssertionError(f"{task}: {got} rows, expected 120")
    for ordering in spec.orderings:
        got = sum(v for (_t, _l, o), v in quotas.items() if o == ordering)
        if got != 360:
            raise AssertionError(f"{ordering}: {got} rows, expected 360")
    for task in spec.tasks:
        for ordering in spec.orderings:
            got = sum(v for (t, _l, o), v in quotas.items() if t == task and o == ordering)
            if got != 30:
                raise AssertionError(f"{task}/{ordering}: {got} rows, expected 30")

    for band_name, levels in bands.items():
        got = sum(v for (_t, l, _o), v in quotas.items() if l in levels)
        if got != int(expected_band_counts[band_name]):
            raise AssertionError(
                f"band {band_name}: {got}, expected {expected_band_counts[band_name]}"
            )
        level_counts = [sum(v for (_t, ll, _o), v in quotas.items() if ll == l) for l in levels]
        if max(level_counts) - min(level_counts) > 1:
            raise AssertionError(f"band {band_name} level imbalance: {level_counts}")

    for task in spec.tasks:
        for ordering in spec.orderings:
            for levels in bands.values():
                counts = [quotas.get((task, int(l), ordering), 0) for l in levels]
                if max(counts) - min(counts) > 1:
                    raise AssertionError(
                        f"within-stratum level imbalance {task}/{ordering}/{levels}: {counts}"
                    )


def _dev_quotas(spec: arggym.TasksetSpec) -> Dict[Tuple[str, int, str], int]:
    return {(t, int(l), o): 1 for t in spec.tasks for l in spec.levels for o in spec.orderings}


def _smoke_quotas(spec: arggym.TasksetSpec, *, dev: bool) -> Dict[Tuple[str, int, str], int]:
    # Explicit small design: every task is present; across tasks we cover all
    # three difficulty bands and all four orderings. Train gets two coordinates
    # per task (24 rows), dev gets one independent coordinate per task (12 rows).
    easy = [3, 4, 5]
    medium = [8, 9, 10]
    hard = [13, 14, 15]
    orders = list(spec.orderings)
    quotas: Dict[Tuple[str, int, str], int] = {}
    for i, task in enumerate(spec.tasks):
        if dev:
            bands = [easy, medium, hard]
            levels = bands[i % 3]
            level = levels[(i // 3) % len(levels)]
            ordering = orders[(i + 1) % len(orders)]
            quotas[(task, level, ordering)] = 1
        else:
            band_a = [easy, medium, hard][i % 3]
            band_b = [easy, medium, hard][(i + 1) % 3]
            level_a = band_a[(i // 3) % len(band_a)]
            level_b = band_b[((i // 3) + 1) % len(band_b)]
            ordering_a = orders[i % len(orders)]
            ordering_b = orders[(i + 2) % len(orders)]
            quotas[(task, level_a, ordering_a)] = 1
            quotas[(task, level_b, ordering_b)] = 1
    return quotas


def _load_curriculum(path: str) -> Dict[str, Any]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeError(f"{path} must contain a mapping")
    return raw


def generate_smoke(a: argparse.Namespace) -> int:
    spec = _load_standard_spec(a.spec)
    frozen_manifest, _rows, forbidden = _frozen_index(a.frozen)
    bands = {"easy": [1, 2, 3, 4, 5], "medium": [6, 7, 8, 9, 10], "hard": [11, 12, 13, 14, 15]}
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train, tm = _build_quota_pool(
        quotas=_smoke_quotas(spec, dev=False),
        split="RL_SMOKE_TRAIN",
        master_seed_namespace=a.master_seed_namespace,
        profile=spec.profile,
        forbidden=forbidden,
        bands=bands,
        max_candidates_per_item=a.max_candidates_per_item,
        curriculum_stage="smoke",
    )
    _add_pool_to_forbidden(forbidden, train)
    dev, dm = _build_quota_pool(
        quotas=_smoke_quotas(spec, dev=True),
        split="RL_SMOKE_DEV",
        master_seed_namespace=a.master_seed_namespace,
        profile=spec.profile,
        forbidden=forbidden,
        bands=bands,
        max_candidates_per_item=a.max_candidates_per_item,
        curriculum_stage="smoke_dev",
    )
    _write_pool(out_dir / "train.jsonl", train, tm, frozen_manifest)
    _write_pool(out_dir / "dev.jsonl", dev, dm, frozen_manifest)
    print(f"wrote smoke train: {len(train)} -> {out_dir / 'train.jsonl'}")
    print(f"wrote smoke dev:   {len(dev)} -> {out_dir / 'dev.jsonl'}")
    return 0


def generate_curriculum(a: argparse.Namespace) -> int:
    spec = _load_standard_spec(a.spec)
    cfg = _load_curriculum(a.curriculum)
    bands = {k: [int(x) for x in v] for k, v in cfg["bands"].items()}
    if bands != {"easy": [1, 2, 3, 4, 5], "medium": [6, 7, 8, 9, 10], "hard": [11, 12, 13, 14, 15]}:
        raise RuntimeError("article curriculum bands must be L1-5 / L6-10 / L11-15")

    frozen_manifest, _rows, forbidden = _frozen_index(a.frozen)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    master = str(cfg["master_seed_namespace"])

    for stage_index, stage_cfg in enumerate(cfg["stages"]):
        quotas = _stage_quotas(spec, stage_index, bands)
        _verify_stage_plan(spec, quotas, stage_cfg["band_counts"], bands)
        rows, manifest = _build_quota_pool(
            quotas=quotas,
            split=str(stage_cfg["namespace"]),
            master_seed_namespace=master,
            profile=str(cfg.get("profile", spec.profile)),
            forbidden=forbidden,
            bands=bands,
            max_candidates_per_item=a.max_candidates_per_item,
            curriculum_stage=str(stage_cfg["name"]),
        )
        if len(rows) != int(stage_cfg["n_rows"]):
            raise AssertionError(f"{stage_cfg['name']}: {len(rows)} rows")
        manifest["curriculum"] = {
            "stage_index": stage_index + 1,
            "stage_name": stage_cfg["name"],
            "band_counts": stage_cfg["band_counts"],
            "task_ordering_rows": 30,
            "task_rows": 120,
            "ordering_rows": 360,
        }
        path = out_dir / f"{stage_cfg['name']}.jsonl"
        _write_pool(path, rows, manifest, frozen_manifest)
        _add_pool_to_forbidden(forbidden, rows)
        print(f"wrote {stage_cfg['name']}: {len(rows)} -> {path}")

    dev_cfg = cfg["dev"]
    dev, dm = _build_quota_pool(
        quotas=_dev_quotas(spec),
        split=str(dev_cfg["namespace"]),
        master_seed_namespace=master,
        profile=str(cfg.get("profile", spec.profile)),
        forbidden=forbidden,
        bands=bands,
        max_candidates_per_item=a.max_candidates_per_item,
        curriculum_stage="dev",
    )
    if len(dev) != int(dev_cfg["n_rows"]):
        raise AssertionError(f"dev: {len(dev)} rows")
    _write_pool(out_dir / "dev.jsonl", dev, dm, frozen_manifest)
    print(f"wrote dev: {len(dev)} -> {out_dir / 'dev.jsonl'}")
    print(f"canonical frozen suite untouched: {a.frozen} ({CANONICAL_FROZEN_HASH})")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--spec", default="tasksets/standard.yaml")
    common.add_argument("--frozen", default="data/taskset.jsonl")
    common.add_argument("--max-candidates-per-item", type=int, default=80)

    s = sub.add_parser("smoke", parents=[common])
    s.add_argument("--out-dir", default="data/rl/smoke")
    s.add_argument("--master-seed-namespace", default="arggym-rl-smoke-v1")

    c = sub.add_parser("curriculum", parents=[common])
    c.add_argument("--curriculum", default="configs/rl/curriculum.yaml")
    c.add_argument("--out-dir", default="data/rl/curriculum")

    a = p.parse_args(argv)
    if a.mode == "smoke":
        return generate_smoke(a)
    if a.mode == "curriculum":
        return generate_curriculum(a)
    raise AssertionError(a.mode)


if __name__ == "__main__":
    raise SystemExit(main())
