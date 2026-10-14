"""The training pool and the dev split, generated fresh and kept apart from evaluation.

    uv run python -m rl.pool rl/configs/qwen3-4b.yaml

Both are frozen with the same `arggym.core.freeze` the evaluation tasksets use,
so every row carries the prompt and scoring versions it was built under and
every reference answer has been checked to score 1.0. What differs is the seed
range: each split scans its own window, and the windows of the training pool,
the dev split and both eval specs (`data/taskset.yaml`, `data/taskset-lite.yaml`)
must not overlap, or `check_windows` refuses before anything is generated.

Distinct seeds are not distinct items: a level-1 theory has few moving parts, and
two seeds can draw the same one. So every row whose question, or whose theory, is
also in an eval taskset is dropped, and the dev split also drops whatever the
training pool holds. The manifest says how many went.

A training row is what TRL reads: the prompt as chat messages, composed by the
eval harness's own code (`evals.prompt.compose`, `evals.client.ChatClient.body`),
and the whole frozen row as JSON for the reward to score against.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import yaml

import arggym
from arggym.core.freeze import freeze, taskset_hash, usable_cores
from arggym.core.spec import ALL_ORDERINGS, SeedPolicy, TasksetSpec
from evals import taskset
from evals.client import ChatClient, Endpoint
from evals.prompt import Elicitation, compose
from rl import config as rl_config

ROOT = Path(__file__).resolve().parents[1]
EVAL_SPECS = ("data/taskset.yaml", "data/taskset-lite.yaml")
EVAL_TASKSETS = ("data/taskset.jsonl", "data/taskset-lite.jsonl")


# ---------------------------------------------------------------- prompts

def elicitation(name: str) -> Elicitation:
    """The elicitation the eval harness would run under, read from its own config file."""
    with open(ROOT / "evals/conf/elicitation" / f"{name}.yaml") as f:
        return Elicitation(**yaml.safe_load(f))


def messages(row: Dict[str, Any], template: str, elicit: Elicitation) -> List[Dict[str, str]]:
    """The chat messages the eval harness sends for this row, built by the harness's code."""
    system, user = compose(row, template, elicit)
    return ChatClient(Endpoint(model="-")).body(system, user)["messages"]


def trl_rows(rows: Iterable[Dict[str, Any]], template: str,
             elicit: Elicitation) -> List[Dict[str, Any]]:
    return [{"prompt": messages(r, template, elicit), "row_json": json.dumps(r),
             "id": r["id"], "task": r["task"], "level": r["metadata"]["level"]}
            for r in rows]


# ---------------------------------------------------------------- identity

def question_key(row: Dict[str, Any]) -> str:
    return hashlib.sha256(row["question"].encode()).hexdigest()


def theory_key(row: Dict[str, Any]) -> str:
    """The theory and what is asked of it, with the task name and the ordering left out.

    Two rows that differ only in which ordering the question names still share
    every fact a model could memorise, so the ordering does not make them
    different items. `formalization` carries its theory in `state` alone.
    """
    m = row["metadata"]
    content = {k: m.get(k) for k in ("base_ops", "pert_ops", "state")}
    return hashlib.sha256(json.dumps(content, sort_keys=True, default=str).encode()).hexdigest()


def keys(rows: Iterable[Dict[str, Any]]) -> Tuple[set, set]:
    rows = list(rows)
    return {question_key(r) for r in rows}, {theory_key(r) for r in rows}


def eval_rows() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for path in EVAL_TASKSETS:
        out.extend(taskset.load(str(ROOT / path))[1])
    return out


# ---------------------------------------------------------------- specs

def split_spec(cfg: Dict[str, Any], split: str) -> TasksetSpec:
    d = cfg["data"]
    s = d[split]
    tasks = tuple(arggym.task_names()) if d["tasks"] == "all" else tuple(d["tasks"])
    return TasksetSpec(
        tasks=tasks, levels=tuple(d["levels"]), orderings=ALL_ORDERINGS,
        seeds=SeedPolicy(start=s["seed_start"], take=s["take"], scan_limit=s["scan_limit"]),
        # A thin cell is a finding for a benchmark and a non-event for a training
        # pool, so only the seed guard stays on, at the eval specs' value.
        min_acceptance=0.3, min_build_acceptance=0.0)


def window(spec: TasksetSpec) -> range:
    return range(spec.seeds.start, spec.seeds.start + spec.seeds.scan_limit)


def check_windows(cfg: Dict[str, Any]) -> None:
    """Refuse seed windows that overlap, before a single item is built."""
    named = [(p, arggym.load_spec(str(ROOT / p))) for p in EVAL_SPECS]
    named += [(f"data.{s}", split_spec(cfg, s)) for s in ("train", "dev")]
    for i, (a, sa) in enumerate(named):
        for b, sb in named[i + 1:]:
            if a in EVAL_SPECS and b in EVAL_SPECS:
                continue  # the lite taskset is a subset of the standard one by design
            wa, wb = window(sa), window(sb)
            if max(wa.start, wb.start) < min(wa.stop, wb.stop):
                raise SystemExit(
                    f"seed windows overlap: {a} scans {wa.start}..{wa.stop - 1} and {b} "
                    f"scans {wb.start}..{wb.stop - 1}. Move data.*.seed_start apart.")


# ---------------------------------------------------------------- building

def _freeze(spec: TasksetSpec, workers: int) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "split.jsonl")
        freeze(spec, path, verbose=False, workers=workers)
        return taskset.load(path)


def drop_seen(rows: Sequence[Dict[str, Any]], seen: Tuple[set, set]
              ) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    questions, theories = seen
    kept, dropped = [], defaultdict(int)
    for r in rows:
        if question_key(r) in questions:
            dropped["question"] += 1
        elif theory_key(r) in theories:
            dropped["theory"] += 1
        else:
            kept.append(r)
    return kept, dict(dropped)


def pick_dev(rows: Sequence[Dict[str, Any]], per_task_level: int) -> List[Dict[str, Any]]:
    """`per_task_level` rows of each (task, level), rotating through the orderings.

    The dev split is evaluated every few steps, so it is sized for a selection
    signal, not for a per-cell curve. Rotating the ordering by level keeps all
    four orderings in it without paying for four rows a level.
    """
    by_cell: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_cell[(r["task"], r["metadata"]["level"])].append(r)
    out = []
    for (task, level), group in sorted(by_cell.items()):
        group.sort(key=lambda r: (ALL_ORDERINGS.index(r["metadata"]["ordering"]),
                                  r["metadata"]["seed"]))
        shift = level % len(group)
        out.extend((group[shift:] + group[:shift])[:per_task_level])
    return out


def build(cfg: Dict[str, Any], workers: int) -> Dict[str, Any]:
    """Freeze both splits, drop what evaluation has seen, write them under `pool_dir`."""
    check_windows(cfg)
    seen = keys(eval_rows())
    out_dir = pool_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Any] = {"data": cfg["data"]}
    for split in ("train", "dev"):
        manifest, rows = _freeze(split_spec(cfg, split), workers)
        rows, dropped = drop_seen(rows, seen)
        if split == "dev":
            rows = pick_dev(rows, cfg["data"]["dev"]["per_task_level"])
        # The dev split must be new to the policy as well as to evaluation.
        if split == "train":
            train_seen = keys(rows)
            seen = (seen[0] | train_seen[0], seen[1] | train_seen[1])
        manifest.update(taskset_hash=taskset_hash(rows), n_items=len(rows),
                        dropped_as_seen=dropped, split=split)
        with open(out_dir / f"{split}.jsonl", "w") as f:
            f.write(json.dumps({"__manifest__": manifest}) + "\n")
            for r in rows:
                f.write(json.dumps(r, default=str) + "\n")
        summary[split] = {"n_items": len(rows), "dropped_as_seen": dropped,
                          "taskset_hash": manifest["taskset_hash"]}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def pool_dir(cfg: Dict[str, Any]) -> Path:
    """Named by a hash of `data`, so a run reuses a pool built from the same settings."""
    digest = hashlib.sha256(json.dumps(cfg["data"], sort_keys=True).encode()).hexdigest()[:12]
    return ROOT / cfg["output_root"] / "pools" / digest


def load_split(cfg: Dict[str, Any], split: str) -> List[Dict[str, Any]]:
    path = pool_dir(cfg) / f"{split}.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} is missing; build it with `python -m rl.pool <config>`")
    return taskset.load(str(path))[1]


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config")
    p.add_argument("overrides", nargs="*")
    p.add_argument("--workers", type=int, default=None)
    a = p.parse_args(argv)
    cfg = rl_config.load(a.config, a.overrides)
    if (pool_dir(cfg) / "summary.json").exists():
        print(f"pool exists: {pool_dir(cfg)}")
        return 0
    summary = build(cfg, a.workers or min(64, usable_cores()))
    print(json.dumps(summary, indent=2))
    print(f"-> {pool_dir(cfg)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
