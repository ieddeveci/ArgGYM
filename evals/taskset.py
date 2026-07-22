"""Materialise the evaluation grid into a frozen, reusable taskset.

Built once on CPU and shared by every model in the roster, so all models see
byte-identical prompts even if kb.json changes between runs.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Iterable, List, Optional

import hydra
from omegaconf import DictConfig, OmegaConf

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aspic_gym import create_dataset, score_answer  # noqa: E402


def sample_id(task: str, mode: str, level: int, idx: int) -> str:
    return f"{task}__{mode}__L{level:02d}__{idx:03d}"


def cell_seed(base_seed: int, task: str, mode: str, level: int) -> int:
    """Stable per-cell seed.

    Derived from a hash rather than an incrementing counter so that adding or
    removing a task from the grid does not renumber every other cell.
    """
    key = f"{base_seed}|{task}|{mode}|{level}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(key, digest_size=4).digest(), "big")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_sha() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent.parent,
            text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def build_rows(tasks: Iterable[str], modes: Iterable[str], levels: Iterable[int],
               n: int, base_seed: int, validate_gold: bool = True,
               progress=None, stats: Optional[dict] = None) -> List[dict]:
    """Generate every (task, mode, level) cell.

    Because ASPICDataset derives its RNG from idx alone, raising `n` later
    extends each cell instead of reshuffling it: the first `n` items stay
    byte-identical.
    """
    rows: List[dict] = []
    gold_failures: List[tuple] = []
    skips: dict = {}
    for task in tasks:
        for mode in modes:
            with_content = (mode == "content")
            for level in levels:
                seed = cell_seed(base_seed, task, mode, level)
                # Some (task, level) combinations occasionally fail to produce a
                # valid theory at a given index even after the generator's own 60
                # retries. Draw from an oversized dataset and skip those indices
                # deterministically rather than shrinking the cell: the skip is a
                # function of (seed, idx) alone, so the taskset stays reproducible
                # and raising n later still extends the existing prefix.
                budget = max(n * 10, n + 50)
                ds = create_dataset(task, seed=seed, size=budget, level=level,
                                    with_content=with_content)
                got, idx, skipped = 0, 0, 0
                while got < n and idx < budget:
                    try:
                        entry = ds[idx]
                    except RuntimeError:
                        skipped += 1
                        idx += 1
                        continue
                    if validate_gold:
                        s = score_answer(entry["answer"], entry)
                        if s < 1.0:
                            gold_failures.append((task, mode, level, idx, s))
                    rows.append({
                        "sample_id": sample_id(task, mode, level, idx),
                        "task": task,
                        "mode": mode,
                        "level": level,
                        "idx": idx,
                        "cell_seed": seed,
                        "prompt": entry["question"],
                        "entry": entry,
                    })
                    got += 1
                    idx += 1
                if got < n:
                    raise RuntimeError(
                        f"cell {task}/{mode}/L{level} produced only {got}/{n} valid "
                        f"items within {budget} attempts")
                if skipped:
                    skips[f"{task}|{mode}|L{level:02d}"] = skipped
                if progress:
                    progress(task, mode, level, len(rows))

    if stats is not None:
        # Cells needing many skips mean the generator is straining at that
        # (task, level) -- worth reporting alongside the scores.
        stats["skipped_indices"] = skips
        stats["total_skipped"] = sum(skips.values())

    if gold_failures:
        raise RuntimeError(
            f"gold self-check failed for {len(gold_failures)} item(s); the answer "
            f"parsers do not accept their own gold answers, so no score from this "
            f"taskset would be trustworthy. First few: {gold_failures[:5]}")
    return rows


def taskset_hash(rows: List[dict], kb_sha: Optional[str]) -> str:
    """Content hash over the prompts plus the KB they were drawn from.

    A taskset built against a different KB hashes differently and therefore
    lands in a different directory, so the two can never be confused.
    """
    h = hashlib.sha256()
    for r in rows:
        h.update(r["sample_id"].encode("utf-8"))
        h.update(b"\0")
        h.update(r["prompt"].encode("utf-8"))
        h.update(b"\0")
    h.update((kb_sha or "no-kb").encode("utf-8"))
    return h.hexdigest()


def write_taskset(root: Path, name: str, rows: List[dict], cfg_node,
                  kb_sha: Optional[str], stats: Optional[dict] = None) -> Path:
    full = taskset_hash(rows, kb_sha)
    out = Path(root) / f"{name}-{full[:8]}"
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "taskset.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    OmegaConf.save(cfg_node, out / "config.yaml")

    cells: dict = {}
    for r in rows:
        cells[f"{r['task']}|{r['mode']}|L{r['level']:02d}"] = \
            cells.get(f"{r['task']}|{r['mode']}|L{r['level']:02d}", 0) + 1

    manifest = {
        "name": name,
        "taskset_hash": full,
        "taskset_id": out.name,
        "n_samples": len(rows),
        "n_cells": len(cells),
        "cells": cells,
        "kb_sha256": kb_sha,
        "git_sha": git_sha(),
        "gold_self_check": "passed (every item's own gold answer scores 1.0)",
        "generator_skips": (stats or {}).get("skipped_indices", {}),
        "total_generator_skips": (stats or {}).get("total_skipped", 0),
    }
    with open(out / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
    return out


def load_taskset(taskset_dir: Path) -> List[dict]:
    """Read a taskset, plain or gzipped.

    The committed snapshot under data/tasksets/ is gzipped (19x smaller, since
    every prompt repeats the shared intro). A local rebuild writes a plain
    taskset.jsonl alongside it (gitignored); when present it is preferred, so a
    fresh build overrides the shipped snapshot.
    """
    d = Path(taskset_dir)
    gz = d / "taskset.jsonl.gz"
    plain = d / "taskset.jsonl"
    if plain.exists():
        opener, path = open, plain
    elif gz.exists():
        opener, path = lambda p: gzip.open(p, "rt"), gz
    else:
        raise FileNotFoundError(f"no taskset.jsonl or taskset.jsonl.gz in {d}")

    rows = []
    with opener(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


@hydra.main(version_base=None, config_path="conf", config_name="build")
def main(cfg: DictConfig) -> None:
    root = Path(__file__).resolve().parent.parent
    ts = cfg.taskset
    kb_path = root / "kb.json"
    kb_sha = sha256_file(kb_path) if kb_path.exists() else None

    done = {"n": 0}

    def progress(task, mode, level, total):
        done["n"] = total
        print(f"  {task:26} {mode:8} L{level:02d}  ({total} rows)", flush=True)

    print(f"building taskset '{ts.name}': {len(ts.tasks)} tasks x {len(ts.modes)} "
          f"modes x {len(ts.levels)} levels x n={ts.n}", flush=True)
    stats: dict = {}
    rows = build_rows(list(ts.tasks), list(ts.modes), list(ts.levels), int(ts.n),
                      int(ts.base_seed), bool(ts.validate_gold), progress=progress,
                      stats=stats)

    out = write_taskset(root / cfg.taskset_root, ts.name, rows, ts, kb_sha, stats)
    print(f"\nwrote {len(rows)} samples to {out}")
    if stats.get("total_skipped"):
        print(f"generator skipped {stats['total_skipped']} unproducible indices "
              f"across {len(stats['skipped_indices'])} cells")
    print(f"taskset_id: {out.name}")


if __name__ == "__main__":
    main()
