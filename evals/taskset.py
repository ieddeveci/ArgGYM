"""Materialise the evaluation grid into a frozen, reusable taskset.

Built once on CPU and shared by every model in the roster, so all models see
byte-identical prompts even if kb.json changes between runs.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional

import hydra
from omegaconf import DictConfig, OmegaConf

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aspic_gym import create_dataset, score_content  # noqa: E402


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


def git_dirty() -> Optional[bool]:
    """True if tracked files had uncommitted changes when the taskset was built.

    Recorded so `git_sha` is not read as a complete description of the code that
    produced the taskset -- a dirty tree means HEAD alone does not reproduce it.
    """
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=Path(__file__).resolve().parent.parent, text=True,
            stderr=subprocess.DEVNULL)
        return bool(out.strip())
    except Exception:
        return None


def _build_cell(spec: tuple) -> dict:
    """Generate one (task, mode, level) cell. Independent and self-seeded, so it
    is safe to run in a worker process; results are reordered by the caller."""
    task, mode, level, n, base_seed, validate_gold = spec
    with_content = (mode == "content")
    seed = cell_seed(base_seed, task, mode, level)
    # Some (task, level) combinations occasionally fail to produce a valid theory
    # at a given index even after the generator's own 60 retries. Draw from an
    # oversized dataset and skip those indices deterministically rather than
    # shrinking the cell: the skip is a function of (seed, idx) alone, so the
    # taskset stays reproducible and raising n later still extends the prefix.
    budget = max(n * 10, n + 50)
    ds = create_dataset(task, seed=seed, size=budget, level=level,
                        with_content=with_content)
    rows: List[dict] = []
    gold_failures: List[tuple] = []
    got, idx, skipped = 0, 0, 0
    while got < n and idx < budget:
        try:
            entry = ds[idx]
        except RuntimeError:
            skipped += 1
            idx += 1
            continue
        if validate_gold:
            # Gold is stored as raw answer content (no submission delimiters), so
            # score it directly rather than through the [answer]-region extractor.
            s = score_content(entry["answer"], entry)
            if s < 1.0:
                gold_failures.append((task, mode, level, idx, s))
        rows.append({
            "sample_id": sample_id(task, mode, level, idx),
            "task": task, "mode": mode, "level": level, "idx": idx,
            "cell_seed": seed, "prompt": entry["question"], "entry": entry,
        })
        got += 1
        idx += 1
    if got < n:
        raise RuntimeError(
            f"cell {task}/{mode}/L{level} produced only {got}/{n} valid "
            f"items within {budget} attempts")
    return {"key": f"{task}|{mode}|L{level:02d}", "rows": rows,
            "skipped": skipped, "gold_failures": gold_failures}


def build_rows(tasks: Iterable[str], modes: Iterable[str], levels: Iterable[int],
               n: int, base_seed: int, validate_gold: bool = True,
               progress=None, stats: Optional[dict] = None,
               workers: Optional[int] = None) -> List[dict]:
    """Generate every (task, mode, level) cell.

    Cells are independent and each derives its RNG from (base_seed, task, mode,
    level) alone, so they run in parallel worker processes with no effect on the
    result -- generation is also PYTHONHASHSEED-independent. Rows are reassembled
    in the original task x mode x level order, so the taskset hash is stable
    whatever the worker count. Because ASPICDataset derives its per-item RNG from
    idx alone, raising `n` later extends each cell instead of reshuffling it.
    """
    specs = [(task, mode, level, n, base_seed, validate_gold)
             for task in tasks for mode in modes for level in levels]
    if workers is None:
        # Independent cells, so scale to cores; cap so a huge-core box does not
        # fork a hundred KB-carrying processes for a marginal tail-latency gain.
        workers = min(len(specs), (os.cpu_count() or 4), 32)
    workers = max(1, min(workers, len(specs)))

    results: dict = {}
    done_rows = 0
    if workers == 1:
        for spec in specs:
            r = _build_cell(spec)
            results[(spec[0], spec[1], spec[2])] = r
            done_rows += len(r["rows"])
            if progress:
                progress(spec[0], spec[1], spec[2], done_rows)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_build_cell, s): (s[0], s[1], s[2]) for s in specs}
            for fut in as_completed(futs):
                key = futs[fut]
                r = fut.result()
                results[key] = r
                done_rows += len(r["rows"])
                if progress:
                    progress(key[0], key[1], key[2], done_rows)

    rows: List[dict] = []
    skips: dict = {}
    gold_failures: List[tuple] = []
    for task in tasks:
        for mode in modes:
            for level in levels:
                r = results[(task, mode, level)]
                rows.extend(r["rows"])
                if r["skipped"]:
                    skips[r["key"]] = r["skipped"]
                gold_failures.extend(r["gold_failures"])

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
                  kb_sha: Optional[str], stats: Optional[dict] = None,
                  allow_dirty: bool = False) -> Path:
    """Freeze `rows` into a taskset directory.

    Refuses a dirty working tree by default. A taskset is a benchmark artifact
    whose identity is (code, config, kb): if the code that produced it was never
    committed, the recorded `git_sha` names a tree that does not regenerate it,
    and nobody -- including its author a week later -- can tell what it measured.
    Recording `git_dirty` is not enough on its own; that only documents the
    problem after the fact.

    `allow_dirty=True` is the escape for local experiments. It stamps `dirty`
    into the directory name so such a build can never be picked up as canonical
    by a glob or mistaken for one by eye.
    """
    dirty = git_dirty()
    if dirty and not allow_dirty:
        raise RuntimeError(
            "refusing to build a taskset from a dirty working tree: the recorded "
            "git_sha would not reproduce it. Commit (or stash) first, or pass "
            "allow_dirty=true for a throwaway local build.")

    full = taskset_hash(rows, kb_sha)
    built = datetime.now(timezone.utc)
    # Time-ordered directory name: the UTC build stamp sorts chronologically, so
    # `ls` and globbing put the newest build last. The content hash is retained
    # after it as a stable identity -- same code+config+kb reproduce the same
    # hash, which is how two builds are told apart from mere re-runs.
    stamp = built.strftime("%Y%m%dT%H%M%SZ")
    tag = "-dirty" if dirty else ""
    out = Path(root) / f"{name}{tag}-{stamp}-{full[:8]}"
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
        "built_at": built.isoformat().replace("+00:00", "Z"),
        "git_sha": git_sha(),
        "git_dirty": dirty,
        "n_samples": len(rows),
        "n_cells": len(cells),
        "cells": cells,
        "kb_sha256": kb_sha,
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
    workers = cfg.get("build_workers")
    rows = build_rows(list(ts.tasks), list(ts.modes), list(ts.levels), int(ts.n),
                      int(ts.base_seed), bool(ts.validate_gold), progress=progress,
                      stats=stats, workers=int(workers) if workers else None)

    out = write_taskset(root / cfg.taskset_root, ts.name, rows, ts, kb_sha, stats,
                        allow_dirty=bool(cfg.get("allow_dirty")))
    print(f"\nwrote {len(rows)} samples to {out}")
    if stats.get("total_skipped"):
        print(f"generator skipped {stats['total_skipped']} unproducible indices "
              f"across {len(stats['skipped_indices'])} cells")
    print(f"taskset_id: {out.name}")


if __name__ == "__main__":
    main()
