"""Score stored generations offline and aggregate.

Deliberately separated from inference: the answer parsers are still under audit
(workspace/benchmark/todo.md item 2), and when one is fixed every number must be
recomputable without touching a GPU.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aspic_gym import score_answer  # noqa: E402
from evals import artifacts  # noqa: E402
from evals.extract import extract  # noqa: E402
from evals.taskset import load_taskset  # noqa: E402


def score_sample(row: dict, gen: dict) -> dict:
    """Score one generation. Failure modes stay separate from the score."""
    err = gen.get("error")
    ex = extract(gen.get("content"), gen.get("reasoning_content"))
    truncated = gen.get("finish_reason") == "length"

    if err:
        score = 0.0
    else:
        try:
            score = float(score_answer(ex["text"], row["entry"]))
        except Exception as e:  # a parser crash must not kill the whole run
            return {
                "sample_id": row["sample_id"], "task": row["task"],
                "mode": row["mode"], "level": row["level"], "idx": row["idx"],
                "score": 0.0, "api_error": False, "truncated": truncated,
                "no_answer_region": not ex["has_region"],
                "zero_score_with_valid_region": False,
                "scorer_crashed": f"{type(e).__name__}: {e}",
                "used_fallback": ex["used_fallback"],
                "answer_in_cot": ex["answer_in_cot"],
                "completion_tokens": (gen.get("usage") or {}).get("completion_tokens"),
                "latency_s": gen.get("latency_s"),
            }

    return {
        "sample_id": row["sample_id"],
        "task": row["task"],
        "mode": row["mode"],
        "level": row["level"],
        "idx": row["idx"],
        "score": score,
        "api_error": bool(err),
        "truncated": bool(truncated),
        "no_answer_region": (not ex["has_region"]) and not err,
        # A well-formed answer region that still scored exactly zero: either a
        # genuine miss or a parser that cannot read a valid answer. High rates
        # here are the signal to inspect a task's parser.
        "zero_score_with_valid_region": bool(ex["has_region"] and score == 0.0 and not err),
        "scorer_crashed": None,
        "used_fallback": ex["used_fallback"],
        "answer_in_cot": ex["answer_in_cot"],
        "completion_tokens": (gen.get("usage") or {}).get("completion_tokens"),
        "latency_s": gen.get("latency_s"),
        "error": err,
    }


_FLAGS = ("api_error", "truncated", "no_answer_region", "zero_score_with_valid_region")


def _agg(rows: List[dict]) -> dict:
    if not rows:
        return {}
    scores = [r["score"] for r in rows]
    toks = [r["completion_tokens"] for r in rows if r.get("completion_tokens")]
    out = {
        "n": len(rows),
        "mean_score": round(statistics.fmean(scores), 4),
        "perfect_rate": round(sum(1 for s in scores if s >= 1.0) / len(scores), 4),
        "zero_rate": round(sum(1 for s in scores if s <= 0.0) / len(scores), 4),
    }
    if len(scores) > 1:
        out["stderr"] = round(statistics.stdev(scores) / (len(scores) ** 0.5), 4)
    for f in _FLAGS:
        out[f + "_rate"] = round(sum(1 for r in rows if r.get(f)) / len(rows), 4)
    if toks:
        out["mean_completion_tokens"] = round(statistics.fmean(toks), 1)
    return out


def aggregate(scored: List[dict]) -> dict:
    by: Dict[str, Dict[str, list]] = {
        "by_task": defaultdict(list), "by_mode": defaultdict(list),
        "by_level": defaultdict(list), "by_task_mode": defaultdict(list),
    }
    for r in scored:
        by["by_task"][r["task"]].append(r)
        by["by_mode"][r["mode"]].append(r)
        by["by_level"][f"L{r['level']:02d}"].append(r)
        by["by_task_mode"][f"{r['task']}|{r['mode']}"].append(r)

    return {
        "overall": _agg(scored),
        **{k: {kk: _agg(vv) for kk, vv in sorted(v.items())} for k, v in by.items()},
    }


def write_metrics_csv(path: Path, metrics: dict) -> None:
    cols = ["group", "key", "n", "mean_score", "stderr", "perfect_rate", "zero_rate",
            "mean_completion_tokens"] + [f + "_rate" for f in _FLAGS]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerow({"group": "overall", "key": "all", **metrics["overall"]})
        for group in ("by_task", "by_mode", "by_level", "by_task_mode"):
            for key, vals in metrics.get(group, {}).items():
                w.writerow({"group": group, "key": key, **vals})


def score_run(run_dir: Path, taskset_dir: Optional[Path] = None) -> dict:
    run_dir = Path(run_dir)
    root = Path(__file__).resolve().parent.parent
    run = artifacts.read_json(run_dir / "run.json") or {}
    if taskset_dir is None:
        taskset_dir = root / "outputs" / "tasksets" / run["taskset_id"]
    rows = {r["sample_id"]: r for r in load_taskset(Path(taskset_dir))}

    scored = []
    for sid, row in rows.items():
        gen = artifacts.read_generation(run_dir, sid)
        if gen is None:
            continue
        s = score_sample(row, gen)
        artifacts.write_score(run_dir, sid, s)
        scored.append(s)

    scored.sort(key=lambda r: r["sample_id"])
    with open(run_dir / "samples.jsonl", "w") as fh:
        for s in scored:
            fh.write(json.dumps(s, ensure_ascii=False) + "\n")

    metrics = aggregate(scored)
    metrics["_meta"] = {
        "model": (run.get("model") or {}).get("name"),
        "taskset_id": run.get("taskset_id"),
        "taskset_hash": run.get("taskset_hash"),
        "kb_sha256": run.get("kb_sha256"),
        "n_scored": len(scored),
        "n_expected": run.get("n_samples"),
    }
    artifacts.write_json(run_dir / "metrics.json", metrics)
    write_metrics_csv(run_dir / "metrics.csv", metrics)
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Score a completed run directory.")
    ap.add_argument("run_dir")
    ap.add_argument("--taskset-dir", default=None)
    a = ap.parse_args()
    m = score_run(Path(a.run_dir), Path(a.taskset_dir) if a.taskset_dir else None)
    o = m["overall"]
    print(f"{m['_meta']['model']}: n={o['n']} mean={o['mean_score']:.4f} "
          f"perfect={o['perfect_rate']:.3f} "
          f"no_region={o['no_answer_region_rate']:.3f} "
          f"truncated={o['truncated_rate']:.3f} "
          f"api_err={o['api_error_rate']:.3f}")


if __name__ == "__main__":
    main()
