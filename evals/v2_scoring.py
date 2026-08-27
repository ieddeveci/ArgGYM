"""Score stored generations against an ArgGYM_v2 taskset, offline.

Separated from inference so that fixing a parser never costs a GPU hour: every
number here is recomputable from the stored generations alone.

Scoring calls back into the v2 task modules rather than reimplementing them, so
the benchmark's own scorers stay the single source of truth. Items are rebuilt
from (task, level, ordering, seed) because the scorers need live item state --
`base_ops`, `line_ops`, `diagnoses` -- that no JSON row carries. Rebuilding is
safe precisely because generation is deterministic and verified reproducible;
`verify_v2_taskset` is what makes that assumption checkable rather than hoped for.
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evals import artifacts  # noqa: E402
from evals.v2_taskset import load_taskset  # noqa: E402

# Tasks whose answer is a set of directives, scored by core.scoring.score_item
# (goal satisfaction plus efficiency against the verified minimum).
_DIRECTIVE_TASKS = {"preference_construction", "counter_argument",
                    "counter_argument_strict", "attack", "defence",
                    "attack_defense"}


def _v2():
    """Import the v2 tree with its own packages winning over v1's."""
    from evals.v2_taskset import _v2_export
    _v2_export()  # puts ArgGYM_v2 on sys.path and clears colliding modules
    import core.scoring as scoring
    from tasks import (attack_defense, claim_chain, counter_argument,
                       defeat_diagnosis, formalization, perturbation,
                       preference_construction, semantics_query, status_query)
    return {
        "scoring": scoring, "attack_defense": attack_defense,
        "claim_chain": claim_chain, "counter_argument": counter_argument,
        "defeat_diagnosis": defeat_diagnosis, "formalization": formalization,
        "perturbation": perturbation,
        "preference_construction": preference_construction,
        # Every task added to evals/v2_taskset.py's TASKS must also be
        # registered here, or _rebuild raises KeyError and score_sample turns
        # every item of that task into a flat 0.0 -- indistinguishable from a
        # model that answered everything wrong. semantics_query scored exactly
        # 0.000 across all 40 items of the first v4 cell for that reason.
        "semantics_query": semantics_query,
        "status_query": status_query,
    }


_MOD: Optional[dict] = None


def _rebuild(task: str, level: int, ordering: str, seed: int):
    """Rebuild the live item a v2 scorer needs, plus the score callable."""
    global _MOD
    if _MOD is None:
        _MOD = _v2()
    M, score_item = _MOD, _MOD["scoring"].score_item

    if task in ("attack", "defence", "attack_defense"):
        it = M["attack_defense"].make_item(level, seed, ordering, mode=task)
        return it, (lambda txt: score_item(txt, it.as_score_input()))
    if task.startswith("counter_argument"):
        ca = M["counter_argument"]
        it = ca.make_item(level, seed, ordering,
                          allow_strict=task.endswith("_strict"))
        return it, (lambda txt: score_item(txt, ca.as_score_input(it)))
    if task == "preference_construction":
        it = M["preference_construction"].make_item(level, seed, ordering)
        return it, (lambda txt: score_item(txt, it.as_score_input()))
    if task == "perturbation":
        pt = M["perturbation"]
        it = pt.make_item(level, seed, ordering)
        return it, (lambda txt: pt.score(txt, it))
    mod = M.get(task)
    if mod is None:
        raise KeyError(f"no v2 scorer for task {task!r}")
    it = mod.make_item(level, seed, ordering)
    return it, (lambda txt: mod.score(txt, it))


def score_sample(row: dict, gen: dict) -> dict:
    """Score one generation. Failure modes stay separate from the score.

    An API error, a truncated generation and a wrong answer are three different
    events; collapsing them into `score=0` would report infrastructure trouble
    as a reasoning result.
    """
    err = gen.get("error")
    text = (gen.get("content") or "")
    reasoning = (gen.get("reasoning_content") or "")
    truncated = gen.get("finish_reason") == "length"

    base = {
        "sample_id": row["sample_id"], "task": row["task"],
        "level": row["level"], "ordering": row["ordering"], "seed": row["seed"],
        "api_error": bool(err), "truncated": bool(truncated),
        "completion_tokens": (gen.get("usage") or {}).get("completion_tokens"),
        "latency_s": gen.get("latency_s"), "error": err,
    }

    if err:
        return {**base, "score": 0.0, "no_answer_region": False,
                "zero_score_with_valid_region": False, "reason": "api_error",
                "scorer_crashed": None, "efficiency": None,
                "n_used": None, "minimum": row.get("min_directives"),
                "used_fallback": False, "answer_in_cot": False}

    global _MOD
    if _MOD is None:
        _MOD = _v2()
    region = _MOD["scoring"].answer_region

    # Score `content` when it carries an answer region; fall back to
    # reasoning+content only when it does not, which covers models served
    # without a reasoning parser. The draft-vs-revision problem is handled
    # inside answer_region, which takes the LAST complete region.
    in_content = region(text) is not None
    in_cot = region(reasoning) is not None
    if in_content:
        scored_text, used_fallback = text, False
    elif reasoning:
        scored_text, used_fallback = (reasoning + "\n" + text), True
    else:
        scored_text, used_fallback = text, False
    has_region = region(scored_text) is not None

    try:
        _, score_fn = _rebuild(row["task"], row["level"], row["ordering"],
                               row["seed"])
        res = score_fn(scored_text)
        score = float(res.get("score", 0.0))
    except Exception as e:  # a scorer crash must not kill the whole run
        return {**base, "score": 0.0, "no_answer_region": not has_region,
                "zero_score_with_valid_region": False,
                "reason": "scorer_crashed",
                "scorer_crashed": f"{type(e).__name__}: {e}",
                "efficiency": None, "n_used": None,
                "minimum": row.get("min_directives"),
                "used_fallback": used_fallback, "answer_in_cot": in_cot}

    diag = res.get("diagnostics") or {}
    return {
        **base,
        "score": score,
        "reason": res.get("reason"),
        "success": res.get("success"),
        # Efficiency is only defined for the directive tasks: it is the ratio of
        # the verified minimum to the moves the model actually used. Absent
        # elsewhere rather than defaulted, so "no minimum" never reads as 1.0.
        "efficiency": res.get("efficiency"),
        "n_used": diag.get("n_used"),
        "minimum": diag.get("minimum", row.get("min_directives")),
        # Separates "created a conflict and failed to win it" from "did
        # nothing" -- two zeroes that mean opposite things.
        "deadlock_not_defeat": res.get("deadlock_not_defeat"),
        "n_unparseable": diag.get("n_unparseable"),
        "n_illegal": len(diag.get("illegal") or []),
        "no_answer_region": not has_region,
        "zero_score_with_valid_region": bool(has_region and score == 0.0),
        "scorer_crashed": None,
        "used_fallback": used_fallback,
        "answer_in_cot": in_cot,
    }


_FLAGS = ("api_error", "truncated", "no_answer_region",
          "zero_score_with_valid_region")


def _agg(rows: List[dict], scores_ok: bool = True) -> dict:
    """Aggregate a group of scored samples.

    `scores_ok=False` keeps the counts and the format/plumbing rates but omits
    every score field. Used for groups that span tasks: `min_directives` means a
    different thing per task -- moves to make for the directive tasks, theory
    size for status_query and formalization -- so a mean over tasks mixes units.
    Withholding the number is the only reliable way to stop it being quoted.
    """
    if not rows:
        return {}
    out = {"n": len(rows)}
    if scores_ok:
        s = [r["score"] for r in rows]
        out["mean_score"] = round(statistics.fmean(s), 4)
        out["perfect_rate"] = round(sum(1 for x in s if x >= 1.0) / len(s), 4)
        out["zero_rate"] = round(sum(1 for x in s if x <= 0.0) / len(s), 4)
        if len(s) > 1:
            out["stderr"] = round(statistics.stdev(s) / (len(s) ** 0.5), 4)
        eff = [r["efficiency"] for r in rows if r.get("efficiency") is not None]
        if eff:
            out["mean_efficiency"] = round(statistics.fmean(eff), 4)
            out["n_with_efficiency"] = len(eff)
        succ = [r for r in rows if r.get("success") is not None]
        if succ:
            out["success_rate"] = round(
                sum(1 for r in succ if r["success"]) / len(succ), 4)
        dl = [r["deadlock_not_defeat"] for r in rows
              if r.get("deadlock_not_defeat") is not None]
        if dl:
            out["mean_deadlocks"] = round(statistics.fmean(dl), 4)
    for f in _FLAGS:
        out[f + "_rate"] = round(sum(1 for r in rows if r.get(f)) / len(rows), 4)
    toks = [r["completion_tokens"] for r in rows if r.get("completion_tokens")]
    if toks:
        out["mean_completion_tokens"] = round(statistics.fmean(toks), 1)
    return out


# Groups whose members share one task, so a mean over them is a quantity.
# `overall` and `by_level` mix tasks and are emitted without score fields.
_TASK_HOMOGENEOUS = ("by_task", "by_task_level", "by_task_ordering")


def aggregate(scored: List[dict]) -> dict:
    by: Dict[str, Dict[str, list]] = {
        "by_task": defaultdict(list), "by_level": defaultdict(list),
        "by_ordering": defaultdict(list), "by_task_level": defaultdict(list),
        "by_task_ordering": defaultdict(list),
    }
    for r in scored:
        o = "ll" if r["ordering"].startswith("last") else "wl"
        by["by_task"][r["task"]].append(r)
        by["by_level"][f"L{r['level']:02d}"].append(r)
        by["by_ordering"][o].append(r)
        by["by_task_level"][f"{r['task']}|L{r['level']:02d}"].append(r)
        by["by_task_ordering"][f"{r['task']}|{o}"].append(r)

    return {
        "overall": _agg(scored, scores_ok=False),
        **{k: {kk: _agg(vv, scores_ok=(k in _TASK_HOMOGENEOUS))
               for kk, vv in sorted(v.items())} for k, v in by.items()},
    }


def write_metrics_csv(path: Path, metrics: dict) -> None:
    cols = ["group", "key", "n", "mean_score", "stderr", "perfect_rate",
            "zero_rate", "success_rate", "mean_efficiency", "mean_deadlocks",
            "mean_completion_tokens"] + [f + "_rate" for f in _FLAGS]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerow({"group": "overall", "key": "all", **metrics["overall"]})
        for g in ("by_task", "by_level", "by_ordering", "by_task_level",
                  "by_task_ordering"):
            for key, vals in metrics.get(g, {}).items():
                w.writerow({"group": g, "key": key, **vals})


def score_run(run_dir: Path, taskset_dir: Optional[Path] = None) -> dict:
    run_dir = Path(run_dir)
    run = artifacts.read_json(run_dir / "run.json") or {}
    if taskset_dir is None:
        taskset_dir = ROOT / "data" / "tasksets" / run["taskset_id"]
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
        "benchmark": "ArgGYM_v2",
        "model": (run.get("model") or {}).get("name"),
        "taskset_id": run.get("taskset_id"),
        "taskset_hash": run.get("taskset_hash"),
        "n_scored": len(scored),
        "n_expected": run.get("n_samples"),
    }
    artifacts.write_json(run_dir / "metrics.json", metrics)
    write_metrics_csv(run_dir / "metrics.csv", metrics)
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Score a completed v2 run.")
    ap.add_argument("run_dir")
    ap.add_argument("--taskset-dir", default=None)
    a = ap.parse_args()
    m = score_run(Path(a.run_dir),
                  Path(a.taskset_dir) if a.taskset_dir else None)
    o = m["overall"]
    # Per task, never one number across tasks: `minimum` means different things
    # per task, so a grand mean would mix units. The rates below are
    # format/plumbing counters and are task-agnostic.
    print(f"{m['_meta']['model']}: n={o['n']} "
          f"no_region={o['no_answer_region_rate']:.3f} "
          f"truncated={o['truncated_rate']:.3f} "
          f"api_err={o['api_error_rate']:.3f}")
    for task, v in sorted(m.get("by_task", {}).items()):
        eff = (f" eff={v['mean_efficiency']:.3f}"
               if "mean_efficiency" in v else "")
        print(f"  {task:26s} n={v['n']:4d} score={v['mean_score']:.4f}{eff}")


if __name__ == "__main__":
    main()
