"""Score a finished run, offline. This program never opens a socket.

Everything it needs is on disk: the completions from `run.py` and the rows from
the frozen taskset. So a scorer fix, a template change or an extraction bug
costs seconds here rather than the hours of inference that produced the
completions.

    uv run python -m evals.score outputs/runs/<dir> --taskset data/taskset.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence

import arggym
from evals import artifacts, taskset, values
from evals.prompt import region


class TasksetMismatch(SystemExit):
    """The run was generated against different questions than these."""


def answer_of(gen: Dict[str, Any], template: Optional[str]) -> Dict[str, Any]:
    """The text to score, and what had to be done to find it.

    The completion is read first. Only if it holds no answer region is the
    reasoning trace consulted, because a model that reaches an answer mid-thought
    and never submits it has done something different from one that submits, and
    the flag says which. The fallback never changes a score that a submitted
    answer would have got; it only rescues one that was never submitted.
    """
    body, found = region(gen.get("completion") or "", template)
    if found or not gen.get("reasoning"):
        return {"answer": body, "no_answer_region": not found, "answer_in_cot": False}
    merged, found_cot = region(f"{gen['reasoning']}\n{gen.get('completion') or ''}",
                               template)
    return {"answer": merged if found_cot else body,
            "no_answer_region": not found_cot, "answer_in_cot": found_cot}


def score_one(gen: Dict[str, Any], row: Dict[str, Any],
              template: Optional[str]) -> Dict[str, Any]:
    """One scored record. An API error is never a zero."""
    out: Dict[str, Any] = {
        "id": gen["id"], "task": gen["task"], "level": gen["level"],
        "ordering": gen["ordering"], "api_error": gen.get("error"),
        "truncated": bool(gen.get("truncated")),
        "completion_tokens": (gen.get("usage") or {}).get("completion_tokens"),
        "latency_s": gen.get("latency_s"), "attempts": gen.get("attempts"),
    }
    # Every documented field is present on every record, whatever happened.
    # `samples.jsonl` is read outside this process, and a reader that has to
    # guess whether a missing key means false or means "not applicable" is
    # being invited to guess wrong.
    out.update(scorer_refused=False, zero_with_region=False)
    if gen.get("error"):
        # No score, no success, no zero. The item was not measured.
        out.update(score=None, success=None, reason="api_error",
                   no_answer_region=None, answer_in_cot=None)
        return out

    by_value = gen.get("value") is not None
    if by_value:
        # A solver that produced the answer as a value did its own parsing, so
        # nothing is extracted and no fence was ever asked for.
        out.update(no_answer_region=False, answer_in_cot=False)
    else:
        found = answer_of(gen, template)
        out.update(no_answer_region=found["no_answer_region"],
                   answer_in_cot=found["answer_in_cot"])

    try:
        result = (arggym.score_row_value(values.decode(gen["value"], row), row)
                  if by_value
                  else arggym.score_row(found["answer"], row))
    except Exception as e:  # noqa: BLE001
        # `score_row` refuses a row it cannot score -- a mismatched engine
        # version, a missing field. That is a fault in the taskset or the
        # install, not an answer worth zero, so it is flagged rather than
        # averaged. The previous harness caught this with a broad except, called
        # it 0.0, and published a cell of forty items scoring exactly 0.000.
        out.update(score=None, success=None, reason=f"{type(e).__name__}: {e}",
                   scorer_refused=True)
        return out

    out.update(score=result.score, success=result.success, reason=result.reason,
               diagnostics=result.diagnostics, scorer_refused=False)
    # A well-formed answer that scores zero is the interesting case: either the
    # model reasoned wrongly, which is a result, or the scorer is wrong, which
    # is a bug. Counting them is how the second gets noticed.
    out["zero_with_region"] = (result.score == 0.0 and not out["no_answer_region"])
    return out


def _stats(records: Sequence[Dict[str, Any]], floor: Optional[float]) -> Dict[str, Any]:
    """One group's numbers, with what was not measured said first."""
    n = len(records)
    ok = [r for r in records if r["score"] is not None]
    untrunc = [r for r in ok if not r["truncated"]]
    mean = sum(r["score"] for r in ok) / len(ok) if ok else None
    out: Dict[str, Any] = {
        "n": n, "n_scored": len(ok),
        "n_api_error": sum(1 for r in records if r["api_error"]),
        "n_scorer_refused": sum(1 for r in records if r.get("scorer_refused")),
        # Read before the score. On the August sweep truncation removed 74-89%
        # of items for three of seven models, and a mean over what survived is
        # a measurement of the token cap.
        "truncated_rate": _rate(sum(r["truncated"] for r in records), n),
        "no_answer_region_rate": _rate(
            sum(bool(r.get("no_answer_region")) for r in ok), len(ok)),
        "answer_in_cot_rate": _rate(
            sum(bool(r.get("answer_in_cot")) for r in ok), len(ok)),
        "mean": round(mean, 4) if mean is not None else None,
        # The same mean over the generations that were not cut off. Reported
        # beside the raw one, never instead of it: on the previous sweep a 9B
        # model scored 0.598 raw and 0.765 censored, and only the pair says why.
        "mean_untruncated": (round(sum(r["score"] for r in untrunc) / len(untrunc), 4)
                             if untrunc else None),
        "success_rate": _rate(sum(bool(r["success"]) for r in ok), len(ok)),
        "zero_with_region": sum(bool(r.get("zero_with_region")) for r in ok),
    }
    if floor is not None:
        out["floor"] = round(floor, 4)
        # Chance-corrected, which is the only form in which two tasks' scores
        # are the same quantity (`docs/dataset-contract.md` section 10). The
        # floor bounds the mean score and nothing else, so nothing else here is
        # corrected by it.
        out["corrected"] = (round(arggym.corrected(mean, floor), 4)
                            if mean is not None else None)
    return out


def coverage(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """What was not measured. No score in here, deliberately.

    A pooled mean over every record is a mean across tasks, which is the one
    number `aggregate` refuses to produce three lines below. Publishing it in
    the file the report reads would make the artifact contradict its own
    docstring, and it is the obvious thing for a reader to quote.
    """
    n = len(records)
    ok = [r for r in records if r["score"] is not None]
    return {
        "n": n, "n_scored": len(ok),
        "n_api_error": sum(1 for r in records if r["api_error"]),
        "n_scorer_refused": sum(1 for r in records if r.get("scorer_refused")),
        "truncated_rate": _rate(sum(r["truncated"] for r in records), n),
        "no_answer_region_rate": _rate(
            sum(bool(r.get("no_answer_region")) for r in ok), len(ok)),
        "answer_in_cot_rate": _rate(
            sum(bool(r.get("answer_in_cot")) for r in ok), len(ok)),
    }


def _rate(numerator: int, denominator: int) -> Optional[float]:
    return round(numerator / denominator, 4) if denominator else None


def _floors_of(rows: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """Measured floors, or none at all if the scorer cannot grade these rows.

    `arggym.floors` scores every row with each constant strategy and does not
    guard, so a single ungradeable row would abort the whole scoring pass --
    after `score_one` has already recorded that row politely. Losing every
    number to one bad row is the wrong trade.
    """
    try:
        return {t: v["floor"] for t, v in arggym.floors(list(rows)).items()}
    except Exception:  # noqa: BLE001 - reported in _meta, never silently zero
        return {}


def aggregate(records: List[Dict[str, Any]], rows: Dict[str, Dict[str, Any]]
              ) -> Dict[str, Any]:
    """Per task, per task and level, per task and ordering. Never one mean.

    There is deliberately no overall average. The twelve metrics are of four
    kinds and their chance floors span half the range, so a mean over them moves
    mostly with which tasks are in the basket. The previous harness published one
    as a "ranking aid"; the contract says not to.

    Each grouping measures its own floor over its own rows. A floor is measured,
    not constant: `docs/dataset-contract.md` section 10 quotes them per level
    because that is what they vary with, so correcting a level-15 mean by a
    floor averaged over levels 3 to 15 produces a well-formed number that means
    nothing.
    """
    groups: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
        "by_task": defaultdict(list), "by_task_level": defaultdict(list),
        "by_task_ordering": defaultdict(list)}
    for r in records:
        groups["by_task"][r["task"]].append(r)
        groups["by_task_level"][f"{r['task']}|L{r['level']}"].append(r)
        groups["by_task_ordering"][f"{r['task']}|{r['ordering']}"].append(r)

    out: Dict[str, Any] = {}
    for name, buckets in groups.items():
        out[name] = {}
        for key, got in sorted(buckets.items()):
            mine = [rows[r["id"]] for r in got if r["id"] in rows]
            floor = _floors_of(mine).get(key.split("|")[0])
            out[name][key] = _stats(got, floor)
    return out


def score_run(run_dir: str, taskset_path: Optional[str] = None) -> Dict[str, Any]:
    with open(os.path.join(run_dir, artifacts.RUN)) as f:
        run = json.load(f)
    path = taskset_path or run["taskset"]
    ts_manifest, all_rows = taskset.load(path)

    got, want = ts_manifest.get("taskset_hash"), run.get("taskset_hash")
    if got != want:
        raise TasksetMismatch(
            f"{path} has taskset_hash {got}, and this run was generated against "
            f"{want}. Scoring answers against questions they were not asked is a "
            f"silent way to publish a wrong number; find the taskset this run "
            f"used, or rerun it.")

    rows = {r["id"]: r for r in all_rows}
    generations = artifacts.best_per_id(run_dir)
    unknown = sorted(set(generations) - set(rows))
    if unknown:
        raise TasksetMismatch(
            f"{len(unknown)} generated ids are in no row of {path} "
            f"({', '.join(unknown[:3])}).")

    if "template" not in run:
        # `null` is a legitimate template -- it means the whole completion is
        # the answer -- so an absent key cannot be read as one. Guessing would
        # hand the strict parser a reasoning preamble and report a plausible
        # low score with no warning.
        raise TasksetMismatch(
            f"{artifacts.RUN} in {run_dir} names no template. It is written by "
            f"run.py and is needed to read the answer back out; without it "
            f"there is no way to tell 'no fence was asked for' from 'the key is "
            f"missing', and the two score differently.")
    template = run["template"]
    records = [score_one(generations[i], rows[i], template) for i in sorted(generations)]

    artifacts.write_jsonl(os.path.join(run_dir, artifacts.SAMPLES), records)

    metrics = {
        "_meta": {
            "run_dir": os.path.abspath(run_dir),
            "endpoint": run.get("endpoint"), "template": template,
            "elicitation": run.get("elicitation"),
            "taskset": path, "taskset_hash": got,
            "taskset_versions": ts_manifest.get("versions", {}),
            "run_status": run.get("status"),
            "arggym": arggym.__version__,
            "n_selected": run.get("n_selected"), "n_scored": len(records),
            # A run interrupted before it finished has fewer generations than
            # it selected, and the difference is the only sign of it once
            # `run.json` is out of view.
            "n_not_generated": (run.get("n_selected") or len(records)) - len(records),
        },
        # What was not measured, before anything that was.
        "coverage": coverage(records),
        **aggregate(records, rows),
    }
    artifacts.write_json(os.path.join(run_dir, artifacts.METRICS), metrics)
    return metrics


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir")
    p.add_argument("--taskset", default=None,
                   help="Defaults to the path the run recorded.")
    a = p.parse_args(argv)
    m = score_run(a.run_dir, a.taskset)

    c = m["coverage"]
    print(f"{c['n_scored']} scored, {c['n_api_error']} API errors, "
          f"{c['n_scorer_refused']} unscorable, {_pct(c['truncated_rate'])} truncated, "
          f"{_pct(c['no_answer_region_rate'])} with no answer region\n")
    print(f"{'task':26s} {'n':>4s} {'scored':>6s} {'mean':>7s} {'untrunc':>8s} "
          f"{'floor':>7s} {'corr':>7s} {'succ':>7s}")
    for task, v in m["by_task"].items():
        print(f"{task:26s} {v['n']:4d} {v['n_scored']:6d} {_f(v['mean'])} "
              f"{_f(v['mean_untruncated'])} {_f(v.get('floor'))} "
              f"{_f(v.get('corrected'))} {_f(v['success_rate'])}")
    print("\nNo overall mean: the per-task metrics are not the same quantity.")
    return 0


def _f(x: Optional[float]) -> str:
    return "      -" if x is None else f"{x:7.3f}"


def _pct(x: Optional[float]) -> str:
    """A rate with no denominator has no percentage, and says so."""
    return "-" if x is None else f"{x:.1%}"


if __name__ == "__main__":
    raise SystemExit(main())
