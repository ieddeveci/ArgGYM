"""A construction answer past twice the minimum scores zero, and the report says so.

The bloat gate zeroes an answer that uses more than `2 * minimum` directives.
On a row whose minimum is 2 the partial-credit band is two directives wide, so
a long answer there is zeroed rather than docked (#136). The zero is the same
number a wrong answer earns, and only `bloat_rate` tells the two apart.
"""
from __future__ import annotations

from conftest import completion
from test_a_reference_answer_scores_one_end_to_end import a_run

from evals.score import _stats, score_run

BLOATED_TASK, KEPT_TASK = "preference_construction", "attack"


def test_bloat_rate_counts_gate_zeros_over_scored_records(tmp_path, rows,
                                                         taskset_file, provider):
    """One row answered three times over, one answered once, the rest never answered."""
    by_task = {r["task"]: r for r in rows}

    def handler(body):
        user = [m for m in body["messages"] if m["role"] == "user"][-1]["content"]
        for task, times in ((BLOATED_TASK, 3), (KEPT_TASK, 1)):
            row = by_task[task]
            if row["question"] in user:
                answer = "\n".join([row["reference_answer"]] * times)
                return completion(f"<answer>\n{answer}\n</answer>")
        return {"__status__": 500, "error": {"message": "boom"}}

    run_dir = a_run(tmp_path, provider(handler).url, rows, taskset_file)
    metrics = score_run(run_dir)

    bloated = metrics["by_task"][BLOATED_TASK]
    assert bloated["mean"] == 0.0
    assert bloated["bloat_rate"] == 1.0
    assert metrics["by_task"][KEPT_TASK]["bloat_rate"] == 0.0
    # An API error is not a scored record, so a group made only of them has no
    # rate at all rather than a rate of zero.
    errored = [v for t, v in metrics["by_task"].items()
               if t not in (BLOATED_TASK, KEPT_TASK)]
    assert errored and all(v["n_scored"] == 0 and v["bloat_rate"] is None
                           for v in errored)
    # Per cell as well as per task.
    for group in ("by_task_level", "by_task_ordering"):
        cells = [v for k, v in metrics[group].items() if k.startswith(BLOATED_TASK + "|")]
        assert cells and all(v["bloat_rate"] == 1.0 for v in cells)


def test_an_api_error_leaves_the_denominator():
    """Same denominator as `no_answer_region_rate`: the records that were scored."""
    base = {"truncated": False, "no_answer_region": False, "answer_in_cot": False}
    records = [
        {**base, "api_error": None, "score": 0.0, "success": False,
         "reason": "bloated:6_used_vs_2_minimum"},
        {**base, "api_error": None, "score": 1.0, "success": True, "reason": "ok"},
        {**base, "api_error": "boom", "api_error_kind": "server", "score": None,
         "success": None, "reason": "api_error"},
    ]
    assert _stats(records, floor=None)["bloat_rate"] == 0.5
