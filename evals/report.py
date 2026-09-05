"""Compare finished runs. Reads `metrics.json`; runs nothing.

    uv run python -m evals.report outputs/runs/* -o outputs/reports/latest

Two rules, both from `docs/dataset-contract.md` section 10.

Chance floors are printed beside the scores, because a score means nothing
without the number a constant answer gets: at level 3 `semantics_query` sits at
0.490 and `status_query` at 0.375, so a model scoring 0.45 on the first is doing
worse than answering the same thing every time.

There is no overall mean. Twelve metrics of four kinds, with floors spanning half
the range, do not average into a quantity; the number that comes out moves mostly
with which tasks are in the basket. Where a single figure is wanted the table
gives the chance-corrected column, which is at least the same quantity across
tasks.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any, Dict, List, Optional, Sequence


class NotScored(SystemExit):
    """A run directory was named that has no metrics."""


def load_metrics(paths: Sequence[str]) -> List[Dict[str, Any]]:
    """Every named run's metrics, or a refusal naming the ones without any.

    Skipping an unscored run quietly turns "we never scored this model" into a
    table that simply lacks its column, and a reader cannot tell that from a
    model that was never run. `make eval` makes this easy to hit: two parallel
    invocations can both score whichever directory was written to last.
    """
    out, missing = [], []
    for p in paths:
        f = p if p.endswith(".json") else os.path.join(p, "metrics.json")
        if not os.path.exists(f):
            missing.append(p)
            continue
        with open(f) as fh:
            m = json.load(fh)
        m["_label"] = label(m)
        out.append(m)
    if missing:
        raise NotScored(
            f"{len(missing)} of {len(paths)} run directories have no "
            f"metrics.json: {', '.join(missing[:4])}"
            + (" ..." if len(missing) > 4 else "")
            + ". Score them first, or leave them out deliberately -- a report "
              "that drops them silently shows 'not scored' as 'not run'.")
    return sorted(out, key=lambda m: m["_label"])


def label(m: Dict[str, Any]) -> str:
    """What names a column, and it is not the model alone.

    Two runs of one model under different elicitations are a comparison; giving
    them the same column header makes them a collision. `evals/prompt.py` states
    the intent: two runs differing only in elicitation are "comparable as an
    experiment rather than confusable as a result".
    """
    meta = m["_meta"]
    e = meta.get("endpoint") or {}
    base = str(e.get("model") or os.path.basename(meta.get("run_dir", "run")))
    parts = [base]
    if meta.get("elicitation") and meta["elicitation"] != "none":
        parts.append(str(meta["elicitation"]))
    if meta.get("template") != "xml_tags":
        parts.append(str(meta.get("template")))
    return "/".join(parts)


def coverage_table(runs: List[Dict[str, Any]]) -> str:
    """What was not measured, printed before anything that was.

    A model that lost most of its items to the token cap has not been measured
    on reasoning, and reading its mean as if it had is how the previous sweep's
    headline numbers went wrong.
    """
    lines = ["| run | status | scored | API errors | truncated | no answer region |",
             "|---|---|---:|---:|---:|---:|"]
    for m in runs:
        c, meta = m["coverage"], m["_meta"]
        lines.append(
            f"| {m['_label']} | {meta.get('run_status')} | {c['n_scored']} | "
            f"{c['n_api_error']} | {_pct(c['truncated_rate'])} | "
            f"{_pct(c['no_answer_region_rate'])} |")
    return "\n".join(lines)


def task_table(runs: List[Dict[str, Any]], field: str, show_floor: bool = True) -> str:
    """One row per task, one column per run, with the count beside the value.

    The count is there because a mean over three surviving items and a mean over
    forty are formatted identically otherwise, and the coverage table above is
    run-wide rather than per task.

    `show_floor` is off for `success_rate`: a floor is the mean *score* of the
    best constant strategy (`arggym/core/floors.py`), so printing it beside a
    success rate invites exactly the comparison it exists to prevent.
    """
    tasks = sorted({t for m in runs for t in m["by_task"]})
    head = "| task |" + (" floor |" if show_floor else "") + \
        " " + " | ".join(m["_label"] for m in runs) + " |"
    rule = "|---|" + ("---:|" if show_floor else "") + "---:|" * len(runs)
    lines = [head, rule]
    for t in tasks:
        seen = [m["by_task"][t]["floor"] for m in runs
                if t in m["by_task"] and m["by_task"][t].get("floor") is not None]
        floor = seen[0] if seen else None
        # A floor is measured over the rows a run actually scored, so a filtered
        # run and a full one produce different ones. Printing a single number
        # then makes `(score - floor) / (1 - floor)` unreproducible from the
        # table it is printed in, so the disagreement is shown rather than hidden.
        floor_cell = _num(floor) + ("*" if len(set(seen)) > 1 else "")
        cells = [_cell(m["by_task"].get(t, {}), field) for m in runs]
        lines.append(f"| {t} |" + (f" {floor_cell} |" if show_floor else "")
                     + " " + " | ".join(cells) + " |")
    if any(len({m["by_task"][t]["floor"] for m in runs
                if t in m["by_task"] and m["by_task"][t].get("floor") is not None}) > 1
           for t in tasks):
        lines.append("")
        lines.append("`*` these runs measured different floors for that task, so "
                     "one column's `corrected` cannot be reproduced from this "
                     "floor. Floors are measured over the rows a run scored.")
    return "\n".join(lines)


def _cell(stats: Dict[str, Any], field: str) -> str:
    if not stats:
        return "-"
    n = stats.get("n_scored")
    return _num(stats.get(field)) + (f" ({n})" if n is not None else "")


def render(runs: List[Dict[str, Any]]) -> str:
    if not runs:
        return "# ArgGYM evaluation\n\nNo scored runs found.\n"
    versions = {json.dumps(m["_meta"].get("taskset_versions"), sort_keys=True)
                for m in runs}
    hashes = {m["_meta"].get("taskset_hash") for m in runs}
    parts = [
        "# ArgGYM evaluation", "",
        f"Taskset `{'`, `'.join(sorted(str(h) for h in hashes))}`.", "",
    ]
    if len(hashes) > 1:
        # Not a warning to be dismissed: the runs answered different questions,
        # so the columns below are not comparable.
        parts += ["**These runs used different tasksets.** The columns are not "
                  "comparable; they answered different questions.", ""]
    if len(versions) > 1:
        parts += ["**These runs used different prompt or scoring versions.** A "
                  "score moves when either does.", ""]
    parts += [
        "## Coverage", "",
        "Read this table first. A run that lost most of its items to the token "
        "cap has not been measured on reasoning, whatever its mean says.", "",
        coverage_table(runs), "",
        "## Mean score, by task", "",
        "Each cell is the mean and, in brackets, how many items it is a mean "
        "of.", "",
        task_table(runs, "mean"), "",
        "## Chance-corrected, by task", "",
        "`(score - floor) / (1 - floor)`. Negative means worse than a constant "
        "answer.", "",
        task_table(runs, "corrected"), "",
        "## Success rate, by task", "",
        "The task's own definition of a fully correct answer, which is not "
        "`score == 1.0` on every task. No floor column: a floor is a mean "
        "score, not a success rate.", "",
        task_table(runs, "success_rate", show_floor=False), "",
        "## Mean over untruncated generations only", "",
        task_table(runs, "mean_untruncated"), "",
        "---", "",
        "No overall mean is given. The twelve metrics are of four kinds and "
        "their chance floors span half the range, so an average over them "
        "measures which tasks are in the basket more than it measures a model.",
        "",
    ]
    return "\n".join(parts)


def write_csv(runs: List[Dict[str, Any]], path: str) -> None:
    # Every stat `_stats` produces. `extrasaction` is left at its default, so
    # adding a stat without adding it here fails loudly rather than dropping it
    # from the artifact people load into a dataframe.
    fields = ["run", "task", "level", "ordering", "n", "n_scored", "n_api_error",
              "n_scorer_refused", "mean", "mean_untruncated", "success_rate",
              "floor", "corrected", "truncated_rate", "no_answer_region_rate",
              "answer_in_cot_rate", "zero_with_region"]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        for m in runs:
            for group, level, ordering in (("by_task", "", ""),
                                           ("by_task_level", None, ""),
                                           ("by_task_ordering", "", None)):
                for key, v in m.get(group, {}).items():
                    task, _, rest = key.partition("|")
                    row = {k: v[k] for k in fields if k in v}
                    row.update(run=m["_label"], task=task,
                               level=rest if level is None else "",
                               ordering=rest if ordering is None else "")
                    w.writerow(row)


def _num(x: Optional[float]) -> str:
    return "-" if x is None else f"{x:.3f}"


def _pct(x: Optional[float]) -> str:
    return "-" if x is None else f"{x:.1%}"


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", nargs="+", help="Run directories, or metrics.json paths.")
    p.add_argument("-o", "--out", default="outputs/reports/latest")
    a = p.parse_args(argv)

    runs = load_metrics(a.runs)
    if not runs:
        raise SystemExit(
            "no metrics.json among those paths. Run `python -m evals.score "
            "<run_dir>` first -- generating and scoring are separate steps on "
            "purpose.")
    os.makedirs(a.out, exist_ok=True)
    md = os.path.join(a.out, "report.md")
    with open(md, "w") as f:
        f.write(render(runs))
    write_csv(runs, os.path.join(a.out, "results.csv"))
    print(render(runs))
    print(f"\nwrote {md} and results.csv", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
