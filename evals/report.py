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


def load_metrics(paths: Sequence[str]) -> List[Dict[str, Any]]:
    out = []
    for p in paths:
        f = p if p.endswith(".json") else os.path.join(p, "metrics.json")
        if not os.path.exists(f):
            continue
        with open(f) as fh:
            m = json.load(fh)
        m["_label"] = label(m)
        out.append(m)
    return sorted(out, key=lambda m: m["_label"])


def label(m: Dict[str, Any]) -> str:
    e = m["_meta"].get("endpoint") or {}
    return str(e.get("model") or os.path.basename(m["_meta"].get("run_dir", "run")))


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


def task_table(runs: List[Dict[str, Any]], field: str) -> str:
    tasks = sorted({t for m in runs for t in m["by_task"]})
    head = "| task | floor | " + " | ".join(m["_label"] for m in runs) + " |"
    rule = "|---|---:|" + "---:|" * len(runs)
    lines = [head, rule]
    for t in tasks:
        floor = next((m["by_task"][t].get("floor") for m in runs
                      if t in m["by_task"] and m["by_task"][t].get("floor") is not None),
                     None)
        cells = [_num(m["by_task"].get(t, {}).get(field)) for m in runs]
        lines.append(f"| {t} | {_num(floor)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


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
        task_table(runs, "mean"), "",
        "## Chance-corrected, by task", "",
        "`(score - floor) / (1 - floor)`. Negative means worse than a constant "
        "answer.", "",
        task_table(runs, "corrected"), "",
        "## Success rate, by task", "",
        "The task's own definition of a fully correct answer, which is not "
        "`score == 1.0` on every task.", "",
        task_table(runs, "success_rate"), "",
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
    fields = ["run", "task", "level", "ordering", "n", "n_scored", "n_api_error",
              "mean", "mean_untruncated", "success_rate", "floor", "corrected",
              "truncated_rate", "no_answer_region_rate", "zero_with_region"]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for m in runs:
            for group, level, ordering in (("by_task", "", ""),
                                           ("by_task_level", None, ""),
                                           ("by_task_ordering", "", None)):
                for key, v in m.get(group, {}).items():
                    task, _, rest = key.partition("|")
                    w.writerow({**v, "run": m["_label"], "task": task,
                                "level": rest if level is None else "",
                                "ordering": rest if ordering is None else ""})


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
