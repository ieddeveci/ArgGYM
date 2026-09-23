"""Compare finished runs. Reads `metrics.json`; runs nothing.

    uv run python -m evals.report outputs/runs/* -o outputs/reports/latest

Two rules and a definition, all from `docs/dataset-contract.md` section 10.

Chance floors are printed beside the scores, because a score means nothing
without the number an uninformed answer gets: at level 3 `semantics_query` sits
at 0.806 and `status_query` at 0.375, so a model scoring 0.45 on the first is
doing worse than answering the same thing every time.

There is no overall mean. Metrics of four kinds, one per task, with floors spanning half
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
from typing import Any, Dict, List, Optional, Sequence, Tuple

from evals.score import NEAR_TIMEOUT_FRACTION


class NotScored(SystemExit):
    """A run directory was named that has no metrics."""


def load_metrics(paths: Sequence[str]) -> List[Dict[str, Any]]:
    """Every named run's metrics, or a refusal naming the ones without any.

    Skipping an unscored run quietly turns "we never scored this model" into a
    table that simply lacks its column, and a reader cannot tell that from a
    model that was never run. `outputs/runs/*` is the documented way to call
    this, and it expands to every directory there -- including the ones a sweep
    generated and never got round to scoring.
    """
    out, missing = [], []
    for p in paths:
        f = p if p.endswith(".json") else os.path.join(p, "metrics.json")
        if not os.path.exists(f):
            missing.append(p)
            continue
        with open(f) as fh:
            m = json.load(fh)
        # Where this file was found, which `_meta.run_dir` does not say: that
        # field is an absolute path written at scoring time, so it names the
        # producing machine once a run directory is copied off it. A caller that
        # needs a sibling artifact -- `evals/figures.py` reads `samples.jsonl`
        # -- has to be told the path it passed in.
        m["_path"] = p
        out.append(m)
    if missing:
        raise NotScored(
            f"{len(missing)} of {len(paths)} run directories have no "
            f"metrics.json: {', '.join(missing[:4])}"
            + (" ..." if len(missing) > 4 else "")
            + ". Score them first, or leave them out deliberately -- a report "
              "that drops them silently shows 'not scored' as 'not run'.")
    for m, name in zip(out, labels(out)):
        m["_label"] = name
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


def labels(runs: Sequence[Dict[str, Any]]) -> List[str]:
    """`label`, made unique across the runs being compared.

    Model, template and elicitation do not identify a run. `run.py` treats a
    changed token cap or temperature as a different run, and rightly: the live
    validation on this branch compared one model against itself at two caps and
    the smaller one scored eight of twelve tasks at exactly 0.000. Under `label`
    alone both columns are headed `Qwen/Qwen3.6-27B`, and `results.csv` carries
    the same string in its `run` field for both.

    So a name is extended only where it collides, and by whatever actually
    differs: `Qwen/Qwen3.6-27B [max_tokens=24576]`. Adding the sampling to every
    header instead would put a line of noise on the common case, where one model
    was run once.
    """
    out = list(map(label, runs))
    for name in {n for n in out if out.count(n) > 1}:
        same = [i for i, n in enumerate(out) if n == name]
        settings = [_settings(runs[i]) for i in same]
        keys = sorted({k for s in settings for k in s
                       if any(t.get(k) != s[k] for t in settings)})
        for i, s in zip(same, settings):
            shown = ", ".join(f"{k}={s[k]}" for k in keys if k in s)
            # Nothing in the endpoint differs, so the directory is the only
            # thing left that does. Two identical configurations run twice is a
            # legitimate thing to compare, and it still needs two headers.
            #
            # The directory this file was *read* from, not the one recorded in
            # it. `_meta.run_dir` is written at scoring time and travels inside
            # the copy, so two copies of one run carried one header between them
            # -- the collision this branch exists to break.
            out[i] += f" [{shown}]" if shown else \
                f" [{os.path.basename(runs[i].get('_path') or str(i))}]"
    return out


def _settings(m: Dict[str, Any]) -> Dict[str, Any]:
    """The endpoint knobs that could tell two same-named runs apart."""
    e = m["_meta"].get("endpoint") or {}
    out = dict(e.get("sampling") or {})
    out.update({k: json.dumps(v, sort_keys=True)
                for k, v in (e.get("extra_body") or {}).items()})
    if e.get("base_url"):
        out["base_url"] = e["base_url"]
    return out


def coverage_table(runs: List[Dict[str, Any]]) -> str:
    """What was not measured, printed before anything that was.

    A model that lost most of its items to the token cap has not been measured
    on reasoning, and reading its mean as if it had is how the previous sweep's
    headline numbers went wrong.
    """
    lines = ["| run | status | scored | API errors | rows lost to timeout "
             "| requests that hit the timeout | truncated | no answer region "
             "| answered only in reasoning |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for m in runs:
        c, meta = m["coverage"], m["_meta"]
        lines.append(
            f"| {m['_label']} | {meta.get('run_status')} | {c['n_scored']} | "
            # Rows lost is a subset of the errors before it. Requests that hit
            # the wall is not a subset of anything: with `retries: 2` a row is
            # lost only when all three of its requests expire, so this column is
            # the one that moves first and the one that is non-zero while every
            # other number on this line still looks healthy.
            f"{c['n_api_error']} | {_int(c.get('n_api_timeout'))} | "
            f"{_int(c.get('n_requests_timed_out'))} | "
            f"{_pct(c['truncated_rate'])} | "
            f"{_pct(c['no_answer_region_rate'])} | "
            f"{_pct(c.get('answer_in_cot_rate'))} |")
    return "\n".join(lines)


def latency_table(runs: List[Dict[str, Any]]) -> str:
    """How much of each run's timeout budget it used.

    The percentile rather than the mean, because a timeout costs the slowest few
    percent of items and the mean says nothing about them; the maximum beside it,
    because a percentile that moved on one outlier should be readable as one. The
    count at the wall is the number that decides anything: a run whose p95 sits
    at four fifths of its timeout is one harder level away from losing items, and
    a lost item is a whole generation regenerated on retry.

    Over the generations that produced an answer. An errored one has a latency
    too and it measures the failure rather than the model, so it is counted in
    the coverage table instead (`evals/score.py:_latency`).

    Which means these columns see a row that hit the wall and recovered -- it
    answered, and its slowest request sits at the timeout -- and never see a row
    that hit the wall and stayed there. The count in the coverage table sees
    both, so it is the one to read when any of these approaches the timeout.
    """
    lines = [f"| run | median | p95 | slowest | timeout "
             f"| at {NEAR_TIMEOUT_FRACTION:.0%} of timeout | answered |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for m in runs:
        c = m["coverage"]
        lines.append(
            f"| {m['_label']} | {_secs(c.get('latency_s_p50'))} | "
            f"{_secs(c.get('latency_s_p95'))} | {_secs(c.get('latency_s_max'))} | "
            f"{_secs(c.get('timeout_s'))} | {_int(c.get('n_near_timeout'))} | "
            f"{_int(c.get('n_latency_measured'))} |")
    return "\n".join(lines)


def error_table(runs: List[Dict[str, Any]]) -> str:
    """One row per cause of API error, or nothing at all when there were none.

    `unclassified` is a generation made before the cause was recorded, not a
    cause. The alternative -- reading the kind back out of the provider's
    message -- would make the taxonomy depend on how a provider words things.
    """
    kinds = sorted({k for m in runs
                    for k in (m["coverage"].get("api_errors_by_kind") or {})})
    if not kinds:
        return ""
    head = "| cause | " + " | ".join(m["_label"] for m in runs) + " |"
    lines = [head, "|---|" + "---:|" * len(runs)]
    for k in kinds:
        cells = [str((m["coverage"].get("api_errors_by_kind") or {}).get(k, 0))
                 for m in runs]
        lines.append(f"| {k} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def task_table(runs: List[Dict[str, Any]], field: str, show_floor: bool = True) -> str:
    """One row per task, one column per run, with the count beside the value.

    The count is there because a mean over three surviving items and a mean over
    forty are formatted identically otherwise, and the coverage table above is
    run-wide rather than per task.

    `show_floor` is off for `success_rate`: a floor is the mean *score* of the
    best uninformed strategy (`arggym/core/floors.py`), so printing it beside a
    success rate invites exactly the comparison it exists to prevent.
    """
    tasks = sorted({t for m in runs for t in m["by_task"]})
    head = "| task |" + (" floor |" if show_floor else "") + \
        " " + " | ".join(m["_label"] for m in runs) + " |"
    rule = "|---|" + ("---:|" if show_floor else "") + "---:|" * len(runs)
    lines = [head, rule]
    marked = False
    for t in tasks:
        floor, disputed = _floor(runs, t)
        floor_cell = _num(floor) + ("*" if disputed else "")
        marked = marked or disputed
        cells = [_cell(m["by_task"].get(t, {}), field) for m in runs]
        lines.append(f"| {t} |" + (f" {floor_cell} |" if show_floor else "")
                     + " " + " | ".join(cells) + " |")
    if marked and show_floor:
        lines.append("")
        lines.append("`*` one column's `corrected` cannot be reproduced from this "
                     "floor: either the runs measured different floors for that "
                     "task, or one of them could not measure a floor at all "
                     "(`_meta.floors_unmeasured` says which). A floor is measured "
                     "over the rows a run scored.")
    return "\n".join(lines)


def _floor(runs: Sequence[Dict[str, Any]],
           task: str) -> Tuple[Optional[float], bool]:
    """The floor to print for a task, and whether printing one is honest.

    A floor is measured over the rows a run actually scored, so a filtered run
    and a full one produce different ones -- and one ungradeable row leaves a
    run with no floor for that task at all. Either way a single number in the
    column is a number some column's `corrected` was not computed from, so the
    disagreement is marked rather than hidden. It was marked only for the first
    case, which is the quieter of the two.
    """
    present = [m["by_task"][task] for m in runs if task in m["by_task"]]
    seen = [s["floor"] for s in present if s.get("floor") is not None]
    if not seen:
        return None, False
    return seen[0], len(set(seen)) > 1 or len(seen) != len(present)


#: Which count belongs beside which value. `mean_untruncated` is a mean over the
#: generations that were not cut off, so `n_scored` beside it prints a mean of
#: four items as `0.850 (40)` -- the exact confusion the bracket exists to
#: prevent. Everything else here is over the scored records.
COUNT_OF = {"mean_untruncated": "n_untruncated"}


def _cell(stats: Dict[str, Any], field: str) -> str:
    if not stats:
        return "-"
    n = stats.get(COUNT_OF.get(field, "n_scored"))
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
    ]
    errors = error_table(runs)
    if errors:
        # Only when something failed. On a clean sweep this section would be a
        # table of zeros in a report whose first instruction is to read the
        # coverage table, and `API errors 0` above has already said it.
        parts += ["### API errors, by cause", "",
                  "A timeout moves with the token cap and the other causes do "
                  "not, so they are counted apart.", "", errors, ""]
    parts += [
        "## Latency headroom", "",
        "The slowest request per row, over the generations that answered. A run "
        "whose p95 is close to its timeout is one harder level away from losing "
        "items, and a request that hits the timeout is retried -- the server "
        "generates the whole completion again.", "",
        "A row that hit the wall and recovered appears here at the wall. A row "
        "that never recovered does not appear here at all -- it is an error, "
        "and the coverage table's `requests that hit the timeout` is the only "
        "number that counts both.", "",
        latency_table(runs), "",
        "## Mean score, by task", "",
        "Each cell is the mean and, in brackets, how many items it is a mean "
        "of.", "",
        task_table(runs, "mean"), "",
        "## Chance-corrected, by task", "",
        "`(score - floor) / (1 - floor)`. Negative means worse than the best "
        "uninformed answer.", "",
        task_table(runs, "corrected"), "",
        "## Success rate, by task", "",
        "The task's own definition of a fully correct answer, which is not "
        "`score == 1.0` on every task. No floor column: a floor is a mean "
        "score, not a success rate.", "",
        task_table(runs, "success_rate", show_floor=False), "",
        "## Bloat rate, by task", "",
        "The share of scored answers zeroed for using more than twice the "
        "minimum number of directives. Nonzero only on the construction tasks; "
        "read it beside their mean, since a bloat zero and a wrong answer score "
        "the same. No floor column: this is a rate, not a score.", "",
        task_table(runs, "bloat_rate", show_floor=False), "",
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
    # Every stat `_stats` produces. The row is handed to `DictWriter` whole and
    # `extrasaction` is left at its default, so adding a stat without adding it
    # here raises rather than dropping it from the artifact people load into a
    # dataframe. Filtering the row to `fields` first, as this did, made that
    # comment describe a check that could not fire.
    fields = ["run", "task", "level", "ordering", "n", "n_scored", "n_untruncated",
              "n_api_error", "n_api_timeout", "n_requests_timed_out",
              "n_scorer_refused", "mean", "mean_untruncated",
              "success_rate", "floor", "floor_strategy", "corrected", "floor_error",
              "truncated_rate", "no_answer_region_rate",
              "answer_in_cot_rate", "zero_with_region", "bloat_rate"]
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
                    row = dict(v)
                    row.update(run=m["_label"], task=task,
                               level=rest if level is None else "",
                               ordering=rest if ordering is None else "")
                    w.writerow(row)


def _num(x: Optional[float]) -> str:
    return "-" if x is None else f"{x:.3f}"


def _pct(x: Optional[float]) -> str:
    return "-" if x is None else f"{x:.1%}"


def _int(x: Optional[int]) -> str:
    """`-` and not `0`, here and in `_secs`. Every `metrics.json` already on disk
    was written before latency was measured and has none of these keys, and an
    absent number rendered as zero prints a run with no timeouts recorded and a
    run with no timeouts the same way."""
    return "-" if x is None else str(x)


def _secs(x: Optional[float]) -> str:
    return "-" if x is None else f"{x:,.0f}s"


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", nargs="+", help="Run directories, or metrics.json paths.")
    p.add_argument("-o", "--out", default="outputs/reports/latest")
    a = p.parse_args(argv)

    # `load_metrics` refuses a path with no metrics.json and argparse guarantees
    # at least one path, so there is no empty case to handle here.
    runs = load_metrics(a.runs)
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
